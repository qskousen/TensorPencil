//! Gemma 3 vision tower (mmproj GGUF, arch "clip", projector "gemma3"):
//! a SigLIP-So400m encoder + the Gemma multimodal projector. CPU forward,
//! f32 compute over the f16/f32 mmproj weights. Ported from llama.cpp
//! tools/mtmd (clip.cpp build_gemma3 + mtmd preprocessing).
//!
//! Pipeline: resize to a fixed 896x896 + normalize ((p/255 - 0.5)/0.5) ->
//! patch conv (14x14 stride 14 -> 64x64 = 4096 patches of 1152) as an
//! im2col GEMM -> add the learned [4096][1152] position embedding -> 27
//! pre-LN SigLIP blocks (LayerNorm w/ bias, separate q/k/v with biases, 16
//! heads x 72, full bidirectional attention scaled 1/sqrt(72), GELU-tanh
//! FFN) -> post_ln -> projector: 4x4 average pool over the 64x64 grid ->
//! 256 tokens, soft_emb_norm (RMSNorm, +1 folded into the GGUF weight),
//! mm.input_projection (1152 -> 3840) -> [256][3840] embeddings for the LLM.
//!
//! Each image is exactly 256 soft tokens; the LLM injects them unscaled
//! (embeddings are NOT multiplied by sqrt(hidden)) at sequential positions.

const std = @import("std");
const gguf_mod = @import("../gguf.zig");
const weights_mod = @import("../weights.zig");
const ops = @import("../ops.zig");
const loader = @import("loader.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    dim: usize,
    n_heads: usize,
    ffn: usize,
    n_blocks: usize,
    patch: usize,
    image_size: usize,
    proj_dim: usize,
    /// Average-pool factor in the projector (gemma3: 4 -> 64x64 patches
    /// become 16x16 = 256 tokens).
    pool: usize,
    image_mean: [3]f32,
    image_std: [3]f32,
    eps: f32,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }
    /// Patches per side (image_size / patch).
    pub fn side(self: Config) usize {
        return self.image_size / self.patch;
    }
    /// Merged tokens per side (side / pool).
    pub fn tokSide(self: Config) usize {
        return self.side() / self.pool;
    }
    pub fn nTokens(self: Config) usize {
        return self.tokSide() * self.tokSide();
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "clip")) return error.UnknownModelConfig;
        const proj = g.getStr("clip.projector_type") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, proj, "gemma3")) return error.UnknownModelConfig;
        const key = struct {
            fn f(gg: *const Gguf, comptime name: []const u8) !usize {
                return @intCast(gg.getUint("clip.vision." ++ name) orelse return error.UnknownModelConfig);
            }
        }.f;
        var mean: [3]f32 = undefined;
        var stdv: [3]f32 = undefined;
        inline for (.{ "image_mean", "image_std" }, .{ &mean, &stdv }) |name, dst| {
            var it = (g.getArr("clip.vision." ++ name) orelse return error.UnknownModelConfig).iterate();
            for (dst) |*v| v.* = @floatCast((it.next() orelse return error.UnknownModelConfig).float);
        }
        const image_size = try key(g, "image_size");
        const patch = try key(g, "patch_size");
        const n_side = image_size / patch;
        // Gemma3 pools 4x4 -> 256 tokens (16x16); the position table is
        // n_side*n_side learned positions.
        const pos = g.get("v.position_embd.weight") orelse return error.UnknownModelConfig;
        const pshape = pos.info.shape.slice();
        if (pshape.len != 2 or pshape[0] != n_side * n_side) return error.UnknownModelConfig;
        return .{
            .dim = try key(g, "embedding_length"),
            .n_heads = try key(g, "attention.head_count"),
            .ffn = try key(g, "feed_forward_length"),
            .n_blocks = try key(g, "block_count"),
            .patch = patch,
            .image_size = image_size,
            .proj_dim = @intCast(g.getUint("clip.vision.projection_dim") orelse return error.UnknownModelConfig),
            .pool = 4,
            .image_mean = mean,
            .image_std = stdv,
            .eps = @floatCast(g.getFloat("clip.vision.attention.layer_norm_epsilon") orelse 1e-6),
        };
    }
};

const Block = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    q: Weight,
    q_b: []const f32,
    k: Weight,
    k_b: []const f32,
    v: Weight,
    v_b: []const f32,
    out: Weight,
    out_b: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    up: Weight,
    up_b: []const f32,
    down: Weight,
    down_b: []const f32,
};

pub const Vit = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// Patch conv kernel as [dim][3*patch*patch] (im2col GEMM weight).
    patch_w: []const f32,
    patch_b: []const f32,
    /// Learned positions [side*side][dim].
    pos_embd: []const f32,
    blocks: []Block,
    post_ln_w: []const f32,
    post_ln_b: []const f32,
    soft_emb_norm: []const f32,
    /// mm.input_projection [proj_dim][dim], no bias.
    mm_proj: Weight,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Vit {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const kdim = 3 * cfg.patch * cfg.patch;
        const patch_w = try loader.vector(alloc, store, "v.patch_embd.weight", cfg.dim * kdim);
        const patch_b = try loader.vector(alloc, store, "v.patch_embd.bias", cfg.dim);
        const pos_embd = try loader.vector(alloc, store, "v.position_embd.weight", cfg.side() * cfg.side() * cfg.dim);

        const blocks = try alloc.alloc(Block, cfg.n_blocks);
        for (blocks, 0..) |*blk, i| {
            blk.* = .{
                .ln1_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ln1.weight", cfg.dim),
                .ln1_b = try loader.indexedVector(alloc, store, "v.blk.", i, "ln1.bias", cfg.dim),
                .q = try loader.indexedMatrix(store, "v.blk.", i, "attn_q.weight", cfg.dim, cfg.dim),
                .q_b = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_q.bias", cfg.dim),
                .k = try loader.indexedMatrix(store, "v.blk.", i, "attn_k.weight", cfg.dim, cfg.dim),
                .k_b = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_k.bias", cfg.dim),
                .v = try loader.indexedMatrix(store, "v.blk.", i, "attn_v.weight", cfg.dim, cfg.dim),
                .v_b = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_v.bias", cfg.dim),
                .out = try loader.indexedMatrix(store, "v.blk.", i, "attn_out.weight", cfg.dim, cfg.dim),
                .out_b = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_out.bias", cfg.dim),
                .ln2_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ln2.weight", cfg.dim),
                .ln2_b = try loader.indexedVector(alloc, store, "v.blk.", i, "ln2.bias", cfg.dim),
                .up = try loader.indexedMatrix(store, "v.blk.", i, "ffn_up.weight", cfg.ffn, cfg.dim),
                .up_b = try loader.indexedVector(alloc, store, "v.blk.", i, "ffn_up.bias", cfg.ffn),
                .down = try loader.indexedMatrix(store, "v.blk.", i, "ffn_down.weight", cfg.dim, cfg.ffn),
                .down_b = try loader.indexedVector(alloc, store, "v.blk.", i, "ffn_down.bias", cfg.dim),
            };
        }

        const post_ln_w = try loader.vector(alloc, store, "v.post_ln.weight", cfg.dim);
        const post_ln_b = try loader.vector(alloc, store, "v.post_ln.bias", cfg.dim);
        const soft_emb_norm = try loader.vector(alloc, store, "mm.soft_emb_norm.weight", cfg.dim);
        // mm.input_projection is stored [dim, proj_dim] (in, out); the engine's
        // matmul is out = x @ Wᵀ with W = [out, in], so dequant + transpose to
        // [proj_dim, dim] f32 (one-time, ~17 MB).
        const mm_view = store.get("mm.input_projection.weight") orelse return error.MissingTensor;
        const ms = mm_view.info.shape.slice();
        if (ms.len != 2 or ms[0] != cfg.dim or ms[1] != cfg.proj_dim) return error.ShapeMismatch;
        const mm_src = try mm_view.toF32Alloc(alloc);
        const mm_w = try alloc.alloc(f32, cfg.proj_dim * cfg.dim);
        for (0..cfg.proj_dim) |o| {
            for (0..cfg.dim) |i| mm_w[o * cfg.dim + i] = mm_src[i * cfg.proj_dim + o];
        }
        const mm_proj = Weight.fromF32(mm_w, cfg.proj_dim, cfg.dim);

        return .{
            .arena = arena,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embd = pos_embd,
            .blocks = blocks,
            .post_ln_w = post_ln_w,
            .post_ln_b = post_ln_b,
            .soft_emb_norm = soft_emb_norm,
            .mm_proj = mm_proj,
        };
    }

    pub fn deinit(self: *Vit) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub const Encoded = struct {
        /// [nTokens][proj_dim] projected image-token embeddings.
        embeds: []f32,
        /// Merged grid dims (tokSide x tokSide); n_tokens = grid_w*grid_h.
        grid_w: usize,
        grid_h: usize,

        pub fn deinit(self: *Encoded, gpa: std.mem.Allocator) void {
            gpa.free(self.embeds);
            self.* = undefined;
        }
    };

    /// Encode interleaved RGB pixels to LLM image-token embeddings.
    pub fn encode(self: *const Vit, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Encoded {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const hd = cfg.headDim();
        const side = cfg.side();
        const np = side * side;
        const kdim = 3 * cfg.patch * cfg.patch;

        // Preprocess + patch matrix (im2col) in row-major patch order.
        const patches = try self.patchMatrix(gpa, rgb, width, height);
        defer gpa.free(patches);

        var x = try gpa.alloc(f32, np * dim);
        defer gpa.free(x);
        try ops.matmul.matmul(io, gpa, x, patches, np, Weight.fromF32(self.patch_w, dim, kdim), self.patch_b);
        for (0..np) |t| {
            const src = self.pos_embd[t * dim ..][0..dim];
            const dst = x[t * dim ..][0..dim];
            for (dst, src) |*d, s| d.* += s;
        }

        var s = try Scratch.init(gpa, np, cfg);
        defer s.deinit(gpa);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        for (self.blocks) |*blk| {
            ops.norm.layerNorm(s.normed, x, blk.ln1_w, blk.ln1_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, s.q, s.normed, np, blk.q, blk.q_b);
            try ops.matmul.matmul(io, gpa, s.k, s.normed, np, blk.k, blk.k_b);
            try ops.matmul.matmul(io, gpa, s.v, s.normed, np, blk.v, blk.v_b);
            try ops.attention.attention(io, gpa, s.attn, s.q, s.k, s.v, .{
                .seq_q = np,
                .seq_kv = np,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = hd,
                .causal = false,
                .scale = scale,
            });
            try ops.matmul.matmul(io, gpa, s.tmp, s.attn, np, blk.out, blk.out_b);
            for (x, s.tmp) |*xi, ti| xi.* += ti;

            ops.norm.layerNorm(s.normed, x, blk.ln2_w, blk.ln2_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, s.ffn, s.normed, np, blk.up, blk.up_b);
            ops.act.geluTanh(s.ffn);
            try ops.matmul.matmul(io, gpa, s.tmp, s.ffn, np, blk.down, blk.down_b);
            for (x, s.tmp) |*xi, ti| xi.* += ti;
        }
        ops.norm.layerNorm(x, x, self.post_ln_w, self.post_ln_b, cfg.eps);
        return self.project(io, gpa, x);
    }

    /// Gemma multimodal projector over post-LN patch hidden states `x`
    /// ([side*side][dim]): 4x4 average pool over the patch grid -> 256 tokens,
    /// soft_emb_norm (RMSNorm; +1 folded into the weight), input projection.
    /// Shared by the CPU encode and the CUDA encode (which runs the encoder
    /// device-side, downloads the post-LN states, and projects here — the
    /// projector is cheap next to the 27 blocks).
    pub fn project(self: *const Vit, io: std.Io, gpa: std.mem.Allocator, x: []const f32) !Encoded {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const side = cfg.side();
        const tside = cfg.tokSide();
        const nt = tside * tside;
        const pooled = try gpa.alloc(f32, nt * dim);
        defer gpa.free(pooled);
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(cfg.pool * cfg.pool));
        for (0..tside) |py| {
            for (0..tside) |px| {
                const dst = pooled[(py * tside + px) * dim ..][0..dim];
                @memset(dst, 0);
                for (0..cfg.pool) |dy| {
                    for (0..cfg.pool) |dx| {
                        const src = x[((py * cfg.pool + dy) * side + (px * cfg.pool + dx)) * dim ..][0..dim];
                        for (dst, src) |*d, sv| d.* += sv;
                    }
                }
                for (dst) |*d| d.* *= inv;
            }
        }
        ops.norm.rmsNorm(pooled, pooled, self.soft_emb_norm, cfg.eps);

        const embeds = try gpa.alloc(f32, nt * cfg.proj_dim);
        errdefer gpa.free(embeds);
        try ops.matmul.matmul(io, gpa, embeds, pooled, nt, self.mm_proj, null);
        return .{ .embeds = embeds, .grid_w = tside, .grid_h = tside };
    }

    /// Build the im2col patch matrix ([side*side][3*patch*patch], row-major
    /// patch order) from preprocessed planar CHW pixels. Shared with the CUDA
    /// encoder (which uploads it for the patch-embed GEMM).
    pub fn patchMatrix(self: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) ![]f32 {
        const cfg = self.cfg;
        const side = cfg.side();
        const kdim = 3 * cfg.patch * cfg.patch;
        const chw = try preprocess(gpa, rgb, width, height, cfg.image_size, cfg.image_mean, cfg.image_std);
        defer gpa.free(chw);
        const patches = try gpa.alloc(f32, side * side * kdim);
        errdefer gpa.free(patches);
        for (0..side) |gy| {
            for (0..side) |gx| {
                const row = patches[(gy * side + gx) * kdim ..][0..kdim];
                for (0..3) |c| {
                    for (0..cfg.patch) |ky| {
                        const src = chw[c * cfg.image_size * cfg.image_size + (gy * cfg.patch + ky) * cfg.image_size + gx * cfg.patch ..][0..cfg.patch];
                        @memcpy(row[(c * cfg.patch + ky) * cfg.patch ..][0..cfg.patch], src);
                    }
                }
            }
        }
        return patches;
    }
};

/// Per-forward activation buffers for `np` patch tokens.
const Scratch = struct {
    normed: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    tmp: []f32,
    ffn: []f32,

    fn init(gpa: std.mem.Allocator, np: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            if (i < done) gpa.free(@field(s, f.name));
        };
        const sizes = [_]usize{
            np * cfg.dim, // normed
            np * cfg.dim, // q
            np * cfg.dim, // k
            np * cfg.dim, // v
            np * cfg.dim, // attn
            np * cfg.dim, // tmp
            np * cfg.ffn, // ffn
        };
        inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try gpa.alloc(f32, sizes[i]);
            done = i + 1;
        }
        return s;
    }

    fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        inline for (@typeInfo(Scratch).@"struct".fields) |f| gpa.free(@field(self, f.name));
        self.* = undefined;
    }
};

/// Resize `rgb` (interleaved, sw x sh) into a `size` x `size` square and
/// normalize to planar CHW f32. Matches llama.cpp mtmd gemma3 (PAD_CEIL):
/// aspect-preserving align-corners bilinear (truncating u8) into a fitted
/// rectangle, centered on a black canvas, then (p/255 - mean)/std — so the
/// black pad normalizes to (0 - mean)/std.
fn preprocess(gpa: std.mem.Allocator, rgb: []const u8, sw: usize, sh: usize, size: usize, mean: [3]f32, stdv: [3]f32) ![]f32 {
    const scale = @min(
        @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(sw)),
        @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(sh)),
    );
    const nw = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sw)) * scale))), size);
    const nh = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sh)) * scale))), size);

    const resized = try gpa.alloc(u8, nw * nh * 3);
    defer gpa.free(resized);
    const x_ratio: f64 = if (nw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(nw - 1)) else 0;
    const y_ratio: f64 = if (nh > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(nh - 1)) else 0;
    for (0..nh) |y| {
        const pyf = @as(f64, @floatFromInt(y)) * y_ratio;
        const y0 = @min(@as(usize, @intFromFloat(pyf)), sh - 1);
        const y1 = @min(y0 + 1, sh - 1);
        const yf = pyf - @as(f64, @floatFromInt(y0));
        for (0..nw) |xx| {
            const pxf = @as(f64, @floatFromInt(xx)) * x_ratio;
            const x0 = @min(@as(usize, @intFromFloat(pxf)), sw - 1);
            const x1 = @min(x0 + 1, sw - 1);
            const xf = pxf - @as(f64, @floatFromInt(x0));
            for (0..3) |c| {
                const p00: f64 = @floatFromInt(rgb[(y0 * sw + x0) * 3 + c]);
                const p01: f64 = @floatFromInt(rgb[(y0 * sw + x1) * 3 + c]);
                const p10: f64 = @floatFromInt(rgb[(y1 * sw + x0) * 3 + c]);
                const p11: f64 = @floatFromInt(rgb[(y1 * sw + x1) * 3 + c]);
                const top = p00 + (p01 - p00) * xf;
                const bottom = p10 + (p11 - p10) * xf;
                resized[(y * nw + xx) * 3 + c] = @intFromFloat(top + (bottom - top) * yf); // truncation, as llama.cpp
            }
        }
    }

    const off_x = (size - nw) / 2;
    const off_y = (size - nh) / 2;
    const out = try gpa.alloc(f32, 3 * size * size);
    errdefer gpa.free(out);
    for (0..3) |c| {
        const plane = out[c * size * size ..][0 .. size * size];
        @memset(plane, (0.0 - mean[c]) / stdv[c]); // black pad
        for (0..nh) |y| {
            for (0..nw) |xx| {
                const p: f32 = @floatFromInt(resized[(y * nw + xx) * 3 + c]);
                plane[(y + off_y) * size + (xx + off_x)] = (p / 255.0 - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

// --- tests -----------------------------------------------------------------

test "gemma vit loads from real mmproj" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/mmproj-F16.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    const cfg = vit.cfg;
    try std.testing.expectEqual(@as(usize, 1152), cfg.dim);
    try std.testing.expectEqual(@as(usize, 16), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 72), cfg.headDim());
    try std.testing.expectEqual(@as(usize, 4304), cfg.ffn);
    try std.testing.expectEqual(@as(usize, 27), cfg.n_blocks);
    try std.testing.expectEqual(@as(usize, 896), cfg.image_size);
    try std.testing.expectEqual(@as(usize, 64), cfg.side());
    try std.testing.expectEqual(@as(usize, 256), cfg.nTokens());
    try std.testing.expectEqual(@as(usize, 3840), cfg.proj_dim);
    try std.testing.expectEqual(@as(usize, 27), vit.blocks.len);
}
