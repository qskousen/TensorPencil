//! Gemma 4 vision tower (mmproj GGUF, arch "clip", projector "gemma4v"): a
//! full SigLIP-style encoder + the Gemma-4 multimodal projector. CPU forward,
//! f32 compute over the Q8_0/F16/F32 mmproj weights. Ported from llama.cpp
//! tools/mtmd (models/gemma4v.cpp + clip.cpp build_vit + mtmd-image smart_resize).
//!
//! This is distinct from BOTH the shallow "gemma4uv" embedder (`gemma4_vit.zig`,
//! no transformer, used by the 12B) AND the gemma3 SigLIP tower (`gemma_vit.zig`,
//! LayerNorm / GELU-tanh / no QK-norm / learned-pos-only). gemma4v adds, per
//! block: RMSNorm sandwich norms (ln1 + attn_post_norm + ln2 + ffn_post_norm),
//! per-head QK-RMSNorm, a 2-D neox RoPE (theta 100, head split x/y), a weightless
//! V-RMSNorm, attention with scale = 1.0 (NOT 1/sqrt(hd)), and a GeGLU-quick FFN.
//! The projector is: 3x3 average-pool "merge" -> x*sqrt(dim) -> (x - std_bias) *
//! std_scale -> weightless RMSNorm -> mm.input_projection (dim -> proj_dim).
//!
//! Preprocessing follows Google's `gemma4_vision_token_budget` reference: an
//! aspect-preserving resize (NO crop, NO letterbox/pad) to a 48-aligned grid
//! sized so the post-merge token count targets a settable budget `nMax`
//! (`Budget` presets 70/140/280/560/1120; default 280). Scale factor
//! f = sqrt(nMax*48^2 / (w*h)); each dim floored to a multiple of 48. Pixels are
//! normalized to 2*(p/255) - 1. (This differs from llama.cpp's dyn_size, which
//! adds min-pixel clamping and a black-bar letterbox we deliberately drop.)
//!
//! Image tokens are injected UNSCALED at their placeholder positions (like the
//! other Gemma towers); the number of tokens per image is variable.

const std = @import("std");
const gguf_mod = @import("tp_core").gguf;
const weights_mod = @import("tp_core").weights;
const ops = @import("tp_ops");
const loader = @import("loader.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// Vision token-budget preset (nMax). Values are Google's own budget ladder from
/// the `gemma4_vision_token_budget` space: the image is resized so the post-merge
/// token count targets nMax (aspect-preserving, no crop/pad). More tokens = more
/// spatial detail, more compute. `high` (280) is the default and matches the
/// llama.cpp mmproj default max.
pub const Budget = enum {
    low,
    medium,
    high,
    ultra,
    max,

    pub fn tokens(self: Budget) usize {
        return switch (self) {
            .low => 70,
            .medium => 140,
            .high => 280,
            .ultra => 560,
            .max => 1120,
        };
    }

    pub fn parse(s: []const u8) ?Budget {
        return std.meta.stringToEnum(Budget, s);
    }
};

pub const Config = struct {
    dim: usize,
    n_heads: usize,
    ffn: usize,
    n_blocks: usize,
    patch: usize,
    proj_dim: usize,
    /// Average-pool factor in the projector (gemma4v: 3 -> a 3x3 merge).
    merge: usize,
    /// 2-D RoPE base (gemma4v vision: 100.0, not the LLM's 1e6/1e4).
    rope_theta: f64,
    /// Vision token budget (nMax): the image is resized so the post-merge token
    /// count targets ~this many (aspect-preserving, no crop/pad — Google's
    /// gemma4_vision_token_budget algorithm). Runtime-settable (CLI
    /// `--vision-budget`, GUI); `detect` seeds the `high` default.
    max_tokens: usize,
    eps: f32,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }
    /// Pixel alignment for smart_resize = patch * merge (48 for gemma4v).
    pub fn alignPx(self: Config) usize {
        return self.patch * self.merge;
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "clip")) return error.UnknownModelConfig;
        // gemma4v stores the projector under clip.vision.projector_type (like
        // the shallow gemma4uv variant), NOT the top-level clip.projector_type
        // that gemma3/qwen3vl use.
        const proj = g.getStr("clip.vision.projector_type") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, proj, "gemma4v")) return error.UnknownModelConfig;
        const key = struct {
            fn f(gg: *const Gguf, comptime name: []const u8) !usize {
                return @intCast(gg.getUint("clip.vision." ++ name) orelse return error.UnknownModelConfig);
            }
        }.f;
        const merge: usize = 3; // gemma4v pooling_kernel_size (clip.cpp)
        const patch = try key(g, "patch_size");
        return .{
            .dim = try key(g, "embedding_length"),
            .n_heads = try key(g, "attention.head_count"),
            .ffn = try key(g, "feed_forward_length"),
            .n_blocks = try key(g, "block_count"),
            .patch = patch,
            .proj_dim = @intCast(g.getUint("clip.vision.projection_dim") orelse return error.UnknownModelConfig),
            .merge = merge,
            .rope_theta = 100.0,
            .max_tokens = Budget.high.tokens(),
            .eps = @floatCast(g.getFloat("clip.vision.attention.layer_norm_epsilon") orelse 1e-6),
        };
    }
};

const Block = struct {
    ln1_w: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    q_norm: []const f32, // per-head QK-norm, head_dim entries
    k_norm: []const f32,
    attn_post_norm_w: []const f32,
    ln2_w: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
    ffn_post_norm_w: []const f32,
};

pub const Vit = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// Patch conv kernel as [dim][3*patch*patch] (im2col GEMM weight, f32, no bias).
    patch_w: []const f32,
    /// 2-D learned position tables, concatenated: [2 * pos_size * dim] with
    /// table x at rows [0, pos_size) and table y at [pos_size, 2*pos_size).
    pos_embd: []const f32,
    pos_size: usize,
    blocks: []Block,
    /// Projector post-pool affine (per channel, dim entries): (x - bias) * scale.
    std_bias: []const f32,
    std_scale: []const f32,
    /// mm.input_projection [proj_dim][dim] (standard [out,in] orientation).
    mm_proj: Weight,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Vit {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const kdim = 3 * cfg.patch * cfg.patch;
        const patch_w = try loader.vector(alloc, store, "v.patch_embd.weight", cfg.dim * kdim);

        // Position table: GGUF ne = [dim, pos_size, 2] -> flat f32 in ne0-fastest
        // order, i.e. flat[((t*pos_size + p)*dim + e)]. Validate the shape.
        const pos_view = store.get("v.position_embd.weight") orelse return error.MissingTensor;
        const ps = pos_view.info.shape.slice(); // reversed: [2, pos_size, dim]
        if (ps.len != 3 or ps[0] != 2 or ps[2] != cfg.dim) return error.ShapeMismatch;
        const pos_size = ps[1];
        const pos_embd = try loader.vector(alloc, store, "v.position_embd.weight", 2 * pos_size * cfg.dim);

        const hd = cfg.headDim();
        const blocks = try alloc.alloc(Block, cfg.n_blocks);
        for (blocks, 0..) |*blk, i| {
            blk.* = .{
                .ln1_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ln1.weight", cfg.dim),
                .q = try loader.indexedMatrix(store, "v.blk.", i, "attn_q.weight", cfg.dim, cfg.dim),
                .k = try loader.indexedMatrix(store, "v.blk.", i, "attn_k.weight", cfg.dim, cfg.dim),
                .v = try loader.indexedMatrix(store, "v.blk.", i, "attn_v.weight", cfg.dim, cfg.dim),
                .out = try loader.indexedMatrix(store, "v.blk.", i, "attn_out.weight", cfg.dim, cfg.dim),
                .q_norm = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_q_norm.weight", hd),
                .k_norm = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_k_norm.weight", hd),
                .attn_post_norm_w = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_post_norm.weight", cfg.dim),
                .ln2_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ln2.weight", cfg.dim),
                .gate = try loader.indexedMatrix(store, "v.blk.", i, "ffn_gate.weight", cfg.ffn, cfg.dim),
                .up = try loader.indexedMatrix(store, "v.blk.", i, "ffn_up.weight", cfg.ffn, cfg.dim),
                .down = try loader.indexedMatrix(store, "v.blk.", i, "ffn_down.weight", cfg.dim, cfg.ffn),
                .ffn_post_norm_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ffn_post_norm.weight", cfg.dim),
            };
        }

        const std_bias = try loader.vector(alloc, store, "v.std_bias", cfg.dim);
        const std_scale = try loader.vector(alloc, store, "v.std_scale", cfg.dim);
        // mm.input_projection is stored in standard [out, in] = [proj_dim, dim]
        // orientation (unlike gemma3, which needs a transpose), so it loads
        // directly as a dtype-preserving Weight (Q8_0, dequant in the GEMM).
        const mm_proj = try loader.matrix(store, "mm.input_projection.weight", cfg.proj_dim, cfg.dim);

        return .{
            .arena = arena,
            .cfg = cfg,
            .patch_w = patch_w,
            .pos_embd = pos_embd,
            .pos_size = pos_size,
            .blocks = blocks,
            .std_bias = std_bias,
            .std_scale = std_scale,
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
        /// Merged grid dims; n_tokens = grid_w * grid_h.
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
        const kdim = 3 * cfg.patch * cfg.patch;

        // Preprocess + smart-resize + im2col (shared with the CUDA encoder).
        var prep = try self.prepare(gpa, rgb, width, height);
        defer prep.deinit(gpa);
        const gx = prep.gx;
        const gy = prep.gy;
        const np = gx * gy;

        const x = try gpa.alloc(f32, np * dim);
        defer gpa.free(x);
        try ops.matmul.matmul(io, gpa, x, prep.patches, np, Weight.fromF32(self.patch_w, dim, kdim), null);

        // 2-D learned position embeddings (table_x[col] + table_y[row] per patch).
        const pos = try self.posEmbedRows(gpa, gx, gy);
        defer gpa.free(pos);
        for (x, pos) |*d, p| d.* += p;

        // Per-patch grid coordinates for the 2-D RoPE.
        const pos_x = try gpa.alloc(usize, np);
        defer gpa.free(pos_x);
        const pos_y = try gpa.alloc(usize, np);
        defer gpa.free(pos_y);
        for (0..np) |t| {
            pos_x[t] = t % gx;
            pos_y[t] = t / gx;
        }
        // neox RoPE over each head's two halves (span hd/2 each), theta 100.
        var freqs = try ops.rope.rotateHalfFreqs(gpa, @max(gx, gy), hd / 2, cfg.rope_theta);
        defer freqs.deinit(gpa);

        var s = try Scratch.init(gpa, np, cfg);
        defer s.deinit(gpa);

        for (self.blocks) |*blk| {
            // --- attention ---
            ops.norm.rmsNorm(s.normed, x, blk.ln1_w, cfg.eps);
            try ops.matmul.matmul(io, gpa, s.q, s.normed, np, blk.q, null);
            try ops.matmul.matmul(io, gpa, s.k, s.normed, np, blk.k, null);
            try ops.matmul.matmul(io, gpa, s.v, s.normed, np, blk.v, null);
            // per-head QK-RMSNorm (rows of head_dim), then 2-D RoPE.
            ops.norm.rmsNorm(s.q, s.q, blk.q_norm, cfg.eps);
            ops.norm.rmsNorm(s.k, s.k, blk.k_norm, cfg.eps);
            for ([_][]f32{ s.q, s.k }) |qk| {
                ops.rope.applyRotateHalfPosSpan(qk, freqs, pos_x, cfg.n_heads, hd, 0, hd / 2);
                ops.rope.applyRotateHalfPosSpan(qk, freqs, pos_y, cfg.n_heads, hd, hd / 2, hd / 2);
            }
            // weightless V-RMSNorm (gemma4v quirk).
            ops.norm.rmsNormUnit(s.v, s.v, hd, cfg.eps);
            try ops.attention.attention(io, gpa, s.attn, s.q, s.k, s.v, .{
                .seq_q = np,
                .seq_kv = np,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = hd,
                .causal = false,
                .scale = 1.0, // gemma4v sets kq_scale = 1.0 (QK-norm handles scaling)
            });
            try ops.matmul.matmul(io, gpa, s.tmp, s.attn, np, blk.out, null);
            ops.norm.rmsNorm(s.tmp, s.tmp, blk.attn_post_norm_w, cfg.eps);
            for (x, s.tmp) |*xi, ti| xi.* += ti;

            // --- FFN (GeGLU-quick) ---
            ops.norm.rmsNorm(s.normed, x, blk.ln2_w, cfg.eps);
            try ops.matmul.matmul(io, gpa, s.gate, s.normed, np, blk.gate, null);
            try ops.matmul.matmul(io, gpa, s.up, s.normed, np, blk.up, null);
            ops.act.geluQuickMul(s.gate, s.up);
            try ops.matmul.matmul(io, gpa, s.tmp, s.gate, np, blk.down, null);
            ops.norm.rmsNorm(s.tmp, s.tmp, blk.ffn_post_norm_w, cfg.eps);
            for (x, s.tmp) |*xi, ti| xi.* += ti;
        }
        return self.project(io, gpa, x, gx, gy);
    }

    /// Preprocessed patch matrix + grid dims, shared by the CPU `encode` and the
    /// CUDA encoder (which uploads `patches` for the device patch-embed GEMM).
    pub const Prepared = struct {
        /// [gx*gy][3*patch*patch] im2col patch matrix, row-major patch order.
        patches: []f32,
        gx: usize,
        gy: usize,
        pub fn np(self: Prepared) usize {
            return self.gx * self.gy;
        }
        pub fn deinit(self: *Prepared, gpa: std.mem.Allocator) void {
            gpa.free(self.patches);
            self.* = undefined;
        }
    };

    /// Smart-resize + normalize + im2col an interleaved-RGB image to a patch
    /// matrix and its pre-pool grid dims.
    pub fn prepare(self: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Prepared {
        const cfg = self.cfg;
        const target = targetSize(width, height, cfg.alignPx(), cfg.max_tokens);
        const gx = target.w / cfg.patch;
        const gy = target.h / cfg.patch;
        const patches = try patchMatrix(gpa, rgb, width, height, target.w, target.h, cfg.patch, gx, gy);
        return .{ .patches = patches, .gx = gx, .gy = gy };
    }

    /// Debug: the EXACT pixels the tower ingests (aspect-preserving resize to the
    /// token-budget grid, reconstructed from the `2·p/255−1` normalization back to
    /// interleaved RGB u8). No crop and no bars — the whole image is present.
    /// Returns owned pixels + the resized dims.
    pub fn preprocessedRgb(self: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !struct { pixels: []u8, width: usize, height: usize } {
        const cfg = self.cfg;
        const target = targetSize(width, height, cfg.alignPx(), cfg.max_tokens);
        const tw = target.w;
        const th = target.h;
        const chw = try preprocess(gpa, rgb, width, height, tw, th);
        defer gpa.free(chw);
        const out = try gpa.alloc(u8, tw * th * 3);
        errdefer gpa.free(out);
        for (0..th) |y| {
            for (0..tw) |x| {
                for (0..3) |c| {
                    const v = chw[c * tw * th + y * tw + x]; // in [-1, 1]
                    const p = std.math.clamp((v + 1.0) * 0.5 * 255.0, 0.0, 255.0);
                    out[(y * tw + x) * 3 + c] = @intFromFloat(p);
                }
            }
        }
        return .{ .pixels = out, .width = tw, .height = th };
    }

    /// Owned [gx*gy][dim] learned 2-D position embeddings: patch i at grid
    /// (col = i%gx, row = i/gx) gets table_x[col] + table_y[row] (table_y starts
    /// at row `pos_size`). Added to the patch-embed output before the blocks.
    pub fn posEmbedRows(self: *const Vit, gpa: std.mem.Allocator, gx: usize, gy: usize) ![]f32 {
        const dim = self.cfg.dim;
        const out = try gpa.alloc(f32, gx * gy * dim);
        errdefer gpa.free(out);
        for (0..gx * gy) |t| {
            const col = t % gx;
            const row = t / gx;
            const ex = self.pos_embd[col * dim ..][0..dim];
            const ey = self.pos_embd[(self.pos_size + row) * dim ..][0..dim];
            const dst = out[t * dim ..][0..dim];
            for (dst, ex, ey) |*d, a, b| d.* = a + b;
        }
        return out;
    }

    /// Gemma-4 multimodal projector over the post-block patch states `x`
    /// ([gx*gy][dim], row-major patch order): 3x3 average-pool merge ->
    /// x*sqrt(dim) -> (x - std_bias)*std_scale -> weightless RMSNorm ->
    /// mm.input_projection. Shared by the CPU encode and a future CUDA encode
    /// (which would run the blocks device-side and project here — cheap).
    pub fn project(self: *const Vit, io: std.Io, gpa: std.mem.Allocator, x: []const f32, gx: usize, gy: usize) !Encoded {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const m = cfg.merge;
        const tx = gx / m;
        const ty = gy / m;
        const nt = tx * ty;
        const pooled = try gpa.alloc(f32, nt * dim);
        defer gpa.free(pooled);
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(m * m));
        const sqrt_dim: f32 = @sqrt(@as(f32, @floatFromInt(dim)));
        for (0..ty) |oy| {
            for (0..tx) |ox| {
                const dst = pooled[(oy * tx + ox) * dim ..][0..dim];
                @memset(dst, 0);
                for (0..m) |dy| {
                    for (0..m) |dx| {
                        const src = x[((oy * m + dy) * gx + (ox * m + dx)) * dim ..][0..dim];
                        for (dst, src) |*d, sv| d.* += sv;
                    }
                }
                // avg-pool, x*sqrt(dim), then per-channel std affine.
                for (dst, self.std_bias, self.std_scale) |*d, b, sc| d.* = (d.* * inv * sqrt_dim - b) * sc;
            }
        }
        // embedding_pre_projection_norm: weightless RMS over dim.
        ops.norm.rmsNormUnit(pooled, pooled, dim, cfg.eps);

        const embeds = try gpa.alloc(f32, nt * cfg.proj_dim);
        errdefer gpa.free(embeds);
        try ops.matmul.matmul(io, gpa, embeds, pooled, nt, self.mm_proj, null);
        return .{ .embeds = embeds, .grid_w = tx, .grid_h = ty };
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
    gate: []f32,
    up: []f32,

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
            np * cfg.ffn, // gate
            np * cfg.ffn, // up
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

const Size = struct { w: usize, h: usize };

/// Google's `gemma4_vision_token_budget` resize: scale the image (aspect-
/// preserving) so its post-merge token count targets `max_tokens`, then floor
/// each dimension to a multiple of `alignp` (= patch*merge = 48). No crop, no
/// pad — the tiny per-dimension floor drift is absorbed by the grid. `f > 1`
/// upscales small images to fill the budget, exactly as Google's tool does.
fn targetSize(w: usize, h: usize, alignp: usize, max_tokens: usize) Size {
    const m: f64 = @floatFromInt(alignp);
    const wf: f64 = @floatFromInt(w);
    const hf: f64 = @floatFromInt(h);
    const t: f64 = @floatFromInt(max_tokens * alignp * alignp); // nMax * m^2
    const f = @sqrt(t / (wf * hf));
    const wb = @max(alignp, @as(usize, @intFromFloat(@floor(f * wf / m))) * alignp);
    const hb = @max(alignp, @as(usize, @intFromFloat(@floor(f * hf / m))) * alignp);
    return .{ .w = wb, .h = hb };
}

/// Build the im2col patch matrix ([gx*gy][3*patch*patch], row-major patch order)
/// from preprocessed planar CHW pixels. Row layout is [c][ky][kx], matching the
/// GGUF conv weight [dim][c*patch*patch].
fn patchMatrix(gpa: std.mem.Allocator, rgb: []const u8, sw: usize, sh: usize, tw: usize, th: usize, patch: usize, gx: usize, gy: usize) ![]f32 {
    const kdim = 3 * patch * patch;
    const chw = try preprocess(gpa, rgb, sw, sh, tw, th);
    defer gpa.free(chw);
    const patches = try gpa.alloc(f32, gx * gy * kdim);
    errdefer gpa.free(patches);
    for (0..gy) |grow| {
        for (0..gx) |gcol| {
            const row = patches[(grow * gx + gcol) * kdim ..][0..kdim];
            for (0..3) |c| {
                for (0..patch) |ky| {
                    const src = chw[c * tw * th + (grow * patch + ky) * tw + gcol * patch ..][0..patch];
                    @memcpy(row[(c * patch + ky) * patch ..][0..patch], src);
                }
            }
        }
    }
    return patches;
}

/// Resize `rgb` (interleaved, sw x sh) DIRECTLY to `tw` x `th` (no crop, no pad
/// — Google's `gemma4_vision_token_budget` uses a plain aspect-preserving canvas
/// resize; `tw`/`th` already encode the near-source aspect) and normalize to
/// planar CHW f32 with `2*(p/255) - 1`. Align-corners bilinear in f32.
fn preprocess(gpa: std.mem.Allocator, rgb: []const u8, sw: usize, sh: usize, tw: usize, th: usize) ![]f32 {
    const out = try gpa.alloc(f32, 3 * tw * th);
    errdefer gpa.free(out);
    const x_ratio: f64 = if (tw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(tw - 1)) else 0;
    const y_ratio: f64 = if (th > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(th - 1)) else 0;
    for (0..th) |y| {
        const pyf = @as(f64, @floatFromInt(y)) * y_ratio;
        const y0 = @min(@as(usize, @intFromFloat(pyf)), sh - 1);
        const y1 = @min(y0 + 1, sh - 1);
        const yf = pyf - @as(f64, @floatFromInt(y0));
        for (0..tw) |x| {
            const pxf = @as(f64, @floatFromInt(x)) * x_ratio;
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
                const p = top + (bottom - top) * yf; // f32 sample, no u8 round-trip
                out[c * tw * th + y * tw + x] = @floatCast(2.0 * (p / 255.0) - 1.0);
            }
        }
    }
    return out;
}

// --- tests -----------------------------------------------------------------

const mmproj_path = "/home/qt/genai/lmstudio/models/DarkIdol-Gemma-4-31B-it.mmproj-Q8_0.gguf";

test "targetSize matches Google gemma4_vision_token_budget" {
    // nMax=280: f=sqrt(280*48^2/1e6)=0.8032 -> floor(0.8032*1000/48)*48=768.
    const s = targetSize(1000, 1000, 48, 280);
    try std.testing.expectEqual(@as(usize, 768), s.w);
    try std.testing.expectEqual(@as(usize, 768), s.h);
    // Token count never exceeds the budget, and the merged grid is 48-aligned.
    try std.testing.expect((s.w / 16) * (s.h / 16) / 9 <= 280);
    // Aspect-preserving (no pad/crop): a 3:1 landscape source stays landscape.
    const wide = targetSize(1200, 400, 48, 280);
    try std.testing.expect(wide.w % 48 == 0 and wide.h % 48 == 0);
    try std.testing.expect(wide.w > wide.h);
    try std.testing.expect((wide.w / 16) * (wide.h / 16) / 9 <= 280);
    // A smaller budget yields fewer pixels; every dim stays >= one macro-block.
    const lo = targetSize(1000, 1000, 48, 70);
    try std.testing.expect(lo.w < s.w and lo.w >= 48);
}

test "gemma4v vit loads from real 31B mmproj" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, mmproj_path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, mmproj_path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    const cfg = vit.cfg;
    try std.testing.expectEqual(@as(usize, 1152), cfg.dim);
    try std.testing.expectEqual(@as(usize, 16), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 72), cfg.headDim());
    try std.testing.expectEqual(@as(usize, 4304), cfg.ffn);
    try std.testing.expectEqual(@as(usize, 27), cfg.n_blocks);
    try std.testing.expectEqual(@as(usize, 16), cfg.patch);
    try std.testing.expectEqual(@as(usize, 5376), cfg.proj_dim);
    try std.testing.expectEqual(@as(usize, 3), cfg.merge);
    try std.testing.expectEqual(@as(usize, 27), vit.blocks.len);
    try std.testing.expectEqual(@as(usize, 10240), vit.pos_size);
}
