//! Qwen3-VL / Qwen3.5 vision tower (mmproj GGUF, projector "qwen3vl_merger").
//! Ported from llama.cpp tools/mtmd (clip.cpp + models/qwen3vl.cpp +
//! mtmd-image.cpp) — CPU forward, f32 compute over the bf16/f32 mmproj.
//!
//! Pipeline: smart-resize (multiples of patch*merge=32, aspect-preserving
//! fit + center pad, align-corners bilinear on u8) -> normalize
//! ((p/255 - mean)/std) -> patch embed (two 16x16 convs over the same image,
//! summed — temporal_patch_size 2 for a still — as an im2col GEMM) with
//! tokens produced directly in the 2x2-merged block order -> interpolated
//! 48x48 learned position embedding (bilinear+antialias, align_corners
//! false) -> 27 pre-LN blocks (fused qkv+bias, 16 heads x 72, 2-D vision
//! RoPE over pairs (d, d+36): pairs 0..17 keyed by patch row, 18..35 by
//! patch column, base 10000; full non-causal attention; GELU-tanh FFN)
//! -> post_ln -> 2x2 merge (4 consecutive tokens -> 4608) -> mm.0 + GELU ->
//! mm.2 -> [n][5120] embeddings for the LLM.
//!
//! No deepstack (the Qwen3.6 mmproj flags none) and no window attention.

const std = @import("std");
const gguf_mod = @import("tp_core").gguf;
const test_gate = @import("../test_gate.zig");
const weights_mod = @import("tp_core").weights;
const ops = @import("tp_ops");
const loader = @import("loader.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    n_blocks: usize,
    dim: usize,
    n_heads: usize,
    ffn: usize,
    patch: usize,
    merge: usize,
    /// Side of the learned position grid (sqrt of position count).
    pos_grid: usize,
    proj_dim: usize,
    image_mean: [3]f32,
    image_std: [3]f32,
    eps: f32,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }

    pub fn detect(g: *const Gguf) !Config {
        const proj = g.getStr("clip.projector_type") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, proj, "qwen3vl_merger")) return error.UnknownModelConfig;
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
        const proj_view = g.get("mm.2.bias") orelse return error.UnknownModelConfig;
        var cfg: Config = .{
            .n_blocks = try key(g, "block_count"),
            .dim = try key(g, "embedding_length"),
            .n_heads = try key(g, "attention.head_count"),
            .ffn = try key(g, "feed_forward_length"),
            .patch = try key(g, "patch_size"),
            .merge = try key(g, "spatial_merge_size"),
            .pos_grid = 0,
            .proj_dim = proj_view.info.elemCount(),
            .image_mean = mean,
            .image_std = stdv,
            .eps = @floatCast(g.getFloat("clip.vision.attention.layer_norm_epsilon") orelse 1e-6),
        };
        const pos = g.get("v.position_embd.weight") orelse return error.UnknownModelConfig;
        const pshape = pos.info.shape.slice();
        if (pshape.len != 2 or pshape[1] != cfg.dim) return error.UnknownModelConfig;
        cfg.pos_grid = std.math.sqrt(pshape[0]);
        if (cfg.pos_grid * cfg.pos_grid != pshape[0]) return error.UnknownModelConfig;
        if (cfg.merge != 2) return error.UnknownModelConfig; // token order is 2x2-specific
        return cfg;
    }
};

const Block = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    qkv: Weight,
    qkv_b: []const f32,
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
    /// Summed still-image patch kernel (W0 + W1), f32 [dim][3*patch*patch].
    patch_w: []const f32,
    patch_b: []const f32,
    /// Learned positions, f32 [pos_grid*pos_grid][dim].
    pos_embd: []const f32,
    blocks: []Block,
    post_ln_w: []const f32,
    post_ln_b: []const f32,
    mm0: Weight,
    mm0_b: []const f32,
    mm2: Weight,
    mm2_b: []const f32,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Vit {
        const cfg = try Config.detect(g);
        if (g.getArr("clip.vision.is_deepstack_layers")) |arr| {
            var it = arr.iterate();
            while (it.next()) |v| if (v.boolean) return error.UnsupportedModelConfig; // deepstack not implemented
        }
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const kdim = 3 * cfg.patch * cfg.patch;
        const w0 = try loader.vector(alloc, store, "v.patch_embd.weight", cfg.dim * kdim);
        const w1 = try loader.vector(alloc, store, "v.patch_embd.weight.1", cfg.dim * kdim);
        const patch_w = try alloc.alloc(f32, cfg.dim * kdim);
        for (patch_w, w0, w1) |*o, a, b| o.* = a + b;

        const blocks = try alloc.alloc(Block, cfg.n_blocks);
        for (blocks, 0..) |*blk, i| {
            blk.* = .{
                .ln1_w = try loader.indexedVector(alloc, store, "v.blk.", i, "ln1.weight", cfg.dim),
                .ln1_b = try loader.indexedVector(alloc, store, "v.blk.", i, "ln1.bias", cfg.dim),
                .qkv = try loader.indexedMatrix(store, "v.blk.", i, "attn_qkv.weight", 3 * cfg.dim, cfg.dim),
                .qkv_b = try loader.indexedVector(alloc, store, "v.blk.", i, "attn_qkv.bias", 3 * cfg.dim),
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

        const merged_dim = cfg.dim * cfg.merge * cfg.merge;
        // All arena allocations must happen BEFORE `.arena = arena` copies
        // the arena state into the result — later allocations would extend a
        // buffer list the snapshot doesn't know about and leak at deinit.
        const patch_b = try loader.vector(alloc, store, "v.patch_embd.bias", cfg.dim);
        const pos_embd = try loader.vector(alloc, store, "v.position_embd.weight", cfg.pos_grid * cfg.pos_grid * cfg.dim);
        const post_ln_w = try loader.vector(alloc, store, "v.post_ln.weight", cfg.dim);
        const post_ln_b = try loader.vector(alloc, store, "v.post_ln.bias", cfg.dim);
        const mm0_b = try loader.vector(alloc, store, "mm.0.bias", merged_dim);
        const mm2_b = try loader.vector(alloc, store, "mm.2.bias", cfg.proj_dim);
        return .{
            .arena = arena,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embd = pos_embd,
            .blocks = blocks,
            .post_ln_w = post_ln_w,
            .post_ln_b = post_ln_b,
            .mm0 = try loader.matrix(store, "mm.0.weight", merged_dim, merged_dim),
            .mm0_b = mm0_b,
            .mm2 = try loader.matrix(store, "mm.2.weight", cfg.proj_dim, merged_dim),
            .mm2_b = mm2_b,
        };
    }

    pub fn deinit(self: *Vit) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub const Encoded = struct {
        /// [grid_w * grid_h][proj_dim] merged-token embeddings.
        embeds: []f32,
        /// Merged grid dims (patches / merge).
        grid_w: usize,
        grid_h: usize,

        pub fn deinit(self: *Encoded, gpa: std.mem.Allocator) void {
            gpa.free(self.embeds);
            self.* = undefined;
        }
    };

    /// Host-side prep shared by the CPU and CUDA encode paths: smart resize
    /// + preprocess, the patch matrix in 2x2-merged token order (with
    /// per-token patch coords), the interpolated position table, and the
    /// 2-D vision-rope cos/sin tables.
    pub const Prepared = struct {
        /// Patch grid dims (pre-merge).
        pw: usize,
        ph: usize,
        /// [np][3*patch*patch] patch rows, merged token order.
        patches: []f32,
        /// Per-token patch coordinates (position lookup + rope).
        py: []u32,
        px: []u32,
        /// [ph][pw][dim] interpolated learned positions (grid order).
        pos: []f32,
        /// [max(pw,ph)][half] rope tables; half = headDim()/4 pairs per axis.
        rope_cos: []f32,
        rope_sin: []f32,

        pub fn np(self: *const Prepared) usize {
            return self.pw * self.ph;
        }

        pub fn deinit(self: *Prepared, gpa: std.mem.Allocator) void {
            gpa.free(self.patches);
            gpa.free(self.py);
            gpa.free(self.px);
            gpa.free(self.pos);
            gpa.free(self.rope_cos);
            gpa.free(self.rope_sin);
            self.* = undefined;
        }
    };

    pub fn prepare(self: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Prepared {
        const cfg = self.cfg;
        const align_px = cfg.patch * cfg.merge;

        // Smart resize to multiples of patch*merge within the token budget.
        const target = smartResize(width, height, align_px, 8 * align_px * align_px, 4096 * align_px * align_px);
        const chw = try preprocess(gpa, rgb, width, height, target.w, target.h, cfg.image_mean, cfg.image_std);
        defer gpa.free(chw);

        const pw = target.w / cfg.patch;
        const ph = target.h / cfg.patch;
        const np = pw * ph;

        // Patch matrix in the 2x2-merged token order, plus per-token patch
        // coordinates for the vision rope.
        const kdim = 3 * cfg.patch * cfg.patch;
        const patches = try gpa.alloc(f32, np * kdim);
        errdefer gpa.free(patches);
        const py = try gpa.alloc(u32, np);
        errdefer gpa.free(py);
        const px = try gpa.alloc(u32, np);
        errdefer gpa.free(px);
        {
            var t: usize = 0;
            var y: usize = 0;
            while (y < ph) : (y += 2) {
                var x: usize = 0;
                while (x < pw) : (x += 2) {
                    for (0..2) |dy| {
                        for (0..2) |dx| {
                            const gy = y + dy;
                            const gx = x + dx;
                            py[t] = @intCast(gy);
                            px[t] = @intCast(gx);
                            const row = patches[t * kdim ..][0..kdim];
                            for (0..3) |c| {
                                for (0..cfg.patch) |ky| {
                                    const src = chw[c * target.h * target.w + (gy * cfg.patch + ky) * target.w + gx * cfg.patch ..][0..cfg.patch];
                                    @memcpy(row[(c * cfg.patch + ky) * cfg.patch ..][0..cfg.patch], src);
                                }
                            }
                            t += 1;
                        }
                    }
                }
            }
        }

        const pos_interp = try interpolatePos(gpa, self.pos_embd, cfg.pos_grid, cfg.dim, pw, ph);
        errdefer gpa.free(pos_interp);

        // Vision rope tables: pair p of 2*half pairs, theta = pos * base^(-2p'/n_dims)
        // with p' resetting at the section boundary (pairs 0..half-1 keyed by
        // row, half..2*half-1 by column); rotation pairs (d, d + n_dims).
        const n_dims = cfg.headDim() / 2; // 36
        const half = n_dims / 2; // 18 pairs per axis
        const max_pos = @max(pw, ph);
        const rope_cos = try gpa.alloc(f32, max_pos * half);
        errdefer gpa.free(rope_cos);
        const rope_sin = try gpa.alloc(f32, max_pos * half);
        errdefer gpa.free(rope_sin);
        for (0..max_pos) |p| {
            for (0..half) |i| {
                const theta = @as(f64, @floatFromInt(p)) * std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_dims)));
                rope_cos[p * half + i] = @floatCast(@cos(theta));
                rope_sin[p * half + i] = @floatCast(@sin(theta));
            }
        }

        return .{
            .pw = pw,
            .ph = ph,
            .patches = patches,
            .py = py,
            .px = px,
            .pos = pos_interp,
            .rope_cos = rope_cos,
            .rope_sin = rope_sin,
        };
    }

    /// Encode interleaved RGB pixels to LLM embeddings.
    pub fn encode(self: *const Vit, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Encoded {
        const cfg = self.cfg;
        const hd = cfg.headDim();
        const half = hd / 4;

        var prep = try self.prepare(gpa, rgb, width, height);
        defer prep.deinit(gpa);
        const pw = prep.pw;
        const ph = prep.ph;
        const np = prep.np();
        const py = prep.py;
        const px = prep.px;
        const kdim = 3 * cfg.patch * cfg.patch;

        // Patch embed GEMM + interpolated position embedding.
        var x = try gpa.alloc(f32, np * cfg.dim);
        defer gpa.free(x);
        try ops.matmul.matmul(io, gpa, x, prep.patches, np, Weight.fromF32(self.patch_w, cfg.dim, kdim), self.patch_b);
        for (0..np) |t| {
            const src = prep.pos[(py[t] * pw + px[t]) * cfg.dim ..][0..cfg.dim];
            const dst = x[t * cfg.dim ..][0..cfg.dim];
            for (dst, src) |*d, s| d.* += s;
        }

        const rope_cos = prep.rope_cos;
        const rope_sin = prep.rope_sin;

        var s = try Scratch.init(gpa, np, cfg);
        defer s.deinit(gpa);

        for (self.blocks) |*blk| {
            ops.norm.layerNorm(s.normed, x, blk.ln1_w, blk.ln1_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, s.qkv, s.normed, np, blk.qkv, blk.qkv_b);
            for (0..np) |t| {
                @memcpy(s.q[t * cfg.dim ..][0..cfg.dim], s.qkv[t * 3 * cfg.dim ..][0..cfg.dim]);
                @memcpy(s.k[t * cfg.dim ..][0..cfg.dim], s.qkv[t * 3 * cfg.dim + cfg.dim ..][0..cfg.dim]);
                @memcpy(s.v[t * cfg.dim ..][0..cfg.dim], s.qkv[t * 3 * cfg.dim + 2 * cfg.dim ..][0..cfg.dim]);
            }
            applyVisionRope(s.q, np, cfg.n_heads, hd, py, px, rope_cos, rope_sin, half);
            applyVisionRope(s.k, np, cfg.n_heads, hd, py, px, rope_cos, rope_sin, half);
            try ops.attention.attention(io, gpa, s.attn, s.q, s.k, s.v, .{
                .seq_q = np,
                .seq_kv = np,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = hd,
                .causal = false,
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

        // 2x2 merge (tokens are already block-ordered: reinterpret rows of 4)
        // then the two-layer projector.
        const nm = np / 4;
        const merged_dim = cfg.dim * 4;
        const h0 = try gpa.alloc(f32, nm * merged_dim);
        defer gpa.free(h0);
        try ops.matmul.matmul(io, gpa, h0, x, nm, self.mm0, self.mm0_b);
        ops.act.geluTanh(h0);
        const embeds = try gpa.alloc(f32, nm * cfg.proj_dim);
        errdefer gpa.free(embeds);
        try ops.matmul.matmul(io, gpa, embeds, h0, nm, self.mm2, self.mm2_b);

        return .{ .embeds = embeds, .grid_w = pw / 2, .grid_h = ph / 2 };
    }
};

/// The 2-D vision rope: per token, pairs (d, d+n_dims) for d < n_dims, the
/// first `half` pairs keyed by the token's patch row, the next `half` by its
/// column (llama.cpp GGML_ROPE_TYPE_VISION with sections {18,18,18,18}).
/// Pub as the CPU reference for the CUDA rope_vision kernel.
pub fn applyVisionRope(qk: []f32, np: usize, n_heads: usize, hd: usize, py: []const u32, px: []const u32, cos: []const f32, sin: []const f32, half: usize) void {
    const n_dims = hd / 2;
    for (0..np) |t| {
        const yc = cos[py[t] * half ..][0..half];
        const ys = sin[py[t] * half ..][0..half];
        const xc = cos[px[t] * half ..][0..half];
        const xs = sin[px[t] * half ..][0..half];
        for (0..n_heads) |h| {
            const base = (t * n_heads + h) * hd;
            for (0..n_dims) |d| {
                const c = if (d < half) yc[d] else xc[d - half];
                const sn = if (d < half) ys[d] else xs[d - half];
                const lo = qk[base + d];
                const hi = qk[base + n_dims + d];
                qk[base + d] = lo * c - hi * sn;
                qk[base + n_dims + d] = hi * c + lo * sn;
            }
        }
    }
}

pub const TargetSize = struct { w: usize, h: usize };

/// HF Qwen smart_resize: round to multiples of `align_px`, scaled into the
/// [min_pixels, max_pixels] budget (llama.cpp mtmd-image.cpp
/// calc_size_preserved_ratio).
pub fn smartResize(width: usize, height: usize, align_px: usize, min_pixels: usize, max_pixels: usize) TargetSize {
    const af: f64 = @floatFromInt(align_px);
    const wf: f64 = @floatFromInt(width);
    const hf: f64 = @floatFromInt(height);
    var w_bar = @max(af, @round(wf / af) * af);
    var h_bar = @max(af, @round(hf / af) * af);
    if (h_bar * w_bar > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(hf * wf / @as(f64, @floatFromInt(max_pixels)));
        h_bar = @max(af, @floor(hf / beta / af) * af);
        w_bar = @max(af, @floor(wf / beta / af) * af);
    } else if (h_bar * w_bar < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (hf * wf));
        h_bar = @ceil(hf * beta / af) * af;
        w_bar = @ceil(wf * beta / af) * af;
    }
    return .{ .w = @intFromFloat(w_bar), .h = @intFromFloat(h_bar) };
}

/// Resize + pad + normalize to planar CHW f32 (llama.cpp img_tool::resize
/// with add_padding=true then img_u8_to_f32): aspect-preserving
/// align-corners bilinear on u8 (truncating), centered zero padding, then
/// (p/255 - mean)/std.
fn preprocess(gpa: std.mem.Allocator, rgb: []const u8, sw: usize, sh: usize, tw: usize, th: usize, mean: [3]f32, stdv: [3]f32) ![]f32 {
    const scale = @min(
        @as(f64, @floatFromInt(tw)) / @as(f64, @floatFromInt(sw)),
        @as(f64, @floatFromInt(th)) / @as(f64, @floatFromInt(sh)),
    );
    const nw = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sw)) * scale))), tw);
    const nh = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sh)) * scale))), th);

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

    // Composite centered onto a zero canvas, then normalize to planar CHW.
    const off_x = (tw - nw) / 2;
    const off_y = (th - nh) / 2;
    const out = try gpa.alloc(f32, 3 * tw * th);
    errdefer gpa.free(out);
    for (0..3) |c| {
        const plane = out[c * tw * th ..][0 .. tw * th];
        const zero = (0.0 - mean[c]) / stdv[c];
        @memset(plane, zero);
        for (0..nh) |y| {
            for (0..nw) |xx| {
                const p: f32 = @floatFromInt(resized[(y * nw + xx) * 3 + c]);
                plane[(y + off_y) * tw + (xx + off_x)] = (p / 255.0 - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

/// Bilinear + antialias interpolation of the learned [grid][grid][dim]
/// position table to [ph][pw][dim] (ggml GGML_SCALE_MODE_BILINEAR |
/// ANTIALIAS, align_corners=false, normalized triangle filter).
fn interpolatePos(gpa: std.mem.Allocator, pos: []const f32, grid: usize, dim: usize, pw: usize, ph: usize) ![]f32 {
    const out = try gpa.alloc(f32, pw * ph * dim);
    errdefer gpa.free(out);
    const gf: f64 = @floatFromInt(grid);
    const sf0 = @as(f64, @floatFromInt(pw)) / gf;
    const sf1 = @as(f64, @floatFromInt(ph)) / gf;
    const support0 = @max(1.0, 1.0 / sf0);
    const support1 = @max(1.0, 1.0 / sf1);
    const invscale0 = 1.0 / support0;
    const invscale1 = 1.0 / support1;
    const gi: i64 = @intCast(grid);

    for (0..ph) |oy| {
        const y = (@as(f64, @floatFromInt(oy)) + 0.5) / sf1;
        const y_min: usize = @intCast(std.math.clamp(@as(i64, @intFromFloat(y - support1 + 0.5)), 0, gi));
        const y_max: usize = @intCast(std.math.clamp(@as(i64, @intFromFloat(y + support1 + 0.5)), 0, gi));
        for (0..pw) |ox| {
            const xq = (@as(f64, @floatFromInt(ox)) + 0.5) / sf0;
            const x_min: usize = @intCast(std.math.clamp(@as(i64, @intFromFloat(xq - support0 + 0.5)), 0, gi));
            const x_max: usize = @intCast(std.math.clamp(@as(i64, @intFromFloat(xq + support0 + 0.5)), 0, gi));
            const dst = out[(oy * pw + ox) * dim ..][0..dim];
            @memset(dst, 0);
            var total: f64 = 0;
            for (y_min..y_max) |sy| {
                const wy = triangle((@as(f64, @floatFromInt(sy)) - y + 0.5) * invscale1);
                for (x_min..x_max) |sx| {
                    const w = triangle((@as(f64, @floatFromInt(sx)) - xq + 0.5) * invscale0) * wy;
                    if (w <= 0) continue;
                    total += w;
                    const src = pos[(sy * grid + sx) * dim ..][0..dim];
                    const wf: f32 = @floatCast(w);
                    for (dst, src) |*d, sv| d.* += sv * wf;
                }
            }
            if (total > 0) {
                const inv: f32 = @floatCast(1.0 / total);
                for (dst) |*d| d.* *= inv;
            }
        }
    }
    return out;
}

fn triangle(x: f64) f64 {
    return @max(1.0 - @abs(x), 0.0);
}

const Scratch = struct {
    normed: []f32,
    qkv: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    ffn: []f32,
    tmp: []f32,

    fn init(gpa: std.mem.Allocator, np: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        const sizes = [_]usize{
            np * cfg.dim, // normed
            np * 3 * cfg.dim, // qkv
            np * cfg.dim, // q
            np * cfg.dim, // k
            np * cfg.dim, // v
            np * cfg.dim, // attn
            np * cfg.ffn, // ffn
            np * cfg.dim, // tmp
        };
        var done: usize = 0;
        errdefer {
            inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
                if (i < done) gpa.free(@field(s, f.name));
            }
        }
        inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try gpa.alloc(f32, sizes[i]);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        inline for (@typeInfo(Scratch).@"struct".fields) |f| {
            gpa.free(@field(s, f.name));
        }
        s.* = undefined;
    }
};

// --- tests -----------------------------------------------------------------

test "smart resize matches the reference rules" {
    // 768x768 rounds to itself (no resize).
    try std.testing.expectEqual(TargetSize{ .w = 768, .h = 768 }, smartResize(768, 768, 32, 8 * 1024, 4096 * 1024));
    // Small images scale UP to the min budget, aligned to 32.
    const up = smartResize(40, 40, 32, 8 * 1024, 4096 * 1024);
    try std.testing.expect(up.w % 32 == 0 and up.h % 32 == 0);
    try std.testing.expect(up.w * up.h >= 8 * 1024);
    // Huge images scale DOWN under the max budget.
    const down = smartResize(10000, 8000, 32, 8 * 1024, 4096 * 1024);
    try std.testing.expect(down.w % 32 == 0 and down.h % 32 == 0);
    try std.testing.expect(down.w * down.h <= 4096 * 1024);
}

test "position interpolation is identity at the native grid" {
    const gpa = std.testing.allocator;
    const grid = 4;
    const dim = 3;
    var pos: [grid * grid * dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(9);
    for (&pos) |*p| p.* = prng.random().floatNorm(f32);
    const out = try interpolatePos(gpa, &pos, grid, dim, grid, grid);
    defer gpa.free(out);
    for (pos, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-6);
}

// Config + wiring against the real Qwen3.6 mmproj; skipped when absent.
test "vit loads from real qwen3.6 mmproj" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/mmproj-Qwen3.6-27B-BF16.gguf";
    try test_gate.requireModelFile(io, path);

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    try std.testing.expectEqual(@as(usize, 27), vit.cfg.n_blocks);
    try std.testing.expectEqual(@as(usize, 1152), vit.cfg.dim);
    try std.testing.expectEqual(@as(usize, 16), vit.cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 72), vit.cfg.headDim());
    try std.testing.expectEqual(@as(usize, 48), vit.cfg.pos_grid);
    try std.testing.expectEqual(@as(usize, 5120), vit.cfg.proj_dim);

    // A tiny 64x64 gradient image end-to-end: 4 patches wide -> 2x2 merged.
    var pixels: [64 * 64 * 3]u8 = undefined;
    for (0..64) |y| {
        for (0..64) |x| {
            const i = (y * 64 + x) * 3;
            pixels[i] = @intCast(x * 4);
            pixels[i + 1] = @intCast(y * 4);
            pixels[i + 2] = 128;
        }
    }
    // 64x64 is under the min budget: smart resize scales it up.
    var enc = try vit.encode(io, gpa, &pixels, 64, 64);
    defer enc.deinit(gpa);
    try std.testing.expect(enc.grid_w >= 2 and enc.grid_h >= 2);
    try std.testing.expectEqual(enc.grid_w * enc.grid_h * vit.cfg.proj_dim, enc.embeds.len);
    for (enc.embeds) |e| try std.testing.expect(std.math.isFinite(e));
}
