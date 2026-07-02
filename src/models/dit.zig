//! Krea 2 diffusion transformer (`SingleStreamDiT`), mirroring
//! comfy/ldm/krea2/model.py.
//!
//! Single-stream MMDiT: text tokens (from the txtfusion adapter) and 2x2
//! patchified latent tokens form one sequence through 28 identical blocks
//! with AdaLN-single modulation, GQA (48/12, head_dim 128) with per-head
//! QK-norm and sigmoid-gated output, SwiGLU MLPs, and 3-axis interleaved RoPE
//! (theta 1000, dims 32/48/48; text tokens sit at position (0,0,0)).
//!
//! All checkpoint tensors are raw fp8-e4m3 (no scales); large weights stay
//! fp8 and dequantize inside the GEMM, small vectors (norm scales, biases,
//! modulation) are dequantized to f32 at load. Norms use the `(1 + scale)`
//! convention with eps 1e-5, folded into the weight at load time. The
//! safetensors mapping must outlive the model.

const std = @import("std");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");

const SafeTensors = safetensors.SafeTensors;
const Weight = ops.matmul.Weight;

pub const features = 6144;
pub const tdim = 256;
pub const n_heads = 48;
pub const n_kv_heads = 12;
pub const head_dim = 128;
pub const n_blocks = 28;
pub const patch = 2;
pub const channels = 16;
pub const mlp_dim = 16384;
pub const rope_theta: f64 = 1000.0;
pub const rope_axes = [3]usize{ 32, 48, 48 };

pub const txt_dim = 2560;
pub const txt_layers = 12;
pub const txt_heads = 20;
pub const txt_mlp_dim = 6912;

const rms_eps: f32 = 1e-5;

const Attn = struct {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    gate: Weight,
    qnorm: []const f32, // effective (1+scale), [head_dim]
    knorm: []const f32,
    heads: usize,
    kv_heads: usize,
};

const Swiglu = struct {
    gate: Weight,
    up: Weight,
    down: Weight,
};

const Block = struct {
    mod: []const f32, // [6 * features]
    prenorm: []const f32,
    postnorm: []const f32,
    attn: Attn,
    mlp: Swiglu,
};

const TxtBlock = struct {
    prenorm: []const f32,
    postnorm: []const f32,
    attn: Attn,
    mlp: Swiglu,
};

const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    first: LinearW, // 64 -> 6144
    blocks: []Block,
    tmlp0: LinearW, // 256 -> 6144
    tmlp2: LinearW, // 6144 -> 6144
    tproj1: LinearW, // 6144 -> 6*6144
    txt_layerwise: [2]TxtBlock,
    txt_projector: Weight, // [1, 12]
    txt_refiner: [2]TxtBlock,
    txtmlp_norm: []const f32,
    txtmlp1: LinearW, // 2560 -> 6144
    txtmlp3: LinearW, // 6144 -> 6144
    last_norm: []const f32,
    last_mod: []const f32, // [2 * features]
    last_linear: LinearW, // 6144 -> 64

    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const blocks = try alloc.alloc(Block, n_blocks);
        for (blocks, 0..) |*blk, i| {
            blk.* = .{
                .mod = try vec(alloc, st, "blocks.{d}.mod.lin", .{i}, 6 * features),
                .prenorm = try normScale(alloc, st, "blocks.{d}.prenorm.scale", .{i}, features),
                .postnorm = try normScale(alloc, st, "blocks.{d}.postnorm.scale", .{i}, features),
                .attn = try loadAttn(alloc, st, "blocks.{d}", .{i}, features, n_heads, n_kv_heads),
                .mlp = try loadSwiglu(st, "blocks.{d}", .{i}, features, mlp_dim),
            };
        }

        var txt_layerwise: [2]TxtBlock = undefined;
        var txt_refiner: [2]TxtBlock = undefined;
        for (0..2) |i| {
            txt_layerwise[i] = try loadTxtBlock(alloc, st, "txtfusion.layerwise_blocks.{d}", i);
            txt_refiner[i] = try loadTxtBlock(alloc, st, "txtfusion.refiner_blocks.{d}", i);
        }

        const first = try loadLinear(alloc, st, "first", features, channels * patch * patch, true);
        const tmlp0 = try loadLinear(alloc, st, "tmlp.0", features, tdim, true);
        const tmlp2 = try loadLinear(alloc, st, "tmlp.2", features, features, true);
        const tproj1 = try loadLinear(alloc, st, "tproj.1", 6 * features, features, true);
        const txt_projector = try mat(st, "txtfusion.projector.weight", .{}, 1, txt_layers);
        const txtmlp_norm = try normScale(alloc, st, "txtmlp.0.scale", .{}, txt_dim);
        const txtmlp1 = try loadLinear(alloc, st, "txtmlp.1", features, txt_dim, true);
        const txtmlp3 = try loadLinear(alloc, st, "txtmlp.3", features, features, true);
        const last_norm = try normScale(alloc, st, "last.norm.scale", .{}, features);
        const last_mod = try vec(alloc, st, "last.modulation.lin", .{}, 2 * features);
        const last_linear = try loadLinear(alloc, st, "last.linear", channels * patch * patch, features, true);

        return .{
            .arena = arena,
            .first = first,
            .blocks = blocks,
            .tmlp0 = tmlp0,
            .tmlp2 = tmlp2,
            .tproj1 = tproj1,
            .txt_layerwise = txt_layerwise,
            .txt_projector = txt_projector,
            .txt_refiner = txt_refiner,
            .txtmlp_norm = txtmlp_norm,
            .txtmlp1 = txtmlp1,
            .txtmlp3 = txtmlp3,
            .last_norm = last_norm,
            .last_mod = last_mod,
            .last_linear = last_linear,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Predict velocity for one latent. `x_lat`/`out` are planar
    /// [channels][lat_h][lat_w]; `ctx` is [seq_txt][12][2560] (post-strip
    /// encoder output); `sigma` is the flow-matching timestep.
    pub fn forward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        x_lat: []const f32,
        lat_h: usize,
        lat_w: usize,
        sigma: f32,
        ctx: []const f32,
        seq_txt: usize,
    ) !void {
        std.debug.assert(lat_h % patch == 0 and lat_w % patch == 0);
        std.debug.assert(x_lat.len == channels * lat_h * lat_w);
        std.debug.assert(out.len == x_lat.len);
        std.debug.assert(ctx.len == seq_txt * txt_layers * txt_dim);
        const h = lat_h / patch;
        const w = lat_w / patch;
        const n_img = h * w;
        const seq = seq_txt + n_img;

        // Timestep path: t = tmlp(sinusoidal(sigma)); tvec = tproj(gelu(t)).
        var temb: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&temb, sigma, 10000.0);
        const t = try gpa.alloc(f32, features);
        defer gpa.free(t);
        {
            const t0 = try gpa.alloc(f32, features);
            defer gpa.free(t0);
            try linear(io, gpa, t0, &temb, 1, self.tmlp0);
            ops.act.geluTanh(t0);
            try linear(io, gpa, t, t0, 1, self.tmlp2);
        }
        const tvec = try gpa.alloc(f32, 6 * features);
        defer gpa.free(tvec);
        {
            const tg = try gpa.alloc(f32, features);
            defer gpa.free(tg);
            @memcpy(tg, t);
            ops.act.geluTanh(tg);
            try linear(io, gpa, tvec, tg, 1, self.tproj1);
        }

        // Text path: txtfusion over the 12-layer stack, then txtmlp to 6144.
        const txt_tokens = try self.txtFusion(io, gpa, ctx, seq_txt);
        defer gpa.free(txt_tokens);

        // Combined sequence: [text | image].
        const x = try gpa.alloc(f32, seq * features);
        defer gpa.free(x);
        {
            ops.norm.rmsNorm(txt_tokens, txt_tokens, self.txtmlp_norm, rms_eps);
            const mid = try gpa.alloc(f32, seq_txt * features);
            defer gpa.free(mid);
            try linear(io, gpa, mid, txt_tokens, seq_txt, self.txtmlp1);
            ops.act.geluTanh(mid);
            try linear(io, gpa, x[0 .. seq_txt * features], mid, seq_txt, self.txtmlp3);
        }
        {
            // Patchify: token (hi, wi), feature (c, ph, pw).
            const img_in = try gpa.alloc(f32, n_img * channels * patch * patch);
            defer gpa.free(img_in);
            for (0..h) |hi| {
                for (0..w) |wi| {
                    const tok = img_in[(hi * w + wi) * channels * patch * patch ..];
                    for (0..channels) |c| {
                        for (0..patch) |ph| {
                            for (0..patch) |pw| {
                                tok[c * patch * patch + ph * patch + pw] =
                                    x_lat[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)];
                            }
                        }
                    }
                }
            }
            try linear(io, gpa, x[seq_txt * features ..], img_in, n_img, self.first);
        }

        // RoPE table: text at (0,0,0), image at (0, row, col).
        var freqs = blk: {
            const pos = try gpa.alloc(f32, seq * 3);
            defer gpa.free(pos);
            @memset(pos[0 .. seq_txt * 3], 0);
            for (0..h) |hi| {
                for (0..w) |wi| {
                    const base = (seq_txt + hi * w + wi) * 3;
                    pos[base] = 0;
                    pos[base + 1] = @floatFromInt(hi);
                    pos[base + 2] = @floatFromInt(wi);
                }
            }
            break :blk try ops.rope.fluxFreqs(gpa, pos, &rope_axes, rope_theta);
        };
        defer freqs.deinit(gpa);

        for (self.blocks) |*blk| {
            try self.blockForward(io, gpa, blk, x, seq, tvec, freqs);
        }

        // Final layer on image tokens only (row-wise, so slicing first is safe).
        const img_rows = x[seq_txt * features ..];
        {
            const scale = try gpa.alloc(f32, features);
            defer gpa.free(scale);
            const shift = try gpa.alloc(f32, features);
            defer gpa.free(shift);
            for (scale, shift, t, 0..) |*sc, *sh, tv, j| {
                sc.* = tv + self.last_mod[j];
                sh.* = tv + self.last_mod[features + j];
            }
            ops.norm.rmsNorm(img_rows, img_rows, self.last_norm, rms_eps);
            var row: usize = 0;
            while (row < img_rows.len) : (row += features) {
                for (img_rows[row..][0..features], scale, shift) |*v, sc, sh| {
                    v.* = (1.0 + sc) * v.* + sh;
                }
            }
        }
        const final = try gpa.alloc(f32, n_img * channels * patch * patch);
        defer gpa.free(final);
        try linear(io, gpa, final, img_rows, n_img, self.last_linear);

        // Unpatchify.
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = final[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            out[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)] =
                                tok[c * patch * patch + ph * patch + pw];
                        }
                    }
                }
            }
        }
    }

    /// TextFusionTransformer: 2 blocks attending across the 12-layer axis per
    /// token, a Linear(12->1) collapse, then 2 refiner blocks over the tokens.
    /// Returns [seq_txt, txt_dim].
    /// Timestep path: t = tmlp(sinusoidal(sigma)); tvec = tproj(gelu(t)).
    /// Caller frees both slices.
    pub fn timestepVectors(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) !struct { t: []f32, tvec: []f32 } {
        var temb: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&temb, sigma, 10000.0);
        const t = try gpa.alloc(f32, features);
        errdefer gpa.free(t);
        {
            const t0 = try gpa.alloc(f32, features);
            defer gpa.free(t0);
            try linear(io, gpa, t0, &temb, 1, self.tmlp0);
            ops.act.geluTanh(t0);
            try linear(io, gpa, t, t0, 1, self.tmlp2);
        }
        const tvec = try gpa.alloc(f32, 6 * features);
        errdefer gpa.free(tvec);
        {
            const tg = try gpa.alloc(f32, features);
            defer gpa.free(tg);
            @memcpy(tg, t);
            ops.act.geluTanh(tg);
            try linear(io, gpa, tvec, tg, 1, self.tproj1);
        }
        return .{ .t = t, .tvec = tvec };
    }

    /// Text conditioning to combined-sequence tokens: txtfusion + txtmlp.
    /// Returns [seq_txt, features]; caller frees.
    pub fn textTokens(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        const fused = try self.txtFusion(io, gpa, ctx, seq_txt);
        defer gpa.free(fused);
        ops.norm.rmsNorm(fused, fused, self.txtmlp_norm, rms_eps);
        const mid = try gpa.alloc(f32, seq_txt * features);
        defer gpa.free(mid);
        try linear(io, gpa, mid, fused, seq_txt, self.txtmlp1);
        ops.act.geluTanh(mid);
        const out_tokens = try gpa.alloc(f32, seq_txt * features);
        errdefer gpa.free(out_tokens);
        try linear(io, gpa, out_tokens, mid, seq_txt, self.txtmlp3);
        return out_tokens;
    }

    /// Patchify a planar latent into [n_img, channels*patch^2] rows.
    pub fn patchify(gpa: std.mem.Allocator, x_lat: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const img_in = try gpa.alloc(f32, h * w * channels * patch * patch);
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = img_in[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            tok[c * patch * patch + ph * patch + pw] =
                                x_lat[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)];
                        }
                    }
                }
            }
        }
        return img_in;
    }

    /// RoPE frequency table for [text | image] positions.
    pub fn ropeFreqs(gpa: std.mem.Allocator, seq_txt: usize, h: usize, w: usize) !ops.rope.Freqs {
        const seq = seq_txt + h * w;
        const pos = try gpa.alloc(f32, seq * 3);
        defer gpa.free(pos);
        @memset(pos[0 .. seq_txt * 3], 0);
        for (0..h) |hi| {
            for (0..w) |wi| {
                const base = (seq_txt + hi * w + wi) * 3;
                pos[base] = 0;
                pos[base + 1] = @floatFromInt(hi);
                pos[base + 2] = @floatFromInt(wi);
            }
        }
        return ops.rope.fluxFreqs(gpa, pos, &rope_axes, rope_theta);
    }

    /// Final layer + unpatchify: img_rows [n_img, features] (modified in
    /// place) -> planar velocity `out`.
    pub fn finalize(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, img_rows: []f32, t: []const f32, lat_h: usize, lat_w: usize) !void {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const n_img = h * w;
        {
            const scale = try gpa.alloc(f32, features);
            defer gpa.free(scale);
            const shift = try gpa.alloc(f32, features);
            defer gpa.free(shift);
            for (scale, shift, t, 0..) |*sc, *sh, tv, j| {
                sc.* = tv + self.last_mod[j];
                sh.* = tv + self.last_mod[features + j];
            }
            ops.norm.rmsNorm(img_rows, img_rows, self.last_norm, rms_eps);
            var row: usize = 0;
            while (row < img_rows.len) : (row += features) {
                for (img_rows[row..][0..features], scale, shift) |*v, sc, sh| {
                    v.* = (1.0 + sc) * v.* + sh;
                }
            }
        }
        const final = try gpa.alloc(f32, n_img * channels * patch * patch);
        defer gpa.free(final);
        try linear(io, gpa, final, img_rows, n_img, self.last_linear);
        unpatchify(out, final, lat_h, lat_w);
    }

    /// Scatter final-layer patch tokens back into the planar latent.
    pub fn unpatchify(out: []f32, final: []const f32, lat_h: usize, lat_w: usize) void {
        const h = lat_h / patch;
        const w = lat_w / patch;
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = final[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            out[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)] =
                                tok[c * patch * patch + ph * patch + pw];
                        }
                    }
                }
            }
        }
    }

    pub fn txtFusion(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        const work = try gpa.alloc(f32, ctx.len);
        defer gpa.free(work);
        @memcpy(work, ctx);

        // seq_txt independent sequences of length 12.
        for (&self.txt_layerwise) |*blk| {
            try self.txtBlockForward(io, gpa, blk, work, seq_txt, txt_layers);
        }

        const projected = try gpa.alloc(f32, seq_txt * txt_dim);
        errdefer gpa.free(projected);
        {
            var pw: [txt_layers]f32 = undefined;
            try safetensors.convertToF32(self.txt_projector.dtype, self.txt_projector.bytes, &pw);
            for (0..seq_txt) |tok| {
                const dst = projected[tok * txt_dim ..][0..txt_dim];
                @memset(dst, 0);
                for (0..txt_layers) |l| {
                    const src = work[(tok * txt_layers + l) * txt_dim ..][0..txt_dim];
                    for (dst, src) |*d, s| d.* += pw[l] * s;
                }
            }
        }

        for (&self.txt_refiner) |*blk| {
            try self.txtBlockForward(io, gpa, blk, projected, 1, seq_txt);
        }
        return projected;
    }

    /// TextFusionBlock: x += attn(prenorm(x)); x += mlp(postnorm(x)).
    /// `x` holds n_seqs sequences of seq_len rows of txt_dim, no RoPE/mask.
    fn txtBlockForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, blk: *const TxtBlock, x: []f32, n_seqs: usize, seq_len: usize) !void {
        _ = self;
        const rows = n_seqs * seq_len;
        const normed = try gpa.alloc(f32, rows * txt_dim);
        defer gpa.free(normed);
        ops.norm.rmsNorm(normed, x, blk.prenorm, rms_eps);

        const a = try gpa.alloc(f32, rows * txt_dim);
        defer gpa.free(a);
        try attnForward(io, gpa, &blk.attn, normed, n_seqs, seq_len, txt_dim, null, a);
        for (x, a) |*xi, ai| xi.* += ai;

        ops.norm.rmsNorm(normed, x, blk.postnorm, rms_eps);
        try swigluForward(io, gpa, &blk.mlp, normed, rows, txt_dim, txt_mlp_dim, a);
        for (x, a) |*xi, ai| xi.* += ai;
    }

    /// SingleStreamBlock with AdaLN-single modulation.
    fn blockForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, blk: *const Block, x: []f32, seq: usize, tvec: []const f32, freqs: ops.rope.Freqs) !void {
        _ = self;
        // Six modulation chunks: tvec + per-block learned offset.
        const mv = try gpa.alloc(f32, 6 * features);
        defer gpa.free(mv);
        for (mv, tvec, blk.mod) |*m, tv, bm| m.* = tv + bm;
        const pre_scale = mv[0 * features ..][0..features];
        const pre_shift = mv[1 * features ..][0..features];
        const pre_gate = mv[2 * features ..][0..features];
        const post_scale = mv[3 * features ..][0..features];
        const post_shift = mv[4 * features ..][0..features];
        const post_gate = mv[5 * features ..][0..features];

        const normed = try gpa.alloc(f32, seq * features);
        defer gpa.free(normed);
        const a = try gpa.alloc(f32, seq * features);
        defer gpa.free(a);

        // x += pre_gate * attn((1+pre_scale) * prenorm(x) + pre_shift)
        ops.norm.rmsNorm(normed, x, blk.prenorm, rms_eps);
        modulate(normed, pre_scale, pre_shift);
        try attnForward(io, gpa, &blk.attn, normed, 1, seq, features, freqs, a);
        gatedAdd(x, a, pre_gate);

        // x += post_gate * mlp((1+post_scale) * postnorm(x) + post_shift)
        ops.norm.rmsNorm(normed, x, blk.postnorm, rms_eps);
        modulate(normed, post_scale, post_shift);
        try swigluForward(io, gpa, &blk.mlp, normed, seq, features, mlp_dim, a);
        gatedAdd(x, a, post_gate);
    }
};

/// Row-wise AdaLN: x = (1 + scale) * x + shift.
fn modulate(x: []f32, scale: []const f32, shift: []const f32) void {
    const dim = scale.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale, shift) |*v, sc, sh| v.* = (1.0 + sc) * v.* + sh;
    }
}

/// Row-wise gated residual: x += gate * delta.
fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, d, g| v.* += g * d;
    }
}

/// Krea 2 attention: q/k/v/gate projections, per-head QK RMSNorm, optional
/// RoPE, GQA softmax attention, sigmoid-gated output projection.
fn attnForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    attn: *const Attn,
    x: []const f32,
    n_seqs: usize,
    seq_len: usize,
    dim: usize,
    freqs: ?ops.rope.Freqs,
    out: []f32,
) !void {
    const rows = n_seqs * seq_len;
    const q_dim = attn.heads * head_dim;
    const kv_dim = attn.kv_heads * head_dim;
    std.debug.assert(x.len == rows * dim and out.len == rows * dim);

    const q = try gpa.alloc(f32, rows * q_dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, rows * kv_dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, rows * kv_dim);
    defer gpa.free(v);
    const g = try gpa.alloc(f32, rows * dim);
    defer gpa.free(g);
    try ops.matmul.matmul(io, gpa, q, x, rows, attn.wq, null);
    try ops.matmul.matmul(io, gpa, k, x, rows, attn.wk, null);
    try ops.matmul.matmul(io, gpa, v, x, rows, attn.wv, null);
    try ops.matmul.matmul(io, gpa, g, x, rows, attn.gate, null);

    ops.norm.rmsNorm(q, q, attn.qnorm, rms_eps);
    ops.norm.rmsNorm(k, k, attn.knorm, rms_eps);
    if (freqs) |f| {
        std.debug.assert(n_seqs == 1);
        ops.rope.applyInterleaved(q, f, seq_len, attn.heads, head_dim);
        ops.rope.applyInterleaved(k, f, seq_len, attn.kv_heads, head_dim);
    }

    const attn_out = try gpa.alloc(f32, rows * q_dim);
    defer gpa.free(attn_out);
    for (0..n_seqs) |s| {
        try ops.attention.attention(
            io,
            gpa,
            attn_out[s * seq_len * q_dim ..][0 .. seq_len * q_dim],
            q[s * seq_len * q_dim ..][0 .. seq_len * q_dim],
            k[s * seq_len * kv_dim ..][0 .. seq_len * kv_dim],
            v[s * seq_len * kv_dim ..][0 .. seq_len * kv_dim],
            .{
                .seq_q = seq_len,
                .seq_kv = seq_len,
                .n_heads = attn.heads,
                .n_kv_heads = attn.kv_heads,
                .head_dim = head_dim,
            },
        );
    }
    ops.act.sigmoidMul(attn_out, g);
    try ops.matmul.matmul(io, gpa, out, attn_out, rows, attn.wo, null);
}

fn swigluForward(io: std.Io, gpa: std.mem.Allocator, mlp: *const Swiglu, x: []const f32, rows: usize, dim: usize, inner: usize, out: []f32) !void {
    std.debug.assert(x.len == rows * dim and out.len == rows * dim);
    const gate = try gpa.alloc(f32, rows * inner);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, rows * inner);
    defer gpa.free(up);
    try ops.matmul.matmul(io, gpa, gate, x, rows, mlp.gate, null);
    try ops.matmul.matmul(io, gpa, up, x, rows, mlp.up, null);
    ops.act.siluMul(gate, up);
    try ops.matmul.matmul(io, gpa, out, gate, rows, mlp.down, null);
}

fn linear(io: std.Io, gpa: std.mem.Allocator, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
}

// --- weight loading --------------------------------------------------------

fn nameOf(buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
    var fbs = std.Io.Writer.fixed(buf);
    try fbs.print(fmt, args);
    try fbs.writeAll(suffix);
    return fbs.buffered();
}

fn mat(st: *const SafeTensors, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    const name = try nameOf(&buf, fmt, args, "");
    const view = st.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

fn vec(alloc: std.mem.Allocator, st: *const SafeTensors, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try nameOf(&buf, fmt, args, "");
    const view = st.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

/// Zero-centered norm scale -> effective (1 + scale) weight.
fn normScale(alloc: std.mem.Allocator, st: *const SafeTensors, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
    const w = try vec(alloc, st, fmt, args, len);
    for (w) |*v| v.* += 1.0;
    return w;
}

fn loadLinear(alloc: std.mem.Allocator, st: *const SafeTensors, comptime prefix: []const u8, rows: usize, cols: usize, bias: bool) !LinearW {
    return .{
        .w = try mat(st, prefix ++ ".weight", .{}, rows, cols),
        .b = if (bias) try vec(alloc, st, prefix ++ ".bias", .{}, rows) else null,
    };
}

fn loadAttn(alloc: std.mem.Allocator, st: *const SafeTensors, comptime prefix: []const u8, args: anytype, dim: usize, heads: usize, kv_heads: usize) !Attn {
    return .{
        .wq = try mat(st, prefix ++ ".attn.wq.weight", args, heads * head_dim, dim),
        .wk = try mat(st, prefix ++ ".attn.wk.weight", args, kv_heads * head_dim, dim),
        .wv = try mat(st, prefix ++ ".attn.wv.weight", args, kv_heads * head_dim, dim),
        .wo = try mat(st, prefix ++ ".attn.wo.weight", args, dim, heads * head_dim),
        .gate = try mat(st, prefix ++ ".attn.gate.weight", args, dim, dim),
        .qnorm = try normScale(alloc, st, prefix ++ ".attn.qknorm.qnorm.scale", args, head_dim),
        .knorm = try normScale(alloc, st, prefix ++ ".attn.qknorm.knorm.scale", args, head_dim),
        .heads = heads,
        .kv_heads = kv_heads,
    };
}

fn loadSwiglu(st: *const SafeTensors, comptime prefix: []const u8, args: anytype, dim: usize, inner: usize) !Swiglu {
    return .{
        .gate = try mat(st, prefix ++ ".mlp.gate.weight", args, inner, dim),
        .up = try mat(st, prefix ++ ".mlp.up.weight", args, inner, dim),
        .down = try mat(st, prefix ++ ".mlp.down.weight", args, dim, inner),
    };
}

fn loadTxtBlock(alloc: std.mem.Allocator, st: *const SafeTensors, comptime prefix: []const u8, i: usize) !TxtBlock {
    return .{
        .prenorm = try normScale(alloc, st, prefix ++ ".prenorm.scale", .{i}, txt_dim),
        .postnorm = try normScale(alloc, st, prefix ++ ".postnorm.scale", .{i}, txt_dim),
        .attn = try loadAttn(alloc, st, prefix, .{i}, txt_dim, txt_heads, txt_heads),
        .mlp = try loadSwiglu(st, prefix, .{i}, txt_dim, txt_mlp_dim),
    };
}

// --- tests -----------------------------------------------------------------

test "modulate and gatedAdd broadcast over rows" {
    var x = [_]f32{ 1, 2, 3, 4 }; // 2 rows, dim 2
    modulate(&x, &.{ 0.5, -1.0 }, &.{ 10, 20 });
    try std.testing.expectEqualSlices(f32, &.{ 11.5, 20, 14.5, 20 }, &x);
    gatedAdd(&x, &.{ 1, 1, 2, 2 }, &.{ 2, 0.5 });
    try std.testing.expectEqualSlices(f32, &.{ 13.5, 20.5, 18.5, 21 }, &x);
}

fn readF32File(gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    return out;
}

// Full-forward parity against ComfyUI (tools/dump_dit_fixture.py). ~28 GEMM
// blocks of a 12B model in Debug mode is minutes of work, so this only runs
// when the marker file `testdata/slow-tests` exists (touch it to enable),
// in addition to model/fixture presence.
test "dit forward matches comfyui" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/slow-tests", .{}) catch return error.SkipZigTest;
    const dit_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, dit_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/dit_out.bin", .{}) catch return error.SkipZigTest;

    const seq_txt = 14;
    const x_lat = try readF32File(gpa, io, "testdata/dit_x.bin", channels * 16 * 16);
    defer gpa.free(x_lat);
    const expected = try readF32File(gpa, io, "testdata/dit_out.bin", channels * 16 * 16);
    defer gpa.free(expected);
    const ctx = try readF32File(gpa, io, "testdata/text_cond.bin", seq_txt * txt_layers * txt_dim);
    defer gpa.free(ctx);

    var st = try SafeTensors.open(gpa, io, dit_path);
    defer st.deinit();
    var model = try DiT.load(gpa, &st);
    defer model.deinit();

    const out = try gpa.alloc(f32, channels * 16 * 16);
    defer gpa.free(out);
    try model.forward(io, gpa, out, x_lat, 16, 16, 0.875, ctx, seq_txt);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var sum_err: f64 = 0;
    for (expected, out) |e, a| {
        max_err = @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        sum_err += @abs(e - a);
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(out.len));
    std.debug.print("dit parity: max_err={d:.5} mean_err={d:.6} max_val={d:.2}\n", .{ max_err, mean_err, max_val });
    try std.testing.expect(max_err < 0.01 * @max(1.0, max_val));
    try std.testing.expect(mean_err < 1e-3 * @as(f64, @max(1.0, max_val)));
}
