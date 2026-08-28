//! MiniMax H3 video VAE, decode side: a ViT3D over the latent grid.
//!
//! Not a CNN decoder. The 24-channel latent is flattened to one token per
//! `[t][h][w]` cell, embedded, run through 36 pre-norm transformer blocks, and
//! projected to a `4 x 16 x 16` pixel patch per token. The encoder half (a 3D
//! causal CNN) is a separate thing and is only needed for image/video
//! conditioning, not for text-to-video.
//!
//! Conventions that are silent wrong answers when got wrong, and note that
//! several DIFFER from the DiT in the same checkpoint family:
//!
//! - **`to_qkv` is fused PER HEAD**: `view(B, N, heads, 3 * head_dim)` then
//!   chunk, so a token's row reads `[h0 q | h0 k | h0 v | h1 q | ...]`. The DiT
//!   fuses the other way (`[all q | all k | all v]`). Reading one as the other is
//!   finite and wrong.
//! - **Q/K norms are WEIGHTLESS** (`elementwise_affine=False`), so there is no
//!   `norm_q.weight` in the checkpoint at all.
//! - **Position ids are per-axis normalized to [-1, 1]**, `2 * (i + 0.5) / n - 1`,
//!   NOT the DiT's area-normalized grid. Same model family, different convention.
//! - Rope is partial split-half again, but over `int(head_dim * 0.75)` and with
//!   an angle scale of `2 * pi` and base 100, where the DiT uses neither.
//! - **Four register tokens plus one zero token** are appended, given position
//!   ids of zero, and dropped after `proj_out`. `mask_token` is in the checkpoint
//!   and unused at inference.
//! - The residual is `x += sublayer(norm(x)) * scale`, a per-channel learned
//!   LayerScale, not a plain add.
//! - `norm_out` is a LayerNorm with bias; every other norm here is RMS.
//! - Latent normalization is per-channel `(mean, std)` from the checkpoint's own
//!   metadata, and the output undoes an IMAGENET normalization and clamps.
//!
//! Reference is ComfyUI `comfy/ldm/minimax/vae.py`. See VIDEO_PLAN.md.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// ImageNet statistics the decoder's output is denormalized by.
pub const pixel_mean = [3]f32{ 0.485, 0.456, 0.406 };
pub const pixel_std = [3]f32{ 0.229, 0.224, 0.225 };

pub const Config = struct {
    /// `heads * head_dim`.
    dim: usize,
    heads: usize,
    head_dim: usize,
    n_layers: usize,
    /// Post-swiglu width, i.e. half of `w1`'s output. The reference builds it as
    /// `dim * 4`, but it is read off the weights here.
    ff: usize,
    /// Spatial patch each token expands to, and the temporal one.
    patch: usize,
    patch_t: usize,
    in_channels: usize,
    out_channels: usize,
    n_register: usize,
    rope_theta: f32 = 100.0,
    /// Fraction of the head that rotates. 0.75 of 64 is 48 dims, i.e. 24 pairs.
    rope_dim_ratio: f32 = 0.75,
    eps: f32 = 1e-5,

    /// Rotation pairs per token. `int(head_dim * ratio)` is the reference's
    /// rope width, and `n_dim = 3` axes share it: the frequency count per axis is
    /// `rope_width / (2 * 3)`, so the pair count is three times that.
    pub fn ropeFreqCount(self: Config) usize {
        const width: usize = @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.rope_dim_ratio);
        return width / 6;
    }

    pub fn ropePairs(self: Config) usize {
        return self.ropeFreqCount() * 3;
    }

    /// Head dims the rotation covers; the rest pass through.
    pub fn ropeRotDim(self: Config) usize {
        return self.ropePairs() * 2;
    }

    /// Values `proj_out` emits per token.
    pub fn patchDim(self: Config) usize {
        return self.out_channels * self.patch_t * self.patch * self.patch;
    }

    pub fn detect(store: WeightStore) !Config {
        const emb = try dims2(store, "decoder.x_embedder.weight");
        const dim = emb[0];
        const in_channels = emb[1];

        const qkv = try dims2(store, "decoder.transformer_blocks.0.attn.to_qkv.weight");
        if (qkv[1] != dim or qkv[0] != dim * 3) return error.ShapeMismatch;

        const w1 = try dims2(store, "decoder.transformer_blocks.0.ff.w1.weight");
        if (w1[1] != dim or w1[0] % 2 != 0) return error.ShapeMismatch;

        const proj = try dims2(store, "decoder.proj_out.weight");
        if (proj[1] != dim) return error.ShapeMismatch;

        const reg = store.get("decoder.register_tokens") orelse return error.MissingTensor;
        const reg_shape = reg.info.shape.slice();
        if (reg_shape.len != 3 or reg_shape[2] != dim) return error.ShapeMismatch;
        const n_register = reg_shape[1];

        // head_dim is not stated by any weight: the reference's default is 64 and
        // the real checkpoint is 32 heads of it. Derive it from the head count,
        // which is likewise not stated... so take the reference's constant and
        // check it divides. A wrong split is a finite wrong answer, so this is
        // the one place a preset is unavoidable.
        const head_dim: usize = 64;
        if (dim % head_dim != 0) return error.UnsupportedCheckpoint;

        var n_layers: usize = 0;
        while (true) : (n_layers += 1) {
            var buf: [96]u8 = undefined;
            const nm = std.fmt.bufPrint(&buf, "decoder.transformer_blocks.{d}.norm1.weight", .{n_layers}) catch break;
            if (store.get(nm) == null) break;
        }

        var cfg: Config = .{
            .dim = dim,
            .heads = dim / head_dim,
            .head_dim = head_dim,
            .n_layers = n_layers,
            .ff = w1[0] / 2,
            .patch = 16,
            .patch_t = 4,
            .in_channels = in_channels,
            .out_channels = 3,
            .n_register = n_register,
        };
        // `proj_out` states the patch volume; with out_channels and patch_t fixed
        // by the architecture, that pins the spatial patch.
        const per_frame = proj[0] / (cfg.out_channels * cfg.patch_t);
        cfg.patch = std.math.sqrt(per_frame);
        if (cfg.patchDim() != proj[0]) return error.ShapeMismatch;
        return cfg;
    }
};

fn dims2(store: WeightStore, name: []const u8) ![2]usize {
    const view = store.get(name) orelse return error.MissingTensor;
    const s = view.info.shape.slice();
    if (s.len != 2) return error.ShapeMismatch;
    return .{ s[0], s[1] };
}

pub const Block = struct {
    norm1: []f32,
    norm2: []f32,
    /// Per-channel LayerScale on each residual: `x += sublayer(...) * scale`.
    scale1: []f32,
    scale2: []f32,
    qkv: Weight,
    qkv_bias: []f32,
    out: Weight,
    out_bias: []f32,
    w1: Weight,
    w1_bias: []f32,
    w2: Weight,
    w2_bias: []f32,
};

pub const VideoDecoder = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// `post_quant_conv`, a 1x1x1 Conv3d, i.e. a per-position linear.
    post_quant: Weight,
    post_quant_bias: []f32,
    x_embedder: Weight,
    x_embedder_bias: []f32,
    /// `[n_register][dim]`.
    register_tokens: []f32,
    blocks: []Block,
    norm_out_w: []f32,
    norm_out_b: []f32,
    proj_out: Weight,
    proj_out_bias: []f32,
    /// Per-channel latent statistics, from the checkpoint's own metadata.
    latents_mean: []f32,
    latents_std: []f32,

    pub fn deinit(self: *VideoDecoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !VideoDecoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const cfg = try Config.detect(store);
        const l: L = .{ .alloc = alloc, .store = store };

        const blocks = try alloc.alloc(Block, cfg.n_layers);
        for (blocks, 0..) |*b, i| b.* = .{
            .norm1 = try l.vec("decoder.transformer_blocks.{d}.norm1.weight", .{i}, cfg.dim),
            .norm2 = try l.vec("decoder.transformer_blocks.{d}.norm2.weight", .{i}, cfg.dim),
            .scale1 = try l.vec("decoder.transformer_blocks.{d}.scale1", .{i}, cfg.dim),
            .scale2 = try l.vec("decoder.transformer_blocks.{d}.scale2", .{i}, cfg.dim),
            .qkv = try l.mat("decoder.transformer_blocks.{d}.attn.to_qkv.weight", .{i}, cfg.dim * 3, cfg.dim),
            .qkv_bias = try l.vec("decoder.transformer_blocks.{d}.attn.to_qkv.bias", .{i}, cfg.dim * 3),
            .out = try l.mat("decoder.transformer_blocks.{d}.attn.to_out.weight", .{i}, cfg.dim, cfg.dim),
            .out_bias = try l.vec("decoder.transformer_blocks.{d}.attn.to_out.bias", .{i}, cfg.dim),
            .w1 = try l.mat("decoder.transformer_blocks.{d}.ff.w1.weight", .{i}, cfg.ff * 2, cfg.dim),
            .w1_bias = try l.vec("decoder.transformer_blocks.{d}.ff.w1.bias", .{i}, cfg.ff * 2),
            .w2 = try l.mat("decoder.transformer_blocks.{d}.ff.w2.weight", .{i}, cfg.dim, cfg.ff),
            .w2_bias = try l.vec("decoder.transformer_blocks.{d}.ff.w2.bias", .{i}, cfg.dim),
        };

        // A 1x1x1 Conv3d is stored [co][ci][1][1][1]; as a GEMM it is [co][ci].
        const pq = store.get("post_quant_conv.weight") orelse return error.MissingTensor;
        if (pq.info.elemCount() != cfg.in_channels * cfg.in_channels) return error.ShapeMismatch;
        const post_quant = Weight.init(pq.bytes, pq.info.dtype, cfg.in_channels, cfg.in_channels);

        const post_quant_bias = try l.vec("post_quant_conv.bias", .{}, cfg.in_channels);
        const x_embedder = try l.mat("decoder.x_embedder.weight", .{}, cfg.dim, cfg.in_channels);
        const x_embedder_bias = try l.vec("decoder.x_embedder.bias", .{}, cfg.dim);
        const register_tokens = try l.vec("decoder.register_tokens", .{}, cfg.n_register * cfg.dim);
        const norm_out_w = try l.vec("decoder.norm_out.weight", .{}, cfg.dim);
        const norm_out_b = try l.vec("decoder.norm_out.bias", .{}, cfg.dim);
        const proj_out = try l.mat("decoder.proj_out.weight", .{}, cfg.patchDim(), cfg.dim);
        const proj_out_bias = try l.vec("decoder.proj_out.bias", .{}, cfg.patchDim());
        const latents_mean = try l.vec("latents_mean", .{}, cfg.in_channels);
        const latents_std = try l.vec("latents_std", .{}, cfg.in_channels);

        return .{
            .arena = arena,
            .cfg = cfg,
            .post_quant = post_quant,
            .post_quant_bias = post_quant_bias,
            .x_embedder = x_embedder,
            .x_embedder_bias = x_embedder_bias,
            .register_tokens = register_tokens,
            .blocks = blocks,
            .norm_out_w = norm_out_w,
            .norm_out_b = norm_out_b,
            .proj_out = proj_out,
            .proj_out_bias = proj_out_bias,
            .latents_mean = latents_mean,
            .latents_std = latents_std,
        };
    }
};

const L = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,

    fn mat(l: L, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [128]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("minimax_h3_vae: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const s = view.info.shape.slice();
        if (s.len != 2 or s[0] != rows or s[1] != cols) {
            std.log.err("minimax_h3_vae: {s} has shape {any}, expected {d}x{d}", .{ nm, s, rows, cols });
            return error.ShapeMismatch;
        }
        if (!ops.matmul.supportsDType(view.info.dtype)) return error.UnsupportedDType;
        var w = Weight.init(view.bytes, view.info.dtype, rows, cols);
        w.tag = try l.alloc.dupe(u8, nm);
        return w;
    }

    fn vec(l: L, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [128]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("minimax_h3_vae: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("minimax_h3_vae: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }
};

/// Per-axis token coordinates, `2 * (i + 0.5) / n - 1` in [-1, 1].
///
/// NOT the DiT's area-normalized grid: this one normalizes each axis by its own
/// extent, so the same latent produces different coordinates in the two models.
pub fn tokenAxis(alloc: std.mem.Allocator, n: usize) ![]f32 {
    const out = try alloc.alloc(f32, n);
    for (out, 0..) |*v, i| {
        const c = (@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(n));
        v.* = 2.0 * c - 1.0;
    }
    return out;
}

/// Rope tables for `t * h * w` grid tokens followed by `n_suffix` tokens pinned
/// at position zero (the register tokens and the trailing zero token).
pub fn ropeFreqs(
    gpa: std.mem.Allocator,
    cfg: Config,
    t: usize,
    h: usize,
    w: usize,
    n_suffix: usize,
) !ops.rope.Freqs {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ta = try tokenAxis(a, t);
    const ha = try tokenAxis(a, h);
    const wa = try tokenAxis(a, w);

    const n_freq = cfg.ropeFreqCount();
    const pairs = cfg.ropePairs();
    const n = t * h * w + n_suffix;
    const cos = try gpa.alloc(f32, n * pairs);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, n * pairs);
    errdefer gpa.free(sin);

    // inv_freq[i] = 1 / theta^(i * 2 * n_dim / rope_width), the reference's
    // `arange(0, 1, 2 * n_dim / dim)` expressed by index.
    const step = 1.0 / @as(f32, @floatFromInt(n_freq));
    const angle_scale = 2.0 * std.math.pi;

    var row: usize = 0;
    for (0..t) |ti| {
        for (0..h) |hi| {
            for (0..w) |wi| {
                const axes = [3]f32{ ta[ti], ha[hi], wa[wi] };
                for (axes, 0..) |p, ax| {
                    for (0..n_freq) |i| {
                        const inv = std.math.pow(f32, cfg.rope_theta, -@as(f32, @floatFromInt(i)) * step);
                        const ang = angle_scale * p * inv;
                        const at = row * pairs + ax * n_freq + i;
                        cos[at] = @cos(ang);
                        sin[at] = @sin(ang);
                    }
                }
                row += 1;
            }
        }
    }
    // Suffix tokens sit at position 0 on every axis: angle 0, so cos 1 / sin 0.
    for (row..n) |r| {
        for (0..pairs) |i| {
            cos[r * pairs + i] = 1.0;
            sin[r * pairs + i] = 0.0;
        }
    }
    return .{ .cos = cos, .sin = sin, .half = pairs };
}

/// De-interleave the per-head fused `to_qkv` output into three planar
/// `[seq][heads][head_dim]` buffers.
///
/// The layout is `[h0 q | h0 k | h0 v | h1 q | ...]` per token, which is NOT how
/// the DiT fuses qkv. Splitting it the DiT's way reads head 0's k as part of q.
fn splitQkv(q: []f32, k: []f32, v: []f32, qkv: []const f32, seq: usize, heads: usize, hd: usize) void {
    const per_head = 3 * hd;
    for (0..seq) |i| {
        const row = qkv[i * heads * per_head ..];
        for (0..heads) |hh| {
            const src = row[hh * per_head ..];
            const dst = (i * heads + hh) * hd;
            @memcpy(q[dst..][0..hd], src[0..hd]);
            @memcpy(k[dst..][0..hd], src[hd..][0..hd]);
            @memcpy(v[dst..][0..hd], src[2 * hd ..][0..hd]);
        }
    }
}

/// The token grid a decode of `[in_channels][t][h][w]` produces.
pub fn outputShape(cfg: Config, t: usize, h: usize, w: usize) struct { frames: usize, height: usize, width: usize } {
    return .{ .frames = t * cfg.patch_t, .height = h * cfg.patch, .width = w * cfg.patch };
}

/// Decode a normalized planar latent `[in_channels][t][h][w]` to planar RGB
/// `[3][t*patch_t][h*patch][w*patch]` in [0, 1].
///
/// This is the reference's `_decode_pixels` plus its latent denormalization and
/// output finalization, i.e. ONE whole-volume decode. The reference also has
/// spatial tiling and temporal chunking around it, which is what determines the
/// frame count of a long clip; neither is implemented yet, so `t` here is the
/// whole clip and the frame count is exactly `t * patch_t`.
pub fn decode(
    dec: *const VideoDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) !void {
    try decodeVolume(dec, io, gpa, out, z, t, h, w);
    const s = outputShape(dec.cfg, t, h, w);
    finalizePixels(out, dec.cfg.out_channels, s.frames * s.height * s.width);
}

/// The raw whole-volume decode: no latent denormalization is skipped, but the
/// ImageNet finalize is NOT applied. `decodeTemporal` needs the raw field.
pub fn decodeVolume(
    dec: *const VideoDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) !void {
    const cfg = dec.cfg;
    const c_in = cfg.in_channels;
    const grid = t * h * w;
    const n_suffix = cfg.n_register + 1;
    const seq = grid + n_suffix;
    const dim = cfg.dim;
    const hd = cfg.head_dim;
    const heads = cfg.heads;
    const shape = outputShape(cfg, t, h, w);
    std.debug.assert(z.len == c_in * grid);
    std.debug.assert(out.len == cfg.out_channels * shape.frames * shape.height * shape.width);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Denormalize and transpose planar [c][t][h][w] to token-major [n][c] in one
    // pass; `post_quant_conv` is a 1x1x1 conv, so it is a plain GEMM over that.
    const rows = try a.alloc(f32, grid * c_in);
    for (0..grid) |i| {
        for (0..c_in) |c| {
            rows[i * c_in + c] = z[c * grid + i] * dec.latents_std[c] + dec.latents_mean[c];
        }
    }
    const pq = try a.alloc(f32, grid * c_in);
    try ops.matmul.matmul(io, gpa, pq, rows, grid, dec.post_quant, dec.post_quant_bias);

    // Embed, then append the register tokens and the trailing zero token.
    const x = try a.alloc(f32, seq * dim);
    try ops.matmul.matmul(io, gpa, x[0 .. grid * dim], pq, grid, dec.x_embedder, dec.x_embedder_bias);
    @memcpy(x[grid * dim ..][0 .. cfg.n_register * dim], dec.register_tokens);
    @memset(x[(grid + cfg.n_register) * dim ..][0..dim], 0);

    var freqs = try ropeFreqs(gpa, cfg, t, h, w, n_suffix);
    defer freqs.deinit(gpa);

    const hn = try a.alloc(f32, seq * dim);
    const qkv = try a.alloc(f32, seq * dim * 3);
    const qb = try a.alloc(f32, seq * dim);
    const kb = try a.alloc(f32, seq * dim);
    const vb = try a.alloc(f32, seq * dim);
    const att = try a.alloc(f32, seq * dim);
    const proj = try a.alloc(f32, seq * dim);
    const ff = try a.alloc(f32, seq * cfg.ff * 2);
    const gate = try a.alloc(f32, seq * cfg.ff);

    for (dec.blocks) |*b| {
        // attention half
        ops.norm.rmsNorm(hn, x, b.norm1, cfg.eps);
        try ops.matmul.matmul(io, gpa, qkv, hn, seq, b.qkv, b.qkv_bias);
        splitQkv(qb, kb, vb, qkv, seq, heads, hd);
        // WEIGHTLESS RMS over each head: `elementwise_affine=False`, so there is
        // no norm_q/norm_k weight in the checkpoint to apply.
        ops.norm.rmsNormUnit(qb, qb, hd, cfg.eps);
        ops.norm.rmsNormUnit(kb, kb, hd, cfg.eps);
        ops.rope.applyRotateHalfPartialAt(qb, freqs, 0, seq, heads, hd, cfg.ropeRotDim());
        ops.rope.applyRotateHalfPartialAt(kb, freqs, 0, seq, heads, hd, cfg.ropeRotDim());
        try ops.attention.attention(io, gpa, att, qb, kb, vb, .{
            .seq_q = seq,
            .seq_kv = seq,
            .n_heads = heads,
            .n_kv_heads = heads,
            .head_dim = hd,
            .causal = false,
        });
        try ops.matmul.matmul(io, gpa, proj, att, seq, b.out, b.out_bias);
        // LayerScale residual, per channel.
        for (0..seq) |i| {
            const dst = x[i * dim ..][0..dim];
            const src = proj[i * dim ..][0..dim];
            for (dst, src, b.scale1) |*d, s, sc| d.* += s * sc;
        }

        // feed-forward half
        ops.norm.rmsNorm(hn, x, b.norm2, cfg.eps);
        try ops.matmul.matmul(io, gpa, ff, hn, seq, b.w1, b.w1_bias);
        // `[gate; value]`, gate first, same order as the DiT's swiglu.
        for (0..seq) |i| {
            const row = ff[i * cfg.ff * 2 ..][0 .. cfg.ff * 2];
            const g = gate[i * cfg.ff ..][0..cfg.ff];
            @memcpy(g, row[0..cfg.ff]);
            ops.act.siluMul(g, row[cfg.ff..][0..cfg.ff]);
        }
        try ops.matmul.matmul(io, gpa, proj, gate, seq, b.w2, b.w2_bias);
        for (0..seq) |i| {
            const dst = x[i * dim ..][0..dim];
            const src = proj[i * dim ..][0..dim];
            for (dst, src, b.scale2) |*d, s, sc| d.* += s * sc;
        }
    }

    // Head: LayerNorm (with bias, unlike every other norm here), then project.
    // Only the GRID tokens are projected; the suffix is dropped.
    ops.norm.layerNorm(hn, x, dec.norm_out_w, dec.norm_out_b, cfg.eps);
    const patch_dim = cfg.patchDim();
    const patches = try a.alloc(f32, grid * patch_dim);
    try ops.matmul.matmul(io, gpa, patches, hn[0 .. grid * dim], grid, dec.proj_out, dec.proj_out_bias);

    unpatchify(cfg, out, patches, t, h, w);
}

/// Undo the ImageNet normalization and clamp, in place, on planar
/// `[out_channels][frames][h][w]` pixels.
///
/// Separate from the decode because the chunked path applies it per WRITTEN
/// part, after blending: blending finalized pixels and blending raw ones are not
/// the same operation, and the reference does the latter.
pub fn finalizePixels(out: []f32, channels: usize, plane: usize) void {
    std.debug.assert(out.len == channels * plane);
    for (0..channels) |c| {
        for (out[c * plane ..][0..plane]) |*p| {
            p.* = std.math.clamp(p.* * pixel_std[c] + pixel_mean[c], 0.0, 1.0);
        }
    }
}

/// `[grid][out_channels * patch_t * patch * patch]` -> planar
/// `[out_channels][t * patch_t][h * patch][w * patch]`, the reference's
/// `permute(0, 4, 1, 5, 2, 6, 3, 7)`.
pub fn unpatchify(cfg: Config, out: []f32, patches: []const f32, t: usize, h: usize, w: usize) void {
    const pt = cfg.patch_t;
    const p = cfg.patch;
    const oc = cfg.out_channels;
    const frames = t * pt;
    const height = h * p;
    const width = w * p;
    const patch_dim = cfg.patchDim();
    std.debug.assert(patches.len == t * h * w * patch_dim);
    std.debug.assert(out.len == oc * frames * height * width);
    for (0..t) |ti| {
        for (0..h) |hi| {
            for (0..w) |wi| {
                const row = ((ti * h + hi) * w + wi) * patch_dim;
                for (0..oc) |c| {
                    for (0..pt) |dt| {
                        for (0..p) |dh| {
                            for (0..p) |dw| {
                                const src = row + ((c * pt + dt) * p + dh) * p + dw;
                                const dst = ((c * frames + (ti * pt + dt)) * height + (hi * p + dh)) * width + (wi * p + dw);
                                out[dst] = patches[src];
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- tests ----------------------------------------------------------------

const vae_fixture = @embedFile("assets/minimax_h3_vae.safetensors");

fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "unpatchify round-trips through a patch index" {
    const gpa = std.testing.allocator;
    const cfg: Config = .{
        .dim = 8,
        .heads = 1,
        .head_dim = 8,
        .n_layers = 1,
        .ff = 8,
        .patch = 2,
        .patch_t = 3,
        .in_channels = 24,
        .out_channels = 3,
        .n_register = 4,
    };
    const t = 2;
    const h = 3;
    const w = 2;
    const n = t * h * w * cfg.patchDim();
    const patches = try gpa.alloc(f32, n);
    defer gpa.free(patches);
    for (patches, 0..) |*v, i| v.* = @floatFromInt(i);

    const s = outputShape(cfg, t, h, w);
    const out = try gpa.alloc(f32, cfg.out_channels * s.frames * s.height * s.width);
    defer gpa.free(out);
    unpatchify(cfg, out, patches, t, h, w);

    // Every value lands exactly once: a permutation, not a scatter with holes.
    try std.testing.expectEqual(n, out.len);
    const seen = try gpa.alloc(bool, n);
    defer gpa.free(seen);
    @memset(seen, false);
    for (out) |v| {
        const i: usize = @intFromFloat(v);
        try std.testing.expect(!seen[i]);
        seen[i] = true;
    }

    // And the mapping itself: token (0,0,0)'s channel c, sub-position (dt,dh,dw)
    // lands at pixel (dt, dh, dw) of channel c.
    for (0..cfg.out_channels) |c| {
        for (0..cfg.patch_t) |dt| {
            for (0..cfg.patch) |dh| {
                for (0..cfg.patch) |dw| {
                    const src = ((c * cfg.patch_t + dt) * cfg.patch + dh) * cfg.patch + dw;
                    const dst = ((c * s.frames + dt) * s.height + dh) * s.width + dw;
                    try std.testing.expectEqual(patches[src], out[dst]);
                }
            }
        }
    }
}

test "token axis is per-axis normalized to [-1, 1], not area-normalized" {
    const gpa = std.testing.allocator;
    // 2*(i+0.5)/n - 1: symmetric about zero, never reaching the endpoints
    const a = try tokenAxis(gpa, 4);
    defer gpa.free(a);
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), a[3], 1e-6);
    // A single-element axis sits at the centre, not at an endpoint.
    const one = try tokenAxis(gpa, 1);
    defer gpa.free(one);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), one[0], 1e-6);

    // Each axis is normalized by its OWN extent, so a 2x4 grid gives the two
    // axes different steps. The DiT's grid is area-normalized and would give
    // them the SAME step; that difference is the whole point of this test.
    const short = try tokenAxis(gpa, 2);
    defer gpa.free(short);
    try std.testing.expect(@abs(short[1] - short[0]) > @abs(a[1] - a[0]));
}

test "the video decode matches the reference at a toy width" {
    // ComfyUI's own `ViT3DDecoder` at a toy width, from
    // tools/gen_minimax_h3_vae.py. head_dim is the REAL 64, because it is the one
    // dimension no weight states and `Config.detect` takes it as a constant.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, vae_fixture);
    defer st.deinit();
    var dec = try VideoDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    const cfg = dec.cfg;
    try std.testing.expectEqual(@as(usize, 128), cfg.dim);
    try std.testing.expectEqual(@as(usize, 2), cfg.heads);
    try std.testing.expectEqual(@as(usize, 64), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 2), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 24), cfg.in_channels);
    try std.testing.expectEqual(@as(usize, 4), cfg.n_register);
    // 24 pairs rotate 48 of the 64-wide head; 16 pass through.
    try std.testing.expectEqual(@as(usize, 24), cfg.ropePairs());
    try std.testing.expectEqual(@as(usize, 48), cfg.ropeRotDim());
    try std.testing.expect(cfg.ropeRotDim() < cfg.head_dim);

    const t: usize = 2;
    const h: usize = 3;
    const w: usize = 4;
    const zv = try st.require("in.z");
    const z = try zv.toF32Alloc(gpa);
    defer gpa.free(z);
    const wantv = try st.require("out.rgb");
    const want = try wantv.toF32Alloc(gpa);
    defer gpa.free(want);

    const s = outputShape(cfg, t, h, w);
    try std.testing.expectEqual(t * cfg.patch_t, s.frames);
    const out = try gpa.alloc(f32, cfg.out_channels * s.frames * s.height * s.width);
    defer gpa.free(out);
    try decode(&dec, io, gpa, out, z, t, h, w);

    try std.testing.expectEqual(want.len, out.len);
    const err = relL2(want, out);
    errdefer std.debug.print("video decode rel L2 {e}\n", .{err});
    try std.testing.expect(err < 1e-5);

    // The output is a clamped [0, 1] image, and a decode that saturated
    // everywhere would match a broken one just as well. The fixture's generator
    // checks the same thing on its side.
    var interior: usize = 0;
    for (out) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
        if (v > 1e-4 and v < 1.0 - 1e-4) interior += 1;
    }
    try std.testing.expect(interior * 2 > out.len);
}

const test_gate = @import("../test_gate.zig");
const real_video_vae = "/home/qt/genai/comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors";

test "the real video VAE loads and its config matches the checkpoint" {
    // The toy-width test pins the MATH; this pins the constants `Config.detect`
    // cannot read off a weight (head_dim, the patch geometry) against the file
    // they have to describe.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, real_video_vae);

    var st = try tp_core.safetensors.SafeTensors.open(gpa, io, real_video_vae);
    defer st.deinit();
    var dec = try VideoDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    const cfg = dec.cfg;
    try std.testing.expectEqual(@as(usize, 2048), cfg.dim);
    try std.testing.expectEqual(@as(usize, 32), cfg.heads);
    try std.testing.expectEqual(@as(usize, 64), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 36), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 8192), cfg.ff);
    try std.testing.expectEqual(@as(usize, 24), cfg.in_channels);
    try std.testing.expectEqual(@as(usize, 4), cfg.n_register);
    // 16x spatial and 4x temporal, i.e. one token becomes a 4x16x16 pixel block
    try std.testing.expectEqual(@as(usize, 16), cfg.patch);
    try std.testing.expectEqual(@as(usize, 4), cfg.patch_t);
    try std.testing.expectEqual(@as(usize, 3072), cfg.patchDim());

    // The latent statistics come from the checkpoint, not a table here.
    try std.testing.expectEqual(@as(usize, 24), dec.latents_mean.len);
    for (dec.latents_std) |s| try std.testing.expect(s > 0);
}

// --- spatial tiling -------------------------------------------------------

/// How a frame is decoded: in overlapping PIXEL tiles, blended.
///
/// This is NOT a memory bound, it is part of the answer. The reference's VAE has
/// `tiling=True` by default and `_adaptive_decode` therefore always goes through
/// `tiled_decode`, so the ViT3D never sees more than `tile / patch` spatial
/// tokens per axis. Decoding a larger frame in one pass runs the trunk at a
/// spatial extent it is never given, and the result is per-patch incoherence: a
/// visible 16 px grid wherever the image has detail, and smooth where it does
/// not. A 256 px frame is exactly ONE tile, which is why that size looked right
/// and 512 did not.
pub const Spatial = struct {
    /// Tile side in PIXELS.
    tile: usize = 256,
    /// Minimum overlap between tiles, in pixels.
    overlap_min: usize = 64,
    /// The VAE's spatial ratio, i.e. `Config.patch`. Overlaps are rounded to it
    /// so every tile boundary lands on a latent cell.
    ratio: usize = 16,

    pub const Split = struct {
        starts: []usize,
        lens: []usize,
        /// `starts.len - 1` entries.
        overlaps: []usize,
    };

    /// Tile placement along one axis of a `input_len`-pixel frame.
    ///
    /// Mirrors the reference's `split_tiles`: grow the tile count until the
    /// tiles plus minimum overlaps cover the axis, then spend the slack by
    /// widening overlaps round-robin in whole latent cells.
    pub fn splitTiles(self: Spatial, alloc: std.mem.Allocator, input_len: usize) !Split {
        if (self.tile >= input_len) {
            const starts = try alloc.alloc(usize, 1);
            starts[0] = 0;
            const lens = try alloc.alloc(usize, 1);
            lens[0] = input_len;
            return .{ .starts = starts, .lens = lens, .overlaps = &.{} };
        }
        var n = std.math.divCeil(usize, input_len, self.tile) catch unreachable;
        var overlaps: []usize = &.{};
        while (true) {
            overlaps = try alloc.alloc(usize, n - 1);
            @memset(overlaps, self.overlap_min);
            const span = self.tile * n;
            const used = self.overlap_min * (n - 1) + input_len;
            if (span < used) {
                alloc.free(overlaps);
                n += 1;
                continue;
            }
            // Slack, spent in whole latent cells so tile edges stay on the grid.
            const units = (span - used) / self.ratio;
            for (0..units) |i| overlaps[i % (n - 1)] += self.ratio;
            break;
        }
        const starts = try alloc.alloc(usize, n);
        const lens = try alloc.alloc(usize, n);
        @memset(lens, self.tile);
        starts[0] = 0;
        for (0..n - 1) |i| starts[i + 1] = starts[i] + self.tile - overlaps[i];
        return .{ .starts = starts, .lens = lens, .overlaps = overlaps };
    }
};

/// Linear cross-fade of `n` rows or columns of a planar `[c][frames][h][w]` tile,
/// in place on `b`, from `a`'s trailing edge.
///
/// Geometry-agnostic, which is why the ENCODE side reuses it over latent tiles
/// (`minimax_h3_vae_encode.zig`) where this file uses it over pixel tiles. `a`
/// must be the neighbour's ORIGINAL trailing edge, not a blended one: the
/// reference blends every tile against its unmodified neighbours.
pub fn blendAxis(
    b: []f32,
    a: []const f32,
    n: usize,
    channels: usize,
    frames: usize,
    comptime vertical: bool,
    a_h: usize,
    a_w: usize,
    b_h: usize,
    b_w: usize,
) void {
    const extent = @min(@min(if (vertical) a_h else a_w, if (vertical) b_h else b_w), n);
    if (extent == 0) return;
    for (0..extent) |i| {
        const wb = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(extent));
        const wa = 1.0 - wb;
        for (0..channels) |c| {
            for (0..frames) |f| {
                const ap = (c * frames + f) * a_h * a_w;
                const bp = (c * frames + f) * b_h * b_w;
                if (vertical) {
                    const arow = a[ap + (a_h - extent + i) * a_w ..][0..a_w];
                    const brow = b[bp + i * b_w ..][0..b_w];
                    for (brow, arow[0..@min(a_w, b_w)]) |*d, av| d.* = av * wa + d.* * wb;
                } else {
                    for (0..b_h) |r| {
                        const av = a[ap + r * a_w + (a_w - extent + i)];
                        const d = &b[bp + r * b_w + i];
                        d.* = av * wa + d.* * wb;
                    }
                }
            }
        }
    }
}

/// Decode one temporal window in overlapping spatial tiles.
///
/// `out` is the whole window at `[out_channels][frames][h*patch][w*patch]`;
/// `vol` decodes ONE tile.
pub fn decodeTiled(
    dec: *const VideoDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    sp: Spatial,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
    vol: ?Volume,
) !void {
    const cfg = dec.cfg;
    const height = h * cfg.patch;
    const width = w * cfg.patch;
    const frames = t * cfg.patch_t;
    const chans = cfg.out_channels;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ys = try sp.splitTiles(a, height);
    const xs = try sp.splitTiles(a, width);
    if (ys.starts.len == 1 and xs.starts.len == 1) {
        return callVolume(dec, vol, io, gpa, out, z, t, h, w);
    }

    // The previous tile ROW's bottom overlap, per column, and the previous tile's
    // right overlap in this row.
    const row_tails = try a.alloc([]f32, xs.starts.len);
    @memset(row_tails, &.{});
    var left_tail: []f32 = &.{};
    var left_w: usize = 0;

    var out_y: usize = 0;
    for (ys.starts, ys.lens, 0..) |ipos, ilen, i| {
        const zi = ipos / cfg.patch;
        const zl = ilen / cfg.patch;
        const new_tails = try a.alloc([]f32, xs.starts.len);
        @memset(new_tails, &.{});
        var out_x: usize = 0;
        var last_h: usize = 0;
        for (xs.starts, xs.lens, 0..) |jpos, jlen, j| {
            const zj = jpos / cfg.patch;
            const zw = jlen / cfg.patch;

            // The tile's latent slice, planar [c][t][zl][zw].
            const sub = try a.alloc(f32, cfg.in_channels * t * zl * zw);
            for (0..cfg.in_channels) |c| {
                for (0..t) |tt| {
                    for (0..zl) |r| {
                        const src = ((c * t + tt) * h + (zi + r)) * w + zj;
                        const dst = ((c * t + tt) * zl + r) * zw;
                        @memcpy(sub[dst..][0..zw], z[src..][0..zw]);
                    }
                }
            }
            var th = zl * cfg.patch;
            var tw = zw * cfg.patch;
            const tile = try a.alloc(f32, chans * frames * th * tw);
            try callVolume(dec, vol, io, gpa, tile, sub, t, zl, zw);

            // Keep this tile's trailing edges BEFORE blending: they are the
            // reference for the next tile, and blending overwrites them.
            if (i + 1 < ys.starts.len) {
                const ov = ys.overlaps[i];
                const keep = try a.alloc(f32, chans * frames * ov * tw);
                for (0..chans * frames) |p| {
                    @memcpy(keep[p * ov * tw ..][0 .. ov * tw], tile[p * th * tw + (th - ov) * tw ..][0 .. ov * tw]);
                }
                new_tails[j] = keep;
            }
            var next_left: []f32 = &.{};
            var next_left_w: usize = 0;
            if (j + 1 < xs.starts.len) {
                const ov = xs.overlaps[j];
                const keep = try a.alloc(f32, chans * frames * th * ov);
                for (0..chans * frames) |p| {
                    for (0..th) |r| {
                        @memcpy(keep[(p * th + r) * ov ..][0..ov], tile[p * th * tw + r * tw + (tw - ov) ..][0..ov]);
                    }
                }
                next_left = keep;
                next_left_w = ov;
            }

            if (i > 0) {
                const ov = ys.overlaps[i - 1];
                blendAxis(tile, row_tails[j], ov, chans, frames, true, ov, tw, th, tw);
            }
            if (j > 0) {
                const ov = xs.overlaps[j - 1];
                blendAxis(tile, left_tail, ov, chans, frames, false, th, left_w, th, tw);
            }
            left_tail = next_left;
            left_w = next_left_w;

            // Crop the trailing overlap: the next tile owns those pixels.
            var vh = th;
            var vw = tw;
            if (i + 1 < ys.starts.len) vh -= ys.overlaps[i];
            if (j + 1 < xs.starts.len) vw -= xs.overlaps[j];

            for (0..chans) |c| {
                for (0..frames) |f| {
                    for (0..vh) |r| {
                        const src = ((c * frames + f) * th + r) * tw;
                        const dst = ((c * frames + f) * height + (out_y + r)) * width + out_x;
                        @memcpy(out[dst..][0..vw], tile[src..][0..vw]);
                    }
                }
            }
            out_x += vw;
            last_h = vh;
            th = th;
            tw = tw;
        }
        @memcpy(row_tails, new_tails);
        out_y += last_h;
    }
    std.debug.assert(out_y == height);
}

fn callVolume(
    dec: *const VideoDecoder,
    vol: ?Volume,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) !void {
    if (vol) |x| return x.call(x.ctx, io, gpa, out, z, t, h, w);
    return decodeVolume(dec, io, gpa, out, z, t, h, w);
}

// --- temporal chunking ----------------------------------------------------

/// How a long clip is decoded: in overlapping token windows, blended.
///
/// This is NOT an optimization that can be skipped. A whole-volume decode gives
/// `t * patch_t` frames, and the reference gives fewer: 2 latent frames become
/// 5 pixel frames, not 8, and 37 become 124, not 148. Those are exactly the
/// counts `minimax_h3.framesForLatentT` predicts from the DiT's own time axis,
/// which is the cross-check that the two halves agree.
///
/// The defaults are the real checkpoint's (`vae_clip_length` 17,
/// `vae_token_drop` 3, `vae_ratio_t` 4).
pub const Temporal = struct {
    clip_length: usize = 17,
    token_drop: usize = 3,
    /// The VAE's temporal ratio, i.e. `Config.patch_t`.
    ratio_t: usize = 4,

    /// Latent tokens per decode window.
    pub fn chunkSize(self: Temporal) usize {
        return std.math.divCeil(usize, self.clip_length, self.ratio_t) catch unreachable;
    }

    /// Frames dropped from the FRONT of every decoded window: the clip length is
    /// not a whole number of tokens, and the remainder is leading padding.
    pub fn framePrePadding(self: Temporal) usize {
        return (self.ratio_t - self.clip_length % self.ratio_t) % self.ratio_t;
    }

    /// Tokens each window shares with the next.
    pub fn tokenOverlap(self: Temporal) usize {
        const cs = self.chunkSize();
        return (cs - self.token_drop % cs) % cs;
    }

    /// Frames blended between consecutive windows.
    pub fn frameOverlap(self: Temporal) usize {
        const a = self.tokenOverlap() * self.ratio_t;
        const b = self.framePrePadding();
        return if (a > b) a - b else 0;
    }

    pub const Plan = struct { pad_tokens: usize, num_chunks: usize };

    /// Tokens to append (by repeating the last) and windows to decode.
    pub fn plan(self: Temporal, z_len: usize) Plan {
        const cs = self.chunkSize();
        var pseudo = z_len + self.token_drop;
        var pad = (cs - pseudo % cs) % cs;
        pseudo += pad;
        var num_chunks = pseudo / cs - @intFromBool(self.token_drop > 0);
        if (num_chunks < 1) {
            // Too few tokens for one window (e.g. latent_t == 2): pad a whole
            // extra window rather than decoding nothing.
            pad += cs;
            num_chunks += 1;
        }
        return .{ .pad_tokens = pad, .num_chunks = num_chunks };
    }

    /// Frames the padding tokens contribute, which are subtracted at the end.
    fn padFrames(self: Temporal, z_len: usize, pad_tokens: usize) usize {
        if (pad_tokens == 0) return 0;
        const intra_tail = self.clip_length % self.ratio_t;
        if (intra_tail == 0) return pad_tokens * self.ratio_t;
        const cs = self.chunkSize();
        const before = z_len - pad_tokens;
        var total: usize = 0;
        for (0..pad_tokens) |k| {
            total += if ((before + k) % cs == 0) intra_tail else self.ratio_t;
        }
        return total;
    }

    /// Pixel frames a latent of `z_len` tokens decodes to.
    pub fn outputFrames(self: Temporal, z_len: usize) usize {
        if (z_len == 1) return 1;
        const p = self.plan(z_len);
        const padded = z_len + p.pad_tokens;
        const chunk_dec = self.chunkSize() * self.ratio_t;
        const split_count: usize = @as(usize, @intFromBool(self.token_drop > 0)) + 1;
        const pre = self.framePrePadding();

        var total: usize = 0;
        var final_overlap: usize = 0;
        for (0..p.num_chunks) |i| {
            const t_start = i * self.chunkSize();
            const t_end = t_start + self.chunkSize() + self.tokenOverlap();
            const lo = @min(t_start, padded);
            const hi = @min(t_end, padded);
            const clip_frame_len = (hi - lo) * self.ratio_t;
            for (0..split_count) |j| {
                const f_start = j * chunk_dec;
                const f_end = @min(f_start + chunk_dec, clip_frame_len);
                const frames = if (f_end > f_start + pre) f_end - f_start - pre else 0;
                if (j == 0) total += frames else final_overlap = frames;
            }
        }
        return total + final_overlap - self.padFrames(padded, p.pad_tokens);
    }
};

/// Linear cross-fade of `n` frames: `a`'s tail into `b`'s head, in place on `b`.
///
/// Planar `[c][frames][plane]`, so a frame is a stride, not a contiguous run.
fn blendFrames(b: []f32, a: []const f32, n: usize, a_frames: usize, b_frames: usize, channels: usize, plane: usize) void {
    const extent = @min(@min(a_frames, b_frames), n);
    if (extent == 0) return;
    for (0..extent) |i| {
        const wb = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(extent));
        const wa = 1.0 - wb;
        for (0..channels) |c| {
            const asrc = a[(c * a_frames + (a_frames - extent + i)) * plane ..][0..plane];
            const bdst = b[(c * b_frames + i) * plane ..][0..plane];
            for (bdst, asrc) |*d, av| d.* = av * wa + d.* * wb;
        }
    }
}

/// Decode a whole clip: overlapping token windows, blended, finalized.
///
/// `out` must be `[out_channels][Temporal.outputFrames(t)][h*patch][w*patch]`.
/// How one window is decoded. `decodeTemporal` drives the chunking and blending
/// and calls this per window, so a backend supplies only the volume decode.
///
/// A closure rather than a backend enum: the device path lives in
/// `minimax_h3_vae_cuda`, which depends on this module, so naming it here would
/// be a cycle.
pub const VolumeFn = *const fn (
    ctx: *anyopaque,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) anyerror!void;

pub const Volume = struct { ctx: *anyopaque, call: VolumeFn };

pub fn decodeTemporal(
    dec: *const VideoDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    tp: Temporal,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) !void {
    return decodeTemporalWith(dec, io, gpa, tp, out, z, t, h, w, null);
}

/// `decodeTemporal` with the per-window decode supplied by the caller.
pub fn decodeTemporalWith(
    dec: *const VideoDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    tp: Temporal,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
    vol: ?Volume,
) !void {
    // Every window goes through the SPATIAL tiling, which is what the
    // reference's `_adaptive_decode` does (`tiling = True` by default). A frame
    // at or below the tile size is one tile and the two are identical.
    const tiles: Spatial = .{ .ratio = dec.cfg.patch };
    const volume = struct {
        fn call(d: *const VideoDecoder, v: ?Volume, s2: Spatial, io2: std.Io, g: std.mem.Allocator, o: []f32, zz: []const f32, tt: usize, hh: usize, ww: usize) !void {
            return decodeTiled(d, io2, g, s2, o, zz, tt, hh, ww, v);
        }
    }.call;
    const cfg = dec.cfg;
    const c_in = cfg.in_channels;
    const height = h * cfg.patch;
    const width = w * cfg.patch;
    const plane = height * width;
    const out_frames = tp.outputFrames(t);
    std.debug.assert(z.len == c_in * t * h * w);
    std.debug.assert(out.len == cfg.out_channels * out_frames * plane);

    // Single-frame latents take the reference's own early path.
    if (t == 1) {
        try volume(dec, vol, tiles, io, gpa, out, z, 1, h, w);
        finalizePixels(out, cfg.out_channels, out_frames * plane);
        return;
    }

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Pad by REPEATING the last latent frame, which is what the reference does.
    const p = tp.plan(t);
    const padded_t = t + p.pad_tokens;
    const zp = try a.alloc(f32, c_in * padded_t * h * w);
    const sp = h * w;
    for (0..c_in) |c| {
        @memcpy(zp[c * padded_t * sp ..][0 .. t * sp], z[c * t * sp ..][0 .. t * sp]);
        for (t..padded_t) |k| {
            @memcpy(zp[(c * padded_t + k) * sp ..][0..sp], z[(c * t + (t - 1)) * sp ..][0..sp]);
        }
    }

    const chunk_dec = tp.chunkSize() * tp.ratio_t;
    const split_count: usize = @as(usize, @intFromBool(tp.token_drop > 0)) + 1;
    const pre = tp.framePrePadding();

    var write_pos: usize = 0;
    // The tail of the previous window, held back to blend into the next one.
    var overlap: ?[]f32 = null;
    var overlap_frames: usize = 0;

    const writePart = struct {
        fn call(dst: []f32, pos: *usize, part: []const f32, n: usize, chans: usize, pl: usize, total: usize) void {
            if (n == 0) return;
            const room = if (total > pos.*) total - pos.* else 0;
            const take = @min(n, room);
            for (0..chans) |c| {
                @memcpy(dst[(c * total + pos.*) * pl ..][0 .. take * pl], part[c * n * pl ..][0 .. take * pl]);
            }
            pos.* += take;
        }
    }.call;

    for (0..p.num_chunks) |i| {
        const t_start = @min(i * tp.chunkSize(), padded_t);
        const t_end = @min(t_start + tp.chunkSize() + tp.tokenOverlap(), padded_t);
        const win = t_end - t_start;
        if (win == 0) continue;

        // The window's latent, sliced on the temporal axis of a planar buffer.
        const clip_z = try a.alloc(f32, c_in * win * sp);
        for (0..c_in) |c| {
            @memcpy(clip_z[c * win * sp ..][0 .. win * sp], zp[(c * padded_t + t_start) * sp ..][0 .. win * sp]);
        }
        const dec_frames = win * cfg.patch_t;
        const clip_dec = try a.alloc(f32, cfg.out_channels * dec_frames * plane);
        try volume(dec, vol, tiles, io, gpa, clip_dec, clip_z, win, h, w);

        for (0..split_count) |j| {
            const f_start = j * chunk_dec;
            const f_end = @min(f_start + chunk_dec, dec_frames);
            if (f_end <= f_start + pre) continue;
            // ...then drop the window's leading padding frames.
            const lo = f_start + pre;
            const n = f_end - lo;
            const part = try a.alloc(f32, cfg.out_channels * n * plane);
            for (0..cfg.out_channels) |c| {
                @memcpy(part[c * n * plane ..][0 .. n * plane], clip_dec[(c * dec_frames + lo) * plane ..][0 .. n * plane]);
            }

            if (j == 0) {
                if (overlap) |ov| {
                    blendFrames(part, ov, tp.frameOverlap(), overlap_frames, n, cfg.out_channels, plane);
                    overlap = null;
                }
                finalizePixels(part, cfg.out_channels, n * plane);
                writePart(out, &write_pos, part, n, cfg.out_channels, plane, out_frames);
            } else {
                overlap = part;
                overlap_frames = n;
            }
        }

        if (i == p.num_chunks - 1) {
            if (overlap) |ov| {
                finalizePixels(ov, cfg.out_channels, overlap_frames * plane);
                writePart(out, &write_pos, ov, overlap_frames, cfg.out_channels, plane, out_frames);
                overlap = null;
            }
        }
    }
    std.debug.assert(write_pos == out_frames);
}

test "the VAE's frame count agrees with the DiT's time axis" {
    // The two halves of H3 compute the clip length independently: the DiT from
    // its per-token frame spans (1, 4, 4, 4, 4), the VAE from its chunking
    // arithmetic. They must agree for every valid length, or the render has more
    // or fewer frames than the model thinks it generated.
    //
    // This is what says the chunking is REQUIRED: a whole-volume decode gives
    // `t * patch_t`, which is 8 where the reference gives 5.
    const minimax_h3 = @import("minimax_h3.zig");
    const tp: Temporal = .{};

    try std.testing.expectEqual(@as(usize, 5), tp.chunkSize());
    try std.testing.expectEqual(@as(usize, 3), tp.framePrePadding());
    try std.testing.expectEqual(@as(usize, 2), tp.tokenOverlap());
    try std.testing.expectEqual(@as(usize, 5), tp.frameOverlap());

    var frames: usize = 5;
    while (frames <= 400) : (frames += 17) {
        const latent_t = minimax_h3.videoLatentT(frames);
        errdefer std.debug.print("frames={d} latent_t={d} vae={d} dit={d}\n", .{
            frames, latent_t, tp.outputFrames(latent_t), minimax_h3.framesForLatentT(latent_t),
        });
        try std.testing.expectEqual(frames, tp.outputFrames(latent_t));
        try std.testing.expectEqual(minimax_h3.framesForLatentT(latent_t), tp.outputFrames(latent_t));
        // ...and the naive whole-volume count is genuinely different, so this
        // test is not vacuously true.
        try std.testing.expect(tp.outputFrames(latent_t) != latent_t * tp.ratio_t);
    }

    // The specific counts the reference produces, spelled out.
    try std.testing.expectEqual(@as(usize, 5), tp.outputFrames(2));
    try std.testing.expectEqual(@as(usize, 22), tp.outputFrames(7));
    try std.testing.expectEqual(@as(usize, 39), tp.outputFrames(12));
    try std.testing.expectEqual(@as(usize, 124), tp.outputFrames(37));
}

test "the chunked decode matches the reference, content and count" {
    // 7 latent tokens is TWO windows, so the blend between them actually runs.
    // The count alone is checked above; this pins the pixels, which differ from
    // a whole-volume decode by more than a trim.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, vae_fixture);
    defer st.deinit();
    var dec = try VideoDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    const t: usize = 7;
    const h: usize = 3;
    const w: usize = 4;
    const tp: Temporal = .{ .ratio_t = dec.cfg.patch_t };
    const frames = tp.outputFrames(t);
    try std.testing.expectEqual(@as(usize, 22), frames);
    try std.testing.expect(frames != t * dec.cfg.patch_t);

    const zv = try st.require("in.z_long");
    const z = try zv.toF32Alloc(gpa);
    defer gpa.free(z);
    const wantv = try st.require("out.chunked");
    const want = try wantv.toF32Alloc(gpa);
    defer gpa.free(want);

    const s = outputShape(dec.cfg, t, h, w);
    const out = try gpa.alloc(f32, dec.cfg.out_channels * frames * s.height * s.width);
    defer gpa.free(out);
    try decodeTemporal(&dec, io, gpa, tp, out, z, t, h, w);

    try std.testing.expectEqual(want.len, out.len);
    const err = relL2(want, out);
    errdefer std.debug.print("chunked decode rel L2 {e}\n", .{err});
    try std.testing.expect(err < 1e-5);
}

test "spatial tile placement matches the reference" {
    const gpa = std.testing.allocator;
    const sp: Spatial = .{};

    // At or below the tile size: ONE tile, which is why a 256 px frame decodes
    // identically with and without tiling.
    {
        const s = try sp.splitTiles(gpa, 256);
        defer gpa.free(s.starts);
        defer gpa.free(s.lens);
        try std.testing.expectEqual(@as(usize, 1), s.starts.len);
        try std.testing.expectEqual(@as(usize, 256), s.lens[0]);
        try std.testing.expectEqual(@as(usize, 0), s.overlaps.len);
    }

    // 320 px is 2 tiles at 0 and 64 with a 192 px overlap; 512 px is 3 tiles at
    // 0, 128, 256 with 128 px overlaps. Both read off the reference's own
    // `split_tiles`.
    for ([_]struct { len: usize, starts: []const usize, ov: []const usize }{
        .{ .len = 320, .starts = &.{ 0, 64 }, .ov = &.{192} },
        .{ .len = 512, .starts = &.{ 0, 128, 256 }, .ov = &.{ 128, 128 } },
    }) |c| {
        const s = try sp.splitTiles(gpa, c.len);
        defer gpa.free(s.starts);
        defer gpa.free(s.lens);
        defer if (s.overlaps.len > 0) gpa.free(s.overlaps);
        errdefer std.debug.print("len {d}: starts {any} overlaps {any}\n", .{ c.len, s.starts, s.overlaps });
        try std.testing.expectEqualSlices(usize, c.starts, s.starts);
        try std.testing.expectEqualSlices(usize, c.ov, s.overlaps);
        // The tiles must COVER the axis exactly, and every boundary must land on
        // a latent cell or a tile would need a fractional latent slice.
        try std.testing.expectEqual(c.len, s.starts[s.starts.len - 1] + s.lens[s.lens.len - 1]);
        for (s.starts) |st| try std.testing.expectEqual(@as(usize, 0), st % sp.ratio);
        for (s.overlaps) |ov| try std.testing.expectEqual(@as(usize, 0), ov % sp.ratio);
    }
}

test "the spatially tiled decode matches the reference" {
    // 20x20 latent is 320x320 pixels: 2x2 tiles with a 192 px overlap and a
    // blend. Every other fixture case is below the tile size and therefore a
    // single tile, which is exactly why they passed before the tiling existed.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, vae_fixture);
    defer st.deinit();
    var dec = try VideoDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    const t: usize = 2;
    const h: usize = 20;
    const w: usize = 20;
    const tp: Temporal = .{ .ratio_t = dec.cfg.patch_t };
    const frames = tp.outputFrames(t);

    const z = try (try st.require("in.z_tiled")).toF32Alloc(gpa);
    defer gpa.free(z);
    const want = try (try st.require("out.tiled")).toF32Alloc(gpa);
    defer gpa.free(want);

    const s = outputShape(dec.cfg, t, h, w);
    const out = try gpa.alloc(f32, dec.cfg.out_channels * frames * s.height * s.width);
    defer gpa.free(out);
    try decodeTemporal(&dec, io, gpa, tp, out, z, t, h, w);

    try std.testing.expectEqual(want.len, out.len);
    const err = relL2(want, out);
    errdefer std.debug.print("tiled decode rel L2 {e}\n", .{err});
    try std.testing.expect(err < 1e-5);
}
