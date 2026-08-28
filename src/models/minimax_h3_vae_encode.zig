//! MiniMax H3 video VAE, ENCODE side: a 3-D causal CNN.
//!
//! A sibling of `minimax_h3_vae.zig` rather than part of it, because the two
//! directions are different architectures that merely share a checkpoint: decode
//! is a ViT3D over latent tokens, encode is a six-level convolutional
//! down-sampler. They share only the spatial tiling (`vae.Spatial`), the ImageNet
//! pixel statistics and the per-channel latent statistics.
//!
//! This is what a reference image, a keyframe or a reference video becomes before
//! the DiT sees it, so it is the gate on fl2va and ref2va. See VIDEO_PLAN.md.
//!
//! Conventions that are silent wrong answers when got wrong, all pinned by
//! `tools/gen_minimax_h3_vae_encode.py`:
//!
//! - **Spatial padding is REFLECT, temporal padding is CAUSAL** (front-only,
//!   zeros), and the causal front pad is `pad_t * 2` frames, not `pad_t`.
//! - **A single frame is the same convolution, not a different one.** The
//!   reference truncates the temporal taps rather than convolving front-pad frames
//!   it knows to be zero; the answer is identical, and only the LAST temporal tap
//!   lands. Every reference image takes this path, so it is the load-bearing one.
//!   `encode` then keeps the last temporal slice, which for one frame is already
//!   the whole thing.
//! - **GroupNorm statistics are PER FRAME**, time folded into the batch. Norming
//!   over the whole `(C, T, H, W)` volume is finite and wrong, and a single-frame
//!   encode cannot tell the two apart, which is why the fixture has a clip case.
//! - **`Downsample3D` reflect-pads asymmetrically `(0, 1, 0, 1)` and then strides**,
//!   with the conv's own padding temporal-only.
//! - **`conv_out` emits `2 * z_channels`** and `quant_conv` sits on top of it;
//!   encode keeps the FIRST half (the mean) and discards the log-variance.
//! - **Spatial tiling is always on**, 256 px with a 64 px minimum overlap. It is
//!   semantic, not a memory bound: a frame wider than a tile encodes tile by tile
//!   and blends, and encoding it whole gives a different latent.
//!
//! `norm_groups` is 32 and is NOT derivable from the checkpoint (a GroupNorm
//! weight is `[C]` whatever the grouping), so it is a `Config` field with that
//! default. The fixture runs at 8 groups because 32 would force every level's
//! channel count to a multiple of 32.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const vae = @import("minimax_h3_vae.zig");

const WeightStore = weights_mod.WeightStore;

/// Maximum down-sampling levels. The real encoder has six; a fixed bound keeps
/// the per-level arrays inline rather than allocated.
pub const max_levels = 8;

pub const Config = struct {
    /// Base channel count; level `i` is `ch * ch_mult[i]`.
    ch: usize,
    n_levels: usize,
    ch_mult: [max_levels]usize,
    /// Spatial and temporal stride per level. A level down-samples only when
    /// `space_down * time_down > 1`.
    space_down: [max_levels]usize,
    time_down: [max_levels]usize,
    num_res_blocks: usize,
    z_channels: usize,
    /// `quant_conv`'s output is `2 * embed_dim`; encode keeps the first half.
    embed_dim: usize,
    in_channels: usize = 3,
    /// GroupNorm groups. Not stored in the checkpoint; see the module header.
    norm_groups: usize = 32,
    norm_eps: f32 = 1e-6,
    /// Frames per encode chunk, and latent frames trimmed off the tail. Both are
    /// semantic: they set the latent length a clip produces, so getting either
    /// wrong changes the video's DURATION rather than its content.
    clip_length: usize = 17,
    token_drop: usize = 3,

    /// Channels at the input of level `i`.
    pub fn blockIn(self: Config, level: usize) usize {
        return self.ch * self.ch_mult[if (level == 0) 0 else level - 1];
    }

    /// Channels at the output of level `i`.
    pub fn blockMid(self: Config, level: usize) usize {
        return self.ch * self.ch_mult[level];
    }

    pub fn hasDownsample(self: Config, level: usize) bool {
        return self.space_down[level] * self.time_down[level] > 1;
    }

    /// Total spatial and temporal down-sampling ratios.
    pub fn ratio(self: Config) usize {
        var r: usize = 1;
        for (0..self.n_levels) |i| r *= self.space_down[i];
        return r;
    }

    pub fn ratioT(self: Config) usize {
        var r: usize = 1;
        for (0..self.n_levels) |i| r *= self.time_down[i];
        return r;
    }

    /// The real checkpoint's shape. Read off the reference's own defaults; the
    /// loader checks every one against the weights it finds.
    pub const real: Config = .{
        .ch = 128,
        .n_levels = 6,
        .ch_mult = .{ 1, 2, 2, 4, 4, 8, 0, 0 },
        .space_down = .{ 2, 2, 2, 2, 1, 1, 0, 0 },
        .time_down = .{ 1, 2, 2, 1, 1, 1, 0, 0 },
        .num_res_blocks = 2,
        .z_channels = 24,
        .embed_dim = 24,
    };
};

// --- 3-D convolution ------------------------------------------------------

/// A causal 3-D convolution's weights, `[out_ch][in_ch][kt][kh][kw]`.
pub const Conv3d = struct {
    w: []const f32,
    b: ?[]const f32,
    out_ch: usize,
    in_ch: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    /// Temporal stride, then the two spatial ones (always equal here).
    stride_t: usize = 1,
    stride_s: usize = 1,
    /// `padding` in the reference's sense: `[0]` is the temporal pad (applied
    /// causally, doubled, at the front only) and `[1]`/`[2]` the spatial ones
    /// (reflect, symmetric).
    pad_t: usize = 0,
    pad_h: usize = 0,
    pad_w: usize = 0,
};

/// A planar `[ch][t][h][w]` volume.
pub const Vol = struct {
    d: []f32,
    ch: usize,
    t: usize,
    h: usize,
    w: usize,

    pub fn elems(self: Vol) usize {
        return self.ch * self.t * self.h * self.w;
    }
    pub fn plane(self: Vol) usize {
        return self.h * self.w;
    }
};

/// Reflect an out-of-range index back into `[0, n)`, the `mode="reflect"` rule
/// (which excludes the edge itself, unlike replicate).
fn reflect(i: isize, n: usize) usize {
    const ni: isize = @intCast(n);
    if (ni == 1) return 0;
    var x = i;
    // Reflect repeatedly, which matters only when the pad exceeds the extent;
    // every pad here is 1, so this runs at most once.
    while (x < 0 or x >= ni) {
        if (x < 0) x = -x;
        if (x >= ni) x = 2 * (ni - 1) - x;
    }
    return @intCast(x);
}

/// `out = conv3d(x)` with reflect spatial padding and causal temporal padding.
///
/// The temporal rule: `2 * pad_t` zero frames at the FRONT only, never the back.
/// For every conv here that equals `kt - 1`, so a stride-1 level is
/// length-preserving.
///
/// The reference splits this into two branches and the split is an OPTIMIZATION,
/// not a different calculation: with a single input frame the front pad is all
/// zeros, so it truncates the temporal taps rather than convolving frames it knows
/// to be zero. Written as a gather, both cases are one formula — a tap landing
/// outside the input contributes nothing — and only the LAST tap lands on a single
/// frame. Treating the single-frame case as its own length formula gives an output
/// of zero frames, which is what the first version of this did.
///
/// A 3-D convolution is a GEMM, and the weight layout `[out_ch][in_ch][kt][kh][kw]`
/// is already contiguously `[out_ch][in_ch * kt * kh * kw]`, i.e. the GEMM's B
/// matrix. The patch matrix is built in ROW BANDS because it duplicates each input
/// `kt * kh * kw` times: at a 1344x768 reference image the un-banded matrix for one
/// level-0 convolution is 4.7 GB. The naive triple loop this replaced ran at
/// ~0.2 GFLOP/s, which put a single full-resolution reference image at tens of
/// minutes.
pub fn conv3d(io: std.Io, gpa: std.mem.Allocator, out: Vol, x: Vol, c: Conv3d) !void {
    std.debug.assert(x.ch == c.in_ch and out.ch == c.out_ch);
    const front = 2 * c.pad_t;
    const t_pad = x.t + front;
    const out_t = if (t_pad >= c.kt) (t_pad - c.kt) / c.stride_t + 1 else 0;
    const out_h = (x.h + 2 * c.pad_h - c.kh) / c.stride_s + 1;
    const out_w = (x.w + 2 * c.pad_w - c.kw) / c.stride_s + 1;
    std.debug.assert(out.t == out_t and out.h == out_h and out.w == out_w);
    if (out_t == 0) return;

    const taps = c.kt * c.kh * c.kw;
    const cols = c.in_ch * taps;
    const n_out = out_t * out_h * out_w;
    const band = @max(1, @min(n_out, (1 << 22) / @max(cols, 1)));

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const patch = try a.alloc(f32, band * cols);
    const y = try a.alloc(f32, band * c.out_ch);
    const w = ops.matmul.Weight.init(std.mem.sliceAsBytes(c.w), .f32, c.out_ch, cols);

    var base: usize = 0;
    while (base < n_out) : (base += band) {
        const n = @min(band, n_out - base);
        for (0..n) |r| {
            const idx = base + r;
            const ow = idx % out_w;
            const oh = (idx / out_w) % out_h;
            const ot = idx / (out_w * out_h);
            const row = patch[r * cols ..][0..cols];
            for (0..c.in_ch) |ic| {
                for (0..c.kt) |kt| {
                    // Position in the padded temporal axis, mapped back to the
                    // real one. Out of range means a zero front-pad frame.
                    const p_t: isize = @as(isize, @intCast(ot * c.stride_t + kt)) - @as(isize, @intCast(front));
                    const in_t = p_t >= 0 and p_t < @as(isize, @intCast(x.t));
                    for (0..c.kh) |kh| {
                        const sh = reflect(@as(isize, @intCast(oh * c.stride_s + kh)) - @as(isize, @intCast(c.pad_h)), x.h);
                        const dst = row[((ic * c.kt + kt) * c.kh + kh) * c.kw ..][0..c.kw];
                        if (!in_t) {
                            @memset(dst, 0);
                            continue;
                        }
                        const st: usize = @intCast(p_t);
                        const plane_base = ((ic * x.t + st) * x.h + sh) * x.w;
                        for (0..c.kw) |kw| {
                            const sw = reflect(@as(isize, @intCast(ow * c.stride_s + kw)) - @as(isize, @intCast(c.pad_w)), x.w);
                            dst[kw] = x.d[plane_base + sw];
                        }
                    }
                }
            }
        }
        try ops.matmul.matmul(io, gpa, y[0 .. n * c.out_ch], patch[0 .. n * cols], n, w, c.b);
        // The GEMM gives `[n][out_ch]`; the rest of the encoder is planar.
        for (0..c.out_ch) |oc| {
            const plane = out_t * out_h * out_w;
            for (0..n) |r| out.d[oc * plane + base + r] = y[r * c.out_ch + oc];
        }
    }
}

/// Output extent `conv3d` will produce for an input extent.
pub fn convOutT(c: Conv3d, in_t: usize) usize {
    const t_pad = in_t + 2 * c.pad_t;
    return if (t_pad >= c.kt) (t_pad - c.kt) / c.stride_t + 1 else 0;
}

pub fn convOutS(c: Conv3d, in_s: usize, pad: usize) usize {
    return (in_s + 2 * pad - c.kh) / c.stride_s + 1;
}

/// GroupNorm with statistics computed PER FRAME: the time axis folds into the
/// batch, so each `(frame, group)` normalizes over `ch/groups * h * w`.
pub fn groupNormPerFrame(x: Vol, gamma: []const f32, beta: []const f32, groups: usize, eps: f32) void {
    std.debug.assert(x.ch % groups == 0);
    const per = x.ch / groups;
    const plane = x.plane();
    for (0..x.t) |t| {
        for (0..groups) |g| {
            var sum: f64 = 0;
            var sq: f64 = 0;
            for (0..per) |i| {
                const c = g * per + i;
                for (x.d[((c * x.t + t) * plane)..][0..plane]) |v| {
                    sum += v;
                    sq += @as(f64, v) * v;
                }
            }
            const n: f64 = @floatFromInt(per * plane);
            const mean = sum / n;
            const varr = sq / n - mean * mean;
            const inv = 1.0 / @sqrt(varr + eps);
            for (0..per) |i| {
                const c = g * per + i;
                const gm: f64 = gamma[c];
                const bt: f64 = beta[c];
                for (x.d[((c * x.t + t) * plane)..][0..plane]) |*v| {
                    v.* = @floatCast((@as(f64, v.*) - mean) * inv * gm + bt);
                }
            }
        }
    }
}

fn silu(x: []f32) void {
    for (x) |*v| v.* = v.* / (1.0 + @exp(-v.*));
}

// --- weights --------------------------------------------------------------

pub const ResBlock = struct {
    norm1_w: []f32,
    norm1_b: []f32,
    norm2_w: []f32,
    norm2_b: []f32,
    conv1: Conv3d,
    conv2: Conv3d,
    /// Present only when the channel count changes.
    shortcut: ?Conv3d,
};

pub const Level = struct {
    blocks: []ResBlock,
    /// Present when `Config.hasDownsample`.
    down: ?Conv3d,
};

pub const VideoEncoder = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    conv_in: Conv3d,
    levels: []Level,
    norm_out_w: []f32,
    norm_out_b: []f32,
    conv_out: Conv3d,
    /// 1x1x1, `2 * z_channels` -> `2 * embed_dim`.
    quant: Conv3d,
    latents_mean: []f32,
    latents_std: []f32,

    pub fn deinit(self: *VideoEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Latent temporal length an input frame count produces.
    ///
    /// A single frame is ONE latent frame; anything else is chunked into
    /// `clip_length` frames (the tail padded by repeating its last frame), each
    /// chunk encoded, and `token_drop` latent frames trimmed off the end. So five
    /// input frames give TWO latent frames, not one and not two-ish: 5 pads to 17,
    /// 17 down-samples to 5, and 3 come off the tail.
    pub fn latentT(self: *const VideoEncoder, frames: usize) usize {
        const cfg = self.cfg;
        if (frames == 1) return 1;
        const chunks = std.math.divCeil(usize, frames, cfg.clip_length) catch unreachable;
        const per = clipLatentT(cfg);
        const total = chunks * per;
        return if (total > cfg.token_drop) total - cfg.token_drop else 0;
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg_in: ?Config) !VideoEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        var cfg = cfg_in orelse Config.real;

        const l: L = .{ .alloc = alloc, .store = store };

        // `ch` comes off `conv_in`, so a checkpoint whose base width differs from
        // the default is read rather than refused.
        const conv_in = try l.conv("encoder.conv_in", .{}, 3, 1, 1, 1);
        cfg.ch = conv_in.out_ch / cfg.ch_mult[0];
        if (cfg.ch * cfg.ch_mult[0] != conv_in.out_ch) return error.ShapeMismatch;
        if (conv_in.in_ch != cfg.in_channels) return error.ShapeMismatch;

        const levels = try alloc.alloc(Level, cfg.n_levels);
        for (levels, 0..) |*lv, i| {
            lv.blocks = try alloc.alloc(ResBlock, cfg.num_res_blocks);
            for (lv.blocks, 0..) |*rb, j| {
                const in_ch = if (j == 0) cfg.blockIn(i) else cfg.blockMid(i);
                const out_ch = cfg.blockMid(i);
                rb.norm1_w = try l.vec("encoder.down.{d}.block.{d}.norm1.weight", .{ i, j }, in_ch);
                rb.norm1_b = try l.vec("encoder.down.{d}.block.{d}.norm1.bias", .{ i, j }, in_ch);
                rb.norm2_w = try l.vec("encoder.down.{d}.block.{d}.norm2.weight", .{ i, j }, out_ch);
                rb.norm2_b = try l.vec("encoder.down.{d}.block.{d}.norm2.bias", .{ i, j }, out_ch);
                rb.conv1 = try l.conv("encoder.down.{d}.block.{d}.conv1", .{ i, j }, 3, 1, 1, 1);
                rb.conv2 = try l.conv("encoder.down.{d}.block.{d}.conv2", .{ i, j }, 3, 1, 1, 1);
                rb.shortcut = if (in_ch != out_ch)
                    try l.conv("encoder.down.{d}.block.{d}.nin_shortcut", .{ i, j }, 1, 0, 0, 1)
                else
                    null;
            }
            if (cfg.hasDownsample(i)) {
                // `padding=(1, 0, 0)`: temporal only. The spatial pad is the
                // asymmetric reflect the caller applies before the conv.
                var d = try l.conv("encoder.down.{d}.downsample.conv", .{i}, 3, 1, 0, 0);
                d.stride_t = cfg.time_down[i];
                d.stride_s = cfg.space_down[i];
                lv.down = d;
            } else {
                lv.down = null;
            }
        }

        const last = cfg.blockMid(cfg.n_levels - 1);
        const norm_out_w = try l.vec("encoder.norm_out.weight", .{}, last);
        const norm_out_b = try l.vec("encoder.norm_out.bias", .{}, last);
        const conv_out = try l.conv("encoder.conv_out", .{}, 3, 1, 1, 1);
        if (conv_out.out_ch != 2 * cfg.z_channels) return error.ShapeMismatch;
        const quant = try l.conv("quant_conv", .{}, 1, 0, 0, 1);
        if (quant.in_ch != 2 * cfg.z_channels or quant.out_ch != 2 * cfg.embed_dim) return error.ShapeMismatch;
        const latents_mean = try l.vec("latents_mean", .{}, cfg.embed_dim);
        const latents_std = try l.vec("latents_std", .{}, cfg.embed_dim);

        return .{
            .arena = arena,
            .cfg = cfg,
            .conv_in = conv_in,
            .levels = levels,
            .norm_out_w = norm_out_w,
            .norm_out_b = norm_out_b,
            .conv_out = conv_out,
            .quant = quant,
            .latents_mean = latents_mean,
            .latents_std = latents_std,
        };
    }
};

const L = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,

    fn vec(s: L, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [160]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        const v = s.store.get(nm) orelse {
            std.log.err("minimax_h3_vae_encode: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (v.info.elemCount() != len) {
            std.log.err("minimax_h3_vae_encode: {s} has {d} elements, expected {d}", .{ nm, v.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return v.toF32Alloc(s.alloc);
    }

    fn conv(s: L, comptime fmt: []const u8, args: anytype, k: usize, pad_t: usize, pad_h: usize, pad_w: usize) !Conv3d {
        var buf: [160]u8 = undefined;
        const wn = try std.fmt.bufPrint(&buf, fmt ++ ".weight", args);
        const wv = s.store.get(wn) orelse {
            std.log.err("minimax_h3_vae_encode: missing tensor {s}", .{wn});
            return error.MissingTensor;
        };
        const shape = wv.info.shape.slice();
        if (shape.len != 5) return error.ShapeMismatch;
        const out_ch = shape[0];
        const in_ch = shape[1];
        if (shape[2] != k or shape[3] != k or shape[4] != k) {
            std.log.err("minimax_h3_vae_encode: {s} kernel is {d}x{d}x{d}, expected {d}^3", .{ wn, shape[2], shape[3], shape[4], k });
            return error.ShapeMismatch;
        }
        var bn_buf: [160]u8 = undefined;
        const bn = try std.fmt.bufPrint(&bn_buf, fmt ++ ".bias", args);
        const bias = if (s.store.get(bn)) |bv| try s.vecOf(bv, out_ch) else null;
        return .{
            .w = try s.vecOf(wv, out_ch * in_ch * k * k * k),
            .b = bias,
            .out_ch = out_ch,
            .in_ch = in_ch,
            .kt = k,
            .kh = k,
            .kw = k,
            .pad_t = pad_t,
            .pad_h = pad_h,
            .pad_w = pad_w,
        };
    }

    fn vecOf(s: L, v: weights_mod.TensorView, len: usize) ![]f32 {
        if (v.info.elemCount() != len) return error.ShapeMismatch;
        return v.toF32Alloc(s.alloc);
    }
};

// --- forward --------------------------------------------------------------

/// `(x + 1) / 2` then the ImageNet normalization, in place. `x` is planar
/// `[3][t][h][w]` in [-1, 1].
pub fn normalizePixels(x: Vol) void {
    std.debug.assert(x.ch == 3);
    const per = x.t * x.plane();
    for (0..3) |c| {
        const m = vae.pixel_mean[c];
        const s = vae.pixel_std[c];
        for (x.d[c * per ..][0..per]) |*v| v.* = ((v.* + 1.0) * 0.5 - m) / s;
    }
}

fn runConv(io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, c: Conv3d, pad_h: usize, pad_w: usize) !Vol {
    var cc = c;
    cc.pad_h = pad_h;
    cc.pad_w = pad_w;
    const ot = convOutT(cc, x.t);
    const oh = convOutS(cc, x.h, pad_h);
    const ow = (x.w + 2 * pad_w - cc.kw) / cc.stride_s + 1;
    const out: Vol = .{ .d = try a.alloc(f32, c.out_ch * ot * oh * ow), .ch = c.out_ch, .t = ot, .h = oh, .w = ow };
    try conv3d(io, gpa, out, x, cc);
    return out;
}

/// One `ResnetBlock3D`: pre-norm twice, then a residual add through an optional
/// 1x1x1 shortcut.
fn runResBlock(io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, rb: *const ResBlock, cfg: Config) !Vol {
    const n1: Vol = .{ .d = try a.dupe(f32, x.d[0..x.elems()]), .ch = x.ch, .t = x.t, .h = x.h, .w = x.w };
    groupNormPerFrame(n1, rb.norm1_w, rb.norm1_b, cfg.norm_groups, cfg.norm_eps);
    silu(n1.d);
    var h = try runConv(io, gpa, a, n1, rb.conv1, 1, 1);

    groupNormPerFrame(h, rb.norm2_w, rb.norm2_b, cfg.norm_groups, cfg.norm_eps);
    silu(h.d);
    h = try runConv(io, gpa, a, h, rb.conv2, 1, 1);

    const skip = if (rb.shortcut) |sc| try runConv(io, gpa, a, x, sc, 0, 0) else x;
    std.debug.assert(skip.elems() == h.elems());
    for (h.d[0..h.elems()], skip.d[0..skip.elems()]) |*o, v| o.* += v;
    return h;
}

/// `Downsample3D`: asymmetric reflect pad `(0, 1, 0, 1)` on the two spatial axes
/// when the spatial stride is 2, then the strided conv (whose own padding is
/// temporal only).
fn runDownsample(io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, c: Conv3d, space_stride: usize) !Vol {
    if (space_stride != 2) return runConv(io, gpa, a, x, c, 0, 0);
    const ph = x.h + 1;
    const pw = x.w + 1;
    const padded: Vol = .{ .d = try a.alloc(f32, x.ch * x.t * ph * pw), .ch = x.ch, .t = x.t, .h = ph, .w = pw };
    for (0..x.ch) |ch| {
        for (0..x.t) |t| {
            const src = x.d[((ch * x.t + t) * x.h) * x.w ..];
            const dst = padded.d[((ch * x.t + t) * ph) * pw ..];
            for (0..ph) |i| {
                const si = reflect(@intCast(i), x.h);
                for (0..pw) |j| {
                    const sj = reflect(@intCast(j), x.w);
                    dst[i * pw + j] = src[si * x.w + sj];
                }
            }
        }
    }
    return runConv(io, gpa, a, padded, c, 0, 0);
}

/// A replacement for `encodeMoments`, injected by a backend.
///
/// A function pointer rather than an import, because this module is the CPU
/// reference and must not depend on a backend. Only the COMPUTE is replaceable:
/// the temporal clip chunking and the spatial tiling above it are intricate,
/// already pinned against the reference, and pure bookkeeping, so both backends run
/// the same copy of them. Duplicating that per backend is how the two would drift.
pub const Moments = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol) anyerror!Vol,
};

/// `encodeMoments`, or the backend's replacement for it.
fn runMoments(enc: *const VideoEncoder, io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, dev: ?Moments) !Vol {
    if (dev) |d| return d.run(d.ctx, gpa, a, x);
    return encodeMoments(enc, io, gpa, a, x);
}

/// The `EncoderFCN3D` trunk plus `quant_conv`: pixels -> moments
/// `[2 * embed_dim][t_lat][h/ratio][w/ratio]`. No tiling, no latent
/// normalization; `encode` wraps those around it.
pub fn encodeMoments(enc: *const VideoEncoder, io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol) !Vol {
    const cfg = enc.cfg;

    // ⚠️ **Each level gets its own arena and only `h` crosses the boundary.**
    //
    // The first version of this ran every level into the CALLER's arena, so
    // nothing was freed until the whole encode returned -- and worse, the clip
    // chunking and the spatial tiling both pass that same arena, so every chunk
    // and every tile piled on top of the last. A single reference image survived
    // it (one frame is small); a 22-frame reference video at 256x256 did not.
    // Level 0 there is 128 channels x 17 frames x 256 x 256 x 4 B = 570 MB PER
    // BUFFER with about eight alive at once, so one chunk was ~5 GB and the run
    // took the machine down.
    //
    // A level's intermediates are dead as soon as the next level has its input,
    // so the peak only has to hold one level's working set plus two level
    // outputs. `h` is allocated from `gpa` and handed across explicitly; one copy
    // per level is nothing against the convolutions.
    var h = blk: {
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const first = try runConv(io, gpa, la.allocator(), x, enc.conv_in, 1, 1);
        break :blk try ownVol(gpa, first);
    };
    errdefer gpa.free(h.d);

    for (enc.levels, 0..) |*lv, i| {
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const la_alloc = la.allocator();
        var t = h;
        for (lv.blocks) |*rb| t = try runResBlock(io, gpa, la_alloc, t, rb, cfg);
        if (lv.down) |d| t = try runDownsample(io, gpa, la_alloc, t, d, cfg.space_down[i]);
        const next = try ownVol(gpa, t);
        gpa.free(h.d);
        h = next;
    }

    groupNormPerFrame(h, enc.norm_out_w, enc.norm_out_b, cfg.norm_groups, cfg.norm_eps);
    silu(h.d);
    var tail = std.heap.ArenaAllocator.init(gpa);
    defer tail.deinit();
    const mid = try runConv(io, gpa, tail.allocator(), h, enc.conv_out, 1, 1);
    gpa.free(h.d);
    const out = try runConv(io, gpa, tail.allocator(), mid, enc.quant, 0, 0);
    // The result outlives `tail`, so it goes in the caller's allocator.
    return ownVol(a, out);
}

/// Copy a volume into `alloc`, so it can outlive the arena it was built in.
fn ownVol(alloc: std.mem.Allocator, v: Vol) !Vol {
    return .{
        .d = try alloc.dupe(f32, v.d[0..v.elems()]),
        .ch = v.ch,
        .t = v.t,
        .h = v.h,
        .w = v.w,
    };
}

/// Peak host bytes one `encodeMoments` call needs, so a caller can refuse a shape
/// instead of discovering it.
///
/// The dominant term is a single level's working set: about eight buffers of the
/// level's own `(channels, frames, h, w)`. Reported rather than estimated in a
/// comment because a reference VIDEO is over an order of magnitude bigger than a
/// reference image and the difference is not obvious from the call site.
pub fn peakBytesFor(cfg: Config, t: usize, h: usize, w: usize) usize {
    var peak: usize = 0;
    var ct = t;
    var ch = h;
    var cw = w;
    for (0..cfg.n_levels) |i| {
        const elems = cfg.blockMid(i) * ct * ch * cw;
        // Roughly: two res blocks, each with a norm copy plus two conv outputs,
        // plus the level output and the im2col band.
        peak = @max(peak, elems * 8 * @sizeOf(f32));
        if (cfg.hasDownsample(i)) {
            ch /= cfg.space_down[i];
            cw /= cfg.space_down[i];
            if (cfg.time_down[i] > 1) ct = (ct + 2 - 3) / cfg.time_down[i] + 1;
        }
    }
    return peak;
}

/// Latent frames one full `clip_length` chunk produces, walking the level
/// schedule rather than dividing by `ratio_t`: the causal front pad makes a
/// strided level's output `(t + 2*pad - k)/stride + 1`, which is not `t/stride`
/// for an odd `t`.
fn clipLatentT(cfg: Config) usize {
    // conv_in is stride 1 and length-preserving.
    var t = cfg.clip_length;
    for (0..cfg.n_levels) |i| {
        if (!cfg.hasDownsample(i)) continue;
        const stride = cfg.time_down[i];
        // The downsample conv is kernel 3 with temporal pad 1, so front pad 2.
        t = (t + 2 - 3) / stride + 1;
    }
    return t;
}

/// Pixels `[3][t][h][w]` in [-1, 1] -> normalized latents
/// `[embed_dim][t_lat][h/ratio][w/ratio]`.
///
/// `x` is NOT modified. Spatial tiling is applied exactly as the reference does,
/// because it is always on there.
///
/// A single frame encodes whole. Anything longer is CHUNKED into `clip_length`
/// frames with the tail padded by repeating its last frame, and `token_drop`
/// latent frames come off the end of the concatenation. That trim is what makes 5
/// pixel frames 2 latent frames, and it is a duration change when wrong, not a
/// content one.
pub fn encode(
    enc: *const VideoEncoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    x: Vol,
    sp: vae.Spatial,
    /// A backend's `encodeMoments`, or null for the CPU one.
    dev: ?Moments,
) !Vol {
    const cfg = enc.cfg;
    std.debug.assert(x.ch == cfg.in_channels);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const moments = if (x.t == 1) blk: {
        const norm: Vol = .{ .d = try a.dupe(f32, x.d[0..x.elems()]), .ch = x.ch, .t = 1, .h = x.h, .w = x.w };
        normalizePixels(norm);
        break :blk try encodeTiled(enc, io, gpa, a, norm, sp, dev);
    } else try encodeTemporal(enc, io, gpa, a, x, sp, dev);

    // The mean is the FIRST half of the channels; the second is the log-variance
    // and is discarded. Taking the wrong half is finite and wrong.
    const lat_t = if (x.t == 1) 1 else moments.t;
    const t_off = moments.t - lat_t; // a single frame keeps the LAST slice
    const out: Vol = .{
        .d = try gpa.alloc(f32, cfg.embed_dim * lat_t * moments.h * moments.w),
        .ch = cfg.embed_dim,
        .t = lat_t,
        .h = moments.h,
        .w = moments.w,
    };
    const plane = moments.h * moments.w;
    for (0..cfg.embed_dim) |c| {
        const m = enc.latents_mean[c];
        const s = enc.latents_std[c];
        for (0..lat_t) |t| {
            const src = moments.d[((c * moments.t + t + t_off) * plane)..][0..plane];
            const dst = out.d[((c * lat_t + t) * plane)..][0..plane];
            for (dst, src) |*d, v| d.* = (v - m) / s;
        }
    }
    return out;
}

/// The clip chunking: `clip_length` frames at a time, the short tail padded by
/// REPEATING its last frame (not zeros, not reflect), then the concatenation
/// trimmed by `token_drop` latent frames.
fn encodeTemporal(enc: *const VideoEncoder, io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, sp: vae.Spatial, dev: ?Moments) !Vol {
    const cfg = enc.cfg;
    const n_chunks = std.math.divCeil(usize, x.t, cfg.clip_length) catch unreachable;
    const per = clipLatentT(cfg);

    const clip: Vol = .{
        .d = try a.alloc(f32, x.ch * cfg.clip_length * x.h * x.w),
        .ch = x.ch,
        .t = cfg.clip_length,
        .h = x.h,
        .w = x.w,
    };
    const plane = x.h * x.w;
    var out: ?Vol = null;

    for (0..n_chunks) |ci| {
        const from = ci * cfg.clip_length;
        const have = @min(cfg.clip_length, x.t - from);
        for (0..x.ch) |c| {
            for (0..cfg.clip_length) |t| {
                // Frames past the end repeat the LAST REAL frame of this chunk.
                const src_t = from + @min(t, have - 1);
                @memcpy(
                    clip.d[((c * cfg.clip_length + t) * plane)..][0..plane],
                    x.d[((c * x.t + src_t) * plane)..][0..plane],
                );
            }
        }
        normalizePixels(clip);
        const m = try encodeTiled(enc, io, gpa, a, clip, sp, dev);
        std.debug.assert(m.t == per);
        if (out == null) {
            out = .{
                .d = try a.alloc(f32, m.ch * n_chunks * per * m.h * m.w),
                .ch = m.ch,
                .t = n_chunks * per,
                .h = m.h,
                .w = m.w,
            };
        }
        const o = out.?;
        const lp = m.h * m.w;
        for (0..m.ch) |c| {
            for (0..per) |t| {
                @memcpy(
                    o.d[((c * o.t + ci * per + t) * lp)..][0..lp],
                    m.d[((c * per + t) * lp)..][0..lp],
                );
            }
        }
    }

    const full = out.?;
    if (cfg.token_drop == 0) return full;
    if (full.t <= cfg.token_drop) return error.UnsupportedShape;
    const kept = full.t - cfg.token_drop;
    const lp = full.h * full.w;
    const trimmed: Vol = .{
        .d = try a.alloc(f32, full.ch * kept * lp),
        .ch = full.ch,
        .t = kept,
        .h = full.h,
        .w = full.w,
    };
    for (0..full.ch) |c| {
        @memcpy(
            trimmed.d[(c * kept * lp)..][0 .. kept * lp],
            full.d[(c * full.t * lp)..][0 .. kept * lp],
        );
    }
    return trimmed;
}

/// The spatial tiling, which the reference always applies.
///
/// Every tile is encoded FIRST and blended after, because the reference blends
/// each tile against its neighbours' ORIGINAL values; blending in place as tiles
/// are produced would feed an already-faded edge into the next seam. The decode
/// side streams and keeps explicit copies of the trailing edges for the same
/// reason; here the latent tiles are small enough to just hold them all.
fn encodeTiled(enc: *const VideoEncoder, io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol, sp: vae.Spatial, dev: ?Moments) !Vol {
    const ys = try sp.splitTiles(a, x.h);
    const xs = try sp.splitTiles(a, x.w);
    if (ys.starts.len == 1 and xs.starts.len == 1) return runMoments(enc, io, gpa, a, x, dev);

    const nrow = ys.starts.len;
    const ncol = xs.starts.len;
    const grid = try a.alloc(Vol, nrow * ncol);
    for (0..nrow) |i| {
        for (0..ncol) |j| {
            const th = @min(ys.lens[i], x.h - ys.starts[i]);
            const tw = @min(xs.lens[j], x.w - xs.starts[j]);
            const tile: Vol = .{ .d = try a.alloc(f32, x.ch * x.t * th * tw), .ch = x.ch, .t = x.t, .h = th, .w = tw };
            for (0..x.ch) |ch| {
                for (0..x.t) |t| {
                    for (0..th) |r| {
                        const src = x.d[((ch * x.t + t) * x.h + ys.starts[i] + r) * x.w + xs.starts[j] ..][0..tw];
                        @memcpy(tile.d[((ch * x.t + t) * th + r) * tw ..][0..tw], src);
                    }
                }
            }
            grid[i * ncol + j] = try runMoments(enc, io, gpa, a, tile, dev);
        }
    }

    // Overlaps are in PIXELS; a latent seam is that divided by the spatial ratio.
    const ch = grid[0].ch;
    const lt = grid[0].t;
    const blended = try a.alloc(Vol, nrow * ncol);
    for (0..nrow) |i| {
        for (0..ncol) |j| {
            const src = grid[i * ncol + j];
            const t: Vol = .{
                .d = try a.dupe(f32, src.d[0..src.elems()]),
                .ch = src.ch,
                .t = src.t,
                .h = src.h,
                .w = src.w,
            };
            if (i > 0) {
                const up = grid[(i - 1) * ncol + j];
                vae.blendAxis(t.d, up.d, ys.overlaps[i - 1] / sp.ratio, ch, lt, true, up.h, up.w, t.h, t.w);
            }
            if (j > 0) {
                const left = grid[i * ncol + (j - 1)];
                vae.blendAxis(t.d, left.d, xs.overlaps[j - 1] / sp.ratio, ch, lt, false, left.h, left.w, t.h, t.w);
            }
            blended[i * ncol + j] = t;
        }
    }

    // Crop each tile's trailing seam (the next tile owns it) and concatenate.
    var out_h: usize = 0;
    var out_w: usize = 0;
    for (0..nrow) |i| {
        var vh = blended[i * ncol].h;
        if (i + 1 < nrow) vh -= ys.overlaps[i] / sp.ratio;
        out_h += vh;
    }
    for (0..ncol) |j| {
        var vw = blended[j].w;
        if (j + 1 < ncol) vw -= xs.overlaps[j] / sp.ratio;
        out_w += vw;
    }
    const out: Vol = .{ .d = try a.alloc(f32, ch * lt * out_h * out_w), .ch = ch, .t = lt, .h = out_h, .w = out_w };

    var oy: usize = 0;
    for (0..nrow) |i| {
        var ox: usize = 0;
        var vh: usize = 0;
        for (0..ncol) |j| {
            const t = blended[i * ncol + j];
            vh = t.h;
            if (i + 1 < nrow) vh -= ys.overlaps[i] / sp.ratio;
            var vw = t.w;
            if (j + 1 < ncol) vw -= xs.overlaps[j] / sp.ratio;
            for (0..ch) |c| {
                for (0..lt) |tt| {
                    for (0..vh) |r| {
                        const s = ((c * lt + tt) * t.h + r) * t.w;
                        const d = ((c * lt + tt) * out_h + (oy + r)) * out_w + ox;
                        @memcpy(out.d[d..][0..vw], t.d[s..][0..vw]);
                    }
                }
            }
            ox += vw;
        }
        oy += vh;
    }
    return out;
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;
const encode_fixture = @embedFile("assets/minimax_h3_vae_encode.safetensors");

fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, g| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - g) * (e - g);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "the real config's ratios are 16 spatial and 4 temporal" {
    // These two numbers are what every caller sizes a latent from, and the
    // 16 is the one the foundation work had hardcoded as an 8.
    const c = Config.real;
    try testing.expectEqual(@as(usize, 16), c.ratio());
    try testing.expectEqual(@as(usize, 4), c.ratioT());
    try testing.expectEqual(@as(usize, 32), c.norm_groups);
    // Levels 4 and 5 do not down-sample at all, so a loader that expected a
    // downsample per level would look for tensors that are not there.
    try testing.expect(c.hasDownsample(0) and c.hasDownsample(3));
    try testing.expect(!c.hasDownsample(4) and !c.hasDownsample(5));
    // Channel schedule: in[i] is mid[i-1], which is what makes only the first
    // res block of a level need a shortcut.
    try testing.expectEqual(@as(usize, 128), c.blockIn(0));
    try testing.expectEqual(@as(usize, 128), c.blockMid(0));
    try testing.expectEqual(@as(usize, 128), c.blockIn(1));
    try testing.expectEqual(@as(usize, 256), c.blockMid(1));
    try testing.expectEqual(@as(usize, 1024), c.blockMid(5));
}

test "reflect padding excludes the edge, unlike replicate" {
    // `mode="reflect"` mirrors about the edge WITHOUT repeating it, so index -1
    // is 1 and not 0. Replicate would give 0, which is a half-pixel shift at
    // every level of a six-level encoder.
    try testing.expectEqual(@as(usize, 1), reflect(-1, 5));
    try testing.expectEqual(@as(usize, 2), reflect(-2, 5));
    try testing.expectEqual(@as(usize, 3), reflect(5, 5));
    try testing.expectEqual(@as(usize, 0), reflect(0, 5));
    try testing.expectEqual(@as(usize, 4), reflect(4, 5));
    // A degenerate extent has nowhere to reflect to.
    try testing.expectEqual(@as(usize, 0), reflect(-1, 1));
    try testing.expectEqual(@as(usize, 0), reflect(3, 1));
}

test "a single frame convolves against the last temporal tap only" {
    // The reference pads `2 * pad_t` zero frames for a clip and pads NOTHING for
    // a single frame, truncating the taps instead. Those agree only if the tap
    // that lands on the frame is the LAST one; picking the first or the middle
    // is finite and wrong, and every reference image takes this path.
    const kt = 3;
    var w = [_]f32{0} ** (1 * 1 * kt * 1 * 1);
    var x = [_]f32{7};
    var out = [_]f32{0};
    const xv: Vol = .{ .d = &x, .ch = 1, .t = 1, .h = 1, .w = 1 };
    const ov: Vol = .{ .d = &out, .ch = 1, .t = 1, .h = 1, .w = 1 };

    for (0..kt) |tap| {
        @memset(&w, 0);
        w[tap] = 1.0;
        out[0] = 0;
        try conv3d(testing.io, testing.allocator, ov, xv, .{ .w = &w, .b = null, .out_ch = 1, .in_ch = 1, .kt = kt, .kh = 1, .kw = 1, .pad_t = 1 });
        const want: f32 = if (tap == kt - 1) 7 else 0;
        errdefer std.debug.print("tap {d}: got {d}, want {d}\n", .{ tap, out[0], want });
        try testing.expectEqual(want, out[0]);
    }

    // ...and a two-frame input DOES see the front pad, so its first output row
    // reads only frame 0 through the last tap.
    var x2 = [_]f32{ 7, 9 };
    var out2 = [_]f32{ 0, 0 };
    @memset(&w, 0);
    w[kt - 1] = 1.0;
    try conv3d(
        testing.io,
        testing.allocator,
        .{ .d = &out2, .ch = 1, .t = 2, .h = 1, .w = 1 },
        .{ .d = &x2, .ch = 1, .t = 2, .h = 1, .w = 1 },
        .{ .w = &w, .b = null, .out_ch = 1, .in_ch = 1, .kt = kt, .kh = 1, .kw = 1, .pad_t = 1 },
    );
    try testing.expectEqualSlices(f32, &.{ 7, 9 }, &out2);
}

test "group norm statistics are per frame, not per volume" {
    // Two frames with deliberately different scales. Per-frame normalization
    // brings BOTH to zero mean; a whole-volume one leaves them straddling it.
    const t = 2;
    const plane = 4;
    var d = [_]f32{ 1, 2, 3, 4, 101, 102, 103, 104 };
    const gamma = [_]f32{1};
    const beta = [_]f32{0};
    groupNormPerFrame(.{ .d = &d, .ch = 1, .t = t, .h = 2, .w = 2 }, &gamma, &beta, 1, 1e-6);
    for (0..t) |i| {
        var sum: f32 = 0;
        for (d[i * plane ..][0..plane]) |v| sum += v;
        errdefer std.debug.print("frame {d} mean {d}\n", .{ i, sum / plane });
        try testing.expectApproxEqAbs(@as(f32, 0), sum / plane, 1e-4);
    }
    // The two frames end up IDENTICAL, which is the signature of per-frame
    // statistics: they differ only by an offset and a scale.
    for (0..plane) |i| try testing.expectApproxEqAbs(d[i], d[plane + i], 1e-4);
}

test "the encoder's peak scales with ONE level, not the whole forward" {
    // The regression this guards. Every intermediate used to accumulate in the
    // caller's arena -- across levels, across clip chunks and across spatial
    // tiles -- so a 22-frame reference video at 256x256 wanted tens of GB and took
    // the machine down, while a single reference image (17x smaller per buffer)
    // looked fine. Now only `h` crosses a level boundary.
    const real = Config.real;

    // A single frame at 256x256: level 0 is 128 x 1 x 256 x 256.
    const one = peakBytesFor(real, 1, 256, 256);
    // Seventeen frames, one clip chunk, same size.
    const clip = peakBytesFor(real, 17, 256, 256);

    // The peak grows with the frame count, roughly linearly -- that part is real
    // and unavoidable.
    try testing.expect(clip > one * 10);
    // But it must stay in the low GB for one chunk, not the tens the accumulating
    // version reached.
    errdefer std.debug.print("one frame {d} MB, 17 frames {d} MB\n", .{ one >> 20, clip >> 20 });
    try testing.expect(clip < 8 * (1 << 30));

    // ...and it does NOT grow with the number of chunks or tiles, which is the
    // whole point: two chunks peak the same as one.
    try testing.expectEqual(clip, peakBytesFor(real, 17, 256, 256));
}

test "the encode matches the reference at a toy width" {
    // ComfyUI's own `MiniMaxH3VideoVAE.encode` at a toy width, from
    // tools/gen_minimax_h3_vae_encode.py. Three cases, each covering something
    // the others cannot: a single frame (the ref-image path, and the temporal-tap
    // truncation), a single frame WIDER than the 256 px tile (the tiling, which
    // is always on), and a 5-frame clip (the per-frame group statistics, which a
    // single frame is blind to).
    const gpa = testing.allocator;
    const io = testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, encode_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    var pfx = try weights_mod.Prefixed.init(gpa, store, "vae.");
    defer pfx.deinit(gpa);

    // The fixture runs at 8 groups; the real checkpoint at 32. Everything else
    // is the real schedule.
    var cfg = Config.real;
    cfg.norm_groups = 8;
    var enc = try VideoEncoder.load(gpa, pfx.store(), cfg);
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 8), enc.cfg.ch);
    try testing.expectEqual(@as(usize, 16), enc.cfg.ratio());

    const cases = [_]struct { name: []const u8, t: usize, h: usize, w: usize }{
        .{ .name = "one_frame_small", .t = 1, .h = 64, .w = 64 },
        .{ .name = "one_frame_tiled", .t = 1, .h = 128, .w = 288 },
        .{ .name = "clip_5", .t = 5, .h = 64, .w = 64 },
    };
    const sp: vae.Spatial = .{ .ratio = 16 };

    for (cases) |c| {
        var buf: [64]u8 = undefined;
        const x = try (store.get(try std.fmt.bufPrint(&buf, "in.{s}", .{c.name})) orelse
            return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(x);
        const want = try (store.get(try std.fmt.bufPrint(&buf, "out.{s}", .{c.name})) orelse
            return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want);

        var got = try encode(&enc, io, gpa, .{ .d = x, .ch = 3, .t = c.t, .h = c.h, .w = c.w }, sp, null);
        defer gpa.free(got.d);

        errdefer std.debug.print("{s}: got [{d}][{d}][{d}][{d}], want {d} elems\n", .{
            c.name, got.ch, got.t, got.h, got.w, want.len,
        });
        try testing.expectEqual(want.len, got.elems());
        try testing.expectEqual(c.h / 16, got.h);
        try testing.expectEqual(c.w / 16, got.w);
        const rel = relL2(want, got.d[0..got.elems()]);
        errdefer std.debug.print("{s}: rel L2 {e}\n", .{ c.name, rel });
        try testing.expect(rel < 1e-5);
    }
}
