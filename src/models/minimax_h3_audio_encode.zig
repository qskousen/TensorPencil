//! MiniMax H3 audio VAE, encode side: a DAC-lineage waveform down-sampler
//! followed by a causal-attention posterior head.
//!
//! This is what turns a reference soundtrack into the `[32][2][t]` latent the DiT
//! packs as `ref_audio` rows. Both stereo channels go through the same MONO
//! network independently, as on the decode side.
//!
//! It shares `Conv1d` and the im2col GEMM with `minimax_h3_audio.zig`, and
//! nothing else: every activation and every block shape differs from the vocoder's.
//! Conventions that are silent wrong answers when got wrong:
//!
//! - **`Snake1d`, not `SnakeBeta`.** The encoder stores alpha LINEAR and uses the
//!   same parameter as beta; the decoder stores alpha and beta separately and in
//!   LOG scale. Three differences in one activation name.
//! - **The residual dilations are 1, 3, 9** here and 1, 3, 5 in the decoder.
//! - A stage's three residual units run at HALF the stage width, i.e. the previous
//!   stage's; only the final strided conv widens. Its kernel is `2 * stride`, which
//!   is what makes the stride recoverable from the checkpoint.
//! - **The posterior head's attention is CAUSAL**, and its output is the MEAN OVER
//!   HEADS pooled along the FEATURE axis down to the latent width. Pooling time
//!   instead gives a plausible shape and no error.
//! - The fused qkv carries no bias of its own: the bias is
//!   `cat(q_bias, zeros, v_bias)`, so the k third is zero.
//! - `AttnProjection` is `proj(norm3(x)) + attn(norm1(x))` then `+ mlp(norm2(x))`.
//!   norm3 feeds the projection and norm1 the attention.
//! - The posterior MEAN is the answer: `mean_proj`, never `logs_proj`, and no
//!   sampling.
//! - The waveform is right-padded with zeros to a multiple of the hop (800).
//!
//! Reference is ComfyUI `comfy/ldm/minimax/audio_vae.py`, lineage
//! descript-audio-codec (MIT). Fixture: tools/gen_minimax_h3_audio_encode.py.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const audio = @import("minimax_h3_audio.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const Conv1d = audio.Conv1d;

/// LayerNorm epsilon, `torch.nn.LayerNorm`'s default.
const ln_eps: f32 = 1e-5;

/// Both residual convolutions in a `ResidualUnit`, and the dilation cycle. None
/// of these is stored: the kernel widths are, the dilations are not.
const res_k: usize = 7;
const res_dilations = [3]usize{ 1, 3, 9 };

/// `x + sin(alpha * x)^2 / (alpha + 1e-9)`, per channel. Alpha is stored LINEAR
/// and serves as beta as well, which is the encoder's `Snake1d`.
pub fn snake1d(x: []f32, alpha: []const f32, channels: usize, len: usize) void {
    std.debug.assert(x.len == channels * len);
    for (0..channels) |c| {
        const a = alpha[c];
        const inv = 1.0 / (a + 1e-9);
        for (x[c * len ..][0..len]) |*v| {
            const s = @sin(a * v.*);
            v.* += s * s * inv;
        }
    }
}

/// PyTorch's `adaptive_avg_pool1d` along the last axis: output bin `i` averages
/// `[floor(i * in / out), ceil((i + 1) * in / out))`. The bins are UNEVEN unless
/// `out` divides `in`; the real model's 256 -> 32 divides, so this generality is
/// only pinned by the fixture (which does not).
pub fn adaptiveAvgPool(out: []f32, x: []const f32, rows: usize, in_dim: usize, out_dim: usize) void {
    std.debug.assert(x.len == rows * in_dim and out.len == rows * out_dim);
    std.debug.assert(in_dim >= out_dim and out_dim > 0);
    for (0..rows) |r| {
        const xs = x[r * in_dim ..][0..in_dim];
        const os = out[r * out_dim ..][0..out_dim];
        for (os, 0..) |*o, i| {
            const s = i * in_dim / out_dim;
            const e = ((i + 1) * in_dim + out_dim - 1) / out_dim;
            var acc: f32 = 0;
            for (xs[s..e]) |v| acc += v;
            o.* = acc / @as(f32, @floatFromInt(e - s));
        }
    }
}

// --- weights --------------------------------------------------------------

/// A plain `nn.Linear`, `[out][in]` row-major, which is already the GEMM's B.
pub const Lin = struct {
    w: []const f32,
    b: ?[]const f32,
    out: usize,
    in: usize,

    fn weight(self: Lin) Weight {
        return Weight.init(std.mem.sliceAsBytes(self.w), .f32, self.out, self.in);
    }
};

pub const LayerNorm = struct { w: []f32, b: []f32 };

/// Snake -> dilated conv (k = 7) -> Snake -> pointwise conv, plus a residual that
/// is CENTER-CROPPED when the convolutions shortened the signal. With the padding
/// the reference uses they never do, so the crop is dead for every real config;
/// it is here because a checkpoint with different padding would silently rely on
/// it (and because the reference's `pad` computation is where the truncation
/// would come from).
pub const ResUnit = struct {
    a1: []f32,
    c1: Conv1d,
    a2: []f32,
    c2: Conv1d,
};

/// One `EncoderBlock`: three residual units at the INPUT width, then a Snake, then
/// the strided convolution that widens to the block's own width.
pub const Stage = struct {
    units: []ResUnit,
    act: []f32,
    down: Conv1d,

    pub fn stride(self: Stage) usize {
        return self.down.stride;
    }
};

/// The `AttnProjection` posterior head, operating on `[t][in_dim]` rows.
pub const Posterior = struct {
    heads: usize,
    in_dim: usize,
    out_dim: usize,
    /// Into the attention.
    norm1: LayerNorm,
    /// Into the residual projection.
    norm3: LayerNorm,
    proj: Lin,
    /// `[3 * in_dim][in_dim]`, no bias of its own.
    qkv: Lin,
    /// `cat(q_bias, zeros, v_bias)`, built at load so the GEMM takes it directly.
    qkv_bias: []f32,
    attn_proj: Lin,
    /// Into the MLP.
    norm2: LayerNorm,
    mlp_norm: LayerNorm,
    w0: Lin,
    w1: Lin,
    w2: Lin,

    pub fn headDim(self: Posterior) usize {
        return self.in_dim / self.heads;
    }
};

pub const AudioEncoder = struct {
    arena: std.heap.ArenaAllocator,
    /// `1 -> d_model`, k = 7.
    conv_in: Conv1d,
    stages: []Stage,
    /// The Snake before `conv_out`.
    act_out: []f32,
    /// `d_model -> d_latent`, k = 3.
    conv_out: Conv1d,
    head: Posterior,
    /// The 1x1 conv that reads the posterior MEAN.
    mean_proj: Conv1d,
    latents_mean: []f32,
    latents_std: []f32,

    pub fn deinit(self: *AudioEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Audio samples per latent frame: the product of the stage strides. 800 for
    /// the real checkpoint, the same figure the decoder upsamples by.
    pub fn hop(self: *const AudioEncoder) usize {
        var h: usize = 1;
        for (self.stages) |s| h *= s.stride();
        return h;
    }

    /// Latent frames a waveform of `samples` encodes to. The waveform is
    /// right-padded to a multiple of the hop, so this rounds UP.
    pub fn latentFrames(self: *const AudioEncoder, samples: usize) usize {
        const h = self.hop();
        return (samples + h - 1) / h;
    }

    /// Latent channels, i.e. `mean_proj`'s width (32 in the real checkpoint).
    pub fn latentChannels(self: *const AudioEncoder) usize {
        return self.mean_proj.out_ch;
    }

    /// Peak host bytes `encode` allocates for one stereo channel, so a caller can
    /// refuse a reference that would not fit rather than discover it by dying.
    ///
    /// The peak is at the FIRST stage, which runs the widest signal: the stage
    /// input, one residual unit's four intermediates, and the banded im2col. Every
    /// later stage is at least `stride` times shorter.
    pub fn peakBytesFor(self: *const AudioEncoder, samples: usize) usize {
        const padded = self.latentFrames(samples) * self.hop();
        if (self.stages.len == 0) return padded * @sizeOf(f32);
        const w = self.stages[0].units[0].c1.in_ch;
        // The stage input plus five live buffers of the same size, which is what a
        // residual unit peaks at, plus the im2col band (a fixed budget).
        return (6 * w * padded + gemm_band_floats) * @sizeOf(f32);
    }

    pub const Options = struct {
        /// Attention heads in the posterior head. NOT recoverable from the
        /// checkpoint: the fused qkv is one `[3 * in_dim][in_dim]` matrix whatever
        /// the split is, so nothing in the file says 8. It changes the answer (the
        /// scale, the causal softmax's grouping and the pooling's input width), so
        /// it is a stated architecture constant rather than a derived one.
        heads: usize = 8,
    };

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, o: Options) !AudioEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l: L = .{ .alloc = alloc, .store = store };

        // A stage is `encoder.block.{1 + i}`; the two tail modules follow it, so
        // the count is however many carry a strided conv at `.block.4`.
        var n_stages: usize = 0;
        while (true) : (n_stages += 1) {
            var buf: [96]u8 = undefined;
            const nm = std.fmt.bufPrint(&buf, "encoder.block.{d}.block.4.weight", .{n_stages + 1}) catch break;
            if (store.get(nm) == null) break;
        }
        if (n_stages == 0) return error.MissingTensor;

        const conv_in = try l.conv("encoder.block.0", .{});
        const stages = try alloc.alloc(Stage, n_stages);
        for (stages, 0..) |*st, i| {
            st.units = try alloc.alloc(ResUnit, res_dilations.len);
            for (st.units, res_dilations, 0..) |*u, dil, ui| {
                u.a1 = try l.vec("encoder.block.{d}.block.{d}.block.0.alpha", .{ i + 1, ui }, null);
                u.c1 = try l.conv("encoder.block.{d}.block.{d}.block.1", .{ i + 1, ui });
                u.a2 = try l.vec("encoder.block.{d}.block.{d}.block.2.alpha", .{ i + 1, ui }, null);
                u.c2 = try l.conv("encoder.block.{d}.block.{d}.block.3", .{ i + 1, ui });
                if (u.c1.k != res_k) return error.UnsupportedCheckpoint;
                // Dilation and its padding are coupled and neither is stored: the
                // padding must be `((k - 1) * d) / 2` or the unit stops being
                // length-preserving and the residual add starts cropping.
                u.c1.dilation = dil;
                u.c1.padding = (res_k - 1) * dil / 2;
            }
            st.act = try l.vec("encoder.block.{d}.block.3.alpha", .{i + 1}, null);
            st.down = try l.conv("encoder.block.{d}.block.4", .{i + 1});
            // `kernel_size = 2 * stride` is the relation that makes the stride
            // recoverable; `padding = ceil(stride / 2)` is what keeps the output
            // exactly `stride` times shorter.
            if (st.down.k % 2 != 0) return error.UnsupportedCheckpoint;
            st.down.stride = st.down.k / 2;
            st.down.padding = (st.down.stride + 1) / 2;
        }
        const act_out = try l.vec("encoder.block.{d}.alpha", .{n_stages + 1}, null);
        const conv_out = try l.conv("encoder.block.{d}", .{n_stages + 2});

        const qkv = try l.lin("pre_block.attn.qkv", .{}, false);
        if (qkv.out != 3 * qkv.in) return error.ShapeMismatch;
        const in_dim = qkv.in;
        const q_bias = try l.vec("pre_block.attn.q_bias", .{}, in_dim);
        const v_bias = try l.vec("pre_block.attn.v_bias", .{}, in_dim);
        const qkv_bias = try alloc.alloc(f32, 3 * in_dim);
        @memcpy(qkv_bias[0..in_dim], q_bias);
        // The k third is zero: `zero_k_bias` is a buffer, not a parameter. Read
        // from the file anyway, so a checkpoint that ever stopped zeroing it is
        // followed rather than second-guessed.
        @memcpy(qkv_bias[in_dim..][0..in_dim], try l.vec("pre_block.attn.zero_k_bias", .{}, in_dim));
        @memcpy(qkv_bias[2 * in_dim ..][0..in_dim], v_bias);

        const proj = try l.lin("pre_block.proj", .{}, true);
        if (o.heads == 0 or in_dim % o.heads != 0) return error.UnsupportedCheckpoint;
        const head: Posterior = .{
            .heads = o.heads,
            .in_dim = in_dim,
            .out_dim = proj.out,
            .norm1 = try l.layerNorm("pre_block.norm1", .{}),
            .norm3 = try l.layerNorm("pre_block.norm3", .{}),
            .proj = proj,
            .qkv = qkv,
            .qkv_bias = qkv_bias,
            .attn_proj = try l.lin("pre_block.attn.proj", .{}, true),
            .norm2 = try l.layerNorm("pre_block.norm2", .{}),
            .mlp_norm = try l.layerNorm("pre_block.mlp.norm", .{}),
            .w0 = try l.lin("pre_block.mlp.w0", .{}, true),
            .w1 = try l.lin("pre_block.mlp.w1", .{}, true),
            .w2 = try l.lin("pre_block.mlp.w2", .{}, true),
        };

        const mean_proj = try l.conv("mean_proj", .{});
        const c_lat = mean_proj.out_ch;
        return .{
            .arena = arena,
            .conv_in = conv_in,
            .stages = stages,
            .act_out = act_out,
            .conv_out = conv_out,
            .head = head,
            .mean_proj = mean_proj,
            .latents_mean = try l.vec("latents_mean", .{}, c_lat),
            .latents_std = try l.vec("latents_std", .{}, c_lat),
        };
    }
};

const L = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,

    fn tensor(l: L, comptime fmt: []const u8, args: anytype) !weights_mod.TensorView {
        var buf: [128]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        return l.store.get(nm) orelse {
            std.log.err("minimax_h3_audio_encode: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
    }

    /// `len == null` accepts whatever is there (the Snake alphas are stored
    /// `[1][c][1]`, and the channel count is what they report).
    fn vec(l: L, comptime fmt: []const u8, args: anytype, len: ?usize) ![]f32 {
        const v = try l.tensor(fmt, args);
        if (len) |n| {
            if (v.info.elemCount() != n) return error.ShapeMismatch;
        }
        return v.toF32Alloc(l.alloc);
    }

    /// A Conv1d with `padding` set to the reference's `(k - 1) / 2`, which is what
    /// every unstrided conv here uses (`k = 7 -> 3`, `k = 3 -> 1`, `k = 1 -> 0`).
    /// The strided convs and the dilated ones are patched by the caller.
    fn conv(l: L, comptime fmt: []const u8, args: anytype) !Conv1d {
        const w = try l.tensor(fmt ++ ".weight", args);
        const s = w.info.shape.slice();
        if (s.len != 3) return error.ShapeMismatch;
        const out_ch = s[0];
        const in_ch = s[1];
        const k = s[2];
        return .{
            .w = try w.toF32Alloc(l.alloc),
            .b = try l.vec(fmt ++ ".bias", args, out_ch),
            .out_ch = out_ch,
            .in_ch = in_ch,
            .k = k,
            .padding = (k - 1) / 2,
        };
    }

    fn lin(l: L, comptime fmt: []const u8, args: anytype, has_bias: bool) !Lin {
        const w = try l.tensor(fmt ++ ".weight", args);
        const s = w.info.shape.slice();
        if (s.len != 2) return error.ShapeMismatch;
        return .{
            .w = try w.toF32Alloc(l.alloc),
            .b = if (has_bias) try l.vec(fmt ++ ".bias", args, s[0]) else null,
            .out = s[0],
            .in = s[1],
        };
    }

    fn layerNorm(l: L, comptime fmt: []const u8, args: anytype) !LayerNorm {
        const w = try l.vec(fmt ++ ".weight", args, null);
        return .{ .w = w, .b = try l.vec(fmt ++ ".bias", args, w.len) };
    }
};

// --- forward --------------------------------------------------------------

/// The im2col band budget in f32 elements. The first residual stage runs 64
/// channels over the FULL sample count, so an unbanded `[out_len][in_ch * k]`
/// column matrix is 573 MB for ten seconds of audio; banding caps it here
/// regardless of length. Mirrors `minimax_h3_vae_encode.conv3d`.
const gemm_band_floats: usize = 1 << 22;

/// A planar `[ch][len]` signal.
const Sig = struct { d: []f32, ch: usize, len: usize };

fn ownSig(a: std.mem.Allocator, s: Sig) !Sig {
    return .{ .d = try a.dupe(f32, s.d), .ch = s.ch, .len = s.len };
}

fn runConv(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, x: Sig, c: Conv1d) !Sig {
    const out_len = c.outLen(x.len);
    const out = try a.alloc(f32, c.out_ch * out_len);
    try audio.conv1d(a, io, gpa, out, x.d, c, x.len);
    return .{ .d = out, .ch = c.out_ch, .len = out_len };
}

/// One `ResidualUnit`. The residual is center-cropped if the convolutions
/// shortened the signal, which with the reference's padding they never do.
fn runResUnit(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, x: Sig, u: *const ResUnit) !Sig {
    var t = try ownSig(a, x);
    snake1d(t.d, u.a1, t.ch, t.len);
    t = try runConv(a, io, gpa, t, u.c1);
    snake1d(t.d, u.a2, t.ch, t.len);
    t = try runConv(a, io, gpa, t, u.c2);
    std.debug.assert(t.ch == x.ch);
    const crop = (x.len - t.len) / 2;
    for (0..t.ch) |c| {
        const dst = t.d[c * t.len ..][0..t.len];
        const src = x.d[c * x.len + crop ..][0..t.len];
        for (dst, src) |*o, v| o.* += v;
    }
    return t;
}

/// The DAC encoder trunk: `conv_in`, the stages, then Snake + `conv_out`.
///
/// Each stage gets its own arena and only the carried signal crosses the
/// boundary, so the peak is one stage's intermediates rather than all of them.
/// Stage 0 is the widest by far (the full sample count), and running everything
/// into one arena is what makes a ten-second reference several gigabytes.
fn encodeTrunk(io: std.Io, gpa: std.mem.Allocator, a: std.mem.Allocator, enc: *const AudioEncoder, x: []const f32, len: usize) !Sig {
    var h = blk: {
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const first = try runConv(la.allocator(), io, gpa, .{ .d = @constCast(x), .ch = 1, .len = len }, enc.conv_in);
        break :blk try ownSig(gpa, first);
    };
    errdefer gpa.free(h.d);

    for (enc.stages) |*st| {
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const sa = la.allocator();
        var t = h;
        for (st.units) |*u| t = try runResUnit(sa, io, gpa, t, u);
        snake1d(t.d, st.act, t.ch, t.len);
        t = try runConv(sa, io, gpa, t, st.down);
        const next = try ownSig(gpa, t);
        gpa.free(h.d);
        h = next;
    }

    var tail = std.heap.ArenaAllocator.init(gpa);
    defer tail.deinit();
    snake1d(h.d, enc.act_out, h.ch, h.len);
    const out = try runConv(tail.allocator(), io, gpa, h, enc.conv_out);
    gpa.free(h.d);
    return ownSig(a, out);
}

/// The posterior head over `[t][in_dim]` rows, in place of nothing: returns
/// `[t][out_dim]`.
fn posterior(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, p: *const Posterior, x: []const f32, t: usize) ![]f32 {
    const din = p.in_dim;
    const dout = p.out_dim;
    const hd = p.headDim();

    // x = proj(norm3(x)) + attn(norm1(x)). norm3 feeds the projection.
    const n3 = try a.alloc(f32, t * din);
    ops.norm.layerNorm(n3, x, p.norm3.w, p.norm3.b, ln_eps);
    const y = try a.alloc(f32, t * dout);
    try ops.matmul.matmul(io, gpa, y, n3, t, p.proj.weight(), p.proj.b);

    const n1 = try a.alloc(f32, t * din);
    ops.norm.layerNorm(n1, x, p.norm1.w, p.norm1.b, ln_eps);
    const qkv = try a.alloc(f32, t * 3 * din);
    try ops.matmul.matmul(io, gpa, qkv, n1, t, p.qkv.weight(), p.qkv_bias);

    // Causal attention, then the MEAN over heads: accumulate every head's output
    // into one `[t][hd]` plane and divide once.
    const pooled_in = try a.alloc(f32, t * hd);
    @memset(pooled_in, 0);
    const scores = try a.alloc(f32, t);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (0..p.heads) |h| {
        for (0..t) |i| {
            const q = qkv[i * 3 * din + h * hd ..][0..hd];
            var max: f32 = -std.math.floatMax(f32);
            for (0..i + 1) |j| {
                const k = qkv[j * 3 * din + din + h * hd ..][0..hd];
                var dot: f32 = 0;
                for (q, k) |qv, kv| dot += qv * kv;
                scores[j] = dot * scale;
                max = @max(max, scores[j]);
            }
            var sum: f32 = 0;
            for (scores[0 .. i + 1]) |*s| {
                s.* = @exp(s.* - max);
                sum += s.*;
            }
            const inv = 1.0 / sum;
            const acc = pooled_in[i * hd ..][0..hd];
            for (0..i + 1) |j| {
                const w = scores[j] * inv;
                const v = qkv[j * 3 * din + 2 * din + h * hd ..][0..hd];
                for (acc, v) |*o, vv| o.* += w * vv;
            }
        }
    }
    const inv_h = 1.0 / @as(f32, @floatFromInt(p.heads));
    for (pooled_in) |*v| v.* *= inv_h;

    // ...pooled along the FEATURE axis down to the latent width, then projected.
    const pooled = try a.alloc(f32, t * dout);
    adaptiveAvgPool(pooled, pooled_in, t, hd, dout);
    const attn = try a.alloc(f32, t * dout);
    try ops.matmul.matmul(io, gpa, attn, pooled, t, p.attn_proj.weight(), p.attn_proj.b);
    for (y, attn) |*o, v| o.* += v;

    // x += mlp(norm2(x)), and the MLP has its own LayerNorm inside.
    const n2 = try a.alloc(f32, t * dout);
    ops.norm.layerNorm(n2, y, p.norm2.w, p.norm2.b, ln_eps);
    ops.norm.layerNorm(n2, n2, p.mlp_norm.w, p.mlp_norm.b, ln_eps);
    const hidden = p.w0.out;
    const g = try a.alloc(f32, t * hidden);
    try ops.matmul.matmul(io, gpa, g, n2, t, p.w0.weight(), p.w0.b);
    const up = try a.alloc(f32, t * hidden);
    try ops.matmul.matmul(io, gpa, up, n2, t, p.w1.weight(), p.w1.b);
    ops.act.geluTanhMul(g, up);
    const mlp = try a.alloc(f32, t * dout);
    try ops.matmul.matmul(io, gpa, mlp, g, t, p.w2.weight(), p.w2.b);
    for (y, mlp) |*o, v| o.* += v;
    return y;
}

/// Encode an interleaved stereo waveform in [-1, 1] to normalized latents,
/// planar `[c_lat][2][t]` -- the layout `minimax_h3_audio.decode` reads back.
///
/// `wav` is `[len][channels]` interleaved, which is what a container gives. A
/// MONO input is accepted and duplicated across both stereo halves, because a
/// mono reference is a normal thing to hand the model and the network has no mono
/// mode. `t` must be `latentFrames(len)`.
pub fn encode(
    enc: *const AudioEncoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    wav: []const f32,
    len: usize,
    channels: usize,
) !void {
    const c_lat = enc.latentChannels();
    const t = enc.latentFrames(len);
    const padded = t * enc.hop();
    std.debug.assert(channels == 1 or channels == audio.stereo);
    std.debug.assert(wav.len == len * channels);
    std.debug.assert(out.len == c_lat * audio.stereo * t);

    for (0..audio.stereo) |s| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Right-padded with zeros to a multiple of the hop, mono, planar.
        const x = try a.alloc(f32, padded);
        @memset(x[len..], 0);
        const src = if (channels == 1) 0 else s;
        for (0..len) |i| x[i] = wav[i * channels + src];

        const trunk = try encodeTrunk(io, gpa, a, enc, x, padded);
        std.debug.assert(trunk.len == t);

        // The head wants `[t][in_dim]` rows; the trunk speaks planar.
        const rows = try a.alloc(f32, t * trunk.ch);
        for (0..t) |i| {
            for (0..trunk.ch) |c| rows[i * trunk.ch + c] = trunk.d[c * t + i];
        }
        const h = try posterior(a, io, gpa, &enc.head, rows, t);

        // ...and `mean_proj` is a 1x1 conv over the planar form again.
        const planar = try a.alloc(f32, enc.head.out_dim * t);
        for (0..t) |i| {
            for (0..enc.head.out_dim) |c| planar[c * t + i] = h[i * enc.head.out_dim + c];
        }
        const z = try runConv(a, io, gpa, .{ .d = planar, .ch = enc.head.out_dim, .len = t }, enc.mean_proj);
        std.debug.assert(z.ch == c_lat);

        for (0..c_lat) |c| {
            const inv = 1.0 / enc.latents_std[c];
            const mean = enc.latents_mean[c];
            const dst = out[(c * audio.stereo + s) * t ..][0..t];
            for (dst, z.d[c * t ..][0..t]) |*o, v| o.* = (v - mean) * inv;
        }
    }
}

// --- tests ----------------------------------------------------------------

const encode_fixture = @embedFile("assets/minimax_h3_audio_encode.safetensors");

test "adaptive average pooling matches torch's uneven bins" {
    // out bin i averages [floor(i*in/out), ceil((i+1)*in/out)). For 4 -> 3 that is
    // [0,2), [1,3), [2,4): OVERLAPPING, which a "block mean" reading is not.
    const x = [_]f32{ 1, 2, 3, 4 };
    var out: [3]f32 = undefined;
    adaptiveAvgPool(&out, &x, 1, 4, 3);
    try std.testing.expectApproxEqAbs(1.5, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(2.5, out[1], 1e-6);
    try std.testing.expectApproxEqAbs(3.5, out[2], 1e-6);
    // An even split is the plain block mean, which is the real model's 256 -> 32.
    var half: [2]f32 = undefined;
    adaptiveAvgPool(&half, &x, 1, 4, 2);
    try std.testing.expectApproxEqAbs(1.5, half[0], 1e-6);
    try std.testing.expectApproxEqAbs(3.5, half[1], 1e-6);
}

test "the encoder snake uses alpha linearly, as its own beta" {
    // x + sin(a x)^2 / a, NOT the decoder's exp(a) / exp(b).
    var x = [_]f32{ 0.5, -0.5 };
    const alpha = [_]f32{2.0};
    snake1d(&x, &alpha, 1, 2);
    const s = @sin(@as(f32, 1.0));
    try std.testing.expectApproxEqAbs(0.5 + s * s / 2.0, x[0], 1e-6);
    try std.testing.expectApproxEqAbs(-0.5 + s * s / 2.0, x[1], 1e-6);
}

test "the strided convolutions shorten by exactly their stride" {
    // `k = 2 * stride` with `padding = ceil(stride / 2)` is the pairing that makes
    // the hop exact, which is why the stride is recovered from the kernel.
    for ([_]usize{ 2, 4, 5 }) |u| {
        const c: Conv1d = .{
            .w = &.{},
            .b = null,
            .out_ch = 1,
            .in_ch = 1,
            .k = 2 * u,
            .stride = u,
            .padding = (u + 1) / 2,
        };
        try std.testing.expectEqual(@as(usize, 11), c.outLen(11 * u));
    }
    // ...and the real strides multiply to the decoder's samples-per-latent-frame.
    var h: usize = 1;
    for ([_]usize{ 2, 4, 4, 5, 5 }) |u| h *= u;
    try std.testing.expectEqual(@as(usize, audio.samples_per_latent), h);
}

test "the residual dilations are the encoder's cycle, not the decoder's" {
    // 1, 3, 9 here; 1, 3, 5 in the vocoder. Same shape, different network.
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 9 }, &res_dilations);
    for (res_dilations) |d| {
        const c: Conv1d = .{
            .w = &.{},
            .b = null,
            .out_ch = 1,
            .in_ch = 1,
            .k = res_k,
            .dilation = d,
            .padding = (res_k - 1) * d / 2,
        };
        // Length-preserving, so the residual add never actually crops.
        try std.testing.expectEqual(@as(usize, 64), c.outLen(64));
    }
}

test "the audio encode matches the reference at a toy width" {
    // ComfyUI's own DAC encoder + posterior head at a toy width, from
    // tools/gen_minimax_h3_audio_encode.py. Two stages so both padding relations
    // run (stride 2, kernel 4, pad 1 and stride 5, kernel 10, pad 3), and a
    // latent width that does NOT divide the head dim, so the adaptive pooling's
    // uneven bins are exercised.
    const gpa = std.testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, encode_fixture);
    defer st.deinit();
    var enc = try AudioEncoder.load(gpa, .{ .safetensors = &st }, .{ .heads = 4 });
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 2), enc.stages.len);
    try std.testing.expectEqual(@as(usize, 2), enc.stages[0].stride());
    try std.testing.expectEqual(@as(usize, 5), enc.stages[1].stride());
    try std.testing.expectEqual(@as(usize, 10), enc.hop());
    try std.testing.expectEqual(@as(usize, 3), enc.stages[0].units.len);
    try std.testing.expectEqual(@as(usize, 9), enc.stages[0].units[2].c1.dilation);
    try std.testing.expectEqual(@as(usize, 6), enc.latentChannels());
    // head_dim 16 over a 6-wide latent: the pooling bins really are uneven
    try std.testing.expectEqual(@as(usize, 16), enc.head.headDim());
    try std.testing.expect(enc.head.headDim() % enc.latentChannels() != 0);
    // the k third of the fused bias is zero
    for (enc.head.qkv_bias[enc.head.in_dim..][0..enc.head.in_dim]) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }

    const wv = try st.require("in.waveform"); // [1][2][len], planar
    const planar = try wv.toF32Alloc(gpa);
    defer gpa.free(planar);
    const len = planar.len / audio.stereo;
    const wantv = try st.require("out.latent"); // [1][c][2][t]
    const want = try wantv.toF32Alloc(gpa);
    defer gpa.free(want);

    // `encode` takes interleaved, which is what a container gives.
    const wav = try gpa.alloc(f32, planar.len);
    defer gpa.free(wav);
    for (0..len) |i| {
        for (0..audio.stereo) |s| wav[i * audio.stereo + s] = planar[s * len + i];
    }

    const t = enc.latentFrames(len);
    try std.testing.expectEqual(want.len, enc.latentChannels() * audio.stereo * t);
    const out = try gpa.alloc(f32, want.len);
    defer gpa.free(out);
    try encode(&enc, std.testing.io, gpa, out, wav, len, audio.stereo);

    const err = relL2(want, out);
    errdefer std.debug.print("audio encode rel L2 {e}\n", .{err});
    try std.testing.expect(err < 1e-5);

    // The two stereo halves must genuinely differ: they share one network, so
    // encoding one and copying it would pass a same-shape check. Compare the
    // per-channel interleave, which is where a stereo mix-up would land.
    var diff: f64 = 0;
    var ref: f64 = 0;
    for (0..enc.latentChannels()) |c| {
        for (0..t) |i| {
            const l0 = out[(c * audio.stereo + 0) * t + i];
            const r0 = out[(c * audio.stereo + 1) * t + i];
            diff += @as(f64, l0 - r0) * (l0 - r0);
            ref += @as(f64, l0) * l0;
        }
    }
    try std.testing.expect(@sqrt(diff / ref) > 0.1);

    // A mono input duplicates rather than being refused, and the duplicate really
    // is the left channel's answer.
    const mono = try gpa.alloc(f32, len);
    defer gpa.free(mono);
    for (0..len) |i| mono[i] = wav[i * audio.stereo];
    const out_mono = try gpa.alloc(f32, want.len);
    defer gpa.free(out_mono);
    try encode(&enc, std.testing.io, gpa, out_mono, mono, len, 1);
    for (0..enc.latentChannels()) |c| {
        const l0 = out[(c * audio.stereo + 0) * t ..][0..t];
        const m0 = out_mono[(c * audio.stereo + 0) * t ..][0..t];
        const m1 = out_mono[(c * audio.stereo + 1) * t ..][0..t];
        try std.testing.expect(relL2(l0, m0) < 1e-6);
        try std.testing.expectEqualSlices(f32, m0, m1);
    }
}

test "the encoder's peak host bytes do not grow with the trunk depth" {
    // The whole point of the per-stage arenas: doubling the waveform must roughly
    // double the peak, not multiply it by the stage count. A regression here is
    // what makes a ten-second reference tens of gigabytes.
    const gpa = std.testing.allocator;
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, encode_fixture);
    defer st.deinit();
    var enc = try AudioEncoder.load(gpa, .{ .safetensors = &st }, .{ .heads = 4 });
    defer enc.deinit();

    const one = enc.peakBytesFor(1000);
    const two = enc.peakBytesFor(2000);
    const grow = @as(f64, @floatFromInt(two - one)) / @as(f64, @floatFromInt(one));
    errdefer std.debug.print("peak 1000 -> {d}, 2000 -> {d}\n", .{ one, two });
    try std.testing.expect(grow < 1.05);
}

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

const test_gate = @import("../test_gate.zig");
const real_audio_vae = "/home/qt/genai/comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors";

test "the real audio VAE's encoder loads and its strides multiply to 800" {
    // Pins the values that are NOT stored: the strides recovered from the kernel
    // widths, the dilation cycle, and the head count.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, real_audio_vae);

    var st = try tp_core.safetensors.SafeTensors.open(gpa, io, real_audio_vae);
    defer st.deinit();
    var enc = try AudioEncoder.load(gpa, .{ .safetensors = &st }, .{});
    defer enc.deinit();

    try std.testing.expectEqual(@as(usize, 5), enc.stages.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 4, 5, 5 }, &.{
        enc.stages[0].stride(), enc.stages[1].stride(), enc.stages[2].stride(),
        enc.stages[3].stride(), enc.stages[4].stride(),
    });
    // 32 kHz at 40 latent fps, the same figure the vocoder upsamples by.
    try std.testing.expectEqual(@as(usize, audio.samples_per_latent), enc.hop());
    try std.testing.expectEqual(@as(usize, audio.latent_channels), enc.latentChannels());
    try std.testing.expectEqual(@as(usize, 1), enc.conv_in.in_ch);
    try std.testing.expectEqual(@as(usize, 2048), enc.conv_out.out_ch);
    // the posterior head: 2048 wide, 8 heads, pooling 256 -> 32
    try std.testing.expectEqual(@as(usize, 2048), enc.head.in_dim);
    try std.testing.expectEqual(@as(usize, 8), enc.head.heads);
    try std.testing.expectEqual(@as(usize, 256), enc.head.headDim());
    try std.testing.expectEqual(@as(usize, audio.latent_channels), enc.head.out_dim);
    // one second of stereo audio is 40 latent frames
    try std.testing.expectEqual(@as(usize, 40), enc.latentFrames(audio.sample_rate));
}

test "a real encode/decode round trip keeps each channel's pitch" {
    // The toy fixture pins the arithmetic; this pins that the arithmetic is wired
    // to the RIGHT weights at the real widths, by sending a signal through both
    // halves of the real VAE and asking what came back.
    //
    // Two different tones, one per stereo channel, is the whole test: a wrong
    // dilation cycle, a pooled TIME axis or a swapped norm all produce a latent
    // that decodes to something, and only a pitch that survives says the encoder
    // and the decoder agree about what a latent means. It also catches a stereo
    // swap, which no norm would.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, real_audio_vae);

    var st = try tp_core.safetensors.SafeTensors.open(gpa, io, real_audio_vae);
    defer st.deinit();
    var enc = try AudioEncoder.load(gpa, .{ .safetensors = &st }, .{});
    defer enc.deinit();
    var dec = try audio.AudioDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    const f_left: f32 = 220.0;
    const f_right: f32 = 660.0; // well separated, and not a harmonic relation
    const len: usize = audio.sample_rate / 2; // half a second
    const wav = try gpa.alloc(f32, len * audio.stereo);
    defer gpa.free(wav);
    const sr: f32 = @floatFromInt(audio.sample_rate);
    for (0..len) |i| {
        const t = @as(f32, @floatFromInt(i)) / sr;
        // A raised-cosine fade over the whole clip, so neither end is a click the
        // vocoder would have to reproduce.
        const env = 0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) /
            @as(f32, @floatFromInt(len - 1))));
        wav[i * 2 + 0] = 0.6 * env * @sin(2.0 * std.math.pi * f_left * t);
        wav[i * 2 + 1] = 0.6 * env * @sin(2.0 * std.math.pi * f_right * t);
    }

    const at = enc.latentFrames(len);
    const z = try gpa.alloc(f32, enc.latentChannels() * audio.stereo * at);
    defer gpa.free(z);
    try encode(&enc, io, gpa, z, wav, len, audio.stereo);
    for (z) |v| try std.testing.expect(std.math.isFinite(v));

    const samples = at * dec.upsampleFactor();
    const out = try gpa.alloc(f32, samples * audio.stereo);
    defer gpa.free(out);
    try audio.decode(&dec, io, gpa, out, z, at);

    // Goertzel energy at one frequency in one interleaved channel.
    const tone = struct {
        fn go(x: []const f32, n: usize, ch: usize, freq: f32, rate: f32) f64 {
            const w = 2.0 * std.math.pi * freq / rate;
            const coeff = 2.0 * @cos(w);
            var s1: f64 = 0;
            var s2: f64 = 0;
            for (0..n) |i| {
                const s = @as(f64, x[i * audio.stereo + ch]) + coeff * s1 - s2;
                s2 = s1;
                s1 = s;
            }
            return s1 * s1 + s2 * s2 - coeff * s1 * s2;
        }
    }.go;

    const ll = tone(out, samples, 0, f_left, sr);
    const lr = tone(out, samples, 0, f_right, sr);
    const rl = tone(out, samples, 1, f_left, sr);
    const rr = tone(out, samples, 1, f_right, sr);
    errdefer std.debug.print(
        "round trip: left {e} at {d} Hz vs {e} at {d} Hz; right {e} vs {e}\n",
        .{ ll, f_left, lr, f_right, rl, rr },
    );
    // Each channel's own tone must dominate the other's by a wide margin. A
    // stereo swap flips both ratios; a broken encode flattens both toward 1.
    try std.testing.expect(ll > 10.0 * lr);
    try std.testing.expect(rr > 10.0 * rl);

    // ...and the output is a real signal, not near-silence the ratios would still
    // satisfy. The clip is at 0.6 peak, so a round trip that lost 30 dB of level
    // is a failure even with the pitch intact.
    var peak: f32 = 0;
    for (out) |v| peak = @max(peak, @abs(v));
    errdefer std.debug.print("round-trip peak {d}\n", .{peak});
    try std.testing.expect(peak > 0.02);
}
