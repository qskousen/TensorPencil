//! MiniMax H3 audio VAE, decode side: a BigVGAN vocoder at 32 kHz.
//!
//! Latents are `[32][2][t]` at 40 frames per second; each stereo channel is
//! decoded INDEPENDENTLY through the same mono vocoder, 800 audio samples per
//! latent frame. This is the one part of H3 that needs 1-D convolution, which
//! nothing else in the engine does.
//!
//! Conventions that are silent wrong answers when got wrong:
//!
//! - **SnakeBeta stores alpha and beta in LOG scale**: the activation is
//!   `x + sin(exp(a) x)^2 / exp(b)`. Using them raw is finite and wrong. (The
//!   encoder's `Snake1d` stores them linearly and shares one parameter for both;
//!   only the decoder is implemented here.)
//! - **The activation is ANTI-ALIASED**: upsample x2, apply snake, downsample x2,
//!   with kaiser-sinc filters and REPLICATE padding. A plain pointwise snake is
//!   the obvious wrong answer and sounds almost right.
//! - The kaiser-sinc filters are **stored in the checkpoint**, so they are loaded
//!   rather than recomputed. Rederiving the window is a needless numeric risk.
//! - Each `AMPBlock1` pairs `activations[::2]` with `convs1` and `activations[1::2]`
//!   with `convs2`; the six activations are NOT three shared pairs.
//! - The three resblocks at each upsampling stage are SUMMED and divided by
//!   three, not chained.
//! - The final output is clamped to [-1, 1] with no tanh and no bias.
//! - Latents are denormalized per channel before `dec_in_proj`.
//!
//! Reference is ComfyUI `comfy/ldm/minimax/audio_vae.py`, whose lineage is
//! descript-audio-codec (MIT) and NVIDIA BigVGAN (MIT). See VIDEO_PLAN.md.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const sample_rate: u32 = 32000;
pub const latent_channels = 32;
pub const stereo = 2;
/// Audio samples per latent frame, i.e. the product of the upsample rates.
pub const samples_per_latent = 800;

// --- 1-D primitives -------------------------------------------------------
//
// Signals are planar `[channels][length]`. These are the only 1-D convolutions
// in the engine, so they live here rather than in `ops/`; hoist them if a second
// consumer appears.

/// A 1-D convolution's weights, `[out_ch][in_ch / groups][k]`.
pub const Conv1d = struct {
    w: []const f32,
    b: ?[]const f32,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize = 1,
    dilation: usize = 1,
    padding: usize = 0,
    groups: usize = 1,

    pub fn outLen(self: Conv1d, in_len: usize) usize {
        const eff = self.dilation * (self.k - 1) + 1;
        const padded = in_len + 2 * self.padding;
        if (padded < eff) return 0;
        return (padded - eff) / self.stride + 1;
    }
};

/// `out[oc][t] = b[oc] + sum_ic sum_j w[oc][ic][j] * x[ic][t*stride - pad + j*dil]`,
/// zero outside the input.
///
/// An UNGROUPED convolution goes through `ops.matmul` as an im2col GEMM, because
/// a Conv1d is a GEMM and the weight layout already is its B matrix: `[out_ch]
/// [in_ch][k]` is contiguously `[out_ch][in_ch * k]`. The naive triple loop this
/// replaced ran the whole vocoder at 0.53 GFLOP/s, which made the audio VAE 303 s
/// of a 311 s decode; the shared GEMM is threaded and vectorized.
///
/// Grouped convolutions (`groups == in_ch`, the kaiser-sinc filters) stay on the
/// direct path: each output channel sees one input channel, so the im2col matrix
/// would be a `k`-wide sliver per group and the GEMM would be all overhead.
pub fn conv1d(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, out: []f32, x: []const f32, c: Conv1d, in_len: usize) !void {
    const out_len = c.outLen(in_len);
    std.debug.assert(x.len == c.in_ch * in_len);
    std.debug.assert(out.len == c.out_ch * out_len);
    std.debug.assert(c.in_ch % c.groups == 0 and c.out_ch % c.groups == 0);
    if (out_len == 0) return;
    if (c.groups == 1) return conv1dGemm(a, io, gpa, out, x, c, in_len, out_len);

    const in_per_group = c.in_ch / c.groups;
    const out_per_group = c.out_ch / c.groups;
    for (0..c.out_ch) |oc| {
        const g = oc / out_per_group;
        const row = out[oc * out_len ..][0..out_len];
        const bias = if (c.b) |bb| bb[oc] else 0.0;
        @memset(row, bias);
        for (0..in_per_group) |ici| {
            const ic = g * in_per_group + ici;
            const xs = x[ic * in_len ..][0..in_len];
            const ws = c.w[(oc * in_per_group + ici) * c.k ..][0..c.k];
            for (0..out_len) |t| {
                const base = @as(isize, @intCast(t * c.stride)) - @as(isize, @intCast(c.padding));
                var acc: f32 = 0;
                for (0..c.k) |j| {
                    const s = base + @as(isize, @intCast(j * c.dilation));
                    if (s < 0 or s >= @as(isize, @intCast(in_len))) continue;
                    acc += ws[j] * xs[@intCast(s)];
                }
                row[t] += acc;
            }
        }
    }
}

/// The im2col band budget in f32 elements, 16 MB. Without it the column matrix
/// is `out_len * in_ch * k`, which for the ENCODER's first residual stage (64
/// channels, k = 7, the full sample count) is 573 MB for ten seconds of audio and
/// grows linearly with the reference's length. Banded, the cost is fixed and the
/// GEMM still sees thousands of rows.
const gemm_band_floats: usize = 1 << 22;

/// im2col + one GEMM per band + transpose back to planar.
///
/// The column order is `(in_ch, tap)`, matching the weight's own `[in_ch][k]`
/// row layout; any other order silently pairs a tap with the wrong channel.
fn conv1dGemm(
    a: std.mem.Allocator,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x: []const f32,
    c: Conv1d,
    in_len: usize,
    out_len: usize,
) !void {
    const cols = c.in_ch * c.k;
    const band = @max(1, @min(out_len, gemm_band_floats / cols));
    const cm = try a.alloc(f32, band * cols);
    defer a.free(cm);
    const y = try a.alloc(f32, band * c.out_ch);
    defer a.free(y);
    const w = Weight.init(std.mem.sliceAsBytes(c.w), .f32, c.out_ch, cols);

    var t0: usize = 0;
    while (t0 < out_len) : (t0 += band) {
        const n = @min(band, out_len - t0);
        for (0..n) |bt| {
            const t = t0 + bt;
            const base = @as(isize, @intCast(t * c.stride)) - @as(isize, @intCast(c.padding));
            const row = cm[bt * cols ..][0..cols];
            for (0..c.in_ch) |ic| {
                const xs = x[ic * in_len ..][0..in_len];
                for (0..c.k) |j| {
                    const sp = base + @as(isize, @intCast(j * c.dilation));
                    row[ic * c.k + j] = if (sp < 0 or sp >= @as(isize, @intCast(in_len))) 0 else xs[@intCast(sp)];
                }
            }
        }

        // `[n][out_ch]` from the GEMM, transposed back to the planar
        // `[out_ch][out_len]` the rest of the network speaks.
        try ops.matmul.matmul(io, gpa, y[0 .. n * c.out_ch], cm[0 .. n * cols], n, w, c.b);
        for (0..c.out_ch) |oc| {
            const row = out[oc * out_len + t0 ..][0..n];
            for (row, 0..) |*o, bt| o.* = y[bt * c.out_ch + oc];
        }
    }
}

/// A 1-D transposed convolution's weights, `[in_ch][out_ch / groups][k]`.
///
/// Note the weight layout is IN-channel major, the opposite of `Conv1d`. That is
/// PyTorch's convention and reading it the other way is a shape mismatch only
/// when the two channel counts differ.
pub const ConvT1d = struct {
    w: []const f32,
    b: ?[]const f32,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize = 1,
    padding: usize = 0,
    groups: usize = 1,

    pub fn outLen(self: ConvT1d, in_len: usize) usize {
        return (in_len - 1) * self.stride + self.k - 2 * self.padding;
    }
};

/// Scatter form: input position `s`, tap `j` contributes to output
/// `s * stride - padding + j`.
pub fn convT1d(out: []f32, x: []const f32, c: ConvT1d, in_len: usize) void {
    const out_len = c.outLen(in_len);
    std.debug.assert(x.len == c.in_ch * in_len);
    std.debug.assert(out.len == c.out_ch * out_len);
    std.debug.assert(c.in_ch % c.groups == 0 and c.out_ch % c.groups == 0);
    const in_per_group = c.in_ch / c.groups;
    const out_per_group = c.out_ch / c.groups;

    for (0..c.out_ch) |oc| {
        const row = out[oc * out_len ..][0..out_len];
        @memset(row, if (c.b) |bb| bb[oc] else 0.0);
    }
    for (0..c.in_ch) |ic| {
        const g = ic / in_per_group;
        const xs = x[ic * in_len ..][0..in_len];
        for (0..out_per_group) |oci| {
            const oc = g * out_per_group + oci;
            const row = out[oc * out_len ..][0..out_len];
            const ws = c.w[(ic * out_per_group + oci) * c.k ..][0..c.k];
            for (0..in_len) |s| {
                const xv = xs[s];
                if (xv == 0) continue;
                const base = @as(isize, @intCast(s * c.stride)) - @as(isize, @intCast(c.padding));
                for (0..c.k) |j| {
                    const t = base + @as(isize, @intCast(j));
                    if (t < 0 or t >= @as(isize, @intCast(out_len))) continue;
                    row[@intCast(t)] += ws[j] * xv;
                }
            }
        }
    }
}

/// Edge-replicate padding, per channel.
pub fn padReplicate(out: []f32, x: []const f32, channels: usize, in_len: usize, left: usize, right: usize) void {
    const out_len = in_len + left + right;
    std.debug.assert(x.len == channels * in_len);
    std.debug.assert(out.len == channels * out_len);
    std.debug.assert(in_len > 0);
    for (0..channels) |c| {
        const src = x[c * in_len ..][0..in_len];
        const dst = out[c * out_len ..][0..out_len];
        @memset(dst[0..left], src[0]);
        @memcpy(dst[left..][0..in_len], src);
        @memset(dst[left + in_len ..][0..right], src[in_len - 1]);
    }
}

/// `x + sin(alpha * x)^2 / beta`, per channel, with alpha and beta given in LOG
/// scale (the decoder's `SnakeBeta`). The `1e-9` matches the reference's guard.
pub fn snakeBeta(x: []f32, log_alpha: []const f32, log_beta: []const f32, channels: usize, len: usize) void {
    std.debug.assert(x.len == channels * len);
    for (0..channels) |c| {
        const a = @exp(log_alpha[c]);
        const inv_b = 1.0 / (@exp(log_beta[c]) + 1e-9);
        for (x[c * len ..][0..len]) |*v| {
            const s = @sin(a * v.*);
            v.* += s * s * inv_b;
        }
    }
}

// --- weights --------------------------------------------------------------

/// Anti-aliased activation: upsample x2 -> snake -> downsample x2. The two
/// kaiser-sinc filters are `[k]` each, applied per channel (`groups = C`).
pub const Activation = struct {
    log_alpha: []f32,
    log_beta: []f32,
    up_filter: []f32,
    down_filter: []f32,
    channels: usize,

    /// Both filters are 12 taps in this checkpoint; the padding constants below
    /// are derived from that rather than assumed.
    pub fn kernel(self: Activation) usize {
        return self.up_filter.len;
    }
};

pub const AmpBlock = struct {
    convs1: []Conv1d,
    convs2: []Conv1d,
    /// `2 * convs1.len`, paired as `activations[::2]` with `convs1` and
    /// `activations[1::2]` with `convs2`.
    activations: []Activation,
};

pub const AudioDecoder = struct {
    arena: std.heap.ArenaAllocator,
    /// `dec_in_proj`, a 1x1 Conv1d from the latent width to the vocoder's.
    dec_in: Conv1d,
    conv_pre: Conv1d,
    /// One transposed conv per upsampling stage.
    ups: []ConvT1d,
    /// `n_stages * n_kernels`, stage-major.
    resblocks: []AmpBlock,
    n_kernels: usize,
    activation_post: Activation,
    conv_post: Conv1d,
    latents_mean: []f32,
    latents_std: []f32,

    pub fn deinit(self: *AudioDecoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn nStages(self: *const AudioDecoder) usize {
        return self.ups.len;
    }

    /// Audio samples one latent frame decodes to: the product of the stage
    /// upsample rates. 800 for the real checkpoint.
    pub fn upsampleFactor(self: *const AudioDecoder) usize {
        var f: usize = 1;
        for (self.ups) |u| f *= u.stride;
        return f;
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !AudioDecoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l: L = .{ .alloc = alloc, .store = store };

        var n_stages: usize = 0;
        while (true) : (n_stages += 1) {
            var buf: [96]u8 = undefined;
            const nm = std.fmt.bufPrint(&buf, "decoder.ups.{d}.0.weight", .{n_stages}) catch break;
            if (store.get(nm) == null) break;
        }
        if (n_stages == 0) return error.MissingTensor;

        var n_res: usize = 0;
        while (true) : (n_res += 1) {
            var buf: [96]u8 = undefined;
            const nm = std.fmt.bufPrint(&buf, "decoder.resblocks.{d}.convs1.0.weight", .{n_res}) catch break;
            if (store.get(nm) == null) break;
        }
        if (n_res == 0 or n_res % n_stages != 0) return error.UnsupportedCheckpoint;
        const n_kernels = n_res / n_stages;

        const ups = try alloc.alloc(ConvT1d, n_stages);
        for (ups, 0..) |*u, i| {
            const w = try l.tensor("decoder.ups.{d}.0.weight", .{i});
            const s = w.info.shape.slice();
            if (s.len != 3) return error.ShapeMismatch;
            // ConvTranspose1d weights are [in][out/groups][k], in-major.
            const in_ch = s[0];
            const out_ch = s[1];
            const k = s[2];
            // `stride` (the upsample rate) is not a stored value; it is recovered
            // from the reference's `padding=(k-u)//2` relation. Every stage here
            // has k = 2u except the first two (k = 9, u = 5), so solve directly:
            // the reference builds `upsample_kernel_sizes` alongside its rates and
            // both are architecture constants. See `upsampleRateFor`.
            u.* = .{
                .w = try l.vecOf(w, in_ch * out_ch * k),
                .b = try l.vec("decoder.ups.{d}.0.bias", .{i}, out_ch),
                .in_ch = in_ch,
                .out_ch = out_ch,
                .k = k,
                .stride = upsampleRateFor(k),
                .padding = (k - upsampleRateFor(k)) / 2,
            };
        }

        const resblocks = try alloc.alloc(AmpBlock, n_res);
        for (resblocks, 0..) |*rb, i| {
            var n_conv: usize = 0;
            while (true) : (n_conv += 1) {
                var buf: [96]u8 = undefined;
                const nm = std.fmt.bufPrint(&buf, "decoder.resblocks.{d}.convs1.{d}.weight", .{ i, n_conv }) catch break;
                if (store.get(nm) == null) break;
            }
            rb.convs1 = try alloc.alloc(Conv1d, n_conv);
            rb.convs2 = try alloc.alloc(Conv1d, n_conv);
            for (0..n_conv) |j| {
                rb.convs1[j] = try l.conv("decoder.resblocks.{d}.convs1.{d}", .{ i, j }, true);
                // `convs1` is DILATED, cycling (1, 3, 5); `convs2` is not. Neither
                // the dilation nor the padding is stored, and they are coupled:
                // the padding must be `get_padding(k, d)` or the block stops being
                // length-preserving and the residual add fails on a shape.
                rb.convs1[j].dilation = dilationFor(j);
                rb.convs1[j].padding = samePadding(rb.convs1[j].k, rb.convs1[j].dilation);
                rb.convs2[j] = try l.conv("decoder.resblocks.{d}.convs2.{d}", .{ i, j }, true);
            }
            rb.activations = try alloc.alloc(Activation, 2 * n_conv);
            for (rb.activations, 0..) |*act, j| {
                act.* = try l.activation("decoder.resblocks.{d}.activations.{d}", .{ i, j });
            }
        }

        const dec_in = try l.conv("dec_in_proj", .{}, true);
        const conv_pre = try l.conv("decoder.conv_pre", .{}, true);
        const conv_post = try l.conv("decoder.conv_post", .{}, false);
        const activation_post = try l.activation("decoder.activation_post", .{});
        const latents_mean = try l.vec("latents_mean", .{}, latent_channels);
        const latents_std = try l.vec("latents_std", .{}, latent_channels);

        return .{
            .arena = arena,
            .dec_in = dec_in,
            .conv_pre = conv_pre,
            .ups = ups,
            .resblocks = resblocks,
            .n_kernels = n_kernels,
            .activation_post = activation_post,
            .conv_post = conv_post,
            .latents_mean = latents_mean,
            .latents_std = latents_std,
        };
    }
};

/// `get_padding(k, d)` from the reference: the padding that keeps a stride-1
/// dilated conv length-preserving.
pub fn samePadding(k: usize, dilation: usize) usize {
    return (k * dilation - dilation) / 2;
}

/// The dilation of `convs1[j]` in an `AMPBlock1`. The reference's
/// `resblock_dilation_sizes` is `((1,3,5), (1,3,5), (1,3,5))`, i.e. the same
/// cycle for every kernel size, and it is not stored anywhere.
pub fn dilationFor(j: usize) usize {
    const cycle = [3]usize{ 1, 3, 5 };
    return cycle[j % cycle.len];
}

/// The upsample rate a stage's kernel size implies.
///
/// The reference pairs `upsample_rates=(5,5,2,2,2,2,2)` with
/// `upsample_kernel_sizes=(9,9,4,4,4,4,4)`, i.e. `k = 2u` except where `u = 5`,
/// which takes `k = 9`. Neither is stored, and the pairing is what makes each
/// stage's output length exactly `u` times its input, so it is recovered from
/// the kernel rather than assumed positionally.
pub fn upsampleRateFor(k: usize) usize {
    return if (k == 9) 5 else k / 2;
}

const L = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,

    fn tensor(l: L, comptime fmt: []const u8, args: anytype) !weights_mod.TensorView {
        var buf: [128]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        return l.store.get(nm) orelse {
            std.log.err("minimax_h3_audio: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
    }

    fn vecOf(l: L, view: weights_mod.TensorView, len: usize) ![]f32 {
        if (view.info.elemCount() != len) return error.ShapeMismatch;
        return view.toF32Alloc(l.alloc);
    }

    fn vec(l: L, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        return l.vecOf(try l.tensor(fmt, args), len);
    }

    /// A Conv1d from `<base>.weight` (+ `<base>.bias`). Padding is `same` for
    /// every conv in this decoder: `(k * dilation - dilation) / 2`, which the
    /// reference spells `get_padding`.
    fn conv(l: L, comptime fmt: []const u8, args: anytype, has_bias: bool) !Conv1d {
        const w = try l.tensor(fmt ++ ".weight", args);
        const s = w.info.shape.slice();
        if (s.len != 3) return error.ShapeMismatch;
        const out_ch = s[0];
        const in_ch = s[1];
        const k = s[2];
        // Dilation is not stored either. Every dilated conv here is a `convs1`
        // whose dilation cycles (1, 3, 5); a `convs2` and the plain convs are
        // dilation 1. Recovering it from the name is fragile, so the caller
        // patches it (see `dilationFor`), and the default is the safe 1.
        return .{
            .w = try l.vecOf(w, out_ch * in_ch * k),
            .b = if (has_bias) try l.vec(fmt ++ ".bias", args, out_ch) else null,
            .out_ch = out_ch,
            .in_ch = in_ch,
            .k = k,
            .padding = samePadding(k, 1),
        };
    }

    fn activation(l: L, comptime fmt: []const u8, args: anytype) !Activation {
        const a = try l.tensor(fmt ++ ".act.alpha", args);
        const channels = a.info.elemCount();
        return .{
            .log_alpha = try l.vecOf(a, channels),
            .log_beta = try l.vec(fmt ++ ".act.beta", args, channels),
            .up_filter = try l.vec(fmt ++ ".upsample.filter", args, 12),
            .down_filter = try l.vec(fmt ++ ".downsample.lowpass.filter", args, 12),
            .channels = channels,
        };
    }
};

// --- forward --------------------------------------------------------------

/// A planar `[channels][len]` signal that grows through the network. Each op
/// allocates its own output from the arena; lengths change at every stage, so
/// reusing one buffer would need the maximum up front and is not worth it for a
/// reference implementation.
const Sig = struct { d: []f32, ch: usize, len: usize };

/// Anti-aliased activation: upsample x2 -> snake -> downsample x2.
///
/// Both halves REPLICATE-pad before convolving, and the upsample slices its
/// result back to exactly `2 * len`. Getting the padding constants wrong shifts
/// the whole signal by a sample or two, which is inaudible in a spectrum and
/// wrong everywhere.
fn antiAliasedSnake(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, x: Sig, act: *const Activation) !Sig {
    const c = x.ch;
    const k = act.kernel();
    std.debug.assert(act.channels == c);

    // --- upsample x2 -------------------------------------------------------
    // pad = k / ratio - 1, then a stride-2 transposed conv, then slice off
    // pad * ratio + (k - ratio) / 2 on the left and the ceil half on the right.
    const ratio = 2;
    const pad = k / ratio - 1;
    const pad_left = pad * ratio + (k - ratio) / 2;
    const pad_right = pad * ratio + (k - ratio + 1) / 2;

    const padded_len = x.len + 2 * pad;
    const padded = try a.alloc(f32, c * padded_len);
    padReplicate(padded, x.d, c, x.len, pad, pad);

    const up_w = try a.alloc(f32, c * k);
    for (0..c) |i| @memcpy(up_w[i * k ..][0..k], act.up_filter);
    const upc: ConvT1d = .{ .w = up_w, .b = null, .in_ch = c, .out_ch = c, .k = k, .stride = ratio, .groups = c };
    const raw_len = upc.outLen(padded_len);
    const raw = try a.alloc(f32, c * raw_len);
    convT1d(raw, padded, upc, padded_len);

    const up_len = raw_len - pad_left - pad_right;
    std.debug.assert(up_len == x.len * ratio);
    const up = try a.alloc(f32, c * up_len);
    for (0..c) |i| {
        @memcpy(up[i * up_len ..][0..up_len], raw[i * raw_len + pad_left ..][0..up_len]);
        // the reference multiplies the transposed conv's output by the ratio
        for (up[i * up_len ..][0..up_len]) |*v| v.* *= @floatFromInt(ratio);
    }

    // --- the activation itself --------------------------------------------
    snakeBeta(up, act.log_alpha, act.log_beta, c, up_len);

    // --- downsample x2 -----------------------------------------------------
    // Asymmetric padding: k/2 - 1 on the left for an even kernel, k/2 right.
    const dl = k / 2 - @intFromBool(k % 2 == 0);
    const dr = k / 2;
    const dpad_len = up_len + dl + dr;
    const dpadded = try a.alloc(f32, c * dpad_len);
    padReplicate(dpadded, up, c, up_len, dl, dr);

    const down_w = try a.alloc(f32, c * k);
    for (0..c) |i| @memcpy(down_w[i * k ..][0..k], act.down_filter);
    const dc: Conv1d = .{ .w = down_w, .b = null, .out_ch = c, .in_ch = c, .k = k, .stride = ratio, .groups = c };
    const out_len = dc.outLen(dpad_len);
    std.debug.assert(out_len == x.len);
    const out = try a.alloc(f32, c * out_len);
    try conv1d(a, io, gpa, out, dpadded, dc, dpad_len);
    return .{ .d = out, .ch = c, .len = out_len };
}

fn runConv(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, x: Sig, c: Conv1d) !Sig {
    const out_len = c.outLen(x.len);
    const out = try a.alloc(f32, c.out_ch * out_len);
    try conv1d(a, io, gpa, out, x.d, c, x.len);
    return .{ .d = out, .ch = c.out_ch, .len = out_len };
}

/// One `AMPBlock1`: three (dilated conv, plain conv) pairs, each preceded by its
/// own anti-aliased activation, accumulated as residuals.
fn runAmpBlock(a: std.mem.Allocator, io: std.Io, gpa: std.mem.Allocator, x: Sig, rb: *const AmpBlock) !Sig {
    var cur = x;
    for (rb.convs1, rb.convs2, 0..) |c1, c2, i| {
        // activations[::2] pairs with convs1, activations[1::2] with convs2:
        // six distinct activations, not three reused.
        var t = try antiAliasedSnake(a, io, gpa, cur, &rb.activations[2 * i]);
        t = try runConv(a, io, gpa, t, c1);
        t = try antiAliasedSnake(a, io, gpa, t, &rb.activations[2 * i + 1]);
        t = try runConv(a, io, gpa, t, c2);
        std.debug.assert(t.len == cur.len and t.ch == cur.ch);
        for (t.d, cur.d) |*o, v| o.* += v;
        cur = t;
    }
    return cur;
}

/// Decode normalized stereo latents `[32][2][t]` (planar) to interleaved samples
/// in [-1, 1], `[len][2]`.
///
/// The two stereo channels go through the SAME mono vocoder independently; the
/// interleave happens only at the end, because that is what a container wants.
pub fn decode(
    dec: *const AudioDecoder,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
) !void {
    const c_lat = dec.dec_in.in_ch;
    const samples = t * dec.upsampleFactor();
    std.debug.assert(z.len == c_lat * stereo * t);
    std.debug.assert(out.len == samples * stereo);

    for (0..stereo) |s| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Denormalize this stereo channel's latent into planar [c][t].
        const lat = try a.alloc(f32, c_lat * t);
        for (0..c_lat) |c| {
            for (0..t) |i| {
                lat[c * t + i] = z[(c * stereo + s) * t + i] * dec.latents_std[c] + dec.latents_mean[c];
            }
        }

        var cur: Sig = .{ .d = lat, .ch = c_lat, .len = t };
        cur = try runConv(a, io, gpa, cur, dec.dec_in);
        cur = try runConv(a, io, gpa, cur, dec.conv_pre);

        for (dec.ups, 0..) |u, i| {
            const up_len = u.outLen(cur.len);
            const up = try a.alloc(f32, u.out_ch * up_len);
            convT1d(up, cur.d, u, cur.len);
            cur = .{ .d = up, .ch = u.out_ch, .len = up_len };

            // The stage's resblocks are SUMMED and averaged, not chained.
            var acc: ?Sig = null;
            for (0..dec.n_kernels) |j| {
                const r = try runAmpBlock(a, io, gpa, cur, &dec.resblocks[i * dec.n_kernels + j]);
                if (acc) |sum| {
                    for (sum.d, r.d) |*o, v| o.* += v;
                } else {
                    acc = .{ .d = try a.dupe(f32, r.d), .ch = r.ch, .len = r.len };
                }
            }
            const sum = acc.?;
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(dec.n_kernels));
            for (sum.d) |*v| v.* *= inv;
            cur = sum;
        }

        cur = try antiAliasedSnake(a, io, gpa, cur, &dec.activation_post);
        cur = try runConv(a, io, gpa, cur, dec.conv_post);
        std.debug.assert(cur.ch == 1);
        std.debug.assert(cur.len == samples);

        // Clamped to [-1, 1], no tanh and no final bias.
        for (0..samples) |i| out[i * stereo + s] = std.math.clamp(cur.d[i], -1.0, 1.0);
    }
}

// --- tests ----------------------------------------------------------------

const audio_fixture = @embedFile("assets/minimax_h3_audio.safetensors");

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

test "conv1d output lengths follow the padding and dilation" {
    // `samePadding` is what makes a dilated stride-1 conv length-preserving, and
    // the residual add in an AMPBlock depends on it exactly.
    for ([_]usize{ 3, 7, 11 }) |k| {
        for ([_]usize{ 1, 3, 5 }) |d| {
            const c: Conv1d = .{
                .w = &.{},
                .b = null,
                .out_ch = 1,
                .in_ch = 1,
                .k = k,
                .dilation = d,
                .padding = samePadding(k, d),
            };
            try std.testing.expectEqual(@as(usize, 40), c.outLen(40));
        }
    }
    // The reference's dilation cycle, which is not stored anywhere.
    try std.testing.expectEqual(@as(usize, 1), dilationFor(0));
    try std.testing.expectEqual(@as(usize, 3), dilationFor(1));
    try std.testing.expectEqual(@as(usize, 5), dilationFor(2));
}

test "transposed conv upsamples by exactly its rate" {
    // `padding = (k - u) / 2` is what makes each stage exactly `u` times longer.
    // The real pairing is rates (5,5,2,2,2,2,2) with kernels (9,9,4,4,4,4,4).
    for ([_][2]usize{ .{ 9, 5 }, .{ 4, 2 } }) |ku| {
        const k = ku[0];
        const u = ku[1];
        try std.testing.expectEqual(u, upsampleRateFor(k));
        const c: ConvT1d = .{
            .w = &.{},
            .b = null,
            .in_ch = 1,
            .out_ch = 1,
            .k = k,
            .stride = u,
            .padding = (k - u) / 2,
        };
        try std.testing.expectEqual(@as(usize, 7 * u), c.outLen(7));
    }
    // ...and the product of the real rates is the samples-per-latent-frame.
    var f: usize = 1;
    for ([_]usize{ 5, 5, 2, 2, 2, 2, 2 }) |r| f *= r;
    try std.testing.expectEqual(@as(usize, samples_per_latent), f);
    try std.testing.expectEqual(@as(usize, sample_rate / 40), samples_per_latent);
}

test "replicate padding extends the edges, not zeros" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 1, 2, 3, 7, 8, 9 }; // two channels of three
    const out = try gpa.alloc(f32, 2 * 7);
    defer gpa.free(out);
    padReplicate(out, &x, 2, 3, 2, 2);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 2, 3, 3, 3 }, out[0..7]);
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 8, 9, 9, 9 }, out[7..14]);
}

test "snake reads alpha and beta in log scale" {
    // x + sin(exp(a) x)^2 / exp(b). Using a and b raw is finite and wrong, which
    // is why the fixture's generator measures that specific substitution.
    var x = [_]f32{ 0.5, -0.5 };
    const la = [_]f32{0.0}; // exp(0) = 1
    const lb = [_]f32{0.0};
    snakeBeta(&x, &la, &lb, 1, 2);
    const s = @sin(@as(f32, 0.5));
    try std.testing.expectApproxEqAbs(0.5 + s * s, x[0], 1e-6);
    // sin is odd and squared, so the correction is symmetric while x is not
    try std.testing.expectApproxEqAbs(-0.5 + s * s, x[1], 1e-6);
}

test "the audio decode matches the reference at a toy width" {
    // ComfyUI's own BigVGAN at a toy width, from tools/gen_minimax_h3_audio.py.
    // Two stages so BOTH kernel/rate pairings run: (9, 5), the odd one, and
    // (4, 2). The kaiser-sinc filters are the real 12-tap ones, loaded from the
    // fixture exactly as they are from the real checkpoint.
    const gpa = std.testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, audio_fixture);
    defer st.deinit();
    var dec = try AudioDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    try std.testing.expectEqual(@as(usize, 2), dec.nStages());
    try std.testing.expectEqual(@as(usize, 3), dec.n_kernels);
    try std.testing.expectEqual(@as(usize, 10), dec.upsampleFactor());
    try std.testing.expectEqual(@as(usize, 5), dec.ups[0].stride);
    try std.testing.expectEqual(@as(usize, 2), dec.ups[1].stride);
    // six activations per block, paired [::2] with convs1 and [1::2] with convs2
    try std.testing.expectEqual(@as(usize, 6), dec.resblocks[0].activations.len);
    try std.testing.expectEqual(@as(usize, 3), dec.resblocks[0].convs1.len);
    try std.testing.expectEqual(@as(usize, 5), dec.resblocks[0].convs1[2].dilation);
    try std.testing.expectEqual(@as(usize, 1), dec.resblocks[0].convs2[2].dilation);

    const t: usize = 3;
    const zv = try st.require("in.z");
    const z = try zv.toF32Alloc(gpa);
    defer gpa.free(z);
    const wantv = try st.require("out.audio");
    const want = try wantv.toF32Alloc(gpa);
    defer gpa.free(want);

    const samples = t * dec.upsampleFactor();
    const out = try gpa.alloc(f32, samples * stereo);
    defer gpa.free(out);
    try decode(&dec, std.testing.io, gpa, out, z, t);

    // The reference emits [B, stereo, L] (channel-major); `decode` interleaves,
    // because that is what a container wants. Compare in the reference's layout.
    try std.testing.expectEqual(want.len, out.len);
    const planar = try gpa.alloc(f32, out.len);
    defer gpa.free(planar);
    for (0..stereo) |s| {
        for (0..samples) |i| planar[s * samples + i] = out[i * stereo + s];
    }
    const err = relL2(want, planar);
    errdefer std.debug.print("audio decode rel L2 {e}\n", .{err});
    try std.testing.expect(err < 1e-5);

    // The two channels must genuinely differ: they share one vocoder, so
    // decoding one and copying it would pass a same-shape check.
    const ch_err = relL2(want[0..samples], planar[samples..][0..samples]);
    try std.testing.expect(ch_err > 0.1);
    for (out) |v| try std.testing.expect(v >= -1.0 and v <= 1.0);
}

const test_gate = @import("../test_gate.zig");
const real_audio_vae = "/home/qt/genai/comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors";

test "the real audio VAE loads and its rates multiply to 800" {
    // Pins the values that are NOT stored -- the upsample rates recovered from
    // the kernel sizes, and the dilation cycle -- against the real file.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, real_audio_vae);

    var st = try tp_core.safetensors.SafeTensors.open(gpa, io, real_audio_vae);
    defer st.deinit();
    var dec = try AudioDecoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();

    try std.testing.expectEqual(@as(usize, 7), dec.nStages());
    try std.testing.expectEqual(@as(usize, 3), dec.n_kernels);
    // The recovered rates must give exactly 800 samples per latent frame, i.e.
    // 32 kHz at 40 latent fps. Any mis-recovered rate changes the audio LENGTH.
    try std.testing.expectEqual(@as(usize, samples_per_latent), dec.upsampleFactor());
    try std.testing.expectEqual(@as(usize, 5), dec.ups[0].stride);
    try std.testing.expectEqual(@as(usize, 5), dec.ups[1].stride);
    try std.testing.expectEqual(@as(usize, 2), dec.ups[2].stride);
    // dec_in_proj widens the 32-channel latent to the vocoder's 2048.
    try std.testing.expectEqual(@as(usize, latent_channels), dec.dec_in.in_ch);
    try std.testing.expectEqual(@as(usize, 2048), dec.dec_in.out_ch);
    try std.testing.expectEqual(@as(usize, 1), dec.conv_post.out_ch);
    // the 12-tap kaiser filters really are in the file
    try std.testing.expectEqual(@as(usize, 12), dec.activation_post.kernel());
}
