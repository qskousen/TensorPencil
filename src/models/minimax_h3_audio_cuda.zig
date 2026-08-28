//! CUDA-backend MiniMax H3 audio VAE decode, the device twin of
//! `minimax_h3_audio.decode`.
//!
//! With the DiT trunk and the video VAE on the device this was the last CPU
//! stage of a clip render, and the largest: ~9 s against ~7 s of video decode and
//! ~5 s of sampling for a 22-frame 512x512 clip.
//!
//! Three things differ from the CPU reference, and each is why this is a port
//! rather than a transcription:
//!
//! 1. **Signals are CHANNEL-LAST `[len][ch]`**, not the reference's planar
//!    `[ch][len]`. Every kernel here then has a warp covering consecutive
//!    channels at one time step, which is consecutive memory, and the GEMM's
//!    output is already the next op's input layout. The CPU path pays a transpose
//!    on both sides of every conv for exactly this reason.
//! 2. **The conv weights are permuted at session build.** An ungrouped conv is an
//!    im2col GEMM, and the patch matrix's column order has to match the weight's.
//!    Channel-last makes `(tap, in_ch)` the coalesced order, so the weight goes
//!    `[out_ch][in_ch][k]` -> `[out_ch][k][in_ch]` (and to f16, for the
//!    tensor-core GEMM). The transposed convs go `[in_ch][out_ch][k]` ->
//!    `[k][in_ch][out_ch]`, which is what makes their gather coalesced.
//! 3. **The anti-aliased activation is two kernels, not six ops.** The reference
//!    is pad -> transposed conv -> slice -> scale -> snake -> pad -> conv. Read as
//!    a gather, the padding is an index clamp and the slice is an index offset,
//!    so the whole first half collapses into `aa_up_snake` and the second into
//!    `aa_down`, with one `2 * len` intermediate and nothing else.
//!
//! ⚠️ **The GEMM runs the activations through f16.** That is the same regime the
//! video VAE's port runs in, but audio is less forgiving of it than an image, so
//! `minimax-h3-audio-cuda-test` reports the relative L2 against the f32 host
//! decode and a per-sample maximum, not just a norm. The weights are f16 rather
//! than bf16 deliberately: 11 mantissa bits against 8, and the range here is tiny
//! (fan-in-normalized conv weights), so precision is the only axis that matters.

const std = @import("std");
const audio = @import("minimax_h3_audio.zig");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const AudioDecoder = audio.AudioDecoder;
const Conv1d = audio.Conv1d;
const ConvT1d = audio.ConvT1d;
const Activation = audio.Activation;

/// DIAGNOSTIC: run the conv GEMMs on tensor cores through f16 instead of in f32.
///
/// Not the default, and the reason is measured. Against the f32 CPU reference on
/// the real vocoder, the f16 route is 2.4e-3 relative with a 8.3e-3 worst sample
/// (about -42 dB, an audible noise floor with 74% of samples past 1e-4), while
/// f32 is 9.5e-6 / 2.8e-5, under 16-bit PCM's own quantum. It is roughly 2x
/// faster, and this stage is no longer the bottleneck, so the trade is not worth
/// taking. Kept as a flag because it is also the isolation that proved the gap
/// was precision and not a defect in the kernels here.
pub var f16_gemm: bool = false;

/// Whether this decoder's shapes are ones the kernels here cover.
///
/// Refuse by name rather than at a launch: every failure mode below would
/// otherwise surface as a bad device read several stages deep, or worse, as
/// audio that is merely shifted.
pub fn supported(dec: *const AudioDecoder) bool {
    // Grouped ungrouped-conv paths do not exist here. The only grouped
    // convolutions in this decoder are the per-channel kaiser filters, which
    // `aa_up_snake`/`aa_down` handle directly rather than as convolutions.
    if (dec.dec_in.groups != 1 or dec.conv_pre.groups != 1 or dec.conv_post.groups != 1) return false;
    if (dec.dec_in.stride != 1 or dec.conv_pre.stride != 1 or dec.conv_post.stride != 1) return false;
    for (dec.ups) |u| if (u.groups != 1) return false;
    for (dec.resblocks) |rb| {
        for (rb.convs1) |c| if (c.groups != 1 or c.stride != 1) return false;
        for (rb.convs2) |c| if (c.groups != 1 or c.stride != 1) return false;
        for (rb.activations) |a| if (!actSupported(a)) return false;
    }
    if (!actSupported(dec.activation_post)) return false;
    return true;
}

fn actSupported(a: Activation) bool {
    const k = a.kernel();
    // `aa_down`'s asymmetric left pad is `k/2 - 1`, which is the even-kernel
    // case; an odd filter would need `k/2` and the round trip would stop being
    // length-preserving. Every kaiser filter in this family is 12 taps.
    if (k % 2 != 0 or k < 2) return false;
    if (a.down_filter.len != k) return false;
    return true;
}

// --- session --------------------------------------------------------------

/// A conv ready for the device: the permuted f16 weight plus the shape the
/// im2col and the GEMM need. Host memory, uploaded on first use through the
/// backend's pointer-keyed weight cache.
const DevConv = struct {
    /// `[out_ch][k * in_ch]` f16.
    w16: []const u8,
    /// The same weight in f32, for the `f32_gemm` diagnostic only.
    w32: []const f32,
    bias: ?[]const f32,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    dilation: usize,
    padding: usize,
    /// 1 for every conv in the vocoder; the ENCODER's downsampling convs are the
    /// only strided ones, and they share this type through `minimax_h3_audio_encode`.
    stride: usize = 1,

    fn plen(self: DevConv) usize {
        return self.k * self.in_ch;
    }
};

/// A transposed conv ready for the device: `[k][in_ch][out_ch]` f32, plus a bias
/// that is always present here (the reference's `ups` all carry one).
const DevConvT = struct {
    w: []const f32,
    bias: []const f32,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    padding: usize,
};

/// An activation ready for the device: the two filters and the EXPONENTIATED
/// snake parameters, interleaved `(exp(alpha), 1/(exp(beta) + 1e-9))` per
/// channel. The checkpoint stores them in log scale; exponentiating on the host
/// keeps it out of the inner loop.
const DevAct = struct {
    up_filter: []const f32,
    down_filter: []const f32,
    /// `[2 * channels]`.
    snake: []const f32,
    channels: usize,
    k: usize,

    /// The reference's upsample constants, derived from the filter length: pad
    /// `k/2 - 1` before a stride-2 transposed conv, then slice `pad*2 + (k-2)/2`
    /// off the left. Getting these wrong shifts the signal by a sample or two,
    /// which is inaudible in a spectrum and wrong everywhere.
    fn upPad(self: DevAct) usize {
        return self.k / 2 - 1;
    }
    fn upSlice(self: DevAct) usize {
        return self.upPad() * 2 + (self.k - 2) / 2;
    }
    /// The downsample's ASYMMETRIC left pad.
    fn downPad(self: DevAct) usize {
        return self.k / 2 - 1;
    }
};

/// Per-decoder device state: every weight permuted and converted once.
///
/// Built once per session rather than per decode: the permute reads ~250 MB of
/// f32 conv weights, which is far more than a decode's own traffic.
pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    dec_in: DevConv,
    conv_pre: DevConv,
    conv_post: DevConv,
    ups: []DevConvT,
    /// `n_stages * n_kernels` blocks, stage-major, matching `dec.resblocks`.
    blocks: []DevBlock,
    act_post: DevAct,

    pub const DevBlock = struct {
        convs1: []DevConv,
        convs2: []DevConv,
        acts: []DevAct,
    };

    pub fn init(gpa: std.mem.Allocator, dec: *const AudioDecoder) !Session {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const ups = try a.alloc(DevConvT, dec.ups.len);
        for (ups, dec.ups) |*d, u| d.* = try devConvT(a, u);

        const blocks = try a.alloc(DevBlock, dec.resblocks.len);
        for (blocks, dec.resblocks) |*b, rb| {
            b.convs1 = try a.alloc(DevConv, rb.convs1.len);
            b.convs2 = try a.alloc(DevConv, rb.convs2.len);
            b.acts = try a.alloc(DevAct, rb.activations.len);
            for (b.convs1, rb.convs1) |*d, c| d.* = try devConv(a, c);
            for (b.convs2, rb.convs2) |*d, c| d.* = try devConv(a, c);
            for (b.acts, rb.activations) |*d, act| d.* = try devAct(a, act);
        }

        const dec_in = try devConv(a, dec.dec_in);
        const conv_pre = try devConv(a, dec.conv_pre);
        const conv_post = try devConv(a, dec.conv_post);
        const act_post = try devAct(a, dec.activation_post);

        return .{
            .arena = arena,
            .dec_in = dec_in,
            .conv_pre = conv_pre,
            .conv_post = conv_post,
            .ups = ups,
            .blocks = blocks,
            .act_post = act_post,
        };
    }

    pub fn deinit(self: *Session) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Host bytes the permuted weights hold, which is also what they cost
    /// resident on the device (f16 for the convs, f32 for the transposed ones).
    pub fn bytes(self: *const Session) usize {
        var n: usize = self.dec_in.w16.len + self.conv_pre.w16.len + self.conv_post.w16.len;
        for (self.ups) |u| n += u.w.len * 4;
        for (self.blocks) |b| {
            for (b.convs1) |c| n += c.w16.len;
            for (b.convs2) |c| n += c.w16.len;
        }
        return n;
    }
};

fn f32ToF16(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// `[out_ch][in_ch][k]` f32 -> `[out_ch][k][in_ch]` f16.
///
/// The permutation is what lets the im2col write its columns in the coalesced
/// `(tap, in_ch)` order. Pairing the two the other way is finite and wrong: every
/// tap convolves against the wrong channel.
fn devConv(a: std.mem.Allocator, c: Conv1d) !DevConv {
    std.debug.assert(c.groups == 1 and c.stride == 1);
    const n = c.out_ch * c.in_ch * c.k;
    std.debug.assert(c.w.len == n);
    const w16 = try a.alloc(u16, n);
    const w32 = try a.alloc(f32, n);
    for (0..c.out_ch) |oc| {
        const src = c.w[oc * c.in_ch * c.k ..][0 .. c.in_ch * c.k];
        const d16 = w16[oc * c.in_ch * c.k ..][0 .. c.in_ch * c.k];
        const d32 = w32[oc * c.in_ch * c.k ..][0 .. c.in_ch * c.k];
        for (0..c.in_ch) |ic| {
            for (0..c.k) |j| {
                const v = src[ic * c.k + j];
                d16[j * c.in_ch + ic] = f32ToF16(v);
                d32[j * c.in_ch + ic] = v;
            }
        }
    }
    return .{
        .w16 = std.mem.sliceAsBytes(w16),
        .w32 = w32,
        .bias = c.b,
        .out_ch = c.out_ch,
        .in_ch = c.in_ch,
        .k = c.k,
        .dilation = c.dilation,
        .padding = c.padding,
    };
}

/// `[in_ch][out_ch][k]` f32 -> `[k][in_ch][out_ch]` f32.
///
/// Note the source is IN-channel major; that is PyTorch's transposed-conv
/// convention and reading it the other way is a shape mismatch only when the two
/// channel counts differ, which for this vocoder's halving stages they always do.
fn devConvT(a: std.mem.Allocator, c: ConvT1d) !DevConvT {
    std.debug.assert(c.groups == 1);
    const n = c.in_ch * c.out_ch * c.k;
    std.debug.assert(c.w.len == n);
    const w = try a.alloc(f32, n);
    for (0..c.in_ch) |ic| {
        for (0..c.out_ch) |oc| {
            for (0..c.k) |j| w[(j * c.in_ch + ic) * c.out_ch + oc] = c.w[(ic * c.out_ch + oc) * c.k + j];
        }
    }
    return .{
        .w = w,
        .bias = c.b orelse return error.MissingTensor,
        .in_ch = c.in_ch,
        .out_ch = c.out_ch,
        .k = c.k,
        .stride = c.stride,
        .padding = c.padding,
    };
}

fn devAct(a: std.mem.Allocator, act: Activation) !DevAct {
    const snake = try a.alloc(f32, 2 * act.channels);
    for (0..act.channels) |c| {
        snake[2 * c] = @exp(act.log_alpha[c]);
        // The reference's `+ 1e-9` guard, folded into the reciprocal.
        snake[2 * c + 1] = 1.0 / (@exp(act.log_beta[c]) + 1e-9);
    }
    return .{
        .up_filter = act.up_filter,
        .down_filter = act.down_filter,
        .snake = snake,
        .channels = act.channels,
        .k = act.kernel(),
    };
}

// --- workspace ------------------------------------------------------------

/// The (channels, length) the signal has at each point in the pipeline.
///
/// Walked rather than assumed: the lengths multiply by the stage rates and the
/// channels halve, and the workspace has to be sized from the maximum of the
/// PRODUCT, which is neither the first nor the last stage.
pub const Shapes = struct {
    /// Largest `ch * len` any signal buffer holds.
    sig: usize,
    /// Largest `ch * len` inside an AMPBlock, i.e. what the `2 * len`
    /// anti-aliased intermediate is sized from.
    aa: usize,
    /// Largest `out_len * k * in_ch` im2col patch.
    patch: usize,
    /// Output samples per stereo channel.
    samples: usize,

    pub fn of(dec: *const AudioDecoder, t: usize) Shapes {
        var s: Shapes = .{ .sig = 0, .aa = 0, .patch = 0, .samples = t * dec.upsampleFactor() };
        const bump = struct {
            fn go(cur: *usize, v: usize) void {
                if (v > cur.*) cur.* = v;
            }
        }.go;

        bump(&s.sig, dec.dec_in.in_ch * t);
        bump(&s.sig, dec.dec_in.out_ch * t);
        bump(&s.patch, t * dec.dec_in.k * dec.dec_in.in_ch);
        bump(&s.sig, dec.conv_pre.out_ch * t);
        bump(&s.patch, t * dec.conv_pre.k * dec.conv_pre.in_ch);

        var len = t;
        for (dec.ups, 0..) |u, i| {
            len = u.outLen(len);
            const ch = u.out_ch;
            bump(&s.sig, ch * len);
            bump(&s.aa, ch * len);
            for (0..dec.n_kernels) |j| {
                const rb = &dec.resblocks[i * dec.n_kernels + j];
                for (rb.convs1) |c| bump(&s.patch, len * c.k * c.in_ch);
                for (rb.convs2) |c| bump(&s.patch, len * c.k * c.in_ch);
            }
        }
        bump(&s.sig, dec.conv_post.out_ch * len);
        bump(&s.patch, len * dec.conv_post.k * dec.conv_post.in_ch);
        return s;
    }
};

pub const Workspace = struct {
    /// The stage's upsampled signal, read by all three of its resblocks.
    up: Buf = .{},
    /// The AMPBlock residual chain's ping-pong pair.
    blk0: Buf = .{},
    blk1: Buf = .{},
    /// The resblock sum, which becomes the next stage's input.
    acc: Buf = .{},
    /// The `2 * len` anti-aliased intermediate.
    aa: Buf = .{},
    /// im2col patch matrix.
    patch: Buf = .{},
    shapes: Shapes = undefined,

    pub fn init(be: *Backend, dec: *const AudioDecoder, t: usize) !Workspace {
        const s = Shapes.of(dec, t);
        var ws: Workspace = .{ .shapes = s };
        errdefer ws.deinit(be);
        ws.up = try be.tensorCreate(s.sig * 4);
        ws.blk0 = try be.tensorCreate(s.sig * 4);
        ws.blk1 = try be.tensorCreate(s.sig * 4);
        ws.acc = try be.tensorCreate(s.sig * 4);
        ws.aa = try be.tensorCreate(2 * s.aa * 4);
        ws.patch = try be.tensorCreate(s.patch * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{ &self.up, &self.blk0, &self.blk1, &self.acc, &self.aa, &self.patch }) |b| {
            be.tensorDestroy(b);
        }
        self.* = .{};
    }

    /// Device bytes this shape needs, so a caller can budget before allocating.
    pub fn bytesFor(dec: *const AudioDecoder, t: usize) usize {
        const s = Shapes.of(dec, t);
        return (4 * s.sig + 2 * s.aa + s.patch) * 4;
    }
};

// --- forward --------------------------------------------------------------

fn devBuf(be: *Backend, data: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(data)), .mem = .null_handle, .size = data.len * 4 };
}

/// `dst[out_len][out_ch] = conv(src[in_len][in_ch])`, im2col + tensor-core GEMM.
///
/// `dst` may alias `src`: the GEMM reads only the patch matrix.
fn conv(be: *Backend, ws: *Workspace, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
    const out_len = in_len; // every conv here is stride 1 and same-padded
    std.debug.assert(out_len * c.plen() <= ws.shapes.patch);
    try be.opIm2col1d(src, ws.patch, 0, out_len, c.k, c.in_ch, in_len, c.dilation, c.padding, c.stride);
    if (f16_gemm) {
        try be.opMatmulF16(dst, ws.patch, out_len, c.w16, c.out_ch, c.plen(), c.bias, false, false);
        return out_len;
    }
    const w32 = std.mem.sliceAsBytes(c.w32);
    // The tiled f32 GEMM where there is one; the hand-PTX arm has none, and its
    // one-thread-per-output fallback is correct at about a third of the speed.
    // Falling back to f16 there instead would be a silent 50 dB of SNR.
    be.opMatmulF32Lt(dst, ws.patch, out_len, w32, c.out_ch, c.plen(), c.bias) catch |err| switch (err) {
        error.UnsupportedKernelArm => try be.opMatmul(dst, 0, ws.patch, 0, out_len, w32, false, c.out_ch, c.plen(), 1.0, c.bias),
        else => return err,
    };
    return out_len;
}

/// The anti-aliased activation, in place: `x[len][ch]` -> `x[len][ch]`.
fn aaSnake(be: *Backend, ws: *Workspace, x: Buf, act: DevAct, len: usize) !void {
    std.debug.assert(act.channels * len <= ws.shapes.aa);
    const up = try devBuf(be, act.up_filter);
    const down = try devBuf(be, act.down_filter);
    const snake = try devBuf(be, act.snake);
    try be.opAaUpSnake(x, ws.aa, up, snake, len, act.channels, act.k, act.upPad(), act.upSlice());
    try be.opAaDown(ws.aa, x, down, len, 2 * len, act.channels, act.k, act.downPad());
}

/// One `AMPBlock1` over `src`, which is NOT modified. Returns the buffer holding
/// the result (one of the ping-pong pair).
fn ampBlock(be: *Backend, ws: *Workspace, src: Buf, blk: *const Session.DevBlock, ch: usize, len: usize) !Buf {
    var cur = src;
    var dst = ws.blk0;
    var other = ws.blk1;
    for (blk.convs1, blk.convs2, 0..) |c1, c2, i| {
        // activations[::2] pairs with convs1 and [1::2] with convs2: six
        // distinct activations per block, not three reused.
        try be.opAaUpSnake(cur, ws.aa, try devBuf(be, blk.acts[2 * i].up_filter), try devBuf(be, blk.acts[2 * i].snake), len, ch, blk.acts[2 * i].k, blk.acts[2 * i].upPad(), blk.acts[2 * i].upSlice());
        try be.opAaDown(ws.aa, dst, try devBuf(be, blk.acts[2 * i].down_filter), len, 2 * len, ch, blk.acts[2 * i].k, blk.acts[2 * i].downPad());
        _ = try conv(be, ws, dst, dst, c1, len);
        try aaSnake(be, ws, dst, blk.acts[2 * i + 1], len);
        _ = try conv(be, ws, dst, dst, c2, len);
        try be.opAdd(dst, cur, ch * len);
        cur = dst;
        // Ping-pong, so the next iteration's residual source stays intact.
        const t = dst;
        dst = other;
        other = t;
    }
    return cur;
}

/// Decode normalized stereo latents `[32][2][t]` (planar, the sampler's layout)
/// to interleaved samples in [-1, 1], `[len][2]`.
///
/// Same signature as `minimax_h3_audio.decode` so the two are drop-in
/// alternatives, and `minimax-h3-audio-cuda-test` compares them directly.
pub fn decode(
    dec: *const AudioDecoder,
    sess: *const Session,
    be: *Backend,
    ws: *Workspace,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const c_lat = dec.dec_in.in_ch;
    const samples = ws.shapes.samples;
    std.debug.assert(z.len == c_lat * audio.stereo * t);
    std.debug.assert(out.len == samples * audio.stereo);
    std.debug.assert(samples == t * dec.upsampleFactor());

    // Channel-last from here on, so the denormalize doubles as the transpose.
    const lat = try gpa.alloc(f32, t * c_lat);
    defer gpa.free(lat);
    const tail = try gpa.alloc(f32, samples);
    defer gpa.free(tail);

    for (0..audio.stereo) |s| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;

        for (0..t) |i| {
            for (0..c_lat) |c| {
                lat[i * c_lat + c] = z[(c * audio.stereo + s) * t + i] * dec.latents_std[c] + dec.latents_mean[c];
            }
        }

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try be.tensorUpload(ws.blk0, std.mem.sliceAsBytes(lat));
        _ = try conv(be, ws, ws.blk1, ws.blk0, sess.dec_in, t);
        _ = try conv(be, ws, ws.acc, ws.blk1, sess.conv_pre, t);

        var len = t;
        for (sess.ups, 0..) |u, i| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            const up_len = (len - 1) * u.stride + u.k - 2 * u.padding;
            try be.opConvT1dCa(
                ws.acc,
                ws.up,
                try devBuf(be, u.w),
                try devBuf(be, u.bias),
                up_len,
                u.out_ch,
                u.in_ch,
                len,
                u.k,
                u.stride,
                u.padding,
            );
            len = up_len;

            // The stage's resblocks are SUMMED and averaged, not chained, so each
            // reads the SAME `up` and the sum lands in `acc`.
            for (0..dec.n_kernels) |j| {
                const r = try ampBlock(be, ws, ws.up, &sess.blocks[i * dec.n_kernels + j], u.out_ch, len);
                if (j == 0) {
                    try be.opCopyOff(ws.acc, 0, r, 0, u.out_ch * len, false);
                } else {
                    try be.opAdd(ws.acc, r, u.out_ch * len);
                }
            }
            try be.opScale(ws.acc, 1.0 / @as(f32, @floatFromInt(dec.n_kernels)), u.out_ch * len);
        }

        try aaSnake(be, ws, ws.acc, sess.act_post, len);
        const post_len = try conv(be, ws, ws.blk0, ws.acc, sess.conv_post, len);
        std.debug.assert(sess.conv_post.out_ch == 1 and post_len == samples);
        try be.endBatch();

        try be.tensorDownload(ws.blk0, std.mem.sliceAsBytes(tail));
        // Clamped to [-1, 1], no tanh and no final bias. The interleave happens
        // only here, because that is what a container wants.
        for (0..samples) |i| out[i * audio.stereo + s] = std.math.clamp(tail[i], -1.0, 1.0);
    }
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

test "the pipeline shape walk peaks in the middle, not at either end" {
    // The workspace is sized from the maximum of `ch * len`, and both factors
    // move in opposite directions across the stages: channels halve while the
    // length multiplies by the stage rate. Sizing from the first or the last
    // stage under-allocates whichever is bigger, which is a device write past
    // the end of a buffer.
    //
    // Modelled on the real decoder: 7 stages, rates (5,5,2,2,2,2,2), channels
    // 1024 -> 512 -> ... -> 8. Halving the channels while doubling the length
    // holds `ch * len` FLAT, so stages 1 through 6 are all 6400T; only the two
    // rate-5 stages break the pattern, and stage 0 is 2560T. Sizing the workspace
    // from the first stage therefore under-allocates by 2.5x.
    const t: usize = 37;
    const rates = [_]usize{ 5, 5, 2, 2, 2, 2, 2 };
    var len = t;
    var ch: usize = 1024;
    var peak: usize = 0;
    var first: usize = 0;
    var last: usize = 0;
    for (rates, 0..) |r, i| {
        len *= r;
        ch /= 2;
        const prod = ch * len;
        if (i == 0) first = prod;
        if (i == rates.len - 1) last = prod;
        peak = @max(peak, prod);
    }
    try testing.expectEqual(@as(usize, 512 * 5 * t), first);
    try testing.expectEqual(@as(usize, 8 * 800 * t), last);
    try testing.expectEqual(@as(usize, 6400 * t), peak);
    try testing.expectEqual(peak, last);
    try testing.expectEqual(@as(usize, 2560 * t), first);
    try testing.expect(peak == 2 * first + first / 2); // 2.5x
}

test "the anti-aliased activation's constants keep the round trip length-preserving" {
    // These four numbers are the whole reason the up/down pair composes: the
    // upsample's slice has to leave exactly `2 * len`, and the downsample's
    // ASYMMETRIC pad has to bring it back to `len`. A symmetric guess shifts the
    // signal by a sample, which is inaudible in a spectrum and wrong everywhere.
    const filt = [_]f32{0} ** 12;
    const snake = [_]f32{0} ** 4;
    const act: DevAct = .{
        .up_filter = &filt,
        .down_filter = &filt,
        .snake = &snake,
        .channels = 2,
        .k = 12,
    };
    try testing.expectEqual(@as(usize, 5), act.upPad());
    try testing.expectEqual(@as(usize, 15), act.upSlice());
    try testing.expectEqual(@as(usize, 5), act.downPad());

    // The upsample: pad, stride-2 transposed conv, slice. What survives must be
    // exactly 2 * len.
    for ([_]usize{ 1, 7, 40, 185 }) |len| {
        const padded = len + 2 * act.upPad();
        const raw = (padded - 1) * 2 + act.k;
        const right = act.upPad() * 2 + (act.k - 2 + 1) / 2;
        try testing.expectEqual(2 * len, raw - act.upSlice() - right);
        // ...and the downsample brings 2*len back to len.
        const dpad = 2 * len + act.downPad() + act.k / 2;
        try testing.expectEqual(len, (dpad - act.k) / 2 + 1);
    }
}

test "the conv weight permutation matches the im2col column order" {
    // `[out_ch][in_ch][k]` -> `[out_ch][k][in_ch]`, because channel-last im2col
    // writes its columns `(tap, in_ch)`. Pairing the two the other way convolves
    // every tap against the wrong channel: finite, plausible, wrong.
    const out_ch: usize = 2;
    const in_ch: usize = 3;
    const k: usize = 2;
    var w: [out_ch * in_ch * k]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @floatFromInt(i);
    const c: Conv1d = .{ .w = &w, .b = null, .out_ch = out_ch, .in_ch = in_ch, .k = k };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const d = try devConv(arena.allocator(), c);
    const got = std.mem.bytesAsSlice(u16, d.w16);
    try testing.expectEqual(@as(usize, out_ch * in_ch * k), got.len);
    // Row 0 source is [ic0k0, ic0k1, ic1k0, ic1k1, ic2k0, ic2k1] = 0..5;
    // permuted it is [ic0k0, ic1k0, ic2k0, ic0k1, ic1k1, ic2k1] = 0, 2, 4, 1, 3, 5.
    const want0 = [_]f32{ 0, 2, 4, 1, 3, 5 };
    for (want0, 0..) |v, i| {
        try testing.expectEqual(v, @as(f32, @floatCast(@as(f16, @bitCast(got[i])))));
    }
    // ...and row 1 is the same pattern offset by the row stride, not a repeat of
    // row 0 (which is what a permutation that dropped `oc` would give).
    try testing.expectEqual(@as(f32, 6), @as(f32, @floatCast(@as(f16, @bitCast(got[6])))));
    try testing.expectEqual(@as(f32, 8), @as(f32, @floatCast(@as(f16, @bitCast(got[7])))));
    try testing.expectEqual(@as(usize, k * in_ch), d.plen());
}

test "the transposed conv weight permutation is in-major to tap-major" {
    // PyTorch stores `[in_ch][out_ch][k]`; the gather kernel wants
    // `[k][in_ch][out_ch]` so consecutive threads (consecutive out_ch) read
    // consecutive weights. The two channel counts differ at every real stage, so
    // reading the source out-major is a shape mismatch rather than a silent
    // transpose -- but the PERMUTATION being wrong is silent.
    const in_ch: usize = 2;
    const out_ch: usize = 3;
    const k: usize = 2;
    var w: [in_ch * out_ch * k]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @floatFromInt(i);
    const bias = [_]f32{ 0, 0, 0 };
    const c: ConvT1d = .{ .w = &w, .b = &bias, .in_ch = in_ch, .out_ch = out_ch, .k = k, .stride = 2, .padding = 0 };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const d = try devConvT(arena.allocator(), c);
    // src[(ic * out_ch + oc) * k + j] -> dst[(j * in_ch + ic) * out_ch + oc]
    for (0..in_ch) |ic| {
        for (0..out_ch) |oc| {
            for (0..k) |j| {
                try testing.expectEqual(
                    w[(ic * out_ch + oc) * k + j],
                    d.w[(j * in_ch + ic) * out_ch + oc],
                );
            }
        }
    }
}

test "snake parameters are exponentiated once, on the host" {
    // The checkpoint stores alpha and beta in LOG scale and the kernel wants
    // them linear. Doing it here keeps `exp` out of the inner loop, and folding
    // the reference's 1e-9 guard into the reciprocal keeps the kernel to one
    // multiply.
    const la = [_]f32{ 0.0, 1.0 };
    const lb = [_]f32{ 0.0, -1.0 };
    const filt = [_]f32{0} ** 12;
    const act: Activation = .{
        .log_alpha = @constCast(&la),
        .log_beta = @constCast(&lb),
        .up_filter = @constCast(&filt),
        .down_filter = @constCast(&filt),
        .channels = 2,
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const d = try devAct(arena.allocator(), act);
    try testing.expectApproxEqAbs(@as(f32, 1.0), d.snake[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), d.snake[1], 1e-6);
    try testing.expectApproxEqAbs(@exp(@as(f32, 1.0)), d.snake[2], 1e-5);
    try testing.expectApproxEqAbs(1.0 / @exp(@as(f32, -1.0)), d.snake[3], 1e-5);
}
