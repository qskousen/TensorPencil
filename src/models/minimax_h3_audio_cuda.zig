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
const conv_cuda = @import("conv_cuda.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const minimax_h3_audio_dev = @import("minimax_h3_audio_dev.zig");

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

// The device-prep types (`Session`, `DevConv`, `DevConvT`, `DevAct`, `Shapes`)
// and the shape gate live in `minimax_h3_audio.zig`: none of them touch a
// backend, and both device arms build the same ones. Re-exported so a caller
// naming this module keeps working.
pub const supported = audio.deviceSupported;
pub const Session = audio.DevSession;
pub const Shapes = audio.Shapes;
const DevConv = audio.DevConv;
const DevConvT = audio.DevConvT;
const DevAct = audio.DevAct;
const devConv = audio.devConv;
const devConvT = audio.devConvT;
const devAct = audio.devAct;

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

/// `dst[out_len][out_ch] = conv(src[in_len][in_ch])`; `dst` may alias `src` (the
/// patch holds the whole input, one band).
fn convAt(be: *Backend, ws: *Workspace, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
    const plan: lin_cuda.Plan = if (f16_gemm) lin_cuda.Plan.tower else f32_plan;
    return conv_cuda.conv1d(be, plan, ws.patch, ws.shapes.patch, dst, src, c, in_len);
}

/// Every conv at its stored f32 (see `f16_gemm`).
const f32_plan: lin_cuda.Plan = blk: {
    var p = lin_cuda.Plan.tower;
    p.dense = .native;
    break :blk p;
};

/// Same signature as `minimax_h3_audio.decode` so the two are drop-in alternatives,
/// and `minimax-h3-audio-cuda-test` compares them directly.
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
    try minimax_h3_audio_dev.decode(Dev{ .be = be, .ws = ws }, dec, sess, gpa, out, z, t, cancel);
}

/// The `minimax_h3_audio_dev` adapter.
const Dev = struct {
    be: *Backend,
    ws: *Workspace,

    pub fn beginBatch(d: Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn upload(d: Dev, b: Buf, data: []const f32) !void {
        try d.be.tensorUpload(b, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: Buf, data: []f32) !void {
        try d.be.tensorDownload(b, std.mem.sliceAsBytes(data));
    }

    pub fn conv(d: Dev, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
        return convAt(d.be, d.ws, dst, src, c, in_len);
    }

    pub fn convT(d: Dev, dst: Buf, src: Buf, u: DevConvT, len: usize, up_len: usize) !void {
        try d.be.opConvT1dCa(src, dst, try devBuf(d.be, u.w), try devBuf(d.be, u.bias), up_len, u.out_ch, u.in_ch, len, u.k, u.stride, u.padding);
    }

    pub fn aaUp(d: Dev, src: Buf, act: DevAct, len: usize) !void {
        try d.be.opAaUpSnake(src, d.ws.aa, try devBuf(d.be, act.up_filter), try devBuf(d.be, act.snake), len, act.channels, act.k, act.upPad(), act.upSlice());
    }

    pub fn aaDown(d: Dev, dst: Buf, act: DevAct, len: usize) !void {
        try d.be.opAaDown(d.ws.aa, dst, try devBuf(d.be, act.down_filter), len, 2 * len, act.channels, act.k, act.downPad());
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn copy(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        try d.be.opCopyOff(dst, 0, src, 0, n);
    }

    pub fn scale(d: Dev, x: Buf, f: f32, n: usize) !void {
        try d.be.opScale(x, f, n);
    }
};

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
    const filt: [12]f32 = @splat(0);
    const snake: [4]f32 = @splat(0);
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
    const got = std.mem.bytesAsSlice(f32, d.w.bytes);
    try testing.expectEqual(@as(usize, out_ch * in_ch * k), got.len);
    // Row 0 source is [ic0k0, ic0k1, ic1k0, ic1k1, ic2k0, ic2k1] = 0..5;
    // permuted it is [ic0k0, ic1k0, ic2k0, ic0k1, ic1k1, ic2k1] = 0, 2, 4, 1, 3, 5.
    const want0 = [_]f32{ 0, 2, 4, 1, 3, 5 };
    for (want0, 0..) |v, i| {
        try testing.expectEqual(v, got[i]);
    }
    // ...and row 1 is the same pattern offset by the row stride, not a repeat of
    // row 0 (which is what a permutation that dropped `oc` would give).
    try testing.expectEqual(@as(f32, 6), got[6]);
    try testing.expectEqual(@as(f32, 8), got[7]);
    try testing.expectEqual(@as(usize, k * in_ch), d.w.cols);
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
    const filt: [12]f32 = @splat(0);
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
