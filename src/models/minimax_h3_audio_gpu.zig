//! Vulkan MiniMax H3 audio VAE (BigVGAN) decode, the twin of
//! `minimax_h3_audio_cuda` and the device form of `minimax_h3_audio.decode`.
//!
//! That arm's header has the three things that make this a port rather than a
//! transcription (channel-last signals, weights permuted at session build, the
//! anti-aliased activation collapsed into two gather kernels). All three are in
//! the shared `DevSession`, so this file is the kernel sequence and nothing else.
//!
//! ⚠️ **The conv GEMMs run in f32**, as on the CUDA arm and for its measured
//! reason: through f16 the vocoder is about -42 dB, which is an audible noise
//! floor, where f32 is under 16-bit PCM's own quantum. `opMatmul` is Vulkan's f32
//! GEMM, so that is the only arm here; there is no f16 fast path to fall into.
//!
//! `minimax-h3-audio-vk-test` checks it against the f32 host decode.

const std = @import("std");
const audio = @import("minimax_h3_audio.zig");
const gpu = @import("tp_gpu").context;

const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const AudioDecoder = audio.AudioDecoder;
const DevConv = audio.DevConv;
const DevAct = audio.DevAct;
pub const Session = audio.DevSession;

const nullBuf: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

pub const supported = audio.deviceSupported;

pub const Workspace = struct {
    /// The stage's upsampled signal, read by all three of its resblocks.
    up: Buf = nullBuf,
    /// The AMPBlock residual chain's ping-pong pair.
    blk0: Buf = nullBuf,
    blk1: Buf = nullBuf,
    /// The stage's running resblock sum.
    acc: Buf = nullBuf,
    /// The anti-aliased activation's 2x intermediate.
    aa: Buf = nullBuf,
    /// The conv patch matrix.
    patch: Buf = nullBuf,
    shapes: audio.Shapes = undefined,

    pub fn init(ctx: *Context, dec: *const AudioDecoder, t: usize) !Workspace {
        const s = audio.Shapes.of(dec, t);
        var ws: Workspace = .{ .shapes = s };
        errdefer ws.deinit(ctx);
        ws.up = try ctx.tensorCreate(s.sig * 4);
        ws.blk0 = try ctx.tensorCreate(s.sig * 4);
        ws.blk1 = try ctx.tensorCreate(s.sig * 4);
        ws.acc = try ctx.tensorCreate(s.sig * 4);
        ws.aa = try ctx.tensorCreate(2 * s.aa * 4);
        ws.patch = try ctx.tensorCreate(s.patch * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (.{ &self.up, &self.blk0, &self.blk1, &self.acc, &self.aa, &self.patch }) |b| ctx.tensorDestroy(b);
    }
};

/// A host vector as a device buffer. `smallBuffer` caches by host POINTER, so the
/// session's permuted weights upload once and are reused every decode.
fn devBuf(ctx: *Context, data: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(data)), .mem = .null_handle, .size = data.len * 4 };
}

/// `dst[out_len][out_ch] = conv(src[in_len][in_ch])`, im2col + f32 GEMM.
/// `dst` may alias `src`: the GEMM reads only the patch matrix.
fn conv(ctx: *Context, ws: *Workspace, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
    const out_len = in_len; // every conv here is stride 1 and same-padded
    std.debug.assert(out_len * c.plen() <= ws.shapes.patch);
    const plen = c.plen();
    const total = out_len * plen;
    try ctx.opElt(.im2col1d, src, ws.patch, null, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(plen),
        .u2 = @intCast(c.in_ch),
        .u3 = @intCast(in_len),
        .u4 = @intCast(c.dilation),
        .u5 = @intCast(c.padding),
        .f0 = @floatFromInt(c.stride),
        .f1 = 0,
    }, total, 1, 1);
    const w32 = std.mem.sliceAsBytes(c.w32);
    try ctx.opMatmul(dst, 0, ws.patch, 0, out_len, w32, false, c.out_ch, plen, 1.0, c.bias);
    return out_len;
}

/// `aa_up_snake` then `aa_down`: the anti-aliased activation, in place.
fn aaUp(ctx: *Context, ws: *Workspace, src: Buf, act: DevAct, len: usize) !void {
    const total = 2 * len * act.channels;
    try ctx.opElt(.aa_up_snake, src, ws.aa, try devBuf(ctx, act.up_filter), try devBuf(ctx, act.snake), .{
        .u0 = @intCast(total),
        .u1 = @intCast(act.channels),
        .u2 = @intCast(len),
        .u3 = @intCast(act.k),
        .u4 = @intCast(act.upPad()),
        .u5 = @intCast(act.upSlice()),
    }, total, 1, 1);
}

fn aaDown(ctx: *Context, ws: *Workspace, out: Buf, act: DevAct, len: usize) !void {
    const total = len * act.channels;
    try ctx.opElt(.aa_down, ws.aa, out, try devBuf(ctx, act.down_filter), null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(act.channels),
        .u2 = @intCast(2 * len),
        .u3 = @intCast(act.k),
        .u4 = @intCast(act.downPad()),
    }, total, 1, 1);
}

fn aaSnake(ctx: *Context, ws: *Workspace, x: Buf, act: DevAct, len: usize) !void {
    std.debug.assert(act.channels * len <= ws.shapes.aa);
    try aaUp(ctx, ws, x, act, len);
    try aaDown(ctx, ws, x, act, len);
}

fn add(ctx: *Context, a: Buf, b: Buf, total: usize) !void {
    try ctx.opElt(.add, a, b, null, null, .{ .u0 = @intCast(total) }, total, 1, 1);
}

/// One `AMPBlock1` over `src`, which is NOT modified. Returns the buffer holding
/// the result (one of the ping-pong pair).
fn ampBlock(ctx: *Context, ws: *Workspace, src: Buf, blk: *const Session.DevBlock, ch: usize, len: usize) !Buf {
    var cur = src;
    var dst = ws.blk0;
    var other = ws.blk1;
    for (blk.convs1, blk.convs2, 0..) |c1, c2, i| {
        // activations[::2] pairs with convs1 and [1::2] with convs2: six distinct
        // activations per block, not three reused.
        try aaUp(ctx, ws, cur, blk.acts[2 * i], len);
        try aaDown(ctx, ws, dst, blk.acts[2 * i], len);
        _ = try conv(ctx, ws, dst, dst, c1, len);
        try aaSnake(ctx, ws, dst, blk.acts[2 * i + 1], len);
        _ = try conv(ctx, ws, dst, dst, c2, len);
        try add(ctx, dst, cur, ch * len);
        cur = dst;
        // Ping-pong, so the next iteration's residual source stays intact.
        const tmp = dst;
        dst = other;
        other = tmp;
    }
    return cur;
}

/// Decode normalized stereo latents `[32][2][t]` (planar, the sampler's layout)
/// to interleaved samples in [-1, 1], `[len][2]`. Same signature as
/// `minimax_h3_audio.decode`, so the two are drop-in alternatives.
pub fn decode(
    dec: *const AudioDecoder,
    sess: *const Session,
    ctx: *Context,
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

        try ctx.beginBatch();
        errdefer ctx.abortBatch();
        try ctx.tensorUpload(ws.blk0, std.mem.sliceAsBytes(lat));
        _ = try conv(ctx, ws, ws.blk1, ws.blk0, sess.dec_in, t);
        _ = try conv(ctx, ws, ws.acc, ws.blk1, sess.conv_pre, t);

        var len = t;
        for (sess.ups, 0..) |u, i| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            const up_len = (len - 1) * u.stride + u.k - 2 * u.padding;
            const total = up_len * u.out_ch;
            try ctx.opElt(.convt1d_ca, ws.acc, ws.up, try devBuf(ctx, u.w), try devBuf(ctx, u.bias), .{
                .u0 = @intCast(total),
                .u1 = @intCast(u.out_ch),
                .u2 = @intCast(u.in_ch),
                .u3 = @intCast(len),
                .u4 = @intCast(u.k),
                .u5 = @intCast(u.stride),
                .f0 = @floatFromInt(u.padding),
            }, total, 1, 1);
            len = up_len;

            // The stage's resblocks are SUMMED and averaged, not chained, so each
            // reads the SAME `up` and the sum lands in `acc`.
            for (0..dec.n_kernels) |j| {
                const r = try ampBlock(ctx, ws, ws.up, &sess.blocks[i * dec.n_kernels + j], u.out_ch, len);
                if (j == 0) {
                    try ctx.opElt(.copy, r, ws.acc, null, null, .{ .u0 = @intCast(u.out_ch * len) }, u.out_ch * len, 1, 1);
                } else {
                    try add(ctx, ws.acc, r, u.out_ch * len);
                }
            }
            try ctx.opElt(.scale_f32, ws.acc, ws.acc, null, null, .{
                .u0 = @intCast(u.out_ch * len),
                .f0 = 1.0 / @as(f32, @floatFromInt(dec.n_kernels)),
            }, u.out_ch * len, 1, 1);
        }

        try aaSnake(ctx, ws, ws.acc, sess.act_post, len);
        const post_len = try conv(ctx, ws, ws.blk0, ws.acc, sess.conv_post, len);
        std.debug.assert(sess.conv_post.out_ch == 1 and post_len == samples);
        try ctx.endBatch();

        try ctx.tensorDownload(ws.blk0, std.mem.sliceAsBytes(tail));
        // Clamped to [-1, 1], no tanh and no final bias. The interleave happens
        // only here, because that is what a container wants.
        for (0..samples) |i| out[i * audio.stereo + s] = std.math.clamp(tail[i], -1.0, 1.0);
    }
}
