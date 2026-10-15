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
const conv_gpu = @import("conv_gpu.zig");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const minimax_h3_audio_dev = @import("minimax_h3_audio_dev.zig");

const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const AudioDecoder = audio.AudioDecoder;
const DevConv = audio.DevConv;
const DevAct = audio.DevAct;
const DevConvT = audio.DevConvT;
pub const Session = audio.DevSession;

const nullBuf: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

pub const supported = audio.deviceSupported;

pub const Workspace = struct {
    /// The conv GEMMs' dispatcher.
    lin: lin_gpu.Lin,
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

    pub fn init(ctx: *Context, gpa: std.mem.Allocator, dec: *const AudioDecoder, t: usize) !Workspace {
        const s = audio.Shapes.of(dec, t);
        var ws: Workspace = .{ .shapes = s, .lin = try lin_gpu.Lin.init(gpa, ctx, @max(dec.dec_in.out_ch, dec.conv_pre.out_ch), 0, 0) };
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
        self.lin.deinit();
    }
};

/// A host vector as a device buffer. `smallBuffer` caches by host POINTER, so the
/// session's permuted weights upload once and are reused every decode.
fn devBuf(ctx: *Context, data: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(data)), .mem = .null_handle, .size = data.len * 4 };
}

/// `dst[out_len][out_ch] = conv(src[in_len][in_ch])`, its GEMM f32 like the CUDA
/// arm's; `dst` may alias `src` (the patch holds the whole input, one band).
fn convAt(ws: *Workspace, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
    return conv_gpu.conv1d(&ws.lin, f32_plan, ws.patch, ws.shapes.patch, dst, src, c, in_len);
}

const f32_plan: lin_gpu.Plan = blk: {
    var p = lin_gpu.Plan.tower;
    p.dense = .native;
    break :blk p;
};

/// Same signature as `minimax_h3_audio.decode`, so the two are drop-in alternatives.
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
    try minimax_h3_audio_dev.decode(Dev{ .ctx = ctx, .ws = ws }, dec, sess, gpa, out, z, t, cancel);
}

/// The `minimax_h3_audio_dev` adapter.
const Dev = struct {
    ctx: *Context,
    ws: *Workspace,

    pub fn beginBatch(d: Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn upload(d: Dev, b: Buf, data: []const f32) !void {
        try d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: Buf, data: []f32) !void {
        try d.ctx.tensorDownload(b, std.mem.sliceAsBytes(data));
    }

    pub fn conv(d: Dev, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
        return convAt(d.ws, dst, src, c, in_len);
    }

    pub fn convT(d: Dev, dst: Buf, src: Buf, u: DevConvT, len: usize, up_len: usize) !void {
        const total = up_len * u.out_ch;
        try d.ctx.opElt(.convt1d_ca, src, dst, try devBuf(d.ctx, u.w), try devBuf(d.ctx, u.bias), .{
            .u0 = @intCast(total),
            .u1 = @intCast(u.out_ch),
            .u2 = @intCast(u.in_ch),
            .u3 = @intCast(len),
            .u4 = @intCast(u.k),
            .u5 = @intCast(u.stride),
            .f0 = @floatFromInt(u.padding),
        }, total, 1, 1);
    }

    pub fn aaUp(d: Dev, src: Buf, act: DevAct, len: usize) !void {
        const total = 2 * len * act.channels;
        try d.ctx.opElt(.aa_up_snake, src, d.ws.aa, try devBuf(d.ctx, act.up_filter), try devBuf(d.ctx, act.snake), .{
            .u0 = @intCast(total),
            .u1 = @intCast(act.channels),
            .u2 = @intCast(len),
            .u3 = @intCast(act.k),
            .u4 = @intCast(act.upPad()),
            .u5 = @intCast(act.upSlice()),
        }, total, 1, 1);
    }

    pub fn aaDown(d: Dev, dst: Buf, act: DevAct, len: usize) !void {
        const total = len * act.channels;
        try d.ctx.opElt(.aa_down, d.ws.aa, dst, try devBuf(d.ctx, act.down_filter), null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(act.channels),
            .u2 = @intCast(2 * len),
            .u3 = @intCast(act.k),
            .u4 = @intCast(act.downPad()),
        }, total, 1, 1);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn copy(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        try d.ctx.opElt(.copy, src, dst, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn scale(d: Dev, x: Buf, f: f32, n: usize) !void {
        try d.ctx.opElt(.scale_f32, x, x, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
    }
};
