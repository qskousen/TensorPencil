//! The Vulkan adapter every encoder tower walk shares (`embed_*_dev.zig`): buffers on
//! demand, transfers, and the handful of ops an encoder block needs, each taking its
//! dimensions explicitly. Linears run in each weight's storage dtype through
//! `lin_gpu.towerLinear`, over the CPU model's own weights (the device cache keys on the
//! host pointer, so nothing is dequantized or copied).

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const Weight = @import("tp_ops").matmul.Weight;

const Buf = gpu.DeviceBuffer;

pub const Dev = struct {
    pub const Buffer = Buf;

    ctx: *gpu.Context,
    tl: *const lin_gpu.Lin,

    pub fn alloc(d: Dev, floats: usize) !Buf {
        return d.ctx.tensorCreate(floats * 4);
    }

    pub fn release(d: Dev, b: *Buf) void {
        d.ctx.tensorDestroy(b);
    }

    /// A buffer of its own holding `data`, released by the caller.
    pub fn table(d: Dev, comptime T: type, data: []const T) !Buf {
        var b = try d.ctx.tensorCreate(data.len * @sizeOf(T));
        errdefer d.ctx.tensorDestroy(&b);
        try d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
        return b;
    }

    pub fn upload(d: Dev, b: Buf, data: []const f32) !void {
        try d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: Buf, dst: []f32) !void {
        try d.ctx.tensorDownloadAt(b, 0, std.mem.sliceAsBytes(dst));
    }

    /// Before the batch opens: the batched attention's descriptor set is persistent, so
    /// binding mid-batch would touch a set already referenced by recorded commands.
    pub fn bindAttention(d: Dev, q: Buf, k: Buf, v: Buf, out: Buf, bounds: Buf) void {
        d.ctx.attnBatchedBind(q, k, v, out, bounds);
    }

    /// One submission for the whole body: these forwards are many tiny ops.
    pub fn beginBatch(d: Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn done(_: Dev) void {}

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_gpu.towerLinear(d.tl, y, x, m, w, bias);
    }

    /// Rotate-half over `rows` rows from `row0`, `stride` floats apart; `freqs` is
    /// `[cos | sin]` with the sine half at `sin_off`.
    pub fn rope(d: Dev, x: Buf, freqs: Buf, row0: usize, rows: usize, heads: usize, stride: usize, half: usize, sin_off: usize) !void {
        const n = rows * heads * half;
        try d.ctx.opElt(.rope_half, x, null, freqs, null, .{
            .u0 = @intCast(n),
            .u1 = @intCast(half),
            .u2 = @intCast(sin_off),
            .u3 = @intCast(heads),
            .u5 = @intCast(row0 * stride),
        }, n, 1, 1);
    }

    /// Unmasked attention over `rows` rows; with `bounds` it is block-diagonal, through
    /// the buffers `bindAttention` bound.
    pub fn attention(d: Dev, q: Buf, k: Buf, v: Buf, out: Buf, rows: usize, bounds: ?Buf, heads: usize, kv_heads: usize, hd: usize) !void {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        if (bounds != null) return d.ctx.attnBatchedDispatch(rows, heads, kv_heads, hd, scale);
        try d.ctx.opElt(.attn_full, q, k, v, out, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(heads),
            .u2 = @intCast(kv_heads),
            .u3 = @intCast(hd),
            .f0 = scale,
        }, rows * heads, 1, 1);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn layerNorm(d: Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try d.ctx.opElt(.layernorm, x, y, try small(d.ctx, w), try small(d.ctx, b), .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
    }

    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try d.ctx.opElt(.rmsnorm, x, y, try small(d.ctx, w), null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn geluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu_mul, g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

fn small(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}
