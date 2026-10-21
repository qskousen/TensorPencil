//! The CUDA adapter every encoder tower walk shares (`embed_*_dev.zig`): buffers on
//! demand, transfers, and the handful of ops an encoder block needs, each taking its
//! dimensions explicitly. Linears run in each weight's storage dtype through
//! `lin_cuda.towerLinear`, the f16 tensor-core regime, so parity against the CPU is
//! relative.
//!
//! No weight scope: the converted GEMM weights stay RESIDENT across calls (they borrow
//! the mmap-stable CPU model), where a scope would re-upload a whole model per call.

const std = @import("std");
const tower = @import("tower.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const Weight = @import("tp_ops").matmul.Weight;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

pub const Dev = struct {
    pub const Buffer = Buf;

    be: *Backend,

    pub fn alloc(d: Dev, floats: usize) !Buf {
        return d.be.tensorCreate(floats * 4);
    }

    pub fn release(d: Dev, b: *Buf) void {
        d.be.tensorDestroy(b);
    }

    /// A buffer of its own holding `data`, released by the caller.
    pub fn table(d: Dev, comptime T: type, data: []const T) !Buf {
        const b = try d.be.tensorCreate(data.len * @sizeOf(T));
        errdefer d.be.tensorDestroy(@constCast(&b));
        try d.be.tensorUpload(b, std.mem.sliceAsBytes(data));
        return b;
    }

    pub fn upload(d: Dev, b: Buf, data: []const f32) !void {
        try d.be.tensorUpload(b.slice(0, data.len * 4), std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: Buf, dst: []f32) !void {
        try d.be.tensorDownload(b.slice(0, dst.len * 4), std.mem.sliceAsBytes(dst));
    }

    pub fn bindAttention(_: Dev, _: Buf, _: Buf, _: Buf, _: Buf, _: Buf) void {}

    pub fn beginBatch(d: Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn done(d: Dev) void {
        d.be.freeAttnScratch();
        d.be.freeConvScratch();
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, bias, null);
    }

    /// Rotate-half over `rows` rows from `row0`, `stride` floats apart; `freqs` is
    /// `[cos | sin]` with the sine half at `sin_off`.
    pub fn rope(d: Dev, x: Buf, freqs: Buf, row0: usize, rows: usize, heads: usize, stride: usize, half: usize, sin_off: usize) !void {
        try d.be.ropeHalf(x.viewF32(row0 * stride), freqs, rows, heads, half, sin_off, 0);
    }

    /// Unmasked attention over `rows` rows; with `bounds` each row attends its own
    /// `[start | end]` range, read `bounds_off` words into `bounds`.
    pub fn attention(d: Dev, q: Buf, k: Buf, v: Buf, out: Buf, rows: usize, bounds: ?Buf, bounds_off: usize, heads: usize, kv_heads: usize, hd: usize, qk_scale: f32) !void {
        if (bounds) |bd| return d.be.opAttnBatched(q, k, v, out, bd.viewBytes(bounds_off * 4), rows, heads, kv_heads, hd, qk_scale);
        try d.be.attn(q, k, v, out, rows, rows, heads, kv_heads, hd, qk_scale, false);
    }

    pub fn uploadWordsAt(d: Dev, b: Buf, off_words: usize, data: []const u32) !void {
        try d.be.tensorUpload(b.slice(off_words * 4, data.len * 4), std.mem.sliceAsBytes(data));
    }

    /// Row r attends keys `[start, end)` from the u32 table `bounds_off` words into
    /// `scratch`, split `nsplit` ways (`Backend.opAttnBounded`).
    pub fn attentionRows(d: Dev, q: Buf, k: Buf, v: Buf, out: Buf, scratch: Buf, bounds_off: usize, rows: usize, heads: usize, kv_heads: usize, hd: usize, nsplit: usize, qk_scale: f32) !void {
        try d.be.opAttnBounded(q, k, v, out, scratch, bounds_off, rows, heads, kv_heads, hd, nsplit, qk_scale);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn scale(d: Dev, x: Buf, n: usize, s: f32) !void {
        try d.be.opScale(x, s, n);
    }

    pub fn layerNorm(d: Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try d.be.opLayerNorm(x, y, w, b, rows, dim, eps);
    }

    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize, eps: f32) !void {
        const wb: Buf = .{ .buf = try d.be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
        try d.be.qkNorm(x, y, wb, rows, dim, eps);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.be.gelu(x, n);
    }

    pub fn geluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.be.geluMul(g, u, n);
    }
};

/// Free every device copy made of `model`'s weights and vectors
/// (`tower.forEachHostBuffer`), before the host buffers they key on are freed.
pub fn evictModel(be: *Backend, model: anytype) void {
    tower.forEachHostBuffer(model, Evict{ .be = be });
}

const Evict = struct {
    be: *Backend,

    pub fn weight(e: Evict, bytes: []const u8) void {
        e.be.evictWeightRange(bytes);
    }

    pub fn vector(e: Evict, bytes: []const u8) void {
        e.be.evictWeightRange(bytes);
    }
};
