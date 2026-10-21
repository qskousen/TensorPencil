//! CUDA decode for the taew2_1 (TAEHV) approx VAE: the adapter for `taehv_dev.zig`.
//! The conv is im2col + f16/f32 GEMM, with a fused nearest-2x upsample.
const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const wan_vae = @import("wan_vae.zig");
const conv_cuda = @import("conv_cuda.zig");
const taehv = @import("taehv.zig");
const taehv_dev = @import("taehv_dev.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

const Bufs = struct {
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{},
    patch: Buf = .{},

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| be.tensorDestroy(&@field(self, name));
    }
};

/// `taehv_dev.decode` on this backend. Caller frees.
pub fn decode(dec: *const taehv.Decoder, be: *Backend, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
    var bufs: Bufs = .{};
    defer bufs.deinit(be);
    return taehv_dev.decode(Dev{ .be = be, .bufs = &bufs }, dec, gpa, z, zh, zw);
}

/// The `taehv_dev` adapter.
const Dev = struct {
    be: *Backend,
    bufs: *Bufs,

    pub fn upload(d: Dev, b: *Buf, data: []const f32) !void {
        try d.be.ensureDeviceBuffer(b, data.len * 4);
        try d.be.tensorUpload(b.*, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: Buf, dst: []f32) !void {
        try d.be.tensorDownload(b, std.mem.sliceAsBytes(dst));
    }

    pub fn beginBatch(d: Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn conv(d: Dev, src: *const Buf, dst: *Buf, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
        try convAt(d.be, d.bufs, src, dst, h, w, cv, up);
    }

    pub fn relu(d: Dev, x: Buf, n: usize) !void {
        try d.be.opRelu(x, n);
    }

    pub fn addRelu(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
        try d.be.opRelu(x, n);
    }
};

/// 3x3 (optionally with a fused nearest-2x upsample) or 1x1 conv, through the shared
/// conv (`conv_cuda.conv2d`).
fn convAt(be: *Backend, bufs: *Bufs, src: *const Buf, dst: *Buf, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
    const n_out = if (up) 4 * h * w else h * w;
    try be.ensureDeviceBuffer(dst, n_out * cv.co * 4);
    try conv_cuda.conv2d(be, &bufs.patch, dst, src, h, w, cv.asOps(), if (up) .upsample2x else .stride1, null, 1.0);
}
