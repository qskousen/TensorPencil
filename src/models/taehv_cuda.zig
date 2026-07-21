//! CUDA decode for the taew2_1 (TAEHV) approx VAE — the GPU path for the live
//! per-step preview. Mirrors vae_cuda's conv (im2col + f16/f32 GEMM, with a
//! fused nearest-2x upsample), driving the layer sequence from taehv.zig.
//!
//! The input Clamp (tanh(x/3)*3) is applied on the host before upload (it's on
//! the small latent). Each stage's spatial Upsample is fused into that stage's
//! 3x3 conv (`up=true`) — valid because it commutes with the per-pixel TGrow
//! 1x1 conv that sits between them.
const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const wan_vae = @import("wan_vae.zig");
const taehv = @import("taehv.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

const patch_band_bytes: usize = 256 << 20;

const Bufs = struct {
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{},
    patch: Buf = .{},

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
    }
};

/// Decode planar [16][zh][zw] sampler latent to RGB8 [8*zh][8*zw][3].
pub fn decode(dec: *const taehv.Decoder, be: *Backend, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
    std.debug.assert(z.len == taehv.latent_channels * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(be);

    var h = zh;
    var w = zw;
    const n0 = zh * zw;
    {
        const rows = try wan_vae.planarToRows(gpa, z, taehv.latent_channels, n0);
        defer gpa.free(rows);
        for (rows) |*v| v.* = std.math.tanh(v.* / 3.0) * 3.0; // Clamp, on host
        try be.ensureDeviceBuffer(&bufs.x, n0 * taehv.latent_channels * 4);
        try be.tensorUpload(bufs.x, std.mem.sliceAsBytes(rows));
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    try conv(be, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, false); // 16 -> 256
    std.mem.swap(Buf, &bufs.x, &bufs.t);
    try be.opRelu(bufs.x, h * w * dec.conv_in.co);

    for (dec.stages) |stage| {
        for (stage.mb) |mb| try memBlock(be, &bufs, h, w, mb);
        try conv(be, &bufs, &bufs.x, &bufs.t, h, w, stage.tgrow, false); // TGrow 1x1 (first-frame slice)
        std.mem.swap(Buf, &bufs.x, &bufs.t);
        try conv(be, &bufs, &bufs.x, &bufs.t, h, w, stage.sc, true); // 3x3 with fused 2x upsample
        std.mem.swap(Buf, &bufs.x, &bufs.t);
        h *= 2;
        w *= 2;
    }

    try be.opRelu(bufs.x, h * w * 64); // ReLU before head
    try conv(be, &bufs, &bufs.x, &bufs.u, h, w, dec.head_conv, false); // 64 -> 3
    try be.endBatch();

    const n = h * w;
    const rgb_rows = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb_rows);
    try be.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb_rows));
    const rgb = try gpa.alloc(u8, n * 3);
    for (rgb, rgb_rows) |*o, v| o.* = @intFromFloat(std.math.clamp(v, 0.0, 1.0) * 255.0);
    return rgb;
}

/// MemBlock (past=0): out = ReLU(conv4(ReLU(conv2(ReLU(conv0(x))))) + x), in place on bufs.x.
fn memBlock(be: *Backend, bufs: *Bufs, h: usize, w: usize, mb: taehv.MemBlock) !void {
    const n = h * w * mb.n;
    try conv(be, bufs, &bufs.x, &bufs.t, h, w, mb.conv0, false);
    try be.opRelu(bufs.t, n);
    try conv(be, bufs, &bufs.t, &bufs.u, h, w, mb.conv2, false);
    try be.opRelu(bufs.u, n);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, mb.conv4, false);
    try be.opAdd(bufs.t, bufs.x, n); // + skip (identity)
    try be.opRelu(bufs.t, n);
    std.mem.swap(Buf, &bufs.x, &bufs.t);
}

/// 3x3 (optionally with fused nearest-2x upsample) or 1x1 conv. Mirrors
/// vae_cuda.conv: coop (co>=96) convs use the f16 tensor-core GEMM, small ones
/// the f32 GEMM; the .libs backend uses a fused cuDNN conv for big non-upsample.
fn conv(be: *Backend, bufs: *Bufs, src: *const Buf, dst: *Buf, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
    const wbytes = std.mem.sliceAsBytes(cv.w);
    const coop = cv.co >= 96;
    if (cv.k == 1) {
        std.debug.assert(!up);
        const n = h * w;
        try be.ensureDeviceBuffer(dst, n * cv.co * 4);
        if (coop) return be.opConvF16(dst.*, 0, src.*, n, wbytes, cv.co, cv.ci, cv.b);
        return be.opMatmul(dst.*, 0, src.*, 0, n, wbytes, false, cv.co, cv.ci, 1.0, cv.b);
    }
    std.debug.assert(cv.k == 3);

    const oh = if (up) 2 * h else h;
    const ow = if (up) 2 * w else w;
    const n_out = oh * ow;
    const patch_len = 9 * cv.ci;
    try be.ensureDeviceBuffer(dst, n_out * cv.co * 4);

    if (be.kernels == .libs and coop and !up) {
        return be.opConvCudnn(dst.*, 0, src.*, h, w, wbytes, cv.co, cv.ci, cv.b);
    }

    const band = @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
    try be.ensureDeviceBuffer(&bufs.patch, band * patch_len * 4);
    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try be.opIm2col(src.*, bufs.patch, bn, patch_len, cv.ci, w, h, p0, up);
        if (coop) {
            try be.opConvF16(dst.*, p0 * cv.co, bufs.patch, bn, wbytes, cv.co, patch_len, cv.b);
        } else {
            try be.opMatmul(dst.*, p0 * cv.co * 4, bufs.patch, 0, bn, wbytes, false, cv.co, patch_len, 1.0, cv.b);
        }
    }
}
