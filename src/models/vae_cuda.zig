//! Wan 2.1 VAE decode on the hand-PTX CUDA backend.
//!
//! The CUDA analogue of `vae_gpu` (Vulkan): 3x3 convs run as banded `im2col`
//! (with an optional fused nearest-exact 2x upsample) + an f32 GEMM; 1x1 convs
//! are direct GEMMs; the per-position channel L2 norms (+ their trailing silus)
//! are the `vae_norm` kernel; residual adds are `opAdd`. The mid-block single
//! head (dim 384) reuses the DiT's tensor-core attention (`opAttnTC` with
//! n_heads=1, hd=384) — no VAE-specific attention kernel. Everything is device
//! resident from the latent upload to the RGB download.
//!
//! Conv/attention weights are f32 and stream through the Backend weight cache,
//! so a small --vram-budget degrades to weight streaming here too.

const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;

const Bufs = struct {
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{}, // also serves as conv2's output (`v` aliases `u` — see res)
    patch: Buf = .{},
    // mid-block attention q/k/v/out (reused proj into aq).
    aq: Buf = .{},
    ak: Buf = .{},
    av: Buf = .{},
    ao: Buf = .{},

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
    }
};

/// Decode a VAE-space latent (planar [16][zh][zw], already denormalized) to
/// planar [3][8*zh][8*zw] pixels in [-1, 1]. Caller frees the result.
pub fn decode(dec: *const wan_vae.Decoder, be: *Backend, io: std.Io, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize, cancel: ?*std.atomic.Value(bool)) ![]f32 {
    _ = io;
    std.debug.assert(z.len == wan_vae.latent_channels * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(be);

    var h = zh;
    var w = zw;
    const n0 = zh * zw;

    {
        const rows = try wan_vae.planarToRows(gpa, z, wan_vae.latent_channels, n0);
        defer gpa.free(rows);
        try be.ensureDeviceBuffer(&bufs.x, n0 * wan_vae.latent_channels * 4);
        try be.tensorUpload(bufs.x, std.mem.sliceAsBytes(rows));
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    try conv(be, &bufs, &bufs.x, &bufs.t, h, w, dec.post_quant, false);
    std.mem.swap(Buf, &bufs.x, &bufs.t);
    try conv(be, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, false);
    std.mem.swap(Buf, &bufs.x, &bufs.t);

    try res(be, &bufs, h, w, dec.mid_res1);
    try attn(be, &bufs, h, w, dec.mid_attn);
    try res(be, &bufs, h, w, dec.mid_res2);

    // The ~seq² attention scores plane is dead now but would otherwise stay
    // resident through the 8× upsampling; flush + free it.
    try be.endBatch();
    be.freeAttnScratch();
    try be.beginBatch();

    for (dec.ups) |layer| {
        // Poll cancel between layers so a stop lands mid-decode; the errdefer
        // above aborts the in-flight batch.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        switch (layer) {
            .res => |rb| try res(be, &bufs, h, w, rb),
            .up => |cv| {
                try conv(be, &bufs, &bufs.x, &bufs.t, h, w, cv, true);
                std.mem.swap(Buf, &bufs.x, &bufs.t);
                h *= 2;
                w *= 2;
            },
        }
    }

    // head: norm + silu + conv
    const n = h * w;
    try norm(be, &bufs.x, &bufs.t, n, dec.head_norm, true);
    try conv(be, &bufs, &bufs.t, &bufs.u, h, w, dec.head_conv, false);

    try be.endBatch();

    const rgb_rows = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb_rows);
    try be.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb_rows));
    return wan_vae.rowsToPlanar(gpa, rgb_rows, 3, n);
}

/// norm(x) (+ optional fused silu) into `dst`.
fn norm(be: *Backend, src: *const Buf, dst: *Buf, n: usize, gamma: []const f32, silu: bool) !void {
    const c = gamma.len;
    try be.ensureDeviceBuffer(dst, n * c * 4);
    const gbuf: Buf = .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(gamma)), .mem = .null_handle, .size = 0 };
    try be.opVaeNorm(src.*, dst.*, gbuf, n, c, silu);
}

/// Conv into `dst` ([n_out][co]); with `up`, the source is read through a fused
/// nearest-exact 2x upsample and n_out covers the doubled dims. All convs use
/// the f32 GEMM (parity-first; the VAE decode is a one-time cost).
fn conv(be: *Backend, bufs: *Bufs, src: *const Buf, dst: *Buf, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
    const wbytes = std.mem.sliceAsBytes(cv.w);
    // Big convs (co>=96) go through the f16 tensor-core GEMM; post_quant (16) and
    // the 3-channel head stay on the f32 GEMM (padding co to the 128 tile would
    // waste more than the tensor cores return).
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

    // Library backend: a fused cuDNN NHWC conv (no im2col) for the big non-upsample
    // convs. Upsample convs keep the fused im2col-upsample path (cuDNN can't do the
    // fused 2x resample); small convs stay on the GEMM (co padding not worth it).
    if (be.kernels == .libs and coop and !up) {
        return be.opConvCudnn(dst.*, 0, src.*, h, w, wbytes, cv.co, cv.ci, cv.b);
    }

    // Band positions: multiple of 4 keeps the GEMM's y byte offset 16-aligned.
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

/// Residual block over bufs.x in place (result swapped back into bufs.x).
/// `u` is reused for both conv outputs: conv1's result is consumed by norm2
/// before conv2 overwrites it, so conv2 (the old `v`) can share the same buffer
/// — one fewer full-resolution buffer (~720 MiB at 1 MP).
fn res(be: *Backend, bufs: *Bufs, h: usize, w: usize, rb: wan_vae.ResBlock) !void {
    const n = h * w;
    try norm(be, &bufs.x, &bufs.t, n, rb.norm1, true);
    try conv(be, bufs, &bufs.t, &bufs.u, h, w, rb.conv1, false);
    try norm(be, &bufs.u, &bufs.t, n, rb.norm2, true); // consumes u
    try conv(be, bufs, &bufs.t, &bufs.u, h, w, rb.conv2, false); // reuses u
    if (rb.shortcut) |sc| {
        try conv(be, bufs, &bufs.x, &bufs.t, h, w, sc, false);
        try be.opAdd(bufs.u, bufs.t, n * rb.conv2.co);
    } else {
        try be.opAdd(bufs.u, bufs.x, n * rb.conv2.co);
    }
    std.mem.swap(Buf, &bufs.x, &bufs.u);
}

/// Mid-block single-head (dim c) self-attention: qkv GEMMs → tensor-core
/// attention (n_heads=1, hd=c) → proj GEMM → residual add.
fn attn(be: *Backend, bufs: *Bufs, h: usize, w: usize, ab: wan_vae.AttnBlock) !void {
    const n = h * w;
    const c = ab.qkv.ci;
    try norm(be, &bufs.x, &bufs.t, n, ab.norm, false);

    const wq = std.mem.sliceAsBytes(ab.qkv.w[0 .. c * c]);
    const wk = std.mem.sliceAsBytes(ab.qkv.w[c * c .. 2 * c * c]);
    const wv = std.mem.sliceAsBytes(ab.qkv.w[2 * c * c .. 3 * c * c]);
    try be.ensureDeviceBuffer(&bufs.aq, n * c * 4);
    try be.ensureDeviceBuffer(&bufs.ak, n * c * 4);
    try be.ensureDeviceBuffer(&bufs.av, n * c * 4);
    try be.ensureDeviceBuffer(&bufs.ao, n * c * 4);
    try be.opMatmul(bufs.aq, 0, bufs.t, 0, n, wq, false, c, c, 1.0, ab.qkv.b[0..c]);
    try be.opMatmul(bufs.ak, 0, bufs.t, 0, n, wk, false, c, c, 1.0, ab.qkv.b[c .. 2 * c]);
    try be.opMatmul(bufs.av, 0, bufs.t, 0, n, wv, false, c, c, 1.0, ab.qkv.b[2 * c .. 3 * c]);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
    try be.opAttnTC(bufs.aq, bufs.ak, bufs.av, bufs.ao, n, 1, 1, c, scale);

    // proj into aq (its q content was consumed by the attention gather).
    try be.opMatmul(bufs.aq, 0, bufs.ao, 0, n, std.mem.sliceAsBytes(ab.proj.w), false, c, c, 1.0, ab.proj.b);
    try be.opAdd(bufs.x, bufs.aq, n * c);
}
