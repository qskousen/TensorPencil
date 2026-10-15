//! The taew2_1 (TAEHV) approx VAE decode, the live per-step preview, written once for
//! every GPU backend over an adapter (`taehv_cuda.Dev`, `taehv_gpu.Dev`).
//!
//! The input Clamp (tanh(x/3)*3) runs on the host before upload, on the small latent.
//! Each stage's spatial Upsample is fused into that stage's 3x3 conv (`up`), which is
//! valid because it commutes with the per-pixel TGrow 1x1 conv between them.
//!
//! The adapter holds `bufs` (`x` the running activation, `t`/`u` scratch, each grown by
//! the conv writing it) and supplies `conv`, `relu`, `addRelu` (`x = relu(x + y)`) and
//! the batch and transfer ops.

const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const taehv = @import("taehv.zig");

/// Planar `[16][zh][zw]` sampler latent to RGB8 `[8*zh][8*zw][3]`. Caller frees.
pub fn decode(d: anytype, dec: *const taehv.Decoder, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
    std.debug.assert(z.len == taehv.latent_channels * zh * zw);
    const b = d.bufs;
    {
        const rows = try wan_vae.planarToRows(gpa, z, taehv.latent_channels, zh * zw);
        defer gpa.free(rows);
        for (rows) |*v| v.* = std.math.tanh(v.* / 3.0) * 3.0;
        try d.upload(&b.x, rows);
    }

    var h = zh;
    var w = zw;
    try d.beginBatch();
    errdefer d.abortBatch();
    try d.conv(&b.x, &b.t, h, w, dec.conv_in, false);
    std.mem.swap(@TypeOf(b.x), &b.x, &b.t);
    try d.relu(b.x, h * w * dec.conv_in.co);

    for (dec.stages) |stage| {
        for (stage.mb) |mb| try memBlock(d, h, w, mb);
        // TGrow 1x1 (the first-frame slice), then the 3x3 with its fused 2x upsample.
        try d.conv(&b.x, &b.t, h, w, stage.tgrow, false);
        std.mem.swap(@TypeOf(b.x), &b.x, &b.t);
        try d.conv(&b.x, &b.t, h, w, stage.sc, true);
        std.mem.swap(@TypeOf(b.x), &b.x, &b.t);
        h *= 2;
        w *= 2;
    }

    try d.relu(b.x, h * w * 64);
    try d.conv(&b.x, &b.u, h, w, dec.head_conv, false);
    try d.endBatch();

    const n = h * w;
    const rgb_rows = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb_rows);
    try d.download(b.u, rgb_rows);
    const rgb = try gpa.alloc(u8, n * 3);
    for (rgb, rgb_rows) |*o, v| o.* = @intFromFloat(std.math.clamp(v, 0.0, 1.0) * 255.0);
    return rgb;
}

/// MemBlock (past=0): `x = relu(conv4(relu(conv2(relu(conv0(x))))) + x)`.
fn memBlock(d: anytype, h: usize, w: usize, mb: taehv.MemBlock) !void {
    const b = d.bufs;
    const n = h * w * mb.n;
    try d.conv(&b.x, &b.t, h, w, mb.conv0, false);
    try d.relu(b.t, n);
    try d.conv(&b.t, &b.u, h, w, mb.conv2, false);
    try d.relu(b.u, n);
    try d.conv(&b.u, &b.t, h, w, mb.conv4, false);
    try d.addRelu(b.t, b.x, n);
    std.mem.swap(@TypeOf(b.x), &b.x, &b.t);
}
