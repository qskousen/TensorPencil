//! The Wan VAE decoder's device walk, written once for every GPU backend over the
//! same adapter as `sd_vae_dev` (`sd_vae_cuda.Dev`, `sd_vae_gpu.Dev`), plus its one
//! op of its own, `channelNorm` (the per-position channel L2 norm, optionally followed
//! by silu). `body` runs between the adapter's upload of the latent rows into
//! `bufs.stage` and its download of the RGB rows from the same buffer.

const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const sd_vae_dev = @import("sd_vae_dev.zig");

/// Returns the decoded extent.
pub fn body(d: anytype, dec: *const wan_vae.Decoder, zh: usize, zw: usize, cancel: ?*std.atomic.Value(bool)) ![2]usize {
    const b = d.bufs;
    var h = zh;
    var w = zw;

    try d.conv(&b.t, &b.stage, h, w, dec.post_quant.asOps(), .stride1, 1.0);
    try d.conv(&b.x, &b.t, h, w, dec.conv_in.asOps(), .stride1, 1.0);

    try res(d, h, w, dec.mid_res1);
    try attn(d, h, w, dec.mid_attn);
    try d.midDone();
    try res(d, h, w, dec.mid_res2);

    var ch = dec.conv_in.co;
    for (dec.ups) |layer| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        switch (layer) {
            .res => |rb| {
                try res(d, h, w, rb);
                ch = rb.conv2.co;
            },
            .up => |cv| {
                try sd_vae_dev.convResidual(d, &b.t, &b.x, h, w, cv.asOps(), .upsample2x);
                std.mem.swap(@TypeOf(b.x), &b.x, &b.t);
                h *= 2;
                w *= 2;
                ch = cv.co;
            },
            .save => try d.copy(b.sv, b.x, h * w * ch),
            .dup_add => |du| try d.dupUpAdd(b.x, b.sv, h / 2, w / 2, du),
        }
    }

    try d.channelNorm(&b.t, &b.x, h * w, dec.head_norm, true);
    // The head writes RGB into the f32 staging buffer; the host wants f32.
    try d.conv(&b.stage, &b.t, h, w, dec.head_conv.asOps(), .stride1, 1.0);
    return .{ h, w };
}

/// Residual block over `x` in place. `u` carries both convolution outputs: conv1's is
/// dead once norm2 has read it.
fn res(d: anytype, h: usize, w: usize, rb: wan_vae.ResBlock) !void {
    const b = d.bufs;
    const n = h * w;
    try d.channelNorm(&b.t, &b.x, n, rb.norm1, true);
    try d.conv(&b.u, &b.t, h, w, rb.conv1.asOps(), .stride1, 1.0);
    try d.channelNorm(&b.t, &b.u, n, rb.norm2, true);
    try d.conv(&b.u, &b.t, h, w, rb.conv2.asOps(), .stride1, 1.0);
    if (rb.shortcut) |sc| {
        try sd_vae_dev.convResidual(d, &b.t, &b.x, h, w, sc.asOps(), .stride1);
        try d.add(b.u, b.t, n * rb.conv2.co);
    } else {
        try d.add(b.u, b.x, n * rb.conv2.co);
    }
    std.mem.swap(@TypeOf(b.x), &b.x, &b.u);
}

/// The mid-block's single head over every channel, `x += proj(attn(norm(x)))`. The
/// fused `to_qkv` is three row ranges of one repacked weight.
fn attn(d: anytype, h: usize, w: usize, ab: wan_vae.AttnBlock) !void {
    const b = d.bufs;
    const n = h * w;
    const c = ab.proj.co;
    try d.channelNorm(&b.t, &b.x, n, ab.norm, false);
    inline for (.{ &b.aq, &b.ak, &b.av }, 0..) |dst, i| {
        try d.conv(dst, &b.t, h, w, wan_vae.qkvPart(ab.qkv, i), .stride1, 1.0);
    }
    try d.attnCore(n, c);
    try d.conv(&b.t, &b.ao, h, w, ab.proj.asOps(), .stride1, 1.0);
    try d.add(b.x, b.t, n * c);
}
