//! The SD-family VAE decoder's device walk (AutoencoderKL), written once for every GPU
//! backend over an adapter (`sd_vae_cuda.Dev`, `sd_vae_gpu.Dev`). `body` runs between
//! the adapter's upload of the latent into `bufs.stage` and its download of the RGB from
//! the same buffer, inside a recording batch that `midDone` may close and reopen.
//!
//! The adapter holds `bufs` (the activation buffers by the names below, which this file
//! swaps between) and supplies `conv` (with an activation divisor), `groupNorm`, `add`,
//! `attnCore` (the mid-block's one head over `aq`/`ak`/`av` into `ao`) and `midDone`
//! (after the mid-block, whose attention scratch is dead for the rest of the decode).

const std = @import("std");
const sd_vae = @import("sd_vae.zig");

/// Divisor applied to the activation of a convolution that reads the RESIDUAL STREAM,
/// before it is cast to a 16-bit GEMM operand, undone exactly in the epilogue.
///
/// Measured, not defensive: an SDXL VAE decoder's residual reaches 4.2e5 against f16's
/// 65504 (probed at a 64² latent: the last upsample conv's output is 1.26e5 and the next
/// block's 1x1 shortcut reads it), so without it the cast produced `inf`, the next
/// GroupNorm spread it to NaN through its mean, and every SDXL render at 512² or larger
/// came out solid white with no error. SD1.5's VAE peaks near 7e3, which is why the family
/// this code was written against never showed it, and why ComfyUI decodes the SDXL VAE in
/// fp32 (or ships the "fp16-fix" weights) by default.
///
/// 256 is a power of two, so the scaling is exact (exponent shift only) with 38x headroom
/// over the measured peak; values below 1.5e-5 underflow, 1e-10 of that peak. Only the
/// residual-reading convolutions take it: every other one reads a GroupNorm output (peak
/// 67 measured) or the latent, where a divisor would only cost low-end precision.
pub const residual_act_div: f32 = 256.0;

/// Decodes the latent in `bufs.stage` (`[lat_h*lat_w][z]`) to RGB in `bufs.stage`; returns
/// the decoded extent.
pub fn body(d: anytype, dec: *const sd_vae.Decoder, lat_h: usize, lat_w: usize, cancel: ?*std.atomic.Value(bool)) ![2]usize {
    const b = d.bufs;
    const Buf = @TypeOf(b.x);
    var h = lat_h;
    var w = lat_w;
    var ch = dec.cfg.innermost();

    // The Flux-lineage 16-channel VAE has NO `post_quant_conv` (BFL dropped it), so
    // `conv_in` reads the staged latent directly; with one it runs first.
    if (dec.post_quant) |pq| {
        try d.conv(&b.t, &b.stage, h, w, pq, .stride1, 1.0);
        try d.conv(&b.x, &b.t, h, w, dec.conv_in, .stride1, 1.0);
    } else {
        try d.conv(&b.x, &b.stage, h, w, dec.conv_in, .stride1, 1.0);
    }

    try resnet(d, h, w, dec.mid1);
    try attn(d, h, w, dec.mid_attn);
    try resnet(d, h, w, dec.mid2);
    try d.midDone();

    for (dec.levels) |level| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        for (level.blocks) |blk| {
            try resnet(d, h, w, blk);
            ch = blk.out_ch;
        }
        if (level.upsample) |up| {
            // Nearest 2x then a 3x3, with the resample fused into the patch gather.
            try convResidual(d, &b.t, &b.x, h, w, up, .upsample2x);
            std.mem.swap(Buf, &b.x, &b.t);
            h *= 2;
            w *= 2;
        }
    }

    try d.groupNorm(&b.t, &b.x, h * w, ch, dec.norm_out, true);
    // The head writes RGB back into the f32 staging buffer; the host wants f32.
    try d.conv(&b.stage, &b.t, h, w, dec.conv_out, .stride1, 1.0);
    return .{ h, w };
}

/// A resnet over `bufs.x` in place (the result swapped back into `x`). `u` carries both
/// convolution outputs: conv1's is dead once norm2 has read it.
fn resnet(d: anytype, h: usize, w: usize, r: sd_vae.Resnet) !void {
    const b = d.bufs;
    const n = h * w;
    try d.groupNorm(&b.t, &b.x, n, r.in_ch, r.norm1, true);
    try d.conv(&b.u, &b.t, h, w, r.conv1, .stride1, 1.0);
    try d.groupNorm(&b.t, &b.u, n, r.out_ch, r.norm2, true);
    try d.conv(&b.u, &b.t, h, w, r.conv2, .stride1, 1.0);
    if (r.nin) |nin| {
        try convResidual(d, &b.t, &b.x, h, w, nin, .stride1);
        try d.add(b.u, b.t, n * r.out_ch);
    } else {
        try d.add(b.u, b.x, n * r.out_ch);
    }
    std.mem.swap(@TypeOf(b.x), &b.x, &b.u);
}

/// Mid-block attention: one head over all channels, `x += proj(attn(norm(x)))`. q/k/v
/// and the output stay f32: at latent resolution they are tens of MB, not where the
/// memory goes.
fn attn(d: anytype, h: usize, w: usize, ab: sd_vae.AttnBlock) !void {
    const b = d.bufs;
    const n = h * w;
    try d.groupNorm(&b.t, &b.x, n, ab.channels, ab.norm, false);
    try d.conv(&b.aq, &b.t, h, w, ab.q, .stride1, 1.0);
    try d.conv(&b.ak, &b.t, h, w, ab.k, .stride1, 1.0);
    try d.conv(&b.av, &b.t, h, w, ab.v, .stride1, 1.0);
    try d.attnCore(n, ab.channels);
    // Into `t`, not back into `aq`: the residual add is against `x`, at the chain's
    // dtype, and `t` is free once the three projections have read it.
    try d.conv(&b.t, &b.ao, h, w, ab.proj, .stride1, 1.0);
    try d.add(b.x, b.t, n * ab.channels);
}

/// A convolution reading the residual stream: the 1x1 shortcut and the level upsamples.
/// An f16 chain takes no divisor, since an f16 buffer could never have held a value that
/// needs one; f32 and bf16 can, so both keep it.
pub fn convResidual(d: anytype, dst: anytype, src: anytype, h: usize, w: usize, cv: anytype, mode: anytype) !void {
    try d.conv(dst, src, h, w, cv, mode, if (src.dt == .f16) 1.0 else residual_act_div);
}
