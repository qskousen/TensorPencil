//! The SD UNet device forward, written once for every GPU backend over an adapter
//! (`sd_unet_cuda.Dev`, `sd_unet_gpu.Dev`). `body` runs between the adapter's upload of
//! the latent into `ws.xin` and its download of `ws.eps`, inside one recording batch.
//!
//! The adapter holds `u` (the UNet), `ws` (the workspace, whose activation buffers go by
//! the names below and which this file swaps between) and `ctx_seq`, and supplies the
//! ops: `groupNorm`, `conv` (LoRA hits included), `resConv` (a ResBlock's first conv
//! plus its timestep projection, which one backend folds into the bias and the other
//! adds after), `add`, `concatSkip`, `setSkip`, `layerNorm`, `gemm`, `gemms` (bias-free
//! linears sharing one input), `crossKv` (the K/V projections off the text
//! conditioning), `selfAttn`, `crossAttn` and `geglu`. Each adapter keeps its
//! backend's own dispatch sequence, which is what keeps a backend's output
//! bit-identical when it moves onto this file.

const std = @import("std");
const sd_unet = @import("sd_unet.zig");

pub const SampleMode = @import("tp_ops").conv.SampleMode;

pub fn body(d: anytype, lat_h: usize, lat_w: usize, cancel: ?*std.atomic.Value(bool)) !void {
    const u = d.u;
    const ws = d.ws;
    const Buf = @TypeOf(ws.cur);
    var h = lat_h;
    var w = lat_w;
    var ch = u.cfg.model_channels;
    // ResBlocks in `ResBlockIter`'s order, which is the order below.
    var rb_ord: usize = 0;

    try d.conv(&ws.cur, &ws.xin, h, w, u.stem, .stride1, null);
    try d.setSkip(0, &ws.cur, h * w * ch, ch, h, w);

    for (u.input_stages, 0..) |stage, si| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (stage.res) |rb| {
            try res(d, rb, rb_ord, h, w);
            rb_ord += 1;
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            ch = rb.out_ch;
        }
        if (stage.attn) |st| try spatial(d, st, h, w);
        if (stage.sample_kind == .down) {
            try d.conv(&ws.alt, &ws.cur, h, w, stage.sample.?, .stride2, null);
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            h = (h + 1) / 2;
            w = (w + 1) / 2;
        }
        try d.setSkip(si + 1, &ws.cur, h * w * ch, ch, h, w);
    }

    try res(d, u.mid_res1, rb_ord, h, w);
    rb_ord += 1;
    std.mem.swap(Buf, &ws.cur, &ws.alt);
    ch = u.mid_res1.out_ch;
    try spatial(d, u.mid_attn, h, w);
    try res(d, u.mid_res2, rb_ord, h, w);
    rb_ord += 1;
    std.mem.swap(Buf, &ws.cur, &ws.alt);
    ch = u.mid_res2.out_ch;

    var skip_top = u.input_stages.len + 1;
    for (u.output_stages) |stage| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        skip_top -= 1;
        const skip_ch = ws.skip_ch[skip_top];
        // Current first, then the popped skip, along channels into `alt`.
        try d.concatSkip(ws.alt, ws.cur, ch, ws.skips[skip_top], skip_ch, h * w);
        std.mem.swap(Buf, &ws.cur, &ws.alt);
        ch += skip_ch;
        const rb = stage.res.?;
        try res(d, rb, rb_ord, h, w);
        rb_ord += 1;
        std.mem.swap(Buf, &ws.cur, &ws.alt);
        ch = rb.out_ch;
        if (stage.attn) |st| try spatial(d, st, h, w);
        if (stage.sample_kind == .up) {
            // LDM's `Upsample`: nearest up then a 3x3, fused into the patch gather. The
            // target is the grid the NEXT skip was stored at, not twice this one: the
            // encoder halved rounding up, so `2 * h` overshoots an odd dimension by a
            // row. The last output stage never upsamples, so a shallower skip exists.
            std.debug.assert(skip_top > 0);
            const target = ws.skip_hw[skip_top - 1];
            try d.conv(&ws.alt, &ws.cur, h, w, stage.sample.?, .upsample2x, target);
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            h = target[0];
            w = target[1];
        }
    }

    try d.groupNorm(&ws.t1, &ws.cur, h * w, ch, u.out_norm, true);
    try d.conv(&ws.eps, &ws.t1, h, w, u.out_conv, .stride1, null);
}

/// A ResBlock from `ws.cur` into `ws.alt` (which it may swap with its own scratch to
/// get there; `t2` is scratch either way). `cur` must survive to the residual add, so
/// the first norm is out of place.
fn res(d: anytype, rb: sd_unet.ResBlock, ordinal: usize, h: usize, w: usize) !void {
    const ws = d.ws;
    const n = h * w;
    try d.groupNorm(&ws.t1, &ws.cur, n, rb.in_ch, rb.in_norm, true);
    try d.resConv(&ws.alt, &ws.t1, h, w, rb, ordinal);
    try d.groupNorm(&ws.t1, &ws.alt, n, rb.out_ch, rb.out_norm, true);
    try d.conv(&ws.t2, &ws.t1, h, w, rb.out_conv, .stride1, null);
    const out_n = n * rb.out_ch;
    if (rb.skip) |sk| {
        // `alt` held the first conv's output, which the second norm has consumed.
        try d.conv(&ws.alt, &ws.cur, h, w, sk, .stride1, null);
        try d.add(ws.alt, ws.t2, out_n);
    } else {
        try d.add(ws.t2, ws.cur, out_n);
        std.mem.swap(@TypeOf(ws.cur), &ws.t2, &ws.alt);
    }
}

/// A SpatialTransformer, in place on `ws.cur` (it ends in an outer residual add).
fn spatial(d: anytype, st: sd_unet.SpatialTransformer, h: usize, w: usize) !void {
    const ws = d.ws;
    const cfg = d.u.cfg;
    const n = h * w;
    const ch = st.channels;
    const heads = cfg.headsAt(ch);
    const hd = ch / heads;

    try d.groupNorm(&ws.nb, &ws.cur, n, ch, st.norm, false);
    try proj(d, &ws.stream, &ws.nb, h, w, st.proj_in);

    for (st.blocks) |b| {
        // attn1: self-attention over pixels.
        try d.layerNorm(&ws.nb, &ws.stream, n, ch, b.norm1);
        try d.gemms(&ws.nb, n, &.{ &ws.q, &ws.k, &ws.v }, &.{ b.attn1.q, b.attn1.k, b.attn1.v });
        try d.selfAttn(n, heads, hd);
        try d.gemm(&ws.nb, &ws.ao, n, b.attn1.out.w, b.attn1.out.b);
        try d.add(ws.stream, ws.nb, n * ch);

        // attn2: cross-attention onto the text conditioning, `ctx_seq` keys, not `n`.
        try d.layerNorm(&ws.nb, &ws.stream, n, ch, b.norm2);
        try d.gemm(&ws.q, &ws.nb, n, b.attn2.q, null);
        try d.crossKv(b.attn2.k, b.attn2.v);
        try d.crossAttn(n, heads, hd);
        try d.gemm(&ws.nb, &ws.ao, n, b.attn2.out.w, b.attn2.out.b);
        try d.add(ws.stream, ws.nb, n * ch);

        // ff: GEGLU, whose halves are (value, gate) and whose gate is erf-GELU.
        try d.layerNorm(&ws.nb, &ws.stream, n, ch, b.norm3);
        try d.gemm(&ws.ff, &ws.nb, n, b.ff_proj.w, b.ff_proj.b);
        try d.geglu(ws.ff, ws.gated, n, b.inner);
        try d.gemm(&ws.nb, &ws.gated, n, b.ff_out.w, b.ff_out.b);
        try d.add(ws.stream, ws.nb, n * ch);
    }

    try proj(d, &ws.nb, &ws.stream, h, w, st.proj_out);
    try d.add(ws.cur, ws.nb, n * ch);
}

/// A SpatialTransformer's in/out projection: a 1x1 convolution in SD1.5, an
/// `nn.Linear` in SDXL. On channel-last activations both are the same GEMM.
fn proj(d: anytype, dst: anytype, src: anytype, h: usize, w: usize, p: sd_unet.Proj) !void {
    switch (p) {
        .conv => |cv| try d.conv(dst, src, h, w, cv, .stride1, null),
        .linear => |lin| try d.gemm(dst, src, h * w, lin.w, lin.b),
    }
}
