//! The Mage-VAE decode, written once for every GPU backend over an adapter
//! (`mage_vae_cuda.Dev`, `mage_vae_gpu.Dev`). `decode` runs between the adapter's
//! upload of the latent (`bufs.stage`) and its constants (`bufs.mods`, `bufs.dct`) and
//! its download of `bufs.rgb`.
//!
//! The adapter holds `bufs`, every buffer pre-sized: `x` the running `[n][hidden]`
//! activation, `t`/`u` scratches of that shape, `wide` the DiCo MLP's, `cond` the CoD
//! output, `cat` the `s_embedder` concatenation, `aq`/`ak`/`av`/`ao` the window
//! projections, and the NeRF band's `y`, `feat`, `nx`, `nt`, `cvec`, `nmod`, `pix`.
//! `windows` is the 32x32 windowed attention core (`aq`/`ak`/`av` to `ao`), `freeAttn`
//! hands its scratch back, and `linear` reads its source from element `x_off`.
//!
//! One batch for the CoD, one for the `s_embedder`, one per DiCo block (a cancel
//! boundary, and one over all of them would hold every intermediate), one per band.

const std = @import("std");
const mage_vae = @import("mage_vae.zig");
const Dt = @import("tp_gpu").context.Dt;

const H = mage_vae.hidden;
const HX = mage_vae.hidden_x;
const FFN = mage_vae.ffn;
const PA = mage_vae.patch_area;
const NERF_IN = mage_vae.nerf_in;

/// Latent cells one NeRF band covers. The pathway is per-cell independent, so this
/// changes only the peak: the widest intermediate is `band * 256 * 99` floats, ~66 MB
/// at 704.
pub const nerf_band: usize = 704;

pub fn decode(d: anytype, model: *const mage_vae.MageVae, lat_h: usize, lat_w: usize, cancel: ?*std.atomic.Value(bool)) !void {
    const b = d.bufs;
    const n = lat_h * lat_w;
    errdefer d.abortBatch();

    try d.beginBatch();
    try cod(d, model, lat_h, lat_w);
    try d.endBatch();
    // Dead from here, and it would otherwise stay resident through the per-pixel
    // pathway, which is where the memory goes.
    d.freeAttn();

    // `proj1` is bias-free and decode's noise is zero, so the image half of the
    // concatenation is zeros; written rather than folded away, so this and the CPU
    // reference read the same.
    const cat_w = mage_vae.bottleneck + H;
    try d.zero(b.cat, n * cat_w);
    try d.beginBatch();
    try d.concatCh(b.cond, b.cat, n, H, cat_w, mage_vae.bottleneck);
    try d.conv(&b.x, &b.cat, lat_h, lat_w, model.s_proj2);
    d.trace("s_embed", b.x, n * H);
    try d.endBatch();

    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.beginBatch();
        try dico(d, lat_h, lat_w, blk, i);
        d.trace("dico x", b.x, n * H);
        try d.endBatch();
    }

    const band: usize = @min(nerf_band, n);
    var l0: usize = 0;
    while (l0 < n) : (l0 += band) {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.beginBatch();
        try nerfBand(d, model, l0, @min(band, n - l0), lat_w);
        try d.endBatch();
    }
    if (mag.on) mag.report();
}

/// `TP_VAE_MAG`: the largest |x| each traced buffer reaches over a decode, which is
/// what decides whether it may be stored at f16. An adapter's `trace` syncs and hands
/// the buffer's raw bytes to `record`.
pub const mag = struct {
    pub var on: bool = false;
    var names: [24][]const u8 = undefined;
    var maxes: [24]f32 = undefined;
    var len: usize = 0;

    pub fn record(name: []const u8, raw: []const u8, dt: Dt) void {
        var m: f32 = 0;
        for (0..raw.len / dt.bytes()) |i| {
            const v: f32 = switch (dt) {
                .f32 => @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little)),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)))),
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16),
            };
            m = if (std.math.isFinite(v)) @max(m, @abs(v)) else std.math.inf(f32);
        }
        for (names[0..len], maxes[0..len]) |nm, *mx| if (std.mem.eql(u8, nm, name)) {
            mx.* = @max(mx.*, m);
            return;
        };
        if (len == names.len) return;
        names[len] = name;
        maxes[len] = m;
        len += 1;
    }

    fn report() void {
        for (names[0..len], maxes[0..len]) |nm, mx| std.debug.print("mage vae max|{s}| {d:.1}\n", .{ nm, mx });
        len = 0;
    }
};

/// `conv_in`, three resnets with two windowed attentions between them, and the head,
/// leaving the conditioning in `bufs.cond`.
pub fn cod(d: anytype, model: *const mage_vae.MageVae, lat_h: usize, lat_w: usize) !void {
    const b = d.bufs;
    const n = lat_h * lat_w;
    try d.conv(&b.x, &b.stage, lat_h, lat_w, model.cod.conv_in);
    d.trace("conv_in", b.x, n * H);
    try resnet(d, lat_h, lat_w, model.cod.r0);
    d.trace("r0", b.x, n * H);
    try winAttn(d, lat_h, lat_w, model.cod.a0);
    d.trace("a0", b.x, n * H);
    try resnet(d, lat_h, lat_w, model.cod.r1);
    d.trace("r1", b.x, n * H);
    try winAttn(d, lat_h, lat_w, model.cod.a1);
    d.trace("a1", b.x, n * H);
    try resnet(d, lat_h, lat_w, model.cod.r2);
    d.trace("r2", b.x, n * H);
    try d.groupNorm(&b.t, &b.x, n, model.cod.norm_out, true);
    try d.conv(&b.cond, &b.t, lat_h, lat_w, model.cod.conv_out);
}

/// `x += conv2(silu(norm2(conv1(silu(norm1(x))))))`, width-preserving.
fn resnet(d: anytype, h: usize, w: usize, r: mage_vae.Resnet) !void {
    const b = d.bufs;
    const n = h * w;
    try d.groupNorm(&b.t, &b.x, n, r.norm1, true);
    try d.conv(&b.u, &b.t, h, w, r.conv1);
    try d.groupNorm(&b.t, &b.u, n, r.norm2, true);
    try d.conv(&b.u, &b.t, h, w, r.conv2);
    try d.add(b.x, b.u, n * H);
}

/// `x += proj(windowed_attention(norm(x)))`. Each 32x32 window attends to itself
/// alone, and out-of-range samples REPLICATE the edge, which is the reference's
/// padding and is not the same as attending over a smaller window.
fn winAttn(d: anytype, h: usize, w: usize, a: mage_vae.WinAttn) !void {
    const b = d.bufs;
    const n = h * w;
    try d.groupNorm(&b.t, &b.x, n, a.norm, false);
    try d.conv(&b.aq, &b.t, h, w, a.q);
    try d.conv(&b.ak, &b.t, h, w, a.k);
    try d.conv(&b.av, &b.t, h, w, a.v);
    try d.windows(h, w);
    try d.conv(&b.t, &b.ao, h, w, a.proj);
    try d.add(b.x, b.t, n * H);
}

/// One DiCo block, in place on `bufs.x`, its modulation rows from the constant table.
fn dico(d: anytype, h: usize, w: usize, blk: *const mage_vae.DiCo, i: usize) !void {
    const b = d.bufs;
    const n = h * w;
    const base = i * 6 * H;

    // x += gate1 * conv3(ca(gelu(dw(conv1((1 + scale1) * layernorm(x) + shift1)))))
    try d.lnMod(b.x, b.t, n, base + 0 * H, base + 1 * H);
    try d.conv(&b.u, &b.t, h, w, blk.conv1);
    try d.depthwise3(b.t, b.u, blk.dw, blk.dw_b, h, w);
    try d.gelu(b.t, n * H);
    // Channel attention: the spatial mean, one 1x1, a sigmoid, broadcast back.
    try d.colMean(b.mean, b.t, n);
    try d.linear(b.gate, b.mean, 0, 1, blk.ca, H, H);
    try d.mulColsSigmoid(b.t, b.gate, n);
    d.trace("dico ca", b.t, n * H);
    try d.conv(&b.u, &b.t, h, w, blk.conv3);
    d.trace("dico conv3", b.u, n * H);
    try d.gatedAdd(b.x, b.u, n * H, base + 2 * H);

    // x += gate2 * conv5(gelu(conv4((1 + scale2) * layernorm(x) + shift2)))
    try d.lnMod(b.x, b.t, n, base + 3 * H, base + 4 * H);
    try d.conv(&b.wide, &b.t, h, w, blk.conv4);
    try d.gelu(b.wide, n * FFN);
    d.trace("dico wide", b.wide, n * FFN);
    try d.conv(&b.u, &b.wide, h, w, blk.conv5);
    d.trace("dico conv5", b.u, n * H);
    try d.gatedAdd(b.x, b.u, n * H, base + 5 * H);
}

/// One band of latent cells through the per-pixel MLP, writing their 16x16 tiles.
fn nerfBand(d: anytype, model: *const mage_vae.MageVae, l0: usize, nb: usize, lat_w: usize) !void {
    const b = d.bufs;
    const rows = nb * PA;
    const off = l0 * H;

    // The patch's per-pixel channels, and the per-pixel conditioning.
    try d.linear(b.y, b.cond, off, nb, model.y_embedder_x, HX * PA, H);
    try d.linear(b.cvec, b.x, off, nb, model.cond_embed, HX * PA, H);
    d.trace("nerf y", b.y, nb * HX * PA);
    d.trace("nerf cvec", b.cvec, nb * HX * PA);

    try d.nerfFeat(b.feat, b.y, nb);
    d.trace("nerf feat", b.feat, rows * NERF_IN);
    try d.linear(b.nx, b.feat, 0, rows, model.x_embedder, HX, NERF_IN);
    try d.linear(b.nt, b.nx, 0, rows, model.input_proj, HX, HX);
    try d.copy(b.nx, b.nt, rows * HX);
    d.trace("nerf in", b.nx, rows * HX);

    for (model.res_blocks) |*r| {
        // `cvec` IS the conditioning `y`; silu it into the scratch rather than in
        // place, because every block reads the same untouched vector.
        try d.copy(b.nt, b.cvec, rows * HX);
        try d.silu(b.nt, rows * HX);
        try d.linear(b.nmod, b.nt, 0, rows, r.adaln, 3 * HX, HX);
        d.trace("nerf mod", b.nmod, rows * 3 * HX);
        try d.layerNorm(b.nx, b.nt, r.in_ln, rows, HX);
        try d.modulatePerRow(b.nt, b.nmod, rows, HX);
        d.trace("nerf modulated", b.nt, rows * HX);
        try d.linear(b.feat, b.nt, 0, rows, r.fc1, HX, HX);
        try d.silu(b.feat, rows * HX);
        d.trace("nerf fc1", b.feat, rows * HX);
        try d.linear(b.nt, b.feat, 0, rows, r.fc2, HX, HX);
        d.trace("nerf fc2", b.nt, rows * HX);
        try d.gatedAddPerRow(b.nx, b.nt, b.nmod, rows, HX);
        d.trace("nerf x", b.nx, rows * HX);
    }

    try d.rmsNorm(b.nx, model.final_norm, rows, HX);
    try d.linear(b.pix, b.nx, 0, rows, model.final_lin, 3, HX);
    d.trace("nerf pix", b.pix, rows * 3);
    try d.patchScatter(b.rgb, b.pix, nb, lat_w, l0);
}
