//! The SenseNova U1.5 device passes, written once for every GPU backend over an adapter
//! (`sensenova_cuda.Dev`, `sensenova_gpu.Dev`).
//!
//! `prefix` runs the base MoT copy over the prompt and hands each layer's K/V to the
//! adapter (`keepKv`); `canvas` runs the vision tower, the `_mot_gen` trunk over the
//! prefix cache concatenated with its own K/V, and `fm_head`, between the adapter's
//! upload of the patch rows and time vector and its download of the prediction. Both go
//! through one `layer`.
//!
//! The adapter holds `b`, the layer's buffers (`x`, `nrm`, `dlt`, `q`, `k`, `v`, `kcat`,
//! `vcat`, `attn`, `mg`, `mu`; in the prefix pass `kcat`/`vcat` are `k`/`v`), and the
//! pass's positions, rope tables and attention form. `qkPrepare` is the per-head span
//! norms and three-span rope; `copy` counts elements and must stay inside the open
//! batch. The op order here is what keeps a backend bit-identical when it moves.
//!
//! The canvas stream (`x`, `nrm`, `mg`, `mu`) is stored at the denoiser's activation
//! dtype; the sublayer delta, the attention planes, the vision tower and the fm_head
//! stay f32, so `copy` casts at the two ends of the trunk.

const std = @import("std");
const sensenova = @import("sensenova.zig");

const Config = sensenova.Config;
const Stream = sensenova.Stream;

/// The dtype the canvas stream is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

/// The exact power of two an f16 canvas stream is stored divided by: the residual
/// reaches ~7e4 by the last layer of the last step, past f16's 65504. Every reader of
/// the stream is an RMSNorm, which takes `eps / div^2` and so computes the same thing.
/// The down projection's own output reaches ~5e4, which is why the delta stays f32.
pub fn streamDiv(dt: Act) f32 {
    return if (dt == .f16) 16 else 1;
}

/// `x += dlt / div`, `x` at the stream's dtype and `dlt` f32.
fn addDelta(d: anytype, x: anytype, dlt: anytype, n: usize, div: f32) !void {
    if (x.dt == dlt.dt and div == 1) return d.add(x, dlt, n);
    try d.addScaled(x, dlt, n, 1 / div);
}

const Act = @import("tp_gpu").context.Dt;

/// One trunk layer over `rows` query rows against `seq_kv` keys, of which the first
/// `prefix_rows` are already in `kcat`/`vcat`.
pub fn layer(d: anytype, cfg: Config, s: *const Stream, rows: usize, seq_kv: usize, prefix_rows: usize) !void {
    const b = d.b;
    const dim = cfg.dim;
    const kv_dim = cfg.kvDim();
    const div = streamDiv(b.x.dt);
    const eps = cfg.norm_eps / (div * div);

    try d.rmsNorm(b.x, b.nrm, s.in_norm, rows, eps);
    try d.gemms(b.nrm, rows, dim, &.{ s.q, s.k, s.v }, &.{ b.q, b.k, b.v });
    try d.qkPrepare(b.q, rows, cfg.n_heads, s.q_norm_pair);
    try d.qkPrepare(b.k, rows, cfg.n_kv_heads, s.k_norm_pair);
    d.trace("q", 0, b.q, rows * cfg.qDim());
    d.trace("k", 0, b.k, rows * kv_dim);
    d.trace("v", 0, b.v, rows * kv_dim);

    // The generation stream attends over the prefix's keys and then its own, with no
    // mask; the prefix pass attends over itself alone.
    if (prefix_rows != 0) {
        try d.copy(b.kcat, prefix_rows * kv_dim, b.k, 0, rows * kv_dim);
        try d.copy(b.vcat, prefix_rows * kv_dim, b.v, 0, rows * kv_dim);
    }
    try d.attention(rows, seq_kv);
    d.trace("attn", 0, b.attn, rows * cfg.qDim());
    try d.linear(b.dlt, b.attn, rows, s.o);
    try addDelta(d, b.x, b.dlt, rows * dim, div);
    d.trace("attn_res", 0, b.x, rows * dim);

    try d.rmsNorm(b.x, b.nrm, s.post_norm, rows, eps);
    try d.gemms(b.nrm, rows, dim, &.{ s.gate, s.up }, &.{ b.mg, b.mu });
    try d.siluMul(b.mg, b.mu, rows * cfg.inter);
    try d.linear(b.dlt, b.mg, rows, s.down);
    try addDelta(d, b.x, b.dlt, rows * dim, div);
}

/// The prompt pass: one batch per layer, its K/V handed to the adapter after each, so
/// the whole cache never sits on the device beside the generation stack. The caller
/// uploads the f32 embedding into `dlt`; the stream takes it at its own dtype and scale.
pub fn prefix(d: anytype, model: *const sensenova.Model, seq: usize, cancel: ?*std.atomic.Value(bool)) !void {
    const b = d.b;
    const n = seq * model.cfg.dim;
    const div = streamDiv(b.x.dt);
    if (div != 1) try d.scaleBy(b.dlt, 1 / div, n);
    try d.copy(b.x, 0, b.dlt, 0, n);
    for (model.layers, 0..) |*l, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (li + 1 < model.layers.len) d.prefetch(&model.layers[li + 1].base);
        try d.beginBatch();
        errdefer d.abortBatch();
        try layer(d, model.cfg, &l.base, seq, seq, 0);
        try d.endBatch();
        try d.keepKv(li);
    }
}

/// One generation forward from the uploaded patch rows (`ws.patch`) and time vector
/// (`ws.tvec`) to the channel-last prediction in `ws.fm_a`.
pub fn canvas(d: anytype, model: *const sensenova.Model, sess: anytype, ws: anytype, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const ch = sensenova.latent_channels;
    const cells = sess.gh * sess.gw;
    const n_img = sess.n_img;

    try d.patchEmbed(ws.cells, ws.patch, cells, model.vision_gen.patch);
    try d.gelu(ws.cells, cells * cfg.vis_dim);
    try d.ropeInter(ws.cells, sess.vis_x, sess.tab_vx, cells, cfg.vis_dim, 0);
    try d.ropeInter(ws.cells, sess.vis_y, sess.tab_vy, cells, cfg.vis_dim, cfg.vis_dim / 2);
    d.trace("cells", 0, ws.cells, cells * cfg.vis_dim);
    const typed = ws.img.dt != .f32;
    const vis = if (typed) ws.fm_a else ws.img;
    try d.convStride(&ws.patch, vis, ws.cells, sess.gh, sess.gw, model.vision_gen.dense);
    d.trace("vision", 0, vis, n_img * cfg.dim);
    try d.addBiasRows(vis, ws.tvec, n_img, cfg.dim);
    const div = streamDiv(ws.img.dt);
    if (div != 1) try d.scaleBy(vis, 1 / div, n_img * cfg.dim);
    if (typed) try d.copy(ws.img, 0, vis, 0, n_img * cfg.dim);
    d.trace("trunk_in", 0, ws.img, n_img * cfg.dim);

    const kv_elems = sess.prefix_seq * cfg.kvDim();
    for (model.layers, 0..) |*l, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (li + 1 < model.layers.len) d.prefetch(&model.layers[li + 1].gen);
        try d.copy(ws.kcat, 0, sess.prefix_kv, (li * 2) * kv_elems, kv_elems);
        try d.copy(ws.vcat, 0, sess.prefix_kv, (li * 2 + 1) * kv_elems, kv_elems);
        try layer(d, cfg, &l.gen, n_img, sess.prefix_seq + n_img, sess.prefix_seq);
        d.trace("layer", li, ws.img, n_img * cfg.dim);
    }
    try d.rmsNorm(ws.img, ws.img, model.norm_gen, n_img, cfg.norm_eps / (div * div));
    d.trace("norm", 0, ws.img, n_img * cfg.dim);

    // The token plane IS the channel-last input the first shuffle reads.
    const th = sess.th;
    const tw = sess.tw;
    if (typed) try d.copy(ws.fm_b, 0, ws.img, 0, n_img * cfg.dim);
    try d.pixelShuffle(ws.fm_a, if (typed) ws.fm_b else ws.img, th, tw, cfg.fmC1(), 2);
    try d.conv3x3(&ws.patch, &ws.fm_b, &ws.fm_a, th * 2, tw * 2, model.fm1);
    try d.gelu(ws.fm_b, th * 2 * tw * 2 * cfg.fmC1());
    try d.pixelShuffle(ws.fm_a, ws.fm_b, th * 2, tw * 2, cfg.fmC2(), 2);
    try d.conv3x3(&ws.patch, &ws.fm_b, &ws.fm_a, th * 4, tw * 4, model.fm2);
    try d.pixelShuffle(ws.fm_a, ws.fm_b, th * 4, tw * 4, ch, 8);
    d.trace("pred", 0, ws.fm_a, sess.ph * sess.pw * ch);
}
