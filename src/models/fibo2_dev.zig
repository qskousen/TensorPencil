//! The Fibo 2.0 device forward, written once for every GPU backend over an adapter
//! (`fibo2_cuda.Dev`, `fibo2_gpu.Dev`). The math is `fibo2.DiT`'s, regrouped.
//!
//! `prefix` runs once per conditioning, in f32, against bundles and gist queries the
//! adapter uploaded beforehand: the Perceiver over bundle 0 into the session's gist rows,
//! then each injection block's normed keys and its values (already divided by `v_div`)
//! over its own bundle. `body` runs per step on `[gists | image]` in
//! `ws.x_d`: the refiner on the image rows in place, the gists copied in, then the trunk.
//! Rows are addressed through `d.rows(buf, row0, n, width)`, a view, so no op takes an
//! offset of its own.
//!
//! Modulation comes from two tables (`Tbl`): `.step`, this sigma's, laid out
//! `[refiner blocks | trunk blocks][4 * dim]`; and `.zero`, t = 0's for the trunk blocks,
//! which the gist rows take. Each block entry is `[scale_msa | gate_msa | scale_mlp |
//! gate_mlp]` with the gates tanh'd; how a scale is stored (with or without the norm
//! weight folded in) is the adapter's own.

const std = @import("std");
const fibo2 = @import("fibo2.zig");

/// Exact power-of-two prescale on V across attention's f16 operands: Q and K are
/// normed, V is not.
pub const v_div: f32 = 1.0 / 64.0;

const Act = @import("tp_gpu").context.Dt;

/// The dtype the stream is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

pub const Tbl = enum { step, zero };

/// The per-conditioning buffers `prefix` runs in, all f32: `g` holds the gist queries on
/// entry and the gist rows on return; the rest is scratch of `max(n_gist, text)` rows,
/// and `n_gist` rows of the FFN halves.
pub fn PrefixBufs(comptime Buf: type) type {
    return struct { g: Buf, nrm: Buf, dlt: Buf, q: Buf, k: Buf, v: Buf, o: Buf, mg: Buf, mu: Buf };
}

pub fn prefix(d: anytype, model: *const fibo2.DiT, p: anytype, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const ng = d.nGist();
    const nt = d.textLen();

    for (model.perceiver) |*pl| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.norm(p.g, p.nrm, pl.cross_norm, ng);
        try xattn(d, cfg, &pl.cross, p, p.nrm, ng, try d.bundle(0), nt);
        try d.add(p.g, p.dlt, ng * cfg.dim);

        try d.norm(p.g, p.nrm, pl.self_norm, ng);
        try xattn(d, cfg, &pl.self, p, p.nrm, ng, p.nrm, ng);
        try d.add(p.g, p.dlt, ng * cfg.dim);

        try d.norm(p.g, p.nrm, pl.ffn_norm, ng);
        const fw = [_]fibo2.Weight{ pl.ffn.w1, pl.ffn.w3 };
        try d.gemms(try d.prep(p.nrm, ng, cfg.dim, &fw), &.{ p.mg, p.mu }, &fw);
        try d.siluMulLinear(p.dlt, p.mg, p.mu, ng, pl.ffn.w2);
        try d.add(p.g, p.dlt, ng * cfg.dim);
    }
    try d.norm(p.g, p.g, model.perceiver_norm, ng);

    var j: usize = 0;
    for (model.layers, 0..) |*blk, li| {
        const c = blk.cross orelse continue;
        const k = try d.crossK(j);
        const v = try d.crossV(j);
        const kv = [_]fibo2.Weight{ c.attn.k, c.attn.v };
        try d.gemms(try d.prep(try d.bundle(cfg.bundleOf(li)), nt, cfg.cap_dim, &kv), &.{ k, v }, &kv);
        try d.headNorm(k, c.attn.knorm, nt);
        try d.scale(v, nt * cfg.dim, v_div);
        j += 1;
    }
}

/// `fibo2.Attention` in f32 scratch: q from `x`, k and v from `ctx` (`kv_rows` of them),
/// result into `p.dlt`.
fn xattn(d: anytype, cfg: fibo2.Config, a: *const fibo2.XAttn, p: anytype, x: anytype, rows: usize, ctx: anytype, kv_rows: usize) !void {
    const dim = cfg.dim;
    const kv_dim = if (a.k.cols == dim) dim else cfg.cap_dim;
    try d.linear(p.q, x, rows, a.q);
    const kv = [_]fibo2.Weight{ a.k, a.v };
    try d.gemms(try d.prep(ctx, kv_rows, kv_dim, &kv), &.{ p.k, p.v }, &kv);
    try d.headNorm(p.q, a.qnorm, rows);
    try d.headNorm(p.k, a.knorm, kv_rows);
    try d.scale(p.v, kv_rows * dim, v_div);
    try d.attention(p.q, p.k, p.v, p.o, rows, kv_rows);
    try d.scale(p.o, rows * dim, 1.0 / v_div);
    try d.linear(p.dlt, p.o, rows, a.out);
}

pub fn body(d: anytype, model: *const fibo2.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const ws = d.ws;
    const dim = cfg.dim;
    const ng = d.nGist();
    const ni = d.nImg();
    const seq = ng + ni;
    const img = try d.rows(ws.x_d, ng, ni, dim);

    try d.embed(img);
    d.trace("x_embed", 0, img, ni * dim);
    for (model.noise_refiner, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(.refiner, i);
        try block(d, cfg, blk, img, ni, 0, ng, i * 4 * dim, 0, null);
        d.trace("refiner", i, img, ni * dim);
    }

    try d.loadGist(try d.rows(ws.x_d, 0, ng, dim));
    var inj: usize = 0;
    for (model.layers, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(.trunk, i);
        const step_off = (model.noise_refiner.len + i) * 4 * dim;
        const cross: ?usize = if (blk.cross != null) inj else null;
        if (blk.cross != null) inj += 1;
        try block(d, cfg, blk, ws.x_d, seq, ng, 0, step_off, i * 4 * dim, cross);
        d.trace("layer", i, ws.x_d, seq * dim);
    }
}

/// One block over `x` `[rows][dim]`: rows `[0, split)` modulate from `.zero` at
/// `zero_off`, the rest from `.step` at `step_off`. `rope_row0` is the first row's place
/// in the joint rope table; `cross` is the injection index when the block has one.
fn block(d: anytype, cfg: fibo2.Config, blk: *const fibo2.Block, x: anytype, rows: usize, split: usize, rope_row0: usize, step_off: usize, zero_off: usize, cross: ?usize) !void {
    const ws = d.ws;
    const dim = cfg.dim;

    // x += tanh(gate_msa) * attn_norm2(attn(modulate(attn_norm1(x), scale_msa)))
    try modNorm(d, x, ws.nrm_d, rows, split, step_off, zero_off, 0, blk.attn_norm1);
    const att = [_]fibo2.Weight{ blk.attn.q, blk.attn.k, blk.attn.v };
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, &att), &.{ ws.q_d, ws.k_d, ws.v_d }, &att);
    try d.headNormRope(ws.q_d, blk.attn.qnorm, rows, rope_row0);
    try d.headNormRope(ws.k_d, blk.attn.knorm, rows, rope_row0);
    try d.scale(ws.v_d, rows * dim, v_div);
    try d.attention(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, rows);
    try d.scale(ws.attn_d, rows * dim, 1.0 / v_div);
    try d.linear(ws.dlt_d, ws.attn_d, rows, blk.attn.out);
    try d.norm(ws.dlt_d, ws.dlt_d, blk.attn_norm2, rows);
    try gatedAdd(d, x, rows, split, step_off, zero_off, 1);

    // x += tanh(gate) * cross_attn_norm2(cross_attention(cross_attn_norm1(x), bundle)),
    // the delta in the f32 `q_d` once the attention has read it: the scaled add takes
    // an f32 addend.
    if (cross) |j| {
        const c = blk.cross.?;
        try d.norm(x, ws.nrm_d, c.norm1, rows);
        try d.linear(ws.q_d, ws.nrm_d, rows, c.attn.q);
        try d.headNorm(ws.q_d, c.attn.qnorm, rows);
        try d.attention(ws.q_d, try d.crossK(j), try d.crossV(j), ws.attn_d, rows, d.textLen());
        try d.scale(ws.attn_d, rows * dim, 1.0 / v_div);
        try d.linear(ws.q_d, ws.attn_d, rows, c.attn.out);
        try d.norm(ws.q_d, ws.q_d, c.norm2, rows);
        try d.addScaled(x, ws.q_d, rows * dim, c.gate);
    }

    // x += tanh(gate_mlp) * ffn_norm2(swiglu(modulate(ffn_norm1(x), scale_mlp)))
    try modNorm(d, x, ws.nrm_d, rows, split, step_off, zero_off, 2, blk.ffn_norm1);
    const ffn = [_]fibo2.Weight{ blk.ffn.w1, blk.ffn.w3 };
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, &ffn), &.{ ws.mg_d, ws.mu_d }, &ffn);
    try d.siluMulLinear(ws.dlt_d, ws.mg_d, ws.mu_d, rows, blk.ffn.w2);
    try d.norm(ws.dlt_d, ws.dlt_d, blk.ffn_norm2, rows);
    try gatedAdd(d, x, rows, split, step_off, zero_off, 3);
}

/// The modulated norm per row range; `chunk` is the scale's slot in the block entry.
fn modNorm(d: anytype, x: anytype, nrm: anytype, rows: usize, split: usize, step_off: usize, zero_off: usize, chunk: usize, norm_w: []const f32) !void {
    const dim = d.model.cfg.dim;
    if (split > 0) try d.modNorm(try d.rows(x, 0, split, dim), try d.rows(nrm, 0, split, dim), split, .zero, zero_off + chunk * dim, norm_w);
    try d.modNorm(try d.rows(x, split, rows - split, dim), try d.rows(nrm, split, rows - split, dim), rows - split, .step, step_off + chunk * dim, norm_w);
}

/// `x += gate * dlt` per row range; `chunk` is the gate's slot in the block entry.
fn gatedAdd(d: anytype, x: anytype, rows: usize, split: usize, step_off: usize, zero_off: usize, chunk: usize) !void {
    const dim = d.model.cfg.dim;
    const dlt = d.ws.dlt_d;
    if (split > 0) try d.gatedAdd(try d.rows(x, 0, split, dim), try d.rows(dlt, 0, split, dim), split, .zero, zero_off + chunk * dim);
    try d.gatedAdd(try d.rows(x, split, rows - split, dim), try d.rows(dlt, split, rows - split, dim), rows - split, .step, step_off + chunk * dim);
}
