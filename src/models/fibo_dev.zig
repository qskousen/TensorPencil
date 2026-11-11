//! The Fibo 1.5 device forward, written once for every GPU backend over an adapter
//! (`fibo_cuda.Dev`, `fibo_gpu.Dev`). The math is `fibo.DiT.forward`'s, regrouped.
//!
//! The joint stream `ws.x_d` is `[text | image]`. Per conditioning the adapter holds the
//! text rows after `context_embedder` and every block's caption projection; per step it
//! uploads the modulation table (`fibo.Table`) and the latent tokens. Before each block
//! the block's projection is written over the second half of every text row's channels.
//! The final layer runs on the host (`DiT.finalize`). Rows are addressed through
//! `d.rows(buf, row0, n, width)`, a view.

const std = @import("std");
const fibo = @import("fibo.zig");

const T = fibo.Table;

/// Exact power-of-two prescale on V across attention's f16 operands.
pub const v_div: f32 = 1.0 / 64.0;

const Act = @import("tp_gpu").context.Dt;

/// The dtype the stream is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

pub fn body(d: anytype, model: *const fibo.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const ws = d.ws;
    const dim = cfg.dim;
    const nt = d.nText();
    const ni = d.nImg();
    const t: T = .{ .dim = dim, .n_double = cfg.n_double, .n_single = cfg.n_single };

    try d.loadText(try d.rows(ws.x_d, 0, nt, dim));
    try d.embed(try d.rows(ws.x_d, nt, ni, dim));
    for (model.doubles, 0..) |*b, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(i);
        try d.caption(i, try d.rows(ws.x_d, 0, nt, dim));
        try double(d, cfg, b, t.double(i));
    }
    for (model.singles, 0..) |*b, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(cfg.n_double + i);
        try d.caption(cfg.n_double + i, try d.rows(ws.x_d, 0, nt, dim));
        try single(d, cfg, b, t.single(i));
    }
}

/// The two streams' rows of `buf`, `width` wide.
fn split(d: anytype, buf: anytype, width: usize) ![2]@TypeOf(buf) {
    return .{ try d.rows(buf, 0, d.nText(), width), try d.rows(buf, d.nText(), d.nImg(), width) };
}

fn double(d: anytype, cfg: fibo.Config, b: *const fibo.Double, base: usize) !void {
    const ws = d.ws;
    const dim = cfg.dim;
    const seq = d.nText() + d.nImg();
    const x = try split(d, ws.x_d, dim);
    const nrm = try split(d, ws.nrm_d, dim);
    const dlt = try split(d, ws.dlt_d, dim);
    const q = try split(d, ws.q_d, dim);
    const k = try split(d, ws.k_d, dim);
    const v = try split(d, ws.v_d, dim);
    const o = try split(d, ws.attn_d, dim);
    const rows = [2]usize{ d.nText(), d.nImg() };
    // Stream 0 is the text, read through the context weights at the table's `txt` slot.
    const attn = [2]*const fibo.Attn{ &b.txt, &b.img };
    const slot = [2]usize{ base + T.txt * dim, base + T.img * dim };

    for (0..2) |s| {
        try d.lnMod(x[s], nrm[s], rows[s], slot[s] + T.premul_a * dim, slot[s] + T.shift_a * dim);
        const lws = [_]fibo.LinearW{ attn[s].q, attn[s].k, attn[s].v };
        try d.gemms(nrm[s], rows[s], dim, &lws, &.{ q[s], k[s], v[s] });
        try d.headNorm(q[s], attn[s].qnorm, rows[s]);
        try d.headNorm(k[s], attn[s].knorm, rows[s]);
    }
    try attend(d, seq);

    const outs = [2]fibo.LinearW{ b.out_ctx, b.out };
    for (0..2) |s| {
        try d.linear(dlt[s], o[s], rows[s], outs[s]);
        try d.gatedAdd(x[s], dlt[s], rows[s], slot[s] + T.gate_a * dim);
    }

    const ffs = [2][2]fibo.LinearW{ .{ b.ffc_in, b.ffc_out }, .{ b.ff_in, b.ff_out } };
    for (0..2) |s| {
        const hid = ffs[s][0].w.rows;
        try d.lnMod(x[s], nrm[s], rows[s], slot[s] + T.premul_f * dim, slot[s] + T.shift_f * dim);
        try d.linear(ws.mg_d, nrm[s], rows[s], ffs[s][0]);
        try d.gelu(ws.mg_d, rows[s] * hid);
        try d.linear(dlt[s], ws.mg_d, rows[s], ffs[s][1]);
        try d.gatedAdd(x[s], dlt[s], rows[s], slot[s] + T.gate_f * dim);
    }
}

/// Rope on the joint q and k (the text rows sit at position zero), then attention.
fn attend(d: anytype, seq: usize) !void {
    const ws = d.ws;
    const dim = d.model.cfg.dim;
    try d.rope(ws.q_d, seq);
    try d.rope(ws.k_d, seq);
    try d.scale(ws.v_d, seq * dim, v_div);
    try d.attention(seq);
    try d.scale(ws.attn_d, seq * dim, 1.0 / v_div);
}

/// `proj_out` over `[attention | gelu(proj_mlp)]`, assembled in the f32 `cat_d`.
fn single(d: anytype, cfg: fibo.Config, b: *const fibo.Single, base: usize) !void {
    const ws = d.ws;
    const dim = cfg.dim;
    const seq = d.nText() + d.nImg();
    const hid = b.mlp.w.rows;

    try d.lnMod(ws.x_d, ws.nrm_d, seq, base + 1 * dim, base + 0 * dim);
    const lws = [_]fibo.LinearW{ b.attn.q, b.attn.k, b.attn.v, b.mlp };
    try d.gemms(ws.nrm_d, seq, dim, &lws, &.{ ws.q_d, ws.k_d, ws.v_d, ws.mlp_d });
    try d.headNorm(ws.q_d, b.attn.qnorm, seq);
    try d.headNorm(ws.k_d, b.attn.knorm, seq);
    try attend(d, seq);
    try d.gelu(ws.mlp_d, seq * hid);
    try d.concat(ws.attn_d, ws.cat_d, seq, dim, dim + hid, 0);
    try d.concat(ws.mlp_d, ws.cat_d, seq, hid, dim + hid, dim);
    try d.linear(ws.dlt_d, ws.cat_d, seq, b.out);
    try d.gatedAdd(ws.x_d, ws.dlt_d, seq, base + 2 * dim);
}
