//! The Mage-Flow (double-stream MMDiT) device forward, written once for every GPU
//! backend over an adapter (`mageflow_cuda.Dev`, `mageflow_gpu.Dev`). `body` runs between
//! the adapter's upload of the step's modulation table and tokens and its download of the
//! canvas rows.
//!
//! Text and image keep their own modulation, norms and MLPs and meet only inside the
//! attention, as `[text | image]`. How a backend lays the two streams out is its own: the
//! adapter's `streams` names where each lives, `qkPrepare` norms, ropes (image only:
//! text sits at position 0, whose rotation is the identity) and assembles the joint q/k/v,
//! and `attnSplit` hands back each stream's part of the attention output. The op order
//! here is what keeps a backend bit-identical when it moves.

const std = @import("std");
const mageflow = @import("mageflow.zig");

const F = mageflow.features;
const mlp_dim = mageflow.mlp_dim;

/// Exact power-of-two prescale on V across the attention's f16 cast: V is the one
/// unnormed operand, and on a REAL conditioning it outgrows f16 (509538 at 1024²) while a
/// synthetic one stays green, so a device check alone cannot see it. Attention is linear
/// in V and a power of two costs no precision.
pub const v_div: f32 = 1.0 / 128.0;

/// The dtype both residual streams, their norm outputs, deltas and MLP buffers are
/// stored at when `want` is asked for, on either backend; the attention block stays f32,
/// so `v_div` and the attention's own f16 cast are untouched.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

const Act = @import("tp_gpu").context.Dt;

/// Which stream a GEMM serves; only a profiler cares.
pub const Stream = enum { txt, img };

pub fn body(d: anytype, model: *const mageflow.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    try d.embed();
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(i);
        try block(d, blk, i);
        d.traceBlock(i);
    }
    // The final layer on the canvas rows alone: it is row-wise, so this is what
    // ComfyUI's "whole stream, then slice" computes.
    try d.final(model.blocks.len * 12 * F);
}

fn block(d: anytype, blk: *const mageflow.Block, i: usize) !void {
    const s = d.streams();
    const ws = d.ws;
    const seq = s.seq_txt + s.n_tok;
    // Per block: the image stream's six chunks, then the text stream's six.
    const img_m = i * 12 * F;
    const txt_m = img_m + 6 * F;

    try d.lnMod(s.txt_x, ws.ntxt_d, txt_m + 0 * F, txt_m + 1 * F, s.seq_txt);
    try d.lnMod(s.img_x, ws.nimg_d, img_m + 0 * F, img_m + 1 * F, s.n_tok);
    const tq = [_]mageflow.LinearW{ blk.attn.txt_q, blk.attn.txt_k, blk.attn.txt_v };
    try d.gemms(.txt, ws.ntxt_d, s.seq_txt, &tq, &.{ s.tq, s.tk, s.tv });
    const iq = [_]mageflow.LinearW{ blk.attn.q, blk.attn.k, blk.attn.v };
    try d.gemms(.img, ws.nimg_d, s.n_tok, &iq, &.{ s.iq, s.ik, s.iv });
    try d.qkPrepare(blk);

    d.trace("  q", i, ws.q_d, seq * F);
    d.trace("  v", i, ws.v_d, seq * F);
    try d.scale(ws.v_d, seq * F, v_div);
    try d.attention(seq);
    try d.scale(ws.attn_d, seq * F, 1.0 / v_div);
    d.trace("  attn", i, ws.attn_d, seq * F);
    const at = try d.attnSplit();
    try d.linear(.txt, ws.dtxt_d, at.txt, s.seq_txt, blk.attn.txt_o);
    try d.linear(.img, ws.dimg_d, at.img, s.n_tok, blk.attn.o);
    try d.gatedAdd(s.txt_x, ws.dtxt_d, s.seq_txt, txt_m + 2 * F);
    try d.gatedAdd(s.img_x, ws.dimg_d, s.n_tok, img_m + 2 * F);

    try d.lnMod(s.txt_x, ws.ntxt_d, txt_m + 3 * F, txt_m + 4 * F, s.seq_txt);
    try d.lnMod(s.img_x, ws.nimg_d, img_m + 3 * F, img_m + 4 * F, s.n_tok);
    try d.linear(.txt, ws.mt_d, ws.ntxt_d, s.seq_txt, blk.txt_mlp.in);
    try d.gelu(ws.mt_d, s.seq_txt * mlp_dim);
    try d.linear(.txt, ws.dtxt_d, ws.mt_d, s.seq_txt, blk.txt_mlp.out);
    try d.linear(.img, ws.mi_d, ws.nimg_d, s.n_tok, blk.img_mlp.in);
    try d.gelu(ws.mi_d, s.n_tok * mlp_dim);
    try d.linear(.img, ws.dimg_d, ws.mi_d, s.n_tok, blk.img_mlp.out);
    try d.gatedAdd(s.txt_x, ws.dtxt_d, s.seq_txt, txt_m + 5 * F);
    try d.gatedAdd(s.img_x, ws.dimg_d, s.n_tok, img_m + 5 * F);
}
