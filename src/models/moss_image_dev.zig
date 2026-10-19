//! The MOSS-Image device forward, written once for every GPU backend over an adapter
//! (`moss_image_cuda.Dev`, `moss_image_gpu.Dev`). `body` runs between the adapter's
//! upload of the step's folded modulation table (`DiT.foldedTable`) and latent tokens
//! and its download of the velocity tokens the head writes.
//!
//! The adapter holds `ws`, whose buffers go by the names below, and `seq`, the joint
//! `[image | text]` row count. `modNorm` is `rmsnorm(x) * premul + shift` with both
//! read from the table at row offsets; `qkNormRope` the per-head norms and split-half
//! rope of q and k over the joint table. The op order here is what keeps a backend
//! bit-identical when it moves.

const std = @import("std");
const moss_image = @import("moss_image.zig");

const T = moss_image.Table;

/// Exact power-of-two prescale on V across the attention's f16 cast: Q and K are
/// normed on the way in, V is not. Attention is linear in V, so this costs nothing.
pub const v_div: f32 = 1.0 / 64.0;

const Act = @import("tp_gpu").context.Dt;

/// The dtype the stream is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

pub fn body(d: anytype, model: *const moss_image.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    try d.embed();
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(i);
        try block(d, model.cfg, blk, i);
    }
    try d.head();
}

fn block(d: anytype, cfg: moss_image.Config, blk: *const moss_image.Block, i: usize) !void {
    const ws = d.ws;
    const seq = d.seq;
    const dim = cfg.dim;

    try d.modNorm(ws.x_d, ws.nrm_d, seq, T.premul1(i), T.shift_a);
    const att = blk.dev[0..3];
    try d.gemms(try d.prep(ws.nrm_d, seq, dim, att), &.{ ws.q_d, ws.k_d, ws.v_d }, att);
    try d.qkNormRope(blk.q_norm, blk.k_norm, seq);
    try d.scale(ws.v_d, seq * dim, v_div);
    try d.attention(seq);
    try d.scale(ws.attn_d, seq * dim, 1.0 / v_div);
    try d.linear(ws.dlt_d, ws.attn_d, seq, blk.dev[3]);
    try d.gatedAdd(ws.x_d, ws.dlt_d, seq, T.gate_a);

    try d.modNorm(ws.x_d, ws.nrm_d, seq, T.premul2(i), T.shift_f);
    const ffn = blk.dev[4..6];
    try d.gemms(try d.prep(ws.nrm_d, seq, dim, ffn), &.{ ws.mh_d, ws.mg_d }, ffn);
    try d.situ(ws.mh_d, ws.mg_d, seq * cfg.ffn);
    try d.linear(ws.dlt_d, ws.mh_d, seq, blk.dev[6]);
    try d.gatedAdd(ws.x_d, ws.dlt_d, seq, T.gate_f);
}
