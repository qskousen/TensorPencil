//! The Qwen-Image 2.1 device forward, written once for every GPU backend over an
//! adapter (`qwen_image21_cuda.Dev`, `qwen_image21_gpu.Dev`).
//!
//! Two passes, because nothing before the target attends to it and the prefix is
//! modulated at t = 0: `prefix` runs once per conditioning over the text (and
//! reference) rows, block-causal, and leaves each block's K and V in the adapter's
//! cache; `target` runs per step over the image rows alone, against `[cached prefix |
//! own]` with no mask. The math is the CPU forward's, regrouped.
//!
//! `d.ws` holds `x_d` (the pass's rows), `nrm_d`, `dlt_d`, `q_d`, `kj_d` / `vj_d`
//! (joint keys and values, prefix rows first), `attn_d`, `mg_d` / `mu_d` (the FFN's
//! gate and up halves) and `mv_d` (the table, `Table`).

const std = @import("std");
const qwen_image21 = @import("qwen_image21.zig");

/// Exact power-of-two prescale on V across the attention's f16 cast: Q and K are
/// normed on the way in, V is not. Attention is linear in V, so this costs nothing.
pub const v_div: f32 = 1.0 / 64.0;

const Act = @import("tp_gpu").context.Dt;

/// The dtype the stream is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

/// Row offsets, in units of `dim`, of the vectors a step's table holds. Premuls carry
/// the `(1 + scale)` fold and gates are already tanh'd; `zero` is the shift the
/// modulated norms do not have.
pub const Table = struct {
    pub const premul_a = 0;
    pub const gate_a = 1;
    pub const premul_f = 2;
    pub const gate_f = 3;
    pub const head_premul = 4;
    pub const zero = 5;
    pub const len = 6;

    /// The table from one sigma's `Mod`: its t = 0 rows when `at_zero`, else the sampled t's.
    pub fn fill(out: []f32, mod: qwen_image21.Mod, at_zero: bool) void {
        const d = mod.head_scale.len;
        std.debug.assert(out.len == len * d);
        for (out[premul_a * d ..][0..d], mod.get(at_zero, 0)) |*o, s| o.* = 1.0 + s;
        @memcpy(out[gate_a * d ..][0..d], mod.get(at_zero, 1));
        for (out[premul_f * d ..][0..d], mod.get(at_zero, 2)) |*o, s| o.* = 1.0 + s;
        @memcpy(out[gate_f * d ..][0..d], mod.get(at_zero, 3));
        for (out[head_premul * d ..][0..d], mod.head_scale) |*o, s| o.* = 1.0 + s;
        @memset(out[zero * d ..][0..d], 0);
    }
};

pub const Pass = enum { prefix, target };

/// The prefix rows through every block, each block's keys and values kept.
pub fn prefix(d: anytype, model: *const qwen_image21.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(i);
        try block(d, model.cfg, blk, i, .prefix);
    }
}

/// One step's image rows: embed, every block against the cached prefix, the head.
pub fn target(d: anytype, model: *const qwen_image21.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    try d.embed();
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(i);
        try block(d, model.cfg, blk, i, .target);
    }
    try d.head();
}

fn block(d: anytype, cfg: qwen_image21.Config, blk: *const qwen_image21.Block, i: usize, pass: Pass) !void {
    const ws = d.ws;
    const rows = d.rows(pass);
    const dim = cfg.dim;
    const k_out = try d.kvRows(ws.kj_d, pass);
    const v_out = try d.kvRows(ws.vj_d, pass);

    try d.modNorm(ws.x_d, ws.nrm_d, rows, Table.premul_a);
    const att = blk.dev[0..3];
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, att), &.{ ws.q_d, k_out, v_out }, att);
    try d.qkNormRope(blk.q_norm, blk.k_norm, k_out, rows, pass);
    try d.scale(v_out, rows * dim, v_div);
    try d.attention(i, pass);
    try d.scale(ws.attn_d, rows * dim, 1.0 / v_div);
    try d.linear(ws.dlt_d, ws.attn_d, rows, blk.dev[3]);
    try d.gatedAdd(ws.x_d, ws.dlt_d, rows, Table.gate_a);

    try d.modNorm(ws.x_d, ws.nrm_d, rows, Table.premul_f);
    const ffn = blk.dev[4..6];
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, ffn), &.{ ws.mg_d, ws.mu_d }, ffn);
    try d.siluMul(ws.mg_d, ws.mu_d, rows * cfg.ffn);
    try d.linear(ws.dlt_d, ws.mg_d, rows, blk.dev[6]);
    try d.gatedAdd(ws.x_d, ws.dlt_d, rows, Table.gate_f);
}
