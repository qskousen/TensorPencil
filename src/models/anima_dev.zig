//! The Anima (Cosmos-Predict2 `MiniTrainDIT`) device forward, written once for every GPU
//! backend over an adapter (`anima_cuda.Dev`, `anima_gpu.Dev`). `body` runs between the
//! adapter's upload of the step's (folded) modulation table and patches and its download
//! of the sequence; the final layer runs on the host (`DiT.finalize`).
//!
//! The adapter holds `ws` (`x_d` the residual, `nrm_d`, `q_d`/`k_d`/`v_d`, `attn_d`,
//! `dlt_d`, `mlp_d`) and supplies the ops: `lnMod` (weightless LayerNorm plus the
//! table's scale and shift), `qkNormRope` (self-attention's per-head norms and
//! split-half rope), `qNorm` (cross-attention's query norm), `selfAttn`, `crossAttn`
//! (against block `bi`'s cached K/V), `gelu` and `gatedAdd`. The op order here is what
//! keeps a backend bit-identical when it moves.

const std = @import("std");
const anima = @import("anima.zig");
const Weight = @import("tp_ops").matmul.Weight;

/// The dtype the residual, the norm output, the delta and the MLP buffer are stored at
/// when `want` is asked for, on either backend; the attention block stays f32.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

const Act = @import("tp_gpu").context.Dt;

pub fn body(d: anytype, model: *const anima.DiT, seq: usize, cancel: ?*std.atomic.Value(bool)) !void {
    try d.embed();
    for (model.blocks, 0..) |*blk, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(bi);
        try block(d, model.cfg, blk, bi, seq, bi * 9 * model.cfg.dim);
    }
}

/// One `Block`: three sublayers of the same shape (modulated LayerNorm, sublayer, gated
/// residual) reading successive thirds of the table `[shift | scale | gate]`, scale
/// already `1 +`'d. Only self-attention takes RoPE.
fn block(d: anytype, cfg: anima.Config, blk: anytype, bi: usize, seq: usize, base: usize) !void {
    const ws = d.ws;
    const dim = cfg.dim;

    try d.lnMod(seq, base + dim, base);
    const qkv = [_]Weight{ blk.self_attn.q, blk.self_attn.k, blk.self_attn.v };
    try d.gemms(try d.prep(ws.nrm_d, seq, dim, &qkv), &.{ ws.q_d, ws.k_d, ws.v_d }, &qkv);
    // Anima's Q/K norms take the BLOCKS' 1e-6, not `finfo(f32).eps` as Z-Image's do.
    try d.qkNormRope(blk.self_attn.qnorm, blk.self_attn.knorm, seq);
    try d.selfAttn(seq);
    try d.linear(ws.dlt_d, ws.attn_d, seq, blk.self_attn.out);
    try d.gatedAdd(seq, base + 2 * dim);

    // Cross-attention onto the adapter's output: no RoPE, and K/V come from the
    // session's per-image cache rather than from two GEMMs.
    try d.lnMod(seq, base + 4 * dim, base + 3 * dim);
    try d.linear(ws.q_d, ws.nrm_d, seq, blk.cross_attn.q);
    try d.qNorm(blk.cross_attn.qnorm, seq);
    try d.crossAttn(bi, seq);
    try d.linear(ws.dlt_d, ws.attn_d, seq, blk.cross_attn.out);
    try d.gatedAdd(seq, base + 5 * dim);

    try d.lnMod(seq, base + 7 * dim, base + 6 * dim);
    try d.linear(ws.mlp_d, ws.nrm_d, seq, blk.mlp1);
    // `nn.GELU()` with the default `approximate='none'`: the erf form, not tanh.
    try d.gelu(seq * cfg.mlp_dim);
    try d.linear(ws.dlt_d, ws.mlp_d, seq, blk.mlp2);
    try d.gatedAdd(seq, base + 8 * dim);
}
