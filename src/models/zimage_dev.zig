//! The Z-Image (NextDiT) device forward, written once for every GPU backend over an
//! adapter (`zimage_cuda.Dev`, `zimage_gpu.Dev`). `body` runs between the adapter's
//! upload of the step's modulation table and patches and its download of the image rows;
//! the final layer runs on the host (`DiT.finalLayer`).
//!
//! The adapter holds `ws`, whose activation buffers go by the names below, and supplies
//! the ops. `modNorm` is the modulated norm, whose table layout is the adapter's own (one
//! backend folds the norm weight into it, the other applies it first); `qkNormRope` the
//! per-head norms and rope of q and k for a `Stage`; `attention` scores q/k/v into
//! `attn_d`. The op order here is what keeps a backend bit-identical when it moves.

const std = @import("std");
const zimage = @import("zimage.zig");
const Weight = @import("tp_ops").matmul.Weight;

/// Exact power-of-two prescale on V across attention's f16 cast. Q and K are RMS-normed
/// on the way in; V is not, and this trunk's V reaches ~98000 by the deep blocks, past
/// f16's 65504: it becomes inf, `softmax @ V` becomes NaN, and the render is solid white
/// with no error. Attention is linear in V, so scaling it down and the output back up is
/// the same arithmetic, and a power of two shifts the exponent at no cost in precision.
/// 2^-6 takes the safe ceiling to ~4.2e6. A constant rather than a measured maximum, which
/// would need a device reduction per attention for a quantity that only has to be bounded.
pub const v_div: f32 = 1.0 / 64.0;

/// The dtype the stream is stored at when `want` is asked for, on either backend: every
/// op it touches takes 16 bits and the attention block stays f32.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

const Act = @import("tp_gpu").context.Dt;

/// Which sequence a block runs over: the image half's noise refiner, or the joint
/// caption-plus-image sequence (each has its own rope table).
pub const Stage = enum { image, joint };

pub fn body(d: anytype, model: *const zimage.DiT, cancel: ?*std.atomic.Value(bool)) !void {
    const ws = d.ws;
    const dim = model.cfg.dim;
    const g = d.geom;

    try d.embed();
    d.trace("x_embed", 0, ws.img_d, g.n_img * dim);
    for (model.noise_refiner, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(.image, i);
        try block(d, model.cfg, blk, ws.img_d, g.img_padded, .image, i * 4 * dim);
        d.trace("refiner", i, ws.img_d, g.img_padded * dim);
    }

    try d.join();
    for (model.layers, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(.joint, i);
        try block(d, model.cfg, blk, ws.x_d, g.seq, .joint, (model.noise_refiner.len + i) * 4 * dim);
        d.trace("layer", i, ws.x_d, g.seq * dim);
    }
}

/// One `JointTransformerBlock`, `x` `[rows][dim]` in place; `base` is this block's
/// offset into the modulation table (`[scale_msa | gate_msa | scale_mlp | gate_mlp]`,
/// the gates `tanh`'d on the host).
fn block(d: anytype, cfg: zimage.Config, blk: anytype, x: anytype, rows: usize, stage: Stage, base: usize) !void {
    const ws = d.ws;
    const dim = cfg.dim;

    // x += tanh(gate_msa) * norm2(attn(modulate(norm1(x), scale_msa)))
    try d.modNorm(x, ws.nrm_d, rows, base + 0 * dim, blk.attn_norm1);
    const att_ws = [_]Weight{ blk.attn.q, blk.attn.k, blk.attn.v };
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, &att_ws), &.{ ws.q_d, ws.k_d, ws.v_d }, &att_ws);
    d.trace("  attn.nrm", 0, ws.nrm_d, rows * dim);
    d.trace("  attn.q", 0, ws.q_d, rows * cfg.qDim());
    d.trace("  attn.v", 0, ws.v_d, rows * cfg.kvDim());
    // The Q/K norms take `finfo(f32).eps`, NOT the blocks' 1e-5 (`Config.qk_eps`).
    try d.qkNormRope(blk.attn.qnorm, blk.attn.knorm, rows, stage);
    try d.scale(ws.v_d, rows * cfg.kvDim(), v_div);
    try d.attention(rows);
    try d.scale(ws.attn_d, rows * cfg.qDim(), 1.0 / v_div);
    d.trace("  attn.o", 0, ws.attn_d, rows * cfg.qDim());
    try d.linear(ws.dlt_d, ws.attn_d, rows, blk.attn.out);
    // The sandwich norm: a SECOND RMSNorm on the sublayer's output, inside the residual.
    try d.rowNorm(ws.dlt_d, blk.attn_norm2, rows);
    d.trace("  attn.dlt", 0, ws.dlt_d, rows * dim);
    try d.gatedAdd(x, ws.dlt_d, rows, base + 1 * dim);
    d.trace("  attn.res", 0, x, rows * dim);

    // x += tanh(gate_mlp) * norm2(swiglu(modulate(norm1(x), scale_mlp)))
    try d.modNorm(x, ws.nrm_d, rows, base + 2 * dim, blk.ffn_norm1);
    const ffn_ws = [_]Weight{ blk.ffn.w1, blk.ffn.w3 };
    try d.gemms(try d.prep(ws.nrm_d, rows, dim, &ffn_ws), &.{ ws.mg_d, ws.mu_d }, &ffn_ws);
    d.trace("  ffn.nrm", 0, ws.nrm_d, rows * dim);
    d.trace("  ffn.w1", 0, ws.mg_d, rows * cfg.mlp_dim);
    try d.siluMul(ws.mg_d, ws.mu_d, rows * cfg.mlp_dim);
    try d.linear(ws.dlt_d, ws.mg_d, rows, blk.ffn.w2);
    d.trace("  ffn.w2", 0, ws.dlt_d, rows * dim);
    try d.rowNorm(ws.dlt_d, blk.ffn_norm2, rows);
    try d.gatedAdd(x, ws.dlt_d, rows, base + 3 * dim);
}
