//! The MiniMax H3 video VAE's transformer, written once for every GPU backend over an
//! adapter (`minimax_h3_vae_cuda.Dev`, `minimax_h3_vae_gpu.Dev`). `body` runs between
//! the adapter's upload of `vae.embedWindow` into `ws.x_d` and the LayerScales into
//! `ws.scale_d`, and its download of the grid tokens' patches from `ws.patch_d`.
//!
//! The residual, the norm outputs, the projections and the FFN planes are at the VAE
//! dial's width; `qkv_d` through the attention output `ao_d` stay f32, so rope, the
//! head padding and the attention never see a 16-bit operand.
//!
//! `to_qkv` is fused PER HEAD (`[h0 q|h0 k|h0 v|h1 q|...]` per token), so the three
//! planes are not row ranges of the weight and are split by `deinterleave3`; `ff.w1`
//! IS `[gate; value]` by rows, so that one splits as two GEMMs. The q/k norms are
//! WEIGHTLESS (`unitNorm`).

const std = @import("std");
const vae = @import("minimax_h3_vae.zig");

pub fn body(d: anytype, dec: *const vae.VideoDecoder, seq: usize, grid: usize) !void {
    const cfg = dec.cfg;
    const dim = cfg.dim;
    const ws = d.ws;

    for (dec.blocks, 0..) |*b, bi| {
        try d.rmsNorm(ws.x_d, ws.hn_d, b.norm1, seq, dim);
        try d.gemm(ws.qkv_d, ws.hn_d, seq, b.qkv, b.qkv_bias);
        try d.deinterleave3(seq);
        try d.unitNorm(ws.q_d, seq);
        try d.unitNorm(ws.k_d, seq);
        try d.rope(ws.q_d, seq);
        try d.rope(ws.k_d, seq);
        try d.attention(seq);
        try d.gemm(ws.proj_d, ws.ao_d, seq, b.out, b.out_bias);
        // LayerScale: the gated add with the learned scale where a DiT has a gate.
        try d.layerScale(ws.x_d, ws.proj_d, seq * dim, (bi * 2) * dim);

        try d.rmsNorm(ws.x_d, ws.hn_d, b.norm2, seq, dim);
        try d.gemm(ws.gate_d, ws.hn_d, seq, vae.rowSlice(b.w1, 0, cfg.ff), b.w1_bias[0..cfg.ff]);
        try d.gemm(ws.up_d, ws.hn_d, seq, vae.rowSlice(b.w1, cfg.ff, cfg.ff), b.w1_bias[cfg.ff..]);
        try d.siluMul(ws.gate_d, ws.up_d, seq * cfg.ff);
        try d.gemm(ws.proj_d, ws.gate_d, seq, b.w2, b.w2_bias);
        try d.layerScale(ws.x_d, ws.proj_d, seq * dim, (bi * 2 + 1) * dim);
        try d.blockDone(bi, seq);
    }

    // A LayerNorm WITH bias, where every other norm here is RMS. Only the GRID tokens
    // are projected; the register/zero suffix is dropped.
    try d.layerNorm(ws.x_d, ws.hn_d, dec.norm_out_w, dec.norm_out_b, seq, dim);
    try d.gemm(ws.patch_d, ws.hn_d, grid, dec.proj_out, dec.proj_out_bias);
}
