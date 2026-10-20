//! The K2 Horizon layer forward, written once for every GPU backend over its adapter
//! (`k2_horizon_cuda.Dev`, `k2_horizon_gpu.Dev`). Each keeps this op order, which is
//! what keeps a backend's output bit-identical when it moves onto this file.
//!
//! The adapter holds `cfg` and `bufs` (the activation buffers by the names below) and
//! supplies `groupNorm`, `linear`, `linears` (several weights over one input, which a
//! backend may stage once), `siluMul`, `add`, `rope` (q and k at the batch's
//! positions), `storeKv`, `attention` and `softplusGate`, plus the two routed blocks,
//! `values` (V from the value experts, into `bufs.v`) and `experts` (the routed FFN,
//! into `bufs.tmp`), whose placement (resident, staged, on the host) is the backend's.
//! `layerBegin` runs after the attention projections, before anything routed.

const k2 = @import("k2_horizon.zig");

/// Layer `l` of a batch of `seq` rows at positions `pos0 ..`, in place on `bufs.x`.
pub fn layer(d: anytype, w: k2.Layer, l: usize, seq: usize, pos0: usize) !void {
    const c = d.cfg;
    const b = d.bufs;
    try d.groupNorm(b.x, b.normed, w.attn_norm, seq);
    // Every projection of `normed` at once; the gate is independent of the attention.
    var ys: [4]@TypeOf(b.q) = .{ b.q, b.k, undefined, undefined };
    var ws: [4]@TypeOf(w.q) = .{ w.q, w.k, undefined, undefined };
    var n: usize = 2;
    if (w.values == null) {
        ys[n] = b.v;
        ws[n] = w.v.?;
        n += 1;
    }
    if (w.attn_gate) |gate| {
        ys[n] = b.gate;
        ws[n] = gate;
        n += 1;
    }
    try d.linears(b.normed, seq, ys[0..n], ws[0..n]);
    try d.layerBegin(l, seq);
    if (w.values) |values| try d.values(l, values, seq);
    try d.rope(seq, pos0);
    try d.storeKv(l, seq, pos0);
    try d.attention(l, seq, pos0);
    if (w.attn_gate != null) try d.softplusGate(seq * c.qDim());
    try d.linear(b.tmp, b.attn, seq, w.o);
    try d.add(b.x, b.tmp, seq * c.hidden);

    try d.groupNorm(b.x, b.normed, w.ffn_norm, seq);
    if (w.experts) |experts| {
        try d.experts(l, experts, seq);
    } else {
        try d.linear(b.expert_a, b.normed, seq, w.dense_gate.?);
        try d.linear(b.expert_b, b.normed, seq, w.dense_up.?);
        try d.siluMul(b.expert_a, b.expert_b, seq * c.intermediate);
        try d.linear(b.tmp, b.expert_a, seq, w.dense_down.?);
    }
    try d.add(b.x, b.tmp, seq * c.hidden);
}
