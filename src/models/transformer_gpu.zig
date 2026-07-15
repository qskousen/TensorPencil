//! Backend-generic GPU transformer decoder layer. `decoderLayer(spec, st, …)`
//! sequences the per-layer ops in the architecture order defined by `LayerSpec`
//! (shared with the CPU `transformer.zig`); the stepper `st` (a `*VulkanLM` /
//! `*CudaLM`) supplies each op as a method that wraps its backend's kernels and
//! perf policy — GEMV grouping, `independent()` scheduling hints, flash-split vs
//! square attention, weight streaming. The op ORDER is single-sourced here; the
//! op IMPL stays per-backend, so output is byte-identical to the hand-written
//! loops this replaces (each method is a faithful lift of one loop block,
//! including its leading scheduling hints).
//!
//! Stepper method contract (all `!void`, operating on the stepper's own
//! device buffers / KV cache):
//!   normInput(layer, seq)           x → normed via input_norm
//!   projectQKV(layer, seq)          normed → q,k,v
//!   normQK(layer, seq)              per-head RMSNorm of q,k
//!   normV(seq)                      weightless RMSNorm of v (v_norm_unit specs)
//!   applyRope(seq, pos0)            rotate-half rope on q,k
//!   appendKV(l, seq, pos0)          write k,v into cache layer l at pos0
//!   attention(l, seq, pos0)         attn over the cached prefix → attn buf
//!   projectO(layer, seq)            attn → t via o_proj
//!   postAttnNorm(layer, seq)        (sandwich) RMSNorm t before residual
//!   addResidual(seq)                x += t
//!   normPreFfn(layer, seq)          x → normed via the pre-MLP norm
//!   projectGateUp(layer, seq)       normed → gate,up
//!   activate(comptime act, seq)     gate = act(gate) * up
//!   projectDown(layer, seq)         gate → t via down_proj
//!   postFfnNorm(layer, seq)         (sandwich) RMSNorm t before residual
//!   outScale(layer, seq)            (gemma4) x *= layer.out_scale

const transformer = @import("transformer.zig");

pub const LayerSpec = transformer.LayerSpec;
pub const Activation = transformer.Activation;

/// One decoder layer, driven by the arch `spec`, over the stepper's buffers.
/// `l` is the layer index (into the KV cache), `seq` the rows this call
/// forwards, `pos0` the absolute base position for rope / cache append.
pub fn decoderLayer(comptime spec: LayerSpec, st: anytype, layer: anytype, l: usize, seq: usize, pos0: usize) !void {
    // --- Attention ---
    // The geometry-sensitive ops (q/k/v/o projections, q/k/v norms) take `l`
    // because per-layer-geometry archs (gemma4) vary head_dim / KV width by
    // layer; uniform archs ignore it.
    try st.normInput(layer, seq);
    try st.projectQKV(l, layer, seq);
    try st.normQK(l, layer, seq);
    if (comptime spec.v_norm_unit) try st.normV(l, seq);
    try st.applyRope(l, seq, pos0);
    try st.appendKV(l, seq, pos0);
    try st.attention(l, seq, pos0);
    try st.projectO(l, layer, seq);
    if (comptime spec.sandwich_norms) try st.postAttnNorm(layer, seq);
    try st.addResidual(seq);

    // --- MLP ---
    try st.normPreFfn(layer, seq);
    try st.projectGateUp(layer, seq);
    try st.activate(spec.activation, seq);
    try st.projectDown(layer, seq);
    if (comptime spec.sandwich_norms) try st.postFfnNorm(layer, seq);
    try st.addResidual(seq);

    if (comptime spec.out_scale) try st.outScale(layer, seq);
}
