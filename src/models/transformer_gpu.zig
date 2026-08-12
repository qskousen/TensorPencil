//! Backend-generic GPU transformer decoder layer. `decoderLayer(spec, st, ...)` sequences
//! the per-layer ops in the order `LayerSpec` defines (shared with the CPU
//! `transformer.zig`); the stepper `st`, a `*VulkanLM` or `*CudaLM`, supplies each op as
//! a method wrapping its backend's kernels and perf policy: GEMV grouping,
//! `independent()` scheduling hints, flash-split versus square attention, weight
//! streaming.
//!
//! The op ORDER is single-sourced here, the op IMPL stays per-backend, and each method is
//! a faithful lift of one hand-written loop block including its leading scheduling hints,
//! so output is byte-identical to the loops this replaces.
//!
//! Stepper method contract, all `!void`, operating on the stepper's own device buffers
//! and KV cache:
//!
//!     normInput(layer, seq)        x -> normed via input_norm
//!     projectQKV(layer, seq)       normed -> q,k,v
//!     normQK(layer, seq)           per-head RMSNorm of q,k
//!     normV(seq)                   weightless RMSNorm of v (v_norm_unit specs)
//!     applyRope(seq, pos0)         rotate-half rope on q,k
//!     appendKV(l, seq, pos0)       write k,v into cache layer l at pos0
//!     attention(l, seq, pos0)      attn over the cached prefix -> attn buf
//!     projectO(layer, seq)         attn -> t via o_proj
//!     postAttnNorm(layer, seq)     (sandwich) RMSNorm t before residual
//!     addResidual(seq)             x += t
//!     normPreFfn(layer, seq)       x -> normed via the pre-MLP norm
//!     projectGateUp(layer, seq)    normed -> gate,up
//!     activate(comptime act, seq)  gate = act(gate) * up
//!     projectDown(layer, seq)      gate -> t via down_proj
//!     postFfnNorm(layer, seq)      (sandwich) RMSNorm t before residual
//!     outScale(layer, seq)         (gemma4) x *= layer.out_scale
const std = @import("std");

/// TP_DUMP_LAYERS: hash the residual stream after each layer, and layer 0's
/// sub-steps, so two runs that disagree on a logit can be bisected to the op
/// that diverged. Read once: this sits on the per-layer path. Only steppers
/// whose `debug_trace` is true are traced; for the rest the calls fold away.
var trace_flag: ?bool = null;

/// A stepper opts in with `pub const debug_trace = true`. Gating on the VALUE,
/// not just the decl, is what makes flipping it to false do what it reads as.
fn tracedStepper(comptime T: type) bool {
    if (!@hasDecl(T, "debug_trace")) return false;
    return T.debug_trace;
}
fn traceOn() bool {
    if (trace_flag) |v| return v;
    const v = std.c.getenv("TP_DUMP_LAYERS") != null;
    trace_flag = v;
    return v;
}

/// Download `n` f32s of `buf` and print their hash under `tag`.
fn traceBuf(st: anytype, tag: []const u8, buf: anytype, n: usize) void {
    const host = st.gpa.alloc(f32, n) catch return;
    defer st.gpa.free(host);
    st.be.tensorDownload(buf, std.mem.sliceAsBytes(host)) catch return;
    std.debug.print("[{s}] hash=0x{x}\n", .{ tag, std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(host)) });
}

const transformer = @import("transformer.zig");

pub const LayerSpec = transformer.LayerSpec;
pub const Activation = transformer.Activation;

/// One decoder layer, driven by the arch `spec`, over the stepper's buffers.
/// `l` is the layer index (into the KV cache), `seq` the rows this call
/// forwards, `pos0` the absolute base position for rope / cache append.
pub fn decoderLayer(comptime spec: LayerSpec, st: anytype, layer: anytype, l: usize, seq: usize, pos0: usize) !void {
    const trace = (comptime tracedStepper(@TypeOf(st.*))) and traceOn();
    // Layer 0's "before" is the embedding, so a divergence visible there is
    // upstream of the whole stack.
    if (trace and l == 0) traceBuf(st, "embed", st.bufs.x, seq * st.cfg.hidden);
    try decoderLayerQKV(spec, st, layer, l, seq, pos0);
    try decoderLayerAttnMlp(spec, st, layer, l, seq, pos0);
    if (trace) {
        var lbl: [24]u8 = undefined;
        traceBuf(st, std.fmt.bufPrint(&lbl, "layer {d}", .{l}) catch "layer", st.bufs.x, seq * st.cfg.hidden);
    }
}

/// First half of a decoder layer: project Q/K/V, norm/rope them, and commit K/V
/// to the cache. Split out so the Vulkan bidirectional image prefill can run
/// this for EVERY block token (committing all K/V) before any attention, the
/// single-query kernel can't see forward within a batch otherwise. `st.q` holds
/// the rope'd Q on return.
pub fn decoderLayerQKV(comptime spec: LayerSpec, st: anytype, layer: anytype, l: usize, seq: usize, pos0: usize) !void {
    // The geometry-sensitive ops (q/k/v projections, q/k/v norms) take `l`
    // because per-layer-geometry archs (gemma4) vary head_dim / KV width by
    // layer; uniform archs ignore it.
    const trace = (comptime tracedStepper(@TypeOf(st.*))) and l == 0 and traceOn();
    try st.normInput(layer, seq);
    if (trace) traceBuf(st, "sub normed", st.bufs.normed, seq * st.cfg.hidden);
    try st.projectQKV(l, layer, seq);
    if (trace) {
        traceBuf(st, "sub q", st.bufs.q, seq * st.cfg.qDim(l));
        traceBuf(st, "sub k", st.bufs.k, seq * st.cfg.kvDim(l));
        traceBuf(st, "sub v", st.bufs.v, seq * st.cfg.kvDim(l));
    }
    try st.normQK(l, layer, seq);
    if (trace) traceBuf(st, "sub qk_normed_q", st.bufs.q, seq * st.cfg.qDim(l));
    if (comptime spec.v_norm_unit) try st.normV(l, seq);
    if (trace) traceBuf(st, "sub v_normed", st.bufs.v, seq * st.cfg.kvDim(l));
    try st.applyRope(l, seq, pos0);
    if (trace) traceBuf(st, "sub roped_q", st.bufs.q, seq * st.cfg.qDim(l));
    try st.appendKV(l, seq, pos0);
}

/// Second half of a decoder layer: attention over the committed cache, output
/// projection, residual, and the MLP block. Expects `decoderLayerQKV` to have
/// run (K/V committed, `st.q` = rope'd Q).
pub fn decoderLayerAttnMlp(comptime spec: LayerSpec, st: anytype, layer: anytype, l: usize, seq: usize, pos0: usize) !void {
    // --- Attention ---
    const trace = (comptime tracedStepper(@TypeOf(st.*))) and l == 0 and traceOn();
    try st.attention(l, seq, pos0);
    if (trace) traceBuf(st, "sub attn_out", st.bufs.attn, seq * st.cfg.qDim(l));
    try st.projectO(l, layer, seq);
    if (trace) traceBuf(st, "sub proj_o", st.bufs.t, seq * st.cfg.hidden);
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
