//! The qwen35 device forward, written once for every GPU backend: `decode` (one row)
//! and `batch` (a prefill chunk of `n` rows), over a backend adapter (`qwen35_cuda.Dev`).
//! Each body keeps its own op order, which is what a captured CUDA graph records and what
//! keeps a backend's output bit-identical when it moves onto this file.
//!
//! The adapter holds `cfg`, `lm` and `bufs` (the activation buffers by the names below)
//! and supplies the ops: `rms`, `addRms` (a residual add then the norm of the sum, which a
//! backend may fuse), `prep` / `gemm` / `linear` (its linear dispatcher),
//! `view` (a byte range of a buffer), `deinterleave2`, `mulSigmoid`, `add`, `siluMul`,
//! `l2normRows`, `l2normRowsGrouped`, the GDN ops, the per-layer recurrent state
//! (`convState`, `ssmState`, `aDt`), the batched GDN route (`gdnRoute`, `abViews`) and
//! the hooks a backend may need around the forward (`forwardBegin`, `layerBegin`,
//! which can run a layer elsewhere, `tapLayer`, `forwardEnd`, `headBegin`).
//!
//! RoPE, the KV store, attention and the normed tap come in two forms, `…Step` for the
//! decode row at the step's position and `…Rows` for a batch with per-row positions,
//! because a batch of one is not a decode step: it takes the batch kernels.

const std = @import("std");
const qwen35 = @import("qwen35.zig");
const ops = @import("tp_ops");
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// How a GDN layer's chunk runs: `chunk` batches the gates, conv, L2 norm and the state
/// recurrence over the whole chunk; `stacked` names the one weight alpha and beta come
/// from, row-padded `[alpha | beta]`, instead of two skinny linears.
pub fn GdnRoute(comptime Weight: type) type {
    return struct { chunk: bool, stacked: ?Weight = null };
}

/// Where a batched chunk's alpha and beta land, and the floats a token between rows.
pub fn AbViews(comptime Buf: type) type {
    return struct { alpha: Buf, beta: Buf, pitch: usize };
}

/// The single-token forward. `graph` is the CUDA graph recording: the adapter's KV store
/// and attention read their position from device state there, and the host taps are off.
/// Does not advance the cache length.
pub fn decode(d: anytype, comptime graph: bool, want_logits: bool) !void {
    const cfg = d.cfg;
    const b = d.bufs;
    const hd = cfg.head_dim;

    try d.forwardBegin(1);
    for (d.lm.layers, 0..) |*layer, l| {
        if (try d.layerBegin(l, 1)) continue;
        switch (layer.*) {
            .attn => |*al| {
                const slot = l / cfg.full_attn_interval;
                try d.rms(b.x, b.normed, al.input_norm, 1, cfg.hidden);
                const g = try d.prep(b.normed, 1, cfg.hidden, &.{ al.qg, al.k, al.v });
                try d.gemm(g, b.qg, al.qg);
                try d.gemm(g, b.k, al.k);
                try d.gemm(g, b.v, al.v);
                try d.deinterleave2(b.qg, b.q, b.gate, cfg.qDim(), hd);
                try d.rms(b.q, b.q, al.q_norm, cfg.n_heads, hd);
                try d.rms(b.k, b.k, al.k_norm, cfg.n_kv_heads, hd);
                try d.ropeStep(b.q, cfg.n_heads);
                try d.ropeStep(b.k, cfg.n_kv_heads);
                try d.storeKvStep(slot, graph);
                try d.attendStep(slot, graph);
                try d.mulSigmoid(b.attn, b.gate, cfg.qDim());
                try d.linear(b.t, b.attn, 1, al.o);
            },
            .linear => |*ll| {
                const lin_idx = l - l / cfg.full_attn_interval;
                const channels = cfg.convChannels();
                const dk = cfg.lin_head_dim;
                const heads = cfg.lin_v_heads;
                try d.rms(b.x, b.normed, ll.input_norm, 1, cfg.hidden);
                if (!graph) try d.tapNormedStep(l);
                const g = try d.prep(b.normed, 1, cfg.hidden, &.{ ll.qkv, ll.z, ll.alpha, ll.beta });
                try d.gemm(g, b.lin_qkv, ll.qkv);
                try d.gemm(g, b.lin_z, ll.z);
                try d.gemm(g, try d.view(b.ab, 0, heads * 4), ll.alpha);
                try d.gemm(g, try d.view(b.ab, heads * 4, heads * 4), ll.beta);
                try d.gdnGates(b.ab, d.aDt(lin_idx), b.gates, heads);
                try d.gdnConvStep(d.convState(lin_idx), b.lin_qkv, ll.conv_w, b.lin_conv, channels);
                try d.l2normRows(try d.view(b.lin_conv, 0, 2 * cfg.linQKDim() * 4), 2 * cfg.lin_k_heads, dk);
                try d.gdnDeltaStep(d.ssmState(lin_idx), b.lin_conv, b.gates, b.lin_o, heads, dk, cfg.lin_k_heads, deltaScale(dk));
                try d.rms(b.lin_o, b.lin_o, ll.ssm_norm, heads, dk);
                try d.siluMul(b.lin_z, b.lin_o, cfg.linVDim());
                try d.linear(b.t, b.lin_z, 1, ll.out);
            },
        }
        try mlp(d, layer, 1, true);
        if (!graph) try d.tapLayer(l, 1);
    }
    try d.forwardEnd(1, true);

    if (want_logits) {
        try d.rms(b.x, b.t, d.lm.final_norm, 1, cfg.hidden);
        d.headBegin();
        try d.linear(b.logits, b.t, 1, d.lm.head);
    }
}

/// The batched forward over `n` embedded rows in `bufs.x`: projections and MLPs as
/// GEMMs, attention as one `n`-query batch, the DeltaNet recurrence chunked where the
/// adapter's route allows and per token otherwise (it is serial in time). Polls
/// `ops.cancel` between layers. Does not advance the cache length.
pub fn batch(d: anytype, n: usize) !void {
    const cfg = d.cfg;
    const b = d.bufs;
    const hd = cfg.head_dim;
    const skip = skipMask();

    try d.forwardBegin(n);
    const ctok = ops.cancel.token;
    for (d.lm.layers, 0..) |*layer, l| {
        if (ops.cancel.canceled(ctok)) return error.Canceled;
        if (try d.layerBegin(l, n)) continue;
        switch (layer.*) {
            .attn => |*al| {
                const slot = l / cfg.full_attn_interval;
                if (skip & 32 == 0) try d.rms(b.x, b.normed, al.input_norm, n, cfg.hidden);
                if (skip & 16 == 0) {
                    const g = try d.prep(b.normed, n, cfg.hidden, &.{ al.qg, al.k, al.v });
                    try d.gemm(g, b.qg, al.qg);
                    try d.gemm(g, b.k, al.k);
                    try d.gemm(g, b.v, al.v);
                }
                if (skip & 64 == 0) try d.deinterleave2(b.qg, b.q, b.gate, n * cfg.qDim(), hd);
                if (skip & 32 == 0) {
                    try d.rms(b.q, b.q, al.q_norm, n * cfg.n_heads, hd);
                    try d.rms(b.k, b.k, al.k_norm, n * cfg.n_kv_heads, hd);
                }
                if (skip & 64 == 0) {
                    try d.ropeRows(b.q, cfg.n_heads, n);
                    try d.ropeRows(b.k, cfg.n_kv_heads, n);
                }
                if (skip & 128 == 0) try d.storeKvRows(slot, n);
                if (skip & 8 == 0) try d.attendRows(slot, n);
                if (skip & 256 == 0) try d.mulSigmoid(b.attn, b.gate, n * cfg.qDim());
                if (skip & 16 == 0) try d.linear(b.t, b.attn, n, al.o);
            },
            .linear => |*ll| {
                const lin_idx = l - l / cfg.full_attn_interval;
                const channels = cfg.convChannels();
                const dk = cfg.lin_head_dim;
                const heads = cfg.lin_v_heads;
                if (skip & 32 == 0) try d.rms(b.x, b.normed, ll.input_norm, n, cfg.hidden);
                try d.tapNormedRows(l, n);
                if (skip & 2 == 0) {
                    const g = try d.prep(b.normed, n, cfg.hidden, &.{ ll.qkv, ll.z });
                    try d.gemm(g, b.lin_qkv, ll.qkv);
                    try d.gemm(g, b.lin_z, ll.z);
                }
                const conv_state = d.convState(lin_idx);
                const ssm = d.ssmState(lin_idx);
                // Everything but the state recurrence is per-token independent, so it
                // batches over the chunk; measured on Bonsai-27B, running all seven per
                // token cost 46% of prefill.
                const route = d.gdnRoute(lin_idx, n);
                const chunked = skip & 4 == 0 and !noGdnChunk() and route.chunk;
                if (chunked) {
                    // One GEMM over the stacked [alpha | beta] weight when there is one,
                    // so the gates read two views of ONE padded output.
                    const ab = try d.abViews(route, n);
                    if (route.stacked) |cat| {
                        try d.linear(b.ab_n, b.normed, n, cat);
                    } else {
                        const g = try d.prep(b.normed, n, cfg.hidden, &.{ ll.alpha, ll.beta });
                        try d.gemm(g, ab.alpha, ll.alpha);
                        try d.gemm(g, ab.beta, ll.beta);
                    }
                    try d.gdnGatesBatch(ab.alpha, ab.beta, d.aDt(lin_idx), b.gates, heads, n, ab.pitch);
                    try d.gdnConvBatch(conv_state, b.lin_qkv, ll.conv_w, b.lin_conv, channels, n);
                    // The q/k slice is NOT contiguous across tokens (v sits between
                    // them), hence the grouped form.
                    try d.l2normRowsGrouped(b.lin_conv, n * 2 * cfg.lin_k_heads, dk, 2 * cfg.lin_k_heads, channels);
                    try d.gdnDeltaChunk(ssm, b.lin_conv, b.gates, b.lin_o, heads, dk, cfg.lin_k_heads, n, channels, deltaScale(dk));
                }
                // Per-token form: one row, or a head dim the chunk recurrence cannot take.
                // Same math, which `TP_NO_GDN_CHUNK` checks bit for bit.
                for (0..if (chunked or skip & 4 != 0) 0 else n) |t| {
                    const g = try d.prep(try d.view(b.normed, t * cfg.hidden * 4, cfg.hidden * 4), 1, cfg.hidden, &.{ ll.alpha, ll.beta });
                    try d.gemm(g, try d.view(b.ab, 0, heads * 4), ll.alpha);
                    try d.gemm(g, try d.view(b.ab, heads * 4, heads * 4), ll.beta);
                    try d.gdnGates(b.ab, d.aDt(lin_idx), b.gates, heads);
                    try d.gdnConvStep(conv_state, try d.view(b.lin_qkv, t * channels * 4, channels * 4), ll.conv_w, b.lin_conv, channels);
                    try d.l2normRows(try d.view(b.lin_conv, 0, 2 * cfg.linQKDim() * 4), 2 * cfg.lin_k_heads, dk);
                    try d.gdnDeltaStep(ssm, b.lin_conv, b.gates, try d.view(b.lin_o, t * cfg.linVDim() * 4, cfg.linVDim() * 4), heads, dk, cfg.lin_k_heads, deltaScale(dk));
                }
                if (skip & 32 == 0) try d.rms(b.lin_o, b.lin_o, ll.ssm_norm, n * heads, dk);
                if (skip & 256 == 0) try d.siluMul(b.lin_z, b.lin_o, n * cfg.linVDim());
                if (skip & 2 == 0) try d.linear(b.t, b.lin_z, n, ll.out);
            },
        }
        // The sublayer's residual add rides the MLP's norm.
        if (skip & 1 == 0) {
            try mlp(d, layer, n, skip & 256 == 0);
        } else if (skip & 256 == 0) try d.add(b.x, b.t, n * cfg.hidden);
        try d.tapLayer(l, n);
    }
    try d.forwardEnd(n, false);
}

/// The MLP block, first adding the sublayer output in `bufs.t` into x when `add`.
fn mlp(d: anytype, layer: *const qwen35.Layer, n: usize, add: bool) !void {
    const cfg = d.cfg;
    const b = d.bufs;
    const m = switch (layer.*) {
        .attn => |*al| &al.mlp,
        .linear => |*ll| &ll.mlp,
    };
    if (add) try d.addRms(b.x, b.t, b.normed, m.post_norm, n, cfg.hidden) else try d.rms(b.x, b.normed, m.post_norm, n, cfg.hidden);
    const g = try d.prep(b.normed, n, cfg.hidden, &.{ m.gate, m.up });
    try d.gemm(g, b.mlp_gate, m.gate);
    try d.gemm(g, b.mlp_up, m.up);
    try d.siluMul(b.mlp_gate, b.mlp_up, n * cfg.intermediate);
    try d.linear(b.t, b.mlp_gate, n, m.down);
    try d.add(b.x, b.t, n * cfg.hidden);
}

fn deltaScale(dk: usize) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
}

/// `TP_SKIP`, the prefill budget bitmask: each bit removes one component from `batch`,
/// so a run measures the cost of everything else. The output is garbage by
/// construction; its value is that the removed costs must sum to the baseline.
///
///   1 MLP (gate/up/down + silu)      8 attention op
///   2 GDN qkv/z/out GEMMs           16 attn qkv/o GEMMs
///   4 GDN recurrence + batched ops   32 norms (input/q/k/ssm; post_norm is in 1)
///  64 rope + deinterleave           128 KV store
/// 256 residual adds + gating (mulSigmoid, GDN siluMul)
fn skipMask() u32 {
    if (skip_cache) |v| return v;
    const s = getenv("TP_SKIP") orelse {
        skip_cache = 0;
        return 0;
    };
    var v: u32 = 0;
    for (std.mem.span(s)) |c| if (c >= '0' and c <= '9') {
        v = v * 10 + (c - '0');
    };
    skip_cache = v;
    return v;
}
var skip_cache: ?u32 = null;

/// `TP_NO_GDN_CHUNK`: the per-token GDN form even where the chunk recurrence applies.
/// The two are meant to be bit-identical, and this is how that is checked on a real
/// checkpoint: same binary, same prompt, greedy, diff.
fn noGdnChunk() bool {
    if (no_chunk_cache) |v| return v;
    const v = getenv("TP_NO_GDN_CHUNK") != null;
    no_chunk_cache = v;
    return v;
}
var no_chunk_cache: ?bool = null;
