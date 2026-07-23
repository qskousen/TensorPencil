//! Data-driven transformer decoder layer, shared by the CPU forward passes of
//! qwen3 / gemma3 / gemma4. One comptime `LayerSpec` selects the architecture's
//! norm placement, activation, V handling, and attention scale; the generic
//! `layerForward` / `layerForwardTree` compose the ops once.
//!
//! `layer`/`cache`/`s` are `anytype`: each model keeps its own `Config`/`Layer`/
//! `Scratch` (unchanged — the GPU steppers depend on their exact shapes), and
//! optional per-arch fields (`post_attn_norm`, `layer.v`, `out_scale`) are read
//! only under `if (comptime spec.<flag>)`, so an arch whose struct lacks a field
//! never compiles that branch. Per-layer geometry, the RoPE table (global vs
//! local), and the sliding window are resolved BY THE CALLER into `Dims` +
//! `freqs`, so this body has no `cfg` coupling and no accessor-arity pitfalls.
//!
//! qwen35 (hybrid DeltaNet) is intentionally NOT covered: its attention path is
//! a superset (fused q+gate, partial RoPE, output gate) and its linear path is a
//! different computation.

const std = @import("std");
const ops = @import("tp_ops");

pub const Activation = enum { silu_mul, gelu_tanh_mul };

/// Per-architecture layer descriptor (comptime). Everything runtime/per-layer
/// (dims, rope table, window) is passed via `Dims`/`freqs`, not here.
pub const LayerSpec = struct {
    /// FFN gate activation: SwiGLU (qwen) vs GeGLU/gelu-tanh (gemma).
    activation: Activation,
    /// Gemma "sandwich" norms: a post-attention norm on the attn output and a
    /// post-ffn norm on the MLP output, each applied BEFORE its residual add,
    /// and the pre-MLP norm is `pre_ffn_norm` (vs qwen3's single `post_norm`).
    sandwich_norms: bool = false,
    /// V projection is optional (`layer.v: ?Weight`); when null the raw K
    /// projection is reused as V (gemma4 global MQA layers).
    optional_v_proj: bool = false,
    /// Weightless RMS-norm over head_dim on V before caching (gemma4).
    v_norm_unit: bool = false,
    /// Per-layer scalar `layer.out_scale` multiplies the whole residual stream
    /// after the MLP (gemma4).
    out_scale: bool = false,
    /// Attention score scale; null = 1/sqrt(head_dim). Gemma4 folds the scale
    /// into its QK norms and passes 1.0.
    attn_scale: ?f32 = null,
    /// Per-head QK RMS-norm before RoPE (qwen3/gemma). false for the plain
    /// llama/Mistral family, which has no q_norm/k_norm tensors — those layer
    /// fields go unread, so the arch's Layer struct may leave them empty.
    qk_norm: bool = true,
};

pub const qwen3_spec: LayerSpec = .{ .activation = .silu_mul };
/// llama/Mistral: SwiGLU, no QK-norm. (RoPE is made rotate-half-compatible by
/// un-permuting q/k weights at load — see qwen3.loadLayersCfg.)
pub const llama_spec: LayerSpec = .{ .activation = .silu_mul, .qk_norm = false };
pub const gemma3_spec: LayerSpec = .{ .activation = .gelu_tanh_mul, .sandwich_norms = true };
pub const gemma4_spec: LayerSpec = .{
    .activation = .gelu_tanh_mul,
    .sandwich_norms = true,
    .optional_v_proj = true,
    .v_norm_unit = true,
    .out_scale = true,
    .attn_scale = 1.0,
};

/// Whether this call is the encoder's full-sequence pass (`fresh`, no persistent
/// KV cache) or a KV-cached decode/prefill (`cached`). Tree verify has its own
/// entry point (`layerForwardTree`).
pub const Mode = enum { fresh, cached };

/// Per-layer dimensions the caller resolves (uniform for qwen3/gemma3, per-layer
/// for gemma4's varying head_dim/KV geometry) plus the sliding window (0 = full
/// causal, also used for global layers). The RoPE table for the layer is passed
/// separately as `freqs` (the caller selects global vs local).
pub const Dims = struct {
    hidden: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    sliding_window: usize = 0,
};

/// The shared MLP block: norm → gate/up → gated activation → down → [post-ffn
/// norm] → residual add. `x` is [seq, hidden].
fn mlpBlock(comptime spec: LayerSpec, io: std.Io, gpa: std.mem.Allocator, layer: anytype, x: []f32, seq: usize, dims: Dims, eps: f32, s: anytype) !void {
    const normed = s.normed[0 .. seq * dims.hidden];
    const gate = s.gate[0 .. seq * dims.intermediate];
    const up = s.up[0 .. seq * dims.intermediate];
    const tmp = s.tmp[0 .. seq * dims.hidden];

    const pre_norm = if (comptime spec.sandwich_norms) layer.pre_ffn_norm else layer.post_norm;
    ops.norm.rmsNorm(normed, x, pre_norm, eps);
    try ops.matmul.matmul(io, gpa, gate, normed, seq, layer.gate, null);
    try ops.matmul.matmul(io, gpa, up, normed, seq, layer.up, null);
    switch (comptime spec.activation) {
        .silu_mul => ops.act.siluMul(gate, up),
        .gelu_tanh_mul => ops.act.geluTanhMul(gate, up),
    }
    try ops.matmul.matmul(io, gpa, tmp, gate, seq, layer.down, null);
    if (comptime spec.sandwich_norms) ops.norm.rmsNorm(tmp, tmp, layer.post_ffn_norm, eps);
    for (x, tmp) |*xi, ti| xi.* += ti;
}

/// Project + norm + rope Q/K/V into `s.q`/`s.k`/`s.v` (sliced to the layer's
/// widths). Shared by the cached/fresh and tree attention blocks. `applyRope`
/// is called by the caller AFTER this only for the tree path (which needs
/// per-node positions); for fresh/cached this applies rope directly with
/// `pos0`. Returns the exact-length q/k/v slices.
fn qkvProject(comptime spec: LayerSpec, io: std.Io, gpa: std.mem.Allocator, layer: anytype, normed: []const f32, seq: usize, dims: Dims, eps: f32, s: anytype) !struct { q: []f32, k: []f32, v: []f32 } {
    const q = s.q[0 .. seq * dims.q_dim];
    const k = s.k[0 .. seq * dims.kv_dim];
    const v = s.v[0 .. seq * dims.kv_dim];
    try ops.matmul.matmul(io, gpa, q, normed, seq, layer.q, null);
    try ops.matmul.matmul(io, gpa, k, normed, seq, layer.k, null);
    // V BEFORE q/k-norm+rope mutate k: gemma4 global layers have no v_proj and
    // reuse the RAW K projection as V.
    if (comptime spec.optional_v_proj) {
        if (layer.v) |vw| {
            try ops.matmul.matmul(io, gpa, v, normed, seq, vw, null);
        } else {
            @memcpy(v, k);
        }
    } else {
        try ops.matmul.matmul(io, gpa, v, normed, seq, layer.v, null);
    }
    if (comptime spec.qk_norm) {
        ops.norm.rmsNorm(q, q, layer.q_norm, eps); // per-head: rows of head_dim
        ops.norm.rmsNorm(k, k, layer.k_norm, eps);
    }
    if (comptime spec.v_norm_unit) ops.norm.rmsNormUnit(v, v, dims.head_dim, eps);
    return .{ .q = q, .k = k, .v = v };
}

/// One decoder layer over `x` [seq, hidden], residuals added in place.
///   - `.fresh`: full-sequence encode, no persistent cache (Krea 2 text encoder).
///   - `.cached`: prefill/decode against `cache` at absolute base `pos0`.
/// `freqs` is the RoPE table the caller selected for this layer; `eps` is the
/// RMSNorm epsilon (module const for qwen3, cfg.rms_eps for gemma).
pub fn layerForward(
    comptime spec: LayerSpec,
    comptime mode: Mode,
    io: std.Io,
    gpa: std.mem.Allocator,
    layer: anytype,
    x: []f32,
    seq: usize,
    dims: Dims,
    freqs: ops.rope.Freqs,
    eps: f32,
    cache: anytype,
    l: usize,
    pos0: usize,
    // Bidirectional attention. In `.cached` prefill: this whole `seq` batch is
    // one image-token block that attends itself in full (forward + backward),
    // causal only to the prefix. In `.fresh`: makes the full-sequence pass a
    // non-causal encoder (EmbeddingGemma) instead of the default causal encoder.
    bidirectional: bool,
    s: anytype,
) !void {
    const normed = s.normed[0 .. seq * dims.hidden];
    const attn_out = s.attn_out[0 .. seq * dims.q_dim];
    const tmp = s.tmp[0 .. seq * dims.hidden];

    // --- Attention ---
    ops.norm.rmsNorm(normed, x, layer.input_norm, eps);
    const qkv = try qkvProject(spec, io, gpa, layer, normed, seq, dims, eps, s);
    // applyRotateHalfAt at pos0 == applyRotateHalf for pos0 == 0 (fresh).
    ops.rope.applyRotateHalfAt(qkv.q, freqs, pos0, seq, dims.n_heads, dims.head_dim);
    ops.rope.applyRotateHalfAt(qkv.k, freqs, pos0, seq, dims.n_kv, dims.head_dim);

    if (comptime mode == .cached) {
        cache.write(l, qkv.k, qkv.v);
        try ops.attention.attention(io, gpa, attn_out, qkv.q, cache.kView(l, seq), cache.vView(l, seq), .{
            .seq_q = seq,
            .seq_kv = pos0 + seq,
            .n_heads = dims.n_heads,
            .n_kv_heads = dims.n_kv,
            .head_dim = dims.head_dim,
            .causal = true,
            .scale = spec.attn_scale,
            .window = dims.sliding_window,
            .bidirectional = bidirectional,
            // LOCAL ring layers store row = pos%ring; kView returns the ring block.
            .ring = cache.ringOf(l),
        });
    } else {
        // `.fresh` full-sequence encode. Causal by default (Qwen text encoder,
        // an autoregressive LM used as an encoder); `bidirectional` flips it to
        // a non-causal encoder (EmbeddingGemma / bidirectional embedding towers)
        // where every position attends the whole sequence. No KV prefix exists
        // here, so seq_kv == seq and causal=false is a plain full attention.
        try ops.attention.attention(io, gpa, attn_out, qkv.q, qkv.k, qkv.v, .{
            .seq_q = seq,
            .seq_kv = seq,
            .n_heads = dims.n_heads,
            .n_kv_heads = dims.n_kv,
            .head_dim = dims.head_dim,
            .causal = !bidirectional,
            .scale = spec.attn_scale,
            .window = dims.sliding_window,
        });
    }
    try ops.matmul.matmul(io, gpa, tmp, attn_out, seq, layer.o, null);
    if (comptime spec.sandwich_norms) ops.norm.rmsNorm(tmp, tmp, layer.post_attn_norm, eps);
    for (x, tmp) |*xi, ti| xi.* += ti;

    // --- MLP ---
    try mlpBlock(spec, io, gpa, layer, x, seq, dims, eps, s);

    // --- Per-layer output scale (gemma4) ---
    if (comptime spec.out_scale) {
        if (layer.out_scale != 1.0) for (x) |*xi| {
            xi.* *= layer.out_scale;
        };
    }
}

/// Batched `.fresh` encoder layer: B independent sequences packed contiguously
/// into `x` [total_rows, hidden] (ragged — item i occupies rows
/// [row_off[i], row_off[i+1])). Every projection / norm / activation runs once
/// over all `total` rows (via the shared `qkvProject`/`mlpBlock`), so the batch
/// amortizes fork/join and weight reuse; only RoPE and attention are per-item
/// (each sequence attends only itself). Produces the identical result to calling
/// `layerForward(.fresh)` on each item. Encoder-only (no KV cache); reuses the
/// caller's per-layer `freqs`/`dims` exactly as `layerForward` does.
pub fn layerForwardBatchedFresh(
    comptime spec: LayerSpec,
    io: std.Io,
    gpa: std.mem.Allocator,
    layer: anytype,
    x: []f32,
    row_off: []const usize, // len B+1, cumulative row offsets; row_off[B] == total
    dims: Dims,
    freqs: ops.rope.Freqs,
    eps: f32,
    bidirectional: bool,
    s: anytype,
) !void {
    const b = row_off.len - 1;
    const total = row_off[b];
    const normed = s.normed[0 .. total * dims.hidden];
    const attn_out = s.attn_out[0 .. total * dims.q_dim];
    const tmp = s.tmp[0 .. total * dims.hidden];

    // --- Attention (projections + norms batched over all rows) ---
    ops.norm.rmsNorm(normed, x, layer.input_norm, eps);
    const qkv = try qkvProject(spec, io, gpa, layer, normed, total, dims, eps, s);
    for (0..b) |i| {
        const L = row_off[i + 1] - row_off[i];
        const qs = row_off[i] * dims.q_dim;
        const ks = row_off[i] * dims.kv_dim;
        ops.rope.applyRotateHalfAt(qkv.q[qs..][0 .. L * dims.q_dim], freqs, 0, L, dims.n_heads, dims.head_dim);
        ops.rope.applyRotateHalfAt(qkv.k[ks..][0 .. L * dims.kv_dim], freqs, 0, L, dims.n_kv, dims.head_dim);
        try ops.attention.attention(io, gpa, attn_out[qs..][0 .. L * dims.q_dim], qkv.q[qs..][0 .. L * dims.q_dim], qkv.k[ks..][0 .. L * dims.kv_dim], qkv.v[ks..][0 .. L * dims.kv_dim], .{
            .seq_q = L,
            .seq_kv = L,
            .n_heads = dims.n_heads,
            .n_kv_heads = dims.n_kv,
            .head_dim = dims.head_dim,
            .causal = !bidirectional,
            .scale = spec.attn_scale,
            .window = dims.sliding_window,
        });
    }
    try ops.matmul.matmul(io, gpa, tmp, attn_out, total, layer.o, null);
    if (comptime spec.sandwich_norms) ops.norm.rmsNorm(tmp, tmp, layer.post_attn_norm, eps);
    for (x, tmp) |*xi, ti| xi.* += ti;

    // --- MLP (batched over all rows) ---
    try mlpBlock(spec, io, gpa, layer, x, total, dims, eps, s);

    if (comptime spec.out_scale) {
        if (layer.out_scale != 1.0) for (x) |*xi| {
            xi.* *= layer.out_scale;
        };
    }
}

/// Tree-verify decoder layer (speculative tree drafting): `n` tree nodes at the
/// per-node absolute `positions`, attending the committed cache prefix plus each
/// node's ancestor chain (`parents`), WITHOUT committing to the cache. This
/// layer's K/V rows are retained in `tree_k`/`tree_v` ([n_layers, n, kv_dim],
/// sliced here by `l`) so the caller can commit the accepted path. qwen3-only,
/// but written against the spec so it stays consistent.
pub fn layerForwardTree(
    comptime spec: LayerSpec,
    io: std.Io,
    gpa: std.mem.Allocator,
    layer: anytype,
    x: []f32,
    n: usize,
    dims: Dims,
    freqs: ops.rope.Freqs,
    positions: []const usize,
    parents: []const u32,
    cache: anytype,
    l: usize,
    tree_k: []f32,
    tree_v: []f32,
    eps: f32,
    s: anytype,
) !void {
    const normed = s.normed[0 .. n * dims.hidden];
    const attn_out = s.attn_out[0 .. n * dims.q_dim];
    const tmp = s.tmp[0 .. n * dims.hidden];

    // --- Attention (tree) ---
    ops.norm.rmsNorm(normed, x, layer.input_norm, eps);
    const qkv = try qkvProject(spec, io, gpa, layer, normed, n, dims, eps, s);
    ops.rope.applyRotateHalfPos(qkv.q, freqs, positions, dims.n_heads, dims.head_dim);
    ops.rope.applyRotateHalfPos(qkv.k, freqs, positions, dims.n_kv, dims.head_dim);

    const lk = tree_k[l * n * dims.kv_dim ..][0 .. n * dims.kv_dim];
    const lv = tree_v[l * n * dims.kv_dim ..][0 .. n * dims.kv_dim];
    @memcpy(lk, qkv.k);
    @memcpy(lv, qkv.v);
    try ops.attention.attentionTree(gpa, attn_out, qkv.q, cache.kView(l, 0), cache.vView(l, 0), lk, lv, parents, .{
        .n_heads = dims.n_heads,
        .n_kv_heads = dims.n_kv,
        .head_dim = dims.head_dim,
    });
    try ops.matmul.matmul(io, gpa, tmp, attn_out, n, layer.o, null);
    for (x, tmp) |*xi, ti| xi.* += ti;

    // --- MLP ---
    try mlpBlock(spec, io, gpa, layer, x, n, dims, eps, s);
}

// --- tests -----------------------------------------------------------------
//
// Ungated CPU guard: for a causal model, one full-sequence `.fresh` pass must
// produce the same final-token hidden state as feeding the tokens one at a time
// through the KV-cached `.cached` path (the prefill == sequential-decode
// property the engine relies on). This exercises rope positioning, cache views,
// residual wiring, norm placement, and the activation on tiny synthetic weights
// — no checkpoint, runs in `zig build test`.

const kv_cache = @import("tp_core").kv_cache;

// Test layer carrying every field the specs touch (v: Weight, so it covers the
// qwen3 + gemma3 shape; gemma4's ?Weight/out_scale variant is validated via
// tp-llm on GPU). `out_scale` defaults to 1.0 (a no-op for the tested specs).
const TLayer = struct {
    input_norm: []const f32,
    q: ops.matmul.Weight,
    k: ops.matmul.Weight,
    v: ops.matmul.Weight,
    o: ops.matmul.Weight,
    q_norm: []const f32,
    k_norm: []const f32,
    post_norm: []const f32, // qwen3 pre-MLP norm
    post_attn_norm: []const f32, // gemma sandwich
    pre_ffn_norm: []const f32,
    post_ffn_norm: []const f32,
    gate: ops.matmul.Weight,
    up: ops.matmul.Weight,
    down: ops.matmul.Weight,
    out_scale: f32 = 1.0,
};

const TScratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,
};

fn freshVsCachedEquiv(comptime spec: LayerSpec, dims: Dims, n_layers: usize) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const rand = prng.random();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const hidden = dims.hidden;
    const q_dim = dims.q_dim;
    const kv_dim = dims.kv_dim;
    const inter = dims.intermediate;

    const rw = struct {
        fn mat(al: std.mem.Allocator, r: std.Random, rows: usize, cols: usize) !ops.matmul.Weight {
            const d = try al.alloc(f32, rows * cols);
            for (d) |*e| e.* = (r.float(f32) - 0.5) * 0.2;
            return ops.matmul.Weight.fromF32(d, rows, cols);
        }
        fn vec(al: std.mem.Allocator, r: std.Random, len: usize) ![]f32 {
            const d = try al.alloc(f32, len);
            for (d) |*e| e.* = 1.0 + (r.float(f32) - 0.5) * 0.1; // norm weights ~1
            return d;
        }
    };

    const layers = try a.alloc(TLayer, n_layers);
    for (layers) |*ly| {
        ly.* = .{
            .input_norm = try rw.vec(a, rand, hidden),
            .q = try rw.mat(a, rand, q_dim, hidden),
            .k = try rw.mat(a, rand, kv_dim, hidden),
            .v = try rw.mat(a, rand, kv_dim, hidden),
            .o = try rw.mat(a, rand, hidden, q_dim),
            .q_norm = try rw.vec(a, rand, dims.head_dim),
            .k_norm = try rw.vec(a, rand, dims.head_dim),
            .post_norm = try rw.vec(a, rand, hidden),
            .post_attn_norm = try rw.vec(a, rand, hidden),
            .pre_ffn_norm = try rw.vec(a, rand, hidden),
            .post_ffn_norm = try rw.vec(a, rand, hidden),
            .gate = try rw.mat(a, rand, inter, hidden),
            .up = try rw.mat(a, rand, inter, hidden),
            .down = try rw.mat(a, rand, hidden, inter),
        };
    }

    const seq = 3;
    const embeds = try a.alloc(f32, seq * hidden);
    for (embeds) |*e| e.* = (rand.float(f32) - 0.5) * 0.5;

    const freqs = try ops.rope.rotateHalfFreqs(a, seq + 2, dims.head_dim, 10000.0);
    const scr: TScratch = .{
        .normed = try a.alloc(f32, seq * hidden),
        .tmp = try a.alloc(f32, seq * hidden),
        .q = try a.alloc(f32, seq * q_dim),
        .k = try a.alloc(f32, seq * kv_dim),
        .v = try a.alloc(f32, seq * kv_dim),
        .attn_out = try a.alloc(f32, seq * q_dim),
        .gate = try a.alloc(f32, seq * inter),
        .up = try a.alloc(f32, seq * inter),
    };

    // Fresh: one full-sequence pass.
    const x_fresh = try a.alloc(f32, seq * hidden);
    @memcpy(x_fresh, embeds);
    for (layers) |*ly| {
        try layerForward(spec, .fresh, io, gpa, ly, x_fresh, seq, dims, freqs, 1e-6, {}, 0, 0, false, &scr);
    }

    // Cached: feed tokens one at a time into a KV cache.
    var cache = try kv_cache.KvCache.init(gpa, n_layers, seq + 2, kv_dim, .f32);
    defer cache.deinit(gpa);
    const xt = try a.alloc(f32, hidden);
    const last = try a.alloc(f32, hidden);
    for (0..seq) |t| {
        @memcpy(xt, embeds[t * hidden ..][0..hidden]);
        const pos0 = cache.len;
        for (layers, 0..) |*ly, l| {
            try layerForward(spec, .cached, io, gpa, ly, xt, 1, dims, freqs, 1e-6, &cache, l, pos0, false, &scr);
        }
        cache.commit(1);
        @memcpy(last, xt);
    }

    // Final token must match between the two paths.
    for (x_fresh[(seq - 1) * hidden ..][0..hidden], last) |fv, cv| {
        try std.testing.expectApproxEqAbs(fv, cv, 1e-4);
    }
}

test "fresh vs cached equivalence — qwen3 spec (silu, pre-norm)" {
    try freshVsCachedEquiv(qwen3_spec, .{
        .hidden = 8,
        .n_heads = 2,
        .n_kv = 1,
        .head_dim = 4,
        .q_dim = 8,
        .kv_dim = 4,
        .intermediate = 16,
    }, 2);
}

test "fresh vs cached equivalence — gemma3 spec (gelu, sandwich norms)" {
    try freshVsCachedEquiv(gemma3_spec, .{
        .hidden = 8,
        .n_heads = 2,
        .n_kv = 1,
        .head_dim = 4,
        .q_dim = 8,
        .kv_dim = 4,
        .intermediate = 16,
    }, 2);
}
