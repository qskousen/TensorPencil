//! GPU-resident Qwen3-VL-4B text encoder (Krea 2 conditioning).
//!
//! Mirrors `qwen3.TextEncoder.encode` with the whole 35-layer transformer on
//! the device in a single batched submission — one upload of the embedded
//! tokens in, one download of the 12-tap conditioning stack out. This
//! replaces the CPU forward's sync-per-GEMM offload (a CPU<->GPU ping-pong
//! that kept the GPU idle and made the encode latency-bound); here every op
//! is recorded into one command buffer so the GPU stays saturated.
//!
//! Parity-first: f32-accumulate GEMMs (opMatmul) and the f32 attention path
//! (attn_scores + fused-online-softmax attn_out, causal), so numerics match
//! the CPU forward up to reordering. Verified against the same ComfyUI text
//! fixture as the CPU test.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const gpu = @import("tp_gpu").context;
const safetensors = @import("tp_core").safetensors;
const ops = @import("tp_ops");
const spec = @import("../llm/spec.zig");
const spec_limits = @import("tp_core").spec_limits;
const sample = @import("tp_core").sample;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

const hidden = qwen3.hidden;
const n_heads = qwen3.n_heads;
const kv_heads = qwen3.n_kv_heads;
const hd = qwen3.head_dim;
const half = hd / 2;
const q_dim = n_heads * hd;
const kv_dim = kv_heads * hd;
const intermediate = qwen3.intermediate;
const n_layers = qwen3.n_layers;
const tap_count = qwen3.tap_count;
const eps = qwen3.rms_eps;
const attn_scale: f32 = 1.0 / @sqrt(@as(f32, hd));

const Buf = gpu.DeviceBuffer;

extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Encode token ids to the Krea 2 conditioning stack, [seq][tap_count][hidden]
/// (same layout the CPU `encode` returns). Caller frees the result.
pub fn encode(enc: *const qwen3.TextEncoder, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, use_f16: bool, cancel: ?*std.atomic.Value(bool)) ![]f32 {
    _ = io;
    const seq = ids.len;
    std.debug.assert(seq > 0);

    // CPU: embedding gather (bf16 -> f32) and the rotate-half rope table.
    const x = try gpa.alloc(f32, seq * hidden);
    defer gpa.free(x);
    for (ids, 0..) |id, t| {
        if (id >= qwen3.vocab_size) return error.TokenIdOutOfRange;
        const row = enc.embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
        try safetensors.convertToF32(.bf16, row, x[t * hidden ..][0..hidden]);
    }

    var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, hd, qwen3.rope_theta);
    defer freqs.deinit(gpa);
    const fp = try gpa.alloc(f32, 2 * seq * half);
    defer gpa.free(fp);
    @memcpy(fp[0 .. seq * half], freqs.cos);
    @memcpy(fp[seq * half ..], freqs.sin);
    const sin_off: u32 = @intCast(seq * half);

    // Coop (tensor-core) GEMMs when requested and available: f32 activations
    // in, f16 weights/accumulate, f32 out — GEMM outputs are 128-row padded.
    // f16 shaves ~0.4s off the encode but ~doubles its image-delta
    // contribution (0.23% -> 0.49%), so the caller decides; default is f32.
    const coop = use_f16 and ctx.pipe_coop != .null_handle;
    const seq_pad = std.mem.alignForward(usize, seq, 128);

    // Device buffers for this (single) forward.
    var bufs = try Bufs.init(ctx, seq, seq_pad);
    defer bufs.deinit(ctx);
    var freqs_d = try ctx.tensorCreate(fp.len * 4);
    defer ctx.tensorDestroy(&freqs_d);
    try ctx.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));
    try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(x));

    const x_d = bufs.x;
    const nd = bufs.normed;
    const q_d = bufs.q;
    const k_d = bufs.k;
    const v_d = bufs.v;
    const qt_d = bufs.qt;
    const kt_d = bufs.kt;
    const s_d = bufs.s;
    const attn_d = bufs.attn;
    const g_d = bufs.gate;
    const u_d = bufs.up;
    const t_d = bufs.t;
    const out_d = bufs.out;

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();

    var tap_idx: usize = 0;
    for (0..n_layers) |l| {
        // Poll cancel between layers so a stop lands mid-encode; the errdefer
        // above aborts the in-flight batch.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (tap_idx < qwen3.tap_layers.len and qwen3.tap_layers[tap_idx] == l) {
            // Snapshot the hidden state entering layer l into the tap-major
            // output buffer (contiguous copy with a per-tap offset).
            try ctx.opElt(.copy, x_d, out_d, null, null, .{
                .u0 = @intCast(seq * hidden),
                .u2 = @intCast(tap_idx * seq * hidden),
            }, seq * hidden, 1, 1);
            tap_idx += 1;
        }
        if (l >= enc.layers.len) break;
        const layer = enc.layers[l];

        // --- Attention ---
        try rmsnorm(ctx, x_d, nd, try nbuf(ctx, layer.input_norm), seq, hidden);
        try gemm(ctx, coop, q_d, nd, seq, seq_pad, layer.q);
        try gemm(ctx, coop, k_d, nd, seq, seq_pad, layer.k);
        try gemm(ctx, coop, v_d, nd, seq, seq_pad, layer.v);
        // Per-head QK-norm (rows of head_dim), then rotate-half rope.
        try rmsnorm(ctx, q_d, q_d, try nbuf(ctx, layer.q_norm), seq * n_heads, hd);
        try rmsnorm(ctx, k_d, k_d, try nbuf(ctx, layer.k_norm), seq * kv_heads, hd);
        try ctx.opElt(.rope_half, q_d, null, freqs_d, null, .{
            .u0 = @intCast(seq * n_heads * half),
            .u1 = half,
            .u2 = sin_off,
            .u3 = n_heads,
        }, seq * n_heads * half, 1, 1);
        try ctx.opElt(.rope_half, k_d, null, freqs_d, null, .{
            .u0 = @intCast(seq * kv_heads * half),
            .u1 = half,
            .u2 = sin_off,
            .u3 = kv_heads,
        }, seq * kv_heads * half, 1, 1);
        // Causal attention: gather k-major, raw scores, fused-softmax P@V.
        try ctx.opElt(.gather_kmajor, q_d, null, null, qt_d, .{
            .u0 = @intCast(seq * n_heads * hd),
            .u1 = n_heads,
            .u2 = hd,
            .u3 = @intCast(seq),
        }, seq * n_heads * hd, 1, 1);
        try ctx.opElt(.gather_kmajor, k_d, null, null, kt_d, .{
            .u0 = @intCast(seq * kv_heads * hd),
            .u1 = kv_heads,
            .u2 = hd,
            .u3 = @intCast(seq),
        }, seq * kv_heads * hd, 1, 1);
        const dc8 = std.math.divCeil(usize, seq, 8) catch unreachable;
        try ctx.opElt(.attn_scores, qt_d, kt_d, null, s_d, .{
            .u0 = @intCast(seq),
            .u1 = n_heads,
            .u2 = kv_heads,
            .u3 = hd,
            .u4 = 0,
            .f0 = attn_scale,
        }, dc8, dc8, n_heads);
        try ctx.opElt(.attn_out, s_d, null, v_d, attn_d, .{
            .u0 = @intCast(seq),
            .u1 = n_heads,
            .u2 = kv_heads,
            .u3 = hd,
            .u4 = 0,
            .u5 = @intCast(seq),
            .f0 = @bitCast(@as(u32, @intCast(seq * seq))),
            .f1 = @bitCast(@as(u32, 1)), // causal
        }, hd / 8, dc8, n_heads);
        try gemm(ctx, coop, t_d, attn_d, seq, seq_pad, layer.o);
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * hidden) }, seq * hidden, 1, 1);

        // --- MLP (SwiGLU) ---
        try rmsnorm(ctx, x_d, nd, try nbuf(ctx, layer.post_norm), seq, hidden);
        try gemm(ctx, coop, g_d, nd, seq, seq_pad, layer.gate);
        try gemm(ctx, coop, u_d, nd, seq, seq_pad, layer.up);
        try ctx.opElt(.silu_mul, g_d, u_d, null, null, .{ .u0 = @intCast(seq * intermediate) }, seq * intermediate, 1, 1);
        try gemm(ctx, coop, t_d, g_d, seq, seq_pad, layer.down);
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * hidden) }, seq * hidden, 1, 1);
    }
    std.debug.assert(tap_idx == tap_count);

    try ctx.endBatch();

    // Download tap-major [tap][seq][hidden] and transpose to the token-major
    // [seq][tap][hidden] layout the DiT context expects.
    const tap_major = try gpa.alloc(f32, tap_count * seq * hidden);
    defer gpa.free(tap_major);
    try ctx.tensorDownload(out_d, std.mem.sliceAsBytes(tap_major));

    const out = try gpa.alloc(f32, seq * tap_count * hidden);
    errdefer gpa.free(out);
    for (0..tap_count) |tp| {
        for (0..seq) |t| {
            const src = tap_major[(tp * seq + t) * hidden ..][0..hidden];
            @memcpy(out[(t * tap_count + tp) * hidden ..][0..hidden], src);
        }
    }
    return out;
}

/// Map a weight's storage dtype to the dense-GEMV weight code. Only fp8 / bf16
/// / f32 reach the Vulkan GEMV path (qwen3 rejects block-quant on vulkan); bf16
/// is read natively (no widening) via the 2-byte transpose + GEMV bf16 branch.
fn wcode(dt: @import("tp_core").dtype.DType) gpu.WCode {
    return switch (dt) {
        .f8_e4m3 => .f8,
        .bf16 => .bf16,
        else => .f32,
    };
}

fn gemm(ctx: *gpu.Context, coop: bool, y: Buf, x: Buf, m: usize, m_pad: usize, w: ops.matmul.Weight) !void {
    if (coop) {
        try ctx.opMatmulCoop(y, x, m, m_pad, w.bytes, w.rows, w.cols, w.scale);
    } else {
        try ctx.opMatmul(y, 0, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, null);
    }
}

fn rmsnorm(ctx: *gpu.Context, in: Buf, out: Buf, weight: Buf, rows: usize, dim: usize) !void {
    try ctx.opElt(.rmsnorm, in, out, weight, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
fn nbuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}

const Bufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    qt: Buf,
    kt: Buf,
    s: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    out: Buf,

    fn init(ctx: *gpu.Context, seq: usize, seq_pad: usize) !Bufs {
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        // GEMM outputs (q/k/v/attn/gate/up/t) are 128-row padded for the coop
        // path (pad rows written zero); the rest are indexed by real seq.
        const sizes = [fields.len]usize{
            seq * hidden * 4, // x
            seq * hidden * 4, // normed
            seq_pad * q_dim * 4, // q
            seq_pad * kv_dim * 4, // k
            seq_pad * kv_dim * 4, // v
            seq * q_dim * 4, // qt (k-major)
            seq * kv_dim * 4, // kt (k-major)
            n_heads * seq * seq * 4, // s (all heads batched)
            seq_pad * q_dim * 4, // attn
            seq_pad * intermediate * 4, // gate
            seq_pad * intermediate * 4, // up
            seq_pad * hidden * 4, // t
            tap_count * seq * hidden * 4, // out (tap-major)
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "qt", "kt", "s", "attn", "gate", "up", "t", "out" };
};

/// KV-cached causal LM on the Vulkan backend (tp-llm --backend vulkan).
/// Config-driven mirror of qwen3_cuda.CudaLM: the whole `cfg.n_layers` stack
/// runs device-resident, one batched submission per step.
///
/// Two weight regimes share this stepper:
///   * Dense (bf16/fp8, tied head — the Qwen3-VL text encoder checkpoints):
///     prefill (seq > 1, empty cache) reuses the encoder's square attention
///     (gather_kmajor + attn_scores + attn_out); decode uses the flash-decoding
///     attn_dsplit/attn_dmerge pair. GEMMs are the f32-accumulate kernel with
///     m = seq (m = 1 is a GEMV). The tied LM head is the embedding converted
///     to f32 once and split into 4 vocab chunks (each under the kernels' 1 GiB
///     type-level buffer bound). Speculative decode (stepAll) is supported.
///   * Block-quant (GGUF q8_0/q4_k/q5_k/q6_k/iq4_nl layers, F16 embed, untied
///     block-quant head — the llama/Mistral arch): every weight matmul routes
///     through `gemvW` (per-row fused-dequant GEMV; no GEMM path on Vulkan), so
///     the whole forward — prefill included — runs one token at a time. The
///     head is a separate block-quant tensor; the F16 embedding is host-gathered
///     via the same f32 copy the dense path uses.
///
/// Hidden-dim norms take the 3-pass parallel rmsnorm (rms_partial/rms_combine/
/// rms_apply_w — one thread per row would serialize rows = 1). Optional per-head
/// QK-norm (`cfg.qk_norm`; llama/Mistral omit it). eps / rope θ / vocab all come
/// from `cfg`.
pub const VulkanLM = struct {
    lm: *const qwen3.CausalLM,
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    /// Model shape (mirrors lm.cfg): dims, vocab, eps, rope θ, qk_norm.
    cfg: qwen3.Config,
    /// Block-quant layer weights (GGUF llama/Mistral): decode is a fused
    /// per-row dequant GEMV. Dense (bf16/fp8) models keep the grouped-GEMV / GEMM
    /// paths.
    quant: bool,
    /// Block-quant prefill runs the tensor-core GEMM (dequant→f16→coopmat) in
    /// one batched pass over the whole prompt instead of a forward per token —
    /// but only when the device has the f16-weight coopmat pipeline. Without it,
    /// prefill falls back to the one-token-at-a-time GEMV.
    can_gemm_prefill: bool,
    /// Route q8_0/iq4_nl through the int8 dp4a GEMV + repacked int8-interleaved
    /// weight layout (TP_VK_DP4A set + device support). ~2.2× faster decode,
    /// opt-in because the repacked weight ~doubles the iq4_nl VRAM footprint.
    use_dp4a: bool,
    /// Route the wide hidden-dim RMSNorm through the one-pass subgroup-reduce
    /// kernel (rmsnorm_sg) instead of the 3-pass rms_partial/combine/apply_w
    /// global round-trip. Requires device subgroup support; opt-in via
    /// TP_VK_SG_RMS while it's being verified against the multi-pass path.
    use_sg_rms: bool,
    /// Route block-quant decode GEMV through the cooperative subgroup kernel
    /// (opGemvQuantSg): raw row-major weight, one subgroup per row, subgroup
    /// reduce — drops the 32-row-group `_t` transpose AND the dp4a repack. Opt-in
    /// via TP_VK_SG_GEMV while it's A/B'd against opGemvQuantT / dp4a.
    use_sg_gemv: bool,
    /// Cooperative dp4a decode GEMV for q8_0/iq4_nl (opGemvQuantSgDp4a): dp4a
    /// speed WITHOUT the repack's ~2× VRAM. Opt-in via TP_VK_SG_DP4A. Takes
    /// precedence over use_dp4a / use_sg_gemv for those two dtypes.
    use_sg_dp4a: bool,
    /// dp4a decode GEMV over the _t layout + k-split (opGemvQuantTDp4a): the
    /// fast repack-dp4a shape with NO int8 repack — reuses the resident _t
    /// buffer (shared with prefill; no cache collision, no VRAM increase). Opt-in
    /// via TP_VK_T_DP4A for q8_0/iq4_nl. Takes precedence over the others.
    use_t_dp4a: bool,
    /// Zero bias for the prefill GEMM projections (the LLM carries no bias);
    /// sized to the largest output dim, passed whole so the cached device
    /// buffer covers every projection's row count.
    zero_bias: []f32,
    /// LM-head vocab-chunk size (dense tied head only); cfg.vocab / vocab_chunks.
    chunk_rows: usize,
    capacity: usize,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    max_rows: usize,
    sin_off: u32,
    /// Token embedding converted to f32 (bf16 or f16 source): the CPU-side
    /// gather scratch, and — for the dense tied-head path — the LM-head weight
    /// source (weightBuffer caches by host pointer, so this must stay alive).
    /// Block-quant models use it only for the gather; their head is `lm.head`.
    embed_f32: []f32,
    k_cache: [qwen3.Config.max_layers]Buf,
    v_cache: [qwen3.Config.max_layers]Buf,
    freqs_d: Buf,
    bufs: LmBufs,

    /// LM-head vocab split (dense tied head): 4 chunks so each stays under the
    /// kernels' 1 GiB type-level buffer bound (cfg.vocab must divide by 4).
    const vocab_chunks = 4;
    /// KV chunks per head in the decode attention split pass.
    const nsplit = 128;
    /// Largest batch that runs the small-batch path (4-input grouped GEMVs +
    /// batched flash-decoding): every speculative verify batch, and the
    /// chunk size for follow-up (pos0 > 0) prefills — which previously went
    /// token-by-token because the square attention kernel is pos0=0-only.
    const gemv_batch_max = spec_limits.max_draft + 1;
    /// Interleaved chunks per row in the 3-pass rmsnorm.
    const rms_chunks = 64;
    /// Interleaved k chunks per output column in the decode GEMV.
    const gemv_nchunk = 32;
    /// Fresh-prompt length at/above which block-quant prefill switches to the
    /// tensor-core GEMM. Below it, the per-token dequant GEMV wins: the GEMM
    /// dequants every weight once up front (~a full extra weight pass), which
    /// only pays off once that fixed cost beats `m` single-token forwards
    /// (~crossover a couple dozen tokens on the 3090).
    const prefill_gemm_min = 32;

    /// Device bytes the KV cache reserves up front for `capacity` tokens (k +
    /// v across all layers). Vulkan has no growable buffers, so this whole
    /// window is allocated in init() — callers sizing a default weight-pin
    /// budget must leave this much VRAM unpinned for it.
    pub fn kvWindowBytes(cfg: qwen3.Config, capacity: usize) usize {
        return 2 * cfg.n_layers * capacity * cfg.kvDim() * 4;
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen3.CausalLM, capacity: usize, first_seq: usize) !VulkanLM {
        const c = lm.cfg;
        if (c.n_layers > qwen3.Config.max_layers) return error.UnsupportedModelConfig;
        // The embedding table is host-gathered into an f32 copy — bf16 / f16
        // only (no Vulkan block-quant gather kernel; those checkpoints run on
        // cpu / zig-cuda / cuda).
        if (lm.embed.dtype != .bf16 and lm.embed.dtype != .f16)
            return error.UnsupportedModelConfig;
        // Block-quant head (llama/Mistral) => per-row GEMV everything. A dense
        // model must use the tied bf16 embedding as its LM head (the 4-chunk
        // head path reads `embed_f32`, which assumes head == embed).
        const quant = lm.head.dtype.isBlockQuant();
        if (!quant) {
            if (lm.embed.dtype != .bf16 or lm.head.bytes.ptr != lm.embed.bytes.ptr)
                return error.UnsupportedModelConfig;
            if (c.vocab % vocab_chunks != 0) return error.UnsupportedModelConfig;
        }

        var self: VulkanLM = undefined;
        self.lm = lm;
        self.ctx = ctx;
        self.gpa = gpa;
        self.cfg = c;
        self.quant = quant;
        self.use_dp4a = quant and ctx.hasIntDot() and getenv("TP_VK_DP4A") != null;
        self.use_sg_rms = ctx.hasSubgroupNorm() and getenv("TP_VK_SG_RMS") != null;
        self.use_sg_gemv = quant and ctx.hasSubgroupGemv() and getenv("TP_VK_SG_GEMV") != null;
        self.use_sg_dp4a = quant and ctx.hasSubgroupDp4a() and getenv("TP_VK_SG_DP4A") != null;
        self.use_t_dp4a = quant and ctx.hasTransposedDp4a() and getenv("TP_VK_T_DP4A") != null;
        // The raw-reading coop GEMV (use_sg_gemv/use_sg_dp4a) reads the RAW
        // weight while the prefill GEMM reads _t/repacked — and the weight cache
        // keys by host pointer, so mixing layouts for one weight returns the
        // wrong bytes. Force token-by-token prefill for those so every weight
        // stays raw-only. use_t_dp4a is exempt: it reads the SAME _t buffer as
        // the prefill, so no collision.
        self.can_gemm_prefill = quant and ctx.hasQuantPrefillGemm() and
            !(self.use_sg_gemv or self.use_sg_dp4a);
        self.chunk_rows = c.vocab / vocab_chunks;
        self.capacity = capacity;
        self.len = 0;
        // Block-quant WITHOUT the prefill GEMM runs one token at a time (no
        // square-attention prefill), so its activation buffers only need the
        // small-batch floor. Everything else (dense, or block-quant with the
        // tensor-core prefill) sizes for the whole first prefill chunk.
        self.max_rows = if (quant and !self.can_gemm_prefill) gemv_batch_max else @max(@max(first_seq, 1), gemv_batch_max);
        self.sin_off = @intCast(capacity * half);

        self.embed_f32 = try gpa.alloc(f32, c.vocab * c.hidden);
        errdefer gpa.free(self.embed_f32);
        try safetensors.convertToF32(lm.embed.dtype, lm.embed.bytes, self.embed_f32);

        // Zero bias for the prefill GEMM projections; sized to the largest
        // output dim so one cached device buffer serves every projection.
        const max_out = @max(@max(c.qDim(), c.kvDim()), @max(c.intermediate, c.hidden));
        self.zero_bias = try gpa.alloc(f32, max_out);
        errdefer gpa.free(self.zero_bias);
        @memset(self.zero_bias, 0);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, capacity, hd, c.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * capacity * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. capacity * half], freqs.cos);
        @memcpy(fp[capacity * half ..], freqs.sin);
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&self.freqs_d);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        const kv_bytes = capacity * c.kvDim() * 4;
        var created: usize = 0;
        errdefer for (self.k_cache[0..created]) |*bf| ctx.tensorDestroy(bf);
        for (self.k_cache[0..c.n_layers]) |*bf| {
            bf.* = try ctx.tensorCreate(kv_bytes);
            created += 1;
        }
        var vcreated: usize = 0;
        errdefer for (self.v_cache[0..vcreated]) |*bf| ctx.tensorDestroy(bf);
        for (self.v_cache[0..c.n_layers]) |*bf| {
            bf.* = try ctx.tensorCreate(kv_bytes);
            vcreated += 1;
        }

        self.bufs = try LmBufs.init(ctx, self.max_rows, c);
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        for (self.k_cache[0..self.cfg.n_layers]) |*bf| self.ctx.tensorDestroy(bf);
        for (self.v_cache[0..self.cfg.n_layers]) |*bf| self.ctx.tensorDestroy(bf);
        self.ctx.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.ctx);
        self.gpa.free(self.embed_f32);
        self.gpa.free(self.zero_bias);
        self.* = undefined;
    }

    pub fn cached(self: *const VulkanLM) usize {
        return self.len;
    }

    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }

    pub fn remaining(self: *const VulkanLM) usize {
        return self.capacity - self.len;
    }

    /// Rows to forward per chunk. Dense: the whole fresh prompt through the
    /// square-attention GEMM path, follow-up (pos0 > 0) through the batched
    /// flash-decoding path. Block-quant WITH the tensor-core prefill: the fresh
    /// prompt in one square-attention chunk (projections via the dequant→f16
    /// GEMM), then one token at a time (decode / short follow-ups stay on the
    /// exact per-row GEMV). Block-quant WITHOUT it: one token at a time always.
    fn chunkRows(self: *const VulkanLM, avail: usize) usize {
        if (self.quant) {
            // GEMM prefill only for a fresh prompt long enough to amortize the
            // dequant-all-weights pass; otherwise one token at a time.
            if (self.can_gemm_prefill and self.len == 0 and avail >= prefill_gemm_min)
                return @min(self.max_rows, avail);
            return @min(@as(usize, 1), avail);
        }
        return if (self.len == 0) @min(self.max_rows, avail) else @min(gemv_batch_max, avail);
    }

    /// Whether a block-quant weight of this dtype routes through the int8 dp4a
    /// path (repacked int8-interleaved layout, ~2.4× decode) — used for BOTH
    /// decode (opGemvDp4a) and prefill (opMatmulCoopQuant repacked=true) so the
    /// weight's cached device layout is consistent (the cache keys by host ptr).
    /// q8_0 is ON by default: its repack is only ~6% larger than raw, so the win
    /// is nearly free. iq4_nl stays opt-in (TP_VK_DP4A): its int8 repack ~doubles
    /// the 4-bit footprint.
    fn dp4aRepack(self: *const VulkanLM, dt: @import("tp_core").dtype.DType) bool {
        return switch (dt) {
            .q8_0 => self.ctx.hasIntDot(),
            .iq4_nl => self.use_dp4a,
            else => false,
        };
    }

    /// A block-quant weight routes through the fused per-row dequant GEMV
    /// (opGemvQuantT, coalesced 32-row-group transpose + k-split); a dense
    /// weight through the bf16/fp8/f32 k-split GEMV. Both write `w.rows`
    /// outputs at element offset `y_off` from a single input vector `x`.
    fn gemvW(self: *VulkanLM, y: Buf, y_off: usize, x: Buf, w: ops.matmul.Weight) !void {
        // dp4a over _t + k-split (q8_0/iq4_nl): repack-dp4a speed, no repack VRAM.
        if (self.use_t_dp4a) switch (w.dtype) {
            .q8_0, .iq4_nl => return self.ctx.opGemvQuantTDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, gemv_nchunk, self.bufs.quant_partials),
            else => {},
        };
        // Cooperative dp4a GEMV (q8_0/iq4_nl): dp4a speed, raw weight, no repack.
        if (self.use_sg_dp4a) switch (w.dtype) {
            .q8_0, .iq4_nl => return self.ctx.opGemvQuantSgDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
            else => {},
        };
        // Cooperative scalar subgroup GEMV: raw layout, no _t transpose / no dp4a
        // repack. Covers all 5 block-quant dtypes; dense still uses the k-split.
        if (self.use_sg_gemv) switch (w.dtype) {
            .q8_0, .iq4_nl, .q4_k, .q5_k, .q6_k => return self.ctx.opGemvQuantSg(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
            else => {},
        };
        switch (w.dtype) {
            // q8_0 / iq4_nl: int8 dp4a decode GEMV over the repacked
            // int8-interleaved layout — MEASURED ~2.4× faster than scalar on the
            // 3090. Default ON for q8_0 (repack ~6% larger than raw); opt-in for
            // iq4_nl (TP_VK_DP4A — its int8 repack ~doubles the 4-bit footprint).
            .q8_0, .iq4_nl => if (self.dp4aRepack(w.dtype))
                try self.ctx.opGemvDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, gemv_nchunk, self.bufs.quant_partials)
            else
                try self.ctx.opGemvQuantT(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, gemv_nchunk, self.bufs.quant_partials),
            .q4_k, .q5_k, .q6_k => try self.ctx.opGemvQuantT(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, gemv_nchunk, self.bufs.quant_partials),
            else => try self.ctx.opGemv(y, y_off, x, self.bufs.gemv_partials[0], w.bytes, wcode(w.dtype), w.rows, w.cols, w.scale, gemv_nchunk),
        }
    }

    /// A block-quant-model projection over `m` rows: the exact per-row dequant
    /// GEMV at m == 1 (decode / short follow-up prefill), else the tensor-core
    /// dequant→f16 GEMM over the whole batch (fresh-prompt prefill — the ~N×
    /// weight-read reuse that turns an O(prompt) stack of forwards into one).
    /// A stray dense weight inside a quant model routes through the dense GEMM.
    fn linearQuant(self: *VulkanLM, y: Buf, x: Buf, m: usize, w: ops.matmul.Weight, rows: usize, cols: usize) !void {
        if (m == 1) return self.gemvW(y, 0, x, w);
        switch (w.dtype) {
            // q8_0 / iq4_nl: dequant from the int8-interleaved layout when the
            // dp4a repack is used for this dtype (shared with decode — one
            // resident copy, consistent cache layout), else from _t.
            .q8_0, .iq4_nl => try self.ctx.opMatmulCoopQuant(w.dtype, y, 0, x, m, w.bytes, rows, cols, w.scale, self.zero_bias, self.dp4aRepack(w.dtype)),
            .q4_k, .q5_k, .q6_k => try self.ctx.opMatmulCoopQuant(w.dtype, y, 0, x, m, w.bytes, rows, cols, w.scale, self.zero_bias, false),
            else => try self.gemm(y, x, m, w, rows, cols),
        }
    }

    /// Single-pass per-head RMSNorm (QK-norm): rows is large (seq * heads), so
    /// one thread per row parallelizes fine, unlike the hidden-dim norm.
    fn rmsRows(self: *VulkanLM, in: Buf, out: Buf, weight: Buf, rows: usize, dim: usize) !void {
        try self.ctx.opElt(.rmsnorm, in, out, weight, null, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(dim),
            .f0 = self.cfg.rms_eps,
        }, rows, 1, 1);
    }

    /// Device VRAM (bytes) this Vulkan context has allocated — the analog of the
    /// CUDA backend's `deviceUsed()`, for the end-of-response telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.ctx.device_used;
    }

    /// Forward `ids` at positions [len, len+ids.len), then write
    /// last-position vocab logits. A fresh-cache prompt chunks by max_rows
    /// through the square-attention GEMM path; follow-up (pos0 > 0) prefills
    /// chunk by gemv_batch_max through the batched flash-decoding path.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n = self.chunkRows(ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], logits);
            off += n;
        }
    }

    /// step, but with vocab logits for every new token ([ids.len, vocab]
    /// row-major) — the speculative-decode verify forward. The batch is
    /// engine-capped at spec_limits.max_draft + 1.
    pub fn stepAll(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        const ctx = self.ctx;
        const seq = ids.len;
        const nvocab = self.cfg.vocab;
        std.debug.assert(logits.len == seq * nvocab);
        std.debug.assert(seq > 0 and seq <= gemv_batch_max);

        // Block-quant has no batched forward — verify each draft position with
        // its own one-token forward (linear chain: position t appends its K/V
        // and attends the committed prefix, exactly as a batched verify would).
        if (self.quant) {
            for (ids, 0..) |id, t| {
                try self.stepChunk(io, &.{id}, logits[t * nvocab ..][0..nvocab]);
            }
            return;
        }

        const b = &self.bufs;
        const hidden_ = self.cfg.hidden;
        try self.layersForward(ids);
        errdefer if (ctx.batching) ctx.abortBatch();

        // Final norm on every new position, then the LM head as 4 vocab
        // chunks x 4-input groups (each weight chunk read once per group).
        try self.normWide(b.x, b.normed, try nbuf(ctx, self.lm.final_norm), seq);
        var g: usize = 0;
        while (g * 4 < seq) : (g += 1) {
            const n: usize = @min(4, seq - g * 4);
            ctx.independent(vocab_chunks);
            for (0..vocab_chunks) |ci| {
                const w = self.embed_f32[ci * self.chunk_rows * hidden_ ..][0 .. self.chunk_rows * hidden_];
                try ctx.opGemvPartial4(b.normed, g * 4 * hidden_, b.gemv_partials[ci], std.mem.sliceAsBytes(w), .f32, self.chunk_rows, hidden_, gemv_nchunk);
            }
            ctx.independent(vocab_chunks);
            for (0..vocab_chunks) |ci| {
                try ctx.opGemvCombine4(b.logits, g * 4 * nvocab + ci * self.chunk_rows, nvocab, b.gemv_partials[ci], self.chunk_rows, 1.0, gemv_nchunk, n);
            }
        }
        try ctx.endBatch();
        self.len += seq;

        try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(logits));
    }

    /// Roll the KV cache back to `new_len` tokens (speculative-decode
    /// rejection); device rows past `new_len` are overwritten by later steps.
    pub fn truncate(self: *VulkanLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    fn stepChunk(self: *VulkanLM, io: std.Io, ids: []const u32, logits: ?[]f32) !void {
        _ = io;
        const ctx = self.ctx;
        const seq = ids.len;
        const b = &self.bufs;
        const hidden_ = self.cfg.hidden;
        const nvocab = self.cfg.vocab;

        try self.layersForward(ids);
        errdefer if (ctx.batching) ctx.abortBatch();

        // Final norm on the last position, then the LM head.
        try ctx.opElt(.copy, b.x, b.t, null, null, .{
            .u0 = @intCast(hidden_),
            .u2 = 0,
            .u3 = @intCast((seq - 1) * hidden_),
        }, hidden_, 1, 1);
        try self.normWide(b.t, b.normed, try nbuf(ctx, self.lm.final_norm), 1);
        if (self.quant) {
            // Untied block-quant head: one fused per-row dequant GEMV.
            try self.gemvW(b.logits, 0, b.normed, self.lm.head);
        } else {
            // Dense tied head: 4 vocab chunks (each weight chunk read once).
            ctx.independent(vocab_chunks);
            for (0..vocab_chunks) |ci| {
                const w = self.embed_f32[ci * self.chunk_rows * hidden_ ..][0 .. self.chunk_rows * hidden_];
                try ctx.opGemvPartial(b.normed, b.gemv_partials[ci], std.mem.sliceAsBytes(w), .f32, self.chunk_rows, hidden_, gemv_nchunk);
            }
            ctx.independent(vocab_chunks);
            for (0..vocab_chunks) |ci| {
                try ctx.opGemvCombine(b.logits, ci * self.chunk_rows, b.gemv_partials[ci], self.chunk_rows, 1.0, gemv_nchunk);
            }
        }
        try ctx.endBatch();
        self.len += seq;

        // `null` leaves the last position's logits resident (stepArgmax runs the
        // device argmax on them instead of paying the ~608 KB vocab download).
        if (logits) |l| try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(l[0..nvocab]));
    }

    /// Greedy decode without the vocab download: forward `ids`, then argmax the
    /// last position's logits on-device and return just that token id. Matches
    /// sample.argmax (temperature 0). The engine uses this for the greedy path.
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits
    /// first (opPenalize; see sample.zig) — keeps penalized greedy decode
    /// on the GPU path instead of the full-vocab download.
    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        var off: usize = 0;
        while (off < ids.len) {
            const n = self.chunkRows(ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], null);
            off += n;
        }
        const ctx = self.ctx;
        const b = &self.bufs;
        try ctx.opPenalize(b.logits, pen, sp);
        try ctx.opArgmax(b.logits, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try ctx.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const VulkanLM) usize {
        _ = self;
        return gpu.topk_lanes * gpu.topk_m;
    }

    /// Stochastic decode: forward `ids`, select the top-k candidates on-device,
    /// and download just those (id,logit) pairs into out_id/out_logit (a few KB
    /// vs the ~608 KB vocab). Returns the candidate count; the engine's Sampler
    /// finishes (softmax/top-p/RNG) on the CPU over this small set.
    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with sampling penalties scattered onto the device logits
    /// before the top-k (opPenalize) — the selected candidates are the true
    /// post-penalty top set, so penalized stochastic decode stays on the GPU.
    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        var off: usize = 0;
        while (off < ids.len) {
            const n = self.chunkRows(ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], null);
            off += n;
        }
        const ctx = self.ctx;
        const b = &self.bufs;
        try ctx.opPenalize(b.logits, pen, sp);
        const count = try ctx.opTopK(b.logits, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try ctx.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try ctx.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    /// The `cfg.n_layers` stack over `ids` at positions [len, len+seq):
    /// embedding upload, then the whole transformer inside an open batch. The
    /// caller finishes the batch (LM head variants differ) — on success the
    /// batch is still open, with the final hidden states in bufs.x.
    fn layersForward(self: *VulkanLM, ids: []const u32) !void {
        const gpa = self.gpa;
        const ctx = self.ctx;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= self.max_rows);
        const pos0 = self.len;
        const hidden_ = self.cfg.hidden;

        // CPU: embedding gather from the f32 copy, upload.
        const x = try gpa.alloc(f32, seq * hidden_);
        defer gpa.free(x);
        for (ids, 0..) |id, t| {
            if (id >= self.cfg.vocab) return error.TokenIdOutOfRange;
            @memcpy(x[t * hidden_ ..][0..hidden_], self.embed_f32[@as(usize, id) * hidden_ ..][0..hidden_]);
        }
        try ctx.tensorUpload(self.bufs.x, std.mem.sliceAsBytes(x));

        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();

        for (self.lm.layers, 0..) |layer, l| {
            try transformer_gpu.decoderLayer(transformer.qwen3_spec, self, layer, l, seq, pos0);
        }
    }

    // --- transformer_gpu.decoderLayer stepper methods (faithful lifts of the
    // former inline layer loop; each keeps its leading independent() hint so op
    // and scheduling order are byte-for-byte preserved). ---

    pub fn normInput(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.normWide(self.bufs.x, self.bufs.normed, try nbuf(self.ctx, layer.input_norm), seq);
    }

    pub fn projectQKV(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l; // qwen3: uniform geometry
        const ctx = self.ctx;
        const b = &self.bufs;
        const c = self.cfg;
        if (self.quant) {
            // Block-quant: per-row dequant GEMV (decode) or tensor-core GEMM (prefill).
            try self.linearQuant(b.q, b.normed, seq, layer.q, c.qDim(), c.hidden);
            try self.linearQuant(b.k, b.normed, seq, layer.k, c.kvDim(), c.hidden);
            try self.linearQuant(b.v, b.normed, seq, layer.v, c.kvDim(), c.hidden);
        } else if (seq == 1) {
            // Group the q/k/v GEMV halves so no barrier drains the GPU between
            // independent dispatches.
            ctx.independent(3);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[0], layer.q.bytes, wcode(layer.q.dtype), c.qDim(), c.hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[1], layer.k.bytes, wcode(layer.k.dtype), c.kvDim(), c.hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[2], layer.v.bytes, wcode(layer.v.dtype), c.kvDim(), c.hidden, gemv_nchunk);
            ctx.independent(3);
            try ctx.opGemvCombine(b.q, 0, b.gemv_partials[0], c.qDim(), layer.q.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.k, 0, b.gemv_partials[1], c.kvDim(), layer.k.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.v, 0, b.gemv_partials[2], c.kvDim(), layer.v.scale, gemv_nchunk);
        } else {
            try self.gemm(b.q, b.normed, seq, layer.q, c.qDim(), c.hidden);
            try self.gemm(b.k, b.normed, seq, layer.k, c.kvDim(), c.hidden);
            try self.gemm(b.v, b.normed, seq, layer.v, c.kvDim(), c.hidden);
        }
    }

    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const c = self.cfg;
        if (!c.qk_norm) return; // llama/Mistral: no per-head QK-norm
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq == 1) ctx.independent(2);
        try self.rmsRows(b.q, b.q, try nbuf(ctx, layer.q_norm), seq * c.n_heads, hd);
        try self.rmsRows(b.k, b.k, try nbuf(ctx, layer.k_norm), seq * c.n_kv_heads, hd);
    }

    pub fn applyRope(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        _ = l; // qwen3: single rope table for all layers
        const ctx = self.ctx;
        const b = &self.bufs;
        const c = self.cfg;
        if (seq == 1) ctx.independent(2);
        try ctx.opElt(.rope_half, b.q, null, self.freqs_d, null, .{
            .u0 = @intCast(seq * c.n_heads * half),
            .u1 = half,
            .u2 = self.sin_off,
            .u3 = @intCast(c.n_heads),
            .u4 = @intCast(pos0),
        }, seq * c.n_heads * half, 1, 1);
        try ctx.opElt(.rope_half, b.k, null, self.freqs_d, null, .{
            .u0 = @intCast(seq * c.n_kv_heads * half),
            .u1 = half,
            .u2 = self.sin_off,
            .u3 = @intCast(c.n_kv_heads),
            .u4 = @intCast(pos0),
        }, seq * c.n_kv_heads * half, 1, 1);
    }

    pub fn appendKV(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        const kvd = self.cfg.kvDim();
        // Append K/V to the cache (in-batch device copy).
        if (seq == 1) ctx.independent(2);
        try ctx.opElt(.copy, b.k, self.k_cache[l], null, null, .{
            .u0 = @intCast(seq * kvd),
            .u2 = @intCast(pos0 * kvd),
        }, seq * kvd, 1, 1);
        try ctx.opElt(.copy, b.v, self.v_cache[l], null, null, .{
            .u0 = @intCast(seq * kvd),
            .u2 = @intCast(pos0 * kvd),
        }, seq * kvd, 1, 1);
    }

    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        const c = self.cfg;
        if (seq <= gemv_batch_max) {
            // Batched flash-decoding split/merge against the cached prefix:
            // query t sees pos0 + 1 + t keys (causal) — covers decode (seq == 1),
            // speculative verify, and follow-up prefill chunks at any pos0.
            try ctx.opElt(.attn_dsplit, b.q, self.k_cache[l], self.v_cache[l], b.attn_scratch, .{
                .u0 = @intCast(pos0 + 1),
                .u1 = @intCast(c.n_heads),
                .u2 = @intCast(c.n_kv_heads),
                .u3 = hd,
                .u4 = nsplit,
                .u5 = @intCast(seq),
                .f0 = attn_scale,
            }, seq * c.n_heads * nsplit, 1, 1);
            try ctx.opElt(.attn_dmerge, b.attn_scratch, null, null, b.attn, .{
                .u0 = @intCast(seq * c.n_heads),
                .u1 = hd,
                .u2 = nsplit,
            }, seq * c.n_heads * hd, 1, 1);
        } else {
            // Square causal attention (prefill starts from an empty cache).
            std.debug.assert(pos0 == 0);
            try ctx.opElt(.gather_kmajor, b.q, null, null, b.qt, .{
                .u0 = @intCast(seq * c.n_heads * hd),
                .u1 = @intCast(c.n_heads),
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * c.n_heads * hd, 1, 1);
            try ctx.opElt(.gather_kmajor, b.k, null, null, b.kt, .{
                .u0 = @intCast(seq * c.n_kv_heads * hd),
                .u1 = @intCast(c.n_kv_heads),
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * c.n_kv_heads * hd, 1, 1);
            const dc8 = std.math.divCeil(usize, seq, 8) catch unreachable;
            try ctx.opElt(.attn_scores, b.qt, b.kt, null, b.s, .{
                .u0 = @intCast(seq),
                .u1 = @intCast(c.n_heads),
                .u2 = @intCast(c.n_kv_heads),
                .u3 = hd,
                .u4 = 0,
                .f0 = attn_scale,
            }, dc8, dc8, c.n_heads);
            try ctx.opElt(.attn_out, b.s, null, b.v, b.attn, .{
                .u0 = @intCast(seq),
                .u1 = @intCast(c.n_heads),
                .u2 = @intCast(c.n_kv_heads),
                .u3 = hd,
                .u4 = 0,
                .u5 = @intCast(seq),
                .f0 = @bitCast(@as(u32, @intCast(seq * seq))),
                .f1 = @bitCast(@as(u32, 1)), // causal
            }, hd / 8, dc8, c.n_heads);
        }
    }

    pub fn projectO(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const c = self.cfg;
        if (self.quant) {
            try self.linearQuant(self.bufs.t, self.bufs.attn, seq, layer.o, c.hidden, c.qDim());
        } else {
            try self.gemm(self.bufs.t, self.bufs.attn, seq, layer.o, c.hidden, c.qDim());
        }
    }

    pub fn addResidual(self: *VulkanLM, seq: usize) !void {
        const n = seq * self.cfg.hidden;
        try self.ctx.opElt(.add, self.bufs.x, self.bufs.t, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn normPreFfn(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.normWide(self.bufs.x, self.bufs.normed, try nbuf(self.ctx, layer.post_norm), seq);
    }

    pub fn projectGateUp(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        const c = self.cfg;
        if (self.quant) {
            try self.linearQuant(b.gate, b.normed, seq, layer.gate, c.intermediate, c.hidden);
            try self.linearQuant(b.up, b.normed, seq, layer.up, c.intermediate, c.hidden);
        } else if (seq == 1) {
            ctx.independent(2);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[0], layer.gate.bytes, wcode(layer.gate.dtype), c.intermediate, c.hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[1], layer.up.bytes, wcode(layer.up.dtype), c.intermediate, c.hidden, gemv_nchunk);
            ctx.independent(2);
            try ctx.opGemvCombine(b.gate, 0, b.gemv_partials[0], c.intermediate, layer.gate.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.up, 0, b.gemv_partials[1], c.intermediate, layer.up.scale, gemv_nchunk);
        } else {
            try self.gemm(b.gate, b.normed, seq, layer.gate, c.intermediate, c.hidden);
            try self.gemm(b.up, b.normed, seq, layer.up, c.intermediate, c.hidden);
        }
    }

    pub fn activate(self: *VulkanLM, comptime act: transformer.Activation, seq: usize) !void {
        const which: gpu.Elt = switch (act) {
            .silu_mul => .silu_mul,
            .gelu_tanh_mul => .gelu_mul,
        };
        const n = seq * self.cfg.intermediate;
        try self.ctx.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        if (self.quant) {
            try self.linearQuant(self.bufs.t, self.bufs.gate, seq, layer.down, c.hidden, c.intermediate);
        } else {
            try self.gemm(self.bufs.t, self.bufs.gate, seq, layer.down, c.hidden, c.intermediate);
        }
    }

    /// Dense linear over `m` rows, kernel picked by batch size: k-split GEMV
    /// (m = 1), grouped 4-input GEMVs (small batches — speculative verify
    /// and follow-up prefill chunks; bitwise equal to the m = 1 path), or
    /// the tiled GEMM (large fresh prefills). bf16 weights have no tiled GEMM
    /// (only fp8/f32), so bf16 always streams through the grouped GEMV — the
    /// weight is read ceil(m/4)x, matching CUDA's bf16 opGemvBf16N.
    fn gemm(self: *VulkanLM, y: Buf, x: Buf, m: usize, w: ops.matmul.Weight, rows: usize, cols: usize) !void {
        const ctx = self.ctx;
        const wc = wcode(w.dtype);
        if (m == 1) {
            try ctx.opGemv(y, 0, x, self.bufs.gemv_partials[0], w.bytes, wc, rows, cols, w.scale, gemv_nchunk);
        } else if (wc == .bf16 or m <= gemv_batch_max) {
            var g: usize = 0;
            while (g * 4 < m) : (g += 1) {
                const n: usize = @min(4, m - g * 4);
                try ctx.opGemvPartial4(x, g * 4 * cols, self.bufs.gemv_partials[0], w.bytes, wc, rows, cols, gemv_nchunk);
                try ctx.opGemvCombine4(y, g * 4 * rows, rows, self.bufs.gemv_partials[0], rows, w.scale, gemv_nchunk, n);
            }
        } else {
            try ctx.opMatmul(y, 0, x, 0, m, w.bytes, wc == .f8, rows, cols, w.scale, null);
        }
    }

    /// 3-pass parallel rmsnorm over [rows][hidden] (one thread per row would
    /// serialize the decode path's rows = 1).
    fn normWide(self: *VulkanLM, in: Buf, out: Buf, weight: Buf, rows: usize) !void {
        const ctx = self.ctx;
        const h: u32 = @intCast(self.cfg.hidden);
        if (self.use_sg_rms) {
            try ctx.opRmsNormSg(in, out, weight, rows, self.cfg.hidden, self.cfg.rms_eps);
            return;
        }
        try ctx.opElt(.rms_partial, in, null, null, self.bufs.rms_partials, .{
            .u0 = @intCast(rows * rms_chunks),
            .u1 = h,
            .u2 = rms_chunks,
        }, rows * rms_chunks, 1, 1);
        try ctx.opElt(.rms_combine, self.bufs.rms_partials, null, null, self.bufs.rms_inv, .{
            .u0 = @intCast(rows),
            .u1 = h,
            .u2 = rms_chunks,
            .f0 = self.cfg.rms_eps,
        }, rows, 1, 1);
        try ctx.opElt(.rms_apply_w, in, out, weight, self.bufs.rms_inv, .{
            .u0 = @intCast(rows * self.cfg.hidden),
            .u1 = h,
        }, rows * self.cfg.hidden, 1, 1);
    }
};

const LmBufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    qt: Buf,
    kt: Buf,
    s: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    attn_scratch: Buf,
    rms_partials: Buf,
    rms_inv: Buf,
    gemv_partials: [4]Buf,
    /// Per-row dequant GEMV partials (gemvW / opGemvQuantT); block-quant only,
    /// sized for the largest output (the untied head's vocab rows).
    quant_partials: Buf,
    logits: Buf,
    // GPU-argmax scratch (opArgmax): per-lane max value + index, and the 1-id out.
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    // GPU top-k scratch (opTopK): per-lane top-M values + indices.
    topk_v: Buf,
    topk_i: Buf,

    fn init(ctx: *gpu.Context, rows: usize, cfg: qwen3.Config) !LmBufs {
        const hidden_ = cfg.hidden;
        const q_dim_ = cfg.qDim();
        const kv_dim_ = cfg.kvDim();
        const inter = cfg.intermediate;
        const nvocab = cfg.vocab;
        const chunk_rows = nvocab / VulkanLM.vocab_chunks;
        var self: LmBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        const r4 = std.mem.alignForward(usize, rows, 4); // grouped-GEMV inputs are read 4 rows at a time
        const sizes = [fields.len]usize{
            rows * hidden_ * 4, // x
            r4 * hidden_ * 4, // normed
            rows * q_dim_ * 4, // q
            rows * kv_dim_ * 4, // k
            rows * kv_dim_ * 4, // v
            rows * q_dim_ * 4, // qt (prefill k-major)
            rows * kv_dim_ * 4, // kt
            cfg.n_heads * rows * rows * 4, // s (prefill scores)
            r4 * q_dim_ * 4, // attn
            r4 * inter * 4, // gate
            rows * inter * 4, // up
            rows * hidden_ * 4, // t (o/down GEMM out; also last-row scratch)
            VulkanLM.gemv_batch_max * cfg.n_heads * VulkanLM.nsplit * (hd + 2) * 4, // attn_scratch (a row per verify query)
            rows * VulkanLM.rms_chunks * 4, // rms_partials
            rows * 4, // rms_inv
            VulkanLM.gemv_batch_max * nvocab * 4, // logits (verify writes a row per position)
            4096 * 4, // argmax_v (>= opArgmax lane count)
            4096 * 4, // argmax_i
            4, // argmax_out (1 id)
            gpu.topk_lanes * gpu.topk_m * 4, // topk_v
            gpu.topk_lanes * gpu.topk_m * 4, // topk_i
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            created += 1;
        }
        // Dequant GEMV partials: largest output row count is the untied vocab
        // head; projections (qDim / intermediate) are smaller.
        self.quant_partials = try ctx.tensorCreate(@max(nvocab, @max(inter, q_dim_)) * VulkanLM.gemv_nchunk * 4);
        errdefer ctx.tensorDestroy(&self.quant_partials);
        // GEMV k-split partials: one per member of an `independent` group
        // (q/k/v, gate/up, the 4 LM-head chunks). Sized for the largest
        // user, times 4 for the 4-input verify variant.
        var pcreated: usize = 0;
        errdefer for (self.gemv_partials[0..pcreated]) |*pb| ctx.tensorDestroy(pb);
        for (&self.gemv_partials) |*pb| {
            pb.* = try ctx.tensorCreate(4 * @max(chunk_rows, inter) * VulkanLM.gemv_nchunk * 4);
            pcreated += 1;
        }
        return self;
    }

    fn deinit(self: *LmBufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        ctx.tensorDestroy(&self.quant_partials);
        for (&self.gemv_partials) |*pb| ctx.tensorDestroy(pb);
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "qt", "kt", "s", "attn", "gate", "up", "t", "attn_scratch", "rms_partials", "rms_inv", "logits", "argmax_v", "argmax_i", "argmax_out", "topk_v", "topk_i" };
};

// Parity against the CPU encode (same weights, same fixture prompt); gated on
// the model + GPU marker. Compares the GPU-resident forward to the CPU forward
// element-wise.
test "gpu encode matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const krea2_text = @import("krea2_text.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(&tok, gpa, "a fluffy orange cat sitting on a windowsill", &ids);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var enc = try qwen3.TextEncoder.load(gpa, &st);
    defer enc.deinit();

    const want = try enc.encode(io, gpa, ids.items, null);
    defer gpa.free(want);

    // f32 path (default): near-bit-parity (reduction-order noise only).
    // f16 coop path: same rounding regime as the DiT's tensor-core GEMMs.
    inline for (.{ .{ false, 1e-3 }, .{ true, 1e-2 } }) |cfg| {
        const got = try encode(&enc, ctx, io, gpa, ids.items, cfg[0], null);
        defer gpa.free(got);
        try std.testing.expectEqual(want.len, got.len);
        var max_err: f32 = 0;
        var max_val: f32 = 0;
        for (want, got) |e, a| {
            max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
            max_val = @max(max_val, @abs(e));
        }
        std.debug.print("qwen gpu parity (f16={}): max_err={d:.5} max_val={d:.2}\n", .{ cfg[0], max_err, max_val });
        try std.testing.expect(max_err < cfg[1] * @max(1.0, max_val));
    }
}

// Speculative decoding on the Vulkan stepper must be byte-identical to
// vanilla greedy: the grouped 4-input GEMVs and the batched flash-decoding
// attention reproduce the decode path's summation orders bitwise. Gated on
// the model + GPU marker; kept tiny (each token is a full 36-layer forward).
test "vulkan spec decode matches vanilla greedy" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const tokenizer_mod = @import("tp_core").tokenizer;
    const chat = @import("../llm/chat.zig");
    const engine = @import("../llm/engine.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    var opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };

    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_spec: std.ArrayList(u32) = .empty;
    defer ids_spec.deinit(gpa);
    try ids_spec.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityFor(opts, ids_vanilla.items.len), ids_vanilla.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.spec_k = 2;
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityFor(opts, ids_spec.items.len), ids_spec.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_spec, opts, null);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_spec.items);
}

// Regression for the bf16-generation bug: the Vulkan dense matmul / decode-GEMV
// kernels used to handle only 1-byte (fp8) and 4-byte (f32) weights, so a bf16
// checkpoint's 2-byte weights were read as f32 and generation produced garbage.
// bf16 is now read natively (2-byte transpose `transpose_bf16` + a bf16 branch
// in gemv_partial/gemv_partial4, weight code `WCode.bf16`). The pre-existing
// spec test above uses the *fp8* encoder checkpoint and only compares
// Vulkan-to-Vulkan, so it never exercised the bf16 path — hence a bf16 model AND
// a comparison against the CPU reference here. Compares the prefill's next-token
// argmax (the crisp signal: garbage diverged at token 1; bf16-GEMV-vs-CPU
// reduction-order drift can diverge multi-token decode, so we don't chase full
// token-identity on this correctness-first path).
test "vulkan bf16 prefill argmax matches cpu" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const tokenizer_mod = @import("tp_core").tokenizer;
    const chat = @import("../llm/chat.zig");
    const engine = @import("../llm/engine.zig");
    const path = "models/text_encoders/qwen3_4b_instruct.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "The capital of France is", &ids);
    try chat.openAssistant(&tok, gpa, &ids);

    const opts: engine.Options = .{ .max_new_tokens = 1, .sampling = .{ .temperature = 0 } };

    const argmaxOf = struct {
        fn f(logits: []const f32) usize {
            var best: usize = 0;
            for (logits, 0..) |v, i| if (v > logits[best]) {
                best = i;
            };
            return best;
        }
    }.f;

    const cpu_logits = try gpa.alloc(f32, qwen3.vocab_size);
    defer gpa.free(cpu_logits);
    const vk_logits = try gpa.alloc(f32, qwen3.vocab_size);
    defer gpa.free(vk_logits);

    // CPU reference: native bf16 forward.
    {
        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
        defer lm.deinit();
        var model = try engine.CpuModel.init(gpa, &lm, try engine.capacityPlanFor(opts, ids.items.len));
        defer model.deinit();
        try model.step(io, ids.items, cpu_logits);
    }
    // Vulkan: native bf16 weights (the fix). Read as f32 before, this was garbage.
    {
        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
        defer lm.deinit();
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityFor(opts, ids.items.len), ids.items.len);
        defer model.deinit();
        try model.step(io, ids.items, vk_logits);
    }

    try std.testing.expectEqual(argmaxOf(cpu_logits), argmaxOf(vk_logits));
}
