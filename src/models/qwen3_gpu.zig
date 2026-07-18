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
const gpu = @import("../gpu/context.zig");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");
const spec = @import("../llm/spec.zig");
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

/// Encode token ids to the Krea 2 conditioning stack, [seq][tap_count][hidden]
/// (same layout the CPU `encode` returns). Caller frees the result.
pub fn encode(enc: *const qwen3.TextEncoder, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, use_f16: bool) ![]f32 {
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
fn wcode(dt: @import("../dtype.zig").DType) gpu.WCode {
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
/// Mirrors qwen3_cuda.CudaLM: device-resident 36-layer stack, one batched
/// submission per step. Prefill (seq > 1, empty cache) reuses the encoder's
/// square attention (gather_kmajor + attn_scores + attn_out); decode uses the
/// flash-decoding attn_dsplit/attn_dmerge pair against the per-layer KV
/// cache. GEMMs are the f32-accumulate fp8 kernel with m = seq (m = 1 is a
/// GEMV: each weight byte read once). Hidden-dim norms take the 3-pass
/// parallel rmsnorm (rms_partial/rms_combine/rms_apply_w — one thread per
/// row would serialize rows = 1). The tied LM head is the bf16 embedding
/// converted to f32 once and split into 4 vocab chunks so each stays under
/// the kernels' 1 GiB type-level buffer bound.
pub const VulkanLM = struct {
    lm: *const qwen3.CausalLM,
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    capacity: usize,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    max_rows: usize,
    sin_off: u32,
    /// bf16 embedding converted to f32: LM-head weight source (weightBuffer
    /// caches by host pointer, so this must stay alive) + CPU-side gather.
    embed_f32: []f32,
    k_cache: [n_layers]Buf,
    v_cache: [n_layers]Buf,
    freqs_d: Buf,
    bufs: LmBufs,

    /// LM-head vocab split (151936 / 4 = 37984 rows per chunk; y descriptor
    /// offsets stay 16-byte aligned).
    const vocab_chunks = 4;
    const chunk_rows = qwen3.vocab_size / vocab_chunks;
    /// KV chunks per head in the decode attention split pass.
    const nsplit = 128;
    /// Largest batch that runs the small-batch path (4-input grouped GEMVs +
    /// batched flash-decoding): every speculative verify batch, and the
    /// chunk size for follow-up (pos0 > 0) prefills — which previously went
    /// token-by-token because the square attention kernel is pos0=0-only.
    const gemv_batch_max = spec.max_draft + 1;
    /// Interleaved chunks per row in the 3-pass rmsnorm.
    const rms_chunks = 64;
    /// Interleaved k chunks per output column in the decode GEMV.
    const gemv_nchunk = 32;

    /// Device bytes the KV cache reserves up front for `capacity` tokens (k +
    /// v across all layers). Vulkan has no growable buffers, so this whole
    /// window is allocated in init() — callers sizing a default weight-pin
    /// budget must leave this much VRAM unpinned for it.
    pub fn kvWindowBytes(capacity: usize) usize {
        return 2 * n_layers * capacity * kv_dim * 4;
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen3.CausalLM, capacity: usize, first_seq: usize) !VulkanLM {
        // This stepper is still hardwired to the 4B dims (module constants);
        // the 0.6B draft model runs on the CPU/CUDA steppers only for now.
        if (lm.cfg.n_layers != n_layers or lm.cfg.hidden != hidden) return error.UnsupportedModelConfig;
        // bf16 embedding doubling as the tied LM head is baked into the
        // kernels; GGUF-quantized models are CPU-only for now.
        if (lm.embed.dtype != .bf16 or lm.head.bytes.ptr != lm.embed.bytes.ptr)
            return error.UnsupportedModelConfig;
        var self: VulkanLM = undefined;
        self.lm = lm;
        self.ctx = ctx;
        self.gpa = gpa;
        self.capacity = capacity;
        self.len = 0;
        // Activation buffers always cover a speculative verify batch.
        self.max_rows = @max(@max(first_seq, 1), gemv_batch_max);
        self.sin_off = @intCast(capacity * half);

        self.embed_f32 = try gpa.alloc(f32, qwen3.vocab_size * hidden);
        errdefer gpa.free(self.embed_f32);
        try safetensors.convertToF32(.bf16, lm.embed.bytes, self.embed_f32);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, capacity, hd, qwen3.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * capacity * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. capacity * half], freqs.cos);
        @memcpy(fp[capacity * half ..], freqs.sin);
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&self.freqs_d);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        var created: usize = 0;
        errdefer for (self.k_cache[0..created]) |*bf| ctx.tensorDestroy(bf);
        for (&self.k_cache) |*bf| {
            bf.* = try ctx.tensorCreate(capacity * kv_dim * 4);
            created += 1;
        }
        var vcreated: usize = 0;
        errdefer for (self.v_cache[0..vcreated]) |*bf| ctx.tensorDestroy(bf);
        for (&self.v_cache) |*bf| {
            bf.* = try ctx.tensorCreate(capacity * kv_dim * 4);
            vcreated += 1;
        }

        self.bufs = try LmBufs.init(ctx, self.max_rows);
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        for (&self.k_cache) |*bf| self.ctx.tensorDestroy(bf);
        for (&self.v_cache) |*bf| self.ctx.tensorDestroy(bf);
        self.ctx.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.ctx);
        self.gpa.free(self.embed_f32);
        self.* = undefined;
    }

    pub fn cached(self: *const VulkanLM) usize {
        return self.len;
    }

    pub fn vocab(self: *const VulkanLM) usize {
        _ = self;
        return qwen3.vocab_size;
    }

    pub fn remaining(self: *const VulkanLM) usize {
        return self.capacity - self.len;
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
            const n = if (self.len == 0)
                @min(self.max_rows, ids.len - off)
            else
                @min(gemv_batch_max, ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], logits);
            off += n;
        }
    }

    /// step, but with vocab logits for every new token ([ids.len, vocab]
    /// row-major) — the speculative-decode verify forward. The batch is
    /// engine-capped at spec.max_draft + 1.
    pub fn stepAll(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        const ctx = self.ctx;
        const seq = ids.len;
        std.debug.assert(logits.len == seq * qwen3.vocab_size);
        std.debug.assert(seq > 0 and seq <= gemv_batch_max);
        const b = &self.bufs;

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
                const w = self.embed_f32[ci * chunk_rows * hidden ..][0 .. chunk_rows * hidden];
                try ctx.opGemvPartial4(b.normed, g * 4 * hidden, b.gemv_partials[ci], std.mem.sliceAsBytes(w), .f32, chunk_rows, hidden, gemv_nchunk);
            }
            ctx.independent(vocab_chunks);
            for (0..vocab_chunks) |ci| {
                try ctx.opGemvCombine4(b.logits, g * 4 * qwen3.vocab_size + ci * chunk_rows, qwen3.vocab_size, b.gemv_partials[ci], chunk_rows, 1.0, gemv_nchunk, n);
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

        try self.layersForward(ids);
        errdefer if (ctx.batching) ctx.abortBatch();

        // Final norm on the last position + LM head (4 vocab chunks).
        try ctx.opElt(.copy, b.x, b.t, null, null, .{
            .u0 = hidden,
            .u2 = 0,
            .u3 = @intCast((seq - 1) * hidden),
        }, hidden, 1, 1);
        try self.normWide(b.t, b.normed, try nbuf(ctx, self.lm.final_norm), 1);
        ctx.independent(vocab_chunks);
        for (0..vocab_chunks) |ci| {
            const w = self.embed_f32[ci * chunk_rows * hidden ..][0 .. chunk_rows * hidden];
            try ctx.opGemvPartial(b.normed, b.gemv_partials[ci], std.mem.sliceAsBytes(w), .f32, chunk_rows, hidden, gemv_nchunk);
        }
        ctx.independent(vocab_chunks);
        for (0..vocab_chunks) |ci| {
            try ctx.opGemvCombine(b.logits, ci * chunk_rows, b.gemv_partials[ci], chunk_rows, 1.0, gemv_nchunk);
        }
        try ctx.endBatch();
        self.len += seq;

        // `null` leaves the last position's logits resident (stepArgmax runs the
        // device argmax on them instead of paying the ~608 KB vocab download).
        if (logits) |l| try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(l[0..qwen3.vocab_size]));
    }

    /// Greedy decode without the vocab download: forward `ids`, then argmax the
    /// last position's logits on-device and return just that token id. Matches
    /// sample.argmax (temperature 0). The engine uses this for the greedy path.
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        var off: usize = 0;
        while (off < ids.len) {
            const n = if (self.len == 0)
                @min(self.max_rows, ids.len - off)
            else
                @min(gemv_batch_max, ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], null);
            off += n;
        }
        const ctx = self.ctx;
        const b = &self.bufs;
        try ctx.opArgmax(b.logits, qwen3.vocab_size, b.argmax_out, &b.argmax_v, &b.argmax_i);
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
        var off: usize = 0;
        while (off < ids.len) {
            const n = if (self.len == 0)
                @min(self.max_rows, ids.len - off)
            else
                @min(gemv_batch_max, ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], null);
            off += n;
        }
        const ctx = self.ctx;
        const b = &self.bufs;
        const count = try ctx.opTopK(b.logits, qwen3.vocab_size, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try ctx.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try ctx.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    /// The 36-layer stack over `ids` at positions [len, len+seq): embedding
    /// upload, then the whole transformer inside an open batch. The caller
    /// finishes the batch (LM head variants differ) — on success the batch
    /// is still open, with the final hidden states in bufs.x.
    fn layersForward(self: *VulkanLM, ids: []const u32) !void {
        const gpa = self.gpa;
        const ctx = self.ctx;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= self.max_rows);
        const pos0 = self.len;

        // CPU: embedding gather from the f32 copy, upload.
        const x = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(x);
        for (ids, 0..) |id, t| {
            if (id >= qwen3.vocab_size) return error.TokenIdOutOfRange;
            @memcpy(x[t * hidden ..][0..hidden], self.embed_f32[@as(usize, id) * hidden ..][0..hidden]);
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
        if (seq == 1) {
            // Group the q/k/v GEMV halves so no barrier drains the GPU between
            // independent dispatches.
            ctx.independent(3);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[0], layer.q.bytes, wcode(layer.q.dtype), q_dim, hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[1], layer.k.bytes, wcode(layer.k.dtype), kv_dim, hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[2], layer.v.bytes, wcode(layer.v.dtype), kv_dim, hidden, gemv_nchunk);
            ctx.independent(3);
            try ctx.opGemvCombine(b.q, 0, b.gemv_partials[0], q_dim, layer.q.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.k, 0, b.gemv_partials[1], kv_dim, layer.k.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.v, 0, b.gemv_partials[2], kv_dim, layer.v.scale, gemv_nchunk);
        } else {
            try self.gemm(b.q, b.normed, seq, layer.q, q_dim, hidden);
            try self.gemm(b.k, b.normed, seq, layer.k, kv_dim, hidden);
            try self.gemm(b.v, b.normed, seq, layer.v, kv_dim, hidden);
        }
    }

    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq == 1) ctx.independent(2);
        try rmsnorm(ctx, b.q, b.q, try nbuf(ctx, layer.q_norm), seq * n_heads, hd);
        try rmsnorm(ctx, b.k, b.k, try nbuf(ctx, layer.k_norm), seq * kv_heads, hd);
    }

    pub fn applyRope(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        _ = l; // qwen3: single rope table for all layers
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq == 1) ctx.independent(2);
        try ctx.opElt(.rope_half, b.q, null, self.freqs_d, null, .{
            .u0 = @intCast(seq * n_heads * half),
            .u1 = half,
            .u2 = self.sin_off,
            .u3 = n_heads,
            .u4 = @intCast(pos0),
        }, seq * n_heads * half, 1, 1);
        try ctx.opElt(.rope_half, b.k, null, self.freqs_d, null, .{
            .u0 = @intCast(seq * kv_heads * half),
            .u1 = half,
            .u2 = self.sin_off,
            .u3 = kv_heads,
            .u4 = @intCast(pos0),
        }, seq * kv_heads * half, 1, 1);
    }

    pub fn appendKV(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        // Append K/V to the cache (in-batch device copy).
        if (seq == 1) ctx.independent(2);
        try ctx.opElt(.copy, b.k, self.k_cache[l], null, null, .{
            .u0 = @intCast(seq * kv_dim),
            .u2 = @intCast(pos0 * kv_dim),
        }, seq * kv_dim, 1, 1);
        try ctx.opElt(.copy, b.v, self.v_cache[l], null, null, .{
            .u0 = @intCast(seq * kv_dim),
            .u2 = @intCast(pos0 * kv_dim),
        }, seq * kv_dim, 1, 1);
    }

    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq <= gemv_batch_max) {
            // Batched flash-decoding split/merge against the cached prefix:
            // query t sees pos0 + 1 + t keys (causal) — covers decode (seq == 1),
            // speculative verify, and follow-up prefill chunks at any pos0.
            try ctx.opElt(.attn_dsplit, b.q, self.k_cache[l], self.v_cache[l], b.attn_scratch, .{
                .u0 = @intCast(pos0 + 1),
                .u1 = n_heads,
                .u2 = kv_heads,
                .u3 = hd,
                .u4 = nsplit,
                .u5 = @intCast(seq),
                .f0 = attn_scale,
            }, seq * n_heads * nsplit, 1, 1);
            try ctx.opElt(.attn_dmerge, b.attn_scratch, null, null, b.attn, .{
                .u0 = @intCast(seq * n_heads),
                .u1 = hd,
                .u2 = nsplit,
            }, seq * n_heads * hd, 1, 1);
        } else {
            // Square causal attention (prefill starts from an empty cache).
            std.debug.assert(pos0 == 0);
            try ctx.opElt(.gather_kmajor, b.q, null, null, b.qt, .{
                .u0 = @intCast(seq * n_heads * hd),
                .u1 = n_heads,
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * n_heads * hd, 1, 1);
            try ctx.opElt(.gather_kmajor, b.k, null, null, b.kt, .{
                .u0 = @intCast(seq * kv_heads * hd),
                .u1 = kv_heads,
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * kv_heads * hd, 1, 1);
            const dc8 = std.math.divCeil(usize, seq, 8) catch unreachable;
            try ctx.opElt(.attn_scores, b.qt, b.kt, null, b.s, .{
                .u0 = @intCast(seq),
                .u1 = n_heads,
                .u2 = kv_heads,
                .u3 = hd,
                .u4 = 0,
                .f0 = attn_scale,
            }, dc8, dc8, n_heads);
            try ctx.opElt(.attn_out, b.s, null, b.v, b.attn, .{
                .u0 = @intCast(seq),
                .u1 = n_heads,
                .u2 = kv_heads,
                .u3 = hd,
                .u4 = 0,
                .u5 = @intCast(seq),
                .f0 = @bitCast(@as(u32, @intCast(seq * seq))),
                .f1 = @bitCast(@as(u32, 1)), // causal
            }, hd / 8, dc8, n_heads);
        }
    }

    pub fn projectO(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        try self.gemm(self.bufs.t, self.bufs.attn, seq, layer.o, hidden, q_dim);
    }

    pub fn addResidual(self: *VulkanLM, seq: usize) !void {
        try self.ctx.opElt(.add, self.bufs.x, self.bufs.t, null, null, .{ .u0 = @intCast(seq * hidden) }, seq * hidden, 1, 1);
    }

    pub fn normPreFfn(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.normWide(self.bufs.x, self.bufs.normed, try nbuf(self.ctx, layer.post_norm), seq);
    }

    pub fn projectGateUp(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq == 1) {
            ctx.independent(2);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[0], layer.gate.bytes, wcode(layer.gate.dtype), intermediate, hidden, gemv_nchunk);
            try ctx.opGemvPartial(b.normed, b.gemv_partials[1], layer.up.bytes, wcode(layer.up.dtype), intermediate, hidden, gemv_nchunk);
            ctx.independent(2);
            try ctx.opGemvCombine(b.gate, 0, b.gemv_partials[0], intermediate, layer.gate.scale, gemv_nchunk);
            try ctx.opGemvCombine(b.up, 0, b.gemv_partials[1], intermediate, layer.up.scale, gemv_nchunk);
        } else {
            try self.gemm(b.gate, b.normed, seq, layer.gate, intermediate, hidden);
            try self.gemm(b.up, b.normed, seq, layer.up, intermediate, hidden);
        }
    }

    pub fn activate(self: *VulkanLM, comptime act: transformer.Activation, seq: usize) !void {
        const which: gpu.Elt = switch (act) {
            .silu_mul => .silu_mul,
            .gelu_tanh_mul => .gelu_mul,
        };
        try self.ctx.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(seq * intermediate) }, seq * intermediate, 1, 1);
    }

    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.gemm(self.bufs.t, self.bufs.gate, seq, layer.down, hidden, intermediate);
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
        try ctx.opElt(.rms_partial, in, null, null, self.bufs.rms_partials, .{
            .u0 = @intCast(rows * rms_chunks),
            .u1 = hidden,
            .u2 = rms_chunks,
        }, rows * rms_chunks, 1, 1);
        try ctx.opElt(.rms_combine, self.bufs.rms_partials, null, null, self.bufs.rms_inv, .{
            .u0 = @intCast(rows),
            .u1 = hidden,
            .u2 = rms_chunks,
            .f0 = eps,
        }, rows, 1, 1);
        try ctx.opElt(.rms_apply_w, in, out, weight, self.bufs.rms_inv, .{
            .u0 = @intCast(rows * hidden),
            .u1 = hidden,
        }, rows * hidden, 1, 1);
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
    logits: Buf,
    // GPU-argmax scratch (opArgmax): per-lane max value + index, and the 1-id out.
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    // GPU top-k scratch (opTopK): per-lane top-M values + indices.
    topk_v: Buf,
    topk_i: Buf,

    fn init(ctx: *gpu.Context, rows: usize) !LmBufs {
        var self: LmBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        const r4 = std.mem.alignForward(usize, rows, 4); // grouped-GEMV inputs are read 4 rows at a time
        const sizes = [fields.len]usize{
            rows * hidden * 4, // x
            r4 * hidden * 4, // normed
            rows * q_dim * 4, // q
            rows * kv_dim * 4, // k
            rows * kv_dim * 4, // v
            rows * q_dim * 4, // qt (prefill k-major)
            rows * kv_dim * 4, // kt
            n_heads * rows * rows * 4, // s (prefill scores)
            r4 * q_dim * 4, // attn
            r4 * intermediate * 4, // gate
            rows * intermediate * 4, // up
            rows * hidden * 4, // t (o/down GEMM out; also last-row scratch)
            VulkanLM.gemv_batch_max * n_heads * VulkanLM.nsplit * (hd + 2) * 4, // attn_scratch (a row per verify query)
            rows * VulkanLM.rms_chunks * 4, // rms_partials
            rows * 4, // rms_inv
            VulkanLM.gemv_batch_max * qwen3.vocab_size * 4, // logits (verify writes a row per position)
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
        // GEMV k-split partials: one per member of an `independent` group
        // (q/k/v, gate/up, the 4 LM-head chunks). Sized for the largest
        // user, times 4 for the 4-input verify variant.
        var pcreated: usize = 0;
        errdefer for (self.gemv_partials[0..pcreated]) |*pb| ctx.tensorDestroy(pb);
        for (&self.gemv_partials) |*pb| {
            pb.* = try ctx.tensorCreate(4 * VulkanLM.chunk_rows * VulkanLM.gemv_nchunk * 4);
            pcreated += 1;
        }
        return self;
    }

    fn deinit(self: *LmBufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
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
    const tokenizer_mod = @import("../tokenizer.zig");
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

    const want = try enc.encode(io, gpa, ids.items);
    defer gpa.free(want);

    // f32 path (default): near-bit-parity (reduction-order noise only).
    // f16 coop path: same rounding regime as the DiT's tensor-core GEMMs.
    inline for (.{ .{ false, 1e-3 }, .{ true, 1e-2 } }) |cfg| {
        const got = try encode(&enc, ctx, io, gpa, ids.items, cfg[0]);
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
    const tokenizer_mod = @import("../tokenizer.zig");
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
    const tokenizer_mod = @import("../tokenizer.zig");
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
