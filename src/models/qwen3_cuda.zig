//! Qwen3-VL-4B text encoder on the hand-PTX CUDA backend.
//!
//! The CUDA analogue of `qwen3_gpu` (Vulkan): the whole 35-layer transformer
//! runs device-resident in one batched submission — one upload of the embedded
//! tokens in, one download of the 12-tap conditioning stack out. GEMMs use the
//! fp8-e4m3 path (`opMatmulFp8`, decode + f16 tensor cores); RMSNorm / per-head
//! QK-norm reuse `qkNorm`; rotate-half RoPE, the SwiGLU gate, residual adds, and
//! the naive causal GQA attention are the CUDA eltwise kernels. Attention stays
//! naive-f32 (parity-first): the encoder sequence is the prompt length (tens to
//! low hundreds of tokens), so the O(seq²) kernel is a sub-second one-time cost.
//! The embedding gather (bf16→f32) and the rope table are CPU-side.
//!
//! fp8 weights stream through the Backend weight cache, so a small --vram-budget
//! degrades to weight streaming here exactly as it does for the DiT.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const cuda = @import("../gpu/cuda.zig");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

const hidden = qwen3.hidden; // 2560
const n_heads = qwen3.n_heads; // 32
const kv_heads = qwen3.n_kv_heads; // 8
const hd = qwen3.head_dim; // 128
const half = hd / 2; // 64
const q_dim = n_heads * hd; // 4096
const kv_dim = kv_heads * hd; // 1024
const intermediate = qwen3.intermediate; // 9728
const n_layers = qwen3.n_layers; // 36
const tap_count = qwen3.tap_count;
const eps = qwen3.rms_eps;
const attn_scale: f32 = 1.0 / @sqrt(@as(f32, hd));

/// Encode token ids to the Krea 2 conditioning stack, [seq][tap_count][hidden]
/// (same token-major layout the CPU `encode` returns). Caller frees the result.
pub fn encode(enc: *const qwen3.TextEncoder, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
    _ = io;
    const seq = ids.len;
    std.debug.assert(seq > 0);
    const seq_pad = std.mem.alignForward(usize, seq, 128);

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
    const sin_off = seq * half;

    var bufs = try Bufs.init(be, seq, seq_pad);
    defer bufs.deinit(be);
    var freqs_d = try be.tensorCreate(fp.len * 4);
    defer be.tensorDestroy(&freqs_d);
    try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));
    try be.tensorUpload(bufs.x, std.mem.sliceAsBytes(x));

    const x_d = bufs.x;
    const nd = bufs.normed;
    const q_d = bufs.q;
    const k_d = bufs.k;
    const v_d = bufs.v;
    const attn_d = bufs.attn;
    const g_d = bufs.gate;
    const u_d = bufs.up;
    const t_d = bufs.t;
    const out_d = bufs.out;

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    var tap_idx: usize = 0;
    for (0..n_layers) |l| {
        if (tap_idx < qwen3.tap_layers.len and qwen3.tap_layers[tap_idx] == l) {
            // Snapshot the hidden state entering layer l into the tap-major output.
            try be.tensorCopy(out_d, tap_idx * seq * hidden * 4, x_d, 0, seq * hidden * 4);
            tap_idx += 1;
        }
        if (l >= enc.layers.len) break;
        const layer = enc.layers[l];

        // --- Attention ---
        try be.qkNorm(x_d, nd, try nbuf(be, layer.input_norm), seq, hidden, eps);
        try be.opMatmulFp8(q_d, nd, seq, layer.q.bytes, layer.q.scale, q_dim, hidden);
        try be.opMatmulFp8(k_d, nd, seq, layer.k.bytes, layer.k.scale, kv_dim, hidden);
        try be.opMatmulFp8(v_d, nd, seq, layer.v.bytes, layer.v.scale, kv_dim, hidden);
        try be.qkNorm(q_d, q_d, try nbuf(be, layer.q_norm), seq * n_heads, hd, eps);
        try be.qkNorm(k_d, k_d, try nbuf(be, layer.k_norm), seq * kv_heads, hd, eps);
        try be.ropeHalf(q_d, freqs_d, seq, n_heads, half, sin_off, 0);
        try be.ropeHalf(k_d, freqs_d, seq, kv_heads, half, sin_off, 0);
        try be.attn(q_d, k_d, v_d, attn_d, seq, seq, n_heads, kv_heads, hd, attn_scale, true);
        try be.opMatmulFp8(t_d, attn_d, seq, layer.o.bytes, layer.o.scale, hidden, q_dim);
        try be.opAdd(x_d, t_d, seq * hidden);

        // --- MLP (SwiGLU) ---
        try be.qkNorm(x_d, nd, try nbuf(be, layer.post_norm), seq, hidden, eps);
        try be.opMatmulFp8(g_d, nd, seq, layer.gate.bytes, layer.gate.scale, intermediate, hidden);
        try be.opMatmulFp8(u_d, nd, seq, layer.up.bytes, layer.up.scale, intermediate, hidden);
        try be.siluMul(g_d, u_d, seq * intermediate);
        try be.opMatmulFp8(t_d, g_d, seq, layer.down.bytes, layer.down.scale, hidden, intermediate);
        try be.opAdd(x_d, t_d, seq * hidden);
    }
    std.debug.assert(tap_idx == tap_count);
    try be.endBatch();

    // Download tap-major [tap][seq][hidden]; transpose to token-major.
    const tap_major = try gpa.alloc(f32, tap_count * seq * hidden);
    defer gpa.free(tap_major);
    try be.tensorDownload(out_d, std.mem.sliceAsBytes(tap_major));

    const out = try gpa.alloc(f32, seq * tap_count * hidden);
    errdefer gpa.free(out);
    for (0..tap_count) |tp| {
        for (0..seq) |t| {
            @memcpy(out[(t * tap_count + tp) * hidden ..][0..hidden], tap_major[(tp * seq + t) * hidden ..][0..hidden]);
        }
    }
    return out;
}

/// Wrap a CPU f32 norm-weight slice as a (pointer-cached) small device buffer.
fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// KV-cached causal LM on the CUDA backend (tp-llm --backend zig-cuda /
/// cuda): the full 36-layer stack runs device-resident per step — prefill is
/// one batched submission over the whole prompt (opMatmulFp8 tensor-core
/// GEMMs + the square attn kernel), decode is one over a single token (fused
/// gemv_fp8 dequant-GEMVs + warp flash-decoding attention). K/V live on
/// device, [capacity][kv_dim] f32 per layer; the final norm + tied bf16 LM
/// head (gemv_bf16) run on device too, so only the sampled token's embedding
/// goes up and the vocab logits come down each step. Engine-compatible
/// stepper (see llm/engine.zig generate()).
pub const CudaLM = struct {
    lm: *const qwen3.CausalLM,
    be: *Backend,
    gpa: std.mem.Allocator,
    capacity: usize,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    /// Activation-buffer row budget: the prompt for prefill, 1 afterwards.
    max_rows: usize,
    sin_off: usize,
    k_cache: [n_layers]Buf,
    v_cache: [n_layers]Buf,
    freqs_d: Buf,
    bufs: LmBufs,

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const qwen3.CausalLM, capacity: usize, first_seq: usize) !CudaLM {
        var self: CudaLM = undefined;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.capacity = capacity;
        self.len = 0;
        self.max_rows = @max(first_seq, 1);
        self.sin_off = capacity * half;

        var freqs = try ops.rope.rotateHalfFreqs(gpa, capacity, hd, qwen3.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * capacity * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. capacity * half], freqs.cos);
        @memcpy(fp[capacity * half ..], freqs.sin);
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&self.freqs_d);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        var created: usize = 0;
        errdefer for (self.k_cache[0..created]) |*b| be.tensorDestroy(b);
        for (&self.k_cache) |*b| {
            b.* = try be.tensorCreate(capacity * kv_dim * 4);
            created += 1;
        }
        var vcreated: usize = 0;
        errdefer for (self.v_cache[0..vcreated]) |*b| be.tensorDestroy(b);
        for (&self.v_cache) |*b| {
            b.* = try be.tensorCreate(capacity * kv_dim * 4);
            vcreated += 1;
        }

        self.bufs = try LmBufs.init(be, self.max_rows);
        return self;
    }

    pub fn deinit(self: *CudaLM) void {
        for (&self.k_cache) |*b| self.be.tensorDestroy(b);
        for (&self.v_cache) |*b| self.be.tensorDestroy(b);
        self.be.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.be);
        self.* = undefined;
    }

    pub fn remaining(self: *const CudaLM) usize {
        return self.capacity - self.len;
    }

    /// Forward `ids` at positions [len, len+ids.len) (prefill on the first
    /// call, single-token decode after), then write last-position vocab
    /// logits. CPU work per step: embedding gather up, hidden-state row down,
    /// final norm + LM head.
    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        const gpa = self.gpa;
        const be = self.be;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= self.max_rows);
        const pos0 = self.len;

        // CPU: embedding gather (bf16 -> f32), upload.
        const x = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(x);
        for (ids, 0..) |id, t| {
            if (id >= qwen3.vocab_size) return error.TokenIdOutOfRange;
            const row = self.lm.embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
            try safetensors.convertToF32(.bf16, row, x[t * hidden ..][0..hidden]);
        }
        try be.tensorUpload(offsetBufSized(self.bufs.x, 0, seq * hidden * 4), std.mem.sliceAsBytes(x));

        const b = &self.bufs;
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (self.lm.layers, 0..) |layer, l| {
            // --- Attention ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.input_norm), seq, hidden, eps);
            if (seq == 1) {
                try be.opGemvFp8(b.q, b.normed, layer.q.bytes, layer.q.scale, q_dim, hidden);
                try be.opGemvFp8(b.k, b.normed, layer.k.bytes, layer.k.scale, kv_dim, hidden);
                try be.opGemvFp8(b.v, b.normed, layer.v.bytes, layer.v.scale, kv_dim, hidden);
            } else {
                try be.opMatmulFp8(b.q, b.normed, seq, layer.q.bytes, layer.q.scale, q_dim, hidden);
                try be.opMatmulFp8(b.k, b.normed, seq, layer.k.bytes, layer.k.scale, kv_dim, hidden);
                try be.opMatmulFp8(b.v, b.normed, seq, layer.v.bytes, layer.v.scale, kv_dim, hidden);
            }
            try be.qkNorm(b.q, b.q, try nbuf(be, layer.q_norm), seq * n_heads, hd, eps);
            try be.qkNorm(b.k, b.k, try nbuf(be, layer.k_norm), seq * kv_heads, hd, eps);
            try be.ropeHalf(b.q, self.freqs_d, seq, n_heads, half, self.sin_off, pos0);
            try be.ropeHalf(b.k, self.freqs_d, seq, kv_heads, half, self.sin_off, pos0);
            try be.tensorCopy(self.k_cache[l], pos0 * kv_dim * 4, b.k, 0, seq * kv_dim * 4);
            try be.tensorCopy(self.v_cache[l], pos0 * kv_dim * 4, b.v, 0, seq * kv_dim * 4);
            if (seq == 1) {
                try be.opAttnDecode(b.q, self.k_cache[l], self.v_cache[l], b.attn, b.attn_scratch, pos0 + 1, n_heads, kv_heads, hd, nsplit, attn_scale);
                try be.opGemvFp8(b.t, b.attn, layer.o.bytes, layer.o.scale, hidden, q_dim);
            } else {
                try be.attn(b.q, self.k_cache[l], self.v_cache[l], b.attn, seq, pos0 + seq, n_heads, kv_heads, hd, attn_scale, true);
                try be.opMatmulFp8(b.t, b.attn, seq, layer.o.bytes, layer.o.scale, hidden, q_dim);
            }
            try be.opAdd(b.x, b.t, seq * hidden);

            // --- MLP (SwiGLU) ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.post_norm), seq, hidden, eps);
            if (seq == 1) {
                try be.opGemvFp8(b.gate, b.normed, layer.gate.bytes, layer.gate.scale, intermediate, hidden);
                try be.opGemvFp8(b.up, b.normed, layer.up.bytes, layer.up.scale, intermediate, hidden);
                try be.siluMul(b.gate, b.up, intermediate);
                try be.opGemvFp8(b.t, b.gate, layer.down.bytes, layer.down.scale, hidden, intermediate);
            } else {
                try be.opMatmulFp8(b.gate, b.normed, seq, layer.gate.bytes, layer.gate.scale, intermediate, hidden);
                try be.opMatmulFp8(b.up, b.normed, seq, layer.up.bytes, layer.up.scale, intermediate, hidden);
                try be.siluMul(b.gate, b.up, seq * intermediate);
                try be.opMatmulFp8(b.t, b.gate, seq, layer.down.bytes, layer.down.scale, hidden, intermediate);
            }
            try be.opAdd(b.x, b.t, seq * hidden);
        }

        // Final norm on the last position + tied bf16 LM head, on device.
        try be.qkNorm(offsetBufSized(b.x, (seq - 1) * hidden * 4, hidden * 4), b.t, try nbuf(be, self.lm.final_norm), 1, hidden, eps);
        try be.opGemvBf16(b.logits, b.t, self.lm.embed_bytes, 1.0, qwen3.vocab_size, hidden);
        try be.endBatch();
        self.len += seq;

        try be.tensorDownload(b.logits, std.mem.sliceAsBytes(logits[0..qwen3.vocab_size]));
        _ = io;
    }

    /// KV chunks per head in the decode attention split pass (one warp each:
    /// 32 heads x 32 splits x 32 lanes = 32k threads).
    const nsplit = 32;
};

/// offsetBuf carrying an explicit size (tensorUpload/Download use db.size).
fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

const LmBufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    attn_scratch: Buf,
    logits: Buf,

    fn init(be: *Backend, rows: usize) !LmBufs {
        const rp = std.mem.alignForward(usize, rows, 128); // GEMM outputs are 128-row padded
        var self: LmBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        const sizes = [fields.len]usize{
            rows * hidden * 4, // x
            rows * hidden * 4, // normed
            rp * q_dim * 4, // q
            rp * kv_dim * 4, // k
            rp * kv_dim * 4, // v
            rows * q_dim * 4, // attn
            rp * intermediate * 4, // gate
            rp * intermediate * 4, // up
            rp * hidden * 4, // t
            n_heads * CudaLM.nsplit * (hd + 4) * 4, // attn_scratch
            qwen3.vocab_size * 4, // logits
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *LmBufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "attn_scratch", "logits" };
};

const Bufs = struct {
    x: Buf, // residual stream [seq][hidden]
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    out: Buf, // tap-major [tap][seq][hidden]

    fn init(be: *Backend, seq: usize, seq_pad: usize) !Bufs {
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        // GEMM outputs (q/k/v/gate/up/t) are 128-row padded (pad rows are zero);
        // x/normed/attn are indexed by real seq.
        const sizes = [fields.len]usize{
            seq * hidden * 4, // x
            seq * hidden * 4, // normed
            seq_pad * q_dim * 4, // q
            seq_pad * kv_dim * 4, // k
            seq_pad * kv_dim * 4, // v
            seq * q_dim * 4, // attn
            seq_pad * intermediate * 4, // gate
            seq_pad * intermediate * 4, // up
            seq_pad * hidden * 4, // t
            tap_count * seq * hidden * 4, // out
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "out" };
};
