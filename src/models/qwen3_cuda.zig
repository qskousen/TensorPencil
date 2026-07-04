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
        try be.ropeHalf(q_d, freqs_d, seq, n_heads, half, sin_off);
        try be.ropeHalf(k_d, freqs_d, seq, kv_heads, half, sin_off);
        try be.attn(q_d, k_d, v_d, attn_d, seq, n_heads, kv_heads, hd, attn_scale, true);
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
