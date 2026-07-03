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
