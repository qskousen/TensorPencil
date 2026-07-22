//! EmbeddingGemma on the Vulkan backend: the bidirectional Gemma-3 body runs
//! device-side, the cheap head (mean-pool + 2 Dense + L2) on the host via
//! `embed_gemma.Model.head`. Reuses the CPU model's f32 weights directly
//! (mmap-stable → Vulkan's pointer-keyed weight cache, no dequant).
//!
//! Gemma-3 primitives, all existing Context ops: RMSNorm (`rmsnorm`, weights
//! carry the +1 already, folded at CPU load), per-head QK-norm (same `rmsnorm`
//! over head_dim rows), dual-RoPE (`rope_half` with a per-theta freqs buffer,
//! global every 6th layer), GQA non-causal `attn_full`, GeGLU (`gelu_mul`), the
//! 4-norm sandwich. Sliding window is run as full attention (correct for the
//! seq ≤ 512 the CPU model already asserts).

const std = @import("std");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const eg = @import("embed_gemma.zig");

const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// Build a [cos | sin] rotate-half freqs buffer for `rows` positions and upload.
fn uploadFreqs(ctx: *gpu.Context, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64) !Buf {
    const half = head_dim / 2;
    var freqs = try ops.rope.rotateHalfFreqs(gpa, rows, head_dim, theta);
    defer freqs.deinit(gpa);
    const packed_f = try gpa.alloc(f32, rows * half * 2);
    defer gpa.free(packed_f);
    @memcpy(packed_f[0 .. rows * half], freqs.cos);
    @memcpy(packed_f[rows * half ..], freqs.sin);
    const b = try ctx.tensorCreate(packed_f.len * 4);
    try ctx.tensorUpload(b, std.mem.sliceAsBytes(packed_f));
    return b;
}

pub const ModelGpu = struct {
    cpu: *const eg.Model,

    pub fn init(cpu: *const eg.Model) ModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const seq = ids.len;
        const w = cfg.hidden;
        const qd = cfg.qDim();
        const kvd = cfg.kvDim();
        const hd = cfg.head_dim;
        const half = hd / 2;
        const heads = cfg.n_heads;
        const kvh = cfg.n_kv_heads;
        const inter = cfg.intermediate;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const sin_off: u32 = @intCast(seq * half);
        std.debug.assert(out.len == eg.embed_dim);
        std.debug.assert(seq <= cfg.sliding_window); // full-attention window valid

        // Host: token embed + sqrt(hidden) scale.
        const x_host = try gpa.alloc(f32, seq * w);
        defer gpa.free(x_host);
        try qwen3.embedTokens(cpu.embed_w, ids, x_host);
        const es = cfg.embedScale();
        for (x_host) |*v| v.* *= es;

        var freqs_g = try uploadFreqs(ctx, gpa, seq, hd, cfg.rope_theta);
        var freqs_l = try uploadFreqs(ctx, gpa, seq, hd, cfg.rope_theta_local);
        defer ctx.tensorDestroy(&freqs_g);
        defer ctx.tensorDestroy(&freqs_l);

        var bufs: [9]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(seq * w * 4); // x
        bufs[1] = try ctx.tensorCreate(seq * w * 4); // normed
        bufs[2] = try ctx.tensorCreate(seq * qd * 4); // q
        bufs[3] = try ctx.tensorCreate(seq * kvd * 4); // k
        bufs[4] = try ctx.tensorCreate(seq * kvd * 4); // v
        bufs[5] = try ctx.tensorCreate(seq * qd * 4); // attn out
        bufs[6] = try ctx.tensorCreate(seq * inter * 4); // gate
        bufs[7] = try ctx.tensorCreate(seq * inter * 4); // up
        bufs[8] = try ctx.tensorCreate(seq * w * 4); // t (residual delta)
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const attn_d = bufs[5];
        const gate_d = bufs[6];
        const up_d = bufs[7];
        const t_d = bufs[8];

        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));

        const eps = cfg.rms_eps;
        for (cpu.layers, 0..) |*l, li| {
            const freqs = if (cfg.isGlobal(li)) freqs_g else freqs_l;
            // --- Attention (input RMSNorm → QKV → QK-norm → RoPE → GQA attn) ---
            try ctx.opElt(.rmsnorm, x_d, normed_d, try nbuf(ctx, l.input_norm), null, .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);
            try ctx.opMatmul(q_d, 0, normed_d, 0, seq, l.q.bytes, false, qd, w, 1.0, null);
            try ctx.opMatmul(k_d, 0, normed_d, 0, seq, l.k.bytes, false, kvd, w, 1.0, null);
            try ctx.opMatmul(v_d, 0, normed_d, 0, seq, l.v.bytes, false, kvd, w, 1.0, null);
            // Per-head QK-norm over head_dim (rows = seq*heads).
            try ctx.opElt(.rmsnorm, q_d, q_d, try nbuf(ctx, l.q_norm), null, .{ .u0 = @intCast(seq * heads), .u1 = @intCast(hd), .f0 = eps }, seq * heads, 1, 1);
            try ctx.opElt(.rmsnorm, k_d, k_d, try nbuf(ctx, l.k_norm), null, .{ .u0 = @intCast(seq * kvh), .u1 = @intCast(hd), .f0 = eps }, seq * kvh, 1, 1);
            try ctx.opElt(.rope_half, q_d, null, freqs, null, .{ .u0 = @intCast(seq * heads * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(heads) }, seq * heads * half, 1, 1);
            try ctx.opElt(.rope_half, k_d, null, freqs, null, .{ .u0 = @intCast(seq * kvh * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(kvh) }, seq * kvh * half, 1, 1);
            try ctx.opElt(.attn_full, q_d, k_d, v_d, attn_d, .{ .u0 = @intCast(seq), .u1 = @intCast(heads), .u2 = @intCast(kvh), .u3 = @intCast(hd), .f0 = scale }, seq * heads, 1, 1);
            try ctx.opMatmul(t_d, 0, attn_d, 0, seq, l.o.bytes, false, w, qd, 1.0, null);
            // Sandwich: post-attn norm on the attn output BEFORE the residual add.
            try ctx.opElt(.rmsnorm, t_d, t_d, try nbuf(ctx, l.post_attn_norm), null, .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * w) }, seq * w, 1, 1);

            // --- MLP (pre-FFN norm → GeGLU → down → post-FFN norm) ---
            try ctx.opElt(.rmsnorm, x_d, normed_d, try nbuf(ctx, l.pre_ffn_norm), null, .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);
            try ctx.opMatmul(gate_d, 0, normed_d, 0, seq, l.gate.bytes, false, inter, w, 1.0, null);
            try ctx.opMatmul(up_d, 0, normed_d, 0, seq, l.up.bytes, false, inter, w, 1.0, null);
            try ctx.opElt(.gelu_mul, gate_d, up_d, null, null, .{ .u0 = @intCast(seq * inter) }, seq * inter, 1, 1);
            try ctx.opMatmul(t_d, 0, gate_d, 0, seq, l.down.bytes, false, w, inter, 1.0, null);
            try ctx.opElt(.rmsnorm, t_d, t_d, try nbuf(ctx, l.post_ffn_norm), null, .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * w) }, seq * w, 1, 1);
        }
        try ctx.opElt(.rmsnorm, x_d, x_d, try nbuf(ctx, cpu.final_norm), null, .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);

        // Host: mean-pool + Dense head + L2 normalize.
        const lhs = try gpa.alloc(f32, seq * w);
        defer gpa.free(lhs);
        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(lhs));
        try cpu.head(io, gpa, lhs, out);
    }
};

// --- tests -----------------------------------------------------------------

test "embeddinggemma Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelGpu.init(&cpu);

    const ids = [_]u32{ 2, 23391, 1902, 1 }; // <bos> "hello world" <eos>
    const cpu_out = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(gpu_out);
    try cpu.embed(io, gpa, &ids, cpu_out);
    try dev.embed(ctx, io, gpa, &ids, gpu_out);

    var dot: f32 = 0;
    var maxad: f32 = 0;
    for (cpu_out, gpu_out) |c, g| {
        dot += c * g;
        maxad = @max(maxad, @abs(c - g));
    }
    errdefer std.debug.print("cosine {d}, max abs diff {d}\n", .{ dot, maxad });
    try std.testing.expect(dot >= 0.9999);
    try std.testing.expect(maxad < 1e-2);
}
