//! EmbeddingGemma on the CUDA backend. Bidirectional Gemma-3 body device-side;
//! head (mean-pool + 2 Dense + L2) on host via `embed_gemma.Model.head`.
//!
//! CUDA RMSNorm (`qkNorm`) takes the norm weight as a DEVICE buffer, so all norm
//! vectors are uploaded to device buffers up front (before the batch). Gemma has
//! no GEMM biases, but opConvF16's bias_compact always reads the bias buffer, so
//! a zero bias (length ≥ co) is passed to every GEMM. Dual-RoPE via two device
//! freqs buffers; GQA + non-causal via the generic `attn`. tanh `geluMul`
//! approximates the gelu-tanh GeGLU (matches within the f16 GEMM parity floor).

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const eg = @import("embed_gemma.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w_bytes: []const u8, co: usize, k: usize, bias: []const f32) !void {
    try be.opConvF16(dst, 0, src, m, w_bytes, co, k, bias);
}

fn uploadFreqs(be: *Backend, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64) !Buf {
    const half = head_dim / 2;
    var freqs = try ops.rope.rotateHalfFreqs(gpa, rows, head_dim, theta);
    defer freqs.deinit(gpa);
    const packed_f = try gpa.alloc(f32, rows * half * 2);
    defer gpa.free(packed_f);
    @memcpy(packed_f[0 .. rows * half], freqs.cos);
    @memcpy(packed_f[rows * half ..], freqs.sin);
    const b = try be.tensorCreate(packed_f.len * 4);
    try be.tensorUpload(b, std.mem.sliceAsBytes(packed_f));
    return b;
}

pub const ModelCuda = struct {
    cpu: *const eg.Model,

    pub fn init(cpu: *const eg.Model) ModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
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
        const sin_off = seq * half;
        const eps = cfg.rms_eps;
        std.debug.assert(out.len == eg.embed_dim);
        std.debug.assert(seq <= cfg.sliding_window);

        // Host: token embed + sqrt(hidden) scale.
        const x_host = try gpa.alloc(f32, seq * w);
        defer gpa.free(x_host);
        try qwen3.embedTokens(cpu.embed_w, ids, x_host);
        const es_scale = cfg.embedScale();
        for (x_host) |*v| v.* *= es_scale;

        // Zero bias (length ≥ every co) — gemma GEMMs are bias-free.
        const zero_bias = try gpa.alloc(f32, @max(qd, inter));
        defer gpa.free(zero_bias);
        @memset(zero_bias, 0);

        be.weightScopeBegin();
        defer {
            be.weightScopeEnd();
            be.freeAttnScratch();
            be.freeConvScratch();
        }

        // Upload freqs + all norm weights to device buffers (before the batch).
        var owned: std.ArrayList(Buf) = .empty;
        defer {
            for (owned.items) |*b| be.tensorDestroy(b);
            owned.deinit(gpa);
        }
        const upNorm = struct {
            fn f(b: *Backend, g: std.mem.Allocator, list: *std.ArrayList(Buf), wv: []const f32) !Buf {
                const db = try b.tensorCreate(wv.len * 4);
                try b.tensorUpload(db, std.mem.sliceAsBytes(wv));
                try list.append(g, db);
                return db;
            }
        }.f;

        var freqs_g = try uploadFreqs(be, gpa, seq, hd, cfg.rope_theta);
        var freqs_l = try uploadFreqs(be, gpa, seq, hd, cfg.rope_theta_local);
        defer be.tensorDestroy(&freqs_g);
        defer be.tensorDestroy(&freqs_l);

        const LN = struct { input: Buf, qn: Buf, kn: Buf, pa: Buf, pf: Buf, pff: Buf };
        const lns = try gpa.alloc(LN, cpu.layers.len);
        defer gpa.free(lns);
        for (cpu.layers, 0..) |*l, i| {
            lns[i] = .{
                .input = try upNorm(be, gpa, &owned, l.input_norm),
                .qn = try upNorm(be, gpa, &owned, l.q_norm),
                .kn = try upNorm(be, gpa, &owned, l.k_norm),
                .pa = try upNorm(be, gpa, &owned, l.post_attn_norm),
                .pf = try upNorm(be, gpa, &owned, l.pre_ffn_norm),
                .pff = try upNorm(be, gpa, &owned, l.post_ffn_norm),
            };
        }
        const final_b = try upNorm(be, gpa, &owned, cpu.final_norm);

        var bufs: [9]Buf = @splat(.{});
        defer for (&bufs) |*b| be.tensorDestroy(b);
        const sizes = [bufs.len]usize{ seq * w, seq * w, seq * qd, seq * kvd, seq * kvd, seq * qd, seq * inter, seq * inter, seq * w };
        for (&bufs, sizes) |*b, s| b.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const attn_d = bufs[5];
        const gate_d = bufs[6];
        const up_d = bufs[7];
        const t_d = bufs[8];

        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (cpu.layers, 0..) |*l, li| {
            const freqs = if (cfg.isGlobal(li)) freqs_g else freqs_l;
            const ln = lns[li];
            // Attention (input RMSNorm → QKV → QK-norm → RoPE → GQA attn).
            try be.qkNorm(x_d, normed_d, ln.input, seq, w, eps);
            try gemm(be, q_d, normed_d, seq, l.q.bytes, qd, w, zero_bias);
            try gemm(be, k_d, normed_d, seq, l.k.bytes, kvd, w, zero_bias);
            try gemm(be, v_d, normed_d, seq, l.v.bytes, kvd, w, zero_bias);
            try be.qkNorm(q_d, q_d, ln.qn, seq * heads, hd, eps);
            try be.qkNorm(k_d, k_d, ln.kn, seq * kvh, hd, eps);
            try be.ropeHalf(q_d, freqs, seq, heads, half, sin_off, 0);
            try be.ropeHalf(k_d, freqs, seq, kvh, half, sin_off, 0);
            try be.attn(q_d, k_d, v_d, attn_d, seq, seq, heads, kvh, hd, scale, false);
            try gemm(be, t_d, attn_d, seq, l.o.bytes, w, qd, zero_bias);
            try be.qkNorm(t_d, t_d, ln.pa, seq, w, eps); // post-attn (sandwich)
            try be.opAdd(x_d, t_d, seq * w);

            // MLP (pre-FFN norm → GeGLU → down → post-FFN norm).
            try be.qkNorm(x_d, normed_d, ln.pf, seq, w, eps);
            try gemm(be, gate_d, normed_d, seq, l.gate.bytes, inter, w, zero_bias);
            try gemm(be, up_d, normed_d, seq, l.up.bytes, inter, w, zero_bias);
            try be.geluMul(gate_d, up_d, seq * inter);
            try gemm(be, t_d, gate_d, seq, l.down.bytes, w, inter, zero_bias);
            try be.qkNorm(t_d, t_d, ln.pff, seq, w, eps);
            try be.opAdd(x_d, t_d, seq * w);
        }
        try be.qkNorm(x_d, x_d, final_b, seq, w, eps);
        try be.endBatch();

        const lhs = try gpa.alloc(f32, seq * w);
        defer gpa.free(lhs);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(lhs));
        try cpu.head(io, gpa, lhs, out);
    }
};

// --- tests -----------------------------------------------------------------

test "embeddinggemma CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const ids = [_]u32{ 2, 23391, 1902, 1 };
    const c = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, &ids, c);
    try dev.embed(be, io, gpa, &ids, g);

    var dot: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    var num: f32 = 0;
    var den: f32 = 0;
    for (c, g) |a, b| {
        dot += a * b;
        na += a * a;
        nb += b * b;
        num += (a - b) * (a - b);
        den += a * a;
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    const rmse = @sqrt(num / den);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ cos, rmse });
    try std.testing.expect(cos > 0.999 and rmse < 0.05);
}
