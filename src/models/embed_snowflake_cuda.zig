//! Snowflake Arctic Embed (GTE) on the CUDA backend. Post-LayerNorm body
//! device-side (opConvF16 GEMMs, opLayerNorm, rotate-half RoPE via a device
//! freqs buffer, generic non-causal `attn`, tanh `geluMul` GeGLU, opAdd); CLS
//! pool + L2 on host. f16 tensor-core GEMM regime → relative parity vs CPU.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const es = @import("embed_snowflake.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w_bytes: []const u8, co: usize, k: usize, bias: []const f32) !void {
    try be.opConvF16(dst, 0, src, m, w_bytes, co, k, bias);
}

/// Upload a [cos | sin] rotate-half freqs buffer for `rows` positions.
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
    cpu: *const es.Model,

    pub fn init(cpu: *const es.Model) ModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        _ = io;
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const seq = ids.len;
        const w = cfg.hidden;
        const hd = cfg.head_dim;
        const half = hd / 2;
        const heads = cfg.n_heads;
        const inter = cfg.intermediate;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const sin_off = seq * half;
        const eps = cfg.ln_eps;
        const seg = w * w * @sizeOf(f32);
        const useg = inter * w * @sizeOf(f32);
        std.debug.assert(out.len == es.embed_dim);

        // opConvF16's bias_compact reads the bias buffer unconditionally, so a
        // no-bias GEMM (up_gate) must still pass a zero bias of length `co`.
        const zero_bias = try gpa.alloc(f32, inter);
        defer gpa.free(zero_bias);
        @memset(zero_bias, 0);

        const x_host = try gpa.alloc(f32, seq * w);
        defer gpa.free(x_host);
        try qwen3.embedTokens(cpu.word_emb, ids, x_host);
        for (0..seq) |t| {
            for (x_host[t * w ..][0..w], cpu.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x_host, x_host, cpu.emb_ln_w, cpu.emb_ln_b, eps);

        be.weightScopeBegin();
        defer {
            be.weightScopeEnd();
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var freqs = try uploadFreqs(be, gpa, seq, hd, cfg.rope_theta);
        defer be.tensorDestroy(&freqs);

        var bufs: [8]Buf = @splat(.{});
        defer for (&bufs) |*b| be.tensorDestroy(b);
        const sizes = [bufs.len]usize{ seq * w, seq * w, seq * w, seq * w, seq * w, seq * inter, seq * inter, seq * w };
        for (&bufs, sizes) |*b, s| b.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const q_d = bufs[1];
        const k_d = bufs[2];
        const v_d = bufs[3];
        const attn_d = bufs[4];
        const up_d = bufs[5];
        const gate_d = bufs[6];
        const t_d = bufs[7];

        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (cpu.layers) |*l| {
            const qkv = l.qkv.bytes;
            const ug = l.up_gate.bytes;
            try gemm(be, q_d, x_d, seq, qkv[0..seg], w, w, l.qkv_bias[0..w]);
            try gemm(be, k_d, x_d, seq, qkv[seg .. 2 * seg], w, w, l.qkv_bias[w .. 2 * w]);
            try gemm(be, v_d, x_d, seq, qkv[2 * seg .. 3 * seg], w, w, l.qkv_bias[2 * w .. 3 * w]);
            try be.ropeHalf(q_d, freqs, seq, heads, half, sin_off, 0);
            try be.ropeHalf(k_d, freqs, seq, heads, half, sin_off, 0);
            try be.attn(q_d, k_d, v_d, attn_d, seq, seq, heads, heads, hd, scale, false);
            try gemm(be, t_d, attn_d, seq, l.o.bytes, w, w, l.o_bias);
            try be.opAdd(x_d, t_d, seq * w);
            try be.opLayerNorm(x_d, x_d, l.attn_ln_w, l.attn_ln_b, seq, w, eps);

            try gemm(be, up_d, x_d, seq, ug[0..useg], inter, w, zero_bias);
            try gemm(be, gate_d, x_d, seq, ug[useg .. 2 * useg], inter, w, zero_bias);
            try be.geluMul(gate_d, up_d, seq * inter);
            try gemm(be, t_d, gate_d, seq, l.down.bytes, w, inter, l.down_bias);
            try be.opAdd(x_d, t_d, seq * w);
            try be.opLayerNorm(x_d, x_d, l.mlp_ln_w, l.mlp_ln_b, seq, w, eps);
        }
        try be.endBatch();

        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        @memcpy(out, x_host[0..w]);
        var ss: f32 = 0;
        for (out) |v| ss += v * v;
        const norm = @sqrt(ss);
        if (norm > 0) {
            const inv = 1.0 / norm;
            for (out) |*v| v.* *= inv;
        }
    }
};

// --- tests -----------------------------------------------------------------

test "snowflake CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const ids = [_]u32{ 0, 33600, 31, 8999, 2 };
    const c = try gpa.alloc(f32, es.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, es.embed_dim);
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
