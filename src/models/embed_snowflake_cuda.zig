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

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
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
        l2normalize(out);
    }

    /// Batched CUDA encode: `ids_list[i]` → `outs[i]`. Mirrors the CPU
    /// `Model.embedBatch` (ragged packing: GEMMs / LayerNorm / GeGLU over
    /// `total = sum(seq_i)`; RoPE + attention loop per item via `dbOffset` views
    /// into the packed q/k/v/attn buffers). One upload → forward → download.
    pub fn embedBatch(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        _ = io;
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const w = cfg.hidden;
        const hd = cfg.head_dim;
        const half = hd / 2;
        const heads = cfg.n_heads;
        const inter = cfg.intermediate;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const eps = cfg.ln_eps;
        const seg = w * w * @sizeOf(f32);
        const useg = inter * w * @sizeOf(f32);
        const b = ids_list.len;
        std.debug.assert(outs.len == b and b > 0);

        const row_off = try gpa.alloc(usize, b + 1);
        defer gpa.free(row_off);
        row_off[0] = 0;
        var max_seq: usize = 0;
        for (ids_list, 0..) |ids, i| {
            row_off[i + 1] = row_off[i] + ids.len;
            max_seq = @max(max_seq, ids.len);
        }
        const total = row_off[b];
        const sin_off = max_seq * half;

        const zero_bias = try gpa.alloc(f32, inter);
        defer gpa.free(zero_bias);
        @memset(zero_bias, 0);

        const x_host = try gpa.alloc(f32, total * w);
        defer gpa.free(x_host);
        for (ids_list, 0..) |ids, i| try qwen3.embedTokens(cpu.word_emb, ids, x_host[row_off[i] * w ..][0 .. ids.len * w]);
        for (0..total) |t| {
            for (x_host[t * w ..][0..w], cpu.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x_host, x_host, cpu.emb_ln_w, cpu.emb_ln_b, eps);

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var freqs = try uploadFreqs(be, gpa, max_seq, hd, cfg.rope_theta);
        defer be.tensorDestroy(&freqs);

        var bufs: [8]Buf = @splat(.{});
        defer for (&bufs) |*bf| be.tensorDestroy(bf);
        const sizes = [bufs.len]usize{ total * w, total * w, total * w, total * w, total * w, total * inter, total * inter, total * w };
        for (&bufs, sizes) |*bf, s| bf.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const q_d = bufs[1];
        const k_d = bufs[2];
        const v_d = bufs[3];
        const attn_d = bufs[4];
        const up_d = bufs[5];
        const gate_d = bufs[6];
        const t_d = bufs[7];

        // Per-query-row item bounds for the block-diagonal batched attention.
        var bounds_d = try be.tensorCreate(2 * total * 4);
        defer be.tensorDestroy(&bounds_d);
        {
            const bounds = try gpa.alloc(u32, 2 * total);
            defer gpa.free(bounds);
            for (0..b) |i| {
                for (row_off[i]..row_off[i + 1]) |r| {
                    bounds[r] = @intCast(row_off[i]);
                    bounds[total + r] = @intCast(row_off[i + 1]);
                }
            }
            try be.tensorUpload(bounds_d, std.mem.sliceAsBytes(bounds));
        }

        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (cpu.layers) |*l| {
            const qkv = l.qkv.bytes;
            const ug = l.up_gate.bytes;
            try gemm(be, q_d, x_d, total, qkv[0..seg], w, w, l.qkv_bias[0..w]);
            try gemm(be, k_d, x_d, total, qkv[seg .. 2 * seg], w, w, l.qkv_bias[w .. 2 * w]);
            try gemm(be, v_d, x_d, total, qkv[2 * seg .. 3 * seg], w, w, l.qkv_bias[2 * w .. 3 * w]);
            for (0..b) |i| {
                const L = ids_list[i].len;
                const eoff = row_off[i] * w;
                try be.ropeHalf(q_d.viewF32(eoff), freqs, L, heads, half, sin_off, 0);
                try be.ropeHalf(k_d.viewF32(eoff), freqs, L, heads, half, sin_off, 0);
            }
            try be.opAttnBatched(q_d, k_d, v_d, attn_d, bounds_d, total, heads, heads, hd, scale);
            try gemm(be, t_d, attn_d, total, l.o.bytes, w, w, l.o_bias);
            try be.opAdd(x_d, t_d, total * w);
            try be.opLayerNorm(x_d, x_d, l.attn_ln_w, l.attn_ln_b, total, w, eps);

            try gemm(be, up_d, x_d, total, ug[0..useg], inter, w, zero_bias);
            try gemm(be, gate_d, x_d, total, ug[useg .. 2 * useg], inter, w, zero_bias);
            try be.geluMul(gate_d, up_d, total * inter);
            try gemm(be, t_d, gate_d, total, l.down.bytes, w, inter, l.down_bias);
            try be.opAdd(x_d, t_d, total * w);
            try be.opLayerNorm(x_d, x_d, l.mlp_ln_w, l.mlp_ln_b, total, w, eps);
        }
        try be.endBatch();

        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        for (0..b) |i| {
            std.debug.assert(outs[i].len == es.embed_dim);
            @memcpy(outs[i], x_host[row_off[i] * w ..][0..w]);
            l2normalize(outs[i]);
        }
    }
};

fn l2normalize(v: []f32) void {
    var ss: f32 = 0;
    for (v) |x| ss += x * x;
    const norm = @sqrt(ss);
    if (norm > 0) {
        const inv = 1.0 / norm;
        for (v) |*x| x.* *= inv;
    }
}

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

test "snowflake CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const item0 = [_]u32{ 0, 33600, 31, 2 };
    const item1 = [_]u32{ 0, 8999, 33600, 31, 2 };
    const item2 = [_]u32{ 0, 31, 2 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][es.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(be, io, gpa, ids, &single[i]);
    var batched: [3][es.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(be, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3); // f16 tensor-core GEMM regime
    }
}
