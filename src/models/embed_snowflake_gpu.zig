//! Snowflake Arctic Embed (GTE) on the Vulkan backend. The bidirectional
//! **post-LayerNorm** GTE body runs device-side; the CLS pool + L2 normalize
//! (trivial) on the host. Reuses the CPU model's f32 weights directly.
//!
//! GTE differs from the pre-norm encoders: attention/MLP read the residual
//! stream directly and the LayerNorm is applied AFTER the residual add. Packed
//! qkv and packed up_gate are sliced into per-projection weight views (f32, so
//! byte-slicing at row boundaries is exact). RoPE is single-theta rotate-half.
//!
//! GELU: GTE's FFN uses exact-erf gelu on the CPU; the Vulkan `gelu_mul` kernel
//! is tanh-approx. The two differ by ~1e-3/elt — within the parity floor here —
//! so we use `gelu_mul`; revisit with a dedicated erf kernel if it ever drifts.

const std = @import("std");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const es = @import("embed_snowflake.zig");

const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

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
    cpu: *const es.Model,

    pub fn init(cpu: *const es.Model) ModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const seq = ids.len;
        const w = cfg.hidden;
        const hd = cfg.head_dim;
        const half = hd / 2;
        const heads = cfg.n_heads;
        const inter = cfg.intermediate;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const sin_off: u32 = @intCast(seq * half);
        const eps = cfg.ln_eps;
        const seg = w * w * @sizeOf(f32);
        const useg = inter * w * @sizeOf(f32);
        std.debug.assert(out.len == es.embed_dim);
        _ = io;

        // Host: word emb + token_type[0], embeddings LayerNorm.
        const x_host = try gpa.alloc(f32, seq * w);
        defer gpa.free(x_host);
        try qwen3.embedTokens(cpu.word_emb, ids, x_host);
        for (0..seq) |t| {
            for (x_host[t * w ..][0..w], cpu.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x_host, x_host, cpu.emb_ln_w, cpu.emb_ln_b, eps);

        var freqs = try uploadFreqs(ctx, gpa, seq, hd, cfg.rope_theta);
        defer ctx.tensorDestroy(&freqs);

        var bufs: [8]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(seq * w * 4); // x
        bufs[1] = try ctx.tensorCreate(seq * w * 4); // q
        bufs[2] = try ctx.tensorCreate(seq * w * 4); // k
        bufs[3] = try ctx.tensorCreate(seq * w * 4); // v
        bufs[4] = try ctx.tensorCreate(seq * w * 4); // attn out
        bufs[5] = try ctx.tensorCreate(seq * inter * 4); // up
        bufs[6] = try ctx.tensorCreate(seq * inter * 4); // gate
        bufs[7] = try ctx.tensorCreate(seq * w * 4); // t (residual delta)
        const x_d = bufs[0];
        const q_d = bufs[1];
        const k_d = bufs[2];
        const v_d = bufs[3];
        const attn_d = bufs[4];
        const up_d = bufs[5];
        const gate_d = bufs[6];
        const t_d = bufs[7];

        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        // Record the whole body into one submission (elide the per-op
        // submit+fence; norm-weight uploads via nbuf are pointer-cached so they
        // don't flush after warmup). Big win for these many-tiny-op forwards.
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();

        for (cpu.layers) |*l| {
            const qkv = l.qkv.bytes; // packed [3w, w]
            const ug = l.up_gate.bytes; // packed [2*inter, w]: up=first, gate=second
            // --- Attention (post-LN: reads x directly) ---
            try ctx.opMatmul(q_d, 0, x_d, 0, seq, qkv[0..seg], false, w, w, 1.0, l.qkv_bias[0..w]);
            try ctx.opMatmul(k_d, 0, x_d, 0, seq, qkv[seg .. 2 * seg], false, w, w, 1.0, l.qkv_bias[w .. 2 * w]);
            try ctx.opMatmul(v_d, 0, x_d, 0, seq, qkv[2 * seg .. 3 * seg], false, w, w, 1.0, l.qkv_bias[2 * w .. 3 * w]);
            try ctx.opElt(.rope_half, q_d, null, freqs, null, .{ .u0 = @intCast(seq * heads * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(heads) }, seq * heads * half, 1, 1);
            try ctx.opElt(.rope_half, k_d, null, freqs, null, .{ .u0 = @intCast(seq * heads * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(heads) }, seq * heads * half, 1, 1);
            try ctx.opElt(.attn_full, q_d, k_d, v_d, attn_d, .{ .u0 = @intCast(seq), .u1 = @intCast(heads), .u2 = @intCast(heads), .u3 = @intCast(hd), .f0 = scale }, seq * heads, 1, 1);
            try ctx.opMatmul(t_d, 0, attn_d, 0, seq, l.o.bytes, false, w, w, 1.0, l.o_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * w) }, seq * w, 1, 1);
            try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, l.attn_ln_w), try nbuf(ctx, l.attn_ln_b), .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);

            // --- MLP (GeGLU: up=first split, gate=second; reads post-attn_ln x) ---
            try ctx.opMatmul(up_d, 0, x_d, 0, seq, ug[0..useg], false, inter, w, 1.0, null);
            try ctx.opMatmul(gate_d, 0, x_d, 0, seq, ug[useg .. 2 * useg], false, inter, w, 1.0, null);
            try ctx.opElt(.gelu_mul, gate_d, up_d, null, null, .{ .u0 = @intCast(seq * inter) }, seq * inter, 1, 1);
            try ctx.opMatmul(t_d, 0, gate_d, 0, seq, l.down.bytes, false, w, inter, 1.0, l.down_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(seq * w) }, seq * w, 1, 1);
            try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, l.mlp_ln_w), try nbuf(ctx, l.mlp_ln_b), .{ .u0 = @intCast(seq), .u1 = @intCast(w), .f0 = eps }, seq, 1, 1);
        }

        try ctx.endBatch();

        // Host: CLS pool (token 0) + L2 normalize.
        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        @memcpy(out, x_host[0..w]);
        l2normalize(out);
    }

    /// Batched Vulkan encode: `ids_list[i]` → `outs[i]`. Mirrors the CPU
    /// `Model.embedBatch` (ragged packing: all GEMMs / LayerNorms / GeGLU over
    /// `total = sum(seq_i)` rows; only RoPE + attention loop per item, via the
    /// `rope_half`/`attn_full` element-offset push fields). One
    /// upload→forward→download for the whole batch.
    pub fn embedBatch(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
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
        const sin_off: u32 = @intCast(max_seq * half);

        // Host: pack embeddings + token_type[0] + LayerNorm per row.
        const x_host = try gpa.alloc(f32, total * w);
        defer gpa.free(x_host);
        for (ids_list, 0..) |ids, i| try qwen3.embedTokens(cpu.word_emb, ids, x_host[row_off[i] * w ..][0 .. ids.len * w]);
        for (0..total) |t| {
            for (x_host[t * w ..][0..w], cpu.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x_host, x_host, cpu.emb_ln_w, cpu.emb_ln_b, eps);

        var freqs = try uploadFreqs(ctx, gpa, max_seq, hd, cfg.rope_theta);
        defer ctx.tensorDestroy(&freqs);

        var bufs: [8]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*bf| ctx.tensorDestroy(bf);
        const sizes = [bufs.len]usize{ total * w, total * w, total * w, total * w, total * w, total * inter, total * inter, total * w };
        for (&bufs, sizes) |*bf, s| bf.* = try ctx.tensorCreate(s * 4);
        const x_d = bufs[0];
        const q_d = bufs[1];
        const k_d = bufs[2];
        const v_d = bufs[3];
        const attn_d = bufs[4];
        const up_d = bufs[5];
        const gate_d = bufs[6];
        const t_d = bufs[7];

        // Per-item bounds for block-diagonal batched attention; bind once.
        var bounds_d = try ctx.tensorCreate(2 * total * 4);
        defer ctx.tensorDestroy(&bounds_d);
        {
            const bounds = try gpa.alloc(u32, 2 * total);
            defer gpa.free(bounds);
            for (0..b) |i| {
                for (row_off[i]..row_off[i + 1]) |r| {
                    bounds[r] = @intCast(row_off[i]);
                    bounds[total + r] = @intCast(row_off[i + 1]);
                }
            }
            try ctx.tensorUpload(bounds_d, std.mem.sliceAsBytes(bounds));
        }

        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        ctx.attnBatchedBind(q_d, k_d, v_d, attn_d, bounds_d);
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();

        for (cpu.layers) |*l| {
            const qkv = l.qkv.bytes;
            const ug = l.up_gate.bytes;
            try ctx.opMatmul(q_d, 0, x_d, 0, total, qkv[0..seg], false, w, w, 1.0, l.qkv_bias[0..w]);
            try ctx.opMatmul(k_d, 0, x_d, 0, total, qkv[seg .. 2 * seg], false, w, w, 1.0, l.qkv_bias[w .. 2 * w]);
            try ctx.opMatmul(v_d, 0, x_d, 0, total, qkv[2 * seg .. 3 * seg], false, w, w, 1.0, l.qkv_bias[2 * w .. 3 * w]);
            for (0..b) |i| {
                const L = ids_list[i].len;
                const roff: u32 = @intCast(row_off[i] * w);
                try ctx.opElt(.rope_half, q_d, null, freqs, null, .{ .u0 = @intCast(L * heads * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(heads), .u5 = roff }, L * heads * half, 1, 1);
                try ctx.opElt(.rope_half, k_d, null, freqs, null, .{ .u0 = @intCast(L * heads * half), .u1 = @intCast(half), .u2 = sin_off, .u3 = @intCast(heads), .u5 = roff }, L * heads * half, 1, 1);
            }
            try ctx.attnBatchedDispatch(total, heads, heads, hd, scale);
            try ctx.opMatmul(t_d, 0, attn_d, 0, total, l.o.bytes, false, w, w, 1.0, l.o_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(total * w) }, total * w, 1, 1);
            try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, l.attn_ln_w), try nbuf(ctx, l.attn_ln_b), .{ .u0 = @intCast(total), .u1 = @intCast(w), .f0 = eps }, total, 1, 1);

            try ctx.opMatmul(up_d, 0, x_d, 0, total, ug[0..useg], false, inter, w, 1.0, null);
            try ctx.opMatmul(gate_d, 0, x_d, 0, total, ug[useg .. 2 * useg], false, inter, w, 1.0, null);
            try ctx.opElt(.gelu_mul, gate_d, up_d, null, null, .{ .u0 = @intCast(total * inter) }, total * inter, 1, 1);
            try ctx.opMatmul(t_d, 0, gate_d, 0, total, l.down.bytes, false, w, inter, 1.0, l.down_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(total * w) }, total * w, 1, 1);
            try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, l.mlp_ln_w), try nbuf(ctx, l.mlp_ln_b), .{ .u0 = @intCast(total), .u1 = @intCast(w), .f0 = eps }, total, 1, 1);
        }
        try ctx.endBatch();

        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
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

test "snowflake Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelGpu.init(&cpu);

    const ids = [_]u32{ 0, 33600, 31, 8999, 2 }; // <s> "hello world" </s>
    const cpu_out = try gpa.alloc(f32, es.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, es.embed_dim);
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
    try std.testing.expect(dot >= 0.9995); // tanh-gelu vs CPU erf-gelu
    try std.testing.expect(maxad < 2e-2);
}

test "snowflake Vulkan embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelGpu.init(&cpu);

    const item0 = [_]u32{ 0, 33600, 31, 2 };
    const item1 = [_]u32{ 0, 8999, 33600, 31, 2 };
    const item2 = [_]u32{ 0, 31, 2 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][es.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(ctx, io, gpa, ids, &single[i]);
    var batched: [3][es.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(ctx, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}
