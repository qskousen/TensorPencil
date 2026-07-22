//! SigLIP2 encoders on the CUDA backend. Device body via `cuda.Backend` ops
//! (opConvF16 for the f32-weight GEMMs, opLayerNorm, generic non-causal `attn`,
//! tanh-gelu, opAdd); the cheap head (last-token pool + projection, or the MAP
//! head) runs on the host. Numerics differ from CPU by the f16 tensor-core GEMM
//! regime (parity is relative: cosine + RMSE, like gemma_vit_cuda).

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const siglip = @import("embed_siglip.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// GEMM in the weight's dtype (f32 here → opConvF16). y[m][co] = x @ Wᵀ + bias.
fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w_bytes: []const u8, co: usize, k: usize, bias: []const f32) !void {
    try be.opConvF16(dst, 0, src, m, w_bytes, co, k, bias);
}

/// SigLIP2 text tower on CUDA. Borrows a loaded CPU `TextModel`'s f32 weights.
pub const TextModelCuda = struct {
    cpu: *const siglip.TextModel,

    pub fn init(cpu: *const siglip.TextModel) TextModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const TextModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const n = cfg.context_length;
        const w = cfg.width;
        const inter = 4 * w;
        const heads = cfg.n_heads;
        const hd = cfg.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const seg = w * w * @sizeOf(f32);
        std.debug.assert(out.len == siglip.embed_dim);

        const x_host = try gpa.alloc(f32, n * w);
        defer gpa.free(x_host);
        {
            const padded = try gpa.alloc(u32, n);
            defer gpa.free(padded);
            for (padded, 0..) |*p, t| p.* = if (t < ids.len) ids[t] else 0;
            try qwen3.embedTokens(cpu.token_emb, padded, x_host);
            for (x_host, cpu.pos_emb) |*xi, pe| xi.* += pe;
        }

        be.weightScopeBegin();
        defer {
            be.weightScopeEnd();
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var bufs: [7]Buf = @splat(.{});
        defer for (&bufs) |*b| be.tensorDestroy(b);
        const sizes = [bufs.len]usize{ n * w, n * w, n * w, n * w, n * w, n * inter, n * w };
        for (&bufs, sizes) |*b, s| b.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (cpu.layers) |*l| {
            const ip = l.in_proj.bytes;
            try be.opLayerNorm(x_d, normed_d, l.ln1_w, l.ln1_b, n, w, cfg.ln_eps);
            try gemm(be, q_d, normed_d, n, ip[0..seg], w, w, l.in_proj_bias[0..w]);
            try gemm(be, k_d, normed_d, n, ip[seg .. 2 * seg], w, w, l.in_proj_bias[w .. 2 * w]);
            try gemm(be, v_d, normed_d, n, ip[2 * seg .. 3 * seg], w, w, l.in_proj_bias[2 * w .. 3 * w]);
            try be.attn(q_d, k_d, v_d, big_d, n, n, heads, heads, hd, scale, false);
            try gemm(be, t_d, big_d, n, l.out_proj.bytes, w, w, l.out_proj_bias);
            try be.opAdd(x_d, t_d, n * w);

            try be.opLayerNorm(x_d, normed_d, l.ln2_w, l.ln2_b, n, w, cfg.ln_eps);
            try gemm(be, big_d, normed_d, n, l.c_fc.bytes, inter, w, l.c_fc_bias);
            try be.gelu(big_d, n * inter);
            try gemm(be, t_d, big_d, n, l.c_proj.bytes, w, inter, l.c_proj_bias);
            try be.opAdd(x_d, t_d, n * w);
        }
        try be.opLayerNorm(x_d, x_d, cpu.ln_final_w, cpu.ln_final_b, n, w, cfg.ln_eps);
        try be.endBatch();

        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        const pooled = x_host[(n - 1) * w ..][0..w];
        try ops.matmul.matmul(io, gpa, out, pooled, 1, cpu.text_proj, cpu.text_proj_bias);
        l2normalize(out);
    }
};

/// SigLIP2 visual tower on CUDA. ViT body device-side, MAP head on host.
pub const VisualModelCuda = struct {
    cpu: *const siglip.VisualModel,

    pub fn init(cpu: *const siglip.VisualModel) VisualModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const VisualModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, img: []const f32, out: []f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const w = cfg.width;
        const inter = cfg.mlp_dim;
        const np = cfg.nPatches();
        const heads = cfg.n_heads;
        const hd = cfg.head_dim;
        const pin = cfg.patchIn();
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const seg = w * w * @sizeOf(f32);

        const patch_in = try cpu.patchify(gpa, img);
        defer gpa.free(patch_in);

        be.weightScopeBegin();
        defer {
            be.weightScopeEnd();
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var bufs: [7]Buf = @splat(.{});
        defer for (&bufs) |*b| be.tensorDestroy(b);
        const sizes = [bufs.len]usize{ np * w, np * w, np * w, np * w, np * w, np * inter, np * @max(w, pin) };
        for (&bufs, sizes) |*b, s| b.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        try be.tensorUpload(t_d, std.mem.sliceAsBytes(patch_in));
        try gemm(be, x_d, t_d, np, cpu.patch_w.bytes, w, pin, cpu.patch_b);
        try be.tensorUpload(t_d, std.mem.sliceAsBytes(cpu.pos_emb));
        try be.opAdd(x_d, t_d, np * w);

        for (cpu.blocks) |*b| {
            const qkv = b.qkv.bytes;
            try be.opLayerNorm(x_d, normed_d, b.norm1_w, b.norm1_b, np, w, cfg.ln_eps);
            try gemm(be, q_d, normed_d, np, qkv[0..seg], w, w, b.qkv_b[0..w]);
            try gemm(be, k_d, normed_d, np, qkv[seg .. 2 * seg], w, w, b.qkv_b[w .. 2 * w]);
            try gemm(be, v_d, normed_d, np, qkv[2 * seg .. 3 * seg], w, w, b.qkv_b[2 * w .. 3 * w]);
            try be.attn(q_d, k_d, v_d, big_d, np, np, heads, heads, hd, scale, false);
            try gemm(be, t_d, big_d, np, b.attn_proj.bytes, w, w, b.attn_proj_b);
            try be.opAdd(x_d, t_d, np * w);

            try be.opLayerNorm(x_d, normed_d, b.norm2_w, b.norm2_b, np, w, cfg.ln_eps);
            try gemm(be, big_d, normed_d, np, b.fc1.bytes, inter, w, b.fc1_b);
            try be.gelu(big_d, np * inter);
            try gemm(be, t_d, big_d, np, b.fc2.bytes, w, inter, b.fc2_b);
            try be.opAdd(x_d, t_d, np * w);
        }
        try be.opLayerNorm(x_d, x_d, cpu.norm_w, cpu.norm_b, np, w, cfg.ln_eps);
        try be.endBatch();

        const x_host = try gpa.alloc(f32, np * w);
        defer gpa.free(x_host);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        try cpu.mapHead(io, gpa, x_host, out);
    }
};

fn l2normalize(out: []f32) void {
    var ss: f32 = 0;
    for (out) |v| ss += v * v;
    const norm = @sqrt(ss);
    if (norm > 0) {
        const inv = 1.0 / norm;
        for (out) |*v| v.* *= inv;
    }
}

// --- tests -----------------------------------------------------------------

fn parity(cpu_out: []const f32, gpu_out: []const f32) struct { cos: f32, rmse: f32 } {
    var dot: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    var num: f32 = 0;
    var den: f32 = 0;
    for (cpu_out, gpu_out) |a, b| {
        dot += a * b;
        na += a * a;
        nb += b * b;
        num += (a - b) * (a - b);
        den += a * a;
    }
    return .{ .cos = dot / (@sqrt(na) * @sqrt(nb)), .rmse = @sqrt(num / den) };
}

test "siglip2 text tower CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelCuda.init(&cpu);

    const ids = [_]u32{ 17534, 2134, 1 };
    const c = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, &ids, c);
    try dev.embed(be, io, gpa, &ids, g);
    const p = parity(c, g);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ p.cos, p.rmse });
    try std.testing.expect(p.cos > 0.999 and p.rmse < 0.05);
}

test "siglip2 visual tower CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_input.f32", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(raw);
    const img = try gpa.alloc(f32, raw.len / @sizeOf(f32));
    defer gpa.free(img);
    @memcpy(std.mem.sliceAsBytes(img), raw);

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelCuda.init(&cpu);

    const c = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, img, c);
    try dev.embed(be, io, gpa, img, g);
    const p = parity(c, g);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ p.cos, p.rmse });
    try std.testing.expect(p.cos > 0.999 and p.rmse < 0.05);
}
