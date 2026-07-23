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

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
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

    /// Batched CUDA text encode: mirrors CPU `TextModel.embedBatch`. Uniform
    /// 64-token windows → one [B*context_length, width] device activation; only
    /// `attn` loops per item (via `dbOffset` views); pool + projection on host.
    pub fn embedBatch(self: *const TextModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const n = cfg.context_length;
        const w = cfg.width;
        const inter = 4 * w;
        const heads = cfg.n_heads;
        const hd = cfg.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const seg = w * w * @sizeOf(f32);
        const b = ids_list.len;
        std.debug.assert(outs.len == b and b > 0);
        const total = b * n;

        const x_host = try gpa.alloc(f32, total * w);
        defer gpa.free(x_host);
        {
            const padded = try gpa.alloc(u32, total);
            defer gpa.free(padded);
            for (ids_list, 0..) |ids, i| {
                for (padded[i * n ..][0..n], 0..) |*pd, t| pd.* = if (t < ids.len) ids[t] else 0;
            }
            try qwen3.embedTokens(cpu.token_emb, padded, x_host);
            for (0..b) |i| {
                for (x_host[i * n * w ..][0 .. n * w], cpu.pos_emb) |*xi, pe| xi.* += pe;
            }
        }

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var bufs: [7]Buf = @splat(.{});
        defer for (&bufs) |*bf| be.tensorDestroy(bf);
        const sizes = [bufs.len]usize{ total * w, total * w, total * w, total * w, total * w, total * inter, total * w };
        for (&bufs, sizes) |*bf, s| bf.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
        // Uniform-length (64) item bounds for the block-diagonal batched attention.
        var bounds_d = try be.tensorCreate(2 * total * 4);
        defer be.tensorDestroy(&bounds_d);
        {
            const bounds = try gpa.alloc(u32, 2 * total);
            defer gpa.free(bounds);
            for (0..b) |i| {
                for (i * n..(i + 1) * n) |r| {
                    bounds[r] = @intCast(i * n);
                    bounds[total + r] = @intCast((i + 1) * n);
                }
            }
            try be.tensorUpload(bounds_d, std.mem.sliceAsBytes(bounds));
        }

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (cpu.layers) |*l| {
            const ip = l.in_proj.bytes;
            try be.opLayerNorm(x_d, normed_d, l.ln1_w, l.ln1_b, total, w, cfg.ln_eps);
            try gemm(be, q_d, normed_d, total, ip[0..seg], w, w, l.in_proj_bias[0..w]);
            try gemm(be, k_d, normed_d, total, ip[seg .. 2 * seg], w, w, l.in_proj_bias[w .. 2 * w]);
            try gemm(be, v_d, normed_d, total, ip[2 * seg .. 3 * seg], w, w, l.in_proj_bias[2 * w .. 3 * w]);
            try be.opAttnBatched(q_d, k_d, v_d, big_d, bounds_d, total, heads, heads, hd, scale);
            try gemm(be, t_d, big_d, total, l.out_proj.bytes, w, w, l.out_proj_bias);
            try be.opAdd(x_d, t_d, total * w);

            try be.opLayerNorm(x_d, normed_d, l.ln2_w, l.ln2_b, total, w, cfg.ln_eps);
            try gemm(be, big_d, normed_d, total, l.c_fc.bytes, inter, w, l.c_fc_bias);
            try be.gelu(big_d, total * inter);
            try gemm(be, t_d, big_d, total, l.c_proj.bytes, w, inter, l.c_proj_bias);
            try be.opAdd(x_d, t_d, total * w);
        }
        try be.opLayerNorm(x_d, x_d, cpu.ln_final_w, cpu.ln_final_b, total, w, cfg.ln_eps);
        try be.endBatch();

        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        const pooled = try gpa.alloc(f32, b * w);
        defer gpa.free(pooled);
        for (0..b) |i| @memcpy(pooled[i * w ..][0..w], x_host[(i * n + n - 1) * w ..][0..w]);
        const projd = try gpa.alloc(f32, b * w);
        defer gpa.free(projd);
        try ops.matmul.matmul(io, gpa, projd, pooled, b, cpu.text_proj, cpu.text_proj_bias);
        for (0..b) |i| {
            std.debug.assert(outs[i].len == siglip.embed_dim);
            @memcpy(outs[i], projd[i * w ..][0..w]);
            l2normalize(outs[i]);
        }
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

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
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

    /// Batched CUDA visual encode: mirrors CPU `VisualModel.embedBatch`. Fixed
    /// 196 patches per image → ViT body over one [B*nPatches, width] activation;
    /// only `attn` loops per image (via `dbOffset` views); MAP head on host.
    pub fn embedBatch(self: *const VisualModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, imgs: []const []const f32, outs: [][]f32) !void {
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
        const b = imgs.len;
        std.debug.assert(outs.len == b and b > 0);
        const total = b * np;

        const patch_all = try gpa.alloc(f32, total * pin);
        defer gpa.free(patch_all);
        for (imgs, 0..) |img, i| {
            const pi = try cpu.patchify(gpa, img);
            defer gpa.free(pi);
            @memcpy(patch_all[i * np * pin ..][0 .. np * pin], pi);
        }
        const pos_tile = try gpa.alloc(f32, total * w);
        defer gpa.free(pos_tile);
        for (0..b) |i| @memcpy(pos_tile[i * np * w ..][0 .. np * w], cpu.pos_emb);

        // No weight scope: the f16-converted GEMM weights stay RESIDENT across
        // embed calls (they borrow the mmap-stable CPU model). A scope would
        // free+re-upload the whole model every call — a per-forward cost that
        // dwarfs the compute. The encoder is opened once and reused.
        defer {
            be.freeAttnScratch();
            be.freeConvScratch();
        }
        var bufs: [7]Buf = @splat(.{});
        defer for (&bufs) |*bf| be.tensorDestroy(bf);
        const sizes = [bufs.len]usize{ total * w, total * w, total * w, total * w, total * w, total * inter, total * @max(w, pin) };
        for (&bufs, sizes) |*bf, s| bf.* = try be.tensorCreate(s * 4);
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        // Uniform-length (196) item bounds for the block-diagonal batched attention.
        var bounds_d = try be.tensorCreate(2 * total * 4);
        defer be.tensorDestroy(&bounds_d);
        {
            const bounds = try gpa.alloc(u32, 2 * total);
            defer gpa.free(bounds);
            for (0..b) |i| {
                for (i * np..(i + 1) * np) |r| {
                    bounds[r] = @intCast(i * np);
                    bounds[total + r] = @intCast((i + 1) * np);
                }
            }
            try be.tensorUpload(bounds_d, std.mem.sliceAsBytes(bounds));
        }

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        try be.tensorUpload(t_d, std.mem.sliceAsBytes(patch_all));
        try gemm(be, x_d, t_d, total, cpu.patch_w.bytes, w, pin, cpu.patch_b);
        try be.tensorUpload(t_d, std.mem.sliceAsBytes(pos_tile));
        try be.opAdd(x_d, t_d, total * w);

        for (cpu.blocks) |*bl| {
            const qkv = bl.qkv.bytes;
            try be.opLayerNorm(x_d, normed_d, bl.norm1_w, bl.norm1_b, total, w, cfg.ln_eps);
            try gemm(be, q_d, normed_d, total, qkv[0..seg], w, w, bl.qkv_b[0..w]);
            try gemm(be, k_d, normed_d, total, qkv[seg .. 2 * seg], w, w, bl.qkv_b[w .. 2 * w]);
            try gemm(be, v_d, normed_d, total, qkv[2 * seg .. 3 * seg], w, w, bl.qkv_b[2 * w .. 3 * w]);
            try be.opAttnBatched(q_d, k_d, v_d, big_d, bounds_d, total, heads, heads, hd, scale);
            try gemm(be, t_d, big_d, total, bl.attn_proj.bytes, w, w, bl.attn_proj_b);
            try be.opAdd(x_d, t_d, total * w);

            try be.opLayerNorm(x_d, normed_d, bl.norm2_w, bl.norm2_b, total, w, cfg.ln_eps);
            try gemm(be, big_d, normed_d, total, bl.fc1.bytes, inter, w, bl.fc1_b);
            try be.gelu(big_d, total * inter);
            try gemm(be, t_d, big_d, total, bl.fc2.bytes, w, inter, bl.fc2_b);
            try be.opAdd(x_d, t_d, total * w);
        }
        try be.opLayerNorm(x_d, x_d, cpu.norm_w, cpu.norm_b, total, w, cfg.ln_eps);
        try be.endBatch();

        const x_host = try gpa.alloc(f32, total * w);
        defer gpa.free(x_host);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        try cpu.mapHeadBatch(io, gpa, x_host, outs);
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

test "siglip2 text tower CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelCuda.init(&cpu);

    const item0 = [_]u32{ 17534, 2134, 1 };
    const item1 = [_]u32{ 2134, 17534, 2134, 1 };
    const item2 = [_]u32{ 17534, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][siglip.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(be, io, gpa, ids, &single[i]);
    var batched: [3][siglip.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(be, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("text item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3);
    }
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

test "siglip2 visual tower CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelCuda.init(&cpu);

    const nimg = 3 * 224 * 224;
    var prng = std.Random.DefaultPrng.init(0x71B2);
    const r = prng.random();
    const imgs_data = try gpa.alloc(f32, 2 * nimg);
    defer gpa.free(imgs_data);
    for (imgs_data) |*e| e.* = (r.float(f32) - 0.5) * 2.0;
    const imgs = [_][]const f32{ imgs_data[0..nimg], imgs_data[nimg..] };

    var single: [2][siglip.embed_dim]f32 = undefined;
    for (imgs, 0..) |img, i| try dev.embed(be, io, gpa, img, &single[i]);
    var batched: [2][siglip.embed_dim]f32 = undefined;
    var outs: [2][]f32 = .{ &batched[0], &batched[1] };
    try dev.embedBatch(be, io, gpa, &imgs, &outs);

    for (0..2) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("visual item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3);
    }
}
