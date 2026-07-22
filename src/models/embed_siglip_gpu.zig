//! SigLIP2 encoders on the Vulkan backend. Thin device forwards that reuse the
//! CPU model's (f32, mmap-stable) weights directly — Vulkan's `opMatmul` takes
//! host weight bytes and caches a device buffer keyed on the pointer, so no
//! dequant is needed (unlike gemma_vit_gpu, whose weights were f16). Every op
//! (LayerNorm, non-causal `attn_full`, tanh-gelu, residual add) is an existing
//! Context entry point; the cheap head (pool + projection) runs on the host.
//!
//! Mirrors the CPU forwards in `embed_siglip.zig` and is validated against them
//! (device parity test at the bottom).

const std = @import("std");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const siglip = @import("embed_siglip.zig");

const Buf = gpu.DeviceBuffer;

/// Wrap a host f32 slice as a small device buffer (norm weights/biases).
fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// SigLIP2 text tower on Vulkan. Borrows a loaded CPU `TextModel`'s weights.
pub const TextModelGpu = struct {
    cpu: *const siglip.TextModel,

    pub fn init(cpu: *const siglip.TextModel) TextModelGpu {
        return .{ .cpu = cpu };
    }

    /// Same contract as `TextModel.embed`: framed ids in, L2-normalized 768-d out.
    pub fn embed(self: *const TextModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cpu = self.cpu;
        const cfg = cpu.cfg;
        const n = cfg.context_length;
        const w = cfg.width;
        const inter = 4 * w;
        const heads = cfg.n_heads;
        const hd = cfg.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        std.debug.assert(out.len == siglip.embed_dim);
        const seg = w * w * @sizeOf(f32); // one [w,w] block of the packed qkv weight

        // Host: token + learned positional embeddings into the fixed 64-window.
        const x_host = try gpa.alloc(f32, n * w);
        defer gpa.free(x_host);
        {
            const padded = try gpa.alloc(u32, n);
            defer gpa.free(padded);
            for (padded, 0..) |*p, t| p.* = if (t < ids.len) ids[t] else 0;
            try qwen3.embedTokens(cpu.token_emb, padded, x_host);
            for (x_host, cpu.pos_emb) |*xi, pe| xi.* += pe;
        }

        var bufs: [7]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(n * w * 4); // x
        bufs[1] = try ctx.tensorCreate(n * w * 4); // normed
        bufs[2] = try ctx.tensorCreate(n * w * 4); // q
        bufs[3] = try ctx.tensorCreate(n * w * 4); // k
        bufs[4] = try ctx.tensorCreate(n * w * 4); // v
        bufs[5] = try ctx.tensorCreate(n * inter * 4); // attn out / ffn hidden
        bufs[6] = try ctx.tensorCreate(n * w * 4); // t (residual delta)
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));

        for (cpu.layers) |*l| {
            const ip = l.in_proj.bytes; // packed [3w, w] f32
            // --- Attention (pre-LN) ---
            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, l.ln1_w), try nbuf(ctx, l.ln1_b), .{ .u0 = @intCast(n), .u1 = @intCast(w), .f0 = cfg.ln_eps }, n, 1, 1);
            try ctx.opMatmul(q_d, 0, normed_d, 0, n, ip[0..seg], false, w, w, 1.0, l.in_proj_bias[0..w]);
            try ctx.opMatmul(k_d, 0, normed_d, 0, n, ip[seg .. 2 * seg], false, w, w, 1.0, l.in_proj_bias[w .. 2 * w]);
            try ctx.opMatmul(v_d, 0, normed_d, 0, n, ip[2 * seg .. 3 * seg], false, w, w, 1.0, l.in_proj_bias[2 * w .. 3 * w]);
            try ctx.opElt(.attn_full, q_d, k_d, v_d, big_d, .{ .u0 = @intCast(n), .u1 = @intCast(heads), .u2 = @intCast(heads), .u3 = @intCast(hd), .f0 = scale }, n * heads, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, n, l.out_proj.bytes, false, w, w, 1.0, l.out_proj_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(n * w) }, n * w, 1, 1);

            // --- MLP (pre-LN, tanh-gelu) ---
            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, l.ln2_w), try nbuf(ctx, l.ln2_b), .{ .u0 = @intCast(n), .u1 = @intCast(w), .f0 = cfg.ln_eps }, n, 1, 1);
            try ctx.opMatmul(big_d, 0, normed_d, 0, n, l.c_fc.bytes, false, inter, w, 1.0, l.c_fc_bias);
            try ctx.opElt(.gelu, big_d, null, null, null, .{ .u0 = @intCast(n * inter) }, n * inter, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, n, l.c_proj.bytes, false, w, inter, 1.0, l.c_proj_bias);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(n * w) }, n * w, 1, 1);
        }
        try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, cpu.ln_final_w), try nbuf(ctx, cpu.ln_final_b), .{ .u0 = @intCast(n), .u1 = @intCast(w), .f0 = cfg.ln_eps }, n, 1, 1);

        // Host: last-token pool → text_projection → L2 normalize.
        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        const pooled = x_host[(n - 1) * w ..][0..w];
        try ops.matmul.matmul(io, gpa, out, pooled, 1, cpu.text_proj, cpu.text_proj_bias);
        var ss: f32 = 0;
        for (out) |val| ss += val * val;
        const norm = @sqrt(ss);
        if (norm > 0) {
            const inv = 1.0 / norm;
            for (out) |*val| val.* *= inv;
        }
    }
};

/// SigLIP2 visual tower on Vulkan. The ViT body (patch-embed + 12 blocks +
/// trunk.norm) runs device-side; the small MAP attention-pool head runs on the
/// host (`VisualModel.mapHead`) — it's a cross-attention (1 latent × 196
/// tokens) the self-attention `attn_full` kernel doesn't cover, and it's cheap.
pub const VisualModelGpu = struct {
    cpu: *const siglip.VisualModel,

    pub fn init(cpu: *const siglip.VisualModel) VisualModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const VisualModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, img: []const f32, out: []f32) !void {
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

        const patch_in = try cpu.patchify(gpa, img); // [np, patchIn]
        defer gpa.free(patch_in);

        var bufs: [7]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(np * w * 4); // x
        bufs[1] = try ctx.tensorCreate(np * w * 4); // normed
        bufs[2] = try ctx.tensorCreate(np * w * 4); // q
        bufs[3] = try ctx.tensorCreate(np * w * 4); // k
        bufs[4] = try ctx.tensorCreate(np * w * 4); // v
        bufs[5] = try ctx.tensorCreate(np * inter * 4); // attn out / ffn hidden
        bufs[6] = try ctx.tensorCreate(np * @max(w, pin) * 4); // t / upload staging
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        // Patch-embed GEMM + learned positional embedding.
        try ctx.tensorUpload(t_d, std.mem.sliceAsBytes(patch_in));
        try ctx.opMatmul(x_d, 0, t_d, 0, np, cpu.patch_w.bytes, false, w, pin, 1.0, cpu.patch_b);
        try ctx.tensorUpload(t_d, std.mem.sliceAsBytes(cpu.pos_emb));
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * w) }, np * w, 1, 1);

        for (cpu.blocks) |*b| {
            const qkv = b.qkv.bytes;
            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, b.norm1_w), try nbuf(ctx, b.norm1_b), .{ .u0 = @intCast(np), .u1 = @intCast(w), .f0 = cfg.ln_eps }, np, 1, 1);
            try ctx.opMatmul(q_d, 0, normed_d, 0, np, qkv[0..seg], false, w, w, 1.0, b.qkv_b[0..w]);
            try ctx.opMatmul(k_d, 0, normed_d, 0, np, qkv[seg .. 2 * seg], false, w, w, 1.0, b.qkv_b[w .. 2 * w]);
            try ctx.opMatmul(v_d, 0, normed_d, 0, np, qkv[2 * seg .. 3 * seg], false, w, w, 1.0, b.qkv_b[2 * w .. 3 * w]);
            try ctx.opElt(.attn_full, q_d, k_d, v_d, big_d, .{ .u0 = @intCast(np), .u1 = @intCast(heads), .u2 = @intCast(heads), .u3 = @intCast(hd), .f0 = scale }, np * heads, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, np, b.attn_proj.bytes, false, w, w, 1.0, b.attn_proj_b);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * w) }, np * w, 1, 1);

            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, b.norm2_w), try nbuf(ctx, b.norm2_b), .{ .u0 = @intCast(np), .u1 = @intCast(w), .f0 = cfg.ln_eps }, np, 1, 1);
            try ctx.opMatmul(big_d, 0, normed_d, 0, np, b.fc1.bytes, false, inter, w, 1.0, b.fc1_b);
            try ctx.opElt(.gelu, big_d, null, null, null, .{ .u0 = @intCast(np * inter) }, np * inter, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, np, b.fc2.bytes, false, w, inter, 1.0, b.fc2_b);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * w) }, np * w, 1, 1);
        }
        try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, cpu.norm_w), try nbuf(ctx, cpu.norm_b), .{ .u0 = @intCast(np), .u1 = @intCast(w), .f0 = cfg.ln_eps }, np, 1, 1);

        const x_host = try gpa.alloc(f32, np * w);
        defer gpa.free(x_host);
        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        try cpu.mapHead(io, gpa, x_host, out);
    }
};

// --- tests -----------------------------------------------------------------

// Device parity: the Vulkan text tower must match the (ONNX-validated) CPU
// TextModel. Self-skips without a Vulkan device / the integration build, or
// when the checkpoint is absent.
test "siglip2 text tower Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelGpu.init(&cpu);

    const ids = [_]u32{ 17534, 2134, 1 }; // "hello world" + <eos>
    const cpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, siglip.embed_dim);
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
    try std.testing.expect(dot >= 0.9999); // both unit vectors
    try std.testing.expect(maxad < 1e-2);
}

test "siglip2 visual tower Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_input.f32", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(raw);
    const img = try gpa.alloc(f32, raw.len / @sizeOf(f32));
    defer gpa.free(img);
    @memcpy(std.mem.sliceAsBytes(img), raw);

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelGpu.init(&cpu);

    const cpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(gpu_out);
    try cpu.embed(io, gpa, img, cpu_out);
    try dev.embed(ctx, io, gpa, img, gpu_out);

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
