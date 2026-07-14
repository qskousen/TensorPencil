//! Gemma 3 vision tower (SigLIP + projector) on the Vulkan backend: the 27
//! encoder blocks run device-side, the cheap projector on the host (like
//! gemma_vit_cuda). Correctness-first, mirroring the Vulkan LLM path.
//!
//! Vulkan has no f16-weight GEMM, so the f16 block weights are dequantized to
//! f32 once at load (stable host pointers → the Context weight cache works)
//! and fed to the f32 `opMatmul` (+bias). Attention is the full non-causal
//! `attn_full` kernel (arbitrary head_dim 72), LayerNorm the `layernorm`
//! kernel, GELU the `gelu` kernel — all added for this port. The patch conv
//! and learned position embedding, and the host projector, are the same as
//! the CPU/CUDA paths.

const std = @import("std");
const gemma_vit = @import("gemma_vit.zig");
const gpu = @import("../gpu/context.zig");
const ops = @import("../ops.zig");
const safetensors = @import("../safetensors.zig");

const Vit = gemma_vit.Vit;
const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// A block's projection weights dequantized to f32 (Vulkan opMatmul needs
/// f32 weights); biases/norms are borrowed from the CPU Vit (already f32).
const BlockF32 = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    o: []f32,
    up: []f32,
    down: []f32,
};

pub const VitGpu = struct {
    arena: std.heap.ArenaAllocator,
    vit: *const Vit,
    blocks: []BlockF32,

    pub fn load(gpa: std.mem.Allocator, vit: *const Vit) !VitGpu {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const blocks = try alloc.alloc(BlockF32, vit.blocks.len);
        for (blocks, vit.blocks) |*dst, *src| {
            dst.* = .{
                .q = try deq(alloc, src.q),
                .k = try deq(alloc, src.k),
                .v = try deq(alloc, src.v),
                .o = try deq(alloc, src.out),
                .up = try deq(alloc, src.up),
                .down = try deq(alloc, src.down),
            };
        }
        return .{ .arena = arena, .vit = vit, .blocks = blocks };
    }

    fn deq(alloc: std.mem.Allocator, w: ops.matmul.Weight) ![]f32 {
        const out = try alloc.alloc(f32, w.rows * w.cols);
        try safetensors.convertToF32(w.dtype, w.bytes, out);
        return out;
    }

    pub fn deinit(self: *VitGpu) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode interleaved RGB pixels to Gemma image-token embeddings on
    /// Vulkan. Same contract as gemma_vit.Vit.encode.
    pub fn encode(self: *const VitGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
        const vit = self.vit;
        const cfg = vit.cfg;
        const dim = cfg.dim;
        const heads = cfg.n_heads;
        const hd = cfg.headDim();
        const side = cfg.side();
        const np = side * side;
        const kdim = 3 * cfg.patch * cfg.patch;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        const patches = try vit.patchMatrix(gpa, rgb, width, height);
        defer gpa.free(patches);

        var bufs: [7]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(np * dim * 4); // x
        bufs[1] = try ctx.tensorCreate(np * dim * 4); // normed
        bufs[2] = try ctx.tensorCreate(np * dim * 4); // q
        bufs[3] = try ctx.tensorCreate(np * dim * 4); // k
        bufs[4] = try ctx.tensorCreate(np * dim * 4); // v
        bufs[5] = try ctx.tensorCreate(np * @max(dim, cfg.ffn) * 4); // attn / ffn hidden
        bufs[6] = try ctx.tensorCreate(np * dim * 4); // t (patch upload / residual delta)
        const x_d = bufs[0];
        const normed_d = bufs[1];
        const q_d = bufs[2];
        const k_d = bufs[3];
        const v_d = bufs[4];
        const big_d = bufs[5];
        const t_d = bufs[6];

        // Patch embed GEMM + learned position embedding.
        try ctx.tensorUpload(t_d, std.mem.sliceAsBytes(patches[0 .. np * kdim]));
        try ctx.opMatmul(x_d, 0, t_d, 0, np, std.mem.sliceAsBytes(vit.patch_w), false, dim, kdim, 1.0, vit.patch_b);
        try ctx.tensorUpload(t_d, std.mem.sliceAsBytes(vit.pos_embd));
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * dim) }, np * dim, 1, 1);

        for (vit.blocks, self.blocks) |*blk, *bf| {
            // --- Attention ---
            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, blk.ln1_w), try nbuf(ctx, blk.ln1_b), .{ .u0 = @intCast(np), .u1 = @intCast(dim), .f0 = cfg.eps }, np, 1, 1);
            try ctx.opMatmul(q_d, 0, normed_d, 0, np, std.mem.sliceAsBytes(bf.q), false, dim, dim, 1.0, blk.q_b);
            try ctx.opMatmul(k_d, 0, normed_d, 0, np, std.mem.sliceAsBytes(bf.k), false, dim, dim, 1.0, blk.k_b);
            try ctx.opMatmul(v_d, 0, normed_d, 0, np, std.mem.sliceAsBytes(bf.v), false, dim, dim, 1.0, blk.v_b);
            try ctx.opElt(.attn_full, q_d, k_d, v_d, big_d, .{ .u0 = @intCast(np), .u1 = @intCast(heads), .u2 = @intCast(heads), .u3 = @intCast(hd), .f0 = scale }, np * heads, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, np, std.mem.sliceAsBytes(bf.o), false, dim, dim, 1.0, blk.out_b);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * dim) }, np * dim, 1, 1);

            // --- MLP (gelu) ---
            try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, blk.ln2_w), try nbuf(ctx, blk.ln2_b), .{ .u0 = @intCast(np), .u1 = @intCast(dim), .f0 = cfg.eps }, np, 1, 1);
            try ctx.opMatmul(big_d, 0, normed_d, 0, np, std.mem.sliceAsBytes(bf.up), false, cfg.ffn, dim, 1.0, blk.up_b);
            try ctx.opElt(.gelu, big_d, null, null, null, .{ .u0 = @intCast(np * cfg.ffn) }, np * cfg.ffn, 1, 1);
            try ctx.opMatmul(t_d, 0, big_d, 0, np, std.mem.sliceAsBytes(bf.down), false, dim, cfg.ffn, 1.0, blk.down_b);
            try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(np * dim) }, np * dim, 1, 1);
        }
        try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, vit.post_ln_w), try nbuf(ctx, vit.post_ln_b), .{ .u0 = @intCast(np), .u1 = @intCast(dim), .f0 = cfg.eps }, np, 1, 1);

        // Download post-LN patch states; the projector runs on the host.
        const x_host = try gpa.alloc(f32, np * dim);
        defer gpa.free(x_host);
        try ctx.tensorDownload(x_d, std.mem.sliceAsBytes(x_host));
        return vit.project(io, gpa, x_host);
    }
};
