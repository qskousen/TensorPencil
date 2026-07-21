//! Gemma 3 vision tower forward on the CUDA backend: the SigLIP encoder
//! (patch embed, learned position add, 27 pre-LN blocks) runs device-side;
//! the cheap projector (4x4 avg-pool -> soft_emb_norm -> input projection)
//! runs on the host via gemma_vit.Vit.project (256 tokens, negligible next
//! to the 27 blocks over 4096 patches — the part that made the CPU encode
//! ~39 s). Mirrors vit35_cuda, minus the vision RoPE (Gemma uses a learned
//! position embedding) and with separate q/k/v (not fused).
//!
//! The 72-dim heads are zero-padded to 128 on device (head_pad) so the
//! tensor-core attention applies; the pads are exact zeros so scores/outputs
//! match the CPU path up to the f16 GEMM regime. f16 block weights feed
//! opMatmulF16 straight from the mmap; the f32 patch kernel and the
//! (transposed f32) input projection take opConvF16. ViT weights and the
//! attention/GEMM scratch are scoped and released after encoding, so nothing
//! image-sized stays resident under the LLM.

const std = @import("std");
const gemma_vit = @import("gemma_vit.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = gemma_vit.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// GEMM in the weight's storage dtype, y[m][w.rows] = x @ Wᵀ + bias.
fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
    const b = bias orelse &.{};
    switch (w.dtype) {
        .f16 => try be.opMatmulF16(dst, src, m, w.bytes, w.rows, w.cols, b),
        .bf16 => try be.opMatmulBf16(dst, src, m, w.bytes, w.rows, w.cols, b),
        .f32 => try be.opConvF16(dst, 0, src, m, w.bytes, w.rows, w.cols, b),
        else => return error.UnsupportedDType,
    }
}

fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// Encode interleaved RGB pixels to Gemma image-token embeddings on CUDA.
/// Same contract as gemma_vit.Vit.encode; numerics differ by the f16 GEMM
/// regime (validated cos > 0.999 vs the CPU path).
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim(); // 72
    const hd_pad = std.mem.alignForward(usize, hd, 128); // 128
    const side = cfg.side();
    const np = side * side; // 4096
    const kdim = 3 * cfg.patch * cfg.patch;

    const patches = try vit.patchMatrix(gpa, rgb, width, height);
    defer gpa.free(patches);

    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }

    var bufs: [8]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{
        np * dim, // x (residual stream)
        np * dim, // normed / compacted attn output
        np * dim, // proj_tmp (raw q/k/v projection before head_pad) / patch upload
        np * heads * hd_pad, // q (padded heads)
        np * heads * hd_pad, // k
        np * heads * hd_pad, // v
        np * @max(heads * hd_pad, cfg.ffn), // attn out / FFN hidden
        np * dim, // t (residual delta)
    };
    for (&bufs, sizes) |*b, size| b.* = try be.tensorCreate(size * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const proj_d = bufs[2];
    const q_d = bufs[3];
    const k_d = bufs[4];
    const v_d = bufs[5];
    const big_d = bufs[6];
    const t_d = bufs[7];

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // Patch embed GEMM (f32 host kernel) + learned position embedding (added
    // as-is: pos_embd is [np][dim] in the same row-major patch order).
    try be.tensorUpload(sized(proj_d, 0, np * kdim * 4), std.mem.sliceAsBytes(patches));
    try gemm(be, x_d, proj_d, np, Weight.fromF32(vit.patch_w, dim, kdim), vit.patch_b);
    try be.tensorUpload(sized(t_d, 0, np * dim * 4), std.mem.sliceAsBytes(vit.pos_embd));
    try be.opAdd(x_d, t_d, np * dim);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (vit.blocks) |*blk| {
        try be.opLayerNorm(x_d, normed_d, blk.ln1_w, blk.ln1_b, np, dim, cfg.eps);
        try gemm(be, proj_d, normed_d, np, blk.q, blk.q_b);
        try be.opHeadPad(q_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        try gemm(be, proj_d, normed_d, np, blk.k, blk.k_b);
        try be.opHeadPad(k_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        try gemm(be, proj_d, normed_d, np, blk.v, blk.v_b);
        try be.opHeadPad(v_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        try be.opAttnTC(q_d, k_d, v_d, big_d, np, heads, heads, hd_pad, scale);
        try be.opHeadPad(normed_d, big_d, np, heads, hd, hd_pad, heads * hd_pad, 0);
        try gemm(be, t_d, normed_d, np, blk.out, blk.out_b);
        try be.opAdd(x_d, t_d, np * dim);

        try be.opLayerNorm(x_d, normed_d, blk.ln2_w, blk.ln2_b, np, dim, cfg.eps);
        try gemm(be, big_d, normed_d, np, blk.up, blk.up_b);
        try be.gelu(big_d, np * cfg.ffn);
        try gemm(be, t_d, big_d, np, blk.down, blk.down_b);
        try be.opAdd(x_d, t_d, np * dim);
    }
    try be.opLayerNorm(x_d, x_d, vit.post_ln_w, vit.post_ln_b, np, dim, cfg.eps);
    try be.endBatch();

    // Download the post-LN patch states; the projector (pool + soft_emb_norm
    // + input projection) runs on the host — 256 tokens, cheap.
    const x_host = try gpa.alloc(f32, np * dim);
    defer gpa.free(x_host);
    try be.tensorDownload(sized(x_d, 0, np * dim * 4), std.mem.sliceAsBytes(x_host));
    return vit.project(io, gpa, x_host);
}

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device + the real Gemma 3 mmproj: the full GPU encode vs
// the CPU reference on a small gradient image. GPU runs f16 tensor-core
// GEMMs, so parity is relative (per-token cosine + global rel RMSE).
test "cuda gemma vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/mmproj-F16.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try @import("tp_core").gguf.Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    var pixels: [128 * 96 * 3]u8 = undefined;
    for (0..96) |y| {
        for (0..128) |x| {
            const i = (y * 128 + x) * 3;
            pixels[i] = @intCast((x * 2) & 0xff);
            pixels[i + 1] = @intCast((y * 2) & 0xff);
            pixels[i + 2] = 100;
        }
    }

    var want = try vit.encode(io, gpa, &pixels, 128, 96);
    defer want.deinit(gpa);
    var got = try encode(&vit, be, io, gpa, &pixels, 128, 96);
    defer got.deinit(gpa);

    try std.testing.expectEqual(want.embeds.len, got.embeds.len);
    const pd = vit.cfg.proj_dim;
    var num: f64 = 0;
    var den: f64 = 0;
    var min_cos: f64 = 1;
    var t: usize = 0;
    while (t * pd < want.embeds.len) : (t += 1) {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (want.embeds[t * pd ..][0..pd], got.embeds[t * pd ..][0..pd]) |a, b| {
            dot += @as(f64, a) * b;
            na += @as(f64, a) * a;
            nb += @as(f64, b) * b;
            num += (@as(f64, a) - b) * (@as(f64, a) - b);
            den += @as(f64, a) * a;
        }
        min_cos = @min(min_cos, dot / (@sqrt(na) * @sqrt(nb)));
    }
    std.debug.print("cuda gemma vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, @sqrt(num / den) });
    try std.testing.expect(min_cos > 0.999);
    try std.testing.expect(@sqrt(num / den) < 0.05);
}
