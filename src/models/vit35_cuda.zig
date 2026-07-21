//! CUDA-backend Qwen3.5/3.6 vision tower forward (LLM_PLAN.md "GPU ViT"):
//! the same qwen3vl_merger pipeline as vit35.Vit.encode, with the patch
//! embed, 27 pre-LN blocks, and the merger projector running device-side.
//! Host keeps only the cheap prep (smart resize, normalize, patch ordering,
//! position interpolation, rope tables) via vit35.Vit.prepare.
//!
//! The bf16 mmproj weights feed the f16 tensor-core GEMM straight from the
//! GGUF mmap (opMatmulBf16 pad-converts per call, like the LLM's dequant
//! path). The 72-dim heads are zero-padded to 128 on device (head_pad) so
//! the tensor-core attention's PV GEMM (n = head_dim, 128-multiple) applies;
//! the pads are exact zeros end to end (K pads don't perturb scores, V pads
//! produce zero output dims that the compaction drops), so the math matches
//! the CPU path up to the f16 GEMM regime.
//!
//! The ViT's weights are cached under a weight scope and released after
//! encoding, along with the attention/GEMM scratch
//! (weightScopeEnd/freeAttnScratch/freeConvScratch): image-sized state must
//! not stay resident under the 19 GB LLM, the cache must not hold pointers
//! into a Vit the caller may free, and — for REPL image turns — resident
//! LLM weights and their captured decode graph must survive the encode.

const std = @import("std");
const vit35 = @import("vit35.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = vit35.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// GEMM in the mmproj weight's storage dtype (bf16 tensors, plus the f32
/// host-summed patch kernel), y[m][w.rows] = x @ Wᵀ + bias, tight rows.
fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w: Weight, bias: []const f32) !void {
    switch (w.dtype) {
        .bf16 => try be.opMatmulBf16(dst, src, m, w.bytes, w.rows, w.cols, bias),
        .f16 => try be.opMatmulF16(dst, src, m, w.bytes, w.rows, w.cols, bias),
        .f32 => try be.opConvF16(dst, 0, src, m, w.bytes, w.rows, w.cols, bias),
        else => return error.UnsupportedDType,
    }
}

/// Encode interleaved RGB pixels to LLM embeddings on the CUDA backend.
/// Same contract as Vit.encode; numerics differ by the f16 GEMM regime.
pub fn encode(vit: *const Vit, be: *Backend, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim(); // 72
    const hd_pad = std.mem.alignForward(usize, hd, 128);
    const half = hd / 4; // 18 rope pairs per axis
    const kdim = 3 * cfg.patch * cfg.patch;

    var prep = try vit.prepare(gpa, rgb, width, height);
    defer prep.deinit(gpa);
    const np = prep.np();
    const nm = np / 4;
    const merged_dim = 4 * dim;

    // The ViT leaves nothing resident: its weights are cached under a
    // scope and dropped as a group below, along with the attention scores
    // plane and the GEMM conversion scratch. The scoped release (rather
    // than a full evictWeights) matters for REPL image turns — a resident
    // LLM and its captured decode graph must survive a mid-session encode.
    // It also protects correctness: the cache is keyed by host pointer,
    // and the Vit arena those keys point into may die with the Vit.
    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }

    var bufs: [8]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{
        np * dim, // x (residual stream; [nm][merged_dim] after post_ln)
        np * dim, // normed / compacted attention output
        np * 3 * dim, // fused qkv
        np * heads * hd_pad, // q (padded heads)
        np * heads * hd_pad, // k
        np * heads * hd_pad, // v
        np * @max(heads * hd_pad, cfg.ffn), // attention out / FFN hidden / patch upload
        @max(np * dim, nm * cfg.proj_dim), // t (residual delta; h0+embeds reuse)
    };
    for (&bufs, sizes) |*b, size| b.* = try be.tensorCreate(size * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const qkv_d = bufs[2];
    const q_d = bufs[3];
    const k_d = bufs[4];
    const v_d = bufs[5];
    const big_d = bufs[6];
    const t_d = bufs[7];

    // Rope tables + per-token (row, col) grid positions.
    const max_pos = @max(prep.pw, prep.ph);
    const sin_off = max_pos * half;
    const freqs_d = try be.tensorCreate(2 * sin_off * 4);
    var freqs_v = freqs_d;
    defer be.tensorDestroy(&freqs_v);
    try be.tensorUpload(sized(freqs_d, 0, sin_off * 4), std.mem.sliceAsBytes(prep.rope_cos));
    try be.tensorUpload(sized(freqs_d, sin_off * 4, sin_off * 4), std.mem.sliceAsBytes(prep.rope_sin));
    const pos2 = try gpa.alloc(u32, np * 2);
    defer gpa.free(pos2);
    for (0..np) |t| {
        pos2[t * 2] = prep.py[t];
        pos2[t * 2 + 1] = prep.px[t];
    }
    const pos2_d = try be.tensorCreate(np * 2 * 4);
    var pos2_v = pos2_d;
    defer be.tensorDestroy(&pos2_v);
    try be.tensorUpload(pos2_d, std.mem.sliceAsBytes(pos2));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // Patch embed GEMM (f32 host-summed kernel) + position embedding, the
    // latter reordered host-side into the merged token order.
    try be.tensorUpload(sized(big_d, 0, np * kdim * 4), std.mem.sliceAsBytes(prep.patches));
    try gemm(be, x_d, big_d, np, Weight.fromF32(vit.patch_w, dim, kdim), vit.patch_b);
    {
        const pos_ordered = try gpa.alloc(f32, np * dim);
        defer gpa.free(pos_ordered);
        for (0..np) |t| {
            const src = prep.pos[(prep.py[t] * prep.pw + prep.px[t]) * dim ..][0..dim];
            @memcpy(pos_ordered[t * dim ..][0..dim], src);
        }
        try be.tensorUpload(sized(t_d, 0, np * dim * 4), std.mem.sliceAsBytes(pos_ordered));
        try be.opAdd(x_d, t_d, np * dim);
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (vit.blocks) |*blk| {
        try be.opLayerNorm(x_d, normed_d, blk.ln1_w, blk.ln1_b, np, dim, cfg.eps);
        try gemm(be, qkv_d, normed_d, np, blk.qkv, blk.qkv_b);
        try be.opHeadPad(q_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, 0);
        try be.opHeadPad(k_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, dim);
        try be.opHeadPad(v_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, 2 * dim);
        try be.opRopeVision(q_d, pos2_d, freqs_d, np, heads, half, sin_off, hd_pad);
        try be.opRopeVision(k_d, pos2_d, freqs_d, np, heads, half, sin_off, hd_pad);
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

    // 2x2 merge is a reinterpret (tokens are block-ordered), then the
    // two-layer projector. h0 reuses normed (nm*merged_dim == np*dim).
    try gemm(be, normed_d, x_d, nm, vit.mm0, vit.mm0_b);
    try be.gelu(normed_d, nm * merged_dim);
    try gemm(be, t_d, normed_d, nm, vit.mm2, vit.mm2_b);
    try be.endBatch();

    const embeds = try gpa.alloc(f32, nm * cfg.proj_dim);
    errdefer gpa.free(embeds);
    try be.tensorDownload(sized(t_d, 0, embeds.len * 4), std.mem.sliceAsBytes(embeds));
    return .{ .embeds = embeds, .grid_w = prep.pw / 2, .grid_h = prep.ph / 2 };
}

/// A sized sub-view of a device buffer (raw pointer arithmetic).
fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device AND the real Qwen3.6 mmproj: the full GPU encode
// against the CPU reference on a small gradient image. The GPU runs f16
// tensor-core GEMMs, so parity is relative (same regime as the DiT's
// CPU-vs-CUDA validation), checked as per-token cosine + global rel RMSE.
test "cuda vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/mmproj-Qwen3.6-27B-BF16.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try @import("tp_core").gguf.Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    var pixels: [64 * 64 * 3]u8 = undefined;
    for (0..64) |y| {
        for (0..64) |x| {
            const i = (y * 64 + x) * 3;
            pixels[i] = @intCast(x * 4);
            pixels[i + 1] = @intCast(y * 4);
            pixels[i + 2] = 128;
        }
    }

    var want = try vit.encode(io, gpa, &pixels, 64, 64);
    defer want.deinit(gpa);
    var got = try encode(&vit, be, gpa, &pixels, 64, 64);
    defer got.deinit(gpa);

    try std.testing.expectEqual(want.grid_w, got.grid_w);
    try std.testing.expectEqual(want.grid_h, got.grid_h);
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
    const rel = @sqrt(num / den);
    std.debug.print("cuda vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, rel });
    try std.testing.expect(min_cos > 0.999);
    try std.testing.expect(rel < 0.05);
}
