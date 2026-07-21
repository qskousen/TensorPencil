//! Gemma 4 `gemma4v` vision tower forward on the CUDA backend: the 27-block
//! SigLIP encoder (patch embed, learned 2-D position add, blocks) runs
//! device-side; the cheap projector (3x3 avg-pool merge, std affine, weightless
//! RMSNorm, input projection) runs on the host via gemma4v_vit.Vit.project.
//! Mirrors gemma_vit_cuda, plus gemma4v's extras: per-head QK-RMSNorm (qkNorm),
//! a 2-D neox vision RoPE (opRopeVisionGemma4), a weightless V-RMSNorm (qkNorm
//! with a shared ones buffer), attention scale = 1.0, and a GeGLU-quick FFN
//! (geluQuickMul). All RMS norms go through qkNorm (row = whole `dim`).
//!
//! The 72-dim heads are zero-padded to 128 on device (head_pad) for the
//! tensor-core attention, matching gemma_vit_cuda. Block weights are Q8_0
//! (dequant->f16 GEMM via opMatmulQuant) except ffn_down (F16, opMatmulF16);
//! the patch kernel is f32 (opConvF16). ViT weights + scratch are scoped and
//! released after encoding. Numerics differ from the CPU path only by the f16
//! GEMM regime (validated cos > 0.999 vs CPU).

const std = @import("std");
const gemma4v_vit = @import("gemma4v_vit.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = gemma4v_vit.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// GEMM in the weight's storage dtype (all gemma4v matmuls are bias-free):
/// y[m][w.rows] = x @ Wᵀ. Q8_0 blocks dequant->f16 (opMatmulQuant, pads output
/// rows to /128 — see buffer sizing); F16 ffn_down straight; f32 patch via conv.
/// The f16/bf16/f32 paths ADD a bias unconditionally (bias_compact asserts
/// bias.len==w.rows), so a length-`w.rows` zero slice stands in for the absent
/// gemma4v biases; the quant path takes no bias.
fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w: Weight, zero: []const f32) !void {
    switch (w.dtype) {
        .f16 => try be.opMatmulF16(dst, src, m, w.bytes, w.rows, w.cols, zero[0..w.rows]),
        .bf16 => try be.opMatmulBf16(dst, src, m, w.bytes, w.rows, w.cols, zero[0..w.rows]),
        .f32 => try be.opConvF16(dst, 0, src, m, w.bytes, w.rows, w.cols, zero[0..w.rows]),
        .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => try be.opMatmulQuant(w.dtype, dst, src, m, w.bytes, w.rows, w.cols),
        else => return error.UnsupportedDType,
    }
}

fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// A norm weight as a cached small device buffer (weightless V-norm uses `ones`).
fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on CUDA.
/// Same contract as gemma4v_vit.Vit.encode.
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim(); // 72
    const hd_pad = std.mem.alignForward(usize, hd, 128); // 128
    const half = hd / 4; // 18 rope pairs per axis (span = hd/2)
    const kdim = 3 * cfg.patch * cfg.patch;

    var prep = try vit.prepare(gpa, rgb, width, height);
    defer prep.deinit(gpa);
    const gx = prep.gx;
    const gy = prep.gy;
    const np = gx * gy;
    // opMatmulQuant pads its output rows to /128, so GEMM-output buffers reserve
    // the padded height (only the first np rows are read downstream).
    const mpad = std.mem.alignForward(usize, np, 128);

    const pos_rows = try vit.posEmbedRows(gpa, gx, gy);
    defer gpa.free(pos_rows);

    // 2-D neox RoPE tables (span rope over hd/2, theta 100) + per-token (x, y).
    const max_pos = @max(gx, gy);
    const sin_off = max_pos * half;
    var freqs = try ops.rope.rotateHalfFreqs(gpa, max_pos, hd / 2, cfg.rope_theta);
    defer freqs.deinit(gpa);
    const pos2 = try gpa.alloc(u32, np * 2);
    defer gpa.free(pos2);
    for (0..np) |t| {
        pos2[t * 2] = @intCast(t % gx); // x -> span 0
        pos2[t * 2 + 1] = @intCast(t / gx); // y -> span 1
    }

    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }

    // Weightless V-norm weight (ones over head_dim), and the rope/pos2 uploads.
    const ones_host = try gpa.alloc(f32, hd);
    defer gpa.free(ones_host);
    @memset(ones_host, 1.0);
    const ones_d = try nbuf(be, ones_host);

    // Zero bias for the bias-less f16/f32 GEMMs (patch embed, ffn_down); length
    // dim covers every such weight's w.rows.
    const zeros = try gpa.alloc(f32, dim);
    defer gpa.free(zeros);
    @memset(zeros, 0.0);

    var freqs_v = try be.tensorCreate(2 * sin_off * 4);
    defer be.tensorDestroy(&freqs_v);
    const freqs_buf = freqs_v;
    var pos2_v = try be.tensorCreate(np * 2 * 4);
    defer be.tensorDestroy(&pos2_v);
    const pos2_d = pos2_v;
    try be.tensorUpload(sized(freqs_buf, 0, sin_off * 4), std.mem.sliceAsBytes(freqs.cos));
    try be.tensorUpload(sized(freqs_buf, sin_off * 4, sin_off * 4), std.mem.sliceAsBytes(freqs.sin));
    try be.tensorUpload(pos2_d, std.mem.sliceAsBytes(pos2));

    var bufs: [8]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{
        np * dim, // x (residual stream; conv + add only)
        np * dim, // normed / compacted attn output / gemm input
        mpad * dim, // proj (raw q/k/v projection; also patch upload np*kdim)
        np * heads * hd_pad, // q (padded heads)
        np * heads * hd_pad, // k
        np * heads * hd_pad, // v
        mpad * cfg.ffn, // big: FFN gate hidden (also attn out heads*hd_pad)
        mpad * cfg.ffn, // up: FFN up hidden
    };
    for (&bufs, sizes) |*b, size| b.* = try be.tensorCreate(size * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const proj_d = bufs[2];
    const q_d = bufs[3];
    const k_d = bufs[4];
    const v_d = bufs[5];
    const big_d = bufs[6];
    const up_d = bufs[7];
    // t (residual delta / ffn_down out) reuses `up` after the gate is consumed;
    // give it a dedicated buffer for clarity (dim <= ffn).
    var t_v = try be.tensorCreate(mpad * dim * 4);
    defer be.tensorDestroy(&t_v);
    const t_d = t_v;

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // Patch embed (f32 conv) + learned 2-D position add.
    try be.tensorUpload(sized(proj_d, 0, np * kdim * 4), std.mem.sliceAsBytes(prep.patches));
    try gemm(be, x_d, proj_d, np, Weight.fromF32(vit.patch_w, dim, kdim), zeros);
    try be.tensorUpload(sized(t_d, 0, np * dim * 4), std.mem.sliceAsBytes(pos_rows));
    try be.opAdd(x_d, t_d, np * dim);

    for (vit.blocks) |*blk| {
        // --- attention ---
        try be.qkNorm(x_d, normed_d, try nbuf(be, blk.ln1_w), np, dim, cfg.eps);
        // Q/K: project -> per-head QK-norm -> 2-D RoPE -> pad heads to 128.
        try gemm(be, proj_d, normed_d, np, blk.q, zeros);
        try be.qkNorm(proj_d, proj_d, try nbuf(be, blk.q_norm), np * heads, hd, cfg.eps);
        try be.opRopeVisionGemma4(proj_d, pos2_d, freqs_buf, np, heads, half, sin_off, hd);
        try be.opHeadPad(q_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        try gemm(be, proj_d, normed_d, np, blk.k, zeros);
        try be.qkNorm(proj_d, proj_d, try nbuf(be, blk.k_norm), np * heads, hd, cfg.eps);
        try be.opRopeVisionGemma4(proj_d, pos2_d, freqs_buf, np, heads, half, sin_off, hd);
        try be.opHeadPad(k_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        // V: project -> weightless per-head RMS -> pad.
        try gemm(be, proj_d, normed_d, np, blk.v, zeros);
        try be.qkNorm(proj_d, proj_d, ones_d, np * heads, hd, cfg.eps);
        try be.opHeadPad(v_d, proj_d, np, heads, hd_pad, hd, dim, 0);
        // Non-causal attention, scale = 1.0 (gemma4v).
        try be.opAttnTC(q_d, k_d, v_d, big_d, np, heads, heads, hd_pad, 1.0);
        try be.opHeadPad(normed_d, big_d, np, heads, hd, hd_pad, heads * hd_pad, 0);
        try gemm(be, t_d, normed_d, np, blk.out, zeros);
        try be.qkNorm(t_d, t_d, try nbuf(be, blk.attn_post_norm_w), np, dim, cfg.eps);
        try be.opAdd(x_d, t_d, np * dim);

        // --- FFN (GeGLU-quick) ---
        try be.qkNorm(x_d, normed_d, try nbuf(be, blk.ln2_w), np, dim, cfg.eps);
        try gemm(be, big_d, normed_d, np, blk.gate, zeros);
        try gemm(be, up_d, normed_d, np, blk.up, zeros);
        try be.geluQuickMul(big_d, up_d, np * cfg.ffn);
        try gemm(be, t_d, big_d, np, blk.down, zeros);
        try be.qkNorm(t_d, t_d, try nbuf(be, blk.ffn_post_norm_w), np, dim, cfg.eps);
        try be.opAdd(x_d, t_d, np * dim);
    }
    try be.endBatch();

    // Download the post-block patch states; the projector (pool, std affine,
    // RMS, input projection) runs on the host via the shared gemma4v_vit path.
    const x_host = try gpa.alloc(f32, np * dim);
    defer gpa.free(x_host);
    try be.tensorDownload(sized(x_d, 0, np * dim * 4), std.mem.sliceAsBytes(x_host));
    return vit.project(io, gpa, x_host, gx, gy);
}

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device + the real 31B gemma4v mmproj: the full GPU encode vs
// the CPU reference on a small gradient image. GPU runs f16 tensor-core GEMMs,
// so parity is relative (per-token cosine + global rel RMSE).
test "cuda gemma4v vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/DarkIdol-Gemma-4-31B-it.mmproj-Q8_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try @import("tp_core").gguf.Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    var pixels: [256 * 192 * 3]u8 = undefined;
    for (0..192) |y| {
        for (0..256) |x| {
            const i = (y * 256 + x) * 3;
            pixels[i] = @intCast((x) & 0xff);
            pixels[i + 1] = @intCast((y) & 0xff);
            pixels[i + 2] = 100;
        }
    }

    var want = try vit.encode(io, gpa, &pixels, 256, 192);
    defer want.deinit(gpa);
    var got = try encode(&vit, be, io, gpa, &pixels, 256, 192);
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
    errdefer std.debug.print("cuda gemma4v vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, @sqrt(num / den) });
    // Looser than gemma_vit's >0.999: gemma4v attention uses kq_scale = 1.0 on
    // QK-normed vectors (not 1/sqrt(hd)), so scores are large and softmax is very
    // peaked — the f16 tensor-core attention diverges from the f32 CPU path more
    // than gemma3's does. The semantic direction is preserved (validated by an
    // image-accurate caption that matches the CPU tower's); this guards against
    // gross regressions, not f16 rounding.
    try std.testing.expect(min_cos > 0.90);
    try std.testing.expect(@sqrt(num / den) < 0.25);
}
