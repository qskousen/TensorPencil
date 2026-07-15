//! Gemma 4 "unified" vision embedder on the CUDA backend. The embedder is
//! shallow (no transformer — see gemma4_vit.zig), so the whole thing runs
//! device-side as a short op chain: upload the im2col patch matrix, then
//! LayerNorm → patch-embed GEMM → LayerNorm → +pos-embed → LayerNorm →
//! weightless RMSNorm → projection GEMM, and download the embeddings. The
//! smart-resize / normalize / im2col preprocessing and the positional-row
//! gather stay on the host (cheap) via gemma4_vit's shared helpers.
//!
//! Weights and GEMM/conv scratch are scoped and released after encoding, so
//! nothing image-sized stays resident under the LLM. Numerics differ from the
//! CPU path only by the f16 GEMM regime (patch-embed f32 weight via opConvF16,
//! projection bf16 via opMatmulBf16).

const std = @import("std");
const gemma4_vit = @import("gemma4_vit.zig");
const cuda = @import("../gpu/cuda.zig");
const ops = @import("../ops.zig");

const Vit = gemma4_vit.Vit;
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

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on CUDA.
/// Same contract as gemma4_vit.Vit.encode.
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    _ = io;
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const kdim = cfg.kdim();

    var pm = try vit.patchMatrix(gpa, rgb, width, height);
    defer pm.deinit(gpa);
    const np = pm.np();
    const pos = try vit.posEmbedRows(gpa, pm.n_cols, pm.n_rows);
    defer gpa.free(pos);

    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeConvScratch();
    }

    // patch_d holds the [np][kdim] patches, later reused for the [np][proj_dim]
    // projection output (kdim=6912 >= proj_dim=3840). x_d is the [np][dim]
    // working stream; t_d carries the uploaded positional rows.
    var bufs: [3]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{ np * kdim, np * dim, np * dim };
    for (&bufs, sizes) |*b, size| b.* = try be.tensorCreate(size * 4);
    const patch_d = bufs[0];
    const x_d = bufs[1];
    const t_d = bufs[2];

    // Ones vector for the weightless RMSNorm (qkNorm with a unit weight).
    const ones = try gpa.alloc(f32, dim);
    defer gpa.free(ones);
    @memset(ones, 1.0);
    const ones_d: Buf = .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(ones)), .mem = .null_handle, .size = 0 };
    // opMatmulBf16 requires a bias of length co; mm.input_projection has none,
    // so feed a zero bias (a no-op add).
    const zbias = try gpa.alloc(f32, cfg.proj_dim);
    defer gpa.free(zbias);
    @memset(zbias, 0.0);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    try be.tensorUpload(sized(patch_d, 0, np * kdim * 4), std.mem.sliceAsBytes(pm.data));
    try be.opLayerNorm(patch_d, patch_d, vit.patch_norm_1_w, vit.patch_norm_1_b, np, kdim, cfg.eps_ln);
    try gemm(be, x_d, patch_d, np, vit.patch_w, vit.patch_b); // patch embed (6912 -> 3840)
    try be.opLayerNorm(x_d, x_d, vit.patch_norm_2_w, vit.patch_norm_2_b, np, dim, cfg.eps_ln);
    try be.tensorUpload(sized(t_d, 0, np * dim * 4), std.mem.sliceAsBytes(pos));
    try be.opAdd(x_d, t_d, np * dim); // + learned positional embedding
    try be.opLayerNorm(x_d, x_d, vit.patch_norm_3_w, vit.patch_norm_3_b, np, dim, cfg.eps_ln);
    try be.qkNorm(x_d, x_d, ones_d, np, dim, cfg.eps_rms); // weightless RMSNorm
    try gemm(be, patch_d, x_d, np, vit.mm_proj, zbias); // projection (3840 -> 3840)
    try be.endBatch();

    const embeds = try gpa.alloc(f32, np * cfg.proj_dim);
    errdefer gpa.free(embeds);
    try be.tensorDownload(sized(patch_d, 0, np * cfg.proj_dim * 4), std.mem.sliceAsBytes(embeds));
    return .{ .embeds = embeds, .grid_w = pm.n_cols, .grid_h = pm.n_rows };
}

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device + the real gemma4 mmproj: the GPU encode vs the CPU
// reference on a small gradient image (GPU runs f16-regime GEMMs, so parity is
// relative — per-token cosine).
test "cuda gemma4 vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try @import("../gguf.zig").Gguf.open(gpa, io, path);
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
        }
        min_cos = @min(min_cos, dot / (@sqrt(na) * @sqrt(nb)));
    }
    std.debug.print("cuda gemma4 vit parity: min token cos {d:.6}\n", .{min_cos});
    try std.testing.expect(min_cos > 0.999);
}
