//! The Qwen3.5/3.6 vision tower on CUDA: the adapter for `vit35_dev.zig`.
//!
//! The bf16 mmproj weights feed the f16 tensor-core GEMM straight from the GGUF mmap.
//! The 72-dim heads are zero-padded to 128 (head_pad) so the tensor-core attention's
//! PV GEMM applies; the pads are exact zeros end to end (K pads don't perturb scores, V
//! pads produce zero output dims that the compaction drops).
//!
//! The ViT's weights are cached under a weight scope and released after encoding, along
//! with the attention/GEMM scratch: image-sized state must not stay resident under the
//! LLM, the cache must not hold pointers into a Vit the caller may free, and a resident
//! LLM's weights and captured decode graph must survive the encode.

const std = @import("std");
const lin_cuda = @import("lin_cuda.zig");
const vit35 = @import("vit35.zig");
const vit35_dev = @import("vit35_dev.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = vit35.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Encode interleaved RGB pixels to LLM embeddings on the CUDA backend.
/// Same contract as Vit.encode; numerics differ by the f16 GEMM regime.
pub fn encode(vit: *const Vit, be: *Backend, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }
    var d: Dev = .{ .be = be, .cfg = vit.cfg };
    defer d.deinit();
    return vit35_dev.encode(&d, vit, gpa, rgb, width, height);
}

const Bufs = struct { x: Buf = .{}, normed: Buf = .{}, big: Buf = .{}, t: Buf = .{} };

/// The `vit35_dev` adapter.
const Dev = struct {
    be: *Backend,
    cfg: vit35.Config,
    b: Bufs = .{},
    qkv: Buf = .{},
    q: Buf = .{},
    k: Buf = .{},
    v: Buf = .{},
    freqs: Buf = .{},
    pos2: Buf = .{},
    sin_off: usize = 0,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.qkv, &d.q, &d.k, &d.v, &d.freqs, &d.pos2 }) |b| d.be.tensorDestroy(b);
    }

    fn hdPad(d: *const Dev) usize {
        return std.mem.alignForward(usize, d.cfg.headDim(), 128);
    }

    /// The buffers for `prep`'s token count, the rope tables and the per-token (row,
    /// column) grid positions.
    pub fn setup(d: *Dev, gpa: std.mem.Allocator, prep: *const Vit.Prepared) !void {
        const cfg = d.cfg;
        const dim = cfg.dim;
        const np = prep.np();
        const nm = np / 4;
        const hp = cfg.n_heads * d.hdPad();
        const sizes = [_]usize{
            np * dim, // x, [nm][merged_dim] after post_ln
            np * dim, // normed, the compacted attention output, h0
            np * @max(hp, cfg.ffn), // attention out, FFN hidden, the patch upload
            @max(np * dim, nm * cfg.proj_dim), // t: each delta, the positions, the embeds
            np * 3 * dim,
            np * hp,
            np * hp,
            np * hp,
        };
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.qkv, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try d.be.tensorCreate(n * 4);

        const half = cfg.headDim() / 4;
        d.sin_off = @max(prep.pw, prep.ph) * half;
        d.freqs = try d.be.tensorCreate(2 * d.sin_off * 4);
        try d.be.tensorUpload(sized(d.freqs, 0, d.sin_off * 4), std.mem.sliceAsBytes(prep.rope_cos));
        try d.be.tensorUpload(sized(d.freqs, d.sin_off * 4, d.sin_off * 4), std.mem.sliceAsBytes(prep.rope_sin));
        const pos2 = try gpa.alloc(u32, np * 2);
        defer gpa.free(pos2);
        for (0..np) |t| {
            pos2[t * 2] = prep.py[t];
            pos2[t * 2 + 1] = prep.px[t];
        }
        d.pos2 = try d.be.tensorCreate(np * 2 * 4);
        try d.be.tensorUpload(d.pos2, std.mem.sliceAsBytes(pos2));
    }

    pub fn beginBatch(d: *Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: *Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: *Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn upload(d: *Dev, b: Buf, data: []const f32) !void {
        try d.be.tensorUpload(sized(b, 0, data.len * 4), std.mem.sliceAsBytes(data));
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        try d.be.tensorDownload(sized(b, 0, dst.len * 4), std.mem.sliceAsBytes(dst));
    }

    /// In the weight's storage dtype (bf16 tensors, plus the f32 host-summed patch kernel).
    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: []const f32) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, bias, null);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize) !void {
        try d.be.opLayerNorm(x, y, w, b, rows, d.cfg.dim, d.cfg.eps);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize) !void {
        try d.be.gelu(x, n);
    }

    /// The fused qkv, its heads padded to 128, the vision rope, the tensor-core
    /// attention, and the output compacted back to 72-wide heads in `normed`.
    pub fn attention(d: *Dev, blk: *const vit35.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const be = d.be;
        const dim = cfg.dim;
        const heads = cfg.n_heads;
        const hd = cfg.headDim();
        const hd_pad = d.hdPad();
        const half = hd / 4;
        try d.linear(d.qkv, d.b.normed, np, blk.qkv, blk.qkv_b);
        try be.opHeadPad(d.q, d.qkv, np, heads, hd_pad, hd, 3 * dim, 0);
        try be.opHeadPad(d.k, d.qkv, np, heads, hd_pad, hd, 3 * dim, dim);
        try be.opHeadPad(d.v, d.qkv, np, heads, hd_pad, hd, 3 * dim, 2 * dim);
        try be.opRopeVision(d.q, d.pos2, d.freqs, np, heads, half, d.sin_off, hd_pad);
        try be.opRopeVision(d.k, d.pos2, d.freqs, np, heads, half, d.sin_off, hd_pad);
        try be.opAttnTC(d.q, d.k, d.v, d.b.big, np, heads, heads, hd_pad, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
        try be.opHeadPad(d.b.normed, d.b.big, np, heads, hd, hd_pad, heads * hd_pad, 0);
        return d.b.normed;
    }
};

/// A sized sub-view of a device buffer (raw pointer arithmetic).
fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @fromBackingInt(@intCast(@backingInt(b.buf) + off_bytes)), .mem = b.mem, .size = size };
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
    errdefer std.debug.print("cuda vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, rel });
    try std.testing.expect(min_cos > 0.999);
    try std.testing.expect(rel < 0.05);
}
