//! The Gemma 4 `gemma4v` vision tower on CUDA: the adapter for `gemma4v_vit_dev.zig`.
//! The 72-dim heads are zero-padded to 128 (head_pad) for the tensor-core attention;
//! the pads are exact zeros on both sides of the scores. ViT weights and scratch are
//! scoped and released after encoding. Numerics differ from the CPU path only by the
//! f16 GEMM regime, which compounds through the 27 blocks (see the test).

const std = @import("std");
const lin_cuda = @import("lin_cuda.zig");
const gemma4v_vit = @import("gemma4v_vit.zig");
const gemma4v_vit_dev = @import("gemma4v_vit_dev.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = gemma4v_vit.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on CUDA.
/// Same contract as gemma4v_vit.Vit.encode.
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }
    var d: Dev = .{ .be = be, .cfg = vit.cfg };
    defer d.deinit();
    return gemma4v_vit_dev.encode(&d, vit, io, gpa, rgb, width, height);
}

const Bufs = struct { x: Buf = .{}, normed: Buf = .{}, proj: Buf = .{}, big: Buf = .{}, up: Buf = .{}, t: Buf = .{} };

/// The `gemma4v_vit_dev` adapter.
const Dev = struct {
    be: *Backend,
    cfg: gemma4v_vit.Config,
    b: Bufs = .{},
    q: Buf = .{},
    k: Buf = .{},
    v: Buf = .{},
    ones: Buf = .{},
    freqs: Buf = .{},
    pos2: Buf = .{},
    sin_off: usize = 0,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.proj, &d.b.big, &d.b.up, &d.b.t, &d.q, &d.k, &d.v, &d.ones, &d.freqs, &d.pos2 }) |b| d.be.tensorDestroy(b);
    }

    fn hdPad(d: *const Dev) usize {
        return std.mem.alignForward(usize, d.cfg.headDim(), 128);
    }

    /// The buffers for a `gx` x `gy` grid, the rope tables and the per-token positions.
    /// GEMM outputs reserve rows to a multiple of 128 (opMatmulQuant pads its output).
    pub fn setup(d: *Dev, gpa: std.mem.Allocator, gx: usize, gy: usize) !void {
        const cfg = d.cfg;
        const be = d.be;
        const np = gx * gy;
        const mpad = std.mem.alignForward(usize, np, 128);
        const hp = cfg.n_heads * d.hdPad();
        const hd = cfg.headDim();
        var rope = try gemma4v_vit_dev.Rope.init(gpa, cfg, gx, gy);
        defer rope.deinit(gpa);
        const ones = try gpa.alloc(f32, hd);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        d.ones = try be.tensorCreate(hd * 4);
        try be.tensorUpload(d.ones, std.mem.sliceAsBytes(ones));
        d.sin_off = rope.sin_off;
        d.freqs = try be.tensorCreate(rope.table.len * 4);
        d.pos2 = try be.tensorCreate(rope.pos2.len * 4);
        try be.tensorUpload(sized(d.freqs, 0, rope.sin_off * 4), std.mem.sliceAsBytes(rope.table[0..rope.sin_off]));
        try be.tensorUpload(sized(d.freqs, rope.sin_off * 4, rope.sin_off * 4), std.mem.sliceAsBytes(rope.table[rope.sin_off..]));
        try be.tensorUpload(d.pos2, std.mem.sliceAsBytes(rope.pos2));
        const sizes = [_]usize{ np * cfg.dim, np * cfg.dim, mpad * cfg.dim, mpad * cfg.ffn, mpad * cfg.ffn, mpad * cfg.dim, np * hp, np * hp, np * hp };
        inline for (.{ &d.b.x, &d.b.normed, &d.b.proj, &d.b.big, &d.b.up, &d.b.t, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try be.tensorCreate(n * 4);
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

    /// In the weight's storage dtype; every gemma4v linear is bias-free.
    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, null, null);
    }

    pub fn rms(d: *Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.be.qkNorm(x, y, try nbuf(d.be, w), rows, dim, d.cfg.eps);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn geluQuickMul(d: *Dev, g: Buf, u: Buf, n: usize) !void {
        try d.be.geluQuickMul(g, u, n);
    }

    /// Each of q and k projected, per-head normed, roped and padded to 128; v projected,
    /// unit-normed and padded; the tensor-core attention at scale 1.0, compacted back to
    /// 72-wide heads in `normed`.
    pub fn attention(d: *Dev, blk: *const gemma4v_vit.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const be = d.be;
        const dim = cfg.dim;
        const heads = cfg.n_heads;
        const hd = cfg.headDim();
        const hd_pad = d.hdPad();
        const half = hd / 4;
        const b = d.b;
        for ([2]*const Weight{ &blk.q, &blk.k }, [2][]const f32{ blk.q_norm, blk.k_norm }, [2]Buf{ d.q, d.k }) |w, nw, dst| {
            try d.linear(b.proj, b.normed, np, w.*);
            try be.qkNorm(b.proj, b.proj, try nbuf(be, nw), np * heads, hd, cfg.eps);
            try be.opRopeVisionGemma4(b.proj, d.pos2, d.freqs, np, heads, half, d.sin_off, hd);
            try be.opHeadPad(dst, b.proj, np, heads, hd_pad, hd, dim, 0);
        }
        try d.linear(b.proj, b.normed, np, blk.v);
        try be.qkNorm(b.proj, b.proj, d.ones, np * heads, hd, cfg.eps);
        try be.opHeadPad(d.v, b.proj, np, heads, hd_pad, hd, dim, 0);
        try be.opAttnTC(d.q, d.k, d.v, b.big, np, heads, heads, hd_pad, 1.0);
        try be.opHeadPad(b.normed, b.big, np, heads, hd, hd_pad, heads * hd_pad, 0);
        return b.normed;
    }
};

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
    // Much looser than gemma_vit's >0.999, and the reason is depth, not attention.
    // Measured by truncating both towers to N blocks and comparing at N:
    //   0 blocks (patch embed + pos): cos 1.000000, rel RMSE 0.000016
    //   1: 0.999997 / 0.0014   2: 0.999947 / 0.0044   4: 0.999928 / 0.0055
    //   8: 0.999806 / 0.0125  16: 0.999255 / 0.0293  24: 0.955371 / 0.0971
    //   27 (all): 0.837371 / 0.163917
    // Every op is fine in isolation and the GEMM path into block 0 is exact; the
    // f16 operand rounding just compounds through 27 residual blocks with a
    // per-block gain above 1. An earlier note here blamed the large scores from
    // kq_scale = 1.0 on QK-normed vectors: that was tested by running BOTH towers
    // at 1/sqrt(hd) instead, which left cos at 0.877 and made rel RMSE worse
    // (0.28), so peaked softmax is not what drives this.
    //
    // These bounds are the measured floor plus margin, not an accuracy target.
    // They have been stable to ~0.002 since the test was written. Tightening them
    // means making the GPU tower more accurate (f32 scores, or f32 operands),
    // not adjusting the number.
    try std.testing.expect(min_cos > 0.80);
    try std.testing.expect(@sqrt(num / den) < 0.20);
}
