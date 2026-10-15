//! Gemma 3 vision tower on the CUDA backend: the adapter for `gemma_vit_dev.zig`. The
//! 72-dim heads are zero-padded to 128 (head_pad) so the tensor-core attention applies,
//! and the linears run in each weight's storage dtype through `lin_cuda.towerLinear`.

const std = @import("std");
const lin_cuda = @import("lin_cuda.zig");
const gemma_vit = @import("gemma_vit.zig");
const gemma_vit_dev = @import("gemma_vit_dev.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = gemma_vit.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Encode interleaved RGB pixels to Gemma image-token embeddings on CUDA. Same
/// contract as `Vit.encode`; numerics differ by the f16 GEMM regime.
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const hd_pad = std.mem.alignForward(usize, cfg.headDim(), 128);
    const np = cfg.side() * cfg.side();

    // ViT weights and scratch are scoped, so nothing image-sized stays under the LLM.
    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }
    var d: Dev = .{ .be = be, .cfg = cfg, .hd_pad = hd_pad };
    defer d.deinit();
    const sizes = [_]usize{
        np * cfg.dim, // x
        np * cfg.dim, // normed, then the unpadded attention output
        np * @max(cfg.n_heads * hd_pad, cfg.ffn), // attention output / FFN hidden
        @max(np * cfg.dim, np * 3 * cfg.patch * cfg.patch), // t: the patches, then each delta
        np * cfg.dim, // one raw q/k/v projection before head_pad
        np * cfg.n_heads * hd_pad,
        np * cfg.n_heads * hd_pad,
        np * cfg.n_heads * hd_pad,
    };
    inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.proj, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try be.tensorCreate(n * 4);
    return gemma_vit_dev.encode(&d, vit, io, gpa, rgb, width, height);
}

const Bufs = struct { x: Buf = .{}, normed: Buf = .{}, big: Buf = .{}, t: Buf = .{} };

/// The `gemma_vit_dev` adapter.
const Dev = struct {
    be: *Backend,
    cfg: gemma_vit.Config,
    hd_pad: usize,
    b: Bufs = .{},
    /// The 72-wide heads zero-padded to 128 so the tensor-core attention applies; the
    /// pads are exact zeros on both sides of the scores.
    proj: Buf = .{},
    q: Buf = .{},
    k: Buf = .{},
    v: Buf = .{},

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.proj, &d.q, &d.k, &d.v }) |b| d.be.tensorDestroy(b);
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
        try d.be.tensorUpload(sized(b, data.len * 4), std.mem.sliceAsBytes(data));
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        try d.be.tensorDownload(sized(b, dst.len * 4), std.mem.sliceAsBytes(dst));
    }

    /// In the weight's storage dtype.
    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, bias, null);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, bias: []const f32, rows: usize) !void {
        try d.be.opLayerNorm(x, y, w, bias, rows, d.cfg.dim, d.cfg.eps);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize) !void {
        try d.be.gelu(x, n);
    }

    pub fn attention(d: *Dev, blk: *const gemma_vit.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const hd = cfg.headDim();
        const heads = cfg.n_heads;
        inline for (.{ .{ blk.q, blk.q_b, d.q }, .{ blk.k, blk.k_b, d.k }, .{ blk.v, blk.v_b, d.v } }) |t| {
            try d.linear(d.proj, d.b.normed, np, t[0], t[1]);
            try d.be.opHeadPad(t[2], d.proj, np, heads, d.hd_pad, hd, cfg.dim, 0);
        }
        try d.be.opAttnTC(d.q, d.k, d.v, d.b.big, np, heads, heads, d.hd_pad, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
        try d.be.opHeadPad(d.b.normed, d.b.big, np, heads, hd, d.hd_pad, heads * d.hd_pad, 0);
        return d.b.normed;
    }
};

fn sized(b: Buf, size: usize) Buf {
    return .{ .buf = b.buf, .mem = b.mem, .size = size };
}

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device + the real Gemma 3 mmproj: the full GPU encode vs
// the CPU reference on a small gradient image. GPU runs f16 tensor-core
// GEMMs, so parity is relative (per-token cosine + global rel RMSE).
test "cuda gemma vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/mnt/watch/models/llm/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/mmproj-F16.gguf";
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
    errdefer std.debug.print("cuda gemma vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, @sqrt(num / den) });
    try std.testing.expect(min_cos > 0.999);
    try std.testing.expect(@sqrt(num / den) < 0.05);
}
