//! The Gemma 4 "unified" vision embedder on CUDA: the adapter for `gemma4_vit_dev.zig`.
//! Weights and GEMM/conv scratch are scoped and released after encoding, so nothing
//! image-sized stays resident under the LLM. Numerics differ from the CPU path only by
//! the f16 GEMM regime.

const std = @import("std");
const lin_cuda = @import("lin_cuda.zig");
const gemma4_vit = @import("gemma4_vit.zig");
const gemma4_vit_dev = @import("gemma4_vit_dev.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = gemma4_vit.Vit;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on CUDA.
/// Same contract as gemma4_vit.Vit.encode.
pub fn encode(vit: *const Vit, be: *Backend, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    _ = io;
    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeConvScratch();
    }
    var d: Dev = .{ .be = be, .gpa = gpa, .cfg = vit.cfg };
    defer d.deinit();
    return gemma4_vit_dev.encode(&d, vit, gpa, rgb, width, height);
}

const Bufs = struct { big: Buf = .{}, x: Buf = .{}, t: Buf = .{} };

/// The `gemma4_vit_dev` adapter.
const Dev = struct {
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: gemma4_vit.Config,
    b: Bufs = .{},
    /// The weightless RMSNorm's weight.
    ones: Buf = .{},
    /// A zero bias for the projection, which has none.
    zbias: []f32 = &.{},

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.big, &d.b.x, &d.b.t, &d.ones }) |b| d.be.tensorDestroy(b);
        d.gpa.free(d.zbias);
    }

    /// `big` holds the [np][kdim] patches, then the [np][proj_dim] projection.
    pub fn setup(d: *Dev, np: usize) !void {
        const cfg = d.cfg;
        inline for (.{ &d.b.big, &d.b.x, &d.b.t }, [_]usize{ np * cfg.kdim(), np * cfg.dim, np * cfg.dim }) |b, n| b.* = try d.be.tensorCreate(n * 4);
        d.zbias = try d.gpa.alloc(f32, cfg.proj_dim);
        @memset(d.zbias, 0.0);
        const ones = try d.gpa.alloc(f32, cfg.dim);
        defer d.gpa.free(ones);
        @memset(ones, 1.0);
        d.ones = try d.be.tensorCreate(cfg.dim * 4);
        try d.be.tensorUpload(d.ones, std.mem.sliceAsBytes(ones));
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

    /// In the weight's storage dtype; a missing bias is the zero one.
    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, bias orelse d.zbias, null);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try d.be.opLayerNorm(x, y, w, b, rows, dim, eps);
    }

    pub fn rmsUnit(d: *Dev, x: Buf, rows: usize, dim: usize, eps: f32) !void {
        try d.be.qkNorm(x, x, d.ones, rows, dim, eps);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }
};

// --- tests -----------------------------------------------------------------

// Gated on a CUDA device + the real gemma4 mmproj: the GPU encode vs the CPU
// reference on a small gradient image (GPU runs f16-regime GEMMs, so parity is
// relative, per-token cosine).
test "cuda gemma4 vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf";
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
    errdefer std.debug.print("cuda gemma4 vit parity: min token cos {d:.6}\n", .{min_cos});
    try std.testing.expect(min_cos > 0.999);
}
