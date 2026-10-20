//! The Gemma 4 "unified" vision embedder on Vulkan: the adapter for
//! `gemma4_vit_dev.zig`. Linears run in each weight's storage dtype through
//! `lin_gpu.towerLinear`; nothing stays resident after an encode (`tower_gpu.Scope`).

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const tower_gpu = @import("tower_gpu.zig");
const gemma4_vit = @import("gemma4_vit.zig");
const gemma4_vit_dev = @import("gemma4_vit_dev.zig");
const gpu = @import("tp_gpu").context;
const Weight = @import("tp_ops").matmul.Weight;

const Vit = gemma4_vit.Vit;
const Buf = gpu.DeviceBuffer;

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on Vulkan. Same
/// contract as `Vit.encode`.
pub fn encode(vit: *const Vit, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    _ = io;
    const scope = tower_gpu.Scope.begin(ctx);
    var d: Dev = .{ .ctx = ctx, .cfg = vit.cfg, .tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0) };
    defer {
        d.deinit();
        scope.end(ctx, vit);
    }
    return gemma4_vit_dev.encode(&d, vit, gpa, rgb, width, height);
}

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
const Bufs = struct { big: Buf = none, x: Buf = none, t: Buf = none };

/// The `gemma4_vit_dev` adapter.
const Dev = struct {
    ctx: *gpu.Context,
    cfg: gemma4_vit.Config,
    tl: lin_gpu.Lin,
    b: Bufs = .{},
    ones: Buf = none,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.big, &d.b.x, &d.b.t, &d.ones }) |b| d.ctx.tensorDestroy(b);
        d.tl.deinit();
    }

    pub fn setup(d: *Dev, np: usize) !void {
        const cfg = d.cfg;
        inline for (.{ &d.b.big, &d.b.x, &d.b.t }, [_]usize{ np * @max(cfg.kdim(), cfg.proj_dim), np * cfg.dim, np * cfg.dim }) |b, n| b.* = try d.ctx.tensorCreate(n * 4);
        d.ones = try d.ctx.tensorCreate(cfg.dim * 4);
        const host = try std.heap.page_allocator.alloc(f32, cfg.dim);
        defer std.heap.page_allocator.free(host);
        @memset(host, 1.0);
        try d.ctx.tensorUpload(d.ones, std.mem.sliceAsBytes(host));
    }

    pub fn beginBatch(d: *Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: *Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: *Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn upload(d: *Dev, b: Buf, data: []const f32) !void {
        try d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        try d.ctx.tensorDownloadAt(b, 0, std.mem.sliceAsBytes(dst));
    }

    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_gpu.towerLinear(&d.tl, y, x, m, w, bias);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try d.ctx.opElt(.layernorm, x, y, try small(d.ctx, w), try small(d.ctx, b), .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
    }

    pub fn rmsUnit(d: *Dev, x: Buf, rows: usize, dim: usize, eps: f32) !void {
        try d.ctx.opElt(.rmsnorm, x, x, d.ones, null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

fn small(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}
