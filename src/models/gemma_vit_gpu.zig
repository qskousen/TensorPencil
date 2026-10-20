//! Gemma 3 vision tower on the Vulkan backend: the adapter for `gemma_vit_dev.zig`.
//! Attention is the non-causal `attn_full` at the true 72-wide head, and the linears
//! run in each weight's storage dtype through `lin_gpu.towerLinear`. Nothing stays
//! resident after an encode (`tower_gpu.Scope`).

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const tower_gpu = @import("tower_gpu.zig");
const Weight = @import("tp_ops").matmul.Weight;
const gemma_vit = @import("gemma_vit.zig");
const gemma_vit_dev = @import("gemma_vit_dev.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const safetensors = @import("tp_core").safetensors;

const Vit = gemma_vit.Vit;
const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// The tower on Vulkan. Its weights are the CPU `Vit`'s own, in their checkpoint
/// dtype: `lin_gpu` runs each by its storage.
pub const VitGpu = struct {
    vit: *const Vit,

    pub fn load(gpa: std.mem.Allocator, vit: *const Vit) !VitGpu {
        _ = gpa;
        return .{ .vit = vit };
    }

    pub fn deinit(self: *VitGpu) void {
        self.* = undefined;
    }

    /// Encode interleaved RGB pixels to Gemma image-token embeddings on Vulkan. Same
    /// contract as `Vit.encode`.
    pub fn encode(self: *const VitGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
        const cfg = self.vit.cfg;
        const np = cfg.side() * cfg.side();
        const scope = tower_gpu.Scope.begin(ctx);
        var d: Dev = .{ .ctx = ctx, .cfg = cfg, .tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0) };
        defer {
            d.deinit();
            scope.end(ctx, self.vit);
        }
        const sizes = [_]usize{ np * cfg.dim, np * cfg.dim, np * @max(cfg.dim, cfg.ffn), np * cfg.dim, np * cfg.dim, np * cfg.dim, np * cfg.dim };
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try ctx.tensorCreate(n * 4);
        return gemma_vit_dev.encode(&d, self.vit, io, gpa, rgb, width, height);
    }
};

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
const Bufs = struct { x: Buf = none, normed: Buf = none, big: Buf = none, t: Buf = none };

/// The `gemma_vit_dev` adapter.
const Dev = struct {
    ctx: *gpu.Context,
    cfg: gemma_vit.Config,
    tl: lin_gpu.Lin,
    b: Bufs = .{},
    q: Buf = none,
    k: Buf = none,
    v: Buf = none,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.q, &d.k, &d.v }) |b| d.ctx.tensorDestroy(b);
        d.tl.deinit();
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
        try d.ctx.tensorDownload(b, std.mem.sliceAsBytes(dst));
    }

    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_gpu.towerLinear(&d.tl, y, x, m, w, bias);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, bias: []const f32, rows: usize) !void {
        try d.ctx.opElt(.layernorm, x, y, try nbuf(d.ctx, w), try nbuf(d.ctx, bias), .{ .u0 = @intCast(rows), .u1 = @intCast(d.cfg.dim), .f0 = d.cfg.eps }, rows, 1, 1);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// `attn_full` at the true 72-wide head.
    pub fn attention(d: *Dev, blk: *const gemma_vit.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const hd = cfg.headDim();
        try d.linear(d.q, d.b.normed, np, blk.q, blk.q_b);
        try d.linear(d.k, d.b.normed, np, blk.k, blk.k_b);
        try d.linear(d.v, d.b.normed, np, blk.v, blk.v_b);
        try d.ctx.opElt(.attn_full, d.q, d.k, d.v, d.b.big, .{
            .u0 = @intCast(np),
            .u1 = @intCast(cfg.n_heads),
            .u2 = @intCast(cfg.n_heads),
            .u3 = @intCast(hd),
            .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, np * cfg.n_heads, 1, 1);
        return d.b.big;
    }
};

test "vulkan gemma vit matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/mnt/watch/models/llm/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/mmproj-F16.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var g = try @import("tp_core").gguf.Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();
    var dev = try VitGpu.load(gpa, &vit);
    defer dev.deinit();

    var pixels: [128 * 96 * 3]u8 = undefined;
    for (0..96) |y| for (0..128) |x| {
        const i = (y * 128 + x) * 3;
        pixels[i] = @intCast((x * 2) & 0xff);
        pixels[i + 1] = @intCast((y * 2) & 0xff);
        pixels[i + 2] = 100;
    };

    var want = try vit.encode(io, gpa, &pixels, 128, 96);
    defer want.deinit(gpa);
    var got = try dev.encode(ctx, io, gpa, &pixels, 128, 96);
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
    errdefer std.debug.print("vulkan gemma vit parity: min token cos {d:.6}, rel RMSE {d:.6}\n", .{ min_cos, @sqrt(num / den) });
    try std.testing.expect(min_cos > 0.999);
    try std.testing.expect(@sqrt(num / den) < 0.05);
}
