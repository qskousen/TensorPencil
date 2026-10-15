//! SigLIP2 encoders on the Vulkan backend: `embed_siglip_dev.zig` over `tower_gpu.Dev`.

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const siglip = @import("embed_siglip.zig");
const embed_siglip_dev = @import("embed_siglip_dev.zig");
const tower_gpu = @import("tower_gpu.zig");

/// SigLIP2 text tower; borrows a loaded CPU `TextModel`'s weights.
pub const TextModelGpu = struct {
    cpu: *const siglip.TextModel,

    pub fn init(cpu: *const siglip.TextModel) TextModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const TextModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(ctx, io, gpa, &.{ids}, &outs);
    }

    pub fn embedBatch(self: *const TextModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        var tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0);
        defer tl.deinit();
        const d: tower_gpu.Dev = .{ .ctx = ctx, .tl = &tl };
        try embed_siglip_dev.text(d, self.cpu, io, gpa, ids_list, outs);
    }
};

/// SigLIP2 visual tower; the MAP head runs on the host.
pub const VisualModelGpu = struct {
    cpu: *const siglip.VisualModel,

    pub fn init(cpu: *const siglip.VisualModel) VisualModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const VisualModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, img: []const f32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(ctx, io, gpa, &.{img}, &outs);
    }

    pub fn embedBatch(self: *const VisualModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, imgs: []const []const f32, outs: [][]f32) !void {
        var tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0);
        defer tl.deinit();
        const d: tower_gpu.Dev = .{ .ctx = ctx, .tl = &tl };
        try embed_siglip_dev.visual(d, self.cpu, io, gpa, imgs, outs);
    }
};

// --- tests -----------------------------------------------------------------

// Device parity: the Vulkan text tower must match the (ONNX-validated) CPU
// TextModel. Self-skips without a Vulkan device / the integration build, or
// when the checkpoint is absent.
test "siglip2 text tower Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelGpu.init(&cpu);

    const ids = [_]u32{ 17534, 2134, 1 }; // "hello world" + <eos>
    const cpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(gpu_out);
    try cpu.embed(io, gpa, &ids, cpu_out);
    try dev.embed(ctx, io, gpa, &ids, gpu_out);

    var dot: f32 = 0;
    var maxad: f32 = 0;
    for (cpu_out, gpu_out) |c, g| {
        dot += c * g;
        maxad = @max(maxad, @abs(c - g));
    }
    errdefer std.debug.print("cosine {d}, max abs diff {d}\n", .{ dot, maxad });
    try std.testing.expect(dot >= 0.9999); // both unit vectors
    try std.testing.expect(maxad < 1e-2);
}

test "siglip2 text tower Vulkan embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelGpu.init(&cpu);

    const item0 = [_]u32{ 17534, 2134, 1 };
    const item1 = [_]u32{ 2134, 17534, 2134, 1 };
    const item2 = [_]u32{ 17534, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][siglip.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(ctx, io, gpa, ids, &single[i]);
    var batched: [3][siglip.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(ctx, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("text item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}

test "siglip2 visual tower Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_input.f32", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(raw);
    const img = try gpa.alloc(f32, raw.len / @sizeOf(f32));
    defer gpa.free(img);
    @memcpy(std.mem.sliceAsBytes(img), raw);

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelGpu.init(&cpu);

    const cpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(gpu_out);
    try cpu.embed(io, gpa, img, cpu_out);
    try dev.embed(ctx, io, gpa, img, gpu_out);

    var dot: f32 = 0;
    var maxad: f32 = 0;
    for (cpu_out, gpu_out) |c, g| {
        dot += c * g;
        maxad = @max(maxad, @abs(c - g));
    }
    errdefer std.debug.print("cosine {d}, max abs diff {d}\n", .{ dot, maxad });
    try std.testing.expect(dot >= 0.9999);
    try std.testing.expect(maxad < 1e-2);
}

test "siglip2 visual tower Vulkan embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelGpu.init(&cpu);

    const nimg = 3 * 224 * 224;
    var prng = std.Random.DefaultPrng.init(0x71B2);
    const r = prng.random();
    const imgs_data = try gpa.alloc(f32, 2 * nimg);
    defer gpa.free(imgs_data);
    for (imgs_data) |*e| e.* = (r.float(f32) - 0.5) * 2.0;
    const imgs = [_][]const f32{ imgs_data[0..nimg], imgs_data[nimg..] };

    var single: [2][siglip.embed_dim]f32 = undefined;
    for (imgs, 0..) |img, i| try dev.embed(ctx, io, gpa, img, &single[i]);
    var batched: [2][siglip.embed_dim]f32 = undefined;
    var outs: [2][]f32 = .{ &batched[0], &batched[1] };
    try dev.embedBatch(ctx, io, gpa, &imgs, &outs);

    for (0..2) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("visual item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}
