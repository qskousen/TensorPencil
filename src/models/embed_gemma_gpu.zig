//! EmbeddingGemma on the Vulkan backend: `embed_gemma_dev.zig` over `tower_gpu.Dev`.

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const eg = @import("embed_gemma.zig");
const embed_gemma_dev = @import("embed_gemma_dev.zig");
const tower_gpu = @import("tower_gpu.zig");

pub const ModelGpu = struct {
    cpu: *const eg.Model,

    pub fn init(cpu: *const eg.Model) ModelGpu {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(ctx, io, gpa, &.{ids}, &outs);
    }

    /// `ids_list[i]` -> `outs[i]`, one upload, forward and download for the batch.
    pub fn embedBatch(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        var tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0);
        defer tl.deinit();
        try embed_gemma_dev.embed(tower_gpu.Dev{ .ctx = ctx, .tl = &tl }, self.cpu, io, gpa, ids_list, outs);
    }
};

// --- tests -----------------------------------------------------------------

test "embeddinggemma Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelGpu.init(&cpu);

    const ids = [_]u32{ 2, 23391, 1902, 1 }; // <bos> "hello world" <eos>
    const cpu_out = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(cpu_out);
    const gpu_out = try gpa.alloc(f32, eg.embed_dim);
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
    try std.testing.expect(dot >= 0.9999);
    try std.testing.expect(maxad < 1e-2);
}

test "embeddinggemma Vulkan embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelGpu.init(&cpu);

    const item0 = [_]u32{ 2, 23391, 1902, 1 };
    const item1 = [_]u32{ 2, 1902, 23391, 1902, 1 };
    const item2 = [_]u32{ 2, 23391, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][eg.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(ctx, io, gpa, ids, &single[i]);
    var batched: [3][eg.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(ctx, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}
