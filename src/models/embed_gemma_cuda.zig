//! EmbeddingGemma on the CUDA backend: `embed_gemma_dev.zig` over `tower_cuda.Dev`.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const eg = @import("embed_gemma.zig");
const embed_gemma_dev = @import("embed_gemma_dev.zig");
const tower_cuda = @import("tower_cuda.zig");

const Backend = cuda.Backend;

pub const ModelCuda = struct {
    cpu: *const eg.Model,

    pub fn init(cpu: *const eg.Model) ModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(be, io, gpa, &.{ids}, &outs);
    }

    /// `ids_list[i]` -> `outs[i]`, one upload, forward and download for the batch.
    pub fn embedBatch(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        try embed_gemma_dev.embed(tower_cuda.Dev{ .be = be }, self.cpu, io, gpa, ids_list, outs);
    }
};

// --- tests -----------------------------------------------------------------

test "embeddinggemma CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const ids = [_]u32{ 2, 23391, 1902, 1 };
    const c = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, eg.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, &ids, c);
    try dev.embed(be, io, gpa, &ids, g);

    var dot: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    var num: f32 = 0;
    var den: f32 = 0;
    for (c, g) |a, b| {
        dot += a * b;
        na += a * a;
        nb += b * b;
        num += (a - b) * (a - b);
        den += a * a;
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    const rmse = @sqrt(num / den);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ cos, rmse });
    try std.testing.expect(cos > 0.999 and rmse < 0.05);
}

test "embeddinggemma CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try eg.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const item0 = [_]u32{ 2, 23391, 1902, 1 };
    const item1 = [_]u32{ 2, 1902, 23391, 1902, 1 };
    const item2 = [_]u32{ 2, 23391, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][eg.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(be, io, gpa, ids, &single[i]);
    var batched: [3][eg.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(be, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3);
    }
}
