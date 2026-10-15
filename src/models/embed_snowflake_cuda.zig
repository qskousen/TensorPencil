//! Snowflake Arctic Embed (GTE) on the CUDA backend: the adapter for
//! `embed_snowflake_dev.zig`. f16 tensor-core GEMM regime, so parity against the CPU
//! is relative.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const es = @import("embed_snowflake.zig");
const embed_snowflake_dev = @import("embed_snowflake_dev.zig");
const tower_cuda = @import("tower_cuda.zig");

const Backend = cuda.Backend;

pub const ModelCuda = struct {
    cpu: *const es.Model,

    pub fn init(cpu: *const es.Model) ModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(be, io, gpa, &.{ids}, &outs);
    }

    /// `ids_list[i]` -> `outs[i]`, one upload, forward and download for the batch.
    pub fn embedBatch(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        _ = io;
        try embed_snowflake_dev.embed(tower_cuda.Dev{ .be = be }, self.cpu, gpa, ids_list, outs);
    }
};

// --- tests -----------------------------------------------------------------

test "snowflake CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const ids = [_]u32{ 0, 33600, 31, 8999, 2 };
    const c = try gpa.alloc(f32, es.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, es.embed_dim);
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

test "snowflake CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try es.Model.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = ModelCuda.init(&cpu);

    const item0 = [_]u32{ 0, 33600, 31, 2 };
    const item1 = [_]u32{ 0, 8999, 33600, 31, 2 };
    const item2 = [_]u32{ 0, 31, 2 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][es.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(be, io, gpa, ids, &single[i]);
    var batched: [3][es.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(be, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3); // f16 tensor-core GEMM regime
    }
}
