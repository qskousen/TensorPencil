//! SigLIP2 encoders on the CUDA backend: `embed_siglip_dev.zig` over `tower_cuda.Dev`.
//! f16 tensor-core GEMM regime, so parity against the CPU is relative.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const siglip = @import("embed_siglip.zig");
const embed_siglip_dev = @import("embed_siglip_dev.zig");
const tower_cuda = @import("tower_cuda.zig");

const Backend = cuda.Backend;

/// SigLIP2 text tower; borrows a loaded CPU `TextModel`'s weights.
pub const TextModelCuda = struct {
    cpu: *const siglip.TextModel,

    pub fn init(cpu: *const siglip.TextModel) TextModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const TextModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(be, io, gpa, &.{ids}, &outs);
    }

    pub fn embedBatch(self: *const TextModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        const d: tower_cuda.Dev = .{ .be = be };
        try embed_siglip_dev.text(d, self.cpu, io, gpa, ids_list, outs);
    }
};

/// SigLIP2 visual tower; the MAP head runs on the host.
pub const VisualModelCuda = struct {
    cpu: *const siglip.VisualModel,

    pub fn init(cpu: *const siglip.VisualModel) VisualModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const VisualModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, img: []const f32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(be, io, gpa, &.{img}, &outs);
    }

    pub fn embedBatch(self: *const VisualModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, imgs: []const []const f32, outs: [][]f32) !void {
        const d: tower_cuda.Dev = .{ .be = be };
        try embed_siglip_dev.visual(d, self.cpu, io, gpa, imgs, outs);
    }
};

// --- tests -----------------------------------------------------------------

fn parity(cpu_out: []const f32, gpu_out: []const f32) struct { cos: f32, rmse: f32 } {
    var dot: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    var num: f32 = 0;
    var den: f32 = 0;
    for (cpu_out, gpu_out) |a, b| {
        dot += a * b;
        na += a * a;
        nb += b * b;
        num += (a - b) * (a - b);
        den += a * a;
    }
    return .{ .cos = dot / (@sqrt(na) * @sqrt(nb)), .rmse = @sqrt(num / den) };
}

test "siglip2 text tower CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelCuda.init(&cpu);

    const ids = [_]u32{ 17534, 2134, 1 };
    const c = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, &ids, c);
    try dev.embed(be, io, gpa, &ids, g);
    const p = parity(c, g);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ p.cos, p.rmse });
    try std.testing.expect(p.cos > 0.999 and p.rmse < 0.05);
}

test "siglip2 text tower CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.TextModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = TextModelCuda.init(&cpu);

    const item0 = [_]u32{ 17534, 2134, 1 };
    const item1 = [_]u32{ 2134, 17534, 2134, 1 };
    const item2 = [_]u32{ 17534, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][siglip.embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try dev.embed(be, io, gpa, ids, &single[i]);
    var batched: [3][siglip.embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try dev.embedBatch(be, io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("text item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3);
    }
}

test "siglip2 visual tower CUDA matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_input.f32", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(raw);
    const img = try gpa.alloc(f32, raw.len / @sizeOf(f32));
    defer gpa.free(img);
    @memcpy(std.mem.sliceAsBytes(img), raw);

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelCuda.init(&cpu);

    const c = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(c);
    const g = try gpa.alloc(f32, siglip.embed_dim);
    defer gpa.free(g);
    try cpu.embed(io, gpa, img, c);
    try dev.embed(be, io, gpa, img, g);
    const p = parity(c, g);
    errdefer std.debug.print("cos {d} rmse {d}\n", .{ p.cos, p.rmse });
    try std.testing.expect(p.cos > 0.999 and p.rmse < 0.05);
}

test "siglip2 visual tower CUDA embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var cpu = try siglip.VisualModel.open(gpa, io, dir);
    defer cpu.deinit();
    const dev = VisualModelCuda.init(&cpu);

    const nimg = 3 * 224 * 224;
    var prng = std.Random.DefaultPrng.init(0x71B2);
    const r = prng.random();
    const imgs_data = try gpa.alloc(f32, 2 * nimg);
    defer gpa.free(imgs_data);
    for (imgs_data) |*e| e.* = (r.float(f32) - 0.5) * 2.0;
    const imgs = [_][]const f32{ imgs_data[0..nimg], imgs_data[nimg..] };

    var single: [2][siglip.embed_dim]f32 = undefined;
    for (imgs, 0..) |img, i| try dev.embed(be, io, gpa, img, &single[i]);
    var batched: [2][siglip.embed_dim]f32 = undefined;
    var outs: [2][]f32 = .{ &batched[0], &batched[1] };
    try dev.embedBatch(be, io, gpa, &imgs, &outs);

    for (0..2) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("visual item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-3);
    }
}
