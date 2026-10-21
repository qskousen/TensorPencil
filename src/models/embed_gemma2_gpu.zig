//! EmbeddingGemma 2 on the Vulkan backend: `embed_gemma2_dev.zig` over `tower_gpu.Dev`.

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const eg2 = @import("embed_gemma2.zig");
const embed_gemma2_dev = @import("embed_gemma2_dev.zig");
const tower_gpu = @import("tower_gpu.zig");
const gemma4v_vit = @import("gemma4v_vit.zig");
const gemma4v_vit_gpu = @import("gemma4v_vit_gpu.zig");
const test_gate = @import("../test_gate.zig");

pub const ModelGpu = struct {
    cpu: *const eg2.Model,

    pub fn init(cpu: *const eg2.Model) ModelGpu {
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
        try embed_gemma2_dev.embed(tower_gpu.Dev{ .ctx = ctx, .tl = &tl }, self.cpu, io, gpa, ids_list, outs);
    }

    /// `images[i]` -> `outs[i]`, the vision tower and the body both on the device. The
    /// weights stay resident until `evict`.
    pub fn embedImageBatch(self: *const ModelGpu, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, images: []const eg2.Rgb, outs: [][]f32) !void {
        var tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0);
        defer tl.deinit();
        try embed_gemma2_dev.embedImages(tower_gpu.Dev{ .ctx = ctx, .tl = &tl }, self.cpu, io, gpa, images, outs, VitEncode{ .ctx = ctx });
    }

    /// Drop the device copies of the model's weights, before the model is freed.
    pub fn evict(self: *const ModelGpu, ctx: *gpu.Context) void {
        tower_gpu.evictModel(ctx, self.cpu);
    }
};

const VitEncode = struct {
    ctx: *gpu.Context,

    pub fn run(v: VitEncode, vit: *const gemma4v_vit.Vit, io: std.Io, gpa: std.mem.Allocator, img: eg2.Rgb) !gemma4v_vit.Vit.Encoded {
        return gemma4v_vit_gpu.encodeResident(vit, v.ctx, io, gpa, img.pixels, img.width, img.height);
    }
};

test "embeddinggemma2 Vulkan matches CPU" {
    try test_gate.requireModelFile(std.testing.io, eg2.test_dir ++ "/model.safetensors");
    const ctx = gpu.Context.init(std.testing.allocator, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();
    try embed_gemma2_dev.testAgainstCpu(ModelGpu, ctx);
}

test "embeddinggemma2 Vulkan images match CPU" {
    try test_gate.requireModelFile(std.testing.io, eg2.test_dir ++ "/model.safetensors");
    const ctx = gpu.Context.init(std.testing.allocator, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();
    try embed_gemma2_dev.testImagesAgainstCpu(ModelGpu, ctx);
}
