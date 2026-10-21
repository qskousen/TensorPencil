//! EmbeddingGemma 2 on the CUDA backend: `embed_gemma2_dev.zig` over `tower_cuda.Dev`.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const eg2 = @import("embed_gemma2.zig");
const embed_gemma2_dev = @import("embed_gemma2_dev.zig");
const tower_cuda = @import("tower_cuda.zig");
const gemma4v_vit = @import("gemma4v_vit.zig");
const gemma4v_vit_cuda = @import("gemma4v_vit_cuda.zig");
const test_gate = @import("../test_gate.zig");

const Backend = cuda.Backend;

pub const ModelCuda = struct {
    cpu: *const eg2.Model,

    pub fn init(cpu: *const eg2.Model) ModelCuda {
        return .{ .cpu = cpu };
    }

    pub fn embed(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(be, io, gpa, &.{ids}, &outs);
    }

    /// `ids_list[i]` -> `outs[i]`, one upload, forward and download for the batch.
    pub fn embedBatch(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        try embed_gemma2_dev.embed(tower_cuda.Dev{ .be = be }, self.cpu, io, gpa, ids_list, outs);
    }

    /// `images[i]` -> `outs[i]`, the vision tower and the body both on the device. The
    /// weights stay resident until `evict`.
    pub fn embedImageBatch(self: *const ModelCuda, be: *Backend, io: std.Io, gpa: std.mem.Allocator, images: []const eg2.Rgb, outs: [][]f32) !void {
        try embed_gemma2_dev.embedImages(tower_cuda.Dev{ .be = be }, self.cpu, io, gpa, images, outs, VitEncode{ .be = be });
    }

    /// Drop the device copies of the model's weights, before the model is freed.
    pub fn evict(self: *const ModelCuda, be: *Backend) void {
        tower_cuda.evictModel(be, self.cpu);
    }
};

const VitEncode = struct {
    be: *Backend,

    pub fn run(v: VitEncode, vit: *const gemma4v_vit.Vit, io: std.Io, gpa: std.mem.Allocator, img: eg2.Rgb) !gemma4v_vit.Vit.Encoded {
        return gemma4v_vit_cuda.encodeResident(vit, v.be, io, gpa, img.pixels, img.width, img.height);
    }
};

test "embeddinggemma2 CUDA matches CPU" {
    try test_gate.requireModelFile(std.testing.io, eg2.test_dir ++ "/model.safetensors");
    const be = Backend.init(std.testing.allocator) catch return error.SkipZigTest;
    defer be.deinit();
    try embed_gemma2_dev.testAgainstCpu(ModelCuda, be);
}

test "embeddinggemma2 CUDA images match CPU" {
    try test_gate.requireModelFile(std.testing.io, eg2.test_dir ++ "/model.safetensors");
    const be = Backend.init(std.testing.allocator) catch return error.SkipZigTest;
    defer be.deinit();
    try embed_gemma2_dev.testImagesAgainstCpu(ModelCuda, be);
}
