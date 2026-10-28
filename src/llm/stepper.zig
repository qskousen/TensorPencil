//! What the chat session calls on a stepper. A stepper on another backend must
//! answer every one of these its CUDA twin answers for the same family: what a
//! family cannot do (a recurrent qwen35 cannot truncate its state) its CUDA
//! stepper lacks too, and the rest is a backend that has not caught up.

const std = @import("std");

pub const methods = [_][]const u8{
    "init",            "deinit",         "cached",          "remaining",         "vocab",
    "capacityMax",     "ensureCapacity", "step",            "prefill",           "prefillImage",
    "truncate",        "checkpoint",     "checkpointBytes", "restoreCheckpoint", "resetCache",
    "reinitCache",     "warmWeights",    "autoOffload",     "resetResidency",    "promoteLayers",
    "promoteLayer",    "promoteCost",    "migrateLayer",    "enableCpuSplit",    "offloadUntilFree",
    "offloadToBudget", "noiseAtLayer",   "vramUsed",        "stepArgmax",        "stepArgmaxPen",
    "stepSelect",      "stepSelectPen",  "maxSelect",       "seqCreate",         "seqSwap",
    "seqDestroy",
};

pub const fields = [_][]const u8{ "io", "boundary", "split", "cfg", "kv_dtype" };

/// The names `Ref` answers and `T` does not, methods then fields.
pub fn missing(comptime Ref: type, comptime T: type) []const []const u8 {
    comptime {
        var out: []const []const u8 = &.{};
        for (methods) |n| {
            if (@hasDecl(Ref, n) and !@hasDecl(T, n)) out = out ++ .{n};
        }
        for (fields) |n| {
            if (@hasField(Ref, n) and !@hasField(T, n)) out = out ++ .{"." ++ n};
        }
        return out;
    }
}

fn joined(comptime names: []const []const u8) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (names) |n| out = out ++ " " ++ n;
        return out;
    }
}

fn expectGaps(comptime Ref: type, comptime T: type, comptime want: []const []const u8) !void {
    errdefer std.debug.print("{s} lacks, against {s}:\n", .{ @typeName(T), @typeName(Ref) });
    try std.testing.expectEqualStrings(comptime joined(want), comptime joined(missing(Ref, T)));
}

// The ledger: what each non-CUDA stepper still lacks. Closing a gap means implementing
// it with its own test and deleting it here; the chat session takes a stepper only
// once its line is empty.
test "each Vulkan stepper's gaps against its CUDA twin are exactly the ledger's" {
    const models = @import("../models.zig");
    try expectGaps(models.qwen3_cuda.CudaLM, models.qwen3_gpu.VulkanLM, &.{});
    try expectGaps(models.qwen35_cuda.CudaLM, models.qwen35_gpu.VulkanLM, &.{});
    try expectGaps(models.gemma3_cuda.CudaLM, models.gemma3_gpu.VulkanLM, &.{});
    try expectGaps(models.gemma4_cuda.CudaLM, models.gemma4_gpu.VulkanLM, &.{});
    try expectGaps(models.k2_horizon_cuda.CudaLM, models.k2_horizon_gpu.VulkanLM, &.{});
}
