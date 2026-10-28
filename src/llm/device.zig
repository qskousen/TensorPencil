//! The device a chat LLM runs on, one arm per backend, so a session, its driver and
//! its host name no backend. Every method switches exhaustively: a new arm must
//! answer all of them.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const vk = @import("tp_gpu").context;
const gguf_mod = @import("tp_core").gguf;
const Container = @import("tp_core").container.Container;
const gpu_context = @import("tp_gpu").context;
const session = @import("session.zig");

pub const MemTag = @import("tp_gpu").mem_tag.MemTag;
pub const MemInfo = struct { free: usize, total: usize };
pub const ForeignReclaim = vk.ForeignReclaim;
pub const WeightNoise = @import("tp_gpu").weight_noise.WeightNoise;

pub const Kind = enum { zig_cuda, cuda, vulkan };

/// Whether weight noise would reach this checkpoint's weights on a `kind` device; null
/// is the CPU, whose steppers draw none. Only GGUF block quants have noised kernels.
pub fn weightNoiseSupported(kind: ?Kind, c: *const Container) bool {
    const gg = switch (c.*) {
        .gguf => |*g| g,
        .safetensors => return false,
    };
    return switch (kind orelse return false) {
        .zig_cuda, .cuda => session.weightNoiseSupportedFor(gg, cuda.Backend.weightNoiseSupported),
        .vulkan => session.weightNoiseSupportedFor(gg, vk.Context.weightNoiseSupported),
    };
}

pub const Device = union(enum) {
    cuda: *cuda.Backend,
    vulkan: *vk.Context,

    pub fn init(arena: std.mem.Allocator, io: std.Io, kind: Kind) !Device {
        return switch (kind) {
            .zig_cuda => .{ .cuda = try cuda.Backend.init(arena) },
            .cuda => .{ .cuda = try cuda.Backend.initLibs(arena) },
            .vulkan => .{ .vulkan = try vk.Context.init(arena, io) },
        };
    }

    pub fn deinit(d: Device) void {
        switch (d) {
            inline else => |b| b.deinit(),
        }
    }

    /// Make the device current on the calling thread; device memory queries and frees
    /// read the current context.
    pub fn bindThread(d: Device) void {
        switch (d) {
            inline else => |b| b.bindThread(),
        }
    }

    /// Device memory this process holds through the device.
    pub fn deviceUsed(d: Device) u64 {
        return switch (d) {
            inline else => |b| b.deviceUsed(),
        };
    }

    /// The card's free and total memory, null when it cannot be asked.
    pub fn memInfo(d: Device) ?MemInfo {
        return switch (d) {
            .cuda => |be| if (be.ctx.memGetInfo()) |mi| .{ .free = mi.free, .total = mi.total } else null,
            .vulkan => |ctx| if (ctx.memGetInfo()) |mi| .{ .free = mi.free, .total = mi.total } else null,
        };
    }

    pub fn isLost(d: Device) bool {
        return switch (d) {
            .cuda => |be| be.ctx.isLost(),
            .vulkan => |ctx| ctx.isLost(),
        };
    }

    /// The table the decode kernels read per launch, so a live edit takes effect mid-reply.
    pub fn weightNoise(d: Device) *WeightNoise {
        return switch (d) {
            inline else => |b| &b.weight_noise,
        };
    }

    /// Pin every weight (an LLM never streams), and name allocations by component.
    /// Before the model allocates anything.
    pub fn prepareForLlm(d: Device) void {
        switch (d) {
            .cuda => |be| {
                be.pinAllWeights();
                be.enableLlmMemTags();
            },
            .vulkan => |ctx| {
                ctx.pin_budget = std.math.maxInt(u64);
                ctx.enableLlmMemTags();
            },
        }
    }

    pub fn memTagBreakdown(d: Device) [MemTag.count]u64 {
        return switch (d) {
            .cuda => |be| be.memTagBreakdown(),
            .vulkan => |ctx| blk: {
                var out: [MemTag.count]u64 = @splat(0);
                for (&out, 0..) |*v, i| v.* = ctx.memTagUsed(@fromBackingInt(@intCast(i)));
                break :blk out;
            },
        };
    }

    /// Kernels compiled rather than loaded: count and total time.
    pub fn jitStats(d: Device) struct { count: u64, ns: u64 } {
        return switch (d) {
            .cuda => |be| blk: {
                const j = be.jitStats();
                break :blk .{ .count = j.count, .ns = j.ns };
            },
            .vulkan => |ctx| .{ .count = ctx.jit_count, .ns = ctx.jit_ns },
        };
    }

    /// What the device's out-of-memory ladder calls last, to free memory another
    /// context on the same card holds.
    pub fn setForeignReclaim(d: Device, r: ?ForeignReclaim) void {
        switch (d) {
            inline else => |b| b.foreign_reclaim = r,
        }
    }

    pub fn backendKind(d: Device) Kind {
        return switch (d) {
            .cuda => .cuda,
            .vulkan => .vulkan,
        };
    }

    /// Read weight bytes from the checkpoint file through `wr` rather than faulting
    /// its mapping (`session.containerReader`, `session.ggufReader`).
    pub fn useFileReads(d: Device, wr: gpu_context.WeightReader) void {
        switch (d) {
            .cuda => |be| be.ctx.addWeightReader(wr),
            .vulkan => |ctx| ctx.addWeightReader(wr),
        }
    }
};
