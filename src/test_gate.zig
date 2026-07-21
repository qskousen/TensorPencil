//! Test gating for the slow integration suite.
//!
//! `zig build test` runs only the fast CPU unit tests. The slow integration
//! tests — GPU device tests and the real-model LLM/parity tests that load
//! multi-GB checkpoints and run inference in Debug — are gated behind
//! `-Dintegration` (see build.zig). GPU tests self-skip because the device
//! `init` fails in test builds when the flag is off; the heavy real-model tests
//! call `requireIntegration` (or `requireModelFile`) at their top.

const std = @import("std");
const build_options = @import("build_options");
const cuda = @import("tp_gpu").cuda;

/// Skip the calling test unless the integration suite is enabled.
pub fn requireIntegration() error{SkipZigTest}!void {
    if (!build_options.integration) return error.SkipZigTest;
}

/// Skip unless the integration suite is enabled AND `path` exists (a real-model
/// checkpoint). Combines the `-Dintegration` gate with the pre-existing
/// file-presence check the model tests already did.
pub fn requireModelFile(io: std.Io, path: []const u8) error{SkipZigTest}!void {
    if (!build_options.integration) return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
}

/// Test helper: occupy device VRAM until only ~`target_free` bytes stay free,
/// simulating another process / a resident image model for tests that assert
/// VRAM-pressure behavior (split planning, eviction, reclaim). Allocates in
/// 1 GiB chunks (one huge allocation can fail on a fragmented card) and stops
/// early if the allocator refuses — the caller asserts on the RESULTING free
/// figure, not on how much was ballooned. Release with `deinit`.
pub const VramBalloon = struct {
    be: *cuda.Backend,
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.backend.DeviceBuffer) = .empty,

    pub fn inflateToFree(gpa: std.mem.Allocator, be: *cuda.Backend, target_free: u64) !VramBalloon {
        var self: VramBalloon = .{ .be = be, .gpa = gpa };
        errdefer self.deinit();
        while (true) {
            const free = be.ctx.memGetInfo().free;
            if (free <= target_free) break;
            const chunk: u64 = @min(1 << 30, free - target_free);
            if (chunk < (64 << 20)) break; // close enough; tiny allocs just churn
            const b = be.tensorCreate(chunk) catch break; // fragmented: keep what we got
            try self.bufs.append(gpa, b);
        }
        return self;
    }

    pub fn deinit(self: *VramBalloon) void {
        for (self.bufs.items) |*b| self.be.tensorDestroy(b);
        self.bufs.deinit(self.gpa);
    }
};
