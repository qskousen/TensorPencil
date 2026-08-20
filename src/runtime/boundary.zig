//! What a stepper's internal chunk loop does at a safe boundary: the prefill
//! end of the handshake `engine.checkpoint` provides between decoded tokens.
//!
//! A frontend's turn intents — a coordinator-published VRAM ceiling, a pause
//! gate, a cancel flag — are polled between decoded tokens. PREFILL never
//! reaches that poll: the frontend calls `model.prefill(ids)` once and the
//! stepper chunks internally, so a long prompt (the whole slow half of a turn,
//! and all of a vision turn) ran with none of them. The frontend installs one
//! `Hook`; the stepper calls `check` where a chunk ends, which is where no batch
//! is open, the split's host shadow is committed, and the committed length is
//! exactly whole chunks — the same conditions that hold between decoded tokens.
//!
//! `error.Stopped` is deliberately NOT `ops/cancel.zig`'s `error.Canceled`.
//! Stopping here leaves every committed token whole, so a caller keeps its KV
//! and resumes from `cached()`. A mid-kernel cancel unwinds a forward
//! half-applied — one layer's recurrent state advanced, its neighbour's not —
//! which `cached()` cannot report and only a context reset fixes. A frontend
//! that arms both has to tell them apart.

const std = @import("std");

/// What a boundary check decides.
pub const Outcome = enum { proceed, stop };

/// Installed by the frontend on the stepper (`model.boundary`). `at` runs on the
/// WORKER's own thread, so it may bind the device context and move residency;
/// the coordinator that published the intent never touches the device itself.
pub const Hook = struct {
    ctx: *anyopaque,
    at: *const fn (ctx: *anyopaque) Outcome,
};

/// Run `st`'s boundary hook. A no-op for a stepper without the field (nothing
/// coordinates it) or a frontend that installed none (the CLI), so a loop can
/// call it unconditionally.
pub fn check(st: anytype) error{Stopped}!void {
    const M = @typeInfo(@TypeOf(st)).pointer.child;
    if (comptime !@hasField(M, "boundary")) return;
    const h = st.boundary orelse return;
    if (h.at(h.ctx) == .stop) return error.Stopped;
}

// --- tests -----------------------------------------------------------------

const Stepper = struct {
    boundary: ?Hook = null,
    chunks: usize = 0,

    fn prefill(self: *Stepper, n: usize) !void {
        var off: usize = 0;
        while (off < n) : (off += 1) {
            try check(self);
            self.chunks += 1;
        }
    }
};

const Coordinator = struct {
    stop_after: usize,
    seen: usize = 0,

    fn at(ctx: *anyopaque) Outcome {
        const self: *Coordinator = @ptrCast(@alignCast(ctx));
        defer self.seen += 1;
        return if (self.seen >= self.stop_after) .stop else .proceed;
    }

    fn hook(self: *Coordinator) Hook {
        return .{ .ctx = self, .at = at };
    }
};

test "no hook installed: every chunk runs" {
    var st: Stepper = .{};
    try st.prefill(4);
    try std.testing.expectEqual(@as(usize, 4), st.chunks);
}

test "a stop unwinds the loop at the boundary, keeping earlier chunks" {
    var co: Coordinator = .{ .stop_after = 2 };
    var st: Stepper = .{ .boundary = co.hook() };
    try std.testing.expectError(error.Stopped, st.prefill(8));
    // Committed exactly the chunks that ran before the stop: the caller's
    // `cached()` stays whole, which is what separates this from a mid-kernel
    // cancel.
    try std.testing.expectEqual(@as(usize, 2), st.chunks);
}

test "a stepper without the field compiles to a no-op" {
    const Bare = struct { chunks: usize = 0 };
    var b: Bare = .{};
    try check(&b);
    try std.testing.expectEqual(@as(usize, 0), b.chunks);
}
