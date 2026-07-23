//! Cooperative pause for long-running generation loops.
//!
//! The sibling of `ops/cancel.zig`: wherever a loop polls its cancel flag at a
//! clean boundary — between diffusion sampling steps, between decoded tokens —
//! it can also consult a pause `Gate`. Where cancel *unwinds* the loop, pause
//! *parks* it: a paused worker blocks on the gate at the boundary, holding its
//! in-flight state (and, by default, its VRAM) until it is unpaused.
//!
//! Unlike cancel, pause is NEVER polled mid-kernel — parking inside a matmul
//! would strand half-computed state and held locks. It is a coarse,
//! loop-boundary primitive, so there is no threadlocal token: the loop simply
//! holds a `*Gate` (in its Options) and calls `checkpoint()` at the same points
//! it checks cancel.
//!
//! One Gate per worker (the LLM decode worker, the diffusion worker) — mirroring
//! cancel's per-worker isolation, so pausing one engine never parks the other.
//! The UI exposes an independent pause button per model (next to each unload
//! button), each driving its own gate.
//!
//! Blocking uses `std.Io.Mutex`/`Condition` (cross-platform, no raw syscalls) so
//! a parked worker consumes no CPU while it waits. Every method takes the
//! `std.Io` the worker already threads through its loop.

const std = @import("std");
const Io = std.Io;

/// A per-worker pause gate. Zero-initialized (`.{}`) it starts unpaused, so a
/// `checkpoint()` on a fresh gate is a cheap uncontended lock + immediate
/// return.
pub const Gate = struct {
    mu: Io.Mutex = Io.Mutex.init,
    cond: Io.Condition = Io.Condition.init,
    /// When true, `checkpoint()` blocks the calling worker at the next boundary.
    paused: bool = false,
    /// Set (only meaningful while paused) to ask a parked worker to shed the
    /// model: snapshot its in-flight state to host RAM and free the weights.
    /// The worker sees this via a `.unload` result from `checkpoint()`, then
    /// parks in `awaitResume()` until unpaused, at which point it reloads and
    /// resumes. Cleared on `unpause()`.
    want_unload: bool = false,

    /// What a paused worker should do when it reaches a boundary.
    pub const Outcome = enum {
        /// Not paused (or just unpaused): keep generating.
        proceed,
        /// Paused *and* an unload was requested: snapshot in-flight state, free
        /// the model, then call `awaitResume()`.
        unload,
        /// The cancel flag passed to `checkpoint` went true while parked: the
        /// caller should unwind exactly as it would for a normal cancel. Lets a
        /// UI stop a *paused* generation without unpausing the gate (which would
        /// desync the pause button) — the parked worker is woken by `wake()`,
        /// sees the cancel here, and exits.
        canceled,
    };

    /// Loop-boundary checkpoint. Fast path: returns `.proceed` immediately when
    /// not paused. While paused, blocks (no CPU spin) until one of: the cancel
    /// flag goes true (`.canceled`), an unload is requested (`.unload`), or the
    /// gate is unpaused (`.proceed`). `cancel` is the same flag the loop polls
    /// itself — passing it lets a cancel land on a *parked* worker.
    pub fn checkpoint(self: *Gate, io: Io, cancel: ?*std.atomic.Value(bool)) Outcome {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        while (self.paused) {
            if (cancel) |c| if (c.load(.acquire)) return .canceled;
            if (self.want_unload) return .unload;
            self.cond.waitUncancelable(io, &self.mu);
        }
        return .proceed;
    }

    /// Park until unpaused, after a worker has handled a `.unload` (snapshotted
    /// its state and freed the model). Returns when the worker should reload and
    /// resume.
    pub fn awaitResume(self: *Gate, io: Io) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        while (self.paused) self.cond.waitUncancelable(io, &self.mu);
    }

    /// Request that workers park at their next boundary. Idempotent.
    pub fn pause(self: *Gate, io: Io) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        self.paused = true;
    }

    /// Release parked workers (both `checkpoint()` and `awaitResume()` waiters).
    /// Clears any pending unload request. Idempotent.
    pub fn unpause(self: *Gate, io: Io) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        self.paused = false;
        self.want_unload = false;
        self.cond.broadcast(io);
    }

    /// Ask a paused worker to snapshot + unload at its next boundary. Only takes
    /// effect while paused; a running worker never observes it (it will unpause
    /// first). Wakes a worker already parked in `checkpoint()`.
    pub fn requestUnload(self: *Gate, io: Io) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        self.want_unload = true;
        self.cond.broadcast(io);
    }

    /// Re-evaluate parked workers WITHOUT changing the pause state — used to
    /// deliver a cancel to a `checkpoint()` waiter (it will re-read its cancel
    /// flag and return `.canceled`). Unlike `unpause`, leaves `paused` set so the
    /// UI pause button stays in sync.
    pub fn wake(self: *Gate, io: Io) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        self.cond.broadcast(io);
    }

    pub fn isPaused(self: *Gate, io: Io) bool {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.paused;
    }

    /// Whether an unload has been requested (and not yet cleared by unpause).
    /// A worker reads this after its loop returns to tell a suspend from a
    /// normal completion.
    pub fn wantsUnload(self: *Gate, io: Io) bool {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.want_unload;
    }
};

test "gate: unpaused checkpoint proceeds without blocking" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var g: Gate = .{};
    try std.testing.expectEqual(Gate.Outcome.proceed, g.checkpoint(io, null));
    try std.testing.expect(!g.isPaused(io));
}

test "gate: paused worker parks and resumes on unpause" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var g: Gate = .{};
    g.pause(io);
    try std.testing.expect(g.isPaused(io));

    const Ctx = struct {
        gate: *Gate,
        io: Io,
        outcome: Gate.Outcome = undefined,
        fn run(c: *@This()) void {
            c.outcome = c.gate.checkpoint(c.io, null);
        }
    };
    var ctx: Ctx = .{ .gate = &g, .io = io };
    const t = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    // The worker is now (or soon) parked in checkpoint(); unpause releases it.
    // Whether unpause wins the race or the worker parks first, the outcome is
    // the same: it proceeds.
    g.unpause(io);
    t.join();
    try std.testing.expectEqual(Gate.Outcome.proceed, ctx.outcome);
}

test "gate: cancel wakes a parked worker with .canceled (pause stays set)" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var g: Gate = .{};
    var cancel = std.atomic.Value(bool).init(false);
    g.pause(io);

    const Ctx = struct {
        gate: *Gate,
        io: Io,
        cancel: *std.atomic.Value(bool),
        outcome: Gate.Outcome = undefined,
        fn run(c: *@This()) void {
            c.outcome = c.gate.checkpoint(c.io, c.cancel);
        }
    };
    var ctx: Ctx = .{ .gate = &g, .io = io, .cancel = &cancel };
    const t = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    cancel.store(true, .release);
    g.wake(io); // deliver the cancel without unpausing
    t.join();
    try std.testing.expectEqual(Gate.Outcome.canceled, ctx.outcome);
    // The gate is still paused: cancel didn't touch the pause state.
    try std.testing.expect(g.isPaused(io));
}

test "gate: unload request wakes a parked worker with .unload" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var g: Gate = .{};
    g.pause(io);

    const Ctx = struct {
        gate: *Gate,
        io: Io,
        outcome: Gate.Outcome = undefined,
        fn run(c: *@This()) void {
            c.outcome = c.gate.checkpoint(c.io, null);
        }
    };
    var ctx: Ctx = .{ .gate = &g, .io = io };
    const t = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    g.requestUnload(io);
    t.join();
    try std.testing.expectEqual(Gate.Outcome.unload, ctx.outcome);
}
