//! A `simple`-mode test runner that reports how long each test took, so the
//! slow ones in the gated suite can be found instead of guessed at.
//! Selected by `zig build test -Dtest-timing`; the normal runs use Zig's own
//! runner, which talks the build-system server protocol.
//!
//! Prints one line per test as it finishes (so a hang is attributable) and a
//! slowest-first summary at the end. Exit code is nonzero if any test failed,
//! which is how a simple runner reports failure.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const Status = enum { pass, skip, fail };

const Result = struct {
    name: []const u8,
    ns: u64,
    status: Status,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    // page_allocator, not a checked one: this runner's own bookkeeping is not
    // what is under test, and a leak check here would report the tests' leaks
    // as the runner's.
    const gpa = std.heap.page_allocator;

    var results: std.ArrayList(Result) = .empty;
    defer results.deinit(gpa);

    var failed: usize = 0;
    var skipped: usize = 0;
    for (builtin.test_functions) |t| {
        // Per-test std.testing setup, the same the stock runner does: without it
        // `testing.io`'s thread pool has no allocator and the first threaded
        // matmul segfaults inside Io.Threaded.groupAsync.
        testing.allocator_instance = .{};
        testing.io_instance = .init(testing.allocator, .{
            .argv0 = .init(init.minimal.args),
            .environ = init.minimal.environ,
        });
        testing.log_level = .warn;
        testing.environ = init.minimal.environ;
        defer {
            testing.io_instance.deinit();
            _ = testing.allocator_instance.deinit();
        }

        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        const outcome: Status = if (t.func()) |_|
            .pass
        else |err| switch (err) {
            error.SkipZigTest => .skip,
            else => blk: {
                std.debug.print("FAIL {s}: {s}\n", .{ t.name, @errorName(err) });
                break :blk .fail;
            },
        };
        const ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);
        switch (outcome) {
            .fail => failed += 1,
            .skip => skipped += 1,
            .pass => {},
        }
        try results.append(gpa, .{ .name = t.name, .ns = ns, .status = outcome });
    }

    const by_time = try gpa.dupe(Result, results.items);
    defer gpa.free(by_time);
    std.mem.sort(Result, by_time, {}, struct {
        fn gt(_: void, a: Result, b: Result) bool {
            return a.ns > b.ns;
        }
    }.gt);

    var total: u64 = 0;
    for (results.items) |r| total += r.ns;

    std.debug.print("\n=== slowest tests ({d} total, {d:.1}s wall in tests) ===\n", .{
        results.items.len, @as(f64, @floatFromInt(total)) / 1e9,
    });
    for (by_time) |r| {
        const ms = @as(f64, @floatFromInt(r.ns)) / 1e6;
        if (ms < 50) break; // everything below this is noise against a 5-minute goal
        std.debug.print("{d:>9.1} ms  {s}{s}\n", .{
            ms, r.name, if (r.status == .skip) "  [skipped]" else "",
        });
    }
    std.debug.print("=== {d} passed, {d} skipped, {d} failed ===\n", .{
        results.items.len - skipped - failed, skipped, failed,
    });

    if (failed != 0) std.process.exit(1);
}
