//! chat-probe: drive a tp-gui `chat.Session` from the command line, with no
//! window and no dvui, and print the raw reply.
//!
//! The session is UI-independent, but a bug that only reproduces there needs the
//! real window to observe, which takes over the screen and the GPU. This builds
//! the session from the same config the app uses (`chat.sessionOptions`), sends
//! each `--message` as its own turn, and dumps what the model generated
//! including the reasoning markers the display splits out.
//!
//! `zig build chat-probe -- --config <path> [--message <text>]... [--seed N] [--repeat N]`
//!
//! `--repeat N` re-runs the messages N times, resetting between: same ids, same
//! sampling seeds, same everything, so a result that varies across repeats is a
//! per-call divergence rather than process state. `reset` alone does NOT rewind
//! the seed stream (see `Session.reseed`), and without that every repeat differs
//! at any temperature above 0 and the probe proves nothing.
//! Point `--config` at a COPY of the settings file.

const std = @import("std");
const config = @import("config.zig");
const chat = @import("chat.zig");

fn noopWake() void {}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var cfg_path: ?[]const u8 = null;
    var seed: u64 = 42; // fixed, unlike the GUI's clock seed, so runs compare
    var repeat: usize = 1;
    var messages: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const has_next = i + 1 < args.len;
        if (std.mem.eql(u8, a, "--config") and has_next) {
            i += 1;
            cfg_path = args[i];
        } else if (std.mem.eql(u8, a, "--message") and has_next) {
            i += 1;
            try messages.append(arena, args[i]);
        } else if (std.mem.eql(u8, a, "--seed") and has_next) {
            i += 1;
            seed = std.fmt.parseInt(u64, args[i], 10) catch seed;
        } else if (std.mem.eql(u8, a, "--repeat") and has_next) {
            i += 1;
            repeat = std.fmt.parseInt(usize, args[i], 10) catch 1;
        }
    }
    if (messages.items.len == 0) try messages.append(arena, "hi");

    var buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.Writer.init(.stdout(), io, &buf);
    const out = &stdout_w.interface;
    // Every exit path, not just the one that prints a reply: a run where every
    // turn failed would otherwise buffer its diagnostics and exit 0 in silence,
    // which reads as "the probe never ran".
    defer out.flush() catch {};

    var cfg = config.Config.load(io, gpa, init.environ_map, cfg_path);
    const opts = chat.sessionOptions(arena, &cfg, seed) catch {
        try out.print("no llm_model in config\n", .{});
        return out.flush();
    };
    const s = try chat.Session.init(arena, gpa, io, noopWake, opts);
    defer s.deinit();

    for (0..repeat) |rep| {
        if (rep > 0) {
            _ = s.reset(); // idle between repeats, so it always takes
            s.reseed(seed);
        }
        for (messages.items, 1..) |message, turn| {
            try s.submit(message);
            while (true) {
                s.poll();
                if (!s.busy() and !s.turnPending()) break;
                std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
            }
            s.poll();

            const n = s.messages.items.len;
            if (n == 0 or s.messages.items[n - 1].role != .assistant) {
                try out.print("[probe] run {d} turn {d}: no assistant reply\n", .{ rep, turn });
                continue;
            }
            const v = s.messages.items[n - 1].activeConst();
            try out.print(
                \\[probe] run {d} turn {d}: {d} bytes, thought_len={d}, thought_primed={}
                \\[probe] ---- RAW REPLY ----
                \\{s}
                \\[probe] ---- END ----
                \\
            , .{ rep, turn, v.text.items.len, chat.Session.thoughtLen(v), v.thought_primed, v.text.items });
            try out.flush();
        }
    }
}
