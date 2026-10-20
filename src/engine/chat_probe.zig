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
//! `zig build chat-probe -- --config <path> --scenario [--seed N]`
//!
//! `--repeat N` re-runs the messages N times, resetting between: same ids, same
//! sampling seeds, same everything, so a result that varies across repeats is a
//! per-call divergence rather than process state. `reset` alone does NOT rewind
//! the seed stream (see `Session.reseed`), and without that every repeat differs
//! at any temperature above 0 and the probe proves nothing.
//! Point `--config` at a COPY of the settings file.
//!
//! `--scenario` walks every turn shape the app drives through one session (follow-up
//! turns, regenerate, switching back to an earlier variant, a KV dtype rebuild, a
//! model reload carrying the transcript, reset, and an image turn where the model has
//! vision) and prints a hash of the reply and of the session's ids after each. Two
//! builds that print the same lines ran the same tokens.

const std = @import("std");
const config = @import("shared").config;
const chat = @import("engine").chat;

fn noopWake() void {}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var cfg_path: ?[]const u8 = null;
    var seed: u64 = 42; // fixed, unlike the GUI's clock seed, so runs compare
    var repeat: usize = 1;
    var scenario = false;
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
        } else if (std.mem.eql(u8, a, "--scenario")) {
            scenario = true;
        } else if (std.mem.eql(u8, a, "--repeat") and has_next) {
            i += 1;
            repeat = std.fmt.parseInt(usize, args[i], 10) catch 1;
        }
    }
    if (messages.items.len == 0) try messages.append(arena, "hi");

    var buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.Writer.initStreaming(.stdout(), io, &buf);
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
    if (scenario) return runScenario(arena, gpa, io, opts, seed, out);
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

fn settle(s: *chat.Session, io: std.Io) void {
    while (true) {
        s.poll();
        if (!s.busy() and !s.turnPending()) break;
        std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
    }
    s.poll();
}

fn report(s: *chat.Session, out: *std.Io.Writer, step: []const u8) !void {
    const n = s.messages.items.len;
    const text: []const u8 = if (n > 0 and s.messages.items[n - 1].role == .assistant) s.messages.items[n - 1].activeConst().text.items else "";
    try out.print("[scenario] {s:<16} reply {x:0>16} ({d} B)  ids {x:0>16} ({d})\n", .{
        step,                                                       std.hash.Wyhash.hash(0, text), text.len,
        std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(s.ids.items)), s.ids.items.len,
    });
    const shown = text[0..@min(text.len, 90)];
    try out.print("[text] {s:<16} {f}\n", .{ step, std.zig.fmtString(shown) });
    try out.flush();
}

fn runTurn(s: *chat.Session, io: std.Io, out: *std.Io.Writer, step: []const u8, text: []const u8) !void {
    try s.submit(text);
    settle(s, io);
    try report(s, out, step);
}

fn runScenario(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, opts: chat.Options, seed: u64, out: *std.Io.Writer) !void {
    var s = try chat.Session.init(arena, gpa, io, noopWake, opts);
    defer s.deinit();
    try runTurn(s, io, out, "turn 1", "Name three rivers in Europe.");
    try runTurn(s, io, out, "turn 2", "Which of them is the longest?");
    try runTurn(s, io, out, "turn 3", "Say that in one short sentence.");
    try s.regenerate();
    settle(s, io);
    try report(s, out, "regenerate");
    s.selectVariant(0);
    settle(s, io);
    try report(s, out, "variant 0");
    try runTurn(s, io, out, "after variant", "Now name a river in Asia.");
    try s.rebuildContext(.f16);
    try runTurn(s, io, out, "kv f16", "And one in Africa.");

    const msgs = s.detachTranscript();
    s.deinit();
    s = try chat.Session.init(arena, gpa, io, noopWake, opts);
    try s.adoptTranscript(msgs);
    try runTurn(s, io, out, "after reload", "Which continent has the most rivers?");

    _ = s.reset();
    s.reseed(seed);
    try runTurn(s, io, out, "reset turn 1", "Name three rivers in Europe.");

    if (s.visionEnabled()) {
        // A deterministic picture, so no file decides the hash: a red-to-blue gradient.
        const w = 224;
        const h = 224;
        const rgb = try arena.alloc(u8, w * h * 3);
        for (0..h) |y| for (0..w) |x| {
            const px = rgb[(y * w + x) * 3 ..][0..3];
            px.* = .{ @intCast(255 * x / w), @intCast(255 * y / h), 128 };
        };
        try s.attachImage(rgb, w, h);
        try runTurn(s, io, out, "image", "Describe the colours in this picture.");
        const n = s.messages.items.len;
        try out.print("[image reply] {s}\n", .{s.messages.items[n - 1].activeConst().text.items});
    } else try out.print("[scenario] image            (no vision tower)\n", .{});
}
