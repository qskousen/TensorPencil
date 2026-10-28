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
//! `zig build chat-probe -- --config <path> --complete [--think] [--seed N]`
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
//!
//! `--complete` checks `Session.complete` leaves the chat alone: a chat with a
//! completion run between its turns must continue byte-identically, prefilling no
//! more, than the same chat without one, and the completion must answer the same
//! in an empty session, mid-chat and after it. The chat-only run is done twice
//! first; when those two disagree the model is not deterministic here and the check
//! exits 2. Completions are compared only at the same layer placement: a dynamic
//! split moves layers to the host as the chat's KV grows, and host layers compute
//! differently.

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
    var complete = false;
    var think = false;
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
        } else if (std.mem.eql(u8, a, "--complete")) {
            complete = true;
        } else if (std.mem.eql(u8, a, "--think")) {
            think = true;
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
    if (complete) {
        const ok = try runCompleteCheck(arena, gpa, io, opts, seed, think, out);
        try out.flush();
        if (ok != .pass) std.process.exit(if (ok == .inconclusive) 2 else 1);
        return;
    }
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

const Verdict = enum { pass, fail, inconclusive };

/// What one chat run ended on: the last two replies, the ids, and how much the
/// third turn prefilled.
const ChatRun = struct {
    turn3: u64,
    regen: u64,
    ids: u64,
    turn3_prefill: usize,
    regen_prefill: usize,
};

fn lastReply(s: *chat.Session) struct { hash: u64, prefill: usize } {
    const n = s.messages.items.len;
    if (n == 0 or s.messages.items[n - 1].role != .assistant) return .{ .hash = 0, .prefill = 0 };
    const v = s.messages.items[n - 1].activeConst();
    return .{ .hash = std.hash.Wyhash.hash(0, v.text.items), .prefill = v.stats.prefill_tokens };
}

/// Three turns and a regenerate, with `side` run after turn 2 when given.
fn chatRun(s: *chat.Session, io: std.Io, out: *std.Io.Writer, seed: u64, side: ?chat.Completion, side_text: *?Answer) !ChatRun {
    _ = s.reset();
    s.reseed(seed);
    try runTurn(s, io, out, "turn 1", "Name three rivers in Europe.");
    try runTurn(s, io, out, "turn 2", "Which of them is the longest?");
    if (side) |c| {
        const before = s.ctxTokens();
        side_text.* = try runCompletion(s, io, out, c);
        if (s.ctxTokens() != before) {
            try out.print("[complete] FAIL: context {d} -> {d} tokens across the completion\n", .{ before, s.ctxTokens() });
            side_text.* = null;
        }
    }
    try runTurn(s, io, out, "turn 3", "Say that in one short sentence.");
    const t3 = lastReply(s);
    try s.regenerate();
    settle(s, io);
    try report(s, out, "regenerate");
    const rg = lastReply(s);
    return .{
        .turn3 = t3.hash,
        .regen = rg.hash,
        .ids = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(s.ids.items)),
        .turn3_prefill = t3.prefill,
        .regen_prefill = rg.prefill,
    };
}

/// A completion's answer hash and the host layer count it ran with. The split
/// migrates in a fixed order, so the count names the placement.
const Answer = struct { hash: u64, host_layers: usize };

fn runCompletion(s: *chat.Session, io: std.Io, out: *std.Io.Writer, c: chat.Completion) !?Answer {
    const gpa = s.gpa;
    const host_layers = s.llmResidency().cpu;
    try s.complete(c);
    while (s.busy()) std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
    s.poll();
    var r = s.takeCompletion() orelse {
        try out.print("[complete] no result\n", .{});
        return null;
    };
    defer r.deinit(gpa);
    if (r.err) |e| {
        try out.print("[complete] error {t}\n", .{e});
        return null;
    }
    const h = std.hash.Wyhash.hash(0, r.text);
    try out.print("[complete] {x:0>16} ({d} B, {d} host layers) {f}\n", .{ h, r.text.len, host_layers, std.zig.fmtString(r.text[0..@min(r.text.len, 160)]) });
    try out.flush();
    return .{ .hash = h, .host_layers = host_layers };
}

fn runCompleteCheck(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, opts: chat.Options, seed: u64, think: bool, out: *std.Io.Writer) !Verdict {
    const s = try chat.Session.init(arena, gpa, io, noopWake, opts);
    defer s.deinit();
    const c: chat.Completion = .{
        .system = "You write prompts for an image model. Expand the idea you are given into one vivid, specific image prompt of two or three sentences. Reply with the prompt only.",
        .user = "adventurer returning home",
        .think = .{ .on = think },
        .seed = 7,
        .max_tokens = if (think) 1536 else 256,
    };

    const fresh = try runCompletion(s, io, out, c);
    var none: ?Answer = null;
    const base_a = try chatRun(s, io, out, seed, null, &none);
    const base_b = try chatRun(s, io, out, seed, null, &none);
    var mid: ?Answer = null;
    const with = try chatRun(s, io, out, seed, c, &mid);
    const after = try runCompletion(s, io, out, c);

    if (!std.meta.eql(base_a, base_b)) {
        try out.print("[complete] INCONCLUSIVE: the chat-only run differs from itself ({any} vs {any})\n", .{ base_a, base_b });
        return .inconclusive;
    }
    if (fresh == null or mid == null or after == null) {
        try out.print("[complete] FAIL: a completion produced no answer\n", .{});
        return .fail;
    }
    var ok = true;
    if (!std.meta.eql(base_a, with)) {
        try out.print("[complete] FAIL: chat without {any}\n[complete]       chat with    {any}\n", .{ base_a, with });
        ok = false;
    }
    // Each pair differs in the chat state the completion ran beside.
    const answers = [_]Answer{ fresh.?, mid.?, after.? };
    const names = [_][]const u8{ "empty session", "mid-chat", "after the chat" };
    var compared: usize = 0;
    for (0..answers.len) |i| for (i + 1..answers.len) |j| {
        if (answers[i].host_layers != answers[j].host_layers) continue;
        compared += 1;
        if (answers[i].hash != answers[j].hash) {
            try out.print("[complete] FAIL: completion {s} {x} vs {s} {x}, both with {d} host layers\n", .{ names[i], answers[i].hash, names[j], answers[j].hash, answers[i].host_layers });
            ok = false;
        }
    };
    if (compared == 0) {
        try out.print("[complete] INCONCLUSIVE: no two completions ran at the same layer placement\n", .{});
        return if (ok) .inconclusive else .fail;
    }
    if (ok) try out.print("[complete] PASS: turn 3 prefilled {d} tok, regenerate {d}, chat identical; {d} completion pairs identical\n", .{ with.turn3_prefill, with.regen_prefill, compared });
    return if (ok) .pass else .fail;
}
