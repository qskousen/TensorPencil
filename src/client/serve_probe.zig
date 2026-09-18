//! `zig build serve-probe -- --config <file> [--message <text>] [--image <prompt>]
//! [--kill-mid-turn [<deltas>]] [--timeout <s>] [--socket <path>]`: what
//! `driver-probe` does, through a tp-serve process over the socket. Without
//! `--socket` it spawns `tp-serve` (the sibling binary this build installed) as
//! tp-gui will, hands it the config, and prints every event as one JSON line.
//! Exits non-zero on an error event, a failed image, or the timeout.
//!
//! `--kill-mid-turn` is tp-gui's reconnect, end to end: after that many deltas
//! of the reply the host is killed, a second one is spawned and handed the
//! transcript this client holds, the message is sent again, and the run
//! passes only if the second host's transcript starts with the first host's
//! partial reply byte for byte.
const std = @import("std");
const Io = std.Io;
const config = @import("shared").config;
const wire = @import("serve").wire;
const link = @import("serve").link;
const Remote = @import("client").remote.Remote;
const Frame = @import("client").remote.Frame;
const Mirror = @import("client").mirror.Mirror;

fn noop() void {}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var cfg_path: ?[]const u8 = null;
    var message: ?[]const u8 = null;
    var image: ?[]const u8 = null;
    var socket: ?[]const u8 = null;
    var kill_after: ?u32 = null;
    var timeout_s: u64 = 900;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--config") and i + 1 < args.len) {
            i += 1;
            cfg_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--message") and i + 1 < args.len) {
            i += 1;
            message = args[i];
        } else if (std.mem.eql(u8, args[i], "--image") and i + 1 < args.len) {
            i += 1;
            image = args[i];
        } else if (std.mem.eql(u8, args[i], "--socket") and i + 1 < args.len) {
            i += 1;
            socket = args[i];
        } else if (std.mem.eql(u8, args[i], "--kill-mid-turn")) {
            kill_after = 5;
            if (i + 1 < args.len) if (std.fmt.parseInt(u32, args[i + 1], 10)) |n| {
                kill_after = n;
                i += 1;
            } else |_| {};
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout_s = try std.fmt.parseInt(u64, args[i], 10);
        } else {
            std.debug.print("usage: serve-probe --config <file> [--message <text>] [--image <prompt>] [--kill-mid-turn [<deltas>]] [--timeout <s>] [--socket <path>]\n", .{});
            return error.BadArgs;
        }
    }
    const path = cfg_path orelse {
        std.debug.print("serve-probe: --config is required (never the real settings file)\n", .{});
        return error.BadArgs;
    };
    if (kill_after != null and (message == null or socket != null)) {
        std.debug.print("serve-probe: --kill-mid-turn needs --message and a spawned host (no --socket)\n", .{});
        return error.BadArgs;
    }

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const out = &stdout.interface;

    var cfg = config.Config.load(io, gpa, init.environ_map, path);
    cfg.applyFraming();
    // The host's share of the settings, the push tp-gui makes at startup.
    const settings_json = try std.json.Stringify.valueAlloc(gpa, config.hostSettings(&cfg), .{});
    defer gpa.free(settings_json);

    // A private socket for the spawned child, so a probe never talks to a
    // host the user is sitting at, and the probe's OWN config, so the child
    // reads the scratch settings and caches its catalog index beside them
    // rather than in the user's config directory.
    var sock_buf: [128]u8 = undefined;
    const sock_path = socket orelse try std.fmt.bufPrint(&sock_buf, "/tmp/tp-serve-probe-{d}.sock", .{Io.Clock.real.now(io).toSeconds()});
    const child_args = [_][]const u8{ "--config", path };
    var remote = if (socket != null)
        try Remote.connect(gpa, io, .{ .unix = sock_path }, noop)
    else
        try Remote.spawn(gpa, io, sock_path, &child_args, noop);
    defer remote.deinit();
    try out.print("{{\"probe\":\"connected\",\"gen\":{d}}}\n", .{remote.gen});

    remote.postRequest(.{ .settings = .{ .json = settings_json } });
    remote.postRequest(.snapshot);
    if (message) |m| remote.postRequest(.{ .chat_submit = .{ .text = m } });
    if (image) |p| remote.postRequest(.{ .img_enqueue = .{
        .client_ref = 1,
        .prompt = p,
        .width = @intCast(cfg.width),
        .height = @intCast(cfg.height),
        .steps = @intCast(cfg.steps),
        .seed = 1234,
        .from_studio = true,
    } });

    // The client's copy of the host, as tp-gui keeps one; what a reconnect
    // hands back.
    var mirror = Mirror.init(gpa);
    defer mirror.deinit();

    var want_turn = message != null;
    var want_image = image != null;
    var image_id: wire.ImageId = 0;
    var want_pixels = false;
    var failed = false;
    var deltas: u32 = 0;
    var killed = false;
    var reconnected = false;
    var partial: ?[]u8 = null;
    defer if (partial) |p| gpa.free(p);
    var frames: std.ArrayList(Frame) = .empty;
    defer frames.deinit(gpa);
    const deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;

    while (want_turn or want_image or want_pixels) {
        if (Io.Clock.real.now(io).nanoseconds > deadline) {
            try out.writeAll("{\"probe\":\"timeout\"}\n");
            try out.flush();
            return error.Timeout;
        }
        if (remote.failed.load(.acquire)) {
            if (!killed or reconnected) {
                try out.writeAll("{\"probe\":\"host lost\"}\n");
                try out.flush();
                return error.HostLost;
            }
            // The killed host is gone: the reconnect, as tp-gui does it.
            const last = &mirror.messages.items[mirror.messages.items.len - 1];
            partial = try gpa.dupe(u8, last.active().text.items);
            mirror.hostRestarted();
            remote.deinit();
            remote = try Remote.spawn(gpa, io, sock_path, &child_args, noop);
            reconnected = true;
            try out.print("{{\"probe\":\"respawned\",\"gen\":{d},\"partial_bytes\":{d}}}\n", .{ remote.gen, partial.?.len });
            var ar = std.heap.ArenaAllocator.init(gpa);
            defer ar.deinit();
            remote.postRequest(.{ .settings = .{ .json = settings_json } });
            remote.postRequest(.{ .chat_adopt = .{ .messages = try mirror.toWire(ar.allocator()) } });
            remote.postRequest(.snapshot);
            remote.postRequest(.{ .chat_submit = .{ .text = message.? } });
            continue;
        }
        remote.take(&frames);
        for (frames.items) |f| {
            defer f.deinit(gpa);
            switch (f) {
                .text => |t| {
                    try out.writeAll(t);
                    try out.writeByte('\n');
                    mirror.apply(.{ .text = t });
                    var ar = std.heap.ArenaAllocator.init(gpa);
                    defer ar.deinit();
                    const ev = wire.decode(wire.Event, ar.allocator(), t) catch continue;
                    switch (ev) {
                        .turn_end => want_turn = false,
                        .err => failed = true,
                        .delta => {
                            deltas += 1;
                            if (kill_after) |k| if (!killed and deltas >= k) {
                                killed = true;
                                try out.writeAll("{\"probe\":\"killing host\"}\n");
                                if (remote.child) |*c| c.kill(io);
                            };
                        },
                        .img => |im| if (im.client_ref == 1) {
                            image_id = im.id;
                            switch (im.status) {
                                .done => if (want_image) {
                                    want_image = false;
                                    want_pixels = true;
                                    remote.postRequest(.{ .img_fetch = .{ .image = im.id, .kind = .pixels } });
                                },
                                .failed, .canceled => {
                                    want_image = false;
                                    failed = true;
                                },
                                else => {},
                            }
                        },
                        else => {},
                    }
                },
                .bin => |b| {
                    try out.print("{{\"bin\":{{\"kind\":\"{t}\",\"id\":{d},\"rev\":{d},\"w\":{d},\"h\":{d},\"len\":{d}}}}}\n", .{
                        b.hdr.kind, b.hdr.id, b.hdr.rev, b.hdr.w, b.hdr.h, b.payload.len,
                    });
                    if (b.hdr.kind == .image_rgba and b.hdr.id == image_id) want_pixels = false;
                },
            }
        }
        try out.flush();
        Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .real) catch {};
    }
    if (kill_after != null) {
        // Two turns on the second host's transcript: the adopted one, cut where
        // the first host died, and the fresh one.
        const msgs = mirror.messages.items;
        const kept = msgs.len == 4 and std.mem.eql(u8, msgs[1].active().text.items, partial.?);
        try out.print("{{\"probe\":\"reconnect\",\"messages\":{d},\"partial_kept\":{}}}\n", .{ msgs.len, kept });
        if (!kept) failed = true;
    }
    try out.flush();
    if (failed) return error.ProbeFailed;
}
