//! `zig build driver-probe -- --config <file> [--message <text>] [--image <prompt>]
//! [--timeout <s>] [--concurrent]`: run the engine host on this thread, feed it
//! wire requests, and print every event it emits as one JSON line.
//! `--concurrent` has three threads hammer the inbox with harmless requests
//! (snapshots, meter moves, fetches of nothing) for the whole run, which is
//! what the engine-thread asserts and the queue locks are for. What a tp-serve client
//! would see, with no socket and no window; the way to look at the protocol's
//! behaviour and at the engine's, headlessly.
//!
//! A message runs until its `turn_end`; an image until its `img` reports a
//! terminal status, after which its pixels are fetched once and the binary
//! frame's header is printed. Exits non-zero on an error event, a failed image,
//! or the timeout.
const std = @import("std");
const Io = std.Io;
const config = @import("shared").config;
const wire = @import("serve").wire;
const host_mod = @import("engine").host;

fn noop() void {}

/// Three of these post from three threads while the engine runs a turn.
const Hammer = struct {
    host: *host_mod.Host,
    io: Io,
    stop: std.atomic.Value(bool) = .init(false),
    posted: std.atomic.Value(u64) = .init(0),

    fn run(self: *Hammer) void {
        var n: u64 = 0;
        while (!self.stop.load(.acquire)) : (n += 1) {
            switch (n % 4) {
                0 => self.host.postRequest(.{ .meter = .{ .split = 0.5 + 0.1 * @as(f32, @floatFromInt(n % 3)), .limit = 0.95 } }),
                1 => self.host.postRequest(.{ .img_fetch = .{ .image = 0 } }),
                2 => self.host.postRequest(.{ .chat_select_variant = .{ .msg = 9999, .variant = 0 } }),
                else => if (n % 40 == 3) self.host.postRequest(.snapshot) else self.host.postRequest(.{ .img_cancel = .{ .image = 0 } }),
            }
            _ = self.posted.fetchAdd(1, .monotonic);
            Io.sleep(self.io, .{ .nanoseconds = std.time.ns_per_ms }, .real) catch {};
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var cfg_path: ?[]const u8 = null;
    var message: ?[]const u8 = null;
    var image: ?[]const u8 = null;
    var timeout_s: u64 = 900;
    var concurrent = false;
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
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout_s = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--concurrent")) {
            concurrent = true;
        } else {
            std.debug.print("usage: driver-probe --config <file> [--message <text>] [--image <prompt>] [--timeout <s>] [--concurrent]\n", .{});
            return error.BadArgs;
        }
    }
    const path = cfg_path orelse {
        std.debug.print("driver-probe: --config is required (never the real settings file)\n", .{});
        return error.BadArgs;
    };

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const out = &stdout.interface;

    var cfg = config.Config.load(io, gpa, init.environ_map, path);
    cfg.applyFraming();

    const host = try gpa.create(host_mod.Host);
    defer gpa.destroy(host);
    host.init(gpa, io, &cfg, .{}, noop);
    defer host.deinit();

    host.postRequest(.{ .hello = .{} });
    host.postRequest(.snapshot);
    if (message) |m| host.postRequest(.{ .chat_submit = .{ .text = m } });
    if (image) |p| host.postRequest(.{ .img_enqueue = .{
        .client_ref = 1,
        .prompt = p,
        .width = @intCast(cfg.width),
        .height = @intCast(cfg.height),
        .steps = @intCast(cfg.steps),
        .seed = 1234,
        .from_studio = true,
    } });

    var hammer: Hammer = .{ .host = host, .io = io };
    var hammer_threads: [3]?std.Thread = .{ null, null, null };
    if (concurrent) for (&hammer_threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Hammer.run, .{&hammer});
    };
    defer {
        hammer.stop.store(true, .release);
        for (hammer_threads) |t| if (t) |th| th.join();
        if (concurrent) std.debug.print("{{\"probe\":\"concurrent\",\"posted\":{d}}}\n", .{hammer.posted.load(.acquire)});
    }

    var want_turn = message != null;
    var want_image = image != null;
    var image_id: wire.ImageId = 0;
    var want_pixels = false;
    var failed = false;
    var frames: std.ArrayList(host_mod.Frame) = .empty;
    defer frames.deinit(gpa);
    const deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;

    while (want_turn or want_image or want_pixels) {
        if (Io.Clock.real.now(io).nanoseconds > deadline) {
            try out.writeAll("{\"probe\":\"timeout\"}\n");
            try out.flush();
            return error.Timeout;
        }
        host.runOnce();
        host.take(&frames);
        for (frames.items) |f| {
            defer f.deinit(gpa);
            switch (f) {
                .text => |t| {
                    try out.writeAll(t);
                    try out.writeByte('\n');
                    var ar = std.heap.ArenaAllocator.init(gpa);
                    defer ar.deinit();
                    const ev = wire.decode(wire.Event, ar.allocator(), t) catch continue;
                    switch (ev) {
                        .turn_end => want_turn = false,
                        .err => failed = true,
                        .img => |im| if (im.client_ref == 1) {
                            image_id = im.id;
                            switch (im.status) {
                                .done => if (want_image) {
                                    want_image = false;
                                    want_pixels = true;
                                    host.postRequest(.{ .img_fetch = .{ .image = im.id, .kind = .pixels } });
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
    // One more pass so the final state lands in the log.
    host.runOnce();
    host.take(&frames);
    for (frames.items) |f| {
        defer f.deinit(gpa);
        if (f == .text) {
            try out.writeAll(f.text);
            try out.writeByte('\n');
        }
    }
    try out.flush();
    if (failed) return error.ProbeFailed;
}
