//! The client's end of a host over the socket: the same `post` / `take`
//! surface as the in-process `Host`, so a mirror never knows which it reads.
//! Requests queue here and a sender thread carries them to the control link
//! in order, so the caller never waits on a round trip; an events link,
//! upgraded to a WebSocket, is read by one thread here into a queue the mirror
//! drains. A local host that is not running is spawned as a child,
//! `--autospawn`, holding its stdin so it leaves when we do.
const std = @import("std");
const Io = std.Io;
const wire = @import("serve").wire;
const link = @import("serve").link;
const httpc = @import("serve").httpc;
const ws = @import("serve").ws;
const queue = @import("serve").queue;

const log = std.log.scoped(.remote);

pub const Frame = queue.Frame;

pub const Remote = struct {
    gpa: std.mem.Allocator,
    io: Io,
    control: *link.Link,
    events: *link.Link,
    inbox: queue.Queue = .{},
    outbox: queue.Queue = .{},
    out_wake: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),
    gen: wire.HostGen = 0,
    /// Called after frames land in the inbox, so the client can repaint.
    on_out: *const fn () void,
    reader: ?std.Thread = null,
    sender: ?std.Thread = null,
    /// A request or the events socket failed; the host is gone for us.
    failed: std.atomic.Value(bool) = .init(false),
    child: ?std.process.Child = null,

    /// Connect to a listening host: hello on the control link, then the events
    /// upgrade and the two threads. Blocks for the round trips; callers run
    /// it off the frame thread.
    pub fn connect(gpa: std.mem.Allocator, io: Io, ep: link.Endpoint, on_out: *const fn () void) !*Remote {
        const control = try link.connect(gpa, io, ep);
        errdefer control.close(gpa);
        const r = try gpa.create(Remote);
        errdefer gpa.destroy(r);
        r.* = .{ .gpa = gpa, .io = io, .control = control, .events = undefined, .on_out = on_out };
        r.gen = try hello(gpa, control);

        r.events = try link.connect(gpa, io, ep);
        errdefer r.events.close(gpa);
        try httpc.upgradeEvents(r.events, "/v1/events", &.{});
        r.sender = try std.Thread.spawn(.{}, senderMain, .{r});
        errdefer r.stopSender();
        r.reader = try std.Thread.spawn(.{}, readerMain, .{r});
        return r;
    }

    /// Reach `ep` and no further: the hello exchange, then close. What a
    /// settings form asks of a host that is not in force yet.
    pub fn check(gpa: std.mem.Allocator, io: Io, ep: link.Endpoint) !void {
        const control = try link.connect(gpa, io, ep);
        defer control.close(gpa);
        _ = try hello(gpa, control);
    }

    fn hello(gpa: std.mem.Allocator, control: *link.Link) !wire.HostGen {
        const resp = try httpc.request(gpa, control, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(resp.body);
        if (resp.status == 401) return error.Unauthorized;
        if (resp.status != 200) return error.HelloRefused;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const ev = try wire.decode(wire.Event, arena.allocator(), resp.body);
        if (ev != .hello) return error.HelloRefused;
        if (ev.hello.proto != wire.proto) {
            log.err("host speaks protocol {d}, this client {d}", .{ ev.hello.proto, wire.proto });
            return error.ProtoMismatch;
        }
        return ev.hello.gen;
    }

    /// Spawn `tp-serve` beside our own executable and connect to it. `extra`
    /// is appended to its command line (a `--config` for a probe).
    pub fn spawn(gpa: std.mem.Allocator, io: Io, socket_path: []const u8, extra: []const []const u8, on_out: *const fn () void) !*Remote {
        const dir = try std.process.executableDirPathAlloc(io, gpa);
        defer gpa.free(dir);
        const exe = try std.fs.path.join(gpa, &.{ dir, "tp-serve" });
        defer gpa.free(exe);
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        try argv.appendSlice(gpa, &.{ exe, "--autospawn", "--socket", socket_path });
        try argv.appendSlice(gpa, extra);
        var child = try std.process.spawn(io, .{
            .argv = argv.items,
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        errdefer child.kill(io);

        // The child says where it listens before anything else.
        var buf: [512]u8 = undefined;
        var out = child.stdout.?.reader(io, &buf);
        const line = out.interface.takeDelimiterInclusive('\n') catch return error.ChildDidNotStart;
        if (!std.mem.startsWith(u8, line, "ready ")) return error.ChildDidNotStart;
        const ep = link.Endpoint.parse(line["ready ".len..]) orelse return error.ChildDidNotStart;
        const r = try connect(gpa, io, ep, on_out);
        r.child = child;
        return r;
    }

    /// The local host: the one listening at the default path, else a fresh child.
    pub fn connectOrSpawn(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map, extra: []const []const u8, on_out: *const fn () void) !*Remote {
        const path = try link.defaultLocalPath(gpa, io, env);
        defer gpa.free(path);
        if (connect(gpa, io, .{ .unix = path }, on_out)) |r| return r else |err| {
            log.info("no host at {s} ({t}); spawning one", .{ path, err });
        }
        return spawn(gpa, io, path, extra, on_out);
    }

    pub fn deinit(self: *Remote) void {
        const gpa = self.gpa;
        self.stopSender();
        // An autospawned child leaves when its events client is gone, and
        // shutting our end of events down is what ends it; it also unblocks
        // the reader, which a plain close would not.
        self.control.close(gpa);
        self.events.shutdown();
        if (self.reader) |t| t.join();
        self.events.close(gpa);
        // A child already killed was reaped by `kill` (its id is null), and
        // `wait` asserts on that.
        if (self.child) |*c| if (c.id != null) {
            if (c.stdin) |f| f.close(self.io);
            c.stdin = null;
            _ = c.wait(self.io) catch {};
        };
        self.inbox.deinit(gpa);
        self.outbox.deinit(gpa);
        gpa.destroy(self);
    }

    // ── Requests ──────────────────────────────────────────────────────────────

    /// Queue `f` for the sender thread. Takes ownership either way.
    pub fn post(self: *Remote, f: Frame) void {
        if (self.failed.load(.acquire)) return f.deinit(self.gpa);
        self.outbox.push(self.io, self.gpa, f) catch return f.deinit(self.gpa);
        self.wakeSender();
    }

    pub fn postRequest(self: *Remote, req: wire.Request) void {
        const bytes = wire.encodeAlloc(self.gpa, req) catch |err| {
            log.err("encode request: {t}", .{err});
            return;
        };
        self.post(.{ .text = bytes });
    }

    fn wakeSender(self: *Remote) void {
        _ = self.out_wake.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_wake.raw, 1);
    }

    fn stopSender(self: *Remote) void {
        const t = self.sender orelse return;
        self.stopping.store(true, .release);
        self.wakeSender();
        t.join();
        self.sender = null;
    }

    fn senderMain(self: *Remote) void {
        var batch: std.ArrayList(Frame) = .empty;
        defer batch.deinit(self.gpa);
        while (true) {
            // Read the counter before draining, so a post that lands between
            // the drain and the wait is not slept through.
            const seen = self.out_wake.load(.acquire);
            self.outbox.drain(self.io, self.gpa, &batch);
            if (batch.items.len == 0) {
                if (self.stopping.load(.acquire)) return;
                self.io.futexWaitUncancelable(u32, &self.out_wake.raw, seen);
                continue;
            }
            for (batch.items) |f| {
                self.sendFrame(f);
                f.deinit(self.gpa);
            }
            batch.clearRetainingCapacity();
        }
    }

    fn sendFrame(self: *Remote, f: Frame) void {
        switch (f) {
            .text => |t| self.send("POST", "/v1/req", "application/json", t),
            .bin => |b| {
                var body: std.ArrayList(u8) = .empty;
                defer body.deinit(self.gpa);
                body.appendSlice(self.gpa, &queue.binHeaderBytes(b.hdr)) catch return;
                body.appendSlice(self.gpa, b.payload) catch return;
                self.send("PUT", "/v1/upload", "application/octet-stream", body.items);
            },
        }
    }

    fn send(self: *Remote, method: []const u8, target: []const u8, ct: []const u8, body: []const u8) void {
        if (self.failed.load(.acquire)) return;
        const resp = httpc.request(self.gpa, self.control, method, target, ct, body, 1 << 16) catch |err| {
            log.err("{s} {s}: {t}", .{ method, target, err });
            self.failed.store(true, .release);
            self.on_out();
            return;
        };
        defer self.gpa.free(resp.body);
        if (resp.status != 202) log.warn("{s} {s}: status {d}", .{ method, target, resp.status });
    }

    /// Move every received frame into `out` (client thread). The caller frees them.
    pub fn take(self: *Remote, out: *std.ArrayList(Frame)) void {
        self.inbox.drain(self.io, self.gpa, out);
    }

    // ── Events ────────────────────────────────────────────────────────────────

    fn readerMain(self: *Remote) void {
        const r = self.events.reader();
        while (true) {
            const h = ws.readHeader(r) catch break;
            if (h.len > (1 << 31)) break;
            const bytes = self.gpa.alloc(u8, @intCast(h.len)) catch break;
            var w: Io.Writer = .fixed(bytes);
            ws.streamPayload(r, h, &w) catch {
                self.gpa.free(bytes);
                break;
            };
            const f: Frame = switch (h.opcode) {
                .text => .{ .text = bytes },
                .binary => queue.binFrameFromBytes(self.gpa, bytes) catch {
                    self.gpa.free(bytes);
                    continue;
                },
                .ping => {
                    const ok = self.pong(bytes);
                    self.gpa.free(bytes);
                    if (!ok) break;
                    continue;
                },
                .connection_close => {
                    self.gpa.free(bytes);
                    break;
                },
                else => {
                    self.gpa.free(bytes);
                    continue;
                },
            };
            self.inbox.push(self.io, self.gpa, f) catch {
                f.deinit(self.gpa);
                break;
            };
            self.on_out();
        }
        self.failed.store(true, .release);
        self.on_out();
    }

    /// Only this thread writes the events link after the upgrade.
    fn pong(self: *Remote, payload: []const u8) bool {
        var mask: [4]u8 = undefined;
        self.io.random(&mask);
        const w = self.events.writer();
        ws.writeFrame(w, .pong, payload, mask) catch return false;
        w.flush() catch return false;
        return true;
    }
};
