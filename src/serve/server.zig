//! The server half of the protocol over one link, engine-free: routes each
//! HTTP request to a `Backend` and, once a client upgrades `/v1/events`, pumps
//! that backend's outgoing frames down the socket until it closes. tp-serve
//! plugs the engine host in; a test plugs a fake in.
//!
//! Verbs: `GET /v1/hello` answers with the host's identity; `POST /v1/req`
//! carries one JSON `wire.Request` and is acknowledged before it runs (a
//! handler never blocks a control request behind engine work); `PUT /v1/upload`
//! carries a `BinHeader` and its payload; `GET /v1/events` is the one
//! server-to-client WebSocket. Pixels a client asked for travel on it too, as
//! binary frames: a client streams a frame of any size, and a 4 MiB image
//! costs a delta a few milliseconds on a local socket.
const std = @import("std");
const Io = std.Io;
const wire = @import("wire.zig");
const queue = @import("queue.zig");
const link_mod = @import("link.zig");
const blob = @import("blob.zig");
const ws = @import("ws.zig");
const Link = link_mod.Link;
const Frame = queue.Frame;

const log = std.log.scoped(.serve);

/// What the routing needs from whoever owns the engine. Every callback may run
/// on a connection thread; the backend queues, it does not touch the engine.
pub const Backend = struct {
    ctx: *anyopaque,
    /// The `hello` event as JSON, gpa-owned.
    hello: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator) anyerror![]u8,
    /// One JSON request; ownership of `bytes` passes.
    request: *const fn (ctx: *anyopaque, bytes: []u8) void,
    /// One binary upload; ownership of `payload` passes.
    upload: *const fn (ctx: *anyopaque, hdr: wire.BinHeader, payload: []u8) void,
    /// The outgoing frames' generation counter and a wait on it, so the events
    /// pump sleeps between bursts: read `outGen`, `take`, and if nothing came,
    /// `waitOut` on the value read, for at most `timeout_ns`. `wakeOut` ends
    /// such a wait with no new frame; the pump's reader thread calls it.
    outGen: *const fn (ctx: *anyopaque) u32,
    waitOut: *const fn (ctx: *anyopaque, seen: u32, timeout_ns: u64) void,
    wakeOut: *const fn (ctx: *anyopaque) void,
    take: *const fn (ctx: *anyopaque, out: *std.ArrayList(Frame)) void,
    /// One events client at a time. `attach` says whether this one may have it.
    eventsAttach: *const fn (ctx: *anyopaque) bool,
    eventsDetach: *const fn (ctx: *anyopaque) void,
    /// Where `/v1/blob/*` lands; null answers 404 (a host that takes no files).
    blobs: ?*blob.Store = null,
    /// Where `/v1/pull/*` lands: the files this host will hand BACK, described
    /// once and cached. Null answers 404 (a host that gives nothing back).
    offers: ?*blob.Offers = null,
    /// A catalog id text to a path on this host, copied into `buf` so a rescan
    /// cannot move the answer under the connection. Null refuses every pull.
    offerPath: ?*const fn (ctx: *anyopaque, id: []const u8, buf: []u8) ?[]const u8 = null,
    /// A committed file, by its final path: the host rescans.
    fileAdded: ?*const fn (ctx: *anyopaque, path: []const u8) void = null,
};

/// Caps on what a client may send in one request. A transcript adopt is the
/// largest JSON; an upload is a raw image.
pub const max_request_bytes: usize = 16 << 20;
pub const max_upload_bytes: usize = 256 << 20;

/// Connections a host serves at once; one past it is closed unserved.
pub const max_connections: u32 = 64;
/// How long a connection may take to finish its TLS handshake and deliver
/// its first request head before the watchdog shuts it down.
pub const first_head_timeout_ns: u64 = 10 * std.time.ns_per_s;
/// How long a request body may go without DELIVERING ANYTHING. On progress, not
/// on the whole body: a 256 MiB upload over a slow link is fine, a peer that
/// declares a length and then stops is not, and left unbounded it pins a
/// connection slot and its buffer for as long as it likes.
pub const body_stall_timeout_ns: u64 = 30 * std.time.ns_per_s;

pub const Options = struct {
    /// What an untrusted link's bearer must hash to; null admits nobody there.
    auth: ?*const link_mod.SecretHash = null,
    /// A refused secret waits this long before its 401, so a guess costs its
    /// sender more than it costs the host.
    unauthorized_delay_ns: u64 = std.time.ns_per_s,
    /// The events pump pings on this cadence whatever else it writes.
    ping_interval_ns: u64 = 15 * std.time.ns_per_s,
    /// Pings in a row with no pong before the client is taken for gone.
    max_unanswered_pings: u32 = 2,
    /// Disarmed once the first request head is in, re-armed around each body
    /// read (see `body_stall_timeout_ns`).
    watch: ?Watchdog.Armed = null,
};

const json_headers = [_]std.http.Header{.{ .name = "content-type", .value = "application/json" }};

/// Serve one connection until the client closes it or `stop` is set. An events
/// upgrade turns the connection into the frame pump and returns when it ends.
/// On a link that is not trusted every request must carry a bearer secret
/// hashing to `opts.auth`, or the connection ends with a 401.
pub fn serve(gpa: std.mem.Allocator, link: *Link, be: Backend, stop: *const std.atomic.Value(bool), opts: Options) !void {
    var http = std.http.Server.init(link.reader(), link.writer());
    var first = true;
    while (!stop.load(.acquire)) {
        var req = http.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing, error.HttpRequestTruncated, error.ReadFailed, error.HttpHeadersOversize => return,
            else => return err,
        };
        if (first) {
            first = false;
            if (opts.watch) |a| a.disarm();
        }
        if (!link.trusted and !authorized(&req, opts.auth)) {
            Io.sleep(link.io, .{ .nanoseconds = opts.unauthorized_delay_ns }, .real) catch {};
            return req.respond("", .{ .status = .unauthorized, .keep_alive = false });
        }
        const target = req.head.target;
        const method = req.head.method;
        if (method == .GET and std.mem.eql(u8, target, "/v1/hello")) {
            const body = try be.hello(be.ctx, gpa);
            defer gpa.free(body);
            try req.respond(body, .{ .extra_headers = &json_headers });
        } else if (method == .POST and std.mem.eql(u8, target, "/v1/req")) {
            const bytes = readBody(gpa, link.io, &req, max_request_bytes, opts.watch) catch |err| return respondErr(&req, err);
            be.request(be.ctx, bytes);
            try req.respond("", .{ .status = .accepted });
        } else if (method == .PUT and std.mem.eql(u8, target, "/v1/upload")) {
            const bytes = readBody(gpa, link.io, &req, max_upload_bytes, opts.watch) catch |err| return respondErr(&req, err);
            const f = queue.binFrameFromBytes(gpa, bytes) catch |err| {
                gpa.free(bytes);
                if (err == error.OutOfMemory) return req.respond("", .{ .status = .insufficient_storage, .keep_alive = false });
                return req.respond(@errorName(err), .{ .status = .bad_request, .keep_alive = false });
            };
            be.upload(be.ctx, f.bin.hdr, f.bin.payload);
            try req.respond("", .{ .status = .accepted });
        } else if (method == .GET and std.mem.eql(u8, target, "/v1/events")) {
            const k = switch (req.upgradeRequested()) {
                .websocket => |k| k orelse return req.respond("missing key", .{ .status = .bad_request, .keep_alive = false }),
                else => return req.respond("not an upgrade", .{ .status = .bad_request, .keep_alive = false }),
            };
            if (!be.eventsAttach(be.ctx)) return req.respond("events already attached", .{ .status = .conflict, .keep_alive = false });
            defer be.eventsDetach(be.ctx);
            var sock = try req.respondWebSocket(.{ .key = k });
            try sock.flush();
            return pumpEvents(gpa, link, &sock, be, stop, opts);
        } else if (std.mem.startsWith(u8, target, "/v1/pull/")) {
            // `try`, not `return`: a pull is many requests on one connection,
            // and returning here would serve exactly one of them.
            try servePull(gpa, link.io, &req, be);
        } else if (std.mem.startsWith(u8, target, "/v1/blob/")) {
            try serveBlob(gpa, link.io, &req, be, opts.watch);
        } else {
            try req.respond("", .{ .status = .not_found });
        }
    }
}

/// `/v1/blob/<digest hex>/{manifest,have,chunk/<i>,commit}` and `DELETE
/// /v1/blob/<digest hex>`, onto the backend's store.
/// A `blob.Store.Tick` that pushes the connection's watchdog deadline out, so a
/// chunk is bounded by how long it goes without DELIVERING anything rather than
/// by its size.
const BodyTick = struct {
    io: Io,
    watch: Watchdog.Armed,

    fn call(ctx: *anyopaque) void {
        const self: *BodyTick = @ptrCast(@alignCast(ctx));
        self.watch.rearm(nowNs(self.io) + body_stall_timeout_ns);
    }
};

fn serveBlob(gpa: std.mem.Allocator, io: Io, req: *std.http.Server.Request, be: Backend, watch: ?Watchdog.Armed) !void {
    const store = be.blobs orelse return req.respond("", .{ .status = .not_found });
    const rest = req.head.target["/v1/blob/".len..];
    if (rest.len < 64) return req.respond("", .{ .status = .not_found });
    const hex = rest[0..64];
    if (link_mod.parseSecretHex(hex) == null) return req.respond("", .{ .status = .not_found });
    const verb = if (rest.len > 65 and rest[64] == '/') rest[65..] else "";
    const method = req.head.method;

    if (method == .POST and std.mem.eql(u8, verb, "manifest")) {
        const json = readBody(gpa, io, req, blob.max_manifest_bytes, watch) catch |err| return respondErr(req, err);
        defer gpa.free(json);
        const have = store.begin(json) catch |err| return respondBlobErr(req, err);
        defer gpa.free(have);
        return req.respond(have, .{ .extra_headers = &json_headers });
    } else if (method == .GET and std.mem.eql(u8, verb, "have")) {
        const have = store.have(hex) catch |err| return respondBlobErr(req, err);
        defer gpa.free(have);
        return req.respond(have, .{ .extra_headers = &json_headers });
    } else if (method == .PUT and std.mem.startsWith(u8, verb, "chunk/")) {
        const index = std.fmt.parseInt(usize, verb["chunk/".len..], 10) catch return req.respond("", .{ .status = .not_found });
        const len = req.head.content_length orelse return req.respond("", .{ .status = .length_required, .keep_alive = false });
        var buf: [8192]u8 = undefined;
        const body = try req.readerExpectContinue(&buf);
        var bt: ?BodyTick = if (watch) |a| .{ .io = io, .watch = a } else null;
        const tick: ?blob.Store.Tick = if (bt) |*b| .{ .ctx = @ptrCast(b), .call = BodyTick.call } else null;
        defer if (watch) |a| a.disarm();
        store.writeChunk(hex, index, body, len, tick) catch |err| switch (err) {
            // The body was consumed: the connection stays usable.
            error.ChunkMismatch => return req.respond("chunk digest mismatch", .{ .status = .bad_request }),
            else => {
                // It was not: end the connection with the answer.
                const status = blobStatus(err);
                return req.respond(@errorName(err), .{ .status = status, .keep_alive = false });
            },
        };
        return req.respond("", .{ .status = .accepted });
    } else if (method == .POST and std.mem.eql(u8, verb, "commit")) {
        const path = store.commit(hex) catch |err| return respondBlobErr(req, err);
        defer gpa.free(path);
        if (be.fileAdded) |f| f(be.ctx, path);
        var out: [512]u8 = undefined;
        const body = std.fmt.bufPrint(&out, "{{\"name\":\"{s}\"}}", .{std.fs.path.basename(path)}) catch "{}";
        return req.respond(body, .{ .extra_headers = &json_headers });
    } else if (method == .DELETE and verb.len == 0) {
        store.abort(hex);
        return req.respond("", .{ .status = .no_content });
    }
    try req.respond("", .{ .status = .not_found });
}

/// `/v1/pull/<catalog id>/{manifest,chunk/<i>}`: the file this host holds
/// under that id, described and then read. The pulling client owns the
/// transfer, so nothing here is per-client state beyond the cached `Offer`.
fn servePull(gpa: std.mem.Allocator, io: Io, req: *std.http.Server.Request, be: Backend) !void {
    _ = gpa;
    const offers = be.offers orelse return req.respond("", .{ .status = .not_found });
    const resolve = be.offerPath orelse return req.respond("", .{ .status = .not_found });
    const rest = req.head.target["/v1/pull/".len..];
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return req.respond("", .{ .status = .not_found });
    const id = rest[0..slash];
    const verb = rest[slash + 1 ..];
    if (req.head.method != .GET) return req.respond("", .{ .status = .method_not_allowed });

    var path_buf: [4096]u8 = undefined;
    const path = resolve(be.ctx, id, &path_buf) orelse return req.respond("", .{ .status = .not_found });
    const offer = offers.acquire(path) catch |err| return req.respond(@errorName(err), .{
        .status = if (err == error.TooManyTransfers) .service_unavailable else .not_found,
    });
    defer offers.release(offer);

    if (std.mem.eql(u8, verb, "manifest")) {
        return req.respond(offer.man_json, .{ .extra_headers = &json_headers });
    }
    if (std.mem.startsWith(u8, verb, "chunk/")) {
        const index = std.fmt.parseInt(usize, verb["chunk/".len..], 10) catch return req.respond("", .{ .status = .not_found });
        const len = offer.chunkLen(index);
        if (len == 0) return req.respond("", .{ .status = .not_found });
        var send_buf: [64 << 10]u8 = undefined;
        var w = try req.respondStreaming(&send_buf, .{ .content_length = len });
        offer.streamChunk(io, index, &w.writer) catch {
            // The length is already promised, so the only honest end is to drop
            // the connection; the puller retries that chunk.
            return error.PullReadFailed;
        };
        try w.end();
        return;
    }
    try req.respond("", .{ .status = .not_found });
}

fn blobStatus(err: blob.StoreError) std.http.Status {
    return switch (err) {
        error.BadManifest, error.BadChunk, error.ChunkMismatch => .bad_request,
        error.InsufficientSpace => .insufficient_storage,
        error.TooManyTransfers => .service_unavailable,
        error.UnknownTransfer => .not_found,
        error.Incomplete, error.NameTaken => .conflict,
        error.DigestMismatch, error.NotAModel => .unprocessable_entity,
        error.OutOfMemory => .insufficient_storage,
        else => .internal_server_error,
    };
}

fn respondBlobErr(req: *std.http.Server.Request, err: blob.StoreError) !void {
    try req.respond(@errorName(err), .{ .status = blobStatus(err) });
}

/// The request carries `authorization: Bearer <secret hex>`, whose hash is
/// compared in constant time; a link with no secret configured admits nobody.
fn authorized(req: *const std.http.Server.Request, auth: ?*const link_mod.SecretHash) bool {
    const want = auth orelse return false;
    var it = req.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "authorization")) continue;
        const prefix = "Bearer ";
        if (!std.ascii.startsWithIgnoreCase(h.value, prefix)) return false;
        const got = link_mod.parseSecretHex(h.value[prefix.len..]) orelse return false;
        return std.crypto.timing_safe.eql(link_mod.SecretHash, link_mod.hashSecret(got), want.*);
    }
    return false;
}

const BodyError = error{ LengthRequired, PayloadTooLarge, OutOfMemory, ReadFailed, EndOfStream, HttpExpectationFailed, WriteFailed };

fn readBody(gpa: std.mem.Allocator, io: Io, req: *std.http.Server.Request, max: usize, watch: ?Watchdog.Armed) BodyError![]u8 {
    const len = req.head.content_length orelse return error.LengthRequired;
    if (len > max) return error.PayloadTooLarge;
    var buf: [8192]u8 = undefined;
    const br = try req.readerExpectContinue(&buf);
    const out = try gpa.alloc(u8, @intCast(len));
    errdefer gpa.free(out);
    // Re-armed per slice, so the deadline is on PROGRESS: the watchdog is what
    // wakes a read blocked on a peer that stopped talking.
    defer if (watch) |a| a.disarm();
    var off: usize = 0;
    while (off < out.len) {
        if (watch) |a| a.rearm(nowNs(io) + body_stall_timeout_ns);
        const n = try br.readSliceShort(out[off..]);
        if (n == 0) return error.EndOfStream;
        off += n;
    }
    return out;
}

fn respondErr(req: *std.http.Server.Request, err: BodyError) !void {
    const status: std.http.Status = switch (err) {
        error.LengthRequired => .length_required,
        error.PayloadTooLarge => .payload_too_large,
        error.OutOfMemory => .insufficient_storage,
        else => return err,
    };
    // The body was not consumed, so this connection cannot be reused.
    try req.respond("", .{ .status = status, .keep_alive = false });
}

/// The events socket's reading half, on its own thread: a client sends only
/// control frames here, and what matters is that it sends them at all. A pong
/// clears the unanswered count, a ping asks the pump for a pong, and a close,
/// EOF or error marks the client gone and wakes the pump to leave.
const Pump = struct {
    link: *Link,
    be: Backend,
    unanswered: std.atomic.Value(u32) = .init(0),
    pong_due: std.atomic.Value(bool) = .init(false),
    gone: std.atomic.Value(bool) = .init(false),

    fn readLoop(p: *Pump) void {
        var discard: Io.Writer.Discarding = .init(&.{});
        const r = p.link.reader();
        while (true) {
            const h = ws.readHeader(r) catch break;
            ws.streamPayload(r, h, &discard.writer) catch break;
            switch (h.opcode) {
                .pong => p.unanswered.store(0, .release),
                .ping => {
                    p.pong_due.store(true, .release);
                    p.be.wakeOut(p.be.ctx);
                },
                .connection_close => break,
                else => {},
            }
        }
        p.gone.store(true, .release);
        p.be.wakeOut(p.be.ctx);
    }
};

/// Write every frame the backend produces until the socket fails, the client
/// stops answering pings, or `stop`.
fn pumpEvents(gpa: std.mem.Allocator, link: *Link, sock: *std.http.Server.WebSocket, be: Backend, stop: *const std.atomic.Value(bool), opts: Options) !void {
    var pump: Pump = .{ .link = link, .be = be };
    const reader = try std.Thread.spawn(.{}, Pump.readLoop, .{&pump});
    defer {
        // A close does not wake a parked read; a shutdown does.
        link.shutdown();
        reader.join();
    }
    var frames: std.ArrayList(Frame) = .empty;
    defer {
        for (frames.items) |f| f.deinit(gpa);
        frames.deinit(gpa);
    }
    var last_ping = nowNs(link.io);
    while (!stop.load(.acquire) and !pump.gone.load(.acquire)) {
        const seen = be.outGen(be.ctx);
        be.take(be.ctx, &frames);
        while (frames.items.len > 0) {
            const f = frames.items[0];
            // Off the list before the write, so a failed write frees it once.
            _ = frames.orderedRemove(0);
            defer f.deinit(gpa);
            switch (f) {
                .text => |t| try sock.writeMessageUnflushed(t, .text),
                .bin => |b| {
                    const hb = queue.binHeaderBytes(b.hdr);
                    var vec = [_][]const u8{ &hb, b.payload };
                    try sock.writeMessageVecUnflushed(&vec, .binary);
                },
            }
        }
        if (pump.pong_due.swap(false, .acq_rel)) try ws.writeControl(link.writer(), .pong, "");
        const since_ping = nowNs(link.io) - last_ping;
        if (since_ping >= opts.ping_interval_ns) {
            if (pump.unanswered.load(.acquire) >= opts.max_unanswered_pings) return error.ClientUnresponsive;
            _ = pump.unanswered.fetchAdd(1, .acq_rel);
            try ws.writeControl(link.writer(), .ping, "");
            last_ping += since_ping;
        }
        try sock.flush();
        be.waitOut(be.ctx, seen, opts.ping_interval_ns -| (nowNs(link.io) - last_ping));
    }
}

/// Monotonic nanoseconds, for deadlines.
pub fn nowNs(io: Io) u64 {
    return @intCast(@max(0, Io.Clock.awake.now(io).nanoseconds));
}

/// Shuts down a connection that stops talking: a stalled TLS handshake, a peer
/// that connects and says nothing, or one that declares a body length and then
/// delivers nothing. A slot is held for the connection's WHOLE life and handed
/// back by `release` (BEFORE it closes the stream, or a sweep may shut down
/// whatever reuses the descriptor); `disarm` only stops watching, so the same
/// connection can go back under the watchdog for its next blocking read.
/// `sweep` runs on one thread for every connection, since a stream read cannot
/// take a timeout.
pub const Watchdog = struct {
    io: Io,
    mu: Io.Mutex = .init,
    slots: [max_connections]Slot = @splat(.{}),

    const Slot = struct {
        /// The connection holding this slot, null while it is free.
        stream: ?Io.net.Stream = null,
        /// When to shut that connection down; 0 while it is not being watched.
        deadline_ns: u64 = 0,
        gen: u32 = 0,
    };

    /// One connection's slot. Every call is idempotent, and a stale one (the
    /// slot has been released and taken by another connection) is a no-op.
    pub const Armed = struct {
        wd: *Watchdog,
        idx: u32,
        gen: u32,
        stream: Io.net.Stream,

        /// Stop watching, keeping the slot.
        pub fn disarm(a: Armed) void {
            a.set(0);
        }

        /// Watch again, with a fresh deadline.
        pub fn rearm(a: Armed, deadline_ns: u64) void {
            a.set(deadline_ns);
        }

        fn set(a: Armed, deadline_ns: u64) void {
            a.wd.mu.lockUncancelable(a.wd.io);
            defer a.wd.mu.unlock(a.wd.io);
            const s = &a.wd.slots[a.idx];
            if (s.gen == a.gen) s.deadline_ns = deadline_ns;
        }

        /// The connection is over: hand the slot back, before the stream closes.
        pub fn release(a: Armed) void {
            a.wd.mu.lockUncancelable(a.wd.io);
            defer a.wd.mu.unlock(a.wd.io);
            const s = &a.wd.slots[a.idx];
            if (s.gen != a.gen) return;
            s.stream = null;
            s.deadline_ns = 0;
        }
    };

    /// Null when every slot is taken, which the connection cap should prevent.
    pub fn arm(self: *Watchdog, stream: Io.net.Stream, deadline_ns: u64) ?Armed {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        for (&self.slots, 0..) |*s, i| {
            if (s.stream != null) continue;
            s.gen +%= 1;
            s.stream = stream;
            s.deadline_ns = deadline_ns;
            return .{ .wd = self, .idx = @intCast(i), .gen = s.gen, .stream = stream };
        }
        return null;
    }

    /// Shuts down every armed stream past `now_ns`; returns how many.
    pub fn sweep(self: *Watchdog, now_ns: u64) usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        var n: usize = 0;
        for (&self.slots) |*s| {
            const st = s.stream orelse continue;
            if (s.deadline_ns == 0 or s.deadline_ns > now_ns) continue;
            st.shutdown(self.io, .both) catch {};
            // Shut down once. The slot stays this connection's until its thread
            // notices the read failed and releases it.
            s.deadline_ns = 0;
            n += 1;
        }
        return n;
    }

    /// The sweep thread: once a second until `stop`.
    pub fn run(self: *Watchdog, stop: *const std.atomic.Value(bool)) void {
        while (!stop.load(.acquire)) {
            Io.sleep(self.io, .{ .nanoseconds = std.time.ns_per_s }, .real) catch {};
            const n = self.sweep(nowNs(self.io));
            if (n > 0) log.info("closed {d} connection(s) that stopped talking", .{n});
        }
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;
const httpc = @import("httpc.zig");

/// A backend with no engine: records requests, hands out whatever the test
/// queued.
const Fake = struct {
    gpa: std.mem.Allocator,
    io: Io,
    requests: std.ArrayList([]u8) = .empty,
    uploads: std.ArrayList(Frame) = .empty,
    mu: Io.Mutex = Io.Mutex.init,
    out: queue.Queue = .{},
    out_gen: std.atomic.Value(u32) = .init(0),
    attached: std.atomic.Value(bool) = .init(false),
    attach_count: std.atomic.Value(u32) = .init(0),
    blobs: ?*blob.Store = null,
    added: std.ArrayList([]u8) = .empty,
    offers: ?*blob.Offers = null,
    /// The one file this fake hands back, under the id "id:0000000000000001".
    offer_file: []const u8 = "",

    fn backend(self: *Fake) Backend {
        return .{
            .ctx = self,
            .hello = hello,
            .request = request,
            .upload = upload,
            .outGen = outGen,
            .waitOut = waitOut,
            .wakeOut = wakeOut,
            .take = take,
            .eventsAttach = eventsAttach,
            .eventsDetach = eventsDetach,
            .blobs = self.blobs,
            .fileAdded = fileAdded,
            .offers = self.offers,
            .offerPath = offerPath,
        };
    }
    fn offerPath(ctx: *anyopaque, id: []const u8, buf: []u8) ?[]const u8 {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        if (self.offer_file.len == 0 or self.offer_file.len > buf.len) return null;
        if (!std.mem.eql(u8, id, "id:0000000000000001")) return null;
        @memcpy(buf[0..self.offer_file.len], self.offer_file);
        return buf[0..self.offer_file.len];
    }
    fn fileAdded(ctx: *anyopaque, path: []const u8) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.added.append(self.gpa, self.gpa.dupe(u8, path) catch return) catch {};
    }
    fn hello(_: *anyopaque, gpa: std.mem.Allocator) anyerror![]u8 {
        return wire.encodeAlloc(gpa, wire.Event{ .hello = .{ .gen = 77 } });
    }
    fn request(ctx: *anyopaque, bytes: []u8) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.requests.append(self.gpa, bytes) catch self.gpa.free(bytes);
    }
    fn upload(ctx: *anyopaque, hdr: wire.BinHeader, payload: []u8) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.uploads.append(self.gpa, .{ .bin = .{ .hdr = hdr, .payload = payload } }) catch self.gpa.free(payload);
    }
    fn outGen(ctx: *anyopaque) u32 {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        return self.out_gen.load(.acquire);
    }
    fn waitOut(ctx: *anyopaque, seen: u32, timeout_ns: u64) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.io.futexWaitTimeout(u32, &self.out_gen.raw, seen, .{ .duration = .{ .raw = .{ .nanoseconds = timeout_ns }, .clock = .awake } }) catch {};
    }
    fn wakeOut(ctx: *anyopaque) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.wakeAll();
    }
    fn take(ctx: *anyopaque, out: *std.ArrayList(Frame)) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.out.drain(self.io, self.gpa, out);
    }
    fn eventsAttach(ctx: *anyopaque) bool {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        _ = self.attach_count.fetchAdd(1, .monotonic);
        return self.attached.cmpxchgStrong(false, true, .acq_rel, .acquire) == null;
    }
    fn eventsDetach(ctx: *anyopaque) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.attached.store(false, .release);
    }
    /// Queue a frame for the events pump and wake it.
    fn push(self: *Fake, f: Frame) !void {
        try self.out.push(self.io, self.gpa, f);
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, 1);
    }
    fn wakeAll(self: *Fake) void {
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }
    fn deinit(self: *Fake) void {
        for (self.requests.items) |r| self.gpa.free(r);
        self.requests.deinit(self.gpa);
        for (self.added.items) |p| self.gpa.free(p);
        self.added.deinit(self.gpa);
        for (self.uploads.items) |u| u.deinit(self.gpa);
        self.uploads.deinit(self.gpa);
        self.out.deinit(self.gpa);
    }
};

/// Accepts links until `finish`, each served on its own thread as tp-serve
/// does; a sequential acceptor would park the events upgrade behind the open
/// control connection. `finish` sets the stop flag, wakes the acceptor with one
/// more connection, and joins everything before the listener may close.
const Acceptor = struct {
    listener: *link_mod.Listener,
    gpa: std.mem.Allocator,
    be: Backend,
    stop: *std.atomic.Value(bool),
    opts: Options = .{},
    threads: [32]?std.Thread = .{null} ** 32,
    main: ?std.Thread = null,

    fn start(self: *Acceptor) !void {
        self.main = try std.Thread.spawn(.{}, run, .{self});
    }
    fn run(self: *Acceptor) void {
        for (&self.threads) |*slot| {
            const l = self.listener.accept(self.gpa) catch return;
            if (self.stop.load(.acquire)) {
                l.close(self.gpa);
                return;
            }
            slot.* = std.Thread.spawn(.{}, one, .{ self, l }) catch {
                l.close(self.gpa);
                return;
            };
        }
    }
    fn one(self: *Acceptor, l: *link_mod.Link) void {
        defer l.close(self.gpa);
        var opts = self.opts;
        opts.auth = self.listener.authHash();
        serve(self.gpa, l, self.be, self.stop, opts) catch {};
    }
    fn finish(self: *Acceptor, io: Io) void {
        self.stop.store(true, .release);
        if (link_mod.connect(self.gpa, io, self.listener.endpoint)) |w| w.close(self.gpa) else |_| {}
        if (self.main) |m| m.join();
        for (&self.threads) |*t| if (t.*) |th| {
            th.join();
            t.* = null;
        };
    }
};

test "hello, a request, an upload and two event frames cross a loopback link" {
    const io = testing.io;
    const gpa = testing.allocator;
    var fake: Fake = .{ .gpa = gpa, .io = io };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop };
    try acc.start();

    // Control connection: three requests on one keep-alive link.
    const control = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer control.close(gpa);
    {
        const r = try httpc.request(gpa, control, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 200), r.status);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const ev = try wire.decode(wire.Event, arena.allocator(), r.body);
        try testing.expectEqual(@as(wire.HostGen, 77), ev.hello.gen);
    }
    {
        const body = try wire.encodeAlloc(gpa, wire.Request{ .chat_submit = .{ .text = "hi" } });
        defer gpa.free(body);
        const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    {
        const hdr: wire.BinHeader = .{ .kind = .rgb_upload, .id = 0, .rev = 0, .w = 2, .h = 1, .len = 6 };
        var body: std.ArrayList(u8) = .empty;
        defer body.deinit(gpa);
        try body.appendSlice(gpa, &queue.binHeaderBytes(hdr));
        try body.appendSlice(gpa, &[_]u8{ 1, 2, 3, 4, 5, 6 });
        const r = try httpc.request(gpa, control, "PUT", "/v1/upload", "application/octet-stream", body.items, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    {
        const r = try httpc.request(gpa, control, "GET", "/v1/nothing", "", "", 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 404), r.status);
    }

    // Events connection: two frames pushed after the upgrade arrive in order.
    const events = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer events.close(gpa);
    try httpc.upgradeEvents(events, "/v1/events", &.{});
    try fake.push(.{ .text = try gpa.dupe(u8, "{\"turn_end\":{}}") });
    try fake.push(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 5, .rev = 1, .w = 1, .h = 1, .len = 4 }, .payload = try gpa.dupe(u8, &[_]u8{ 9, 8, 7, 6 }) } });
    {
        const h = try ws.readHeader(events.reader());
        try testing.expectEqual(ws.Opcode.text, h.opcode);
        var got: Io.Writer.Allocating = .init(gpa);
        defer got.deinit();
        try ws.streamPayload(events.reader(), h, &got.writer);
        try testing.expectEqualStrings("{\"turn_end\":{}}", got.written());
    }
    {
        const h = try ws.readHeader(events.reader());
        try testing.expectEqual(ws.Opcode.binary, h.opcode);
        var got: Io.Writer.Allocating = .init(gpa);
        defer got.deinit();
        try ws.streamPayload(events.reader(), h, &got.writer);
        const f = try queue.binFrameFromBytes(gpa, try got.toOwnedSlice());
        defer f.deinit(gpa);
        try testing.expectEqual(@as(u64, 5), f.bin.hdr.id);
        try testing.expectEqualSlices(u8, &.{ 9, 8, 7, 6 }, f.bin.payload);
    }
    // Stop the pump (the flag FIRST, then a wake: a parked pump is not woken
    // by the flag alone) and close both links, so every server thread returns
    // before it is joined.
    stop.store(true, .release);
    fake.wakeAll();
    control.close(gpa);
    events.close(gpa);
    acc.finish(io);

    try testing.expectEqual(@as(usize, 1), fake.requests.items.len);
    try testing.expect(std.mem.indexOf(u8, fake.requests.items[0], "\"chat_submit\"") != null);
    try testing.expectEqual(@as(usize, 1), fake.uploads.items.len);
    try testing.expectEqual(wire.BinHeader.Kind.rgb_upload, fake.uploads.items[0].bin.hdr.kind);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6 }, fake.uploads.items[0].bin.payload);
    try testing.expectEqual(@as(u32, 1), fake.attach_count.load(.acquire));
}

test "a second events client is refused with 409 and an oversize request with 413" {
    const io = testing.io;
    const gpa = testing.allocator;
    var fake: Fake = .{ .gpa = gpa, .io = io };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop };
    try acc.start();

    // The first holder never reads; it only has to hold the slot.
    fake.attached.store(true, .release);
    const second = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer second.close(gpa);
    try testing.expectError(error.EventsTaken, httpc.upgradeEvents(second, "/v1/events", &.{}));

    const control = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer control.close(gpa);
    // A head claiming more than the cap is refused before any body is read.
    const w = control.writer();
    try w.print("POST /v1/req HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {s}\r\ncontent-length: {d}\r\n\r\n", .{ &link_mod.secretHex(listener.endpoint.loopback.cookie), max_request_bytes + 1 });
    try w.flush();
    const h = try httpc.readHead(control.reader());
    try testing.expectEqual(@as(u16, 413), h.status);
    second.close(gpa);
    control.close(gpa);
    acc.finish(io);
}

test "a loopback link without the host's cookie gets 401 on every verb" {
    const io = testing.io;
    const gpa = testing.allocator;
    var fake: Fake = .{ .gpa = gpa, .io = io };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop, .opts = .{ .unauthorized_delay_ns = 1 } };
    try acc.start();

    // No header at all.
    const bare = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer bare.close(gpa);
    bare.auth = null;
    {
        const r = try httpc.request(gpa, bare, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 401), r.status);
    }
    // The wrong cookie, on the upgrade.
    var wrong = listener.endpoint;
    wrong.loopback.cookie[0] +%= 1;
    const forged = try link_mod.connect(gpa, io, wrong);
    errdefer forged.close(gpa);
    try testing.expectError(error.Unauthorized, httpc.upgradeEvents(forged, "/v1/events", &.{}));
    // The right one, taken from the endpoint as a client would.
    const good = try link_mod.connect(gpa, io, listener.endpoint);
    errdefer good.close(gpa);
    {
        const r = try httpc.request(gpa, good, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 200), r.status);
    }
    bare.close(gpa);
    forged.close(gpa);
    good.close(gpa);
    acc.finish(io);
    try testing.expectEqual(@as(u32, 0), fake.attach_count.load(.acquire));
}

/// The next frame on the events link, its payload dropped.
fn nextOpcode(events: *Link) !ws.Opcode {
    const h = try ws.readHeader(events.reader());
    var discard: Io.Writer.Discarding = .init(&.{});
    try ws.streamPayload(events.reader(), h, &discard.writer);
    return h.opcode;
}

test "an events client that never answers pings is let go, and the slot is free again" {
    const io = testing.io;
    const gpa = testing.allocator;
    var fake: Fake = .{ .gpa = gpa, .io = io };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop, .opts = .{ .ping_interval_ns = 20 * std.time.ns_per_ms } };
    try acc.start();

    const events = try link_mod.connect(gpa, io, listener.endpoint);
    try httpc.upgradeEvents(events, "/v1/events", &.{});
    // Two pings, then the host gives up on us: the read hits EOF.
    try testing.expectEqual(ws.Opcode.ping, try nextOpcode(events));
    try testing.expectEqual(ws.Opcode.ping, try nextOpcode(events));
    try testing.expectError(error.EndOfStream, nextOpcode(events));
    events.close(gpa);
    // Our EOF comes from the pump's shutdown, which runs before the detach, so
    // an early retry may still see 409; a 409 closes its connection, so every
    // try is a fresh one.
    var tries: u32 = 0;
    const again = while (true) : (tries += 1) {
        const c = try link_mod.connect(gpa, io, listener.endpoint);
        if (httpc.upgradeEvents(c, "/v1/events", &.{})) |_| break c else |err| {
            c.close(gpa);
            try testing.expect(err == error.EventsTaken and tries < 24);
            Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .real) catch {};
        }
    };
    stop.store(true, .release);
    fake.wakeAll();
    again.close(gpa);
    acc.finish(io);
    try testing.expect(fake.attach_count.load(.acquire) >= 2);
}

test "an events client that answers pings stays attached and its own ping is answered" {
    const io = testing.io;
    const gpa = testing.allocator;
    var fake: Fake = .{ .gpa = gpa, .io = io };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop, .opts = .{ .ping_interval_ns = 40 * std.time.ns_per_ms } };
    try acc.start();

    const events = try link_mod.connect(gpa, io, listener.endpoint);
    try httpc.upgradeEvents(events, "/v1/events", &.{});
    // Answer five pings, more than the two that would end an unanswered client.
    for (0..5) |_| {
        try testing.expectEqual(ws.Opcode.ping, try nextOpcode(events));
        try ws.writePong(events.writer(), "", .{ 1, 2, 3, 4 });
        try events.writer().flush();
    }
    // Our own ping comes back as a pong, with a frame pushed meanwhile still delivered.
    try ws.writeFrame(events.writer(), .ping, "hb", .{ 4, 3, 2, 1 });
    try events.writer().flush();
    try fake.push(.{ .text = try gpa.dupe(u8, "{\"turn_end\":{}}") });
    var saw_pong = false;
    var saw_text = false;
    while (!saw_pong or !saw_text) {
        switch (try nextOpcode(events)) {
            .pong => saw_pong = true,
            .text => saw_text = true,
            .ping => {
                try ws.writePong(events.writer(), "", .{ 1, 2, 3, 4 });
                try events.writer().flush();
            },
            else => return error.UnexpectedFrame,
        }
    }
    // Still attached: a second client is refused.
    const second = try link_mod.connect(gpa, io, listener.endpoint);
    try testing.expectError(error.EventsTaken, httpc.upgradeEvents(second, "/v1/events", &.{}));
    stop.store(true, .release);
    fake.wakeAll();
    second.close(gpa);
    events.close(gpa);
    acc.finish(io);
    try testing.expectEqual(@as(u32, 2), fake.attach_count.load(.acquire));
}

test "the watchdog wakes a reader blocked on a connection past its deadline and spares one disarmed" {
    const io = testing.io;
    const gpa = testing.allocator;
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var wd: Watchdog = .{ .io = io };

    const Side = struct {
        fn readUntilGone(side_io: Io, stream: Io.net.Stream, saw_end: *std.atomic.Value(bool)) void {
            var buf: [64]u8 = undefined;
            var r = stream.reader(side_io, &buf);
            _ = r.interface.takeByte() catch {};
            saw_end.store(true, .release);
        }
    };
    // Two silent clients; the server side arms both, disarms one.
    const c1 = try link_mod.connect(gpa, io, listener.endpoint);
    defer c1.close(gpa);
    const s1 = try listener.acceptStream();
    defer s1.close(io);
    const c2 = try link_mod.connect(gpa, io, listener.endpoint);
    defer c2.close(gpa);
    const s2 = try listener.acceptStream();
    defer s2.close(io);
    const a1 = wd.arm(s1, 100).?;
    const a2 = wd.arm(s2, 100).?;
    a2.disarm();
    a2.disarm();
    var ended: std.atomic.Value(bool) = .init(false);
    const th = try std.Thread.spawn(.{}, Side.readUntilGone, .{ io, s1, &ended });

    try testing.expectEqual(@as(usize, 0), wd.sweep(99));
    try testing.expect(!ended.load(.acquire));
    try testing.expectEqual(@as(usize, 1), wd.sweep(100));
    th.join();
    try testing.expect(ended.load(.acquire));
    // The slot was released by the sweep; a stale disarm changes nothing.
    a1.disarm();
    try testing.expectEqual(@as(usize, 0), wd.sweep(1000));
    // c2 was spared: its socket still carries bytes.
    try c2.writer().writeAll("x");
    try c2.writer().flush();
    var buf: [1]u8 = undefined;
    var r2 = s2.reader(io, &buf);
    try testing.expectEqual(@as(u8, 'x'), try r2.interface.takeByte());
}

test "a connection keeps its watchdog slot while disarmed, and can be watched again" {
    const io = testing.io;
    const gpa = testing.allocator;
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var wd: Watchdog = .{ .io = io };

    const c1 = try link_mod.connect(gpa, io, listener.endpoint);
    defer c1.close(gpa);
    const s1 = try listener.acceptStream();
    defer s1.close(io);
    const c2 = try link_mod.connect(gpa, io, listener.endpoint);
    defer c2.close(gpa);
    const s2 = try listener.acceptStream();
    defer s2.close(io);

    const a1 = wd.arm(s1, 100).?;
    a1.disarm(); // its first request head is in
    // The next connection gets a slot of its own: taking this one would make
    // every later rearm here a no-op, which is the body timeout going silent.
    const a2 = wd.arm(s2, 100).?;
    try testing.expect(a1.idx != a2.idx);
    // A body read puts the first connection back under the watchdog.
    a1.rearm(50);
    try testing.expectEqual(@as(usize, 2), wd.sweep(100));
    // Shut down once, not once a second, and the slot is the connection's
    // until it hands it back.
    try testing.expectEqual(@as(usize, 0), wd.sweep(1000));
    a1.release();
    const a3 = wd.arm(s1, 100).?;
    try testing.expectEqual(a1.idx, a3.idx);
}

/// The sender's connector over a loopback listener.
const LoopConnector = struct {
    gpa: std.mem.Allocator,
    io: Io,
    ep: link_mod.Endpoint,
    fn connect(ctx: *anyopaque) anyerror!*Link {
        const self: *LoopConnector = @ptrCast(@alignCast(ctx));
        return link_mod.connect(self.gpa, self.io, self.ep);
    }
};

test "a model crosses the link in chunks, survives a cut mid-chunk, and lands in the store as a scanned file" {
    const io = testing.io;
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const store_dir = try std.fs.path.join(gpa, &.{ root, "models" });
    defer gpa.free(store_dir);
    var store = try blob.Store.init(gpa, io, store_dir);
    defer store.deinit();
    var fake: Fake = .{ .gpa = gpa, .io = io, .blobs = &store };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop };
    try acc.start();

    // 11 chunks of 64 KiB, cut after 3.5 of them.
    const cb: u64 = blob.min_chunk_bytes;
    const total = try blob.writeTestModel(io, tmp.dir, "sent.safetensors", 10 * @as(usize, @intCast(cb)) + 12344, 11);
    const src = try std.fs.path.join(gpa, &.{ root, "sent.safetensors" });
    defer gpa.free(src);
    var sender = try blob.Sender.init(gpa, io, src, .{ .chunk_bytes = cb, .cut_after_bytes = 3 * cb + cb / 2 });
    defer sender.deinit();
    var conn: LoopConnector = .{ .gpa = gpa, .io = io, .ep = listener.endpoint };
    var prog: blob.Progress = .{};
    sender.run(.{ .ctx = &conn, .connect = LoopConnector.connect }, &prog);
    errdefer std.debug.print("send ended {t}: {s}\n", .{ prog.phase.load(.acquire), prog.errName() });
    try testing.expectEqual(blob.Phase.done, prog.phase.load(.acquire));
    try testing.expectEqual(@as(u32, 2), prog.connects.load(.acquire));
    const chunks = prog.chunks_total.load(.acquire);
    // One connection lost one chunk in flight: at most that chunk went twice.
    try testing.expect(prog.puts.load(.acquire) <= chunks + 1);
    try testing.expectEqual(@as(u64, total), prog.bytes_done.load(.acquire));

    const final = try std.fs.path.join(gpa, &.{ store_dir, "sent.safetensors" });
    defer gpa.free(final);
    const a = try Io.Dir.cwd().readFileAlloc(io, src, gpa, .limited(4 << 20));
    defer gpa.free(a);
    const b = try Io.Dir.cwd().readFileAlloc(io, final, gpa, .limited(4 << 20));
    defer gpa.free(b);
    try testing.expectEqualSlices(u8, a, b);
    try testing.expectEqual(@as(usize, 1), fake.added.items.len);
    try testing.expectEqualStrings(final, fake.added.items[0]);

    // A stranger to the store: 404 for a digest nobody announced.
    const c = try link_mod.connect(gpa, io, listener.endpoint);
    const r = try httpc.request(gpa, c, "GET", "/v1/blob/" ++ ("ab" ** 32) ++ "/have", "", "", 4096);
    defer gpa.free(r.body);
    try testing.expectEqual(@as(u16, 404), r.status);
    fake.wakeAll();
    c.close(gpa);
    acc.finish(io);
}

test "a model comes BACK over the link, checked the same way, and a file the host does not offer is a 404" {
    const io = testing.io;
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);

    // What the host holds, and where the puller puts what it gets.
    const cb: u64 = blob.min_chunk_bytes;
    const total = try blob.writeTestModel(io, tmp.dir, "held.safetensors", 6 * @as(usize, @intCast(cb)) + 4096, 23);
    const src = try std.fs.path.join(gpa, &.{ root, "held.safetensors" });
    defer gpa.free(src);
    const dest_dir = try std.fs.path.join(gpa, &.{ root, "incoming" });
    defer gpa.free(dest_dir);
    try Io.Dir.cwd().createDirPath(io, dest_dir);

    var offers = blob.Offers.init(gpa, io);
    offers.chunk_bytes = cb;
    defer offers.deinit();
    var fake: Fake = .{ .gpa = gpa, .io = io, .offers = &offers, .offer_file = src };
    defer fake.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = fake.backend(), .stop = &stop };
    try acc.start();

    var dest = try blob.Store.init(gpa, io, dest_dir);
    defer dest.deinit();
    var fetcher = try blob.Fetcher.init(gpa, io, &dest, "id:0000000000000001", .{});
    defer fetcher.deinit();
    var conn: LoopConnector = .{ .gpa = gpa, .io = io, .ep = listener.endpoint };
    var prog: blob.Progress = .{};
    fetcher.run(.{ .ctx = &conn, .connect = LoopConnector.connect }, &prog);
    errdefer std.debug.print("pull ended {t}: {s}\n", .{ prog.phase.load(.acquire), prog.errName() });
    try testing.expectEqual(blob.Phase.done, prog.phase.load(.acquire));
    try testing.expectEqual(@as(u64, total), prog.bytes_done.load(.acquire));

    // The file that landed is the file the host holds, byte for byte, under the
    // name the host called it: a puller names nothing itself.
    const final = try std.fs.path.join(gpa, &.{ dest_dir, "held.safetensors" });
    defer gpa.free(final);
    const want = try Io.Dir.cwd().readFileAlloc(io, src, gpa, .limited(4 << 20));
    defer gpa.free(want);
    const got = try Io.Dir.cwd().readFileAlloc(io, final, gpa, .limited(4 << 20));
    defer gpa.free(got);
    try testing.expectEqualSlices(u8, want, got);

    // An id this host does not hold hands back nothing, rather than an empty file.
    const c = try link_mod.connect(gpa, io, listener.endpoint);
    const r = try httpc.request(gpa, c, "GET", "/v1/pull/id:00000000000000ff/manifest", "", "", 4096);
    defer gpa.free(r.body);
    try testing.expectEqual(@as(u16, 404), r.status);
    fake.wakeAll();
    c.close(gpa);
    acc.finish(io);
}
