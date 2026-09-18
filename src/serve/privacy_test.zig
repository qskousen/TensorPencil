//! The canary suite: the check behind "tp-serve keeps none of what you type".
//!
//! A string nobody else would produce is driven through the whole protocol as
//! a prompt, an image request and a reply, against a stub backend; then every
//! file the host could have written is searched for those bytes. It runs in
//! the fast suite because the protocol layer is engine-free by construction,
//! so no GPU and no model are involved.
//!
//! The walk is checked for TEETH in the same test: one that visits nothing
//! passes exactly like a clean one, so a run that plants the canary must find
//! it. Three things this cannot see, and which the receipt in
//! `src/serve_main.zig`'s module doc covers with a real daemon instead: a
//! write outside the directories the test hands the host, anything the real
//! pipeline does once a model is loaded, and what reaches the log (a test
//! binary's root module is its test runner, so `std_options` declared here
//! would never bind).
const std = @import("std");
const Io = std.Io;
const server = @import("server.zig");
const link_mod = @import("link.zig");
const httpc = @import("httpc.zig");
const ws = @import("ws.zig");
const queue = @import("queue.zig");
const wire = @import("wire.zig");
const blob = @import("blob.zig");

const testing = std.testing;

/// Distinct enough that a hit is never a coincidence, and short enough to sit
/// inside a prompt a model would answer.
const canary = "CANARY-7f3a91-PROMPT";
const reply_canary = "CANARY-7f3a91-REPLY";

/// Every file under `dir`, recursively, searched for `needle`. Returns the
/// path that held it (caller frees) so a failure names the file.
fn treeHolds(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, needle: []const u8) !?[]u8 {
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var files: usize = 0;
    while (try walker.next(io)) |ent| {
        if (ent.kind != .file) continue;
        files += 1;
        const bytes = ent.dir.readFileAlloc(io, ent.basename, gpa, .limited(8 << 20)) catch continue;
        defer gpa.free(bytes);
        if (std.mem.indexOf(u8, bytes, needle) != null) return try gpa.dupe(u8, ent.path);
    }
    return null;
}

/// A host that holds the request in memory and answers with a reply, writing
/// nothing: what tp-serve's backend does with the engine behind it.
const Stub = struct {
    gpa: std.mem.Allocator,
    io: Io,
    out: queue.Queue = .{},
    out_gen: std.atomic.Value(u32) = .init(0),
    attached: std.atomic.Value(bool) = .init(false),
    /// The last request's bytes, held as an engine would hold a prompt.
    held: std.ArrayList(u8) = .empty,
    blobs: ?*blob.Store = null,

    fn backend(self: *Stub) server.Backend {
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
        };
    }
    fn hello(_: *anyopaque, gpa: std.mem.Allocator) anyerror![]u8 {
        return wire.encodeAlloc(gpa, wire.Event{ .hello = .{ .gen = 1 } });
    }
    fn request(ctx: *anyopaque, bytes: []u8) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        defer self.gpa.free(bytes);
        self.held.clearRetainingCapacity();
        self.held.appendSlice(self.gpa, bytes) catch {};
        // Answer as a turn would: the reply is content too, and it leaves
        // through the socket and nowhere else.
        const ev = wire.encodeAlloc(self.gpa, wire.Event{
            .delta = .{ .msg = 1, .variant = 0, .text = reply_canary },
        }) catch return;
        self.push(.{ .text = ev }) catch self.gpa.free(ev);
    }
    fn upload(ctx: *anyopaque, _: wire.BinHeader, payload: []u8) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        self.gpa.free(payload);
    }
    fn outGen(ctx: *anyopaque) u32 {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        return self.out_gen.load(.acquire);
    }
    fn waitOut(ctx: *anyopaque, seen: u32, timeout_ns: u64) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        self.io.futexWaitTimeout(u32, &self.out_gen.raw, seen, .{ .duration = .{ .raw = .{ .nanoseconds = timeout_ns }, .clock = .awake } }) catch {};
    }
    fn wakeOut(ctx: *anyopaque) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        self.wakeAll();
    }
    fn take(ctx: *anyopaque, out: *std.ArrayList(queue.Frame)) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        self.out.drain(self.io, self.gpa, out);
    }
    fn eventsAttach(ctx: *anyopaque) bool {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        return self.attached.cmpxchgStrong(false, true, .acq_rel, .acquire) == null;
    }
    fn eventsDetach(ctx: *anyopaque) void {
        const self: *Stub = @ptrCast(@alignCast(ctx));
        self.attached.store(false, .release);
    }
    fn push(self: *Stub, f: queue.Frame) !void {
        try self.out.push(self.io, self.gpa, f);
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }
    fn wakeAll(self: *Stub) void {
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }
    fn deinit(self: *Stub) void {
        self.held.deinit(self.gpa);
        self.out.deinit(self.gpa);
    }
};

const Acceptor = struct {
    listener: *link_mod.Listener,
    gpa: std.mem.Allocator,
    be: server.Backend,
    stop: *std.atomic.Value(bool),
    threads: [4]?std.Thread = .{null} ** 4,
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
        server.serve(self.gpa, l, self.be, self.stop, .{ .auth = self.listener.authHash() }) catch {};
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

test "a whole job's text reaches no file the host wrote and no line it logged" {
    const io = testing.io;
    const gpa = testing.allocator;
    // `.iterate`: the walk below reads this directory, and a handle opened
    // without it fails the read with EBADF rather than a permission error.
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);

    // Everything this host may write lives under the temp tree: its incoming
    // model store, and whatever else a handler might decide to open.
    const store_dir = try std.fs.path.join(gpa, &.{ root, "models" });
    defer gpa.free(store_dir);
    var store = try blob.Store.init(gpa, io, store_dir);
    defer store.deinit();

    var stub: Stub = .{ .gpa = gpa, .io = io, .blobs = &store };
    defer stub.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = stub.backend(), .stop = &stop };

    try acc.start();

    // A turn and a render, both carrying the canary, plus an upload: every
    // verb that takes something the user wrote.
    const control = try link_mod.connect(gpa, io, listener.endpoint);
    {
        const body = try wire.encodeAlloc(gpa, wire.Request{ .chat_submit = .{ .text = canary ++ " draw me a fox" } });
        defer gpa.free(body);
        const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    {
        const body = try wire.encodeAlloc(gpa, wire.Request{ .img_enqueue = .{
            .prompt = canary ++ " a fox in snow",
            .negative = canary ++ " blurry",
        } });
        defer gpa.free(body);
        const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    {
        // An attachment: pixels are content too.
        const hdr: wire.BinHeader = .{ .kind = .rgb_upload, .id = 0, .rev = 0, .w = 2, .h = 1, .len = 6 };
        var body: std.ArrayList(u8) = .empty;
        defer body.deinit(gpa);
        try body.appendSlice(gpa, &queue.binHeaderBytes(hdr));
        try body.appendSlice(gpa, "PIXELS");
        const r = try httpc.request(gpa, control, "PUT", "/v1/upload", "application/octet-stream", body.items, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    // A refused blob, named with the canary: the failure path must not write
    // the name it refused either.
    {
        const manifest = "{\"name\":\"" ++ canary ++ ".safetensors\",\"size\":1,\"chunk_bytes\":1,\"digest\":\"\",\"chunks\":[]}";
        const r = try httpc.request(gpa, control, "POST", "/v1/blob/" ++ ("ab" ** 32) ++ "/manifest", "application/json", manifest, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 400), r.status);
    }
    // The reply comes back over the socket, which is where it belongs.
    const events = try link_mod.connect(gpa, io, listener.endpoint);
    try httpc.upgradeEvents(events, "/v1/events", &.{});
    stub.wakeAll();
    {
        const body = try wire.encodeAlloc(gpa, wire.Request{ .chat_submit = .{ .text = canary ++ " again" } });
        defer gpa.free(body);
        const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
        defer gpa.free(r.body);
        try testing.expectEqual(@as(u16, 202), r.status);
    }
    {
        const h = try ws.readHeader(events.reader());
        var got: Io.Writer.Allocating = .init(gpa);
        defer got.deinit();
        try ws.streamPayload(events.reader(), h, &got.writer);
        try testing.expect(std.mem.indexOf(u8, got.written(), reply_canary) != null);
    }

    stop.store(true, .release);
    stub.wakeAll();
    control.close(gpa);
    events.close(gpa);
    acc.finish(io);

    // 1. Nothing under the tree it was given holds either canary.
    for ([_][]const u8{ canary, reply_canary }) |needle| {
        if (try treeHolds(gpa, io, tmp.dir, needle)) |path| {
            defer gpa.free(path);
            std.debug.print("canary '{s}' found in {s}\n", .{ needle, path });
            return error.ContentWrittenToDisk;
        }
    }

    // 2. TEETH. A walk that visits nothing passes exactly like a clean one:
    // plant the canary, in a subdirectory so this also proves the walk
    // descends, and require the walk to find it.
    try tmp.dir.createDirPath(io, "leak");
    try tmp.dir.writeFile(io, .{ .sub_path = "leak/note.txt", .data = "a handler wrote " ++ canary });
    const found = try treeHolds(gpa, io, tmp.dir, canary);
    defer if (found) |p| gpa.free(p);
    try testing.expect(found != null);
    try tmp.dir.deleteTree(io, "leak");
}

test "the request's own bytes are the only copy the protocol layer keeps" {
    // The server hands the body straight to the backend and keeps no copy of
    // its own: the stub's buffer is where the text is, and freeing it is the
    // backend's business. A regression that logged, buffered or cached the
    // body would show up in the canary walk above; this pins the ownership
    // that makes that true.
    const io = testing.io;
    const gpa = testing.allocator;
    var stub: Stub = .{ .gpa = gpa, .io = io };
    defer stub.deinit();
    var stop: std.atomic.Value(bool) = .init(false);
    var listener = try link_mod.listenLoopback(io);
    defer listener.deinit(gpa);
    var acc: Acceptor = .{ .listener = &listener, .gpa = gpa, .be = stub.backend(), .stop = &stop };
    try acc.start();

    const control = try link_mod.connect(gpa, io, listener.endpoint);
    const body = try wire.encodeAlloc(gpa, wire.Request{ .chat_submit = .{ .text = canary } });
    defer gpa.free(body);
    const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
    defer gpa.free(r.body);
    try testing.expectEqual(@as(u16, 202), r.status);
    // The 202 is written before the handler runs, so wait for the echo.
    while (stub.out_gen.load(.acquire) == 0) std.Thread.yield() catch {};
    try testing.expect(std.mem.indexOf(u8, stub.held.items, canary) != null);

    stop.store(true, .release);
    stub.wakeAll();
    control.close(gpa);
    acc.finish(io);
}
