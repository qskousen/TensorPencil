//! The remote link's gate, on a loopback socket with no engine: the daemon's
//! server end (`serve_tls.zig`, tls.zig behind the real `server.serve`
//! routing) against the client's end (`link.connect` on a `tp://` endpoint,
//! std's TLS client pinned to the minted certificate). Checks: hello with the
//! right token, a wrong token refused with 401 and the connection closed, a
//! wrong pin refused at the handshake, a 4 MiB events frame and an 8 MiB
//! upload intact. Exits 0 only when all of it holds. `zig build tls-spike`.
const std = @import("std");
const Io = std.Io;
const serve = @import("serve");
const link = serve.link;
const server = serve.server;
const httpc = serve.httpc;
const ws = serve.ws;
const queue = serve.queue;
const wire = serve.wire;
const x509 = serve.x509;
const serve_tls = @import("serve_tls.zig");
const Blake3 = std.crypto.hash.Blake3;

const push_bytes: usize = 4 << 20;
const upload_bytes: usize = 8 << 20;

const log = std.log.scoped(.tls_spike);

fn fillPattern(buf: []u8, offset: u64) void {
    for (buf, 0..) |*b, i| b.* = @truncate(((offset + i) *% 0x9E3779B1) >> 13);
}

fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Blake3.hash(bytes, &out, .{});
    return out;
}

/// The backend: answers hello, keeps the last upload's digest, pushes what
/// the test queues.
const Fake = struct {
    gpa: std.mem.Allocator,
    io: Io,
    out: queue.Queue = .{},
    out_gen: std.atomic.Value(u32) = .init(0),
    attached: std.atomic.Value(bool) = .init(false),
    requests: std.atomic.Value(u32) = .init(0),
    upload_digest: [32]u8 = .{0} ** 32,
    upload_len: usize = 0,

    fn backend(self: *Fake) server.Backend {
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
        };
    }
    fn hello(_: *anyopaque, gpa: std.mem.Allocator) anyerror![]u8 {
        return wire.encodeAlloc(gpa, wire.Event{ .hello = .{ .gen = 42 } });
    }
    fn request(ctx: *anyopaque, bytes: []u8) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        _ = self.requests.fetchAdd(1, .monotonic);
        self.gpa.free(bytes);
    }
    fn upload(ctx: *anyopaque, _: wire.BinHeader, payload: []u8) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.upload_digest = digest(payload);
        self.upload_len = payload.len;
        self.gpa.free(payload);
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
    fn take(ctx: *anyopaque, out: *std.ArrayList(queue.Frame)) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.out.drain(self.io, self.gpa, out);
    }
    fn eventsAttach(ctx: *anyopaque) bool {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        return self.attached.cmpxchgStrong(false, true, .acq_rel, .acquire) == null;
    }
    fn eventsDetach(ctx: *anyopaque) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.attached.store(false, .release);
    }
    fn push(self: *Fake, f: queue.Frame) !void {
        try self.out.push(self.io, self.gpa, f);
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }
    fn wakeAll(self: *Fake) void {
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }
};

/// tp-serve's accept loop in miniature: `n` connections, each handshaken and
/// served on its own thread.
const Acceptor = struct {
    gpa: std.mem.Allocator,
    io: Io,
    listener: *link.Listener,
    ident: *serve_tls.Identity,
    be: server.Backend,
    stop: *const std.atomic.Value(bool),
    n: usize,
    opts: server.Options,
    threads: [8]?std.Thread = .{null} ** 8,
    handshake_failures: std.atomic.Value(u32) = .init(0),

    fn run(self: *Acceptor) void {
        for (0..self.n) |i| {
            const stream = self.listener.acceptStream() catch return;
            self.threads[i] = std.Thread.spawn(.{}, one, .{ self, stream }) catch {
                stream.close(self.io);
                return;
            };
        }
    }
    fn one(self: *Acceptor, stream: Io.net.Stream) void {
        const l = serve_tls.accept(self.gpa, self.io, stream, self.ident) catch |err| {
            log.info("server side refused a handshake: {t}", .{err});
            _ = self.handshake_failures.fetchAdd(1, .monotonic);
            stream.close(self.io);
            return;
        };
        defer l.close(self.gpa);
        server.serve(self.gpa, l, self.be, self.stop, self.opts) catch |err| log.info("connection ended: {t}", .{err});
    }
    fn join(self: *Acceptor) void {
        for (&self.threads) |*t| if (t.*) |th| {
            th.join();
            t.* = null;
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const now_sec = Io.Clock.real.now(io).toSeconds();

    // The daemon's side of pairing: an identity, a token, its hash.
    var id = try x509.selfSigned(gpa, io, x509.Ecdsa.KeyPair.generate(io), .{ .not_before = now_sec - 60, .not_after = now_sec + 3600 });
    defer id.deinit(gpa);
    const cert_pem = try id.certPemAlloc(gpa);
    defer gpa.free(cert_pem);
    const key_pem = try id.keyPemAlloc(gpa);
    defer gpa.free(key_pem);
    var ident = try serve_tls.identityFromPem(gpa, io, cert_pem, key_pem);
    defer ident.deinit(gpa);
    var token: link.Secret = undefined;
    io.random(&token);

    var listener = try link.listenTcp(io, "127.0.0.1", 0, link.hashSecret(token));
    defer listener.deinit(gpa);
    const port = listener.endpoint.tcp;

    // The pairing string, as the daemon prints it and the client parses it.
    const enc = std.base64.url_safe_no_pad.Encoder;
    const b64 = try gpa.alloc(u8, enc.calcSize(id.cert_der.len));
    defer gpa.free(b64);
    var pair_buf: [4096]u8 = undefined;
    var pw: Io.Writer = .fixed(&pair_buf);
    try pw.print("{f}", .{link.Endpoint{ .tls = .{ .host = "127.0.0.1", .port = port, .cert_b64 = enc.encode(b64, id.cert_der), .token = token } }});
    const ep = link.Endpoint.parse(pw.buffered()) orelse return error.PairingStringUnparsable;

    var fake: Fake = .{ .gpa = gpa, .io = io };
    var stop: std.atomic.Value(bool) = .init(false);
    var acc: Acceptor = .{
        .gpa = gpa,
        .io = io,
        .listener = &listener,
        .ident = &ident,
        .be = fake.backend(),
        .stop = &stop,
        .n = 4,
        .opts = .{ .auth = listener.authHash(), .unauthorized_delay_ns = 50 * std.time.ns_per_ms },
    };
    const th = try std.Thread.spawn(.{}, Acceptor.run, .{&acc});

    // 1. Control: hello, a request, an upload, all under the right token.
    const control = try link.connect(gpa, io, ep);
    {
        const r = try httpc.request(gpa, control, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(r.body);
        if (r.status != 200) return error.HelloRefused;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const ev = try wire.decode(wire.Event, arena.allocator(), r.body);
        if (ev.hello.gen != 42) return error.WrongHello;
        log.info("hello over TLS with the right token: gen {d}", .{ev.hello.gen});
    }
    {
        const body = try wire.encodeAlloc(gpa, wire.Request{ .chat_submit = .{ .text = "hi" } });
        defer gpa.free(body);
        const r = try httpc.request(gpa, control, "POST", "/v1/req", "application/json", body, 1 << 16);
        defer gpa.free(r.body);
        if (r.status != 202) return error.RequestRefused;
    }
    {
        const hdr: wire.BinHeader = .{ .kind = .rgb_upload, .id = 1, .rev = 0, .w = 2048, .h = 1365, .len = upload_bytes };
        const body = try gpa.alloc(u8, @sizeOf(wire.BinHeader) + upload_bytes);
        defer gpa.free(body);
        @memcpy(body[0..@sizeOf(wire.BinHeader)], &queue.binHeaderBytes(hdr));
        fillPattern(body[@sizeOf(wire.BinHeader)..], 7);
        const r = try httpc.request(gpa, control, "PUT", "/v1/upload", "application/octet-stream", body, 1 << 16);
        defer gpa.free(r.body);
        if (r.status != 202) return error.UploadRefused;
        // The handler ran on the connection thread before the 202 was written.
        if (fake.upload_len != upload_bytes or !std.mem.eql(u8, &fake.upload_digest, &digest(body[@sizeOf(wire.BinHeader)..]))) return error.UploadCorrupt;
        log.info("8 MiB upload intact through the TLS link", .{});
    }

    // 2. Events: the upgrade, then a 4 MiB frame pushed by the backend.
    const events = try link.connect(gpa, io, ep);
    try httpc.upgradeEvents(events, "/v1/events", &.{});
    {
        const payload = try gpa.alloc(u8, push_bytes);
        errdefer gpa.free(payload);
        fillPattern(payload, 0);
        const want = digest(payload);
        try fake.push(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 5, .rev = 1, .w = 1024, .h = 1024, .len = push_bytes }, .payload = payload } });
        const h = try ws.readHeader(events.reader());
        if (h.opcode != .binary or h.len != @sizeOf(wire.BinHeader) + push_bytes) return error.BadFrame;
        var got: Io.Writer.Allocating = .init(gpa);
        defer got.deinit();
        try ws.streamPayload(events.reader(), h, &got.writer);
        if (!std.mem.eql(u8, &want, &digest(got.written()[@sizeOf(wire.BinHeader)..]))) return error.FrameCorrupt;
        log.info("4 MiB events frame intact through the TLS link", .{});
    }

    // 3. A wrong token: 401, and the connection is closed under us.
    {
        var wrong = ep;
        wrong.tls.token[0] +%= 1;
        const c = try link.connect(gpa, io, wrong);
        defer c.close(gpa);
        const r = try httpc.request(gpa, c, "GET", "/v1/hello", "", "", 1 << 16);
        defer gpa.free(r.body);
        if (r.status != 401) return error.WrongTokenAccepted;
        if (httpc.request(gpa, c, "GET", "/v1/hello", "", "", 1 << 16)) |again| {
            gpa.free(again.body);
            return error.ConnectionKeptAfter401;
        } else |_| {}
        log.info("wrong token: 401 and the connection closed", .{});
    }

    // 4. A wrong pin: another key's certificate under the same name is refused
    //    before a byte of HTTP moves, so the token never reaches an impostor.
    {
        var other = try x509.selfSigned(gpa, io, x509.Ecdsa.KeyPair.generate(io), .{ .not_before = now_sec - 60, .not_after = now_sec + 3600 });
        defer other.deinit(gpa);
        const ob64 = try gpa.alloc(u8, enc.calcSize(other.cert_der.len));
        defer gpa.free(ob64);
        var wrong = ep;
        wrong.tls.cert_b64 = enc.encode(ob64, other.cert_der);
        if (link.connect(gpa, io, wrong)) |c| {
            c.close(gpa);
            return error.WrongPinAccepted;
        } else |err| switch (err) {
            error.TlsCertificateNotVerified, error.CertificateSignatureInvalid => log.info("wrong pin refused at the handshake: {t}", .{err}),
            else => return err,
        }
    }

    stop.store(true, .release);
    fake.wakeAll();
    control.close(gpa);
    events.close(gpa);
    th.join();
    acc.join();
    fake.out.deinit(gpa);
    if (fake.requests.load(.acquire) != 1) return error.RequestLost;

    var out_buf: [64]u8 = undefined;
    var stdout = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    try stdout.interface.writeAll("tls-spike: ok\n");
    try stdout.interface.flush();
}
