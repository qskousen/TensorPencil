//! One byte stream between a client and a host, whatever carries it: a unix
//! socket for the local child, a loopback TCP port where the Io backend has no
//! unix sockets, TCP under TLS for a remote host. Above this nothing knows
//! which; `std.http.Server` and `httpc` take the reader and writer.
//!
//! Unix sockets are per Io BACKEND, not per OS: `Io.Threaded` has them on POSIX
//! and Windows, `Uring` and `Dispatch` report `AddressFamilyUnsupported`, and
//! `Kqueue` panics. The error gate below covers the middle case; the process
//! Io must be `Threaded` (it is, from `std.process.Init`) to never meet the last.
//!
//! A remote host is reached under TLS with its certificate pinned: the client
//! trusts exactly the one certificate in the pairing string, and every request
//! carries the token from the same string. The client half is std's TLS
//! client; the server half needs tls.zig and lives with the daemon
//! (`serve_tls.zig`), which hands its plaintext pair in through `Link.VTable`.
//! One std hole worth knowing: on Windows, std's client falls back to the OS
//! certificate store when the pinned bundle has no entry for the presented
//! issuer, so there a peer holding any OS-trusted certificate gets past the
//! pin (and is then sent the token). Linux and macOS pin strictly.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const x509 = @import("x509.zig");

/// Sized for a whole HTTP head plus a control frame on each side.
pub const buf_len: usize = 64 * 1024;
/// std's TLS client asserts at least one max ciphertext record on each side.
pub const tls_buf_len: usize = 32 * 1024;
/// A pairing certificate is a self-signed P-256 leaf, a few hundred bytes.
pub const max_cert_der: usize = 1024;

pub const Link = struct {
    io: Io,
    in: *Io.Reader,
    out: *Io.Writer,
    /// Confined to this user by the filesystem (a unix socket), so auth skips
    /// it. Anything else must carry a secret on every request.
    trusted: bool,
    /// The secret this client end sends (`authorization: Bearer <hex>`).
    auth: ?Secret = null,
    vt: *const VTable,

    pub const VTable = struct {
        /// Flush, close the transport, free everything, the link included.
        close: *const fn (*Link, std.mem.Allocator) void,
        /// Unblock a thread parked reading this link.
        shutdown: *const fn (*Link) void,
    };

    /// A link straight over a stream: a unix socket or the loopback port.
    pub fn init(gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, trusted: bool) !*Link {
        const p = try Plain.create(gpa, io, stream, trusted);
        return &p.link;
    }

    pub fn reader(l: *Link) *Io.Reader {
        return l.in;
    }

    pub fn writer(l: *Link) *Io.Writer {
        return l.out;
    }

    pub fn close(l: *Link, gpa: std.mem.Allocator) void {
        l.vt.close(l, gpa);
    }

    pub fn shutdown(l: *Link) void {
        l.vt.shutdown(l);
    }
};

/// Built in place: the interfaces recover their parent by pointer.
const Plain = struct {
    link: Link,
    stream: Io.net.Stream,
    sr: Io.net.Stream.Reader,
    sw: Io.net.Stream.Writer,
    in_buf: [buf_len]u8,
    out_buf: [buf_len]u8,

    const vtable: Link.VTable = .{ .close = close, .shutdown = shutdown };

    fn create(gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, trusted: bool) !*Plain {
        const p = try gpa.create(Plain);
        p.stream = stream;
        p.sr = stream.reader(io, &p.in_buf);
        p.sw = stream.writer(io, &p.out_buf);
        p.link = .{ .io = io, .in = &p.sr.interface, .out = &p.sw.interface, .trusted = trusted, .vt = &vtable };
        return p;
    }

    fn close(l: *Link, gpa: std.mem.Allocator) void {
        const p: *Plain = @fieldParentPtr("link", l);
        p.sw.interface.flush() catch {};
        p.stream.close(l.io);
        gpa.destroy(p);
    }

    fn shutdown(l: *Link) void {
        const p: *Plain = @fieldParentPtr("link", l);
        p.stream.shutdown(l.io, .both) catch {};
    }
};

/// A TLS layer's plaintext writer, whose flush reaches the socket. Both std's
/// client and tls.zig's connection encrypt into the socket writer's buffer on
/// flush and stop there, and the protocol above flushes once per message
/// expecting the peer to see it. Buffered, since `std.http.Server` asks its
/// writer for writable slices.
pub const Through = struct {
    interface: Io.Writer,
    tls: *Io.Writer,
    sock: *Io.Writer,
    buf: [16 * 1024]u8,

    pub fn init(self: *Through, tls: *Io.Writer, sock: *Io.Writer) void {
        self.tls = tls;
        self.sock = sock;
        self.interface = .{ .vtable = &vtable, .buffer = &self.buf };
    }

    const vtable: Io.Writer.VTable = .{ .drain = drain, .flush = flush };

    fn drain(w: *Io.Writer, data: []const []const u8, splat: usize) Io.Writer.Error!usize {
        const t: *Through = @fieldParentPtr("interface", w);
        try t.tls.writeAll(w.buffered());
        w.end = 0;
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |d| {
            try t.tls.writeAll(d);
            n += d.len;
        }
        const pattern = data[data.len - 1];
        for (0..splat) |_| try t.tls.writeAll(pattern);
        return n + pattern.len * splat;
    }

    fn flush(w: *Io.Writer) Io.Writer.Error!void {
        const t: *Through = @fieldParentPtr("interface", w);
        try t.tls.writeAll(w.buffered());
        w.end = 0;
        try t.tls.flush();
        try t.sock.flush();
    }
};

/// The client end of a pinned TLS connection, std's client over a TCP stream.
const TlsClient = struct {
    link: Link,
    stream: Io.net.Stream,
    sr: Io.net.Stream.Reader,
    sw: Io.net.Stream.Writer,
    lock: Io.RwLock,
    bundle: std.crypto.Certificate.Bundle,
    client: std.crypto.tls.Client,
    through: Through,
    in_buf: [tls_buf_len]u8,
    out_buf: [tls_buf_len]u8,
    /// The plaintext side; an HTTP head must fit.
    rd_buf: [buf_len]u8,
    wr_buf: [tls_buf_len]u8,

    const vtable: Link.VTable = .{ .close = close, .shutdown = shutdown };

    fn create(gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, cert_der: []const u8, token: Secret) !*TlsClient {
        const c = try gpa.create(TlsClient);
        errdefer gpa.destroy(c);
        const now = Io.Clock.real.now(io);
        c.bundle = try x509.bundleFromDer(gpa, cert_der, now.toSeconds());
        errdefer c.bundle.deinit(gpa);
        c.lock = .init;
        c.stream = stream;
        c.sr = stream.reader(io, &c.in_buf);
        c.sw = stream.writer(io, &c.out_buf);
        var entropy: [std.crypto.tls.Client.Options.entropy_len]u8 = undefined;
        io.random(&entropy);
        c.client = try std.crypto.tls.Client.init(&c.sr.interface, &c.sw.interface, .{
            .host = .no_verification,
            .ca = .{ .bundle = .{ .gpa = gpa, .io = io, .lock = &c.lock, .bundle = &c.bundle } },
            .write_buffer = &c.wr_buf,
            .read_buffer = &c.rd_buf,
            .entropy = &entropy,
            .realtime_now = now,
        });
        c.through.init(&c.client.writer, &c.sw.interface);
        c.link = .{ .io = io, .in = &c.client.reader, .out = &c.through.interface, .trusted = false, .auth = token, .vt = &vtable };
        return c;
    }

    fn close(l: *Link, gpa: std.mem.Allocator) void {
        const c: *TlsClient = @fieldParentPtr("link", l);
        c.client.end() catch {};
        c.sw.interface.flush() catch {};
        c.stream.close(l.io);
        c.bundle.deinit(gpa);
        gpa.destroy(c);
    }

    fn shutdown(l: *Link) void {
        const c: *TlsClient = @fieldParentPtr("link", l);
        c.stream.shutdown(l.io, .both) catch {};
    }
};

/// What an untrusted client presents on every request, 64 hex on the wire: a
/// loopback host's one-run cookie, or a remote host's token. The host keeps
/// only the hash, so a copied state directory yields no credential.
pub const Secret = [32]u8;
pub const SecretHash = [32]u8;

pub fn secretHex(s: Secret) [64]u8 {
    return std.fmt.bytesToHex(s, .lower);
}

/// A token's short public name: the first 8 hex of its hash. The daemon keeps
/// only the hash and the client only the token, so both ends can print this,
/// and "the token was refused" becomes two numbers to compare.
pub fn fingerprint(h: SecretHash) [8]u8 {
    return secretHex(h)[0..8].*;
}

pub fn parseSecretHex(hex: []const u8) ?Secret {
    if (hex.len != 64) return null;
    var s: Secret = undefined;
    _ = std.fmt.hexToBytes(&s, hex) catch return null;
    return s;
}

pub fn hashSecret(s: Secret) SecretHash {
    var out: SecretHash = undefined;
    std.crypto.hash.Blake3.hash(&s, &out, .{});
    return out;
}

pub const HostPort = struct { host: []const u8, port: u16 };

/// `host:port`, with an IPv6 literal in brackets (`[::1]:7777`).
pub fn splitHostPort(text: []const u8) ?HostPort {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return null;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return null;
    var host = text[0..colon];
    if (host.len >= 2 and host[0] == '[' and host[host.len - 1] == ']') host = host[1 .. host.len - 1];
    if (host.len == 0) return null;
    return .{ .host = host, .port = port };
}

/// Where a host listens, as a client names it. The text form is what a spawned
/// child prints on its `ready` line (`unix <path>` / `loopback <port> <cookie>`)
/// or, for a remote host, the pairing string the daemon prints once:
/// `tp://<host>:<port>/<certificate DER, base64url>#<token hex>`.
pub const Endpoint = union(enum) {
    unix: []const u8,
    loopback: Loopback,
    tls: Tls,
    /// A plain TCP port: what a remote daemon listens on, never a client's
    /// destination (a remote host is only reached under TLS).
    tcp: u16,

    pub const Loopback = struct { port: u16, cookie: Secret };
    pub const Tls = struct {
        host: []const u8,
        port: u16,
        /// The one certificate the client trusts, DER in base64url.
        cert_b64: []const u8,
        token: Secret,

        pub fn certDer(self: *const Tls, buf: *[max_cert_der]u8) ![]u8 {
            const dec = std.base64.url_safe_no_pad.Decoder;
            const n = try dec.calcSizeForSlice(self.cert_b64);
            if (n > buf.len) return error.CertificateTooLong;
            try dec.decode(buf[0..n], self.cert_b64);
            return buf[0..n];
        }
    };

    /// The full text form, secret included: what the peer needs, never a log.
    pub fn format(self: Endpoint, w: *Io.Writer) Io.Writer.Error!void {
        switch (self) {
            .unix => |p| try w.print("unix {s}", .{p}),
            .loopback => |l| try w.print("loopback {d} {s}", .{ l.port, &secretHex(l.cookie) }),
            .tls => |t| {
                try self.public().format(w);
                try w.print("/{s}#{s}", .{ t.cert_b64, &secretHex(t.token) });
            },
            .tcp => |port| try w.print("tcp {d}", .{port}),
        }
    }

    /// The same endpoint with its secret left out, for a log line.
    pub fn public(self: Endpoint) Public {
        return .{ .ep = self };
    }

    pub const Public = struct {
        ep: Endpoint,

        pub fn format(self: Public, w: *Io.Writer) Io.Writer.Error!void {
            switch (self.ep) {
                .unix => |p| try w.print("unix {s}", .{p}),
                .loopback => |l| try w.print("loopback {d}", .{l.port}),
                .tls => |t| {
                    const v6 = std.mem.indexOfScalar(u8, t.host, ':') != null;
                    try w.print("tp://{s}{s}{s}:{d}", .{ if (v6) "[" else "", t.host, if (v6) "]" else "", t.port });
                },
                .tcp => |port| try w.print("tcp {d}", .{port}),
            }
        }
    };

    pub fn parse(text: []const u8) ?Endpoint {
        const t = std.mem.trim(u8, text, " \t\r\n");
        if (std.mem.startsWith(u8, t, "unix ")) return .{ .unix = t[5..] };
        if (std.mem.startsWith(u8, t, "loopback ")) {
            var it = std.mem.tokenizeScalar(u8, t[9..], ' ');
            const port = std.fmt.parseInt(u16, it.next() orelse return null, 10) catch return null;
            const cookie = parseSecretHex(it.next() orelse return null) orelse return null;
            return .{ .loopback = .{ .port = port, .cookie = cookie } };
        }
        if (std.mem.startsWith(u8, t, "tcp ")) {
            return .{ .tcp = std.fmt.parseInt(u16, t[4..], 10) catch return null };
        }
        if (std.mem.startsWith(u8, t, "tp://")) {
            const rest = t[5..];
            const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
            const hp = splitHostPort(rest[0..slash]) orelse return null;
            const hash = std.mem.indexOfScalar(u8, rest, '#') orelse return null;
            if (hash <= slash + 1) return null;
            const token = parseSecretHex(rest[hash + 1 ..]) orelse return null;
            return .{ .tls = .{ .host = hp.host, .port = hp.port, .cert_b64 = rest[slash + 1 .. hash], .token = token } };
        }
        return null;
    }
};

pub const Listener = struct {
    io: Io,
    server: Io.net.Server,
    /// The unix path is a gpa-owned copy, unlinked on `deinit`.
    endpoint: Endpoint,
    /// What an untrusted client's bearer must hash to; null when every link
    /// is trusted (a unix socket).
    auth: ?SecretHash = null,

    /// A plain link over the next connection. A TLS listener takes
    /// `acceptStream` instead and wraps the stream itself.
    pub fn accept(self: *Listener, gpa: std.mem.Allocator) !*Link {
        const s = try self.acceptStream();
        errdefer s.close(self.io);
        return Link.init(gpa, self.io, s, self.endpoint == .unix);
    }

    pub fn acceptStream(self: *Listener) !Io.net.Stream {
        return self.server.accept(self.io);
    }

    pub fn authHash(self: *const Listener) ?*const SecretHash {
        return if (self.auth) |*h| h else null;
    }

    pub fn deinit(self: *Listener, gpa: std.mem.Allocator) void {
        self.server.deinit(self.io);
        switch (self.endpoint) {
            .unix => |p| {
                Io.Dir.cwd().deleteFile(self.io, p) catch {};
                gpa.free(p);
            },
            else => {},
        }
    }
};

/// Listen for local clients at `path`. A socket file nobody answers on is a
/// leftover and is replaced; one somebody answers on is `AddressInUse`. Where
/// the backend has no unix sockets this binds an ephemeral loopback port
/// instead, and the endpoint says so.
pub fn listenLocal(gpa: std.mem.Allocator, io: Io, path: []const u8) !Listener {
    if (!Io.net.has_unix_sockets) return listenLoopback(io);
    const ua = try Io.net.UnixAddress.init(path);
    const server = ua.listen(io, .{}) catch |err| switch (err) {
        error.AddressFamilyUnsupported => return listenLoopback(io),
        error.AddressInUse => blk: {
            if (ua.connect(io)) |s| {
                s.close(io);
                return error.AddressInUse;
            } else |_| {}
            Io.Dir.cwd().deleteFile(io, path) catch {};
            break :blk try ua.listen(io, .{});
        },
        else => return err,
    };
    return .{ .io = io, .server = server, .endpoint = .{ .unix = try gpa.dupe(u8, path) } };
}

pub fn listenLoopback(io: Io) !Listener {
    const bind = try Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    const server = try bind.listen(io, .{ .reuse_address = true });
    var cookie: Secret = undefined;
    io.random(&cookie);
    return .{
        .io = io,
        .server = server,
        .endpoint = .{ .loopback = .{ .port = server.socket.address.getPort(), .cookie = cookie } },
        .auth = hashSecret(cookie),
    };
}

/// Listen for remote clients on `bind`:`port` (an IP literal; `0.0.0.0` for
/// every interface). Every connection must be wrapped in TLS by the caller
/// and present a secret hashing to `auth`.
pub fn listenTcp(io: Io, bind: []const u8, port: u16, auth: SecretHash) !Listener {
    const addr = try Io.net.IpAddress.parse(bind, port);
    const server = try addr.listen(io, .{ .reuse_address = true });
    return .{ .io = io, .server = server, .endpoint = .{ .tcp = server.socket.address.getPort() }, .auth = auth };
}

pub fn connect(gpa: std.mem.Allocator, io: Io, ep: Endpoint) !*Link {
    switch (ep) {
        .unix => |p| {
            const stream = try (try Io.net.UnixAddress.init(p)).connect(io);
            errdefer stream.close(io);
            return Link.init(gpa, io, stream, true);
        },
        .loopback => |l| {
            const stream = try (try Io.net.IpAddress.parseIp4("127.0.0.1", l.port)).connect(io, .{ .mode = .stream });
            errdefer stream.close(io);
            const link = try Link.init(gpa, io, stream, false);
            link.auth = l.cookie;
            return link;
        },
        .tls => |t| {
            var der_buf: [max_cert_der]u8 = undefined;
            const der = try t.certDer(&der_buf);
            const stream = try connectHost(io, t.host, t.port);
            errdefer stream.close(io);
            const c = try TlsClient.create(gpa, io, stream, der, t.token);
            return &c.link;
        },
        .tcp => return error.PlainTcpRefused,
    }
}

/// An IP literal as given, else a name through the resolver.
fn connectHost(io: Io, host: []const u8, port: u16) !Io.net.Stream {
    if (Io.net.IpAddress.parse(host, port)) |a| return a.connect(io, .{ .mode = .stream }) else |_| {}
    const hn = try Io.net.HostName.init(host);
    return hn.connect(io, port, .{ .mode = .stream });
}

/// The local host's socket: `$XDG_RUNTIME_DIR/tensorpencil/local.sock`, the
/// directory created 0700. Falls back to the temp dir, which is why the name
/// carries the user.
///
/// That directory is the local link's ONLY authentication: a unix link is
/// trusted and carries no bearer secret, so anyone who can put a socket at this
/// path is the host as far as a client is concerned. The fallback base is
/// world-writable, where the name can be pre-created, so the mode is CHECKED and
/// not merely requested. A directory someone else owns is either group- or
/// other-accessible, which this refuses, or 0700 and untraversable by us, which
/// fails on first use -- a symlink to either lands in the same two cases.
pub fn defaultLocalPath(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map) ![]u8 {
    const base = env.get("XDG_RUNTIME_DIR") orelse env.get("TMPDIR") orelse env.get("TEMP") orelse "/tmp";
    const user = env.get("USER") orelse env.get("USERNAME") orelse "tp";
    const dir = try std.fmt.allocPrint(gpa, "{s}/tensorpencil-{s}", .{ base, user });
    defer gpa.free(dir);
    const perms: Io.Dir.Permissions = if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700);
    _ = Io.Dir.cwd().createDirPathStatus(io, dir, perms) catch {};
    if (builtin.os.tag != .windows) {
        const st = try Io.Dir.cwd().statFile(io, dir, .{});
        if (st.kind != .directory or st.permissions.toMode() & 0o077 != 0) return error.UnsafeSocketDir;
    }
    return std.fmt.allocPrint(gpa, "{s}/local.sock", .{dir});
}

test "an endpoint's text form parses back" {
    var buf: [64]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try w.print("{f}", .{Endpoint{ .unix = "/run/x.sock" }});
    try std.testing.expectEqualStrings("/run/x.sock", Endpoint.parse(w.buffered()).?.unix);
    var buf2: [128]u8 = undefined;
    var w2: Io.Writer = .fixed(&buf2);
    const cookie: Secret = .{7} ** 32;
    try w2.print("{f}", .{Endpoint{ .loopback = .{ .port = 4321, .cookie = cookie } }});
    const lb = Endpoint.parse(w2.buffered()).?.loopback;
    try std.testing.expectEqual(@as(u16, 4321), lb.port);
    try std.testing.expectEqual(cookie, lb.cookie);
    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("ready"));
    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("loopback 4321"));
    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("loopback 4321 abcd"));
    try std.testing.expectEqual(@as(u16, 7777), Endpoint.parse("tcp 7777").?.tcp);
}

test "the public form of an endpoint carries no secret" {
    const cookie: Secret = .{0x5c} ** 32;
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try w.print("{f}", .{(Endpoint{ .loopback = .{ .port = 4321, .cookie = cookie } }).public()});
    try std.testing.expectEqualStrings("loopback 4321", w.buffered());
    var w2: Io.Writer = .fixed(&buf);
    try w2.print("{f}", .{(Endpoint{ .tls = .{ .host = "fd7a::8", .port = 7777, .cert_b64 = "MIIB", .token = cookie } }).public()});
    try std.testing.expectEqualStrings("tp://[fd7a::8]:7777", w2.buffered());
    try std.testing.expect(std.mem.indexOf(u8, w2.buffered(), &secretHex(cookie)) == null);
}

test "a pairing string round-trips, brackets an IPv6 host, and refuses a short token" {
    const token: Secret = .{0xab} ** 32;
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try w.print("{f}", .{Endpoint{ .tls = .{ .host = "lydia", .port = 7777, .cert_b64 = "MIIB_-cert", .token = token } }});
    try std.testing.expect(std.mem.startsWith(u8, w.buffered(), "tp://lydia:7777/MIIB_-cert#abab"));
    const t = Endpoint.parse(w.buffered()).?.tls;
    try std.testing.expectEqualStrings("lydia", t.host);
    try std.testing.expectEqual(@as(u16, 7777), t.port);
    try std.testing.expectEqualStrings("MIIB_-cert", t.cert_b64);
    try std.testing.expectEqual(token, t.token);

    var w6: Io.Writer = .fixed(&buf);
    try w6.print("{f}", .{Endpoint{ .tls = .{ .host = "fd7a::8", .port = 1, .cert_b64 = "AA", .token = token } }});
    try std.testing.expect(std.mem.startsWith(u8, w6.buffered(), "tp://[fd7a::8]:1/AA#"));
    try std.testing.expectEqualStrings("fd7a::8", Endpoint.parse(w6.buffered()).?.tls.host);

    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("tp://lydia:7777/AA#abab"));
    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("tp://lydia/AA#" ++ ("ab" ** 32)));
    try std.testing.expectEqual(@as(?Endpoint, null), Endpoint.parse("tp://lydia:7777/#" ++ ("ab" ** 32)));
    try std.testing.expectEqual(@as(?HostPort, null), splitHostPort("nohost"));
    try std.testing.expectEqualStrings("::1", splitHostPort("[::1]:80").?.host);
}

test "the certificate in a pairing string decodes to the DER it was made from" {
    const der = [_]u8{ 0x30, 0x82, 0x01, 0x02, 0xff, 0x00, 0x7e };
    const enc = std.base64.url_safe_no_pad.Encoder;
    var b64: [16]u8 = undefined;
    const text = enc.encode(&b64, &der);
    const t: Endpoint.Tls = .{ .host = "h", .port = 1, .cert_b64 = text, .token = .{0} ** 32 };
    var out: [max_cert_der]u8 = undefined;
    try std.testing.expectEqualSlices(u8, &der, try t.certDer(&out));
}

test "a loopback listener accepts a link and bytes cross it both ways" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var listener = try listenLoopback(io);
    defer listener.deinit(gpa);
    try std.testing.expectEqual(hashSecret(listener.endpoint.loopback.cookie), listener.authHash().?.*);
    const Side = struct {
        fn serve(l: *Listener, a: std.mem.Allocator) void {
            const link = l.accept(a) catch return;
            defer link.close(a);
            const line = link.reader().takeDelimiterInclusive('\n') catch return;
            link.writer().print("echo {s}", .{line}) catch return;
            link.writer().flush() catch return;
        }
    };
    const th = try std.Thread.spawn(.{}, Side.serve, .{ &listener, gpa });
    const c = try connect(gpa, io, listener.endpoint);
    defer c.close(gpa);
    try std.testing.expect(!c.trusted);
    try std.testing.expectEqual(listener.endpoint.loopback.cookie, c.auth.?);
    try c.writer().writeAll("ping\n");
    try c.writer().flush();
    const back = try c.reader().takeDelimiterInclusive('\n');
    try std.testing.expectEqualStrings("echo ping\n", back);
    th.join();
}
