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
//! carries the token from the same string. Both ends are tls.zig's
//! (`serve_tls.zig`), which hands its plaintext pair in through `Link.VTable`;
//! a binary that dials a remote host installs the client (`tls_dial`), so this
//! module stays pure std. Not std's client: on Windows it falls back to the OS
//! certificate store when the pinned bundle lacks the presented issuer, which
//! lets any OS-trusted certificate past the pin, and the token with it.
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

/// Opens a link under TLS over `stream`, trusting only the certificate `cert_der`, with
/// every request carrying `token`. On failure the stream is still the caller's.
pub const TlsDial = *const fn (gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, cert_der: []const u8, token: Secret) anyerror!*Link;

/// The binary's TLS client (`serve_tls.installClient`); none refuses a `tp://` endpoint.
pub var tls_dial: ?TlsDial = null;

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

    /// Bring a thread blocked in `accept` back by connecting to ourselves; false when
    /// the connect fails. Closing the socket under the accept is not portable: Windows
    /// cancels it, and std's `netAcceptWindows` takes a cancel as unreachable.
    pub fn wake(self: *const Listener) bool {
        const stream = switch (self.endpoint) {
            .unix => |p| (Io.net.UnixAddress.init(p) catch return false).connect(self.io),
            .loopback, .tls, .tcp => blk: {
                // A listener bound to every address is reached on loopback.
                var a = self.server.socket.address;
                switch (a) {
                    .ip4 => |*v| if (std.mem.allEqual(u8, &v.bytes, 0)) {
                        v.bytes = .{ 127, 0, 0, 1 };
                    },
                    .ip6 => |*v| if (std.mem.allEqual(u8, &v.bytes, 0)) {
                        v.bytes[15] = 1;
                    },
                }
                break :blk a.connect(self.io, .{ .mode = .stream });
            },
        } catch return false;
        stream.close(self.io);
        return true;
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
            // On Windows std returns Unexpected (a stack trace in Debug) for every
            // failed AF_UNIX connect, a socket that is not there included.
            Io.Dir.cwd().access(io, p, .{}) catch return error.FileNotFound;
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
            const dial = tls_dial orelse return error.TlsClientNotInstalled;
            const stream = try connectHost(io, t.host, t.port);
            errdefer stream.close(io);
            return dial(gpa, io, stream, der, t.token);
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

/// The local host's socket, `localPath`'s `local.sock`.
pub fn defaultLocalPath(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map) ![]u8 {
    return localPath(gpa, io, env, "local.sock");
}

/// A socket named `name` in `$XDG_RUNTIME_DIR/tensorpencil-<user>`, the directory
/// created 0700. Falls back to the temp dir, which is why the name carries the user.
///
/// That directory is the local link's ONLY authentication: a unix link is
/// trusted and carries no bearer secret, so anyone who can put a socket at this
/// path is the host as far as a client is concerned. The fallback base is
/// world-writable, where the name can be pre-created, so the mode is CHECKED and
/// not merely requested. A directory someone else owns is either group- or
/// other-accessible, which this refuses, or 0700 and untraversable by us, which
/// fails on first use -- a symlink to either lands in the same two cases.
pub fn localPath(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map, name: []const u8) ![]u8 {
    const base = env.get("XDG_RUNTIME_DIR") orelse env.get("TMPDIR") orelse env.get("TEMP") orelse "/tmp";
    const user = env.get("USER") orelse env.get("USERNAME") orelse "tp";
    // The native separator: Windows hands a unix socket's path to its socket driver
    // as is, and that driver takes no '/'.
    const sep = std.fs.path.sep_str;
    const dir = try std.fmt.allocPrint(gpa, "{s}" ++ sep ++ "tensorpencil-{s}", .{ base, user });
    defer gpa.free(dir);
    const perms: Io.Dir.Permissions = if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700);
    _ = try Io.Dir.cwd().createDirPathStatus(io, dir, perms);
    if (builtin.os.tag != .windows) {
        const st = try Io.Dir.cwd().statFile(io, dir, .{});
        if (st.kind != .directory or st.permissions.toMode() & 0o077 != 0) return error.UnsafeSocketDir;
    }
    return std.fmt.allocPrint(gpa, "{s}" ++ sep ++ "{s}", .{ dir, name });
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

// What tp-serve's shutdown relies on, unix, loopback and a TCP listener on every address.
test "wake brings a thread blocked in accept back" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "wake.sock" });
    defer a.free(path);
    const Waiter = struct {
        done: std.atomic.Value(bool) = .init(false),
        fn run(w: *@This(), l: *Listener) void {
            if (l.acceptStream()) |st| st.close(l.io) else |_| {}
            w.done.store(true, .release);
        }
    };
    for (0..3) |kind| {
        var l = switch (kind) {
            0 => if (Io.net.has_unix_sockets) try listenLocal(a, io, path) else continue,
            1 => try listenLoopback(io),
            else => try listenTcp(io, "0.0.0.0", 0, undefined),
        };
        var w: Waiter = .{};
        const t = try std.Thread.spawn(.{}, Waiter.run, .{ &w, &l });
        try Io.sleep(io, .{ .nanoseconds = 50 * std.time.ns_per_ms }, .real);
        try std.testing.expect(l.wake());
        var waited: usize = 0;
        while (!w.done.load(.acquire) and waited < 500) : (waited += 1) try Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .real);
        const woke = w.done.load(.acquire);
        // A thread `wake` failed to reach is freed by a plain connect, so the test fails
        // instead of hanging.
        if (!woke) {
            const st = switch (l.endpoint) {
                .unix => |p| (try Io.net.UnixAddress.init(p)).connect(io),
                else => (try Io.net.IpAddress.parseIp4("127.0.0.1", l.server.socket.address.getPort())).connect(io, .{ .mode = .stream }),
            } catch null;
            if (st) |x| x.close(io);
        }
        t.join();
        errdefer std.debug.print("listener kind {d} did not wake\n", .{kind});
        try std.testing.expect(woke);
        l.deinit(a);
    }
}

// Where std reports a missing socket as Unexpected (Windows), with a stack trace.
test "a unix socket with nothing at its path is FileNotFound" {
    if (!Io.net.has_unix_sockets) return error.SkipZigTest;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const a = std.testing.allocator;
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "nobody.sock" });
    defer a.free(path);
    try std.testing.expectError(error.FileNotFound, connect(a, io, .{ .unix = path }));
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
