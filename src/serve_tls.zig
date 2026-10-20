//! Both ends of a remote link over tls.zig, handed to the protocol as a `Link`: the
//! daemon's server over an accepted TCP stream, and the client a dialing binary
//! installs (`installClient`). Kept out of the `serve` module so the fast suite never
//! pulls tls.zig.
const std = @import("std");
const Io = std.Io;
const tls = @import("tls");
const link_mod = @import("serve").link;
const Link = link_mod.Link;

pub const Identity = tls.config.CertKeyPair;

/// The certificate and key PEM files, absolute or relative to cwd.
pub fn loadIdentity(gpa: std.mem.Allocator, io: Io, cert_path: []const u8, key_path: []const u8) !Identity {
    return Identity.fromFilePath(gpa, io, Io.Dir.cwd(), cert_path, key_path);
}

pub fn identityFromPem(gpa: std.mem.Allocator, io: Io, cert_pem: []const u8, key_pem: []const u8) !Identity {
    return Identity.fromSlice(gpa, io, cert_pem, key_pem);
}

/// Built in place: the interfaces recover their parent by pointer.
const TlsServer = struct {
    link: Link,
    stream: Io.net.Stream,
    sr: Io.net.Stream.Reader,
    sw: Io.net.Stream.Writer,
    rng: std.Random.IoSource,
    conn: tls.Connection,
    cr: tls.Connection.Reader,
    cw: tls.Connection.Writer,
    through: link_mod.Through,
    in_buf: [link_mod.tls_buf_len]u8,
    out_buf: [link_mod.tls_buf_len]u8,
    /// The plaintext side; an HTTP head must fit.
    rd_buf: [link_mod.buf_len]u8,
    wr_buf: [link_mod.tls_buf_len]u8,

    const vtable: Link.VTable = .{ .close = close, .shutdown = shutdown };

    fn close(l: *Link, gpa: std.mem.Allocator) void {
        const s: *TlsServer = @fieldParentPtr("link", l);
        s.cw.interface.flush() catch {};
        s.conn.close() catch {};
        s.sw.interface.flush() catch {};
        s.stream.close(l.io);
        gpa.destroy(s);
    }

    fn shutdown(l: *Link) void {
        const s: *TlsServer = @fieldParentPtr("link", l);
        s.stream.shutdown(l.io, .both) catch {};
    }
};

/// Handshake on `stream` (the caller's thread, so a slow peer stalls nobody
/// else) and wrap the plaintext pair. The link is untrusted: every request
/// must carry the token. On failure the stream is still the caller's to close,
/// after it has told the watchdog.
pub fn accept(gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, ident: *Identity) !*Link {
    const s = try gpa.create(TlsServer);
    errdefer gpa.destroy(s);
    s.stream = stream;
    s.rng = .{ .io = io };
    s.sr = stream.reader(io, &s.in_buf);
    s.sw = stream.writer(io, &s.out_buf);
    s.conn = try tls.server(&s.sr.interface, &s.sw.interface, .{
        .rng = s.rng.interface(),
        .auth = ident,
        .now = Io.Clock.real.now(io),
    });
    s.cr = s.conn.reader(&s.rd_buf);
    s.cw = s.conn.writer(&s.wr_buf);
    s.through.init(&s.cw.interface, &s.sw.interface);
    s.link = .{ .io = io, .in = &s.cr.interface, .out = &s.through.interface, .trusted = false, .vt = &TlsServer.vtable };
    return &s.link;
}

/// The client end, built in place like the server's: the interfaces recover their
/// parent by pointer.
const TlsClient = struct {
    link: Link,
    stream: Io.net.Stream,
    sr: Io.net.Stream.Reader,
    sw: Io.net.Stream.Writer,
    rng: std.Random.IoSource,
    bundle: std.crypto.Certificate.Bundle,
    conn: tls.Connection,
    cr: tls.Connection.Reader,
    cw: tls.Connection.Writer,
    through: link_mod.Through,
    in_buf: [link_mod.tls_buf_len]u8,
    out_buf: [link_mod.tls_buf_len]u8,
    /// The plaintext side; an HTTP head must fit.
    rd_buf: [link_mod.buf_len]u8,
    wr_buf: [link_mod.tls_buf_len]u8,

    const vtable: Link.VTable = .{ .close = close, .shutdown = shutdown };

    fn close(l: *Link, gpa: std.mem.Allocator) void {
        const c: *TlsClient = @fieldParentPtr("link", l);
        c.cw.interface.flush() catch {};
        c.conn.close() catch {};
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

/// `link.TlsDial` over tls.zig: the one certificate is the whole trust store, so the
/// server's chain must verify under the pinned key, and the server must prove it holds
/// that key. No host name is checked (an empty one skips it): the pin names the server.
pub fn dial(gpa: std.mem.Allocator, io: Io, stream: Io.net.Stream, cert_der: []const u8, token: link_mod.Secret) !*Link {
    const c = try gpa.create(TlsClient);
    errdefer gpa.destroy(c);
    const now = Io.Clock.real.now(io);
    c.bundle = try @import("serve").x509.bundleFromDer(gpa, cert_der, now.toSeconds());
    errdefer c.bundle.deinit(gpa);
    c.stream = stream;
    c.rng = .{ .io = io };
    c.sr = stream.reader(io, &c.in_buf);
    c.sw = stream.writer(io, &c.out_buf);
    c.conn = try tls.client(&c.sr.interface, &c.sw.interface, .{
        .rng = c.rng.interface(),
        .now = now,
        .host = "",
        .root_ca = c.bundle,
    });
    c.cr = c.conn.reader(&c.rd_buf);
    c.cw = c.conn.writer(&c.wr_buf);
    c.through.init(&c.cw.interface, &c.sw.interface);
    c.link = .{ .io = io, .in = &c.cr.interface, .out = &c.through.interface, .trusted = false, .auth = token, .vt = &TlsClient.vtable };
    return &c.link;
}

/// Make this binary's `link.connect` reach `tp://` hosts. Once, before any dial.
pub fn installClient() void {
    link_mod.tls_dial = dial;
}
