//! The server end of a remote link: tls.zig's server over an accepted TCP
//! stream, handed to the protocol as a `Link`. Kept out of the `serve` module
//! so the fast suite never pulls tls.zig; the client end is std's, in
//! `serve/link.zig`.
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
