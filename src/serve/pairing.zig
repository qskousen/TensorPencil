//! A remote host's state directory: its certificate and key, and the HASH of
//! its token, never the token. A copied directory yields a peer that can be
//! impersonated but not a credential to reach it with. The pairing string a
//! client pastes (`link.Endpoint.tls`'s text form) is the only place the
//! token appears, and it exists once, when the token is minted.
const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");
const link = @import("link.zig");
const x509 = @import("x509.zig");

pub const Paths = struct {
    cert: []const u8,
    key: []const u8,
    token_hash: []const u8,

    pub fn init(arena: std.mem.Allocator, dir: []const u8) !Paths {
        return .{
            .cert = try std.fs.path.join(arena, &.{ dir, "cert.pem" }),
            .key = try std.fs.path.join(arena, &.{ dir, "key.pem" }),
            .token_hash = try std.fs.path.join(arena, &.{ dir, "token.hash" }),
        };
    }
};

/// Ten years: std refuses an expired certificate at the handshake, so a short
/// life is a scheduled outage.
pub const cert_life_s: i64 = 10 * 365 * std.time.s_per_day;

fn writePrivate(io: Io, path: []const u8, data: []const u8) !void {
    const perms: Io.Dir.Permissions = if (builtin.os.tag == .windows) .default_file else .fromMode(0o600);
    const f = try Io.Dir.cwd().createFile(io, path, .{ .permissions = perms });
    defer f.close(io);
    try f.writeStreamingAll(io, data);
}

/// Create the directory (0700) and the certificate and key when either is
/// missing. True when they were minted this call.
pub fn ensureIdentity(gpa: std.mem.Allocator, io: Io, dir: []const u8, p: Paths, now_sec: i64) !bool {
    const dir_perms: Io.Dir.Permissions = if (builtin.os.tag == .windows) .default_dir else .fromMode(0o700);
    _ = Io.Dir.cwd().createDirPathStatus(io, dir, dir_perms) catch {};
    const have_cert = Io.Dir.cwd().statFile(io, p.cert, .{}) catch null;
    const have_key = Io.Dir.cwd().statFile(io, p.key, .{}) catch null;
    if (have_cert != null and have_key != null) return false;
    var id = try x509.selfSigned(gpa, io, x509.Ecdsa.KeyPair.generate(io), .{
        .not_before = now_sec - 60,
        .not_after = now_sec + cert_life_s,
    });
    defer id.deinit(gpa);
    const cert_pem = try id.certPemAlloc(gpa);
    defer gpa.free(cert_pem);
    const key_pem = try id.keyPemAlloc(gpa);
    defer gpa.free(key_pem);
    try writePrivate(io, p.key, key_pem);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = p.cert, .data = cert_pem });
    return true;
}

/// The stored token hash, null when there is none (or it does not parse).
pub fn readTokenHash(io: Io, p: Paths) ?link.SecretHash {
    var buf: [256]u8 = undefined;
    const f = Io.Dir.cwd().openFile(io, p.token_hash, .{}) catch return null;
    defer f.close(io);
    var r = f.reader(io, &buf);
    const text = r.interface.takeDelimiterExclusive('\n') catch |err| switch (err) {
        error.EndOfStream => r.interface.buffered(),
        else => return null,
    };
    return link.parseSecretHex(std.mem.trim(u8, text, " \r\n"));
}

/// Mint a token, store only its hash, return the token: the caller prints it
/// now or never.
pub fn newToken(io: Io, p: Paths) !link.Secret {
    var token: link.Secret = undefined;
    io.random(&token);
    try writePrivate(io, p.token_hash, &link.secretHex(link.hashSecret(token)));
    return token;
}

/// The certificate's DER as the pairing string carries it (base64url).
pub fn certB64Alloc(arena: std.mem.Allocator, io: Io, p: Paths) ![]const u8 {
    const pem = try Io.Dir.cwd().readFileAlloc(io, p.cert, arena, .limited(64 << 10));
    const der = try x509.pemToDerAlloc(arena, pem);
    const enc = std.base64.url_safe_no_pad.Encoder;
    const b64 = try arena.alloc(u8, enc.calcSize(der.len));
    return enc.encode(b64, der);
}

pub fn pairingAlloc(arena: std.mem.Allocator, io: Io, p: Paths, host: []const u8, port: u16, token: link.Secret) ![]u8 {
    const ep: link.Endpoint = .{ .tls = .{ .host = host, .port = port, .cert_b64 = try certB64Alloc(arena, io, p), .token = token } };
    return std.fmt.allocPrint(arena, "{f}", .{ep});
}

test "the state directory holds a reusable identity and the token's hash, never the token" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const state = try std.fs.path.join(a, &.{ dir, "serve" });
    const p = try Paths.init(a, state);
    const now: i64 = 1_800_000_000;

    try std.testing.expect(try ensureIdentity(gpa, io, state, p, now));
    try std.testing.expect(!try ensureIdentity(gpa, io, state, p, now));
    const der = try x509.pemToDerAlloc(a, try Io.Dir.cwd().readFileAlloc(io, p.cert, a, .limited(64 << 10)));
    const parsed = try (std.crypto.Certificate{ .buffer = der, .index = 0 }).parse();
    try std.testing.expectEqual(@as(u64, @intCast(now + cert_life_s)), parsed.validity.not_after);

    try std.testing.expect(readTokenHash(io, p) == null);
    const token = try newToken(io, p);
    const stored = try Io.Dir.cwd().readFileAlloc(io, p.token_hash, a, .limited(256));
    try std.testing.expectEqualStrings(&link.secretHex(link.hashSecret(token)), stored);
    try std.testing.expect(!std.mem.eql(u8, &link.secretHex(token), stored));
    try std.testing.expectEqual(link.hashSecret(token), readTokenHash(io, p).?);
    // A second daemon start reads the same hash; a new token replaces it.
    const again = try newToken(io, p);
    try std.testing.expect(!std.mem.eql(u8, &again, &token));
    try std.testing.expectEqual(link.hashSecret(again), readTokenHash(io, p).?);

    // The pairing string parses back to the certificate on disk and the token.
    const pair = try pairingAlloc(a, io, p, "lydia", 7777, again);
    const ep = link.Endpoint.parse(pair) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("lydia", ep.tls.host);
    try std.testing.expectEqual(again, ep.tls.token);
    var der_buf: [link.max_cert_der]u8 = undefined;
    try std.testing.expectEqualSlices(u8, der, try ep.tls.certDer(&der_buf));
}
