//! Self-signed ECDSA P-256 certificates without openssl: a small DER writer,
//! the X.509 v3 and RFC 5915 key encodings, PEM, and a one-cert pin bundle.
//! std verifies what this writes, which is the test.
const std = @import("std");
const Io = std.Io;
const Certificate = std.crypto.Certificate;

pub const Ecdsa = std.crypto.sign.ecdsa.EcdsaP256Sha256;

// OID contents (tag and length are written by the encoder).
const oid_ecdsa_with_sha256 = [_]u8{ 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02 };
const oid_id_ec_public_key = [_]u8{ 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01 };
const oid_prime256v1 = [_]u8{ 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07 };
const oid_common_name = [_]u8{ 0x55, 0x04, 0x03 };
const oid_subject_alt_name = [_]u8{ 0x55, 0x1D, 0x11 };

pub const Tag = struct {
    pub const integer: u8 = 0x02;
    pub const bitstring: u8 = 0x03;
    pub const octetstring: u8 = 0x04;
    pub const oid: u8 = 0x06;
    pub const utf8string: u8 = 0x0C;
    pub const sequence: u8 = 0x30;
    pub const set: u8 = 0x31;
    pub const generalized_time: u8 = 0x18;
    /// Constructed, context-specific [n].
    pub fn context(n: u8) u8 {
        return 0xA0 | n;
    }
    /// Primitive, context-specific [n].
    pub fn contextPrimitive(n: u8) u8 {
        return 0x80 | n;
    }
};

/// DER writer over a fixed buffer. A container is `begin`, its contents, `end`;
/// the length is patched in at `end`, so nothing is sized up front.
pub const Der = struct {
    buf: []u8,
    pos: usize = 0,

    pub const Error = error{NoSpaceLeft};

    pub fn init(buf: []u8) Der {
        return .{ .buf = buf };
    }

    pub fn bytes(d: *const Der) []u8 {
        return d.buf[0..d.pos];
    }

    pub fn raw(d: *Der, b: []const u8) Error!void {
        if (d.pos + b.len > d.buf.len) return error.NoSpaceLeft;
        @memcpy(d.buf[d.pos..][0..b.len], b);
        d.pos += b.len;
    }

    /// Writes the tag and reserves three length bytes; returns the mark for `end`.
    pub fn begin(d: *Der, tag: u8) Error!usize {
        try d.raw(&.{ tag, 0, 0, 0 });
        return d.pos - 4;
    }

    pub fn end(d: *Der, mark: usize) void {
        const content_start = mark + 4;
        const len = d.pos - content_start;
        var lb: [3]u8 = undefined;
        const n = encodeLength(len, &lb);
        const dst = mark + 1 + n;
        std.mem.copyForwards(u8, d.buf[dst .. dst + len], d.buf[content_start .. content_start + len]);
        @memcpy(d.buf[mark + 1 ..][0..n], lb[0..n]);
        d.pos = dst + len;
    }

    pub fn tlv(d: *Der, tag: u8, content: []const u8) Error!void {
        const m = try d.begin(tag);
        try d.raw(content);
        d.end(m);
    }

    pub fn oid(d: *Der, content: []const u8) Error!void {
        return d.tlv(Tag.oid, content);
    }

    /// Non-negative INTEGER from big-endian bytes, minimal form.
    pub fn integer(d: *Der, be: []const u8) Error!void {
        var v = std.mem.trimStart(u8, be, &.{0});
        if (v.len == 0) v = &.{0};
        const m = try d.begin(Tag.integer);
        if (v[0] & 0x80 != 0) try d.raw(&.{0});
        try d.raw(v);
        d.end(m);
    }

    /// BIT STRING with no unused bits.
    pub fn bitString(d: *Der, content: []const u8) Error!void {
        const m = try d.begin(Tag.bitstring);
        try d.raw(&.{0});
        try d.raw(content);
        d.end(m);
    }

    fn encodeLength(len: usize, out: *[3]u8) usize {
        if (len < 0x80) {
            out[0] = @intCast(len);
            return 1;
        }
        if (len < 0x100) {
            out[0] = 0x81;
            out[1] = @intCast(len);
            return 2;
        }
        std.debug.assert(len < 0x10000);
        out[0] = 0x82;
        out[1] = @intCast(len >> 8);
        out[2] = @intCast(len & 0xff);
        return 3;
    }
};

pub const Options = struct {
    common_name: []const u8 = "tp-serve",
    /// dNSName entries of a subjectAltName extension; empty writes no extension.
    dns_names: []const []const u8 = &.{},
    /// Unix seconds. std rejects a certificate outside this window at handshake.
    not_before: i64,
    not_after: i64,
};

pub const Identity = struct {
    key_pair: Ecdsa.KeyPair,
    cert_der: []u8,
    /// RFC 5915 ECPrivateKey, what an "EC PRIVATE KEY" PEM block holds.
    key_der: []u8,

    pub fn deinit(self: *Identity, gpa: std.mem.Allocator) void {
        gpa.free(self.cert_der);
        std.crypto.secureZero(u8, self.key_der);
        gpa.free(self.key_der);
        self.* = undefined;
    }

    pub fn certPemAlloc(self: *const Identity, gpa: std.mem.Allocator) ![]u8 {
        return pemAlloc(gpa, "CERTIFICATE", self.cert_der);
    }

    pub fn keyPemAlloc(self: *const Identity, gpa: std.mem.Allocator) ![]u8 {
        return pemAlloc(gpa, "EC PRIVATE KEY", self.key_der);
    }

    /// SHA-256 of the DER, the fingerprint every TLS tool prints.
    pub fn fingerprint(self: *const Identity) [32]u8 {
        var out: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(self.cert_der, &out, .{});
        return out;
    }
};

/// A fresh random serial; everything else is `selfSignedWithSerial`.
pub fn selfSigned(gpa: std.mem.Allocator, io: Io, key_pair: Ecdsa.KeyPair, opts: Options) !Identity {
    var serial: [16]u8 = undefined;
    io.random(&serial);
    return selfSignedWithSerial(gpa, key_pair, serial, opts);
}

pub fn selfSignedWithSerial(gpa: std.mem.Allocator, key_pair: Ecdsa.KeyPair, serial_in: [16]u8, opts: Options) !Identity {
    // Positive and with a non-zero leading byte, so the INTEGER is 16 bytes.
    var serial = serial_in;
    serial[0] = (serial[0] & 0x7f) | 0x40;

    const pub_sec1 = key_pair.public_key.toUncompressedSec1();

    var tbs_buf: [3072]u8 = undefined;
    var t = Der.init(&tbs_buf);
    {
        const tbs = try t.begin(Tag.sequence);
        {
            const v = try t.begin(Tag.context(0));
            try t.tlv(Tag.integer, &.{2}); // v3
            t.end(v);
        }
        try t.integer(&serial);
        try algorithmId(&t);
        try name(&t, opts.common_name);
        {
            const v = try t.begin(Tag.sequence);
            var nb: [15]u8 = undefined;
            var na: [15]u8 = undefined;
            generalizedTime(opts.not_before, &nb);
            generalizedTime(opts.not_after, &na);
            try t.tlv(Tag.generalized_time, &nb);
            try t.tlv(Tag.generalized_time, &na);
            t.end(v);
        }
        try name(&t, opts.common_name);
        {
            const spki = try t.begin(Tag.sequence);
            const alg = try t.begin(Tag.sequence);
            try t.oid(&oid_id_ec_public_key);
            try t.oid(&oid_prime256v1);
            t.end(alg);
            try t.bitString(&pub_sec1);
            t.end(spki);
        }
        if (opts.dns_names.len > 0) {
            const x = try t.begin(Tag.context(3));
            const exts = try t.begin(Tag.sequence);
            const ext = try t.begin(Tag.sequence);
            try t.oid(&oid_subject_alt_name);
            const val = try t.begin(Tag.octetstring);
            const names = try t.begin(Tag.sequence);
            for (opts.dns_names) |n| try t.tlv(Tag.contextPrimitive(2), n);
            t.end(names);
            t.end(val);
            t.end(ext);
            t.end(exts);
            t.end(x);
        }
        t.end(tbs);
    }
    const tbs_bytes = t.bytes();

    // The signature covers the whole TBS element, tag and length included.
    const sig = try key_pair.sign(tbs_bytes, null);
    var sig_buf: [Ecdsa.Signature.der_encoded_length_max]u8 = undefined;
    const sig_der = sig.toDer(&sig_buf);

    var cert_buf: [4096]u8 = undefined;
    var c = Der.init(&cert_buf);
    {
        const cert = try c.begin(Tag.sequence);
        try c.raw(tbs_bytes);
        try algorithmId(&c);
        try c.bitString(sig_der);
        c.end(cert);
    }

    var key_buf: [256]u8 = undefined;
    var k = Der.init(&key_buf);
    {
        const seq = try k.begin(Tag.sequence);
        try k.tlv(Tag.integer, &.{1});
        try k.tlv(Tag.octetstring, &key_pair.secret_key.toBytes());
        const params = try k.begin(Tag.context(0));
        try k.oid(&oid_prime256v1);
        k.end(params);
        const pk = try k.begin(Tag.context(1));
        try k.bitString(&pub_sec1);
        k.end(pk);
        k.end(seq);
    }

    const cert_der = try gpa.dupe(u8, c.bytes());
    errdefer gpa.free(cert_der);
    const key_der = try gpa.dupe(u8, k.bytes());
    return .{ .key_pair = key_pair, .cert_der = cert_der, .key_der = key_der };
}

fn algorithmId(d: *Der) Der.Error!void {
    const m = try d.begin(Tag.sequence);
    try d.oid(&oid_ecdsa_with_sha256);
    d.end(m);
}

fn name(d: *Der, cn: []const u8) Der.Error!void {
    const m = try d.begin(Tag.sequence);
    const s = try d.begin(Tag.set);
    const a = try d.begin(Tag.sequence);
    try d.oid(&oid_common_name);
    try d.tlv(Tag.utf8string, cn);
    d.end(a);
    d.end(s);
    d.end(m);
}

/// "YYYYMMDDHHMMSSZ". Seconds before 1970 are clamped to the epoch.
fn generalizedTime(secs: i64, out: *[15]u8) void {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(secs, 0)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    _ = std.fmt.bufPrint(out, "{d:0>4}{d:0>2}{d:0>2}{d:0>2}{d:0>2}{d:0>2}Z", .{
        yd.year,
        md.month.numeric(),
        @as(u8, md.day_index) + 1,
        ds.getHoursIntoDay(),
        ds.getMinutesIntoHour(),
        ds.getSecondsIntoMinute(),
    }) catch unreachable;
}

pub fn pemAlloc(gpa: std.mem.Allocator, label: []const u8, der: []const u8) ![]u8 {
    const enc = std.base64.standard.Encoder;
    const b64 = try gpa.alloc(u8, enc.calcSize(der.len));
    defer gpa.free(b64);
    _ = enc.encode(b64, der);

    var aw: Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try aw.writer.print("-----BEGIN {s}-----\n", .{label});
    var i: usize = 0;
    while (i < b64.len) : (i += 64) {
        try aw.writer.writeAll(b64[i..@min(i + 64, b64.len)]);
        try aw.writer.writeByte('\n');
    }
    try aw.writer.print("-----END {s}-----\n", .{label});
    return aw.toOwnedSlice();
}

/// The DER inside one PEM block, whatever its label. gpa-owned.
pub fn pemToDerAlloc(gpa: std.mem.Allocator, pem: []const u8) ![]u8 {
    var b64: std.ArrayList(u8) = .empty;
    defer b64.deinit(gpa);
    var lines = std.mem.splitScalar(u8, pem, '\n');
    var inside = false;
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "-----BEGIN")) {
            inside = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, "-----END")) break;
        if (inside) try b64.appendSlice(gpa, line);
    }
    if (!inside) return error.NotPem;
    const dec = std.base64.standard.Decoder;
    const out = try gpa.alloc(u8, try dec.calcSizeForSlice(b64.items));
    errdefer gpa.free(out);
    try dec.decode(out, b64.items);
    return out;
}

/// A bundle holding exactly this certificate: what a client pins against.
/// `parseCert` drops an expired certificate without an error, so that case is
/// turned into one here.
pub fn bundleFromDer(gpa: std.mem.Allocator, der: []const u8, now_sec: i64) !Certificate.Bundle {
    var b: Certificate.Bundle = .empty;
    errdefer b.deinit(gpa);
    try b.bytes.appendSlice(gpa, der);
    try b.parseCert(gpa, 0, now_sec);
    if (b.map.size != 1) return error.CertificateRejected;
    return b;
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

fn testIdentity(dns: []const []const u8) !Identity {
    const seed = [_]u8{0x42} ** Ecdsa.KeyPair.seed_length;
    const kp = try Ecdsa.KeyPair.generateDeterministic(seed);
    return selfSignedWithSerial(testing.allocator, kp, [_]u8{0x11} ** 16, .{
        .common_name = "tp-serve test",
        .dns_names = dns,
        .not_before = 1_700_000_000,
        .not_after = 2_000_000_000,
    });
}

test "der length forms round-trip through std's element parser" {
    var buf: [70000]u8 = undefined;
    var d = Der.init(&buf);
    const content = [_]u8{0xAB} ** 65000;
    for ([_]usize{ 0, 1, 127, 128, 255, 256, 65000 }) |n| {
        d.pos = 0;
        try d.tlv(Tag.octetstring, content[0..n]);
        const el = try Certificate.der.Element.parse(d.bytes(), 0);
        try testing.expectEqual(Certificate.der.Tag.octetstring, el.identifier.tag);
        try testing.expectEqual(n, el.slice.end - el.slice.start);
        try testing.expectEqual(d.pos, el.slice.end);
    }
}

test "der integer is minimal and non-negative" {
    var buf: [64]u8 = undefined;
    var d = Der.init(&buf);
    try d.integer(&.{ 0, 0, 0x80, 1 });
    try testing.expectEqualSlices(u8, &.{ 0x02, 0x03, 0x00, 0x80, 0x01 }, d.bytes());
    d.pos = 0;
    try d.integer(&.{ 0, 0 });
    try testing.expectEqualSlices(u8, &.{ 0x02, 0x01, 0x00 }, d.bytes());
}

test "generalized time" {
    var out: [15]u8 = undefined;
    generalizedTime(0, &out);
    try testing.expectEqualStrings("19700101000000Z", &out);
    generalizedTime(1_700_000_000, &out);
    try testing.expectEqualStrings("20231114221320Z", &out);
}

test "self-signed certificate parses, verifies, and rejects tampering" {
    var id = try testIdentity(&.{"localhost"});
    defer id.deinit(testing.allocator);
    const now: i64 = 1_800_000_000;

    const parsed = try (Certificate{ .buffer = id.cert_der, .index = 0 }).parse();
    try testing.expectEqual(Certificate.Version.v3, parsed.version);
    try testing.expectEqualStrings("tp-serve test", parsed.commonName());
    try testing.expectEqual(Certificate.Algorithm.ecdsa_with_SHA256, parsed.signature_algorithm);
    try testing.expectEqual(Certificate.NamedCurve.X9_62_prime256v1, parsed.pub_key_algo.X9_62_id_ecPublicKey);
    try testing.expectEqual(@as(u64, 1_700_000_000), parsed.validity.not_before);
    try testing.expectEqual(@as(u64, 2_000_000_000), parsed.validity.not_after);
    try parsed.verifyHostName("localhost");
    try testing.expectError(error.CertificateHostMismatch, parsed.verifyHostName("evil"));

    var bundle = try bundleFromDer(testing.allocator, id.cert_der, now);
    defer bundle.deinit(testing.allocator);
    try bundle.verify(parsed, now);
    try testing.expectError(error.CertificateExpired, bundle.verify(parsed, 2_100_000_000));
    try testing.expectError(error.CertificateNotYetValid, bundle.verify(parsed, 1_600_000_000));

    // Teeth: one flipped signature byte must fail, one flipped TBS byte too.
    const bad = try testing.allocator.dupe(u8, id.cert_der);
    defer testing.allocator.free(bad);
    bad[bad.len - 1] ^= 0x01;
    const bad_parsed = try (Certificate{ .buffer = bad, .index = 0 }).parse();
    try testing.expectError(error.CertificateSignatureInvalid, bundle.verify(bad_parsed, now));

    @memcpy(bad, id.cert_der);
    bad[parsed.subject_slice.end - 1] ^= 0x01;
    // The signed bytes changed under an intact issuer name.
    const bad2 = try (Certificate{ .buffer = bad, .index = 0 }).parse();
    try testing.expectError(error.CertificateSignatureInvalid, bundle.verify(bad2, now));

    @memcpy(bad, id.cert_der);
    bad[parsed.issuer_slice.end - 1] ^= 0x01;
    // The issuer name changed, so the pin bundle has no entry for it.
    const bad3 = try (Certificate{ .buffer = bad, .index = 0 }).parse();
    try testing.expectError(error.CertificateIssuerNotFound, bundle.verify(bad3, now));

    // A different key's certificate under the same name is not this pin.
    const other_seed = [_]u8{0x43} ** Ecdsa.KeyPair.seed_length;
    const other_kp = try Ecdsa.KeyPair.generateDeterministic(other_seed);
    var other = try selfSignedWithSerial(testing.allocator, other_kp, [_]u8{0x11} ** 16, .{
        .common_name = "tp-serve test",
        .not_before = 1_700_000_000,
        .not_after = 2_000_000_000,
    });
    defer other.deinit(testing.allocator);
    const other_parsed = try (Certificate{ .buffer = other.cert_der, .index = 0 }).parse();
    try testing.expectError(error.CertificateSignatureInvalid, bundle.verify(other_parsed, now));
}

test "no SAN means no extension and only the common name checks" {
    var id = try testIdentity(&.{});
    defer id.deinit(testing.allocator);
    const parsed = try (Certificate{ .buffer = id.cert_der, .index = 0 }).parse();
    try testing.expectEqual(@as(u32, 0), parsed.subject_alt_name_slice.end);
    try parsed.verifyHostName("tp-serve test");
}

test "expired certificate is refused by the pin bundle" {
    var id = try testIdentity(&.{});
    defer id.deinit(testing.allocator);
    try testing.expectError(error.CertificateRejected, bundleFromDer(testing.allocator, id.cert_der, 2_100_000_000));
}

test "pem wraps at 64 columns and decodes back to the der" {
    var id = try testIdentity(&.{});
    defer id.deinit(testing.allocator);
    const pem = try id.certPemAlloc(testing.allocator);
    defer testing.allocator.free(pem);
    try testing.expect(std.mem.startsWith(u8, pem, "-----BEGIN CERTIFICATE-----\n"));
    try testing.expect(std.mem.endsWith(u8, pem, "-----END CERTIFICATE-----\n"));
    var lines = std.mem.splitScalar(u8, pem, '\n');
    _ = lines.next();
    var b64: std.ArrayList(u8) = .empty;
    defer b64.deinit(testing.allocator);
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "-----END")) break;
        try testing.expect(line.len <= 64);
        try b64.appendSlice(testing.allocator, line);
    }
    const dec = std.base64.standard.Decoder;
    const out = try testing.allocator.alloc(u8, try dec.calcSizeForSlice(b64.items));
    defer testing.allocator.free(out);
    try dec.decode(out, b64.items);
    try testing.expectEqualSlices(u8, id.cert_der, out);
    const back = try pemToDerAlloc(testing.allocator, pem);
    defer testing.allocator.free(back);
    try testing.expectEqualSlices(u8, id.cert_der, back);
    try testing.expectError(error.NotPem, pemToDerAlloc(testing.allocator, "not a pem"));
}

test "key der is an RFC 5915 ECPrivateKey holding the secret scalar" {
    var id = try testIdentity(&.{});
    defer id.deinit(testing.allocator);
    const der = Certificate.der;
    const seq = try der.Element.parse(id.key_der, 0);
    const version = try der.Element.parse(id.key_der, seq.slice.start);
    try testing.expectEqualSlices(u8, &.{1}, id.key_der[version.slice.start..version.slice.end]);
    const key = try der.Element.parse(id.key_der, version.slice.end);
    try testing.expectEqual(der.Tag.octetstring, key.identifier.tag);
    try testing.expectEqualSlices(u8, &id.key_pair.secret_key.toBytes(), id.key_der[key.slice.start..key.slice.end]);
    const params = try der.Element.parse(id.key_der, key.slice.end);
    const curve = try der.Element.parse(id.key_der, params.slice.start);
    try testing.expectEqual(Certificate.NamedCurve.X9_62_prime256v1, try Certificate.parseNamedCurve(id.key_der, curve));
}
