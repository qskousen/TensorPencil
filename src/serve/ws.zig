//! WebSocket framing over any Io.Reader and Io.Writer, the client half plus
//! the control frames both ends send. std ships only the server half
//! (std.http.Server.WebSocket); its header types are reused here so the two
//! ends cannot disagree about the bit layout. Client frames are masked because
//! the std reader refuses unmasked ones; server frames arrive unmasked and
//! stream straight through without a payload-sized buffer.
const std = @import("std");
const Io = std.Io;
const Ws = std.http.Server.WebSocket;

pub const Opcode = Ws.Opcode;
pub const Header0 = Ws.Header0;
pub const Header1 = Ws.Header1;

/// Base64 of 16 random bytes, the Sec-WebSocket-Key value.
pub const Key = [24]u8;

pub fn key(random: [16]u8) Key {
    var k: Key = undefined;
    _ = std.base64.standard.Encoder.encode(&k, &random);
    return k;
}

/// The Sec-WebSocket-Accept a server must answer `k` with (RFC 6455 4.2.2).
pub fn accept(k: *const Key) [28]u8 {
    var sha1 = std.crypto.hash.Sha1.init(.{});
    sha1.update(k);
    sha1.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    var digest: [std.crypto.hash.Sha1.digest_length]u8 = undefined;
    sha1.final(&digest);
    var out: [28]u8 = undefined;
    _ = std.base64.standard.Encoder.encode(&out, &digest);
    return out;
}

pub fn writeUpgrade(w: *Io.Writer, target: []const u8, host: []const u8, k: *const Key, extra: []const std.http.Header) Io.Writer.Error!void {
    try w.print(
        "GET {s} HTTP/1.1\r\nhost: {s}\r\nconnection: upgrade\r\nupgrade: websocket\r\nsec-websocket-version: 13\r\nsec-websocket-key: {s}\r\n",
        .{ target, host, k },
    );
    for (extra) |h| try w.print("{s}: {s}\r\n", .{ h.name, h.value });
    try w.writeAll("\r\n");
    try w.flush();
}

pub const UpgradeError = error{
    /// The server answered with a status other than 101; `status` has it.
    UpgradeRefused,
    /// 101 without the accept value derived from our key.
    BadAccept,
    HttpHeadersInvalid,
} || Io.Reader.DelimiterError;

/// Consumes the response head. On `UpgradeRefused` the head has been consumed
/// but any body has not; the caller closes.
pub fn readUpgrade(r: *Io.Reader, k: *const Key, status: *u16) UpgradeError!void {
    const status_line = try r.takeDelimiterInclusive('\n');
    if (status_line.len < 12 or !std.mem.startsWith(u8, status_line, "HTTP/1.")) return error.HttpHeadersInvalid;
    status.* = std.fmt.parseInt(u16, status_line[9..12], 10) catch return error.HttpHeadersInvalid;

    const want = accept(k);
    var accept_ok = false;
    while (true) {
        const line = std.mem.trimEnd(u8, try r.takeDelimiterInclusive('\n'), "\r\n");
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.HttpHeadersInvalid;
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(line[0..colon], "sec-websocket-accept")) accept_ok = std.mem.eql(u8, value, &want);
    }
    if (status.* != 101) return error.UpgradeRefused;
    if (!accept_ok) return error.BadAccept;
}

pub const Header = struct {
    opcode: Opcode,
    fin: bool,
    len: u64,
    mask: ?[4]u8,
};

pub fn readHeader(r: *Io.Reader) Io.Reader.Error!Header {
    const hb = try r.takeArray(2);
    const h0: Header0 = @bitCast(hb[0]);
    const h1: Header1 = @bitCast(hb[1]);
    const len: u64 = switch (h1.payload_len) {
        .len16 => try r.takeInt(u16, .big),
        .len64 => try r.takeInt(u64, .big),
        else => @intFromEnum(h1.payload_len),
    };
    const mask: ?[4]u8 = if (h1.mask) (try r.takeArray(4)).* else null;
    return .{ .opcode = h0.opcode, .fin = h0.fin, .len = len, .mask = mask };
}

/// Streams `h.len` payload bytes into `w`, unmasking when the frame is masked.
pub fn streamPayload(r: *Io.Reader, h: Header, w: *Io.Writer) Io.Reader.StreamError!void {
    const m = h.mask orelse return r.streamExact64(w, h.len);
    var remaining = h.len;
    var i: usize = 0;
    while (remaining > 0) {
        try r.fill(1);
        const avail = r.buffered();
        const n: usize = @intCast(@min(@as(u64, avail.len), remaining));
        for (avail[0..n]) |*b| {
            b.* ^= m[i & 3];
            i += 1;
        }
        try w.writeAll(avail[0..n]);
        r.toss(n);
        remaining -= n;
    }
}

/// One masked, final frame. Not flushed.
pub fn writeFrame(w: *Io.Writer, opcode: Opcode, payload: []const u8, mask: [4]u8) Io.Writer.Error!void {
    try writeHeader(w, opcode, payload.len, true);
    try w.writeAll(&mask);
    var chunk: [512]u8 = undefined;
    var i: usize = 0;
    while (i < payload.len) {
        const n = @min(chunk.len, payload.len - i);
        for (chunk[0..n], payload[i..][0..n], 0..) |*o, b, j| o.* = b ^ mask[(i + j) & 3];
        try w.writeAll(chunk[0..n]);
        i += n;
    }
}

/// A pong answering a ping, or an unsolicited one as a heartbeat. Masked, so
/// std's server reader takes it; a server reading through `readHeader` takes
/// either. Not flushed.
pub fn writePong(w: *Io.Writer, payload: []const u8, mask: [4]u8) Io.Writer.Error!void {
    try writeFrame(w, .pong, payload, mask);
}

/// A server's unmasked control frame (ping, pong, close). Not flushed.
pub fn writeControl(w: *Io.Writer, opcode: Opcode, payload: []const u8) Io.Writer.Error!void {
    try writeHeader(w, opcode, payload.len, false);
    try w.writeAll(payload);
}

fn writeHeader(w: *Io.Writer, opcode: Opcode, len: usize, masked: bool) Io.Writer.Error!void {
    try w.writeByte(@bitCast(Header0{ .opcode = opcode, .fin = true }));
    switch (len) {
        0...125 => try w.writeByte(@bitCast(Header1{ .payload_len = @enumFromInt(@as(u7, @intCast(len))), .mask = masked })),
        126...0xffff => {
            try w.writeByte(@bitCast(Header1{ .payload_len = .len16, .mask = masked }));
            try w.writeInt(u16, @intCast(len), .big);
        },
        else => {
            try w.writeByte(@bitCast(Header1{ .payload_len = .len64, .mask = masked }));
            try w.writeInt(u64, len, .big);
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "accept matches the RFC 6455 worked example" {
    const k: Key = "dGhlIHNhbXBsZSBub25jZQ==".*;
    try testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &accept(&k));
}

test "client frames are read by std's server-side reader" {
    const payload_big = [_]u8{0x5A} ** 70000;
    for ([_][]const u8{ "", "hello", payload_big[0..200], &payload_big }) |payload| {
        var buf: [70100]u8 = undefined;
        var w: Io.Writer = .fixed(&buf);
        try writeFrame(&w, .binary, payload, .{ 1, 2, 3, 4 });
        var r: Io.Reader = .fixed(w.buffered());
        var ws: Ws = .{ .key = "", .input = &r, .output = undefined };
        const msg = try ws.readSmallMessage();
        try testing.expectEqual(Opcode.binary, msg.opcode);
        try testing.expectEqualSlices(u8, payload, msg.data);
    }
}

test "std's server frames stream through the client reader" {
    const payload = [_]u8{0xC3} ** 65537;
    var buf: [65600]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    var ws: Ws = .{ .key = "", .input = undefined, .output = &w };
    try ws.writeMessageUnflushed(&payload, .binary);
    var fixed: Io.Reader = .fixed(w.buffered());
    var got: Io.Writer.Allocating = .init(testing.allocator);
    defer got.deinit();
    const h = try readHeader(&fixed);
    try testing.expectEqual(Opcode.binary, h.opcode);
    try testing.expect(h.fin);
    try testing.expectEqual(@as(u64, payload.len), h.len);
    try testing.expectEqual(@as(?[4]u8, null), h.mask);
    try streamPayload(&fixed, h, &got.writer);
    try got.writer.flush();
    try testing.expectEqualSlices(u8, &payload, got.written());
}

test "masked frames unmask through the streaming path" {
    var payload: [3001]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @truncate(i * 7);
    var buf: [3100]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeFrame(&w, .text, &payload, .{ 0xDE, 0xAD, 0xBE, 0xEF });
    var r: Io.Reader = .fixed(w.buffered());
    var got: Io.Writer.Allocating = .init(testing.allocator);
    defer got.deinit();
    const h = try readHeader(&r);
    try testing.expectEqual(Opcode.text, h.opcode);
    try testing.expectEqual([4]u8{ 0xDE, 0xAD, 0xBE, 0xEF }, h.mask.?);
    try streamPayload(&r, h, &got.writer);
    try got.writer.flush();
    try testing.expectEqualSlices(u8, &payload, got.written());
}

test "a masked pong is read by std's server reader and an unmasked ping by ours" {
    var buf: [64]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writePong(&w, "hb", .{ 9, 8, 7, 6 });
    var r: Io.Reader = .fixed(w.buffered());
    const h = try readHeader(&r);
    try testing.expectEqual(Opcode.pong, h.opcode);
    try testing.expectEqual(@as(u64, 2), h.len);
    var got: [2]u8 = undefined;
    var gw: Io.Writer = .fixed(&got);
    try streamPayload(&r, h, &gw);
    try testing.expectEqualStrings("hb", &got);

    var w2: Io.Writer = .fixed(&buf);
    try writeControl(&w2, .ping, "");
    var r2: Io.Reader = .fixed(w2.buffered());
    const h2 = try readHeader(&r2);
    try testing.expectEqual(Opcode.ping, h2.opcode);
    try testing.expectEqual(@as(u64, 0), h2.len);
    try testing.expectEqual(@as(?[4]u8, null), h2.mask);
}

test "upgrade response is checked for status and accept" {
    const k: Key = "dGhlIHNhbXBsZSBub25jZQ==".*;
    var status: u16 = 0;
    {
        var r: Io.Reader = .fixed("HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\nconnection: upgrade\r\nsec-websocket-accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");
        try readUpgrade(&r, &k, &status);
        try testing.expectEqual(@as(u16, 101), status);
        try testing.expectEqual(@as(usize, 0), r.buffered().len);
    }
    {
        var r: Io.Reader = .fixed("HTTP/1.1 101 Switching Protocols\r\nsec-websocket-accept: AAAAAAAAAAAAAAAAAAAAAAAAAAA=\r\n\r\n");
        try testing.expectError(error.BadAccept, readUpgrade(&r, &k, &status));
    }
    {
        var r: Io.Reader = .fixed("HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n");
        try testing.expectError(error.UpgradeRefused, readUpgrade(&r, &k, &status));
        try testing.expectEqual(@as(u16, 401), status);
    }
    {
        var r: Io.Reader = .fixed("garbage\r\n\r\n");
        try testing.expectError(error.HttpHeadersInvalid, readUpgrade(&r, &k, &status));
    }
}

test "upgrade request carries the key and is parsed by std's head parser" {
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    const k = key([_]u8{7} ** 16);
    try writeUpgrade(&w, "/v1/events", "gpu-box", &k, &.{.{ .name = "authorization", .value = "Bearer x" }});
    const head = try std.http.Server.Request.Head.parse(w.buffered());
    try testing.expectEqual(std.http.Method.GET, head.method);
    try testing.expectEqualStrings("/v1/events", head.target);
    var found = false;
    var it = std.http.HeaderIterator.init(w.buffered());
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "sec-websocket-key")) {
            try testing.expectEqualStrings(&k, h.value);
            found = true;
        }
    }
    try testing.expect(found);
}
