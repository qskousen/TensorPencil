//! The HTTP/1.1 client half of the protocol, over a `Link`. Ours by necessity:
//! `std.http.Client` binds its connections to a socket it opens itself and has
//! no WebSocket upgrade, so it cannot be handed a link. Only what the protocol
//! uses: keep-alive requests with a content-length body, and the events
//! upgrade (framing is in ws.zig).
const std = @import("std");
const Io = std.Io;
const link_mod = @import("link.zig");
const Link = link_mod.Link;
const ws = @import("ws.zig");

pub const Response = struct {
    status: u16,
    /// gpa-owned.
    body: []u8,
};

pub const Head = struct {
    status: u16,
    content_length: ?u64,
    close: bool,
};

pub const Error = error{
    HttpHeadersInvalid,
    /// A response with neither content-length nor a close: this client does
    /// not speak chunked.
    UnsupportedResponse,
    /// The body exceeds the caller's cap.
    BodyTooLarge,
} || Io.Reader.DelimiterError || Io.Reader.Error || Io.Writer.Error || std.mem.Allocator.Error;

/// Status line and headers, consumed up to and including the blank line.
pub fn readHead(r: *Io.Reader) Error!Head {
    const status_line = try r.takeDelimiterInclusive('\n');
    if (status_line.len < 12 or !std.mem.startsWith(u8, status_line, "HTTP/1.")) return error.HttpHeadersInvalid;
    var h: Head = .{
        .status = std.fmt.parseInt(u16, status_line[9..12], 10) catch return error.HttpHeadersInvalid,
        .content_length = null,
        .close = false,
    };
    while (true) {
        const line = std.mem.trimEnd(u8, try r.takeDelimiterInclusive('\n'), "\r\n");
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.HttpHeadersInvalid;
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            h.content_length = std.fmt.parseInt(u64, value, 10) catch return error.HttpHeadersInvalid;
        } else if (std.ascii.eqlIgnoreCase(name, "connection")) {
            h.close = std.ascii.eqlIgnoreCase(value, "close");
        }
    }
    return h;
}

/// The request line and headers, up to and including the blank line. The
/// caller writes `body_len` body bytes and flushes.
pub fn writeHead(link: *Link, method: []const u8, target: []const u8, content_type: []const u8, body_len: u64) Io.Writer.Error!void {
    const w = link.writer();
    try w.print("{s} {s} HTTP/1.1\r\nhost: tp-serve\r\ncontent-length: {d}\r\n", .{ method, target, body_len });
    if (link.auth) |c| try w.print("authorization: Bearer {s}\r\n", .{&link_mod.secretHex(c)});
    if (content_type.len > 0) try w.print("content-type: {s}\r\n", .{content_type});
    try w.writeAll("\r\n");
}

/// One request and its whole response. `body` may be empty. The response body
/// is gpa-owned; `max_body` caps it.
pub fn request(gpa: std.mem.Allocator, link: *Link, method: []const u8, target: []const u8, content_type: []const u8, body: []const u8, max_body: usize) Error!Response {
    return requestInto(link, method, target, content_type, body, gpa, max_body);
}

/// `request` with the response body taken from `body_gpa`, which may be a
/// fixed-buffer allocator: what a transfer loop uses to allocate nothing.
pub fn requestInto(link: *Link, method: []const u8, target: []const u8, content_type: []const u8, body: []const u8, body_gpa: std.mem.Allocator, max_body: usize) Error!Response {
    try writeHead(link, method, target, content_type, body.len);
    try link.writer().writeAll(body);
    try link.writer().flush();
    return readResponse(link, body_gpa, max_body);
}

/// `request` with the body streamed from `body`, `body_len` bytes of it, so a
/// multi-MiB chunk never sits in memory whole.
pub fn requestStreaming(link: *Link, method: []const u8, target: []const u8, content_type: []const u8, body: *Io.Reader, body_len: u64, body_gpa: std.mem.Allocator, max_body: usize) Error!Response {
    try writeHead(link, method, target, content_type, body_len);
    body.streamExact64(link.writer(), body_len) catch |err| return switch (err) {
        error.EndOfStream => error.EndOfStream,
        error.ReadFailed => error.ReadFailed,
        error.WriteFailed => error.WriteFailed,
    };
    try link.writer().flush();
    return readResponse(link, body_gpa, max_body);
}

fn readResponse(link: *Link, gpa: std.mem.Allocator, max_body: usize) Error!Response {
    return readResponseFrom(link.reader(), gpa, max_body);
}

fn readResponseFrom(r: *Io.Reader, gpa: std.mem.Allocator, max_body: usize) Error!Response {
    const h = try readHead(r);
    const len = h.content_length orelse {
        if (!h.close) return error.UnsupportedResponse;
        var acc: Io.Writer.Allocating = .init(gpa);
        errdefer acc.deinit();
        // One byte past the cap is enough to know it was exceeded.
        while (acc.written().len <= max_body) {
            _ = r.stream(&acc.writer, .limited(max_body + 1 - acc.written().len)) catch |err| switch (err) {
                error.EndOfStream => break,
                error.ReadFailed => return error.ReadFailed,
                error.WriteFailed => return error.OutOfMemory,
            };
        }
        if (acc.written().len > max_body) return error.BodyTooLarge;
        return .{ .status = h.status, .body = try acc.toOwnedSlice() };
    };
    if (len > max_body) return error.BodyTooLarge;
    if (len == 0) return .{ .status = h.status, .body = &.{} };
    const out = try gpa.alloc(u8, @intCast(len));
    errdefer gpa.free(out);
    try r.readSliceAll(out);
    return .{ .status = h.status, .body = out };
}

/// Upgrade the link to the server-to-client events socket. After this the
/// link carries WebSocket frames only (`ws.readHeader` / `streamPayload`), and
/// the client answers the host's pings with `ws.writePong`.
pub fn upgradeEvents(link: *Link, target: []const u8, extra: []const std.http.Header) !void {
    var rnd: [16]u8 = undefined;
    link.io.random(&rnd);
    const k = ws.key(rnd);
    // The secret rides as one more header; `extra` is the caller's few.
    var hdrs: [8]std.http.Header = undefined;
    std.debug.assert(extra.len < hdrs.len);
    @memcpy(hdrs[0..extra.len], extra);
    var n = extra.len;
    var bearer: [7 + 64]u8 = undefined;
    if (link.auth) |c| {
        _ = std.fmt.bufPrint(&bearer, "Bearer {s}", .{&link_mod.secretHex(c)}) catch unreachable;
        hdrs[n] = .{ .name = "authorization", .value = &bearer };
        n += 1;
    }
    try ws.writeUpgrade(link.writer(), target, "tp-serve", &k, hdrs[0..n]);
    var status: u16 = 0;
    ws.readUpgrade(link.reader(), &k, &status) catch |err| switch (err) {
        error.UpgradeRefused => return if (status == 409) error.EventsTaken else if (status == 401) error.Unauthorized else error.UpgradeRefused,
        else => return err,
    };
}

test "a response head parses status, length and close" {
    var r: Io.Reader = .fixed("HTTP/1.1 202 Accepted\r\ncontent-length: 5\r\nConnection: close\r\n\r\nhello");
    const h = try readHead(&r);
    try std.testing.expectEqual(@as(u16, 202), h.status);
    try std.testing.expectEqual(@as(?u64, 5), h.content_length);
    try std.testing.expect(h.close);
    try std.testing.expectEqualStrings("hello", try r.take(5));
    var bad: Io.Reader = .fixed("nope\r\n\r\n");
    try std.testing.expectError(error.HttpHeadersInvalid, readHead(&bad));
}

test "a close-delimited body is capped like a content-length one" {
    const gpa = std.testing.allocator;
    const head = "HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n";
    var ok: Io.Reader = .fixed(head ++ "0123456789");
    const r = try readResponseFrom(&ok, gpa, 10);
    defer gpa.free(r.body);
    try std.testing.expectEqualStrings("0123456789", r.body);
    var big: Io.Reader = .fixed(head ++ "0123456789");
    try std.testing.expectError(error.BodyTooLarge, readResponseFrom(&big, gpa, 9));
    var sized: Io.Reader = .fixed("HTTP/1.1 200 OK\r\ncontent-length: 10\r\n\r\n0123456789");
    try std.testing.expectError(error.BodyTooLarge, readResponseFrom(&sized, gpa, 9));
}
