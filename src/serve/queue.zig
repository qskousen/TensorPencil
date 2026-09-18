//! Frames between a host and a client, in either direction: the unit both
//! queues carry and exactly what one WebSocket message holds. JSON text, or a
//! binary payload behind a `BinHeader`. gpa-owned; the receiver frees.
const std = @import("std");
const Io = std.Io;
const wire = @import("wire.zig");

pub const Frame = union(enum) {
    text: []u8,
    bin: struct { hdr: wire.BinHeader, payload: []u8 },

    pub fn deinit(self: Frame, gpa: std.mem.Allocator) void {
        switch (self) {
            .text => |t| gpa.free(t),
            .bin => |b| gpa.free(b.payload),
        }
    }

    pub fn len(self: Frame) usize {
        return switch (self) {
            .text => |t| t.len,
            .bin => |b| b.payload.len,
        };
    }
};

/// Multi-producer queue of frames. Pushing is cheap enough to do from any
/// thread; draining moves everything out under one lock.
pub const Queue = struct {
    mu: Io.Mutex = Io.Mutex.init,
    items: std.ArrayList(Frame) = .empty,
    /// Bytes queued right now.
    bytes: usize = 0,
    /// A push that would take `bytes` past this fails; 0 = no cap.
    max_bytes: usize = 0,

    /// On error the caller still owns `f`.
    pub fn push(self: *Queue, io: Io, gpa: std.mem.Allocator, f: Frame) error{ OutOfMemory, QueueFull }!void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        if (self.max_bytes != 0 and self.bytes + f.len() > self.max_bytes) return error.QueueFull;
        try self.items.append(gpa, f);
        self.bytes += f.len();
    }

    /// Moves every queued frame into `out` (cleared first), oldest first.
    pub fn drain(self: *Queue, io: Io, gpa: std.mem.Allocator, out: *std.ArrayList(Frame)) void {
        out.clearRetainingCapacity();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        out.appendSlice(gpa, self.items.items) catch {
            // Nothing lost: the frames stay queued for the next drain.
            return;
        };
        self.items.clearRetainingCapacity();
        self.bytes = 0;
    }

    /// Frees every queued frame: what a host does when its reader is gone.
    pub fn clear(self: *Queue, io: Io, gpa: std.mem.Allocator) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (self.items.items) |f| f.deinit(gpa);
        self.items.clearRetainingCapacity();
        self.bytes = 0;
    }

    pub fn deinit(self: *Queue, gpa: std.mem.Allocator) void {
        for (self.items.items) |f| f.deinit(gpa);
        self.items.deinit(gpa);
    }
};

/// A binary frame's wire form: the 32-byte header, then the payload.
pub fn binHeaderBytes(hdr: wire.BinHeader) [@sizeOf(wire.BinHeader)]u8 {
    return @bitCast(hdr);
}

pub const BinFrameError = error{ ShortFrame, BadMagic, BadKind, BadDims, OutOfMemory };

/// Split one received binary message into a frame, checking the header. The
/// payload is moved to the front of `bytes` and the buffer shrunk, so `bytes`
/// must be gpa-owned; on error the caller still owns it.
pub fn binFrameFromBytes(gpa: std.mem.Allocator, bytes: []u8) BinFrameError!Frame {
    const n = @sizeOf(wire.BinHeader);
    if (bytes.len < n) return error.ShortFrame;
    var hb: [n]u8 = undefined;
    @memcpy(&hb, bytes[0..n]);
    // The enum field is checked as an integer BEFORE the bit cast makes it one.
    const magic = std.mem.readInt(u32, hb[@offsetOf(wire.BinHeader, "magic")..][0..4], .little);
    if (magic != wire.BinHeader.magic_value) return error.BadMagic;
    const raw_kind = std.mem.readInt(u16, hb[@offsetOf(wire.BinHeader, "kind")..][0..2], .little);
    if (std.enums.fromInt(wire.BinHeader.Kind, raw_kind) == null) return error.BadKind;
    const hdr: wire.BinHeader = @bitCast(hb);
    // The dimensions and the length are what consumers size their pixel
    // arithmetic from, so they are held to the payload here rather than at each
    // of them.
    if (hdr.w > wire.BinHeader.max_edge or hdr.h > wire.BinHeader.max_edge) return error.BadDims;
    if (hdr.len != bytes.len - n) return error.ShortFrame;
    std.mem.copyForwards(u8, bytes, bytes[n..]);
    const payload = try gpa.realloc(bytes, bytes.len - n);
    return .{ .bin = .{ .hdr = hdr, .payload = payload } };
}

test "a binary frame survives its wire form" {
    const gpa = std.testing.allocator;
    const hdr: wire.BinHeader = .{ .kind = .preview_rgba, .id = 9, .rev = 3, .w = 2, .h = 1, .len = 8 };
    const payload = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var msg: std.ArrayList(u8) = .empty;
    defer msg.deinit(gpa);
    try msg.appendSlice(gpa, &binHeaderBytes(hdr));
    try msg.appendSlice(gpa, &payload);
    const f = try binFrameFromBytes(gpa, try msg.toOwnedSlice(gpa));
    defer f.deinit(gpa);
    try std.testing.expectEqual(hdr.id, f.bin.hdr.id);
    try std.testing.expectEqual(hdr.kind, f.bin.hdr.kind);
    try std.testing.expectEqualSlices(u8, &payload, f.bin.payload);
    try std.testing.expectError(error.ShortFrame, binFrameFromBytes(gpa, &[_]u8{}));
}

test "a header with an unknown kind or the wrong magic is refused, and the buffer stays the caller's" {
    const gpa = std.testing.allocator;
    const hdr: wire.BinHeader = .{ .kind = .image_rgba, .id = 1, .rev = 0, .w = 1, .h = 1, .len = 4 };
    var bad_kind = try gpa.alloc(u8, @sizeOf(wire.BinHeader) + 4);
    defer gpa.free(bad_kind);
    @memcpy(bad_kind[0..@sizeOf(wire.BinHeader)], &binHeaderBytes(hdr));
    std.mem.writeInt(u16, bad_kind[@offsetOf(wire.BinHeader, "kind")..][0..2], 0xBEEF, .little);
    try std.testing.expectError(error.BadKind, binFrameFromBytes(gpa, bad_kind));

    var bad_magic = try gpa.alloc(u8, @sizeOf(wire.BinHeader) + 4);
    defer gpa.free(bad_magic);
    @memcpy(bad_magic[0..@sizeOf(wire.BinHeader)], &binHeaderBytes(hdr));
    bad_magic[0] ^= 0xff;
    try std.testing.expectError(error.BadMagic, binFrameFromBytes(gpa, bad_magic));
}

test "a header whose dimensions or length do not match the payload is refused" {
    const gpa = std.testing.allocator;
    const n = @sizeOf(wire.BinHeader);
    // 0xFFFFFFFF squared times bytes-per-pixel is what a consumer would compute.
    const huge: wire.BinHeader = .{ .kind = .rgb_upload, .id = 0, .rev = 0, .w = 0xFFFFFFFF, .h = 0xFFFFFFFF, .len = 0 };
    var buf = try gpa.alloc(u8, n);
    defer gpa.free(buf);
    @memcpy(buf[0..n], &binHeaderBytes(huge));
    try std.testing.expectError(error.BadDims, binFrameFromBytes(gpa, buf));

    const lying: wire.BinHeader = .{ .kind = .image_rgba, .id = 1, .rev = 0, .w = 1, .h = 1, .len = 4 };
    var short = try gpa.alloc(u8, n);
    defer gpa.free(short);
    @memcpy(short[0..n], &binHeaderBytes(lying));
    try std.testing.expectError(error.ShortFrame, binFrameFromBytes(gpa, short));
}

test "a failed shrink is an error, never a slice of the wrong length" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1, .resize_fail_index = 0 });
    const gpa = failing.allocator();
    const hdr: wire.BinHeader = .{ .kind = .image_rgba, .id = 1, .rev = 0, .w = 1, .h = 1, .len = 4 };
    const bytes = try gpa.alloc(u8, @sizeOf(wire.BinHeader) + 4);
    defer gpa.free(bytes);
    @memcpy(bytes[0..@sizeOf(wire.BinHeader)], &binHeaderBytes(hdr));
    try std.testing.expectError(error.OutOfMemory, binFrameFromBytes(gpa, bytes));
}

test "the queue hands frames back oldest first and counts bytes" {
    const gpa = std.testing.allocator;
    var q: Queue = .{};
    defer q.deinit(gpa);
    try q.push(std.testing.io, gpa, .{ .text = try gpa.dupe(u8, "ab") });
    try q.push(std.testing.io, gpa, .{ .text = try gpa.dupe(u8, "cde") });
    try std.testing.expectEqual(@as(usize, 5), q.bytes);
    var out: std.ArrayList(Frame) = .empty;
    defer out.deinit(gpa);
    q.drain(std.testing.io, gpa, &out);
    defer for (out.items) |f| f.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqualStrings("ab", out.items[0].text);
    try std.testing.expectEqual(@as(usize, 0), q.bytes);
}

test "a capped queue refuses the frame that would overflow it and clear empties it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var q: Queue = .{ .max_bytes = 5 };
    defer q.deinit(gpa);
    try q.push(io, gpa, .{ .text = try gpa.dupe(u8, "abc") });
    const too_big = try gpa.dupe(u8, "defg");
    try std.testing.expectError(error.QueueFull, q.push(io, gpa, .{ .text = too_big }));
    gpa.free(too_big);
    try std.testing.expectEqual(@as(usize, 3), q.bytes);
    try q.push(io, gpa, .{ .text = try gpa.dupe(u8, "de") });
    try std.testing.expectEqual(@as(usize, 5), q.bytes);
    q.clear(io, gpa);
    try std.testing.expectEqual(@as(usize, 0), q.bytes);
    try std.testing.expectEqual(@as(usize, 0), q.items.items.len);
    try q.push(io, gpa, .{ .text = try gpa.dupe(u8, "fresh") });
    try std.testing.expectEqual(@as(usize, 5), q.bytes);
}
