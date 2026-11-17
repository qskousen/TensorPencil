//! The conditioning rows of recent side encodes (a steering term's text, the empty
//! prompt), so a batch pays one encode per term instead of one per image.
//!
//! An entry is only as good as its key: the caller hashes everything that changes
//! the encode into `key`, and empties the cache whenever the encoder itself changes
//! (a LoRA, a precision dial). `text` is compared exactly on top of the hash.
const std = @import("std");

pub const max_entries = 8;

pub const Rows = struct {
    /// Owned by the cache and valid until its next `put` or `clear`. Read only.
    data: []f32,
    seq: usize,
};

const Entry = struct {
    key: u64,
    text: []u8,
    rows: Rows,
    used: u64,
};

pub const RowCache = struct {
    entries: [max_entries]?Entry = @splat(null),
    tick: u64 = 0,

    pub fn get(self: *RowCache, key: u64, text: []const u8) ?Rows {
        for (&self.entries) |*slot| if (slot.*) |*e| {
            if (e.key == key and std.mem.eql(u8, e.text, text)) {
                self.tick += 1;
                e.used = self.tick;
                return e.rows;
            }
        };
        return null;
    }

    /// Store a copy of `data`, evicting the least recently used entry when full.
    pub fn put(self: *RowCache, gpa: std.mem.Allocator, key: u64, text: []const u8, data: []const f32, seq: usize) !Rows {
        const own_text = try gpa.dupe(u8, text);
        errdefer gpa.free(own_text);
        const own_data = try gpa.dupe(f32, data);
        errdefer gpa.free(own_data);

        var victim: usize = 0;
        for (self.entries, 0..) |slot, i| {
            const e = slot orelse {
                victim = i;
                break;
            };
            if (e.used < self.entries[victim].?.used) victim = i;
        }
        if (self.entries[victim]) |e| free(gpa, e);
        self.tick += 1;
        const rows: Rows = .{ .data = own_data, .seq = seq };
        self.entries[victim] = .{ .key = key, .text = own_text, .rows = rows, .used = self.tick };
        return rows;
    }

    pub fn clear(self: *RowCache, gpa: std.mem.Allocator) void {
        for (&self.entries) |*slot| if (slot.*) |e| {
            free(gpa, e);
            slot.* = null;
        };
    }

    fn free(gpa: std.mem.Allocator, e: Entry) void {
        gpa.free(e.text);
        gpa.free(e.rows.data);
    }
};

test "RowCache hits on key and text, and misses on either alone" {
    const gpa = std.testing.allocator;
    var c: RowCache = .{};
    defer c.clear(gpa);

    try std.testing.expect(c.get(1, "moon") == null);
    _ = try c.put(gpa, 1, "moon", &.{ 1, 2, 3 }, 1);
    const hit = c.get(1, "moon") orelse return error.TestExpectedHit;
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, hit.data);
    try std.testing.expectEqual(@as(usize, 1), hit.seq);
    try std.testing.expect(c.get(2, "moon") == null);
    try std.testing.expect(c.get(1, "mood") == null);

    c.clear(gpa);
    try std.testing.expect(c.get(1, "moon") == null);
}

test "RowCache evicts the least recently used entry" {
    const gpa = std.testing.allocator;
    var c: RowCache = .{};
    defer c.clear(gpa);

    for (0..max_entries) |i| _ = try c.put(gpa, i, "t", &.{@floatFromInt(i)}, 1);
    // Touch entry 0 so entry 1 is now the oldest.
    _ = c.get(0, "t") orelse return error.TestExpectedHit;
    _ = try c.put(gpa, 100, "t", &.{100}, 1);

    try std.testing.expect(c.get(0, "t") != null);
    try std.testing.expect(c.get(1, "t") == null);
    try std.testing.expect(c.get(100, "t") != null);
    for (2..max_entries) |i| try std.testing.expect(c.get(i, "t") != null);
}
