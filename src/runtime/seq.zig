//! A stepper's per-sequence context, held apart from its weights so a second
//! prompt can run on the same model without touching the first one's KV.
//!
//! Each stepper declares a `Seq` whose fields are named exactly like the
//! stepper's own per-sequence fields, and exchanges the two with `swapShared`.
//! A `Seq` field the stepper has no field of that name for (a split's host
//! side, which lives inside `split`) is the stepper's to swap by hand.
//!
//! No layer may move between host and device while a sequence is parked: a
//! migration copies the live sequence's state only, and the parked one would
//! come back with KV on the wrong side.

const std = @import("std");

/// Exchange every field `S` shares by name with `lm.*`.
pub fn swapShared(comptime S: type, lm: anytype, s: *S) void {
    const L = @typeInfo(@TypeOf(lm)).pointer.child;
    inline for (@typeInfo(S).@"struct".field_names) |name| {
        if (comptime @hasField(L, name)) {
            const T = @TypeOf(@field(s.*, name));
            std.mem.swap(T, &@field(lm.*, name), &@field(s.*, name));
        }
    }
}

test "swapShared exchanges the named fields and leaves the rest" {
    const Lm = struct { len: usize, cache: []const u8, weights: u32 };
    const S = struct { len: usize, cache: []const u8, host: ?u8 = null };
    var lm: Lm = .{ .len = 7, .cache = "main", .weights = 42 };
    var s: S = .{ .len = 0, .cache = "side", .host = 3 };
    swapShared(S, &lm, &s);
    try std.testing.expectEqual(@as(usize, 0), lm.len);
    try std.testing.expectEqualStrings("side", lm.cache);
    try std.testing.expectEqual(@as(u32, 42), lm.weights);
    try std.testing.expectEqual(@as(usize, 7), s.len);
    try std.testing.expectEqualStrings("main", s.cache);
    try std.testing.expectEqual(@as(?u8, 3), s.host);
    swapShared(S, &lm, &s);
    try std.testing.expectEqual(@as(usize, 7), lm.len);
    try std.testing.expectEqualStrings("main", lm.cache);
}
