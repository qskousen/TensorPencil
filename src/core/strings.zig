//! Comptime string helpers.

const std = @import("std");

/// `s` repeated `n` times, as a string literal would be.
pub fn repeat(comptime s: []const u8, comptime n: usize) *const [s.len * n:0]u8 {
    return comptime blk: {
        @setEvalBranchQuota(1000 + 10 * n);
        var out: [s.len * n:0]u8 = undefined;
        for (0..n) |i| @memcpy(out[i * s.len ..][0..s.len], s);
        const final = out;
        break :blk &final;
    };
}

/// The code point a whole UTF-8 sequence encodes, `bytes.len` being its length as
/// its lead byte declares. A single byte decodes to itself.
pub fn utf8DecodeSequence(bytes: []const u8) !u21 {
    return switch (bytes.len) {
        1 => bytes[0],
        2 => std.unicode.utf8Decode2(bytes[0..2].*),
        3 => std.unicode.utf8Decode3(bytes[0..3].*),
        4 => std.unicode.utf8Decode4(bytes[0..4].*),
        else => error.Utf8InvalidSequenceLength,
    };
}

test utf8DecodeSequence {
    try std.testing.expectEqual(@as(u21, 'a'), try utf8DecodeSequence("a"));
    try std.testing.expectEqual(@as(u21, 0xE9), try utf8DecodeSequence("\xC3\xA9"));
    try std.testing.expectEqual(@as(u21, 0x1F600), try utf8DecodeSequence("\xF0\x9F\x98\x80"));
    try std.testing.expectError(error.Utf8ExpectedContinuation, utf8DecodeSequence("\xC3a"));
}

test repeat {
    try std.testing.expectEqualStrings("abababab", repeat("ab", 4));
    try std.testing.expectEqualStrings("", repeat("ab", 0));
}
