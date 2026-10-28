//! The text half of the studio's Expand mode: what the model wrote, cleaned of
//! the wrapping models add however they are told, and joined with the parts the
//! user wrote into the prompt a render gets. Pure, so the rules are tested here.

const std = @import("std");

/// The model's answer as prompt text: trimmed, with a leading "Prompt:"-style
/// label and quotes around the whole of it removed. Anything inside is left as
/// written. A slice of `text`.
pub fn cleanAnswer(text: []const u8) []const u8 {
    var t = trim(text);
    // A first line that only labels what follows: "Prompt:", "Image prompt:".
    if (std.mem.findScalar(u8, t, ':')) |colon| {
        const head = trim(t[0..colon]);
        if (head.len <= 24 and std.ascii.endsWithIgnoreCase(head, "prompt")) t = trim(t[colon + 1 ..]);
    }
    for ([_][2][]const u8{ .{ "\"", "\"" }, .{ "\u{201C}", "\u{201D}" }, .{ "'", "'" } }) |q| {
        if (t.len >= q[0].len + q[1].len and std.mem.startsWith(u8, t, q[0]) and std.mem.endsWith(u8, t, q[1])) {
            const inner = t[q[0].len .. t.len - q[1].len];
            // Only when the pair encloses everything: `"a" and "b"` is two quotes.
            if (std.mem.find(u8, inner, q[1]) == null) t = trim(inner);
            break;
        }
    }
    return t;
}

/// `prefix`, the model's `answer` and `suffix`, one per line, each trimmed and
/// an empty one skipped. Caller owns the result.
pub fn join(gpa: std.mem.Allocator, prefix: []const u8, answer: []const u8, suffix: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for ([_][]const u8{ trim(prefix), cleanAnswer(answer), trim(suffix) }) |part| {
        if (part.len == 0) continue;
        if (out.items.len > 0) try out.append(gpa, '\n');
        try out.appendSlice(gpa, part);
    }
    return out.toOwnedSlice(gpa);
}

fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

const testing = std.testing;

test "the three parts join one per line, and an empty part leaves no blank line" {
    const gpa = testing.allocator;
    const cases = [_][4][]const u8{
        .{ "masterpiece, oil painting", "a knight at a gate", "golden hour", "masterpiece, oil painting\na knight at a gate\ngolden hour" },
        .{ "", "a knight at a gate", "", "a knight at a gate" },
        .{ "  prefix  \n", "answer", "   ", "prefix\nanswer" },
        .{ "", "", "suffix", "suffix" },
    };
    for (cases) |c| {
        const got = try join(gpa, c[0], c[1], c[2]);
        defer gpa.free(got);
        errdefer std.debug.print("got {f}\n", .{std.zig.fmtString(got)});
        try testing.expectEqualStrings(c[3], got);
    }
}

test "an answer loses a label and enclosing quotes, and keeps quotes inside it" {
    const cases = [_][2][]const u8{
        .{ "A knight at a gate.", "A knight at a gate." },
        .{ "  Prompt: A knight at a gate.\n", "A knight at a gate." },
        .{ "Image prompt:\nA knight at a gate.", "A knight at a gate." },
        .{ "\"A knight at a gate.\"", "A knight at a gate." },
        .{ "\u{201C}A knight at a gate.\u{201D}", "A knight at a gate." },
        .{ "Prompt: \"A knight.\"", "A knight." },
        // Not a label: a colon later in an ordinary sentence.
        .{ "A sign reads: welcome home", "A sign reads: welcome home" },
        .{ "Two signs, \"left\" and \"right\"", "Two signs, \"left\" and \"right\"" },
        .{ "\"left\" and \"right\"", "\"left\" and \"right\"" },
    };
    for (cases) |c| {
        errdefer std.debug.print("{f} -> {f}\n", .{ std.zig.fmtString(c[0]), std.zig.fmtString(cleanAnswer(c[0])) });
        try testing.expectEqualStrings(c[1], cleanAnswer(c[0]));
    }
}
