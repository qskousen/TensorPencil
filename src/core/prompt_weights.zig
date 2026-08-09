//! ComfyUI's prompt emphasis syntax, resolved into flat weighted segments.
//!
//! This is `comfy.sd1_clip`'s `escape_important` / `token_weights` / `parse_parentheses`
//! trio and nothing else: it turns a prompt string into a list of `(text, weight)` runs.
//! What a TOKENIZER then does with those runs (CLIP's 77-token chunking, T5's
//! per-segment SentencePiece pass) lives in its own file.
//!
//! It is a module because two unrelated tokenizers consume it, `clip_tokenizer` for the
//! SD family and `t5_tokenizer` for Anima's adapter branch, and a second copy of its
//! rules is the drift that makes one prompt path silently disagree with another. The
//! three non-obvious rules are documented at `segments`, which implements them.

const std = @import("std");

/// One run of prompt text and the attention weight its enclosing parentheses gave
/// it. `text` borrows from the arena passed to `segments`, still in escaped form,
/// call `unescape` before handing it to a tokenizer.
pub const Segment = struct { text: []const u8, weight: f64 };

/// The deepest `(((...)))` nesting accepted. Python's own recursion limit stops the
/// reference at ~1000 frames; a real prompt never exceeds two or three, so this is a
/// stack guard against a pathological input rather than a semantic limit, and it is
/// an error instead of a silent literal reading, since a prompt this malformed should
/// be reported rather than rendered differently than it looks.
pub const max_nesting: usize = 32;

pub const Error = error{ OutOfMemory, PromptNestingTooDeep };

/// Resolve `text`'s emphasis syntax into flat weighted segments, appended to `out`.
/// Everything allocated (including each segment's `text`) comes from `arena`.
///
/// Three rules that are not obvious, and that both consuming tokenizers must share:
///
/// 1. A bare paren MULTIPLIES by 1.1; an explicit `:w` REPLACES the weight. `((a))` is
///    1.21 but `((a:1.5))` is 1.5, not 1.65: the innermost absolute wins over every
///    enclosing multiplier. A1111 multiplies instead; that dialect is `prompt_a1111`.
/// 2. The product accumulates in f64 and narrows once, because Python's `1.1 * 1.1` is
///    1.2100000000000002 where f32 gives 1.2100001.
/// 3. Unbalanced parentheses are TEXT, not an error. `a)b` is one plain segment and
///    `(a` is one segment that simply fails the paren test; the reference lets its
///    nesting counter go negative for exactly this reason.
///
/// An empty segment is reported here and dropped by the caller: ComfyUI filters
/// `to_tokenize` for `x != ""` after unescaping, so `a () mat` contributes no token for
/// the `()`, but only the caller knows what "empty" means after its own unescaping.
pub fn segments(arena: std.mem.Allocator, out: *std.ArrayList(Segment), text: []const u8) Error!void {
    try tokenWeights(arena, out, try escape(arena, text), 1.0, 0);
}

/// `\(` and `\)` mean literal parentheses, so they are hidden from the weight parser
/// behind byte pairs that cannot occur in text. Both replacements are the same length
/// as what they replace, which is why this can be one left-to-right pass where the
/// reference does two `str.replace` sweeps.
pub fn escape(arena: std.mem.Allocator, text: []const u8) ![]u8 {
    const out = try arena.alloc(u8, text.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\\' and i + 1 < text.len and (text[i + 1] == ')' or text[i + 1] == '(')) {
            out[n] = 0;
            out[n + 1] = if (text[i + 1] == ')') 1 else 2;
            n += 2;
            i += 2;
        } else {
            out[n] = text[i];
            n += 1;
            i += 1;
        }
    }
    return out[0..n];
}

/// Undo `escape` on one segment, restoring the literal parentheses that the
/// tokenizer should see as text.
pub fn unescape(arena: std.mem.Allocator, text: []const u8) ![]u8 {
    const out = try arena.alloc(u8, text.len);
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == 0 and i + 1 < text.len and (text[i + 1] == 1 or text[i + 1] == 2)) {
            out[n] = if (text[i + 1] == 1) ')' else '(';
            n += 1;
            i += 2;
        } else {
            out[n] = text[i];
            n += 1;
            i += 1;
        }
    }
    return out[0..n];
}

/// Split `string` at its top-level parenthesised groups, so each returned item is
/// either one whole `(...)` group (outer parens included) or a run of text between
/// them. Every item is a contiguous slice of the input, which is what lets the whole
/// weight parse stay zero-copy.
fn parseParentheses(arena: std.mem.Allocator, out: *std.ArrayList([]const u8), string: []const u8) !void {
    var start: usize = 0;
    var nesting: isize = 0;
    for (string, 0..) |c, i| {
        if (c == '(') {
            if (nesting == 0 and i > start) {
                try out.append(arena, string[start..i]);
                start = i;
            }
            nesting += 1;
        } else if (c == ')') {
            nesting -= 1;
            if (nesting == 0) {
                try out.append(arena, string[start .. i + 1]);
                start = i + 1;
            }
        }
    }
    if (start < string.len) try out.append(arena, string[start..]);
}

/// See rules 1-3 in the module header. The multiply happens first and the explicit
/// assignment overwrites it, which is the reference's order and the only way to get
/// the `((a:1.5))` case right.
fn tokenWeights(
    arena: std.mem.Allocator,
    out: *std.ArrayList(Segment),
    string: []const u8,
    current_weight: f64,
    depth: usize,
) Error!void {
    if (depth > max_nesting) return error.PromptNestingTooDeep;

    var items: std.ArrayList([]const u8) = .empty;
    try parseParentheses(arena, &items, string);

    for (items.items) |item| {
        if (!(item.len >= 2 and item[0] == '(' and item[item.len - 1] == ')')) {
            try out.append(arena, .{ .text = item, .weight = current_weight });
            continue;
        }
        var x = item[1 .. item.len - 1];
        var weight = current_weight * 1.1;
        if (std.mem.lastIndexOfScalar(u8, x, ':')) |at| {
            // `at > 0`: a leading colon is text, not a weight separator. This is why
            // `(:1.5)` is a bare-paren group over the literal ":1.5" rather than an
            // empty segment at weight 1.5.
            if (at > 0) {
                if (parsePyFloat(x[at + 1 ..])) |w| {
                    weight = w;
                    x = x[0..at];
                }
            }
        }
        try tokenWeights(arena, out, x, weight, depth + 1);
    }
}

/// `float(s)` as Python accepts it: surrounding whitespace stripped, and nothing else.
/// Null when Python would have raised, which the caller treats as "this colon was not
/// a weight" and leaves the text alone.
///
/// Zig's `parseFloat` is the more permissive of the two, it also takes hex floats and
/// `_` digit separators, so those are rejected explicitly rather than silently
/// accepted, which would read `(a:0x1p4)` as a weight where ComfyUI reads it as text.
fn parsePyFloat(s: []const u8) ?f64 {
    const t = std.mem.trim(u8, s, " \t\n\r\x0b\x0c");
    if (t.len == 0) return null;
    for (t) |c| if (c == '_' or c == 'x' or c == 'X') return null;
    return std.fmt.parseFloat(f64, t) catch null;
}

const testing = std.testing;

test "a bare paren multiplies and an explicit weight replaces" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var segs: std.ArrayList(Segment) = .empty;
    try segments(a, &segs, "((a)) ((b:1.5)) ((c:1.2):1.5)");
    // Only the weights are pinned here; the text split is pinned by the fixture
    // tests in clip_tokenizer.zig and t5_tokenizer.zig.
    var seen: std.ArrayList(f64) = .empty;
    for (segs.items) |s| {
        if (std.mem.trim(u8, s.text, " ").len == 0) continue;
        try seen.append(a, s.weight);
    }
    errdefer std.debug.print("weights: {any}\n", .{seen.items});
    try testing.expectEqual(@as(usize, 3), seen.items.len);
    try testing.expectApproxEqAbs(@as(f64, 1.1 * 1.1), seen.items[0], 1e-12);
    // 1.5, not 1.5 * 1.21: the inner absolute wins.
    try testing.expectEqual(@as(f64, 1.5), seen.items[1]);
    // The shape that tells "replace" apart from "multiply": the outer `:1.5`
    // sets 1.5, then the inner `:1.2` REPLACES it, 1.2, not 1.5 * 1.2 = 1.8.
    // Verified against `comfy.sd1_clip.token_weights`.
    try testing.expectEqual(@as(f64, 1.2), seen.items[2]);
}

test "unbalanced parentheses are literal text, not an error" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Verbatim from `comfy.sd1_clip.token_weights(escape_important(t), 1.0)`.
    // `((a)` is the interesting one and it is NOT weight 1.0: `parseParentheses`
    // never closes the outer group, so the whole string arrives as one item whose
    // first and last bytes are `(` and `)`, it IS read as a group, dropping one
    // paren and applying the 1.1 multiplier to `(a`. Anything that "fixes" the
    // unbalanced input diverges here.
    const Case = struct { text: []const u8, want: []const Segment };
    for ([_]Case{
        .{ .text = "a)b", .want = &.{.{ .text = "a)b", .weight = 1.0 }} },
        .{ .text = "(a", .want = &.{.{ .text = "(a", .weight = 1.0 }} },
        .{ .text = "a(b", .want = &.{ .{ .text = "a", .weight = 1.0 }, .{ .text = "(b", .weight = 1.0 } } },
        .{ .text = ")(", .want = &.{.{ .text = ")(", .weight = 1.0 }} },
        .{ .text = "((a)", .want = &.{.{ .text = "(a", .weight = 1.1 }} },
    }) |c| {
        var segs: std.ArrayList(Segment) = .empty;
        try segments(a, &segs, c.text);
        errdefer {
            std.debug.print("text {s} ->", .{c.text});
            for (segs.items) |s| std.debug.print(" ({s}, {d})", .{ s.text, s.weight });
            std.debug.print("\n", .{});
        }
        try testing.expectEqual(c.want.len, segs.items.len);
        for (c.want, segs.items) |w, got| {
            try testing.expectEqualStrings(w.text, got.text);
            try testing.expectApproxEqAbs(w.weight, got.weight, 1e-12);
        }
    }
}

test "escaped parentheses survive the round trip and carry no weight" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var segs: std.ArrayList(Segment) = .empty;
    try segments(a, &segs, "x \\(y\\) z");
    var out: std.ArrayList(u8) = .empty;
    for (segs.items) |s| {
        try testing.expectEqual(@as(f64, 1.0), s.weight);
        try out.appendSlice(a, try unescape(a, s.text));
    }
    try testing.expectEqualStrings("x (y) z", out.items);
}

test "nesting past the guard is reported rather than read literally" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var text: std.ArrayList(u8) = .empty;
    for (0..max_nesting + 2) |_| try text.append(a, '(');
    try text.append(a, 'x');
    for (0..max_nesting + 2) |_| try text.append(a, ')');

    var segs: std.ArrayList(Segment) = .empty;
    try testing.expectError(error.PromptNestingTooDeep, segments(a, &segs, text.items));
}
