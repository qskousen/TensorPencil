//! Unicode NFC, what transformers' slow `Qwen2Tokenizer` applies to a prompt before
//! splitting it. Canonical decomposition (tables plus algorithmic Hangul), canonical
//! ordering of combining marks, then canonical composition, per UAX #15. Tables are
//! `nfc_tables.zig`, generated from python's `unicodedata`, the module the reference
//! calls.

const std = @import("std");
const t = @import("nfc_tables.zig");

const s_base: u21 = 0xAC00;
const l_base: u21 = 0x1100;
const v_base: u21 = 0x1161;
const t_base: u21 = 0x11A7;
const l_count = 19;
const v_count = 21;
const t_count = 28;
const n_count = v_count * t_count;
const s_count = l_count * n_count;

/// `text` in NFC, or null when it already is: ASCII, or input that is not valid
/// UTF-8 (passed on as the bytes it is). Caller frees.
pub fn normalize(gpa: std.mem.Allocator, text: []const u8) !?[]u8 {
    for (text) |c| {
        if (c >= 0x80) break;
    } else return null;
    if (!std.unicode.utf8ValidateSlice(text)) return null;

    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(gpa);
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepoint()) |cp| try decompose(gpa, &cps, cp);
    reorder(cps.items);
    const n = compose(cps.items);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, text.len);
    for (cps.items[0..n]) |cp| {
        var buf: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(cp, &buf) catch unreachable;
        try out.appendSlice(gpa, buf[0..len]);
    }
    if (std.mem.eql(u8, out.items, text)) {
        out.deinit(gpa);
        return null;
    }
    return try out.toOwnedSlice(gpa);
}

fn decompose(gpa: std.mem.Allocator, out: *std.ArrayList(u21), cp: u21) !void {
    if (cp >= s_base and cp < s_base + s_count) {
        const s = cp - s_base;
        try out.append(gpa, l_base + s / n_count);
        try out.append(gpa, v_base + (s % n_count) / t_count);
        if (s % t_count != 0) try out.append(gpa, t_base + s % t_count);
        return;
    }
    const i = std.sort.binarySearch(u21, &t.decomp_cp, cp, order) orelse return out.append(gpa, cp);
    for (t.decomp_pool[t.decomp_off[i]..t.decomp_off[i + 1]]) |d| try decompose(gpa, out, d);
}

fn order(key: u21, x: u21) std.math.Order {
    return std.math.order(key, x);
}

pub fn combiningClass(cp: u21) u8 {
    const i = std.sort.binarySearch(t.Ccc, &t.ccc, cp, struct {
        fn f(key: u21, e: t.Ccc) std.math.Order {
            return std.math.order(key, e.cp);
        }
    }.f) orelse return 0;
    return t.ccc[i].class;
}

/// A stable sort by class over each run of non-starters. Runs are short, so insertion.
fn reorder(cps: []u21) void {
    var i: usize = 1;
    while (i < cps.len) : (i += 1) {
        const c = combiningClass(cps[i]);
        if (c == 0) continue;
        var j = i;
        while (j > 0) : (j -= 1) {
            const p = combiningClass(cps[j - 1]);
            if (p == 0 or p <= c) break;
            std.mem.swap(u21, &cps[j], &cps[j - 1]);
        }
    }
}

fn composePair(a: u21, b: u21) ?u21 {
    if (a >= l_base and a < l_base + l_count and b >= v_base and b < v_base + v_count)
        return s_base + ((a - l_base) * v_count + (b - v_base)) * t_count;
    if (a >= s_base and a < s_base + s_count and (a - s_base) % t_count == 0 and b > t_base and b < t_base + t_count)
        return a + (b - t_base);
    const i = std.sort.binarySearch(t.Pair, &t.comp, [2]u21{ a, b }, struct {
        fn f(key: [2]u21, e: t.Pair) std.math.Order {
            return switch (std.math.order(key[0], e.a)) {
                .eq => std.math.order(key[1], e.b),
                else => |o| o,
            };
        }
    }.f) orelse return null;
    return t.comp[i].c;
}

/// Canonical composition in place; returns the new length. A mark joins the last
/// starter unless something between them has class 0 or a class no lower than its own.
fn compose(cps: []u21) usize {
    if (cps.len == 0) return 0;
    var starter: usize = 0;
    var last: u16 = if (combiningClass(cps[0]) == 0) 0 else 256;
    var w: usize = 1;
    for (cps[1..]) |cp| {
        const c = combiningClass(cp);
        if (last < c or last == 0) {
            if (composePair(cps[starter], cp)) |comp| {
                cps[starter] = comp;
                continue;
            }
        }
        if (c == 0) starter = w;
        last = c;
        cps[w] = cp;
        w += 1;
    }
    return w;
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;

test "NFC matches python's unicodedata on every canonical decomposition" {
    const gpa = testing.allocator;
    const Fixture = struct { unicode_version: []const u8, cases: []const [2][]const u8 };
    const parsed = try std.json.parseFromSlice(Fixture, gpa, @embedFile("assets/nfc_fixtures.json"), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(t.unicode_version, parsed.value.unicode_version);
    for (parsed.value.cases) |c| {
        const got = try normalize(gpa, c[0]);
        defer if (got) |g| gpa.free(g);
        errdefer std.debug.print("NFC of {any}: want {any}, got {any}\n", .{ c[0], c[1], got orelse c[0] });
        try testing.expectEqualStrings(c[1], got orelse c[0]);
    }
}

test "NFC leaves ASCII and invalid UTF-8 alone" {
    try testing.expectEqual(@as(?[]u8, null), try normalize(testing.allocator, "plain ascii"));
    try testing.expectEqual(@as(?[]u8, null), try normalize(testing.allocator, "\xff\xfe broken"));
}
