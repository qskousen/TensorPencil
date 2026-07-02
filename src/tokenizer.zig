//! Qwen2 byte-level BPE tokenizer (pure Zig port of transformers'
//! `Qwen2Tokenizer`, which ComfyUI uses for the Krea 2 text encoder).
//!
//! Pipeline: split out special tokens -> pretokenize (hand-rolled scanner for
//! the fixed Qwen2 regex, using generated Unicode tables) -> byte-level BPE
//! over vocab.json/merges.txt. The vocab and merges are embedded in the
//! binary from src/assets/qwen_tokenizer/.

const std = @import("std");
const tables = @import("unicode_tables.zig");

const vocab_json = @embedFile("assets/qwen_tokenizer/vocab.json");
const merges_txt = @embedFile("assets/qwen_tokenizer/merges.txt");

pub const pad_token: u32 = 151643; // <|endoftext|>
pub const im_start: u32 = 151644;
pub const im_end: u32 = 151645;
pub const think_open: u32 = 151667;
pub const think_close: u32 = 151668;

/// Added special tokens (from tokenizer_config.json); matched verbatim
/// before pretokenization.
const special_tokens = [_]struct { text: []const u8, id: u32 }{
    .{ .text = "<|endoftext|>", .id = 151643 },
    .{ .text = "<|im_start|>", .id = 151644 },
    .{ .text = "<|im_end|>", .id = 151645 },
    .{ .text = "<|object_ref_start|>", .id = 151646 },
    .{ .text = "<|object_ref_end|>", .id = 151647 },
    .{ .text = "<|box_start|>", .id = 151648 },
    .{ .text = "<|box_end|>", .id = 151649 },
    .{ .text = "<|quad_start|>", .id = 151650 },
    .{ .text = "<|quad_end|>", .id = 151651 },
    .{ .text = "<|vision_start|>", .id = 151652 },
    .{ .text = "<|vision_end|>", .id = 151653 },
    .{ .text = "<|vision_pad|>", .id = 151654 },
    .{ .text = "<|image_pad|>", .id = 151655 },
    .{ .text = "<|video_pad|>", .id = 151656 },
    .{ .text = "<tool_call>", .id = 151657 },
    .{ .text = "</tool_call>", .id = 151658 },
    .{ .text = "<|fim_prefix|>", .id = 151659 },
    .{ .text = "<|fim_middle|>", .id = 151660 },
    .{ .text = "<|fim_suffix|>", .id = 151661 },
    .{ .text = "<|fim_pad|>", .id = 151662 },
    .{ .text = "<|repo_name|>", .id = 151663 },
    .{ .text = "<|file_sep|>", .id = 151664 },
    .{ .text = "<tool_response>", .id = 151665 },
    .{ .text = "</tool_response>", .id = 151666 },
    .{ .text = "<think>", .id = 151667 },
    .{ .text = "</think>", .id = 151668 },
};

// --- GPT-2 byte<->unicode mapping ----------------------------------------

/// Bytes that byte-level BPE keeps as their own codepoint.
inline fn byteKeptAsIs(b: u16) bool {
    return (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
}

/// codepoint -> raw byte for vocab/merges keys; codepoints are < 0x100 + 68.
const cp_to_byte: [0x144]u16 = blk: {
    var map: [0x144]u16 = @splat(0xFFFF);
    var shifted: u16 = 0;
    for (0..256) |b| {
        if (byteKeptAsIs(@intCast(b))) {
            map[b] = b;
        } else {
            map[0x100 + shifted] = b;
            shifted += 1;
        }
    }
    break :blk map;
};

/// Decode a byte-level token key (UTF-8) back to the raw bytes it stands for.
/// Returns the decoded length, in place (decoded is never longer than utf8).
fn decodeTokenKey(key: []const u8, out: []u8) !usize {
    var it = std.unicode.Utf8Iterator{ .bytes = key, .i = 0 };
    var n: usize = 0;
    while (it.nextCodepoint()) |cp| {
        if (cp >= cp_to_byte.len or cp_to_byte[cp] == 0xFFFF) return error.InvalidTokenKey;
        if (n >= out.len) return error.TokenKeyTooLong;
        out[n] = @intCast(cp_to_byte[cp]);
        n += 1;
    }
    return n;
}

// --- Unicode classes -------------------------------------------------------

fn inRanges(comptime ranges: []const tables.Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp > ranges[mid].hi) {
            lo = mid + 1;
        } else if (cp < ranges[mid].lo) {
            hi = mid;
        } else {
            return true;
        }
    }
    return false;
}

fn isLetter(cp: u21) bool {
    return inRanges(&tables.letter_ranges, cp);
}
fn isNumber(cp: u21) bool {
    return inRanges(&tables.number_ranges, cp);
}
fn isWs(cp: u21) bool {
    return inRanges(&tables.whitespace_ranges, cp);
}

// --- Tokenizer -------------------------------------------------------------

const Merge = struct { rank: u32, merged: u32 };

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// id -> raw bytes (regular vocab only, not special tokens).
    id_to_bytes: [][]const u8,
    /// single byte -> vocab id.
    byte_id: [256]u32,
    /// (left_id << 32 | right_id) -> merge.
    merges: std.AutoHashMapUnmanaged(u64, Merge),

    pub fn init(gpa: std.mem.Allocator) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // Vocab: token key (byte-level unicode string) -> id.
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, vocab_json, .{});
        if (parsed != .object) return error.InvalidVocab;
        const vocab_obj = parsed.object;

        const id_to_bytes = try alloc.alloc([]const u8, vocab_obj.count());
        // raw bytes -> id, for resolving merge targets.
        var raw_to_id: std.StringHashMapUnmanaged(u32) = .empty;
        try raw_to_id.ensureTotalCapacity(alloc, @intCast(vocab_obj.count()));

        var byte_id: [256]u32 = @splat(0xFFFF_FFFF);
        var it = vocab_obj.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* != .integer) return error.InvalidVocab;
            const id: u32 = @intCast(entry.value_ptr.integer);
            if (id >= id_to_bytes.len) return error.InvalidVocab;
            const key = entry.key_ptr.*;
            const raw = try alloc.alloc(u8, key.len);
            const n = try decodeTokenKey(key, raw);
            id_to_bytes[id] = raw[0..n];
            raw_to_id.putAssumeCapacity(raw[0..n], id);
            if (n == 1) byte_id[raw[0]] = id;
        }
        for (byte_id) |id| if (id == 0xFFFF_FFFF) return error.IncompleteByteVocab;

        // Merges.
        var merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty;
        var rank: u32 = 0;
        var lines = std.mem.splitScalar(u8, merges_txt, '\n');
        var first = true;
        // Longest key in this vocab is 128 UTF-8 bytes; decoded is never longer.
        var pair_buf: [256]u8 = undefined; // left ++ right, concatenated
        var right_buf: [128]u8 = undefined;
        while (lines.next()) |line_raw| {
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            if (line.len == 0) continue;
            if (first and std.mem.startsWith(u8, line, "#version")) {
                first = false;
                continue;
            }
            first = false;
            const space = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidMerges;
            const left_n = try decodeTokenKey(line[0..space], pair_buf[0..128]);
            const right_n = try decodeTokenKey(line[space + 1 ..], &right_buf);
            const left = raw_to_id.get(pair_buf[0..left_n]) orelse return error.InvalidMerges;
            const right = raw_to_id.get(right_buf[0..right_n]) orelse return error.InvalidMerges;
            @memcpy(pair_buf[left_n .. left_n + right_n], right_buf[0..right_n]);
            const merged = raw_to_id.get(pair_buf[0 .. left_n + right_n]) orelse return error.InvalidMerges;
            try merges.put(alloc, pairKey(left, right), .{ .rank = rank, .merged = merged });
            rank += 1;
        }

        return .{ .arena = arena, .id_to_bytes = id_to_bytes, .byte_id = byte_id, .merges = merges };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    inline fn pairKey(left: u32, right: u32) u64 {
        return (@as(u64, left) << 32) | right;
    }

    /// Append the token ids of `text` to `out`.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < text.len) {
            // Earliest special-token occurrence from i.
            var next_special: ?struct { pos: usize, idx: usize } = null;
            for (special_tokens, 0..) |st, idx| {
                if (std.mem.indexOfPos(u8, text, i, st.text)) |pos| {
                    if (next_special == null or pos < next_special.?.pos or
                        (pos == next_special.?.pos and st.text.len > special_tokens[next_special.?.idx].text.len))
                    {
                        next_special = .{ .pos = pos, .idx = idx };
                    }
                }
            }
            const seg_end = if (next_special) |ns| ns.pos else text.len;
            if (i < seg_end) try self.encodeSegment(gpa, text[i..seg_end], out);
            if (next_special) |ns| {
                try out.append(gpa, special_tokens[ns.idx].id);
                i = ns.pos + special_tokens[ns.idx].text.len;
            } else {
                i = text.len;
            }
        }
    }

    /// Encode text containing no special tokens.
    fn encodeSegment(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        // Decode to codepoints once; keep byte offsets for slicing.
        const n_cps = try std.unicode.utf8CountCodepoints(text);
        const cps = try gpa.alloc(u21, n_cps);
        defer gpa.free(cps);
        const offs = try gpa.alloc(usize, n_cps + 1);
        defer gpa.free(offs);
        {
            var it = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
            var n: usize = 0;
            while (true) {
                offs[n] = it.i;
                const cp = it.nextCodepoint() orelse break;
                cps[n] = cp;
                n += 1;
            }
        }

        var symbols: std.ArrayList(u32) = .empty;
        defer symbols.deinit(gpa);

        var i: usize = 0;
        while (i < n_cps) {
            const end = pretokenEnd(cps, i);
            try self.bpe(gpa, text[offs[i]..offs[end]], &symbols, out);
            i = end;
        }
    }

    /// End index (exclusive, in codepoints) of the pretoken starting at `i`.
    /// Hand-rolled match of the Qwen2 pretokenizer regex alternation:
    ///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
    ///   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    fn pretokenEnd(cps: []const u21, i: usize) usize {
        const n = cps.len;
        const c0 = cps[i];

        // 1: contractions, ASCII case-insensitive.
        if (c0 == '\'' and i + 1 < n) {
            const c1 = asciiLower(cps[i + 1]);
            if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return i + 2;
            if (i + 2 < n) {
                const c2 = asciiLower(cps[i + 2]);
                if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return i + 3;
            }
        }

        // 2: optional non-letter/number/CRLF prefix + letters.
        if (isLetter(c0)) {
            var j = i + 1;
            while (j < n and isLetter(cps[j])) j += 1;
            return j;
        }
        if (c0 != '\r' and c0 != '\n' and !isNumber(c0) and i + 1 < n and isLetter(cps[i + 1])) {
            var j = i + 2;
            while (j < n and isLetter(cps[j])) j += 1;
            return j;
        }

        // 3: single number char.
        if (isNumber(c0)) return i + 1;

        // 4: optional space + punctuation run + trailing newlines.
        {
            var j = i;
            if (cps[j] == ' ') j += 1;
            var k = j;
            while (k < n and !isWs(cps[k]) and !isLetter(cps[k]) and !isNumber(cps[k])) k += 1;
            if (k > j) {
                while (k < n and (cps[k] == '\r' or cps[k] == '\n')) k += 1;
                return k;
            }
        }

        // 5-7: whitespace runs.
        if (isWs(c0)) {
            var e = i;
            var last_rn: ?usize = null;
            while (e < n and isWs(cps[e])) : (e += 1) {
                if (cps[e] == '\r' or cps[e] == '\n') last_rn = e;
            }
            if (last_rn) |lr| return lr + 1; // \s*[\r\n]+
            if (e == n) return e; // \s+(?!\S) at end of text
            if (e - i >= 2) return e - 1; // \s+(?!\S) backs off one
            return e; // \s+
        }

        // Unreachable for the regex above, but never loop forever.
        return i + 1;
    }

    fn asciiLower(cp: u21) u21 {
        return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
    }

    /// Byte-level BPE of one pretoken; appends ids to `out`.
    fn bpe(self: *const Tokenizer, gpa: std.mem.Allocator, bytes: []const u8, symbols: *std.ArrayList(u32), out: *std.ArrayList(u32)) !void {
        symbols.clearRetainingCapacity();
        try symbols.ensureUnusedCapacity(gpa, bytes.len);
        for (bytes) |b| symbols.appendAssumeCapacity(self.byte_id[b]);

        while (symbols.items.len >= 2) {
            // Lowest-rank adjacent pair.
            var best_rank: u32 = std.math.maxInt(u32);
            var best_at: usize = 0;
            var best_merged: u32 = 0;
            for (0..symbols.items.len - 1) |j| {
                if (self.merges.get(pairKey(symbols.items[j], symbols.items[j + 1]))) |m| {
                    if (m.rank < best_rank) {
                        best_rank = m.rank;
                        best_at = j;
                        best_merged = m.merged;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;
            // Merge every occurrence of that pair, left to right.
            const left = symbols.items[best_at];
            const right = symbols.items[best_at + 1];
            var read: usize = 0;
            var write: usize = 0;
            while (read < symbols.items.len) {
                if (read + 1 < symbols.items.len and symbols.items[read] == left and symbols.items[read + 1] == right) {
                    symbols.items[write] = best_merged;
                    read += 2;
                } else {
                    symbols.items[write] = symbols.items[read];
                    read += 1;
                }
                write += 1;
            }
            symbols.shrinkRetainingCapacity(write);
        }
        try out.appendSlice(gpa, symbols.items);
    }

    /// Decode ids back to text (debugging / tests).
    pub fn decodeAlloc(self: *const Tokenizer, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        outer: for (ids) |id| {
            if (id < self.id_to_bytes.len) {
                try buf.appendSlice(gpa, self.id_to_bytes[id]);
            } else {
                for (special_tokens) |st| {
                    if (st.id == id) {
                        try buf.appendSlice(gpa, st.text);
                        continue :outer;
                    }
                }
                return error.UnknownToken;
            }
        }
        return buf.toOwnedSlice(gpa);
    }
};

// --- tests -----------------------------------------------------------------

fn expectEncode(tok: *const Tokenizer, text: []const u8, expected: []const u32) !void {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try tok.encode(gpa, text, &out);
    try std.testing.expectEqualSlices(u32, expected, out.items);
}

// Reference ids from transformers' Qwen2Tokenizer (slow) on the same
// vocab/merges — see the fixture dump in tools/gen_op_fixtures.py history.
test "tokenizer matches transformers Qwen2Tokenizer" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    try expectEncode(&tok, "a photo of a cat", &.{ 64, 6548, 315, 264, 8251 });
    try expectEncode(&tok, "Hello, World! 123 café ☕ 你好", &.{ 9707, 11, 4337, 0, 220, 16, 17, 18, 51950, 25125, 243, 220, 108386 });
    try expectEncode(&tok, "don't we'll I'M IT'S", &.{ 15007, 944, 582, 3278, 358, 27603, 8700, 13272 });
    try expectEncode(&tok, "a  b\n\nc\t d ", &.{ 64, 220, 293, 271, 66, 197, 294, 220 });
    try expectEncode(&tok, "12345", &.{ 16, 17, 18, 19, 20 });
    try expectEncode(&tok, "  leading and trailing  ", &.{ 220, 6388, 323, 27748, 256 });
    try expectEncode(&tok, "", &.{});
    try expectEncode(&tok, "ééé 😀😀", &.{ 963, 963, 963, 90316, 141334 });
    try expectEncode(
        &tok,
        "<|im_start|>system\nDescribe the image:<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n",
        &.{ 151644, 8948, 198, 74785, 279, 2168, 25, 151645, 198, 151644, 872, 198, 64, 8251, 151645, 198, 151644, 77091, 198 },
    );
}

test "decode round trips" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const text = "<|im_start|>Hello, café world 123!<|im_end|>";
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try tok.encode(gpa, text, &out);
    const round = try tok.decodeAlloc(gpa, out.items);
    defer gpa.free(round);
    try std.testing.expectEqualStrings(text, round);
}
