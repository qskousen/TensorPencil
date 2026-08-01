//! CLIP's BPE tokenizer — the SD family's prompt front end, a pure-Zig port of
//! `transformers.CLIPTokenizer` (which is OpenAI CLIP's `SimpleTokenizer`).
//!
//! ## Why this is not a `Pretok` variant of `tokenizer.zig`
//!
//! That file is byte-level BPE: a word's initial symbols are its raw bytes, and
//! merges are looked up by `(left_id, right_id)`. CLIP starts differently — the
//! **last character of every word carries a `</w>` suffix**, so the initial symbol
//! sequence is `c₀, c₁, …, cₙ</w>` and the first lookup is by *text*, not by byte.
//! Bending the general BPE around that would complicate the path five other models
//! use. What is genuinely shared — the GPT-2 byte↔codepoint mapping and the Unicode
//! letter/number predicates — is small and re-derived here from `unicode_tables`.
//!
//! ## The pipeline, and the three places a from-scratch port goes wrong
//!
//! ```
//! text -> collapse whitespace -> lowercase -> regex pretokenize
//!      -> byte-encode each word -> append </w> to its last char -> BPE by rank
//!      -> [BOS] ids… [EOS] padded to context with EOS
//! ```
//!
//! 1. **Whitespace is not a token.** CLIP's regex has no whitespace alternative, so
//!    runs of it vanish entirely rather than attaching to the next word — the
//!    opposite of the GPT-2/Qwen convention where a leading space is part of the
//!    token. A port that keeps the space produces valid-looking ids for a different
//!    prompt.
//! 2. **Digits are one token each**, and letters/punctuation come in runs.
//! 3. **Padding is EOS, not a dedicated pad id**, and truncation keeps BOS while
//!    overwriting the final slot with EOS — so a truncated prompt still terminates.
//!    (`clip_text.pooled` finds the *first* EOS for exactly this reason.)

const std = @import("std");
const tables = @import("unicode_tables.zig");

const vocab_json = @embedFile("assets/clip_tokenizer/vocab.json");
const merges_txt = @embedFile("assets/clip_tokenizer/merges.txt");

pub const bos_id: u32 = 49406;
pub const eos_id: u32 = 49407;
/// SD1.5 and SDXL both condition on exactly this many token slots; the positional
/// embedding table is this long, so it is a property of the weights.
pub const context_length: usize = 77;

/// Binary search over the generated Unicode ranges — the same shape
/// `tokenizer.zig` uses (its copy is private, and duplicating fifteen lines beats
/// making a hot predicate public across modules).
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

fn isSpace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
}

/// Bytes byte-level BPE keeps as their own codepoint; everything else is shifted
/// into the 0x100+ range. Identical to GPT-2's table (and `tokenizer.zig`'s, in the
/// opposite direction).
inline fn byteKeptAsIs(b: u16) bool {
    return (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
}

const byte_to_cp: [256]u21 = blk: {
    var map: [256]u21 = undefined;
    var shifted: u21 = 0;
    for (0..256) |b| {
        if (byteKeptAsIs(@intCast(b))) {
            map[b] = @intCast(b);
        } else {
            map[b] = 0x100 + shifted;
            shifted += 1;
        }
    }
    break :blk map;
};

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// Byte-encoded token text -> id.
    vocab: std.StringHashMapUnmanaged(u32),
    /// `left ++ '\x00' ++ right` -> merge rank (lower applies first). Token texts
    /// contain no NUL, so the key is unambiguous.
    ranks: std.StringHashMapUnmanaged(u32),

    pub fn init(gpa: std.mem.Allocator) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        {
            const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, vocab_json, .{});
            const obj = parsed.object;
            try vocab.ensureTotalCapacity(alloc, @intCast(obj.count()));
            var it = obj.iterator();
            while (it.next()) |e| {
                const id: u32 = switch (e.value_ptr.*) {
                    .integer => |i| @intCast(i),
                    else => return error.InvalidVocab,
                };
                vocab.putAssumeCapacity(e.key_ptr.*, id);
            }
        }

        var ranks: std.StringHashMapUnmanaged(u32) = .empty;
        {
            var rank: u32 = 0;
            var lines = std.mem.splitScalar(u8, merges_txt, '\n');
            while (lines.next()) |raw| {
                const line = std.mem.trimEnd(u8, raw, "\r");
                if (line.len == 0 or line[0] == '#') continue; // "#version: 0.2"
                const sp = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
                const key = try std.fmt.allocPrint(alloc, "{s}\x00{s}", .{ line[0..sp], line[sp + 1 ..] });
                try ranks.put(alloc, key, rank);
                rank += 1;
            }
        }

        return .{ .arena = arena, .vocab = vocab, .ranks = ranks };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode `text` into exactly `context_length` ids: `[BOS] … [EOS]`, padded with
    /// EOS. Truncation keeps BOS and forces the last slot to EOS. Caller frees.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try gpa.alloc(u32, context_length);
        errdefer gpa.free(out);
        @memset(out, eos_id);
        out[0] = bos_id;

        var n: usize = 1;
        var words = WordIterator{ .text = text };
        while (try words.next(gpa)) |word| {
            defer gpa.free(word);
            // One slot is reserved for the terminating EOS, so stop at len - 1.
            if (n >= context_length - 1) break;
            var pieces = try self.bpe(gpa, word);
            defer {
                for (pieces.items) |p| gpa.free(p);
                pieces.deinit(gpa);
            }
            for (pieces.items) |piece| {
                if (n >= context_length - 1) break;
                out[n] = self.vocab.get(piece) orelse eos_id;
                n += 1;
            }
        }
        out[n] = eos_id; // explicit terminator; the rest is already EOS padding
        return out;
    }

    /// BPE over one byte-encoded word, returning its pieces in order. The pieces
    /// borrow from an arena internal to the call, so they are duped into the returned
    /// list's own storage.
    fn bpe(self: *const Tokenizer, gpa: std.mem.Allocator, word: []const u8) !std.ArrayList([]const u8) {
        var out: std.ArrayList([]const u8) = .empty;
        errdefer out.deinit(gpa);
        if (word.len == 0) return out;

        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const alloc = scratch.allocator();

        // Initial symbols: one per *codepoint*, with `</w>` glued to the last.
        var syms: std.ArrayList([]const u8) = .empty;
        {
            var i: usize = 0;
            while (i < word.len) {
                const len = std.unicode.utf8ByteSequenceLength(word[i]) catch 1;
                const end = @min(i + len, word.len);
                try syms.append(alloc, word[i..end]);
                i = end;
            }
            const last = syms.items[syms.items.len - 1];
            syms.items[syms.items.len - 1] = try std.fmt.allocPrint(alloc, "{s}</w>", .{last});
        }

        // Repeatedly merge the lowest-ranked adjacent pair. O(len²) in the worst
        // case, on words that are a handful of characters long.
        var key_buf: std.ArrayList(u8) = .empty;
        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_at: ?usize = null;
            for (0..syms.items.len - 1) |i| {
                key_buf.clearRetainingCapacity();
                try key_buf.appendSlice(alloc, syms.items[i]);
                try key_buf.append(alloc, 0);
                try key_buf.appendSlice(alloc, syms.items[i + 1]);
                if (self.ranks.get(key_buf.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_at = i;
                    }
                }
            }
            const at = best_at orelse break;
            const merged = try std.fmt.allocPrint(alloc, "{s}{s}", .{ syms.items[at], syms.items[at + 1] });
            syms.items[at] = merged;
            _ = syms.orderedRemove(at + 1);
        }

        // Duped out of the scratch arena, which dies with this call.
        for (syms.items) |s| try out.append(gpa, try gpa.dupe(u8, s));
        return out;
    }
};

/// Splits text into CLIP's pretokens, byte-encoded and ready for BPE. Whitespace is
/// skipped rather than emitted (see the module header), and matching is on the
/// lowercased codepoint.
const WordIterator = struct {
    text: []const u8,
    pos: usize = 0,

    /// Next word as a byte-encoded string, or null at the end. Caller frees.
    fn next(self: *WordIterator, gpa: std.mem.Allocator) !?[]const u8 {
        // Decode lazily: CLIP lowercases before matching, so both the classification
        // and the bytes handed to BPE come from the lowered codepoint.
        while (self.pos < self.text.len) {
            const first = try self.decode(self.pos);
            if (!isSpace(first.cp)) break;
            self.pos += first.len;
        }
        if (self.pos >= self.text.len) return null;

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);

        const start = try self.decode(self.pos);
        const c0 = lowerCp(start.cp);

        if (c0 == '\'' and self.pos + 1 < self.text.len) {
            // Contractions: 's 't 're 've 'm 'll 'd, ASCII case-insensitive.
            const rest = self.text[self.pos + 1 ..];
            const two = [_][]const u8{ "re", "ve", "ll" };
            for (two) |t| {
                if (rest.len >= 2 and std.ascii.eqlIgnoreCase(rest[0..2], t)) {
                    try self.emit(gpa, &out, self.pos, self.pos + 3);
                    self.pos += 3;
                    return try out.toOwnedSlice(gpa);
                }
            }
            const one = "stmd";
            if (rest.len >= 1 and std.mem.indexOfScalar(u8, one, std.ascii.toLower(rest[0])) != null) {
                try self.emit(gpa, &out, self.pos, self.pos + 2);
                self.pos += 2;
                return try out.toOwnedSlice(gpa);
            }
        }

        if (isLetter(c0)) {
            var end = self.pos + start.len;
            while (end < self.text.len) {
                const c = try self.decode(end);
                if (!isLetter(lowerCp(c.cp))) break;
                end += c.len;
            }
            try self.emit(gpa, &out, self.pos, end);
            self.pos = end;
            return try out.toOwnedSlice(gpa);
        }

        if (isNumber(c0)) {
            // One digit per token — not a run.
            try self.emit(gpa, &out, self.pos, self.pos + start.len);
            self.pos += start.len;
            return try out.toOwnedSlice(gpa);
        }

        // Everything else: a run of non-space, non-letter, non-digit.
        var end = self.pos + start.len;
        while (end < self.text.len) {
            const c = try self.decode(end);
            const lc = lowerCp(c.cp);
            if (isSpace(lc) or isLetter(lc) or isNumber(lc)) break;
            end += c.len;
        }
        try self.emit(gpa, &out, self.pos, end);
        self.pos = end;
        return try out.toOwnedSlice(gpa);
    }

    const Decoded = struct { cp: u21, len: usize };

    fn decode(self: *const WordIterator, at: usize) !Decoded {
        const len = std.unicode.utf8ByteSequenceLength(self.text[at]) catch return .{ .cp = self.text[at], .len = 1 };
        if (at + len > self.text.len) return .{ .cp = self.text[at], .len = 1 };
        const cp = std.unicode.utf8Decode(self.text[at..][0..len]) catch return .{ .cp = self.text[at], .len = 1 };
        return .{ .cp = cp, .len = len };
    }

    /// Lowercase, byte-encode, and append `self.text[from..to]` to `out`.
    fn emit(self: *const WordIterator, gpa: std.mem.Allocator, out: *std.ArrayList(u8), from: usize, to: usize) !void {
        var i = from;
        var buf: [4]u8 = undefined;
        while (i < to) {
            const d = try self.decode(i);
            const lc = lowerCp(d.cp);
            const n = std.unicode.utf8Encode(lc, &buf) catch blk: {
                buf[0] = @truncate(d.cp);
                break :blk 1;
            };
            // Byte-level encoding: each UTF-8 byte becomes its mapped codepoint,
            // which is then itself UTF-8 encoded — the form the vocab keys are in.
            for (buf[0..n]) |b| {
                var enc: [4]u8 = undefined;
                const m = std.unicode.utf8Encode(byte_to_cp[b], &enc) catch unreachable;
                try out.appendSlice(gpa, enc[0..m]);
            }
            i += d.len;
        }
    }
};

/// ASCII plus the Latin-1 range, which is all CLIP's `text.lower()` reaches for the
/// prompts SD models see. A full Unicode case fold would need the tables' own
/// mapping; this is deliberately narrow and documented rather than silently partial.
fn lowerCp(cp: u21) u21 {
    if (cp >= 'A' and cp <= 'Z') return cp + 32;
    if (cp >= 0xC0 and cp <= 0xDE and cp != 0xD7) return cp + 32;
    return cp;
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;

const TokCase = struct { text: []const u8, ids: []const u32 };
const Fixtures = struct { clip_tokenizer: []const TokCase };

test "CLIP tokenization matches transformers.CLIPTokenizer" {
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        Fixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for (parsed.value.clip_tokenizer) |c| {
        const got = try tok.encode(gpa, c.text);
        defer gpa.free(got);
        try testing.expectEqual(context_length, c.ids.len);
        for (c.ids, got, 0..) |e, a, i| {
            errdefer std.debug.print("'{s}' slot {d}: expected {d} got {d}\n  full: {any}\n", .{ c.text, i, e, a, got });
            try testing.expectEqual(e, a);
        }
    }
}

test "the encoding always terminates, even when the prompt overruns the window" {
    // Truncation must keep BOS and force EOS into the last slot: a prompt that runs
    // past 77 tokens otherwise ends mid-word, and `clip_text.pooled` (first EOS)
    // would find nothing.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(gpa);
    for (0..200) |_| try long.appendSlice(gpa, "detailed ");

    const ids = try tok.encode(gpa, long.items);
    defer gpa.free(ids);
    try testing.expectEqual(context_length, ids.len);
    try testing.expectEqual(bos_id, ids[0]);
    try testing.expectEqual(eos_id, ids[context_length - 1]);
    // Everything between is real content, not padding.
    for (ids[1 .. context_length - 1]) |id| try testing.expect(id != eos_id);
}

test "whitespace is dropped rather than attached to the next word" {
    // The GPT-2/Qwen convention makes a leading space part of the token; CLIP's regex
    // has no whitespace alternative at all. Both spellings tokenize identically here,
    // which is the property that convention difference amounts to.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const a = try tok.encode(gpa, "a red   cat");
    defer gpa.free(a);
    const b = try tok.encode(gpa, "  a red cat  ");
    defer gpa.free(b);
    try testing.expectEqualSlices(u32, a, b);
}
