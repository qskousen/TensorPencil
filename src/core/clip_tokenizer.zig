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
const prompt_a1111 = @import("prompt_a1111.zig");
const prompt_weights = @import("prompt_weights.zig");

const vocab_json = @embedFile("assets/clip_tokenizer/vocab.json");
const merges_txt = @embedFile("assets/clip_tokenizer/merges.txt");

pub const bos_id: u32 = 49406;
pub const eos_id: u32 = 49407;
/// SD1.5 and SDXL both condition on exactly this many token slots; the positional
/// embedding table is this long, so it is a property of the weights.
pub const context_length: usize = 77;

/// A tokenized *segment* at least this long is split across a chunk boundary rather
/// than pushed whole into the next chunk (ComfyUI's `SDTokenizer.max_word_length`).
/// Short segments are kept intact so a chunk boundary never lands mid-word.
pub const max_word_length: usize = 8;

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
        return self.encodePadded(gpa, text, eos_id);
    }

    /// `encode`, with the padding id spelled out.
    ///
    /// ⚠️ **SDXL's two towers pad differently and both paddings are conditioning.**
    /// CLIP-L pads with EOS (`encode`); CLIP-G pads with **0** (`"!"`), which is what
    /// OpenCLIP trained with and what both ComfyUI and diffusers use. The padded slots
    /// are part of the 77-token window the UNet cross-attends to — a causal tower gives
    /// them different hidden states — so this is not cosmetic, and the two towers must be
    /// tokenized twice rather than once and shared.
    pub fn encodePadded(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, pad_id: u32) ![]u32 {
        const ids = try self.contentIds(gpa, text);
        defer gpa.free(ids);

        const out = try gpa.alloc(u32, context_length);
        errdefer gpa.free(out);
        @memset(out, pad_id);
        out[0] = bos_id;

        // One slot is reserved for the terminating EOS, so at most len - 2 content ids.
        const n = @min(ids.len, context_length - 2);
        @memcpy(out[1 .. 1 + n], ids[0..n]);
        // The terminator is always EOS even when the padding is not — it is what
        // `clip_text.pooled` looks for, and with pad 0 there is exactly one of them.
        out[1 + n] = eos_id;
        return out;
    }

    /// The bare content ids of `text` — no BOS, no EOS, no padding and no
    /// truncation. This is `transformers`' `tokenizer(text)["input_ids"][1:-1]`,
    /// which is the unit both `encodePadded` and `encodeWeighted` are built from.
    /// Caller frees.
    pub fn contentIds(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var words = WordIterator{ .text = text };
        while (try words.next(gpa)) |word| {
            defer gpa.free(word);
            var pieces = try self.bpe(gpa, word);
            defer {
                for (pieces.items) |p| gpa.free(p);
                pieces.deinit(gpa);
            }
            // An unknown piece cannot happen with a well-formed vocab; CLIP's unk
            // token is `<|endoftext|>`, so falling back to it matches transformers.
            for (pieces.items) |piece| try out.append(gpa, self.vocab.get(piece) orelse eos_id);
        }
        return out.toOwnedSlice(gpa);
    }

    /// Chunk already-parsed A1111 weighted parts into whole `context_length` windows —
    /// the `tokenize_line` half of A1111's prompt path, where `prompt_a1111.parseAttention`
    /// is the other half.
    ///
    /// Two things differ from `encodeWeighted`'s ComfyUI chunker, and both are visible in
    /// a long prompt:
    ///
    /// - ⚠️ **A `BREAK` part closes the current chunk immediately**, padding the rest. That
    ///   is what `BREAK` is *for*, and ComfyUI ignores it entirely (tokenizing the literal
    ///   word `break`).
    /// - ⚠️ **A boundary backtracks to the last comma.** On reaching 75 content tokens with
    ///   a comma no more than `comma_backtrack` tokens behind, everything after that comma
    ///   is *relocated* into the next chunk, so a boundary does not land mid-phrase. Since
    ///   each chunk is encoded independently by a causal tower, a phrase split across the
    ///   seam is conditioning the model never sees whole.
    ///
    /// Note the reference sets `last_comma` to the index the comma is *about* to occupy,
    /// before appending it, and resets it whenever a chunk closes — including on the
    /// relocation path, where the relocated tokens land in a chunk with no known comma.
    ///
    /// `parts` come from `prompt_a1111.parseAttention`. Caller `deinit`s the result.
    pub fn encodeParts(
        self: *const Tokenizer,
        gpa: std.mem.Allocator,
        parts: []const prompt_a1111.Part,
        pad_id: u32,
    ) !Prompt {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();

        // `,</w>` — the comma as its own token, which is what a boundary looks for.
        const comma_id: ?u32 = self.vocab.get(",</w>");

        var out: std.ArrayList(Weighted) = .empty;
        errdefer out.deinit(gpa);
        // The current chunk's CONTENT only; BOS/EOS are added when it closes.
        var cur: std.ArrayList(Weighted) = .empty;
        var last_comma: ?usize = null;

        const Closer = struct {
            fn go(g: std.mem.Allocator, o: *std.ArrayList(Weighted), c: *std.ArrayList(Weighted), lc: *?usize, pad: u32) !void {
                try o.append(g, .{ .id = bos_id, .weight = 1.0 });
                try o.appendSlice(g, c.items);
                // EOS terminates the content; the rest is padding. Both carry weight 1.0,
                // so a weighted prompt never scales its own delimiters.
                try o.append(g, .{ .id = eos_id, .weight = 1.0 });
                for (c.items.len..chunk_content) |_| try o.append(g, .{ .id = pad, .weight = 1.0 });
                c.clearRetainingCapacity();
                lc.* = null;
            }
        };

        for (parts) |part| {
            if (part.isBreak()) {
                try Closer.go(gpa, &out, &cur, &last_comma, pad_id);
                continue;
            }
            const ids = try self.contentIds(arena, part.text);
            for (ids) |id| {
                if (comma_id != null and id == comma_id.?) {
                    last_comma = cur.items.len;
                } else if (comma_backtrack != 0 and cur.items.len == chunk_content) {
                    if (last_comma) |lc| if (cur.items.len - lc <= comma_backtrack) {
                        const at = lc + 1;
                        const reloc = try arena.dupe(Weighted, cur.items[at..]);
                        cur.shrinkRetainingCapacity(at);
                        try Closer.go(gpa, &out, &cur, &last_comma, pad_id);
                        try cur.appendSlice(gpa, reloc);
                    };
                }
                if (cur.items.len == chunk_content) try Closer.go(gpa, &out, &cur, &last_comma, pad_id);
                try cur.append(gpa, .{ .id = id, .weight = part.weight });
            }
        }
        // A trailing partial chunk, and the empty-prompt case (which still needs one).
        if (cur.items.len > 0 or out.items.len == 0) {
            try Closer.go(gpa, &out, &cur, &last_comma, pad_id);
        }
        cur.deinit(gpa);

        std.debug.assert(out.items.len % context_length == 0);
        const tokens = try out.toOwnedSlice(gpa);
        return .{ .tokens = tokens, .chunks = tokens.len / context_length };
    }

    /// Tokenize a prompt the way ComfyUI does: parse the `(text:weight)` emphasis
    /// syntax, then pack the result into **as many whole `context_length` chunks as
    /// it takes** rather than truncating at one.
    ///
    /// ⚠️ **This is what `encode`/`encodePadded` get wrong for any real prompt**, and
    /// it is not a subtle difference. A booru-style prompt is routinely 100+ tokens;
    /// truncating at 77 silently drops the tail (typically the entire quality-tag
    /// block), and tokenizing `(shiny skin:1.1)` literally spends nine content slots
    /// on punctuation that ComfyUI strips — so the truncation bites *earlier* than the
    /// prompt's real length suggests. Measured on one real 115-token prompt: ComfyUI
    /// built 154 conditioning rows, `encode` built 77 and lost `lens flare` through
    /// `newest`. The render was a different image, at the same seed.
    ///
    /// The two conventions worth knowing, because neither is derivable:
    ///
    /// - **`BREAK` is not honoured.** A1111 pads to the next chunk on it; ComfyUI has
    ///   no such rule, so it tokenizes as the literal word `break`. Verified against
    ///   `comfy.sd1_clip.SDTokenizer` — the chunk split is purely length-driven.
    /// - **A weight is absolute, not cumulative, once a `:` gives one.** `(a:1.5)`
    ///   inside another paren group is 1.5, not 1.5 × 1.1. Bare nesting *is*
    ///   cumulative (`((a))` is 1.21).
    ///
    /// `pad_id` is the trailing filler for a short final chunk — EOS for CLIP-L, 0 for
    /// CLIP-G, exactly as in `encodePadded`. Caller `deinit`s the result.
    pub fn encodeWeighted(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, pad_id: u32) !Prompt {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();

        var segs: std.ArrayList(prompt_weights.Segment) = .empty;
        try prompt_weights.segments(arena, &segs, text);

        // Each weighted *segment* is tokenized on its own — not each word — because
        // that is the unit ComfyUI hands to the tokenizer, and BPE at a segment
        // boundary is not always what it would be mid-string.
        var groups: std.ArrayList(Group) = .empty;
        for (segs.items) |s| {
            const plain = try prompt_weights.unescape(arena, s.text);
            if (plain.len == 0) continue; // ComfyUI drops empty segments outright.
            try groups.append(arena, .{ .ids = try self.contentIds(arena, plain), .weight = s.weight });
        }

        var out: std.ArrayList(Weighted) = .empty;
        errdefer out.deinit(gpa);

        // `len` tracks the current chunk's fill; a chunk always opens with BOS and
        // always closes with EOS, so `context_length - 1` is the content ceiling.
        try out.append(gpa, .{ .id = bos_id, .weight = 1.0 });
        var len: usize = 1;
        for (groups.items) |g| {
            const is_large = g.ids.len >= max_word_length;
            // The nesting product was accumulated in f64 to match Python; this is the
            // single narrowing to the f32 the multiply is actually done in.
            const w: f32 = @floatCast(g.weight);
            var rest = g.ids;
            while (rest.len > 0) {
                if (rest.len + len <= context_length - 1) {
                    for (rest) |id| try out.append(gpa, .{ .id = id, .weight = w });
                    len += rest.len;
                    break;
                }
                const room = context_length - len - 1;
                if (is_large) {
                    // Split it: a long segment is not a word, so a boundary inside it
                    // costs nothing. ⚠️ EOS goes directly after the content it
                    // terminates — padding follows it, never precedes it.
                    for (rest[0..room]) |id| try out.append(gpa, .{ .id = id, .weight = w });
                    rest = rest[room..];
                    try out.append(gpa, .{ .id = eos_id, .weight = 1.0 });
                } else {
                    // Keep it whole: close this chunk early and retry in the next one.
                    // `rest` is deliberately not advanced.
                    try out.append(gpa, .{ .id = eos_id, .weight = 1.0 });
                    for (0..room) |_| try out.append(gpa, .{ .id = pad_id, .weight = 1.0 });
                }
                try out.append(gpa, .{ .id = bos_id, .weight = 1.0 });
                len = 1;
            }
        }
        try out.append(gpa, .{ .id = eos_id, .weight = 1.0 });
        len += 1;
        while (len < context_length) : (len += 1) try out.append(gpa, .{ .id = pad_id, .weight = 1.0 });

        std.debug.assert(out.items.len % context_length == 0);
        const tokens = try out.toOwnedSlice(gpa);
        return .{ .tokens = tokens, .chunks = tokens.len / context_length };
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

/// One token id plus the attention weight the prompt's emphasis syntax gave it.
pub const Weighted = struct { id: u32, weight: f32 };

/// How far back a chunk boundary may move to land after a comma (A1111's
/// `comma_padding_backtrack`, default 20). Zero disables it.
pub const comma_backtrack: usize = 20;

/// Content tokens per chunk — 75, with BOS and EOS making up `context_length`.
pub const chunk_content: usize = context_length - 2;

/// A prompt tokenized into whole `context_length` chunks. `tokens` is
/// `[chunks][context_length]` row-major, so every chunk is a complete
/// `[BOS] … [EOS] pad…` sequence the tower can be run on directly.
pub const Prompt = struct {
    tokens: []Weighted,
    chunks: usize,

    pub fn deinit(self: *Prompt, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        self.* = undefined;
    }

    /// Chunk `i`'s `context_length` weighted ids.
    pub fn chunk(self: *const Prompt, i: usize) []const Weighted {
        return self.tokens[i * context_length ..][0..context_length];
    }

    /// Chunk `i`'s ids alone, into a caller buffer of exactly `context_length`.
    pub fn idsInto(self: *const Prompt, dst: []u32, i: usize) void {
        std.debug.assert(dst.len == context_length);
        for (self.chunk(i), dst) |t, *d| d.* = t.id;
    }

    /// Total conditioning rows this prompt produces — what `Cond.seq` becomes.
    pub fn seq(self: *const Prompt) usize {
        return self.chunks * context_length;
    }

    /// True when any token's weight is not exactly 1.0. This is ComfyUI's
    /// `has_weights`, and it is what decides whether the empty-prompt reference
    /// forward is needed at all (see `clip_text.applyWeights`) — so an unweighted
    /// prompt costs exactly what it did before.
    pub fn hasWeights(self: *const Prompt) bool {
        for (self.tokens) |t| if (t.weight != 1.0) return true;
        return false;
    }
};

/// ComfyUI's `gen_empty_tokens`: the sequence whose hidden states are the reference
/// point attention weighting interpolates away from — `[BOS] [EOS] pad…`, filled to
/// `dst.len`.
pub fn emptyIds(dst: []u32, pad_id: u32) void {
    std.debug.assert(dst.len >= 2);
    dst[0] = bos_id;
    dst[1] = eos_id;
    @memset(dst[2..], pad_id);
}

/// A run of prompt text sharing one weight, still pointing into the escaped input.
/// That run, tokenized. ComfyUI calls this a "t_group".
const Group = struct { ids: []const u32, weight: f64 };

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

const SdxlCase = struct { text: []const u8, ids_l: []const u32, ids_g: []const u32 };
const SdxlFixtures = struct { sdxl_tokenizer: []const SdxlCase };

test "SDXL's two paddings match ComfyUI's own SDXL tokenizer" {
    // The reference is ComfyUI (`comfy.sdxl_clip.SDXLTokenizer`) rather than
    // transformers, because ComfyUI is the compatibility target and it is what settles
    // the pad-0 convention for the CLIP-G tower. Same cases as the SD1.5 fixture above,
    // deliberately: any difference between the two columns is the padding and nothing
    // else, which is also what this test asserts below.
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        SdxlFixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures_sdxl.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for (parsed.value.sdxl_tokenizer) |c| {
        const l = try tok.encodePadded(gpa, c.text, eos_id);
        defer gpa.free(l);
        const g = try tok.encodePadded(gpa, c.text, 0);
        defer gpa.free(g);
        for (c.ids_l, l, 0..) |e, a, i| {
            errdefer std.debug.print("clip_l '{s}' slot {d}: expected {d} got {d}\n", .{ c.text, i, e, a });
            try testing.expectEqual(e, a);
        }
        for (c.ids_g, g, 0..) |e, a, i| {
            errdefer std.debug.print("clip_g '{s}' slot {d}: expected {d} got {d}\n", .{ c.text, i, e, a });
            try testing.expectEqual(e, a);
        }
        // And the relation between them, which is the whole claim: identical up to and
        // including the terminating EOS, then EOS against 0. A fixture that disagreed
        // here would mean the two towers see different *content*, not different padding.
        var term: usize = 0;
        while (term < context_length and c.ids_g[term] != eos_id) term += 1;
        for (c.ids_l[0 .. term + 1], c.ids_g[0 .. term + 1]) |a, b| try testing.expectEqual(a, b);
        for (c.ids_l[term + 1 ..]) |id| try testing.expectEqual(eos_id, id);
        for (c.ids_g[term + 1 ..]) |id| try testing.expectEqual(@as(u32, 0), id);
    }
}

const ChunkRows = struct { ids: []const u32, weights: []const f32 };
const PromptCase = struct { text: []const u8, clip_l: []const ChunkRows, clip_g: []const ChunkRows };
const PromptFixtures = struct { clip_prompt: []const PromptCase };

test "prompt weights and 77-token chunking match ComfyUI's own tokenizer" {
    // 31 prompts x 2 paddings, from `tools/gen_clip_prompt_fixtures.py`. The cases are
    // adversarial on purpose: the two real prompts that exposed the truncation bug, bare
    // vs absolute nesting, escaped parens, unbalanced parens, an unparseable weight, and
    // both sides of the keep-short-segments-whole rule at a chunk boundary.
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        PromptFixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures_weighted.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for (parsed.value.clip_prompt) |c| {
        for ([_]struct { u32, []const ChunkRows }{
            .{ eos_id, c.clip_l },
            .{ 0, c.clip_g },
        }) |arm| {
            const pad, const want = arm;
            var got = try tok.encodeWeighted(gpa, c.text, pad);
            defer got.deinit(gpa);

            errdefer std.debug.print(
                "'{s}' pad {d}: expected {d} chunks, got {d}\n",
                .{ c.text, pad, want.len, got.chunks },
            );
            try testing.expectEqual(want.len, got.chunks);

            for (want, 0..) |wc, ci| {
                const gc = got.chunk(ci);
                for (wc.ids, wc.weights, gc, 0..) |wid, ww, g, j| {
                    errdefer std.debug.print(
                        "'{s}' pad {d} chunk {d} slot {d}: expected id {d} w {d}, got id {d} w {d}\n",
                        .{ c.text, pad, ci, j, wid, ww, g.id, g.weight },
                    );
                    try testing.expectEqual(wid, g.id);
                    // Exact: the weight is a product of literal 1.1s and a parsed
                    // decimal, and both sides narrow f64 -> f32 at the same point.
                    try testing.expectEqual(ww, g.weight);
                }
            }
        }
    }
}

test "a long prompt keeps every token instead of truncating at one chunk" {
    // The regression this whole path exists for, stated as a property rather than a
    // fixture: `encode` drops the tail of a 100+ token prompt, `encodeWeighted` does not.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(gpa);
    for (0..100) |_| try long.appendSlice(gpa, "cat ");
    try long.appendSlice(gpa, "unicorn");

    const truncated = try tok.encode(gpa, long.items);
    defer gpa.free(truncated);
    var p = try tok.encodeWeighted(gpa, long.items, eos_id);
    defer p.deinit(gpa);

    const unicorn = tok.vocab.get("unicorn</w>").?;
    try testing.expect(std.mem.indexOfScalar(u32, truncated, unicorn) == null);
    var found = false;
    for (p.tokens) |t| if (t.id == unicorn) {
        found = true;
    };
    try testing.expect(found);
    // Two chunks, and every chunk is a complete BOS…EOS sequence — not one long run.
    try testing.expectEqual(@as(usize, 2), p.chunks);
    for (0..p.chunks) |c| {
        try testing.expectEqual(bos_id, p.chunk(c)[0].id);
        var has_eos = false;
        for (p.chunk(c)) |t| if (t.id == eos_id) {
            has_eos = true;
        };
        try testing.expect(has_eos);
    }
    try testing.expect(!p.hasWeights());
}

test "an unweighted prompt's first chunk is exactly what encodePadded produced" {
    // The compatibility claim that keeps the fixtures above meaningful: chunking is
    // additive. A prompt that fit in one window before must tokenize identically now,
    // padding included, or every existing SD/SDXL parity fixture has silently moved.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for ([_][]const u8{ "a red cat", "", "one two three four five", "a photo of a dog on a beach" }) |text| {
        for ([_]u32{ eos_id, 0 }) |pad| {
            const want = try tok.encodePadded(gpa, text, pad);
            defer gpa.free(want);
            var p = try tok.encodeWeighted(gpa, text, pad);
            defer p.deinit(gpa);
            try testing.expectEqual(@as(usize, 1), p.chunks);
            for (want, p.chunk(0), 0..) |w, g, i| {
                errdefer std.debug.print("'{s}' pad {d} slot {d}: {d} vs {d}\n", .{ text, pad, i, w, g.id });
                try testing.expectEqual(w, g.id);
            }
        }
    }
}

test "a pathologically nested prompt is reported rather than silently mis-read" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var deep: std.ArrayList(u8) = .empty;
    defer deep.deinit(gpa);
    for (0..64) |_| try deep.append(gpa, '(');
    try deep.appendSlice(gpa, "cat");
    for (0..64) |_| try deep.append(gpa, ')');

    try testing.expectError(error.PromptNestingTooDeep, tok.encodeWeighted(gpa, deep.items, eos_id));
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

const A1111ChunkRows = struct { ids: []const u32, mults: []const f32 };
const A1111ChunkCase = struct { text: []const u8, clip_l: []const A1111ChunkRows, clip_g: []const A1111ChunkRows };
const A1111ChunkFixtures = struct { a1111_chunks: []const A1111ChunkCase };

test "A1111 chunking matches tokenize_line, BREAK and comma backtrack included" {
    // The full A1111 front end end to end: `parseAttention` then `encodeParts`, against
    // A1111's own `parse_prompt_attention` piped through its own `tokenize_line`. The
    // cases straddle the 75-token boundary with and without a nearby comma, which is the
    // only way to see the backtrack at all.
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        A1111ChunkFixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures_a1111.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for (parsed.value.a1111_chunks) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const parts = try prompt_a1111.parseAttention(arena.allocator(), c.text);

        for ([_]struct { u32, []const A1111ChunkRows }{
            .{ eos_id, c.clip_l },
            .{ 0, c.clip_g },
        }) |arm| {
            const pad, const want = arm;
            var got = try tok.encodeParts(gpa, parts, pad);
            defer got.deinit(gpa);

            errdefer std.debug.print(
                "'{s}' pad {d}: expected {d} chunks, got {d}\n",
                .{ c.text, pad, want.len, got.chunks },
            );
            try testing.expectEqual(want.len, got.chunks);
            for (want, 0..) |wc, ci| {
                for (wc.ids, wc.mults, got.chunk(ci), 0..) |wid, wm, g, j| {
                    errdefer std.debug.print(
                        "'{s}' pad {d} chunk {d} slot {d}: want id {d} w {d}, got id {d} w {d}\n",
                        .{ c.text, pad, ci, j, wid, wm, g.id, g.weight },
                    );
                    try testing.expectEqual(wid, g.id);
                    try testing.expectEqual(wm, g.weight);
                }
            }
        }
    }
}

test "BREAK closes a chunk in the A1111 dialect and is a word in ComfyUI's" {
    // The two dialects side by side on the same prompt, which is the clearest statement
    // of why `Options.prompt_syntax` is not cosmetic.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const parts = try prompt_a1111.parseAttention(arena.allocator(), "a BREAK b");
    var a1111 = try tok.encodeParts(gpa, parts, eos_id);
    defer a1111.deinit(gpa);
    var comfy = try tok.encodeWeighted(gpa, "a BREAK b", eos_id);
    defer comfy.deinit(gpa);

    // A1111 splits into two windows; ComfyUI keeps one and spells `break` as a token.
    try testing.expectEqual(@as(usize, 2), a1111.chunks);
    try testing.expectEqual(@as(usize, 1), comfy.chunks);
    const break_id = tok.vocab.get("break</w>").?;
    var comfy_has_break = false;
    for (comfy.chunk(0)) |t| if (t.id == break_id) {
        comfy_has_break = true;
    };
    try testing.expect(comfy_has_break);
    for (a1111.chunk(0)) |t| try testing.expect(t.id != break_id);
}
