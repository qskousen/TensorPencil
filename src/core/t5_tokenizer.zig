//! T5 SentencePiece-**Unigram** tokenizer — the second of Anima's two prompt
//! tokenizers, and the one that is genuinely new to this engine.
//!
//! Anima tokenizes every prompt TWICE (`comfy/text_encoders/anima.py`): once with
//! Qwen2 byte-level BPE (`tokenizer.zig`) for the Qwen3-0.6B encoder whose hidden
//! states are the `llm_adapter`'s cross-attention *source*, and once with this,
//! whose ids index the adapter's own `Embedding(32128, 1024)` as
//! `target_input_ids`. So this file does not feed a T5 model — there is no T5 here
//! at all. It feeds a 1024-wide learned embedding that happens to have been
//! trained against T5's vocabulary.
//!
//! ⚠️ **The emphasis weights live on THIS branch only.** `AnimaTokenizer` forces
//! every Qwen3 weight to 1.0 and keeps the T5 ones, which then multiply the
//! adapter's output rows. `(a:1.5)` in an Anima prompt is a T5-side effect.
//!
//! Reference: HuggingFace `tokenizers`' Unigram model + `Precompiled`/`Strip`/
//! `Replace` normalizers + `Metaspace` pre-tokenizer, wrapped in ComfyUI's
//! `sd1_clip.SDTokenizer`. Pinned by `tools/gen_anima_prompt_fixtures.py`, which
//! *executes* `AnimaTokenizer` rather than re-deriving it.
//!
//! ## The five conventions that are not derivable, each a silent wrong answer
//!
//! 1. **Weighted segments are tokenized SEPARATELY** — one tokenizer call each.
//!    `Metaspace`'s prefix therefore fires per segment, so `a cat(s:1.1)` is
//!    `▁a ▁cat ▁s …`, never `▁a ▁cats …`. ⚠️ A split at a SPACE is transparent
//!    (the segment's own prefix reproduces the space's `▁`), so only a mid-word
//!    split reveals a wrong port — which is why the fixture carries one.
//! 2. **`</s>` appears once, at the very end, and per-segment ones are dropped.**
//!    `SDTokenizer` slices `input_ids[0:-1]` per segment — removing what
//!    `TemplateProcessing` appended — then appends `self.end_token` once for the
//!    whole chunk. `end_token` is 1 because `SDTokenizer.__init__` resolved it
//!    from `tokenizer("")["input_ids"][0]`, not because it was configured.
//! 3. **The `▁` prefix is added iff the span starts at byte 0 of the call's
//!    input** — `Metaspace`'s `prepend_scheme = "first"` tests
//!    `offsets_original().0 == 0`, NOT "is this the first split". An added token
//!    splits the input into spans, and a later span gets no prefix:
//!    `prefix<extra_id_0>suffix` ends `… [s][uff][ix]`, with no `▁`.
//! 4. **An empty normalized span produces NOTHING**, not a bare `▁`. `" "`,
//!    `"   "` and `"\x01"` all tokenize to the empty id list (then the one
//!    trailing `</s>` of rule 2). The prefix is never prepended to an empty
//!    string, so a naive "always prepend" gives every whitespace-only prompt an
//!    extra token.
//! 5. **Unknown pieces FUSE.** Consecutive `<unk>`s inside one pre-token collapse
//!    to a single `<unk>` (`fuse_unk`), so the two Gothic letters of `𐌰𐌱` are one
//!    id, while two unknown characters separated by a space are two — they land in
//!    different pre-tokens, and fusion does not cross them.
//!
//! ## The normalizer is a real piece of SentencePiece, not an approximation
//!
//! `Precompiled(precompiled_charsmap)` is a serialized darts-clone double-array
//! trie plus a NUL-separated replacement blob — SentencePiece's `nmt_nfkc`. It is
//! embedded verbatim (237 KB) and walked here, rather than approximated by "NFKC
//! is identity on ASCII": it maps NBSP/ZWSP/ZWJ/ideographic space to a plain
//! space, deletes 30 control characters, folds ligatures (`ﬁ`→`fi`), fullwidth
//! Latin, superscripts and circled digits, expands `Ⅸ`→`IX` (one char to two),
//! and composes `e`+U+0301 into `é`. Every one of those changes the id stream of a
//! realistic prompt.
//!
//! ⚠️ **Grapheme clustering, and why "base + marks" is exact here rather than a
//! shortcut.** The reference segments into UAX#29 extended grapheme clusters and
//! only tries a whole-cluster charsmap lookup when the cluster is **under 6
//! bytes**, falling back to per-codepoint otherwise. Every cluster that can pass
//! that gate is a base codepoint plus one or two combining marks (a Hangul L+V+T
//! jamo sequence is 9 bytes, an emoji ZWJ sequence 11, a flag 8 — all take the
//! per-codepoint path in the reference too), so clustering as base+Extend is
//! equivalent to full UAX#29 for every input the gate admits. CR LF is the one
//! non-mark cluster short enough to matter and is handled explicitly.

//! ## Why this is not a variant of `tokenizer.zig`'s existing `.unigram` kind
//!
//! ⚠️ `tokenizer.zig` already has a SentencePiece-Unigram path
//! (`initUnigramFromTokenizerJson`, for Snowflake Arctic Embed / GTE), and it is a
//! **different tokenizer**, not a configuration of this one. Three differences, each
//! decisive:
//!
//! * Its pre-tokenization splits on ASCII whitespace and prepends `▁` to EVERY word.
//!   Metaspace with `prepend_scheme = "first"` prepends to exactly one span
//!   (rule 3), and its split is on `▁` after a `Replace`, not on whitespace.
//! * It has **no normalizer at all** — no charsmap, so no NFKC, no NBSP-to-space and
//!   no control-character deletion.
//! * Its lattice has no `<unk>` NODES: a word it cannot segment becomes a single
//!   `<unk>` for the whole word, where `tokenizers` inserts one per unsegmentable
//!   *character* at `min_score - 10` and then fuses runs. Its tie-break also prefers
//!   the shortest incoming piece where `tokenizers` prefers the longest.
//!
//! Those last two are worth a look by whoever owns the embedding tokenizers — its own
//! fixture passes, so its corpus evidently never exercises a partially-unknown word —
//! but they are NOT this file's to change, since altering them would move every
//! embedding this repo has computed. Kept separate deliberately.

const std = @import("std");
const tables = @import("unicode_tables.zig");
const prompt_weights = @import("prompt_weights.zig");

const vocab_bin = @embedFile("assets/t5_tokenizer/vocab.bin");
const charsmap_bin = @embedFile("assets/t5_tokenizer/charsmap.bin");

/// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's visible stand-in for a space,
/// and `Metaspace`'s `replacement`.
pub const meta = "\u{2581}";

/// `</s>`. Appended once per prompt; see rule 2 in the module header.
pub const eos_id: u32 = 1;

/// `tokenizers`' `K_UNK_PENALTY`: an unknown character's lattice score is the
/// vocabulary's minimum score minus this.
const unk_penalty: f64 = 10.0;

/// One token id plus the weight its enclosing parentheses gave it.
pub const Weighted = struct { id: u32, weight: f32 };

fn inRanges(comptime ranges: []const tables.Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo) {
            hi = mid;
        } else if (cp > ranges[mid].hi) {
            lo = mid + 1;
        } else return true;
    }
    return false;
}

/// Rust's `char::is_whitespace` (Unicode `White_Space`), which is what the `Strip`
/// normalizer trims.
fn isWhitespace(cp: u21) bool {
    return inRanges(&tables.whitespace_ranges, cp);
}

/// UAX#29 `Extend`, restricted to what can appear in a sub-6-byte cluster: combining
/// marks, ZWJ, and the two variation-selector blocks. See the module header for why
/// that restriction is exact rather than approximate.
fn isExtend(cp: u21) bool {
    if (cp == 0x200D) return true; // ZWJ
    if (cp >= 0xFE00 and cp <= 0xFE0F) return true; // variation selectors
    if (cp >= 0xE0100 and cp <= 0xE01EF) return true; // variation selectors supplement
    return inRanges(&tables.mark_ranges, cp);
}

fn cpLen(b: u8) usize {
    return std.unicode.utf8ByteSequenceLength(b) catch 1;
}

fn cpAt(s: []const u8, i: usize) struct { cp: u21, len: usize } {
    const n = @min(cpLen(s[i]), s.len - i);
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch {
        // Invalid UTF-8 cannot reach here from a Zig string literal or a validated
        // prompt, but a prompt is caller data. Treat the byte as its own scalar so
        // the tokenizer degrades to <unk> rather than failing the render.
        return .{ .cp = 0xFFFD, .len = 1 };
    };
    return .{ .cp = cp, .len = n };
}

/// Byte length of the grapheme cluster starting at `i`.
fn graphemeLen(s: []const u8, i: usize) usize {
    if (s[i] == '\r' and i + 1 < s.len and s[i + 1] == '\n') return 2;
    const first = cpAt(s, i);
    var end = i + first.len;
    while (end < s.len) {
        const nxt = cpAt(s, end);
        if (!isExtend(nxt.cp)) break;
        end += nxt.len;
    }
    return end - i;
}

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// id -> piece text.
    pieces: []const []const u8,
    /// id -> log-probability.
    scores: []const f64,
    /// piece text -> id.
    ids: std.StringHashMapUnmanaged(u32),
    /// Ids of the tokens matched verbatim before normalization, longest literal
    /// first so a greedy scan finds `<extra_id_99>` before any prefix of it.
    added: []const u32,
    /// Which bytes can begin an added token, so the scan costs nothing on ordinary
    /// text (for T5 the only member is `<`).
    added_first: [256]bool,
    unk_id: u32,
    unk_score: f64,
    max_piece: usize,
    /// darts-clone double-array units, byte-swapped into host order at init.
    trie: []const u32,
    /// NUL-separated replacement strings the trie's leaf values index into.
    blob: []const u8,

    pub fn init(gpa: std.mem.Allocator) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // --- vocab.bin ------------------------------------------------------
        if (vocab_bin.len < 20 or !std.mem.eql(u8, vocab_bin[0..4], "TPT5")) return error.InvalidVocabAsset;
        const version = std.mem.readInt(u32, vocab_bin[4..8], .little);
        if (version != 2) return error.InvalidVocabAsset;
        const count = std.mem.readInt(u32, vocab_bin[8..12], .little);
        const unk_id = std.mem.readInt(u32, vocab_bin[12..16], .little);
        const n_added = std.mem.readInt(u32, vocab_bin[16..20], .little);
        if (unk_id >= count) return error.InvalidVocabAsset;

        const pieces = try alloc.alloc([]const u8, count);
        const scores = try alloc.alloc(f64, count);
        var ids: std.StringHashMapUnmanaged(u32) = .empty;
        try ids.ensureTotalCapacity(alloc, count);

        var off: usize = 20;
        var max_piece: usize = 0;
        var min_score: f64 = std.math.inf(f64);
        for (0..count) |id| {
            if (off + 5 > vocab_bin.len) return error.InvalidVocabAsset;
            const score: f32 = @bitCast(std.mem.readInt(u32, vocab_bin[off..][0..4], .little));
            const len = vocab_bin[off + 4];
            off += 5;
            if (off + len > vocab_bin.len) return error.InvalidVocabAsset;
            const text = vocab_bin[off..][0..len];
            off += len;
            pieces[id] = text;
            scores[id] = score;
            max_piece = @max(max_piece, len);
            min_score = @min(min_score, @as(f64, score));
            // The generator asserts uniqueness, so a collision here is a corrupt
            // asset rather than a vocabulary with duplicates.
            if (ids.getOrPutAssumeCapacity(text).found_existing) return error.InvalidVocabAsset;
            ids.putAssumeCapacity(text, @intCast(id));
        }

        const added = try alloc.alloc(u32, n_added);
        for (added, 0..) |*slot, i| {
            if (off + 4 > vocab_bin.len) return error.InvalidVocabAsset;
            const id = std.mem.readInt(u32, vocab_bin[off..][0..4], .little);
            if (id >= count) return error.InvalidVocabAsset;
            slot.* = id;
            off += 4;
            _ = i;
        }
        if (off != vocab_bin.len) return error.InvalidVocabAsset;
        // Longest literal first: a greedy scan must prefer `<extra_id_99>` to any
        // shorter added token that prefixes it.
        std.mem.sort(u32, added, pieces, struct {
            fn lt(ctx: []const []const u8, a: u32, b: u32) bool {
                if (ctx[a].len != ctx[b].len) return ctx[a].len > ctx[b].len;
                return a < b;
            }
        }.lt);
        var added_first = [_]bool{false} ** 256;
        for (added) |id| if (pieces[id].len > 0) {
            added_first[pieces[id][0]] = true;
        };

        // --- charsmap.bin ---------------------------------------------------
        if (charsmap_bin.len < 4) return error.InvalidCharsmapAsset;
        const trie_bytes = std.mem.readInt(u32, charsmap_bin[0..4], .little);
        if (trie_bytes % 4 != 0 or 4 + trie_bytes > charsmap_bin.len) return error.InvalidCharsmapAsset;
        // `@embedFile` gives byte alignment, so the u32 units are copied rather than
        // cast — and read little-endian explicitly, since the blob's endianness is a
        // property of the file, not of the host.
        const trie = try alloc.alloc(u32, trie_bytes / 4);
        for (trie, 0..) |*u, i| {
            u.* = std.mem.readInt(u32, charsmap_bin[4 + i * 4 ..][0..4], .little);
        }

        return .{
            .arena = arena,
            .pieces = pieces,
            .scores = scores,
            .ids = ids,
            .added = added,
            .added_first = added_first,
            .unk_id = unk_id,
            .unk_score = min_score - unk_penalty,
            .max_piece = max_piece,
            .trie = trie,
            .blob = charsmap_bin[4 + trie_bytes ..],
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The full ComfyUI prompt path: resolve `(a:1.2)` emphasis, tokenize each
    /// weighted segment on its own, concatenate, and append one `</s>`. Caller frees.
    pub fn encodeWeighted(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]Weighted {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();

        var segs: std.ArrayList(prompt_weights.Segment) = .empty;
        try prompt_weights.segments(arena, &segs, text);

        var out: std.ArrayList(Weighted) = .empty;
        errdefer out.deinit(gpa);

        for (segs.items) |s| {
            const plain = try prompt_weights.unescape(arena, s.text);
            if (plain.len == 0) continue; // ComfyUI drops empty segments outright.
            // The nesting product accumulated in f64 to match Python; this is the
            // single narrowing to the f32 the model multiplies with.
            const w: f32 = @floatCast(s.weight);
            var ids: std.ArrayList(u32) = .empty;
            try self.tokenizeCall(arena, &ids, plain);
            for (ids.items) |id| try out.append(gpa, .{ .id = id, .weight = w });
        }

        // Rule 2: exactly one `</s>`, for the whole prompt, at weight 1.0.
        try out.append(gpa, .{ .id = eos_id, .weight = 1.0 });
        return out.toOwnedSlice(gpa);
    }

    /// `encodeWeighted` with the weights dropped — for callers (and tests) that only
    /// want the id stream. Caller frees.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        const w = try self.encodeWeighted(gpa, text);
        defer gpa.free(w);
        const out = try gpa.alloc(u32, w.len);
        for (w, out) |src, *dst| dst.* = src.id;
        return out;
    }

    /// One `self.tokenizer(word)` call: split out added tokens, then normalize and
    /// pre-tokenize each remaining span and run the Unigram lattice over its
    /// pre-tokens.
    fn tokenizeCall(
        self: *const Tokenizer,
        arena: std.mem.Allocator,
        out: *std.ArrayList(u32),
        text: []const u8,
    ) !void {
        var span_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            const hit: ?u32 = if (self.added_first[text[i]]) self.addedAt(text[i..]) else null;
            if (hit) |id| {
                if (i > span_start) try self.tokenizeSpan(arena, out, text[span_start..i], span_start == 0);
                try out.append(arena, id);
                i += self.pieces[id].len;
                span_start = i;
            } else i += 1;
        }
        if (span_start < text.len) {
            try self.tokenizeSpan(arena, out, text[span_start..], span_start == 0);
        }
    }

    /// The longest added token that is a prefix of `rest`, or null.
    fn addedAt(self: *const Tokenizer, rest: []const u8) ?u32 {
        for (self.added) |id| {
            const lit = self.pieces[id];
            if (lit.len <= rest.len and std.mem.eql(u8, rest[0..lit.len], lit)) return id;
        }
        return null;
    }

    /// Normalize one non-added span, split it into `Metaspace` pre-tokens, and
    /// append each one's ids. `at_zero` is rule 3's condition.
    fn tokenizeSpan(
        self: *const Tokenizer,
        arena: std.mem.Allocator,
        out: *std.ArrayList(u32),
        span: []const u8,
        at_zero: bool,
    ) !void {
        const normalized = try self.normalize(arena, span);
        // Rule 4: an empty normalized span contributes nothing — in particular NOT
        // a bare `▁` from the prefix that would otherwise be prepended.
        if (normalized.len == 0) return;

        // Metaspace, in the reference's order: replace every space with `▁`, THEN test
        // whether the result already starts with one, THEN prepend.
        var replaced: std.ArrayList(u8) = .empty;
        for (normalized) |b| {
            if (b == ' ') try replaced.appendSlice(arena, meta) else try replaced.append(arena, b);
        }

        // Rule 3: the prefix goes on iff this span began at byte 0 of the tokenizer
        // call's input and the (space-replaced) text does not already start with `▁`.
        // ⚠️ The order matters for readability rather than for the answer — testing
        // `normalized` instead would have to special-case a leading plain space, which
        // is the same condition written less obviously.
        var s: std.ArrayList(u8) = .empty;
        if (at_zero and !std.mem.startsWith(u8, replaced.items, meta)) {
            try s.appendSlice(arena, meta);
        }
        try s.appendSlice(arena, replaced.items);

        // The MergedWithNext split on `▁`: each piece starts with one, text before the
        // first forms its own piece, and empty pieces are dropped.
        var at: usize = 0;
        while (at < s.items.len) {
            var end = at + 1;
            while (end < s.items.len) {
                if (std.mem.startsWith(u8, s.items[end..], meta)) break;
                end += 1;
            }
            try self.viterbi(arena, out, s.items[at..end]);
            at = end;
        }
    }

    /// `Precompiled(charsmap)` -> `Strip(right)` -> `Replace(/ {2,}/, "▁")`.
    fn normalize(self: *const Tokenizer, arena: std.mem.Allocator, span: []const u8) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;

        // 1. The charsmap, over grapheme clusters. ⚠️ A cluster lookup that matches
        //    only a PREFIX of the cluster still wins and the rest of the cluster is
        //    dropped — `common_prefix_search` returns every leaf on the path and the
        //    reference takes the last. Faithful, and the reason this is not written
        //    as "look up the whole cluster or nothing".
        var i: usize = 0;
        while (i < span.len) {
            const glen = graphemeLen(span, i);
            if (glen < 6) {
                if (self.transform(span[i..][0..glen])) |rep| {
                    try buf.appendSlice(arena, rep);
                    i += glen;
                    continue;
                }
            }
            const end = i + glen;
            while (i < end) {
                const c = cpAt(span, i);
                if (self.transform(span[i..][0..c.len])) |rep| {
                    try buf.appendSlice(arena, rep);
                } else {
                    try buf.appendSlice(arena, span[i..][0..c.len]);
                }
                i += c.len;
            }
        }

        // 2. Strip trailing whitespace.
        var n = buf.items.len;
        while (n > 0) {
            var start = n - 1;
            while (start > 0 and (buf.items[start] & 0xC0) == 0x80) start -= 1;
            const c = cpAt(buf.items, start);
            if (start + c.len != n or !isWhitespace(c.cp)) break;
            n = start;
        }
        const stripped = buf.items[0..n];

        // 3. Collapse runs of two or more ASCII spaces into one `▁`. ⚠️ This runs
        //    AFTER the charsmap, so an NBSP that became a space participates.
        var out: std.ArrayList(u8) = .empty;
        var j: usize = 0;
        while (j < stripped.len) {
            if (stripped[j] == ' ') {
                var k = j;
                while (k < stripped.len and stripped[k] == ' ') k += 1;
                if (k - j >= 2) try out.appendSlice(arena, meta) else try out.append(arena, ' ');
                j = k;
            } else {
                try out.append(arena, stripped[j]);
                j += 1;
            }
        }
        return out.items;
    }

    /// Longest charsmap replacement for `key`, or null when the trie has no leaf on
    /// its byte path. This is darts-clone's `common_prefix_search` keeping only the
    /// last (longest) hit, then reading the NUL-terminated string at that offset.
    fn transform(self: *const Tokenizer, key: []const u8) ?[]const u8 {
        if (self.trie.len == 0) return null;
        var node: usize = 0;
        var unit = self.trie[node];
        node ^= unitOffset(unit);
        var best: ?u32 = null;
        for (key) |c| {
            if (c == 0) break;
            node ^= c;
            if (node >= self.trie.len) return null;
            unit = self.trie[node];
            if (unitLabel(unit) != c) break;
            node ^= unitOffset(unit);
            if (unitHasLeaf(unit)) {
                if (node >= self.trie.len) return null;
                best = unitValue(self.trie[node]);
            }
        }
        const at = best orelse return null;
        if (at >= self.blob.len) return null;
        const rest = self.blob[at..];
        const end = std.mem.indexOfScalar(u8, rest, 0) orelse rest.len;
        return rest[0..end];
    }

    /// Unigram Viterbi over one pre-token, appending the best path's ids.
    ///
    /// Node insertion order is reproduced exactly because it decides ties: begin
    /// position ascending, and within a begin, piece length ascending, with the
    /// `<unk>` node last. The reference keeps the FIRST strict maximum, so at a
    /// given end position the smallest begin (longest piece) wins a tie.
    fn viterbi(
        self: *const Tokenizer,
        arena: std.mem.Allocator,
        out: *std.ArrayList(u32),
        text: []const u8,
    ) !void {
        if (text.len == 0) return;
        const n = text.len;

        const best = try arena.alloc(f64, n + 1);
        const from = try arena.alloc(usize, n + 1);
        const tok = try arena.alloc(u32, n + 1);
        @memset(best, -std.math.inf(f64));
        best[0] = 0;

        var begin: usize = 0;
        while (begin < n) {
            if (best[begin] == -std.math.inf(f64)) {
                begin += 1;
                continue;
            }
            const mblen = @min(cpLen(text[begin]), n - begin);
            var has_single = false;
            var len: usize = 1;
            const cap = @min(self.max_piece, n - begin);
            while (len <= cap) : (len += 1) {
                const id = self.ids.get(text[begin..][0..len]) orelse continue;
                if (len == mblen) has_single = true;
                const cand = best[begin] + self.scores[id];
                const end = begin + len;
                if (cand > best[end]) {
                    best[end] = cand;
                    from[end] = begin;
                    tok[end] = id;
                }
            }
            if (!has_single) {
                const cand = best[begin] + self.unk_score;
                const end = begin + mblen;
                if (cand > best[end]) {
                    best[end] = cand;
                    from[end] = begin;
                    tok[end] = self.unk_id;
                }
            }
            begin += 1;
        }

        // Every codepoint boundary is reachable (a piece or an `<unk>` always spans
        // to the next one), so `n` is too.
        std.debug.assert(best[n] != -std.math.inf(f64));

        // Backtrack, then emit forwards, fusing consecutive `<unk>` (rule 5).
        var path: std.ArrayList(u32) = .empty;
        var at = n;
        while (at > 0) {
            try path.append(arena, tok[at]);
            at = from[at];
        }
        var k = path.items.len;
        var prev_unk = false;
        while (k > 0) {
            k -= 1;
            const id = path.items[k];
            const is_unk = id == self.unk_id;
            if (is_unk and prev_unk) continue;
            try out.append(arena, id);
            prev_unk = is_unk;
        }
    }
};

fn unitHasLeaf(u: u32) bool {
    return ((u >> 8) & 1) == 1;
}
fn unitValue(u: u32) u32 {
    return u & ((@as(u32, 1) << 31) - 1);
}
fn unitLabel(u: u32) u32 {
    return u & ((@as(u32, 1) << 31) | 0xFF);
}
fn unitOffset(u: u32) usize {
    return @as(usize, (u >> 10) << @intCast((u & (@as(u32, 1) << 9)) >> 6));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const Case = struct {
    label: []const u8,
    text: []const u8,
    t5_ids: []const u32,
    t5_weights: []const f32,
    qwen_ids: []const u32,
};
const Fixtures = struct {
    t5_tokenizer_sha256: []const u8,
    t5_unk_id: u32,
    qwen_pad_token: u32,
    llm_adapter_min_rows: usize,
    cases: []const Case,
};

test "the T5 tokenizer matches ComfyUI's AnimaTokenizer on every fixture case" {
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        Fixtures,
        gpa,
        @embedFile("assets/t5_tokenizer/fixtures.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    try testing.expectEqual(parsed.value.t5_unk_id, tok.unk_id);

    for (parsed.value.cases) |c| {
        const got = try tok.encodeWeighted(gpa, c.text);
        defer gpa.free(got);
        errdefer {
            std.debug.print("case \"{s}\" text {any}\n", .{ c.label, c.text });
            std.debug.print("  want {d} ids:", .{c.t5_ids.len});
            for (c.t5_ids) |id| std.debug.print(" {d}", .{id});
            std.debug.print("\n  got  {d} ids:", .{got.len});
            for (got) |w| std.debug.print(" {d}", .{w.id});
            std.debug.print("\n", .{});
        }
        try testing.expectEqual(c.t5_ids.len, got.len);
        for (c.t5_ids, c.t5_weights, got) |want_id, want_w, g| {
            try testing.expectEqual(want_id, g.id);
            try testing.expectApproxEqAbs(want_w, g.weight, 1e-6);
        }
    }
}

test "an empty normalized span contributes no token, not a bare metaspace" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    // Rule 4. Each of these normalizes to "" — whitespace stripped, control chars
    // deleted by the charsmap — and must yield only the trailing `</s>`.
    for ([_][]const u8{ "", " ", "   ", "\t", "\x01", "\x01\x02", " \t\n " }) |text| {
        const got = try tok.encode(gpa, text);
        defer gpa.free(got);
        errdefer std.debug.print("text {any} -> {any}\n", .{ text, got });
        try testing.expectEqual(@as(usize, 1), got.len);
        try testing.expectEqual(eos_id, got[0]);
    }
}

test "the charsmap normalizer is applied, not assumed to be identity" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Direct `normalize` checks, so a failure localizes to the charsmap rather than
    // to the lattice. Values verified against `tokenizers.normalizers.Precompiled`.
    const Pair = struct { in: []const u8, want: []const u8 };
    for ([_]Pair{
        .{ .in = "the \u{FB01}re", .want = "the fire" }, // ﬁ ligature
        .{ .in = "\u{FF28}\u{FF45}", .want = "He" }, // fullwidth latin
        .{ .in = "\u{2168}", .want = "IX" }, // one char -> two
        .{ .in = "x\u{00B2}", .want = "x2" }, // superscript
        .{ .in = "\u{2460}", .want = "1" }, // circled digit
        .{ .in = "a\u{00A0}b", .want = "a b" }, // NBSP -> space
        .{ .in = "a\u{200B}b", .want = "a b" }, // ZWSP -> space
        .{ .in = "a\u{200D}b", .want = "a b" }, // ZWJ -> space
        .{ .in = "a\u{3000}b", .want = "a b" }, // ideographic space
        // ⚠️ Needs grapheme clustering: `e` + U+0301 is one cluster with a charsmap
        // entry, where per-codepoint lookup leaves both characters alone.
        .{ .in = "cafe\u{0301}", .want = "caf\u{00E9}" },
        .{ .in = "a\u{0301}b", .want = "\u{00E1}b" },
        // ...but two stacked marks exceed nothing and simply have no entry.
        .{ .in = "x\u{0301}\u{0301}", .want = "x\u{0301}\u{0301}" },
        .{ .in = "a\x01b", .want = "ab" }, // control char deleted
        .{ .in = "a\tb", .want = "a b" }, // tab -> space
        .{ .in = "a\r\nb", .want = "a b" }, // ⚠️ CR LF is ONE cluster -> ONE space
        // Strip(right), then the 2+-space collapse.
        .{ .in = "trailing   ", .want = "trailing" },
        .{ .in = "a  b", .want = "a" ++ meta ++ "b" },
        .{ .in = "   a", .want = meta ++ "a" },
    }) |p| {
        const got = try tok.normalize(a, p.in);
        errdefer std.debug.print("normalize({any}) = {any}, want {any}\n", .{ p.in, got, p.want });
        try testing.expectEqualStrings(p.want, got);
    }
}

test "unknown characters fuse inside a pre-token but not across one" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    // Rule 5. Two Gothic letters are outside T5's vocabulary; adjacent they are one
    // `<unk>`, separated by a space they are two.
    {
        const got = try tok.encode(gpa, "\u{10330}\u{10331}");
        defer gpa.free(got);
        errdefer std.debug.print("fused -> {any}\n", .{got});
        try testing.expectEqual(@as(usize, 3), got.len); // ▁, <unk>, </s>
        try testing.expectEqual(tok.unk_id, got[1]);
    }
    {
        const got = try tok.encode(gpa, "\u{10330} \u{10331}");
        defer gpa.free(got);
        errdefer std.debug.print("split -> {any}\n", .{got});
        var unks: usize = 0;
        for (got) |id| if (id == tok.unk_id) {
            unks += 1;
        };
        try testing.expectEqual(@as(usize, 2), unks);
    }
}

test "the metaspace prefix keys off byte offset zero, not the first span" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    // Rule 3. `<extra_id_0>` (id 32099) is an added token, so it splits the input;
    // the span after it starts at a nonzero offset and gets NO `▁`. If the prefix
    // were per-span, `suffix` would tokenize as `▁suffi` `x` instead.
    const got = try tok.encode(gpa, "prefix<extra_id_0>suffix");
    defer gpa.free(got);
    errdefer {
        std.debug.print("ids:", .{});
        for (got) |id| std.debug.print(" {d}({s})", .{ id, tok.pieces[id] });
        std.debug.print("\n", .{});
    }
    const at = std.mem.indexOfScalar(u32, got, 32099) orelse return error.AddedTokenMissing;
    try testing.expect(!std.mem.startsWith(u8, tok.pieces[got[at + 1]], meta));
    // ...while the span that DOES start at offset 0 gets one.
    try testing.expect(std.mem.startsWith(u8, tok.pieces[got[0]], meta));
}
