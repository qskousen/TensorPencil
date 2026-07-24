//! Qwen2 byte-level BPE tokenizer (pure Zig port of transformers'
//! `Qwen2Tokenizer`, which ComfyUI uses for the Krea 2 text encoder).
//!
//! Pipeline: split out special tokens -> pretokenize (hand-rolled scanner for
//! the fixed Qwen2 regex, using generated Unicode tables) -> byte-level BPE
//! over vocab.json/merges.txt. The vocab and merges are embedded in the
//! binary from src/core/assets/qwen_tokenizer/.

const std = @import("std");
const tables = @import("unicode_tables.zig");
const gguf_mod = @import("gguf.zig");

const vocab_json = @embedFile("assets/qwen_tokenizer/vocab.json");
const merges_txt = @embedFile("assets/qwen_tokenizer/merges.txt");

pub const pad_token: u32 = 151643; // <|endoftext|>
pub const im_start: u32 = 151644;
pub const im_end: u32 = 151645;
pub const think_open: u32 = 151667;
pub const think_close: u32 = 151668;

/// A token matched verbatim before pretokenization.
pub const Special = struct { text: []const u8, id: u32 };

/// Added special tokens of the embedded Qwen3 tokenizer (from
/// tokenizer_config.json).
const special_tokens = [_]Special{
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
fn isMark(cp: u21) bool {
    return inRanges(&tables.mark_ranges, cp);
}
fn isLowercase(cp: u21) bool {
    return inRanges(&tables.lowercase_ranges, cp); // \p{Ll}
}
fn isUppercase(cp: u21) bool {
    return inRanges(&tables.uppercase_ranges, cp); // \p{Lu} ∪ \p{Lt}
}
/// tekken "upper" letter class [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] = letters that
/// aren't lowercase, plus marks.
fn isTekHi(cp: u21) bool {
    return isMark(cp) or (isLetter(cp) and !isLowercase(cp));
}
/// tekken "lower" letter class [\p{Ll}\p{Lm}\p{Lo}\p{M}] = letters that aren't
/// upper/title-case, plus marks.
fn isTekLo(cp: u21) bool {
    return isMark(cp) or (isLetter(cp) and !isUppercase(cp));
}

/// Pretokenizer regex variant (tokenizer.ggml.pre). qwen35 differs from
/// qwen2 in two places: letter runs are [\p{L}\p{M}]+ (combining marks join
/// the run) and the punctuation class also excludes \p{M}.
pub const Pretok = enum { qwen2, qwen35, tekken };

/// Tokenizer algorithm. `bpe` is GPT-2 byte-level BPE (Qwen family);
/// `spm` is SentencePiece (tokenizer.ggml.model == "llama"; Gemma 3, Llama);
/// `gemma4` is "SPM-style BPE" (tokenizer.ggml.model == "gemma4"): rank-based
/// BPE merges over ▁-escaped raw UTF-8, newline-only pre-split, `<0xNN>` byte
/// fallback — shares the SPM vocab layout but merges by rank, not score.
/// `unigram` is SentencePiece Unigram (XLM-RoBERTa / GTE; Snowflake Arctic
/// Embed): whitespace pre-split, ▁-prefix per word, Viterbi over per-piece
/// log-scores, whole-word `<unk>` fallback (no byte fallback).
pub const Kind = enum { bpe, spm, gemma4, unigram };

/// SentencePiece meta symbol ▁ (U+2581) standing in for a space.
const spm_space = "\xe2\x96\x81";

// --- Tokenizer -------------------------------------------------------------

const Merge = struct { rank: u32, merged: u32 };

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// id -> raw bytes. The embedded vocab covers regular tokens only;
    /// GGUF vocabs also cover their control tokens (literal text).
    id_to_bytes: [][]const u8,
    /// single byte -> vocab id.
    byte_id: [256]u32,
    /// (left_id << 32 | right_id) -> merge.
    merges: std.AutoHashMapUnmanaged(u64, Merge),
    /// Special tokens matched verbatim before pretokenization.
    specials: []const Special = &special_tokens,
    /// Vocab-dependent template/stop ids (chat.zig reads these through
    /// chat.applyTokenizer): end-of-turn (<|im_end|> / eos), pad
    /// (<|endoftext|>), and the "\n" glue token.
    turn_end: u32 = im_end,
    pad: u32 = pad_token,
    newline: u32 = 198,
    /// Beginning-of-sequence token to prepend once at the start of the prompt,
    /// or null when the model doesn't use one. Set from a GGUF's
    /// `tokenizer.ggml.add_bos_token` + `bos_token_id` (Mistral/llama want it;
    /// Qwen3 does not). chat.zig prepends it via `appendBos`.
    bos: ?u32 = null,
    /// End-of-sequence token (`tokenizer.ggml.eos_token_id`), when the vocab
    /// declares one distinct from `turn_end`. A model can end a turn on EITHER
    /// its turn marker OR raw `<eos>` (some finetunes emit eos), so chat.isStop
    /// treats both as stops — else the engine runs past the model's own end and
    /// degenerates. Null when unknown/absent.
    eos: ?u32 = null,
    pretok: Pretok = .qwen2,
    kind: Kind = .bpe,
    /// SentencePiece data (kind == .spm; empty for BPE): per-id merge score,
    /// stored-text -> id (for merge lookups / text_to_token), byte-fallback
    /// map (byte -> "<0xNN>" token id), and the unknown-token id.
    spm_scores: []const f32 = &.{},
    spm_text_id: std.StringHashMapUnmanaged(u32) = .empty,
    spm_byte_id: [256]u32 = @splat(0),
    spm_unk: u32 = 0,
    /// Gemma 4 BPE merge ranks (kind == .gemma4): key is `left ++ '\x00' ++
    /// right` (token texts contain no NUL), value is the merge rank (lower =
    /// applied first). Lookups reuse `spm_text_id` (escaped-form text -> id)
    /// and `spm_byte_id` (byte-fallback) exactly like the SPM path.
    gemma4_ranks: std.StringHashMapUnmanaged(u32) = .empty,
    /// SentencePiece add_dummy_prefix: prepend ▁ at the start of a raw
    /// fragment following a special token. False for Gemma.
    add_space_prefix: bool = false,
    /// Unigram (kind == .unigram): longest piece in bytes, bounding the Viterbi
    /// back-scan. `spm_scores` holds per-id log-scores; `spm_text_id` holds the
    /// ▁-escaped piece -> id map used for lattice lookups.
    unigram_max_piece: usize = 0,

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
        while (lines.next()) |line_raw| {
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            if (line.len == 0) continue;
            if (first and std.mem.startsWith(u8, line, "#version")) {
                first = false;
                continue;
            }
            first = false;
            try addMerge(alloc, &merges, &raw_to_id, line, rank);
            rank += 1;
        }

        return .{ .arena = arena, .id_to_bytes = id_to_bytes, .byte_id = byte_id, .merges = merges };
    }

    /// Build from a GGUF's embedded tokenizer (tokenizer.ggml.* kv arrays):
    /// gpt2-style byte-level BPE with the same pipeline as the embedded
    /// Qwen3 tokenizer. CONTROL / USER_DEFINED tokens (token_type 3 / 4)
    /// become verbatim-matched specials; the template/stop ids resolve from
    /// the control tokens and the eos/padding kv entries. Everything is
    /// copied into the tokenizer's arena — the Gguf may be closed after.
    pub fn initFromGguf(gpa: std.mem.Allocator, g: *const gguf_mod.Gguf) !Tokenizer {
        const model = g.getStr("tokenizer.ggml.model") orelse return error.MissingTokenizer;
        if (std.mem.eql(u8, model, "llama")) return initSpmFromGguf(gpa, g);
        if (std.mem.eql(u8, model, "gemma4")) return initGemma4FromGguf(gpa, g);
        if (!std.mem.eql(u8, model, "gpt2")) return error.UnsupportedTokenizer;
        var pretok: Pretok = .qwen2;
        if (g.getStr("tokenizer.ggml.pre")) |pre| {
            if (std.mem.eql(u8, pre, "qwen35")) {
                pretok = .qwen35;
            } else if (std.mem.eql(u8, pre, "tekken")) {
                pretok = .tekken; // Mistral (Nemo etc.): case-split letter runs, single-digit
            } else if (!std.mem.eql(u8, pre, "qwen2")) {
                std.log.warn("gguf pretokenizer '{s}' not implemented; using the qwen2 regex", .{pre});
            }
        }
        const tokens_arr = g.getArr("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        const merges_arr = g.getArr("tokenizer.ggml.merges") orelse return error.MissingTokenizer;
        const types_arr = g.getArr("tokenizer.ggml.token_type");

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const n_tokens = tokens_arr.len;
        const id_to_bytes = try alloc.alloc([]const u8, n_tokens);
        var raw_to_id: std.StringHashMapUnmanaged(u32) = .empty;
        try raw_to_id.ensureTotalCapacity(alloc, @intCast(n_tokens));
        var specials: std.ArrayList(Special) = .empty;
        var byte_id: [256]u32 = @splat(0xFFFF_FFFF);

        var types_it: ?gguf_mod.Array.Iterator = if (types_arr) |ta| ta.iterate() else null;
        var it = tokens_arr.iterate();
        var id: u32 = 0;
        while (it.next()) |v| : (id += 1) {
            if (v != .str) return error.InvalidVocab;
            const key = v.str;
            const ttype: i64 = if (types_it) |*ti| valueInt(ti.next() orelse return error.InvalidVocab) else 1;
            if (ttype == 3 or ttype == 4) { // CONTROL / USER_DEFINED: literal text
                const text = try alloc.dupe(u8, key);
                id_to_bytes[id] = text;
                try specials.append(alloc, .{ .text = text, .id = id });
                continue;
            }
            const raw = try alloc.alloc(u8, key.len);
            const nb = decodeTokenKey(key, raw) catch return error.InvalidVocab;
            id_to_bytes[id] = raw[0..nb];
            raw_to_id.putAssumeCapacity(raw[0..nb], id);
            if (nb == 1) byte_id[raw[0]] = id;
        }
        for (byte_id) |b| if (b == 0xFFFF_FFFF) return error.IncompleteByteVocab;

        var merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty;
        var rank: u32 = 0;
        var mit = merges_arr.iterate();
        while (mit.next()) |mv| : (rank += 1) {
            if (mv != .str) return error.InvalidMerges;
            try addMerge(alloc, &merges, &raw_to_id, mv.str, rank);
        }

        var t: Tokenizer = .{
            .arena = arena,
            .id_to_bytes = id_to_bytes,
            .byte_id = byte_id,
            .merges = merges,
            .specials = try specials.toOwnedSlice(alloc),
            .newline = byte_id['\n'],
            .pretok = pretok,
        };
        const eos: ?u32 = if (g.getUint("tokenizer.ggml.eos_token_id")) |e| @intCast(e) else null;
        t.turn_end = findSpecial(t.specials, "<|im_end|>") orelse eos orelse return error.MissingTokenizer;
        t.eos = eos;
        t.pad = if (g.getUint("tokenizer.ggml.padding_token_id")) |p| @intCast(p) else eos orelse t.turn_end;
        // Prepend BOS only when the model asks for it (Mistral/llama: true;
        // Qwen3: absent/false).
        if ((g.getBool("tokenizer.ggml.add_bos_token") orelse false))
            t.bos = if (g.getUint("tokenizer.ggml.bos_token_id")) |b| @intCast(b) else null;
        return t;
    }

    /// Build from a GGUF's embedded SentencePiece tokenizer
    /// (tokenizer.ggml.model == "llama"): score-ranked bigram merges over
    /// ▁-escaped text, byte fallback (<0xNN> tokens), and CONTROL /
    /// USER_DEFINED tokens matched verbatim before merging (llama.cpp's
    /// tokenizer_st_partition, longest-first). Used by Gemma 3.
    pub fn initSpmFromGguf(gpa: std.mem.Allocator, g: *const gguf_mod.Gguf) !Tokenizer {
        const tokens_arr = g.getArr("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        const scores_arr = g.getArr("tokenizer.ggml.scores") orelse return error.MissingTokenizer;
        const types_arr = g.getArr("tokenizer.ggml.token_type") orelse return error.MissingTokenizer;

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const n_tokens = tokens_arr.len;
        const id_to_bytes = try alloc.alloc([]const u8, n_tokens);
        const scores = try alloc.alloc(f32, n_tokens);
        var text_id: std.StringHashMapUnmanaged(u32) = .empty;
        try text_id.ensureTotalCapacity(alloc, @intCast(n_tokens));
        var specials: std.ArrayList(Special) = .empty;

        var scores_it = scores_arr.iterate();
        var types_it = types_arr.iterate();
        var it = tokens_arr.iterate();
        var id: u32 = 0;
        while (it.next()) |v| : (id += 1) {
            if (v != .str) return error.InvalidVocab;
            const raw = try alloc.dupe(u8, v.str);
            scores[id] = @floatCast(valueFloat(scores_it.next() orelse return error.InvalidVocab));
            const ttype: i64 = valueInt(types_it.next() orelse return error.InvalidVocab);
            // stored-text -> id (all tokens; last id wins on duplicate text,
            // matching llama.cpp token_to_id).
            try text_id.put(alloc, raw, id);
            switch (ttype) {
                3, 4 => { // CONTROL / USER_DEFINED: literal text, verbatim-matched
                    id_to_bytes[id] = raw;
                    try specials.append(alloc, .{ .text = raw, .id = id });
                },
                6 => { // BYTE: "<0xNN>" -> the raw byte
                    const b = parseByteToken(raw) orelse return error.InvalidVocab;
                    id_to_bytes[id] = try alloc.dupe(u8, &[_]u8{b});
                },
                else => id_to_bytes[id] = try unescapeSpm(alloc, raw), // NORMAL / UNKNOWN
            }
        }

        // Special-token cache order: longest text first (llama.cpp), so a
        // longer whitespace/marker run wins over a shorter prefix.
        std.mem.sort(Special, specials.items, {}, struct {
            fn lt(_: void, a: Special, b: Special) bool {
                if (a.text.len != b.text.len) return a.text.len > b.text.len;
                return a.id < b.id;
            }
        }.lt);

        const unk: u32 = if (g.getUint("tokenizer.ggml.unknown_token_id")) |u| @intCast(u) else 0;
        var byte_id: [256]u32 = undefined;
        for (0..256) |b| {
            var buf: [6]u8 = undefined;
            const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{@as(u8, @intCast(b))}) catch unreachable;
            byte_id[b] = text_id.get(hex) orelse text_id.get(&[_]u8{@intCast(b)}) orelse unk;
        }

        var t: Tokenizer = .{
            .arena = arena,
            .id_to_bytes = id_to_bytes,
            .byte_id = @splat(0), // unused for SPM
            .merges = .empty,
            .specials = try specials.toOwnedSlice(alloc),
            .kind = .spm,
            .spm_scores = scores,
            .spm_text_id = text_id,
            .spm_byte_id = byte_id,
            .spm_unk = unk,
            .add_space_prefix = g.getBool("tokenizer.ggml.add_space_prefix") orelse true,
        };
        const eos: ?u32 = if (g.getUint("tokenizer.ggml.eos_token_id")) |e| @intCast(e) else null;
        t.turn_end = findSpecial(t.specials, "<end_of_turn>") orelse eos orelse return error.MissingTokenizer;
        t.eos = eos;
        t.pad = if (g.getUint("tokenizer.ggml.padding_token_id")) |p| @intCast(p) else eos orelse t.turn_end;
        t.newline = findSpecial(t.specials, "\n") orelse 0;
        // BOS: Gemma requires a leading <bos>; the chat_template renders it via
        // `{{ bos_token }}`, so this must be populated for the render-driven path
        // (else no BOS is emitted and the model degenerates). Metadata-driven, so
        // it stays null for SPM models that genuinely have no BOS.
        t.bos = if (g.getUint("tokenizer.ggml.bos_token_id")) |b| @intCast(b) else findSpecial(t.specials, "<bos>");
        return t;
    }

    /// Build from a GGUF's "gemma4" tokenizer (tokenizer.ggml.model ==
    /// "gemma4"): SPM-style vocab (▁-escaped tokens, `<0xNN>` byte fallback,
    /// CONTROL/USER_DEFINED specials) but merges are RANK-ordered BPE rules
    /// (llama.cpp LLAMA_VOCAB_TYPE_BPE + pre-type "gemma4"). Encoding escapes
    /// spaces to ▁ (no dummy prefix), splits on newline runs, then runs
    /// rank-BPE with byte fallback — see `encodeGemma4`.
    pub fn initGemma4FromGguf(gpa: std.mem.Allocator, g: *const gguf_mod.Gguf) !Tokenizer {
        const tokens_arr = g.getArr("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        const types_arr = g.getArr("tokenizer.ggml.token_type") orelse return error.MissingTokenizer;
        const merges_arr = g.getArr("tokenizer.ggml.merges") orelse return error.MissingTokenizer;

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const n_tokens = tokens_arr.len;
        const id_to_bytes = try alloc.alloc([]const u8, n_tokens);
        var text_id: std.StringHashMapUnmanaged(u32) = .empty;
        try text_id.ensureTotalCapacity(alloc, @intCast(n_tokens));
        var specials: std.ArrayList(Special) = .empty;

        var types_it = types_arr.iterate();
        var it = tokens_arr.iterate();
        var id: u32 = 0;
        while (it.next()) |v| : (id += 1) {
            if (v != .str) return error.InvalidVocab;
            const raw = try alloc.dupe(u8, v.str); // ▁-escaped stored form
            const ttype: i64 = valueInt(types_it.next() orelse return error.InvalidVocab);
            try text_id.put(alloc, raw, id); // escaped-form -> id (for merges/lookup)
            switch (ttype) {
                3, 4 => { // CONTROL / USER_DEFINED: literal text, verbatim-matched
                    id_to_bytes[id] = raw;
                    try specials.append(alloc, .{ .text = raw, .id = id });
                },
                6 => { // BYTE: "<0xNN>" -> the raw byte
                    const b = parseByteToken(raw) orelse return error.InvalidVocab;
                    id_to_bytes[id] = try alloc.dupe(u8, &[_]u8{b});
                },
                else => id_to_bytes[id] = try unescapeSpm(alloc, raw), // NORMAL / UNKNOWN
            }
        }

        std.mem.sort(Special, specials.items, {}, struct {
            fn lt(_: void, a: Special, b: Special) bool {
                if (a.text.len != b.text.len) return a.text.len > b.text.len;
                return a.id < b.id;
            }
        }.lt);

        const unk: u32 = if (g.getUint("tokenizer.ggml.unknown_token_id")) |u| @intCast(u) else 0;
        var byte_id: [256]u32 = undefined;
        for (0..256) |b| {
            var buf: [6]u8 = undefined;
            const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{@as(u8, @intCast(b))}) catch unreachable;
            byte_id[b] = text_id.get(hex) orelse text_id.get(&[_]u8{@intCast(b)}) orelse unk;
        }

        // Merge ranks: split each "left right" at the first space >= index 1
        // (llama.cpp word.find(' ', 1)); key by `left ++ '\x00' ++ right`.
        var ranks: std.StringHashMapUnmanaged(u32) = .empty;
        try ranks.ensureTotalCapacity(alloc, @intCast(merges_arr.len));
        var mit = merges_arr.iterate();
        var rank: u32 = 0;
        while (mit.next()) |mv| : (rank += 1) {
            if (mv != .str) return error.InvalidMerges;
            const word = mv.str;
            if (word.len < 2) return error.InvalidMerges;
            const pos = (std.mem.indexOfScalarPos(u8, word, 1, ' ')) orelse return error.InvalidMerges;
            const left = word[0..pos];
            const right = word[pos + 1 ..];
            const key = try alloc.alloc(u8, left.len + 1 + right.len);
            @memcpy(key[0..left.len], left);
            key[left.len] = 0;
            @memcpy(key[left.len + 1 ..], right);
            // First occurrence wins the (lowest) rank, matching emplace.
            const gop = try ranks.getOrPut(alloc, key);
            if (!gop.found_existing) gop.value_ptr.* = rank;
        }

        var t: Tokenizer = .{
            .arena = arena,
            .id_to_bytes = id_to_bytes,
            .byte_id = @splat(0), // unused for gemma4
            .merges = .empty,
            .specials = try specials.toOwnedSlice(alloc),
            .kind = .gemma4,
            .spm_text_id = text_id,
            .spm_byte_id = byte_id,
            .spm_unk = unk,
            .gemma4_ranks = ranks,
        };
        const eos: ?u32 = if (g.getUint("tokenizer.ggml.eos_token_id")) |e| @intCast(e) else null;
        // Gemma 4 turn markers: <|turn> (open) / <turn|> (close); the model
        // stops on the closing marker or eos.
        t.turn_end = findSpecial(t.specials, "<turn|>") orelse eos orelse return error.MissingTokenizer;
        t.eos = eos;
        t.pad = if (g.getUint("tokenizer.ggml.padding_token_id")) |p| @intCast(p) else eos orelse t.turn_end;
        t.newline = text_id.get("\n") orelse 0;
        // BOS: Gemma prompts REQUIRE a leading <bos>. The chat_template renders
        // it via `{{ bos_token }}`, so `bos` must be populated — otherwise the
        // render-driven path emits no BOS and the model degenerates (badly on
        // larger models). The hand glue adds it via specialId, but the template
        // path relies on this field.
        t.bos = if (g.getUint("tokenizer.ggml.bos_token_id")) |b| @intCast(b) else findSpecial(t.specials, "<bos>");
        return t;
    }

    /// Build a Gemma-style BPE tokenizer (kind == .gemma4) from a HuggingFace
    /// `tokenizer.json` (model.type == "BPE"): metaspace ▁, `<0xNN>` byte
    /// fallback, rank-ordered merges. Covers models sharing the SentencePiece
    /// BPE layout regardless of vocab size — EmbeddingGemma (262144) and
    /// SigLIP2's text tower (256000) both parse here. `json_bytes` is the raw
    /// file contents; everything needed is copied into the tokenizer's arena,
    /// so the buffer may be freed afterward.
    ///
    /// This reuses the exact `encodeGemma4` merge path used for GGUF gemma4
    /// vocabs (validated bit-identical to llama.cpp / transformers): the vocab
    /// pieces and merge halves are the ▁-escaped stored form, keyed the same
    /// way. HF's normalizer (Replace " "→"▁") and pre_tokenizer (Split " ")
    /// reduce to the ▁-escaping `gemma4EncodeRaw` already performs, so no
    /// separate normalization step is needed. Added tokens are matched verbatim
    /// as specials. Callers that need the model's post-processor frame
    /// (`<bos>…<eos>` vs `…<eos>`) add it themselves — `encode` emits content
    /// ids only.
    pub fn initGemma4FromTokenizerJson(gpa: std.mem.Allocator, json_bytes: []const u8) !Tokenizer {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, sa, json_bytes, .{});
        if (parsed != .object) return error.InvalidTokenizerJson;
        const root = parsed.object;

        const model_v = root.get("model") orelse return error.InvalidTokenizerJson;
        if (model_v != .object) return error.InvalidTokenizerJson;
        const model = model_v.object;
        if (model.get("type")) |mt| {
            if (mt == .string and !std.mem.eql(u8, mt.string, "BPE")) return error.UnsupportedTokenizer;
        }
        const vocab_v = model.get("vocab") orelse return error.InvalidTokenizerJson;
        if (vocab_v != .object) return error.InvalidTokenizerJson;
        const vocab = vocab_v.object;
        const merges_v = model.get("merges") orelse return error.InvalidTokenizerJson;
        if (merges_v != .array) return error.InvalidTokenizerJson;
        const merges = merges_v.array.items;

        // added_tokens (optional): control/user tokens matched verbatim.
        var added: []const std.json.Value = &.{};
        if (root.get("added_tokens")) |av| {
            if (av == .array) added = av.array.items;
        }

        // Vocab size = max id + 1 over vocab entries and added tokens.
        var n: usize = 0;
        {
            var vit = vocab.iterator();
            while (vit.next()) |e| {
                if (e.value_ptr.* != .integer) return error.InvalidVocab;
                const id: usize = @intCast(e.value_ptr.integer);
                if (id + 1 > n) n = id + 1;
            }
            for (added) |av| {
                if (av != .object) continue;
                const idv = av.object.get("id") orelse continue;
                if (idv == .integer) {
                    const id: usize = @intCast(idv.integer);
                    if (id + 1 > n) n = id + 1;
                }
            }
        }
        if (n == 0) return error.InvalidVocab;

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const id_to_bytes = try a.alloc([]const u8, n);
        for (id_to_bytes) |*e| e.* = ""; // ids with no vocab/added entry stay empty
        var text_id: std.StringHashMapUnmanaged(u32) = .empty;
        try text_id.ensureTotalCapacity(a, @intCast(n + added.len));

        // Regular vocab: keys are ▁-escaped SentencePiece pieces (stored form).
        {
            var vit = vocab.iterator();
            while (vit.next()) |e| {
                const id: u32 = @intCast(e.value_ptr.integer);
                const raw = try a.dupe(u8, e.key_ptr.*); // escaped form
                try text_id.put(a, raw, id);
                if (parseByteToken(raw)) |b| {
                    id_to_bytes[id] = try a.dupe(u8, &[_]u8{b});
                } else {
                    id_to_bytes[id] = try unescapeSpm(a, raw);
                }
            }
        }

        // Added tokens: literal content (not ▁-escaped), matched verbatim.
        var specials: std.ArrayList(Special) = .empty;
        for (added) |av| {
            if (av != .object) continue;
            const o = av.object;
            const idv = o.get("id") orelse continue;
            const cv = o.get("content") orelse continue;
            if (idv != .integer or cv != .string) continue;
            const id: u32 = @intCast(idv.integer);
            const content = try a.dupe(u8, cv.string);
            id_to_bytes[id] = content;
            try text_id.put(a, content, id);
            try specials.append(a, .{ .text = content, .id = id });
        }
        // Longest-first so a special that is a prefix of another still matches.
        std.mem.sort(Special, specials.items, {}, struct {
            fn lt(_: void, x: Special, y: Special) bool {
                if (x.text.len != y.text.len) return x.text.len > y.text.len;
                return x.id < y.id;
            }
        }.lt);

        const unk: u32 = blk: {
            if (model.get("unk_token")) |uv| {
                if (uv == .string) if (text_id.get(uv.string)) |uid| break :blk uid;
            }
            break :blk 0;
        };

        // Byte-fallback map: "<0xNN>" -> id (else raw single byte, else unk).
        var byte_id: [256]u32 = undefined;
        for (0..256) |b| {
            var buf: [6]u8 = undefined;
            const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{@as(u8, @intCast(b))}) catch unreachable;
            byte_id[b] = text_id.get(hex) orelse text_id.get(&[_]u8{@intCast(b)}) orelse unk;
        }

        // Merge ranks: key `left ++ '\x00' ++ right`; first occurrence wins.
        // HF 0.20+ stores merges as [left, right] pairs; older exports use a
        // single "left right" string.
        var ranks: std.StringHashMapUnmanaged(u32) = .empty;
        try ranks.ensureTotalCapacity(a, @intCast(merges.len));
        for (merges, 0..) |mv, rank| {
            var left: []const u8 = undefined;
            var right: []const u8 = undefined;
            switch (mv) {
                .array => |pair| {
                    if (pair.items.len != 2 or pair.items[0] != .string or pair.items[1] != .string)
                        return error.InvalidMerges;
                    left = pair.items[0].string;
                    right = pair.items[1].string;
                },
                .string => |s| {
                    const pos = std.mem.indexOfScalar(u8, s, ' ') orelse return error.InvalidMerges;
                    left = s[0..pos];
                    right = s[pos + 1 ..];
                },
                else => return error.InvalidMerges,
            }
            const key = try a.alloc(u8, left.len + 1 + right.len);
            @memcpy(key[0..left.len], left);
            key[left.len] = 0;
            @memcpy(key[left.len + 1 ..], right);
            const gop = try ranks.getOrPut(a, key);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(rank);
        }

        var t: Tokenizer = .{
            .arena = arena,
            .id_to_bytes = id_to_bytes,
            .byte_id = @splat(0), // unused for gemma4
            .merges = .empty,
            .specials = try specials.toOwnedSlice(a),
            .kind = .gemma4,
            .spm_text_id = text_id,
            .spm_byte_id = byte_id,
            .spm_unk = unk,
            .gemma4_ranks = ranks,
        };
        // Best-effort template/stop ids; embedding callers add their own frame.
        t.turn_end = findSpecial(t.specials, "<end_of_turn>") orelse
            findSpecial(t.specials, "<eos>") orelse unk;
        t.pad = findSpecial(t.specials, "<pad>") orelse t.turn_end;
        t.newline = text_id.get("\n") orelse 0;
        return t;
    }

    /// Build a SentencePiece **Unigram** tokenizer (kind == .unigram) from a
    /// HuggingFace `tokenizer.json` (model.type == "Unigram"; XLM-RoBERTa / GTE,
    /// e.g. Snowflake Arctic Embed). `model.vocab` is an array of `[piece,
    /// score]` (id = index); `model.unk_id` is the fallback id. Pieces are the
    /// ▁-escaped SentencePiece form.
    ///
    /// Matches the deployed DiffKeep `onnx_tokenizers.zig` behavior exactly: the
    /// `Precompiled` (SentencePiece charsmap / NFKC) normalizer is **not**
    /// applied — DiffKeep's index was built without it, so reproducing its
    /// vectors means tokenizing the same way. `encode` emits content ids only
    /// (whitespace-split → ▁-prefix → Viterbi); the embedding façade adds the
    /// `<s> … </s>` frame.
    pub fn initUnigramFromTokenizerJson(gpa: std.mem.Allocator, json_bytes: []const u8) !Tokenizer {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, sa, json_bytes, .{});
        if (parsed != .object) return error.InvalidTokenizerJson;
        const root = parsed.object;

        const model_v = root.get("model") orelse return error.InvalidTokenizerJson;
        if (model_v != .object) return error.InvalidTokenizerJson;
        const model = model_v.object;
        if (model.get("type")) |mt| {
            if (mt == .string and !std.mem.eql(u8, mt.string, "Unigram")) return error.UnsupportedTokenizer;
        }
        const vocab_v = model.get("vocab") orelse return error.InvalidTokenizerJson;
        if (vocab_v != .array) return error.InvalidTokenizerJson;
        const vocab = vocab_v.array.items;
        if (vocab.len == 0) return error.InvalidVocab;

        var added: []const std.json.Value = &.{};
        if (root.get("added_tokens")) |av| {
            if (av == .array) added = av.array.items;
        }
        var n: usize = vocab.len;
        for (added) |av| {
            if (av != .object) continue;
            if (av.object.get("id")) |idv| if (idv == .integer) {
                const id: usize = @intCast(idv.integer);
                if (id + 1 > n) n = id + 1;
            };
        }

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const id_to_bytes = try a.alloc([]const u8, n);
        for (id_to_bytes) |*e| e.* = "";
        const scores = try a.alloc(f32, n);
        @memset(scores, 0);
        var text_id: std.StringHashMapUnmanaged(u32) = .empty;
        try text_id.ensureTotalCapacity(a, @intCast(n + added.len));

        var max_piece: usize = 0;
        for (vocab, 0..) |entry, id| {
            if (entry != .array or entry.array.items.len != 2) return error.InvalidVocab;
            const pv = entry.array.items;
            if (pv[0] != .string) return error.InvalidVocab;
            const raw = try a.dupe(u8, pv[0].string); // ▁-escaped piece
            try text_id.put(a, raw, @intCast(id));
            scores[id] = switch (pv[1]) {
                .float => |f| @floatCast(f),
                .integer => |iv| @floatFromInt(iv),
                else => return error.InvalidVocab,
            };
            id_to_bytes[id] = try unescapeSpm(a, raw);
            if (raw.len > max_piece) max_piece = raw.len;
        }

        var specials: std.ArrayList(Special) = .empty;
        for (added) |av| {
            if (av != .object) continue;
            const o = av.object;
            const idv = o.get("id") orelse continue;
            const cv = o.get("content") orelse continue;
            if (idv != .integer or cv != .string) continue;
            const id: u32 = @intCast(idv.integer);
            const content = try a.dupe(u8, cv.string);
            id_to_bytes[id] = content;
            try text_id.put(a, content, id);
            try specials.append(a, .{ .text = content, .id = id });
        }
        std.mem.sort(Special, specials.items, {}, struct {
            fn lt(_: void, x: Special, y: Special) bool {
                if (x.text.len != y.text.len) return x.text.len > y.text.len;
                return x.id < y.id;
            }
        }.lt);

        const unk: u32 = if (model.get("unk_id")) |uv| (switch (uv) {
            .integer => @intCast(uv.integer),
            else => 3,
        }) else 3;

        var t: Tokenizer = .{
            .arena = arena,
            .id_to_bytes = id_to_bytes,
            .byte_id = @splat(0), // unused for unigram
            .merges = .empty,
            .specials = try specials.toOwnedSlice(a),
            .kind = .unigram,
            .spm_scores = scores,
            .spm_text_id = text_id,
            .spm_unk = unk,
            .unigram_max_piece = max_piece,
        };
        t.turn_end = findSpecial(t.specials, "</s>") orelse text_id.get("</s>") orelse unk;
        t.pad = findSpecial(t.specials, "<pad>") orelse text_id.get("<pad>") orelse t.turn_end;
        t.newline = text_id.get("\n") orelse 0;
        return t;
    }

    /// Id of a special token by its literal text (e.g. "<|image_pad|>").
    pub fn specialId(self: *const Tokenizer, text: []const u8) ?u32 {
        return findSpecial(self.specials, text);
    }

    fn findSpecial(specials: []const Special, text: []const u8) ?u32 {
        for (specials) |s| {
            if (std.mem.eql(u8, s.text, text)) return s.id;
        }
        return null;
    }

    fn valueInt(v: gguf_mod.Value) i64 {
        return switch (v) {
            .int => |i| i,
            .uint => |u| @intCast(u),
            else => 1, // treat malformed type entries as NORMAL
        };
    }

    fn valueFloat(v: gguf_mod.Value) f64 {
        return switch (v) {
            .float => |f| f,
            .uint => |u| @floatFromInt(u),
            .int => |i| @floatFromInt(i),
            else => 0,
        };
    }

    /// Byte value of a "<0xNN>" SentencePiece byte token, or null.
    fn parseByteToken(s: []const u8) ?u8 {
        if (s.len != 6 or s[0] != '<' or s[1] != '0' or s[2] != 'x' or s[5] != '>') return null;
        return std.fmt.parseInt(u8, s[3..5], 16) catch null;
    }

    /// Decode a SentencePiece piece to raw bytes: ▁ (U+2581) -> space.
    fn unescapeSpm(alloc: std.mem.Allocator, raw: []const u8) ![]const u8 {
        const out = try alloc.alloc(u8, raw.len); // decoded never longer
        var w: usize = 0;
        var i: usize = 0;
        while (i < raw.len) {
            if (i + 3 <= raw.len and raw[i] == 0xE2 and raw[i + 1] == 0x96 and raw[i + 2] == 0x81) {
                out[w] = ' ';
                w += 1;
                i += 3;
            } else {
                out[w] = raw[i];
                w += 1;
                i += 1;
            }
        }
        return out[0..w];
    }

    /// Register one merge rule ("left right", byte-level-unicode form).
    /// Longest key in these vocabs is 128 UTF-8 bytes; decoded never longer.
    fn addMerge(
        alloc: std.mem.Allocator,
        merges: *std.AutoHashMapUnmanaged(u64, Merge),
        raw_to_id: *const std.StringHashMapUnmanaged(u32),
        line: []const u8,
        rank: u32,
    ) !void {
        var pair_buf: [256]u8 = undefined; // left ++ right, concatenated
        var right_buf: [128]u8 = undefined;
        const space = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidMerges;
        const left_n = try decodeTokenKey(line[0..space], pair_buf[0..128]);
        const right_n = try decodeTokenKey(line[space + 1 ..], &right_buf);
        const left = raw_to_id.get(pair_buf[0..left_n]) orelse return error.InvalidMerges;
        const right = raw_to_id.get(right_buf[0..right_n]) orelse return error.InvalidMerges;
        @memcpy(pair_buf[left_n .. left_n + right_n], right_buf[0..right_n]);
        const merged = raw_to_id.get(pair_buf[0 .. left_n + right_n]) orelse return error.InvalidMerges;
        try merges.put(alloc, pairKey(left, right), .{ .rank = rank, .merged = merged });
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
        if (self.kind == .spm) return self.encodeSpm(gpa, text, out);
        if (self.kind == .gemma4) return self.encodeGemma4(gpa, text, out);
        if (self.kind == .unigram) return self.encodeUnigram(gpa, text, out);
        var i: usize = 0;
        while (i < text.len) {
            // Earliest special-token occurrence from i.
            var next_special: ?struct { pos: usize, idx: usize } = null;
            for (self.specials, 0..) |st, idx| {
                if (std.mem.indexOfPos(u8, text, i, st.text)) |pos| {
                    if (next_special == null or pos < next_special.?.pos or
                        (pos == next_special.?.pos and st.text.len > self.specials[next_special.?.idx].text.len))
                    {
                        next_special = .{ .pos = pos, .idx = idx };
                    }
                }
            }
            const seg_end = if (next_special) |ns| ns.pos else text.len;
            if (i < seg_end) try self.encodeSegment(gpa, text[i..seg_end], out);
            if (next_special) |ns| {
                try out.append(gpa, self.specials[ns.idx].id);
                i = ns.pos + self.specials[ns.idx].text.len;
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
            const end = pretokenEnd(self.pretok, cps, i);
            try self.bpe(gpa, text[offs[i]..offs[end]], &symbols, out);
            i = end;
        }
    }

    /// End index (exclusive, in codepoints) of the pretoken starting at `i`.
    /// Hand-rolled match of the Qwen2 pretokenizer regex alternation:
    ///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
    ///   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    /// The qwen35 variant widens letter runs to [\p{L}\p{M}]+ and excludes
    /// \p{M} from the punctuation class.
    fn pretokenEnd(pretok: Pretok, cps: []const u21, i: usize) usize {
        if (pretok == .tekken) return pretokenEndTekken(cps, i);
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

        const wordish = struct {
            fn f(p: Pretok, cp: u21) bool {
                return isLetter(cp) or (p == .qwen35 and isMark(cp));
            }
        }.f;

        // 2: optional non-letter/number/CRLF prefix + letter(+mark) run.
        if (wordish(pretok, c0)) {
            var j = i + 1;
            while (j < n and wordish(pretok, cps[j])) j += 1;
            return j;
        }
        if (c0 != '\r' and c0 != '\n' and !isNumber(c0) and i + 1 < n and wordish(pretok, cps[i + 1])) {
            var j = i + 2;
            while (j < n and wordish(pretok, cps[j])) j += 1;
            return j;
        }

        // 3: single number char.
        if (isNumber(c0)) return i + 1;

        // 4: optional space + punctuation run + trailing newlines.
        {
            var j = i;
            if (cps[j] == ' ') j += 1;
            var k = j;
            while (k < n and !isWs(cps[k]) and !isLetter(cps[k]) and !isNumber(cps[k]) and
                !(pretok == .qwen35 and isMark(cps[k]))) k += 1;
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

    /// End index (exclusive) of the tekken (Mistral) pretoken starting at `i`.
    /// Hand-rolled match of the tekken regex alternation (tried in order):
    ///   [^\r\n\p{L}\p{N}]?([\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
    ///     |[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*)
    ///   | \p{N} |  ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    /// Differs from qwen2: no contractions, letter runs split on case, digits
    /// are single, and the punctuation tail allows '/' (not just \r\n).
    fn pretokenEndTekken(cps: []const u21, i: usize) usize {
        const n = cps.len;
        const c0 = cps[i];

        // 1: optional non-letter/number/CRLF prefix + a case-split letter run.
        // The letter run is `U* L+ | U+ L*` (U/L = the tekken upper/lower
        // classes); `caseRun` returns the end of exactly one such segment.
        const isLM = struct {
            fn f(cp: u21) bool {
                return isLetter(cp) or isMark(cp);
            }
        }.f;
        if (isLM(c0)) return caseRun(cps, i);
        // prefix char: not CRLF, not letter, not number (space/punct qualify).
        if (c0 != '\r' and c0 != '\n' and !isLetter(c0) and !isNumber(c0) and
            i + 1 < n and isLM(cps[i + 1]))
            return caseRun(cps, i + 1);

        // 2: single number char.
        if (isNumber(c0)) return i + 1;

        // 3: optional space + punctuation run + trailing [\r\n/] run.
        {
            var j = i;
            if (cps[j] == ' ') j += 1;
            var k = j;
            while (k < n and !isWs(cps[k]) and !isLetter(cps[k]) and !isNumber(cps[k])) k += 1;
            if (k > j) {
                while (k < n and (cps[k] == '\r' or cps[k] == '\n' or cps[k] == '/')) k += 1;
                return k;
            }
        }

        // 4-6: whitespace runs (identical to the qwen2 tail).
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
        return i + 1; // never loop forever
    }

    /// End (exclusive) of one tekken case-split letter segment starting at `s`
    /// (`s` is a letter or mark). Consumes the maximal upper-class prefix, then
    /// the following lower-class run — i.e. `U* L+`, falling back to `U+ L*`
    /// when no lower-class char follows (an all-uppercase run). Ambiguous
    /// letters (Lm/Lo/M) are in both classes and greedily join the upper prefix.
    fn caseRun(cps: []const u21, s: usize) usize {
        const n = cps.len;
        var hi = s;
        while (hi < n and isTekHi(cps[hi])) hi += 1; // greedy U*
        // U* L+ : the char after the upper prefix (if any) is a pure-lowercase
        // letter, so it opens the lower run; if instead the upper prefix ran to
        // the segment boundary, this is the all-upper `U+ L*` case (lo == hi).
        var lo = hi;
        while (lo < n and isTekLo(cps[lo])) lo += 1;
        return if (lo > hi) lo else hi;
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

    // --- SentencePiece (SPM) encode path ---------------------------------

    /// SPM encode: partition on special tokens (verbatim, longest-first),
    /// then run the score-ranked merge over each ▁-escaped raw fragment.
    fn encodeSpm(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        const Frag = union(enum) { raw: []const u8, tok: u32 };
        var frags: std.ArrayList(Frag) = .empty;
        defer frags.deinit(gpa);
        var next: std.ArrayList(Frag) = .empty;
        defer next.deinit(gpa);
        try frags.append(gpa, .{ .raw = text });

        for (self.specials) |sp| {
            if (sp.text.len == 0) continue;
            next.clearRetainingCapacity();
            for (frags.items) |f| switch (f) {
                .tok => try next.append(gpa, f),
                .raw => |r| {
                    var start: usize = 0;
                    while (std.mem.indexOfPos(u8, r, start, sp.text)) |pos| {
                        if (pos > start) try next.append(gpa, .{ .raw = r[start..pos] });
                        try next.append(gpa, .{ .tok = sp.id });
                        start = pos + sp.text.len;
                    }
                    if (start < r.len) try next.append(gpa, .{ .raw = r[start..] });
                },
            };
            std.mem.swap(std.ArrayList(Frag), &frags, &next);
        }

        var is_prev_special = true;
        for (frags.items) |f| switch (f) {
            .tok => |t| {
                try out.append(gpa, t);
                is_prev_special = true;
            },
            .raw => |r| {
                try self.spmEncodeRaw(gpa, r, is_prev_special, out);
                is_prev_special = false;
            },
        };
    }

    /// Escape a raw fragment (spaces -> ▁, optional add_dummy_prefix) and
    /// run the SPM merge.
    fn spmEncodeRaw(self: *const Tokenizer, gpa: std.mem.Allocator, r: []const u8, is_prev_special: bool, out: *std.ArrayList(u32)) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        if (self.add_space_prefix and is_prev_special) try buf.appendSlice(gpa, spm_space);
        for (r) |c| {
            if (c == ' ') try buf.appendSlice(gpa, spm_space) else try buf.append(gpa, c);
        }
        if (buf.items.len == 0) return;
        try self.spmMerge(gpa, buf.items, out);
    }

    fn tryAddBigram(self: *const Tokenizer, gpa: std.mem.Allocator, syms: []const SpmSymbol, pq: *SpmQueue, left: i32, right: i32) !void {
        if (left == -1 or right == -1) return;
        const l = syms[@intCast(left)];
        const r = syms[@intCast(right)];
        // Symbols are consecutive slices of the escaped buffer; l is
        // immediately followed by r, so their combined text is contiguous.
        const combined = l.text.ptr[0 .. l.text.len + r.text.len];
        const id = self.spm_text_id.get(combined) orelse return;
        try pq.push(gpa, .{ .left = left, .right = right, .score = self.spm_scores[id], .size = combined.len });
    }

    /// Bigram-merge `text` (already ▁-escaped) into token ids, appending to
    /// `out`. Mirrors llama.cpp's llm_tokenizer_spm.
    fn spmMerge(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        var syms: std.ArrayList(SpmSymbol) = .empty;
        defer syms.deinit(gpa);
        {
            var offs: usize = 0;
            var idx: i32 = 0;
            while (offs < text.len) {
                const len = std.unicode.utf8ByteSequenceLength(text[offs]) catch 1;
                const n = @min(@as(usize, len), text.len - offs);
                try syms.append(gpa, .{
                    .text = text[offs .. offs + n],
                    .prev = idx - 1,
                    .next = if (offs + n == text.len) -1 else idx + 1,
                });
                offs += n;
                idx += 1;
            }
        }
        if (syms.items.len == 0) return;

        var pq: SpmQueue = .empty;
        defer pq.deinit(gpa);
        for (1..syms.items.len) |i| try self.tryAddBigram(gpa, syms.items, &pq, @intCast(i - 1), @intCast(i));

        while (pq.pop()) |bg| {
            const l = &syms.items[@intCast(bg.left)];
            const r = &syms.items[@intCast(bg.right)];
            if (l.text.len == 0 or r.text.len == 0 or l.text.len + r.text.len != bg.size) continue;
            l.text = l.text.ptr[0 .. l.text.len + r.text.len];
            r.text = r.text[0..0];
            l.next = r.next;
            if (r.next >= 0) syms.items[@intCast(r.next)].prev = bg.left;
            try self.tryAddBigram(gpa, syms.items, &pq, l.prev, bg.left);
            try self.tryAddBigram(gpa, syms.items, &pq, bg.left, l.next);
        }

        var i: i32 = 0;
        while (i != -1) : (i = syms.items[@intCast(i)].next) {
            const sym = syms.items[@intCast(i)];
            if (self.spm_text_id.get(sym.text)) |id| {
                try out.append(gpa, id);
            } else {
                for (sym.text) |b| try out.append(gpa, self.spm_byte_id[b]);
            }
        }
    }

    // --- Gemma 4 ("SPM-style BPE") encode path ---------------------------

    /// Gemma 4 encode: partition on specials (verbatim, longest-first), then
    /// per raw fragment escape spaces to ▁ (no dummy prefix), split into
    /// newline vs non-newline runs, and rank-BPE each run. Mirrors llama.cpp's
    /// llm_tokenizer_bpe_session with pre-type GEMMA4.
    fn encodeGemma4(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        const Frag = union(enum) { raw: []const u8, tok: u32 };
        var frags: std.ArrayList(Frag) = .empty;
        defer frags.deinit(gpa);
        var next: std.ArrayList(Frag) = .empty;
        defer next.deinit(gpa);
        try frags.append(gpa, .{ .raw = text });

        for (self.specials) |sp| {
            if (sp.text.len == 0) continue;
            next.clearRetainingCapacity();
            for (frags.items) |f| switch (f) {
                .tok => try next.append(gpa, f),
                .raw => |r| {
                    var start: usize = 0;
                    while (std.mem.indexOfPos(u8, r, start, sp.text)) |pos| {
                        if (pos > start) try next.append(gpa, .{ .raw = r[start..pos] });
                        try next.append(gpa, .{ .tok = sp.id });
                        start = pos + sp.text.len;
                    }
                    if (start < r.len) try next.append(gpa, .{ .raw = r[start..] });
                },
            };
            std.mem.swap(std.ArrayList(Frag), &frags, &next);
        }

        for (frags.items) |f| switch (f) {
            .tok => |t| try out.append(gpa, t),
            .raw => |r| try self.gemma4EncodeRaw(gpa, r, out),
        };
    }

    /// Escape a raw fragment (spaces -> ▁, no dummy prefix), split into
    /// maximal newline / non-newline runs, and rank-BPE each run.
    fn gemma4EncodeRaw(self: *const Tokenizer, gpa: std.mem.Allocator, r: []const u8, out: *std.ArrayList(u32)) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        for (r) |c| {
            if (c == ' ') try buf.appendSlice(gpa, spm_space) else try buf.append(gpa, c);
        }
        const b = buf.items;
        if (b.len == 0) return;
        // Scratch for bigram keys: `left ++ '\x00' ++ right` never exceeds the
        // word length + 1.
        const scratch = try gpa.alloc(u8, b.len + 1);
        defer gpa.free(scratch);
        var i: usize = 0;
        while (i < b.len) {
            const is_nl = b[i] == '\n';
            var j = i + 1;
            while (j < b.len and (b[j] == '\n') == is_nl) j += 1;
            try self.gemma4Word(gpa, b[i..j], is_nl, scratch, out);
            i = j;
        }
    }

    /// Rank-BPE one word (a newline or non-newline run) into ids, with
    /// `<0xNN>` byte fallback. `all_newline` enables llama.cpp's gemma4 fix:
    /// a newline run that is itself a token is emitted whole (never split).
    fn gemma4Word(self: *const Tokenizer, gpa: std.mem.Allocator, word: []const u8, all_newline: bool, scratch: []u8, out: *std.ArrayList(u32)) !void {
        if (all_newline) {
            if (self.spm_text_id.get(word)) |wid| {
                try out.append(gpa, wid);
                return;
            }
        }

        var syms: std.ArrayList(SpmSymbol) = .empty;
        defer syms.deinit(gpa);
        {
            var offs: usize = 0;
            var idx: i32 = 0;
            while (offs < word.len) {
                const len = std.unicode.utf8ByteSequenceLength(word[offs]) catch 1;
                const n = @min(@as(usize, len), word.len - offs);
                try syms.append(gpa, .{
                    .text = word[offs .. offs + n],
                    .prev = idx - 1,
                    .next = if (offs + n == word.len) -1 else idx + 1,
                });
                offs += n;
                idx += 1;
            }
        }
        if (syms.items.len == 0) return;

        var pq: Gemma4Queue = .empty;
        defer pq.deinit(gpa);
        for (1..syms.items.len) |i| try self.gemma4AddBigram(gpa, syms.items, &pq, @intCast(i - 1), @intCast(i), scratch);

        while (pq.pop()) |bg| {
            const l = &syms.items[@intCast(bg.left)];
            const rr = &syms.items[@intCast(bg.right)];
            if (l.text.len == 0 or rr.text.len == 0 or l.text.len + rr.text.len != bg.size) continue;
            l.text = l.text.ptr[0 .. l.text.len + rr.text.len];
            rr.text = rr.text[0..0];
            l.next = rr.next;
            if (rr.next >= 0) syms.items[@intCast(rr.next)].prev = bg.left;
            try self.gemma4AddBigram(gpa, syms.items, &pq, l.prev, bg.left, scratch);
            try self.gemma4AddBigram(gpa, syms.items, &pq, bg.left, l.next, scratch);
        }

        var i: i32 = 0;
        while (i != -1) : (i = syms.items[@intCast(i)].next) {
            const sym = syms.items[@intCast(i)];
            if (sym.text.len == 0) continue;
            if (self.spm_text_id.get(sym.text)) |wid| {
                try out.append(gpa, wid);
            } else {
                for (sym.text) |bb| try out.append(gpa, self.spm_byte_id[bb]);
            }
        }
    }

    fn gemma4AddBigram(self: *const Tokenizer, gpa: std.mem.Allocator, syms: []const SpmSymbol, pq: *Gemma4Queue, left: i32, right: i32, scratch: []u8) !void {
        if (left == -1 or right == -1) return;
        const l = syms[@intCast(left)].text;
        const rt = syms[@intCast(right)].text;
        if (l.len == 0 or rt.len == 0) return;
        @memcpy(scratch[0..l.len], l);
        scratch[l.len] = 0;
        @memcpy(scratch[l.len + 1 ..][0..rt.len], rt);
        const rank = self.gemma4_ranks.get(scratch[0 .. l.len + 1 + rt.len]) orelse return;
        try pq.push(gpa, .{ .left = left, .right = right, .rank = rank, .size = l.len + rt.len });
    }

    // --- Unigram (SentencePiece) encode path ----------------------------

    /// Unigram encode: split on ASCII whitespace, prepend ▁ to each word, and
    /// Viterbi-segment each word into ids. Mirrors DiffKeep's `onnx_tokenizers`
    /// (no Precompiled/NFKC normalization). Content ids only — no `<s>`/`</s>`.
    fn encodeUnigram(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        var word: std.ArrayList(u8) = .empty;
        defer word.deinit(gpa);
        var it = std.mem.splitAny(u8, text, " \t\n\r");
        while (it.next()) |w| {
            if (w.len == 0) continue;
            word.clearRetainingCapacity();
            try word.appendSlice(gpa, spm_space);
            try word.appendSlice(gpa, w);
            try self.viterbiWord(gpa, word.items, out);
        }
    }

    /// Viterbi over `word` (a ▁-prefixed word): maximize the summed piece
    /// log-scores, emitting piece ids. If no full segmentation exists (an
    /// unknown byte, no byte fallback), the whole word becomes one `<unk>`.
    fn viterbiWord(self: *const Tokenizer, gpa: std.mem.Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        const n = word.len;
        const neg_inf = -std.math.inf(f64);
        const dp = try gpa.alloc(f64, n + 1);
        defer gpa.free(dp);
        const prev_start = try gpa.alloc(usize, n + 1);
        defer gpa.free(prev_start);
        const prev_id = try gpa.alloc(u32, n + 1);
        defer gpa.free(prev_id);

        @memset(dp, neg_inf);
        dp[0] = 0;
        for (1..n + 1) |i| {
            const max_back = @min(i, self.unigram_max_piece);
            var l: usize = 1;
            while (l <= max_back) : (l += 1) {
                const j = i - l;
                if (dp[j] == neg_inf) continue;
                if (self.spm_text_id.get(word[j..i])) |id| {
                    const score = dp[j] + @as(f64, self.spm_scores[id]);
                    if (score > dp[i]) {
                        dp[i] = score;
                        prev_start[i] = j;
                        prev_id[i] = id;
                    }
                }
            }
        }

        if (dp[n] == neg_inf) {
            try out.append(gpa, self.spm_unk);
            return;
        }
        // Backtrack, then append the pieces in forward order.
        const start = out.items.len;
        var pos = n;
        while (pos > 0) {
            try out.append(gpa, prev_id[pos]);
            pos = prev_start[pos];
        }
        std.mem.reverse(u32, out.items[start..]);
    }

    /// Decode ids back to text (debugging / tests).
    pub fn decodeAlloc(self: *const Tokenizer, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        outer: for (ids) |id| {
            if (id < self.id_to_bytes.len) {
                try buf.appendSlice(gpa, self.id_to_bytes[id]);
            } else {
                for (self.specials) |st| {
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

/// One symbol in the SPM merge: a slice of the escaped buffer plus
/// doubly-linked neighbours (index into the symbol list, -1 = none).
const SpmSymbol = struct { text: []const u8, prev: i32, next: i32 };

/// A candidate SPM merge of two adjacent symbols.
const SpmBigram = struct { left: i32, right: i32, score: f32, size: usize };

/// Highest score first; ties broken by smaller left index (llama.cpp
/// llm_bigram_spm::comparator). PriorityQueue pops the `.lt`-most element.
fn spmBigramOrder(_: void, a: SpmBigram, b: SpmBigram) std.math.Order {
    if (a.score != b.score) return if (a.score > b.score) .lt else .gt;
    if (a.left != b.left) return if (a.left < b.left) .lt else .gt;
    return .eq;
}

const SpmQueue = std.PriorityQueue(SpmBigram, void, spmBigramOrder);

/// A candidate Gemma 4 rank-BPE merge of two adjacent symbols.
const Gemma4Bigram = struct { left: i32, right: i32, rank: u32, size: usize };

/// Lowest rank first; ties broken by smaller left index (llama.cpp
/// llm_bigram_bpe::comparator). PriorityQueue pops the `.lt`-most element.
fn gemma4BigramOrder(_: void, a: Gemma4Bigram, b: Gemma4Bigram) std.math.Order {
    if (a.rank != b.rank) return if (a.rank < b.rank) .lt else .gt;
    if (a.left != b.left) return if (a.left < b.left) .lt else .gt;
    return .eq;
}

const Gemma4Queue = std.PriorityQueue(Gemma4Bigram, void, gemma4BigramOrder);

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

// The Qwen3-4B GGUF ships the same vocab/merges as the embedded tokenizer,
// so a tokenizer built from its kv arrays must encode identically; skipped
// when the checkpoint is absent.
test "gguf tokenizer matches embedded tokenizer" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var gt = try Tokenizer.initFromGguf(gpa, &g);
    defer gt.deinit();
    var et = try Tokenizer.init(gpa);
    defer et.deinit();

    try std.testing.expectEqual(im_end, gt.turn_end);
    try std.testing.expectEqual(@as(u32, 151654), gt.pad); // <|vision_pad|>, per the gguf kv
    try std.testing.expectEqual(@as(u32, 198), gt.newline);
    try std.testing.expectEqual(@as(usize, 151936), gt.id_to_bytes.len);

    const samples = [_][]const u8{
        "a photo of a cat",
        "Hello, World! 123 café ☕ 你好",
        "don't we'll I'M IT'S",
        "a  b\n\nc\t d ",
        "  leading and trailing  ",
        "ééé 😀😀",
        "<|im_start|>system\nDescribe the image:<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n",
        "<think>reasoning</think> answer",
    };
    var a: std.ArrayList(u32) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u32) = .empty;
    defer b.deinit(gpa);
    for (samples) |text| {
        a.clearRetainingCapacity();
        b.clearRetainingCapacity();
        try et.encode(gpa, text, &a);
        try gt.encode(gpa, text, &b);
        try std.testing.expectEqualSlices(u32, a.items, b.items);
        const round = try gt.decodeAlloc(gpa, b.items);
        defer gpa.free(round);
        try std.testing.expectEqualStrings(text, round);
    }
}

// Golden ids from llama.cpp's llama-tokenize on the same file (the qwen35
// pretokenizer: 248k vocab, combining-mark handling); skipped when absent.
test "qwen3.6 gguf tokenizer matches llama-tokenize" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Q5_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();

    try std.testing.expectEqual(Pretok.qwen35, tok.pretok);
    try std.testing.expectEqual(@as(u32, 248046), tok.turn_end); // <|im_end|> == eos
    try std.testing.expectEqual(@as(u32, 248044), tok.pad);
    try std.testing.expectEqual(@as(usize, 248320), tok.id_to_bytes.len);

    try expectEncode(&tok, "a photo of a cat", &.{ 64, 6345, 314, 264, 7993 });
    try expectEncode(&tok, "Hello, World! 123 café ☕ 你好", &.{ 9419, 11, 4196, 0, 220, 16, 17, 18, 50203, 24329, 243, 220, 109266 });
    try expectEncode(&tok, "don't we'll I'M IT'S", &.{ 14572, 914, 567, 3172, 353, 26708, 8435, 12887 });
    try expectEncode(&tok, "cafe\u{0301} nai\u{0308}ve", &.{ 895, 1795, 52033, 238883, 136, 230, 571 });
    try expectEncode(&tok, "<|im_start|>user\nhi there<|im_end|>\n", &.{ 248045, 846, 198, 5834, 1017, 248046, 198 });
    try expectEncode(&tok, "  leading and trailing  ", &.{ 220, 6187, 321, 26849, 256 });
    try expectEncode(&tok, "नमस्ते dost", &.{ 58069, 84237, 150104, 153348, 46704 });
    try expectEncode(&tok, "x… —dash", &.{ 87, 1873, 1892, 42080 });
}

// Golden ids from llama.cpp's `llama-tokenize --no-bos` on the same file
// (gpt2 byte-level BPE, tokenizer.ggml.pre == "tekken"; Mistral-Nemo). Skipped
// when absent. Exercises the tekken-specific pretokenizer: case-split letter
// runs (iOS -> " i","OS"; camelCase -> "camel","Case"), single-digit numbers
// (42.50 -> 4,2,.,5,0), no contractions ("don't" is not glued), and the '/'
// punctuation tail ("/path").
test "tekken gguf tokenizer matches llama-tokenize" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/Impish_Bloodmoon_12B-ARM_HA_NL.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();

    try std.testing.expectEqual(Pretok.tekken, tok.pretok);

    try expectEncode(&tok, "Hello World! The iOS API costs $42.50, right? café_test/path", &.{
        22177, 5325, 1033, 1531, 1623, 6964, 10523, 12889, 1659, 1052, 1050, 1046, 1053, 1048, 1044, 3169, 1063, 35858, 13683, 109366,
    });
    try expectEncode(&tok, "don't we'll I'M IT'S", &.{ 21797, 2405, 1729, 7534, 1362, 1039, 1077, 22784, 44161 });
    try expectEncode(&tok, "camelCase HTTPRequest snake_case", &.{ 32587, 1299, 11139, 23733, 4967, 48726, 69982 });
    try expectEncode(&tok, "  leading and trailing  ", &.{ 1032, 8924, 1321, 49875, 1256 });
}

// Golden ids from llama.cpp's llama-tokenize --no-bos on the same file
// (SentencePiece / tokenizer.ggml.model == "llama"); skipped when absent.
test "gemma3 gguf spm tokenizer matches llama-tokenize" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();

    try std.testing.expectEqual(Kind.spm, tok.kind);
    try std.testing.expectEqual(@as(u32, 106), tok.turn_end); // <end_of_turn>
    try std.testing.expectEqual(@as(u32, 0), tok.pad); // <pad>
    try std.testing.expectEqual(@as(usize, 262145), tok.id_to_bytes.len);
    try std.testing.expect(!tok.add_space_prefix);

    try expectEncode(&tok, "a photo of a cat", &.{ 236746, 4429, 529, 496, 5866 });
    try expectEncode(&tok, "Hello, World! 123 café ☕ 你好", &.{ 9259, 236764, 4109, 236888, 236743, 236770, 236778, 236800, 33443, 236743, 244360, 43758, 237389 });
    try expectEncode(&tok, "don't we'll", &.{ 13246, 236789, 236745, 692, 236789, 859 });
    try expectEncode(&tok, "  leading and trailing  ", &.{ 138, 26016, 532, 45330, 138 });
    try expectEncode(&tok, "The quick brown fox.", &.{ 818, 3823, 8864, 37423, 236761 });

    // Full Gemma chat template tokenizes bit-identically to llama-tokenize
    // (golden ids minus the leading BOS the CLI adds).
    try expectEncode(
        &tok,
        "<start_of_turn>user\nGive me three tips for staying focused while working.<end_of_turn>\n<start_of_turn>model\n",
        &.{ 105, 2364, 107, 46762, 786, 1806, 11221, 573, 19447, 10317, 1651, 2844, 236761, 106, 107, 105, 4368, 107 },
    );

    // Control tokens matched verbatim (parse_special), and decode round-trips.
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "<start_of_turn>user\nhi<end_of_turn>\n", &ids);
    try std.testing.expectEqual(@as(u32, 105), ids.items[0]); // <start_of_turn>
    const round = try tok.decodeAlloc(gpa, ids.items);
    defer gpa.free(round);
    try std.testing.expectEqualStrings("<start_of_turn>user\nhi<end_of_turn>\n", round);
}

// Golden ids from llama.cpp's llama-tokenize --no-bos on the gemma4 12B GGUF
// (tokenizer.ggml.model == "gemma4": SPM-style rank BPE); skipped when absent.
test "gemma4 gguf tokenizer matches llama-tokenize" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();

    try std.testing.expectEqual(Kind.gemma4, tok.kind);
    try std.testing.expectEqual(@as(u32, 106), tok.turn_end); // <turn|>
    try std.testing.expectEqual(@as(u32, 107), tok.newline); // "\n"
    try std.testing.expectEqual(@as(usize, 262144), tok.id_to_bytes.len);

    try expectEncode(&tok, "a photo of a cat", &.{ 236746, 4429, 529, 496, 5866 });
    try expectEncode(&tok, "Hello, World! 123 café ☕ 你好", &.{ 9259, 236764, 4109, 236888, 236743, 236770, 236778, 236800, 33443, 236743, 244360, 43758, 237389 });
    try expectEncode(&tok, "  leading and trailing  ", &.{ 138, 26016, 532, 45330, 138 });
    try expectEncode(&tok, "The quick brown fox.", &.{ 818, 3823, 8864, 37423, 236761 });
    try expectEncode(&tok, "don't we'll", &.{ 13246, 236789, 236745, 692, 236789, 859 });
    try expectEncode(&tok, "a  b\n\nc\td ", &.{ 236746, 138, 236763, 108, 236755, 255968, 236753, 236743 });
    try expectEncode(&tok, "café naïve", &.{ 123125, 236859, 120362 });
    try expectEncode(&tok, "नमस्ते dost", &.{ 226767, 24873 });
    try expectEncode(&tok, "x… —dash", &.{ 236781, 237064, 2192, 56057 });

    // Turn markers matched verbatim (<|turn> = 105, <turn|> = 106, \n = 107).
    try expectEncode(&tok, "<|turn>user\nhi there<turn|>\n", &.{ 105, 2364, 107, 2202, 993, 106, 107 });

    // Decode round-trips (byte-concat of stored/unescaped forms).
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "The quick brown fox.", &ids);
    const round = try tok.decodeAlloc(gpa, ids.items);
    defer gpa.free(round);
    try std.testing.expectEqualStrings("The quick brown fox.", round);
}

// tokenizer.json (HuggingFace) loader parity: build the gemma4 BPE path from
// each model's tokenizer.json and require exact token-id match against golden
// ids produced by HF `tokenizers` (testdata/embed_tokenizer_golden.json,
// `ids_no_special` — encode emits content ids only). Covers EmbeddingGemma
// (262144 vocab) and SigLIP2's text tower (256000 vocab). Skipped when the
// DiffKeep model checkpoints are absent.
test "gemma4 tokenizer.json matches HF tokenizers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const golden_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/embed_tokenizer_golden.json", gpa, .limited(1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(golden_bytes);

    var golden = try std.json.parseFromSlice(std.json.Value, gpa, golden_bytes, .{});
    defer golden.deinit();
    const groot = golden.value.object;

    const Model = struct { name: []const u8, path: []const u8 };
    const models = [_]Model{
        .{ .name = "embeddinggemma", .path = "../DiffKeep/Models/embeddinggemma-300m/tokenizer.json" },
        .{ .name = "siglip2_text", .path = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm/tokenizer.json" },
    };

    var ran = false;
    for (models) |m| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, m.path, gpa, .limited(128 * 1024 * 1024)) catch continue;
        defer gpa.free(bytes);
        ran = true;

        var tok = try Tokenizer.initGemma4FromTokenizerJson(gpa, bytes);
        defer tok.deinit();

        const cases = groot.get(m.name).?.object.get("cases").?.array.items;
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        for (cases) |cv| {
            const co = cv.object;
            const text = co.get("text").?.string;
            const want = co.get("ids_no_special").?.array.items;
            out.clearRetainingCapacity();
            try tok.encode(gpa, text, &out);
            errdefer {
                std.debug.print("MISMATCH [{s}] text=\"{s}\"\n  got : {any}\n  want:", .{ m.name, text, out.items });
                for (want) |w| std.debug.print(" {d}", .{w.integer});
                std.debug.print("\n", .{});
            }
            try std.testing.expectEqual(want.len, out.items.len);
            for (want, out.items) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
        }
    }
    if (!ran) return error.SkipZigTest;
}

// Unigram tokenizer.json parity: build the Snowflake Arctic Embed Unigram
// tokenizer and require exact token-id match vs HF `tokenizers` golden
// (`ids_no_special`; encode emits content ids, no <s>/</s>). Skipped when the
// DiffKeep model checkpoint is absent.
test "unigram tokenizer.json matches HF tokenizers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const golden_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/embed_tokenizer_golden.json", gpa, .limited(1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(golden_bytes);
    var golden = try std.json.parseFromSlice(std.json.Value, gpa, golden_bytes, .{});
    defer golden.deinit();

    const path = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0/tokenizer.json";
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(bytes);
    var tok = try Tokenizer.initUnigramFromTokenizerJson(gpa, bytes);
    defer tok.deinit();
    try std.testing.expectEqual(Kind.unigram, tok.kind);

    const cases = golden.value.object.get("snowflake").?.object.get("cases").?.array.items;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    for (cases) |cv| {
        const co = cv.object;
        const text = co.get("text").?.string;
        const want = co.get("ids_no_special").?.array.items;
        out.clearRetainingCapacity();
        try tok.encode(gpa, text, &out);
        errdefer {
            std.debug.print("MISMATCH text=\"{s}\"\n  got :{any}\n  want:", .{ text, out.items });
            for (want) |w| std.debug.print(" {d}", .{w.integer});
            std.debug.print("\n", .{});
        }
        try std.testing.expectEqual(want.len, out.items.len);
        for (want, out.items) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
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
