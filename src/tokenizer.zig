//! Qwen2 byte-level BPE tokenizer (pure Zig port of transformers'
//! `Qwen2Tokenizer`, which ComfyUI uses for the Krea 2 text encoder).
//!
//! Pipeline: split out special tokens -> pretokenize (hand-rolled scanner for
//! the fixed Qwen2 regex, using generated Unicode tables) -> byte-level BPE
//! over vocab.json/merges.txt. The vocab and merges are embedded in the
//! binary from src/assets/qwen_tokenizer/.

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

/// Pretokenizer regex variant (tokenizer.ggml.pre). qwen35 differs from
/// qwen2 in two places: letter runs are [\p{L}\p{M}]+ (combining marks join
/// the run) and the punctuation class also excludes \p{M}.
pub const Pretok = enum { qwen2, qwen35 };

/// Tokenizer algorithm. `bpe` is GPT-2 byte-level BPE (Qwen family);
/// `spm` is SentencePiece (tokenizer.ggml.model == "llama"; Gemma, Llama).
pub const Kind = enum { bpe, spm };

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
    pretok: Pretok = .qwen2,
    kind: Kind = .bpe,
    /// SentencePiece data (kind == .spm; empty for BPE): per-id merge score,
    /// stored-text -> id (for merge lookups / text_to_token), byte-fallback
    /// map (byte -> "<0xNN>" token id), and the unknown-token id.
    spm_scores: []const f32 = &.{},
    spm_text_id: std.StringHashMapUnmanaged(u32) = .empty,
    spm_byte_id: [256]u32 = @splat(0),
    spm_unk: u32 = 0,
    /// SentencePiece add_dummy_prefix: prepend ▁ at the start of a raw
    /// fragment following a special token. False for Gemma.
    add_space_prefix: bool = false,

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
        if (!std.mem.eql(u8, model, "gpt2")) return error.UnsupportedTokenizer;
        var pretok: Pretok = .qwen2;
        if (g.getStr("tokenizer.ggml.pre")) |pre| {
            if (std.mem.eql(u8, pre, "qwen35")) {
                pretok = .qwen35;
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
        t.pad = if (g.getUint("tokenizer.ggml.padding_token_id")) |p| @intCast(p) else eos orelse t.turn_end;
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
        t.pad = if (g.getUint("tokenizer.ggml.padding_token_id")) |p| @intCast(p) else eos orelse t.turn_end;
        t.newline = findSpecial(t.specials, "\n") orelse 0;
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
