//! Qwen3 ChatML templating for tp-llm.
//!
//! Turns are appended to a growing token-id list so multi-turn chat is just
//! more appends:
//!   <|im_start|>role\n ... <|im_end|>\n
//! Generation starts from an open assistant turn and stops at <|im_end|> (or
//! <|endoftext|>). Qwen3-VL-Instruct is a non-thinking model: no <think>
//! block, and no implicit default system prompt.
//!
//! User text is tokenized with the same special-token scanner as everything
//! else, so a prompt containing e.g. "<|im_end|>" maps to the special id —
//! matching transformers' behavior with special tokens left unescaped.

const std = @import("std");
const tokenizer_mod = @import("../tokenizer.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

/// "\n" as a single token, used by the template glue.
pub const newline_id: u32 = 198;

/// Vocab-dependent template/stop ids. Defaults match the embedded Qwen3
/// tokenizer; a GGUF-embedded tokenizer overrides them via applyTokenizer
/// (process-global, like the tokenizer itself in tp-llm).
pub var turn_end: u32 = tokenizer_mod.im_end;
pub var pad: u32 = tokenizer_mod.pad_token;
pub var newline: u32 = newline_id;

/// Chat template family. `chatml` is Qwen's `<|im_start|>role\n…<|im_end|>`;
/// `gemma` is Gemma 3's `<start_of_turn>role\n…<end_of_turn>\n`; `gemma4` is
/// Gemma 4's `<|turn>role\n…<turn|>\n` (roles system / user / model, each its
/// own turn; the system turn also carries the `<|think|>` reasoning cue).
pub const Family = enum { chatml, gemma, gemma4 };
pub var family: Family = .chatml;
/// Gemma: a user turn was opened by appendSystem (the system prefix) and the
/// next appendUser continues it rather than starting a fresh turn.
var gemma_user_open: bool = false;

/// Whether the model should emit a reasoning ("thought") block before its
/// answer. Process-global like `family`; only reasoning-capable families honor
/// it (see `reasoning`). When off, `openAssistant` primes an empty thought
/// block so the model answers directly (each family's non-thinking template
/// convention); when on, the turn is left unprimed and the model opens its own
/// block (gemma4 is additionally cued by `<|think|>` in the system turn).
pub var enable_thinking: bool = true;

/// Turn reasoning on/off (process-global). Forward-looking: a new reasoning
/// family just adds its markers to `reasoning` and its priming to
/// `openAssistant`/`appendSystem`; this stays untouched.
pub fn setThinking(on: bool) void {
    enable_thinking = on;
}

/// Delimiters of a family's reasoning block as they appear in the *generated
/// text* (decoded tokens): `open` starts the thought, `close` ends it. null =>
/// the family has no reasoning channel and the thinking toggle is a no-op.
pub const Reasoning = struct { open: []const u8, close: []const u8 };

/// A given family's reasoning-block markers, or null if it can't reason. The
/// single source of truth; `reasoning`/`supportsThinking` read the active
/// family, callers probing a not-yet-active family pass it explicitly.
pub fn reasoningFor(f: Family) ?Reasoning {
    return switch (f) {
        .chatml => .{ .open = "<think>", .close = "</think>" },
        .gemma4 => .{ .open = "<|channel>thought", .close = "<channel|>" },
        .gemma => null,
    };
}

/// The active family's reasoning-block markers, or null if it can't reason.
pub fn reasoning() ?Reasoning {
    return reasoningFor(family);
}

/// Whether a given family supports a reasoning block (pure — lets the GUI probe
/// a configured-but-not-loaded model without disturbing the process-global
/// `family`).
pub fn familySupportsThinking(f: Family) bool {
    return reasoningFor(f) != null;
}

/// Whether the active family supports a reasoning block (drives the GUI toggle).
pub fn supportsThinking() bool {
    return familySupportsThinking(family);
}

/// Map a GGUF `general.architecture` string to its chat template family, or
/// null if it isn't a family we template here. Single source of truth for the
/// arch→family mapping (used by the session loader and the GUI's pre-load
/// capability probe).
pub fn familyForArch(arch: []const u8) ?Family {
    if (std.mem.eql(u8, arch, "qwen3")) return .chatml;
    if (std.mem.eql(u8, arch, "qwen35")) return .chatml;
    if (std.mem.eql(u8, arch, "gemma3")) return .gemma;
    if (std.mem.eql(u8, arch, "gemma4")) return .gemma4;
    return null;
}

/// Select the chat template family (process-global, like the tokenizer).
pub fn setFamily(f: Family) void {
    family = f;
    gemma_user_open = false;
}

/// Point the template glue and stop check at `tok`'s vocab.
pub fn applyTokenizer(tok: *const Tokenizer) void {
    turn_end = tok.turn_end;
    pad = tok.pad;
    newline = tok.newline;
}

pub fn isStop(id: u32) bool {
    return id == turn_end or id == pad;
}

pub fn appendSystem(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    switch (family) {
        .chatml => try appendTurn(tok, gpa, "system", text, out),
        .gemma => {
            // Gemma has no system role: the content prefixes the first user
            // turn (first_user_prefix + "\n\n"), which appendUser completes.
            try tok.encode(gpa, "<start_of_turn>user\n", out);
            try tok.encode(gpa, text, out);
            try tok.encode(gpa, "\n\n", out);
            gemma_user_open = true;
        },
        .gemma4 => {
            // Gemma 4 has a dedicated system turn (unlike Gemma 3): a full
            // <|turn>system\n … <turn|>\n block. In thinking mode the <|think|>
            // reasoning cue is injected at the very top of it, before the
            // system content — matching the reference template. The system turn
            // stands alone, so the next appendUser opens a fresh user turn.
            try tok.encode(gpa, "<|turn>system\n", out);
            if (enable_thinking) try tok.encode(gpa, "<|think|>\n", out);
            try tok.encode(gpa, text, out);
            try tok.encode(gpa, "<turn|>\n", out);
        },
    }
}

pub fn appendUser(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    switch (family) {
        .chatml => try appendTurn(tok, gpa, "user", text, out),
        .gemma => {
            if (!gemma_user_open) try tok.encode(gpa, "<start_of_turn>user\n", out);
            gemma_user_open = false;
            try tok.encode(gpa, text, out);
            try tok.encode(gpa, "<end_of_turn>\n", out);
        },
        .gemma4 => {
            if (!gemma_user_open) try tok.encode(gpa, "<|turn>user\n", out);
            gemma_user_open = false;
            try tok.encode(gpa, text, out);
            try tok.encode(gpa, "<turn|>\n", out);
        },
    }
}

/// One piece of an interleaved user turn.
pub const Segment = union(enum) {
    text: []const u8,
    /// An encoded image occupying grid_w*grid_h cache rows.
    image: struct { grid_w: usize, grid_h: usize },
};

/// Append a user turn with interleaved text and image segments (Qwen3-VL:
/// each image is <|vision_start|> + grid_w*grid_h <|image_pad|> rows +
/// <|vision_end|>, inline in the turn at its mention point). The
/// placeholder pads keep ids aligned with cache rows — sampling penalties
/// and cached()-based prefill index by row — and each image's first pad
/// row index is appended to image_rows so the caller can interleave
/// prefill() with prefillImage().
pub fn appendUserSegments(tok: *const Tokenizer, gpa: std.mem.Allocator, segments: []const Segment, out: *std.ArrayList(u32), image_rows: *std.ArrayList(usize)) !void {
    switch (family) {
        .gemma => {
            try tok.encode(gpa, "<start_of_turn>user\n", out);
            for (segments) |seg| switch (seg) {
                .text => |t| try tok.encode(gpa, t, out),
                .image => |im| {
                    const row = try appendGemmaImage(tok, gpa, im.grid_w * im.grid_h, out);
                    try image_rows.append(gpa, row);
                },
            };
            try tok.encode(gpa, "<end_of_turn>\n", out);
        },
        .gemma4 => {
            try tok.encode(gpa, "<|turn>user\n", out);
            for (segments) |seg| switch (seg) {
                .text => |t| try tok.encode(gpa, t, out),
                .image => |im| {
                    const row = try appendGemma4Image(tok, gpa, im.grid_w * im.grid_h, out);
                    try image_rows.append(gpa, row);
                },
            };
            try tok.encode(gpa, "<turn|>\n", out);
        },
        .chatml => {
            const pad_id = tok.specialId("<|image_pad|>") orelse tok.pad;
            try tok.encode(gpa, "<|im_start|>user\n", out);
            for (segments) |seg| switch (seg) {
                .text => |t| try tok.encode(gpa, t, out),
                .image => |im| {
                    try tok.encode(gpa, "<|vision_start|>", out);
                    try image_rows.append(gpa, out.items.len);
                    try out.appendNTimes(gpa, pad_id, im.grid_w * im.grid_h);
                    try tok.encode(gpa, "<|vision_end|>", out);
                },
            };
            try out.append(gpa, turn_end);
            try out.append(gpa, newline);
        },
    }
}

/// Gemma image block: `<start_of_image>` + `n_tokens` `<image_soft_token>`
/// placeholders + `<end_of_image>`. Returns the index of the first
/// placeholder row so the caller can splice the ViT embeddings in with
/// prefillImage (the placeholders keep `ids` aligned with cache rows).
pub fn appendGemmaImage(tok: *const Tokenizer, gpa: std.mem.Allocator, n_tokens: usize, out: *std.ArrayList(u32)) !usize {
    const soft = tok.specialId("<image_soft_token>") orelse return error.MissingImageToken;
    try tok.encode(gpa, "<start_of_image>", out);
    const first = out.items.len;
    try out.appendNTimes(gpa, soft, n_tokens);
    try tok.encode(gpa, "<end_of_image>", out);
    return first;
}

/// Gemma 4 image block: `<|image>` + `n_tokens` placeholder rows (pad) +
/// `<image|>`. Returns the first placeholder row so the caller can splice the
/// embedder output in with prefillImage. Unlike Gemma 3 there is no dedicated
/// soft token — the placeholder id is immaterial (those rows are overwritten by
/// the projected image embeddings), so `pad` keeps `ids` aligned with cache rows.
pub fn appendGemma4Image(tok: *const Tokenizer, gpa: std.mem.Allocator, n_tokens: usize, out: *std.ArrayList(u32)) !usize {
    try tok.encode(gpa, "<|image>", out);
    const first = out.items.len;
    try out.appendNTimes(gpa, tok.pad, n_tokens);
    try tok.encode(gpa, "<image|>", out);
    return first;
}

/// Start the assistant turn the model completes (ChatML: `<|im_start|>
/// assistant\n`; Gemma: `<start_of_turn>model\n`).
pub fn openAssistant(tok: *const Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !void {
    switch (family) {
        .chatml => {
            try tok.encode(gpa, "<|im_start|>assistant\n", out);
            // Non-thinking: prime an empty reasoning block (Qwen's
            // <think>\n\n</think> convention) so the model answers directly.
            // Thinking: unprimed — the model emits its own <think>…</think>.
            if (!enable_thinking) try tok.encode(gpa, "<think>\n\n</think>\n\n", out);
        },
        .gemma => try tok.encode(gpa, "<start_of_turn>model\n", out),
        .gemma4 => {
            try tok.encode(gpa, "<|turn>model\n", out);
            // Non-thinking: prime an empty thought channel so the model answers
            // directly. Thinking: unprimed — the model opens its own
            // <|channel>thought … <channel|> (cued by <|think|> in the system).
            if (!enable_thinking) try tok.encode(gpa, "<|channel>thought\n<channel|>", out);
        },
    }
}

/// Close a generated assistant turn so another user turn can follow (the
/// stop token itself is never appended by the engine).
pub fn closeAssistant(gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !void {
    try out.append(gpa, turn_end);
    try out.append(gpa, newline);
}

/// One parsed piece of an interactive chat line: literal text or an
/// @-mentioned image path.
pub const Part = union(enum) { text: []const u8, image: []const u8 };

/// Image file extensions accepted as @mentions (what the CLI's vips-backed
/// decoder handles); anything else after '@' stays literal text.
const image_exts = [_][]const u8{ ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff" };

fn hasImageExt(path: []const u8) bool {
    for (image_exts) |ext| {
        if (std.ascii.endsWithIgnoreCase(path, ext)) return true;
    }
    return false;
}

/// Split an interactive chat line into text and @image mentions:
/// `@path.jpg` or `@"path with spaces.png"` becomes an image part (the
/// path must end in a known image extension — anything else, e.g. @handles
/// or emails, stays text), and trailing sentence punctuation after an
/// unquoted path returns to the text. All parts are slices of `line`.
pub fn parseImageMentions(gpa: std.mem.Allocator, line: []const u8, parts: *std.ArrayList(Part)) !void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < line.len) {
        if (line[i] != '@') {
            i += 1;
            continue;
        }
        var pstart = i + 1;
        var pend: usize = undefined;
        var next: usize = undefined;
        if (pstart < line.len and line[pstart] == '"') {
            pstart += 1;
            pend = std.mem.indexOfScalarPos(u8, line, pstart, '"') orelse {
                i += 1; // unterminated quote: not a mention
                continue;
            };
            next = pend + 1;
        } else {
            pend = std.mem.indexOfAnyPos(u8, line, pstart, " \t\n") orelse line.len;
            while (pend > pstart and std.mem.indexOfScalar(u8, ",.;:!?)", line[pend - 1]) != null) pend -= 1;
            next = pend;
        }
        const path = line[pstart..pend];
        if (!hasImageExt(path)) {
            i += 1; // not an image mention: the '@' stays literal text
            continue;
        }
        if (i > start) try parts.append(gpa, .{ .text = line[start..i] });
        try parts.append(gpa, .{ .image = path });
        start = next;
        i = next;
    }
    if (start < line.len) try parts.append(gpa, .{ .text = line[start..] });
}

fn appendTurn(tok: *const Tokenizer, gpa: std.mem.Allocator, role: []const u8, text: []const u8, out: *std.ArrayList(u32)) !void {
    try tok.encode(gpa, "<|im_start|>", out);
    try tok.encode(gpa, role, out);
    try out.append(gpa, newline);
    try tok.encode(gpa, text, out);
    try out.append(gpa, turn_end);
    try out.append(gpa, newline);
}

// --- tests -----------------------------------------------------------------

test "turn building matches whole-template tokenization" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try appendSystem(&tok, gpa, "Describe the image:", &ids);
    try appendUser(&tok, gpa, "a cat", &ids);
    try openAssistant(&tok, gpa, &ids);

    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    try tok.encode(
        gpa,
        "<|im_start|>system\nDescribe the image:<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n",
        &ref,
    );
    try std.testing.expectEqualSlices(u32, ref.items, ids.items);
}

test "reasoning descriptor is family-scoped" {
    const saved = family;
    defer setFamily(saved);

    setFamily(.chatml);
    try std.testing.expect(supportsThinking());
    try std.testing.expectEqualStrings("<think>", reasoning().?.open);
    try std.testing.expectEqualStrings("</think>", reasoning().?.close);

    setFamily(.gemma4);
    try std.testing.expect(supportsThinking());
    try std.testing.expectEqualStrings("<|channel>thought", reasoning().?.open);
    try std.testing.expectEqualStrings("<channel|>", reasoning().?.close);

    setFamily(.gemma);
    try std.testing.expect(!supportsThinking());
    try std.testing.expectEqual(@as(?Reasoning, null), reasoning());
}

test "arch→family mapping and family-scoped thinking probe (no global mutation)" {
    // The GUI probes a configured-but-unloaded model's reasoning support via
    // these pure helpers; they must not disturb the process-global `family`.
    const saved = family;
    defer setFamily(saved);
    setFamily(.gemma); // a distinctive sentinel to prove the probe leaves it be

    try std.testing.expectEqual(@as(?Family, .chatml), familyForArch("qwen35"));
    try std.testing.expectEqual(@as(?Family, .gemma), familyForArch("gemma3"));
    try std.testing.expectEqual(@as(?Family, .gemma4), familyForArch("gemma4"));
    try std.testing.expectEqual(@as(?Family, null), familyForArch("llama"));
    try std.testing.expectEqual(@as(?Family, null), familyForArch(""));

    try std.testing.expect(familySupportsThinking(.chatml));
    try std.testing.expect(familySupportsThinking(.gemma4));
    try std.testing.expect(!familySupportsThinking(.gemma));

    try std.testing.expectEqual(Family.gemma, family); // untouched by the probes
}

// Thinking off must prepend an empty reasoning block to the assistant open (so
// the model answers directly); thinking on must leave the open untouched (the
// model emits its own block). Uses the embedded chatml tokenizer.
test "openAssistant primes an empty reasoning block only when thinking is off" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const saved_family = family;
    const saved_thinking = enable_thinking;
    defer {
        setFamily(saved_family);
        setThinking(saved_thinking);
    }
    setFamily(.chatml);

    var open_ref: std.ArrayList(u32) = .empty;
    defer open_ref.deinit(gpa);
    try tok.encode(gpa, "<|im_start|>assistant\n", &open_ref);

    // Thinking on: bare assistant open, model reasons on its own.
    setThinking(true);
    var on: std.ArrayList(u32) = .empty;
    defer on.deinit(gpa);
    try openAssistant(&tok, gpa, &on);
    try std.testing.expectEqualSlices(u32, open_ref.items, on.items);

    // Thinking off: same open plus a primed empty <think></think>, so it is a
    // strict prefix-extension of the bare open.
    setThinking(false);
    var off: std.ArrayList(u32) = .empty;
    defer off.deinit(gpa);
    try openAssistant(&tok, gpa, &off);
    try std.testing.expect(off.items.len > on.items.len);
    try std.testing.expectEqualSlices(u32, on.items, off.items[0..on.items.len]);
}

// Gemma turn building (family = gemma) must match one-shot tokenization of
// the same template string; gated on the Gemma 3 GGUF. Also confirms the
// BOS/turn glue against llama.cpp's chat template shape.
test "gemma turn building matches whole-template tokenization" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const gguf_mod = @import("../gguf.zig");
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();
    // applyTokenizer + setFamily mutate process-global template state; capture
    // and restore ALL of it (not just family) so later tests still see the
    // chatml/Qwen defaults — otherwise gemma's stop/glue ids leak out.
    const saved_turn_end = turn_end;
    const saved_pad = pad;
    const saved_newline = newline;
    const saved_family = family;
    applyTokenizer(&tok);
    setFamily(.gemma);
    defer {
        turn_end = saved_turn_end;
        pad = saved_pad;
        newline = saved_newline;
        setFamily(saved_family);
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    const bos = tok.specialId("<bos>").?;
    try ids.append(gpa, bos);
    try appendSystem(&tok, gpa, "You are terse.", &ids);
    try appendUser(&tok, gpa, "Hi there", &ids);
    try openAssistant(&tok, gpa, &ids);

    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    try ref.append(gpa, bos);
    try tok.encode(
        gpa,
        "<start_of_turn>user\nYou are terse.\n\nHi there<end_of_turn>\n<start_of_turn>model\n",
        &ref,
    );
    try std.testing.expectEqualSlices(u32, ref.items, ids.items);

    // closeAssistant emits <end_of_turn>\n via the vocab-driven glue ids.
    try std.testing.expectEqual(@as(u32, 106), turn_end); // <end_of_turn>
    ids.clearRetainingCapacity();
    try closeAssistant(gpa, &ids);
    var ref2: std.ArrayList(u32) = .empty;
    defer ref2.deinit(gpa);
    try tok.encode(gpa, "<end_of_turn>\n", &ref2);
    try std.testing.expectEqualSlices(u32, ref2.items, ids.items);
}

test "segment turn interleaves image pads at the mention points" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    const segs = [_]Segment{
        .{ .text = "compare " },
        .{ .image = .{ .grid_w = 2, .grid_h = 2 } },
        .{ .text = " and " },
        .{ .image = .{ .grid_w = 3, .grid_h = 1 } },
    };
    try appendUserSegments(&tok, gpa, &segs, &ids, &rows);

    // Piecewise reference: same encoder over the glue strings, pads spliced
    // at the recorded rows (independent of the vocab's vision specials).
    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    const pad_id = tok.specialId("<|image_pad|>") orelse tok.pad;
    try tok.encode(gpa, "<|im_start|>user\ncompare <|vision_start|>", &ref);
    try std.testing.expectEqual(ref.items.len, rows.items[0]);
    try ref.appendNTimes(gpa, pad_id, 4);
    try tok.encode(gpa, "<|vision_end|> and <|vision_start|>", &ref);
    try std.testing.expectEqual(ref.items.len, rows.items[1]);
    try ref.appendNTimes(gpa, pad_id, 3);
    try tok.encode(gpa, "<|vision_end|>", &ref);
    try ref.append(gpa, turn_end);
    try ref.append(gpa, newline);
    try std.testing.expectEqualSlices(u32, ref.items, ids.items);
}

test "closeAssistant appends im_end + newline" {
    const gpa = std.testing.allocator;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try closeAssistant(gpa, &ids);
    try std.testing.expectEqualSlices(u32, &.{ tokenizer_mod.im_end, newline_id }, ids.items);
}

test "image mention parsing" {
    const gpa = std.testing.allocator;
    var parts: std.ArrayList(Part) = .empty;
    defer parts.deinit(gpa);

    // Interleaved text + plain and quoted mentions, trailing punctuation.
    try parseImageMentions(gpa, "compare @a.png and @\"b c.png\", or @d.png.", &parts);
    try std.testing.expectEqual(@as(usize, 7), parts.items.len);
    try std.testing.expectEqualStrings("compare ", parts.items[0].text);
    try std.testing.expectEqualStrings("a.png", parts.items[1].image);
    try std.testing.expectEqualStrings(" and ", parts.items[2].text);
    try std.testing.expectEqualStrings("b c.png", parts.items[3].image);
    try std.testing.expectEqualStrings(", or ", parts.items[4].text);
    try std.testing.expectEqualStrings("d.png", parts.items[5].image);
    try std.testing.expectEqualStrings(".", parts.items[6].text);

    // Image-only line.
    parts.clearRetainingCapacity();
    try parseImageMentions(gpa, "@x.png", &parts);
    try std.testing.expectEqual(@as(usize, 1), parts.items.len);
    try std.testing.expectEqualStrings("x.png", parts.items[0].image);

    // Non-.png @mentions (emails, handles) stay literal text.
    parts.clearRetainingCapacity();
    try parseImageMentions(gpa, "mail me@foo.org about @alice", &parts);
    try std.testing.expectEqual(@as(usize, 1), parts.items.len);
    try std.testing.expectEqualStrings("mail me@foo.org about @alice", parts.items[0].text);

    // Unterminated quote stays text.
    parts.clearRetainingCapacity();
    try parseImageMentions(gpa, "say @\"oops.png", &parts);
    try std.testing.expectEqual(@as(usize, 1), parts.items.len);
    try std.testing.expectEqualStrings("say @\"oops.png", parts.items[0].text);

    // Non-PNG formats (vips-decoded) and case-insensitive extensions.
    parts.clearRetainingCapacity();
    try parseImageMentions(gpa, "@photo.JPG vs @anim.webp", &parts);
    try std.testing.expectEqual(@as(usize, 3), parts.items.len);
    try std.testing.expectEqualStrings("photo.JPG", parts.items[0].image);
    try std.testing.expectEqualStrings(" vs ", parts.items[1].text);
    try std.testing.expectEqualStrings("anim.webp", parts.items[2].image);
}

test "stop tokens" {
    try std.testing.expect(isStop(tokenizer_mod.im_end));
    try std.testing.expect(isStop(tokenizer_mod.pad_token));
    try std.testing.expect(!isStop(newline_id));
}
