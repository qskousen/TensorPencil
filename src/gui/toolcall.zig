//! Parsing of the `<image ...>…</image>` tool call that tp-gui's LLM emits to
//! request image generation.
//!
//! Pure string logic (std only) so it unit-tests cheaply and both consumers —
//! the generation scanner (chat.zig) and the display hider (app.zig) — share
//! one definition of "what counts as a call". Keeping them in lockstep means a
//! call that fires a generation is exactly a call that's hidden from the reply.
//!
//! Two guards keep the model from *accidentally* triggering the tool:
//!  1. Callers scan only the answer, never the reasoning block (`answerText`):
//!     models routinely write out the `<image>` tag while *thinking about* what
//!     to generate.
//!  2. Only line-anchored tags count (`nextImageCall`), matching the tool
//!     prompt's "on its own line" contract: a casual inline mention of the tag
//!     (e.g. the model explaining how the tool works) stays ordinary text.
const std = @import("std");

/// A family's reasoning-block delimiters as they appear in generated text
/// (mirrors `TensorPencil`'s `llm.chat.Reasoning`; kept local so this module
/// stays std-only and independently testable).
pub const Reasoning = struct { open: []const u8, close: []const u8 };

/// Result of scanning for the next `<image ...>…</image>` tool call.
///  - `.call` — a complete call: `text_before` (ordinary text to render),
///    the parsed `attrs`/`prompt`, and `after` (remaining text to keep scanning).
///  - `.partial` — a line-anchored `<image` whose open tag or body is still
///    streaming: everything from it onward is pending (display hides it; the
///    turn-complete scanner stops).
///  - `.none` — no line-anchored tool call remains; the whole buffer is text.
pub const ScanResult = union(enum) {
    none,
    partial: struct { text_before: []const u8 },
    call: struct { text_before: []const u8, attrs: []const u8, prompt: []const u8, after: []const u8 },
};

/// True if `idx` sits at the start of a line in `buf` (only whitespace precedes
/// it back to the previous newline or the buffer start).
fn atLineStart(buf: []const u8, idx: usize) bool {
    var i = idx;
    while (i > 0) {
        i -= 1;
        switch (buf[i]) {
            '\n' => return true,
            ' ', '\t', '\r' => {},
            else => return false,
        }
    }
    return true;
}

/// Find the next `<image ...>…</image>` tool call in `buf`. Only tags that
/// begin a line (after optional leading whitespace) count — an inline/casual
/// mention of the tag is left as ordinary text, so it neither fires a
/// generation nor gets hidden from the reply. Callers must strip the reasoning
/// block first (see `answerText`); this scans only text.
pub fn nextImageCall(buf: []const u8) ScanResult {
    const close = "</image>";
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, buf, from, "<image")) |a| {
        if (!atLineStart(buf, a)) {
            // Casual mention, not a tool call: skip past it, keep it as text.
            from = a + "<image".len;
            continue;
        }
        const after_open = buf[a + "<image".len ..];
        const gt = std.mem.indexOfScalar(u8, after_open, '>') orelse
            return .{ .partial = .{ .text_before = buf[0..a] } };
        const body = after_open[gt + 1 ..];
        const b = std.mem.indexOf(u8, body, close) orelse
            return .{ .partial = .{ .text_before = buf[0..a] } };
        return .{ .call = .{
            .text_before = buf[0..a],
            .attrs = after_open[0..gt],
            .prompt = std.mem.trim(u8, body[0..b], " \n\r\t"),
            .after = body[b + close.len ..],
        } };
    }
    return .none;
}

/// The answer portion of a completed assistant turn: everything after the
/// model's reasoning block closes (or the whole text if `r` is null — the
/// family doesn't reason, or thinking is off so no block is emitted). Tool
/// calls are scanned only here, NEVER inside the thought block, because models
/// routinely write out the `<image>` tag while *reasoning about* what to
/// generate. Mirrors app.zig's `parseThink` split.
pub fn answerText(text: []const u8, r: ?Reasoning) []const u8 {
    const rr = r orelse return text;
    const t = std.mem.trimStart(u8, text, " \n\r\t");
    if (std.mem.startsWith(u8, t, rr.open)) {
        const rest = t[rr.open.len..];
        if (std.mem.indexOf(u8, rest, rr.close)) |end|
            return rest[end + rr.close.len ..];
        return ""; // reasoning still open — no answer to scan yet
    }
    return text;
}

const testing = std.testing;

test "nextImageCall: a line-anchored call is parsed with surrounding text" {
    const r = nextImageCall("Sure!\n<image>a red fox</image>\nDone.");
    try testing.expectEqualStrings("Sure!\n", r.call.text_before);
    try testing.expectEqualStrings("", r.call.attrs);
    try testing.expectEqualStrings("a red fox", r.call.prompt);
    try testing.expectEqualStrings("\nDone.", r.call.after);
}

test "nextImageCall: attributes are captured verbatim" {
    const r = nextImageCall("<image width=1024 height=1536 seed=42>tall tower</image>");
    try testing.expectEqualStrings(" width=1024 height=1536 seed=42", r.call.attrs);
    try testing.expectEqualStrings("tall tower", r.call.prompt);
}

test "nextImageCall: an inline/casual mention is NOT a call" {
    // The model explaining the tool, mid-sentence — must stay ordinary text.
    try testing.expectEqual(ScanResult.none, nextImageCall("Just write <image>…</image> on its own line."));
    // Backtick-wrapped mention is likewise not line-anchored.
    try testing.expectEqual(ScanResult.none, nextImageCall("Use the `<image>desc</image>` tag."));
}

test "nextImageCall: leading whitespace still counts as line-anchored" {
    const r = nextImageCall("  \t<image>indented but on its own line</image>");
    try testing.expectEqualStrings("indented but on its own line", r.call.prompt);
}

test "nextImageCall: incomplete open tag or body is partial" {
    switch (nextImageCall("here we go\n<image width=102")) {
        .partial => |p| try testing.expectEqualStrings("here we go\n", p.text_before),
        else => return error.TestUnexpectedResult,
    }
    switch (nextImageCall("\n<image>still streaming the promp")) {
        .partial => |p| try testing.expectEqualStrings("\n", p.text_before),
        else => return error.TestUnexpectedResult,
    }
}

test "nextImageCall: plain text yields none" {
    try testing.expectEqual(ScanResult.none, nextImageCall("no tool calls at all here"));
}

test "nextImageCall: multiple calls scanned in sequence via .after" {
    var rest: []const u8 = "<image>one</image>\n<image>two</image>";
    const a = nextImageCall(rest);
    try testing.expectEqualStrings("one", a.call.prompt);
    rest = a.call.after;
    const b = nextImageCall(rest);
    try testing.expectEqualStrings("two", b.call.prompt);
    try testing.expectEqual(ScanResult.none, nextImageCall(b.call.after));
}

test "answerText: a tag inside the reasoning block is excluded" {
    const think: Reasoning = .{ .open = "<think>", .close = "</think>" };
    // The model wrote the tag while reasoning; only the post-think answer scans.
    const txt = "<think>I'll emit <image>a cat</image> for this.</think>\nSure!\n<image>a dog</image>";
    const ans = answerText(txt, think);
    try testing.expectEqualStrings("\nSure!\n<image>a dog</image>", ans);
    // The excluded thought's tag must not survive into the scanned answer.
    try testing.expectEqual(ScanResult.none, nextImageCall(nextImageCall(ans).call.after));
    try testing.expectEqualStrings("a dog", nextImageCall(ans).call.prompt);
}

test "answerText: reasoning still open yields empty answer" {
    const think: Reasoning = .{ .open = "<think>", .close = "</think>" };
    try testing.expectEqualStrings("", answerText("<think>still thinking about <image>a cat</image>", think));
}

test "answerText: null reasoning returns the whole text" {
    try testing.expectEqualStrings("<image>x</image>", answerText("<image>x</image>", null));
}

test "answerText: gemma4-style channel markers" {
    const ch: Reasoning = .{ .open = "<|channel>thought", .close = "<channel|>" };
    const txt = "<|channel>thought maybe <image>skip me</image><channel|>Here:\n<image>keep me</image>";
    try testing.expectEqualStrings("keep me", nextImageCall(answerText(txt, ch)).call.prompt);
}
