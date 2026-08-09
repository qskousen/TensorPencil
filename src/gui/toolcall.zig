//! Parsing of the `<image ...>…</image>` tool call that tp-gui's LLM emits to
//! request image generation.
//!
//! Pure string logic so it unit-tests cheaply and both consumers — the
//! generation scanner (chat.zig) and the display hider (app.zig) — share one
//! definition of "what counts as a call". Keeping them in lockstep means a call
//! that fires a generation is exactly a call that's hidden from the reply.
//!
//! Two guards keep the model from *accidentally* triggering the tool:
//!  1. Callers scan only the answer, never the reasoning block (`answerText`):
//!     models routinely write out the `<image>` tag while *thinking about* what
//!     to generate.
//!  2. Only line-anchored tags count (`nextImageCall`), matching the tool
//!     prompt's "on its own line" contract: a casual inline mention of the tag
//!     (e.g. the model explaining how the tool works) stays ordinary text.
const std = @import("std");
const tool_call = @import("TensorPencil").llm.tool_call;

// The reasoning-block splitter moved into the library (`llm/tool_call.zig`)
// when the trained-format tool parser landed: BOTH scanners need "where does the
// answer start", and two definitions of it is exactly the drift that lets one
// caller fire a tool the other hides. Re-exported so this module stays the one
// place the GUI reaches for tool-call string logic.
pub const Reasoning = tool_call.Reasoning;
pub const Split = tool_call.Split;
pub const splitThought = tool_call.splitThought;
pub const endsInsideThought = tool_call.endsInsideThought;
pub const answerText = tool_call.answerText;

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
