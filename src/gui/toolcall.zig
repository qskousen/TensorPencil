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

/// A generated assistant turn split into its reasoning block and its answer.
/// `open` is true while the block has not closed yet (still streaming, or
/// generation stopped mid-thought).
pub const Split = struct {
    /// The reasoning block's contents, or null when the turn has no block.
    think: ?[]const u8,
    /// Everything after the block closed; "" while it is still open.
    answer: []const u8,
    open: bool,
};

/// Split a generated assistant turn around its reasoning block. ONE definition
/// shared by the display (`app.zig`'s thought expander) and the tool-call
/// scanner, so a call that fires a generation is exactly a call that's hidden
/// from the reply — and so both agree on what counts as "inside the thought".
///
/// ⚠️ **`primed` is the whole subtlety, and getting it wrong is what broke
/// thinking for Qwen3.5/Bonsai in the GUI.** The two prompt builders here
/// disagree about who writes the OPENING marker:
///
///   - the hand glue (`llm/chat.zig openAssistant`) leaves a thinking turn
///     UNPRIMED, so the model emits its own `<think>` and the generated text
///     starts with it;
///   - a model's own jinja template (the render-driven path) PRIMES
///     `…assistant\n<think>\n` into the prompt, so the generated text starts
///     *inside* the block and the only marker it ever emits is the CLOSE.
///
/// With `primed` false against a primed prompt, nothing matches, the thought is
/// rendered as part of the answer with a stray `</think>` in the middle, and —
/// far worse — `answerText` returns the whole text, so an `<image>` tag the model
/// wrote *while reasoning about* what to generate FIRES a generation. That is
/// precisely the failure this module exists to prevent, so `primed` is measured
/// from the rendered prompt (`endsInsideThought`) rather than assumed per family.
///
/// A primed turn whose model redundantly repeats the marker is handled too: the
/// leading `open` is consumed either way.
pub fn splitThought(text: []const u8, r: ?Reasoning, primed: bool) Split {
    const ws = " \n\r\t";
    const rr = r orelse return .{ .think = null, .answer = text, .open = false };
    const t = std.mem.trimStart(u8, text, ws);
    const body = if (std.mem.startsWith(u8, t, rr.open))
        t[rr.open.len..]
    else if (primed)
        t
    else
        return .{ .think = null, .answer = text, .open = false };
    if (std.mem.indexOf(u8, body, rr.close)) |end| return .{
        .think = std.mem.trim(u8, body[0..end], ws),
        .answer = std.mem.trimStart(u8, body[end + rr.close.len ..], ws),
        .open = false,
    };
    return .{ .think = std.mem.trimStart(u8, body, ws), .answer = "", .open = true };
}

/// Whether a RENDERED prompt ends inside an open reasoning block — i.e. whether
/// the generation that follows it starts already inside the model's thought.
/// This is the `primed` input to `splitThought`, and it is measured rather than
/// derived from the family so it stays right for any template, including ones
/// that prime `<think>\n\n</think>` (thinking OFF: closed, so NOT primed).
///
/// Scan the prompt's TAIL only (the caller decodes the last handful of tokens):
/// an earlier user turn quoting `</think>` must not decide this.
pub fn endsInsideThought(prompt_tail: []const u8, r: ?Reasoning) bool {
    const rr = r orelse return false;
    const last_open = std.mem.lastIndexOf(u8, prompt_tail, rr.open);
    const last_close = std.mem.lastIndexOf(u8, prompt_tail, rr.close);
    const o = last_open orelse return false;
    return if (last_close) |c| o > c else true;
}

/// The answer portion of a completed assistant turn: everything after the
/// model's reasoning block closes (or the whole text if `r` is null — the
/// family doesn't reason, or thinking is off so no block is emitted). Tool
/// calls are scanned only here, NEVER inside the thought block, because models
/// routinely write out the `<image>` tag while *reasoning about* what to
/// generate. See `splitThought` for `primed`.
pub fn answerText(text: []const u8, r: ?Reasoning, primed: bool) []const u8 {
    const s = splitThought(text, r, primed);
    if (s.think == null) return s.answer;
    return s.answer; // "" while the block is still open — nothing to scan yet
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

const think_markers: Reasoning = .{ .open = "<think>", .close = "</think>" };

test "answerText: a tag inside the reasoning block is excluded" {
    // The model wrote the tag while reasoning; only the post-think answer scans.
    const txt = "<think>I'll emit <image>a cat</image> for this.</think>\nSure!\n<image>a dog</image>";
    const ans = answerText(txt, think_markers, false);
    try testing.expectEqualStrings("Sure!\n<image>a dog</image>", ans);
    // The excluded thought's tag must not survive into the scanned answer.
    try testing.expectEqual(ScanResult.none, nextImageCall(nextImageCall(ans).call.after));
    try testing.expectEqualStrings("a dog", nextImageCall(ans).call.prompt);
}

test "answerText: reasoning still open yields empty answer" {
    try testing.expectEqualStrings("", answerText("<think>still thinking about <image>a cat</image>", think_markers, false));
}

test "answerText: null reasoning returns the whole text" {
    try testing.expectEqualStrings("<image>x</image>", answerText("<image>x</image>", null, false));
}

test "answerText: gemma4-style channel markers" {
    const ch: Reasoning = .{ .open = "<|channel>thought", .close = "<channel|>" };
    const txt = "<|channel>thought maybe <image>skip me</image><channel|>Here:\n<image>keep me</image>";
    try testing.expectEqualStrings("keep me", nextImageCall(answerText(txt, ch, false)).call.prompt);
}

// --- primed thoughts (the render-driven template path) ----------------------
// The prompt ends `…assistant\n<think>\n`, so the generated text starts INSIDE
// the block and the only marker it emits is the CLOSE. This is what Qwen3.5 /
// Bonsai actually produce in the GUI.

test "splitThought: primed turn with no opening marker folds the thought" {
    const txt = "Here's a thinking process:\n1. multiply.\n</think>\n\n17 x 4 = 68.";
    const s = splitThought(txt, think_markers, true);
    try testing.expectEqualStrings("Here's a thinking process:\n1. multiply.", s.think.?);
    try testing.expectEqualStrings("17 x 4 = 68.", s.answer);
    try testing.expect(!s.open);
}

test "splitThought: the SAME text unprimed is all answer (the bug)" {
    // Without `primed` nothing matches: the thought renders as part of the reply
    // with a stray `</think>` in it. Pins the regression the fix is for.
    const txt = "Here's a thinking process:\n1. multiply.\n</think>\n\n17 x 4 = 68.";
    const s = splitThought(txt, think_markers, false);
    try testing.expectEqual(@as(?[]const u8, null), s.think);
    try testing.expectEqualStrings(txt, s.answer);
}

test "answerText: a primed thought's image tag must NOT reach the scanner" {
    // The serious half of the bug: an <image> tag written while reasoning would
    // fire a real generation if the thought were treated as answer text.
    const txt = "I could draw it:\n<image>a cat</image>\n</think>\nSure!\n<image>a dog</image>";
    const ans = answerText(txt, think_markers, true);
    try testing.expectEqualStrings("Sure!\n<image>a dog</image>", ans);
    try testing.expectEqualStrings("a dog", nextImageCall(ans).call.prompt);
    try testing.expectEqual(ScanResult.none, nextImageCall(nextImageCall(ans).call.after));
    // Unprimed, the thought's tag DOES leak through — which is the bug.
    try testing.expectEqualStrings("a cat", nextImageCall(answerText(txt, think_markers, false)).call.prompt);
}

test "splitThought: a primed turn that redundantly repeats the marker" {
    // Observed on Bonsai's one-shot path: the prompt primes `<think>` and the
    // model emits another. The leading marker is consumed either way.
    const txt = "\n<think>\nthinking…\n</think>\n\nDone.";
    for ([_]bool{ true, false }) |primed| {
        const s = splitThought(txt, think_markers, primed);
        try testing.expectEqualStrings("thinking…", s.think.?);
        try testing.expectEqualStrings("Done.", s.answer);
    }
}

test "splitThought: primed and still streaming stays open" {
    const s = splitThought("weighing the options", think_markers, true);
    try testing.expectEqualStrings("weighing the options", s.think.?);
    try testing.expectEqualStrings("", s.answer);
    try testing.expect(s.open);
}

test "endsInsideThought: measures the render rather than trusting the family" {
    // Thinking ON: the template primes an OPEN block.
    try testing.expect(endsInsideThought("<|im_start|>assistant\n<think>\n", think_markers));
    // Thinking OFF: it primes a CLOSED empty block — not primed.
    try testing.expect(!endsInsideThought("<|im_start|>assistant\n<think>\n\n</think>\n\n", think_markers));
    // Hand glue leaves the turn unprimed.
    try testing.expect(!endsInsideThought("<|im_start|>assistant\n", think_markers));
    // A non-reasoning family never primes.
    try testing.expect(!endsInsideThought("<think>\n", null));
    // gemma4's channel markers work the same way.
    const ch: Reasoning = .{ .open = "<|channel>thought", .close = "<channel|>" };
    try testing.expect(endsInsideThought("<|turn>model\n<|channel>thought\n", ch));
    try testing.expect(!endsInsideThought("<|turn>model\n<|channel>thought\n<channel|>", ch));
}
