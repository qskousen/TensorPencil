//! Parsing of the `<image ...>...</image>` tool call that tp-gui's LLM emits to
//! request image generation.
//!
//! Pure string logic so it unit-tests cheaply and both consumers, the
//! generation scanner (chat.zig) and the display hider (app.zig), share one
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

// The reasoning-block splitter lives in the library (`llm/tool_call.zig`) because
// BOTH scanners need "where does the answer start", and two definitions of it is the
// drift that lets one caller fire a tool the other hides. Re-exported so this module
// stays the one place the GUI reaches for tool-call string logic.
pub const Reasoning = tool_call.Reasoning;
pub const Split = tool_call.Split;
pub const splitThought = tool_call.splitThought;
pub const endsInsideThought = tool_call.endsInsideThought;
pub const answerText = tool_call.answerText;

/// Result of scanning for the next `<image ...>...</image>` tool call.
///  - `.call`, a complete call: `text_before` (ordinary text to render),
///    the parsed `attrs`/`prompt`, and `after` (remaining text to keep scanning).
///  - `.partial`, a line-anchored `<image` whose open tag or body is still
///    streaming: everything from it onward is pending (display hides it; the
///    turn-complete scanner stops).
///  - `.none`, no line-anchored tool call remains; the whole buffer is text.
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

/// Find the next `<image ...>...</image>` tool call in `buf`. Only tags that
/// begin a line (after optional leading whitespace) count, an inline/casual
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
    // The model explaining the tool, mid-sentence, must stay ordinary text.
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

/// One piece of a reply, in document order: either prose to render, or a run of
/// consecutive tool calls that should share a single card.
pub const Segment = union(enum) {
    prose: []const u8,
    calls: struct {
        /// Indices into the variant's image list, `[start, start+len)`. `len`
        /// can be FEWER than `n_calls`: a reopened conversation only rebuilt the
        /// renders whose files it could find, and images are only on disk at all
        /// when saving is on.
        start: usize,
        len: usize,
        /// How many calls this run actually contains.
        n_calls: usize,
        /// The literal span of the reply these calls occupy, markup and all.
        /// Carried rather than reconstructed so the section shows what the model
        /// wrote, and so it still shows when no image survived.
        text: []const u8,
    },
};

/// Split a reply into prose and call runs, in the order the model wrote them.
///
/// Consecutive calls (nothing but whitespace between) group into one run, which
/// is what makes a four-image request one 2x2 card rather than four stacked
/// cards. A call with an empty prompt produces no image (see chat.zig), so it
/// advances neither the index nor the run — otherwise every card after one
/// would show the wrong picture.
///
/// `out` is filled up to its length; returns the slice written.
pub fn segments(reply: []const u8, n_images: usize, out: []Segment) []Segment {
    const base = @intFromPtr(reply.ptr);
    var rest = reply;
    var n: usize = 0;
    var next_img: usize = 0;
    var run_start: usize = 0;
    var run_len: usize = 0;
    var run_calls: usize = 0;
    var run_from: usize = 0;
    var run_to: usize = 0;

    const push = struct {
        fn prose(o: []Segment, i: *usize, t: []const u8) void {
            if (std.mem.trim(u8, t, " \t\r\n").len == 0) return;
            if (i.* < o.len) {
                o[i.*] = .{ .prose = t };
                i.* += 1;
            }
        }
        fn run(o: []Segment, i: *usize, start: usize, len: usize, calls: usize, text: []const u8) void {
            if (calls == 0) return;
            if (i.* < o.len) {
                o[i.*] = .{ .calls = .{ .start = start, .len = len, .n_calls = calls, .text = text } };
                i.* += 1;
            }
        }
    };

    while (true) {
        switch (nextImageCall(rest)) {
            .call => |c| {
                if (std.mem.trim(u8, c.text_before, " \t\r\n").len > 0) {
                    push.run(out, &n, run_start, run_len, run_calls, reply[run_from..run_to]);
                    run_len = 0;
                    run_calls = 0;
                    push.prose(out, &n, c.text_before);
                }
                // The call's own span: from where `text_before` ends to where
                // `after` begins. All three are slices of `reply`, so this is
                // pointer arithmetic on one buffer, not a search.
                const from = @intFromPtr(c.text_before.ptr) + c.text_before.len - base;
                const to = @intFromPtr(c.after.ptr) - base;
                if (run_calls == 0) {
                    run_start = next_img;
                    run_from = from;
                }
                run_to = to;
                run_calls += 1;
                // chat.zig only creates an image for a call with a prompt, so
                // the index walk has to skip the same ones or every card after
                // an empty call shows the wrong picture.
                if (c.prompt.len > 0 and next_img < n_images) {
                    run_len += 1;
                    next_img += 1;
                }
                rest = c.after;
            },
            .partial => |pp| {
                push.run(out, &n, run_start, run_len, run_calls, reply[run_from..run_to]);
                push.prose(out, &n, pp.text_before);
                break;
            },
            .none => {
                push.run(out, &n, run_start, run_len, run_calls, reply[run_from..run_to]);
                push.prose(out, &n, rest);
                break;
            },
        }
    }
    return out[0..n];
}

test "a reply splits into prose and call runs in document order" {
    var buf: [8]Segment = undefined;

    // Four back-to-back calls are ONE run: that is the 2x2 card.
    const four =
        "Here you go.\n" ++
        "<image>a</image>\n<image>b</image>\n<image>c</image>\n<image>d</image>\n" ++
        "Enjoy.";
    const s1 = segments(four, 4, &buf);
    try std.testing.expectEqual(@as(usize, 3), s1.len);
    try std.testing.expectEqualStrings("Here you go.\n", s1[0].prose);
    try std.testing.expectEqual(@as(usize, 0), s1[1].calls.start);
    try std.testing.expectEqual(@as(usize, 4), s1[1].calls.len);
    try std.testing.expectEqual(@as(usize, 4), s1[1].calls.n_calls);
    // The span covers every call in the run and nothing else.
    try std.testing.expectEqualStrings(
        "<image>a</image>\n<image>b</image>\n<image>c</image>\n<image>d</image>",
        s1[1].calls.text,
    );
    // The slice keeps the newline that followed the last `</image>`: prose is
    // passed through verbatim rather than trimmed, so nothing is silently
    // dropped from a reply. The markdown parser skips leading blank lines.
    try std.testing.expectEqualStrings("\nEnjoy.", s1[2].prose);

    // Talking between generations splits the run, and the SECOND card must map
    // to the second image, not the first.
    const talked =
        "<image>a</image>\nNow a variation.\n<image>b</image>";
    const s2 = segments(talked, 2, &buf);
    try std.testing.expectEqual(@as(usize, 3), s2.len);
    try std.testing.expectEqual(@as(usize, 0), s2[0].calls.start);
    try std.testing.expectEqual(@as(usize, 1), s2[0].calls.len);
    try std.testing.expectEqualStrings("\nNow a variation.\n", s2[1].prose);
    try std.testing.expectEqual(@as(usize, 1), s2[2].calls.start);
    try std.testing.expectEqual(@as(usize, 1), s2[2].calls.len);
}

test "plain prose and a bare call are each a single segment" {
    var buf: [8]Segment = undefined;
    const only_prose = segments("just talking", 0, &buf);
    try std.testing.expectEqual(@as(usize, 1), only_prose.len);
    try std.testing.expectEqualStrings("just talking", only_prose[0].prose);

    const only_call = segments("<image>a</image>", 1, &buf);
    try std.testing.expectEqual(@as(usize, 1), only_call.len);
    try std.testing.expectEqual(@as(usize, 1), only_call[0].calls.len);
    try std.testing.expectEqualStrings("<image>a</image>", only_call[0].calls.text);

    // Nothing at all produces nothing, rather than an empty prose block that
    // would draw a stray gap in the transcript.
    try std.testing.expectEqual(@as(usize, 0), segments("", 0, &buf).len);
    try std.testing.expectEqual(@as(usize, 0), segments("   \n ", 0, &buf).len);
}

test "an empty-prompt call creates no image, so it must not shift the indices" {
    var buf: [8]Segment = undefined;
    // chat.zig skips a call with no prompt; a naive walk would still advance
    // and the following card would show the wrong picture.
    const s = segments("<image></image>\n<image>real</image>", 1, &buf);
    try std.testing.expectEqual(@as(usize, 1), s.len);
    try std.testing.expectEqual(@as(usize, 0), s[0].calls.start);
    try std.testing.expectEqual(@as(usize, 1), s[0].calls.len);
    // Both calls are in the run's span even though only one made an image.
    try std.testing.expectEqual(@as(usize, 2), s[0].calls.n_calls);
}

test "a half-streamed call renders the prose before it and stops" {
    var buf: [8]Segment = undefined;
    const s = segments("thinking out loud\n<image>half", 0, &buf);
    try std.testing.expectEqual(@as(usize, 1), s.len);
    try std.testing.expectEqualStrings("thinking out loud\n", s[0].prose);
}

test "a run still reports its calls when no image survived" {
    var buf: [8]Segment = undefined;
    // A reopened conversation whose PNGs were never saved: the call is still
    // part of the reply and must remain visible, with no images behind it.
    const s = segments("done\n<image>a</image>\n<image>b</image>", 0, &buf);
    try std.testing.expectEqual(@as(usize, 2), s.len);
    try std.testing.expectEqualStrings("done\n", s[0].prose);
    try std.testing.expectEqual(@as(usize, 0), s[1].calls.len);
    try std.testing.expectEqual(@as(usize, 2), s[1].calls.n_calls);
    try std.testing.expectEqualStrings("<image>a</image>\n<image>b</image>", s[1].calls.text);
}
