//! Parse the tool calls a model EMITS, back into the typed `chat_template.ToolCall`
//! replayed on the next turn, and split a generated block into thought and answer.
//!
//! This is the read half of tool calling; `chat_template.zig` is the write half. The two
//! must be INVERSES: what `parse` returns for a block, handed back as a `ToolCall`, has
//! to re-render as that same block, or the model sees a garbled version of its own
//! request one turn later. The tests pin that round trip.
//!
//! Two wire formats, auto-detected on the first non-space byte, because the models this
//! engine runs do not agree. qwen3.5 and Bonsai are trained on an XML-ish form their own
//! chat template documents in the system prompt:
//!
//!     <tool_call>
//!     <function=get_weather>
//!     <parameter=city>
//!     Paris
//!     </parameter>
//!     </function>
//!     </tool_call>
//!
//! Plain qwen3 and most llama finetunes emit the Hermes/qwen2 JSON body instead:
//! `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
//!
//! K2 Horizon has its own tags (its template's default `xml` tool_call_format), all
//! calls of a turn inside one wrapper:
//!
//!     <ifm|tool_calls>
//!     <ifm|tool_call>get_weather
//!     <ifm|arg_key>city</ifm|arg_key>
//!     <ifm|arg_value>Paris</ifm|arg_value>
//!     </ifm|tool_call>
//!     </ifm|tool_calls>
//!
//! `nextBlock` yields one `<ifm|tool_call>` body at a time and folds the wrapper tags
//! into the block's span, so a display that hides blocks hides the wrapper too.
//!
//! `splitThought` / `answerText` / `endsInsideThought` live here rather than in the GUI
//! because both the CLI and the GUI scanners need "where does the answer start", and two
//! definitions of that is the drift that lets one caller fire a tool the other hides. A
//! caller must scan only the ANSWER, never the reasoning block: a model routinely writes
//! a tool call out while thinking about whether to make it.
//!
//! Pure string and JSON logic, std only, so it unit-tests without a model.

const std = @import("std");

// --- where a reply's ANSWER starts ------------------------------------------
// Lives here rather than in `gui/toolcall.zig` (where it grew up) because BOTH
// tool scanners need it and two definitions of "the answer" is exactly the drift
// that lets one caller fire a tool the other hides. `chat.Reasoning` aliases the
// type below, so a family's markers pass straight through with no conversion.

/// A family's reasoning-block delimiters as they appear in generated text.
pub const Reasoning = struct { open: []const u8, close: []const u8 };

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

const Close = struct { at: usize, len: usize };

fn findClose(body: []const u8, expected: []const u8) ?Close {
    var best: ?Close = if (std.mem.indexOf(u8, body, expected)) |at| .{ .at = at, .len = expected.len } else null;
    if (!std.mem.startsWith(u8, expected, "</ifm|think")) return best;
    for ([_][]const u8{ "</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>" }) |close| {
        const at = std.mem.indexOf(u8, body, close) orelse continue;
        if (best == null or at < best.?.at) best = .{ .at = at, .len = close.len };
    }
    return best;
}

/// Split a generated assistant turn around its reasoning block. ONE definition
/// shared by the display (the GUI's thought expander) and every tool-call
/// scanner, so a call that fires is exactly a call that's hidden from the reply.
///
/// `primed` is the whole subtlety, and getting it wrong is what broke
/// thinking for Qwen3.5/Bonsai in the GUI. The two prompt builders disagree
/// about who writes the OPENING marker:
///
///   - the hand glue (`llm/chat.zig openAssistant`) leaves a thinking turn
///     UNPRIMED, so the model emits its own `<think>` and the generated text
///     starts with it;
///   - a model's own jinja template (the render-driven path) PRIMES
///     `...assistant\n<think>\n` into the prompt, so the generated text starts
///     *inside* the block and the only marker it ever emits is the CLOSE.
///
/// With `primed` false against a primed prompt, nothing matches, the thought is
/// rendered as part of the answer with a stray `</think>` in the middle, and,
/// far worse, `answerText` returns the whole text, so a tool call the model
/// wrote *while reasoning about* whether to make it FIRES. That is precisely the
/// failure this guard exists to prevent, so `primed` is measured from the
/// rendered prompt (`endsInsideThought`) rather than assumed per family.
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
    if (findClose(body, rr.close)) |close| return .{
        .think = std.mem.trim(u8, body[0..close.at], ws),
        .answer = std.mem.trimStart(u8, body[close.at + close.len ..], ws),
        .open = false,
    };
    return .{ .think = std.mem.trimStart(u8, body, ws), .answer = "", .open = true };
}

/// Whether a RENDERED prompt ends inside an open reasoning block, i.e. whether
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
/// model's reasoning block closes (or the whole text if `r` is null, the family
/// doesn't reason, or thinking is off so no block is emitted). Tool calls are
/// scanned only here, NEVER inside the thought. See `splitThought` for `primed`.
pub fn answerText(text: []const u8, r: ?Reasoning, primed: bool) []const u8 {
    const s = splitThought(text, r, primed);
    if (s.think == null) return s.answer;
    return s.answer; // "" while the block is still open, nothing to scan yet
}

/// One parsed call. Owns both fields (`deinit`), because neither can be a slice
/// of the input in general: the XML form's arguments have to be *rebuilt* as
/// JSON, and the JSON form's have to be re-serialized out of the parsed doc.
pub const Call = struct {
    name: []const u8,
    /// The arguments as a JSON object, in JSON text, the exact shape
    /// `chat_template.ToolCall.arguments_json` takes.
    arguments_json: []const u8,

    pub fn deinit(self: *Call, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
        gpa.free(self.arguments_json);
    }
};

/// Result of scanning for the next `<tool_call>...</tool_call>` block.
///  - `.block`, a complete one: `text_before` (ordinary reply text), the raw
///    `body` to hand to `parse`, and `after` to keep scanning.
///  - `.partial`, an opened block still streaming: everything from it onward is
///    pending, so a live display can hide it and a turn-complete scanner stops.
///  - `.none`, no call remains; the whole buffer is text.
pub const Found = union(enum) {
    none,
    partial: struct { text_before: []const u8 },
    /// `k2` marks the K2 tag form, whose body carries no format marker of its
    /// own (a bare name line), so the scanner's knowledge picks the parser.
    block: struct { text_before: []const u8, body: []const u8, after: []const u8, k2: bool = false },
};

const open_tag = "<tool_call>";
const close_tag = "</tool_call>";
const k2_open = "<ifm|tool_call>";
const k2_close = "</ifm|tool_call>";
const k2_wrap_open = "<ifm|tool_calls>";
const k2_wrap_close = "</ifm|tool_calls>";

/// Find the next complete (or still-streaming) call block in `buf`. Pure.
///
/// Unlike `gui/toolcall.nextImageCall` this does NOT require the tag to start a
/// line: `<tool_call>` is a token the model was trained to emit, not a
/// convention invented by a prompt, and the templates themselves render it
/// mid-line (`...\n\n<tool_call>\n<function=...`). The reasoning-block guard is what
/// keeps a merely-contemplated call from firing here.
pub fn nextBlock(buf: []const u8) Found {
    const qa = std.mem.indexOf(u8, buf, open_tag);
    const ka = std.mem.indexOf(u8, buf, k2_open);
    const kw = std.mem.indexOf(u8, buf, k2_wrap_open);
    // K2's wrapper may have arrived before its first call: everything from it on
    // is pending.
    const k_first = if (ka) |k| (if (kw) |w| @min(w, k) else k) else kw;
    if (qa == null and k_first == null) return .none;
    if (k_first != null and (qa == null or k_first.? < qa.?)) return nextK2Block(buf, k_first.?);
    const a = qa.?;
    const body_start = a + open_tag.len;
    const b = std.mem.indexOfPos(u8, buf, body_start, close_tag) orelse
        return .{ .partial = .{ .text_before = buf[0..a] } };
    return .{ .block = .{
        .text_before = buf[0..a],
        .body = buf[body_start..b],
        .after = buf[b + close_tag.len ..],
    } };
}

/// The K2 form from `first` (the wrapper or a call tag, whichever came first):
/// the block spans a leading wrapper open and a trailing wrapper close when they
/// are adjacent to the call, so neither leaks into the surrounding text.
fn nextK2Block(buf: []const u8, first: usize) Found {
    const a = std.mem.indexOfPos(u8, buf, first, k2_open) orelse
        return .{ .partial = .{ .text_before = buf[0..first] } };
    const body_start = a + k2_open.len;
    const b = std.mem.indexOfPos(u8, buf, body_start, k2_close) orelse
        return .{ .partial = .{ .text_before = buf[0..first] } };
    var before_end = a;
    const lead = std.mem.trimEnd(u8, buf[0..a], " \t\r\n");
    if (std.mem.endsWith(u8, lead, k2_wrap_open)) before_end = lead.len - k2_wrap_open.len;
    var after_start = b + k2_close.len;
    const tail = std.mem.trimStart(u8, buf[after_start..], " \t\r\n");
    if (std.mem.startsWith(u8, tail, k2_wrap_close)) after_start = buf.len - tail.len + k2_wrap_close.len;
    return .{ .block = .{
        .text_before = buf[0..before_end],
        .body = buf[body_start..b],
        .after = buf[after_start..],
        .k2 = true,
    } };
}

/// One `<ifm|arg_key>`/`<ifm|arg_value>` pair of a K2 body, values as the raw text
/// between the tags. Pure, for callers that cannot allocate (the GUI's per-frame
/// display scan); `parse` re-types them.
pub const K2Arg = struct { key: []const u8, type_name: ?[]const u8, value: []const u8 };

pub const K2Args = struct {
    rest: []const u8,

    pub fn next(self: *K2Args) ?K2Arg {
        const ks = std.mem.indexOf(u8, self.rest, "<ifm|arg_key>") orelse return null;
        const key_start = ks + "<ifm|arg_key>".len;
        const key_end = std.mem.indexOfPos(u8, self.rest, key_start, "</ifm|arg_key>") orelse return null;
        var pos = key_end + "</ifm|arg_key>".len;
        var type_name: ?[]const u8 = null;
        const t = std.mem.trimStart(u8, self.rest[pos..], " \t\r\n");
        if (std.mem.startsWith(u8, t, "<ifm|arg_type>")) {
            const ts = self.rest.len - t.len + "<ifm|arg_type>".len;
            const te = std.mem.indexOfPos(u8, self.rest, ts, "</ifm|arg_type>") orelse return null;
            type_name = std.mem.trim(u8, self.rest[ts..te], " \t\r\n");
            pos = te + "</ifm|arg_type>".len;
        }
        const vs0 = std.mem.indexOfPos(u8, self.rest, pos, "<ifm|arg_value>") orelse return null;
        const vs = vs0 + "<ifm|arg_value>".len;
        const ve = std.mem.indexOfPos(u8, self.rest, vs, "</ifm|arg_value>") orelse self.rest.len;
        const arg: K2Arg = .{ .key = std.mem.trim(u8, self.rest[key_start..key_end], " \t\r\n"), .type_name = type_name, .value = self.rest[vs..ve] };
        self.rest = self.rest[@min(self.rest.len, ve + "</ifm|arg_value>".len)..];
        return arg;
    }
};

/// The function name a K2 body opens with (its first line), or null.
pub fn k2Name(body: []const u8) ?[]const u8 {
    const t = std.mem.trimStart(u8, body, " \t\r\n");
    const end = std.mem.indexOfAny(u8, t, "\r\n<") orelse t.len;
    const name = std.mem.trim(u8, t[0..end], " \t");
    return if (name.len == 0) null else name;
}

/// Iterate a K2 body's arguments (the text after the name line).
pub fn k2Args(body: []const u8) K2Args {
    return .{ .rest = body };
}

/// Whether `buf` holds any tool-call block, complete or streaming, in any form.
pub fn hasBlock(buf: []const u8) bool {
    return nextBlock(buf) != .none;
}

/// Parse one block body (what `Found.block.body` gives) into a `Call`.
/// Auto-detects the two formats: a body whose first non-space byte is `{` is the
/// JSON form, anything else the XML form.
pub fn parse(gpa: std.mem.Allocator, body: []const u8) !Call {
    const t = std.mem.trim(u8, body, " \t\r\n");
    if (t.len == 0) return error.EmptyToolCall;
    if (t[0] == '{') return parseJson(gpa, t);
    if (std.mem.indexOf(u8, t, "<ifm|arg_key>") != null) return parseK2(gpa, body);
    return parseXml(gpa, body);
}

/// The K2 form: `NAME` on the first line, then arg pairs. A declared
/// `<ifm|arg_type>` of string keeps the value verbatim; otherwise the value is
/// re-typed the way the qwen form is (JSON literal if it parses, else a string).
/// `parse` only reaches this on an arg tag; a no-argument call is a bare name
/// line, which only the scanner (`Found.block.k2`) can vouch for.
pub fn parseK2(gpa: std.mem.Allocator, body: []const u8) !Call {
    const name = k2Name(body) orelse return error.NoFunctionTag;
    var aw: std.Io.Writer.Allocating = try .initCapacity(gpa, 128);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{");
    var it = k2Args(body);
    var n: usize = 0;
    while (it.next()) |arg| {
        if (arg.key.len == 0) continue;
        if (n > 0) try w.writeAll(", ");
        try std.json.Stringify.encodeJsonString(arg.key, .{}, w);
        try w.writeAll(": ");
        const val = stripDelimiterNewlines(arg.value);
        if (arg.type_name != null and std.mem.eql(u8, arg.type_name.?, "string"))
            try std.json.Stringify.encodeJsonString(val, .{}, w)
        else
            try writeArgValue(gpa, w, val);
        n += 1;
    }
    try w.writeAll("}");
    const args = try aw.toOwnedSlice();
    errdefer gpa.free(args);
    return .{ .name = try gpa.dupe(u8, name), .arguments_json = args };
}

/// Every call in `buf`, appended to `out` (caller `deinit`s each). A block that
/// fails to parse is SKIPPED rather than aborting the scan, one malformed call
/// in a reply must not discard the well-formed ones beside it, and the reason
/// is returned via `bad` so a caller can tell the user.
pub fn parseAll(gpa: std.mem.Allocator, buf: []const u8, out: *std.ArrayList(Call), bad: ?*usize) !void {
    var rest = buf;
    while (true) {
        const blk = switch (nextBlock(rest)) {
            .none, .partial => return,
            .block => |b| b,
        };
        rest = blk.after;
        const c = (if (blk.k2) parseK2(gpa, blk.body) else parse(gpa, blk.body)) catch {
            if (bad) |p| p.* += 1;
            continue;
        };
        errdefer {
            var cc = c;
            cc.deinit(gpa);
        }
        try out.append(gpa, c);
    }
}

// --- the XML-ish form -------------------------------------------------------

/// `<function=NAME>` + zero or more `<parameter=KEY>VALUE</parameter>`.
fn parseXml(gpa: std.mem.Allocator, body: []const u8) !Call {
    const fs = std.mem.indexOf(u8, body, "<function=") orelse return error.NoFunctionTag;
    const name_start = fs + "<function=".len;
    const name_end = std.mem.indexOfScalarPos(u8, body, name_start, '>') orelse return error.NoFunctionTag;
    const name = std.mem.trim(u8, body[name_start..name_end], " \t\r\n");
    if (name.len == 0) return error.NoFunctionTag;

    var aw: std.Io.Writer.Allocating = try .initCapacity(gpa, 128);
    errdefer aw.deinit();
    const w = &aw.writer;
    try w.writeAll("{");

    var pos = name_end + 1;
    // A `</function>` bounds the parameter list, so a stray `<parameter=` in
    // trailing prose cannot be swept in. Absent (truncated generation), the rest
    // of the body is the list.
    const limit = std.mem.indexOfPos(u8, body, pos, "</function>") orelse body.len;
    var n: usize = 0;
    while (std.mem.indexOfPos(u8, body[0..limit], pos, "<parameter=")) |ps| {
        const key_start = ps + "<parameter=".len;
        const key_end = std.mem.indexOfScalarPos(u8, body[0..limit], key_start, '>') orelse break;
        const key = std.mem.trim(u8, body[key_start..key_end], " \t\r\n");
        const val_start = key_end + 1;
        const val_end = std.mem.indexOfPos(u8, body[0..limit], val_start, "</parameter>") orelse limit;
        const val = stripDelimiterNewlines(body[val_start..val_end]);
        if (key.len > 0) {
            if (n > 0) try w.writeAll(", ");
            try std.json.Stringify.encodeJsonString(key, .{}, w);
            try w.writeAll(": ");
            try writeArgValue(gpa, w, val);
            n += 1;
        }
        pos = @min(val_end + "</parameter>".len, limit);
        if (pos >= limit) break;
    }
    try w.writeAll("}");

    const args = try aw.toOwnedSlice();
    errdefer gpa.free(args);
    return .{ .name = try gpa.dupe(u8, name), .arguments_json = args };
}

/// Strip exactly the ONE newline the format puts on each side of a value,
/// never a full trim.
///
/// The template's own `args_value` writes `'\n' + value + '\n'`, so removing
/// one newline per side is its exact inverse; a `std.mem.trim` would additionally
/// eat leading indentation, which is real content for the multi-line values the
/// format explicitly supports ("a value that can span multiple lines"). Models
/// also emit the inline form `<parameter=k>v</parameter>`, which has no newlines
/// to strip and is left alone.
fn stripDelimiterNewlines(v: []const u8) []const u8 {
    var s = v;
    if (std.mem.startsWith(u8, s, "\r\n")) s = s[2..] else if (std.mem.startsWith(u8, s, "\n")) s = s[1..];
    if (std.mem.endsWith(u8, s, "\r\n")) s = s[0 .. s.len - 2] else if (std.mem.endsWith(u8, s, "\n")) s = s[0 .. s.len - 1];
    return s;
}

/// Re-type one parameter's raw text, inverting the template's `args_value`:
/// a value it stringified as JSON (dict/list/number/bool/null) comes back as
/// that JSON, anything else as a JSON string.
///
/// The inverse is AMBIGUOUS for a string that happens to look like JSON, an
/// argument whose value is literally the text `3` returns as the number 3. That
/// is a property of the wire format (it carries no types), not of this parser:
/// the template renders the string `3` and the number 3 identically, so no
/// reader can distinguish them. Declaring the parameter's type in the tool
/// schema is what a consumer has to rely on.
fn writeArgValue(gpa: std.mem.Allocator, w: *std.Io.Writer, raw: []const u8) !void {
    const t = std.mem.trim(u8, raw, " \t\r\n");
    if (t.len > 0 and isJsonScalarOrContainer(t)) {
        if (std.json.validate(gpa, t) catch false) {
            try w.writeAll(t);
            return;
        }
    }
    try std.json.Stringify.encodeJsonString(raw, .{}, w);
}

/// Cheap pre-filter so `json.validate` is only asked about text that could
/// plausibly be JSON, a bare word like `Paris` never reaches the allocator.
fn isJsonScalarOrContainer(t: []const u8) bool {
    return switch (t[0]) {
        '{', '[', '-', '0'...'9' => true,
        't' => std.mem.eql(u8, t, "true"),
        'f' => std.mem.eql(u8, t, "false"),
        'n' => std.mem.eql(u8, t, "null"),
        else => false,
    };
}

// --- the JSON form ----------------------------------------------------------

/// `{"name": "f", "arguments": {...}}`. `arguments` may also arrive as a JSON
/// string (OpenAI's wire shape, which some finetunes imitate); it is then
/// used verbatim once validated, so a nested object is not double-encoded.
fn parseJson(gpa: std.mem.Allocator, body: []const u8) !Call {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return error.MalformedToolCall;
    if (v != .object) return error.MalformedToolCall;
    const name_v = v.object.get("name") orelse return error.NoFunctionTag;
    if (name_v != .string) return error.NoFunctionTag;

    const args: []u8 = blk: {
        const av = v.object.get("arguments") orelse
            v.object.get("parameters") orelse
            break :blk try gpa.dupe(u8, "{}");
        switch (av) {
            .string => |s| {
                if (std.json.validate(a, s) catch false) break :blk try gpa.dupe(u8, s);
                // Not JSON at all: a single unnamed argument is not a shape the
                // typed side can replay, so refuse rather than invent a key.
                return error.MalformedToolCall;
            },
            .object => break :blk try std.json.Stringify.valueAlloc(gpa, av, .{}),
            else => return error.MalformedToolCall,
        }
    };
    errdefer gpa.free(args);
    return .{ .name = try gpa.dupe(u8, name_v.string), .arguments_json = args };
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;

const think_markers: Reasoning = .{ .open = "<think>", .close = "</think>" };

test "answerText: a call inside the reasoning block is excluded" {
    // The model wrote the call while reasoning; only the post-think answer scans.
    const txt = "<think>maybe <tool_call>a</tool_call>?</think>\nSure.\n<tool_call>b</tool_call>";
    const ans = answerText(txt, think_markers, false);
    try testing.expectEqualStrings("Sure.\n<tool_call>b</tool_call>", ans);
    try testing.expectEqualStrings("b", nextBlock(ans).block.body);
}

test "answerText: reasoning still open yields empty; null reasoning yields everything" {
    try testing.expectEqualStrings("", answerText("<think>weighing <tool_call>x</tool_call>", think_markers, false));
    try testing.expectEqualStrings("<tool_call>x</tool_call>", answerText("<tool_call>x</tool_call>", null, false));
}

test "answerText: gemma4-style channel markers" {
    const ch: Reasoning = .{ .open = "<|channel>thought", .close = "<channel|>" };
    const txt = "<|channel>thought skip <tool_call>a</tool_call><channel|>Here:\n<tool_call>b</tool_call>";
    try testing.expectEqualStrings("b", nextBlock(answerText(txt, ch, false)).block.body);
}

// --- primed thoughts (the render-driven template path) ----------------------
// The prompt ends `...assistant\n<think>\n`, so the generated text starts INSIDE
// the block and the only marker it emits is the CLOSE. This is what Qwen3.5 /
// Bonsai actually produce.

test "splitThought: primed turn with no opening marker folds the thought" {
    const txt = "Here's a thinking process:\n1. multiply.\n</think>\n\n17 x 4 = 68.";
    const s = splitThought(txt, think_markers, true);
    try testing.expectEqualStrings("Here's a thinking process:\n1. multiply.", s.think.?);
    try testing.expectEqualStrings("17 x 4 = 68.", s.answer);
    try testing.expect(!s.open);
}

test "splitThought: K2 accepts any effort close marker" {
    const low: Reasoning = .{ .open = "<ifm|think_faster>", .close = "</ifm|think_faster>" };
    const s = splitThought("brief\n</ifm|think>\nanswer", low, true);
    try testing.expectEqualStrings("brief", s.think.?);
    try testing.expectEqualStrings("answer", s.answer);
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

test "answerText: a primed thought's tool call must NOT reach the scanner" {
    // The serious half of the bug: a call written while reasoning would fire.
    const txt = "I could:\n<tool_call>a</tool_call>\n</think>\nSure!\n<tool_call>b</tool_call>";
    try testing.expectEqualStrings("b", nextBlock(answerText(txt, think_markers, true)).block.body);
    // Unprimed, the thought's call DOES leak through, which is the bug.
    try testing.expectEqualStrings("a", nextBlock(answerText(txt, think_markers, false)).block.body);
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
    // Thinking OFF: it primes a CLOSED empty block, not primed.
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

test "nextBlock: a complete call is split from the surrounding text" {
    const txt = "Sure, one moment.\n<tool_call>\n<function=f>\n</function>\n</tool_call>\nDone.";
    switch (nextBlock(txt)) {
        .block => |b| {
            try testing.expectEqualStrings("Sure, one moment.\n", b.text_before);
            try testing.expectEqualStrings("\n<function=f>\n</function>\n", b.body);
            try testing.expectEqualStrings("\nDone.", b.after);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "nextBlock: an unclosed call is partial, plain text is none" {
    switch (nextBlock("thinking...\n<tool_call>\n<function=get_wea")) {
        .partial => |p| try testing.expectEqualStrings("thinking...\n", p.text_before),
        else => return error.TestUnexpectedResult,
    }
    try testing.expectEqual(Found.none, nextBlock("no calls in here at all"));
}

// The exact block Bonsai's own template renders for the golden fixture case
// (see chat_template.zig's byte-exact test). Parsing it back must reproduce the
// arguments the template was given, that round trip is the whole contract.
const golden_body =
    \\
    \\<function=get_weather>
    \\<parameter=city>
    \\Paris
    \\</parameter>
    \\<parameter=opts>
    \\{"units": "c"}
    \\</parameter>
    \\<parameter=days>
    \\[1, 2]
    \\</parameter>
    \\<parameter=count>
    \\3
    \\</parameter>
    \\<parameter=verbose>
    \\true
    \\</parameter>
    \\<parameter=note>
    \\null
    \\</parameter>
    \\</function>
    \\
;

test "parse: the XML form round-trips the template's own rendering" {
    const gpa = testing.allocator;
    var c = try parse(gpa, golden_body);
    defer c.deinit(gpa);
    try testing.expectEqualStrings("get_weather", c.name);
    // Every value comes back with the type the template stringified it from,
    // a bare word as a string, everything else as the JSON it was written as.
    try testing.expectEqualStrings(
        \\{"city": "Paris", "opts": {"units": "c"}, "days": [1, 2], "count": 3, "verbose": true, "note": null}
    , c.arguments_json);
}

test "parse: a value's interior layout and newlines survive" {
    const gpa = testing.allocator;
    // Only the ONE delimiting newline per side is removed: the indentation of a
    // multi-line value is content (this is the format's documented use).
    var c = try parse(gpa,
        \\<function=write_file>
        \\<parameter=body>
        \\def f():
        \\    return 1
        \\</parameter>
        \\</function>
    );
    defer c.deinit(gpa);
    try testing.expectEqualStrings(
        \\{"body": "def f():\n    return 1"}
    , c.arguments_json);
}

test "parse: the inline (no-newline) spelling and a no-argument call" {
    const gpa = testing.allocator;
    var c = try parse(gpa, "<function=ping>\n<parameter=host>example.com</parameter>\n</function>");
    defer c.deinit(gpa);
    try testing.expectEqualStrings("ping", c.name);
    try testing.expectEqualStrings(
        \\{"host": "example.com"}
    , c.arguments_json);

    var d = try parse(gpa, "\n<function=now>\n</function>\n");
    defer d.deinit(gpa);
    try testing.expectEqualStrings("now", d.name);
    try testing.expectEqualStrings("{}", d.arguments_json);
}

test "parse: a JSON body (Hermes/qwen style) is detected and normalized" {
    const gpa = testing.allocator;
    var c = try parse(gpa,
        \\{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}
    );
    defer c.deinit(gpa);
    try testing.expectEqualStrings("get_weather", c.name);
    try testing.expectEqualStrings(
        \\{"city":"Paris","days":2}
    , c.arguments_json);

    // OpenAI's shape, where `arguments` is itself a JSON string: used verbatim
    // rather than double-encoded into a string-valued argument.
    var d = try parse(gpa,
        \\{"name": "f", "arguments": "{\"a\": 1}"}
    );
    defer d.deinit(gpa);
    try testing.expectEqualStrings(
        \\{"a": 1}
    , d.arguments_json);
}

test "parse: malformed bodies are refused, not guessed at" {
    const gpa = testing.allocator;
    try testing.expectError(error.EmptyToolCall, parse(gpa, "  \n "));
    try testing.expectError(error.NoFunctionTag, parse(gpa, "just some prose"));
    try testing.expectError(error.MalformedToolCall, parse(gpa, "{\"name\": \"f\", \"arguments\": 3}"));
    try testing.expectError(error.NoFunctionTag, parse(gpa, "{\"arguments\": {}}"));
}

test "parseAll: several calls in one reply; a broken one is counted, not fatal" {
    const gpa = testing.allocator;
    var out: std.ArrayList(Call) = .empty;
    defer {
        for (out.items) |*c| c.deinit(gpa);
        out.deinit(gpa);
    }
    var bad: usize = 0;
    try parseAll(gpa,
        "First:\n<tool_call>\n<function=a>\n</function>\n</tool_call>\n" ++
            "oops:\n<tool_call>\nnonsense\n</tool_call>\n" ++
            "then:\n<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        &out,
        &bad,
    );
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(@as(usize, 1), bad);
    try testing.expectEqualStrings("a", out.items[0].name);
    try testing.expectEqualStrings("b", out.items[1].name);
    try testing.expectEqualStrings("{\"x\": 1}", out.items[1].arguments_json);
}

test "K2 form: wrapper folded into the block, name line + arg pairs parsed" {
    const gpa = testing.allocator;
    const txt = "Sure.\n<ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_type>integer</ifm|arg_type>\n<ifm|arg_value>3</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>\nDone.";
    switch (nextBlock(txt)) {
        .block => |b| {
            try testing.expectEqualStrings("Sure.\n", b.text_before);
            try testing.expectEqualStrings("\nDone.", b.after);
            var c = try parse(gpa, b.body);
            defer c.deinit(gpa);
            try testing.expectEqualStrings("get_weather", c.name);
            try testing.expectEqualStrings("{\"city\": \"Paris\", \"days\": 3}", c.arguments_json);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "K2 form: a typed string that looks like a number stays a string" {
    const gpa = testing.allocator;
    var c = try parse(gpa, "f\n<ifm|arg_key>id</ifm|arg_key><ifm|arg_type>string</ifm|arg_type><ifm|arg_value>42</ifm|arg_value>");
    defer c.deinit(gpa);
    try testing.expectEqualStrings("{\"id\": \"42\"}", c.arguments_json);
}

test "K2 form: a wrapper without a finished call is partial" {
    switch (nextBlock("Let me draw that.\n<ifm|tool_calls>\n<ifm|tool_call>generate_image\n<ifm|arg_key>prom")) {
        .partial => |p| try testing.expectEqualStrings("Let me draw that.\n", p.text_before),
        else => return error.TestUnexpectedResult,
    }
    switch (nextBlock("text\n<ifm|tool_calls>")) {
        .partial => |p| try testing.expectEqualStrings("text\n", p.text_before),
        else => return error.TestUnexpectedResult,
    }
}

test "K2 form: two calls in one wrapper scan in sequence" {
    const gpa = testing.allocator;
    var out: std.ArrayList(Call) = .empty;
    defer {
        for (out.items) |*c| c.deinit(gpa);
        out.deinit(gpa);
    }
    try parseAll(gpa, "<ifm|tool_calls>\n<ifm|tool_call>a\n</ifm|tool_call>\n<ifm|tool_call>b\n<ifm|arg_key>x</ifm|arg_key>\n<ifm|arg_value>[1, 2]</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>", &out, null);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqualStrings("a", out.items[0].name);
    try testing.expectEqualStrings("{}", out.items[0].arguments_json);
    try testing.expectEqualStrings("{\"x\": [1, 2]}", out.items[1].arguments_json);
}
