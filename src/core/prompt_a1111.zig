//! AUTOMATIC1111 prompt syntax: emphasis and per-step scheduling.
//!
//! A port of `modules/prompt_parser.py`, pinned against that file EXECUTED (see
//! tools/gen_a1111_prompt_fixtures.py, which downloads and runs it rather than
//! re-deriving it). This is the second prompt dialect here; `clip_tokenizer` has
//! ComfyUI's. They are not variations on a theme, and each difference changes the image:
//!
//!     (x:w)              ComfyUI replaces the weight; A1111 MULTIPLIES the range
//!     [x]                ComfyUI literal text;        A1111 1/1.1 de-emphasis
//!     BREAK              ComfyUI the literal word;    A1111 forces a chunk boundary
//!     75-token boundary  ComfyUI hard cut;            A1111 backtracks to last comma
//!
//! The first row surprises: because `:w` multiplies, `(((house:1.3))` is
//! `1.3 * 1.1 * 1.1 = 1.573`, not 1.3, and the two unclosed parens still apply because
//! A1111 flushes open brackets at end of input rather than discarding them. The
//! weighting is applied differently downstream too (`clip_text.applyWeightsA1111`):
//! A1111 multiplies the hidden states and restores the chunk's mean, where ComfyUI
//! interpolates away from the empty prompt's states per position.
//!
//! Two passes, in this order:
//!
//!     text --schedule(steps)--> per-step texts --parseAttention--> (text, weight) parts
//!
//! Scheduling runs FIRST and its output still contains emphasis brackets, exactly as
//! upstream does it. The other order resolves `[a|b]` after the brackets have already
//! been consumed as de-emphasis.
//!
//! Upstream parses the scheduling syntax with Lark's Earley parser over an ambiguous
//! grammar; this is recursive descent, so equivalence is MEASURED rather than assumed:
//! 283 schedule cases agree exactly, 43 hand-picked plus a 240-case seeded random corpus
//! over the stressing alphabet, at 4/10/20 steps. It rests on reproducing upstream's
//! FAILURE behaviour as well, since on any parse error it discards scheduling and
//! renders verbatim; `unschedulable` recognizes the two constraints that make real
//! prompts fail.

const std = @import("std");

/// One run of prompt text sharing a weight. `weight == break_marker` is not a weight at
/// all but a chunk-boundary marker, which is how upstream threads `BREAK` through.
pub const Part = struct {
    text: []const u8,
    weight: f32,

    /// Upstream's sentinel: `["BREAK", -1]`. Compared exactly, and paired with the
    /// text being exactly `BREAK`, because a *real* weight of -1 is legal
    /// (`(x:-1)` parses) and must not be mistaken for a boundary.
    pub fn isBreak(self: Part) bool {
        return self.weight == break_marker and std.mem.eql(u8, self.text, "BREAK");
    }
};

pub const break_marker: f32 = -1.0;

const round_mul: f64 = 1.1;
const square_mul: f64 = 1.0 / 1.1;

/// Parse A1111 emphasis into flat weighted parts. `arena` owns the returned text.
///
/// The algorithm is upstream's and its shape matters: weights are applied
/// retroactively by multiplying every part emitted since the opening bracket, so an
/// unbalanced `(` still weights everything after it (`"(unbalanced"` -> 1.1), and
/// brackets left open at end of input are flushed rather than dropped.
pub fn parseAttention(arena: std.mem.Allocator, text: []const u8) ![]Part {
    var res: std.ArrayList(Part) = .empty;
    var round: std.ArrayList(usize) = .empty;
    var square: std.ArrayList(usize) = .empty;

    // Weights accumulate in f64 and narrow once at the end: upstream multiplies Python
    // floats, and 1.1*1.1*1.3 differs in the last bits between f32 and f64.
    var w64: std.ArrayList(f64) = .empty;

    const multiplyRange = struct {
        fn f(ws: *std.ArrayList(f64), from: usize, m: f64) void {
            for (ws.items[from..]) |*x| x.* *= m;
        }
    }.f;

    var it: TokenIter = .{ .text = text };
    while (it.next()) |tk| {
        if (tk.kind == .escaped) {
            // `\(` -> `(`. The token includes the backslash; upstream appends `text[1:]`,
            // so a lone trailing backslash appends an empty part (kept, not skipped).
            try res.append(arena, .{ .text = tk.slice[1..], .weight = 1.0 });
            try w64.append(arena, 1.0);
        } else if (tk.kind == .open_round) {
            try round.append(arena, res.items.len);
        } else if (tk.kind == .open_square) {
            try square.append(arena, res.items.len);
        } else if (tk.kind == .weight and round.items.len > 0) {
            // MULTIPLIES the range, this is the ComfyUI difference.
            multiplyRange(&w64, round.pop().?, tk.weight);
        } else if (tk.kind == .close_round and round.items.len > 0) {
            multiplyRange(&w64, round.pop().?, round_mul);
        } else if (tk.kind == .close_square and square.items.len > 0) {
            multiplyRange(&w64, square.pop().?, square_mul);
        } else {
            // Plain text, and also a `)`/`]`/`:w)` with no matching opener, which
            // upstream falls through to here and keeps as literal text.
            var parts = breakSplit(tk.slice);
            var first = true;
            while (parts.next()) |piece| {
                if (!first) {
                    try res.append(arena, .{ .text = "BREAK", .weight = break_marker });
                    try w64.append(arena, break_marker);
                }
                first = false;
                try res.append(arena, .{ .text = piece, .weight = 1.0 });
                try w64.append(arena, 1.0);
            }
        }
    }

    // Brackets still open at end of input apply anyway.
    for (round.items) |pos| multiplyRange(&w64, pos, round_mul);
    for (square.items) |pos| multiplyRange(&w64, pos, square_mul);

    if (res.items.len == 0) {
        try res.append(arena, .{ .text = "", .weight = 1.0 });
        try w64.append(arena, 1.0);
    }

    // Narrow, then merge runs of equal weight (upstream concatenates their text).
    for (res.items, w64.items) |*p, w| p.weight = @floatCast(w);
    var i: usize = 0;
    while (i + 1 < res.items.len) {
        if (res.items[i].weight == res.items[i + 1].weight and
            !res.items[i].isBreak() and !res.items[i + 1].isBreak())
        {
            // The merged text is not contiguous in the input (escapes drop a
            // backslash, BREAK drops surrounding whitespace), so it must be built.
            res.items[i].text = try std.mem.concat(arena, u8, &.{ res.items[i].text, res.items[i + 1].text });
            _ = res.orderedRemove(i + 1);
        } else i += 1;
    }
    return res.toOwnedSlice(arena);
}

/// The tokens of upstream's `re_attention`, in its alternation order, which is also
/// the matching priority, since Python's `re` takes the first alternative that matches
/// at a position.
const Token = struct {
    const Kind = enum { escaped, open_round, open_square, weight, close_round, close_square, plain };
    kind: Kind,
    slice: []const u8,
    /// Only meaningful for `.weight`.
    weight: f64 = 0,
};

/// Hand-rolled equivalent of
/// `\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:`
/// scanned with `finditer`. Every byte is covered by some alternative, so this never
/// skips input.
const TokenIter = struct {
    text: []const u8,
    pos: usize = 0,

    fn next(self: *TokenIter) ?Token {
        if (self.pos >= self.text.len) return null;
        const s = self.text[self.pos..];

        // 1-6: a backslash escape, or a bare backslash.
        if (s[0] == '\\') {
            const n: usize = if (s.len >= 2 and (s[1] == '(' or s[1] == ')' or s[1] == '[' or s[1] == ']' or s[1] == '\\')) 2 else 1;
            self.pos += n;
            return .{ .kind = .escaped, .slice = s[0..n] };
        }
        // 7-8: an opening bracket.
        if (s[0] == '(') {
            self.pos += 1;
            return .{ .kind = .open_round, .slice = s[0..1] };
        }
        if (s[0] == '[') {
            self.pos += 1;
            return .{ .kind = .open_square, .slice = s[0..1] };
        }
        // 9: `:\s*([+-]?[.\d]+)\s*\)`, tried BEFORE the bare `:` and before `)`, and
        // it is the only alternative with a capture group.
        if (s[0] == ':') {
            if (matchWeight(s)) |m| {
                self.pos += m.len;
                return .{ .kind = .weight, .slice = s[0..m.len], .weight = m.value };
            }
        }
        // 10-11: a closing bracket.
        if (s[0] == ')') {
            self.pos += 1;
            return .{ .kind = .close_round, .slice = s[0..1] };
        }
        if (s[0] == ']') {
            self.pos += 1;
            return .{ .kind = .close_square, .slice = s[0..1] };
        }
        // 13: a lone colon (reached only when 9 did not match).
        if (s[0] == ':') {
            self.pos += 1;
            return .{ .kind = .plain, .slice = s[0..1] };
        }
        // 12: `[^\\()\[\]:]+`
        var n: usize = 0;
        while (n < s.len) : (n += 1) {
            switch (s[n]) {
                '\\', '(', ')', '[', ']', ':' => break,
                else => {},
            }
        }
        self.pos += n;
        return .{ .kind = .plain, .slice = s[0..n] };
    }

    const WeightMatch = struct { len: usize, value: f64 };

    /// `:\s*([+-]?[.\d]+)\s*\)`. The character class is `[.\d]`, so the numeric run
    /// may contain several dots (`1.2.3`), the regex matches it and `float()` then
    /// raises... except upstream never guards it, so a malformed number is a Python
    /// exception rather than a fallback. In practice the class only ever sees a real
    /// number; an unparseable run here declines the match and the `:` becomes plain
    /// text, which is the safe reading rather than a crash.
    fn matchWeight(s: []const u8) ?WeightMatch {
        var i: usize = 1; // past ':'
        while (i < s.len and isSpace(s[i])) i += 1;
        const num_start = i;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        const digits_start = i;
        while (i < s.len and (s[i] == '.' or (s[i] >= '0' and s[i] <= '9'))) i += 1;
        if (i == digits_start) return null; // `[.\d]+` needs at least one
        const num = s[num_start..i];
        while (i < s.len and isSpace(s[i])) i += 1;
        if (i >= s.len or s[i] != ')') return null;
        const v = std.fmt.parseFloat(f64, num) catch return null;
        return .{ .len = i + 1, .value = v };
    }

    fn isSpace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
    }
};

/// `re.split(/\s*\bBREAK\b\s*/, text)`, the pieces between `BREAK` occurrences, with
/// the surrounding whitespace consumed. The `\b` matters: `BREAKfast` is not a break.
const BreakSplit = struct {
    text: []const u8,
    pos: usize = 0,
    done: bool = false,

    fn next(self: *BreakSplit) ?[]const u8 {
        if (self.done) return null;
        var i = self.pos;
        while (i + 5 <= self.text.len) : (i += 1) {
            if (!std.mem.eql(u8, self.text[i..][0..5], "BREAK")) continue;
            // Word boundaries on both sides.
            if (i > 0 and isWord(self.text[i - 1])) continue;
            if (i + 5 < self.text.len and isWord(self.text[i + 5])) continue;
            // Whitespace before and after is part of the separator.
            var lo = i;
            while (lo > self.pos and TokenIter.isSpace(self.text[lo - 1])) lo -= 1;
            var hi = i + 5;
            while (hi < self.text.len and TokenIter.isSpace(self.text[hi])) hi += 1;
            const piece = self.text[self.pos..lo];
            self.pos = hi;
            return piece;
        }
        self.done = true;
        return self.text[self.pos..];
    }

    fn isWord(c: u8) bool {
        return c == '_' or (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
    }
};

fn breakSplit(text: []const u8) BreakSplit {
    return .{ .text = text };
}

// --- per-step scheduling ----------------------------------------------------

/// One span of the render: `text` is the flattened prompt for every step up to and
/// including `until` (1-based, inclusive) that no earlier entry covers.
pub const Entry = struct { until: usize, text: []const u8 };

/// Resolve `[a:b:when]` (prompt editing) and `[a|b]` (alternating words) into the
/// distinct prompts a render passes through, exactly as
/// `get_learned_conditioning_prompt_schedules` does. Always returns at least one entry,
/// whose `until` is `steps`.
///
/// This runs BEFORE `parseAttention`, and its output still contains emphasis
/// brackets, upstream's order. Resolving emphasis first would consume the `[` that
/// `[a|b]` needs.
///
/// `when` is a fraction of `steps` only when the literal contains a `.`, and an
/// integer is an absolute step: `[b:3]` is step 3 but `[b:.5]` is half way. So `[b:1]`
/// and `[b:1.0]` are different schedules, 1 vs `steps`. That is upstream's rule (its
/// non-hires branch) and it is not derivable from anything.
///
/// An `[a|b]` makes EVERY step its own entry (upstream's `collect_steps` extends
/// the set with `range(1, steps+1)`), so a 35-step render with one alternation yields 35
/// entries, of which only 2 are distinct texts. Deduplicate downstream by text, or a
/// render builds 35 conditionings for 2 prompts.
///
/// `arena` owns the returned text.
pub fn schedule(arena: std.mem.Allocator, text: []const u8, steps: usize) ![]Entry {
    // Upstream parses with Lark and, on ANY parse error, falls back to the whole prompt
    // verbatim. The one error a real prompt hits is a misplaced `|` (see
    // `unschedulable`), so that case is reproduced exactly rather than approximated.
    if (unschedulable(text)) {
        const one = try arena.alloc(Entry, 1);
        one[0] = .{ .until = steps, .text = text };
        return one;
    }

    var nodes: std.ArrayList(Node) = .empty;
    var p: Parser = .{ .arena = arena, .text = text, .steps = steps };
    try p.parseSeq(&nodes, text);

    // The step set: always `steps`, plus each scheduled boundary that lands at 1 or
    // later, plus every step when an alternation is present.
    var set: std.ArrayList(usize) = .empty;
    try set.append(arena, steps);
    try collectSteps(arena, nodes.items, steps, &set);
    std.mem.sort(usize, set.items, {}, std.sort.asc(usize));

    var out: std.ArrayList(Entry) = .empty;
    var last: ?usize = null;
    for (set.items) |t| {
        if (last != null and last.? == t) continue; // dedupe the sorted set
        last = t;
        var buf: std.ArrayList(u8) = .empty;
        try render(arena, nodes.items, t, &buf);
        try out.append(arena, .{ .until = t, .text = try buf.toOwnedSlice(arena) });
    }
    return out.toOwnedSlice(arena);
}

const Node = union(enum) {
    /// Verbatim, including any emphasis characters, `parseAttention` handles those.
    text: []const u8,
    /// A `(...)` or de-emphasis `[...]` span. Re-emitted with its delimiters, because the
    /// emphasis pass still has to see them.
    group: struct { open: u8, close: u8, children: []const Node },
    scheduled: struct { before: []const Node, after: []const Node, when: i64 },
    alternate: []const []const Node,
};

const Parser = struct {
    const Err = error{OutOfMemory};

    arena: std.mem.Allocator,
    text: []const u8,
    steps: usize,

    /// Parse `span` into nodes. Nested spans are parsed by slicing, so this only ever
    /// walks to the end of what it is given.
    ///
    /// The three parse functions carry an EXPLICIT error set: they are mutually
    /// recursive, and inferred sets across a cycle are a compile error
    /// ("dependency loop with length 3").
    fn parseSeq(self: *Parser, out: *std.ArrayList(Node), span: []const u8) Err!void {
        var i: usize = 0;
        var lit_start: usize = 0;
        while (i < span.len) {
            const c = span[i];
            // `plain` accepts `\\.`, so an escaped bracket is literal and must not
            // affect nesting.
            if (c == '\\') {
                i += if (i + 1 < span.len) 2 else 1;
                continue;
            }
            if (c != '[' and c != '(') {
                i += 1;
                continue;
            }
            const close: u8 = if (c == '[') ']' else ')';
            const end = matchClose(span, i) orelse {
                // Unbalanced: upstream's `!start` keeps stray `][():` verbatim.
                i += 1;
                continue;
            };
            if (i > lit_start) try out.append(self.arena, .{ .text = span[lit_start..i] });
            try self.parseBracket(out, span[i + 1 .. end], c, close);
            i = end + 1;
            lit_start = i;
        }
        if (lit_start < span.len) try out.append(self.arena, .{ .text = span[lit_start..] });
    }

    /// Classify a bracketed span. Only `[...]` can be scheduled or alternating; `(...)` is
    /// always an emphasis group (its `:w` is the emphasis pass's business).
    fn parseBracket(self: *Parser, out: *std.ArrayList(Node), inner: []const u8, open: u8, close: u8) Err!void {
        if (open == '[') {
            // Alternation wins: upstream's `prompt` rule has no `|` alternative, so a
            // top-level `|` cannot be part of an emphasis or scheduled span.
            var bars: std.ArrayList(usize) = .empty;
            try topLevel(self.arena, inner, '|', &bars);
            if (bars.items.len > 0) {
                var opts: std.ArrayList([]const Node) = .empty;
                var from: usize = 0;
                for (bars.items) |at| {
                    try opts.append(self.arena, try self.seq(inner[from..at]));
                    from = at + 1;
                }
                try opts.append(self.arena, try self.seq(inner[from..]));
                try out.append(self.arena, .{ .alternate = try opts.toOwnedSlice(self.arena) });
                return;
            }
            // Scheduled needs a trailing `:` NUMBER at top level. Without it the `:`
            // is just text inside a de-emphasis group (so `[a:b]` de-emphasizes "a:b").
            var colons: std.ArrayList(usize) = .empty;
            try topLevel(self.arena, inner, ':', &colons);
            if (colons.items.len > 0) {
                const last_colon = colons.items[colons.items.len - 1];
                if (parseWhen(inner[last_colon + 1 ..], self.steps)) |when| {
                    const has_before = colons.items.len >= 2;
                    const split = if (has_before) colons.items[colons.items.len - 2] else last_colon;
                    const before = if (has_before) try self.seq(inner[0..split]) else &[_]Node{};
                    const after = try self.seq(inner[if (has_before) split + 1 else 0 .. last_colon]);
                    try out.append(self.arena, .{ .scheduled = .{ .before = before, .after = after, .when = when } });
                    return;
                }
            }
        }
        try out.append(self.arena, .{ .group = .{ .open = open, .close = close, .children = try self.seq(inner) } });
    }

    fn seq(self: *Parser, span: []const u8) Err![]Node {
        var out: std.ArrayList(Node) = .empty;
        try self.parseSeq(&out, span);
        return out.toOwnedSlice(self.arena);
    }
};

/// Deepest bracket nesting `unschedulable` tracks. Far past any real prompt; deeper
/// than this is reported as unschedulable, which degrades to "no scheduling" rather
/// than to a wrong reading.
const max_depth: usize = 64;

/// Whether upstream's grammar would fail to parse this prompt at all, in which case it
/// renders verbatim with no scheduling.
///
/// The rule is about `|`, not about braces. The grammar gives `|` exactly one
/// home, separating the options of an `alternate`, i.e. directly inside a MATCHED
/// `[...]`. A `|` anywhere else fails to lex, and upstream's `except lark.LarkError`
/// turns the whole prompt into a single verbatim entry. Measured against the real
/// parser: `[a|b]` and `[[a|b]]` parse; `a|b`, `(a|b)`, `[(a|b)]`, `{a|b}`,
/// `a [b|c] d|e` and even `a [b|c` (unbalanced, so the `[` is literal text) all fail.
///
/// That last group matters in practice, NovelAI-style `{a|b}` prompts silently lose
/// all scheduling in A1111, and reproducing that is the difference between matching
/// the reference and being "reasonable".
///
/// This is also the one place a hand-written parser cannot be equivalent to Lark's
/// Earley parser in general; it is equivalent on the failure mode that actually occurs,
/// and the fixture corpus pins nine such cases.
fn unschedulable(text: []const u8) bool {
    return badSpan(text, false, 0);
}

/// Recognizer for the two constraints above. `alternation_allowed` is true exactly when
/// `span` is the direct interior of a matched `[...]`, the one context where `|` has a
/// grammar rule.
fn badSpan(span: []const u8, alternation_allowed: bool, depth: usize) bool {
    if (depth > max_depth) return true;

    if (hasTopLevel(span, '|')) {
        if (!alternation_allowed) return true;
        // Each option must be derivable as a `prompt`, and `prompt` has no bare-`:`
        // alternative, so a top-level `:` in any option kills the whole prompt.
        var from: usize = 0;
        var i: usize = 0;
        var d: isize = 0;
        while (i <= span.len) : (i += 1) {
            const at_end = i == span.len;
            if (!at_end) {
                switch (span[i]) {
                    '\\' => {
                        i += 1;
                        continue;
                    },
                    '[', '(' => {
                        d += 1;
                        continue;
                    },
                    ']', ')' => {
                        d -= 1;
                        continue;
                    },
                    '|' => if (d != 0) continue,
                    else => continue,
                }
            }
            const opt = span[from..i];
            if (hasTopLevel(opt, ':')) return true;
            if (badSpan(opt, false, depth + 1)) return true;
            from = i + 1;
        }
        return false;
    }

    // No alternation here: descend into each matched bracket. `[...]` opens an
    // alternation context, `(...)` does not.
    var i: usize = 0;
    while (i < span.len) : (i += 1) {
        switch (span[i]) {
            '\\' => i += 1,
            '[', '(' => {
                // An UNBALANCED bracket is literal text, not a nesting level, which is
                // why `a [b|c` fails: its `|` stays at the top level.
                if (matchClose(span, i)) |e| {
                    if (badSpan(span[i + 1 .. e], span[i] == '[', depth + 1)) return true;
                    i = e;
                }
            },
            else => {},
        }
    }
    return false;
}

/// Whether `needle` occurs at bracket-nesting depth zero in `span`, skipping escapes.
fn hasTopLevel(span: []const u8, needle: u8) bool {
    var d: isize = 0;
    var i: usize = 0;
    while (i < span.len) : (i += 1) {
        switch (span[i]) {
            '\\' => i += 1,
            '[', '(' => d += 1,
            ']', ')' => d -= 1,
            else => if (d == 0 and span[i] == needle) return true,
        }
    }
    return false;
}

/// Index of the `]`/`)` closing the bracket at `from`, honouring nesting of BOTH
/// bracket kinds and skipping `\\.` escapes. Null when unbalanced.
fn matchClose(span: []const u8, from: usize) ?usize {
    var depth_sq: isize = 0;
    var depth_rd: isize = 0;
    var i = from;
    while (i < span.len) {
        switch (span[i]) {
            '\\' => {
                i += 1;
            },
            '[' => depth_sq += 1,
            '(' => depth_rd += 1,
            ']' => {
                depth_sq -= 1;
                if (depth_sq == 0 and depth_rd == 0 and span[from] == '[') return i;
            },
            ')' => {
                depth_rd -= 1;
                if (depth_rd == 0 and depth_sq == 0 and span[from] == '(') return i;
            },
            else => {},
        }
        i += 1;
    }
    return null;
}

/// Positions of `needle` at nesting depth zero within `span`.
fn topLevel(arena: std.mem.Allocator, span: []const u8, needle: u8, out: *std.ArrayList(usize)) !void {
    var depth: isize = 0;
    var i: usize = 0;
    while (i < span.len) : (i += 1) {
        switch (span[i]) {
            '\\' => i += 1,
            '[', '(' => depth += 1,
            ']', ')' => depth -= 1,
            else => if (depth == 0 and span[i] == needle) try out.append(arena, i),
        }
    }
}

/// `[WHITESPACE] NUMBER [WHITESPACE]` filling the whole slice -> the resolved step.
///
/// The conversion is upstream's non-hires branch: a literal containing `.` is a
/// FRACTION of `steps`, an integer is absolute, then `min(steps, int(v))` with `int`
/// truncating toward zero.
fn parseWhen(s: []const u8, steps: usize) ?i64 {
    const t = std.mem.trim(u8, s, " \t\n\r\x0b\x0c");
    if (t.len == 0) return null;
    // Lark's `SIGNED_NUMBER`: an optional sign then digits with an optional fraction.
    var i: usize = 0;
    if (t[0] == '+' or t[0] == '-') i += 1;
    const ds = i;
    var dots: usize = 0;
    while (i < t.len) : (i += 1) {
        if (t[i] == '.') {
            dots += 1;
            if (dots > 1) return null;
        } else if (t[i] < '0' or t[i] > '9') return null;
    }
    if (i == ds or (dots == 1 and i == ds + 1)) return null; // needs a digit
    const v = std.fmt.parseFloat(f64, t) catch return null;
    const scaled = if (dots == 1) v * @as(f64, @floatFromInt(steps)) else v;
    const truncated: i64 = @intFromFloat(@trunc(scaled));
    return @min(@as(i64, @intCast(steps)), truncated);
}

fn collectSteps(arena: std.mem.Allocator, nodes: []const Node, steps: usize, out: *std.ArrayList(usize)) !void {
    for (nodes) |n| switch (n) {
        .text => {},
        .group => |g| try collectSteps(arena, g.children, steps, out),
        .scheduled => |s| {
            if (s.when >= 1) try out.append(arena, @intCast(s.when));
            try collectSteps(arena, s.before, steps, out);
            try collectSteps(arena, s.after, steps, out);
        },
        .alternate => |opts| {
            // Every step becomes its own entry.
            for (1..steps + 1) |t| try out.append(arena, t);
            for (opts) |o| try collectSteps(arena, o, steps, out);
        },
    };
}

fn render(arena: std.mem.Allocator, nodes: []const Node, step: usize, buf: *std.ArrayList(u8)) !void {
    for (nodes) |n| switch (n) {
        .text => |t| try buf.appendSlice(arena, t),
        .group => |g| {
            try buf.append(arena, g.open);
            try render(arena, g.children, step, buf);
            try buf.append(arena, g.close);
        },
        .scheduled => |s| try render(arena, if (@as(i64, @intCast(step)) <= s.when) s.before else s.after, step, buf),
        // `args[(step - 1) % len(args)]`, 1-based step.
        .alternate => |opts| try render(arena, opts[(step - 1) % opts.len], step, buf),
    };
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;

const AttnPart = struct { text: []const u8, weight: f32 };
const AttnCase = struct { text: []const u8, parts: []const AttnPart };
const A1111Fixtures = struct { a1111_attention: []const AttnCase };

test "A1111 emphasis parsing matches parse_prompt_attention" {
    // 22 cases from `tools/gen_a1111_prompt_fixtures.py`, which runs A1111's own
    // `parse_prompt_attention`, including every one of its upstream doctests.
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        A1111Fixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures_a1111.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    for (parsed.value.a1111_attention) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const got = try parseAttention(arena.allocator(), c.text);

        errdefer {
            std.debug.print("'{s}':\n  want:", .{c.text});
            for (c.parts) |p| std.debug.print(" ('{s}' {d})", .{ p.text, p.weight });
            std.debug.print("\n  got: ", .{});
            for (got) |p| std.debug.print(" ('{s}' {d})", .{ p.text, p.weight });
            std.debug.print("\n", .{});
        }
        try testing.expectEqual(c.parts.len, got.len);
        for (c.parts, got) |w, g| {
            try testing.expectEqualStrings(w.text, g.text);
            // Exact: both sides multiply the same literals in f64 and narrow once.
            try testing.expectEqual(w.weight, g.weight);
        }
    }
}

test "an A1111 weight multiplies the range rather than replacing it" {
    // The single most consequential difference from ComfyUI, stated as a property so it
    // cannot regress into "replace" without failing here: two unclosed parens around an
    // explicit 1.3 give 1.3 x 1.1 x 1.1, not 1.3.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const got = try parseAttention(arena.allocator(), "(((house:1.3))");
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqualStrings("house", got[0].text);
    try testing.expectApproxEqAbs(@as(f32, 1.573), got[0].weight, 1e-6);
}

test "square brackets de-emphasize, which ComfyUI reads as literal text" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const got = try parseAttention(arena.allocator(), "[quiet]");
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqualStrings("quiet", got[0].text);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 1.1), got[0].weight, 1e-6);
}

test "BREAK is a boundary marker, and BREAKfast is not" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    {
        const got = try parseAttention(arena.allocator(), "a BREAK b");
        try testing.expectEqual(@as(usize, 3), got.len);
        try testing.expectEqualStrings("a", got[0].text);
        try testing.expect(got[1].isBreak());
        try testing.expectEqualStrings("b", got[2].text);
    }
    {
        const got = try parseAttention(arena.allocator(), "BREAKfast");
        try testing.expectEqual(@as(usize, 1), got.len);
        try testing.expect(!got[0].isBreak());
        try testing.expectEqualStrings("BREAKfast", got[0].text);
    }
}

const SchedEntry = struct { until: usize, text: []const u8 };
const SchedCase = struct { text: []const u8, steps: usize, entries: []const SchedEntry };
const SchedFixtures = struct { a1111_schedule: []const SchedCase };

test "A1111 prompt scheduling matches get_learned_conditioning_prompt_schedules" {
    // 28 cases from A1111's own Lark-based scheduler, including every upstream doctest
    // (`a[b:[c:d:2]:1]e`, `[fe|||]male`, the unbalanced and stray-bracket cases) and the
    // module docstring's worked 100-step example.
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(
        SchedFixtures,
        gpa,
        @embedFile("assets/clip_tokenizer/fixtures_a1111.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    for (parsed.value.a1111_schedule) |c| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const got = try schedule(arena.allocator(), c.text, c.steps);

        errdefer {
            std.debug.print("'{s}' steps={d}:\n  want:", .{ c.text, c.steps });
            for (c.entries) |e| std.debug.print(" [{d} '{s}']", .{ e.until, e.text });
            std.debug.print("\n  got: ", .{});
            for (got) |e| std.debug.print(" [{d} '{s}']", .{ e.until, e.text });
            std.debug.print("\n", .{});
        }
        try testing.expectEqual(c.entries.len, got.len);
        for (c.entries, got) |w, g| {
            try testing.expectEqual(w.until, g.until);
            try testing.expectEqualStrings(w.text, g.text);
        }
    }
}

test "an integer `when` is an absolute step but a decimal is a fraction" {
    // `[b:1]` and `[b:1.0]` are DIFFERENT schedules, which is upstream's rule and the
    // easiest thing to get wrong by normalizing the literal before reading it.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const int_form = try schedule(a, "x [b:1]", 10);
    try testing.expectEqual(@as(usize, 2), int_form.len);
    try testing.expectEqual(@as(usize, 1), int_form[0].until);

    const flt_form = try schedule(a, "x [b:1.0]", 10);
    // 1.0 * 10 = 10 = steps, so the boundary coincides with the end: one entry.
    try testing.expectEqual(@as(usize, 1), flt_form.len);
    try testing.expectEqual(@as(usize, 10), flt_form[0].until);
}

test "scheduling runs before emphasis, so an alternation's brackets survive to it" {
    // The pass order, as a property: `[a|(b:1.2)]` must hand `(b:1.2)` to the emphasis
    // parser on even steps, if emphasis ran first, the `[`...`]` would already have been
    // eaten as de-emphasis and there would be no alternation left.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const sched = try schedule(a, "[a|(b:1.2)]", 4);
    try testing.expectEqual(@as(usize, 4), sched.len);
    try testing.expectEqualStrings("a", sched[0].text);
    try testing.expectEqualStrings("(b:1.2)", sched[1].text);

    const parts = try parseAttention(a, sched[1].text);
    try testing.expectEqual(@as(usize, 1), parts.len);
    try testing.expectEqualStrings("b", parts[0].text);
    try testing.expectApproxEqAbs(@as(f32, 1.2), parts[0].weight, 1e-6);
}

test "a prompt with no scheduling is one entry covering every step" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const got = try schedule(arena.allocator(), "1girl, (shiny skin:1.1), [blurry]", 35);
    try testing.expectEqual(@as(usize, 1), got.len);
    try testing.expectEqual(@as(usize, 35), got[0].until);
    try testing.expectEqualStrings("1girl, (shiny skin:1.1), [blurry]", got[0].text);
}
