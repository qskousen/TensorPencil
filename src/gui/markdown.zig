//! Streaming-tolerant markdown-subset parser for chat text.
//!
//! Pure string logic (std only) so it unit-tests cheaply — the dvui-facing
//! rendering lives in markdown_view.zig. The subset is what LLMs actually
//! emit: headings, fenced code, lists, blockquotes, horizontal rules, and
//! inline bold / italic / strikethrough / code / links.
//!
//! Chat text re-parses every frame while the model is still streaming, so
//! partial input must render sanely, never wrongly:
//!  - an unterminated inline marker (`**bol`) stays literal text — a span
//!    only opens once its closer is already in the buffer;
//!  - an unterminated code fence IS an open code block (`closed = false`),
//!    so streaming code displays as code live.
//!
//! Everything yields slices into the source — no allocation.
const std = @import("std");

// ---------------------------------------------------------------- blocks

pub const Heading = struct { level: u3, text: []const u8 };
pub const Code = struct { lang: []const u8, body: []const u8, closed: bool };
pub const ListItem = struct { indent: usize, number: ?u32, text: []const u8 };

pub const Block = union(enum) {
    /// Consecutive plain lines (may contain '\n' soft breaks); inline-styled.
    paragraph: []const u8,
    heading: Heading,
    /// Fenced code. `closed = false` while the fence is still streaming.
    code: Code,
    /// One `- item` / `1. item` line. `number == null` means a bullet.
    list_item: ListItem,
    /// Raw source slice of consecutive `> …` lines — the renderer strips the
    /// marker per line (`stripQuote`), keeping this module allocation-free.
    quote: []const u8,
    rule,
};

pub fn blocks(src: []const u8) BlockIterator {
    return .{ .src = src };
}

pub const BlockIterator = struct {
    src: []const u8,
    pos: usize = 0,

    pub fn next(self: *BlockIterator) ?Block {
        // Skip blank lines between blocks.
        while (self.pos < self.src.len) {
            const line = lineAt(self.src, self.pos);
            if (classify(line) != .blank) break;
            self.pos = lineEnd(self.src, self.pos);
        }
        if (self.pos >= self.src.len) return null;

        const line = lineAt(self.src, self.pos);
        switch (classify(line)) {
            .blank => unreachable,
            .fence => return self.fenceBlock(line),
            .heading => {
                self.pos = lineEnd(self.src, self.pos);
                const l = std.mem.trimStart(u8, line, " ");
                var level: u3 = 0;
                while (level < l.len and l[level] == '#') level += 1;
                return .{ .heading = .{
                    .level = level,
                    .text = std.mem.trim(u8, l[level..], " \t"),
                } };
            },
            .rule => {
                self.pos = lineEnd(self.src, self.pos);
                return .rule;
            },
            .list_item => {
                self.pos = lineEnd(self.src, self.pos);
                var indent: usize = 0;
                var i: usize = 0;
                while (i < line.len and (line[i] == ' ' or line[i] == '\t')) : (i += 1)
                    indent += if (line[i] == '\t') 4 else 1;
                if (line[i] == '-' or line[i] == '*' or line[i] == '+') {
                    return .{ .list_item = .{
                        .indent = indent,
                        .number = null,
                        .text = std.mem.trimStart(u8, line[i + 1 ..], " \t"),
                    } };
                }
                var num: u32 = 0;
                while (i < line.len and std.ascii.isDigit(line[i])) : (i += 1)
                    num = num *% 10 +% (line[i] - '0');
                return .{ .list_item = .{
                    .indent = indent,
                    .number = num,
                    .text = std.mem.trimStart(u8, line[i + 1 ..], " \t"),
                } };
            },
            .quote => {
                const start = self.pos;
                var end = self.pos;
                while (self.pos < self.src.len) {
                    const l = lineAt(self.src, self.pos);
                    if (classify(l) != .quote) break;
                    end = self.pos + l.len;
                    self.pos = lineEnd(self.src, self.pos);
                }
                return .{ .quote = self.src[start..end] };
            },
            .text => {
                const start = self.pos;
                var end = self.pos;
                while (self.pos < self.src.len) {
                    const l = lineAt(self.src, self.pos);
                    if (classify(l) != .text) break;
                    end = self.pos + l.len;
                    self.pos = lineEnd(self.src, self.pos);
                }
                return .{ .paragraph = self.src[start..end] };
            },
        }
    }

    fn fenceBlock(self: *BlockIterator, open_line: []const u8) Block {
        const opened = std.mem.trimStart(u8, open_line, " ");
        var ticks: usize = 0;
        while (ticks < opened.len and opened[ticks] == '`') ticks += 1;
        const lang = std.mem.trim(u8, opened[ticks..], " \t");
        const body_start = lineEnd(self.src, self.pos);
        self.pos = body_start;
        while (self.pos < self.src.len) {
            const l = lineAt(self.src, self.pos);
            const t = std.mem.trimStart(u8, l, " ");
            var n: usize = 0;
            while (n < t.len and t[n] == '`') n += 1;
            if (n >= ticks and std.mem.trimEnd(u8, t[n..], " \t").len == 0) {
                // Closing fence: body ends just before this line.
                const body_end = if (self.pos > body_start) self.pos - 1 else body_start;
                const body = self.src[body_start..body_end];
                self.pos = lineEnd(self.src, self.pos);
                return .{ .code = .{ .lang = lang, .body = body, .closed = true } };
            }
            self.pos = lineEnd(self.src, self.pos);
        }
        // Still streaming (or the model forgot to close): open code block.
        return .{ .code = .{ .lang = lang, .body = self.src[body_start..], .closed = false } };
    }
};

const LineClass = enum { blank, fence, heading, rule, list_item, quote, text };

/// The line beginning at `pos` (which must be a line start), without the '\n'.
fn lineAt(src: []const u8, pos: usize) []const u8 {
    const nl = std.mem.indexOfScalarPos(u8, src, pos, '\n') orelse src.len;
    return src[pos..nl];
}

/// Position of the next line start after the line beginning at `pos`.
fn lineEnd(src: []const u8, pos: usize) usize {
    const nl = std.mem.indexOfScalarPos(u8, src, pos, '\n') orelse return src.len;
    return nl + 1;
}

fn classify(line: []const u8) LineClass {
    const l = std.mem.trimStart(u8, line, " \t");
    if (l.len == 0) return .blank;
    if (std.mem.startsWith(u8, l, "```")) return .fence;
    if (l[0] == '>') return .quote;
    if (l[0] == '#') {
        var n: usize = 0;
        while (n < l.len and l[n] == '#') n += 1;
        if (n <= 6 and (n == l.len or l[n] == ' ')) return .heading;
        return .text;
    }
    if (isRule(l)) return .rule;
    // Bullet: `- x` / `* x` / `+ x` (marker must be followed by a space —
    // `*emphasis*` at line start is not a list).
    if ((l[0] == '-' or l[0] == '*' or l[0] == '+') and l.len >= 2 and (l[1] == ' ' or l[1] == '\t'))
        return .list_item;
    // Ordered: 1-9 digits then `.` or `)` then a space.
    var d: usize = 0;
    while (d < l.len and d < 9 and std.ascii.isDigit(l[d])) d += 1;
    if (d > 0 and d + 1 < l.len and (l[d] == '.' or l[d] == ')') and (l[d + 1] == ' ' or l[d + 1] == '\t'))
        return .list_item;
    return .text;
}

/// A horizontal rule: 3+ of the same `-` `_` `*`, nothing else but spaces.
fn isRule(l: []const u8) bool {
    const c = l[0];
    if (c != '-' and c != '_' and c != '*') return false;
    var n: usize = 0;
    for (l) |ch| {
        if (ch == c) n += 1 else if (ch != ' ' and ch != '\t') return false;
    }
    return n >= 3;
}

/// Strip the `>` marker (plus one optional following space) from one line of
/// a `.quote` block's source.
pub fn stripQuote(line: []const u8) []const u8 {
    var l = std.mem.trimStart(u8, line, " \t");
    if (l.len > 0 and l[0] == '>') l = l[1..];
    if (l.len > 0 and l[0] == ' ') l = l[1..];
    return l;
}

// ---------------------------------------------------------------- inlines

pub const Style = struct {
    bold: bool = false,
    italic: bool = false,
    code: bool = false,
    strike: bool = false,
};

pub const Span = struct {
    text: []const u8,
    style: Style = .{},
    /// Set for `[text](url)` and bare-URL autolinks.
    link: ?[]const u8 = null,
};

pub fn spans(text: []const u8) SpanIterator {
    return .{ .src = text };
}

pub const SpanIterator = struct {
    src: []const u8,
    pos: usize = 0,
    start: usize = 0, // pending literal run start
    style: Style = .{},
    queued: ?Span = null, // span to emit right after the pending flush

    pub fn next(self: *SpanIterator) ?Span {
        if (self.queued) |q| {
            self.queued = null;
            return q;
        }
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            switch (c) {
                '`' => if (self.codeSpan()) |s| return s,
                '*', '_' => if (self.emphasis(c)) |s| return s,
                '~' => if (self.strike()) |s| return s,
                '[' => if (self.link()) |s| return s,
                'h' => if (self.autolink()) |s| return s,
                '\\' => {
                    // Escaped punctuation: drop the backslash, keep the char.
                    if (self.pos + 1 < self.src.len and isPunct(self.src[self.pos + 1])) {
                        const flushed = self.take(self.pos);
                        self.start = self.pos + 1;
                        self.pos = self.pos + 2;
                        if (flushed) |s| return s;
                    } else self.pos += 1;
                },
                else => self.pos += 1,
            }
        }
        return self.take(self.src.len);
    }

    /// Emit the pending literal run [start..end) with the current style, if
    /// non-empty, and advance `start`.
    fn take(self: *SpanIterator, end: usize) ?Span {
        if (end <= self.start) return null;
        const s: Span = .{ .text = self.src[self.start..end], .style = self.style };
        self.start = end;
        return s;
    }

    /// Flush pending text up to `at`, queue `q` behind it, and jump to `to`.
    fn emit(self: *SpanIterator, at: usize, q: Span, to: usize) ?Span {
        const flushed = self.take(at);
        self.pos = to;
        self.start = to;
        if (flushed) |s| {
            self.queued = q;
            return s;
        }
        return q;
    }

    fn runLen(self: *const SpanIterator, at: usize, ch: u8) usize {
        var n: usize = 0;
        while (at + n < self.src.len and self.src[at + n] == ch) n += 1;
        return n;
    }

    /// `` `code` `` — the closer must be a backtick run of the same length.
    fn codeSpan(self: *SpanIterator) ?Span {
        const n = self.runLen(self.pos, '`');
        var j = self.pos + n;
        while (j < self.src.len) {
            if (self.src[j] == '`') {
                const k = self.runLen(j, '`');
                if (k == n) {
                    var st = self.style;
                    st.code = true;
                    return self.emit(self.pos, .{
                        .text = self.src[self.pos + n .. j],
                        .style = st,
                    }, j + k);
                }
                j += k;
            } else j += 1;
        }
        self.pos += n; // no closer yet: literal backticks
        return null;
    }

    fn isSpace(ch: u8) bool {
        return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r';
    }

    fn alnumAt(self: *const SpanIterator, at: usize) bool {
        return at < self.src.len and (std.ascii.isAlphanumeric(self.src[at]) or self.src[at] >= 0x80);
    }

    /// Can a delimiter run of `need` chars starting at `at` close a span?
    /// The char before it must be non-space; `_` additionally requires a
    /// word boundary after the run (protects snake_case).
    fn canClose(self: *const SpanIterator, at: usize, ch: u8, need: usize) bool {
        if (at == 0 or isSpace(self.src[at - 1])) return false;
        if (ch == '_' and self.alnumAt(at + need)) return false;
        return true;
    }

    /// Is there a closer for (ch, need) strictly after `from`? Spans only
    /// open when their closer is already buffered (streaming rule).
    fn hasCloser(self: *const SpanIterator, from: usize, ch: u8, need: usize) bool {
        var j = from;
        while (j < self.src.len) {
            if (self.src[j] == ch) {
                const k = self.runLen(j, ch);
                if (k >= need and j > from and self.canClose(j, ch, need)) return true;
                j += k;
            } else j += 1;
        }
        return false;
    }

    /// `**bold**` / `*italic*` / `__bold__` / `_italic_` (and `***both***`
    /// via consuming 2 then 1 from the same run).
    fn emphasis(self: *SpanIterator, ch: u8) ?Span {
        const run = self.runLen(self.pos, ch);
        var left = run;
        while (left > 0) {
            const need: usize = if (left >= 2) 2 else 1;
            const active = if (need == 2) self.style.bold else self.style.italic;
            const delim_at = self.pos;
            if (active and self.canClose(delim_at, ch, need)) {
                const flushed = self.take(delim_at);
                if (need == 2) self.style.bold = false else self.style.italic = false;
                self.pos = delim_at + need;
                self.start = self.pos;
                left -= need;
                if (flushed) |s| return s;
                continue;
            }
            // Opener: next char must be non-space, `_` must sit at a word
            // boundary, and a closer must already exist downstream.
            const after = delim_at + need;
            const opens = after < self.src.len and !isSpace(self.src[after]) and
                !(ch == '_' and delim_at > 0 and self.alnumAt(delim_at - 1)) and
                self.hasCloser(after, ch, need);
            if (opens) {
                const flushed = self.take(delim_at);
                if (need == 2) self.style.bold = true else self.style.italic = true;
                self.pos = after;
                self.start = self.pos;
                left -= need;
                if (flushed) |s| return s;
                continue;
            }
            // Literal delimiter(s): keep in the pending run.
            self.pos = delim_at + need;
            left -= need;
        }
        return null;
    }

    /// `~~strikethrough~~`.
    fn strike(self: *SpanIterator) ?Span {
        const run = self.runLen(self.pos, '~');
        if (run < 2) {
            self.pos += run;
            return null;
        }
        const delim_at = self.pos;
        if (self.style.strike and self.canClose(delim_at, '~', 2)) {
            const flushed = self.take(delim_at);
            self.style.strike = false;
            self.pos = delim_at + 2;
            self.start = self.pos;
            return flushed;
        }
        const after = delim_at + 2;
        if (after < self.src.len and !isSpace(self.src[after]) and self.hasCloser(after, '~', 2)) {
            const flushed = self.take(delim_at);
            self.style.strike = true;
            self.pos = after;
            self.start = self.pos;
            return flushed;
        }
        self.pos += run;
        return null;
    }

    /// `[text](url)` — label styling is not nested (rendered as link text).
    fn link(self: *SpanIterator) ?Span {
        const open = self.pos;
        const rb = std.mem.indexOfScalarPos(u8, self.src, open + 1, ']') orelse {
            self.pos += 1;
            return null;
        };
        if (rb + 1 >= self.src.len or self.src[rb + 1] != '(') {
            self.pos += 1;
            return null;
        }
        const rp = std.mem.indexOfScalarPos(u8, self.src, rb + 2, ')') orelse {
            self.pos += 1;
            return null;
        };
        const label = self.src[open + 1 .. rb];
        const url = std.mem.trim(u8, self.src[rb + 2 .. rp], " ");
        if (label.len == 0 or url.len == 0 or std.mem.indexOfScalar(u8, url, '\n') != null) {
            self.pos += 1;
            return null;
        }
        return self.emit(open, .{ .text = label, .style = self.style, .link = url }, rp + 1);
    }

    /// Bare `http(s)://…` URL at a word boundary, trailing punctuation
    /// excluded (`see https://x.org.` links `https://x.org`).
    fn autolink(self: *SpanIterator) ?Span {
        const at = self.pos;
        const rest = self.src[at..];
        const scheme: usize = if (std.mem.startsWith(u8, rest, "https://"))
            8
        else if (std.mem.startsWith(u8, rest, "http://"))
            7
        else {
            self.pos += 1;
            return null;
        };
        if (at > 0 and !isSpace(self.src[at - 1]) and self.src[at - 1] != '(') {
            self.pos += 1;
            return null;
        }
        var end = at + scheme;
        while (end < self.src.len and !isSpace(self.src[end]) and self.src[end] != '<' and self.src[end] != '"') end += 1;
        while (end > at + scheme and std.mem.indexOfScalar(u8, ").,;:!?'`", self.src[end - 1]) != null) end -= 1;
        if (end == at + scheme) {
            self.pos += 1;
            return null;
        }
        const url = self.src[at..end];
        return self.emit(at, .{ .text = url, .style = self.style, .link = url }, end);
    }
};

fn isPunct(ch: u8) bool {
    return switch (ch) {
        '!'...'/', ':'...'@', '['...'`', '{'...'~' => true,
        else => false,
    };
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

fn expectBlockCount(src: []const u8, n: usize) !void {
    var it = blocks(src);
    var count: usize = 0;
    while (it.next()) |_| count += 1;
    errdefer std.debug.print("expected {d} blocks, got {d} in \"{s}\"\n", .{ n, count, src });
    try testing.expectEqual(n, count);
}

test "blocks: paragraphs split on blank lines, soft breaks kept" {
    var it = blocks("first line\nstill first\n\nsecond");
    const a = it.next().?;
    try testing.expectEqualStrings("first line\nstill first", a.paragraph);
    const b = it.next().?;
    try testing.expectEqualStrings("second", b.paragraph);
    try testing.expectEqual(null, it.next());
}

test "blocks: heading levels and missing space" {
    var it = blocks("# Title\n### Sub\n#nospace");
    try testing.expectEqual(1, it.next().?.heading.level);
    const h = it.next().?;
    try testing.expectEqual(3, h.heading.level);
    try testing.expectEqualStrings("Sub", h.heading.text);
    // '#' without a space is ordinary text, not a heading.
    try testing.expectEqualStrings("#nospace", it.next().?.paragraph);
}

test "blocks: closed fence with language" {
    var it = blocks("intro\n```zig\nconst x = 1;\nconst y = 2;\n```\nafter");
    try testing.expectEqualStrings("intro", it.next().?.paragraph);
    const c = it.next().?.code;
    try testing.expectEqualStrings("zig", c.lang);
    try testing.expectEqualStrings("const x = 1;\nconst y = 2;", c.body);
    try testing.expect(c.closed);
    try testing.expectEqualStrings("after", it.next().?.paragraph);
}

test "blocks: streaming (unterminated) fence stays an open code block" {
    var it = blocks("```py\nprint(1)\nprint(2");
    const c = it.next().?.code;
    try testing.expectEqualStrings("py", c.lang);
    try testing.expectEqualStrings("print(1)\nprint(2", c.body);
    try testing.expect(!c.closed);
    try testing.expectEqual(null, it.next());
}

test "blocks: empty and just-opened fences" {
    var it = blocks("```\n```");
    const c = it.next().?.code;
    try testing.expectEqualStrings("", c.lang);
    try testing.expectEqualStrings("", c.body);
    try testing.expect(c.closed);
    // Fence opener alone (first streamed line).
    var it2 = blocks("```zig");
    const c2 = it2.next().?.code;
    try testing.expect(!c2.closed);
    try testing.expectEqualStrings("", c2.body);
}

test "blocks: bullets, ordered items, nesting indent" {
    var it = blocks("- one\n* two\n  - nested\n1. first\n2) second\n10. tenth");
    const a = it.next().?.list_item;
    try testing.expectEqual(null, a.number);
    try testing.expectEqualStrings("one", a.text);
    try testing.expectEqualStrings("two", it.next().?.list_item.text);
    const n = it.next().?.list_item;
    try testing.expectEqual(2, n.indent);
    try testing.expectEqualStrings("nested", n.text);
    const o1 = it.next().?.list_item;
    try testing.expectEqual(1, o1.number.?);
    try testing.expectEqualStrings("first", o1.text);
    try testing.expectEqual(2, it.next().?.list_item.number.?);
    try testing.expectEqual(10, it.next().?.list_item.number.?);
}

test "blocks: dash without space is text, not a bullet" {
    var it = blocks("-not a list\n--also not");
    try testing.expectEqualStrings("-not a list\n--also not", it.next().?.paragraph);
}

test "blocks: quote lines group into one block, stripQuote per line" {
    var it = blocks("> quoted a\n> quoted b\nplain");
    const q = it.next().?.quote;
    try testing.expectEqualStrings("> quoted a\n> quoted b", q);
    var lines = std.mem.splitScalar(u8, q, '\n');
    try testing.expectEqualStrings("quoted a", stripQuote(lines.next().?));
    try testing.expectEqualStrings("quoted b", stripQuote(lines.next().?));
    try testing.expectEqualStrings("plain", it.next().?.paragraph);
}

test "blocks: horizontal rules" {
    var it = blocks("---\n* * *\n___");
    try testing.expectEqual(Block.rule, it.next().?);
    try testing.expectEqual(Block.rule, it.next().?);
    try testing.expectEqual(Block.rule, it.next().?);
    try testing.expectEqual(null, it.next());
}

test "blocks: empty and whitespace-only input" {
    try expectBlockCount("", 0);
    try expectBlockCount("  \n\n \t\n", 0);
}

fn collectSpans(src: []const u8, buf: []Span) []Span {
    var it = spans(src);
    var n: usize = 0;
    while (it.next()) |s| : (n += 1) buf[n] = s;
    return buf[0..n];
}

test "spans: plain text is one span" {
    var buf: [8]Span = undefined;
    const s = collectSpans("hello world", &buf);
    try testing.expectEqual(1, s.len);
    try testing.expectEqualStrings("hello world", s[0].text);
    try testing.expectEqual(Style{}, s[0].style);
}

test "spans: bold, italic, and nesting" {
    var buf: [8]Span = undefined;
    const s = collectSpans("a **b *c* d** e", &buf);
    try testing.expectEqual(5, s.len);
    try testing.expectEqualStrings("a ", s[0].text);
    try testing.expect(s[1].style.bold and !s[1].style.italic);
    try testing.expectEqualStrings("b ", s[1].text);
    try testing.expect(s[2].style.bold and s[2].style.italic);
    try testing.expectEqualStrings("c", s[2].text);
    try testing.expect(s[3].style.bold and !s[3].style.italic);
    try testing.expectEqualStrings(" d", s[3].text);
    try testing.expectEqualStrings(" e", s[4].text);
    try testing.expectEqual(Style{}, s[4].style);
}

test "spans: triple asterisk is bold+italic" {
    var buf: [8]Span = undefined;
    const s = collectSpans("***both***", &buf);
    try testing.expectEqual(1, s.len);
    try testing.expect(s[0].style.bold and s[0].style.italic);
    try testing.expectEqualStrings("both", s[0].text);
}

test "spans: unterminated markers stay literal (streaming)" {
    var buf: [8]Span = undefined;
    // Mid-stream: "**bol" — the closer hasn't arrived, so no style flips on.
    const s = collectSpans("this is **bol", &buf);
    try testing.expectEqual(1, s.len);
    try testing.expectEqualStrings("this is **bol", s[0].text);
    try testing.expectEqual(Style{}, s[0].style);
    const t = collectSpans("`code without close", &buf);
    try testing.expectEqual(1, t.len);
    try testing.expectEqualStrings("`code without close", t[0].text);
}

test "spans: snake_case underscores stay literal" {
    var buf: [8]Span = undefined;
    const s = collectSpans("use foo_bar_baz here", &buf);
    try testing.expectEqual(1, s.len);
    try testing.expectEqualStrings("use foo_bar_baz here", s[0].text);
    // But a real underscore emphasis still works at word boundaries.
    const t = collectSpans("an _emphasized_ word", &buf);
    try testing.expectEqual(3, t.len);
    try testing.expect(t[1].style.italic);
    try testing.expectEqualStrings("emphasized", t[1].text);
}

test "spans: inline code, double-backtick, code wins over emphasis" {
    var buf: [8]Span = undefined;
    const s = collectSpans("run `zig build` now", &buf);
    try testing.expectEqual(3, s.len);
    try testing.expect(s[1].style.code);
    try testing.expectEqualStrings("zig build", s[1].text);
    const t = collectSpans("``a ` b``", &buf);
    try testing.expectEqual(1, t.len);
    try testing.expectEqualStrings("a ` b", t[0].text);
    // '*' inside code is literal.
    const u = collectSpans("`a * b` *i*", &buf);
    try testing.expect(u[0].style.code);
    try testing.expectEqualStrings("a * b", u[0].text);
    try testing.expect(u[2].style.italic);
}

test "spans: strikethrough" {
    var buf: [8]Span = undefined;
    const s = collectSpans("~~gone~~ kept", &buf);
    try testing.expectEqual(2, s.len);
    try testing.expect(s[0].style.strike);
    try testing.expectEqualStrings("gone", s[0].text);
    try testing.expectEqualStrings(" kept", s[1].text);
}

test "spans: markdown link and bare-URL autolink" {
    var buf: [8]Span = undefined;
    const s = collectSpans("see [the docs](https://ziglang.org/doc) ok", &buf);
    try testing.expectEqual(3, s.len);
    try testing.expectEqualStrings("the docs", s[1].text);
    try testing.expectEqualStrings("https://ziglang.org/doc", s[1].link.?);
    const t = collectSpans("go to https://example.com/x. Then stop.", &buf);
    try testing.expectEqual(3, t.len);
    try testing.expectEqualStrings("https://example.com/x", t[1].text);
    try testing.expectEqualStrings("https://example.com/x", t[1].link.?);
    try testing.expectEqualStrings(". Then stop.", t[2].text);
}

test "spans: bold inside a sentence keeps surrounding style clean" {
    var buf: [8]Span = undefined;
    const s = collectSpans("**Bold:** rest", &buf);
    try testing.expectEqual(2, s.len);
    try testing.expect(s[0].style.bold);
    try testing.expectEqualStrings("Bold:", s[0].text);
    try testing.expectEqualStrings(" rest", s[1].text);
    try testing.expectEqual(Style{}, s[1].style);
}

test "spans: escaped punctuation is literal" {
    var buf: [8]Span = undefined;
    const s = collectSpans("a \\*not italic\\* b", &buf);
    // Backslashes are dropped, asterisks kept, style never flips.
    var total: usize = 0;
    for (s) |sp| {
        try testing.expectEqual(Style{}, sp.style);
        total += sp.text.len;
    }
    try testing.expectEqual("a *not italic* b".len, total);
    try testing.expectEqualStrings("a ", s[0].text);
    try testing.expectEqualStrings("*not italic", s[1].text);
    try testing.expectEqualStrings("* b", s[2].text);
}

test "spans: multiplication asterisks stay literal" {
    var buf: [8]Span = undefined;
    // "3 * 4 * 5": openers are followed by spaces → never emphasis.
    const s = collectSpans("3 * 4 * 5", &buf);
    try testing.expectEqual(1, s.len);
    try testing.expectEqualStrings("3 * 4 * 5", s[0].text);
}

test "spans: empty input yields nothing" {
    var it = spans("");
    try testing.expectEqual(null, it.next());
}
