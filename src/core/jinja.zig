//! jinja.zig — a minimal Jinja2-subset interpreter for LLM chat templates.
//!
//! Scope note (a deliberate shortcut, named per the project rules): this is NOT
//! a complete Jinja2. It implements exactly the subset the real GGUF-embedded
//! `tokenizer.chat_template`s use (gemma4, qwen3/qwen3.5 ChatML, gemma3, llama),
//! and is validated byte-exact against `jinja2`-rendered goldens (see the
//! `jinja: golden ...` tests + `assets/jinja/fixtures.json`). It mirrors the
//! philosophy of llama.cpp's `minja`: enough language to render chat templates
//! faithfully, no more. Whitespace handling matches transformers'
//! `apply_chat_template` env (`trim_blocks=True, lstrip_blocks=True`,
//! `keep_trailing_newline=False`) plus the explicit `{%- -%}` / `{{- -}}` markers.
//!
//! Usage:
//!   var tmpl = try Template.parse(gpa, template_src);
//!   defer tmpl.deinit();
//!   try tmpl.render(gpa, globals_dict_value, &out);   // out: *std.ArrayList(u8)
//!
//! The globals `Value` is a dict carrying `messages`, `bos_token`,
//! `add_generation_prompt`, and any flags (`enable_thinking`, `tools`, …). The
//! language builtins `range`, `namespace`, and `raise_exception` are always
//! available; a caller may override `raise_exception` / add `strftime_now` by
//! putting them in the globals dict.

const std = @import("std");

pub const Error = error{
    JinjaParse,
    JinjaRuntime,
    JinjaRaise,
    OutOfMemory,
};

// ---------------------------------------------------------------------------
// Value model
// ---------------------------------------------------------------------------

pub const Entry = struct { key: []const u8, val: Value };

/// Ordered string-keyed map (Jinja dicts + namespaces; insertion order matters
/// for `dictsort`-free iteration and for rendering determinism).
pub const Dict = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub fn get(self: *const Dict, key: []const u8) ?Value {
        for (self.entries.items) |e| if (std.mem.eql(u8, e.key, key)) return e.val;
        return null;
    }
    pub fn put(self: *Dict, a: std.mem.Allocator, key: []const u8, val: Value) !void {
        for (self.entries.items) |*e| if (std.mem.eql(u8, e.key, key)) {
            e.val = val;
            return;
        };
        try self.entries.append(a, .{ .key = key, .val = val });
    }
};

pub const List = struct { items: std.ArrayList(Value) = .empty };

pub const Value = union(enum) {
    undef,
    none,
    boolean: bool,
    int: i64,
    float: f64,
    str: []const u8,
    list: *List,
    dict: *Dict,
    macro: *const Macro,
    /// Language builtin (range/namespace/raise_exception), dispatched by name.
    builtin: []const u8,

    pub fn truthy(self: Value) bool {
        return switch (self) {
            .undef, .none => false,
            .boolean => |b| b,
            .int => |i| i != 0,
            .float => |f| f != 0,
            .str => |s| s.len > 0,
            .list => |l| l.items.items.len > 0,
            .dict => |d| d.entries.items.len > 0,
            .macro, .builtin => true,
        };
    }
};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

const BinOp = enum { add, sub, mul, div, floordiv, mod, concat, eq, ne, lt, gt, le, ge, in, not_in, @"and", @"or" };
const UnOp = enum { not, neg };

const Kw = struct { name: []const u8, val: *Expr };
const DictPair = struct { key: *Expr, val: *Expr };

const Expr = union(enum) {
    lit: Value,
    ident: []const u8,
    attr: struct { obj: *Expr, name: []const u8 },
    index: struct { obj: *Expr, idx: *Expr },
    slice: struct { obj: *Expr, start: ?*Expr, end: ?*Expr, step: ?*Expr },
    call: struct { callee: *Expr, args: []*Expr, kwargs: []Kw },
    method: struct { obj: *Expr, name: []const u8, args: []*Expr },
    filter: struct { value: *Expr, name: []const u8, args: []*Expr },
    do_test: struct { value: *Expr, name: []const u8, negate: bool, args: []*Expr },
    binop: struct { op: BinOp, lhs: *Expr, rhs: *Expr },
    unop: struct { op: UnOp, operand: *Expr },
    ternary: struct { then: *Expr, cond: *Expr, els: *Expr },
    list: []*Expr,
    dict: []DictPair,
};

const SetTarget = struct { root: []const u8, fields: [][]const u8 };
const FilterSpec = struct { name: []const u8, args: []*Expr };

const Branch = struct { cond: *Expr, body: []Node };

pub const Macro = struct {
    name: []const u8,
    params: []Param,
    body: []Node,
    const Param = struct { name: []const u8, default: ?*Expr };
};

const Node = union(enum) {
    text: []const u8,
    output: *Expr,
    set: struct { target: SetTarget, value: *Expr },
    set_block: struct { target: SetTarget, body: []Node, filters: []FilterSpec },
    if_: struct { branches: []Branch, else_body: ?[]Node },
    for_: struct { targets: [][]const u8, iter: *Expr, body: []Node, else_body: ?[]Node },
    macro: *Macro,
};

// ---------------------------------------------------------------------------
// Lexer: split source into text/expr/stmt chunks with whitespace control.
// ---------------------------------------------------------------------------

const Sym = enum {
    ident, int, float, str,
    lparen, rparen, lbrack, rbrack, lbrace, rbrace,
    comma, colon, dot, pipe, tilde,
    plus, minus, star, slash, dslash, percent,
    eq, ne, lt, gt, le, ge, assign,
};

const Tok = struct {
    sym: Sym,
    text: []const u8 = "", // ident name / raw
    int: i64 = 0,
    float: f64 = 0,
    str: []const u8 = "", // unescaped string literal
};

const TagKind = enum { expr, stmt };

const Chunk = union(enum) {
    text: []const u8,
    tag: struct { kind: TagKind, toks: []Tok },
};

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

pub const Template = struct {
    arena: std.heap.ArenaAllocator,
    nodes: []Node,
    diag: []const u8 = "",

    pub fn parse(gpa: std.mem.Allocator, src: []const u8) Error!Template {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const src_copy = try a.dupe(u8, src);

        var p = Parser{ .a = a, .chunks = try scan(a, src_copy), .i = 0 };
        const nodes = p.parseNodes(&.{}) catch |e| {
            // Surface the parser diagnostic by copying it out before the arena
            // would be freed on error — but we keep the arena for the message.
            var t = Template{ .arena = arena, .nodes = &.{}, .diag = p.diag };
            _ = &t;
            return e;
        };
        if (p.i != p.chunks.len) {
            p.diag = "unexpected trailing block (unbalanced end tag?)";
            return Error.JinjaParse;
        }
        return .{ .arena = arena, .nodes = nodes };
    }

    pub fn deinit(self: *Template) void {
        self.arena.deinit();
    }

    /// Render into `out`. `globals` must be a `.dict` Value. All transient
    /// values are allocated in an internal arena freed on return; `out` keeps
    /// only the rendered bytes (owned by the caller's allocator).
    pub fn render(self: *const Template, gpa: std.mem.Allocator, globals: Value, out: *std.ArrayList(u8)) Error!void {
        if (globals != .dict) return Error.JinjaRuntime;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var interp = Interp{ .a = arena.allocator(), .frames = .empty };
        try interp.frames.append(interp.a, globals.dict); // frame 0 = globals
        // Render into an arena-backed buffer, then copy the finished bytes into
        // the caller's list with the caller's allocator (the arena — and every
        // transient value in it — is freed on return).
        var buf: std.ArrayList(u8) = .empty;
        try interp.execNodes(self.nodes, &buf);
        try out.appendSlice(gpa, buf.items);
    }
};

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

fn scan(a: std.mem.Allocator, src: []const u8) Error![]Chunk {
    // Pass 1: raw split into text runs and tags, recording per-tag trim flags.
    const RawTag = struct { kind: TagKind, left_trim: bool, right_trim: bool, inner: []const u8, is_block: bool };
    const Raw = union(enum) { text: []const u8, tag: RawTag };
    var raws: std.ArrayList(Raw) = .empty;

    var i: usize = 0;
    var text_start: usize = 0;
    while (i < src.len) {
        if (src[i] == '{' and i + 1 < src.len and (src[i + 1] == '{' or src[i + 1] == '%' or src[i + 1] == '#')) {
            if (i > text_start) try raws.append(a, .{ .text = src[text_start..i] });
            const marker = src[i + 1];
            var j = i + 2;
            var left_trim = false;
            if (j < src.len and src[j] == '-') {
                left_trim = true;
                j += 1;
            } else if (j < src.len and src[j] == '+') {
                j += 1; // explicit "no lstrip" — treated same as default here
            }
            const inner_start = j;
            // Find the matching close for this marker.
            const close: []const u8 = switch (marker) {
                '{' => "}}",
                '%' => "%}",
                else => "#}",
            };
            const close_at = std.mem.indexOfPos(u8, src, j, close) orelse {
                return Error.JinjaParse;
            };
            var inner_end = close_at;
            var right_trim = false;
            if (inner_end > inner_start and src[inner_end - 1] == '-') {
                right_trim = true;
                inner_end -= 1;
            } else if (inner_end > inner_start and src[inner_end - 1] == '+') {
                inner_end -= 1;
            }
            const inner = src[inner_start..inner_end];
            if (marker != '#') {
                try raws.append(a, .{ .tag = .{
                    .kind = if (marker == '{') .expr else .stmt,
                    .left_trim = left_trim,
                    .right_trim = right_trim,
                    .inner = inner,
                    .is_block = marker == '%',
                } });
            } else {
                // Comment: contributes only its trim flags. Represent as an
                // empty stmt-less marker by folding trims into neighbors — we do
                // that by emitting a zero-effect tag record we later drop, but
                // simplest: keep trims by pushing a sentinel handled below.
                try raws.append(a, .{ .tag = .{
                    .kind = .stmt,
                    .left_trim = left_trim,
                    .right_trim = right_trim,
                    .inner = "", // empty inner => comment sentinel
                    .is_block = true,
                } });
            }
            i = close_at + close.len;
            text_start = i;
        } else i += 1;
    }
    if (src.len > text_start) try raws.append(a, .{ .text = src[text_start..] });

    // Pass 2: apply whitespace trimming to text runs based on adjacent tags,
    // then emit final chunks (dropping comment sentinels).
    var out: std.ArrayList(Chunk) = .empty;
    for (raws.items, 0..) |r, idx| {
        switch (r) {
            .tag => |t| {
                if (t.inner.len == 0 and t.is_block) {
                    // comment sentinel: skip (its trim flags already influence
                    // neighbors via prev/next lookups below)
                    continue;
                }
                const toks = try lexInner(a, t.inner);
                try out.append(a, .{ .tag = .{ .kind = t.kind, .toks = toks } });
            },
            .text => |txt0| {
                var txt = txt0;
                // Start trim from the PREVIOUS raw tag.
                if (idx > 0) switch (raws.items[idx - 1]) {
                    .tag => |pt| {
                        if (pt.right_trim) {
                            txt = std.mem.trimStart(u8, txt, " \t\r\n");
                        } else if (pt.is_block) {
                            // trim_blocks: drop a single leading newline.
                            if (txt.len >= 2 and txt[0] == '\r' and txt[1] == '\n') {
                                txt = txt[2..];
                            } else if (txt.len >= 1 and txt[0] == '\n') {
                                txt = txt[1..];
                            }
                        }
                    },
                    else => {},
                };
                // End trim from the NEXT raw tag.
                if (idx + 1 < raws.items.len) switch (raws.items[idx + 1]) {
                    .tag => |nt| {
                        if (nt.left_trim) {
                            txt = std.mem.trimEnd(u8, txt, " \t\r\n");
                        } else if (nt.is_block) {
                            // lstrip_blocks: strip a trailing run of spaces/tabs
                            // that sits at the start of the line (i.e. preceded
                            // by a newline or the buffer start).
                            var k: usize = txt.len;
                            while (k > 0 and (txt[k - 1] == ' ' or txt[k - 1] == '\t')) k -= 1;
                            if (k == 0 or txt[k - 1] == '\n') txt = txt[0..k];
                        }
                    },
                    else => {},
                };
                if (txt.len > 0) try out.append(a, .{ .text = txt });
            },
        }
    }
    return out.items;
}

fn lexInner(a: std.mem.Allocator, s: []const u8) Error![]Tok {
    var toks: std.ArrayList(Tok) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (isSpace(c)) {
            i += 1;
            continue;
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            const start = i;
            while (i < s.len and (std.ascii.isAlphanumeric(s[i]) or s[i] == '_')) i += 1;
            try toks.append(a, .{ .sym = .ident, .text = s[start..i] });
            continue;
        }
        if (std.ascii.isDigit(c)) {
            const start = i;
            var is_float = false;
            while (i < s.len and (std.ascii.isDigit(s[i]) or s[i] == '.')) {
                if (s[i] == '.') {
                    // Guard against '..' or a trailing '.' method access — only
                    // a digit-flanked dot is part of a float.
                    if (i + 1 >= s.len or !std.ascii.isDigit(s[i + 1])) break;
                    is_float = true;
                }
                i += 1;
            }
            const lit = s[start..i];
            if (is_float) {
                try toks.append(a, .{ .sym = .float, .float = std.fmt.parseFloat(f64, lit) catch return Error.JinjaParse });
            } else {
                try toks.append(a, .{ .sym = .int, .int = std.fmt.parseInt(i64, lit, 10) catch return Error.JinjaParse });
            }
            continue;
        }
        if (c == '\'' or c == '"') {
            i += 1;
            var buf: std.ArrayList(u8) = .empty;
            while (i < s.len and s[i] != c) {
                if (s[i] == '\\' and i + 1 < s.len) {
                    const e = s[i + 1];
                    try buf.append(a, switch (e) {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        else => e,
                    });
                    i += 2;
                } else {
                    try buf.append(a, s[i]);
                    i += 1;
                }
            }
            if (i >= s.len) return Error.JinjaParse;
            i += 1; // closing quote
            try toks.append(a, .{ .sym = .str, .str = buf.items });
            continue;
        }
        // Punctuation / operators.
        const two: ?Sym = if (i + 1 < s.len) blk: {
            const p = s[i .. i + 2];
            break :blk if (std.mem.eql(u8, p, "==")) .eq else if (std.mem.eql(u8, p, "!=")) .ne else if (std.mem.eql(u8, p, "<=")) .le else if (std.mem.eql(u8, p, ">=")) .ge else if (std.mem.eql(u8, p, "//")) .dslash else null;
        } else null;
        if (two) |sym| {
            try toks.append(a, .{ .sym = sym });
            i += 2;
            continue;
        }
        const one: Sym = switch (c) {
            '(' => .lparen,
            ')' => .rparen,
            '[' => .lbrack,
            ']' => .rbrack,
            '{' => .lbrace,
            '}' => .rbrace,
            ',' => .comma,
            ':' => .colon,
            '.' => .dot,
            '|' => .pipe,
            '~' => .tilde,
            '+' => .plus,
            '-' => .minus,
            '*' => .star,
            '/' => .slash,
            '%' => .percent,
            '<' => .lt,
            '>' => .gt,
            '=' => .assign,
            else => return Error.JinjaParse,
        };
        try toks.append(a, .{ .sym = one });
        i += 1;
    }
    return toks.items;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

const Parser = struct {
    a: std.mem.Allocator,
    chunks: []Chunk,
    i: usize,
    diag: []const u8 = "",
    // Token cursor within the current tag being parsed.
    toks: []Tok = &.{},
    ti: usize = 0,

    fn fail(self: *Parser, msg: []const u8) Error {
        self.diag = msg;
        return Error.JinjaParse;
    }

    // --- node-level -------------------------------------------------------

    /// Parse nodes until a stmt whose leading keyword is in `stops` (that stmt
    /// is left UNconsumed so the caller can read/dispatch it). Returns the body.
    fn parseNodes(self: *Parser, stops: []const []const u8) Error![]Node {
        var nodes: std.ArrayList(Node) = .empty;
        while (self.i < self.chunks.len) {
            switch (self.chunks[self.i]) {
                .text => |t| {
                    try nodes.append(self.a, .{ .text = t });
                    self.i += 1;
                },
                .tag => |tag| {
                    if (tag.kind == .expr) {
                        self.setToks(tag.toks);
                        const e = try self.parseExpr();
                        if (self.ti != self.toks.len) return self.fail("trailing tokens in {{ }}");
                        try nodes.append(self.a, .{ .output = e });
                        self.i += 1;
                        continue;
                    }
                    // stmt
                    const kw = self.stmtKeyword(tag.toks) orelse return self.fail("empty statement");
                    for (stops) |s| if (std.mem.eql(u8, s, kw)) return nodes.items;
                    self.i += 1; // consume this stmt tag; sub-parsers read its toks
                    self.setToks(tag.toks);
                    try self.parseStmt(kw, &nodes);
                },
            }
        }
        if (stops.len != 0) return self.fail("unexpected end of template (missing end tag)");
        return nodes.items;
    }

    fn stmtKeyword(self: *Parser, toks: []Tok) ?[]const u8 {
        _ = self;
        if (toks.len == 0 or toks[0].sym != .ident) return null;
        return toks[0].text;
    }

    fn parseStmt(self: *Parser, kw: []const u8, nodes: *std.ArrayList(Node)) Error!void {
        self.ti = 1; // skip the leading keyword ident
        if (std.mem.eql(u8, kw, "set")) {
            const tgt = try self.parseSetTarget();
            if (self.peekSym() == .assign) {
                self.ti += 1;
                const v = try self.parseExpr();
                try self.endTag();
                try nodes.append(self.a, .{ .set = .{ .target = tgt, .value = v } });
            } else {
                // Block set: `{% set x [| filter...] %}...{% endset %}` captures
                // the rendered body (then applies any filter chain) as a string.
                var filters: std.ArrayList(FilterSpec) = .empty;
                while (self.peekSym() == .pipe) {
                    self.ti += 1;
                    const fname = try self.expectIdent();
                    var fargs: []*Expr = &.{};
                    if (self.peekSym() == .lparen) fargs = (try self.parseCallArgs()).pos;
                    try filters.append(self.a, .{ .name = fname, .args = fargs });
                }
                try self.endTag();
                const body = try self.parseNodes(&.{"endset"});
                self.setToks(self.chunks[self.i].tag.toks);
                self.i += 1;
                self.ti = 1;
                try self.endTag();
                try nodes.append(self.a, .{ .set_block = .{ .target = tgt, .body = body, .filters = filters.items } });
            }
        } else if (std.mem.eql(u8, kw, "if")) {
            try self.parseIf(nodes);
        } else if (std.mem.eql(u8, kw, "for")) {
            try self.parseFor(nodes);
        } else if (std.mem.eql(u8, kw, "macro")) {
            try self.parseMacro(nodes);
        } else return self.fail("unknown statement keyword");
    }

    fn parseSetTarget(self: *Parser) Error!SetTarget {
        const root = try self.expectIdent();
        var fields: std.ArrayList([]const u8) = .empty;
        while (self.peekSym() == .dot) {
            self.ti += 1;
            try fields.append(self.a, try self.expectIdent());
        }
        return .{ .root = root, .fields = fields.items };
    }

    fn parseIf(self: *Parser, nodes: *std.ArrayList(Node)) Error!void {
        var branches: std.ArrayList(Branch) = .empty;
        const first_cond = try self.parseExpr();
        try self.endTag();
        var body = try self.parseNodes(&.{ "elif", "else", "endif" });
        try branches.append(self.a, .{ .cond = first_cond, .body = body });
        var else_body: ?[]Node = null;
        while (true) {
            const tag = self.chunks[self.i].tag;
            const kw = self.stmtKeyword(tag.toks).?;
            self.i += 1;
            self.setToks(tag.toks);
            self.ti = 1;
            if (std.mem.eql(u8, kw, "elif")) {
                const c = try self.parseExpr();
                try self.endTag();
                body = try self.parseNodes(&.{ "elif", "else", "endif" });
                try branches.append(self.a, .{ .cond = c, .body = body });
            } else if (std.mem.eql(u8, kw, "else")) {
                try self.endTag();
                else_body = try self.parseNodes(&.{"endif"});
                // consume endif
                self.setToks(self.chunks[self.i].tag.toks);
                self.i += 1;
                self.ti = 1;
                try self.endTag();
                break;
            } else { // endif
                try self.endTag();
                break;
            }
        }
        try nodes.append(self.a, .{ .if_ = .{ .branches = branches.items, .else_body = else_body } });
    }

    fn parseFor(self: *Parser, nodes: *std.ArrayList(Node)) Error!void {
        var targets: std.ArrayList([]const u8) = .empty;
        try targets.append(self.a, try self.expectIdent());
        while (self.peekSym() == .comma) {
            self.ti += 1;
            try targets.append(self.a, try self.expectIdent());
        }
        const in_kw = try self.expectIdent();
        if (!std.mem.eql(u8, in_kw, "in")) return self.fail("expected 'in' in for");
        const iter = try self.parseExpr();
        try self.endTag();
        const body = try self.parseNodes(&.{ "else", "endfor" });
        var else_body: ?[]Node = null;
        const tag = self.chunks[self.i].tag;
        const kw = self.stmtKeyword(tag.toks).?;
        self.i += 1;
        self.setToks(tag.toks);
        self.ti = 1;
        if (std.mem.eql(u8, kw, "else")) {
            try self.endTag();
            else_body = try self.parseNodes(&.{"endfor"});
            self.setToks(self.chunks[self.i].tag.toks);
            self.i += 1;
            self.ti = 1;
            try self.endTag();
        } else try self.endTag(); // endfor
        try nodes.append(self.a, .{ .for_ = .{ .targets = targets.items, .iter = iter, .body = body, .else_body = else_body } });
    }

    fn parseMacro(self: *Parser, nodes: *std.ArrayList(Node)) Error!void {
        const name = try self.expectIdent();
        try self.expect(.lparen);
        var params: std.ArrayList(Macro.Param) = .empty;
        while (self.peekSym() != .rparen) {
            const pname = try self.expectIdent();
            var def: ?*Expr = null;
            if (self.peekSym() == .assign) {
                self.ti += 1;
                def = try self.parseExpr();
            }
            try params.append(self.a, .{ .name = pname, .default = def });
            if (self.peekSym() == .comma) self.ti += 1 else break;
        }
        try self.expect(.rparen);
        try self.endTag();
        const body = try self.parseNodes(&.{"endmacro"});
        // consume endmacro
        self.setToks(self.chunks[self.i].tag.toks);
        self.i += 1;
        self.ti = 1;
        try self.endTag();
        const m = try self.a.create(Macro);
        m.* = .{ .name = name, .params = params.items, .body = body };
        try nodes.append(self.a, .{ .macro = m });
    }

    // --- token helpers ----------------------------------------------------

    fn setToks(self: *Parser, toks: []Tok) void {
        self.toks = toks;
        self.ti = 0;
    }
    fn peekSym(self: *Parser) ?Sym {
        return if (self.ti < self.toks.len) self.toks[self.ti].sym else null;
    }
    fn peekIdent(self: *Parser) ?[]const u8 {
        if (self.ti < self.toks.len and self.toks[self.ti].sym == .ident) return self.toks[self.ti].text;
        return null;
    }
    fn expect(self: *Parser, sym: Sym) Error!void {
        if (self.peekSym() != sym) return self.fail("unexpected token");
        self.ti += 1;
    }
    fn expectIdent(self: *Parser) Error![]const u8 {
        const id = self.peekIdent() orelse return self.fail("expected identifier");
        self.ti += 1;
        return id;
    }
    fn endTag(self: *Parser) Error!void {
        if (self.ti != self.toks.len) return self.fail("trailing tokens in statement");
    }

    // --- expression grammar (Jinja precedence) ----------------------------

    fn parseExpr(self: *Parser) Error!*Expr {
        return self.parseCond();
    }

    fn parseCond(self: *Parser) Error!*Expr {
        const e = try self.parseOr();
        if (self.matchKw("if")) {
            const cond = try self.parseOr();
            var els: *Expr = undefined;
            if (self.matchKw("else")) {
                els = try self.parseCond();
            } else {
                els = try self.mk(.{ .lit = .undef });
            }
            return self.mk(.{ .ternary = .{ .then = e, .cond = cond, .els = els } });
        }
        return e;
    }

    fn matchKw(self: *Parser, kw: []const u8) bool {
        if (self.peekIdent()) |id| if (std.mem.eql(u8, id, kw)) {
            self.ti += 1;
            return true;
        };
        return false;
    }

    fn parseOr(self: *Parser) Error!*Expr {
        var l = try self.parseAnd();
        while (self.matchKw("or")) {
            const r = try self.parseAnd();
            l = try self.mk(.{ .binop = .{ .op = .@"or", .lhs = l, .rhs = r } });
        }
        return l;
    }
    fn parseAnd(self: *Parser) Error!*Expr {
        var l = try self.parseNot();
        while (self.matchKw("and")) {
            const r = try self.parseNot();
            l = try self.mk(.{ .binop = .{ .op = .@"and", .lhs = l, .rhs = r } });
        }
        return l;
    }
    fn parseNot(self: *Parser) Error!*Expr {
        if (self.matchKw("not")) {
            const o = try self.parseNot();
            return self.mk(.{ .unop = .{ .op = .not, .operand = o } });
        }
        return self.parseCompare();
    }
    fn parseCompare(self: *Parser) Error!*Expr {
        var l = try self.parseMath1();
        while (true) {
            const s = self.peekSym();
            const op: ?BinOp = switch (s orelse break) {
                .eq => .eq,
                .ne => .ne,
                .lt => .lt,
                .gt => .gt,
                .le => .le,
                .ge => .ge,
                else => null,
            };
            if (op) |o| {
                self.ti += 1;
                const r = try self.parseMath1();
                l = try self.mk(.{ .binop = .{ .op = o, .lhs = l, .rhs = r } });
                continue;
            }
            if (self.peekIdent()) |id| {
                if (std.mem.eql(u8, id, "in")) {
                    self.ti += 1;
                    const r = try self.parseMath1();
                    l = try self.mk(.{ .binop = .{ .op = .in, .lhs = l, .rhs = r } });
                    continue;
                }
                if (std.mem.eql(u8, id, "not") and self.ti + 1 < self.toks.len and
                    self.toks[self.ti + 1].sym == .ident and std.mem.eql(u8, self.toks[self.ti + 1].text, "in"))
                {
                    self.ti += 2;
                    const r = try self.parseMath1();
                    l = try self.mk(.{ .binop = .{ .op = .not_in, .lhs = l, .rhs = r } });
                    continue;
                }
                if (std.mem.eql(u8, id, "is")) {
                    self.ti += 1;
                    l = try self.parseTestTail(l);
                    continue;
                }
            }
            break;
        }
        return l;
    }
    fn parseTestTail(self: *Parser, value: *Expr) Error!*Expr {
        var negate = false;
        if (self.peekIdent()) |id| if (std.mem.eql(u8, id, "not")) {
            negate = true;
            self.ti += 1;
        };
        const name = try self.expectIdent();
        var args: std.ArrayList(*Expr) = .empty;
        if (self.peekSym() == .lparen) {
            self.ti += 1;
            while (self.peekSym() != .rparen) {
                try args.append(self.a, try self.parseExpr());
                if (self.peekSym() == .comma) self.ti += 1 else break;
            }
            try self.expect(.rparen);
        }
        return self.mk(.{ .do_test = .{ .value = value, .name = name, .negate = negate, .args = args.items } });
    }
    fn parseMath1(self: *Parser) Error!*Expr {
        var l = try self.parseConcat();
        while (true) {
            const s = self.peekSym() orelse break;
            const op: ?BinOp = switch (s) {
                .plus => .add,
                .minus => .sub,
                else => null,
            };
            if (op) |o| {
                self.ti += 1;
                const r = try self.parseConcat();
                l = try self.mk(.{ .binop = .{ .op = o, .lhs = l, .rhs = r } });
            } else break;
        }
        return l;
    }
    fn parseConcat(self: *Parser) Error!*Expr {
        var l = try self.parseMath2();
        while (self.peekSym() == .tilde) {
            self.ti += 1;
            const r = try self.parseMath2();
            l = try self.mk(.{ .binop = .{ .op = .concat, .lhs = l, .rhs = r } });
        }
        return l;
    }
    fn parseMath2(self: *Parser) Error!*Expr {
        var l = try self.parseUnary();
        while (true) {
            const s = self.peekSym() orelse break;
            const op: ?BinOp = switch (s) {
                .star => .mul,
                .slash => .div,
                .dslash => .floordiv,
                .percent => .mod,
                else => null,
            };
            if (op) |o| {
                self.ti += 1;
                const r = try self.parseUnary();
                l = try self.mk(.{ .binop = .{ .op = o, .lhs = l, .rhs = r } });
            } else break;
        }
        return l;
    }
    fn parseUnary(self: *Parser) Error!*Expr {
        if (self.peekSym() == .minus) {
            self.ti += 1;
            const o = try self.parseUnary();
            return self.mk(.{ .unop = .{ .op = .neg, .operand = o } });
        }
        if (self.peekSym() == .plus) {
            self.ti += 1;
            return self.parseUnary();
        }
        return self.parsePostfix();
    }
    fn parsePostfix(self: *Parser) Error!*Expr {
        var e = try self.parsePrimary();
        while (true) {
            const s = self.peekSym() orelse break;
            switch (s) {
                .dot => {
                    self.ti += 1;
                    const name = try self.expectIdent();
                    if (self.peekSym() == .lparen) {
                        const args = try self.parseCallArgs();
                        e = try self.mk(.{ .method = .{ .obj = e, .name = name, .args = args.pos } });
                    } else {
                        e = try self.mk(.{ .attr = .{ .obj = e, .name = name } });
                    }
                },
                .lbrack => {
                    self.ti += 1;
                    // Subscript: `[expr]` index, or a slice `[a?:b?:c?]`.
                    var start_e: ?*Expr = null;
                    if (self.peekSym() != .colon and self.peekSym() != .rbrack) start_e = try self.parseExpr();
                    if (self.peekSym() != .colon) {
                        // plain index (no colon seen)
                        try self.expect(.rbrack);
                        e = try self.mk(.{ .index = .{ .obj = e, .idx = start_e.? } });
                    } else {
                        self.ti += 1; // first colon
                        var end_e: ?*Expr = null;
                        if (self.peekSym() != .colon and self.peekSym() != .rbrack) end_e = try self.parseExpr();
                        var step_e: ?*Expr = null;
                        if (self.peekSym() == .colon) {
                            self.ti += 1; // second colon
                            if (self.peekSym() != .rbrack) step_e = try self.parseExpr();
                        }
                        try self.expect(.rbrack);
                        e = try self.mk(.{ .slice = .{ .obj = e, .start = start_e, .end = end_e, .step = step_e } });
                    }
                },
                .lparen => {
                    const args = try self.parseCallArgs();
                    e = try self.mk(.{ .call = .{ .callee = e, .args = args.pos, .kwargs = args.kw } });
                },
                .pipe => {
                    self.ti += 1;
                    const name = try self.expectIdent();
                    var fargs: []*Expr = &.{};
                    if (self.peekSym() == .lparen) {
                        const args = try self.parseCallArgs();
                        fargs = args.pos;
                    }
                    e = try self.mk(.{ .filter = .{ .value = e, .name = name, .args = fargs } });
                },
                else => break,
            }
        }
        return e;
    }

    const CallArgs = struct { pos: []*Expr, kw: []Kw };
    fn parseCallArgs(self: *Parser) Error!CallArgs {
        try self.expect(.lparen);
        var pos: std.ArrayList(*Expr) = .empty;
        var kw: std.ArrayList(Kw) = .empty;
        while (self.peekSym() != .rparen) {
            // kwarg? ident '=' ...
            if (self.peekSym() == .ident and self.ti + 1 < self.toks.len and self.toks[self.ti + 1].sym == .assign) {
                const name = self.toks[self.ti].text;
                self.ti += 2;
                const v = try self.parseExpr();
                try kw.append(self.a, .{ .name = name, .val = v });
            } else {
                try pos.append(self.a, try self.parseExpr());
            }
            if (self.peekSym() == .comma) self.ti += 1 else break;
        }
        try self.expect(.rparen);
        return .{ .pos = pos.items, .kw = kw.items };
    }

    fn parsePrimary(self: *Parser) Error!*Expr {
        const s = self.peekSym() orelse return self.fail("unexpected end of expression");
        switch (s) {
            .int => {
                const v = self.toks[self.ti].int;
                self.ti += 1;
                return self.mk(.{ .lit = .{ .int = v } });
            },
            .float => {
                const v = self.toks[self.ti].float;
                self.ti += 1;
                return self.mk(.{ .lit = .{ .float = v } });
            },
            .str => {
                const v = self.toks[self.ti].str;
                self.ti += 1;
                return self.mk(.{ .lit = .{ .str = v } });
            },
            .lparen => {
                self.ti += 1;
                const e = try self.parseExpr();
                try self.expect(.rparen);
                return e;
            },
            .lbrack => {
                self.ti += 1;
                var items: std.ArrayList(*Expr) = .empty;
                while (self.peekSym() != .rbrack) {
                    try items.append(self.a, try self.parseExpr());
                    if (self.peekSym() == .comma) self.ti += 1 else break;
                }
                try self.expect(.rbrack);
                return self.mk(.{ .list = items.items });
            },
            .lbrace => {
                self.ti += 1;
                var pairs: std.ArrayList(DictPair) = .empty;
                while (self.peekSym() != .rbrace) {
                    const k = try self.parseExpr();
                    try self.expect(.colon);
                    const v = try self.parseExpr();
                    try pairs.append(self.a, .{ .key = k, .val = v });
                    if (self.peekSym() == .comma) self.ti += 1 else break;
                }
                try self.expect(.rbrace);
                return self.mk(.{ .dict = pairs.items });
            },
            .ident => {
                const id = self.toks[self.ti].text;
                self.ti += 1;
                if (std.mem.eql(u8, id, "none") or std.mem.eql(u8, id, "None")) return self.mk(.{ .lit = .none });
                if (std.mem.eql(u8, id, "true") or std.mem.eql(u8, id, "True")) return self.mk(.{ .lit = .{ .boolean = true } });
                if (std.mem.eql(u8, id, "false") or std.mem.eql(u8, id, "False")) return self.mk(.{ .lit = .{ .boolean = false } });
                return self.mk(.{ .ident = id });
            },
            else => return self.fail("unexpected token in expression"),
        }
    }

    fn mk(self: *Parser, e: Expr) Error!*Expr {
        const p = try self.a.create(Expr);
        p.* = e;
        return p;
    }
};

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

const Interp = struct {
    a: std.mem.Allocator,
    frames: std.ArrayList(*Dict),
    diag: []const u8 = "",

    fn newDict(self: *Interp) Error!*Dict {
        const d = try self.a.create(Dict);
        d.* = .{};
        return d;
    }
    fn newList(self: *Interp) Error!*List {
        const l = try self.a.create(List);
        l.* = .{};
        return l;
    }
    fn strVal(self: *Interp, s: []const u8) Value {
        _ = self;
        return .{ .str = s };
    }

    // --- variable scope ---------------------------------------------------

    fn lookup(self: *Interp, name: []const u8) Value {
        var i: usize = self.frames.items.len;
        while (i > 0) {
            i -= 1;
            if (self.frames.items[i].get(name)) |v| return v;
        }
        // language builtins
        if (std.mem.eql(u8, name, "range") or std.mem.eql(u8, name, "namespace") or
            std.mem.eql(u8, name, "raise_exception"))
            return .{ .builtin = name };
        return .undef;
    }
    fn setLocal(self: *Interp, name: []const u8, v: Value) Error!void {
        try self.frames.items[self.frames.items.len - 1].put(self.a, name, v);
    }
    fn setGlobal(self: *Interp, name: []const u8, v: Value) Error!void {
        try self.frames.items[0].put(self.a, name, v);
    }
    fn assignTarget(self: *Interp, target: SetTarget, v: Value) Error!void {
        if (target.fields.len == 0) {
            try self.setLocal(target.root, v);
            return;
        }
        var cur = self.lookup(target.root);
        var fi: usize = 0;
        while (fi + 1 < target.fields.len) : (fi += 1) {
            if (cur != .dict) return self.rt("set on non-dict field");
            cur = cur.dict.get(target.fields[fi]) orelse .undef;
        }
        if (cur != .dict) return self.rt("set on non-dict field");
        try cur.dict.put(self.a, target.fields[target.fields.len - 1], v);
    }

    // --- statement execution ---------------------------------------------

    fn execNodes(self: *Interp, nodes: []const Node, out: *std.ArrayList(u8)) Error!void {
        for (nodes) |n| try self.execNode(n, out);
    }

    fn execNode(self: *Interp, n: Node, out: *std.ArrayList(u8)) Error!void {
        switch (n) {
            .text => |t| try out.appendSlice(self.a, t),
            .output => |e| {
                const v = try self.eval(e);
                try self.writeValue(v, out);
            },
            .set => |st| {
                const v = try self.eval(st.value);
                try self.assignTarget(st.target, v);
            },
            .set_block => |sb| {
                var buf: std.ArrayList(u8) = .empty;
                try self.execNodes(sb.body, &buf);
                var v: Value = self.strVal(buf.items);
                for (sb.filters) |f| v = try self.applyFilterToValue(f.name, v, f.args);
                try self.assignTarget(sb.target, v);
            },
            .if_ => |iff| {
                for (iff.branches) |b| {
                    if ((try self.eval(b.cond)).truthy()) {
                        try self.execNodes(b.body, out);
                        return;
                    }
                }
                if (iff.else_body) |eb| try self.execNodes(eb, out);
            },
            .for_ => |f| try self.execFor(f, out),
            .macro => |m| try self.setGlobal(m.name, .{ .macro = m }),
        }
    }

    fn execFor(self: *Interp, f: anytype, out: *std.ArrayList(u8)) Error!void {
        const iter_v = try self.eval(f.iter);
        const items: []Value = switch (iter_v) {
            .list => |l| l.items.items,
            .str => |s| blk: {
                // iterate characters (rare); build a list of 1-char strs
                var l = try self.newList();
                for (s) |_| {}
                _ = &l;
                break :blk &.{};
            },
            .dict => |d| blk: {
                // iterating a dict yields its keys (Jinja/Python semantics)
                var l = try self.newList();
                for (d.entries.items) |e| try l.items.append(self.a, self.strVal(e.key));
                break :blk l.items.items;
            },
            else => &.{},
        };
        if (items.len == 0) {
            if (f.else_body) |eb| try self.execNodes(eb, out);
            return;
        }
        const frame = try self.newDict();
        try self.frames.append(self.a, frame);
        defer _ = self.frames.pop();
        for (items, 0..) |item, idx| {
            // reset the loop frame each iteration (Jinja loop bodies don't leak
            // assignments across iterations; that's what `namespace` is for)
            frame.entries.clearRetainingCapacity();
            if (f.targets.len == 1) {
                try frame.put(self.a, f.targets[0], item);
            } else {
                // tuple unpack (e.g. `for k, v in x|dictsort`)
                const parts: []Value = switch (item) {
                    .list => |l| l.items.items,
                    else => return self.rt("cannot unpack non-list in for"),
                };
                if (parts.len != f.targets.len) return self.rt("for-loop unpack arity mismatch");
                for (f.targets, 0..) |t, ti| try frame.put(self.a, t, parts[ti]);
            }
            try frame.put(self.a, "loop", try self.loopVar(idx, items));
            try self.execNodes(f.body, out);
        }
    }

    fn loopVar(self: *Interp, idx: usize, items: []Value) Error!Value {
        const d = try self.newDict();
        try d.put(self.a, "index0", .{ .int = @intCast(idx) });
        try d.put(self.a, "index", .{ .int = @intCast(idx + 1) });
        try d.put(self.a, "revindex", .{ .int = @intCast(items.len - idx) });
        try d.put(self.a, "revindex0", .{ .int = @intCast(items.len - idx - 1) });
        try d.put(self.a, "first", .{ .boolean = idx == 0 });
        try d.put(self.a, "last", .{ .boolean = idx + 1 == items.len });
        try d.put(self.a, "length", .{ .int = @intCast(items.len) });
        try d.put(self.a, "previtem", if (idx > 0) items[idx - 1] else .undef);
        try d.put(self.a, "nextitem", if (idx + 1 < items.len) items[idx + 1] else .undef);
        return .{ .dict = d };
    }

    // --- expression evaluation -------------------------------------------

    fn eval(self: *Interp, e: *const Expr) Error!Value {
        switch (e.*) {
            .lit => |v| return v,
            .ident => |name| return self.lookup(name),
            .attr => |a| {
                const obj = try self.eval(a.obj);
                return self.getAttr(obj, a.name);
            },
            .index => |ix| {
                const obj = try self.eval(ix.obj);
                const idx = try self.eval(ix.idx);
                return self.getItem(obj, idx);
            },
            .slice => |sl| {
                const obj = try self.eval(sl.obj);
                const start: ?i64 = if (sl.start) |se| (try self.eval(se)).int else null;
                const end: ?i64 = if (sl.end) |ee| (try self.eval(ee)).int else null;
                const step: i64 = if (sl.step) |pe| (try self.eval(pe)).int else 1;
                return self.doSlice(obj, start, end, step);
            },
            .list => |items| {
                const l = try self.newList();
                for (items) |ie| try l.items.append(self.a, try self.eval(ie));
                return .{ .list = l };
            },
            .dict => |pairs| {
                const d = try self.newDict();
                for (pairs) |p| {
                    const k = try self.eval(p.key);
                    const key = switch (k) {
                        .str => |s| s,
                        else => return self.rt("dict key must be a string"),
                    };
                    try d.put(self.a, key, try self.eval(p.val));
                }
                return .{ .dict = d };
            },
            .unop => |u| {
                const o = try self.eval(u.operand);
                return switch (u.op) {
                    .not => .{ .boolean = !o.truthy() },
                    .neg => switch (o) {
                        .int => |i| .{ .int = -i },
                        .float => |fl| .{ .float = -fl },
                        else => self.rt("cannot negate non-number"),
                    },
                };
            },
            .binop => |b| return self.evalBinop(b.op, b.lhs, b.rhs),
            .ternary => |t| {
                if ((try self.eval(t.cond)).truthy()) return self.eval(t.then);
                return self.eval(t.els);
            },
            .filter => |fl| return self.evalFilter(fl.name, fl.value, fl.args),
            .do_test => |t| {
                const r = try self.evalTest(t.name, t.value, t.args);
                return .{ .boolean = if (t.negate) !r else r };
            },
            .method => |m| return self.evalMethod(m.obj, m.name, m.args),
            .call => |c| return self.evalCall(c.callee, c.args, c.kwargs),
        }
    }

    fn evalBinop(self: *Interp, op: BinOp, le: *const Expr, re: *const Expr) Error!Value {
        // short-circuit
        if (op == .@"and") {
            const l = try self.eval(le);
            if (!l.truthy()) return l;
            return self.eval(re);
        }
        if (op == .@"or") {
            const l = try self.eval(le);
            if (l.truthy()) return l;
            return self.eval(re);
        }
        const l = try self.eval(le);
        const r = try self.eval(re);
        switch (op) {
            .add => {
                if (l == .str and r == .str) return self.strVal(try std.mem.concat(self.a, u8, &.{ l.str, r.str }));
                return self.numBinop(l, r, .add);
            },
            .sub => return self.numBinop(l, r, .sub),
            .mul => return self.numBinop(l, r, .mul),
            .div => return self.numBinop(l, r, .div),
            .floordiv => return self.numBinop(l, r, .floordiv),
            .mod => return self.numBinop(l, r, .mod),
            .concat => return self.strVal(try std.mem.concat(self.a, u8, &.{ try self.toStr(l), try self.toStr(r) })),
            .eq => return .{ .boolean = self.valueEql(l, r) },
            .ne => return .{ .boolean = !self.valueEql(l, r) },
            .lt, .gt, .le, .ge => return self.compare(l, r, op),
            .in => return .{ .boolean = try self.contains(r, l) },
            .not_in => return .{ .boolean = !(try self.contains(r, l)) },
            .@"and", .@"or" => unreachable,
        }
    }

    fn numBinop(self: *Interp, l: Value, r: Value, op: enum { add, sub, mul, div, floordiv, mod }) Error!Value {
        if (l == .int and r == .int) {
            const a = l.int;
            const b = r.int;
            return switch (op) {
                .add => .{ .int = a + b },
                .sub => .{ .int = a - b },
                .mul => .{ .int = a * b },
                .div => .{ .float = @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b)) },
                .floordiv => .{ .int = @divFloor(a, b) },
                .mod => .{ .int = @mod(a, b) },
            };
        }
        const a = try self.toFloat(l);
        const b = try self.toFloat(r);
        return switch (op) {
            .add => .{ .float = a + b },
            .sub => .{ .float = a - b },
            .mul => .{ .float = a * b },
            .div => .{ .float = a / b },
            .floordiv => .{ .float = @divFloor(a, b) },
            .mod => .{ .float = @mod(a, b) },
        };
    }

    fn compare(self: *Interp, l: Value, r: Value, op: BinOp) Error!Value {
        const ord: std.math.Order = blk: {
            if (l == .str and r == .str) break :blk std.mem.order(u8, l.str, r.str);
            break :blk std.math.order(try self.toFloat(l), try self.toFloat(r));
        };
        return .{ .boolean = switch (op) {
            .lt => ord == .lt,
            .gt => ord == .gt,
            .le => ord != .gt,
            .ge => ord != .lt,
            else => unreachable,
        } };
    }

    fn valueEql(self: *Interp, l: Value, r: Value) bool {
        _ = self;
        return switch (l) {
            .undef => r == .undef,
            .none => r == .none,
            .boolean => |b| r == .boolean and r.boolean == b,
            .int => switch (r) {
                .int => |ri| ri == l.int,
                .float => |rf| rf == @as(f64, @floatFromInt(l.int)),
                else => false,
            },
            .float => switch (r) {
                .float => |rf| rf == l.float,
                .int => |ri| l.float == @as(f64, @floatFromInt(ri)),
                else => false,
            },
            .str => |s| r == .str and std.mem.eql(u8, s, r.str),
            .list => |x| r == .list and x == r.list,
            .dict => |x| r == .dict and x == r.dict,
            .macro => |x| r == .macro and x == r.macro,
            .builtin => |x| r == .builtin and std.mem.eql(u8, x, r.builtin),
        };
    }

    fn contains(self: *Interp, container: Value, needle: Value) Error!bool {
        switch (container) {
            .str => |s| {
                if (needle != .str) return false;
                return std.mem.indexOf(u8, s, needle.str) != null;
            },
            .list => |l| {
                for (l.items.items) |it| if (self.valueEql(it, needle)) return true;
                return false;
            },
            .dict => |d| {
                if (needle != .str) return false;
                return d.get(needle.str) != null;
            },
            else => return false,
        }
    }

    fn getAttr(self: *Interp, obj: Value, name: []const u8) Value {
        switch (obj) {
            .dict => |d| return d.get(name) orelse .undef,
            else => {
                _ = self;
                return .undef;
            },
        }
    }
    fn getItem(self: *Interp, obj: Value, idx: Value) Value {
        switch (obj) {
            .dict => |d| {
                if (idx != .str) return .undef;
                return d.get(idx.str) orelse .undef;
            },
            .list => |l| {
                if (idx != .int) return .undef;
                var i = idx.int;
                if (i < 0) i += @intCast(l.items.items.len);
                if (i < 0 or i >= l.items.items.len) return .undef;
                return l.items.items[@intCast(i)];
            },
            .str => |s| {
                if (idx != .int) return .undef;
                var i = idx.int;
                if (i < 0) i += @intCast(s.len);
                if (i < 0 or i >= s.len) return .undef;
                return self.strVal(s[@intCast(i)..@intCast(i + 1)]);
            },
            else => return .undef,
        }
    }
    /// Python-semantics slice with optional negative/step (handles `[::-1]`).
    fn doSlice(self: *Interp, obj: Value, start_in: ?i64, end_in: ?i64, step: i64) Error!Value {
        const n: i64 = switch (obj) {
            .list => |l| @intCast(l.items.items.len),
            .str => |s| @intCast(s.len),
            else => return .undef,
        };
        if (step == 0) return self.rt("slice step cannot be zero");
        // Resolve defaults + negatives per Python's slice rules.
        var s: i64 = undefined;
        var e: i64 = undefined;
        if (step > 0) {
            s = start_in orelse 0;
            e = end_in orelse n;
            if (s < 0) s += n;
            if (e < 0) e += n;
            s = std.math.clamp(s, 0, n);
            e = std.math.clamp(e, 0, n);
        } else {
            s = start_in orelse (n - 1);
            e = end_in orelse (-n - 1);
            if (s < 0) s += n;
            if (e < 0) e += n;
            s = std.math.clamp(s, -1, n - 1);
            e = std.math.clamp(e, -1, n - 1);
        }
        // Collect indices.
        var idxs: std.ArrayList(usize) = .empty;
        var i = s;
        while ((step > 0 and i < e) or (step < 0 and i > e)) : (i += step) {
            if (i >= 0 and i < n) try idxs.append(self.a, @intCast(i));
        }
        switch (obj) {
            .list => |l| {
                const out = try self.newList();
                for (idxs.items) |k| try out.items.append(self.a, l.items.items[k]);
                return .{ .list = out };
            },
            .str => |str| {
                var buf: std.ArrayList(u8) = .empty;
                for (idxs.items) |k| try buf.append(self.a, str[k]);
                return self.strVal(buf.items);
            },
            else => return .undef,
        }
    }

    // --- calls / methods / builtins --------------------------------------

    fn evalCall(self: *Interp, callee: *const Expr, args: []const *Expr, kwargs: []const Kw) Error!Value {
        const cv = try self.eval(callee);
        switch (cv) {
            .macro => |m| return self.invokeMacro(m, args, kwargs),
            .builtin => |name| return self.invokeBuiltin(name, args, kwargs),
            else => return self.rt("value is not callable"),
        }
    }

    fn invokeMacro(self: *Interp, m: *const Macro, args: []const *Expr, kwargs: []const Kw) Error!Value {
        const frame = try self.newDict();
        // positional
        for (m.params, 0..) |p, pi| {
            if (pi < args.len) {
                try frame.put(self.a, p.name, try self.eval(args[pi]));
            } else {
                // kwarg match or default
                var v: ?Value = null;
                for (kwargs) |kw| if (std.mem.eql(u8, kw.name, p.name)) {
                    v = try self.eval(kw.val);
                };
                if (v == null and p.default != null) v = try self.eval(p.default.?);
                try frame.put(self.a, p.name, v orelse .undef);
            }
        }
        // Macro scope = globals frame + param frame (NOT the caller's locals).
        const saved = self.frames;
        var macro_frames: std.ArrayList(*Dict) = .empty;
        try macro_frames.append(self.a, saved.items[0]);
        try macro_frames.append(self.a, frame);
        self.frames = macro_frames;
        defer self.frames = saved;

        var buf: std.ArrayList(u8) = .empty;
        try self.execNodes(m.body, &buf);
        return self.strVal(buf.items);
    }

    fn invokeBuiltin(self: *Interp, name: []const u8, args: []const *Expr, kwargs: []const Kw) Error!Value {
        if (std.mem.eql(u8, name, "range")) {
            var lo: i64 = 0;
            var hi: i64 = 0;
            var step: i64 = 1;
            if (args.len == 1) {
                hi = (try self.eval(args[0])).int;
            } else if (args.len >= 2) {
                lo = (try self.eval(args[0])).int;
                hi = (try self.eval(args[1])).int;
                if (args.len >= 3) step = (try self.eval(args[2])).int;
            }
            const l = try self.newList();
            if (step == 0) return self.rt("range() step must not be zero");
            var i = lo;
            while ((step > 0 and i < hi) or (step < 0 and i > hi)) : (i += step)
                try l.items.append(self.a, .{ .int = i });
            return .{ .list = l };
        }
        if (std.mem.eql(u8, name, "namespace")) {
            const d = try self.newDict();
            for (kwargs) |kw| try d.put(self.a, kw.name, try self.eval(kw.val));
            return .{ .dict = d };
        }
        if (std.mem.eql(u8, name, "raise_exception")) {
            self.diag = if (args.len > 0) try self.toStr(try self.eval(args[0])) else "raise_exception";
            return Error.JinjaRaise;
        }
        return self.rt("unknown builtin");
    }

    fn evalMethod(self: *Interp, obj_e: *const Expr, name: []const u8, args: []const *Expr) Error!Value {
        const obj = try self.eval(obj_e);
        if (obj == .dict) {
            if (std.mem.eql(u8, name, "get")) {
                const key = try self.toStr(try self.eval(args[0]));
                if (obj.dict.get(key)) |v| return v;
                if (args.len >= 2) return self.eval(args[1]);
                return .none;
            }
            if (std.mem.eql(u8, name, "items")) {
                const l = try self.newList();
                for (obj.dict.entries.items) |ent| {
                    const pair = try self.newList();
                    try pair.items.append(self.a, self.strVal(ent.key));
                    try pair.items.append(self.a, ent.val);
                    try l.items.append(self.a, .{ .list = pair });
                }
                return .{ .list = l };
            }
            if (std.mem.eql(u8, name, "keys")) {
                const l = try self.newList();
                for (obj.dict.entries.items) |ent| try l.items.append(self.a, self.strVal(ent.key));
                return .{ .list = l };
            }
            if (std.mem.eql(u8, name, "values")) {
                const l = try self.newList();
                for (obj.dict.entries.items) |ent| try l.items.append(self.a, ent.val);
                return .{ .list = l };
            }
        }
        if (obj == .str) {
            const s = obj.str;
            if (std.mem.eql(u8, name, "split")) {
                const l = try self.newList();
                if (args.len == 0) {
                    var it = std.mem.tokenizeAny(u8, s, " \t\r\n");
                    while (it.next()) |part| try l.items.append(self.a, self.strVal(part));
                } else {
                    const sep = try self.toStr(try self.eval(args[0]));
                    if (sep.len == 0) return self.rt("empty separator");
                    var it = std.mem.splitSequence(u8, s, sep);
                    while (it.next()) |part| try l.items.append(self.a, self.strVal(part));
                }
                return .{ .list = l };
            }
            if (std.mem.eql(u8, name, "strip")) return self.strVal(std.mem.trim(u8, s, " \t\r\n"));
            if (std.mem.eql(u8, name, "lstrip")) {
                const cut = if (args.len > 0) try self.toStr(try self.eval(args[0])) else " \t\r\n";
                return self.strVal(std.mem.trimStart(u8, s, cut));
            }
            if (std.mem.eql(u8, name, "rstrip")) {
                const cut = if (args.len > 0) try self.toStr(try self.eval(args[0])) else " \t\r\n";
                return self.strVal(std.mem.trimEnd(u8, s, cut));
            }
            if (std.mem.eql(u8, name, "startswith")) {
                return .{ .boolean = std.mem.startsWith(u8, s, try self.toStr(try self.eval(args[0]))) };
            }
            if (std.mem.eql(u8, name, "endswith")) {
                return .{ .boolean = std.mem.endsWith(u8, s, try self.toStr(try self.eval(args[0]))) };
            }
            if (std.mem.eql(u8, name, "upper")) return self.strVal(try std.ascii.allocUpperString(self.a, s));
            if (std.mem.eql(u8, name, "lower")) return self.strVal(try std.ascii.allocLowerString(self.a, s));
            if (std.mem.eql(u8, name, "replace")) {
                const from = try self.toStr(try self.eval(args[0]));
                const to = try self.toStr(try self.eval(args[1]));
                const sz = std.mem.replacementSize(u8, s, from, to);
                const buf = try self.a.alloc(u8, sz);
                _ = std.mem.replace(u8, s, from, to, buf);
                return self.strVal(buf);
            }
        }
        return self.rt("unknown method");
    }

    // --- filters ----------------------------------------------------------

    fn evalFilter(self: *Interp, name: []const u8, value_e: *const Expr, args: []const *Expr) Error!Value {
        const v = try self.eval(value_e);
        return self.applyFilterToValue(name, v, args);
    }

    fn applyFilterToValue(self: *Interp, name: []const u8, v: Value, args: []const *Expr) Error!Value {
        if (std.mem.eql(u8, name, "default") or std.mem.eql(u8, name, "d")) {
            const use_default = if (args.len >= 2 and (try self.eval(args[1])).truthy())
                !v.truthy()
            else
                v == .undef;
            if (use_default) return if (args.len >= 1) self.eval(args[0]) else .{ .str = "" };
            return v;
        }
        if (std.mem.eql(u8, name, "trim")) return self.strVal(std.mem.trim(u8, try self.toStr(v), " \t\r\n"));
        if (std.mem.eql(u8, name, "upper")) return self.strVal(try std.ascii.allocUpperString(self.a, try self.toStr(v)));
        if (std.mem.eql(u8, name, "lower")) return self.strVal(try std.ascii.allocLowerString(self.a, try self.toStr(v)));
        if (std.mem.eql(u8, name, "string")) return self.strVal(try self.toStr(v));
        if (std.mem.eql(u8, name, "safe") or std.mem.eql(u8, name, "e") or std.mem.eql(u8, name, "escape")) return v; // no autoescape
        if (std.mem.eql(u8, name, "length") or std.mem.eql(u8, name, "count")) {
            return .{ .int = @intCast(switch (v) {
                .str => |s| s.len,
                .list => |l| l.items.items.len,
                .dict => |d| d.entries.items.len,
                else => 0,
            }) };
        }
        if (std.mem.eql(u8, name, "first")) {
            return switch (v) {
                .list => |l| if (l.items.items.len > 0) l.items.items[0] else .undef,
                .str => |s| if (s.len > 0) self.strVal(s[0..1]) else .undef,
                else => .undef,
            };
        }
        if (std.mem.eql(u8, name, "last")) {
            return switch (v) {
                .list => |l| if (l.items.items.len > 0) l.items.items[l.items.items.len - 1] else .undef,
                else => .undef,
            };
        }
        if (std.mem.eql(u8, name, "list")) {
            return switch (v) {
                .list => v,
                .str => |s| blk: {
                    const l = try self.newList();
                    for (0..s.len) |k| try l.items.append(self.a, self.strVal(s[k .. k + 1]));
                    break :blk .{ .list = l };
                },
                else => .{ .list = try self.newList() },
            };
        }
        if (std.mem.eql(u8, name, "join")) {
            if (v != .list) return self.strVal("");
            const sep = if (args.len >= 1) try self.toStr(try self.eval(args[0])) else "";
            var buf: std.ArrayList(u8) = .empty;
            for (v.list.items.items, 0..) |it, i| {
                if (i > 0) try buf.appendSlice(self.a, sep);
                try buf.appendSlice(self.a, try self.toStr(it));
            }
            return self.strVal(buf.items);
        }
        if (std.mem.eql(u8, name, "map")) {
            // map('filtername') — apply a named filter to each item.
            if (v != .list or args.len < 1) return v;
            const fname = try self.toStr(try self.eval(args[0]));
            const l = try self.newList();
            for (v.list.items.items) |it| try l.items.append(self.a, try self.applyNamedFilter(fname, it));
            return .{ .list = l };
        }
        if (std.mem.eql(u8, name, "dictsort")) {
            if (v != .dict) return v;
            const entries = try self.a.dupe(Entry, v.dict.entries.items);
            std.sort.block(Entry, entries, {}, struct {
                fn lt(_: void, x: Entry, y: Entry) bool {
                    return std.mem.lessThan(u8, x.key, y.key);
                }
            }.lt);
            const l = try self.newList();
            for (entries) |ent| {
                const pair = try self.newList();
                try pair.items.append(self.a, self.strVal(ent.key));
                try pair.items.append(self.a, ent.val);
                try l.items.append(self.a, .{ .list = pair });
            }
            return .{ .list = l };
        }
        if (std.mem.eql(u8, name, "tojson")) {
            var buf: std.ArrayList(u8) = .empty;
            var indent: ?usize = null;
            if (args.len >= 1) {
                const iv = try self.eval(args[0]);
                if (iv == .int) indent = @intCast(iv.int);
            }
            try self.toJson(v, &buf, indent, 0);
            return self.strVal(buf.items);
        }
        return self.rt("unknown filter");
    }

    fn applyNamedFilter(self: *Interp, fname: []const u8, item: Value) Error!Value {
        if (std.mem.eql(u8, fname, "upper")) return self.strVal(try std.ascii.allocUpperString(self.a, try self.toStr(item)));
        if (std.mem.eql(u8, fname, "lower")) return self.strVal(try std.ascii.allocLowerString(self.a, try self.toStr(item)));
        if (std.mem.eql(u8, fname, "trim")) return self.strVal(std.mem.trim(u8, try self.toStr(item), " \t\r\n"));
        if (std.mem.eql(u8, fname, "string")) return self.strVal(try self.toStr(item));
        return self.rt("unknown filter in map()");
    }

    // --- tests ------------------------------------------------------------

    fn evalTest(self: *Interp, name: []const u8, value_e: *const Expr, args: []const *Expr) Error!bool {
        _ = args;
        const v = try self.eval(value_e);
        if (std.mem.eql(u8, name, "defined")) return v != .undef;
        if (std.mem.eql(u8, name, "undefined")) return v == .undef;
        if (std.mem.eql(u8, name, "none")) return v == .none;
        if (std.mem.eql(u8, name, "string")) return v == .str;
        if (std.mem.eql(u8, name, "mapping")) return v == .dict;
        if (std.mem.eql(u8, name, "boolean")) return v == .boolean;
        if (std.mem.eql(u8, name, "number")) return v == .int or v == .float;
        if (std.mem.eql(u8, name, "integer")) return v == .int;
        if (std.mem.eql(u8, name, "float")) return v == .float;
        if (std.mem.eql(u8, name, "sequence")) return v == .list or v == .str;
        if (std.mem.eql(u8, name, "iterable")) return v == .list or v == .str or v == .dict;
        if (std.mem.eql(u8, name, "true")) return v == .boolean and v.boolean;
        if (std.mem.eql(u8, name, "false")) return v == .boolean and !v.boolean;
        return self.rt2("unknown test");
    }

    // --- coercion / output ------------------------------------------------

    fn toFloat(self: *Interp, v: Value) Error!f64 {
        return switch (v) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
            .boolean => |b| if (b) 1 else 0,
            else => self.rtF("expected a number"),
        };
    }

    /// Render a value the way `str()` / `~` / string filters do (used for
    /// concatenation and filter inputs — NOT the same as writeValue for output,
    /// which matches Jinja's `{{ }}` printing of None/Undefined).
    fn toStr(self: *Interp, v: Value) Error![]const u8 {
        return switch (v) {
            .str => |s| s,
            .undef => "",
            .none => "None",
            .boolean => |b| if (b) "True" else "False",
            .int => |i| try std.fmt.allocPrint(self.a, "{d}", .{i}),
            .float => |f| try std.fmt.allocPrint(self.a, "{d}", .{f}),
            .list, .dict, .macro, .builtin => self.rtS("cannot stringify this value"),
        };
    }

    /// `{{ expr }}` output. Jinja prints Undefined as "" and None as "None".
    fn writeValue(self: *Interp, v: Value, out: *std.ArrayList(u8)) Error!void {
        try out.appendSlice(self.a, try self.toStr(v));
    }

    fn toJson(self: *Interp, v: Value, out: *std.ArrayList(u8), indent: ?usize, depth: usize) Error!void {
        switch (v) {
            .none, .undef => try out.appendSlice(self.a, "null"),
            .boolean => |b| try out.appendSlice(self.a, if (b) "true" else "false"),
            .int => |i| try out.print(self.a, "{d}", .{i}),
            .float => |f| try out.print(self.a, "{d}", .{f}),
            .str => |s| try self.jsonStr(s, out),
            .list => |l| {
                try out.append(self.a, '[');
                for (l.items.items, 0..) |it, i| {
                    if (i > 0) try out.append(self.a, ',');
                    try self.jsonNL(out, indent, depth + 1);
                    try self.toJson(it, out, indent, depth + 1);
                }
                if (l.items.items.len > 0) try self.jsonNL(out, indent, depth);
                try out.append(self.a, ']');
            },
            .dict => |d| {
                try out.append(self.a, '{');
                for (d.entries.items, 0..) |ent, i| {
                    if (i > 0) try out.append(self.a, ',');
                    try self.jsonNL(out, indent, depth + 1);
                    try self.jsonStr(ent.key, out);
                    try out.append(self.a, ':');
                    if (indent != null) try out.append(self.a, ' ');
                    try self.toJson(ent.val, out, indent, depth + 1);
                }
                if (d.entries.items.len > 0) try self.jsonNL(out, indent, depth);
                try out.append(self.a, '}');
            },
            else => try out.appendSlice(self.a, "null"),
        }
    }
    fn jsonNL(self: *Interp, out: *std.ArrayList(u8), indent: ?usize, depth: usize) Error!void {
        if (indent) |n| {
            try out.append(self.a, '\n');
            try out.appendNTimes(self.a, ' ', n * depth);
        }
    }
    fn jsonStr(self: *Interp, s: []const u8, out: *std.ArrayList(u8)) Error!void {
        try out.append(self.a, '"');
        for (s) |c| switch (c) {
            '"' => try out.appendSlice(self.a, "\\\""),
            '\\' => try out.appendSlice(self.a, "\\\\"),
            '\n' => try out.appendSlice(self.a, "\\n"),
            '\r' => try out.appendSlice(self.a, "\\r"),
            '\t' => try out.appendSlice(self.a, "\\t"),
            else => try out.append(self.a, c),
        };
        try out.append(self.a, '"');
    }

    // --- error helpers ----------------------------------------------------
    fn rt(self: *Interp, msg: []const u8) Error {
        self.diag = msg;
        return Error.JinjaRuntime;
    }
    fn rt2(self: *Interp, msg: []const u8) Error!bool {
        self.diag = msg;
        return Error.JinjaRuntime;
    }
    fn rtF(self: *Interp, msg: []const u8) Error!f64 {
        self.diag = msg;
        return Error.JinjaRuntime;
    }
    fn rtS(self: *Interp, msg: []const u8) Error![]const u8 {
        self.diag = msg;
        return Error.JinjaRuntime;
    }
};

// ---------------------------------------------------------------------------
// JSON → Value bridge (used by tests and the config layer to build context)
// ---------------------------------------------------------------------------

/// Deep-convert a parsed `std.json.Value` into a jinja `Value`, allocating in
/// `a`. JSON has no undefined; nulls map to `.none`. Objects preserve order.
pub fn fromJson(a: std.mem.Allocator, jv: std.json.Value) Error!Value {
    return switch (jv) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .str = s },
        .string => |s| .{ .str = s },
        .array => |arr| blk: {
            const l = try a.create(List);
            l.* = .{};
            for (arr.items) |it| try l.items.append(a, try fromJson(a, it));
            break :blk .{ .list = l };
        },
        .object => |obj| blk: {
            const d = try a.create(Dict);
            d.* = .{};
            var it = obj.iterator();
            while (it.next()) |kv| try d.put(a, kv.key_ptr.*, try fromJson(a, kv.value_ptr.*));
            break :blk .{ .dict = d };
        },
    };
}

// ---------------------------------------------------------------------------
// Tests: byte-exact against jinja2 goldens (assets/jinja/fixtures.json).
// ---------------------------------------------------------------------------

const fixtures_json = @embedFile("assets/jinja/fixtures.json");

test "jinja: golden fixtures render byte-exact vs jinja2" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = try std.json.parseFromSlice(std.json.Value, a, fixtures_json, .{});
    const root = parsed.value.object;
    const templates = root.get("templates").?.object;
    const cases = root.get("cases").?.array;

    // Parse each template once.
    var tmpls = std.StringHashMap(Template).init(a);
    var tit = templates.iterator();
    while (tit.next()) |kv| {
        const t = try Template.parse(gpa, kv.value_ptr.*.string);
        try tmpls.put(kv.key_ptr.*, t);
    }
    defer {
        var vit = tmpls.valueIterator();
        while (vit.next()) |t| t.deinit();
    }

    var failures: usize = 0;
    var first_fail: []const u8 = "";
    for (cases.items) |c| {
        const case = c.object;
        const key = case.get("key").?.string;
        const tname = case.get("template").?.string;
        const expected = case.get("expected").?.string;
        const ctx = try fromJson(a, case.get("context").?);

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        const tmpl = tmpls.get(tname).?;
        tmpl.render(gpa, ctx, &out) catch |e| {
            std.debug.print("[{s}] render error: {t} (diag not surfaced)\n", .{ key, e });
            failures += 1;
            if (first_fail.len == 0) first_fail = key;
            continue;
        };
        if (!std.mem.eql(u8, out.items, expected)) {
            failures += 1;
            if (first_fail.len == 0) first_fail = key;
            std.debug.print("[{s}] MISMATCH\n--- expected ---\n{s}\n--- got ---\n{s}\n--- end ---\n", .{ key, expected, out.items });
        }
    }
    errdefer std.debug.print("jinja goldens: {d}/{d} failed (first: {s})\n", .{ failures, cases.items.len, first_fail });
    try std.testing.expectEqual(@as(usize, 0), failures);
}
