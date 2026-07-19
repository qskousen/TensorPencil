//! dvui rendering of the markdown subset parsed by markdown.zig.
//!
//! Hybrid layout, chosen to keep text selection seamless: all consecutive
//! prose blocks (paragraphs, headings, lists, quotes, rules) flow into ONE
//! TextLayoutWidget — dvui selection spans a whole layout, so styled runs,
//! links and headings select/copy together like plain text. Only a fenced
//! code block breaks the flow: it renders as its own card (full-width
//! background + mono face + copy button), which a single text layout can't
//! express — per-run backgrounds hug the glyphs and child widgets are
//! corner-only. Selecting across a fence boundary is the one thing this
//! gives up; each code card is independently selectable and has its own
//! copy button.
//!
//! Text re-parses and re-renders every frame (immediate mode); the parser
//! is a linear scan over slices, which is noise next to dvui's own layout
//! work, and its streaming rules keep partially-generated markdown sane.
const std = @import("std");
const dvui = @import("dvui");
const md = @import("markdown.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");

pub const Opts = struct {
    /// Applied to every prose text layout — colors/background/border for
    /// the surrounding context (e.g. the dimmed thought block).
    prose: dvui.Options = .{},
    /// Base for widget ids when one parent renders several documents.
    id_extra: usize = 0,
};

/// Render `text` as markdown into the current parent (typically a chat
/// bubble box). Creates one or more widgets; ids derive from `src` +
/// `opts.id_extra` + an internal counter.
pub fn render(src: std.builtin.SourceLocation, text: []const u8, opts: Opts) void {
    var r: Renderer = .{ .src = src, .opts = opts };
    defer r.flushProse();
    var it = md.blocks(text);
    while (it.next()) |b| switch (b) {
        .code => |c| {
            r.flushProse();
            r.codeCard(c);
        },
        else => r.proseBlock(b),
    };
}

const Renderer = struct {
    src: std.builtin.SourceLocation,
    opts: Opts,
    /// Open prose layout, lazily created; flushed before each code card so
    /// widgets appear in document order (an open layout is the current
    /// parent — a card created inside it would nest as a corner widget).
    tl: ?*dvui.TextLayoutWidget = null,
    /// Tag of the previous prose block, for separator choice.
    last: ?std.meta.Tag(md.Block) = null,
    /// Running widget id disambiguator (several widgets share `src`).
    n: usize = 0,

    fn nextId(self: *Renderer) usize {
        self.n += 1;
        return self.opts.id_extra *% 1000 +% self.n;
    }

    fn proseTl(self: *Renderer) *dvui.TextLayoutWidget {
        if (self.tl) |t| return t;
        var o = self.opts.prose;
        o.id_extra = self.nextId();
        if (o.expand == null) o.expand = .horizontal;
        self.tl = dvui.textLayout(self.src, .{}, o);
        self.last = null;
        return self.tl.?;
    }

    fn flushProse(self: *Renderer) void {
        if (self.tl) |t| {
            t.deinit();
            self.tl = null;
        }
    }

    /// Blank line between prose blocks; consecutive list items and quote
    /// lines sit on adjacent lines instead.
    fn separate(self: *Renderer, tl: *dvui.TextLayoutWidget, tag: std.meta.Tag(md.Block)) void {
        if (self.last) |last| {
            const tight = tag == .list_item and last == .list_item;
            tl.addText(if (tight) "\n" else "\n\n", self.opts.prose);
        }
        self.last = tag;
    }

    fn proseBlock(self: *Renderer, b: md.Block) void {
        const tl = self.proseTl();
        const theme = dvui.themeGet();
        self.separate(tl, b);
        switch (b) {
            .code => unreachable,
            .paragraph => |text| self.emitSpans(tl, text, .{}, self.opts.prose),
            .heading => |h| {
                // Heading = bold at a size stepped by level; inline styles
                // still apply within it.
                const base = self.opts.prose.font orelse dvui.themeGet().font_body;
                const mult: f32 = switch (h.level) {
                    1 => 1.5,
                    2 => 1.3,
                    3 => 1.15,
                    else => 1.0,
                };
                var o = self.opts.prose;
                o.font = base.withSize(base.size * mult);
                self.emitSpans(tl, h.text, .{ .bold = true }, o);
            },
            .list_item => |li| {
                // Two indent columns per nesting level, glyph by depth.
                // (Type the @min result explicitly: @min(x, 3) narrows to u2
                // by design, and `level * 2` overflows it — Zig #14039.)
                const level: usize = @min(li.indent / 2, 3);
                var buf: [32]u8 = undefined;
                const marker = if (li.number) |num|
                    std.fmt.bufPrint(&buf, "{s}{d}. ", .{ "        "[0 .. level * 2], num }) catch "1. "
                else
                    std.fmt.bufPrint(&buf, "{s}{s} ", .{
                        "        "[0 .. level * 2],
                        switch (level) {
                            0 => "•",
                            1 => "◦",
                            else => "▪",
                        },
                    }) catch "• ";
                tl.addText(marker, self.opts.prose);
                self.emitSpans(tl, li.text, .{}, self.opts.prose);
            },
            .quote => |q| {
                // Blockquote: an accent bar glyph per line, dimmed text.
                var o = self.opts.prose;
                o.color_text = (self.opts.prose.color_text orelse theme.text).lerp(theme.fill, 0.35);
                var lines = std.mem.splitScalar(u8, q, '\n');
                var first = true;
                while (lines.next()) |line| {
                    if (!first) tl.addText("\n", o);
                    first = false;
                    var bar = o;
                    bar.color_text = theme.focus;
                    tl.addText("▎", bar);
                    self.emitSpans(tl, md.stripQuote(line), .{}, o);
                }
            },
            .rule => {
                var o = self.opts.prose;
                o.color_text = (self.opts.prose.color_text orelse theme.text).lerp(theme.fill, 0.6);
                tl.addText("─" ** 26, o);
            },
        }
    }

    /// Inline-parse `text` and add styled runs. `force` ORs into every
    /// span's style (headings force bold); `o` carries block-level options
    /// (font/base size, colors).
    fn emitSpans(self: *Renderer, tl: *dvui.TextLayoutWidget, text: []const u8, force: fonts.Style, o: dvui.Options) void {
        _ = self;
        const theme = dvui.themeGet();
        var it = md.spans(text);
        while (it.next()) |span| {
            const style: fonts.Style = .{
                .bold = span.style.bold or force.bold,
                .italic = span.style.italic or force.italic,
                .code = span.style.code or force.code,
                .strike = span.style.strike or force.strike,
            };
            var so = o;
            if (style.code) {
                // Inline code: per-run background highlight.
                so.color_fill = theme.fill.lerp(theme.text, 0.14);
            }
            if (span.link) |url| {
                // Underlined accent run that opens the URL. Single run (no
                // emoji splitting): link text is overwhelmingly ASCII and a
                // click region shouldn't fragment.
                so.color_text = theme.focus;
                const base = so.font orelse theme.font_body;
                so.font = base.withUnderline(.{});
                if (tl.addTextClick(span.text, so)) |click| {
                    const new_window = click == .mouse and
                        (click.mouse.button == .middle or click.mouse.mod.matchBind("ctrl/cmd"));
                    _ = dvui.openURL(.{ .url = url, .new_window = new_window });
                }
                continue;
            }
            fonts.addStyled(tl, span.text, style, so);
        }
    }

    fn codeCard(self: *Renderer, c: md.Code) void {
        const theme = dvui.themeGet();
        var card = dvui.box(self.src, .{ .dir = .vertical }, .{
            .id_extra = self.nextId(),
            .expand = .horizontal,
            .background = true,
            .color_fill = theme.fill.lerp(theme.text, 0.10),
            .corner_radius = dvui.Rect.all(6),
            .padding = dvui.Rect.all(6),
            .margin = .{ .y = 4, .h = 4 },
        });
        defer card.deinit();

        // Header: language tag left, copy-to-clipboard right. Rendered even
        // with no language so the copy affordance is always there.
        {
            var hb = dvui.box(self.src, .{ .dir = .horizontal }, .{
                .id_extra = self.nextId(),
                .expand = .horizontal,
            });
            defer hb.deinit();
            if (c.lang.len > 0) {
                dvui.label(self.src, "{s}", .{c.lang}, .{
                    .id_extra = self.nextId(),
                    .font = dvui.themeGet().font_mono.withSize(11),
                    .color_text = theme.text.lerp(theme.fill, 0.4),
                    .gravity_y = 0.5,
                    .padding = .{},
                    .margin = .{},
                });
            }
            var wd: dvui.WidgetData = undefined;
            if (dvui.buttonIcon(self.src, "copy code", dvui.entypo.clipboard, .{}, .{}, .{
                .id_extra = self.nextId(),
                .gravity_x = 1.0,
                .min_size_content = .{ .h = 14 },
                .color_text = theme.text.lerp(theme.fill, 0.4),
                .padding = dvui.Rect.all(2),
                .margin = .{},
                .data_out = &wd,
            })) dvui.clipboardTextSet(c.body);
            hint.hover(self.src, &wd, "Copy code");
        }

        var tl = dvui.textLayout(self.src, .{}, .{
            .id_extra = self.nextId(),
            .expand = .horizontal,
            .background = false,
            .padding = .{ .y = 4 },
        });
        defer tl.deinit();
        fonts.addStyled(tl, c.body, .{ .code = true }, .{});
    }
};
