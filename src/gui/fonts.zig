//! Bundled broad-coverage fonts + style-aware run splitting.
//!
//! dvui renders one font face per text run (no per-glyph fallback across
//! fonts), and its built-in font is Latin-only, so LLM output with CJK /
//! arrows / math / box-drawing shows tofu boxes. We bundle:
//!
//! - NotoSansCJK Regular + Bold (pan-CJK + Latin + symbols) — body text.
//! - NotoSansMonoCJK JP Regular (pan-CJK monospace) — code, `font_mono`.
//! - NotoSans Italic + BoldItalic (Latin/Greek/Cyrillic only — CJK has no
//!   italic tradition and Noto ships none) — italic runs fall back to the
//!   upright CJK face outside LGC coverage.
//! - NotoEmoji (monochrome outlines — dvui's rasterizer is monochrome, so
//!   color emoji fonts can't be used) — rendered per-run for emoji.
//!
//! `addStyled` is the single place that maps (codepoint, style) → font: it
//! splits text into same-font runs so every glyph lands on a face that has
//! it. Use it (or `addRich`/`richLabel`) for ANY user/LLM-visible text.
const std = @import("std");
const dvui = @import("dvui");

pub const family = "NotoSansCJK";
pub const mono_family = "NotoSansMonoCJK";
/// Latin/Greek/Cyrillic family — only italic faces are bundled (used solely
/// for italic runs; upright text stays on the CJK family).
pub const lgc_family = "NotoSans";
pub const emoji_family = "NotoEmoji";

const regular_bytes = @embedFile("fonts/NotoSansCJK-Regular.ttc");
const bold_bytes = @embedFile("fonts/NotoSansCJK-Bold.ttc");
const mono_bytes = @embedFile("fonts/NotoSansMonoCJKjp-Regular.otf");
const italic_bytes = @embedFile("fonts/NotoSans-Italic.ttf");
const bold_italic_bytes = @embedFile("fonts/NotoSans-BoldItalic.ttf");
const emoji_bytes = @embedFile("fonts/NotoEmoji.ttf");

/// Weight/style-tagged font sources, registered via `Theme.embedded_fonts`
/// (plain `dvui.addFont` can only register normal weight/style). Note dvui
/// dedups sources by bytes pointer, so the same bytes can't alias a second
/// weight — code runs therefore never ask for bold (see `fontFor`).
const sources = [_]dvui.Font.Source{
    .{ .family = dvui.Font.array(family), .bytes = regular_bytes },
    .{ .family = dvui.Font.array(family), .weight = .bold, .bytes = bold_bytes },
    .{ .family = dvui.Font.array(mono_family), .bytes = mono_bytes },
    .{ .family = dvui.Font.array(lgc_family), .style = .italic, .bytes = italic_bytes },
    .{ .family = dvui.Font.array(lgc_family), .weight = .bold, .style = .italic, .bytes = bold_italic_bytes },
    .{ .family = dvui.Font.array(emoji_family), .bytes = emoji_bytes },
};

/// Register the bundled fonts and point the current theme's text styles at
/// them. Must run inside a `Window.begin`/`end` pair (needs the current
/// window); `themeSet` pulls `embedded_fonts` into the window's font database.
pub fn install() void {
    var theme = dvui.themeGet();
    theme.embedded_fonts = &sources;
    theme.font_body = theme.font_body.withFamily(family);
    theme.font_heading = theme.font_heading.withFamily(family);
    theme.font_title = theme.font_title.withFamily(family);
    theme.font_mono = theme.font_mono.withFamily(mono_family);
    dvui.themeSet(theme);
}

/// Inline text style, as produced by the markdown parser. `addStyled` turns
/// this plus per-codepoint coverage into a concrete font per run.
pub const Style = struct {
    bold: bool = false,
    italic: bool = false,
    code: bool = false,
    strike: bool = false,
};

/// The bundled emoji font at the current body text size.
pub fn emojiFont() dvui.Font {
    return dvui.themeGet().font_body.withFamily(emoji_family);
}

/// Codepoints routed to the monochrome emoji font. Kept to clearly-emoji
/// blocks — arrows / CJK / punctuation stay in NotoSansCJK, which covers them.
fn isEmoji(cp: u21) bool {
    return switch (cp) {
        0x1F000...0x1FAFF => true, // emoji + pictographs (incl. flags, skin tones)
        0x2600...0x27BF => true, // misc symbols + dingbats (⚙ ⚠ …)
        0x2B00...0x2BFF => true, // misc symbols & arrows (emoji-presentation)
        0xFE00...0xFE0F => true, // variation selectors — keep with the emoji run
        0x200D => true, // ZWJ — keep emoji sequences together
        else => false,
    };
}

/// Codepoints the bundled NotoSans italic faces cover (Latin/Greek/Cyrillic
/// and shared punctuation). Italic runs outside this render upright in the
/// CJK face — matching how CJK text is conventionally emphasized elsewhere.
fn isLgc(cp: u21) bool {
    return switch (cp) {
        0x00...0x2FF => true, // ASCII + Latin-1 + Latin Extended-A/B + IPA + modifiers
        0x300...0x36F => true, // combining diacritics
        0x370...0x3FF => true, // Greek
        0x400...0x52F => true, // Cyrillic + supplement
        0x1E00...0x1EFF => true, // Latin Extended Additional
        0x1F00...0x1FFF => true, // Greek Extended
        0x2000...0x206F => true, // general punctuation (quotes, dashes, …)
        0x20A0...0x20CF => true, // currency symbols
        0x2100...0x214F => true, // letterlike (™ Ω …)
        else => false,
    };
}

/// Resolve the font for one codepoint under `style`, deriving size (and any
/// underline/strike the caller set) from `base`.
fn fontFor(cp: u21, style: Style, base: dvui.Font) dvui.Font {
    var f = base;
    if (style.strike) f = f.withStrike(.{});
    if (isEmoji(cp)) return f.withFamily(emoji_family);
    // No bold mono face is bundled (and dvui can't alias one), so code runs
    // stay normal weight even inside bold text.
    if (style.code) return f.withFamily(mono_family).withWeight(.normal);
    const weight: dvui.Font.Weight = if (style.bold or base.weight == .bold) .bold else .normal;
    if (style.italic and isLgc(cp)) return f.withFamily(lgc_family).withWeight(weight).withStyle(.italic);
    return f.withFamily(family).withWeight(weight);
}

/// Add `text` to a text layout, split into same-font runs: emoji go to the
/// emoji face, code to the mono face, italic to the LGC italic face where
/// covered, everything else to the CJK face (bold-aware). `opts` is applied
/// to every run (colors, background for inline code, …); `opts.font` (or the
/// theme body font) sets the base size.
pub fn addStyled(tl: *dvui.TextLayoutWidget, text: []const u8, style: Style, opts: dvui.Options) void {
    const base = opts.font orelse dvui.themeGet().font_body;
    var start: usize = 0;
    var i: usize = 0;
    var cur: dvui.Font = undefined;
    var have = false;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const cp: u21 = if (n > 1 and i + n <= text.len)
            (std.unicode.utf8Decode(text[i..][0..n]) catch 0xFFFD)
        else
            text[i];
        const f = fontFor(cp, style, base);
        if (!have) {
            cur = f;
            have = true;
        } else if (!std.meta.eql(f, cur)) {
            var o = opts;
            o.font = cur;
            tl.addText(text[start..i], o);
            start = i;
            cur = f;
        }
        i += n;
    }
    if (start < text.len) {
        var o = opts;
        o.font = cur;
        tl.addText(text[start..], o);
    }
}

/// Plain-styled `addStyled` — drop-in for the common "just show this string
/// with emoji/symbol coverage" case.
pub fn addRich(tl: *dvui.TextLayoutWidget, text: []const u8) void {
    addStyled(tl, text, .{}, .{});
}

/// A short label that may contain emoji/symbols, rendered through a text
/// layout so the per-run font fallback applies. Drop-in for `dvui.label` at
/// symbol-bearing sites (plain labels/buttons can't mix fonts within a run).
pub fn richLabel(src: std.builtin.SourceLocation, text: []const u8, opts: dvui.Options) void {
    var tl = dvui.textLayout(src, .{}, opts);
    defer tl.deinit();
    addRich(tl, text);
}
