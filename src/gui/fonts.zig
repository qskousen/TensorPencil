//! Bundled broad-coverage font. dvui renders one font family per text style
//! (no per-glyph fallback across fonts), and its built-in font is Latin-only,
//! so LLM output with CJK / arrows / math / box-drawing / punctuation shows
//! tofu boxes. Noto Sans CJK covers those ranges. (Pictographic color emoji
//! are still out of reach — dvui's rasterizer is monochrome.)
const std = @import("std");
const dvui = @import("dvui");

pub const family = "NotoSansCJK";
pub const emoji_family = "NotoEmoji";

const regular_bytes = @embedFile("fonts/NotoSansCJK-Regular.ttc");
// Monochrome Noto Emoji (outline glyphs — dvui's rasterizer is monochrome, so
// the color emoji font on the system can't be used). Rendered per-run for
// emoji codepoints, since dvui has no per-glyph fallback across fonts.
const emoji_bytes = @embedFile("fonts/NotoEmoji.ttf");

/// The bundled emoji font at the current body text size, for the emoji runs in
/// `addRich`.
pub fn emojiFont() dvui.Font {
    return dvui.themeGet().font_body.withFamily(emoji_family);
}

/// Codepoints routed to the monochrome emoji font. Kept to clearly-emoji blocks
/// — arrows / CJK / punctuation stay in NotoSansCJK, which covers them.
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

/// Add `text` to a text layout, drawing emoji codepoints in the bundled emoji
/// font and everything else in the default (body) font. dvui renders a whole
/// run in one font with no per-glyph fallback, so we split into same-class runs.
///
/// This is the single place that handles mixed script/symbol text — use it (or
/// `richLabel`) for ANY UI text that might contain emoji/symbols, so glyphs like
/// ⚙ / ⚠ / 🖼 render instead of tofu boxes.
pub fn addRich(tl: *dvui.TextLayoutWidget, text: []const u8) void {
    const emoji_font = emojiFont();
    var start: usize = 0;
    var i: usize = 0;
    var cur_emoji = false;
    var have = false;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const cp: u21 = if (n > 1 and i + n <= text.len)
            (std.unicode.utf8Decode(text[i..][0..n]) catch 0xFFFD)
        else
            text[i];
        const e = isEmoji(cp);
        if (!have) {
            cur_emoji = e;
            have = true;
        } else if (e != cur_emoji) {
            tl.addText(text[start..i], if (cur_emoji) .{ .font = emoji_font } else .{});
            start = i;
            cur_emoji = e;
        }
        i += n;
    }
    if (start < text.len) tl.addText(text[start..], if (cur_emoji) .{ .font = emoji_font } else .{});
}

/// A short label that may contain emoji/symbols, rendered through a text layout
/// so `addRich`'s per-run font fallback applies. Drop-in for `dvui.label` at
/// symbol-bearing sites (plain labels/buttons can't mix fonts within a run).
pub fn richLabel(src: std.builtin.SourceLocation, text: []const u8, opts: dvui.Options) void {
    var tl = dvui.textLayout(src, .{}, opts);
    defer tl.deinit();
    addRich(tl, text);
}

/// Register the bundled font and point the current theme's text styles at it.
/// Must run inside a `Window.begin`/`end` pair (needs the current window).
pub fn install() (std.mem.Allocator.Error || dvui.FontError)!void {
    try dvui.addFont(family, regular_bytes, null);
    // dvui resolves a bold text style as "<family> Bold"; alias it to the
    // regular bytes so headings/expanders don't log a missing-font warning
    // (they just won't be visually bolder — no separate bold asset shipped).
    try dvui.addFont(family ++ " Bold", regular_bytes, null);
    try dvui.addFont(emoji_family, emoji_bytes, null);
    var theme = dvui.themeGet();
    theme.font_body = theme.font_body.withFamily(family);
    theme.font_heading = theme.font_heading.withFamily(family);
    theme.font_title = theme.font_title.withFamily(family);
    theme.font_mono = theme.font_mono.withFamily(family);
    dvui.themeSet(theme);
}
