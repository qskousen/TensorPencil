//! Bundled fonts + per-codepoint run splitting.
//!
//! dvui renders one face per text run (no per-glyph fallback), so any string
//! that leaves one face's coverage shows tofu. `addStyled` is the single place
//! that maps (codepoint, style) -> face: it splits text into same-face runs so
//! every glyph lands somewhere that has it. Use it (or `addRich`/`richLabel`)
//! for ANY user- or LLM-visible text.
//!
//! Three tiers, resolved per codepoint:
//!
//!   1. IBM Plex Sans / Plex Mono — the design's face. Latin, Greek, Cyrillic,
//!      Vietnamese. 895 codepoints, all BMP.
//!   2. Noto Sans CJK / Noto Sans Mono CJK — everything else with a glyph:
//!      CJK, arrows, math, box drawing, the long tail.
//!   3. Noto Emoji — monochrome outlines, since dvui's rasterizer is monochrome
//!      and cannot use a color emoji font.
//!
//! Tier 1 membership is decided by parsing each Plex face's own `cmap` at
//! startup (`Coverage`), never by a hand-written codepoint range table: the
//! table drifts the moment a font is updated, and the failure is a tofu box in
//! a screenshot rather than an error.
//!
//! UI affordance marks (`▾ ▸ ✕ ⌘ ⏏`) are NOT text here. Plex has none of them
//! and three are missing from every face we bundle, so they are drawn as vector
//! icons (see style.zig `mark`), which also keeps them crisp at any scale.
//!
//! dvui's `Font.size` is the height of a capital M, not an em size. That gives
//! the cross-face cap-height normalization for free: Plex and Noto CJK at the
//! same `size` have the same cap height, so a CJK run in a Latin line does not
//! change the line's optical size. See style.zig for the nominal-px conversion.
const std = @import("std");
const dvui = @import("dvui");

/// Sans tier 1. Regular / SemiBold / Italic / SemiBold Italic live under this
/// one family; dvui's `Weight` is binary (normal|bold), so 600 is registered as
/// this family's `.bold`.
pub const sans = "PlexSans";
/// Plex Sans Medium (500). A separate family for the same reason: dvui has no
/// third weight, and the design uses 500 for every UI label and button.
pub const sans_med = "PlexSansMed";
pub const mono = "PlexMono";
pub const mono_med = "PlexMonoMed";
/// Tier 2, the coverage backstop.
pub const cjk = "NotoSansCJK";
pub const cjk_mono = "NotoSansMonoCJK";
pub const emoji_family = "NotoEmoji";

const plex_sans_regular = @embedFile("fonts/IBMPlexSans-Regular.ttf");
const plex_sans_medium = @embedFile("fonts/IBMPlexSans-Medium.ttf");
const plex_sans_semibold = @embedFile("fonts/IBMPlexSans-SemiBold.ttf");
const plex_sans_italic = @embedFile("fonts/IBMPlexSans-Italic.ttf");
const plex_sans_semibold_italic = @embedFile("fonts/IBMPlexSans-SemiBoldItalic.ttf");
const plex_mono_regular = @embedFile("fonts/IBMPlexMono-Regular.ttf");
const plex_mono_medium = @embedFile("fonts/IBMPlexMono-Medium.ttf");
const cjk_regular = @embedFile("fonts/NotoSansCJK-Regular.ttc");
const cjk_bold = @embedFile("fonts/NotoSansCJK-Bold.ttc");
const cjk_mono_regular = @embedFile("fonts/NotoSansMonoCJKjp-Regular.otf");
const emoji_bytes = @embedFile("fonts/NotoEmoji.ttf");

/// Registered via `Theme.embedded_fonts` (plain `dvui.addFont` can only
/// register normal weight/style). dvui dedups sources by bytes pointer, so one
/// file cannot alias two weights, which is why Medium gets its own family.
const sources = [_]dvui.Font.Source{
    .{ .family = dvui.Font.array(sans), .bytes = plex_sans_regular },
    .{ .family = dvui.Font.array(sans), .weight = .bold, .bytes = plex_sans_semibold },
    .{ .family = dvui.Font.array(sans), .style = .italic, .bytes = plex_sans_italic },
    .{ .family = dvui.Font.array(sans), .weight = .bold, .style = .italic, .bytes = plex_sans_semibold_italic },
    .{ .family = dvui.Font.array(sans_med), .bytes = plex_sans_medium },
    .{ .family = dvui.Font.array(mono), .bytes = plex_mono_regular },
    .{ .family = dvui.Font.array(mono_med), .bytes = plex_mono_medium },
    .{ .family = dvui.Font.array(cjk), .bytes = cjk_regular },
    .{ .family = dvui.Font.array(cjk), .weight = .bold, .bytes = cjk_bold },
    .{ .family = dvui.Font.array(cjk_mono), .bytes = cjk_mono_regular },
    .{ .family = dvui.Font.array(emoji_family), .bytes = emoji_bytes },
};

// ---------------------------------------------------------------- coverage

/// A face's BMP coverage as a bitset. Plex is BMP-only (verified: its `cmap`
/// has a single format-4 subtable and no codepoint above U+FFFD), so a BMP
/// bitset is exact for tier 1, and anything above U+FFFF goes to tier 2 or 3
/// without a lookup.
const Coverage = struct {
    words: [0x10000 / 64]u64 = @splat(0),

    fn mark(self: *Coverage, cp: u16) void {
        self.words[cp >> 6] |= @as(u64, 1) << @truncate(cp);
    }
    fn has(self: *const Coverage, cp: u21) bool {
        if (cp > 0xFFFF) return false;
        return self.words[cp >> 6] & (@as(u64, 1) << @truncate(cp)) != 0;
    }
    fn count(self: *const Coverage) usize {
        var n: usize = 0;
        for (self.words) |w| n += @popCount(w);
        return n;
    }
};

var cov_sans: Coverage = .{};
var cov_mono: Coverage = .{};
var cov_ready = false;

fn rdU16(b: []const u8, off: usize) u16 {
    if (off + 2 > b.len) return 0;
    return std.mem.readInt(u16, b[off..][0..2], .big);
}
fn rdU32(b: []const u8, off: usize) u32 {
    if (off + 4 > b.len) return 0;
    return std.mem.readInt(u32, b[off..][0..4], .big);
}

/// Byte offset of an sfnt table, or null. Plain TTF/OTF only (the Plex faces);
/// the Noto collections are tier 2 and never need a coverage lookup.
fn tableOffset(ttf: []const u8, tag: *const [4]u8) ?usize {
    const n = rdU16(ttf, 4);
    for (0..n) |i| {
        const rec = 12 + 16 * i;
        if (rec + 16 > ttf.len) return null;
        if (std.mem.eql(u8, ttf[rec..][0..4], tag)) return rdU32(ttf, rec + 8);
    }
    return null;
}

/// Fill `out` from a face's Windows-BMP (platform 3, encoding 1) format-4
/// subtable. Codepoints that map to glyph 0 are NOT coverage: a font may map a
/// whole range and resolve most of it to .notdef, which renders as tofu just
/// like no mapping at all.
fn parseCoverage(ttf: []const u8, out: *Coverage) void {
    const cmap = tableOffset(ttf, "cmap") orelse return;
    const n_sub = rdU16(ttf, cmap + 2);
    var sub: ?usize = null;
    for (0..n_sub) |i| {
        const rec = cmap + 4 + 8 * i;
        const plat = rdU16(ttf, rec);
        const enc = rdU16(ttf, rec + 2);
        const off = cmap + rdU32(ttf, rec + 4);
        if (plat == 3 and enc == 1 and rdU16(ttf, off) == 4) sub = off;
    }
    const t = sub orelse return;

    const seg2 = rdU16(ttf, t + 6);
    const seg = seg2 / 2;
    const ends = t + 14;
    const starts = ends + seg2 + 2;
    const deltas = starts + seg2;
    const ranges = deltas + seg2;

    for (0..seg) |i| {
        const end = rdU16(ttf, ends + 2 * i);
        const start = rdU16(ttf, starts + 2 * i);
        if (start > end or start == 0xFFFF) continue;
        const delta = rdU16(ttf, deltas + 2 * i);
        const ro = rdU16(ttf, ranges + 2 * i);
        var cp: u32 = start;
        while (cp <= end) : (cp += 1) {
            const gid: u16 = if (ro == 0)
                @truncate(cp +% delta)
            else blk: {
                // The offset is from the range-offset slot itself, in glyph
                // ids, hence the doubled index arithmetic.
                const gi = ranges + 2 * i + ro + 2 * (cp - start);
                const g = rdU16(ttf, gi);
                break :blk if (g == 0) 0 else @truncate(g +% delta);
            };
            if (gid != 0) out.mark(@truncate(cp));
        }
    }
}

// ------------------------------------------------------------ tier routing

/// How one role's runs escalate when tier 1 lacks a codepoint. `fallback_weight`
/// exists because Plex Medium has no Noto twin: a 500-weight run drops to Noto
/// Regular rather than jumping to Bold.
const Chain = struct {
    cover: *const Coverage,
    fallback: []const u8,
    fallback_weight: dvui.Font.Weight = .normal,
};

const sans_chain: Chain = .{ .cover = &cov_sans, .fallback = cjk };
const sans_bold_chain: Chain = .{ .cover = &cov_sans, .fallback = cjk, .fallback_weight = .bold };
const mono_chain: Chain = .{ .cover = &cov_mono, .fallback = cjk_mono };

fn chainFor(fam: []const u8, weight: dvui.Font.Weight) Chain {
    if (std.mem.eql(u8, fam, mono) or std.mem.eql(u8, fam, mono_med)) return mono_chain;
    return if (weight == .bold) sans_bold_chain else sans_chain;
}

/// Register the bundled fonts, build the tier-1 coverage index, and point the
/// theme's text styles at the Plex families. Must run inside a
/// `Window.begin`/`end` pair; `themeSet` pulls `embedded_fonts` into the
/// window's font database.
pub fn install() void {
    if (!cov_ready) {
        parseCoverage(plex_sans_regular, &cov_sans);
        parseCoverage(plex_mono_regular, &cov_mono);
        cov_ready = true;
    }
    var theme = dvui.themeGet();
    theme.embedded_fonts = &sources;
    theme.font_body = theme.font_body.withFamily(sans);
    theme.font_heading = theme.font_heading.withFamily(sans).withWeight(.bold);
    theme.font_title = theme.font_title.withFamily(sans).withWeight(.bold);
    theme.font_mono = theme.font_mono.withFamily(mono);
    dvui.themeSet(theme);
}

/// Inline text style, as produced by the markdown parser. `addStyled` turns
/// this plus per-codepoint coverage into a concrete face per run.
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

/// Codepoints routed to the emoji face ahead of any coverage test, because the
/// text faces map some of them to flat, wrong-looking glyphs. Kept to clearly
/// pictographic blocks; arrows and punctuation stay on the text tiers.
fn isEmoji(cp: u21) bool {
    return switch (cp) {
        0x1F000...0x1FAFF => true, // emoji + pictographs (incl. flags, skin tones)
        0x23E9...0x23FA => true, // media controls (⏸ ⏯ ⏹ ...), no text face has these
        0x2600...0x27BF => true, // misc symbols + dingbats (⚙ ✅ ...)
        0x2B00...0x2BFF => true, // misc symbols & arrows (emoji-presentation)
        0xFE00...0xFE0F => true, // variation selectors, keep with the emoji run
        0x200D => true, // ZWJ, keep emoji sequences together
        else => false,
    };
}

/// Resolve the face for one codepoint under `style`, deriving size (and any
/// strike the caller set) from `base`.
fn fontFor(cp: u21, style: Style, base: dvui.Font) dvui.Font {
    var f = base;
    if (style.strike) f = f.withStrike(.{});
    if (isEmoji(cp)) return f.withFamily(emoji_family);

    if (style.code) {
        // Mono deliberately does not fall back to mono for content: the role
        // only ever renders digits, identifiers and units, and a proportional
        // CJK run beats a tofu row. No bold mono face is bundled.
        const fam: []const u8 = if (std.mem.eql(u8, base.familyName(), mono_med)) mono_med else mono;
        const c = chainFor(fam, .normal);
        if (c.cover.has(cp)) return f.withFamily(fam).withWeight(.normal);
        return f.withFamily(c.fallback).withWeight(.normal);
    }

    const bold = style.bold or base.weight == .bold;
    const fam = base.familyName();
    // Medium (500) is its own family, and only in the upright roman: a bold or
    // italic run inside 500-weight text resolves to the regular family's own
    // SemiBold / Italic face rather than a face that does not exist.
    const primary: []const u8 = if (std.mem.eql(u8, fam, sans_med) and !bold and !style.italic)
        sans_med
    else if (std.mem.eql(u8, fam, mono) or std.mem.eql(u8, fam, mono_med))
        fam
    else
        sans;

    const c = chainFor(primary, if (bold) .bold else .normal);
    if (c.cover.has(cp)) {
        var r = f.withFamily(primary);
        if (!std.mem.eql(u8, primary, sans_med)) r = r.withWeight(if (bold) .bold else .normal);
        return if (style.italic) r.withStyle(.italic) else r;
    }
    // Tier 2. Italic is dropped rather than faked: Noto CJK ships no italic,
    // and CJK is conventionally emphasized without one.
    return f.withFamily(c.fallback).withWeight(c.fallback_weight);
}

/// Add `text` to a text layout, split into same-face runs. `opts` applies to
/// every run (colors, background for inline code, ...); `opts.font` (or the
/// theme body font) sets the base size and role.
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

/// Plain-styled `addStyled`, for the common "show this string with full
/// coverage" case.
pub fn addRich(tl: *dvui.TextLayoutWidget, text: []const u8) void {
    addStyled(tl, text, .{}, .{});
}

/// A short label that may leave tier 1, rendered through a text layout so the
/// per-run fallback applies. Drop-in for `dvui.label` at such sites (a plain
/// label cannot mix faces within a run).
pub fn richLabel(src: std.builtin.SourceLocation, text: []const u8, opts: dvui.Options) void {
    var tl = dvui.textLayout(src, .{}, noBackground(opts));
    defer tl.deinit();
    addRich(tl, text);
}

/// A label is a label, not a panel. `TextLayoutWidget.defaults` sets
/// `.background = true` with `.style = .content`, so a text layout left alone
/// paints a CANVAS-COLOURED block behind itself — which over a sidebar row or
/// any other surface covers what it is sitting on. A caller that genuinely
/// wants a background still says so.
fn noBackground(opts: dvui.Options) dvui.Options {
    var o = opts;
    if (o.background == null) o.background = false;
    if (o.padding == null) o.padding = .{};
    return o;
}

/// `richLabel` that never wraps: for row titles and other single-line slots in
/// a fixed-width rail, where wrapping would reflow the whole list. Pair it with
/// `style.ellipsize` so the overflow is cut deliberately rather than clipped.
pub fn richLine(src: std.builtin.SourceLocation, text: []const u8, opts: dvui.Options) void {
    // Built by hand rather than via `dvui.textLayout`, purely to SKIP
    // `processEvents`. A text layout claims the mouse to support text
    // selection, so a row title drawn with one swallowed every click that
    // landed on the words — you had to aim at the gap between the text and the
    // row's edge to open a conversation. A one-line label in a rail has nothing
    // worth selecting, and being inert lets the click reach the row.
    var tl: dvui.TextLayoutWidget = undefined;
    tl.init(src, .{ .break_lines = false }, noBackground(opts));
    defer tl.deinit();
    addStyled(&tl, text, .{}, .{ .font = opts.font, .color_text = opts.color_text });
}

// ------------------------------------------------------------------- tests

fn testCoverage() void {
    if (!cov_ready) {
        parseCoverage(plex_sans_regular, &cov_sans);
        parseCoverage(plex_mono_regular, &cov_mono);
        cov_ready = true;
    }
}

test "plex coverage parses to the face's real cmap" {
    testCoverage();
    // Both Plex faces report 895 / 983 mapped codepoints from their format-4
    // subtable; a parse that silently found nothing would pass every "has ASCII"
    // check below by falling through to tier 2, so assert the magnitude too.
    try std.testing.expect(cov_sans.count() > 800);
    try std.testing.expect(cov_mono.count() > 900);
    for ("ABCXYZabcxyz0189") |ch| try std.testing.expect(cov_sans.has(ch));
    // Greek, Cyrillic, Vietnamese: the rest of tier 1's remit.
    try std.testing.expect(cov_sans.has('Ω'));
    try std.testing.expect(cov_sans.has('Д'));
    try std.testing.expect(cov_sans.has('ế'));
}

test "codepoints Plex lacks escalate past tier 1" {
    testCoverage();
    // The marks this design would otherwise want as text. All three are absent
    // from every face we bundle, which is why style.zig draws them as icons;
    // the point here is that coverage REPORTS them absent rather than routing
    // them to a face that renders tofu.
    for ([_]u21{ '▾', '▸', '✕' }) |cp| try std.testing.expect(!cov_sans.has(cp));
    // CJK and emoji are simply not tier 1.
    try std.testing.expect(!cov_sans.has('日'));
    try std.testing.expect(!cov_sans.has(0x1F98A));
}

test "run splitting sends each codepoint to a face that has it" {
    testCoverage();
    const base = dvui.Font{ .family = dvui.Font.array(sans), .size = 12 };
    const Case = struct { cp: u21, want: []const u8 };
    for ([_]Case{
        .{ .cp = 'A', .want = sans },
        .{ .cp = '·', .want = sans }, // the design's separator, Plex has it
        .{ .cp = '×', .want = sans },
        .{ .cp = '→', .want = sans },
        .{ .cp = '日', .want = cjk }, // past tier 1
        .{ .cp = '⌘', .want = cjk }, // Plex lacks it, Noto CJK has it
        // Markdown's own decorations. Plex has the level-0 bullet and nothing
        // below it, so these are the run splitter's job, not the caller's --
        // emitting them with a bare `addText` is what put tofu in the probe.
        .{ .cp = '•', .want = sans },
        .{ .cp = '◦', .want = cjk },
        .{ .cp = '▪', .want = cjk },
        .{ .cp = '▎', .want = cjk },
        .{ .cp = '─', .want = cjk },
        .{ .cp = 0x1F98A, .want = emoji_family },
        .{ .cp = '⏸', .want = emoji_family },
    }) |c| {
        const f = fontFor(c.cp, .{}, base);
        errdefer std.debug.print("cp U+{X} -> {s}, want {s}\n", .{ c.cp, f.familyName(), c.want });
        try std.testing.expectEqualStrings(c.want, f.familyName());
    }
}

test "medium weight keeps its own family but borrows real bold and italic faces" {
    testCoverage();
    const med = dvui.Font{ .family = dvui.Font.array(sans_med), .size = 11 };
    // Upright 500 stays on the Medium family...
    try std.testing.expectEqualStrings(sans_med, fontFor('A', .{}, med).familyName());
    // ...but there is no Medium Bold or Medium Italic file, so those resolve to
    // the regular family's SemiBold / Italic rather than a face that is absent.
    const b = fontFor('A', .{ .bold = true }, med);
    try std.testing.expectEqualStrings(sans, b.familyName());
    try std.testing.expectEqual(dvui.Font.Weight.bold, b.weight);
    const i = fontFor('A', .{ .italic = true }, med);
    try std.testing.expectEqualStrings(sans, i.familyName());
    try std.testing.expectEqual(dvui.Font.Style.italic, i.style);
}

test "mono content leaves mono rather than showing tofu" {
    testCoverage();
    const m = dvui.Font{ .family = dvui.Font.array(mono), .size = 10 };
    try std.testing.expectEqualStrings(mono, fontFor('4', .{ .code = true }, m).familyName());
    // A CJK chat title in a mono slot renders proportional, not as boxes.
    try std.testing.expectEqualStrings(cjk_mono, fontFor('日', .{ .code = true }, m).familyName());
}
