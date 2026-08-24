//! The tp-gui design system: one palette, one type scale, one set of radii,
//! and the handful of primitives every screen shares. Nothing here knows about
//! chat, diffusion or VRAM — it is the vocabulary the views are written in.
//!
//! Two rules the whole UI depends on:
//!
//!   AMBER means the machine is doing something (progress, a running job, VRAM
//!   pressure). BLUE means the user owns it (selection, focus, a value they
//!   set). Never amber a button, never blue a progress bar. Against neutrals,
//!   these two are the only accents on the screen, which is what makes either
//!   of them readable at a glance.
//!
//!   Hairlines are always white at low alpha over the surface, never a solid
//!   grey, so one edge color works over `chrome`, `rail` and `canvas` alike.
//!
//! Sizes here are NOMINAL px (the size a designer means by "12.5px text"), but
//! dvui's `Font.size` is the height of a capital M. `sans`/`monoFont` convert,
//! so `cap_ratio` is the single place the two worlds meet. A pleasant
//! side effect: two faces at the same dvui size have the same cap height, so a
//! CJK fallback run cannot change a line's optical size (spec §12.4 asks for
//! exactly that normalization, and dvui gives it for free).
const std = @import("std");
const dvui = @import("dvui");
const fonts = @import("fonts.zig");

pub const Color = dvui.Color;

/// Surfaces, text and accents. Reserved meanings are marked; a color with a
/// meaning is never reused for decoration.
pub const C = struct {
    // surfaces
    pub const canvas = Color.fromHex("#0b0d0f"); // chat pane / card bg
    pub const chrome = Color.fromHex("#101317"); // title bar, status bar
    pub const rail = Color.fromHex("#0e1114"); // side rails
    pub const sunken = Color.fromHex("#12161a"); // composer, track bg
    pub const spark_bg = Color.fromHex("#0f1215"); // sparkline well
    pub const card = Color.fromHex("#0f1316"); // tool-call card
    pub const raised = Color.fromHex("#1c2126"); // buttons
    pub const raised_dim = Color.fromHex("#151a1e"); // secondary buttons
    pub const chip = Color.fromHex("#141a1e"); // quick-setting chips
    pub const chip_hi = Color.fromHex("#171b1f"); // title-bar chips
    pub const sel_row = Color.fromHex("#1a1f24"); // selected conversation
    pub const bubble_user = Color.fromHex("#1e2429");
    pub const seg_track = Color.fromHex("#181c20"); // Chat/Studio toggle well
    pub const queue_active = Color.fromHex("#141a1f");
    pub const queue_idle = Color.fromHex("#111417");
    pub const meter_track = Color.fromHex("#22282d");

    // text
    pub const text_hi = Color.fromHex("#e6e8ea"); // primary
    pub const text = Color.fromHex("#c9cdd1"); // agent prose
    pub const text_dim = Color.fromHex("#8b9196"); // secondary / labels
    pub const text_faint = Color.fromHex("#6d7378"); // mono captions
    pub const text_ghost = Color.fromHex("#5c6368"); // placeholders, heads
    pub const text_ink = Color.fromHex("#0b0d0f"); // on light fills

    // accents — RESERVED MEANINGS, see the module header
    pub const amber = Color.fromHex("#f0a03c");
    pub const blue = Color.fromHex("#6ea8f0");
    pub const danger = Color.fromHex("#d9584f");

    // Telemetry sparklines. The design gave all three the same blue; three
    // identical graphs side by side are three things you have to read the
    // label of. A hue each makes them tellable apart at a glance, which is the
    // entire point of a sparkline.
    pub const meter_gpu = Color.fromHex("#5cc08a");
    pub const meter_cpu = Color.fromHex("#6ea8f0");
    /// VRAM's RESTING colour only. It escalates to `amber` at 80% and `danger`
    /// above 95%: it is the resource that actually fails, so it is the only one
    /// whose colour carries a threshold.
    pub const meter_vram = Color.fromHex("#b083e0");

    // VRAM segments: the LLM side is cool, the diffusion side warm, so which
    // engine owns the card reads before any label does.
    pub const vram_sys = Color.fromHex("#3f464c"); // hatch fg, other programs
    pub const vram_sys_alt = Color.fromHex("#343b41"); // hatch bg
    pub const vram_llm = Color.fromHex("#3f9c8f"); // weights
    pub const vram_ctx = Color.fromHex("#4f7fb0"); // kv cache
    pub const vram_ovh = Color.fromHex("#6d7378"); // runtime overhead
    pub const vram_free = Color.fromHex("#1b2024"); // rail gap
    pub const vram_lat = Color.fromHex("#d9584f"); // latents
    pub const vram_vae = Color.fromHex("#9a6fd8");
    pub const vram_te = Color.fromHex("#e8c33f"); // text encoder
    pub const vram_dit = Color.fromHex("#e08a2e"); // unet / dit
};

/// Edges. Always alpha over whatever is beneath.
pub const hairline = Color{ .r = 255, .g = 255, .b = 255, .a = 18 }; // chrome edges
pub const hairline_soft = Color{ .r = 255, .g = 255, .b = 255, .a = 15 }; // inside rails
pub const hairline_hi = Color{ .r = 255, .g = 255, .b = 255, .a = 28 }; // composer
pub const hover_wash = Color{ .r = 255, .g = 255, .b = 255, .a = 10 }; // row hover

/// A translucent accent, for BORDERS and hairlines, where sitting over an
/// unknown surface is the point.
pub fn tint(c: Color, alpha: u8) Color {
    return .{ .r = c.r, .g = c.g, .b = c.b, .a = alpha };
}

/// An accent laid over a known surface at `t`, resolved to an OPAQUE color.
/// Use this for FILLS. dvui fills a widget's whole border rect with the border
/// color and then insets the background, so a translucent background over a
/// colored border shows the border rather than the blend — which is how a 5%
/// blue wash renders as a solid blue slab.
pub fn over(base: Color, accent: Color, t: f32) Color {
    var c = base.lerp(accent, t);
    c.a = 255;
    return c;
}

// ---------------------------------------------------------------- type

/// Cap height as a fraction of the em, for the bundled Plex faces. Every
/// nominal size in the app passes through this; the test at the bottom reads it
/// back out of the font file so a face swap cannot silently resize the UI.
pub const cap_ratio: f32 = 0.698;

/// The hhea ascent-to-descent span, in ems, of the bundled Plex faces (all
/// seven share 1025/-275). dvui's `line_height_factor` multiplies THIS, not the
/// cap height, so a designer's "1.65 line height" is 1.65/1.30 as a factor.
/// Getting this wrong is invisible in a unit test and doubles every paragraph's
/// leading on screen.
pub const natural_line: f32 = 1.30;

/// A type role: nominal px, plus the line height as a multiple of that nominal
/// size (i.e. what a designer means by "12.5/1.65"), converted into dvui's two
/// different worlds — cap height for size, hhea span for leading.
fn role(family: []const u8, nominal: f32, line: f32, weight: dvui.Font.Weight) dvui.Font {
    return (dvui.Font{})
        .withFamily(family)
        .withSize(nominal * cap_ratio)
        .withWeight(weight)
        .withLineHeight(line / natural_line);
}

/// The type scale. Names are roles, never faces: a call site asks for `F.ui`,
/// not for Plex Sans Medium 11.
pub const F = struct {
    /// App name, 12/600.
    pub const app = role(fonts.sans, 12, 1.0, .bold);
    /// UI label / button, 11/500.
    pub const ui = role(fonts.sans_med, 11, 1.0, .normal);
    /// Smaller UI label, 10.5/500.
    pub const ui_sm = role(fonts.sans_med, 10.5, 1.0, .normal);
    /// Conversation row title, 11.5/400.
    pub const row = role(fonts.sans, 11.5, 1.35, .normal);
    /// Row title in a rail that must not reflow, 11/500.
    pub const row_hi = role(fonts.sans_med, 11, 1.2, .normal);
    /// Chat prose and bubbles, 12.5/1.65. The one role that carries long text.
    pub const prose = role(fonts.sans, 12.5, 1.65, .normal);
    /// Composer placeholder, 12.5/1.0.
    pub const input = role(fonts.sans, 12.5, 1.2, .normal);
    /// Metadata / params, 10.5/400. A SCANNING role: chips, one-line captions,
    /// figures compared against each other. Not for anything read left to right.
    pub const mono = role(fonts.mono, 10.5, 1.0, .normal);
    /// Monospaced text that is READ, not scanned: a tool call's body, an image
    /// card's prompt. Sized against `prose` rather than `mono`, because it sits in
    /// the reading column and is the same kind of content, and given real leading
    /// since it wraps to several lines.
    pub const code = role(fonts.mono, 12, 1.45, .normal);
    /// The smallest role, for dense captions that are scanned, not read.
    pub const mono_sm = role(fonts.mono, 10, 1.0, .normal);
    /// Status-bar legend and meter labels. The design says 8.5, which is a size
    /// you can see but not read; these are numbers a user compares against each
    /// other, so they get two steps up.
    pub const mono_legend = role(fonts.mono, 11.5, 1.0, .normal);
    /// Section head (TODAY, RECENT), 9.5/500 uppercase.
    pub const mono_hd = role(fonts.mono_med, 9.5, 1.0, .normal);
    /// Telemetry value (the big percentage on a meter).
    pub const mono_val = role(fonts.mono_med, 12.5, 1.0, .normal);
    /// Sub-lines inside rail rows, 9.5/400.
    pub const mono_row = role(fonts.mono, 9.5, 1.25, .normal);
};

// ---------------------------------------------------------------- geometry

pub const R = struct {
    pub const chip = dvui.Rect.all(5);
    pub const button = dvui.Rect.all(6);
    pub const panel = dvui.Rect.all(7); // queue rows, New chat
    pub const seg = dvui.Rect.all(8); // segmented-toggle well
    pub const card = dvui.Rect.all(9);
    pub const input = dvui.Rect.all(10);
    pub const track = dvui.Rect.all(2); // vram bar, sparkline well
    pub const none = dvui.Rect.all(0);
};

/// Fixed band sizes. The chat column is what flexes.
pub const Layout = struct {
    pub const title_h: f32 = 44;
    /// OUTER widths, chrome included. Each rail subtracts its own padding and
    /// border to get a content width, so this one number stays the thing the
    /// design specifies and the chat column can be sized by subtracting it.
    pub const sidebar_w: f32 = 206;
    pub const sidebar_pad: f32 = 12;
    pub const rail_w: f32 = 272;
    pub const status_h: f32 = 74;
    /// Widest a user bubble / agent paragraph / card is allowed to run. Long
    /// measure is the fastest way to make prose unreadable.
    pub const bubble_max: f32 = 400;
    pub const prose_max: f32 = 560;
    pub const card_max: f32 = 600;
};

/// A 1px border on one edge only, as `Options.border` wants it.
pub const Edge = struct {
    pub const top = dvui.Rect{ .y = 1 };
    pub const bottom = dvui.Rect{ .h = 1 };
    pub const left = dvui.Rect{ .x = 1 };
    pub const right = dvui.Rect{ .w = 1 };
    pub const all = dvui.Rect.all(1);
};

// ---------------------------------------------------------------- theme

/// Build the dvui theme from the palette, so stock widgets (buttons, text
/// entries, scrollbars, dialogs) land in the same world as the hand-drawn
/// chrome instead of fighting it.
pub fn theme() dvui.Theme {
    return .{
        .name = "TensorPencil",
        .dark = true,
        .focus = C.blue,
        .text_select = tint(C.blue, 70),
        .fill = C.canvas,
        .fill_hover = C.raised_dim,
        .fill_press = C.raised,
        .text = C.text,
        .border = hairline,
        .font_body = F.prose,
        .font_heading = role(fonts.sans, 12.5, 1.4, .bold),
        .font_title = role(fonts.sans, 15, 1.3, .bold),
        .font_mono = F.mono,
        .max_default_corner_radius = 8,
        .control = .{
            .fill = C.raised,
            .fill_hover = C.raised.lighten(6),
            .fill_press = C.raised.lighten(12),
            .text = C.text_hi,
            .border = hairline_hi,
        },
        .window = .{
            .fill = C.chrome,
            .text = C.text,
            .border = hairline,
        },
        .highlight = .{
            .fill = tint(C.blue, 40),
            .fill_hover = tint(C.blue, 60),
            .text = C.text_hi,
            .border = tint(C.blue, 90),
        },
        .err = .{
            .fill = tint(C.danger, 40),
            .fill_hover = tint(C.danger, 70),
            .text = C.text_hi,
            .border = C.danger,
        },
    };
}

/// A copy of the theme with a TRANSPARENT focus color, for text entries that
/// sit inside our own chrome.
///
/// `TextEntryWidget` unconditionally strokes a 2px focus ring on its OWN border
/// rect. Our entries are stripped of background and padding so the surrounding
/// chip or composer frame provides the chrome, which puts that ring a pixel
/// from the glyphs and makes selected text unreadable. Handing the entry this
/// theme removes the ring; the caller then draws focus on the outer edge, where
/// there is room for it. The caret survives, because `drawCursor` reads the
/// CURRENT theme rather than the widget's.
var no_focus: dvui.Theme = undefined;

pub fn noFocusTheme() *const dvui.Theme {
    return &no_focus;
}

/// Register the fonts and install the theme. Must run inside a
/// `Window.begin`/`end` pair.
pub fn install() void {
    fonts.install(); // registers the sources + builds tier-1 coverage
    var t = theme();
    t.embedded_fonts = dvui.themeGet().embedded_fonts; // keep what fonts.install put there
    dvui.themeSet(t);
    no_focus = t;
    no_focus.focus = .{ .r = 0, .g = 0, .b = 0, .a = 0 };
}

// ---------------------------------------------------------------- marks

/// UI affordance marks. These are vector icons, NOT text: Plex has none of
/// `▾ ▸ ✕ ⌘ ⏏`, and three of them are missing from every face we bundle, so as
/// glyphs they would be tofu. As icons they also stay crisp at any scale.
pub const Mark = enum {
    /// A dropdown's "there is a menu here".
    caret_down,
    /// A collapsed disclosure.
    disclosure,
    /// An expanded disclosure.
    disclosure_open,
    close,
    /// "Leaves this screen" (All settings in Studio ↗).
    external,
    check,
    pause,
    play,
    eject,
    plus,
    gear,

    fn tvg(self: Mark) []const u8 {
        return switch (self) {
            .caret_down => dvui.entypo.chevron_small_down,
            .disclosure => dvui.entypo.triangle_right,
            .disclosure_open => dvui.entypo.triangle_down,
            .close => dvui.entypo.cross,
            .external => dvui.entypo.popup, // box-with-arrow, the ↗ in the mockup
            .check => dvui.entypo.check,
            .pause => dvui.entypo.controller_pause,
            .play => dvui.entypo.controller_play,
            .eject => dvui.entypo.log_out, // entypo has no ⏏; "unload" reads the same
            .plus => dvui.entypo.plus,
            .gear => dvui.entypo.cog,
        };
    }
};

/// Draw a mark inline at `size` logical px in `color`.
pub fn mark(src: std.builtin.SourceLocation, m: Mark, size: f32, color: Color, opts: dvui.Options) void {
    var o = opts;
    o.min_size_content = .{ .w = size, .h = size };
    o.color_text = color;
    if (o.gravity_y == null) o.gravity_y = 0.5;
    dvui.icon(src, @tagName(m), m.tvg(), .{}, o);
}

// ---------------------------------------------------------------- widgets

pub const ChipOpts = struct {
    /// Show a caret and report clicks.
    dropdown: bool = false,
    font: dvui.Font = F.mono,
    fill: Color = C.chip,
    border: Color = hairline,
    text: Color = C.text_dim,
    id_extra: usize = 0,
    /// Left gap, for chips laid out in a row.
    margin_x: f32 = 0,
    /// `chipInput` only: the EDITABLE text's color. Separate from `text`, which
    /// colors a chip's label: a typed value wants full contrast where a label
    /// wants less, so null keeps the bright default and a caller overrides it
    /// only to show the field is inert.
    text_input: ?Color = null,
};

/// A small bordered pill: model names in the title bar, quick settings in the
/// composer. Returns true when clicked (only meaningful for `dropdown` chips,
/// but a plain chip reports it too so a caller can make one interactive).
pub fn chip(src: std.builtin.SourceLocation, text: []const u8, o: ChipOpts) bool {
    var wd: dvui.WidgetData = undefined;
    return chipEx(src, text, o, &wd);
}

/// `chip` that also reports its `WidgetData`, so a caller can anchor a floating
/// menu to the chip's own rect.
pub fn chipEx(src: std.builtin.SourceLocation, text: []const u8, o: ChipOpts, out: *dvui.WidgetData) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .id_extra = o.id_extra,
        .background = true,
        .color_fill = o.fill,
        .color_fill_hover = o.fill.lighten(5),
        .color_fill_press = o.fill.lighten(10),
        .border = Edge.all,
        .color_border = o.border,
        .corner_radius = R.button,
        .padding = .{ .x = 9, .y = 6, .w = if (o.dropdown) 6 else 9, .h = 6 },
        .margin = .{ .x = o.margin_x },
        .gravity_y = 0.5,
    });
    bw.processEvents();
    bw.drawBackground();
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
        defer row.deinit();
        dvui.labelNoFmt(@src(), text, .{}, .{
            .font = o.font,
            .color_text = o.text,
            .padding = .{},
            .gravity_y = 0.5,
        });
        if (o.dropdown) mark(@src(), .caret_down, 11, o.text, .{ .margin = .{ .x = 4 } });
    }
    const clicked = bw.clicked();
    out.* = bw.data().*;
    bw.drawFocus();
    bw.deinit();
    return clicked;
}

/// A chip that opens a menu. `choice` is mutated in place; returns true when
/// the user picked something new.
///
/// Hand-rolled rather than `dvui.dropdown`, which hardcodes `.avoid = .none` on
/// its floating menu and so always opens DOWNWARD. These chips live in the
/// composer at the foot of the window, where downward means most of the list is
/// off-screen and the control reads as broken. `.avoid = .vertical` flips it
/// above when there is no room below.
pub fn chipDropdown(src: std.builtin.SourceLocation, entries: []const []const u8, choice: *usize, o: ChipOpts) bool {
    var changed = false;
    var wrap = dvui.box(src, .{ .dir = .horizontal }, .{ .id_extra = o.id_extra });
    defer wrap.deinit();

    const id = wrap.data().id;
    var open = dvui.dataGet(null, id, "_open", bool) orelse false;

    var co = o;
    co.dropdown = true;
    co.id_extra = 0;
    const label = if (choice.* < entries.len) entries[choice.*] else "";
    var chip_wd: dvui.WidgetData = undefined;
    if (chipEx(@src(), label, co, &chip_wd)) open = !open;

    if (open) {
        const rs = chip_wd.borderRectScale().r;
        var fw = dvui.floatingMenu(@src(), .{
            .from = rs.toNatural(),
            .avoid = .vertical,
        }, .{});
        for (entries, 0..) |e, i| {
            if (dvui.menuItemLabel(@src(), e, .{}, .{
                .id_extra = i,
                .expand = .horizontal,
                .font = o.font,
                .corner_radius = R.chip,
            }) != null) {
                choice.* = i;
                changed = true;
                open = false;
                fw.close();
            }
        }
        // Dismissal: the menu takes focus when it opens, so losing it means the
        // user clicked away or pressed escape. Without this the flag stays set
        // and the menu reopens forever.
        //
        // The first-frame guard is on the MENU's id, not the chip's: on the
        // frame the menu is created it does not own focus yet, and guarding on
        // the (long-lived) chip closed it again before it ever drew.
        const still = dvui.focusedSubwindowId() == fw.data().id or dvui.firstFrame(fw.data().id);
        fw.deinit();
        if (!still) open = false;
    }

    dvui.dataSet(null, id, "_open", open);
    return changed;
}

pub const InputResult = struct {
    /// The user finished editing: Enter, or focus left the field.
    committed: bool = false,
    /// Being edited right now. A caller syncing the buffer from elsewhere must
    /// skip while this is true, or it overwrites what is being typed.
    editing: bool = false,
};

/// A chip-shaped text field with a dimmed, non-editable suffix ("1.8 MP").
/// Same skin as `chip`, so a typed value and a picked one read as the same
/// kind of control.
pub fn chipInput(src: std.builtin.SourceLocation, buf: []u8, suffix: []const u8, width: f32, o: ChipOpts) InputResult {
    var res: InputResult = .{};
    // Focus is shown on the CHIP's edge, not the entry's. Whether the entry had
    // focus is known from last frame, which is enough: the ring is decoration,
    // and being one frame late is invisible.
    const was_focused = dvui.dataGet(null, dvui.parentGet().extendId(src, o.id_extra), "_focus_ring", bool) orelse false;
    var box = dvui.box(src, .{ .dir = .horizontal }, .{
        .id_extra = o.id_extra,
        .background = true,
        .color_fill = o.fill,
        .border = Edge.all,
        .color_border = if (was_focused) C.blue else o.border,
        .corner_radius = R.button,
        .padding = .{ .x = 9, .y = 5, .w = 9, .h = 5 },
        .margin = .{ .x = o.margin_x },
        .gravity_y = 0.5,
    });
    defer box.deinit();

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = buf },
        .scroll_horizontal = false,
    }, .{
        .background = false,
        .border = .{},
        .padding = .{},
        .margin = .{},
        .min_size_content = .{ .w = width },
        .max_size_content = .width(width),
        .font = o.font,
        .color_text = o.text_input orelse C.text_hi,
        .gravity_y = 0.5,
        .theme = noFocusTheme(),
    });
    const id = te.data().id;
    const focused = dvui.focusedWidgetId() == id;
    res.editing = focused;
    dvui.dataSet(null, box.data().id, "_focus_ring", focused);

    // Enter commits. Deliberately NOT skipping already-handled events: the
    // text entry processes its own input inside `textEntry` above and marks the
    // key handled before this loop ever runs, so filtering on `handled` here
    // meant Enter was never seen. An Enter while THIS field has focus means
    // commit, whoever else looked at it.
    if (focused) {
        for (dvui.events()) |*e| {
            if (e.evt != .key) continue;
            const k = e.evt.key;
            if ((k.code == .enter or k.code == .kp_enter) and k.action == .down) {
                e.handled = true;
                res.committed = true;
                dvui.focusWidget(null, null, null);
            }
        }
    }
    te.deinit();

    // Focus leaving the field commits too, so a click elsewhere is not silently
    // discarded. Edge-triggered off last frame's state.
    const was = dvui.dataGet(null, id, "_had_focus", bool) orelse false;
    if (was and !focused) res.committed = true;
    dvui.dataSet(null, id, "_had_focus", focused);

    dvui.labelNoFmt(@src(), suffix, .{}, .{
        .font = o.font,
        .color_text = o.text,
        .padding = .{},
        .margin = .{ .x = 4 },
        .gravity_y = 0.5,
    });
    return res;
}

/// The Chat|Studio well: a sunken track with one lit segment. `active` is an
/// index into `labels`; returns the index the user picked, or null.
pub fn segmented(src: std.builtin.SourceLocation, labels: []const []const u8, active: usize) ?usize {
    var picked: ?usize = null;
    var well = dvui.box(src, .{ .dir = .horizontal }, .{
        .background = true,
        .color_fill = C.seg_track,
        .border = Edge.all,
        .color_border = hairline,
        .corner_radius = R.seg,
        .padding = dvui.Rect.all(2),
        .gravity_y = 0.5,
    });
    defer well.deinit();

    for (labels, 0..) |lbl, i| {
        const on = i == active;
        var bw: dvui.ButtonWidget = undefined;
        bw.init(@src(), .{}, .{
            .id_extra = i,
            .background = on,
            .color_fill = C.text_hi,
            .color_fill_hover = if (on) C.text_hi else hover_wash,
            .color_fill_press = if (on) C.text_hi else hover_wash,
            .corner_radius = R.button,
            .padding = .{ .x = 12, .y = 6, .w = 12, .h = 6 },
            .margin = if (i == 0) .{} else .{ .x = 2 },
        });
        bw.processEvents();
        bw.drawBackground();
        dvui.labelNoFmt(@src(), lbl, .{}, .{
            .font = F.ui,
            .color_text = if (on) C.text_ink else C.text_dim,
            .padding = .{},
            .gravity_x = 0.5,
            .gravity_y = 0.5,
        });
        if (bw.clicked()) picked = i;
        bw.drawFocus();
        bw.deinit();
    }
    return picked;
}

/// Truncate `text` to `max_w` logical px in `font`, appending an ellipsis.
/// Returns a slice of `buf`, or `text` itself when it already fits.
///
/// MEASURED, never counted: an ideograph is twice the width of a Latin letter,
/// so a character budget cuts a Japanese title at half the room it has and lets
/// a Latin one overflow. Only valid between `Window.begin`/`end`.
pub fn ellipsize(buf: []u8, text: []const u8, font: dvui.Font, max_w: f32) []const u8 {
    if (font.textSize(text).w <= max_w) return text;
    const ell = "…";
    const ell_w = font.textSize(ell).w;
    if (max_w <= ell_w) return "";

    // Walk whole codepoints so a cut never lands inside a UTF-8 sequence.
    var end: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const next = @min(text.len, i + n);
        if (font.textSize(text[0..next]).w + ell_w > max_w) break;
        end = next;
        i = next;
    }
    if (end + ell.len > buf.len) end = @min(end, buf.len -| ell.len);
    @memcpy(buf[0..end], text[0..end]);
    @memcpy(buf[end..][0..ell.len], ell);
    return buf[0 .. end + ell.len];
}

/// An uppercase mono section head (TODAY, RECENT).
pub fn sectionHead(src: std.builtin.SourceLocation, text: []const u8, opts: dvui.Options) void {
    var o = opts;
    o.font = F.mono_hd;
    o.color_text = C.text_ghost;
    if (o.padding == null) o.padding = .{ .x = 8, .h = 7 };
    dvui.labelNoFmt(src, text, .{}, o);
}

/// A vertical hairline the height of its parent's content, for splitting groups
/// inside a bar.
pub fn vsep(src: std.builtin.SourceLocation) void {
    var b = dvui.box(src, .{ .dir = .vertical }, .{
        .expand = .vertical,
        .min_size_content = .{ .w = 1 },
        .background = true,
        .color_fill = hairline,
        .margin = .{ .x = 4, .y = 2, .w = 4, .h = 2 },
    });
    b.deinit();
}

/// Fill `r` with 45° hatching: the "not ours to reclaim" texture, used for VRAM
/// held by other processes. Drawn as stripes under a clip rather than a tiled
/// texture so it stays correct at any DPI and needs no startup asset.
pub fn hatch(r: dvui.Rect.Physical, fg: Color, bg: Color, period: f32) void {
    if (r.w <= 0 or r.h <= 0) return;
    r.fill(.{}, .{ .color = bg });

    const prev = dvui.clip(r);
    defer dvui.clipSet(prev);

    var pb = dvui.Path.Builder.init(dvui.currentWindow().lifo());
    defer pb.deinit();

    // A stripe is a parallelogram sheared by its own height, so the run starts
    // one height to the left of the rect to cover the top-left corner.
    const w = @max(1.0, period * 0.5);
    var x = r.x - r.h;
    while (x < r.x + r.w) : (x += period) {
        pb.points.clearRetainingCapacity();
        pb.addPoint(.{ .x = x, .y = r.y + r.h });
        pb.addPoint(.{ .x = x + r.h, .y = r.y });
        pb.addPoint(.{ .x = x + r.h + w, .y = r.y });
        pb.addPoint(.{ .x = x + w, .y = r.y + r.h });
        pb.build().fillConvex(.{ .color = fg });
    }
}

/// A rolling bar chart. `values` are 0..1, oldest first; drawn bottom-aligned
/// inside a sunken well. One manual pass, because 26 widgets per meter × 3
/// meters × 60 fps is a lot of layout for a graph.
pub fn sparkline(src: std.builtin.SourceLocation, values: []const f32, color: Color, h: f32) void {
    var well = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = h },
        .background = true,
        .color_fill = C.spark_bg,
        .corner_radius = R.track,
        .padding = dvui.Rect.all(2),
    });
    defer well.deinit();
    if (values.len == 0) return;

    const r = well.data().contentRectScale().r;
    if (r.w <= 0 or r.h <= 0) return;
    const n: f32 = @floatFromInt(values.len);
    const gap: f32 = 1;
    const bw = @max(1.0, (r.w - (n - 1) * gap) / n);
    for (values, 0..) |v, i| {
        const frac = std.math.clamp(v, 0, 1);
        const bh = @max(1.0, frac * r.h);
        const bar: dvui.Rect.Physical = .{
            .x = r.x + @as(f32, @floatFromInt(i)) * (bw + gap),
            .y = r.y + r.h - bh,
            .w = bw,
            .h = bh,
        };
        bar.fill(.{}, .{ .color = color });
    }
}

/// One 86px telemetry meter: a label/value header over a sparkline well. The
/// number and the history ARE the meter — no clock speeds, no `x / y GB`, no
/// second progress bar. `values` are 0..1, oldest first.
pub fn historyMeter(src: std.builtin.SourceLocation, id: usize, label: []const u8, value: f32, values: []const f32, color: Color) void {
    var b = dvui.box(src, .{ .dir = .vertical }, .{
        .id_extra = id,
        .min_size_content = .{ .w = 86 },
        .max_size_content = .width(86),
        .gravity_y = 0.5,
        .margin = .{ .w = 10 },
    });
    defer b.deinit();

    {
        var hdr = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .h = 4 } });
        defer hdr.deinit();
        dvui.labelNoFmt(@src(), label, .{}, .{
            .font = F.mono_legend,
            .color_text = C.text_faint,
            .padding = .{},
            .gravity_y = 0.5,
        });
        var buf: [8]u8 = undefined;
        dvui.labelNoFmt(@src(), std.fmt.bufPrint(&buf, "{d:.0}%", .{value * 100}) catch "", .{}, .{
            .font = F.mono_val,
            .color_text = color,
            .padding = .{},
            .gravity_x = 1.0,
            .gravity_y = 0.5,
        });
    }
    sparkline(@src(), values, color, 22);
}

/// A 6×6 legend swatch.
pub fn swatch(src: std.builtin.SourceLocation, color: Color) void {
    var b = dvui.box(src, .{}, .{
        .min_size_content = .{ .w = 6, .h = 6 },
        .background = true,
        .color_fill = color,
        .corner_radius = dvui.Rect.all(1),
        .gravity_y = 0.5,
        .margin = .{ .w = 5 },
    });
    b.deinit();
}

// ------------------------------------------------------------------- tests

test {
    // The widget helpers below take dvui widgets and so are never called from a
    // test; without this the test build skips analyzing them and an API break
    // only surfaces when a view happens to use one.
    std.testing.refAllDecls(@This());
}

test "cap_ratio matches the bundled face" {
    // The whole type scale is nominal px divided through this number. If a font
    // update changed the face's cap height, every size in the app would shift
    // and nothing would fail — so read it back out of the file.
    const ttf = @embedFile("fonts/IBMPlexSans-Regular.ttf");
    const n = std.mem.readInt(u16, ttf[4..6], .big);
    var head: ?usize = null;
    var os2: ?usize = null;
    for (0..n) |i| {
        const rec = 12 + 16 * i;
        const tag = ttf[rec..][0..4];
        const off = std.mem.readInt(u32, ttf[rec + 8 ..][0..4], .big);
        if (std.mem.eql(u8, tag, "head")) head = off;
        if (std.mem.eql(u8, tag, "OS/2")) os2 = off;
    }
    const upem: f32 = @floatFromInt(std.mem.readInt(u16, ttf[head.? + 18 ..][0..2], .big));
    const cap: f32 = @floatFromInt(std.mem.readInt(i16, ttf[os2.? + 88 ..][0..2], .big));
    try std.testing.expectApproxEqAbs(cap_ratio, cap / upem, 0.001);
}

test "type roles convert nominal px to dvui cap-height sizes" {
    // 12.5px prose at 1.65 line spacing, as the design states it.
    try std.testing.expectApproxEqAbs(@as(f32, 12.5 * cap_ratio), F.prose.size, 0.001);
    // dvui multiplies its line-height factor by the hhea span, NOT the cap
    // height, so the factor carries that division; recovering 1.65 proves the
    // round trip through both conversions.
    try std.testing.expectApproxEqAbs(@as(f32, 1.65), F.prose.line_height_factor * natural_line, 0.001);
    // Every role must resolve to a family we actually register, or dvui
    // silently substitutes its built-in Vera and the screen goes generic.
    for ([_]dvui.Font{ F.app, F.ui, F.ui_sm, F.row, F.row_hi, F.prose, F.input, F.mono, F.mono_sm, F.mono_hd, F.mono_val, F.mono_row }) |f| {
        const fam = f.familyName();
        const known = std.mem.eql(u8, fam, fonts.sans) or std.mem.eql(u8, fam, fonts.sans_med) or
            std.mem.eql(u8, fam, fonts.mono) or std.mem.eql(u8, fam, fonts.mono_med);
        errdefer std.debug.print("role family {s} is not a registered family\n", .{fam});
        try std.testing.expect(known);
    }
}

test "accents keep their reserved meanings distinct" {
    // Amber and blue must not drift toward each other: the entire screen's
    // legibility rests on "warm = machine, cool = user" being obvious.
    try std.testing.expect(C.amber.r > C.amber.b);
    try std.testing.expect(C.blue.b > C.blue.r);
    // The diffusion side of the VRAM bar is warm, the LLM side cool, for the
    // same reason.
    try std.testing.expect(C.vram_dit.r > C.vram_dit.b);
    try std.testing.expect(C.vram_llm.b > C.vram_llm.r);
    try std.testing.expect(C.vram_ctx.b > C.vram_ctx.r);
}
