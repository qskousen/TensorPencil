//! The window chrome: the three vertical bands, the title bar and the left
//! sidebar. Everything here renders from plain data and reports back through
//! callbacks, so the same code draws the live app and the canned `ui-probe`
//! screenshot. No chat, diffusion or VRAM types reach this module.
//!
//! The band split is fixed-flex-fixed horizontally (sidebar | chat | rail) and
//! fixed-flex-fixed vertically (title | body | status). The status bar spans
//! the WHOLE window, under all three columns, because telemetry is a property
//! of the app and not of the queue.
//!
//! Heights are computed rather than left to the box layout: a scroll area
//! reports its full content height as its min size, so as a plain flex child it
//! would push the bars off-screen. `Bands` does that arithmetic once.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;
const F = style.F;
const R = style.R;
const L = style.Layout;

pub const Tab = enum { chat, studio };

/// Vertical band heights for one frame, from the root box's content rect.
pub const Bands = struct {
    body: f32,

    pub fn from(root: *dvui.BoxWidget) Bands {
        const h = root.data().contentRect().h;
        return .{ .body = @max(120, h - L.title_h - L.status_h) };
    }
};

// ---------------------------------------------------------------- title bar

pub const Title = struct {
    tab: Tab,
    /// The diffusion checkpoint, as a dropdown chip ("sdxl-turbo · fp16").
    diff_model: []const u8,
    /// What is resident right now, static ("qwen-7b + sdxl-turbo"). Empty
    /// hides the chip.
    residents: []const u8 = "",
};

pub const TitleActions = struct {
    on_tab: *const fn (Tab) void,
    on_model_menu: *const fn () void,
};

pub fn titleBar(m: Title, cb: TitleActions) void {
    var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = L.title_h },
        .max_size_content = .height(L.title_h),
        .background = true,
        .color_fill = C.chrome,
        .border = style.Edge.bottom,
        .color_border = style.hairline,
        .padding = .{ .x = 14, .w = 14 },
    });
    defer bar.deinit();

    // No traffic lights and no app name: the mockup was imitating a macOS
    // window, and the real one has a title bar of its own saying TensorPencil.
    // The bar starts with the only thing here that is a control.
    if (style.segmented(@src(), &.{ "Chat", "Studio" }, @intFromEnum(m.tab))) |i| {
        cb.on_tab(@enumFromInt(i));
    }

    // Right group. `gravity_x = 1` inside an expanded spacer is dvui's
    // margin-left:auto.
    var right = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
    });
    defer right.deinit();
    {
        var grp = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 1.0, .gravity_y = 0.5 });
        defer grp.deinit();
        if (style.chip(@src(), m.diff_model, .{ .dropdown = true, .fill = C.chip_hi })) cb.on_model_menu();
        if (m.residents.len > 0) {
            _ = style.chip(@src(), m.residents, .{ .fill = C.chip_hi, .id_extra = 1 });
        }
    }
}

// ------------------------------------------------------------------ sidebar

/// One conversation in the list. `sub` is only drawn on the selected row and is
/// blue, because it records what the USER did by hand (studio edits), not what
/// the machine did.
pub const ConvRow = struct {
    id: u64,
    title: []const u8,
    sub: ?[]const u8 = null,
};

/// A dated group of conversations (TODAY, YESTERDAY, EARLIER).
pub const ConvGroup = struct {
    head: []const u8,
    rows: []const ConvRow,
};

pub const Sidebar = struct {
    groups: []const ConvGroup,
    selected: ?u64 = null,
    /// Download progress for the Models row, 0..1, or null when idle. Amber:
    /// this is the machine working.
    models_pct: ?f32 = null,
};

pub const SidebarActions = struct {
    on_new_chat: *const fn () void,
    on_select: *const fn (u64) void,
    on_delete: *const fn (u64) void,
    on_models: *const fn () void,
    on_settings: *const fn () void,
};

pub fn sidebar(m: Sidebar, cb: SidebarActions) void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{
        // Expand into the band rather than being told a height: a fixed
        // min_size_content is the CONTENT box, so padding would push the rail
        // past the bottom of the body by exactly its own padding.
        .expand = .vertical,
        // Content width = the outer width the design gives, less this box's
        // own padding and its 1px right border.
        .min_size_content = .{ .w = L.sidebar_w - 2 * L.sidebar_pad - 1 },
        .max_size_content = .width(L.sidebar_w - 2 * L.sidebar_pad - 1),
        .background = true,
        .color_fill = C.rail,
        .border = style.Edge.right,
        .color_border = style.hairline_soft,
        .padding = .{ .x = L.sidebar_pad, .y = 14, .w = L.sidebar_pad, .h = 14 },
    });
    defer col.deinit();

    // Measured now, before the children are laid out: dvui hands a widget its
    // rect at init, so this is the real inner height of the rail.
    const inner = col.data().contentRect().h;

    if (newChatButton(@src())) cb.on_new_chat();

    // The list scrolls; the footer below is pinned. Cap the scroll area's
    // height to what is left, or its content min-height pushes the footer out.
    const new_chat_h: f32 = 34;
    const footer_h: f32 = 62;
    const list_h = @max(40, inner - new_chat_h - 16 - footer_h);
    {
        var sc = dvui.scrollArea(@src(), .{ .horizontal = .none }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = list_h },
            .max_size_content = .height(list_h),
            .background = false,
            .margin = .{ .y = 16 },
        });
        defer sc.deinit();

        for (m.groups, 0..) |g, gi| {
            if (g.rows.len == 0) continue;
            style.sectionHead(@src(), g.head, .{
                .id_extra = gi,
                .padding = .{ .x = 8, .y = if (gi == 0) 0 else 10, .h = 7 },
            });
            for (g.rows, 0..) |r, ri| {
                convRow(@src(), gi * 1024 + ri, r, m.selected == r.id, cb);
            }
        }
        if (m.groups.len == 0) {
            dvui.labelNoFmt(@src(), "No conversations yet.", .{}, .{
                .font = F.row,
                .color_text = C.text_ghost,
                .padding = .{ .x = 8, .y = 6 },
            });
        }
    }

    footer(@src(), m.models_pct, cb);
}

fn newChatButton(src: std.builtin.SourceLocation) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = C.raised,
        .color_fill_hover = C.raised.lighten(6),
        .color_fill_press = C.raised.lighten(12),
        .border = style.Edge.all,
        .color_border = style.hairline_hi,
        .corner_radius = R.panel,
        .padding = dvui.Rect.all(9),
        .margin = .{},
    });
    bw.processEvents();
    bw.drawBackground();
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
        defer row.deinit();
        style.mark(@src(), .plus, 11, C.text_hi, .{ .margin = .{ .w = 5 } });
        dvui.labelNoFmt(@src(), "New chat", .{}, .{
            .font = F.row_hi,
            .color_text = C.text_hi,
            .padding = .{},
            .gravity_y = 0.5,
        });
    }
    const clicked = bw.clicked();
    bw.drawFocus();
    bw.deinit();
    return clicked;
}

/// Which row is armed for deletion, by conversation id. One click on the ✕ arms
/// it, a second confirms.
///
/// Two steps rather than a dialog: the same shape the VRAM meter's unload button
/// already uses, and it keeps a destructive action from being one stray click on
/// a row you were only trying to open.
var arm_delete: ?u64 = null;

fn convRow(src: std.builtin.SourceLocation, id_extra: usize, r: ConvRow, selected: bool, cb: SidebarActions) void {
    // A plain box, NOT a ButtonWidget: a button processes its events before its
    // children exist, so it would swallow the click meant for the ✕ inside it.
    // Here the row's own click is tested AFTER the children have had their turn.
    //
    // The cost is that hover has to come from last frame (the background is
    // painted at init, before we can know), which is invisible at frame rate.
    const row_id = dvui.parentGet().extendId(src, id_extra);
    const hovered = dvui.dataGet(null, row_id, "_hover", bool) orelse false;
    const armed = arm_delete == r.id;

    var row = dvui.box(src, .{ .dir = .horizontal }, .{
        .id_extra = id_extra,
        .expand = .horizontal,
        .background = selected or hovered,
        .color_fill = if (selected) C.sel_row else style.over(C.rail, C.text_hi, 0.04),
        .corner_radius = R.button,
        .padding = dvui.Rect.all(8),
        .margin = .{ .h = 3 },
    });

    {
        var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
        defer col.deinit();
        var tbuf: [160]u8 = undefined;
        // Leave room for the ✕ so the title does not run under it.
        const avail = col.data().contentRect().w - (if (hovered or armed) @as(f32, 20) else 0);
        fonts.richLine(@src(), style.ellipsize(&tbuf, r.title, F.row, avail), .{
            .font = F.row,
            .color_text = if (selected) C.text_hi else C.text_dim,
            .padding = .{},
            .margin = .{},
        });
        if (selected) if (r.sub) |sub| dvui.labelNoFmt(@src(), sub, .{}, .{
            .font = F.mono_row,
            .color_text = C.blue,
            .padding = .{ .y = 2 },
        });
    }

    // The ✕ appears on hover, and stays while armed so the second click has
    // something to hit even if the pointer drifts.
    var deleted = false;
    if (hovered or armed) {
        var x: dvui.ButtonWidget = undefined;
        x.init(@src(), .{}, .{
            .gravity_x = 1.0,
            .gravity_y = 0.5,
            .background = armed,
            .color_fill = style.tint(C.danger, 60),
            .color_fill_hover = style.tint(C.danger, 90),
            .color_fill_press = style.tint(C.danger, 120),
            .corner_radius = R.chip,
            .padding = dvui.Rect.all(3),
            .margin = .{},
        });
        x.processEvents();
        x.drawBackground();
        style.mark(@src(), .close, 11, if (armed) C.danger else C.text_ghost, .{});
        const hit = x.clicked();
        x.deinit();
        if (hit) {
            if (armed) {
                deleted = true;
                arm_delete = null;
            } else arm_delete = r.id;
        }
    }

    var now_hovered = false;
    const clicked = dvui.clicked(row.data(), .{ .hovered = &now_hovered });
    const row_rect = row.data().borderRectScale().r;
    dvui.dataSet(null, row.data().id, "_hover", now_hovered);
    row.deinit();

    // Right-click still works, and is the only way to reach delete without a
    // pointer hovering (touch, keyboard-driven use).
    {
        const ctx = dvui.context(@src(), .{ .rect = row_rect }, .{ .id_extra = id_extra });
        defer ctx.deinit();
        if (ctx.activePoint()) |cp| {
            var fw = dvui.floatingMenu(@src(), .{ .from = dvui.Rect.Natural.fromPoint(cp) }, .{});
            defer fw.deinit();
            if (dvui.menuItemLabel(@src(), "Delete conversation", .{}, .{ .expand = .horizontal }) != null) {
                fw.close();
                deleted = true;
            }
        }
    }

    if (deleted) {
        cb.on_delete(r.id);
        return;
    }
    // Clicking anywhere else disarms, so an armed row cannot linger.
    if (clicked) {
        if (armed) arm_delete = null else cb.on_select(r.id);
    }
}

fn footer(src: std.builtin.SourceLocation, models_pct: ?f32, cb: SidebarActions) void {
    var f = dvui.box(src, .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .gravity_y = 1.0,
        // The sidebar's surface; see queue_rail's tab strip for why.
        .background = true,
        .color_fill = C.rail,
        .border = style.Edge.top,
        .color_border = style.hairline_soft,
        .padding = .{ .y = 10 },
    });
    defer f.deinit();

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal });
        defer row.deinit();
        if (footerLabel(@src(), "Models")) cb.on_models();
        if (models_pct) |p| {
            var buf: [8]u8 = undefined;
            dvui.labelNoFmt(@src(), std.fmt.bufPrint(&buf, "{d:.0}%", .{p * 100}) catch "", .{}, .{
                .font = F.mono,
                .color_text = C.amber, // a download is the machine working
                .gravity_x = 1.0,
                .gravity_y = 0.5,
                .padding = .{},
            });
        }
    }
    if (footerLabel(@src(), "Settings")) cb.on_settings();
}

fn footerLabel(src: std.builtin.SourceLocation, text: []const u8) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .background = false,
        .color_fill_hover = style.hover_wash,
        .color_fill_press = style.hover_wash,
        .corner_radius = R.button,
        .padding = .{ .x = 8, .y = 6, .w = 8, .h = 6 },
        .margin = .{},
    });
    bw.processEvents();
    bw.drawBackground();
    dvui.labelNoFmt(@src(), text, .{}, .{
        .font = F.row,
        .color_text = C.text_dim,
        .padding = .{},
        .gravity_y = 0.5,
    });
    const clicked = bw.clicked();
    bw.deinit();
    return clicked;
}
