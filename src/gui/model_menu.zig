//! The title-bar model chips: a chip naming the current model that opens a menu
//! of architecture groups, each a submenu of files. Pure rendering over plain
//! data, so `ui-probe` can draw it and the app builds the data from the catalog
//! (see model_lib.zig).
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");

const C = style.C;

pub const Item = struct {
    label: []const u8,
    /// What a pick reports; the file path.
    path: []const u8,
    selected: bool = false,
    /// Shown dimmed and inert, with `note` after the name saying why.
    greyed: bool = false,
    note: []const u8 = "",
};

pub const Group = struct {
    label: []const u8,
    items: []const Item,
    /// Every item is greyed: the whole group reads dim.
    greyed: bool = false,
};

pub const Menu = struct {
    groups: []const Group = &.{},
    /// The "none" entry ("no chat model"). Empty hides it.
    none_label: []const u8 = "",
    none_selected: bool = false,
    /// A line at the bottom when there is nothing to pick from, or a hint.
    empty_note: []const u8 = "",
};

pub const Pick = union(enum) {
    none,
    path: []const u8,
    /// The "Model folders…" footer: open Settings.
    settings,
};

pub const Chip = struct {
    label: []const u8,
    /// Loaded right now: a green dot before the name.
    resident: bool = false,
    /// Something is missing for this model to run: amber text.
    warn: bool = false,
    /// Nothing configured: ghost text.
    empty: bool = false,
};

/// Draw the chip and, while open, its menu. Returns what the user picked this
/// frame, if anything.
///
/// The chip is a submenu item inside a one-item menu, the way `dvui.dropdown`
/// is built, so the menu machinery owns open and close: a click opens, a pick
/// or a click elsewhere closes, and focus moving into a nested submenu is not a
/// close. Tracking "open" by hand with the focused subwindow closed the whole
/// menu the moment the pointer entered a group's file list.
pub fn chip(src: std.builtin.SourceLocation, c: Chip, menu: Menu, id_extra: usize) ?Pick {
    var pick: ?Pick = null;
    var m = dvui.menu(src, .horizontal, .{
        .id_extra = id_extra,
        .background = false,
        .border = .{},
        .padding = .{},
        .margin = .{ .x = 8 },
        .gravity_y = 0.5,
    });
    defer m.deinit();

    const text_color = if (c.warn) C.amber else if (c.empty) C.text_ghost else C.text_dim;
    var mi = dvui.menuItem(@src(), .{ .submenu = true, .focus_as_outline = true }, .{
        .background = true,
        .color_fill = C.chip_hi,
        .color_fill_hover = C.chip_hi.lighten(5),
        .border = style.Edge.all,
        .color_border = style.hairline,
        .corner_radius = style.R.button,
        .padding = .{ .x = 9, .y = 6, .w = 6, .h = 6 },
        .gravity_y = 0.5,
    });
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_y = 0.5 });
        defer row.deinit();
        if (c.resident) {
            var dot = dvui.box(@src(), .{}, .{
                .min_size_content = .{ .w = 6, .h = 6 },
                .max_size_content = .{ .w = 6, .h = 6 },
                .background = true,
                .color_fill = C.meter_gpu,
                .corner_radius = dvui.Rect.all(3),
                .margin = .{ .w = 6 },
                .gravity_y = 0.5,
            });
            dot.deinit();
        }
        dvui.labelNoFmt(@src(), c.label, .{}, .{
            .font = style.F.mono,
            .color_text = text_color,
            .padding = .{},
            .gravity_y = 0.5,
        });
        style.mark(@src(), .caret_down, 11, text_color, .{ .margin = .{ .x = 4 } });
    }
    const active = mi.activeRect();
    mi.deinit();

    if (active) |r| {
        var fw = dvui.floatingMenu(@src(), .{ .from = r, .avoid = .vertical }, .{});
        defer fw.deinit();
        // What `dvui.dropdown` does too: without it a hovered row highlights but
        // is never focused, and a release only activates a focused row.
        fw.menu.submenus_activated = true;
        if (menuBody(menu)) |p| {
            pick = p;
            fw.close();
        }
    }
    return pick;
}

/// The menu's rows. Groups are submenus; a group of one item is still a submenu
/// so the shape stays the same as the list changes.
fn menuBody(menu: Menu) ?Pick {
    var pick: ?Pick = null;
    if (menu.none_label.len > 0) {
        if (pickRow(@src(), menu.none_label, menu.none_selected, false, "", 0)) pick = .none;
    }
    for (menu.groups, 0..) |g, gi| {
        const opts: dvui.Options = .{
            .id_extra = gi,
            .expand = .horizontal,
            .font = style.F.ui,
            .color_text = if (g.greyed) C.text_ghost else C.text,
            .corner_radius = style.R.chip,
        };
        if (dvui.menuItemLabel(@src(), g.label, .{ .submenu = true }, opts)) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r, .avoid = .horizontal }, .{});
            defer fw.deinit();
            fw.menu.submenus_activated = true;
            for (g.items, 0..) |it, ii| {
                if (pickRow(@src(), it.label, it.selected, it.greyed, it.note, ii)) pick = .{ .path = it.path };
            }
        }
    }
    if (menu.groups.len == 0 and menu.empty_note.len > 0) {
        dvui.labelNoFmt(@src(), menu.empty_note, .{}, .{
            .font = style.F.ui_sm,
            .color_text = C.text_ghost,
            .padding = .{ .x = 8, .y = 4, .w = 8, .h = 4 },
        });
    }
    if (pickRow(@src(), "Model folders…", false, false, "", 0)) pick = .settings;
    return pick;
}

/// One pickable row: the name, blue when it is the current choice, ghosted with
/// its note when greyed. A greyed row draws but never picks.
///
/// One label straight inside the menu item, as `dvui.menuItemLabel` does. A row
/// built from a box with a mark and two labels looked the same but never
/// activated inside a nested submenu: the press closed the menu chain before the
/// release arrived. Measured by swapping the two on the same menu.
fn pickRow(src: std.builtin.SourceLocation, label: []const u8, selected: bool, greyed: bool, note: []const u8, id_extra: usize) bool {
    var mi = dvui.menuItem(src, .{}, .{
        .id_extra = id_extra,
        .expand = .horizontal,
        .corner_radius = style.R.chip,
    });
    defer mi.deinit();
    var buf: [512]u8 = undefined;
    const text = if (greyed and note.len > 0)
        std.fmt.bufPrint(&buf, "{s}   {s}", .{ label, note }) catch label
    else
        label;
    const color = if (greyed) C.text_ghost else if (selected) C.blue else C.text;
    dvui.labelNoFmt(@src(), text, .{}, mi.style().strip().override(.{
        .label = .{ .for_id = mi.data().id },
        .font = style.F.ui,
        .color_text = color,
    }));
    return !greyed and mi.activeRect() != null;
}
