//! Hover hints for buttons whose meaning isn't obvious at a glance —
//! icon-only buttons especially. One-line wrapper over `dvui.tooltip` so a
//! call site is a single line: create the widget with `.data_out = &wd`,
//! then `hint.hover(@src(), &wd, "…")` right after it (same parent).
const std = @import("std");
const dvui = @import("dvui");

pub fn hover(src: std.builtin.SourceLocation, wd: *const dvui.WidgetData, text: []const u8) void {
    // Inherit the target widget's id_extra so hints on widgets created from a
    // shared source location (helper fns, loops) get distinct ids too.
    dvui.tooltip(src, .{ .active_rect = wd.borderRectScale().r }, "{s}", .{text}, .{
        .id_extra = wd.options.idExtra(),
    });
}
