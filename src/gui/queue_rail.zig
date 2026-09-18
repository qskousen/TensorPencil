//! The right-hand queue rail: what the image engine is doing now, what is
//! waiting, and what it just finished. Renders from plain data so the same code
//! draws the live queue and the canned `ui-probe` screenshot.
//!
//! No telemetry here — that all lives in the status bar, which spans the whole
//! window. The rail answers "what is my queue doing", nothing else.
//!
//! The Queue tab is WAITING WORK ONLY and drains to empty: a render nobody has
//! started, plus the failures nothing else keeps. A render in motion is drawn
//! where its pixels are -- the studio canvas, or the transcript's tool card --
//! so it is never on screen twice. Finished images belong to the transcript
//! that asked for them; everything ever made lives under Library, which is a
//! different activity (finding an old image) rather than a second copy of the
//! current conversation.
//!
//! A row and a Library thumbnail mean different things when clicked, which is
//! why they are separate callbacks: a row is work to watch, a thumbnail is a
//! picture to look at.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;
const F = style.F;
const R = style.R;
const L = style.Layout;

/// What to draw in a Library tile.
pub const Thumb = union(enum) {
    /// Still being read back from disk: the same two greys as a real
    /// placeholder tile, so nothing pops when the texture swaps in.
    loading,
    rgba: struct { px: []const u8, w: u32, h: u32 },
};

pub const JobState = union(enum) {
    queued: struct {
        eta_s: ?f32 = null,
        /// Why it is still waiting, when the reason is not "a host is busy".
        note: []const u8 = "",
    },
    /// It will not be rendered. The row stays, because a job that vanishes
    /// leaves nothing to read and nothing to click.
    failed: struct { why: []const u8 },
};

pub const Job = struct {
    id: u64,
    title: []const u8,
    state: JobState,
    /// Provenance. A job the user set up by hand in Studio is worth
    /// distinguishing from one the chat model asked for.
    from_studio: bool = false,
    /// The engine host rendering it, "" when there is only one.
    host: []const u8 = "",
    /// Only a job still in this client's own queue can be reordered; one a host
    /// has taken is where it is.
    draggable: bool = true,
};

/// One finished image in the Library grid.
pub const LibraryItem = struct {
    id: u64,
    thumb: Thumb = .loading,
    /// The host that made it, "" when there is only one.
    host: []const u8 = "",
};

pub const Tab = enum { queue, library };

pub const Model = struct {
    tab: Tab = .queue,
    jobs: []const Job = &.{},
    /// Every finished image, newest first. Only the Library tab draws these.
    library: []const LibraryItem = &.{},
    paused: bool = false,
};

pub const Actions = struct {
    on_tab: *const fn (Tab) void,
    on_pause_all: *const fn () void,
    /// A library thumbnail: "look at that picture". Always the viewer, in both
    /// views -- the canvas is for work in motion and may be busy with some.
    on_open_library: *const fn (u64) void,
    on_cancel: *const fn (u64) void,
    /// Have another go at a failed job. The row is where a failure lives, in
    /// both views, so this is the only place it can be asked for.
    on_retry: *const fn (u64) void,
    /// Move the job with the first id so it sits before the job with the
    /// second, or at the end when that is null.
    on_reorder: *const fn (u64, ?u64) void,
};

pub fn render(m: Model, cb: Actions) void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .vertical,
        // Outer width less this box's 1px left border; its padding is on the
        // inner body, not here.
        .min_size_content = .{ .w = L.rail_w - 1 },
        .max_size_content = .width(L.rail_w - 1),
        .background = true,
        .color_fill = C.rail,
        .border = style.Edge.left,
        .color_border = style.hairline_soft,
    });
    defer col.deinit();

    const inner = col.data().contentRect().h;
    const strip_h: f32 = 40;
    tabStrip(@src(), m, cb);

    // Clipped and scrollable: a long queue must not push the rail's own
    // background off, and dvui boxes do not clip.
    var sc = dvui.scrollArea(@src(), .{ .horizontal = .none }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = @max(40, inner - strip_h) },
        .max_size_content = .height(@max(40, inner - strip_h)),
        .background = false,
    });
    defer sc.deinit();

    var body = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .padding = dvui.Rect.all(12),
    });
    defer body.deinit();

    switch (m.tab) {
        .queue => {
            if (m.jobs.len == 0) {
                dvui.labelNoFmt(@src(), "Nothing queued.", .{}, .{
                    .font = F.row,
                    .color_text = C.text_ghost,
                    .padding = .{ .x = 2, .y = 6 },
                });
            } else {
                jobList(@src(), m, cb);
                hintRow(@src(), m, cb);
            }
        },
        .library => {
            if (m.library.len == 0) {
                dvui.labelNoFmt(@src(), "No images yet.", .{}, .{
                    .font = F.row,
                    .color_text = C.text_ghost,
                    .padding = .{ .x = 2, .y = 6 },
                });
            } else thumbGrid(@src(), m.library, 3, cb);
        },
    }
}

fn tabStrip(src: std.builtin.SourceLocation, m: Model, cb: Actions) void {
    var strip = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        // The rail's surface, not the theme's: a forced background here painted
        // a canvas-coloured block over the rail behind the tabs.
        .background = true,
        .color_fill = C.rail,
        .border = style.Edge.bottom,
        .color_border = style.hairline_soft,
        .padding = .{ .x = 12, .y = 10, .w = 12, .h = 10 },
    });
    defer strip.deinit();

    var buf: [24]u8 = undefined;
    const q_label = if (m.jobs.len > 0)
        std.fmt.bufPrint(&buf, "Queue · {d}", .{m.jobs.len}) catch "Queue"
    else
        "Queue";
    if (tab(@src(), 0, q_label, m.tab == .queue)) cb.on_tab(.queue);
    if (tab(@src(), 1, "Library", m.tab == .library)) cb.on_tab(.library);
}

fn tab(src: std.builtin.SourceLocation, id: usize, label: []const u8, on: bool) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .id_extra = id,
        .expand = .horizontal,
        .background = on,
        .color_fill = C.text_hi,
        .color_fill_hover = if (on) C.text_hi else style.hover_wash,
        .color_fill_press = if (on) C.text_hi else style.hover_wash,
        .corner_radius = R.chip,
        .padding = dvui.Rect.all(7),
        .margin = .{ .x = if (id == 0) 0 else 2 },
    });
    bw.processEvents();
    bw.drawBackground();
    dvui.labelNoFmt(@src(), label, .{}, .{
        .font = F.ui,
        .color_text = if (on) C.text_ink else C.text_dim,
        .padding = .{},
        .gravity_x = 0.5,
        .gravity_y = 0.5,
    });
    const clicked = bw.clicked();
    bw.drawFocus();
    bw.deinit();
    return clicked;
}

fn jobList(src: std.builtin.SourceLocation, m: Model, cb: Actions) void {
    if (m.jobs.len == 0) return;

    // The reorder widget must wrap the whole list; it reports the drag as a
    // (removed, insert-before) pair once, on drop.
    var reo = dvui.reorder(src, .{}, .{ .expand = .horizontal });
    defer reo.deinit();

    var removed: ?usize = null;
    var insert_before: ?usize = null;

    {
        var list = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
        defer list.deinit();

        for (m.jobs, 0..) |j, i| {
            var r = reo.reorderable(@src(), .{}, .{ .id_extra = i, .expand = .horizontal });
            defer r.deinit();
            if (r.removed()) {
                removed = i;
            } else if (r.insertBefore()) {
                insert_before = i;
            }
            jobRow(@src(), j, r, cb);
        }
    }
    if (reo.finalSlot()) insert_before = m.jobs.len;

    if (insert_before) |to| if (removed) |from|
        cb.on_reorder(m.jobs[from].id, if (to < m.jobs.len) m.jobs[to].id else null);
}

fn jobRow(src: std.builtin.SourceLocation, j: Job, r: *dvui.Reorderable, cb: Actions) void {
    const failed = j.state == .failed;
    var row = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = C.queue_idle,
        .border = style.Edge.all,
        // A failed job is the only thing in this rail allowed to be red.
        .color_border = if (failed) style.tint(C.danger, 56) else style.hairline_soft,
        .corner_radius = R.panel,
        .padding = dvui.Rect.all(9),
        .margin = .{ .h = 9 },
    });
    defer row.deinit();

    // No picture slot: nothing waiting has pixels, and the one it would get on
    // starting is drawn where the render is watched, never here.
    //
    // Widths are computed, not left to the box layout: a child that expands
    // takes the whole row and leaves the controls beside it nothing.
    const ctl_w: f32 = 18;
    const text_w = @max(40, row.data().contentRect().w - ctl_w - 10);

    var text = dvui.box(@src(), .{ .dir = .vertical }, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = text_w },
        .max_size_content = .width(text_w),
    });

    var tbuf: [160]u8 = undefined;
    fonts.richLine(@src(), style.ellipsize(&tbuf, j.title, F.row_hi, text_w), .{
        .font = F.row_hi,
        .color_text = C.text,
        .padding = .{},
        .margin = .{},
    });

    var buf: [96]u8 = undefined;
    var host_buf: [48]u8 = undefined;
    const on_host = if (j.host.len > 0) std.fmt.bufPrint(&host_buf, " · on {s}", .{j.host}) catch "" else "";
    switch (j.state) {
        .queued => |s| {
            const line = if (s.note.len > 0)
                std.fmt.bufPrint(&buf, "queued · {s}", .{s.note}) catch "queued"
            else if (s.eta_s) |e|
                std.fmt.bufPrint(&buf, "queued · ~{d:.0} s{s}{s}", .{ e, if (j.from_studio) " · from Studio" else "", on_host }) catch "queued"
            else
                std.fmt.bufPrint(&buf, "queued{s}{s}", .{ if (j.from_studio) " · from Studio" else "", on_host }) catch "queued";
            fonts.richLine(@src(), style.ellipsize(&tbuf, line, F.mono_row, text_w), .{
                .font = F.mono_row,
                .color_text = if (s.note.len > 0) C.amber else C.text_ghost,
                .padding = .{ .y = 2 },
                .margin = .{},
            });
        },
        .failed => |s| {
            const line = std.fmt.bufPrint(&buf, "{s}{s}", .{ s.why, on_host }) catch "failed";
            fonts.richLine(@src(), style.ellipsize(&tbuf, line, F.mono_row, text_w), .{
                .font = F.mono_row,
                .color_text = C.danger,
                .padding = .{ .y = 2 },
                .margin = .{},
            });
            // In its own row: a chip centres itself vertically, which in this
            // column lands it on top of the line above.
            var act = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .y = 4 } });
            defer act.deinit();
            if (style.chip(@src(), "Try again", .{
                .font = F.mono_row,
                .border = style.tint(C.danger, 90),
                .text = C.text,
            })) cb.on_retry(j.id);
        },
    }

    text.deinit();

    var ctl = dvui.box(@src(), .{ .dir = .vertical }, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = ctl_w },
        .max_size_content = .width(ctl_w),
    });
    defer ctl.deinit();

    // Stop this one render, or clear away a row that is only a record of what
    // happened. Dim like the drag handle: always there, never what the eye
    // lands on.
    if (dvui.buttonIcon(@src(), "cancel", dvui.entypo.cross, .{}, .{}, .{
        .gravity_x = 0.5,
        .min_size_content = .{ .w = 12, .h = 12 },
        .color_text = if (j.state == .failed) style.tint(C.danger, 150) else C.text_ghost,
        .background = false,
        .corner_radius = R.chip,
        .padding = dvui.Rect.all(2),
        .margin = .{},
    })) cb.on_cancel(j.id);

    // Drag handle. The mockup shows none, but an invisible drag target is a
    // feature nobody finds; this is the dimmest thing in the row.
    if (j.draggable) _ = dvui.ReorderWidget.draggable(@src(), .{ .reorderable = r }, .{
        .gravity_x = 0.5,
        .min_size_content = .{ .w = 12, .h = 12 },
        .margin = .{ .y = 4 },
        .color_text = C.text_ghost,
    });
}

fn hintRow(src: std.builtin.SourceLocation, m: Model, cb: Actions) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 2 } });
    defer row.deinit();
    var any_drag = false;
    for (m.jobs) |j| {
        if (j.draggable) {
            any_drag = true;
            break;
        }
    }
    if (any_drag) dvui.labelNoFmt(@src(), "drag to reorder", .{}, .{
        .font = F.mono,
        .color_text = C.text_ghost,
        .padding = .{ .x = 2 },
        .gravity_y = 0.5,
    });
    var bw: dvui.ButtonWidget = undefined;
    bw.init(@src(), .{}, .{
        .gravity_x = 1.0,
        .background = false,
        .color_fill_hover = style.hover_wash,
        .color_fill_press = style.hover_wash,
        .corner_radius = R.chip,
        .padding = .{ .x = 5, .y = 3, .w = 5, .h = 3 },
        .margin = .{},
    });
    bw.processEvents();
    bw.drawBackground();
    dvui.labelNoFmt(@src(), if (m.paused) "Resume all" else "Pause all", .{}, .{
        .font = F.mono,
        .color_text = if (m.paused) C.amber else C.text_ghost,
        .padding = .{},
        .gravity_y = 0.5,
    });
    const clicked = bw.clicked();
    bw.deinit();
    if (clicked) cb.on_pause_all();
}

fn thumbGrid(src: std.builtin.SourceLocation, items: []const LibraryItem, cols: usize, cb: Actions) void {
    var grid = dvui.box(src, .{ .dir = .vertical }, .{ .expand = .horizontal });
    defer grid.deinit();

    const gap: f32 = 7;
    const avail = grid.data().contentRect().w;
    const cell = @max(24, (avail - gap * @as(f32, @floatFromInt(cols - 1))) / @as(f32, @floatFromInt(cols)));

    var i: usize = 0;
    while (i < items.len) : (i += cols) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .id_extra = i,
            .expand = .horizontal,
            .margin = .{ .h = gap },
        });
        defer row.deinit();
        var c: usize = 0;
        while (c < cols and i + c < items.len) : (c += 1) {
            // Overlay, not a box: the host tag sits ON the picture rather than
            // after it.
            var cellbox = dvui.overlay(@src(), .{
                .id_extra = c,
                .min_size_content = .{ .w = cell, .h = cell },
                .max_size_content = .size(.{ .w = cell, .h = cell }),
                .margin = .{ .w = if (c + 1 < cols) gap else 0 },
                .corner_radius = R.button,
            });
            const crs = cellbox.data().contentRectScale();
            switch (items[i + c].thumb) {
                .rgba => |px| _ = dvui.image(@src(), .{
                    .source = .{ .pixels = .{ .rgba = px.px, .width = px.w, .height = px.h } },
                    .shrink = .ratio,
                }, .{
                    .min_size_content = .{ .w = cell, .h = cell },
                    .max_size_content = .size(.{ .w = cell, .h = cell }),
                    .corner_radius = R.button,
                }),
                .loading => style.hatch(crs.r, C.raised, dvui.Color.fromHex("#161a1e"), 6 * @max(1, crs.s)),
            }
            if (items[i + c].host.len > 0) hostTag(@src(), items[i + c].host, cell);
            const clicked = dvui.clicked(cellbox.data(), .{});
            cellbox.deinit();
            if (clicked) cb.on_open_library(items[i + c].id);
        }
    }
}

/// Who made this picture, in its bottom-left corner. The wash under it is what
/// keeps the name readable over a bright image.
fn hostTag(src: std.builtin.SourceLocation, host: []const u8, cell: f32) void {
    var pill = dvui.box(src, .{ .dir = .horizontal }, .{
        .gravity_x = 0,
        .gravity_y = 1.0,
        .background = true,
        .color_fill = style.tint(C.canvas, 200),
        .corner_radius = R.chip,
        .padding = .{ .x = 4, .y = 1, .w = 4, .h = 1 },
        .margin = dvui.Rect.all(3),
    });
    defer pill.deinit();

    var buf: [64]u8 = undefined;
    fonts.richLine(@src(), style.ellipsize(&buf, host, F.mono_row, @max(8, cell - 14)), .{
        .font = F.mono_row,
        .color_text = C.text_dim,
        .padding = .{},
        .margin = .{},
    });
}
