//! The right-hand queue rail: what the image engine is doing now, what is
//! waiting, and what it just finished. Renders from plain data so the same code
//! draws the live queue and the canned `ui-probe` screenshot.
//!
//! No telemetry here — that all lives in the status bar, which spans the whole
//! window. The rail answers "what is my queue doing", nothing else.
//!
//! The Queue tab shows IN-FLIGHT WORK ONLY and drains to empty. Finished images
//! belong to the transcript that asked for them; showing them here as well
//! would put the same picture on screen twice. Everything ever made lives under
//! Library, which is a different activity (finding an old image) rather than a
//! second copy of the current conversation.
//!
//! The running row is the ONLY amber thing in the rail. If a second element
//! ever goes amber, the eye stops finding the one that is actually working.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;
const F = style.F;
const R = style.R;
const L = style.Layout;

/// What to draw in a job's 52px slot.
pub const Thumb = union(enum) {
    /// Nothing decoded yet: a dashed outline, not a grey box, so a queued job
    /// does not read as a finished one that came out black.
    dashed,
    /// Decoding / loading: the same two greys as a real placeholder tile, so
    /// nothing pops when the texture swaps in.
    loading,
    rgba: struct { px: []const u8, w: u32, h: u32 },
};

pub const JobState = union(enum) {
    running: struct { step: u32, steps: u32, it_s: f32 },
    queued: struct { eta_s: ?f32 = null },
};

pub const Job = struct {
    id: u64,
    title: []const u8,
    thumb: Thumb = .dashed,
    state: JobState,
    /// Provenance. A job the user set up by hand in Studio is worth
    /// distinguishing from one the chat model asked for.
    from_studio: bool = false,
};

/// One finished image in the Library grid.
pub const LibraryItem = struct {
    id: u64,
    thumb: Thumb = .loading,
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
    on_open: *const fn (u64) void,
    on_cancel: *const fn (u64) void,
    /// Move the job at `from` to sit before index `to`, both indices into
    /// `Model.jobs`.
    on_reorder: *const fn (usize, usize) void,
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
            jobRow(@src(), j, r);
        }
    }
    if (reo.finalSlot()) insert_before = m.jobs.len;

    if (insert_before) |to| if (removed) |from| cb.on_reorder(from, to);
}

fn jobRow(src: std.builtin.SourceLocation, j: Job, r: *dvui.Reorderable) void {
    const running = j.state == .running;
    var row = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = if (running) C.queue_active else C.queue_idle,
        .border = style.Edge.all,
        // The running job is the one thing in this rail allowed to be amber.
        .color_border = if (running) style.tint(C.amber, 56) else style.hairline_soft,
        .corner_radius = R.panel,
        .padding = dvui.Rect.all(9),
        .margin = .{ .h = 9 },
    });
    defer row.deinit();

    thumb(@src(), j.thumb, 52, running);

    var text = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        .margin = .{ .x = 10 },
    });
    defer text.deinit();

    var tbuf: [160]u8 = undefined;
    fonts.richLine(@src(), style.ellipsize(&tbuf, j.title, F.row_hi, text.data().contentRect().w), .{
        .font = F.row_hi,
        .color_text = if (running) C.text_hi else C.text,
        .padding = .{},
        .margin = .{},
    });

    var buf: [64]u8 = undefined;
    switch (j.state) {
        .running => |s| {
            const line = std.fmt.bufPrint(&buf, "{d} / {d} · {d:.1} it/s", .{ s.step, s.steps, s.it_s }) catch "";
            dvui.labelNoFmt(@src(), line, .{}, .{
                .font = F.mono_row,
                .color_text = C.text_dim,
                .padding = .{ .y = 2 },
            });
            const frac: f32 = if (s.steps > 0)
                @as(f32, @floatFromInt(s.step)) / @as(f32, @floatFromInt(s.steps))
            else
                0;
            progressTrack(@src(), frac);
            dvui.refresh(null, @src(), null); // it/s and the bar both move
        },
        .queued => |s| {
            const line = if (s.eta_s) |e|
                std.fmt.bufPrint(&buf, "queued · ~{d:.0} s{s}", .{ e, if (j.from_studio) " · from Studio" else "" }) catch "queued"
            else if (j.from_studio)
                "queued · from Studio"
            else
                "queued";
            dvui.labelNoFmt(@src(), line, .{}, .{
                .font = F.mono_row,
                .color_text = C.text_ghost,
                .padding = .{ .y = 2 },
            });
        },
    }

    // Drag handle. The mockup shows none, but an invisible drag target is a
    // feature nobody finds; this is the dimmest thing in the row.
    _ = dvui.ReorderWidget.draggable(@src(), .{ .reorderable = r }, .{
        .expand = .vertical,
        .gravity_x = 1.0,
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 14, .h = 14 },
        .color_text = C.text_ghost,
    });
}

/// A 3px progress track. Amber, always: this is the machine working.
fn progressTrack(src: std.builtin.SourceLocation, frac: f32) void {
    var t = dvui.box(src, .{}, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = 3 },
        .max_size_content = .height(3),
        .background = true,
        .color_fill = C.meter_track,
        .corner_radius = dvui.Rect.all(1.5),
        .margin = .{ .y = 4 },
    });
    defer t.deinit();
    const r = t.data().contentRectScale().r;
    if (r.w <= 0) return;
    const fill: dvui.Rect.Physical = .{
        .x = r.x,
        .y = r.y,
        .w = r.w * std.math.clamp(frac, 0, 1),
        .h = r.h,
    };
    fill.fill(.{}, .{ .color = C.amber });
}

fn thumb(src: std.builtin.SourceLocation, t: Thumb, size: f32, running: bool) void {
    var b = dvui.box(src, .{}, .{
        .min_size_content = .{ .w = size, .h = size },
        .max_size_content = .size(.{ .w = size, .h = size }),
        .gravity_y = 0.5,
        .corner_radius = R.chip,
    });
    defer b.deinit();
    const rs = b.data().contentRectScale();
    switch (t) {
        .dashed => dashedBox(rs.r, rs.s, if (running) style.tint(C.amber, 56) else style.hairline_hi),
        .loading => style.hatch(rs.r, C.raised, dvui.Color.fromHex("#161a1e"), 6 * @max(1, rs.s)),
        .rgba => |px| {
            _ = dvui.image(@src(), .{
                .source = .{ .pixels = .{ .rgba = px.px, .width = px.w, .height = px.h } },
                .shrink = .ratio,
            }, .{
                .min_size_content = .{ .w = size, .h = size },
                .max_size_content = .size(.{ .w = size, .h = size }),
                .corner_radius = R.chip,
            });
        },
    }
}

/// A dashed 1px outline: "a slot, nothing in it yet".
fn dashedBox(r: dvui.Rect.Physical, scale: f32, color: dvui.Color) void {
    if (r.w <= 0 or r.h <= 0) return;
    const dash: f32 = 3 * @max(1, scale);
    const gap: f32 = 3 * @max(1, scale);
    const th: f32 = @max(1, scale);
    var x = r.x;
    while (x < r.x + r.w) : (x += dash + gap) {
        const w = @min(dash, r.x + r.w - x);
        (dvui.Rect.Physical{ .x = x, .y = r.y, .w = w, .h = th }).fill(.{}, .{ .color = color });
        (dvui.Rect.Physical{ .x = x, .y = r.y + r.h - th, .w = w, .h = th }).fill(.{}, .{ .color = color });
    }
    var y = r.y;
    while (y < r.y + r.h) : (y += dash + gap) {
        const h = @min(dash, r.y + r.h - y);
        (dvui.Rect.Physical{ .x = r.x, .y = y, .w = th, .h = h }).fill(.{}, .{ .color = color });
        (dvui.Rect.Physical{ .x = r.x + r.w - th, .y = y, .w = th, .h = h }).fill(.{}, .{ .color = color });
    }
}

fn hintRow(src: std.builtin.SourceLocation, m: Model, cb: Actions) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 2 } });
    defer row.deinit();
    dvui.labelNoFmt(@src(), "drag to reorder", .{}, .{
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
            var cellbox = dvui.box(@src(), .{}, .{
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
                .dashed => dashedBox(crs.r, crs.s, style.hairline_hi),
            }
            const clicked = dvui.clicked(cellbox.data(), .{});
            cellbox.deinit();
            if (clicked) cb.on_open(items[i + c].id);
        }
    }
}
