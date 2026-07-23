//! The VRAM meter: one bar over the whole card, with two draggable handles
//! (split = LLM|diffusion boundary, limit = ceiling) and an unload (⏏) button
//! at each end. Segments show what's actually resident by component; the
//! handles are a soft, live VRAM policy the app applies (offload / stream /
//! cap). See the design mockup / `vram-meter-design` memory.
//!
//! This module is pure presentation + input: it draws the `Model` the app fills
//! each frame and mutates `Model.split`/`Model.limit` (fractions of the card) on
//! drag, calling back so the app can persist + apply the new policy live.
const std = @import("std");
const dvui = @import("dvui");
const hint = @import("hint.zig");

const C = dvui.Color;
// Segment colors (match the mockup). Fixed semantics.
const col = struct {
    const free = C{ .r = 10, .g = 13, .b = 18 };
    const sys = C{ .r = 138, .g = 109, .b = 74 };
    const llm_w = C{ .r = 63, .g = 134, .b = 214 };
    const llm_ctx = C{ .r = 225, .g = 233, .b = 244 };
    const te = C{ .r = 216, .g = 194, .b = 74 };
    const dit = C{ .r = 224, .g = 138, .b = 60 };
    const latent = C{ .r = 216, .g = 82, .b = 74 };
    const vae = C{ .r = 154, .g = 109, .b = 208 };
    const head = C{ .r = 128, .g = 138, .b = 156, .a = 46 };
    const split = C{ .r = 234, .g = 241, .b = 251 };
    const limit = C{ .r = 239, .g = 162, .b = 60 };
};

pub const Model = struct {
    total: u64,
    system: u64,
    llm_w: u64,
    llm_ctx: u64,
    te: u64,
    dit: u64,
    latent: u64,
    vae: u64,
    /// Handle positions as fractions of the card [0,1]; mutated on drag.
    split: *f32,
    limit: *f32,
    /// Dynamic floors (fractions): split can't go left of `floor_llm`
    /// (system+context); the limit↔split gap can't close below `floor_diff`.
    floor_llm: f32,
    floor_diff: f32,
    llm_loaded: bool,
    diff_loaded: bool,
    llm_armed: bool,
    diff_armed: bool,
    /// Whether each worker's pause gate is currently engaged (parks at the next
    /// boundary, holding in-flight state + VRAM). Drives the blinking pause
    /// button next to each unload button. See ops/pause.zig.
    llm_paused: bool,
    diff_paused: bool,
};

pub const Actions = struct {
    /// Fired every drag-motion frame — CHEAP only (repaint / preview). The
    /// fractions are already mutated in place by the drag; this just notifies.
    on_change: *const fn () void,
    /// Fired once on drag RELEASE — the place to do heavy work (persist the
    /// fractions, apply the offload/budget policy live). Never fires mid-drag,
    /// so we don't shuffle layers CPU↔GPU on every pixel of movement.
    on_commit: *const fn () void,
    on_eject_llm: *const fn () void,
    on_eject_diff: *const fn () void,
    /// Toggle each worker's pause independently — parks/releases the LLM decode
    /// worker or the diffusion sampling worker at its next boundary (holding
    /// in-flight state + VRAM). See ops/pause.zig.
    on_toggle_pause_llm: *const fn () void,
    on_toggle_pause_diff: *const fn () void,
};

const height: f32 = 30;
var dragging: ?enum { split, limit } = null;
/// The limit handle never goes fully to the edge: it stays a hair inside so the
/// grip is always drawn on the bar (grabbable) and there's always a sliver of
/// headroom to its right.
const limit_max: f32 = 0.985;

/// Render the full status-bar row: LLM numbers · ⏏ · bar · ⏏ · diffusion
/// numbers. One horizontal row (no vertical nesting).
pub fn render(m: *Model, a: Actions) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .gravity_y = 0.5 });
    defer row.deinit();

    // LLM numbers (left): weights + context, stacked.
    numCol(0, &.{ .{ .name = "weights", .c = col.llm_w, .v = m.llm_w }, .{ .name = "ctx", .c = col.llm_ctx, .v = m.llm_ctx } });

    if (pauseBtn(0, m.llm_loaded, m.llm_paused, "Pause the LLM — parks generation, keeps it resident")) a.on_toggle_pause_llm();
    if (eject(0, m.llm_loaded, m.llm_armed, "Unload the LLM — frees its VRAM (reloads on the next message)")) a.on_eject_llm();

    {
        // The box's own background paints the "free" color (reliable); segments
        // overlay on top. min height reserves the row so it can't collapse onto
        // the readout row below.
        var bar = dvui.box(@src(), .{}, .{
            .expand = .horizontal,
            .gravity_y = 0.5,
            .min_size_content = .{ .h = height },
            .background = true,
            .color_fill = col.free,
            .corner_radius = dvui.Rect.all(4),
            .margin = .{ .x = 8, .w = 8 },
        });
        defer bar.deinit();
        const wd = bar.data();
        const crs = wd.rectScale();
        if (crs.r.w > 1 and crs.r.h > 1) drawBar(m, crs.r, crs.s);
        handleDrag(m, a, wd, crs);
    }

    if (pauseBtn(1, m.diff_loaded, m.diff_paused, "Pause diffusion — parks generation, keeps it resident")) a.on_toggle_pause_diff();
    if (eject(1, m.diff_loaded, m.diff_armed, "Unload the diffusion model — frees its VRAM (reloads on the next image)")) a.on_eject_diff();

    // Diffusion numbers (right): TE/DiT over latent/VAE, two mini-columns.
    numCol(1, &.{ .{ .name = "TE", .c = col.te, .v = m.te }, .{ .name = "lat", .c = col.latent, .v = m.latent } });
    numCol(2, &.{ .{ .name = "DiT", .c = col.dit, .v = m.dit }, .{ .name = "VAE", .c = col.vae, .v = m.vae } });
}

const Item = struct { name: []const u8, c: C, v: u64 };

/// A stacked mini-column of memory readouts (swatch · label · size), dimmed
/// when a part isn't loaded. Fixed row height so the strip never pops.
fn numCol(id: usize, items: []const Item) void {
    var colb = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = id, .gravity_y = 0.5, .margin = .{ .x = 6, .w = 6 } });
    defer colb.deinit();
    const th = dvui.themeGet();
    for (items, 0..) |it, i| {
        var r = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = i });
        defer r.deinit();
        const dim = it.v < (1 << 20);
        _ = dvui.box(@src(), .{}, .{
            .min_size_content = .{ .w = 8, .h = 8 },
            .gravity_y = 0.5,
            .background = true,
            .color_fill = if (dim) th.fill.lerp(th.text, 0.2) else it.c,
            .corner_radius = dvui.Rect.all(2),
            .margin = .{ .w = 5 },
        }).deinit();
        var buf: [24]u8 = undefined;
        const txt = std.fmt.bufPrint(&buf, "{s} {d:.1}G", .{ it.name, @as(f64, @floatFromInt(it.v)) / (1 << 30) }) catch it.name;
        dvui.label(@src(), "{s}", .{txt}, .{
            .gravity_y = 0.5,
            .font = dvui.Font.theme(.body).withSize(9),
            .padding = .{},
            .margin = .{ .y = 1, .h = 1 },
            .color_text = if (dim) th.fill.lerp(th.text, 0.4) else null,
        });
    }
}

fn eject(id: usize, loaded: bool, armed: bool, hint_text: []const u8) bool {
    const th = dvui.themeGet();
    // ⏏ isn't in the bundled font; use the entypo eject glyph. Dimmed when the
    // model isn't loaded; accent-colored + bordered when armed (deferred unload).
    var wd: dvui.WidgetData = undefined;
    const clicked = dvui.buttonIcon(@src(), "eject", dvui.entypo.circle_with_minus, .{}, .{}, .{
        .id_extra = id,
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 18, .h = 20 },
        .margin = .{ .x = 3, .w = 3 },
        .padding = dvui.Rect.all(3),
        .corner_radius = dvui.Rect.all(5),
        .color_text = if (armed) col.limit else if (loaded) null else th.fill.lerp(th.text, 0.28),
        .color_border = if (armed) col.limit else null,
        .border = if (armed) dvui.Rect.all(1) else .{},
        .data_out = &wd,
    });
    hint.hover(@src(), &wd, if (armed) "Unloading when idle…" else hint_text);
    return clicked and loaded;
}

/// Pause/resume toggle for one model, styled to match the eject button. While
/// paused the glyph BLINKS orange (a ~1.25 Hz square wave) so the parked state
/// is unmissable, and flips to ▶ (resume). Dimmed + inert when the model isn't
/// loaded. Returns true when clicked (and loaded).
fn pauseBtn(id: usize, loaded: bool, paused: bool, hint_text: []const u8) bool {
    const th = dvui.themeGet();

    // Square-wave blink phase off the frame clock: 400 ms lit, 400 ms dim.
    const half_ns: i128 = 400_000_000;
    const now = dvui.frameTimeNS();
    const phase = @mod(now, 2 * half_ns);
    const lit = phase < half_ns;

    const color_text: ?C = if (paused)
        (if (lit) col.limit else col.limit.lerp(th.fill, 0.72))
    else if (loaded) null else th.fill.lerp(th.text, 0.28);

    var wd: dvui.WidgetData = undefined;
    const clicked = dvui.buttonIcon(@src(), "pause", if (paused) dvui.entypo.controller_play else dvui.entypo.controller_pause, .{}, .{}, .{
        .id_extra = id,
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 18, .h = 20 },
        .margin = .{ .x = 3, .w = 3 },
        .padding = dvui.Rect.all(3),
        .corner_radius = dvui.Rect.all(5),
        .color_text = color_text,
        .color_border = if (paused) col.limit else null,
        .border = if (paused) dvui.Rect.all(1) else .{},
        .data_out = &wd,
    });
    // Keep the blink alive while paused: schedule the next repaint at the coming
    // phase flip (the UI wakes ~2.5×/s, not every frame).
    if (paused) {
        const to_flip: i128 = if (lit) half_ns - phase else 2 * half_ns - phase;
        dvui.timer(wd.id, @intCast(@divFloor(to_flip, 1000) + 1));
    }
    hint.hover(@src(), &wd, if (paused) "Resume" else hint_text);
    return clicked and loaded;
}

fn frac(m: *const Model, bytes: u64) f32 {
    if (m.total == 0) return 0;
    return @as(f32, @floatFromInt(bytes)) / @as(f32, @floatFromInt(m.total));
}

fn drawBar(m: *Model, R: dvui.Rect.Physical, scale: f32) void {
    const x0 = R.x;
    const W = R.w;
    const seg = struct {
        fn at(rr: dvui.Rect.Physical, ox0: f32, ow: f32, a: f32, b: f32, c: C) void {
            if (b <= a) return;
            const r: dvui.Rect.Physical = .{ .x = ox0 + a * ow, .y = rr.y, .w = (b - a) * ow, .h = rr.h };
            r.fill(.{}, .{ .color = c });
        }
    };
    // ONE contiguous left→right sweep of the whole card — no anchoring, no
    // overlap — so nothing hides beneath anything and usage that overruns the
    // limit visibly crosses the limit marker. Order: everything WE allocate
    // first (LLM weights, context, then the diffusion stack), packed contiguous
    // on the left, THEN the free gap (bar background shows through), THEN
    // system/reserved at the far right. All segments sum to the card total.
    var used: u64 = 0;
    inline for (.{ m.llm_w, m.llm_ctx, m.te, m.dit, m.latent, m.vae, m.system }) |b| used += b;
    const free_b: u64 = m.total -| used;

    var x: f32 = 0;
    // Our stuff: LLM (weights, context) then the diffusion stack, contiguous.
    inline for (.{
        .{ m.llm_w, col.llm_w },
        .{ m.llm_ctx, col.llm_ctx },
        .{ m.te, col.te },
        .{ m.dit, col.dit },
        .{ m.latent, col.latent },
        .{ m.vae, col.vae },
    }) |it| {
        const w = frac(m, it[0]);
        seg.at(R, x0, W, x, x + w, it[1]);
        x += w;
    }
    // Free VRAM — advance across the gap; the box's "free"-colored background
    // shows through (no fill needed).
    x += frac(m, free_b);
    // System / reserved (OS + context overhead) at the far right.
    seg.at(R, x0, W, x, x + frac(m, m.system), col.sys);

    // Headroom: dim wash from the limit handle to the bar's right edge (VRAM we
    // won't allocate). Round its RIGHT corners to match the bar's rounded
    // background so the wash doesn't poke past the end.
    if (m.limit.* < 1) {
        const rad = 4 * scale;
        const hr: dvui.Rect.Physical = .{ .x = x0 + m.limit.* * W, .y = R.y, .w = (1 - m.limit.*) * W, .h = R.h };
        hr.fill(.{ .x = 0, .y = rad, .w = rad, .h = 0 }, .{ .color = col.head });
    }

    // Handles are policy MARKERS overlaid on top (not usage) — the split is the
    // LLM|diffusion contention boundary, the limit is the ceiling.
    handle(R, x0, W, m.split.*, col.split);
    handle(R, x0, W, m.limit.*, col.limit);
}

fn handle(R: dvui.Rect.Physical, x0: f32, W: f32, f: f32, c: C) void {
    const hx = x0 + f * W;
    // stem
    (dvui.Rect.Physical{ .x = hx - 1, .y = R.y - 3, .w = 2, .h = R.h + 6 }).fill(.{}, .{ .color = c });
    // grip
    (dvui.Rect.Physical{ .x = hx - 4, .y = R.y + R.h / 2 - 8, .w = 8, .h = 16 }).fill(dvui.Rect.Physical.all(2), .{ .color = c });
}

fn handleDrag(m: *Model, a: Actions, wd: *dvui.WidgetData, crs: dvui.RectScale) void {
    const R = crs.r;
    const grab = 11 * crs.s;
    for (dvui.events()) |*e| {
        if (e.handled or e.evt != .mouse) continue;
        const me = e.evt.mouse;
        switch (me.action) {
            .press => {
                if (!me.button.pointer() or !dvui.eventMatchSimple(e, wd)) continue;
                const sx = R.x + m.split.* * R.w;
                const lx = R.x + m.limit.* * R.w;
                const ds = @abs(me.p.x - sx);
                const dl = @abs(me.p.x - lx);
                // Grab the nearer handle if within reach; otherwise ignore the
                // click (a click OFF a handle must not move anything).
                dragging = if (dl <= grab and dl <= ds) .limit else if (ds <= grab) .split else null;
                if (dragging != null) {
                    e.handled = true;
                    dvui.captureMouse(wd, e.num);
                    dvui.dragPreStart(me.p, .{});
                }
            },
            // Move by the per-event DELTA (viewer.zig's proven idiom), never by
            // absolute cursor position — so the handle tracks the drag smoothly
            // from where it started and can't jump to the cursor.
            .motion => |delta| {
                if (!dvui.captured(wd.id) or dragging == null) continue;
                if (dvui.dragging(me.p, null) == null) continue; // past the click threshold
                e.handled = true;
                dragBy(m, delta.x / R.w);
                a.on_change();
            },
            .release => {
                if (!dvui.captured(wd.id)) continue;
                e.handled = true;
                dvui.captureMouse(null, e.num);
                dragging = null;
                a.on_commit(); // apply + persist the settled handle position
            },
            else => {},
        }
    }
}

/// Nudge the active handle by `df` (a fraction of the bar), clamped. Ranges are
/// inversion-guarded (hi ≥ lo) so a large floor / tight limit can pin but never
/// lock the handle.
fn dragBy(m: *Model, df: f32) void {
    switch (dragging.?) {
        .split => {
            const hi = @max(m.floor_llm, m.limit.* - m.floor_diff);
            m.split.* = std.math.clamp(m.split.* + df, m.floor_llm, hi);
        },
        .limit => {
            const lo = @min(m.floor_llm + m.floor_diff, limit_max);
            m.limit.* = std.math.clamp(m.limit.* + df, lo, limit_max);
            // Push the split left if the limit closed in on it.
            const split_hi = m.limit.* - m.floor_diff;
            if (m.split.* > split_hi) m.split.* = @max(m.floor_llm, split_hi);
        },
    }
}
