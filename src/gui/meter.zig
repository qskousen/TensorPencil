//! The two-sided VRAM meter: one continuous scale over the whole card, read
//! from both ends. The LLM grows rightward, diffusion grows leftward, and the
//! gap between them IS free VRAM — so which engine is winning the card is the
//! fastest thing on the screen to read.
//!
//! Layout, left to right:
//!
//!   [sys hatch][reserve wash]  [weights|ctx|ovh]  ....free....  [lat|vae|te|dit]
//!    ^limit handle                                     ^split
//!
//! `sys` is other processes, pinned far left and hatched: not ours to reclaim.
//! The two handles are the live VRAM policy, and they match `vram.resolve`:
//!
//!   limit -> the RESERVE the user asks to keep free, measured from the LEFT.
//!            The effective reserve is `max(requested, sys)`, so our block
//!            starts at whichever is further right. When other programs hold
//!            more than the request, the handle sits left of the block's real
//!            edge, which is the meter SHOWING why the handle stopped binding.
//!   split -> the LLM's guaranteed share under contention. It sits in the free
//!            gap, which is exactly where the two sides meet.
//!
//! NO TEXT INSIDE THE BARS, ever. No segment labels, no amounts, no free
//! readout. The bar is pure color; every number lives in the legend beneath it.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");

const C = dvui.Color;
const P = style.C;
const F = style.F;

pub const Model = struct {
    total: u64,
    /// VRAM held by OTHER processes (desktop, browsers, ...), never ours.
    system: u64,
    /// OURS, but outside what the allocators track: the CUDA context(s) + JIT'd
    /// modules, cuBLASLt/cuDNN internals, the SDL/GL window and image textures,
    /// and anything a model has uploaded before its session is published.
    overhead: u64,
    llm_w: u64,
    llm_ctx: u64,
    /// Tokens currently in the KV cache, for the `ctx 0.8G · 6k` legend entry.
    ctx_tokens: usize = 0,
    te: u64,
    dit: u64,
    latent: u64,
    vae: u64,
    /// Handle positions as fractions of the card [0,1]; mutated on drag.
    /// `limit` is a CEILING fraction (the reserve is `1 - limit`), kept in that
    /// form because that is what `vram.resolve` and the config file store.
    split: *f32,
    limit: *f32,
    /// Dynamic floors (fractions): the split can't go left of the LLM's
    /// incompressible context, and diffusion keeps a small gap when loaded.
    floor_llm: f32,
    floor_diff: f32,
    llm_loaded: bool,
    diff_loaded: bool,
    llm_armed: bool,
    diff_armed: bool,
    /// Whether each worker's pause gate is engaged (parks at the next boundary,
    /// holding in-flight state + VRAM). See ops/pause.zig.
    llm_paused: bool,
    diff_paused: bool,

    fn llmTotal(m: *const Model) u64 {
        return m.llm_w + m.llm_ctx + m.overhead;
    }
    fn difTotal(m: *const Model) u64 {
        return m.te + m.dit + m.latent + m.vae;
    }
    fn freeBytes(m: *const Model) u64 {
        return m.total -| m.system -| m.llmTotal() -| m.difTotal();
    }
};

pub const Actions = struct {
    /// Fired every drag-motion frame, CHEAP only (repaint / preview).
    on_change: *const fn () void,
    /// Fired once on drag RELEASE, the place to persist and apply the policy.
    /// Never mid-drag, so we don't shuffle layers CPU<->GPU per pixel.
    on_commit: *const fn () void,
    on_eject_llm: *const fn () void,
    on_eject_diff: *const fn () void,
    on_toggle_pause_llm: *const fn () void,
    on_toggle_pause_diff: *const fn () void,
};

const rail_h: f32 = 3;
/// The track is deliberately SHORT. It is a scale, not a chart: making it tall
/// buys no precision, and at 20px it grew into the ownership rail above it so
/// the two rows read as one smeared graphic.
const track_h: f32 = 16;
/// Clear air between the rail and the track. Without it the rail's LLM span and
/// the track's `llm` segment are the same colour touching, which looks like one
/// block with a notch in it.
const rail_gap: f32 = 5;
/// The reserve handle stays a hair inside the left edge so the grip is always
/// on the bar and grabbable.
const limit_max: f32 = 0.985;
const limit_min: f32 = 0.30;

var dragging: ?enum { split, limit } = null;

/// Segment geometry, in physical px, computed ONCE and used by both the
/// ownership rail and the track. Deriving the rail separately is how the two
/// drift apart.
const Geom = struct {
    x: f32,
    w: f32,
    sys_w: f32,
    /// Left edge of our block: `max(requested reserve, sys)`.
    ours_x: f32,
    llm: [3]f32, // weights, ctx, ovh
    llm_w_total: f32,
    dif: [4]f32, // lat, vae, te, dit (left to right)
    dif_x: f32,
    dif_w_total: f32,
    free_x: f32,
    free_w: f32,

    fn build(m: *const Model, r: dvui.Rect.Physical) Geom {
        const cap: f32 = @floatFromInt(@max(m.total, 1));
        const px = r.w / cap;
        const b = struct {
            fn f(v: u64, scale: f32) f32 {
                return @round(@as(f32, @floatFromInt(v)) * scale);
            }
        };

        const sys_w = b.f(m.system, px);
        // The user's requested reserve, and the effective one. `vram.resolve`
        // takes the max of the two; so does the picture.
        const requested = @max(0, 1 - m.limit.*) * r.w;
        const ours_x = r.x + @max(requested, sys_w);

        var g: Geom = .{
            .x = r.x,
            .w = r.w,
            .sys_w = sys_w,
            .ours_x = ours_x,
            .llm = .{ b.f(m.llm_w, px), b.f(m.llm_ctx, px), b.f(m.overhead, px) },
            .llm_w_total = 0,
            .dif = .{ b.f(m.latent, px), b.f(m.vae, px), b.f(m.te, px), b.f(m.dit, px) },
            .dif_w_total = 0,
            .dif_x = 0,
            .free_x = 0,
            .free_w = 0,
        };
        for (g.llm) |v| g.llm_w_total += v;
        for (g.dif) |v| g.dif_w_total += v;

        // Diffusion right-anchors at the card's right edge and grows left. If
        // the two blocks would collide the right one is pushed right, so an
        // over-committed card reads as "no gap" rather than as overlap.
        const llm_end = g.ours_x + g.llm_w_total;
        g.dif_x = @max(llm_end, r.x + r.w - g.dif_w_total);
        g.free_x = llm_end;
        g.free_w = @max(0, g.dif_x - llm_end);
        return g;
    }
};

/// Draw the meter: ownership rail, track, legend. Fills the width it is given.
pub fn render(m: *Model, a: Actions) void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .min_size_content = .{ .w = 0 },
        .gravity_y = 0.5,
    });
    defer col.deinit();

    // (a) ownership rail: which SIDE owns what, nothing finer.
    var rail = dvui.box(@src(), .{}, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = rail_h },
        .max_size_content = .height(rail_h),
        .margin = .{ .h = rail_gap },
    });
    const rail_rs = rail.data().contentRectScale();
    rail.deinit();

    // (b) the track.
    var track = dvui.box(@src(), .{}, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = track_h },
        .max_size_content = .height(track_h),
        .background = true,
        .color_fill = P.sunken,
        .corner_radius = style.R.track,
        .margin = .{ .h = 5 },
    });
    const wd = track.data();
    // CONTENT rect, not `rectScale()`. `rectScale` is the widget's full box
    // INCLUDING its margin, so filling it painted the segments 5px past the
    // background they are supposed to sit inside — straight over the ownership
    // rail above and the legend below.
    const rs = wd.contentRectScale();
    if (rs.r.w > 1 and rs.r.h > 1) {
        const g = Geom.build(m, rs.r);
        drawRail(m, g, rail_rs.r);
        drawTrack(m, g, rs.r, rs.s);
        handleDrag(m, a, wd, rs);
    }
    track.deinit();

    // (c) legend.
    legend(@src(), m, a);
}

fn seg(r: dvui.Rect.Physical, x: f32, w: f32, c: C) void {
    if (w <= 0) return;
    (dvui.Rect.Physical{ .x = x, .y = r.y, .w = w, .h = r.h }).fill(.{}, .{ .color = c });
}

/// The rail says only "sys | LLM | free | diffusion", derived by summing the
/// track's own segments so the two can never disagree.
fn drawRail(m: *const Model, g: Geom, r: dvui.Rect.Physical) void {
    if (r.w <= 0 or r.h <= 0) return;
    seg(r, g.x, g.sys_w, P.vram_sys);
    // A 1px canvas gap: what other programs hold is not part of our bracket.
    if (m.llmTotal() > 0) seg(r, g.ours_x + 1, g.llm_w_total - 1, P.vram_llm);
    seg(r, g.free_x, g.free_w, P.vram_free);
    if (m.difTotal() > 0) seg(r, g.dif_x, g.dif_w_total, P.vram_dit);
}

fn drawTrack(m: *Model, g: Geom, r: dvui.Rect.Physical, scale: f32) void {
    // 1. Other processes: hatched, because it is the one block we can never
    //    reclaim, and a flat fill would read as just another segment.
    if (g.sys_w > 0) {
        style.hatch(.{ .x = g.x, .y = r.y, .w = g.sys_w, .h = r.h }, P.vram_sys, P.vram_sys_alt, 4 * scale);
    }
    // 2. Reserve we asked to keep free beyond what others hold: a dim wash, so
    //    the gap between the handle and our block is legible as "held back".
    if (g.ours_x > g.x + g.sys_w) {
        seg(r, g.x + g.sys_w, g.ours_x - (g.x + g.sys_w), style.tint(P.vram_sys, 60));
    }
    // 3. A hard edge: not ours to reclaim.
    seg(r, g.ours_x, 1, P.canvas);

    // 4. The LLM, growing rightward.
    var x = g.ours_x + 1;
    inline for (.{ P.vram_llm, P.vram_ctx, P.vram_ovh }, 0..) |c, i| {
        seg(r, x, g.llm[i], c);
        x += g.llm[i];
    }

    // 5. free is the gap; the track's own sunken background is it.

    // 6. Diffusion, mirrored: its order runs lat, vae, te, dit so that the
    //    biggest block (the DiT) lands against the right edge.
    x = g.dif_x;
    inline for (.{ P.vram_lat, P.vram_vae, P.vram_te, P.vram_dit }, 0..) |c, i| {
        seg(r, x, g.dif[i], c);
        x += g.dif[i];
    }

    // Handles last, on top: they are policy markers, not usage.
    handle(r, g.x + (1 - m.limit.*) * g.w, P.amber);
    handle(r, g.x + m.split.* * g.w, P.blue);
}

/// A handle is a 2px stem with a grip. Amber for the reserve (a machine limit),
/// blue for the split (a value the user sets).
fn handle(r: dvui.Rect.Physical, hx: f32, c: C) void {
    (dvui.Rect.Physical{ .x = hx - 1, .y = r.y - 3, .w = 2, .h = r.h + 6 }).fill(.{}, .{ .color = c });
    (dvui.Rect.Physical{ .x = hx - 3.5, .y = r.y + r.h / 2 - 6, .w = 7, .h = 12 })
        .fill(dvui.Rect.Physical.all(2), .{ .color = c });
}

// ------------------------------------------------------------------- legend

fn gib(v: u64) f64 {
    return @as(f64, @floatFromInt(v)) / (1 << 30);
}

/// One formatted legend entry, measured before anything is drawn.
const Item = struct {
    color: C,
    text: []const u8,
    /// 3 = essential, 0 = first to go. The totals and `free` are never
    /// droppable: they are the headline numbers the rest elaborate on.
    importance: u8,
    present: bool,
    w: f32 = 0,
};

fn legend(src: std.builtin.SourceLocation, m: *Model, a: Actions) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{ .expand = .horizontal });
    defer row.deinit();

    // Everything is formatted and MEASURED first, then entries are dropped in
    // priority order until the row fits. Laying it out optimistically is how
    // the right-hand group -- which carries the diffusion total -- silently
    // fell off the end of a 1200px window while looking fine on a 1526px one.
    var bufs: [8][48]u8 = undefined;
    var tok: [12]u8 = undefined;
    // Importance is what the segment TELLS you. `llm`/`ctx` and `dit` are the
    // blocks a user acts on; `ovh` and `lat` are derived detail that only
    // matters when you are already staring at the bar.
    const left = [_]Item{
        entry(&bufs[0], P.vram_llm, "llm", m.llm_w, m.llm_loaded, null, 3),
        entry(&bufs[1], P.vram_ctx, "ctx", m.llm_ctx, m.llm_loaded, if (m.ctx_tokens > 0) fmtTokens(&tok, m.ctx_tokens) else null, 3),
        entry(&bufs[2], P.vram_ovh, "ovh", m.overhead, m.llm_loaded or m.diff_loaded, null, 1),
        entry(&bufs[3], P.vram_sys, "sys", m.system, true, null, 2),
    };
    const right = [_]Item{
        entry(&bufs[4], P.vram_te, "TE", m.te, m.diff_loaded, null, 2),
        entry(&bufs[5], P.vram_dit, "dit", m.dit, m.diff_loaded, null, 3),
        entry(&bufs[6], P.vram_vae, "VAE", m.vae, m.diff_loaded, null, 1),
        entry(&bufs[7], P.vram_lat, "lat", m.latent, m.diff_loaded, null, 1),
    };

    var tbuf: [40]u8 = undefined;
    var dbuf: [40]u8 = undefined;
    var fbuf: [24]u8 = undefined;
    const llm_total = std.fmt.bufPrint(&tbuf, "LLM {d:.1}G", .{gib(m.llmTotal())}) catch "LLM";
    const dif_total = std.fmt.bufPrint(&dbuf, "DIFFUSION {d:.1}G", .{gib(m.difTotal())}) catch "DIFFUSION";
    const free_b = m.freeBytes();
    const free_txt = std.fmt.bufPrint(&fbuf, "{d:.1}G free", .{gib(free_b)}) catch "";

    // Fixed cost: the two totals with their arrows, the free readout, the four
    // control buttons, and the gaps around them.
    const fixed: f32 = F.mono_val.textSize(llm_total).w + F.mono_val.textSize(dif_total).w +
        F.mono_legend.textSize(free_txt).w +
        4 * 21 + // the four control buttons
        2 * 34 + // the totals' arrows and their margins
        40; // gaps around the free readout
    const budget = row.data().contentRect().w - fixed;

    // Keep every entry at or above `keep`, raising the bar until the row fits.
    // 1 keeps everything; 4 keeps only the totals.
    var keep: u8 = 1;
    while (keep < 4) : (keep += 1) {
        var need: f32 = 0;
        for (left) |it| {
            if (it.importance >= keep) need += it.w;
        }
        for (right) |it| {
            if (it.importance >= keep) need += it.w;
        }
        if (need <= budget) break;
    }

    {
        var grp = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_y = 0.5 });
        defer grp.deinit();

        // This side's controls sit with this side's numbers, so the handedness
        // of the bar carries through to the buttons that act on it.
        if (pauseBtn(0, m.llm_paused, "Pause the LLM — holds generation (nothing loads until resumed)")) a.on_toggle_pause_llm();
        if (eject(0, m.llm_loaded, m.llm_armed, "Unload the LLM — frees its VRAM (reloads on the next message)")) a.on_eject_llm();

        total(@src(), llm_total, P.vram_llm, .left);
        for (left, 0..) |it, i| if (it.importance >= keep) drawItem(@src(), i, it);
    }

    {
        var mid = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .min_size_content = .{ .w = 40 },
            .margin = .{ .x = 8, .w = 8 },
        });
        defer mid.deinit();
        dvui.labelNoFmt(@src(), free_txt, .{}, .{
            .font = F.mono_legend,
            // The one place the bar shouts: nearly out of card.
            .color_text = if (free_b < (2 << 30)) P.amber else P.text_ghost,
            .gravity_x = 0.5,
            .gravity_y = 0.5,
            .padding = .{},
        });
    }

    {
        var grp = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_y = 0.5 });
        defer grp.deinit();
        for (right, 0..) |it, i| if (it.importance >= keep) drawItem(@src(), 4 + i, it);
        total(@src(), dif_total, P.vram_dit, .right);

        if (eject(1, m.diff_loaded, m.diff_armed, "Unload the diffusion model — frees its VRAM (reloads on the next image)")) a.on_eject_diff();
        if (pauseBtn(1, m.diff_paused, "Pause diffusion — holds generation (nothing loads until resumed)")) a.on_toggle_pause_diff();
    }
}

/// Format one entry and measure it. An UNLOADED engine's entries read `llm —`
/// rather than `llm 0.0G`: the bar states what is resident right now, and a
/// zero is a different claim from an absence.
fn entry(buf: []u8, c: C, name: []const u8, bytes: u64, loaded: bool, suffix: ?[]const u8, importance: u8) Item {
    const present = loaded and bytes >= (1 << 20);
    const text = if (!present)
        std.fmt.bufPrint(buf, "{s} —", .{name}) catch name
    else if (suffix) |sfx|
        std.fmt.bufPrint(buf, "{s} {d:.1}G · {s}", .{ name, gib(bytes), sfx }) catch name
    else
        std.fmt.bufPrint(buf, "{s} {d:.1}G", .{ name, gib(bytes) }) catch name;
    // swatch (6) + its gap (5) + the entry's own trailing margin (8)
    return .{ .color = c, .text = text, .importance = importance, .present = present, .w = F.mono_legend.textSize(text).w + 19 };
}

fn drawItem(src: std.builtin.SourceLocation, id: usize, it: Item) void {
    var b = dvui.box(src, .{ .dir = .horizontal }, .{ .id_extra = id, .gravity_y = 0.5, .margin = .{ .w = 8 } });
    defer b.deinit();
    style.swatch(@src(), if (it.present) it.color else style.over(P.chrome, it.color, 0.28));
    dvui.labelNoFmt(@src(), it.text, .{}, .{
        .font = F.mono_legend,
        .color_text = if (it.present) P.text_dim else P.text_ghost,
        .padding = .{},
        .gravity_y = 0.5,
    });
}

/// A group total, with the arrow pointing the way its side grows.
fn total(src: std.builtin.SourceLocation, text: []const u8, c: C, side: enum { left, right }) void {
    var b = dvui.box(src, .{ .dir = .horizontal }, .{ .gravity_y = 0.5, .margin = .{ .x = 6, .w = 10 } });
    defer b.deinit();
    if (side == .right) arrow(@src(), "←", c);
    dvui.labelNoFmt(@src(), text, .{}, .{
        .font = F.mono_val,
        .color_text = c,
        .padding = .{},
        .gravity_y = 0.5,
    });
    if (side == .left) arrow(@src(), "→", c);
}

fn arrow(src: std.builtin.SourceLocation, glyph: []const u8, c: C) void {
    dvui.labelNoFmt(src, glyph, .{}, .{
        .font = F.mono_val,
        .color_text = c,
        .padding = .{},
        .margin = .{ .x = 4, .w = 4 },
        .gravity_y = 0.5,
    });
}

/// "823", "3.2k", "128k".
fn fmtTokens(buf: []u8, n: usize) []const u8 {
    if (n >= 1000) return std.fmt.bufPrint(buf, "{d:.1}k", .{@as(f64, @floatFromInt(n)) / 1000.0}) catch "?";
    return std.fmt.bufPrint(buf, "{d}", .{n}) catch "?";
}

// ------------------------------------------------------------------ controls

fn eject(id: usize, loaded: bool, armed: bool, hint_text: []const u8) bool {
    var wd: dvui.WidgetData = undefined;
    const clicked = dvui.buttonIcon(@src(), "eject", dvui.entypo.circle_with_minus, .{}, .{}, .{
        .id_extra = id,
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 13, .h = 13 },
        .margin = .{ .x = 2, .w = 2 },
        .padding = dvui.Rect.all(3),
        .corner_radius = style.R.chip,
        .background = false,
        .color_fill_hover = style.hover_wash,
        .color_fill_press = style.hover_wash,
        .color_text = if (armed) P.amber else if (loaded) P.text_faint else P.text_ghost,
        .color_border = if (armed) P.amber else null,
        .border = if (armed) dvui.Rect.all(1) else .{},
        .data_out = &wd,
    });
    hint.hover(@src(), &wd, if (armed) "Unloading when idle…" else hint_text);
    return clicked and loaded;
}

/// Pause/resume for one engine. While paused the glyph BLINKS amber (~1.25 Hz)
/// so a parked state is unmissable, and flips to ▶. ALWAYS clickable: pausing
/// with nothing resident pre-arms the gate, so the next message or image is
/// held and nothing loads at all. See app.toggleLlmPause and Diffuser.pump.
fn pauseBtn(id: usize, paused: bool, hint_text: []const u8) bool {
    const half_ns: i128 = 400_000_000;
    const now = dvui.frameTimeNS();
    const phase = @mod(now, 2 * half_ns);
    const lit = phase < half_ns;

    var wd: dvui.WidgetData = undefined;
    const clicked = dvui.buttonIcon(@src(), "pause", if (paused) dvui.entypo.controller_play else dvui.entypo.controller_pause, .{}, .{}, .{
        .id_extra = id,
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 13, .h = 13 },
        .margin = .{ .x = 2, .w = 2 },
        .padding = dvui.Rect.all(3),
        .corner_radius = style.R.chip,
        .background = false,
        .color_fill_hover = style.hover_wash,
        .color_fill_press = style.hover_wash,
        .color_text = if (paused) (if (lit) P.amber else style.over(P.chrome, P.amber, 0.35)) else P.text_faint,
        .color_border = if (paused) P.amber else null,
        .border = if (paused) dvui.Rect.all(1) else .{},
        .data_out = &wd,
    });
    // Keep the blink alive: schedule a repaint at the coming phase flip, since
    // the UI wakes ~2.5x/s rather than every frame.
    if (paused) {
        const to_flip: i128 = if (lit) half_ns - phase else 2 * half_ns - phase;
        dvui.timer(wd.id, @intCast(@divFloor(to_flip, 1000) + 1));
    }
    hint.hover(@src(), &wd, if (paused) "Resume" else hint_text);
    return clicked;
}

// ---------------------------------------------------------------------- drag

fn handleDrag(m: *Model, a: Actions, wd: *dvui.WidgetData, crs: dvui.RectScale) void {
    const r = crs.r;
    const grab = 11 * crs.s;
    for (dvui.events()) |*e| {
        if (e.handled or e.evt != .mouse) continue;
        const me = e.evt.mouse;
        switch (me.action) {
            .press => {
                if (!me.button.pointer() or !dvui.eventMatchSimple(e, wd)) continue;
                const sx = r.x + m.split.* * r.w;
                const lx = r.x + (1 - m.limit.*) * r.w;
                const ds = @abs(me.p.x - sx);
                const dl = @abs(me.p.x - lx);
                // Grab the nearer handle if within reach; a click OFF a handle
                // must not move anything.
                dragging = if (dl <= grab and dl <= ds) .limit else if (ds <= grab) .split else null;
                if (dragging != null) {
                    e.handled = true;
                    dvui.captureMouse(wd, e.num);
                    dvui.dragPreStart(me.p, .{});
                }
            },
            // Move by the per-event DELTA, never by absolute cursor position, so
            // the handle tracks from where the drag started and cannot jump.
            .motion => |delta| {
                if (!dvui.captured(wd.id) or dragging == null) continue;
                if (dvui.dragging(me.p, null) == null) continue; // past the click threshold
                e.handled = true;
                dragBy(m, delta.x / r.w);
                a.on_change();
            },
            .release => {
                if (!dvui.captured(wd.id)) continue;
                e.handled = true;
                dvui.captureMouse(null, e.num);
                dragging = null;
                a.on_commit(); // apply + persist the settled position
            },
            else => {},
        }
    }
}

/// Nudge the active handle by `df` (a fraction of the card), clamped. Ranges
/// are inversion-guarded (hi >= lo) so a large floor can pin a handle but never
/// lock it.
fn dragBy(m: *Model, df: f32) void {
    switch (dragging.?) {
        .split => {
            const hi = @max(m.floor_llm, 1 - m.floor_diff);
            m.split.* = std.math.clamp(m.split.* + df, m.floor_llm, hi);
        },
        .limit => {
            // The handle is drawn at `1 - limit`, so dragging it right SHRINKS
            // the ceiling. Inverting the delta here keeps the grip under the
            // pointer, which is the only thing the user is tracking.
            m.limit.* = std.math.clamp(m.limit.* - df, limit_min, limit_max);
        },
    }
}

test "geometry places our block at the larger of the requested reserve and sys" {
    var split: f32 = 0.5;
    var limit: f32 = 0.9; // asks to keep 10% free
    var m: Model = .{
        .total = 100,
        .system = 25, // other programs hold more than the request
        .overhead = 0,
        .llm_w = 10,
        .llm_ctx = 0,
        .te = 0,
        .dit = 20,
        .latent = 0,
        .vae = 0,
        .split = &split,
        .limit = &limit,
        .floor_llm = 0.04,
        .floor_diff = 0.04,
        .llm_loaded = true,
        .diff_loaded = true,
        .llm_armed = false,
        .diff_armed = false,
        .llm_paused = false,
        .diff_paused = false,
    };
    const r: dvui.Rect.Physical = .{ .x = 0, .y = 0, .w = 100, .h = 20 };

    // sys (25) beats the 10% request, so our block starts at 25.
    var g = Geom.build(&m, r);
    try std.testing.expectApproxEqAbs(@as(f32, 25), g.ours_x, 0.01);
    // Diffusion right-anchors: 20 wide against the right edge.
    try std.testing.expectApproxEqAbs(@as(f32, 80), g.dif_x, 0.01);
    // Free is the gap between the two blocks: 100 - 25 - 10 - 20 = 45.
    try std.testing.expectApproxEqAbs(@as(f32, 45), g.free_w, 0.01);

    // Now ask to keep 40% free: the request wins and our block moves right.
    limit = 0.6;
    g = Geom.build(&m, r);
    try std.testing.expectApproxEqAbs(@as(f32, 40), g.ours_x, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 30), g.free_w, 0.01);
}

test "an over-committed card shows no gap rather than overlapping blocks" {
    var split: f32 = 0.5;
    var limit: f32 = 1.0;
    var m: Model = .{
        .total = 100,
        .system = 10,
        .overhead = 0,
        .llm_w = 60,
        .llm_ctx = 0,
        .te = 0,
        .dit = 50, // 10 + 60 + 50 > 100
        .latent = 0,
        .vae = 0,
        .split = &split,
        .limit = &limit,
        .floor_llm = 0.04,
        .floor_diff = 0.04,
        .llm_loaded = true,
        .diff_loaded = true,
        .llm_armed = false,
        .diff_armed = false,
        .llm_paused = false,
        .diff_paused = false,
    };
    const g = Geom.build(&m, .{ .x = 0, .y = 0, .w = 100, .h = 20 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), g.free_w, 0.01);
    // The diffusion block is pushed right of the LLM's end, never under it.
    try std.testing.expect(g.dif_x >= g.ours_x + g.llm_w_total);
}

test "dragging the reserve handle right shrinks the ceiling" {
    var split: f32 = 0.5;
    var limit: f32 = 0.9;
    var m: Model = .{
        .total = 100,
        .system = 0,
        .overhead = 0,
        .llm_w = 0,
        .llm_ctx = 0,
        .te = 0,
        .dit = 0,
        .latent = 0,
        .vae = 0,
        .split = &split,
        .limit = &limit,
        .floor_llm = 0.04,
        .floor_diff = 0.04,
        .llm_loaded = false,
        .diff_loaded = false,
        .llm_armed = false,
        .diff_armed = false,
        .llm_paused = false,
        .diff_paused = false,
    };
    dragging = .limit;
    defer dragging = null;
    dragBy(&m, 0.1); // grip moves right by 10% of the card
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), limit, 0.001);
    // And it cannot be dragged off either end.
    dragBy(&m, -10);
    try std.testing.expectApproxEqAbs(limit_max, limit, 0.001);
    dragBy(&m, 10);
    try std.testing.expectApproxEqAbs(limit_min, limit, 0.001);
}
