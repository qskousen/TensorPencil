//! The tp-gui status bar: three history meters (GPU / CPU / VRAM) on the left,
//! then the two-sided VRAM meter (see meter.zig) filling the rest. Drawn from
//! the host's telemetry as the mirror holds it; the host does the sampling.
//!
//! It spans the WHOLE window, under all three columns, and never migrates into
//! a panel: this telemetry is a property of the app, not of any one view. Its
//! fixed height (`bar_height`) is subtracted from the body band by the caller.
//!
//! This module owns the model-building; meter.zig owns the track's drawing and
//! interaction, and style.zig owns the history meter, which the `ui-probe`
//! harness also draws.
const std = @import("std");
const dvui = @import("dvui");
const mirror = @import("client").mirror;
const wire = @import("serve").wire;
const meter = @import("meter.zig");
const vram_split = @import("shared").vram_split;
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;
const F = style.F;

/// Fixed bar height (logical px), reserved by the caller. Spans the WHOLE
/// window, under all three columns: telemetry is a property of the app, not of
/// the queue rail it used to sit in.
pub const bar_height: f32 = style.Layout.status_h;
const pad_top: f32 = 9;
const pad_bottom: f32 = 10;

/// What one bar actually costs the layout: its content plus its own padding
/// and top hairline. The CALLER reserves this, not `bar_height`; reserving the
/// content height alone overruns the window by the padding, which with one bar
/// costs an invisible hairline and with two clips the second one's readout.
pub const bar_outer_height: f32 = bar_height + pad_top + pad_bottom + 1;

/// Samples kept per sparkline. At the host's ~2 Hz idle cadence this is a
/// ~13 s window, which is long enough to show a generation start and short
/// enough that the graph still moves.
const hist_n = 26;

/// A small fixed-capacity rolling history, iterated oldest->newest via `at`.
const Ring = struct {
    data: [hist_n]f32 = [_]f32{0} ** hist_n,
    len: usize = 0,
    head: usize = 0,

    fn push(self: *Ring, v: f32) void {
        self.data[self.head] = v;
        self.head = (self.head + 1) % hist_n;
        if (self.len < hist_n) self.len += 1;
    }
    fn at(self: *const Ring, i: usize) f32 {
        const start = (self.head + hist_n - self.len) % hist_n;
        return self.data[(start + i) % hist_n];
    }
};


/// The most recent sample, rendered every frame regardless of when it arrived.
///
/// EVERY VRAM number here comes from ONE telemetry event, the whole-card total,
/// our process's footprint, and each component. They stay coherent because the
/// host reads them in one pass; mixing a sampled total with live per-frame
/// component reads is what made "system" bounce during diffusion (see
/// vram_split.zig).
const Sample = struct {
    t: wire.Telemetry = .{},
    has_session: bool = false,
    /// Smoothed residuals: ours-but-untracked, and other processes'.
    parts: vram_split.Parts = .{},

    fn diffTotal(s: *const Sample) u64 {
        return s.t.diff_te + s.t.diff_dit + s.t.diff_latent + s.t.diff_vae;
    }
};
/// One host's bar state. Every field here used to be a module global, which
/// is exactly why there could only ever be one bar: two hosts sampling into
/// one ring draw each other's history. The caller keeps one of these per host
/// and hands the right one in.
pub const View = struct {
    h_vram: Ring = .{},
    h_cpu: Ring = .{},
    h_gpu: Ring = .{},
    /// The most recent sample, rendered every frame regardless of when it came.
    cur: Sample = .{},
    split_smoother: vram_split.Smoother = .{},
    seen_seq: u64 = 0,
};

/// Take the mirror's newest telemetry into `cur` and push the time-series rings.
/// Once per telemetry event, not per frame.
fn sampleInto(v: *View, m: *const mirror.Mirror) void {
    var n: Sample = .{ .t = m.telemetry, .has_session = m.state.llm_resident };
    // Split the card between us and the rest of the system, from this snapshot
    // only (see vram_split.zig for why the residuals are derived this way).
    if (n.t.vram_total > 0) n.parts = v.split_smoother.update(.{
        .total = n.t.vram_total,
        .device_used = n.t.vram_used,
        .proc_used = n.t.vram_proc,
        .ours = n.t.llm_used + n.diffTotal(),
        .loading = m.state.loading,
    });
    // TP_DUMP_VRAM: the meter's attribution, per sample, in MiB. `ovh` is a
    // RESIDUAL (our process's NVML footprint minus what our allocators count),
    // so when it reads high this is the only way to see which side moved: a real
    // untracked allocation grows `proc` alone, while an accounting gap shows up
    // as `proc - ours` widening while `dev_used` holds still.
    if (std.c.getenv("TP_DUMP_VRAM") != null) std.debug.print(
        "[vram-dbg] dev_used={d} proc={d} ours={d} (llm {d} + diff {d}) -> ovh={d} sys={d} loading={d} · split: cpu={d}/{d} host_w={d} MiB (kv {d})\n",
        .{
            n.t.vram_used >> 20,                 n.t.vram_proc >> 20,
            (n.t.llm_used + n.diffTotal()) >> 20, n.t.llm_used >> 20,
            n.diffTotal() >> 20,                  n.parts.overhead >> 20,
            n.parts.system >> 20,                 n.parts.loading >> 20,
            n.t.layers_cpu,                       n.t.layers_cpu + n.t.layers_gpu,
            n.t.llm_host >> 20,                   n.t.ctx_kv >> 20,
        },
    );
    const totf: f32 = if (n.t.vram_total > 0) @floatFromInt(n.t.vram_total) else 0;
    v.h_cpu.push(n.t.cpu);
    v.h_gpu.push(n.t.gpu_util);
    if (totf > 0) v.h_vram.push(@as(f32, @floatFromInt(n.t.vram_used)) / totf * 100.0);
    v.cur = n;
}

/// Draw one host's bar from its mirror. `split`/`limit` are that host's meter
/// handles, which the caller owns; a drag mutates them in place and
/// `acts.on_commit` is where they are persisted and sent. `name` labels the
/// bar and is empty when there is only one host; `index` separates the widget
/// ids of one bar from the next.
pub fn render(v: *View, m: *const mirror.Mirror, name: []const u8, trouble: []const u8, note: []const u8, index: usize, split: *f32, limit: *f32, acts: meter.Actions) void {
    var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .id_extra = index,
        .expand = .horizontal,
        .min_size_content = .{ .h = bar_height },
        .max_size_content = .height(bar_height),
        .color_fill = C.chrome,
        .background = true,
        .border = style.Edge.top,
        .color_border = style.hairline,
        .padding = .{ .x = 14, .y = pad_top, .w = 14, .h = pad_bottom },
    });
    defer bar.deinit();

    if (m.telemetry_seq != v.seen_seq) {
        v.seen_seq = m.telemetry_seq;
        sampleInto(v, m);
    }

    // Whose card this is. Only drawn with more than one host, so the single
    // host case is the bar it has always been. A host that is up but taking no
    // renders says so right under its name: without it a host that quietly
    // never gets a job looks the same as one that is merely idle.
    if (name.len > 0 or note.len > 0) {
        {
            var who = dvui.box(@src(), .{ .dir = .vertical }, .{
                .gravity_y = 0.5,
                .min_size_content = .{ .w = 64 },
                .max_size_content = .width(120),
            });
            defer who.deinit();
            if (name.len > 0) fonts.richLabel(@src(), name, .{ .color_text = C.text_dim });
            if (note.len > 0) {
                fonts.richLabel(@src(), note, .{ .font = F.mono_row, .color_text = C.amber });
            }
        }
        style.vsep(@src());
    }

    // A host that is not answering says so HERE, on its own row. Its last
    // numbers are stale and drawing them would be a lie; the reason is the
    // only useful thing this row can show.
    if (trouble.len > 0) {
        fonts.richLabel(@src(), trouble, .{
            .gravity_y = 0.5,
            .expand = .horizontal,
            .margin = .{ .x = 8 },
            .color_text = C.danger,
        });
        return;
    }

    // Left: three history meters. The number and the history ARE the meter --
    // no clock speeds, no x/y GB, no second progress bar.
    if (v.cur.t.have_gpu) historyMeter(0, "GPU", v.cur.t.gpu_util / 100.0, &v.h_gpu, C.meter_gpu);
    historyMeter(1, "CPU", v.cur.t.cpu / 100.0, &v.h_cpu, C.meter_cpu);
    if (v.cur.t.vram_total > 0) {
        const frac: f32 = @floatCast(@as(f64, @floatFromInt(v.cur.t.vram_used)) / @as(f64, @floatFromInt(v.cur.t.vram_total)));
        // The only place the bar shouts. VRAM is the resource that actually
        // fails, so it is the only one whose color carries a threshold.
        const c = if (frac > 0.95) C.danger else if (frac >= 0.80) C.amber else C.meter_vram;
        historyMeter(2, "VRAM", frac, &v.h_vram, c);
    }
    style.vsep(@src());

    // The rest of the bar is the two-sided VRAM meter (see meter.zig).
    renderMeter(v, &m.state, split, limit, acts);
}

/// Unpack a ring into the flat 0..1 slice `style.historyMeter` wants.
fn historyMeter(id: usize, label: []const u8, value: f32, ring: *const Ring, color: dvui.Color) void {
    var vals: [hist_n]f32 = undefined;
    for (0..ring.len) |i| vals[i] = ring.at(i) / 100.0;
    style.historyMeter(@src(), id, label, value, vals[0..ring.len], color);
}

/// Build the meter model from the sampled device accounting and draw it. The
/// diffusion segments (TE / DiT / latent / VAE) are MEASURED per-tag allocator
/// counters (see pipeline.vramBreakdown); `latent` is the per-image working
/// set (GPU session + activation workspace + preview decode), populated
/// mid-generation.
fn renderMeter(v: *const View, st: *const wire.State, split: *f32, limit: *f32, acts: meter.Actions) void {
    // EVERY byte count below comes from `cur`, one snapshot (see Sample). The
    // whole-card total is the same reading the left VRAM meter shows, so the
    // two always agree.
    //
    // A host that has not said what card it has gets NO track. This used to
    // draw a 24 GiB one, which on a second host with a 4 GB card read as
    // "24 GB free" and was simply false.
    const total: u64 = v.cur.t.vram_total;
    if (total == 0) {
        fonts.richLabel(@src(), "waiting for this host's card", .{
            .gravity_y = 0.5,
            .expand = .horizontal,
            .color_text = C.text_dim,
        });
        return;
    }
    const ctx_b: u64 = v.cur.t.ctx_kv;
    const tf: f32 = @floatFromInt(@max(total, 1));
    // The loaded/unloaded FLAGS (button dimming, drag floors) stay live: they're
    // booleans, not residuals, so they can't reintroduce the bounce, and a
    // half-second lag on an eject button would feel broken.
    const diff_b = v.cur.diffTotal();

    var model: meter.Model = .{
        .total = total,
        // Ours but untracked (CUDA contexts + kernels, library internals, UI
        // textures) vs genuinely other processes'. A model mid-load is NOT here;
        // it goes to `llm_w` below.
        .overhead = v.cur.parts.overhead,
        .system = v.cur.parts.system,
        // A load in flight has weights on the card that nothing tracks yet;
        // they belong to the model, not to `ovh` (see vram_split.Parts).
        .llm_w = (v.cur.t.llm_used -| ctx_b) + v.cur.parts.loading,
        .llm_ctx = ctx_b,
        .ctx_tokens = @intCast(v.cur.t.ctx_tokens),
        // Not on the card: LLM layers computing on the CPU, pipeline weights
        // streaming in from host RAM. Sampled with every other byte count above.
        .llm_host = v.cur.t.llm_host,
        .llm_host_layers = v.cur.t.layers_cpu,
        .llm_layers = v.cur.t.layers_cpu + v.cur.t.layers_gpu,
        .diff_off = v.cur.t.diff_off,
        // MEASURED per-component diffusion breakdown (see pipeline.vramBreakdown).
        .te = v.cur.t.diff_te,
        .dit = v.cur.t.diff_dit,
        .latent = v.cur.t.diff_latent,
        .vae = v.cur.t.diff_vae,
        .split = split,
        .limit = limit,
        // Floors are soft UX guardrails, not hard reservations. The split can't
        // be dragged left of the LLM's incompressible context (KV can't evict);
        // diffusion keeps a small gap when loaded. Both are CAPPED well below the
        // limit so a noisy byte-accounting reading can never invert the drag
        // range and lock the handles (system VRAM is NOT counted here, it lives
        // in the right-hand block against the ceiling, not the LLM's share).
        .floor_llm = std.math.clamp(0.04 + @as(f32, @floatFromInt(ctx_b)) / tf, 0.04, 0.80),
        .floor_diff = if (diff_b > 0) @as(f32, 0.04) else 0.01,
        .llm_loaded = st.llm_resident,
        .diff_loaded = diff_b > 0,
        .llm_armed = st.llm_eject_armed,
        .diff_armed = st.diff_eject_armed,
        .llm_paused = st.llm_paused,
        .diff_paused = st.diff_paused,
    };
    meter.render(&model, acts);
}

test "Ring pushes and reads oldest→newest with wraparound" {
    var r: Ring = .{};
    for (0..hist_n + 3) |i| r.push(@floatFromInt(i));
    try std.testing.expectEqual(@as(usize, hist_n), r.len);
    // After hist_n+3 pushes, oldest is value 3, newest is hist_n+2.
    try std.testing.expectEqual(@as(f32, 3), r.at(0));
    try std.testing.expectEqual(@as(f32, @floatFromInt(hist_n + 2)), r.at(r.len - 1));
}
