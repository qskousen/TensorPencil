//! The tp-gui status bar: three history meters (GPU / CPU / VRAM) on the left,
//! then the two-sided VRAM meter (see meter.zig) filling the rest. Sampled from
//! `sysmon` (CPU via /proc/stat, GPU via NVML) and the engines' own device
//! accounting.
//!
//! It spans the WHOLE window, under all three columns, and never migrates into
//! a panel: this telemetry is a property of the app, not of any one view. Its
//! fixed height (`bar_height`) is subtracted from the body band by the caller.
//!
//! This module owns SAMPLING and model-building; meter.zig owns the track's
//! drawing and interaction, and style.zig owns the history meter, which the
//! `ui-probe` harness also draws.
const std = @import("std");
const dvui = @import("dvui");
const chat = @import("chat.zig");
const diffuser = @import("diffuser.zig");
const sysmon = @import("sysmon.zig");
const meter = @import("meter.zig");
const vram_split = @import("vram_split.zig");
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;
const F = style.F;

/// Fixed bar height (logical px), reserved by the caller. Spans the WHOLE
/// window, under all three columns: telemetry is a property of the app, not of
/// the queue rail it used to sit in.
pub const bar_height: f32 = style.Layout.status_h;

/// Samples kept per sparkline. At the ~2 Hz cadence below this is a ~24 s
/// window, which is long enough to show a generation start and short enough
/// that the graph still moves.
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

var cpu_meter: sysmon.CpuMeter = .{};
// NVML now lives in sysmon as a process-wide singleton: the VRAM budget reads the
// same handle, so the meter and the policy can never disagree about the card.

var h_vram: Ring = .{};
var h_cpu: Ring = .{};
var h_gpu: Ring = .{};

const gib: f64 = 1 << 30;
/// Sampling cadence (µs), decoupled from the frame rate. A dvui timer fires on
/// this interval, which also wakes the (event-driven) main loop when idle so the
/// meters keep advancing even with no UI activity. Faster while a worker is busy,
/// since that's when the segments actually move.
const sample_interval_us: i32 = 500_000;
const sample_interval_busy_us: i32 = 200_000;

/// The most recent sample, rendered every frame regardless of when it was taken.
///
/// EVERY VRAM number here is read in one pass in `sampleInto`, the whole-card
/// total, our process's footprint, and each component. They must stay coherent:
/// mixing a timer-sampled total with live per-frame component reads is what made
/// "system" bounce during diffusion (see vram_split.zig).
const Sample = struct {
    cpu: f32 = 0,
    cpu_mhz: f32 = 0,
    gpu_util: f32 = 0,
    gpu_mhz: u32 = 0,
    vram_used: u64 = 0,
    vram_total: u64 = 0,
    /// Our process's whole card footprint (NVML per-process), 0 = unavailable.
    vram_proc: u64 = 0,
    have_gpu: bool = false,
    has_session: bool = false,
    llm_used: u64 = 0,
    ctx_tokens: usize = 0,
    ctx_kv: u64 = 0,
    layers_gpu: usize = 0,
    layers_cpu: usize = 0,
    diffusing: bool = false,
    /// Resident diffusion VRAM, per component (sums to the backend's device_used).
    diff: diffuser.VramBreakdown = .{},
    limit: u64 = 0,
    /// Smoothed residuals: ours-but-untracked, and other processes'.
    parts: vram_split.Parts = .{},
};
var cur: Sample = .{};
var split_smoother: vram_split.Smoother = .{};

/// Release the NVML handle at process exit.
pub fn deinit() void {
    sysmon.nvmlClose();
}

/// Take a fresh sample of every meter into `cur` and push the time-series rings.
/// Called on the timer cadence, not per frame.
fn sampleInto(s: ?*chat.Session, loading: bool, diff_busy: bool, diff: diffuser.VramBreakdown) void {
    var n: Sample = .{};
    n.cpu = cpu_meter.sample();
    n.cpu_mhz = sysmon.cpuFreqMhz();
    if (sysmon.nvml()) |nv| {
        if (nv.query()) |g| {
            n.gpu_util = @floatFromInt(g.util);
            n.gpu_mhz = g.clock_mhz;
            n.vram_used = g.mem_used;
            n.vram_total = g.mem_total;
            n.have_gpu = true;
        }
        // Same pass as the whole-card numbers above, the meter subtracts the two.
        n.vram_proc = nv.selfUsed() orelse 0;
    }
    // Diffusion state comes from the app-level engine (not the session).
    n.diffusing = diff_busy;
    n.diff = diff; // MEASURED per-component resident diffusion VRAM
    if (s) |sess| {
        n.has_session = true;
        n.llm_used = sess.be.deviceUsed();
        n.limit = sess.vram_limit; // the CONFIGURED cap, not the LLM's internal offload budget
        n.ctx_tokens = sess.ctxTokens();
        n.ctx_kv = sess.ctxKvBytes();
        const res = sess.llmResidency();
        n.layers_gpu = res.gpu;
        n.layers_cpu = res.cpu;
        if (n.vram_total == 0) {
            const mi = sess.be.ctx.memGetInfo();
            n.vram_total = mi.total;
            n.vram_used = mi.total -| mi.free;
        }
    }
    // Split the card between us and the rest of the system, from this snapshot
    // only (see vram_split.zig for why the residuals are derived this way).
    if (n.vram_total > 0) n.parts = split_smoother.update(.{
        .total = n.vram_total,
        .device_used = n.vram_used,
        .proc_used = n.vram_proc,
        .ours = n.llm_used + n.diff.total(),
        .loading = loading,
    });
    // TP_DUMP_VRAM: the meter's attribution, per sample, in MiB. `ovh` is a
    // RESIDUAL (our process's NVML footprint minus what our allocators count),
    // so when it reads high this is the only way to see which side moved: a real
    // untracked allocation grows `proc` alone, while an accounting gap shows up
    // as `proc - ours` widening while `dev_used` holds still.
    if (std.c.getenv("TP_DUMP_VRAM") != null) std.debug.print(
        "[vram-dbg] dev_used={d} proc={d} ours={d} (llm {d} + diff {d}) -> ovh={d} sys={d} loading={d}\n",
        .{
            n.vram_used >> 20,                   n.vram_proc >> 20,
            (n.llm_used + n.diff.total()) >> 20, n.llm_used >> 20,
            n.diff.total() >> 20,                n.parts.overhead >> 20,
            n.parts.system >> 20,                n.parts.loading >> 20,
        },
    );
    const totf: f32 = if (n.vram_total > 0) @floatFromInt(n.vram_total) else 0;
    h_cpu.push(n.cpu);
    h_gpu.push(n.gpu_util);
    if (totf > 0) h_vram.push(@as(f32, @floatFromInt(n.vram_used)) / totf * 100.0);
    cur = n;
}

/// Draw the bar. `s` is the live session (null before a model loads, the bar
/// still shows CPU/GPU/total VRAM). `diff_busy`/`diff_used` come from the
/// app-level diffusion engine.
pub fn render(s: ?*chat.Session, loading: bool, diff_busy: bool, diff: diffuser.VramBreakdown, split: *f32, limit: *f32, llm_armed: bool, diff_armed: bool, llm_paused: bool, diff_paused: bool, acts: meter.Actions) void {
    _ = sysmon.nvml(); // opens on first use

    var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = bar_height },
        .max_size_content = .height(bar_height),
        .color_fill = C.chrome,
        .background = true,
        .border = style.Edge.top,
        .color_border = style.hairline,
        .padding = .{ .x = 14, .y = 9, .w = 14, .h = 10 },
    });
    defer bar.deinit();

    // Time-based sampling: EVERY meter number (card total, our process
    // footprint, per-component residency) is taken here, in one pass, and the
    // meter renders that snapshot, never a mix of sampled and live reads.
    if (dvui.timerDoneOrNone(bar.data().id)) {
        sampleInto(s, loading, diff_busy, diff);
        dvui.timer(bar.data().id, if (diff_busy) sample_interval_busy_us else sample_interval_us);
    }

    // Left: three history meters. The number and the history ARE the meter --
    // no clock speeds, no x/y GB, no second progress bar.
    if (cur.have_gpu) historyMeter(0, "GPU", cur.gpu_util / 100.0, &h_gpu, C.meter_gpu);
    historyMeter(1, "CPU", cur.cpu / 100.0, &h_cpu, C.meter_cpu);
    if (cur.vram_total > 0) {
        const frac: f32 = @floatCast(@as(f64, @floatFromInt(cur.vram_used)) / @as(f64, @floatFromInt(cur.vram_total)));
        // The only place the bar shouts. VRAM is the resource that actually
        // fails, so it is the only one whose color carries a threshold.
        const c = if (frac > 0.95) C.danger else if (frac >= 0.80) C.amber else C.meter_vram;
        historyMeter(2, "VRAM", frac, &h_vram, c);
    }
    style.vsep(@src());

    // The rest of the bar is the two-sided VRAM meter (see meter.zig).
    renderMeter(s, diff, split, limit, llm_armed, diff_armed, llm_paused, diff_paused, acts);
}

/// Unpack a ring into the flat 0..1 slice `style.historyMeter` wants.
fn historyMeter(id: usize, label: []const u8, value: f32, ring: *const Ring, color: dvui.Color) void {
    var vals: [hist_n]f32 = undefined;
    for (0..ring.len) |i| vals[i] = ring.at(i) / 100.0;
    style.historyMeter(@src(), id, label, value, vals[0..ring.len], color);
}

/// Build the meter model from live device accounting and draw it. The diffusion
/// segments (TE / DiT / latent / VAE) are MEASURED per-tag allocator counters
/// (see pipeline.vramBreakdown); `latent` is the per-image working set (GPU
/// session + activation workspace + preview decode), populated mid-generation.
fn renderMeter(s: ?*chat.Session, diff: diffuser.VramBreakdown, split: *f32, limit: *f32, llm_armed: bool, diff_armed: bool, llm_paused: bool, diff_paused: bool, acts: meter.Actions) void {
    // EVERY byte count below comes from `cur`, one snapshot (see Sample). The
    // whole-card total is the same NVML reading the left VRAM meter shows, so the
    // two always agree; with no NVML at all we fall back to the LLM context's
    // cuMemGetInfo, then to a sane default.
    var total: u64 = cur.vram_total;
    if (total == 0) total = 24 << 30;
    const ctx_b: u64 = cur.ctx_kv;
    const tf: f32 = @floatFromInt(@max(total, 1));
    // The loaded/unloaded FLAGS (button dimming, drag floors) stay live: they're
    // booleans, not residuals, so they can't reintroduce the bounce, and a
    // half-second lag on an eject button would feel broken.
    const diff_b = diff.total();

    var model: meter.Model = .{
        .total = total,
        // Ours but untracked (CUDA contexts + kernels, library internals, UI
        // textures) vs genuinely other processes'. A model mid-load is NOT here;
        // it goes to `llm_w` below.
        .overhead = cur.parts.overhead,
        .system = cur.parts.system,
        // A load in flight has weights on the card that nothing tracks yet;
        // they belong to the model, not to `ovh` (see vram_split.Parts).
        .llm_w = (cur.llm_used -| ctx_b) + cur.parts.loading,
        .llm_ctx = ctx_b,
        .ctx_tokens = cur.ctx_tokens,
        // MEASURED per-component diffusion breakdown (see pipeline.vramBreakdown).
        .te = cur.diff.te,
        .dit = cur.diff.dit,
        .latent = cur.diff.latent,
        .vae = cur.diff.vae,
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
        .llm_loaded = s != null,
        .diff_loaded = diff_b > 0,
        .llm_armed = llm_armed,
        .diff_armed = diff_armed,
        .llm_paused = llm_paused,
        .diff_paused = diff_paused,
    };
    meter.render(&model, acts);
}

/// Format a token count compactly ("823", "3.2k", "128k").
fn fmtTokens(buf: []u8, n: usize) []const u8 {
    if (n >= 1000) return std.fmt.bufPrint(buf, "{d:.1}k", .{@as(f64, @floatFromInt(n)) / 1000.0}) catch "?";
    return std.fmt.bufPrint(buf, "{d}", .{n}) catch "?";
}

test "Ring pushes and reads oldest→newest with wraparound" {
    var r: Ring = .{};
    for (0..hist_n + 3) |i| r.push(@floatFromInt(i));
    try std.testing.expectEqual(@as(usize, hist_n), r.len);
    // After hist_n+3 pushes, oldest is value 3, newest is hist_n+2.
    try std.testing.expectEqual(@as(f32, 3), r.at(0));
    try std.testing.expectEqual(@as(f32, @floatFromInt(hist_n + 2)), r.at(r.len - 1));
}
