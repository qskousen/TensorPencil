//! The tp-gui bottom status bar (GUI_VRAM.md, Phase 6): live VRAM (total + the
//! chat model's resident footprint), the active VRAM limit, and CPU / GPU
//! utilization — each with a small rolling sparkline. Sampled from `sysmon`
//! (CPU via /proc/stat, GPU via NVML) and the LLM backend's device accounting.
//!
//! Rendered as the last child of the chat frame's root vbox; its fixed height
//! (`bar_height`) is subtracted from the message-list height so it never
//! overlaps the list.
const std = @import("std");
const dvui = @import("dvui");
const chat = @import("chat.zig");
const diffuser = @import("diffuser.zig");
const sysmon = @import("sysmon.zig");
const meter = @import("meter.zig");

/// Fixed bar height (logical px), reserved by the caller. The VRAM meter row
/// (top) plus the readout row (bottom).
pub const bar_height: f32 = 52;

/// Number of samples kept per sparkline.
const hist_n = 48;

/// A small fixed-capacity rolling history, iterated oldest→newest via `at`.
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
var nvml: ?sysmon.Nvml = null;
var nvml_tried: bool = false;

var h_vram: Ring = .{};
var h_cpu: Ring = .{};
var h_gpu: Ring = .{};

const gib: f64 = 1 << 30;
/// Sampling cadence (µs) — decoupled from the frame rate. A dvui timer fires on
/// this interval, which also wakes the (event-driven) main loop when idle so the
/// meters keep advancing even with no UI activity.
const sample_interval_us: i32 = 500_000;

/// The most recent sample, rendered every frame regardless of when it was taken.
const Sample = struct {
    cpu: f32 = 0,
    cpu_mhz: f32 = 0,
    gpu_util: f32 = 0,
    gpu_mhz: u32 = 0,
    vram_used: u64 = 0,
    vram_total: u64 = 0,
    have_gpu: bool = false,
    has_session: bool = false,
    llm_used: u64 = 0,
    ctx_tokens: usize = 0,
    ctx_kv: u64 = 0,
    layers_gpu: usize = 0,
    layers_cpu: usize = 0,
    diffusing: bool = false,
    diff_used: u64 = 0, // resident diffusion VRAM (the diffusion backend's device_used)
    limit: u64 = 0,
};
var cur: Sample = .{};

/// Release the NVML handle at process exit.
pub fn deinit() void {
    if (nvml) |*n| n.close();
    nvml = null;
}

/// Take a fresh sample of every meter into `cur` and push the time-series rings.
/// Called on the timer cadence, not per frame.
fn sampleInto(s: ?*chat.Session, diff_busy: bool, diff_used: u64) void {
    var n: Sample = .{};
    n.cpu = cpu_meter.sample();
    n.cpu_mhz = sysmon.cpuFreqMhz();
    if (nvml) |*nv| if (nv.query()) |g| {
        n.gpu_util = @floatFromInt(g.util);
        n.gpu_mhz = g.clock_mhz;
        n.vram_used = g.mem_used;
        n.vram_total = g.mem_total;
        n.have_gpu = true;
    };
    // Diffusion state comes from the app-level engine (not the session).
    n.diffusing = diff_busy;
    n.diff_used = diff_used; // accurate resident diffusion VRAM (backend device_used)
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
    const totf: f32 = if (n.vram_total > 0) @floatFromInt(n.vram_total) else 0;
    h_cpu.push(n.cpu);
    h_gpu.push(n.gpu_util);
    if (totf > 0) h_vram.push(@as(f32, @floatFromInt(n.vram_used)) / totf * 100.0);
    cur = n;
}

/// Draw the bar. `s` is the live session (null before a model loads — the bar
/// still shows CPU/GPU/total VRAM). `diff_busy`/`diff_used` come from the
/// app-level diffusion engine.
pub fn render(s: ?*chat.Session, diff_busy: bool, diff: diffuser.VramBreakdown, split: *f32, limit: *f32, llm_armed: bool, diff_armed: bool, acts: meter.Actions) void {
    if (!nvml_tried) {
        nvml = sysmon.Nvml.open();
        nvml_tried = true;
    }

    const theme = dvui.themeGet();
    var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = bar_height },
        .color_fill = theme.fill.lerp(theme.text, 0.06),
        .background = true,
        .padding = .{ .x = 8, .y = 3, .w = 8 },
    });
    defer bar.deinit();

    // Time-based sampling (drives the meter's total/system baseline when no LLM
    // session is loaded to read the card from). Resample on the timer.
    if (dvui.timerDoneOrNone(bar.data().id)) {
        sampleInto(s, diff_busy, diff.total());
        dvui.timer(bar.data().id, sample_interval_us);
    }

    var top: [24]u8 = undefined;
    var bot: [24]u8 = undefined;
    // Left: CPU/GPU/VRAM — util% on top, clock/size beneath (a 2-row stack that
    // matches the meter's number columns), plus a trend sparkline.
    if (cur.have_gpu) {
        metric(0, 72, &h_gpu, .{ .r = 120, .g = 200, .b = 120, .a = 235 }, std.fmt.bufPrint(&top, "GPU {d:.0}%", .{cur.gpu_util}) catch "GPU", std.fmt.bufPrint(&bot, "{d:.2} GHz", .{@as(f64, @floatFromInt(cur.gpu_mhz)) / 1000.0}) catch "");
    }
    metric(6, 72, &h_cpu, .{ .r = 230, .g = 170, .b = 110, .a = 235 }, std.fmt.bufPrint(&top, "CPU {d:.0}%", .{cur.cpu}) catch "CPU", std.fmt.bufPrint(&bot, "{d:.2} GHz", .{cur.cpu_mhz / 1000.0}) catch "");
    if (cur.vram_total > 0) {
        metric(2, 88, &h_vram, .{ .r = 110, .g = 160, .b = 230, .a = 235 }, std.fmt.bufPrint(&top, "VRAM {d:.0}%", .{@as(f64, @floatFromInt(cur.vram_used)) / @as(f64, @floatFromInt(cur.vram_total)) * 100}) catch "VRAM", std.fmt.bufPrint(&bot, "{d:.1}/{d:.0} GB", .{ @as(f64, @floatFromInt(cur.vram_used)) / gib, @as(f64, @floatFromInt(cur.vram_total)) / gib }) catch "");
    }
    sep(10);

    // The rest of the bar is the live VRAM meter (see meter.zig).
    renderMeter(s, diff, split, limit, llm_armed, diff_armed, acts);
}

/// Build the meter model from live device accounting and draw it. The diffusion
/// segments (TE / DiT / latent / VAE) are MEASURED per-tag allocator counters
/// (see pipeline.vramBreakdown); `latent` is the per-image working set (GPU
/// session + activation workspace + preview decode), populated mid-generation.
fn renderMeter(s: ?*chat.Session, diff: diffuser.VramBreakdown, split: *f32, limit: *f32, llm_armed: bool, diff_armed: bool, acts: meter.Actions) void {
    // Whole-card totals come from the SAME source as the left VRAM meter — the
    // NVML sample — so the two always agree. (Reading the LLM context's own
    // cuMemGetInfo here made the segments misbehave whenever a session was
    // resident; NVML is the authoritative whole-device view.) Fall back to the
    // context query, then a sane default, only when NVML is unavailable.
    var total: u64 = cur.vram_total;
    var used_all: u64 = cur.vram_used;
    if (total == 0) {
        if (s) |ss| {
            const mi = ss.be.ctx.memGetInfo();
            total = mi.total;
            used_all = mi.total -| mi.free;
        } else total = 24 << 30;
    }
    const llm_used: u64 = if (s) |ss| ss.be.deviceUsed() else 0;
    const ctx_b: u64 = if (s) |ss| ss.ctxKvBytes() else 0;
    const diff_b = diff.total();
    const system = used_all -| llm_used -| diff_b;
    const tf: f32 = @floatFromInt(@max(total, 1));

    var model: meter.Model = .{
        .total = total,
        .system = system,
        .llm_w = llm_used -| ctx_b,
        .llm_ctx = ctx_b,
        // MEASURED per-component diffusion breakdown (see pipeline.vramBreakdown).
        .te = diff.te,
        .dit = diff.dit,
        .latent = diff.latent,
        .vae = diff.vae,
        .split = split,
        .limit = limit,
        // Floors are soft UX guardrails, not hard reservations. The split can't
        // be dragged left of the LLM's incompressible context (KV can't evict);
        // diffusion keeps a small gap when loaded. Both are CAPPED well below the
        // limit so a noisy byte-accounting reading can never invert the drag
        // range and lock the handles (system VRAM is NOT counted here — it lives
        // in the right-hand block against the ceiling, not the LLM's share).
        .floor_llm = std.math.clamp(0.04 + @as(f32, @floatFromInt(ctx_b)) / tf, 0.04, 0.80),
        .floor_diff = if (diff_b > 0) @as(f32, 0.04) else 0.01,
        .llm_loaded = s != null,
        .diff_loaded = diff_b > 0,
        .llm_armed = llm_armed,
        .diff_armed = diff_armed,
    };
    meter.render(&model, acts);
}

/// Format a token count compactly ("823", "3.2k", "128k").
fn fmtTokens(buf: []u8, n: usize) []const u8 {
    if (n >= 1000) return std.fmt.bufPrint(buf, "{d:.1}k", .{@as(f64, @floatFromInt(n)) / 1000.0}) catch "?";
    return std.fmt.bufPrint(buf, "{d}", .{n}) catch "?";
}

/// One left-hand metric: a 2-row text stack (top = util%, bottom = clock/size)
/// beside its trend sparkline. `wtext` fixes the text column width (pick it wide
/// enough for the max string) so digit-count changes never shift the bar.
fn metric(id: usize, wtext: f32, ring: *const Ring, color: dvui.Color, top: []const u8, bottom: []const u8) void {
    var box = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = id, .gravity_y = 0.5, .margin = .{ .w = 4 } });
    defer box.deinit();
    {
        var txt = dvui.box(@src(), .{ .dir = .vertical }, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = wtext },
            .max_size_content = .width(wtext),
        });
        defer txt.deinit();
        stackLine(0, top, false);
        stackLine(1, bottom, true);
    }
    spark(id, ring, color);
}

fn stackLine(id: usize, text: []const u8, dim: bool) void {
    const th = dvui.themeGet();
    dvui.label(@src(), "{s}", .{text}, .{
        .id_extra = id,
        .gravity_x = 0.0,
        .font = dvui.Font.theme(.body).withSize(9),
        .padding = .{},
        .margin = .{ .y = 1 },
        .color_text = if (dim) th.fill.lerp(th.text, 0.45) else null,
    });
}

/// A thin vertical separator between meter groups.
fn sep(id: usize) void {
    dvui.label(@src(), "│", .{}, .{ .id_extra = id, .gravity_y = 0.5, .font = dvui.Font.theme(.body).withSize(12), .margin = .{ .x = 4 } });
}

/// Draw a rolling sparkline (bars scaled to a 0..100 range) inside a small box.
fn spark(id: usize, ring: *const Ring, color: dvui.Color) void {
    var bx = dvui.box(@src(), .{}, .{
        .id_extra = id,
        .min_size_content = .{ .w = 56, .h = 30 },
        .gravity_y = 0.5,
        .margin = .{ .x = 2, .w = 10 },
    });
    defer bx.deinit();
    const r = bx.data().rectScale().r;
    if (ring.len == 0 or r.w <= 0 or r.h <= 0) return;
    const bw = r.w / @as(f32, @floatFromInt(hist_n));
    var i: usize = 0;
    while (i < ring.len) : (i += 1) {
        const frac = std.math.clamp(ring.at(i) / 100.0, 0.0, 1.0);
        const bh = @max(1.0, r.h * frac);
        const bar_r: dvui.Rect.Physical = .{
            .x = r.x + @as(f32, @floatFromInt(i)) * bw,
            .y = r.y + r.h - bh,
            .w = @max(1.0, bw - 1.0),
            .h = bh,
        };
        bar_r.fill(.{}, .{ .color = color });
    }
}

test "Ring pushes and reads oldest→newest with wraparound" {
    var r: Ring = .{};
    for (0..hist_n + 3) |i| r.push(@floatFromInt(i));
    try std.testing.expectEqual(@as(usize, hist_n), r.len);
    // After hist_n+3 pushes, oldest is value 3, newest is hist_n+2.
    try std.testing.expectEqual(@as(f32, 3), r.at(0));
    try std.testing.expectEqual(@as(f32, @floatFromInt(hist_n + 2)), r.at(r.len - 1));
}
