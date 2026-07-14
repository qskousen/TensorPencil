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
const config = @import("config.zig");
const sysmon = @import("sysmon.zig");

/// Fixed bar height (logical px), reserved by the caller.
pub const bar_height: f32 = 32;

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
    gpu_util: f32 = 0,
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
    priority: config.Priority = .chat,
};
var cur: Sample = .{};

/// Release the NVML handle at process exit.
pub fn deinit() void {
    if (nvml) |*n| n.close();
    nvml = null;
}

/// Take a fresh sample of every meter into `cur` and push the time-series rings.
/// Called on the timer cadence, not per frame.
fn sampleInto(s: ?*chat.Session) void {
    var n: Sample = .{};
    n.cpu = cpu_meter.sample();
    if (nvml) |*nv| if (nv.query()) |g| {
        n.gpu_util = @floatFromInt(g.util);
        n.vram_used = g.mem_used;
        n.vram_total = g.mem_total;
        n.have_gpu = true;
    };
    if (s) |sess| {
        n.has_session = true;
        n.llm_used = sess.be.deviceUsed();
        n.limit = sess.vram_limit; // the CONFIGURED cap, not the LLM's internal offload budget
        n.priority = sess.vram_priority;
        n.ctx_tokens = sess.ctxTokens();
        n.ctx_kv = sess.ctxKvBytes();
        const res = sess.llmResidency();
        n.layers_gpu = res.gpu;
        n.layers_cpu = res.cpu;
        n.diffusing = sess.diffusing();
        if (n.vram_total == 0) {
            const mi = sess.be.ctx.memGetInfo();
            n.vram_total = mi.total;
            n.vram_used = mi.total -| mi.free;
        }
        // Accurate resident diffusion VRAM (the diffusion backend's own
        // device_used), not an NVML-minus-LLM proxy that would also count the
        // desktop's VRAM.
        n.diff_used = sess.diffVramBytes();
    }
    const totf: f32 = if (n.vram_total > 0) @floatFromInt(n.vram_total) else 0;
    h_cpu.push(n.cpu);
    h_gpu.push(n.gpu_util);
    if (totf > 0) h_vram.push(@as(f32, @floatFromInt(n.vram_used)) / totf * 100.0);
    cur = n;
}

/// Draw the bar. `s` is the live session (null before a model loads — the bar
/// still shows CPU/GPU/total VRAM).
pub fn render(s: ?*chat.Session) void {
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
        .padding = .{ .x = 10, .y = 4, .w = 10 },
    });
    defer bar.deinit();

    // Time-based sampling: resample only when the timer fires, then reschedule
    // (which also sets the main loop's wait so idle frames still tick).
    if (dvui.timerDoneOrNone(bar.data().id)) {
        sampleInto(s);
        dvui.timer(bar.data().id, sample_interval_us);
    }

    var buf: [128]u8 = undefined;

    // Time-series meters (fixed-width labels so 2→3 digit changes don't shift).
    if (cur.have_gpu) {
        meterLabel(0, 46, std.fmt.bufPrint(&buf, "GPU {d:.0}%", .{cur.gpu_util}) catch "GPU");
        spark(1, &h_gpu, .{ .r = 120, .g = 200, .b = 120, .a = 235 });
    }
    if (cur.vram_total > 0) {
        meterLabel(2, 104, std.fmt.bufPrint(&buf, "VRAM {d:.1}/{d:.0} GB", .{
            @as(f64, @floatFromInt(cur.vram_used)) / gib, @as(f64, @floatFromInt(cur.vram_total)) / gib,
        }) catch "VRAM");
        spark(3, &h_vram, .{ .r = 110, .g = 160, .b = 230, .a = 235 });
    }
    meterLabel(6, 46, std.fmt.bufPrint(&buf, "CPU {d:.0}%", .{cur.cpu}) catch "CPU");
    spark(7, &h_cpu, .{ .r = 230, .g = 170, .b = 110, .a = 235 });

    // Session readouts (text; no sparkline — they change slowly / on generation).
    if (cur.has_session) {
        sep(10);
        // LLM footprint + layer split (GPU/total; cpu = total − gpu).
        meterLabel(11, 118, std.fmt.bufPrint(&buf, "LLM {d:.1}G · {d}/{d} gpu", .{
            @as(f64, @floatFromInt(cur.llm_used)) / gib, cur.layers_gpu, cur.layers_gpu + cur.layers_cpu,
        }) catch "LLM");
        // Context length + its KV footprint.
        var tb: [16]u8 = undefined;
        meterLabel(12, 128, std.fmt.bufPrint(&buf, "ctx {s} tok · {d:.2}G KV", .{
            fmtTokens(&tb, cur.ctx_tokens), @as(f64, @floatFromInt(cur.ctx_kv)) / gib,
        }) catch "ctx");
        // Diffusion residency (accurate: the diffusion backend's device_used),
        // shown while the image model is loaded.
        if (cur.diff_used > 0) {
            meterLabel(13, 88, std.fmt.bufPrint(&buf, "diff {d:.1}G", .{@as(f64, @floatFromInt(cur.diff_used)) / gib}) catch "diff");
        }
    }

    // Limit + priority, pushed to the right.
    {
        var spb = dvui.box(@src(), .{}, .{ .expand = .horizontal });
        spb.deinit();
    }
    if (cur.has_session) {
        const pri = @tagName(cur.priority);
        const txt = if (cur.limit > 0)
            std.fmt.bufPrint(&buf, "limit {d:.1} GB · {s}", .{ @as(f64, @floatFromInt(cur.limit)) / gib, pri }) catch ""
        else
            std.fmt.bufPrint(&buf, "limit: auto · {s}", .{pri}) catch "";
        dvui.label(@src(), "{s}", .{txt}, .{ .gravity_y = 0.5, .font = dvui.Font.theme(.body).withSize(12) });
    }
}

/// Format a token count compactly ("823", "3.2k", "128k").
fn fmtTokens(buf: []u8, n: usize) []const u8 {
    if (n >= 1000) return std.fmt.bufPrint(buf, "{d:.1}k", .{@as(f64, @floatFromInt(n)) / 1000.0}) catch "?";
    return std.fmt.bufPrint(buf, "{d}", .{n}) catch "?";
}

/// A fixed-width label for one meter (width reserved so digit-count changes
/// don't reflow the row).
fn meterLabel(id: usize, width: f32, text: []const u8) void {
    dvui.label(@src(), "{s}", .{text}, .{
        .id_extra = id,
        .gravity_y = 0.5,
        .gravity_x = 0.0,
        .font = dvui.Font.theme(.body).withSize(12),
        .min_size_content = .{ .w = width },
        .margin = .{ .x = 2 },
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
        .min_size_content = .{ .w = 56, .h = 18 },
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
