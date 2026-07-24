//! Attribution math for the status-bar VRAM meter: turns one coherent snapshot
//! of four readings — whole-card used, OUR process's card footprint, the LLM's
//! tracked bytes, diffusion's tracked bytes — into the meter's two residual
//! blocks: `overhead` (ours, untracked) and `system` (other processes/driver).
//!
//! Pure math, no dvui/NVML: unit-tested via `zig build gui-test`.
//!
//! ## Why this module exists (two measured bugs it fixes)
//!
//! 1. **"system" bounced during diffusion.** The meter computed
//!    `system = device_used − llm − diffusion` from a device total sampled every
//!    500 ms but per-component reads taken LIVE every frame. Every diffusion
//!    alloc/free therefore moved "system" the opposite way at frame rate until
//!    the next device sample landed — worst when VRAM-constrained (the DiT
//!    evicts + reloads weights every step, so the swings are GB-sized), and
//!    invisible during LLM decode (its tracked bytes barely move). The fix is
//!    structural, not cosmetic: `system` is now `device_used − proc_used`, i.e.
//!    derived ONLY from two numbers out of the same driver snapshot, so our own
//!    allocation churn cannot enter it at all.
//!
//! 2. **~1 GiB of OUR VRAM was drawn as "system".** Whatever our allocators
//!    don't count — the CUDA context(s) + JIT'd modules, cuBLASLt/cuDNN
//!    internals, the SDL/GL window and image textures — fell into the residual.
//!    Measured on a 3090 with a 12.3 G LLM + a 5.8 G diffusion model resident:
//!    card 23.4 G used, our process 21.9 G, our tracked components 20.9 G → the
//!    ~1 G difference is ours, not the system's. It gets its own block now.
//!    (Same hole covered the model-LOAD window, where the session isn't
//!    published yet and its uploads counted as "system".)
//!
//! ## Why the residuals are low-passed
//!
//! Both residuals are physically slow-moving (other processes' usage; our
//! context/kernel/texture footprint), while `ours` swings hard mid-generation
//! and the driver's per-process number lags our exact counters by up to a
//! sample. Smoothing keeps that lag out of the display; the leftover skew lands
//! in `free`, which is *defined* as the unaccounted gap and is drawn as bare
//! background.
const std = @import("std");

/// One coherent snapshot. `device_used`/`proc_used` MUST come from the same
/// driver query pass as each other, and `ours` from the same instant — mixing a
/// stale total with live component reads is bug 1 above.
pub const Reading = struct {
    /// Card capacity.
    total: u64,
    /// Whole-card bytes in use (all processes).
    device_used: u64,
    /// Bytes charged to OUR process on this card, or 0 when the driver has no
    /// per-process data (then `overhead` collapses to 0 and `system` degrades to
    /// the old whole-residual meaning).
    proc_used: u64,
    /// Bytes our own allocators track (LLM resident + diffusion resident).
    ours: u64,
};

/// The two residual blocks, in bytes.
pub const Parts = struct {
    /// Ours but untracked: CUDA contexts, JIT'd modules, library internals, UI
    /// textures — plus anything resident before its session is published.
    overhead: u64 = 0,
    /// Not ours: other processes and driver-side allocations.
    system: u64 = 0,
};

/// Exponential smoother over successive readings. One instance per meter.
pub const Smoother = struct {
    /// Smoothed state; null until the first reading seeds it (so the meter never
    /// ramps up from zero on the first frame).
    parts: ?Parts = null,
    /// Weight of a NEW reading. At the 200–500 ms sample cadence 0.25 gives a
    /// ~1.5 s time constant: fast enough to follow a real change (unloading a
    /// model frees its context), slow enough to ignore driver lag.
    alpha: f32 = 0.25,

    pub fn update(self: *Smoother, r: Reading) Parts {
        // Our process can't hold less than we've already tracked: the driver's
        // per-process number lags our exact counters (and is 0 when there's no
        // per-process data at all), which would otherwise wrap `overhead`.
        const proc = @max(r.proc_used, r.ours);
        const inst: Parts = .{
            .overhead = proc - r.ours,
            .system = r.device_used -| proc,
        };
        const sm: Parts = if (self.parts) |p| .{
            .overhead = ema(p.overhead, inst.overhead, self.alpha),
            .system = ema(p.system, inst.system, self.alpha),
        } else inst;
        // Keep the UNCLAMPED smoothed state: the clamp below is a display
        // artifact of a full card, and `inst` itself always fits
        // (ours + overhead + system ≤ device_used ≤ total), so this can't run away.
        self.parts = sm;
        return fit(sm, r);
    }
};

/// Shrink the residuals so `ours + overhead + system` can't exceed the card —
/// otherwise the meter's segments would draw past the end of the bar. Only
/// reachable transiently, when smoothed residuals meet a freshly grown `ours`.
fn fit(p: Parts, r: Reading) Parts {
    const room = r.total -| r.ours;
    var out = p;
    if (out.overhead > room) out.overhead = room;
    if (out.system > room - out.overhead) out.system = room - out.overhead;
    return out;
}

fn ema(prev: u64, new: u64, alpha: f32) u64 {
    const p: f64 = @floatFromInt(prev);
    const n: f64 = @floatFromInt(new);
    const v = p + (n - p) * @as(f64, alpha);
    return if (v <= 0) 0 else @intFromFloat(v);
}

const gib: u64 = 1 << 30;
const mib: u64 = 1 << 20;

test "first reading splits ours / overhead / system with no smoothing ramp" {
    // The measured 3090 case from the module doc.
    var s: Smoother = .{};
    const p = s.update(.{
        .total = 24 * gib,
        .device_used = 23400 * mib,
        .proc_used = 21900 * mib,
        .ours = 20900 * mib,
    });
    try std.testing.expectEqual(@as(u64, 1000 * mib), p.overhead);
    try std.testing.expectEqual(@as(u64, 1500 * mib), p.system);
}

test "a diffusion allocation cannot move system (the reported bounce)" {
    var s: Smoother = .{};
    const steady: Reading = .{
        .total = 24 * gib,
        .device_used = 23400 * mib,
        .proc_used = 21900 * mib,
        .ours = 20900 * mib,
    };
    const before = s.update(steady);
    // Diffusion allocates 500 MiB of step scratch. Our tracked number jumps at
    // once; the driver's two numbers are still last sample's.
    var churned = steady;
    churned.ours += 500 * mib;
    const after = s.update(churned);
    errdefer std.debug.print("system {d} MiB -> {d} MiB\n", .{ before.system / mib, after.system / mib });
    try std.testing.expectEqual(before.system, after.system); // EXACTLY unmoved
    // The lag lands in `overhead` (damped) — never in `system`.
    try std.testing.expect(after.overhead < before.overhead);
}

test "with no per-process data, system degrades to the whole residual" {
    var s: Smoother = .{};
    const p = s.update(.{
        .total = 24 * gib,
        .device_used = 20 * gib,
        .proc_used = 0, // NVML per-process unavailable
        .ours = 18 * gib,
    });
    try std.testing.expectEqual(@as(u64, 0), p.overhead);
    try std.testing.expectEqual(@as(u64, 2 * gib), p.system);
}

test "residuals never overflow the card" {
    var s: Smoother = .{};
    // Seed a big overhead + system, then have `ours` swallow nearly the card.
    _ = s.update(.{ .total = 24 * gib, .device_used = 23 * gib, .proc_used = 21 * gib, .ours = 18 * gib });
    const p = s.update(.{ .total = 24 * gib, .device_used = 23 * gib, .proc_used = 21 * gib, .ours = 23 * gib + 512 * mib });
    try std.testing.expect(23 * gib + 512 * mib + p.overhead + p.system <= 24 * gib);
}

test "smoothing converges on a real change within a couple seconds" {
    var s: Smoother = .{};
    _ = s.update(.{ .total = 24 * gib, .device_used = 20 * gib, .proc_used = 18 * gib, .ours = 17 * gib });
    // Another process grabs 2 GiB and stays. At 0.25/sample (≈2–5 Hz) the meter
    // should be within 5% after 10 samples (≈2–5 s).
    const changed: Reading = .{ .total = 24 * gib, .device_used = 22 * gib, .proc_used = 18 * gib, .ours = 17 * gib };
    var p: Parts = .{};
    for (0..10) |_| p = s.update(changed);
    const want: f64 = @floatFromInt(4 * gib); // 22 − 18
    const got: f64 = @floatFromInt(p.system);
    errdefer std.debug.print("system {d:.2} GiB, want {d:.2}\n", .{ got / 1073741824.0, want / 1073741824.0 });
    try std.testing.expect(@abs(got - want) / want < 0.05);
}

test "fit shrinks system before overhead" {
    // Overhead is ours and measured; system is the guess — so the guess yields first.
    const p = fit(
        .{ .overhead = 1 * gib, .system = 2 * gib },
        .{ .total = 24 * gib, .device_used = 23 * gib, .proc_used = 0, .ours = 23 * gib },
    );
    try std.testing.expectEqual(@as(u64, 1 * gib), p.overhead);
    try std.testing.expectEqual(@as(u64, 0), p.system);
}
