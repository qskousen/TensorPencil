//! Attribution math for the status-bar VRAM meter: turns one coherent snapshot of four
//! readings (whole-card used, OUR process's card footprint, the LLM's tracked bytes,
//! diffusion's tracked bytes) into the meter's two residual blocks, `overhead` (ours,
//! untracked) and `system` (other processes and the driver).
//!
//! Pure math, no dvui or NVML: unit-tested via `zig build gui-test`.
//!
//! `system` is `device_used - proc_used`, derived only from two numbers out of the SAME
//! driver snapshot. Computing it as `device_used - llm - diffusion` instead mixes a
//! device total sampled every 500 ms with per-component reads taken live every frame, so
//! every diffusion alloc/free moves it the opposite way at frame rate, in GB-sized swings
//! when the DiT is evicting and reloading weights each step.
//!
//! `overhead` exists because a large part of our own VRAM is untracked: the CUDA contexts
//! and JIT'd modules, cuBLASLt/cuDNN internals, the SDL/GL window and image textures.
//! Measured on a 3090 with a 12.3 G LLM and a 5.8 G diffusion model resident, the card
//! reads 23.4 G used, our process 21.9 G and our tracked components 20.9 G, so ~1 G is
//! ours rather than the system's.
//!
//! `loading` splits the model-load window back out of that hole. A load's uploads are on
//! the card with nothing tracking them yet, and left in `overhead` they read as gigabytes
//! of non-model VRAM, which is the one thing that segment is meant to rule out.
//!
//! Both residuals are low-passed because they are physically slow-moving, while `ours`
//! swings hard mid-generation and the driver's per-process number lags our own counters
//! by up to a sample. Smoothing keeps that lag out of the display; the leftover skew
//! lands in `free`, which is defined as the unaccounted gap.
const std = @import("std");

/// One coherent snapshot. `device_used`/`proc_used` MUST come from the same
/// driver query pass as each other, and `ours` from the same instant, mixing a
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
    /// A model load is in flight. Its weights are already on the card but not in
    /// `ours` yet: the loader builds the session on its own thread and publishes
    /// it only once it is whole, so nothing tracks those bytes meanwhile.
    loading: bool = false,
};

/// The residual blocks, in bytes.
pub const Parts = struct {
    /// Ours but untracked: CUDA contexts, JIT'd modules, library internals, UI
    /// textures.
    overhead: u64 = 0,
    /// Not ours: other processes and driver-side allocations.
    system: u64 = 0,
    /// Untracked growth during a load, i.e. weights being uploaded. Reported
    /// separately so the meter can draw it as the model it is becoming rather
    /// than as `overhead`, which means non-model VRAM of ours (the CUDA context,
    /// SDL's textures). Watching 17 GiB of a 31B accumulate under "ovh" says the
    /// opposite of what the segment is for.
    loading: u64 = 0,
};

/// Exponential smoother over successive readings. One instance per meter.
pub const Smoother = struct {
    /// Smoothed state; null until the first reading seeds it (so the meter never
    /// ramps up from zero on the first frame).
    parts: ?Parts = null,
    /// Weight of a NEW reading. At the 200-500 ms sample cadence 0.25 gives a
    /// ~1.5 s time constant: fast enough to follow a real change (unloading a
    /// model frees its context), slow enough to ignore driver lag.
    alpha: f32 = 0.25,

    pub fn update(self: *Smoother, r: Reading) Parts {
        // Our process can't hold less than we've already tracked: the driver's
        // per-process number lags our exact counters (and is 0 when there's no
        // per-process data at all), which would otherwise wrap `overhead`.
        const proc = @max(r.proc_used, r.ours);
        var inst: Parts = .{
            .overhead = proc - r.ours,
            .system = r.device_used -| proc,
        };
        if (r.loading) {
            // Hold `overhead` at the level it had before the load and call the
            // rest the incoming model. The pre-load value is the honest one: the
            // context and textures it describes do not grow while weights
            // upload, so everything above it is the model.
            const base = if (self.parts) |p| @min(p.overhead, inst.overhead) else inst.overhead;
            inst.loading = inst.overhead - base;
            inst.overhead = base;
        }
        const sm: Parts = if (self.parts) |p| .{
            .overhead = ema(p.overhead, inst.overhead, self.alpha),
            .system = ema(p.system, inst.system, self.alpha),
            // NOT smoothed: this is a progress readout, and lagging it by a
            // second only makes the bar disagree with the load it is showing.
            .loading = inst.loading,
        } else inst;
        // Keep the UNCLAMPED smoothed state: the clamp below is a display
        // artifact of a full card, and `inst` itself always fits
        // (ours + overhead + system ≤ device_used ≤ total), so this can't run away.
        self.parts = sm;
        return fit(sm, r);
    }
};

/// Shrink the residuals so `ours + overhead + system` can't exceed the card,
/// otherwise the meter's segments would draw past the end of the bar. Only
/// reachable transiently, when smoothed residuals meet a freshly grown `ours`.
fn fit(p: Parts, r: Reading) Parts {
    var out = p;
    // `loading` is drawn with the model, so it comes off the room the residuals
    // have to share, exactly as `ours` does.
    const room = r.total -| r.ours -| out.loading;
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
    // The lag lands in `overhead` (damped), never in `system`.
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
    // Another process grabs 2 GiB and stays. At 0.25/sample (≈2-5 Hz) the meter
    // should be within 5% after 10 samples (≈2-5 s).
    const changed: Reading = .{ .total = 24 * gib, .device_used = 22 * gib, .proc_used = 18 * gib, .ours = 17 * gib };
    var p: Parts = .{};
    for (0..10) |_| p = s.update(changed);
    const want: f64 = @floatFromInt(4 * gib); // 22 − 18
    const got: f64 = @floatFromInt(p.system);
    errdefer std.debug.print("system {d:.2} GiB, want {d:.2}\n", .{ got / 1073741824.0, want / 1073741824.0 });
    try std.testing.expect(@abs(got - want) / want < 0.05);
}

test "fit shrinks system before overhead" {
    // Overhead is ours and measured; system is the guess, so the guess yields first.
    const p = fit(
        .{ .overhead = 1 * gib, .system = 2 * gib },
        .{ .total = 24 * gib, .device_used = 23 * gib, .proc_used = 0, .ours = 23 * gib },
    );
    try std.testing.expectEqual(@as(u64, 1 * gib), p.overhead);
    try std.testing.expectEqual(@as(u64, 0), p.system);
}

test "a model mid-load is reported as the model, not as overhead" {
    // The reported display bug: the loader builds the session on its own thread
    // and publishes it only when whole, so a 31B's ~17 GiB of uploads are on the
    // card with nothing tracking them. They landed in `overhead`, which is meant
    // for the CUDA context and SDL's textures, i.e. VRAM of ours that is NOT a
    // model.
    var s: Smoother = .{};
    // Idle: 800 MiB of genuine overhead (context + UI), nothing loaded.
    const idle: Reading = .{ .total = 24 * gib, .device_used = 2 * gib, .proc_used = 800 * mib, .ours = 0 };
    const before = s.update(idle);
    try std.testing.expectEqual(@as(u64, 800 * mib), before.overhead);
    try std.testing.expectEqual(@as(u64, 0), before.loading);

    // Mid-load: 12 GiB uploaded so far. Overhead holds at its pre-load level and
    // the growth is attributed to the model.
    const mid = s.update(.{
        .total = 24 * gib,
        .device_used = 14 * gib,
        .proc_used = 800 * mib + 12 * gib,
        .ours = 0, // session not published yet
        .loading = true,
    });
    try std.testing.expectEqual(@as(u64, 12 * gib), mid.loading);
    try std.testing.expectEqual(@as(u64, 800 * mib), mid.overhead);

    // Published: the same bytes are tracked now, so `loading` collapses and
    // overhead is back to being just the overhead. No jump in what is drawn.
    const done = s.update(.{
        .total = 24 * gib,
        .device_used = 22 * gib,
        .proc_used = 800 * mib + 20 * gib,
        .ours = 20 * gib,
    });
    try std.testing.expectEqual(@as(u64, 0), done.loading);
    try std.testing.expectEqual(@as(u64, 800 * mib), done.overhead);
}

test "a load that outgrows the card still fits the bar" {
    // `loading` is drawn alongside `ours`, so it has to be inside `fit`'s room
    // like `ours` is, or the segments run off the end of the meter.
    const p = fit(
        .{ .overhead = 1 * gib, .system = 2 * gib, .loading = 20 * gib },
        .{ .total = 24 * gib, .device_used = 23 * gib, .proc_used = 0, .ours = 2 * gib, .loading = true },
    );
    try std.testing.expect(2 * gib + p.loading + p.overhead + p.system <= 24 * gib);
}
