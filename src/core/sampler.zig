//! Flow-matching (rectified flow) sampling for Krea 2.
//!
//! Schedule: ComfyUI's `ModelSamplingFlux` with shift 1.15 —
//! sigma(t) = e^mu / (e^mu + (1/t - 1)), tabulated at t = (i+1)/10000 — and
//! the "simple" scheduler indexing that table top-down. Prediction is
//! velocity (CONST): denoised = x - sigma*v; Euler integrates
//! x += (sigma_next - sigma) * v. CFG mixes velocities, which is equivalent
//! to ComfyUI mixing denoised predictions at fixed x.

const std = @import("std");
const torch_rng = @import("torch_rng.zig");

pub const default_shift: f32 = 1.15;
const table_len = 10000;

/// sigma(t) = e^mu / (e^mu + (1/t - 1)); t in (0, 1].
pub fn sigmaAt(shift_mu: f64, t: f64) f64 {
    const e = @exp(shift_mu);
    return e / (e + (1.0 / t - 1.0));
}

/// ComfyUI "simple" scheduler: steps+1 sigmas from sigma_max down to 0,
/// indexed out of the 10000-entry table exactly like the reference.
pub fn simpleSchedule(gpa: std.mem.Allocator, steps: usize, shift_mu: f64) ![]f32 {
    std.debug.assert(steps >= 1);
    const sigs = try gpa.alloc(f32, steps + 1);
    const ss = @as(f64, @floatFromInt(table_len)) / @as(f64, @floatFromInt(steps));
    for (0..steps) |x| {
        const k: usize = @intFromFloat(@as(f64, @floatFromInt(x)) * ss);
        // table[table_len - 1 - k] holds sigma((table_len - k) / table_len).
        const t = @as(f64, @floatFromInt(table_len - k)) / @as(f64, @floatFromInt(table_len));
        sigs[x] = @floatCast(sigmaAt(shift_mu, t));
    }
    sigs[steps] = 0.0;
    return sigs;
}

/// One Euler step: x += (sigma_next - sigma) * v.
pub fn eulerStep(x: []f32, v: []const f32, sigma: f32, sigma_next: f32) void {
    const dt = sigma_next - sigma;
    for (x, v) |*xi, vi| xi.* += dt * vi;
}

/// Classifier-free guidance on velocities, in place into `v_pos`:
/// v = v_neg + cfg * (v_pos - v_neg).
pub fn applyCfg(v_pos: []f32, v_neg: []const f32, cfg: f32) void {
    for (v_pos, v_neg) |*p, n| p.* = n + cfg * (p.* - n);
}

/// Seeded standard-normal noise, bit-identical to torch.randn on the CPU
/// (ComfyUI's prepare_noise), so the same seed reproduces ComfyUI's initial
/// latent exactly. Requires x.len >= 16 (always true for latents).
pub fn fillNoise(x: []f32, seed: u64) void {
    torch_rng.randn(x, seed);
}

// Golden values from the reference math (see tools history): shift 1.15.
test "simple schedule matches comfyui" {
    const gpa = std.testing.allocator;
    {
        const sigs = try simpleSchedule(gpa, 8, default_shift);
        defer gpa.free(sigs);
        const expected = [_]f32{ 1, 0.95672375, 0.904530764, 0.84034878, 0.759510934, 0.654566824, 0.512844086, 0.310901046, 0 };
        for (expected, sigs) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-6);
    }
    {
        const sigs = try simpleSchedule(gpa, 3, default_shift);
        defer gpa.free(sigs);
        const expected = [_]f32{ 1, 0.863338172, 0.612338543, 0 };
        for (expected, sigs) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-6);
    }
}

test "euler and cfg math" {
    var x = [_]f32{ 1.0, 2.0 };
    eulerStep(&x, &.{ 0.5, -0.5 }, 0.8, 0.6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.1), x[1], 1e-6);

    var vp = [_]f32{ 2.0, 0.0 };
    applyCfg(&vp, &.{ 1.0, 1.0 }, 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), vp[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), vp[1], 1e-6);
}

test "noise is deterministic per seed" {
    var a: [32]f32 = undefined;
    var b: [32]f32 = undefined;
    fillNoise(&a, 42);
    fillNoise(&b, 42);
    try std.testing.expectEqualSlices(f32, &a, &b);
    fillNoise(&b, 43);
    try std.testing.expect(!std.mem.eql(f32, &a, &b));
}

// ---------------------------------------------------------------------------
// Discrete-noise (eps-prediction) sampling — the SD family
// ---------------------------------------------------------------------------
//
// Krea 2 is rectified flow: the model predicts a velocity and sigma runs over a
// continuous schedule. SD1.5/SDXL are the older formulation and differ in three ways
// that all have to line up or the model is being run off-distribution:
//
//  1. **Sigma comes from a discrete beta schedule**, not a formula:
//     `sigma_i = sqrt((1 - alpha_bar_i) / alpha_bar_i)` over 1000 training steps.
//  2. **The model is conditioned on a timestep index, not on sigma** — so a sampler
//     that has chosen a sigma must map back to the (fractional) index that produced
//     it. `timestepForSigma` is that inverse.
//  3. **The input is pre-scaled by `1/sqrt(sigma^2 + 1)`.** SD's UNet expects a
//     unit-variance input; feeding it the raw latent silently runs a model outside
//     its training distribution, which looks like "the sampler is bad" rather than
//     like a missing scale.
//
// The *step* is the same Euler as above once eps is in hand, because for this
// parameterization the trajectory derivative **is** eps: `denoised = x - sigma·eps`,
// so `d = (x - denoised)/sigma = eps`. That is what lets one measurement harness
// (ggufy's teacher-forced level 2) compare both families without knowing which is
// which.

/// SD1.5 / SDXL training betas ("scaled_linear"): 1000 steps between these bounds,
/// squared. Both numbers are part of the checkpoint's identity, not tunables.
pub const sd_beta_start: f64 = 0.00085;
pub const sd_beta_end: f64 = 0.012;
pub const sd_train_steps: usize = 1000;

/// The full per-training-step sigma ladder, ascending. Caller frees.
pub fn sdSigmasFull(gpa: std.mem.Allocator) ![]f32 {
    const out = try gpa.alloc(f32, sd_train_steps);
    var alpha_bar: f64 = 1.0;
    const lo = @sqrt(sd_beta_start);
    const hi = @sqrt(sd_beta_end);
    for (out, 0..) |*s, i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(sd_train_steps - 1));
        const beta_sqrt = lo + (hi - lo) * t;
        const beta = beta_sqrt * beta_sqrt;
        alpha_bar *= 1.0 - beta;
        s.* = @floatCast(@sqrt((1.0 - alpha_bar) / alpha_bar));
    }
    return out;
}

/// The `steps + 1` sigma schedule for a run, descending and ending at exactly 0 —
/// the same shape `simpleSchedule` returns, so a caller's sampling loop is identical.
///
/// ⚠️ **The training indices are evenly spaced and the sigmas are read off the ladder
/// by *linear interpolation in index***, which is what diffusers' `EulerDiscreteScheduler`
/// does. Sampling at rounded integer indices instead is a different (and measurably
/// worse) schedule that still produces images — see `sdTrainIndex` for the spacing
/// convention, which matters far more than the interpolation does.
pub fn sdSchedule(gpa: std.mem.Allocator, steps: usize) ![]f32 {
    std.debug.assert(steps > 0);
    const full = try sdSigmasFull(gpa);
    defer gpa.free(full);

    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| {
        out[i] = interpLadder(full, sdTrainIndex(i, steps));
    }
    out[steps] = 0;
    return out;
}

/// The (fractional) training index the `i`-th step of a `steps`-step run samples at:
/// `linspace(0, 999, steps)` taken in reverse, so the run starts at the **top** of the
/// ladder and ends at the bottom.
///
/// ⚠️ **This is `timestep_spacing = "linspace"`, and the alternative is not a detail.**
/// diffusers' "leading" spacing (the old PNDM discretization, `steps_offset = 1`)
/// starts a 4-step run at index 751 — **sigma 4.12 instead of 14.615**. The sampler is
/// then told the latent is only moderately noisy while `scaleInitialNoise` has scaled
/// it as pure noise, so no global structure forms and the image comes out as
/// noise-textured mush. It is a *correct* implementation of the wrong convention, which
/// is why a parity test against a wrongly-configured reference passes happily.
fn sdTrainIndex(i: usize, steps: usize) f64 {
    const last: f64 = @floatFromInt(sd_train_steps - 1);
    if (steps == 1) return last;
    const from_top: f64 = @floatFromInt(steps - 1 - i);
    return from_top * last / @as(f64, @floatFromInt(steps - 1));
}

/// The timesteps matching `sdSchedule`'s sigmas — what the model is conditioned on.
/// Fractional, because the schedule's sigmas generally fall between training steps.
pub fn sdTimesteps(gpa: std.mem.Allocator, steps: usize) ![]f32 {
    const out = try gpa.alloc(f32, steps);
    for (0..steps) |i| out[i] = @floatCast(sdTrainIndex(i, steps));
    return out;
}

/// Linear interpolation into an ascending ladder at a fractional index.
fn interpLadder(ladder: []const f32, idx: f64) f32 {
    const lo: usize = @intFromFloat(@floor(idx));
    const hi = @min(lo + 1, ladder.len - 1);
    const frac: f32 = @floatCast(idx - @floor(idx));
    return ladder[lo] * (1 - frac) + ladder[hi] * frac;
}

/// The (fractional) training index whose sigma is `sigma` — the inverse of the
/// ladder, for a sampler that picked a sigma off-schedule (an off-list sigma is
/// exactly what a teacher-forced measurement feeds in).
pub fn sdTimestepForSigma(ladder: []const f32, sigma: f32) f32 {
    if (sigma <= ladder[0]) return 0;
    if (sigma >= ladder[ladder.len - 1]) return @floatFromInt(ladder.len - 1);
    var lo: usize = 0;
    var hi: usize = ladder.len - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (ladder[mid] <= sigma) lo = mid else hi = mid;
    }
    const span = ladder[hi] - ladder[lo];
    const frac = if (span > 0) (sigma - ladder[lo]) / span else 0;
    return @as(f32, @floatFromInt(lo)) + frac;
}

/// Scale a freshly drawn unit-normal latent to the schedule's starting noise level:
/// `x *= sigmas[0]`.
///
/// ⚠️ **Both families need this and only one of them notices.** Flow matching starts at
/// sigma = 1, so for krea2 this is a multiply by exactly 1.0 — bit-identical, which is
/// what lets it be applied unconditionally. SD's ladder starts near **14.6**, and
/// skipping it hands the UNet a latent ~15x too small: it denoises something that was
/// never noisy, and produces a washed, low-contrast image with no error anywhere.
pub fn scaleInitialNoise(x: []f32, sigma0: f32) void {
    if (sigma0 == 1.0) return; // exact no-op; skip the pass entirely
    for (x) |*v| v.* *= sigma0;
}

/// The input scaling SD's UNet expects: `x / sqrt(sigma^2 + 1)`.
pub fn sdScaleInput(dst: []f32, x: []const f32, sigma: f32) void {
    std.debug.assert(dst.len == x.len);
    const c: f32 = 1.0 / @sqrt(sigma * sigma + 1.0);
    for (dst, x) |*d, v| d.* = v * c;
}

// --- SD-family sampler tests ------------------------------------------------

test "the SD sigma ladder and schedule match diffusers' EulerDiscreteScheduler" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("safetensors.zig");
    // Self-skip rather than go through `test_gate`: that lives outside this module,
    // and the fixture is 1.5 MB, so this belongs in the fast suite when present.
    const ref_path = "src/models/assets/sd15_ref.safetensors";
    var ref = safetensors.SafeTensors.open(gpa, io, ref_path) catch return error.SkipZigTest;
    defer ref.deinit();

    // The full ladder first: it is derived from the betas alone, so a mismatch here
    // means the training schedule is wrong and nothing downstream can be right.
    {
        const want = try ref.get("sched.sigmas_full").?.toF32Alloc(gpa);
        defer gpa.free(want);
        const got = try sdSigmasFull(gpa);
        defer gpa.free(got);
        try std.testing.expectEqual(want.len, got.len);
        for (want, got, 0..) |e, a, i| {
            errdefer std.debug.print("sigma[{d}]: expected {d:.6} got {d:.6}\n", .{ i, e, a });
            try std.testing.expectApproxEqRel(e, a, 1e-5);
        }
    }

    // Then the N-step ladders and their timesteps — the interpolation-in-index
    // convention, which is the part a hand-rolled schedule gets wrong while still
    // producing plausible images.
    inline for (.{ 4, 10 }) |n| {
        const want_s = try ref.get(std.fmt.comptimePrint("sched.sigmas_{d}", .{n})).?.toF32Alloc(gpa);
        defer gpa.free(want_s);
        const want_t = try ref.get(std.fmt.comptimePrint("sched.timesteps_{d}", .{n})).?.toF32Alloc(gpa);
        defer gpa.free(want_t);

        const got_s = try sdSchedule(gpa, n);
        defer gpa.free(got_s);
        const got_t = try sdTimesteps(gpa, n);
        defer gpa.free(got_t);

        try std.testing.expectEqual(want_s.len, got_s.len);
        for (want_s, got_s, 0..) |e, a, i| {
            errdefer std.debug.print("{d}-step sigma[{d}]: expected {d:.6} got {d:.6}\n", .{ n, i, e, a });
            try std.testing.expectApproxEqAbs(e, a, 2e-4);
        }
        try std.testing.expectEqual(want_t.len, got_t.len);
        for (want_t, got_t, 0..) |e, a, i| {
            errdefer std.debug.print("{d}-step timestep[{d}]: expected {d:.4} got {d:.4}\n", .{ n, i, e, a });
            try std.testing.expectApproxEqAbs(e, a, 1e-3);
        }
    }
}

test "sigma -> timestep inverts the ladder, including off-schedule sigmas" {
    // A teacher-forced measurement hands the model a sigma the schedule never
    // visited, so the inverse has to be continuous rather than a table lookup.
    const gpa = std.testing.allocator;
    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);

    for ([_]usize{ 0, 1, 250, 500, 999 }) |i| {
        const t = sdTimestepForSigma(ladder, ladder[i]);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), t, 1e-2);
    }
    // Half way between two rungs comes back half way between two indices.
    const mid = (ladder[400] + ladder[401]) / 2;
    const t_mid = sdTimestepForSigma(ladder, mid);
    try std.testing.expect(t_mid > 400.0 and t_mid < 401.0);
    // Out of range clamps rather than extrapolating.
    try std.testing.expectEqual(@as(f32, 0), sdTimestepForSigma(ladder, 0));
    try std.testing.expectEqual(@as(f32, 999), sdTimestepForSigma(ladder, 1e6));
}

test "the input scaling is what keeps SD's UNet in distribution" {
    var x = [_]f32{ 2.0, -4.0, 0.0 };
    var out: [3]f32 = undefined;
    sdScaleInput(&out, &x, 0.0); // sigma 0 => no scaling at all
    try std.testing.expectEqualSlices(f32, &x, &out);
    sdScaleInput(&out, &x, 1.0); // sigma 1 => 1/sqrt(2)
    const inv = 1.0 / @sqrt(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * inv), out[0], 1e-6);
}
