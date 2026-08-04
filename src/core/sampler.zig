//! **How to move between two sigmas** — the samplers (steppers).
//!
//! Where the steps *go* is `schedule.zig`'s job (the nine ComfyUI schedulers and the
//! two families' sigma tables); this file owns what happens between two of them, and
//! re-exports the schedule names that used to live here so existing callers are
//! unaffected.
//!
//! Two samplers today, `Kind` selects between them:
//!
//!  - **Euler** — one model evaluation, first order, stateless, deterministic. For
//!    krea2 the model predicts a velocity (`CONST`), for the SD family it predicts eps;
//!    either way the prediction IS the trajectory derivative, so `denoised = x - sigma*v`
//!    and one `eulerStep` serves both families. CFG mixes derivatives, which is
//!    equivalent to ComfyUI mixing denoised predictions at fixed x.
//!  - **DPM++ 2M SDE** (midpoint and Heun) — second-order multistep in half-logSNR,
//!    with noise from a Brownian tree. See the section at the bottom of this file.

const std = @import("std");
const noise = @import("noise.zig");
const brownian = @import("brownian.zig");
const schedule = @import("schedule.zig");

// Re-exported from `schedule.zig`, which now owns them. Kept here because
// `pipeline`, `main` and ggufy's measurement ladder reach for `sampler.<name>`.
/// The scheduler module itself, for a caller that wants `sampler.schedule.build`.
pub const schedule_mod = schedule;
pub const default_shift = schedule.default_shift;
pub const Scheduler = schedule.Scheduler;
pub const SigmaTable = schedule.SigmaTable;
pub const sigmaAt = schedule.sigmaAt;
pub const simpleSchedule = schedule.simpleSchedule;
pub const sdSchedule = schedule.sdSchedule;
pub const sdSigmasFull = schedule.sdSigmasFull;
pub const sdTimesteps = schedule.sdTimesteps;
pub const sd_beta_start = schedule.sd_beta_start;
pub const sd_beta_end = schedule.sd_beta_end;
pub const sd_train_steps = schedule.sd_train_steps;

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
    fillNoiseFrom(x, seed, .torch_cpu);
}

/// The same, from an explicit generator — `.nv_philox` is what A1111 draws with. Kept as
/// a separate entry point so every existing caller keeps ComfyUI's behaviour by name.
pub fn fillNoiseFrom(x: []f32, seed: u64, src: noise.Source) void {
    noise.randn(x, seed, src);
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
    // ⚠️ Interpolate in LOG space, because that is what `interpLadder` (and hence the
    // schedule) does — so this is the actual inverse rather than an approximate one,
    // and `sdModelTimestep`'s rounding of it is then *exactly* ComfyUI's
    // `argmin |log sigma - log sigma_i|` instead of merely agreeing with it in the
    // common case.
    const log_sigma = @log(sigma);
    const log_lo = @log(ladder[lo]);
    const span = @log(ladder[hi]) - log_lo;
    const frac = if (span > 0) (log_sigma - log_lo) / span else 0;
    return @as(f32, @floatFromInt(lo)) + frac;
}

/// **The timestep the SD UNet is actually conditioned on at `sigma`**: the nearest
/// *trained* index, not the fractional one.
///
/// ⚠️ **The second place ComfyUI and diffusers disagree** (the first is
/// `sdScaleInitialNoise`), and the more expensive one to get wrong:
///
/// - **ComfyUI**'s `model_sampling.timestep(sigma)` is `argmin |log sigma - log sigma_i|`
///   — it snaps to an integer index the model was trained at.
/// - **diffusers** passes its fractional `timesteps` straight to the UNet.
///
/// The sinusoidal embedding is continuous, so both are well-defined and neither errors.
/// **Measured on SDXL** (512², 8 steps, CFG 7.5, same seed, against a ComfyUI render):
/// fractional gives **30.9 dB**, snapping gives **56.1 dB** — i.e. the fractional form is
/// visibly a different image, and this one is pixel-level agreement (RMSE 0.4/255). At 8
/// steps the fractional indices land ~0.3 off an integer, and that offset is present in
/// *every* step's conditioning.
///
/// Rounding the fractional index is equivalent to ComfyUI's log-space `argmin` for any
/// sigma that came off the ladder, which is every sigma a schedule produces. A sigma
/// sitting almost exactly between two indices — only a teacher-forced probe would supply
/// one — could round the other way; the two indices are adjacent, so it does not matter.
pub fn sdModelTimestep(ladder: []const f32, sigma: f32) f32 {
    return @round(sdTimestepForSigma(ladder, sigma));
}

/// Scale a freshly drawn unit-normal latent to the schedule's starting noise level:
/// `x *= sigmas[0]`. This is the **flow-matching** form; the SD family uses
/// `sdScaleInitialNoise`, and `Session.scaleInitialNoise` picks between them.
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

/// The SD family's initial noise scaling when starting from **full** noise:
/// `x *= sqrt(1 + sigma_max²)`, *not* `x *= sigma_max`.
///
/// ⚠️ **A 0.23% difference worth ~28 dB of render agreement, and the two reference
/// implementations disagree about it** — which is why it needs saying out loud rather
/// than being derived:
///
/// - **ComfyUI** always uses `sqrt(1 + sigma²)` at max denoise
///   (`model_sampling.noise_scaling` under `Sampler.max_denoise`), for any spacing.
/// - **diffusers**' `EulerDiscreteScheduler.init_noise_sigma` returns a bare `max_sigma`
///   when `timestep_spacing in {"linspace", "trailing"}` and `sqrt(max_sigma² + 1)`
///   otherwise — and `linspace` is the spacing this engine samples with, so diffusers'
///   figure here is the bare sigma.
///
/// **ComfyUI's is the target**: it is the compatibility target for renders here, it is
/// what k-diffusion/A1111 do, and it is the principled one — the UNet's input is
/// pre-scaled by `1/sqrt(sigma²+1)` (`sdScaleInput`), so scaling pure noise by
/// `sqrt(1+sigma²)` is exactly what hands the model a unit-variance input at step 0.
///
/// Measured on SDXL (512², 8 steps, CFG 7.5, same seed): the bare-sigma start renders the
/// *same composition* with visibly different fine detail and colour fringing — **22.0 dB**
/// against ComfyUI, where this form gives 50+. An 0.23% perturbation of the starting
/// latent is not small once eight denoising steps have amplified it.
pub fn sdScaleInitialNoise(x: []f32, sigma_max: f32) void {
    const s: f32 = @sqrt(1.0 + sigma_max * sigma_max);
    for (x) |*v| v.* *= s;
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

test "the SD schedule matches ComfyUI's `normal` scheduler, not diffusers'" {
    // ⚠️ **This is the test the diffusers fixture above could not be.** Both
    // schedules read the same (bit-identical) beta ladder at the same indices; they
    // disagree only on the interpolation *space* — ComfyUI lerps `log_sigmas`,
    // diffusers lerps sigma. That is worth up to 4.4e-5, so a comparison at the
    // 2e-4 the diffusers fixture needs is blind to it, and Euler barely notices
    // (53.8 dB against a ComfyUI render either way). DPM++ 2M SDE notices
    // completely: `brownian.zig` quantises sigma to 1e-6 as a tree key, so the
    // wrong space re-rolls the whole noise path (~20 dB, measured).
    //
    // Hence a TIGHT bound here — a few f32 ulp, which is all that torch's and Zig's
    // `exp`/`log` differ by — against ComfyUI's own `normal_scheduler` output.
    const gpa = std.testing.allocator;
    const json_text = @embedFile("assets/dpmpp_sde_fixtures.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
    defer parsed.deinit();

    var it = parsed.value.object.get("sd_schedules").?.object.iterator();
    while (it.next()) |entry| {
        const steps = try std.fmt.parseInt(usize, entry.key_ptr.*, 10);
        const want = entry.value_ptr.array.items;
        const got = try sdSchedule(gpa, steps);
        defer gpa.free(got);
        try std.testing.expectEqual(want.len, got.len);

        var same_cell: usize = 0;
        for (want, got, 0..) |w, a, i| {
            const e: f32 = @floatCast(w.float);
            errdefer std.debug.print("{d}-step sigma[{d}]: ComfyUI {d:.9} got {d:.9}\n", .{ steps, i, e, a });
            if (e == 0) {
                try std.testing.expectEqual(@as(f32, 0), a);
            } else {
                // ~2 f32 ulp. Both remaining conventions this test exists to catch are
                // an order of magnitude coarser (log-space interpolation: 4e-6 rel;
                // the f32 linspace index: 5e-7 rel), while torch's and Zig's `exp`/`log`
                // legitimately differ by one ulp.
                try std.testing.expectApproxEqRel(e, a, 2.4e-7);
            }
            if (brownian.round6(e) == brownian.round6(a)) same_cell += 1;
        }

        // ⚠️ **The bound above is not the whole story, because the SDE sampler consumes
        // these as QUANTISED tree keys**: `round6` cells are 1e-6 wide and an f32 ulp
        // near sigma 10 is 9.5e-7, so a legitimate one-ulp libm difference sometimes
        // lands in the neighbouring cell and that step draws different noise.
        //
        // Measured against ComfyUI on this machine: 5/5 and 11/11 cells at 4 and 10
        // steps (bit-exact), 19/21 at 20, 29/31 at 30. Asserted as a fraction rather
        // than a count so a one-ulp shift in Zig's libm is not a failure — while a
        // reverted convention, which puts nearly every sigma in the wrong cell, still
        // is. See the SdeStepper section for what the residual costs.
        const frac = @as(f64, @floatFromInt(same_cell)) / @as(f64, @floatFromInt(want.len));
        errdefer std.debug.print("{d}-step: only {d}/{d} sigmas in ComfyUI's round6 cell\n", .{ steps, same_cell, want.len });
        try std.testing.expect(frac >= 0.85);
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

test "the two ComfyUI sampling conventions the SD family follows" {
    // ⚠️ Both of these differ from diffusers, both are silent when wrong, and both were
    // *measured* rather than derived — a 512² SDXL render against a ComfyUI render of the
    // same seed went 22.0 dB -> 30.9 dB -> 56.1 dB as they were fixed in turn. This test
    // exists so neither can quietly revert to the diffusers form.
    const gpa = std.testing.allocator;

    // 1. The initial noise is scaled by sqrt(1 + sigma_max²), not sigma_max. The gap is
    //    only 0.23%, and it is the *starting* latent of every trajectory.
    {
        const sigmas = try sdSchedule(gpa, 8);
        defer gpa.free(sigmas);
        var x = [_]f32{ 1.0, -1.0, 0.5 };
        sdScaleInitialNoise(&x, sigmas[0]);
        const want: f32 = @sqrt(1.0 + sigmas[0] * sigmas[0]);
        try std.testing.expectApproxEqRel(want, x[0], 1e-6);
        try std.testing.expectApproxEqRel(-want, x[1], 1e-6);
        // And it is NOT the flow-matching form, which is what makes this worth pinning.
        try std.testing.expect(@abs(want - sigmas[0]) > 0.03);
    }

    // 2. The UNet is conditioned on the nearest *trained* index. At 8 steps the
    //    schedule's own indices are ~0.3 off an integer, so this is not a no-op.
    {
        const ladder = try sdSigmasFull(gpa);
        defer gpa.free(ladder);
        const sigmas = try sdSchedule(gpa, 8);
        defer gpa.free(sigmas);
        const fractional = try sdTimesteps(gpa, 8);
        defer gpa.free(fractional);

        var any_fractional = false;
        for (sigmas[0..8], fractional) |sg, f| {
            const t = sdModelTimestep(ladder, sg);
            try std.testing.expectEqual(@round(t), t); // integral
            try std.testing.expectApproxEqAbs(@round(f), t, 1e-2); // == round(diffusers')
            if (@abs(f - @round(f)) > 0.1) any_fractional = true;
        }
        try std.testing.expect(any_fractional);
    }
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

// ---------------------------------------------------------------------------
// DPM-Solver++(2M) SDE — the second sampler
// ---------------------------------------------------------------------------
//
// Everything above integrates with **Euler**: one model evaluation, one first-order
// step, no state carried between steps, and no stochasticity. `dpmpp_2m_sde` differs
// on all three counts, and each difference is a place a plausible-looking
// implementation goes quietly wrong:
//
//  1. **It works in half-logSNR, not in sigma.** `lambda = log(alpha_t / sigma_t)`,
//     and the two families compute it *differently* — see `SdeStepper.Parameterization`.
//     The exponential-integrator coefficients are then exact for the linear part of
//     the ODE, which is where the accuracy over Euler comes from.
//  2. **It is multistep (the "2M").** The second-order correction reuses the
//     PREVIOUS step's denoised prediction, so the stepper is stateful, and a resumed
//     render has to restore that state or its first step silently degrades to first
//     order. `Snapshot` carries it for exactly that reason.
//  3. **It injects noise (the "SDE"), from a Brownian tree.** Not a fresh `randn`
//     per step — see `brownian.zig`. This is the piece with no shortcut: the noise
//     is a single seed-determined path over the sigma axis, and reproducing
//     ComfyUI's image requires reproducing it exactly.
//
// `solver_type` picks between two ways of applying the multistep correction, and
// ComfyUI exposes both as separate sampler names: `dpmpp_2m_sde` is `midpoint`,
// `dpmpp_2m_sde_heun` is `heun`. They are the same order of accuracy and give
// visibly different images.
//
// ⚠️ **Coefficients are computed in f64 here where the reference computes them on
// 0-dim f32 tensors**, deliberately, for the same reason `timestepEmbedding` uses
// f64: being more accurate than the reference bounds the disagreement by the
// reference's own rounding instead of stacking two errors. It costs nothing — there
// are ~6 scalars per step — and it is why the trajectory fixture is compared at 1e-4
// relative rather than bit-for-bit. The element-wise arithmetic stays f32, matching.

/// Which sampler drives the loop. Names match ComfyUI's.
pub const Kind = enum {
    euler,
    dpmpp_2m_sde,
    dpmpp_2m_sde_heun,

    /// The `--sampler` spelling, and the GUI config value.
    pub fn parse(s: []const u8) ?Kind {
        return std.meta.stringToEnum(Kind, s);
    }

    pub fn label(self: Kind) []const u8 {
        return switch (self) {
            .euler => "euler",
            .dpmpp_2m_sde => "dpmpp_2m_sde",
            .dpmpp_2m_sde_heun => "dpmpp_2m_sde_heun",
        };
    }

    /// The AUTOMATIC1111 `parameters` spelling, for PNG metadata. Different from
    /// `label` (which is ComfyUI's/the CLI's) because a1111 is what reads that field —
    /// including ComfyUI's own metadata importers.
    pub fn a1111Name(self: Kind) []const u8 {
        return switch (self) {
            .euler => "Euler",
            .dpmpp_2m_sde => "DPM++ 2M SDE",
            .dpmpp_2m_sde_heun => "DPM++ 2M SDE Heun",
        };
    }

    /// True when the loop needs an `SdeStepper` rather than `eulerStep`.
    pub fn isSde(self: Kind) bool {
        return self != .euler;
    }
};

/// The half-logSNR the noise level is expressed in — a property of the model's
/// prediction target, not of the sampler.
///
/// ⚠️ **These are not interchangeable and neither errors on the other's schedule.**
/// `flow` on an SD ladder takes `log` of a negative number for any sigma above 1
/// (every step of an SD run) and produces NaN; `eps` on a krea2 schedule is
/// perfectly finite and simply integrates the wrong ODE.
pub const Parameterization = enum {
    /// Rectified flow / `CONST` (krea2): `sigma` is the interpolation coefficient, so
    /// `alpha = 1 - sigma` and `lambda = log((1 - sigma) / sigma)` — ComfyUI's
    /// `sigma.logit().neg()`. Requires `sigma < 1`, hence `offsetFirstSigma`.
    flow,
    /// `EPS` (SD1.5 / SDXL): `alpha = 1` and `lambda = -log(sigma)`.
    eps,

    fn halfLogSnr(self: Parameterization, sigma: f64) f64 {
        return switch (self) {
            // `1 - sigma` is exact in f32/f64 for sigma in [0.5, 1) by Sterbenz, so
            // this has no cancellation problem despite looking like it should.
            .flow => @log((1.0 - sigma) / sigma),
            .eps => -@log(sigma),
        };
    }

    /// `alpha_t = sigma * exp(lambda_t)`, i.e. `1 - sigma` for `flow` and `1` for
    /// `eps`. Written as the reference writes it rather than simplified, so a new
    /// parameterization only has to supply `halfLogSnr`.
    fn alpha(self: Parameterization, sigma: f64, lambda: f64) f64 {
        _ = self;
        return sigma * @exp(lambda);
    }
};

/// ComfyUI's `offset_first_sigma_for_snr`, in place.
///
/// ⚠️ **Without this a flow-matching SDE run is all NaN from the first step.** A
/// krea2 schedule starts at sigma **exactly 1**, where `lambda = log(0/1)` is -inf,
/// so `h` is -inf and every coefficient is NaN. ComfyUI nudges the first sigma to
/// `percent_to_sigma(1e-4)` — the sigma at 0.01% denoising — which for shift 1.15 is
/// 0.99996833. The model is then evaluated at that sigma rather than at 1.0, so this
/// is a (tiny) change to the render and not merely a guard.
///
/// Returns true if it changed anything. A no-op for `eps` (an SD ladder starts near
/// 14.6, where the logarithm is unremarkable) and for any `flow` schedule already
/// below 1.
pub fn offsetFirstSigma(sigmas: []f32, param: Parameterization, shift: f32) bool {
    if (param != .flow or sigmas.len <= 1 or sigmas[0] < 1.0) return false;
    sigmas[0] = @floatCast(sigmaAt(shift, 1.0 - 1e-4));
    return true;
}

/// A DPM-Solver++(2M) SDE stepper: the per-render state Euler does not need.
///
/// Bound to one schedule and one latent length. **`init` mutates `sigmas[0]`**
/// (`offsetFirstSigma`), so it must run before anything that caches per-sigma data
/// off that array — `Session.denoiser` precomputes a timestep vector per entry — and
/// after `scaleInitialNoise`, which ComfyUI applies to the *unoffset* first sigma.
/// The slice is borrowed and must outlive the stepper.
pub const SdeStepper = struct {
    gpa: std.mem.Allocator,
    sigmas: []f32,
    param: Parameterization,
    solver: Solver,
    eta: f64,
    s_noise: f64,
    noise: brownian.NoiseSampler,
    /// The clean-image estimate for the step just taken — the same quantity a
    /// preview wants, and better than reconstructing it from `x` afterwards.
    denoised: []f32,
    old_denoised: []f32,
    have_old: bool,
    h_last: f64,
    noise_buf: []f32,

    pub const Solver = enum { heun, midpoint };

    pub const Options = struct {
        /// Noise level. 0 makes the sampler deterministic (and equal to plain
        /// DPM++(2M)); ComfyUI's default for the SDE variants is 1.
        eta: f64 = 1.0,
        /// Multiplier on the injected noise. ComfyUI's default is 1.
        s_noise: f64 = 1.0,
        solver: Solver = .heun,
        /// Brownian-path seed. ComfyUI passes the render's own seed here
        /// (`extra_args["seed"]`), so the noise and the initial latent share it.
        seed: u64 = 0,
        /// Which generator the tree's per-node draws come from — the same choice as the
        /// initial latent's, and for the same reason. See `noise.zig`: A1111's pinned
        /// k-diffusion builds the tree on the CUDA tensor's device, so an A1111 SDE
        /// render's noise is Philox at *every* node, not just at step 0.
        noise_src: noise.Source = .torch_cpu,
    };

    /// `sigmas` is the full `steps + 1` schedule ending at 0, exactly as the loop
    /// will index it.
    pub fn init(
        gpa: std.mem.Allocator,
        sigmas: []f32,
        n: usize,
        param: Parameterization,
        opts: Options,
        shift: f32,
    ) !SdeStepper {
        std.debug.assert(sigmas.len >= 2);
        std.debug.assert(n > 0);

        // ⚠️ The tree's span comes from the schedule **before** the first-sigma
        // offset — that is the order in `sample_dpmpp_2m_sde`, and the span is part
        // of the path's identity, so getting it from the offset array would change
        // every sample.
        var t0: f32 = std.math.floatMax(f32);
        var t1: f32 = 0;
        for (sigmas) |s| {
            if (s > 0 and s < t0) t0 = s;
            if (s > t1) t1 = s;
        }
        if (!(t0 < t1)) return error.DegenerateSchedule;

        var tree = try brownian.NoiseSampler.init(gpa, n, t0, t1, opts.seed, opts.noise_src);
        errdefer tree.deinit();

        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const old_denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(old_denoised);
        const noise_buf = try gpa.alloc(f32, n);
        errdefer gpa.free(noise_buf);

        _ = offsetFirstSigma(sigmas, param, shift);

        return .{
            .gpa = gpa,
            .sigmas = sigmas,
            .param = param,
            .solver = opts.solver,
            .eta = opts.eta,
            .s_noise = opts.s_noise,
            .noise = tree,
            .denoised = denoised,
            .old_denoised = old_denoised,
            .have_old = false,
            .h_last = 0,
            .noise_buf = noise_buf,
        };
    }

    pub fn deinit(self: *SdeStepper) void {
        self.noise.deinit();
        self.gpa.free(self.denoised);
        self.gpa.free(self.old_denoised);
        self.gpa.free(self.noise_buf);
        self.* = undefined;
    }

    /// One step of the loop: advance `x` from `sigmas[i]` to `sigmas[i + 1]`, given
    /// the model's derivative prediction `v` at `sigmas[i]`. Leaves the step's
    /// clean-image estimate in `self.denoised`.
    ///
    /// `v` is the trajectory derivative either family's forward already returns —
    /// krea2's velocity or SD's eps — so `denoised = x - sigma * v` covers both, the
    /// same identity that lets `eulerStep` be family-neutral.
    pub fn step(self: *SdeStepper, x: []f32, v: []const f32, i: usize) !void {
        std.debug.assert(x.len == self.denoised.len);
        std.debug.assert(v.len == x.len);
        std.debug.assert(i + 1 < self.sigmas.len);

        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }

        if (sigma_next == 0) {
            // The final step is a pure denoising step: no drift, no noise. (Reaching
            // sigma 0 by the exponential-integrator formula would need lambda = +inf.)
            // ⚠️ `h_last` is deliberately left alone here, matching the reference: it
            // never assigns `h` in this branch, so the trailing `h_last = h` carries
            // the previous iteration's value forward. No step follows either way.
            @memcpy(x, self.denoised);
        } else {
            const s: f64 = sigma;
            const sn: f64 = sigma_next;
            const lambda_s = self.param.halfLogSnr(s);
            const lambda_t = self.param.halfLogSnr(sn);
            const h = lambda_t - lambda_s;
            const h_eta = h * (self.eta + 1.0);
            const alpha_t = self.param.alpha(sn, lambda_t);
            // `1 - exp(-h_eta)`, the phi_1 factor. `expm1` rather than `exp` because
            // h_eta is small at high step counts and the difference is the whole value.
            const phi1 = -std.math.expm1(-h_eta);

            const c_x: f32 = @floatCast((sn / s) * @exp(-h * self.eta));
            const c_d: f32 = @floatCast(alpha_t * phi1);
            for (x, self.denoised) |*xi, d| xi.* = c_x * xi.* + c_d * d;

            if (self.have_old) {
                // r = h_last / h; the correction is scaled by 1/r.
                const inv_r = h / self.h_last;
                const c: f32 = @floatCast(switch (self.solver) {
                    .heun => alpha_t * (phi1 / -h_eta + 1.0) * inv_r,
                    .midpoint => 0.5 * alpha_t * phi1 * inv_r,
                });
                for (x, self.denoised, self.old_denoised) |*xi, d, od| xi.* += c * (d - od);
            }

            if (self.eta > 0 and self.s_noise > 0) {
                try self.noise.sample(self.noise_buf, sigma, sigma_next);
                const c: f32 = @floatCast(sn * @sqrt(-std.math.expm1(-2.0 * h * self.eta)) * self.s_noise);
                for (x, self.noise_buf) |*xi, z| xi.* += c * z;
            }
            self.h_last = h;
        }

        @memcpy(self.old_denoised, self.denoised);
        self.have_old = true;
    }

    /// Restore the multistep history when resuming a suspended render, so the first
    /// step after a resume is second-order like every other step rather than
    /// silently first-order. `old` must have the stepper's length.
    pub fn restore(self: *SdeStepper, old: []const f32, h_last: f64) void {
        std.debug.assert(old.len == self.old_denoised.len);
        @memcpy(self.old_denoised, old);
        self.h_last = h_last;
        self.have_old = true;
    }
};

// --- DPM++ 2M SDE tests -----------------------------------------------------

test "sampler kind round-trips its CLI spelling" {
    try std.testing.expectEqual(Kind.dpmpp_2m_sde_heun, Kind.parse("dpmpp_2m_sde_heun").?);
    try std.testing.expectEqual(Kind.euler, Kind.parse("euler").?);
    try std.testing.expectEqual(@as(?Kind, null), Kind.parse("dpmpp_2m_sde_heun_gpu"));
    inline for (comptime std.enums.values(Kind)) |k| {
        try std.testing.expectEqual(k, Kind.parse(k.label()).?);
    }
    try std.testing.expect(!Kind.euler.isSde());
    try std.testing.expect(Kind.dpmpp_2m_sde.isSde());
}

test "the two half-logSNR parameterizations, and why they are not interchangeable" {
    // flow: lambda = log((1-sigma)/sigma), alpha = 1 - sigma.
    {
        const p = Parameterization.flow;
        const lam = p.halfLogSnr(0.25);
        try std.testing.expectApproxEqRel(@as(f64, @log(3.0)), lam, 1e-12);
        try std.testing.expectApproxEqRel(@as(f64, 0.75), p.alpha(0.25, lam), 1e-12);
    }
    // eps: lambda = -log(sigma), alpha = 1 exactly.
    {
        const p = Parameterization.eps;
        const lam = p.halfLogSnr(4.0);
        try std.testing.expectApproxEqRel(@as(f64, -@log(4.0)), lam, 1e-12);
        try std.testing.expectApproxEqRel(@as(f64, 1.0), p.alpha(4.0, lam), 1e-12);
    }
    // lambda must increase as sigma falls, for both — the step direction depends on it.
    try std.testing.expect(Parameterization.flow.halfLogSnr(0.2) > Parameterization.flow.halfLogSnr(0.9));
    try std.testing.expect(Parameterization.eps.halfLogSnr(0.2) > Parameterization.eps.halfLogSnr(9.0));
    // An SD sigma through the flow branch is NaN, not merely inaccurate: that is the
    // failure mode `Session` dispatch has to prevent.
    try std.testing.expect(std.math.isNan(Parameterization.flow.halfLogSnr(14.6)));
}

test "the first-sigma offset is required for flow matching and a no-op elsewhere" {
    const gpa = std.testing.allocator;
    {
        const sigmas = try simpleSchedule(gpa, 8, default_shift);
        defer gpa.free(sigmas);
        try std.testing.expectEqual(@as(f32, 1.0), sigmas[0]);
        // Unoffset, the very first coefficient is -inf and the whole render is NaN.
        try std.testing.expect(std.math.isNegativeInf(Parameterization.flow.halfLogSnr(1.0)));

        try std.testing.expect(offsetFirstSigma(sigmas, .flow, default_shift));
        try std.testing.expectApproxEqAbs(@as(f32, 0.99996833), sigmas[0], 1e-7);
        try std.testing.expect(std.math.isFinite(Parameterization.flow.halfLogSnr(sigmas[0])));
        // Idempotent: sigmas[0] is now below 1.
        try std.testing.expect(!offsetFirstSigma(sigmas, .flow, default_shift));
    }
    {
        // An SD ladder is untouched — `eps` has no singularity at the top.
        const sigmas = try sdSchedule(gpa, 8);
        defer gpa.free(sigmas);
        const before = sigmas[0];
        try std.testing.expect(!offsetFirstSigma(sigmas, .eps, default_shift));
        try std.testing.expectEqual(before, sigmas[0]);
    }
}

test "eta = 0 reduces the SDE stepper to deterministic DPM++(2M)" {
    // The noise term is the only stochastic part, so eta = 0 must be repeatable
    // across seeds. This isolates the multistep solver from the Brownian tree.
    const gpa = std.testing.allocator;
    const n = 32;
    var out: [2][n]f32 = undefined;
    for (&out, [_]u64{ 1, 999 }) |*dst, seed| {
        const sigmas = try sdSchedule(gpa, 6);
        defer gpa.free(sigmas);
        var st = try SdeStepper.init(gpa, sigmas, n, .eps, .{ .eta = 0, .seed = seed }, default_shift);
        defer st.deinit();

        var x: [n]f32 = undefined;
        for (&x, 0..) |*xi, j| xi.* = @floatFromInt(j % 7);
        var v: [n]f32 = undefined;
        for (0..6) |i| {
            // A fixed, sigma-dependent "model": enough structure that the multistep
            // correction is exercised.
            for (&v, x) |*vi, xi| vi.* = 0.1 * xi + 0.3;
            try st.step(&x, &v, i);
        }
        dst.* = x;
    }
    try std.testing.expectEqualSlices(f32, &out[0], &out[1]);
}

test "the SDE stepper's last step lands exactly on the denoised estimate" {
    // sigma_next == 0 is a pure denoising step; anything else leaves visible noise
    // in the final image.
    const gpa = std.testing.allocator;
    const n = 16;
    const sigmas = try sdSchedule(gpa, 3);
    defer gpa.free(sigmas);
    var st = try SdeStepper.init(gpa, sigmas, n, .eps, .{}, default_shift);
    defer st.deinit();

    var x: [n]f32 = undefined;
    for (&x, 0..) |*xi, j| xi.* = @as(f32, @floatFromInt(j)) - 8.0;
    var v: [n]f32 = undefined;
    for (0..3) |i| {
        for (&v, x) |*vi, xi| vi.* = 0.05 * xi;
        try st.step(&x, &v, i);
    }
    try std.testing.expectEqual(@as(f32, 0), sigmas[3]);
    try std.testing.expectEqualSlices(f32, st.denoised, &x);
}

test "restore reinstates the multistep history bit-identically" {
    // The pause/unload path re-creates the stepper mid-render; without `restore` the
    // step after a resume is first-order and the image changes.
    const gpa = std.testing.allocator;
    const n = 24;

    const run = struct {
        fn straight(a: std.mem.Allocator, x: *[n]f32) !void {
            const sigmas = try sdSchedule(a, 5);
            defer a.free(sigmas);
            var st = try SdeStepper.init(a, sigmas, n, .eps, .{ .seed = 5 }, default_shift);
            defer st.deinit();
            var v: [n]f32 = undefined;
            for (0..5) |i| {
                for (&v, x.*) |*vi, xi| vi.* = 0.07 * xi + 0.2;
                try st.step(x, &v, i);
            }
        }
        fn split(a: std.mem.Allocator, x: *[n]f32) !void {
            var carry: [n]f32 = undefined;
            var h_last: f64 = 0;
            {
                const sigmas = try sdSchedule(a, 5);
                defer a.free(sigmas);
                var st = try SdeStepper.init(a, sigmas, n, .eps, .{ .seed = 5 }, default_shift);
                defer st.deinit();
                var v: [n]f32 = undefined;
                for (0..3) |i| {
                    for (&v, x.*) |*vi, xi| vi.* = 0.07 * xi + 0.2;
                    try st.step(x, &v, i);
                }
                @memcpy(&carry, st.old_denoised);
                h_last = st.h_last;
            }
            const sigmas = try sdSchedule(a, 5);
            defer a.free(sigmas);
            var st = try SdeStepper.init(a, sigmas, n, .eps, .{ .seed = 5 }, default_shift);
            defer st.deinit();
            st.restore(&carry, h_last);
            var v: [n]f32 = undefined;
            for (3..5) |i| {
                for (&v, x.*) |*vi, xi| vi.* = 0.07 * xi + 0.2;
                try st.step(x, &v, i);
            }
        }
    };

    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    for (&a, &b, 0..) |*ai, *bi, j| {
        ai.* = @floatFromInt(j % 5);
        bi.* = @floatFromInt(j % 5);
    }
    try run.straight(gpa, &a);
    try run.split(gpa, &b);
    try std.testing.expectEqualSlices(f32, &a, &b);
}

test "DPM++ 2M SDE matches ComfyUI's own solver on both families" {
    // ⚠️ The fixture is generated by driving ComfyUI's `sample_dpmpp_2m_sde` /
    // `..._heun` over a toy analytic denoiser (`tools/gen_sampler_fixtures.py`), so
    // the solver, the half-logSNR branch, the first-sigma offset AND the Brownian
    // tree are all under test at once. The toy model is `(x + c) / (1 + sigma)` —
    // pure f32 add/divide, no transcendental — so a disagreement here is this code's
    // and not libm's.
    //
    // Self-skip rather than go through `test_gate`: that lives outside `tp_core`,
    // and the fixture is ~46 KB, so this belongs in the fast suite.
    const gpa = std.testing.allocator;
    const json_text = @embedFile("assets/dpmpp_sde_fixtures.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
    defer parsed.deinit();

    const getF32 = struct {
        fn arr(a: std.mem.Allocator, v: std.json.Value) ![]f32 {
            const items = v.array.items;
            const out = try a.alloc(f32, items.len);
            for (out, items) |*o, it| {
                o.* = switch (it) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    else => return error.BadFixture,
                };
            }
            return out;
        }
    }.arr;

    for (parsed.value.object.get("trajectories").?.array.items) |t| {
        const obj = t.object;
        const name = obj.get("name").?.string;
        errdefer std.debug.print("trajectory {s}\n", .{name});

        const sigmas = try getF32(gpa, obj.get("sigmas").?);
        defer gpa.free(sigmas);
        const c = try getF32(gpa, obj.get("c").?);
        defer gpa.free(c);
        const x = try getF32(gpa, obj.get("x0").?);
        defer gpa.free(x);
        const want = try getF32(gpa, obj.get("x_out").?);
        defer gpa.free(want);

        const param: Parameterization = if (std.mem.eql(u8, obj.get("family").?.string, "const")) .flow else .eps;
        const solver: SdeStepper.Solver = if (std.mem.eql(u8, obj.get("solver_type").?.string, "heun")) .heun else .midpoint;
        var st = try SdeStepper.init(gpa, sigmas, x.len, param, .{
            .eta = obj.get("eta").?.float,
            .s_noise = obj.get("s_noise").?.float,
            .solver = solver,
            .seed = @intCast(obj.get("seed").?.integer),
        }, default_shift);
        defer st.deinit();

        const v = try gpa.alloc(f32, x.len);
        defer gpa.free(v);
        for (0..sigmas.len - 1) |i| {
            // The toy model returns `denoised`; the stepper wants the derivative, so
            // invert the same identity it uses: v = (x - denoised) / sigma.
            const sigma = sigmas[i];
            const inv_one_plus: f32 = @floatCast(1.0 + @as(f64, sigma));
            for (v, x, c) |*vi, xi, ci| {
                const denoised = (xi + ci) / inv_one_plus;
                vi.* = (xi - denoised) / sigma;
            }
            try st.step(x, v, i);
        }

        // 1e-4 relative: the reference computes its ~6 scalar coefficients per step on
        // 0-dim f32 tensors, this computes them in f64 (see the section header), and
        // the difference compounds over the run. The Brownian samples themselves are
        // bit-exact — pinned separately below — so this bound is about the solver
        // arithmetic only.
        var max_rel: f64 = 0;
        for (want, x, 0..) |e, a, j| {
            const rel = @abs(@as(f64, e) - @as(f64, a)) / @max(1e-3, @abs(@as(f64, e)));
            if (rel > max_rel) max_rel = rel;
            errdefer std.debug.print("{s}[{d}]: expected {d:.6} got {d:.6}\n", .{ name, j, e, a });
            try std.testing.expect(rel < 1e-4);
        }
    }
}

