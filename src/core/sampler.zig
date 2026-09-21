//! How to move between two sigmas, the samplers (steppers).
//!
//! Where the steps *go* is `schedule.zig`'s job (the nine ComfyUI schedulers and the
//! two families' sigma tables); this file owns what happens between two of them, and
//! re-exports the schedule names a sampler caller also needs.
//!
//! `Kind` selects the sampler and `Stepper` is the per-render state it needs. Eleven
//! of ComfyUI's, which fall into four machines rather than eleven:
//!
//!  - Euler alone: one evaluation, first order, no state, no noise. For krea2 the
//!    model predicts a velocity (`CONST`), for the SD family it predicts eps; either
//!    way the prediction IS the trajectory derivative, so `denoised = x - sigma*v` and
//!    one `eulerStep` serves both families. CFG mixes derivatives, which is equivalent
//!    to ComfyUI mixing denoised predictions at fixed x.
//!  - `TwoStageStepper` (heun, dpm_2): a probe evaluation and the real step off
//!    what it found. Deterministic, no state.
//!  - `AncestralStepper` (euler_ancestral, dpm_2_ancestral, dpmpp_2s_ancestral):
//!    step SHORT of the next sigma and make the gap up with a fresh draw. The last two
//!    place a probe as well, so they cost two evaluations.
//!  - `MultistepStepper` (dpmpp_2m) and `SdeStepper` (dpmpp_2m_sde, its heun
//!    variant, dpmpp_3m_sde): second and third order from the PREVIOUS steps' estimates
//!    instead of a probe, so one evaluation each, and stateful, so a resumed render has
//!    to be handed that state back. `SdeSingleStepper` (dpmpp_sde) is the other
//!    trade: a probe and a Brownian path, no history.
//!
//! Three things here are not derivable and come from reading the reference:
//!
//!  - the stochastic samplers do NOT share a generator (`stepNoiseSource`);
//!  - `dpmpp_2m`, `heun` and `dpm_2` have no `CONST` dispatch at all, so ComfyUI runs
//!    one body over both families where the others run two;
//!  - `dpmpp_sde` splits its step on `exp(-lambda)` rather than on the sigmas.
//!
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

/// The same, from an explicit generator, `.nv_philox` is what A1111 draws with. Kept as
/// a separate entry point so every existing caller keeps ComfyUI's behaviour by name.
pub fn fillNoiseFrom(x: []f32, seed: u64, src: noise.Source) void {
    noise.randn(x, seed, src);
}

// ---------------------------------------------------------------------------
// Choosing a sampler
// ---------------------------------------------------------------------------

/// Which sampler drives the loop. Names match ComfyUI's.
pub const Kind = enum {
    euler,
    euler_ancestral,
    heun,
    dpm_2,
    dpm_2_ancestral,
    dpmpp_2s_ancestral,
    dpmpp_sde,
    dpmpp_2m,
    dpmpp_2m_sde,
    dpmpp_2m_sde_heun,
    dpmpp_3m_sde,

    /// The `--sampler` spelling, and the GUI config value.
    pub fn parse(s: []const u8) ?Kind {
        return std.meta.stringToEnum(Kind, s);
    }

    pub fn label(self: Kind) []const u8 {
        return switch (self) {
            .euler => "euler",
            .euler_ancestral => "euler_ancestral",
            .heun => "heun",
            .dpm_2 => "dpm_2",
            .dpm_2_ancestral => "dpm_2_ancestral",
            .dpmpp_2s_ancestral => "dpmpp_2s_ancestral",
            .dpmpp_sde => "dpmpp_sde",
            .dpmpp_2m => "dpmpp_2m",
            .dpmpp_2m_sde => "dpmpp_2m_sde",
            .dpmpp_2m_sde_heun => "dpmpp_2m_sde_heun",
            .dpmpp_3m_sde => "dpmpp_3m_sde",
        };
    }

    /// The AUTOMATIC1111 `parameters` spelling, for PNG metadata. Different from
    /// `label` (which is ComfyUI's/the CLI's) because a1111 is what reads that field,
    /// including ComfyUI's own metadata importers.
    pub fn a1111Name(self: Kind) []const u8 {
        return switch (self) {
            .euler => "Euler",
            .euler_ancestral => "Euler a",
            .heun => "Heun",
            .dpm_2 => "DPM2",
            .dpm_2_ancestral => "DPM2 a",
            .dpmpp_2s_ancestral => "DPM++ 2S a",
            .dpmpp_sde => "DPM++ SDE",
            .dpmpp_2m => "DPM++ 2M",
            .dpmpp_2m_sde => "DPM++ 2M SDE",
            .dpmpp_2m_sde_heun => "DPM++ 2M SDE Heun",
            .dpmpp_3m_sde => "DPM++ 3M SDE",
        };
    }

    /// True when the sampler draws noise every step, so `Options.eta` and
    /// `Options.s_noise` mean something and the render is not reproducible from the
    /// latent alone.
    pub fn isStochastic(self: Kind) bool {
        return switch (self) {
            .euler, .heun, .dpm_2, .dpmpp_2m => false,
            .euler_ancestral,
            .dpm_2_ancestral,
            .dpmpp_2s_ancestral,
            .dpmpp_sde,
            .dpmpp_2m_sde,
            .dpmpp_2m_sde_heun,
            .dpmpp_3m_sde,
            => true,
        };
    }

    /// How many model evaluations one step costs. The second-order single-step
    /// samplers take two, so the same step count is twice the work; a caller
    /// estimating a render's time has to ask rather than assume one.
    pub fn evalsPerStep(self: Kind) u8 {
        return switch (self) {
            .euler, .euler_ancestral, .dpmpp_2m, .dpmpp_2m_sde, .dpmpp_2m_sde_heun, .dpmpp_3m_sde => 1,
            .heun, .dpm_2, .dpm_2_ancestral, .dpmpp_2s_ancestral, .dpmpp_sde => 2,
        };
    }
};

/// Which generator a sampler's own per-step noise comes from, given the one the
/// INITIAL LATENT was drawn from.
///
/// Not the same answer for both stochastic samplers, and neither is derivable, both
/// come from reading ComfyUI:
///
///  - the SDE samplers' Brownian tree is built with `cpu=True`, so its nodes are
///    torch's CPU generator whatever the latent sits on;
///  - `default_noise_sampler`, which the ancestral samplers use, builds its generator
///    on `x.device` instead, so on any GPU render those draws are Philox.
///
/// Under ComfyUI's defaults that means one render can mix the two: a CPU latent and
/// Philox ancestral noise, from the same seed. A1111 routes this through its own
/// `randn_source` hijack and continues the image's generator rather than starting a
/// fresh one, which is not reproduced here.
pub fn stepNoiseSource(kind: Kind, latent_src: noise.Source) noise.Source {
    return switch (kind) {
        .euler_ancestral, .dpm_2_ancestral, .dpmpp_2s_ancestral => .nv_philox,
        else => latent_src,
    };
}

/// The half-logSNR the noise level is expressed in, a property of the model's
/// prediction target, not of the sampler. Both stochastic samplers dispatch on it,
/// for unrelated reasons (`SdeStepper` needs `lambda`; the ancestral sampler needs to
/// know that alpha moves).
///
/// These are not interchangeable and neither errors on the other's schedule.
/// `flow` on an SD ladder takes `log` of a negative number for any sigma above 1
/// (every step of an SD run) and produces NaN; `eps` on a krea2 schedule is
/// perfectly finite and simply integrates the wrong ODE.
pub const Parameterization = enum {
    /// Rectified flow / `CONST` (krea2): `sigma` is the interpolation coefficient, so
    /// `alpha = 1 - sigma` and `lambda = log((1 - sigma) / sigma)`, ComfyUI's
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

    /// `halfLogSnr` inverted, which the samplers that place a probe in lambda need to
    /// get back to a sigma the model can be evaluated at.
    fn sigmaFor(self: Parameterization, lambda: f64) f64 {
        return switch (self) {
            .flow => 1.0 / (1.0 + @exp(lambda)),
            .eps => @exp(-lambda),
        };
    }
};

/// What a sampler calls to evaluate the model at a point the loop did not evaluate.
///
/// The loop already has the forward at `sigmas[i]` and hands it to `step` as `v`; this
/// is for the second-order samplers, which ask for another at a sigma that is not on
/// the schedule at all. That is safe on every backend because each device session
/// COMPUTES the timestep for a sigma it has no cached entry for, rather than matching
/// the nearest one, so an intermediate sigma is exact and not merely accepted.
///
/// A sampler that takes two evaluations per step costs two forwards per step, not two
/// per render: at equal step counts it is twice the work, which is the trade it makes
/// for the order.
pub const Model = struct {
    ctx: *anyopaque,
    predictFn: *const fn (ctx: *anyopaque, v_out: []f32, x: []const f32, sigma: f32) anyerror!void,

    /// `v_out` is the trajectory derivative at `sigma` for the latent `x`, the same
    /// quantity the loop hands `step`.
    pub fn predict(self: Model, v_out: []f32, x: []const f32, sigma: f32) anyerror!void {
        return self.predictFn(self.ctx, v_out, x, sigma);
    }
};

/// The per-render state a sampler needs, and the one thing a sampling loop drives.
///
/// Euler carries nothing, so the union has an empty arm rather than the loop carrying
/// an optional and a branch. `init` is the only place that knows which sampler wants
/// which state, which is what makes adding one a local change.
pub const Stepper = union(enum) {
    euler,
    two_stage: TwoStageStepper,
    ancestral: AncestralStepper,
    multistep: MultistepStepper,
    sde: SdeStepper,
    sde_2s: SdeSingleStepper,

    pub const Options = struct {
        /// Noise level. 0 makes a stochastic sampler deterministic; ComfyUI's default
        /// is 1 for all of them.
        eta: f64 = 1.0,
        /// Multiplier on the injected noise. ComfyUI's default is 1.
        s_noise: f64 = 1.0,
        /// ComfyUI passes the render's own seed (`extra_args["seed"]`), so the
        /// sampler's noise and the initial latent share it.
        seed: u64 = 0,
        /// Which generator the INITIAL LATENT was drawn from. What each sampler's own
        /// per-step noise comes from is derived from it by `stepNoiseSource`, and is
        /// not always the same thing.
        latent_noise_src: noise.Source = .torch_cpu,
    };

    /// `sigmas` is the full `steps + 1` schedule ending at 0, exactly as the loop will
    /// index it. It is borrowed and must outlive the stepper, and `SdeStepper` mutates
    /// `sigmas[0]`, so nothing may cache per-sigma data off it before this runs.
    pub fn init(
        gpa: std.mem.Allocator,
        kind: Kind,
        sigmas: []f32,
        n: usize,
        param: Parameterization,
        opts: Options,
        shift: f32,
    ) !Stepper {
        const src = stepNoiseSource(kind, opts.latent_noise_src);
        return switch (kind) {
            .euler => .euler,
            .heun, .dpm_2 => .{ .two_stage = try .init(gpa, kind, sigmas, n) },
            .euler_ancestral, .dpm_2_ancestral, .dpmpp_2s_ancestral => .{ .ancestral = try .init(gpa, kind, sigmas, n, param, .{
                .eta = opts.eta,
                .s_noise = opts.s_noise,
                .seed = opts.seed,
                .noise_src = src,
            }) },
            .dpmpp_sde => .{ .sde_2s = try .init(gpa, sigmas, n, param, .{
                .eta = opts.eta,
                .s_noise = opts.s_noise,
                .seed = opts.seed,
                .noise_src = src,
            }, shift) },
            .dpmpp_2m => .{ .multistep = try .init(gpa, sigmas, n) },
            .dpmpp_2m_sde, .dpmpp_2m_sde_heun, .dpmpp_3m_sde => .{ .sde = try .init(gpa, sigmas, n, param, .{
                .eta = opts.eta,
                .s_noise = opts.s_noise,
                .solver = switch (kind) {
                    .dpmpp_2m_sde => .midpoint,
                    // 3M's two-history correction IS the heun one written differently
                    // (`phi_2` against `phi1 / -h_eta + 1`), so it degrades to that
                    // shape for its first correction rather than to midpoint.
                    .dpmpp_2m_sde_heun, .dpmpp_3m_sde => .heun,
                    else => unreachable,
                },
                .third_order = kind == .dpmpp_3m_sde,
                .seed = opts.seed,
                .noise_src = src,
            }, shift) },
        };
    }

    pub fn deinit(self: *Stepper) void {
        switch (self.*) {
            .euler => {},
            inline else => |*s| s.deinit(),
        }
    }

    /// Advance `x` from `sigmas[i]` to `sigmas[i + 1]`, given the model's derivative
    /// prediction `v` at `sigmas[i]`.
    pub fn step(self: *Stepper, x: []f32, v: []const f32, sigmas: []const f32, i: usize, model: Model) !void {
        switch (self.*) {
            .euler => eulerStep(x, v, sigmas[i], sigmas[i + 1]),
            .two_stage => |*s| try s.step(x, v, i, model),
            .ancestral => |*s| try s.step(x, v, i, model),
            .multistep => |*s| s.step(x, v, i),
            .sde => |*s| try s.step(x, v, i),
            .sde_2s => |*s| try s.step(x, v, i, model),
        }
    }

    /// The clean-image estimate for the step just taken, or null when the sampler
    /// keeps none. A caller wanting a preview out of the `null` case must reconstruct
    /// it from the Euler step it knows was taken; reading it back out of any other
    /// sampler's latent gives a differently-scaled image that looks plausible.
    pub fn denoised(self: *const Stepper) ?[]const f32 {
        return switch (self.*) {
            .euler => null,
            inline else => |*s| s.denoised,
        };
    }

    /// What a multistep sampler has to be handed back to resume bit-identically.
    /// `prev2` is null for every sampler but DPM++(3M) SDE, which reaches two steps
    /// back.
    pub const History = struct {
        prev: []const f32,
        prev2: ?[]const f32 = null,
        h: f64 = 0,
        h2: f64 = 0,
    };

    /// The multistep history a bit-identical resume needs, or null when there is none
    /// yet (the first step of a run, and every step of a single-step sampler).
    pub fn history(self: *const Stepper) ?History {
        return switch (self.*) {
            .sde => |*s| if (s.n_old == 0) null else .{
                .prev = s.old_denoised,
                .prev2 = if (s.n_old > 1) s.old_denoised2 else null,
                .h = s.h_last,
                .h2 = s.h_last2,
            },
            .multistep => |*s| if (s.have_old) .{ .prev = s.old_denoised } else null,
            else => null,
        };
    }

    /// Rebuild the state a render resumed at `start_step` needs.
    ///
    /// Two different things, one per stochastic sampler, both silent when missed. The
    /// SDE stepper's multistep history has to be handed back (`old`, from `history`),
    /// while the ancestral sampler's generator has to be wound forward past the draws
    /// the steps before the pause made: its noise is a SEQUENCE, not a path addressed
    /// by sigma, so a generator that restarts at the resume point gives every
    /// remaining step the wrong field.
    pub fn resumeFrom(self: *Stepper, start_step: usize, hist: ?History) void {
        switch (self.*) {
            .euler, .two_stage => {},
            .ancestral => |*s| s.fastForward(start_step),
            .multistep => |*s| if (hist) |h| s.restore(h.prev),
            .sde => |*s| if (hist) |h| s.restore(h),
            // The Brownian path is addressed by sigma, so a resumed render queries
            // the same intervals it would have; nothing to wind forward.
            .sde_2s => {},
        }
    }
};

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
// Discrete-noise (eps-prediction) sampling, the SD family
// ---------------------------------------------------------------------------
//
// Krea 2 is rectified flow: the model predicts a velocity and sigma runs over a
// continuous schedule. SD1.5/SDXL are the older formulation and differ in three ways
// that all have to line up or the model is being run off-distribution:
//
//  1. Sigma comes from a discrete beta schedule, not a formula:
//     `sigma_i = sqrt((1 - alpha_bar_i) / alpha_bar_i)` over 1000 training steps.
//  2. The model is conditioned on a timestep index, not on sigma, so a sampler
//     that has chosen a sigma must map back to the (fractional) index that produced
//     it. `timestepForSigma` is that inverse.
//  3. The input is pre-scaled by `1/sqrt(sigma^2 + 1)`. SD's UNet expects a
//     unit-variance input; feeding it the raw latent silently runs a model outside
//     its training distribution, which looks like "the sampler is bad" rather than
//     like a missing scale.
//
// The *step* is the same Euler as above once eps is in hand, because for this
// parameterization the trajectory derivative is eps: `denoised = x - sigma*eps`,
// so `d = (x - denoised)/sigma = eps`. That is what lets one measurement harness
// (ggufy's teacher-forced level 2) compare both families without knowing which is
// which.

/// The (fractional) training index whose sigma is `sigma`, the inverse of the
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
    // Interpolate in LOG space, because that is what `interpLadder` (and hence the
    // schedule) does, so this is the actual inverse rather than an approximate one,
    // and `sdModelTimestep`'s rounding of it is then *exactly* ComfyUI's
    // `argmin |log sigma - log sigma_i|` instead of merely agreeing with it in the
    // common case.
    const log_sigma = @log(sigma);
    const log_lo = @log(ladder[lo]);
    const span = @log(ladder[hi]) - log_lo;
    const frac = if (span > 0) (log_sigma - log_lo) / span else 0;
    return @as(f32, @floatFromInt(lo)) + frac;
}

/// The timestep the SD UNet is actually conditioned on at `sigma`: the nearest
/// *trained* index, not the fractional one.
///
/// The second place ComfyUI and diffusers disagree (the first is
/// `sdScaleInitialNoise`), and the more expensive one to get wrong:
///
/// - ComfyUI's `model_sampling.timestep(sigma)` is `argmin |log sigma - log sigma_i|`
///   it snaps to an integer index the model was trained at.
/// - diffusers passes its fractional `timesteps` straight to the UNet.
///
/// The sinusoidal embedding is continuous, so both are well-defined and neither errors.
/// Measured on SDXL (512², 8 steps, CFG 7.5, same seed, against a ComfyUI render):
/// fractional gives 30.9 dB, snapping gives 56.1 dB, i.e. the fractional form is
/// visibly a different image, and this one is pixel-level agreement (RMSE 0.4/255). At 8
/// steps the fractional indices land ~0.3 off an integer, and that offset is present in
/// *every* step's conditioning.
///
/// Rounding the fractional index is equivalent to ComfyUI's log-space `argmin` for any
/// sigma that came off the ladder, which is every sigma a schedule produces. A sigma
/// sitting almost exactly between two indices, only a teacher-forced probe would supply
/// one, could round the other way; the two indices are adjacent, so it does not matter.
pub fn sdModelTimestep(ladder: []const f32, sigma: f32) f32 {
    return @round(sdTimestepForSigma(ladder, sigma));
}

/// Scale a freshly drawn unit-normal latent to the schedule's starting noise level:
/// `x *= sigmas[0]`. This is the flow-matching form; the SD family uses
/// `sdScaleInitialNoise`, and `Session.scaleInitialNoise` picks between them.
///
/// Both families need this and only one of them notices. Flow matching starts at
/// sigma = 1, so for krea2 this is a multiply by exactly 1.0, bit-identical, which is
/// what lets it be applied unconditionally. SD's ladder starts near 14.6, and
/// skipping it hands the UNet a latent ~15x too small: it denoises something that was
/// never noisy, and produces a washed, low-contrast image with no error anywhere.
pub fn scaleInitialNoise(x: []f32, sigma0: f32) void {
    if (sigma0 == 1.0) return; // exact no-op; skip the pass entirely
    for (x) |*v| v.* *= sigma0;
}

/// The SD family's initial noise scaling when starting from full noise:
/// `x *= sqrt(1 + sigma_max²)`, *not* `x *= sigma_max`.
///
/// A 0.23% difference worth ~28 dB of render agreement, and the two reference
/// implementations disagree about it, which is why it needs saying out loud rather
/// than being derived:
///
/// - ComfyUI always uses `sqrt(1 + sigma²)` at max denoise
///   (`model_sampling.noise_scaling` under `Sampler.max_denoise`), for any spacing.
/// - diffusers' `EulerDiscreteScheduler.init_noise_sigma` returns a bare `max_sigma`
///   when `timestep_spacing in {"linspace", "trailing"}` and `sqrt(max_sigma² + 1)`
///   otherwise, and `linspace` is the spacing this engine samples with, so diffusers'
///   figure here is the bare sigma.
///
/// ComfyUI's is the target: it is the compatibility target for renders here, it is
/// what k-diffusion/A1111 do, and it is the principled one, the UNet's input is
/// pre-scaled by `1/sqrt(sigma²+1)` (`sdScaleInput`), so scaling pure noise by
/// `sqrt(1+sigma²)` is exactly what hands the model a unit-variance input at step 0.
///
/// Measured on SDXL (512², 8 steps, CFG 7.5, same seed): the bare-sigma start renders the
/// *same composition* with visibly different fine detail and colour fringing, 22.0 dB
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

    // Then the N-step ladders and their timesteps, the interpolation-in-index
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
    // This is the test the diffusers fixture above could not be. Both
    // schedules read the same (bit-identical) beta ladder at the same indices; they
    // disagree only on the interpolation *space*, ComfyUI lerps `log_sigmas`,
    // diffusers lerps sigma. That is worth up to 4.4e-5, so a comparison at the
    // 2e-4 the diffusers fixture needs is blind to it, and Euler barely notices
    // (53.8 dB against a ComfyUI render either way). DPM++ 2M SDE notices
    // completely: `brownian.zig` quantises sigma to 1e-6 as a tree key, so the
    // wrong space re-rolls the whole noise path (~20 dB, measured).
    //
    // Hence a TIGHT bound here, a few f32 ulp, which is all that torch's and Zig's
    // `exp`/`log` differ by, against ComfyUI's own `normal_scheduler` output.
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

        // The bound above is not the whole story, because the SDE sampler consumes
        // these as QUANTISED tree keys: `round6` cells are 1e-6 wide and an f32 ulp
        // near sigma 10 is 9.5e-7, so a legitimate one-ulp libm difference sometimes
        // lands in the neighbouring cell and that step draws different noise.
        //
        // Measured against ComfyUI on this machine: 5/5 and 11/11 cells at 4 and 10
        // steps (bit-exact), 19/21 at 20, 29/31 at 30. Asserted as a fraction rather
        // than a count so a one-ulp shift in Zig's libm is not a failure, while a
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
    // Both of these differ from diffusers, both are silent when wrong, and both were
    // *measured* rather than derived, a 512² SDXL render against a ComfyUI render of the
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
// DPM-Solver++(2M) SDE, the second sampler
// ---------------------------------------------------------------------------
//
// Euler above is one model evaluation, one first-order step, no state between steps and
// no stochasticity. `dpmpp_2m_sde` differs on all three counts, and each is a place a
// plausible-looking implementation goes quietly wrong:
//
//  1. It works in half-logSNR, not in sigma: `lambda = log(alpha_t / sigma_t)`, which
//     the two families compute DIFFERENTLY (see `SdeStepper.Parameterization`). The
//     exponential-integrator coefficients are then exact for the ODE's linear part,
//     which is where the accuracy over Euler comes from.
//  2. It is multistep (the "2M"): the second-order correction reuses the PREVIOUS
//     step's denoised prediction, so the stepper is stateful and a resumed render must
//     restore that state or its first step silently degrades to first order.
//     `Snapshot` carries it.
//  3. It injects noise (the "SDE") from a Brownian tree, not a fresh `randn` per step
//     (see `brownian.zig`). The noise is a single seed-determined path over the sigma
//     axis, and reproducing ComfyUI's image requires reproducing it exactly.
//
// `solver_type` picks between two ways of applying the multistep correction, which
// ComfyUI exposes as separate sampler names: `dpmpp_2m_sde` is `midpoint`,
// `dpmpp_2m_sde_heun` is `heun`. Same order of accuracy, visibly different images.
//
// Coefficients are computed in f64 where the reference uses 0-dim f32 tensors, for the
// same reason `timestepEmbedding` does: being more accurate than the reference bounds
// the disagreement by the reference's own rounding instead of stacking two errors. It
// costs nothing at ~6 scalars per step, and it is why the trajectory fixture compares at
// 1e-4 relative rather than bit for bit. The element-wise arithmetic stays f32.

/// ComfyUI's `offset_first_sigma_for_snr`, in place.
///
/// Without this a flow-matching SDE run is all NaN from the first step. A
/// krea2 schedule starts at sigma exactly 1, where `lambda = log(0/1)` is -inf,
/// so `h` is -inf and every coefficient is NaN. ComfyUI nudges the first sigma to
/// `percent_to_sigma(1e-4)`, the sigma at 0.01% denoising, which for shift 1.15 is
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
/// Bound to one schedule and one latent length. `init` mutates `sigmas[0]`
/// (`offsetFirstSigma`), so it must run before anything that caches per-sigma data
/// off that array, `Session.denoiser` precomputes a timestep vector per entry, and
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
    /// The clean-image estimate for the step just taken, the same quantity a
    /// preview wants, and better than reconstructing it from `x` afterwards.
    denoised: []f32,
    /// The previous two steps' estimates and step sizes, newest first. The 2M
    /// solvers read one, 3M reads both; `n_old` says how many are valid, which is
    /// what makes the first steps of a run degrade in order exactly as the
    /// reference's `None` checks do.
    old_denoised: []f32,
    old_denoised2: []f32,
    n_old: u8,
    h_last: f64,
    h_last2: f64,
    /// True for `dpmpp_3m_sde`: take the third-order correction once two steps of
    /// history exist.
    third_order: bool,
    noise_buf: []f32,

    pub const Solver = enum { heun, midpoint };

    pub const Options = struct {
        /// Noise level. 0 makes the sampler deterministic (and equal to plain
        /// DPM++(2M)); ComfyUI's default for the SDE variants is 1.
        eta: f64 = 1.0,
        /// Multiplier on the injected noise. ComfyUI's default is 1.
        s_noise: f64 = 1.0,
        solver: Solver = .heun,
        /// DPM++(3M) SDE: take the third-order correction once two steps of history
        /// exist. Its two-history correction is the `heun` one, so `solver` must be
        /// `.heun` alongside this.
        third_order: bool = false,
        /// Brownian-path seed. ComfyUI passes the render's own seed here
        /// (`extra_args["seed"]`), so the noise and the initial latent share it.
        seed: u64 = 0,
        /// Which generator the tree's per-node draws come from, the same choice as the
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

        // The tree's span comes from the schedule before the first-sigma
        // offset, that is the order in `sample_dpmpp_2m_sde`, and the span is part
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
        // Allocated whatever the solver: one latent, and a solver-dependent
        // allocation is a second thing to get wrong on a resume.
        const old_denoised2 = try gpa.alloc(f32, n);
        errdefer gpa.free(old_denoised2);
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
            .old_denoised2 = old_denoised2,
            .n_old = 0,
            .h_last = 0,
            .h_last2 = 0,
            .third_order = opts.third_order,
            .noise_buf = noise_buf,
        };
    }

    pub fn deinit(self: *SdeStepper) void {
        self.noise.deinit();
        self.gpa.free(self.denoised);
        self.gpa.free(self.old_denoised);
        self.gpa.free(self.old_denoised2);
        self.gpa.free(self.noise_buf);
        self.* = undefined;
    }

    /// One step of the loop: advance `x` from `sigmas[i]` to `sigmas[i + 1]`, given
    /// the model's derivative prediction `v` at `sigmas[i]`. Leaves the step's
    /// clean-image estimate in `self.denoised`.
    ///
    /// `v` is the trajectory derivative either family's forward already returns,
    /// krea2's velocity or SD's eps, so `denoised = x - sigma * v` covers both, the
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
            // `h_last` is deliberately left alone here, matching the reference: it
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

            // `phi_2` in the reference's 3M body, and the same quantity as the `heun`
            // factor below: `phi1 / -h_eta` IS `expm1(-h_eta) / h_eta`.
            const phi2 = std.math.expm1(-h_eta) / h_eta + 1.0;

            if (self.third_order and self.n_old > 1) {
                // DPM-Solver++(3M) SDE: two finite differences over the denoised
                // history, extrapolated. Written with the reference's operation order
                // (divide, then scale) rather than folded into one scalar, because
                // this branch is compared against it directly.
                const r0: f32 = @floatCast(self.h_last / h);
                const r1: f32 = @floatCast(self.h_last2 / h);
                const rsum: f32 = r0 + r1;
                const phi3 = phi2 / h_eta - 0.5;
                const c1: f32 = @floatCast(alpha_t * phi2);
                const c2: f32 = @floatCast(alpha_t * phi3);
                for (x, self.denoised, self.old_denoised, self.old_denoised2) |*xi, d, od, od2| {
                    const d1_0 = (d - od) / r0;
                    const d1_1 = (od - od2) / r1;
                    const diff = d1_0 - d1_1;
                    xi.* += c1 * (d1_0 + diff * r0 / rsum) - c2 * (diff / rsum);
                }
            } else if (self.n_old > 0) {
                if (self.third_order) {
                    // 3M with only one step of history is 2M SDE heun, but it writes
                    // it as a divide by r rather than a fold into the coefficient, so
                    // it rounds differently by an ulp. Kept separate rather than
                    // shared, since both forms are pinned against the reference.
                    const r: f32 = @floatCast(self.h_last / h);
                    const c: f32 = @floatCast(alpha_t * phi2);
                    for (x, self.denoised, self.old_denoised) |*xi, d, od| xi.* += c * ((d - od) / r);
                } else {
                    // r = h_last / h; the correction is scaled by 1/r.
                    const inv_r = h / self.h_last;
                    const c: f32 = @floatCast(switch (self.solver) {
                        .heun => alpha_t * (phi1 / -h_eta + 1.0) * inv_r,
                        .midpoint => 0.5 * alpha_t * phi1 * inv_r,
                    });
                    for (x, self.denoised, self.old_denoised) |*xi, d, od| xi.* += c * (d - od);
                }
            }

            if (self.eta > 0 and self.s_noise > 0) {
                try self.noise.sample(self.noise_buf, sigma, sigma_next);
                const c: f32 = @floatCast(sn * @sqrt(-std.math.expm1(-2.0 * h * self.eta)) * self.s_noise);
                for (x, self.noise_buf) |*xi, z| xi.* += c * z;
            }
            self.h_last2 = self.h_last;
            self.h_last = h;
        }

        // Newest first, so the oldest falls off the end. The slices are swapped
        // rather than copied; `history` hands both out and `restore` writes them back.
        std.mem.swap([]f32, &self.old_denoised, &self.old_denoised2);
        @memcpy(self.old_denoised, self.denoised);
        if (self.n_old < 2) self.n_old += 1;
    }

    /// Restore the multistep history when resuming a suspended render, so the first
    /// step after a resume is the order every other step is rather than silently
    /// dropping one. The vectors must have the stepper's length.
    pub fn restore(self: *SdeStepper, hist: Stepper.History) void {
        std.debug.assert(hist.prev.len == self.old_denoised.len);
        @memcpy(self.old_denoised, hist.prev);
        self.h_last = hist.h;
        self.n_old = 1;
        if (hist.prev2) |p2| {
            std.debug.assert(p2.len == self.old_denoised2.len);
            @memcpy(self.old_denoised2, p2);
            self.h_last2 = hist.h2;
            self.n_old = 2;
        }
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
    // `dpmpp_2m` is the one that is easy to get wrong here: it is a second-order
    // sampler with no noise at all, so it sits with euler on this axis.
    try std.testing.expect(!Kind.euler.isStochastic());
    try std.testing.expect(!Kind.dpmpp_2m.isStochastic());
    try std.testing.expect(Kind.euler_ancestral.isStochastic());
    try std.testing.expect(Kind.dpmpp_3m_sde.isStochastic());
    // And the two stochastic samplers do NOT share a generator under ComfyUI's
    // defaults, which is the fact most likely to be "cleaned up".
    try std.testing.expectEqual(noise.Source.nv_philox, stepNoiseSource(.euler_ancestral, .torch_cpu));
    try std.testing.expectEqual(noise.Source.torch_cpu, stepNoiseSource(.dpmpp_2m_sde, .torch_cpu));
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
    // lambda must increase as sigma falls, for both, the step direction depends on it.
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
        // An SD ladder is untouched, `eps` has no singularity at the top.
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
            st.restore(.{ .prev = &carry, .h = h_last });
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

// --- fixture helpers, shared by the trajectory tests ------------------------

/// Drive one fixture trajectory through the public `Stepper` and compare it with what
/// ComfyUI's own sampler produced. Every fixture key carries the same fields; only
/// which `Kind` reads them differs, so this is the whole of every trajectory test.
///
/// `tol` is a fraction of the trajectory's OWN SCALE, not a per-element relative
/// bound. Two reasons, and the second is why a per-element bound was wrong: the error
/// here is additive rather than proportional (f64 coefficients against the reference's
/// 0-dim f32 tensors, plus ~1e-7 per injected field, since `philox_rng.zig` reproduces
/// A1111's numpy imitation of the device generator rather than curand's f32
/// arithmetic), and a latent element that happens to land near zero would otherwise
/// need a floor pulled out of the air. Measured across every trajectory in the fixture,
/// the worst is 6.3e-7 of scale, so 5e-6 is ~8x headroom while every failure a sampler
/// can actually have is O(1).
fn checkTrajectory(gpa: std.mem.Allocator, obj: std.json.ObjectMap, kind: Kind, tol: f64) !void {
    const name = obj.get("name").?.string;
    errdefer std.debug.print("trajectory {s} through {s}\n", .{ name, kind.label() });

    const sigmas = try fixtureF32(gpa, obj.get("sigmas").?);
    defer gpa.free(sigmas);
    const c = try fixtureF32(gpa, obj.get("c").?);
    defer gpa.free(c);
    const x = try fixtureF32(gpa, obj.get("x0").?);
    defer gpa.free(x);
    const want = try fixtureF32(gpa, obj.get("x_out").?);
    defer gpa.free(want);

    const param: Parameterization = if (std.mem.eql(u8, obj.get("family").?.string, "const")) .flow else .eps;
    var st = try Stepper.init(gpa, kind, sigmas, x.len, param, .{
        .eta = obj.get("eta").?.float,
        .s_noise = obj.get("s_noise").?.float,
        .seed = @intCast(obj.get("seed").?.integer),
        // ComfyUI's own default. `stepNoiseSource` turns it into Philox for the
        // ancestral samplers and leaves it alone for the tree ones, which is exactly
        // the split the fixtures were generated under.
        .latent_noise_src = .torch_cpu,
    }, default_shift);
    defer st.deinit();

    var toy: ToyModel = .{ .c = c };
    const v = try gpa.alloc(f32, x.len);
    defer gpa.free(v);
    for (0..sigmas.len - 1) |i| {
        // The stepper wants the trajectory derivative; the toy model returns the
        // clean image, so invert the identity the steppers themselves use.
        toyModel(v, x, c, sigmas[i]);
        try st.step(x, v, sigmas, i, toy.model());
    }

    var scale: f64 = 0;
    for (want) |e| scale = @max(scale, @abs(@as(f64, e)));
    const bound = tol * scale;
    for (want, x, 0..) |e, a, j| {
        errdefer std.debug.print(
            "{s}[{d}]: expected {d:.8} got {d:.8} (bound {e} on scale {d:.3})\n",
            .{ name, j, e, a, bound, scale },
        );
        try std.testing.expect(@abs(@as(f64, e) - @as(f64, a)) < bound);
    }
}

/// How far a trajectory may drift from ComfyUI's, as a fraction of its own scale.
/// One number for every sampler: see `checkTrajectory`.
const traj_tol: f64 = 5e-6;

/// The fixture document, parsed. Every trajectory test wants it.
fn openFixtures(gpa: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, gpa, @embedFile("assets/dpmpp_sde_fixtures.json"), .{});
}

/// A JSON array of numbers as f32. The generator writes f32 values through Python's
/// double repr, so this narrowing is exact.
fn fixtureF32(a: std.mem.Allocator, v: std.json.Value) ![]f32 {
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

/// A `Model` over `toyModel`, for the tests that drive a second-order sampler with no
/// checkpoint anywhere in sight.
const ToyModel = struct {
    c: []const f32,

    fn predictFn(ctx: *anyopaque, v_out: []f32, x: []const f32, sigma: f32) anyerror!void {
        const self: *ToyModel = @ptrCast(@alignCast(ctx));
        toyModel(v_out, x, self.c, sigma);
    }

    fn model(self: *ToyModel) Model {
        return .{ .ctx = self, .predictFn = predictFn };
    }
};

/// `v = a * x + b`, for the unit tests that need a stand-in with some structure in it
/// rather than a reference to match.
const LinearModel = struct {
    a: f32,
    b: f32,

    fn predictFn(ctx: *anyopaque, v_out: []f32, x: []const f32, sigma: f32) anyerror!void {
        _ = sigma;
        const self: *LinearModel = @ptrCast(@alignCast(ctx));
        for (v_out, x) |*o, xi| o.* = self.a * xi + self.b;
    }

    fn model(self: *LinearModel) Model {
        return .{ .ctx = self, .predictFn = predictFn };
    }
};

/// The generator's toy denoiser, `denoised = (x + c) / (1 + sigma)`, handed to a
/// stepper as the trajectory derivative it expects: `v = (x - denoised) / sigma`,
/// the same identity the steppers invert. Pure f32 add and divide, so Zig and torch
/// agree exactly and a trajectory disagreement is the solver's.
fn toyModel(v: []f32, x: []const f32, c: []const f32, sigma: f32) void {
    const inv_one_plus: f32 = @floatCast(1.0 + @as(f64, sigma));
    for (v, x, c) |*vi, xi, ci| {
        const denoised = (xi + ci) / inv_one_plus;
        vi.* = (xi - denoised) / sigma;
    }
}

test "DPM++ 2M SDE and 3M SDE match ComfyUI's own solvers on both families" {
    // The fixture is generated by driving ComfyUI's `sample_dpmpp_2m_sde` / `..._heun`
    // / `..._3m_sde` over a toy analytic denoiser (`tools/gen_sampler_fixtures.py`), so
    // the solver, the half-logSNR branch, the first-sigma offset AND the Brownian tree
    // are all under test at once. The toy model is `(x + c) / (1 + sigma)`, pure f32
    // add/divide with no transcendental, so a disagreement here is this code's and not
    // libm's.
    //
    // Self-skip rather than go through `test_gate`: that lives outside `tp_core`, and
    // the fixture is a few hundred KB, so this belongs in the fast suite.
    const gpa = std.testing.allocator;
    const parsed = try openFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.object.get("trajectories").?.array.items) |t| {
        const obj = t.object;
        const solver = obj.get("solver_type").?.string;
        const kind: Kind = if (std.mem.eql(u8, solver, "3m"))
            .dpmpp_3m_sde
        else if (std.mem.eql(u8, solver, "heun"))
            .dpmpp_2m_sde_heun
        else
            .dpmpp_2m_sde;
        // The Brownian samples themselves are bit-exact, pinned separately above, so
        // what this bound covers is the solver arithmetic only.
        try checkTrajectory(gpa, obj, kind, traj_tol);
    }
}

test "the deterministic samplers match ComfyUI, including the two-evaluation ones" {
    // `plain` is every sampler that takes no eta and draws no noise: dpmpp_2m, heun
    // and dpm_2. Nothing here is stochastic, so a disagreement is arithmetic.
    //
    // The krea2 entries are the point as much as the SD ones: NONE of these three
    // dispatches on `CONST`, so ComfyUI runs one body for both families and a port
    // that reached for the model's own half-logSNR would be more principled and would
    // not be ComfyUI.
    const gpa = std.testing.allocator;
    const parsed = try openFixtures(gpa);
    defer parsed.deinit();

    const plain = parsed.value.object.get("plain") orelse return error.SkipZigTest;
    for (plain.array.items) |t| {
        const obj = t.object;
        const name = obj.get("name").?.string;
        const kind: Kind = if (std.mem.indexOf(u8, name, "_2m_") != null)
            .dpmpp_2m
        else if (std.mem.indexOf(u8, name, "_heun2_") != null)
            .heun
        else
            .dpm_2;
        try checkTrajectory(gpa, obj, kind, traj_tol);
    }
}


// ---------------------------------------------------------------------------
// Euler ancestral, the stochastic first-order sampler
// ---------------------------------------------------------------------------
//
// One evaluation per step like Euler, but the step lands SHORT of the next sigma and
// the gap is made up with fresh noise, re-randomizing the trajectory every step rather
// than following one Brownian path.
//
// ComfyUI ships two bodies, dispatched on `CONST`:
//
//  - `eps`: sigma alone carries the noise level, so the variance splits into a
//    down-step and an up-noise (`get_ancestral_step`) and the noise is just added.
//  - `flow`: alpha moves with sigma, so a split does not close. The step goes to
//    `sigma_down`, x is rescaled by `alpha_next / alpha_down` to put the signal back
//    at the right level, and the residual noise is added.
//
// At eta = 0 the eps arm is bit-identical to `eulerStep`; the flow arm is
// algebraically equal there but written as a lerp toward `denoised`, so it rounds
// differently.
//
// ⚠️ The noise is a SEQUENCE from one generator, not a Brownian path: step i's field
// depends on every earlier step having drawn. That is what `fastForward` is for, and
// it is why the first-sigma offset the SDE stepper needs has no counterpart here (no
// logarithm is taken).

/// The RF bodies' down-step: `sigma_next` pulled toward `sigma` by eta. There is no
/// variance split here (see the section header), so this is the whole of it.
fn rfDown(sigma_from: f32, sigma_to: f32, eta: f64) f64 {
    const s: f64 = sigma_from;
    const to: f64 = sigma_to;
    return to * (1.0 + (to / s - 1.0) * eta);
}

/// Halfway between two sigmas in LOG space, where the second-order probes sit.
/// `lerp(log a, log b, 0.5).exp()` as the reference writes it; the same number as
/// `sqrt(a * b)`.
fn logMid(a: f32, b: f32) f32 {
    const la = @log(@as(f64, a));
    const lb = @log(@as(f64, b));
    return @floatCast(@exp(la + 0.5 * (lb - la)));
}

/// ComfyUI's `get_ancestral_step`: how far down to step, and how much noise to put
/// back, splitting the next sigma's variance between the two.
fn ancestralStep(sigma_from: f32, sigma_to: f32, eta: f64) struct { down: f64, up: f64 } {
    // The reference's own early out, and the reason eta = 0 is EXACTLY Euler rather
    // than Euler plus a rounding: it returns `sigma_to` itself, not `sqrt(sigma_to^2)`.
    if (eta == 0) return .{ .down = sigma_to, .up = 0 };
    const from: f64 = sigma_from;
    const to: f64 = sigma_to;
    // Clamped to `sigma_to`, which is what keeps `down` real at eta > 1.
    const up = @min(to, eta * @sqrt(to * to * (from * from - to * to) / (from * from)));
    return .{ .down = @sqrt(to * to - up * up), .up = up };
}

/// An `euler_ancestral` stepper: the generator and the scratch Euler does not need.
///
/// Bound to one schedule and one latent length; the schedule is borrowed, read-only
/// (unlike `SdeStepper`, which offsets its first sigma) and must outlive the stepper.
pub const AncestralStepper = struct {
    gpa: std.mem.Allocator,
    kind: Kind,
    sigmas: []const f32,
    param: Parameterization,
    eta: f64,
    s_noise: f64,
    rng: noise.Generator,
    /// The clean-image estimate for the step just taken, the same quantity ComfyUI
    /// hands its preview callback.
    denoised: []f32,
    noise_buf: []f32,
    /// The probe latent and its forward, for the two second-order members. Empty for
    /// `euler_ancestral`, which never evaluates twice.
    x2: []f32,
    v2: []f32,

    pub const Options = struct {
        eta: f64 = 1.0,
        s_noise: f64 = 1.0,
        seed: u64 = 0,
        /// Philox by default because that is what ComfyUI draws here on a GPU render;
        /// see `stepNoiseSource`, which is what a caller should ask.
        noise_src: noise.Source = .nv_philox,
    };

    pub fn init(
        gpa: std.mem.Allocator,
        kind: Kind,
        sigmas: []const f32,
        n: usize,
        param: Parameterization,
        opts: Options,
    ) !AncestralStepper {
        std.debug.assert(sigmas.len >= 2);
        std.debug.assert(n > 0);

        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const noise_buf = try gpa.alloc(f32, n);
        errdefer gpa.free(noise_buf);
        const two = kind.evalsPerStep() > 1;
        const x2 = try gpa.alloc(f32, if (two) n else 0);
        errdefer gpa.free(x2);
        const v2 = try gpa.alloc(f32, if (two) n else 0);
        errdefer gpa.free(v2);

        return .{
            .gpa = gpa,
            .kind = kind,
            .sigmas = sigmas,
            .param = param,
            .eta = opts.eta,
            .s_noise = opts.s_noise,
            .rng = .init(opts.seed, opts.noise_src),
            .denoised = denoised,
            .noise_buf = noise_buf,
            .x2 = x2,
            .v2 = v2,
        };
    }

    pub fn deinit(self: *AncestralStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.noise_buf);
        self.gpa.free(self.x2);
        self.gpa.free(self.v2);
        self.* = undefined;
    }

    /// Whether step `i` draws. Shared with `fastForward` so a resumed render cannot
    /// disagree with the run it is resuming about how many draws have happened.
    ///
    /// Not one rule: the euler and dpm_2 bodies add their noise INSIDE the down-step
    /// branch, so a step with nowhere to go does not draw, while 2S adds it after the
    /// branch and so draws even when it fell back to Euler.
    fn draws(self: *const AncestralStepper, i: usize) bool {
        const sigma = self.sigmas[i];
        const sn = self.sigmas[i + 1];
        return switch (self.kind) {
            .euler_ancestral => switch (self.param) {
                .eps => ancestralStep(sigma, sn, self.eta).down != 0,
                .flow => sn != 0 and self.eta > 0,
            },
            .dpm_2_ancestral => switch (self.param) {
                .eps => ancestralStep(sigma, sn, self.eta).down != 0,
                // No `eta > 0` guard in this one, unlike every other RF body. It
                // changes no image (at eta 0 the tail scales x by 1 and the draw by 0,
                // and every step skips together), so this is here only to keep the
                // predicate and the body saying the same thing.
                .flow => rfDown(sigma, sn, self.eta) != 0,
            },
            .dpmpp_2s_ancestral => sn > 0 and (self.param == .eps or self.eta > 0),
            else => unreachable,
        };
    }

    /// One step: advance `x` from `sigmas[i]` to `sigmas[i + 1]`, given the model's
    /// derivative prediction `v` at `sigmas[i]`. Leaves the step's clean-image
    /// estimate in `self.denoised`, which for the second-order members is the FIRST
    /// evaluation's, the one the reference reports.
    pub fn step(self: *AncestralStepper, x: []f32, v: []const f32, i: usize, model: Model) !void {
        std.debug.assert(x.len == self.denoised.len);
        std.debug.assert(v.len == x.len);
        std.debug.assert(i + 1 < self.sigmas.len);

        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }

        switch (self.kind) {
            .euler_ancestral => switch (self.param) {
                .eps => self.eulerEps(x, v, sigma, sigma_next),
                .flow => self.eulerFlow(x, sigma, sigma_next),
            },
            .dpm_2_ancestral => switch (self.param) {
                .eps => try self.dpm2Eps(x, v, sigma, sigma_next, model),
                .flow => try self.dpm2Flow(x, v, sigma, sigma_next, model),
            },
            .dpmpp_2s_ancestral => switch (self.param) {
                // These two work from the clean-image estimates alone, so unlike
                // every other arm they never look at the derivative.
                .eps => try self.dpmpp2sEps(x, sigma, sigma_next, model),
                .flow => try self.dpmpp2sFlow(x, sigma, sigma_next, model),
            },
            else => unreachable,
        }
    }

    fn eulerEps(self: *AncestralStepper, x: []f32, v: []const f32, sigma: f32, sigma_next: f32) void {
        const a = ancestralStep(sigma, sigma_next, self.eta);
        if (a.down == 0) {
            // Nothing left to step down to: a pure denoising step, the last one of any
            // schedule that ends at 0. `x + v * (0 - sigma)` to the bit.
            @memcpy(x, self.denoised);
            return;
        }
        const dt: f32 = @floatCast(a.down - @as(f64, sigma));
        self.rng.randn(self.noise_buf);
        const c: f32 = @floatCast(a.up * self.s_noise);
        for (x, v, self.noise_buf) |*xi, vi, z| xi.* += dt * vi + c * z;
    }

    fn eulerFlow(self: *AncestralStepper, x: []f32, sigma: f32, sigma_next: f32) void {
        if (sigma_next == 0) {
            @memcpy(x, self.denoised);
            return;
        }
        const down = rfDown(sigma, sigma_next, self.eta);
        // Written as the lerp the reference writes rather than as
        // `x += (down - sigma) * v`, which is the same step: the two round differently
        // and this arm is compared against the reference.
        const ratio: f32 = @floatCast(down / @as(f64, sigma));
        for (x, self.denoised) |*xi, d| xi.* = ratio * xi.* + (1.0 - ratio) * d;
        if (self.eta > 0) self.renoiseFlow(x, sigma_next, down);
    }

    fn dpm2Eps(self: *AncestralStepper, x: []f32, v: []const f32, sigma: f32, sigma_next: f32, model: Model) !void {
        const a = ancestralStep(sigma, sigma_next, self.eta);
        if (a.down == 0) {
            @memcpy(x, self.denoised);
            return;
        }
        const sigma_mid = logMid(sigma, @floatCast(a.down));
        const dt1: f32 = sigma_mid - sigma;
        const dt2: f32 = @floatCast(a.down - @as(f64, sigma));
        for (self.x2, x, v) |*x2, xi, vi| x2.* = xi + dt1 * vi;
        try model.predict(self.v2, self.x2, sigma_mid);
        // The probe's derivative carries the whole step, as in `dpm_2`.
        for (x, self.v2) |*xi, d2| xi.* += dt2 * d2;
        self.rng.randn(self.noise_buf);
        const c: f32 = @floatCast(a.up * self.s_noise);
        for (x, self.noise_buf) |*xi, z| xi.* += c * z;
    }

    fn dpm2Flow(self: *AncestralStepper, x: []f32, v: []const f32, sigma: f32, sigma_next: f32, model: Model) !void {
        const down = rfDown(sigma, sigma_next, self.eta);
        if (down == 0) {
            @memcpy(x, self.denoised);
            return;
        }
        const sigma_mid = logMid(sigma, @floatCast(down));
        const dt1: f32 = sigma_mid - sigma;
        const dt2: f32 = @floatCast(down - @as(f64, sigma));
        for (self.x2, x, v) |*x2, xi, vi| x2.* = xi + dt1 * vi;
        try model.predict(self.v2, self.x2, sigma_mid);
        for (x, self.v2) |*xi, d2| xi.* += dt2 * d2;
        // Unconditional, which is the reference's own shape here; see `draws`.
        self.renoiseFlow(x, sigma_next, down);
    }

    fn dpmpp2sEps(self: *AncestralStepper, x: []f32, sigma: f32, sigma_next: f32, model: Model) !void {
        const a = ancestralStep(sigma, sigma_next, self.eta);
        if (a.down == 0) {
            @memcpy(x, self.denoised);
        } else {
            // DPM-Solver++(2S) in `t = -log(sigma)`, with the probe half a step along
            // in t. Both lines scale the ORIGINAL x, not the probe.
            const t = -@log(@as(f64, sigma));
            const t_next = -@log(a.down);
            const h = t_next - t;
            const t_mid = t + 0.5 * h;
            const sigma_s: f32 = @floatCast(@exp(-t_mid));
            {
                const c_x: f32 = @floatCast(@exp(-t_mid) / @exp(-t));
                const c_d: f32 = @floatCast(-std.math.expm1(-h * 0.5));
                for (self.x2, x, self.denoised) |*x2, xi, d| x2.* = c_x * xi + c_d * d;
            }
            try model.predict(self.v2, self.x2, sigma_s);
            const c_x: f32 = @floatCast(@exp(-t_next) / @exp(-t));
            const c_d: f32 = @floatCast(-std.math.expm1(-h));
            for (x, self.x2, self.v2) |*xi, x2, v2| {
                // The reference passes `denoised_2` to the second line, so invert the
                // probe's forward the same way the first estimate was formed.
                const d2 = x2 - sigma_s * v2;
                xi.* = c_x * xi.* + c_d * d2;
            }
        }
        // OUTSIDE the branch, unlike the other two: an Euler fallback still gets noise.
        if (sigma_next > 0) {
            self.rng.randn(self.noise_buf);
            const c: f32 = @floatCast(a.up * self.s_noise);
            for (x, self.noise_buf) |*xi, z| xi.* += c * z;
        }
    }

    fn dpmpp2sFlow(self: *AncestralStepper, x: []f32, sigma: f32, sigma_next: f32, model: Model) !void {
        const down = rfDown(sigma, sigma_next, self.eta);
        if (sigma_next == 0) {
            @memcpy(x, self.denoised);
        } else {
            // ⚠️ The reference hardcodes the probe at sigma 0.9999 when the schedule
            // starts at exactly 1, where the half-logSNR is -inf. It does NOT use
            // `offset_first_sigma_for_snr` here, so this literal is the whole guard and
            // a flow render's first step depends on it.
            const sigma_s: f32 = if (sigma == 1.0) 0.9999 else blk: {
                const l_i = Parameterization.flow.halfLogSnr(@as(f64, sigma));
                const l_d = Parameterization.flow.halfLogSnr(down);
                const mid = l_i + 0.5 * (l_d - l_i);
                break :blk @floatCast(Parameterization.flow.sigmaFor(mid));
            };
            {
                const r: f32 = sigma_s / sigma;
                for (self.x2, x, self.denoised) |*x2, xi, d| x2.* = r * xi + (1.0 - r) * d;
            }
            try model.predict(self.v2, self.x2, sigma_s);
            const r: f32 = @floatCast(down / @as(f64, sigma));
            for (x, self.x2, self.v2) |*xi, x2, v2| {
                const d2 = x2 - sigma_s * v2;
                xi.* = r * xi.* + (1.0 - r) * d2;
            }
        }
        if (sigma_next > 0 and self.eta > 0) self.renoiseFlow(x, sigma_next, down);
    }

    /// The RF bodies' shared tail: put the signal back at the next sigma's alpha and
    /// add the residual noise.
    fn renoiseFlow(self: *AncestralStepper, x: []f32, sigma_next: f32, down: f64) void {
        const sn: f64 = sigma_next;
        const scale = (1.0 - sn) / (1.0 - down);
        const renoise = @sqrt(sn * sn - down * down * scale * scale);
        self.rng.randn(self.noise_buf);
        const c_x: f32 = @floatCast(scale);
        const c_n: f32 = @floatCast(renoise * self.s_noise);
        for (x, self.noise_buf) |*xi, z| xi.* = c_x * xi.* + c_n * z;
    }

    /// Wind the generator forward past the draws steps `[0, steps_done)` made, for a
    /// render resumed at `steps_done`. Which steps drew depends only on the schedule
    /// and eta, never on the model, so it replays without one.
    pub fn fastForward(self: *AncestralStepper, steps_done: usize) void {
        for (0..@min(steps_done, self.sigmas.len - 1)) |i| {
            if (self.draws(i)) self.rng.randn(self.noise_buf);
        }
    }
};

// --- euler_ancestral tests --------------------------------------------------

test "eta = 0 makes the eps arm bit-identical to plain Euler" {
    // `get_ancestral_step`'s early out is what buys this, and it is worth pinning:
    // a "simplification" that always went through the sqrt would leave a sampler
    // that is Euler to eight digits and reproduces nothing exactly.
    const gpa = std.testing.allocator;
    const n = 32;
    const sigmas = try sdSchedule(gpa, 6);
    defer gpa.free(sigmas);

    var a: [n]f32 = undefined;
    for (&a, 0..) |*ai, j| ai.* = @floatFromInt(j % 7);
    var b = a;

    var st = try AncestralStepper.init(gpa, .euler_ancestral, sigmas, n, .eps, .{ .eta = 0, .seed = 3 });
    defer st.deinit();
    // euler_ancestral never asks for a second forward; the stand-in is there because
    // `step` takes one whatever the sampler.
    var lin: LinearModel = .{ .a = 0.1, .b = 0.3 };
    var v: [n]f32 = undefined;
    for (0..6) |i| {
        for (&v, a) |*vi, ai| vi.* = 0.1 * ai + 0.3;
        try st.step(&a, &v, i, lin.model());
        for (&v, b) |*vi, bi| vi.* = 0.1 * bi + 0.3;
        eulerStep(&b, &v, sigmas[i], sigmas[i + 1]);
    }
    try std.testing.expectEqualSlices(f32, &b, &a);
}

test "the ancestral samplers land on the denoised estimate and depend on the seed" {
    const gpa = std.testing.allocator;
    const n = 32;

    // Both families: the eps arm reaches sigma_down == 0 on the last step, the flow
    // arm takes its own `sigmas[i + 1] == 0` branch.
    for ([_]Parameterization{ .eps, .flow }) |param| {
        const sigmas = if (param == .eps)
            try sdSchedule(gpa, 5)
        else
            try simpleSchedule(gpa, 5, default_shift);
        defer gpa.free(sigmas);

        var out: [2][n]f32 = undefined;
        for (&out, [_]u64{ 11, 12 }) |*dst, seed| {
            var st = try AncestralStepper.init(gpa, .euler_ancestral, sigmas, n, param, .{ .seed = seed });
            defer st.deinit();
            var lin: LinearModel = .{ .a = 0.05, .b = 0 };
            var x: [n]f32 = undefined;
            for (&x, 0..) |*xi, j| xi.* = @as(f32, @floatFromInt(j)) * 0.01;
            var v: [n]f32 = undefined;
            for (0..5) |i| {
                for (&v, x) |*vi, xi| vi.* = 0.05 * xi;
                try st.step(&x, &v, i, lin.model());
            }
            // The last step is a pure denoising step whatever the family: anything
            // else leaves visible noise in the final image.
            try std.testing.expectEqualSlices(f32, st.denoised, &x);
            dst.* = x;
        }
        // And the seed is actually consumed, i.e. the noise is not a constant field.
        try std.testing.expect(!std.mem.eql(f32, &out[0], &out[1]));
    }
}

test "a resumed ancestral render continues the noise sequence" {
    // Restarting the generator at the resume point is the silent failure: every
    // remaining step gets the field an uninterrupted render used EARLIER, so the image
    // is plausible and different.
    const gpa = std.testing.allocator;
    const n = 32;
    const steps = 6;

    const run = struct {
        fn go(a: std.mem.Allocator, x: *[n]f32, sigmas: []const f32, from: usize) !void {
            var st = try AncestralStepper.init(a, .euler_ancestral, sigmas, n, .eps, .{ .seed = 77 });
            defer st.deinit();
            st.fastForward(from);
            var lin: LinearModel = .{ .a = 0.07, .b = 0.2 };
            var v: [n]f32 = undefined;
            for (from..steps) |i| {
                for (&v, x.*) |*vi, xi| vi.* = 0.07 * xi + 0.2;
                try st.step(x, &v, i, lin.model());
            }
        }
    };

    const sigmas = try sdSchedule(gpa, steps);
    defer gpa.free(sigmas);

    var straight: [n]f32 = undefined;
    for (&straight, 0..) |*xi, j| xi.* = @floatFromInt(j % 5);
    var split = straight;

    try run.go(gpa, &straight, sigmas, 0);
    // The same render, stopped after three steps and resumed.
    {
        var st = try AncestralStepper.init(gpa, .euler_ancestral, sigmas, n, .eps, .{ .seed = 77 });
        defer st.deinit();
        var lin: LinearModel = .{ .a = 0.07, .b = 0.2 };
        var v: [n]f32 = undefined;
        for (0..3) |i| {
            for (&v, split) |*vi, xi| vi.* = 0.07 * xi + 0.2;
            try st.step(&split, &v, i, lin.model());
        }
    }
    try run.go(gpa, &split, sigmas, 3);
    try std.testing.expectEqualSlices(f32, &straight, &split);
}

test "the ancestral samplers match ComfyUI's own, both families, one and two stage" {
    // Same shape as the DPM++ fixtures and generated by the same script: ComfyUI's
    // `sample_euler_ancestral` / `sample_dpm_2_ancestral` / `sample_dpmpp_2s_ancestral`
    // driven over the toy analytic denoiser, through the DISPATCHING entry points, so
    // the `CONST` split, the variance split, the probe placement and the noise SEQUENCE
    // are all under test at once. Generated on the GPU, because that is where
    // `default_noise_sampler` puts its generator and it is the whole reason these draws
    // are Philox.
    const gpa = std.testing.allocator;
    const parsed = try openFixtures(gpa);
    defer parsed.deinit();

    // A wrong variance split, a re-seeded generator or a missed draw is off by O(1),
    // so `traj_tol` has room to spare for the noise's own ~1e-7 per draw.
    const one = parsed.value.object.get("ancestral") orelse return error.SkipZigTest;
    for (one.array.items) |t| try checkTrajectory(gpa, t.object, .euler_ancestral, traj_tol);

    const two = parsed.value.object.get("ancestral2") orelse return error.SkipZigTest;
    for (two.array.items) |t| {
        const name = t.object.get("name").?.string;
        const kind: Kind = if (std.mem.indexOf(u8, name, "_dpm2a_") != null) .dpm_2_ancestral else .dpmpp_2s_ancestral;
        try checkTrajectory(gpa, t.object, kind, traj_tol);
    }
}

test "dpmpp_sde matches ComfyUI, tree and probe" {
    // The one sampler here that takes two evaluations AND a Brownian path. Its
    // ancestral split runs on `exp(-lambda)` rather than on the sigmas, which is
    // invisible on the SD arm (there they are the same number) and decides the krea2
    // arm entirely.
    const gpa = std.testing.allocator;
    const parsed = try openFixtures(gpa);
    defer parsed.deinit();

    const fx = parsed.value.object.get("sde_2s") orelse return error.SkipZigTest;
    for (fx.array.items) |t| try checkTrajectory(gpa, t.object, .dpmpp_sde, traj_tol);
}

test "the per-step draws are CUDA's, sequence and all" {
    // What the trajectory test above cannot localize: whether draw 2 is draw 2. The
    // fixture is successive `torch.randn` calls from ONE `torch.Generator(device=cuda)`,
    // which is exactly what `default_noise_sampler` holds, so this pins the offset
    // bookkeeping (one counter block per draw) as well as the values.
    const gpa = std.testing.allocator;
    const json_text = @embedFile("assets/dpmpp_sde_fixtures.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
    defer parsed.deinit();

    const fx = (parsed.value.object.get("cuda_randn_seq") orelse return error.SkipZigTest).object;
    const seed: u64 = @intCast(fx.get("seed").?.integer);
    var g: noise.Generator = .init(seed, .nv_philox);

    for (fx.get("draws").?.array.items, 0..) |draw, k| {
        const items = draw.array.items;
        const got = try gpa.alloc(f32, items.len);
        defer gpa.free(got);
        g.randn(got);
        for (items, got, 0..) |w, a, j| {
            const e: f32 = @floatCast(w.float);
            errdefer std.debug.print("draw {d}[{d}]: CUDA {d:.8} got {d:.8}\n", .{ k, j, e, a });
            // Not bit-exact by construction: f64 Box-Muller against curand's f32.
            try std.testing.expectApproxEqAbs(e, a, 2e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// DPM-Solver++(2M), the deterministic multistep sampler
// ---------------------------------------------------------------------------
//
// `dpmpp_2m_sde` with the noise taken out is NOT this: the SDE solvers are written in
// the model's own half-logSNR, while `sample_dpmpp_2m` uses `t = -log(sigma)` and
// `sigma_fn(t) = exp(-t)` for EVERY family, with no `CONST` dispatch and no first-sigma
// offset. So on krea2 or Z-Image this integrates a different (still valid) ODE than
// `dpmpp_2m_sde` at eta 0 would, and matching ComfyUI means matching that choice rather
// than the principled one.
//
// One model evaluation per step, second order from the PREVIOUS step's estimate
// (the "2M"), so the stepper is stateful and a resumed render must be handed that
// estimate back or its first step is silently first order.
//
// ⚠️ This family needs a schedule with a smooth log tail, and the SD `normal` one is
// not that: its last rung before zero is the ladder's own minimum, so at 8 steps the
// step goes sigma 0.44 -> 0.029 while every step before it moved ~0.5 in log sigma.
// `h` jumps to 2.7, the extrapolation coefficient `1/(2r)` reaches 2.3, and the
// correction throws the latent to 3x its scale, decoding as saturated colour blocks.
// ComfyUI does the same (`tools/render_sd_ref.py` renders one), so a blown low-step
// `normal` render is the sampler, not a defect here. Karras or exponential fixes it.

/// A `dpmpp_2m` stepper: the previous step's clean-image estimate, and nothing else.
pub const MultistepStepper = struct {
    gpa: std.mem.Allocator,
    sigmas: []const f32,
    /// The clean-image estimate for the step just taken.
    denoised: []f32,
    old_denoised: []f32,
    have_old: bool,

    pub fn init(gpa: std.mem.Allocator, sigmas: []const f32, n: usize) !MultistepStepper {
        std.debug.assert(sigmas.len >= 2);
        std.debug.assert(n > 0);
        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const old_denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(old_denoised);
        return .{
            .gpa = gpa,
            .sigmas = sigmas,
            .denoised = denoised,
            .old_denoised = old_denoised,
            .have_old = false,
        };
    }

    pub fn deinit(self: *MultistepStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.old_denoised);
        self.* = undefined;
    }

    pub fn step(self: *MultistepStepper, x: []f32, v: []const f32, i: usize) void {
        std.debug.assert(x.len == self.denoised.len);
        std.debug.assert(v.len == x.len);
        std.debug.assert(i + 1 < self.sigmas.len);

        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }

        // t = -log(sigma) rises as sigma falls, so h > 0 and `expm1(-h)` is in (-1, 0].
        const t = -@log(@as(f64, sigma));
        const t_next = -@log(@as(f64, sigma_next));
        const h = t_next - t;

        // `i == 0` is redundant with `have_old` for a straight run and is not for a
        // resumed one: the correction below reads `sigmas[i - 1]`.
        if (!self.have_old or i == 0 or sigma_next == 0) {
            // First order. At sigma_next == 0 the reference reaches this through
            // infinities (t_next = inf makes the x term vanish and `-expm1(-h)` one),
            // which lands exactly on `x = denoised`.
            if (sigma_next == 0) {
                @memcpy(x, self.denoised);
            } else {
                const c_x: f32 = @floatCast(@exp(-t_next) / @exp(-t));
                const c_d: f32 = @floatCast(-std.math.expm1(-h));
                for (x, self.denoised) |*xi, d| xi.* = c_x * xi.* + c_d * d;
            }
        } else {
            // `h_last` comes from the SCHEDULE, not from what the last step did, which
            // is why this stepper stores no step size: the reference reads
            // `sigmas[i - 1]` right here.
            const h_last = t - -@log(@as(f64, self.sigmas[i - 1]));
            const r = h_last / h;
            const fac: f32 = @floatCast(1.0 / (2.0 * r));
            const c_x: f32 = @floatCast(@exp(-t_next) / @exp(-t));
            const c_d: f32 = @floatCast(-std.math.expm1(-h));
            for (x, self.denoised, self.old_denoised) |*xi, d, od| {
                const dd = (1.0 + fac) * d - fac * od;
                xi.* = c_x * xi.* + c_d * dd;
            }
        }

        @memcpy(self.old_denoised, self.denoised);
        self.have_old = true;
    }

    /// Reinstate the previous step's estimate on a resume. The step size is not
    /// carried because `step` reads it back off the schedule.
    pub fn restore(self: *MultistepStepper, old: []const f32) void {
        std.debug.assert(old.len == self.old_denoised.len);
        @memcpy(self.old_denoised, old);
        self.have_old = true;
    }
};

// --- dpmpp_2m tests ---------------------------------------------------------

test "dpmpp_2m restores its history rather than dropping to first order" {
    const gpa = std.testing.allocator;
    const n = 24;
    const steps = 5;
    const sigmas = try sdSchedule(gpa, steps);
    defer gpa.free(sigmas);

    var straight: [n]f32 = undefined;
    for (&straight, 0..) |*xi, j| xi.* = @floatFromInt(j % 5);
    var split = straight;
    var dropped = straight;

    const drive = struct {
        fn go(st: *MultistepStepper, x: []f32, from: usize, to: usize) void {
            var v: [n]f32 = undefined;
            for (from..to) |i| {
                for (&v, x) |*vi, xi| vi.* = 0.07 * xi + 0.2;
                st.step(x, &v, i);
            }
        }
    }.go;

    {
        var st = try MultistepStepper.init(gpa, sigmas, n);
        defer st.deinit();
        drive(&st, &straight, 0, steps);
    }

    // Split at step 3, carrying the history across.
    var carry: [n]f32 = undefined;
    {
        var st = try MultistepStepper.init(gpa, sigmas, n);
        defer st.deinit();
        drive(&st, &split, 0, 3);
        @memcpy(&carry, st.old_denoised);
        @memcpy(&dropped, &split);
    }
    {
        var st = try MultistepStepper.init(gpa, sigmas, n);
        defer st.deinit();
        st.restore(&carry);
        drive(&st, &split, 3, steps);
    }
    try std.testing.expectEqualSlices(f32, &straight, &split);

    // And without it the resumed render is a DIFFERENT image, not a rounding away,
    // which is what makes the restore worth carrying.
    {
        var st = try MultistepStepper.init(gpa, sigmas, n);
        defer st.deinit();
        drive(&st, &dropped, 3, steps);
    }
    try std.testing.expect(!std.mem.eql(f32, &straight, &dropped));
}

// ---------------------------------------------------------------------------
// Heun and DPM-Solver-2, the deterministic second-order samplers
// ---------------------------------------------------------------------------
//
// Two model evaluations per step, no state between steps, no noise. Both take an
// Euler step to a probe point, evaluate there, and use THAT derivative to make the
// real step; they differ in where the probe sits and how the two derivatives combine:
//
//  - `heun` probes at the next sigma and averages the two derivatives (a trapezoid).
//  - `dpm_2` probes halfway in LOG sigma and uses the probe's derivative alone (a
//    midpoint).
//
// Neither dispatches on the family: ComfyUI runs one body for both, and `dpm_2`'s
// log-space midpoint is taken on the raw sigmas whatever they mean.
//
// ⚠️ ComfyUI's bodies also carry Karras `s_churn`, which its KSampler never sets and
// which is not implemented here: with `s_churn = 0` the reference's `sigma_hat` is
// `sigmas[i]` exactly and the noise injection it guards never runs.

/// A `heun` / `dpm_2` stepper: the probe latent and its derivative.
pub const TwoStageStepper = struct {
    gpa: std.mem.Allocator,
    kind: Kind,
    sigmas: []const f32,
    /// The clean-image estimate from the step's FIRST evaluation, which is the one
    /// the reference reports to its callback.
    denoised: []f32,
    x2: []f32,
    v2: []f32,

    pub fn init(gpa: std.mem.Allocator, kind: Kind, sigmas: []const f32, n: usize) !TwoStageStepper {
        std.debug.assert(sigmas.len >= 2);
        std.debug.assert(n > 0);
        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const x2 = try gpa.alloc(f32, n);
        errdefer gpa.free(x2);
        const v2 = try gpa.alloc(f32, n);
        errdefer gpa.free(v2);
        return .{ .gpa = gpa, .kind = kind, .sigmas = sigmas, .denoised = denoised, .x2 = x2, .v2 = v2 };
    }

    pub fn deinit(self: *TwoStageStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.x2);
        self.gpa.free(self.v2);
        self.* = undefined;
    }

    pub fn step(self: *TwoStageStepper, x: []f32, v: []const f32, i: usize, model: Model) !void {
        std.debug.assert(x.len == self.denoised.len);
        std.debug.assert(v.len == x.len);
        std.debug.assert(i + 1 < self.sigmas.len);

        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }

        const dt = sigma_next - sigma;
        if (sigma_next == 0) {
            // Both reference bodies fall back to Euler for the last step: the probe
            // would sit at sigma 0, where the derivative is not defined.
            eulerStep(x, v, sigma, sigma_next);
            return;
        }

        switch (self.kind) {
            .heun => {
                for (self.x2, x, v) |*x2, xi, vi| x2.* = xi + dt * vi;
                try model.predict(self.v2, self.x2, sigma_next);
                // (d + d_2) / 2, as the reference forms it before scaling by dt.
                for (x, v, self.v2) |*xi, d, d2| xi.* += ((d + d2) / 2.0) * dt;
            },
            .dpm_2 => {
                const sigma_mid = logMid(sigma, sigma_next);
                const dt1 = sigma_mid - sigma;
                for (self.x2, x, v) |*x2, xi, vi| x2.* = xi + dt1 * vi;
                try model.predict(self.v2, self.x2, sigma_mid);
                // The probe's derivative carries the WHOLE step, which is what makes
                // this a midpoint rule rather than an average.
                for (x, self.v2) |*xi, d2| xi.* += dt * d2;
            },
            else => unreachable,
        }
    }
};

// ---------------------------------------------------------------------------
// DPM-Solver++ SDE, the single-step stochastic second-order sampler
// ---------------------------------------------------------------------------
//
// Two model evaluations per step and a Brownian path, where `dpmpp_2m_sde` gets its
// second order from the previous step instead and evaluates once. So it keeps no
// history at all, and a resumed render needs nothing restored: the path is addressed
// by sigma, so the same intervals come back with the same noise.
//
// The structure is the exponential integrator twice over, at a probe half a step along
// in half-logSNR and then at the destination. What is easy to miss is WHERE the
// ancestral split happens: `get_ancestral_step` is applied to `exp(-lambda)` rather
// than to the sigmas, so on a flow model the numbers it splits are not this schedule's
// sigmas at all, and the result comes back through `lambda` to become the step actually
// taken. Reading it as "the sigmas, split" gives a plausible sampler that is not
// ComfyUI's.

/// A `dpmpp_sde` stepper: a Brownian path, the probe latent and its forward.
pub const SdeSingleStepper = struct {
    gpa: std.mem.Allocator,
    sigmas: []f32,
    param: Parameterization,
    eta: f64,
    s_noise: f64,
    r: f64,
    noise: brownian.NoiseSampler,
    /// The clean-image estimate for the step just taken.
    denoised: []f32,
    x2: []f32,
    v2: []f32,
    noise_buf: []f32,

    pub const Options = struct {
        eta: f64 = 1.0,
        s_noise: f64 = 1.0,
        /// Where the probe sits, as a fraction of the step in half-logSNR. ComfyUI
        /// exposes it and defaults to the midpoint; nothing here changes it.
        r: f64 = 0.5,
        seed: u64 = 0,
        noise_src: noise.Source = .torch_cpu,
    };

    pub fn init(
        gpa: std.mem.Allocator,
        sigmas: []f32,
        n: usize,
        param: Parameterization,
        opts: Options,
        shift: f32,
    ) !SdeSingleStepper {
        std.debug.assert(sigmas.len >= 2);
        std.debug.assert(n > 0);

        // Span from the schedule BEFORE the first-sigma offset, the order the
        // reference builds it in; the span is part of the path's identity.
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
        const x2 = try gpa.alloc(f32, n);
        errdefer gpa.free(x2);
        const v2 = try gpa.alloc(f32, n);
        errdefer gpa.free(v2);
        const noise_buf = try gpa.alloc(f32, n);
        errdefer gpa.free(noise_buf);

        _ = offsetFirstSigma(sigmas, param, shift);

        return .{
            .gpa = gpa,
            .sigmas = sigmas,
            .param = param,
            .eta = opts.eta,
            .s_noise = opts.s_noise,
            .r = opts.r,
            .noise = tree,
            .denoised = denoised,
            .x2 = x2,
            .v2 = v2,
            .noise_buf = noise_buf,
        };
    }

    pub fn deinit(self: *SdeSingleStepper) void {
        self.noise.deinit();
        self.gpa.free(self.denoised);
        self.gpa.free(self.x2);
        self.gpa.free(self.v2);
        self.gpa.free(self.noise_buf);
        self.* = undefined;
    }

    pub fn step(self: *SdeSingleStepper, x: []f32, v: []const f32, i: usize, model: Model) !void {
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
            @memcpy(x, self.denoised);
            return;
        }

        const lambda_s = self.param.halfLogSnr(@as(f64, sigma));
        const lambda_t = self.param.halfLogSnr(@as(f64, sigma_next));
        const h = lambda_t - lambda_s;
        const lambda_mid = lambda_s + self.r * h;
        const sigma_mid = self.param.sigmaFor(lambda_mid);

        const alpha_s = self.param.alpha(@as(f64, sigma), lambda_s);
        const alpha_mid = self.param.alpha(sigma_mid, lambda_mid);
        const alpha_t = self.param.alpha(@as(f64, sigma_next), lambda_t);

        // Stage 1, to the probe. The ancestral split runs on `exp(-lambda)`, NOT on
        // the sigmas (see the section header), and its down-step comes back through
        // the logarithm as the lambda actually stepped to.
        {
            const a = ancestralStep(@floatCast(@exp(-lambda_s)), @floatCast(@exp(-lambda_mid)), self.eta);
            const h_ = -@log(a.down) - lambda_s;
            const c_x: f32 = @floatCast((alpha_mid / alpha_s) * @exp(-h_));
            const c_d: f32 = @floatCast(-alpha_mid * std.math.expm1(-h_));
            for (self.x2, x, self.denoised) |*x2, xi, d| x2.* = c_x * xi + c_d * d;
            if (self.eta > 0 and self.s_noise > 0) {
                try self.noise.sample(self.noise_buf, sigma, @floatCast(sigma_mid));
                const c: f32 = @floatCast(alpha_mid * a.up * self.s_noise);
                for (self.x2, self.noise_buf) |*x2, z| x2.* += c * z;
            }
        }
        try model.predict(self.v2, self.x2, @floatCast(sigma_mid));

        // Stage 2, to the destination, off a blend of the two clean-image estimates.
        {
            const a = ancestralStep(@floatCast(@exp(-lambda_s)), @floatCast(@exp(-lambda_t)), self.eta);
            const h_ = -@log(a.down) - lambda_s;
            const fac: f32 = @floatCast(1.0 / (2.0 * self.r));
            const sm: f32 = @floatCast(sigma_mid);
            const c_x: f32 = @floatCast((alpha_t / alpha_s) * @exp(-h_));
            const c_d: f32 = @floatCast(-alpha_t * std.math.expm1(-h_));
            for (x, self.denoised, self.x2, self.v2) |*xi, d, x2, v2| {
                const d2 = x2 - sm * v2;
                xi.* = c_x * xi.* + c_d * ((1.0 - fac) * d + fac * d2);
            }
            if (self.eta > 0 and self.s_noise > 0) {
                try self.noise.sample(self.noise_buf, sigma, sigma_next);
                const c: f32 = @floatCast(alpha_t * a.up * self.s_noise);
                for (x, self.noise_buf) |*xi, z| xi.* += c * z;
            }
        }
    }
};

test "the euler arm of Stepper is bit-identical to calling eulerStep" {
    // The union exists for the samplers that carry state; euler carries none, and a
    // render driven through it has to come out the same BITS as one driven through
    // `eulerStep` directly, or every measurement taken through the stage API
    // (`Session.generate` composed by hand, ggufy's ladder) describes a different
    // model than `generate` renders.
    const gpa = std.testing.allocator;
    const n = 48;
    const sigmas = try sdSchedule(gpa, 7);
    defer gpa.free(sigmas);

    var a: [n]f32 = undefined;
    for (&a, 0..) |*xi, j| xi.* = @as(f32, @floatFromInt(j)) * 0.37 - 8.0;
    var b = a;

    var st = try Stepper.init(gpa, .euler, sigmas, n, .eps, .{}, default_shift);
    defer st.deinit();
    var lin: LinearModel = .{ .a = 0.11, .b = -0.4 };
    var v: [n]f32 = undefined;
    for (0..7) |i| {
        for (&v, a) |*vi, xi| vi.* = 0.11 * xi - 0.4;
        try st.step(&a, &v, sigmas, i, lin.model());
        for (&v, b) |*vi, xi| vi.* = 0.11 * xi - 0.4;
        eulerStep(&b, &v, sigmas[i], sigmas[i + 1]);
    }
    try std.testing.expectEqualSlices(f32, &b, &a);
    // And it keeps no clean-image estimate, which is what makes the preview path
    // reconstruct one instead of reading it back off a latent that has none.
    try std.testing.expectEqual(@as(?[]const f32, null), st.denoised());
}
