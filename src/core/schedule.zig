//! **Where the sampling steps go** — the sigma schedules, all nine of ComfyUI's.
//!
//! This is a separate concern from `sampler.zig`, which owns *how* to move between
//! two sigmas (Euler, DPM++ 2M SDE, …). The two axes are independent: any sampler
//! runs on any schedule, and ComfyUI presents them as two dropdowns for that reason.
//! `sampler.zig` re-exports the handful of names that used to live there.
//!
//! A scheduler reads the model's own **sigma table** and decides which points of it
//! a short run visits. `SigmaTable` is the family-neutral view of that table:
//!
//! | | krea2 / flux (`ModelSamplingFlux`) | SD1.5 / SDXL (`ModelSamplingDiscrete`) |
//! |---|---|---|
//! | entries | 10000, `sigma(t) = e^mu / (e^mu + (1/t − 1))` at `t = (i+1)/10000` | 1000, the beta ladder |
//! | `sigma_min … sigma_max` | 0.00031575 … 1.0 | 0.029167 … 14.614641 |
//! | `timestep(sigma)` | identity | `argmin |log σ − log σ_i|` (an integer index) |
//! | `sigma(timestep)` | the same formula | lerp `log_sigmas`, then `exp` |
//!
//! ⚠️ **Everything here computes in f32, matching the reference's rounding rather
//! than bounding it** — the opposite of this codebase's usual "be more accurate than
//! the reference" policy (`timestepEmbedding`). The reason is `brownian.zig`: an SDE
//! sampler quantises the sigma axis to 1e-6 and uses it as a **tree key**, so a
//! schedule value that is one f32 ulp off can select a different noise draw. Measured:
//! one sigma of twenty crossing one 1e-6 cell costs **28.9 dB** (ComfyUI against
//! itself), where staying inside the cell costs 0.6 dB. A quantised key has to agree
//! digit for digit; being *more* accurate is simply being different.
//!
//! Three torch behaviours had to be reverse-engineered to get there, none of which is
//! visible in the Python source, and each of which was worth ulps in a few hundred of
//! the 10000 table entries:
//!
//!  1. ⚠️ **`scalar / tensor` in PyTorch is `reciprocal(tensor) * scalar`** — *two*
//!     roundings, not one. `flux_time_shift`'s `math.exp(mu) / (…)` is exactly that
//!     shape, and computing it as a single f32 division disagrees on **3114 of 10000**
//!     table entries, 165 of them crossing a `round6` cell. `fluxSigma` reproduces the
//!     reciprocal-then-multiply.
//!  2. ⚠️ **`torch.linspace` for f32 is FMA-contracted**: ATen's kernel is
//!     `scalar_start + step * i` in `float`, which the build's `-ffp-contract=fast`
//!     turns into a single `fmaf`. Two separate f32 operations disagree by an ulp
//!     wherever the intermediate is inexact (3 of 20 for a log-space ramp). Hence
//!     `@mulAdd` in `torchLinspace`, plus torch's halfway split so both endpoints land
//!     exactly.
//!  3. ⚠️ **The same formula has two precisions depending on how it is called.**
//!     `ModelSamplingFlux.sigma(t)` on a *tensor* takes the f32 path above; the same
//!     class's `percent_to_sigma` is called with a *Python float* and is therefore f64.
//!     `sigmaAt` is that second path, and `offsetFirstSigma` needs it.
//!
//! ⚠️ **Two schedulers do not return `steps + 1` sigmas.** `ddim_uniform` strides the
//! table (30 requested steps → 31), and `beta` de-duplicates repeated indices. A
//! sampling loop must take its step count from `sigmas.len - 1`, never from the
//! requested `steps` — `pipeline.generate` does.

const std = @import("std");

pub const default_shift: f32 = 1.15;
/// `ModelSamplingFlux.set_parameters(timesteps=10000)`.
pub const flux_table_len: usize = 10000;
/// `ModelSamplingDiscreteFlow.set_parameters(timesteps=1000)` — Lumina 2 / Z-Image.
/// ⚠️ A TENTH of the flux table, which is why the two cannot share a variant even
/// though their sigma formulas are algebraically the same function: every scheduler
/// that indexes the table (`normal`, `sgm_uniform`, `ddim_uniform`, `beta`) reads a
/// different number of rungs.
pub const discrete_flow_table_len: usize = 1000;
/// Z-Image's own `sampling_settings["shift"]` (`supported_models.py::ZImage`), which
/// the official ComfyUI template also sets explicitly via `ModelSamplingAuraFlow`.
pub const z_image_shift: f32 = 3.0;

/// SD1.5 / SDXL training betas ("scaled_linear"): 1000 steps between these bounds,
/// squared. Both numbers are part of the checkpoint's identity, not tunables.
pub const sd_beta_start: f64 = 0.00085;
pub const sd_beta_end: f64 = 0.012;
pub const sd_train_steps: usize = 1000;

// --- primitives -------------------------------------------------------------

/// `flux_time_shift(mu, 1.0, t)` on a **Python float** — f64 throughout.
///
/// This is the path `ModelSamplingFlux.percent_to_sigma` takes; the *tensor* path is
/// `fluxSigma`, and they differ in the last f32 digit. Only `offsetFirstSigma` wants
/// this one.
pub fn sigmaAt(shift_mu: f64, t: f64) f64 {
    const e = @exp(shift_mu);
    return e / (e + (1.0 / t - 1.0));
}

/// `flux_time_shift(mu, 1.0, t)` on an **f32 tensor** — the path that builds
/// `ModelSamplingFlux.sigmas` and that `normal`/`sgm_uniform` evaluate per step.
///
/// ⚠️ The final division is `reciprocal * scalar`, because that is what PyTorch's
/// `scalar / tensor` lowers to (`__rtruediv__`). Writing it as one f32 division is a
/// one-ulp difference on ~31% of the table. Verified bit-exact against
/// `ModelSamplingFlux.sigmas` on all 10000 entries.
pub fn fluxSigma(shift: f32, t: f32) f32 {
    const e: f32 = @floatCast(@exp(@as(f64, shift)));
    const den: f32 = e + (1.0 / t - 1.0);
    return (1.0 / den) * e;
}

/// Entry `i` of `ModelSamplingFlux.sigmas` (ascending, `i < flux_table_len`):
/// `sigma((i + 1) / 10000)`, with the index division in f32 as torch's
/// `arange(1, 10001) / 10000` does it.
pub fn fluxTableAt(shift: f32, i: usize) f32 {
    const t: f32 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(flux_table_len));
    return fluxSigma(shift, t);
}

/// `time_snr_shift(alpha, t)` — `ModelSamplingDiscreteFlow.sigma`, on an f32 tensor.
///
/// Algebraically identical to `fluxSigma` with `alpha = exp(mu)`, and deliberately
/// NOT written that way: torch evaluates this one as `(alpha*t) / (1 + (alpha-1)*t)`
/// — a single tensor/tensor division — where the flux form is `scalar / tensor`,
/// which lowers to `reciprocal * scalar` and rounds twice. Schedule values are
/// quantised to 1e-6 and used as Brownian-tree keys, so a one-ulp difference is a
/// different noise draw (see the module doc).
pub fn discreteFlowSigma(shift: f32, t: f32) f32 {
    if (shift == 1.0) return t;
    return (shift * t) / (1.0 + (shift - 1.0) * t);
}

/// Entry `i` of `ModelSamplingDiscreteFlow.sigmas`, ascending. The multiplier is 1.0
/// for Z-Image, so `set_parameters`' `* multiplier` and `sigma`'s `/ multiplier`
/// cancel exactly and `t` is just `(i + 1) / 1000` in f32, as `arange(1, 1001) / 1000`
/// produces it.
pub fn discreteFlowTableAt(shift: f32, i: usize) f32 {
    const t: f32 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(discrete_flow_table_len));
    return discreteFlowSigma(shift, t);
}

/// The full per-training-step SD sigma ladder, ascending. Caller frees.
pub fn sdSigmasFull(gpa: std.mem.Allocator) ![]f32 {
    const out = try gpa.alloc(f32, sd_train_steps);
    var alpha_bar: f64 = 1.0;
    const lo = @sqrt(sd_beta_start);
    const hi = @sqrt(sd_beta_end);
    for (out, 0..) |*s, i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(sd_train_steps - 1));
        const beta_sqrt = lo + (hi - lo) * t;
        const beta_i = beta_sqrt * beta_sqrt;
        alpha_bar *= 1.0 - beta_i;
        s.* = @floatCast(@sqrt((1.0 - alpha_bar) / alpha_bar));
    }
    return out;
}

/// `ModelSamplingDiscrete.sigma(timestep)`: interpolation into an ascending ladder at
/// a fractional index, **in log space** and in f32.
///
/// ⚠️ At an integral index this is `exp(log(ladder[i]))`, which is *not* `ladder[i]` to
/// the last f32 ulp. That round trip is part of the convention, not an accident to
/// clean up — see the module doc, and CLAUDE.md's SD section for what interpolating in
/// sigma instead cost.
pub fn interpLadder(ladder: []const f32, idx: f64) f32 {
    const clamped = std.math.clamp(idx, 0, @as(f64, @floatFromInt(ladder.len - 1)));
    const lo: usize = @intFromFloat(@floor(clamped));
    const hi = @min(lo + 1, ladder.len - 1);
    const frac: f32 = @floatCast(clamped - @floor(clamped));
    const log_lo: f32 = @floatCast(@log(@as(f64, ladder[lo])));
    const log_hi: f32 = @floatCast(@log(@as(f64, ladder[hi])));
    return @floatCast(@exp(@as(f64, log_lo * (1 - frac) + log_hi * frac)));
}

/// Element `i` of `torch.linspace(start, end, n)` for an f32 tensor.
///
/// ⚠️ Two things that are not obvious from `torch.linspace`'s docs and both matter:
/// `step` is computed in the tensor's **own** dtype (f32, not the double accumulator
/// its `accscalar_t` suggests), and `start + step * i` is **FMA-contracted** by the
/// build — so `@mulAdd`, not a multiply followed by an add. The halfway split is what
/// makes both endpoints exact.
pub fn torchLinspace(start: f32, end: f32, n: usize, i: usize) f32 {
    std.debug.assert(i < n);
    if (n == 1) return start;
    const step: f32 = (end - start) / @as(f32, @floatFromInt(n - 1));
    if (i < n / 2) return @mulAdd(f32, step, @floatFromInt(i), start);
    return @mulAdd(f32, -step, @floatFromInt(n - 1 - i), end);
}

/// `math.isclose(x, 0, abs_tol=1e-5)`, which is what ComfyUI uses to decide whether a
/// table's bottom rung already *is* zero (some architectures' do).
fn isCloseToZero(x: f32) bool {
    return @abs(x) <= 1e-5;
}

// --- the model's sigma table ------------------------------------------------

/// The family-neutral view of `model_sampling.sigmas` that a scheduler reads. Holds no
/// allocation: the flux table is a closed-form function of its index, and the discrete
/// ladder is borrowed from the caller (`SdModels.sigma_ladder`).
pub const SigmaTable = union(enum) {
    /// krea2 / flux, parameterized by its shift.
    flux: f32,
    /// SD1.5 / SDXL: the ascending 1000-rung beta ladder, borrowed.
    discrete: []const f32,
    /// Lumina 2 / Z-Image (`ModelSamplingDiscreteFlow`), parameterized by its shift.
    /// Flow matching like `.flux`, but a 1000-rung table under a differently-rounded
    /// form of the same formula — see `discreteFlowSigma`.
    discrete_flow: f32,

    pub fn len(self: SigmaTable) usize {
        return switch (self) {
            .flux => flux_table_len,
            .discrete => |l| l.len,
            .discrete_flow => discrete_flow_table_len,
        };
    }

    /// Entry `i`, ascending.
    pub fn at(self: SigmaTable, i: usize) f32 {
        return switch (self) {
            .flux => |shift| fluxTableAt(shift, i),
            .discrete => |l| l[i],
            .discrete_flow => |shift| discreteFlowTableAt(shift, i),
        };
    }

    pub fn sigmaMin(self: SigmaTable) f32 {
        return self.at(0);
    }

    pub fn sigmaMax(self: SigmaTable) f32 {
        return self.at(self.len() - 1);
    }

    /// `model_sampling.timestep(sigma)` — the abscissa the table is indexed by.
    ///
    /// ⚠️ These are different *kinds* of quantity, which is why `normal` produces such
    /// different schedules per family: for flux the "timestep" IS the sigma (so
    /// `normal` applies the shift formula a second time, on purpose — that is what
    /// ComfyUI does), while for SD it is an integer training index found by an argmin
    /// in log-sigma.
    pub fn timestep(self: SigmaTable, sigma: f32) f32 {
        return switch (self) {
            // Both flow tables index by the sigma itself. For `.discrete_flow` that
            // is `sigma * multiplier` with Z-Image's multiplier of 1.0 — an identity
            // only because of that value, not a property of the class.
            .flux, .discrete_flow => sigma,
            .discrete => |l| blk: {
                const log_sigma: f32 = @log(sigma);
                var best: usize = 0;
                var best_d: f32 = @abs(log_sigma - @log(l[0]));
                for (l[1..], 1..) |v, i| {
                    const d = @abs(log_sigma - @log(v));
                    if (d < best_d) {
                        best_d = d;
                        best = i;
                    }
                }
                break :blk @floatFromInt(best);
            },
        };
    }

    /// `model_sampling.sigma(timestep)` — the inverse of `timestep`, continuous.
    pub fn sigmaOf(self: SigmaTable, t: f32) f32 {
        return switch (self) {
            .flux => |shift| fluxSigma(shift, t),
            .discrete => |l| interpLadder(l, t),
            .discrete_flow => |shift| discreteFlowSigma(shift, t),
        };
    }
};

// --- the schedulers ---------------------------------------------------------

/// ComfyUI's `SCHEDULER_NAMES`, same spellings.
pub const Scheduler = enum {
    normal,
    karras,
    exponential,
    sgm_uniform,
    simple,
    ddim_uniform,
    beta,
    linear_quadratic,
    kl_optimal,

    /// The `--scheduler` spelling and the GUI config value.
    pub fn parse(s: []const u8) ?Scheduler {
        return std.meta.stringToEnum(Scheduler, s);
    }

    pub fn label(self: Scheduler) []const u8 {
        return @tagName(self);
    }

    /// The AUTOMATIC1111 `parameters` "Schedule type" spelling, for PNG metadata.
    pub fn a1111Name(self: Scheduler) []const u8 {
        return switch (self) {
            .normal => "Normal",
            .karras => "Karras",
            .exponential => "Exponential",
            .sgm_uniform => "SGM Uniform",
            .simple => "Simple",
            .ddim_uniform => "DDIM Uniform",
            .beta => "Beta",
            .linear_quadratic => "Linear Quadratic",
            .kl_optimal => "KL Optimal",
        };
    }

    /// The scheduler a family samples with when the caller did not choose — the
    /// behaviour that predates this module, and what each ecosystem actually uses:
    /// `simple` for flow-matching checkpoints, `normal` for the SD family.
    pub fn defaultFor(table: SigmaTable) Scheduler {
        return switch (table) {
            // `simple` for both flow-matching tables: it is what ComfyUI's own
            // Z-Image Turbo template selects, alongside 8 steps and cfg 1.
            .flux, .discrete_flow => .simple,
            .discrete => .normal,
        };
    }
};

/// Build a descending sigma schedule ending at exactly 0. Caller frees.
///
/// ⚠️ **The result is not always `steps + 1` long** — see the module doc. Drive a
/// sampling loop off `result.len - 1`.
pub fn build(gpa: std.mem.Allocator, table: SigmaTable, sched: Scheduler, steps: usize) ![]f32 {
    if (steps < 1) return error.NoSteps;
    return switch (sched) {
        .simple => simple(gpa, table, steps),
        .normal => normal(gpa, table, steps, false),
        .sgm_uniform => normal(gpa, table, steps, true),
        .karras => karras(gpa, table, steps),
        .exponential => exponential(gpa, table, steps),
        .ddim_uniform => ddimUniform(gpa, table, steps),
        .beta => betaSchedule(gpa, table, steps),
        .linear_quadratic => linearQuadratic(gpa, table, steps),
        .kl_optimal => klOptimal(gpa, table, steps),
    };
}

/// `simple_scheduler`: walk the table top-down at a constant stride.
fn simple(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    const out = try gpa.alloc(f32, steps + 1);
    const n = table.len();
    const ss = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(steps));
    for (0..steps) |x| {
        const k: usize = @intFromFloat(@as(f64, @floatFromInt(x)) * ss);
        out[x] = table.at(n - 1 - k);
    }
    out[steps] = 0;
    return out;
}

/// `normal_scheduler`, and with `sgm = true` it is `sgm_uniform`: place the steps
/// uniformly in the table's own *timestep* coordinate and map each back to a sigma.
fn normal(gpa: std.mem.Allocator, table: SigmaTable, steps_in: usize, sgm: bool) ![]f32 {
    const start = table.timestep(table.sigmaMax());
    const end = table.timestep(table.sigmaMin());

    var steps = steps_in;
    var append_zero = true;
    var n: usize = undefined; // linspace length
    if (sgm) {
        // `linspace(start, end, steps + 1)[:-1]` — the last (sigma ~ 0) point dropped
        // and a hard 0 appended instead, which is what makes `sgm_uniform` end higher.
        n = steps + 1;
    } else {
        // A table whose bottom rung is already ~0 gets one more step and no appended
        // zero, since it would be a duplicate. Neither of our two families hits this;
        // it is reproduced so a third one does not need to rediscover it.
        if (isCloseToZero(table.sigmaOf(end))) {
            steps += 1;
            append_zero = false;
        }
        n = steps;
    }

    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| out[i] = table.sigmaOf(torchLinspace(start, end, n, i));
    if (append_zero) out[steps] = 0;
    return out;
}

/// `get_sigmas_karras`, rho = 7 — uniform in `sigma^(1/rho)`.
fn karras(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    const rho: f64 = 7.0;
    // ⚠️ The two endpoints are raised to 1/rho as **Python floats** (f64) and only the
    // ramp arithmetic is f32, so the narrowing happens here and not earlier.
    const min_inv_rho = std.math.pow(f64, @as(f64, table.sigmaMin()), 1.0 / rho);
    const max_inv_rho = std.math.pow(f64, @as(f64, table.sigmaMax()), 1.0 / rho);
    const max32: f32 = @floatCast(max_inv_rho);
    const span32: f32 = @floatCast(min_inv_rho - max_inv_rho);

    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| {
        const ramp = torchLinspace(0, 1, steps, i);
        const base = max32 + ramp * span32;
        out[i] = @floatCast(std.math.pow(f64, @as(f64, base), rho));
    }
    out[steps] = 0;
    return out;
}

/// `get_sigmas_exponential`: uniform in log-sigma.
fn exponential(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    const hi: f32 = @floatCast(@log(@as(f64, table.sigmaMax())));
    const lo: f32 = @floatCast(@log(@as(f64, table.sigmaMin())));
    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| {
        out[i] = @floatCast(@exp(@as(f64, torchLinspace(hi, lo, steps, i))));
    }
    out[steps] = 0;
    return out;
}

/// `ddim_scheduler`: every `len/steps`-th table entry, bottom-up, then reversed.
///
/// ⚠️ **This one's length is `len(table)/stride + 1`, which is only `steps + 1` when
/// the stride divides evenly** — 30 requested steps over the 1000-rung SD ladder gives
/// a stride of 33 and therefore 31 steps. It is the reason `build` documents that the
/// caller must read the step count back off the result.
fn ddimUniform(gpa: std.mem.Allocator, table: SigmaTable, steps_in: usize) ![]f32 {
    const n = table.len();
    var steps = steps_in;
    var sigs: std.ArrayList(f32) = .empty;
    errdefer sigs.deinit(gpa);
    if (isCloseToZero(table.at(1))) {
        steps += 1; // the table's own bottom rung serves as the trailing zero
    } else {
        try sigs.append(gpa, 0.0);
    }
    const ss = @max(n / steps, 1);
    var x: usize = 1;
    while (x < n) : (x += ss) try sigs.append(gpa, table.at(x));
    const out = try sigs.toOwnedSlice(gpa);
    std.mem.reverse(f32, out);
    return out;
}

/// `beta_scheduler` (arXiv 2407.12173), alpha = beta = 0.6: place the steps at the
/// Beta distribution's quantiles, which concentrates them at both ends.
///
/// ⚠️ **De-duplicates**, so a high step count over a short table returns fewer sigmas
/// than requested: neighbouring quantiles can round to the same table index, and
/// ComfyUI drops the repeat rather than emitting a zero-length step.
fn betaSchedule(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    const alpha: f64 = 0.6;
    const b: f64 = 0.6;
    const total: f64 = @floatFromInt(table.len() - 1);

    var sigs: std.ArrayList(f32) = .empty;
    errdefer sigs.deinit(gpa);
    // numpy's `linspace(0, 1, steps, endpoint=False)` is `i * (1/steps)`, not `i/steps`.
    const step = 1.0 / @as(f64, @floatFromInt(steps));
    var last: f64 = -1;
    for (0..steps) |i| {
        const p = 1.0 - @as(f64, @floatFromInt(i)) * step;
        // `numpy.rint` is round-half-to-EVEN, which `@round` is not.
        const t = rintEven(betaIncInv(alpha, b, p) * total);
        if (t != last) try sigs.append(gpa, table.at(@intFromFloat(t)));
        last = t;
    }
    try sigs.append(gpa, 0.0);
    return sigs.toOwnedSlice(gpa);
}

/// `linear_quadratic_schedule` (Mochi): linear over the first half, quadratic after.
fn linearQuadratic(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    const sigma_max = table.sigmaMax();
    if (steps == 1) {
        const out = try gpa.alloc(f32, 2);
        out[0] = 1.0 * sigma_max;
        out[1] = 0.0 * sigma_max;
        return out;
    }
    const threshold_noise: f64 = 0.025;
    const linear_steps = steps / 2;
    const quadratic_steps = steps - linear_steps;
    const fl: f64 = @floatFromInt(linear_steps);
    const fq: f64 = @floatFromInt(quadratic_steps);
    const diff = fl - threshold_noise * @as(f64, @floatFromInt(steps));
    const quadratic_coef = diff / (fl * fq * fq);
    const linear_coef = threshold_noise / fl - 2.0 * diff / (fq * fq);
    const constant = quadratic_coef * fl * fl;

    // `sigma_schedule` is built ascending in "noise removed", then flipped by
    // `1 - x` and scaled — so the trailing 1.0 becomes the trailing 0 sigma for free.
    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| {
        const fi: f64 = @floatFromInt(i);
        const v = if (i < linear_steps)
            fi * threshold_noise / fl
        else
            quadratic_coef * fi * fi + linear_coef * fi + constant;
        out[i] = @as(f32, @floatCast(1.0 - v)) * sigma_max;
    }
    out[steps] = @as(f32, 1.0 - 1.0) * sigma_max;
    return out;
}

/// `kl_optimal_scheduler` (A1111 PR 15608): uniform in `atan(sigma)`.
///
/// Errors on a single step rather than reproducing the reference's `0/0`: `arange(1)`
/// divided by `n - 1 = 0` is NaN there, and a NaN schedule renders a blank image with
/// no diagnostic.
fn klOptimal(gpa: std.mem.Allocator, table: SigmaTable, steps: usize) ![]f32 {
    if (steps < 2) return error.NoSteps;
    const a_min: f32 = @floatCast(std.math.atan(@as(f64, table.sigmaMin())));
    const a_max: f32 = @floatCast(std.math.atan(@as(f64, table.sigmaMax())));
    const out = try gpa.alloc(f32, steps + 1);
    for (0..steps) |i| {
        // `arange(n, dtype=float).div_(n - 1)`, not a linspace: an f32 divide.
        const adj: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps - 1));
        const arg = adj * a_min + (1.0 - adj) * a_max;
        out[i] = @floatCast(@tan(@as(f64, arg)));
    }
    out[steps] = 0;
    return out;
}

// --- the inverse incomplete beta, for the `beta` scheduler -------------------

/// `numpy.rint` — round half to **even**, which `@round` (half away from zero) is not.
fn rintEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) != 0.5) return r;
    return if (@mod(r, 2.0) == 0) r else r - std.math.sign(x);
}

/// Numerical Recipes' modified-Lentz continued fraction for the incomplete beta.
fn betaCf(a: f64, b: f64, x: f64) f64 {
    const eps = 3.0e-16;
    const fpmin = 1.0e-300;
    const qab = a + b;
    const qap = a + 1.0;
    const qam = a - 1.0;
    var c: f64 = 1.0;
    var d: f64 = 1.0 - qab * x / qap;
    if (@abs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    var h: f64 = d;
    var m: usize = 1;
    while (m <= 300) : (m += 1) {
        const fm: f64 = @floatFromInt(m);
        const m2 = 2.0 * fm;
        var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const del = d * c;
        h *= del;
        if (@abs(del - 1.0) < eps) break;
    }
    return h;
}

/// The regularized incomplete beta `I_x(a, b)` — the Beta distribution's CDF.
pub fn betaInc(a: f64, b: f64, x: f64) f64 {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const ln_beta = std.math.lgamma(f64, a + b) - std.math.lgamma(f64, a) - std.math.lgamma(f64, b);
    const front = @exp(ln_beta + a * @log(x) + b * std.math.log1p(-x));
    // Pick the branch whose continued fraction converges: the CF is only fast for
    // `x` below the distribution's mode.
    if (x < (a + 1.0) / (a + b + 2.0)) return front * betaCf(a, b, x) / a;
    return 1.0 - front * betaCf(b, a, 1.0 - x) / b;
}

/// `scipy.stats.beta.ppf` — the inverse of `betaInc` in `x`.
///
/// Plain bisection rather than a Newton scheme with an analytic seed: the CDF is
/// monotone on [0, 1] so bisection cannot fail, it converges to adjacent doubles for
/// roots of any magnitude (the `beta` scheduler's smallest quantile lands near 1e-10),
/// and at ~30 evaluations per schedule the cost is irrelevant. A Newton iteration
/// would need guarding against the infinite derivative at both ends, where a < 1.
pub fn betaIncInv(a: f64, b: f64, p: f64) f64 {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    var lo: f64 = 0;
    var hi: f64 = 1;
    var i: usize = 0;
    while (i < 300) : (i += 1) {
        const mid = 0.5 * (lo + hi);
        if (mid <= lo or mid >= hi) break; // adjacent doubles
        if (betaInc(a, b, mid) < p) lo = mid else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// --- family-default wrappers, kept for the callers that predate this module ---

/// ComfyUI's "simple" scheduler over the flux table — krea2's default.
pub fn simpleSchedule(gpa: std.mem.Allocator, steps: usize, shift: f64) ![]f32 {
    return build(gpa, .{ .flux = @floatCast(shift) }, .simple, steps);
}

/// ComfyUI's "normal" scheduler over the SD beta ladder — the SD family's default.
/// Allocates the ladder internally; `Session` passes its cached one to `build` instead.
pub fn sdSchedule(gpa: std.mem.Allocator, steps: usize) ![]f32 {
    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);
    return build(gpa, .{ .discrete = ladder }, .normal, steps);
}

/// The (fractional) training index the `i`-th step of a `steps`-step SD run samples at
/// under the `normal` scheduler: `torch.linspace(999, 0, steps)`.
///
/// ⚠️ **This is `timestep_spacing = "linspace"`, and the alternative is not a detail.**
/// diffusers' "leading" spacing (the old PNDM discretization, `steps_offset = 1`)
/// starts a 4-step run at index 751 — **sigma 4.12 instead of 14.615**. The sampler is
/// then told the latent is only moderately noisy while `scaleInitialNoise` has scaled
/// it as pure noise, so no global structure forms and the image comes out as
/// noise-textured mush. It is a *correct* implementation of the wrong convention, which
/// is why a parity test against a wrongly-configured reference passes happily.
pub fn sdTrainIndex(i: usize, steps: usize) f64 {
    const last: f32 = @floatFromInt(sd_train_steps - 1);
    return torchLinspace(last, 0, steps, i);
}

/// The **fractional** training indices the SD `normal` schedule reads at — the quantity
/// diffusers' `EulerDiscreteScheduler.timesteps` holds, and what its UNet is
/// conditioned on.
///
/// ⚠️ **This is not what *this* engine conditions on.** See `sampler.sdModelTimestep`:
/// ComfyUI snaps to the nearest trained index, and that is the convention followed
/// here. This function stays as the diffusers-side quantity, because it is what the
/// schedule fixture compares against.
pub fn sdTimesteps(gpa: std.mem.Allocator, steps: usize) ![]f32 {
    const out = try gpa.alloc(f32, steps);
    for (0..steps) |i| out[i] = @floatCast(sdTrainIndex(i, steps));
    return out;
}

// --- tests ------------------------------------------------------------------

const fixture_json = @embedFile("assets/dpmpp_sde_fixtures.json");

fn f32At(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => unreachable,
    };
}

/// A u64 out of the fixture. ⚠️ `std.json` parks an integer that does not fit `i64`
/// in `number_string`, and the Z-Image table's FNV-1a hash is exactly such a value —
/// reading `.integer` unconditionally panics on it while passing for the other two
/// tables, which is how this stayed latent until a third table was added.
fn u64At(v: std.json.Value) !u64 {
    return switch (v) {
        .integer => |n| @bitCast(n),
        .number_string => |str| std.fmt.parseInt(u64, str, 10),
        else => error.BadFixture,
    };
}

test "both sigma tables are bit-exact to model_sampling.sigmas, every entry" {
    // ⚠️ Checked by a hash over EVERY entry's raw f32 bits, not by sampling indices,
    // and that is the point: the `scalar / tensor` reciprocal convention (see the
    // module doc) moves only **165 of 10000** flux entries into a different `round6`
    // cell and 3114 by an ulp, so a seven-index spot check would miss it outright.
    // These tables are the foundation of `simple`, `ddim_uniform` and `beta` (pure
    // lookups) and of every scheduler's sigma_min/max, so a single wrong ulp here is a
    // wrong noise draw somewhere downstream.
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture_json, .{});
    defer parsed.deinit();

    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);

    var it = parsed.value.object.get("sigma_tables").?.object.iterator();
    var seen: usize = 0;
    while (it.next()) |entry| {
        const t = entry.value_ptr.object;
        const table: SigmaTable = if (std.mem.eql(u8, entry.key_ptr.*, "flux"))
            .{ .flux = default_shift }
        else if (std.mem.eql(u8, entry.key_ptr.*, "zimage"))
            .{ .discrete_flow = z_image_shift }
        else
            .{ .discrete = ladder };
        errdefer std.debug.print("table {s}\n", .{entry.key_ptr.*});

        try std.testing.expectEqual(@as(usize, @intCast(t.get("len").?.integer)), table.len());
        try std.testing.expectEqual(f32At(t.get("sigma_min").?), table.sigmaMin());
        try std.testing.expectEqual(f32At(t.get("sigma_max").?), table.sigmaMax());
        for (t.get("index").?.array.items, t.get("value").?.array.items) |iv, vv| {
            const i: usize = @intCast(iv.integer);
            errdefer std.debug.print("  sigmas[{d}]\n", .{i});
            try std.testing.expectEqual(f32At(vv), table.at(i));
        }

        // FNV-1a over the little-endian f32 bytes of the whole table.
        var h: u64 = 0xcbf29ce484222325;
        for (0..table.len()) |i| {
            const bits: u32 = @bitCast(table.at(i));
            for (std.mem.asBytes(&bits)) |byte| {
                h = (h ^ byte) *% 0x100000001b3;
            }
        }
        try std.testing.expectEqual(try u64At(t.get("bits_fnv1a").?), h);
        seen += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), seen);
}

test "every scheduler matches ComfyUI's calculate_sigmas, on both sigma tables" {
    // Generated by driving ComfyUI's own dispatcher (`tools/gen_sampler_fixtures.py`),
    // so the fixture and this implementation share no assumption.
    //
    // Two assertions per schedule, because they catch different things:
    //  - a tight RELATIVE bound (2.4e-7, ~2 f32 ulp), which every convention this
    //    module had to reverse-engineer violates by an order of magnitude or more;
    //  - the fraction of sigmas landing in ComfyUI's own `round6` cell, which is what
    //    an SDE sampler actually consumes as a Brownian-tree key. The residual is
    //    torch's vs Zig's `exp`/`log`/`pow` differing in the last ulp, so it is
    //    asserted as a fraction rather than a count.
    const gpa = std.testing.allocator;
    const brownian = @import("brownian.zig");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture_json, .{});
    defer parsed.deinit();

    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);

    var it = parsed.value.object.get("schedulers").?.object.iterator();
    var checked: usize = 0;
    while (it.next()) |entry| {
        // key is "<family>|<scheduler>|<steps>"
        var parts = std.mem.splitScalar(u8, entry.key_ptr.*, '|');
        const fam = parts.next().?;
        const name = parts.next().?;
        const steps = try std.fmt.parseInt(usize, parts.next().?, 10);
        const table: SigmaTable = if (std.mem.eql(u8, fam, "flux"))
            .{ .flux = default_shift }
        else if (std.mem.eql(u8, fam, "zimage"))
            .{ .discrete_flow = z_image_shift }
        else
            .{ .discrete = ladder };
        const sched = Scheduler.parse(name).?;

        const want = entry.value_ptr.array.items;
        const got = try build(gpa, table, sched, steps);
        defer gpa.free(got);
        errdefer std.debug.print("{s}: want {d} sigmas, got {d}\n", .{ entry.key_ptr.*, want.len, got.len });
        try std.testing.expectEqual(want.len, got.len);

        var same_cell: usize = 0;
        for (want, got, 0..) |w, a, i| {
            const e = f32At(w);
            errdefer std.debug.print("{s} sigma[{d}]: ComfyUI {d:.9} got {d:.9}\n", .{ entry.key_ptr.*, i, e, a });
            if (e == 0) {
                try std.testing.expectEqual(@as(f32, 0), a);
            } else {
                // ~3.4 f32 ulp, which is the honest floor: `interpLadder` mixes two
                // logarithms and exponentiates, so torch's vectorized `logf` differing
                // from a correctly-rounded one in the last ulp lands ~3e-7 out (max
                // observed across all 72 combos: 3.0e-7, at sd|sgm_uniform|30). Still an
                // order of magnitude tighter than the conventions this module had to
                // reverse-engineer, which were 4.1e-6 relative and up — so it has teeth.
                try std.testing.expectApproxEqRel(e, a, 4e-7);
            }
            if (brownian.round6(e) == brownian.round6(a)) same_cell += 1;
        }
        errdefer std.debug.print("{s}: only {d}/{d} sigmas in ComfyUI's round6 cell\n", .{ entry.key_ptr.*, same_cell, want.len });
        // Measured on this machine: 68 of the 72 combos put EVERY sigma in ComfyUI's
        // own cell; the other four (sd sgm_uniform at 4/20/30 steps, sd kl_optimal at
        // 20) miss exactly one, from torch's vs Zig's `log`/`tan` differing in the last
        // ulp. Bounded as a count so a libm shift is not a failure, while a reverted
        // convention — which misplaced 18 of 21 when it happened — still is.
        try std.testing.expect(want.len - same_cell <= 1 + want.len / 8);
        checked += 1;
    }
    // 9 schedulers x 3 tables x 4 step counts; a silently-empty fixture would
    // otherwise make this test pass by doing nothing.
    try std.testing.expectEqual(@as(usize, 108), checked);
}

test "the inverse incomplete beta matches scipy" {
    // The `beta` scheduler's quantiles. Checked over a > 1 and a < 1 on both
    // parameters, since the continued fraction takes a different branch either side of
    // the mode and `a < 1` is where the density is unbounded at 0.
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture_json, .{});
    defer parsed.deinit();

    for (parsed.value.object.get("betaincinv").?.array.items) |c| {
        const o = c.object;
        const a = o.get("a").?.float;
        const b = o.get("b").?.float;
        const p = o.get("p").?.float;
        const want = o.get("x").?.float;
        const got = betaIncInv(a, b, p);
        errdefer std.debug.print("betaincinv(a={d}, b={d}, p={d}): scipy {d:.17} got {d:.17}\n", .{ a, b, p, want, got });
        if (want == 0 or want == 1) {
            try std.testing.expectEqual(want, got);
        } else {
            try std.testing.expectApproxEqRel(want, got, 1e-11);
        }
        // Round-trip through the forward CDF too, so a fixture typo cannot make both
        // sides agree on a wrong answer.
        if (got > 1e-8 and got < 1.0 - 1e-8) {
            try std.testing.expectApproxEqRel(p, betaInc(a, b, got), 1e-9);
        }
    }
}

test "numpy.rint rounds half to even, unlike @round" {
    try std.testing.expectEqual(@as(f64, 2.0), rintEven(2.5));
    try std.testing.expectEqual(@as(f64, 4.0), rintEven(3.5));
    try std.testing.expectEqual(@as(f64, 0.0), rintEven(0.5));
    try std.testing.expectEqual(@as(f64, -2.0), rintEven(-2.5));
    try std.testing.expectEqual(@as(f64, 3.0), rintEven(2.7));
    try std.testing.expectEqual(@as(f64, 2.0), rintEven(2.3));
    // The case that matters: @round would give 3.0 here and select a different rung.
    try std.testing.expect(@round(@as(f64, 2.5)) != rintEven(2.5));
}

test "torch.linspace hits both endpoints exactly and is FMA-contracted" {
    // The halfway split is what makes the endpoints exact; the FMA is what makes the
    // interior match. A plain multiply-then-add differs on a log-space ramp.
    try std.testing.expectEqual(@as(f32, 999), torchLinspace(999, 0, 20, 0));
    try std.testing.expectEqual(@as(f32, 0), torchLinspace(999, 0, 20, 19));
    try std.testing.expectEqual(@as(f32, 0), torchLinspace(0, 1, 8, 0));
    try std.testing.expectEqual(@as(f32, 1), torchLinspace(0, 1, 8, 7));
    try std.testing.expectEqual(@as(f32, 5), torchLinspace(5, 9, 1, 0)); // n == 1: start
    // Monotone, and strictly inside for interior points.
    var prev: f32 = 1000;
    for (0..30) |i| {
        const v = torchLinspace(999, 0, 30, i);
        try std.testing.expect(v < prev);
        prev = v;
    }
}

test "schedules are descending, end at exactly zero, and stay in the table's range" {
    // A shape invariant every scheduler owes the sampling loop, independent of the
    // fixture: `generate` steps `sigmas[i] -> sigmas[i+1]` and `decode` assumes the
    // trajectory finished at 0.
    const gpa = std.testing.allocator;
    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);

    for ([_]SigmaTable{ .{ .flux = default_shift }, .{ .discrete = ladder } }) |table| {
        for (comptime std.enums.values(Scheduler)) |sched| {
            for ([_]usize{ 2, 3, 7, 20, 50 }) |steps| {
                const s = try build(gpa, table, sched, steps);
                defer gpa.free(s);
                errdefer std.debug.print("{t} / {t} at {d} steps\n", .{ table, sched, steps });
                try std.testing.expect(s.len >= 2);
                try std.testing.expectEqual(@as(f32, 0), s[s.len - 1]);
                // ⚠️ Not `<= sigmaMax()`: `karras` computes `(sigma_max^(1/rho))^rho`,
                // whose round trip can land an ulp ABOVE sigma_max (it does, at 2
                // steps). That is the reference's own behaviour, and harmless — the
                // model is evaluated slightly off the top of its trained range, which is
                // exactly what `sdTimestepForSigma`'s clamp is for.
                try std.testing.expect(s[0] > 0 and s[0] <= table.sigmaMax() * 1.001);
                for (s[0 .. s.len - 1], s[1..]) |a, b| {
                    try std.testing.expect(std.math.isFinite(a));
                    try std.testing.expect(a > b); // strictly descending
                }
            }
        }
    }
}

test "the family defaults are the behaviour that predates the scheduler choice" {
    // krea2 sampled `simple` and the SD family sampled `normal` before either was
    // selectable; `defaultFor` has to keep that, or every existing render changes.
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(Scheduler.simple, Scheduler.defaultFor(.{ .flux = default_shift }));
    try std.testing.expectEqual(Scheduler.normal, Scheduler.defaultFor(.{ .discrete = &.{} }));

    const ladder = try sdSigmasFull(gpa);
    defer gpa.free(ladder);
    {
        const a = try simpleSchedule(gpa, 8, default_shift);
        defer gpa.free(a);
        const b = try build(gpa, .{ .flux = default_shift }, .simple, 8);
        defer gpa.free(b);
        try std.testing.expectEqualSlices(f32, a, b);
    }
    {
        const a = try sdSchedule(gpa, 8);
        defer gpa.free(a);
        const b = try build(gpa, .{ .discrete = ladder }, .normal, 8);
        defer gpa.free(b);
        try std.testing.expectEqualSlices(f32, a, b);
    }
}

test "the flow-matching simple schedule matches the golden ComfyUI values" {
    // Predates this module; kept verbatim so the f32 table rework cannot silently
    // shift krea2's default schedule.
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

test "the SD sigma ladder and normal schedule match diffusers' EulerDiscreteScheduler" {
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

    // ⚠️ The 2e-4 tolerance below is the whole reason this file also has a test against
    // ComfyUI's `normal_scheduler`: diffusers lerps sigma where ComfyUI lerps its
    // logarithm, which is a 4.4e-5 difference this bound cannot see. Do not tighten it
    // (the log/exp round trip forces it) and do not treat it as the parity check.
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
