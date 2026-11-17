//! Two exponential-integrator samplers that keep the previous step's estimate and
//! draw from the latent device's generator (`default_noise_sampler`):
//!
//!  - `res_multistep` (eta 0, draws nothing) and `res_multistep_ancestral` (eta 1):
//!    second order from the previous estimate, after `get_ancestral_step`'s split
//!    of the sigmas. No `CONST` dispatch, so `t = -log(sigma)` on every family.
//!  - `er_sde`: the three-stage ER-SDE solver. It DOES dispatch on `CONST`: it
//!    offsets the first sigma and works in `sigma / alpha`, which is `sigma` for
//!    eps and `sigma / (1 - sigma)` for flow.
const std = @import("std");
const noise = @import("noise.zig");
const sampler = @import("sampler.zig");

pub const Options = struct {
    eta: f64 = 0,
    s_noise: f64 = 1,
    seed: u64 = 0,
    noise_src: noise.Source = .nv_philox,
};

pub const ResStepper = struct {
    gpa: std.mem.Allocator,
    sigmas: []const f32,
    eta: f64,
    s_noise: f64,
    rng: noise.Generator,
    denoised: []f32,
    old_denoised: []f32,
    noise_buf: []f32,
    have_old: bool = false,
    old_sigma_down: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, sigmas: []const f32, n: usize, opts: Options) !ResStepper {
        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const old_denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(old_denoised);
        const noise_buf = try gpa.alloc(f32, n);
        return .{
            .gpa = gpa,
            .sigmas = sigmas,
            .eta = opts.eta,
            .s_noise = opts.s_noise,
            .rng = .init(opts.seed, opts.noise_src),
            .denoised = denoised,
            .old_denoised = old_denoised,
            .noise_buf = noise_buf,
        };
    }

    pub fn deinit(self: *ResStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.old_denoised);
        self.gpa.free(self.noise_buf);
        self.* = undefined;
    }

    fn draws(self: *const ResStepper, i: usize) bool {
        return sampler.ancestralStep(self.sigmas[i], self.sigmas[i + 1], self.eta).up > 0;
    }

    pub fn step(self: *ResStepper, x: []f32, v: []const f32, i: usize) void {
        const sigma = self.sigmas[i];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }
        const split = sampler.ancestralStep(sigma, self.sigmas[i + 1], self.eta);

        if (split.down == 0 or !self.have_old) {
            const dt: f32 = @floatCast(split.down - sigma);
            for (x, v) |*xi, d| xi.* += d * dt;
        } else {
            const t = -@log(@as(f64, sigma));
            const t_old = -@log(self.old_sigma_down);
            const t_next = -@log(split.down);
            const t_prev = -@log(@as(f64, self.sigmas[i - 1]));
            const h = t_next - t;
            const c2 = (t_prev - t_old) / h;
            const phi1 = std.math.expm1(-h) / -h;
            const phi2 = (phi1 - 1) / -h;
            const b1 = nanToZero(phi1 - phi2 / c2);
            const b2 = nanToZero(phi2 / c2);
            const cx: f32 = @floatCast(@exp(-h));
            const hb1: f32 = @floatCast(b1);
            const hb2: f32 = @floatCast(b2);
            const hh: f32 = @floatCast(h);
            for (x, self.denoised, self.old_denoised) |*xi, d, od| xi.* = cx * xi.* + hh * (hb1 * d + hb2 * od);
        }

        if (split.up > 0) {
            self.rng.randn(self.noise_buf);
            const k: f32 = @floatCast(self.s_noise);
            const up: f32 = @floatCast(split.up);
            for (x, self.noise_buf) |*xi, z| xi.* += z * k * up;
        }

        @memcpy(self.old_denoised, self.denoised);
        self.old_sigma_down = split.down;
        self.have_old = true;
    }

    pub fn fastForward(self: *ResStepper, steps_done: usize) void {
        for (0..@min(steps_done, self.sigmas.len - 1)) |i| {
            if (self.draws(i)) self.rng.randn(self.noise_buf);
        }
    }

    /// `old_sigma_down` is `sigmas[i - 1]`'s down-step, recomputed rather than carried.
    pub fn restore(self: *ResStepper, start_step: usize, old: []const f32) void {
        if (start_step == 0) return;
        @memcpy(self.old_denoised, old);
        self.old_sigma_down = sampler.ancestralStep(self.sigmas[start_step - 1], self.sigmas[start_step], self.eta).down;
        self.have_old = true;
    }
};

fn nanToZero(v: f64) f64 {
    return if (std.math.isNan(v)) 0 else v;
}

/// `x * (exp(x^0.3) + 10)`, ER-SDE's default noise scaler.
fn erScaler(x: f64) f64 {
    return x * (@exp(std.math.pow(f64, x, 0.3)) + 10);
}

const er_points = 200;

pub const ErSdeStepper = struct {
    gpa: std.mem.Allocator,
    sigmas: []f32,
    s_noise: f64,
    rng: noise.Generator,
    /// `sigma / alpha` per schedule entry.
    lambdas: []f64,
    denoised: []f32,
    old_denoised: []f32,
    old_d: []f32,
    noise_buf: []f32,
    /// Previous estimates held: 1 for `old_denoised`, 2 once `old_d` is valid too.
    n_old: usize = 0,

    /// Offsets `sigmas[0]` on a flow model, as `SdeStepper` does.
    pub fn init(gpa: std.mem.Allocator, sigmas: []f32, n: usize, param: sampler.Parameterization, opts: Options, table: sampler.SigmaTable) !ErSdeStepper {
        _ = sampler.offsetFirstSigma(sigmas, param, table);
        const lambdas = try gpa.alloc(f64, sigmas.len);
        errdefer gpa.free(lambdas);
        for (lambdas, sigmas) |*l, s| l.* = @exp(-param.halfLogSnr(s));
        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const old_denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(old_denoised);
        const old_d = try gpa.alloc(f32, n);
        errdefer gpa.free(old_d);
        const noise_buf = try gpa.alloc(f32, n);
        return .{
            .gpa = gpa,
            .sigmas = sigmas,
            .s_noise = opts.s_noise,
            .rng = .init(opts.seed, opts.noise_src),
            .lambdas = lambdas,
            .denoised = denoised,
            .old_denoised = old_denoised,
            .old_d = old_d,
            .noise_buf = noise_buf,
        };
    }

    pub fn deinit(self: *ErSdeStepper) void {
        self.gpa.free(self.lambdas);
        self.gpa.free(self.denoised);
        self.gpa.free(self.old_denoised);
        self.gpa.free(self.old_d);
        self.gpa.free(self.noise_buf);
        self.* = undefined;
    }

    fn draws(self: *const ErSdeStepper, i: usize) bool {
        return self.s_noise > 0 and self.sigmas[i + 1] != 0;
    }

    pub fn step(self: *ErSdeStepper, x: []f32, v: []const f32, i: usize) void {
        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }
        const stage = @min(3, i + 1, self.n_old + 1);

        if (sigma_next == 0) {
            @memcpy(x, self.denoised);
        } else {
            const ls = self.lambdas[i];
            const lt = self.lambdas[i + 1];
            const alpha_s = sigma / ls;
            const alpha_t = sigma_next / lt;
            const r = erScaler(lt) / erScaler(ls);
            {
                const cx: f32 = @floatCast(alpha_t / alpha_s * r);
                const cd: f32 = @floatCast(alpha_t * (1 - r));
                for (x, self.denoised) |*xi, d| xi.* = cx * xi.* + cd * d;
            }

            if (stage >= 2) {
                const dt = lt - ls;
                const h = -dt / er_points;
                var s1: f64 = 0;
                var s_u: f64 = 0;
                for (0..er_points) |k| {
                    const pos = lt + @as(f64, @floatFromInt(k)) * h;
                    const sc = erScaler(pos);
                    s1 += 1 / sc;
                    s_u += (pos - ls) / sc;
                }
                s1 *= h;
                s_u *= h;

                const inv_d: f32 = @floatCast(1 / (ls - self.lambdas[i - 1]));
                const c2: f32 = @floatCast(alpha_t * (dt + s1 * erScaler(lt)));
                for (x, self.denoised, self.old_denoised, self.old_d) |*xi, d, od, *dd| {
                    const nd = (d - od) * inv_d;
                    xi.* += c2 * nd;
                    if (stage >= 3) dd.* = nd - dd.* else dd.* = nd;
                }
                if (stage >= 3) {
                    // `old_d` holds `nd - old_d` here; finish stage 3, then put `nd` back.
                    const inv_u: f32 = @floatCast(2 / (ls - self.lambdas[i - 2]));
                    const c3: f32 = @floatCast(alpha_t * (dt * dt / 2 + s_u * erScaler(lt)));
                    for (x, self.denoised, self.old_denoised, self.old_d) |*xi, d, od, *dd| {
                        xi.* += c3 * (dd.* * inv_u);
                        dd.* = (d - od) * inv_d;
                    }
                }
            }

            if (self.draws(i)) {
                self.rng.randn(self.noise_buf);
                const amp = @sqrt(lt * lt - ls * ls * r * r);
                const k: f32 = @floatCast(alpha_t * self.s_noise * nanToZero(amp));
                for (x, self.noise_buf) |*xi, z| xi.* += k * z;
            }
            if (stage >= 2) self.n_old = 2;
        }

        @memcpy(self.old_denoised, self.denoised);
        self.n_old = @max(self.n_old, 1);
    }

    pub fn fastForward(self: *ErSdeStepper, steps_done: usize) void {
        for (0..@min(steps_done, self.sigmas.len - 1)) |i| {
            if (self.draws(i)) self.rng.randn(self.noise_buf);
        }
    }

    pub fn restore(self: *ErSdeStepper, old: []const f32, old_d: ?[]const f32) void {
        @memcpy(self.old_denoised, old);
        self.n_old = 1;
        if (old_d) |d| {
            @memcpy(self.old_d, d);
            self.n_old = 2;
        }
    }
};
