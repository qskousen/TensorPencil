//! UniPC (`uni_pc`, `uni_pc_bh2`), ComfyUI's `extra_samplers/uni_pc.py` as its
//! `sample_unipc` drives it: multistep, data prediction, order `min(3, steps - 1)`,
//! lower order for the last steps.
//!
//! The reference does not run in the loop's space. It divides the latent by
//! `sqrt(1 + sigma^2)` (VP space), replaces a final sigma of 0 with 0.001, and
//! divides by that sigma's alpha at the end, so the result is NOT `denoised` exactly.
//! Its one model call per step is the corrector's, at the PREDICTED point for the
//! next sigma; so step `i` here first corrects the update to `sigmas[i]` with the
//! loop's forward there, then predicts the update to `sigmas[i + 1]` and hands the
//! loop that point to evaluate.
//!
//! The schedule functions are ComfyUI's f32 tensor arithmetic: near sigma 0.001
//! `1 - exp(2 * log_alpha)` cancels, and exact math misses ComfyUI's result by
//! about 1e-5 of its scale, twice the trajectory fixture's bound.
const std = @import("std");

pub const Variant = enum { bh1, bh2 };

pub const max_order = 3;
/// The sigma ComfyUI substitutes for a final 0.
const t_final: f32 = 0.001;

pub const UniPcStepper = struct {
    gpa: std.mem.Allocator,
    variant: Variant,
    sigmas: []const f32,
    order: usize,
    denoised: []f32,
    /// The current point in VP space, as predicted for `sigmas[i]` before step `i`.
    x_vp: []f32,
    /// The VP point the pending update started from; its corrector recomputes from it.
    x_prev: []f32,
    /// Model x0 predictions, newest first, at `sigmas[i - 1], sigmas[i - 2], ...`.
    models: [max_order][]f32,
    n_models: usize = 0,
    /// The newest forward's x0 prediction until it joins `models`.
    model_t: []f32,

    pub fn init(gpa: std.mem.Allocator, variant: Variant, sigmas: []const f32, n: usize) !UniPcStepper {
        std.debug.assert(sigmas.len >= 2);
        const steps = sigmas.len - 1;
        var bufs: [4 + max_order][]f32 = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |b| gpa.free(b);
        for (&bufs) |*b| {
            b.* = try gpa.alloc(f32, n);
            made += 1;
        }
        return .{
            .gpa = gpa,
            .variant = variant,
            .sigmas = sigmas,
            .order = @min(max_order, steps - 1),
            .denoised = bufs[0],
            .x_vp = bufs[1],
            .x_prev = bufs[2],
            .model_t = bufs[3],
            .models = bufs[4..][0..max_order].*,
        };
    }

    pub fn deinit(self: *UniPcStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.x_vp);
        self.gpa.free(self.x_prev);
        self.gpa.free(self.model_t);
        for (self.models) |m| self.gpa.free(m);
        self.* = undefined;
    }

    fn t(self: *const UniPcStepper, k: usize) f32 {
        return if (k == self.sigmas.len - 1 and self.sigmas[k] == 0) t_final else self.sigmas[k];
    }

    /// The order of the update to `sigmas[k]`.
    fn orderAt(self: *const UniPcStepper, k: usize) usize {
        const steps = self.sigmas.len - 1;
        return if (k < self.order) k else @min(self.order, steps + 1 - k);
    }

    pub fn step(self: *UniPcStepper, x: []f32, v: []const f32, i: usize) void {
        const steps = self.sigmas.len - 1;
        {
            const s: f32 = self.sigmas[i];
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }
        if (i == 0) {
            const den = f32Sqrt(1.0 + self.t(0) * self.t(0));
            for (self.x_vp, x) |*p, xi| p.* = xi / den;
            self.n_models = 0;
        }

        // The forward at the predicted point, as an x0 prediction.
        const ti = self.t(i);
        const alpha_i = marginalAlpha(ti);
        const std_i = marginalStd(ti);
        const model_t = self.model_t;
        for (model_t, self.x_vp, v) |*m, xv, e| m.* = (xv - std_i * e) / alpha_i;

        if (i > 0) self.update(i, self.orderAt(i), self.x_prev, model_t, self.x_vp);

        // Newest first; the oldest buffer becomes the next scratch.
        const oldest = self.models[max_order - 1];
        var j: usize = max_order - 1;
        while (j > 0) : (j -= 1) self.models[j] = self.models[j - 1];
        self.models[0] = model_t;
        self.model_t = oldest;
        self.n_models = @min(self.n_models + 1, @max(self.order, 1));

        const k = i + 1;
        if (self.order > 0) {
            @memcpy(self.x_prev, self.x_vp);
            self.update(k, self.orderAt(k), self.x_prev, null, self.x_vp);
        }
        if (k == steps) {
            const a = marginalAlpha(self.t(k));
            for (x, self.x_vp) |*xi, p| xi.* = p / a;
        } else {
            const tk = self.t(k);
            const sc = f32Pow05(tk * tk + 1.0);
            for (x, self.x_vp) |*xi, p| xi.* = p * sc;
        }
    }

    /// `multistep_uni_pc_bh_update` to `sigmas[k]` from `x_in`, the predictor when
    /// `model_t` is null and the corrector with it.
    fn update(self: *const UniPcStepper, k: usize, o: usize, x_in: []const f32, model_t: ?[]const f32, out: []f32) void {
        std.debug.assert(o >= 1 and o <= self.n_models);
        const tt = self.t(k);
        const tp0 = self.t(k - 1);
        const lambda_p0 = marginalLambda(tp0);
        const lambda_t = marginalLambda(tt);
        const sigma_p0 = marginalStd(tp0);
        const sigma_t = marginalStd(tt);
        const alpha_t = f32Exp(marginalLogMean(tt));
        const h = lambda_t - lambda_p0;

        var rks: [max_order]f32 = undefined;
        for (1..o) |q| rks[q - 1] = (marginalLambda(self.t(k - 1 - q)) - lambda_p0) / h;
        rks[o - 1] = 1;

        const hh = -h;
        const h_phi_1 = f32Expm1(hh);
        var h_phi_k = h_phi_1 / hh - 1;
        const b_h: f32 = switch (self.variant) {
            .bh1 => hh,
            .bh2 => f32Expm1(hh),
        };
        var r_mat: [max_order][max_order]f64 = undefined;
        var b: [max_order]f64 = undefined;
        var factorial: f32 = 1;
        for (1..o + 1) |q| {
            for (0..o) |c| r_mat[q - 1][c] = std.math.pow(f32, rks[c], @floatFromInt(q - 1));
            b[q - 1] = h_phi_k * factorial / b_h;
            factorial *= @floatFromInt(q + 1);
            h_phi_k = h_phi_k / hh - 1 / factorial;
        }

        // Weights on D1s (and, for the corrector, on D1_t last).
        var rhos: [max_order]f32 = @splat(0);
        const n_d1 = o - 1;
        if (model_t != null) {
            if (o == 1) rhos[0] = 0.5 else solve(o, r_mat, b, &rhos);
        } else if (n_d1 > 0) {
            if (o == 2) rhos[0] = 0.5 else solve(o - 1, r_mat, b, &rhos);
        }

        const c_x = sigma_t / sigma_p0;
        const c_m = alpha_t * h_phi_1;
        const c_r = alpha_t * b_h;
        const m0 = self.models[0];
        for (out, x_in, m0, 0..) |*y, xv, mp0, e| {
            const base = c_x * xv - c_m * mp0;
            var res: f32 = 0;
            for (0..n_d1) |q| res += rhos[q] * ((self.models[q + 1][e] - mp0) / rks[q]);
            if (model_t) |mt| res += rhos[o - 1] * (mt[e] - mp0);
            y.* = base - c_r * res;
        }
    }

    /// For a resume at step `i`: the predicted point, the update's start, and the
    /// models, newest first.
    pub fn history(self: *const UniPcStepper, out: [][]const f32) usize {
        out[0] = self.x_vp;
        out[1] = self.x_prev;
        for (self.models[0..self.n_models], 0..) |m, q| out[2 + q] = m;
        return 2 + self.n_models;
    }

    pub fn restore(self: *UniPcStepper, bufs: []const []const f32) void {
        @memcpy(self.x_vp, bufs[0]);
        @memcpy(self.x_prev, bufs[1]);
        self.n_models = bufs.len - 2;
        for (bufs[2..], 0..) |m, q| @memcpy(self.models[q], m);
    }
};

/// `torch.linalg.solve` on the leading `n` x `n` block, partial pivoting, in f64.
fn solve(n: usize, a_in: [max_order][max_order]f64, b_in: [max_order]f64, out: *[max_order]f32) void {
    var a = a_in;
    var b = b_in;
    for (0..n) |c| {
        var p = c;
        for (c + 1..n) |r| if (@abs(a[r][c]) > @abs(a[p][c])) {
            p = r;
        };
        std.mem.swap([max_order]f64, &a[c], &a[p]);
        std.mem.swap(f64, &b[c], &b[p]);
        for (c + 1..n) |r| {
            const f = a[r][c] / a[c][c];
            for (c..n) |q| a[r][q] -= f * a[c][q];
            b[r] -= f * b[c];
        }
    }
    var x: [max_order]f64 = undefined;
    var r = n;
    while (r > 0) {
        r -= 1;
        var s = b[r];
        for (r + 1..n) |q| s -= a[r][q] * x[q];
        x[r] = s / a[r][r];
    }
    for (0..n) |q| out[q] = @floatCast(x[q]);
}

// The reference's f32 tensor ops. Each transcendental is computed in f64 and
// rounded once, which is what a correctly rounded libm returns.
fn f32Log(v: f32) f32 {
    return @floatCast(@log(@as(f64, v)));
}
fn f32Exp(v: f32) f32 {
    return @floatCast(@exp(@as(f64, v)));
}
fn f32Expm1(v: f32) f32 {
    return @floatCast(std.math.expm1(@as(f64, v)));
}
fn f32Sqrt(v: f32) f32 {
    return @sqrt(v);
}
fn f32Pow05(v: f32) f32 {
    return @floatCast(std.math.pow(f64, v, 0.5));
}

fn marginalLogMean(s: f32) f32 {
    return 0.5 * f32Log(1 / ((s * s) + 1));
}
fn marginalAlpha(s: f32) f32 {
    return f32Exp(marginalLogMean(s));
}
fn marginalStd(s: f32) f32 {
    return f32Sqrt(1.0 - f32Exp(2.0 * marginalLogMean(s)));
}
fn marginalLambda(s: f32) f32 {
    const lm = marginalLogMean(s);
    return lm - 0.5 * f32Log(1.0 - f32Exp(2.0 * lm));
}

test "solve inverts a pivoting 3x3" {
    const a: [max_order][max_order]f64 = .{ .{ 0, 1, 2 }, .{ 1, 0, 3 }, .{ 4, -3, 8 } };
    var x: [max_order]f32 = undefined;
    // a * [1, 2, 3] = [8, 10, 22]
    solve(3, a, .{ 8, 10, 22 }, &x);
    for (x, [_]f32{ 1, 2, 3 }) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-5);
}
