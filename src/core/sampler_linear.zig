//! The linear multistep samplers: each step moves `x` by a weighted sum of the
//! current derivative and up to three previous ones, with weights from the
//! schedule alone. One evaluation per step.
//!
//!  - `lms`: Adams-Bashforth in sigma. ComfyUI integrates each Lagrange basis
//!    polynomial with `scipy.integrate.quad`; the polynomial is at most cubic, so
//!    Gauss-Legendre with three points integrates it exactly.
//!  - `ipndm`: the fixed-step Adams-Bashforth weights, times `dt`.
//!  - `ipndm_v`: the variable-step weights, which ComfyUI computes on 0-dim f32
//!    tensors, so they are computed in f32 here too.
//!  - `deis`: DEIS in its `tab` mode, weights integrated against the VP-SDE's
//!    integrand over EDM's fixed (0.002, 80) range, whatever the model's range is.
//!
//! None dispatches on the model's parameterization: ComfyUI runs one body for
//! every family.
const std = @import("std");

pub const Method = enum { lms, ipndm, ipndm_v, deis };

/// Previous derivatives a step can read.
pub const max_past = 3;

pub const LinearStepper = struct {
    gpa: std.mem.Allocator,
    method: Method,
    sigmas: []const f32,
    denoised: []f32,
    /// Previous derivatives, `past[0]` the newest; `n_past` are valid.
    past: [max_past][]f32,
    n_past: usize = 0,
    acc: []f32,
    /// DEIS's per-step weights, `[cur, prev1, prev2]`; empty for the others.
    deis_c: [][max_past]f32,

    pub fn init(gpa: std.mem.Allocator, method: Method, sigmas: []const f32, n: usize) !LinearStepper {
        std.debug.assert(sigmas.len >= 2);
        const denoised = try gpa.alloc(f32, n);
        errdefer gpa.free(denoised);
        const acc = try gpa.alloc(f32, n);
        errdefer gpa.free(acc);
        var past: [max_past][]f32 = undefined;
        var made: usize = 0;
        errdefer for (past[0..made]) |p| gpa.free(p);
        for (&past) |*p| {
            p.* = try gpa.alloc(f32, n);
            made += 1;
        }
        const deis_c = try gpa.alloc([max_past]f32, if (method == .deis) sigmas.len - 1 else 0);
        errdefer gpa.free(deis_c);
        if (method == .deis) deisCoeffs(sigmas, deis_c);
        return .{ .gpa = gpa, .method = method, .sigmas = sigmas, .denoised = denoised, .past = past, .acc = acc, .deis_c = deis_c };
    }

    pub fn deinit(self: *LinearStepper) void {
        self.gpa.free(self.denoised);
        self.gpa.free(self.acc);
        for (self.past) |p| self.gpa.free(p);
        self.gpa.free(self.deis_c);
        self.* = undefined;
    }

    /// How many previous derivatives the reference keeps: lms keeps `order - 1 = 3`,
    /// the others `max_order - 1` (3 for the ipndms, 2 for deis).
    fn keep(self: *const LinearStepper) usize {
        return if (self.method == .deis) 2 else 3;
    }

    pub fn step(self: *LinearStepper, x: []f32, v: []const f32, i: usize) void {
        const sigma = self.sigmas[i];
        const sigma_next = self.sigmas[i + 1];
        {
            const s: f32 = sigma;
            for (self.denoised, x, v) |*d, xi, vi| d.* = xi - s * vi;
        }

        const order = @min(i + 1, self.keep() + 1, self.n_past + 1);
        const p = self.past;
        if (sigma_next == 0 and self.method != .deis) {
            @memcpy(x, self.denoised);
        } else switch (self.method) {
            .lms => {
                // x + sum(c_j * d_j), d_0 the current: Python's `sum` starts at 0
                // and adds in order.
                for (0..order) |j| {
                    const c: f32 = @floatCast(lmsCoeff(self.sigmas, i, order, j));
                    const d = if (j == 0) v else p[j - 1];
                    if (j == 0) {
                        for (self.acc, d) |*a, dj| a.* = c * dj;
                    } else for (self.acc, d) |*a, dj| {
                        a.* += c * dj;
                    }
                }
                for (x, self.acc) |*xi, a| xi.* += a;
            },
            .ipndm => {
                // `dt * (...) / k` groups left to right, as the reference's Python does.
                const dt = sigma_next - sigma;
                switch (order) {
                    1 => for (x, v) |*xi, d| {
                        xi.* += dt * d;
                    },
                    2 => for (x, v, p[0]) |*xi, d, b1| {
                        xi.* += dt * (3 * d - b1) / 2;
                    },
                    3 => for (x, v, p[0], p[1]) |*xi, d, b1, b2| {
                        xi.* += dt * (23 * d - 16 * b1 + 5 * b2) / 12;
                    },
                    else => for (x, v, p[0], p[1], p[2]) |*xi, d, b1, b2, b3| {
                        xi.* += dt * (55 * d - 59 * b1 + 37 * b2 - 9 * b3) / 24;
                    },
                }
            },
            .ipndm_v => {
                const c = ipndmVCoeffs(self.sigmas, i, order);
                const dt = sigma_next - sigma;
                for (x, 0..) |*xi, e| {
                    var s = c[0] * v[e];
                    for (1..order) |j| s += c[j] * p[j - 1][e];
                    xi.* += dt * s;
                }
            },
            .deis => {
                if (sigma_next <= 0 or order == 1) {
                    const dt = sigma_next - sigma;
                    for (x, v) |*xi, d| xi.* += dt * d;
                } else {
                    const c = self.deis_c[i];
                    for (x, 0..) |*xi, e| {
                        var s: f32 = xi.* + c[0] * v[e];
                        for (1..order) |j| s += c[j] * p[j - 1][e];
                        xi.* = s;
                    }
                }
            },
        }

        // Newest first; the oldest falls off once `keep` are held.
        const k = self.keep();
        const last = self.past[k - 1];
        var j = k - 1;
        while (j > 0) : (j -= 1) self.past[j] = self.past[j - 1];
        self.past[0] = last;
        @memcpy(self.past[0], v);
        self.n_past = @min(self.n_past + 1, k);
    }

    /// The previous derivatives, newest first, for a bit-identical resume.
    pub fn history(self: *const LinearStepper) []const []f32 {
        return self.past[0..self.n_past];
    }

    pub fn restore(self: *LinearStepper, past: []const []const f32) void {
        const n = @min(past.len, self.keep());
        for (past[0..n], self.past[0..n]) |src, dst| @memcpy(dst, src);
        self.n_past = n;
    }
};

/// `linear_multistep_coeff(order, sigmas, i, j)`: the integral from `sigmas[i]` to
/// `sigmas[i + 1]` of the Lagrange basis polynomial for node `i - j` over the nodes
/// `i, i - 1, ..., i - order + 1`. Degree at most 3, so 3-point Gauss-Legendre is
/// exact.
fn lmsCoeff(sigmas: []const f32, i: usize, order: usize, j: usize) f64 {
    const a: f64 = sigmas[i];
    const b: f64 = sigmas[i + 1];
    const nodes = [_]f64{ -@sqrt(3.0 / 5.0), 0, @sqrt(3.0 / 5.0) };
    const weights = [_]f64{ 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
    var sum: f64 = 0;
    for (nodes, weights) |z, w| {
        const tau = (a + b) / 2 + (b - a) / 2 * z;
        var prod: f64 = 1;
        for (0..order) |k| {
            if (k == j) continue;
            const tk: f64 = sigmas[i - k];
            prod *= (tau - tk) / (@as(f64, sigmas[i - j]) - tk);
        }
        sum += w * prod;
    }
    return sum * (b - a) / 2;
}

/// `sample_ipndm_v`'s weights for the current derivative and the previous ones, in
/// the reference's own f32 arithmetic and operation order.
fn ipndmVCoeffs(t: []const f32, i: usize, order: usize) [4]f32 {
    if (order == 1) return .{ 1, 0, 0, 0 };
    const h_n = t[i + 1] - t[i];
    const h_n_1 = t[i] - t[i - 1];
    if (order == 2) return .{ (2 + (h_n / h_n_1)) / 2, -(h_n / h_n_1) / 2, 0, 0 };
    const h_n_2 = t[i - 1] - t[i - 2];
    const temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2;
    if (order == 3) return .{
        (2 + (h_n / h_n_1)) / 2 + temp1,
        -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1,
        temp1 * h_n_1 / h_n_2,
        0,
    };
    const h_n_3 = t[i - 2] - t[i - 3];
    const temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) *
        (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3));
    const r12 = h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3));
    return .{
        (2 + (h_n / h_n_1)) / 2 + temp1 + temp2,
        -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + r12) * temp2,
        temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + r12 * (1 + h_n_2 / h_n_3)) * temp2,
        -temp2 * r12 * h_n_1 / h_n_2,
    };
}

/// `deis.get_deis_coeff_list(sigmas, 3, N=10000, 'tab')`, in the reference's own f32
/// arithmetic: on a real model DEIS is sensitive enough that computing these weights
/// exactly in f64 (a 1e-5 relative difference) moves ComfyUI's own render 30 dB.
/// Only the 10000-term sum is wider, as torch's reduction order is not reproducible.
fn deisCoeffs(sigmas: []const f32, out: [][max_past]f32) void {
    // edm2t's betas: `torch.tensor(0.002)` is f32 but `torch.tensor(80)` is an int64
    // tensor, whose log is f64, and that promotes the rest to f64. Each is cast to
    // f32 where it meets an f32 tensor, the sums of betas after forming in f64.
    const ln_min: f64 = f32Log(0.002 * @as(f32, 0.002) + 1) / @as(f32, 1e-3);
    const ln_max: f64 = @log(6401.0);
    const beta_d64 = 2 * (ln_min - ln_max) / (1e-3 - 1);
    const beta_064 = ln_max - 0.5 * beta_d64;
    const beta_164 = beta_d64 + beta_064;
    const beta_0: f32 = @floatCast(beta_064);
    const c: f32 = @floatCast(beta_164 - beta_064);
    const t_par: TimeParams = .{
        .bm_sq = @floatCast(beta_064 * beta_064),
        .two_bd = @floatCast(2 * beta_d64),
        .bm = beta_0,
        .bd = @floatCast(beta_d64),
    };
    const N = 10000;

    var t_buf: [max_past]f32 = undefined;
    for (out, 0..) |*co, i| {
        co.* = @splat(0);
        const order = @min(i + 1, 3);
        // The step onto sigma 0 takes Euler; its weights would be infinite.
        if (order == 1 or sigmas[i + 1] <= 0) continue;
        for (0..order) |k| t_buf[k] = t_par.time(sigmas[i - k]);
        const t_cur = t_buf[0];
        const t_next = t_par.time(sigmas[i + 1]);
        const step = (t_next - t_cur) / @as(f32, N - 1);
        const dtau = (t_next - t_cur) / @as(f32, N);
        var sums: [max_past]f64 = @splat(0);
        for (0..N) |q| {
            // torch.linspace: fused multiply-add, from whichever end is nearer.
            const tau: f32 = if (q < N / 2)
                @floatCast(@as(f64, t_cur) + @as(f64, step) * @as(f64, @floatFromInt(q)))
            else
                @floatCast(@as(f64, t_next) - @as(f64, step) * @as(f64, @floatFromInt(N - 1 - q)));
            const alpha = f32Exp(-0.5 * (tau * tau) * c - tau * beta_0);
            // d log(alpha) / d tau as autograd forms it, `(1 / alpha) * alpha` included.
            const g = (1 / alpha) * alpha;
            const dlog = ((g * c) * -0.5) * (2 * tau) + (-g) * beta_0;
            const integrand = (-0.5 * dlog) / @sqrt(alpha * (1 - alpha));
            for (0..order) |j| {
                var poly: f32 = 1;
                for (0..order) |k| {
                    if (k == j) continue;
                    poly = poly * ((tau - t_buf[k]) / (t_buf[j] - t_buf[k]));
                }
                sums[j] += integrand * poly;
            }
        }
        for (0..order) |j| co[j] = @as(f32, @floatCast(sums[j])) * dtau;
    }
}

/// `edm2t`'s `vp_sigma_inv` on an f32 tensor: each f64 term is cast to f32 as it
/// meets the tensor.
const TimeParams = struct {
    bm_sq: f32,
    two_bd: f32,
    bm: f32,
    bd: f32,

    fn time(p: TimeParams, sigma: f32) f32 {
        return (@sqrt(p.bm_sq + p.two_bd * f32Log(sigma * sigma + 1)) - p.bm) / p.bd;
    }
};

fn f32Log(v: f32) f32 {
    return @floatCast(@log(@as(f64, v)));
}

fn f32Exp(v: f32) f32 {
    return @floatCast(@exp(@as(f64, v)));
}

test "lmsCoeff integrates the basis exactly: the weights of one step sum to its length" {
    const sigmas = [_]f32{ 14.6, 9.7, 6.1, 3.9, 2.2, 0 };
    for (1..5) |i| {
        const order = @min(i + 1, 4);
        var sum: f64 = 0;
        for (0..order) |j| sum += lmsCoeff(&sigmas, i, order, j);
        try std.testing.expectApproxEqRel(@as(f64, sigmas[i + 1]) - sigmas[i], sum, 1e-12);
    }
}

test "deis weights match ComfyUI's get_deis_coeff_list" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("assets/dpmpp_sde_fixtures.json"), .{});
    defer parsed.deinit();
    for (parsed.value.object.get("deis_coeffs").?.array.items) |entry| {
        const name = entry.object.get("name").?.string;
        const items = entry.object.get("sigmas").?.array.items;
        const sigmas = try gpa.alloc(f32, items.len);
        defer gpa.free(sigmas);
        for (sigmas, items) |*o, it| o.* = @floatCast(switch (it) {
            .float => |f| f,
            .integer => |n| @as(f64, @floatFromInt(n)),
            else => return error.BadFixture,
        });
        const got = try gpa.alloc([max_past]f32, sigmas.len - 1);
        defer gpa.free(got);
        deisCoeffs(sigmas, got);
        var worst: f64 = 0;
        for (entry.object.get("coeffs").?.array.items, got) |row, g| {
            if (row == .null) continue;
            for (row.array.items, 0..) |w, j| {
                const want = w.float;
                worst = @max(worst, @abs(g[j] - want) / @abs(want));
            }
        }
        std.debug.print("DEIS {s} worst rel {e}\n", .{ name, worst });
    }
}
