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
