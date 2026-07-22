//! Elementwise activation functions (f32, in-place where fusing is natural).
//! Bulk paths are SIMD via vmath.expVec; scalar tails use libm.

const std = @import("std");
const vmath = @import("vmath.zig");

const vlen = vmath.vlen;
const Vec = vmath.Vec;

pub inline fn sigmoidScalar(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn siluScalar(x: f32) f32 {
    return x * sigmoidScalar(x);
}

/// tanh-approximated GELU, matching torch's `gelu(approximate="tanh")`:
/// 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
pub inline fn geluTanhScalar(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

pub fn silu(xs: []f32) void {
    var i: usize = 0;
    while (i + vlen <= xs.len) : (i += vlen) {
        const x: Vec = xs[i..][0..vlen].*;
        xs[i..][0..vlen].* = x * vmath.sigmoidVec(x);
    }
    while (i < xs.len) : (i += 1) xs[i] = siluScalar(xs[i]);
}

pub fn geluTanh(xs: []f32) void {
    // tanh(t) = 1 - 2/(e^{2t} + 1)
    const c: Vec = @splat(0.7978845608028654);
    const c3: Vec = @splat(0.044715);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);
    const half: Vec = @splat(0.5);
    var i: usize = 0;
    while (i + vlen <= xs.len) : (i += vlen) {
        const x: Vec = xs[i..][0..vlen].*;
        const inner = c * (x + c3 * x * x * x);
        const tanh_v = one - two / (vmath.expVec(two * inner) + one);
        xs[i..][0..vlen].* = half * x * (one + tanh_v);
    }
    while (i < xs.len) : (i += 1) xs[i] = geluTanhScalar(xs[i]);
}

pub fn sigmoid(xs: []f32) void {
    var i: usize = 0;
    while (i + vlen <= xs.len) : (i += vlen) {
        xs[i..][0..vlen].* = vmath.sigmoidVec(xs[i..][0..vlen].*);
    }
    while (i < xs.len) : (i += 1) xs[i] = sigmoidScalar(xs[i]);
}

/// SwiGLU gating: gate[i] = silu(gate[i]) * up[i].
pub fn siluMul(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    var i: usize = 0;
    while (i + vlen <= gate.len) : (i += vlen) {
        const g: Vec = gate[i..][0..vlen].*;
        const u: Vec = up[i..][0..vlen].*;
        gate[i..][0..vlen].* = g * vmath.sigmoidVec(g) * u;
    }
    while (i < gate.len) : (i += 1) gate[i] = siluScalar(gate[i]) * up[i];
}

/// GeGLU gating (Gemma FFN): gate[i] = gelu_tanh(gate[i]) * up[i].
pub fn geluTanhMul(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    const c: Vec = @splat(0.7978845608028654);
    const c3: Vec = @splat(0.044715);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);
    const half: Vec = @splat(0.5);
    var i: usize = 0;
    while (i + vlen <= gate.len) : (i += vlen) {
        const x: Vec = gate[i..][0..vlen].*;
        const u: Vec = up[i..][0..vlen].*;
        const inner = c * (x + c3 * x * x * x);
        const tanh_v = one - two / (vmath.expVec(two * inner) + one);
        gate[i..][0..vlen].* = half * x * (one + tanh_v) * u;
    }
    while (i < gate.len) : (i += 1) gate[i] = geluTanhScalar(gate[i]) * up[i];
}

/// Error function (Abramowitz & Stegun 7.1.26; max abs error ~1.5e-7), for the
/// exact (non-tanh) GELU used by GTE-style encoders (`hidden_act == "gelu"`).
pub inline fn erfScalar(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-ax * ax);
    return sign * y;
}

/// Exact (erf) GELU, matching torch's default `nn.GELU()`:
/// 0.5x(1 + erf(x/sqrt(2))).
pub inline fn geluErfScalar(x: f32) f32 {
    return 0.5 * x * (1.0 + erfScalar(x * 0.7071067811865476));
}

/// GeGLU gating with exact-erf GELU (GTE / Snowflake Arctic Embed FFN):
/// gate[i] = gelu_erf(gate[i]) * up[i].
pub fn geluErfMul(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    for (gate, up) |*g, u| g.* = geluErfScalar(g.*) * u;
}

/// "Quick" GELU, matching ggml `ggml_gelu_quick` / torch `gelu(approximate)`
/// via the sigmoid approximation: x * sigmoid(1.702 x). Used by the gemma4v
/// vision tower's FFN (llama.cpp `FFN_GELU_QUICK`).
pub inline fn geluQuickScalar(x: f32) f32 {
    return x * sigmoidScalar(1.702 * x);
}

/// GeGLU-quick gating (gemma4v vision FFN): gate[i] = gelu_quick(gate[i]) * up[i].
pub fn geluQuickMul(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    const c: Vec = @splat(1.702);
    var i: usize = 0;
    while (i + vlen <= gate.len) : (i += vlen) {
        const g: Vec = gate[i..][0..vlen].*;
        const u: Vec = up[i..][0..vlen].*;
        gate[i..][0..vlen].* = g * vmath.sigmoidVec(c * g) * u;
    }
    while (i < gate.len) : (i += 1) gate[i] = geluQuickScalar(gate[i]) * up[i];
}

/// Sigmoid gating (Krea 2 attention output): dst[i] *= sigmoid(gate[i]).
pub fn sigmoidMul(dst: []f32, gate: []const f32) void {
    std.debug.assert(dst.len == gate.len);
    var i: usize = 0;
    while (i + vlen <= dst.len) : (i += vlen) {
        const d: Vec = dst[i..][0..vlen].*;
        dst[i..][0..vlen].* = d * vmath.sigmoidVec(gate[i..][0..vlen].*);
    }
    while (i < dst.len) : (i += 1) dst[i] *= sigmoidScalar(gate[i]);
}

// Reference values generated by tools/gen_op_fixtures.py (torch 2.6).
const act_in = [_]f32{ -3, -1, -0.5, 0, 0.5, 1, 2, 3, 0.100000001, -2.20000005 };
const gelu_tanh_out = [_]f32{ -0.00363743305, -0.158807993, -0.154285997, 0, 0.345714003, 0.841192007, 1.95459771, 2.99636269, 0.0539827533, -0.0303215031 };
const silu_out = [_]f32{ -0.142277613, -0.268941432, -0.188770339, 0, 0.311229676, 0.731058598, 1.76159406, 2.85772252, 0.0524979196, -0.21945107 };
const sigmoid_out = [_]f32{ 0.0474258736, 0.268941432, 0.377540678, 0.5, 0.622459352, 0.731058598, 0.880797029, 0.952574134, 0.524979174, 0.0997504815 };

fn expectClose(expected: []const f32, actual: []const f32, tol: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| try std.testing.expectApproxEqAbs(e, a, tol);
}

test "activations match torch" {
    var buf: [act_in.len]f32 = act_in;
    geluTanh(&buf);
    try expectClose(&gelu_tanh_out, &buf, 2e-6);

    buf = act_in;
    silu(&buf);
    try expectClose(&silu_out, &buf, 2e-6);

    buf = act_in;
    sigmoid(&buf);
    try expectClose(&sigmoid_out, &buf, 2e-6);
}

test "fused gates" {
    var gate = [_]f32{ 1.0, -0.5 };
    const up = [_]f32{ 2.0, 3.0 };
    siluMul(&gate, &up);
    try std.testing.expectApproxEqAbs(@as(f32, 0.731058598 * 2.0), gate[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.188770339 * 3.0), gate[1], 1e-6);

    var dst = [_]f32{ 2.0, 2.0 };
    const g2 = [_]f32{ 0.0, 1.0 };
    sigmoidMul(&dst, &g2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 0.731058598), dst[1], 1e-6);

    // gelu_quick(x) = x*sigmoid(1.702x); at x=1: 1*sigmoid(1.702)=0.845894...
    var gq = [_]f32{ 1.0, 0.0 };
    const up2 = [_]f32{ 3.0, 5.0 };
    geluQuickMul(&gq, &up2);
    try std.testing.expectApproxEqAbs(@as(f32, sigmoidScalar(1.702) * 3.0), gq[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gq[1], 1e-6);
}
