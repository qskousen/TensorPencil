//! Vectorized transcendentals. The hot paths (softmax, SiLU/sigmoid gating)
//! evaluate exp() hundreds of millions of times per DiT step; libm-per-lane
//! is far too slow.

const std = @import("std");

pub const vlen = std.simd.suggestVectorLength(f32) orelse 8;
pub const Vec = @Vector(vlen, f32);

/// Vectorized exp(x), cephes-style: n = round(x*log2e), e^x = 2^n * e^r with a
/// degree-5 polynomial for e^r. ~1-2 ulp over normal range; underflows to 0
/// below ~-87.3, saturates to +inf above ~88.7.
pub fn expVec(x: Vec) Vec {
    const log2e: Vec = @splat(1.44269504088896341);
    const ln2_hi: Vec = @splat(0.693359375);
    const ln2_lo: Vec = @splat(-2.12194440e-4);

    const max_x: Vec = @splat(88.72283905206835);
    const min_x: Vec = @splat(-87.33654475055310);
    const xc = @min(@max(x, min_x), max_x);

    const n = @round(xc * log2e);
    var r = xc - n * ln2_hi;
    r -= n * ln2_lo;

    // e^r = 1 + r + r^2 * P(r) (cephes expf coefficients).
    const c0: Vec = @splat(1.9875691500e-4);
    const c1: Vec = @splat(1.3981999507e-3);
    const c2: Vec = @splat(8.3334519073e-3);
    const c3: Vec = @splat(4.1665795894e-2);
    const c4: Vec = @splat(1.6666665459e-1);
    const c5: Vec = @splat(5.0000001201e-1);
    var p = c0;
    p = @mulAdd(Vec, p, r, c1);
    p = @mulAdd(Vec, p, r, c2);
    p = @mulAdd(Vec, p, r, c3);
    p = @mulAdd(Vec, p, r, c4);
    p = @mulAdd(Vec, p, r, c5);
    var y = @mulAdd(Vec, p, r * r, r) + @as(Vec, @splat(1.0));

    // Scale by 2^n via exponent bits, split in two so n up to 128 (the
    // saturation clamp) stays representable at each step.
    const bias: @Vector(vlen, i32) = @splat(127);
    const ni: @Vector(vlen, i32) = @intFromFloat(n);
    const n1 = ni >> @as(@Vector(vlen, u5), @splat(1));
    const n2 = ni - n1;
    const shift: @Vector(vlen, u5) = @splat(23);
    y *= @as(Vec, @bitCast((n1 + bias) << shift));
    y *= @as(Vec, @bitCast((n2 + bias) << shift));

    // Exact zero below the underflow threshold (saturation handles overflow).
    y = @select(f32, x < min_x, @as(Vec, @splat(0.0)), y);
    return y;
}

pub inline fn sigmoidVec(x: Vec) Vec {
    const one: Vec = @splat(1.0);
    return one / (one + expVec(-x));
}

test "expVec matches libm within tolerance" {
    var x: f32 = -90.0;
    while (x <= 89.0) : (x += 0.137) {
        var input: Vec = @splat(x);
        input[0] = x; // exercise all lanes with the same value anyway
        const got = expVec(input)[0];
        const want = @exp(x);
        if (x < -87.33654475) {
            try std.testing.expectEqual(@as(f32, 0), got);
        } else if (want == 0 or std.math.isInf(want)) {
            try std.testing.expectEqual(want, got);
        } else {
            const rel = @abs(got - want) / want;
            try std.testing.expect(rel < 3e-7);
        }
    }
    // Exactly zero input.
    try std.testing.expectEqual(@as(f32, 1.0), expVec(@as(Vec, @splat(0.0)))[0]);
}

test "sigmoidVec matches scalar" {
    var x: f32 = -30.0;
    while (x <= 30.0) : (x += 0.31) {
        const got = sigmoidVec(@as(Vec, @splat(x)))[0];
        const want = 1.0 / (1.0 + @exp(-x));
        try std.testing.expectApproxEqAbs(want, got, 2e-7);
    }
}
