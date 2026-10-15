//! The convrot rotation and the integer weight decodes as pure math, so the dual
//! kernels in quant.zig and the host tests in `models/lin_cuda.zig` run one body.
//!
//! The size-256 Hadamard is kron(h4, h4, h4, h4): one 4-point butterfly per base-4
//! digit of the index, and the four digit passes commute. On the device a lane holds
//! 16 contiguous values of a group and 16 lanes hold the group: `radix16` does digits
//! 0 and 1 in registers, `bfly4` digits 2 and 3 across lanes `^ 1, 2, 3` and
//! `^ 4, 8, 12`. Neither applies the 1/16 gain.

/// The two digit passes over 16 values, the radix-4 butterfly `ops.convrot` uses.
pub inline fn radix16(v: *[16]f32) void {
    inline for (.{ 1, 4 }) |s| {
        inline for (0..16) |p| {
            // A butterfly starts where digit `s` of p is zero.
            if ((p / s) % 4 == 0) {
                const a = v[p];
                const b = v[p + s];
                const c = v[p + 2 * s];
                const d = v[p + 3 * s];
                v[p] = a + b + c - d;
                v[p + s] = a + b - c + d;
                v[p + 2 * s] = a - b + c + d;
                v[p + 3 * s] = -a + b + c + d;
            }
        }
    }
}

/// One output of a 4-point butterfly from the four inputs as seen from digit `d`:
/// `own` at d, `x1..x3` at `d ^ 1..3`. Row d of h4 is all ones but a -1 at `3 - d`,
/// which is `d ^ 3`.
pub inline fn bfly4(own: f32, x1: f32, x2: f32, x3: f32) f32 {
    return (own + x1 + x2 + x3) - 2 * x3;
}

/// Sign-extend the low byte.
pub inline fn sext8(v: u32) i32 {
    return @as(i32, @bitCast(v << 24)) >> 24;
}

/// Sign-extend the low nibble.
pub inline fn sext4(v: u32) i32 {
    return @as(i32, @bitCast(v << 28)) >> 28;
}

/// Four int8 weights packed in one word, times their row's scale.
pub inline fn i8Quad(word: u32, scale: f32) [4]f32 {
    var out: [4]f32 = undefined;
    inline for (0..4) |j| out[j] = @as(f32, @floatFromInt(sext8(word >> (8 * j)))) * scale;
    return out;
}
