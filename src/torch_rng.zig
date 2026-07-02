//! torch.randn-compatible standard-normal generation, bit-exact with
//! PyTorch's CPU generator so a ComfyUI seed reproduces the identical
//! initial latent noise.
//!
//! Torch's pipeline for contiguous float tensors of >= 16 elements
//! (normal_fill_AVX2 in aten's DistributionTemplates.h): an MT19937 engine
//! writes one uniform per element ((random() & 0xFFFFFF) * 2^-24), then
//! each 16-block is Box-Mullered in place (pairs j / j+8) using the Cephes
//! polynomial log/sincos from avx_mathfun.h — NOT libm; sizes not divisible
//! by 16 redraw fresh uniforms for a final, overlapping 16-block.
//!
//! The polynomials here replicate the *compiled* instruction sequence of
//! the shipping torch wheel, fused multiply-adds included (the build
//! contracts every mul-feeding-add, and the cosine polynomial's final
//! multiply fuses into the following subtraction as an fmsub, not an
//! fnmadd). Verified bit-for-bit against fixtures dumped from the
//! reference venv (tools/dump_randn.py), including a full 470400-element
//! latent.
//!
//! Portability note: torch dispatches an MKL VSL path on Intel CPUs with
//! different output; this matches the AVX2 normal_fill path torch uses on
//! this AMD box.

const std = @import("std");

/// Mersenne twister matching torch's at::mt19937 (the reference MT19937
/// with the standard Knuth seeding).
pub const Mt19937 = struct {
    state: [n]u32,
    index: usize,

    const n = 624;
    const m = 397;

    pub fn init(seed: u64) Mt19937 {
        var s: [n]u32 = undefined;
        s[0] = @truncate(seed);
        for (1..n) |i| {
            s[i] = 1812433253 *% (s[i - 1] ^ (s[i - 1] >> 30)) +% @as(u32, @intCast(i));
        }
        return .{ .state = s, .index = n };
    }

    pub fn random(self: *Mt19937) u32 {
        if (self.index == n) self.twist();
        var y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    fn twist(self: *Mt19937) void {
        const matrix_a: u32 = 0x9908b0df;
        for (0..n) |i| {
            const y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % n] & 0x7fffffff);
            var v = y >> 1;
            if (y & 1 != 0) v ^= matrix_a;
            self.state[i] = self.state[(i + m) % n] ^ v;
        }
        self.index = 0;
    }
};

/// One 24-bit-mantissa uniform in [0, 1), torch's
/// uniform_real_distribution<float>(0, 1).
inline fn uniform(g: *Mt19937) f32 {
    const divisor: f32 = 1.0 / @as(f32, 1 << 24);
    return @as(f32, @floatFromInt(g.random() & 0xFFFFFF)) * divisor;
}

/// avx_mathfun's log256_ps, one lane, with the wheel's fma contraction.
fn cephesLog(x0: f32) f32 {
    const min_norm_pos: f32 = @bitCast(@as(u32, 0x00800000));
    var x = @max(x0, min_norm_pos);
    var imm0: i32 = @intCast(@as(u32, @bitCast(x)) >> 23);
    x = @bitCast((@as(u32, @bitCast(x)) & ~@as(u32, 0x7f800000)) | @as(u32, @bitCast(@as(f32, 0.5))));
    imm0 -= 0x7f;
    var e: f32 = @floatFromInt(imm0);
    e += 1.0;
    const sqrthf: f32 = 0.707106781186547524;
    const mask = x < sqrthf;
    const tmp: f32 = if (mask) x else 0.0;
    x -= 1.0;
    if (mask) e -= 1.0;
    x += tmp;

    const z = x * x;
    var y: f32 = 7.0376836292e-2;
    y = @mulAdd(f32, y, x, -1.1514610310e-1);
    y = @mulAdd(f32, y, x, 1.1676998740e-1);
    y = @mulAdd(f32, y, x, -1.2420140846e-1);
    y = @mulAdd(f32, y, x, 1.4249322787e-1);
    y = @mulAdd(f32, y, x, -1.6668057665e-1);
    y = @mulAdd(f32, y, x, 2.0000714765e-1);
    y = @mulAdd(f32, y, x, -2.4999993993e-1);
    y = @mulAdd(f32, y, x, 3.3333331174e-1);
    y = y * x;
    // The y*z multiply fuses into the e*q1 addition (the build contracts
    // each pending multiply into its consuming add — same choice as the
    // cosine polynomial's fmsub below).
    const eq1 = e * -2.12194440e-4;
    y = @mulAdd(f32, y, z, eq1);
    y = @mulAdd(f32, z, -0.5, y);
    x = x + y;
    x = @mulAdd(f32, e, 0.693359375, x);
    return x;
}

/// avx_mathfun's sincos256_ps, one lane, with the wheel's fma contraction.
fn cephesSincos(theta: f32) struct { sin: f32, cos: f32 } {
    var sign_sin: u32 = @as(u32, @bitCast(theta)) & 0x80000000;
    var x: f32 = @bitCast(@as(u32, @bitCast(theta)) & 0x7fffffff);

    const fopi: f32 = 1.27323954473516; // 4/pi
    const y0 = x * fopi;
    var imm2: i32 = @intFromFloat(y0); // cvttps: truncate
    imm2 = (imm2 + 1) & ~@as(i32, 1);
    const yq: f32 = @floatFromInt(imm2);

    const swap_sign: u32 = @as(u32, @bitCast(imm2 & 4)) << 29;
    const poly = (imm2 & 2) == 0;
    sign_sin ^= swap_sign;
    const sign_cos: u32 = @as(u32, @bitCast(~(imm2 - 2) & 4)) << 29;

    // Extended-precision modular arithmetic.
    x = @mulAdd(f32, yq, -0.78515625, x);
    x = @mulAdd(f32, yq, -2.4187564849853515625e-4, x);
    x = @mulAdd(f32, yq, -3.77489497744594108e-8, x);

    const z = x * x;
    // Cosine polynomial: the final multiply by z fuses into the following
    // subtraction (fmsub(y, z, z*0.5)); the alternative fnmadd contraction
    // rounds differently on ~0.1% of inputs.
    var yc: f32 = @mulAdd(f32, 2.443315711809948e-5, z, -1.388731625493765e-3);
    yc = @mulAdd(f32, yc, z, 4.166664568298827e-2);
    yc = yc * z;
    const half_z = z * 0.5;
    yc = @mulAdd(f32, yc, z, -half_z);
    yc = yc + 1.0;

    var ys: f32 = @mulAdd(f32, -1.9515295891e-4, z, 8.3321608736e-3);
    ys = @mulAdd(f32, ys, z, -1.6666654611e-1);
    ys = ys * z;
    ys = @mulAdd(f32, ys, x, x);

    const sin_v = if (poly) ys else yc;
    const cos_v = if (poly) yc else ys;
    return .{
        .sin = @bitCast(@as(u32, @bitCast(sin_v)) ^ sign_sin),
        .cos = @bitCast(@as(u32, @bitCast(cos_v)) ^ sign_cos),
    };
}

/// torch's normal_fill_16_AVX2: Box-Muller over 8 uniform pairs (j, j+8),
/// in place. theta uses float(2.0f * pi<double>) like _mm256_set1_ps does.
fn fill16(data: *[16]f32) void {
    const two_pi: f32 = @floatCast(2.0 * std.math.pi);
    for (0..8) |j| {
        const ua: f32 = 1.0 - data[j]; // [0, 1) -> (0, 1]
        const ub = data[j + 8];
        const radius = @sqrt(-2.0 * cephesLog(ua));
        const sc = cephesSincos(two_pi * ub);
        data[j] = radius * sc.cos;
        data[j + 8] = radius * sc.sin;
    }
}

/// Fill `out` with standard-normal samples, bit-identical to
/// torch.manual_seed(seed) + torch.randn(out.len) on the CPU. Requires
/// out.len >= 16 (below that torch switches to a cached element-wise
/// Box-Muller this doesn't implement; latents are always >= 16).
pub fn randn(out: []f32, seed: u64) void {
    std.debug.assert(out.len >= 16);
    var g = Mt19937.init(seed);
    for (out) |*v| v.* = uniform(&g);
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) {
        fill16(out[i..][0..16]);
    }
    if (out.len % 16 != 0) {
        // Torch redraws fresh uniforms for the final 16, overlapping the
        // already-transformed region.
        const tail = out[out.len - 16 ..][0..16];
        for (tail) |*v| v.* = uniform(&g);
        fill16(tail);
    }
}

test "randn matches torch fixtures bit-for-bit" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    const cases = [_]struct { path: []const u8, len: usize, seed: u64 }{
        .{ .path = "testdata/randn_42_1024.bin", .len = 1024, .seed = 42 },
        .{ .path = "testdata/randn_7_40.bin", .len = 40, .seed = 7 },
        .{ .path = "testdata/randn_12345_1000.bin", .len = 1000, .seed = 12345 },
        // The comparison workflow's full [1, 16, 210, 140] latent.
        .{ .path = "testdata/randn_80085_full.bin", .len = 470400, .seed = 80085 },
    };
    for (cases) |case| {
        const expected = try gpa.alloc(f32, case.len);
        defer gpa.free(expected);
        {
            const file = std.Io.Dir.cwd().openFile(io, case.path, .{ .mode = .read_only }) catch return error.SkipZigTest;
            defer file.close(io);
            const bytes = std.mem.sliceAsBytes(expected);
            if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
        }
        const got = try gpa.alloc(f32, case.len);
        defer gpa.free(got);
        randn(got, case.seed);
        try std.testing.expectEqualSlices(u32, @ptrCast(expected), @ptrCast(got));
    }
}
