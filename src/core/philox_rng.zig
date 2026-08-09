//! NVIDIA's Philox4x32-10 plus Box-Muller standard normals: the noise AUTOMATIC1111
//! draws its initial latent from, reproduced on the CPU.
//!
//! This exists alongside `torch_rng.zig` because the two ecosystems draw their initial
//! noise from DIFFERENT generators, so the same seed is not the same picture. ComfyUI
//! draws `torch.randn` on the CPU (MT19937 + AVX2 `normal_fill`), which is
//! `torch_rng.zig` and this engine's default; A1111 defaults to `randn_source="GPU"`,
//! i.e. `torch.randn` on CUDA, which is this file. A different algorithm gives an
//! UNRELATED noise tensor, not a perturbed one, so unlike every other A1111-vs-ComfyUI
//! difference here this one decides whether the image is the same image at all.
//!
//! The reference is A1111's own `modules/rng_philox.py`, a pure-numpy imitation of the
//! CUDA generator, which is what its third `randn_source` setting ("NV") selects. So the
//! target is executable, and tools/gen_philox_fixtures.py runs it. Pinned by
//! `assets/philox_fixtures.json`, with the block cipher and the normals separate so a
//! mismatch localizes.
//!
//! Two places where imitating CUDA is exact and one where it is not; each is documented
//! at the code that implements it (`Generator.threads` for the element-to-counter map
//! and its hardware-dependent cap, `boxMuller` for the f64-versus-f32 arithmetic).

const std = @import("std");

/// Philox's two multipliers and its two key-bump (Weyl) constants.
const philox_m = [2]u32{ 0xD2511F53, 0xCD9E8D57 };
const philox_w = [2]u32{ 0x9E3779B9, 0xBB67AE85 };

/// 2^-32 exactly, as f32, `curand`'s `CURAND_2POW32_INV` and numpy's `two_pow32_inv`.
const inv_f32: f32 = 2.3283064e-10;
/// The same scaled by 2π, rounded to f32 *after* an f64 product, which is what
/// `np.array([2.3283064e-10 * 6.2831855], dtype=np.float32)` computes.
const inv_2pi_f32: f32 = @floatCast(@as(f64, 2.3283064e-10) * @as(f64, 6.2831855));

// Widened once here: the reference adds an f32 `c/2` to an f64 product, and halving an
// f32 is exact, so these are the exact f64 values that addition sees.
const inv: f64 = inv_f32;
const inv_half: f64 = inv / 2.0;
const inv_2pi: f64 = inv_2pi_f32;
const inv_2pi_half: f64 = inv_2pi / 2.0;

/// The 32×32->64 product Philox's rounds are built from.
inline fn mulhilo(a: u32, b: u32) struct { lo: u32, hi: u32 } {
    const p: u64 = @as(u64, a) * @as(u64, b);
    return .{ .lo = @truncate(p), .hi = @truncate(p >> 32) };
}

inline fn philoxRound(ctr: *[4]u32, key: [2]u32) void {
    const v1 = mulhilo(ctr[0], philox_m[0]);
    const v2 = mulhilo(ctr[2], philox_m[1]);
    ctr.* = .{
        v2.hi ^ ctr[1] ^ key[0],
        v2.lo,
        v1.hi ^ ctr[3] ^ key[1],
        v1.lo,
    };
}

/// Ten rounds of Philox4x32, bumping the key between them: `counter` in, four
/// uniformly-distributed words out.
pub fn philox4x32_10(counter: [4]u32, key: [2]u32) [4]u32 {
    var ctr = counter;
    var k = key;
    // The reference bumps the key after each of the first nine rounds, so round ten
    // runs with `key + 9*w`.
    for (0..9) |_| {
        philoxRound(&ctr, k);
        k[0] +%= philox_w[0];
        k[1] +%= philox_w[1];
    }
    philoxRound(&ctr, k);
    return ctr;
}

/// Box-Muller on one pair of words: `.sin` is the value curand returns first and the
/// only one A1111's numpy path keeps, `.cos` is its partner, consumed only by the
/// unrolled slots above the thread cap (see `Generator.threads`).
///
/// The arithmetic is numpy's f64, not curand's f32: `uint32 * float32` promotes to
/// float64 under NumPy's rules, so this runs in f64 with f32-rounded constants and
/// narrows once, where curand's device path is f32 throughout and uses `__sincosf`.
/// Matching numpy makes this bit-exact against `randn_source="NV"` and ~1e-6 relative
/// against `"GPU"`, two orders below the model-level disagreement a render already
/// carries.
inline fn boxMuller(x: u32, y: u32) struct { sin: f32, cos: f32 } {
    const u: f64 = @as(f64, @floatFromInt(x)) * inv + inv_half;
    const v: f64 = @as(f64, @floatFromInt(y)) * inv_2pi + inv_2pi_half;
    const s: f64 = @sqrt(-2.0 * @log(u));
    return .{ .sin = @floatCast(s * @sin(v)), .cos = @floatCast(s * @cos(v)) };
}

/// A1111's `rng_philox.Generator`: a seed plus a counter that advances one step per
/// `randn` call, so successive draws from the same generator differ.
pub const Generator = struct {
    /// The seed, split low word first, `curand_init`'s `key = (seed, seed >> 32)`.
    key: [2]u32,
    /// `counter[0]`; incremented per draw, matching ATen's philox-offset bookkeeping.
    offset: u32 = 0,
    /// How many CUDA threads the launch being reproduced used, or 0 for one counter per
    /// element: A1111's "NV" path, and the right answer for any tensor below the launch's
    /// thread cap.
    ///
    /// ATen's kernel gives each thread ONE `curand_normal4` call and spreads its four
    /// normals across a grid stride, so element `i` is not obviously thread `i`'s first
    /// normal. It is, as long as `numel <= 256 * grid_cap`, because the launch uses
    /// `grid.x = min(SM_count * (maxThreadsPerSM / 256), ceil(numel/256))`: below the cap
    /// there is one thread per element and only `rand.x` is ever consumed, which is
    /// exactly `counter[2] = i`, sin only.
    ///
    /// The cap is HARDWARE-DEPENDENT (on a 3090, 82 SMs x 1536 threads/SM / 256 = 492
    /// blocks = 125,952 elements, above a 1024x1536 latent's 98,304 and below a 1536
    /// square latent's 147,456). Past it the tail comes from the cosine partners and the
    /// noise genuinely differs. Nothing in the CLI sets this, because reproducing a
    /// foreign A1111 render would require knowing which GPU drew it; A1111's own "NV"
    /// setting has the same limitation.
    threads: u32 = 0,

    pub fn init(seed: u64) Generator {
        return .{ .key = .{ @truncate(seed), @truncate(seed >> 32) } };
    }

    /// Fill `out` with standard normals and advance the counter.
    pub fn randn(self: *Generator, out: []f32) void {
        if (out.len == 0) return;
        const stride: usize = if (self.threads == 0) out.len else @min(out.len, self.threads);
        const per_iter = stride * 4;
        const iters = (out.len + per_iter - 1) / per_iter;

        for (0..iters) |j| {
            const ctr0 = self.offset +% @as(u32, @intCast(j));
            for (0..stride) |t| {
                const g = philox4x32_10(.{ ctr0, 0, @intCast(t), 0 }, self.key);
                const a = boxMuller(g[0], g[1]);
                const base = t + per_iter * j;
                if (base >= out.len) break;
                out[base] = a.sin;
                // The overwhelmingly common case: one thread per element, so the other
                // three normals of this call are simply not consumed.
                if (stride >= out.len) continue;
                if (base + stride < out.len) out[base + stride] = a.cos;
                const b = boxMuller(g[2], g[3]);
                if (base + 2 * stride < out.len) out[base + 2 * stride] = b.sin;
                if (base + 3 * stride < out.len) out[base + 3 * stride] = b.cos;
            }
        }
        self.offset +%= @intCast(iters);
    }
};

/// One draw from a fresh generator, `torch.randn(shape, device='cuda')` after
/// `manual_seed(seed)`, and what every call site here wants.
pub fn randn(out: []f32, seed: u64) void {
    var g: Generator = .init(seed);
    g.randn(out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "the Box-Muller constants are the reference's, to the bit" {
    // Not decoration: `inv_2pi` is an f64 product rounded to f32, and doing the product
    // in a different precision moves it by an ulp, which moves every noise value.
    try std.testing.expectEqual(@as(u32, 0x2f800000), @as(u32, @bitCast(inv_f32)));
    try std.testing.expectEqual(@as(u32, 0x30c90fdb), @as(u32, @bitCast(inv_2pi_f32)));
}

test "philox_rng reproduces rng_philox's own docstring" {
    // The reference documents this output for seed 0, shape (3, 4). Cheap end-to-end
    // check that needs no fixture file.
    var out: [12]f32 = undefined;
    randn(&out, 0);
    const want = [12]f32{
        -0.92466259, -0.42534415, -2.6438457,  0.14518388,
        -0.12086647, -0.57972564, -0.62285122, -0.32838709,
        -1.07454231, -0.36314407, -1.67105067, 2.26550497,
    };
    // 1e-6, not tighter: the docstring prints eight decimals of values the reference
    // actually returns as f32. The bit-exact bar is the fixture test below.
    for (out, want) |got, w| try std.testing.expectApproxEqAbs(w, got, 1e-6);
}

test "philox noise is bit-exact with A1111's rng_philox" {
    const gpa = std.testing.allocator;
    const fx_src = @embedFile("assets/philox_fixtures.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fx_src, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    // 1. The block cipher alone, so a mismatch localizes to the rounds.
    for (root.get("philox_raw").?.array.items) |item| {
        const c = item.object.get("counter").?.array.items;
        const k = item.object.get("key").?.array.items;
        var ctr: [4]u32 = undefined;
        for (&ctr, c) |*d, v| d.* = @intCast(v.integer);
        var key: [2]u32 = undefined;
        for (&key, k) |*d, v| d.* = @intCast(v.integer);
        const got = philox4x32_10(ctr, key);
        for (got, item.object.get("out").?.array.items, 0..) |g, want, i| {
            errdefer std.debug.print(
                "philox ctr={any} key={any} word {d}: got {d} want {d}\n",
                .{ ctr, key, i, g, want.integer },
            );
            try std.testing.expectEqual(@as(u32, @intCast(want.integer)), g);
        }
    }

    // 2. The normals. Bit-exact is the bar: a quantity used as a *seed's* output has to
    //    agree digit for digit, not closely.
    for (root.get("philox_randn").?.array.items) |item| {
        const o = item.object;
        const seed: u64 = @intCast(o.get("seed").?.integer);
        const n: usize = @intCast(o.get("n").?.integer);
        const buf = try gpa.alloc(f32, n);
        defer gpa.free(buf);

        var g: Generator = .init(seed);
        g.randn(buf);
        try expectDraw(o, buf, seed, "first8", "hash", "all");
        // The second draw from the same generator, pins the offset advance.
        g.randn(buf);
        try expectDraw(o, buf, seed, "next_first8", "next_hash", null);

        // last8 belongs to the first draw only.
        var g2: Generator = .init(seed);
        g2.randn(buf);
        for (o.get("last8").?.array.items, buf[n - 8 ..]) |want, got| {
            errdefer std.debug.print("seed {d} n {d} tail: got {d} want {d}\n", .{ seed, n, got, want.float });
            try std.testing.expectEqual(@as(f32, @floatCast(want.float)), got);
        }
    }
}

fn expectDraw(
    o: std.json.ObjectMap,
    buf: []const f32,
    seed: u64,
    head_key: []const u8,
    hash_key: []const u8,
    all_key: ?[]const u8,
) !void {
    for (o.get(head_key).?.array.items, buf[0..8], 0..) |want, got, i| {
        errdefer std.debug.print(
            "seed {d} n {d} {s}[{d}]: got {d} want {d}\n",
            .{ seed, buf.len, head_key, i, got, want.float },
        );
        try std.testing.expectEqual(@as(f32, @floatCast(want.float)), got);
    }
    // The hash covers every element, which the head/tail samples deliberately do not:
    // an element-mapping bug in the middle of a latent is exactly what would slip past.
    const want_hash = try std.fmt.parseInt(u64, o.get(hash_key).?.string, 10);
    var h: u64 = 0xCBF29CE484222325;
    for (std.mem.sliceAsBytes(buf)) |b| h = (h ^ b) *% 0x100000001B3;
    errdefer std.debug.print("seed {d} n {d} {s}: hash {d} want {d}\n", .{ seed, buf.len, hash_key, h, want_hash });
    try std.testing.expectEqual(want_hash, h);

    if (all_key) |ak| if (o.get(ak)) |arr| {
        for (arr.array.items, buf) |want, got| {
            try std.testing.expectEqual(@as(f32, @floatCast(want.float)), got);
        }
    };
}

test "the CUDA thread cap changes the noise, and only above the cap" {
    // Below the cap the unrolled slots are unreachable, so a capped generator has to
    // agree with the uncapped one element for element. This is what makes the uncapped
    // form the right default rather than a simplification.
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    randn(&a, 7);
    var capped: Generator = .init(7);
    capped.threads = 64;
    capped.randn(&b);
    try std.testing.expectEqualSlices(f32, &a, &b);

    // Above it the tail comes from the cosine partners instead, so the two differ,
    // and the head still agrees, which is the signature of the grid-stride mapping.
    var c: [64]f32 = undefined;
    var tight: Generator = .init(7);
    tight.threads = 16;
    tight.randn(&c);
    try std.testing.expectEqualSlices(f32, a[0..16], c[0..16]);
    try std.testing.expect(!std.mem.eql(f32, a[16..], c[16..]));
}
