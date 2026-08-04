//! torchsde's Brownian tree, as ComfyUI's SDE samplers use it.
//!
//! `dpmpp_2m_sde` / `dpmpp_3m_sde` / `dpmpp_sde` do not draw fresh noise per step.
//! They query a **Brownian motion sampled over the sigma axis**:
//!
//!     noise(sigma, sigma_next) = W([sigma_next, sigma]) / sqrt(|sigma - sigma_next|)
//!
//! where `W` is one path fixed by the seed alone. That is the whole point of the
//! construction: the increments over *adjacent* sigma intervals are consistent, so
//! the same seed at 8 and at 20 steps walks the same underlying path and produces
//! recognisably the same image. A per-step `randn` gives statistically valid noise
//! and none of that property — and, more to the point here, does not reproduce
//! ComfyUI.
//!
//! ComfyUI's default is `BrownianTreeNoiseSampler`, i.e.
//! `torchsde.BrownianTree(entropy=seed, tol=1e-6, pool_size=24)`, which is
//! `BrownianInterval(halfway_tree=True, levy_area_approximation='none')`. Three
//! pieces make it reproducible, and all three are ported here:
//!
//!  1. **The dyadic interval tree.** `halfway_tree=True` means the path is
//!     determined by the seed *alone* — not by the order or location of queries.
//!     An interval is only ever split at its own (rounded) midpoint, recursively,
//!     until the requested boundary falls on a node edge.
//!  2. **`numpy.random.SeedSequence`** derives each node's RNG seed from
//!     `(entropy, spawn_key=(node, depth), pool_size=24)` — see `seed_seq.zig`.
//!  3. **`torch.randn`** seeded with that u32 — `noise.zig`, whose two engines are the
//!     ecosystems' two generators (see there: A1111's tree runs on the GPU's Philox),
//!     already bit-exact.
//!
//! ⚠️ **Time is quantised by Python's `round(x, 6)`** (`tol = 1e-6`), and that is
//! not the same function as "multiply by 1e6, round, divide". Python rounds the
//! **exact binary value** with ties-to-even; the naive form double-rounds. A single
//! ulp of disagreement on one midpoint gives a different tree and therefore a
//! completely different image, so `round6` below does it exactly. Zig's own float
//! formatter is *also* not a substitute: it rounds the shortest round-trip decimal,
//! half-up.
//!
//! ⚠️ **The sign flip is real.** `BatchedBrownianTree` sorts its query times and
//! multiplies by the product of the construction-order and query-order signs. Since
//! sampling runs *down* the sigma axis while the tree was built over
//! `[sigma_min, sigma_max]`, every returned increment is **negated**. Statistically
//! irrelevant (the law is symmetric), bit-exactly essential.
//!
//! Levy area is never needed (`levy_area_approximation='none'`), so the `H`/`A`
//! machinery of `brownian_interval.py` is deliberately absent — only the `W`
//! branch of `_increment_and_space_time_levy_area` is ported. `_randn(H_seed)` in
//! the reference draws from an independent generator, so omitting it changes
//! nothing.

const std = @import("std");
const seed_seq = @import("seed_seq.zig");
const noise_src = @import("noise.zig");

/// `tol = 1e-6` → `round(x, 6)`.
const round_digits: comptime_int = 6;
const round_scale: u64 = 1_000_000;
/// torchsde's `BrownianTree` default.
const pool_size: usize = 24;

/// **Python's `round(x, 6)`.**
///
/// CPython's `double_round` renders `x` to 6 fractional digits with `_Py_dg_dtoa`
/// (exact, ties-to-even) and parses the decimal back with a correctly-rounded
/// strtod. This reproduces both halves: exact integer arithmetic on the mantissa
/// for the rounding, and `std.fmt.parseFloat` (correctly rounded) for the parse.
///
/// The naive `@round(x * 1e6) / 1e6` differs from this on values where `x * 1e6`
/// lands near a half-integer, because the multiply itself rounds first. Ties are
/// reachable in practice: an interval midpoint that happens to be an odd multiple
/// of 1/128 has exactly 7 decimal digits ending in 5.
pub fn round6(x: f64) f64 {
    if (!std.math.isFinite(x) or x == 0) return x;

    const bits: u64 = @bitCast(x);
    const negative = bits >> 63 != 0;
    const exp_bits: u11 = @truncate(bits >> 52);
    const frac = bits & ((@as(u64, 1) << 52) - 1);

    // |x| = m * 2^e, exactly.
    var m: u64 = frac;
    var e: i32 = undefined;
    if (exp_bits == 0) {
        e = -1074;
    } else {
        m |= @as(u64, 1) << 52;
        e = @as(i32, exp_bits) - 1075;
    }
    // e >= 0 means |x| is an integer, so rounding to any number of fractional
    // digits is the identity — and skips the overflow question entirely.
    if (e >= 0) return x;

    const k: u32 = @intCast(-e);
    // |x| * 10^6 = m * 10^6 / 2^k. m < 2^53 and 10^6 < 2^20, so the numerator is
    // under 2^73 and u128 is exact.
    const num: u128 = @as(u128, m) * round_scale;
    var q: u128 = undefined;
    if (k >= 128) {
        // |x| < 2^-75; the scaled value is far below 1/2.
        q = 0;
    } else {
        q = num >> @intCast(k);
        const rem = num - (q << @intCast(k));
        const half = @as(u128, 1) << @intCast(k - 1);
        if (rem > half or (rem == half and q & 1 == 1)) q += 1;
    }

    // Parse back the decimal N / 10^6, exactly as CPython hands the dtoa string to
    // strtod. Doing the division in floating point instead would round twice.
    var buf: [48]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{s}{d}.{d:0>6}", .{
        if (negative) "-" else "",
        q / round_scale,
        @as(u64, @intCast(q % round_scale)),
    }) catch unreachable;
    return std.fmt.parseFloat(f64, s) catch unreachable;
}

/// One node of the dyadic tree. `midway == null` means "not split yet" (a leaf).
const Node = struct {
    start: f64,
    end: f64,
    parent: u32,
    is_left: bool,
    /// `_set_spawn_key_and_depth`: `2 * parent.spawn + is_right`, root at 0. Together
    /// with `depth` this is the `spawn_key` handed to `SeedSequence`, so it is an
    /// intrinsic property of the node's position — the same tree always derives the
    /// same seeds regardless of the order nodes were created in.
    spawn: u64,
    depth: u32,
    /// Rounded midpoint once split.
    midway: ?f64 = null,
    left: u32 = no_node,
    right: u32 = no_node,
    /// Derived at split time (`_split_exact`) from the spawn key.
    w_seed: u32 = 0,

    const no_node: u32 = std.math.maxInt(u32);
};

/// A depth guard. The reference can itself recurse forever if an interval narrows
/// to adjacent 6-decimal values, so bail with an error rather than hang: the sigma
/// span is ~1e-2..1e2, i.e. at most ~28 halvings before hitting the 1e-6 grid.
const max_depth: u32 = 64;

/// `BrownianTreeNoiseSampler` — a Brownian path over `[t0, t1]`, queried by
/// interval and normalised to unit variance.
pub const NoiseSampler = struct {
    gpa: std.mem.Allocator,
    /// Element count of one sample (the latent's length).
    n: usize,
    nodes: std.ArrayList(Node),
    entropy: u64,
    /// The root increment `W([t0, t1])`, held permanently (the reference keeps it
    /// pinned in its cache).
    root_w: []f32,
    /// Scratch: the increment being carried down the tree, and one `randn` draw.
    path_w: []f32,
    noise: []f32,
    /// Which generator every node's draw comes from. ⚠️ NOT a detail: the k-diffusion
    /// commit A1111 pins builds the tree on the latent's own (CUDA) device, so under
    /// A1111 these are Philox draws, while ComfyUI's fork forces them to the CPU.
    src: noise_src.Source,

    /// `t0`/`t1` are the schedule's `min(sigma > 0)` and `max(sigma)` — taken from
    /// the sigmas the caller will actually query, **before** any first-sigma SNR
    /// offset, matching the order in `sample_dpmpp_2m_sde`.
    pub fn init(gpa: std.mem.Allocator, n: usize, t0: f32, t1: f32, seed: u64, src: noise_src.Source) !NoiseSampler {
        std.debug.assert(t0 < t1);

        var nodes: std.ArrayList(Node) = .empty;
        errdefer nodes.deinit(gpa);
        try nodes.append(gpa, .{
            .start = round6(t0),
            .end = round6(t1),
            .parent = Node.no_node,
            .is_left = false,
            .spawn = 0,
            .depth = 0,
        });

        const root_w = try gpa.alloc(f32, n);
        errdefer gpa.free(root_w);
        const path_w = try gpa.alloc(f32, n);
        errdefer gpa.free(path_w);
        const noise = try gpa.alloc(f32, n);
        errdefer gpa.free(noise);

        // The root's increment. ⚠️ `math.sqrt(t1 - t0)` in the reference uses the
        // **unrounded** endpoints, unlike every child computation, which uses the
        // rounded `_start`/`_end`.
        var state: [3]u32 = undefined;
        seed_seq.SeedSequence.init(seed, &.{}, pool_size).generateState(&state);
        noise_src.randn(root_w, state[0], src);
        // f32 arithmetic: torch multiplies a f32 tensor by a Python float in f32.
        const scale: f32 = @floatCast(@sqrt(@as(f64, t1) - @as(f64, t0)));
        for (root_w) |*w| w.* *= scale;

        return .{
            .gpa = gpa,
            .n = n,
            .nodes = nodes,
            .entropy = seed,
            .root_w = root_w,
            .path_w = path_w,
            .noise = noise,
            .src = src,
        };
    }

    pub fn deinit(self: *NoiseSampler) void {
        self.nodes.deinit(self.gpa);
        self.gpa.free(self.root_w);
        self.gpa.free(self.path_w);
        self.gpa.free(self.noise);
        self.* = undefined;
    }

    /// The unit-variance noise for a sampler step from `sigma` down to `sigma_next`.
    ///
    /// This is `BrownianTreeNoiseSampler.__call__`: locate `[sigma_next, sigma]` in
    /// the tree, sum the increments of the covering nodes, **negate** (the sort-sign
    /// product, see the module doc), and divide by `sqrt(|sigma_next - sigma|)`
    /// computed in f32 from the *unrounded* sigmas.
    pub fn sample(self: *NoiseSampler, out: []f32, sigma: f32, sigma_next: f32) !void {
        std.debug.assert(out.len == self.n);
        std.debug.assert(sigma != sigma_next);

        const lo: f32 = @min(sigma, sigma_next);
        const hi: f32 = @max(sigma, sigma_next);
        // Clamp to the root's (rounded) span the way `__call__` does before it
        // rounds; `round6` is monotonic, so clamping either side of it agrees.
        const root = self.nodes.items[0];
        const ta = round6(std.math.clamp(@as(f64, lo), root.start, root.end));
        const tb = round6(std.math.clamp(@as(f64, hi), root.start, root.end));

        @memset(out, 0);
        if (ta < tb) try self.accumulate(0, ta, tb, out, 0);

        // The construction sort ([sigma_min, sigma_max]) gives +1; a descending
        // query ([sigma, sigma_next]) gives -1.
        const sign: f32 = if (sigma > sigma_next) -1 else 1;
        const denom: f32 = @sqrt(@abs(sigma_next - sigma));
        for (out) |*o| o.* = o.* * sign / denom;
    }

    /// `_loc_inner` + increment summation, fused: walk the tree for the nodes
    /// covering `[ta, tb]`, splitting as needed, and add each one's increment into
    /// `out` left to right (the reference's summation order, so the f32 rounding
    /// matches).
    ///
    /// ⚠️ **The "bounce up to the parent" branch is load-bearing, not defensive.**
    /// A node only has jurisdiction over its own span, and `_split` in halfway mode
    /// cuts at the *midpoint* rather than at the requested boundary — so after
    /// splitting, the child the reference descends into is routinely too narrow for
    /// the query. It bounces back to the parent, which by then has a midpoint and
    /// takes the straddle path. Omitting the bounce sends the walk down the right
    /// edge forever.
    fn accumulate(self: *NoiseSampler, node_idx: u32, ta: f64, tb: f64, out: []f32, depth: u32) !void {
        if (depth > max_depth) return error.BrownianTreeTooDeep;
        var idx = node_idx;
        var a = ta;
        while (true) {
            const node = self.nodes.items[idx];
            if (a < node.start or tb > node.end) {
                if (node.parent == Node.no_node) return error.BrownianQueryOutOfRange;
                idx = node.parent;
                continue;
            }
            if (a == node.start and tb == node.end) {
                try self.incrementInto(idx);
                for (out, self.path_w) |*o, w| o.* += w;
                return;
            }
            if (node.midway == null) {
                if (a == node.start) {
                    try self.split(idx, tb);
                    idx = self.nodes.items[idx].left;
                } else {
                    try self.split(idx, a);
                    idx = self.nodes.items[idx].right;
                }
                continue;
            }
            const midway = node.midway.?;
            if (tb <= midway) {
                idx = node.left;
                continue;
            }
            if (a >= midway) {
                idx = node.right;
                continue;
            }
            // The query straddles the midpoint: take the left part, then continue
            // into the right child.
            try self.accumulate(node.left, a, midway, out, depth + 1);
            idx = node.right;
            a = midway;
        }
    }

    /// `_split`: in halfway-tree mode a node is always cut at its own midpoint, then
    /// the child containing `midway` is cut again, until `midway` lands on an edge.
    fn split(self: *NoiseSampler, node_idx: u32, midway: f64) !void {
        var idx = node_idx;
        while (true) {
            const node = self.nodes.items[idx];
            std.debug.assert(node.midway == null);
            if (node.depth >= max_depth) return error.BrownianTreeTooDeep;
            const half = round6(0.5 * (node.end + node.start));
            // Both endpoints sit on the 1e-6 grid and `midway` is strictly inside, so
            // the span is at least 2e-6 and the midpoint is strictly inside too. If it
            // is not, the grid is exhausted and the reference would recurse forever.
            if (half <= node.start or half >= node.end) return error.BrownianTreeTooDeep;
            try self.splitExact(idx, half);
            if (midway > half) {
                idx = self.nodes.items[idx].right;
            } else if (midway < half) {
                idx = self.nodes.items[idx].left;
            } else return;
        }
    }

    /// `_split_exact`: record the midpoint, derive the node's seeds from its spawn
    /// key, and create both children.
    fn splitExact(self: *NoiseSampler, node_idx: u32, midway: f64) !void {
        const parent = self.nodes.items[node_idx];
        var state: [4]u32 = undefined;
        seed_seq.SeedSequence.init(self.entropy, &.{ parent.spawn, parent.depth }, pool_size)
            .generateState(&state);

        const left: u32 = @intCast(self.nodes.items.len);
        try self.nodes.ensureUnusedCapacity(self.gpa, 2);
        self.nodes.appendAssumeCapacity(.{
            .start = parent.start,
            .end = midway,
            .parent = node_idx,
            .is_left = true,
            .spawn = 2 * parent.spawn,
            .depth = parent.depth + 1,
        });
        self.nodes.appendAssumeCapacity(.{
            .start = midway,
            .end = parent.end,
            .parent = node_idx,
            .is_left = false,
            .spawn = 2 * parent.spawn + 1,
            .depth = parent.depth + 1,
        });

        const node = &self.nodes.items[node_idx];
        node.midway = midway;
        node.w_seed = state[0]; // state[1..3] are the H and Levy-area seeds
        node.left = left;
        node.right = left + 1;
    }

    /// The increment `W` over one node's interval, left in `self.path_w`.
    ///
    /// The reference recurses upward through an LRU cache; this walks the root→node
    /// path and carries `W` down, which costs one `randn` per level and no cache.
    /// The values are identical — the cache is purely a speed device there.
    fn incrementInto(self: *NoiseSampler, node_idx: u32) !void {
        // Collect the path root→node. Depth is bounded by `max_depth` by construction.
        var path: [max_depth + 2]u32 = undefined;
        var len: usize = 0;
        var cur = node_idx;
        while (true) {
            if (len >= path.len) return error.BrownianTreeTooDeep;
            path[len] = cur;
            len += 1;
            if (cur == 0) break;
            cur = self.nodes.items[cur].parent;
        }
        std.mem.reverse(u32, path[0..len]);

        @memcpy(self.path_w, self.root_w);
        for (path[0 .. len - 1], path[1..len]) |parent_idx, child_idx| {
            const parent = self.nodes.items[parent_idx];
            const child = self.nodes.items[child_idx];
            const h_reciprocal = 1.0 / (parent.end - parent.start);
            const left_diff = parent.midway.? - parent.start;
            const right_diff = parent.end - parent.midway.?;

            noise_src.randn(self.noise, parent.w_seed, self.src);

            // ⚠️ f32 throughout, and in the reference's order: a Python float times
            // a f32 tensor is computed in f32, so each scalar is narrowed *before*
            // it multiplies. Folding `left_diff * h_reciprocal` into one f64 scalar
            // would be a different (more accurate, non-matching) computation.
            const ld: f32 = @floatCast(left_diff);
            const hr: f32 = @floatCast(h_reciprocal);
            const sd: f32 = @floatCast(@sqrt(left_diff * right_diff * h_reciprocal));
            if (child.is_left) {
                for (self.path_w, self.noise) |*w, z| w.* = ld * w.* * hr + sd * z;
            } else {
                for (self.path_w, self.noise) |*w, z| w.* = w.* - (ld * w.* * hr + sd * z);
            }
        }
    }
};

// --- tests -----------------------------------------------------------------

test "round6 is Python's round(x, 6), not a scaled @round" {
    // Ground truth from CPython 3.12 (`round(x, 6)`).
    try std.testing.expectEqual(@as(f64, 0.007812), round6(0.0078125));
    try std.testing.expectEqual(@as(f64, 0.023438), round6(0.0234375)); // 3/128, ties up to even
    try std.testing.expectEqual(@as(f64, 0.0), round6(0.0000004));
    try std.testing.expectEqual(@as(f64, 0.000001), round6(0.0000006));
    try std.testing.expectEqual(@as(f64, 14.614642), round6(14.6146416664123535));
    try std.testing.expectEqual(@as(f64, 0.029168), round6(0.0291675031185150));
    try std.testing.expectEqual(@as(f64, -0.007812), round6(-0.0078125));
    // Integers and exactly-representable values pass through.
    try std.testing.expectEqual(@as(f64, 1.0), round6(1.0));
    try std.testing.expectEqual(@as(f64, 0.5), round6(0.5));
    try std.testing.expectEqual(@as(f64, 0.0), round6(0.0));

    // The tie cases above are exactly where the naive form disagrees, which is the
    // reason this function exists.
    try std.testing.expect(@round(0.0078125 * 1e6) / 1e6 != round6(0.0078125));
}

test "the Brownian path is additive over adjacent intervals" {
    // The defining property, and the one a per-step randn does not have:
    // W([a,c]) == W([a,b]) + W([b,c]). Checked through the public `sample`, which
    // divides by sqrt(dt), so the increments are recovered by multiplying back.
    const gpa = std.testing.allocator;
    var ns = try NoiseSampler.init(gpa, 64, 0.03, 14.6, 1234, .torch_cpu);
    defer ns.deinit();

    var w_ac: [64]f32 = undefined;
    var w_ab: [64]f32 = undefined;
    var w_bc: [64]f32 = undefined;
    const a: f32 = 8.0;
    const b: f32 = 4.0;
    const c: f32 = 1.0;
    try ns.sample(&w_ac, a, c);
    try ns.sample(&w_ab, a, b);
    try ns.sample(&w_bc, b, c);
    for (0..64) |i| {
        errdefer std.debug.print("i={d} ac={d} ab={d} bc={d}\n", .{ i, w_ac[i], w_ab[i], w_bc[i] });
        const lhs = w_ac[i] * @sqrt(a - c);
        const rhs = w_ab[i] * @sqrt(a - b) + w_bc[i] * @sqrt(b - c);
        try std.testing.expectApproxEqAbs(lhs, rhs, 1e-4);
    }
}

test "the Brownian path depends on the seed and only on the seed" {
    const gpa = std.testing.allocator;
    var out_a: [32]f32 = undefined;
    var out_b: [32]f32 = undefined;

    // Same seed, same query -> identical, regardless of what was queried before.
    {
        var ns = try NoiseSampler.init(gpa, 32, 0.03, 14.6, 7, .torch_cpu);
        defer ns.deinit();
        try ns.sample(&out_a, 5.0, 3.0);
    }
    {
        var ns = try NoiseSampler.init(gpa, 32, 0.03, 14.6, 7, .torch_cpu);
        defer ns.deinit();
        // A different query order must not perturb the path (`halfway_tree=True`).
        var scratch: [32]f32 = undefined;
        try ns.sample(&scratch, 14.0, 9.0);
        try ns.sample(&scratch, 9.0, 5.0);
        try ns.sample(&out_b, 5.0, 3.0);
    }
    try std.testing.expectEqualSlices(f32, &out_a, &out_b);

    // A different seed gives a different path.
    {
        var ns = try NoiseSampler.init(gpa, 32, 0.03, 14.6, 8, .torch_cpu);
        defer ns.deinit();
        try ns.sample(&out_b, 5.0, 3.0);
    }
    try std.testing.expect(!std.mem.eql(f32, &out_a, &out_b));
}

test "matches torchsde's BrownianTree through ComfyUI's BrownianTreeNoiseSampler" {
    // ⚠️ This is the test that makes the whole port worth doing: **bit-exact**, not
    // approximate. Every ingredient has to be right simultaneously — numpy's
    // SeedSequence, the spawn key of every node on the path, Python's `round(x, 6)`,
    // torch's `randn`, the f32 ordering of the increment recursion, and the sort-sign
    // negation. Any one of them being wrong gives statistically indistinguishable
    // noise, so a tolerance-based check here would prove nothing.
    //
    // The fixture's queries deliberately include out-of-sweep-order ones (pinning the
    // halfway-tree property) and one interval spanning several nodes (pinning the
    // multi-node summation order). Generated by `tools/gen_sampler_fixtures.py`.
    const gpa = std.testing.allocator;
    const json_text = @embedFile("assets/dpmpp_sde_fixtures.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
    defer parsed.deinit();
    const fx = parsed.value.object.get("brownian").?.object;

    const n: usize = @intCast(fx.get("n").?.integer);
    const t0: f32 = @floatCast(fx.get("t0").?.float);
    const t1: f32 = @floatCast(fx.get("t1").?.float);
    const seed: u64 = @intCast(fx.get("seed").?.integer);

    var ns = try NoiseSampler.init(gpa, n, t0, t1, seed, .torch_cpu);
    defer ns.deinit();

    const got = try gpa.alloc(f32, n);
    defer gpa.free(got);

    const queries = fx.get("queries").?.array.items;
    const samples = fx.get("samples").?.array.items;
    try std.testing.expectEqual(queries.len, samples.len);
    for (queries, samples, 0..) |q, s, qi| {
        const sigma: f32 = @floatCast(q.array.items[0].float);
        const sigma_next: f32 = @floatCast(q.array.items[1].float);
        try ns.sample(got, sigma, sigma_next);
        for (s.array.items, got, 0..) |want_v, a, j| {
            const want: f32 = @floatCast(want_v.float);
            errdefer std.debug.print(
                "query {d} ({d} -> {d}) element {d}: expected {d} got {d}\n",
                .{ qi, sigma, sigma_next, j, want, a },
            );
            try std.testing.expectEqual(want, a);
        }
    }
}

test "the increment is unit-variance after normalisation" {
    // sample() divides by sqrt(dt), so each component should look standard normal.
    const gpa = std.testing.allocator;
    const n = 4096;
    var ns = try NoiseSampler.init(gpa, n, 0.0292, 14.6146, 42, .torch_cpu);
    defer ns.deinit();
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    try ns.sample(out, 3.0, 1.5);

    var mean: f64 = 0;
    for (out) |v| mean += v;
    mean /= n;
    var variance: f64 = 0;
    for (out) |v| variance += (v - mean) * (v - mean);
    variance /= n;
    errdefer std.debug.print("mean {d:.4} var {d:.4}\n", .{ mean, variance });
    try std.testing.expect(@abs(mean) < 0.06);
    try std.testing.expect(@abs(variance - 1.0) < 0.06);
}
