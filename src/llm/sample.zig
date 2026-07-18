//! Logits -> token id: greedy argmax, or temperature + top-k + top-p
//! (nucleus) sampling with an optional repetition penalty.
//!
//! Defaults follow Qwen3's non-thinking-mode recommendation
//! (temperature 0.7, top_p 0.8, top_k 20). temperature == 0 is greedy.

const std = @import("std");

pub const Params = struct {
    /// 0 = greedy (argmax); otherwise logits are divided by this.
    temperature: f32 = 0.7,
    /// Keep only the k highest logits (0 = no limit).
    top_k: usize = 20,
    /// Nucleus: smallest prefix of the (sorted) distribution with cumulative
    /// probability >= top_p. 1.0 = no limit.
    top_p: f32 = 0.8,
    /// llama.cpp-style repetition penalty over the recent window (1.0 = off):
    /// positive logits of recent ids are divided by this, negative multiplied.
    repeat_penalty: f32 = 1.0,
    /// How many trailing context tokens the repetition penalty looks at.
    repeat_last_n: usize = 64,
};

/// Hard cap on sampling candidates when top_k = 0; softmax over the full
/// 152k vocab would be pointless (everything past the top few hundred is
/// noise) and this keeps the candidate buffer fixed-size.
const max_candidates = 512;

/// The fully processed next-token distribution (repetition penalty,
/// temperature, top-k, top-p all applied): the exact distribution `next`
/// draws from, exposed so speculative decoding can verify drafted tokens
/// against it. Greedy (temperature 0) is a point mass on the argmax.
pub const Dist = struct {
    ids: [max_candidates]u32,
    /// Normalized over the kept candidates, descending.
    probs: [max_candidates]f32,
    n: usize,

    /// Probability of `id` under this distribution (0 when filtered out).
    pub fn probOf(self: *const Dist, id: u32) f32 {
        for (self.ids[0..self.n], self.probs[0..self.n]) |i, p| {
            if (i == id) return p;
        }
        return 0;
    }

    pub fn sample(self: *const Dist, rand: std.Random) u32 {
        if (self.n == 1) return self.ids[0]; // greedy: no rng draw
        var r = rand.float(f32);
        for (self.ids[0..self.n], self.probs[0..self.n]) |id, p| {
            r -= p;
            if (r <= 0) return id;
        }
        return self.ids[self.n - 1]; // float round-off fallthrough
    }

    /// Draw from the distribution with `excl` removed and the rest
    /// renormalized — the speculative-decode residual for a rejected token
    /// proposed by a deterministic drafter (max(p - q, 0) with q a point
    /// mass at `excl`).
    pub fn sampleExcluding(self: *const Dist, rand: std.Random, excl: u32) u32 {
        std.debug.assert(self.n > 0);
        const mass = 1 - self.probOf(excl);
        if (mass <= 0) return self.ids[0]; // degenerate: p was a point mass on excl
        var fallback: u32 = self.ids[0];
        if (self.n == 1) return fallback; // greedy: excl lost to the argmax, no rng draw
        var r = rand.float(f32) * mass;
        for (self.ids[0..self.n], self.probs[0..self.n]) |id, p| {
            if (id == excl) continue;
            r -= p;
            if (r <= 0) return id;
            fallback = id;
        }
        return fallback;
    }
};

pub const Sampler = struct {
    params: Params,
    rng: std.Random.DefaultPrng,

    pub fn init(params: Params, seed: u64) Sampler {
        return .{ .params = params, .rng = std.Random.DefaultPrng.init(seed) };
    }

    /// Pick the next token. `logits` is modified in place (repetition
    /// penalty); `recent` is the trailing context window for the penalty.
    pub fn next(self: *Sampler, logits: []f32, recent: []const u32) u32 {
        const d = self.dist(logits, recent);
        return d.sample(self.rng.random());
    }

    /// Speculative-decode acceptance: keep a drafted token with probability
    /// equal to its mass under the target distribution (the drafter is
    /// deterministic, so q(draft) = 1). Greedy draws no randomness.
    pub fn accept(self: *Sampler, d: *const Dist, draft: u32) bool {
        const p = d.probOf(draft);
        if (p >= 1) return true;
        if (p <= 0) return false;
        return self.rng.random().float(f32) < p;
    }

    /// Build the processed next-token distribution. `logits` is modified in
    /// place (repetition penalty); `recent` is the penalty's context window.
    pub fn dist(self: *Sampler, logits: []f32, recent: []const u32) Dist {
        const p = self.params;
        applyRepetitionPenalty(logits, recent, p);
        if (p.temperature <= 0) {
            return distFromSorted(p, &.{.{ .id = argmax(logits), .logit = 0 }});
        }
        // Top-k candidates (id, logit), highest first — the only full-vocab
        // step; everything downstream is on this <= max_candidates set.
        const k = candidateCount(p);
        var cand: [max_candidates]Candidate = undefined;
        const cands = topK(logits, cand[0..k]);
        return distFromSorted(p, cands);
    }

    /// GPU-sampling entry point: the device already applied the repetition
    /// penalty and selected the top-k, downloading just these candidates
    /// (a few KB vs the full ~608 KB vocab). We sort them descending — k is
    /// tiny — and run the identical softmax / top-p / normalize / RNG tail as
    /// the full-vocab path, so the emitted token is bit-identical to what the
    /// CPU sampler would have produced from the same logits.
    pub fn nextFromCandidates(self: *Sampler, cands: []Candidate) u32 {
        std.mem.sort(Candidate, cands, {}, candDesc);
        // Trim the (possibly larger, e.g. GPU per-lane) candidate pool to the
        // top-k the params consider — matching the CPU path's topK(k) so the
        // softmax runs over the same set.
        const k = @min(candidateCount(self.params), cands.len);
        const d = distFromSorted(self.params, cands[0..k]);
        return d.sample(self.rng.random());
    }
};

/// Number of top-k candidates the sampler considers for the given params
/// (0 = "no limit" caps at max_candidates). The GPU top-k selects the same
/// count so its candidate set matches the CPU path's.
pub fn candidateCount(p: Params) usize {
    return @min(if (p.top_k == 0) max_candidates else p.top_k, max_candidates);
}

/// llama.cpp-style repetition penalty over the recent window, applied in place
/// to the full logits (`p.repeat_penalty == 1.0` is a no-op). Extracted so the
/// GPU path can mirror the exact formula in a device kernel before its top-k.
pub fn applyRepetitionPenalty(logits: []f32, recent: []const u32, p: Params) void {
    if (p.repeat_penalty == 1.0) return;
    const n = @min(p.repeat_last_n, recent.len);
    for (recent[recent.len - n ..]) |id| {
        const l = &logits[id];
        l.* = if (l.* > 0) l.* / p.repeat_penalty else l.* * p.repeat_penalty;
    }
}

/// The post-top-k tail: temperature softmax + nucleus (top-p) cut + normalize,
/// over candidates already SORTED descending by logit. Shared by the full-vocab
/// `dist` and the GPU candidate path so there is one source of truth for the
/// stochastic math (and thus exact CPU/GPU parity). `temperature <= 0` collapses
/// to a point mass on `cands[0]` (which must be the argmax). Repetition penalty
/// is assumed already applied to the logits these candidates came from.
pub fn distFromSorted(p: Params, cands: []const Candidate) Dist {
    std.debug.assert(cands.len >= 1);
    var d: Dist = undefined;
    if (p.temperature <= 0) {
        d.n = 1;
        d.ids[0] = cands[0].id;
        d.probs[0] = 1;
        return d;
    }

    // Softmax with temperature over the candidates.
    var probs: [max_candidates]f32 = undefined;
    var denom: f32 = 0;
    for (cands, 0..) |c, i| {
        probs[i] = @exp((c.logit - cands[0].logit) / p.temperature);
        denom += probs[i];
    }

    // Nucleus cut: candidates are sorted, so walk until cumulative >= top_p.
    var keep: usize = cands.len;
    if (p.top_p < 1.0) {
        var cum: f32 = 0;
        for (probs[0..cands.len], 0..) |prob, i| {
            cum += prob / denom;
            if (cum >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }

    // Normalize the kept prefix.
    var kept_mass: f32 = 0;
    for (probs[0..keep]) |prob| kept_mass += prob;
    for (cands[0..keep], probs[0..keep], 0..) |c, prob, i| {
        d.ids[i] = c.id;
        d.probs[i] = prob / kept_mass;
    }
    d.n = keep;
    return d;
}

pub const Candidate = struct { id: u32, logit: f32 };

/// Descending by logit, ties broken by ascending id (matches argmax's
/// lowest-index-wins so a GPU-selected set sorts identically to topK's output).
pub fn candDesc(_: void, a: Candidate, b: Candidate) bool {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

/// Fill `out` with the highest-logit candidates, sorted descending.
/// Selection over the full vocab via a min-heap keyed on the smallest kept
/// logit (out.len is small: <= max_candidates).
fn topK(logits: []const f32, out: []Candidate) []Candidate {
    const k = @min(out.len, logits.len);
    const heap = out[0..k];
    // Heap-order the first k, min at root.
    for (logits[0..k], 0..) |l, i| siftUp(heap, i, .{ .id = @intCast(i), .logit = l });
    for (logits[k..], k..) |l, i| {
        if (l > heap[0].logit) {
            heap[0] = .{ .id = @intCast(i), .logit = l };
            siftDown(heap, 0);
        }
    }
    // Heap-sort into descending order: repeatedly move the min to the back.
    var end = k;
    while (end > 1) {
        end -= 1;
        std.mem.swap(Candidate, &heap[0], &heap[end]);
        siftDown(heap[0..end], 0);
    }
    return heap;
}

fn siftUp(heap: []Candidate, at: usize, val: Candidate) void {
    var i = at;
    heap[i] = val;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (heap[parent].logit <= heap[i].logit) break;
        std.mem.swap(Candidate, &heap[parent], &heap[i]);
        i = parent;
    }
}

fn siftDown(heap: []Candidate, at: usize) void {
    var i = at;
    while (true) {
        var min = i;
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        if (l < heap.len and heap[l].logit < heap[min].logit) min = l;
        if (r < heap.len and heap[r].logit < heap[min].logit) min = r;
        if (min == i) break;
        std.mem.swap(Candidate, &heap[i], &heap[min]);
        i = min;
    }
}

/// Lowest index wins ties, matching torch.argmax.
pub fn argmax(logits: []const f32) u32 {
    std.debug.assert(logits.len > 0);
    var best: u32 = 0;
    var best_val = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best = @intCast(i);
        }
    }
    return best;
}

// --- tests -----------------------------------------------------------------

test "argmax picks the maximum" {
    try std.testing.expectEqual(@as(u32, 2), argmax(&.{ -1.0, 0.5, 3.0, 2.9 }));
    try std.testing.expectEqual(@as(u32, 0), argmax(&.{5.0}));
}

test "argmax tie-breaks to the lowest index" {
    try std.testing.expectEqual(@as(u32, 1), argmax(&.{ 0.0, 7.0, 7.0, 7.0 }));
}

test "topK returns the k largest, descending" {
    const logits = [_]f32{ 0.1, 5.0, -2.0, 3.0, 4.0, -0.5 };
    var buf: [3]Candidate = undefined;
    const got = topK(&logits, &buf);
    try std.testing.expectEqual(@as(u32, 1), got[0].id);
    try std.testing.expectEqual(@as(u32, 4), got[1].id);
    try std.testing.expectEqual(@as(u32, 3), got[2].id);
}

test "temperature 0 is greedy" {
    var s = Sampler.init(.{ .temperature = 0 }, 42);
    var logits = [_]f32{ 1.0, 9.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), s.next(&logits, &.{}));
}

test "top_k 1 always picks the max regardless of seed" {
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 1, .top_p = 1.0 }, 7);
    const logits = [_]f32{ 1.0, 2.0, 9.0, 3.0 };
    for (0..8) |_| {
        var copy = logits;
        try std.testing.expectEqual(@as(u32, 2), s.next(&copy, &.{}));
    }
}

test "top_p excludes the tail" {
    // Two dominant, equal logits and a far-off tail: top_p = 0.9 keeps
    // exactly the two dominant ids (each ~0.5 mass).
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, 1234);
    var seen_tail = false;
    for (0..64) |_| {
        var logits = [_]f32{ 10.0, 10.0, -10.0, -10.0 };
        const id = s.next(&logits, &.{});
        if (id >= 2) seen_tail = true;
    }
    try std.testing.expect(!seen_tail);
}

test "repetition penalty suppresses a repeated token" {
    var s = Sampler.init(.{ .temperature = 0, .repeat_penalty = 2.0 }, 0);
    // Token 0 leads, but was just emitted; penalty halves it below token 1.
    var logits = [_]f32{ 3.0, 2.0, 0.0 };
    const recent = [_]u32{0};
    try std.testing.expectEqual(@as(u32, 1), s.next(&logits, &recent));
}

test "dist matches next() and normalizes" {
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 3, .top_p = 1.0 }, 5);
    var logits = [_]f32{ 2.0, 1.0, 0.0, -50.0 };
    const d = s.dist(&logits, &.{});
    try std.testing.expectEqual(@as(usize, 3), d.n);
    try std.testing.expectEqual(@as(u32, 0), d.ids[0]);
    var sum: f32 = 0;
    for (d.probs[0..d.n]) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    try std.testing.expectEqual(@as(f32, 0), d.probOf(3)); // filtered out
    try std.testing.expect(d.probOf(0) > d.probOf(1));
}

test "greedy dist is a point mass and draws no rng" {
    var s = Sampler.init(.{ .temperature = 0 }, 0);
    var logits = [_]f32{ 0.0, 4.0, 1.0 };
    const d = s.dist(&logits, &.{});
    try std.testing.expectEqual(@as(usize, 1), d.n);
    try std.testing.expectEqual(@as(f32, 1), d.probOf(1));
    try std.testing.expect(s.accept(&d, 1));
    try std.testing.expect(!s.accept(&d, 0));
    // Residual after a greedy rejection is the argmax itself.
    try std.testing.expectEqual(@as(u32, 1), d.sampleExcluding(s.rng.random(), 0));
}

// Speculative acceptance is lossless: accept draft w.p. p(draft), else
// resample from the renormalized residual. The emitted-token distribution
// must equal the target distribution itself.
test "accept + sampleExcluding preserves the target distribution" {
    const logits_base = [_]f32{ 1.5, 1.0, 0.5, 0.0 };
    const draft: u32 = 1; // a mid-probability candidate
    var counts_spec = [_]usize{0} ** 4;
    var counts_direct = [_]usize{0} ** 4;
    const trials = 20000;

    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 424242);
    for (0..trials) |_| {
        var l1 = logits_base;
        const d = s.dist(&l1, &.{});
        const emitted = if (s.accept(&d, draft)) draft else d.sampleExcluding(s.rng.random(), draft);
        counts_spec[emitted] += 1;
        var l2 = logits_base;
        counts_direct[s.next(&l2, &.{})] += 1;
    }
    for (counts_spec, counts_direct) |a, b| {
        const fa = @as(f64, @floatFromInt(a)) / trials;
        const fb = @as(f64, @floatFromInt(b)) / trials;
        try std.testing.expectApproxEqAbs(fa, fb, 0.02); // ~4 sigma at n=20k
    }
}

// The GPU path (device top-k → download candidates → nextFromCandidates) must
// emit the SAME token as the CPU full-vocab next() for the same logits + seed.
test "nextFromCandidates matches full-vocab next()" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    const params_sets = [_]Params{
        .{ .temperature = 0.0 }, // greedy
        .{ .temperature = 1.0, .top_k = 20, .top_p = 1.0 },
        .{ .temperature = 0.7, .top_k = 0, .top_p = 0.8 }, // top_k=0 → cap
        .{ .temperature = 1.3, .top_k = 5, .top_p = 0.95 },
    };
    for (params_sets) |p| {
        for (0..16) |trial| {
            var logits: [4096]f32 = undefined;
            for (&logits) |*l| l.* = rand.floatNorm(f32) * 4.0;
            const seed: u64 = 100 + trial;

            // CPU full-vocab path.
            var cpu = Sampler.init(p, seed);
            var l_cpu = logits;
            const want = cpu.next(&l_cpu, &.{});

            // GPU path: select the same top-k the device would, hand it over.
            var l_gpu = logits;
            applyRepetitionPenalty(&l_gpu, &.{}, p);
            const k = if (p.temperature <= 0) 1 else candidateCount(p);
            var cand: [max_candidates]Candidate = undefined;
            const cands = topK(&l_gpu, cand[0..k]);
            var gpu = Sampler.init(p, seed);
            const got = gpu.nextFromCandidates(cands);

            try std.testing.expectEqual(want, got);
        }
    }
}

test "sampling stays within the top-k set and is seed-deterministic" {
    const logits_base = [_]f32{ 5.0, 4.9, 4.8, -100.0, -100.0 };
    var a = Sampler.init(.{ .temperature = 1.0, .top_k = 3, .top_p = 1.0 }, 99);
    var b = Sampler.init(.{ .temperature = 1.0, .top_k = 3, .top_p = 1.0 }, 99);
    for (0..32) |_| {
        var la = logits_base;
        var lb = logits_base;
        const ia = a.next(&la, &.{});
        const ib = b.next(&lb, &.{});
        try std.testing.expectEqual(ia, ib);
        try std.testing.expect(ia < 3);
    }
}
