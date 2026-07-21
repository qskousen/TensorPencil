//! Logits -> token id: greedy argmax, or temperature + top-k + top-p
//! (nucleus) + min-p sampling with optional repetition / presence /
//! frequency penalties (llama.cpp semantics).
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
    /// Min-p: drop candidates whose probability is below min_p times the top
    /// candidate's (0 = off). Thresholded in raw-logit space, independent of
    /// temperature — matches llama.cpp's min_p sampler.
    min_p: f32 = 0.0,
    /// llama.cpp-style repetition penalty over the recent window (1.0 = off):
    /// positive logits of recent ids are divided by this, negative multiplied.
    /// Applied once per unique token regardless of its count in the window.
    repeat_penalty: f32 = 1.0,
    /// How many trailing context tokens the penalties look at
    /// (capped at max_penalty_window; 0 disables all penalties).
    repeat_last_n: usize = 64,
    /// Flat penalty subtracted from the logit of every token that appears in
    /// the recent window (0 = off). Negative values boost repeats.
    presence_penalty: f32 = 0.0,
    /// Penalty subtracted per occurrence: logit -= count * frequency_penalty
    /// (0 = off).
    frequency_penalty: f32 = 0.0,

    /// Whether any recent-window penalty is enabled. The GPU sampling paths
    /// (on-device argmax / top-k) select candidates BEFORE penalties could be
    /// applied, so the engine falls back to the full-logit-download CPU path
    /// whenever this is true.
    pub fn penaltiesActive(p: Params) bool {
        return p.repeat_last_n != 0 and
            (p.repeat_penalty != 1.0 or p.presence_penalty != 0 or p.frequency_penalty != 0);
    }
};

/// Hard cap on sampling candidates when top_k = 0; softmax over the full
/// 152k vocab would be pointless (everything past the top few hundred is
/// noise) and this keeps the candidate buffer fixed-size.
const max_candidates = 512;

/// The fully processed next-token distribution (penalties, temperature,
/// top-k, top-p, min-p all applied): the exact distribution `next`
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

/// Per-turn seed sequence for multi-turn sessions. A Sampler is constructed
/// fresh for every generate call from `Options.seed`, so a session that reuses
/// one seed replays the identical RNG stream each turn — a repeated prompt
/// (new chat, or a regenerated reply) then reproduces the identical response.
/// Chat drivers hold one SeedSeq per session and pull `next()` at every turn
/// boundary instead: turns never share an RNG stream, while a fixed base seed
/// still reproduces the whole session deterministically.
pub const SeedSeq = struct {
    state: u64,

    pub fn init(base: u64) SeedSeq {
        return .{ .state = base };
    }

    /// The next per-turn seed: splitmix64 over an advancing counter. The
    /// finalizer is a bijection, so seeds within a session never collide.
    pub fn next(self: *SeedSeq) u64 {
        self.state +%= 0x9E3779B97F4A7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }
};

pub const Sampler = struct {
    params: Params,
    rng: std.Random.DefaultPrng,

    pub fn init(params: Params, seed: u64) Sampler {
        return .{ .params = params, .rng = std.Random.DefaultPrng.init(seed) };
    }

    /// Pick the next token. `logits` is modified in place (penalties);
    /// `recent` is the trailing context window for the penalties.
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
    /// place (penalties); `recent` is the penalties' context window.
    pub fn dist(self: *Sampler, logits: []f32, recent: []const u32) Dist {
        const p = self.params;
        applyPenalties(logits, recent, p);
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

    /// GPU-sampling entry point: the device selected the top-k, downloading
    /// just these candidates (a few KB vs the full ~608 KB vocab); the engine
    /// only routes here when no penalty is active (penalties need the full
    /// logits). We sort them descending — k is tiny — and run the identical
    /// softmax / min-p / top-p / normalize / RNG tail as the full-vocab path,
    /// so the emitted token is bit-identical to what the CPU sampler would
    /// have produced from the same logits.
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

/// Cap on how many trailing context tokens the penalties scan per step (bounds
/// the fixed-size scratch below); `repeat_last_n` above this is clamped.
pub const max_penalty_window = 2048;

/// One unique token of the penalty window: its id and occurrence count.
/// `count` is f32 (not integer) because it only ever feeds the frequency
/// term's float math — and so the GPU penalize kernels can consume the
/// entry buffer without an int→float convert.
pub const PenaltyEntry = extern struct { id: u32, count: f32 };

/// Collect the unique (id, count) pairs of the trailing penalty window,
/// sorted by id. This is THE penalty list: the CPU path applies it below and
/// the GPU paths upload it to their penalize kernel, so both penalize the
/// exact same tokens with the same counts. Empty when penalties are off.
pub fn collectPenalties(recent: []const u32, p: Params, out: *[max_penalty_window]PenaltyEntry) []PenaltyEntry {
    if (!p.penaltiesActive()) return out[0..0];
    const n: usize = @min(@min(p.repeat_last_n, recent.len), max_penalty_window);
    // Sort a copy of the window so each unique id is one run (id + count).
    var window: [max_penalty_window]u32 = undefined;
    @memcpy(window[0..n], recent[recent.len - n ..]);
    std.mem.sort(u32, window[0..n], {}, std.sort.asc(u32));
    var m: usize = 0;
    var i: usize = 0;
    while (i < n) {
        const id = window[i];
        var end = i + 1;
        while (end < n and window[end] == id) end += 1;
        out[m] = .{ .id = id, .count = @floatFromInt(end - i) };
        m += 1;
        i = end;
    }
    return out[0..m];
}

/// The per-token penalty formula (mirrored bit-for-bit by the device penalize
/// kernels): repetition divides a positive logit / multiplies a negative one —
/// once per unique token — then presence/frequency subtract.
pub fn penalizeLogit(l: f32, count: f32, p: Params) f32 {
    const r = if (l > 0) l / p.repeat_penalty else l * p.repeat_penalty;
    return r - (count * p.frequency_penalty + p.presence_penalty);
}

/// Wire scratch for the device penalize kernels: one (id, subtract) pair per
/// entry. The subtract term `count * frequency_penalty + presence_penalty` is
/// precomputed on the HOST with the exact two f32 ops penalizeLogit uses, so
/// the kernels need only the repeat-penalty scalar and stay bit-faithful to
/// the CPU formula.
pub const PenaltyWire = [2 * max_penalty_window]u32;

/// Pack entries for the CUDA penalize kernel: interleaved u32 words —
/// the token id, then the f32 bits of the precomputed subtract term.
pub fn packPenaltyWireU32(entries: []const PenaltyEntry, p: Params, out: *PenaltyWire) []const u32 {
    for (entries, 0..) |e, i| {
        out[2 * i] = e.id;
        out[2 * i + 1] = @bitCast(e.count * p.frequency_penalty + p.presence_penalty);
    }
    return out[0 .. 2 * entries.len];
}

/// Pack entries for the Vulkan penalize kernel, whose storage buffers are
/// f32-only: the token id stored AS f32 (exact below 2^24 — every vocab is)
/// and the same precomputed subtract term.
pub fn packPenaltyWireF32(entries: []const PenaltyEntry, p: Params, out: *[2 * max_penalty_window]f32) []const f32 {
    for (entries, 0..) |e, i| {
        out[2 * i] = @floatFromInt(e.id);
        out[2 * i + 1] = e.count * p.frequency_penalty + p.presence_penalty;
    }
    return out[0 .. 2 * entries.len];
}

/// llama.cpp-style penalties over the recent window, applied in place to the
/// full logits (a no-op when `p.penaltiesActive()` is false). Per unique token
/// in the window (count = its occurrences):
///   - repetition: positive logits divided by repeat_penalty, negative
///     multiplied — applied ONCE, regardless of count;
///   - presence/frequency: logit -= count * frequency_penalty + presence_penalty.
/// Matches llama.cpp's llama_sampler_penalties formula exactly.
pub fn applyPenalties(logits: []f32, recent: []const u32, p: Params) void {
    var scratch: [max_penalty_window]PenaltyEntry = undefined;
    for (collectPenalties(recent, p, &scratch)) |e| {
        logits[e.id] = penalizeLogit(logits[e.id], e.count, p);
    }
}

/// The post-top-k tail: temperature softmax + nucleus (top-p) + min-p cuts +
/// normalize, over candidates already SORTED descending by logit. Shared by the
/// full-vocab `dist` and the GPU candidate path so there is one source of truth
/// for the stochastic math (and thus exact CPU/GPU parity). `temperature <= 0`
/// collapses to a point mass on `cands[0]` (which must be the argmax).
/// Penalties are assumed already applied to the logits these candidates came
/// from. Both cuts are prefix cuts computed on the full candidate set and then
/// intersected — the same result as llama.cpp's top_p-then-min_p chain order.
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

    // Min-p cut: drop candidates with probability below min_p times the top's.
    // Thresholded in raw-logit space (temperature-independent, llama.cpp
    // semantics); the top candidate always survives.
    if (p.min_p > 0) {
        const min_logit = cands[0].logit + @log(p.min_p);
        var m: usize = 1;
        while (m < keep and cands[m].logit >= min_logit) m += 1;
        keep = m;
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

test "repetition penalty hits a repeated token once (llama.cpp semantics)" {
    var s = Sampler.init(.{ .temperature = 0, .repeat_penalty = 2.0 }, 0);
    // Token 0 appears TWICE in the window; the penalty still divides once
    // (3/2 = 1.5 > 1.0), not per occurrence (3/4 = 0.75 < 1.0).
    var logits = [_]f32{ 3.0, 1.0 };
    const recent = [_]u32{ 0, 0 };
    try std.testing.expectEqual(@as(u32, 0), s.next(&logits, &recent));
}

test "presence and frequency penalties scale with occurrence count" {
    // Token 0 leads by 0.1 and appeared twice: frequency subtracts per
    // occurrence (2 * 1.0), presence once (0.5) -> 2.0 - 2.5 = -0.5 < 1.9.
    var s = Sampler.init(.{ .temperature = 0, .presence_penalty = 0.5, .frequency_penalty = 1.0 }, 0);
    var logits = [_]f32{ 2.0, 1.9, 0.0 };
    const recent = [_]u32{ 0, 0 };
    try std.testing.expectEqual(@as(u32, 1), s.next(&logits, &recent));

    // Presence alone is flat: one vs three occurrences penalize the same.
    var p1 = [_]f32{ 2.0, 1.0 };
    var p3 = [_]f32{ 2.0, 1.0 };
    applyPenalties(&p1, &.{0}, .{ .presence_penalty = 0.7 });
    applyPenalties(&p3, &.{ 0, 0, 0 }, .{ .presence_penalty = 0.7 });
    try std.testing.expectEqual(p1[0], p3[0]);
}

test "repeat_last_n 0 disables all penalties" {
    var logits = [_]f32{ 2.0, 1.0 };
    applyPenalties(&logits, &.{ 0, 0 }, .{ .repeat_penalty = 2.0, .presence_penalty = 1.0, .repeat_last_n = 0 });
    try std.testing.expectEqual(@as(f32, 2.0), logits[0]);
}

test "collectPenalties dedups the window into id-sorted (id, count) entries" {
    const p: Params = .{ .repeat_penalty = 1.1 };
    var scratch: [max_penalty_window]PenaltyEntry = undefined;
    const recent = [_]u32{ 7, 3, 7, 7, 3, 9 };
    const entries = collectPenalties(&recent, p, &scratch);
    try std.testing.expectEqual(@as(usize, 3), entries.len);
    try std.testing.expectEqual(PenaltyEntry{ .id = 3, .count = 2 }, entries[0]);
    try std.testing.expectEqual(PenaltyEntry{ .id = 7, .count = 3 }, entries[1]);
    try std.testing.expectEqual(PenaltyEntry{ .id = 9, .count = 1 }, entries[2]);

    // Window trimming: only the trailing repeat_last_n tokens count.
    const short = collectPenalties(&recent, .{ .repeat_penalty = 1.1, .repeat_last_n = 2 }, &scratch);
    try std.testing.expectEqual(@as(usize, 2), short.len);
    try std.testing.expectEqual(@as(u32, 3), short[0].id);
    try std.testing.expectEqual(@as(u32, 9), short[1].id);

    // Penalties off -> empty (the GPU paths key "upload nothing" off this).
    try std.testing.expectEqual(@as(usize, 0), collectPenalties(&recent, .{}, &scratch).len);
}

// gemma4_cuda's device suppress mask reuses the penalize kernel with an
// infinite presence penalty: rp=1 leaves the logit exactly unchanged (x/1),
// and any finite value minus +inf is exactly -inf — this test pins that
// contract so a formula change can't silently break the mask.
test "infinite presence penalty is an exact -inf mask" {
    const p: Params = .{ .repeat_penalty = 1.0, .presence_penalty = std.math.inf(f32) };
    for ([_]f32{ 12.5, -3.25, 0.0, 3.4e38 }) |l| {
        try std.testing.expectEqual(-std.math.inf(f32), penalizeLogit(l, 1, p));
    }
}

test "penalty wire packing reproduces penalizeLogit exactly" {
    const p: Params = .{ .repeat_penalty = 1.3, .presence_penalty = 0.4, .frequency_penalty = 0.17 };
    const entries = [_]PenaltyEntry{ .{ .id = 5, .count = 1 }, .{ .id = 9, .count = 3 } };
    var wu: PenaltyWire = undefined;
    var wf: [2 * max_penalty_window]f32 = undefined;
    const u = packPenaltyWireU32(&entries, p, &wu);
    const f = packPenaltyWireF32(&entries, p, &wf);
    try std.testing.expectEqual(@as(usize, 4), u.len);
    for (entries, 0..) |e, i| {
        try std.testing.expectEqual(e.id, u[2 * i]);
        try std.testing.expectEqual(@as(f32, @floatFromInt(e.id)), f[2 * i]);
        const sub: f32 = @bitCast(u[2 * i + 1]);
        try std.testing.expectEqual(sub, f[2 * i + 1]);
        // Applying (l -> l/rp or l*rp, then - sub) must equal penalizeLogit.
        for ([_]f32{ 2.5, -1.75 }) |l| {
            const r = if (l > 0) l / p.repeat_penalty else l * p.repeat_penalty;
            try std.testing.expectEqual(penalizeLogit(l, e.count, p), r - sub);
        }
    }
}

test "min-p drops candidates far below the top" {
    // Two near-equal leaders, one distant tail: min_p 0.5 keeps exactly the
    // leaders (tail is e^-10 of the max), so the tail is never sampled and the
    // dist reports it filtered out.
    var s = Sampler.init(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.5 }, 77);
    var logits = [_]f32{ 10.0, 10.0, 0.0, -5.0 };
    const d = s.dist(&logits, &.{});
    try std.testing.expectEqual(@as(usize, 2), d.n);
    try std.testing.expectEqual(@as(f32, 0), d.probOf(2));
    for (0..32) |_| {
        var l = [_]f32{ 10.0, 10.0, 0.0, -5.0 };
        try std.testing.expect(s.next(&l, &.{}) < 2);
    }
}

test "min-p threshold is temperature-independent (raw-logit space)" {
    // ln(0.5) ~ -0.693: token 1 sits 0.5 under the max, so min_p = 0.5 keeps
    // it at ANY temperature (the cut ignores temperature by design).
    for ([_]f32{ 0.3, 1.0, 2.0 }) |t| {
        var s = Sampler.init(.{ .temperature = t, .top_k = 0, .top_p = 1.0, .min_p = 0.5 }, 1);
        var logits = [_]f32{ 3.0, 2.5, 1.0 };
        const d = s.dist(&logits, &.{});
        try std.testing.expectEqual(@as(usize, 2), d.n);
    }
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
        .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .min_p = 0.1 },
        .{ .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.3 },
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
            applyPenalties(&l_gpu, &.{}, p);
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

test "SeedSeq is deterministic per base and collision-free across turns" {
    var a = SeedSeq.init(42);
    var b = SeedSeq.init(42);
    var seen: [64]u64 = undefined;
    for (&seen) |*s| {
        s.* = a.next();
        try std.testing.expectEqual(s.*, b.next()); // same base -> same sequence
    }
    for (seen, 0..) |si, i| { // bijective finalizer: no repeats within a session
        for (seen[i + 1 ..]) |sj| try std.testing.expect(si != sj);
    }
    // A different base yields a different first turn (new chat != old chat).
    var c = SeedSeq.init(43);
    try std.testing.expect(c.next() != seen[0]);
}

test "SeedSeq turns produce independent sampling streams" {
    // Two turns of the same session must not replay the same tokens for the
    // same logits (the "identical response on a repeated prompt" bug).
    var seq = SeedSeq.init(7);
    var s1 = Sampler.init(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, seq.next());
    var s2 = Sampler.init(.{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, seq.next());
    const logits_base = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var differs = false;
    for (0..64) |_| {
        var l1 = logits_base;
        var l2 = logits_base;
        if (s1.next(&l1, &.{}) != s2.next(&l2, &.{})) differs = true;
    }
    try std.testing.expect(differs); // p(false positive) = 8^-64
}
