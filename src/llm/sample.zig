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

pub const Sampler = struct {
    params: Params,
    rng: std.Random.DefaultPrng,

    pub fn init(params: Params, seed: u64) Sampler {
        return .{ .params = params, .rng = std.Random.DefaultPrng.init(seed) };
    }

    /// Pick the next token. `logits` is modified in place (repetition
    /// penalty); `recent` is the trailing context window for the penalty.
    pub fn next(self: *Sampler, logits: []f32, recent: []const u32) u32 {
        const p = self.params;
        if (p.repeat_penalty != 1.0) {
            const n = @min(p.repeat_last_n, recent.len);
            for (recent[recent.len - n ..]) |id| {
                const l = &logits[id];
                l.* = if (l.* > 0) l.* / p.repeat_penalty else l.* * p.repeat_penalty;
            }
        }
        if (p.temperature <= 0) return argmax(logits);

        // Top-k candidates (id, logit), highest first.
        const k = @min(if (p.top_k == 0) max_candidates else p.top_k, max_candidates);
        var cand: [max_candidates]Candidate = undefined;
        const cands = topK(logits, cand[0..k]);

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

        // Categorical draw over the kept prefix.
        var kept_mass: f32 = 0;
        for (probs[0..keep]) |prob| kept_mass += prob;
        var r = self.rng.random().float(f32) * kept_mass;
        for (cands[0..keep], probs[0..keep]) |c, prob| {
            r -= prob;
            if (r <= 0) return c.id;
        }
        return cands[keep - 1].id; // float round-off fallthrough
    }
};

const Candidate = struct { id: u32, logit: f32 };

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
