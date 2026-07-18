//! Softmax attention with grouped-query support.
//!
//! Streaming formulation: scores for one (head, query) pair at a time, so
//! memory is O(seq_kv) per task instead of O(seq^2). Activations are
//! [seq, n_heads * head_dim] row-major, matching the rest of the engine.

const std = @import("std");
const vmath = @import("vmath.zig");

pub const Params = struct {
    seq_q: usize,
    seq_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Causal masking (Qwen text encoder / LLM decode). Queries are the LAST
    /// seq_q positions of the kv sequence: query i attends to keys
    /// [0, seq_kv - seq_q + i]. seq_q == seq_kv is the classic full-sequence
    /// case; seq_q == 1 with a longer seq_kv is KV-cached decode.
    causal: bool = false,
    /// Per-key boolean mask, true = attend (Krea 2 text refiner). Length seq_kv.
    key_mask: ?[]const bool = null,
    /// Score scale; defaults to 1/sqrt(head_dim).
    scale: ?f32 = null,
    /// Sliding-window attention (Gemma 3 local layers): 0 = disabled (full
    /// causal). When > 0 (and causal), query at absolute position p attends
    /// only keys in [p - window + 1, p]. Requires causal.
    window: usize = 0,
    /// Ring-buffer KV storage (gemma3/gemma4 LOCAL layers, TODO lever 1): 0 =
    /// linear (row = absolute position). When > 0, key/value for absolute
    /// position j lives at row `j % ring`, so k/v hold `ring` rows instead of
    /// `seq_kv`. Scores/positions stay absolute; only the K/V data row wraps.
    ring: usize = 0,
};

pub const Error = error{OutOfMemory} || std.Io.Cancelable;

/// out[seq_q, n_heads*head_dim] = softmax(q k^T * scale) v, per head.
/// KV head h_kv = h / (n_heads / n_kv_heads) (repeat_interleave semantics).
pub fn attention(
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    p: Params,
) Error!void {
    const hd = p.head_dim;
    // Ring layers store `ring` rows instead of the full seq_kv (row = pos%ring).
    const kv_rows = if (p.ring != 0) p.ring else p.seq_kv;
    std.debug.assert(q.len == p.seq_q * p.n_heads * hd);
    std.debug.assert(k.len == kv_rows * p.n_kv_heads * hd);
    std.debug.assert(v.len == kv_rows * p.n_kv_heads * hd);
    std.debug.assert(out.len == q.len);
    std.debug.assert(p.n_heads % p.n_kv_heads == 0);
    if (p.causal) std.debug.assert(p.seq_q <= p.seq_kv);
    if (p.key_mask) |mask| std.debug.assert(mask.len == p.seq_kv);

    // Split each head's queries into chunks so even single-head attention
    // (the VAE mid-block: 1 head over all spatial positions) parallelizes.
    const n_threads = std.Thread.getCpuCount() catch 1;
    const small = p.seq_q * p.seq_kv * hd < (1 << 16);
    const want_tasks: usize = if (small or n_threads == 1) 1 else 2 * n_threads;
    const q_chunks = @max(1, std.math.divCeil(usize, want_tasks, p.n_heads) catch unreachable);
    const q_chunk_len = std.math.divCeil(usize, p.seq_q, q_chunks) catch unreachable;
    const n_tasks = p.n_heads * q_chunks;

    const scratch = try gpa.alloc(f32, n_tasks * p.seq_kv);
    defer gpa.free(scratch);

    if (n_tasks == 1) {
        headTask(out, q, k, v, p, 0, 0, p.seq_q, scratch);
        return;
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    var task: usize = 0;
    for (0..p.n_heads) |h| {
        var q_start: usize = 0;
        while (q_start < p.seq_q) : (q_start += q_chunk_len) {
            const q_end = @min(q_start + q_chunk_len, p.seq_q);
            group.async(io, headTask, .{ out, q, k, v, p, h, q_start, q_end, scratch[task * p.seq_kv ..][0..p.seq_kv] });
            task += 1;
        }
    }
    try group.await(io);
}

fn headTask(
    out: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    p: Params,
    h: usize,
    q_start: usize,
    q_end: usize,
    scores: []f32,
) void {
    const hd = p.head_dim;
    const rep = p.n_heads / p.n_kv_heads;
    const h_kv = h / rep;
    const q_stride = p.n_heads * hd;
    const kv_stride = p.n_kv_heads * hd;
    const scale = p.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    for (q_start..q_end) |i| {
        const qrow = q[i * q_stride + h * hd ..][0..hd];
        const kv_end = if (p.causal) p.seq_kv - p.seq_q + i + 1 else p.seq_kv;
        // Sliding window: drop keys older than `window` positions back.
        const kv_start = if (p.window != 0 and kv_end > p.window) kv_end - p.window else 0;

        var max_score = -std.math.inf(f32);
        for (kv_start..kv_end) |j| {
            if (p.key_mask) |mask| if (!mask[j]) {
                scores[j] = -std.math.inf(f32);
                continue;
            };
            const jr = if (p.ring != 0) j % p.ring else j;
            const krow = k[jr * kv_stride + h_kv * hd ..][0..hd];
            const s = dot(qrow, krow) * scale;
            scores[j] = s;
            max_score = @max(max_score, s);
        }

        const orow = out[i * q_stride + h * hd ..][0..hd];
        @memset(orow, 0);
        if (max_score == -std.math.inf(f32)) continue; // fully masked row

        var denom: f32 = 0;
        {
            const vlen = vmath.vlen;
            const mx: vmath.Vec = @splat(max_score);
            var acc: vmath.Vec = @splat(0);
            var j: usize = kv_start;
            while (j + vlen <= kv_end) : (j += vlen) {
                const e = vmath.expVec(@as(vmath.Vec, scores[j..][0..vlen].*) - mx);
                scores[j..][0..vlen].* = e;
                acc += e;
            }
            denom = @reduce(.Add, acc);
            while (j < kv_end) : (j += 1) {
                const e = @exp(scores[j] - max_score);
                scores[j] = e;
                denom += e;
            }
        }
        const inv = 1.0 / denom;
        for (kv_start..kv_end) |j| {
            const weight = scores[j] * inv;
            if (weight == 0) continue;
            const jr = if (p.ring != 0) j % p.ring else j;
            const vrow = v[jr * kv_stride + h_kv * hd ..][0..hd];
            for (orow, vrow) |*o, vv| o.* += weight * vv;
        }
    }
}

pub const TreeParams = struct {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Score scale; defaults to 1/sqrt(head_dim).
    scale: ?f32 = null,
};

/// Tree-verify attention (speculative tree drafting, LLM_PLAN.md M8):
/// `parents.len` query nodes, node i attending the full committed prefix
/// plus its own ancestor chain among the batch rows (parents[i] < i;
/// parents[0] == 0 is the root, which attends only the prefix and itself).
/// q/out are [n][n_heads*hd]; k/v_prefix are the cached rows
/// [prefix_len][n_kv_heads*hd]; k/v_tree are the batch rows [n][kv_dim].
/// Serial per (node, head) — verify batches are tiny and this is the
/// reference implementation for the GPU kernels, not a production CPU path.
pub fn attentionTree(
    gpa: std.mem.Allocator,
    out: []f32,
    q: []const f32,
    k_prefix: []const f32,
    v_prefix: []const f32,
    k_tree: []const f32,
    v_tree: []const f32,
    parents: []const u32,
    p: TreeParams,
) error{OutOfMemory}!void {
    const hd = p.head_dim;
    const n = parents.len;
    const q_stride = p.n_heads * hd;
    const kv_stride = p.n_kv_heads * hd;
    const prefix_len = k_prefix.len / kv_stride;
    std.debug.assert(q.len == n * q_stride and out.len == q.len);
    std.debug.assert(k_prefix.len == prefix_len * kv_stride and v_prefix.len == k_prefix.len);
    std.debug.assert(k_tree.len == n * kv_stride and v_tree.len == k_tree.len);
    std.debug.assert(p.n_heads % p.n_kv_heads == 0);
    const rep = p.n_heads / p.n_kv_heads;
    const scale = p.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    const scores = try gpa.alloc(f32, prefix_len + n);
    defer gpa.free(scores);
    var anc: [64]u32 = undefined; // ancestor chain, root first (n <= spec.max_tree_nodes)
    std.debug.assert(n <= anc.len);

    for (0..n) |i| {
        // Node i's ancestor chain (including itself), depth order.
        var depth: usize = 0;
        {
            var j: u32 = @intCast(i);
            while (true) {
                anc[depth] = j;
                depth += 1;
                if (j == 0) break;
                std.debug.assert(parents[j] < j);
                j = parents[j];
            }
            std.mem.reverse(u32, anc[0..depth]);
        }
        const kv_len = prefix_len + depth;

        for (0..p.n_heads) |h| {
            const h_kv = h / rep;
            const qrow = q[i * q_stride + h * hd ..][0..hd];

            var max_score = -std.math.inf(f32);
            for (0..kv_len) |j| {
                const krow = if (j < prefix_len)
                    k_prefix[j * kv_stride + h_kv * hd ..][0..hd]
                else
                    k_tree[anc[j - prefix_len] * kv_stride + h_kv * hd ..][0..hd];
                const s = dot(qrow, krow) * scale;
                scores[j] = s;
                max_score = @max(max_score, s);
            }

            var denom: f32 = 0;
            for (scores[0..kv_len]) |*s| {
                s.* = @exp(s.* - max_score);
                denom += s.*;
            }
            const inv = 1.0 / denom;

            const orow = out[i * q_stride + h * hd ..][0..hd];
            @memset(orow, 0);
            for (0..kv_len) |j| {
                const weight = scores[j] * inv;
                const vrow = if (j < prefix_len)
                    v_prefix[j * kv_stride + h_kv * hd ..][0..hd]
                else
                    v_tree[anc[j - prefix_len] * kv_stride + h_kv * hd ..][0..hd];
                for (orow, vrow) |*o, vv| o.* += weight * vv;
            }
        }
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    const vlen = comptime std.simd.suggestVectorLength(f32) orelse 8;
    const Vec = @Vector(vlen, f32);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + vlen <= a.len) : (i += vlen) {
        const av: Vec = a[i..][0..vlen].*;
        const bv: Vec = b[i..][0..vlen].*;
        acc += av * bv;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

// Reference values generated by tools/gen_op_fixtures.py (torch sdpa):
// seq 3, heads 2, kv_heads 1, head_dim 4, layout [seq, heads*hd].
const attn_q = [_]f32{ -1.12583983, -1.1523602, -0.250578582, -0.433878809, 0.166455343, 0.87438184, -0.143473849, -0.111609332, 0.848710358, 0.692009151, -0.31601277, -2.11521935, 0.931826591, 1.25900924, 2.00498056, 0.0537369028, 0.468096405, -0.157712445, 1.44366014, 0.266049415, 0.618056655, -0.412802219, -0.841064811, -2.31604195 };
const attn_k = [_]f32{ 0.370393544, 1.45650256, 0.939809918, 0.774848819, 0.191869423, 1.26379478, -1.29043508, -0.791102707, -0.0208794735, -0.718480051, 0.518636763, -1.31252193 };
const attn_v = [_]f32{ 0.191995069, 0.542771339, -2.21877933, 0.258984536, -1.02970219, -0.50075841, 0.273367733, -0.918099999, -0.0404210873, 0.288116813, -0.00753730815, -0.914495468 };
const attn_out = [_]f32{ -0.233936831, 0.140385717, -0.156303599, -0.80384618, -0.3932935, 0.0334626585, -0.734680712, -0.463540882, -0.583657682, -0.13896063, -0.0424246602, -0.813088059, 0.0828519836, 0.443475634, -1.80830443, 0.0511539057, 0.00270056119, 0.361810803, -1.23623109, -0.249404013, -0.469629675, -0.0521637946, 0.05305776, -0.882563472 };
const attn_causal_out = [_]f32{ 0.191995069, 0.542771339, -2.21877933, 0.258984536, 0.191995069, 0.542771339, -2.21877933, 0.258984536, -0.866005778, -0.360934854, -0.0605574213, -0.760381341, 0.0977972746, 0.46231094, -2.02662444, 0.168226555, 0.00270056119, 0.361810803, -1.23623109, -0.249404013, -0.469629675, -0.0521637946, 0.05305776, -0.882563472 };

test "gqa attention matches torch sdpa" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var out: [24]f32 = undefined;
    try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, .{
        .seq_q = 3,
        .seq_kv = 3,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 4,
    });
    for (attn_out, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
}

test "causal attention matches torch sdpa" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var out: [24]f32 = undefined;
    try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, .{
        .seq_q = 3,
        .seq_kv = 3,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 4,
        .causal = true,
    });
    for (attn_causal_out, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
}

test "sliding-window attention" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const p: Params = .{ .seq_q = 3, .seq_kv = 3, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .causal = true };

    // window >= seq_kv is identical to full causal.
    {
        var out: [24]f32 = undefined;
        var pw = p;
        pw.window = 3;
        try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, pw);
        for (attn_causal_out, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
    }
    // window == 1: query i attends only key i, so each output row is that
    // position's V (softmax over a single key), for both GQA heads.
    {
        var out: [24]f32 = undefined;
        var pw = p;
        pw.window = 1;
        try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, pw);
        for (0..3) |i| {
            for (0..2) |h| {
                for (0..4) |d| {
                    try std.testing.expectApproxEqAbs(attn_v[i * 4 + d], out[i * 8 + h * 4 + d], 3e-6);
                }
            }
        }
    }
}

test "cached decode attention matches the full causal rows" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // seq_q < seq_kv, causal: query i is absolute position seq_kv - seq_q + i,
    // so outputs must equal the corresponding rows of the full causal run.
    inline for (.{ 1, 2 }) |seq_q| {
        var out: [seq_q * 8]f32 = undefined;
        try attention(io, gpa, &out, attn_q[(3 - seq_q) * 8 ..], &attn_k, &attn_v, .{
            .seq_q = seq_q,
            .seq_kv = 3,
            .n_heads = 2,
            .n_kv_heads = 1,
            .head_dim = 4,
            .causal = true,
        });
        for (attn_causal_out[(3 - seq_q) * 8 ..], out) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
    }
}

test "key mask restricts attention" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // With only key 0 visible, every output row equals v[0] (per kv head).
    var out: [24]f32 = undefined;
    const mask = [_]bool{ true, false, false };
    try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, .{
        .seq_q = 3,
        .seq_kv = 3,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 4,
        .key_mask = &mask,
    });
    for (0..3) |i| {
        for (0..2) |h| {
            for (0..4) |d| {
                try std.testing.expectApproxEqAbs(attn_v[d], out[i * 8 + h * 4 + d], 1e-6);
            }
        }
    }
}

test "tree attention on a chain matches causal attention" {
    const gpa = std.testing.allocator;
    // Prefix = position 0; tree = a 2-node chain at positions 1, 2. Every
    // node's output must equal the corresponding full-causal row.
    const parents = [_]u32{ 0, 0 }; // node 1 is the root's child
    var out: [16]f32 = undefined;
    try attentionTree(
        gpa,
        &out,
        attn_q[8..24], // queries for positions 1, 2
        attn_k[0..4], // prefix K (position 0)
        attn_v[0..4],
        attn_k[4..12], // batch K (positions 1, 2)
        attn_v[4..12],
        &parents,
        .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 4 },
    );
    for (attn_causal_out[8..24], out) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
}

test "tree attention: a node never sees its sibling branch" {
    const gpa = std.testing.allocator;
    // Nodes: A (root, position 1), B and C both children of A (position 2).
    // B carries garbage K/V; C's output must still equal the full-causal row
    // for the chain prefix->A->C — proof it attends only its own ancestors.
    const junk = [_]f32{ 100, -100, 100, -100 };
    const q = attn_q[8..16] ++ junk ++ junk ++ attn_q[16..24];
    const k_tree = attn_k[4..8] ++ junk ++ attn_k[8..12];
    const v_tree = attn_v[4..8] ++ junk ++ attn_v[8..12];
    const parents = [_]u32{ 0, 0, 0 };
    var out: [24]f32 = undefined;
    try attentionTree(
        gpa,
        &out,
        q,
        attn_k[0..4],
        attn_v[0..4],
        k_tree,
        v_tree,
        &parents,
        .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 4 },
    );
    // Node A (row 0) = causal row 1; node C (row 2) = causal row 2.
    for (attn_causal_out[8..16], out[0..8]) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
    for (attn_causal_out[16..24], out[16..24]) |e, a| try std.testing.expectApproxEqAbs(e, a, 3e-6);
}

test "fully masked rows produce zeros" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var out: [24]f32 = undefined;
    const mask = [_]bool{ false, false, false };
    try attention(io, gpa, &out, &attn_q, &attn_k, &attn_v, .{
        .seq_q = 3,
        .seq_kv = 3,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 4,
        .key_mask = &mask,
    });
    for (out) |o| try std.testing.expectEqual(@as(f32, 0), o);
}
