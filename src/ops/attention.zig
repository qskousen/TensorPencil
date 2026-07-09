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
    std.debug.assert(q.len == p.seq_q * p.n_heads * hd);
    std.debug.assert(k.len == p.seq_kv * p.n_kv_heads * hd);
    std.debug.assert(v.len == p.seq_kv * p.n_kv_heads * hd);
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

        var max_score = -std.math.inf(f32);
        for (0..kv_end) |j| {
            if (p.key_mask) |mask| if (!mask[j]) {
                scores[j] = -std.math.inf(f32);
                continue;
            };
            const krow = k[j * kv_stride + h_kv * hd ..][0..hd];
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
            var j: usize = 0;
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
        for (0..kv_end) |j| {
            const weight = scores[j] * inv;
            if (weight == 0) continue;
            const vrow = v[j * kv_stride + h_kv * hd ..][0..hd];
            for (orow, vrow) |*o, vv| o.* += weight * vv;
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
