//! Rotary position embeddings.
//!
//! Krea 2 uses the FLUX convention: multi-axis frequency tables concatenated
//! along the pair dimension, applied to *consecutive* element pairs
//! (x0, x1) -> (x0 c - x1 s, x0 s + x1 c). Angles are computed in f64 and the
//! tables stored as f32, matching `comfy/ldm/flux/math.py::rope`.
//!
//! (Qwen3's half-split "rotate_half" variant will be added with the text
//! encoder in milestone 5.)

const std = @import("std");

/// Per-position cos/sin tables, `half = head_dim / 2` entries per position.
pub const Freqs = struct {
    cos: []f32,
    sin: []f32,
    half: usize,

    pub fn deinit(self: *Freqs, gpa: std.mem.Allocator) void {
        gpa.free(self.cos);
        gpa.free(self.sin);
        self.* = undefined;
    }
};

/// FLUX/Krea-2 multi-axis table. `positions` is [seq, n_axes] row-major
/// (Krea 2: n_axes = 3, text tokens (0,0,0), image tokens (0, row, col));
/// `axes_dim` are per-axis rotary dims summing to head_dim (Krea 2: 32/48/48).
pub fn fluxFreqs(
    gpa: std.mem.Allocator,
    positions: []const f32,
    axes_dim: []const usize,
    theta: f64,
) !Freqs {
    const n_axes = axes_dim.len;
    std.debug.assert(positions.len % n_axes == 0);
    const seq = positions.len / n_axes;
    var half: usize = 0;
    for (axes_dim) |d| {
        std.debug.assert(d % 2 == 0);
        half += d / 2;
    }

    const cos = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(sin);

    for (0..seq) |p| {
        var offset: usize = 0;
        for (axes_dim, 0..) |dim, axis| {
            const pos: f64 = positions[p * n_axes + axis];
            const dh = dim / 2;
            for (0..dh) |i| {
                // omega_i = theta^(-2i/dim); linspace(0, (dim-2)/dim, dim/2).
                const exponent = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim));
                const omega = 1.0 / std.math.pow(f64, theta, exponent);
                const angle = pos * omega;
                cos[p * half + offset + i] = @floatCast(@cos(angle));
                sin[p * half + offset + i] = @floatCast(@sin(angle));
            }
            offset += dh;
        }
    }
    return .{ .cos = cos, .sin = sin, .half = half };
}

/// Apply interleaved-pair rotation in place.
/// `x` is [seq, n_heads * head_dim] row-major; `freqs.half` must equal head_dim/2.
pub fn applyInterleaved(x: []f32, freqs: Freqs, seq: usize, n_heads: usize, head_dim: usize) void {
    const half = head_dim / 2;
    std.debug.assert(freqs.half == half);
    std.debug.assert(x.len == seq * n_heads * head_dim);
    for (0..seq) |p| {
        const cos = freqs.cos[p * half ..][0..half];
        const sin = freqs.sin[p * half ..][0..half];
        for (0..n_heads) |h| {
            const base = (p * n_heads + h) * head_dim;
            for (0..half) |i| {
                const x0 = x[base + 2 * i];
                const x1 = x[base + 2 * i + 1];
                x[base + 2 * i] = x0 * cos[i] - x1 * sin[i];
                x[base + 2 * i + 1] = x0 * sin[i] + x1 * cos[i];
            }
        }
    }
}

/// Half-split ("rotate half") frequency table for Qwen/Llama-style RoPE:
/// freq_i = pos * theta^(-2i/head_dim), i in [0, head_dim/2).
pub fn rotateHalfFreqs(gpa: std.mem.Allocator, seq: usize, head_dim: usize, theta: f64) !Freqs {
    return rotateHalfFreqsScaled(gpa, seq, head_dim, theta, 1.0);
}

/// rotateHalfFreqs with a linear position scale (llama.cpp `freq_scale` /
/// HF "linear" rope scaling): the effective position is `p * freq_scale`
/// (position interpolation). `freq_scale == 1.0` reproduces rotateHalfFreqs.
/// Gemma 3 global layers use freq_scale = 1/8; local (sliding-window) layers
/// use 1.0.
pub fn rotateHalfFreqsScaled(gpa: std.mem.Allocator, seq: usize, head_dim: usize, theta: f64, freq_scale: f64) !Freqs {
    const half = head_dim / 2;
    const cos = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(sin);
    for (0..seq) |p| {
        for (0..half) |i| {
            const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
            const angle = @as(f64, @floatFromInt(p)) * freq_scale * inv;
            cos[p * half + i] = @floatCast(@cos(angle));
            sin[p * half + i] = @floatCast(@sin(angle));
        }
    }
    return .{ .cos = cos, .sin = sin, .half = half };
}

/// rotateHalfFreqsScaled with optional per-dimension frequency factors
/// (llama.cpp `rope_freqs` / freq_factors — "proportional"/long-context RoPE):
/// the effective angle divides the inverse frequency by `freq_factors[i]`,
/// matching ggml's `theta/ff` in ggml_rope_cache_init. `freq_factors` must
/// have head_dim/2 entries (or be null for the plain scaled table). Gemma 4's
/// global layers pass the model's `rope_freqs.weight`; local layers pass null.
pub fn rotateHalfFreqsFactored(
    gpa: std.mem.Allocator,
    seq: usize,
    head_dim: usize,
    theta: f64,
    freq_scale: f64,
    freq_factors: ?[]const f32,
) !Freqs {
    const half = head_dim / 2;
    if (freq_factors) |ff| std.debug.assert(ff.len == half);
    const cos = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(sin);
    for (0..seq) |p| {
        for (0..half) |i| {
            const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
            const ff: f64 = if (freq_factors) |f| f[i] else 1.0;
            const angle = @as(f64, @floatFromInt(p)) * freq_scale * (inv / ff);
            cos[p * half + i] = @floatCast(@cos(angle));
            sin[p * half + i] = @floatCast(@sin(angle));
        }
    }
    return .{ .cos = cos, .sin = sin, .half = half };
}

/// Build parameters for one rotate-half RoPE table. Every LLM stepper's freq
/// table is captured by these four fields — the three `rotateHalfFreqs*`
/// builders above all reduce to `rotateHalfFreqsFactored`. Capturing the spec
/// (rather than a pre-built table) lets a growable context rebuild the table at
/// a larger row count without the caller re-deriving theta/scale/factors.
/// `freq_factors`, when set, must outlive the tables (it points into model
/// weights, e.g. Gemma 4's `rope_freqs`).
pub const RopeSpec = struct {
    head_dim: usize,
    theta: f64,
    freq_scale: f64 = 1.0,
    freq_factors: ?[]const f32 = null,

    pub fn build(self: RopeSpec, gpa: std.mem.Allocator, rows: usize) !Freqs {
        return rotateHalfFreqsFactored(gpa, rows, self.head_dim, self.theta, self.freq_scale, self.freq_factors);
    }
};

/// A fixed set of `N` RoPE tables built from specs and rebuilt together when a
/// growable context grows. `N` is 1 for single-RoPE archs (Qwen) and 2 for
/// dual-RoPE archs (Gemma global + local). Owns the `Freqs`; `deinit` frees
/// them. Single-sources the "rebuild every RoPE table at the new row count"
/// dance that each CPU stepper's `ensureCapacity` open-coded.
pub fn RopeTables(comptime N: usize) type {
    return struct {
        specs: [N]RopeSpec,
        tables: [N]Freqs,

        const Self = @This();

        pub fn init(gpa: std.mem.Allocator, specs: [N]RopeSpec, rows: usize) !Self {
            var tables: [N]Freqs = undefined;
            var built: usize = 0;
            errdefer for (tables[0..built]) |*t| t.deinit(gpa);
            for (&tables, specs) |*t, s| {
                t.* = try s.build(gpa, rows);
                built += 1;
            }
            return .{ .specs = specs, .tables = tables };
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            for (&self.tables) |*t| t.deinit(gpa);
            self.* = undefined;
        }

        /// The `i`-th table (0 = first spec; for Gemma, 0 = global, 1 = local).
        pub fn get(self: *const Self, i: usize) Freqs {
            return self.tables[i];
        }

        /// Rebuild all tables at `rows` rows (context growth). Builds the fresh
        /// tables first and only swaps in on full success, so a mid-build OOM
        /// leaves the existing tables untouched.
        pub fn regrow(self: *Self, gpa: std.mem.Allocator, rows: usize) !void {
            var fresh: [N]Freqs = undefined;
            var built: usize = 0;
            errdefer for (fresh[0..built]) |*t| t.deinit(gpa);
            for (&fresh, self.specs) |*t, s| {
                t.* = try s.build(gpa, rows);
                built += 1;
            }
            for (&self.tables) |*t| t.deinit(gpa);
            self.tables = fresh;
        }
    };
}

/// Apply rotate-half rotation in place: for i < half,
/// (x[i], x[i+half]) -> (x[i] c_i - x[i+half] s_i, x[i+half] c_i + x[i] s_i).
/// `x` is [seq, n_heads * head_dim] row-major.
pub fn applyRotateHalf(x: []f32, freqs: Freqs, seq: usize, n_heads: usize, head_dim: usize) void {
    applyRotateHalfAt(x, freqs, 0, seq, n_heads, head_dim);
}

/// applyRotateHalf for tokens whose absolute positions start at `pos0`
/// (KV-cached decode: x holds only the new tokens). `freqs` must cover
/// pos0 + seq positions.
pub fn applyRotateHalfAt(x: []f32, freqs: Freqs, pos0: usize, seq: usize, n_heads: usize, head_dim: usize) void {
    const half = head_dim / 2;
    std.debug.assert(freqs.half == half);
    std.debug.assert(x.len == seq * n_heads * head_dim);
    std.debug.assert(freqs.cos.len >= (pos0 + seq) * half);
    for (0..seq) |p| {
        const cos = freqs.cos[(pos0 + p) * half ..][0..half];
        const sin = freqs.sin[(pos0 + p) * half ..][0..half];
        for (0..n_heads) |h| {
            const base = (p * n_heads + h) * head_dim;
            for (0..half) |i| {
                const lo = x[base + i];
                const hi = x[base + half + i];
                x[base + i] = lo * cos[i] - hi * sin[i];
                x[base + half + i] = hi * cos[i] + lo * sin[i];
            }
        }
    }
}

/// applyRotateHalfAt over only the first `rot_dim` dims of each head —
/// partial RoPE (Qwen3.5 rotates 64 of 256 head dims; the rest pass
/// through). `freqs` must be built with head_dim = rot_dim; pairs are
/// (i, i + rot_dim/2) within the rotated span.
pub fn applyRotateHalfPartialAt(
    x: []f32,
    freqs: Freqs,
    pos0: usize,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    rot_dim: usize,
) void {
    const half = rot_dim / 2;
    std.debug.assert(rot_dim <= head_dim);
    std.debug.assert(freqs.half == half);
    std.debug.assert(x.len == seq * n_heads * head_dim);
    std.debug.assert(freqs.cos.len >= (pos0 + seq) * half);
    for (0..seq) |p| {
        const cos = freqs.cos[(pos0 + p) * half ..][0..half];
        const sin = freqs.sin[(pos0 + p) * half ..][0..half];
        for (0..n_heads) |h| {
            const base = (p * n_heads + h) * head_dim;
            for (0..half) |i| {
                const lo = x[base + i];
                const hi = x[base + half + i];
                x[base + i] = lo * cos[i] - hi * sin[i];
                x[base + half + i] = hi * cos[i] + lo * sin[i];
            }
        }
    }
}

/// applyRotateHalf with an explicit absolute position per row (speculative
/// tree-verify batches: node positions are depth-based, not consecutive).
/// `freqs` must cover max(positions) + 1.
pub fn applyRotateHalfPos(x: []f32, freqs: Freqs, positions: []const usize, n_heads: usize, head_dim: usize) void {
    const half = head_dim / 2;
    std.debug.assert(freqs.half == half);
    std.debug.assert(x.len == positions.len * n_heads * head_dim);
    for (positions, 0..) |pos, p| {
        std.debug.assert((pos + 1) * half <= freqs.cos.len);
        const cos = freqs.cos[pos * half ..][0..half];
        const sin = freqs.sin[pos * half ..][0..half];
        for (0..n_heads) |h| {
            const base = (p * n_heads + h) * head_dim;
            for (0..half) |i| {
                const lo = x[base + i];
                const hi = x[base + half + i];
                x[base + i] = lo * cos[i] - hi * sin[i];
                x[base + half + i] = hi * cos[i] + lo * sin[i];
            }
        }
    }
}

/// FLUX sinusoidal timestep embedding: out = [cos(t' w_i) .. sin(t' w_i) ..]
/// with t' = 1000 t, w_i = max_period^(-i/half). `out.len` must be even.
pub fn timestepEmbedding(out: []f32, t: f32, max_period: f32) void {
    const half = out.len / 2;
    std.debug.assert(out.len == half * 2);
    const tf: f32 = 1000.0 * t;
    for (0..half) |i| {
        const freq = @exp(-@log(max_period) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = tf * freq;
        out[i] = @cos(arg);
        out[half + i] = @sin(arg);
    }
}

// Reference values generated by tools/gen_op_fixtures.py (torch 2.6):
// dim 8, theta 1000, positions [0, 1, 7], q laid out [seq=3, heads=1, d=8].
const rope_q_in = [_]f32{ -1, -0.899999976, -0.800000012, -0.699999988, -0.600000024, -0.5, -0.399999976, -0.300000012, -0.199999988, -0.100000024, 0, 0.100000024, 0.200000048, 0.299999952, 0.399999976, 0.5, 0.600000024, 0.700000048, 0.799999952, 0.899999976, 1, 1.0999999, 1.20000005, 1.29999995 };
const rope_q_out = [_]f32{ -1, -0.899999976, -0.800000012, -0.699999988, -0.600000024, -0.5, -0.399999976, -0.300000012, -0.0239133313, -0.222324416, -0.0176892225, 0.0984230489, 0.190414816, 0.306173474, 0.397181958, 0.502241433, -0.00754925609, 0.921923578, -0.596392035, 1.04609585, 0.734088182, 1.29271591, 1.14791059, 1.34621739 };

test "flux rope matches torch" {
    const gpa = std.testing.allocator;
    const positions = [_]f32{ 0, 1, 7 };
    var freqs = try fluxFreqs(gpa, &positions, &.{8}, 1000.0);
    defer freqs.deinit(gpa);

    var q: [24]f32 = rope_q_in;
    applyInterleaved(&q, freqs, 3, 1, 8);
    for (rope_q_out, q) |e, a| try std.testing.expectApproxEqAbs(e, a, 2e-6);
}

test "multi-axis table equals concatenated single-axis tables" {
    const gpa = std.testing.allocator;
    // Two positions with distinct per-axis coordinates.
    const positions = [_]f32{ 0, 2, 5, 1, 3, 4 };
    var multi = try fluxFreqs(gpa, &positions, &.{ 4, 2, 2 }, 1000.0);
    defer multi.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), multi.half);

    var ax0 = try fluxFreqs(gpa, &.{ 0, 1 }, &.{4}, 1000.0);
    defer ax0.deinit(gpa);
    var ax1 = try fluxFreqs(gpa, &.{ 2, 3 }, &.{2}, 1000.0);
    defer ax1.deinit(gpa);
    var ax2 = try fluxFreqs(gpa, &.{ 5, 4 }, &.{2}, 1000.0);
    defer ax2.deinit(gpa);

    for (0..2) |p| {
        try std.testing.expectEqualSlices(f32, ax0.cos[p * 2 ..][0..2], multi.cos[p * 4 ..][0..2]);
        try std.testing.expectEqualSlices(f32, ax1.cos[p..][0..1], multi.cos[p * 4 + 2 ..][0..1]);
        try std.testing.expectEqualSlices(f32, ax2.sin[p..][0..1], multi.sin[p * 4 + 3 ..][0..1]);
    }
}

test "position zero is identity" {
    const gpa = std.testing.allocator;
    var freqs = try fluxFreqs(gpa, &.{ 0, 0, 0 }, &.{ 4, 2, 2 }, 1000.0);
    defer freqs.deinit(gpa);
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    applyInterleaved(&x, freqs, 1, 1, 8);
    for ([_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }, x) |e, a| try std.testing.expectEqual(e, a);
}

test "RopeTables matches direct builders and regrows" {
    const gpa = std.testing.allocator;

    // N=2 (Gemma-style global+local): tables equal the direct factored builds.
    const factors = [_]f32{ 1.0, 2.0, 4.0, 8.0 }; // head_dim/2 = 4
    var rt = try RopeTables(2).init(gpa, .{
        .{ .head_dim = 8, .theta = 1e6, .freq_scale = 0.125, .freq_factors = &factors },
        .{ .head_dim = 8, .theta = 1e4 },
    }, 3);
    defer rt.deinit(gpa);

    var g = try rotateHalfFreqsFactored(gpa, 3, 8, 1e6, 0.125, &factors);
    defer g.deinit(gpa);
    var l = try rotateHalfFreqsScaled(gpa, 3, 8, 1e4, 1.0);
    defer l.deinit(gpa);
    try std.testing.expectEqualSlices(f32, g.cos, rt.get(0).cos);
    try std.testing.expectEqualSlices(f32, g.sin, rt.get(0).sin);
    try std.testing.expectEqualSlices(f32, l.cos, rt.get(1).cos);

    // regrow to more rows: prefix (first 3 rows) is preserved bit-exactly, and
    // the table now covers the larger row count.
    try rt.regrow(gpa, 5);
    var g5 = try rotateHalfFreqsFactored(gpa, 5, 8, 1e6, 0.125, &factors);
    defer g5.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5 * 4), rt.get(0).cos.len);
    try std.testing.expectEqualSlices(f32, g5.cos, rt.get(0).cos);
}

// Rotate-half fixture: head_dim 8, theta 5e6, positions [0,1,2], same q as above.
const rh_q_out = [_]f32{ -1, -0.899999976, -0.800000012, -0.699999988, -0.600000024, -0.5, -0.399999976, -0.300000012, -0.2763547, -0.106321417, -0.000178885428, 0.0999952927, -0.0602336824, 0.297818273, 0.399999917, 0.500000954, -1.1589855, 0.652863562, 0.798926294, 0.899975359, 0.129431635, 1.12861371, 1.20071507, 1.300017 };

test "rotate-half rope matches torch" {
    const gpa = std.testing.allocator;
    var freqs = try rotateHalfFreqs(gpa, 3, 8, 5000000.0);
    defer freqs.deinit(gpa);
    var q: [24]f32 = rope_q_in;
    applyRotateHalf(&q, freqs, 3, 1, 8);
    for (rh_q_out, q) |e, a| try std.testing.expectApproxEqAbs(e, a, 2e-6);
}

// Fixture: timestep_embedding([0.25, 1.0], dim=8), generated by tools/gen_op_fixtures.py.
const temb_out = [_]f32{ 0.240988299, 0.991202533, -0.801143587, 0.968912423, -0.970528007, -0.132353634, 0.598472118, 0.247403949, 0.562379062, 0.862314999, -0.839071512, 0.540302336, 0.826879561, -0.506372213, -0.54402113, 0.841470957 };

test "timestep embedding matches torch" {
    var out: [8]f32 = undefined;
    timestepEmbedding(&out, 0.25, 10000.0);
    for (temb_out[0..8], out) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
    timestepEmbedding(&out, 1.0, 10000.0);
    for (temb_out[8..16], out) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
}
