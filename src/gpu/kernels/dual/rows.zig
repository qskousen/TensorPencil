//! Row dual-target kernel bodies: one subgroup per row, lanes strided over the
//! row, subgroups striding over rows (see ../dual.zig). The host passes the ROW
//! count and sizes the launch with `dual_table.rowGroups`. Every lane of a subgroup
//! shares its row, so the early return on the row bound is uniform and the
//! reductions see every lane.

const k = @import("../dual.zig");
const Env = k.Env;

/// Row iteration shared by every body here.
inline fn Rows(e: Env) struct { row: u32, step: u32, lane: u32, lanes: u32 } {
    return .{ .row = k.sgGlobal(e), .step = k.sgTotal(e), .lane = e.sgLane(), .lanes = e.sgSize() };
}

/// Lanes-strided sum over a row with `unroll` independent accumulators, so each
/// lane keeps that many loads in flight: one load per iteration is latency-bound
/// on a single wide row (an LLM decode norm). `f` maps the loaded value.
inline fn stridedSum(e: Env, r: anytype, base: u32, dim: u32, comptime f16_in: bool, comptime f: anytype) f32 {
    const unroll = 8;
    var acc: [unroll]f32 = @splat(0);
    var i = r.lane;
    while (i + (unroll - 1) * r.lanes < dim) : (i += unroll * r.lanes) {
        inline for (0..unroll) |u_| {
            const u: u32 = @intCast(u_);
            const v = if (f16_in) e.h16(.a, base + i + u * r.lanes) else e.ld(.a, base + i + u * r.lanes);
            acc[u] += f(v);
        }
    }
    while (i < dim) : (i += r.lanes) acc[0] += f(if (f16_in) e.h16(.a, base + i) else e.ld(.a, base + i));
    var s: f32 = 0;
    inline for (0..unroll) |u| s += acc[u];
    return s;
}

// The apply loops below are NOT unrolled the way `stridedSum` is: eight strided
// loads followed by eight stores in one loop body, in any form tried, crashes the
// NVIDIA SPIR-V compiler at pipeline creation (ZIG.md). The one shape that pays for
// it, a single wide row, is the LLM decode norm the CUDA wrapper keeps on its hand
// block-per-row kernel.

inline fn sq(v: f32) f32 {
    return v * v;
}
inline fn id(v: f32) f32 {
    return v;
}

/// ggml_l2_norm: a[row] /= max(|a[row]|, f0). u0 = rows, u1 = dim.
pub inline fn l2normRows(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        const scale = 1.0 / @max(@sqrt(k.sgSum(stridedSum(e, r, base, dim, false, sq))), e.f(0));
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.a, base + i, e.ld(.a, base + i) * scale);
    }
}

/// `l2normRows` over rows that come in groups strided by u3 elements, the first u2
/// consecutive dim-wide rows of each group. u0 = rows, u1 = dim, u2 = rows per
/// group, u3 = group stride (elements).
pub inline fn l2normRowsG(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = (row / e.u(2)) * e.u(3) + (row % e.u(2)) * dim;
        const scale = 1.0 / @max(@sqrt(k.sgSum(stridedSum(e, r, base, dim, false, sq))), e.f(0));
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.a, base + i, e.ld(.a, base + i) * scale);
    }
}

inline fn rmsInv(e: Env, r: anytype, base: u32, dim: u32) f32 {
    return 1.0 / @sqrt(k.sgSum(stridedSum(e, r, base, dim, false, sq)) / @as(f32, @floatFromInt(dim)) + e.f(0));
}

/// b[row] = a[row] * rsqrt(mean(a[row]^2) + f0) * c. u0 = rows, u1 = dim.
pub inline fn rmsnorm(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        const inv = rmsInv(e, r, base, dim);
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.b, base + i, e.ld(.a, base + i) * inv * e.ld(.c, i));
    }
}

/// `rmsnorm` over groups: flattened groups are the rows, u1 their width, u2 the
/// groups per original row (the weight is u2*u1 wide).
pub inline fn groupRmsnorm(e: Env) void {
    const r = Rows(e);
    const width = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * width;
        const wbase = (row % e.u(2)) * width;
        const inv = rmsInv(e, r, base, width);
        var i = r.lane;
        while (i < width) : (i += r.lanes) e.st(.b, base + i, e.ld(.a, base + i) * inv * e.ld(.c, wbase + i));
    }
}

/// Fused rmsnorm + AdaLN: b = a * inv * c[premul + col] + c[shift + col], the
/// pre-norm weight already folded into premul. Offsets are u2 and u3, or per row
/// u2 + d[row]*u4 and u3 + d[row]*u4 when u4 != 0 (d holds u32 row labels, a
/// denoise mask). u0 = rows, u1 = dim, f0 = eps.
pub inline fn rmsMod(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        var pre = e.u(2);
        var sh = e.u(3);
        if (e.u(4) != 0) {
            const lab = e.ldW(.d, row) * e.u(4);
            pre += lab;
            sh += lab;
        }
        const inv = rmsInv(e, r, base, dim);
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.b, base + i, e.ld(.a, base + i) * inv * e.ld(.c, pre + i) + e.ld(.c, sh + i));
    }
}

/// `rmsMod` writing f16 pairs scaled by f0, for a GEMM that reads half precision.
/// Rows at or past u5 are real padding (a tile-aligned batch), so they are zeroed
/// rather than normalized: their input is uninitialized and an rms over it would
/// be an inf. b = out words, dim even. u0 = rows, u1 = dim, u2 = premul offset,
/// u3 = shift offset, u5 = real rows, f0 = eps, f1 = scale.
pub inline fn rmsModH16(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        if (row >= e.u(5)) {
            var z = r.lane * 2;
            while (z < dim) : (z += r.lanes * 2) e.stW(.b, (base + z) >> 1, 0);
            continue;
        }
        const inv = rmsInv(e, r, base, dim);
        var i = r.lane * 2;
        while (i < dim) : (i += r.lanes * 2) {
            const lo = (e.ld(.a, base + i) * inv * e.ld(.c, e.u(2) + i) + e.ld(.c, e.u(3) + i)) * e.f(1);
            const hi = (e.ld(.a, base + i + 1) * inv * e.ld(.c, e.u(2) + i + 1) + e.ld(.c, e.u(3) + i + 1)) * e.f(1);
            e.stW(.b, (base + i) >> 1, k.packH16(lo, hi));
        }
    }
}

/// Two-pass mean and inverse deviation of a row (deviation-based variance, since
/// the shifted E[x^2]-E[x]^2 form cancels once the mean is large).
inline fn lnStats(e: Env, r: anytype, base: u32, dim: u32, comptime f16_in: bool) [2]f32 {
    const dimf: f32 = @floatFromInt(dim);
    const mean = k.sgSum(stridedSum(e, r, base, dim, f16_in, id)) / dimf;
    // Deviations need the mean, so this pass is written out.
    const unroll = 8;
    var acc: [unroll]f32 = @splat(0);
    var i = r.lane;
    while (i + (unroll - 1) * r.lanes < dim) : (i += unroll * r.lanes) {
        inline for (0..unroll) |u_| {
            const u: u32 = @intCast(u_);
            const dv = (if (f16_in) e.h16(.a, base + i + u * r.lanes) else e.ld(.a, base + i + u * r.lanes)) - mean;
            acc[u] += dv * dv;
        }
    }
    while (i < dim) : (i += r.lanes) {
        const dv = (if (f16_in) e.h16(.a, base + i) else e.ld(.a, base + i)) - mean;
        acc[0] += dv * dv;
    }
    var v: f32 = 0;
    inline for (0..unroll) |u| v += acc[u];
    return .{ mean, 1.0 / @sqrt(k.sgSum(v) / dimf + e.f(0)) };
}

/// LayerNorm with weight and bias: b = (a - mean) * inv * c + d. u0 = rows,
/// u1 = dim, f0 = eps.
pub inline fn layernorm(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        const st = lnStats(e, r, base, dim, false);
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.b, base + i, (e.ld(.a, base + i) - st[0]) * st[1] * e.ld(.c, i) + e.ld(.d, i));
    }
}

/// `layernorm` over f16 activations (a, b f16; c, d f32), lanes on pairs; dim even.
pub inline fn layernormH16(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        const st = lnStats(e, r, base, dim, true);
        var i = r.lane * 2;
        while (i < dim) : (i += r.lanes * 2) {
            const lo = (e.h16(.a, base + i) - st[0]) * st[1] * e.ld(.c, i) + e.ld(.d, i);
            const hi = (e.h16(.a, base + i + 1) - st[0]) * st[1] * e.ld(.c, i + 1) + e.ld(.d, i + 1);
            e.stW(.b, (base + i) >> 1, k.packH16(lo, hi));
        }
    }
}

/// Fused weightless LayerNorm + AdaLN: b = (a - mean) * inv * c[u2 + col] + c[u3 +
/// col], the (1 + scale) fold already in premul. u0 = rows, u1 = dim, f0 = eps.
pub inline fn lnMod(e: Env) void {
    const r = Rows(e);
    const dim = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        const base = row * dim;
        const st = lnStats(e, r, base, dim, false);
        var i = r.lane;
        while (i < dim) : (i += r.lanes) e.st(.b, base + i, (e.ld(.a, base + i) - st[0]) * st[1] * e.ld(.c, e.u(2) + i) + e.ld(.c, e.u(3) + i));
    }
}

/// GroupNorm pass 1, one subgroup per (group, partial): lanes walk a contiguous
/// slice of the group's positions x channels and write {count, mean, m2} to
/// d[partial*3]. Shifted sums about the group's first element, then one merge
/// across lanes, so the cancellation the plain sum-of-squares form suffers at a
/// large mean stays bounded by the spread. a = x [n][c] (f16 when `h16`).
/// u0 = groups*chunks (partials), u1 = channels, u2 = chunks per group,
/// u3 = per_group, u4 = positions.
inline fn gnStatsImpl(e: Env, comptime h16: bool) void {
    const r = Rows(e);
    const nch = e.u(2);
    const per_group = e.u(3);
    var p = r.row;
    while (p < e.u(0)) : (p += r.step) {
        const g = p / nch;
        const chunk = p % nch;
        const total = e.u(4) * per_group;
        const c0 = g * per_group;
        const shift = if (h16) e.h16(.a, c0) else e.ld(.a, c0);
        var n: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var i = chunk * r.lanes + r.lane;
        while (i < total) : (i += nch * r.lanes) {
            const at = (i / per_group) * e.u(1) + c0 + i % per_group;
            const dv = (if (h16) e.h16(.a, at) else e.ld(.a, at)) - shift;
            n += 1;
            s1 += dv;
            s2 += dv * dv;
        }
        const count = k.sgSum(n);
        const sum1 = k.sgSum(s1);
        const sum2 = k.sgSum(s2);
        if (r.lane == 0) {
            var mean: f32 = 0;
            var m2: f32 = 0;
            if (count > 0) {
                const t = sum1 / count;
                mean = shift + t;
                m2 = @max(sum2 - sum1 * t, 0);
            }
            e.st(.d, p * 3, count);
            e.st(.d, p * 3 + 1, mean);
            e.st(.d, p * 3 + 2, m2);
        }
    }
}

pub inline fn gnStats(e: Env) void {
    gnStatsImpl(e, false);
}

pub inline fn gnStatsH16(e: Env) void {
    gnStatsImpl(e, true);
}

/// Per-row dynamic int8 scale = max|a[row]| / 127, clamped off zero, to b[row].
/// u0 = rows, u1 = cols.
pub inline fn rowmaxI8(e: Env) void {
    const r = Rows(e);
    const cols = e.u(1);
    var row = r.row;
    while (row < e.u(0)) : (row += r.step) {
        var amax: f32 = 0;
        var i = r.lane;
        while (i < cols) : (i += r.lanes) amax = @max(amax, @abs(e.ld(.a, row * cols + i)));
        const m = k.sgMax(amax);
        if (r.lane == 0) e.st(.b, row, @max(m / 127.0, 1e-12));
    }
}
