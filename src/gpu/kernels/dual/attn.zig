//! Batched attention against a KV cache and the chunked gated-delta recurrence, one
//! source for both GPU arms. An LLM prefill chunk runs them with `seq_q` / `n` rows; a
//! decode step is the same kernel at one row.

const k = @import("../dual.zig");
const Env = k.Env;
const Slot = k.Slot;

/// How the K and V caches store an element.
pub const KvFmt = enum { f32, f16, q8_0 };

inline fn kvAt(e: Env, comptime s: Slot, comptime fmt: KvFmt, i: u32) f32 {
    return switch (fmt) {
        .f32 => e.ld(s, i),
        .f16 => e.h16(s, i),
        // ggml q8_0: 34-byte blocks of an f16 scale and 32 int8.
        .q8_0 => blk: {
            const base = (i / 32) * 34;
            const q: i32 = @as(i8, @bitCast(@as(u8, @truncate(e.byte(s, base + 2 + i % 32)))));
            break :blk e.f16At(s, base) * @as(f32, @floatFromInt(q));
        },
    };
}

/// Dims of the head a lane keeps in registers per pass; a narrower subgroup takes
/// more passes over its keys.
const per = 8;

/// The keys chunk `i` of query `t` reads: its causal prefix `u5 + t`, cut to the last
/// `window` keys and run on to a bidirectional block's end, split `nsplit` ways.
/// u6 = window | ring << 16 (0: no window, linear rows); f1 = the block end as u32 bits
/// (0: causal). Key j sits at row `j % ring` of a ring cache.
const Span = struct {
    kv0: u32,
    kv1: u32,
    ring: u32,

    inline fn row(s: Span, j: u32) u32 {
        return if (s.ring != 0) j % s.ring else j;
    }
};

inline fn span(e: Env, t: u32, nsplit: u32, i: u32) Span {
    const kv_len = e.u(5) + t;
    const window = e.u(6) & 0xffff;
    const bidir: u32 = @bitCast(e.f(1));
    const start = if (window != 0 and kv_len > window) kv_len - window else 0;
    const end = if (bidir != 0) bidir else kv_len;
    const chunk = (end - start + nsplit - 1) / nsplit;
    const kv0 = start + i * chunk;
    return .{ .kv0 = kv0, .kv1 = @min(kv0 + chunk, end), .ring = e.u(6) >> 16 };
}

/// Flash-decoding split pass for `seq_q` queries over the keys `span` gives query `t`:
/// one subgroup per (query, head, KV chunk), the head dim across its lanes. Writes the
/// unnormalized partial (m, sum, acc[hd]) to d at [row][hd + 2], rows ordered (query,
/// head, chunk), which `attnMerge` reads.
/// a = q [seq_q][heads][hd], b = k, c = v ([row][kv_heads][hd]), d = scratch.
/// u0 = seq_q * heads * nsplit, u1 = heads, u2 = kv_heads, u3 = hd, u4 = nsplit,
/// u5 = kv_len0, u6 and f1 as `span`. f0 = scale.
pub inline fn attnRows(e: Env, comptime fmt: KvFmt) void {
    const heads = e.u(1);
    const kv_heads = e.u(2);
    const hd = e.u(3);
    const nsplit = e.u(4);
    const lane = e.sgLane();
    const lanes = e.sgSize();
    var row = k.sgGlobal(e);
    while (row < e.u(0)) : (row += k.sgTotal(e)) {
        const i = row % nsplit;
        const qh = row / nsplit;
        const h = qh % heads;
        const t = qh / heads;
        const kvh = h / (heads / kv_heads);
        const sp = span(e, t, nsplit, i);
        const qbase = qh * hd;
        const out = row * (hd + 2);
        var d0: u32 = 0;
        while (d0 < hd) : (d0 += per * lanes) {
            var acc: [per]f32 = @splat(0);
            var m: f32 = -3.0e38;
            var sum: f32 = 0;
            var j = sp.kv0;
            while (j < sp.kv1) : (j += 1) {
                const kbase = (sp.row(j) * kv_heads + kvh) * hd;
                var part: f32 = 0;
                var dd = lane;
                while (dd < hd) : (dd += lanes) part += e.ld(.a, qbase + dd) * kvAt(e, .b, fmt, kbase + dd);
                const s = k.sgSum(part) * e.f(0);
                const m2 = @max(m, s);
                const corr = k.exp(m - m2);
                const p = k.exp(s - m2);
                sum = sum * corr + p;
                m = m2;
                inline for (0..per) |c_| {
                    const dim = d0 + @as(u32, @intCast(c_)) * lanes + lane;
                    if (dim < hd) acc[c_] = acc[c_] * corr + p * kvAt(e, .c, fmt, kbase + dim);
                }
            }
            if (d0 == 0 and lane == 0) {
                e.st(.d, out, m);
                e.st(.d, out + 1, sum);
            }
            inline for (0..per) |c_| {
                const dim = d0 + @as(u32, @intCast(c_)) * lanes + lane;
                if (dim < hd) e.st(.d, out + 2 + dim, acc[c_]);
            }
        }
    }
}

/// `attnRows` for the `G` query heads that share a KV head, in one subgroup per (query,
/// KV head, KV chunk): each key and value is read once for the group rather than once
/// per head, and q stays in registers. Bit-identical per head to `attnRows` (same key
/// order, dot order and reduction), so decode, a verify batch and a prefill chunk may
/// take either. Needs `hd <= per * lanes`. u0 = seq_q * kv_heads * nsplit, the rest as
/// `attnRows`; the scratch rows are `attnRows`' too.
pub inline fn attnRowsG(e: Env, comptime fmt: KvFmt, comptime G: u32) void {
    const heads = e.u(1);
    const kv_heads = e.u(2);
    const hd = e.u(3);
    const nsplit = e.u(4);
    const lane = e.sgLane();
    const lanes = e.sgSize();
    var row = k.sgGlobal(e);
    while (row < e.u(0)) : (row += k.sgTotal(e)) {
        const i = row % nsplit;
        const tk = row / nsplit;
        const kvh = tk % kv_heads;
        const t = tk / kv_heads;
        const sp = span(e, t, nsplit, i);
        var q: [G][per]f32 = undefined;
        inline for (0..G) |g| {
            const qbase = (t * heads + kvh * G + @as(u32, @intCast(g))) * hd;
            inline for (0..per) |c_| {
                const dim = @as(u32, @intCast(c_)) * lanes + lane;
                q[g][c_] = if (dim < hd) e.ld(.a, qbase + dim) else 0;
            }
        }
        var acc: [G][per]f32 = @splat(@splat(0));
        var m: [G]f32 = @splat(-3.0e38);
        var sum: [G]f32 = @splat(0);
        var j = sp.kv0;
        while (j < sp.kv1) : (j += 1) {
            const kbase = (sp.row(j) * kv_heads + kvh) * hd;
            var kr: [per]f32 = undefined;
            var vr: [per]f32 = undefined;
            inline for (0..per) |c_| {
                const dim = @as(u32, @intCast(c_)) * lanes + lane;
                kr[c_] = if (dim < hd) kvAt(e, .b, fmt, kbase + dim) else 0;
                vr[c_] = if (dim < hd) kvAt(e, .c, fmt, kbase + dim) else 0;
            }
            inline for (0..G) |g| {
                var part: f32 = 0;
                inline for (0..per) |c_| {
                    if (@as(u32, @intCast(c_)) * lanes + lane < hd) part += q[g][c_] * kr[c_];
                }
                const s = k.sgSum(part) * e.f(0);
                const m2 = @max(m[g], s);
                const corr = k.exp(m[g] - m2);
                const p = k.exp(s - m2);
                sum[g] = sum[g] * corr + p;
                m[g] = m2;
                inline for (0..per) |c_| acc[g][c_] = acc[g][c_] * corr + p * vr[c_];
            }
        }
        inline for (0..G) |g| {
            const out = ((t * heads + kvh * G + @as(u32, @intCast(g))) * nsplit + i) * (hd + 2);
            if (lane == 0) {
                e.st(.d, out, m[g]);
                e.st(.d, out + 1, sum[g]);
            }
            inline for (0..per) |c_| {
                const dim = @as(u32, @intCast(c_)) * lanes + lane;
                if (dim < hd) e.st(.d, out + 2 + dim, acc[g][c_]);
            }
        }
    }
}

/// The `attn_rows_g<G>_<fmt>` entry body, as `.run`.
pub fn AttnRowsG(comptime fmt: KvFmt, comptime G: u32) type {
    return struct {
        pub inline fn run(e: Env) void {
            attnRowsG(e, fmt, G);
        }
    };
}

pub inline fn attnRowsF32(e: Env) void {
    attnRows(e, .f32);
}
pub inline fn attnRowsF16(e: Env) void {
    attnRows(e, .f16);
}
pub inline fn attnRowsQ8(e: Env) void {
    attnRows(e, .q8_0);
}

/// Merge `attnRows`' partials: d[qh * hd + ch] over the nsplit chunks of query-head
/// `qh`. a = scratch. u0 = seq_q * heads * hd, u1 = hd, u2 = nsplit.
pub inline fn attnMerge(e: Env) void {
    const idx = k.elem(e) orelse return;
    const hd = e.u(1);
    const nsplit = e.u(2);
    const qh = idx / hd;
    const ch = idx % hd;
    const stride = hd + 2;
    const base = qh * nsplit * stride;
    // Loads in blocks of 8, so they are in flight together; the sums still run in split
    // order.
    const blk = 8;
    var mx: f32 = -3.0e38;
    var i: u32 = 0;
    while (i + blk <= nsplit) : (i += blk) {
        var m: [blk]f32 = undefined;
        inline for (0..blk) |j| m[j] = e.ld(.a, base + (i + @as(u32, j)) * stride);
        inline for (0..blk) |j| mx = @max(mx, m[j]);
    }
    while (i < nsplit) : (i += 1) mx = @max(mx, e.ld(.a, base + i * stride));
    var sum: f32 = 0;
    var o: f32 = 0;
    i = 0;
    while (i + blk <= nsplit) : (i += blk) {
        var m: [blk]f32 = undefined;
        var sm: [blk]f32 = undefined;
        var acc: [blk]f32 = undefined;
        inline for (0..blk) |j| {
            const r = base + (i + @as(u32, j)) * stride;
            m[j] = e.ld(.a, r);
            sm[j] = e.ld(.a, r + 1);
            acc[j] = e.ld(.a, r + 2 + ch);
        }
        inline for (0..blk) |j| {
            const w = k.exp(m[j] - mx);
            sum += sm[j] * w;
            o += acc[j] * w;
        }
    }
    while (i < nsplit) : (i += 1) {
        const w = k.exp(e.ld(.a, base + i * stride) - mx);
        sum += e.ld(.a, base + i * stride + 1) * w;
        o += e.ld(.a, base + i * stride + 2 + ch) * w;
    }
    e.st(.d, idx, o / sum);
}

/// Gated-delta recurrence over a chunk of `n` tokens with the state in registers: one
/// thread per (v-head, state column j), column j of the `[D][D]` state held for the
/// whole chunk and read and written once. CUDA's `kernels.buildGdnDeltaChunk` in the
/// same arithmetic and order, which is what makes the two exact against each other
/// and a re-prefill exact against decode: ascending i, a multiply then a fused add in
/// pass 1, two fused adds in pass 2, `(v - m) * beta`.
/// a = S [heads][D][D], b = conv_out [n][channels] (`[q | k | v]` per token),
/// c = gates [n][2 * heads] (`[decay | beta]`), d = o [n][heads * D].
/// u0 = heads (one thread per head and column, so not the thread count), u2 = k_heads,
/// u3 = n, u4 = channels, u5 = consecutive v-heads per k-head (0 or 1: tiled, v-head h
/// reads k-head h % k_heads). f0 = the readout scale.
pub inline fn gdnDeltaChunk(e: Env, comptime D: u32) void {
    @setEvalBranchQuota(16 * D);
    const heads = e.u(0);
    const idx = e.gid();
    if (idx >= heads * D) return;
    const kh = e.u(2);
    const channels = e.u(4);
    const h = idx / D;
    const j = idx % D;
    const sbase = h * D * D;
    var s: [D]f32 = undefined;
    inline for (0..D) |i_| s[i_] = e.ld(.a, sbase + @as(u32, @intCast(i_)) * D + j);
    const kidx = (h / @max(e.u(5), 1)) % kh;
    const qoff = kidx * D;
    const koff = kh * D + kidx * D;
    const voff = 2 * kh * D + h * D;
    var t: u32 = 0;
    while (t < e.u(3)) : (t += 1) {
        const tb = t * channels;
        const decay = e.ld(.c, t * 2 * heads + h);
        const beta = e.ld(.c, t * 2 * heads + heads + h);
        var m: f32 = 0;
        inline for (0..D) |i_| {
            const i: u32 = @intCast(i_);
            s[i_] = s[i_] * decay;
            m = @mulAdd(f32, s[i_], e.ld(.b, tb + koff + i), m);
        }
        const dl = (e.ld(.b, tb + voff + j) - m) * beta;
        var o: f32 = 0;
        inline for (0..D) |i_| {
            const i: u32 = @intCast(i_);
            const qi = e.ld(.b, tb + qoff + i) * e.f(0);
            s[i_] = @mulAdd(f32, e.ld(.b, tb + koff + i), dl, s[i_]);
            o = @mulAdd(f32, s[i_], qi, o);
        }
        e.st(.d, t * heads * D + h * D + j, o);
    }
    inline for (0..D) |i_| e.st(.a, sbase + @as(u32, @intCast(i_)) * D + j, s[i_]);
}

pub inline fn gdnDeltaChunkD64(e: Env) void {
    gdnDeltaChunk(e, 64);
}
pub inline fn gdnDeltaChunkD128(e: Env) void {
    gdnDeltaChunk(e, 128);
}
