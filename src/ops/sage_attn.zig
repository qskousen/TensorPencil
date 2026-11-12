//! ComfyUI's int8 attention (comfy-kitchen's "sage" SDPA), which a checkpoint asks for
//! with a `comfy_attention.config` tensor holding `{"attention": "comfy_kitchen_int8"}`.
//! A CPU reference of what its CUDA kernels compute at head dim 128 with more than 1024
//! keys, the path a 1024² DiT takes; `supported` says which shapes that is.
//!
//! - K has a "representative key" subtracted when that tightens its range (`anchor`),
//!   which softmax cannot see. Q and K are then rotated by the same signed H128 in a
//!   fixed fp32 order, so the rotation's rounding is part of the answer.
//! - One Q scale covers rows {32a + o + 8j}, one K scale the 32 keys of a 128-key tile
//!   one MMA lane owns ((n % 8) / 2), one V scale a channel over the whole sequence.
//! - Probabilities are taken against each key TILE's maximum (scaled so it lands near
//!   255) and rounded to u8; P.V is an exact integer product folded in with one fma
//!   per tile. The last tile differs: its denominator sums the unrounded fp32
//!   probabilities, and its per-lane maximum is floored at -50000 in score units.
//!
//! The GPU computes exp2 and reciprocals with approximate instructions, so this matches
//! comfy-kitchen to a u8 flip near a rounding boundary, not to the bit. Pinned against
//! comfy-kitchen's own kernels by `tools/gen_sage_attn_fixture.py`.

const std = @import("std");
const dtypes = @import("tp_core").dtype;
const WeightStore = @import("tp_core").weights.WeightStore;

pub const head_dim = 128;
pub const key_tile = 128;

/// The tile maximum is offset so the largest probability is 2^off, about 255.
const p_offset: f32 = 7.9943534;
const log2e: f32 = 1.44269504088896340736;

/// Whether this reference (and the kernels that match it) covers the shape. Fewer
/// keys take comfy-kitchen's 64-key tiles, and at 256 or fewer another rotation.
pub fn supported(hd: usize, lk: usize) bool {
    return hd == head_dim and lk > 1024;
}

/// `auto` runs int8 attention where the checkpoint asks for it, as ComfyUI does;
/// `off` is the control row, ordinary attention everywhere.
pub const Mode = enum { auto, on, off };

/// Set once by the CLI (`--int8-attention`) before anything runs.
pub var mode: Mode = .auto;

/// Whether one attention runs int8: what its checkpoint asked for under `mode`,
/// at a shape this path covers.
pub fn use(requested_by_file: bool, hd: usize, lk: usize) bool {
    const want = switch (mode) {
        .auto => requested_by_file,
        .on => true,
        .off => false,
    };
    return want and supported(hd, lk);
}

/// Whether the `<prefix>.comfy_attention.config` tensor asks for this attention,
/// the way ComfyUI's `ComfyAttention` reads it: a JSON object, or a list of them,
/// whose `attention` is `comfy_kitchen_int8`.
pub fn requested(store: WeightStore, prefix: []const u8) bool {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.comfy_attention.config", .{prefix}) catch return false;
    const t = store.get(name) orelse return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), t.bytes, .{}) catch return false;
    const items: []const std.json.Value = switch (v) {
        .array => |a| a.items,
        .object => &.{v},
        else => return false,
    };
    for (items) |it| {
        if (it != .object) continue;
        const m = it.object.get("attention") orelse continue;
        if (m == .string and std.mem.eql(u8, m.string, "comfy_kitchen_int8")) return true;
    }
    return false;
}

/// The rotation's sign vector: entry i is negated when bit i of these is clear.
const sign_words = [4]u32{ 0x1035997b, 0x8087f5ee, 0xee2e4e1a, 0x71132418 };

/// The signed H128 in comfy-kitchen's fp32 order: an H4 on each lane's four channels,
/// butterflies across the 32 lanes, one final scale.
pub fn rotate(x: *[head_dim]f32) void {
    for (x, 0..) |*v, i| {
        if ((sign_words[i / 32] >> @intCast(i % 32)) & 1 == 0) v.* = -v.*;
    }
    for (0..32) |l| {
        const g = x[l * 4 ..][0..4];
        const a0 = g[0] + g[1];
        const a1 = g[0] - g[1];
        const a2 = g[2] + g[3];
        const a3 = g[2] - g[3];
        g.* = .{ (a0 + a2) * 0.5, (a1 + a3) * 0.5, (a0 - a2) * 0.5, (a1 - a3) * 0.5 };
    }
    var bit: usize = 1;
    while (bit < 32) : (bit <<= 1) {
        for (0..32) |l| {
            if (l & bit != 0) continue;
            for (0..4) |c| {
                const lo = x[l * 4 + c];
                const hi = x[(l | bit) * 4 + c];
                x[l * 4 + c] = lo + hi;
                x[(l | bit) * 4 + c] = lo - hi;
            }
        }
    }
    for (x) |*v| v.* *= 0.1767766952966369;
}

/// `cvt.rni.sat.s8.f32`.
fn toS8(v: f32) i8 {
    return @intFromFloat(std.math.clamp(@round(v) - roundTieAdjust(v), -128, 127));
}

/// `@round` rounds ties away from zero; the GPU rounds them to even.
fn roundTieAdjust(v: f32) f32 {
    const r = @round(v);
    if (@abs(r - v) != 0.5) return 0;
    return if (@mod(r, 2) != 0) std.math.sign(v) else 0;
}

/// The Q/K scale: `amax / 127 + 1e-7`, the 1e-7 after the division.
fn qkScale(amax: f32) f32 {
    return amax / 127.0 + 1e-7;
}

pub fn qScaleLen(lq: usize) usize {
    return (lq + 127) / 128 * 32;
}

pub fn kRows(lk: usize) usize {
    return (lk + key_tile - 1) / key_tile * key_tile;
}

pub fn kScaleLen(lk: usize) usize {
    return (lk + key_tile - 1) / key_tile * 4;
}

fn qScaleIndex(r: usize) usize {
    return (r / 32) * 8 + r % 8;
}

fn kScaleIndex(n: usize) usize {
    return (n / key_tile) * 4 + (n % 8) / 2;
}

/// Sum of 128 per-channel values the way the anchor kernel reduces them: a
/// shuffle-down tree in each 32-lane warp, then the four warp sums in order.
fn blockSum(vals: *const [head_dim]f32) f32 {
    var total: f32 = 0;
    for (0..4) |w| {
        var v = vals[w * 32 ..][0..32].*;
        var off: usize = 16;
        while (off > 0) : (off /= 2) {
            for (0..off) |i| v[i] += v[i + off];
        }
        total += v[0];
    }
    return total;
}

fn blockMax(vals: *const [head_dim]f32) f32 {
    var m: f32 = 0;
    for (vals) |v| m = @max(m, v);
    return m;
}

/// The key row subtracted from every key of one head, or null. From nine sampled
/// keys, the one nearest their mean, kept only when it lowers their energy without
/// raising any channel's range by more than 1/8.
pub fn anchor(k: []const f32, lk: usize, stride: usize) ?usize {
    const ns = 9;
    var xs: [ns][head_dim]f32 = undefined;
    for (0..ns) |s| @memcpy(&xs[s], k[(s * (lk - 1) / 8) * stride ..][0..head_dim]);

    var energy: [head_dim]f32 = undefined;
    var amax: [head_dim]f32 = undefined;
    var cand: [ns][head_dim]f32 = undefined;
    for (0..head_dim) |c| {
        var e: f32 = 0;
        var m: f32 = 0;
        var sum: f32 = 0;
        for (0..ns) |s| {
            e = @mulAdd(f32, xs[s][c], xs[s][c], e);
            m = @max(m, @abs(xs[s][c]));
            sum += xs[s][c];
        }
        energy[c] = e;
        amax[c] = m;
        for (0..ns) |a| {
            const dist = 9 * xs[a][c] - sum;
            cand[a][c] = dist * dist;
        }
    }
    var best: usize = 0;
    var best_cand = std.math.floatMax(f32);
    for (0..ns) |a| {
        const v = blockSum(&cand[a]);
        if (v < best_cand) {
            best_cand = v;
            best = a;
        }
    }
    var res_energy: [head_dim]f32 = undefined;
    var res_max: [head_dim]f32 = undefined;
    for (0..head_dim) |c| {
        var e: f32 = 0;
        var m: f32 = 0;
        for (0..ns) |s| {
            const d = xs[s][c] - xs[best][c];
            e = @mulAdd(f32, d, d, e);
            m = @max(m, @abs(d));
        }
        res_energy[c] = e;
        res_max[c] = m;
    }
    const keep = blockSum(&res_energy) < blockSum(&energy) and blockMax(&res_max) <= blockMax(&amax) * 1.125;
    return if (keep) best * (lk - 1) / 8 else null;
}

/// The quantized operands of one attention, head-major.
pub const Quant = struct {
    heads: usize,
    lq: usize,
    lk: usize,
    /// [heads][lq][128], rotated
    q: []i8,
    /// [heads][qScaleLen(lq)]
    q_scale: []f32,
    /// [heads][lk][128], anchor-shifted and rotated
    k: []i8,
    /// [heads][kScaleLen(lk)]
    k_scale: []f32,
    /// [heads][128][kRows(lk)], channel-major, zero past lk
    v: []i8,
    /// [heads][128]
    v_scale: []f32,
    /// [heads], -1 for none
    anchor: []i32,

    pub fn deinit(self: *Quant, gpa: std.mem.Allocator) void {
        inline for (.{ "q", "q_scale", "k", "k_scale", "v", "v_scale", "anchor" }) |f| gpa.free(@field(self, f));
    }
};

/// Quantize token-major Q [lq][heads*128], K and V [lk][heads*128].
pub fn quantize(gpa: std.mem.Allocator, q: []const f32, k: []const f32, v: []const f32, lq: usize, lk: usize, heads: usize) !Quant {
    std.debug.assert(supported(head_dim, lk));
    const stride = heads * head_dim;
    var out: Quant = .{
        .heads = heads,
        .lq = lq,
        .lk = lk,
        .q = try gpa.alloc(i8, heads * lq * head_dim),
        .q_scale = try gpa.alloc(f32, heads * qScaleLen(lq)),
        .k = try gpa.alloc(i8, heads * lk * head_dim),
        .k_scale = try gpa.alloc(f32, heads * kScaleLen(lk)),
        .v = try gpa.alloc(i8, heads * head_dim * kRows(lk)),
        .v_scale = try gpa.alloc(f32, heads * head_dim),
        .anchor = try gpa.alloc(i32, heads),
    };
    errdefer out.deinit(gpa);
    const rot = try gpa.alloc(f32, @max(lq, lk) * head_dim);
    defer gpa.free(rot);

    for (0..heads) |h| {
        // Q: rotate every row, then one scale per row group
        for (0..lq) |r| {
            const x = rot[r * head_dim ..][0..head_dim];
            @memcpy(x, q[r * stride + h * head_dim ..][0..head_dim]);
            rotate(x);
        }
        try quantRows(rot, out.q[h * lq * head_dim ..][0 .. lq * head_dim], out.q_scale[h * qScaleLen(lq) ..][0..qScaleLen(lq)], lq, qScaleIndex);

        // K: subtract the anchor, rotate, one scale per lane group of each tile
        const a = anchor(k[h * head_dim ..], lk, stride);
        out.anchor[h] = if (a) |i| @intCast(i) else -1;
        for (0..lk) |n| {
            const x = rot[n * head_dim ..][0..head_dim];
            @memcpy(x, k[n * stride + h * head_dim ..][0..head_dim]);
            if (a) |i| {
                for (x, k[i * stride + h * head_dim ..][0..head_dim]) |*xv, b| xv.* -= b;
            }
            rotate(x);
        }
        try quantRows(rot, out.k[h * lk * head_dim ..][0 .. lk * head_dim], out.k_scale[h * kScaleLen(lk) ..][0..kScaleLen(lk)], lk, kScaleIndex);

        // V: one scale per channel over the whole sequence, no rotation
        for (0..head_dim) |c| {
            var amax: f32 = 0;
            for (0..lk) |n| amax = @max(amax, @abs(v[n * stride + h * head_dim + c]));
            const sc = @max(amax * @as(f32, 1.0 / 127.0), 1e-12);
            const inv = 1 / sc;
            out.v_scale[h * head_dim + c] = sc;
            const row = out.v[(h * head_dim + c) * kRows(lk) ..][0..kRows(lk)];
            for (row, 0..) |*o, n| o.* = if (n < lk) toS8(v[n * stride + h * head_dim + c] * inv) else 0;
        }
    }
    return out;
}

/// Quantize `rows` rotated rows, each scale covering the rows `index` maps to it.
fn quantRows(rot: []const f32, codes: []i8, scales: []f32, rows: usize, comptime index: fn (usize) usize) !void {
    @memset(scales, 0);
    for (0..rows) |r| {
        var m: f32 = 0;
        for (rot[r * head_dim ..][0..head_dim]) |x| m = @max(m, @abs(x));
        scales[index(r)] = @max(scales[index(r)], m);
    }
    for (scales) |*s| s.* = qkScale(s.*);
    for (0..rows) |r| {
        const inv = 1 / scales[index(r)];
        for (codes[r * head_dim ..][0..head_dim], rot[r * head_dim ..][0..head_dim]) |*o, x| o.* = toS8(x * inv);
    }
}

/// u8 probability rounding: at most 255, half to even.
fn toU8(p: f32) u8 {
    const c = std.math.clamp(p, 0, 255);
    return @intFromFloat(@round(c) - roundTieAdjust(c));
}

/// Attention over quantized operands into token-major `out` [lq][heads*128], each
/// value rounded to bf16 as comfy-kitchen stores it. One task per head.
pub fn attend(io: std.Io, qt: Quant, out: []f32, scale: f32) !void {
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (0..qt.heads) |h| group.async(io, attendHead, .{ qt, out, scale, h });
    try group.await(io);
}

const V = @Vector(head_dim, i32);

fn attendHead(qt: Quant, out: []f32, scale: f32, h: usize) void {
    const lq = qt.lq;
    const lk = qt.lk;
    const kpad = kRows(lk);
    const stride = qt.heads * head_dim;
    const sm2 = scale * log2e;
    const tiles = kpad / key_tile;
    const qs_h = qt.q_scale[h * qScaleLen(lq) ..];
    const ks_h = qt.k_scale[h * kScaleLen(lk) ..];
    const k_h = qt.k[h * lk * head_dim ..][0 .. lk * head_dim];
    const v_h = qt.v[h * head_dim * kpad ..][0 .. head_dim * kpad];
    for (0..lq) |r| {
        const qv: V = @as(@Vector(head_dim, i8), qt.q[(h * lq + r) * head_dim ..][0..head_dim].*);
        const qs = qs_h[qScaleIndex(r)];
        var m: f32 = -50000;
        var d: [4]f32 = @splat(1);
        var ro: [head_dim]f32 = @splat(0);
        for (0..tiles) |t| {
            const last = t + 1 == tiles;
            const n0 = t * key_tile;
            var sc: [4]f32 = undefined;
            for (&sc, 0..) |*s, l| s.* = sm2 * (qs * ks_h[t * 4 + l]);

            var score: [key_tile]f32 = undefined;
            var lane_max: [4]f32 = @splat(if (last) -50000 else -std.math.inf(f32));
            for (0..key_tile) |j| {
                const n = n0 + j;
                if (n >= lk) {
                    score[j] = -1e30;
                } else {
                    const kv: V = @as(@Vector(head_dim, i8), k_h[n * head_dim ..][0..head_dim].*);
                    score[j] = @floatFromInt(@reduce(.Add, qv * kv));
                }
                const l = (j % 8) / 2;
                lane_max[l] = @max(lane_max[l], score[j]);
            }
            var tile_m = -std.math.inf(f32);
            for (lane_max, sc) |lm, s| tile_m = @max(tile_m, @mulAdd(f32, lm, s, -p_offset));

            const step = std.math.exp2(-@abs(m - tile_m));
            const o_scale: f32 = if (m < tile_m) step else 1;
            const t_scale: f32 = if (tile_m < m) step else 1;
            m = @max(m, tile_m);
            for (&d) |*x| x.* *= o_scale;
            for (&ro) |*x| x.* *= o_scale;

            var p: [key_tile]f32 = undefined;
            var u: [key_tile]u8 = undefined;
            for (0..key_tile) |j| {
                p[j] = std.math.exp2(@mulAdd(f32, score[j], sc[(j % 8) / 2], -tile_m));
                u[j] = toU8(p[j]);
            }
            for (0..8) |b| {
                for (0..4) |l| {
                    const ks = [4]usize{ 16 * b + 2 * l, 16 * b + 2 * l + 1, 16 * b + 2 * l + 8, 16 * b + 2 * l + 9 };
                    const den: f32 = if (last)
                        (p[ks[0]] + p[ks[1]]) + (p[ks[2]] + p[ks[3]])
                    else
                        @floatFromInt(@as(u32, u[ks[0]]) + u[ks[1]] + u[ks[2]] + u[ks[3]]);
                    d[l] = @mulAdd(f32, den, t_scale, d[l]);
                }
            }
            // keys past lk have u = 0 and zero codes
            const uv: V = @as(@Vector(key_tile, u8), u);
            for (0..head_dim) |c| {
                const vv: V = @as(@Vector(key_tile, i8), v_h[c * kpad + n0 ..][0..key_tile].*);
                ro[c] = @mulAdd(f32, @floatFromInt(@reduce(.Add, uv * vv)), t_scale, ro[c]);
            }
        }
        const rcp = 1 / ((d[0] + d[1]) + (d[2] + d[3]));
        for (0..head_dim) |c| {
            const o = (ro[c] * rcp) * qt.v_scale[h * head_dim + c];
            out[r * stride + h * head_dim + c] = dtypes.bf16ToF32(dtypes.f32ToBf16(o));
        }
    }
}

/// `quantize` then `attend`.
pub fn attention(io: std.Io, gpa: std.mem.Allocator, out: []f32, q: []const f32, k: []const f32, v: []const f32, lq: usize, lk: usize, heads: usize, scale: f32) !void {
    var qt = try quantize(gpa, q, k, v, lq, lk, heads);
    defer qt.deinit(gpa);
    try attend(io, qt, out, scale);
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const safetensors = @import("tp_core").safetensors;

/// The fixture, its head-major bf16 inputs widened and laid out token-major.
pub const Fixture = struct {
    st: safetensors.SafeTensors,
    heads: usize,
    lq: usize,
    lk: usize,
    q: []f32,
    k: []f32,
    v: []f32,
    scale: f32,

    pub fn open(gpa: std.mem.Allocator) !Fixture {
        const st = try safetensors.SafeTensors.initFromSlice(gpa, @embedFile("assets/sage_attn.safetensors"));
        const qs = st.get("q").?.info.shape.slice();
        const ks = st.get("k").?.info.shape.slice();
        var f: Fixture = .{ .st = st, .heads = qs[0], .lq = qs[1], .lk = ks[1], .q = undefined, .k = undefined, .v = undefined, .scale = 0 };
        f.q = try f.tokenMajor(gpa, "q", f.lq);
        f.k = try f.tokenMajor(gpa, "k", f.lk);
        f.v = try f.tokenMajor(gpa, "v", f.lk);
        f.scale = 1.0 / @sqrt(@as(f32, head_dim));
        return f;
    }

    pub fn deinit(f: *Fixture, gpa: std.mem.Allocator) void {
        gpa.free(f.q);
        gpa.free(f.k);
        gpa.free(f.v);
        f.st.deinit();
    }

    pub fn bytes(f: *const Fixture, name: []const u8) []const u8 {
        return f.st.get(name).?.bytes;
    }

    fn tokenMajor(f: *const Fixture, gpa: std.mem.Allocator, name: []const u8, len: usize) ![]f32 {
        const src = f.bytes(name);
        const dst = try gpa.alloc(f32, len * f.heads * head_dim);
        for (0..f.heads) |h| for (0..len) |n| for (0..head_dim) |c| {
            const i = (h * len + n) * head_dim + c;
            dst[(n * f.heads + h) * head_dim + c] = dtypes.bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
        };
        return dst;
    }

    /// The reference output, token-major.
    pub fn output(f: *const Fixture, gpa: std.mem.Allocator) ![]f32 {
        return f.tokenMajor(gpa, "out", f.lq);
    }
};

fn f32At(b: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
}

test "the rotation is orthonormal: dot products survive it" {
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var a: [head_dim]f32 = undefined;
    var b: [head_dim]f32 = undefined;
    for (&a, &b) |*x, *y| {
        x.* = rnd.floatNorm(f32);
        y.* = rnd.floatNorm(f32);
    }
    var dot0: f64 = 0;
    for (a, b) |x, y| dot0 += @as(f64, x) * y;
    rotate(&a);
    rotate(&b);
    var dot1: f64 = 0;
    for (a, b) |x, y| dot1 += @as(f64, x) * y;
    errdefer std.debug.print("dot {d} -> {d}\n", .{ dot0, dot1 });
    try testing.expectApproxEqRel(dot0, dot1, 1e-5);
}

test "int8 attention quantizes Q, K and V as comfy-kitchen does" {
    const gpa = testing.allocator;
    var f = try Fixture.open(gpa);
    defer f.deinit(gpa);
    var qt = try quantize(gpa, f.q, f.k, f.v, f.lq, f.lk, f.heads);
    defer qt.deinit(gpa);

    const want_anchor = std.mem.bytesAsSlice(i32, f.bytes("anchor"));
    for (qt.anchor, want_anchor) |a, w| try testing.expectEqual(w, a);

    // The GPU divides and takes reciprocals approximately, so a scale may sit an ulp
    // off and a code that lands on a rounding boundary may flip.
    const Pair = struct { name: []const u8, scale: []const u8, got_s: []const f32, got: []const i8 };
    for ([_]Pair{
        .{ .name = "q_int8", .scale = "q_scale", .got_s = qt.q_scale, .got = qt.q },
        .{ .name = "k_int8", .scale = "k_scale", .got_s = qt.k_scale, .got = qt.k },
    }) |p| {
        const ws = f.bytes(p.scale);
        for (p.got_s, 0..) |s, i| {
            errdefer std.debug.print("{s}[{d}]: {e} vs {e}\n", .{ p.scale, i, s, f32At(ws, i) });
            try testing.expectApproxEqRel(f32At(ws, i), s, 4e-7);
        }
        const want: []const i8 = @ptrCast(f.bytes(p.name));
        var off: usize = 0;
        for (p.got, want) |a, b| {
            if (a != b) off += 1;
            try testing.expect(@abs(@as(i16, a) - b) <= 1);
        }
        errdefer std.debug.print("{s}: {d} of {d} codes off by one\n", .{ p.name, off, want.len });
        try testing.expect(off * 1000 < want.len);
    }

    // V: the fixture stores [heads*128][128-padded keys], each 16 keys permuted.
    const ws = f.bytes("v_scale");
    for (qt.v_scale, 0..) |s, i| try testing.expectApproxEqRel(f32At(ws, i), s, 4e-7);
    const wv: []const i8 = @ptrCast(f.bytes("v_int8"));
    const pad = (f.lk + key_tile - 1) / key_tile * key_tile;
    var off: usize = 0;
    for (0..f.heads * head_dim) |row| for (0..f.lk) |n| {
        const w = n & 15;
        const dst = (n & ~@as(usize, 15)) | (w & 1) | ((w >> 3) & 1) << 1 | ((w >> 1) & 1) << 2 | ((w >> 2) & 1) << 3;
        const a = qt.v[row * kRows(f.lk) + n];
        const b = wv[row * pad + dst];
        if (a != b) off += 1;
        try testing.expect(@abs(@as(i16, a) - b) <= 1);
    };
    errdefer std.debug.print("v_int8: {d} codes off by one\n", .{off});
    try testing.expect(off * 1000 < f.heads * head_dim * f.lk);
}

test "int8 attention matches comfy-kitchen's output" {
    const gpa = testing.allocator;
    var f = try Fixture.open(gpa);
    defer f.deinit(gpa);
    const got = try gpa.alloc(f32, f.lq * f.heads * head_dim);
    defer gpa.free(got);
    try attention(testing.io, gpa, got, f.q, f.k, f.v, f.lq, f.lk, f.heads, f.scale);
    const want = try f.output(gpa);
    defer gpa.free(want);

    // An exact-attention port lands 3% away (the generator asserts it); a faithful one
    // differs only where an approximate exp2 flipped a u8 probability.
    var num: f64 = 0;
    var den: f64 = 0;
    var same: usize = 0;
    for (got, want) |a, b| {
        num += (a - b) * (a - b);
        den += b * b;
        if (a == b) same += 1;
    }
    const rel = @sqrt(num / den);
    errdefer std.debug.print("rel L2 {e}, {d} of {d} bf16 values identical\n", .{ rel, same, got.len });
    try testing.expect(rel < 2e-3);
    try testing.expect(same * 10 > got.len * 9);
}
