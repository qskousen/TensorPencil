//! Elementwise dual-target kernel bodies (see ../dual.zig for the prelude).
//!
//! One thread per element. A typed body (`ldT` / `stT`) stores one 16-bit element per
//! thread (the SPIR-V module carries StorageBuffer16BitAccess); the older f16 bodies
//! that index PAIRS of words are launched with `elems / 2` threads (`dualPairs` on
//! CUDA).
//! Layouts are the Vulkan push layouts, since Vulkan model files carry them at
//! the call site; the CUDA wrappers in gpu/cuda/backend.zig remap their arguments.
//! Each body's doc names the slots and scalars it reads.

const k = @import("../dual.zig");
const Dt = k.Dt;
const Env = k.Env;

// ---- activations ------------------------------------------------------------

/// a += b, both `dt`. u0 = n.
pub inline fn add(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) + e.ldT(.b, dt, i));
}

/// d[0] = 1 when any of a[0..u0) (`dt`) is not finite, else d is left as it was: the
/// caller zeroes it. Every writer stores the same word, so the race is benign.
pub inline fn flagNonFinite(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const bits: u32 = @bitCast(e.ldT(.a, dt, i));
    if (bits & 0x7f80_0000 == 0x7f80_0000) e.stW(.d, 0, 1);
}

/// a = max(0, a + b). u0 = n.
pub inline fn addRelu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, @max(0.0, e.ld(.a, i) + e.ld(.b, i)));
}

/// Row-major f32 `[u3][u1]` into `[u1][u2]`, rows past u3 zeroed: the transposed,
/// padded layout the f32 GEMM reads its weight in. u0 = u1 * u2.
pub inline fn transposePadF32(e: Env) void {
    const i = k.elem(e) orelse return;
    const stride = e.u(2);
    const c = i / stride;
    const r = i % stride;
    e.st(.b, i, if (r < e.u(3)) e.ld(.a, r * e.u(1) + c) else 0);
}

/// a += f0 * b, a `dt` and b f32. u0 = n.
pub inline fn addScaled(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) + e.f(0) * e.ld(.b, i));
}

/// a = max(0, a). u0 = n.
pub inline fn relu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, @max(0.0, e.ld(.a, i)));
}

/// a = silu(a), `dt`. u0 = n.
pub inline fn silu(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, k.silu(e.ldT(.a, dt, i)));
}

/// a = silu(a) * b, both `dt`. u0 = n.
pub inline fn siluMul(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, k.silu(e.ldT(.a, dt, i)) * e.ldT(.b, dt, i));
}

/// MOSS-Image's soft-capped gated pair: a = 4 tanh(b / 4) sigmoid(b) * 25 tanh(a / 25),
/// a the hidden half and b the gate, both `dt`. u0 = n.
pub inline fn situ(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const h = e.ldT(.a, dt, i);
    const g = e.ldT(.b, dt, i);
    e.stT(.a, dt, i, (4.0 * k.tanh(g * 0.25) * k.sigmoid(g)) * (25.0 * k.tanh(h * 0.04)));
}

/// a *= sigmoid(b), both `dt`. u0 = n.
pub inline fn sigmoidMul(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) * k.sigmoid(e.ldT(.b, dt, i)));
}

/// a = geluTanh(a). u0 = n.
pub inline fn gelu(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, k.geluTanh(e.ldT(.a, dt, i)));
}

/// a = geluTanh(a) * b. u0 = n.
pub inline fn geluMul(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.geluTanh(e.ld(.a, i)) * e.ld(.b, i));
}

/// a = geluQuick(a). u0 = n. Not interchangeable with the tanh or erf forms:
/// the three agree to ~1e-2, which shifts style.
pub inline fn geluQuick(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.geluQuick(e.ld(.a, i)));
}

/// a = geluQuick(a) * b. u0 = n.
pub inline fn geluQuickMul(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.geluQuick(e.ld(.a, i)) * e.ld(.b, i));
}

/// a = geluErf(a). u0 = n.
pub inline fn geluErf(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, k.geluErf(e.ldT(.a, dt, i)));
}

/// GEGLU: b[p][j] = a[p][j] * geluErf(a[p][inner + j]), value then gate in that
/// order. a = src [n][2*u1], b = dst [n][u1], both `dt`. u0 = n*u1, u1 = inner.
pub inline fn geglu(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const inner = e.u(1);
    const base = (i / inner) * 2 * inner + i % inner;
    e.stT(.b, dt, i, e.ldT(.a, dt, base) * k.geluErf(e.ldT(.a, dt, base + inner)));
}

/// a *= softplus(gate*ln2)/ln2, as log2(1 + 2^gate), stable on both signs. u0 = n.
pub inline fn softplusGate(e: Env) void {
    const i = k.elem(e) orelse return;
    const g = e.ld(.b, i);
    const sp = if (g > 0) g + k.log2(1.0 + @exp2(-g)) else k.log2(1.0 + @exp2(g));
    e.st(.a, i, e.ld(.a, i) * sp);
}

// ---- copies, scales, concatenation ------------------------------------------

/// b[u2 + i] = a[u3 + i], a in `in` and b in `out`. u0 = n.
pub inline fn copy(e: Env, comptime in: Dt, comptime out: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.b, out, e.u(2) + i, e.ldT(.a, in, e.u(3) + i));
}

/// b[u2 + i] = a[u3 + i] * f0, in place when a and b are one buffer. u0 = n.
pub inline fn scaleF32(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, e.u(2) + i, e.ld(.a, e.u(3) + i) * e.f(0));
}

/// Channel concatenation, one thread per SOURCE element: b[p][u3 + j] = a[p][j].
/// u0 = n*u1, u1 = src channels, u2 = dst channels, u3 = dst channel offset.
pub inline fn concatCh(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const j = i % e.u(1);
    e.stT(.b, dt, (i / e.u(1)) * e.u(2) + e.u(3) + j, e.ldT(.a, dt, i));
}

/// Fused int8 GEMM scale buffer = [act(m_pad) | weight(rows)]. a = act scales
/// (u1 = m of them), c = weight scales, b = out. u0 = m_pad + rows, u2 = m_pad.
pub inline fn scaleConcat(e: Env) void {
    const i = k.elem(e) orelse return;
    if (i < e.u(1)) {
        e.st(.b, i, e.ld(.a, i));
    } else if (i >= e.u(2)) {
        e.st(.b, i, e.ld(.c, i - e.u(2)));
    } else {
        e.st(.b, i, 0);
    }
}

/// y = s32_acc * act_scale[row] * weight_scale[col], y in `dt`. a = acc (s32 bits),
/// c = weight_scale [u1], d = act_scale [rows], b = y. u0 = rows*u1, u1 = nout.
pub inline fn scaleI32(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const acc: i32 = @bitCast(e.ldW(.a, i));
    e.stT(.b, dt, i, @as(f32, @floatFromInt(acc)) * e.ld(.d, i / e.u(1)) * e.ld(.c, i % e.u(1)));
}

/// Pack 4 int8 per word: q = clamp(round(a/scale[row]), -127, 127). a = x,
/// d = scale [rows], b = packed. u0 = words, u1 = cols, u2 = real elements
/// (words past it, GEMM pad rows, zero).
pub inline fn quantizeI8(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 4;
    if (e0 >= e.u(2)) {
        e.stW(.b, w, 0);
        return;
    }
    const inv = 1.0 / e.ld(.d, e0 / e.u(1));
    var out: u32 = 0;
    inline for (0..4) |j_| {
        const j: u32 = @intCast(j_);
        var qi: i32 = @intFromFloat(@round(e.ld(.a, e0 + j) * inv));
        qi = @max(@as(i32, -127), @min(@as(i32, 127), qi));
        out |= @as(u32, @as(u8, @bitCast(@as(i8, @intCast(qi))))) << @intCast(8 * j);
    }
    e.stW(.b, w, out);
}

// ---- modulation, residual gates, rms epilogues -------------------------------

/// a = (1 + c[u2 + col]) * a + c[u3 + col]. u0 = n, u1 = dim.
pub inline fn modulate(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const col = i % e.u(1);
    e.stT(.a, dt, i, (1.0 + e.ld(.c, e.u(2) + col)) * e.ldT(.a, dt, i) + e.ld(.c, e.u(3) + col));
}

/// a += c[gate + col] * b, gate = u2, or per row u2 + d[row]*u3 when u3 != 0
/// (d holds one u32 label per row: a denoise mask relabels rows inside a segment).
/// a and b in `dt`. u0 = n, u1 = dim, u2 = gate_off, u3 = label stride (0 = uniform).
pub inline fn gatedAdd(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const dim = e.u(1);
    const col = i % dim;
    var off = e.u(2);
    if (e.u(3) != 0) off += e.ldW(.d, e.u(6) + i / dim) * e.u(3);
    // u4/u5 are ELEMENT offsets into a and b, u6 a ROW offset into the index buffer,
    // for the same reason `rmsMod` takes them. Zero for a whole-buffer caller.
    e.stT(.a, dt, e.u(4) + i, e.ldT(.a, dt, e.u(4) + i) + e.ld(.c, off + col) * e.ldT(.b, dt, e.u(5) + i));
}

/// a += f0 * gate * b, a `dt` and b f32: `gatedAdd` into a narrower residual than the
/// delta, `f0` the residual's storage scale. Same pushes as `gatedAdd`.
pub inline fn gatedAddScaled(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const dim = e.u(1);
    const col = i % dim;
    var off = e.u(2);
    if (e.u(3) != 0) off += e.ldW(.d, e.u(6) + i / dim) * e.u(3);
    e.stT(.a, dt, e.u(4) + i, e.ldT(.a, dt, e.u(4) + i) + e.f(0) * e.ld(.c, off + col) * e.ld(.b, e.u(5) + i));
}

/// `gatedAdd` with an f16 delta. a = x (f32), b = delta words, c = vectors.
/// u0 = words (n/2), u1 = dim, u2 = gate_off.
pub inline fn gatedAdd16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const word = e.ldW(.b, w);
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const idx = e0 + j;
        const col = idx % e.u(1);
        const dv = k.h16(word >> (16 * j));
        e.st(.a, idx, e.ld(.a, idx) + e.ld(.c, e.u(2) + col) * dv);
    }
}

/// y = x * inv[row] * w[col]. a = x, b = out, c = weight, d = inv. u0 = n, u1 = dim.
pub inline fn rmsApplyW(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, i, e.ld(.a, i) * e.ld(.d, i / e.u(1)) * e.ld(.c, i % e.u(1)));
}

/// Sum of squares over the interleaved slice i = chunk, chunk + u2, ... of one
/// row, the first pass of the 3-pass rmsnorm. a = x, d = partials [row][u2].
/// u0 = rows*u2, u1 = dim, u2 = chunks per row.
pub inline fn rmsPartial(e: Env) void {
    const i = k.elem(e) orelse return;
    const nch = e.u(2);
    const base = (i / nch) * e.u(1);
    var sum: f32 = 0;
    var j = i % nch;
    while (j < e.u(1)) : (j += nch) {
        const v = e.ld(.a, base + j);
        sum += v * v;
    }
    e.st(.d, i, sum);
}

/// inv_rms from the chunk partials. a = partials, d = inv [rows]. u0 = rows,
/// u1 = dim, u2 = chunks, f0 = eps.
pub inline fn rmsCombine(e: Env) void {
    const row = k.elem(e) orelse return;
    var sum: f32 = 0;
    var j: u32 = 0;
    while (j < e.u(2)) : (j += 1) sum += e.ld(.a, row * e.u(2) + j);
    e.st(.d, row, 1.0 / @sqrt(sum / @as(f32, @floatFromInt(e.u(1))) + e.f(0)));
}

// ---- rope -------------------------------------------------------------------

/// Interleaved RoPE in place, one thread per (pos, head, pair). a = qk, c = freqs
/// (cos then sin at u2), qk in `dt`. u0 = total pairs, u1 = half, u2 = sin_off,
/// u3 = n_heads.
pub inline fn ropeInter(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const pair = i % half;
    const pos = i / (half * e.u(3));
    const cos_v = e.ld(.c, pos * half + pair);
    const sin_v = e.ld(.c, e.u(2) + pos * half + pair);
    const at = i * 2;
    const x0 = e.ldT(.a, dt, at);
    const x1 = e.ldT(.a, dt, at + 1);
    e.stT(.a, dt, at, x0 * cos_v - x1 * sin_v);
    e.stT(.a, dt, at + 1, x0 * sin_v + x1 * cos_v);
}

inline fn rotateHalf(e: Env, base: u32, half: u32, cos_v: f32, sin_v: f32) void {
    const lo = e.ld(.a, base);
    const hi = e.ld(.a, base + half);
    e.st(.a, base, lo * cos_v - hi * sin_v);
    e.st(.a, base + half, hi * cos_v + lo * sin_v);
}

/// Rotate-half RoPE in place (pairs i with i+half). a = qk, c = freqs.
/// u0 = total pairs, u1 = half, u2 = sin_off, u3 = n_heads, u4 = position of
/// row 0, u5 = element base offset into a (a batched item's sub-region).
pub inline fn ropeHalf(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const pos = i / (half * e.u(3)) + e.u(4);
    const row = i / half;
    rotateHalf(e, e.u(5) + row * 2 * half + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

/// `ropeHalf` with one absolute position per row from b (u32). u0..u3 as above.
pub inline fn ropeHalfPos(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const pos = e.ldW(.b, i / (half * e.u(3)));
    const row = i / half;
    rotateHalf(e, row * 2 * half + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

/// Partial rotate-half RoPE: the first 2*half dims of u5-wide heads rotate, the
/// rest pass through. u0 = total pairs, u1 = half, u2 = sin_off, u3 = n_heads,
/// u4 = pos0, u5 = head_dim.
pub inline fn ropeHalfPart(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const pos = i / (half * e.u(3)) + e.u(4);
    const row = i / half;
    rotateHalf(e, row * e.u(5) + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

// ---- qwen35 hybrid ------------------------------------------------------------

/// Split per-head [q(hd) gate(hd)] into c = q and d = gate. u0 = heads*hd, u1 = hd.
pub inline fn deinterleave2(e: Env) void {
    const i = k.elem(e) orelse return;
    const hd = e.u(1);
    const src = (i / hd) * 2 * hd + i % hd;
    e.st(.c, i, e.ld(.a, src));
    e.st(.d, i, e.ld(.a, src + hd));
}

/// Split per-head fused [q | k | v] into b, c, d. u0 = tokens*heads*hd, u1 = hd.
pub inline fn deinterleave3(e: Env) void {
    const i = k.elem(e) orelse return;
    const hd = e.u(1);
    const src = (i / hd) * 3 * hd + i % hd;
    e.st(.b, i, e.ld(.a, src));
    e.st(.c, i, e.ld(.a, src + hd));
    e.st(.d, i, e.ld(.a, src + 2 * hd));
}

inline fn gdnGate(alpha: f32, beta_raw: f32, av: f32, dt: f32) [2]f32 {
    return .{ k.exp(av * k.softplus(alpha + dt)), k.sigmoid(beta_raw) };
}

/// Per-head gates: decay = exp(a*softplus(alpha+dt)), beta = sigmoid(beta_raw).
/// a = [alpha | beta_raw], b = [a | dt], d = [decay | beta]. u0 = heads.
pub inline fn gdnGates(e: Env) void {
    const h = k.elem(e) orelse return;
    const heads = e.u(0);
    const g = gdnGate(e.ld(.a, h), e.ld(.a, heads + h), e.ld(.b, h), e.ld(.b, heads + h));
    e.st(.d, h, g[0]);
    e.st(.d, heads + h, g[1]);
}

/// `gdnGates` for a whole chunk. a = alpha [n][stride], b = beta [n][stride],
/// c = [a | dt], d = out [n][2*heads]. u0 = n*heads, u1 = heads, u2 = row stride
/// of a and b (0 = heads).
pub inline fn gdnGatesBatch(e: Env) void {
    const i = k.elem(e) orelse return;
    const heads = e.u(1);
    const stride = if (e.u(2) != 0) e.u(2) else heads;
    const t = i / heads;
    const h = i % heads;
    const g = gdnGate(e.ld(.a, t * stride + h), e.ld(.b, t * stride + h), e.ld(.c, h), e.ld(.c, heads + h));
    e.st(.d, t * 2 * heads + h, g[0]);
    e.st(.d, t * 2 * heads + heads + h, g[1]);
}

/// One step of the depthwise causal conv + SiLU; the (taps-1)-column state rolls
/// forward. a = state [ch][taps-1] (in place), b = x [ch], c = w [ch][taps],
/// d = out [ch]. u0 = channels, u1 = taps.
pub inline fn gdnConvStep(e: Env) void {
    const ch = k.elem(e) orelse return;
    const taps = e.u(1);
    const stb = ch * (taps - 1);
    const wb = ch * taps;
    var acc = e.ld(.c, wb + taps - 1) * e.ld(.b, ch);
    var t: u32 = 0;
    while (t < taps - 1) : (t += 1) acc += e.ld(.c, wb + t) * e.ld(.a, stb + t);
    t = 0;
    while (t < taps - 2) : (t += 1) e.st(.a, stb + t, e.ld(.a, stb + t + 1));
    e.st(.a, stb + taps - 2, e.ld(.b, ch));
    e.st(.d, ch, k.silu(acc));
}

/// The 4-tap conv over a whole chunk, one thread per (token, channel), state
/// read-only; taps summed in `gdnConvStep`'s order (w3*x first) so the two agree
/// bit for bit. a = state [ch][3], b = x [n][ch], c = w [ch][4], d = out [n][ch].
/// u0 = n*ch, u1 = ch, u2 = n.
pub inline fn gdnConvBatch(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(1);
    const t = i / ch;
    const c = i % ch;
    var acc = e.ld(.c, c * 4 + 3) * e.ld(.b, i);
    inline for (0..3) |tap_| {
        const tap: u32 = @intCast(tap_);
        // Column t-3+tap: from x when it exists, else the carried state.
        const s = if (t + tap >= 3) e.ld(.b, (t + tap - 3) * ch + c) else e.ld(.a, c * 3 + t + tap);
        acc += e.ld(.c, c * 4 + tap) * s;
    }
    e.st(.d, i, k.silu(acc));
}

/// Roll the conv state to the chunk's last 3 columns; the old values are read
/// before any write since a short chunk keeps some. a = state [ch][3] (in place),
/// b = x [n][ch]. u0 = ch, u1 = n.
pub inline fn gdnConvState(e: Env) void {
    const c = k.elem(e) orelse return;
    const n = e.u(1);
    const old = [3]f32{ e.ld(.a, c * 3), e.ld(.a, c * 3 + 1), e.ld(.a, c * 3 + 2) };
    inline for (0..3) |col_| {
        const col: u32 = @intCast(col_);
        const v = if (n + col >= 3) e.ld(.b, (n + col - 3) * e.u(0) + c) else old[n + col];
        e.st(.a, c * 3 + col, v);
    }
}

// ---- sampling ---------------------------------------------------------------

/// Penalties, one thread per unique recent token: a[id] = (l > 0 ? l/rp : l*rp)
/// - sub. b = wire of (id as exact f32, sub) pairs (sample.packPenaltyWireF32).
/// u0 = entries, f0 = repeat penalty.
pub inline fn penalize(e: Env) void {
    const i = k.elem(e) orelse return;
    const id: u32 = @intFromFloat(e.ld(.b, i * 2));
    const sub = e.ld(.b, i * 2 + 1);
    const l = e.ld(.a, id);
    const r = if (l > 0) l / e.f(0) else l * e.f(0);
    e.st(.a, id, r - sub);
}

/// Argmax pass 1: lane i stride-scans a[i], a[i+L], ... keeping its max and the
/// LOWEST index achieving it. c = out_val [L], d = out_idx [L] (index as an
/// exact f32). u0 = L, u1 = vocab.
pub inline fn argmaxReduce(e: Env) void {
    const lane = k.elem(e) orelse return;
    var best_i: u32 = lane;
    var best_v: f32 = e.ld(.a, lane);
    var i: u32 = lane + e.u(0);
    while (i < e.u(1)) : (i += e.u(0)) {
        const v = e.ld(.a, i);
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    e.st(.c, lane, best_v);
    e.st(.d, lane, @floatFromInt(best_i));
}

/// Argmax pass 2, one subgroup: the lane winners to the global argmax, lowest index
/// on ties. a = vals [L], b = idx [L], d = out [1]. u0 = L.
pub inline fn argmaxFinal(e: Env) void {
    if (k.sgGlobal(e) != 0) return;
    const lane = e.sgLane();
    // -inf, so an all -inf row still yields its lowest index.
    var best_v: f32 = @bitCast(@as(u32, 0xff800000));
    var best_i: f32 = 3.0e38;
    var i = lane;
    while (i < e.u(0)) : (i += e.sgSize()) {
        const v = e.ld(.a, i);
        const idx = e.ld(.b, i);
        if (v > best_v or (v == best_v and idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }
    const top = k.sgMax(best_v);
    // The lowest index among the lanes holding the max, as a max of negations.
    const lowest = -k.sgMax(if (best_v == top) -best_i else -3.0e38);
    if (lane == 0) e.st(.d, 0, lowest);
}

/// Lanes per top-k launch on both hosts.
pub const topk_m = 8;

/// Top-k pass: L lanes each keep their `topk_m` highest (value, index) over their
/// stride-slice by min-slot tracking. c = out_val [L*M], d = out_idx [L*M].
/// u0 = L, u1 = vocab. The host does the exact top-k over the L*M candidates.
pub inline fn topkReduce(e: Env) void {
    const lane = k.elem(e) orelse return;
    const neg_max: f32 = -3.4028235e38;
    var bv: [topk_m]f32 = @splat(neg_max);
    var bi: [topk_m]u32 = @splat(0);
    var i: u32 = lane;
    while (i < e.u(1)) : (i += e.u(0)) {
        const v = e.ld(.a, i);
        var mj: u32 = 0;
        var mv: f32 = bv[0];
        var j: u32 = 1;
        while (j < topk_m) : (j += 1) {
            if (bv[j] < mv) {
                mv = bv[j];
                mj = j;
            }
        }
        if (v > mv) {
            bv[mj] = v;
            bi[mj] = i;
        }
    }
    const base = lane * topk_m;
    var j: u32 = 0;
    while (j < topk_m) : (j += 1) {
        e.st(.c, base + j, bv[j]);
        e.st(.d, base + j, @floatFromInt(bi[j]));
    }
}

// ---- head padding / gathers for the tensor-core attention --------------------

/// Tight f32 [seq][heads*hd_src] -> f16 [seq_pad][heads*hd_out], each head
/// zero-extended and rows past seq zeroed, f0 folded in. a = src, d = f16 pairs.
/// u0 = out words, u1 = hd_src, u2 = hd_out (even), u3 = seq, u4 = heads.
pub inline fn headPadH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const hd_out = e.u(2);
    const row_out = e.u(4) * hd_out;
    const row = e0 / row_out;
    const rem = e0 % row_out;
    const h = rem / hd_out;
    const t0 = rem % hd_out;
    var out: u32 = 0;
    if (row < e.u(3)) {
        inline for (0..2) |j_| {
            const j: u32 = @intCast(j_);
            const t = t0 + j;
            if (t < e.u(1)) out |= @as(u32, k.f16Bits(e.ld(.a, (row * e.u(4) + h) * e.u(1) + t) * e.f(0))) << (16 * j);
        }
    }
    e.stW(.d, w, out);
}

/// The f32 inverse: [seq_pad][heads*hd_out] -> tight [seq][heads*hd_src].
/// a = src, b = dst. u0 = seq*heads*hd_src, u1 = hd_src, u2 = hd_out, u4 = heads.
pub inline fn headUnpad(e: Env) void {
    const i = k.elem(e) orelse return;
    const row_in = e.u(4) * e.u(1);
    const row = i / row_in;
    const rem = i % row_in;
    e.st(.b, i, e.ld(.a, row * e.u(4) * e.u(2) + (rem / e.u(1)) * e.u(2) + rem % e.u(1)));
}

/// f32 head restride: out[t][h][d] = d < in_hd ? in[t*in_stride + in_off + h*in_hd
/// + d] : 0. a = in, b = out. u0 = rows*heads*out_hd, u1 = out_hd, u2 = in_hd,
/// u3 = in_stride, u4 = in_off, u5 = heads.
pub inline fn headPad(e: Env) void {
    const i = k.elem(e) orelse return;
    const hp = i / e.u(1);
    const d = i % e.u(1);
    const t = hp / e.u(5);
    const h = hp % e.u(5);
    const v: f32 = if (d < e.u(2)) e.ld(.a, t * e.u(3) + e.u(4) + h * e.u(2) + d) else 0;
    e.st(.b, i, v);
}

/// One head's rows from interleaved [seq][nheads][hd] f32 into a [mpad][hd] f16
/// tile, rows >= seq zero. a = src, b = f16 pairs. u0 = seq, u1 = nheads,
/// u2 = head, u3 = hd (even), u4 = total elements (mpad*hd).
pub inline fn gatherHead(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(4)) return;
    const hd = e.u(3);
    const row = e0 / hd;
    var out: u32 = 0;
    if (row < e.u(0)) {
        const src = (row * e.u(1) + e.u(2)) * hd + e0 % hd;
        out = k.packH16(e.ld(.a, src), e.ld(.a, src + 1));
    }
    e.stW(.b, w, out);
}

/// One KV head's V transposed to [hd][mpad] f16, columns >= seq zero. a = src
/// [seq][kv_heads][hd], b = f16 pairs. u0 = seq, u1 = kv_heads, u2 = kv head,
/// u3 = hd, u4 = mpad (even), u5 = total elements (hd*mpad).
pub inline fn gatherVt(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(5)) return;
    const mpad = e.u(4);
    const c = e0 / mpad;
    const j0 = e0 % mpad;
    var v: [2]f32 = .{ 0, 0 };
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (j0 + j < e.u(0)) v[j] = e.ld(.a, ((j0 + j) * e.u(1) + e.u(2)) * e.u(3) + c);
    }
    e.stW(.b, w, k.packH16(v[0], v[1]));
}

/// One head's [mpad][hd] f32 output (rows 0..seq) into interleaved
/// [seq][heads][hd]. a = src, b = dst. u0 = seq, u1 = heads, u2 = head, u3 = hd,
/// u4 = total (seq*hd).
pub inline fn scatterHead(e: Env) void {
    const i = e.gid();
    if (i >= e.u(4)) return;
    const hd = e.u(3);
    e.st(.b, ((i / hd) * e.u(1) + e.u(2)) * hd + i % hd, e.ld(.a, i));
}

/// `gatherHead` for `gsize` heads at once: b = [gsize][mpad][hd] f16, head =
/// (base_h + z) / group_div. u0 = seq, u1 = nheads, u2 = base_h, u3 = group_div,
/// u4 = hd, u5 = mpad, u6 = total elements.
pub inline fn gatherHeadB(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(6)) return;
    const hd = e.u(4);
    const sl = hd * e.u(5);
    const z = e0 / sl;
    const rem = e0 % sl;
    const row = rem / hd;
    var out: u32 = 0;
    if (row < e.u(0)) {
        const head = (e.u(2) + z) / e.u(3);
        const src = (row * e.u(1) + head) * hd + rem % hd;
        out = k.packH16(e.ld(.a, src), e.ld(.a, src + 1));
    }
    e.stW(.b, w, out);
}

/// `gatherVt` for `gsize` heads: b = [gsize][hd][mpad] f16, kv head = (base_h + z)
/// / group. u0 = seq, u1 = kv_heads, u2 = base_h, u3 = group, u4 = hd, u5 = mpad,
/// u6 = total elements.
pub inline fn gatherVtB(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(6)) return;
    const mpad = e.u(5);
    const sl = e.u(4) * mpad;
    const z = e0 / sl;
    const rem = e0 % sl;
    const c = rem / mpad;
    const j0 = rem % mpad;
    const head = (e.u(2) + z) / e.u(3);
    var v: [2]f32 = .{ 0, 0 };
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (j0 + j < e.u(0)) v[j] = e.ld(.a, ((j0 + j) * e.u(1) + head) * e.u(4) + c);
    }
    e.stW(.b, w, k.packH16(v[0], v[1]));
}

/// `scatterHead` for `gsize` heads: a = [gsize][mpad][hd] f32 -> b[row][base_h+z][hd].
/// u0 = seq, u1 = heads, u2 = base_h, u3 = hd, u4 = mpad, u5 = total (gsize*seq*hd).
pub inline fn scatterHeadB(e: Env) void {
    const i = e.gid();
    if (i >= e.u(5)) return;
    const hd = e.u(3);
    const sl = e.u(0) * hd;
    const z = i / sl;
    const rem = i % sl;
    e.st(.b, ((rem / hd) * e.u(1) + e.u(2) + z) * hd + rem % hd, e.ld(.a, z * e.u(4) * hd + rem));
}

/// d[(h*hd + k)*seq + s] = a[(s*n_heads + h)*hd + k], per-head k-major.
/// u0 = seq*n_heads*hd, u1 = n_heads, u2 = hd, u3 = seq.
pub inline fn gatherKmajor(e: Env) void {
    const i = k.elem(e) orelse return;
    const hd = e.u(2);
    const kk = i % hd;
    const h = (i / hd) % e.u(1);
    const s = i / (hd * e.u(1));
    e.st(.d, (h * hd + kk) * e.u(3) + s, e.ld(.a, i));
}

/// f16 per-head k-major gather: d = [kv][hd][s_stride] pairs from a = f32
/// [seq][kv][hd_src]; positions >= seq and head rows >= hd_src write zero.
/// u0 = out words, u1 = hd, u2 = s_stride (even), u3 = seq, u4 = n_kv_heads,
/// u5 = source hd when narrower than u1 (0 = same).
pub inline fn gatherKmajorH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const hd = e.u(1);
    const hd_src = if (e.u(5) == 0) hd else e.u(5);
    const plane = hd * e.u(2);
    const h = e0 / plane;
    const rem = e0 % plane;
    const kk = rem / e.u(2);
    const s0 = rem % e.u(2);
    var out: u32 = 0;
    if (kk < hd_src) {
        inline for (0..2) |j_| {
            const j: u32 = @intCast(j_);
            const s = s0 + j;
            if (s < e.u(3)) out |= @as(u32, k.f16Bits(e.ld(.a, (s * e.u(4) + h) * hd_src + kk))) << (16 * j);
        }
    }
    e.stW(.d, w, out);
}

/// `gatherKmajorH16` from an f16 source, raw 16-bit moves. a = f16 [seq][kv][hd]
/// words. u0 = out words, u1 = hd, u2 = s_stride, u3 = seq, u4 = n_kv_heads.
pub inline fn gatherKmajor16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const hd = e.u(1);
    const plane = hd * e.u(2);
    const h = e0 / plane;
    const rem = e0 % plane;
    const kk = rem / e.u(2);
    const s0 = rem % e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const s = s0 + j;
        if (s < e.u(3)) out |= @as(u32, e.h16Bits(.a, (s * e.u(4) + h) * hd + kk)) << (16 * j);
    }
    e.stW(.d, w, out);
}

// ---- f16 / bf16 conversions and padded copies -------------------------------

/// f32 -> f16 pairs, zero past u1 real elements, scaled by f0. a = src, d = words.
/// u0 = out words, u1 = real elems.
pub inline fn f32ToH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (e0 + j < e.u(1)) out |= @as(u32, k.f16Bits(e.ld(.a, e0 + j) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// x [m][cols] (`dt`, rows u4 apart, or cols when u4 = 0) to a GEMM operand
/// [m_pad][k_pad] scaled by f0, zeros in both pads, two elements a thread: f16 for
/// `padH16`, bf16 for `padBf16`. u0 = output words, u1 = cols, u2 = k_pad, u3 = m.
pub inline fn padH16(e: Env, comptime dt: Dt) void {
    padTo(e, dt, false);
}

pub inline fn padBf16(e: Env, comptime dt: Dt) void {
    padTo(e, dt, true);
}

inline fn padTo(e: Env, comptime dt: Dt, comptime bf: bool) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const row = e0 / e.u(2);
    const col = e0 % e.u(2);
    const ld = if (e.u(4) != 0) e.u(4) else e.u(1);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const cc = col + j;
        if (cc < e.u(1) and row < e.u(3)) {
            const v = e.ldT(.a, dt, row * ld + cc) * e.f(0);
            out |= @as(u32, if (bf) k.bf16Bits(v) else k.f16Bits(v)) << @intCast(16 * j);
        }
    }
    e.stW(.d, w, out);
}

/// A weight to `dt` (f16 or bf16) in conv patch order `[co][kk][ci]`, two elements per thread. `a`
/// holds f32 (u3 = 0), f16 (1) or bf16 (2), in patch order (u4 = 0) or torch's
/// `[co][ci][kt][kk]` (1), read at temporal slice `kt - 1` (u6 = kt, 1 for a plain
/// conv). u0 = pairs, u1 = ci, u2 = kk, u5 = elements. A linear is `kk = 1`, where
/// the two orders coincide.
pub inline fn convW16(e: Env, comptime dt: Dt) void {
    if (dt == .f32) @compileError("convW16 writes a 16-bit weight");
    const w = k.elem(e) orelse return;
    const ci = e.u(1);
    const kk = e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const d = w * 2 + j;
        if (d < e.u(5)) {
            const s = if (e.u(4) == 0) d else blk: {
                const i = d % ci;
                const t = (d / ci) % kk;
                const o = d / (ci * kk);
                const kt = e.u(6);
                break :blk ((o * ci + i) * kt + (kt - 1)) * kk + t;
            };
            const v = if (e.u(3) == 0) e.ld(.a, s) else if (e.u(3) == 1) e.h16(.a, s) else e.bf16(.a, s);
            out |= @as(u32, if (dt == .bf16) k.bf16Bits(v) else k.f16Bits(v)) << (16 * j);
        }
    }
    e.stW(.d, w, out);
}

/// b[i] = f32(a[i]) from f16. u0 = n.
pub inline fn f16ToF32(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, i, e.h16(.a, i));
}

/// b[i] = f32(a[i]) from bf16. u0 = n. The widening every bf16 consumer needs on
/// the way back to an f32 activation stream.
pub inline fn bf16ToF32(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, i, e.bf16(.a, i));
}

/// d = f16(silu(a) * b * f0) pairs, zero past u1 real elements. u0 = out words.
pub inline fn siluMulH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (e0 + j < e.u(1)) out |= @as(u32, k.f16Bits(k.silu(e.ld(.a, e0 + j)) * e.ld(.b, e0 + j) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// d = f16(a * sigmoid(b) * f0) pairs, zero past u1 real elements. u0 = out words.
pub inline fn sigmoidMulH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (e0 + j < e.u(1)) out |= @as(u32, k.f16Bits(e.ld(.a, e0 + j) * k.sigmoid(e.ld(.b, e0 + j)) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// `siluMulH16` with f16 gate (a) and up (b) words. u0 = words, f0 = scale.
pub inline fn siluMul16(e: Env) void {
    const w = k.elem(e) orelse return;
    const gw = e.ldW(.a, w);
    const uw = e.ldW(.b, w);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const g = k.h16(gw >> (16 * j));
        const u = k.h16(uw >> (16 * j));
        out |= @as(u32, k.f16Bits(k.silu(g) * u * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// `sigmoidMulH16` with an f16 gate (b words); a stays f32, bound by u1 real
/// elements. u0 = words, f0 = scale.
pub inline fn sigmoidMulG16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const gw = e.ldW(.b, w);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        if (e0 + j < e.u(1)) out |= @as(u32, k.f16Bits(e.ld(.a, e0 + j) * k.sigmoid(k.h16(gw >> (16 * j))) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// f32 K/V rows into an f16 cache, one thread per destination word: b[u2 + w] =
/// pack(a[u3 + 2w], a[u3 + 2w + 1]). u0 = words.
pub inline fn kvStoreF16(e: Env) void {
    const w = k.elem(e) orelse return;
    e.stW(.b, e.u(2) + w, k.packH16(e.ld(.a, e.u(3) + 2 * w), e.ld(.a, e.u(3) + 2 * w + 1)));
}

/// 16-bit rows [rows][cols] in a -> k-major [k_pad][n_pad] in d (element (k, row) at
/// k*n_pad + row), zeros in both pads. A thread takes one row pair and 16 k: it reads a
/// 32-byte run of each row and writes 16 words, one per output row, its neighbours
/// beside it, so both sides move whole sectors with no shared memory. u0 = threads,
/// u1 = rows, u2 = cols, u3 = n_pad / 2.
pub inline fn transposeKmajor16(e: Env) void {
    const t = k.elem(e) orelse return;
    const rp = t % e.u(3);
    const kc = t / e.u(3);
    const r0 = 2 * rp;
    const half = e.u(2) / 2;
    // cols % 32 == 0, so a chunk is wholly inside the row or wholly in the k pad, and
    // both of its 16-byte halves are aligned.
    var w: [2][8]u32 = @splat(@splat(0));
    if (kc * 8 < half) {
        inline for (0..2) |r_| {
            const r: u32 = @intCast(r_);
            if (r0 + r < e.u(1)) inline for (0..2) |h_| {
                const h: u32 = @intCast(h_);
                const v = e.ld4(.a, (r0 + r) * half + kc * 8 + 4 * h);
                inline for (0..4) |q| w[r][4 * h + q] = @bitCast(v[q]);
            };
        }
    }
    inline for (0..8) |j_| {
        const j: u32 = @intCast(j_);
        const kw = kc * 8 + j;
        e.stW(.d, (2 * kw) * e.u(3) + rp, (w[0][j] & 0xFFFF) | (w[1][j] << 16));
        e.stW(.d, (2 * kw + 1) * e.u(3) + rp, (w[0][j] >> 16) | (w[1][j] & 0xFFFF0000));
    }
}

// ---- bias, groupnorm apply, VAE norm ---------------------------------------

/// After a column-padded GEMM: d[u3 + i] = f0*a[(i/u1)*u2 + i%u1] + b[u4 + i%u1],
/// d in `dt`. f0 is the GEMM output scale and must be 1.0 for the plain case.
/// u0 = rows*u1, u1 = co, u2 = padded row stride, u3 = dst offset, u4 = bias offset.
pub inline fn biasCompact(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const cc = i % e.u(1);
    e.stT(.d, dt, e.u(3) + i, e.ld(.a, (i / e.u(1)) * e.u(2) + cc) * e.f(0) + e.ld(.b, e.u(4) + cc));
}

/// `biasCompact` accumulating, bias-free: d[i] += f0*a[(i/u1)*u2 + i%u1], d in `dt`.
/// u0 = rows*u1.
pub inline fn compactAcc(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.d, dt, i, e.ldT(.d, dt, i) + e.f(0) * e.ld(.a, (i / e.u(1)) * e.u(2) + i % e.u(1)));
}

/// c[u2 + i] = a[i] + b[i % u1], a in `in` and c in `out`: the cuDNN conv output
/// plus its bias. u0 = n*co, u1 = co.
pub inline fn biasAdd(e: Env, comptime in: Dt, comptime out: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.c, out, e.u(2) + i, e.ldT(.a, in, i) + e.ld(.b, i % e.u(1)));
}

/// a[p][c] += b[u3 + c], a in `dt`. u0 = n*ch, u1 = ch.
pub inline fn addBiasRows(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) + e.ld(.b, e.u(3) + i % e.u(1)));
}

/// GroupNorm apply: b = (a - mean[g])*inv[g]*w[ch] + bias[ch], then silu when u5.
/// c = weight ++ bias (bias at u4), d = mean ++ inv (inv at u3). u0 = n*ch,
/// u1 = ch, u2 = per_group, u3 = groups, u4 = bias offset, u5 = silu. a and b `dt`.
pub inline fn gnApply(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const ch = i % e.u(1);
    const g = ch / e.u(2);
    var v = (e.ldT(.a, dt, i) - e.ld(.d, g)) * e.ld(.d, e.u(3) + g) * e.ld(.c, ch) + e.ld(.c, e.u(4) + ch);
    if (e.u(5) != 0) v = k.silu(v);
    e.stT(.b, dt, i, v);
}

// ---- im2col -----------------------------------------------------------------

/// Patch matrix of a zero-padded 3x3 conv over channel-last [h*w][ci], for a band
/// of output positions; f0 != 0 reads a fused nearest-exact 2x upsample. a = src,
/// d = patches. u0 = bn*u1, u1 = 9*ci, u2 = ci, u3 = src w, u4 = src h,
/// u5 = first output position.
pub inline fn im2col(e: Env) void {
    const i = k.elem(e) orelse return;
    const plen = e.u(1);
    const ci = e.u(2);
    const up: u5 = if (e.f(0) != 0) 1 else 0;
    const ow = e.u(3) << up;
    const oh = e.u(4) << up;
    const col = i % plen;
    const p = e.u(5) + i / plen;
    const tap = col / ci;
    const yk = p / ow + tap / 3;
    const xk = p % ow + tap % 3;
    var v: f32 = 0;
    if (yk >= 1 and yk <= oh and xk >= 1 and xk <= ow) {
        v = e.ld(.a, (((yk - 1) >> up) * e.u(3) + ((xk - 1) >> up)) * ci + col % ci);
    }
    e.st(.d, i, v);
}

/// The SD family's 3x3 patch matrix, with stride-2 convolutions. a = src (`dt`), d =
/// patches (f32). u0 = bn*u1, u1 = 9*ci, u2 = ci, u3 = src w, u4 = grid extent (rows),
/// u5 = band start, u6 = OUT width, f0 = 0 stride 1 / 1 fused 2x upsample / 2
/// stride 2.
pub inline fn im2colSd(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const plen = e.u(1);
    const ci = e.u(2);
    const sw = e.u(3);
    const up: u5 = if (e.f(0) == 1) 1 else 0;
    const stride: u32 = if (e.f(0) == 2) 2 else 1;
    const ow = e.u(6);
    // The grid the taps are addressed in decides where the conv pads: when
    // upsampling it is the OUTPUT grid, which may be 2s-1, not 2s, so u4 carries
    // the extent rather than the source height.
    const gw = if (up == 1) ow else sw;
    const gh = e.u(4);
    const col = i % plen;
    const p = e.u(5) + i / plen;
    const tap = col / ci;
    const yk = (p / ow) * stride + tap / 3;
    const xk = (p % ow) * stride + tap % 3;
    var v: f32 = 0;
    if (yk >= 1 and yk <= gh and xk >= 1 and xk <= gw) {
        const src = (((yk - 1) >> up) * sw + ((xk - 1) >> up)) * ci + col % ci;
        v = e.ldT(.a, dt, src);
    }
    e.st(.d, i, v);
}

/// `im2colSd` written straight into a coop GEMM operand `[m_pad][k_pad]`, f16 for
/// `im2colH16` and bf16 for `im2colBf16`, scaled by f1, zeros in both pads, eight
/// elements (four words) a thread. a = src (`dt`), d = operand. u0 = threads, u1 = 9*ci,
/// u2 = ci | k_pad << 16, u3 = src w | grid extent << 16, u4 = OUT width, u5 = band
/// start, u6 = band rows; f0 as `im2colSd`. k_pad is a multiple of 8.
pub inline fn im2colH16(e: Env, comptime dt: Dt) void {
    im2colTo(e, dt, false);
}

pub inline fn im2colBf16(e: Env, comptime dt: Dt) void {
    im2colTo(e, dt, true);
}

inline fn im2colTo(e: Env, comptime dt: Dt, comptime bf: bool) void {
    const t = k.elem(e) orelse return;
    const plen = e.u(1);
    const ci = e.u(2) & 0xffff;
    const kp = e.u(2) >> 16;
    const e0 = t * 8;
    const row = e0 / kp;
    const col0 = e0 % kp;
    const w0 = t * 4;
    var words: [4]u32 = @splat(0);
    if (row < e.u(6) and col0 < plen) {
        const p = e.u(5) + row;
        if (ci % 8 == 0) {
            // All eight from one tap: one address, and a straight word copy when the
            // source is already the operand's type.
            if (im2colSrc(e, p, col0 / ci)) |base| {
                const at = base * ci + col0 % ci;
                if (comptime dt == (if (bf) Dt.bf16 else Dt.f16)) {
                    if (e.f(1) == 1.0) {
                        inline for (0..4) |q| words[q] = e.ldW(.a, at / 2 + @as(u32, q));
                        inline for (0..4) |q| e.stW(.d, w0 + @as(u32, q), words[q]);
                        return;
                    }
                }
                inline for (0..8) |j| {
                    const v = e.ldT(.a, dt, at + @as(u32, j)) * e.f(1);
                    words[j / 2] |= @as(u32, if (bf) k.bf16Bits(v) else k.f16Bits(v)) << @intCast(16 * (j % 2));
                }
            }
        } else {
            inline for (0..8) |j_| {
                const j: u32 = @intCast(j_);
                const col = col0 + j;
                if (col < plen) {
                    if (im2colSrc(e, p, col / ci)) |base| {
                        const v = e.ldT(.a, dt, base * ci + col % ci) * e.f(1);
                        words[j_ / 2] |= @as(u32, if (bf) k.bf16Bits(v) else k.f16Bits(v)) << @intCast(16 * (j_ % 2));
                    }
                }
            }
        }
    }
    inline for (0..4) |q| e.stW(.d, w0 + @as(u32, q), words[q]);
}

/// The source pixel output position `p` reads at `tap`, or null in the conv's padding.
inline fn im2colSrc(e: Env, p: u32, tap: u32) ?u32 {
    const sw = e.u(3) & 0xffff;
    const gh = e.u(3) >> 16;
    const ow = e.u(4);
    const up: u5 = if (e.f(0) == 1) 1 else 0;
    const stride: u32 = if (e.f(0) == 2) 2 else 1;
    const gw = if (up == 1) ow else sw;
    const yk = (p / ow) * stride + tap / 3;
    const xk = (p % ow) * stride + tap % 3;
    if (yk < 1 or yk > gh or xk < 1 or xk > gw) return null;
    return ((yk - 1) >> up) * sw + ((xk - 1) >> up);
}

// ---- MoE / row routing ------------------------------------------------------

/// b[i] = a[c[i / u1] * u1 + i % u1]: gather rows by u32 id. u0 = rows*u1, u1 = width.
pub inline fn gatherRows(e: Env) void {
    const i = k.elem(e) orelse return;
    const w = e.u(1);
    e.st(.b, i, e.ld(.a, e.ldW(.c, i / w) * w + i % w));
}

/// a[c[r] * u1 + j] += d[r] * b[r * u1 + j]; ids of one launch must be distinct.
/// u0 = rows*u1, u1 = width.
pub inline fn scatterAddRows(e: Env) void {
    const i = k.elem(e) orelse return;
    const w = e.u(1);
    const r = i / w;
    const dst = e.ldW(.c, r) * w + i % w;
    e.st(.a, dst, e.ld(.a, dst) + e.ld(.b, i) * e.ld(.d, r));
}

/// a[t][j] = sum over the token's u2 route rows r = c[t*u2 + k] of d[r] * b[r][j],
/// in slot order. u0 = tokens*u1, u1 = width, u2 = used.
pub inline fn moeCombine(e: Env) void {
    const i = k.elem(e) orelse return;
    const w = e.u(1);
    const t = i / w;
    const j = i % w;
    var acc: f32 = 0;
    var s: u32 = 0;
    while (s < e.u(2)) : (s += 1) {
        const r = e.ldW(.c, t * e.u(2) + s);
        acc += e.ld(.b, r * w + j) * e.ld(.d, r);
    }
    e.st(.a, i, acc);
}

const inf_f32: f32 = @bitCast(@as(u32, 0x7F800000));

/// MoE routing, one subgroup per token t < u0: gate its row of the logits (a, u1 experts
/// wide; sigmoid when u3 is 0, softmax when 1), take the u2 best by gate plus bias (b),
/// highest first and the lower index on a tie, and write their ids as words to
/// c[u4 + t * u2 ..] and their weights to c[u5 + t * u2 ..]: the gates, divided by their
/// sum (at least 2^-14) when u6 is 1, times f0. That is `k2_horizon.route`'s selection,
/// one pass per pick, each the best strictly below the last; the weights match it to
/// rounding.
pub inline fn moeRoute(e: Env) void {
    const n = e.u(1);
    const used = e.u(2);
    const softmax = e.u(3) == 1;
    const lane = e.sgLane();
    const lanes = e.sgSize();
    var t = k.sgGlobal(e);
    while (t < e.u(0)) : (t += k.sgTotal(e)) {
        const row = t * n;
        var mx: f32 = -inf_f32;
        var total: f32 = 0;
        if (softmax) {
            var x = lane;
            while (x < n) : (x += lanes) mx = @max(mx, e.ld(.a, row + x));
            mx = k.sgMax(mx);
            x = lane;
            while (x < n) : (x += lanes) total += k.exp(e.ld(.a, row + x) - mx);
            total = k.sgSum(total);
        }
        var last_score: f32 = inf_f32;
        var last: f32 = -1;
        var sum: f32 = 0;
        var j: u32 = 0;
        while (j < used) : (j += 1) {
            var best_score: f32 = -inf_f32;
            var best: f32 = 3.0e38;
            var best_gate: f32 = 0;
            var x = lane;
            while (x < n) : (x += lanes) {
                const v = e.ld(.a, row + x);
                const gate = if (softmax) k.exp(v - mx) / total else k.sigmoid(v);
                const score = gate + e.ld(.b, x);
                const xf: f32 = @floatFromInt(x);
                const below = j == 0 or score < last_score or (score == last_score and xf > last);
                if (below and score > best_score) {
                    best_score = score;
                    best = xf;
                    best_gate = gate;
                }
            }
            const top = k.sgMax(best_score);
            // The lowest index among the lanes holding the top score, as a max of negations.
            const pick = -k.sgMax(if (best_score == top) -best else -3.0e38);
            const gate = k.sgSum(if (best_score == top and best == pick) best_gate else 0);
            last_score = top;
            last = pick;
            sum += gate;
            if (lane == 0) {
                e.stW(.c, e.u(4) + t * used + j, if (top == -inf_f32) n else @intFromFloat(pick));
                e.st(.c, e.u(5) + t * used + j, gate);
            }
        }
        if (lane == 0) {
            const div = if (e.u(6) == 1) @max(sum, 6.103515625e-5) else 1;
            j = 0;
            while (j < used) : (j += 1) {
                const at = e.u(5) + t * used + j;
                e.st(.c, at, e.ld(.c, at) / div * e.f(0));
            }
        }
    }
}

// ---- k-split GEMV reduce halves ----------------------------------------------

/// d[u2 + col] = f0 * sum over u1 chunks of a[ch*u0 + col]. u0 = rows.
pub inline fn gemvCombine(e: Env) void {
    const col = k.elem(e) orelse return;
    var sum: f32 = 0;
    var ch: u32 = 0;
    while (ch < e.u(1)) : (ch += 1) sum += e.ld(.a, ch * e.u(0) + col);
    e.st(.d, e.u(2) + col, sum * e.f(0));
}

/// `gemvCombine` for u4 inputs: d[u2 + i*u3 + col] = f0 * sum_ch a[(ch*4 + i)*rows
/// + col]. u0 = rows, u1 = chunks, u3 = dest row stride, u4 = inputs (1..4).
pub inline fn gemvCombine4(e: Env) void {
    const i = e.gid();
    const rows = e.u(0);
    if (i >= e.u(4) * rows) return;
    const inp = i / rows;
    const col = i % rows;
    var sum: f32 = 0;
    var ch: u32 = 0;
    while (ch < e.u(1)) : (ch += 1) sum += e.ld(.a, (ch * 4 + inp) * rows + col);
    e.st(.d, e.u(2) + inp * e.u(3) + col, sum * e.f(0));
}

// ---- M-RoPE and vision RoPE ---------------------------------------------------

/// ggml's imrope channel for pair `p`: 1 (h) when p%3==1 and p < 3*s1, 2 (w) when
/// p%3==2 and p < 3*s2, else 0 (t). `sections` packs s0 | s1<<8 | s2<<16.
inline fn imropeChannel(p: u32, sections: u32) u32 {
    const s1 = (sections >> 8) & 255;
    const s2 = (sections >> 16) & 255;
    return switch (p % 3) {
        1 => if (p < 3 * s1) @as(u32, 1) else 0,
        2 => if (p < 3 * s2) @as(u32, 2) else 0,
        else => 0,
    };
}

/// Interleaved M-RoPE for one row: the position per pair comes from b = pos3
/// (u32 t, h, w). a = qk, c = freqs. u0 = n_heads*half, u1 = half, u2 = sin_off,
/// u3 = n_heads, u4 = sections, u5 = head_dim.
pub inline fn ropeImrope(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const pos = e.ldW(.b, imropeChannel(p, e.u(4)));
    rotateHalf(e, (i / half) * e.u(5) + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

/// `ropeImrope` with per-row position triples, b = pos3s [rows][3].
/// u0 = rows*n_heads*half, u1..u5 as above.
pub inline fn ropeImropePos(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const row = i / (half * e.u(3));
    const pos = e.ldW(.b, row * 3 + imropeChannel(p, e.u(4)));
    rotateHalf(e, (i / half) * e.u(5) + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

/// Qwen3-VL 2-D vision rope: pairs (p, p + 2*half) for p < 2*half; the first half
/// keyed by the patch row, the next by its column, frequency index p % half.
/// a = qk, b = pos2 [rows][2] u32, c = freqs. u0 = rows*n_heads*2*half, u1 = half,
/// u2 = sin_off, u3 = n_heads, u4 = head_dim.
pub inline fn ropeVision(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const pairs = 2 * half;
    const p = i % pairs;
    const hp = i / pairs;
    const row = i / (pairs * e.u(3));
    const axis: u32 = if (p >= half) 1 else 0;
    const fi = if (p >= half) p - half else p;
    const pos = e.ldW(.b, row * 2 + axis);
    rotateHalf(e, hp * e.u(4) + p, pairs, e.ld(.c, pos * half + fi), e.ld(.c, e.u(2) + pos * half + fi));
}

/// gemma4v 2-D rope (neox): each head is two spans [0,2h) and [2h,4h); span 0
/// rotates against grid x (pos2[t][0]), span 1 against y, pairing (off+i, off+h+i)
/// within the span. Layout as `ropeVision`.
pub inline fn ropeVisionGemma4(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const pairs = 2 * half;
    const p = i % pairs;
    const hp = i / pairs;
    const row = hp / e.u(3);
    const span = p / half;
    const fi = p % half;
    const pos = e.ldW(.b, row * 2 + span);
    rotateHalf(e, hp * e.u(4) + span * pairs + fi, half, e.ld(.c, pos * half + fi), e.ld(.c, e.u(2) + pos * half + fi));
}

/// Rotate-half RoPE over a contiguous span `[u5, u5 + 2*half)` of each `u4`-wide
/// head, with one absolute position per ROW from b (u32).
///
/// SenseNova's trunk: a head splits 64 (sequence position, theta 5e6) + 32 (token
/// row) + 32 (token column), each rotated against its own table, so a span form
/// with its own positions is what the three calls need.
/// u0 = rows*n_heads*half, u1 = half, u2 = sin_off, u3 = n_heads, u4 = head_dim,
/// u5 = span offset within the head.
pub inline fn ropeHalfSpanPos(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const hp = i / half;
    const pos = e.ldW(.b, hp / e.u(3));
    rotateHalf(e, hp * e.u(4) + e.u(5) + p, half, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

inline fn rotateInter(e: Env, base: u32, cos_v: f32, sin_v: f32) void {
    const x0 = e.ld(.a, base);
    const x1 = e.ld(.a, base + 1);
    e.st(.a, base, x0 * cos_v - x1 * sin_v);
    e.st(.a, base + 1, x0 * sin_v + x1 * cos_v);
}

/// Interleaved-pair RoPE over a contiguous span `[u4, u4 + 2*half)` of each
/// `u3`-wide row, one position per row from b (u32). Pairs are (2i, 2i+1), NOT
/// (i, i+half): SenseNova's vision patch embedder rotates interleaved pairs while
/// its trunk rotates split halves, and the two are the same shape and different
/// arithmetic. u0 = rows*half, u1 = half, u2 = sin_off, u3 = row_dim, u4 = offset.
pub inline fn ropeInterSpanPos(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const p = i % half;
    const row = i / half;
    const pos = e.ldW(.b, row);
    rotateInter(e, row * e.u(3) + e.u(4) + 2 * p, e.ld(.c, pos * half + p), e.ld(.c, e.u(2) + pos * half + p));
}

/// `nn.PixelShuffle(r)` on a channel-last plane: `[h][w][c*r*r] -> [h*r][w*r][c]`,
/// the input channel split as `c * r * r + i * r + j`. a = src, b = dst.
/// u0 = total output elements, u1 = out channels, u2 = r, u3 = input width.
pub inline fn pixelShuffle(e: Env) void {
    const i = k.elem(e) orelse return;
    const c = e.u(1);
    const r = e.u(2);
    const w = e.u(3);
    const ow = w * r;
    const ch = i % c;
    const px = (i / c) % ow;
    const py = i / (c * ow);
    e.st(.b, i, e.ld(.a, ((py / r) * w + px / r) * (c * r * r) + ch * r * r + (py % r) * r + px % r));
}

/// `DupUp3D` at spatial factor 2, added: a `[2h][2w][u1]` (`dt`) += b, the `[h][w][u2]`
/// stage input with each channel repeated and pixel-shuffled, temporal phase `ft - 1`
/// kept (`wan_vae.DupUp.src`). u0 = total, u1 = co, u2 = ci, u3 = w, u4 = ft.
pub inline fn dupUpAdd(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const co = e.u(1);
    const ci = e.u(2);
    const w = e.u(3);
    const ft = e.u(4);
    const ch = i % co;
    const px = (i / co) % (2 * w);
    const py = i / (co * 2 * w);
    const f = ft * 4;
    const src = (ch * f + (ft - 1) * 4 + (py % 2) * 2 + px % 2) / (co * f / ci);
    const s = e.ldT(.b, dt, ((py / 2) * w + px / 2) * ci + src);
    e.stT(.a, dt, i, e.ldT(.a, dt, i) + s);
}

/// The int8 levels of a W6A8 byte pair: column `2j` from the low nibble of `byte` and
/// bits `sh..sh+1` of `hi`, column `2j + 1` from the high nibble and bits `sh+2..sh+3`,
/// each through the `[256][64]` table row of scale byte `s` (`ops.w4a8.Levels6`).
inline fn w6Pair(e: Env, byte: u32, hi: u32, sh: u32, s: u32) [2]u32 {
    const lo = (byte & 0xF) | (((hi >> @intCast(sh)) & 3) << 4);
    const up = (byte >> 4) | (((hi >> @intCast(sh + 2)) & 3) << 4);
    return .{ e.byte(.d, s * 64 + lo), e.byte(.d, s * 64 + up) };
}

/// A packed W6A8 weight to row-major int8, four columns a thread (the CUDA layout).
/// a = packed `[rows][3*cols/4]` bytes, b = int8 `[rows][cols]`, c = s_rel fp8 bytes
/// `[rows][cols/group]`, d = the level table. u0 = rows*cols/4, u1 = cols, u2 = group.
pub inline fn w6a8Decode(e: Env) void {
    const t = k.elem(e) orelse return;
    const cols = e.u(1);
    const quads = cols / 4;
    const row = t / quads;
    const q = t % quads;
    const base = row * (cols * 3 / 4);
    const hi = e.byte(.a, base + cols / 2 + q);
    const s = e.byte(.c, row * (cols / e.u(2)) + (4 * q) / e.u(2));
    const p0 = w6Pair(e, e.byte(.a, base + 2 * q), hi, 0, s);
    const p1 = w6Pair(e, e.byte(.a, base + 2 * q + 1), hi, 4, s);
    e.stW(.b, t, p0[0] | (p0[1] << 8) | (p1[0] << 16) | (p1[1] << 24));
}

/// `w6a8Decode` into the k-major layout Vulkan's int8 GEMM reads, from inputs the
/// weight upload already byte-transposed (see `w4a8_decode_t` in eltwise.zig, whose
/// layout this keeps): a = packed `[3*cols/4][stride]`, c = s_rel `[cols/group][stride]`,
/// b = int8 `[cols][stride]`. A thread owns one nibble byte-column `j` for four rows.
/// u0 = (cols/2)*(stride/4), u1 = group/2, u2 = stride/4, u3 = cols/2.
pub inline fn w6a8DecodeT(e: Env) void {
    const t = k.elem(e) orelse return;
    const rq = e.u(2);
    const j = t / rq;
    const r = t % rq;
    const pw = e.ldW(.a, t);
    const hw = e.ldW(.a, (e.u(3) + j / 2) * rq + r);
    const sw = e.ldW(.c, (j / e.u(1)) * rq + r);
    const sh = 4 * (j % 2);
    var lo_out: u32 = 0;
    var hi_out: u32 = 0;
    inline for (0..4) |i| {
        const v = w6Pair(e, (pw >> (8 * i)) & 0xFF, (hw >> (8 * i)) & 0xFF, sh, (sw >> (8 * i)) & 0xFF);
        lo_out |= v[0] << (8 * i);
        hi_out |= v[1] << (8 * i);
    }
    e.stW(.b, 2 * j * rq + r, lo_out);
    e.stW(.b, (2 * j + 1) * rq + r, hi_out);
}

/// Patch matrix of a NON-OVERLAPPING k x k stride-k convolution over channel-last
/// `[h][w][ci]`: `b[oy*ow + ox][(dy*k + dx)*ci + c] = a[(oy*k + dy)*w + ox*k + dx][c]`.
/// No padding, because a stride-k window never leaves the input. a = src, b = dst.
/// u0 = total, u1 = ci, u2 = k, u3 = input width.
pub inline fn im2colStride(e: Env) void {
    const i = k.elem(e) orelse return;
    const ci = e.u(1);
    const kk = e.u(2);
    const w = e.u(3);
    const plen = kk * kk * ci;
    const col = i % plen;
    const tok = i / plen;
    const ow = w / kk;
    const tap = col / ci;
    e.st(.b, i, e.ld(.a, (((tok / ow) * kk + tap / kk) * w + (tok % ow) * kk + tap % kk) * ci + col % ci));
}

/// Patch matrix of a 3-D conv over a channel-last `[t][h][w][ci]` volume, columns
/// ordered (kt, kh, kw, ci). Temporal padding is CAUSAL (front only, zero); spatial
/// padding REFLECTS, and a width of 1 collapses to index 0 rather than reflecting.
///
/// a = src, b = patch, c = shape params as u32:
/// 0 cols, 1 in_ch, 2 kt, 3 kh, 4 kw, 5 out_h, 6 out_w, 7 x_t, 8 x_h, 9 x_w,
/// 10 stride_t, 11 stride_s, 12 front, 13 pad_h_lo, 14 pad_w_lo.
/// u0 = rows*cols of this band, f0 = the band's first output row.
///
/// Those params are the same for every band of a volume, so `c` is uploaded once and
/// only `f0` moves; a per-band params slot would need a buffer offset, which a Vulkan
/// `DeviceBuffer` has no room for.
pub inline fn im2col3d(e: Env) void {
    const i = k.elem(e) orelse return;
    const cols = e.ldW(.c, 0);
    const ci = e.ldW(.c, 1);
    const x_h: i32 = @intCast(e.ldW(.c, 8));
    const x_w: i32 = @intCast(e.ldW(.c, 9));

    // Output row of the whole volume, unpacked to (ot, oh, ow).
    const row = i / cols + @as(u32, @intFromFloat(e.f(0)));
    const out_w = e.ldW(.c, 6);
    const out_h = e.ldW(.c, 5);
    const ow = row % out_w;
    const oh = (row / out_w) % out_h;
    const ot = (row / out_w) / out_h;

    // Column, unpacked to (kt, kh, kw, ic).
    const col = i % cols;
    const ic = col % ci;
    const kw = e.ldW(.c, 4);
    const kh = e.ldW(.c, 3);
    const w_i = (col / ci) % kw;
    const h_i = ((col / ci) / kw) % kh;
    const t_i = ((col / ci) / kw) / kh;

    var v: f32 = 0;
    const p_t = @as(i32, @intCast(ot * e.ldW(.c, 10) + t_i)) - @as(i32, @intCast(e.ldW(.c, 12)));
    if (p_t >= 0 and p_t < @as(i32, @intCast(e.ldW(.c, 7)))) {
        const stride_s = e.ldW(.c, 11);
        const sh = reflect(@as(i32, @intCast(oh * stride_s + h_i)) - @as(i32, @intCast(e.ldW(.c, 13))), x_h);
        const sw = reflect(@as(i32, @intCast(ow * stride_s + w_i)) - @as(i32, @intCast(e.ldW(.c, 14))), x_w);
        v = e.ld(.a, @intCast(((p_t * x_h + sh) * x_w + sw) * @as(i32, @intCast(ci)) + @as(i32, @intCast(ic))));
    }
    e.st(.b, i, v);
}

/// `torch` reflect padding: fold about 0 and about `n - 1` until inside. Bounded
/// rather than `while (true)`: a pad smaller than the extent converges in two folds
/// and SPIR-V would rather not be handed an unbounded loop.
inline fn reflect(v0: i32, n: i32) i32 {
    if (n == 1) return 0;
    var v = v0;
    for (0..8) |_| {
        if (v < 0) v = -v;
        if (v >= n) v = 2 * (n - 1) - v else break;
    }
    return v;
}

// ---- MiniMax H3 audio (channel-last 1-D) ---------------------------------------

/// Patch matrix of a 1-D conv over channel-last [len][ci], columns ordered
/// (tap, ci) so a warp's loads coalesce; out of range is zero. a = src, b = patch.
/// u0 = out_len*plen, u1 = plen (k*ci), u2 = ci, u3 = in_len, u4 = dilation,
/// u5 = padding, f0 = stride, f1 = first output row of the band.
pub inline fn im2col1d(e: Env) void {
    const i = k.elem(e) orelse return;
    const plen = e.u(1);
    const ci = e.u(2);
    const col = i % plen;
    const stride: i32 = @intFromFloat(e.f(0));
    const t: i32 = @as(i32, @intCast(i / plen)) + @as(i32, @intFromFloat(e.f(1)));
    const s: i32 = t * stride - @as(i32, @intCast(e.u(5))) + @as(i32, @intCast((col / ci) * e.u(4)));
    var v: f32 = 0;
    if (s >= 0 and s < @as(i32, @intCast(e.u(3)))) v = e.ld(.a, @as(u32, @intCast(s)) * ci + col % ci);
    e.st(.b, i, v);
}

/// Anti-aliased 2x upsample + SnakeBeta as one gather: output t reads the transposed
/// conv's position P = t + pad_left, taps j == P (mod 2), replicate padding as an
/// index clamp, then x += sin(x*ea)^2 * ib with (ea, ib) = (exp(alpha), 1/(exp(beta)
/// + 1e-9)) interleaved per channel in d. a = src [len][ch], b = out [2*len][ch],
/// c = filter [k]. u0 = 2*len*ch, u1 = ch, u2 = len, u3 = k, u4 = pad, u5 = pad_left.
pub inline fn aaUpSnake(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(1);
    const len: i32 = @intCast(e.u(2));
    const kk: i32 = @intCast(e.u(3));
    const pad: i32 = @intCast(e.u(4));
    const c = i % ch;
    const P: i32 = @as(i32, @intCast(i / ch)) + @as(i32, @intCast(e.u(5)));
    const padded_len = len + 2 * pad;
    var acc: f32 = 0;
    var j: i32 = P & 1;
    while (j < kk) : (j += 2) {
        const pj = P - j;
        if (pj < 0) continue;
        const sidx = pj >> 1;
        if (sidx >= padded_len) continue;
        const sp = @max(0, @min(sidx - pad, len - 1));
        acc += e.ld(.a, @as(u32, @intCast(sp)) * ch + c) * e.ld(.c, @as(u32, @intCast(j)));
    }
    acc += acc;
    const ea = e.ld(.d, 2 * c);
    const ib = e.ld(.d, 2 * c + 1);
    const sn = k.sin(acc * ea);
    e.st(.b, i, acc + sn * sn * ib);
}

/// The matching downsample: replicate-pad by pad_left and kaiser-sinc filter x2.
/// a = up [up_len][ch], b = out, c = filter [k]. u0 = out_len*ch, u1 = ch,
/// u2 = up_len, u3 = k, u4 = pad_left.
pub inline fn aaDown(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(1);
    const c = i % ch;
    const base: i32 = 2 * @as(i32, @intCast(i / ch)) - @as(i32, @intCast(e.u(4)));
    const last: i32 = @as(i32, @intCast(e.u(2))) - 1;
    var acc: f32 = 0;
    var j: u32 = 0;
    while (j < e.u(3)) : (j += 1) {
        const pidx = @max(0, @min(base + @as(i32, @intCast(j)), last));
        acc += e.ld(.a, @as(u32, @intCast(pidx)) * ch + c) * e.ld(.c, j);
    }
    e.st(.b, i, acc);
}

/// Transposed 1-D conv as a gather: output t reads P = t + pad, taps j == P (mod
/// stride), over in_ch; w is permuted to [k][in_ch][out_ch]. a = x [in_len][in_ch],
/// b = out [out_len][out_ch], c = w, d = bias [out_ch]. u0 = out_len*out_ch,
/// u1 = out_ch, u2 = in_ch, u3 = in_len, u4 = k, u5 = stride, f0 = padding.
pub inline fn convt1dCa(e: Env) void {
    const i = k.elem(e) orelse return;
    const out_ch = e.u(1);
    const in_ch = e.u(2);
    const stride = e.u(5);
    const oc = i % out_ch;
    const P: i32 = @as(i32, @intCast(i / out_ch)) + @as(i32, @intFromFloat(e.f(0)));
    var acc = e.ld(.d, oc);
    var j: u32 = @as(u32, @intCast(P)) % stride;
    while (j < e.u(4)) : (j += stride) {
        const pj = P - @as(i32, @intCast(j));
        if (pj < 0) continue;
        const sidx = @as(u32, @intCast(pj)) / stride;
        if (sidx >= e.u(3)) continue;
        var ic: u32 = 0;
        while (ic < in_ch) : (ic += 1) {
            acc += e.ld(.c, (j * in_ch + ic) * out_ch + oc) * e.ld(.a, sidx * in_ch + ic);
        }
    }
    e.st(.b, i, acc);
}

/// The encoder's Snake1d in place: x += sin(a*x)^2 / (a + 1e-9), a = alpha[ch]
/// as both parameters. a = x, b = alpha. u0 = n, u1 = ch.
pub inline fn snake1dCa(e: Env) void {
    const i = k.elem(e) orelse return;
    const x = e.ld(.a, i);
    const al = e.ld(.b, i % e.u(1));
    const sn = k.sin(al * x);
    e.st(.a, i, x + sn * sn / (al + 1e-9));
}

/// Mean over heads, then adaptive average pool along the feature axis to out_dim:
/// bin o of row r averages a[r][h][i] over every h and i in [floor(o*hd/out),
/// ceil((o+1)*hd/out)). a = src [rows][heads*hd], b = out [rows][out_dim].
/// u0 = rows*out_dim, u1 = out_dim, u2 = heads, u3 = hd.
pub inline fn meanHeadsPool(e: Env) void {
    const i = k.elem(e) orelse return;
    const out = e.u(1);
    const heads = e.u(2);
    const hd = e.u(3);
    const r = i / out;
    const o = i % out;
    const start = o * hd / out;
    const end = ((o + 1) * hd + out - 1) / out;
    var acc: f32 = 0;
    var h: u32 = 0;
    while (h < heads) : (h += 1) {
        var j = start;
        while (j < end) : (j += 1) acc += e.ld(.a, (r * heads + h) * hd + j);
    }
    e.st(.b, i, acc / @as(f32, @floatFromInt((end - start) * heads)));
}

// ---- attention softmax partials and int8 row scales (Vulkan DiT paths) -------

/// Online-softmax partial over an interleaved slice of f16-pair WORDS of a scores
/// row. a = S (f16 pairs), d = partials [(z*rows + q)*nchunks + chunk] x {m, d}.
/// u0 = threads, u1 = nchunks, u2 = valid cols, u3 = S row stride, u4 = rows per
/// plane (0 for u2, square), u5 = S plane stride (elements).
pub inline fn softmaxPartial(e: Env) void {
    const i = k.elem(e) orelse return;
    const nch = e.u(1);
    const chunk = i % nch;
    const qz = i / nch;
    const rows = if (e.u(4) != 0) e.u(4) else e.u(2);
    const q = qz % rows;
    const z = qz / rows;
    const base_w = (z * e.u(5) + q * e.u(3)) / 2;
    const nw = (e.u(2) + 1) / 2;
    var m: f32 = -3.4e38;
    var dsum: f32 = 0;
    var wi = chunk;
    while (wi < nw) : (wi += nch) {
        const word = e.ldW(.a, base_w + wi);
        inline for (0..2) |h_| {
            const h: u32 = @intCast(h_);
            const j = wi * 2 + h;
            if (j < e.u(2)) {
                const sc = k.h16(word >> (16 * h));
                const mn = @max(m, sc);
                dsum = dsum * k.exp(m - mn) + k.exp(sc - mn);
                m = mn;
            }
        }
    }
    e.st(.d, i * 2, m);
    e.st(.d, i * 2 + 1, dsum);
}

/// Fold the chunk partials of one (z, q) into the row max and reciprocal sum.
/// a = partials, d = md [z][rows_pad] x {m, 1/d}. u0 = z*rows, u1 = nchunks,
/// u2 = rows, u3 = rows_pad.
pub inline fn softmaxCombine(e: Env) void {
    const i = k.elem(e) orelse return;
    const q = i % e.u(2);
    const z = i / e.u(2);
    const pbase = i * e.u(1) * 2;
    var m: f32 = -3.4e38;
    var c: u32 = 0;
    while (c < e.u(1)) : (c += 1) m = @max(m, e.ld(.a, pbase + c * 2));
    var dsum: f32 = 0;
    c = 0;
    while (c < e.u(1)) : (c += 1) dsum += e.ld(.a, pbase + c * 2 + 1) * k.exp(e.ld(.a, pbase + c * 2) - m);
    e.st(.d, (z * e.u(3) + q) * 2, m);
    e.st(.d, (z * e.u(3) + q) * 2 + 1, 1.0 / dsum);
}

/// Per-row int8 scale from the per-group partial abs-maxes: b[row] = max(max_g a[row*ng
/// + g] / 127, 1e-12). u0 = rows, u1 = ng.
pub inline fn rowscaleI8(e: Env) void {
    const row = k.elem(e) orelse return;
    var amax: f32 = 0;
    var g: u32 = 0;
    while (g < e.u(1)) : (g += 1) amax = @max(amax, e.ld(.a, row * e.u(1) + g));
    e.st(.b, row, @max(amax / 127.0, 1e-12));
}

/// Merge a group's chunk statistics (Chan) into d[g] = mean and d[groups + g] = inv.
/// a = stats [groups*u2][3]. u0 = groups, u2 = chunks, f0 = eps. Chunks with
/// count 0 (a group shorter than its chunk count) merge as identities.
pub inline fn gnCombine(e: Env) void {
    const g = k.elem(e) orelse return;
    const nch = e.u(2);
    var count: f32 = 0;
    var mean: f32 = 0;
    var m2: f32 = 0;
    var c: u32 = 0;
    while (c < nch) : (c += 1) {
        const base = (g * nch + c) * 3;
        const cb = e.ld(.a, base);
        if (cb == 0) continue;
        const mb = e.ld(.a, base + 1);
        const m2b = e.ld(.a, base + 2);
        const n = count + cb;
        const delta = mb - mean;
        mean += delta * cb / n;
        m2 += m2b + delta * delta * count * cb / n;
        count = n;
    }
    e.st(.d, g, mean);
    e.st(.d, e.u(0) + g, 1.0 / @sqrt(m2 / count + e.f(0)));
}

/// Fused per-head rmsnorm + interleaved rope + output scale, in place, one thread per
/// (pos, head) row. a = x [rows][2*half] (`dt`), b = norm weight, c = freqs.
/// u0 = rows, u1 = half, u2 = sin_off, u3 = n_heads, f0 = scale, f1 = eps.
pub inline fn qknormRope(e: Env, comptime dt: Dt) void {
    const row = k.elem(e) orelse return;
    const half = e.u(1);
    const pos = row / e.u(3);
    const base = row * half * 2;
    var sum: f32 = 0;
    var w: u32 = 0;
    while (w < half * 2) : (w += 1) {
        const v = e.ldT(.a, dt, base + w);
        sum += v * v;
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(half * 2)) + e.f(1));
    w = 0;
    while (w < half) : (w += 1) {
        const x0 = e.ldT(.a, dt, base + w * 2) * inv * e.ld(.b, w * 2);
        const x1 = e.ldT(.a, dt, base + w * 2 + 1) * inv * e.ld(.b, w * 2 + 1);
        const cos_v = e.ld(.c, pos * half + w);
        const sin_v = e.ld(.c, e.u(2) + pos * half + w);
        e.stT(.a, dt, base + w * 2, (x0 * cos_v - x1 * sin_v) * e.f(0));
        e.stT(.a, dt, base + w * 2 + 1, (x0 * sin_v + x1 * cos_v) * e.f(0));
    }
}

// ---- Mage-VAE ----------------------------------------------------------------

/// Depthwise 3x3, stride 1, pad 1, over a channel-last `[h][w][ch]` activation:
/// `a[y][x][c] = Σ b[y+ky-1][x+kx-1][c] * c_[c][ky][kx] + d[c]`, zero outside.
///
/// Not an im2col GEMM, because each output channel reads only its own input
/// channel: the patch matrix a GEMM needs would be block-diagonal and almost all
/// zero. The innermost index is the CHANNEL, so a warp covers consecutive memory.
/// u0 = h*w*ch, u1 = h, u2 = w, u3 = ch. a and b `dt`.
pub inline fn dwConv3(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(3);
    const w = e.u(2);
    const h = e.u(1);
    const c = i % ch;
    const x = (i / ch) % w;
    const y = i / (ch * w);
    var acc = e.ld(.d, c);
    var ky: u32 = 0;
    while (ky < 3) : (ky += 1) {
        if (y + ky < 1 or y + ky - 1 >= h) continue;
        const sy = y + ky - 1;
        var kx: u32 = 0;
        while (kx < 3) : (kx += 1) {
            if (x + kx < 1 or x + kx - 1 >= w) continue;
            const sx = x + kx - 1;
            acc += e.ldT(.b, dt, (sy * w + sx) * ch + c) * e.ld(.c, (c * 3 + ky) * 3 + kx);
        }
    }
    e.stT(.a, dt, i, acc);
}

/// Column sums of a `[u1][u3]` activation over row chunks of `u2`, times f0: thread i
/// takes column `i % u3` of chunk `i / u3`, so a warp reads a row contiguously. a =
/// `[chunks][u3]` f32, b `dt`. u0 = chunks * u3. Run twice, a mean over many rows is a
/// pass over chunks and one over the partials (`AdaptiveAvgPool2d(1)` in Mage-VAE).
pub inline fn colMean(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const dim = e.u(3);
    const c = i % dim;
    const r0 = (i / dim) * e.u(2);
    const r1 = @min(r0 + e.u(2), e.u(1));
    var acc: f32 = 0;
    var r = r0;
    while (r < r1) : (r += 1) acc += e.ldT(.b, dt, r * dim + c);
    e.st(.a, i, acc * e.f(0));
}

/// `a[i] *= sigmoid(b[i % u1])`: a per-column gate broadcast over rows, which is
/// the second half of the channel attention. u0 = total, u1 = dim. a `dt`, b f32.
pub inline fn mulColsSigmoid(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) * k.sigmoid(e.ld(.b, i % e.u(1))));
}

/// Build the per-pixel MLP's input rows for Mage-VAE's decode:
///
///     a[(t * P + p) * W + f] = 0                        f < 3
///                            = b[t * H * P + (f-3) * P + p]   3 <= f < 3 + H
///                            = c[p * D + (f - 3 - H)]         otherwise
///
/// The first three are the image pixel, which decode's zero noise makes zero; the
/// next `H` are that patch's own channels for this pixel; the rest are the fixed
/// cosine position table. u0 = total, u1 = W (3+H+D), u2 = P, u3 = H, u4 = D.
/// a and b `dt`, c f32.
pub inline fn nerfFeat(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const wide = e.u(1);
    const p_area = e.u(2);
    const hx = e.u(3);
    const dct = e.u(4);
    const f = i % wide;
    const row = i / wide;
    const p = row % p_area;
    const t = row / p_area;
    if (f < 3) {
        e.stT(.a, dt, i, 0);
    } else if (f < 3 + hx) {
        e.stT(.a, dt, i, e.ldT(.b, dt, t * hx * p_area + (f - 3) * p_area + p));
    } else {
        e.stT(.a, dt, i, e.ld(.c, p * dct + (f - 3 - hx)));
    }
}

/// LoKr's cross-group mix: `a[p * u4 + u5 + i * u2 + q] += f0 * sum_j c[i * u3 + j] *
/// b[p * f1 + j * u6 + q]` over `i < u1`, `q < u2`, `j < u3`, `a` at `dt` and `b`, `c` f32.
/// `f1` is the per-row stride of `b` as a u32's bits: `[rows][im][ok]` for a linear,
/// `ok` beside a group stride `u6` for a conv's stacked groups. With `u1 = u3 = 1` and
/// `c = [1]` it is a strided scaled add. u0 = rows * u1 * u2.
pub inline fn kronMix(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const ok = e.u(2);
    const ol = e.u(1);
    const im = e.u(3);
    const q = i % ok;
    const row = (i / ok) % ol;
    const p = i / (ok * ol);
    const hp: u32 = @bitCast(e.f(1));
    var acc: f32 = 0;
    var j: u32 = 0;
    while (j < im) : (j += 1) acc += e.ld(.c, row * im + j) * e.ld(.b, p * hp + j * e.u(6) + q);
    const o = p * e.u(4) + e.u(5) + row * ok + q;
    e.stT(.a, dt, o, e.ldT(.a, dt, o) + e.f(0) * acc);
}

/// LoKr's channel groups stacked as images with one zero row between: `a` is
/// `[u5][u2 + 1][u1][u3]`, group `j`'s row `y` the source's channels
/// `[(u6 + j) * u3, +u3)` of row `y` (`b` `[u2][u1][u4]`), row `u2` zero. A 3x3 conv
/// with padding 1 over the stack is then each group's conv alone. Both `dt`.
/// u0 = u5 * (u2 + 1) * u1 * u3.
pub inline fn kronStack(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const in_n = e.u(3);
    const w = e.u(1);
    const h = e.u(2);
    const n = i % in_n;
    const x = (i / in_n) % w;
    const r = i / (in_n * w);
    const j = r / (h + 1);
    const y = r % (h + 1);
    const v: f32 = if (y == h) 0 else e.ldT(.b, dt, (y * w + x) * e.u(4) + (e.u(6) + j) * in_n + n);
    e.stT(.a, dt, i, v);
}

/// An output-axis DoRA's rescale of the base product: `a[p][o] += f0 * (b[o] - 1) *
/// (a[p][o] - c[o])`, `c` the bias the base added (read only when u2 != 0). `a` `dt`.
/// u0 = rows * u1, u1 = width.
pub inline fn doraRows(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const o = i % e.u(1);
    const y = e.ldT(.a, dt, i);
    const bias: f32 = if (e.u(2) != 0) e.ld(.c, o) else 0;
    e.stT(.a, dt, i, y + e.f(0) * (e.ld(.b, o) - 1) * (y - bias));
}

/// An input-axis DoRA's share of the base product's input: `a[p][j] = (c[j] - 1) *
/// b[p][j]`, `a` f32 and `b` `dt`. u0 = rows * u1, u1 = width.
pub inline fn doraCols(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, (e.ld(.c, i % e.u(1)) - 1) * e.ldT(.b, dt, i));
}

/// Scatter per-patch pixels back to the image: patch `t` of a `u1`-wide latent
/// grid covers a `u2 x u2` tile, and its pixel `p` is `(p / u2, p % u2)` inside
/// it. `a` is `[h][w][3]`, `b` is `[tiles][u2 * u2][3]`.
/// u0 = total, u1 = latent width, u2 = patch side, u3 = first tile index.
pub inline fn patchScatter(e: Env) void {
    const i = k.elem(e) orelse return;
    const side = e.u(2);
    const area = side * side;
    const c = i % 3;
    const p = (i / 3) % area;
    const tile = e.u(3) + i / (3 * area);
    const img_w = e.u(1) * side;
    const y = (tile / e.u(1)) * side + p / side;
    const x = (tile % e.u(1)) * side + p % side;
    e.st(.a, (y * img_w + x) * 3 + c, e.ld(.b, i));
}

/// Gather a `[h][w][ch]` activation into 32x32 attention windows laid out as
/// `[win_area][n_win][ch]`, i.e. with the WINDOW as an attention head, so one
/// batched call attends each window against itself alone.
///
/// Out-of-range samples REPLICATE the edge rather than reading zero, which is the
/// reference's `F.pad(mode="replicate")` and is not interchangeable with it: a
/// latent smaller than the window is padded up and its edge cells attend to
/// duplicates of themselves.
/// u0 = total, u1 = h, u2 = w, u3 = ch, u4 = windows across, u5 = window side,
/// u6 = window count.
pub inline fn winGather(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(3);
    const npw = e.u(4);
    const side = e.u(5);
    const n_win = e.u(6);
    const c = i % ch;
    const win = (i / ch) % n_win;
    const t = i / (ch * n_win);
    const sy0 = (win / npw) * side + t / side;
    const sx0 = (win % npw) * side + t % side;
    const sy = if (sy0 >= e.u(1)) e.u(1) - 1 else sy0;
    const sx = if (sx0 >= e.u(2)) e.u(2) - 1 else sx0;
    e.st(.a, i, e.ld(.b, (sy * e.u(2) + sx) * ch + c));
}

/// The inverse of `winGather`, dropping the replicated pad: one thread per
/// SOURCE element, so a padded position is simply never written.
/// Same scalars as `winGather`; `a` is the image, `b` the windows.
pub inline fn winScatter(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = e.u(3);
    const npw = e.u(4);
    const side = e.u(5);
    const n_win = e.u(6);
    const c = i % ch;
    const win = (i / ch) % n_win;
    const t = i / (ch * n_win);
    const sy = (win / npw) * side + t / side;
    const sx = (win % npw) * side + t % side;
    if (sy >= e.u(1) or sx >= e.u(2)) return;
    e.st(.a, (sy * e.u(2) + sx) * ch + c, e.ld(.b, i));
}

/// Per-ROW AdaLN, where every row carries its own shift/scale rather than
/// sharing a per-column vector: `a[r][j] = a[r][j] * (1 + c[r][D+j]) + c[r][j]`,
/// `c` being `[rows][3 * D]` as `shift, scale, gate`.
///
/// Mage-VAE's per-pixel MLP conditions each PIXEL on its own patch vector, which
/// is what makes this per-row where every transformer's AdaLN here is per-column.
/// u0 = rows*D, u1 = D. a and c `dt`.
pub inline fn modulatePerRow(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const d = e.u(1);
    const r = i / d;
    const j = i % d;
    e.stT(.a, dt, i, e.ldT(.a, dt, i) * (1.0 + e.ldT(.c, dt, r * 3 * d + d + j)) + e.ldT(.c, dt, r * 3 * d + j));
}

/// The gated residual half of `modulatePerRow`: `a[r][j] += c[r][2D+j] * b[r][j]`.
/// u0 = rows*D, u1 = D. The residual a is f32 at every `dt`; b and c are `dt`.
pub inline fn gatedAddPerRow(e: Env, comptime dt: Dt) void {
    const i = k.elem(e) orelse return;
    const d = e.u(1);
    const r = i / d;
    const j = i % d;
    e.st(.a, i, e.ld(.a, i) + e.ldT(.c, dt, r * 3 * d + 2 * d + j) * e.ldT(.b, dt, i));
}
