//! Elementwise dual-target kernel bodies (see ../dual.zig for the prelude).
//!
//! One thread per element, or per PAIR of f16 elements: an f16 store must own its
//! whole u32 word on SPIR-V, so every body that writes f16 indexes pairs, on both
//! targets, and its host launches `elems / 2` threads (`dualPairs` on CUDA).
//! Layouts are the Vulkan push layouts, since Vulkan model files carry them at
//! the call site; the CUDA wrappers in gpu/cuda/backend.zig remap their arguments.
//! Each body's doc names the slots and scalars it reads.

const k = @import("../dual.zig");
const Env = k.Env;

// ---- activations ------------------------------------------------------------

/// a += b. u0 = n.
pub inline fn add(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, e.ld(.a, i) + e.ld(.b, i));
}

/// a = max(0, a + b). u0 = n.
pub inline fn addRelu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, @max(0.0, e.ld(.a, i) + e.ld(.b, i)));
}

/// a += f0 * b. u0 = n.
pub inline fn addScaled(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, e.ld(.a, i) + e.f(0) * e.ld(.b, i));
}

/// a = max(0, a). u0 = n.
pub inline fn relu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, @max(0.0, e.ld(.a, i)));
}

/// a = silu(a). u0 = n.
pub inline fn silu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.silu(e.ld(.a, i)));
}

/// a = silu(a) * b. u0 = n.
pub inline fn siluMul(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.silu(e.ld(.a, i)) * e.ld(.b, i));
}

/// a *= sigmoid(b). u0 = n.
pub inline fn sigmoidMul(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, e.ld(.a, i) * k.sigmoid(e.ld(.b, i)));
}

/// a = geluTanh(a). u0 = n.
pub inline fn gelu(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.geluTanh(e.ld(.a, i)));
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
pub inline fn geluErf(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, k.geluErf(e.ld(.a, i)));
}

/// GEGLU: b[p][j] = a[p][j] * geluErf(a[p][inner + j]), value then gate in that
/// order. a = src [n][2*u1], b = dst [n][u1]. u0 = n*u1, u1 = inner.
pub inline fn geglu(e: Env) void {
    const i = k.elem(e) orelse return;
    const inner = e.u(1);
    const base = (i / inner) * 2 * inner + i % inner;
    e.st(.b, i, e.ld(.a, base) * k.geluErf(e.ld(.a, base + inner)));
}

/// `geglu` over f16 activations, one thread per output pair (inner is even).
pub inline fn gegluH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    const inner = e.u(1);
    const base = (e0 / inner) * 2 * inner + e0 % inner;
    var out: [2]f32 = undefined;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        out[j] = e.h16(.a, base + j) * k.geluErf(e.h16(.a, base + inner + j));
    }
    e.stW(.b, w, k.packH16(out[0], out[1]));
}

/// a *= softplus(gate*ln2)/ln2, as log2(1 + 2^gate), stable on both signs. u0 = n.
pub inline fn softplusGate(e: Env) void {
    const i = k.elem(e) orelse return;
    const g = e.ld(.b, i);
    const sp = if (g > 0) g + k.log2(1.0 + @exp2(-g)) else k.log2(1.0 + @exp2(g));
    e.st(.a, i, e.ld(.a, i) * sp);
}

// ---- copies, scales, concatenation ------------------------------------------

/// b[u2 + i] = a[u3 + i]. u0 = n.
pub inline fn copy(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, e.u(2) + i, e.ld(.a, e.u(3) + i));
}

/// b[u2 + i] = a[u3 + i] * f0, in place when a and b are one buffer. u0 = n.
pub inline fn scaleF32(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, e.u(2) + i, e.ld(.a, e.u(3) + i) * e.f(0));
}

/// Channel concatenation, one thread per SOURCE element: b[p][u3 + j] = a[p][j].
/// u0 = n*u1, u1 = src channels, u2 = dst channels, u3 = dst channel offset.
pub inline fn concatCh(e: Env) void {
    const i = k.elem(e) orelse return;
    const j = i % e.u(1);
    e.st(.b, (i / e.u(1)) * e.u(2) + e.u(3) + j, e.ld(.a, i));
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

/// y = s32_acc * act_scale[row] * weight_scale[col]. a = acc (s32 bits),
/// c = weight_scale [u1], d = act_scale [rows], b = y. u0 = rows*u1, u1 = nout.
pub inline fn scaleI32(e: Env) void {
    const i = k.elem(e) orelse return;
    const acc: i32 = @bitCast(e.ldW(.a, i));
    e.st(.b, i, @as(f32, @floatFromInt(acc)) * e.ld(.d, i / e.u(1)) * e.ld(.c, i % e.u(1)));
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
pub inline fn modulate(e: Env) void {
    const i = k.elem(e) orelse return;
    const col = i % e.u(1);
    e.st(.a, i, (1.0 + e.ld(.c, e.u(2) + col)) * e.ld(.a, i) + e.ld(.c, e.u(3) + col));
}

/// a += c[gate + col] * b, gate = u2, or per row u2 + d[row]*u3 when u3 != 0
/// (d holds one u32 label per row: a denoise mask relabels rows inside a segment).
/// u0 = n, u1 = dim, u2 = gate_off, u3 = label stride (0 = uniform).
pub inline fn gatedAdd(e: Env) void {
    const i = k.elem(e) orelse return;
    const dim = e.u(1);
    const col = i % dim;
    var off = e.u(2);
    if (e.u(3) != 0) off += e.ldW(.d, i / dim) * e.u(3);
    e.st(.a, i, e.ld(.a, i) + e.ld(.c, off + col) * e.ld(.b, i));
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

/// y = x * inv[row] * premul[col] + shift[col]. a = x, b = out, c = vectors,
/// d = inv. u0 = n, u1 = dim, u2 = premul offset, u3 = shift offset.
pub inline fn rmsApplyMod(e: Env) void {
    const i = k.elem(e) orelse return;
    const col = i % e.u(1);
    e.st(.b, i, e.ld(.a, i) * e.ld(.d, i / e.u(1)) * e.ld(.c, e.u(2) + col) + e.ld(.c, e.u(3) + col));
}

/// `rmsApplyMod` emitting f16 pairs scaled by f0, zero past u4 real elements.
/// b = out words. u0 = words, u1 = dim, u2, u3 as above, u4 = real elems.
pub inline fn rmsApplyModH16(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const idx = e0 + j;
        if (idx < e.u(4)) {
            const col = idx % e.u(1);
            const v = (e.ld(.a, idx) * e.ld(.d, idx / e.u(1)) * e.ld(.c, e.u(2) + col) + e.ld(.c, e.u(3) + col)) * e.f(0);
            out |= @as(u32, k.f16Bits(v)) << (16 * j);
        }
    }
    e.stW(.b, w, out);
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
/// (cos then sin at u2). u0 = total pairs, u1 = half, u2 = sin_off, u3 = n_heads.
pub inline fn ropeInter(e: Env) void {
    const i = k.elem(e) orelse return;
    const half = e.u(1);
    const pair = i % half;
    const pos = i / (half * e.u(3));
    const cos_v = e.ld(.c, pos * half + pair);
    const sin_v = e.ld(.c, e.u(2) + pos * half + pair);
    const at = i * 2;
    const x0 = e.ld(.a, at);
    const x1 = e.ld(.a, at + 1);
    e.st(.a, at, x0 * cos_v - x1 * sin_v);
    e.st(.a, at + 1, x0 * sin_v + x1 * cos_v);
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

/// Argmax pass 2, one thread: the lane winners to the global argmax, lowest index
/// on ties. a = vals [L], b = idx [L], d = out [1]. u0 = L.
pub inline fn argmaxFinal(e: Env) void {
    if (e.gid() != 0) return;
    var best_v: f32 = e.ld(.a, 0);
    var best_i: u32 = @intFromFloat(e.ld(.b, 0));
    var i: u32 = 1;
    while (i < e.u(0)) : (i += 1) {
        const v = e.ld(.a, i);
        const idx: u32 = @intFromFloat(e.ld(.b, i));
        if (v > best_v or (v == best_v and idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }
    e.st(.d, 0, @floatFromInt(best_i));
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

/// Tight [rows][u1] f32 -> [*][u2] f16 rows (u2 >= u1, even), zero in the column
/// tail and past u3 rows, scaled by f0. a = src, d = words. u0 = out words.
pub inline fn f32ToH16Pad(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const row = e0 / e.u(2);
    const col = e0 % e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const cc = col + j;
        if (cc < e.u(1) and row < e.u(3)) out |= @as(u32, k.f16Bits(e.ld(.a, row * e.u(1) + cc) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// `f32ToH16Pad` emitting bf16 (round to nearest even).
pub inline fn f32ToBf16Pad(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const row = e0 / e.u(2);
    const col = e0 % e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const cc = col + j;
        if (cc < e.u(1) and row < e.u(3)) out |= @as(u32, k.bf16Bits(e.ld(.a, row * e.u(1) + cc) * e.f(0))) << @intCast(16 * j);
    }
    e.stW(.d, w, out);
}

/// `f32ToH16Pad` whose source is already f16 (a = f16 elements).
pub inline fn h16ToH16Pad(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const row = e0 / e.u(2);
    const col = e0 % e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const cc = col + j;
        if (cc < e.u(1) and row < e.u(3)) out |= @as(u32, k.f16Bits(e.h16(.a, row * e.u(1) + cc) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// `h16ToH16Pad` from a bf16 source (a = bf16 elements).
pub inline fn bf16ToH16Pad(e: Env) void {
    const w = k.elem(e) orelse return;
    const e0 = w * 2;
    const row = e0 / e.u(2);
    const col = e0 % e.u(2);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const cc = col + j;
        if (cc < e.u(1) and row < e.u(3)) out |= @as(u32, k.f16Bits(e.bf16(.a, row * e.u(1) + cc) * e.f(0))) << (16 * j);
    }
    e.stW(.d, w, out);
}

/// b[i] = f32(a[i]) from f16. u0 = n.
pub inline fn f16ToF32(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.b, i, e.h16(.a, i));
}

/// Two f16 activations summed in f32, in place. u0 = n (even).
pub inline fn addH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    e.stW(.a, w, k.packH16(e.h16(.a, e0) + e.h16(.b, e0), e.h16(.a, e0 + 1) + e.h16(.b, e0 + 1)));
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

/// f32 row-major [rows][cols] -> f16 k-major [k_pad][n_pad] (element (k, row) at
/// k*n_pad + row), zeros in both pads; one thread per output word holding two
/// adjacent rows. a = src, d = words. u0 = words, u1 = rows, u2 = cols, u3 = n_pad.
pub inline fn packH16Kmajor(e: Env) void {
    const w = k.elem(e) orelse return;
    const base = w * 2;
    const kk = base / e.u(3);
    const row0 = base % e.u(3);
    var out: u32 = 0;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const row = row0 + j;
        if (kk < e.u(2) and row < e.u(1)) out |= @as(u32, k.f16Bits(e.ld(.a, row * e.u(2) + kk))) << @intCast(16 * j);
    }
    e.stW(.d, w, out);
}

// ---- bias, groupnorm apply, VAE norm ---------------------------------------

/// After a column-padded GEMM: d[u3 + i] = f0*a[(i/u1)*u2 + i%u1] + b[u4 + i%u1].
/// f0 is the GEMM output scale and must be 1.0 for the plain case. u0 = rows*u1,
/// u1 = co, u2 = padded row stride, u3 = dst offset, u4 = bias offset.
pub inline fn biasCompact(e: Env) void {
    const i = k.elem(e) orelse return;
    const cc = i % e.u(1);
    e.st(.d, e.u(3) + i, e.ld(.a, (i / e.u(1)) * e.u(2) + cc) * e.f(0) + e.ld(.b, e.u(4) + cc));
}

/// `biasCompact` writing f16 pairs; u3 must be even. u0 = elements.
pub inline fn biasCompactH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    var out: [2]f32 = undefined;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const idx = e0 + j;
        const cc = idx % e.u(1);
        out[j] = e.ld(.a, (idx / e.u(1)) * e.u(2) + cc) * e.f(0) + e.ld(.b, e.u(4) + cc);
    }
    e.stW(.d, (e.u(3) + e0) >> 1, k.packH16(out[0], out[1]));
}

/// c[u2 + i] = f32(a[i] as f16) + b[i % u1], the cuDNN conv output plus its bias.
/// u0 = n*co, u1 = co.
pub inline fn biasAddF16(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.c, e.u(2) + i, e.h16(.a, i) + e.ld(.b, i % e.u(1)));
}

/// `biasAddF16` writing f16 pairs; u2 and co even.
pub inline fn biasAddH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    var out: [2]f32 = undefined;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        out[j] = e.h16(.a, e0 + j) + e.ld(.b, (e0 + j) % e.u(1));
    }
    e.stW(.c, (e.u(2) + e0) >> 1, k.packH16(out[0], out[1]));
}

/// a[p][c] += b[u3 + c]. u0 = n*ch, u1 = ch.
pub inline fn addBiasRows(e: Env) void {
    const i = k.elem(e) orelse return;
    e.st(.a, i, e.ld(.a, i) + e.ld(.b, e.u(3) + i % e.u(1)));
}

/// `addBiasRows` over an f16 activation (ch even).
pub inline fn addBiasRowsH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    var out: [2]f32 = undefined;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        out[j] = e.h16(.a, e0 + j) + e.ld(.b, e.u(3) + (e0 + j) % e.u(1));
    }
    e.stW(.a, w, k.packH16(out[0], out[1]));
}

/// GroupNorm apply: b = (a - mean[g])*inv[g]*w[ch] + bias[ch], then silu when u5.
/// c = weight ++ bias (bias at u4), d = mean ++ inv (inv at u3). u0 = n*ch,
/// u1 = ch, u2 = per_group, u3 = groups, u4 = bias offset, u5 = silu.
pub inline fn gnApply(e: Env) void {
    const i = k.elem(e) orelse return;
    const ch = i % e.u(1);
    const g = ch / e.u(2);
    var v = (e.ld(.a, i) - e.ld(.d, g)) * e.ld(.d, e.u(3) + g) * e.ld(.c, ch) + e.ld(.c, e.u(4) + ch);
    if (e.u(5) != 0) v = k.silu(v);
    e.st(.b, i, v);
}

/// `gnApply` reading and writing f16; the two halves of a pair may fall in
/// different groups, so the lookup is per element.
pub inline fn gnApplyH16(e: Env) void {
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    var out: [2]f32 = undefined;
    inline for (0..2) |j_| {
        const j: u32 = @intCast(j_);
        const idx = e0 + j;
        const ch = idx % e.u(1);
        const g = ch / e.u(2);
        var v = (e.h16(.a, idx) - e.ld(.d, g)) * e.ld(.d, e.u(3) + g) * e.ld(.c, ch) + e.ld(.c, e.u(4) + ch);
        if (e.u(5) != 0) v = k.silu(v);
        out[j] = v;
    }
    e.stW(.b, w, k.packH16(out[0], out[1]));
}

/// Per-position channel L2 norm, x * sqrt(c)/max(|x|, f0) * gamma, then silu when
/// u2. One thread per position. a = x [n][c], b = out, c = gamma. u0 = n, u1 = c.
pub inline fn vaeNorm(e: Env) void {
    const row = k.elem(e) orelse return;
    const dim = e.u(1);
    const base = row * dim;
    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < dim) : (i += 1) {
        const v = e.ld(.a, base + i);
        sum += v * v;
    }
    const inv = @sqrt(@as(f32, @floatFromInt(dim))) / @max(@sqrt(sum), e.f(0));
    i = 0;
    while (i < dim) : (i += 1) {
        var v = e.ld(.a, base + i) * inv * e.ld(.c, i);
        if (e.u(2) != 0) v = k.silu(v);
        e.st(.b, base + i, v);
    }
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

inline fn im2colSdImpl(e: Env, comptime src_f16: bool) void {
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
        v = if (src_f16) e.h16(.a, src) else e.ld(.a, src);
    }
    e.st(.d, i, v);
}

/// The SD family's 3x3 patch matrix, with stride-2 convolutions. a = src, d =
/// patches. u0 = bn*u1, u1 = 9*ci, u2 = ci, u3 = src w, u4 = grid extent (rows),
/// u5 = band start, u6 = OUT width, f0 = 0 stride 1 / 1 fused 2x upsample / 2
/// stride 2.
pub inline fn im2colSd(e: Env) void {
    im2colSdImpl(e, false);
}

/// `im2colSd` reading an f16 source; the patch stays f32.
pub inline fn im2colSdH16(e: Env) void {
    im2colSdImpl(e, true);
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
/// u0 = threads, u1 = nchunks, u2 = rows (= valid cols), u3 = S row stride,
/// u5 = S plane stride (elements).
pub inline fn softmaxPartial(e: Env) void {
    const i = k.elem(e) orelse return;
    const nch = e.u(1);
    const chunk = i % nch;
    const qz = i / nch;
    const q = qz % e.u(2);
    const z = qz / e.u(2);
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

/// Fused per-head rmsnorm + interleaved rope + output scale, in place on f16 pair
/// words, one thread per (pos, head) row. a = qk words, b = norm weight,
/// c = freqs. u0 = rows, u1 = half, u2 = sin_off, u3 = n_heads, f0 = scale, f1 = eps.
pub inline fn qknormRope16(e: Env) void {
    const row = k.elem(e) orelse return;
    const half = e.u(1);
    const pos = row / e.u(3);
    const base_w = row * half;
    var sum: f32 = 0;
    var w: u32 = 0;
    while (w < half) : (w += 1) {
        const word = e.ldW(.a, base_w + w);
        inline for (0..2) |j_| {
            const j: u32 = @intCast(j_);
            const v = k.h16(word >> (16 * j));
            sum += v * v;
        }
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(half * 2)) + e.f(1));
    w = 0;
    while (w < half) : (w += 1) {
        const word = e.ldW(.a, base_w + w);
        const x0 = k.h16(word) * inv * e.ld(.b, w * 2);
        const x1 = k.h16(word >> 16) * inv * e.ld(.b, w * 2 + 1);
        const cos_v = e.ld(.c, pos * half + w);
        const sin_v = e.ld(.c, e.u(2) + pos * half + w);
        e.stW(.a, base_w + w, k.packH16((x0 * cos_v - x1 * sin_v) * e.f(0), (x0 * sin_v + x1 * cos_v) * e.f(0)));
    }
}

/// `qknormRope16` entirely in f32, in place. a = x [rows][2*half].
pub inline fn qknormRopeF32(e: Env) void {
    const row = k.elem(e) orelse return;
    const half = e.u(1);
    const pos = row / e.u(3);
    const base = row * half * 2;
    var sum: f32 = 0;
    var w: u32 = 0;
    while (w < half * 2) : (w += 1) {
        const v = e.ld(.a, base + w);
        sum += v * v;
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(half * 2)) + e.f(1));
    w = 0;
    while (w < half) : (w += 1) {
        const x0 = e.ld(.a, base + w * 2) * inv * e.ld(.b, w * 2);
        const x1 = e.ld(.a, base + w * 2 + 1) * inv * e.ld(.b, w * 2 + 1);
        const cos_v = e.ld(.c, pos * half + w);
        const sin_v = e.ld(.c, e.u(2) + pos * half + w);
        e.st(.a, base + w * 2, (x0 * cos_v - x1 * sin_v) * e.f(0));
        e.st(.a, base + w * 2 + 1, (x0 * sin_v + x1 * cos_v) * e.f(0));
    }
}
