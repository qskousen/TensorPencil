//! Device-side elementwise / normalization / attention kernels for the
//! GPU-resident DiT forward. No workgroup memory anywhere (see ZIG.md on the
//! NVIDIA + Zig-SPIR-V workgroup issue), so all entries share this module.
//!
//! Universal binding layout (set 0): four storage buffers a, b, c, d whose
//! roles depend on the entry point; unused bindings get a dummy buffer.
//! Push constants: six u32 (u0..u5) + two f32 (f0, f1); meaning per entry.
//!
//! Entries (thread mapping / bindings):
//!   rmsnorm     x = row.               a=in, b=out, c=weight.
//!               u0=n_rows u1=dim f0=eps.
//!   modulate    x = element.           a=x (in place), c=vectors.
//!               u0=n u1=dim u2=scale_off u3=shift_off  (y = (1+s)*x + sh)
//!   gated_add   x = element.           a=x, b=delta, c=vectors.
//!               u0=n u1=dim u2=gate_off                (x += g*delta)
//!   add         x = element.           a=x, b=delta. u0=n.
//!   silu_mul    x = element.           a=gate, b=up. u0=n. (a = silu(a)*b)
//!   sigmoid_mul x = element.           a=dst, b=gate. u0=n. (a *= sigmoid(b))
//!   rope_inter  x = (pos, head, pair) flattened. a=qk (in place), c=freqs
//!               (cos then sin halves). u0=total_pairs u1=half u2=sin_off
//!               u3=n_heads.
//!   attention   x = head*chunks+chunk, y = query. a=q, b=k, c=v, d=out.
//!               u0=seq u1=n_heads u2=n_kv_heads u3=head_dim f0=scale.
//!               Online softmax; each thread owns a 32-wide slice of the
//!               head dim and recomputes the (cheap relative to PV) dots.

const gpu = @import("std").gpu;

// Type-level maximum, not an allocation: the VAE decoder's final-stage
// activations reach ~180M f32 at 5.4 MP, past the old 1 << 27 bound.
const FBuf = extern struct { data: [1 << 28]f32 };

pub const Push = extern struct {
    u0: u32,
    u1: u32,
    u2: u32,
    u3: u32,
    u4: u32,
    u5: u32,
    f0: f32,
    f1: f32,
};

extern var a: FBuf addrspace(.storage_buffer);
extern var b: FBuf addrspace(.storage_buffer);
extern var c: FBuf addrspace(.storage_buffer);
extern var d: FBuf addrspace(.storage_buffer);
extern var pc: Push addrspace(.push_constant);

inline fn decorate() void {
    asm volatile (
        \\OpDecorate %ft Block
        \\OpMemberDecorate %ft 0 Offset 0
        \\OpDecorate %pt Block
        \\OpMemberDecorate %pt 0 Offset 0
        \\OpMemberDecorate %pt 1 Offset 4
        \\OpMemberDecorate %pt 2 Offset 8
        \\OpMemberDecorate %pt 3 Offset 12
        \\OpMemberDecorate %pt 4 Offset 16
        \\OpMemberDecorate %pt 5 Offset 20
        \\OpMemberDecorate %pt 6 Offset 24
        \\OpMemberDecorate %pt 7 Offset 28
        \\OpDecorate %ba DescriptorSet 0
        \\OpDecorate %ba Binding 0
        \\OpDecorate %bb DescriptorSet 0
        \\OpDecorate %bb Binding 1
        \\OpDecorate %bc DescriptorSet 0
        \\OpDecorate %bc Binding 2
        \\OpDecorate %bd DescriptorSet 0
        \\OpDecorate %bd Binding 3
        :
        : [ft] "t" (FBuf),
          [pt] "t" (Push),
          [ba] "" (&a),
          [bb] "" (&b),
          [bc] "" (&c),
          [bd] "" (&d),
    );
}

export fn rmsnorm() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const dim = pc.u1;
    const base = row * dim;
    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < dim) : (i += 1) {
        const v = a.data[base + i];
        sum += v * v;
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(dim)) + pc.f0);
    i = 0;
    while (i < dim) : (i += 1) {
        b.data[base + i] = a.data[base + i] * inv * c.data[i];
    }
}

// --- parallel rmsnorm + modulation (wide rows) ----------------------------
// The one-thread-per-row rmsnorm is latency-bound at dim 6144 (a few
// thousand serial-loop threads leave the GPU mostly idle), so wide rows use
// three fully parallel passes; the norm weight is prefolded into the
// modulation scale on the CPU (premul = (1+mod_scale)*w).
// rms_partial: one thread per (row, chunk): sum of squares over the
//   INTERLEAVED slice i = chunk, chunk+u2, ... (consecutive threads read
//   consecutive addresses; see softmax_partial). a = x, d = partials
//   [row][u2]. u0 = total threads (rows*u2), u1 = dim, u2 = chunks per row.
export fn rms_partial() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const nch = pc.u2;
    const row = idx / nch;
    const ch = idx % nch;
    const base = row * pc.u1;
    var sum: f32 = 0;
    var i: u32 = ch;
    while (i < pc.u1) : (i += nch) {
        const v = a.data[base + i];
        sum += v * v;
    }
    d.data[idx] = sum;
}

// rms_combine: one thread per row: inv_rms from the chunk partials.
//   a = partials, d = inv [rows]. u0 = rows, u1 = dim, u2 = chunks, f0 = eps.
export fn rms_combine() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < pc.u2) : (i += 1) sum += a.data[row * pc.u2 + i];
    d.data[row] = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(pc.u1)) + pc.f0);
}

// rms_apply_mod: one thread per element: y = x*inv[row]*premul[col] +
//   shift[col]. a = x, b = out, c = vectors, d = inv.
//   u0 = n, u1 = dim, u2 = premul offset, u3 = shift offset.
export fn rms_apply_mod() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const col = idx % pc.u1;
    b.data[idx] = a.data[idx] * d.data[idx / pc.u1] * c.data[pc.u2 + col] + c.data[pc.u3 + col];
}

export fn modulate() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const col = idx % pc.u1;
    a.data[idx] = (1.0 + c.data[pc.u2 + col]) * a.data[idx] + c.data[pc.u3 + col];
}

export fn gated_add() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const col = idx % pc.u1;
    a.data[idx] += c.data[pc.u2 + col] * b.data[idx];
}

export fn add() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    a.data[idx] += b.data[idx];
}

export fn silu_mul() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const g = a.data[idx];
    a.data[idx] = g / (1.0 + @exp(-g)) * b.data[idx];
}

export fn sigmoid_mul() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    a.data[idx] *= 1.0 / (1.0 + @exp(-b.data[idx]));
}

export fn rope_inter() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const half = pc.u1;
    const pair = idx % half;
    const pos = idx / (half * pc.u3);
    const cos_v = c.data[pos * half + pair];
    const sin_v = c.data[pc.u2 + pos * half + pair];
    const at = idx * 2;
    const x0 = a.data[at];
    const x1 = a.data[at + 1];
    a.data[at] = x0 * cos_v - x1 * sin_v;
    a.data[at + 1] = x0 * sin_v + x1 * cos_v;
}

// rope_half: rotate-half RoPE (Qwen/Llama convention) in place — pairs
// element i with i+half rather than adjacent lanes. a = qk, c = freqs (cos
// then sin halves). u0 = total (seq*n_heads*half), u1 = half, u2 = sin_off,
// u3 = n_heads. Matches ops.rope.applyRotateHalf exactly.
export fn rope_half() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const half = pc.u1;
    const i = idx % half;
    const pos = idx / (half * pc.u3);
    const row = idx / half; // pos*n_heads + head
    const cos_v = c.data[pos * half + i];
    const sin_v = c.data[pc.u2 + pos * half + i];
    const base = row * (2 * half);
    const lo = a.data[base + i];
    const hi = a.data[base + half + i];
    a.data[base + i] = lo * cos_v - hi * sin_v;
    a.data[base + half + i] = hi * cos_v + lo * sin_v;
}

// copy: b[u2 + idx] = a[idx]. Contiguous copy with a destination offset;
// used to snapshot hidden-state taps into a tap-major output buffer on the
// device (keeps the encoder forward in one batch). u0 = element count,
// u2 = dest offset.
export fn copy() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    b.data[pc.u2 + idx] = a.data[idx];
}

// --- GEMM-ified attention (scores buffer batched over head groups) -------
// attn_scores: S[z][q][j] = scale * dot(Q[q, head, :], K[j, kv(head), :])
//   where head = u4 + z. 4x4 register tile per thread; x = key tile,
//   y = query tile, z = head-in-batch.
//   a=Q, b=K, d=S. u0=seq u1=n_heads u2=n_kv u3=hd u4=head_off f0=scale.
// softmax_rows: in-place row softmax. a=S. u0=n_rows u1=row_len. x = row.
// attn_out: out[q][head][c] = sum_j S[z][q][j] * V[j][kv(head)][c].
//   4 queries x 4 channels per thread; x = channel tile, y = query tile,
//   z = head-in-batch. a=S, c=V, d=out. u0..u4 as attn_scores.
const amm_tile = 8;

// gather_kmajor: d[(h*hd + k)*seq + s] = a[(s*n_heads + h)*hd + k] —
// per-head k-major layout so the scores kernel loads contiguously.
// u0=seq*n_heads*hd u1=n_heads u2=hd u3=seq. x = flat source index.
export fn gather_kmajor() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const hd = pc.u2;
    const k = idx % hd;
    const h = (idx / hd) % pc.u1;
    const s = idx / (hd * pc.u1);
    d.data[(h * hd + k) * pc.u3 + s] = a.data[idx];
}

// gather_kmajor_h16: f16 per-head k-major gather for the tensor-core scores
// GEMM. d = packed f16 pairs [kv][hd][s_stride] <- a = f32 [seq][kv][hd];
// positions >= seq write zero (column padding). x = dest word index.
// u0 = out words, u1 = hd, u2 = s_stride (even), u3 = seq, u4 = n_kv_heads.
export fn gather_kmajor_h16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const hd = pc.u1;
    const sstride = pc.u2;
    const plane = hd * sstride;
    const h = e0 / plane;
    const rem = e0 % plane;
    const k = rem / sstride;
    const s0 = rem % sstride;
    var out: u32 = 0;
    inline for (0..2) |j| {
        const s = s0 + j;
        if (s < pc.u3) {
            const v: f16 = @floatCast(a.data[(s * pc.u4 + h) * hd + k]);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

export fn attn_scores() callconv(.spirv_kernel) void {
    decorate();
    const seq = pc.u0;
    const hd = pc.u3;
    const head = pc.u4 + gpu.global_invocation_id[2];
    const kv_head = head / (pc.u1 / pc.u2);
    const j0 = gpu.global_invocation_id[0] * amm_tile;
    const q0 = gpu.global_invocation_id[1] * amm_tile;
    if (j0 >= seq or q0 >= seq) return;

    // Inputs are per-head k-major (gather_kmajor): row k of head h starts at
    // (h*hd + k) * seq. All tile loads are contiguous runs of amm_tile.
    const qrow0 = head * hd * seq;
    const krow0 = kv_head * hd * seq;

    var acc: [amm_tile][amm_tile]f32 = @splat(@splat(0.0));
    var k: u32 = 0;
    while (k < hd) : (k += 1) {
        var qv: [amm_tile]f32 = undefined;
        var kv: [amm_tile]f32 = undefined;
        inline for (0..amm_tile) |i| {
            qv[i] = a.data[qrow0 + k * seq + q0 + i];
            kv[i] = b.data[krow0 + k * seq + j0 + i];
        }
        inline for (0..amm_tile) |i| {
            inline for (0..amm_tile) |jj| {
                acc[i][jj] += qv[i] * kv[jj];
            }
        }
    }

    const z = gpu.global_invocation_id[2];
    inline for (0..amm_tile) |i| {
        const q = q0 + i;
        if (q < seq) {
            inline for (0..amm_tile) |jj| {
                const j = j0 + jj;
                if (j < seq) {
                    d.data[(z * seq + q) * seq + j] = acc[i][jj] * pc.f0;
                }
            }
        }
    }
}

// --- two-pass softmax feeding the tensor-core P@V GEMM -------------------
// softmax_partial: one thread per (head z, row q, chunk): online max/sum-exp
// over an INTERLEAVED slice of f16-pair WORDS (the scores kernel stores S
// half-precision) — consecutive threads read consecutive words every
// iteration (a contiguous-block split makes warp lanes stride kilobytes
// apart and costs ~30x bandwidth). Max and exp-sum are order-independent,
// so the interleaving is free. a = S (f16 pairs read through the f32 view),
// d = partials [(z*rows+q)*nchunks+chunk] x {m, d}. u0 = total threads,
// u1 = nchunks, u2 = rows == valid cols (seq), u3 = S row stride (elems),
// u5 = S plane stride (elems).
export fn softmax_partial() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const nch = pc.u1;
    const chunk = idx % nch;
    const qz = idx / nch;
    const q = qz % pc.u2;
    const z = qz / pc.u2;
    const base_w = (z * pc.u5 + q * pc.u3) / 2;
    const nw = (pc.u2 + 1) / 2;
    var m: f32 = -3.4e38;
    var dsum: f32 = 0;
    var wi = chunk;
    while (wi < nw) : (wi += nch) {
        const word: u32 = @bitCast(a.data[base_w + wi]);
        inline for (0..2) |h| {
            const j = wi * 2 + h;
            if (j < pc.u2) {
                const hv: f16 = @bitCast(@as(u16, @truncate(word >> (16 * h))));
                const s: f32 = @floatCast(hv);
                const mn = @max(m, s);
                dsum = dsum * @exp(m - mn) + @exp(s - mn);
                m = mn;
            }
        }
    }
    d.data[idx * 2] = m;
    d.data[idx * 2 + 1] = dsum;
}

// softmax_combine: one thread per (z, q): fold the chunk partials into the
// row max and reciprocal denominator. a = partials, d = md [z][rows_pad]
// x {m, 1/d} (pad rows stay garbage — their P@V output rows are never
// read). u0 = total threads (z*rows), u1 = nchunks, u2 = rows (seq),
// u3 = rows_pad (md rows per head plane).
export fn softmax_combine() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const q = idx % pc.u2;
    const z = idx / pc.u2;
    const pbase = idx * pc.u1 * 2;
    var m: f32 = -3.4e38;
    var i: u32 = 0;
    while (i < pc.u1) : (i += 1) m = @max(m, a.data[pbase + i * 2]);
    var dsum: f32 = 0;
    i = 0;
    while (i < pc.u1) : (i += 1) {
        dsum += a.data[pbase + i * 2 + 1] * @exp(a.data[pbase + i * 2] - m);
    }
    d.data[(z * pc.u3 + q) * 2] = m;
    d.data[(z * pc.u3 + q) * 2 + 1] = 1.0 / dsum;
}

export fn softmax_rows() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const n = pc.u1;
    const base = row * n;
    var m: f32 = -3.4e38;
    var i: u32 = 0;
    while (i < n) : (i += 1) m = @max(m, a.data[base + i]);
    var denom: f32 = 0;
    i = 0;
    while (i < n) : (i += 1) {
        const e = @exp(a.data[base + i] - m);
        a.data[base + i] = e;
        denom += e;
    }
    const inv = 1.0 / denom;
    i = 0;
    while (i < n) : (i += 1) a.data[base + i] *= inv;
}

export fn attn_out() callconv(.spirv_kernel) void {
    decorate();
    const seq = pc.u0;
    const hd = pc.u3;
    const head = pc.u4 + gpu.global_invocation_id[2];
    const kv_head = head / (pc.u1 / pc.u2);
    const c0 = gpu.global_invocation_id[0] * amm_tile;
    const q0 = gpu.global_invocation_id[1] * amm_tile;
    if (c0 >= hd or q0 >= seq) return;

    // Scores layout: row stride u5, per-head plane stride in f0 (u32 bits) —
    // the tensor-core scores path pads both to multiples of 128.
    const z = gpu.global_invocation_id[2];
    const kv_stride = pc.u2 * hd;
    const s_max = seq - 1;
    const sstr = pc.u5;
    const splane: u32 = @bitCast(pc.f0);
    var pb: [amm_tile]u32 = undefined;
    inline for (0..amm_tile) |i| {
        pb[i] = z * splane + @min(q0 + i, s_max) * sstr;
    }
    const vb = kv_head * hd + c0;
    // f1 != 0 => causal: row q attends only to keys j <= q (encoder path;
    // the DiT leaves f1 = 0 for full attention).
    const causal: u32 = @bitCast(pc.f1);

    // Online softmax fused over raw scores: no separate softmax pass, no
    // extra scores traffic. Per-row running max/denominator in registers.
    var m: [amm_tile]f32 = @splat(-3.4e38);
    var denom: [amm_tile]f32 = @splat(0.0);
    var acc: [amm_tile][amm_tile]f32 = @splat(@splat(0.0));
    var j: u32 = 0;
    while (j < seq) : (j += 1) {
        var vv: [amm_tile]f32 = undefined;
        inline for (0..amm_tile) |u| {
            vv[u] = c.data[j * kv_stride + vb + u];
        }
        inline for (0..amm_tile) |i| {
            if (causal == 0 or j <= q0 + i) {
                const s = a.data[pb[i] + j];
                const m_new = @max(m[i], s);
                const corr = @exp(m[i] - m_new);
                const pv = @exp(s - m_new);
                denom[i] = denom[i] * corr + pv;
                m[i] = m_new;
                inline for (0..amm_tile) |u| {
                    acc[i][u] = acc[i][u] * corr + pv * vv[u];
                }
            }
        }
    }

    const q_stride = pc.u1 * hd;
    inline for (0..amm_tile) |i| {
        const q = q0 + i;
        if (q < seq) {
            const inv = 1.0 / denom[i];
            inline for (0..amm_tile) |u| {
                d.data[q * q_stride + head * hd + c0 + u] = acc[i][u] * inv;
            }
        }
    }
}

// rms_apply_mod_h16: fused modulated rmsnorm + f16 conversion feeding the
// coop GEMMs — when every consumer of the normed rows shares one dequant
// scale, the f32 intermediate (and its round trip) is skipped entirely:
// b[word] = pack(f16((x*inv[row]*premul[col] + shift[col]) * f0)), zero pad
// words past the real element count. a = x, b = out (f16 pair words),
// c = vectors, d = inv. u0 = out words, u1 = dim, u2 = premul offset,
// u3 = shift offset, u4 = real elems, f0 = scale.
export fn rms_apply_mod_h16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    var out: u32 = 0;
    inline for (0..2) |j| {
        const e = e0 + j;
        if (e < pc.u4) {
            const col = e % pc.u1;
            const v: f16 = @floatCast((a.data[e] * d.data[e / pc.u1] * c.data[pc.u2 + col] + c.data[pc.u3 + col]) * pc.f0);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    b.data[idx] = @bitCast(out);
}

// f16 conversion for the cooperative-matrix path. Output is packed f16
// pairs in u32 words (binding d). Inputs beyond the real element count
// (u1) write zero — used to pad rows up to multiples of the GEMM tile.
// f32_to_h16: a = f32 source. u0 = out words, u1 = real elems, f0 = scale
// (the weight dequant scale rides on the activations; the coop kernel
// decodes raw fp8 weights unscaled).
export fn f32_to_h16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const v0: f16 = if (e0 < pc.u1) @floatCast(a.data[e0] * pc.f0) else 0;
    const v1: f16 = if (e0 + 1 < pc.u1) @floatCast(a.data[e0 + 1] * pc.f0) else 0;
    d.data[idx] = @bitCast(@as(u32, @as(u16, @bitCast(v0))) | (@as(u32, @as(u16, @bitCast(v1))) << 16));
}

// f32_to_h16_pad: strided variant for the f16-weight coop path (VAE convs):
// tight [rows][u1] f32 source rows become [*][u2] f16 rows (u2 >= u1, even),
// zero in the column tail and beyond u3 source rows — the coop GEMM needs
// k%64 and the im2col patch length is 9*ci, which usually isn't.
// a = f32 source, d = packed f16 pairs. u0 = out words, u1 = src cols,
// u2 = dst cols, u3 = src rows, f0 = scale.
export fn f32_to_h16_pad() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const row = e0 / pc.u2;
    const col = e0 % pc.u2; // u2 is even: e0+1 stays in the same row
    var out: u32 = 0;
    inline for (0..2) |j| {
        const cc = col + j;
        if (cc < pc.u1 and row < pc.u3) {
            const v: f16 = @floatCast(a.data[row * pc.u1 + cc] * pc.f0);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

// silu_mul_h16: fused SwiGLU gate + f16 conversion feeding the coop GEMM:
// d[word] = pack(f16(silu(a)*b*scale)) — skips the f32 intermediate the
// GEMM conversion would immediately re-read. Same push layout as
// f32_to_h16 (u0 = out words, u1 = real elems, f0 = scale).
export fn silu_mul_h16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    var out: u32 = 0;
    inline for (0..2) |j| {
        const e = e0 + j;
        if (e < pc.u1) {
            const g = a.data[e];
            const v: f16 = @floatCast(g / (1.0 + @exp(-g)) * b.data[e] * pc.f0);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

// sigmoid_mul_h16: fused attention gate + f16 conversion:
// d[word] = pack(f16(a * sigmoid(b) * scale)). Push as f32_to_h16.
export fn sigmoid_mul_h16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    var out: u32 = 0;
    inline for (0..2) |j| {
        const e = e0 + j;
        if (e < pc.u1) {
            const v: f16 = @floatCast(a.data[e] / (1.0 + @exp(-b.data[e])) * pc.f0);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

// --- f16-input variants for the f16-C-store coop GEMM path ----------------
// With f16 accumulators the coop GEMM's C values are exactly representable
// in f16, so these kernels read the half-precision C directly and compute
// in f32 — value-identical to the old f32-C + convert chain, at half the
// GEMM-output traffic.

// qknorm_rope16: fused per-head RMS norm + interleaved rope + output scale,
// in place on the f16 QKV GEMM output. One thread per (pos, head) row; the
// element pair of each u32 word is exactly one rope pair, and the operation
// order matches the rmsnorm + rope_inter + f32_to_h16 chain it replaces.
// a = q/k (f16 pair words, in place), b = norm weight, c = freqs (cos then
// sin halves). u0 = rows (seq*n_heads), u1 = half (dim/2), u2 = sin_off,
// u3 = n_heads, f0 = output scale, f1 = eps.
export fn qknorm_rope16() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const half = pc.u1;
    const pos = row / pc.u3;
    const base_w = row * half;
    var sum: f32 = 0;
    var w: u32 = 0;
    while (w < half) : (w += 1) {
        const word: u32 = @bitCast(a.data[base_w + w]);
        inline for (0..2) |j| {
            const hv: f16 = @bitCast(@as(u16, @truncate(word >> (16 * j))));
            const v: f32 = @floatCast(hv);
            sum += v * v;
        }
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(half * 2)) + pc.f1);
    w = 0;
    while (w < half) : (w += 1) {
        const word: u32 = @bitCast(a.data[base_w + w]);
        const h0: f16 = @bitCast(@as(u16, @truncate(word)));
        const h1: f16 = @bitCast(@as(u16, @truncate(word >> 16)));
        const x0 = @as(f32, @floatCast(h0)) * inv * b.data[w * 2];
        const x1 = @as(f32, @floatCast(h1)) * inv * b.data[w * 2 + 1];
        const cos_v = c.data[pos * half + w];
        const sin_v = c.data[pc.u2 + pos * half + w];
        const o0: f16 = @floatCast((x0 * cos_v - x1 * sin_v) * pc.f0);
        const o1: f16 = @floatCast((x0 * sin_v + x1 * cos_v) * pc.f0);
        a.data[base_w + w] = @bitCast(@as(u32, @as(u16, @bitCast(o0))) | (@as(u32, @as(u16, @bitCast(o1))) << 16));
    }
}

// gather_kmajor16: gather_kmajor_h16 with an f16 source — raw u16 moves, no
// conversion. a = f16 [seq][kv][hd] pair words, d = packed f16 pairs
// [kv][hd][s_stride]; positions >= seq write zero. u0 = out words, u1 = hd,
// u2 = s_stride (even), u3 = seq, u4 = n_kv_heads.
export fn gather_kmajor16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const hd = pc.u1;
    const sstride = pc.u2;
    const plane = hd * sstride;
    const h = e0 / plane;
    const rem = e0 % plane;
    const k = rem / sstride;
    const s0 = rem % sstride;
    var out: u32 = 0;
    inline for (0..2) |j| {
        const s = s0 + j;
        if (s < pc.u3) {
            const e = (s * pc.u4 + h) * hd + k;
            const word: u32 = @bitCast(a.data[e / 2]);
            const hv: u16 = @truncate(word >> @as(u5, @intCast((e & 1) * 16)));
            out |= @as(u32, hv) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

// silu_mul16: silu_mul_h16 with f16 gate/up inputs (the GEMM's zero pad
// rows pass through as silu(0)*0 = 0, so no bound is needed).
// a = gate words, b = up words, d = out words. u0 = words, f0 = scale.
export fn silu_mul16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const gw: u32 = @bitCast(a.data[idx]);
    const uw: u32 = @bitCast(b.data[idx]);
    var out: u32 = 0;
    inline for (0..2) |j| {
        const g: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(gw >> (16 * j))))));
        const u: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(uw >> (16 * j))))));
        const v: f16 = @floatCast(g / (1.0 + @exp(-g)) * u * pc.f0);
        out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
    }
    d.data[idx] = @bitCast(out);
}

// sigmoid_mul_g16: sigmoid_mul_h16 with an f16 gate. a = dst (f32, bound by
// u1 — attention pad rows are never read), b = gate words, d = out words.
// u0 = words, u1 = real elems, f0 = scale.
export fn sigmoid_mul_g16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const gw: u32 = @bitCast(b.data[idx]);
    var out: u32 = 0;
    inline for (0..2) |j| {
        const e = e0 + j;
        if (e < pc.u1) {
            const g: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(gw >> (16 * j))))));
            const v: f16 = @floatCast(a.data[e] / (1.0 + @exp(-g)) * pc.f0);
            out |= @as(u32, @as(u16, @bitCast(v))) << (16 * j);
        }
    }
    d.data[idx] = @bitCast(out);
}

// gated_add16: gated_add with an f16 delta (the wo / mlp.down GEMM output).
// a = x (f32, in place), b = delta words, c = vectors. u0 = words (n/2),
// u1 = dim, u2 = gate_off.
export fn gated_add16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const e0 = idx * 2;
    const w: u32 = @bitCast(b.data[idx]);
    inline for (0..2) |j| {
        const e = e0 + j;
        const col = e % pc.u1;
        const dv: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w >> (16 * j))))));
        a.data[e] += c.data[pc.u2 + col] * dv;
    }
}

// --- VAE decoder kernels ---------------------------------------------------

// vae_norm: per-position channel L2 norm (F.normalize * sqrt(c) * gamma, the
// Wan VAE convention), optionally fused with the silu that always follows it
// in the decoder. One thread per spatial position; activations are tight
// channel-last [n][c]. a = x, b = out, c = gamma.
// u0 = rows (positions), u1 = channels, u2 = 1 to apply silu.
export fn vae_norm() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const dim = pc.u1;
    const base = row * dim;
    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < dim) : (i += 1) {
        const v = a.data[base + i];
        sum += v * v;
    }
    const inv = @sqrt(@as(f32, @floatFromInt(dim))) / @max(@sqrt(sum), 1e-12);
    i = 0;
    while (i < dim) : (i += 1) {
        var v = a.data[base + i] * inv * c.data[i];
        if (pc.u2 != 0) v = v / (1.0 + @exp(-v));
        b.data[base + i] = v;
    }
}

// im2col: build the f32 patch matrix for a band of output positions of a
// zero-padded 3x3 conv over tight channel-last activations, so the conv is
// a plain GEMM (patches [bn][9*ci] @ W^T). With f0 != 0 the source is
// sampled nearest-exact 2x upsampled (coordinates halve) — the decoder's
// upsample+conv pairs never materialize the upsampled tensor. One thread
// per output f32. a = src [h*w][ci], d = patches.
// u0 = bn*patch_len threads, u1 = patch_len (9*ci), u2 = ci, u3 = src w,
// u4 = src h, u5 = first output position of the band, f0 = upsample flag.
export fn im2col() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const plen = pc.u1;
    const ci = pc.u2;
    const up: u5 = if (pc.f0 != 0) 1 else 0;
    const ow = pc.u3 << up;
    const oh = pc.u4 << up;
    const col = idx % plen;
    const p = pc.u5 + idx / plen;
    const tap = col / ci;
    const cc = col % ci;
    // Tap coordinate (y + ky - 1, x + kx - 1) via +1 offsets to stay
    // unsigned: valid iff 1 <= y + ky <= oh (same for x).
    const yk = p / ow + tap / 3;
    const xk = p % ow + tap % 3;
    var v: f32 = 0;
    if (yk >= 1 and yk <= oh and xk >= 1 and xk <= ow) {
        const sy = (yk - 1) >> up;
        const sx = (xk - 1) >> up;
        v = a.data[(sy * pc.u3 + sx) * ci + cc];
    }
    d.data[idx] = v;
}

// bias_compact: after the column-padded coop GEMM, strip the padding and
// add the conv bias in one pass: d[u3 + i] = a[(i/u1)*u2 + i%u1] + b[i%u1].
// One thread per real output element. a = padded GEMM out [*][u2],
// b = bias [u1], d = tight dst. u0 = positions*u1, u1 = co, u2 = n_pad,
// u3 = dst offset (elements).
export fn bias_compact() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const cc = idx % pc.u1;
    d.data[pc.u3 + idx] = a.data[(idx / pc.u1) * pc.u2 + cc] + b.data[cc];
}

const attn_chunk = 32;

export fn attention() callconv(.spirv_kernel) void {
    decorate();
    const seq = pc.u0;
    const n_heads = pc.u1;
    const n_kv = pc.u2;
    const hd = pc.u3;
    const chunks = hd / attn_chunk;
    const flat = gpu.global_invocation_id[0];
    const q_pos = gpu.global_invocation_id[1];
    if (flat >= n_heads * chunks or q_pos >= seq) return;
    const head = flat / chunks;
    const chunk = flat % chunks;
    const kv_head = head / (n_heads / n_kv);

    const q_stride = n_heads * hd;
    const kv_stride = n_kv * hd;
    const q_base = q_pos * q_stride + head * hd;
    const c_off = chunk * attn_chunk;

    var m: f32 = -3.4e38;
    var denom: f32 = 0;
    var acc: [attn_chunk]f32 = @splat(0.0);

    var j: u32 = 0;
    while (j < seq) : (j += 1) {
        const k_base = j * kv_stride + kv_head * hd;
        var dot: f32 = 0;
        var t: u32 = 0;
        while (t < hd) : (t += 1) {
            dot += a.data[q_base + t] * b.data[k_base + t];
        }
        const s = dot * pc.f0;
        const m_new = @max(m, s);
        const corr = @exp(m - m_new);
        const p = @exp(s - m_new);
        denom = denom * corr + p;
        const v_base = j * kv_stride + kv_head * hd + c_off;
        inline for (0..attn_chunk) |u| {
            acc[u] = acc[u] * corr + p * c.data[v_base + u];
        }
        m = m_new;
    }
    const inv = 1.0 / denom;
    const o_base = q_pos * q_stride + head * hd + c_off;
    inline for (0..attn_chunk) |u| {
        d.data[o_base + u] = acc[u] * inv;
    }
}
