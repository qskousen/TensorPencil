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

// rms_apply_w: y = x*inv[row]*w[col] — the plain-weight epilogue of the
//   3-pass parallel rmsnorm (rms_partial/rms_combine), for wide rows with a
//   norm weight but no AdaLN modulation (the LLM decode path's rows=1
//   hidden norms). a = x, b = out, c = weight, d = inv. u0 = n, u1 = dim.
export fn rms_apply_w() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    b.data[idx] = a.data[idx] * d.data[idx / pc.u1] * c.data[idx % pc.u1];
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
    // u4 = absolute position of row 0 (0 for the full-sequence encoder;
    // the KV-cached decode passes the cache length).
    const pos = idx / (half * pc.u3) + pc.u4;
    const row = idx / half; // pos*n_heads + head
    const cos_v = c.data[pos * half + i];
    const sin_v = c.data[pc.u2 + pos * half + i];
    const base = row * (2 * half);
    const lo = a.data[base + i];
    const hi = a.data[base + half + i];
    a.data[base + i] = lo * cos_v - hi * sin_v;
    a.data[base + half + i] = hi * cos_v + lo * sin_v;
}

// --- k-split GEMV (m=1 decode; the tiled GEMM leaves rows/8 threads) ------
// gemv_partial: y[col] = dot(W[:, col], x) split over u2 interleaved k
//   chunks; one thread per (chunk, 4-column group) so an fp8 thread reads a
//   whole u32 word per k (a warp touches 128 consecutive bytes — per-column
//   threads would touch 32 and waste 4x bandwidth). Weight is the k-major
//   transposed layout of the matmul kernels: element (k, col) at
//   k*w_stride + col; rows must be a multiple of 4. a = W (raw words through
//   the f32 view), b = x, d = partials [ch][rows]. u0 = (rows/4)*nchunk,
//   u1 = cols, u2 = nchunk, u3 = w_stride, u4 = is_f8, u5 = rows.
export fn gemv_partial() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const groups = pc.u5 / 4;
    const ch = idx / groups;
    const col0 = (idx % groups) * 4;
    var sums: [4]f32 = @splat(0.0);
    var k: u32 = ch;
    if (pc.u4 != 0) {
        while (k < pc.u1) : (k += pc.u2) {
            const word: u32 = @bitCast(a.data[(k * pc.u3 + col0) / 4]);
            const xv = b.data[k];
            inline for (0..4) |j| {
                sums[j] += e4m3ToF32((word >> (8 * j)) & 0xFF) * xv;
            }
        }
    } else {
        while (k < pc.u1) : (k += pc.u2) {
            const base = k * pc.u3 + col0;
            const xv = b.data[k];
            inline for (0..4) |j| {
                sums[j] += a.data[base + j] * xv;
            }
        }
    }
    const out = ch * pc.u5 + col0;
    inline for (0..4) |j| {
        d.data[out + j] = sums[j];
    }
}

// gemv_combine: y[u2 + col] = scale * sum_ch partials[ch][rows + col].
//   a = partials, d = y. u0 = rows, u1 = nchunk, u2 = dest element offset
//   (the chunked LM head), f0 = scale.
export fn gemv_combine() callconv(.spirv_kernel) void {
    decorate();
    const col = gpu.global_invocation_id[0];
    if (col >= pc.u0) return;
    var sum: f32 = 0;
    var ch: u32 = 0;
    while (ch < pc.u1) : (ch += 1) sum += a.data[ch * pc.u0 + col];
    d.data[pc.u2 + col] = sum * pc.f0;
}

// gemv_q8_0: y[row] = scale * dot(dequant(W[row]), x), ONE thread per output
//   row (no cross-thread reduction, so no workgroup memory). GGUF q8_0 row
//   layout is row-major blocks of 32: cols/32 blocks x 34 bytes =
//   [f16 d][32 x i8 qs]. Weight is uploaded RAW (weightBufferRaw, no k-major
//   transpose) and read through the u32 view of a. a = W bytes, b = x [cols],
//   d = y [rows]. u0 = rows, u1 = cols, f0 = scale.
export fn gemv_q8_0() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const nblk = pc.u1 / 32;
    const row_base = row * nblk * 34; // byte offset of this weight row
    var acc: f32 = 0;
    var blk: u32 = 0;
    while (blk < nblk) : (blk += 1) {
        const bb = row_base + blk * 34;
        const dword: u32 = @bitCast(a.data[bb / 4]);
        const dbits: u16 = if (bb % 4 == 0) @truncate(dword) else @truncate(dword >> 16);
        const sc: f32 = @floatCast(@as(f16, @bitCast(dbits)));
        var bsum: f32 = 0;
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            const bo = bb + 2 + i;
            const w: u32 = @bitCast(a.data[bo / 4]);
            const sh: u5 = @intCast(8 * (bo % 4));
            const ub: u32 = (w >> sh) & 0xFF;
            const q: i32 = @as(i32, @bitCast(ub << 24)) >> 24; // sign-extend low byte
            bsum += @as(f32, @floatFromInt(q)) * b.data[blk * 32 + i];
        }
        acc += sc * bsum;
    }
    d.data[pc.u2 + row] = acc * pc.f0;
}

// --- block-quant readers (weight buffer `a`, read through its u32 view) ---
inline fn wbyte(bo: u32) u32 {
    const word: u32 = @bitCast(a.data[bo / 4]);
    const sh: u5 = @intCast(8 * (bo % 4));
    return (word >> sh) & 0xFF;
}
inline fn wf16(bo: u32) f32 { // bo is 2-byte aligned
    const word: u32 = @bitCast(a.data[bo / 4]);
    const bits: u16 = if (bo % 4 == 0) @truncate(word) else @truncate(word >> 16);
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn wi8(bo: u32) i32 {
    return @as(i32, @bitCast(wbyte(bo) << 24)) >> 24; // sign-extend low byte
}

// q4_k / q5_k 6-bit sub-block scale+min unpack (ggml get_scale_min_k4).
// sbase = byte offset of the 12 packed scale bytes; j = sub-block 0..7.
const ScaleMin = struct { sc: u32, m: u32 };
inline fn scaleMinK4(sbase: u32, j: u32) ScaleMin {
    if (j < 4) {
        return .{ .sc = wbyte(sbase + j) & 63, .m = wbyte(sbase + j + 4) & 63 };
    }
    return .{
        .sc = (wbyte(sbase + j + 4) & 0x0F) | ((wbyte(sbase + j - 4) >> 6) << 4),
        .m = (wbyte(sbase + j + 4) >> 4) | ((wbyte(sbase + j) >> 6) << 4),
    };
}

// gemv_q4_k: y[row] = scale * dot(dequant(W[row]), x), one thread per row.
//   GGUF q4_k super-block (256 elems / 144 B): f16 d, f16 dmin, 12 B packed
//   sub-block scales/mins, 128 B low nibbles. v = d*sc*q - dmin*m (ggml
//   dequantize_row_q4_K element order). u0 = rows, u1 = cols, f0 = scale.
export fn gemv_q4_k() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 144;
    var acc: f32 = 0;
    var sb: u32 = 0;
    while (sb < nsb) : (sb += 1) {
        const bb = row_base + sb * 144;
        const sd = wf16(bb);
        const sdmin = wf16(bb + 2);
        const sbase = bb + 4;
        const qbase = bb + 16;
        const xb = sb * 256;
        var g: u32 = 0;
        while (g < 4) : (g += 1) {
            const s1 = scaleMinK4(sbase, 2 * g);
            const s2 = scaleMinK4(sbase, 2 * g + 1);
            const d1 = sd * @as(f32, @floatFromInt(s1.sc));
            const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
            const d2 = sd * @as(f32, @floatFromInt(s2.sc));
            const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
            const qg = qbase + g * 32;
            const xg = xb + g * 64;
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                const q = wbyte(qg + l);
                const wlo = d1 * @as(f32, @floatFromInt(q & 0xF)) - m1;
                const whi = d2 * @as(f32, @floatFromInt(q >> 4)) - m2;
                acc += wlo * b.data[xg + l];
                acc += whi * b.data[xg + 32 + l];
            }
        }
    }
    d.data[pc.u2 + row] = acc * pc.f0;
}

// gemv_q5_k: q4_k layout + 32 B of per-element 5th bits (qh) after the scales.
//   super-block 176 B: f16 d, f16 dmin, 12 B scales, 32 B qh, 128 B qs.
//   v = d*sc*(nibble + 16*bit) - dmin*m. u0 = rows, u1 = cols, f0 = scale.
export fn gemv_q5_k() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 176;
    var acc: f32 = 0;
    var sb: u32 = 0;
    while (sb < nsb) : (sb += 1) {
        const bb = row_base + sb * 176;
        const sd = wf16(bb);
        const sdmin = wf16(bb + 2);
        const sbase = bb + 4;
        const qhbase = bb + 16;
        const qbase = bb + 48;
        const xb = sb * 256;
        var g: u32 = 0;
        while (g < 4) : (g += 1) {
            const s1 = scaleMinK4(sbase, 2 * g);
            const s2 = scaleMinK4(sbase, 2 * g + 1);
            const d1 = sd * @as(f32, @floatFromInt(s1.sc));
            const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
            const d2 = sd * @as(f32, @floatFromInt(s2.sc));
            const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
            const qg = qbase + g * 32;
            const xg = xb + g * 64;
            const mlo: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g));
            const mhi: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g + 1));
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                const q = wbyte(qg + l);
                const qh = wbyte(qhbase + l);
                const lo: u32 = (q & 0xF) + (if (qh & mlo != 0) @as(u32, 16) else 0);
                const hi: u32 = (q >> 4) + (if (qh & mhi != 0) @as(u32, 16) else 0);
                acc += (d1 * @as(f32, @floatFromInt(lo)) - m1) * b.data[xg + l];
                acc += (d2 * @as(f32, @floatFromInt(hi)) - m2) * b.data[xg + 32 + l];
            }
        }
    }
    d.data[pc.u2 + row] = acc * pc.f0;
}

// gemv_q6_k: super-block 210 B / 256 elems: 128 B low nibbles (ql), 64 B high
//   2-bit pairs (qh), 16 x i8 sub-block scales, f16 d. v = d*sc*(q - 32),
//   16 sub-blocks of 16 (ggml dequantize_row_q6_K). u0=rows u1=cols f0=scale.
export fn gemv_q6_k() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 210;
    var acc: f32 = 0;
    var sb: u32 = 0;
    while (sb < nsb) : (sb += 1) {
        const bb = row_base + sb * 210;
        const sd = wf16(bb + 208);
        const xb = sb * 256;
        var half: u32 = 0;
        while (half < 2) : (half += 1) {
            const qlh = bb + half * 64; // ql
            const qhh = bb + 128 + half * 32; // qh
            const sch = bb + 192 + half * 8; // scales (i8)
            const xh = xb + half * 128;
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                const is = l / 16;
                const ql_l = wbyte(qlh + l);
                const ql_h = wbyte(qlh + l + 32);
                const qh = wbyte(qhh + l);
                const q1 = @as(i32, @intCast((ql_l & 0xF) | (((qh >> 0) & 3) << 4))) - 32;
                const q2 = @as(i32, @intCast((ql_h & 0xF) | (((qh >> 2) & 3) << 4))) - 32;
                const q3 = @as(i32, @intCast((ql_l >> 4) | (((qh >> 4) & 3) << 4))) - 32;
                const q4 = @as(i32, @intCast((ql_h >> 4) | (((qh >> 6) & 3) << 4))) - 32;
                const sc1 = wi8(sch + is + 0);
                const sc2 = wi8(sch + is + 2);
                const sc3 = wi8(sch + is + 4);
                const sc4 = wi8(sch + is + 6);
                acc += sd * @as(f32, @floatFromInt(sc1 * q1)) * b.data[xh + l];
                acc += sd * @as(f32, @floatFromInt(sc2 * q2)) * b.data[xh + l + 32];
                acc += sd * @as(f32, @floatFromInt(sc3 * q3)) * b.data[xh + l + 64];
                acc += sd * @as(f32, @floatFromInt(sc4 * q4)) * b.data[xh + l + 96];
            }
        }
    }
    d.data[pc.u2 + row] = acc * pc.f0;
}

// --- qwen35 hybrid (gated DeltaNet) kernels -----------------------------

// l2norm_rows: in-place per-row L2 normalize (ggml_l2_norm, clamped by eps).
//   a = x (in place). u0 = rows, u1 = dim, f0 = eps. One thread per row.
export fn l2norm_rows() callconv(.spirv_kernel) void {
    decorate();
    const r = gpu.global_invocation_id[0];
    if (r >= pc.u0) return;
    const dim = pc.u1;
    const base = r * dim;
    var ss: f32 = 0;
    var i: u32 = 0;
    while (i < dim) : (i += 1) {
        const v = a.data[base + i];
        ss += v * v;
    }
    const scale = 1.0 / @max(@sqrt(ss), pc.f0);
    i = 0;
    while (i < dim) : (i += 1) a.data[base + i] *= scale;
}

// deinterleave2: split per-head interleaved [q(hd) gate(hd)] into q and gate.
//   a = qg [heads*2*hd], c = q [heads*hd], d = gate [heads*hd]. u0 = heads*hd
//   (total output elems), u1 = hd. One thread per output element.
export fn deinterleave2() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const hd = pc.u1;
    const h = idx / hd;
    const j = idx % hd;
    c.data[idx] = a.data[h * 2 * hd + j];
    d.data[idx] = a.data[h * 2 * hd + hd + j];
}

// gdn_gates: per-head decay = exp(a * softplus(alpha + dt_bias)),
//   beta = sigmoid(beta_raw). a = [alpha(heads) | beta_raw(heads)],
//   b = a_dt [a(heads) | dt_bias(heads)], d = [decay(heads) | beta(heads)].
//   u0 = heads. One thread per head.
export fn gdn_gates() callconv(.spirv_kernel) void {
    decorate();
    const h = gpu.global_invocation_id[0];
    if (h >= pc.u0) return;
    const heads = pc.u0;
    const alpha = a.data[h];
    const beta_raw = a.data[heads + h];
    const av = b.data[h];
    const dt = b.data[heads + h];
    const s = alpha + dt;
    const sp = if (s > 20.0) s else @log(1.0 + @exp(s));
    d.data[h] = @exp(av * sp);
    d.data[heads + h] = 1.0 / (1.0 + @exp(-beta_raw));
}

// gdn_conv_step: per-channel depthwise causal conv over [state | current] +
//   SiLU; the 3-column state rolls forward. a = conv_state [ch*(taps-1)] (in
//   place), b = x [ch], c = conv_w [ch*taps], d = out [ch]. u0 = channels,
//   u1 = taps (4). One thread per channel.
export fn gdn_conv_step() callconv(.spirv_kernel) void {
    decorate();
    const ch = gpu.global_invocation_id[0];
    if (ch >= pc.u0) return;
    const taps = pc.u1;
    const stb = ch * (taps - 1);
    const wb = ch * taps;
    var acc = c.data[wb + taps - 1] * b.data[ch];
    var k: u32 = 0;
    while (k < taps - 1) : (k += 1) acc += c.data[wb + k] * a.data[stb + k];
    k = 0;
    while (k < taps - 2) : (k += 1) a.data[stb + k] = a.data[stb + k + 1];
    a.data[stb + taps - 2] = b.data[ch];
    d.data[ch] = acc / (1.0 + @exp(-acc));
}

// gdn_delta_step: per-v-head delta rule over a dd x dd state (ggml
//   build_delta_net_autoregressive, one token). a = state [heads*dd*dd] (in
//   place), b = conv_out [q(kheads*dd) | k(kheads*dd) | v(heads*dd)],
//   c = gates [decay(heads) | beta(heads)], d = o [heads*dd]. u0 = heads,
//   u1 = dd (<=128), u2 = kheads, f0 = readout scale. One thread per head.
export fn gdn_delta_step() callconv(.spirv_kernel) void {
    decorate();
    const h = gpu.global_invocation_id[0];
    if (h >= pc.u0) return;
    const heads = pc.u0;
    const dd = pc.u1;
    const kheads = pc.u2;
    const scale = pc.f0;
    const qkdim = kheads * dd;
    const qbase = (h % kheads) * dd;
    const kbase = qkdim + (h % kheads) * dd;
    const vbase = 2 * qkdim + h * dd;
    const sbase = h * dd * dd;
    const decay = c.data[h];
    const beta = c.data[heads + h];

    var m: [128]f32 = undefined;
    var j: u32 = 0;
    while (j < dd) : (j += 1) m[j] = 0;
    var i: u32 = 0;
    while (i < dd) : (i += 1) {
        const ki = b.data[kbase + i];
        const rb = sbase + i * dd;
        j = 0;
        while (j < dd) : (j += 1) {
            const sij = a.data[rb + j] * decay;
            a.data[rb + j] = sij;
            m[j] += sij * ki;
        }
    }
    var dl: [128]f32 = undefined;
    j = 0;
    while (j < dd) : (j += 1) dl[j] = (b.data[vbase + j] - m[j]) * beta;
    var o: [128]f32 = undefined;
    j = 0;
    while (j < dd) : (j += 1) o[j] = 0;
    i = 0;
    while (i < dd) : (i += 1) {
        const ki = b.data[kbase + i];
        const qi = b.data[qbase + i] * scale;
        const rb = sbase + i * dd;
        j = 0;
        while (j < dd) : (j += 1) {
            const sij = a.data[rb + j] + ki * dl[j];
            a.data[rb + j] = sij;
            o[j] += sij * qi;
        }
    }
    j = 0;
    while (j < dd) : (j += 1) d.data[h * dd + j] = o[j];
}

// rope_qwen35: partial rotate-half RoPE over the first rope_dim head dims,
//   in place (text-only decode: single position for all M-RoPE sections).
//   a = qk [n_heads*head_dim], c = freqs (cos then sin). u0 = n_heads*half
//   (total pairs), u1 = half (rope_dim/2), u2 = sin_off, u3 = head_dim,
//   u4 = pos. Matches ops.rope.applyRotateHalfPartialAt.
export fn rope_qwen35() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const half = pc.u1;
    const sin_off = pc.u2;
    const hd = pc.u3;
    const pos = pc.u4;
    const h = idx / half;
    const i = idx % half;
    const cosv = c.data[pos * half + i];
    const sinv = c.data[sin_off + pos * half + i];
    const base = h * hd + i;
    const lo = a.data[base];
    const hi = a.data[base + half];
    a.data[base] = lo * cosv - hi * sinv;
    a.data[base + half] = hi * cosv + lo * sinv;
}

// attn_decode_q35: causal GQA attention for one query (decode), one thread per
//   query head, online softmax. a = q [n_heads*hd], b = k_cache
//   [kv_len*kvDim], c = v_cache [kv_len*kvDim] (kvDim = n_kv*hd, position j at
//   j*kvDim + kvh*hd), d = out [n_heads*hd]. u0 = n_heads, u1 = n_kv_heads,
//   u2 = head_dim (<=256), u3 = kv_len, f0 = scale.
export fn attn_decode_q35() callconv(.spirv_kernel) void {
    decorate();
    const h = gpu.global_invocation_id[0];
    if (h >= pc.u0) return;
    const n_heads = pc.u0;
    const n_kv = pc.u1;
    const hd = pc.u2;
    const kv_len = pc.u3;
    const scale = pc.f0;
    const kvh = h / (n_heads / n_kv);
    const qb = h * hd;
    const kvdim = n_kv * hd;
    const kvbase = kvh * hd;
    var acc: [256]f32 = undefined;
    var t: u32 = 0;
    while (t < hd) : (t += 1) acc[t] = 0;
    var mx: f32 = -3.4e38;
    var denom: f32 = 0;
    var j: u32 = 0;
    while (j < kv_len) : (j += 1) {
        const kb = j * kvdim + kvbase;
        var sc: f32 = 0;
        t = 0;
        while (t < hd) : (t += 1) sc += a.data[qb + t] * b.data[kb + t];
        sc *= scale;
        const newmax = @max(mx, sc);
        const corr = @exp(mx - newmax);
        const p = @exp(sc - newmax);
        denom = denom * corr + p;
        t = 0;
        while (t < hd) : (t += 1) acc[t] = acc[t] * corr + p * c.data[kb + t];
        mx = newmax;
    }
    const inv = 1.0 / denom;
    t = 0;
    while (t < hd) : (t += 1) d.data[qb + t] = acc[t] * inv;
}

// gemv_partial4: gemv_partial for FOUR input vectors at once (speculative-
//   decode verify): one thread per (chunk, 8-column group) computes 32 dots,
//   reading each weight word once for all four inputs and each x value once
//   for eight columns. Per-(column, input) k order is identical to
//   gemv_partial (k = ch, stride nchunk), so results are bitwise equal to
//   four single-input GEMVs — greedy speculative decode stays byte-identical
//   to vanilla. x must have 4 rows of backing store past the offset (garbage
//   rows beyond the live count are discarded by gemv_combine4's n). rows
//   must be a multiple of 8. a = W (k-major), b = x, d = partials
//   [ch][4][rows]. u0 = (rows/8)*nchunk, u1 = cols, u2 = nchunk,
//   u3 = w_stride, u4 = is_f8, u5 = rows, f1 = x element offset (bitcast u32
//   — the input-group base for seq > 4 verifies).
export fn gemv_partial4() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u5;
    const cols = pc.u1;
    const x0: u32 = @bitCast(pc.f1);
    const groups = rows / 8;
    const ch = idx / groups;
    const col0 = (idx % groups) * 8;
    var sums: [4][8]f32 = @splat(@splat(0.0));
    var k: u32 = ch;
    if (pc.u4 != 0) {
        while (k < cols) : (k += pc.u2) {
            const base = (k * pc.u3 + col0) / 4;
            const w0: u32 = @bitCast(a.data[base]);
            const w1: u32 = @bitCast(a.data[base + 1]);
            inline for (0..4) |i| {
                const xv = b.data[x0 + i * cols + k];
                inline for (0..4) |j| {
                    sums[i][j] += e4m3ToF32((w0 >> (8 * j)) & 0xFF) * xv;
                    sums[i][4 + j] += e4m3ToF32((w1 >> (8 * j)) & 0xFF) * xv;
                }
            }
        }
    } else {
        while (k < cols) : (k += pc.u2) {
            const base = k * pc.u3 + col0;
            inline for (0..4) |i| {
                const xv = b.data[x0 + i * cols + k];
                inline for (0..8) |j| {
                    sums[i][j] += a.data[base + j] * xv;
                }
            }
        }
    }
    inline for (0..4) |i| {
        const out = (ch * 4 + i) * rows + col0;
        inline for (0..8) |j| {
            d.data[out + j] = sums[i][j];
        }
    }
}

// gemv_combine4: y[u2 + i*u3 + col] = scale * sum_ch partials[ch][i][col]
//   for the n live inputs — the reduce half of gemv_partial4. Same ascending
//   chunk order as gemv_combine (bitwise equal). a = partials [ch][4][rows],
//   d = y. u0 = rows, u1 = nchunk, u2 = dest element offset, u3 = dest row
//   stride (elements between consecutive inputs' outputs), u4 = n (1..4),
//   f0 = scale. Dispatch n * rows threads.
export fn gemv_combine4() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const rows = pc.u0;
    if (idx >= pc.u4 * rows) return;
    const i = idx / rows;
    const col = idx % rows;
    var sum: f32 = 0;
    var ch: u32 = 0;
    while (ch < pc.u1) : (ch += 1) sum += a.data[(ch * 4 + i) * rows + col];
    d.data[pc.u2 + i * pc.u3 + col] = sum * pc.f0;
}

// e4m3 -> f32, branchless (same as common.zig's; duplicated so this module
// stays free of common's buffer bindings).
inline fn e4m3ToF32(byte: u32) f32 {
    const man = byte & 0x7;
    const sign: u32 = (byte & 0x80) << 24;
    const magnitude = byte & 0x7F;
    const normal: f32 = @bitCast(sign | ((magnitude << 20) + (120 << 23)));
    const subnormal: f32 = @as(f32, @bitCast(sign | @as(u32, 0x3F800000))) *
        (@as(f32, @floatFromInt(man)) * 0x1p-9);
    return if (magnitude >= 8) normal else subnormal;
}

// --- flash-decoding attention (queries vs. the KV cache) ------------------
// attn_dsplit: pass 1 — one thread per (query, head, kv chunk): online
//   softmax over the chunk, unnormalized partial (m, d, acc[hd]) to scratch
//   at [idx*(hd+2)] ([t][h][i] order — attn_dmerge runs with heads' =
//   seq_q*heads). Queries are consecutive causal positions: query t sees
//   kv_len0 + t keys, so seq_q == 1 is plain decode and seq_q > 1 the
//   speculative-verify batch / multi-turn prefill chunk. Empty chunks write
//   (m=-3e38, d=0, acc=0), which the merge weights to zero.
//   a = q [seq_q][heads][hd], b = k [seq_kv][kv_dim], c = v, d = scratch.
//   u0=kv_len0, u1=heads, u2=kv_heads, u3=hd(<=128), u4=nsplit,
//   u5=seq_q (0 = 1), f0=scale.
export fn attn_dsplit() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const nsplit = pc.u4;
    const seq_q = @max(pc.u5, 1);
    if (idx >= seq_q * pc.u1 * nsplit) return;
    const hd = pc.u3;
    const per_q = pc.u1 * nsplit;
    const tq = idx / per_q;
    const h = (idx % per_q) / nsplit;
    const i = idx % nsplit;
    const kv_len = pc.u0 + tq; // causal: query tq's visible keys
    const chunk = (kv_len + nsplit - 1) / nsplit;
    const kv0 = i * chunk;
    const kv1 = @min(kv0 + chunk, kv_len);
    const kvh = h / (pc.u1 / pc.u2);
    const qbase = (tq * pc.u1 + h) * hd;

    var acc: [128]f32 = @splat(0.0); // type-level max; loops bound by hd
    var m: f32 = -3.0e38;
    var dsum: f32 = 0;
    var j = kv0;
    while (j < kv1) : (j += 1) {
        const kbase = (j * pc.u2 + kvh) * hd;
        var s: f32 = 0;
        var t: u32 = 0;
        while (t < hd) : (t += 1) s += a.data[qbase + t] * b.data[kbase + t];
        s *= pc.f0;
        const m2 = @max(m, s);
        const corr = @exp(m - m2);
        const p = @exp(s - m2);
        dsum = dsum * corr + p;
        m = m2;
        var t2: u32 = 0;
        while (t2 < hd) : (t2 += 1) acc[t2] = acc[t2] * corr + p * c.data[kbase + t2];
    }
    const base = idx * (hd + 2);
    d.data[base] = m;
    d.data[base + 1] = dsum;
    var t: u32 = 0;
    while (t < hd) : (t += 1) d.data[base + 2 + t] = acc[t];
}

// attn_dmerge: pass 2 — one thread per (head, dim c): M = max_i m_i,
//   D = sum_i d_i*exp(m_i-M), out[h][c] = sum_i acc_i[c]*exp(m_i-M) / D.
//   a = scratch (see attn_dsplit), d = out [heads][hd].
//   u0=heads, u1=hd, u2=nsplit.
export fn attn_dmerge() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const hd = pc.u1;
    if (idx >= pc.u0 * hd) return;
    const h = idx / hd;
    const ch = idx % hd;
    const nsplit = pc.u2;
    const stride = hd + 2;
    const base = h * nsplit * stride;
    var mx: f32 = -3.0e38;
    var i: u32 = 0;
    while (i < nsplit) : (i += 1) mx = @max(mx, a.data[base + i * stride]);
    var dsum: f32 = 0;
    var o: f32 = 0;
    i = 0;
    while (i < nsplit) : (i += 1) {
        const w = @exp(a.data[base + i * stride] - mx);
        dsum += a.data[base + i * stride + 1] * w;
        o += a.data[base + i * stride + 2 + ch] * w;
    }
    d.data[idx] = o / dsum;
}

// copy: b[u2 + idx] = a[u3 + idx]. Contiguous copy with destination and
// source element offsets; keeps tap snapshots / KV-cache appends / last-row
// extraction inside one batch (tensorCopy would flush it). u0 = element
// count, u2 = dest offset, u3 = src offset.
export fn copy() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    b.data[pc.u2 + idx] = a.data[pc.u3 + idx];
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

// qknorm_rope_f32: qknorm_rope16 entirely in f32, in place — fuses rmsnorm +
// rope (2 int8-path passes into 1). a = x (f32 [rows][hd], in place), b = norm
// weight, c = freqs. u0 = rows, u1 = half, u2 = sin_off, u3 = n_heads,
// f0 = scale, f1 = eps.
export fn qknorm_rope_f32() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const half = pc.u1;
    const pos = row / pc.u3;
    const base = row * half * 2;
    var sum: f32 = 0;
    var w: u32 = 0;
    while (w < half * 2) : (w += 1) {
        const v = a.data[base + w];
        sum += v * v;
    }
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(half * 2)) + pc.f1);
    w = 0;
    while (w < half) : (w += 1) {
        const x0 = a.data[base + w * 2] * inv * b.data[w * 2];
        const x1 = a.data[base + w * 2 + 1] * inv * b.data[w * 2 + 1];
        const cos_v = c.data[pos * half + w];
        const sin_v = c.data[pc.u2 + pos * half + w];
        a.data[base + w * 2] = (x0 * cos_v - x1 * sin_v) * pc.f0;
        a.data[base + w * 2 + 1] = (x0 * sin_v + x1 * cos_v) * pc.f0;
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

// --- int8 (convrot) activation prep + output scaling ----------------------
// The int8 tensor-core GEMM path quantizes activations dynamically per row
// after the group-Hadamard rotation that matches the pre-rotated weights.

// rotate: x_rot = x @ H per group_size (256) block along the columns (the
// ConvRot rotation, applied online to activations). One thread per output
// element; a = x, c = H (row-major [gs][gs]), b = x_rot. u0 = n (rows*cols),
// u1 = gs (256). Both cols and rows*cols are multiples of gs, so the group a
// column belongs to is just `idx - idx % gs`.
export fn rotate() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const gs = pc.u1;
    const local = idx % gs;
    const base = idx - local;
    var acc: f32 = 0;
    var l: u32 = 0;
    // H is symmetric (H[local][l] == H[l][local]); index as [l][local] so that
    // consecutive threads (consecutive `local`) read consecutive H addresses —
    // coalesced instead of stride-gs.
    while (l < gs) : (l += 1) acc += c.data[l * gs + local] * a.data[base + l];
    b.data[idx] = acc;
}

// rotate_fwht: same rotation as `rotate`, but one thread owns a whole 256
// group and runs the radix-4 fast Walsh-Hadamard (4 passes, strides 1/4/16/64,
// then /16) — ~16x fewer ops than the matvec, at the cost of a 256-f32 private
// array per thread. Also emits the group's abs-max as a partial (d), so the
// per-row quant scale becomes a cheap O(groups/row) reduction instead of a
// latency-bound O(cols) row scan. a = x, b = x_rot, d = partial abs-max per
// group. u0 = group count (n/256).
export fn rotate_fwht() callconv(.spirv_kernel) void {
    decorate();
    const g = gpu.global_invocation_id[0];
    if (g >= pc.u0) return;
    const base = g * 256;
    var v: [256]f32 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) v[i] = a.data[base + i];
    inline for (.{ 1, 4, 16, 64 }) |s| {
        var bb: u32 = 0;
        while (bb < 256) : (bb += s * 4) {
            var o: u32 = 0;
            while (o < s) : (o += 1) {
                const p = bb + o;
                const x0 = v[p];
                const x1 = v[p + s];
                const x2 = v[p + 2 * s];
                const x3 = v[p + 3 * s];
                v[p] = x0 + x1 + x2 - x3;
                v[p + s] = x0 + x1 - x2 + x3;
                v[p + 2 * s] = x0 - x1 + x2 + x3;
                v[p + 3 * s] = -x0 + x1 + x2 + x3;
            }
        }
    }
    var amax: f32 = 0;
    i = 0;
    while (i < 256) : (i += 1) {
        const r = v[i] / 16.0;
        b.data[base + i] = r;
        amax = @max(amax, @abs(r));
    }
    d.data[g] = amax;
}

// rowscale_i8: per-row int8 scale from the per-group partial abs-maxes.
// a = partials [rows*ng], b = scale [rows]. u0 = rows, u1 = ng (groups/row).
export fn rowscale_i8() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const ng = pc.u1;
    var amax: f32 = 0;
    var i: u32 = 0;
    while (i < ng) : (i += 1) amax = @max(amax, a.data[row * ng + i]);
    b.data[row] = @max(amax / 127.0, 1e-12);
}

// rowmax_i8: per-row dynamic int8 scale = max|x_rot[row]| / 127 (clamped off
// zero). One thread per row. a = x_rot, b = scale [rows]. u0 = rows, u1 = cols.
export fn rowmax_i8() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const cols = pc.u1;
    const base = row * cols;
    var amax: f32 = 0;
    var i: u32 = 0;
    while (i < cols) : (i += 1) amax = @max(amax, @abs(a.data[base + i]));
    b.data[row] = @max(amax / 127.0, 1e-12);
}

// quantize_i8: pack 4 int8 per u32 word, q = clamp(round(x_rot/scale[row]),
// -127, 127). a = x_rot, d = scale [rows], b = packed int8. u0 = words (n/4),
// u1 = cols, u2 = real element count (words past it — GEMM pad rows — zero).
export fn quantize_i8() callconv(.spirv_kernel) void {
    decorate();
    const w = gpu.global_invocation_id[0];
    if (w >= pc.u0) return;
    const e0 = w * 4;
    if (e0 >= pc.u2) {
        b.data[w] = 0; // padding row -> zeros
        return;
    }
    const inv = 1.0 / d.data[e0 / pc.u1];
    var out: u32 = 0;
    inline for (0..4) |j| {
        var qi: i32 = @intFromFloat(@round(a.data[e0 + j] * inv));
        qi = @max(@as(i32, -127), @min(@as(i32, 127), qi));
        out |= @as(u32, @as(u8, @bitCast(@as(i8, @intCast(qi))))) << @intCast(8 * j);
    }
    b.data[w] = @bitCast(out);
}

// scale_concat: assemble the fused int8 GEMM's scale buffer = [act(m_pad) |
// weight(rows)] in one dispatch (batch-barrier-safe, no transfer/compute race).
// a = act_scale (m entries on device), c = weight_scale (rows), b = concat out.
// u0 = m_pad + rows (total), u1 = m (act count), u2 = m_pad (weight base).
export fn scale_concat() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    if (idx < pc.u1) {
        b.data[idx] = a.data[idx];
    } else if (idx >= pc.u2) {
        b.data[idx] = c.data[idx - pc.u2];
    } else {
        b.data[idx] = 0; // act pad rows (m..m_pad); outputs discarded
    }
}

// scale_i32: y = int32_acc * act_scale[row] * weight_scale[col]. Reads the
// s32 GEMM output through the f32 view. a = acc (s32 bits), c = weight_scale
// [nout], d = act_scale [rows], b = y (f32). u0 = rows*nout, u1 = nout.
export fn scale_i32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const acc: i32 = @bitCast(a.data[idx]);
    b.data[idx] = @as(f32, @floatFromInt(acc)) * d.data[idx / pc.u1] * c.data[idx % pc.u1];
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
