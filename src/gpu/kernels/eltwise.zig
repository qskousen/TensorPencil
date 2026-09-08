//! Device-side elementwise, normalization and attention kernels for the GPU-resident
//! DiT forward. The plain elementwise ops live in dual.zig, which serves both arms. No workgroup memory anywhere, because the NVIDIA driver faults on
//! Zig-emitted workgroup-storage kernels, so all entries share this module.
//!
//! Universal binding layout (set 0): four storage buffers a, b, c, d whose
//! roles depend on the entry point; unused bindings get a dummy buffer.
//! Push constants: six u32 (u0..u5) + two f32 (f0, f1); meaning per entry.
//!
//! Entries (thread mapping / bindings):
//!   rmsnorm     x = row.               a=in, b=out, c=weight.
//!               u0=n_rows u1=dim f0=eps.

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
    // Appended after f0/f1 so existing kernels' field offsets are unchanged
    // (matches host EltPush). attn_dsplit_gemma uses it as the bidir kv_end.
    u6: u32,
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

// topk_reduce: L lanes each keep their M highest (value, index) over their
//   stride-slice, via min-slot tracking (strict `>` keeps the lower index on
//   ties). a=logits, c=out_val[L*M], d=out_idx[L*M] (index as exact f32).
//   u0=L, u1=vocab. The host does exact top-k over the L*M candidates, this
//   is a superset selection, exact unless a single lane holds >M of the global
//   top-k (astronomically unlikely for k<=512 over 1024 lanes). No workgroup
//   memory, no atomics.
const topk_m = 8; // must match context.zig topk_m
// Full non-causal attention, one thread per (query, head), arbitrary
// head_dim (unlike the 32-slice `attention` kernel). a=q, b=k, c=v, d=out,
// each [seq][n_heads*hd] (position j at j*n_kv*hd + kvh*hd). u0=seq,
// u1=n_heads, u2=n_kv_heads, u3=head_dim (<=256), f0=scale. Online softmax.
export fn attn_full() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const seq = pc.u0;
    const n_heads = pc.u1;
    if (idx >= seq * n_heads) return;
    const n_kv = pc.u2;
    const hd = pc.u3;
    const scale = pc.f0;
    const head = idx % n_heads;
    const kvh = head / (n_heads / n_kv);
    // u4/u5 = element base offsets into the q/out and k/v buffers (batched
    // ragged packing, one item's sub-region; 0 for the single-item path). q
    // and k/v may have different per-position strides (GQA), hence two offsets.
    const qoff = pc.u4;
    const kvoff = pc.u5;
    const qb = qoff + idx * hd;
    const kvdim = n_kv * hd;
    const kvbase = kvoff + kvh * hd;
    var acc: [256]f32 = undefined;
    var t: u32 = 0;
    while (t < hd) : (t += 1) acc[t] = 0;
    var mx: f32 = -3.4e38;
    var denom: f32 = 0;
    var j: u32 = 0;
    while (j < seq) : (j += 1) {
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

// Block-diagonal batched CAUSAL attention, `attn_full_batched` with the key
// range clipped to `[item_start, query]` instead of the whole item.
//
// This is CLIP's text tower: a causal LM body used as an encoder, so unlike every
// other encoder here it must not see forward. A non-causal CLIP still encodes
// every prompt and still renders every image, it is simply a different model, so
// there is no failure to observe, which is why this gets its own kernel rather than
// a flag on `attn_full_batched` that a caller could forget.
//
// The batch axis is the *prompt chunk*: a 77-token window gives only 77*heads
// threads, which does not fill a GPU, and a long prompt has two or three chunks
// that are independent by construction.
//
// u0=total rows, u1=n_heads, u2=n_kv, u3=hd, u4=s_rows (per-item length),
// f0=scale. a=q, b=k, c=v, d=out.
export fn attn_causal_batched() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const total = pc.u0;
    const n_heads = pc.u1;
    if (idx >= total * n_heads) return;
    const n_kv = pc.u2;
    const hd = pc.u3;
    const s_rows = pc.u4;
    const scale = pc.f0;
    const q_global = idx / n_heads;
    const head = idx % n_heads;
    const kvh = head / (n_heads / n_kv);
    const kv_start = (q_global / s_rows) * s_rows;
    // Inclusive of the query's own row, exclusive above it, the only difference
    // from the non-causal form, and the whole point of the kernel.
    const kv_end = q_global + 1;
    const qb = idx * hd;
    const kvdim = n_kv * hd;
    var acc: [256]f32 = undefined;
    var t: u32 = 0;
    while (t < hd) : (t += 1) acc[t] = 0;
    var mx: f32 = -3.4e38;
    var denom: f32 = 0;
    var j: u32 = kv_start;
    while (j < kv_end) : (j += 1) {
        const kb = j * kvdim + kvh * hd;
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

// --- k-split GEMV (m=1 decode; the tiled GEMM leaves rows/8 threads) ------
// gemv_partial: y[col] = dot(W[:, col], x) split over u2 interleaved k
//   chunks; one thread per (chunk, 4-column group) so an fp8 thread reads a
//   whole u32 word per k (a warp touches 128 consecutive bytes, per-column
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
    if (pc.u4 == 1) { // fp8: one aligned u32 = 4 columns
        while (k < pc.u1) : (k += pc.u2) {
            const word: u32 = @bitCast(a.data[(k * pc.u3 + col0) / 4]);
            const xv = b.data[k];
            inline for (0..4) |j| {
                sums[j] += e4m3ToF32((word >> (8 * j)) & 0xFF) * xv;
            }
        }
    } else if (pc.u4 == 2) { // bf16: one u32 = 2 columns (col0 even, stride even)
        while (k < pc.u1) : (k += pc.u2) {
            const e = k * pc.u3 + col0;
            const w0: u32 = @bitCast(a.data[e >> 1]);
            const w1: u32 = @bitCast(a.data[(e >> 1) + 1]);
            const xv = b.data[k];
            sums[0] += bf16ToF32(w0) * xv;
            sums[1] += bf16ToF32(w0 >> 16) * xv;
            sums[2] += bf16ToF32(w1) * xv;
            sums[3] += bf16ToF32(w1 >> 16) * xv;
        }
    } else { // f32
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

// --- transposed block-quant readers (32-row-group byte interleave) --------
// The `*_t` GEMV kernels read a weight repacked so logical byte `j` of row
// `row` lives at physical byte grp_base + j*32 (grp_base = (row/32)*row_bytes*32
// + row%32). A 32-lane warp = one full row-group, so at every read the warp
// touches 32 consecutive bytes = one coalesced transaction (the raw row-major
// layout has lanes row_bytes apart, ~1.7% of peak bandwidth). `gb` is the
// caller's grp_base; `j` is the logical byte within the row.
const GROUP = 32;
inline fn tbyte(gb: u32, j: u32) u32 {
    const p = gb + j * GROUP;
    const word: u32 = @bitCast(a.data[p / 4]);
    const sh: u5 = @intCast(8 * (p % 4));
    return (word >> sh) & 0xFF;
}
inline fn tf16(gb: u32, j: u32) f32 { // f16's two bytes are 32 B apart here
    const bits: u16 = @intCast(tbyte(gb, j) | (tbyte(gb, j + 1) << 8));
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn ti8(gb: u32, j: u32) i32 {
    return @as(i32, @bitCast(tbyte(gb, j) << 24)) >> 24; // sign-extend low byte
}
// scaleMinK4 over the transposed layout (logical byte offsets via tbyte).
inline fn scaleMinK4T(gb: u32, sbase: u32, j: u32) ScaleMin {
    if (j < 4) {
        return .{ .sc = tbyte(gb, sbase + j) & 63, .m = tbyte(gb, sbase + j + 4) & 63 };
    }
    return .{
        .sc = (tbyte(gb, sbase + j + 4) & 0x0F) | ((tbyte(gb, sbase + j - 4) >> 6) << 4),
        .m = (tbyte(gb, sbase + j + 4) >> 4) | ((tbyte(gb, sbase + j) >> 6) << 4),
    };
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

// gemv_q6_k_t: gemv_q6_k over the 32-row-group transposed weight layout
//   (see the transposed readers) AND k-split over superblocks. Thread
//   (row, ch) sums the contiguous superblock range [ch*chunk, (ch+1)*chunk)
//   into partials[ch*rows + row]; gemv_combine reduces over ch and applies
//   the scale. Transposed => a 32-lane warp (consecutive rows, same ch) reads
//   coalesced; split => rows*nchunk threads give the SM enough warps to hide
//   memory latency (one-thread-per-row is only ~1.4 warps/SM). Per-superblock
//   math is byte-identical to gemv_q6_k. a = W (transposed), b = x [cols],
//   d = partials [nchunk][rows]. u0 = rows*nchunk, u1 = cols, u2 = nchunk,
//   u3 = rows.
export fn gemv_q6_k_t() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nsb = pc.u1 / 256;
    const row_bytes = nsb * 210;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const chunk = (nsb + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nsb);
    var acc: f32 = 0;
    var sb: u32 = start;
    while (sb < stop) : (sb += 1) {
        const bb = sb * 210; // logical byte offset of this superblock in the row
        const sd = tf16(gb, bb + 208);
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
                const ql_l = tbyte(gb, qlh + l);
                const ql_h = tbyte(gb, qlh + l + 32);
                const qh = tbyte(gb, qhh + l);
                const q1 = @as(i32, @intCast((ql_l & 0xF) | (((qh >> 0) & 3) << 4))) - 32;
                const q2 = @as(i32, @intCast((ql_h & 0xF) | (((qh >> 2) & 3) << 4))) - 32;
                const q3 = @as(i32, @intCast((ql_l >> 4) | (((qh >> 4) & 3) << 4))) - 32;
                const q4 = @as(i32, @intCast((ql_h >> 4) | (((qh >> 6) & 3) << 4))) - 32;
                const sc1 = ti8(gb, sch + is + 0);
                const sc2 = ti8(gb, sch + is + 2);
                const sc3 = ti8(gb, sch + is + 4);
                const sc4 = ti8(gb, sch + is + 6);
                acc += sd * @as(f32, @floatFromInt(sc1 * q1)) * b.data[xh + l];
                acc += sd * @as(f32, @floatFromInt(sc2 * q2)) * b.data[xh + l + 32];
                acc += sd * @as(f32, @floatFromInt(sc3 * q3)) * b.data[xh + l + 64];
                acc += sd * @as(f32, @floatFromInt(sc4 * q4)) * b.data[xh + l + 96];
            }
        }
    }
    d.data[ch * rows + row] = acc;
}

// gemv_q8_0_t: transposed + block-split gemv_q8_0. Thread (row, ch) sums the
//   contiguous 32-elem-block range for its chunk into partials[ch*rows+row].
//   a = W (transposed), b = x, d = partials. u0=rows*nchunk u1=cols u2=nchunk
//   u3=rows.
export fn gemv_q8_0_t() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nblk = pc.u1 / 32;
    const row_bytes = nblk * 34;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const chunk = (nblk + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nblk);
    var acc: f32 = 0;
    var blk: u32 = start;
    while (blk < stop) : (blk += 1) {
        const bb = blk * 34; // logical byte offset of this block in the row
        const sc = tf16(gb, bb);
        var bsum: f32 = 0;
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            const q = ti8(gb, bb + 2 + i);
            bsum += @as(f32, @floatFromInt(q)) * b.data[blk * 32 + i];
        }
        acc += sc * bsum;
    }
    d.data[ch * rows + row] = acc;
}

// IQ4_NL non-linear 4-bit codebook (ggml kvalues_iq4nl). Same block layout as
// q4_0 (18 B/32 elems: f16 d + 16 nibble bytes) but v = d * kvalues[nibble].
const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

// gemv_iq4_nl: y[row] = scale * dot(dequant(W[row]), x), one thread per row.
//   Block j: low nibble of qs[i] -> element i, high nibble -> element i+16
//   (ggml dequantize_row_iq4_nl order). a = W bytes, b = x, d = y.
//   u0 = rows, u1 = cols, f0 = scale.
export fn gemv_iq4_nl() callconv(.spirv_kernel) void {
    decorate();
    const row = gpu.global_invocation_id[0];
    if (row >= pc.u0) return;
    const nblk = pc.u1 / 32;
    const row_base = row * nblk * 18;
    var acc: f32 = 0;
    var blk: u32 = 0;
    while (blk < nblk) : (blk += 1) {
        const bb = row_base + blk * 18;
        const sc = wf16(bb);
        var bsum: f32 = 0;
        var i: u32 = 0;
        while (i < 16) : (i += 1) {
            const q = wbyte(bb + 2 + i);
            const lo: f32 = @floatFromInt(kvalues_iq4nl[@intCast(q & 0xF)]);
            const hi: f32 = @floatFromInt(kvalues_iq4nl[@intCast(q >> 4)]);
            bsum += lo * b.data[blk * 32 + i];
            bsum += hi * b.data[blk * 32 + 16 + i];
        }
        acc += sc * bsum;
    }
    d.data[pc.u2 + row] = acc * pc.f0;
}

// gemv_iq4_nl_t: transposed + block-split gemv_iq4_nl (matches gemv_q8_0_t).
//   a = W (transposed), b = x, d = partials. u0=rows*nchunk u1=cols u2=nchunk u3=rows.
export fn gemv_iq4_nl_t() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nblk = pc.u1 / 32;
    const row_bytes = nblk * 18;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const chunk = (nblk + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nblk);
    var acc: f32 = 0;
    var blk: u32 = start;
    while (blk < stop) : (blk += 1) {
        const bb = blk * 18;
        const sc = tf16(gb, bb);
        var bsum: f32 = 0;
        var i: u32 = 0;
        while (i < 16) : (i += 1) {
            const q = tbyte(gb, bb + 2 + i);
            const lo: f32 = @floatFromInt(kvalues_iq4nl[@intCast(q & 0xF)]);
            const hi: f32 = @floatFromInt(kvalues_iq4nl[@intCast(q >> 4)]);
            bsum += lo * b.data[blk * 32 + i];
            bsum += hi * b.data[blk * 32 + 16 + i];
        }
        acc += sc * bsum;
    }
    d.data[ch * rows + row] = acc;
}

// gemv_q4_k_t: transposed + superblock-split gemv_q4_k. Same math as gemv_q4_k
//   (v = d*sc*q - dmin*m) over the transposed layout. u0=rows*nchunk u1=cols
//   u2=nchunk u3=rows.
export fn gemv_q4_k_t() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nsb = pc.u1 / 256;
    const row_bytes = nsb * 144;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const chunk = (nsb + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nsb);
    var acc: f32 = 0;
    var sb: u32 = start;
    while (sb < stop) : (sb += 1) {
        const bb = sb * 144;
        const sd = tf16(gb, bb);
        const sdmin = tf16(gb, bb + 2);
        const sbase = bb + 4;
        const qbase = bb + 16;
        const xb = sb * 256;
        var g: u32 = 0;
        while (g < 4) : (g += 1) {
            const s1 = scaleMinK4T(gb, sbase, 2 * g);
            const s2 = scaleMinK4T(gb, sbase, 2 * g + 1);
            const d1 = sd * @as(f32, @floatFromInt(s1.sc));
            const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
            const d2 = sd * @as(f32, @floatFromInt(s2.sc));
            const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
            const qg = qbase + g * 32;
            const xg = xb + g * 64;
            var l: u32 = 0;
            while (l < 32) : (l += 1) {
                const q = tbyte(gb, qg + l);
                const wlo = d1 * @as(f32, @floatFromInt(q & 0xF)) - m1;
                const whi = d2 * @as(f32, @floatFromInt(q >> 4)) - m2;
                acc += wlo * b.data[xg + l];
                acc += whi * b.data[xg + 32 + l];
            }
        }
    }
    d.data[ch * rows + row] = acc;
}

// gemv_q5_k_t: transposed + superblock-split gemv_q5_k (q4_k layout + 32 B qh
//   of 5th bits). u0=rows*nchunk u1=cols u2=nchunk u3=rows.
export fn gemv_q5_k_t() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nsb = pc.u1 / 256;
    const row_bytes = nsb * 176;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const chunk = (nsb + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nsb);
    var acc: f32 = 0;
    var sb: u32 = start;
    while (sb < stop) : (sb += 1) {
        const bb = sb * 176;
        const sd = tf16(gb, bb);
        const sdmin = tf16(gb, bb + 2);
        const sbase = bb + 4;
        const qhbase = bb + 16;
        const qbase = bb + 48;
        const xb = sb * 256;
        var g: u32 = 0;
        while (g < 4) : (g += 1) {
            const s1 = scaleMinK4T(gb, sbase, 2 * g);
            const s2 = scaleMinK4T(gb, sbase, 2 * g + 1);
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
                const q = tbyte(gb, qg + l);
                const qh = tbyte(gb, qhbase + l);
                const lo: u32 = (q & 0xF) + (if (qh & mlo != 0) @as(u32, 16) else 0);
                const hi: u32 = (q >> 4) + (if (qh & mhi != 0) @as(u32, 16) else 0);
                acc += (d1 * @as(f32, @floatFromInt(lo)) - m1) * b.data[xg + l];
                acc += (d2 * @as(f32, @floatFromInt(hi)) - m2) * b.data[xg + 32 + l];
            }
        }
    }
    d.data[ch * rows + row] = acc;
}

// --- block-quant dequant -> f32 row-major (prefill tensor-core GEMM path) ---
// These read the SAME 32-row-group transposed weight the gemv_*_t kernels read
// (so the resident decode weight is reused, no second copy) and write the
// dequantized weight in natural [rows][cols] row-major order (element (row,col)
// at row*cols + col; the value `col` would multiply in the GEMV). One thread
// per (row, block/superblock); idx = unit*rows + row so a 32-lane warp reads
// one coalesced row-group. u0 = rows*nunits, u1 = cols, u2 = rows, f0 = scale.
// a = W (transposed), d = f32 row-major out. The per-format dequant math is
// byte-identical to the matching gemv_*_t kernel; only the store differs.
// `pack_h16_kmajor` then repacks the f32 output into the k-major padded f16
// layout the coop GEMM consumes.

export fn dequant_q8_0_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nblk = cols / 32;
    const blk = idx / rows;
    const row = idx % rows;
    const row_bytes = nblk * 34;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const bb = blk * 34;
    const sc = tf16(gb, bb) * pc.f0;
    const obase = row * cols + blk * 32;
    var i: u32 = 0;
    while (i < 32) : (i += 1) {
        d.data[obase + i] = sc * @as(f32, @floatFromInt(ti8(gb, bb + 2 + i)));
    }
}

export fn dequant_iq4_nl_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nblk = cols / 32;
    const blk = idx / rows;
    const row = idx % rows;
    const row_bytes = nblk * 18;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const bb = blk * 18;
    const sc = tf16(gb, bb) * pc.f0;
    const obase = row * cols + blk * 32;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        const q = tbyte(gb, bb + 2 + i);
        d.data[obase + i] = sc * @as(f32, @floatFromInt(kvalues_iq4nl[@intCast(q & 0xF)]));
        d.data[obase + 16 + i] = sc * @as(f32, @floatFromInt(kvalues_iq4nl[@intCast(q >> 4)]));
    }
}

export fn dequant_q4_k_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nsb = cols / 256;
    const sb = idx / rows;
    const row = idx % rows;
    const row_bytes = nsb * 144;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const bb = sb * 144;
    const sd = tf16(gb, bb);
    const sdmin = tf16(gb, bb + 2);
    const sbase = bb + 4;
    const qbase = bb + 16;
    const obase = row * cols + sb * 256;
    var g: u32 = 0;
    while (g < 4) : (g += 1) {
        const s1 = scaleMinK4T(gb, sbase, 2 * g);
        const s2 = scaleMinK4T(gb, sbase, 2 * g + 1);
        const d1 = sd * @as(f32, @floatFromInt(s1.sc));
        const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
        const d2 = sd * @as(f32, @floatFromInt(s2.sc));
        const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
        const qg = qbase + g * 32;
        const og = obase + g * 64;
        var l: u32 = 0;
        while (l < 32) : (l += 1) {
            const q = tbyte(gb, qg + l);
            d.data[og + l] = (d1 * @as(f32, @floatFromInt(q & 0xF)) - m1) * pc.f0;
            d.data[og + 32 + l] = (d2 * @as(f32, @floatFromInt(q >> 4)) - m2) * pc.f0;
        }
    }
}

export fn dequant_q5_k_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nsb = cols / 256;
    const sb = idx / rows;
    const row = idx % rows;
    const row_bytes = nsb * 176;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const bb = sb * 176;
    const sd = tf16(gb, bb);
    const sdmin = tf16(gb, bb + 2);
    const sbase = bb + 4;
    const qhbase = bb + 16;
    const qbase = bb + 48;
    const obase = row * cols + sb * 256;
    var g: u32 = 0;
    while (g < 4) : (g += 1) {
        const s1 = scaleMinK4T(gb, sbase, 2 * g);
        const s2 = scaleMinK4T(gb, sbase, 2 * g + 1);
        const d1 = sd * @as(f32, @floatFromInt(s1.sc));
        const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
        const d2 = sd * @as(f32, @floatFromInt(s2.sc));
        const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
        const qg = qbase + g * 32;
        const og = obase + g * 64;
        const mlo: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g));
        const mhi: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g + 1));
        var l: u32 = 0;
        while (l < 32) : (l += 1) {
            const q = tbyte(gb, qg + l);
            const qh = tbyte(gb, qhbase + l);
            const lo: u32 = (q & 0xF) + (if (qh & mlo != 0) @as(u32, 16) else 0);
            const hi: u32 = (q >> 4) + (if (qh & mhi != 0) @as(u32, 16) else 0);
            d.data[og + l] = (d1 * @as(f32, @floatFromInt(lo)) - m1) * pc.f0;
            d.data[og + 32 + l] = (d2 * @as(f32, @floatFromInt(hi)) - m2) * pc.f0;
        }
    }
}

export fn dequant_q6_k_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nsb = cols / 256;
    const sb = idx / rows;
    const row = idx % rows;
    const row_bytes = nsb * 210;
    const gb = (row / GROUP) * (row_bytes * GROUP) + (row % GROUP);
    const bb = sb * 210;
    const sd = tf16(gb, bb + 208);
    const obase = row * cols + sb * 256;
    var half: u32 = 0;
    while (half < 2) : (half += 1) {
        const qlh = bb + half * 64;
        const qhh = bb + 128 + half * 32;
        const sch = bb + 192 + half * 8;
        const og = obase + half * 128;
        var l: u32 = 0;
        while (l < 32) : (l += 1) {
            const is = l / 16;
            const ql_l = tbyte(gb, qlh + l);
            const ql_h = tbyte(gb, qlh + l + 32);
            const qh = tbyte(gb, qhh + l);
            const q1 = @as(i32, @intCast((ql_l & 0xF) | (((qh >> 0) & 3) << 4))) - 32;
            const q2 = @as(i32, @intCast((ql_h & 0xF) | (((qh >> 2) & 3) << 4))) - 32;
            const q3 = @as(i32, @intCast((ql_l >> 4) | (((qh >> 4) & 3) << 4))) - 32;
            const q4 = @as(i32, @intCast((ql_h >> 4) | (((qh >> 6) & 3) << 4))) - 32;
            const sc1 = ti8(gb, sch + is + 0);
            const sc2 = ti8(gb, sch + is + 2);
            const sc3 = ti8(gb, sch + is + 4);
            const sc4 = ti8(gb, sch + is + 6);
            d.data[og + l] = sd * @as(f32, @floatFromInt(sc1 * q1)) * pc.f0;
            d.data[og + l + 32] = sd * @as(f32, @floatFromInt(sc2 * q2)) * pc.f0;
            d.data[og + l + 64] = sd * @as(f32, @floatFromInt(sc3 * q3)) * pc.f0;
            d.data[og + l + 96] = sd * @as(f32, @floatFromInt(sc4 * q4)) * pc.f0;
        }
    }
}

// --- qwen35 hybrid (gated DeltaNet) kernels -----------------------------

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

// gemv_partial4: gemv_partial for FOUR input vectors at once (speculative-
//   decode verify): one thread per (chunk, 8-column group) computes 32 dots,
//   reading each weight word once for all four inputs and each x value once
//   for eight columns. Per-(column, input) k order is identical to
//   gemv_partial (k = ch, stride nchunk), so results are bitwise equal to
//   four single-input GEMVs, greedy speculative decode stays byte-identical
//   to vanilla. x must have 4 rows of backing store past the offset (garbage
//   rows beyond the live count are discarded by gemv_combine4's n). rows
//   must be a multiple of 8. a = W (k-major), b = x, d = partials
//   [ch][4][rows]. u0 = (rows/8)*nchunk, u1 = cols, u2 = nchunk,
//   u3 = w_stride, u4 = is_f8, u5 = rows, f1 = x element offset (bitcast u32
//   the input-group base for seq > 4 verifies).
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
    if (pc.u4 == 1) { // fp8: two aligned u32 = 8 columns
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
    } else if (pc.u4 == 2) { // bf16: four u32 = 8 columns (col0, stride both even)
        while (k < cols) : (k += pc.u2) {
            const base = (k * pc.u3 + col0) >> 1;
            const w0: u32 = @bitCast(a.data[base]);
            const w1: u32 = @bitCast(a.data[base + 1]);
            const w2: u32 = @bitCast(a.data[base + 2]);
            const w3: u32 = @bitCast(a.data[base + 3]);
            inline for (0..4) |i| {
                const xv = b.data[x0 + i * cols + k];
                sums[i][0] += bf16ToF32(w0) * xv;
                sums[i][1] += bf16ToF32(w0 >> 16) * xv;
                sums[i][2] += bf16ToF32(w1) * xv;
                sums[i][3] += bf16ToF32(w1 >> 16) * xv;
                sums[i][4] += bf16ToF32(w2) * xv;
                sums[i][5] += bf16ToF32(w2 >> 16) * xv;
                sums[i][6] += bf16ToF32(w3) * xv;
                sums[i][7] += bf16ToF32(w3 >> 16) * xv;
            }
        }
    } else { // f32
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

// bf16 -> f32: bf16 IS the top 16 bits of an f32, so widening is a left shift
// of the (low-16) bits into the high half. `bits` must be a 16-bit value.
inline fn bf16ToF32(bits: u32) f32 {
    return @bitCast((bits & 0xFFFF) << 16);
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
// attn_dsplit: pass 1, one thread per (query, head, kv chunk): online
//   softmax over the chunk, unnormalized partial (m, d, acc[hd]) to scratch
//   at [idx*(hd+2)] ([t][h][i] order, attn_dmerge runs with heads' =
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

// attn_dmerge: pass 2, one thread per (head, dim c): M = max_i m_i,
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

// attn_dsplit_gemma: flash-decoding split for ONE query (decode), hd<=256,
//   with sliding-window + ring addressing (gemma3 LOCAL layers) and GQA. One
//   thread per (head, kv chunk): online softmax over its chunk of the visible
//   window, unnormalized partial (m, d, acc[hd]) to scratch at [idx*(hd+2)]
//   ([h][i] order, attn_dmerge runs with heads' = heads). Pairs with
//   attn_dmerge (hd-agnostic). Parallelizes the old one-thread-per-head
//   attn_decode_q35 by `nsplit`.
//   a = q [heads][hd], b = k_cache, c = v_cache, d = scratch.
//   u0=kv_len (pos+1), u1=heads, u2=kv_heads, u3=hd(<=256), u4=nsplit,
//   u5=window (0 = full causal), f0=scale, f1=ring (bitcast u32; 0 = linear),
//   u6=kv_end (bidirectional image block upper bound; 0 = causal = kv_len). The
//   window kv_start always tracks the query's OWN causal position (kv_len).
export fn attn_dsplit_gemma() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const nsplit = pc.u4;
    const heads = pc.u1;
    if (idx >= heads * nsplit) return;
    const hd = pc.u3;
    const kv_len = pc.u0;
    const kv_heads = pc.u2;
    const window = pc.u5;
    const ring: u32 = @bitCast(pc.f1);
    const scale = pc.f0;
    // Bidirectional image block extends the upper bound forward to the whole
    // block (u6); default (0) keeps the causal bound. kv_start stays on kv_len.
    const kv_end = if (pc.u6 != 0) pc.u6 else kv_len;
    const h = idx / nsplit;
    const i = idx % nsplit;
    const kvh = h / (heads / kv_heads);
    const kv_start: u32 = if (window != 0 and kv_len > window) kv_len - window else 0;
    const span = kv_end - kv_start;
    const chunk = (span + nsplit - 1) / nsplit;
    const kv0 = kv_start + i * chunk;
    const kv1 = @min(kv0 + chunk, kv_end);
    const qbase = h * hd;
    var acc: [256]f32 = @splat(0.0); // type-level max; loops bound by hd
    var m: f32 = -3.0e38;
    var dsum: f32 = 0;
    var j = kv0;
    while (j < kv1) : (j += 1) {
        const row = if (ring != 0) j % ring else j;
        const kbase = (row * kv_heads + kvh) * hd;
        var s: f32 = 0;
        var t: u32 = 0;
        while (t < hd) : (t += 1) s += a.data[qbase + t] * b.data[kbase + t];
        s *= scale;
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

// f16 K/V readers: the K (b) / V (c) caches hold f16, two per f32 slot. Element
// e lives in slot e/2, low half if e even else high (little-endian). Mirrors
// `wf16` (buffer a) for the K/V bindings, used by attn_dsplit_gemma_f16.
inline fn kf16(e: u32) f32 {
    const word: u32 = @bitCast(b.data[e / 2]);
    const bits: u16 = if (e % 2 == 0) @truncate(word) else @truncate(word >> 16);
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn vf16(e: u32) f32 {
    const word: u32 = @bitCast(c.data[e / 2]);
    const bits: u16 = if (e % 2 == 0) @truncate(word) else @truncate(word >> 16);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// attn_dsplit_gemma_f16: f16-KV variant of attn_dsplit_gemma. Identical online
// softmax + sliding-window + ring, but K/V are read from the f16-packed caches
// (kf16/vf16) and widened to f32; Q (a) and scratch (d) stay f32. Params match
// attn_dsplit_gemma exactly. Lossy vs f32.
export fn attn_dsplit_gemma_f16() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const nsplit = pc.u4;
    const heads = pc.u1;
    if (idx >= heads * nsplit) return;
    const hd = pc.u3;
    const kv_len = pc.u0;
    const kv_heads = pc.u2;
    const window = pc.u5;
    const ring: u32 = @bitCast(pc.f1);
    const scale = pc.f0;
    // Bidirectional image block (u6, 0 = causal); kv_start stays on kv_len.
    const kv_end = if (pc.u6 != 0) pc.u6 else kv_len;
    const h = idx / nsplit;
    const i = idx % nsplit;
    const kvh = h / (heads / kv_heads);
    const kv_start: u32 = if (window != 0 and kv_len > window) kv_len - window else 0;
    const span = kv_end - kv_start;
    const chunk = (span + nsplit - 1) / nsplit;
    const kv0 = kv_start + i * chunk;
    const kv1 = @min(kv0 + chunk, kv_end);
    const qbase = h * hd;
    var acc: [256]f32 = @splat(0.0); // type-level max; loops bound by hd
    var m: f32 = -3.0e38;
    var dsum: f32 = 0;
    var j = kv0;
    while (j < kv1) : (j += 1) {
        const row = if (ring != 0) j % ring else j;
        const kbase = (row * kv_heads + kvh) * hd;
        var s: f32 = 0;
        var t: u32 = 0;
        while (t < hd) : (t += 1) s += a.data[qbase + t] * kf16(kbase + t);
        s *= scale;
        const m2 = @max(m, s);
        const corr = @exp(m - m2);
        const p = @exp(s - m2);
        dsum = dsum * corr + p;
        m = m2;
        var t2: u32 = 0;
        while (t2 < hd) : (t2 += 1) acc[t2] = acc[t2] * corr + p * vf16(kbase + t2);
    }
    const base = idx * (hd + 2);
    d.data[base] = m;
    d.data[base + 1] = dsum;
    var t: u32 = 0;
    while (t < hd) : (t += 1) d.data[base + 2 + t] = acc[t];
}

// q8_0 K/V readers: the K (b) / V (c) caches hold ggml block_q8_0 (34 bytes per
// 32 elements: f16 scale d at +0, 32 x i8 quants at +2), byte-packed into the
// f32 words. Element e's block starts at byte (e/32)*34 (2-byte aligned; the
// quant byte is arbitrary). Split into scale/quant readers so the attention
// loop hoists the scale once per 32-aligned chunk. value = quant * scale.
inline fn kq8s(e: u32) f32 {
    const boff = (e / 32) * 34;
    const w: u32 = @bitCast(b.data[boff / 4]);
    const bits: u16 = if (boff % 4 == 0) @truncate(w) else @truncate(w >> 16);
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn kq8q(e: u32) f32 {
    const qoff = (e / 32) * 34 + 2 + (e % 32);
    const w: u32 = @bitCast(b.data[qoff / 4]);
    const sh: u5 = @intCast(24 - (qoff % 4) * 8);
    return @floatFromInt(@as(i32, @bitCast(w << sh)) >> 24);
}
inline fn vq8s(e: u32) f32 {
    const boff = (e / 32) * 34;
    const w: u32 = @bitCast(c.data[boff / 4]);
    const bits: u16 = if (boff % 4 == 0) @truncate(w) else @truncate(w >> 16);
    return @floatCast(@as(f16, @bitCast(bits)));
}
inline fn vq8q(e: u32) f32 {
    const qoff = (e / 32) * 34 + 2 + (e % 32);
    const w: u32 = @bitCast(c.data[qoff / 4]);
    const sh: u5 = @intCast(24 - (qoff % 4) * 8);
    return @floatFromInt(@as(i32, @bitCast(w << sh)) >> 24);
}

// attn_dsplit_gemma_q8: q8_0-KV variant of attn_dsplit_gemma. Identical online
// softmax + sliding-window + ring, but K/V are read from the q8_0 block caches
// (kq8*/vq8*) and widened to f32, the row base is a block multiple (kv_dim is),
// so 32-wide chunks hoist each block's scale once. Q (a) and scratch (d) stay
// f32. Params match attn_dsplit_gemma exactly. Lossy vs f32.
export fn attn_dsplit_gemma_q8() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const nsplit = pc.u4;
    const heads = pc.u1;
    if (idx >= heads * nsplit) return;
    const hd = pc.u3;
    const kv_len = pc.u0;
    const kv_heads = pc.u2;
    const window = pc.u5;
    const ring: u32 = @bitCast(pc.f1);
    const scale = pc.f0;
    // Bidirectional image block (u6, 0 = causal); kv_start stays on kv_len.
    const kv_end = if (pc.u6 != 0) pc.u6 else kv_len;
    const h = idx / nsplit;
    const i = idx % nsplit;
    const kvh = h / (heads / kv_heads);
    const kv_start: u32 = if (window != 0 and kv_len > window) kv_len - window else 0;
    const span = kv_end - kv_start;
    const chunk = (span + nsplit - 1) / nsplit;
    const kv0 = kv_start + i * chunk;
    const kv1 = @min(kv0 + chunk, kv_end);
    const qbase = h * hd;
    var acc: [256]f32 = @splat(0.0); // type-level max; loops bound by hd
    var m: f32 = -3.0e38;
    var dsum: f32 = 0;
    var j = kv0;
    while (j < kv1) : (j += 1) {
        const row = if (ring != 0) j % ring else j;
        const kbase = (row * kv_heads + kvh) * hd;
        var s: f32 = 0;
        var t: u32 = 0;
        while (t < hd) : (t += 32) {
            const dk = kq8s(kbase + t);
            var u: u32 = 0;
            while (u < 32) : (u += 1) s += a.data[qbase + t + u] * (kq8q(kbase + t + u) * dk);
        }
        s *= scale;
        const m2 = @max(m, s);
        const corr = @exp(m - m2);
        const p = @exp(s - m2);
        dsum = dsum * corr + p;
        m = m2;
        var t2: u32 = 0;
        while (t2 < hd) : (t2 += 32) {
            const dv = vq8s(kbase + t2);
            var u: u32 = 0;
            while (u < 32) : (u += 1) acc[t2 + u] = acc[t2 + u] * corr + p * (vq8q(kbase + t2 + u) * dv);
        }
    }
    const base = idx * (hd + 2);
    d.data[base] = m;
    d.data[base + 1] = dsum;
    var t: u32 = 0;
    while (t < hd) : (t += 1) d.data[base + 2 + t] = acc[t];
}

// kv_store_q8_0: quantize-store f32 K/V into a q8_0 block cache. One thread per
// PAIR of 32-element blocks (68 bytes = 17 whole u32 words), because a single
// 34-byte block ends mid-word and adjacent threads would race on the shared
// word; kv_dim is a multiple of 64, so rows always hold whole pairs. Reads
// a.data[u3 + idx*64 ..][64] (f32), writes 17 words at b word (u2/64)*17 +
// idx*17. Per block: d = absmax/127 (f16), q = roundEven(x/d), same
// round-to-nearest-even as the host packQ80 and the CUDA cvt.rni kernels.
// u0 = pair count, u2 = dst element offset (pair-aligned), u3 = src elem off.
export fn kv_store_q8_0() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const src = pc.u3 + idx * 64;
    // Byte staging as u32 (one byte value per slot): the SPIR-V path avoids
    // 8-bit types everywhere (no Int8 capability), like the gemv_q8_0 readers.
    var bytes: [68]u32 = undefined;
    var blk: u32 = 0;
    while (blk < 2) : (blk += 1) {
        const s0 = src + blk * 32;
        const o = blk * 34;
        var amax: f32 = 0;
        var i: u32 = 0;
        while (i < 32) : (i += 1) amax = @max(amax, @abs(a.data[s0 + i]));
        const dq = amax / 127.0;
        const id: f32 = if (dq != 0) 1.0 / dq else 0.0;
        const dbits: u32 = @as(u16, @bitCast(@as(f16, @floatCast(dq))));
        bytes[o] = dbits & 0xff;
        bytes[o + 1] = dbits >> 8;
        i = 0;
        while (i < 32) : (i += 1) {
            // Round-to-nearest-even from exact primitives (floor and the
            // fraction compare are exact for |v| <= 127.5): the 2^23 magic-add
            // trick gets folded away by GPU shader compilers, leaving trunc.
            const v = a.data[s0 + i] * id;
            const fl = @floor(v);
            const fr = v - fl;
            var q: i32 = @intFromFloat(fl);
            if (fr > 0.5) {
                q += 1;
            } else if (fr == 0.5) {
                q += q & 1; // ties to even: bump odd down-rounded values up
            }
            bytes[o + 2 + i] = @as(u32, @bitCast(q)) & 0xff;
        }
    }
    const dst_word = (pc.u2 / 64) * 17 + idx * 17;
    var w: u32 = 0;
    while (w < 17) : (w += 1) {
        const p4 = w * 4;
        const word = bytes[p4] | (bytes[p4 + 1] << 8) | (bytes[p4 + 2] << 16) | (bytes[p4 + 3] << 24);
        b.data[dst_word + w] = @bitCast(word);
    }
}

// --- f16 ACTIVATION STORAGE (VAE decode) ------------------------------------
//
// Storage-format twins, NOT new maths: each reads and/or writes the big VAE
// activation buffers as f16 while computing in f32 exactly as its f32 original a
// few lines away. A 16-channel VAE decoding 1056x1584 holds 256 channels at FULL
// image resolution (428M floats, 1.71 GB) in each of two buffers; f32 storage made
// a whole-image decode cost 4.3 GB of activations. `sd_vae.Config.act_f16` gates
// them, on RANGE not precision, SDXL's residual reaches 4.2e5, past f16.
//
// f16 rides as u32 PAIRS, because every storage buffer here is `[*]f32`:
// element `i` is half `i & 1` of word `i >> 1`. Reads may therefore be
// per-element, but a WRITE must own the whole word or two threads race on it, so
// every store kernel below indexes PAIRS and does two elements per thread. That is
// the one structural difference from the CUDA twins, which have real `b16` stores.
//
// Each takes its f32 original's push layout EXACTLY (Vulkan's, which is not
// CUDA's, `im2col_sd` carries the source width in u3 and the output width in u6,
// and `bias_compact` carries a bias offset in u4 and `act_div` in f0). A first
// attempt ported the PTX conventions instead; it compiled and would have been
// silently wrong.

inline fn ldH16(buf: *addrspace(.storage_buffer) FBuf, i: u32) f32 {
    const w: u32 = @bitCast(buf.data[i >> 1]);
    const sh: u5 = @intCast((i & 1) * 16);
    return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w >> sh)))));
}

inline fn packH16(lo: f32, hi: f32) f32 {
    const a16: u16 = @bitCast(@as(f16, @floatCast(lo)));
    const b16: u16 = @bitCast(@as(f16, @floatCast(hi)));
    return @bitCast(@as(u32, a16) | (@as(u32, b16) << 16));
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

export fn attn_scores() callconv(.spirv_kernel) void {
    decorate();
    const seq = pc.u0;
    const hd = pc.u3;
    const head = pc.u4 + gpu.global_invocation_id[2];
    const kv_head = head / (pc.u1 / pc.u2);
    const j0 = gpu.global_invocation_id[0] * amm_tile;
    const q0 = gpu.global_invocation_id[1] * amm_tile;
    // Query banding (u6 != 0): compute rows [u5, u5+u6) of the scores plane into a
    // band-local buffer of `u6` rows. u6 == 0 is the whole plane, exactly as before,
    // every pre-existing caller passes 0 for both, so their behaviour is unchanged
    // by construction. Exists because the VAE mid-block attends over EVERY latent
    // position: at a 132x198 latent that is seq = 26,136 and a full f32 plane is
    // 2.73 GB, which dwarfs the decode it belongs to.
    const qb = if (pc.u6 == 0) seq else pc.u6;
    const qoff = pc.u5;
    if (j0 >= seq or q0 >= qb) return;

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
            qv[i] = a.data[qrow0 + k * seq + qoff + q0 + i];
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
        // Band-local: `q` indexes the band, `qoff + q` the real query. The row
        // stride stays `seq` (a whole key row) and only the PLANE shrinks to `qb`.
        if (q < qb and qoff + q < seq) {
            inline for (0..amm_tile) |jj| {
                const j = j0 + jj;
                if (j < seq) {
                    d.data[(z * qb + q) * seq + j] = acc[i][jj] * pc.f0;
                }
            }
        }
    }
}

export fn attn_out() callconv(.spirv_kernel) void {
    decorate();
    const seq = pc.u0;
    const hd = pc.u3;
    const head = pc.u4 + gpu.global_invocation_id[2];
    const kv_head = head / (pc.u1 / pc.u2);
    const c0 = gpu.global_invocation_id[0] * amm_tile;
    const q0 = gpu.global_invocation_id[1] * amm_tile;
    // `u6` carries the query-band offset plus one, so that 0 keeps meaning
    // "not banded", the kernel has exactly one free push slot and needs to
    // distinguish "band starting at row 0" from "no banding". See `attn_scores`.
    const banded = pc.u6 != 0;
    const qoff = if (banded) pc.u6 - 1 else 0;
    if (c0 >= hd or (!banded and q0 >= seq) or qoff + q0 >= seq) return;

    // Scores layout: row stride u5, per-head plane stride in f0 (u32 bits),
    // the tensor-core scores path pads both to multiples of 128.
    const z = gpu.global_invocation_id[2];
    const kv_stride = pc.u2 * hd;
    const s_max = seq - 1;
    const sstr = pc.u5;
    const splane: u32 = @bitCast(pc.f0);
    var pb: [amm_tile]u32 = undefined;
    inline for (0..amm_tile) |i| {
        // Banded: rows are band-local and the caller sizes the band buffer to a whole
        // multiple of the tile, so `q0 + i` is always in range and needs no clamp.
        const row: u32 = if (banded) q0 + @as(u32, @intCast(i)) else @min(q0 + @as(u32, @intCast(i)), s_max);
        pb[i] = z * splane + row * sstr;
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
            if (causal == 0 or j <= qoff + q0 + i) {
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
        // Banded: the OUTPUT row is the real query, `qoff + q0 + i`.
        const q = qoff + q0 + i;
        if (q < seq) {
            const inv = 1.0 / denom[i];
            inline for (0..amm_tile) |u| {
                d.data[q * q_stride + head * hd + c0 + u] = acc[i][u] * inv;
            }
        }
    }
}

// --- f16-input variants for the f16-C-store coop GEMM path ----------------
// With f16 accumulators the coop GEMM's C values are exactly representable
// in f16, so these kernels read the half-precision C directly and compute
// in f32, value-identical to the old f32-C + convert chain, at half the
// GEMM-output traffic.

// --- int8 (convrot) activation prep + output scaling ----------------------
// The int8 tensor-core GEMM path quantizes activations dynamically per row
// after the group-Hadamard rotation that matches the pre-rotated weights.

// rotate_fwht: same rotation as `rotate`, but one thread owns a whole 256
// group and runs the radix-4 fast Walsh-Hadamard (4 passes, strides 1/4/16/64,
// then /16), ~16x fewer ops than the matvec, at the cost of a 256-f32 private
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

// w4a8_decode_t: decode a packed ComfyUI `asym_w4a8_int8` weight into the k-major int8
// layout the coop GEMM reads. The packed form stays resident and this runs PER GEMM, a
// decoded krea2 is 12.2 GB of int8 against 6.1 GB packed, which is the entire difference
// between this format and int8 (see ops/w4a8.zig). `levels` is the [256][16] table the
// host built, so this kernel does no arithmetic at all and is bit-identical to the CPU
// decode by construction rather than by agreement.
//
// Both inputs arrive ALREADY BYTE-TRANSPOSED, through the same cached `weightBuffer`
// path every other weight uses, and that is what makes this kernel fast rather than
// merely correct. Transposing inside the decode instead measures 1757 ms/step against a
// ~20 ms bandwidth roof: a transpose has one unavoidably strided side, so a warp's 32
// lanes hit 32 separate sectors per load instruction, and paying that PER GEMM rather
// than once at load is the whole 50x. Transposing the PACKED bytes at load costs the same
// one-time pass every dense weight already pays and leaves every stream here coalesced.
//
// Layout, with `stride = align(rows, tile_n)` shared by both inputs and the output:
//   a  = pT   packed,  [cols/2][stride], byte (j, r) holds columns 2j, 2j+1 of row r
//   c  = sT   s_rel,   [cols/group_size][stride] fp8 bytes
//   b  = out  int8,    [cols][stride]
// One thread owns 4 consecutive rows of one packed byte-column: it reads ONE u32 of `a`
// (at index `t` exactly, `j*(stride/4) + rq` is the thread id) and one u32 of `c`, and
// writes one u32 into each of the two output planes 2j and 2j+1. Adjacent threads hold
// adjacent row quads, so every read and both writes are fully coalesced.
//
// The row PADDING needs no special case: `weightBuffer` zeroes it in both inputs, and a
// zero scale byte is fp8 0.0, whose level table row is all zeros, so a pad row decodes
// to 0, which is what the GEMM must see.
//
// u0 = total threads = (cols/2)*(stride/4), u1 = group_size/2 (packed byte-columns per
// group), u2 = stride/4.
export fn w4a8_decode_t() callconv(.spirv_kernel) void {
    decorate();
    const t = gpu.global_invocation_id[0];
    if (t >= pc.u0) return;
    const rq = pc.u2; // row quads per byte-column plane
    const j = t / rq; // packed byte-column
    const r = t % rq; // row quad within it
    const pw: u32 = @bitCast(a.data[t]);
    const sw: u32 = @bitCast(c.data[(j / pc.u1) * rq + r]);
    var lo_out: u32 = 0;
    var hi_out: u32 = 0;
    inline for (0..4) |i| {
        const byte: u32 = (pw >> @intCast(8 * i)) & 0xFF;
        const s: u32 = (sw >> @intCast(8 * i)) & 0xFF;
        const base = s * 16;
        const llo = base + (byte & 0xF);
        const lhi = base + (byte >> 4);
        const wlo: u32 = @bitCast(d.data[llo >> 2]);
        const whi: u32 = @bitCast(d.data[lhi >> 2]);
        lo_out |= ((wlo >> @intCast(8 * (llo & 3))) & 0xFF) << @intCast(8 * i);
        hi_out |= ((whi >> @intCast(8 * (lhi & 3))) & 0xFF) << @intCast(8 * i);
    }
    b.data[2 * j * rq + r] = @bitCast(lo_out);
    b.data[(2 * j + 1) * rq + r] = @bitCast(hi_out);
}

// i4_decode_t: unpack a nibble-packed int4-convrot weight into the k-major int8 layout the
// coop GEMM reads. `w4a8_decode_t` above with the scale plane and the level table removed,
// a `.i4` nibble sign-extends straight to int8, and its per-OUTPUT-ROW scale is applied
// afterwards by `opI8GemmBuf`'s rescale, exactly as it is for a plain int8 weight. So this
// kernel does no arithmetic beyond a 4-bit sign extension.
//
// This exists so int4 stops costing int8's VRAM on Vulkan. There is no `sint4`
// cooperative matrix on this device, so the nibbles must reach the GEMM as int8 either way,
// but widening every weight ONCE AT LOAD keeps a full int8 copy resident. Measured on a
// 2B trunk: int4 and W4A8 hold the same 1050.7 MB of packed weight bytes, both being
// 4-bit, yet widening at load costs 1778 MB against W4A8's 1118 purely because W4A8
// decodes per GEMM. Same policy, same footprint.
//
// The result is W4A8-shaped arithmetic, NOT the CUDA arms' W4A4: the weight is 4-bit and
// the activation stays int8 (`opI8Prep`). That is what the widening already did, so this
// changes residency only, renders are expected bit-identical, and the test asserts it.
//
// Input arrives ALREADY BYTE-TRANSPOSED through the same cached `weightBuffer` path,
// for the reason `w4a8_decode_t` documents at length: a transposing decode has one strided
// side and cost that kernel 1757 ms/step against a ~20 ms roof before its inputs were
// pre-transposed. Do not "simplify" this by reading the row-major storage.
//
// Layout, `stride = align(rows, tile_n)` shared by input and output:
//   a = pT  packed, [cols/2][stride], byte (j, r) holds columns 2j, 2j+1 of row r
//   b = out int8,   [cols][stride]
// One thread owns 4 consecutive rows of one packed byte-column: one u32 in, one u32 into
// each of the two output planes 2j and 2j+1. Every access is coalesced.
//
// The row PADDING needs no special case: `weightBuffer` zeroes it, and nibble 0 sign-extends
// to int8 0, which is what the GEMM must see for a pad row.
//
// u0 = total threads = (cols/2)*(stride/4), u1 = stride/4.
export fn i4_decode_t() callconv(.spirv_kernel) void {
    decorate();
    const t = gpu.global_invocation_id[0];
    if (t >= pc.u0) return;
    const pw: u32 = @bitCast(a.data[t]);
    const rq = pc.u1; // row quads per byte-column plane
    const j = t / rq; // packed byte-column
    const r = t % rq; // row quad within it
    var lo_out: u32 = 0;
    var hi_out: u32 = 0;
    inline for (0..4) |i| {
        const byte: u32 = (pw >> @intCast(8 * i)) & 0xFF;
        // LOW nibble = EVEN element, sign-extended, the packing `ops.matmul`'s i4 path
        // and ComfyUI's W4A4 converter both use. Swapping the halves is not an error, it is
        // a different (wrong) weight matrix, and it is rms-preserving.
        const lo: u32 = @bitCast(@as(i32, @as(i4, @bitCast(@as(u4, @truncate(byte))))));
        const hi: u32 = @bitCast(@as(i32, @as(i4, @bitCast(@as(u4, @truncate(byte >> 4))))));
        lo_out |= (lo & 0xFF) << @intCast(8 * i);
        hi_out |= (hi & 0xFF) << @intCast(8 * i);
    }
    b.data[2 * j * rq + r] = @bitCast(lo_out);
    b.data[(2 * j + 1) * rq + r] = @bitCast(hi_out);
}

// nvfp4_decode_t: decode a packed ComfyUI NVFP4 weight into the f16 `[k_pad][n_pad]`
// k-major layout the coop f16 GEMM reads. Weight-only quantization, per GEMM, so the
// 4-bit form stays resident, which is what NVFP4 is on anything below Blackwell and what
// ComfyUI itself does there (see ops/nvfp4.zig).
//
// Element 2k is the HIGH nibble (`hi_first`), the opposite of `.i4`/`.w4a8` here.
// Both inputs arrive ALREADY BYTE-TRANSPOSED through `weightBuffer`, and the scales
// arrive already UNSWIZZLED from the loader, the same arrangement `w4a8_decode_t` needs
// and for the same measured reason: transposing per GEMM puts 32 sectors on every warp
// load, which cost that kernel 1757 ms/step before its inputs were pre-transposed.
//
// Layouts (`sin` = the inputs' shared row stride, `n_pad` = the output's):
//   a = pT      packed, [cols/2][sin], byte (j, r) holds columns 2j, 2j+1 of row r
//   c = sT      fp8 block scales, [cols/16][sin], unswizzled
//   d = levels  f16 [256][16], the host's table (E2M1 * per_tensor * block already folded)
//   b = out     f16 [k_pad][n_pad]
// One thread owns 4 consecutive rows of one packed byte-column: one u32 of `a`, one u32 of
// `c`, and two pairs of u32 written into output planes 2j and 2j+1, all coalesced.
//
// It writes BOTH paddings. `n_pad`/`k_pad` round rows/cols up to 128/64 and the GEMM
// reads all of it, while this scratch is shared between weights of different shapes, so
// an unwritten pad slot would feed the next GEMM the previous weight's values.
//
// The INPUT planes and the OUTPUT have DIFFERENT row strides, `weightBuffer` pads rows
// to `tile_n` (8) while the coop GEMM's layout pads to 128, so the dispatch must cover the
// OUTPUT's quads and guard the input read. Getting that backwards leaves every row-padding
// slot unwritten, which the first version did: a 40-row weight wrote 40 of 128 rows and the
// GEMM read the previous weight's values from the rest.
//
// u0 = total threads = (k_pad/2)*(n_pad/4), u1 = n_pad/4 (output row quads), u2 = n_pad,
// u3 = rows, u4 = cols, u5 = sin/4 (INPUT row quads).
export fn nvfp4_decode_t() callconv(.spirv_kernel) void {
    decorate();
    const t = gpu.global_invocation_id[0];
    if (t >= pc.u0) return;
    const rq = pc.u1;
    const j = t / rq; // packed byte-column -> output planes 2j, 2j+1
    const r = t % rq;
    const row0 = r * 4;
    // Only the real region has data; k- and row-padding decode to zero.
    const live_k = 2 * j < pc.u4 and r < pc.u5;
    const pw: u32 = if (live_k) @bitCast(a.data[j * pc.u5 + r]) else 0;
    const sw: u32 = if (live_k) @bitCast(c.data[(j / 8) * pc.u5 + r]) else 0;
    var hi0: u32 = 0;
    var hi1: u32 = 0;
    var lo0: u32 = 0;
    var lo1: u32 = 0;
    inline for (0..4) |i| {
        if (live_k and row0 + i < pc.u3) {
            const byte: u32 = (pw >> @intCast(8 * i)) & 0xFF;
            const s: u32 = (sw >> @intCast(8 * i)) & 0xFF;
            const base = s * 8; // 16 f16 entries = 8 u32 words per scale byte
            const chi = byte >> 4; // element 2j, HIGH nibble first
            const clo = byte & 0xF; // element 2j+1
            const whi: u32 = @bitCast(d.data[base + (chi >> 1)]);
            const wlo: u32 = @bitCast(d.data[base + (clo >> 1)]);
            const vhi: u32 = (whi >> @intCast(16 * (chi & 1))) & 0xFFFF;
            const vlo: u32 = (wlo >> @intCast(16 * (clo & 1))) & 0xFFFF;
            // Four f16 per plane pack into two u32: rows r0,r0+1 then r0+2,r0+3.
            if (i < 2) {
                hi0 |= vhi << @intCast(16 * i);
                lo0 |= vlo << @intCast(16 * i);
            } else {
                hi1 |= vhi << @intCast(16 * (i - 2));
                lo1 |= vlo << @intCast(16 * (i - 2));
            }
        }
    }
    const w_hi = (2 * j * pc.u2 + row0) >> 1; // f16 -> u32 word index
    const w_lo = ((2 * j + 1) * pc.u2 + row0) >> 1;
    b.data[w_hi] = @bitCast(hi0);
    b.data[w_hi + 1] = @bitCast(hi1);
    b.data[w_lo] = @bitCast(lo0);
    b.data[w_lo + 1] = @bitCast(lo1);
}

// --- VAE decoder kernels ---------------------------------------------------

// attn_cross: non-causal attention where the KEYS ARE A DIFFERENT LENGTH from
// the queries, the UNet's cross-attention onto the 77-row text conditioning.
// `attn_full` folds both into one `seq`, and padding K/V up to seq_q to reuse
// it would build an n×n scores plane where an n×77 one is wanted (16M against
// 315K entries at a 512² SD1.5 latent). One thread per (query, head); K/V are
// small enough to stay hot in L2 across the whole launch.
// a = q [seq_q][n_heads*hd], b = k, c = v (both [seq_kv][n_heads*hd]),
// d = out [seq_q][n_heads*hd].
// u0 = seq_q, u1 = n_heads, u2 = head_dim (<=256), u3 = seq_kv, f0 = scale.
export fn attn_cross() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const n_heads = pc.u1;
    if (idx >= pc.u0 * n_heads) return;
    const hd = pc.u2;
    const seq_kv = pc.u3;
    const scale = pc.f0;
    const head = idx % n_heads;
    const qb = idx * hd;
    const dim = n_heads * hd;
    var acc: [256]f32 = undefined;
    var t: u32 = 0;
    while (t < hd) : (t += 1) acc[t] = 0;
    var mx: f32 = -3.4e38;
    var denom: f32 = 0;
    var j: u32 = 0;
    while (j < seq_kv) : (j += 1) {
        const kb = j * dim + head * hd;
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

const attn_chunk = 32;
