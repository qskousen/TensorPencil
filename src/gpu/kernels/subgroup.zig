//! Subgroup-cooperative kernels: reductions done WITHIN a subgroup via
//! OpGroupNonUniform* ops, which need no workgroup storage class. This is the
//! verified escape hatch from the NVIDIA hang on Zig-emitted workgroup memory
//! (see ZIG.md / VULKAN_MEMORY.md): the Zig SPIR-V backend can emit these
//! subgroup ops through inline asm (same trick as OpSDot), and they RUN on
//! 580.173.02 where Zig-emitted `var addrspace(.shared)` reductions DEVICE_LOST.
//!
//! One subgroup (32 lanes on NVIDIA) cooperates on one row: lanes read the row
//! coalesced+strided, each accumulates a partial, then a single subgroup reduce
//! yields the full result in every lane — replacing the *_partial -> *_combine
//! kernel pairs that round-tripped partials through global memory.
//!
//! Bindings mirror the eltwise module (a,b,c,d + EltPush) so these pipelines
//! reuse pipeline_layout_e. Dispatch these with LocalSize x = 32 (one subgroup
//! per workgroup on wave32 devices) and one workgroup per row.
//!
//! GOTCHA: the Zig SPIR-V backend SEGFAULTS compiling a single-kernel module
//! that uses a subgroup inline-asm op; it compiles fine in a multi-kernel
//! module. Keep >= 2 kernels here.

const gpu = @import("std").gpu;

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
        \\OpMemberDecorate %pt 8 Offset 32
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

// Sum `v` across the subgroup (Subgroup scope = 3) via OpGroupNonUniformFAdd
// Reduce; every lane receives the total. Needs GroupNonUniform (61) +
// GroupNonUniformArithmetic (63) capabilities (injected host-side). The reduce
// is over ACTIVE invocations, so any early-return guard must be uniform across
// the subgroup (all lanes of a subgroup share the same row here).
inline fn subgroupReduceAdd(v: f32) f32 {
    return asm (
        \\%r = OpGroupNonUniformFAdd %f32t %scope Reduce %val
        : [r] "" (-> f32),
        : [f32t] "t" (f32),
          [scope] "" (@as(u32, 3)),
          [val] "" (v),
    );
}

// Max of `v` across the subgroup (OpGroupNonUniformFMax Reduce); every lane
// receives the max. Same GroupNonUniformArithmetic capability as the add.
inline fn subgroupReduceMax(v: f32) f32 {
    return asm (
        \\%r = OpGroupNonUniformFMax %f32t %scope Reduce %val
        : [r] "" (-> f32),
        : [f32t] "t" (f32),
          [scope] "" (@as(u32, 3)),
          [val] "" (v),
    );
}

// subgroup_sum: capability probe (verification). Each lane reduces its input
// across the subgroup and writes the total. Dispatched as one 32-lane group
// over 32 ones -> every lane reads back 32.0. Confirms a Zig-emitted
// subgroup-scope kernel EXECUTES on the device (no workgroup storage class),
// the escape hatch from the NVIDIA hang on Zig-emitted workgroup memory.
// a = in, d = out.
export fn subgroup_sum() callconv(.spirv_kernel) void {
    decorate();
    const i = gpu.global_invocation_id[0];
    d.data[i] = subgroupReduceAdd(a.data[i]);
}

// --- block-quant weight readers (weight buffer `a`, u32 view) -------------
// Byte-addressed reads into the RAW row-major GGUF block layout (identical to
// eltwise.zig's wbyte/wf16/wi8). The cooperative GEMV coalesces by spreading a
// block's 32 elements across the 32 subgroup lanes, so it reads this raw layout
// directly — no 32-row-group transpose (`weightBufferRawT`) and no dp4a repack.
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
const ScaleMin = struct { sc: u32, m: u32 };
inline fn scaleMinK4(sbase: u32, j: u32) ScaleMin { // ggml get_scale_min_k4
    if (j < 4) {
        return .{ .sc = wbyte(sbase + j) & 63, .m = wbyte(sbase + j + 4) & 63 };
    }
    return .{
        .sc = (wbyte(sbase + j + 4) & 0x0F) | ((wbyte(sbase + j - 4) >> 6) << 4),
        .m = (wbyte(sbase + j + 4) >> 4) | ((wbyte(sbase + j) >> 6) << 4),
    };
}
const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

// Cooperative block-quant decode GEMV: one subgroup (32 lanes) per output row.
// Lane `l` owns element `l` within every block, accumulates its per-lane
// partial dot (block scale folded in), then a single subgroup reduce yields the
// full row dot — the exact math of the serial gemv_q*_* kernels with the inner
// `while (l < 32)` loop parallelized across lanes. Reads are coalesced (a
// block's 32 elements are contiguous) with no transpose/repack. Bindings /
// push mirror opGemvQuant: a = W (raw), b = x, d = y; u0 = rows, u1 = cols,
// u2 = y elem offset, f0 = scale. Dispatch LocalSize x = 32, one workgroup/row.

// q8_0: block 34 B = [f16 d][32 i8].
export fn gemv_q8_0_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return;
    const nblk = pc.u1 / 32;
    const row_base = row * nblk * 34;
    var p: f32 = 0;
    var blk: u32 = 0;
    while (blk < nblk) : (blk += 1) {
        const bb = row_base + blk * 34;
        const sc = wf16(bb);
        const q = wi8(bb + 2 + lane);
        p += sc * @as(f32, @floatFromInt(q)) * b.data[blk * 32 + lane];
    }
    const sum = subgroupReduceAdd(p);
    if (lane == 0) d.data[pc.u2 + row] = sum * pc.f0;
}

// q4_k: super-block 144 B = f16 d, f16 dmin, 12 B scales, 128 B low nibbles.
export fn gemv_q4_k_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 144;
    var p: f32 = 0;
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
            const q = wbyte(qbase + g * 32 + lane);
            const xg = xb + g * 64;
            const wlo = d1 * @as(f32, @floatFromInt(q & 0xF)) - m1;
            const whi = d2 * @as(f32, @floatFromInt(q >> 4)) - m2;
            p += wlo * b.data[xg + lane];
            p += whi * b.data[xg + 32 + lane];
        }
    }
    const sum = subgroupReduceAdd(p);
    if (lane == 0) d.data[pc.u2 + row] = sum * pc.f0;
}

// q5_k: q4_k layout + 32 B of per-element 5th bits (qh). super-block 176 B.
export fn gemv_q5_k_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 176;
    var p: f32 = 0;
    var sb: u32 = 0;
    while (sb < nsb) : (sb += 1) {
        const bb = row_base + sb * 176;
        const sd = wf16(bb);
        const sdmin = wf16(bb + 2);
        const sbase = bb + 4;
        const qhbase = bb + 16;
        const qbase = bb + 48;
        const xb = sb * 256;
        const qh = wbyte(qhbase + lane);
        var g: u32 = 0;
        while (g < 4) : (g += 1) {
            const s1 = scaleMinK4(sbase, 2 * g);
            const s2 = scaleMinK4(sbase, 2 * g + 1);
            const d1 = sd * @as(f32, @floatFromInt(s1.sc));
            const m1 = sdmin * @as(f32, @floatFromInt(s1.m));
            const d2 = sd * @as(f32, @floatFromInt(s2.sc));
            const m2 = sdmin * @as(f32, @floatFromInt(s2.m));
            const q = wbyte(qbase + g * 32 + lane);
            const xg = xb + g * 64;
            const mlo: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g));
            const mhi: u32 = @as(u32, 1) << @as(u5, @intCast(2 * g + 1));
            const lo: u32 = (q & 0xF) + (if (qh & mlo != 0) @as(u32, 16) else 0);
            const hi: u32 = (q >> 4) + (if (qh & mhi != 0) @as(u32, 16) else 0);
            p += (d1 * @as(f32, @floatFromInt(lo)) - m1) * b.data[xg + lane];
            p += (d2 * @as(f32, @floatFromInt(hi)) - m2) * b.data[xg + 32 + lane];
        }
    }
    const sum = subgroupReduceAdd(p);
    if (lane == 0) d.data[pc.u2 + row] = sum * pc.f0;
}

// q6_k: super-block 210 B = 128 B ql, 64 B qh (2-bit), 16 i8 scales, f16 d.
export fn gemv_q6_k_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return;
    const nsb = pc.u1 / 256;
    const row_base = row * nsb * 210;
    var p: f32 = 0;
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
            const is = lane / 16;
            const ql_l = wbyte(qlh + lane);
            const ql_h = wbyte(qlh + lane + 32);
            const qh = wbyte(qhh + lane);
            const q1 = @as(i32, @intCast((ql_l & 0xF) | (((qh >> 0) & 3) << 4))) - 32;
            const q2 = @as(i32, @intCast((ql_h & 0xF) | (((qh >> 2) & 3) << 4))) - 32;
            const q3 = @as(i32, @intCast((ql_l >> 4) | (((qh >> 4) & 3) << 4))) - 32;
            const q4 = @as(i32, @intCast((ql_h >> 4) | (((qh >> 6) & 3) << 4))) - 32;
            const sc1 = wi8(sch + is + 0);
            const sc2 = wi8(sch + is + 2);
            const sc3 = wi8(sch + is + 4);
            const sc4 = wi8(sch + is + 6);
            p += sd * @as(f32, @floatFromInt(sc1 * q1)) * b.data[xh + lane];
            p += sd * @as(f32, @floatFromInt(sc2 * q2)) * b.data[xh + lane + 32];
            p += sd * @as(f32, @floatFromInt(sc3 * q3)) * b.data[xh + lane + 64];
            p += sd * @as(f32, @floatFromInt(sc4 * q4)) * b.data[xh + lane + 96];
        }
    }
    const sum = subgroupReduceAdd(p);
    if (lane == 0) d.data[pc.u2 + row] = sum * pc.f0;
}

// iq4_nl: block 18 B = f16 d + 16 nibble bytes; v = d * kvalues[nibble].
// Lane l reads byte l%16; lanes 0..15 take low nibbles, 16..31 high (both
// groups touch the same coalesced 16 bytes); x index = blk*32 + lane for all.
export fn gemv_iq4_nl_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return;
    const nblk = pc.u1 / 32;
    const row_base = row * nblk * 18;
    const i = lane % 16;
    var p: f32 = 0;
    var blk: u32 = 0;
    while (blk < nblk) : (blk += 1) {
        const bb = row_base + blk * 18;
        const sc = wf16(bb);
        const q = wbyte(bb + 2 + i);
        const nib: u32 = if (lane < 16) q & 0xF else q >> 4;
        const v: f32 = @floatFromInt(kvalues_iq4nl[@intCast(nib)]);
        p += sc * v * b.data[blk * 32 + lane];
    }
    const sum = subgroupReduceAdd(p);
    if (lane == 0) d.data[pc.u2 + row] = sum * pc.f0;
}

// attn_decode_sg: flash-decoding attention for ONE decode query, folded — one
// subgroup (32 lanes) per head, lane = one of 32 KV chunks. Each lane runs the
// online softmax over its chunk (m, dsum, acc[hd]) exactly like
// attn_dsplit_gemma, then the cross-chunk merge is done IN-SUBGROUP (max-reduce
// the running maxes, then add-reduce the reweighted dsum and each acc element)
// and lane 0 writes out[h][hd] directly — no global scratch, no attn_dmerge
// dispatch. f32 KV. Supports GQA + sliding-window + ring + bidirectional block,
// same push as attn_dsplit_gemma (u4 nsplit is ignored; lanes ARE the 32 split).
// a = q [heads][hd], b = k_cache, c = v_cache, d = out [heads][hd].
// u0=kv_len, u1=heads, u2=kv_heads, u3=hd(<=256), u5=window, f0=scale,
// f1=ring (bitcast u32), u6=kv_end. Dispatch LocalSize 32, one wg per head.
export fn attn_decode_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32; // KV chunk index (the 32-way split)
    const h = gid / 32; // head
    const heads = pc.u1;
    if (h >= heads) return; // uniform across the subgroup
    const hd = pc.u3;
    const kv_len = pc.u0;
    const kv_heads = pc.u2;
    const window = pc.u5;
    const ring: u32 = @bitCast(pc.f1);
    const scale = pc.f0;
    const kv_end = if (pc.u6 != 0) pc.u6 else kv_len;
    const kvh = h / (heads / kv_heads);
    const kv_start: u32 = if (window != 0 and kv_len > window) kv_len - window else 0;
    const span = kv_end - kv_start;
    const chunk = (span + 32 - 1) / 32;
    const kv0 = kv_start + lane * chunk;
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
    // In-subgroup online-softmax merge across the 32 chunk-lanes.
    const gmax = subgroupReduceMax(m);
    const w = @exp(m - gmax); // this lane's reweight (0 for an empty chunk: m=-inf)
    const gden = subgroupReduceAdd(dsum * w);
    const inv = if (gden > 0) 1.0 / gden else 0;
    var t: u32 = 0;
    while (t < hd) : (t += 1) {
        const num = subgroupReduceAdd(acc[t] * w);
        if (lane == 0) d.data[qbase + t] = num * inv;
    }
}

// rmsnorm_sg: one subgroup per row, one-pass RMSNorm with a plain norm weight.
//   y[row][i] = x[row][i] * inv * w[i],  inv = 1/sqrt(mean(x^2) + eps).
// Replaces the 3-pass rms_partial -> rms_combine -> rms_apply_w that round-trips
// per-chunk partials through global memory. Each lane strides the row summing
// squares, a single subgroup reduce gives the full sum in every lane, then each
// lane writes its strided elements. Dispatch LocalSize x = 32, one workgroup
// per row. a = x, b = out, c = weight. u0 = rows, u1 = dim, f0 = eps.
export fn rmsnorm_sg() callconv(.spirv_kernel) void {
    decorate();
    const gid = gpu.global_invocation_id[0];
    const lane = gid % 32;
    const row = gid / 32;
    if (row >= pc.u0) return; // uniform across the subgroup (all 32 lanes share row)
    const dim = pc.u1;
    const base = row * dim;
    var partial: f32 = 0;
    var i: u32 = lane;
    while (i < dim) : (i += 32) {
        const v = a.data[base + i];
        partial += v * v;
    }
    const sum = subgroupReduceAdd(partial);
    const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(dim)) + pc.f0);
    i = lane;
    while (i < dim) : (i += 32) {
        b.data[base + i] = a.data[base + i] * inv * c.data[i];
    }
}
