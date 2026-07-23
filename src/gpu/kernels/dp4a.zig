//! int8 dot-product (dp4a) decode GEMV for GGUF block-quant weights, over a
//! REPACKED int8-interleaved weight layout (see transpose.zig repack_* kernels).
//!
//! Decode is compute-bound (~12% of memory bandwidth at the scalar-f32 GEMV's
//! throughput), so the lever is faster arithmetic. The weight is pre-repacked so
//! four consecutive quant values of a row sit contiguously in one u32 and a
//! 32-lane warp reads coalesced — so a thread issues one `OpSDot` (hardware
//! dp4a: 4 int8 MACs) per quad with NO gather/pack. The iq4_nl codebook is
//! pre-applied during repack, so q8_0 and iq4_nl share this single kernel
//! (uniform int8 + f32-scale blocks).
//!
//! Repacked layout, per 32-row group (block-index b, row r in 0..31, nblk
//! blocks/row, block region = 288 u32 = 256 quad-u32 + 32 f32 scales):
//!   quad qg (0..7): u32 at gi*(nblk*288) + b*288 + qg*32 + r  (4 packed int8)
//!   row scale:      f32 at gi*(nblk*288) + b*288 + 256 + r
//!
//! Separate SPIR-V module so the injected DotProduct capability never touches
//! the shared eltwise module. Bindings mirror eltwise (a,b,c,d + push).

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

// A u16-typed ALIAS of binding 0 (the raw weight `a`): lets the aligned-u16
// dp4a GEMV kernels read quant blocks as native 16-bit words (2 loads/quad, 4
// bytes) instead of assembling from u32 loads (2 loads, 8 bytes) — matching
// llama.cpp's block_q*_packed16 views. Needs StorageBuffer16BitAccess (injected
// host-side when the device supports it).
const WU16 = extern struct { data: [1 << 27]u16 };
extern var aw: WU16 addrspace(.storage_buffer);

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

// Decoration for the u16-alias kernels: everything decorate() does, plus the
// aw view aliased onto binding 0 (u16 element type, Block, offset 0).
inline fn decorate16() void {
    decorate();
    asm volatile (
        \\OpDecorate %wt Block
        \\OpMemberDecorate %wt 0 Offset 0
        \\OpDecorate %baw DescriptorSet 0
        \\OpDecorate %baw Binding 0
        :
        : [wt] "t" (WU16),
          [baw] "" (&aw),
    );
}

// Aligned 4-int8 quad from an EVEN byte offset: 2 native u16 loads → u32 for
// OpSDot. Reads exactly 4 bytes (vs the u32-view's 8) and is 16-bit-aligned.
inline fn wquad16(bo: u32) u32 {
    const i = bo / 2;
    return @as(u32, aw.data[i]) | (@as(u32, aw.data[i + 1]) << 16);
}
// The block's f16 scale as f32, read via the u16 view (bo even).
inline fn wf16a(bo: u32) f32 {
    return @floatCast(@as(f16, @bitCast(aw.data[bo / 2])));
}

// 4×int8 dot product (packed u32 lanes) -> i32 (SPIR-V DotProduct capability,
// PackedVectorFormat4x8Bit = literal 0). The driver lowers this to dp4a.
inline fn dot4(w4: u32, x4: u32) i32 {
    return asm (
        \\%r = OpSDot %i32t %w %x $fmt
        : [r] "" (-> i32),
        : [i32t] "t" (i32),
          [w] "" (w4),
          [x] "" (x4),
          [fmt] "c" (0),
    );
}

// Repacked block region size in u32: 8 quad-u32 × 32 rows + 32 f32 scales.
const BLOCK_U32 = 256 + 32;

// --- raw block readers + subgroup reduce (cooperative dp4a GEMV) -----------
// Byte-addressed reads into the RAW row-major GGUF layout (the cooperative
// kernels read raw, not repacked — no VRAM doubling).
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
// Assemble a u32 of 4 consecutive raw bytes at byte offset `bo` (q8_0 quads sit
// at +2 mod 4, so a plain u32 load would be misaligned — this is the alignment
// tax the repack avoids). Reads the (at most two) overlapping aligned u32s and
// shift-combines — 1–2 loads, not 4.
inline fn wquad(bo: u32) u32 {
    const i = bo / 4;
    const sh: u5 = @intCast((bo % 4) * 8);
    const lo: u32 = @bitCast(a.data[i]);
    if (sh == 0) return lo;
    const hi: u32 = @bitCast(a.data[i + 1]);
    return (lo >> sh) | (hi << @intCast(32 - @as(u32, sh)));
}
inline fn subgroupReduceAdd(v: f32) f32 {
    return asm (
        \\%r = OpGroupNonUniformFAdd %f32t %scope Reduce %val
        : [r] "" (-> f32),
        : [f32t] "t" (f32),
          [scope] "" (@as(u32, 3)),
          [val] "" (v),
    );
}

// iq4_nl non-linear 4-bit codebook (ggml kvalues_iq4nl).
const kvalues_iq4nl = [16]i32{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

// --- 32-row-group transposed readers (the `_t` / weightBufferRawT layout) ---
// Logical byte `j` of row `row` lives at gb + j*32 (gb = (row/32)*row_bytes*32
// + row%32). A 32-lane warp of consecutive rows reads 32 contiguous bytes per
// `j` (coalesced). This is the SAME layout the scalar `gemv_*_t` kernels and the
// GEMM prefill (weightBufferRawT) read — so a dp4a-over-_t kernel adds NO extra
// VRAM and shares the resident buffer.
const GROUP = 32;
inline fn tbyte(gb: u32, j: u32) u32 {
    const p = gb + j * GROUP;
    const word: u32 = @bitCast(a.data[p / 4]);
    const sh: u5 = @intCast(8 * (p % 4));
    return (word >> sh) & 0xFF;
}
inline fn tf16(gb: u32, j: u32) f32 { // the f16's two bytes are 32 B apart here
    const bits: u16 = @intCast(tbyte(gb, j) | (tbyte(gb, j + 1) << 8));
    return @floatCast(@as(f16, @bitCast(bits)));
}
// Assemble the 4 int8 at logical bytes j..j+3 into a u32 for OpSDot.
inline fn tquad(gb: u32, j: u32) u32 {
    return tbyte(gb, j) | (tbyte(gb, j + 1) << 8) | (tbyte(gb, j + 2) << 16) | (tbyte(gb, j + 3) << 24);
}

// gemv_q8_0_t_dp4a: dp4a decode GEMV over the 32-row-group TRANSPOSED q8_0
// weight (weightBufferRawT) + k-split — the fast repack-dp4a shape but reading
// the _t layout, so NO int8 repack (no extra VRAM, shares the prefill buffer).
// Thread (row, ch) dp4a's the contiguous block range [ch*chunk,(ch+1)*chunk) of
// its row into partials[ch*rows+row]; gemv_combine reduces over ch + applies the
// weight scale. a = W (_t), b = xi8 (packed 4/u32, 8/block), c = xscale, d =
// partials. u0 = rows*nchunk, u1 = cols, u2 = nchunk, u3 = rows.
export fn gemv_q8_0_t_dp4a() callconv(.spirv_kernel) void {
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
        const dsc = tf16(gb, bb) * c.data[blk];
        var isum: i32 = 0;
        var q: u32 = 0;
        while (q < 8) : (q += 1) {
            const wq = tquad(gb, bb + 2 + q * 4);
            const xq: u32 = @bitCast(b.data[blk * 8 + q]);
            isum += dot4(wq, xq);
        }
        acc += dsc * @as(f32, @floatFromInt(isum));
    }
    d.data[ch * rows + row] = acc;
}

// gemv_iq4_nl_t_dp4a: dp4a decode GEMV over the transposed iq4_nl weight
// (weightBufferRawT, 18 B block) + k-split. Nibbles are LUT'd to int8 in-
// register (no int8 repack → stays 4-bit in VRAM, HALF the repack footprint).
// quad q<4 = low nibbles of logical bytes 4q.., q>=4 = high nibbles of 4(q-4)..;
// matches quant_act_i8's element order. a = W (_t), b = xi8, c = xscale,
// d = partials. u0 = rows*nchunk, u1 = cols, u2 = nchunk, u3 = rows.
export fn gemv_iq4_nl_t_dp4a() callconv(.spirv_kernel) void {
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
        const dsc = tf16(gb, bb) * c.data[blk];
        var isum: i32 = 0;
        var q: u32 = 0;
        while (q < 8) : (q += 1) {
            const byte0 = (q % 4) * 4;
            const hi = q >= 4;
            var wq: u32 = 0;
            var t: u32 = 0;
            while (t < 4) : (t += 1) {
                const nyb = tbyte(gb, bb + 2 + byte0 + t);
                const nib: u32 = if (hi) nyb >> 4 else nyb & 0xF;
                const v: i32 = kvalues_iq4nl[@intCast(nib)];
                wq |= (@as(u32, @bitCast(v)) & 0xFF) << @intCast(8 * t);
            }
            const xq: u32 = @bitCast(b.data[blk * 8 + q]);
            isum += dot4(wq, xq);
        }
        acc += dsc * @as(f32, @floatFromInt(isum));
    }
    d.data[ch * rows + row] = acc;
}

// gemv_q8_0_sg_dp4a: dp4a decode GEMV over the RAW q8_0 weight (block 34 B =
// f16 d + 32 i8), ONE subgroup (32 lanes) per output row, K split across lanes
// in CONTIGUOUS 8-column chunks (the llama.cpp mul_mat_vecq mapping): each
// iteration the 32 lanes cover 8 consecutive blocks (256 cols) so the warp
// streams contiguous bytes = coalesced. Lane tid owns block (base+tid/4), chunk
// (tid%4) = 8 int8 = 2 OpSDots. subgroupReduceAdd over the 32 partials. NO _t
// transpose, NO int8 repack. a = raw W, b = xi8 (packed 4/u32, 8/block),
// c = xscale (f32/block), d = y. u0 = rows, u1 = cols, u2 = y elem offset,
// f0 = weight scale. Dispatch LocalSize 32 (1 subgroup/wg), one row per wg.
// Output rows computed per subgroup (llama.cpp's NUM_ROWS): the activation quad
// + xscale loaded once are reused across NR weight rows, amortizing the load and
// loop overhead over NR dp4a groups.
const NR = 4;

export fn gemv_q8_0_sg_dp4a() callconv(.spirv_kernel) void {
    decorate16();
    const gid = gpu.global_invocation_id[0];
    const tid = gid % 32;
    const row0 = (gid / 32) * NR;
    if (row0 >= pc.u0) return;
    const rows = pc.u0;
    const nblk = pc.u1 / 32;
    const rstride = nblk * 34;
    const blk_in_grp = tid / 4; // which of the 8 blocks in a group (0..7)
    const chunk = tid % 4; // which 8-int8 chunk of the block (0..3)
    var p: [NR]f32 = @splat(0);
    var base: u32 = 0;
    while (base < nblk) : (base += 8) {
        const blk = base + blk_in_grp;
        if (blk < nblk) {
            const x0: u32 = @bitCast(b.data[blk * 8 + chunk * 2]);
            const x1: u32 = @bitCast(b.data[blk * 8 + chunk * 2 + 1]);
            const xs = c.data[blk];
            const off = blk * 34 + 2 + chunk * 8; // byte offset within a row
            inline for (0..NR) |r| {
                const rr = row0 + @as(u32, @intCast(r));
                if (rr < rows) {
                    const bb = rr * rstride + off;
                    const isum = dot4(wquad16(bb), x0) + dot4(wquad16(bb + 4), x1);
                    p[r] += wf16a(rr * rstride + blk * 34) * xs * @as(f32, @floatFromInt(isum));
                }
            }
        }
    }
    inline for (0..NR) |r| {
        const sum = subgroupReduceAdd(p[r]);
        if (tid == 0 and row0 + r < rows) d.data[pc.u2 + row0 + r] = sum * pc.f0;
    }
}

// gemv_iq4_nl_sg_dp4a: as above for RAW iq4_nl (block 18 B = f16 d + 16 nibble
// bytes; v = kvalues[nibble]). Lane tid → block (base+tid/4), chunk (tid%4) =
// elements [chunk*8, chunk*8+8): chunk<2 = low nibbles of bytes chunk*8.., chunk
// >=2 = high nibbles of bytes (chunk-2)*8.. (matches quant_act_i8 element order).
// LUT'd to int8 in-register (no repack → stays 4-bit in VRAM). a = raw W, b =
// xi8, c = xscale, d = y. Dispatch LocalSize 32, one row per wg.
export fn gemv_iq4_nl_sg_dp4a() callconv(.spirv_kernel) void {
    decorate16();
    const gid = gpu.global_invocation_id[0];
    const tid = gid % 32;
    const row0 = (gid / 32) * NR;
    if (row0 >= pc.u0) return;
    const rows = pc.u0;
    const nblk = pc.u1 / 32;
    const rstride = nblk * 18;
    const chunk = tid % 4;
    const blk_in_grp = tid / 4;
    const byte0 = (chunk % 2) * 8; // base nibble byte for this 8-element chunk
    const hi = chunk >= 2; // high nibbles cover elements 16..31
    var p: [NR]f32 = @splat(0);
    var base: u32 = 0;
    while (base < nblk) : (base += 8) {
        const blk = base + blk_in_grp;
        if (blk < nblk) {
            const x0: u32 = @bitCast(b.data[blk * 8 + chunk * 2]);
            const x1: u32 = @bitCast(b.data[blk * 8 + chunk * 2 + 1]);
            const xs = c.data[blk];
            const noff = (blk * 18 + 2 + byte0) / 2; // u16 index within a row
            inline for (0..NR) |r| {
                const rr = row0 + @as(u32, @intCast(r));
                if (rr < rows) {
                    const nb = rr * (rstride / 2) + noff;
                    var wq: u32 = 0;
                    var wq1: u32 = 0;
                    var m: u32 = 0;
                    while (m < 2) : (m += 1) {
                        const wl: u32 = aw.data[nb + m];
                        const wh: u32 = aw.data[nb + 2 + m];
                        inline for (0..2) |h| {
                            const bl = (wl >> @intCast(8 * h)) & 0xFF;
                            const bh = (wh >> @intCast(8 * h)) & 0xFF;
                            const v0: i32 = kvalues_iq4nl[@intCast(if (hi) bl >> 4 else bl & 0xF)];
                            const v1: i32 = kvalues_iq4nl[@intCast(if (hi) bh >> 4 else bh & 0xF)];
                            const sh: u5 = @intCast(8 * (2 * m + h));
                            wq |= (@as(u32, @bitCast(v0)) & 0xFF) << sh;
                            wq1 |= (@as(u32, @bitCast(v1)) & 0xFF) << sh;
                        }
                    }
                    const isum = dot4(wq, x0) + dot4(wq1, x1);
                    p[r] += wf16a(rr * rstride + blk * 18) * xs * @as(f32, @floatFromInt(isum));
                }
            }
        }
    }
    inline for (0..NR) |r| {
        const sum = subgroupReduceAdd(p[r]);
        if (tid == 0 and row0 + r < rows) d.data[pc.u2 + row0 + r] = sum * pc.f0;
    }
}

// quant_act_i8: quantize the activation x[cols] to int8 per 32-block. One thread
// per block: scale = max|x|/127, xi8[k] = round(x[k]/scale). a = x (f32),
// c = xi8 (packed 4/u32), d = xscale (f32, one per block). u0 = nblocks.
export fn quant_act_i8() callconv(.spirv_kernel) void {
    decorate();
    const blk = gpu.global_invocation_id[0];
    if (blk >= pc.u0) return;
    const base = blk * 32;
    var amax: f32 = 0;
    var i: u32 = 0;
    while (i < 32) : (i += 1) amax = @max(amax, @abs(a.data[base + i]));
    const inv = if (amax > 0) 127.0 / amax else 0;
    d.data[blk] = amax / 127.0;
    var g: u32 = 0;
    while (g < 8) : (g += 1) {
        var word: u32 = 0;
        var t: u32 = 0;
        while (t < 4) : (t += 1) {
            const v = a.data[base + g * 4 + t] * inv;
            const q: i32 = @intFromFloat(@round(@max(-127.0, @min(127.0, v))));
            word |= (@as(u32, @bitCast(q)) & 0xFF) << @intCast(8 * t);
        }
        c.data[blk * 8 + g] = @bitCast(word);
    }
}

// gemv_repack_dp4a: y[row] partials over the repacked int8 weight. Thread
// (row, ch) dots block range [ch*chunk, (ch+1)*chunk) of its row via dp4a.
// a = repacked W, b = xi8 (packed), c = xscale, d = partials. u0=rows*nchunk
// u1=cols u2=nchunk u3=rows.
export fn gemv_repack_dp4a() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u3;
    const nchunk = pc.u2;
    const ch = idx / rows;
    const row = idx % rows;
    const nblk = pc.u1 / 32;
    const gi = row / 32;
    const r = row % 32;
    const gstride = nblk * BLOCK_U32;
    const chunk = (nblk + nchunk - 1) / nchunk;
    const start = ch * chunk;
    const stop = @min(start + chunk, nblk);
    var acc: f32 = 0;
    var blk: u32 = start;
    while (blk < stop) : (blk += 1) {
        const base = gi * gstride + blk * BLOCK_U32;
        var isum: i32 = 0;
        var qg: u32 = 0;
        while (qg < 8) : (qg += 1) {
            const w4: u32 = @bitCast(a.data[base + qg * 32 + r]);
            const x4: u32 = @bitCast(b.data[blk * 8 + qg]);
            isum += dot4(w4, x4);
        }
        const wscale = a.data[base + 256 + r];
        acc += wscale * c.data[blk] * @as(f32, @floatFromInt(isum));
    }
    d.data[ch * rows + row] = acc;
}

// dequant_repack_f32: repacked int8 weight -> f32 row-major [rows][cols]
// (element (row, col) at row*cols + col) for the prefill tensor-core GEMM path.
// a = repacked W, d = f32 out. u0 = rows*nblk, u1 = cols, u2 = rows.
export fn dequant_repack_f32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.u0) return;
    const rows = pc.u2;
    const cols = pc.u1;
    const nblk = cols / 32;
    const blk = idx / rows;
    const row = idx % rows;
    const gi = row / 32;
    const r = row % 32;
    const base = gi * (nblk * BLOCK_U32) + blk * BLOCK_U32;
    const wscale = a.data[base + 256 + r];
    const obase = row * cols + blk * 32;
    var qg: u32 = 0;
    while (qg < 8) : (qg += 1) {
        const w4: u32 = @bitCast(a.data[base + qg * 32 + r]);
        var t: u32 = 0;
        while (t < 4) : (t += 1) {
            const lane: i32 = @as(i32, @bitCast((w4 << @intCast(24 - 8 * t)))) >> 24; // sign-extend byte t
            d.data[obase + qg * 4 + t] = wscale * @as(f32, @floatFromInt(lane));
        }
    }
}
