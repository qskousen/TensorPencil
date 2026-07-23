//! Weight transpose kernels: row-major [rows, cols] -> k-major
//! (element (k, col) at k * stride + col, stride = align(rows, tn)).
//! Runs on upload so the 12+ GiB CPU-side transpose disappears; output reads
//! are coalesced, input reads strided (fine at device bandwidth).
//!
//! Bindings (set 0): 0 = src raw weights, 1 = dst transposed. Both u32
//! words: fp8 bytes for `transpose_f8`, f32 bits for `transpose_f32`.
//! Dispatch: x = ceil((stride/4) / wg), y = cols (one k per y).

const gpu = @import("std").gpu;

const Words = extern struct { w: [1 << 28]u32 };

pub const Push = extern struct {
    rows: u32,
    cols: u32,
    stride: u32,
    k_pad: u32, // bf16_to_f16_coopw only (the f8/f32 transposes ignore it)
};

extern var src: Words addrspace(.storage_buffer);
extern var dst: Words addrspace(.storage_buffer);
extern var pc: Push addrspace(.push_constant);

inline fn decorate() void {
    asm volatile (
        \\OpDecorate %wt Block
        \\OpMemberDecorate %wt 0 Offset 0
        \\OpDecorate %pt Block
        \\OpMemberDecorate %pt 0 Offset 0
        \\OpMemberDecorate %pt 1 Offset 4
        \\OpMemberDecorate %pt 2 Offset 8
        \\OpMemberDecorate %pt 3 Offset 12
        \\OpDecorate %s DescriptorSet 0
        \\OpDecorate %s Binding 0
        \\OpDecorate %d DescriptorSet 0
        \\OpDecorate %d Binding 1
        :
        : [wt] "t" (Words),
          [pt] "t" (Push),
          [s] "" (&src),
          [d] "" (&dst),
    );
}

/// One output u32 (4 fp8 columns at one k) per invocation.
export fn transpose_f8() callconv(.spirv_kernel) void {
    decorate();
    const col_base = gpu.global_invocation_id[0] * 4;
    const k = gpu.global_invocation_id[1];
    if (col_base >= pc.stride or k >= pc.cols) return;

    var word: u32 = 0;
    inline for (0..4) |j| {
        const col = col_base + j;
        if (col < pc.rows) {
            const idx = col * pc.cols + k; // byte index into src
            const byte = (src.w[idx >> 2] >> @intCast((idx & 3) * 8)) & 0xFF;
            word |= byte << (8 * j);
        }
    }
    dst.w[(k * pc.stride + col_base) >> 2] = word;
}

/// bf16 weight [rows][cols] -> bf16 k-major [cols][stride] for the dense GEMV
/// path: element (k, col) at k*stride + col (stride = align(rows, tn)), 2 bytes
/// each, packed two columns per u32. Twin of `transpose_f8` (which packs four
/// 1-byte columns). One output u32 (2 columns at one k) per invocation; x =
/// column-pair in [0, stride/2), y = k in [0, cols).
export fn transpose_bf16() callconv(.spirv_kernel) void {
    decorate();
    const col_base = gpu.global_invocation_id[0] * 2;
    const k = gpu.global_invocation_id[1];
    if (col_base >= pc.stride or k >= pc.cols) return;
    var word: u32 = 0;
    inline for (0..2) |j| {
        const col = col_base + j;
        if (col < pc.rows) {
            const idx = col * pc.cols + k; // bf16 element index into src
            const bits: u32 = (src.w[idx >> 1] >> @intCast((idx & 1) * 16)) & 0xFFFF;
            word |= bits << @intCast(16 * j);
        }
    }
    dst.w[(k * pc.stride + col_base) >> 1] = word;
}

/// Raw block-quant weight [rows][row_bytes] -> the 32-row-group byte-transposed
/// layout the *_t GEMV / dequant kernels read: logical byte j of row `row` lands
/// at (row/32)*row_bytes*32 + row%32 + j*32. A pure byte permutation (dequant-
/// neutral), replacing weightBufferRawT's single-thread ~8 GiB CPU scatter with
/// one device-bandwidth pass. One OUTPUT u32 per invocation, holding four
/// CONSECUTIVE rows' byte j: a word's first dst byte starts at row%32 that is a
/// multiple of 4 and j is constant across the 4 bytes, so they are exactly rows
/// {base..base+3} at column j. Pad rows (>= `rows`) and the 4-byte tail contribute
/// zero. rows = real rows, stride = row_bytes, cols = total output words (bound).
export fn transpose_grp32() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    if (idx >= pc.cols) return;
    const row_bytes = pc.stride;
    const group_bytes = row_bytes * 32;
    const p0 = idx * 4; // first dst byte of this word
    const group = p0 / group_bytes;
    const local = p0 % group_bytes;
    const j = local / 32; // logical byte within the row
    const r0 = local % 32; // first row-in-group (multiple of 4)
    const base_row = group * 32 + r0;
    var out: u32 = 0;
    inline for (0..4) |t| {
        const row = base_row + t;
        if (row < pc.rows) {
            const q = row * row_bytes + j; // src byte index (row-major)
            const byte = (src.w[q >> 2] >> @intCast((q & 3) * 8)) & 0xFF;
            out |= byte << @intCast(8 * t);
        }
    }
    dst.w[idx] = out;
}

/// One f32 element per invocation.
export fn transpose_f32() callconv(.spirv_kernel) void {
    decorate();
    const col = gpu.global_invocation_id[0];
    const k = gpu.global_invocation_id[1];
    if (col >= pc.stride or k >= pc.cols) return;
    const v: u32 = if (col < pc.rows) src.w[col * pc.cols + k] else 0;
    dst.w[k * pc.stride + col] = v;
}

/// bf16 weight [rows][cols] -> f16 k-major [k_pad][stride] (the f16-weight coop
/// GEMM layout: element (k, r) at k*stride + r, stride = n_pad = align(rows,128),
/// k_pad = align(cols,64), both pads zeroed). Converts bf16->f16 on the GPU so
/// the whole conversion + transpose runs at device bandwidth instead of a
/// single CPU thread. One output u32 (two adjacent r as packed f16) per
/// invocation. x = r-pair in [0, stride/2), y = k in [0, k_pad).
export fn bf16_to_f16_coopw() callconv(.spirv_kernel) void {
    decorate();
    const rp = gpu.global_invocation_id[0];
    const k = gpu.global_invocation_id[1];
    if (rp * 2 >= pc.stride or k >= pc.k_pad) return;
    var out: u32 = 0;
    if (k < pc.cols) {
        inline for (0..2) |j| {
            const r = rp * 2 + j;
            if (r < pc.rows) {
                const idx = r * pc.cols + k; // bf16 element index
                const bits: u32 = (src.w[idx >> 1] >> @intCast((idx & 1) * 16)) & 0xFFFF;
                const f: f32 = @bitCast(bits << 16); // bf16 -> f32
                const h: f16 = @floatCast(f);
                out |= @as(u32, @as(u16, @bitCast(h))) << @intCast(16 * j);
            }
        }
    }
    dst.w[(k * pc.stride + rp * 2) >> 1] = out;
}

/// bf16 weight [rows][cols] -> bf16 k-major [k_pad][stride] (native bf16 coop
/// GEMM). Same layout as bf16_to_f16_coopw but the 16-bit bits are copied
/// VERBATIM (bf16 in, bf16 out) — no conversion, so the tensor cores consume
/// the raw checkpoint values. Two adjacent r packed per output u32.
export fn bf16_coopw() callconv(.spirv_kernel) void {
    decorate();
    const rp = gpu.global_invocation_id[0];
    const k = gpu.global_invocation_id[1];
    if (rp * 2 >= pc.stride or k >= pc.k_pad) return;
    var out: u32 = 0;
    if (k < pc.cols) {
        inline for (0..2) |j| {
            const r = rp * 2 + j;
            if (r < pc.rows) {
                const idx = r * pc.cols + k; // bf16 element index
                const bits: u32 = (src.w[idx >> 1] >> @intCast((idx & 1) * 16)) & 0xFFFF;
                out |= bits << @intCast(16 * j);
            }
        }
    }
    dst.w[(k * pc.stride + rp * 2) >> 1] = out;
}
