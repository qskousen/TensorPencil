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

/// One f32 element per invocation.
export fn transpose_f32() callconv(.spirv_kernel) void {
    decorate();
    const col = gpu.global_invocation_id[0];
    const k = gpu.global_invocation_id[1];
    if (col >= pc.stride or k >= pc.cols) return;
    const v: u32 = if (col < pc.rows) src.w[col * pc.cols + k] else 0;
    dst.w[k * pc.stride + col] = v;
}
