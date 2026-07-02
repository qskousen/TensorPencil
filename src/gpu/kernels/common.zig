//! Shared definitions for the GPU matmul kernels, compiled to SPIR-V by the
//! Zig self-hosted backend (build.zig: -target spirv64-vulkan -fno-llvm,
//! SPIR-V 1.5).
//!
//! y[m, rows] = x[m, cols] @ W^T (+ bias). The weight buffer holds W
//! TRANSPOSED to k-major with the output dimension padded to a multiple of 4
//! (`w_stride`): element (k, col) sits at k * w_stride + col. That makes
//! reads coalesced without workgroup shared memory — deliberately, since the
//! NVIDIA 580 driver faults on Zig-emitted workgroup-storage kernels (see
//! ZIG.md). Each thread computes a tm x tn register tile instead.
//!
//! Bindings (set 0): 0 = W (u32 words: fp8 bytes or f32 bits), 1 = x,
//! 2 = y, 3 = bias. Workgroup size is patched in at load time (spv.zig);
//! dispatch is x = ceil(rows / (wg_x * tn)), y = ceil(m / (wg_y * tm)).
//!
//! Array bounds below are type-level maxima, not allocations.

pub const gpu = @import("std").gpu;

/// Outputs per thread: tm tokens x tn columns (tn a multiple of 4: each
/// aligned u32 of the fp8 layout covers 4 columns). Must match Context in
/// gpu/context.zig. Tuned on RTX 3090: 8x8 with ku=4 beats 8x4 and 4x8.
pub const tm = 8;
pub const tn = 8;

pub const WBuf = extern struct { words: [1 << 27]u32 };
pub const FBuf = extern struct { data: [1 << 27]f32 };
pub const BBuf = extern struct { data: [1 << 20]f32 };

pub const Push = extern struct {
    m: u32,
    rows: u32,
    cols: u32,
    w_stride: u32, // rows padded to a multiple of 4 (transposed layout)
    has_bias: u32,
    scale: f32,
};

pub extern var wbuf: WBuf addrspace(.storage_buffer);
pub extern var xbuf: FBuf addrspace(.storage_buffer);
pub extern var ybuf: FBuf addrspace(.storage_buffer);
pub extern var bbuf: BBuf addrspace(.storage_buffer);
pub extern var pc: Push addrspace(.push_constant);

pub inline fn decorate() void {
    asm volatile (
        \\OpDecorate %wt Block
        \\OpMemberDecorate %wt 0 Offset 0
        \\OpDecorate %ft Block
        \\OpMemberDecorate %ft 0 Offset 0
        \\OpDecorate %bt Block
        \\OpMemberDecorate %bt 0 Offset 0
        \\OpDecorate %pt Block
        \\OpMemberDecorate %pt 0 Offset 0
        \\OpMemberDecorate %pt 1 Offset 4
        \\OpMemberDecorate %pt 2 Offset 8
        \\OpMemberDecorate %pt 3 Offset 12
        \\OpMemberDecorate %pt 4 Offset 16
        \\OpMemberDecorate %pt 5 Offset 20
        \\OpDecorate %w DescriptorSet 0
        \\OpDecorate %w Binding 0
        \\OpDecorate %x DescriptorSet 0
        \\OpDecorate %x Binding 1
        \\OpDecorate %y DescriptorSet 0
        \\OpDecorate %y Binding 2
        \\OpDecorate %b DescriptorSet 0
        \\OpDecorate %b Binding 3
        :
        : [wt] "t" (WBuf),
          [ft] "t" (FBuf),
          [bt] "t" (BBuf),
          [pt] "t" (Push),
          [w] "" (&wbuf),
          [x] "" (&xbuf),
          [y] "" (&ybuf),
          [b] "" (&bbuf),
    );
}

pub inline fn e4m3ToF32(byte: u32) f32 {
    const man = byte & 0x7;
    const exp = (byte >> 3) & 0xF;
    _ = exp;
    // Branchless: the normal case assembles exponent/mantissa directly
    // ((byte & 0x7f) << 20 lands man at bit 20 and exp at bit 23, +120
    // rebias); subnormal (exp == 0, i.e. magnitude < 8) = ±man * 2^-9.
    const sign: u32 = (byte & 0x80) << 24;
    const magnitude = byte & 0x7F;
    const normal: f32 = @bitCast(sign | ((magnitude << 20) + (120 << 23)));
    const subnormal: f32 = @as(f32, @bitCast(sign | @as(u32, 0x3F800000))) *
        (@as(f32, @floatFromInt(man)) * 0x1p-9);
    return if (magnitude >= 8) normal else subnormal;
}

/// Register-tiled GEMM over the transposed weight layout. Threads whose
/// whole tile is out of range exit early; partial tiles read x with a
/// clamped token index (always in bounds) and mask at the store.
pub inline fn tiledMatmul(comptime f8: bool) void {
    decorate();
    const col_base = gpu.global_invocation_id[0] * tn;
    const t_base = gpu.global_invocation_id[1] * tm;
    if (col_base >= pc.rows or t_base >= pc.m) return;

    var acc: [tm][tn]f32 = @splat(@splat(0.0));
    const m_max = pc.m - 1;

    // Hoist per-row x offsets (clamped: padded rows recompute row m-1 and
    // are masked at the store) so the k loop is pure loads + FMAs.
    var xoff: [tm]u32 = undefined;
    inline for (0..tm) |i| {
        xoff[i] = @min(t_base + i, m_max) * pc.cols;
    }

    // Unroll k to keep several independent weight loads in flight.
    const ku = 4;
    const cols_main = pc.cols & ~@as(u32, ku - 1);
    var k: u32 = 0;
    while (k < cols_main) : (k += ku) {
        var w: [ku][tn]f32 = undefined;
        inline for (0..ku) |u| {
            const wbase = (k + u) * pc.w_stride + col_base;
            if (f8) {
                // Each aligned u32 covers 4 of the thread's fp8 columns.
                inline for (0..tn / 4) |wi| {
                    const word = wbuf.words[(wbase >> 2) + wi];
                    inline for (0..4) |j| {
                        w[u][wi * 4 + j] = e4m3ToF32((word >> (8 * j)) & 0xFF);
                    }
                }
            } else {
                inline for (0..tn) |j| {
                    w[u][j] = @bitCast(wbuf.words[wbase + j]);
                }
            }
        }
        inline for (0..tm) |i| {
            inline for (0..ku) |u| {
                const xv = xbuf.data[xoff[i] + k + u];
                inline for (0..tn) |j| {
                    acc[i][j] += xv * w[u][j];
                }
            }
        }
    }
    while (k < pc.cols) : (k += 1) {
        const wbase = k * pc.w_stride + col_base;
        var w: [tn]f32 = undefined;
        if (f8) {
            inline for (0..tn / 4) |wi| {
                const word = wbuf.words[(wbase >> 2) + wi];
                inline for (0..4) |j| {
                    w[wi * 4 + j] = e4m3ToF32((word >> (8 * j)) & 0xFF);
                }
            }
        } else {
            inline for (0..tn) |j| {
                w[j] = @bitCast(wbuf.words[wbase + j]);
            }
        }
        inline for (0..tm) |i| {
            const xv = xbuf.data[xoff[i] + k];
            inline for (0..tn) |j| {
                acc[i][j] += xv * w[j];
            }
        }
    }

    inline for (0..tm) |i| {
        const t = t_base + i;
        if (t < pc.m) {
            inline for (0..tn) |j| {
                const col = col_base + j;
                if (col < pc.rows) {
                    var v = acc[i][j] * pc.scale;
                    if (pc.has_bias != 0) v += bbuf.data[col];
                    ybuf.data[t * pc.rows + col] = v;
                }
            }
        }
    }
}
