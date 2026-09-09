//! Kernels with ONE source for both GPU arms: this file compiles to SPIR-V for the
//! Vulkan `Context` (self-hosted backend, like eltwise.zig) and to PTX for the CUDA
//! `Backend` (LLVM, then build.zig's ptx_unalias + zig cc lowering).
//!
//! Every entry shares the eltwise launch ABI: four f32 buffers a..d, seven u32
//! u0..u6 and two f32 f0,f1. On Vulkan those are the storage-buffer bindings and
//! the push-constant block; on CUDA they are the kernel parameters, in that order,
//! which is what `Backend.dualLaunch` passes. A body reads them only through `Env`,
//! so the two preludes below are the whole per-target difference. Workgroup sizes
//! live in dual_table.zig, which both hosts import.
//!
//! Rules the two compilers impose:
//! - A body must be `inline`. The SPIR-V backend crashes on a real call from a
//!   kernel into a function that touches a storage buffer.
//! - `@exp` has no nvptx libcall; use `exp` below, which is `ex2.approx` on PTX.
//! - Storage-buffer many-pointers cannot be indexed on SPIR-V (logical
//!   addressing), so `Env` exposes `ld`/`st`, never a pointer.
//! - No workgroup memory: Zig-emitted workgroup storage hangs NVIDIA under Vulkan.
//!   Cross-thread work is subgroup ops (`sgSum`, `sgMax`), one subgroup per row.
//!
//! Row kernels stride over rows by subgroup (`sgGlobal` / `sgTotal`), so the host
//! sizes their launch with `dual_table.rowGroups` without knowing the subgroup width.
//!
//! Bodies live in dual/elt.zig (per element or per f16 pair) and dual/rows.zig
//! (per subgroup-row); each documents its slots and scalars.

const std = @import("std");
const builtin = @import("builtin");
const gpu = std.gpu;
const table = @import("dual_table.zig");

pub const is_ptx = builtin.cpu.arch == .nvptx64;

pub const Slot = enum { a, b, c, d };

// ---- SPIR-V prelude ---------------------------------------------------------

const FBuf = extern struct { data: [1 << 28]f32 };
const Push = extern struct { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, f0: f32, f1: f32, u6: u32 };
extern var a: FBuf addrspace(.storage_buffer);
extern var b: FBuf addrspace(.storage_buffer);
extern var c: FBuf addrspace(.storage_buffer);
extern var d: FBuf addrspace(.storage_buffer);
extern var pc: Push addrspace(.push_constant);
// Subgroup builtins std.gpu does not declare; the BuiltIn decorations are in
// `decorate`.
extern const sg_size: u32 addrspace(.input);
extern const sg_lane: u32 addrspace(.input);
extern const sg_id: u32 addrspace(.input);

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
        \\OpDecorate %ss BuiltIn SubgroupSize
        \\OpDecorate %sl BuiltIn SubgroupLocalInvocationId
        \\OpDecorate %si BuiltIn SubgroupId
        :
        : [ft] "t" (FBuf),
          [pt] "t" (Push),
          [ba] "" (&a),
          [bb] "" (&b),
          [bc] "" (&c),
          [bd] "" (&d),
          [ss] "" (&sg_size),
          [sl] "" (&sg_lane),
          [si] "" (&sg_id),
    );
}

// ---- PTX prelude ------------------------------------------------------------

const G = [*]addrspace(.global) f32;

extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() u32;

/// What a body sees: buffer loads/stores by slot, the scalar arguments, and the
/// thread, workgroup and subgroup indices.
pub const Env = if (is_ptx) struct {
    a: G,
    b: G,
    c: G,
    d: G,
    us: [7]u32,
    fs: [2]f32,
    pub inline fn ld(e: Env, comptime s: Slot, i: u32) f32 {
        return switch (s) {
            .a => e.a[i],
            .b => e.b[i],
            .c => e.c[i],
            .d => e.d[i],
        };
    }
    pub inline fn st(e: Env, comptime s: Slot, i: u32, v: f32) void {
        switch (s) {
            .a => e.a[i] = v,
            .b => e.b[i] = v,
            .c => e.c[i] = v,
            .d => e.d[i] = v,
        }
    }
    pub inline fn ldW(e: Env, comptime s: Slot, i: u32) u32 {
        return @bitCast(e.ld(s, i));
    }
    pub inline fn stW(e: Env, comptime s: Slot, i: u32, w: u32) void {
        e.st(s, i, @bitCast(w));
    }
    /// f16 element `i` of a buffer of f16 pairs.
    pub inline fn h16(e: Env, comptime s: Slot, i: u32) f32 {
        return @floatCast(@as(f16, @bitCast(e.h16Bits(s, i))));
    }
    pub inline fn h16Bits(e: Env, comptime s: Slot, i: u32) u16 {
        const sh: u5 = @intCast((i & 1) * 16);
        return @truncate(e.ldW(s, i >> 1) >> sh);
    }
    /// bf16 element `i` of a buffer of bf16 pairs.
    pub inline fn bf16(e: Env, comptime s: Slot, i: u32) f32 {
        return @bitCast(@as(u32, e.h16Bits(s, i)) << 16);
    }
    /// Byte `off` of a buffer of raw bytes.
    pub inline fn byte(e: Env, comptime s: Slot, off: u32) u32 {
        return (e.ldW(s, off >> 2) >> @intCast((off & 3) * 8)) & 0xFF;
    }
    /// 16-bit word at byte offset `off` (2-aligned).
    pub inline fn u16At(e: Env, comptime s: Slot, off: u32) u32 {
        return (e.ldW(s, off >> 2) >> @intCast((off & 2) * 8)) & 0xFFFF;
    }
    /// f16 at byte offset `off` (2-aligned), as f32.
    pub inline fn f16At(e: Env, comptime s: Slot, off: u32) f32 {
        return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(e.u16At(s, off))))));
    }
    pub inline fn u(e: Env, comptime i: usize) u32 {
        return e.us[i];
    }
    pub inline fn f(e: Env, comptime i: usize) f32 {
        return e.fs[i];
    }
    pub inline fn gid(_: Env) u32 {
        return @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    }
    pub inline fn lid(_: Env) u32 {
        return @workItemId(0);
    }
    pub inline fn wgid(_: Env) u32 {
        return @workGroupId(0);
    }
    pub inline fn nwg(_: Env) u32 {
        return @"llvm.nvvm.read.ptx.sreg.nctaid.x"();
    }
    pub inline fn sgSize(_: Env) u32 {
        return 32;
    }
    pub inline fn sgLane(_: Env) u32 {
        return @workItemId(0) % 32;
    }
    pub inline fn sgId(_: Env) u32 {
        return @workItemId(0) / 32;
    }
} else struct {
    pub inline fn ld(_: Env, comptime s: Slot, i: u32) f32 {
        return switch (s) {
            .a => a.data[i],
            .b => b.data[i],
            .c => c.data[i],
            .d => d.data[i],
        };
    }
    pub inline fn st(_: Env, comptime s: Slot, i: u32, v: f32) void {
        switch (s) {
            .a => a.data[i] = v,
            .b => b.data[i] = v,
            .c => c.data[i] = v,
            .d => d.data[i] = v,
        }
    }
    pub inline fn ldW(e: Env, comptime s: Slot, i: u32) u32 {
        return @bitCast(e.ld(s, i));
    }
    pub inline fn stW(e: Env, comptime s: Slot, i: u32, w: u32) void {
        e.st(s, i, @bitCast(w));
    }
    /// f16 element `i` of a buffer of f16 pairs.
    pub inline fn h16(e: Env, comptime s: Slot, i: u32) f32 {
        return @floatCast(@as(f16, @bitCast(e.h16Bits(s, i))));
    }
    pub inline fn h16Bits(e: Env, comptime s: Slot, i: u32) u16 {
        const sh: u5 = @intCast((i & 1) * 16);
        return @truncate(e.ldW(s, i >> 1) >> sh);
    }
    /// bf16 element `i` of a buffer of bf16 pairs.
    pub inline fn bf16(e: Env, comptime s: Slot, i: u32) f32 {
        return @bitCast(@as(u32, e.h16Bits(s, i)) << 16);
    }
    /// Byte `off` of a buffer of raw bytes.
    pub inline fn byte(e: Env, comptime s: Slot, off: u32) u32 {
        return (e.ldW(s, off >> 2) >> @intCast((off & 3) * 8)) & 0xFF;
    }
    /// 16-bit word at byte offset `off` (2-aligned).
    pub inline fn u16At(e: Env, comptime s: Slot, off: u32) u32 {
        return (e.ldW(s, off >> 2) >> @intCast((off & 2) * 8)) & 0xFFFF;
    }
    /// f16 at byte offset `off` (2-aligned), as f32.
    pub inline fn f16At(e: Env, comptime s: Slot, off: u32) f32 {
        return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(e.u16At(s, off))))));
    }
    pub inline fn u(_: Env, comptime i: usize) u32 {
        return switch (i) {
            0 => pc.u0,
            1 => pc.u1,
            2 => pc.u2,
            3 => pc.u3,
            4 => pc.u4,
            5 => pc.u5,
            6 => pc.u6,
            else => unreachable,
        };
    }
    pub inline fn f(_: Env, comptime i: usize) f32 {
        return switch (i) {
            0 => pc.f0,
            1 => pc.f1,
            else => unreachable,
        };
    }
    pub inline fn gid(_: Env) u32 {
        return gpu.global_invocation_id[0];
    }
    pub inline fn lid(_: Env) u32 {
        return gpu.local_invocation_id[0];
    }
    pub inline fn wgid(_: Env) u32 {
        return gpu.workgroup_id[0];
    }
    pub inline fn nwg(_: Env) u32 {
        return gpu.num_workgroups[0];
    }
    pub inline fn sgSize(_: Env) u32 {
        return sg_size;
    }
    pub inline fn sgLane(_: Env) u32 {
        return sg_lane;
    }
    pub inline fn sgId(_: Env) u32 {
        return sg_id;
    }
};

/// Index of this subgroup among every subgroup of the launch, and their count.
/// Row kernels are `row_wg` wide, so both follow from the subgroup width.
pub inline fn sgGlobal(e: Env) u32 {
    return e.wgid() * (table.row_wg / e.sgSize()) + e.sgId();
}
pub inline fn sgTotal(e: Env) u32 {
    return e.nwg() * (table.row_wg / e.sgSize());
}

// ---- shared math ------------------------------------------------------------

/// Thread index of a per-element body, or null past u0.
pub inline fn elem(e: Env) ?u32 {
    const i = e.gid();
    return if (i >= e.u(0)) null else i;
}

pub inline fn exp(x: f32) f32 {
    return if (is_ptx) @exp2(x * 1.4426950408889634) else @exp(x);
}

pub inline fn log2(x: f32) f32 {
    if (is_ptx) {
        return asm ("lg2.approx.f32 %[r], %[x];"
            : [r] "=f" (-> f32),
            : [x] "f" (x),
        );
    } else {
        return @log2(x);
    }
}

pub inline fn sin(x: f32) f32 {
    if (is_ptx) {
        return asm ("sin.approx.f32 %[r], %[x];"
            : [r] "=f" (-> f32),
            : [x] "f" (x),
        );
    } else {
        return @sin(x);
    }
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + exp(-x));
}

pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// log(1 + exp(x)), the linear tail past 20 where the f32 exp would overflow.
pub inline fn softplus(x: f32) f32 {
    return if (x > 20.0) x else log2(1.0 + exp(x)) * 0.6931471805599453;
}

// Tanh-gelu folded to x*sigmoid(2c*(x + c3*x^3)), c = sqrt(2/pi), c3 = 0.044715,
// the form ops.act.geluTanh uses.
pub inline fn geluTanh(x: f32) f32 {
    return x * sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x));
}

pub inline fn geluQuick(x: f32) f32 {
    return x * sigmoid(1.702 * x);
}

/// 0.5x(1 + erf(x/sqrt2)) with the A&S 7.1.26 erf, matching ops.act.geluErfScalar.
pub inline fn geluErf(x: f32) f32 {
    const t_in = x * 0.7071067811865476;
    const ax = @abs(t_in);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    var er = 1.0 - poly * exp(-ax * ax);
    if (t_in < 0) er = -er;
    return 0.5 * x * (1.0 + er);
}

/// The f16 in the low 16 bits of `w`, as f32.
pub inline fn h16(w: u32) f32 {
    return @floatCast(@as(f16, @bitCast(@as(u16, @truncate(w)))));
}

pub inline fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

pub inline fn packH16(lo: f32, hi: f32) u32 {
    return @as(u32, f16Bits(lo)) | (@as(u32, f16Bits(hi)) << 16);
}

/// bf16 by round-to-nearest-even of the top 16 bits, what cvt.rn.bf16.f32 does.
pub inline fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const rne: u32 = ((bits >> 16) & 1) +% 0x7FFF;
    return @truncate((bits +% rne) >> 16);
}

/// Sum of `v` over the subgroup, in every lane. Every lane of the subgroup must
/// reach the call: the reduce is over active lanes.
pub inline fn sgSum(v: f32) f32 {
    if (is_ptx) {
        var x = v;
        comptime var off: u32 = 16;
        inline while (off >= 1) : (off /= 2) {
            const o: u32 = asm ("shfl.sync.bfly.b32 %[r], %[v], %[o], 0x1f, 0xffffffff;"
                : [r] "=r" (-> u32),
                : [v] "r" (@as(u32, @bitCast(x))),
                  [o] "r" (off),
            );
            x += @as(f32, @bitCast(o));
        }
        return x;
    } else {
        return asm (
            \\%r = OpGroupNonUniformFAdd %f32t %scope Reduce %val
            : [r] "" (-> f32),
            : [f32t] "t" (f32),
              [scope] "" (@as(u32, 3)),
              [val] "" (v),
        );
    }
}

/// Max of `v` over the subgroup, in every lane. Same participation rule as `sgSum`.
pub inline fn sgMax(v: f32) f32 {
    if (is_ptx) {
        var x = v;
        comptime var off: u32 = 16;
        inline while (off >= 1) : (off /= 2) {
            const o: u32 = asm ("shfl.sync.bfly.b32 %[r], %[v], %[o], 0x1f, 0xffffffff;"
                : [r] "=r" (-> u32),
                : [v] "r" (@as(u32, @bitCast(x))),
                  [o] "r" (off),
            );
            x = @max(x, @as(f32, @bitCast(o)));
        }
        return x;
    } else {
        return asm (
            \\%r = OpGroupNonUniformFMax %f32t %scope Reduce %val
            : [r] "" (-> f32),
            : [f32t] "t" (f32),
              [scope] "" (@as(u32, 3)),
              [val] "" (v),
        );
    }
}

// ---- entry points -----------------------------------------------------------

const elt = @import("dual/elt.zig");
const rows = @import("dual/rows.zig");
const quant = @import("dual/quant.zig");

fn Entry(comptime body: anytype) type {
    return struct {
        fn spv() callconv(.spirv_kernel) void {
            decorate();
            body(.{});
        }
        fn ptx(pa: G, pb: G, pc_: G, pd: G, k0: u32, k1: u32, k2: u32, k3: u32, k4: u32, k5: u32, k6: u32, g0: f32, g1: f32) callconv(.nvptx_kernel) void {
            body(Env{ .a = pa, .b = pb, .c = pc_, .d = pd, .us = .{ k0, k1, k2, k3, k4, k5, k6 }, .fs = .{ g0, g1 } });
        }
        pub const run = if (is_ptx) ptx else spv;
    };
}

/// Every body, by the entry name both backends launch it under, in dual_table.zig's
/// order. One body per name: NVPTX cannot export one kernel twice.
const bodies = .{
    .{ "add", elt.add },
    .{ "add_relu", elt.addRelu },
    .{ "add_scaled", elt.addScaled },
    .{ "relu", elt.relu },
    .{ "silu", elt.silu },
    .{ "silu_mul", elt.siluMul },
    .{ "sigmoid_mul", elt.sigmoidMul },
    .{ "gelu", elt.gelu },
    .{ "gelu_mul", elt.geluMul },
    .{ "gelu_quick", elt.geluQuick },
    .{ "gelu_quick_mul", elt.geluQuickMul },
    .{ "gelu_erf", elt.geluErf },
    .{ "geglu", elt.geglu },
    .{ "geglu_h16", elt.gegluH16 },
    .{ "softplus_gate", elt.softplusGate },
    .{ "copy", elt.copy },
    .{ "scale_f32", elt.scaleF32 },
    .{ "concat_ch", elt.concatCh },
    .{ "scale_concat", elt.scaleConcat },
    .{ "scale_i32", elt.scaleI32 },
    .{ "quantize_i8", elt.quantizeI8 },
    .{ "rms_apply_w", elt.rmsApplyW },
    .{ "rms_partial", elt.rmsPartial },
    .{ "rms_combine", elt.rmsCombine },
    .{ "modulate", elt.modulate },
    .{ "gated_add", elt.gatedAdd },
    .{ "gated_add16", elt.gatedAdd16 },
    .{ "rope_inter", elt.ropeInter },
    .{ "rope_half", elt.ropeHalf },
    .{ "rope_half_pos", elt.ropeHalfPos },
    .{ "rope_half_part", elt.ropeHalfPart },
    .{ "deinterleave2", elt.deinterleave2 },
    .{ "deinterleave3", elt.deinterleave3 },
    .{ "gdn_gates", elt.gdnGates },
    .{ "gdn_gates_batch", elt.gdnGatesBatch },
    .{ "gdn_conv_step", elt.gdnConvStep },
    .{ "gdn_conv_batch", elt.gdnConvBatch },
    .{ "gdn_conv_state", elt.gdnConvState },
    .{ "penalize", elt.penalize },
    .{ "argmax_reduce", elt.argmaxReduce },
    .{ "argmax_final", elt.argmaxFinal },
    .{ "topk_reduce", elt.topkReduce },
    .{ "head_pad_h16", elt.headPadH16 },
    .{ "head_unpad", elt.headUnpad },
    .{ "head_pad", elt.headPad },
    .{ "gather_head", elt.gatherHead },
    .{ "gather_vt", elt.gatherVt },
    .{ "scatter_head", elt.scatterHead },
    .{ "gather_head_b", elt.gatherHeadB },
    .{ "gather_vt_b", elt.gatherVtB },
    .{ "scatter_head_b", elt.scatterHeadB },
    .{ "gather_kmajor", elt.gatherKmajor },
    .{ "gather_kmajor_h16", elt.gatherKmajorH16 },
    .{ "gather_kmajor16", elt.gatherKmajor16 },
    .{ "f32_to_h16", elt.f32ToH16 },
    .{ "f32_to_h16_pad", elt.f32ToH16Pad },
    .{ "f32_to_bf16_pad", elt.f32ToBf16Pad },
    .{ "h16_to_h16_pad", elt.h16ToH16Pad },
    .{ "bf16_to_h16_pad", elt.bf16ToH16Pad },
    .{ "f16_to_f32", elt.f16ToF32 },
    .{ "add_h16", elt.addH16 },
    .{ "silu_mul_h16", elt.siluMulH16 },
    .{ "sigmoid_mul_h16", elt.sigmoidMulH16 },
    .{ "silu_mul16", elt.siluMul16 },
    .{ "sigmoid_mul_g16", elt.sigmoidMulG16 },
    .{ "kv_store_f16", elt.kvStoreF16 },
    .{ "pack_h16_kmajor", elt.packH16Kmajor },
    .{ "bias_compact", elt.biasCompact },
    .{ "bias_compact_h16", elt.biasCompactH16 },
    .{ "bias_add_f16", elt.biasAddF16 },
    .{ "bias_add_h16", elt.biasAddH16 },
    .{ "add_bias_rows", elt.addBiasRows },
    .{ "add_bias_rows_h16", elt.addBiasRowsH16 },
    .{ "gn_apply", elt.gnApply },
    .{ "gn_apply_h16", elt.gnApplyH16 },
    .{ "vae_norm", elt.vaeNorm },
    .{ "im2col", elt.im2col },
    .{ "im2col_sd", elt.im2colSd },
    .{ "im2col_sd_h16", elt.im2colSdH16 },
    .{ "gather_rows", elt.gatherRows },
    .{ "scatter_add_rows", elt.scatterAddRows },
    .{ "moe_combine", elt.moeCombine },
    .{ "gemv_combine", elt.gemvCombine },
    .{ "gemv_combine4", elt.gemvCombine4 },
    .{ "rope_imrope", elt.ropeImrope },
    .{ "rope_imrope_pos", elt.ropeImropePos },
    .{ "rope_vision", elt.ropeVision },
    .{ "rope_vision_gemma4", elt.ropeVisionGemma4 },
    .{ "im2col1d", elt.im2col1d },
    .{ "aa_up_snake", elt.aaUpSnake },
    .{ "aa_down", elt.aaDown },
    .{ "convt1d_ca", elt.convt1dCa },
    .{ "snake1d_ca", elt.snake1dCa },
    .{ "mean_heads_pool", elt.meanHeadsPool },
    .{ "dequant_fp8_f16", quant.fp8F16 },
    .{ "dequant_fp8_bf16", quant.fp8Bf16 },
    .{ "dequant_fp8_f32", quant.fp8F32 },
    .{ "dequant_q8_0_f16", quant.q8_0F16 },
    .{ "dequant_q8_0_bf16", quant.q8_0Bf16 },
    .{ "dequant_q8_0_f32", quant.q8_0F32 },
    .{ "dequant_q4_0_f16", quant.q4_0F16 },
    .{ "dequant_q4_0_bf16", quant.q4_0Bf16 },
    .{ "dequant_q4_0_f32", quant.q4_0F32 },
    .{ "dequant_q1_0_f16", quant.q1_0F16 },
    .{ "dequant_q1_0_bf16", quant.q1_0Bf16 },
    .{ "dequant_q1_0_f32", quant.q1_0F32 },
    .{ "dequant_q2_0_g64_f16", quant.q2_0G64F16 },
    .{ "dequant_q2_0_g64_bf16", quant.q2_0G64Bf16 },
    .{ "dequant_q2_0_g64_f32", quant.q2_0G64F32 },
    .{ "dequant_q2_0_g128_f16", quant.q2_0G128F16 },
    .{ "dequant_q2_0_g128_bf16", quant.q2_0G128Bf16 },
    .{ "dequant_q2_0_g128_f32", quant.q2_0G128F32 },
    .{ "dequant_iq4_nl_f16", quant.iq4NlF16 },
    .{ "dequant_iq4_nl_bf16", quant.iq4NlBf16 },
    .{ "dequant_iq4_nl_f32", quant.iq4NlF32 },
    .{ "dequant_iq4_xs_f16", quant.iq4XsF16 },
    .{ "dequant_iq4_xs_bf16", quant.iq4XsBf16 },
    .{ "dequant_iq4_xs_f32", quant.iq4XsF32 },
    .{ "dequant_q4_k_f16", quant.q4_kF16 },
    .{ "dequant_q4_k_bf16", quant.q4_kBf16 },
    .{ "dequant_q4_k_f32", quant.q4_kF32 },
    .{ "dequant_q5_k_f16", quant.q5_kF16 },
    .{ "dequant_q5_k_bf16", quant.q5_kBf16 },
    .{ "dequant_q5_k_f32", quant.q5_kF32 },
    .{ "dequant_q6_k_f16", quant.q6_kF16 },
    .{ "dequant_q6_k_bf16", quant.q6_kBf16 },
    .{ "dequant_q6_k_f32", quant.q6_kF32 },
    .{ "gn_combine", elt.gnCombine },
    .{ "softmax_partial", elt.softmaxPartial },
    .{ "softmax_combine", elt.softmaxCombine },
    .{ "rowscale_i8", elt.rowscaleI8 },
    .{ "qknorm_rope16", elt.qknormRope16 },
    .{ "qknorm_rope_f32", elt.qknormRopeF32 },
    .{ "l2norm_rows", rows.l2normRows },
    .{ "l2norm_rows_g", rows.l2normRowsG },
    .{ "rmsnorm", rows.rmsnorm },
    .{ "group_rmsnorm", rows.groupRmsnorm },
    .{ "rms_mod", rows.rmsMod },
    .{ "rms_mod_h16", rows.rmsModH16 },
    .{ "layernorm", rows.layernorm },
    .{ "layernorm_h16", rows.layernormH16 },
    .{ "ln_mod", rows.lnMod },
    .{ "gn_stats", rows.gnStats },
    .{ "gn_stats_h16", rows.gnStatsH16 },
    .{ "rowmax_i8", rows.rowmaxI8 },
};

comptime {
    if (bodies.len != table.entries.len) @compileError("dual.zig bodies and dual_table.zig entries differ in count");
    for (bodies, table.entries) |b_, t| {
        if (!std.mem.eql(u8, b_[0], t.name)) @compileError("dual.zig body order differs from dual_table.zig at " ++ b_[0]);
        @export(&Entry(b_[1]).run, .{ .name = b_[0] });
    }
}
