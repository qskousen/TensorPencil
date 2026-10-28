//! Kernels with ONE source for both GPU arms: this file compiles to SPIR-V for the
//! Vulkan `Context` (self-hosted backend, like eltwise.zig) and to PTX for the CUDA
//! `Backend` (the LLVM backend's assembly).
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
const abi = @import("abi.zig");
const builtin = @import("builtin");
const gpu = std.spirv;
const table = @import("dual_table.zig");

pub const is_ptx = builtin.target.cpu.arch == .nvptx64;

pub const Slot = enum { a, b, c, d };

/// An activation's element type, per slot: `ldT` / `stT` read and write it, so one
/// body serves f32, f16 and bf16 chains. A 16-bit store is a real 16-bit store (the
/// module carries StorageBuffer16BitAccess), so a thread owns one element, not a pair.
pub const Dt = table.Dt;

// ---- SPIR-V prelude ---------------------------------------------------------

const FBuf = extern struct { data: [1 << 28]f32 };
const Push = extern struct { u0: u32, u1: u32, u2: u32, u3: u32, u4: u32, u5: u32, f0: f32, f1: f32, u6: u32 };
const a = abi.buffer(FBuf, "a", 0);
const b = abi.buffer(FBuf, "b", 1);
const c = abi.buffer(FBuf, "c", 2);
const d = abi.buffer(FBuf, "d", 3);
/// A `vec4` ALIAS of binding 1, the activation slot: logical addressing has no
/// pointer to re-cast, so a whole-quad load has to come from a second variable
/// on the same binding with a wider element type (the trick `dp4a.zig` used for
/// its u16 weight view). Without it `ld4` is four scalar loads, and these GEMVs
/// are bound by load instructions.
const VBuf = extern struct { data: [1 << 26]@Vector(4, f32) };
const bv = abi.buffer(VBuf, "bv", 1);
/// 16-bit aliases of every binding, for `ldT` / `stT`: f16 as itself, bf16 as its bits.
const HBuf = extern struct { data: [1 << 29]f16 };
const UBuf = extern struct { data: [1 << 29]u16 };
const ah = abi.buffer(HBuf, "ah", 0);
const bh = abi.buffer(HBuf, "bh", 1);
const ch = abi.buffer(HBuf, "ch", 2);
const dh = abi.buffer(HBuf, "dh", 3);
const au = abi.buffer(UBuf, "au", 0);
const bu = abi.buffer(UBuf, "bu", 1);
const cu = abi.buffer(UBuf, "cu", 2);
const du = abi.buffer(UBuf, "du", 3);
const pc = abi.pushConstants(Push, "pc");
// Subgroup builtins std.spirv does not declare. The compiler decorates an input
// by its name, so these names are fixed.
extern const subgroup_size: u32 addrspace(.input);
extern const subgroup_local_invocation_id: u32 addrspace(.input);
extern const subgroup_id: u32 addrspace(.input);

/// `body`, which takes the activations' element type, at `dt`, as an entry body.
fn At2(comptime body: anytype, comptime in: Dt, comptime out: Dt) type {
    return struct {
        pub inline fn run(e: Env) void {
            body(e, in, out);
        }
    };
}

fn At(comptime body: anytype, comptime dt: Dt) type {
    return struct {
        pub inline fn run(e: Env) void {
            body(e, dt);
        }
    };
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
    inline fn ptr(e: Env, comptime s: Slot) G {
        return switch (s) {
            .a => e.a,
            .b => e.b,
            .c => e.c,
            .d => e.d,
        };
    }
    /// Element `i` of slot `s` holding `dt`, as f32.
    pub inline fn ldT(e: Env, comptime s: Slot, comptime dt: Dt, i: u32) f32 {
        return switch (dt) {
            .f32 => e.ptr(s)[i],
            .f16 => @floatCast(@as([*]addrspace(.global) f16, @ptrCast(e.ptr(s)))[i]),
            .bf16 => @bitCast(@as(u32, @as([*]addrspace(.global) u16, @ptrCast(e.ptr(s)))[i]) << 16),
        };
    }
    /// Store `v` as element `i` of slot `s` holding `dt`, rounding to nearest even.
    pub inline fn stT(e: Env, comptime s: Slot, comptime dt: Dt, i: u32, v: f32) void {
        switch (dt) {
            .f32 => e.ptr(s)[i] = v,
            .f16 => @as([*]addrspace(.global) f16, @ptrCast(e.ptr(s)))[i] = @floatCast(v),
            .bf16 => @as([*]addrspace(.global) u16, @ptrCast(e.ptr(s)))[i] = bf16Bits(v),
        }
    }
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
    /// Four consecutive f32 from a 4-ALIGNED index, as one 16-byte load. A body
    /// that reads a whole group of activations or quant words takes this rather
    /// than four `ld`s: these kernels are bound by load instructions, not by
    /// bandwidth, so the count is what the time follows.
    ///
    /// Every slot is a device allocation (256-byte aligned) offset by whole rows,
    /// which is what makes the 16-byte cast sound; `i % 4 == 0` is the caller's.
    pub inline fn ld4(e: Env, comptime s: Slot, i: u32) [4]f32 {
        const V = @Vector(4, f32);
        const base: [*]addrspace(.global) f32 = switch (s) {
            .a => e.a,
            .b => e.b,
            .c => e.c,
            .d => e.d,
        };
        const v: [*]align(16) addrspace(.global) const V = @ptrCast(@alignCast(base));
        return v[i >> 2];
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
    pub inline fn ldT(e: Env, comptime s: Slot, comptime dt: Dt, i: u32) f32 {
        return switch (dt) {
            .f32 => e.ld(s, i),
            .f16 => @floatCast(switch (s) {
                .a => ah.data[i],
                .b => bh.data[i],
                .c => ch.data[i],
                .d => dh.data[i],
            }),
            .bf16 => @bitCast(@as(u32, switch (s) {
                .a => au.data[i],
                .b => bu.data[i],
                .c => cu.data[i],
                .d => du.data[i],
            }) << 16),
        };
    }
    pub inline fn stT(e: Env, comptime s: Slot, comptime dt: Dt, i: u32, v: f32) void {
        switch (dt) {
            .f32 => e.st(s, i, v),
            .f16 => switch (s) {
                .a => ah.data[i] = @floatCast(v),
                .b => bh.data[i] = @floatCast(v),
                .c => ch.data[i] = @floatCast(v),
                .d => dh.data[i] = @floatCast(v),
            },
            .bf16 => switch (s) {
                .a => au.data[i] = bf16Bits(v),
                .b => bu.data[i] = bf16Bits(v),
                .c => cu.data[i] = bf16Bits(v),
                .d => du.data[i] = bf16Bits(v),
            },
        }
    }
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
    /// `ld4` on SPIR-V. Slot b reads through the `vec4` alias, one load for the
    /// quad; the rest have no alias and fall back to four scalars.
    pub inline fn ld4(e: Env, comptime s: Slot, i: u32) [4]f32 {
        if (s == .b) return bv.data[i >> 2];
        return .{ e.ld(s, i), e.ld(s, i + 1), e.ld(s, i + 2), e.ld(s, i + 3) };
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
        return subgroup_size;
    }
    pub inline fn sgLane(_: Env) u32 {
        return subgroup_local_invocation_id;
    }
    pub inline fn sgId(_: Env) u32 {
        return subgroup_id;
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
pub inline fn sgPerWg(e: Env) u32 {
    return table.row_wg / e.sgSize();
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

/// Saturates cleanly at both ends: the exp under `sigmoid` goes to inf or 0.
pub inline fn tanh(x: f32) f32 {
    return 2.0 * sigmoid(2.0 * x) - 1.0;
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

/// Sign-extend byte 0 of `v`.
pub inline fn sext8(v: u32) i32 {
    return @as(i32, @bitCast(v << 24)) >> 24;
}

/// Four int8 multiply-accumulates in one go: `acc + dot(w4, x4)` over the packed
/// signed bytes. The hardware instruction on PTX; on SPIR-V the arithmetic, since
/// `OpSDot` needs a capability the shared module must not carry (it would refuse
/// to build on a device without the integer dot, taking all 160 kernels with it).
pub inline fn dp4a(w4: u32, x4: u32, acc: i32) i32 {
    if (is_ptx) {
        return asm ("dp4a.s32.s32 %[r], %[w], %[x], %[c];"
            : [r] "=r" (-> i32),
            : [w] "r" (w4),
              [x] "r" (x4),
              [c] "r" (acc),
        );
    }
    var s = acc;
    inline for (0..4) |i| s += sext8(w4 >> @intCast(i * 8)) * sext8(x4 >> @intCast(i * 8));
    return s;
}

/// `dp4a` as SPIR-V's `OpSDot` over packed bytes, for the `_sdot` kernels alone: the
/// capability it needs is added to their pipelines only, on a device that has it.
pub inline fn sdot4(w4: u32, x4: u32, acc: i32) i32 {
    if (is_ptx) return dp4a(w4, x4, acc);
    const dot = asm (
        \\%r = OpSDot %i32t %w %x PackedVectorFormat4x8Bit
        : [r] "" (-> i32),
        : [i32t] "t" (i32),
          [w] "" (w4),
          [x] "" (x4),
    );
    return acc + dot;
}

/// Per-byte `a - b`, no borrow between bytes. One instruction on PTX; on SPIR-V
/// the two byte lanes are done apart so an add cannot carry out of its byte (the
/// naive `(a ^ m) + 1` form is wrong for a zero byte, which the codebooks do not
/// hold today and a future one might).
pub inline fn subBytes(x: u32, y: u32) u32 {
    if (is_ptx) {
        // The addend is an operand, not a literal: ptxas reads a bare `0` there
        // as a video selector and refuses it.
        return asm ("vsub4.u32.u32.u32 %[r], %[x], %[y], %[z];"
            : [r] "=r" (-> u32),
            : [x] "r" (x),
              [y] "r" (y),
              [z] "r" (@as(u32, 0)),
        );
    }
    // The guard bit above each byte absorbs its borrow. Without it the borrow
    // ripples through the zeroed gap into the NEXT lane of the same word, which
    // is right for three bytes in four and silently wrong for the rest.
    const lo = ((x & 0x00FF00FF) | 0x01000100) -% (y & 0x00FF00FF);
    const hi = (((x >> 8) & 0x00FF00FF) | 0x01000100) -% ((y >> 8) & 0x00FF00FF);
    return (lo & 0x00FF00FF) | ((hi & 0x00FF00FF) << 8);
}

/// The four low bits of `bits` as a per-byte mask of 0x00 / 0xFF.
///
/// The multiply lays four copies of the nibble at bits 0, 7, 14 and 21, whose
/// 4-bit fields do not overlap, so bit `i` lands at bit `8i` and the mask picks
/// it out. Three instructions where shifting each bit into place takes eleven,
/// which is a fifth of these GEMVs' time (measured: isolating the sign
/// application entirely moved iq2_xxs 32%).
pub inline fn byteMask(bits: u32) u32 {
    return ((bits *% 0x00204081) & 0x01010101) *% 255;
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

/// `v` from lane `lane ^ mask`, every lane of the subgroup participating. PTX only:
/// the SPIR-V module declares no shuffle capability, and adding one would refuse the
/// whole module on a device without it.
pub inline fn sgXor(v: f32, comptime mask: u32) f32 {
    if (!is_ptx) @compileError("sgXor has no SPIR-V arm");
    const o: u32 = asm ("shfl.sync.bfly.b32 %[r], %[v], %[o], 0x1f, 0xffffffff;"
        : [r] "=r" (-> u32),
        : [v] "r" (@as(u32, @bitCast(v))),
          [o] "r" (mask),
    );
    return @bitCast(o);
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
const attn = @import("dual/attn.zig");

fn Entry(comptime body: anytype) type {
    return struct {
        // The body stays a function of its own: inlined into the entry point (Zig
        // 0.17's default), the NVIDIA driver built the q4_0/q6_k/iq GEMVs 4-9% slower
        // from otherwise identical SPIR-V.
        fn spv() callconv(abi.kernel) void {
            @call(.never_inline, spvBody, .{});
        }
        fn spvBody() void {
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
    .{ "add", At(elt.add, .f32).run },
    .{ "add_relu", elt.addRelu },
    .{ "add_scaled", At(elt.addScaled, .f32).run },
    .{ "compact_acc", At(elt.compactAcc, .f32).run },
    .{ "relu", elt.relu },
    .{ "silu", At(elt.silu, .f32).run },
    .{ "silu_mul", At(elt.siluMul, .f32).run },
    .{ "sigmoid_mul", At(elt.sigmoidMul, .f32).run },
    .{ "gelu", At(elt.gelu, .f32).run },
    .{ "gelu_mul", elt.geluMul },
    .{ "gelu_quick", elt.geluQuick },
    .{ "gelu_quick_mul", elt.geluQuickMul },
    .{ "gelu_erf", At(elt.geluErf, .f32).run },
    .{ "geglu", At(elt.geglu, .f32).run },
    .{ "softplus_gate", elt.softplusGate },
    .{ "copy", At2(elt.copy, .f32, .f32).run },
    .{ "scale_f32", elt.scaleF32 },
    .{ "concat_ch", At(elt.concatCh, .f32).run },
    .{ "scale_concat", elt.scaleConcat },
    .{ "scale_i32", At(elt.scaleI32, .f32).run },
    .{ "quantize_i8", elt.quantizeI8 },
    .{ "rms_apply_w", elt.rmsApplyW },
    .{ "rms_partial", elt.rmsPartial },
    .{ "rms_combine", elt.rmsCombine },
    .{ "modulate", At(elt.modulate, .f32).run },
    .{ "gated_add", At(elt.gatedAdd, .f32).run },
    .{ "gated_add16", elt.gatedAdd16 },
    .{ "rope_inter", At(elt.ropeInter, .f32).run },
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
    .{ "pad_h16", At(elt.padH16, .f32).run },
    .{ "pad_bf16", At(elt.padBf16, .f32).run },
    .{ "conv_w_h16", At(elt.convW16, .f16).run },
    .{ "bf16_to_f32", elt.bf16ToF32 },
    .{ "f16_to_f32", elt.f16ToF32 },
    .{ "silu_mul_h16", elt.siluMulH16 },
    .{ "sigmoid_mul_h16", elt.sigmoidMulH16 },
    .{ "silu_mul16", elt.siluMul16 },
    .{ "sigmoid_mul_g16", elt.sigmoidMulG16 },
    .{ "kv_store_f16", elt.kvStoreF16 },
    .{ "transpose_kmajor16", elt.transposeKmajor16 },
    .{ "bias_compact", At(elt.biasCompact, .f32).run },
    .{ "bias_add", At2(elt.biasAdd, .f16, .f32).run },
    .{ "add_bias_rows", At(elt.addBiasRows, .f32).run },
    .{ "gn_apply", At(elt.gnApply, .f32).run },
    .{ "im2col", elt.im2col },
    .{ "im2col_sd", At(elt.im2colSd, .f32).run },
    .{ "gather_rows", elt.gatherRows },
    .{ "scatter_add_rows", elt.scatterAddRows },
    .{ "moe_combine", elt.moeCombine },
    .{ "moe_route", elt.moeRoute },
    .{ "gemv_combine", elt.gemvCombine },
    .{ "gemv_combine4", elt.gemvCombine4 },
    .{ "modulate_pr", At(elt.modulatePerRow, .f32).run },
    .{ "gated_add_pr", At(elt.gatedAddPerRow, .f32).run },
    .{ "dw_conv3", At(elt.dwConv3, .f32).run },
    .{ "col_mean", At(elt.colMean, .f32).run },
    .{ "mul_cols_sigmoid", At(elt.mulColsSigmoid, .f32).run },
    .{ "nerf_feat", At(elt.nerfFeat, .f32).run },
    .{ "patch_scatter", elt.patchScatter },
    .{ "win_gather", elt.winGather },
    .{ "win_scatter", elt.winScatter },
    .{ "rope_imrope", elt.ropeImrope },
    .{ "rope_imrope_pos", elt.ropeImropePos },
    .{ "rope_vision", elt.ropeVision },
    .{ "rope_vision_gemma4", elt.ropeVisionGemma4 },
    .{ "rope_half_span_pos", elt.ropeHalfSpanPos },
    .{ "rope_inter_span_pos", elt.ropeInterSpanPos },
    .{ "pixel_shuffle", elt.pixelShuffle },
    .{ "im2col_stride", elt.im2colStride },
    .{ "im2col1d", elt.im2col1d },
    .{ "im2col3d", elt.im2col3d },
    .{ "aa_up_snake", elt.aaUpSnake },
    .{ "aa_down", elt.aaDown },
    .{ "convt1d_ca", elt.convt1dCa },
    .{ "snake1d_ca", elt.snake1dCa },
    .{ "mean_heads_pool", elt.meanHeadsPool },
    .{ "dequant_fp8_f16", quant.fp8F16 },
    .{ "dequant_fp8_bf16", quant.fp8Bf16 },
    .{ "dequant_q8_0_f16", quant.q8_0F16 },
    .{ "dequant_q8_0_bf16", quant.q8_0Bf16 },
    .{ "dequant_q4_0_f16", quant.q4_0F16 },
    .{ "dequant_q4_0_bf16", quant.q4_0Bf16 },
    .{ "dequant_q1_0_f16", quant.q1_0F16 },
    .{ "dequant_q1_0_bf16", quant.q1_0Bf16 },
    .{ "dequant_q2_0_g64_f16", quant.q2_0G64F16 },
    .{ "dequant_q2_0_g64_bf16", quant.q2_0G64Bf16 },
    .{ "dequant_q2_0_g128_f16", quant.q2_0G128F16 },
    .{ "dequant_q2_0_g128_bf16", quant.q2_0G128Bf16 },
    .{ "dequant_iq4_nl_f16", quant.iq4NlF16 },
    .{ "dequant_iq4_nl_bf16", quant.iq4NlBf16 },
    .{ "dequant_iq4_xs_f16", quant.iq4XsF16 },
    .{ "dequant_iq4_xs_bf16", quant.iq4XsBf16 },
    .{ "dequant_q4_k_f16", quant.q4_kF16 },
    .{ "dequant_q4_k_bf16", quant.q4_kBf16 },
    .{ "dequant_q5_k_f16", quant.q5_kF16 },
    .{ "dequant_q5_k_bf16", quant.q5_kBf16 },
    .{ "dequant_q6_k_f16", quant.q6_kF16 },
    .{ "dequant_q6_k_bf16", quant.q6_kBf16 },
    .{ "dequant_q2_k_f16", quant.q2_kF16 },
    .{ "dequant_q2_k_bf16", quant.q2_kBf16 },
    .{ "dequant_q3_k_f16", quant.q3_kF16 },
    .{ "dequant_q3_k_bf16", quant.q3_kBf16 },
    .{ "dequant_iq2_xxs_f16", quant.iq2XxsF16 },
    .{ "dequant_iq2_xxs_bf16", quant.iq2XxsBf16 },
    .{ "dequant_iq2_xs_f16", quant.iq2XsF16 },
    .{ "dequant_iq2_xs_bf16", quant.iq2XsBf16 },
    .{ "dequant_iq3_xxs_f16", quant.iq3XxsF16 },
    .{ "dequant_iq3_xxs_bf16", quant.iq3XxsBf16 },
    .{ "dequant_iq3_s_f16", quant.iq3SF16 },
    .{ "dequant_iq3_s_bf16", quant.iq3SBf16 },
    .{ "gn_combine", elt.gnCombine },
    .{ "softmax_partial", elt.softmaxPartial },
    .{ "softmax_combine", elt.softmaxCombine },
    .{ "rowscale_i8", elt.rowscaleI8 },
    .{ "i8row_rot_bf16", quant.i8RowRotBf16 },
    .{ "i4row_rot_bf16", quant.i4RowRotBf16 },
    .{ "dequant_i8row_bf16", quant.i8RowBf16 },
    .{ "i8row_rot_f16", quant.i8RowRotF16 },
    .{ "i4row_rot_f16", quant.i4RowRotF16 },
    .{ "dequant_i8row_f16", quant.i8RowF16 },
    .{ "i8row_rot_f32", quant.i8RowRotF32 },
    .{ "act_rot_f32", quant.actRotF32 },
    .{ "act_rot_direct", quant.actRotDirect },
    .{ "gemv_i8row", quant.gemvI8Row },
    .{ "i4row_rot_f32", quant.i4RowRotF32 },
    .{ "dequant_i8row_f32", quant.i8RowF32 },
    .{ "wo_unpack_f32", quant.woUnpackF32 },
    .{ "transpose_pad_f32", elt.transposePadF32 },
    .{ "im2col_h16", At(elt.im2colH16, .f32).run },
    .{ "im2col_h16_f16", At(elt.im2colH16, .f16).run },
    .{ "im2col_h16_bf16", At(elt.im2colH16, .bf16).run },
    .{ "im2col_bf16", At(elt.im2colBf16, .f32).run },
    .{ "im2col_bf16_f16", At(elt.im2colBf16, .f16).run },
    .{ "im2col_bf16_bf16", At(elt.im2colBf16, .bf16).run },
    .{ "qknorm_rope", At(elt.qknormRope, .f32).run },
    .{ "l2norm_rows", rows.l2normRows },
    .{ "l2norm_rows_g", rows.l2normRowsG },
    .{ "act_steer", At(rows.actSteer, .f32).run },
    .{ "attn_rows_f32", attn.attnRowsF32 },
    .{ "attn_rows_f16", attn.attnRowsF16 },
    .{ "attn_rows_q8", attn.attnRowsQ8 },
    .{ "attn_merge", attn.attnMerge },
    .{ "quantize_q8_1", quant.quantizeQ81 },
    .{ "gdn_delta_chunk_d64", attn.gdnDeltaChunkD64 },
    .{ "gdn_delta_chunk_d128", attn.gdnDeltaChunkD128 },
    .{ "rmsnorm", At(rows.rmsnorm, .f32).run },
    .{ "qk_norm_rope", rows.qkNormRope },
    .{ "group_rmsnorm", rows.groupRmsnorm },
    .{ "gemv_rows_f32", rows.gemvRowsF32 },
    .{ "rms_mod", At(rows.rmsMod, .f32).run },
    .{ "rms_mod_h16", At(rows.rmsModH16, .f32).run },
    .{ "layernorm", At(rows.layernorm, .f32).run },
    .{ "ln_mod", At(rows.lnMod, .f32).run },
    .{ "gn_stats", At(rows.gnStats, .f32).run },
    .{ "rowmax_i8", rows.rowmaxI8 },
    .{ "gemv_dual_q2_k", quant.gemvQ2_k },
    .{ "gemv_dual_q2_k_q8", quant.gemvQ2_kQ8 },
    .{ "gemv_dual_q2_k_sdot", quant.GemvQ8Sdot(.q2_k).run },
    .{ "gemv_dual_q3_k", quant.gemvQ3_k },
    .{ "gemv_dual_q3_k_q8", quant.gemvQ3_kQ8 },
    .{ "gemv_dual_q3_k_sdot", quant.GemvQ8Sdot(.q3_k).run },
    .{ "gemv_dual_q4_k", quant.gemvQ4_k },
    .{ "gemv_dual_q4_k_q8", quant.gemvQ4_kQ8 },
    .{ "gemv_dual_q4_k_sdot", quant.GemvQ8Sdot(.q4_k).run },
    .{ "gemv_dual_q8_0", quant.gemvQ8_0 },
    .{ "gemv_dual_q8_0_q8", quant.gemvQ8_0Q8 },
    .{ "gemv_dual_q8_0_sdot", quant.GemvQ8Sdot(.q8_0).run },
    .{ "gemv_dual_q4_0", quant.gemvQ4_0 },
    .{ "gemv_dual_q4_0_q8", quant.gemvQ4_0Q8 },
    .{ "gemv_dual_q4_0_sdot", quant.GemvQ8Sdot(.q4_0).run },
    .{ "gemv_dual_iq4_nl", quant.gemvIq4Nl },
    .{ "gemv_dual_iq4_nl_q8", quant.gemvIq4NlQ8 },
    .{ "gemv_dual_iq4_nl_sdot", quant.GemvQ8Sdot(.iq4_nl).run },
    .{ "gemv_dual_iq4_xs", quant.gemvIq4Xs },
    .{ "gemv_dual_iq4_xs_q8", quant.gemvIq4XsQ8 },
    .{ "gemv_dual_iq4_xs_sdot", quant.GemvQ8Sdot(.iq4_xs).run },
    .{ "gemv_dual_q5_k", quant.gemvQ5_k },
    .{ "gemv_dual_q5_k_q8", quant.gemvQ5_kQ8 },
    .{ "gemv_dual_q5_k_sdot", quant.GemvQ8Sdot(.q5_k).run },
    .{ "gemv_dual_q6_k", quant.gemvQ6_k },
    .{ "gemv_dual_q6_k_q8", quant.gemvQ6_kQ8 },
    .{ "gemv_dual_q6_k_sdot", quant.GemvQ8Sdot(.q6_k).run },
    .{ "gemv_dual_q1_0", quant.gemvQ1_0 },
    .{ "gemv_dual_q1_0_q8", quant.gemvQ1_0Q8 },
    .{ "gemv_dual_q1_0_sdot", quant.GemvQ8Sdot(.q1_0).run },
    .{ "gemv_dual_q2_0_g64", quant.gemvQ2_0G64 },
    .{ "gemv_dual_q2_0_g64_q8", quant.gemvQ2_0G64Q8 },
    .{ "gemv_dual_q2_0_g64_sdot", quant.GemvQ8Sdot(.q2_0_g64).run },
    .{ "gemv_dual_q2_0_g128", quant.gemvQ2_0G128 },
    .{ "gemv_dual_q2_0_g128_q8", quant.gemvQ2_0G128Q8 },
    .{ "gemv_dual_q2_0_g128_sdot", quant.GemvQ8Sdot(.q2_0_g128).run },
    .{ "gemv_dual_iq2_xxs", quant.gemvIq2Xxs },
    .{ "gemv_dual_iq2_xxs_q8", quant.gemvIq2XxsQ8 },
    .{ "gemv_dual_iq2_xxs_sdot", quant.GemvQ8Sdot(.iq2_xxs).run },
    .{ "gemv_dual_iq2_xs", quant.gemvIq2Xs },
    .{ "gemv_dual_iq2_xs_q8", quant.gemvIq2XsQ8 },
    .{ "gemv_dual_iq2_xs_sdot", quant.GemvQ8Sdot(.iq2_xs).run },
    .{ "gemv_dual_iq3_xxs", quant.gemvIq3Xxs },
    .{ "gemv_dual_iq3_xxs_q8", quant.gemvIq3XxsQ8 },
    .{ "gemv_dual_iq3_xxs_sdot", quant.GemvQ8Sdot(.iq3_xxs).run },
    .{ "gemv_dual_iq3_s", quant.gemvIq3S },
    .{ "gemv_dual_iq3_s_q8", quant.gemvIq3SQ8 },
    .{ "gemv_dual_iq3_s_sdot", quant.GemvQ8Sdot(.iq3_s).run },
    .{ "gemv4_dual_q2_k", quant.GemvN(.q2_k).run },
    .{ "gemv4_dual_q2_k_sdot", quant.GemvQ8N(.q2_k).run },
    .{ "gemv4_dual_q3_k", quant.GemvN(.q3_k).run },
    .{ "gemv4_dual_q3_k_sdot", quant.GemvQ8N(.q3_k).run },
    .{ "gemv4_dual_q4_k", quant.GemvN(.q4_k).run },
    .{ "gemv4_dual_q4_k_sdot", quant.GemvQ8N(.q4_k).run },
    .{ "gemv4_dual_q8_0", quant.GemvN(.q8_0).run },
    .{ "gemv4_dual_q8_0_sdot", quant.GemvQ8N(.q8_0).run },
    .{ "gemv4_dual_q4_0", quant.GemvN(.q4_0).run },
    .{ "gemv4_dual_q4_0_sdot", quant.GemvQ8N(.q4_0).run },
    .{ "gemv4_dual_iq4_nl", quant.GemvN(.iq4_nl).run },
    .{ "gemv4_dual_iq4_nl_sdot", quant.GemvQ8N(.iq4_nl).run },
    .{ "gemv4_dual_iq4_xs", quant.GemvN(.iq4_xs).run },
    .{ "gemv4_dual_iq4_xs_sdot", quant.GemvQ8N(.iq4_xs).run },
    .{ "gemv4_dual_q5_k", quant.GemvN(.q5_k).run },
    .{ "gemv4_dual_q5_k_sdot", quant.GemvQ8N(.q5_k).run },
    .{ "gemv4_dual_q6_k", quant.GemvN(.q6_k).run },
    .{ "gemv4_dual_q6_k_sdot", quant.GemvQ8N(.q6_k).run },
    .{ "gemv4_dual_q1_0", quant.GemvN(.q1_0).run },
    .{ "gemv4_dual_q1_0_sdot", quant.GemvQ8N(.q1_0).run },
    .{ "gemv4_dual_q2_0_g64", quant.GemvN(.q2_0_g64).run },
    .{ "gemv4_dual_q2_0_g64_sdot", quant.GemvQ8N(.q2_0_g64).run },
    .{ "gemv4_dual_q2_0_g128", quant.GemvN(.q2_0_g128).run },
    .{ "gemv4_dual_q2_0_g128_sdot", quant.GemvQ8N(.q2_0_g128).run },
    .{ "gemv4_dual_iq2_xxs", quant.GemvN(.iq2_xxs).run },
    .{ "gemv4_dual_iq2_xxs_sdot", quant.GemvQ8N(.iq2_xxs).run },
    .{ "gemv4_dual_iq2_xs", quant.GemvN(.iq2_xs).run },
    .{ "gemv4_dual_iq2_xs_sdot", quant.GemvQ8N(.iq2_xs).run },
    .{ "gemv4_dual_iq3_xxs", quant.GemvN(.iq3_xxs).run },
    .{ "gemv4_dual_iq3_xxs_sdot", quant.GemvQ8N(.iq3_xxs).run },
    .{ "gemv4_dual_iq3_s", quant.GemvN(.iq3_s).run },
    .{ "gemv4_dual_iq3_s_sdot", quant.GemvQ8N(.iq3_s).run },
    .{ "gemvx_dual_q2_k", quant.GemvX(.q2_k, false).run },
    .{ "gemvx_dual_q2_k_sdot", quant.GemvX(.q2_k, true).run },
    .{ "gemvx_dual_q3_k", quant.GemvX(.q3_k, false).run },
    .{ "gemvx_dual_q3_k_sdot", quant.GemvX(.q3_k, true).run },
    .{ "gemvx_dual_q4_k", quant.GemvX(.q4_k, false).run },
    .{ "gemvx_dual_q4_k_sdot", quant.GemvX(.q4_k, true).run },
    .{ "gemvx_dual_q8_0", quant.GemvX(.q8_0, false).run },
    .{ "gemvx_dual_q8_0_sdot", quant.GemvX(.q8_0, true).run },
    .{ "gemvx_dual_q4_0", quant.GemvX(.q4_0, false).run },
    .{ "gemvx_dual_q4_0_sdot", quant.GemvX(.q4_0, true).run },
    .{ "gemvx_dual_iq4_nl", quant.GemvX(.iq4_nl, false).run },
    .{ "gemvx_dual_iq4_nl_sdot", quant.GemvX(.iq4_nl, true).run },
    .{ "gemvx_dual_iq4_xs", quant.GemvX(.iq4_xs, false).run },
    .{ "gemvx_dual_iq4_xs_sdot", quant.GemvX(.iq4_xs, true).run },
    .{ "gemvx_dual_q5_k", quant.GemvX(.q5_k, false).run },
    .{ "gemvx_dual_q5_k_sdot", quant.GemvX(.q5_k, true).run },
    .{ "gemvx_dual_q6_k", quant.GemvX(.q6_k, false).run },
    .{ "gemvx_dual_q6_k_sdot", quant.GemvX(.q6_k, true).run },
    .{ "gemvx_dual_q1_0", quant.GemvX(.q1_0, false).run },
    .{ "gemvx_dual_q1_0_sdot", quant.GemvX(.q1_0, true).run },
    .{ "gemvx_dual_q2_0_g64", quant.GemvX(.q2_0_g64, false).run },
    .{ "gemvx_dual_q2_0_g64_sdot", quant.GemvX(.q2_0_g64, true).run },
    .{ "gemvx_dual_q2_0_g128", quant.GemvX(.q2_0_g128, false).run },
    .{ "gemvx_dual_q2_0_g128_sdot", quant.GemvX(.q2_0_g128, true).run },
    .{ "gemvx_dual_iq2_xxs", quant.GemvX(.iq2_xxs, false).run },
    .{ "gemvx_dual_iq2_xxs_sdot", quant.GemvX(.iq2_xxs, true).run },
    .{ "gemvx_dual_iq2_xs", quant.GemvX(.iq2_xs, false).run },
    .{ "gemvx_dual_iq2_xs_sdot", quant.GemvX(.iq2_xs, true).run },
    .{ "gemvx_dual_iq3_xxs", quant.GemvX(.iq3_xxs, false).run },
    .{ "gemvx_dual_iq3_xxs_sdot", quant.GemvX(.iq3_xxs, true).run },
    .{ "gemvx_dual_iq3_s", quant.GemvX(.iq3_s, false).run },
    .{ "gemvx_dual_iq3_s_sdot", quant.GemvX(.iq3_s, true).run },
    .{ "attn_rows_g2_f32", attn.AttnRowsG(.f32, 2).run },
    .{ "attn_rows_g2_f16", attn.AttnRowsG(.f16, 2).run },
    .{ "attn_rows_g2_q8", attn.AttnRowsG(.q8_0, 2).run },
    .{ "attn_rows_g4_f32", attn.AttnRowsG(.f32, 4).run },
    .{ "attn_rows_g4_f16", attn.AttnRowsG(.f16, 4).run },
    .{ "attn_rows_g4_q8", attn.AttnRowsG(.q8_0, 4).run },
    .{ "attn_rows_g8_f32", attn.AttnRowsG(.f32, 8).run },
    .{ "attn_rows_g8_f16", attn.AttnRowsG(.f16, 8).run },
    .{ "attn_rows_g8_q8", attn.AttnRowsG(.q8_0, 8).run },
    .{ "add_f16", At(elt.add, .f16).run },
    .{ "add_bf16", At(elt.add, .bf16).run },
    .{ "geglu_f16", At(elt.geglu, .f16).run },
    .{ "geglu_bf16", At(elt.geglu, .bf16).run },
    .{ "bias_compact_f16", At(elt.biasCompact, .f16).run },
    .{ "bias_compact_bf16", At(elt.biasCompact, .bf16).run },
    .{ "add_bias_rows_f16", At(elt.addBiasRows, .f16).run },
    .{ "add_bias_rows_bf16", At(elt.addBiasRows, .bf16).run },
    .{ "gn_apply_f16", At(elt.gnApply, .f16).run },
    .{ "gn_apply_bf16", At(elt.gnApply, .bf16).run },
    .{ "im2col_sd_f16", At(elt.im2colSd, .f16).run },
    .{ "im2col_sd_bf16", At(elt.im2colSd, .bf16).run },
    .{ "qknorm_rope_f16", At(elt.qknormRope, .f16).run },
    .{ "qknorm_rope_bf16", At(elt.qknormRope, .bf16).run },
    .{ "layernorm_f16", At(rows.layernorm, .f16).run },
    .{ "layernorm_bf16", At(rows.layernorm, .bf16).run },
    .{ "layernorm_f32_f16", At(rows.layernormFrom32, .f16).run },
    .{ "layernorm_f32_bf16", At(rows.layernormFrom32, .bf16).run },
    .{ "gn_stats_f16", At(rows.gnStats, .f16).run },
    .{ "gn_stats_bf16", At(rows.gnStats, .bf16).run },
    .{ "pad_h16_f16", At(elt.padH16, .f16).run },
    .{ "pad_h16_bf16", At(elt.padH16, .bf16).run },
    .{ "pad_bf16_f16", At(elt.padBf16, .f16).run },
    .{ "pad_bf16_bf16", At(elt.padBf16, .bf16).run },
    .{ "bias_add_f16", At2(elt.biasAdd, .f16, .f16).run },
    .{ "bias_add_bf16", At2(elt.biasAdd, .bf16, .bf16).run },
    .{ "conv_w_bf16", At(elt.convW16, .bf16).run },
    .{ "bias_add_bf16_f32", At2(elt.biasAdd, .bf16, .f32).run },
    .{ "scale_i32_f16", At(elt.scaleI32, .f16).run },
    .{ "scale_i32_bf16", At(elt.scaleI32, .bf16).run },
    .{ "copy_f16", At2(elt.copy, .f16, .f16).run },
    .{ "copy_bf16", At2(elt.copy, .bf16, .bf16).run },
    .{ "copy_f32_f16", At2(elt.copy, .f32, .f16).run },
    .{ "copy_f32_bf16", At2(elt.copy, .f32, .bf16).run },
    .{ "copy_f16_f32", At2(elt.copy, .f16, .f32).run },
    .{ "copy_bf16_f32", At2(elt.copy, .bf16, .f32).run },
    .{ "concat_ch_f16", At(elt.concatCh, .f16).run },
    .{ "concat_ch_bf16", At(elt.concatCh, .bf16).run },
    .{ "compact_acc_f16", At(elt.compactAcc, .f16).run },
    .{ "compact_acc_bf16", At(elt.compactAcc, .bf16).run },
    .{ "silu_mul_f16", At(elt.siluMul, .f16).run },
    .{ "silu_mul_bf16", At(elt.siluMul, .bf16).run },
    .{ "sigmoid_mul_f16", At(elt.sigmoidMul, .f16).run },
    .{ "sigmoid_mul_bf16", At(elt.sigmoidMul, .bf16).run },
    .{ "gated_add_f16", At(elt.gatedAdd, .f16).run },
    .{ "gated_add_bf16", At(elt.gatedAdd, .bf16).run },
    .{ "rope_inter_f16", At(elt.ropeInter, .f16).run },
    .{ "rope_inter_bf16", At(elt.ropeInter, .bf16).run },
    .{ "act_steer_f16", At(rows.actSteer, .f16).run },
    .{ "act_steer_bf16", At(rows.actSteer, .bf16).run },
    .{ "vae_norm", At(rows.vaeNorm, .f32).run },
    .{ "vae_norm_f16", At(rows.vaeNorm, .f16).run },
    .{ "vae_norm_bf16", At(rows.vaeNorm, .bf16).run },
    .{ "rmsnorm_f16", At(rows.rmsnorm, .f16).run },
    .{ "rmsnorm_bf16", At(rows.rmsnorm, .bf16).run },
    .{ "rms_mod_f16", At(rows.rmsMod, .f16).run },
    .{ "rms_mod_bf16", At(rows.rmsMod, .bf16).run },
    .{ "rms_mod_h16_f16", At(rows.rmsModH16, .f16).run },
    .{ "rms_mod_h16_bf16", At(rows.rmsModH16, .bf16).run },
    .{ "modulate_f16", At(elt.modulate, .f16).run },
    .{ "modulate_bf16", At(elt.modulate, .bf16).run },
    .{ "kron_mix", At(elt.kronMix, .f32).run },
    .{ "kron_mix_f16", At(elt.kronMix, .f16).run },
    .{ "kron_mix_bf16", At(elt.kronMix, .bf16).run },
    .{ "kron_stack", At(elt.kronStack, .f32).run },
    .{ "kron_stack_f16", At(elt.kronStack, .f16).run },
    .{ "kron_stack_bf16", At(elt.kronStack, .bf16).run },
    .{ "dora_rows", At(elt.doraRows, .f32).run },
    .{ "dora_rows_f16", At(elt.doraRows, .f16).run },
    .{ "dora_rows_bf16", At(elt.doraRows, .bf16).run },
    .{ "dora_cols", At(elt.doraCols, .f32).run },
    .{ "dora_cols_f16", At(elt.doraCols, .f16).run },
    .{ "dora_cols_bf16", At(elt.doraCols, .bf16).run },
    .{ "gelu_erf_f16", At(elt.geluErf, .f16).run },
    .{ "gelu_erf_bf16", At(elt.geluErf, .bf16).run },
    .{ "silu_f16", At(elt.silu, .f16).run },
    .{ "silu_bf16", At(elt.silu, .bf16).run },
    .{ "modulate_pr_f16", At(elt.modulatePerRow, .f16).run },
    .{ "modulate_pr_bf16", At(elt.modulatePerRow, .bf16).run },
    .{ "gated_add_pr_f16", At(elt.gatedAddPerRow, .f16).run },
    .{ "gated_add_pr_bf16", At(elt.gatedAddPerRow, .bf16).run },
    .{ "dw_conv3_f16", At(elt.dwConv3, .f16).run },
    .{ "dw_conv3_bf16", At(elt.dwConv3, .bf16).run },
    .{ "col_mean_f16", At(elt.colMean, .f16).run },
    .{ "col_mean_bf16", At(elt.colMean, .bf16).run },
    .{ "mul_cols_sigmoid_f16", At(elt.mulColsSigmoid, .f16).run },
    .{ "mul_cols_sigmoid_bf16", At(elt.mulColsSigmoid, .bf16).run },
    .{ "nerf_feat_f16", At(elt.nerfFeat, .f16).run },
    .{ "nerf_feat_bf16", At(elt.nerfFeat, .bf16).run },
    .{ "ln_mod_f16", At(rows.lnMod, .f16).run },
    .{ "ln_mod_bf16", At(rows.lnMod, .bf16).run },
    .{ "gelu_f16", At(elt.gelu, .f16).run },
    .{ "gelu_bf16", At(elt.gelu, .bf16).run },
    .{ "flag_nonfinite", At(elt.flagNonFinite, .f32).run },
    .{ "flag_nonfinite_f16", At(elt.flagNonFinite, .f16).run },
    .{ "flag_nonfinite_bf16", At(elt.flagNonFinite, .bf16).run },
    .{ "add_scaled_f16", At(elt.addScaled, .f16).run },
    .{ "add_scaled_bf16", At(elt.addScaled, .bf16).run },
    .{ "gated_add_scaled", At(elt.gatedAddScaled, .f32).run },
    .{ "gated_add_scaled_f16", At(elt.gatedAddScaled, .f16).run },
    .{ "gated_add_scaled_bf16", At(elt.gatedAddScaled, .bf16).run },
    .{ "situ", At(elt.situ, .f32).run },
    .{ "situ_f16", At(elt.situ, .f16).run },
    .{ "situ_bf16", At(elt.situ, .bf16).run },
    .{ "dup_up_add", At(elt.dupUpAdd, .f32).run },
    .{ "dup_up_add_f16", At(elt.dupUpAdd, .f16).run },
    .{ "dup_up_add_bf16", At(elt.dupUpAdd, .bf16).run },
    .{ "w6a8_decode", elt.w6a8Decode },
    .{ "w6a8_decode_t", elt.w6a8DecodeT },
    .{ "attn_bounded", attn.attnBounded },
    .{ "mul_cols", At(elt.mulCols, .f32).run },
    .{ "mul_cols_f16", At(elt.mulCols, .f16).run },
    .{ "mul_cols_bf16", At(elt.mulCols, .bf16).run },
    .{ "svdq_decode", elt.svdqDecode },
    .{ "svdq_decode_t", elt.svdqDecodeT },
    .{ "svdq_unpack", At(elt.svdqUnpack, .f32).run },
    .{ "svdq_unpack_f16", At(elt.svdqUnpack, .f16).run },
    .{ "svdq_unpack_bf16", At(elt.svdqUnpack, .bf16).run },
};

comptime {
    @setEvalBranchQuota(100_000);
    if (bodies.len != table.entries.len) @compileError("dual.zig bodies and dual_table.zig entries differ in count");
    for (bodies, table.entries) |b_, t| {
        if (!std.mem.eql(u8, b_[0], t.name)) @compileError("dual.zig body order differs from dual_table.zig at " ++ b_[0]);
        // On PTX an `_sdot` kernel is its `_q8` twin, which LLVM would merge into it.
        if (is_ptx and std.mem.endsWith(u8, b_[0], "_sdot")) continue;
        @export(&Entry(b_[1]).run, .{ .name = b_[0] });
    }
}
