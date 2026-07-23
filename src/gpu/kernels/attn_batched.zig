//! Standalone SPIR-V module for block-diagonal BATCHED attention. It lives in
//! its own module (not `eltwise.zig`) because the Zig 0.16 SPIR-V backend
//! segfaults once `eltwise` — already ~80 entry points — gains one more; a fresh
//! small module compiles fine and also lets this kernel use a 5th storage buffer
//! (the per-item bounds table) that the shared 4-buffer eltwise layout can't.
//!
//! Binding layout (set 0): a=q, b=k, c=v, d=out (each the packed
//! [total, heads*hd] / [total, kv*hd] batch activation), e=bounds (u32[2*total]).
//! Push: u0=total, u1=n_heads, u2=n_kv, u3=hd, f0=scale.

const gpu = @import("std").gpu;

const FBuf = extern struct { data: [1 << 28]f32 };
const UBuf = extern struct { data: [1 << 24]u32 };

const Push = extern struct {
    u0: u32,
    u1: u32,
    u2: u32,
    u3: u32,
    f0: f32,
};

extern var a: FBuf addrspace(.storage_buffer);
extern var b: FBuf addrspace(.storage_buffer);
extern var c: FBuf addrspace(.storage_buffer);
extern var d: FBuf addrspace(.storage_buffer);
extern var e: UBuf addrspace(.storage_buffer);
extern var pc: Push addrspace(.push_constant);

inline fn decorate() void {
    asm volatile (
        \\OpDecorate %ft Block
        \\OpMemberDecorate %ft 0 Offset 0
        \\OpDecorate %ut Block
        \\OpMemberDecorate %ut 0 Offset 0
        \\OpDecorate %pt Block
        \\OpMemberDecorate %pt 0 Offset 0
        \\OpMemberDecorate %pt 1 Offset 4
        \\OpMemberDecorate %pt 2 Offset 8
        \\OpMemberDecorate %pt 3 Offset 12
        \\OpMemberDecorate %pt 4 Offset 16
        \\OpDecorate %ba DescriptorSet 0
        \\OpDecorate %ba Binding 0
        \\OpDecorate %bb DescriptorSet 0
        \\OpDecorate %bb Binding 1
        \\OpDecorate %bc DescriptorSet 0
        \\OpDecorate %bc Binding 2
        \\OpDecorate %bd DescriptorSet 0
        \\OpDecorate %bd Binding 3
        \\OpDecorate %be DescriptorSet 0
        \\OpDecorate %be Binding 4
        :
        : [ft] "t" (FBuf),
          [ut] "t" (UBuf),
          [pt] "t" (Push),
          [ba] "" (&a),
          [bb] "" (&b),
          [bc] "" (&c),
          [bd] "" (&d),
          [be] "" (&e),
    );
}

// Ragged block-diagonal non-causal attention. Query row q attends only keys
// [e[q], e[total+q]); one thread per (query,head). Online softmax, f32.
export fn attn_batched() callconv(.spirv_kernel) void {
    decorate();
    const idx = gpu.global_invocation_id[0];
    const total = pc.u0;
    const n_heads = pc.u1;
    if (idx >= total * n_heads) return;
    const n_kv = pc.u2;
    const hd = pc.u3;
    const scale = pc.f0;
    const q_global = idx / n_heads;
    const head = idx % n_heads;
    const kvh = head / (n_heads / n_kv);
    const start = e.data[q_global];
    const end = e.data[total + q_global];
    const qb = idx * hd;
    const kvdim = n_kv * hd;
    var acc: [256]f32 = undefined;
    var t: u32 = 0;
    while (t < hd) : (t += 1) acc[t] = 0;
    var mx: f32 = -3.4e38;
    var denom: f32 = 0;
    var j: u32 = start;
    while (j < end) : (j += 1) {
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
