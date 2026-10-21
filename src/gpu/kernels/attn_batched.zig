//! Standalone SPIR-V module for block-diagonal BATCHED attention. It lives in
//! its own module (not `eltwise.zig`) because the Zig SPIR-V backend
//! segfaulted (0.16) once `eltwise`, already ~80 entry points, gains one more; a fresh
//! small module compiles fine and also lets this kernel use a 5th storage buffer
//! (the per-item bounds table) that the shared 4-buffer eltwise layout can't.
//!
//! Binding layout (set 0): a=q, b=k, c=v, d=out (each the packed
//! [total, heads*hd] / [total, kv*hd] batch activation), e=bounds (u32[2*total]
//! starting u4 words in, so one buffer can hold several tables).
//! Push: u0=total, u1=n_heads, u2=n_kv, u3=hd, f0=scale, u4=bounds offset.
//! Heads up to `max_hd` wide; the dispatch refuses wider.

const abi = @import("abi.zig");
const gpu = @import("std").spirv;

const FBuf = extern struct { data: [1 << 28]f32 };
const UBuf = extern struct { data: [1 << 24]u32 };

const Push = extern struct {
    u0: u32,
    u1: u32,
    u2: u32,
    u3: u32,
    f0: f32,
    u4: u32,
};

pub const max_hd = 512;

const a = abi.buffer(FBuf, "a", 0);
const b = abi.buffer(FBuf, "b", 1);
const c = abi.buffer(FBuf, "c", 2);
const d = abi.buffer(FBuf, "d", 3);
const e = abi.buffer(UBuf, "e", 4);
const pc = abi.pushConstants(Push, "pc");

// Ragged block-diagonal non-causal attention. Query row q attends only keys
// [e[q], e[total+q]); one thread per (query,head). Online softmax, f32.
export fn attn_batched() callconv(abi.kernel) void {
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
    const start = e.data[pc.u4 + q_global];
    const end = e.data[pc.u4 + total + q_global];
    const qb = idx * hd;
    const kvdim = n_kv * hd;
    var acc: [max_hd]f32 = undefined;
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
