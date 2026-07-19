//! Correctness-first elementwise / attention PTX kernels for the CUDA DiT
//! forward (the f32 fallback path). One thread per element / row / (query,head);
//! no tiling or shared memory — simple and obviously-correct, matching the CPU
//! DiT numerics (exp via ex2.approx + log2e, negligible vs the int8 regime).
//! All kernels share a uniform 12-parameter signature so one launcher fits all:
//!   (p0,p1,p2,p3 : .u64 buffers)  (u0..u5 : .u32)  (f0,f1 : .f32)
//! log2(e) = 0f3FB8AA3B; exp(x) = ex2.approx(x * log2e).

/// rms + modulation, fused, ONE BLOCK (256 threads) per row with a parallel
/// shared-memory reduction (replaces the serial-per-row rms_mod — that launched
/// only `rows` threads, e.g. 264 at 256px = 2 blocks on 82 SMs, each doing a
/// 6144-long serial reduction). Same math/order-independent result. b0=x, b1=out,
/// b2=mod. u0=rows, u1=dim, u2=premul_off, u3=shift_off, f0=eps. grid=(rows,1,1).
const std = @import("std");

pub const rms_mod_par_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rms_mod_par(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<16>;
    \\  .reg .f32 %f<12>;
    \\  .reg .b64 %rd<16>;
    \\  .shared .align 4 .b8 red[1024];       // 256 f32 partials
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;                    // tid 0..255
    \\  ld.param.u32 %r4,[u1];                 // dim
    \\  ld.param.u32 %r5,[u2];                 // premul_off
    \\  ld.param.u32 %r6,[u3];                 // shift_off
    \\  ld.param.f32 %f1,[f0];                 // eps
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mul.lo.s32 %r7,%r1,%r4;                // base = row*dim
    \\  mul.wide.u32 %rd4,%r7,4; add.s64 %rd5,%rd1,%rd4;   // x row ptr
    \\  add.s64 %rd6,%rd2,%rd4;                // out row ptr
    \\  mov.f32 %f2,0f00000000; mov.u32 %r8,%r3;           // partial, i=tid
    \\SS:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra SSD;
    \\  mul.wide.u32 %rd7,%r8,4; add.s64 %rd8,%rd5,%rd7;
    \\  ld.global.f32 %f3,[%rd8]; fma.rn.f32 %f2,%f3,%f3,%f2;
    \\  add.u32 %r8,%r8,256; bra SS;
    \\SSD:
    \\  mov.u32 %r9,red; shl.b32 %r10,%r3,2; add.u32 %r10,%r10,%r9;
    \\  st.shared.f32 [%r10],%f2; bar.sync 0;
    \\  mov.u32 %r11,128;
    \\RED:
    \\  setp.eq.u32 %p2,%r11,0; @%p2 bra REDD;
    \\  setp.ge.u32 %p3,%r3,%r11; @%p3 bra REDS;
    \\  ld.shared.f32 %f4,[%r10]; shl.b32 %r12,%r11,2; add.u32 %r12,%r10,%r12;
    \\  ld.shared.f32 %f5,[%r12]; add.f32 %f4,%f4,%f5; st.shared.f32 [%r10],%f4;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r11,%r11,1; bra RED;
    \\REDD:
    \\  ld.shared.f32 %f6,[%r9];               // sum of squares
    \\  cvt.rn.f32.u32 %f7,%r4; div.rn.f32 %f6,%f6,%f7; add.f32 %f6,%f6,%f1;
    \\  rsqrt.approx.f32 %f8,%f6;              // inv
    \\  mov.u32 %r8,%r3;
    \\AP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra END;
    \\  mul.wide.u32 %rd7,%r8,4; add.s64 %rd8,%rd5,%rd7; ld.global.f32 %f3,[%rd8];
    \\  add.s32 %r13,%r8,%r5; mul.wide.u32 %rd9,%r13,4; add.s64 %rd10,%rd3,%rd9; ld.global.f32 %f9,[%rd10];
    \\  add.s32 %r14,%r8,%r6; mul.wide.u32 %rd11,%r14,4; add.s64 %rd12,%rd3,%rd11; ld.global.f32 %f10,[%rd12];
    \\  mul.f32 %f3,%f3,%f8; fma.rn.f32 %f3,%f3,%f9,%f10;
    \\  add.s64 %rd13,%rd6,%rd7; st.global.f32 [%rd13],%f3;
    \\  add.u32 %r8,%r8,256; bra AP;
    \\END:
    \\  ret;
    \\}
;

/// per-head RMS norm, one thread per row. b0=x, b1=out, b2=weight[dim].
/// out[row*dim+c] = x*inv*weight[c], inv=1/sqrt(mean(x^2)+f0). u0=rows,u1=dim(hd),f0=eps.
pub const qk_rmsnorm_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry qk_rmsnorm(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.f32 %f1,[f0];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mul.lo.s32 %r7,%r4,%r6; mul.wide.u32 %rd4,%r7,4;
    \\  add.s64 %rd5,%rd1,%rd4; add.s64 %rd6,%rd2,%rd4;
    \\  mov.f32 %f2,0f00000000; mov.u32 %r8,0; mov.b64 %rd7,%rd5;
    \\SS:
    \\  setp.ge.u32 %p2,%r8,%r6; @%p2 bra SSD;
    \\  ld.global.f32 %f3,[%rd7]; fma.rn.f32 %f2,%f3,%f3,%f2;
    \\  add.s64 %rd7,%rd7,4; add.u32 %r8,%r8,1; bra SS;
    \\SSD:
    \\  cvt.rn.f32.u32 %f4,%r6; div.rn.f32 %f2,%f2,%f4; add.f32 %f2,%f2,%f1; rsqrt.approx.f32 %f5,%f2;
    \\  mov.u32 %r8,0; mov.b64 %rd8,%rd3;
    \\AP:
    \\  setp.ge.u32 %p2,%r8,%r6; @%p2 bra END;
    \\  ld.global.f32 %f6,[%rd5]; ld.global.f32 %f7,[%rd8];
    \\  mul.f32 %f6,%f6,%f5; mul.f32 %f6,%f6,%f7; st.global.f32 [%rd6],%f6;
    \\  add.s64 %rd5,%rd5,4; add.s64 %rd6,%rd6,4; add.s64 %rd8,%rd8,4; add.u32 %r8,%r8,1; bra AP;
    \\END:
    \\  ret;
    \\}
;

/// interleaved RoPE, in place, one thread per (row,pair). b0=qk, b2=freqs.
/// u0=total(rows*nheads*half), u1=half, u2=sin_off, u3=nheads.
pub const rope_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<16>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];  // half
    \\  ld.param.u32 %r7,[u2];  // sin_off
    \\  ld.param.u32 %r8,[u3];  // nheads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair = idx % half
    \\  mul.lo.s32 %r10,%r6,%r8;              // half*nheads
    \\  div.u32 %r11,%r4,%r10;                // pos = idx/(half*nheads)
    \\  mad.lo.s32 %r12,%r11,%r6,%r9;         // pos*half + pair  (cos index)
    \\  mul.wide.u32 %rd4,%r12,4; add.s64 %rd5,%rd3,%rd4; ld.global.f32 %f1,[%rd5]; // cos
    \\  add.s32 %r13,%r12,%r7;               // + sin_off
    \\  mul.wide.u32 %rd6,%r13,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f2,[%rd7]; // sin
    \\  shl.b32 %r14,%r4,1;                  // at = idx*2
    \\  mul.wide.u32 %rd8,%r14,4; add.s64 %rd9,%rd1,%rd8;
    \\  ld.global.f32 %f3,[%rd9]; ld.global.f32 %f4,[%rd9+4];
    \\  // x0*cos - x1*sin
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd9],%f5;
    \\  // x0*sin + x1*cos
    \\  mul.f32 %f6,%f3,%f2; fma.rn.f32 %f6,%f4,%f1,%f6; st.global.f32 [%rd9+4],%f6;
    \\END:
    \\  ret;
    \\}
;

/// naive attention, one thread per (query,head), online softmax, GQA.
/// b0=q[seq_q][heads][hd], b1=k[seq_kv][kv][hd], b2=v[seq_kv][kv][hd],
/// b3=out[seq_q][heads][hd]. u0=seq_q, u1=heads, u2=kv_heads, u3=hd,
/// u4=causal, u5=seq_kv, f0=scale. acc[hd] in .local.
/// Causal treats the queries as the LAST seq_q positions of the kv sequence
/// (query i attends to keys [0, seq_kv - seq_q + i]) — seq_q == seq_kv is the
/// classic square case, seq_q == 1 with longer seq_kv is KV-cached decode.
pub const attn_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .local .align 4 .b8 accl[2048];       // up to hd=512 f32 (gemma4 global layers)
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<32>;
    \\  .reg .f32 %f<20>;
    \\  .reg .b64 %rd<32>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // idx
    \\  ld.param.u32 %r5,[u0];                // seq
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  mul.lo.s32 %r7,%r5,%r6;               // seq*heads
    \\  setp.ge.u32 %p1,%r4,%r7; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r10,%r4,%r6;                 // q = idx/heads
    \\  rem.u32 %r11,%r4,%r6;                 // h = idx%heads
    \\  div.u32 %r12,%r6,%r8;                 // group = heads/kv
    \\  div.u32 %r13,%r11,%r12;               // kv = h/group
    \\  // qbase = (q*heads + h)*hd  (elements)
    \\  mad.lo.s32 %r14,%r10,%r6,%r11; mul.lo.s32 %r14,%r14,%r9;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;  // Q row ptr
    \\  // init acc[hd]=0, m=-inf, d=0
    \\  mov.u32 %r15,0;
    \\ZINIT:
    \\  setp.ge.u32 %p2,%r15,%r9; @%p2 bra ZD;
    \\  mul.wide.u32 %rd7,%r15,4; mov.u32 %r16, accl; cvt.u64.u32 %rd8,%r16; add.s64 %rd8,%rd8,%rd7;
    \\  mov.f32 %f2,0f00000000; st.local.f32 [%rd8],%f2;
    \\  add.u32 %r15,%r15,1; bra ZINIT;
    \\ZD:
    \\  mov.f32 %f10,0fFF800000;              // m = -inf
    \\  mov.f32 %f11,0f00000000;              // d = 0
    \\  ld.param.u32 %r28,[u5];               // seq_kv
    \\  sub.u32 %r29,%r28,%r5;                // kv_off = seq_kv - seq_q
    \\  add.u32 %r31,%r10,1; add.u32 %r31,%r31,%r29; // causal bound = q+1+kv_off
    \\  ld.param.u32 %r30,[u4]; setp.ne.u32 %p2,%r30,0; selp.b32 %r30,%r31,%r28,%p2; // else bound = seq_kv
    \\  mov.u32 %r17,0;                       // j
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r30; @%p2 bra JD;
    \\  // kbase=(j*kv+kv_head)*hd ; vbase same
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9;
    \\  mul.wide.u32 %rd9,%r18,4; add.s64 %rd10,%rd2,%rd9;  // K row
    \\  add.s64 %rd11,%rd3,%rd9;                            // V row
    \\  // s = scale * dot(Q,K)
    \\  mov.f32 %f3,0f00000000; mov.u32 %r19,0; mov.b64 %rd12,%rd6; mov.b64 %rd13,%rd10;
    \\DOT:
    \\  setp.ge.u32 %p3,%r19,%r9; @%p3 bra DOTD;
    \\  ld.global.f32 %f4,[%rd12]; ld.global.f32 %f5,[%rd13]; fma.rn.f32 %f3,%f4,%f5,%f3;
    \\  add.s64 %rd12,%rd12,4; add.s64 %rd13,%rd13,4; add.u32 %r19,%r19,1; bra DOT;
    \\DOTD:
    \\  mul.f32 %f3,%f3,%f1;                  // s
    \\  // online softmax update
    \\  max.f32 %f12,%f10,%f3;                // m2
    \\  sub.f32 %f6,%f10,%f12; mul.f32 %f6,%f6,0f3FB8AA3B; ex2.approx.f32 %f6,%f6; // corr=exp(m-m2)
    \\  sub.f32 %f7,%f3,%f12; mul.f32 %f7,%f7,0f3FB8AA3B; ex2.approx.f32 %f7,%f7;  // p=exp(s-m2)
    \\  mul.f32 %f11,%f11,%f6; add.f32 %f11,%f11,%f7;      // d = d*corr + p
    \\  mov.f32 %f10,%f12;                    // m = m2
    \\  // acc[c] = acc[c]*corr + p*V[c]
    \\  mov.u32 %r19,0; mov.b64 %rd14,%rd11;
    \\ACC:
    \\  setp.ge.u32 %p3,%r19,%r9; @%p3 bra ACCD;
    \\  mul.wide.u32 %rd15,%r19,4; mov.u32 %r20, accl; cvt.u64.u32 %rd16,%r20; add.s64 %rd16,%rd16,%rd15;
    \\  ld.local.f32 %f8,[%rd16]; ld.global.f32 %f9,[%rd14];
    \\  mul.f32 %f8,%f8,%f6; fma.rn.f32 %f8,%f7,%f9,%f8; st.local.f32 [%rd16],%f8;
    \\  add.s64 %rd14,%rd14,4; add.u32 %r19,%r19,1; bra ACC;
    \\ACCD:
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  // out[obase+c] = acc[c]/d   (obase = qbase)
    \\  rcp.approx.f32 %f13,%f11;
    \\  add.s64 %rd17,%rd4,%rd5;              // out row ptr (same layout as Q)
    \\  mov.u32 %r19,0;
    \\WR:
    \\  setp.ge.u32 %p3,%r19,%r9; @%p3 bra END;
    \\  mul.wide.u32 %rd15,%r19,4; mov.u32 %r20, accl; cvt.u64.u32 %rd16,%r20; add.s64 %rd16,%rd16,%rd15;
    \\  ld.local.f32 %f8,[%rd16]; mul.f32 %f8,%f8,%f13; add.s64 %rd18,%rd17,%rd15; st.global.f32 [%rd18],%f8;
    \\  add.u32 %r19,%r19,1; bra WR;
    \\END:
    \\  ret;
    \\}
;

/// qk_rmsnorm with one 256-thread block per row (shared-memory reduction) —
/// the LLM decode path norms rows=1 x dim=2560, where the one-thread-per-row
/// kernel serializes the whole row on a single lane. Same math/params:
/// out = x * rsqrt(mean(x^2)+eps) * w. b0=x, b1=out, b2=w. u0=rows, u1=dim,
/// f0=eps.
pub const qk_rmsnorm_par_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry qk_rmsnorm_par(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<12>;
    \\  .reg .b64 %rd<16>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // dim
    \\  ld.param.f32 %f1,[f0];                 // eps
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mul.lo.s32 %r7,%r1,%r4; mul.wide.u32 %rd4,%r7,4;
    \\  add.s64 %rd5,%rd1,%rd4;                // x row
    \\  add.s64 %rd6,%rd2,%rd4;                // out row
    \\  mov.f32 %f2,0f00000000; mov.u32 %r8,%r3;
    \\SS:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra SSD;
    \\  mul.wide.u32 %rd7,%r8,4; add.s64 %rd8,%rd5,%rd7;
    \\  ld.global.f32 %f3,[%rd8]; fma.rn.f32 %f2,%f3,%f3,%f2;
    \\  add.u32 %r8,%r8,256; bra SS;
    \\SSD:
    \\  mov.u32 %r9,red; shl.b32 %r10,%r3,2; add.u32 %r10,%r10,%r9;
    \\  st.shared.f32 [%r10],%f2; bar.sync 0;
    \\  mov.u32 %r11,128;
    \\RED:
    \\  setp.eq.u32 %p2,%r11,0; @%p2 bra REDD;
    \\  setp.ge.u32 %p3,%r3,%r11; @%p3 bra REDS;
    \\  ld.shared.f32 %f4,[%r10]; shl.b32 %r12,%r11,2; add.u32 %r12,%r10,%r12;
    \\  ld.shared.f32 %f5,[%r12]; add.f32 %f4,%f4,%f5; st.shared.f32 [%r10],%f4;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r11,%r11,1; bra RED;
    \\REDD:
    \\  ld.shared.f32 %f6,[%r9];
    \\  cvt.rn.f32.u32 %f7,%r4; div.rn.f32 %f6,%f6,%f7; add.f32 %f6,%f6,%f1;
    \\  rsqrt.approx.f32 %f8,%f6;              // inv
    \\  mov.u32 %r8,%r3;
    \\AP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra END;
    \\  mul.wide.u32 %rd7,%r8,4; add.s64 %rd8,%rd5,%rd7; ld.global.f32 %f3,[%rd8];
    \\  add.s64 %rd9,%rd3,%rd7; ld.global.f32 %f9,[%rd9];
    \\  mul.f32 %f3,%f3,%f8; mul.f32 %f3,%f3,%f9;
    \\  add.s64 %rd10,%rd6,%rd7; st.global.f32 [%rd10],%f3;
    \\  add.u32 %r8,%r8,256; bra AP;
    \\END:
    \\  ret;
    \\}
;

/// Classic LayerNorm with weight and bias (qwen3vl ViT ln1/ln2/post_ln):
/// out = (x - mean) / sqrt(var + eps) * w + b, one 256-thread block per row.
/// Two-pass variance (sum -> mean -> sum (x-mean)^2), matching
/// ops.norm.layerNorm's math. b0=x, b1=out, b2=w, b3=b. u0=rows, u1=dim,
/// f0=eps. grid=(rows,1,1).
pub const ln_bias_par_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry ln_bias_par(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<16>;
    \\  .reg .b64 %rd<20>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // dim
    \\  ld.param.f32 %f1,[f0];                 // eps
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  mul.lo.s32 %r7,%r1,%r4; mul.wide.u32 %rd5,%r7,4;
    \\  add.s64 %rd6,%rd1,%rd5;                // x row
    \\  add.s64 %rd7,%rd2,%rd5;                // out row
    \\  cvt.rn.f32.u32 %f8,%r4;                // dim as f32
    \\  mov.u32 %r9,red; shl.b32 %r10,%r3,2; add.u32 %r10,%r10,%r9;
    \\  // pass 1: mean
    \\  mov.f32 %f2,0f00000000; mov.u32 %r8,%r3;
    \\S1:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra S1D;
    \\  mul.wide.u32 %rd8,%r8,4; add.s64 %rd9,%rd6,%rd8;
    \\  ld.global.f32 %f4,[%rd9]; add.f32 %f2,%f2,%f4;
    \\  add.u32 %r8,%r8,256; bra S1;
    \\S1D:
    \\  st.shared.f32 [%r10],%f2; bar.sync 0;
    \\  mov.u32 %r11,128;
    \\R1:
    \\  setp.eq.u32 %p2,%r11,0; @%p2 bra R1D;
    \\  setp.ge.u32 %p3,%r3,%r11; @%p3 bra R1S;
    \\  shl.b32 %r12,%r11,2; add.u32 %r12,%r10,%r12;
    \\  ld.shared.f32 %f4,[%r10]; ld.shared.f32 %f5,[%r12]; add.f32 %f4,%f4,%f5; st.shared.f32 [%r10],%f4;
    \\R1S:
    \\  bar.sync 0; shr.u32 %r11,%r11,1; bra R1;
    \\R1D:
    \\  ld.shared.f32 %f6,[%r9]; div.rn.f32 %f6,%f6,%f8; // mean
    \\  bar.sync 0;
    \\  // pass 2: var = mean((x-mean)^2)
    \\  mov.f32 %f3,0f00000000; mov.u32 %r8,%r3;
    \\S2:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra S2D;
    \\  mul.wide.u32 %rd8,%r8,4; add.s64 %rd9,%rd6,%rd8;
    \\  ld.global.f32 %f4,[%rd9]; sub.f32 %f4,%f4,%f6; fma.rn.f32 %f3,%f4,%f4,%f3;
    \\  add.u32 %r8,%r8,256; bra S2;
    \\S2D:
    \\  st.shared.f32 [%r10],%f3; bar.sync 0;
    \\  mov.u32 %r11,128;
    \\R2:
    \\  setp.eq.u32 %p2,%r11,0; @%p2 bra R2D;
    \\  setp.ge.u32 %p3,%r3,%r11; @%p3 bra R2S;
    \\  shl.b32 %r12,%r11,2; add.u32 %r12,%r10,%r12;
    \\  ld.shared.f32 %f4,[%r10]; ld.shared.f32 %f5,[%r12]; add.f32 %f4,%f4,%f5; st.shared.f32 [%r10],%f4;
    \\R2S:
    \\  bar.sync 0; shr.u32 %r11,%r11,1; bra R2;
    \\R2D:
    \\  ld.shared.f32 %f7,[%r9]; div.rn.f32 %f7,%f7,%f8; add.f32 %f7,%f7,%f1;
    \\  rsqrt.approx.f32 %f10,%f7;             // inv = 1/sqrt(var+eps)
    \\  mov.u32 %r8,%r3;
    \\AP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra END;
    \\  mul.wide.u32 %rd8,%r8,4; add.s64 %rd9,%rd6,%rd8; ld.global.f32 %f4,[%rd9];
    \\  add.s64 %rd10,%rd3,%rd8; ld.global.f32 %f11,[%rd10]; // w
    \\  add.s64 %rd11,%rd4,%rd8; ld.global.f32 %f12,[%rd11]; // b
    \\  sub.f32 %f4,%f4,%f6; mul.f32 %f4,%f4,%f10; fma.rn.f32 %f4,%f4,%f11,%f12;
    \\  add.s64 %rd12,%rd7,%rd8; st.global.f32 [%rd12],%f4;
    \\  add.u32 %r8,%r8,256; bra AP;
    \\END:
    \\  ret;
    \\}
;

/// Fused fp8-e4m3 GEMV for KV-cached decode (m=1): y[row] = scale * dot(W[row], x),
/// W fp8 [rows][cols] dequantized inline via the 256-entry LUT staged in shared.
/// One 256-thread block per row (ctaid = row): thread t strides c = 8t, 8t+2048, ...
/// loading 8 weights as one v2.u32 (coalesced 2 KiB per block iteration) and x as
/// v4.f32, then a shared-memory tree reduction. cols must be a multiple of 8.
/// b0=W, b1=x, b2=y, b3=lut(f32[256] global). u0=rows, u1=cols, f0=scale.
pub const gemv_fp8_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_fp8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<28>;
    \\  .reg .f32 %f<16>;
    \\  .reg .b64 %rd<20>;
    \\  .shared .align 4 .b8 lut_s[1024];
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END; // uniform per block
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  // stage the fp8->f32 LUT: lut_s[tid] = lut[tid]
    \\  shl.b32 %r5,%r3,2;
    \\  mul.wide.u32 %rd5,%r3,4; add.s64 %rd6,%rd4,%rd5; ld.global.f32 %f2,[%rd6];
    \\  mov.u32 %r6,lut_s; add.u32 %r7,%r6,%r5; st.shared.f32 [%r7],%f2;
    \\  bar.sync 0;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shl.b32 %r8,%r3,3;                     // c = tid*8
    \\  mul.wide.u32 %rd7,%r1,%r4; add.s64 %rd8,%rd1,%rd7; // W row base (byte = row*cols)
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  cvt.u64.u32 %rd9,%r8; add.s64 %rd10,%rd8,%rd9; ld.global.v2.u32 {%r9,%r18},[%rd10]; // 8 fp8
    \\  mul.wide.u32 %rd11,%r8,4; add.s64 %rd12,%rd2,%rd11;
    \\  ld.global.v4.f32 {%f4,%f5,%f6,%f7},[%rd12];
    \\  ld.global.v4.f32 {%f12,%f13,%f14,%f15},[%rd12+16];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f4,%f3;
    \\  shr.u32 %r10,%r9,8;  and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f5,%f3;
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f6,%f3;
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f7,%f3;
    \\  and.b32 %r10,%r18,255;               shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f12,%f3;
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f13,%f3;
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f14,%f3;
    \\  shr.u32 %r10,%r18,24;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f8,[%r12]; fma.rn.f32 %f3,%f8,%f15,%f3;
    \\  add.u32 %r8,%r8,2048; bra LOOP;
    \\LD:
    \\  mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd13,%r1,4; add.s64 %rd14,%rd3,%rd13; st.global.f32 [%rd14],%f11;
    \\END:
    \\  ret;
    \\}
;

/// bf16 GEMV (the tied LM head): y[row] = scale * dot(W[row], x), W bf16
/// [rows][cols]. Same block-per-row layout as gemv_fp8; thread t loads one u32
/// (two bf16, elems c and c+1 at c = 2t stride 512) and x as v2.f32; bf16 ->
/// f32 is a 16-bit shift. cols must be a multiple of 2.
/// b0=W, b1=x, b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_bf16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_bf16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<14>;
    \\  .reg .b64 %rd<18>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shl.b32 %r8,%r3,1;                     // c = tid*2
    \\  mul.wide.u32 %rd7,%r1,%r4; shl.b64 %rd7,%rd7,1; add.s64 %rd8,%rd1,%rd7; // W row base (bytes = row*cols*2)
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  mul.wide.u32 %rd9,%r8,2; add.s64 %rd10,%rd8,%rd9; ld.global.u32 %r9,[%rd10]; // 2 bf16
    \\  mul.wide.u32 %rd11,%r8,4; add.s64 %rd12,%rd2,%rd11;
    \\  ld.global.v2.f32 {%f4,%f5},[%rd12];
    \\  shl.b32 %r10,%r9,16; mov.b32 %f6,%r10; fma.rn.f32 %f3,%f6,%f4,%f3;           // elem c
    \\  and.b32 %r10,%r9,0xffff0000; mov.b32 %f6,%r10; fma.rn.f32 %f3,%f6,%f5,%f3;   // elem c+1
    \\  add.u32 %r8,%r8,512; bra LOOP;
    \\LD:
    \\  shl.b32 %r5,%r3,2; mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd13,%r1,4; add.s64 %rd14,%rd3,%rd13; st.global.f32 [%rd14],%f11;
    \\END:
    \\  ret;
    \\}
;

/// f16 GEMV: identical to `gemv_bf16` (same [rows][cols] 2-byte layout, thread
/// t loads one u32 = two f16 at c = 2t stride 512, x as v2.f32) except each
/// 16-bit lane is decoded with a true `cvt.rn.f32.f16` instead of a bf16 shift.
/// Used for the small f16 weights some GGUF quants keep at higher precision
/// (e.g. Unsloth's ssm_alpha/ssm_beta). cols must be a multiple of 2.
/// b0=W, b1=x, b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b16 %rs<4>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<14>;
    \\  .reg .b64 %rd<18>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shl.b32 %r8,%r3,1;                     // c = tid*2
    \\  mul.wide.u32 %rd7,%r1,%r4; shl.b64 %rd7,%rd7,1; add.s64 %rd8,%rd1,%rd7; // W row base (bytes = row*cols*2)
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  mul.wide.u32 %rd9,%r8,2; add.s64 %rd10,%rd8,%rd9; ld.global.u32 %r9,[%rd10]; // 2 f16
    \\  mul.wide.u32 %rd11,%r8,4; add.s64 %rd12,%rd2,%rd11;
    \\  ld.global.v2.f32 {%f4,%f5},[%rd12];
    \\  mov.b32 {%rs1,%rs2},%r9;                                     // split into two f16
    \\  cvt.f32.f16 %f6,%rs1; fma.rn.f32 %f3,%f6,%f4,%f3;           // elem c
    \\  cvt.f32.f16 %f6,%rs2; fma.rn.f32 %f3,%f6,%f5,%f3;           // elem c+1
    \\  add.u32 %r8,%r8,512; bra LOOP;
    \\LD:
    \\  shl.b32 %r5,%r3,2; mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd13,%r1,4; add.s64 %rd14,%rd3,%rd13; st.global.f32 [%rd14],%f11;
    \\END:
    \\  ret;
    \\}
;

/// ggml q8_0 GEMV: y[row] = scale * dot(W[row], x), W q8_0 [rows][cols/32
/// blocks of 34 B: f16 d + 32 i8]. Same block-per-row layout as gemv_bf16;
/// thread t owns elems c = 2t stride 512. Blocks are only 2-byte aligned
/// (34 B stride), so quants load as u16 pairs and d as b16. cols % 32 == 0.
/// b0=W, b1=x, b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_q8_0_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q8_0(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<14>;
    \\  .reg .b64 %rd<20>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r5,%r4,5; mul.lo.u32 %r6,%r5,34;          // row bytes = cols/32*34
    \\  mul.wide.u32 %rd7,%r1,%r6; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shl.b32 %r8,%r3,1;                     // e = tid*2
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  shr.u32 %r9,%r8,5; mul.lo.u32 %r10,%r9,34;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // &block
    \\  ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f6,%h0;    // d
    \\  and.b32 %r11,%r8,31; cvt.u64.u32 %rd11,%r11; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.u16 %r13,[%rd12+2];                      // 2 quants
    \\  mul.wide.u32 %rd13,%r8,4; add.s64 %rd14,%rd2,%rd13;
    \\  ld.global.v2.f32 {%f4,%f5},[%rd14];
    \\  shl.b32 %r14,%r13,24; shr.s32 %r14,%r14,24; cvt.rn.f32.s32 %f7,%r14;  // q0 (i8)
    \\  mul.f32 %f7,%f7,%f6; fma.rn.f32 %f3,%f7,%f4,%f3;
    \\  shl.b32 %r14,%r13,16; shr.s32 %r14,%r14,24; cvt.rn.f32.s32 %f7,%r14;  // q1
    \\  mul.f32 %f7,%f7,%f6; fma.rn.f32 %f3,%f7,%f5,%f3;
    \\  add.u32 %r8,%r8,512; bra LOOP;
    \\LD:
    \\  shl.b32 %r5,%r3,2; mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd15,%r1,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f11;
    \\END:
    \\  ret;
    \\}
;

/// ggml q4_0 GEMV (Gemma 4 QAT decode): y[row] = scale * dot(W[row], x), W q4_0
/// [rows][cols/32 blocks of 18 B: f16 d + 16 nibble bytes]. One block per row,
/// 256 threads, shared reduction (gemv_q8_0 shape). Iterates the 16 qs BYTES
/// per block: byte g holds element (b*32 + jj) in its LOW nibble and element
/// (b*32 + jj + 16) in its HIGH nibble, so each weight byte is read ONCE and
/// contributes two FMAs — v = (nibble - 8) * d. cols % 32 == 0.
pub const gemv_q4_0_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q4_0(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<16>;
    \\  .reg .b64 %rd<20>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r5,%r4,5; mul.lo.u32 %r6,%r5,18;          // row bytes = cols/32*18
    \\  mul.wide.u32 %rd7,%r1,%r6; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shr.u32 %r7,%r4,1;                     // nbytes = cols/2
    \\  mov.u32 %r8,%r3;                       // g = tid
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r7; @%p2 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // block b = g/16
    \\  mul.lo.u32 %r10,%r9,18; cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;  // &block
    \\  ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f6,%h0;    // d
    \\  and.b32 %r11,%r8,15;                   // jj
    \\  cvt.u64.u32 %rd11,%r11; add.s64 %rd12,%rd10,%rd11; ld.global.u8 %r13,[%rd12+2];  // qs byte
    \\  and.b32 %r14,%r13,15; sub.s32 %r14,%r14,8; cvt.rn.f32.s32 %f7,%r14; mul.f32 %f7,%f7,%f6;  // lo*d
    \\  shr.u32 %r15,%r13,4; sub.s32 %r15,%r15,8; cvt.rn.f32.s32 %f8,%r15; mul.f32 %f8,%f8,%f6;   // hi*d
    \\  shl.b32 %r16,%r9,5; add.u32 %r16,%r16,%r11;        // elo = b*32 + jj
    \\  mul.wide.u32 %rd13,%r16,4; add.s64 %rd14,%rd2,%rd13; ld.global.f32 %f4,[%rd14];  // act_lo
    \\  add.u32 %r17,%r16,16; mul.wide.u32 %rd15,%r17,4; add.s64 %rd16,%rd2,%rd15; ld.global.f32 %f5,[%rd16];  // act_hi
    \\  fma.rn.f32 %f3,%f7,%f4,%f3; fma.rn.f32 %f3,%f8,%f5,%f3;
    \\  add.u32 %r8,%r8,256; bra LOOP;
    \\LD:
    \\  shl.b32 %r5,%r3,2; mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd17,%r1,4; add.s64 %rd18,%rd3,%rd17; st.global.f32 [%rd18],%f11;
    \\END:
    \\  ret;
    \\}
;

/// ggml q4_k GEMV: y[row] = scale * dot(W[row], x), W q4_k [rows][cols/256
/// super-blocks of 144 B: f16 d, f16 dmin, 12 B 6-bit sub-block scales/mins,
/// 128 B nibbles]. Phase 1 decodes every sub-block's (d*sc, dmin*m) pair
/// into shared once per row (8 pairs/super-block, get_scale_min_k4 packing);
/// phase 2 is the gemv_bf16-shaped loop with the value nibble decoded as
/// (q >> 4*((j>>5)&1)) & 15 from byte (j>>6)*32 + (j&31), v = dsc*q - dm.
/// cols % 256 == 0 and cols <= 32768 (shared scale table).
/// b0=W, b1=x, b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_q4_k_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q4_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<16>;
    \\  .reg .b64 %rd<24>;
    \\  .shared .align 8 .b8 scales_s[8192];
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r5,%r4,8; mul.lo.u32 %r6,%r5,144;         // nsb, row bytes
    \\  mul.wide.u32 %rd7,%r1,%r6; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r7,%r5,3;                     // nsb*8 sub-blocks
    \\  mov.u32 %r16,%r3;
    \\PRE:
    \\  setp.ge.u32 %p2,%r16,%r7; @%p2 bra PRED;
    \\  shr.u32 %r17,%r16,3; and.b32 %r18,%r16,7;          // sb, j
    \\  mul.lo.u32 %r19,%r17,144; cvt.u64.u32 %rd9,%r19; add.s64 %rd10,%rd8,%rd9;
    \\  ld.global.b16 %h0,[%rd10];   cvt.f32.f16 %f6,%h0;  // d
    \\  ld.global.b16 %h1,[%rd10+2]; cvt.f32.f16 %f7,%h1;  // dmin
    \\  cvt.u64.u32 %rd11,%r18; add.s64 %rd12,%rd10,%rd11; // A: scales[j] at [A+4]
    \\  setp.lt.u32 %p3,%r18,4; @%p3 bra PLO;
    \\  ld.global.u8 %r21,[%rd12+8];                       // s[j+4]
    \\  ld.global.u8 %r22,[%rd12];                         // s[j-4]
    \\  ld.global.u8 %r23,[%rd12+4];                       // s[j]
    \\  and.b32 %r24,%r21,15; shr.u32 %r25,%r22,6; shl.b32 %r25,%r25,4; or.b32 %r24,%r24,%r25; // sc
    \\  shr.u32 %r26,%r21,4; shr.u32 %r27,%r23,6; shl.b32 %r27,%r27,4; or.b32 %r26,%r26,%r27;  // m
    \\  bra PST;
    \\PLO:
    \\  ld.global.u8 %r21,[%rd12+4];                       // s[j]
    \\  ld.global.u8 %r22,[%rd12+8];                       // s[j+4]
    \\  and.b32 %r24,%r21,63; and.b32 %r26,%r22,63;
    \\PST:
    \\  cvt.rn.f32.u32 %f8,%r24; mul.f32 %f8,%f8,%f6;      // d*sc
    \\  cvt.rn.f32.u32 %f9,%r26; mul.f32 %f9,%f9,%f7;      // dmin*m
    \\  shl.b32 %r28,%r16,3; mov.u32 %r29,scales_s; add.u32 %r29,%r29,%r28;
    \\  st.shared.v2.f32 [%r29],{%f8,%f9};
    \\  add.u32 %r16,%r16,256; bra PRE;
    \\PRED:
    \\  bar.sync 0;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shl.b32 %r8,%r3,1;                     // e = tid*2
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r4; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,8; mul.lo.u32 %r10,%r9,144;        // super-block byte base
    \\  and.b32 %r11,%r8,255;                              // j
    \\  shr.u32 %r12,%r11,6; shl.b32 %r12,%r12,5;
    \\  and.b32 %r13,%r11,31; add.u32 %r12,%r12,%r13; add.u32 %r12,%r12,%r10;
    \\  cvt.u64.u32 %rd9,%r12; add.s64 %rd10,%rd8,%rd9;
    \\  ld.global.u16 %r14,[%rd10+16];                     // two nibble bytes (qs at +16)
    \\  shr.u32 %r15,%r11,5; and.b32 %r15,%r15,1; shl.b32 %r15,%r15,2;  // nibble shift
    \\  shr.u32 %r17,%r8,5; shl.b32 %r17,%r17,3; mov.u32 %r18,scales_s; add.u32 %r18,%r18,%r17;
    \\  ld.shared.v2.f32 {%f6,%f7},[%r18];                 // d*sc, dmin*m
    \\  mul.wide.u32 %rd13,%r8,4; add.s64 %rd14,%rd2,%rd13;
    \\  ld.global.v2.f32 {%f4,%f5},[%rd14];
    \\  and.b32 %r19,%r14,255; shr.u32 %r19,%r19,%r15; and.b32 %r19,%r19,15;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7;
    \\  fma.rn.f32 %f3,%f8,%f4,%f3;
    \\  shr.u32 %r19,%r14,8; and.b32 %r19,%r19,255; shr.u32 %r19,%r19,%r15; and.b32 %r19,%r19,15;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7;
    \\  fma.rn.f32 %f3,%f8,%f5,%f3;
    \\  add.u32 %r8,%r8,512; bra LOOP;
    \\LD:
    \\  shl.b32 %r5,%r3,2; mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14],%f3; bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p5,%r15,0; @%p5 bra REDD;
    \\  setp.ge.u32 %p6,%r3,%r15; @%p6 bra REDS;
    \\  ld.shared.f32 %f9,[%r14]; shl.b32 %r16,%r15,2; add.u32 %r16,%r14,%r16;
    \\  ld.shared.f32 %f10,[%r16]; add.f32 %f9,%f9,%f10; st.shared.f32 [%r14],%f9;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p7,%r3,0; @%p7 bra END;
    \\  ld.shared.f32 %f11,[%r13]; mul.f32 %f11,%f11,%f1;
    \\  mul.wide.u32 %rd15,%r1,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f11;
    \\END:
    \\  ret;
    \\}
;

/// ggml q5_k GEMV, warp-per-row (8 rows per 256-thread block): each lane
/// walks quads (one aligned u32 of nibbles = elems j..j+3 low + j+32..j+35
/// high) strided 32, decoding the two 6-bit sub-block scales inline from
/// the L1-resident block header; butterfly-shuffle reduction, no shared
/// memory or barriers. Every qs/qh byte is read exactly once.
/// cols % 256 == 0, rows % 8 == 0. b0=W, b1=x, b2=y. u0=rows, u1=cols,
/// f0=scale.
pub const gemv_q5_k_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q5_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b32 %r<48>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,176;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  mov.f32 %f3,0f00000000;                // acc_lo
    \\  mov.f32 %f30,0f00000000;               // acc_hi
    \\  shr.u32 %r30,%r4,3;                    // nq = cols/8
    \\  mov.u32 %r8,%r6;                       // quad = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,5;                     // sb
    \\  and.b32 %r11,%r8,31;                   // lq
    \\  mul.lo.u32 %r10,%r9,176;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shl.b32 %r12,%r11,2;                   // lq*4
    \\  cvt.u64.u32 %rd11,%r12; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.u32 %r14,[%rd12+48];         // 4 qs bytes
    \\  and.b32 %r13,%r11,7; shl.b32 %r13,%r13,2;          // lo4*4
    \\  cvt.u64.u32 %rd17,%r13; add.s64 %rd18,%rd10,%rd17;
    \\  ld.global.u32 %r20,[%rd18+16];         // 4 qh bytes
    \\  shr.u32 %r21,%r11,3; shl.b32 %r22,%r21,1;          // grp, is0 = 2*grp
    \\  // inline scale decode: d, dmin + packed 6-bit table
    \\  ld.global.b16 %h0,[%rd10];   cvt.f32.f16 %f24,%h0; // d
    \\  ld.global.b16 %h1,[%rd10+2]; cvt.f32.f16 %f25,%h1; // dmin
    \\  ld.global.u32 %r36,[%rd10+4];          // s[0..3]
    \\  ld.global.u32 %r37,[%rd10+8];          // s[4..7]
    \\  ld.global.u32 %r38,[%rd10+12];         // s[8..11]
    \\  shl.b32 %r39,%r22,3;                   // is0*8
    \\  setp.ge.u32 %p5,%r22,4; @%p5 bra SHI;
    \\  // is < 4: sc = s[is]&63, m = s[is+4]&63 (is0 and is0+1)
    \\  shr.u32 %r31,%r36,%r39; and.b32 %r31,%r31,63;      // sc0
    \\  shr.u32 %r32,%r37,%r39; and.b32 %r32,%r32,63;      // m0
    \\  add.u32 %r40,%r39,8;
    \\  shr.u32 %r33,%r36,%r40; and.b32 %r33,%r33,63;      // sc1
    \\  shr.u32 %r34,%r37,%r40; and.b32 %r34,%r34,63;      // m1
    \\  bra SDONE;
    \\SHI:
    \\  // is >= 4: k = is-4; sc = (s8[k]&15)|((s0[k]>>6)<<4), m = (s8[k]>>4)|((s4[k]>>6)<<4)
    \\  add.u32 %r41,%r39,-32;                 // k0*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r31,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r31,%r31,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r32,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r32,%r32,%r43;
    \\  add.u32 %r41,%r41,8;                   // k1*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r33,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r33,%r33,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r34,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r34,%r34,%r43;
    \\SDONE:
    \\  cvt.rn.f32.u32 %f6,%r31; mul.f32 %f6,%f6,%f24;     // d1
    \\  cvt.rn.f32.u32 %f7,%r32; mul.f32 %f7,%f7,%f25;     // m1
    \\  cvt.rn.f32.u32 %f26,%r33; mul.f32 %f26,%f26,%f24;  // d2
    \\  cvt.rn.f32.u32 %f27,%r34; mul.f32 %f27,%f27,%f25;  // m2
    \\  // j_lo = sb*256 + grp*64 + lo4*4
    \\  shr.u32 %r23,%r8,5; shl.b32 %r23,%r23,8;
    \\  shl.b32 %r24,%r21,6; add.u32 %r23,%r23,%r24; add.u32 %r23,%r23,%r13;
    \\  mul.wide.u32 %rd13,%r23,4; add.s64 %rd14,%rd2,%rd13;
    \\  ld.global.v4.f32 {%f4,%f5,%f10,%f11},[%rd14];      // x_lo
    \\  ld.global.v4.f32 {%f12,%f13,%f14,%f15},[%rd14+128]; // x_hi
    \\  shr.u32 %r28,%r20,%r22;                            // qh: bit0/1 per byte
    \\  // byte 0
    \\  and.b32 %r19,%r14,15;
    \\  and.b32 %r24,%r28,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7; fma.rn.f32 %f3,%f8,%f4,%f3;
    \\  shr.u32 %r19,%r14,4; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,1; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f26; sub.f32 %f8,%f8,%f27; fma.rn.f32 %f30,%f8,%f12,%f30;
    \\  // byte 1
    \\  shr.u32 %r19,%r14,8; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,8; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7; fma.rn.f32 %f3,%f8,%f5,%f3;
    \\  shr.u32 %r19,%r14,12; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,9; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f26; sub.f32 %f8,%f8,%f27; fma.rn.f32 %f30,%f8,%f13,%f30;
    \\  // byte 2
    \\  shr.u32 %r19,%r14,16; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,16; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7; fma.rn.f32 %f3,%f8,%f10,%f3;
    \\  shr.u32 %r19,%r14,20; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,17; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f26; sub.f32 %f8,%f8,%f27; fma.rn.f32 %f30,%f8,%f14,%f30;
    \\  // byte 3
    \\  shr.u32 %r19,%r14,24; and.b32 %r19,%r19,15;
    \\  shr.u32 %r24,%r28,24; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f6; sub.f32 %f8,%f8,%f7; fma.rn.f32 %f3,%f8,%f11,%f3;
    \\  shr.u32 %r19,%r14,28;
    \\  shr.u32 %r24,%r28,25; and.b32 %r24,%r24,1; shl.b32 %r24,%r24,4; add.u32 %r19,%r19,%r24;
    \\  cvt.rn.f32.u32 %f8,%r19; mul.f32 %f8,%f8,%f26; sub.f32 %f8,%f8,%f27; fma.rn.f32 %f30,%f8,%f15,%f30;
    \\  add.u32 %r8,%r8,32; bra LOOP;
    \\LD:
    \\  add.f32 %f3,%f3,%f30;
    \\  // butterfly warp reduction
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,16,0x1f,0xffffffff; mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,8,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,4,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,2,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,1,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  setp.ne.u32 %p7,%r6,0; @%p7 bra END;
    \\  mul.f32 %f3,%f3,%f1;
    \\  mul.wide.u32 %rd15,%r7,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f3;
    \\END:
    \\  ret;
    \\}
;

/// ggml q6_k GEMV, warp-per-row (8 rows per 256-thread block): each lane
/// walks 16-elem units (4 consecutive l-bytes of one half: 4 ql+4 ql32+4 qh
/// bytes decode 4 elems in each of the 4 y-groups) strided 32, i8 sub-block
/// scales read inline; butterfly-shuffle reduction. Every ql/qh byte read
/// once. cols % 256 == 0, rows % 8 == 0. b0=W, b1=x, b2=y. u0=rows,
/// u1=cols, f0=scale.
pub const gemv_q6_k_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q6_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<48>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,210;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  mov.f32 %f3,0f00000000;                // acc g0
    \\  mov.f32 %f28,0f00000000;               // acc g1
    \\  mov.f32 %f29,0f00000000;               // acc g2
    \\  mov.f32 %f31,0f00000000;               // acc g3
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // lu
    \\  mul.lo.u32 %r10,%r9,210;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shr.u32 %r12,%r11,3;                   // half
    \\  and.b32 %r13,%r11,7; shl.b32 %r13,%r13,2;          // lb
    \\  shl.b32 %r15,%r12,6; add.u32 %r15,%r15,%r13;       // ql off
    \\  cvt.u64.u32 %rd11,%r15; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.u16 %r16,[%rd12]; ld.global.u16 %r17,[%rd12+2];
    \\  shl.b32 %r17,%r17,16; or.b32 %r16,%r16,%r17;       // w_lo
    \\  ld.global.u16 %r18,[%rd12+32]; ld.global.u16 %r19,[%rd12+34];
    \\  shl.b32 %r19,%r19,16; or.b32 %r18,%r18,%r19;       // w_32
    \\  shl.b32 %r20,%r12,5; add.u32 %r20,%r20,%r13;       // half*32 + lb
    \\  cvt.u64.u32 %rd13,%r20; add.s64 %rd17,%rd10,%rd13;
    \\  ld.global.u16 %r21,[%rd17+128]; ld.global.u16 %r22,[%rd17+130];
    \\  shl.b32 %r22,%r22,16; or.b32 %r21,%r21,%r22;       // w_h
    \\  ld.global.b16 %h0,[%rd10+208]; cvt.f32.f16 %f24,%h0; // d
    \\  // i8 scales at +192 + half*8 + lb/16 (+2 per group)
    \\  shl.b32 %r23,%r12,3; shr.u32 %r25,%r13,4; add.u32 %r23,%r23,%r25;
    \\  cvt.u64.u32 %rd14,%r23; add.s64 %rd15,%rd10,%rd14;
    \\  ld.global.s8 %r31,[%rd15+192]; cvt.rn.f32.s32 %f6,%r31; mul.f32 %f6,%f24,%f6;
    \\  ld.global.s8 %r31,[%rd15+194]; cvt.rn.f32.s32 %f7,%r31; mul.f32 %f7,%f24,%f7;
    \\  ld.global.s8 %r31,[%rd15+196]; cvt.rn.f32.s32 %f26,%r31; mul.f32 %f26,%f24,%f26;
    \\  ld.global.s8 %r31,[%rd15+198]; cvt.rn.f32.s32 %f27,%r31; mul.f32 %f27,%f24,%f27;
    \\  // x fragments at j0 = sb*256 + half*128 + lb
    \\  shl.b32 %r28,%r9,8; shl.b32 %r29,%r12,7; add.u32 %r28,%r28,%r29; add.u32 %r28,%r28,%r13;
    \\  mul.wide.u32 %rd18,%r28,4; add.s64 %rd19,%rd2,%rd18;
    \\  ld.global.v4.f32 {%f4,%f5,%f10,%f11},[%rd19];
    \\  ld.global.v4.f32 {%f12,%f13,%f14,%f15},[%rd19+128];
    \\  ld.global.v4.f32 {%f16,%f17,%f18,%f19},[%rd19+256];
    \\  ld.global.v4.f32 {%f20,%f21,%f22,%f23},[%rd19+384];
    \\  // byte 0
    \\  and.b32 %r31,%r16,15; and.b32 %r32,%r21,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f6,%f8; fma.rn.f32 %f3,%f8,%f4,%f3;
    \\  and.b32 %r31,%r18,15; shr.u32 %r32,%r21,2; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f7,%f8; fma.rn.f32 %f28,%f8,%f12,%f28;
    \\  shr.u32 %r31,%r16,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r21,4; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f26,%f8; fma.rn.f32 %f29,%f8,%f16,%f29;
    \\  shr.u32 %r31,%r18,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r21,6; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f27,%f8; fma.rn.f32 %f31,%f8,%f20,%f31;
    \\  // byte 1
    \\  shr.u32 %r33,%r16,8; shr.u32 %r34,%r18,8; shr.u32 %r35,%r21,8;
    \\  and.b32 %r31,%r33,15; and.b32 %r32,%r35,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f6,%f8; fma.rn.f32 %f3,%f8,%f5,%f3;
    \\  and.b32 %r31,%r34,15; shr.u32 %r32,%r35,2; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f7,%f8; fma.rn.f32 %f28,%f8,%f13,%f28;
    \\  shr.u32 %r31,%r33,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,4; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f26,%f8; fma.rn.f32 %f29,%f8,%f17,%f29;
    \\  shr.u32 %r31,%r34,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,6; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f27,%f8; fma.rn.f32 %f31,%f8,%f21,%f31;
    \\  // byte 2
    \\  shr.u32 %r33,%r16,16; shr.u32 %r34,%r18,16; shr.u32 %r35,%r21,16;
    \\  and.b32 %r31,%r33,15; and.b32 %r32,%r35,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f6,%f8; fma.rn.f32 %f3,%f8,%f10,%f3;
    \\  and.b32 %r31,%r34,15; shr.u32 %r32,%r35,2; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f7,%f8; fma.rn.f32 %f28,%f8,%f14,%f28;
    \\  shr.u32 %r31,%r33,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,4; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f26,%f8; fma.rn.f32 %f29,%f8,%f18,%f29;
    \\  shr.u32 %r31,%r34,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,6; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f27,%f8; fma.rn.f32 %f31,%f8,%f22,%f31;
    \\  // byte 3
    \\  shr.u32 %r33,%r16,24; shr.u32 %r34,%r18,24; shr.u32 %r35,%r21,24;
    \\  and.b32 %r31,%r33,15; and.b32 %r32,%r35,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f6,%f8; fma.rn.f32 %f3,%f8,%f11,%f3;
    \\  and.b32 %r31,%r34,15; shr.u32 %r32,%r35,2; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f7,%f8; fma.rn.f32 %f28,%f8,%f15,%f28;
    \\  shr.u32 %r31,%r33,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,4; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f26,%f8; fma.rn.f32 %f29,%f8,%f19,%f29;
    \\  shr.u32 %r31,%r34,4; and.b32 %r31,%r31,15; shr.u32 %r32,%r35,6; and.b32 %r32,%r32,3; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;
    \\  sub.s32 %r31,%r31,32; cvt.rn.f32.s32 %f8,%r31; mul.f32 %f8,%f27,%f8; fma.rn.f32 %f31,%f8,%f23,%f31;
    \\  add.u32 %r8,%r8,32; bra LOOP;
    \\LD:
    \\  add.f32 %f3,%f3,%f28; add.f32 %f29,%f29,%f31; add.f32 %f3,%f3,%f29;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,16,0x1f,0xffffffff; mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,8,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,4,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,2,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,1,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  setp.ne.u32 %p7,%r6,0; @%p7 bra END;
    \\  mul.f32 %f3,%f3,%f1;
    \\  mul.wide.u32 %rd15,%r7,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f3;
    \\END:
    \\  ret;
    \\}
;

/// Quantize a decode activation vector to q8 blocks for the dp4a GEMV path,
/// SoA so the GEMVs use vector loads: f32 d[nblk] then i8 qs[cols], with
/// d = amax/127 per 32-elem block, q = rni(x*127/amax) (llama.cpp
/// quantize_q8_1 semantics; no stored sum — the GEMVs recover per-quad sums
/// with dp4a against 0x01010101). Warp per block: lane loads one elem,
/// butterfly-max for amax. b0=x f32[cols], b1=xq out. u0=nblk (cols/32).
pub const quantize_q8_1_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry quantize_q8_1(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<12>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%tid.x;
    \\  shr.u32 %r3,%r2,5;                     // warp
    \\  and.b32 %r4,%r2,31;                    // lane
    \\  shl.b32 %r5,%r1,3; add.u32 %r5,%r5,%r3;            // blk
    \\  ld.param.u32 %r6,[u0]; setp.ge.u32 %p1,%r5,%r6; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shl.b32 %r7,%r5,5; add.u32 %r7,%r7,%r4;            // elem
    \\  mul.wide.u32 %rd3,%r7,4; add.s64 %rd4,%rd1,%rd3;
    \\  ld.global.f32 %f1,[%rd4];              // xi
    \\  abs.f32 %f2,%f1;
    \\  mov.b32 %r8,%f2; shfl.sync.bfly.b32 %r9,%r8,16,0x1f,0xffffffff; mov.b32 %f3,%r9; max.f32 %f2,%f2,%f3;
    \\  mov.b32 %r8,%f2; shfl.sync.bfly.b32 %r9,%r8,8,0x1f,0xffffffff;  mov.b32 %f3,%r9; max.f32 %f2,%f2,%f3;
    \\  mov.b32 %r8,%f2; shfl.sync.bfly.b32 %r9,%r8,4,0x1f,0xffffffff;  mov.b32 %f3,%r9; max.f32 %f2,%f2,%f3;
    \\  mov.b32 %r8,%f2; shfl.sync.bfly.b32 %r9,%r8,2,0x1f,0xffffffff;  mov.b32 %f3,%r9; max.f32 %f2,%f2,%f3;
    \\  mov.b32 %r8,%f2; shfl.sync.bfly.b32 %r9,%r8,1,0x1f,0xffffffff;  mov.b32 %f3,%r9; max.f32 %f2,%f2,%f3;
    \\  mov.f32 %f4,0f42FE0000;                // 127.0
    \\  div.rn.f32 %f5,%f4,%f2;                // 127/amax
    \\  setp.eq.f32 %p2,%f2,0f00000000;
    \\  @%p2 mov.f32 %f5,0f00000000;
    \\  mul.f32 %f6,%f1,%f5;
    \\  cvt.rni.s32.f32 %r10,%f6;
    \\  shl.b32 %r11,%r6,2; add.u32 %r12,%r11,%r7;         // nblk*4 + elem
    \\  cvt.u64.u32 %rd5,%r12; add.s64 %rd6,%rd2,%rd5;
    \\  st.global.s8 [%rd6],%r10;
    \\  setp.ne.u32 %p3,%r4,0; @%p3 bra END;
    \\  div.rn.f32 %f7,%f2,%f4;                // d = amax/127
    \\  mul.wide.u32 %rd7,%r5,4; add.s64 %rd8,%rd2,%rd7;
    \\  st.global.f32 [%rd8],%f7;
    \\END:
    \\  ret;
    \\}
;

/// ggml q5_k GEMV against a quantize_q8_1 activation (dp4a int8 dot
/// products, llama.cpp vec_dot_q5_K_q8_1_impl_vmmq math): warp-per-row, each
/// lane owns a 16-elem unit (8 qs bytes = quads j..j+7 low + j+32..j+35+4
/// high) strided 32 units, so the inline 6-bit scale decode amortizes over
/// 16 elems (v3 paid it per 8). Per unit: 8 dp4a (4 value dots + 4 sums of u
/// for the dmin*m term), integer sc/m muls, 4 cvt+fma; six vector LDGs
/// (v2 qs, v2 qh, v4 header, v2 d8, 2x v2 u) — the kernel is load-slot
/// bound, so every 32-bit scalar load matters more than ALU here.
/// cols % 256 == 0, rows % 8 == 0. b0=W, b1=xq (SoA: f32 d[cols/32] then
/// i8 qs[cols]), b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_q5_k_q8_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q5_k_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<32>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,176;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shr.u32 %r47,%r4,3;                    // SoA qs base = xq + cols/8
    \\  cvt.u64.u32 %rd4,%r47; add.s64 %rd4,%rd2,%rd4;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // within
    \\  shr.u32 %r12,%r11,2;                   // grp
    \\  and.b32 %r13,%r11,3; shl.b32 %r13,%r13,3;          // ch8 = chunk*8
    \\  mul.lo.u32 %r10,%r9,176;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shl.b32 %r14,%r12,5; add.u32 %r14,%r14,%r13; add.u32 %r14,%r14,48;
    \\  cvt.u64.u32 %rd11,%r14; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.v2.u32 {%r15,%r16},[%rd12];              // qs0, qs1
    \\  cvt.u64.u32 %rd13,%r13; add.s64 %rd14,%rd10,%rd13;
    \\  ld.global.v2.u32 {%r17,%r18},[%rd14+16];           // qh0, qh1
    \\  ld.global.v4.u32 {%r35,%r36,%r37,%r38},[%rd10];    // d|dmin, s[0..3], s[4..7], s[8..11]
    \\  mov.b32 {%h0,%h1},%r35;
    \\  cvt.f32.f16 %f24,%h0;                  // d
    \\  cvt.f32.f16 %f25,%h1;                  // dmin
    \\  shl.b32 %r22,%r12,1;                   // is0 = grp*2
    \\  shl.b32 %r39,%r22,3;                   // is0*8
    \\  setp.ge.u32 %p5,%r22,4; @%p5 bra SHI;
    \\  // is < 4: sc = s[is]&63, m = s[is+4]&63 (is0 and is0+1)
    \\  shr.u32 %r31,%r36,%r39; and.b32 %r31,%r31,63;      // sc0
    \\  shr.u32 %r32,%r37,%r39; and.b32 %r32,%r32,63;      // m0
    \\  add.u32 %r40,%r39,8;
    \\  shr.u32 %r33,%r36,%r40; and.b32 %r33,%r33,63;      // sc1
    \\  shr.u32 %r34,%r37,%r40; and.b32 %r34,%r34,63;      // m1
    \\  bra SDONE;
    \\SHI:
    \\  // is >= 4: k = is-4; sc = (s8[k]&15)|((s0[k]>>6)<<4), m = (s8[k]>>4)|((s4[k]>>6)<<4)
    \\  add.u32 %r41,%r39,-32;                 // k0*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r31,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r31,%r31,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r32,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r32,%r32,%r43;
    \\  add.u32 %r41,%r41,8;                   // k1*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r33,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r33,%r33,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r34,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r34,%r34,%r43;
    \\SDONE:
    \\  // xq SoA: d8 pair for blocks sb*8+is0, +1; quants at qs base
    \\  // + sb*256 + grp*64 + ch8 (lo) / +32 (hi)
    \\  shl.b32 %r45,%r9,3; add.u32 %r45,%r45,%r22; shl.b32 %r46,%r45,2;
    \\  cvt.u64.u32 %rd15,%r46; add.s64 %rd16,%rd2,%rd15;
    \\  ld.global.v2.f32 {%f26,%f27},[%rd16];  // d8_lo, d8_hi
    \\  shl.b32 %r46,%r9,8; shl.b32 %r20,%r12,6; add.u32 %r46,%r46,%r20; add.u32 %r46,%r46,%r13;
    \\  cvt.u64.u32 %rd17,%r46; add.s64 %rd18,%rd4,%rd17;
    \\  ld.global.v2.u32 {%r19,%r20},[%rd18];  // u_lo0, u_lo1
    \\  ld.global.v2.u32 {%r23,%r24},[%rd18+32]; // u_hi0, u_hi1
    \\  and.b32 %r47,%r15,0x0f0f0f0f;
    \\  shr.u32 %r48,%r17,%r22; shl.b32 %r48,%r48,4; and.b32 %r48,%r48,0x10101010; or.b32 %r47,%r47,%r48; // v_lo0
    \\  and.b32 %r49,%r16,0x0f0f0f0f;
    \\  shr.u32 %r50,%r18,%r22; shl.b32 %r50,%r50,4; and.b32 %r50,%r50,0x10101010; or.b32 %r49,%r49,%r50; // v_lo1
    \\  add.u32 %r21,%r22,1;
    \\  shr.u32 %r51,%r15,4; and.b32 %r51,%r51,0x0f0f0f0f;
    \\  shr.u32 %r52,%r17,%r21; shl.b32 %r52,%r52,4; and.b32 %r52,%r52,0x10101010; or.b32 %r51,%r51,%r52; // v_hi0
    \\  shr.u32 %r53,%r16,4; and.b32 %r53,%r53,0x0f0f0f0f;
    \\  shr.u32 %r54,%r18,%r21; shl.b32 %r54,%r54,4; and.b32 %r54,%r54,0x10101010; or.b32 %r53,%r53,%r54; // v_hi1
    \\  mov.u32 %r55,0; mov.u32 %r57,0x01010101;
    \\  dp4a.u32.s32 %r56,%r49,%r20,%r55;
    \\  dp4a.u32.s32 %r56,%r47,%r19,%r56;      // dot_lo
    \\  dp4a.u32.s32 %r58,%r57,%r20,%r55;
    \\  dp4a.u32.s32 %r58,%r57,%r19,%r58;      // su_lo
    \\  dp4a.u32.s32 %r59,%r53,%r24,%r55;
    \\  dp4a.u32.s32 %r59,%r51,%r23,%r59;      // dot_hi
    \\  dp4a.u32.s32 %r60,%r57,%r24,%r55;
    \\  dp4a.u32.s32 %r60,%r57,%r23,%r60;      // su_hi
    \\  mul.lo.s32 %r56,%r56,%r31;             // dot_lo*sc0
    \\  mul.lo.s32 %r58,%r58,%r32;             // su_lo*m0
    \\  mul.lo.s32 %r59,%r59,%r33;             // dot_hi*sc1
    \\  mul.lo.s32 %r60,%r60,%r34;             // su_hi*m1
    \\  cvt.rn.f32.s32 %f6,%r56; cvt.rn.f32.s32 %f7,%r58;
    \\  cvt.rn.f32.s32 %f8,%r59; cvt.rn.f32.s32 %f9,%r60;
    \\  mul.f32 %f10,%f6,%f26; fma.rn.f32 %f10,%f8,%f27,%f10;   // fd
    \\  mul.f32 %f11,%f7,%f26; fma.rn.f32 %f11,%f9,%f27,%f11;   // fm
    \\  fma.rn.f32 %f3,%f10,%f24,%f3;          // acc += d*fd
    \\  neg.f32 %f12,%f25;
    \\  fma.rn.f32 %f3,%f11,%f12,%f3;          // acc -= dmin*fm
    \\  add.u32 %r8,%r8,32; bra LOOP;
    \\LD:
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,16,0x1f,0xffffffff; mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,8,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,4,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,2,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,1,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  setp.ne.u32 %p7,%r6,0; @%p7 bra END;
    \\  mul.f32 %f3,%f3,%f1;
    \\  mul.wide.u32 %rd15,%r7,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f3;
    \\END:
    \\  ret;
    \\}
;

/// ggml q6_k GEMV against a quantize_q8_1 activation (dp4a, llama.cpp
/// vec_dot_q6_K_q8_1_impl_mmvq math): v3's warp-per-row 16-elem unit walk
/// (4 ql+4 ql32+4 qh bytes decode one quad in each of the 4 y-groups), with
/// the per-element f32 math replaced by 4 value dp4a + 4 sum-of-u dp4a; the
/// -32 offset folds in as dot - 32*sum(u), scales stay integer until one
/// cvt+fma per group; the SoA activation loads as one v4.f32 of d8 + 4 u32.
/// cols % 256 == 0, rows % 8 == 0. b0=W, b1=xq (SoA: f32 d[cols/32] then
/// i8 qs[cols]), b2=y. u0=rows, u1=cols, f0=scale.
pub const gemv_q6_k_q8_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q6_k_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<32>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,210;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shr.u32 %r24,%r4,3;                    // SoA qs base = xq + cols/8
    \\  cvt.u64.u32 %rd4,%r24; add.s64 %rd4,%rd2,%rd4;
    \\  mov.f32 %f3,0f00000000;                // acc
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // lu
    \\  mul.lo.u32 %r10,%r9,210;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shr.u32 %r12,%r11,3;                   // half
    \\  and.b32 %r13,%r11,7; shl.b32 %r13,%r13,2;          // lb
    \\  shl.b32 %r15,%r12,6; add.u32 %r15,%r15,%r13;       // ql off
    \\  cvt.u64.u32 %rd11,%r15; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.u16 %r16,[%rd12]; ld.global.u16 %r17,[%rd12+2];
    \\  shl.b32 %r17,%r17,16; or.b32 %r16,%r16,%r17;       // w_lo
    \\  ld.global.u16 %r18,[%rd12+32]; ld.global.u16 %r19,[%rd12+34];
    \\  shl.b32 %r19,%r19,16; or.b32 %r18,%r18,%r19;       // w_32
    \\  shl.b32 %r20,%r12,5; add.u32 %r20,%r20,%r13;       // half*32 + lb
    \\  cvt.u64.u32 %rd13,%r20; add.s64 %rd17,%rd10,%rd13;
    \\  ld.global.u16 %r21,[%rd17+128]; ld.global.u16 %r22,[%rd17+130];
    \\  shl.b32 %r22,%r22,16; or.b32 %r21,%r21,%r22;       // w_h
    \\  ld.global.b16 %h0,[%rd10+208]; cvt.f32.f16 %f24,%h0; // d
    \\  // i8 scales at +192 + half*8 + lb/16 (+2 per group), kept integer
    \\  shl.b32 %r23,%r12,3; shr.u32 %r25,%r13,4; add.u32 %r23,%r23,%r25;
    \\  cvt.u64.u32 %rd14,%r23; add.s64 %rd15,%rd10,%rd14;
    \\  ld.global.s8 %r26,[%rd15+192];         // sc0
    \\  ld.global.s8 %r27,[%rd15+194];         // sc1
    \\  ld.global.s8 %r28,[%rd15+196];         // sc2
    \\  ld.global.s8 %r29,[%rd15+198];         // sc3
    \\  // xq SoA: d8 quad for blocks sb*8 + half*4 + k; quants at qs base
    \\  // + sb*256 + half*128 + lb (+32 per group)
    \\  shl.b32 %r33,%r9,3; shl.b32 %r34,%r12,2; add.u32 %r33,%r33,%r34;
    \\  shl.b32 %r34,%r33,2;
    \\  cvt.u64.u32 %rd18,%r34; add.s64 %rd19,%rd2,%rd18;
    \\  ld.global.v4.f32 {%f4,%f5,%f10,%f11},[%rd19];      // d8_0..3
    \\  shl.b32 %r34,%r9,8; shl.b32 %r35,%r12,7; add.u32 %r34,%r34,%r35; add.u32 %r34,%r34,%r13;
    \\  cvt.u64.u32 %rd20,%r34; add.s64 %rd21,%rd4,%rd20;
    \\  ld.global.u32 %r36,[%rd21];            // u_0
    \\  ld.global.u32 %r37,[%rd21+32];         // u_1
    \\  ld.global.u32 %r38,[%rd21+64];         // u_2
    \\  ld.global.u32 %r39,[%rd21+96];         // u_3
    \\  and.b32 %r40,%r16,0x0f0f0f0f; and.b32 %r41,%r21,0x03030303; shl.b32 %r41,%r41,4; or.b32 %r40,%r40,%r41;                  // v_0
    \\  and.b32 %r42,%r18,0x0f0f0f0f; shr.u32 %r43,%r21,2; and.b32 %r43,%r43,0x03030303; shl.b32 %r43,%r43,4; or.b32 %r42,%r42,%r43; // v_1
    \\  shr.u32 %r44,%r16,4; and.b32 %r44,%r44,0x0f0f0f0f; shr.u32 %r45,%r21,4; and.b32 %r45,%r45,0x03030303; shl.b32 %r45,%r45,4; or.b32 %r44,%r44,%r45; // v_2
    \\  shr.u32 %r46,%r18,4; and.b32 %r46,%r46,0x0f0f0f0f; shr.u32 %r47,%r21,6; and.b32 %r47,%r47,0x03030303; shl.b32 %r47,%r47,4; or.b32 %r46,%r46,%r47; // v_3
    \\  mov.u32 %r48,0; mov.u32 %r49,0x01010101;
    \\  dp4a.u32.s32 %r50,%r40,%r36,%r48;      // dot_0
    \\  dp4a.u32.s32 %r51,%r49,%r36,%r48;      // su_0
    \\  dp4a.u32.s32 %r52,%r42,%r37,%r48;
    \\  dp4a.u32.s32 %r53,%r49,%r37,%r48;
    \\  dp4a.u32.s32 %r54,%r44,%r38,%r48;
    \\  dp4a.u32.s32 %r55,%r49,%r38,%r48;
    \\  dp4a.u32.s32 %r56,%r46,%r39,%r48;
    \\  dp4a.u32.s32 %r57,%r49,%r39,%r48;
    \\  shl.b32 %r51,%r51,5; sub.s32 %r50,%r50,%r51; mul.lo.s32 %r50,%r50,%r26;
    \\  shl.b32 %r53,%r53,5; sub.s32 %r52,%r52,%r53; mul.lo.s32 %r52,%r52,%r27;
    \\  shl.b32 %r55,%r55,5; sub.s32 %r54,%r54,%r55; mul.lo.s32 %r54,%r54,%r28;
    \\  shl.b32 %r57,%r57,5; sub.s32 %r56,%r56,%r57; mul.lo.s32 %r56,%r56,%r29;
    \\  cvt.rn.f32.s32 %f6,%r50; cvt.rn.f32.s32 %f7,%r52;
    \\  cvt.rn.f32.s32 %f8,%r54; cvt.rn.f32.s32 %f9,%r56;
    \\  mul.f32 %f12,%f6,%f4;
    \\  fma.rn.f32 %f12,%f7,%f5,%f12;
    \\  fma.rn.f32 %f12,%f8,%f10,%f12;
    \\  fma.rn.f32 %f12,%f9,%f11,%f12;
    \\  fma.rn.f32 %f3,%f12,%f24,%f3;          // acc += d * unit sum
    \\  add.u32 %r8,%r8,32; bra LOOP;
    \\LD:
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,16,0x1f,0xffffffff; mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,8,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,4,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,2,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  mov.b32 %r19,%f3; shfl.sync.bfly.b32 %r24,%r19,1,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f3,%f3,%f9;
    \\  setp.ne.u32 %p7,%r6,0; @%p7 bra END;
    \\  mul.f32 %f3,%f3,%f1;
    \\  mul.wide.u32 %rd15,%r7,4; add.s64 %rd16,%rd3,%rd15; st.global.f32 [%rd16],%f3;
    \\END:
    \\  ret;
    \\}
;

/// Grouped dp4a GEMV (small-batch prefill): one pass streams each weight
/// row once and dots it against up to 8 quantized activation rows staged by
/// a single quantize_q8_1 over all n rows (llama.cpp mmvq ncols_y<=8 idea;
/// gemv_fp8n precedent). The W-side walk, scale decode, and v-word assembly
/// are shared per 16-elem unit; the per-input block (v2 d8 + 2x v2 u loads,
/// 8 dp4a, scale muls, acc fma) is comptime-generated 8x with a warp-uniform
/// early-out at ng. xq layout: f32 d[n*cols/32] then i8 qs[n*cols]; input i
/// (global row row_off+i) reads d at (row_off+i)*cols/32 and qs at
/// (row_off+i)*cols. y[i][rows] per group. cols % 256 == 0, rows % 8 == 0.
/// b0=W, b1=xq, b2=y. u0=rows, u1=cols, u2=ng (1..8), u3=row_off,
/// u4=nblk_total (n*cols/32). f0=scale.
pub const gemv_q5_k_q8n_ptx: [:0]const u8 = q5n_head ++ q8nInputs(q5nInput) ++ q8n_step ++ q8n_epi_head ++ q8nInputs(q8nEpilogue) ++ q8n_tail;

const q5n_head =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q5_k_q8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<48>;
    \\  .reg .b64 %rd<28>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r61,[u2];                // ng
    \\  ld.param.u32 %r62,[u3];                // row_off
    \\  ld.param.u32 %r63,[u4];                // nblk_total
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,176;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r10,%r63,2; cvt.u64.u32 %rd4,%r10; add.s64 %rd4,%rd2,%rd4; // qs region base
    \\  shr.u32 %r10,%r4,3; cvt.u64.u32 %rd25,%r10;        // d row step (cols/8 bytes)
    \\  mul.lo.u32 %r11,%r62,%r10; cvt.u64.u32 %rd5,%r11; add.s64 %rd5,%rd2,%rd5; // d base at row_off
    \\  mul.lo.u32 %r11,%r62,%r4; cvt.u64.u32 %rd6,%r11; add.s64 %rd6,%rd4,%rd6;  // qs base at row_off
    \\  cvt.u64.u32 %rd26,%r4;                 // qs row step
    \\  mov.u32 %r55,0; mov.u32 %r57,0x01010101;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\  mov.f32 %f44,0f00000000; mov.f32 %f45,0f00000000; mov.f32 %f46,0f00000000; mov.f32 %f47,0f00000000;
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // within
    \\  shr.u32 %r12,%r11,2;                   // grp
    \\  and.b32 %r13,%r11,3; shl.b32 %r13,%r13,3;          // ch8 = chunk*8
    \\  mul.lo.u32 %r10,%r9,176;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shl.b32 %r14,%r12,5; add.u32 %r14,%r14,%r13; add.u32 %r14,%r14,48;
    \\  cvt.u64.u32 %rd11,%r14; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.v2.u32 {%r15,%r16},[%rd12];              // qs0, qs1
    \\  cvt.u64.u32 %rd13,%r13; add.s64 %rd14,%rd10,%rd13;
    \\  ld.global.v2.u32 {%r17,%r18},[%rd14+16];           // qh0, qh1
    \\  ld.global.v4.u32 {%r35,%r36,%r37,%r38},[%rd10];    // d|dmin, s[0..3], s[4..7], s[8..11]
    \\  mov.b32 {%h0,%h1},%r35;
    \\  cvt.f32.f16 %f24,%h0;                  // d
    \\  cvt.f32.f16 %f25,%h1;                  // dmin
    \\  neg.f32 %f12,%f25;
    \\  shl.b32 %r22,%r12,1;                   // is0 = grp*2
    \\  shl.b32 %r39,%r22,3;                   // is0*8
    \\  setp.ge.u32 %p5,%r22,4; @%p5 bra SHI;
    \\  shr.u32 %r31,%r36,%r39; and.b32 %r31,%r31,63;      // sc0
    \\  shr.u32 %r32,%r37,%r39; and.b32 %r32,%r32,63;      // m0
    \\  add.u32 %r40,%r39,8;
    \\  shr.u32 %r33,%r36,%r40; and.b32 %r33,%r33,63;      // sc1
    \\  shr.u32 %r34,%r37,%r40; and.b32 %r34,%r34,63;      // m1
    \\  bra SDONE;
    \\SHI:
    \\  add.u32 %r41,%r39,-32;                 // k0*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r31,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r31,%r31,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r32,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r32,%r32,%r43;
    \\  add.u32 %r41,%r41,8;                   // k1*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r33,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r33,%r33,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r34,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r34,%r34,%r43;
    \\SDONE:
    \\  // v words (shared across inputs)
    \\  and.b32 %r47,%r15,0x0f0f0f0f;
    \\  shr.u32 %r48,%r17,%r22; shl.b32 %r48,%r48,4; and.b32 %r48,%r48,0x10101010; or.b32 %r47,%r47,%r48; // v_lo0
    \\  and.b32 %r49,%r16,0x0f0f0f0f;
    \\  shr.u32 %r50,%r18,%r22; shl.b32 %r50,%r50,4; and.b32 %r50,%r50,0x10101010; or.b32 %r49,%r49,%r50; // v_lo1
    \\  add.u32 %r21,%r22,1;
    \\  shr.u32 %r51,%r15,4; and.b32 %r51,%r51,0x0f0f0f0f;
    \\  shr.u32 %r52,%r17,%r21; shl.b32 %r52,%r52,4; and.b32 %r52,%r52,0x10101010; or.b32 %r51,%r51,%r52; // v_hi0
    \\  shr.u32 %r53,%r16,4; and.b32 %r53,%r53,0x0f0f0f0f;
    \\  shr.u32 %r54,%r18,%r21; shl.b32 %r54,%r54,4; and.b32 %r54,%r54,0x10101010; or.b32 %r53,%r53,%r54; // v_hi1
    \\  // moving input pointers for this unit (d pair, quants)
    \\  shl.b32 %r45,%r9,3; add.u32 %r45,%r45,%r22; shl.b32 %r46,%r45,2;
    \\  cvt.u64.u32 %rd15,%r46; add.s64 %rd16,%rd5,%rd15;
    \\  shl.b32 %r46,%r9,8; shl.b32 %r20,%r12,6; add.u32 %r46,%r46,%r20; add.u32 %r46,%r46,%r13;
    \\  cvt.u64.u32 %rd17,%r46; add.s64 %rd18,%rd6,%rd17;
    \\
;

/// One grouped-GEMV input block (q5_k): dot the unit's shared v words with
/// input i's quants, scale, accumulate into %f4{0+i}.
fn q5nInput(comptime i: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\  setp.le.u32 %p6,%r61,{d}; @%p6 bra UDONE;
        \\  ld.global.v2.f32 {{%f26,%f27}},[%rd16];
        \\  ld.global.v2.u32 {{%r19,%r20}},[%rd18];
        \\  ld.global.v2.u32 {{%r23,%r24}},[%rd18+32];
        \\  dp4a.u32.s32 %r56,%r49,%r20,%r55;
        \\  dp4a.u32.s32 %r56,%r47,%r19,%r56;
        \\  dp4a.u32.s32 %r58,%r57,%r20,%r55;
        \\  dp4a.u32.s32 %r58,%r57,%r19,%r58;
        \\  dp4a.u32.s32 %r59,%r53,%r24,%r55;
        \\  dp4a.u32.s32 %r59,%r51,%r23,%r59;
        \\  dp4a.u32.s32 %r60,%r57,%r24,%r55;
        \\  dp4a.u32.s32 %r60,%r57,%r23,%r60;
        \\  mul.lo.s32 %r56,%r56,%r31;
        \\  mul.lo.s32 %r58,%r58,%r32;
        \\  mul.lo.s32 %r59,%r59,%r33;
        \\  mul.lo.s32 %r60,%r60,%r34;
        \\  cvt.rn.f32.s32 %f6,%r56; cvt.rn.f32.s32 %f7,%r58;
        \\  cvt.rn.f32.s32 %f8,%r59; cvt.rn.f32.s32 %f9,%r60;
        \\  mul.f32 %f10,%f6,%f26; fma.rn.f32 %f10,%f8,%f27,%f10;
        \\  mul.f32 %f11,%f7,%f26; fma.rn.f32 %f11,%f9,%f27,%f11;
        \\  fma.rn.f32 %f{d},%f10,%f24,%f{d};
        \\  fma.rn.f32 %f{d},%f11,%f12,%f{d};
        \\  add.s64 %rd16,%rd16,%rd25;
        \\  add.s64 %rd18,%rd18,%rd26;
        \\
    , .{ i, 40 + i, 40 + i, 40 + i, 40 + i });
}

/// Concat the 8 per-input blocks of a grouped kernel.
fn q8nInputs(comptime block: anytype) []const u8 {
    comptime {
        var s: []const u8 = "";
        var i: u32 = 0;
        while (i < 8) : (i += 1) s = s ++ block(i);
        return s;
    }
}

const q8n_step =
    \\UDONE:
    \\  add.u32 %r8,%r8,32; bra LOOP;
    \\
;

const q8n_epi_head =
    \\LD:
    \\  mul.wide.u32 %rd15,%r7,4; add.s64 %rd24,%rd3,%rd15; // y + row*4
    \\  shl.b32 %r10,%r2,2; cvt.u64.u32 %rd27,%r10;         // y row step
    \\
;

/// Grouped-GEMV epilogue block: butterfly-reduce acc i across the warp
/// (uniform early-out at ng — all 32 lanes branch together), lane 0 stores
/// y[i*rows + row].
fn q8nEpilogue(comptime i: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\  setp.le.u32 %p6,%r61,{d}; @%p6 bra END;
        \\  mov.b32 %r19,%f{d}; shfl.sync.bfly.b32 %r24,%r19,16,0x1f,0xffffffff; mov.b32 %f9,%r24; add.f32 %f{d},%f{d},%f9;
        \\  mov.b32 %r19,%f{d}; shfl.sync.bfly.b32 %r24,%r19,8,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f{d},%f{d},%f9;
        \\  mov.b32 %r19,%f{d}; shfl.sync.bfly.b32 %r24,%r19,4,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f{d},%f{d},%f9;
        \\  mov.b32 %r19,%f{d}; shfl.sync.bfly.b32 %r24,%r19,2,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f{d},%f{d},%f9;
        \\  mov.b32 %r19,%f{d}; shfl.sync.bfly.b32 %r24,%r19,1,0x1f,0xffffffff;  mov.b32 %f9,%r24; add.f32 %f{d},%f{d},%f9;
        \\  setp.ne.u32 %p7,%r6,0; @%p7 bra SK{d};
        \\  mul.f32 %f{d},%f{d},%f1; st.global.f32 [%rd24],%f{d};
        \\SK{d}:
        \\  add.s64 %rd24,%rd24,%rd27;
        \\
    , .{
        i,
        40 + i, 40 + i, 40 + i,
        40 + i, 40 + i, 40 + i,
        40 + i, 40 + i, 40 + i,
        40 + i, 40 + i, 40 + i,
        40 + i, 40 + i, 40 + i,
        i,
        40 + i, 40 + i, 40 + i,
        i,
    });
}

const q8n_tail =
    \\END:
    \\  ret;
    \\}
;

/// gemv_q5_k_q8n's q6_k twin: shared W-side v assembly and integer scales
/// per 16-elem unit, comptime-generated per-input blocks (v4 d8 + 4 u32
/// loads, 8 dp4a, dot - 32*sum, acc fma). Same xq layout and params.
pub const gemv_q6_k_q8n_ptx: [:0]const u8 = q6n_head ++ q8nInputs(q6nInput) ++ q8n_step ++ q8n_epi_head ++ q8nInputs(q8nEpilogue) ++ q8n_tail;

const q6n_head =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q6_k_q8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<48>;
    \\  .reg .b64 %rd<28>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r61,[u2];                // ng
    \\  ld.param.u32 %r62,[u3];                // row_off
    \\  ld.param.u32 %r63,[u4];                // nblk_total
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,210;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r10,%r63,2; cvt.u64.u32 %rd4,%r10; add.s64 %rd4,%rd2,%rd4; // qs region base
    \\  shr.u32 %r10,%r4,3; cvt.u64.u32 %rd25,%r10;        // d row step (cols/8 bytes)
    \\  mul.lo.u32 %r11,%r62,%r10; cvt.u64.u32 %rd5,%r11; add.s64 %rd5,%rd2,%rd5; // d base at row_off
    \\  mul.lo.u32 %r11,%r62,%r4; cvt.u64.u32 %rd6,%r11; add.s64 %rd6,%rd4,%rd6;  // qs base at row_off
    \\  cvt.u64.u32 %rd26,%r4;                 // qs row step
    \\  mov.u32 %r48,0; mov.u32 %r49,0x01010101;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\  mov.f32 %f44,0f00000000; mov.f32 %f45,0f00000000; mov.f32 %f46,0f00000000; mov.f32 %f47,0f00000000;
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // lu
    \\  mul.lo.u32 %r10,%r9,210;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shr.u32 %r12,%r11,3;                   // half
    \\  and.b32 %r13,%r11,7; shl.b32 %r13,%r13,2;          // lb
    \\  shl.b32 %r15,%r12,6; add.u32 %r15,%r15,%r13;       // ql off
    \\  cvt.u64.u32 %rd11,%r15; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.u16 %r16,[%rd12]; ld.global.u16 %r17,[%rd12+2];
    \\  shl.b32 %r17,%r17,16; or.b32 %r16,%r16,%r17;       // w_lo
    \\  ld.global.u16 %r18,[%rd12+32]; ld.global.u16 %r19,[%rd12+34];
    \\  shl.b32 %r19,%r19,16; or.b32 %r18,%r18,%r19;       // w_32
    \\  shl.b32 %r20,%r12,5; add.u32 %r20,%r20,%r13;       // half*32 + lb
    \\  cvt.u64.u32 %rd13,%r20; add.s64 %rd17,%rd10,%rd13;
    \\  ld.global.u16 %r21,[%rd17+128]; ld.global.u16 %r22,[%rd17+130];
    \\  shl.b32 %r22,%r22,16; or.b32 %r21,%r21,%r22;       // w_h
    \\  ld.global.b16 %h0,[%rd10+208]; cvt.f32.f16 %f24,%h0; // d
    \\  shl.b32 %r23,%r12,3; shr.u32 %r25,%r13,4; add.u32 %r23,%r23,%r25;
    \\  cvt.u64.u32 %rd14,%r23; add.s64 %rd15,%rd10,%rd14;
    \\  ld.global.s8 %r26,[%rd15+192];         // sc0
    \\  ld.global.s8 %r29,[%rd15+194];         // sc1
    \\  ld.global.s8 %r31,[%rd15+196];         // sc2
    \\  ld.global.s8 %r32,[%rd15+198];         // sc3
    \\  // v words (shared across inputs)
    \\  and.b32 %r40,%r16,0x0f0f0f0f; and.b32 %r41,%r21,0x03030303; shl.b32 %r41,%r41,4; or.b32 %r40,%r40,%r41;                  // v_0
    \\  and.b32 %r42,%r18,0x0f0f0f0f; shr.u32 %r43,%r21,2; and.b32 %r43,%r43,0x03030303; shl.b32 %r43,%r43,4; or.b32 %r42,%r42,%r43; // v_1
    \\  shr.u32 %r44,%r16,4; and.b32 %r44,%r44,0x0f0f0f0f; shr.u32 %r45,%r21,4; and.b32 %r45,%r45,0x03030303; shl.b32 %r45,%r45,4; or.b32 %r44,%r44,%r45; // v_2
    \\  shr.u32 %r46,%r18,4; and.b32 %r46,%r46,0x0f0f0f0f; shr.u32 %r47,%r21,6; and.b32 %r47,%r47,0x03030303; shl.b32 %r47,%r47,4; or.b32 %r46,%r46,%r47; // v_3
    \\  // moving input pointers for this unit (d quad, quants)
    \\  shl.b32 %r33,%r9,3; shl.b32 %r34,%r12,2; add.u32 %r33,%r33,%r34;
    \\  shl.b32 %r34,%r33,2;
    \\  cvt.u64.u32 %rd18,%r34; add.s64 %rd22,%rd5,%rd18;
    \\  shl.b32 %r34,%r9,8; shl.b32 %r35,%r12,7; add.u32 %r34,%r34,%r35; add.u32 %r34,%r34,%r13;
    \\  cvt.u64.u32 %rd20,%r34; add.s64 %rd23,%rd6,%rd20;
    \\
;

/// One grouped-GEMV input block (q6_k): shared v words vs input i's quants;
/// the -32 offset folds in as dot - 32*sum(u), scales stay integer.
fn q6nInput(comptime i: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\  setp.le.u32 %p6,%r61,{d}; @%p6 bra UDONE;
        \\  ld.global.v4.f32 {{%f4,%f5,%f10,%f11}},[%rd22];
        \\  ld.global.u32 %r36,[%rd23];
        \\  ld.global.u32 %r37,[%rd23+32];
        \\  ld.global.u32 %r38,[%rd23+64];
        \\  ld.global.u32 %r39,[%rd23+96];
        \\  dp4a.u32.s32 %r50,%r40,%r36,%r48;
        \\  dp4a.u32.s32 %r51,%r49,%r36,%r48;
        \\  dp4a.u32.s32 %r52,%r42,%r37,%r48;
        \\  dp4a.u32.s32 %r53,%r49,%r37,%r48;
        \\  dp4a.u32.s32 %r54,%r44,%r38,%r48;
        \\  dp4a.u32.s32 %r55,%r49,%r38,%r48;
        \\  dp4a.u32.s32 %r56,%r46,%r39,%r48;
        \\  dp4a.u32.s32 %r57,%r49,%r39,%r48;
        \\  shl.b32 %r51,%r51,5; sub.s32 %r50,%r50,%r51; mul.lo.s32 %r50,%r50,%r26;
        \\  shl.b32 %r53,%r53,5; sub.s32 %r52,%r52,%r53; mul.lo.s32 %r52,%r52,%r29;
        \\  shl.b32 %r55,%r55,5; sub.s32 %r54,%r54,%r55; mul.lo.s32 %r54,%r54,%r31;
        \\  shl.b32 %r57,%r57,5; sub.s32 %r56,%r56,%r57; mul.lo.s32 %r56,%r56,%r32;
        \\  cvt.rn.f32.s32 %f6,%r50; cvt.rn.f32.s32 %f7,%r52;
        \\  cvt.rn.f32.s32 %f8,%r54; cvt.rn.f32.s32 %f9,%r56;
        \\  mul.f32 %f12,%f6,%f4;
        \\  fma.rn.f32 %f12,%f7,%f5,%f12;
        \\  fma.rn.f32 %f12,%f8,%f10,%f12;
        \\  fma.rn.f32 %f12,%f9,%f11,%f12;
        \\  fma.rn.f32 %f{d},%f12,%f24,%f{d};
        \\  add.s64 %rd22,%rd22,%rd25;
        \\  add.s64 %rd23,%rd23,%rd26;
        \\
    , .{ i, 40 + i, 40 + i });
}

/// gemv_q5_k_q8n's q4_k sibling: identical scale decode, dp4a dot (dot*sc -
/// su*m min term), and epilogue — the only difference is the weight value is
/// just the 4-bit nibble (q4_k has no 5th-bit qh plane), so the block is 144 B
/// and qs sits at offset 16. Reuses q5nInput unchanged: the per-input block
/// only touches the shared v words, which q4n_head builds without the qh OR.
pub const gemv_q4_k_q8n_ptx: [:0]const u8 = q4n_head ++ q8nInputs(q5nInput) ++ q8n_step ++ q8n_epi_head ++ q8nInputs(q8nEpilogue) ++ q8n_tail;

const q4n_head =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q4_k_q8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<48>;
    \\  .reg .b64 %rd<28>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r61,[u2];                // ng
    \\  ld.param.u32 %r62,[u3];                // row_off
    \\  ld.param.u32 %r63,[u4];                // nblk_total
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,8; mul.lo.u32 %r10,%r9,144;        // row bytes
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r10,%r63,2; cvt.u64.u32 %rd4,%r10; add.s64 %rd4,%rd2,%rd4; // qs region base
    \\  shr.u32 %r10,%r4,3; cvt.u64.u32 %rd25,%r10;        // d row step (cols/8 bytes)
    \\  mul.lo.u32 %r11,%r62,%r10; cvt.u64.u32 %rd5,%r11; add.s64 %rd5,%rd2,%rd5; // d base at row_off
    \\  mul.lo.u32 %r11,%r62,%r4; cvt.u64.u32 %rd6,%r11; add.s64 %rd6,%rd4,%rd6;  // qs base at row_off
    \\  cvt.u64.u32 %rd26,%r4;                 // qs row step
    \\  mov.u32 %r55,0; mov.u32 %r57,0x01010101;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\  mov.f32 %f44,0f00000000; mov.f32 %f45,0f00000000; mov.f32 %f46,0f00000000; mov.f32 %f47,0f00000000;
    \\  shr.u32 %r30,%r4,4;                    // nu = cols/16
    \\  mov.u32 %r8,%r6;                       // unit = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  shr.u32 %r9,%r8,4;                     // sb
    \\  and.b32 %r11,%r8,15;                   // within
    \\  shr.u32 %r12,%r11,2;                   // grp
    \\  and.b32 %r13,%r11,3; shl.b32 %r13,%r13,3;          // ch8 = chunk*8
    \\  mul.lo.u32 %r10,%r9,144;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // super-block base
    \\  shl.b32 %r14,%r12,5; add.u32 %r14,%r14,%r13; add.u32 %r14,%r14,16;  // qs off (q4_k: 16, no qh)
    \\  cvt.u64.u32 %rd11,%r14; add.s64 %rd12,%rd10,%rd11;
    \\  ld.global.v2.u32 {%r15,%r16},[%rd12];              // qs0, qs1
    \\  ld.global.v4.u32 {%r35,%r36,%r37,%r38},[%rd10];    // d|dmin, s[0..3], s[4..7], s[8..11]
    \\  mov.b32 {%h0,%h1},%r35;
    \\  cvt.f32.f16 %f24,%h0;                  // d
    \\  cvt.f32.f16 %f25,%h1;                  // dmin
    \\  neg.f32 %f12,%f25;
    \\  shl.b32 %r22,%r12,1;                   // is0 = grp*2
    \\  shl.b32 %r39,%r22,3;                   // is0*8
    \\  setp.ge.u32 %p5,%r22,4; @%p5 bra SHI;
    \\  shr.u32 %r31,%r36,%r39; and.b32 %r31,%r31,63;      // sc0
    \\  shr.u32 %r32,%r37,%r39; and.b32 %r32,%r32,63;      // m0
    \\  add.u32 %r40,%r39,8;
    \\  shr.u32 %r33,%r36,%r40; and.b32 %r33,%r33,63;      // sc1
    \\  shr.u32 %r34,%r37,%r40; and.b32 %r34,%r34,63;      // m1
    \\  bra SDONE;
    \\SHI:
    \\  add.u32 %r41,%r39,-32;                 // k0*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r31,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r31,%r31,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r32,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r32,%r32,%r43;
    \\  add.u32 %r41,%r41,8;                   // k1*8
    \\  shr.u32 %r42,%r38,%r41; and.b32 %r33,%r42,15;
    \\  shr.u32 %r43,%r36,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r33,%r33,%r43;
    \\  shr.u32 %r44,%r42,4; and.b32 %r34,%r44,15;
    \\  shr.u32 %r43,%r37,%r41; shr.u32 %r43,%r43,6; and.b32 %r43,%r43,3; shl.b32 %r43,%r43,4; or.b32 %r34,%r34,%r43;
    \\SDONE:
    \\  // v words (shared across inputs); q4_k: low/high nibble, no 5th-bit plane
    \\  and.b32 %r47,%r15,0x0f0f0f0f;                      // v_lo0
    \\  and.b32 %r49,%r16,0x0f0f0f0f;                      // v_lo1
    \\  shr.u32 %r51,%r15,4; and.b32 %r51,%r51,0x0f0f0f0f; // v_hi0
    \\  shr.u32 %r53,%r16,4; and.b32 %r53,%r53,0x0f0f0f0f; // v_hi1
    \\  // moving input pointers for this unit (d pair, quants)
    \\  shl.b32 %r45,%r9,3; add.u32 %r45,%r45,%r22; shl.b32 %r46,%r45,2;
    \\  cvt.u64.u32 %rd15,%r46; add.s64 %rd16,%rd5,%rd15;
    \\  shl.b32 %r46,%r9,8; shl.b32 %r20,%r12,6; add.u32 %r46,%r46,%r20; add.u32 %r46,%r46,%r13;
    \\  cvt.u64.u32 %rd17,%r46; add.s64 %rd18,%rd6,%rd17;
    \\
;

/// Grouped dp4a GEMV for q8_0 (llama.cpp vec_dot_q8_0_q8_1 math): q8_0 has no
/// sub-block scales or mins, so the dot is just d_w * d_x * Σ(qw·qx) per 32-elem
/// block. Weights are SIGNED int8 -> dp4a.s32.s32 (the k-quant nibbles are
/// unsigned -> .u32.s32). The 34-B block stride leaves the quant bytes only
/// 2-B aligned, so each weight u32 is assembled from a u16 pair (q6_k does the
/// same); the q8_1 activation region stays 16-B aligned and loads as v4.u32.
/// Shares q8n_step/epi with the k-quant grouped kernels. Same xq layout/params.
pub const gemv_q8_0_q8n_ptx: [:0]const u8 = q8_0n_head ++ q8nInputs(q8_0nInput) ++ q8n_step ++ q8n_epi_head ++ q8nInputs(q8nEpilogue) ++ q8n_tail;

const q8_0n_head =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q8_0_q8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<48>;
    \\  .reg .b64 %rd<28>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r61,[u2];                // ng
    \\  ld.param.u32 %r62,[u3];                // row_off
    \\  ld.param.u32 %r63,[u4];                // nblk_total
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,5; mul.lo.u32 %r10,%r9,34;        // row bytes = nblk*34
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r10,%r63,2; cvt.u64.u32 %rd4,%r10; add.s64 %rd4,%rd2,%rd4; // qs region base
    \\  shr.u32 %r10,%r4,3; cvt.u64.u32 %rd25,%r10;        // d row step (cols/8 = nblk*4 bytes)
    \\  mul.lo.u32 %r11,%r62,%r10; cvt.u64.u32 %rd5,%r11; add.s64 %rd5,%rd2,%rd5; // d base at row_off
    \\  mul.lo.u32 %r11,%r62,%r4; cvt.u64.u32 %rd6,%r11; add.s64 %rd6,%rd4,%rd6;  // qs base at row_off
    \\  cvt.u64.u32 %rd26,%r4;                 // qs row step (cols bytes)
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\  mov.f32 %f44,0f00000000; mov.f32 %f45,0f00000000; mov.f32 %f46,0f00000000; mov.f32 %f47,0f00000000;
    \\  shr.u32 %r30,%r4,5;                    // nblk = cols/32
    \\  mov.u32 %r8,%r6;                       // block = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  mul.lo.u32 %r10,%r8,34;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // weight block base
    \\  ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f24,%h0;   // weight d
    \\  ld.global.u16 %r40,[%rd10+2];  ld.global.u16 %r12,[%rd10+4];  shl.b32 %r12,%r12,16; or.b32 %r40,%r40,%r12;
    \\  ld.global.u16 %r41,[%rd10+6];  ld.global.u16 %r12,[%rd10+8];  shl.b32 %r12,%r12,16; or.b32 %r41,%r41,%r12;
    \\  ld.global.u16 %r42,[%rd10+10]; ld.global.u16 %r12,[%rd10+12]; shl.b32 %r12,%r12,16; or.b32 %r42,%r42,%r12;
    \\  ld.global.u16 %r43,[%rd10+14]; ld.global.u16 %r12,[%rd10+16]; shl.b32 %r12,%r12,16; or.b32 %r43,%r43,%r12;
    \\  ld.global.u16 %r44,[%rd10+18]; ld.global.u16 %r12,[%rd10+20]; shl.b32 %r12,%r12,16; or.b32 %r44,%r44,%r12;
    \\  ld.global.u16 %r45,[%rd10+22]; ld.global.u16 %r12,[%rd10+24]; shl.b32 %r12,%r12,16; or.b32 %r45,%r45,%r12;
    \\  ld.global.u16 %r46,[%rd10+26]; ld.global.u16 %r12,[%rd10+28]; shl.b32 %r12,%r12,16; or.b32 %r46,%r46,%r12;
    \\  ld.global.u16 %r47,[%rd10+30]; ld.global.u16 %r12,[%rd10+32]; shl.b32 %r12,%r12,16; or.b32 %r47,%r47,%r12;
    \\  // moving input pointers for this block: act d at rd5 + block*4, qs at rd6 + block*32
    \\  shl.b32 %r13,%r8,2; cvt.u64.u32 %rd15,%r13; add.s64 %rd16,%rd5,%rd15;
    \\  shl.b32 %r13,%r8,5; cvt.u64.u32 %rd17,%r13; add.s64 %rd18,%rd6,%rd17;
    \\
;

/// One grouped-GEMV input block (q8_0): dot the block's 32 signed weight int8
/// with input i's 32 activation int8 (8 dp4a.s32.s32), scale by d_w*d_x, acc.
fn q8_0nInput(comptime i: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\  setp.le.u32 %p6,%r61,{d}; @%p6 bra UDONE;
        \\  ld.global.f32 %f26,[%rd16];
        \\  ld.global.v4.u32 {{%r48,%r49,%r50,%r51}},[%rd18];
        \\  ld.global.v4.u32 {{%r52,%r53,%r54,%r55}},[%rd18+16];
        \\  mov.u32 %r56,0;
        \\  dp4a.s32.s32 %r56,%r40,%r48,%r56;
        \\  dp4a.s32.s32 %r56,%r41,%r49,%r56;
        \\  dp4a.s32.s32 %r56,%r42,%r50,%r56;
        \\  dp4a.s32.s32 %r56,%r43,%r51,%r56;
        \\  dp4a.s32.s32 %r56,%r44,%r52,%r56;
        \\  dp4a.s32.s32 %r56,%r45,%r53,%r56;
        \\  dp4a.s32.s32 %r56,%r46,%r54,%r56;
        \\  dp4a.s32.s32 %r56,%r47,%r55,%r56;
        \\  cvt.rn.f32.s32 %f6,%r56;
        \\  mul.f32 %f10,%f24,%f26;
        \\  fma.rn.f32 %f{d},%f6,%f10,%f{d};
        \\  add.s64 %rd16,%rd16,%rd25;
        \\  add.s64 %rd18,%rd18,%rd26;
        \\
    , .{ i, 40 + i, 40 + i });
}

/// dp4a q4_0 GEMV (Gemma 4 decode), grouped-N q8_1 activation. Same warp-per-8-
/// rows / lane-per-block structure as gemv_q8_0_q8n; the only differences are
/// the 18-byte q4_0 block (f16 d + 16 nibble bytes) and the -8 weight offset,
/// applied as dot(w-8,a) = dp4a(nibble, a) - 8*sum(a) (sum via dp4a with
/// 0x01010101 in r58). Nibbles decode to 0..15 (positive int8, so dp4a.s32 is
/// exact). Reuses q8nInputs(q4_0nInput) ++ q8n_step ++ epilogue ++ tail.
pub const gemv_q4_0_q8n_ptx: [:0]const u8 = q4_0n_head ++ q8nInputs(q4_0nInput) ++ q8n_step ++ q8n_epi_head ++ q8nInputs(q8nEpilogue) ++ q8n_tail;

const q4_0n_head =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_q4_0_q8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b32 %r<64>;
    \\  .reg .f32 %f<48>;
    \\  .reg .b64 %rd<28>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r3,%tid.x;
    \\  shr.u32 %r5,%r3,5;                     // warp
    \\  and.b32 %r6,%r3,31;                    // lane
    \\  shl.b32 %r7,%r1,3; add.u32 %r7,%r7,%r5;            // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r7,%r2; @%p1 bra END;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r61,[u2];                // ng
    \\  ld.param.u32 %r62,[u3];                // row_off
    \\  ld.param.u32 %r63,[u4];                // nblk_total
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shr.u32 %r9,%r4,5; mul.lo.u32 %r10,%r9,18;        // row bytes = nblk*18
    \\  mul.wide.u32 %rd7,%r7,%r10; add.s64 %rd8,%rd1,%rd7; // W row base
    \\  shl.b32 %r10,%r63,2; cvt.u64.u32 %rd4,%r10; add.s64 %rd4,%rd2,%rd4; // act qs region base
    \\  shr.u32 %r10,%r4,3; cvt.u64.u32 %rd25,%r10;        // act d row step (cols/8 bytes)
    \\  mul.lo.u32 %r11,%r62,%r10; cvt.u64.u32 %rd5,%r11; add.s64 %rd5,%rd2,%rd5; // act d base at row_off
    \\  mul.lo.u32 %r11,%r62,%r4; cvt.u64.u32 %rd6,%r11; add.s64 %rd6,%rd4,%rd6;  // act qs base at row_off
    \\  cvt.u64.u32 %rd26,%r4;                 // act qs row step (cols bytes)
    \\  mov.u32 %r58,0x01010101;               // dp4a sum multiplicand
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\  mov.f32 %f44,0f00000000; mov.f32 %f45,0f00000000; mov.f32 %f46,0f00000000; mov.f32 %f47,0f00000000;
    \\  shr.u32 %r30,%r4,5;                    // nblk = cols/32
    \\  mov.u32 %r8,%r6;                       // block = lane
    \\LOOP:
    \\  setp.ge.u32 %p4,%r8,%r30; @%p4 bra LD;
    \\  mul.lo.u32 %r10,%r8,18;
    \\  cvt.u64.u32 %rd9,%r10; add.s64 %rd10,%rd8,%rd9;    // weight block base
    \\  ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f24,%h0;   // weight d
    \\  ld.global.u16 %r40,[%rd10+2];  ld.global.u16 %r12,[%rd10+4];  shl.b32 %r12,%r12,16; or.b32 %r40,%r40,%r12; // W0=qs[0..3]
    \\  ld.global.u16 %r41,[%rd10+6];  ld.global.u16 %r12,[%rd10+8];  shl.b32 %r12,%r12,16; or.b32 %r41,%r41,%r12; // W1=qs[4..7]
    \\  ld.global.u16 %r42,[%rd10+10]; ld.global.u16 %r12,[%rd10+12]; shl.b32 %r12,%r12,16; or.b32 %r42,%r42,%r12; // W2=qs[8..11]
    \\  ld.global.u16 %r43,[%rd10+14]; ld.global.u16 %r12,[%rd10+16]; shl.b32 %r12,%r12,16; or.b32 %r43,%r43,%r12; // W3=qs[12..15]
    \\  // unpack: high nibbles (elements 16..31) -> r44..r47, low nibbles (0..15) -> r40..r43
    \\  shr.u32 %r44,%r40,4; and.b32 %r44,%r44,0x0f0f0f0f; and.b32 %r40,%r40,0x0f0f0f0f;
    \\  shr.u32 %r45,%r41,4; and.b32 %r45,%r45,0x0f0f0f0f; and.b32 %r41,%r41,0x0f0f0f0f;
    \\  shr.u32 %r46,%r42,4; and.b32 %r46,%r46,0x0f0f0f0f; and.b32 %r42,%r42,0x0f0f0f0f;
    \\  shr.u32 %r47,%r43,4; and.b32 %r47,%r47,0x0f0f0f0f; and.b32 %r43,%r43,0x0f0f0f0f;
    \\  shl.b32 %r13,%r8,2; cvt.u64.u32 %rd15,%r13; add.s64 %rd16,%rd5,%rd15;   // act d ptr
    \\  shl.b32 %r13,%r8,5; cvt.u64.u32 %rd17,%r13; add.s64 %rd18,%rd6,%rd17;   // act qs ptr
    \\
;

/// dp4a q4_0 grouped-GEMV input block: dp4a the 32 nibble weights (unsigned
/// 0..15) with input i's 32 activation int8, subtract 8*sum(act), scale by
/// d_w*d_a, acc. sum(act) via dp4a with 0x01010101 (r58).
fn q4_0nInput(comptime i: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\  setp.le.u32 %p6,%r61,{d}; @%p6 bra UDONE;
        \\  ld.global.f32 %f26,[%rd16];
        \\  ld.global.v4.u32 {{%r48,%r49,%r50,%r51}},[%rd18];
        \\  ld.global.v4.u32 {{%r52,%r53,%r54,%r55}},[%rd18+16];
        \\  mov.u32 %r56,0;
        \\  dp4a.s32.s32 %r56,%r40,%r48,%r56;
        \\  dp4a.s32.s32 %r56,%r41,%r49,%r56;
        \\  dp4a.s32.s32 %r56,%r42,%r50,%r56;
        \\  dp4a.s32.s32 %r56,%r43,%r51,%r56;
        \\  dp4a.s32.s32 %r56,%r44,%r52,%r56;
        \\  dp4a.s32.s32 %r56,%r45,%r53,%r56;
        \\  dp4a.s32.s32 %r56,%r46,%r54,%r56;
        \\  dp4a.s32.s32 %r56,%r47,%r55,%r56;
        \\  mov.u32 %r57,0;
        \\  dp4a.s32.s32 %r57,%r58,%r48,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r49,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r50,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r51,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r52,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r53,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r54,%r57;
        \\  dp4a.s32.s32 %r57,%r58,%r55,%r57;
        \\  shl.b32 %r57,%r57,3; sub.s32 %r56,%r56,%r57;   // isum - 8*sum(act)
        \\  cvt.rn.f32.s32 %f6,%r56;
        \\  mul.f32 %f10,%f24,%f26;
        \\  fma.rn.f32 %f{d},%f6,%f10,%f{d};
        \\  add.s64 %rd16,%rd16,%rd25;
        \\  add.s64 %rd18,%rd18,%rd26;
        \\
    , .{ i, 40 + i, 40 + i });
}

/// KV storage format of the flash-decode attention kernels (and the KV store/
/// append ops): mirrors llm.kv_cache.KvDtype without importing the LLM layer.
/// q8_0 is the ggml 34-byte block (f16 scale + 32 x i8).
pub const KvFmt = enum { f32, f16, q8_0 };

/// Comptime generator for the Gemma flash-decode "attn_split" kernels. The
/// hand-written PTX blobs this replaced (h256/h512 x KV format) differ only in
/// three axes, all derived here from (`hd`, `kvf`):
///   - dims-per-lane = hd/32 (each of the 32 warp lanes owns this many head
///     dims), which sets the unroll of the Q load, dot product, V accumulate,
///     and scratch store;
///   - KV load: f32 reads v4.f32 (4 dims each); f16 reads v4.u32 (8 packed
///     halfs) and widens with cvt.f32.f16, at a *2 byte stride instead of *4;
///     q8_0 addresses the row's 34-byte block (a lane's fragment never
///     straddles one: fragments are dims-aligned and dims divides 32), loads
///     the f16 scale once and the quants as u16 pairs, sign-extends and
///     multiplies — see emitKVLoad;
///   - ring addressing (u6) wraps the KV row (j%ring) for LOCAL sliding-window
///     layers; GLOBAL layers pass ring=0 (linear) — same code, no branch cost.
/// The masking is uniform: causal kv_len = kv_len0+t, a bidirectional image
/// block (u7) extends the upper bound (kv_end) to kv_len0+seq_q-1 while the
/// sliding window (f1) keeps kv_start on the query's own causal position.
/// Register banks: %f1..%f9 scalars (%f10 = q8_0 block scale); Q dims
/// [16..16+D); KV-temp [16+D..16+2D) (K then V); acc [16+2D..16+3D).
/// Validated by `ptxas` + the on-device opAttnDecode tests (causal / window /
/// f16 / q8_0 / bidirectional).
fn genAttnSplit(comptime hd: u32, comptime kvf: KvFmt) [:0]const u8 {
    @setEvalBranchQuota(200000);
    const cp = std.fmt.comptimePrint;
    const dims = hd / 32; // head dims per lane (8 for hd256, 16 for hd512)
    const qshl = std.math.log2_int(u32, dims); // lane element offset shift
    const sshl = qshl + 2; // lane byte offset for the scratch acc store
    const QB = 16; // Q register base
    const KB = 16 + dims; // KV-temp register base
    const AB = 16 + 2 * dims; // accumulator register base
    const suffix = switch (kvf) {
        .f32 => "",
        .f16 => "_f16",
        .q8_0 => "_q8",
    };
    const name = cp("attn_split_h{d}{s}", .{ hd, suffix });

    comptime var s: []const u8 = ".version 8.0\n.target sm_86\n.address_size 64\n";
    s = s ++ cp(".visible .entry {s}(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,\n", .{name});
    s = s ++ "  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .u32 u6,.param .u32 u7,.param .f32 f0,.param .f32 f1)\n{\n";
    s = s ++ "  .reg .pred %p<8>;\n  .reg .b32 %r<64>;\n  .reg .f32 %f<128>;\n";
    if (kvf != .f32) s = s ++ "  .reg .b16 %hs<2>;\n";
    s = s ++ "  .reg .b64 %rd<32>;\n";

    // --- thread / warp / lane, params, per-query kv_len (causal) + kv_end ---
    s = s ++
        \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
        \\  mad.lo.s32 %r4,%r1,%r2,%r3;
        \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
        \\  and.b32 %r28,%r4,31;                  // lane
        \\  ld.param.u32 %r5,[u0];                // kv_len0
        \\  ld.param.u32 %r6,[u1];                // heads
        \\  ld.param.u32 %r26,[u4];               // nsplit
        \\  ld.param.u32 %r30,[u5];               // seq_q
        \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
        \\  mul.lo.s32 %r31,%r7,%r30;
        \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
        \\  ld.param.u32 %r8,[u2];                // kv_heads
        \\  ld.param.u32 %r9,[u3];                // hd
        \\  ld.param.f32 %f1,[f0];                // scale
        \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
        \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
        \\  div.u32 %r31,%r27,%r7;                // query t
        \\  rem.u32 %r2,%r27,%r7;                 // warp within query
        \\  add.u32 %r5,%r5,%r31;                 // causal kv len (kv_len0+t)
        \\  ld.param.u32 %r34,[u7];               // bidir (0/1)
        \\  mov.u32 %r35,%r5;                      // kv_end = causal len
        \\  setp.eq.u32 %p6,%r34,0; @%p6 bra NOBIDIR;
        \\  add.u32 %r35,%r5,%r30; sub.u32 %r35,%r35,1; sub.u32 %r35,%r35,%r31; // kv_len0+seq_q-1
        \\NOBIDIR:
        \\  div.u32 %r10,%r2,%r26;                // h
        \\  rem.u32 %r21,%r2,%r26;                // split i
        \\  ld.param.f32 %f9,[f1]; cvt.rzi.u32.f32 %r11,%f9;   // window (0 = full causal)
        \\  mov.u32 %r16,0;                       // kv_start
        \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
        \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
        \\  sub.u32 %r16,%r5,%r11;                // kv_start = causal_len - window
        \\NOWIN:
        \\  sub.u32 %r22,%r35,%r16;               // span = kv_end - kv_start
        \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
        \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
        \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r35; // kv1
        \\  div.u32 %r12,%r6,%r8;                 // group
        \\  div.u32 %r13,%r10,%r12;               // kv head
        \\
    ;

    // --- Q fragment load: (t*heads+h)*hd + lane*dims elements, dims f32 ---
    s = s ++ cp("  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,{d}; add.u32 %r14,%r14,%r15;\n", .{qshl});
    s = s ++ "  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;\n";
    inline for (0..dims / 4) |L| {
        s = s ++ cp("  ld.global.v4.f32 {{%f{d},%f{d},%f{d},%f{d}}},[%rd6+{d}];\n", .{ QB + L * 4, QB + L * 4 + 1, QB + L * 4 + 2, QB + L * 4 + 3, L * 16 });
    }

    // --- online-softmax accumulators: m=%f6, d=%f7, acc=[AB..AB+dims) ---
    s = s ++ "  mov.f32 %f6,0fFF800000;              // m\n  mov.f32 %f7,0f00000000;              // d\n";
    inline for (0..dims) |dd| s = s ++ cp("  mov.f32 %f{d},0f00000000;\n", .{AB + dd});
    s = s ++ "  ld.param.u32 %r32,[u6];              // ring (0 = linear KV addressing)\n";
    if (kvf == .q8_0) {
        // Loop-invariant quant byte offset within the lane's block: rows are
        // block-aligned, so elem%32 == lane_off%32; quants start at +2.
        s = s ++ "  and.b32 %r45,%r15,31; add.u32 %r45,%r45,2; cvt.u64.u32 %rd21,%r45;\n";
    }

    // --- KV loop ---
    s = s ++
        \\JLOOP:
        \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
        \\  mov.u32 %r33,%r17;
        \\  setp.eq.u32 %p5,%r32,0; @%p5 bra ROWJ;
        \\  rem.u32 %r33,%r17,%r32;               // ring wrap: j % ring
        \\ROWJ:
        \\  mad.lo.s32 %r18,%r33,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
        \\
    ;
    if (kvf == .q8_0) {
        // Byte offset of the lane's 34-byte block: (elem >> 5) * 34. The
        // fragment (dims <= 16 elems, dims-aligned) never straddles a block.
        s = s ++ "  shr.u32 %r44,%r18,5; mul.wide.u32 %rd9,%r44,34; add.s64 %rd10,%rd2,%rd9;\n";
    } else {
        const kv_stride = if (kvf == .f16) 2 else 4;
        s = s ++ cp("  mul.wide.u32 %rd9,%r18,{d}; add.s64 %rd10,%rd2,%rd9;\n", .{kv_stride});
    }
    s = s ++ emitKVLoad(kvf, dims, KB, "%rd10");

    // --- dot(q, k) into %f2, then butterfly all-reduce across the 32 lanes ---
    s = s ++ cp("  mul.f32 %f2,%f{d},%f{d};", .{ QB, KB });
    inline for (1..dims) |dd| s = s ++ cp(" fma.rn.f32 %f2,%f{d},%f{d},%f2;", .{ QB + dd, KB + dd });
    s = s ++ "\n";
    inline for ([_]u32{ 16, 8, 4, 2, 1 }) |off| {
        s = s ++ cp("  mov.b32 %r19,%f2; shfl.sync.bfly.b32 %r20,%r19,{d},0x1f,0xffffffff; mov.b32 %f3,%r20; add.f32 %f2,%f2,%f3;\n", .{off});
    }

    // --- online softmax update (m=%f6, d=%f7, m2=%f8, corr=%f4, p=%f5) ---
    s = s ++
        \\  mul.f32 %f2,%f2,%f1;                  // s = dot*scale
        \\  max.f32 %f8,%f6,%f2;                  // m2
        \\  sub.f32 %f4,%f6,%f8; mul.f32 %f4,%f4,0f3FB8AA3B; ex2.approx.f32 %f4,%f4;  // corr
        \\  sub.f32 %f5,%f2,%f8; mul.f32 %f5,%f5,0f3FB8AA3B; ex2.approx.f32 %f5,%f5;  // p
        \\  mul.f32 %f7,%f7,%f4; add.f32 %f7,%f7,%f5;   // d = d*corr + p
        \\  mov.f32 %f6,%f8;                      // m = m2
        \\
    ;

    // --- V load (same row offsets, base rd3) then acc = acc*corr + p*v ---
    s = s ++ "  add.s64 %rd11,%rd3,%rd9;\n";
    s = s ++ emitKVLoad(kvf, dims, KB, "%rd11");
    inline for (0..dims) |dd| {
        s = s ++ cp("  mul.f32 %f{d},%f{d},%f4; fma.rn.f32 %f{d},%f5,%f{d},%f{d};\n", .{ AB + dd, AB + dd, AB + dd, KB + dd, AB + dd });
    }
    s = s ++ "  add.u32 %r17,%r17,1; bra JLOOP;\n";

    // --- store partials: scratch row = warp*(hd+4); lane 0 writes m,d ---
    s = s ++
        \\JD:
        \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
        \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
        \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
        \\  st.global.f32 [%rd18],%f6; st.global.f32 [%rd18+4],%f7;   // m, d
        \\WRACC:
        \\
    ;
    s = s ++ cp("  shl.b32 %r29,%r28,{d}; add.u32 %r29,%r29,16;   // byte off = 16 + lane*dims*4\n", .{sshl});
    s = s ++ "  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;\n";
    inline for (0..dims / 4) |L| {
        s = s ++ cp("  st.global.v4.f32 [%rd20+{d}],{{%f{d},%f{d},%f{d},%f{d}}};\n", .{ L * 16, AB + L * 4, AB + L * 4 + 1, AB + L * 4 + 2, AB + L * 4 + 3 });
    }
    s = s ++ "END:\n  ret;\n}\n";

    return (s ++ [_]u8{0})[0..s.len :0];
}

/// Emit the load of `dims` head-dim f32 values into registers [base..base+dims)
/// from the global address in `addr`. f32 reads v4.f32 (4 dims/load); f16 reads
/// v4.u32 (8 packed halfs/load) and widens each with cvt.f32.f16. q8_0 expects
/// `addr` to point at the lane's 34-byte block (and %rd21 = 2 + elem%32, the
/// loop-invariant quant offset): it loads the block's f16 scale once, then the
/// quants as u16 pairs (2-byte aligned — dims is even), sign-extends each i8
/// with shl/shr.s32 and multiplies by the scale into the same f32 registers.
fn emitKVLoad(comptime kvf: KvFmt, comptime dims: u32, comptime base: u32, comptime addr: []const u8) []const u8 {
    const cp = std.fmt.comptimePrint;
    comptime var s: []const u8 = "";
    switch (kvf) {
        .f16 => {
            inline for (0..dims / 8) |L| {
                s = s ++ cp("  ld.global.v4.u32 {{%r40,%r41,%r42,%r43}},[{s}+{d}];\n", .{ addr, L * 16 });
                inline for (0..4) |w| {
                    const d0 = base + L * 8 + w * 2;
                    s = s ++ cp("  mov.b32 {{%hs0,%hs1}},%r{d}; cvt.f32.f16 %f{d},%hs0; cvt.f32.f16 %f{d},%hs1;\n", .{ 40 + w, d0, d0 + 1 });
                }
            }
        },
        .q8_0 => {
            s = s ++ cp("  ld.global.b16 %hs0,[{s}]; cvt.f32.f16 %f10,%hs0;\n", .{addr});
            s = s ++ cp("  add.s64 %rd12,{s},%rd21;\n", .{addr});
            inline for (0..dims / 2) |w| {
                const d0 = base + w * 2;
                s = s ++ cp("  ld.global.u16 %r46,[%rd12+{d}];\n", .{w * 2});
                s = s ++ cp("  shl.b32 %r47,%r46,24; shr.s32 %r47,%r47,24; cvt.rn.f32.s32 %f{d},%r47; mul.f32 %f{d},%f{d},%f10;\n", .{ d0, d0, d0 });
                s = s ++ cp("  shl.b32 %r47,%r46,16; shr.s32 %r47,%r47,24; cvt.rn.f32.s32 %f{d},%r47; mul.f32 %f{d},%f{d},%f10;\n", .{ d0 + 1, d0 + 1, d0 + 1 });
            }
        },
        .f32 => {
            inline for (0..dims / 4) |L| {
                s = s ++ cp("  ld.global.v4.f32 {{%f{d},%f{d},%f{d},%f{d}}},[{s}+{d}];\n", .{ base + L * 4, base + L * 4 + 1, base + L * 4 + 2, base + L * 4 + 3, addr, L * 16 });
            }
        },
    }
    return s;
}

pub const attn_split_h256_ptx = genAttnSplit(256, .f32);
pub const attn_split_h512_ptx = genAttnSplit(512, .f32);
pub const attn_split_h256_f16_ptx = genAttnSplit(256, .f16);
pub const attn_split_h512_f16_ptx = genAttnSplit(512, .f16);
pub const attn_split_h256_q8_ptx = genAttnSplit(256, .q8_0);
pub const attn_split_h512_q8_ptx = genAttnSplit(512, .q8_0);

/// Multi-input fp8 GEMV (speculative-decode verify / short prefills):
/// y[i][row] = scale * dot(W[row], x_i) for n <= 4 input vectors. One block
/// per EIGHT weight rows (rows % 8 == 0): with block-per-row, every block
/// re-reads all four x rows and the kernel saturates L2 at ~80 GB/s of
/// weight traffic; amortizing x over 8 rows restores the W-stream-bound
/// regime. Per-thread element order matches gemv_fp8 (c = tid*8, stride
/// 2048) and each accumulator sums in that same order, so results are
/// bitwise identical to the single-input kernel — greedy speculative decode
/// stays byte-identical to vanilla. x must have 4 rows of backing store
/// (garbage rows beyond n are computed and discarded via predicated stores);
/// W streams with .cs (evict-first). 32 accumulators (8 rows x 4 inputs)
/// reduce through 32 shared arrays.
/// b0=W, b1=x [4][cols], b2=y [n][rows], b3=lut. u0=rows, u1=cols, u2=n, f0=scale.
pub const gemv_fp8n_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_fp8n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<12>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<80>;
    \\  .reg .b64 %rd<40>;
    \\  .shared .align 4 .b8 lut_s[1024];
    \\  .shared .align 4 .b8 red[32768];
    \\  mov.u32 %r1,%ctaid.x;                  // row group (8 weight rows)
    \\  ld.param.u32 %r2,[u0];                 // rows
    \\  shl.b32 %r35,%r1,3;                    // r0 = group*8
    \\  setp.ge.u32 %p1,%r35,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r20,[u2];                // n inputs (1..4)
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  // stage the fp8->f32 LUT: lut_s[tid] = lut[tid]
    \\  shl.b32 %r5,%r3,2;
    \\  mul.wide.u32 %rd5,%r3,4; add.s64 %rd6,%rd4,%rd5; ld.global.f32 %f2,[%rd6];
    \\  mov.u32 %r6,lut_s; add.u32 %r7,%r6,%r5; st.shared.f32 [%r7],%f2;
    \\  bar.sync 0;
    \\  // x row base pointers rd20..rd23 = x + i*cols*4
    \\  mul.wide.u32 %rd16,%r4,4;
    \\  mov.u64 %rd20,%rd2;
    \\  add.s64 %rd21,%rd20,%rd16;
    \\  add.s64 %rd22,%rd21,%rd16;
    \\  add.s64 %rd23,%rd22,%rd16;
    \\  // W row base pointers rd24..rd31 = W + (r0+j)*cols
    \\  mul.wide.u32 %rd7,%r35,%r4; add.s64 %rd24,%rd1,%rd7;
    \\  cvt.u64.u32 %rd8,%r4;
    \\  add.s64 %rd25,%rd24,%rd8;
    \\  add.s64 %rd26,%rd25,%rd8;
    \\  add.s64 %rd27,%rd26,%rd8;
    \\  add.s64 %rd28,%rd27,%rd8;
    \\  add.s64 %rd29,%rd28,%rd8;
    \\  add.s64 %rd30,%rd29,%rd8;
    \\  add.s64 %rd31,%rd30,%rd8;
    \\  mov.f32 %f44,0f00000000;
    \\  mov.f32 %f45,0f00000000;
    \\  mov.f32 %f46,0f00000000;
    \\  mov.f32 %f47,0f00000000;
    \\  mov.f32 %f48,0f00000000;
    \\  mov.f32 %f49,0f00000000;
    \\  mov.f32 %f50,0f00000000;
    \\  mov.f32 %f51,0f00000000;
    \\  mov.f32 %f52,0f00000000;
    \\  mov.f32 %f53,0f00000000;
    \\  mov.f32 %f54,0f00000000;
    \\  mov.f32 %f55,0f00000000;
    \\  mov.f32 %f56,0f00000000;
    \\  mov.f32 %f57,0f00000000;
    \\  mov.f32 %f58,0f00000000;
    \\  mov.f32 %f59,0f00000000;
    \\  mov.f32 %f60,0f00000000;
    \\  mov.f32 %f61,0f00000000;
    \\  mov.f32 %f62,0f00000000;
    \\  mov.f32 %f63,0f00000000;
    \\  mov.f32 %f64,0f00000000;
    \\  mov.f32 %f65,0f00000000;
    \\  mov.f32 %f66,0f00000000;
    \\  mov.f32 %f67,0f00000000;
    \\  mov.f32 %f68,0f00000000;
    \\  mov.f32 %f69,0f00000000;
    \\  mov.f32 %f70,0f00000000;
    \\  mov.f32 %f71,0f00000000;
    \\  mov.f32 %f72,0f00000000;
    \\  mov.f32 %f73,0f00000000;
    \\  mov.f32 %f74,0f00000000;
    \\  mov.f32 %f75,0f00000000;
    \\  shl.b32 %r8,%r3,3;                     // c = tid*8 (same order as gemv_fp8: bitwise-stable)
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  // all 4 inputs' elems c..c+7 (32 registers), loaded once per row GROUP:
    \\  // amortizing x L2 traffic over 8 weight rows is the whole point.
    \\  mul.wide.u32 %rd11,%r8,4;
    \\  add.s64 %rd12,%rd20,%rd11; ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd12];    ld.global.v4.f32 {%f6,%f7,%f8,%f9},[%rd12+16];
    \\  add.s64 %rd13,%rd21,%rd11; ld.global.v4.f32 {%f10,%f11,%f12,%f13},[%rd13]; ld.global.v4.f32 {%f14,%f15,%f16,%f17},[%rd13+16];
    \\  add.s64 %rd14,%rd22,%rd11; ld.global.v4.f32 {%f18,%f19,%f20,%f21},[%rd14]; ld.global.v4.f32 {%f22,%f23,%f24,%f25},[%rd14+16];
    \\  add.s64 %rd15,%rd23,%rd11; ld.global.v4.f32 {%f26,%f27,%f28,%f29},[%rd15]; ld.global.v4.f32 {%f30,%f31,%f32,%f33},[%rd15+16];
    \\  cvt.u64.u32 %rd9,%r8;
    \\  // row 0: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd24,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f44,%f34,%f2,%f44; fma.rn.f32 %f45,%f34,%f10,%f45; fma.rn.f32 %f46,%f34,%f18,%f46; fma.rn.f32 %f47,%f34,%f26,%f47;
    \\  fma.rn.f32 %f44,%f35,%f3,%f44; fma.rn.f32 %f45,%f35,%f11,%f45; fma.rn.f32 %f46,%f35,%f19,%f46; fma.rn.f32 %f47,%f35,%f27,%f47;
    \\  fma.rn.f32 %f44,%f36,%f4,%f44; fma.rn.f32 %f45,%f36,%f12,%f45; fma.rn.f32 %f46,%f36,%f20,%f46; fma.rn.f32 %f47,%f36,%f28,%f47;
    \\  fma.rn.f32 %f44,%f37,%f5,%f44; fma.rn.f32 %f45,%f37,%f13,%f45; fma.rn.f32 %f46,%f37,%f21,%f46; fma.rn.f32 %f47,%f37,%f29,%f47;
    \\  fma.rn.f32 %f44,%f38,%f6,%f44; fma.rn.f32 %f45,%f38,%f14,%f45; fma.rn.f32 %f46,%f38,%f22,%f46; fma.rn.f32 %f47,%f38,%f30,%f47;
    \\  fma.rn.f32 %f44,%f39,%f7,%f44; fma.rn.f32 %f45,%f39,%f15,%f45; fma.rn.f32 %f46,%f39,%f23,%f46; fma.rn.f32 %f47,%f39,%f31,%f47;
    \\  fma.rn.f32 %f44,%f40,%f8,%f44; fma.rn.f32 %f45,%f40,%f16,%f45; fma.rn.f32 %f46,%f40,%f24,%f46; fma.rn.f32 %f47,%f40,%f32,%f47;
    \\  fma.rn.f32 %f44,%f41,%f9,%f44; fma.rn.f32 %f45,%f41,%f17,%f45; fma.rn.f32 %f46,%f41,%f25,%f46; fma.rn.f32 %f47,%f41,%f33,%f47;
    \\  // row 1: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd25,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f48,%f34,%f2,%f48; fma.rn.f32 %f49,%f34,%f10,%f49; fma.rn.f32 %f50,%f34,%f18,%f50; fma.rn.f32 %f51,%f34,%f26,%f51;
    \\  fma.rn.f32 %f48,%f35,%f3,%f48; fma.rn.f32 %f49,%f35,%f11,%f49; fma.rn.f32 %f50,%f35,%f19,%f50; fma.rn.f32 %f51,%f35,%f27,%f51;
    \\  fma.rn.f32 %f48,%f36,%f4,%f48; fma.rn.f32 %f49,%f36,%f12,%f49; fma.rn.f32 %f50,%f36,%f20,%f50; fma.rn.f32 %f51,%f36,%f28,%f51;
    \\  fma.rn.f32 %f48,%f37,%f5,%f48; fma.rn.f32 %f49,%f37,%f13,%f49; fma.rn.f32 %f50,%f37,%f21,%f50; fma.rn.f32 %f51,%f37,%f29,%f51;
    \\  fma.rn.f32 %f48,%f38,%f6,%f48; fma.rn.f32 %f49,%f38,%f14,%f49; fma.rn.f32 %f50,%f38,%f22,%f50; fma.rn.f32 %f51,%f38,%f30,%f51;
    \\  fma.rn.f32 %f48,%f39,%f7,%f48; fma.rn.f32 %f49,%f39,%f15,%f49; fma.rn.f32 %f50,%f39,%f23,%f50; fma.rn.f32 %f51,%f39,%f31,%f51;
    \\  fma.rn.f32 %f48,%f40,%f8,%f48; fma.rn.f32 %f49,%f40,%f16,%f49; fma.rn.f32 %f50,%f40,%f24,%f50; fma.rn.f32 %f51,%f40,%f32,%f51;
    \\  fma.rn.f32 %f48,%f41,%f9,%f48; fma.rn.f32 %f49,%f41,%f17,%f49; fma.rn.f32 %f50,%f41,%f25,%f50; fma.rn.f32 %f51,%f41,%f33,%f51;
    \\  // row 2: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd26,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f52,%f34,%f2,%f52; fma.rn.f32 %f53,%f34,%f10,%f53; fma.rn.f32 %f54,%f34,%f18,%f54; fma.rn.f32 %f55,%f34,%f26,%f55;
    \\  fma.rn.f32 %f52,%f35,%f3,%f52; fma.rn.f32 %f53,%f35,%f11,%f53; fma.rn.f32 %f54,%f35,%f19,%f54; fma.rn.f32 %f55,%f35,%f27,%f55;
    \\  fma.rn.f32 %f52,%f36,%f4,%f52; fma.rn.f32 %f53,%f36,%f12,%f53; fma.rn.f32 %f54,%f36,%f20,%f54; fma.rn.f32 %f55,%f36,%f28,%f55;
    \\  fma.rn.f32 %f52,%f37,%f5,%f52; fma.rn.f32 %f53,%f37,%f13,%f53; fma.rn.f32 %f54,%f37,%f21,%f54; fma.rn.f32 %f55,%f37,%f29,%f55;
    \\  fma.rn.f32 %f52,%f38,%f6,%f52; fma.rn.f32 %f53,%f38,%f14,%f53; fma.rn.f32 %f54,%f38,%f22,%f54; fma.rn.f32 %f55,%f38,%f30,%f55;
    \\  fma.rn.f32 %f52,%f39,%f7,%f52; fma.rn.f32 %f53,%f39,%f15,%f53; fma.rn.f32 %f54,%f39,%f23,%f54; fma.rn.f32 %f55,%f39,%f31,%f55;
    \\  fma.rn.f32 %f52,%f40,%f8,%f52; fma.rn.f32 %f53,%f40,%f16,%f53; fma.rn.f32 %f54,%f40,%f24,%f54; fma.rn.f32 %f55,%f40,%f32,%f55;
    \\  fma.rn.f32 %f52,%f41,%f9,%f52; fma.rn.f32 %f53,%f41,%f17,%f53; fma.rn.f32 %f54,%f41,%f25,%f54; fma.rn.f32 %f55,%f41,%f33,%f55;
    \\  // row 3: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd27,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f56,%f34,%f2,%f56; fma.rn.f32 %f57,%f34,%f10,%f57; fma.rn.f32 %f58,%f34,%f18,%f58; fma.rn.f32 %f59,%f34,%f26,%f59;
    \\  fma.rn.f32 %f56,%f35,%f3,%f56; fma.rn.f32 %f57,%f35,%f11,%f57; fma.rn.f32 %f58,%f35,%f19,%f58; fma.rn.f32 %f59,%f35,%f27,%f59;
    \\  fma.rn.f32 %f56,%f36,%f4,%f56; fma.rn.f32 %f57,%f36,%f12,%f57; fma.rn.f32 %f58,%f36,%f20,%f58; fma.rn.f32 %f59,%f36,%f28,%f59;
    \\  fma.rn.f32 %f56,%f37,%f5,%f56; fma.rn.f32 %f57,%f37,%f13,%f57; fma.rn.f32 %f58,%f37,%f21,%f58; fma.rn.f32 %f59,%f37,%f29,%f59;
    \\  fma.rn.f32 %f56,%f38,%f6,%f56; fma.rn.f32 %f57,%f38,%f14,%f57; fma.rn.f32 %f58,%f38,%f22,%f58; fma.rn.f32 %f59,%f38,%f30,%f59;
    \\  fma.rn.f32 %f56,%f39,%f7,%f56; fma.rn.f32 %f57,%f39,%f15,%f57; fma.rn.f32 %f58,%f39,%f23,%f58; fma.rn.f32 %f59,%f39,%f31,%f59;
    \\  fma.rn.f32 %f56,%f40,%f8,%f56; fma.rn.f32 %f57,%f40,%f16,%f57; fma.rn.f32 %f58,%f40,%f24,%f58; fma.rn.f32 %f59,%f40,%f32,%f59;
    \\  fma.rn.f32 %f56,%f41,%f9,%f56; fma.rn.f32 %f57,%f41,%f17,%f57; fma.rn.f32 %f58,%f41,%f25,%f58; fma.rn.f32 %f59,%f41,%f33,%f59;
    \\  // row 4: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd28,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f60,%f34,%f2,%f60; fma.rn.f32 %f61,%f34,%f10,%f61; fma.rn.f32 %f62,%f34,%f18,%f62; fma.rn.f32 %f63,%f34,%f26,%f63;
    \\  fma.rn.f32 %f60,%f35,%f3,%f60; fma.rn.f32 %f61,%f35,%f11,%f61; fma.rn.f32 %f62,%f35,%f19,%f62; fma.rn.f32 %f63,%f35,%f27,%f63;
    \\  fma.rn.f32 %f60,%f36,%f4,%f60; fma.rn.f32 %f61,%f36,%f12,%f61; fma.rn.f32 %f62,%f36,%f20,%f62; fma.rn.f32 %f63,%f36,%f28,%f63;
    \\  fma.rn.f32 %f60,%f37,%f5,%f60; fma.rn.f32 %f61,%f37,%f13,%f61; fma.rn.f32 %f62,%f37,%f21,%f62; fma.rn.f32 %f63,%f37,%f29,%f63;
    \\  fma.rn.f32 %f60,%f38,%f6,%f60; fma.rn.f32 %f61,%f38,%f14,%f61; fma.rn.f32 %f62,%f38,%f22,%f62; fma.rn.f32 %f63,%f38,%f30,%f63;
    \\  fma.rn.f32 %f60,%f39,%f7,%f60; fma.rn.f32 %f61,%f39,%f15,%f61; fma.rn.f32 %f62,%f39,%f23,%f62; fma.rn.f32 %f63,%f39,%f31,%f63;
    \\  fma.rn.f32 %f60,%f40,%f8,%f60; fma.rn.f32 %f61,%f40,%f16,%f61; fma.rn.f32 %f62,%f40,%f24,%f62; fma.rn.f32 %f63,%f40,%f32,%f63;
    \\  fma.rn.f32 %f60,%f41,%f9,%f60; fma.rn.f32 %f61,%f41,%f17,%f61; fma.rn.f32 %f62,%f41,%f25,%f62; fma.rn.f32 %f63,%f41,%f33,%f63;
    \\  // row 5: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd29,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f64,%f34,%f2,%f64; fma.rn.f32 %f65,%f34,%f10,%f65; fma.rn.f32 %f66,%f34,%f18,%f66; fma.rn.f32 %f67,%f34,%f26,%f67;
    \\  fma.rn.f32 %f64,%f35,%f3,%f64; fma.rn.f32 %f65,%f35,%f11,%f65; fma.rn.f32 %f66,%f35,%f19,%f66; fma.rn.f32 %f67,%f35,%f27,%f67;
    \\  fma.rn.f32 %f64,%f36,%f4,%f64; fma.rn.f32 %f65,%f36,%f12,%f65; fma.rn.f32 %f66,%f36,%f20,%f66; fma.rn.f32 %f67,%f36,%f28,%f67;
    \\  fma.rn.f32 %f64,%f37,%f5,%f64; fma.rn.f32 %f65,%f37,%f13,%f65; fma.rn.f32 %f66,%f37,%f21,%f66; fma.rn.f32 %f67,%f37,%f29,%f67;
    \\  fma.rn.f32 %f64,%f38,%f6,%f64; fma.rn.f32 %f65,%f38,%f14,%f65; fma.rn.f32 %f66,%f38,%f22,%f66; fma.rn.f32 %f67,%f38,%f30,%f67;
    \\  fma.rn.f32 %f64,%f39,%f7,%f64; fma.rn.f32 %f65,%f39,%f15,%f65; fma.rn.f32 %f66,%f39,%f23,%f66; fma.rn.f32 %f67,%f39,%f31,%f67;
    \\  fma.rn.f32 %f64,%f40,%f8,%f64; fma.rn.f32 %f65,%f40,%f16,%f65; fma.rn.f32 %f66,%f40,%f24,%f66; fma.rn.f32 %f67,%f40,%f32,%f67;
    \\  fma.rn.f32 %f64,%f41,%f9,%f64; fma.rn.f32 %f65,%f41,%f17,%f65; fma.rn.f32 %f66,%f41,%f25,%f66; fma.rn.f32 %f67,%f41,%f33,%f67;
    \\  // row 6: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd30,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f68,%f34,%f2,%f68; fma.rn.f32 %f69,%f34,%f10,%f69; fma.rn.f32 %f70,%f34,%f18,%f70; fma.rn.f32 %f71,%f34,%f26,%f71;
    \\  fma.rn.f32 %f68,%f35,%f3,%f68; fma.rn.f32 %f69,%f35,%f11,%f69; fma.rn.f32 %f70,%f35,%f19,%f70; fma.rn.f32 %f71,%f35,%f27,%f71;
    \\  fma.rn.f32 %f68,%f36,%f4,%f68; fma.rn.f32 %f69,%f36,%f12,%f69; fma.rn.f32 %f70,%f36,%f20,%f70; fma.rn.f32 %f71,%f36,%f28,%f71;
    \\  fma.rn.f32 %f68,%f37,%f5,%f68; fma.rn.f32 %f69,%f37,%f13,%f69; fma.rn.f32 %f70,%f37,%f21,%f70; fma.rn.f32 %f71,%f37,%f29,%f71;
    \\  fma.rn.f32 %f68,%f38,%f6,%f68; fma.rn.f32 %f69,%f38,%f14,%f69; fma.rn.f32 %f70,%f38,%f22,%f70; fma.rn.f32 %f71,%f38,%f30,%f71;
    \\  fma.rn.f32 %f68,%f39,%f7,%f68; fma.rn.f32 %f69,%f39,%f15,%f69; fma.rn.f32 %f70,%f39,%f23,%f70; fma.rn.f32 %f71,%f39,%f31,%f71;
    \\  fma.rn.f32 %f68,%f40,%f8,%f68; fma.rn.f32 %f69,%f40,%f16,%f69; fma.rn.f32 %f70,%f40,%f24,%f70; fma.rn.f32 %f71,%f40,%f32,%f71;
    \\  fma.rn.f32 %f68,%f41,%f9,%f68; fma.rn.f32 %f69,%f41,%f17,%f69; fma.rn.f32 %f70,%f41,%f25,%f70; fma.rn.f32 %f71,%f41,%f33,%f71;
    \\  // row 7: decode 8 fp8 once, 32 FMAs interleaved across the 4 accumulators
    \\  add.s64 %rd10,%rd31,%rd9; ld.global.cs.v2.u32 {%r9,%r18},[%rd10];
    \\  and.b32 %r10,%r9,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f34,[%r12];
    \\  shr.u32 %r10,%r9,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f35,[%r12];
    \\  shr.u32 %r10,%r9,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f36,[%r12];
    \\  shr.u32 %r10,%r9,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f37,[%r12];
    \\  and.b32 %r10,%r18,255;                shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f38,[%r12];
    \\  shr.u32 %r10,%r18,8; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f39,[%r12];
    \\  shr.u32 %r10,%r18,16; and.b32 %r10,%r10,255; shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f40,[%r12];
    \\  shr.u32 %r10,%r18,24;                 shl.b32 %r11,%r10,2; add.u32 %r12,%r6,%r11; ld.shared.f32 %f41,[%r12];
    \\  fma.rn.f32 %f72,%f34,%f2,%f72; fma.rn.f32 %f73,%f34,%f10,%f73; fma.rn.f32 %f74,%f34,%f18,%f74; fma.rn.f32 %f75,%f34,%f26,%f75;
    \\  fma.rn.f32 %f72,%f35,%f3,%f72; fma.rn.f32 %f73,%f35,%f11,%f73; fma.rn.f32 %f74,%f35,%f19,%f74; fma.rn.f32 %f75,%f35,%f27,%f75;
    \\  fma.rn.f32 %f72,%f36,%f4,%f72; fma.rn.f32 %f73,%f36,%f12,%f73; fma.rn.f32 %f74,%f36,%f20,%f74; fma.rn.f32 %f75,%f36,%f28,%f75;
    \\  fma.rn.f32 %f72,%f37,%f5,%f72; fma.rn.f32 %f73,%f37,%f13,%f73; fma.rn.f32 %f74,%f37,%f21,%f74; fma.rn.f32 %f75,%f37,%f29,%f75;
    \\  fma.rn.f32 %f72,%f38,%f6,%f72; fma.rn.f32 %f73,%f38,%f14,%f73; fma.rn.f32 %f74,%f38,%f22,%f74; fma.rn.f32 %f75,%f38,%f30,%f75;
    \\  fma.rn.f32 %f72,%f39,%f7,%f72; fma.rn.f32 %f73,%f39,%f15,%f73; fma.rn.f32 %f74,%f39,%f23,%f74; fma.rn.f32 %f75,%f39,%f31,%f75;
    \\  fma.rn.f32 %f72,%f40,%f8,%f72; fma.rn.f32 %f73,%f40,%f16,%f73; fma.rn.f32 %f74,%f40,%f24,%f74; fma.rn.f32 %f75,%f40,%f32,%f75;
    \\  fma.rn.f32 %f72,%f41,%f9,%f72; fma.rn.f32 %f73,%f41,%f17,%f73; fma.rn.f32 %f74,%f41,%f25,%f74; fma.rn.f32 %f75,%f41,%f33,%f75;
    \\  add.u32 %r8,%r8,2048; bra LOOP;
    \\LD:
    \\  // red is 32 arrays of 256 f32: &red[a][tid] = red + a*1024 + tid*4, a = j*4 + i
    \\  mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14+0],%f44;
    \\  st.shared.f32 [%r14+1024],%f45;
    \\  st.shared.f32 [%r14+2048],%f46;
    \\  st.shared.f32 [%r14+3072],%f47;
    \\  st.shared.f32 [%r14+4096],%f48;
    \\  st.shared.f32 [%r14+5120],%f49;
    \\  st.shared.f32 [%r14+6144],%f50;
    \\  st.shared.f32 [%r14+7168],%f51;
    \\  st.shared.f32 [%r14+8192],%f52;
    \\  st.shared.f32 [%r14+9216],%f53;
    \\  st.shared.f32 [%r14+10240],%f54;
    \\  st.shared.f32 [%r14+11264],%f55;
    \\  st.shared.f32 [%r14+12288],%f56;
    \\  st.shared.f32 [%r14+13312],%f57;
    \\  st.shared.f32 [%r14+14336],%f58;
    \\  st.shared.f32 [%r14+15360],%f59;
    \\  st.shared.f32 [%r14+16384],%f60;
    \\  st.shared.f32 [%r14+17408],%f61;
    \\  st.shared.f32 [%r14+18432],%f62;
    \\  st.shared.f32 [%r14+19456],%f63;
    \\  st.shared.f32 [%r14+20480],%f64;
    \\  st.shared.f32 [%r14+21504],%f65;
    \\  st.shared.f32 [%r14+22528],%f66;
    \\  st.shared.f32 [%r14+23552],%f67;
    \\  st.shared.f32 [%r14+24576],%f68;
    \\  st.shared.f32 [%r14+25600],%f69;
    \\  st.shared.f32 [%r14+26624],%f70;
    \\  st.shared.f32 [%r14+27648],%f71;
    \\  st.shared.f32 [%r14+28672],%f72;
    \\  st.shared.f32 [%r14+29696],%f73;
    \\  st.shared.f32 [%r14+30720],%f74;
    \\  st.shared.f32 [%r14+31744],%f75;
    \\  bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  shl.b32 %r16,%r15,2; add.u32 %r17,%r14,%r16;
    \\  ld.shared.f32 %f2,[%r14+0]; ld.shared.f32 %f3,[%r17+0]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+0],%f2;
    \\  ld.shared.f32 %f2,[%r14+1024]; ld.shared.f32 %f3,[%r17+1024]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+1024],%f2;
    \\  ld.shared.f32 %f2,[%r14+2048]; ld.shared.f32 %f3,[%r17+2048]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+2048],%f2;
    \\  ld.shared.f32 %f2,[%r14+3072]; ld.shared.f32 %f3,[%r17+3072]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+3072],%f2;
    \\  ld.shared.f32 %f2,[%r14+4096]; ld.shared.f32 %f3,[%r17+4096]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+4096],%f2;
    \\  ld.shared.f32 %f2,[%r14+5120]; ld.shared.f32 %f3,[%r17+5120]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+5120],%f2;
    \\  ld.shared.f32 %f2,[%r14+6144]; ld.shared.f32 %f3,[%r17+6144]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+6144],%f2;
    \\  ld.shared.f32 %f2,[%r14+7168]; ld.shared.f32 %f3,[%r17+7168]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+7168],%f2;
    \\  ld.shared.f32 %f2,[%r14+8192]; ld.shared.f32 %f3,[%r17+8192]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+8192],%f2;
    \\  ld.shared.f32 %f2,[%r14+9216]; ld.shared.f32 %f3,[%r17+9216]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+9216],%f2;
    \\  ld.shared.f32 %f2,[%r14+10240]; ld.shared.f32 %f3,[%r17+10240]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+10240],%f2;
    \\  ld.shared.f32 %f2,[%r14+11264]; ld.shared.f32 %f3,[%r17+11264]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+11264],%f2;
    \\  ld.shared.f32 %f2,[%r14+12288]; ld.shared.f32 %f3,[%r17+12288]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+12288],%f2;
    \\  ld.shared.f32 %f2,[%r14+13312]; ld.shared.f32 %f3,[%r17+13312]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+13312],%f2;
    \\  ld.shared.f32 %f2,[%r14+14336]; ld.shared.f32 %f3,[%r17+14336]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+14336],%f2;
    \\  ld.shared.f32 %f2,[%r14+15360]; ld.shared.f32 %f3,[%r17+15360]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+15360],%f2;
    \\  ld.shared.f32 %f2,[%r14+16384]; ld.shared.f32 %f3,[%r17+16384]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+16384],%f2;
    \\  ld.shared.f32 %f2,[%r14+17408]; ld.shared.f32 %f3,[%r17+17408]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+17408],%f2;
    \\  ld.shared.f32 %f2,[%r14+18432]; ld.shared.f32 %f3,[%r17+18432]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+18432],%f2;
    \\  ld.shared.f32 %f2,[%r14+19456]; ld.shared.f32 %f3,[%r17+19456]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+19456],%f2;
    \\  ld.shared.f32 %f2,[%r14+20480]; ld.shared.f32 %f3,[%r17+20480]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+20480],%f2;
    \\  ld.shared.f32 %f2,[%r14+21504]; ld.shared.f32 %f3,[%r17+21504]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+21504],%f2;
    \\  ld.shared.f32 %f2,[%r14+22528]; ld.shared.f32 %f3,[%r17+22528]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+22528],%f2;
    \\  ld.shared.f32 %f2,[%r14+23552]; ld.shared.f32 %f3,[%r17+23552]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+23552],%f2;
    \\  ld.shared.f32 %f2,[%r14+24576]; ld.shared.f32 %f3,[%r17+24576]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+24576],%f2;
    \\  ld.shared.f32 %f2,[%r14+25600]; ld.shared.f32 %f3,[%r17+25600]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+25600],%f2;
    \\  ld.shared.f32 %f2,[%r14+26624]; ld.shared.f32 %f3,[%r17+26624]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+26624],%f2;
    \\  ld.shared.f32 %f2,[%r14+27648]; ld.shared.f32 %f3,[%r17+27648]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+27648],%f2;
    \\  ld.shared.f32 %f2,[%r14+28672]; ld.shared.f32 %f3,[%r17+28672]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+28672],%f2;
    \\  ld.shared.f32 %f2,[%r14+29696]; ld.shared.f32 %f3,[%r17+29696]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+29696],%f2;
    \\  ld.shared.f32 %f2,[%r14+30720]; ld.shared.f32 %f3,[%r17+30720]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+30720],%f2;
    \\  ld.shared.f32 %f2,[%r14+31744]; ld.shared.f32 %f3,[%r17+31744]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+31744],%f2;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  // y[i*rows + r0 + j] = scale * red[j*4+i][0], stores predicated on i < n
    \\  mul.wide.u32 %rd13,%r35,4; add.s64 %rd14,%rd3,%rd13;
    \\  mul.wide.u32 %rd15,%r2,4;
    \\  setp.lt.u32 %p6,1,%r20; setp.lt.u32 %p7,2,%r20; setp.lt.u32 %p8,3,%r20;
    \\  ld.shared.f32 %f2,[%r13+0]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+4096]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+8192]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+12288]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+16384]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+20480]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+24576]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+28672]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+1024]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+5120]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+9216]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+13312]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+17408]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+21504]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+25600]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+29696]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+2048]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+6144]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+10240]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+14336]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+18432]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+22528]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+26624]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+30720]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+3072]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+7168]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+11264]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+15360]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+19456]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+23552]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+27648]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+31744]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+28],%f2;
    \\END:
    \\  ret;
    \\}
;

/// Multi-input bf16 GEMV (speculative-decode LM head): y[i][row] =
/// scale * dot(W[row], x_i) for n <= 4 input vectors. One block per EIGHT
/// weight rows (rows % 8 == 0) so the four x rows are re-read from L2 once
/// per 8 rows instead of once per row (at vocab-size rows that is GBs of L2
/// traffic otherwise). Per-thread element order matches gemv_bf16 (c =
/// tid*2, stride 512) and each accumulator sums in that order, so results
/// are bitwise identical to the single-input kernel. x must have 4 rows of
/// backing store; W streams with .cs. 32 accumulators (8 rows x 4 inputs)
/// reduce through 32 shared arrays.
/// b0=W, b1=x [4][cols], b2=y [n][rows]. u0=rows, u1=cols, u2=n, f0=scale.
pub const gemv_bf16n_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gemv_bf16n(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<12>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<80>;
    \\  .reg .b64 %rd<40>;
    \\  .shared .align 4 .b8 red[32768];
    \\  mov.u32 %r1,%ctaid.x;                  // row group (8 weight rows)
    \\  ld.param.u32 %r2,[u0];                 // rows
    \\  shl.b32 %r35,%r1,3;                    // r0 = group*8
    \\  setp.ge.u32 %p1,%r35,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // cols
    \\  ld.param.u32 %r20,[u2];                // n inputs (1..4)
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shl.b32 %r5,%r3,2;
    \\  // x row base pointers rd20..rd23 = x + i*cols*4
    \\  mul.wide.u32 %rd16,%r4,4;
    \\  mov.u64 %rd20,%rd2;
    \\  add.s64 %rd21,%rd20,%rd16;
    \\  add.s64 %rd22,%rd21,%rd16;
    \\  add.s64 %rd23,%rd22,%rd16;
    \\  // W row base pointers rd24..rd31 = W + (r0+j)*cols*2
    \\  mul.wide.u32 %rd7,%r35,%r4; shl.b64 %rd7,%rd7,1; add.s64 %rd24,%rd1,%rd7;
    \\  cvt.u64.u32 %rd8,%r4; shl.b64 %rd8,%rd8,1;
    \\  add.s64 %rd25,%rd24,%rd8;
    \\  add.s64 %rd26,%rd25,%rd8;
    \\  add.s64 %rd27,%rd26,%rd8;
    \\  add.s64 %rd28,%rd27,%rd8;
    \\  add.s64 %rd29,%rd28,%rd8;
    \\  add.s64 %rd30,%rd29,%rd8;
    \\  add.s64 %rd31,%rd30,%rd8;
    \\  mov.f32 %f44,0f00000000;
    \\  mov.f32 %f45,0f00000000;
    \\  mov.f32 %f46,0f00000000;
    \\  mov.f32 %f47,0f00000000;
    \\  mov.f32 %f48,0f00000000;
    \\  mov.f32 %f49,0f00000000;
    \\  mov.f32 %f50,0f00000000;
    \\  mov.f32 %f51,0f00000000;
    \\  mov.f32 %f52,0f00000000;
    \\  mov.f32 %f53,0f00000000;
    \\  mov.f32 %f54,0f00000000;
    \\  mov.f32 %f55,0f00000000;
    \\  mov.f32 %f56,0f00000000;
    \\  mov.f32 %f57,0f00000000;
    \\  mov.f32 %f58,0f00000000;
    \\  mov.f32 %f59,0f00000000;
    \\  mov.f32 %f60,0f00000000;
    \\  mov.f32 %f61,0f00000000;
    \\  mov.f32 %f62,0f00000000;
    \\  mov.f32 %f63,0f00000000;
    \\  mov.f32 %f64,0f00000000;
    \\  mov.f32 %f65,0f00000000;
    \\  mov.f32 %f66,0f00000000;
    \\  mov.f32 %f67,0f00000000;
    \\  mov.f32 %f68,0f00000000;
    \\  mov.f32 %f69,0f00000000;
    \\  mov.f32 %f70,0f00000000;
    \\  mov.f32 %f71,0f00000000;
    \\  mov.f32 %f72,0f00000000;
    \\  mov.f32 %f73,0f00000000;
    \\  mov.f32 %f74,0f00000000;
    \\  mov.f32 %f75,0f00000000;
    \\  shl.b32 %r8,%r3,1;                     // c = tid*2 (same order as gemv_bf16: bitwise-stable)
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r4; @%p2 bra LD;
    \\  // all 4 inputs' elems c,c+1 loaded once per 8-row group (x L2 amortization)
    \\  mul.wide.u32 %rd11,%r8,4;
    \\  add.s64 %rd12,%rd20,%rd11; ld.global.v2.f32 {%f2,%f3},[%rd12];
    \\  add.s64 %rd13,%rd21,%rd11; ld.global.v2.f32 {%f4,%f5},[%rd13];
    \\  add.s64 %rd14,%rd22,%rd11; ld.global.v2.f32 {%f6,%f7},[%rd14];
    \\  add.s64 %rd15,%rd23,%rd11; ld.global.v2.f32 {%f8,%f9},[%rd15];
    \\  mul.wide.u32 %rd9,%r8,2;
    \\  add.s64 %rd10,%rd24,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 0: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f44,%f10,%f2,%f44; fma.rn.f32 %f45,%f10,%f4,%f45; fma.rn.f32 %f46,%f10,%f6,%f46; fma.rn.f32 %f47,%f10,%f8,%f47;
    \\  fma.rn.f32 %f44,%f11,%f3,%f44; fma.rn.f32 %f45,%f11,%f5,%f45; fma.rn.f32 %f46,%f11,%f7,%f46; fma.rn.f32 %f47,%f11,%f9,%f47;
    \\  add.s64 %rd10,%rd25,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 1: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f48,%f10,%f2,%f48; fma.rn.f32 %f49,%f10,%f4,%f49; fma.rn.f32 %f50,%f10,%f6,%f50; fma.rn.f32 %f51,%f10,%f8,%f51;
    \\  fma.rn.f32 %f48,%f11,%f3,%f48; fma.rn.f32 %f49,%f11,%f5,%f49; fma.rn.f32 %f50,%f11,%f7,%f50; fma.rn.f32 %f51,%f11,%f9,%f51;
    \\  add.s64 %rd10,%rd26,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 2: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f52,%f10,%f2,%f52; fma.rn.f32 %f53,%f10,%f4,%f53; fma.rn.f32 %f54,%f10,%f6,%f54; fma.rn.f32 %f55,%f10,%f8,%f55;
    \\  fma.rn.f32 %f52,%f11,%f3,%f52; fma.rn.f32 %f53,%f11,%f5,%f53; fma.rn.f32 %f54,%f11,%f7,%f54; fma.rn.f32 %f55,%f11,%f9,%f55;
    \\  add.s64 %rd10,%rd27,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 3: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f56,%f10,%f2,%f56; fma.rn.f32 %f57,%f10,%f4,%f57; fma.rn.f32 %f58,%f10,%f6,%f58; fma.rn.f32 %f59,%f10,%f8,%f59;
    \\  fma.rn.f32 %f56,%f11,%f3,%f56; fma.rn.f32 %f57,%f11,%f5,%f57; fma.rn.f32 %f58,%f11,%f7,%f58; fma.rn.f32 %f59,%f11,%f9,%f59;
    \\  add.s64 %rd10,%rd28,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 4: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f60,%f10,%f2,%f60; fma.rn.f32 %f61,%f10,%f4,%f61; fma.rn.f32 %f62,%f10,%f6,%f62; fma.rn.f32 %f63,%f10,%f8,%f63;
    \\  fma.rn.f32 %f60,%f11,%f3,%f60; fma.rn.f32 %f61,%f11,%f5,%f61; fma.rn.f32 %f62,%f11,%f7,%f62; fma.rn.f32 %f63,%f11,%f9,%f63;
    \\  add.s64 %rd10,%rd29,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 5: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f64,%f10,%f2,%f64; fma.rn.f32 %f65,%f10,%f4,%f65; fma.rn.f32 %f66,%f10,%f6,%f66; fma.rn.f32 %f67,%f10,%f8,%f67;
    \\  fma.rn.f32 %f64,%f11,%f3,%f64; fma.rn.f32 %f65,%f11,%f5,%f65; fma.rn.f32 %f66,%f11,%f7,%f66; fma.rn.f32 %f67,%f11,%f9,%f67;
    \\  add.s64 %rd10,%rd30,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 6: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f68,%f10,%f2,%f68; fma.rn.f32 %f69,%f10,%f4,%f69; fma.rn.f32 %f70,%f10,%f6,%f70; fma.rn.f32 %f71,%f10,%f8,%f71;
    \\  fma.rn.f32 %f68,%f11,%f3,%f68; fma.rn.f32 %f69,%f11,%f5,%f69; fma.rn.f32 %f70,%f11,%f7,%f70; fma.rn.f32 %f71,%f11,%f9,%f71;
    \\  add.s64 %rd10,%rd31,%rd9; ld.global.cs.u32 %r9,[%rd10]; // row 7: 2 bf16
    \\  shl.b32 %r10,%r9,16; mov.b32 %f10,%r10; and.b32 %r10,%r9,0xffff0000; mov.b32 %f11,%r10;
    \\  fma.rn.f32 %f72,%f10,%f2,%f72; fma.rn.f32 %f73,%f10,%f4,%f73; fma.rn.f32 %f74,%f10,%f6,%f74; fma.rn.f32 %f75,%f10,%f8,%f75;
    \\  fma.rn.f32 %f72,%f11,%f3,%f72; fma.rn.f32 %f73,%f11,%f5,%f73; fma.rn.f32 %f74,%f11,%f7,%f74; fma.rn.f32 %f75,%f11,%f9,%f75;
    \\  add.u32 %r8,%r8,512; bra LOOP;
    \\LD:
    \\  // red is 32 arrays of 256 f32: &red[a][tid] = red + a*1024 + tid*4, a = j*4 + i
    \\  mov.u32 %r13,red; add.u32 %r14,%r13,%r5;
    \\  st.shared.f32 [%r14+0],%f44;
    \\  st.shared.f32 [%r14+1024],%f45;
    \\  st.shared.f32 [%r14+2048],%f46;
    \\  st.shared.f32 [%r14+3072],%f47;
    \\  st.shared.f32 [%r14+4096],%f48;
    \\  st.shared.f32 [%r14+5120],%f49;
    \\  st.shared.f32 [%r14+6144],%f50;
    \\  st.shared.f32 [%r14+7168],%f51;
    \\  st.shared.f32 [%r14+8192],%f52;
    \\  st.shared.f32 [%r14+9216],%f53;
    \\  st.shared.f32 [%r14+10240],%f54;
    \\  st.shared.f32 [%r14+11264],%f55;
    \\  st.shared.f32 [%r14+12288],%f56;
    \\  st.shared.f32 [%r14+13312],%f57;
    \\  st.shared.f32 [%r14+14336],%f58;
    \\  st.shared.f32 [%r14+15360],%f59;
    \\  st.shared.f32 [%r14+16384],%f60;
    \\  st.shared.f32 [%r14+17408],%f61;
    \\  st.shared.f32 [%r14+18432],%f62;
    \\  st.shared.f32 [%r14+19456],%f63;
    \\  st.shared.f32 [%r14+20480],%f64;
    \\  st.shared.f32 [%r14+21504],%f65;
    \\  st.shared.f32 [%r14+22528],%f66;
    \\  st.shared.f32 [%r14+23552],%f67;
    \\  st.shared.f32 [%r14+24576],%f68;
    \\  st.shared.f32 [%r14+25600],%f69;
    \\  st.shared.f32 [%r14+26624],%f70;
    \\  st.shared.f32 [%r14+27648],%f71;
    \\  st.shared.f32 [%r14+28672],%f72;
    \\  st.shared.f32 [%r14+29696],%f73;
    \\  st.shared.f32 [%r14+30720],%f74;
    \\  st.shared.f32 [%r14+31744],%f75;
    \\  bar.sync 0;
    \\  mov.u32 %r15,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r15,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r15; @%p4 bra REDS;
    \\  shl.b32 %r16,%r15,2; add.u32 %r17,%r14,%r16;
    \\  ld.shared.f32 %f2,[%r14+0]; ld.shared.f32 %f3,[%r17+0]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+0],%f2;
    \\  ld.shared.f32 %f2,[%r14+1024]; ld.shared.f32 %f3,[%r17+1024]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+1024],%f2;
    \\  ld.shared.f32 %f2,[%r14+2048]; ld.shared.f32 %f3,[%r17+2048]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+2048],%f2;
    \\  ld.shared.f32 %f2,[%r14+3072]; ld.shared.f32 %f3,[%r17+3072]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+3072],%f2;
    \\  ld.shared.f32 %f2,[%r14+4096]; ld.shared.f32 %f3,[%r17+4096]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+4096],%f2;
    \\  ld.shared.f32 %f2,[%r14+5120]; ld.shared.f32 %f3,[%r17+5120]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+5120],%f2;
    \\  ld.shared.f32 %f2,[%r14+6144]; ld.shared.f32 %f3,[%r17+6144]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+6144],%f2;
    \\  ld.shared.f32 %f2,[%r14+7168]; ld.shared.f32 %f3,[%r17+7168]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+7168],%f2;
    \\  ld.shared.f32 %f2,[%r14+8192]; ld.shared.f32 %f3,[%r17+8192]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+8192],%f2;
    \\  ld.shared.f32 %f2,[%r14+9216]; ld.shared.f32 %f3,[%r17+9216]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+9216],%f2;
    \\  ld.shared.f32 %f2,[%r14+10240]; ld.shared.f32 %f3,[%r17+10240]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+10240],%f2;
    \\  ld.shared.f32 %f2,[%r14+11264]; ld.shared.f32 %f3,[%r17+11264]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+11264],%f2;
    \\  ld.shared.f32 %f2,[%r14+12288]; ld.shared.f32 %f3,[%r17+12288]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+12288],%f2;
    \\  ld.shared.f32 %f2,[%r14+13312]; ld.shared.f32 %f3,[%r17+13312]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+13312],%f2;
    \\  ld.shared.f32 %f2,[%r14+14336]; ld.shared.f32 %f3,[%r17+14336]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+14336],%f2;
    \\  ld.shared.f32 %f2,[%r14+15360]; ld.shared.f32 %f3,[%r17+15360]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+15360],%f2;
    \\  ld.shared.f32 %f2,[%r14+16384]; ld.shared.f32 %f3,[%r17+16384]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+16384],%f2;
    \\  ld.shared.f32 %f2,[%r14+17408]; ld.shared.f32 %f3,[%r17+17408]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+17408],%f2;
    \\  ld.shared.f32 %f2,[%r14+18432]; ld.shared.f32 %f3,[%r17+18432]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+18432],%f2;
    \\  ld.shared.f32 %f2,[%r14+19456]; ld.shared.f32 %f3,[%r17+19456]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+19456],%f2;
    \\  ld.shared.f32 %f2,[%r14+20480]; ld.shared.f32 %f3,[%r17+20480]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+20480],%f2;
    \\  ld.shared.f32 %f2,[%r14+21504]; ld.shared.f32 %f3,[%r17+21504]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+21504],%f2;
    \\  ld.shared.f32 %f2,[%r14+22528]; ld.shared.f32 %f3,[%r17+22528]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+22528],%f2;
    \\  ld.shared.f32 %f2,[%r14+23552]; ld.shared.f32 %f3,[%r17+23552]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+23552],%f2;
    \\  ld.shared.f32 %f2,[%r14+24576]; ld.shared.f32 %f3,[%r17+24576]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+24576],%f2;
    \\  ld.shared.f32 %f2,[%r14+25600]; ld.shared.f32 %f3,[%r17+25600]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+25600],%f2;
    \\  ld.shared.f32 %f2,[%r14+26624]; ld.shared.f32 %f3,[%r17+26624]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+26624],%f2;
    \\  ld.shared.f32 %f2,[%r14+27648]; ld.shared.f32 %f3,[%r17+27648]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+27648],%f2;
    \\  ld.shared.f32 %f2,[%r14+28672]; ld.shared.f32 %f3,[%r17+28672]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+28672],%f2;
    \\  ld.shared.f32 %f2,[%r14+29696]; ld.shared.f32 %f3,[%r17+29696]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+29696],%f2;
    \\  ld.shared.f32 %f2,[%r14+30720]; ld.shared.f32 %f3,[%r17+30720]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+30720],%f2;
    \\  ld.shared.f32 %f2,[%r14+31744]; ld.shared.f32 %f3,[%r17+31744]; add.f32 %f2,%f2,%f3; st.shared.f32 [%r14+31744],%f2;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r15,%r15,1; bra RED;
    \\REDD:
    \\  setp.ne.u32 %p5,%r3,0; @%p5 bra END;
    \\  // y[i*rows + r0 + j] = scale * red[j*4+i][0], stores predicated on i < n
    \\  mul.wide.u32 %rd13,%r35,4; add.s64 %rd14,%rd3,%rd13;
    \\  mul.wide.u32 %rd15,%r2,4;
    \\  setp.lt.u32 %p6,1,%r20; setp.lt.u32 %p7,2,%r20; setp.lt.u32 %p8,3,%r20;
    \\  ld.shared.f32 %f2,[%r13+0]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+4096]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+8192]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+12288]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+16384]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+20480]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+24576]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+28672]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+1024]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+5120]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+9216]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+13312]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+17408]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+21504]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+25600]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+29696]; mul.f32 %f2,%f2,%f1; @%p6 st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+2048]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+6144]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+10240]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+14336]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+18432]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+22528]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+26624]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+30720]; mul.f32 %f2,%f2,%f1; @%p7 st.global.f32 [%rd14+28],%f2;
    \\  add.s64 %rd14,%rd14,%rd15;
    \\  ld.shared.f32 %f2,[%r13+3072]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+0],%f2;
    \\  ld.shared.f32 %f2,[%r13+7168]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+4],%f2;
    \\  ld.shared.f32 %f2,[%r13+11264]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+8],%f2;
    \\  ld.shared.f32 %f2,[%r13+15360]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+12],%f2;
    \\  ld.shared.f32 %f2,[%r13+19456]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+16],%f2;
    \\  ld.shared.f32 %f2,[%r13+23552]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+20],%f2;
    \\  ld.shared.f32 %f2,[%r13+27648]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+24],%f2;
    \\  ld.shared.f32 %f2,[%r13+31744]; mul.f32 %f2,%f2,%f1; @%p8 st.global.f32 [%rd14+28],%f2;
    \\END:
    \\  ret;
    \\}
;

/// Flash-decoding pass 1: split each query's KV range across nsplit chunks,
/// one WARP per (query, head, split). Queries are consecutive positions with
/// causal attention: query t sees kv_len0 + t keys (kv_len0 = pos0 + 1), so
/// seq_q == 1 is plain decode and seq_q > 1 is the speculative-verify batch.
/// Requires hd == 128: each lane owns 4 dims (q/k/v as v4.f32), the k·q dot
/// closes with a shfl.bfly tree (all lanes get the sum), softmax scalars are
/// computed redundantly per lane, and the accumulator lives in 4 registers
/// per lane — no local memory. Partial (m, d, pad, pad, acc[hd]) rows go to
/// scratch at row `warp` — [t][h][split] order — stride hd+4 so the lane v4
/// stores stay 16B-aligned (attn_merge then runs with heads' = seq_q*heads).
/// b0=q[seq_q][heads][hd], b1=k[seq_kv][kv][hd], b2=v, b3=scratch.
/// u0=kv_len0, u1=heads, u2=kv_heads, u3=hd(=128), u4=nsplit, u5=seq_q, f0=scale.
pub const attn_split_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn_split(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<32>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.param.u32 %r5,[u0];                // kv_len0
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal: kv_len0 + t)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  // Sliding window (f1, 0 = full causal): kv_start = max(0, kv_len - window).
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment: q[(t*heads + h)*hd + lane*4 ..][4]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  // kv row fragment base: ((j*kv_heads + kvh)*hd + lane*4)
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,4; add.s64 %rd10,%rd2,%rd9;
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd10];
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  // butterfly all-reduce: every lane ends with the full dot
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V fragment (same offsets)
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd11];
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  // scratch row = warp*(hd+4): lane 0 stores m,d; every lane its acc4
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
;

/// f16-KV variant of `attn_split` (hd128, qwen3 prefill): each 4-dim K/V
/// fragment loaded as v2.u32 (8 B = 4 halfs) and widened to f32; *2 stride.
/// q/scratch/out stay f32. Otherwise identical to attn_split. Lossy vs f32.
pub const attn_split_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn_split_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<34>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.param.u32 %r5,[u0];                // kv_len0
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal: kv_len0 + t)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  // Sliding window (f1, 0 = full causal): kv_start = max(0, kv_len - window).
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[(t*heads + h)*hd + lane*4 ..][4]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,2; add.s64 %rd10,%rd2,%rd9;   // f16 K (*2)
    \\  ld.global.v2.u32 {%r32,%r33},[%rd10];                // 4 halfs
    \\  mov.b32 {%hs0,%hs1},%r32; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r33; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // f16 V (same offsets)
    \\  ld.global.v2.u32 {%r32,%r33},[%rd11];
    \\  mov.b32 {%hs0,%hs1},%r32; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r33; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
;

/// q8_0-KV variant of `attn_split` (hd128, qwen3): each 4-dim K/V fragment
/// lives inside one 34-byte ggml block (f16 scale d + 32 x i8). The lane loads
/// its block's scale, then its 4 quants as two u16s (2-byte aligned), sign-
/// extends and multiplies by d. q/scratch/out stay f32. Lossy vs f32.
pub const attn_split_q8_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn_split_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.param.u32 %r5,[u0];                // kv_len0
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal: kv_len0 + t)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  // Sliding window (f1, 0 = full causal): kv_start = max(0, kv_len - window).
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[(t*heads + h)*hd + lane*4 ..][4]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  // Loop-invariant quant byte offset within the lane's block (+2 header).
    \\  and.b32 %r34,%r15,31; add.u32 %r34,%r34,2; cvt.u64.u32 %rd21,%r34;
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  shr.u32 %r35,%r18,5; mul.wide.u32 %rd9,%r35,34; add.s64 %rd10,%rd2,%rd9;   // K block
    \\  ld.global.b16 %hs0,[%rd10]; cvt.f32.f16 %f31,%hs0;   // block scale d
    \\  add.s64 %rd12,%rd10,%rd21;
    \\  ld.global.u16 %r32,[%rd12];
    \\  shl.b32 %r36,%r32,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f24,%r36; mul.f32 %f24,%f24,%f31;
    \\  shl.b32 %r36,%r32,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f25,%r36; mul.f32 %f25,%f25,%f31;
    \\  ld.global.u16 %r33,[%rd12+2];
    \\  shl.b32 %r36,%r33,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f26,%r36; mul.f32 %f26,%f26,%f31;
    \\  shl.b32 %r36,%r33,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f27,%r36; mul.f32 %f27,%f27,%f31;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V block (same offsets)
    \\  ld.global.b16 %hs0,[%rd11]; cvt.f32.f16 %f31,%hs0;
    \\  add.s64 %rd12,%rd11,%rd21;
    \\  ld.global.u16 %r32,[%rd12];
    \\  shl.b32 %r36,%r32,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f24,%r36; mul.f32 %f24,%f24,%f31;
    \\  shl.b32 %r36,%r32,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f25,%r36; mul.f32 %f25,%f25,%f31;
    \\  ld.global.u16 %r33,[%rd12+2];
    \\  shl.b32 %r36,%r33,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f26,%r36; mul.f32 %f26,%f26,%f31;
    \\  shl.b32 %r36,%r33,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f27,%r36; mul.f32 %f27,%f27,%f31;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
;

/// Flash-decoding pass 2: merge the nsplit partials of each head. One thread
/// per (head, dim c): M = max_i m_i, D = sum_i d_i*exp(m_i-M),
/// out[h][c] = sum_i acc_i[c]*exp(m_i-M) / D.
/// b0=scratch[heads*nsplit][hd+4] (see attn_split), b1=out[heads][hd].
/// u0=heads, u1=hd, u2=nsplit.
pub const attn_merge_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn_merge(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<16>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // idx
    \\  ld.param.u32 %r5,[u0];                // heads
    \\  ld.param.u32 %r6,[u1];                // hd
    \\  mul.lo.s32 %r7,%r5,%r6;
    \\  setp.ge.u32 %p1,%r4,%r7; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // nsplit
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  div.u32 %r9,%r4,%r6;                  // h
    \\  rem.u32 %r10,%r4,%r6;                 // c
    \\  add.u32 %r11,%r6,4;                   // stride = hd+4
    \\  mul.lo.s32 %r12,%r9,%r8; mul.lo.s32 %r12,%r12,%r11; // h*nsplit*(hd+4)
    \\  mul.wide.u32 %rd3,%r12,4; add.s64 %rd4,%rd1,%rd3;   // partial base
    \\  // pass 1: M = max m_i
    \\  mov.f32 %f1,0fFF800000; mov.u32 %r13,0; mov.b64 %rd5,%rd4;
    \\M1:
    \\  setp.ge.u32 %p2,%r13,%r8; @%p2 bra M1D;
    \\  ld.global.f32 %f2,[%rd5]; max.f32 %f1,%f1,%f2;
    \\  mul.wide.u32 %rd6,%r11,4; add.s64 %rd5,%rd5,%rd6; add.u32 %r13,%r13,1; bra M1;
    \\M1D:
    \\  // pass 2: D and O
    \\  mov.f32 %f3,0f00000000; mov.f32 %f4,0f00000000; mov.u32 %r13,0; mov.b64 %rd5,%rd4;
    \\  add.u32 %r14,%r10,4;                  // acc elem offset = 4+c
    \\M2:
    \\  setp.ge.u32 %p2,%r13,%r8; @%p2 bra M2D;
    \\  ld.global.f32 %f5,[%rd5]; ld.global.f32 %f6,[%rd5+4];
    \\  sub.f32 %f7,%f5,%f1; mul.f32 %f7,%f7,0f3FB8AA3B; ex2.approx.f32 %f7,%f7; // w = exp(m_i - M)
    \\  fma.rn.f32 %f3,%f6,%f7,%f3;           // D += d_i*w
    \\  mul.wide.u32 %rd7,%r14,4; add.s64 %rd8,%rd5,%rd7; ld.global.f32 %f8,[%rd8];
    \\  fma.rn.f32 %f4,%f8,%f7,%f4;           // O += acc_i[c]*w
    \\  mul.wide.u32 %rd6,%r11,4; add.s64 %rd5,%rd5,%rd6; add.u32 %r13,%r13,1; bra M2;
    \\M2D:
    \\  rcp.approx.f32 %f9,%f3; mul.f32 %f4,%f4,%f9;
    \\  mul.wide.u32 %rd9,%r4,4; add.s64 %rd10,%rd2,%rd9; st.global.f32 [%rd10],%f4;
    \\END:
    \\  ret;
    \\}
;

/// Tree-verify attn_split (speculative tree drafting, LLM_PLAN.md M8):
/// seq_q tree-node queries, query t attending kv rows [0, L) of the linear
/// cache plus its ancestor chain, whose K/V live at rows tree_base+idx of
/// the SAME k/v buffers. Per-query kv lengths and ancestor row lists come
/// from a meta table at the scratch tail (element offset
/// seq_q*heads*nsplit*(hd+4), row stride seq_q+1, row t = [kv_len_t,
/// anc_0, anc_1, ...] u32, ancestors in depth order): kv index j maps to
/// row j when j < L, else tree_base + anc[j - L]. Chunking, math, and
/// iteration order are identical to attn_split at the same kv_len, so
/// merged outputs stay bitwise-identical to plain decode. hd is hardcoded
/// 128 (as attn_split requires); its param slot carries tree_base.
/// b0=q[seq_q][heads][128], b1=k, b2=v, b3=scratch(+meta).
/// u0=L, u1=heads, u2=kv_heads, u3=tree_base, u4=nsplit, u5=seq_q, f0=scale.
pub const attn_split_tree_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry attn_split_tree(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b64 %rd<26>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.param.u32 %r36,[u0];               // L (committed prefix len)
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r35,[u3];               // tree_base (batch K/V row offset)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  // meta row t: scratch elem off = seq_q*heads*nsplit*(hd+4) + t*(seq_q+1)
    \\  mul.lo.s32 %r32,%r7,%r30;
    \\  mul.lo.s32 %r32,%r32,132;             // partial region elems (hd+4 = 132)
    \\  add.u32 %r33,%r30,1;                  // meta row stride = seq_q+1
    \\  mad.lo.s32 %r34,%r31,%r33,%r32;
    \\  mul.wide.u32 %rd21,%r34,4; add.s64 %rd22,%rd4,%rd21;
    \\  ld.global.u32 %r5,[%rd22];            // this query's kv len (L + depth + 1)
    \\  add.s64 %rd23,%rd22,4;                // ancestor row list base
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  // Sliding window (f1, 0 = full causal): kv_start = max(0, kv_len - window).
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment: q[(t*heads + h)*hd + lane*4 ..][4]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,128; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  // kv row: j < L -> j (cache), else tree_base + anc[j - L] (batch row)
    \\  mov.u32 %r37,%r17;
    \\  setp.lt.u32 %p4,%r17,%r36; @%p4 bra HAVEROW;
    \\  sub.u32 %r38,%r17,%r36;
    \\  mul.wide.u32 %rd19,%r38,4; add.s64 %rd20,%rd23,%rd19; ld.global.u32 %r37,[%rd20];
    \\  add.u32 %r37,%r37,%r35;
    \\HAVEROW:
    \\  mad.lo.s32 %r18,%r37,%r8,%r13; mul.lo.s32 %r18,%r18,128; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,4; add.s64 %rd10,%rd2,%rd9;
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd10];
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  // butterfly all-reduce: every lane ends with the full dot
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V fragment (same offsets)
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd11];
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  // scratch row = warp*(hd+4): lane 0 stores m,d; every lane its acc4
    \\  mul.lo.s32 %r25,%r27,132;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
;

/// rotate-half RoPE, in place, one thread per (position, head, pair): the head
/// vector splits into halves [0:half] and [half:2*half]; for pair i,
/// lo' = lo*cos[i] - hi*sin[i], hi' = hi*cos[i] + lo*sin[i]. cos/sin are
/// [seq][half] with sin offset u2. b0=qk(f32), b2=freqs(f32). u0=total
/// (=seq*n_heads*half), u1=half, u2=sin_off, u3=n_heads.
pub const rope_half_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_half(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair = idx % half
    \\  div.u32 %r10,%r4,%r6;                 // hp = idx/half = pos*n_heads + head
    \\  mul.lo.s32 %r11,%r6,%r8;              // half*n_heads
    \\  div.u32 %r12,%r4,%r11;                // pos
    \\  ld.param.u32 %r18,[u4]; add.u32 %r12,%r12,%r18; // pos += pos0 (u4): cached decode offset
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;         // cos idx = pos*half + pair
    \\  mul.wide.u32 %rd4,%r13,4; add.s64 %rd5,%rd3,%rd4; ld.global.f32 %f1,[%rd5]; // cos
    \\  add.s32 %r14,%r13,%r7;                // + sin_off
    \\  mul.wide.u32 %rd6,%r14,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f2,[%rd7]; // sin
    \\  shl.b32 %r15,%r6,1;                   // head_dim = 2*half
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;        // lo_idx = hp*head_dim + pair
    \\  add.s32 %r17,%r16,%r6;                // hi_idx = lo_idx + half
    \\  mul.wide.u32 %rd8,%r16,4; add.s64 %rd9,%rd1,%rd8; ld.global.f32 %f3,[%rd9];   // lo
    \\  mul.wide.u32 %rd10,%r17,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f4,[%rd11]; // hi
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd9],%f5;  // lo*cos - hi*sin
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd11],%f6;               // hi*cos + lo*sin
    \\END:
    \\  ret;
    \\}
;

/// rope_half with an explicit absolute position per row (tree-verify
/// batches: node positions are depth-based, not consecutive): pos =
/// b1[row] (u32), row = idx / (half*n_heads). Per-element math identical
/// to rope_half at the same position. b0=qk(f32), b1=positions(u32),
/// b2=freqs(f32). u0=total (=rows*n_heads*half), u1=half, u2=sin_off,
/// u3=n_heads.
pub const rope_half_pos_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_half_pos(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair = idx % half
    \\  div.u32 %r10,%r4,%r6;                 // hp = idx/half = row*n_heads + head
    \\  mul.lo.s32 %r11,%r6,%r8;              // half*n_heads
    \\  div.u32 %r12,%r4,%r11;                // row
    \\  mul.wide.u32 %rd12,%r12,4; add.s64 %rd13,%rd2,%rd12; ld.global.u32 %r12,[%rd13]; // pos = positions[row]
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;         // cos idx = pos*half + pair
    \\  mul.wide.u32 %rd4,%r13,4; add.s64 %rd5,%rd3,%rd4; ld.global.f32 %f1,[%rd5]; // cos
    \\  add.s32 %r14,%r13,%r7;                // + sin_off
    \\  mul.wide.u32 %rd6,%r14,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f2,[%rd7]; // sin
    \\  shl.b32 %r15,%r6,1;                   // head_dim = 2*half
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;        // lo_idx = hp*head_dim + pair
    \\  add.s32 %r17,%r16,%r6;                // hi_idx = lo_idx + half
    \\  mul.wide.u32 %rd8,%r16,4; add.s64 %rd9,%rd1,%rd8; ld.global.f32 %f3,[%rd9];   // lo
    \\  mul.wide.u32 %rd10,%r17,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f4,[%rd11]; // hi
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd9],%f5;  // lo*cos - hi*sin
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd11],%f6;               // hi*cos + lo*sin
    \\END:
    \\  ret;
    \\}
;

/// rope_half with an explicit per-head stride (partial RoPE, qwen35: rotate
/// the first 2*half dims of head_dim-wide heads; the rest pass through).
/// Same math as rope_half; u5 = head_dim replaces the implicit 2*half.
/// b0=qk(f32), b2=freqs. u0=total(=seq*n_heads*half), u1=half, u2=sin_off,
/// u3=n_heads, u4=pos0, u5=head_dim.
pub const rope_half_part_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_half_part(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half (of the ROTATED span)
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair
    \\  div.u32 %r10,%r4,%r6;                 // hp = pos*n_heads + head
    \\  mul.lo.s32 %r11,%r6,%r8;
    \\  div.u32 %r12,%r4,%r11;                // pos
    \\  ld.param.u32 %r18,[u4]; add.u32 %r12,%r12,%r18; // pos += pos0
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;         // cos idx
    \\  mul.wide.u32 %rd4,%r13,4; add.s64 %rd5,%rd3,%rd4; ld.global.f32 %f1,[%rd5];
    \\  add.s32 %r14,%r13,%r7;
    \\  mul.wide.u32 %rd6,%r14,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f2,[%rd7];
    \\  ld.param.u32 %r15,[u5];               // head_dim (per-head stride)
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;        // lo_idx = hp*head_dim + pair
    \\  add.s32 %r17,%r16,%r6;                // hi_idx = lo_idx + half
    \\  mul.wide.u32 %rd8,%r16,4; add.s64 %rd9,%rd1,%rd8; ld.global.f32 %f3,[%rd9];
    \\  mul.wide.u32 %rd10,%r17,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f4,[%rd11];
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd9],%f5;
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd11],%f6;
    \\END:
    \\  ret;
    \\}
;

/// Interleaved M-RoPE (qwen35 with images): like rope_half_part but the
/// position for pair p comes from one of three channels (t, h, w) selected
/// by ggml's imrope round-robin — p%3==1 and p<3*s1 -> h, p%3==2 and
/// p<3*s2 -> w, p%3==0 and p<3*s0 -> t, else t. Frequencies stay tied to
/// the global pair index, so equal positions reproduce rope_half_part
/// exactly. Single-row (seq=1) decode stepping.
/// b0=qk, b1=pos3 (u32[3]: t,h,w), b2=freqs. u0=total(=n_heads*half),
/// u1=half, u2=sin_off, u3=n_heads, u4=sections s0|s1<<8|s2<<16, u5=head_dim.
pub const rope_imrope_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_imrope(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<28>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair p (sector)
    \\  div.u32 %r10,%r4,%r6;                 // head
    \\  // channel select: default t (pos3[0])
    \\  ld.param.u32 %r19,[u4];
    \\  and.b32 %r20,%r19,255;                // s0
    \\  shr.u32 %r21,%r19,8; and.b32 %r21,%r21,255;   // s1
    \\  shr.u32 %r22,%r19,16; and.b32 %r22,%r22,255;  // s2
    \\  rem.u32 %r23,%r9,3;                   // p % 3
    \\  mov.u32 %r24,0;                       // channel index
    \\  setp.ne.u32 %p2,%r23,1; @%p2 bra CH2;
    \\  mul.lo.s32 %r25,%r21,3; setp.ge.u32 %p3,%r9,%r25; @%p3 bra CHT;
    \\  mov.u32 %r24,1; bra CHT;
    \\CH2:
    \\  setp.ne.u32 %p4,%r23,2; @%p4 bra CHT;
    \\  mul.lo.s32 %r25,%r22,3; setp.ge.u32 %p5,%r9,%r25; @%p5 bra CHT;
    \\  mov.u32 %r24,2;
    \\CHT:
    \\  mul.wide.u32 %rd4,%r24,4; add.s64 %rd5,%rd2,%rd4; ld.global.u32 %r12,[%rd5]; // pos
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;         // cos idx = pos*half + p
    \\  mul.wide.u32 %rd6,%r13,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f1,[%rd7];
    \\  add.s32 %r14,%r13,%r7;
    \\  mul.wide.u32 %rd8,%r14,4; add.s64 %rd9,%rd3,%rd8; ld.global.f32 %f2,[%rd9];
    \\  ld.param.u32 %r15,[u5];               // head_dim
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;        // lo idx
    \\  add.s32 %r17,%r16,%r6;
    \\  mul.wide.u32 %rd10,%r16,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f3,[%rd11];
    \\  mul.wide.u32 %rd12,%r17,4; add.s64 %rd13,%rd1,%rd12; ld.global.f32 %f4,[%rd13];
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd11],%f5;
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd13],%f6;
    \\END:
    \\  ret;
    \\}
;

/// rope_imrope with per-ROW position triples (batched qwen35 prefill over
/// mixed text/image rows): pos3s is [rows][3] u32; row = idx/(n_heads*half).
/// b0=qk, b1=pos3s, b2=freqs. u0=total(=rows*n_heads*half), u1=half,
/// u2=sin_off, u3=n_heads, u4=sections s0|s1<<8|s2<<16, u5=head_dim.
pub const rope_imrope_pos_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_imrope_pos(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<32>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair p
    \\  div.u32 %r10,%r4,%r6;                 // hp = row*n_heads + head
    \\  mul.lo.s32 %r11,%r6,%r8;
    \\  div.u32 %r26,%r4,%r11;                // row
    \\  // channel select (imrope round-robin)
    \\  ld.param.u32 %r19,[u4];
    \\  and.b32 %r20,%r19,255;
    \\  shr.u32 %r21,%r19,8; and.b32 %r21,%r21,255;
    \\  shr.u32 %r22,%r19,16; and.b32 %r22,%r22,255;
    \\  rem.u32 %r23,%r9,3;
    \\  mov.u32 %r24,0;
    \\  setp.ne.u32 %p2,%r23,1; @%p2 bra CH2;
    \\  mul.lo.s32 %r25,%r21,3; setp.ge.u32 %p3,%r9,%r25; @%p3 bra CHT;
    \\  mov.u32 %r24,1; bra CHT;
    \\CH2:
    \\  setp.ne.u32 %p4,%r23,2; @%p4 bra CHT;
    \\  mul.lo.s32 %r25,%r22,3; setp.ge.u32 %p5,%r9,%r25; @%p5 bra CHT;
    \\  mov.u32 %r24,2;
    \\CHT:
    \\  mul.lo.s32 %r27,%r26,3; add.u32 %r27,%r27,%r24;    // pos idx = row*3 + ch
    \\  mul.wide.u32 %rd4,%r27,4; add.s64 %rd5,%rd2,%rd4; ld.global.u32 %r12,[%rd5];
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;
    \\  mul.wide.u32 %rd6,%r13,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f1,[%rd7];
    \\  add.s32 %r14,%r13,%r7;
    \\  mul.wide.u32 %rd8,%r14,4; add.s64 %rd9,%rd3,%rd8; ld.global.f32 %f2,[%rd9];
    \\  ld.param.u32 %r15,[u5];
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;
    \\  add.s32 %r17,%r16,%r6;
    \\  mul.wide.u32 %rd10,%r16,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f3,[%rd11];
    \\  mul.wide.u32 %rd12,%r17,4; add.s64 %rd13,%rd1,%rd12; ld.global.f32 %f4,[%rd13];
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd11],%f5;
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd13],%f6;
    \\END:
    \\  ret;
    \\}
;

/// 2-D vision rope (qwen3vl ViT, llama.cpp GGML_ROPE_TYPE_VISION): rows of
/// [n_heads][head_dim], rotation pairs (p, p + 2*half) for p < 2*half; the
/// first `half` pairs are keyed by the token's patch ROW, the next `half`
/// by its COLUMN, both with frequency index p % half (theta resets at the
/// section boundary). head_dim may exceed 4*half (zero-padded heads: only
/// the first 4*half dims are rotated). b0=qk(f32), b1=pos2 (u32 [rows][2]:
/// row,col), b2=freqs (cos [max_pos][half], sin at +sin_off).
/// u0=total(=rows*n_heads*2*half), u1=half, u2=sin_off, u3=n_heads,
/// u4=head_dim.
pub const rope_vision_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry rope_vision(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<26>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  shl.b32 %r9,%r6,1;                    // n_dims = 2*half (pairs per head)
    \\  rem.u32 %r10,%r4,%r9;                 // pair p
    \\  div.u32 %r11,%r4,%r9;                 // hp = row*n_heads + head
    \\  mul.lo.s32 %r12,%r9,%r8;              // pairs*n_heads
    \\  div.u32 %r13,%r4,%r12;                // row (token)
    \\  // axis select: p < half -> patch row (pos2[t][0]), else column ([1])
    \\  setp.ge.u32 %p2,%r10,%r6;
    \\  selp.b32 %r14,1,0,%p2;                // axis
    \\  sub.u32 %r15,%r10,%r6;
    \\  selp.b32 %r16,%r15,%r10,%p2;          // fi = p % half
    \\  shl.b32 %r17,%r13,1; add.u32 %r17,%r17,%r14;
    \\  mul.wide.u32 %rd4,%r17,4; add.s64 %rd5,%rd2,%rd4; ld.global.u32 %r18,[%rd5]; // pos
    \\  mad.lo.s32 %r19,%r18,%r6,%r16;        // cos idx = pos*half + fi
    \\  mul.wide.u32 %rd6,%r19,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f1,[%rd7]; // cos
    \\  add.s32 %r20,%r19,%r7;
    \\  mul.wide.u32 %rd8,%r20,4; add.s64 %rd9,%rd3,%rd8; ld.global.f32 %f2,[%rd9]; // sin
    \\  ld.param.u32 %r21,[u4];               // head_dim
    \\  mad.lo.s32 %r22,%r11,%r21,%r10;       // lo idx = hp*head_dim + p
    \\  add.s32 %r23,%r22,%r9;                // hi idx = lo + 2*half
    \\  mul.wide.u32 %rd10,%r22,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f3,[%rd11];
    \\  mul.wide.u32 %rd12,%r23,4; add.s64 %rd13,%rd1,%rd12; ld.global.f32 %f4,[%rd13];
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd11],%f5;  // lo*cos - hi*sin
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd13],%f6;               // hi*cos + lo*sin
    \\END:
    \\  ret;
    \\}
;

/// Deinterleave the qwen35 attention q projection: per 2*hd-wide head slot,
/// q[h*hd+d] = qg[h*2*hd + d], gate[h*hd+d] = qg[h*2*hd + hd + d].
/// b0=qg, b1=q, b2=gate. u0=total q elems (n_heads*hd), u1=hd.
pub const deinterleave2_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry deinterleave2(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<14>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // hd
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  div.u32 %r7,%r4,%r6;                  // h
    \\  rem.u32 %r8,%r4,%r6;                  // d
    \\  shl.b32 %r9,%r6,1; mad.lo.s32 %r10,%r7,%r9,%r8;    // qg lo idx
    \\  mul.wide.u32 %rd4,%r10,4; add.s64 %rd5,%rd1,%rd4; ld.global.f32 %f1,[%rd5];
    \\  add.u32 %r11,%r10,%r6;
    \\  mul.wide.u32 %rd6,%r11,4; add.s64 %rd7,%rd1,%rd6; ld.global.f32 %f2,[%rd7];
    \\  mul.wide.u32 %rd8,%r4,4;
    \\  add.s64 %rd9,%rd2,%rd8; st.global.f32 [%rd9],%f1;
    \\  add.s64 %rd10,%rd3,%rd8; st.global.f32 [%rd10],%f2;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] *= sigmoid(b[idx]) — the qwen35 attention output gate.
/// b0=a, b1=b. u0=total.
pub const mul_sigmoid_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry mul_sigmoid(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<6>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd5];
    \\  neg.f32 %f3,%f2; mul.f32 %f3,%f3,0f3FB8AA3B; ex2.approx.f32 %f3,%f3; add.f32 %f3,%f3,0f3F800000; rcp.approx.f32 %f3,%f3;
    \\  mul.f32 %f1,%f1,%f3; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Row-wise L2 normalization (ggml_l2_norm): x_row /= max(||x_row||, eps).
/// One 256-thread block per row (rows of dim <= 256), shared tree reduction.
/// b0=x (in place). u0=rows, u1=dim, f0=eps.
pub const l2norm_rows_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry l2norm_rows(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<18>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b64 %rd<10>;
    \\  .shared .align 4 .b8 red[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // row
    \\  ld.param.u32 %r2,[u0]; setp.ge.u32 %p1,%r1,%r2; @%p1 bra END;
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // dim
    \\  ld.param.f32 %f1,[f0];                 // eps
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mad.lo.s32 %r5,%r1,%r4,%r3;            // elem = row*dim + tid
    \\  mul.wide.u32 %rd2,%r5,4; add.s64 %rd3,%rd1,%rd2;
    \\  mov.f32 %f2,0f00000000;
    \\  setp.ge.u32 %p2,%r3,%r4; @%p2 bra RED0;
    \\  ld.global.f32 %f3,[%rd3]; mul.f32 %f2,%f3,%f3;
    \\RED0:
    \\  shl.b32 %r6,%r3,2; mov.u32 %r7,red; add.u32 %r8,%r7,%r6;
    \\  st.shared.f32 [%r8],%f2; bar.sync 0;
    \\  mov.u32 %r9,128;
    \\RED:
    \\  setp.eq.u32 %p3,%r9,0; @%p3 bra REDD;
    \\  setp.ge.u32 %p4,%r3,%r9; @%p4 bra REDS;
    \\  ld.shared.f32 %f4,[%r8]; shl.b32 %r10,%r9,2; add.u32 %r10,%r8,%r10;
    \\  ld.shared.f32 %f5,[%r10]; add.f32 %f4,%f4,%f5; st.shared.f32 [%r8],%f4;
    \\REDS:
    \\  bar.sync 0; shr.u32 %r9,%r9,1; bra RED;
    \\REDD:
    \\  ld.shared.f32 %f6,[%r7];               // sum of squares
    \\  sqrt.rn.f32 %f6,%f6; max.f32 %f6,%f6,%f1; rcp.rn.f32 %f6,%f6;
    \\  setp.ge.u32 %p5,%r3,%r4; @%p5 bra END;
    \\  ld.global.f32 %f7,[%rd3]; mul.f32 %f7,%f7,%f6; st.global.f32 [%rd3],%f7;
    \\END:
    \\  ret;
    \\}
;

/// One step of the qwen35 depthwise causal conv (kernel 4) + SiLU: per
/// channel, out = silu(w0*s0 + w1*s1 + w2*s2 + w3*x), then the 3-column
/// state rolls forward (s = {s1, s2, x}). One thread per channel.
/// b0=conv_state [channels][3] (in/out), b1=x (new column [channels]),
/// b2=conv_w [channels][4] (w[0] oldest), b3=out. u0=channels.
pub const gdn_conv_step_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gdn_conv_step(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<14>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  mul.lo.s32 %r6,%r4,3; mul.wide.u32 %rd5,%r6,4; add.s64 %rd6,%rd1,%rd5;   // &state[c][0]
    \\  shl.b32 %r7,%r4,2; mul.wide.u32 %rd7,%r7,4; add.s64 %rd8,%rd3,%rd7;      // &w[c][0]
    \\  mul.wide.u32 %rd9,%r4,4; add.s64 %rd10,%rd2,%rd9; ld.global.f32 %f1,[%rd10]; // x
    \\  ld.global.f32 %f2,[%rd6]; ld.global.f32 %f3,[%rd6+4]; ld.global.f32 %f4,[%rd6+8];
    \\  ld.global.f32 %f5,[%rd8]; ld.global.f32 %f6,[%rd8+4]; ld.global.f32 %f7,[%rd8+8]; ld.global.f32 %f8,[%rd8+12];
    \\  mul.f32 %f9,%f8,%f1;
    \\  fma.rn.f32 %f9,%f5,%f2,%f9; fma.rn.f32 %f9,%f6,%f3,%f9; fma.rn.f32 %f9,%f7,%f4,%f9;
    \\  st.global.f32 [%rd6],%f3; st.global.f32 [%rd6+4],%f4; st.global.f32 [%rd6+8],%f1;  // roll
    \\  // silu(acc) = acc * sigmoid(acc)
    \\  neg.f32 %f10,%f9; mul.f32 %f10,%f10,0f3FB8AA3B; ex2.approx.f32 %f10,%f10; add.f32 %f10,%f10,0f3F800000; rcp.approx.f32 %f10,%f10;
    \\  mul.f32 %f9,%f9,%f10;
    \\  add.s64 %rd11,%rd4,%rd9; st.global.f32 [%rd11],%f9;
    \\END:
    \\  ret;
    \\}
;

/// Per-head gated-delta-net gates: decay[h] = exp(a[h] * softplus(alpha[h]
/// + dt[h])), beta_out[h] = sigmoid(beta[h]). alpha/beta arrive as one GEMV
/// output buffer [alpha(heads) | beta(heads)]; a/dt as a const buffer
/// [a(heads) | dt(heads)]; out is [decay(heads) | beta(heads)].
/// b0=alpha_beta, b1=a_dt, b2=out. u0=heads.
pub const gdn_gates_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gdn_gates(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<14>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mul.wide.u32 %rd4,%r4,4;
    \\  mul.wide.u32 %rd5,%r5,4;
    \\  add.s64 %rd6,%rd1,%rd4; ld.global.f32 %f1,[%rd6];            // alpha[h]
    \\  add.s64 %rd7,%rd6,%rd5; ld.global.f32 %f2,[%rd7];            // beta[h]
    \\  add.s64 %rd8,%rd2,%rd4; ld.global.f32 %f3,[%rd8];            // a[h]
    \\  add.s64 %rd9,%rd8,%rd5; ld.global.f32 %f4,[%rd9];            // dt[h]
    \\  add.f32 %f5,%f1,%f4;                                         // alpha + dt
    \\  // softplus(x) = log(1+exp(x)); x > 20 -> x
    \\  setp.gt.f32 %p2,%f5,0f41A00000; @%p2 bra SPBIG;
    \\  mul.f32 %f6,%f5,0f3FB8AA3B; ex2.approx.f32 %f6,%f6; add.f32 %f6,%f6,0f3F800000;
    \\  lg2.approx.f32 %f6,%f6; mul.f32 %f5,%f6,0f3F317218;          // ln2 * log2(1+e^x)
    \\SPBIG:
    \\  mul.f32 %f7,%f3,%f5;                                         // a * softplus
    \\  mul.f32 %f7,%f7,0f3FB8AA3B; ex2.approx.f32 %f7,%f7;          // exp -> decay
    \\  add.s64 %rd10,%rd3,%rd4; st.global.f32 [%rd10],%f7;
    \\  // sigmoid(beta)
    \\  neg.f32 %f8,%f2; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8; add.f32 %f8,%f8,0f3F800000; rcp.approx.f32 %f8,%f8;
    \\  add.s64 %rd11,%rd10,%rd5; st.global.f32 [%rd11],%f8;
    \\END:
    \\  ret;
    \\}
;

/// One decode step of the gated-delta-net recurrence, one 256-thread block
/// per v-head over the [d][d] state (k-dim rows i, v-dim columns j; thread
/// j < d owns column j):
///   S[i,j] *= decay;  m_j = sum_i S[i,j] k_i
///   d_j = (v_j - m_j) * beta
///   S[i,j] += k_i d_j;  o_j = sum_i S[i,j] (q_i * scale)
/// q/k are the L2-normalized conv outputs; v-head h uses k-head h % k_heads.
/// Threads 0..d-1 stage k into shared, d..2d-1 stage q*scale. Both state
/// walks are unrolled x4 (same fma order — bitwise-identical m/o): with
/// only `heads` CTAs of 4 active warps the kernel is load-latency bound,
/// so per-thread memory-level parallelism is the lever. d % 4 == 0.
/// b0=S (all heads, [heads][d][d]), b1=conv_out ([q(k_heads*d) |
/// k(k_heads*d) | v(heads*d)]), b2=gates ([decay(heads) | beta(heads)]),
/// b3=o out [heads*d]. u0=heads, u1=d, u2=k_heads. f0=readout scale.
pub const gdn_delta_step_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gdn_delta_step(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<28>;
    \\  .reg .f32 %f<20>;
    \\  .reg .b64 %rd<28>;
    \\  .shared .align 4 .b8 sk[1024];
    \\  .shared .align 4 .b8 sq[1024];
    \\  mov.u32 %r1,%ctaid.x;                  // h
    \\  mov.u32 %r3,%tid.x;
    \\  ld.param.u32 %r4,[u1];                 // d
    \\  ld.param.u32 %r5,[u2];                 // k_heads
    \\  ld.param.f32 %f1,[f0];                 // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  rem.u32 %r6,%r1,%r5;                   // kh = h % k_heads
    \\  mul.lo.s32 %r7,%r6,%r4;                // kh*d
    \\  ld.param.u32 %r8,[u0];                 // heads
    \\  mul.lo.s32 %r9,%r5,%r4;                // qk span = k_heads*d
    \\  // stage: tid < d loads k[kh*d + tid] (at conv_out + qk span);
    \\  //        d <= tid < 2d loads q[kh*d + tid-d] * scale (at offset 0).
    \\  shl.b32 %r10,%r4,1;
    \\  setp.ge.u32 %p1,%r3,%r10; @%p1 bra STAGED;
    \\  setp.ge.u32 %p2,%r3,%r4; @%p2 bra STQ;
    \\  add.u32 %r11,%r9,%r7; add.u32 %r11,%r11,%r3;       // k elem
    \\  mul.wide.u32 %rd5,%r11,4; add.s64 %rd6,%rd2,%rd5; ld.global.f32 %f2,[%rd6];
    \\  shl.b32 %r12,%r3,2; mov.u32 %r13,sk; add.u32 %r13,%r13,%r12; st.shared.f32 [%r13],%f2;
    \\  bra STAGED;
    \\STQ:
    \\  sub.u32 %r14,%r3,%r4;
    \\  add.u32 %r11,%r7,%r14;                              // q elem
    \\  mul.wide.u32 %rd5,%r11,4; add.s64 %rd6,%rd2,%rd5; ld.global.f32 %f2,[%rd6];
    \\  mul.f32 %f2,%f2,%f1;
    \\  shl.b32 %r12,%r14,2; mov.u32 %r13,sq; add.u32 %r13,%r13,%r12; st.shared.f32 [%r13],%f2;
    \\STAGED:
    \\  bar.sync 0;
    \\  setp.ge.u32 %p3,%r3,%r4; @%p3 bra END;  // only threads j < d continue
    \\  // decay/beta for this head
    \\  mul.wide.u32 %rd7,%r1,4; add.s64 %rd8,%rd3,%rd7; ld.global.f32 %f3,[%rd8];   // decay
    \\  mul.wide.u32 %rd9,%r8,4; add.s64 %rd10,%rd8,%rd9; ld.global.f32 %f4,[%rd10]; // beta
    \\  // v_j at conv_out[2*qk span + h*d + j]
    \\  shl.b32 %r15,%r9,1; mad.lo.s32 %r16,%r1,%r4,%r3; add.u32 %r16,%r16,%r15;
    \\  mul.wide.u32 %rd11,%r16,4; add.s64 %rd12,%rd2,%rd11; ld.global.f32 %f5,[%rd12];
    \\  // state column j: base = S + (h*d*d + j)*4, stride d*4
    \\  mul.lo.s32 %r17,%r4,%r4; mad.lo.s32 %r18,%r1,%r17,%r3;
    \\  mul.wide.u32 %rd13,%r18,4; add.s64 %rd14,%rd1,%rd13;
    \\  mul.wide.u32 %rd15,%r4,4;              // column stride bytes
    \\  shl.b64 %rd19,%rd15,1;                 // 2 rows
    \\  add.s64 %rd20,%rd19,%rd15;             // 3 rows
    \\  shl.b64 %rd21,%rd15,2;                 // 4 rows (loop step)
    \\  // pass 1: decay + memory readout m (x4 unrolled, same fma order)
    \\  mov.f32 %f6,0f00000000;                // m
    \\  mov.b64 %rd16,%rd14;
    \\  mov.u32 %r19,0; mov.u32 %r20,sk;
    \\P1:
    \\  setp.ge.u32 %p4,%r19,%r4; @%p4 bra P1D;
    \\  ld.global.f32 %f7,[%rd16];
    \\  add.s64 %rd22,%rd16,%rd15; ld.global.f32 %f12,[%rd22];
    \\  add.s64 %rd23,%rd16,%rd19; ld.global.f32 %f13,[%rd23];
    \\  add.s64 %rd24,%rd16,%rd20; ld.global.f32 %f14,[%rd24];
    \\  mul.f32 %f7,%f7,%f3;   st.global.f32 [%rd16],%f7;
    \\  mul.f32 %f12,%f12,%f3; st.global.f32 [%rd22],%f12;
    \\  mul.f32 %f13,%f13,%f3; st.global.f32 [%rd23],%f13;
    \\  mul.f32 %f14,%f14,%f3; st.global.f32 [%rd24],%f14;
    \\  ld.shared.f32 %f8,[%r20];    fma.rn.f32 %f6,%f7,%f8,%f6;
    \\  ld.shared.f32 %f8,[%r20+4];  fma.rn.f32 %f6,%f12,%f8,%f6;
    \\  ld.shared.f32 %f8,[%r20+8];  fma.rn.f32 %f6,%f13,%f8,%f6;
    \\  ld.shared.f32 %f8,[%r20+12]; fma.rn.f32 %f6,%f14,%f8,%f6;
    \\  add.s64 %rd16,%rd16,%rd21; add.u32 %r20,%r20,16; add.u32 %r19,%r19,4; bra P1;
    \\P1D:
    \\  sub.f32 %f9,%f5,%f6; mul.f32 %f9,%f9,%f4;          // d_j
    \\  // pass 2: rank-1 update + readout (x4 unrolled)
    \\  mov.f32 %f10,0f00000000;               // o
    \\  mov.b64 %rd16,%rd14;
    \\  mov.u32 %r19,0; mov.u32 %r20,sk; mov.u32 %r21,sq;
    \\P2:
    \\  setp.ge.u32 %p5,%r19,%r4; @%p5 bra P2D;
    \\  ld.global.f32 %f7,[%rd16];
    \\  add.s64 %rd22,%rd16,%rd15; ld.global.f32 %f12,[%rd22];
    \\  add.s64 %rd23,%rd16,%rd19; ld.global.f32 %f13,[%rd23];
    \\  add.s64 %rd24,%rd16,%rd20; ld.global.f32 %f14,[%rd24];
    \\  ld.shared.f32 %f8,[%r20];    fma.rn.f32 %f7,%f8,%f9,%f7;   st.global.f32 [%rd16],%f7;
    \\  ld.shared.f32 %f8,[%r20+4];  fma.rn.f32 %f12,%f8,%f9,%f12; st.global.f32 [%rd22],%f12;
    \\  ld.shared.f32 %f8,[%r20+8];  fma.rn.f32 %f13,%f8,%f9,%f13; st.global.f32 [%rd23],%f13;
    \\  ld.shared.f32 %f8,[%r20+12]; fma.rn.f32 %f14,%f8,%f9,%f14; st.global.f32 [%rd24],%f14;
    \\  ld.shared.f32 %f11,[%r21];    fma.rn.f32 %f10,%f7,%f11,%f10;
    \\  ld.shared.f32 %f11,[%r21+4];  fma.rn.f32 %f10,%f12,%f11,%f10;
    \\  ld.shared.f32 %f11,[%r21+8];  fma.rn.f32 %f10,%f13,%f11,%f10;
    \\  ld.shared.f32 %f11,[%r21+12]; fma.rn.f32 %f10,%f14,%f11,%f10;
    \\  add.s64 %rd16,%rd16,%rd21; add.u32 %r20,%r20,16; add.u32 %r21,%r21,16; add.u32 %r19,%r19,4; bra P2;
    \\P2D:
    \\  mad.lo.s32 %r22,%r1,%r4,%r3;           // o elem = h*d + j
    \\  mul.wide.u32 %rd17,%r22,4; add.s64 %rd18,%rd4,%rd17; st.global.f32 [%rd18],%f10;
    \\END:
    \\  ret;
    \\}
;

/// VAE per-position channel L2 norm: out[row][ch] = x[row][ch] * inv * gamma[ch],
/// inv = sqrt(c)/max(||x_row||_2, eps); optional fused silu. One thread per
/// position (channel-last [n][c]). b0=x, b1=out, b2=gamma. u0=n, u1=c, u2=silu,
/// f0=eps (1e-12).
pub const vae_norm_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry vae_norm(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.f32 %f8,[f0];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  mul.lo.s32 %r8,%r4,%r6;               // base = row*c
    \\  mul.wide.u32 %rd4,%r8,4; add.s64 %rd5,%rd1,%rd4; add.s64 %rd6,%rd2,%rd4;  // x/out row ptrs
    \\  mov.f32 %f1,0f00000000; mov.u32 %r9,0; mov.b64 %rd7,%rd5;
    \\SS:
    \\  setp.ge.u32 %p2,%r9,%r6; @%p2 bra SSD;
    \\  ld.global.f32 %f2,[%rd7]; fma.rn.f32 %f1,%f2,%f2,%f1;
    \\  add.s64 %rd7,%rd7,4; add.u32 %r9,%r9,1; bra SS;
    \\SSD:
    \\  sqrt.rn.f32 %f3,%f1; max.f32 %f3,%f3,%f8;         // max(||x||_2, eps)
    \\  cvt.rn.f32.u32 %f4,%r6; sqrt.rn.f32 %f4,%f4;      // sqrt(c)
    \\  div.rn.f32 %f5,%f4,%f3;                            // inv
    \\  mov.u32 %r9,0; mov.b64 %rd7,%rd5; mov.b64 %rd8,%rd3; mov.b64 %rd9,%rd6;
    \\AP:
    \\  setp.ge.u32 %p2,%r9,%r6; @%p2 bra END;
    \\  ld.global.f32 %f2,[%rd7]; ld.global.f32 %f6,[%rd8];
    \\  mul.f32 %f2,%f2,%f5; mul.f32 %f2,%f2,%f6;         // v = x*inv*gamma
    \\  setp.eq.u32 %p3,%r7,0; @%p3 bra STORE;
    \\  neg.f32 %f7,%f2; mul.f32 %f7,%f7,0f3FB8AA3B; ex2.approx.f32 %f7,%f7; add.f32 %f7,%f7,0f3F800000; rcp.approx.f32 %f7,%f7;
    \\  mul.f32 %f2,%f2,%f7;                               // silu(v) = v*sigmoid(v)
    \\STORE:
    \\  st.global.f32 [%rd9],%f2;
    \\  add.s64 %rd7,%rd7,4; add.s64 %rd8,%rd8,4; add.s64 %rd9,%rd9,4; add.u32 %r9,%r9,1; bra AP;
    \\END:
    \\  ret;
    \\}
;

/// im2col for a zero-padded 3x3 conv over channel-last [h*w][ci] activations,
/// producing a patch matrix [bn][9*ci] so the conv is a GEMM. With f0!=0 the
/// source is read through a fused nearest-exact 2x upsample (coords halve; the
/// upsampled tensor never materializes). One thread per output f32.
/// b0=src, b1=out(patch). u0=bn*plen, u1=plen(9*ci), u2=ci, u3=src w, u4=src h,
/// u5=first output position of the band, f0=upsample flag.
pub const im2col_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry im2col(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<6>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3]; ld.param.u32 %r9,[u4]; ld.param.u32 %r10,[u5];
    \\  ld.param.f32 %f1,[f0]; setp.neu.f32 %p2,%f1,0f00000000; selp.b32 %r11,1,0,%p2; // up
    \\  shl.b32 %r12,%r8,%r11; shl.b32 %r13,%r9,%r11;      // ow, oh
    \\  rem.u32 %r14,%r4,%r6; div.u32 %r15,%r4,%r6;        // col, band-row
    \\  add.u32 %r16,%r10,%r15;                             // p = band start + band-row
    \\  div.u32 %r17,%r14,%r7; rem.u32 %r18,%r14,%r7;      // tap, cc
    \\  div.u32 %r19,%r16,%r12; rem.u32 %r20,%r16,%r12;    // oy, ox
    \\  div.u32 %r21,%r17,3; rem.u32 %r22,%r17,3;          // ky, kx
    \\  add.u32 %r19,%r19,%r21; add.u32 %r20,%r20,%r22;    // yk, xk
    \\  mov.f32 %f2,0f00000000;
    \\  setp.lt.u32 %p3,%r19,1; @%p3 bra STORE;
    \\  setp.gt.u32 %p3,%r19,%r13; @%p3 bra STORE;
    \\  setp.lt.u32 %p3,%r20,1; @%p3 bra STORE;
    \\  setp.gt.u32 %p3,%r20,%r12; @%p3 bra STORE;
    \\  sub.u32 %r19,%r19,1; shr.u32 %r19,%r19,%r11;       // sy
    \\  sub.u32 %r20,%r20,1; shr.u32 %r20,%r20,%r11;       // sx
    \\  mad.lo.s32 %r23,%r19,%r8,%r20; mad.lo.s32 %r23,%r23,%r7,%r18; // (sy*w+sx)*ci+cc
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd2,%r23,4; add.s64 %rd3,%rd1,%rd2; ld.global.f32 %f2,[%rd3];
    \\STORE:
    \\  ld.param.u64 %rd4,[p1]; cvta.to.global.u64 %rd4,%rd4;
    \\  mul.wide.u32 %rd5,%r4,4; add.s64 %rd6,%rd4,%rd5; st.global.f32 [%rd6],%f2;
    \\END:
    \\  ret;
    \\}
;

/// Convert a tight f32 [rows][cols] matrix to a zero-padded f16 [*][cols_pad]
/// (rows padded implicitly by the launch size) so it feeds the 128×n / k%32
/// tensor-core GEMM. out[idx] with r=idx/cols_pad, c=idx%cols_pad =
/// (r<rows and c<cols) ? f16(src[r*cols+c]) : 0. b0=src(f32), b1=out(f16).
/// u0=total(rows_pad*cols_pad), u1=cols_pad, u2=rows, u3=cols.
pub const f32_to_f16_pad2d_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32_to_f16_pad2d(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3];
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // &out[idx] f16
    \\  div.u32 %r9,%r4,%r6; rem.u32 %r10,%r4,%r6;         // r, c
    \\  mov.b16 %h0,0x0000;
    \\  setp.ge.u32 %p2,%r9,%r7; @%p2 bra STORE;
    \\  setp.ge.u32 %p3,%r10,%r8; @%p3 bra STORE;
    \\  mad.lo.s32 %r11,%r9,%r8,%r10;                       // r*cols + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r11,4; add.s64 %rd6,%rd1,%rd5; ld.global.f32 %f1,[%rd6]; cvt.rn.f16.f32 %h0,%f1;
    \\STORE:
    \\  st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// f32_to_f16_pad2d but converting to bf16 (native bf16 tensor-core GEMM): the
/// activation feeds the bf16 MMA alongside the raw-bf16 weight, so no f16 round
/// trip. out[idx] with r=idx/cols_pad, c=idx%cols_pad = (r<rows and c<cols) ?
/// bf16(src_f32[r*cols+c]) : 0. b0=src(f32), b1=out(bf16). u1=cols_pad, u2=rows,
/// u3=cols.
pub const f32_to_bf16_pad2d_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32_to_bf16_pad2d(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3];
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // &out[idx] bf16
    \\  div.u32 %r9,%r4,%r6; rem.u32 %r10,%r4,%r6;         // r, c
    \\  mov.b16 %h0,0x0000;
    \\  setp.ge.u32 %p2,%r9,%r7; @%p2 bra STORE;
    \\  setp.ge.u32 %p3,%r10,%r8; @%p3 bra STORE;
    \\  mad.lo.s32 %r11,%r9,%r8,%r10;                       // r*cols + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r11,4; add.s64 %rd6,%rd1,%rd5; ld.global.f32 %f1,[%rd6]; cvt.rn.bf16.f32 %h0,%f1;
    \\STORE:
    \\  st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// f32_to_f16_pad2d's bf16-source twin (ViT GEMMs consume the mmproj's bf16
/// weights straight from the mmap): out[idx] with r=idx/cols_pad,
/// c=idx%cols_pad = (r<rows and c<cols) ? f16(f32(src_bf16[r*cols+c])) : 0.
/// b0=src(bf16), b1=out(f16). u0=total(rows_pad*cols_pad), u1=cols_pad,
/// u2=rows, u3=cols.
pub const bf16_to_f16_pad2d_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry bf16_to_f16_pad2d(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<14>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3];
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // &out[idx] f16
    \\  div.u32 %r9,%r4,%r6; rem.u32 %r10,%r4,%r6;         // r, c
    \\  mov.b16 %h0,0x0000;
    \\  setp.ge.u32 %p2,%r9,%r7; @%p2 bra STORE;
    \\  setp.ge.u32 %p3,%r10,%r8; @%p3 bra STORE;
    \\  mad.lo.s32 %r11,%r9,%r8,%r10;                       // r*cols + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r11,2; add.s64 %rd6,%rd1,%rd5; ld.global.u16 %r12,[%rd6];
    \\  shl.b32 %r13,%r12,16; mov.b32 %f1,%r13;             // bf16 -> f32
    \\  cvt.rn.f16.f32 %h0,%f1;
    \\STORE:
    \\  st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// f16 → f16 2-D pad: copy W[co][cols] into a [rows][cols_pad] f16 tile,
/// zero-padding the tail. Same shape as `bf16_to_f16_pad2d` but the source is
/// already f16, so each lane is a straight 16-bit copy (no conversion). Used by
/// `opMatmulF16` for GGUF mmproj weights stored as f16.
/// b0=W(f16), b1=out(f16). u0=total(rows*cols_pad), u1=cols_pad, u2=co, u3=cols.
pub const f16_pad2d_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f16_pad2d(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<12>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3];
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // &out[idx] f16
    \\  div.u32 %r9,%r4,%r6; rem.u32 %r10,%r4,%r6;         // r, c (padded)
    \\  mov.b16 %h0,0x0000;
    \\  setp.ge.u32 %p2,%r9,%r7; @%p2 bra STORE;
    \\  setp.ge.u32 %p3,%r10,%r8; @%p3 bra STORE;
    \\  mad.lo.s32 %r11,%r9,%r8,%r10;                       // r*cols + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r11,2; add.s64 %rd6,%rd1,%rd5; ld.global.u16 %h0,[%rd6]; // f16 as-is
    \\STORE:
    \\  st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Strip the column padding from a [*][co_pad] f32 GEMM output and add the conv
/// bias in one pass: dst[dst_off + i] = C[(i/co)*co_pad + i%co] + bias[i%co].
/// b0=C(f32 padded), b1=bias(f32[co]), b2=dst(f32). u0=total(m*co), u1=co,
/// u2=co_pad, u3=dst offset (elements).
pub const bias_compact_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry bias_compact(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2]; ld.param.u32 %r8,[u3];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  div.u32 %r9,%r4,%r6; rem.u32 %r10,%r4,%r6;          // r, c
    \\  mad.lo.s32 %r11,%r9,%r7,%r10;                        // r*co_pad + c
    \\  mul.wide.u32 %rd4,%r11,4; add.s64 %rd5,%rd1,%rd4; ld.global.f32 %f1,[%rd5];  // C
    \\  mul.wide.u32 %rd6,%r10,4; add.s64 %rd7,%rd2,%rd6; ld.global.f32 %f2,[%rd7];  // bias[c]
    \\  add.f32 %f1,%f1,%f2;
    \\  add.s32 %r11,%r8,%r4;                                // dst_off + i
    \\  mul.wide.u32 %rd8,%r11,4; add.s64 %rd9,%rd3,%rd8; st.global.f32 [%rd9],%f1;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] += b[idx], in place (plain residual add). b0=a, b1=b. u0=total.
pub const add_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry add(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd5]; add.f32 %f1,%f1,%f2; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// ReLU in place: a[idx] = max(0, a[idx]). b0=a. u0=total.
pub const relu_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry relu(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<6>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b64 %rd<5>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; max.f32 %f1,%f1,0f00000000; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// argmax pass 1: `lanes` stride-scanners. Lane `l` scans logits[l], logits[l+
/// lanes], ... keeping its max and the LOWEST index achieving it (ascending
/// scan + strict `>`), writing them to out_val[l]/out_idx[l] (index as an exact
/// f32; vocab < 2^24). b0=logits, b1=out_val, b2=out_idx. u0=lanes, u1=vocab.
pub const argmax_reduce_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry argmax_reduce(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd2,%r4,4; add.s64 %rd3,%rd1,%rd2;
    \\  ld.global.f32 %f1,[%rd3];
    \\  mov.u32 %r7,%r4;
    \\  add.u32 %r8,%r4,%r5;
    \\LOOP:
    \\  setp.ge.u32 %p2,%r8,%r6; @%p2 bra DONE;
    \\  mul.wide.u32 %rd4,%r8,4; add.s64 %rd5,%rd1,%rd4;
    \\  ld.global.f32 %f2,[%rd5];
    \\  setp.gt.f32 %p1,%f2,%f1;
    \\  @%p1 mov.f32 %f1,%f2;
    \\  @%p1 mov.u32 %r7,%r8;
    \\  add.u32 %r8,%r8,%r5;
    \\  bra LOOP;
    \\DONE:
    \\  ld.param.u64 %rd6,[p1]; cvta.to.global.u64 %rd6,%rd6; add.s64 %rd7,%rd6,%rd2; st.global.f32 [%rd7],%f1;
    \\  ld.param.u64 %rd8,[p2]; cvta.to.global.u64 %rd8,%rd8; add.s64 %rd9,%rd8,%rd2; cvt.rn.f32.u32 %f3,%r7; st.global.f32 [%rd9],%f3;
    \\END:
    \\  ret;
    \\}
;

/// argmax pass 2: one thread reduces the `lanes` lane winners to the global
/// argmax, tie-breaking to the lowest index (matches sample.argmax). b0=vals,
/// b1=idx, b2=out (id as f32). u0=lanes.
pub const argmax_final_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry argmax_final(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<6>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  setp.ne.u32 %p1,%r4,0; @%p1 bra END;
    \\  ld.param.u32 %r5,[u0];
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  ld.global.f32 %f1,[%rd1];
    \\  ld.global.f32 %f2,[%rd2]; cvt.rzi.u32.f32 %r6,%f2;
    \\  mov.u32 %r7,1;
    \\LOOP:
    \\  setp.ge.u32 %p1,%r7,%r5; @%p1 bra DONE;
    \\  mul.wide.u32 %rd3,%r7,4;
    \\  add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f3,[%rd4];
    \\  add.s64 %rd5,%rd2,%rd3; ld.global.f32 %f4,[%rd5]; cvt.rzi.u32.f32 %r8,%f4;
    \\  setp.gt.f32 %p2,%f3,%f1;
    \\  setp.eq.f32 %p3,%f3,%f1;
    \\  setp.lt.u32 %p4,%r8,%r6;
    \\  and.pred %p3,%p3,%p4;
    \\  or.pred %p2,%p2,%p3;
    \\  @%p2 mov.f32 %f1,%f3;
    \\  @%p2 mov.u32 %r6,%r8;
    \\  add.u32 %r7,%r7,1;
    \\  bra LOOP;
    \\DONE:
    \\  ld.param.u64 %rd6,[p2]; cvta.to.global.u64 %rd6,%rd6; cvt.rn.f32.u32 %f5,%r6; st.global.f32 [%rd6],%f5;
    \\END:
    \\  ret;
    \\}
;

/// top-k pass: L lanes each keep their 8 highest (value, index) over their
/// stride-slice via min-slot tracking (strict `>` keeps the lower index on
/// ties). b0=logits, b1=out_val[L*8], b2=out_idx[L*8]. u0=L, u1=vocab. The host
/// does exact top-k over the L*8 candidates. Uses per-thread .local arrays (no
/// shared memory), matching the Vulkan topk_reduce kernel. Must keep M=8 in sync.
pub const topk_reduce_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry topk_reduce(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<4>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<12>;
    \\  .local .align 4 .b32 bv[8];
    \\  .local .align 4 .b32 bi[8];
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];
    \\  mov.b32 %r7,0fFF7FFFFF; mov.u32 %r8,0;
    \\  st.local.u32 [bv+0],%r7; st.local.u32 [bv+4],%r7; st.local.u32 [bv+8],%r7; st.local.u32 [bv+12],%r7;
    \\  st.local.u32 [bv+16],%r7; st.local.u32 [bv+20],%r7; st.local.u32 [bv+24],%r7; st.local.u32 [bv+28],%r7;
    \\  st.local.u32 [bi+0],%r8; st.local.u32 [bi+4],%r8; st.local.u32 [bi+8],%r8; st.local.u32 [bi+12],%r8;
    \\  st.local.u32 [bi+16],%r8; st.local.u32 [bi+20],%r8; st.local.u32 [bi+24],%r8; st.local.u32 [bi+28],%r8;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mov.u32 %r9,%r4;
    \\OUTER:
    \\  setp.ge.u32 %p1,%r9,%r6; @%p1 bra WRITE;
    \\  mul.wide.u32 %rd2,%r9,4; add.s64 %rd3,%rd1,%rd2; ld.global.f32 %f1,[%rd3];
    \\  mov.u32 %r10,0; ld.local.f32 %f2,[bv+0]; mov.u32 %r11,1;
    \\INNER:
    \\  setp.ge.u32 %p2,%r11,8; @%p2 bra DECIDE;
    \\  mov.u32 %r13,bv; mul.lo.u32 %r12,%r11,4; add.u32 %r14,%r13,%r12; ld.local.f32 %f3,[%r14];
    \\  setp.lt.f32 %p3,%f3,%f2; @%p3 mov.f32 %f2,%f3; @%p3 mov.u32 %r10,%r11;
    \\  add.u32 %r11,%r11,1; bra INNER;
    \\DECIDE:
    \\  setp.gt.f32 %p2,%f1,%f2; @!%p2 bra NEXT;
    \\  mul.lo.u32 %r12,%r10,4;
    \\  mov.u32 %r13,bv; add.u32 %r14,%r13,%r12; st.local.f32 [%r14],%f1;
    \\  mov.u32 %r13,bi; add.u32 %r14,%r13,%r12; st.local.u32 [%r14],%r9;
    \\NEXT:
    \\  add.u32 %r9,%r9,%r5; bra OUTER;
    \\WRITE:
    \\  ld.param.u64 %rd4,[p1]; cvta.to.global.u64 %rd4,%rd4;
    \\  ld.param.u64 %rd5,[p2]; cvta.to.global.u64 %rd5,%rd5;
    \\  mul.lo.u32 %r15,%r4,8; mul.wide.u32 %rd6,%r15,4; add.s64 %rd7,%rd4,%rd6; add.s64 %rd8,%rd5,%rd6;
    \\  mov.u32 %r11,0;
    \\WLOOP:
    \\  setp.ge.u32 %p1,%r11,8; @%p1 bra END;
    \\  mul.lo.u32 %r12,%r11,4;
    \\  mov.u32 %r13,bv; add.u32 %r14,%r13,%r12; ld.local.f32 %f1,[%r14];
    \\  mul.wide.u32 %rd9,%r11,4; add.s64 %rd10,%rd7,%rd9; st.global.f32 [%rd10],%f1;
    \\  mov.u32 %r13,bi; add.u32 %r14,%r13,%r12; ld.local.u32 %r16,[%r14]; cvt.rn.f32.u32 %f2,%r16;
    \\  add.s64 %rd11,%rd8,%rd9; st.global.f32 [%rd11],%f2;
    \\  add.u32 %r11,%r11,1; bra WLOOP;
    \\END:
    \\  ret;
    \\}
;

/// Sampling penalties (llm/sample.zig penalizeLogit), one thread per unique
/// recent token: logits[id] = (l>0 ? l/rp : l*rp) - sub, with the presence +
/// frequency subtract term `sub` precomputed on the host (packPenaltyWireU32).
/// b0=logits, b1=wire (interleaved u32 pairs: id, f32-bits sub). u0=entry
/// count, f0=repeat penalty. Ids are unique, so no two threads touch the same
/// logit; div.rn/mul.rn/sub.rn keep the math bit-identical to the CPU path.
pub const penalize_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry penalize(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<7>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,8; add.s64 %rd4,%rd2,%rd3;
    \\  ld.global.u32 %r6,[%rd4]; ld.global.f32 %f2,[%rd4+4];
    \\  mul.wide.u32 %rd5,%r6,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.f32 %f1,[%rd6];
    \\  ld.param.f32 %f3,[f0];
    \\  div.rn.f32 %f4,%f1,%f3;
    \\  mul.rn.f32 %f5,%f1,%f3;
    \\  setp.gt.f32 %p2,%f1,0f00000000;
    \\  selp.f32 %f6,%f4,%f5,%p2;
    \\  sub.rn.f32 %f6,%f6,%f2;
    \\  st.global.f32 [%rd6],%f6;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] *= sigmoid(b[idx]), in place. b0=a, b1=b. u0=total.
pub const sigmoid_mul_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry sigmoid_mul(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd5];
    \\  neg.f32 %f3,%f2; mul.f32 %f3,%f3,0f3FB8AA3B; ex2.approx.f32 %f3,%f3; add.f32 %f3,%f3,0f3F800000; rcp.approx.f32 %f3,%f3;
    \\  mul.f32 %f1,%f1,%f3; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] = silu(a[idx]) * b[idx], in place. b0=a(gate), b1=b(up). u0=total.
/// f16 SwiGLU gate: a = silu(a) * b, all f16 in/out (the c16 chain). Same math
/// as silu_mul, ×2 byte strides + b16 load/store. b0=a (gate), b1=b (up).
pub const silu_mul_h16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry silu_mul_h16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.b16 %h0,[%rd4]; cvt.f32.f16 %f1,%h0;
    \\  ld.global.b16 %h1,[%rd5]; cvt.f32.f16 %f2,%h1;
    \\  neg.f32 %f3,%f1; mul.f32 %f3,%f3,0f3FB8AA3B; ex2.approx.f32 %f3,%f3; add.f32 %f3,%f3,0f3F800000; rcp.approx.f32 %f3,%f3;
    \\  mul.f32 %f1,%f1,%f3;                  // silu(g) = g*sigmoid(g)
    \\  mul.f32 %f1,%f1,%f2; cvt.rn.f16.f32 %h2,%f1; st.global.b16 [%rd4],%h2;
    \\END:
    \\  ret;
    \\}
;

pub const silu_mul_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry silu_mul(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd5];
    \\  neg.f32 %f3,%f1; mul.f32 %f3,%f3,0f3FB8AA3B; ex2.approx.f32 %f3,%f3; add.f32 %f3,%f3,0f3F800000; rcp.approx.f32 %f3,%f3;
    \\  mul.f32 %f1,%f1,%f3;                  // silu(g) = g*sigmoid(g)
    \\  mul.f32 %f1,%f1,%f2; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] = geluTanh(a[idx]), in place. b0=a. u0=total. Tanh-gelu folds to
/// x·sigmoid(w), w = x·(c1 + c2·x²) with c1=2·√(2/π), c2=c1·0.044715 — matches
/// ops.act.geluTanh to f32 rounding (sigmoid via ex2.approx, as in silu_mul).
pub const gelu_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gelu(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<6>;
    \\  .reg .f32 %f<6>;
    \\  .reg .b64 %rd<6>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3;
    \\  ld.global.f32 %f1,[%rd4];
    \\  mul.f32 %f2,%f1,%f1;                        // x^2
    \\  fma.rn.f32 %f3,%f2,0f3D922279,0f3FCC422A;   // c1 + c2*x^2
    \\  mul.f32 %f3,%f3,%f1;                        // w = x*(c1 + c2*x^2)
    \\  neg.f32 %f4,%f3; mul.f32 %f4,%f4,0f3FB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; rcp.approx.f32 %f4,%f4;
    \\  mul.f32 %f1,%f1,%f4;                        // x*sigmoid(w)
    \\  st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// GeGLU gate: a[idx] = geluTanh(a[idx]) * b[idx], in place (Gemma FFN).
/// Same folded tanh-gelu as `gelu` (x·sigmoid(w), w = x·(c1 + c2·x²)),
/// then multiplied by the up projection b — the fused twin of silu_mul.
/// b0=a (gate), b1=b (up). u0=total.
pub const gelu_mul_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gelu_mul(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<6>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; add.s64 %rd5,%rd2,%rd3;
    \\  ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd5];
    \\  mul.f32 %f5,%f1,%f1;                        // x^2
    \\  fma.rn.f32 %f3,%f5,0f3D922279,0f3FCC422A;   // c1 + c2*x^2
    \\  mul.f32 %f3,%f3,%f1;                        // w = x*(c1 + c2*x^2)
    \\  neg.f32 %f4,%f3; mul.f32 %f4,%f4,0f3FB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; rcp.approx.f32 %f4,%f4;
    \\  mul.f32 %f1,%f1,%f4;                        // geluTanh(x) = x*sigmoid(w)
    \\  mul.f32 %f1,%f1,%f2; st.global.f32 [%rd4],%f1;
    \\END:
    \\  ret;
    \\}
;

/// a[idx] += mod[u2 + idx%u1] * b[idx], in place. b0=a, b1=b(delta), b2=mod.
/// u0=total, u1=dim(F), u2=gate_off.
pub const gated_add_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gated_add(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<6>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r8,%r4,%r6; add.s32 %r9,%r8,%r7;           // gate col = u2 + idx%u1
    \\  mul.wide.u32 %rd4,%r4,4; add.s64 %rd5,%rd1,%rd4; add.s64 %rd6,%rd2,%rd4;
    \\  mul.wide.u32 %rd7,%r9,4; add.s64 %rd8,%rd3,%rd7;
    \\  ld.global.f32 %f1,[%rd5]; ld.global.f32 %f2,[%rd6]; ld.global.f32 %f3,[%rd8];
    \\  fma.rn.f32 %f1,%f3,%f2,%f1; st.global.f32 [%rd5],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Restride per-head slices between packed layouts (ViT head-dim padding:
/// 72-dim heads -> 128-dim zero-padded for the tensor-core attention, and
/// back). out[t][h][d] (d < out_hd, contiguous [rows][heads*out_hd]) =
/// d < in_hd ? in[t*in_stride + in_off + h*in_hd + d] : 0. One thread per
/// out element. b0=in(f32), b1=out(f32). u0=total(=rows*heads*out_hd),
/// u1=out_hd, u2=in_hd, u3=in_stride (elements per token row), u4=in_off,
/// u5=heads.
pub const head_pad_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry head_pad(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // out_hd
    \\  ld.param.u32 %r7,[u2];               // in_hd
    \\  ld.param.u32 %r8,[u3];               // in_stride
    \\  ld.param.u32 %r9,[u4];               // in_off
    \\  ld.param.u32 %r10,[u5];              // heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  div.u32 %r11,%r4,%r6; rem.u32 %r12,%r4,%r6;        // hp, d
    \\  div.u32 %r13,%r11,%r10; rem.u32 %r14,%r11,%r10;    // t, h
    \\  mov.f32 %f1,0f00000000;
    \\  setp.ge.u32 %p2,%r12,%r7; @%p2 bra STORE;          // pad dim -> 0
    \\  mad.lo.s32 %r15,%r13,%r8,%r9;                      // t*in_stride + in_off
    \\  mad.lo.s32 %r15,%r14,%r7,%r15; add.u32 %r15,%r15,%r12;
    \\  mul.wide.u32 %rd3,%r15,4; add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f1,[%rd4];
    \\STORE:
    \\  mul.wide.u32 %rd5,%r4,4; add.s64 %rd6,%rd2,%rd5; st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Gather one head's rows from an interleaved [seq][nheads][hd] f32 tensor into a
/// contiguous [mpad][hd] f16 tile, zero-padding rows >= seq (so tensor-core GEMM
/// sees clean pad rows). b0=src(f32), b1=dst(f16). u0=seq, u1=nheads, u2=head,
/// u3=hd, u4=total (=mpad*hd). One thread per dst element.
pub const gather_head_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gather_head(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<14>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u4]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u0]; ld.param.u32 %r9,[u3];  // seq, hd
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // dst[idx] (f16)
    \\  div.u32 %r10,%r4,%r9;                              // row = idx/hd
    \\  setp.ge.u32 %p2,%r10,%r6; @%p2 bra ZERO;
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2];      // nheads, head
    \\  rem.u32 %r11,%r4,%r9;                              // c = idx%hd
    \\  mad.lo.s32 %r10,%r10,%r7,%r8;                       // row*nheads + head
    \\  mad.lo.s32 %r10,%r10,%r9,%r11;                      // *hd + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r10,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.f32 %f1,[%rd6]; cvt.rn.f16.f32 %h0,%f1; st.global.b16 [%rd4],%h0; bra END;
    \\ZERO:
    \\  mov.b16 %h0,0x0000; st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Gather one KV head's V, transposed to [hd][mpad] f16, zero-padding cols >= seq.
/// V source is interleaved [seq][kv_heads][hd] f32. dst[c*mpad + j] = V[j][kvh][c].
/// b0=src(f32), b1=dst(f16). u0=seq, u1=kv_heads, u2=kvhead, u3=hd, u4=mpad,
/// u5=total (=hd*mpad). One thread per dst element.
pub const gather_vt_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gather_vt(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<16>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u5]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u0]; ld.param.u32 %r12,[u4];    // seq, mpad
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;   // dst[idx] (f16)
    \\  div.u32 %r10,%r4,%r12;                             // c = idx/mpad
    \\  rem.u32 %r11,%r4,%r12;                             // j = idx%mpad
    \\  setp.ge.u32 %p2,%r11,%r6; @%p2 bra ZERO;
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2]; ld.param.u32 %r9,[u3];  // kv_heads, kvhead, hd
    \\  mad.lo.s32 %r13,%r11,%r7,%r8;                       // j*kv_heads + kvhead
    \\  mad.lo.s32 %r13,%r13,%r9,%r10;                      // *hd + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r13,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.f32 %f1,[%rd6]; cvt.rn.f16.f32 %h0,%f1; st.global.b16 [%rd4],%h0; bra END;
    \\ZERO:
    \\  mov.b16 %h0,0x0000; st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Scatter one head's attention output [mpad][hd] f32 (rows 0..seq) into the
/// interleaved [seq][heads][hd] f32 output. b0=src(f32), b1=dst(f32). u0=seq,
/// u1=heads, u2=head, u3=hd, u4=total (=seq*hd). One thread per src element.
pub const scatter_head_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry scatter_head(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<14>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u4]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2]; ld.param.u32 %r9,[u3];  // heads, head, hd
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd5,%rd1,%rd3;   // src[idx]
    \\  ld.global.f32 %f1,[%rd5];
    \\  div.u32 %r10,%r4,%r9;                              // row = idx/hd
    \\  rem.u32 %r11,%r4,%r9;                              // c = idx%hd
    \\  mad.lo.s32 %r10,%r10,%r7,%r8;                       // row*heads + head
    \\  mad.lo.s32 %r10,%r10,%r9,%r11;                      // *hd + c
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd4,%r10,4; add.s64 %rd6,%rd2,%rd4;
    \\  st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
;

// ---- batched (head-group) gather/scatter: one launch handles `gsize` heads ----
// gid selects a flat element of the [gsize][...] output; the head index is
// derived from base_head + z, so a single launch feeds a grid.z-batched hgemm.

/// Batched Q/K gather: dst[z][mpad][hd] f16 for heads base_h..base_h+gsize.
/// GQA is unified via group_div (1 for Q, group for K): head=(base_h+z)/group_div.
/// b0=src(f32 [seq][nheads][hd]), b1=dst(f16). u0=seq, u1=nheads, u2=base_h,
/// u3=group_div, u4=hd, u5=mpad, u6=total (=gsize*mpad*hd).
pub const gather_head_b_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gather_head_b(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .u32 u6)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u6]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;   // total
    \\  ld.param.u32 %r9,[u4]; ld.param.u32 %r12,[u5];                  // hd, mpad
    \\  mul.lo.s32 %r13,%r9,%r12;                                       // sl = hd*mpad
    \\  div.u32 %r14,%r4,%r13; rem.u32 %r15,%r4,%r13;                   // z, rem
    \\  div.u32 %r10,%r15,%r9;                                          // row = rem/hd
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;                // dst[idx] f16
    \\  ld.param.u32 %r6,[u0]; setp.ge.u32 %p2,%r10,%r6; @%p2 bra ZERO; // seq
    \\  rem.u32 %r11,%r15,%r9;                                          // c = rem%hd
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2]; ld.param.u32 %r16,[u3]; // nheads, base_h, group_div
    \\  add.s32 %r17,%r8,%r14; div.u32 %r17,%r17,%r16;                  // head=(base_h+z)/group_div
    \\  mad.lo.s32 %r10,%r10,%r7,%r17; mad.lo.s32 %r10,%r10,%r9,%r11;   // (row*nheads+head)*hd + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r10,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.f32 %f1,[%rd6]; cvt.rn.f16.f32 %h0,%f1; st.global.b16 [%rd4],%h0; bra END;
    \\ZERO:
    \\  mov.b16 %h0,0x0000; st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Batched V→Vt gather: dst[z][hd][mpad] f16 for kv heads (base_h+z)/group.
/// b0=src(f32 [seq][kv_heads][hd]), b1=dst(f16). u0=seq, u1=kv_heads, u2=base_h,
/// u3=group, u4=hd, u5=mpad, u6=total (=gsize*hd*mpad).
pub const gather_vt_b_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry gather_vt_b(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .u32 u6)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u6]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r9,[u4]; ld.param.u32 %r12,[u5];                  // hd, mpad
    \\  mul.lo.s32 %r13,%r9,%r12;                                       // sl = hd*mpad
    \\  div.u32 %r14,%r4,%r13; rem.u32 %r15,%r4,%r13;                   // z, rem
    \\  div.u32 %r10,%r15,%r12; rem.u32 %r11,%r15,%r12;                 // c=rem/mpad, j=rem%mpad
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;
    \\  ld.param.u32 %r6,[u0]; setp.ge.u32 %p2,%r11,%r6; @%p2 bra ZERO; // seq; skip if j>=seq
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2]; ld.param.u32 %r16,[u3]; // kv_heads, base_h, group
    \\  add.s32 %r17,%r8,%r14; div.u32 %r17,%r17,%r16;                  // head=(base_h+z)/group
    \\  mad.lo.s32 %r18,%r11,%r7,%r17; mad.lo.s32 %r18,%r18,%r9,%r10;   // (j*kv_heads+head)*hd + c
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r18,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.f32 %f1,[%rd6]; cvt.rn.f16.f32 %h0,%f1; st.global.b16 [%rd4],%h0; bra END;
    \\ZERO:
    \\  mov.b16 %h0,0x0000; st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Batched scatter: src[z][mpad][hd] f32 (rows 0..seq) → out[row][base_h+z][hd].
/// b0=src(f32), b1=dst(f32 [seq][heads][hd]). u0=seq, u1=heads, u2=base_h, u3=hd,
/// u4=mpad, u5=total (=gsize*seq*hd).
pub const scatter_head_b_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry scatter_head_b(.param .u64 p0,.param .u64 p1,.param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u5]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;   // total
    \\  ld.param.u32 %r6,[u0]; ld.param.u32 %r9,[u3];                   // seq, hd
    \\  mul.lo.s32 %r13,%r6,%r9;                                        // sl = seq*hd
    \\  div.u32 %r14,%r4,%r13; rem.u32 %r15,%r4,%r13;                   // z, rem
    \\  div.u32 %r10,%r15,%r9; rem.u32 %r11,%r15,%r9;                   // row=rem/hd, c=rem%hd
    \\  ld.param.u32 %r12,[u4]; mul.lo.s32 %r16,%r12,%r9;               // mpad*hd
    \\  mad.lo.s32 %r16,%r14,%r16,%r15;                                 // z*(mpad*hd) + rem
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd3,%r16,4; add.s64 %rd5,%rd1,%rd3; ld.global.f32 %f1,[%rd5];
    \\  ld.param.u32 %r7,[u1]; ld.param.u32 %r8,[u2];                   // heads, base_h
    \\  add.s32 %r17,%r8,%r14; mad.lo.s32 %r10,%r10,%r7,%r17; mad.lo.s32 %r10,%r10,%r9,%r11; // (row*heads+base_h+z)*hd+c
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd4,%r10,4; add.s64 %rd6,%rd2,%rd4;
    \\  st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize fp8-e4m3 weights to f16: out[i] = f16(lut[in[i]] * scale). The
/// e4m3 byte indexes a 256-entry f32 lookup table (dtype.f8_e4m3_to_f32_table),
/// then the per-tensor weight scale is folded in. b0=in(u8 fp8), b1=lut(f32[256]),
/// b2=out(f16). u0=total. f0=scale. One thread per element.
pub const dequant_fp8_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_fp8_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.f32 %f1,[f0];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    \\  cvt.u64.u32 %rd4,%r4; add.s64 %rd5,%rd1,%rd4;      // &in[idx]
    \\  ld.global.u8 %r6,[%rd5];                            // fp8 byte (0..255)
    \\  mul.wide.u32 %rd6,%r6,4; add.s64 %rd7,%rd2,%rd6;    // &lut[byte]
    \\  ld.global.f32 %f2,[%rd7]; mul.f32 %f2,%f2,%f1;      // lut[byte]*scale
    \\  cvt.rn.f16.f32 %h0,%f2;
    \\  mul.wide.u32 %rd8,%r4,2; add.s64 %rd9,%rd3,%rd8; st.global.b16 [%rd9],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize ggml q8_0 weights to f16 (prefill GEMM scratch): one thread per
/// element, global element index e -> block e>>5 (34 B), quant i8 at +2+(e&31),
/// f16 d at +0; out[e] = f16(q*d). Blocks never straddle rows (cols % 32 == 0).
/// b0=in(q8_0), b1=out(f16). u0=total.
pub const dequant_q8_0_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_q8_0_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r6,%r4,5; mul.wide.u32 %rd4,%r6,34; add.s64 %rd5,%rd1,%rd4;  // &block
    \\  ld.global.b16 %h0,[%rd5]; cvt.f32.f16 %f1,%h0;                        // d
    \\  and.b32 %r7,%r4,31; cvt.u64.u32 %rd6,%r7; add.s64 %rd7,%rd5,%rd6;
    \\  ld.global.s8 %r8,[%rd7+2]; cvt.rn.f32.s32 %f2,%r8;                    // q
    \\  mul.f32 %f3,%f2,%f1;
    \\  cvt.rn.f16.f32 %h1,%f3;
    \\  mul.wide.u32 %rd8,%r4,2; add.s64 %rd9,%rd2,%rd8; st.global.b16 [%rd9],%h1;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize ggml q4_0 weights to f16 (Gemma 4 QAT). One thread per element
/// e: block b = e>>5 (18 B: f16 d + 16 nibble bytes); within-block j = e&31 maps
/// to the LOW nibble of qs[j] for j<16, else the HIGH nibble of qs[j-16];
/// out = f16((nibble - 8) * d).
pub const dequant_q4_0_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_q4_0_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<16>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r6,%r4,5; mul.wide.u32 %rd4,%r6,18; add.s64 %rd5,%rd1,%rd4;   // &block
    \\  ld.global.b16 %h0,[%rd5]; cvt.f32.f16 %f1,%h0;                         // d
    \\  and.b32 %r7,%r4,31;                                                    // j
    \\  setp.lt.u32 %p2,%r7,16;                                                // low nibble?
    \\  sub.u32 %r9,%r7,16; selp.b32 %r8,%r7,%r9,%p2;                          // qs byte index
    \\  cvt.u64.u32 %rd6,%r8; add.s64 %rd7,%rd5,%rd6;
    \\  ld.global.u8 %r10,[%rd7+2];                                            // qs byte
    \\  shr.u32 %r11,%r10,4; and.b32 %r12,%r10,15; selp.b32 %r13,%r12,%r11,%p2; // nibble
    \\  sub.s32 %r14,%r13,8; cvt.rn.f32.s32 %f2,%r14;                          // q - 8
    \\  mul.f32 %f3,%f2,%f1;
    \\  cvt.rn.f16.f32 %h1,%f3;
    \\  mul.wide.u32 %rd8,%r4,2; add.s64 %rd9,%rd2,%rd8; st.global.b16 [%rd9],%h1;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize ggml q4_k weights to f16: element e -> super-block e>>8 (144 B),
/// sub-block scale/min via get_scale_min_k4, nibble from byte
/// (j>>6)*32 + (j&31); out = f16(d*sc*q - dmin*m). cols % 256 == 0.
/// b0=in(q4_k), b1=out(f16). u0=total.
pub const dequant_q4_k_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_q4_k_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r6,%r4,8; mul.wide.u32 %rd4,%r6,144; add.s64 %rd5,%rd1,%rd4; // &super-block
    \\  ld.global.b16 %h0,[%rd5];   cvt.f32.f16 %f1,%h0;                      // d
    \\  ld.global.b16 %h1,[%rd5+2]; cvt.f32.f16 %f2,%h1;                      // dmin
    \\  and.b32 %r7,%r4,255;                                                  // j
    \\  shr.u32 %r8,%r7,5;                                                    // is
    \\  cvt.u64.u32 %rd6,%r8; add.s64 %rd7,%rd5,%rd6;                         // A: s[is] at [A+4]
    \\  setp.lt.u32 %p2,%r8,4; @%p2 bra LO;
    \\  ld.global.u8 %r10,[%rd7+8];
    \\  ld.global.u8 %r11,[%rd7];
    \\  ld.global.u8 %r12,[%rd7+4];
    \\  and.b32 %r13,%r10,15; shr.u32 %r14,%r11,6; shl.b32 %r14,%r14,4; or.b32 %r13,%r13,%r14; // sc
    \\  shr.u32 %r15,%r10,4; shr.u32 %r16,%r12,6; shl.b32 %r16,%r16,4; or.b32 %r15,%r15,%r16;  // m
    \\  bra DEQ;
    \\LO:
    \\  ld.global.u8 %r10,[%rd7+4];
    \\  ld.global.u8 %r11,[%rd7+8];
    \\  and.b32 %r13,%r10,63; and.b32 %r15,%r11,63;
    \\DEQ:
    \\  shr.u32 %r17,%r7,6; shl.b32 %r17,%r17,5; and.b32 %r18,%r7,31; add.u32 %r17,%r17,%r18;
    \\  cvt.u64.u32 %rd8,%r17; add.s64 %rd9,%rd5,%rd8;
    \\  ld.global.u8 %r19,[%rd9+16];                                          // nibble byte
    \\  shr.u32 %r20,%r7,5; and.b32 %r20,%r20,1; shl.b32 %r20,%r20,2;
    \\  shr.u32 %r19,%r19,%r20; and.b32 %r19,%r19,15;
    \\  cvt.rn.f32.u32 %f3,%r13; mul.f32 %f3,%f3,%f1;                         // d*sc
    \\  cvt.rn.f32.u32 %f4,%r15; mul.f32 %f4,%f4,%f2;                         // dmin*m
    \\  cvt.rn.f32.u32 %f5,%r19; mul.f32 %f5,%f5,%f3; sub.f32 %f5,%f5,%f4;
    \\  cvt.rn.f16.f32 %h2,%f5;
    \\  mul.wide.u32 %rd10,%r4,2; add.s64 %rd11,%rd2,%rd10; st.global.b16 [%rd11],%h2;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize ggml q5_k weights to f16: q4_k plus the 5th bit from qh[j&31]
/// (bit (j>>5)&7); blocks are 176 B, qh at +16, qs at +48.
/// b0=in(q5_k), b1=out(f16). u0=total.
pub const dequant_q5_k_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_q5_k_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<26>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b16 %h<4>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r6,%r4,8; mul.wide.u32 %rd4,%r6,176; add.s64 %rd5,%rd1,%rd4; // &super-block
    \\  ld.global.b16 %h0,[%rd5];   cvt.f32.f16 %f1,%h0;                      // d
    \\  ld.global.b16 %h1,[%rd5+2]; cvt.f32.f16 %f2,%h1;                      // dmin
    \\  and.b32 %r7,%r4,255;                                                  // j
    \\  shr.u32 %r8,%r7,5;                                                    // is
    \\  cvt.u64.u32 %rd6,%r8; add.s64 %rd7,%rd5,%rd6;
    \\  setp.lt.u32 %p2,%r8,4; @%p2 bra LO;
    \\  ld.global.u8 %r10,[%rd7+8];
    \\  ld.global.u8 %r11,[%rd7];
    \\  ld.global.u8 %r12,[%rd7+4];
    \\  and.b32 %r13,%r10,15; shr.u32 %r14,%r11,6; shl.b32 %r14,%r14,4; or.b32 %r13,%r13,%r14;
    \\  shr.u32 %r15,%r10,4; shr.u32 %r16,%r12,6; shl.b32 %r16,%r16,4; or.b32 %r15,%r15,%r16;
    \\  bra DEQ;
    \\LO:
    \\  ld.global.u8 %r10,[%rd7+4];
    \\  ld.global.u8 %r11,[%rd7+8];
    \\  and.b32 %r13,%r10,63; and.b32 %r15,%r11,63;
    \\DEQ:
    \\  and.b32 %r18,%r7,31;
    \\  cvt.u64.u32 %rd12,%r18; add.s64 %rd13,%rd5,%rd12;
    \\  ld.global.u8 %r21,[%rd13+16];                                         // qh byte
    \\  shr.u32 %r17,%r7,6; shl.b32 %r17,%r17,5; add.u32 %r17,%r17,%r18;
    \\  cvt.u64.u32 %rd8,%r17; add.s64 %rd9,%rd5,%rd8;
    \\  ld.global.u8 %r19,[%rd9+48];                                          // nibble byte
    \\  shr.u32 %r20,%r7,5; and.b32 %r22,%r20,1; shl.b32 %r22,%r22,2;
    \\  shr.u32 %r19,%r19,%r22; and.b32 %r19,%r19,15;
    \\  and.b32 %r23,%r20,7; shr.u32 %r21,%r21,%r23; and.b32 %r21,%r21,1; shl.b32 %r21,%r21,4;
    \\  add.u32 %r19,%r19,%r21;
    \\  cvt.rn.f32.u32 %f3,%r13; mul.f32 %f3,%f3,%f1;
    \\  cvt.rn.f32.u32 %f4,%r15; mul.f32 %f4,%f4,%f2;
    \\  cvt.rn.f32.u32 %f5,%r19; mul.f32 %f5,%f5,%f3; sub.f32 %f5,%f5,%f4;
    \\  cvt.rn.f16.f32 %h2,%f5;
    \\  mul.wide.u32 %rd10,%r4,2; add.s64 %rd11,%rd2,%rd10; st.global.b16 [%rd11],%h2;
    \\END:
    \\  ret;
    \\}
;

/// Dequantize ggml q6_k weights to f16: blocks are 210 B; out =
/// f16((d*sc[j>>4]) * (q - 32)) with the 4+2-bit q recombined from ql/qh.
/// b0=in(q6_k), b1=out(f16). u0=total.
pub const dequant_q6_k_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry dequant_q6_k_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<26>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r6,%r4,8; mul.wide.u32 %rd4,%r6,210; add.s64 %rd5,%rd1,%rd4; // &super-block
    \\  ld.global.b16 %h0,[%rd5+208]; cvt.f32.f16 %f1,%h0;                    // d
    \\  and.b32 %r7,%r4,255;                                                  // j
    \\  shr.u32 %r8,%r7,4; cvt.u64.u32 %rd6,%r8; add.s64 %rd7,%rd5,%rd6;
    \\  ld.global.s8 %r9,[%rd7+192]; cvt.rn.f32.s32 %f2,%r9;                  // sc
    \\  mul.f32 %f3,%f1,%f2;                                                  // d*sc
    \\  and.b32 %r10,%r7,31;                                                  // l
    \\  shr.u32 %r11,%r7,7;                                                   // half
    \\  shl.b32 %r12,%r11,6;                                                  // half*64
    \\  shr.u32 %r13,%r7,5; and.b32 %r14,%r13,1; shl.b32 %r14,%r14,5;
    \\  add.u32 %r12,%r12,%r14; add.u32 %r12,%r12,%r10;
    \\  cvt.u64.u32 %rd8,%r12; add.s64 %rd9,%rd5,%rd8;
    \\  ld.global.u8 %r15,[%rd9];                                             // ql byte
    \\  shl.b32 %r16,%r11,5; add.u32 %r16,%r16,%r10;
    \\  cvt.u64.u32 %rd10,%r16; add.s64 %rd11,%rd5,%rd10;
    \\  ld.global.u8 %r17,[%rd11+128];                                        // qh byte
    \\  shr.u32 %r18,%r7,6; and.b32 %r18,%r18,1; shl.b32 %r18,%r18,2;         // nibble shift
    \\  and.b32 %r19,%r13,3; shl.b32 %r19,%r19,1;                             // qh shift
    \\  shr.u32 %r15,%r15,%r18; and.b32 %r15,%r15,15;
    \\  shr.u32 %r17,%r17,%r19; and.b32 %r17,%r17,3; shl.b32 %r17,%r17,4;
    \\  or.b32 %r15,%r15,%r17; sub.s32 %r15,%r15,32; cvt.rn.f32.s32 %f4,%r15;
    \\  mul.f32 %f5,%f3,%f4;
    \\  cvt.rn.f16.f32 %h1,%f5;
    \\  mul.wide.u32 %rd12,%r4,2; add.s64 %rd13,%rd2,%rd12; st.global.b16 [%rd13],%h1;
    \\END:
    \\  ret;
    \\}
;

/// Convert f32 activations to f16, zero-padding rows past the real count so the
/// 128-row-padded GEMM sees clean pad rows. out[i] = i<u1 ? f16(in[i]) : 0.
/// b0=in(f32), b1=out(f16). u0=total(padded elems), u1=real elems. One thread/elem.
/// f16 -> f32 flat elementwise convert (p0 f16 in, p1 f32 out, u0 = count). Used
/// to bring the cuDNN fused-SDPA O (f16) back to the DiT's f32 attention buffer.
pub const f16_to_f32_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f16_to_f32(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<6>;
    \\  .reg .f32 %f<2>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd1,%rd3; ld.global.b16 %h0,[%rd4];
    \\  cvt.f32.f16 %f1,%h0;
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd5,%r4,4; add.s64 %rd6,%rd2,%rd5; st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Add a per-channel bias to an NHWC f16 conv output, writing f32: for a
/// [n][co] tile, out[u2 + i*co + j] = f16_in[i*co + j] + bias[j]. p0 f16 in,
/// p1 f32 bias[co], p2 f32 out, u0 = n*co, u1 = co, u2 = dst offset (elements).
pub const bias_add_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry bias_add_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; rem.u32 %r7,%r4,%r6;          // j = idx % co
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd2,%r4,2; add.s64 %rd3,%rd1,%rd2; ld.global.b16 %h0,[%rd3]; cvt.f32.f16 %f1,%h0;
    \\  ld.param.u64 %rd4,[p1]; cvta.to.global.u64 %rd4,%rd4;
    \\  mul.wide.u32 %rd5,%r7,4; add.s64 %rd6,%rd4,%rd5; ld.global.f32 %f2,[%rd6];
    \\  add.f32 %f3,%f1,%f2;
    \\  ld.param.u32 %r8,[u2]; add.s32 %r9,%r8,%r4;
    \\  ld.param.u64 %rd7,[p2]; cvta.to.global.u64 %rd7,%rd7;
    \\  mul.wide.u32 %rd8,%r9,4; add.s64 %rd9,%rd7,%rd8; st.global.f32 [%rd9],%f3;
    \\END:
    \\  ret;
    \\}
;

pub const f32_to_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32_to_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<8>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<8>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,2; add.s64 %rd4,%rd2,%rd3;    // &out[idx] (f16)
    \\  setp.ge.u32 %p2,%r4,%r6; @%p2 bra ZERO;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd5,%r4,4; add.s64 %rd6,%rd1,%rd5; ld.global.f32 %f1,[%rd6];
    \\  cvt.rn.f16.f32 %h0,%f1; st.global.b16 [%rd4],%h0; bra END;
    \\ZERO:
    \\  mov.b16 %h0,0x0000; st.global.b16 [%rd4],%h0;
    \\END:
    \\  ret;
    \\}
;

/// Quantize f32 into ggml q8_0 blocks: one thread per 32-element block.
/// b0 = src f32, b1 = dst blocks (34 B each), u0 = block count. Per block:
/// d = absmax/127 stored as f16, q[i] = rni(x[i]/d) as i8 — div.rn + cvt.rni
/// (round-to-nearest-EVEN) are bit-identical to the host packQ80, so a row
/// quantized on either side of a CPU-offload split produces the same bytes.
pub const f32_to_q8_0_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32_to_q8_0(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<12>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  ld.param.u64 %rd2,[p1]; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,128; add.s64 %rd4,%rd1,%rd3;   // src + b*32*4
    \\  mul.wide.u32 %rd5,%r4,34;  add.s64 %rd6,%rd2,%rd5;   // dst + b*34
    \\  // amax over the 32 elements
    \\  mov.f32 %f1,0f00000000;
    \\  mov.u32 %r6,0;
    \\AMAX:
    \\  cvt.u64.u32 %rd7,%r6; add.s64 %rd8,%rd4,%rd7;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd8];
    \\  abs.f32 %f2,%f2; max.f32 %f1,%f1,%f2;
    \\  abs.f32 %f3,%f3; max.f32 %f1,%f1,%f3;
    \\  abs.f32 %f4,%f4; max.f32 %f1,%f1,%f4;
    \\  abs.f32 %f5,%f5; max.f32 %f1,%f1,%f5;
    \\  add.u32 %r6,%r6,16; setp.lt.u32 %p2,%r6,128; @%p2 bra AMAX;
    \\  // d = amax/127 (f16 header), id = d == 0 ? 0 : 1/d
    \\  div.rn.f32 %f6,%f1,0f42FE0000;
    \\  cvt.rn.f16.f32 %h0,%f6; st.global.b16 [%rd6],%h0;
    \\  mov.f32 %f7,0f00000000;
    \\  setp.eq.f32 %p2,%f6,0f00000000; @%p2 bra QZ;
    \\  div.rn.f32 %f7,0f3F800000,%f6;
    \\QZ:
    \\  // quants: pairs -> u16 stores at +2 (2-byte aligned)
    \\  mov.u32 %r6,0;
    \\QLOOP:
    \\  cvt.u64.u32 %rd7,%r6; shl.b64 %rd7,%rd7,3; add.s64 %rd8,%rd4,%rd7;
    \\  ld.global.v2.f32 {%f2,%f3},[%rd8];
    \\  mul.f32 %f2,%f2,%f7; cvt.rni.s32.f32 %r7,%f2;
    \\  mul.f32 %f3,%f3,%f7; cvt.rni.s32.f32 %r8,%f3;
    \\  and.b32 %r7,%r7,255; shl.b32 %r8,%r8,8; or.b32 %r7,%r7,%r8;
    \\  cvt.u64.u32 %rd7,%r6; shl.b64 %rd7,%rd7,1; add.s64 %rd9,%rd6,%rd7;
    \\  st.global.u16 [%rd9+2],%r7;
    \\  add.u32 %r6,%r6,1; setp.lt.u32 %p2,%r6,16; @%p2 bra QLOOP;
    \\END:
    \\  ret;
    \\}
;

/// In-place scalar multiply: buf[i] *= f0, for i < u0. Gemma 4's per-layer
/// `out_scale` (a scalar the whole layer output is multiplied by).
pub const f32_scale_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32_scale(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<6>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b64 %rd<4>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u64 %rd1,[p0]; cvta.to.global.u64 %rd1,%rd1;
    \\  mul.wide.u32 %rd2,%r4,4; add.s64 %rd3,%rd1,%rd2;
    \\  ld.global.f32 %f1,[%rd3]; ld.param.f32 %f2,[f0]; mul.f32 %f1,%f1,%f2; st.global.f32 [%rd3],%f1;
    \\END:
    \\  ret;
    \\}
;

/// Decode-graph state module (CUDA graphs, LLM_PLAN.md M6): the per-token
/// dynamic values — sampled token id and cache position — live in the
/// g_state module global instead of kernel parameters, so the captured
/// decode graph replays unmodified: one 8-byte HtoD + one cuGraphLaunch per
/// token. Entries mirror their param-driven twins exactly (same math, same
/// order — byte-identical logits): embed_gather_s replaces the CPU
/// embedding gather + upload, rope_half_s takes pos0 from g_state[1],
/// kv_append_s replaces the KV-append memcpy, attn_split_s is the seq_q=1
/// flash-decode split with kv_len = g_state[1] + 1. g_state = [token, pos0].
pub const decode_state_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .global .align 8 .b32 g_state[2];
    \\
    \\// x[i] = f32(embed_bf16[g_state[0]*hidden + i]); b0=embed, b1=x, u0=hidden
    \\.visible .entry embed_gather_s(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<4>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.global.u32 %r6,[g_state];          // token id
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mad.lo.s32 %r7,%r6,%r5,%r4;           // elem = token*hidden + i
    \\  mul.wide.u32 %rd3,%r7,2; add.s64 %rd4,%rd1,%rd3; ld.global.u16 %r8,[%rd4];
    \\  shl.b32 %r9,%r8,16; mov.b32 %f1,%r9;  // bf16 -> f32
    \\  mul.wide.u32 %rd5,%r4,4; add.s64 %rd6,%rd2,%rd5; st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
    \\
    \\// x[i] = dequant_q8_0(embed[g_state[0]], i); b0=embed(q8_0), b1=x, u0=hidden
    \\.visible .entry embed_gather_q8_0(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<14>;
    \\  .reg .f32 %f<6>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<14>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.global.u32 %r6,[g_state];                                          // token id
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r7,%r5,5; mul.lo.u32 %r7,%r7,34;                             // row bytes
    \\  mul.wide.u32 %rd3,%r6,%r7; add.s64 %rd4,%rd1,%rd3;                    // row base
    \\  shr.u32 %r8,%r4,5; mul.lo.u32 %r8,%r8,34; cvt.u64.u32 %rd5,%r8; add.s64 %rd6,%rd4,%rd5;
    \\  ld.global.b16 %h0,[%rd6]; cvt.f32.f16 %f1,%h0;                        // d
    \\  and.b32 %r9,%r4,31; cvt.u64.u32 %rd7,%r9; add.s64 %rd8,%rd6,%rd7;
    \\  ld.global.s8 %r10,[%rd8+2]; cvt.rn.f32.s32 %f2,%r10;                  // q
    \\  mul.f32 %f3,%f2,%f1;
    \\  mul.wide.u32 %rd9,%r4,4; add.s64 %rd10,%rd2,%rd9; st.global.f32 [%rd10],%f3;
    \\END:
    \\  ret;
    \\}
    \\
    \\// x[i] = dequant_q4_k(embed[g_state[0]], i); b0=embed(q4_k), b1=x, u0=hidden
    \\.visible .entry embed_gather_q4_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<24>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<16>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.global.u32 %r6,[g_state];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r7,%r5,8; mul.lo.u32 %r7,%r7,144;
    \\  mul.wide.u32 %rd3,%r6,%r7; add.s64 %rd4,%rd1,%rd3;                    // row base
    \\  shr.u32 %r8,%r4,8; mul.lo.u32 %r8,%r8,144; cvt.u64.u32 %rd5,%r8; add.s64 %rd6,%rd4,%rd5;
    \\  ld.global.b16 %h0,[%rd6];   cvt.f32.f16 %f1,%h0;                      // d
    \\  ld.global.b16 %h1,[%rd6+2]; cvt.f32.f16 %f2,%h1;                      // dmin
    \\  and.b32 %r9,%r4,255;                                                  // j
    \\  shr.u32 %r10,%r9,5; cvt.u64.u32 %rd7,%r10; add.s64 %rd8,%rd6,%rd7;
    \\  setp.lt.u32 %p2,%r10,4; @%p2 bra LO;
    \\  ld.global.u8 %r11,[%rd8+8];
    \\  ld.global.u8 %r12,[%rd8];
    \\  ld.global.u8 %r13,[%rd8+4];
    \\  and.b32 %r14,%r11,15; shr.u32 %r15,%r12,6; shl.b32 %r15,%r15,4; or.b32 %r14,%r14,%r15;
    \\  shr.u32 %r16,%r11,4; shr.u32 %r17,%r13,6; shl.b32 %r17,%r17,4; or.b32 %r16,%r16,%r17;
    \\  bra DEQ;
    \\LO:
    \\  ld.global.u8 %r11,[%rd8+4];
    \\  ld.global.u8 %r12,[%rd8+8];
    \\  and.b32 %r14,%r11,63; and.b32 %r16,%r12,63;
    \\DEQ:
    \\  shr.u32 %r18,%r9,6; shl.b32 %r18,%r18,5; and.b32 %r19,%r9,31; add.u32 %r18,%r18,%r19;
    \\  cvt.u64.u32 %rd9,%r18; add.s64 %rd10,%rd6,%rd9;
    \\  ld.global.u8 %r20,[%rd10+16];
    \\  shr.u32 %r21,%r9,5; and.b32 %r21,%r21,1; shl.b32 %r21,%r21,2;
    \\  shr.u32 %r20,%r20,%r21; and.b32 %r20,%r20,15;
    \\  cvt.rn.f32.u32 %f3,%r14; mul.f32 %f3,%f3,%f1;
    \\  cvt.rn.f32.u32 %f4,%r16; mul.f32 %f4,%f4,%f2;
    \\  cvt.rn.f32.u32 %f5,%r20; mul.f32 %f5,%f5,%f3; sub.f32 %f5,%f5,%f4;
    \\  mul.wide.u32 %rd11,%r4,4; add.s64 %rd12,%rd2,%rd11; st.global.f32 [%rd12],%f5;
    \\END:
    \\  ret;
    \\}
    \\
    \\// x[i] = dequant_q5_k(embed[g_state[0]], i); b0=embed(q5_k), b1=x, u0=hidden
    \\.visible .entry embed_gather_q5_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<28>;
    \\  .reg .f32 %f<10>;
    \\  .reg .b16 %h<3>;
    \\  .reg .b64 %rd<18>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.global.u32 %r6,[g_state];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r7,%r5,8; mul.lo.u32 %r7,%r7,176;
    \\  mul.wide.u32 %rd3,%r6,%r7; add.s64 %rd4,%rd1,%rd3;
    \\  shr.u32 %r8,%r4,8; mul.lo.u32 %r8,%r8,176; cvt.u64.u32 %rd5,%r8; add.s64 %rd6,%rd4,%rd5;
    \\  ld.global.b16 %h0,[%rd6];   cvt.f32.f16 %f1,%h0;
    \\  ld.global.b16 %h1,[%rd6+2]; cvt.f32.f16 %f2,%h1;
    \\  and.b32 %r9,%r4,255;
    \\  shr.u32 %r10,%r9,5; cvt.u64.u32 %rd7,%r10; add.s64 %rd8,%rd6,%rd7;
    \\  setp.lt.u32 %p2,%r10,4; @%p2 bra LO;
    \\  ld.global.u8 %r11,[%rd8+8];
    \\  ld.global.u8 %r12,[%rd8];
    \\  ld.global.u8 %r13,[%rd8+4];
    \\  and.b32 %r14,%r11,15; shr.u32 %r15,%r12,6; shl.b32 %r15,%r15,4; or.b32 %r14,%r14,%r15;
    \\  shr.u32 %r16,%r11,4; shr.u32 %r17,%r13,6; shl.b32 %r17,%r17,4; or.b32 %r16,%r16,%r17;
    \\  bra DEQ;
    \\LO:
    \\  ld.global.u8 %r11,[%rd8+4];
    \\  ld.global.u8 %r12,[%rd8+8];
    \\  and.b32 %r14,%r11,63; and.b32 %r16,%r12,63;
    \\DEQ:
    \\  and.b32 %r19,%r9,31;
    \\  cvt.u64.u32 %rd13,%r19; add.s64 %rd14,%rd6,%rd13;
    \\  ld.global.u8 %r22,[%rd14+16];                                         // qh byte
    \\  shr.u32 %r18,%r9,6; shl.b32 %r18,%r18,5; add.u32 %r18,%r18,%r19;
    \\  cvt.u64.u32 %rd9,%r18; add.s64 %rd10,%rd6,%rd9;
    \\  ld.global.u8 %r20,[%rd10+48];                                         // nibble byte
    \\  shr.u32 %r21,%r9,5; and.b32 %r23,%r21,1; shl.b32 %r23,%r23,2;
    \\  shr.u32 %r20,%r20,%r23; and.b32 %r20,%r20,15;
    \\  and.b32 %r24,%r21,7; shr.u32 %r22,%r22,%r24; and.b32 %r22,%r22,1; shl.b32 %r22,%r22,4;
    \\  add.u32 %r20,%r20,%r22;
    \\  cvt.rn.f32.u32 %f3,%r14; mul.f32 %f3,%f3,%f1;
    \\  cvt.rn.f32.u32 %f4,%r16; mul.f32 %f4,%f4,%f2;
    \\  cvt.rn.f32.u32 %f5,%r20; mul.f32 %f5,%f5,%f3; sub.f32 %f5,%f5,%f4;
    \\  mul.wide.u32 %rd11,%r4,4; add.s64 %rd12,%rd2,%rd11; st.global.f32 [%rd12],%f5;
    \\END:
    \\  ret;
    \\}
    \\
    \\// x[i] = dequant_q6_k(embed[g_state[0]], i); b0=embed(q6_k), b1=x, u0=hidden
    \\.visible .entry embed_gather_q6_k(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<26>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<18>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.global.u32 %r6,[g_state];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  shr.u32 %r7,%r5,8; mul.lo.u32 %r7,%r7,210;
    \\  mul.wide.u32 %rd3,%r6,%r7; add.s64 %rd4,%rd1,%rd3;
    \\  shr.u32 %r8,%r4,8; mul.lo.u32 %r8,%r8,210; cvt.u64.u32 %rd5,%r8; add.s64 %rd6,%rd4,%rd5;
    \\  ld.global.b16 %h0,[%rd6+208]; cvt.f32.f16 %f1,%h0;                    // d
    \\  and.b32 %r9,%r4,255;                                                  // j
    \\  shr.u32 %r10,%r9,4; cvt.u64.u32 %rd7,%r10; add.s64 %rd8,%rd6,%rd7;
    \\  ld.global.s8 %r11,[%rd8+192]; cvt.rn.f32.s32 %f2,%r11;                // sc
    \\  mul.f32 %f3,%f1,%f2;
    \\  and.b32 %r12,%r9,31;                                                  // l
    \\  shr.u32 %r13,%r9,7;                                                   // half
    \\  shl.b32 %r14,%r13,6;
    \\  shr.u32 %r15,%r9,5; and.b32 %r16,%r15,1; shl.b32 %r16,%r16,5;
    \\  add.u32 %r14,%r14,%r16; add.u32 %r14,%r14,%r12;
    \\  cvt.u64.u32 %rd9,%r14; add.s64 %rd10,%rd6,%rd9;
    \\  ld.global.u8 %r17,[%rd10];                                            // ql byte
    \\  shl.b32 %r18,%r13,5; add.u32 %r18,%r18,%r12;
    \\  cvt.u64.u32 %rd11,%r18; add.s64 %rd12,%rd6,%rd11;
    \\  ld.global.u8 %r19,[%rd12+128];                                        // qh byte
    \\  shr.u32 %r20,%r9,6; and.b32 %r20,%r20,1; shl.b32 %r20,%r20,2;
    \\  and.b32 %r21,%r15,3; shl.b32 %r21,%r21,1;
    \\  shr.u32 %r17,%r17,%r20; and.b32 %r17,%r17,15;
    \\  shr.u32 %r19,%r19,%r21; and.b32 %r19,%r19,3; shl.b32 %r19,%r19,4;
    \\  or.b32 %r17,%r17,%r19; sub.s32 %r17,%r17,32; cvt.rn.f32.s32 %f4,%r17;
    \\  mul.f32 %f5,%f3,%f4;
    \\  mul.wide.u32 %rd13,%r4,4; add.s64 %rd14,%rd2,%rd13; st.global.f32 [%rd14],%f5;
    \\END:
    \\  ret;
    \\}
    \\
    \\// dst[u2 + g_state[1]*u1 + i] = src[i]; b0=src, b1=dst, u0=count,
    \\// u1=row stride, u2=base offset (KV appends and decode-graph tap rows).
    \\.visible .entry kv_append_s(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];
    \\  ld.param.u32 %r9,[u2];
    \\  ld.global.u32 %r7,[g_state+4];        // pos0
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f1,[%rd4];
    \\  mad.lo.s32 %r8,%r7,%r6,%r4;           // dst = base + pos0*stride + i
    \\  add.u32 %r8,%r8,%r9;
    \\  mul.wide.u32 %rd5,%r8,4; add.s64 %rd6,%rd2,%rd5; st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
    \\
    \\// rope_half with pos0 = g_state[1] (otherwise identical to rope_half).
    \\.visible .entry rope_half_s(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<20>;
    \\  .reg .f32 %f<8>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];               // half
    \\  ld.param.u32 %r7,[u2];               // sin_off
    \\  ld.param.u32 %r8,[u3];               // n_heads
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd3,[p2];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd3,%rd3;
    \\  rem.u32 %r9,%r4,%r6;                  // pair = idx % half
    \\  div.u32 %r10,%r4,%r6;                 // hp = idx/half = pos*n_heads + head
    \\  mul.lo.s32 %r11,%r6,%r8;              // half*n_heads
    \\  div.u32 %r12,%r4,%r11;                // pos
    \\  ld.global.u32 %r18,[g_state+4]; add.u32 %r12,%r12,%r18; // pos += pos0 (device state)
    \\  mad.lo.s32 %r13,%r12,%r6,%r9;         // cos idx = pos*half + pair
    \\  mul.wide.u32 %rd4,%r13,4; add.s64 %rd5,%rd3,%rd4; ld.global.f32 %f1,[%rd5]; // cos
    \\  add.s32 %r14,%r13,%r7;                // + sin_off
    \\  mul.wide.u32 %rd6,%r14,4; add.s64 %rd7,%rd3,%rd6; ld.global.f32 %f2,[%rd7]; // sin
    \\  shl.b32 %r15,%r6,1;                   // head_dim = 2*half
    \\  mad.lo.s32 %r16,%r10,%r15,%r9;        // lo_idx = hp*head_dim + pair
    \\  add.s32 %r17,%r16,%r6;                // hi_idx = lo_idx + half
    \\  mul.wide.u32 %rd8,%r16,4; add.s64 %rd9,%rd1,%rd8; ld.global.f32 %f3,[%rd9];   // lo
    \\  mul.wide.u32 %rd10,%r17,4; add.s64 %rd11,%rd1,%rd10; ld.global.f32 %f4,[%rd11]; // hi
    \\  mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6; st.global.f32 [%rd9],%f5;  // lo*cos - hi*sin
    \\  mul.f32 %f6,%f4,%f1; fma.rn.f32 %f6,%f3,%f2,%f6; st.global.f32 [%rd11],%f6;               // hi*cos + lo*sin
    \\END:
    \\  ret;
    \\}
    \\
    \\// attn_split for the single decode query with kv_len = g_state[1] + 1
    \\// (otherwise identical math/order to attn_split at seq_q = 1).
    \\.visible .entry attn_split_s(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<32>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = pos0 + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps
    \\  setp.ge.u32 %p1,%r27,%r7; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r10,%r27,%r26;               // h
    \\  rem.u32 %r21,%r27,%r26;               // split i
    \\  add.u32 %r22,%r5,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk
    \\  mul.lo.s32 %r17,%r21,%r22;            // kv0
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment: q[h*hd + lane*4 ..][4]
    \\  mul.lo.s32 %r14,%r10,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,4; add.s64 %rd10,%rd2,%rd9;
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd10];
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V fragment (same offsets)
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd11];
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
    \\
    \\// attn_split_h256 (qwen35 decode, 8 dims/lane) with kv_len =
    \\// g_state[1] + 1 instead of u0 (otherwise identical math/order;
    \\// launch with u5 = seq_q = 1).
    \\.visible .entry attn_split_h256_s(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<32>;
    \\  .reg .f32 %f<64>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = len + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=256)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  // Sliding window (f1, 0 = full causal): kv_start = max(0, kv_len - window).
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment: q[(t*heads + h)*hd + lane*8 ..][8]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,3; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  ld.global.v4.f32 {%f32,%f33,%f34,%f35},[%rd6+16];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\JLOOPH:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JDH;
    \\  // kv row fragment base: ((j*kv_heads + kvh)*hd + lane*8)
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,4; add.s64 %rd10,%rd2,%rd9;
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd10];
    \\  ld.global.v4.f32 {%f36,%f37,%f38,%f39},[%rd10+16];
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  fma.rn.f32 %f6,%f32,%f36,%f6; fma.rn.f32 %f6,%f33,%f37,%f6; fma.rn.f32 %f6,%f34,%f38,%f6; fma.rn.f32 %f6,%f35,%f39,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V fragment (same offsets)
    \\  ld.global.v4.f32 {%f24,%f25,%f26,%f27},[%rd11];
    \\  ld.global.v4.f32 {%f36,%f37,%f38,%f39},[%rd11+16];
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  mul.f32 %f40,%f40,%f8; fma.rn.f32 %f40,%f9,%f36,%f40;
    \\  mul.f32 %f41,%f41,%f8; fma.rn.f32 %f41,%f9,%f37,%f41;
    \\  mul.f32 %f42,%f42,%f8; fma.rn.f32 %f42,%f9,%f38,%f42;
    \\  mul.f32 %f43,%f43,%f8; fma.rn.f32 %f43,%f9,%f39,%f43;
    \\  add.u32 %r17,%r17,1; bra JLOOPH;
    \\JDH:
    \\  // scratch row = warp*(hd+4): lane 0 stores m,d; every lane its acc8
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACCH;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACCH:
    \\  shl.b32 %r29,%r28,5; add.u32 %r29,%r29,16;   // byte off = 16 + lane*32
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\  st.global.v4.f32 [%rd20+16],{%f40,%f41,%f42,%f43};
    \\END:
    \\  ret;
    \\}
    \\
    \\// f16-KV variant of kv_append_s: read f32 src (*4), convert, store f16 (*2).
    \\.visible .entry kv_append_s_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<12>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];
    \\  ld.param.u32 %r9,[u2];
    \\  ld.global.u32 %r7,[g_state+4];        // pos0
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,4; add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f1,[%rd4];   // src f32
    \\  mad.lo.s32 %r8,%r7,%r6,%r4;           // dst = base + pos0*stride + i
    \\  add.u32 %r8,%r8,%r9;
    \\  cvt.rn.f16.f32 %h0,%f1;
    \\  mul.wide.u32 %rd5,%r8,2; add.s64 %rd6,%rd2,%rd5; st.global.b16 [%rd6],%h0;   // dst f16
    \\END:
    \\  ret;
    \\}
    \\
    \\// f16-KV variant of attn_split_s (hd128): K/V loaded as v2.u32 (4 halfs) and
    \\// widened to f32; *2 stride. q/scratch stay f32. kv_len = g_state[1] + 1.
    \\.visible .entry attn_split_s_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<34>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = pos0 + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps
    \\  setp.ge.u32 %p1,%r27,%r7; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r10,%r27,%r26;               // h
    \\  rem.u32 %r21,%r27,%r26;               // split i
    \\  add.u32 %r22,%r5,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk
    \\  mul.lo.s32 %r17,%r21,%r22;            // kv0
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[h*hd + lane*4 ..][4]
    \\  mul.lo.s32 %r14,%r10,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,2; add.s64 %rd10,%rd2,%rd9;   // f16 K (*2)
    \\  ld.global.v2.u32 {%r32,%r33},[%rd10];                // 4 halfs
    \\  mov.b32 {%hs0,%hs1},%r32; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r33; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // f16 V (same offsets)
    \\  ld.global.v2.u32 {%r32,%r33},[%rd11];
    \\  mov.b32 {%hs0,%hs1},%r32; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r33; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
    \\
    \\// f16-KV variant of attn_split_h256_s (hd256): K/V as v4.u32 (8 halfs) ->
    \\// f32, *2 stride; kv_len = g_state[1] + 1. q/scratch stay f32.
    \\.visible .entry attn_split_h256_s_f16(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<44>;
    \\  .reg .f32 %f<64>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = len + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=256)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[(t*heads + h)*hd + lane*8 ..][8]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,3; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  ld.global.v4.f32 {%f32,%f33,%f34,%f35},[%rd6+16];
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\JLOOPH:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JDH;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  mul.wide.u32 %rd9,%r18,2; add.s64 %rd10,%rd2,%rd9;   // f16 K (*2)
    \\  ld.global.v4.u32 {%r40,%r41,%r42,%r43},[%rd10];      // 8 halfs
    \\  mov.b32 {%hs0,%hs1},%r40; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r41; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r42; cvt.f32.f16 %f36,%hs0; cvt.f32.f16 %f37,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r43; cvt.f32.f16 %f38,%hs0; cvt.f32.f16 %f39,%hs1;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  fma.rn.f32 %f6,%f32,%f36,%f6; fma.rn.f32 %f6,%f33,%f37,%f6; fma.rn.f32 %f6,%f34,%f38,%f6; fma.rn.f32 %f6,%f35,%f39,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // f16 V (same offsets)
    \\  ld.global.v4.u32 {%r40,%r41,%r42,%r43},[%rd11];
    \\  mov.b32 {%hs0,%hs1},%r40; cvt.f32.f16 %f24,%hs0; cvt.f32.f16 %f25,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r41; cvt.f32.f16 %f26,%hs0; cvt.f32.f16 %f27,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r42; cvt.f32.f16 %f36,%hs0; cvt.f32.f16 %f37,%hs1;
    \\  mov.b32 {%hs0,%hs1},%r43; cvt.f32.f16 %f38,%hs0; cvt.f32.f16 %f39,%hs1;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  mul.f32 %f40,%f40,%f8; fma.rn.f32 %f40,%f9,%f36,%f40;
    \\  mul.f32 %f41,%f41,%f8; fma.rn.f32 %f41,%f9,%f37,%f41;
    \\  mul.f32 %f42,%f42,%f8; fma.rn.f32 %f42,%f9,%f38,%f42;
    \\  mul.f32 %f43,%f43,%f8; fma.rn.f32 %f43,%f9,%f39,%f43;
    \\  add.u32 %r17,%r17,1; bra JLOOPH;
    \\JDH:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACCH;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACCH:
    \\  shl.b32 %r29,%r28,5; add.u32 %r29,%r29,16;   // byte off = 16 + lane*32
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\  st.global.v4.f32 [%rd20+16],{%f40,%f41,%f42,%f43};
    \\END:
    \\  ret;
    \\}
    \\
    \\// q8_0-KV variant of kv_append_s: ONE THREAD PER 32-ELEM BLOCK (u0 = block
    \\// count, not elements). u1 = row stride and u2 = base offset stay ELEMENT
    \\// counts (both block multiples). Quantization (div.rn + cvt.rni, ties to
    \\// even) is bit-identical to the host packQ80 and f32_to_q8_0.
    \\.visible .entry kv_append_s_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<3>;
    \\  .reg .b32 %r<16>;
    \\  .reg .f32 %f<12>;
    \\  .reg .b16 %h<2>;
    \\  .reg .b64 %rd<12>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1];                // stride (elements)
    \\  ld.param.u32 %r9,[u2];                // base (elements)
    \\  ld.global.u32 %r7,[g_state+4];        // pos0
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  mul.wide.u32 %rd3,%r4,128; add.s64 %rd4,%rd1,%rd3;   // src + b*32*4
    \\  shr.u32 %r10,%r6,5; shr.u32 %r11,%r9,5;
    \\  mad.lo.s32 %r12,%r7,%r10,%r11; add.u32 %r12,%r12,%r4; // dst block index
    \\  mul.wide.u32 %rd5,%r12,34; add.s64 %rd6,%rd2,%rd5;
    \\  mov.f32 %f1,0f00000000;
    \\  mov.u32 %r6,0;
    \\AMAX:
    \\  cvt.u64.u32 %rd7,%r6; add.s64 %rd8,%rd4,%rd7;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd8];
    \\  abs.f32 %f2,%f2; max.f32 %f1,%f1,%f2;
    \\  abs.f32 %f3,%f3; max.f32 %f1,%f1,%f3;
    \\  abs.f32 %f4,%f4; max.f32 %f1,%f1,%f4;
    \\  abs.f32 %f5,%f5; max.f32 %f1,%f1,%f5;
    \\  add.u32 %r6,%r6,16; setp.lt.u32 %p2,%r6,128; @%p2 bra AMAX;
    \\  div.rn.f32 %f6,%f1,0f42FE0000;        // d = amax/127
    \\  cvt.rn.f16.f32 %h0,%f6; st.global.b16 [%rd6],%h0;
    \\  mov.f32 %f7,0f00000000;
    \\  setp.eq.f32 %p2,%f6,0f00000000; @%p2 bra QZ;
    \\  div.rn.f32 %f7,0f3F800000,%f6;        // id = 1/d
    \\QZ:
    \\  mov.u32 %r6,0;
    \\QLOOP:
    \\  cvt.u64.u32 %rd7,%r6; shl.b64 %rd7,%rd7,3; add.s64 %rd8,%rd4,%rd7;
    \\  ld.global.v2.f32 {%f2,%f3},[%rd8];
    \\  mul.f32 %f2,%f2,%f7; cvt.rni.s32.f32 %r7,%f2;
    \\  mul.f32 %f3,%f3,%f7; cvt.rni.s32.f32 %r8,%f3;
    \\  and.b32 %r7,%r7,255; shl.b32 %r8,%r8,8; or.b32 %r7,%r7,%r8;
    \\  cvt.u64.u32 %rd7,%r6; shl.b64 %rd7,%rd7,1; add.s64 %rd9,%rd6,%rd7;
    \\  st.global.u16 [%rd9+2],%r7;
    \\  add.u32 %r6,%r6,1; setp.lt.u32 %p2,%r6,16; @%p2 bra QLOOP;
    \\END:
    \\  ret;
    \\}
    \\
    \\// q8_0-KV variant of attn_split_s (hd128): each lane's 4-dim K/V fragment
    \\// sits inside one 34-byte block; load the f16 scale + two u16 quant pairs,
    \\// sign-extend, multiply. kv_len = g_state[1] + 1. q/scratch stay f32.
    \\.visible .entry attn_split_s_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<40>;
    \\  .reg .f32 %f<40>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = pos0 + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps
    \\  setp.ge.u32 %p1,%r27,%r7; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=128)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r10,%r27,%r26;               // h
    \\  rem.u32 %r21,%r27,%r26;               // split i
    \\  add.u32 %r22,%r5,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk
    \\  mul.lo.s32 %r17,%r21,%r22;            // kv0
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[h*hd + lane*4 ..][4]
    \\  mul.lo.s32 %r14,%r10,%r9; shl.b32 %r15,%r28,2; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  and.b32 %r34,%r15,31; add.u32 %r34,%r34,2; cvt.u64.u32 %rd21,%r34; // quant byte off
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000; // acc
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JD;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  shr.u32 %r35,%r18,5; mul.wide.u32 %rd9,%r35,34; add.s64 %rd10,%rd2,%rd9;   // K block
    \\  ld.global.b16 %hs0,[%rd10]; cvt.f32.f16 %f31,%hs0;
    \\  add.s64 %rd12,%rd10,%rd21;
    \\  ld.global.u16 %r32,[%rd12];
    \\  shl.b32 %r36,%r32,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f24,%r36; mul.f32 %f24,%f24,%f31;
    \\  shl.b32 %r36,%r32,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f25,%r36; mul.f32 %f25,%f25,%f31;
    \\  ld.global.u16 %r33,[%rd12+2];
    \\  shl.b32 %r36,%r33,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f26,%r36; mul.f32 %f26,%f26,%f31;
    \\  shl.b32 %r36,%r33,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f27,%r36; mul.f32 %f27,%f27,%f31;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V block (same offsets)
    \\  ld.global.b16 %hs0,[%rd11]; cvt.f32.f16 %f31,%hs0;
    \\  add.s64 %rd12,%rd11,%rd21;
    \\  ld.global.u16 %r32,[%rd12];
    \\  shl.b32 %r36,%r32,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f24,%r36; mul.f32 %f24,%f24,%f31;
    \\  shl.b32 %r36,%r32,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f25,%r36; mul.f32 %f25,%f25,%f31;
    \\  ld.global.u16 %r33,[%rd12+2];
    \\  shl.b32 %r36,%r33,24; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f26,%r36; mul.f32 %f26,%f26,%f31;
    \\  shl.b32 %r36,%r33,16; shr.s32 %r36,%r36,24; cvt.rn.f32.s32 %f27,%r36; mul.f32 %f27,%f27,%f31;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  add.u32 %r17,%r17,1; bra JLOOP;
    \\JD:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACC;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACC:
    \\  shl.b32 %r29,%r28,4; add.u32 %r29,%r29,16;   // byte off = 16 + lane*16
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\END:
    \\  ret;
    \\}
    \\
    \\// q8_0-KV variant of attn_split_h256_s (hd256): each lane's 8-dim fragment
    \\// sits inside one 34-byte block; scale + four u16 quant pairs -> f32.
    \\// kv_len = g_state[1] + 1. q/scratch stay f32.
    \\.visible .entry attn_split_h256_s_q8(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<5>;
    \\  .reg .b32 %r<44>;
    \\  .reg .f32 %f<64>;
    \\  .reg .b16 %hs<2>;
    \\  .reg .b64 %rd<24>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
    \\  mad.lo.s32 %r4,%r1,%r2,%r3;           // global thread
    \\  shr.u32 %r27,%r4,5;                   // warp = idx/32
    \\  and.b32 %r28,%r4,31;                  // lane
    \\  ld.global.u32 %r5,[g_state+4]; add.u32 %r5,%r5,1; // kv_len = len + 1
    \\  ld.param.u32 %r6,[u1];                // heads
    \\  ld.param.u32 %r26,[u4];               // nsplit
    \\  ld.param.u32 %r30,[u5];               // seq_q
    \\  mul.lo.s32 %r7,%r6,%r26;              // heads*nsplit warps per query
    \\  mul.lo.s32 %r31,%r7,%r30;
    \\  setp.ge.u32 %p1,%r27,%r31; @%p1 bra END;
    \\  ld.param.u32 %r8,[u2];                // kv_heads
    \\  ld.param.u32 %r9,[u3];                // hd (=256)
    \\  ld.param.f32 %f1,[f0];                // scale
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1]; ld.param.u64 %rd3,[p2]; ld.param.u64 %rd4,[p3];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3; cvta.to.global.u64 %rd4,%rd4;
    \\  div.u32 %r31,%r27,%r7;                // query t
    \\  rem.u32 %r2,%r27,%r7;                 // warp within query
    \\  add.u32 %r5,%r5,%r31;                 // this query's kv len (causal)
    \\  div.u32 %r10,%r2,%r26;                // h
    \\  rem.u32 %r21,%r2,%r26;                // split i
    \\  ld.param.f32 %f30,[f1]; cvt.rzi.u32.f32 %r11,%f30;
    \\  mov.u32 %r16,0;                       // kv_start
    \\  setp.eq.u32 %p4,%r11,0; @%p4 bra NOWIN;
    \\  setp.le.u32 %p4,%r5,%r11; @%p4 bra NOWIN;
    \\  sub.u32 %r16,%r5,%r11;                // kv_start = kv_len - window
    \\NOWIN:
    \\  sub.u32 %r22,%r5,%r16;                // span = kv_len - kv_start
    \\  add.u32 %r22,%r22,%r26; sub.u32 %r22,%r22,1; div.u32 %r22,%r22,%r26; // chunk = ceil(span/nsplit)
    \\  mad.lo.s32 %r17,%r21,%r22,%r16;       // kv0 = kv_start + split_i*chunk
    \\  add.u32 %r23,%r17,%r22; min.u32 %r23,%r23,%r5; // kv1
    \\  div.u32 %r12,%r6,%r8;                 // group
    \\  div.u32 %r13,%r10,%r12;               // kv head
    \\  // q fragment (f32): q[(t*heads + h)*hd + lane*8 ..][8]
    \\  mad.lo.s32 %r14,%r31,%r6,%r10; mul.lo.s32 %r14,%r14,%r9; shl.b32 %r15,%r28,3; add.u32 %r14,%r14,%r15;
    \\  mul.wide.u32 %rd5,%r14,4; add.s64 %rd6,%rd1,%rd5;
    \\  ld.global.v4.f32 {%f2,%f3,%f4,%f5},[%rd6];
    \\  ld.global.v4.f32 {%f32,%f33,%f34,%f35},[%rd6+16];
    \\  and.b32 %r34,%r15,31; add.u32 %r34,%r34,2; cvt.u64.u32 %rd21,%r34; // quant byte off
    \\  mov.f32 %f10,0fFF800000;              // m
    \\  mov.f32 %f11,0f00000000;              // d
    \\  mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000; mov.f32 %f23,0f00000000;
    \\  mov.f32 %f40,0f00000000; mov.f32 %f41,0f00000000; mov.f32 %f42,0f00000000; mov.f32 %f43,0f00000000;
    \\JLOOPH:
    \\  setp.ge.u32 %p2,%r17,%r23; @%p2 bra JDH;
    \\  mad.lo.s32 %r18,%r17,%r8,%r13; mul.lo.s32 %r18,%r18,%r9; add.u32 %r18,%r18,%r15;
    \\  shr.u32 %r35,%r18,5; mul.wide.u32 %rd9,%r35,34; add.s64 %rd10,%rd2,%rd9;   // K block
    \\  ld.global.b16 %hs0,[%rd10]; cvt.f32.f16 %f50,%hs0;
    \\  add.s64 %rd12,%rd10,%rd21;
    \\  ld.global.u16 %r36,[%rd12];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f24,%r38; mul.f32 %f24,%f24,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f25,%r38; mul.f32 %f25,%f25,%f50;
    \\  ld.global.u16 %r36,[%rd12+2];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f26,%r38; mul.f32 %f26,%f26,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f27,%r38; mul.f32 %f27,%f27,%f50;
    \\  ld.global.u16 %r36,[%rd12+4];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f36,%r38; mul.f32 %f36,%f36,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f37,%r38; mul.f32 %f37,%f37,%f50;
    \\  ld.global.u16 %r36,[%rd12+6];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f38,%r38; mul.f32 %f38,%f38,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f39,%r38; mul.f32 %f39,%f39,%f50;
    \\  mul.f32 %f6,%f2,%f24; fma.rn.f32 %f6,%f3,%f25,%f6; fma.rn.f32 %f6,%f4,%f26,%f6; fma.rn.f32 %f6,%f5,%f27,%f6;
    \\  fma.rn.f32 %f6,%f32,%f36,%f6; fma.rn.f32 %f6,%f33,%f37,%f6; fma.rn.f32 %f6,%f34,%f38,%f6; fma.rn.f32 %f6,%f35,%f39,%f6;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,16,0x1f,0xffffffff; mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,8,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,4,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,2,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mov.b32 %r19,%f6; shfl.sync.bfly.b32 %r20,%r19,1,0x1f,0xffffffff;  mov.b32 %f7,%r20; add.f32 %f6,%f6,%f7;
    \\  mul.f32 %f6,%f6,%f1;                  // s
    \\  max.f32 %f12,%f10,%f6;                // m2
    \\  sub.f32 %f8,%f10,%f12; mul.f32 %f8,%f8,0f3FB8AA3B; ex2.approx.f32 %f8,%f8;  // corr
    \\  sub.f32 %f9,%f6,%f12; mul.f32 %f9,%f9,0f3FB8AA3B; ex2.approx.f32 %f9,%f9;   // p
    \\  mul.f32 %f11,%f11,%f8; add.f32 %f11,%f11,%f9;
    \\  mov.f32 %f10,%f12;
    \\  add.s64 %rd11,%rd3,%rd9;              // V block (same offsets)
    \\  ld.global.b16 %hs0,[%rd11]; cvt.f32.f16 %f50,%hs0;
    \\  add.s64 %rd12,%rd11,%rd21;
    \\  ld.global.u16 %r36,[%rd12];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f24,%r38; mul.f32 %f24,%f24,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f25,%r38; mul.f32 %f25,%f25,%f50;
    \\  ld.global.u16 %r36,[%rd12+2];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f26,%r38; mul.f32 %f26,%f26,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f27,%r38; mul.f32 %f27,%f27,%f50;
    \\  ld.global.u16 %r36,[%rd12+4];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f36,%r38; mul.f32 %f36,%f36,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f37,%r38; mul.f32 %f37,%f37,%f50;
    \\  ld.global.u16 %r36,[%rd12+6];
    \\  shl.b32 %r38,%r36,24; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f38,%r38; mul.f32 %f38,%f38,%f50;
    \\  shl.b32 %r38,%r36,16; shr.s32 %r38,%r38,24; cvt.rn.f32.s32 %f39,%r38; mul.f32 %f39,%f39,%f50;
    \\  mul.f32 %f20,%f20,%f8; fma.rn.f32 %f20,%f9,%f24,%f20;
    \\  mul.f32 %f21,%f21,%f8; fma.rn.f32 %f21,%f9,%f25,%f21;
    \\  mul.f32 %f22,%f22,%f8; fma.rn.f32 %f22,%f9,%f26,%f22;
    \\  mul.f32 %f23,%f23,%f8; fma.rn.f32 %f23,%f9,%f27,%f23;
    \\  mul.f32 %f40,%f40,%f8; fma.rn.f32 %f40,%f9,%f36,%f40;
    \\  mul.f32 %f41,%f41,%f8; fma.rn.f32 %f41,%f9,%f37,%f41;
    \\  mul.f32 %f42,%f42,%f8; fma.rn.f32 %f42,%f9,%f38,%f42;
    \\  mul.f32 %f43,%f43,%f8; fma.rn.f32 %f43,%f9,%f39,%f43;
    \\  add.u32 %r17,%r17,1; bra JLOOPH;
    \\JDH:
    \\  add.u32 %r24,%r9,4; mul.lo.s32 %r25,%r27,%r24;
    \\  mul.wide.u32 %rd17,%r25,4; add.s64 %rd18,%rd4,%rd17;
    \\  setp.ne.u32 %p3,%r28,0; @%p3 bra WRACCH;
    \\  st.global.f32 [%rd18],%f10; st.global.f32 [%rd18+4],%f11;
    \\WRACCH:
    \\  shl.b32 %r29,%r28,5; add.u32 %r29,%r29,16;   // byte off = 16 + lane*32
    \\  cvt.u64.u32 %rd19,%r29; add.s64 %rd20,%rd18,%rd19;
    \\  st.global.v4.f32 [%rd20],{%f20,%f21,%f22,%f23};
    \\  st.global.v4.f32 [%rd20+16],{%f40,%f41,%f42,%f43};
    \\END:
    \\  ret;
    \\}
;

/// Plain element copy with source/destination element offsets — keeps
/// hidden-state tap snapshots inside a recorded batch (cuMemcpyDtoD runs on
/// the null stream, which a graph capture rejects and a batch would flush).
/// b0=src, b1=dst. u0=count, u1=dst offset, u2=src offset.
pub const copy_off_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry copy_off(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 p3,
    \\  .param .u32 u0,.param .u32 u1,.param .u32 u2,.param .u32 u3,.param .u32 u4,.param .u32 u5,.param .f32 f0,.param .f32 f1)
    \\{
    \\  .reg .pred %p<2>;
    \\  .reg .b32 %r<10>;
    \\  .reg .f32 %f<3>;
    \\  .reg .b64 %rd<10>;
    \\  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
    \\  ld.param.u32 %r5,[u0]; setp.ge.u32 %p1,%r4,%r5; @%p1 bra END;
    \\  ld.param.u32 %r6,[u1]; ld.param.u32 %r7,[u2];
    \\  ld.param.u64 %rd1,[p0]; ld.param.u64 %rd2,[p1];
    \\  cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    \\  add.u32 %r8,%r4,%r7; mul.wide.u32 %rd3,%r8,4; add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f1,[%rd4];
    \\  add.u32 %r9,%r4,%r6; mul.wide.u32 %rd5,%r9,4; add.s64 %rd6,%rd2,%rd5; st.global.f32 [%rd6],%f1;
    \\END:
    \\  ret;
    \\}
;
