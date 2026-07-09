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
    \\  .local .align 4 .b8 accl[512];        // hd=128 f32
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

/// Flash-decoding pass 1: split the KV range of a single query (seq_q == 1)
/// across nsplit chunks, one WARP per (head, split). Requires hd == 128: each
/// lane owns 4 dims (q/k/v as v4.f32), the k·q dot closes with a shfl.bfly
/// tree (all lanes get the sum), softmax scalars are computed redundantly per
/// lane, and the accumulator lives in 4 registers per lane — no local memory.
/// Partial (m, d, pad, pad, acc[hd]) rows go to scratch, stride hd+4 so the
/// lane v4 stores stay 16B-aligned.
/// b0=q[heads][hd], b1=k[seq_kv][kv][hd], b2=v, b3=scratch.
/// u0=seq_kv, u1=heads, u2=kv_heads, u3=hd(=128), u4=nsplit, f0=scale.
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
    \\  ld.param.u32 %r5,[u0];                // seq_kv
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
