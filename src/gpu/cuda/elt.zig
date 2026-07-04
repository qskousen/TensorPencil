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
/// b0=q[seq][heads][hd], b1=k[seq][kv][hd], b2=v[seq][kv][hd], b3=out[seq][heads][hd].
/// u0=seq, u1=heads, u2=kv_heads, u3=hd, f0=scale. acc[hd] in .local.
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
    \\  mov.u32 %r17,0;                       // j
    \\JLOOP:
    \\  setp.ge.u32 %p2,%r17,%r5; @%p2 bra JD;
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
