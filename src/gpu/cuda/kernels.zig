//! Hand-written / hand-emitted PTX kernels for the CUDA backend, plus the
//! bring-up smoke test. GEMM/prep/attention kernels are added here as Phase 1
//! progresses; each is authored as PTX (validated offline with
//! `ptxas -arch=sm_86`) and JIT-compiled by the driver at load time.

const std = @import("std");
const cu = @import("cu.zig");
const ctxmod = @import("context.zig");
const Context = ctxmod.Context;

/// Trivial element-wise `c = a + b` over `n` f32. The toolchain smoke test:
/// validates bindings -> PTX JIT -> launch -> readback end to end. Assembles
/// cleanly under `ptxas -arch=sm_86`.
pub const vadd_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\
    \\.visible .entry vadd(
    \\    .param .u64 p_a,
    \\    .param .u64 p_b,
    \\    .param .u64 p_c,
    \\    .param .u32 p_n
    \\)
    \\{
    \\    .reg .pred  %p<2>;
    \\    .reg .b32   %r<8>;
    \\    .reg .f32   %f<4>;
    \\    .reg .b64   %rd<11>;
    \\
    \\    ld.param.u64    %rd1, [p_a];
    \\    ld.param.u64    %rd2, [p_b];
    \\    ld.param.u64    %rd3, [p_c];
    \\    ld.param.u32    %r1, [p_n];
    \\    mov.u32         %r2, %ctaid.x;
    \\    mov.u32         %r3, %ntid.x;
    \\    mov.u32         %r4, %tid.x;
    \\    mad.lo.s32      %r5, %r2, %r3, %r4;
    \\    setp.ge.s32     %p1, %r5, %r1;
    \\    @%p1 bra        DONE;
    \\    mul.wide.s32    %rd4, %r5, 4;
    \\    cvta.to.global.u64 %rd5, %rd1;
    \\    add.s64         %rd6, %rd5, %rd4;
    \\    cvta.to.global.u64 %rd7, %rd2;
    \\    add.s64         %rd8, %rd7, %rd4;
    \\    ld.global.f32   %f1, [%rd6];
    \\    ld.global.f32   %f2, [%rd8];
    \\    add.f32         %f3, %f1, %f2;
    \\    cvta.to.global.u64 %rd9, %rd3;
    \\    add.s64         %rd10, %rd9, %rd4;
    \\    st.global.f32   [%rd10], %f3;
    \\DONE:
    \\    ret;
    \\}
;

/// End-to-end bring-up: JIT the vadd PTX, run it on a small vector, and verify
/// the result against a host reference. Returns error.CudaError on any mismatch.
pub fn smokeTest(ctx: *Context) !void {
    var mod = try ctx.loadModule(vadd_ptx);
    defer mod.unload(ctx);
    const f = try mod.getFunction(ctx, "vadd");

    const n: u32 = 4096;
    const a = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(a);
    const b = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(b);
    const c = try std.heap.page_allocator.alloc(f32, n);
    defer std.heap.page_allocator.free(c);
    for (a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (b, 0..) |*v, i| v.* = @floatFromInt(2 * i);

    var da = try ctx.alloc(n * 4);
    defer ctx.free(&da);
    var db = try ctx.alloc(n * 4);
    defer ctx.free(&db);
    var dc = try ctx.alloc(n * 4);
    defer ctx.free(&dc);

    try ctx.upload(da, std.mem.sliceAsBytes(a));
    try ctx.upload(db, std.mem.sliceAsBytes(b));

    var pa = da.ptr;
    var pb = db.ptr;
    var pc = dc.ptr;
    var pn = n;
    var params = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn) };

    const block: u32 = 256;
    const grid: u32 = (n + block - 1) / block;
    try ctx.launch(f, .{ grid, 1, 1 }, .{ block, 1, 1 }, 0, &params);
    try ctx.download(dc, std.mem.sliceAsBytes(c));

    for (c, 0..) |v, i| {
        const want: f32 = @floatFromInt(3 * i);
        if (v != want) {
            std.debug.print("vadd mismatch at {d}: got {d} want {d}\n", .{ i, v, want });
            return error.CudaError;
        }
    }
}

// ---------------------------------------------------------------------------
// int8 IMMA GEMM.  C[m][n] (s32) = A(s8)[m][k] @ B(s8)[n][k]^T, i.e.
// C[i][j] = sum_k A[i][k]*B[j][k].  For `mma.row.col` the B operand must be
// col-major K x N, which is exactly the natural row-major weight W[n][k] (k
// contiguous) — so NO k-major transpose is needed (unlike the Vulkan coopmat
// path). Both A and B fragments load 4-consecutive-k s8 as a u32 from global.
//
// m16n8k32 s8 fragment layout (verified): groupID = lane>>2, tid = lane&3.
//   A: a0=(row gid, k tid*4+0..3), a1=(row gid+8, same k), a2/a3 = +16 in k.
//   B: b0=(col gid, k tid*4+0..3), b1 = +16 in k.
//   C: c0=(row gid, col tid*2+0), c1=(gid,tid*2+1), c2=(gid+8,..), c3=(gid+8,..).
// ---------------------------------------------------------------------------

/// v0 — correctness reference: one warp per 16x8 output tile, fragments loaded
/// straight from global, s32 accumulate over the full k. Obviously-correct,
/// slow (no reuse). Requires m%16==0, n%8==0, k%32==0. Grid (n/8, m/16), 32 thr.
pub const igemm_v0_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\
    \\.visible .entry igemm_v0(
    \\    .param .u64 p_a,
    \\    .param .u64 p_b,
    \\    .param .u64 p_c,
    \\    .param .u32 p_n,
    \\    .param .u32 p_k
    \\)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<40>;
    \\    .reg .b64 %rd<20>;
    \\    .reg .b32 %c<4>;
    \\    ld.param.u64 %rd1, [p_a];
    \\    ld.param.u64 %rd2, [p_b];
    \\    ld.param.u64 %rd3, [p_c];
    \\    ld.param.u32 %r1, [p_n];
    \\    ld.param.u32 %r2, [p_k];
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    cvta.to.global.u64 %rd3, %rd3;
    \\    mov.u32 %r3, %tid.x;
    \\    and.b32 %r3, %r3, 31;
    \\    shr.u32 %r4, %r3, 2;
    \\    and.b32 %r5, %r3, 3;
    \\    mov.u32 %r6, %ctaid.y;
    \\    mov.u32 %r7, %ctaid.x;
    \\    shl.b32 %r8, %r6, 4;
    \\    shl.b32 %r9, %r7, 3;
    \\    add.u32 %r10, %r8, %r4;
    \\    add.u32 %r11, %r10, 8;
    \\    add.u32 %r12, %r9, %r4;
    \\    shl.b32 %r13, %r5, 2;
    \\    mul.wide.u32 %rd4, %r10, %r2;
    \\    add.s64 %rd4, %rd1, %rd4;
    \\    mul.wide.u32 %rd5, %r11, %r2;
    \\    add.s64 %rd5, %rd1, %rd5;
    \\    mul.wide.u32 %rd6, %r12, %r2;
    \\    add.s64 %rd6, %rd2, %rd6;
    \\    mov.u32 %c0, 0;
    \\    mov.u32 %c1, 0;
    \\    mov.u32 %c2, 0;
    \\    mov.u32 %c3, 0;
    \\    mov.u32 %r14, 0;
    \\LOOP:
    \\    setp.ge.u32 %p1, %r14, %r2;
    \\    @%p1 bra ENDLOOP;
    \\    add.u32 %r15, %r14, %r13;
    \\    cvt.u64.u32 %rd7, %r15;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.u32 %r20, [%rd8];
    \\    ld.global.u32 %r22, [%rd8+16];
    \\    add.s64 %rd9, %rd5, %rd7;
    \\    ld.global.u32 %r21, [%rd9];
    \\    ld.global.u32 %r23, [%rd9+16];
    \\    add.s64 %rd10, %rd6, %rd7;
    \\    ld.global.u32 %r24, [%rd10];
    \\    ld.global.u32 %r25, [%rd10+16];
    \\    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    \\      {%c0,%c1,%c2,%c3},
    \\      {%r20,%r21,%r22,%r23},
    \\      {%r24,%r25},
    \\      {%c0,%c1,%c2,%c3};
    \\    add.u32 %r14, %r14, 32;
    \\    bra LOOP;
    \\ENDLOOP:
    \\    shl.b32 %r16, %r5, 1;
    \\    add.u32 %r17, %r9, %r16;
    \\    mad.lo.u32 %r18, %r10, %r1, %r17;
    \\    mul.wide.u32 %rd11, %r18, 4;
    \\    add.s64 %rd11, %rd3, %rd11;
    \\    st.global.u32 [%rd11], %c0;
    \\    st.global.u32 [%rd11+4], %c1;
    \\    mad.lo.u32 %r19, %r11, %r1, %r17;
    \\    mul.wide.u32 %rd12, %r19, 4;
    \\    add.s64 %rd12, %rd3, %rd12;
    \\    st.global.u32 [%rd12], %c2;
    \\    st.global.u32 [%rd12+4], %c3;
    \\    ret;
    \\}
;

// ---------------------------------------------------------------------------
// int4 IMMA GEMM (W4A4).  C[m][n] (s32) = A(s4)[m][k] @ B(s4)[n][k]^T.
//
// A and B are nibble-packed: two signed 4-bit values per byte, element 2j in
// the low nibble, 2j+1 in the high (the on-disk convrot weight layout, and the
// same layout opI4Prep writes for activations). k is contiguous, so 8
// consecutive-k s4 values are exactly one u32 — loadable straight into an mma
// fragment register, no repack (mirrors the s8 path's 4-consecutive-k u32).
//
// m16n8k64 s4 fragment layout: groupID = lane>>2, tid = lane&3.
//   A: a0=(row gid, k tid*8+0..7), a1=(row gid+8, same k),
//      a2=(row gid, k tid*8+32..39), a3=(row gid+8, k+32).  (8 s4 = 1 u32)
//   B: b0=(col gid, k tid*8+0..7), b1=(col gid, k tid*8+32..39).
//   C: identical to the s8 m16n8k32 case (s32 16x8 tile).
// A/B byte addr of element (row,kk): row*(k/2) + kk/2 ; a u32 at
//   row*(k/2) + tid*4 + k0/2 covers k = k0+tid*8 .. +7. a2/b1 sit +16 bytes.
// ---------------------------------------------------------------------------

/// v0 — correctness reference for the s4 tensor-core GEMM: one warp per 16x8
/// output tile, fragments loaded straight from global, s32 accumulate over the
/// full k. Slow (no reuse) but obviously correct. Requires m%16==0, n%8==0,
/// k%64==0. Grid (n/8, m/16), 32 threads.
pub const i4gemm_v0_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\
    \\.visible .entry i4gemm_v0(
    \\    .param .u64 p_a,
    \\    .param .u64 p_b,
    \\    .param .u64 p_c,
    \\    .param .u32 p_n,
    \\    .param .u32 p_k
    \\)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<40>;
    \\    .reg .b64 %rd<20>;
    \\    .reg .b32 %c<4>;
    \\    ld.param.u64 %rd1, [p_a];
    \\    ld.param.u64 %rd2, [p_b];
    \\    ld.param.u64 %rd3, [p_c];
    \\    ld.param.u32 %r1, [p_n];
    \\    ld.param.u32 %r2, [p_k];
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    cvta.to.global.u64 %rd3, %rd3;
    \\    mov.u32 %r3, %tid.x;
    \\    and.b32 %r3, %r3, 31;
    \\    shr.u32 %r4, %r3, 2;          // gid = lane>>2
    \\    and.b32 %r5, %r3, 3;          // tid = lane&3
    \\    mov.u32 %r6, %ctaid.y;
    \\    mov.u32 %r7, %ctaid.x;
    \\    shl.b32 %r8, %r6, 4;          // row0 = ctaid.y*16
    \\    shl.b32 %r9, %r7, 3;          // col0 = ctaid.x*8
    \\    add.u32 %r10, %r8, %r4;       // rowA = row0 + gid
    \\    add.u32 %r11, %r10, 8;        // rowA8
    \\    add.u32 %r12, %r9, %r4;       // colB = col0 + gid
    \\    shr.u32 %r26, %r2, 1;         // khb = k/2 (row stride in bytes)
    \\    shl.b32 %r13, %r5, 2;         // tid*4 (byte offset within row)
    \\    // A row bases: rd4 = A + rowA*khb + tid*4 ; rd5 = A + rowA8*khb + tid*4
    \\    mul.wide.u32 %rd4, %r10, %r26;
    \\    add.s64 %rd4, %rd1, %rd4;
    \\    mul.wide.u32 %rd5, %r11, %r26;
    \\    add.s64 %rd5, %rd1, %rd5;
    \\    // B row base: rd6 = B + colB*khb + tid*4
    \\    mul.wide.u32 %rd6, %r12, %r26;
    \\    add.s64 %rd6, %rd2, %rd6;
    \\    cvt.u64.u32 %rd7, %r13;
    \\    add.s64 %rd4, %rd4, %rd7;
    \\    add.s64 %rd5, %rd5, %rd7;
    \\    add.s64 %rd6, %rd6, %rd7;
    \\    mov.u32 %c0, 0;
    \\    mov.u32 %c1, 0;
    \\    mov.u32 %c2, 0;
    \\    mov.u32 %c3, 0;
    \\    mov.u32 %r14, 0;              // koff (bytes), 0..khb step 32
    \\LOOP:
    \\    setp.ge.u32 %p1, %r14, %r26;
    \\    @%p1 bra ENDLOOP;
    \\    cvt.u64.u32 %rd8, %r14;
    \\    add.s64 %rd9, %rd4, %rd8;
    \\    ld.global.u32 %r20, [%rd9];       // a0 (rowA, k0..)
    \\    ld.global.u32 %r22, [%rd9+16];    // a2 (rowA, k0+32..)
    \\    add.s64 %rd10, %rd5, %rd8;
    \\    ld.global.u32 %r21, [%rd10];      // a1 (rowA8, k0..)
    \\    ld.global.u32 %r23, [%rd10+16];   // a3 (rowA8, k0+32..)
    \\    add.s64 %rd11, %rd6, %rd8;
    \\    ld.global.u32 %r24, [%rd11];      // b0 (colB, k0..)
    \\    ld.global.u32 %r25, [%rd11+16];   // b1 (colB, k0+32..)
    \\    mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
    \\      {%c0,%c1,%c2,%c3},
    \\      {%r20,%r21,%r22,%r23},
    \\      {%r24,%r25},
    \\      {%c0,%c1,%c2,%c3};
    \\    add.u32 %r14, %r14, 32;
    \\    bra LOOP;
    \\ENDLOOP:
    \\    shl.b32 %r16, %r5, 1;         // tid*2
    \\    add.u32 %r17, %r9, %r16;      // col = col0 + tid*2
    \\    mad.lo.u32 %r18, %r10, %r1, %r17;
    \\    mul.wide.u32 %rd12, %r18, 4;
    \\    add.s64 %rd12, %rd3, %rd12;
    \\    st.global.u32 [%rd12], %c0;
    \\    st.global.u32 [%rd12+4], %c1;
    \\    mad.lo.u32 %r19, %r11, %r1, %r17;
    \\    mul.wide.u32 %rd13, %r19, 4;
    \\    add.s64 %rd13, %rd3, %rd13;
    \\    st.global.u32 [%rd13], %c2;
    \\    st.global.u32 [%rd13+4], %c3;
    \\    ret;
    \\}
;

const ptx = @import("ptx.zig");

/// v1 — shared-memory register-tiled IMMA GEMM. 128x128 block tile, 4 warps
/// (2x2 grid of 64x64 warp tiles), 128 s32 accumulators/thread, K_STEP=64.
/// A/B staged synchronously into 16 KB static shared, fragments loaded with
/// plain `ld.shared.b32`. Requires m%128==0, n%128==0, k%64==0. Grid (n/128,
/// m/128), 128 threads. (cp.async + dynamic shared come in v2.)
///
/// Generated with the PTX emitter — the 32 MMAs/k-step and their fragment loads
/// are unrolled here rather than hand-typed. Caller frees the returned bytes.
pub fn buildIgemmSmem(alloc: std.mem.Allocator) ![:0]u8 {
    const BM = 128;
    const KSTEP = 64;
    const MT = 4; // 16-row m-tiles per warp (WM=64)
    const NT = 8; // 8-col n-tiles per warp (WN=64)
    const KS = KSTEP / 32; // 2 k-substeps of 32
    const BS_BASE = BM * KSTEP; // 8192
    const SH_BYTES = 2 * BM * KSTEP; // 16384

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    // Accumulators: acc[(mi*NT+nj)*4 + e], all init 0.
    const acc = try b.regs(.b32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4); // A fragments for current ks
    const bf = try b.regs(.b32, NT * 2); // B fragments for current ks

    // ---- params ----
    const rd_a = try b.reg(.b64);
    const rd_b = try b.reg(.b64);
    const rd_c = try b.reg(.b64);
    const r_n = try b.reg(.b32);
    const r_k = try b.reg(.b32);
    try b.linef("ld.param.u64 {s}, [p_a];", .{rd_a});
    try b.linef("ld.param.u64 {s}, [p_b];", .{rd_b});
    try b.linef("ld.param.u64 {s}, [p_c];", .{rd_c});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.u32 {s}, [p_k];", .{r_k});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_a, rd_a });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_b, rd_b });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_c, rd_c });

    // ---- thread/warp/tile indices ----
    const r_t = try b.reg(.b32);
    const r_rowq = try b.reg(.b32);
    const r_kq = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("shr.u32 {s}, {s}, 4;", .{ r_rowq, r_t }); // t>>4 (0..7)
    try b.linef("and.b32 {s}, {s}, 15;", .{ r_kq, r_t }); // t&15
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp }); // warp_m
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp }); // warp_n
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane }); // gid
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane }); // tid_f
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 }); // *128
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });

    // ---- staging base global byte pointers (k0 added in loop) ----
    // A: base_a + (row0 + rowq)*k + kq*4 ; advance +8*k per staging step.
    const r_arow = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    const rd_abase = try b.reg(.b64);
    const rd_bbase = try b.reg(.b64);
    const rd_8k = try b.reg(.b64);
    const r_kq4 = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_kq4, r_kq }); // kq*4
    // rd_8k = 8*k
    try b.linef("mul.wide.u32 {s}, {s}, 8;", .{ rd_8k, r_k });
    // A base
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_arow, r_row0, r_rowq }); // row0+rowq
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_arow, r_k }); // (row0+rowq)*k
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_abase, rd_a, rd_tmp });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_kq4 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_abase, rd_abase, rd_tmp }); // + kq*4
    // B base (col0+rowq)*k + kq*4
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_arow, r_col0, r_rowq });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_arow, r_k });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bbase, rd_b, rd_tmp });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_kq4 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bbase, rd_bbase, rd_tmp });

    // shared-window base of `smem` (all shared addresses are smem-relative).
    const r_smem = try b.reg(.b32);
    try b.linef("mov.u32 {s}, smem;", .{r_smem});

    // shared store base addresses: smem + As(0) + t*4 ; smem + Bs + t*4
    const r_shA = try b.reg(.b32);
    const r_shB = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_shA, r_t }); // t*4  (As base 0)
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_shA, r_shA, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_shB, r_shA, BS_BASE });

    // shared fragment-load base offsets (bytes):
    //   As_lane = ((warp_m*64 + gid)*16 + tf)*4
    //   Bs_lane = BS_BASE + ((warp_n*64 + gid)*16 + tf)*4
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wm }); // warp_m*64
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_tmp, r_tmp }); // *16
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_tf });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_asl, r_tmp }); // *4
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wn }); // warp_n*64
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_tmp, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_tf });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_bsl, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, BS_BASE });

    // init accumulators
    for (acc) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    // ---- k loop ----
    const r_k0 = try b.reg(.b32);
    const rd_ap = try b.reg(.b64);
    const rd_bp = try b.reg(.b64);
    const rd_k0 = try b.reg(.b64);
    const r_tA = try b.reg(.b32);
    const r_tB = try b.reg(.b32);
    const p0 = try b.reg(.pred);
    try b.linef("mov.u32 {s}, 0;", .{r_k0});
    try b.label("LOOP");
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_k0, r_k0 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_abase, rd_k0 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bbase, rd_k0 });
    // stage 16 quads/thread for A and B
    var i: usize = 0;
    while (i < (BM * KSTEP / 4) / 128) : (i += 1) {
        try b.linef("ld.global.u32 {s}, [{s}];", .{ r_tA, rd_ap });
        try b.linef("st.shared.u32 [{s}+{d}], {s};", .{ r_shA, i * 512, r_tA });
        try b.linef("ld.global.u32 {s}, [{s}];", .{ r_tB, rd_bp });
        try b.linef("st.shared.u32 [{s}+{d}], {s};", .{ r_shB, i * 512, r_tB });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_ap, rd_8k });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bp, rd_8k });
    }
    try b.line("bar.sync 0;");
    // compute
    var ks: usize = 0;
    while (ks < KS) : (ks += 1) {
        var mi: usize = 0;
        while (mi < MT) : (mi += 1) {
            const o = (mi * 256 + ks * 8) * 4;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 512 });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 528 });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = (nj * 128 + ks * 8) * 4;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
        }
        mi = 0;
        while (mi < MT) : (mi += 1) {
            nj = 0;
            while (nj < NT) : (nj += 1) {
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                try b.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                    a[0],              a[1],              a[2],              a[3],
                    af[mi * 4 + 0],    af[mi * 4 + 1],    af[mi * 4 + 2],    af[mi * 4 + 3],
                    bf[nj * 2 + 0],    bf[nj * 2 + 1],    a[0],              a[1],
                    a[2],              a[3],
                });
            }
        }
    }
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_k0, r_k0, KSTEP });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p0, r_k0, r_k });
    try b.linef("@{s} bra LOOP;", .{p0});

    // ---- store ----
    // Crow = row0 + warp_m*64 + gid ; Ccol = col0 + warp_n*64 + tf*2
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    const rd_8n4 = try b.reg(.b64);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ r_tmp, r_tf }); // tf*2
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, 32;", .{ rd_8n4, r_n }); // 8*n*4
    const r_row_mi = try b.reg(.b32);
    const r_idx = try b.reg(.b32);
    const rd_cp = try b.reg(.b64);
    const rd_cp2 = try b.reg(.b64);
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_row_mi, r_crow, mi2 * 16 });
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const a = acc[(mi2 * NT + nj) * 4 ..][0..4];
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_idx, r_row_mi, r_n, r_ccol });
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_idx, r_idx, nj * 8 });
            try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_cp, r_idx });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp, rd_c, rd_cp });
            try b.linef("st.global.u32 [{s}], {s};", .{ rd_cp, a[0] });
            try b.linef("st.global.u32 [{s}+4], {s};", .{ rd_cp, a[1] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp2, rd_cp, rd_8n4 });
            try b.linef("st.global.u32 [{s}], {s};", .{ rd_cp2, a[2] });
            try b.linef("st.global.u32 [{s}+4], {s};", .{ rd_cp2, a[3] });
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH_BYTES});
    defer alloc.free(shared_decl);
    return b.build(
        "igemm_smem",
        "    .param .u64 p_a,\n    .param .u64 p_b,\n    .param .u64 p_c,\n    .param .u32 p_n,\n    .param .u32 p_k",
        shared_decl,
    );
}

/// v2 — cp.async double-buffered IMMA GEMM. Same 128x128 tile / 2x2 warps / 128
/// accumulators as v1, but A/B slabs are streamed global->shared with
/// `cp.async.cg` (the Ampere `LDGSTS` the Vulkan path can't emit) and
/// double-buffered so the next slab loads while the current one computes.
/// K_STEP is a parameter: 64 -> 32 KB shared (no opt-in); 128 -> 64 KB (needs
/// cuFuncSetAttribute opt-in, the >48 KB lever). Requires m%128==0, n%128==0,
/// k%K_STEP==0. Entry `igemm_pipe`.
pub fn buildIgemmPipe(alloc: std.mem.Allocator, kstep: usize, fuse: bool, bits: usize) ![:0]u8 {
    std.debug.assert(bits == 8 or bits == 4);
    // s8: one m16n8k32 per 32-byte substep (32 k). s4: one m16n8k64 per 32-byte
    // substep (64 k). Staging/tile math is byte-based and identical; only the
    // mma opcode and the global row byte-stride (k vs k/2) differ.
    const mma_op = if (bits == 8)
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
    else
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32";
    const BM = 128;
    const MT = 4;
    const NT = 8;
    const KS = kstep / 32; // k-substeps of 32
    const TILE = BM * kstep; // bytes per A (or B) tile
    const BUFSZ = 2 * TILE; // A+B per buffer
    const SH_BYTES = 2 * BUFSZ; // double-buffered
    const cpr = kstep / 16; // 16B chunks per row
    const cpt = TILE / 16 / 128; // 16B cp.async ops per thread per tile
    const stage_stride = TILE / cpt; // dst byte advance per staging step (=2048)

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.b32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4);
    const bf = try b.regs(.b32, NT * 2);

    const rd_a = try b.reg(.b64);
    const rd_b = try b.reg(.b64);
    const rd_c = try b.reg(.b64);
    const r_n = try b.reg(.b32);
    const r_k = try b.reg(.b32);
    const rd_as = try b.reg(.b64);
    const rd_ws = try b.reg(.b64);
    try b.linef("ld.param.u64 {s}, [p_a];", .{rd_a});
    try b.linef("ld.param.u64 {s}, [p_b];", .{rd_b});
    try b.linef("ld.param.u64 {s}, [p_c];", .{rd_c});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.u32 {s}, [p_k];", .{r_k});
    // r_k is used only as the global row byte-stride and the slab count divisor.
    // s4 packs two elements per byte, so the byte stride is k/2.
    if (bits == 4) try b.linef("shr.u32 {s}, {s}, 1;", .{ r_k, r_k });
    if (fuse) {
        try b.linef("ld.param.u64 {s}, [p_as];", .{rd_as});
        try b.linef("ld.param.u64 {s}, [p_ws];", .{rd_ws});
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_as, rd_as });
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_ws, rd_ws });
    }
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_a, rd_a });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_b, rd_b });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_c, rd_c });

    const r_t = try b.reg(.b32);
    const r_srow = try b.reg(.b32);
    const r_cpos = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("div.u32 {s}, {s}, {d};", .{ r_srow, r_t, cpr });
    try b.linef("rem.u32 {s}, {s}, {d};", .{ r_cpos, r_t, cpr });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_cpos, r_cpos });
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });

    const r_smem = try b.reg(.b32);
    try b.linef("mov.u32 {s}, smem;", .{r_smem});

    // Staging global base pointers (k0 added per slab):
    const r_tmp = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    const rd_astg = try b.reg(.b64);
    const rd_bstg = try b.reg(.b64);
    const rd_rowstep_k = try b.reg(.b64); // (128/cpr)*k bytes per staging step
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_rowstep_k, r_k, BM / cpr });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_row0, r_srow });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_tmp, r_k });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_astg, rd_a, rd_tmp });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_cpos });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_astg, rd_astg, rd_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_col0, r_srow });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_tmp, r_k });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bstg, rd_b, rd_tmp });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_cpos });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bstg, rd_bstg, rd_tmp });

    // Staging shared dst offset within buffer's A region: srow*kstep + cpos.
    const r_stdst = try b.reg(.b32);
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_stdst, r_srow, kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_stdst, r_stdst, r_cpos });

    // Fragment-load lane bases (relative to buffer): As lane / Bs lane.
    const r_asl0 = try b.reg(.b32);
    const r_bsl0 = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl0, r_tmp, kstep });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_tmp, r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl0, r_asl0, r_tmp });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl0, r_tmp, kstep });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_tmp, r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl0, r_bsl0, r_tmp });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl0, r_bsl0, TILE });

    for (acc) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    const r_i = try b.reg(.b32);
    const r_ip1 = try b.reg(.b32);
    const r_nslab = try b.reg(.b32);
    const rd_ap = try b.reg(.b64);
    const rd_bp = try b.reg(.b64);
    const rd_koff = try b.reg(.b64);
    const r_buf = try b.reg(.b32); // buffer base (smem + b*BUFSZ)
    const r_dst = try b.reg(.b32);
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    const r_bit = try b.reg(.b32);
    const p_more = try b.reg(.pred);
    const p_loop = try b.reg(.pred);

    try b.linef("div.u32 {s}, {s}, {d};", .{ r_nslab, r_k, kstep });

    // ---- prologue: stage slab 0 into buffer 0 (smem + 0) ----
    try b.linef("mov.b64 {s}, {s};", .{ rd_ap, rd_astg });
    try b.linef("mov.b64 {s}, {s};", .{ rd_bp, rd_bstg });
    // A into buffer0
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_dst, r_smem, r_stdst });
    {
        var j: usize = 0;
        while (j < cpt) : (j += 1) {
            try b.linef("cp.async.cg.shared.global [{s}+{d}], [{s}], 16;", .{ r_dst, j * stage_stride, rd_ap });
            if (j + 1 < cpt) try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_ap, rd_rowstep_k });
        }
    }
    // B into buffer0
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_dst, r_smem, r_stdst });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_dst, r_dst, TILE });
    {
        var j: usize = 0;
        while (j < cpt) : (j += 1) {
            try b.linef("cp.async.cg.shared.global [{s}+{d}], [{s}], 16;", .{ r_dst, j * stage_stride, rd_bp });
            if (j + 1 < cpt) try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bp, rd_rowstep_k });
        }
    }
    try b.line("cp.async.commit_group;");

    try b.linef("mov.u32 {s}, 0;", .{r_i});
    try b.label("LOOP");
    try b.linef("add.u32 {s}, {s}, 1;", .{ r_ip1, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_more, r_ip1, r_nslab });
    try b.linef("@!{s} bra NOSTAGE;", .{p_more});
    // rd_ap/bp = base + (i+1)*kstep
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_koff, r_ip1, kstep });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_astg, rd_koff });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bstg, rd_koff });
    // nb buffer base = smem + ((i+1)&1)*BUFSZ
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_bit, r_ip1 });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_buf, r_bit, BUFSZ });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_buf, r_buf, r_smem });
    // A
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_dst, r_buf, r_stdst });
    {
        var j: usize = 0;
        while (j < cpt) : (j += 1) {
            try b.linef("cp.async.cg.shared.global [{s}+{d}], [{s}], 16;", .{ r_dst, j * stage_stride, rd_ap });
            if (j + 1 < cpt) try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_ap, rd_rowstep_k });
        }
    }
    // B
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_dst, r_buf, r_stdst });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_dst, r_dst, TILE });
    {
        var j: usize = 0;
        while (j < cpt) : (j += 1) {
            try b.linef("cp.async.cg.shared.global [{s}+{d}], [{s}], 16;", .{ r_dst, j * stage_stride, rd_bp });
            if (j + 1 < cpt) try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bp, rd_rowstep_k });
        }
    }
    try b.line("cp.async.commit_group;");
    try b.label("NOSTAGE");
    // wait: keep the next-slab group in flight (wait_group 1) when we staged
    // one, else drain (wait_group 0) so the current slab is complete.
    try b.linef("@{s} bra WAIT1;", .{p_more});
    try b.line("cp.async.wait_group 0;");
    try b.line("bra WAITED;");
    try b.label("WAIT1");
    try b.line("cp.async.wait_group 1;");
    try b.label("WAITED");
    try b.line("bar.sync 0;");

    // ---- compute on buffer cb = i&1 ----
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_bit, r_i });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_buf, r_bit, BUFSZ });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_buf, r_buf, r_smem });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_buf, r_asl0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_buf, r_bsl0 });
    var ks: usize = 0;
    while (ks < KS) : (ks += 1) {
        var mi: usize = 0;
        while (mi < MT) : (mi += 1) {
            const o = mi * 16 * kstep + ks * 32;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 8 * kstep });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 8 * kstep + 16 });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = nj * 8 * kstep + ks * 32;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
        }
        mi = 0;
        while (mi < MT) : (mi += 1) {
            nj = 0;
            while (nj < NT) : (nj += 1) {
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                try b.linef("{s} {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                    mma_op,
                    a[0],           a[1],           a[2],           a[3],
                    af[mi * 4 + 0], af[mi * 4 + 1], af[mi * 4 + 2], af[mi * 4 + 3],
                    bf[nj * 2 + 0], bf[nj * 2 + 1], a[0],           a[1],
                    a[2],           a[3],
                });
            }
        }
    }
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, 1;", .{ r_i, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_loop, r_i, r_nslab });
    try b.linef("@{s} bra LOOP;", .{p_loop});

    // ---- store ----
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    const rd_8n4 = try b.reg(.b64);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ r_tmp, r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, 32;", .{ rd_8n4, r_n });
    const r_row_mi = try b.reg(.b32);
    const r_idx = try b.reg(.b32);
    const rd_cp = try b.reg(.b64);
    const rd_cp2 = try b.reg(.b64);
    // fused-rescale temporaries
    const f_as0 = try b.reg(.f32);
    const f_as8 = try b.reg(.f32);
    const f_ws0 = try b.reg(.f32);
    const f_ws1 = try b.reg(.f32);
    const f_y = try b.reg(.f32);
    const rd_sc = try b.reg(.b64);
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_row_mi, r_crow, mi2 * 16 });
        if (fuse) {
            // act_scale[row_mi] and act_scale[row_mi+8]
            try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_sc, r_row_mi });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sc, rd_as, rd_sc });
            try b.linef("ld.global.f32 {s}, [{s}];", .{ f_as0, rd_sc });
            try b.linef("ld.global.f32 {s}, [{s}+32];", .{ f_as8, rd_sc });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const a = acc[(mi2 * NT + nj) * 4 ..][0..4];
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_idx, r_row_mi, r_n, r_ccol });
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_idx, r_idx, nj * 8 });
            try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_cp, r_idx });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp, rd_c, rd_cp });
            if (!fuse) {
                try b.linef("st.global.u32 [{s}], {s};", .{ rd_cp, a[0] });
                try b.linef("st.global.u32 [{s}+4], {s};", .{ rd_cp, a[1] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp2, rd_cp, rd_8n4 });
                try b.linef("st.global.u32 [{s}], {s};", .{ rd_cp2, a[2] });
                try b.linef("st.global.u32 [{s}+4], {s};", .{ rd_cp2, a[3] });
            } else {
                // weight_scale[col] and [col+1], col = r_ccol + nj*8
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_idx, r_ccol, nj * 8 });
                try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_sc, r_idx });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sc, rd_ws, rd_sc });
                try b.linef("ld.global.f32 {s}, [{s}];", .{ f_ws0, rd_sc });
                try b.linef("ld.global.f32 {s}, [{s}+4];", .{ f_ws1, rd_sc });
                // y = f32(acc) * act_scale[row] * weight_scale[col]
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_y, a[0] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_as0 });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_ws0 });
                try b.linef("st.global.f32 [{s}], {s};", .{ rd_cp, f_y });
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_y, a[1] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_as0 });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_ws1 });
                try b.linef("st.global.f32 [{s}+4], {s};", .{ rd_cp, f_y });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp2, rd_cp, rd_8n4 });
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_y, a[2] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_as8 });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_ws0 });
                try b.linef("st.global.f32 [{s}], {s};", .{ rd_cp2, f_y });
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_y, a[3] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_as8 });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ f_y, f_y, f_ws1 });
                try b.linef("st.global.f32 [{s}+4], {s};", .{ rd_cp2, f_y });
            }
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH_BYTES});
    defer alloc.free(shared_decl);
    const params = if (fuse)
        "    .param .u64 p_a,\n    .param .u64 p_b,\n    .param .u64 p_c,\n    .param .u32 p_n,\n    .param .u32 p_k,\n    .param .u64 p_as,\n    .param .u64 p_ws"
    else
        "    .param .u64 p_a,\n    .param .u64 p_b,\n    .param .u64 p_c,\n    .param .u32 p_n,\n    .param .u32 p_k";
    const entry = if (bits == 8)
        (if (fuse) "igemm_pipe_fused" else "igemm_pipe")
    else
        (if (fuse) "i4gemm_pipe_fused" else "i4gemm_pipe");
    return b.build(entry, params, shared_decl);
}

/// Fused activation prep, one block (256 threads) per row: load x[row] into
/// dynamic shared f32, radix-4 FWHT per 256-group (bit-identical to
/// convrot.rotate — each butterfly output is a fixed 4-input sum, so the
/// parallel order matches the serial CPU order exactly), /16 normalize +
/// per-row abs-max, dynamic scale = max(absmax/maxq, 1e-12), then round-half-away
/// quantize + pack. `bits` selects the output format: 8 → int8, 4 s8/u32,
/// entry `iprep`, clamp [-128,127]; 4 → int4, 8 s4/u32, entry `i4prep`, clamp
/// [-8,7]. Packed row is [m][cols/(32/bits)] u32. Uses >48 KB dynamic shared for
/// cols=16384 (the Vulkan path was forced to f16 there by the 48 KB cap; here
/// f32 rotation is exact). Requires cols%256==0, (cols/256)%4==0 (FWHT) and
/// (cols/(32/bits))%256==0 (pack). block 256, grid (m,1,1).
pub fn buildPrep(alloc: std.mem.Allocator, cols: usize, bits: usize, in_f16: bool) ![:0]u8 {
    std.debug.assert(bits == 8 or bits == 4);
    const in_bytes: usize = if (in_f16) 2 else 4; // activation elem width (f16 chain)
    const per_word = 32 / bits; // s8: 4 elements/u32 ; s4: 8 elements/u32
    const per_word_log2 = std.math.log2_int(usize, per_word);
    const maxq: i32 = (@as(i32, 1) << @intCast(bits - 1)) - 1; // 127 / 7
    const minq: i32 = -(@as(i32, 1) << @intCast(bits - 1)); // -128 / -8
    const elt_mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1; // 0xFF / 0xF
    const maxq_hex: u32 = @bitCast(@as(f32, @floatFromInt(maxq))); // 127.0 / 7.0 bits
    const ngroups = cols / 256;
    const nbf = ngroups * 64 / 256; // butterflies/thread/pass (all 256 threads busy)
    const load_iters = cols / 256;
    const word_iters = cols / (per_word * 256); // packed u32 words / thread
    const SMAX_OFF = cols * 4; // smax[256] f32 region
    const SCALE_OFF = SMAX_OFF + 256 * 4; // scale broadcast slot

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const rd_x = try b.reg(.b64);
    const rd_q = try b.reg(.b64);
    const rd_s = try b.reg(.b64);
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_q];", .{rd_q});
    try b.linef("ld.param.u64 {s}, [p_s];", .{rd_s});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_q, rd_q });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_s, rd_s });

    const r_t = try b.reg(.b32);
    const r_row = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const rd_xrow = try b.reg(.b64); // p_x + row*cols*in_bytes
    const rd_tmp = try b.reg(.b64);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_row});
    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_xrow, r_row, cols * in_bytes });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_xrow, rd_x, rd_xrow });

    // ---- load x[row] -> shared f32 (rotation runs in f32 regardless of input) ----
    const r_sh = try b.reg(.b32); // smem + t*4
    const rd_g = try b.reg(.b64);
    const r_ftmp = try b.reg(.f32);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_sh, r_t });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_sh, r_sh, r_smem });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_t });
    try b.linef("shl.b64 {s}, {s}, {d};", .{ rd_tmp, rd_tmp, if (in_f16) @as(usize, 1) else 2 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_g, rd_xrow, rd_tmp });
    {
        const r_h = if (in_f16) try b.reg(.b16) else "";
        var i: usize = 0;
        while (i < load_iters) : (i += 1) {
            if (in_f16) {
                try b.linef("ld.global.b16 {s}, [{s}+{d}];", .{ r_h, rd_g, i * 256 * 2 });
                try b.linef("cvt.f32.f16 {s}, {s};", .{ r_ftmp, r_h });
            } else {
                try b.linef("ld.global.f32 {s}, [{s}+{d}];", .{ r_ftmp, rd_g, i * 256 * 4 });
            }
            try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ r_sh, i * 256 * 4, r_ftmp });
        }
    }
    try b.line("bar.sync 0;");

    // ---- FWHT: 4 passes over strides 1,4,16,64 ----
    // bidx = t + bi*256 ; group = bidx>>6 ; bwithin = bidx&63 ;
    // p = group*256 + (bwithin/s)*4s + (bwithin%s).
    const r_bidx = try b.reg(.b32);
    const r_grp = try b.reg(.b32);
    const r_bw = try b.reg(.b32);
    const r_p = try b.reg(.b32);
    const r_sha = try b.reg(.b32); // smem + p*4
    const fa = try b.reg(.f32);
    const fb = try b.reg(.f32);
    const fc = try b.reg(.f32);
    const fd = try b.reg(.f32);
    const fo = try b.reg(.f32);
    const strides = [_]usize{ 1, 4, 16, 64 };
    for (strides) |s| {
        const logs = std.math.log2_int(usize, s);
        var bi: usize = 0;
        while (bi < nbf) : (bi += 1) {
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_bidx, r_t, bi * 256 });
            try b.linef("shr.u32 {s}, {s}, 6;", .{ r_grp, r_bidx }); // group
            try b.linef("and.b32 {s}, {s}, 63;", .{ r_bw, r_bidx }); // bwithin
            // p_in = (bw>>logs)<<(logs+2) + (bw & (s-1))
            if (s == 1) {
                try b.linef("shl.b32 {s}, {s}, 2;", .{ r_p, r_bw }); // bw*4
            } else {
                try b.linef("shr.u32 {s}, {s}, {d};", .{ r_p, r_bw, logs });
                try b.linef("shl.b32 {s}, {s}, {d};", .{ r_p, r_p, logs + 2 });
                try b.linef("and.b32 {s}, {s}, {d};", .{ r_bidx, r_bw, s - 1 }); // bw & (s-1) into r_bidx temp
                try b.linef("add.u32 {s}, {s}, {s};", .{ r_p, r_p, r_bidx });
            }
            // p += group*256
            try b.linef("shl.b32 {s}, {s}, 8;", .{ r_grp, r_grp });
            try b.linef("add.u32 {s}, {s}, {s};", .{ r_p, r_p, r_grp });
            // sha = smem + p*4
            try b.linef("shl.b32 {s}, {s}, 2;", .{ r_sha, r_p });
            try b.linef("add.u32 {s}, {s}, {s};", .{ r_sha, r_sha, r_smem });
            // load a,b,c,d at sha, +s*4, +2s*4, +3s*4
            try b.linef("ld.shared.f32 {s}, [{s}];", .{ fa, r_sha });
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fb, r_sha, s * 4 });
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fc, r_sha, s * 8 });
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fd, r_sha, s * 12 });
            // out0 = a+b+c-d
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fa, fb });
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fo, fc });
            try b.linef("sub.f32 {s}, {s}, {s};", .{ fo, fo, fd });
            try b.linef("st.shared.f32 [{s}], {s};", .{ r_sha, fo });
            // out1 = a+b-c+d
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fa, fb });
            try b.linef("sub.f32 {s}, {s}, {s};", .{ fo, fo, fc });
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fo, fd });
            try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ r_sha, s * 4, fo });
            // out2 = a-b+c+d
            try b.linef("sub.f32 {s}, {s}, {s};", .{ fo, fa, fb });
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fo, fc });
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fo, fd });
            try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ r_sha, s * 8, fo });
            // out3 = -a+b+c+d  = (b+c+d) - a
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fb, fc });
            try b.linef("add.f32 {s}, {s}, {s};", .{ fo, fo, fd });
            try b.linef("sub.f32 {s}, {s}, {s};", .{ fo, fo, fa });
            try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ r_sha, s * 12, fo });
        }
        try b.line("bar.sync 0;");
    }

    // ---- /16 normalize + per-thread abs-max ----
    const famax = try b.reg(.f32);
    const fav = try b.reg(.f32);
    try b.linef("mov.f32 {s}, 0f00000000;", .{famax});
    {
        var i: usize = 0;
        while (i < load_iters) : (i += 1) {
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fo, r_sh, i * 256 * 4 });
            try b.linef("mul.f32 {s}, {s}, 0f3D800000;", .{ fo, fo }); // *0.0625 == /16
            try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ r_sh, i * 256 * 4, fo });
            try b.linef("abs.f32 {s}, {s};", .{ fav, fo });
            try b.linef("max.f32 {s}, {s}, {s};", .{ famax, famax, fav });
        }
    }
    // write per-thread amax to smax[t], reduce 256->1
    const r_smx = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_smx, r_t });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_smx, r_smx, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_smx, r_smx, SMAX_OFF });
    try b.linef("st.shared.f32 [{s}], {s};", .{ r_smx, famax });
    try b.line("bar.sync 0;");
    const p_red = try b.reg(.pred);
    const foth = try b.reg(.f32);
    const steps = [_]usize{ 128, 64, 32, 16, 8, 4, 2, 1 };
    for (steps) |st| {
        const lbl = try b.newLabel("red");
        try b.linef("setp.ge.u32 {s}, {s}, {d};", .{ p_red, r_t, st });
        try b.linef("@{s} bra {s};", .{ p_red, lbl });
        try b.linef("ld.shared.f32 {s}, [{s}];", .{ fo, r_smx });
        try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ foth, r_smx, st * 4 });
        try b.linef("max.f32 {s}, {s}, {s};", .{ fo, fo, foth });
        try b.linef("st.shared.f32 [{s}], {s};", .{ r_smx, fo });
        try b.label(lbl);
        try b.line("bar.sync 0;");
    }
    // thread 0: scale = max(absmax/127, 1e-12); write act_scale[row] + broadcast.
    const lbl_sk = try b.newLabel("sk");
    const r_scaleaddr = try b.reg(.b32);
    const fsc = try b.reg(.f32);
    const rd_srow = try b.reg(.b64);
    try b.linef("setp.ne.u32 {s}, {s}, 0;", .{ p_red, r_t });
    try b.linef("@{s} bra {s};", .{ p_red, lbl_sk });
    try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fsc, r_smem, SMAX_OFF }); // smax[0] = absmax
    try b.linef("div.rn.f32 {s}, {s}, 0f{X:0>8};", .{ fsc, fsc, maxq_hex }); // /127.0 (s8) or /7.0 (s4)
    try b.linef("max.f32 {s}, {s}, 0f2B8CBCCC;", .{ fsc, fsc }); // 1e-12 zero-guard
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_scaleaddr, r_smem, SCALE_OFF });
    try b.linef("st.shared.f32 [{s}], {s};", .{ r_scaleaddr, fsc });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_srow, r_row });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_srow, rd_s, rd_srow });
    try b.linef("st.global.f32 [{s}], {s};", .{ rd_srow, fsc });
    try b.label(lbl_sk);
    try b.line("bar.sync 0;");

    // ---- quantize + pack (round half away, clamp [-127,127]) ----
    const r_word = try b.reg(.b32);
    const r_out = try b.reg(.b32);
    const r_q = try b.reg(.b32);
    const fv = try b.reg(.f32);
    const fr = try b.reg(.f32);
    const fh = try b.reg(.f32);
    const rd_qrow = try b.reg(.b64);
    const r_ecol = try b.reg(.b32);
    const r_esha = try b.reg(.b32);
    // scale broadcast
    try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fsc, r_smem, SCALE_OFF });
    // q row base = p_q + row*(cols/per_word)*4 bytes  (= row*cols for s8, row*cols/2 for s4)
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_qrow, r_row, (cols / per_word) * 4 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qrow, rd_q, rd_qrow });
    {
        var i: usize = 0;
        while (i < word_iters) : (i += 1) {
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_word, r_t, i * 256 });
            try b.linef("mov.u32 {s}, 0;", .{r_out});
            // element base col = word*per_word ; shared byte = smem + col*4
            try b.linef("shl.b32 {s}, {s}, {d};", .{ r_ecol, r_word, per_word_log2 });
            try b.linef("shl.b32 {s}, {s}, 2;", .{ r_esha, r_ecol }); // byte = col*4
            try b.linef("add.u32 {s}, {s}, {s};", .{ r_esha, r_esha, r_smem });
            var kk: usize = 0;
            while (kk < per_word) : (kk += 1) {
                try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fv, r_esha, kk * 4 });
                try b.linef("div.rn.f32 {s}, {s}, {s};", .{ fr, fv, fsc });
                try b.linef("copysign.f32 {s}, {s}, 0f3F000000;", .{ fh, fr }); // copysign(0.5, r)
                try b.linef("add.f32 {s}, {s}, {s};", .{ fr, fr, fh });
                try b.linef("cvt.rzi.s32.f32 {s}, {s};", .{ r_q, fr }); // round half away
                try b.linef("max.s32 {s}, {s}, {d};", .{ r_q, r_q, minq });
                try b.linef("min.s32 {s}, {s}, {d};", .{ r_q, r_q, maxq });
                try b.linef("and.b32 {s}, {s}, {d};", .{ r_q, r_q, elt_mask });
                if (kk == 0) {
                    try b.linef("mov.b32 {s}, {s};", .{ r_out, r_q });
                } else {
                    try b.linef("shl.b32 {s}, {s}, {d};", .{ r_q, r_q, kk * bits });
                    try b.linef("or.b32 {s}, {s}, {s};", .{ r_out, r_out, r_q });
                }
            }
            // store word: qrow + word*4
            try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_word });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_g, rd_qrow, rd_tmp });
            try b.linef("st.global.u32 [{s}], {s};", .{ rd_g, r_out });
        }
    }

    return b.build(
        if (bits == 8) "iprep" else "i4prep",
        "    .param .u64 p_x,\n    .param .u64 p_q,\n    .param .u64 p_s",
        ".extern .shared .align 16 .b8 smem[];",
    );
}

/// dynamic-shared byte requirement for a prep launch (bit-width-independent:
/// the FWHT runs on the f32 activations in shared regardless of output bits).
pub fn prepSharedBytes(cols: usize) usize {
    return cols * 4 + 256 * 4 + 256;
}

/// int8 rescale: y[i][j] = f32(acc_s32[i][j]) * act_scale[i] * weight_scale[j].
/// acc is [m][rows] s32; grid ceil(total/256), block 256. Entry `irescale`.
pub const irescale_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry irescale(
    \\    .param .u64 p_acc,
    \\    .param .u64 p_y,
    \\    .param .u64 p_as,
    \\    .param .u64 p_ws,
    \\    .param .u32 p_rows,
    \\    .param .u32 p_total
    \\)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<12>;
    \\    .reg .f32 %f<5>;
    \\    .reg .b64 %rd<16>;
    \\    ld.param.u64 %rd1, [p_acc];
    \\    ld.param.u64 %rd2, [p_y];
    \\    ld.param.u64 %rd3, [p_as];
    \\    ld.param.u64 %rd4, [p_ws];
    \\    ld.param.u32 %r1, [p_rows];
    \\    ld.param.u32 %r2, [p_total];
    \\    mov.u32 %r3, %ctaid.x;
    \\    mov.u32 %r4, %ntid.x;
    \\    mov.u32 %r5, %tid.x;
    \\    mad.lo.s32 %r6, %r3, %r4, %r5;
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra DONE;
    \\    div.u32 %r7, %r6, %r1;
    \\    rem.u32 %r8, %r6, %r1;
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    cvta.to.global.u64 %rd3, %rd3;
    \\    cvta.to.global.u64 %rd4, %rd4;
    \\    mul.wide.u32 %rd5, %r6, 4;
    \\    add.s64 %rd6, %rd1, %rd5;
    \\    ld.global.s32 %r9, [%rd6];
    \\    cvt.rn.f32.s32 %f1, %r9;
    \\    mul.wide.u32 %rd7, %r7, 4;
    \\    add.s64 %rd8, %rd3, %rd7;
    \\    ld.global.f32 %f2, [%rd8];
    \\    mul.wide.u32 %rd9, %r8, 4;
    \\    add.s64 %rd10, %rd4, %rd9;
    \\    ld.global.f32 %f3, [%rd10];
    \\    mul.f32 %f4, %f1, %f2;
    \\    mul.f32 %f4, %f4, %f3;
    \\    add.s64 %rd11, %rd2, %rd5;
    \\    st.global.f32 [%rd11], %f4;
    \\DONE:
    \\    ret;
    \\}
;

/// f16-output int8 rescale (the c16 chain): y[i][j] (f16) = f32(acc_s32[i][j]) *
/// act_scale[i] * weight_scale[j]. acc is [m][rows] s32 (×4), y is f16 (×2).
/// grid ceil(total/256), block 256. Entry `irescale_h16`.
pub const irescale_h16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry irescale_h16(
    \\    .param .u64 p_acc,
    \\    .param .u64 p_y,
    \\    .param .u64 p_as,
    \\    .param .u64 p_ws,
    \\    .param .u32 p_rows,
    \\    .param .u32 p_total
    \\)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<12>;
    \\    .reg .f32 %f<5>;
    \\    .reg .b16 %h<2>;
    \\    .reg .b64 %rd<16>;
    \\    ld.param.u64 %rd1, [p_acc];
    \\    ld.param.u64 %rd2, [p_y];
    \\    ld.param.u64 %rd3, [p_as];
    \\    ld.param.u64 %rd4, [p_ws];
    \\    ld.param.u32 %r1, [p_rows];
    \\    ld.param.u32 %r2, [p_total];
    \\    mov.u32 %r3, %ctaid.x;
    \\    mov.u32 %r4, %ntid.x;
    \\    mov.u32 %r5, %tid.x;
    \\    mad.lo.s32 %r6, %r3, %r4, %r5;
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra DONE;
    \\    div.u32 %r7, %r6, %r1;
    \\    rem.u32 %r8, %r6, %r1;
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    cvta.to.global.u64 %rd3, %rd3;
    \\    cvta.to.global.u64 %rd4, %rd4;
    \\    mul.wide.u32 %rd5, %r6, 4;
    \\    add.s64 %rd6, %rd1, %rd5;
    \\    ld.global.s32 %r9, [%rd6];
    \\    cvt.rn.f32.s32 %f1, %r9;
    \\    mul.wide.u32 %rd7, %r7, 4;
    \\    add.s64 %rd8, %rd3, %rd7;
    \\    ld.global.f32 %f2, [%rd8];
    \\    mul.wide.u32 %rd9, %r8, 4;
    \\    add.s64 %rd10, %rd4, %rd9;
    \\    ld.global.f32 %f3, [%rd10];
    \\    mul.f32 %f4, %f1, %f2;
    \\    mul.f32 %f4, %f4, %f3;
    \\    cvt.rn.f16.f32 %h0, %f4;
    \\    mul.wide.u32 %rd12, %r6, 2;
    \\    add.s64 %rd11, %rd2, %rd12;
    \\    st.global.b16 [%rd11], %h0;
    \\DONE:
    \\    ret;
    \\}
;

/// f16 tensor-core GEMM: C[m][n] (f32) = A(f16)[m][k] @ B(f16)[n][k]^T. The
/// attention building block (scores = Q@K^T; P@V with V pre-transposed). Same
/// 128x128 tile / 2x2 warps / 128 accumulators as the int8 GEMM, but mma.m16n8k16
/// f16->f32, K_STEP=32 (2 substeps of 16), f16 static shared. Bit-comparable to
/// an f16-rounded CPU reference. Requires m%128==0, n%128==0, k%32==0. Entry
/// `hgemm`. block 128, grid (n/128, m/128).
/// f16 tensor-core GEMM C[m][n] = A[m][k] @ B[n][k]ᵀ (128×128 tile, 4×8 warp
/// register tile, k stepped 32 through shared). Three modes threaded via flags:
///   batched — gid.z selects an independent GEMM (per-head strides p_sa/sb/sc);
///     the C-store also folds a scalar p_scale into the accumulators (used by the
///     scores GEMM to prefold the softmax scale so f16 S can't overflow).
///   c_f16   — store C as f16 (scores→softmax path; halves the S write).
///   attnout — the FUSED attention output: A operand is the raw scores S (f16),
///     and during A-staging each element is turned into a softmax probability
///     P[q][j] = exp2((S[q][j]-max[q])·log2e)·inv[q] (pad keys j≥seq → 0), read
///     from the per-row MD={max,1/sum} table (`softmax_md_f16`). This eliminates
///     the P materialization entirely (no P write in softmax, no P read here) —
///     the Vulkan-parity win. attnout implies batched; C is f32; p_scale is 1.
pub fn buildHgemm(alloc: std.mem.Allocator, batched: bool, c_f16: bool, attnout: bool) ![:0]u8 {
    const BM = 128;
    const KSTEP = 32;
    const MT = 4;
    const NT = 8;
    const KS = KSTEP / 16; // 2 substeps of k=16
    const BS_BASE = BM * KSTEP * 2; // bytes (f16): 128*32*2 = 8192
    const SH_BYTES = 2 * BM * KSTEP * 2; // 16384

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4); // 2 f16 each
    const bf = try b.regs(.b32, NT * 2);

    const rd_a = try b.reg(.b64);
    const rd_b = try b.reg(.b64);
    const rd_c = try b.reg(.b64);
    const r_n = try b.reg(.b32);
    const r_k = try b.reg(.b32);
    const f_scale = if (batched) try b.reg(.f32) else "";
    try b.linef("ld.param.u64 {s}, [p_a];", .{rd_a});
    try b.linef("ld.param.u64 {s}, [p_b];", .{rd_b});
    try b.linef("ld.param.u64 {s}, [p_c];", .{rd_c});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.u32 {s}, [p_k];", .{r_k});
    // scores prefold: the C accumulators are multiplied by p_scale before store,
    // so the f16-C path stores scale·(Q·K). The true score can exceed f16's 65504
    // max (large qk-norm weights → Inf → NaN in softmax); the f32 accumulator
    // holds the true value and the scaled store stays in range. PV passes 1.0.
    if (batched) try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_a, rd_a });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_b, rd_b });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_c, rd_c });

    // Batched variant: gid.z selects an independent GEMM; offset each base
    // pointer by gid.z * per-head stride (element strides passed as params;
    // 64-bit math so 48-head × mpad² offsets don't overflow). A/B are f16
    // (×2 bytes), C is f32 (×4 bytes).
    if (batched) {
        const r_z = try b.reg(.b32);
        const r_st = try b.reg(.b32);
        const rd_off = try b.reg(.b64);
        try b.linef("mov.u32 {s}, %ctaid.z;", .{r_z});
        try b.linef("ld.param.u32 {s}, [p_sa];", .{r_st});
        try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_off, r_z, r_st });
        try b.linef("shl.b64 {s}, {s}, 1;", .{ rd_off, rd_off });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_a, rd_a, rd_off });
        try b.linef("ld.param.u32 {s}, [p_sb];", .{r_st});
        try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_off, r_z, r_st });
        try b.linef("shl.b64 {s}, {s}, 1;", .{ rd_off, rd_off });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_b, rd_b, rd_off });
        try b.linef("ld.param.u32 {s}, [p_sc];", .{r_st});
        try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_off, r_z, r_st });
        try b.linef("shl.b64 {s}, {s}, {d};", .{ rd_off, rd_off, @as(usize, if (c_f16) 1 else 2) });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_c, rd_c, rd_off });
    }

    // attnout: read the MD (max/inv-sum) table + seq + per-head MD row stride,
    // and the log2e / 0 constants used by the per-element exp transform below.
    var rd_md: []const u8 = undefined;
    var r_seq: []const u8 = undefined;
    var r_mds: []const u8 = undefined;
    var r_mds1: []const u8 = undefined;
    var r_zz: []const u8 = undefined;
    var r_l2e: []const u8 = undefined;
    var f_zero: []const u8 = undefined;
    var r_qbase: []const u8 = undefined;
    if (attnout) {
        rd_md = try b.reg(.b64);
        r_seq = try b.reg(.b32);
        r_mds = try b.reg(.b32);
        r_mds1 = try b.reg(.b32);
        r_zz = try b.reg(.b32);
        r_l2e = try b.reg(.f32);
        f_zero = try b.reg(.f32);
        r_qbase = try b.reg(.b32);
        try b.linef("ld.param.u64 {s}, [p_md];", .{rd_md});
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_md, rd_md });
        try b.linef("ld.param.u32 {s}, [p_seq];", .{r_seq});
        try b.linef("ld.param.u32 {s}, [p_mds];", .{r_mds});
        try b.linef("sub.u32 {s}, {s}, 1;", .{ r_mds1, r_mds }); // mpad-1 (clamp)
        try b.linef("mov.u32 {s}, %ctaid.z;", .{r_zz}); // head index
        try b.linef("mov.f32 {s}, 0f3FB8AA3B;", .{r_l2e});
        try b.linef("mov.f32 {s}, 0f00000000;", .{f_zero});
    }

    const r_t = try b.reg(.b32);
    const r_rowq = try b.reg(.b32);
    const r_kq = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("shr.u32 {s}, {s}, 4;", .{ r_rowq, r_t }); // t>>4 (16 u32/row)
    try b.linef("and.b32 {s}, {s}, 15;", .{ r_kq, r_t });
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });
    // attnout: q of the first-staged A row (i=0) = row0 + rowq (rowq = t>>4).
    if (attnout) try b.linef("add.u32 {s}, {s}, {s};", .{ r_qbase, r_row0, r_rowq });

    // staging global byte pointers. row/col in f16; each u32 = 2 f16 = 4 bytes.
    // A base = rd_a + ((row0+rowq)*k + kq*2)*2 ; advance 8 rows = 16*k bytes.
    const r_arow = try b.reg(.b32);
    const r_kq2 = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    const rd_abase = try b.reg(.b64);
    const rd_bbase = try b.reg(.b64);
    const rd_8k = try b.reg(.b64);
    try b.linef("shl.b32 {s}, {s}, 1;", .{ r_kq2, r_kq }); // kq*2 (f16 col)
    try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_8k, r_k }); // 8 rows * k f16 * 2 bytes
    // A
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_arow, r_row0, r_rowq });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_arow, r_k });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_abase, r_kq2 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_abase }); // (row0+rowq)*k + kq2 (f16)
    try b.linef("shl.b64 {s}, {s}, 1;", .{ rd_tmp, rd_tmp }); // *2 bytes
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_abase, rd_a, rd_tmp });
    // B
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_arow, r_col0, r_rowq });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_tmp, r_arow, r_k });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_bbase, r_kq2 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_bbase });
    try b.linef("shl.b64 {s}, {s}, 1;", .{ rd_tmp, rd_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bbase, rd_b, rd_tmp });

    const r_smem = try b.reg(.b32);
    const r_shA = try b.reg(.b32);
    const r_shB = try b.reg(.b32);
    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_shA, r_t }); // t*4 (u32 slot)
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_shA, r_shA, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_shB, r_shA, BS_BASE });

    // fragment lane bases (bytes): row*64 (=32 f16*2) + tf*4.
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, 64;", .{ r_asl, r_tmp }); // row*64 bytes
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_tmp, r_tf }); // tf*4
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, 64;", .{ r_bsl, r_tmp });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_tmp, r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, BS_BASE });

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});

    const r_k0 = try b.reg(.b32);
    const rd_ap = try b.reg(.b64);
    const rd_bp = try b.reg(.b64);
    const rd_k0 = try b.reg(.b64);
    const r_tA = try b.reg(.b32);
    const r_tB = try b.reg(.b32);
    const p0 = try b.reg(.pred);
    // attnout: per-element exp-transform temporaries (declared once, reused).
    var r_j0: []const u8 = undefined;
    var r_j1: []const u8 = undefined;
    var p_j0: []const u8 = undefined;
    var p_j1: []const u8 = undefined;
    var r_qi: []const u8 = undefined;
    var r_mdrow: []const u8 = undefined;
    var rd_mdp: []const u8 = undefined;
    var f_m: []const u8 = undefined;
    var f_inv: []const u8 = undefined;
    var rs_lo: []const u8 = undefined;
    var rs_hi: []const u8 = undefined;
    var f_x0: []const u8 = undefined;
    var f_x1: []const u8 = undefined;
    var f_p0: []const u8 = undefined;
    var f_p1: []const u8 = undefined;
    if (attnout) {
        r_j0 = try b.reg(.b32);
        r_j1 = try b.reg(.b32);
        p_j0 = try b.reg(.pred);
        p_j1 = try b.reg(.pred);
        r_qi = try b.reg(.b32);
        r_mdrow = try b.reg(.b32);
        rd_mdp = try b.reg(.b64);
        f_m = try b.reg(.f32);
        f_inv = try b.reg(.f32);
        rs_lo = try b.reg(.b16);
        rs_hi = try b.reg(.b16);
        f_x0 = try b.reg(.f32);
        f_x1 = try b.reg(.f32);
        f_p0 = try b.reg(.f32);
        f_p1 = try b.reg(.f32);
    }
    try b.linef("mov.u32 {s}, 0;", .{r_k0});
    try b.label("HLOOP");
    try b.linef("mul.wide.u32 {s}, {s}, 2;", .{ rd_k0, r_k0 }); // k0 f16 -> bytes
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_abase, rd_k0 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bbase, rd_k0 });
    if (attnout) {
        // key columns of this k-slab's A pair: j0 = k0 + kq*2, j1 = j0+1 (constant
        // across the row-advancing i-loop); pad keys j>=seq contribute P=0.
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_j0, r_k0, r_kq2 });
        try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_j0, r_j0, r_seq });
        try b.linef("add.u32 {s}, {s}, 1;", .{ r_j1, r_j0 });
        try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_j1, r_j1, r_seq });
    }
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        try b.linef("ld.global.u32 {s}, [{s}];", .{ r_tA, rd_ap });
        if (attnout) {
            // this iteration's A row q = qbase + i*8 (clamp to mpad-1 so the MD read
            // for redundant/pad staging rows stays in-bounds); load {max, 1/sum}.
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_qi, r_qbase, i * 8 });
            try b.linef("min.u32 {s}, {s}, {s};", .{ r_qi, r_qi, r_mds1 });
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_mdrow, r_zz, r_mds, r_qi });
            try b.linef("mul.wide.u32 {s}, {s}, 8;", .{ rd_mdp, r_mdrow });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_mdp, rd_md, rd_mdp });
            try b.linef("ld.global.f32 {s}, [{s}];", .{ f_m, rd_mdp });
            try b.linef("ld.global.f32 {s}, [{s}+4];", .{ f_inv, rd_mdp });
            // unpack the 2 packed f16 scores → P = exp2((S-max)*log2e)*inv → repack.
            try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ rs_lo, rs_hi, r_tA });
            try b.linef("cvt.f32.f16 {s}, {s};", .{ f_x0, rs_lo });
            try b.linef("cvt.f32.f16 {s}, {s};", .{ f_x1, rs_hi });
            try b.linef("sub.f32 {s}, {s}, {s}; mul.f32 {s}, {s}, {s}; ex2.approx.f32 {s}, {s}; mul.f32 {s}, {s}, {s};", .{ f_p0, f_x0, f_m, f_p0, f_p0, r_l2e, f_p0, f_p0, f_p0, f_p0, f_inv });
            try b.linef("sub.f32 {s}, {s}, {s}; mul.f32 {s}, {s}, {s}; ex2.approx.f32 {s}, {s}; mul.f32 {s}, {s}, {s};", .{ f_p1, f_x1, f_m, f_p1, f_p1, r_l2e, f_p1, f_p1, f_p1, f_p1, f_inv });
            try b.linef("selp.f32 {s}, {s}, {s}, {s};", .{ f_p0, f_p0, f_zero, p_j0 });
            try b.linef("selp.f32 {s}, {s}, {s}, {s};", .{ f_p1, f_p1, f_zero, p_j1 });
            try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rs_lo, f_p0 });
            try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rs_hi, f_p1 });
            try b.linef("mov.b32 {s}, {{{s}, {s}}};", .{ r_tA, rs_lo, rs_hi });
        }
        try b.linef("st.shared.u32 [{s}+{d}], {s};", .{ r_shA, i * 512, r_tA });
        try b.linef("ld.global.u32 {s}, [{s}];", .{ r_tB, rd_bp });
        try b.linef("st.shared.u32 [{s}+{d}], {s};", .{ r_shB, i * 512, r_tB });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ap, rd_ap, rd_8k });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bp, rd_bp, rd_8k });
    }
    try b.line("bar.sync 0;");
    var ks: usize = 0;
    while (ks < KS) : (ks += 1) {
        var mi: usize = 0;
        while (mi < MT) : (mi += 1) {
            const o = mi * 1024 + ks * 32; // mi*16 rows*64B + ks*16 f16*2B
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 512 });
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 528 });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = nj * 512 + ks * 32;
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
            try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
        }
        mi = 0;
        while (mi < MT) : (mi += 1) {
            nj = 0;
            while (nj < NT) : (nj += 1) {
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                try b.linef("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                    a[0],           a[1],           a[2],           a[3],
                    af[mi * 4 + 0], af[mi * 4 + 1], af[mi * 4 + 2], af[mi * 4 + 3],
                    bf[nj * 2 + 0], bf[nj * 2 + 1], a[0],           a[1],
                    a[2],           a[3],
                });
            }
        }
    }
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_k0, r_k0, KSTEP });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p0, r_k0, r_k });
    try b.linef("@{s} bra HLOOP;", .{p0});

    // store C [m][n]: f32 (4 B) or, for the scores→softmax path, f16 (2 B) — the
    // accumulators are f32 and converted at store, halving the S write + softmax
    // read traffic (the memory-bound cost at large seq). elem = C element size.
    const elem: usize = if (c_f16) 2 else 4;
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    const rd_8n4 = try b.reg(.b64);
    const rh = try b.regs(.b16, 4);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ r_tmp, r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_8n4, r_n, 8 * elem }); // 8-row byte stride
    const r_row_mi = try b.reg(.b32);
    const r_idx = try b.reg(.b32);
    const rd_cp = try b.reg(.b64);
    const rd_cp2 = try b.reg(.b64);
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_row_mi, r_crow, mi2 * 16 });
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const a = acc[(mi2 * NT + nj) * 4 ..][0..4];
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_idx, r_row_mi, r_n, r_ccol });
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_idx, r_idx, nj * 8 });
            try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_cp, r_idx, elem });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp, rd_c, rd_cp });
            if (batched) {
                // fold the (scores) scale into the accumulator before store.
                for (a) |ai| try b.linef("mul.f32 {s}, {s}, {s};", .{ ai, ai, f_scale });
            }
            if (c_f16) {
                try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rh[0], a[0] });
                try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rh[1], a[1] });
                try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rh[2], a[2] });
                try b.linef("cvt.rn.f16.f32 {s}, {s};", .{ rh[3], a[3] });
                try b.linef("st.global.b16 [{s}], {s};", .{ rd_cp, rh[0] });
                try b.linef("st.global.b16 [{s}+2], {s};", .{ rd_cp, rh[1] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp2, rd_cp, rd_8n4 });
                try b.linef("st.global.b16 [{s}], {s};", .{ rd_cp2, rh[2] });
                try b.linef("st.global.b16 [{s}+2], {s};", .{ rd_cp2, rh[3] });
                continue;
            }
            try b.linef("st.global.f32 [{s}], {s};", .{ rd_cp, a[0] });
            try b.linef("st.global.f32 [{s}+4], {s};", .{ rd_cp, a[1] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_cp2, rd_cp, rd_8n4 });
            try b.linef("st.global.f32 [{s}], {s};", .{ rd_cp2, a[2] });
            try b.linef("st.global.f32 [{s}+4], {s};", .{ rd_cp2, a[3] });
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH_BYTES});
    defer alloc.free(shared_decl);
    const name = if (attnout) "hgemm_attnout" else if (batched and c_f16) "hgemm_batched_c16" else if (batched) "hgemm_batched" else "hgemm";
    const batched_params = "    .param .u64 p_a,\n    .param .u64 p_b,\n    .param .u64 p_c,\n    .param .u32 p_n,\n    .param .u32 p_k,\n    .param .u32 p_sa,\n    .param .u32 p_sb,\n    .param .u32 p_sc,\n    .param .f32 p_scale";
    return b.build(
        name,
        if (attnout)
            batched_params ++ ",\n    .param .u64 p_md,\n    .param .u32 p_seq,\n    .param .u32 p_mds"
        else if (batched)
            batched_params
        else
            "    .param .u64 p_a,\n    .param .u64 p_b,\n    .param .u64 p_c,\n    .param .u32 p_n,\n    .param .u32 p_k",
        shared_decl,
    );
}

/// Row softmax with a prefolded scale, f32 in -> f16 out: for each of `m` rows of
/// width `n` (padded to seq_pad; valid cols 0..seq), P[q][j] = exp(scale*S[q][j] -
/// max) / sum, written f16. One block (256 threads) per row; dynamic shared for
/// the per-thread partials. Entry `softmax_row`. params: p_s(f32 in [m][n]),
/// p_p(f16 out [m][n]), p_n(u32 padded width), p_seq(u32 valid), p_scale(f32).
pub const softmax_row_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.extern .shared .align 8 .b8 smem[];
    \\.visible .entry softmax_row(
    \\    .param .u64 p_s,
    \\    .param .u64 p_p,
    \\    .param .u32 p_n,
    \\    .param .u32 p_seq,
    \\    .param .f32 p_scale
    \\)
    \\{
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<20>;
    \\    .reg .f32 %f<12>;
    \\    .reg .b16 %h<2>;
    \\    .reg .b64 %rd<12>;
    \\    ld.param.u64 %rd1, [p_s];
    \\    ld.param.u64 %rd2, [p_p];
    \\    ld.param.u32 %r1, [p_n];
    \\    ld.param.u32 %r2, [p_seq];
    \\    ld.param.f32 %f1, [p_scale];
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    mov.u32 %r3, %ctaid.x;        // row
    \\    mov.u32 %r4, %tid.x;          // 0..255
    \\    mov.u32 %r5, smem;            // shared base (256 f32 partials)
    \\    // row base byte offsets
    \\    mul.wide.u32 %rd3, %r3, %r1;
    \\    shl.b64 %rd3, %rd3, 2;        // *4 (f32) for S
    \\    add.s64 %rd4, %rd1, %rd3;     // S row
    \\    mul.wide.u32 %rd5, %r3, %r1;
    \\    shl.b64 %rd5, %rd5, 1;        // *2 (f16) for P
    \\    add.s64 %rd6, %rd2, %rd5;     // P row
    \\    // pass 1: thread-local max over j = r4, r4+256, ... < seq
    \\    mov.f32 %f2, 0fFF800000;      // -inf
    \\    mov.u32 %r6, %r4;
    \\MX:
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra MXD;
    \\    mul.wide.u32 %rd7, %r6, 4;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.f32 %f3, [%rd8];
    \\    mul.f32 %f3, %f3, %f1;        // scale
    \\    max.f32 %f2, %f2, %f3;
    \\    add.u32 %r6, %r6, 256;
    \\    bra MX;
    \\MXD:
    \\    // reduce max across 256 threads
    \\    shl.b32 %r7, %r4, 2;
    \\    add.u32 %r7, %r7, %r5;
    \\    st.shared.f32 [%r7], %f2;
    \\    bar.sync 0;
    \\    mov.u32 %r8, 128;
    \\RMX:
    \\    setp.eq.u32 %p2, %r8, 0;
    \\    @%p2 bra RMXD;
    \\    setp.ge.u32 %p1, %r4, %r8;
    \\    @%p1 bra RMXS;
    \\    ld.shared.f32 %f4, [%r7];
    \\    shl.b32 %r9, %r8, 2;
    \\    add.u32 %r9, %r7, %r9;
    \\    ld.shared.f32 %f5, [%r9];
    \\    max.f32 %f4, %f4, %f5;
    \\    st.shared.f32 [%r7], %f4;
    \\RMXS:
    \\    bar.sync 0;
    \\    shr.u32 %r8, %r8, 1;
    \\    bra RMX;
    \\RMXD:
    \\    ld.shared.f32 %f6, [%r5];     // row max (broadcast)
    \\    bar.sync 0;
    \\    // pass 2: thread-local sum of exp(scale*S - max)
    \\    mov.f32 %f7, 0f00000000;
    \\    mov.u32 %r6, %r4;
    \\SM:
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra SMD;
    \\    mul.wide.u32 %rd7, %r6, 4;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.f32 %f3, [%rd8];
    \\    mul.f32 %f3, %f3, %f1;
    \\    sub.f32 %f3, %f3, %f6;
    \\    mul.f32 %f3, %f3, 0f3FB8AA3B;    // * log2(e) for ex2
    \\    ex2.approx.f32 %f8, %f3;      // note: exp via ex2 needs *log2e; done below
    \\    add.f32 %f7, %f7, %f8;
    \\    add.u32 %r6, %r6, 256;
    \\    bra SM;
    \\SMD:
    \\    st.shared.f32 [%r7], %f7;
    \\    bar.sync 0;
    \\    mov.u32 %r8, 128;
    \\RSM:
    \\    setp.eq.u32 %p2, %r8, 0;
    \\    @%p2 bra RSMD;
    \\    setp.ge.u32 %p1, %r4, %r8;
    \\    @%p1 bra RSMS;
    \\    ld.shared.f32 %f4, [%r7];
    \\    shl.b32 %r9, %r8, 2;
    \\    add.u32 %r9, %r7, %r9;
    \\    ld.shared.f32 %f5, [%r9];
    \\    add.f32 %f4, %f4, %f5;
    \\    st.shared.f32 [%r7], %f4;
    \\RSMS:
    \\    bar.sync 0;
    \\    shr.u32 %r8, %r8, 1;
    \\    bra RSM;
    \\RSMD:
    \\    ld.shared.f32 %f9, [%r5];     // row sum
    \\    rcp.approx.f32 %f9, %f9;      // 1/sum
    \\    // pass 3: write P = exp(...)*inv (f16), pad cols -> 0
    \\    mov.u32 %r6, %r4;
    \\WR:
    \\    setp.ge.u32 %p1, %r6, %r1;
    \\    @%p1 bra WRD;
    \\    setp.ge.u32 %p3, %r6, %r2;    // j >= seq -> pad 0
    \\    mul.wide.u32 %rd9, %r6, 2;
    \\    add.s64 %rd10, %rd6, %rd9;
    \\    @%p3 bra ZERO;
    \\    mul.wide.u32 %rd7, %r6, 4;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.f32 %f3, [%rd8];
    \\    mul.f32 %f3, %f3, %f1;
    \\    sub.f32 %f3, %f3, %f6;
    \\    mul.f32 %f3, %f3, 0f3FB8AA3B;    // * log2(e) for ex2
    \\    ex2.approx.f32 %f8, %f3;
    \\    mul.f32 %f8, %f8, %f9;
    \\    cvt.rn.f16.f32 %h0, %f8;
    \\    st.global.b16 [%rd10], %h0;
    \\    bra WRN;
    \\ZERO:
    \\    mov.b16 %h0, 0x0000;
    \\    st.global.b16 [%rd10], %h0;
    \\WRN:
    \\    add.u32 %r6, %r6, 256;
    \\    bra WR;
    \\WRD:
    \\    ret;
    \\}
;

/// f16-input row softmax (S is f16 from the hgemm_batched_c16 scores kernel):
/// identical to `softmax_row` but S is read as b16 + converted, and the S row
/// stride is *2. Halves the S write + all three S reads (the memory-bound cost
/// at large seq). Entry `softmax_row_f16`.
pub const softmax_row_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.extern .shared .align 8 .b8 smem[];
    \\.visible .entry softmax_row_f16(
    \\    .param .u64 p_s,
    \\    .param .u64 p_p,
    \\    .param .u32 p_n,
    \\    .param .u32 p_seq,
    \\    .param .f32 p_scale
    \\)
    \\{
    \\    .reg .pred %p<4>;
    \\    .reg .b32 %r<20>;
    \\    .reg .f32 %f<12>;
    \\    .reg .b16 %h<3>;
    \\    .reg .b64 %rd<12>;
    \\    ld.param.u64 %rd1, [p_s];
    \\    ld.param.u64 %rd2, [p_p];
    \\    ld.param.u32 %r1, [p_n];
    \\    ld.param.u32 %r2, [p_seq];
    \\    ld.param.f32 %f1, [p_scale];
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    mov.u32 %r3, %ctaid.x;
    \\    mov.u32 %r4, %tid.x;
    \\    mov.u32 %r5, smem;
    \\    mul.wide.u32 %rd3, %r3, %r1;
    \\    shl.b64 %rd3, %rd3, 1;        // *2 (f16) for S
    \\    add.s64 %rd4, %rd1, %rd3;
    \\    mul.wide.u32 %rd5, %r3, %r1;
    \\    shl.b64 %rd5, %rd5, 1;        // *2 (f16) for P
    \\    add.s64 %rd6, %rd2, %rd5;
    \\    mov.f32 %f2, 0fFF800000;
    \\    mov.u32 %r6, %r4;
    \\MX:
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra MXD;
    \\    mul.wide.u32 %rd7, %r6, 2;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.b16 %h1, [%rd8]; cvt.f32.f16 %f3, %h1;
    \\    mul.f32 %f3, %f3, %f1;
    \\    max.f32 %f2, %f2, %f3;
    \\    add.u32 %r6, %r6, 256;
    \\    bra MX;
    \\MXD:
    \\    shl.b32 %r7, %r4, 2;
    \\    add.u32 %r7, %r7, %r5;
    \\    st.shared.f32 [%r7], %f2;
    \\    bar.sync 0;
    \\    mov.u32 %r8, 128;
    \\RMX:
    \\    setp.eq.u32 %p2, %r8, 0;
    \\    @%p2 bra RMXD;
    \\    setp.ge.u32 %p1, %r4, %r8;
    \\    @%p1 bra RMXS;
    \\    ld.shared.f32 %f4, [%r7];
    \\    shl.b32 %r9, %r8, 2;
    \\    add.u32 %r9, %r7, %r9;
    \\    ld.shared.f32 %f5, [%r9];
    \\    max.f32 %f4, %f4, %f5;
    \\    st.shared.f32 [%r7], %f4;
    \\RMXS:
    \\    bar.sync 0;
    \\    shr.u32 %r8, %r8, 1;
    \\    bra RMX;
    \\RMXD:
    \\    ld.shared.f32 %f6, [%r5];
    \\    bar.sync 0;
    \\    mov.f32 %f7, 0f00000000;
    \\    mov.u32 %r6, %r4;
    \\SM:
    \\    setp.ge.u32 %p1, %r6, %r2;
    \\    @%p1 bra SMD;
    \\    mul.wide.u32 %rd7, %r6, 2;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.b16 %h1, [%rd8]; cvt.f32.f16 %f3, %h1;
    \\    mul.f32 %f3, %f3, %f1;
    \\    sub.f32 %f3, %f3, %f6;
    \\    mul.f32 %f3, %f3, 0f3FB8AA3B;
    \\    ex2.approx.f32 %f8, %f3;
    \\    add.f32 %f7, %f7, %f8;
    \\    add.u32 %r6, %r6, 256;
    \\    bra SM;
    \\SMD:
    \\    st.shared.f32 [%r7], %f7;
    \\    bar.sync 0;
    \\    mov.u32 %r8, 128;
    \\RSM:
    \\    setp.eq.u32 %p2, %r8, 0;
    \\    @%p2 bra RSMD;
    \\    setp.ge.u32 %p1, %r4, %r8;
    \\    @%p1 bra RSMS;
    \\    ld.shared.f32 %f4, [%r7];
    \\    shl.b32 %r9, %r8, 2;
    \\    add.u32 %r9, %r7, %r9;
    \\    ld.shared.f32 %f5, [%r9];
    \\    add.f32 %f4, %f4, %f5;
    \\    st.shared.f32 [%r7], %f4;
    \\RSMS:
    \\    bar.sync 0;
    \\    shr.u32 %r8, %r8, 1;
    \\    bra RSM;
    \\RSMD:
    \\    ld.shared.f32 %f9, [%r5];
    \\    rcp.approx.f32 %f9, %f9;
    \\    mov.u32 %r6, %r4;
    \\WR:
    \\    setp.ge.u32 %p1, %r6, %r1;
    \\    @%p1 bra WRD;
    \\    setp.ge.u32 %p3, %r6, %r2;
    \\    mul.wide.u32 %rd9, %r6, 2;
    \\    add.s64 %rd10, %rd6, %rd9;
    \\    @%p3 bra ZERO;
    \\    mul.wide.u32 %rd7, %r6, 2;
    \\    add.s64 %rd8, %rd4, %rd7;
    \\    ld.global.b16 %h1, [%rd8]; cvt.f32.f16 %f3, %h1;
    \\    mul.f32 %f3, %f3, %f1;
    \\    sub.f32 %f3, %f3, %f6;
    \\    mul.f32 %f3, %f3, 0f3FB8AA3B;
    \\    ex2.approx.f32 %f8, %f3;
    \\    mul.f32 %f8, %f8, %f9;
    \\    cvt.rn.f16.f32 %h0, %f8;
    \\    st.global.b16 [%rd10], %h0;
    \\    bra WRN;
    \\ZERO:
    \\    mov.b16 %h0, 0x0000;
    \\    st.global.b16 [%rd10], %h0;
    \\WRN:
    \\    add.u32 %r6, %r6, 256;
    \\    bra WR;
    \\WRD:
    \\    ret;
    \\}
;

/// Single-pass "flash" softmax reduction: for each row of S (f16 [rows][pn],
/// scale already folded in by the scores GEMM), read S ONCE and emit per-row
/// MD[row] = {max, 1/sum} (f32 pair). No P materialization — the fused attn-out
/// GEMM (`hgemm_attnout`) recomputes P = exp(S-max)/sum from S + this MD during
/// its A-staging. One block (256) per row; the block-reduce combines running
/// (m, d) partials the FlashAttention way: M=max(mᵢ), D=Σ dᵢ·exp2((mᵢ-M)·log2e).
///
/// The running-max is initialised to -FLT_MAX (not -inf) so that combining two
/// empty partials gives m-M = 0 (finite) rather than -inf-(-inf) = NaN; every
/// real score is > -FLT_MAX so the max result is unchanged, and empty lanes
/// (d=0) contribute 0·anything = 0. (In practice attention seq ≥ 256 so every
/// lane has a valid column, but the sentinel keeps it robust for any seq ≥ 1.)
/// Entry `softmax_md_f16`. params: p_s(f16 [rows][pn]), p_md(f32 [rows][2]),
/// p_n(u32 pn=mpad), p_seq(u32 valid cols). grid=(rows,1,1), block=256.
pub const softmax_md_f16_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry softmax_md_f16(
    \\    .param .u64 p_s,
    \\    .param .u64 p_md,
    \\    .param .u32 p_n,
    \\    .param .u32 p_seq
    \\)
    \\{
    \\    .reg .pred %p<5>;
    \\    .reg .b32 %r<16>;
    \\    .reg .f32 %f<24>;
    \\    .reg .b16 %h<2>;
    \\    .reg .b64 %rd<10>;
    \\    .shared .align 4 .b8 sm[2048];        // [0,1024) m partials, [1024,2048) d partials
    \\    ld.param.u64 %rd1, [p_s];
    \\    ld.param.u64 %rd2, [p_md];
    \\    ld.param.u32 %r1, [p_n];              // pn (mpad)
    \\    ld.param.u32 %r2, [p_seq];            // seq (valid cols)
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    mov.u32 %r3, %ctaid.x;                // row
    \\    mov.u32 %r4, %tid.x;                  // 0..255
    \\    mov.f32 %f10, 0f3FB8AA3B;             // log2(e)
    \\    mul.wide.u32 %rd3, %r3, %r1;
    \\    shl.b64 %rd3, %rd3, 1;                // *2 (f16) S row byte offset
    \\    add.s64 %rd4, %rd1, %rd3;             // S row ptr
    \\    // per-thread online reduction over j = tid, tid+256, ... < seq (one S read)
    \\    mov.f32 %f1, 0fFF7FFFFF;              // ml = -FLT_MAX (finite sentinel)
    \\    mov.f32 %f2, 0f00000000;              // dl = 0
    \\    mov.u32 %r5, %r4;                     // j
    \\LP:
    \\    setp.ge.u32 %p1, %r5, %r2;
    \\    @%p1 bra LPD;
    \\    mul.wide.u32 %rd5, %r5, 2;
    \\    add.s64 %rd6, %rd4, %rd5;
    \\    ld.global.b16 %h0, [%rd6];
    \\    cvt.f32.f16 %f3, %h0;                 // x
    \\    max.f32 %f4, %f1, %f3;                // m2
    \\    sub.f32 %f5, %f1, %f4; mul.f32 %f5, %f5, %f10; ex2.approx.f32 %f5, %f5;  // exp2((ml-m2)*log2e)
    \\    sub.f32 %f6, %f3, %f4; mul.f32 %f6, %f6, %f10; ex2.approx.f32 %f6, %f6;  // exp2((x-m2)*log2e)
    \\    fma.rn.f32 %f2, %f2, %f5, %f6;        // dl = dl*c1 + c2
    \\    mov.f32 %f1, %f4;                     // ml = m2
    \\    add.u32 %r5, %r5, 256;
    \\    bra LP;
    \\LPD:
    \\    mov.u32 %r6, sm;
    \\    shl.b32 %r7, %r4, 2;                  // tid*4
    \\    add.u32 %r8, %r6, %r7;                // m slot = sm + tid*4
    \\    st.shared.f32 [%r8], %f1;
    \\    st.shared.f32 [%r8+1024], %f2;        // d slot
    \\    bar.sync 0;
    \\    mov.u32 %r9, 128;                     // reduction offset
    \\RED:
    \\    setp.eq.u32 %p2, %r9, 0;
    \\    @%p2 bra REDD;
    \\    setp.ge.u32 %p3, %r4, %r9;
    \\    @%p3 bra REDS;
    \\    ld.shared.f32 %f11, [%r8];            // m_a
    \\    ld.shared.f32 %f12, [%r8+1024];       // d_a
    \\    shl.b32 %r10, %r9, 2;
    \\    add.u32 %r11, %r8, %r10;              // partner slot
    \\    ld.shared.f32 %f13, [%r11];           // m_b
    \\    ld.shared.f32 %f14, [%r11+1024];      // d_b
    \\    max.f32 %f15, %f11, %f13;             // M
    \\    sub.f32 %f16, %f11, %f15; mul.f32 %f16, %f16, %f10; ex2.approx.f32 %f16, %f16;
    \\    sub.f32 %f17, %f13, %f15; mul.f32 %f17, %f17, %f10; ex2.approx.f32 %f17, %f17;
    \\    mul.f32 %f18, %f12, %f16;
    \\    fma.rn.f32 %f18, %f14, %f17, %f18;    // D = d_a*c_a + d_b*c_b
    \\    st.shared.f32 [%r8], %f15;
    \\    st.shared.f32 [%r8+1024], %f18;
    \\REDS:
    \\    bar.sync 0;
    \\    shr.u32 %r9, %r9, 1;
    \\    bra RED;
    \\REDD:
    \\    setp.ne.u32 %p4, %r4, 0;
    \\    @%p4 bra END;
    \\    ld.shared.f32 %f19, [%r6];            // M = sm[0]
    \\    ld.shared.f32 %f20, [%r6+1024];       // D
    \\    rcp.approx.f32 %f20, %f20;            // 1/sum
    \\    mul.wide.u32 %rd7, %r3, 8;            // MD[row] byte offset (2 f32)
    \\    add.s64 %rd8, %rd2, %rd7;
    \\    st.global.f32 [%rd8], %f19;
    \\    st.global.f32 [%rd8+4], %f20;
    \\END:
    \\    ret;
    \\}
;


const gpa = std.heap.page_allocator;

/// Which GEMM kernel a case exercises.
const Kernel = enum { v0, smem, pipe };

/// int8 IMMA GEMM validation (exact integer match vs a CPU reference) + timing.
pub fn i8GemmTest(ctx: *Context, io: anytype, stdout: anytype) !void {
    var mod0 = try ctx.loadModule(igemm_v0_ptx);
    defer mod0.unload(ctx);
    const f_v0 = try mod0.getFunction(ctx, "igemm_v0");

    // Build + JIT the tiled shared-memory kernel; dump its PTX for offline
    // inspection so a JIT failure is debuggable.
    const smem_ptx = try buildIgemmSmem(gpa);
    defer gpa.free(smem_ptx);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "/tmp/claude-1000/-dump-projects-zig-TensorPencil/eccfce6f-7c1f-4c32-b182-cc9c60d44a58/scratchpad/igemm_smem.gen.ptx", .data = smem_ptx }) catch {};
    var mod1 = try ctx.loadModule(smem_ptx);
    defer mod1.unload(ctx);
    const f_smem = try mod1.getFunction(ctx, "igemm_smem");
    const pipe_ptx = try buildIgemmPipe(gpa, 64, false, 8);
    defer gpa.free(pipe_ptx);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "/tmp/claude-1000/-dump-projects-zig-TensorPencil/eccfce6f-7c1f-4c32-b182-cc9c60d44a58/scratchpad/igemm_pipe.gen.ptx", .data = pipe_ptx }) catch {};
    var mod2 = try ctx.loadModule(pipe_ptx);
    defer mod2.unload(ctx);
    const f_pipe = try mod2.getFunction(ctx, "igemm_pipe");
    {
        var rs: c_int = 0;
        var rp: c_int = 0;
        _ = ctx.api.cuFuncGetAttribute(&rs, cu.CU_FUNC_ATTRIBUTE_NUM_REGS, f_smem);
        _ = ctx.api.cuFuncGetAttribute(&rp, cu.CU_FUNC_ATTRIBUTE_NUM_REGS, f_pipe);
        try stdout.print("regs/thread: smem={d} pipe={d}\n", .{ rs, rp });
    }

    const Case = struct { m: usize, n: usize, k: usize, check: bool, kern: Kernel };
    const cases = [_]Case{
        .{ .m = 16, .n = 8, .k = 32, .check = true, .kern = .v0 },
        .{ .m = 64, .n = 256, .k = 128, .check = true, .kern = .v0 },
        .{ .m = 128, .n = 128, .k = 64, .check = true, .kern = .smem },
        .{ .m = 256, .n = 384, .k = 320, .check = true, .kern = .smem },
        .{ .m = 128, .n = 256, .k = 6144, .check = true, .kern = .smem },
        .{ .m = 128, .n = 128, .k = 64, .check = true, .kern = .pipe },
        .{ .m = 256, .n = 384, .k = 320, .check = true, .kern = .pipe },
        .{ .m = 128, .n = 256, .k = 6144, .check = true, .kern = .pipe },
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false, .kern = .smem },
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false, .kern = .pipe },
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false, .kern = .smem }, // DiT qkv @1120x1680
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false, .kern = .pipe },
        .{ .m = 7680, .n = 16384, .k = 6144, .check = false, .kern = .pipe }, // mlp gate/up
        .{ .m = 7680, .n = 6144, .k = 16384, .check = false, .kern = .pipe }, // mlp.down
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const timer = try ctx.timerCreate();
    defer ctx.timerDestroy(timer);

    for (cases) |c| {
        const m = c.m;
        const n = c.n;
        const k = c.k;
        const ab = try gpa.alloc(u8, m * k);
        defer gpa.free(ab);
        const bb = try gpa.alloc(u8, n * k);
        defer gpa.free(bb);
        for (ab) |*v| v.* = @bitCast(rand.int(i8));
        for (bb) |*v| v.* = @bitCast(rand.int(i8));

        var da = try ctx.alloc(m * k);
        defer ctx.free(&da);
        var db = try ctx.alloc(n * k);
        defer ctx.free(&db);
        var dc = try ctx.alloc(m * n * 4);
        defer ctx.free(&dc);
        try ctx.upload(da, ab);
        try ctx.upload(db, bb);

        var pa = da.ptr;
        var pb = db.ptr;
        var pc = dc.ptr;
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(k);
        var params = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };

        const f = switch (c.kern) {
            .v0 => f_v0,
            .smem => f_smem,
            .pipe => f_pipe,
        };
        const grid: [3]u32 = switch (c.kern) {
            .v0 => .{ @intCast(n / 8), @intCast(m / 16), 1 },
            .smem, .pipe => .{ @intCast(n / 128), @intCast(m / 128), 1 },
        };
        const block: [3]u32 = switch (c.kern) {
            .v0 => .{ 32, 1, 1 },
            .smem, .pipe => .{ 128, 1, 1 },
        };
        const tag = @tagName(c.kern);

        if (c.check) {
            try ctx.launch(f, grid, block, 0, &params);
            const cg = try gpa.alloc(u8, m * n * 4);
            defer gpa.free(cg);
            try ctx.download(dc, cg);
            const ci: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, cg));
            var mism: usize = 0;
            for (0..m) |i| {
                for (0..n) |j| {
                    var acc: i32 = 0;
                    for (0..k) |kk| {
                        acc += @as(i32, @as(i8, @bitCast(ab[i * k + kk]))) * @as(i32, @as(i8, @bitCast(bb[j * k + kk])));
                    }
                    if (ci[i * n + j] != acc) {
                        if (mism < 5) try stdout.print("  MISMATCH [{d},{d}]: gpu={d} cpu={d}\n", .{ i, j, ci[i * n + j], acc });
                        mism += 1;
                    }
                }
            }
            try stdout.print("{s} {d}x{d}x{d}: {d}/{d} mismatches\n", .{ tag, m, n, k, mism, m * n });
            if (mism != 0) return error.CudaError;
        } else {
            try ctx.launch(f, grid, block, 0, &params);
            try ctx.synchronize();
            var best: f32 = std.math.floatMax(f32);
            for (0..12) |_| {
                try ctx.timerBegin(timer);
                try ctx.launch(f, grid, block, 0, &params);
                const ms = try ctx.timerEndMs(timer);
                best = @min(best, ms);
            }
            const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * n * k));
            try stdout.print("{s} {d}x{d}x{d}: {d:.3} ms, {d:.1} GOP/s (min of 12)\n", .{ tag, m, n, k, best, flops / (best * 1e6) });
        }
    }
}

/// Validate the raw int4 tensor-core GEMM (s4*s4->s32) against a CPU oracle.
/// Random s4 [-8,7] operands, nibble-packed 2/byte, checked exactly (integer
/// mma is bit-exact). This proves the m16n8k64.s4 fragment layout on this GPU.
pub fn i4GemmTest(ctx: *Context, io: anytype, stdout: anytype) !void {
    _ = io;
    var mod0 = try ctx.loadModule(i4gemm_v0_ptx);
    defer mod0.unload(ctx);
    const f_v0 = try mod0.getFunction(ctx, "i4gemm_v0");

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{
        .{ .m = 16, .n = 8, .k = 64 },
        .{ .m = 64, .n = 256, .k = 128 },
        .{ .m = 128, .n = 128, .k = 256 },
        .{ .m = 48, .n = 72, .k = 320 },
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    for (cases) |c| {
        const m = c.m;
        const n = c.n;
        const k = c.k;
        // Unpacked s4 values (as i8 in [-8,7]) for the oracle, plus the packed
        // nibble buffers the kernel reads.
        const au = try gpa.alloc(i8, m * k);
        defer gpa.free(au);
        const bu = try gpa.alloc(i8, n * k);
        defer gpa.free(bu);
        for (au) |*v| v.* = rand.intRangeAtMost(i4, -8, 7);
        for (bu) |*v| v.* = rand.intRangeAtMost(i4, -8, 7);
        const ab = try gpa.alloc(u8, m * k / 2);
        defer gpa.free(ab);
        const bb = try gpa.alloc(u8, n * k / 2);
        defer gpa.free(bb);
        for (ab, 0..) |*p, i| p.* = @as(u8, @as(u4, @bitCast(@as(i4, @intCast(au[2 * i]))))) |
            (@as(u8, @as(u4, @bitCast(@as(i4, @intCast(au[2 * i + 1]))))) << 4);
        for (bb, 0..) |*p, i| p.* = @as(u8, @as(u4, @bitCast(@as(i4, @intCast(bu[2 * i]))))) |
            (@as(u8, @as(u4, @bitCast(@as(i4, @intCast(bu[2 * i + 1]))))) << 4);

        var da = try ctx.alloc(m * k / 2);
        defer ctx.free(&da);
        var db = try ctx.alloc(n * k / 2);
        defer ctx.free(&db);
        var dc = try ctx.alloc(m * n * 4);
        defer ctx.free(&dc);
        try ctx.upload(da, ab);
        try ctx.upload(db, bb);

        var pa = da.ptr;
        var pb = db.ptr;
        var pc = dc.ptr;
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(k);
        var params = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };
        try ctx.launch(f_v0, .{ @intCast(n / 8), @intCast(m / 16), 1 }, .{ 32, 1, 1 }, 0, &params);

        const cg = try gpa.alloc(u8, m * n * 4);
        defer gpa.free(cg);
        try ctx.download(dc, cg);
        const ci: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, cg));
        var mism: usize = 0;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: i32 = 0;
                for (0..k) |kk| acc += @as(i32, au[i * k + kk]) * @as(i32, bu[j * k + kk]);
                if (ci[i * n + j] != acc) {
                    if (mism < 5) try stdout.print("  MISMATCH [{d},{d}]: gpu={d} cpu={d}\n", .{ i, j, ci[i * n + j], acc });
                    mism += 1;
                }
            }
        }
        try stdout.print("i4 v0 {d}x{d}x{d}: {d}/{d} mismatches\n", .{ m, n, k, mism, m * n });
        if (mism != 0) return error.CudaError;
    }
}

/// Full int4 (W4A4) convrot linear validation: i4prep (rotate + per-row
/// quantize x to s4 [-8,7], pack 2/byte) -> s4 tensor-core GEMM (v0) ->
/// irescale, checked against a CPU replica. rel-vs-cpu-sim = wiring exactness;
/// rel-vs-f32 = int4 accuracy (naturally coarser than int8).
pub fn i4LinearTest(ctx: *Context, io: anytype, stdout: anytype) !void {
    _ = io;
    const convrot = @import("../../ops/convrot.zig");

    // Performant fused s4 GEMM (rescale folded into the C-store), the path
    // dit_cuda's opI4Gemm uses. Requires m%128, rows%128, cols%128.
    const pipe_ptx = try buildIgemmPipe(gpa, 64, true, 4);
    defer gpa.free(pipe_ptx);
    var gmod = try ctx.loadModule(pipe_ptx);
    defer gmod.unload(ctx);
    const f_gemm = try gmod.getFunction(ctx, "i4gemm_pipe_fused");

    const LCase = struct { m: usize, rows: usize, cols: usize };
    const lcases = [_]LCase{
        .{ .m = 128, .rows = 128, .cols = 2048 },
        .{ .m = 128, .rows = 256, .cols = 6144 },
        .{ .m = 128, .rows = 128, .cols = 16384 }, // mlp.down cols, >48KB shared in prep
    };
    var prng = std.Random.DefaultPrng.init(9);
    const rand = prng.random();

    for (lcases) |c| {
        const m = c.m;
        const rows = c.rows;
        const cols = c.cols;

        const prep_ptx = try buildPrep(gpa, cols, 4, false);
        defer gpa.free(prep_ptx);
        var pmod = try ctx.loadModule(prep_ptx);
        defer pmod.unload(ctx);
        const f_prep = try pmod.getFunction(ctx, "i4prep");
        const shb = prepSharedBytes(cols);
        try ctx.setMaxDynamicShared(f_prep, shb);

        const xf = try gpa.alloc(f32, m * cols);
        defer gpa.free(xf);
        for (xf) |*v| v.* = rand.floatNorm(f32);
        // Pre-rotated int4 weight, unpacked oracle + nibble-packed device bytes.
        const wu = try gpa.alloc(i8, rows * cols);
        defer gpa.free(wu);
        for (wu) |*v| v.* = rand.intRangeAtMost(i4, -8, 7);
        const wb = try gpa.alloc(u8, rows * cols / 2);
        defer gpa.free(wb);
        for (wb, 0..) |*p, i| p.* = @as(u8, @as(u4, @bitCast(@as(i4, @intCast(wu[2 * i]))))) |
            (@as(u8, @as(u4, @bitCast(@as(i4, @intCast(wu[2 * i + 1]))))) << 4);
        const wscale = try gpa.alloc(f32, rows);
        defer gpa.free(wscale);
        for (wscale) |*s| s.* = 0.001 + rand.float(f32) * 0.02;

        var x_d = try ctx.alloc(m * cols * 4);
        defer ctx.free(&x_d);
        var q_d = try ctx.alloc(m * cols / 2); // packed int4 activations
        defer ctx.free(&q_d);
        var as_d = try ctx.alloc(m * 4);
        defer ctx.free(&as_d);
        var w_d = try ctx.alloc(rows * cols / 2);
        defer ctx.free(&w_d);
        var ws_d = try ctx.alloc(rows * 4);
        defer ctx.free(&ws_d);
        var y_d = try ctx.alloc(m * rows * 4);
        defer ctx.free(&y_d);
        try ctx.upload(x_d, std.mem.sliceAsBytes(xf));
        try ctx.upload(w_d, wb);
        try ctx.upload(ws_d, std.mem.sliceAsBytes(wscale));

        // prep: x[m][cols] f32 -> q_d (packed s4) + as_d (per-row scale).
        var px = x_d.ptr;
        var pq = q_d.ptr;
        var pas = as_d.ptr;
        var pp = [_]?*anyopaque{ @ptrCast(&px), @ptrCast(&pq), @ptrCast(&pas) };
        try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
        // fused gemm: A=q [m][cols], B=w [rows][cols] (packed s4) -> y [m][rows]
        // f32, rescale (act_scale[row]*weight_scale[col]) folded into the store.
        var pa = q_d.ptr;
        var pb = w_d.ptr;
        var pc = y_d.ptr;
        var pn: u32 = @intCast(rows);
        var pk: u32 = @intCast(cols);
        var pas2 = as_d.ptr;
        var pws2 = ws_d.ptr;
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk), @ptrCast(&pas2), @ptrCast(&pws2) };
        try ctx.launch(f_gemm, .{ @intCast(rows / 128), @intCast(m / 128), 1 }, .{ 128, 1, 1 }, 0, &pg);

        const yg = try gpa.alloc(u8, m * rows * 4);
        defer gpa.free(yg);
        try ctx.download(y_d, yg);
        const y_gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yg));

        // CPU replica of the same int4 pipeline.
        const xr = try gpa.dupe(f32, xf);
        defer gpa.free(xr);
        const xi4 = try gpa.alloc(i8, m * cols);
        defer gpa.free(xi4);
        const ascale = try gpa.alloc(f32, m);
        defer gpa.free(ascale);
        for (0..m) |i| {
            convrot.rotate(xr[i * cols ..][0..cols]);
            var amax: f32 = 0;
            for (xr[i * cols ..][0..cols]) |v| amax = @max(amax, @abs(v));
            const s = @max(amax / 7.0, 1e-12);
            ascale[i] = s;
            for (0..cols) |k| {
                var qi: i32 = @intFromFloat(@round(xr[i * cols + k] / s));
                qi = @max(@as(i32, -8), @min(@as(i32, 7), qi));
                xi4[i * cols + k] = @intCast(qi);
            }
        }
        var num_sim: f64 = 0;
        var num_truth: f64 = 0;
        var den: f64 = 0;
        for (0..m) |i| {
            for (0..rows) |j| {
                var acc: i32 = 0;
                var truth: f64 = 0;
                for (0..cols) |k| {
                    acc += @as(i32, xi4[i * cols + k]) * @as(i32, wu[j * cols + k]);
                    truth += @as(f64, xr[i * cols + k]) * (@as(f64, @floatFromInt(wu[j * cols + k])) * wscale[j]);
                }
                const sim: f64 = @as(f64, @floatFromInt(acc)) * ascale[i] * wscale[j];
                const g: f64 = y_gpu[i * rows + j];
                num_sim += (g - sim) * (g - sim);
                num_truth += (g - truth) * (g - truth);
                den += truth * truth;
            }
        }
        const rel_sim = @sqrt(num_sim / den);
        const rel_truth = @sqrt(num_truth / den);
        try stdout.print("i4 linear {d}x{d}x{d}: rel vs cpu-sim {d:.6} (wiring), rel vs f32 {d:.4} (int4 acc)\n", .{ m, rows, cols, rel_sim, rel_truth });
        try stdout.flush();
        if (rel_sim > 1e-3) return error.CudaError;
    }
}

/// Full int8 convrot linear validation: prep (rotate+quantize) -> IMMA GEMM ->
/// rescale, checked against the same CPU replica as `gpu-i8-test`
/// (rel-vs-cpu-sim = wiring exactness; rel-vs-f32 = int8 accuracy). Exercises
/// >48 KB dynamic shared for cols=16384.
pub fn i8LinearTest(ctx: *Context, io: anytype, stdout: anytype) !void {
    const convrot = @import("../../ops/convrot.zig");

    const pipe_ptx = try buildIgemmPipe(gpa, 64, false, 8);
    defer gpa.free(pipe_ptx);
    var gmod = try ctx.loadModule(pipe_ptx);
    defer gmod.unload(ctx);
    const f_gemm = try gmod.getFunction(ctx, "igemm_pipe");

    var rmod = try ctx.loadModule(irescale_ptx);
    defer rmod.unload(ctx);
    const f_rescale = try rmod.getFunction(ctx, "irescale");

    // Stage-A fused GEMM: rescale folded into the C-store (no s32 acc buffer,
    // no separate rescale pass), output f32 y directly.
    const fused_ptx = try buildIgemmPipe(gpa, 64, true, 8);
    defer gpa.free(fused_ptx);
    var fmod = try ctx.loadModule(fused_ptx);
    defer fmod.unload(ctx);
    const f_fused = try fmod.getFunction(ctx, "igemm_pipe_fused");

    const LCase = struct { m: usize, rows: usize, cols: usize };
    const lcases = [_]LCase{
        .{ .m = 128, .rows = 128, .cols = 1024 },
        .{ .m = 128, .rows = 256, .cols = 6144 },
        .{ .m = 128, .rows = 128, .cols = 16384 }, // mlp.down cols, f32 rotation in >48KB shared
        .{ .m = 256, .rows = 384, .cols = 6144 },
    };
    var prng = std.Random.DefaultPrng.init(9);
    const rand = prng.random();

    for (lcases) |c| {
        const m = c.m;
        const rows = c.rows;
        const cols = c.cols;

        const prep_ptx = try buildPrep(gpa, cols, 8, false);
        defer gpa.free(prep_ptx);
        var pmod = try ctx.loadModule(prep_ptx);
        defer pmod.unload(ctx);
        const f_prep = try pmod.getFunction(ctx, "iprep");
        const shb = prepSharedBytes(cols);
        try ctx.setMaxDynamicShared(f_prep, shb);

        const xf = try gpa.alloc(f32, m * cols);
        defer gpa.free(xf);
        for (xf) |*v| v.* = rand.floatNorm(f32);
        const wb = try gpa.alloc(u8, rows * cols); // pre-rotated int8 weight
        defer gpa.free(wb);
        for (wb) |*v| v.* = @bitCast(rand.int(i8));
        const wscale = try gpa.alloc(f32, rows);
        defer gpa.free(wscale);
        for (wscale) |*s| s.* = 0.001 + rand.float(f32) * 0.02;

        var x_d = try ctx.alloc(m * cols * 4);
        defer ctx.free(&x_d);
        var q_d = try ctx.alloc(m * cols); // packed int8
        defer ctx.free(&q_d);
        var as_d = try ctx.alloc(m * 4); // act_scale
        defer ctx.free(&as_d);
        var w_d = try ctx.alloc(rows * cols);
        defer ctx.free(&w_d);
        var ws_d = try ctx.alloc(rows * 4);
        defer ctx.free(&ws_d);
        var acc_d = try ctx.alloc(m * rows * 4);
        defer ctx.free(&acc_d);
        var y_d = try ctx.alloc(m * rows * 4);
        defer ctx.free(&y_d);
        try ctx.upload(x_d, std.mem.sliceAsBytes(xf));
        try ctx.upload(w_d, wb);
        try ctx.upload(ws_d, std.mem.sliceAsBytes(wscale));

        // prep
        var px = x_d.ptr;
        var pq = q_d.ptr;
        var pas = as_d.ptr;
        var pp = [_]?*anyopaque{ @ptrCast(&px), @ptrCast(&pq), @ptrCast(&pas) };
        try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
        // gemm: A=q [m][cols], B=w [rows][cols] -> acc [m][rows]
        var pa = q_d.ptr;
        var pb = w_d.ptr;
        var pc = acc_d.ptr;
        var pn: u32 = @intCast(rows);
        var pk: u32 = @intCast(cols);
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };
        try ctx.launch(f_gemm, .{ @intCast(rows / 128), @intCast(m / 128), 1 }, .{ 128, 1, 1 }, 0, &pg);
        // rescale
        const total: u32 = @intCast(m * rows);
        var racc = acc_d.ptr;
        var ry = y_d.ptr;
        var ras = as_d.ptr;
        var rws = ws_d.ptr;
        var rrows: u32 = @intCast(rows);
        var rtot: u32 = total;
        var pr = [_]?*anyopaque{ @ptrCast(&racc), @ptrCast(&ry), @ptrCast(&ras), @ptrCast(&rws), @ptrCast(&rrows), @ptrCast(&rtot) };
        try ctx.launch(f_rescale, .{ (total + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, 0, &pr);

        const yg = try gpa.alloc(u8, m * rows * 4);
        defer gpa.free(yg);
        try ctx.download(y_d, yg);
        const y_gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yg));

        // CPU replica of the same int8 pipeline (mirrors main.zig gpuI8Test).
        const xr = try gpa.dupe(f32, xf);
        defer gpa.free(xr);
        const xi8 = try gpa.alloc(i8, m * cols);
        defer gpa.free(xi8);
        const ascale = try gpa.alloc(f32, m);
        defer gpa.free(ascale);
        for (0..m) |i| {
            convrot.rotate(xr[i * cols ..][0..cols]);
            var amax: f32 = 0;
            for (xr[i * cols ..][0..cols]) |v| amax = @max(amax, @abs(v));
            const s = @max(amax / 127.0, 1e-12);
            ascale[i] = s;
            for (0..cols) |k| {
                var qi: i32 = @intFromFloat(@round(xr[i * cols + k] / s));
                qi = @max(@as(i32, -127), @min(@as(i32, 127), qi));
                xi8[i * cols + k] = @intCast(qi);
            }
        }
        const sim_arr = try gpa.alloc(f64, m * rows);
        defer gpa.free(sim_arr);
        var num_sim: f64 = 0;
        var num_truth: f64 = 0;
        var den: f64 = 0;
        for (0..m) |i| {
            for (0..rows) |j| {
                var acc: i32 = 0;
                var truth: f64 = 0;
                for (0..cols) |k| {
                    acc += @as(i32, xi8[i * cols + k]) * @as(i32, @as(i8, @bitCast(wb[j * cols + k])));
                    truth += @as(f64, xr[i * cols + k]) * (@as(f64, @floatFromInt(@as(i8, @bitCast(wb[j * cols + k])))) * wscale[j]);
                }
                const sim: f64 = @as(f64, @floatFromInt(acc)) * ascale[i] * wscale[j];
                sim_arr[i * rows + j] = sim;
                const g: f64 = y_gpu[i * rows + j];
                num_sim += (g - sim) * (g - sim);
                num_truth += (g - truth) * (g - truth);
                den += truth * truth;
            }
        }
        const rel_sim = @sqrt(num_sim / den);
        const rel_truth = @sqrt(num_truth / den);

        // Stage-A fused GEMM path: prep -> fused gemm (rescale in C-store) -> y.
        var fa2 = q_d.ptr;
        var fb2 = w_d.ptr;
        var fc2 = y_d.ptr;
        var fn2: u32 = @intCast(rows);
        var fk2: u32 = @intCast(cols);
        var fas = as_d.ptr;
        var fws = ws_d.ptr;
        var pf = [_]?*anyopaque{ @ptrCast(&fa2), @ptrCast(&fb2), @ptrCast(&fc2), @ptrCast(&fn2), @ptrCast(&fk2), @ptrCast(&fas), @ptrCast(&fws) };
        try ctx.launch(f_fused, .{ @intCast(rows / 128), @intCast(m / 128), 1 }, .{ 128, 1, 1 }, 0, &pf);
        try ctx.download(y_d, yg);
        var num_fused: f64 = 0;
        for (0..m * rows) |ix| {
            const d = @as(f64, y_gpu[ix]) - sim_arr[ix];
            num_fused += d * d;
        }
        const rel_fused = @sqrt(num_fused / den);
        try stdout.print("i8 linear {d}x{d}x{d}: rel vs cpu-sim {d:.6} (wiring), rel vs f32 {d:.4} (int8 acc), fused {d:.6}\n", .{ m, rows, cols, rel_sim, rel_truth, rel_fused });
        try stdout.flush();
        if (rel_sim > 1e-3 or rel_truth > 0.03 or rel_fused > 1e-3) return error.CudaError;
    }
    _ = io;

    // --- DiT-shape timing: decompose the full linear (prep|gemm|rescale). ---
    const timer = try ctx.timerCreate();
    defer ctx.timerDestroy(timer);
    const TCase = struct { m: usize, rows: usize, cols: usize };
    const tcases = [_]TCase{
        .{ .m = 7680, .rows = 6144, .cols = 6144 }, // qkv/gate @1120x1680
        .{ .m = 7680, .rows = 16384, .cols = 6144 }, // mlp gate/up
        .{ .m = 7680, .rows = 6144, .cols = 16384 }, // mlp.down
    };
    for (tcases) |c| {
        const m = c.m;
        const rows = c.rows;
        const cols = c.cols;
        const prep_ptx = try buildPrep(gpa, cols, 8, false);
        defer gpa.free(prep_ptx);
        var pmod = try ctx.loadModule(prep_ptx);
        defer pmod.unload(ctx);
        const f_prep = try pmod.getFunction(ctx, "iprep");
        const shb = prepSharedBytes(cols);
        try ctx.setMaxDynamicShared(f_prep, shb);

        var x_d = try ctx.alloc(m * cols * 4);
        defer ctx.free(&x_d);
        var q_d = try ctx.alloc(m * cols);
        defer ctx.free(&q_d);
        var as_d = try ctx.alloc(m * 4);
        defer ctx.free(&as_d);
        var w_d = try ctx.alloc(rows * cols);
        defer ctx.free(&w_d);
        var ws_d = try ctx.alloc(rows * 4);
        defer ctx.free(&ws_d);
        var acc_d = try ctx.alloc(m * rows * 4);
        defer ctx.free(&acc_d);
        var y_d = try ctx.alloc(m * rows * 4);
        defer ctx.free(&y_d);

        var px = x_d.ptr;
        var pq = q_d.ptr;
        var pas = as_d.ptr;
        var pp = [_]?*anyopaque{ @ptrCast(&px), @ptrCast(&pq), @ptrCast(&pas) };
        var pa = q_d.ptr;
        var pb = w_d.ptr;
        var pc = acc_d.ptr;
        var pn: u32 = @intCast(rows);
        var pk: u32 = @intCast(cols);
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };
        const total: u32 = @intCast(m * rows);
        var racc = acc_d.ptr;
        var ry = y_d.ptr;
        var ras = as_d.ptr;
        var rws = ws_d.ptr;
        var rrows: u32 = @intCast(rows);
        var rtot: u32 = total;
        var pr = [_]?*anyopaque{ @ptrCast(&racc), @ptrCast(&ry), @ptrCast(&ras), @ptrCast(&rws), @ptrCast(&rrows), @ptrCast(&rtot) };

        // fused GEMM params (rescale in C-store; writes y_d directly)
        var fas = as_d.ptr;
        var fws = ws_d.ptr;
        var pfz = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&ry), @ptrCast(&pn), @ptrCast(&pk), @ptrCast(&fas), @ptrCast(&fws) };
        const gg = [3]u32{ @intCast(rows / 128), @intCast(m / 128), 1 };
        const gb = [3]u32{ 128, 1, 1 };

        var best_prep: f32 = std.math.floatMax(f32);
        var best_gemm: f32 = std.math.floatMax(f32);
        var best_resc: f32 = std.math.floatMax(f32);
        var best_full: f32 = std.math.floatMax(f32);
        var best_fused: f32 = std.math.floatMax(f32); // prep + fused gemm
        // warm
        try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
        try ctx.launch(f_gemm, gg, gb, 0, &pg);
        try ctx.launch(f_rescale, .{ (total + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, 0, &pr);
        try ctx.launch(f_fused, gg, gb, 0, &pfz);
        try ctx.synchronize();
        for (0..10) |_| {
            try ctx.timerBegin(timer);
            try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
            best_prep = @min(best_prep, try ctx.timerEndMs(timer));
            try ctx.timerBegin(timer);
            try ctx.launch(f_gemm, gg, gb, 0, &pg);
            best_gemm = @min(best_gemm, try ctx.timerEndMs(timer));
            try ctx.timerBegin(timer);
            try ctx.launch(f_rescale, .{ (total + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, 0, &pr);
            best_resc = @min(best_resc, try ctx.timerEndMs(timer));
            try ctx.timerBegin(timer);
            try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
            try ctx.launch(f_gemm, gg, gb, 0, &pg);
            try ctx.launch(f_rescale, .{ (total + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, 0, &pr);
            best_full = @min(best_full, try ctx.timerEndMs(timer));
            try ctx.timerBegin(timer);
            try ctx.launch(f_prep, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(shb), &pp);
            try ctx.launch(f_fused, gg, gb, 0, &pfz);
            best_fused = @min(best_fused, try ctx.timerEndMs(timer));
        }
        try stdout.print("i8 linear {d}x{d}x{d}: unfused {d:.3} ms (prep {d:.3}+gemm {d:.3}+rescale {d:.3}) | FUSED {d:.3} ms (prep+gemm)\n", .{ m, rows, cols, best_full, best_prep, best_gemm, best_resc, best_fused });
        try stdout.flush();
    }

    try stdout.print("cuda int8 linear OK\n", .{});
}

fn f16bits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}
fn f16val(u: u16) f32 {
    return @floatCast(@as(f16, @bitCast(u)));
}

/// Validate the f16 tensor-core GEMM (hgemm) and full single-head attention
/// (scores = Q@K^T via hgemm -> softmax_row -> P@V via hgemm with V^T) against
/// CPU references. Attention primitives on the hand-PTX backend.
pub fn attnTest(ctx: *Context, io: anytype, stdout: anytype) !void {
    _ = io;
    const hg_ptx = try buildHgemm(gpa, false, false, false);
    defer gpa.free(hg_ptx);
    var hmod = try ctx.loadModule(hg_ptx);
    defer hmod.unload(ctx);
    const f_hg = try hmod.getFunction(ctx, "hgemm");

    var smod = try ctx.loadModule(softmax_row_ptx);
    defer smod.unload(ctx);
    const f_sm = try smod.getFunction(ctx, "softmax_row");

    // f16-C batched hgemm (used for scores in the DiT attention path).
    const hgc16_ptx = try buildHgemm(gpa, true, true, false);
    defer gpa.free(hgc16_ptx);
    var hc16mod = try ctx.loadModule(hgc16_ptx);
    defer hc16mod.unload(ctx);
    const f_hgc16 = try hc16mod.getFunction(ctx, "hgemm_batched_c16");

    var smf16mod = try ctx.loadModule(softmax_row_f16_ptx);
    defer smf16mod.unload(ctx);
    const f_smf16 = try smf16mod.getFunction(ctx, "softmax_row_f16");

    var prng = std.Random.DefaultPrng.init(21);
    const rand = prng.random();

    // ---- 1. hgemm correctness: C[m][n] = A(f16)[m][k] @ B(f16)[n][k]^T ----
    const HC = struct { m: usize, n: usize, k: usize };
    const hcases = [_]HC{ .{ .m = 128, .n = 128, .k = 128 }, .{ .m = 128, .n = 256, .k = 256 }, .{ .m = 256, .n = 128, .k = 512 } };
    for (hcases) |c| {
        const m = c.m;
        const n = c.n;
        const k = c.k;
        const a16 = try gpa.alloc(u16, m * k);
        defer gpa.free(a16);
        const b16 = try gpa.alloc(u16, n * k);
        defer gpa.free(b16);
        for (a16) |*v| v.* = f16bits(rand.floatNorm(f32) * 0.5);
        for (b16) |*v| v.* = f16bits(rand.floatNorm(f32) * 0.5);
        var a_d = try ctx.alloc(m * k * 2);
        defer ctx.free(&a_d);
        var b_d = try ctx.alloc(n * k * 2);
        defer ctx.free(&b_d);
        var c_d = try ctx.alloc(m * n * 4);
        defer ctx.free(&c_d);
        try ctx.upload(a_d, std.mem.sliceAsBytes(a16));
        try ctx.upload(b_d, std.mem.sliceAsBytes(b16));
        var pa = a_d.ptr;
        var pb = b_d.ptr;
        var pc = c_d.ptr;
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(k);
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };
        try ctx.launch(f_hg, .{ @intCast(n / 128), @intCast(m / 128), 1 }, .{ 128, 1, 1 }, 0, &pg);
        const cg = try gpa.alloc(u8, m * n * 4);
        defer gpa.free(cg);
        try ctx.download(c_d, cg);
        const cf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, cg));
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += f16val(a16[i * k + kk]) * f16val(b16[j * k + kk]);
                const d = @as(f64, cf[i * n + j]) - acc;
                num += d * d;
                den += @as(f64, acc) * acc;
            }
        }
        const rel = @sqrt(num / den);
        try stdout.print("hgemm {d}x{d}x{d}: rel vs f16-cpu {d:.5}\n", .{ m, n, k, rel });
        if (rel > 2e-2) return error.CudaError;

        // f16-C batched variant, gs=2 (two identical batches, exercises the
        // grid.z batch C offset with shl-1). Compare BOTH batches to cf.
        var a2_d = try ctx.alloc(2 * m * k * 2);
        defer ctx.free(&a2_d);
        var b2_d = try ctx.alloc(2 * n * k * 2);
        defer ctx.free(&b2_d);
        var c16_d = try ctx.alloc(2 * m * n * 2);
        defer ctx.free(&c16_d);
        try ctx.upload(.{ .ptr = a2_d.ptr, .bytes = m * k * 2 }, std.mem.sliceAsBytes(a16));
        try ctx.upload(.{ .ptr = a2_d.ptr + m * k * 2, .bytes = m * k * 2 }, std.mem.sliceAsBytes(a16));
        try ctx.upload(.{ .ptr = b2_d.ptr, .bytes = n * k * 2 }, std.mem.sliceAsBytes(b16));
        try ctx.upload(.{ .ptr = b2_d.ptr + n * k * 2, .bytes = n * k * 2 }, std.mem.sliceAsBytes(b16));
        var qa = a2_d.ptr;
        var qb = b2_d.ptr;
        var qc = c16_d.ptr;
        var qn: u32 = @intCast(n);
        var qk: u32 = @intCast(k);
        var sa: u32 = @intCast(m * k);
        var sb: u32 = @intCast(n * k);
        var sc: u32 = @intCast(m * n);
        var sc1: f32 = 1.0;
        var qg = [_]?*anyopaque{ @ptrCast(&qa), @ptrCast(&qb), @ptrCast(&qc), @ptrCast(&qn), @ptrCast(&qk), @ptrCast(&sa), @ptrCast(&sb), @ptrCast(&sc), @ptrCast(&sc1) };
        try ctx.launch(f_hgc16, .{ @intCast(n / 128), @intCast(m / 128), 2 }, .{ 128, 1, 1 }, 0, &qg);
        const c16g = try gpa.alloc(u8, 2 * m * n * 2);
        defer gpa.free(c16g);
        try ctx.download(c16_d, c16g);
        const c16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, c16g));
        for ([_]usize{ 0, 1 }) |z| {
            var n2: f64 = 0;
            var d2: f64 = 0;
            for (0..m * n) |i| {
                const dd = @as(f64, f16val(c16[z * m * n + i])) - cf[i];
                n2 += dd * dd;
                d2 += @as(f64, cf[i]) * cf[i];
            }
            const rel16 = @sqrt(n2 / d2);
            try stdout.print("hgemm_c16 {d}x{d}x{d} batch{d}: rel vs f32-C {d:.5}\n", .{ m, n, k, z, rel16 });
            if (rel16 > 3e-3) return error.CudaError; // f16 rounding of C only
        }
    }

    // ---- 2. single-head attention: seq x hd ----
    const seq = 256;
    const hd = 128;
    const scale: f32 = 1.0 / @sqrt(@as(f32, hd));
    const qf = try gpa.alloc(f32, seq * hd);
    defer gpa.free(qf);
    const kf = try gpa.alloc(f32, seq * hd);
    defer gpa.free(kf);
    const vf = try gpa.alloc(f32, seq * hd);
    defer gpa.free(vf);
    for (qf) |*v| v.* = rand.floatNorm(f32) * 0.3;
    for (kf) |*v| v.* = rand.floatNorm(f32) * 0.3;
    for (vf) |*v| v.* = rand.floatNorm(f32) * 0.3;
    // f16 device copies; Vt = V^T [hd][seq]
    const q16 = try gpa.alloc(u16, seq * hd);
    defer gpa.free(q16);
    const k16 = try gpa.alloc(u16, seq * hd);
    defer gpa.free(k16);
    const vt16 = try gpa.alloc(u16, hd * seq);
    defer gpa.free(vt16);
    for (0..seq * hd) |i| {
        q16[i] = f16bits(qf[i]);
        k16[i] = f16bits(kf[i]);
    }
    for (0..seq) |j| for (0..hd) |c| {
        vt16[c * seq + j] = f16bits(vf[j * hd + c]);
    };

    var q_d = try ctx.alloc(seq * hd * 2);
    defer ctx.free(&q_d);
    var k_d = try ctx.alloc(seq * hd * 2);
    defer ctx.free(&k_d);
    var vt_d = try ctx.alloc(hd * seq * 2);
    defer ctx.free(&vt_d);
    var s_d = try ctx.alloc(seq * seq * 4); // scores f32
    defer ctx.free(&s_d);
    var p_d = try ctx.alloc(seq * seq * 2); // P f16
    defer ctx.free(&p_d);
    var o_d = try ctx.alloc(seq * hd * 4); // out f32
    defer ctx.free(&o_d);
    try ctx.upload(q_d, std.mem.sliceAsBytes(q16));
    try ctx.upload(k_d, std.mem.sliceAsBytes(k16));
    try ctx.upload(vt_d, std.mem.sliceAsBytes(vt16));

    // scores = Q @ K^T  (m=seq, n=seq, k=hd)
    var sa = q_d.ptr;
    var sb = k_d.ptr;
    var sc = s_d.ptr;
    var sn: u32 = @intCast(seq);
    var sk: u32 = @intCast(hd);
    var sp = [_]?*anyopaque{ @ptrCast(&sa), @ptrCast(&sb), @ptrCast(&sc), @ptrCast(&sn), @ptrCast(&sk) };
    try ctx.launch(f_hg, .{ @intCast(seq / 128), @intCast(seq / 128), 1 }, .{ 128, 1, 1 }, 0, &sp);
    // softmax(scale*S) -> P f16
    var ms = s_d.ptr;
    var mp = p_d.ptr;
    var mn: u32 = @intCast(seq);
    var mseq: u32 = @intCast(seq);
    var msc: f32 = scale;
    var mpar = [_]?*anyopaque{ @ptrCast(&ms), @ptrCast(&mp), @ptrCast(&mn), @ptrCast(&mseq), @ptrCast(&msc) };
    try ctx.launch(f_sm, .{ @intCast(seq), 1, 1 }, .{ 256, 1, 1 }, @intCast(seq * 4), &mpar);
    // out = P @ Vt^T  (m=seq, n=hd, k=seq)
    var oa = p_d.ptr;
    var ob = vt_d.ptr;
    var oc = o_d.ptr;
    var on: u32 = @intCast(hd);
    var ok: u32 = @intCast(seq);
    var op = [_]?*anyopaque{ @ptrCast(&oa), @ptrCast(&ob), @ptrCast(&oc), @ptrCast(&on), @ptrCast(&ok) };
    try ctx.launch(f_hg, .{ @intCast(hd / 128), @intCast(seq / 128), 1 }, .{ 128, 1, 1 }, 0, &op);

    const og = try gpa.alloc(u8, seq * hd * 4);
    defer gpa.free(og);
    try ctx.download(o_d, og);
    const o_gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, og));

    // CPU f32 attention reference
    const prow = try gpa.alloc(f32, seq);
    defer gpa.free(prow);
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..seq) |q| {
        var mx: f32 = -std.math.inf(f32);
        for (0..seq) |j| {
            var dot: f32 = 0;
            for (0..hd) |c| dot += qf[q * hd + c] * kf[j * hd + c];
            prow[j] = dot * scale;
            mx = @max(mx, prow[j]);
        }
        var sum: f32 = 0;
        for (0..seq) |j| {
            prow[j] = @exp(prow[j] - mx);
            sum += prow[j];
        }
        for (0..seq) |j| prow[j] /= sum;
        for (0..hd) |c| {
            var acc: f32 = 0;
            for (0..seq) |j| acc += prow[j] * vf[j * hd + c];
            const d = @as(f64, o_gpu[q * hd + c]) - acc;
            num += d * d;
            den += @as(f64, acc) * acc;
        }
    }
    const rel = @sqrt(num / den);
    try stdout.print("attention seq={d} hd={d}: rel vs f32-cpu {d:.5}\n", .{ seq, hd, rel });
    if (rel > 5e-2) return error.CudaError;

    // ---- 3. f16-scores path (hgemm_batched_c16 -> softmax_row_f16 -> P@V) ----
    var s16_d = try ctx.alloc(seq * seq * 2); // scores f16
    defer ctx.free(&s16_d);
    var o2_d = try ctx.alloc(seq * hd * 4);
    defer ctx.free(&o2_d);
    // scores f16: gs=1, strides seq*hd / seq*hd / seq*seq
    var za = q_d.ptr;
    var zb = k_d.ptr;
    var zc = s16_d.ptr;
    var zn: u32 = @intCast(seq);
    var zk: u32 = @intCast(hd);
    var zsa: u32 = @intCast(seq * hd);
    var zsb: u32 = @intCast(seq * hd);
    var zsc: u32 = @intCast(seq * seq);
    var zscale: f32 = scale; // prefold the softmax scale into the f16 scores
    var zp = [_]?*anyopaque{ @ptrCast(&za), @ptrCast(&zb), @ptrCast(&zc), @ptrCast(&zn), @ptrCast(&zk), @ptrCast(&zsa), @ptrCast(&zsb), @ptrCast(&zsc), @ptrCast(&zscale) };
    try ctx.launch(f_hgc16, .{ @intCast(seq / 128), @intCast(seq / 128), 1 }, .{ 128, 1, 1 }, 0, &zp);
    // Zero P first so a broken softmax_f16 can't be masked by stale (correct) P.
    try ctx.memsetD8(.{ .ptr = p_d.ptr, .bytes = seq * seq * 2 }, 0, seq * seq * 2);
    // softmax_f16(S) -> P f16 (scale already folded into S, so scale=1)
    var ws = s16_d.ptr;
    var wp = p_d.ptr;
    var wn: u32 = @intCast(seq);
    var wseq: u32 = @intCast(seq);
    var wsc: f32 = 1.0;
    var wpar = [_]?*anyopaque{ @ptrCast(&ws), @ptrCast(&wp), @ptrCast(&wn), @ptrCast(&wseq), @ptrCast(&wsc) };
    try ctx.launch(f_smf16, .{ @intCast(seq), 1, 1 }, .{ 256, 1, 1 }, 256 * 4, &wpar);
    // P @ Vt^T
    var ya = p_d.ptr;
    var yb = vt_d.ptr;
    var yc = o2_d.ptr;
    var yn: u32 = @intCast(hd);
    var yk: u32 = @intCast(seq);
    var yp = [_]?*anyopaque{ @ptrCast(&ya), @ptrCast(&yb), @ptrCast(&yc), @ptrCast(&yn), @ptrCast(&yk) };
    try ctx.launch(f_hg, .{ @intCast(hd / 128), @intCast(seq / 128), 1 }, .{ 128, 1, 1 }, 0, &yp);
    const o2g = try gpa.alloc(u8, seq * hd * 4);
    defer gpa.free(o2g);
    try ctx.download(o2_d, o2g);
    const o2: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, o2g));
    var n3: f64 = 0;
    var d3: f64 = 0;
    for (0..seq) |q| {
        var mx: f32 = -std.math.inf(f32);
        for (0..seq) |j| {
            var dot: f32 = 0;
            for (0..hd) |c| dot += qf[q * hd + c] * kf[j * hd + c];
            prow[j] = dot * scale;
            mx = @max(mx, prow[j]);
        }
        var sum: f32 = 0;
        for (0..seq) |j| {
            prow[j] = @exp(prow[j] - mx);
            sum += prow[j];
        }
        for (0..seq) |j| prow[j] /= sum;
        for (0..hd) |c| {
            var acc: f32 = 0;
            for (0..seq) |j| acc += prow[j] * vf[j * hd + c];
            const d = @as(f64, o2[q * hd + c]) - acc;
            n3 += d * d;
            d3 += @as(f64, acc) * acc;
        }
    }
    const rel3 = @sqrt(n3 / d3);
    try stdout.print("attention (f16 scores) seq={d} hd={d}: rel vs f32-cpu {d:.5}\n", .{ seq, hd, rel3 });
    if (rel3 > 5e-2) return error.CudaError;
    try stdout.print("cuda attention OK\n", .{});
}

/// Naive f32 register GEMM: y[m][rows] = scale*(x[m][cols] @ W(f32)[rows][cols]^T)
/// (+bias[rows]); y_off/x_off are ELEMENT offsets to the first row. One thread
/// per output element. For the non-int8 first/last DiT linears (small). Entry
/// `f32gemm`. grid ceil(m*rows/256), block 256.
pub const f32gemm_ptx: [:0]const u8 =
    \\.version 8.0
    \\.target sm_86
    \\.address_size 64
    \\.visible .entry f32gemm(
    \\    .param .u64 p_y, .param .u64 p_x, .param .u64 p_w, .param .u64 p_bias,
    \\    .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_total,
    \\    .param .f32 p_scale, .param .u32 p_yoff, .param .u32 p_xoff, .param .u32 p_hasbias
    \\)
    \\{
    \\    .reg .pred %p<3>;
    \\    .reg .b32 %r<20>;
    \\    .reg .f32 %f<6>;
    \\    .reg .b64 %rd<16>;
    \\    mov.u32 %r1, %ctaid.x;
    \\    mov.u32 %r2, %ntid.x;
    \\    mov.u32 %r3, %tid.x;
    \\    mad.lo.s32 %r4, %r1, %r2, %r3;
    \\    ld.param.u32 %r5, [p_total];
    \\    setp.ge.u32 %p1, %r4, %r5;
    \\    @%p1 bra DONE;
    \\    ld.param.u32 %r6, [p_rows];
    \\    ld.param.u32 %r7, [p_cols];
    \\    ld.param.f32 %f1, [p_scale];
    \\    ld.param.u32 %r8, [p_yoff];
    \\    ld.param.u32 %r9, [p_xoff];
    \\    ld.param.u32 %r10, [p_hasbias];
    \\    ld.param.u64 %rd1, [p_y];
    \\    ld.param.u64 %rd2, [p_x];
    \\    ld.param.u64 %rd3, [p_w];
    \\    ld.param.u64 %rd4, [p_bias];
    \\    cvta.to.global.u64 %rd1, %rd1;
    \\    cvta.to.global.u64 %rd2, %rd2;
    \\    cvta.to.global.u64 %rd3, %rd3;
    \\    cvta.to.global.u64 %rd4, %rd4;
    \\    div.u32 %r11, %r4, %r6;
    \\    mul.lo.s32 %r12, %r11, %r6;
    \\    sub.s32 %r13, %r4, %r12;
    \\    mad.lo.s32 %r14, %r11, %r7, %r9;
    \\    mul.lo.s32 %r15, %r13, %r7;
    \\    mul.wide.u32 %rd5, %r14, 4;
    \\    add.s64 %rd6, %rd2, %rd5;
    \\    mul.wide.u32 %rd7, %r15, 4;
    \\    add.s64 %rd8, %rd3, %rd7;
    \\    mov.f32 %f2, 0f00000000;
    \\    mov.u32 %r16, 0;
    \\LOOP:
    \\    setp.ge.u32 %p2, %r16, %r7;
    \\    @%p2 bra ENDL;
    \\    ld.global.f32 %f3, [%rd6];
    \\    ld.global.f32 %f4, [%rd8];
    \\    fma.rn.f32 %f2, %f3, %f4, %f2;
    \\    add.s64 %rd6, %rd6, 4;
    \\    add.s64 %rd8, %rd8, 4;
    \\    add.u32 %r16, %r16, 1;
    \\    bra LOOP;
    \\ENDL:
    \\    mul.f32 %f2, %f2, %f1;
    \\    setp.eq.u32 %p2, %r10, 0;
    \\    @%p2 bra NOBIAS;
    \\    mul.wide.u32 %rd9, %r13, 4;
    \\    add.s64 %rd10, %rd4, %rd9;
    \\    ld.global.f32 %f5, [%rd10];
    \\    add.f32 %f2, %f2, %f5;
    \\NOBIAS:
    \\    mad.lo.s32 %r17, %r11, %r6, %r13;
    \\    add.s32 %r18, %r17, %r8;
    \\    mul.wide.u32 %rd11, %r18, 4;
    \\    add.s64 %rd12, %rd1, %rd11;
    \\    st.global.f32 [%rd12], %f2;
    \\DONE:
    \\    ret;
    \\}
;
