//! Hand-written / hand-emitted PTX kernels for the CUDA backend, plus the
//! bring-up smoke test. GEMM/prep/attention kernels are added here as Phase 1
//! progresses; each is authored as PTX (validated offline with
//! `ptxas -arch=sm_86`) and JIT-compiled by the driver at load time.

const std = @import("std");
const cu = @import("cu.zig");
const ctxmod = @import("context.zig");
const elt = @import("elt.zig"); // KvFmt only (elt imports nothing local, so no cycle)
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
pub fn buildIgemmPipe(alloc: std.mem.Allocator, kstep: usize, fuse: bool, bits: usize, use_ldmatrix: bool) ![:0]u8 {
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
    // XOR-swizzle mask for the ldmatrix path (bits [4:6], keyed on row&7). Zero
    // when swizzling is off. Applied identically to the cp.async store offset and
    // the ldmatrix load offset -> pure permutation of shared, so bit-exact, but
    // it makes the 8 rows of each 8x8 tile hit 8 distinct 16B banks (conflict-
    // free) where the plain srow*kstep layout collides every 2 rows.
    const r_lmask = try b.reg(.b32);
    try b.linef("mov.u32 {s}, 0;", .{r_lmask});
    if (!use_ldmatrix) {
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
    } else {
        // ldmatrix.x4 lane bases: each lane supplies the 16B-aligned start of one
        // row of one 8x8 b16 tile. For the m16n8kK A fragment the 32 lanes map to
        // 4 tiles: matrix = lane>>3, row_in_tile = lane&7.
        //   A: row_within = (lane&7) + (matrix&1)*8 ; col_byte = (matrix&2)*8
        //   B (col-major [n][k], ldmatrix.x2): row = lane&7 ; col_byte = (lane&8)<<1
        const r_l8 = try b.reg(.b32);
        const r_mtx = try b.reg(.b32);
        const r_colb = try b.reg(.b32);
        try b.linef("and.b32 {s}, {s}, 7;", .{ r_l8, r_lane });
        try b.linef("shr.u32 {s}, {s}, 3;", .{ r_mtx, r_lane });
        // A: row_within into r_tmp
        try b.linef("and.b32 {s}, {s}, 1;", .{ r_asl0, r_mtx });
        try b.linef("shl.b32 {s}, {s}, 3;", .{ r_asl0, r_asl0 });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl0, r_asl0, r_l8 }); // row_within
        try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wm }); // wm*64
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl0, r_asl0, r_tmp });
        try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl0, r_asl0, kstep });
        try b.linef("and.b32 {s}, {s}, 2;", .{ r_colb, r_mtx });
        try b.linef("shl.b32 {s}, {s}, 3;", .{ r_colb, r_colb }); // (matrix&2)*8 -> 0 or 16
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl0, r_asl0, r_colb });
        // B: row = lane&7 within the nj*8 block; col_byte = (lane&8)<<1
        try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wn }); // wn*64
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_l8 });
        try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl0, r_tmp, kstep });
        try b.linef("and.b32 {s}, {s}, 8;", .{ r_colb, r_lane });
        try b.linef("shl.b32 {s}, {s}, 1;", .{ r_colb, r_colb }); // 0 or 16
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl0, r_bsl0, r_colb });
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl0, r_bsl0, TILE });
        // Read-side swizzle mask = (row&7)<<4. For every A/B ldmatrix load the
        // fragment offset has (off>>6)&7 == lane&7 (proved: mi adds *1024, ks/col
        // add <64, none carry into bit 6), so the mask is lane-constant.
        try b.linef("shl.b32 {s}, {s}, 4;", .{ r_lmask, r_l8 });
        // Write-side swizzle: XOR (srow&7)<<4 into the cp.async store offset. sw
        // distributes over the +stage_stride (2048=bit11) chunk advance, so one
        // XOR here swizzles all four staged chunks.
        try b.linef("and.b32 {s}, {s}, 7;", .{ r_tmp, r_srow });
        try b.linef("shl.b32 {s}, {s}, 4;", .{ r_tmp, r_tmp });
        try b.linef("xor.b32 {s}, {s}, {s};", .{ r_stdst, r_stdst, r_tmp });
    }

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
    const r_lda = try b.reg(.b32); // swizzled ldmatrix address temp
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
            if (use_ldmatrix) {
                // One warp-cooperative load pulls the whole 16x32 (s8) / 16x64 (s4)
                // A fragment into the exact {a0,a1,a2,a3} MMA register layout.
                // XOR-swizzle the address (r_lmask=0 when swizzle off).
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_lda, r_asl, o });
                try b.linef("xor.b32 {s}, {s}, {s};", .{ r_lda, r_lda, r_lmask });
                try b.linef("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{s}, {s}, {s}, {s}}}, [{s}];", .{
                    af[mi * 4 + 0], af[mi * 4 + 1], af[mi * 4 + 2], af[mi * 4 + 3], r_lda,
                });
            } else {
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 8 * kstep });
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 8 * kstep + 16 });
            }
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = nj * 8 * kstep + ks * 32;
            if (use_ldmatrix) {
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_lda, r_bsl, o });
                try b.linef("xor.b32 {s}, {s}, {s};", .{ r_lda, r_lda, r_lmask });
                try b.linef("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{s}, {s}}}, [{s}];", .{
                    bf[nj * 2 + 0], bf[nj * 2 + 1], r_lda,
                });
            } else {
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
                try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
            }
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
pub fn buildHgemm(alloc: std.mem.Allocator, batched: bool, c_f16: bool, attnout: bool, bf16: bool, use_ldmatrix: bool) ![:0]u8 {
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
    // XOR-swizzle mask for the ldmatrix path (bits [4:6], keyed on row&7); 0 when
    // off. Same permutation as buildIgemmPipe (rows are 64B apart here too:
    // KSTEP=32 f16 * 2B). Applied to the st.shared store offset and the ldmatrix
    // read offset -> pure permutation of shared, bit-exact, conflict-free reads.
    const r_lmask = try b.reg(.b32);
    try b.linef("mov.u32 {s}, 0;", .{r_lmask});
    if (!use_ldmatrix) {
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
    } else {
        // ldmatrix.x4 (A) / .x2 (B) lane bases. matrix = lane>>3, row = lane&7.
        //   A: row_within = (lane&7) + (matrix&1)*8 ; col_byte = (matrix&2)*8
        //   B (col-major [n][k]): row = lane&7 ; col_byte = (lane&8)<<1
        const r_l8 = try b.reg(.b32);
        const r_mtx = try b.reg(.b32);
        const r_colb = try b.reg(.b32);
        try b.linef("and.b32 {s}, {s}, 7;", .{ r_l8, r_lane });
        try b.linef("shr.u32 {s}, {s}, 3;", .{ r_mtx, r_lane });
        // A row_within
        try b.linef("and.b32 {s}, {s}, 1;", .{ r_asl, r_mtx });
        try b.linef("shl.b32 {s}, {s}, 3;", .{ r_asl, r_asl });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_l8 });
        try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wm }); // wm*64 rows
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_tmp });
        try b.linef("mul.lo.u32 {s}, {s}, 64;", .{ r_asl, r_asl }); // *64 bytes/row
        try b.linef("and.b32 {s}, {s}, 2;", .{ r_colb, r_mtx });
        try b.linef("shl.b32 {s}, {s}, 3;", .{ r_colb, r_colb }); // 0 or 16
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_colb });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
        // B
        try b.linef("shl.b32 {s}, {s}, 6;", .{ r_tmp, r_wn }); // wn*64 rows
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_l8 });
        try b.linef("mul.lo.u32 {s}, {s}, 64;", .{ r_bsl, r_tmp });
        try b.linef("and.b32 {s}, {s}, 8;", .{ r_colb, r_lane });
        try b.linef("shl.b32 {s}, {s}, 1;", .{ r_colb, r_colb }); // 0 or 16
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_colb });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, BS_BASE });
        // read mask = (lane&7)<<4
        try b.linef("shl.b32 {s}, {s}, 4;", .{ r_lmask, r_l8 });
        // write swizzle: XOR (rowq&7)<<4 into both shared store bases (rowq=t>>4=
        // row&7 of the staged row; constant across the i*512 chunk advance).
        try b.linef("shl.b32 {s}, {s}, 4;", .{ r_colb, r_rowq }); // rowq<<4 (rowq is 0..7)
        try b.linef("xor.b32 {s}, {s}, {s};", .{ r_shA, r_shA, r_colb });
        try b.linef("xor.b32 {s}, {s}, {s};", .{ r_shB, r_shB, r_colb });
    }

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});

    const r_k0 = try b.reg(.b32);
    const rd_ap = try b.reg(.b64);
    const rd_bp = try b.reg(.b64);
    const rd_k0 = try b.reg(.b64);
    const r_tA = try b.reg(.b32);
    const r_tB = try b.reg(.b32);
    const r_lda = try b.reg(.b32); // swizzled ldmatrix address temp
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
            if (use_ldmatrix) {
                // f16/bf16 are natively b16 -> ldmatrix.x4.b16 lands the exact
                // {a0,a1,a2,a3} m16n8k16 fragment. XOR-swizzle the address.
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_lda, r_asl, o });
                try b.linef("xor.b32 {s}, {s}, {s};", .{ r_lda, r_lda, r_lmask });
                try b.linef("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{s}, {s}, {s}, {s}}}, [{s}];", .{
                    af[mi * 4 + 0], af[mi * 4 + 1], af[mi * 4 + 2], af[mi * 4 + 3], r_lda,
                });
            } else {
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 512 });
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 528 });
            }
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = nj * 512 + ks * 32;
            if (use_ldmatrix) {
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_lda, r_bsl, o });
                try b.linef("xor.b32 {s}, {s}, {s};", .{ r_lda, r_lda, r_lmask });
                try b.linef("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{s}, {s}}}, [{s}];", .{
                    bf[nj * 2 + 0], bf[nj * 2 + 1], r_lda,
                });
            } else {
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
                try b.linef("ld.shared.b32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
            }
        }
        mi = 0;
        while (mi < MT) : (mi += 1) {
            nj = 0;
            while (nj < NT) : (nj += 1) {
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                try b.linef("mma.sync.aligned.m16n8k16.row.col.f32.{s}.{s}.f32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                    if (bf16) "bf16" else "f16", if (bf16) "bf16" else "f16",
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
const Kernel = enum { v0, smem, pipe, pipe_lm };

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
    const pipe_ptx = try buildIgemmPipe(gpa, 64, false, 8, false);
    defer gpa.free(pipe_ptx);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "/tmp/claude-1000/-dump-projects-zig-TensorPencil/eccfce6f-7c1f-4c32-b182-cc9c60d44a58/scratchpad/igemm_pipe.gen.ptx", .data = pipe_ptx }) catch {};
    var mod2 = try ctx.loadModule(pipe_ptx);
    defer mod2.unload(ctx);
    const f_pipe = try mod2.getFunction(ctx, "igemm_pipe");
    // ldmatrix fragment-load variant (warp-cooperative frag loads; must stay
    // bit-exact vs the plain ld.shared path — same math, different load path).
    const pipe_lm_ptx = try buildIgemmPipe(gpa, 64, false, 8, true);
    defer gpa.free(pipe_lm_ptx);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "/tmp/claude-1000/-dump-projects-zig-TensorPencil/eccfce6f-7c1f-4c32-b182-cc9c60d44a58/scratchpad/igemm_pipe_lm.gen.ptx", .data = pipe_lm_ptx }) catch {};
    var mod_lm = try ctx.loadModule(pipe_lm_ptx);
    defer mod_lm.unload(ctx);
    const f_pipe_lm = try mod_lm.getFunction(ctx, "igemm_pipe");
    {
        var rs: c_int = 0;
        var rp: c_int = 0;
        var rl: c_int = 0;
        _ = ctx.api.cuFuncGetAttribute(&rs, cu.CU_FUNC_ATTRIBUTE_NUM_REGS, f_smem);
        _ = ctx.api.cuFuncGetAttribute(&rp, cu.CU_FUNC_ATTRIBUTE_NUM_REGS, f_pipe);
        _ = ctx.api.cuFuncGetAttribute(&rl, cu.CU_FUNC_ATTRIBUTE_NUM_REGS, f_pipe_lm);
        try stdout.print("regs/thread: smem={d} pipe={d} pipe_lm={d}\n", .{ rs, rp, rl });
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
        .{ .m = 128, .n = 128, .k = 64, .check = true, .kern = .pipe_lm },
        .{ .m = 256, .n = 384, .k = 320, .check = true, .kern = .pipe_lm },
        .{ .m = 128, .n = 256, .k = 6144, .check = true, .kern = .pipe_lm },
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false, .kern = .smem },
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false, .kern = .pipe },
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false, .kern = .pipe_lm },
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false, .kern = .smem }, // DiT qkv @1120x1680
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false, .kern = .pipe },
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false, .kern = .pipe_lm },
        .{ .m = 7680, .n = 16384, .k = 6144, .check = false, .kern = .pipe }, // mlp gate/up
        .{ .m = 7680, .n = 16384, .k = 6144, .check = false, .kern = .pipe_lm },
        .{ .m = 7680, .n = 6144, .k = 16384, .check = false, .kern = .pipe }, // mlp.down
        .{ .m = 7680, .n = 6144, .k = 16384, .check = false, .kern = .pipe_lm },
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
            .pipe_lm => f_pipe_lm,
        };
        const grid: [3]u32 = switch (c.kern) {
            .v0 => .{ @intCast(n / 8), @intCast(m / 16), 1 },
            .smem, .pipe, .pipe_lm => .{ @intCast(n / 128), @intCast(m / 128), 1 },
        };
        const block: [3]u32 = switch (c.kern) {
            .v0 => .{ 32, 1, 1 },
            .smem, .pipe, .pipe_lm => .{ 128, 1, 1 },
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
    const convrot = @import("tp_ops").convrot;

    // Performant fused s4 GEMM (rescale folded into the C-store), the path
    // dit_cuda's opI4Gemm uses. Requires m%128, rows%128, cols%128.
    const pipe_ptx = try buildIgemmPipe(gpa, 64, true, 4, true);
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
    const convrot = @import("tp_ops").convrot;

    const pipe_ptx = try buildIgemmPipe(gpa, 64, false, 8, true);
    defer gpa.free(pipe_ptx);
    var gmod = try ctx.loadModule(pipe_ptx);
    defer gmod.unload(ctx);
    const f_gemm = try gmod.getFunction(ctx, "igemm_pipe");

    var rmod = try ctx.loadModule(irescale_ptx);
    defer rmod.unload(ctx);
    const f_rescale = try rmod.getFunction(ctx, "irescale");

    // Stage-A fused GEMM: rescale folded into the C-store (no s32 acc buffer,
    // no separate rescale pass), output f32 y directly.
    const fused_ptx = try buildIgemmPipe(gpa, 64, true, 8, true);
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
    const hg_ptx = try buildHgemm(gpa, false, false, false, false, true);
    defer gpa.free(hg_ptx);
    var hmod = try ctx.loadModule(hg_ptx);
    defer hmod.unload(ctx);
    const f_hg = try hmod.getFunction(ctx, "hgemm");

    var smod = try ctx.loadModule(softmax_row_ptx);
    defer smod.unload(ctx);
    const f_sm = try smod.getFunction(ctx, "softmax_row");

    // f16-C batched hgemm (used for scores in the DiT attention path).
    const hgc16_ptx = try buildHgemm(gpa, true, true, false, false, true);
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



// ---------------------------------------------------------------------------
// MMQ — q4_k weight x q8_1 activation on the s8 tensor cores, with the
// dequantized weight NEVER materialized in global memory.
//
// The m>1 fallback (`opMatmulQuant`) expands the whole weight matrix to f16 in
// global memory and then runs an f16 GEMM, so a prefill that steps the model in
// chunks re-expands every weight once per chunk. Measured on gemma4 31B Q4_K_M:
// that expansion alone is ~23% of prefill wall time, and the GEMM it feeds then
// re-reads the result at 16 bits/elem instead of q4_k's 4.5.
//
// MMQ feeds `mma.m16n8k32.s8` straight from the packed nibbles. A k-quant's
// scale changes every 32 elements, so the integer accumulator can only run for
// ONE 32-elem sub-block before it must be folded into f32: the loop is one mma
// per sub-block followed by a scaled accumulate (the same shape as llama.cpp's
// mma mmq).
//
//   v = d*sc*q - dmin*m            (q4_k; sc/m per 32-elem sub-block)
//   Σ_k v_k·a_k = d*sc*da*Σ(q·qa) - dmin*m*da*Σ(qa)
//
// The first term is the mma's s32 output; the second needs Σ(qa) per activation
// block, which `quantize_q8_1` stores in its trailing `s` region.
//
// Tiling: one warp owns 16 rows x nt*8 tokens and holds nt*4 f32 accumulators.
// The weight fragments are loaded once per k-step and reused across all nt
// column tiles, so the weight is read once per row-tile per nt*8 tokens. nt*8
// must be >= 128 to beat the f16 path's traffic: at nt=16 a 256-token chunk
// reads the weight twice (~35 GB for this model) versus the f16 path's expand
// + re-read (~122 GB).
//
// Only the ACTIVATION scales go through shared memory. Reading them straight
// from global would cost 4 scalar loads per mma (they are indexed by both the
// column and the sub-block, so nothing hoists); staged per super-block by the
// whole block they cost 8 vector-ish loads per 8 mmas instead.
//
// Requires cols % 256 == 0 and rows % (16*warps) == 0. `n` is free — the column
// tiles are predicated. Entry `mmq_q4_k`.
// ---------------------------------------------------------------------------

/// Emit the q4_k sub-block scale/min decode for a COMPILE-TIME sub-block index
/// `is` (0..7), mirroring ggml's `get_scale_min_k4`. At comptime the reference
/// implementation's data-dependent branch and variable shifts fold into
/// immediates. `s1/s2/s3` hold header bytes 4..7, 8..11, 12..15; `t` is scratch.
fn q4kScaleMin(
    b: *ptx.Builder,
    is: usize,
    s1: []const u8,
    s2: []const u8,
    s3: []const u8,
    sc: []const u8,
    mn: []const u8,
    t: []const u8,
) !void {
    if (is < 4) {
        const sh = is * 8;
        try b.linef("shr.u32 {s}, {s}, {d};", .{ sc, s1, sh });
        try b.linef("and.b32 {s}, {s}, 63;", .{ sc, sc });
        try b.linef("shr.u32 {s}, {s}, {d};", .{ mn, s2, sh });
        try b.linef("and.b32 {s}, {s}, 63;", .{ mn, mn });
    } else {
        // sc = (scales[is+4] & 0xF) | ((scales[is-4] >> 6) << 4)
        // mn = (scales[is+4] >>  4) | ((scales[is]   >> 6) << 4)
        const sh = (is - 4) * 8;
        try b.linef("shr.u32 {s}, {s}, {d};", .{ t, s3, sh });
        try b.linef("and.b32 {s}, {s}, 15;", .{ sc, t });
        try b.linef("shr.u32 {s}, {s}, 4;", .{ mn, t });
        try b.linef("and.b32 {s}, {s}, 15;", .{ mn, mn });
        try b.linef("shr.u32 {s}, {s}, {d};", .{ t, s1, sh });
        try b.linef("shr.u32 {s}, {s}, 6;", .{ t, t });
        try b.linef("and.b32 {s}, {s}, 3;", .{ t, t });
        try b.linef("shl.b32 {s}, {s}, 4;", .{ t, t });
        try b.linef("or.b32 {s}, {s}, {s};", .{ sc, sc, t });
        try b.linef("shr.u32 {s}, {s}, {d};", .{ t, s2, sh });
        try b.linef("shr.u32 {s}, {s}, 6;", .{ t, t });
        try b.linef("and.b32 {s}, {s}, 3;", .{ t, t });
        try b.linef("shl.b32 {s}, {s}, 4;", .{ t, t });
        try b.linef("or.b32 {s}, {s}, {s};", .{ mn, mn, t });
    }
}

/// q4_k MMQ (see the block comment above). `nt` = 8-token column tiles per warp,
/// `warps` = warps per block (each owning its own 16 rows, all sharing columns).
/// Grid: x = ceil(n/(nt*8)), y = rows/(16*warps), block = 32*warps threads.
///   p_w      q4_k weight, row-major, row stride (cols/256)*144
///   p_x      quantize_q8_1 output: f32 d[n*cols/32] | i8 qs[n*cols] | f32 s[n*cols/32]
///   p_y      f32 [n][rows] (y[t*rows + r]) — the layout both existing paths write
///   p_scale  multiplies the result
pub fn buildMmqQ4K(alloc: std.mem.Allocator, nt: usize, warps: usize) ![:0]u8 {
    std.debug.assert(nt >= 1 and warps >= 1);
    const nthreads = 32 * warps;
    const sd_bytes = nt * 8 * 8 * 4; // [nt*8 cols][8 sub-blocks] f32
    const sh_bytes = 2 * sd_bytes; // sd then ss

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, nt * 4);
    const cfr = try b.regs(.b32, 4);
    const afr = try b.regs(.b32, 4);
    const bfr = try b.regs(.b32, 2);
    const wq = try b.regs(.b32, 4); // wa0, wa16, wb0, wb16

    const rd_w = try b.reg(.b64);
    const rd_x = try b.reg(.b64);
    const rd_y = try b.reg(.b64);
    const r_rows = try b.reg(.b32);
    const r_cols = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const f_scale = try b.reg(.f32);
    try b.linef("ld.param.u64 {s}, [p_w];", .{rd_w});
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_y];", .{rd_y});
    try b.linef("ld.param.u32 {s}, [p_rows];", .{r_rows});
    try b.linef("ld.param.u32 {s}, [p_cols];", .{r_cols});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_w, rd_w });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_y, rd_y });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_t4 = try b.reg(.b32);
    const r_rowa = try b.reg(.b32);
    const r_rowb = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    const r_nsb = try b.reg(.b32);
    const r_rb = try b.reg(.b32);
    const r_bpt = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const tm = try b.regs(.b32, 6);
    const rdt = try b.regs(.b64, 4);

    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_t4, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{tm[0]});
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ tm[0], tm[0], 16 * warps });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ tm[1], r_warp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_rowa, tm[0], r_gid });
    try b.linef("add.u32 {s}, {s}, 8;", .{ r_rowb, r_rowa });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_col0, r_col0, nt * 8 });
    try b.linef("shr.u32 {s}, {s}, 8;", .{ r_nsb, r_cols });
    try b.linef("mul.lo.u32 {s}, {s}, 144;", .{ r_rb, r_nsb });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_bpt, r_cols });

    const rd_wa = try b.reg(.b64);
    const rd_wb = try b.reg(.b64);
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wa, r_rowa, r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wa, rd_w, rd_wa });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wb, r_rowb, r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wb, rd_w, rd_wb });

    // qs region at n*bpt*4; s region a further n*cols on.
    const rd_qs = try b.reg(.b64);
    const rd_sm = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[0], r_n, r_bpt });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_qs, tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qs, rd_x, rd_qs });
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[1], r_n, r_cols });
    try b.linef("mul.wide.u32 {s}, {s}, 1;", .{ rdt[0], tm[1] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sm, rd_qs, rdt[0] });

    for (acc) |a| try b.linef("mov.f32 {s}, 0f00000000;", .{a});

    // ---- super-block loop -------------------------------------------------
    const r_sb = try b.reg(.b32);
    const pr = try b.regs(.pred, 4);
    const lp = try b.newLabel("sb");
    const lp_end = try b.newLabel("sbend");
    try b.linef("mov.u32 {s}, 0;", .{r_sb});
    try b.label(lp);
    try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ pr[0], r_sb, r_nsb });
    try b.linef("@{s} bra {s};", .{ pr[0], lp_end });

    // Stage this super-block's activation d/Σq for all nt*8 columns x 8 sub-blocks.
    // bar.sync BEFORE the fill too: the previous iteration's readers must be done.
    const r_fi = try b.reg(.b32);
    const fdv = try b.regs(.f32, 2);
    const lf = try b.newLabel("fill");
    const lf_end = try b.newLabel("fillend");
    const lf_oob = try b.newLabel("filloob");
    try b.line("bar.sync 0;");
    try b.linef("mov.u32 {s}, {s};", .{ r_fi, r_t });
    try b.label(lf);
    try b.linef("setp.ge.u32 {s}, {s}, {d};", .{ pr[1], r_fi, nt * 64 });
    try b.linef("@{s} bra {s};", .{ pr[1], lf_end });
    try b.linef("shr.u32 {s}, {s}, 3;", .{ tm[0], r_fi }); // column within tile-set
    try b.linef("and.b32 {s}, {s}, 7;", .{ tm[1], r_fi }); // sub-block
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[2], r_col0, tm[0] });
    try b.linef("mov.f32 {s}, 0f00000000;", .{fdv[0]});
    try b.linef("mov.f32 {s}, 0f00000000;", .{fdv[1]});
    try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ pr[2], tm[2], r_n });
    try b.linef("@{s} bra {s};", .{ pr[2], lf_oob });
    try b.linef("shl.b32 {s}, {s}, 3;", .{ tm[3], r_sb });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[3], tm[3], tm[1] });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[4], tm[2], r_bpt, tm[3] });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[1], tm[4] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rd_x, rdt[1] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ fdv[0], rdt[2] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[3], rd_sm, rdt[1] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ fdv[1], rdt[3] });
    try b.label(lf_oob);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[5], r_fi });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[5], tm[5], r_smem });
    try b.linef("st.shared.f32 [{s}], {s};", .{ tm[5], fdv[0] });
    try b.linef("st.shared.f32 [{s}+{d}], {s};", .{ tm[5], sd_bytes, fdv[1] });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_fi, r_fi, nthreads });
    try b.linef("bra {s};", .{lf});
    try b.label(lf_end);
    try b.line("bar.sync 0;");

    // Super-block headers for the warp's two mma rows.
    const ha = try b.regs(.b32, 4);
    const hb = try b.regs(.b32, 4);
    const rd_sba = try b.reg(.b64);
    const rd_sbb = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, 144;", .{ tm[0], r_sb });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sba, rd_wa, rdt[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sbb, rd_wb, rdt[0] });
    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ ha[0], ha[1], ha[2], ha[3], rd_sba });
    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ hb[0], hb[1], hb[2], hb[3], rd_sbb });
    const fda = try b.reg(.f32);
    const fmia = try b.reg(.f32);
    const fdb = try b.reg(.f32);
    const fmib = try b.reg(.f32);
    const h16 = try b.regs(.b16, 4);
    try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ h16[0], h16[1], ha[0] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fda, h16[0] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fmia, h16[1] });
    try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ h16[2], h16[3], hb[0] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fdb, h16[2] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fmib, h16[3] });

    // Byte offset of this lane's weight words inside a 64-elem group.
    const r_qoff = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_qoff, r_t4 });

    const sc = try b.reg(.b32);
    const mn = try b.reg(.b32);
    const sc2 = try b.reg(.b32);
    const mn2 = try b.reg(.b32);
    const ft = try b.regs(.f32, 8);
    // Float scratch for the per-mma fold. These must be .f32 registers: the
    // builder's .b32 class is for integer/bit values, and ptxas types operands.
    const fx = try b.regs(.f32, 2);
    const rd_b = try b.reg(.b64);
    const r_kbase = try b.reg(.b32);

    var is: usize = 0;
    while (is < 8) : (is += 1) {
        const grp = is / 2;
        const half = is % 2;
        try b.comment("sub-block {d} (group {d}, {s} nibble)", .{ is, grp, if (half == 0) "low" else "high" });
        if (half == 0) {
            // One pair of loads per row serves both halves of this 64-elem group.
            try b.linef("add.u32 {s}, {s}, {d};", .{ tm[0], r_qoff, 16 + grp * 32 });
            try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[0] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_sba, rdt[0] });
            try b.linef("ld.global.u32 {s}, [{s}];", .{ wq[0], rdt[1] });
            try b.linef("ld.global.u32 {s}, [{s}+16];", .{ wq[1], rdt[1] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rd_sbb, rdt[0] });
            try b.linef("ld.global.u32 {s}, [{s}];", .{ wq[2], rdt[2] });
            try b.linef("ld.global.u32 {s}, [{s}+16];", .{ wq[3], rdt[2] });
        }
        // A fragment order is {row_a k, row_b k, row_a k+16, row_b k+16}.
        const src = [_][]const u8{ wq[0], wq[2], wq[1], wq[3] };
        for (afr, src) |dst, s| {
            if (half == 0) {
                try b.linef("and.b32 {s}, {s}, 0x0f0f0f0f;", .{ dst, s });
            } else {
                try b.linef("shr.u32 {s}, {s}, 4;", .{ dst, s });
                try b.linef("and.b32 {s}, {s}, 0x0f0f0f0f;", .{ dst, dst });
            }
        }
        try q4kScaleMin(&b, is, ha[1], ha[2], ha[3], sc, mn, tm[0]);
        try q4kScaleMin(&b, is, hb[1], hb[2], hb[3], sc2, mn2, tm[0]);
        // ft0 = d_a*sc_a, ft1 = dmin_a*m_a, ft2/ft3 the same for row_b.
        try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ ft[0], sc });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[0], ft[0], fda });
        try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ ft[1], mn });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[1], ft[1], fmia });
        try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ ft[2], sc2 });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[2], ft[2], fdb });
        try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ ft[3], mn2 });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[3], ft[3], fmib });
        // k offset of this sub-block within the row: sb*256 + is*32 + t4*4
        try b.linef("shl.b32 {s}, {s}, 8;", .{ r_kbase, r_sb });
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_kbase, r_kbase, is * 32 });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_kbase, r_kbase, r_qoff });

        var j: usize = 0;
        while (j < nt) : (j += 1) {
            const lb_oob = try b.newLabel("bz");
            const lb_st = try b.newLabel("bs");
            // B fragment: this lane supplies column (col0 + j*8 + gid).
            try b.linef("add.u32 {s}, {s}, {s};", .{ tm[1], r_col0, r_gid });
            try b.linef("add.u32 {s}, {s}, {d};", .{ tm[1], tm[1], j * 8 });
            try b.linef("mov.u32 {s}, 0;", .{bfr[0]});
            try b.linef("mov.u32 {s}, 0;", .{bfr[1]});
            try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ pr[3], tm[1], r_n });
            try b.linef("@{s} bra {s};", .{ pr[3], lb_oob });
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[2], tm[1], r_cols, r_kbase });
            try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[2] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_b, rd_qs, rdt[0] });
            try b.linef("ld.global.u32 {s}, [{s}];", .{ bfr[0], rd_b });
            try b.linef("ld.global.u32 {s}, [{s}+16];", .{ bfr[1], rd_b });
            try b.label(lb_oob);
            for (cfr) |c| try b.linef("mov.u32 {s}, 0;", .{c});
            try b.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                cfr[0], cfr[1], cfr[2], cfr[3],
                afr[0], afr[1], afr[2], afr[3],
                bfr[0], bfr[1],
                cfr[0], cfr[1], cfr[2], cfr[3],
            });
            _ = lb_st;
            // Activation d/Σq for this lane's two C columns, from shared.
            // shared index = (j*8 + t4*2 + {0,1})*8 + is
            try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[3], r_t4 });
            try b.linef("add.u32 {s}, {s}, {d};", .{ tm[3], tm[3], j * 8 });
            try b.linef("shl.b32 {s}, {s}, 3;", .{ tm[3], tm[3] });
            try b.linef("add.u32 {s}, {s}, {d};", .{ tm[3], tm[3], is });
            try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[3], tm[3] });
            try b.linef("add.u32 {s}, {s}, {s};", .{ tm[3], tm[3], r_smem });
            try b.linef("ld.shared.f32 {s}, [{s}];", .{ ft[4], tm[3] }); // da0
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ ft[5], tm[3], sd_bytes }); // sa0
            try b.linef("ld.shared.f32 {s}, [{s}+32];", .{ ft[6], tm[3] }); // da1
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ ft[7], tm[3], sd_bytes + 32 }); // sa1
            // acc += d*sc*da*c  -  dmin*m*da*Σq
            const pairs = [_][3]usize{ .{ 0, 4, 5 }, .{ 1, 6, 7 }, .{ 2, 4, 5 }, .{ 3, 6, 7 } };
            for (pairs, 0..) |pp, ci| {
                const vscale = if (ci < 2) ft[0] else ft[2];
                const mscale = if (ci < 2) ft[1] else ft[3];
                const da = ft[pp[1]];
                const sa = ft[pp[2]];
                const a = acc[j * 4 + ci];
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ fx[0], cfr[ci] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fx[1], vscale, da });
                try b.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ a, fx[1], fx[0], a });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fx[1], mscale, da });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fx[1], fx[1], sa });
                try b.linef("sub.f32 {s}, {s}, {s};", .{ a, a, fx[1] });
                _ = pp[0];
            }
        }
    }

    try b.linef("add.u32 {s}, {s}, 1;", .{ r_sb, r_sb });
    try b.linef("bra {s};", .{lp});
    try b.label(lp_end);

    // ---- epilogue: y[col*rows + row] --------------------------------------
    var j2: usize = 0;
    while (j2 < nt) : (j2 += 1) {
        const l_skip0 = try b.newLabel("st0");
        const l_skip1 = try b.newLabel("st1");
        try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_t4 });
        try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_col0 });
        try b.linef("add.u32 {s}, {s}, {d};", .{ tm[0], tm[0], j2 * 8 });
        // column c
        try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ pr[0], tm[0], r_n });
        try b.linef("@{s} bra {s};", .{ pr[0], l_skip0 });
        try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], tm[0], r_rows, r_rowa });
        try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[1] });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[0], acc[j2 * 4 + 0], f_scale });
        try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], ft[0] });
        try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], tm[0], r_rows, r_rowb });
        try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[1], tm[1] });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_y, rdt[1] });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[1], acc[j2 * 4 + 2], f_scale });
        try b.linef("st.global.f32 [{s}], {s};", .{ rdt[1], ft[1] });
        try b.label(l_skip0);
        // column c+1
        try b.linef("add.u32 {s}, {s}, 1;", .{ tm[2], tm[0] });
        try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ pr[1], tm[2], r_n });
        try b.linef("@{s} bra {s};", .{ pr[1], l_skip1 });
        try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], tm[2], r_rows, r_rowa });
        try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[1] });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[0], acc[j2 * 4 + 1], f_scale });
        try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], ft[0] });
        try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], tm[2], r_rows, r_rowb });
        try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[1], tm[1] });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_y, rdt[1] });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[1], acc[j2 * 4 + 3], f_scale });
        try b.linef("st.global.f32 [{s}], {s};", .{ rdt[1], ft[1] });
        try b.label(l_skip1);
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{sh_bytes});
    defer alloc.free(shared_decl);
    return b.build(
        "mmq_q4_k",
        "    .param .u64 p_w,\n    .param .u64 p_x,\n    .param .u64 p_y,\n    .param .u32 p_rows,\n    .param .u32 p_cols,\n    .param .u32 p_n,\n    .param .f32 p_scale",
        shared_decl,
    );
}

/// q4_k MMQ on the igemm-pipe tiling — the v1 `buildMmqQ4K` above is correct but
/// loses to cuBLASLt at prefill batch sizes because its warp owns only 16 rows,
/// so every A+B fragment-load pair feeds exactly ONE mma. This version uses the
/// proven 128x128 / 2x2-warp / MT=4 / NT=8 shape from `buildIgemmPipe`: 32 mmas
/// per 24 fragment loads, with the weight tile decoded ONCE into shared memory
/// and reused by all four warps.
///
/// The pipe already issues one `mma.m16n8k32` per 32-k substep, which is exactly
/// a q4_k sub-block — so the per-sub-block scale fold drops straight in after
/// each substep. The mma writes a scratch s32 quad (C operand held at zero) and
/// the result is folded into f32 accumulators:
///     acc += (d*sc)*da*c - (dmin*m)*da*sum(qa)
/// Both scale pairs are staged into shared alongside the tiles, so the fold costs
/// 48 shared f32 loads per 32 mmas instead of 4 per mma.
///
/// Differences from `buildIgemmPipe`, and why:
///   - A staging decodes q4_k nibbles instead of `cp.async`-copying raw s8. One
///     thread per row: 32 qs bytes in, 64 int8 out, plus the row's two sub-block
///     scale pairs. cp.async can't be used for A because the bytes are
///     transformed on the way in.
///   - Single-buffered. Without cp.async on A there is no async copy to overlap,
///     so double buffering would mean holding the next slab in registers on top
///     of 128 f32 accumulators. Fixing the arithmetic intensity is the dominant
///     term; pipelining is the next lever, not this one.
///   - No ldmatrix/swizzle (plain `ld.shared.u32` fragment loads, the pipe's
///     `use_ldmatrix=false` path). kstep is 64 here, so the pipe's (row&7)<<4
///     swizzle would XOR bit 6 and cross into the next row; a correct mask would
///     be (row&3)<<4. Left off until the tiling change is measured.
///
/// Requires cols % 256 == 0, rows % 128 == 0, and `n` padded to a multiple of 128
/// with the padding zero-filled (see `opMatmulQuantMmq`). Entry `mmq_pipe_q4_k`.
pub fn buildMmqPipeQ4K(alloc: std.mem.Allocator) ![:0]u8 {
    const BM = 128;
    const BN = 128;
    const kstep = 64;
    const KS = kstep / 32; // 32-k substeps == q4_k sub-blocks per slab
    const MT = 4;
    const NT = 8;
    const ATILE = BM * kstep;
    const BTILE = BN * kstep;
    const ASC = BM * KS * 2 * 4; // (d*sc, dmin*m) f32 pair per row per substep
    const BSC = BN * KS * 2 * 4; // (da, sum qa) f32 pair per col per substep
    const B_OFF = ATILE;
    const ASC_OFF = ATILE + BTILE;
    const BSC_OFF = ASC_OFF + ASC;
    const SH = BSC_OFF + BSC;

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4);
    const bf = try b.regs(.b32, NT * 2);
    const ct = try b.regs(.b32, 4); // mma destination (C operand is the zero quad)
    const zq = try b.regs(.b32, 4); // permanently zero
    const rsc = try b.regs(.f32, MT * 4); // per-mi: (dsc,dm) for rows gid and gid+8
    const csc = try b.regs(.f32, 4); // per-nj: (da,sa) for cols tf*2 and tf*2+1

    const rd_w = try b.reg(.b64);
    const rd_x = try b.reg(.b64);
    const rd_y = try b.reg(.b64);
    const r_rows = try b.reg(.b32);
    const r_cols = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const f_scale = try b.reg(.f32);
    try b.linef("ld.param.u64 {s}, [p_w];", .{rd_w});
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_y];", .{rd_y});
    try b.linef("ld.param.u32 {s}, [p_rows];", .{r_rows});
    try b.linef("ld.param.u32 {s}, [p_cols];", .{r_cols});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_w, rd_w });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_y, rd_y });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const r_nsb = try b.reg(.b32);
    const r_rb = try b.reg(.b32);
    const r_bpt = try b.reg(.b32);
    const tm = try b.regs(.b32, 8);
    const rdt = try b.regs(.b64, 4);

    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });
    try b.linef("shr.u32 {s}, {s}, 8;", .{ r_nsb, r_cols });
    try b.linef("mul.lo.u32 {s}, {s}, 144;", .{ r_rb, r_nsb });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_bpt, r_cols });

    // Activation regions: qs at n*bpt*4, sums a further n*cols on.
    const rd_qs = try b.reg(.b64);
    const rd_sm = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[0], r_n, r_bpt });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_qs, tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qs, rd_x, rd_qs });
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[1], r_n, r_cols });
    try b.linef("mul.wide.u32 {s}, {s}, 1;", .{ rdt[0], tm[1] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_sm, rd_qs, rdt[0] });

    // Staging identities: one thread per A row and per B column (BM=BN=128=nthreads).
    const rd_wrow = try b.reg(.b64); // W + (row0+tid)*row_bytes
    const rd_bcol = try b.reg(.b64); // qs + (col0+tid)*cols
    const r_gcol = try b.reg(.b32); // col0 + tid (for the activation scale gather)
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], r_row0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wrow, tm[0], r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wrow, rd_w, rd_wrow });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_gcol, r_col0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_bcol, r_gcol, r_cols });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bcol, rd_qs, rd_bcol });

    // Shared destinations for this thread's staged row/column.
    const r_adst = try b.reg(.b32);
    const r_bdst = try b.reg(.b32);
    const r_ascd = try b.reg(.b32);
    const r_bscd = try b.reg(.b32);
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_adst, r_t, kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_adst, r_adst, r_smem });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bdst, r_t, kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bdst, r_bdst, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bdst, r_bdst, B_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_ascd, r_t, KS * 8 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascd, r_ascd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascd, r_ascd, ASC_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscd, r_t, KS * 8 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscd, r_bscd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscd, r_bscd, BSC_OFF });

    // Fragment-load bases (pipe's use_ldmatrix=false layout).
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl, tm[0], kstep });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl, tm[0], kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, B_OFF });

    // Scale-read bases: row (wm*64 + mi*16 + gid) and col (wn*64 + nj*8 + tf*2).
    const r_ascr = try b.reg(.b32);
    const r_bscr = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_ascr, tm[0], KS * 8 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascr, r_ascr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascr, r_ascr, ASC_OFF });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], tm[1] });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscr, tm[0], KS * 8 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscr, r_bscr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscr, r_bscr, BSC_OFF });

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});
    for (zq) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    // ---- slab loop --------------------------------------------------------
    const r_i = try b.reg(.b32);
    const r_nslab = try b.reg(.b32);
    const pr = try b.regs(.pred, 3);
    const lp = try b.newLabel("slab");
    const hi_lbl = try b.newLabel("schi");
    const sd_lbl = try b.newLabel("scdone");
    try b.linef("shr.u32 {s}, {s}, 6;", .{ r_nslab, r_cols }); // cols/kstep
    try b.linef("mov.u32 {s}, 0;", .{r_i});
    try b.label(lp);

    // -- stage A: decode this thread's row of q4_k nibbles into shared -------
    const hh = try b.regs(.b32, 4);
    const qw = try b.regs(.b32, 8);
    const nib = try b.regs(.b32, 8);
    try b.linef("shr.u32 {s}, {s}, 2;", .{ tm[0], r_i }); // sb = i/4
    try b.linef("and.b32 {s}, {s}, 3;", .{ tm[1], r_i }); // grp = i&3
    try b.linef("mul.lo.u32 {s}, {s}, 144;", .{ tm[2], tm[0] });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[2] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_wrow, rdt[0] });
    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ hh[0], hh[1], hh[2], hh[3], rdt[1] });
    try b.linef("shl.b32 {s}, {s}, 5;", .{ tm[3], tm[1] }); // grp*32
    try b.linef("add.u32 {s}, {s}, 16;", .{ tm[3], tm[3] });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[3] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rdt[1], rdt[0] });
    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ qw[0], qw[1], qw[2], qw[3], rdt[2] });
    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}+16];", .{ qw[4], qw[5], qw[6], qw[7], rdt[2] });
    for (0..8) |j| try b.linef("and.b32 {s}, {s}, 0x0f0f0f0f;", .{ nib[j], qw[j] });
    try b.linef("st.shared.v4.u32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_adst, nib[0], nib[1], nib[2], nib[3] });
    try b.linef("st.shared.v4.u32 [{s}+16], {{{s}, {s}, {s}, {s}}};", .{ r_adst, nib[4], nib[5], nib[6], nib[7] });
    for (0..8) |j| {
        try b.linef("shr.u32 {s}, {s}, 4;", .{ nib[j], qw[j] });
        try b.linef("and.b32 {s}, {s}, 0x0f0f0f0f;", .{ nib[j], nib[j] });
    }
    try b.linef("st.shared.v4.u32 [{s}+32], {{{s}, {s}, {s}, {s}}};", .{ r_adst, nib[0], nib[1], nib[2], nib[3] });
    try b.linef("st.shared.v4.u32 [{s}+48], {{{s}, {s}, {s}, {s}}};", .{ r_adst, nib[4], nib[5], nib[6], nib[7] });

    // -- stage A scales: (d*sc, dmin*m) for sub-blocks grp*2 and grp*2+1 -----
    // Runtime `is` here (grp comes from the slab index), so this is ggml's
    // get_scale_min_k4 branch rather than the comptime form used elsewhere.
    const sc0 = try b.reg(.b32);
    const mn0 = try b.reg(.b32);
    const sc1 = try b.reg(.b32);
    const mn1 = try b.reg(.b32);
    const h16 = try b.regs(.b16, 2);
    const fd = try b.reg(.f32);
    const fdm = try b.reg(.f32);
    const fs = try b.regs(.f32, 4);
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[4], tm[1] }); // is0 = grp*2
    try b.linef("shl.b32 {s}, {s}, 3;", .{ tm[5], tm[4] }); // is0*8
    try b.linef("setp.ge.u32 {s}, {s}, 4;", .{ pr[0], tm[4] });
    try b.linef("@{s} bra {s};", .{ pr[0], hi_lbl });
    try b.linef("shr.u32 {s}, {s}, {s};", .{ sc0, hh[1], tm[5] });
    try b.linef("and.b32 {s}, {s}, 63;", .{ sc0, sc0 });
    try b.linef("shr.u32 {s}, {s}, {s};", .{ mn0, hh[2], tm[5] });
    try b.linef("and.b32 {s}, {s}, 63;", .{ mn0, mn0 });
    try b.linef("add.u32 {s}, {s}, 8;", .{ tm[6], tm[5] });
    try b.linef("shr.u32 {s}, {s}, {s};", .{ sc1, hh[1], tm[6] });
    try b.linef("and.b32 {s}, {s}, 63;", .{ sc1, sc1 });
    try b.linef("shr.u32 {s}, {s}, {s};", .{ mn1, hh[2], tm[6] });
    try b.linef("and.b32 {s}, {s}, 63;", .{ mn1, mn1 });
    try b.linef("bra {s};", .{sd_lbl});
    try b.label(hi_lbl);
    try b.linef("add.u32 {s}, {s}, -32;", .{ tm[6], tm[5] }); // (is0-4)*8
    inline for (.{ 0, 1 }) |half| {
        const sc = if (half == 0) sc0 else sc1;
        const mn = if (half == 0) mn0 else mn1;
        if (half == 1) try b.linef("add.u32 {s}, {s}, 8;", .{ tm[6], tm[6] });
        try b.linef("shr.u32 {s}, {s}, {s};", .{ tm[7], hh[3], tm[6] });
        try b.linef("and.b32 {s}, {s}, 15;", .{ sc, tm[7] });
        try b.linef("shr.u32 {s}, {s}, 4;", .{ mn, tm[7] });
        try b.linef("and.b32 {s}, {s}, 15;", .{ mn, mn });
        try b.linef("shr.u32 {s}, {s}, {s};", .{ tm[7], hh[1], tm[6] });
        try b.linef("shr.u32 {s}, {s}, 6;", .{ tm[7], tm[7] });
        try b.linef("and.b32 {s}, {s}, 3;", .{ tm[7], tm[7] });
        try b.linef("shl.b32 {s}, {s}, 4;", .{ tm[7], tm[7] });
        try b.linef("or.b32 {s}, {s}, {s};", .{ sc, sc, tm[7] });
        try b.linef("shr.u32 {s}, {s}, {s};", .{ tm[7], hh[2], tm[6] });
        try b.linef("shr.u32 {s}, {s}, 6;", .{ tm[7], tm[7] });
        try b.linef("and.b32 {s}, {s}, 3;", .{ tm[7], tm[7] });
        try b.linef("shl.b32 {s}, {s}, 4;", .{ tm[7], tm[7] });
        try b.linef("or.b32 {s}, {s}, {s};", .{ mn, mn, tm[7] });
    }
    try b.label(sd_lbl);
    try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ h16[0], h16[1], hh[0] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fd, h16[0] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fdm, h16[1] });
    try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ fs[0], sc0 });
    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[0], fs[0], fd });
    try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ fs[1], mn0 });
    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[1], fs[1], fdm });
    try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ fs[2], sc1 });
    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[2], fs[2], fd });
    try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ fs[3], mn1 });
    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[3], fs[3], fdm });
    try b.linef("st.shared.v4.f32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_ascd, fs[0], fs[1], fs[2], fs[3] });

    // -- stage B: activation int8 (already in the right layout) + its scales --
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_i }); // i*kstep
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_bcol, rdt[0] });
    // ⚠️ All FOUR loads issue before any store. Interleaving load/store/load/store
    // makes the second pair wait on the first pair's registers, serializing two
    // ~500-cycle global latencies that should overlap; this costs 16 registers of
    // staging and measured +7% on the whole kernel.
    for (0..4) |q4| try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ qw[q4 * 4], qw[q4 * 4 + 1], qw[q4 * 4 + 2], qw[q4 * 4 + 3], rdt[1], q4 * 16 });
    for (0..4) |q4| try b.linef("st.shared.v4.u32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ r_bdst, q4 * 16, qw[q4 * 4], qw[q4 * 4 + 1], qw[q4 * 4 + 2], qw[q4 * 4 + 3] });
    // blk = i*KS + ks; d and sum live in separate regions, both indexed col*bpt+blk.
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ tm[0], r_i, KS });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], r_gcol, r_bpt, tm[0] });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[1] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_x, rdt[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rd_sm, rdt[0] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ fs[0], rdt[1] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ fs[1], rdt[2] });
    try b.linef("ld.global.f32 {s}, [{s}+4];", .{ fs[2], rdt[1] });
    try b.linef("ld.global.f32 {s}, [{s}+4];", .{ fs[3], rdt[2] });
    try b.linef("st.shared.v4.f32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_bscd, fs[0], fs[1], fs[2], fs[3] });

    try b.line("bar.sync 0;");

    // ---- compute: KS substeps of MT x NT mmas, folding scales each substep --
    var ks: usize = 0;
    while (ks < KS) : (ks += 1) {
        var mi: usize = 0;
        while (mi < MT) : (mi += 1) {
            const o = mi * 16 * kstep + ks * 32;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 0], r_asl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 1], r_asl, o + 8 * kstep });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 2], r_asl, o + 16 });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 4 + 3], r_asl, o + 8 * kstep + 16 });
            // Row scale pairs for this mi: rows (mi*16+gid) and (mi*16+gid+8).
            const so = mi * 16 * KS * 8 + ks * 8;
            try b.linef("ld.shared.v2.f32 {{{s}, {s}}}, [{s}+{d}];", .{ rsc[mi * 4 + 0], rsc[mi * 4 + 1], r_ascr, so });
            try b.linef("ld.shared.v2.f32 {{{s}, {s}}}, [{s}+{d}];", .{ rsc[mi * 4 + 2], rsc[mi * 4 + 3], r_ascr, so + 8 * KS * 8 });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            const o = nj * 8 * kstep + ks * 32;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 0], r_bsl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bf[nj * 2 + 1], r_bsl, o + 16 });
        }
        nj = 0;
        while (nj < NT) : (nj += 1) {
            // Column scale pairs for this nj: cols (nj*8+tf*2) and +1.
            const co = nj * 8 * KS * 8 + ks * 8;
            try b.linef("ld.shared.v2.f32 {{{s}, {s}}}, [{s}+{d}];", .{ csc[0], csc[1], r_bscr, co });
            try b.linef("ld.shared.v2.f32 {{{s}, {s}}}, [{s}+{d}];", .{ csc[2], csc[3], r_bscr, co + KS * 8 });
            mi = 0;
            while (mi < MT) : (mi += 1) {
                try b.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                    ct[0],          ct[1],          ct[2],          ct[3],
                    af[mi * 4 + 0], af[mi * 4 + 1], af[mi * 4 + 2], af[mi * 4 + 3],
                    bf[nj * 2 + 0], bf[nj * 2 + 1],
                    zq[0],          zq[1],          zq[2],          zq[3],
                });
                // c0=(row gid,col tf*2) c1=(gid,tf*2+1) c2=(gid+8,tf*2) c3=(gid+8,+1)
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                const rows2 = [_]usize{ 0, 0, 2, 2 }; // index into rsc for row a/b
                const cols2 = [_]usize{ 0, 2, 0, 2 }; // index into csc for col 0/1
                for (0..4) |ci| {
                    const dsc = rsc[mi * 4 + rows2[ci]];
                    const dm = rsc[mi * 4 + rows2[ci] + 1];
                    const da = csc[cols2[ci]];
                    const sa = csc[cols2[ci] + 1];
                    try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ fs[0], ct[ci] });
                    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[1], dsc, da });
                    try b.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ a[ci], fs[1], fs[0], a[ci] });
                    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[1], dm, da });
                    try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[1], fs[1], sa });
                    try b.linef("sub.f32 {s}, {s}, {s};", .{ a[ci], a[ci], fs[1] });
                }
            }
        }
    }
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, 1;", .{ r_i, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ pr[1], r_i, r_nslab });
    try b.linef("@{s} bra {s};", .{ pr[1], lp });

    // ---- store: y[col*rows + row] -----------------------------------------
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, tm[0] });
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        var nj2: usize = 0;
        while (nj2 < NT) : (nj2 += 1) {
            const a = acc[(mi2 * NT + nj2) * 4 ..][0..4];
            const rowoff = [_]usize{ 0, 0, 8, 8 };
            const coloff = [_]usize{ 0, 1, 0, 1 };
            for (0..4) |ci| {
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[1], r_crow, mi2 * 16 + rowoff[ci] });
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[2], r_ccol, nj2 * 8 + coloff[ci] });
                try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[3], tm[2], r_rows, tm[1] });
                try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[3] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[0], a[ci], f_scale });
                try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], fs[0] });
            }
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH});
    defer alloc.free(shared_decl);
    return b.build(
        "mmq_pipe_q4_k",
        "    .param .u64 p_w,\n    .param .u64 p_x,\n    .param .u64 p_y,\n    .param .u32 p_rows,\n    .param .u32 p_cols,\n    .param .u32 p_n,\n    .param .f32 p_scale",
        shared_decl,
    );
}

/// q6_k MMQ, the pipe-tiled twin of `buildMmqPipeQ4K`. Same 128x128 / 2x2-warp /
/// MT=4 / NT=8 shape and the same "decode the weight tile once into shared"
/// idea; three things differ, all forced by the format:
///
///   1. q6_k's scale changes every 16 elements, not 32, so the mma is
///      `m16n8k16` (one per sub-block) instead of `m16n8k32`. Same MACs per k,
///      twice the instructions — but no wasted lanes, which zero-padding a
///      k32 mma to cover one 16-element scale would cost.
///   2. `v = d*sc*(q-32)` has NO min term, so there is no sum(qa) to carry and
///      the fold is just `acc += d*sc*da*c`. The -32 is folded into the staged
///      weight instead (SWAR byte subtract, see below), which keeps the mma
///      operand a plain s8 in [-32,31].
///   3. The 210-byte block is only 2-BYTE aligned (210 = 2 mod 4, and the row
///      stride nsb*210 inherits that), so the weight bytes come in as u16 pairs
///      assembled into u32 — the same dance `gemv_q6_k_q8n` does. q4_k's 144-byte
///      block is 16-aligned and could use v4.u32.
///
/// Element mapping, checked against ggml's `dequantize_row_q6_K`: for slab
/// `i_local` (0..3) of a super-block, with h = i_local>>1 and parity = i_local&1,
/// element e (0..63) of the slab reads ql[h*64 + e] (low nibble when parity==0,
/// high when 1), qh[h*32 + (e&31)] shifted right by 4*parity + 2*(e>>5), and uses
/// scale index i_local*4 + s for substep s = e>>4. The natural element order lines
/// up with the activation layout, so no k-permutation is needed.
///
/// Requires cols % 256 == 0 and rows % 128 == 0; `n` padded to 128 by the caller.
/// Entry `mmq_pipe_q6_k`.
pub fn buildMmqPipeQ6K(alloc: std.mem.Allocator) ![:0]u8 {
    const BM = 128;
    const BN = 128;
    const kstep = 64;
    const KS = kstep / 16; // 16-k substeps == q6_k sub-blocks per slab
    const MT = 4;
    const NT = 8;
    const ATILE = BM * kstep;
    const BTILE = BN * kstep;
    const ASC = BM * KS * 4; // d*sc f32 per row per substep
    const BSC = BN * KS * 4; // da f32 per col per substep
    const B_OFF = ATILE;
    const ASC_OFF = ATILE + BTILE;
    const BSC_OFF = ASC_OFF + ASC;
    const SH = BSC_OFF + BSC;

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 2);
    const bfr = try b.regs(.b32, NT);
    const ct = try b.regs(.b32, 4);
    const zq = try b.regs(.b32, 4);
    const rsc = try b.regs(.f32, MT * 2);
    const csc = try b.regs(.f32, 2);

    const rd_w = try b.reg(.b64);
    const rd_x = try b.reg(.b64);
    const rd_y = try b.reg(.b64);
    const r_rows = try b.reg(.b32);
    const r_cols = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const f_scale = try b.reg(.f32);
    try b.linef("ld.param.u64 {s}, [p_w];", .{rd_w});
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_y];", .{rd_y});
    try b.linef("ld.param.u32 {s}, [p_rows];", .{r_rows});
    try b.linef("ld.param.u32 {s}, [p_cols];", .{r_cols});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_w, rd_w });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_y, rd_y });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const r_nsb = try b.reg(.b32);
    const r_rb = try b.reg(.b32);
    const r_bpt = try b.reg(.b32);
    const tm = try b.regs(.b32, 8);
    const rdt = try b.regs(.b64, 4);

    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });
    try b.linef("shr.u32 {s}, {s}, 8;", .{ r_nsb, r_cols });
    try b.linef("mul.lo.u32 {s}, {s}, 210;", .{ r_rb, r_nsb });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_bpt, r_cols });

    const rd_qs = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[0], r_n, r_bpt });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_qs, tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qs, rd_x, rd_qs });

    const rd_wrow = try b.reg(.b64);
    const rd_bcol = try b.reg(.b64);
    const r_gcol = try b.reg(.b32);
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], r_row0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wrow, tm[0], r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wrow, rd_w, rd_wrow });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_gcol, r_col0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_bcol, r_gcol, r_cols });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bcol, rd_qs, rd_bcol });

    const r_adst = try b.reg(.b32);
    const r_bdst = try b.reg(.b32);
    const r_ascd = try b.reg(.b32);
    const r_bscd = try b.reg(.b32);
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_adst, r_t, kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_adst, r_adst, r_smem });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bdst, r_t, kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bdst, r_bdst, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bdst, r_bdst, B_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_ascd, r_t, KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascd, r_ascd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascd, r_ascd, ASC_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscd, r_t, KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscd, r_bscd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscd, r_bscd, BSC_OFF });

    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl, tm[0], kstep });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl, tm[0], kstep });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, B_OFF });

    const r_ascr = try b.reg(.b32);
    const r_bscr = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_ascr, tm[0], KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascr, r_ascr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascr, r_ascr, ASC_OFF });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], tm[1] });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscr, tm[0], KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscr, r_bscr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscr, r_bscr, BSC_OFF });

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});
    for (zq) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    const r_i = try b.reg(.b32);
    const r_nslab = try b.reg(.b32);
    const pr = try b.regs(.pred, 2);
    const lp = try b.newLabel("slab6");
    try b.linef("shr.u32 {s}, {s}, 6;", .{ r_nslab, r_cols });
    try b.linef("mov.u32 {s}, 0;", .{r_i});
    try b.label(lp);

    // -- stage A: decode this thread's row of q6_k into shared ---------------
    const rs = try b.regs(.b16, 2);
    const qh = try b.regs(.b32, 8); // qh groups g=0..7, reused by g=8..15
    const gw = try b.regs(.b32, 4); // one v4 store's worth of decoded bytes
    const r_par = try b.reg(.b32);
    const r_sh4 = try b.reg(.b32);
    const rd_ql = try b.reg(.b64);
    const rd_qh = try b.reg(.b64);
    try b.linef("shr.u32 {s}, {s}, 2;", .{ tm[0], r_i }); // sb
    try b.linef("and.b32 {s}, {s}, 3;", .{ tm[1], r_i }); // i_local
    try b.linef("mul.lo.u32 {s}, {s}, 210;", .{ tm[2], tm[0] });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[2] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_wrow, rdt[0] }); // super-block base
    try b.linef("shr.u32 {s}, {s}, 1;", .{ tm[3], tm[1] }); // h
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_par, tm[1] }); // parity
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_sh4, r_par }); // 4*parity
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[4], tm[3] }); // h*64
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[4] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ql, rdt[1], rdt[0] });
    try b.linef("shl.b32 {s}, {s}, 5;", .{ tm[4], tm[3] }); // h*32
    try b.linef("add.u32 {s}, {s}, 128;", .{ tm[4], tm[4] });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[4] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qh, rdt[1], rdt[0] });
    // Round both pointers down to 4 bytes and recombine with a funnel shift of
    // (addr & 3) * 8, which is 0 or 16 (the base is always even).
    const r_shb = try b.reg(.b32);
    const rd_qla = try b.reg(.b64);
    const rd_qha = try b.reg(.b64);
    const wcar = try b.reg(.b32);
    const wnew = try b.regs(.b32, 4);
    try b.linef("cvt.u32.u64 {s}, {s};", .{ tm[5], rd_ql });
    try b.linef("and.b32 {s}, {s}, 3;", .{ tm[5], tm[5] });
    try b.linef("shl.b32 {s}, {s}, 3;", .{ r_shb, tm[5] });
    try b.linef("and.b64 {s}, {s}, -4;", .{ rd_qla, rd_ql });
    try b.linef("and.b64 {s}, {s}, -4;", .{ rd_qha, rd_qh });
    // qh groups (8 u32 = 32 bytes): 9 aligned loads feed 8 funnels.
    try b.linef("ld.global.u32 {s}, [{s}];", .{ wcar, rd_qha });
    for (0..8) |g| {
        try b.linef("ld.global.u32 {s}, [{s}+{d}];", .{ wnew[0], rd_qha, g * 4 + 4 });
        try b.linef("shf.r.clamp.b32 {s}, {s}, {s}, {s};", .{ qh[g], wcar, wnew[0], r_shb });
        try b.linef("mov.b32 {s}, {s};", .{ wcar, wnew[0] });
    }
    // 16 groups of 4 elements; 4 groups per v4.u32 shared store.
    for (0..4) |gg| {
        for (0..4) |k| {
            const g = gg * 4 + k;
            if (g == 0) try b.linef("ld.global.u32 {s}, [{s}];", .{ wcar, rd_qla });
            try b.linef("ld.global.u32 {s}, [{s}+{d}];", .{ wnew[k], rd_qla, g * 4 + 4 });
            try b.linef("shf.r.clamp.b32 {s}, {s}, {s}, {s};", .{ gw[k], wcar, wnew[k], r_shb });
            try b.linef("mov.b32 {s}, {s};", .{ wcar, wnew[k] });
            // low nibble when parity==0, high when 1 -> one runtime shift
            try b.linef("shr.u32 {s}, {s}, {s};", .{ gw[k], gw[k], r_sh4 });
            try b.linef("and.b32 {s}, {s}, 0x0f0f0f0f;", .{ gw[k], gw[k] });
            // qh bits: shift = 4*parity + 2*(e>>5); e>>5 is 0 for g<8, 1 for g>=8
            try b.linef("add.u32 {s}, {s}, {d};", .{ tm[5], r_sh4, if (g >= 8) @as(usize, 2) else 0 });
            try b.linef("shr.u32 {s}, {s}, {s};", .{ tm[6], qh[g & 7], tm[5] });
            try b.linef("and.b32 {s}, {s}, 0x03030303;", .{ tm[6], tm[6] });
            try b.linef("shl.b32 {s}, {s}, 4;", .{ tm[6], tm[6] });
            try b.linef("or.b32 {s}, {s}, {s};", .{ gw[k], gw[k], tm[6] });
            // byte-wise -32. Each byte is in [0,63] so bit7 is clear; setting it
            // first stops the borrow crossing into the next byte, and the final
            // XOR removes it again. (v|0x80)-0x20 never borrows, and
            // ((b+96)^128) == (b-32) mod 256 for b in [0,63].
            try b.linef("or.b32 {s}, {s}, 0x80808080;", .{ gw[k], gw[k] });
            try b.linef("sub.u32 {s}, {s}, 0x20202020;", .{ gw[k], gw[k] });
            try b.linef("xor.b32 {s}, {s}, 0x80808080;", .{ gw[k], gw[k] });
        }
        try b.linef("st.shared.v4.u32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ r_adst, gg * 16, gw[0], gw[1], gw[2], gw[3] });
    }
    // -- stage A scales: d * (signed int8) sc[i_local*4 + s] -----------------
    const fd6 = try b.reg(.f32);
    const fsv = try b.regs(.f32, 4);
    try b.linef("ld.global.u16 {s}, [{s}+208];", .{ rs[0], rdt[1] });
    try b.linef("cvt.f32.f16 {s}, {s};", .{ fd6, rs[0] });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[5], tm[1] }); // i_local*4
    try b.linef("add.u32 {s}, {s}, 192;", .{ tm[5], tm[5] });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[5] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rdt[1], rdt[0] });
    try b.linef("cvt.u32.u64 {s}, {s};", .{ tm[7], rdt[2] });
    try b.linef("and.b32 {s}, {s}, 3;", .{ tm[7], tm[7] });
    try b.linef("shl.b32 {s}, {s}, 3;", .{ tm[7], tm[7] });
    try b.linef("and.b64 {s}, {s}, -4;", .{ rdt[3], rdt[2] });
    try b.linef("ld.global.u32 {s}, [{s}];", .{ tm[6], rdt[3] });
    try b.linef("ld.global.u32 {s}, [{s}+4];", .{ wcar, rdt[3] });
    try b.linef("shf.r.clamp.b32 {s}, {s}, {s}, {s};", .{ tm[6], tm[6], wcar, tm[7] });
    for (0..4) |s| {
        try b.linef("shl.b32 {s}, {s}, {d};", .{ tm[7], tm[6], 24 - 8 * s });
        try b.linef("shr.s32 {s}, {s}, 24;", .{ tm[7], tm[7] }); // sign-extend
        try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ fsv[s], tm[7] });
        try b.linef("mul.f32 {s}, {s}, {s};", .{ fsv[s], fsv[s], fd6 });
    }
    try b.linef("st.shared.v4.f32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_ascd, fsv[0], fsv[1], fsv[2], fsv[3] });

    // -- stage B: activations (16-byte aligned) + their per-32-block scale ----
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_i });
    try b.linef("cvt.u64.u32 {s}, {s};", .{ rdt[0], tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[1], rd_bcol, rdt[0] });
    for (0..4) |q| {
        try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ qh[0], qh[1], qh[2], qh[3], rdt[1], q * 16 });
        try b.linef("st.shared.v4.u32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ r_bdst, q * 16, qh[0], qh[1], qh[2], qh[3] });
    }
    // Substeps 0,1 fall in activation block i*2 and 2,3 in i*2+1 (q8_1 blocks are
    // 32 elements, q6_k sub-blocks 16), so each da is used by two substeps.
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_i });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[1], r_gcol, r_bpt, tm[0] });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[1] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[2], rd_x, rdt[0] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ fsv[0], rdt[2] });
    try b.linef("ld.global.f32 {s}, [{s}+4];", .{ fsv[2], rdt[2] });
    try b.linef("mov.f32 {s}, {s};", .{ fsv[1], fsv[0] });
    try b.linef("mov.f32 {s}, {s};", .{ fsv[3], fsv[2] });
    try b.linef("st.shared.v4.f32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_bscd, fsv[0], fsv[1], fsv[2], fsv[3] });

    try b.line("bar.sync 0;");

    // ---- compute -----------------------------------------------------------
    var s: usize = 0;
    while (s < KS) : (s += 1) {
        var mi: usize = 0;
        while (mi < MT) : (mi += 1) {
            const o = mi * 16 * kstep + s * 16;
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 2 + 0], r_asl, o });
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ af[mi * 2 + 1], r_asl, o + 8 * kstep });
            const so = mi * 16 * KS * 4 + s * 4;
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ rsc[mi * 2 + 0], r_ascr, so });
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ rsc[mi * 2 + 1], r_ascr, so + 8 * KS * 4 });
        }
        var nj: usize = 0;
        while (nj < NT) : (nj += 1) {
            try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ bfr[nj], r_bsl, nj * 8 * kstep + s * 16 });
        }
        nj = 0;
        while (nj < NT) : (nj += 1) {
            const co = nj * 8 * KS * 4 + s * 4;
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ csc[0], r_bscr, co });
            try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ csc[1], r_bscr, co + KS * 4 });
            mi = 0;
            while (mi < MT) : (mi += 1) {
                try b.linef("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s}}}, {{{s},{s},{s},{s}}};", .{
                    ct[0],          ct[1],          ct[2], ct[3],
                    af[mi * 2 + 0], af[mi * 2 + 1],
                    bfr[nj],
                    zq[0],          zq[1],          zq[2], zq[3],
                });
                const a = acc[(mi * NT + nj) * 4 ..][0..4];
                const rows2 = [_]usize{ 0, 0, 1, 1 };
                const cols2 = [_]usize{ 0, 1, 0, 1 };
                for (0..4) |ci| {
                    try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ fsv[0], ct[ci] });
                    try b.linef("mul.f32 {s}, {s}, {s};", .{ fsv[1], rsc[mi * 2 + rows2[ci]], csc[cols2[ci]] });
                    try b.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ a[ci], fsv[1], fsv[0], a[ci] });
                }
            }
        }
    }
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, 1;", .{ r_i, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ pr[0], r_i, r_nslab });
    try b.linef("@{s} bra {s};", .{ pr[0], lp });

    // ---- store -------------------------------------------------------------
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, tm[0] });
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        var nj2: usize = 0;
        while (nj2 < NT) : (nj2 += 1) {
            const a = acc[(mi2 * NT + nj2) * 4 ..][0..4];
            const rowoff = [_]usize{ 0, 0, 8, 8 };
            const coloff = [_]usize{ 0, 1, 0, 1 };
            for (0..4) |ci| {
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[1], r_crow, mi2 * 16 + rowoff[ci] });
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[2], r_ccol, nj2 * 8 + coloff[ci] });
                try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[3], tm[2], r_rows, tm[1] });
                try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[3] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fsv[0], a[ci], f_scale });
                try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], fsv[0] });
            }
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH});
    defer alloc.free(shared_decl);
    return b.build(
        "mmq_pipe_q6_k",
        "    .param .u64 p_w,\n    .param .u64 p_x,\n    .param .u64 p_y,\n    .param .u32 p_rows,\n    .param .u32 p_cols,\n    .param .u32 p_n,\n    .param .f32 p_scale",
        shared_decl,
    );
}

/// Decode one lane's `dims`-wide K or V fragment from `addr` into `dst`, for one of
/// the three KV cache formats. **Mirrors `elt.emitKVLoad` instruction for
/// instruction** — same load widths, same sign-extend shifts, same multiply order —
/// so `attn_split_g*` stays bit-identical to the `attn_split*` kernel it replaces.
/// A reassociation here would silently change the KV cache that decode reads.
///
/// `kt` are b32 scratch, `hs` b16 scratch (empty for f32), `f_kd` the q8_0 block
/// scale, `rd_t` a scratch address register, and `rd_qoff` the loop-invariant q8_0
/// quant offset `(lane*dims & 31) + 2`.
fn emitKvFrag(
    b: *ptx.Builder,
    kvf: elt.KvFmt,
    dims: usize,
    dst: []const []const u8,
    addr: []const u8,
    kt: []const []const u8,
    hs: []const []const u8,
    f_kd: []const u8,
    rd_t: []const u8,
    rd_qoff: []const u8,
) !void {
    switch (kvf) {
        .f32 => for (0..dims / 4) |L| try b.linef("ld.global.v4.f32 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ dst[L * 4], dst[L * 4 + 1], dst[L * 4 + 2], dst[L * 4 + 3], addr, L * 16 }),
        .f16 => {
            // 4 halfs per u32; v4.u32 (8 halfs) when the fragment is wide enough.
            if (dims % 8 == 0) {
                for (0..dims / 8) |L| {
                    try b.linef("ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ kt[0], kt[1], kt[2], kt[3], addr, L * 16 });
                    for (0..4) |w| {
                        const d0 = L * 8 + w * 2;
                        try b.linef("mov.b32 {{{s}, {s}}}, {s}; cvt.f32.f16 {s}, {s}; cvt.f32.f16 {s}, {s};", .{ hs[0], hs[1], kt[w], dst[d0], hs[0], dst[d0 + 1], hs[1] });
                    }
                }
            } else {
                for (0..dims / 4) |L| {
                    try b.linef("ld.global.v2.u32 {{{s}, {s}}}, [{s}+{d}];", .{ kt[0], kt[1], addr, L * 8 });
                    for (0..2) |w| {
                        const d0 = L * 4 + w * 2;
                        try b.linef("mov.b32 {{{s}, {s}}}, {s}; cvt.f32.f16 {s}, {s}; cvt.f32.f16 {s}, {s};", .{ hs[0], hs[1], kt[w], dst[d0], hs[0], dst[d0 + 1], hs[1] });
                    }
                }
            }
        },
        .q8_0 => {
            try b.linef("ld.global.b16 {s}, [{s}]; cvt.f32.f16 {s}, {s};", .{ hs[0], addr, f_kd, hs[0] });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_t, addr, rd_qoff });
            for (0..dims / 2) |w| {
                try b.linef("ld.global.u16 {s}, [{s}+{d}];", .{ kt[0], rd_t, w * 2 });
                try b.linef("shl.b32 {s}, {s}, 24; shr.s32 {s}, {s}, 24; cvt.rn.f32.s32 {s}, {s}; mul.f32 {s}, {s}, {s};", .{ kt[1], kt[0], kt[1], kt[1], dst[w * 2], kt[1], dst[w * 2], dst[w * 2], f_kd });
                try b.linef("shl.b32 {s}, {s}, 16; shr.s32 {s}, {s}, 24; cvt.rn.f32.s32 {s}, {s}; mul.f32 {s}, {s}, {s};", .{ kt[1], kt[0], kt[1], kt[1], dst[w * 2 + 1], kt[1], dst[w * 2 + 1], dst[w * 2 + 1], f_kd });
            }
        },
    }
}

/// Flash-decoding pass 1 for PREFILL, with the KV fragment shared across the whole
/// query-head group: one warp per (query, **kv head**, split) instead of per
/// (query, **head**, split), looping the group's `group = heads/kv_heads` q heads
/// inside. Entry `attn_split_g`. Emits the same scratch layout as `attn_split`, so
/// `attn_merge` is unchanged.
///
/// ⚠️ **The win is L2 traffic, not arithmetic, and `attn_split` is ALREADY exactly
/// causal** — each warp there uses `kv_len = kv_len0 + t`, so no masked work is
/// computed and thrown away. What it does do is read the same KV head once per
/// *query head*: with 24 heads over 4 kv heads and 256 queries that is a 1536x
/// re-read of K and V. Measured on Bonsai prefill: 6.29 GB of L2→SM traffic per
/// attention call, 503 GB over a 1155-token prefill, ~252 ms at the 3090's ~2 TB/s
/// L2 — against 217 ms measured for the whole `attn` bucket. Sharing the fragment
/// across the group divides that by `group`.
///
/// The instruction count per warp rises by ~`group` (each head still needs its own
/// butterfly reduction and softmax update) while the warp count falls by `group`, so
/// total instructions are unchanged — and they were ~27 ms against a 252 ms traffic
/// bound, i.e. 9x of headroom to spend.
///
/// ⚠️ **Bit-identical to `attn_split` per head**: same j order, same dot order, same
/// butterfly, same `ex2.approx` softmax sequence. Prefill attention feeds the KV
/// cache that decode then reads, and this model is validated token-identical, so a
/// reassociation here would be a behaviour change.
///
/// Narrow on purpose — hd ∈ {128, 256}, full causal (no sliding window, no ring, no
/// bidirectional). All three KV formats are supported; the per-format fragment decode
/// mirrors `elt.emitKVLoad` instruction for instruction, so each variant stays
/// bit-identical to the `attn_split*` kernel it replaces. Anything else keeps the
/// general kernel; see `Backend.attnSplitGroupOk`.
/// b0=q[seq_q][heads][hd], b1=k[seq_kv][kv_heads][hd], b2=v, b3=scratch.
/// u0=kv_len0, u1=heads, u2=kv_heads, u3=hd(=128), u4=nsplit, u5=seq_q, f0=scale.
pub fn buildAttnSplitGroup(alloc: std.mem.Allocator, hd: usize, group: usize, kvf: elt.KvFmt) ![:0]u8 {
    std.debug.assert(group >= 1 and group <= 16);
    std.debug.assert(hd == 128 or hd == 256);
    const dims = hd / 32; // head dims per lane
    const dshl = std.math.log2_int(usize, dims);
    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    // Per-head state, unrolled over the group.
    const q4 = try b.regs(.f32, group * dims);
    const fm = try b.regs(.f32, group);
    const fd = try b.regs(.f32, group);
    const ac = try b.regs(.f32, group * dims);
    const k4 = try b.regs(.f32, dims);
    const v4 = try b.regs(.f32, dims);
    const ft = try b.regs(.f32, 6); // dot, shfl temp, m2, corr, p, scratch
    const f_scale = try b.reg(.f32);
    // f16 widening / q8_0 dequant temporaries (unused for f32 KV).
    const kt = try b.regs(.b32, 4);
    const hs = if (kvf == .f32) &[_][]const u8{} else try b.regs(.b16, 2);
    const f_kd = try b.reg(.f32); // q8_0 block scale

    const r = try b.regs(.b32, 20);
    const rd = try b.regs(.b64, 12);
    const pr = try b.regs(.pred, 4);

    try b.linef("mov.u32 {s}, %ctaid.x; mov.u32 {s}, %ntid.x; mov.u32 {s}, %tid.x;", .{ r[0], r[1], r[2] });
    try b.linef("mad.lo.s32 {s}, {s}, {s}, {s};", .{ r[3], r[0], r[1], r[2] });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r[4], r[3] }); // global warp
    try b.linef("and.b32 {s}, {s}, 31;", .{ r[5], r[3] }); // lane
    try b.linef("ld.param.u32 {s}, [p_kvlen0];", .{r[6]});
    try b.linef("ld.param.u32 {s}, [p_heads];", .{r[7]});
    try b.linef("ld.param.u32 {s}, [p_kvheads];", .{r[8]});
    try b.linef("ld.param.u32 {s}, [p_nsplit];", .{r[9]});
    try b.linef("ld.param.u32 {s}, [p_seqq];", .{r[10]});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("mul.lo.s32 {s}, {s}, {s};", .{ r[11], r[8], r[9] }); // warps per query
    try b.linef("mul.lo.s32 {s}, {s}, {s};", .{ r[12], r[11], r[10] });
    try b.linef("setp.ge.u32 {s}, {s}, {s}; @{s} bra END;", .{ pr[0], r[4], r[12], pr[0] });
    try b.linef("ld.param.u64 {s}, [p_q];", .{rd[0]});
    try b.linef("ld.param.u64 {s}, [p_k];", .{rd[1]});
    try b.linef("ld.param.u64 {s}, [p_v];", .{rd[2]});
    try b.linef("ld.param.u64 {s}, [p_sc];", .{rd[3]});
    for (0..4) |i| try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd[i], rd[i] });
    try b.linef("div.u32 {s}, {s}, {s};", .{ r[13], r[4], r[11] }); // t
    try b.linef("rem.u32 {s}, {s}, {s};", .{ r[14], r[4], r[11] });
    try b.linef("div.u32 {s}, {s}, {s};", .{ r[15], r[14], r[9] }); // kv head
    try b.linef("rem.u32 {s}, {s}, {s};", .{ r[16], r[14], r[9] }); // split
    try b.linef("add.u32 {s}, {s}, {s};", .{ r[6], r[6], r[13] }); // causal kv_len = kv_len0 + t
    // Split the causal range evenly: chunk = ceil(kv_len / nsplit).
    try b.linef("add.u32 {s}, {s}, {s}; add.u32 {s}, {s}, -1; div.u32 {s}, {s}, {s};", .{ r[17], r[6], r[9], r[17], r[17], r[17], r[17], r[9] });
    try b.linef("mul.lo.s32 {s}, {s}, {s};", .{ r[18], r[16], r[17] }); // kv0
    try b.linef("add.u32 {s}, {s}, {s}; min.u32 {s}, {s}, {s};", .{ r[19], r[18], r[17], r[19], r[19], r[6] }); // kv1
    try b.linef("shl.b32 {s}, {s}, {d};", .{ r[1], r[5], dshl }); // lane*dims (dim offset)
    if (kvf == .q8_0) {
        // Rows are block-aligned and the fragment is dims-aligned with dims | 32, so
        // a lane's fragment never straddles a 34-byte block: elem%32 == (lane*dims)%32,
        // and the quants start 2 bytes past the block's f16 scale.
        try b.linef("and.b32 {s}, {s}, 31; add.u32 {s}, {s}, 2; cvt.u64.u32 {s}, {s};", .{ kt[0], r[1], kt[0], kt[0], rd[8], kt[0] });
    }

    // q fragments: head h = kv_head*group + g, at (t*heads + h)*hd + lane*4.
    try b.linef("mul.lo.s32 {s}, {s}, {d};", .{ r[0], r[15], group });
    for (0..group) |g| {
        try b.linef("add.u32 {s}, {s}, {d};", .{ r[2], r[0], g }); // h
        try b.linef("mad.lo.s32 {s}, {s}, {s}, {s}; mul.lo.s32 {s}, {s}, {d}; add.u32 {s}, {s}, {s};", .{ r[3], r[13], r[7], r[2], r[3], r[3], hd, r[3], r[3], r[1] });
        try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd[4], r[3], rd[5], rd[0], rd[4] });
        for (0..dims / 4) |L| try b.linef("ld.global.v4.f32 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ q4[g * dims + L * 4], q4[g * dims + L * 4 + 1], q4[g * dims + L * 4 + 2], q4[g * dims + L * 4 + 3], rd[5], L * 16 });
        try b.linef("mov.f32 {s}, 0fFF800000; mov.f32 {s}, 0f00000000;", .{ fm[g], fd[g] });
        for (0..dims) |i| try b.linef("mov.f32 {s}, 0f00000000;", .{ac[g * dims + i]});
    }

    const lp = try b.newLabel("j");
    const done = try b.newLabel("jd");
    try b.label(lp);
    try b.linef("setp.ge.u32 {s}, {s}, {s}; @{s} bra {s};", .{ pr[1], r[18], r[19], pr[1], done });
    // The shared fragment: k and v at ((j*kv_heads + kvh)*hd + lane*4), read ONCE
    // for the whole group — this is the entire point of the kernel.
    try b.linef("mad.lo.s32 {s}, {s}, {s}, {s}; mul.lo.s32 {s}, {s}, {d}; add.u32 {s}, {s}, {s};", .{ r[2], r[18], r[8], r[15], r[2], r[2], hd, r[2], r[2], r[1] });
    // Byte offset of the lane's fragment, per KV format. q8_0 addresses the row's
    // 34-byte block; f16/f32 are a plain element stride.
    switch (kvf) {
        .f32 => try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd[4], r[2] }),
        .f16 => try b.linef("mul.wide.u32 {s}, {s}, 2;", .{ rd[4], r[2] }),
        .q8_0 => try b.linef("shr.u32 {s}, {s}, 5; mul.wide.u32 {s}, {s}, 34;", .{ kt[1], r[2], rd[4], kt[1] }),
    }
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd[5], rd[1], rd[4] }); // K
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd[6], rd[2], rd[4] }); // V
    try emitKvFrag(&b, kvf, dims, k4, rd[5], kt, hs, f_kd, rd[9], rd[8]);
    try emitKvFrag(&b, kvf, dims, v4, rd[6], kt, hs, f_kd, rd[9], rd[8]);
    for (0..group) |g| {
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[0], q4[g * dims], k4[0] });
        for (1..dims) |i| try b.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ ft[0], q4[g * dims + i], k4[i], ft[0] });
        // Butterfly all-reduce, identical to attn_split's.
        for ([_]u32{ 16, 8, 4, 2, 1 }) |d| {
            try b.linef("mov.b32 {s}, {s}; shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff; mov.b32 {s}, {s}; add.f32 {s}, {s}, {s};", .{ r[3], ft[0], r[2], r[3], d, ft[1], r[2], ft[0], ft[0], ft[1] });
        }
        try b.linef("mul.f32 {s}, {s}, {s};", .{ ft[0], ft[0], f_scale });
        try b.linef("max.f32 {s}, {s}, {s};", .{ ft[2], fm[g], ft[0] });
        try b.linef("sub.f32 {s}, {s}, {s}; mul.f32 {s}, {s}, 0f3FB8AA3B; ex2.approx.f32 {s}, {s};", .{ ft[3], fm[g], ft[2], ft[3], ft[3], ft[3], ft[3] });
        try b.linef("sub.f32 {s}, {s}, {s}; mul.f32 {s}, {s}, 0f3FB8AA3B; ex2.approx.f32 {s}, {s};", .{ ft[4], ft[0], ft[2], ft[4], ft[4], ft[4], ft[4] });
        try b.linef("mul.f32 {s}, {s}, {s}; add.f32 {s}, {s}, {s};", .{ fd[g], fd[g], ft[3], fd[g], fd[g], ft[4] });
        try b.linef("mov.f32 {s}, {s};", .{ fm[g], ft[2] });
        for (0..dims) |i| try b.linef("mul.f32 {s}, {s}, {s}; fma.rn.f32 {s}, {s}, {s}, {s};", .{ ac[g * dims + i], ac[g * dims + i], ft[3], ac[g * dims + i], ft[4], v4[i], ac[g * dims + i] });
    }
    try b.linef("add.u32 {s}, {s}, 1; bra {s};", .{ r[18], r[18], lp });
    try b.label(done);

    // Scratch row ((t*heads + h)*nsplit + split)*(hd+4) — attn_split's own layout.
    try b.linef("mul.lo.s32 {s}, {s}, {d};", .{ r[0], r[15], group });
    for (0..group) |g| {
        try b.linef("add.u32 {s}, {s}, {d};", .{ r[2], r[0], g });
        try b.linef("mad.lo.s32 {s}, {s}, {s}, {s}; mad.lo.s32 {s}, {s}, {s}, {s};", .{ r[3], r[13], r[7], r[2], r[3], r[3], r[9], r[16] });
        try b.linef("mul.lo.s32 {s}, {s}, {d};", .{ r[3], r[3], hd + 4 });
        try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd[4], r[3], rd[5], rd[3], rd[4] });
        const sk = try b.newLabel("wracc");
        try b.linef("setp.ne.u32 {s}, {s}, 0; @{s} bra {s};", .{ pr[2], r[5], pr[2], sk });
        try b.linef("st.global.f32 [{s}], {s}; st.global.f32 [{s}+4], {s};", .{ rd[5], fm[g], rd[5], fd[g] });
        try b.label(sk);
        try b.linef("shl.b32 {s}, {s}, {d}; add.u32 {s}, {s}, 16;", .{ r[2], r[5], dshl + 2, r[2], r[2] });
        try b.linef("cvt.u64.u32 {s}, {s}; add.s64 {s}, {s}, {s};", .{ rd[6], r[2], rd[7], rd[5], rd[6] });
        for (0..dims / 4) |L| try b.linef("st.global.v4.f32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ rd[7], L * 16, ac[g * dims + L * 4], ac[g * dims + L * 4 + 1], ac[g * dims + L * 4 + 2], ac[g * dims + L * 4 + 3] });
    }
    try b.label("END");
    _ = pr[3];

    return b.build(
        switch (kvf) {
            .f32 => "attn_split_g",
            .f16 => "attn_split_g_f16",
            .q8_0 => "attn_split_g_q8",
        },
        "    .param .u64 p_q,\n    .param .u64 p_k,\n    .param .u64 p_v,\n    .param .u64 p_sc,\n    .param .u32 p_kvlen0,\n    .param .u32 p_heads,\n    .param .u32 p_kvheads,\n    .param .u32 p_hd,\n    .param .u32 p_nsplit,\n    .param .u32 p_seqq,\n    .param .f32 p_scale",
        "",
    );
}

/// Gated-delta-net recurrence over a WHOLE prefill chunk with the state held in
/// REGISTERS: one block per v-head, thread `j` owning column `j` of the `[d][d]`
/// state for the entire chunk. Entry `gdn_delta_chunk`.
///
/// This is `elt.gdn_delta_step_ptx`'s exact arithmetic, in the exact same order,
/// restructured so the state is read and written ONCE per chunk instead of once
/// per token. That distinction is the whole point:
///
///   - The per-token kernel streams the state through global memory every token —
///     `[heads][d][d]` f32 read AND written, 3.1 MB each way per layer per token,
///     ~350 GB over a 1157-token prefill. Its own doc comment says it is
///     "load-latency bound", and it measured ~33% of prefill.
///   - Registers hold `d` floats per thread (128 for qwen3.5), so the state never
///     leaves the SM between tokens. Only the staged k/q vectors and the small
///     per-token gates/v/o touch memory.
///
/// ⚠️ **Bit-identical to the per-token kernel, and that is a requirement, not a
/// nicety.** The recurrence is reassociation-sensitive, decode can only ever run
/// the per-token form, and this model is validated token-identical to llama.cpp —
/// so prefill and decode must agree exactly or a re-prefill (regenerate, variant
/// rollback, suspend/resume) would diverge from the incremental path. Hence:
/// ascending `i` in both passes, `mul` then `fma` in pass 1, `fma` then `fma` in
/// pass 2, and `(v - m) * beta` in that order. A chunked/parallel DeltaNet
/// formulation would NOT have this property — it is the reason this restructuring
/// was chosen over one.
///
/// ⚠️ The `i` walk is FULLY unrolled because PTX cannot index a register file;
/// that is what forces this kernel to be generated rather than written. `d` is
/// baked in, so the caller must rebuild when it changes.
///
/// b0=S [heads][d][d], b1=conv_out [n][channels] (`[q | k | v]` per token),
/// b2=gates [n][2*heads] (`[decay | beta]` per token), b3=o [n][heads*d].
/// u0=heads, u1=d, u2=k_heads, u3=n, u4=channels. f0=readout scale.
/// Launch: grid=(heads), block=(d).
pub fn buildGdnDeltaChunk(alloc: std.mem.Allocator, d: usize) ![:0]u8 {
    std.debug.assert(d % 32 == 0 and d >= 32);
    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    // The resident state column. Everything else is small working state.
    const s = try b.regs(.f32, d);
    const f_scale = try b.reg(.f32);
    const f_decay = try b.reg(.f32);
    const f_beta = try b.reg(.f32);
    const f_v = try b.reg(.f32);
    const f_m = try b.reg(.f32);
    const f_o = try b.reg(.f32);
    const f_dj = try b.reg(.f32);
    const f_t = try b.reg(.f32);

    const r_h = try b.reg(.b32);
    const r_j = try b.reg(.b32);
    const r_heads = try b.reg(.b32);
    const r_kheads = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const r_ch = try b.reg(.b32);
    const r_kh = try b.reg(.b32);
    const r_t = try b.regs(.b32, 6);
    const r_sk = try b.reg(.b32);
    const r_sq = try b.reg(.b32);
    const r_tok = try b.reg(.b32);

    const rd_s = try b.reg(.b64);
    const rd_cv = try b.reg(.b64);
    const rd_g = try b.reg(.b64);
    const rd_o = try b.reg(.b64);
    const rd_scol = try b.reg(.b64); // this thread's state column base
    const rd_t = try b.regs(.b64, 4);

    try b.linef("ld.param.u64 {s}, [p_s];", .{rd_s});
    try b.linef("ld.param.u64 {s}, [p_cv];", .{rd_cv});
    try b.linef("ld.param.u64 {s}, [p_g];", .{rd_g});
    try b.linef("ld.param.u64 {s}, [p_o];", .{rd_o});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_s, rd_s });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_cv, rd_cv });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_g, rd_g });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_o, rd_o });
    try b.linef("ld.param.u32 {s}, [p_heads];", .{r_heads});
    try b.linef("ld.param.u32 {s}, [p_kheads];", .{r_kheads});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.u32 {s}, [p_ch];", .{r_ch});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_h});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_j});
    try b.linef("rem.u32 {s}, {s}, {s};", .{ r_kh, r_h, r_kheads }); // kh = h % k_heads
    try b.linef("mov.u32 {s}, sk;", .{r_sk});
    try b.linef("mov.u32 {s}, sq;", .{r_sq});

    // This thread's state column: S + (h*d*d + j)*4, row stride d*4.
    try b.linef("mad.lo.s32 {s}, {s}, {d}, {s};", .{ r_t[0], r_h, d * d, r_j });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[0], r_t[0], rd_scol, rd_s, rd_t[0] });

    // Load the column once for the whole chunk.
    try b.comment("state column -> registers (once per chunk, not once per token)", .{});
    for (0..d) |i| try b.linef("ld.global.f32 {s}, [{s}+{d}];", .{ s[i], rd_scol, i * d * 4 });

    // qk_span = k_heads*d; k/q source offsets within a token's conv_out row.
    const r_qk = try b.reg(.b32);
    try b.linef("mul.lo.s32 {s}, {s}, {d};", .{ r_qk, r_kheads, d });
    try b.linef("mad.lo.s32 {s}, {s}, {d}, {s};", .{ r_t[1], r_kh, d, r_j }); // kh*d + j

    const lp = try b.newLabel("tok");
    const done = try b.newLabel("tokdone");
    const pr = try b.regs(.pred, 2);
    try b.linef("mov.u32 {s}, 0;", .{r_tok});
    try b.label(lp);
    try b.linef("setp.ge.u32 {s}, {s}, {s}; @{s} bra {s};", .{ pr[0], r_tok, r_n, pr[0], done });

    // -- stage k and q for this token (thread j takes element j) --------------
    try b.linef("mul.lo.s32 {s}, {s}, {s};", .{ r_t[2], r_tok, r_ch }); // token base in conv_out
    try b.linef("add.u32 {s}, {s}, {s}; add.u32 {s}, {s}, {s};", .{ r_t[3], r_t[2], r_qk, r_t[3], r_t[3], r_t[1] });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[1], r_t[3], rd_t[1], rd_cv, rd_t[1] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ f_t, rd_t[1] }); // k
    try b.linef("shl.b32 {s}, {s}, 2; add.u32 {s}, {s}, {s};", .{ r_t[4], r_j, r_t[4], r_t[4], r_sk });
    try b.linef("st.shared.f32 [{s}], {s};", .{ r_t[4], f_t });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_t[3], r_t[2], r_t[1] });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[1], r_t[3], rd_t[1], rd_cv, rd_t[1] });
    try b.linef("ld.global.f32 {s}, [{s}]; mul.f32 {s}, {s}, {s};", .{ f_t, rd_t[1], f_t, f_t, f_scale }); // q*scale
    try b.linef("shl.b32 {s}, {s}, 2; add.u32 {s}, {s}, {s};", .{ r_t[5], r_j, r_t[5], r_t[5], r_sq });
    try b.linef("st.shared.f32 [{s}], {s};", .{ r_t[5], f_t });
    try b.line("bar.sync 0;");

    // -- per-token scalars: decay, beta, v_j ---------------------------------
    try b.linef("shl.b32 {s}, {s}, 1; mul.lo.s32 {s}, {s}, {s};", .{ r_t[3], r_heads, r_t[3], r_t[3], r_tok });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_t[3], r_t[3], r_h });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[2], r_t[3], rd_t[2], rd_g, rd_t[2] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ f_decay, rd_t[2] });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[3], r_heads, rd_t[3], rd_t[2], rd_t[3] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ f_beta, rd_t[3] });
    // v_j at conv_out[token][2*qk_span + h*d + j]
    try b.linef("shl.b32 {s}, {s}, 1; add.u32 {s}, {s}, {s};", .{ r_t[3], r_qk, r_t[3], r_t[3], r_t[2] });
    try b.linef("mad.lo.s32 {s}, {s}, {d}, {s};", .{ r_t[4], r_h, d, r_j });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_t[3], r_t[3], r_t[4] });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[1], r_t[3], rd_t[1], rd_cv, rd_t[1] });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ f_v, rd_t[1] });

    // -- pass 1: decay the column, accumulate the memory readout m -----------
    // Ascending i, `mul` then `fma`, matching gdn_delta_step exactly.
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_m});
    for (0..d) |i| {
        try b.linef("mul.f32 {s}, {s}, {s};", .{ s[i], s[i], f_decay });
        try b.linef("ld.shared.f32 {s}, [{s}+{d}]; fma.rn.f32 {s}, {s}, {s}, {s};", .{ f_t, r_sk, i * 4, f_m, s[i], f_t, f_m });
    }
    try b.linef("sub.f32 {s}, {s}, {s}; mul.f32 {s}, {s}, {s};", .{ f_dj, f_v, f_m, f_dj, f_dj, f_beta });

    // -- pass 2: rank-1 update, accumulate the output readout ---------------
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_o});
    for (0..d) |i| {
        try b.linef("ld.shared.f32 {s}, [{s}+{d}]; fma.rn.f32 {s}, {s}, {s}, {s};", .{ f_t, r_sk, i * 4, s[i], f_t, f_dj, s[i] });
        try b.linef("ld.shared.f32 {s}, [{s}+{d}]; fma.rn.f32 {s}, {s}, {s}, {s};", .{ f_t, r_sq, i * 4, f_o, s[i], f_t, f_o });
    }

    // -- o[token][h*d + j] ---------------------------------------------------
    try b.linef("mul.lo.s32 {s}, {s}, {d};", .{ r_t[3], r_heads, d });
    try b.linef("mad.lo.s32 {s}, {s}, {s}, {s};", .{ r_t[3], r_tok, r_t[3], r_t[4] });
    try b.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ rd_t[1], r_t[3], rd_t[1], rd_o, rd_t[1] });
    try b.linef("st.global.f32 [{s}], {s};", .{ rd_t[1], f_o });

    // The next token restages sk/sq, which every thread reads in full.
    try b.line("bar.sync 0;");
    try b.linef("add.u32 {s}, {s}, 1; bra {s};", .{ r_tok, r_tok, lp });
    try b.label(done);

    try b.comment("state column -> memory (once per chunk)", .{});
    for (0..d) |i| try b.linef("st.global.f32 [{s}+{d}], {s};", .{ rd_scol, i * d * 4, s[i] });

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 4 .b8 sk[{d}];\n    .shared .align 4 .b8 sq[{d}];", .{ d * 4, d * 4 });
    defer alloc.free(shared_decl);
    return b.build(
        "gdn_delta_chunk",
        "    .param .u64 p_s,\n    .param .u64 p_cv,\n    .param .u64 p_g,\n    .param .u64 p_o,\n    .param .u32 p_heads,\n    .param .u32 p_kheads,\n    .param .u32 p_n,\n    .param .u32 p_ch,\n    .param .f32 p_scale",
        shared_decl,
    );
}

/// Emit the sign-bit -> ±1-int8 unpack: 16 bits from the low half of `q` become
/// four u32 of four s8 each, in element order (v[0] = elements 0..3). Uses `l1`
/// = 0x11100100 and `l2` = 0x000001FF as `prmt` byte LUTs and `t` as a scratch.
///
/// This is `elt.zig`'s `q1Dp4aHalf` unpack without the dp4a — see there for what
/// the two LUTs do and, in particular, why the `0x33333333` mask on the selector
/// is load-bearing (`prmt` reads bit 3 of each nibble as a sign-replicate flag,
/// and every byte of `l1` has its msb clear).
fn emitQ1Unpack(
    b: *ptx.Builder,
    q: []const u8,
    l1: []const u8,
    l2: []const u8,
    t: []const u8,
    n: []const []const u8, // 2 scratch regs
    s: []const []const u8, // 4 scratch regs
    v: []const []const u8, // 4 outputs
) !void {
    try b.linef("and.b32 {s}, {s}, 0x33333333;", .{ t, q });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ n[0], l1, l1, t });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ t, q });
    try b.linef("and.b32 {s}, {s}, 0x33333333;", .{ t, t });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ n[1], l1, l1, t });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[0], l2, l2, n[0] });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[1], l2, l2, n[1] });
    try b.linef("shr.u32 {s}, {s}, 16;", .{ t, n[0] });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[2], l2, l2, t });
    try b.linef("shr.u32 {s}, {s}, 16;", .{ t, n[1] });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[3], l2, l2, t });
    try b.linef("mov.u32 {s}, 0x5410;", .{t});
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[0], s[0], s[1], t });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[2], s[2], s[3], t });
    try b.linef("mov.u32 {s}, 0x7632;", .{t});
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[1], s[0], s[1], t });
    try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[3], s[2], s[3], t });
}

/// Decode 16 q2_0 codes (a u32, packed 2 bits each) into 16 SIGNED bytes, as four
/// u32 in element order — the q2_0 twin of `emitQ1Unpack`.
///
/// `l2` holds the symbol table `{-1, 0, +1, +2}` as bytes (0x020100FF) and `l1` the
/// selector mask 0x33333333, so one `prmt` maps four codes straight to their
/// symbols. Per 8-code half:
///
///   sel_even = q & 0x33333333        -> nibbles c0 c2 c4 c6
///   sel_odd  = (q >> 2) & 0x33333333 -> nibbles c1 c3 c5 c7
///   qe/qo    = prmt(l2, l2, sel)     -> their symbol bytes
///   prmt(qe, qo, 0x5140 / 0x7362)    -> back to element order
///
/// ⚠️ **`prmt` reads only the LOW 16 BITS of the selector**, which is why a u32 of
/// 16 codes has to be split into two halves rather than masked in one go — the top
/// four nibbles would simply be ignored.
///
/// ⚠️ **The 0x33333333 mask is load-bearing**: bit 3 of a `prmt` selector nibble
/// means SIGN-REPLICATE, and an unmasked nibble carries the neighbouring code in
/// its high bits, so any code >= 2 would silently corrupt its neighbour's symbol.
/// Same trap `emitQ1Unpack` masks for. (CUDA's `__byte_perm` intrinsic ignores
/// bit 3, which is why llama.cpp's C++ needs no mask and this does.)
///
/// Symbols come out signed, so the caller needs no activation `sum(q)` min term —
/// the property that makes q2_0's MMQ as simple as q1_0's.
fn emitQ2Unpack(
    b: *ptx.Builder,
    q: []const u8,
    l1: []const u8, // 0x33333333
    l2: []const u8, // 0x020100FF
    t: []const u8,
    s: []const []const u8, // 4 scratch regs
    v: []const []const u8, // 4 outputs
) !void {
    for (0..2) |half| {
        // half 0 works on q[15:0], half 1 on q[31:16].
        if (half == 0) {
            try b.linef("and.b32 {s}, {s}, {s};", .{ s[0], q, l1 });
            try b.linef("shr.u32 {s}, {s}, 2;", .{ t, q });
        } else {
            try b.linef("shr.u32 {s}, {s}, 16;", .{ t, q });
            try b.linef("and.b32 {s}, {s}, {s};", .{ s[0], t, l1 });
            try b.linef("shr.u32 {s}, {s}, 2;", .{ t, t });
        }
        try b.linef("and.b32 {s}, {s}, {s};", .{ s[1], t, l1 });
        try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[2], l2, l2, s[0] }); // even symbols
        try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ s[3], l2, l2, s[1] }); // odd symbols
        try b.linef("mov.u32 {s}, 0x5140;", .{t});
        try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[half * 2 + 0], s[2], s[3], t });
        try b.linef("mov.u32 {s}, 0x7362;", .{t});
        try b.linef("prmt.b32 {s}, {s}, {s}, {s};", .{ v[half * 2 + 1], s[2], s[3], t });
    }
}

/// q2_0 MMQ — the q1_0 kernel's twin, and deliberately a near-copy of it.
///
/// The two formats line up on exactly the two properties that make
/// `buildMmqPipeQ1_0` the simplest of the MMQ family, so the whole pipe
/// (128x128 / 2x2 warps / MT=4 / NT=8, cp.async double-buffering, the 80-byte
/// padded row stride that removes the 4-way fragment bank conflict) is reused
/// unchanged:
///
///   1. **No min term.** `emitQ2Unpack` emits SIGNED symbols straight out of a
///      `prmt` table, so like q1_0's `+-d` there is no zero point to cancel and
///      the B-side sums region and half the fold arithmetic never appear.
///   2. **One scale per block, constant across a 64-k slab.** g128's block is 128
///      elements (two slabs) and g64's is 64 (one), so `d` is loaded once per slab
///      either way — where q4_k reloads a pair every 32-k and q6_k every 16.
///
/// What actually differs from q1_0, and it is only addressing plus the unpack:
/// a slab needs **16** code bytes per row rather than 8, the block stride is
/// 34/18 rather than 18, and a block spans `SPB` slabs rather than always 2.
///
/// ⚠️ The 16 bytes are read as eight `u16` and OR'd into four u32 immediately.
/// They must stay u16 because `block + 2 + sub*16` is only 2-byte aligned (the
/// block stride is 2 mod 4, so its parity alternates), and they must be packed
/// rather than held because `mmq_pipe_q1_0` measures at **255 registers with zero
/// spill** — there is no room for four more live values. The high halves reuse the
/// addressing temps, which are dead by then.
///
/// Requires cols % 256 == 0, rows % 128 == 0, and `n` padded to a multiple of 128
/// with the padding zero-filled. Entry `mmq_pipe_q2_0_{g64,g128}`.
pub fn buildMmqPipeQ2_0(alloc: std.mem.Allocator, g: elt.Q2Geom) ![:0]u8 {
    // Slabs (64 k-elements) spanned by one weight block: 2 for g128, 1 for g64.
    const SPB = g.qk / 64;
    const BB = 2 + g.qk / 4; // block bytes: 34 (g128) / 18 (g64)
    const QK_LOG = std.math.log2_int(u32, @intCast(g.qk));
    const BM = 128;
    const BN = 128;
    const kstep = 64;
    const KS = kstep / 32; // 32-k substeps (== q8_1 activation blocks) per slab
    const MT = 4;
    const NT = 8;
    // ⚠️ Shared rows are padded to 80 bytes, not the 64 they hold. The fragment
    // load has lane L read row `base + L/4` at word `ks*8 + L%4`, so with a
    // 64-byte (16-word) stride the 32 lanes touch only 8 of the 32 banks — a
    // **4-way bank conflict on every fragment load**, and there are 32 of them per
    // substep. At a 20-word stride the row term is `20*gid mod 32` =
    // {0,20,8,28,16,4,24,12}, eight distinct multiples of 4, so adding `tf` 0..3
    // covers all 32 banks exactly once: conflict-free.
    //
    // This is the PADDING alternative to an XOR swizzle, and it is the right one
    // here: at kstep=64 a row is only 16 words, so the usual `(row&7)<<4` swizzle
    // would cross into the next row, and the `(row&3)<<4` mask that does fit still
    // leaves 2-way conflicts because gid 4..7 alias gid 0..3. It also beats
    // `ldmatrix`, which on sm_86 is `.b16` only (awkward for an int8 fragment) and
    // would save just ~4% of the instruction stream.
    const stride = 80;
    const ATILE = BM * stride;
    const BTILE = BN * stride;
    const ASC = BM * 4; // one f32 `d` per row (constant across the slab)
    const BSC = BN * KS * 4; // one f32 `da` per column per substep
    const B_OFF = ATILE;
    const ASC_OFF = ATILE + BTILE;
    const BSC_OFF = ASC_OFF + ASC;
    // One buffer's worth of shared. The pipeline holds TWO (see the slab loop).
    const SH_HALF = BSC_OFF + BSC;
    const SH = 2 * SH_HALF;

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4);
    const bf = try b.regs(.b32, NT * 2);
    const ct = try b.regs(.b32, 4); // mma destination (C operand is the zero quad)
    const zq = try b.regs(.b32, 4); // permanently zero
    const rsc = try b.regs(.f32, MT * 2); // per-mi: `d` for rows gid and gid+8
    const csc = try b.regs(.f32, 2); // per-nj: `da` for cols tf*2 and tf*2+1

    const rd_w = try b.reg(.b64);
    const rd_x = try b.reg(.b64);
    const rd_y = try b.reg(.b64);
    const r_rows = try b.reg(.b32);
    const r_cols = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const f_scale = try b.reg(.f32);
    try b.linef("ld.param.u64 {s}, [p_w];", .{rd_w});
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_y];", .{rd_y});
    try b.linef("ld.param.u32 {s}, [p_rows];", .{r_rows});
    try b.linef("ld.param.u32 {s}, [p_cols];", .{r_cols});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_w, rd_w });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_y, rd_y });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const r_nblk = try b.reg(.b32);
    const r_rb = try b.reg(.b32);
    const r_bpt = try b.reg(.b32);
    const r_l1 = try b.reg(.b32);
    const r_l2 = try b.reg(.b32);
    const tm = try b.regs(.b32, 8);
    const rdt = try b.regs(.b64, 4);

    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });
    try b.linef("shr.u32 {s}, {s}, {d};", .{ r_nblk, r_cols, QK_LOG }); // q2_0 blocks per row
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_rb, r_nblk, BB }); // row bytes
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_bpt, r_cols }); // q8 blocks per token
    try b.linef("mov.u32 {s}, 0x33333333;", .{r_l1}); // prmt selector mask
    try b.linef("mov.u32 {s}, 0x020100FF;", .{r_l2}); // symbols {-1,0,1,2}

    // Activation qs region (no sums region: q1_0 needs no min term).
    const rd_qs = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[0], r_n, r_bpt });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_qs, tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qs, rd_x, rd_qs });

    // Staging identities: one thread per A row and per B column (BM=BN=128=nthreads).
    const rd_wrow = try b.reg(.b64);
    const rd_bcol = try b.reg(.b64);
    const r_gcol = try b.reg(.b32);
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], r_row0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wrow, tm[0], r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wrow, rd_w, rd_wrow });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_gcol, r_col0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_bcol, r_gcol, r_cols });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bcol, rd_qs, rd_bcol });

    // Shared destinations for this thread's staged row/column.
    const r_adst = try b.reg(.b32);
    const r_bdst = try b.reg(.b32);
    const r_ascd = try b.reg(.b32);
    const r_bscd = try b.reg(.b32);
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_adst, r_t, stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_adst, r_adst, r_smem });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bdst, r_t, stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bdst, r_bdst, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bdst, r_bdst, B_OFF });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_ascd, r_t });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascd, r_ascd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascd, r_ascd, ASC_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscd, r_t, KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscd, r_bscd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscd, r_bscd, BSC_OFF });

    // Fragment-load bases (pipe's use_ldmatrix=false layout).
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl, tm[0], stride });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl, tm[0], stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, B_OFF });

    // Scale-read bases: row (wm*64 + mi*16 + gid), col (wn*64 + nj*8 + tf*2).
    const r_ascr = try b.reg(.b32);
    const r_bscr = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_ascr, tm[0] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascr, r_ascr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascr, r_ascr, ASC_OFF });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], tm[1] });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscr, tm[0], KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscr, r_bscr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscr, r_bscr, BSC_OFF });

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});
    for (zq) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    // ---- slab loop: cp.async double-buffered software pipeline -------------
    //
    // ⚠️ The single-buffered version this replaces measured 19% of the kernel in
    // EXPOSED global-load latency: staging wrote shared, `bar.sync`d, computed,
    // `bar.sync`d again, so nothing overlapped the ~500-cycle loads. Removing the
    // staging entirely (garbage output, timing only) gave 80.4 TOPS against 65.0
    // with it — that measurement is what justified this rewrite, and the fold is
    // NOT the limiter (cutting it 4x was worth only +10%).
    //
    // Three things make it fit:
    //   - **B goes global -> shared with `cp.async`**, so its 16 staging registers
    //     disappear. At 254 registers there was no headroom for prefetch state
    //     otherwise; cp.async pays for itself before any latency is hidden.
    //   - **A cannot use cp.async** (its bytes are transformed by the prmt decode),
    //     so its raw u16s are prefetched into REGISTERS a slab ahead and decoded
    //     into shared after the compute that hides their latency.
    //   - **The slab loop is unrolled 2x so the buffer parity is COMPILE-TIME**, and
    //     every shared offset stays an immediate (`buf * SH_HALF`). `cols % 256 == 0`
    //     makes `cols/kstep` a multiple of 4, so nslab is always even.
    //
    // One `bar.sync` per slab instead of two, and the prefetch index is CLAMPED to
    // the last slab rather than branched on: the extra fetch is garbage that is
    // never consumed, and clamping keeps the address in bounds.
    const r_i = try b.reg(.b32);
    const r_nslab = try b.reg(.b32);
    const r_last = try b.reg(.b32);
    const r_pf = try b.reg(.b32); // clamped prefetch slab index
    const pr = try b.regs(.pred, 2);
    const ar = try b.regs(.b32, 4); // A's prefetched raw sign words
    const h16 = try b.regs(.b16, 2);
    const f_ad = try b.reg(.f32); // A's prefetched d
    const f_bd = try b.regs(.f32, 2); // B's prefetched scales
    const fs = try b.regs(.f32, 4);
    const us = try b.regs(.b32, 4);
    const uv = try b.regs(.b32, 4);

    try b.linef("shr.u32 {s}, {s}, 6;", .{ r_nslab, r_cols }); // cols/kstep
    try b.linef("add.u32 {s}, {s}, -1;", .{ r_last, r_nslab });

    // Issue the global side of slab `r_pf` into buffer `buf`: A's raw words into
    // registers, B straight into shared with cp.async.
    const prefetch = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            // A: block = slab/SPB, sub = slab%SPB; qs is only 2-byte aligned.
            if (args.spb == 2) {
                try bb.linef("shr.u32 {s}, {s}, 1;", .{ args.tm[0], args.r_pf });
                try bb.linef("and.b32 {s}, {s}, 1;", .{ args.tm[1], args.r_pf });
            } else {
                try bb.linef("mov.u32 {s}, {s};", .{ args.tm[0], args.r_pf });
                try bb.linef("mov.u32 {s}, 0;", .{args.tm[1]});
            }
            try bb.linef("mul.lo.u32 {s}, {s}, {d};", .{ args.tm[2], args.tm[0], args.bb });
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[2] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[1], args.rd_wrow, args.rdt[0] });
            try bb.linef("ld.global.b16 {s}, [{s}];", .{ args.h16[0], args.rdt[1] }); // d
            try bb.linef("shl.b32 {s}, {s}, 4; add.u32 {s}, {s}, 2;", .{ args.tm[3], args.tm[1], args.tm[3], args.tm[3] });
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[3] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[2], args.rdt[1], args.rdt[0] });
            // 16 code bytes per row per slab (q1_0 needs 8). ⚠️ They must stay u16 —
            // a q2_0 block is 34/18 bytes, so `block + 2 + sub*16` is only 2-byte
            // aligned and its parity alternates with the block index.
            //
            // ⚠️ The pairs are OR'd into 4 u32 immediately rather than held as 8
            // live u16: `mmq_pipe_q1_0` measures at 255 registers with zero spill,
            // so there is no headroom for four more. The high halves land in the
            // addressing temps, which are dead by this point. A u32 of 16 codes is
            // also exactly what `emitQ2Unpack` consumes.
            for (0..4) |j| try bb.linef("ld.global.u16 {s}, [{s}+{d}];", .{ args.ar[j], args.rdt[2], j * 4 });
            for (0..4) |j| try bb.linef("ld.global.u16 {s}, [{s}+{d}];", .{ args.tm[4 + j], args.rdt[2], j * 4 + 2 });
            for (0..4) |j| try bb.linef("shl.b32 {s}, {s}, 16; or.b32 {s}, {s}, {s};", .{ args.tm[4 + j], args.tm[4 + j], args.ar[j], args.ar[j], args.tm[4 + j] });
            try bb.linef("cvt.f32.f16 {s}, {s};", .{ args.f_ad, args.h16[0] });
            // B: 64 bytes per column, global -> shared, no registers.
            try bb.linef("shl.b32 {s}, {s}, 6;", .{ args.tm[0], args.r_pf }); // slab*kstep
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[0] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[1], args.rd_bcol, args.rdt[0] });
            for (0..4) |q4| try bb.linef("cp.async.ca.shared.global [{s}+{d}], [{s}+{d}], 16;", .{ args.r_bdst, off + q4 * 16, args.rdt[1], q4 * 16 });
            try bb.line("cp.async.commit_group;");
            // B's per-substep scales (tiny; plain loads, stored at commit).
            try bb.linef("mul.lo.u32 {s}, {s}, {d};", .{ args.tm[0], args.r_pf, args.ks });
            try bb.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ args.tm[1], args.r_gcol, args.r_bpt, args.tm[0] });
            try bb.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ args.rdt[0], args.tm[1], args.rdt[1], args.rd_x, args.rdt[0] });
            try bb.linef("ld.global.f32 {s}, [{s}];", .{ args.f_bd[0], args.rdt[1] });
            try bb.linef("ld.global.f32 {s}, [{s}+4];", .{ args.f_bd[1], args.rdt[1] });
        }
    }.emit;

    // Land the prefetched slab in buffer `buf`: decode A into shared, store both
    // scale sets, then wait for this buffer's cp.async and sync.
    const commit = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            for (0..4) |j| {
                try emitQ2Unpack(bb, args.ar[j], args.r_l1, args.r_l2, args.tm[4], args.us, args.uv);
                try bb.linef("st.shared.v4.u32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ args.r_adst, off + j * 16, args.uv[0], args.uv[1], args.uv[2], args.uv[3] });
            }
            try bb.linef("st.shared.f32 [{s}+{d}], {s};", .{ args.r_ascd, off, args.f_ad });
            try bb.linef("st.shared.v2.f32 [{s}+{d}], {{{s}, {s}}};", .{ args.r_bscd, off, args.f_bd[0], args.f_bd[1] });
            try bb.line("cp.async.wait_group 0;");
            try bb.line("bar.sync 0;");
        }
    }.emit;

    // MT x NT mmas over the buffer's KS substeps, folding d*da per substep.
    const compute = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            var mi0: usize = 0;
            while (mi0 < args.mt) : (mi0 += 1) {
                try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.rsc[mi0 * 2 + 0], args.r_ascr, off + mi0 * 16 * 4 });
                try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.rsc[mi0 * 2 + 1], args.r_ascr, off + mi0 * 16 * 4 + 8 * 4 });
            }
            var ks: usize = 0;
            while (ks < args.ks) : (ks += 1) {
                var mi: usize = 0;
                while (mi < args.mt) : (mi += 1) {
                    const o = off + mi * 16 * args.stride + ks * 32;
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 0], args.r_asl, o });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 1], args.r_asl, o + 8 * args.stride });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 2], args.r_asl, o + 16 });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 3], args.r_asl, o + 8 * args.stride + 16 });
                }
                var nj: usize = 0;
                while (nj < args.nt) : (nj += 1) {
                    const o = off + nj * 8 * args.stride + ks * 32;
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.bf[nj * 2 + 0], args.r_bsl, o });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.bf[nj * 2 + 1], args.r_bsl, o + 16 });
                }
                nj = 0;
                while (nj < args.nt) : (nj += 1) {
                    const co = off + nj * 8 * args.ks * 4 + ks * 4;
                    try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.csc[0], args.r_bscr, co });
                    try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.csc[1], args.r_bscr, co + args.ks * 4 });
                    mi = 0;
                    while (mi < args.mt) : (mi += 1) {
                        try bb.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                            args.ct[0],          args.ct[1],          args.ct[2],          args.ct[3],
                            args.af[mi * 4 + 0], args.af[mi * 4 + 1], args.af[mi * 4 + 2], args.af[mi * 4 + 3],
                            args.bf[nj * 2 + 0], args.bf[nj * 2 + 1],
                            args.zq[0],          args.zq[1],          args.zq[2],          args.zq[3],
                        });
                        const a = args.acc[(mi * args.nt + nj) * 4 ..][0..4];
                        const rows2 = [_]usize{ 0, 0, 1, 1 };
                        const cols2 = [_]usize{ 0, 1, 0, 1 };
                        for (0..4) |ci| {
                            try bb.linef("cvt.rn.f32.s32 {s}, {s};", .{ args.fs[0], args.ct[ci] });
                            try bb.linef("mul.f32 {s}, {s}, {s};", .{ args.fs[1], args.rsc[mi * 2 + rows2[ci]], args.csc[cols2[ci]] });
                            try bb.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ a[ci], args.fs[1], args.fs[0], a[ci] });
                        }
                    }
                }
            }
        }
    }.emit;

    const A = .{
        .sh_half = SH_HALF, .ks = KS,   .kstep = kstep, .stride = stride, .mt = MT, .nt = NT,
        .tm = tm,           .rdt = rdt, .r_pf = r_pf,   .rd_wrow = rd_wrow, .rd_bcol = rd_bcol,
        .h16 = h16,         .ar = ar,   .f_ad = f_ad,   .f_bd = f_bd,   .r_bdst = r_bdst,
        .r_adst = r_adst,   .r_ascd = r_ascd, .r_bscd = r_bscd, .r_gcol = r_gcol, .r_bpt = r_bpt,
        .rd_x = rd_x,       .r_l1 = r_l1, .r_l2 = r_l2, .spb = SPB,     .bb = BB,   .us = us,
        .uv = uv,           .r_ascr = r_ascr, .r_asl = r_asl, .r_bsl = r_bsl, .r_bscr = r_bscr,
        .rsc = rsc,         .csc = csc, .af = af,       .bf = bf,       .ct = ct,
        .zq = zq,           .acc = acc, .fs = fs,
    };

    // prologue: slab 0 -> buffer 0, then start slab 1 into buffer 1.
    try b.linef("mov.u32 {s}, 0;", .{r_pf});
    try prefetch(&b, 0, A);
    try commit(&b, 0, A);
    try b.linef("mov.u32 {s}, 1; min.u32 {s}, {s}, {s};", .{ r_pf, r_pf, r_pf, r_last });
    try prefetch(&b, 1, A);

    const lp = try b.newLabel("slab");
    try b.linef("mov.u32 {s}, 0;", .{r_i});
    try b.label(lp);
    try compute(&b, 0, A);
    try commit(&b, 1, A);
    try b.linef("add.u32 {s}, {s}, 2; min.u32 {s}, {s}, {s};", .{ r_pf, r_i, r_pf, r_pf, r_last });
    try prefetch(&b, 0, A);
    try compute(&b, 1, A);
    try commit(&b, 0, A);
    try b.linef("add.u32 {s}, {s}, 3; min.u32 {s}, {s}, {s};", .{ r_pf, r_i, r_pf, r_pf, r_last });
    try prefetch(&b, 1, A);
    try b.linef("add.u32 {s}, {s}, 2;", .{ r_i, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ pr[1], r_i, r_nslab });
    try b.linef("@{s} bra {s};", .{ pr[1], lp });
    _ = pr[0];
    // ---- store: y[col*rows + row] -----------------------------------------
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, tm[0] });
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        var nj2: usize = 0;
        while (nj2 < NT) : (nj2 += 1) {
            const a = acc[(mi2 * NT + nj2) * 4 ..][0..4];
            const rowoff = [_]usize{ 0, 0, 8, 8 };
            const coloff = [_]usize{ 0, 1, 0, 1 };
            for (0..4) |ci| {
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[1], r_crow, mi2 * 16 + rowoff[ci] });
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[2], r_ccol, nj2 * 8 + coloff[ci] });
                try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[3], tm[2], r_rows, tm[1] });
                try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[3] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[0], a[ci], f_scale });
                try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], fs[0] });
            }
        }
    }

    const entry_name = try std.fmt.allocPrint(alloc, "mmq_pipe_q2_0_{s}", .{g.tag});
    defer alloc.free(entry_name);
    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH});
    defer alloc.free(shared_decl);
    return b.build(
        entry_name,
        "    .param .u64 p_w,\n    .param .u64 p_x,\n    .param .u64 p_y,\n    .param .u32 p_rows,\n    .param .u32 p_cols,\n    .param .u32 p_n,\n    .param .f32 p_scale",
        shared_decl,
    );
}

/// q1_0 MMQ, the pipe-tiled sibling of `buildMmqPipeQ4K`/`Q6K`: same 128x128 /
/// 2x2-warp / MT=4 / NT=8 tiling, same "decode the weight tile once into shared
/// and let all four warps reuse it" idea.
///
/// This is the SIMPLEST of the three, because q1_0's format removes both of the
/// things that complicate the others:
///
///   1. **No min term.** q4_k reconstructs `d*sc*q - dmin*m`, so it needs the
///      activation's per-block quant SUM to cancel the offset, and stages a
///      (da, sum) pair per column. q1_0 is `v = ±d` — symmetric, no zero point —
///      so the whole B-side sums region and half the fold arithmetic disappear:
///      `acc += (d * da) * c`.
///   2. **One scale per 128 elements.** A 64-k slab is half a block, so `d` is
///      constant across the slab — the row scale is loaded ONCE per slab, outside
///      the substep loop, where q4_k reloads a pair per 32-k substep and q6_k per
///      16. Shared memory drops to 17.9 KB from q4_k's 20.5 KB.
///
/// The A staging decodes one row's 64 weights from 8 bytes of sign bits with
/// `emitQ1Unpack` (4 u16 loads -> 16 u32 of ±1). ⚠️ Those loads must stay u16:
/// a q1_0 block is 18 bytes, so `block + 2 + h*8` is only ever 2-byte aligned
/// (its alignment alternates with the block parity), and a u32 or v4 load there
/// is misaligned for half the blocks.
///
/// Requires cols % 256 == 0, rows % 128 == 0, and `n` padded to a multiple of 128
/// with the padding zero-filled (see `opMatmulQuantMmqPipe`). Entry `mmq_pipe_q1_0`.
pub fn buildMmqPipeQ1_0(alloc: std.mem.Allocator) ![:0]u8 {
    const BM = 128;
    const BN = 128;
    const kstep = 64;
    const KS = kstep / 32; // 32-k substeps (== q8_1 activation blocks) per slab
    const MT = 4;
    const NT = 8;
    // ⚠️ Shared rows are padded to 80 bytes, not the 64 they hold. The fragment
    // load has lane L read row `base + L/4` at word `ks*8 + L%4`, so with a
    // 64-byte (16-word) stride the 32 lanes touch only 8 of the 32 banks — a
    // **4-way bank conflict on every fragment load**, and there are 32 of them per
    // substep. At a 20-word stride the row term is `20*gid mod 32` =
    // {0,20,8,28,16,4,24,12}, eight distinct multiples of 4, so adding `tf` 0..3
    // covers all 32 banks exactly once: conflict-free.
    //
    // This is the PADDING alternative to an XOR swizzle, and it is the right one
    // here: at kstep=64 a row is only 16 words, so the usual `(row&7)<<4` swizzle
    // would cross into the next row, and the `(row&3)<<4` mask that does fit still
    // leaves 2-way conflicts because gid 4..7 alias gid 0..3. It also beats
    // `ldmatrix`, which on sm_86 is `.b16` only (awkward for an int8 fragment) and
    // would save just ~4% of the instruction stream.
    const stride = 80;
    const ATILE = BM * stride;
    const BTILE = BN * stride;
    const ASC = BM * 4; // one f32 `d` per row (constant across the slab)
    const BSC = BN * KS * 4; // one f32 `da` per column per substep
    const B_OFF = ATILE;
    const ASC_OFF = ATILE + BTILE;
    const BSC_OFF = ASC_OFF + ASC;
    // One buffer's worth of shared. The pipeline holds TWO (see the slab loop).
    const SH_HALF = BSC_OFF + BSC;
    const SH = 2 * SH_HALF;

    var b = ptx.Builder.init(alloc);
    defer b.deinit();

    const acc = try b.regs(.f32, MT * NT * 4);
    const af = try b.regs(.b32, MT * 4);
    const bf = try b.regs(.b32, NT * 2);
    const ct = try b.regs(.b32, 4); // mma destination (C operand is the zero quad)
    const zq = try b.regs(.b32, 4); // permanently zero
    const rsc = try b.regs(.f32, MT * 2); // per-mi: `d` for rows gid and gid+8
    const csc = try b.regs(.f32, 2); // per-nj: `da` for cols tf*2 and tf*2+1

    const rd_w = try b.reg(.b64);
    const rd_x = try b.reg(.b64);
    const rd_y = try b.reg(.b64);
    const r_rows = try b.reg(.b32);
    const r_cols = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const f_scale = try b.reg(.f32);
    try b.linef("ld.param.u64 {s}, [p_w];", .{rd_w});
    try b.linef("ld.param.u64 {s}, [p_x];", .{rd_x});
    try b.linef("ld.param.u64 {s}, [p_y];", .{rd_y});
    try b.linef("ld.param.u32 {s}, [p_rows];", .{r_rows});
    try b.linef("ld.param.u32 {s}, [p_cols];", .{r_cols});
    try b.linef("ld.param.u32 {s}, [p_n];", .{r_n});
    try b.linef("ld.param.f32 {s}, [p_scale];", .{f_scale});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_w, rd_w });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_x, rd_x });
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_y, rd_y });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_wm = try b.reg(.b32);
    const r_wn = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_row0 = try b.reg(.b32);
    const r_col0 = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const r_nblk = try b.reg(.b32);
    const r_rb = try b.reg(.b32);
    const r_bpt = try b.reg(.b32);
    const r_l1 = try b.reg(.b32);
    const r_l2 = try b.reg(.b32);
    const tm = try b.regs(.b32, 8);
    const rdt = try b.regs(.b64, 4);

    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("and.b32 {s}, {s}, 1;", .{ r_wm, r_warp });
    try b.linef("shr.u32 {s}, {s}, 1;", .{ r_wn, r_warp });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_row0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_row0, r_row0 });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_col0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_col0, r_col0 });
    try b.linef("shr.u32 {s}, {s}, 7;", .{ r_nblk, r_cols }); // q1_0 blocks per row
    try b.linef("mul.lo.u32 {s}, {s}, 18;", .{ r_rb, r_nblk }); // row bytes
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_bpt, r_cols }); // q8 blocks per token
    try b.linef("mov.u32 {s}, 0x11100100;", .{r_l1});
    try b.linef("mov.u32 {s}, 0x000001FF;", .{r_l2});

    // Activation qs region (no sums region: q1_0 needs no min term).
    const rd_qs = try b.reg(.b64);
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ tm[0], r_n, r_bpt });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_qs, tm[0] });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qs, rd_x, rd_qs });

    // Staging identities: one thread per A row and per B column (BM=BN=128=nthreads).
    const rd_wrow = try b.reg(.b64);
    const rd_bcol = try b.reg(.b64);
    const r_gcol = try b.reg(.b32);
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], r_row0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_wrow, tm[0], r_rb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_wrow, rd_w, rd_wrow });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_gcol, r_col0, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_bcol, r_gcol, r_cols });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_bcol, rd_qs, rd_bcol });

    // Shared destinations for this thread's staged row/column.
    const r_adst = try b.reg(.b32);
    const r_bdst = try b.reg(.b32);
    const r_ascd = try b.reg(.b32);
    const r_bscd = try b.reg(.b32);
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_adst, r_t, stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_adst, r_adst, r_smem });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bdst, r_t, stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bdst, r_bdst, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bdst, r_bdst, B_OFF });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_ascd, r_t });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascd, r_ascd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascd, r_ascd, ASC_OFF });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscd, r_t, KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscd, r_bscd, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscd, r_bscd, BSC_OFF });

    // Fragment-load bases (pipe's use_ldmatrix=false layout).
    const r_asl = try b.reg(.b32);
    const r_bsl = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_asl, tm[0], stride });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_asl, r_asl, r_smem });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bsl, tm[0], stride });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, tm[1] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bsl, r_bsl, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bsl, r_bsl, B_OFF });

    // Scale-read bases: row (wm*64 + mi*16 + gid), col (wn*64 + nj*8 + tf*2).
    const r_ascr = try b.reg(.b32);
    const r_bscr = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], r_gid });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_ascr, tm[0] });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ascr, r_ascr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ascr, r_ascr, ASC_OFF });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ tm[0], r_wn });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[1], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ tm[0], tm[0], tm[1] });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_bscr, tm[0], KS * 4 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_bscr, r_bscr, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_bscr, r_bscr, BSC_OFF });

    for (acc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});
    for (zq) |r| try b.linef("mov.u32 {s}, 0;", .{r});

    // ---- slab loop: cp.async double-buffered software pipeline -------------
    //
    // ⚠️ The single-buffered version this replaces measured 19% of the kernel in
    // EXPOSED global-load latency: staging wrote shared, `bar.sync`d, computed,
    // `bar.sync`d again, so nothing overlapped the ~500-cycle loads. Removing the
    // staging entirely (garbage output, timing only) gave 80.4 TOPS against 65.0
    // with it — that measurement is what justified this rewrite, and the fold is
    // NOT the limiter (cutting it 4x was worth only +10%).
    //
    // Three things make it fit:
    //   - **B goes global -> shared with `cp.async`**, so its 16 staging registers
    //     disappear. At 254 registers there was no headroom for prefetch state
    //     otherwise; cp.async pays for itself before any latency is hidden.
    //   - **A cannot use cp.async** (its bytes are transformed by the prmt decode),
    //     so its raw u16s are prefetched into REGISTERS a slab ahead and decoded
    //     into shared after the compute that hides their latency.
    //   - **The slab loop is unrolled 2x so the buffer parity is COMPILE-TIME**, and
    //     every shared offset stays an immediate (`buf * SH_HALF`). `cols % 256 == 0`
    //     makes `cols/kstep` a multiple of 4, so nslab is always even.
    //
    // One `bar.sync` per slab instead of two, and the prefetch index is CLAMPED to
    // the last slab rather than branched on: the extra fetch is garbage that is
    // never consumed, and clamping keeps the address in bounds.
    const r_i = try b.reg(.b32);
    const r_nslab = try b.reg(.b32);
    const r_last = try b.reg(.b32);
    const r_pf = try b.reg(.b32); // clamped prefetch slab index
    const pr = try b.regs(.pred, 2);
    const ar = try b.regs(.b32, 4); // A's prefetched raw sign words
    const h16 = try b.regs(.b16, 2);
    const f_ad = try b.reg(.f32); // A's prefetched d
    const f_bd = try b.regs(.f32, 2); // B's prefetched scales
    const fs = try b.regs(.f32, 4);
    const un = try b.regs(.b32, 2);
    const us = try b.regs(.b32, 4);
    const uv = try b.regs(.b32, 4);

    try b.linef("shr.u32 {s}, {s}, 6;", .{ r_nslab, r_cols }); // cols/kstep
    try b.linef("add.u32 {s}, {s}, -1;", .{ r_last, r_nslab });

    // Issue the global side of slab `r_pf` into buffer `buf`: A's raw words into
    // registers, B straight into shared with cp.async.
    const prefetch = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            // A: block = slab/2, half = slab&1; qs is only 2-byte aligned.
            try bb.linef("shr.u32 {s}, {s}, 1;", .{ args.tm[0], args.r_pf });
            try bb.linef("and.b32 {s}, {s}, 1;", .{ args.tm[1], args.r_pf });
            try bb.linef("mul.lo.u32 {s}, {s}, 18;", .{ args.tm[2], args.tm[0] });
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[2] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[1], args.rd_wrow, args.rdt[0] });
            try bb.linef("ld.global.b16 {s}, [{s}];", .{ args.h16[0], args.rdt[1] }); // d
            try bb.linef("shl.b32 {s}, {s}, 3; add.u32 {s}, {s}, 2;", .{ args.tm[3], args.tm[1], args.tm[3], args.tm[3] });
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[3] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[2], args.rdt[1], args.rdt[0] });
            for (0..4) |j| try bb.linef("ld.global.u16 {s}, [{s}+{d}];", .{ args.ar[j], args.rdt[2], j * 2 });
            try bb.linef("cvt.f32.f16 {s}, {s};", .{ args.f_ad, args.h16[0] });
            // B: 64 bytes per column, global -> shared, no registers.
            try bb.linef("shl.b32 {s}, {s}, 6;", .{ args.tm[0], args.r_pf }); // slab*kstep
            try bb.linef("cvt.u64.u32 {s}, {s};", .{ args.rdt[0], args.tm[0] });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ args.rdt[1], args.rd_bcol, args.rdt[0] });
            for (0..4) |q4| try bb.linef("cp.async.ca.shared.global [{s}+{d}], [{s}+{d}], 16;", .{ args.r_bdst, off + q4 * 16, args.rdt[1], q4 * 16 });
            try bb.line("cp.async.commit_group;");
            // B's per-substep scales (tiny; plain loads, stored at commit).
            try bb.linef("mul.lo.u32 {s}, {s}, {d};", .{ args.tm[0], args.r_pf, args.ks });
            try bb.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ args.tm[1], args.r_gcol, args.r_bpt, args.tm[0] });
            try bb.linef("mul.wide.u32 {s}, {s}, 4; add.s64 {s}, {s}, {s};", .{ args.rdt[0], args.tm[1], args.rdt[1], args.rd_x, args.rdt[0] });
            try bb.linef("ld.global.f32 {s}, [{s}];", .{ args.f_bd[0], args.rdt[1] });
            try bb.linef("ld.global.f32 {s}, [{s}+4];", .{ args.f_bd[1], args.rdt[1] });
        }
    }.emit;

    // Land the prefetched slab in buffer `buf`: decode A into shared, store both
    // scale sets, then wait for this buffer's cp.async and sync.
    const commit = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            for (0..4) |j| {
                try emitQ1Unpack(bb, args.ar[j], args.r_l1, args.r_l2, args.tm[4], args.un, args.us, args.uv);
                try bb.linef("st.shared.v4.u32 [{s}+{d}], {{{s}, {s}, {s}, {s}}};", .{ args.r_adst, off + j * 16, args.uv[0], args.uv[1], args.uv[2], args.uv[3] });
            }
            try bb.linef("st.shared.f32 [{s}+{d}], {s};", .{ args.r_ascd, off, args.f_ad });
            try bb.linef("st.shared.v2.f32 [{s}+{d}], {{{s}, {s}}};", .{ args.r_bscd, off, args.f_bd[0], args.f_bd[1] });
            try bb.line("cp.async.wait_group 0;");
            try bb.line("bar.sync 0;");
        }
    }.emit;

    // MT x NT mmas over the buffer's KS substeps, folding d*da per substep.
    const compute = struct {
        fn emit(bb: *ptx.Builder, buf: usize, args: anytype) !void {
            const off = buf * args.sh_half;
            var mi0: usize = 0;
            while (mi0 < args.mt) : (mi0 += 1) {
                try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.rsc[mi0 * 2 + 0], args.r_ascr, off + mi0 * 16 * 4 });
                try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.rsc[mi0 * 2 + 1], args.r_ascr, off + mi0 * 16 * 4 + 8 * 4 });
            }
            var ks: usize = 0;
            while (ks < args.ks) : (ks += 1) {
                var mi: usize = 0;
                while (mi < args.mt) : (mi += 1) {
                    const o = off + mi * 16 * args.stride + ks * 32;
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 0], args.r_asl, o });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 1], args.r_asl, o + 8 * args.stride });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 2], args.r_asl, o + 16 });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.af[mi * 4 + 3], args.r_asl, o + 8 * args.stride + 16 });
                }
                var nj: usize = 0;
                while (nj < args.nt) : (nj += 1) {
                    const o = off + nj * 8 * args.stride + ks * 32;
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.bf[nj * 2 + 0], args.r_bsl, o });
                    try bb.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ args.bf[nj * 2 + 1], args.r_bsl, o + 16 });
                }
                nj = 0;
                while (nj < args.nt) : (nj += 1) {
                    const co = off + nj * 8 * args.ks * 4 + ks * 4;
                    try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.csc[0], args.r_bscr, co });
                    try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ args.csc[1], args.r_bscr, co + args.ks * 4 });
                    mi = 0;
                    while (mi < args.mt) : (mi += 1) {
                        try bb.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                            args.ct[0],          args.ct[1],          args.ct[2],          args.ct[3],
                            args.af[mi * 4 + 0], args.af[mi * 4 + 1], args.af[mi * 4 + 2], args.af[mi * 4 + 3],
                            args.bf[nj * 2 + 0], args.bf[nj * 2 + 1],
                            args.zq[0],          args.zq[1],          args.zq[2],          args.zq[3],
                        });
                        const a = args.acc[(mi * args.nt + nj) * 4 ..][0..4];
                        const rows2 = [_]usize{ 0, 0, 1, 1 };
                        const cols2 = [_]usize{ 0, 1, 0, 1 };
                        for (0..4) |ci| {
                            try bb.linef("cvt.rn.f32.s32 {s}, {s};", .{ args.fs[0], args.ct[ci] });
                            try bb.linef("mul.f32 {s}, {s}, {s};", .{ args.fs[1], args.rsc[mi * 2 + rows2[ci]], args.csc[cols2[ci]] });
                            try bb.linef("fma.rn.f32 {s}, {s}, {s}, {s};", .{ a[ci], args.fs[1], args.fs[0], a[ci] });
                        }
                    }
                }
            }
        }
    }.emit;

    const A = .{
        .sh_half = SH_HALF, .ks = KS,   .kstep = kstep, .stride = stride, .mt = MT, .nt = NT,
        .tm = tm,           .rdt = rdt, .r_pf = r_pf,   .rd_wrow = rd_wrow, .rd_bcol = rd_bcol,
        .h16 = h16,         .ar = ar,   .f_ad = f_ad,   .f_bd = f_bd,   .r_bdst = r_bdst,
        .r_adst = r_adst,   .r_ascd = r_ascd, .r_bscd = r_bscd, .r_gcol = r_gcol, .r_bpt = r_bpt,
        .rd_x = rd_x,       .r_l1 = r_l1, .r_l2 = r_l2, .un = un,       .us = us,
        .uv = uv,           .r_ascr = r_ascr, .r_asl = r_asl, .r_bsl = r_bsl, .r_bscr = r_bscr,
        .rsc = rsc,         .csc = csc, .af = af,       .bf = bf,       .ct = ct,
        .zq = zq,           .acc = acc, .fs = fs,
    };

    // prologue: slab 0 -> buffer 0, then start slab 1 into buffer 1.
    try b.linef("mov.u32 {s}, 0;", .{r_pf});
    try prefetch(&b, 0, A);
    try commit(&b, 0, A);
    try b.linef("mov.u32 {s}, 1; min.u32 {s}, {s}, {s};", .{ r_pf, r_pf, r_pf, r_last });
    try prefetch(&b, 1, A);

    const lp = try b.newLabel("slab");
    try b.linef("mov.u32 {s}, 0;", .{r_i});
    try b.label(lp);
    try compute(&b, 0, A);
    try commit(&b, 1, A);
    try b.linef("add.u32 {s}, {s}, 2; min.u32 {s}, {s}, {s};", .{ r_pf, r_i, r_pf, r_pf, r_last });
    try prefetch(&b, 0, A);
    try compute(&b, 1, A);
    try commit(&b, 0, A);
    try b.linef("add.u32 {s}, {s}, 3; min.u32 {s}, {s}, {s};", .{ r_pf, r_i, r_pf, r_pf, r_last });
    try prefetch(&b, 1, A);
    try b.linef("add.u32 {s}, {s}, 2;", .{ r_i, r_i });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ pr[1], r_i, r_nslab });
    try b.linef("@{s} bra {s};", .{ pr[1], lp });
    _ = pr[0];
    // ---- store: y[col*rows + row] -----------------------------------------
    const r_crow = try b.reg(.b32);
    const r_ccol = try b.reg(.b32);
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_crow, r_wm });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_row0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_crow, r_crow, r_gid });
    try b.linef("shl.b32 {s}, {s}, 6;", .{ r_ccol, r_wn });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, r_col0 });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ tm[0], r_tf });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_ccol, r_ccol, tm[0] });
    var mi2: usize = 0;
    while (mi2 < MT) : (mi2 += 1) {
        var nj2: usize = 0;
        while (nj2 < NT) : (nj2 += 1) {
            const a = acc[(mi2 * NT + nj2) * 4 ..][0..4];
            const rowoff = [_]usize{ 0, 0, 8, 8 };
            const coloff = [_]usize{ 0, 1, 0, 1 };
            for (0..4) |ci| {
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[1], r_crow, mi2 * 16 + rowoff[ci] });
                try b.linef("add.u32 {s}, {s}, {d};", .{ tm[2], r_ccol, nj2 * 8 + coloff[ci] });
                try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ tm[3], tm[2], r_rows, tm[1] });
                try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rdt[0], tm[3] });
                try b.linef("add.s64 {s}, {s}, {s};", .{ rdt[0], rd_y, rdt[0] });
                try b.linef("mul.f32 {s}, {s}, {s};", .{ fs[0], a[ci], f_scale });
                try b.linef("st.global.f32 [{s}], {s};", .{ rdt[0], fs[0] });
            }
        }
    }

    const shared_decl = try std.fmt.allocPrint(alloc, ".shared .align 16 .b8 smem[{d}];", .{SH});
    defer alloc.free(shared_decl);
    return b.build(
        "mmq_pipe_q1_0",
        "    .param .u64 p_w,\n    .param .u64 p_x,\n    .param .u64 p_y,\n    .param .u32 p_rows,\n    .param .u32 p_cols,\n    .param .u32 p_n,\n    .param .f32 p_scale",
        shared_decl,
    );
}
