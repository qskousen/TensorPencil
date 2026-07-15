//! SPIR-V cooperative-matrix (tensor-core) GEMM and attention kernels.
//!
//! Zig's SPIR-V backend cannot express OpTypeCooperativeMatrixKHR, so these
//! kernels are authored as SPIR-V *assembly text* and turned into the binary
//! word stream at runtime by the small in-tree assembler in `spirv_asm.zig`
//! (no external tool). Each `build*` function emits the text for one kernel
//! variant and returns `sasm.assembleChecked(...)`.
//!
//! The GEMM kernels compute C = A @ B with the k-major (transposed) weight
//! layout used across the matmul backends, so B needs no rearranging; C is
//! stored directly. Dimensions are multiples of the fragment tile (callers
//! pad m; n/k already align). Bindings (set 0): 0 = B (weights), 1 = A (x),
//! 2 = C (y); push constants carry {m, n, k, stride}.

const std = @import("std");
const sasm = @import("spirv_asm.zig");

/// Helper for authoring a kernel as SPIR-V assembly text (see spirv_asm.zig).
/// `id()` hands out fresh `%tN` names for anonymous SSA temporaries (mirroring
/// the old `Asm.id()` counter, so there are never naming collisions); fixed
/// types/constants/labels are referenced by readable names in the text.
const Emit = struct {
    w: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    idc: u32 = 0,

    fn id(self: *Emit) u32 {
        self.idc += 1;
        return self.idc;
    }

    fn line(self: *Emit, comptime fmt: []const u8, args: anytype) !void {
        try self.w.print(self.gpa, fmt, args);
        try self.w.append(self.gpa, '\n');
    }
};

/// int8 tensor-core GEMM: `C(s32) = A(s8) @ B(s8)` with a 16x16x32 subgroup
/// cooperative matrix (sint8 in, sint32 accumulate). One subgroup (a 32-thread
/// workgroup) computes an `mt` x `nt` grid of 16x16 s32 output tiles, stepping
/// k by 32. Register-tiled: each k-step loads `mt` A fragments and `nt` B
/// fragments and issues `mt*nt` MMAs, so each A fragment is reused `nt` times
/// and each B fragment `mt` times — cutting the global traffic of the naive
/// (mt=nt=1) kernel without any workgroup memory (so no shared-memory driver
/// hazard). A is x row-major s8 [m][k]; B is the k-major s8 weight layout
/// [k][stride] (same transpose as fp8 — both 1-byte); C is s32 [m][n]. Scaling
/// by per-row activation/weight scales is a separate pass. Dims: n a multiple
/// of 16*nt, m of 16*mt, k of 32. Bindings (set 0): 0 = B (s8), 1 = A (s8),
/// 2 = C (s32). Push: {m,n,k,stride}. `mt`,`nt` in 1..4.
pub fn buildGemmI8(gpa: std.mem.Allocator, mt: u32, nt: u32) ![]align(4) u8 {
    std.debug.assert(mt >= 1 and mt <= 8 and nt >= 1 and nt <= 8);

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    const w = &t;

    // u32 index-constant pool, deduped by value and emitted in a fixed order;
    // each value v is referenced as `%cu_<v>`, so the assembler interns it once.
    var pool: [40]u32 = undefined;
    var pn: usize = 0;
    const addC = struct {
        fn f(p: []u32, n: *usize, v: u32) void {
            for (p[0..n.*]) |ex| if (ex == v) return;
            p[n.*] = v;
            n.* += 1;
        }
    }.f;
    addC(&pool, &pn, 0);
    addC(&pool, &pn, 1);
    addC(&pool, &pn, 2);
    addC(&pool, &pn, 3);
    addC(&pool, &pn, 16);
    addC(&pool, &pn, 32);
    addC(&pool, &pn, 16 * mt);
    addC(&pool, &pn, 16 * nt);
    for (0..mt) |i| addC(&pool, &pn, @intCast(16 * i));
    for (0..nt) |j| addC(&pool, &pn, @intCast(16 * j));

    // --- header / decorations ------------------------------------------------
    try w.print(gpa,
        \\OpCapability Shader
        \\OpCapability Int8
        \\OpCapability StorageBuffer8BitAccess
        \\OpCapability VulkanMemoryModel
        \\OpCapability CooperativeMatrixKHR
        \\OpExtension "SPV_KHR_cooperative_matrix"
        \\OpExtension "SPV_KHR_vulkan_memory_model"
        \\OpExtension "SPV_KHR_8bit_storage"
        \\OpMemoryModel Logical Vulkan
        \\OpEntryPoint GLCompute %main "main" %gid %vb %va %vc %vpush
        \\OpExecutionMode %main LocalSize 32 1 1
        \\OpDecorate %gid BuiltIn WorkgroupId
        \\OpDecorate %arr_s8 ArrayStride 1
        \\OpDecorate %arr_s32 ArrayStride 4
        \\OpDecorate %sb Block
        \\OpMemberDecorate %sb 0 Offset 0
        \\OpDecorate %sa Block
        \\OpMemberDecorate %sa 0 Offset 0
        \\OpDecorate %sc Block
        \\OpMemberDecorate %sc 0 Offset 0
        \\OpDecorate %push Block
        \\OpMemberDecorate %push 0 Offset 0
        \\OpMemberDecorate %push 1 Offset 4
        \\OpMemberDecorate %push 2 Offset 8
        \\OpMemberDecorate %push 3 Offset 12
        \\OpDecorate %vb DescriptorSet 0
        \\OpDecorate %vb Binding 0
        \\OpDecorate %va DescriptorSet 0
        \\OpDecorate %va Binding 1
        \\OpDecorate %vc DescriptorSet 0
        \\OpDecorate %vc Binding 2
        \\
    , .{});

    // --- types --------------------------------------------------------------
    try w.print(gpa,
        \\%void = OpTypeVoid
        \\%fnvoid = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%s8 = OpTypeInt 8 1
        \\%s32 = OpTypeInt 32 1
        \\%v3u = OpTypeVector %u32 3
        \\%ptr_in_v3 = OpTypePointer Input %v3u
        \\%gid = OpVariable %ptr_in_v3 Input
        \\%c_arrlen = OpConstant %u32 268435456
        \\%arr_s8 = OpTypeArray %s8 %c_arrlen
        \\%arr_s32 = OpTypeArray %s32 %c_arrlen
        \\%sb = OpTypeStruct %arr_s8
        \\%sa = OpTypeStruct %arr_s8
        \\%sc = OpTypeStruct %arr_s32
        \\%ptr_sb = OpTypePointer StorageBuffer %sb
        \\%ptr_sa = OpTypePointer StorageBuffer %sa
        \\%ptr_sc = OpTypePointer StorageBuffer %sc
        \\%vb = OpVariable %ptr_sb StorageBuffer
        \\%va = OpVariable %ptr_sa StorageBuffer
        \\%vc = OpVariable %ptr_sc StorageBuffer
        \\%push = OpTypeStruct %u32 %u32 %u32 %u32
        \\%ptr_push = OpTypePointer PushConstant %push
        \\%vpush = OpVariable %ptr_push PushConstant
        \\%ptr_pc_u32 = OpTypePointer PushConstant %u32
        \\%ptr_sb_s8 = OpTypePointer StorageBuffer %s8
        \\%ptr_sa_s8 = OpTypePointer StorageBuffer %s8
        \\%ptr_sc_s32 = OpTypePointer StorageBuffer %s32
        \\
    , .{});

    // Index-constant pool.
    for (pool[0..pn]) |v| try w.print(gpa, "%cu_{d} = OpConstant %u32 {d}\n", .{ v, v });

    // Cooperative-matrix types (scope Subgroup = %cu_3). A=16x32 use 0,
    // B=32x16 use 1, C=16x16 use 2. Accumulator initialised to zeros.
    try w.print(gpa,
        \\%bool = OpTypeBool
        \\%mat_a = OpTypeCooperativeMatrixKHR %s8 %cu_3 %cu_16 %cu_32 %cu_0
        \\%mat_b = OpTypeCooperativeMatrixKHR %s8 %cu_3 %cu_32 %cu_16 %cu_1
        \\%mat_c = OpTypeCooperativeMatrixKHR %s32 %cu_3 %cu_16 %cu_16 %cu_2
        \\%c_s32_0 = OpConstant %s32 0
        \\%c_acc0 = OpConstantComposite %mat_c %c_s32_0
        \\%ptr_fn_matc = OpTypePointer Function %mat_c
        \\%ptr_fn_u32 = OpTypePointer Function %u32
        \\
    , .{});

    // --- function -----------------------------------------------------------
    const nt_tiles = mt * nt;
    try w.print(gpa,
        \\%main = OpFunction %void None %fnvoid
        \\%entry = OpLabel
        \\
    , .{});
    // mt*nt accumulator vars + the k0 induction var (all in the entry block).
    for (0..nt_tiles) |i| try w.print(gpa, "%acc_{d} = OpVariable %ptr_fn_matc Function\n", .{i});
    try w.print(gpa,
        \\%k0_var = OpVariable %ptr_fn_u32 Function
        \\%gidv = OpLoad %v3u %gid
        \\%tile_c = OpCompositeExtract %u32 %gidv 0
        \\%tile_r = OpCompositeExtract %u32 %gidv 1
        \\
    , .{});
    try w.print(gpa, "%col0 = OpIMul %u32 %tile_c %cu_{d}\n", .{16 * nt});
    try w.print(gpa, "%row0 = OpIMul %u32 %tile_r %cu_{d}\n", .{16 * mt});

    // Push constants: {m, n, k, stride}. We use n (1), k (2), stride (3).
    for (0..4) |m| {
        try w.print(gpa, "%pptr_{d} = OpAccessChain %ptr_pc_u32 %vpush %cu_{d}\n", .{ m, m });
        try w.print(gpa, "%pval_{d} = OpLoad %u32 %pptr_{d}\n", .{ m, m });
    }

    // Per-tile invariants. row = row0 + 16*mi; a_row_base = row*k;
    // c_row_n = row*n; col_n = col0 + 16*nj.
    for (0..mt) |mi| {
        try w.print(gpa, "%rowmi_{d} = OpIAdd %u32 %row0 %cu_{d}\n", .{ mi, 16 * mi });
        try w.print(gpa, "%arb_{d} = OpIMul %u32 %rowmi_{d} %pval_2\n", .{ mi, mi });
        try w.print(gpa, "%crn_{d} = OpIMul %u32 %rowmi_{d} %pval_1\n", .{ mi, mi });
    }
    for (0..nt) |nj| try w.print(gpa, "%coln_{d} = OpIAdd %u32 %col0 %cu_{d}\n", .{ nj, 16 * nj });

    for (0..nt_tiles) |i| try w.print(gpa, "OpStore %acc_{d} %c_acc0\n", .{i});
    try w.print(gpa,
        \\OpStore %k0_var %cu_0
        \\OpBranch %head
        \\%head = OpLabel
        \\OpLoopMerge %merge %cont None
        \\OpBranch %cond
        \\%cond = OpLabel
        \\%k0v = OpLoad %u32 %k0_var
        \\%cmp = OpULessThan %bool %k0v %pval_2
        \\OpBranchConditional %cmp %body %merge
        \\%body = OpLabel
        \\
    , .{});

    // Load mt A fragments and nt B fragments for this k-slab.
    for (0..mt) |mi| {
        try w.print(gpa, "%aoff_{d} = OpIAdd %u32 %arb_{d} %k0v\n", .{ mi, mi });
        try w.print(gpa, "%aptr_{d} = OpAccessChain %ptr_sa_s8 %va %cu_0 %aoff_{d}\n", .{ mi, mi });
        try w.print(gpa, "%ma_{d} = OpCooperativeMatrixLoadKHR %mat_a %aptr_{d} %cu_0 %pval_2\n", .{ mi, mi });
    }
    try w.print(gpa, "%b_rowmul = OpIMul %u32 %k0v %pval_3\n", .{});
    for (0..nt) |nj| {
        try w.print(gpa, "%boff_{d} = OpIAdd %u32 %b_rowmul %coln_{d}\n", .{ nj, nj });
        try w.print(gpa, "%bptr_{d} = OpAccessChain %ptr_sb_s8 %vb %cu_0 %boff_{d}\n", .{ nj, nj });
        try w.print(gpa, "%mb_{d} = OpCooperativeMatrixLoadKHR %mat_b %bptr_{d} %cu_0 %pval_3\n", .{ nj, nj });
    }
    // mt*nt MMAs, reusing the loaded fragments. Operands-mask 15 = A|B|C|Result
    // all signed.
    for (0..mt) |mi| {
        for (0..nt) |nj| {
            const ti = mi * nt + nj;
            try w.print(gpa, "%accin_{d} = OpLoad %mat_c %acc_{d}\n", .{ ti, ti });
            try w.print(gpa, "%accout_{d} = OpCooperativeMatrixMulAddKHR %mat_c %ma_{d} %mb_{d} %accin_{d} 15\n", .{ ti, mi, nj, ti });
            try w.print(gpa, "OpStore %acc_{d} %accout_{d}\n", .{ ti, ti });
        }
    }

    try w.print(gpa,
        \\OpBranch %cont
        \\%cont = OpLabel
        \\
    , .{});
    try w.print(gpa, "%k0n = OpIAdd %u32 %k0v %cu_32\n", .{}); // k0 += 32
    try w.print(gpa,
        \\OpStore %k0_var %k0n
        \\OpBranch %head
        \\%merge = OpLabel
        \\
    , .{});

    // Store each C tile.
    for (0..mt) |mi| {
        for (0..nt) |nj| {
            const ti = mi * nt + nj;
            try w.print(gpa, "%cbase_{d} = OpIAdd %u32 %crn_{d} %coln_{d}\n", .{ ti, mi, nj });
            try w.print(gpa, "%cptr_{d} = OpAccessChain %ptr_sc_s32 %vc %cu_0 %cbase_{d}\n", .{ ti, ti });
            try w.print(gpa, "%accfin_{d} = OpLoad %mat_c %acc_{d}\n", .{ ti, ti });
            try w.print(gpa, "OpCooperativeMatrixStoreKHR %cptr_{d} %accfin_{d} %cu_0 %pval_1\n", .{ ti, ti });
        }
    }
    try w.print(gpa,
        \\OpReturn
        \\OpFunctionEnd
        \\
    , .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}

/// Register tile (16x16 fragments per subgroup) for the int8 coop GEMM.
pub const i8_mt: u32 = 4;
pub const i8_nt: u32 = 4;

/// Route the DiT int8 GEMM through the shared-memory kernel below (vs the
/// register-tiled buildGemmI8). Fork #2 from PLAN.md — the shared kernel
/// stages s8 A/B through workgroup memory (no decode; half the bytes of the
/// fp8 f16 staging), so all fragment loads are ldmatrix-from-shared.
pub const i8_shared = true;

/// 8-warp (256-thread) variant of the shared int8 GEMM: 2x4 warp grid,
/// 8 accumulator fragments/warp = 64 s32/thread -> ~2 workgroups/SM (vs the
/// 4-warp config's 16 frags = 128 regs -> ~1 wg/SM). Occupancy lever for the
/// in-DiT matmul (int8 accumulators are forced s32 and can't shrink like fp8's
/// f16 accs, so more warps is the only way to raise occupancy).
pub const coop_i8_warps8 = false;

/// Double-buffer the shared int8 GEMM: issue the NEXT k-slab's global loads
/// (into registers) before the current slab's MMA section, so the ~400-cycle
/// global-load latency hides under tensor-core work. The int8 GEMM benches at
/// ~85 TF/s (~30% of the 3090's int8 tensor peak) single-buffered — the
/// single-buffer stall (barrier / stage / barrier / MMA, load latency exposed)
/// is the ceiling, not the MMA rate.
pub const coop_i8_double_buf = true;

/// Shared-memory staged int8 (s8*s8->s32) GEMM: `C(s32)[m][n] = A(s8)[m][k] @ B^T`
/// where B is the k-major s8 weight [k][stride] (same transposed layout as the
/// fp8/f16 paths). LocalSize (32,4) = 4 subgroups; each workgroup computes a
/// 128(m) x 128(n) tile of C. A 128x64 s8 tile of A and a 64x128 s8 tile of B
/// stage into ONE workgroup u32 array per k-step of 64 (single-buffered: barrier
/// / stage / barrier / MMA), then all cooperative fragment loads are
/// ldmatrix-from-shared. The shared array is u32-typed (staging is a plain u32
/// copy — no decode) and the s8 cooperative loads read it with u32-unit strides
/// (the standard int8-matrix-from-uint-buffer pattern). Each warp is a 2x2 grid
/// of 64x64 tiles (4 A-frags 16x32 x 4 B-frags 32x16 = 16 MMAs per 32-k, 2 ks
/// per step). s32 accumulators, phi-carried.
///
/// Bindings: 0 = B (weights, viewed u32), 1 = A (x, viewed u32), 2 = C (y, s32).
/// Push: {m, n, k, stride} u32. m/n multiples of 128, k a multiple of 64,
/// stride a multiple of 4.
pub fn buildGemmSharedI8(gpa: std.mem.Allocator, warps8: bool, fuse_scale: bool, double_buf: bool, c_h16: bool) ![]align(4) u8 {
    std.debug.assert(!c_h16 or fuse_scale);
    const WGN: u32 = 128;
    const K_STEP: u32 = 64;
    const NWARPS: u32 = if (warps8) 8 else 4;
    const THREADS: u32 = 32 * NWARPS;
    const WARP_N: u32 = NWARPS / 2; // warps along n
    const WARP_W: u32 = WGN / WARP_N; // n-cols per warp (64 or 32)
    const MT: u32 = 4;
    const NT: u32 = WARP_W / 16; // 4 or 2
    const A_U32: u32 = 128 * (K_STEP / 4); // 2048; B slab starts here (B_BASE)
    const SH_LEN: u32 = A_U32 + K_STEP * (WGN / 4); // 4096 (16 KB)
    const A_QUADS: u32 = A_U32 / THREADS; // u32/thread for A staging (16 or 8)
    const B_QUADS: u32 = (K_STEP * (WGN / 4)) / THREADS; // (16 or 8) for B staging

    const c_elem = if (c_h16) "f16" else if (fuse_scale) "f32" else "s32";

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    // u32 index-constant pool, deduped by value and emitted in a fixed order;
    // referenced as `%cu_<v>`.
    var pool: [96]u32 = undefined;
    var pn: usize = 0;
    const addC = struct {
        fn f(p: []u32, n: *usize, v: u32) void {
            for (p[0..n.*]) |ex| if (ex == v) return;
            p[n.*] = v;
            n.* += 1;
        }
    }.f;
    for ([_]u32{ 0, 1, 2, 3, 4, 5, 8, 15, 16, 31, 32, 48, 64, 128, 256, 512, 768, 1024, 2048, 0x108 }) |v|
        addC(&pool, &pn, v);
    for (0..@max(A_QUADS, B_QUADS)) |tt| addC(&pool, &pn, @intCast(tt * THREADS));
    for (0..MT) |mi| addC(&pool, &pn, @intCast(mi * 256));
    for (0..NT) |nj| addC(&pool, &pn, @intCast(nj * 4));
    addC(&pool, &pn, WARP_W / 4);
    if (fuse_scale) {
        addC(&pool, &pn, NWARPS * 256);
        for (0..8) |i| addC(&pool, &pn, @intCast(i * 32));
    }

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Int8", .{});
    if (c_h16) try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpCapability CooperativeMatrixKHR", .{});
    try em.line("OpExtension \"SPV_KHR_cooperative_matrix\"", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    if (fuse_scale)
        try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vb %va %vc %vpush %vsh %vscale %vscr", .{})
    else
        try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vb %va %vc %vpush %vsh", .{});
    try em.line("OpExecutionMode %main LocalSize 32 {d} 1", .{NWARPS});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    try em.line("OpDecorate %arr_u32 ArrayStride 4", .{});
    try em.line("OpDecorate %arr_s32 ArrayStride {d}", .{@as(u32, if (c_h16) 2 else 4)});
    for ([_][]const u8{ "sb", "sa", "sc" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }
    try em.line("OpDecorate %push Block", .{});
    for (0..4) |m| try em.line("OpMemberDecorate %push {d} Offset {d}", .{ m, m * 4 });
    for ([_][]const u8{ "vb", "va", "vc" }, 0..) |v, b| {
        try em.line("OpDecorate %{s} DescriptorSet 0", .{v});
        try em.line("OpDecorate %{s} Binding {d}", .{ v, b });
    }
    if (fuse_scale) {
        try em.line("OpDecorate %arr_f32 ArrayStride 4", .{});
        try em.line("OpDecorate %scale Block", .{});
        try em.line("OpMemberDecorate %scale 0 Offset 0", .{});
        try em.line("OpDecorate %vscale DescriptorSet 0", .{});
        try em.line("OpDecorate %vscale Binding 3", .{});
    }

    // --- types / constants ---
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%s8 = OpTypeInt 8 1", .{});
    try em.line("%s32 = OpTypeInt 32 1", .{});
    try em.line("%bool = OpTypeBool", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    if (fuse_scale) try em.line("%f32 = OpTypeFloat 32", .{});
    if (c_h16) try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_u32 = OpTypeArray %u32 %c_arrlen", .{});
    try em.line("%arr_s32 = OpTypeArray %{s} %c_arrlen", .{c_elem});
    try em.line("%sb = OpTypeStruct %arr_u32", .{});
    try em.line("%sa = OpTypeStruct %arr_u32", .{});
    try em.line("%sc = OpTypeStruct %arr_s32", .{});
    try em.line("%ptr_sb = OpTypePointer StorageBuffer %sb", .{});
    try em.line("%ptr_sa = OpTypePointer StorageBuffer %sa", .{});
    try em.line("%ptr_sc = OpTypePointer StorageBuffer %sc", .{});
    try em.line("%vb = OpVariable %ptr_sb StorageBuffer", .{});
    try em.line("%va = OpVariable %ptr_sa StorageBuffer", .{});
    try em.line("%vc = OpVariable %ptr_sc StorageBuffer", .{});
    try em.line("%ptr_sb_u32 = OpTypePointer StorageBuffer %u32", .{});
    try em.line("%ptr_sa_u32 = OpTypePointer StorageBuffer %u32", .{});
    try em.line("%ptr_sc_s32 = OpTypePointer StorageBuffer %{s}", .{c_elem});
    try em.line("%push = OpTypeStruct %u32 %u32 %u32 %u32", .{});
    try em.line("%ptr_push = OpTypePointer PushConstant %push", .{});
    try em.line("%vpush = OpVariable %ptr_push PushConstant", .{});
    try em.line("%ptr_pc_u32 = OpTypePointer PushConstant %u32", .{});
    if (fuse_scale) {
        try em.line("%arr_f32 = OpTypeArray %f32 %c_arrlen", .{});
        try em.line("%scale = OpTypeStruct %arr_f32", .{});
        try em.line("%ptr_scale = OpTypePointer StorageBuffer %scale", .{});
        try em.line("%vscale = OpVariable %ptr_scale StorageBuffer", .{});
        try em.line("%ptr_scale_f32 = OpTypePointer StorageBuffer %f32", .{});
        try em.line("%c_scrlen = OpConstant %u32 {d}", .{NWARPS * 256});
        try em.line("%scr = OpTypeArray %s32 %c_scrlen", .{});
        try em.line("%ptr_wg_scr = OpTypePointer Workgroup %scr", .{});
        try em.line("%vscr = OpVariable %ptr_wg_scr Workgroup", .{});
        try em.line("%ptr_wg_s32 = OpTypePointer Workgroup %s32", .{});
    }
    try em.line("%c_shlen = OpConstant %u32 {d}", .{SH_LEN});
    try em.line("%sh = OpTypeArray %u32 %c_shlen", .{});
    try em.line("%ptr_wg_sh = OpTypePointer Workgroup %sh", .{});
    try em.line("%vsh = OpVariable %ptr_wg_sh Workgroup", .{});
    try em.line("%ptr_wg_u32 = OpTypePointer Workgroup %u32", .{});

    for (pool[0..pn]) |v| try em.line("%cu_{d} = OpConstant %u32 {d}", .{ v, v });

    try em.line("%mat_a = OpTypeCooperativeMatrixKHR %s8 %cu_3 %cu_16 %cu_32 %cu_0", .{});
    try em.line("%mat_b = OpTypeCooperativeMatrixKHR %s8 %cu_3 %cu_32 %cu_16 %cu_1", .{});
    try em.line("%mat_c = OpTypeCooperativeMatrixKHR %s32 %cu_3 %cu_16 %cu_16 %cu_2", .{});
    try em.line("%c_s32_0 = OpConstant %s32 0", .{});
    try em.line("%c_acc0 = OpConstantComposite %mat_c %c_s32_0", .{});

    // --- function ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});

    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const tile_c = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ tile_c, gidv });
    const tile_r = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ tile_r, gidv });
    const col0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_128", .{ col0, tile_c });
    const row0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_128", .{ row0, tile_r });

    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const lx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ lx, lidv });
    const ly = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ ly, lidv });
    const lymul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_32", .{ lymul, ly });
    const flat = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ flat, lymul, lx });

    // push values: {m, n, k, stride}.
    const pnames = [_][]const u8{ "pm", "pn", "pk", "pstride" };
    for (0..4) |m| {
        const pptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_pc_u32 %vpush %cu_{d}", .{ pptr, m });
        try em.line("%{s} = OpLoad %u32 %t{d}", .{ pnames[m], pptr });
    }

    const warp_m = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_1", .{ warp_m, ly });
    const warp_n = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_1", .{ warp_n, ly });
    const a_warp = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_1024", .{ a_warp, warp_m });
    const b_warp = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ b_warp, warp_n, WARP_W / 4 });
    const wm64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_64", .{ wm64, warp_m });
    const c_row0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ c_row0, row0, wm64 });
    const wn64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ wn64, warp_n, WARP_W });
    const c_col0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ c_col0, col0, wn64 });

    // A staging index precompute.
    var a_shidx: [16]u32 = undefined;
    var a_rowk: [16]u32 = undefined;
    var a_kq: [16]u32 = undefined;
    for (0..A_QUADS) |tt| {
        var q = flat;
        if (tt > 0) {
            const qn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ qn, flat, tt * THREADS });
            q = qn;
        }
        a_shidx[tt] = q;
        const row = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_4", .{ row, q });
        a_kq[tt] = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_15", .{ a_kq[tt], q });
        const grow = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ grow, row0, row });
        a_rowk[tt] = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %pk", .{ a_rowk[tt], grow });
    }
    // B staging index precompute.
    var b_shidx: [16]u32 = undefined;
    var b_krow: [16]u32 = undefined;
    var b_cq: [16]u32 = undefined;
    for (0..B_QUADS) |tt| {
        var q = flat;
        if (tt > 0) {
            const qn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ qn, flat, tt * THREADS });
            q = qn;
        }
        b_shidx[tt] = q;
        b_krow[tt] = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_5", .{ b_krow[tt], q });
        b_cq[tt] = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_31", .{ b_cq[tt], q });
    }

    const DB = struct {
        aq: usize,
        bq: usize,
        a_rowk: [16]u32,
        a_kq: [16]u32,
        a_shidx: [16]u32,
        b_krow: [16]u32,
        b_cq: [16]u32,
        b_shidx: [16]u32,
        col0: u32,
        fn load(st: @This(), e: *Emit, kb: []const u8) ![32]u32 {
            var regs: [32]u32 = undefined;
            for (0..st.aq) |tt| {
                const gs8 = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} {s}", .{ gs8, st.a_rowk[tt], kb });
                const gu = e.id();
                try e.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_2", .{ gu, gs8 });
                const gidx = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gidx, gu, st.a_kq[tt] });
                const gptr = e.id();
                try e.line("%t{d} = OpAccessChain %ptr_sa_u32 %va %cu_0 %t{d}", .{ gptr, gidx });
                regs[tt] = e.id();
                try e.line("%t{d} = OpLoad %u32 %t{d}", .{ regs[tt], gptr });
            }
            for (0..st.bq) |tt| {
                const kk = e.id();
                try e.line("%t{d} = OpIAdd %u32 {s} %t{d}", .{ kk, kb, st.b_krow[tt] });
                const kmul = e.id();
                try e.line("%t{d} = OpIMul %u32 %t{d} %pstride", .{ kmul, kk });
                const gs8 = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gs8, kmul, st.col0 });
                const gu = e.id();
                try e.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_2", .{ gu, gs8 });
                const gidx = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gidx, gu, st.b_cq[tt] });
                const gptr = e.id();
                try e.line("%t{d} = OpAccessChain %ptr_sb_u32 %vb %cu_0 %t{d}", .{ gptr, gidx });
                regs[16 + tt] = e.id();
                try e.line("%t{d} = OpLoad %u32 %t{d}", .{ regs[16 + tt], gptr });
            }
            return regs;
        }
        fn store(st: @This(), e: *Emit, regs: [32]u32) !void {
            for (0..st.aq) |tt| {
                const sptr = e.id();
                try e.line("%t{d} = OpAccessChain %ptr_wg_u32 %vsh %t{d}", .{ sptr, st.a_shidx[tt] });
                try e.line("OpStore %t{d} %t{d}", .{ sptr, regs[tt] });
            }
            for (0..st.bq) |tt| {
                const sidx = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %cu_2048", .{ sidx, st.b_shidx[tt] });
                const sptr = e.id();
                try e.line("%t{d} = OpAccessChain %ptr_wg_u32 %vsh %t{d}", .{ sptr, sidx });
                try e.line("OpStore %t{d} %t{d}", .{ sptr, regs[16 + tt] });
            }
        }
    };
    const db: DB = .{
        .aq = A_QUADS,
        .bq = B_QUADS,
        .a_rowk = a_rowk,
        .a_kq = a_kq,
        .a_shidx = a_shidx,
        .b_krow = b_krow,
        .b_cq = b_cq,
        .b_shidx = b_shidx,
        .col0 = col0,
    };

    // Pre-allocate phi-carried ids (k0 induction + accumulators), referenced in
    // the head-block phis before their defining instructions.
    const k0n = em.id();
    var acc_next: [4][4]u32 = undefined;
    for (0..MT) |mi| for (0..NT) |nj| {
        acc_next[mi][nj] = em.id();
    };

    if (double_buf) {
        const regs0 = try db.load(&em, "%cu_0");
        try db.store(&em, regs0);
        try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
    }

    try em.line("OpBranch %head", .{});
    try em.line("%head = OpLabel", .{});
    const k0v = em.id();
    try em.line("%t{d} = OpPhi %u32 %cu_0 %entry %t{d} %cont", .{ k0v, k0n });
    var acc_phi: [4][4]u32 = undefined;
    for (0..MT) |mi| for (0..NT) |nj| {
        acc_phi[mi][nj] = em.id();
        try em.line("%t{d} = OpPhi %mat_c %c_acc0 %entry %t{d} %cont", .{ acc_phi[mi][nj], acc_next[mi][nj] });
    };
    try em.line("OpLoopMerge %merge %cont None", .{});
    try em.line("OpBranch %cond", .{});
    try em.line("%cond = OpLabel", .{});
    const cmp = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %pk", .{ cmp, k0v });
    try em.line("OpBranchConditional %t{d} %body %merge", .{cmp});
    try em.line("%body = OpLabel", .{});

    var regs_next: [32]u32 = undefined;
    if (!double_buf) {
        try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
        const k0v_name = try std.fmt.allocPrint(gpa, "%t{d}", .{k0v});
        defer gpa.free(k0v_name);
        const regs = try db.load(&em, k0v_name);
        try db.store(&em, regs);
        try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
    } else {
        const kb = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_64", .{ kb, k0v });
        const ok = em.id();
        try em.line("%t{d} = OpULessThan %bool %t{d} %pk", .{ ok, kb });
        const kbc = em.id();
        try em.line("%t{d} = OpSelect %u32 %t{d} %t{d} %cu_0", .{ kbc, ok, kb });
        const kbc_name = try std.fmt.allocPrint(gpa, "%t{d}", .{kbc});
        defer gpa.free(kbc_name);
        regs_next = try db.load(&em, kbc_name);
    }

    // MMA: 2 ks of 32 k each, reading the current shared slab.
    var acc_cur = acc_phi;
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..MT) |mi| {
            const off1 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ off1, a_warp, mi * 256 });
            const off = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ off, off1, ks * 8 });
            const aptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_u32 %vsh %t{d}", .{ aptr, off });
            ma[mi] = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %cu_0 %cu_16", .{ ma[mi], aptr });
        }
        for (0..NT) |nj| {
            const off1 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %cu_2048 %cu_{d}", .{ off1, ks * 1024 });
            const off2 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ off2, off1, b_warp });
            const off = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ off, off2, nj * 4 });
            const bptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_u32 %vsh %t{d}", .{ bptr, off });
            const mb = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %cu_0 %cu_32", .{ mb, bptr });
            for (0..MT) |mi| {
                const acc_out = if (ks == 1) acc_next[mi][nj] else em.id();
                try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %t{d} 15", .{ acc_out, ma[mi], mb, acc_cur[mi][nj] });
                acc_cur[mi][nj] = acc_out;
            }
        }
    }

    if (double_buf) {
        try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
        try db.store(&em, regs_next);
        try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
    }
    try em.line("OpBranch %cont", .{});
    try em.line("%cont = OpLabel", .{});
    try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_64", .{ k0n, k0v });
    try em.line("OpBranch %head", .{});

    try em.line("%merge = OpLabel", .{});
    if (!fuse_scale) {
        for (0..MT) |mi| {
            const rowmi = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ rowmi, c_row0, mi * 16 });
            const rowmul = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %pn", .{ rowmul, rowmi });
            for (0..NT) |nj| {
                const ccol = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ ccol, c_col0, nj * 16 });
                const cbase = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ cbase, rowmul, ccol });
                const cptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_sc_s32 %vc %cu_0 %t{d}", .{ cptr, cbase });
                try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %cu_0 %pn", .{ cptr, acc_phi[mi][nj] });
            }
        }
    } else {
        const scr_base = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_256", .{ scr_base, ly });
        for (0..MT) |mi| {
            const rowmi = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ rowmi, c_row0, mi * 16 });
            const rowmul = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %pn", .{ rowmul, rowmi });
            for (0..NT) |nj| {
                const ccol = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ ccol, c_col0, nj * 16 });
                const scrptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_wg_s32 %vscr %t{d}", .{ scrptr, scr_base });
                try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %cu_0 %cu_16", .{ scrptr, acc_phi[mi][nj] });
                try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
                for (0..8) |i| {
                    var elem = lx;
                    if (i > 0) {
                        const en = em.id();
                        try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ en, lx, i * 32 });
                        elem = en;
                    }
                    const lrow = em.id();
                    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_4", .{ lrow, elem });
                    const lcol = em.id();
                    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_15", .{ lcol, elem });
                    const grow = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ grow, rowmi, lrow });
                    const gcol = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gcol, ccol, lcol });
                    const sidx = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ sidx, scr_base, elem });
                    const sptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_wg_s32 %vscr %t{d}", .{ sptr, sidx });
                    const s32v = em.id();
                    try em.line("%t{d} = OpLoad %s32 %t{d}", .{ s32v, sptr });
                    const fv = em.id();
                    try em.line("%t{d} = OpConvertSToF %f32 %t{d}", .{ fv, s32v });
                    const aptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_scale_f32 %vscale %cu_0 %t{d}", .{ aptr, grow });
                    const asv = em.id();
                    try em.line("%t{d} = OpLoad %f32 %t{d}", .{ asv, aptr });
                    const widx = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %pm %t{d}", .{ widx, gcol });
                    const wptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_scale_f32 %vscale %cu_0 %t{d}", .{ wptr, widx });
                    const wsv = em.id();
                    try em.line("%t{d} = OpLoad %f32 %t{d}", .{ wsv, wptr });
                    const prod = em.id();
                    try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ prod, asv, wsv });
                    const yv = em.id();
                    try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ yv, fv, prod });
                    const lrowmul = em.id();
                    try em.line("%t{d} = OpIMul %u32 %t{d} %pn", .{ lrowmul, lrow });
                    const yb = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ yb, rowmul, lrowmul });
                    const yidx = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ yidx, yb, gcol });
                    const yptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_sc_s32 %vc %cu_0 %t{d}", .{ yptr, yidx });
                    if (c_h16) {
                        const yh = em.id();
                        try em.line("%t{d} = OpFConvert %f16 %t{d}", .{ yh, yv });
                        try em.line("OpStore %t{d} %t{d}", .{ yptr, yh });
                    } else {
                        try em.line("OpStore %t{d} %t{d}", .{ yptr, yv });
                    }
                }
                try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});
            }
        }
    }
    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}

/// Stage B — fused int8 prep: rotate (radix-4 FWHT) + per-row abs-max +
/// dynamic quantize, all in ONE hand-assembled workgroup kernel (Zig-emitted
/// workgroup kernels DEVICE_LOST on this NVIDIA driver, so this must be hand-
/// assembled). Replaces the 3-pass eltwise chain (rotate_fwht -> rowscale_i8 ->
/// quantize_i8) and its xr(f32) DRAM round-trip; the FWHT runs IN SHARED (no
/// private-array spill — the measured prep floor). SPECIALIZED for cols=6144
/// (ng=24 groups of 256); mlp.down (cols=16384, 64 KB row > 48 KB shared) keeps
/// the 3-pass fallback.
///
/// One workgroup (32 threads) per row: (0) coalesced load of the row's 6144 f32
/// into a padded shared buffer [24 groups][257] (PAD 257 = conflict-free: lane
/// t hits bank (t*257)%32 = t), (1) threads 0..23 each FWHT + normalize (/16)
/// their group in shared and emit the group abs-max, (2) thread 0 reduces the
/// 24 maxes -> per-row scale (=max/127), (3) all 32 threads coalesced-quantize
/// shared -> packed int8 (4/word). Bindings: 0=x(f32 in [m][6144]),
/// 1=x_i8(u32 out [m][1536]), 2=scale(f32 out [m]). Dispatch m workgroups.
pub fn buildFusedPrepI8(gpa: std.mem.Allocator, cols: u32) ![]align(4) u8 {
    const NG: u32 = cols / 256;
    const PAD: u32 = 257;
    const ROT: u32 = NG * PAD;
    const INV: u32 = ROT + NG; // inv-scale broadcast slot
    const SH_LEN: u32 = INV + 2;
    const WG: u32 = if (NG <= 32) 32 else 64;
    const LOADS: u32 = cols / WG;
    const WORDS: u32 = (cols / 4) / WG;

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    var arena_inst = std.heap.ArenaAllocator.init(gpa);
    defer arena_inst.deinit();
    const ar = arena_inst.allocator();
    const H = struct {
        fn cst(a2: std.mem.Allocator, v: u32) ![]const u8 {
            return std.fmt.allocPrint(a2, "%cu_{d}", .{v});
        }
        fn tmp(a2: std.mem.Allocator, i: u32) ![]const u8 {
            return std.fmt.allocPrint(a2, "%t{d}", .{i});
        }
    };

    var pool: [80]u32 = undefined;
    var pn: usize = 0;
    const addC = struct {
        fn f(p: []u32, n: *usize, v: u32) void {
            for (p[0..n.*]) |ex| if (ex == v) return;
            p[n.*] = v;
            n.* += 1;
        }
    }.f;
    for ([_]u32{ 0, 1, 2, 3, 4, 8, 12, 15, 16, 24, 32, 48, 63, 64, 128, 192, 255, 256, 257, 0xFF, 0x108, NG, ROT, INV, WG, LOADS, WORDS, cols, cols / 4 }) |v|
        addC(&pool, &pn, v);

    // Loop helpers: phi-carried counter loops (open emits the header/cond/body,
    // close emits the increment/back-edge/merge).
    const Loop = struct { head: u32, cont: u32, merge: u32, iv: u32, inx: u32 };
    const openLoop = struct {
        fn f(e: *Emit, tu: []const u8, tb: []const u8, cz: []const u8, count: []const u8, pred: []const u8) !Loop {
            const head = e.id();
            const cont = e.id();
            const merge = e.id();
            const iv = e.id();
            const inx = e.id();
            const cond = e.id();
            const body = e.id();
            try e.line("OpBranch %t{d}", .{head});
            try e.line("%t{d} = OpLabel", .{head});
            try e.line("%t{d} = OpPhi {s} {s} {s} %t{d} %t{d}", .{ iv, tu, cz, pred, inx, cont });
            try e.line("OpLoopMerge %t{d} %t{d} None", .{ merge, cont });
            try e.line("OpBranch %t{d}", .{cond});
            try e.line("%t{d} = OpLabel", .{cond});
            const cmp = e.id();
            try e.line("%t{d} = OpULessThan {s} %t{d} {s}", .{ cmp, tb, iv, count });
            try e.line("OpBranchConditional %t{d} %t{d} %t{d}", .{ cmp, body, merge });
            try e.line("%t{d} = OpLabel", .{body});
            return .{ .head = head, .cont = cont, .merge = merge, .iv = iv, .inx = inx };
        }
    }.f;
    const closeLoop = struct {
        fn f(e: *Emit, tu: []const u8, one: []const u8, L: Loop) !void {
            try e.line("OpBranch %t{d}", .{L.cont});
            try e.line("%t{d} = OpLabel", .{L.cont});
            try e.line("%t{d} = OpIAdd {s} %t{d} {s}", .{ L.inx, tu, L.iv, one });
            try e.line("OpBranch %t{d}", .{L.head});
            try e.line("%t{d} = OpLabel", .{L.merge});
        }
    }.f;

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("%ext = OpExtInstImport \"GLSL.std.450\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vx %vq %vs %vsh", .{});
    try em.line("OpExecutionMode %main LocalSize {d} 1 1", .{WG});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    try em.line("OpDecorate %arr_f32 ArrayStride 4", .{});
    try em.line("OpDecorate %arr_u32 ArrayStride 4", .{});
    for ([_][]const u8{ "sx", "sq", "ss" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }
    for ([_][]const u8{ "vx", "vq", "vs" }, 0..) |v, b| {
        try em.line("OpDecorate %{s} DescriptorSet 0", .{v});
        try em.line("OpDecorate %{s} Binding {d}", .{ v, b });
    }

    // --- types / constants ---
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%s32 = OpTypeInt 32 1", .{});
    try em.line("%f32 = OpTypeFloat 32", .{});
    try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%bool = OpTypeBool", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_f32 = OpTypeArray %f32 %c_arrlen", .{});
    try em.line("%arr_u32 = OpTypeArray %u32 %c_arrlen", .{});
    try em.line("%sx = OpTypeStruct %arr_f32", .{});
    try em.line("%sq = OpTypeStruct %arr_u32", .{});
    try em.line("%ss = OpTypeStruct %arr_f32", .{});
    try em.line("%ptr_sx = OpTypePointer StorageBuffer %sx", .{});
    try em.line("%ptr_sq = OpTypePointer StorageBuffer %sq", .{});
    try em.line("%ptr_ss = OpTypePointer StorageBuffer %ss", .{});
    try em.line("%vx = OpVariable %ptr_sx StorageBuffer", .{});
    try em.line("%vq = OpVariable %ptr_sq StorageBuffer", .{});
    try em.line("%vs = OpVariable %ptr_ss StorageBuffer", .{});
    try em.line("%ptr_sx_f32 = OpTypePointer StorageBuffer %f32", .{});
    try em.line("%ptr_sq_u32 = OpTypePointer StorageBuffer %u32", .{});
    try em.line("%ptr_ss_f32 = OpTypePointer StorageBuffer %f32", .{});
    try em.line("%c_shlen = OpConstant %u32 {d}", .{SH_LEN});
    try em.line("%sh = OpTypeArray %f16 %c_shlen", .{});
    try em.line("%ptr_wg_sh = OpTypePointer Workgroup %sh", .{});
    try em.line("%vsh = OpVariable %ptr_wg_sh Workgroup", .{});
    try em.line("%ptr_wg_f16 = OpTypePointer Workgroup %f16", .{});

    for (pool[0..pn]) |v| try em.line("%cu_{d} = OpConstant %u32 {d}", .{ v, v });
    try em.line("%cf16_inv16 = OpConstant %f16 {d}", .{@as(u32, @as(u16, @bitCast(@as(f16, 1.0 / 16.0))))});
    try em.line("%cf16_0 = OpConstant %f16 0", .{});
    try em.line("%cf_inv127 = OpConstant %f32 {d}", .{@as(u32, @bitCast(@as(f32, 1.0 / 127.0)))});
    try em.line("%cf_tiny = OpConstant %f32 {d}", .{@as(u32, @bitCast(@as(f32, 1e-12)))});
    try em.line("%cf_half = OpConstant %f32 {d}", .{@as(u32, @bitCast(@as(f32, 0.5)))});
    try em.line("%cf_1 = OpConstant %f32 {d}", .{@as(u32, @bitCast(@as(f32, 1.0)))});
    try em.line("%cs_127 = OpConstant %s32 127", .{});
    try em.line("%cs_m127 = OpConstant %s32 {d}", .{@as(u32, @bitCast(@as(i32, -127)))});

    // --- function ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});
    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const row = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ row, gidv });
    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const tid = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ tid, lidv });
    const rowcols = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ rowcols, row, cols });
    var cur: []const u8 = "%entry";

    // Phase 0: coalesced load f32 -> f16 into padded shared.
    {
        const L = try openLoop(&em, "%u32", "%bool", "%cu_0", try H.cst(ar, LOADS), cur);
        const iw = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ iw, L.iv, WG });
        const j = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ j, tid, iw });
        const g = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_8", .{ g, j });
        const l = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_255", .{ l, j });
        const gp = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_257", .{ gp, g });
        const shi = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ shi, gp, l });
        const gi = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gi, rowcols, j });
        const xp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_sx_f32 %vx %cu_0 %t{d}", .{ xp, gi });
        const valf = em.id();
        try em.line("%t{d} = OpLoad %f32 %t{d}", .{ valf, xp });
        const valh = em.id();
        try em.line("%t{d} = OpFConvert %f16 %t{d}", .{ valh, valf });
        const sp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %t{d}", .{ sp, shi });
        try em.line("OpStore %t{d} %t{d}", .{ sp, valh });
        try closeLoop(&em, "%u32", "%cu_1", L);
        cur = try H.tmp(ar, L.merge);
    }
    try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});

    // Phase 1: FWHT (f16) + normalize (tid < NG).
    {
        const sel = em.id();
        const do = em.id();
        const cmp = em.id();
        try em.line("%t{d} = OpULessThan %bool %t{d} %cu_{d}", .{ cmp, tid, NG });
        try em.line("OpSelectionMerge %t{d} None", .{sel});
        try em.line("OpBranchConditional %t{d} %t{d} %t{d}", .{ cmp, do, sel });
        try em.line("%t{d} = OpLabel", .{do});
        var cur1: []const u8 = try H.tmp(ar, do);
        const t257 = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_257", .{ t257, tid });
        inline for (.{ 1, 4, 16, 64 }) |s| {
            const L = try openLoop(&em, "%u32", "%bool", "%cu_0", "%cu_64", cur1);
            var lo: []const u8 = "%cu_0";
            if (s != 1) {
                const x = em.id();
                try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_{d}", .{ x, L.iv, s - 1 });
                lo = try H.tmp(ar, x);
            }
            var hi: []const u8 = try H.tmp(ar, L.iv);
            if (s != 1) {
                const x = em.id();
                try em.line("%t{d} = OpISub %u32 %t{d} {s}", .{ x, L.iv, lo });
                hi = try H.tmp(ar, x);
            }
            const hi4 = em.id();
            try em.line("%t{d} = OpShiftLeftLogical %u32 {s} %cu_2", .{ hi4, hi });
            var p0: []const u8 = try H.tmp(ar, hi4);
            if (s != 1) {
                const x = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} {s}", .{ x, hi4, lo });
                p0 = try H.tmp(ar, x);
            }
            const base = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} {s}", .{ base, t257, p0 });
            const offs = [_][]const u8{ "%cu_0", try H.cst(ar, s), try H.cst(ar, 2 * s), try H.cst(ar, 3 * s) };
            var e: [4]u32 = undefined;
            for (offs, 0..) |off, k| {
                var idx: []const u8 = try H.tmp(ar, base);
                if (k != 0) {
                    const x = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} {s}", .{ x, base, off });
                    idx = try H.tmp(ar, x);
                }
                const pp = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh {s}", .{ pp, idx });
                e[k] = em.id();
                try em.line("%t{d} = OpLoad %f16 %t{d}", .{ e[k], pp });
            }
            const pq = em.id();
            try em.line("%t{d} = OpFAdd %f16 %t{d} %t{d}", .{ pq, e[0], e[1] });
            const qd = em.id();
            try em.line("%t{d} = OpFSub %f16 %t{d} %t{d}", .{ qd, e[2], e[3] });
            const rd = em.id();
            try em.line("%t{d} = OpFSub %f16 %t{d} %t{d}", .{ rd, e[0], e[1] });
            const sd = em.id();
            try em.line("%t{d} = OpFAdd %f16 %t{d} %t{d}", .{ sd, e[2], e[3] });
            var nn: [4]u32 = undefined;
            nn[0] = em.id();
            try em.line("%t{d} = OpFAdd %f16 %t{d} %t{d}", .{ nn[0], pq, qd });
            nn[1] = em.id();
            try em.line("%t{d} = OpFSub %f16 %t{d} %t{d}", .{ nn[1], pq, qd });
            nn[2] = em.id();
            try em.line("%t{d} = OpFAdd %f16 %t{d} %t{d}", .{ nn[2], rd, sd });
            nn[3] = em.id();
            try em.line("%t{d} = OpFSub %f16 %t{d} %t{d}", .{ nn[3], sd, rd });
            for (offs, 0..) |off, k| {
                var idx: []const u8 = try H.tmp(ar, base);
                if (k != 0) {
                    const x = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} {s}", .{ x, base, off });
                    idx = try H.tmp(ar, x);
                }
                const pp = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh {s}", .{ pp, idx });
                try em.line("OpStore %t{d} %t{d}", .{ pp, nn[k] });
            }
            try closeLoop(&em, "%u32", "%cu_1", L);
            cur1 = try H.tmp(ar, L.merge);
        }
        // normalize /16 + abs-max (2-phi loop, f16)
        const nh = em.id();
        const ncond = em.id();
        const nbd = em.id();
        const nct = em.id();
        const nmg = em.id();
        const niv = em.id();
        const ninx = em.id();
        const namax = em.id();
        const namaxn = em.id();
        try em.line("OpBranch %t{d}", .{nh});
        try em.line("%t{d} = OpLabel", .{nh});
        try em.line("%t{d} = OpPhi %u32 %cu_0 {s} %t{d} %t{d}", .{ niv, cur1, ninx, nct });
        try em.line("%t{d} = OpPhi %f16 %cf16_0 {s} %t{d} %t{d}", .{ namax, cur1, namaxn, nct });
        try em.line("OpLoopMerge %t{d} %t{d} None", .{ nmg, nct });
        try em.line("OpBranch %t{d}", .{ncond});
        try em.line("%t{d} = OpLabel", .{ncond});
        const ncmp = em.id();
        try em.line("%t{d} = OpULessThan %bool %t{d} %cu_256", .{ ncmp, niv });
        try em.line("OpBranchConditional %t{d} %t{d} %t{d}", .{ ncmp, nbd, nmg });
        try em.line("%t{d} = OpLabel", .{nbd});
        const nidx = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ nidx, t257, niv });
        const npp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %t{d}", .{ npp, nidx });
        const raw = em.id();
        try em.line("%t{d} = OpLoad %f16 %t{d}", .{ raw, npp });
        const v16 = em.id();
        try em.line("%t{d} = OpFMul %f16 %t{d} %cf16_inv16", .{ v16, raw });
        try em.line("OpStore %t{d} %t{d}", .{ npp, v16 });
        const av = em.id();
        try em.line("%t{d} = OpExtInst %f16 %ext 4 %t{d}", .{ av, v16 });
        try em.line("%t{d} = OpExtInst %f16 %ext 40 %t{d} %t{d}", .{ namaxn, namax, av });
        try em.line("OpBranch %t{d}", .{nct});
        try em.line("%t{d} = OpLabel", .{nct});
        try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_1", .{ ninx, niv });
        try em.line("OpBranch %t{d}", .{nh});
        try em.line("%t{d} = OpLabel", .{nmg});
        const gidx = em.id();
        try em.line("%t{d} = OpIAdd %u32 %cu_{d} %t{d}", .{ gidx, ROT, tid });
        const gpp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %t{d}", .{ gpp, gidx });
        try em.line("OpStore %t{d} %t{d}", .{ gpp, namax });
        try em.line("OpBranch %t{d}", .{sel});
        try em.line("%t{d} = OpLabel", .{sel});
        cur = try H.tmp(ar, sel);
    }
    try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});

    // Phase 2: reduce over gmax[0..NG] -> scale[row], inv (tid == 0).
    {
        const sel = em.id();
        const do = em.id();
        const cmp = em.id();
        try em.line("%t{d} = OpIEqual %bool %t{d} %cu_0", .{ cmp, tid });
        try em.line("OpSelectionMerge %t{d} None", .{sel});
        try em.line("OpBranchConditional %t{d} %t{d} %t{d}", .{ cmp, do, sel });
        try em.line("%t{d} = OpLabel", .{do});
        const mh = em.id();
        const mcond = em.id();
        const mbd = em.id();
        const mct = em.id();
        const mmg = em.id();
        const miv = em.id();
        const minx = em.id();
        const mx = em.id();
        const mxn = em.id();
        try em.line("OpBranch %t{d}", .{mh});
        try em.line("%t{d} = OpLabel", .{mh});
        try em.line("%t{d} = OpPhi %u32 %cu_0 %t{d} %t{d} %t{d}", .{ miv, do, minx, mct });
        try em.line("%t{d} = OpPhi %f16 %cf16_0 %t{d} %t{d} %t{d}", .{ mx, do, mxn, mct });
        try em.line("OpLoopMerge %t{d} %t{d} None", .{ mmg, mct });
        try em.line("OpBranch %t{d}", .{mcond});
        try em.line("%t{d} = OpLabel", .{mcond});
        const mcmp = em.id();
        try em.line("%t{d} = OpULessThan %bool %t{d} %cu_{d}", .{ mcmp, miv, NG });
        try em.line("OpBranchConditional %t{d} %t{d} %t{d}", .{ mcmp, mbd, mmg });
        try em.line("%t{d} = OpLabel", .{mbd});
        const gidx = em.id();
        try em.line("%t{d} = OpIAdd %u32 %cu_{d} %t{d}", .{ gidx, ROT, miv });
        const gpp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %t{d}", .{ gpp, gidx });
        const gv = em.id();
        try em.line("%t{d} = OpLoad %f16 %t{d}", .{ gv, gpp });
        try em.line("%t{d} = OpExtInst %f16 %ext 40 %t{d} %t{d}", .{ mxn, mx, gv });
        try em.line("OpBranch %t{d}", .{mct});
        try em.line("%t{d} = OpLabel", .{mct});
        try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_1", .{ minx, miv });
        try em.line("OpBranch %t{d}", .{mh});
        try em.line("%t{d} = OpLabel", .{mmg});
        const mxf = em.id();
        try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ mxf, mx });
        const sc0 = em.id();
        try em.line("%t{d} = OpFMul %f32 %t{d} %cf_inv127", .{ sc0, mxf });
        const sc = em.id();
        try em.line("%t{d} = OpExtInst %f32 %ext 40 %t{d} %cf_tiny", .{ sc, sc0 });
        const scp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_ss_f32 %vs %cu_0 %t{d}", .{ scp, row });
        try em.line("OpStore %t{d} %t{d}", .{ scp, sc });
        const invf = em.id();
        try em.line("%t{d} = OpFDiv %f32 %cf_1 %t{d}", .{ invf, sc });
        const invh = em.id();
        try em.line("%t{d} = OpFConvert %f16 %t{d}", .{ invh, invf });
        const ip = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %cu_{d}", .{ ip, INV });
        try em.line("OpStore %t{d} %t{d}", .{ ip, invh });
        try em.line("OpBranch %t{d}", .{sel});
        try em.line("%t{d} = OpLabel", .{sel});
        cur = try H.tmp(ar, sel);
    }
    try em.line("OpControlBarrier %cu_2 %cu_2 %cu_264", .{});

    // Phase 3: coalesced quantize (f16 shared -> f32 math -> packed int8).
    {
        const ip = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh %cu_{d}", .{ ip, INV });
        const invh = em.id();
        try em.line("%t{d} = OpLoad %f16 %t{d}", .{ invh, ip });
        const inv = em.id();
        try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ inv, invh });
        const roww = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ roww, row, cols / 4 });
        const L = try openLoop(&em, "%u32", "%bool", "%cu_0", try H.cst(ar, WORDS), cur);
        const iw = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_{d}", .{ iw, L.iv, WG });
        const word = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ word, tid, iw });
        const be = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %cu_2", .{ be, word });
        const g = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %cu_8", .{ g, be });
        const l = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_255", .{ l, be });
        const gp = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %cu_257", .{ gp, g });
        const base = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ base, gp, l });
        var out: []const u8 = "%cu_0";
        inline for (0..4) |k| {
            var idx: []const u8 = try H.tmp(ar, base);
            if (k != 0) {
                const x = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %cu_{d}", .{ x, base, k });
                idx = try H.tmp(ar, x);
            }
            const pp = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vsh {s}", .{ pp, idx });
            const vh = em.id();
            try em.line("%t{d} = OpLoad %f16 %t{d}", .{ vh, pp });
            const v = em.id();
            try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ v, vh });
            const r = em.id();
            try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ r, v, inv });
            const sgn = em.id();
            try em.line("%t{d} = OpExtInst %f32 %ext 6 %t{d}", .{ sgn, r });
            const hh = em.id();
            try em.line("%t{d} = OpFMul %f32 %t{d} %cf_half", .{ hh, sgn });
            const added = em.id();
            try em.line("%t{d} = OpFAdd %f32 %t{d} %t{d}", .{ added, r, hh });
            const qi = em.id();
            try em.line("%t{d} = OpConvertFToS %s32 %t{d}", .{ qi, added });
            const qc = em.id();
            try em.line("%t{d} = OpExtInst %s32 %ext 45 %t{d} %cs_m127 %cs_127", .{ qc, qi });
            const qub = em.id();
            try em.line("%t{d} = OpBitcast %u32 %t{d}", .{ qub, qc });
            const qb = em.id();
            try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %cu_255", .{ qb, qub });
            const shk = em.id();
            try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %cu_{d}", .{ shk, qb, 8 * k });
            const no = em.id();
            try em.line("%t{d} = OpBitwiseOr %u32 {s} %t{d}", .{ no, out, shk });
            out = try H.tmp(ar, no);
        }
        const wi = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ wi, roww, word });
        const qp = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_sq_u32 %vq %cu_0 %t{d}", .{ qp, wi });
        try em.line("OpStore %t{d} {s}", .{ qp, out });
        try closeLoop(&em, "%u32", "%cu_1", L);
        cur = try H.tmp(ar, L.merge);
    }
    _ = &cur;

    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}
/// Shared-memory staged variant with fused fp8 decode and double-buffered
/// k sub-slabs: LocalSize (32,4) = 4 subgroups; each workgroup computes a
/// 128(m) x 128(n) tile of C, stepping k by 64 as two 32-deep sub-slabs.
/// BOTH operands stage through one workgroup f16 array: B decodes into
/// [0, 8192) (2 x 32x128), A copies into [8192, 16384) (2 x 128x32) — 32 KB
/// total, still 3 workgroups/SM by shared. Each half-step, the 128 threads
/// first issue the uvec4 global loads for the *next* sub-slab (2 raw-e4m3
/// quads for B, 4 f16 quads for A), then run the MMAs against the current
/// one (all fragment loads ldmatrix from shared), then store the loaded
/// words into the other buffer — the global-load latency hides under
/// tensor-core work. 2 barriers per 64 k. The dequant scale is folded into
/// A by the caller (f32_to_h16). Bindings: 0 = A viewed as uvec4 (staging
/// loads), 2 = C (f32), 3 = B (raw fp8 viewed as uvec4).
/// Push: {m, n, k, stride_bytes} u32; n, stride multiples of 128, k a
/// multiple of 64, m of 128.
/// GEMM occupancy experiment (lever 1) toggles for the fp8 coop pipeline;
/// the f16-weight (VAE) pipeline always builds 4-warp/f32-acc since its
/// padded widths aren't 256-multiples. coop_wgn is the wg tile N width the
/// dispatch must divide rows by.
pub const coop_warps8 = true;
pub const coop_acc_h16 = true;
pub const coop_wgn: u32 = if (coop_warps8) 256 else 128;

/// Lanes per subgroup that every coop kernel in this file is authored for:
/// each shader declares `LocalSize (subgroup_lanes, NWARPS, 1)` so one warp
/// (the X dimension) maps to exactly one subgroup, and the cooperative-matrix
/// fragments are distributed across those lanes. The value is a property of
/// how these kernels are written (a "warp" is 32 wide, the NVIDIA/PTX model),
/// NOT of any particular GPU. A device whose native subgroup size differs
/// (AMD RADV runs wave64) must have its coop pipelines pinned to this size via
/// VK_EXT_subgroup_size_control, or the fragments span the wrong lanes and the
/// GEMM returns zeros. Single source of truth for the pin in context.zig.
pub const subgroup_lanes: u32 = 32;

/// `b_f16 = false`: B is raw k-major e4m3 (the DiT weights), SWAR-decoded
/// to f16 in the staging loop. `b_f16 = true`: B is pre-converted k-major
/// f16 (the VAE conv weights, zero-padded to rows%128 / cols%64 on the
/// CPU) and staging is a plain uvec4 copy like the A slab. Everything else
/// (tiling, double buffering, MMA section, C store) is identical.
///
/// `warps8`: 2x4 warp grid over a 128x256 wg tile (LocalSize (32,8), B
/// slabs double to 32 KB -> 48 KB total shared, rows must be %256) instead
/// of the 2x2 grid over 128x128 (32 KB, 3 wgs/SM). At ~165 regs/thread the
/// wide tile is register-gated (occupancy experiment — lever 1); pair with
/// `acc_h16` (f16 accumulators, converted to f32 at the C store) to fit
/// 2 wgs/SM at <= 128 regs. acc_h16 changes accumulation NUMERICS — gate
/// any keep on the DiT parity fixture.
pub fn buildGemmShared(gpa: std.mem.Allocator, b_f16: bool, warps8: bool, acc_h16: bool, c_h16: bool, bf16: bool) ![]align(4) u8 {
    // bf16 mode: the A/B operands (activations + weights) are bfloat16 instead
    // of IEEE f16 — same 16-bit storage, so the staging is a bitwise copy and
    // only the type annotations change. Requires b_f16 (weight already 16-bit,
    // no fp8 dequant) and f32 accumulate. ` E`/`v2E` name the A/B element type.
    std.debug.assert(!bf16 or (b_f16 and !acc_h16 and !c_h16));
    const E = if (bf16) "bf16" else "f16";
    const v2E = if (bf16) "v2bf16" else "v2f16";
    // c_h16 stores C half-precision (binding 2 becomes an f16 array). Only
    // meaningful with f16 accumulators, where it is exact: the f32 store it
    // replaces was just a widening of the f16 accumulator values.
    std.debug.assert(!c_h16 or acc_h16);
    // A slab geometry: 128 rows x 32 k f16 per sub-slab at base A_BASE of
    // the shared array. The row stride can carry padding to spread shared-
    // memory banks, but stride 34 measured neutral vs 32 (the driver's coop
    // loads evidently swizzle already), so no padding.
    const A_STRIDE: u32 = 32;
    const A_SLAB: u32 = 128 * A_STRIDE;
    // Workgroup geometry. THREADS == WGN in both configs, which keeps the
    // B staging at exactly 2 sixteen-element chunks per thread.
    const WGN: u32 = if (warps8) 256 else 128; // wg tile N width
    const THREADS: u32 = if (warps8) 256 else 128;
    const NWARPS: u32 = if (warps8) 8 else 4;
    const B_SLAB: u32 = 32 * WGN; // one 32-deep k sub-slab
    const A_BASE: u32 = 2 * B_SLAB;
    const AQ: usize = 512 / THREADS; // A staging quads per thread (512 uvec4/sub-slab)

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    // Derived constant/index names are formatted from indices; allocated ones
    // are tracked here and freed at the end.
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |s| gpa.free(s);
        names.deinit(gpa);
    }
    const Nm = struct {
        nl: *std.ArrayList([]u8),
        g: std.mem.Allocator,
        fn f(self: @This(), comptime fmt: []const u8, args: anytype) ![]const u8 {
            const s = try std.fmt.allocPrint(self.g, fmt, args);
            try self.nl.append(self.g, s);
            return s;
        }
    };
    const nm = Nm{ .nl = &names, .g = gpa };

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpCapability CooperativeMatrixKHR", .{});
    if (bf16) {
        try em.line("OpCapability BFloat16TypeKHR", .{});
        try em.line("OpCapability BFloat16CooperativeMatrixKHR", .{});
    }
    try em.line("OpExtension \"SPV_KHR_cooperative_matrix\"", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("OpExtension \"SPV_KHR_16bit_storage\"", .{});
    if (bf16) try em.line("OpExtension \"SPV_KHR_bfloat16\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vc %vb4 %va4 %vpush %vbsh", .{});
    try em.line("OpExecutionMode %main LocalSize 32 {d} 1", .{NWARPS});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    try em.line("OpDecorate %arr_c ArrayStride {d}", .{@as(u32, if (c_h16) 2 else 4)});
    try em.line("OpDecorate %sc Block", .{});
    try em.line("OpMemberDecorate %sc 0 Offset 0", .{});
    try em.line("OpDecorate %push Block", .{});
    for (0..4) |m| try em.line("OpMemberDecorate %push {d} Offset {d}", .{ m, m * 4 });
    try em.line("OpDecorate %vc DescriptorSet 0", .{});
    try em.line("OpDecorate %vc Binding 2", .{});
    try em.line("OpDecorate %vb4 DescriptorSet 0", .{});
    try em.line("OpDecorate %vb4 Binding 3", .{});
    try em.line("OpDecorate %va4 DescriptorSet 0", .{});
    try em.line("OpDecorate %va4 Binding 0", .{});
    try em.line("OpDecorate %arr_v4 ArrayStride 16", .{});
    for ([_][]const u8{ "sb4", "sa4" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }

    // --- types ---
    const c_elem = if (c_h16) "f16" else "f32";
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%f32 = OpTypeFloat 32", .{});
    try em.line("%bool = OpTypeBool", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_c = OpTypeArray %{s} %c_arrlen", .{c_elem});
    try em.line("%v4u32 = OpTypeVector %u32 4", .{});
    try em.line("%arr_v4 = OpTypeArray %v4u32 %c_arrlen", .{});
    try em.line("%sb4 = OpTypeStruct %arr_v4", .{});
    try em.line("%ptr_sb4 = OpTypePointer StorageBuffer %sb4", .{});
    try em.line("%vb4 = OpVariable %ptr_sb4 StorageBuffer", .{});
    try em.line("%ptr_sb4_v4 = OpTypePointer StorageBuffer %v4u32", .{});
    try em.line("%sa4 = OpTypeStruct %arr_v4", .{});
    try em.line("%ptr_sa4 = OpTypePointer StorageBuffer %sa4", .{});
    try em.line("%va4 = OpVariable %ptr_sa4 StorageBuffer", .{});
    try em.line("%sc = OpTypeStruct %arr_c", .{});
    try em.line("%ptr_sc = OpTypePointer StorageBuffer %sc", .{});
    try em.line("%vc = OpVariable %ptr_sc StorageBuffer", .{});
    try em.line("%push = OpTypeStruct %u32 %u32 %u32 %u32", .{});
    try em.line("%ptr_push = OpTypePointer PushConstant %push", .{});
    try em.line("%vpush = OpVariable %ptr_push PushConstant", .{});
    try em.line("%ptr_pc_u32 = OpTypePointer PushConstant %u32", .{});
    try em.line("%ptr_sc_f32 = OpTypePointer StorageBuffer %{s}", .{c_elem});

    // Workgroup slab (no layout decorations): B 2 x [32][WGN] f16 at 0,
    // A 2 x [128][32] f16 at A_BASE.
    if (bf16) try em.line("%bf16 = OpTypeFloat 16 BFloat16KHR", .{});
    try em.line("%c_bsh_len = OpConstant %u32 {d}", .{A_BASE + 2 * A_SLAB});
    try em.line("%t_bsh = OpTypeArray %{s} %c_bsh_len", .{E});
    try em.line("%ptr_wg_bsh = OpTypePointer Workgroup %t_bsh", .{});
    try em.line("%vbsh = OpVariable %ptr_wg_bsh Workgroup", .{});
    try em.line("%ptr_wg_f16 = OpTypePointer Workgroup %{s}", .{E});

    // Named u32 constant block (mirrors the original — note duplicate values
    // like c_u3 vs c_scope_sub are intentional and must be preserved).
    for ([_]struct { []const u8, u32 }{
        .{ "c_u0", 0 },   .{ "c_u1", 1 },   .{ "c_u2", 2 },     .{ "c_u3", 3 },
        .{ "c_u4", 4 },   .{ "c_u5", 5 },   .{ "c_u6", 6 },     .{ "c_u7", 7 },
        .{ "c_u8", 8 },   .{ "c_u12", 12 }, .{ "c_u31", 31 },   .{ "c_u16", 16 },
        .{ "c_u32c", 32 }, .{ "c_u64", 64 }, .{ "c_u128", 128 }, .{ "c_u4096", 4096 },
        .{ "c_mag_mask", 0x007F007F }, .{ "c_sgn_mask", 0x00800080 }, .{ "c_u264", 0x108 },
        .{ "c_scope_sub", 3 }, .{ "c_scope_wg", 2 },
    }) |cv| try em.line("%{s} = OpConstant %u32 {d}", .{ cv[0], cv[1] });

    const c_acc_elem = if (acc_h16) "f16" else "f32";
    try em.line("%mat_a = OpTypeCooperativeMatrixKHR %{s} %c_scope_sub %c_u16 %c_u16 %c_u0", .{E});
    try em.line("%mat_b = OpTypeCooperativeMatrixKHR %{s} %c_scope_sub %c_u16 %c_u16 %c_u1", .{E});
    try em.line("%mat_c = OpTypeCooperativeMatrixKHR %{s} %c_scope_sub %c_u16 %c_u16 %c_u2", .{c_acc_elem});
    if (acc_h16 and !c_h16) try em.line("%mat_c32 = OpTypeCooperativeMatrixKHR %f32 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%v2f16 = OpTypeVector %f16 2", .{});
    if (bf16) try em.line("%v2bf16 = OpTypeVector %bf16 2", .{});
    try em.line("%c_f32_0 = OpConstant %f32 0", .{});
    if (acc_h16) try em.line("%c_f16_0 = OpConstant %f16 0", .{});
    try em.line("%c_h256 = OpConstant %f16 23552", .{}); // 0x5C00
    try em.line("%c_v2_256 = OpConstantComposite %v2f16 %c_h256 %c_h256", .{});
    try em.line("%c_acc0 = OpConstantComposite %mat_c %{s}", .{if (acc_h16) "c_f16_0" else "c_f32_0"});

    // Body index constants (with the original's value-aliasing so emission
    // order and set match golden exactly).
    const c_wgn = if (WGN == 128) "c_u128" else blk: {
        try em.line("%c_wgn = OpConstant %u32 {d}", .{WGN});
        break :blk "c_wgn";
    };
    const c_bslab = if (B_SLAB == 4096) "c_u4096" else blk: {
        try em.line("%c_bslab = OpConstant %u32 {d}", .{B_SLAB});
        break :blk "c_bslab";
    };
    const c_bmask = if (WGN == 128) "c_u7" else blk: {
        try em.line("%c_bmask = OpConstant %u32 {d}", .{WGN / 16 - 1});
        break :blk "c_bmask";
    };
    const c_bshift = if (WGN == 128) "c_u3" else "c_u4"; // log2(WGN/16)
    var c_stage: [4][]const u8 = undefined; // t*THREADS
    c_stage[0] = "c_u0";
    c_stage[1] = c_wgn;
    for (2..4) |i| {
        try em.line("%c_stage_{d} = OpConstant %u32 {d}", .{ i, @as(u32, @intCast(i)) * THREADS });
        c_stage[i] = try nm.f("c_stage_{d}", .{i});
    }
    var c_k16: [4][]const u8 = undefined;
    c_k16[0] = "c_u0";
    c_k16[1] = "c_u16";
    c_k16[2] = "c_u32c";
    try em.line("%c_k16_3 = OpConstant %u32 48", .{});
    c_k16[3] = "c_k16_3";
    var c_col16: [8][]const u8 = undefined;
    c_col16[0] = "c_u0";
    c_col16[1] = "c_u16";
    c_col16[2] = "c_u32c";
    c_col16[3] = "c_k16_3";
    c_col16[4] = "c_u64";
    for (5..8) |i| {
        try em.line("%c_col16_{d} = OpConstant %u32 {d}", .{ i, @as(u32, @intCast(i)) * 16 });
        c_col16[i] = try nm.f("c_col16_{d}", .{i});
    }
    var c_bt: [2][2][8][]const u8 = undefined;
    for (0..2) |par| for (0..2) |ks| for (0..8) |nt| {
        if (par == 0 and ks == 0) {
            c_bt[par][ks][nt] = c_col16[nt];
        } else {
            try em.line("%c_bt_{d}_{d}_{d} = OpConstant %u32 {d}", .{ par, ks, nt, @as(u32, @intCast(par * B_SLAB + ks * (16 * WGN) + nt * 16)) });
            c_bt[par][ks][nt] = try nm.f("c_bt_{d}_{d}_{d}", .{ par, ks, nt });
        }
    };
    const c_astride = if (A_STRIDE == 32) "c_u32c" else blk: {
        try em.line("%c_astride = OpConstant %u32 {d}", .{A_STRIDE});
        break :blk "c_astride";
    };
    const c_aslab = if (A_SLAB == 4096) "c_u4096" else blk: {
        try em.line("%c_aslab = OpConstant %u32 {d}", .{A_SLAB});
        break :blk "c_aslab";
    };
    try em.line("%c_ash0 = OpConstant %u32 {d}", .{A_BASE});
    var c_at: [2][2][4][]const u8 = undefined;
    for (0..2) |par| for (0..2) |ks| for (0..4) |r| {
        if (par == 0 and ks == 0 and r == 0) {
            c_at[par][ks][r] = "c_ash0";
        } else {
            try em.line("%c_at_{d}_{d}_{d} = OpConstant %u32 {d}", .{ par, ks, r, @as(u32, @intCast(A_BASE + par * A_SLAB + r * 16 * A_STRIDE + ks * 16)) });
            c_at[par][ks][r] = try nm.f("c_at_{d}_{d}_{d}", .{ par, ks, r });
        }
    };

    // --- function ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});

    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const tile_c = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ tile_c, gidv });
    const tile_r = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ tile_r, gidv });
    const col0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %{s}", .{ col0, tile_c, c_wgn });
    const row0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ row0, tile_r });

    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const lx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ lx, lidv });
    const ly = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ ly, lidv });
    const lymul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u32c", .{ lymul, ly });
    const flat = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ flat, lymul, lx });

    const pnames = [_][]const u8{ "pm", "pn", "pk", "pstride" };
    for (0..4) |m| {
        const pptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_pc_u32 %vpush %c_u{d}", .{ pptr, m });
        try em.line("%{s} = OpLoad %u32 %t{d}", .{ pnames[m], pptr });
    }

    const warp_m = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u1", .{ warp_m, ly });
    const warp_n = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ warp_n, ly });
    const wm64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wm64, warp_m });
    const row_s = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ row_s, row0, wm64 });
    const wn64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wn64, warp_n });
    const col_s = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ col_s, col0, wn64 });
    const a_shbase = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %{s}", .{ a_shbase, wm64, c_astride });

    // Loop-invariant B staging indices.
    var brow_t: [2]u32 = undefined;
    var bco_t: [2]u32 = undefined;
    var sbase0_t: [2]u32 = undefined;
    var sbase1_t: [2]u32 = undefined;
    for (0..2) |i| {
        var v = flat;
        if (i > 0) {
            const vn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ vn, flat, c_wgn });
            v = vn;
        }
        brow_t[i] = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %{s}", .{ brow_t[i], v, c_bshift });
        const vmod = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %{s}", .{ vmod, v, c_bmask });
        const bcol16 = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u4", .{ bcol16, vmod });
        bco_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ bco_t[i], col0, bcol16 });
        sbase0_t[i] = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u4", .{ sbase0_t[i], v });
        sbase1_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ sbase1_t[i], sbase0_t[i], c_bslab });
    }

    // Loop-invariant A staging indices.
    var a_inv_t: [4]u32 = undefined;
    var asb0_t: [4]u32 = undefined;
    var asb1_t: [4]u32 = undefined;
    for (0..AQ) |i| {
        var q = flat;
        if (i > 0) {
            const qn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ qn, flat, c_stage[i] });
            q = qn;
        }
        const arow = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u2", .{ arow, q });
        const aqc = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u3", .{ aqc, q });
        const grow = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ grow, row0, arow });
        const gmul = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %pk", .{ gmul, grow });
        const gq = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u3", .{ gq, gmul });
        a_inv_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ a_inv_t[i], gq, aqc });
        const srow = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %{s}", .{ srow, arow, c_astride });
        const scol = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u3", .{ scol, aqc });
        const s0 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ s0, srow, scol });
        asb0_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %c_ash0", .{ asb0_t[i], s0 });
        asb1_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ asb1_t[i], asb0_t[i], c_aslab });
    }

    // Staging emitters (loads separated from decode+store).
    const Stage = struct {
        em: *Emit,
        b_f16: bool,
        e_type: []const u8, // A/B element: "f16" or "bf16"
        v2_type: []const u8, // 2-vector of the above
        c_v2_256: []const u8,
        c_j: [4][]const u8,
        c_j8: [8][]const u8,
        c_w4: [4][]const u8,
        brow: [2]u32,
        bco: [2]u32,

        fn loads(st: @This(), kb: []const u8) ![4]u32 {
            const e = st.em;
            var quads: [4]u32 = undefined;
            for (0..2) |i| {
                const bk = e.id();
                try e.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ bk, kb, st.brow[i] });
                const bmul = e.id();
                try e.line("%t{d} = OpIMul %u32 %t{d} %pstride", .{ bmul, bk });
                const boff = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ boff, bmul, st.bco[i] });
                if (st.b_f16) {
                    const qidx0 = e.id();
                    try e.line("%t{d} = OpShiftRightLogical %u32 %t{d} %{s}", .{ qidx0, boff, st.c_j[3] }); // /8
                    for (0..2) |qi| {
                        var qidx = qidx0;
                        if (qi > 0) {
                            const qn = e.id();
                            try e.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ qn, qidx0, st.c_j[1] });
                            qidx = qn;
                        }
                        const qptr = e.id();
                        try e.line("%t{d} = OpAccessChain %ptr_sb4_v4 %vb4 %c_u0 %t{d}", .{ qptr, qidx });
                        quads[2 * i + qi] = e.id();
                        try e.line("%t{d} = OpLoad %v4u32 %t{d}", .{ quads[2 * i + qi], qptr });
                    }
                } else {
                    const qidx = e.id();
                    try e.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u4", .{ qidx, boff }); // /16
                    const qptr = e.id();
                    try e.line("%t{d} = OpAccessChain %ptr_sb4_v4 %vb4 %c_u0 %t{d}", .{ qptr, qidx });
                    quads[i] = e.id();
                    try e.line("%t{d} = OpLoad %v4u32 %t{d}", .{ quads[i], qptr });
                }
            }
            return quads;
        }

        fn stores(st: @This(), quads: [4]u32, sbase: [2]u32) !void {
            const e = st.em;
            if (st.b_f16) {
                for (0..2) |i| {
                    for (0..2) |qi| {
                        var qbase = sbase[i];
                        if (qi > 0) {
                            const qb = e.id();
                            try e.line("%t{d} = OpIAdd %u32 %t{d} %c_u8", .{ qb, sbase[i] });
                            qbase = qb;
                        }
                        for (0..4) |wi| {
                            const wv = e.id();
                            try e.line("%t{d} = OpCompositeExtract %u32 %t{d} {d}", .{ wv, quads[2 * i + qi], wi });
                            const hv2 = e.id();
                            try e.line("%t{d} = OpBitcast %{s} %t{d}", .{ hv2, st.v2_type, wv });
                            for (0..2) |j| {
                                const hval = e.id();
                                try e.line("%t{d} = OpCompositeExtract %{s} %t{d} {d}", .{ hval, st.e_type, hv2, j });
                                var eidx = qbase;
                                if (wi * 2 + j > 0) {
                                    const ei = e.id();
                                    try e.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ei, qbase, st.c_j8[wi * 2 + j] });
                                    eidx = ei;
                                }
                                const bsptr = e.id();
                                try e.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ bsptr, eidx });
                                try e.line("OpStore %t{d} %t{d}", .{ bsptr, hval });
                            }
                        }
                    }
                }
                return;
            }
            for (0..2) |tq| {
                for (0..4) |wi| {
                    const wv = e.id();
                    try e.line("%t{d} = OpCompositeExtract %u32 %t{d} {d}", .{ wv, quads[tq], wi });
                    var pair_vals: [2]u32 = undefined;
                    for (0..2) |half| {
                        var src = wv;
                        if (half == 1) {
                            const shd = e.id();
                            try e.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u8", .{ shd, wv });
                            src = shd;
                        }
                        const magp = e.id();
                        try e.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_mag_mask", .{ magp, src });
                        const sgnp = e.id();
                        try e.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_sgn_mask", .{ sgnp, src });
                        const mag_sh = e.id();
                        try e.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u7", .{ mag_sh, magp });
                        const sgn_sh = e.id();
                        try e.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u8", .{ sgn_sh, sgnp });
                        const hbits = e.id();
                        try e.line("%t{d} = OpBitwiseOr %u32 %t{d} %t{d}", .{ hbits, mag_sh, sgn_sh });
                        const hv2 = e.id();
                        try e.line("%t{d} = OpBitcast %v2f16 %t{d}", .{ hv2, hbits });
                        pair_vals[half] = e.id();
                        try e.line("%t{d} = OpFMul %v2f16 %t{d} %{s}", .{ pair_vals[half], hv2, st.c_v2_256 });
                    }
                    var wbase = sbase[tq];
                    if (wi > 0) {
                        const wb = e.id();
                        try e.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ wb, sbase[tq], st.c_w4[wi] });
                        wbase = wb;
                    }
                    for (0..4) |j| {
                        const hval = e.id();
                        try e.line("%t{d} = OpCompositeExtract %f16 %t{d} {d}", .{ hval, pair_vals[j & 1], j >> 1 });
                        var eidx = wbase;
                        if (j > 0) {
                            const ei = e.id();
                            try e.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ei, wbase, st.c_j[j] });
                            eidx = ei;
                        }
                        const bsptr = e.id();
                        try e.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ bsptr, eidx });
                        try e.line("OpStore %t{d} %t{d}", .{ bsptr, hval });
                    }
                }
            }
        }
    };
    const stage: Stage = .{
        .em = &em,
        .b_f16 = b_f16,
        .e_type = E,
        .v2_type = v2E,
        .c_v2_256 = "c_v2_256",
        .c_j = .{ "c_u0", "c_u1", "c_u2", "c_u3" },
        .c_j8 = .{ "c_u0", "c_u1", "c_u2", "c_u3", "c_u4", "c_u5", "c_u6", "c_u7" },
        .c_w4 = .{ "c_u0", "c_u4", "c_u8", "c_u12" },
        .brow = brow_t,
        .bco = bco_t,
    };

    // A staging: plain f16 copy into the shared A region.
    const AStage = struct {
        em: *Emit,
        aq: usize,
        e_type: []const u8,
        v2_type: []const u8,
        c_j: [8][]const u8,
        a_inv: [4]u32,

        fn loads(st: @This(), kb: []const u8) ![4]u32 {
            const e = st.em;
            const kbq = e.id();
            try e.line("%t{d} = OpShiftRightLogical %u32 %{s} %c_u3", .{ kbq, kb }); // kb/8
            var quads: [4]u32 = undefined;
            for (0..st.aq) |i| {
                const qidx = e.id();
                try e.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qidx, st.a_inv[i], kbq });
                const qptr = e.id();
                try e.line("%t{d} = OpAccessChain %ptr_sb4_v4 %va4 %c_u0 %t{d}", .{ qptr, qidx });
                quads[i] = e.id();
                try e.line("%t{d} = OpLoad %v4u32 %t{d}", .{ quads[i], qptr });
            }
            return quads;
        }

        fn stores(st: @This(), quads: [4]u32, sbase: [4]u32) !void {
            const e = st.em;
            for (0..st.aq) |i| {
                for (0..4) |wi| {
                    const wv = e.id();
                    try e.line("%t{d} = OpCompositeExtract %u32 %t{d} {d}", .{ wv, quads[i], wi });
                    const hv2 = e.id();
                    try e.line("%t{d} = OpBitcast %{s} %t{d}", .{ hv2, st.v2_type, wv });
                    for (0..2) |j| {
                        const hval = e.id();
                        try e.line("%t{d} = OpCompositeExtract %{s} %t{d} {d}", .{ hval, st.e_type, hv2, j });
                        var eidx = sbase[i];
                        if (wi * 2 + j > 0) {
                            const ei = e.id();
                            try e.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ei, sbase[i], st.c_j[wi * 2 + j] });
                            eidx = ei;
                        }
                        const bsptr = e.id();
                        try e.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ bsptr, eidx });
                        try e.line("OpStore %t{d} %t{d}", .{ bsptr, hval });
                    }
                }
            }
        }
    };
    const astage: AStage = .{
        .em = &em,
        .aq = AQ,
        .e_type = E,
        .v2_type = v2E,
        .c_j = .{ "c_u0", "c_u1", "c_u2", "c_u3", "c_u4", "c_u5", "c_u6", "c_u7" },
        .a_inv = a_inv_t,
    };

    // Prologue: fill sub-slab 0 with the first 32 k-rows/columns.
    {
        const w0 = try stage.loads("c_u0");
        const x0 = try astage.loads("c_u0");
        try stage.stores(w0, sbase0_t);
        try astage.stores(x0, asb0_t);
    }

    // Pre-allocated ids for the loop-carried values (phi back-edges).
    const k0n = em.id();
    var acc_next: [4][4]u32 = undefined;
    for (&acc_next) |*row| for (row) |*v| {
        v.* = em.id();
    };

    try em.line("OpBranch %head", .{});
    try em.line("%head = OpLabel", .{});
    const k0v = em.id();
    try em.line("%t{d} = OpPhi %u32 %c_u0 %entry %t{d} %cont", .{ k0v, k0n });
    var acc_phi: [4][4]u32 = undefined;
    for (&acc_phi, acc_next) |*prow, nrow| {
        for (prow, nrow) |*ap, an| {
            ap.* = em.id();
            try em.line("%t{d} = OpPhi %mat_c %c_acc0 %entry %t{d} %cont", .{ ap.*, an });
        }
    }
    try em.line("OpLoopMerge %merge %cont None", .{});
    try em.line("OpBranch %cond", .{});
    try em.line("%cond = OpLabel", .{});
    const cmp = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %pk", .{ cmp, k0v });
    try em.line("OpBranchConditional %t{d} %body %merge", .{cmp});

    try em.line("%body = OpLabel", .{});
    var acc_cur = acc_phi;
    // Half-step 0: consume sub-slab 0, prefetch k0+32 into sub-slab 1.
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
    const kb1 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u32c", .{ kb1, k0v });
    const kb1n = try nm.f("t{d}", .{kb1});
    const w1 = try stage.loads(kb1n);
    const x1 = try astage.loads(kb1n);
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..4) |r| {
            const aoff = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ aoff, a_shbase, c_at[0][ks][r] });
            const aptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ aptr, aoff });
            ma[r] = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %{s}", .{ ma[r], aptr, c_astride });
        }
        for (0..4) |nt| {
            const boff = em.id();
            try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ boff, c_bt[0][ks][nt], wn64 });
            const bptr2 = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ bptr2, boff });
            const mb = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %{s}", .{ mb, bptr2, c_wgn });
            for (0..4) |r| {
                const acc_out = em.id();
                try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %t{d}", .{ acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try stage.stores(w1, sbase1_t);
    try astage.stores(x1, asb1_t);
    // Half-step 1: consume sub-slab 1, prefetch k0+64 (clamped) into sub-slab 0.
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
    const kb2 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u64", .{ kb2, k0v });
    const kb2_ok = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %pk", .{ kb2_ok, kb2 });
    const kb2s = em.id();
    try em.line("%t{d} = OpSelect %u32 %t{d} %t{d} %c_u0", .{ kb2s, kb2_ok, kb2 });
    const kb2sn = try nm.f("t{d}", .{kb2s});
    const w2 = try stage.loads(kb2sn);
    const x2 = try astage.loads(kb2sn);
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..4) |r| {
            const aoff = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ aoff, a_shbase, c_at[1][ks][r] });
            const aptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ aptr, aoff });
            ma[r] = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %{s}", .{ ma[r], aptr, c_astride });
        }
        for (0..4) |nt| {
            const boff = em.id();
            try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ boff, c_bt[1][ks][nt], wn64 });
            const bptr2 = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vbsh %t{d}", .{ bptr2, boff });
            const mb = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %{s}", .{ mb, bptr2, c_wgn });
            for (0..4) |r| {
                const acc_out = if (ks == 1) acc_next[r][nt] else em.id();
                try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %t{d}", .{ acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try stage.stores(w2, sbase0_t);
    try astage.stores(x2, asb0_t);
    try em.line("OpBranch %cont", .{});

    try em.line("%cont = OpLabel", .{});
    try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u64", .{ k0n, k0v });
    try em.line("OpBranch %head", .{});

    // merge: store 4x4 C tiles at (row_s + r*16, col_s + nt*16)
    try em.line("%merge = OpLabel", .{});
    const rn16 = em.id();
    try em.line("%t{d} = OpIMul %u32 %c_u16 %pn", .{rn16});
    var c_rowmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %pn", .{ c_rowmul, row_s });
    for (0..4) |r| {
        if (r > 0) {
            const nx = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ nx, c_rowmul, rn16 });
            c_rowmul = nx;
        }
        for (0..4) |nt| {
            const ccol = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ccol, col_s, c_col16[nt] });
            const cbase = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ cbase, c_rowmul, ccol });
            const cptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_sc_f32 %vc %c_u0 %t{d}", .{ cptr, cbase });
            var cval = acc_phi[r][nt];
            if (acc_h16 and !c_h16) {
                const cv = em.id();
                try em.line("%t{d} = OpFConvert %mat_c32 %t{d}", .{ cv, cval });
                cval = cv;
            }
            try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u0 %pn", .{ cptr, cval });
        }
    }
    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}

/// K staging for buildGemmScores: measured NEUTRAL at DiT/VAE shapes
/// (scores ~700 ms/step at 1120x1680 both ways, back-to-back A/B), so the
/// direct global-load variant stays the default; the staged path is kept
/// for retest on other drivers. The global K fragment loads were NOT the
/// scores bottleneck — same lesson as the A-stride-34 experiment.
pub const scores_stage_k = false;

/// Attention-scores GEMM on tensor cores: S[z][q][j] = Qh . Kh^T for one
/// batch of heads (gid.z = head-in-batch). Q/K are f16 in global memory and
/// cooperative-loadable directly; the f16 OUTPUT tile bounces through a
/// 32 KB workgroup slab so the global S writes are fully coalesced u32
/// stores (direct cooperative stores scatter 32 B tile rows across the
/// s_stride and measured ~184 GB/s — a 4x loss). LocalSize (32,4): 2x2
/// warps each computing a 64x64 quarter of the workgroup's 128(q) x 128(j)
/// tile.
///
/// Bindings: 0 = K (f16, per-head k-major [kv][hd][s_stride], zero-padded
/// cols), 1 = Q (f16, [seq][heads][hd], softmax scale prefolded), 2 = S
/// (f32, [z][m_pad][s_stride]). Push (EltPush words): u0 = q row stride
/// (heads*hd), u1 = s_stride, u2 = head_off, u3 = heads-per-kv group,
/// u4 = K head stride (hd*s_stride), u5 = S plane stride (m_pad*s_stride).
/// `hd` (multiple of 16) sets the unrolled k-depth; the DiT uses 128, the
/// VAE mid-block 384.
pub fn buildGemmScores(gpa: std.mem.Allocator, hd: u32, stage_k: bool) ![]align(4) u8 {
    std.debug.assert(hd % 16 == 0 and hd >= 64);
    std.debug.assert(!stage_k or hd % 64 == 0);

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |s| gpa.free(s);
        names.deinit(gpa);
    }
    const Nm = struct {
        nl: *std.ArrayList([]u8),
        g: std.mem.Allocator,
        fn f(self: @This(), comptime fmt: []const u8, args: anytype) ![]const u8 {
            const s = try std.fmt.allocPrint(self.g, fmt, args);
            try self.nl.append(self.g, s);
            return s;
        }
    };
    const nm = Nm{ .nl = &names, .g = gpa };

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpCapability CooperativeMatrixKHR", .{});
    try em.line("OpExtension \"SPV_KHR_cooperative_matrix\"", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("OpExtension \"SPV_KHR_16bit_storage\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vk %vq %vs %vs4 %vpush %vssh", .{});
    try em.line("OpExecutionMode %main LocalSize 32 4 1", .{});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    try em.line("OpDecorate %arr_f16 ArrayStride 2", .{});
    try em.line("OpDecorate %arr_f16s ArrayStride 2", .{});
    try em.line("OpDecorate %arr_u32s ArrayStride 4", .{});
    for ([_][]const u8{ "sk", "sq", "ss", "ss4" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }
    try em.line("OpDecorate %push Block", .{});
    for (0..8) |m| try em.line("OpMemberDecorate %push {d} Offset {d}", .{ m, m * 4 });
    for ([_][]const u8{ "vk", "vq", "vs" }, 0..) |v, b| {
        try em.line("OpDecorate %{s} DescriptorSet 0", .{v});
        try em.line("OpDecorate %{s} Binding {d}", .{ v, b });
    }
    try em.line("OpDecorate %vs4 DescriptorSet 0", .{});
    try em.line("OpDecorate %vs4 Binding 3", .{});

    // --- types ---
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%f32 = OpTypeFloat 32", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_f16 = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%arr_f16s = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%arr_u32s = OpTypeArray %u32 %c_arrlen", .{});
    try em.line("%sk = OpTypeStruct %arr_f16", .{});
    try em.line("%sq = OpTypeStruct %arr_f16", .{});
    try em.line("%ss = OpTypeStruct %arr_f16s", .{});
    try em.line("%ss4 = OpTypeStruct %arr_u32s", .{});
    try em.line("%ptr_sk = OpTypePointer StorageBuffer %sk", .{});
    try em.line("%ptr_sq = OpTypePointer StorageBuffer %sq", .{});
    try em.line("%ptr_ss = OpTypePointer StorageBuffer %ss", .{});
    try em.line("%ptr_ss4 = OpTypePointer StorageBuffer %ss4", .{});
    try em.line("%vk = OpVariable %ptr_sk StorageBuffer", .{});
    try em.line("%vq = OpVariable %ptr_sq StorageBuffer", .{});
    try em.line("%vs = OpVariable %ptr_ss StorageBuffer", .{});
    try em.line("%vs4 = OpVariable %ptr_ss4 StorageBuffer", .{});
    try em.line("%v2f16 = OpTypeVector %f16 2", .{});
    try em.line("%push = OpTypeStruct %u32 %u32 %u32 %u32 %u32 %u32 %u32 %u32", .{});
    try em.line("%ptr_push = OpTypePointer PushConstant %push", .{});
    try em.line("%vpush = OpVariable %ptr_push PushConstant", .{});
    try em.line("%ptr_pc_u32 = OpTypePointer PushConstant %u32", .{});
    try em.line("%ptr_sk_f16 = OpTypePointer StorageBuffer %f16", .{});
    try em.line("%ptr_sq_f16 = OpTypePointer StorageBuffer %f16", .{});
    try em.line("%ptr_ss_f32 = OpTypePointer StorageBuffer %f16", .{});
    try em.line("%ptr_ss4_u32 = OpTypePointer StorageBuffer %u32", .{});

    // Workgroup bounce slab: [128][128] f16.
    try em.line("%c_ssh_len = OpConstant %u32 16384", .{});
    try em.line("%t_ssh = OpTypeArray %f16 %c_ssh_len", .{});
    try em.line("%ptr_wg_ssh = OpTypePointer Workgroup %t_ssh", .{});
    try em.line("%vssh = OpVariable %ptr_wg_ssh Workgroup", .{});
    try em.line("%ptr_wg_f16 = OpTypePointer Workgroup %f16", .{});

    for ([_]struct { []const u8, u32 }{
        .{ "c_u0", 0 },  .{ "c_u1", 1 },   .{ "c_u2", 2 },     .{ "c_u3", 3 },
        .{ "c_u4", 4 },  .{ "c_u5", 5 },   .{ "c_u6", 6 },     .{ "c_u63", 63 },
        .{ "c_u16", 16 }, .{ "c_u32c", 32 }, .{ "c_u64", 64 }, .{ "c_u128", 128 },
        .{ "c_u264", 0x108 }, .{ "c_scope_sub", 3 }, .{ "c_scope_wg", 2 },
    }) |cv| try em.line("%{s} = OpConstant %u32 {d}", .{ cv[0], cv[1] });

    try em.line("%mat_a = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u0", .{});
    try em.line("%mat_b = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u1", .{});
    try em.line("%mat_c = OpTypeCooperativeMatrixKHR %f32 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%mat_h = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%c_f32_0 = OpConstant %f32 0", .{});
    try em.line("%c_acc0 = OpConstantComposite %mat_c %c_f32_0", .{});

    // ks*16 k-offsets up to hd (also reused as nt*16 column offsets).
    const klen = @max(hd / 16, 4);
    const c_k16 = try gpa.alloc([]const u8, klen);
    defer gpa.free(c_k16);
    c_k16[0] = "c_u0";
    c_k16[1] = "c_u16";
    c_k16[2] = "c_u32c";
    for (3..klen) |i| {
        if (i * 16 == 128) {
            c_k16[i] = "c_u128";
            continue;
        }
        try em.line("%c_k16_{d} = OpConstant %u32 {d}", .{ i, @as(u32, @intCast(i * 16)) });
        c_k16[i] = try nm.f("c_k16_{d}", .{i});
    }
    const c_hd = switch (hd) {
        128 => "c_u128",
        64 => "c_u64",
        else => blk: {
            try em.line("%c_hd = OpConstant %u32 {d}", .{hd});
            break :blk "c_hd";
        },
    };
    // s_sh tile offsets within the bounce slab: r*16*128 + nt*16.
    var c_st: [4][4][]const u8 = undefined;
    for (0..4) |r| for (0..4) |nt| {
        if (r == 0) {
            c_st[r][nt] = c_k16[nt];
        } else {
            try em.line("%c_st_{d}_{d} = OpConstant %u32 {d}", .{ r, nt, @as(u32, @intCast(r * 2048 + nt * 16)) });
            c_st[r][nt] = try nm.f("c_st_{d}_{d}", .{ r, nt });
        }
    };
    const c_u256 = if (klen > 16) c_k16[16] else blk: {
        try em.line("%c_u256 = OpConstant %u32 256", .{});
        break :blk "c_u256";
    };

    // --- function (straight-line) ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});

    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const tile_c = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ tile_c, gidv });
    const tile_r = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ tile_r, gidv });
    const zidx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 2", .{ zidx, gidv });
    const col0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ col0, tile_c });
    const row0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ row0, tile_r });

    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const lx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ lx, lidv });
    const ly = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ ly, lidv });
    const lymul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u32c", .{ lymul, ly });
    const flat = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ flat, lymul, lx });

    const pnames = [_][]const u8{ "pqstride", "psstride", "pheadoff", "pgroup", "pkhead", "psplane" };
    for (0..6) |m| {
        const pptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_pc_u32 %vpush %c_u{d}", .{ pptr, m });
        try em.line("%{s} = OpLoad %u32 %t{d}", .{ pnames[m], pptr });
    }

    // head = head_off + z; kv = head / group.
    const head = em.id();
    try em.line("%t{d} = OpIAdd %u32 %pheadoff %t{d}", .{ head, zidx });
    const kvh = em.id();
    try em.line("%t{d} = OpUDiv %u32 %t{d} %pgroup", .{ kvh, head });

    // Warp tiling: 2x2 grid of 64x64 tiles.
    const warp_m = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u1", .{ warp_m, ly });
    const warp_n = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ warp_n, ly });
    const wm64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wm64, warp_m });
    const row_s = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ row_s, row0, wm64 });
    const wn64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wn64, warp_n });
    const col_w = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ col_w, col0, wn64 });

    // A (Q) base: row_s*q_stride + head*hd; row stride q_stride.
    const a_rowmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %pqstride", .{ a_rowmul, row_s });
    const headmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %{s}", .{ headmul, head, c_hd });
    const a_base = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ a_base, a_rowmul, headmul });
    const a_row16 = em.id();
    try em.line("%t{d} = OpIMul %u32 %c_u16 %pqstride", .{a_row16});

    // K block base: kv*k_head_stride.
    const kvmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %pkhead", .{ kvmul, kvh });

    // Copy-out invariants.
    const zmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %psplane", .{ zmul, zidx });
    const srow0 = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u6", .{ srow0, flat }); // flat / 64
    const fmask = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u63", .{ fmask, flat });
    const scol2 = em.id();
    try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u1", .{ scol2, fmask }); // * 2
    const grow0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ grow0, row0, srow0 });
    const growmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %psstride", .{ growmul, grow0 });
    const gsum0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gsum0, zmul, growmul });
    const gsum1 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gsum1, gsum0, col0 });
    const gsum2 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gsum2, gsum1, scol2 });
    const gword0 = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ gword0, gsum2 });
    const erow = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ erow, srow0 });
    const e0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ e0, erow, scol2 });

    // acc[r][nt] as reference names (init to the zero composite).
    var acc: [4][4][]const u8 = undefined;
    for (&acc) |*row| for (row) |*v| {
        v.* = "c_acc0";
    };
    if (stage_k) {
        const kbase0 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ kbase0, kvmul, col0 });
        const kbase = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ kbase, kbase0, scol2 });
        const c2s = em.id();
        try em.line("%t{d} = OpIAdd %u32 %psstride %psstride", .{c2s}); // 2 rows/iter
        for (0..hd / 64) |s| {
            const krow = em.id();
            try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ krow, c_k16[s * 4], srow0 });
            const krowmul = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %psstride", .{ krowmul, krow });
            var g = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ g, kbase, krowmul });
            var e = e0;
            for (0..32) |i| {
                if (i > 0) {
                    const gn = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gn, g, c2s });
                    g = gn;
                    const en = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ en, e, c_u256 });
                    e = en;
                }
                const g1 = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ g1, g });
                const e1 = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ e1, e });
                const pairs = [2][2]u32{ .{ g, e }, .{ g1, e1 } };
                for (pairs) |pair| {
                    const kp = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_sk_f16 %vk %c_u0 %t{d}", .{ kp, pair[0] });
                    const kv = em.id();
                    try em.line("%t{d} = OpLoad %f16 %t{d}", .{ kv, kp });
                    const sp = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vssh %t{d}", .{ sp, pair[1] });
                    try em.line("OpStore %t{d} %t{d}", .{ sp, kv });
                }
            }
            try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
            for (0..4) |kk| {
                const ks = s * 4 + kk;
                var a_off = a_base;
                if (ks > 0) {
                    const ao = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ao, a_base, c_k16[ks] });
                    a_off = ao;
                }
                var ma: [4]u32 = undefined;
                var ao_cur = a_off;
                for (0..4) |r| {
                    if (r > 0) {
                        const a2 = em.id();
                        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ a2, ao_cur, a_row16 });
                        ao_cur = a2;
                    }
                    const aptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_sq_f16 %vq %c_u0 %t{d}", .{ aptr, ao_cur });
                    ma[r] = em.id();
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %pqstride", .{ ma[r], aptr });
                }
                for (0..4) |nt| {
                    const bidx = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ bidx, c_st[kk][nt], wn64 });
                    const bptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vssh %t{d}", .{ bptr, bidx });
                    const mb = em.id();
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %c_u128", .{ mb, bptr });
                    for (0..4) |r| {
                        const acc_out = em.id();
                        try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %{s}", .{ acc_out, ma[r], mb, acc[r][nt] });
                        acc[r][nt] = try nm.f("t{d}", .{acc_out});
                    }
                }
            }
            try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
        }
    } else {
        const b_base = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ b_base, kvmul, col_w });
        for (0..hd / 16) |ks| {
            var a_off = a_base;
            if (ks > 0) {
                const ao = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ao, a_base, c_k16[ks] });
                a_off = ao;
            }
            var ma: [4]u32 = undefined;
            var ao_cur = a_off;
            for (0..4) |r| {
                if (r > 0) {
                    const a2 = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ a2, ao_cur, a_row16 });
                    ao_cur = a2;
                }
                const aptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_sq_f16 %vq %c_u0 %t{d}", .{ aptr, ao_cur });
                ma[r] = em.id();
                try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %pqstride", .{ ma[r], aptr });
            }
            const krowmul = em.id();
            try em.line("%t{d} = OpIMul %u32 %{s} %psstride", .{ krowmul, c_k16[ks] });
            const b_ks = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ b_ks, b_base, krowmul });
            for (0..4) |nt| {
                var b_off = b_ks;
                if (nt > 0) {
                    const bo = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ bo, b_ks, c_k16[nt] });
                    b_off = bo;
                }
                const bptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_sk_f16 %vk %c_u0 %t{d}", .{ bptr, b_off });
                const mb = em.id();
                try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %psstride", .{ mb, bptr });
                for (0..4) |r| {
                    const acc_out = em.id();
                    try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %{s}", .{ acc_out, ma[r], mb, acc[r][nt] });
                    acc[r][nt] = try nm.f("t{d}", .{acc_out});
                }
            }
        }
    }

    // Stage the warp's 4x4 f16 tiles into the bounce slab, then copy out.
    const wbase0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ wbase0, wm64 });
    const wbase = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ wbase, wbase0, wn64 });
    for (0..4) |r| {
        for (0..4) |nt| {
            const hacc = em.id();
            try em.line("%t{d} = OpFConvert %mat_h %{s}", .{ hacc, acc[r][nt] });
            const sidx = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ sidx, wbase, c_st[r][nt] });
            const sptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vssh %t{d}", .{ sptr, sidx });
            try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u0 %c_u128", .{ sptr, hacc });
        }
    }
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
    // 8192 words -> 64 per thread, 2 rows apart.
    var e = e0;
    var gw = gword0;
    for (0..64) |i| {
        if (i > 0) {
            const en = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ en, e, c_u256 });
            e = en;
            const gn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %psstride", .{ gn, gw });
            gw = gn;
        }
        const p0 = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vssh %t{d}", .{ p0, e });
        const h0 = em.id();
        try em.line("%t{d} = OpLoad %f16 %t{d}", .{ h0, p0 });
        const e1 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ e1, e });
        const p1 = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vssh %t{d}", .{ p1, e1 });
        const h1 = em.id();
        try em.line("%t{d} = OpLoad %f16 %t{d}", .{ h1, p1 });
        const pair = em.id();
        try em.line("%t{d} = OpCompositeConstruct %v2f16 %t{d} %t{d}", .{ pair, h0, h1 });
        const word = em.id();
        try em.line("%t{d} = OpBitcast %u32 %t{d}", .{ word, pair });
        const gptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_ss4_u32 %vs4 %c_u0 %t{d}", .{ gptr, gw });
        try em.line("OpStore %t{d} %t{d}", .{ gptr, word });
    }
    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}

/// Flash attention on tensor cores (head_dim 128), two-pass recompute: the
/// scores matrix never exists in global memory. Each workgroup (LocalSize
/// (32,4), grid (1, seq_pad/128, heads)) owns 128 q rows of one head; Q
/// stages once into a 32 KB shared slab and is reused across the whole j
/// loop. Per 64-wide j block, S = Q@K^T is computed on cooperative matrices
/// and stored COLUMN-MAJOR into a 16 KB shared tile (s_sh[j][q] — so the
/// per-row scalar passes read lane-consecutive addresses, conflict-free).
///
/// buildFlashMd (pass 1): each thread owns one q row and folds the shared S
/// block into a running online {max, sum-exp}; at the end writes
/// {m, 1/d} to the MD table. Replaces scores + softmax_partial/combine.
///
/// buildFlashOut (pass 2): recomputes each S block the same way, transforms
/// it in place to P = exp(S - m) * invd (padded j forced to zero — see the
/// f16 overflow note on buildGemmAttnOut), then accumulates P@V on
/// cooperative matrices (P loads are column-major from the shared tile).
/// Replaces buildGemmAttnOut. The MD table lives in the TAIL of the f32
/// output buffer (binding 2) at push offset u5, because K/Q/OUT/V exhaust
/// the four bindings.
///
/// Bindings: 0 = K (f16 per-head k-major [kv][128][s_stride], zero-padded
/// cols), 1 = Q (f16 [seq_pad][heads*128], softmax scale prefolded, zero
/// pad rows), 2 = OUT+MD (f32; out rows [seq_pad][heads*128] then MD
/// [z][s_stride] x {m, 1/d} at u5), 3 = V (f16 [seq_pad][kv*128], zero pad
/// rows; unused by pass 1). Push (EltPush words): u0 = q/out row stride
/// (heads*128), u1 = s_stride == MD plane stride, u2 = head_off, u3 =
/// heads-per-kv group, u4 = V row stride (kv*128), u5 = MD offset in f32
/// elements, f0 = valid j count (u32 bits).
fn buildFlashAttn(gpa: std.mem.Allocator, out_phase: bool, stage_k: bool) ![]align(4) u8 {
    const STAGE_Q = false;
    const Q_SH: u32 = if (STAGE_Q) 16384 else 0; // s_sh region base
    const K_SH: u32 = Q_SH + 8192; // k_sh region base (stage_k)

    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |s| gpa.free(s);
        names.deinit(gpa);
    }
    const Nm = struct {
        nl: *std.ArrayList([]u8),
        g: std.mem.Allocator,
        fn f(self: @This(), comptime fmt: []const u8, args: anytype) ![]const u8 {
            const s = try std.fmt.allocPrint(self.g, fmt, args);
            try self.nl.append(self.g, s);
            return s;
        }
    };
    const nm = Nm{ .nl = &names, .g = gpa };

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpCapability CooperativeMatrixKHR", .{});
    try em.line("OpExtension \"SPV_KHR_cooperative_matrix\"", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("OpExtension \"SPV_KHR_16bit_storage\"", .{});
    try em.line("%glsl = OpExtInstImport \"GLSL.std.450\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vk %vq %vo %vv %vpush %vwsh", .{});
    try em.line("OpExecutionMode %main LocalSize 32 4 1", .{});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    for ([_][]const u8{ "arr_f16k", "arr_f16q", "arr_f16v" }) |s| try em.line("OpDecorate %{s} ArrayStride 2", .{s});
    try em.line("OpDecorate %arr_f32o ArrayStride 4", .{});
    for ([_][]const u8{ "sk", "sq", "so", "sv" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }
    try em.line("OpDecorate %push Block", .{});
    for (0..8) |m| try em.line("OpMemberDecorate %push {d} Offset {d}", .{ m, m * 4 });
    for ([_][]const u8{ "vk", "vq", "vo", "vv" }, 0..) |v, b| {
        try em.line("OpDecorate %{s} DescriptorSet 0", .{v});
        try em.line("OpDecorate %{s} Binding {d}", .{ v, b });
    }

    // --- types ---
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%f32 = OpTypeFloat 32", .{});
    try em.line("%bool = OpTypeBool", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_f16k = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%arr_f16q = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%arr_f32o = OpTypeArray %f32 %c_arrlen", .{});
    try em.line("%arr_f16v = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%sk = OpTypeStruct %arr_f16k", .{});
    try em.line("%sq = OpTypeStruct %arr_f16q", .{});
    try em.line("%so = OpTypeStruct %arr_f32o", .{});
    try em.line("%sv = OpTypeStruct %arr_f16v", .{});
    try em.line("%ptr_sk = OpTypePointer StorageBuffer %sk", .{});
    try em.line("%ptr_sq = OpTypePointer StorageBuffer %sq", .{});
    try em.line("%ptr_so = OpTypePointer StorageBuffer %so", .{});
    try em.line("%ptr_sv = OpTypePointer StorageBuffer %sv", .{});
    try em.line("%vk = OpVariable %ptr_sk StorageBuffer", .{});
    try em.line("%vq = OpVariable %ptr_sq StorageBuffer", .{});
    try em.line("%vo = OpVariable %ptr_so StorageBuffer", .{});
    try em.line("%vv = OpVariable %ptr_sv StorageBuffer", .{});
    try em.line("%push = OpTypeStruct %u32 %u32 %u32 %u32 %u32 %u32 %u32 %u32", .{});
    try em.line("%ptr_push = OpTypePointer PushConstant %push", .{});
    try em.line("%vpush = OpVariable %ptr_push PushConstant", .{});
    try em.line("%ptr_pc_u32 = OpTypePointer PushConstant %u32", .{});
    try em.line("%ptr_g_f16 = OpTypePointer StorageBuffer %f16", .{});
    try em.line("%ptr_o_f32 = OpTypePointer StorageBuffer %f32", .{});

    // Workgroup slab.
    try em.line("%c_wsh_len = OpConstant %u32 {d}", .{Q_SH + 8192 + @as(u32, if (stage_k) 8192 else 0)});
    try em.line("%t_wsh = OpTypeArray %f16 %c_wsh_len", .{});
    try em.line("%ptr_wg_wsh = OpTypePointer Workgroup %t_wsh", .{});
    try em.line("%vwsh = OpVariable %ptr_wg_wsh Workgroup", .{});
    try em.line("%ptr_wg_f16 = OpTypePointer Workgroup %f16", .{});

    for ([_]struct { []const u8, u32 }{
        .{ "c_u0", 0 },  .{ "c_u1", 1 },  .{ "c_u2", 2 },  .{ "c_u3", 3 },
        .{ "c_u4", 4 },  .{ "c_u5", 5 },  .{ "c_u6", 6 },  .{ "c_u7", 7 },
        .{ "c_u16", 16 }, .{ "c_u32c", 32 }, .{ "c_u64", 64 }, .{ "c_u128", 128 },
        .{ "c_u264", 0x108 }, .{ "c_scope_sub", 3 }, .{ "c_scope_wg", 2 },
    }) |cv| try em.line("%{s} = OpConstant %u32 {d}", .{ cv[0], cv[1] });

    try em.line("%mat_a = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u0", .{});
    try em.line("%mat_b = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u1", .{});
    try em.line("%mat_c = OpTypeCooperativeMatrixKHR %f32 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%mat_h = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%c_f32_0 = OpConstant %f32 0", .{});
    try em.line("%c_f32_1 = OpConstant %f32 1065353216", .{}); // 0x3F800000
    try em.line("%c_f32_ninf = OpConstant %f32 {d}", .{@as(u32, @bitCast(@as(f32, -3.4e38)))});
    try em.line("%c_f32_minf = OpConstant %f32 4286578688", .{}); // 0xFF800000
    try em.line("%c_f16_0 = OpConstant %f16 0", .{});
    try em.line("%c_acc0 = OpConstantComposite %mat_c %c_f32_0", .{});

    // i*16 offsets.
    var c_k16: [8][]const u8 = undefined;
    c_k16[0] = "c_u0";
    c_k16[1] = "c_u16";
    c_k16[2] = "c_u32c";
    for (3..8) |i| {
        try em.line("%c_k16_{d} = OpConstant %u32 {d}", .{ i, i * 16 });
        c_k16[i] = try nm.f("c_k16_{d}", .{i});
    }
    // Per-column s_sh offsets col*128.
    var c_col: [64][]const u8 = undefined;
    c_col[0] = "c_u0";
    c_col[1] = "c_u128";
    for (2..64) |i| {
        try em.line("%c_col_{d} = OpConstant %u32 {d}", .{ i, i * 128 });
        c_col[i] = try nm.f("c_col_{d}", .{i});
    }
    // s_sh tile bases Q_SH + ct*16*128 (alias c_col when Q_SH == 0).
    const c_ssh0 = if (Q_SH == 0) "c_u0" else blk: {
        try em.line("%c_ssh0 = OpConstant %u32 {d}", .{Q_SH});
        break :blk "c_ssh0";
    };
    var c_sct: [4][]const u8 = undefined;
    c_sct[0] = c_ssh0;
    for (1..4) |ct| {
        const val: u32 = Q_SH + @as(u32, @intCast(ct)) * 16 * 128;
        if (Q_SH == 0) {
            c_sct[ct] = c_col[val / 128];
        } else {
            try em.line("%c_sct_{d} = OpConstant %u32 {d}", .{ ct, val });
            c_sct[ct] = try nm.f("c_sct_{d}", .{ct});
        }
    }
    var c_kst: [8][]const u8 = undefined;
    if (stage_k) {
        for (0..8) |ks| {
            try em.line("%c_kst_{d} = OpConstant %u32 {d}", .{ ks, K_SH + @as(u32, @intCast(ks)) * 1024 });
            c_kst[ks] = try nm.f("c_kst_{d}", .{ks});
        }
        try em.line("%c_u63 = OpConstant %u32 63", .{});
    }

    // --- function ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});

    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const tile_r = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ tile_r, gidv });
    const zidx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 2", .{ zidx, gidv });
    const row0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ row0, tile_r });

    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const lx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ lx, lidv });
    const ly = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ ly, lidv });
    const lymul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u32c", .{ lymul, ly });
    const flat = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ flat, lymul, lx });

    const pnames = [_][]const u8{ "pqstride", "psstride", "pheadoff", "pgroup", "pvstride", "pmdoff", "pseq" };
    for (0..7) |m| {
        const pptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_pc_u32 %vpush %c_u{d}", .{ pptr, m });
        try em.line("%{s} = OpLoad %u32 %t{d}", .{ pnames[m], pptr });
    }

    const head = em.id();
    try em.line("%t{d} = OpIAdd %u32 %pheadoff %t{d}", .{ head, zidx });
    const kvh = em.id();
    try em.line("%t{d} = OpUDiv %u32 %t{d} %pgroup", .{ kvh, head });
    const headmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ headmul, head });
    const khead = em.id();
    try em.line("%t{d} = OpShiftLeftLogical %u32 %psstride %c_u7", .{khead});
    const kbase = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %t{d}", .{ kbase, kvh, khead });
    const kvmul_v = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ kvmul_v, kvh });

    // Per-warp Q row-block bases.
    const ly32 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u32c", .{ ly32, ly });
    const qrow_g0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qrow_g0, row0, ly32 });
    var qg_base: [2]u32 = undefined;
    for (0..2) |rw| {
        var grow = qrow_g0;
        if (rw != 0) {
            const g = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u16", .{ g, qrow_g0 });
            grow = g;
        }
        const gmul = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %pqstride", .{ gmul, grow });
        qg_base[rw] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qg_base[rw], gmul, headmul });
    }
    if (STAGE_Q) {
        for (0..2) |rw| {
            const smul = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ smul, ly32, if (rw == 0) "c_u0" else "c_u16" });
            const sbase = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ sbase, smul });
            for (0..8) |kt| {
                const goff = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ goff, qg_base[rw], c_k16[kt] });
                const gptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_g_f16 %vq %c_u0 %t{d}", .{ gptr, goff });
                const mq = em.id();
                try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %pqstride", .{ mq, gptr });
                const soff = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ soff, sbase, c_k16[kt] });
                const sptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ sptr, soff });
                try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u0 %c_u128", .{ sptr, mq });
            }
        }
    }

    // Per-thread row MD index.
    const mdrow = em.id();
    {
        const zr = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %psstride", .{ zr, zidx });
        const qr = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qr, zr, row0 });
        const qr2 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qr2, qr, flat });
        const dbl = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u1", .{ dbl, qr2 });
        try em.line("%t{d} = OpIAdd %u32 %t{d} %pmdoff", .{ mdrow, dbl });
    }
    var m_hoist: u32 = undefined;
    var invd_hoist: u32 = undefined;
    if (out_phase) {
        const mptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_o_f32 %vo %c_u0 %t{d}", .{ mptr, mdrow });
        m_hoist = em.id();
        try em.line("%t{d} = OpLoad %f32 %t{d}", .{ m_hoist, mptr });
        const di = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ di, mdrow });
        const dptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_o_f32 %vo %c_u0 %t{d}", .{ dptr, di });
        invd_hoist = em.id();
        try em.line("%t{d} = OpLoad %f32 %t{d}", .{ invd_hoist, dptr });
    }
    const s_thread0 = em.id();
    try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ s_thread0, c_ssh0, flat });

    var k_inv: u32 = undefined;
    var k_e0: u32 = undefined;
    var c2s: u32 = undefined;
    if (stage_k) {
        const kj = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u63", .{ kj, flat });
        const kr0 = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u6", .{ kr0, flat });
        const gj = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gj, kbase, kj });
        const gr = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %psstride", .{ gr, kr0 });
        k_inv = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ k_inv, gj, gr });
        k_e0 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ k_e0, c_kst[0], flat });
        c2s = em.id();
        try em.line("%t{d} = OpIAdd %u32 %psstride %psstride", .{c2s});
    }

    // Warp tiling.
    const warp_m = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u1", .{ warp_m, ly });
    const warp_n = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ warp_n, ly });
    const wm64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wm64, warp_m });
    const wn64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wn64, warp_n });
    var wmr: [4]u32 = undefined;
    wmr[0] = wm64;
    for (1..4) |r| {
        wmr[r] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ wmr[r], wm64, c_k16[r] });
    }
    const vcol_base = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ vcol_base, kvmul_v, wn64 });

    // Loop-carried back-edge ids.
    const j0n = em.id();
    const mn_next = em.id();
    const dn_next = em.id();
    var acc_next: [4][4]u32 = undefined;
    for (&acc_next) |*row| for (row) |*v| {
        v.* = em.id();
    };

    try em.line("OpBranch %head", .{});
    try em.line("%head = OpLabel", .{});
    const j0v = em.id();
    try em.line("%t{d} = OpPhi %u32 %c_u0 %entry %t{d} %cont", .{ j0v, j0n });
    var m_phi: u32 = undefined;
    var d_phi: u32 = undefined;
    var acc_phi: [4][4]u32 = undefined;
    if (!out_phase) {
        m_phi = em.id();
        try em.line("%t{d} = OpPhi %f32 %c_f32_ninf %entry %t{d} %cont", .{ m_phi, mn_next });
        d_phi = em.id();
        try em.line("%t{d} = OpPhi %f32 %c_f32_0 %entry %t{d} %cont", .{ d_phi, dn_next });
    } else {
        for (&acc_phi, acc_next) |*prow, nrow| {
            for (prow, nrow) |*ap, an| {
                ap.* = em.id();
                try em.line("%t{d} = OpPhi %mat_c %c_acc0 %entry %t{d} %cont", .{ ap.*, an });
            }
        }
    }
    try em.line("OpLoopMerge %merge %cont None", .{});
    try em.line("OpBranch %cond", .{});
    try em.line("%cond = OpLabel", .{});
    const cmp = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %psstride", .{ cmp, j0v });
    try em.line("OpBranchConditional %t{d} %body %merge", .{cmp});

    try em.line("%body = OpLabel", .{});
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});

    if (stage_k) {
        var g = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ g, k_inv, j0v });
        var e = k_e0;
        for (0..64) |i| {
            if (i > 0) {
                const gn = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ gn, g, c2s });
                g = gn;
                const en = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u128", .{ en, e });
                e = en;
            }
            const gp = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_g_f16 %vk %c_u0 %t{d}", .{ gp, g });
            const hv = em.id();
            try em.line("%t{d} = OpLoad %f16 %t{d}", .{ hv, gp });
            const sp = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ sp, e });
            try em.line("OpStore %t{d} %t{d}", .{ sp, hv });
        }
        try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
    }

    // S block.
    for (0..2) |rw| {
        const q_shbase = em.id();
        {
            const qr = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ qr, ly32, if (rw == 0) "c_u0" else "c_u16" });
            try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ q_shbase, qr });
        }
        const s_qoff = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ s_qoff, ly32, if (rw == 0) "c_u0" else "c_u16" });
        for (0..4) |ct| {
            var acc: []const u8 = "c_acc0";
            const jg = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ jg, j0v, c_k16[ct] });
            for (0..8) |ks| {
                const qoff = em.id();
                if (STAGE_Q) {
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ qoff, q_shbase, c_k16[ks] });
                } else {
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ qoff, qg_base[rw], c_k16[ks] });
                }
                const qptr = em.id();
                if (STAGE_Q) {
                    try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ qptr, qoff });
                } else {
                    try em.line("%t{d} = OpAccessChain %ptr_g_f16 %vq %c_u0 %t{d}", .{ qptr, qoff });
                }
                const ma = em.id();
                if (STAGE_Q) {
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %c_u128", .{ ma, qptr });
                } else {
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %pqstride", .{ ma, qptr });
                }
                var mb: u32 = undefined;
                if (stage_k) {
                    const koff = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %{s} %{s}", .{ koff, c_kst[ks], c_k16[ct] });
                    const kptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ kptr, koff });
                    mb = em.id();
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %c_u64", .{ mb, kptr });
                } else {
                    const krow = em.id();
                    try em.line("%t{d} = OpIMul %u32 %{s} %psstride", .{ krow, c_k16[ks] });
                    const koff0 = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ koff0, kbase, krow });
                    const koff = em.id();
                    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ koff, koff0, jg });
                    const kptr = em.id();
                    try em.line("%t{d} = OpAccessChain %ptr_g_f16 %vk %c_u0 %t{d}", .{ kptr, koff });
                    mb = em.id();
                    try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %psstride", .{ mb, kptr });
                }
                const acc_out = em.id();
                try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %{s}", .{ acc_out, ma, mb, acc });
                acc = try nm.f("t{d}", .{acc_out});
            }
            const hacc = em.id();
            try em.line("%t{d} = OpFConvert %mat_h %{s}", .{ hacc, acc });
            const soff = em.id();
            try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ soff, c_sct[ct], s_qoff });
            const sptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ sptr, soff });
            try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u1 %c_u128", .{ sptr, hacc }); // ColumnMajor
        }
    }
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});

    // Validity horizon.
    const over = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %pseq", .{ over, j0v });
    const rem = em.id();
    try em.line("%t{d} = OpISub %u32 %pseq %t{d}", .{ rem, j0v });
    const limit = em.id();
    try em.line("%t{d} = OpSelect %u32 %t{d} %t{d} %c_u0", .{ limit, over, rem });
    const limit128 = em.id();
    try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u7", .{ limit128, limit });

    if (!out_phase) {
        var m_cur = m_phi;
        var d_cur = d_phi;
        for (0..64) |col| {
            var eidx = s_thread0;
            if (col > 0) {
                const ei = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ei, s_thread0, c_col[col] });
                eidx = ei;
            }
            const sptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ sptr, eidx });
            const hval = em.id();
            try em.line("%t{d} = OpLoad %f16 %t{d}", .{ hval, sptr });
            const s32 = em.id();
            try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ s32, hval });
            const valid = em.id();
            try em.line("%t{d} = OpULessThan %bool %{s} %t{d}", .{ valid, c_col[col], limit128 });
            const s_eff = em.id();
            try em.line("%t{d} = OpSelect %f32 %t{d} %t{d} %c_f32_minf", .{ s_eff, valid, s32 });
            const m_new = if (col == 63) mn_next else em.id();
            try em.line("%t{d} = OpExtInst %f32 %glsl 40 %t{d} %t{d}", .{ m_new, m_cur, s_eff }); // FMax
            const dm = em.id();
            try em.line("%t{d} = OpFSub %f32 %t{d} %t{d}", .{ dm, m_cur, m_new });
            const corr = em.id();
            try em.line("%t{d} = OpExtInst %f32 %glsl 27 %t{d}", .{ corr, dm }); // Exp
            const ds = em.id();
            try em.line("%t{d} = OpFSub %f32 %t{d} %t{d}", .{ ds, s_eff, m_new });
            const p = em.id();
            try em.line("%t{d} = OpExtInst %f32 %glsl 27 %t{d}", .{ p, ds });
            const dscaled = em.id();
            try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ dscaled, d_cur, corr });
            const d_new = if (col == 63) dn_next else em.id();
            try em.line("%t{d} = OpFAdd %f32 %t{d} %t{d}", .{ d_new, dscaled, p });
            m_cur = m_new;
            d_cur = d_new;
        }
    } else {
        for (0..64) |col| {
            var eidx = s_thread0;
            if (col > 0) {
                const ei = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ei, s_thread0, c_col[col] });
                eidx = ei;
            }
            const sptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ sptr, eidx });
            const hval = em.id();
            try em.line("%t{d} = OpLoad %f16 %t{d}", .{ hval, sptr });
            const s32 = em.id();
            try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ s32, hval });
            const ds = em.id();
            try em.line("%t{d} = OpFSub %f32 %t{d} %t{d}", .{ ds, s32, m_hoist });
            const e = em.id();
            try em.line("%t{d} = OpExtInst %f32 %glsl 27 %t{d}", .{ e, ds });
            const p = em.id();
            try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ p, e, invd_hoist });
            const valid = em.id();
            try em.line("%t{d} = OpULessThan %bool %{s} %t{d}", .{ valid, c_col[col], limit128 });
            const pm = em.id();
            try em.line("%t{d} = OpSelect %f32 %t{d} %t{d} %c_f32_0", .{ pm, valid, p });
            const hp = em.id();
            try em.line("%t{d} = OpFConvert %f16 %t{d}", .{ hp, pm });
            try em.line("OpStore %t{d} %t{d}", .{ sptr, hp });
        }
        try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});

        var acc_cur = acc_phi;
        for (0..4) |kk| {
            var ma: [4]u32 = undefined;
            for (0..4) |r| {
                const aoff = em.id();
                try em.line("%t{d} = OpIAdd %u32 %{s} %t{d}", .{ aoff, c_sct[kk], wmr[r] });
                const aptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vwsh %t{d}", .{ aptr, aoff });
                ma[r] = em.id();
                try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u1 %c_u128", .{ ma[r], aptr }); // ColMajor
            }
            const jg2 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ jg2, j0v, c_k16[kk] });
            const vrow = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %pvstride", .{ vrow, jg2 });
            const vb0 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ vb0, vrow, vcol_base });
            for (0..4) |ct| {
                const voff = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ voff, vb0, c_k16[ct] });
                const vptr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_g_f16 %vv %c_u0 %t{d}", .{ vptr, voff });
                const mb = em.id();
                try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %pvstride", .{ mb, vptr });
                for (0..4) |r| {
                    const acc_out = if (kk == 3) acc_next[r][ct] else em.id();
                    try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %t{d}", .{ acc_out, ma[r], mb, acc_cur[r][ct] });
                    acc_cur[r][ct] = acc_out;
                }
            }
        }
    }
    try em.line("OpBranch %cont", .{});

    try em.line("%cont = OpLabel", .{});
    try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u64", .{ j0n, j0v });
    try em.line("OpBranch %head", .{});

    try em.line("%merge = OpLabel", .{});
    if (!out_phase) {
        const invd = em.id();
        try em.line("%t{d} = OpFDiv %f32 %c_f32_1 %t{d}", .{ invd, d_phi });
        const mptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_o_f32 %vo %c_u0 %t{d}", .{ mptr, mdrow });
        try em.line("OpStore %t{d} %t{d}", .{ mptr, m_phi });
        const di = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ di, mdrow });
        const dptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_o_f32 %vo %c_u0 %t{d}", .{ dptr, di });
        try em.line("OpStore %t{d} %t{d}", .{ dptr, invd });
    } else {
        for (0..4) |r| {
            const orow = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ orow, row0, wmr[r] });
            const omul = em.id();
            try em.line("%t{d} = OpIMul %u32 %t{d} %pqstride", .{ omul, orow });
            const ob = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ ob, omul, headmul });
            const ob2 = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ ob2, ob, wn64 });
            for (0..4) |ct| {
                const ooff = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ooff, ob2, c_k16[ct] });
                const optr = em.id();
                try em.line("%t{d} = OpAccessChain %ptr_o_f32 %vo %c_u0 %t{d}", .{ optr, ooff });
                try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u0 %pqstride", .{ optr, acc_phi[r][ct] });
            }
        }
    }
    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}

/// Flash pass 1: online {m, 1/d} per q row (see buildFlashAttn).
/// Stage each j-block's K tile in shared for both flash passes (S 16 KB +
/// K 16 KB = 32 KB, 3 wgs/SM vs 6): the experiment that tests whether the
/// flash S-recompute is fragment-load-bound (all four warps consume
/// identical K fragments, so staging removes 4x-redundant global loads).
pub const flash_stage_k = false;

pub fn buildFlashMd(gpa: std.mem.Allocator) ![]align(4) u8 {
    return buildFlashAttn(gpa, false, flash_stage_k);
}

/// Flash pass 2: P@V with S recomputed per block (see buildFlashAttn).
pub fn buildFlashOut(gpa: std.mem.Allocator) ![]align(4) u8 {
    return buildFlashAttn(gpa, true, flash_stage_k);
}

/// Attention P@V GEMM on tensor cores: out[q][head][c] = sum_j P[q,j] *
/// V[j][kv][c] for one batch of heads (gid.z = head-in-batch), where
/// P = exp(S - m[q]) * invd[q] is computed from the raw f32 scores during
/// A staging (GLSL.std.450 Exp) using the per-row max / reciprocal
/// denominator from the two-pass softmax kernels. LocalSize (32,8): each
/// workgroup covers 128 q rows x the full head_dim (128) over 64-deep j
/// slabs staged as f16 in workgroup memory; V tiles are cooperative loads
/// straight from the f16 V buffer (zero-padded rows, so padded-j P values
/// contribute nothing).
///
/// Bindings: 0 = S (f32, [z][*][s_stride]), 1 = V (f16, [seq_pad][kv*128]),
/// 2 = OUT (f32, [q][out_stride]), 3 = MD (f32, [z][mdplane] x {m, 1/d}).
/// Push (EltPush words): u0 = S row stride == j count (seq_pad), u1 = S
/// plane stride, u2 = head_off, u3 = heads-per-kv group, u4 = V row stride
/// (kv_heads*128), u5 = OUT row stride (heads*128), f0 = valid j count
/// (u32 bits), f1 = MD plane stride in rows (u32 bits; the DiT passes
/// seq_pad, wider-head callers batching a head as several 128-column fake
/// heads pass 0 along with u1 = 0 so all fakes share plane 0).
pub fn buildGemmAttnOut(gpa: std.mem.Allocator) ![]align(4) u8 {
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(gpa);
    var em = Emit{ .w = &t, .gpa = gpa };

    // --- header ---
    try em.line("OpCapability Shader", .{});
    try em.line("OpCapability Float16", .{});
    try em.line("OpCapability VulkanMemoryModel", .{});
    try em.line("OpCapability CooperativeMatrixKHR", .{});
    try em.line("OpExtension \"SPV_KHR_cooperative_matrix\"", .{});
    try em.line("OpExtension \"SPV_KHR_vulkan_memory_model\"", .{});
    try em.line("OpExtension \"SPV_KHR_16bit_storage\"", .{});
    try em.line("%glsl = OpExtInstImport \"GLSL.std.450\"", .{});
    try em.line("OpMemoryModel Logical Vulkan", .{});
    try em.line("OpEntryPoint GLCompute %main \"main\" %gid %lid %vs %vv %vo %vmd %vpush %vpsh", .{});
    try em.line("OpExecutionMode %main LocalSize 32 4 1", .{});

    // --- decorations ---
    try em.line("OpDecorate %gid BuiltIn WorkgroupId", .{});
    try em.line("OpDecorate %lid BuiltIn LocalInvocationId", .{});
    try em.line("OpDecorate %arr_f16 ArrayStride 2", .{});
    try em.line("OpDecorate %arr_ss ArrayStride 4", .{});
    try em.line("OpDecorate %arr_o ArrayStride 4", .{});
    try em.line("OpDecorate %arr_md ArrayStride 4", .{});
    for ([_][]const u8{ "ss", "sv", "so", "smd" }) |s| {
        try em.line("OpDecorate %{s} Block", .{s});
        try em.line("OpMemberDecorate %{s} 0 Offset 0", .{s});
    }
    try em.line("OpDecorate %push Block", .{});
    for (0..8) |m| try em.line("OpMemberDecorate %push {d} Offset {d}", .{ m, m * 4 });
    for ([_][]const u8{ "vs", "vv", "vo", "vmd" }, 0..) |v, b| {
        try em.line("OpDecorate %{s} DescriptorSet 0", .{v});
        try em.line("OpDecorate %{s} Binding {d}", .{ v, b });
    }

    // --- types ---
    try em.line("%void = OpTypeVoid", .{});
    try em.line("%fnvoid = OpTypeFunction %void", .{});
    try em.line("%u32 = OpTypeInt 32 0", .{});
    try em.line("%f16 = OpTypeFloat 16", .{});
    try em.line("%f32 = OpTypeFloat 32", .{});
    try em.line("%bool = OpTypeBool", .{});
    try em.line("%v3u = OpTypeVector %u32 3", .{});
    try em.line("%ptr_in_v3 = OpTypePointer Input %v3u", .{});
    try em.line("%gid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%lid = OpVariable %ptr_in_v3 Input", .{});
    try em.line("%c_arrlen = OpConstant %u32 268435456", .{});
    try em.line("%arr_f16 = OpTypeArray %f16 %c_arrlen", .{});
    try em.line("%arr_ss = OpTypeArray %u32 %c_arrlen", .{}); // S words (f16 pairs)
    try em.line("%arr_o = OpTypeArray %f32 %c_arrlen", .{});
    try em.line("%arr_md = OpTypeArray %f32 %c_arrlen", .{});
    try em.line("%v2f16 = OpTypeVector %f16 2", .{});
    try em.line("%ss = OpTypeStruct %arr_ss", .{});
    try em.line("%sv = OpTypeStruct %arr_f16", .{});
    try em.line("%so = OpTypeStruct %arr_o", .{});
    try em.line("%smd = OpTypeStruct %arr_md", .{});
    try em.line("%ptr_ss = OpTypePointer StorageBuffer %ss", .{});
    try em.line("%ptr_sv = OpTypePointer StorageBuffer %sv", .{});
    try em.line("%ptr_so = OpTypePointer StorageBuffer %so", .{});
    try em.line("%ptr_smd = OpTypePointer StorageBuffer %smd", .{});
    try em.line("%vs = OpVariable %ptr_ss StorageBuffer", .{});
    try em.line("%vv = OpVariable %ptr_sv StorageBuffer", .{});
    try em.line("%vo = OpVariable %ptr_so StorageBuffer", .{});
    try em.line("%vmd = OpVariable %ptr_smd StorageBuffer", .{});
    try em.line("%push = OpTypeStruct %u32 %u32 %u32 %u32 %u32 %u32 %u32 %u32", .{});
    try em.line("%ptr_push = OpTypePointer PushConstant %push", .{});
    try em.line("%vpush = OpVariable %ptr_push PushConstant", .{});
    try em.line("%ptr_pc_u32 = OpTypePointer PushConstant %u32", .{});
    try em.line("%ptr_ss_f32 = OpTypePointer StorageBuffer %u32", .{});
    try em.line("%ptr_sv_f16 = OpTypePointer StorageBuffer %f16", .{});
    try em.line("%ptr_so_f32 = OpTypePointer StorageBuffer %f32", .{});
    try em.line("%ptr_smd_f32 = OpTypePointer StorageBuffer %f32", .{});

    // Workgroup P slab: [128][64] f16.
    try em.line("%c_psh_len = OpConstant %u32 8192", .{});
    try em.line("%t_psh = OpTypeArray %f16 %c_psh_len", .{});
    try em.line("%ptr_wg_psh = OpTypePointer Workgroup %t_psh", .{});
    try em.line("%vpsh = OpVariable %ptr_wg_psh Workgroup", .{});
    try em.line("%ptr_wg_f16 = OpTypePointer Workgroup %f16", .{});

    for ([_]struct { []const u8, u32 }{
        .{ "c_u0", 0 },  .{ "c_u1", 1 },   .{ "c_u2", 2 },     .{ "c_u3", 3 },
        .{ "c_u4", 4 },  .{ "c_u5", 5 },   .{ "c_u6", 6 },     .{ "c_u7", 7 },
        .{ "c_u31", 31 }, .{ "c_u16", 16 }, .{ "c_u32c", 32 }, .{ "c_u63", 63 },
        .{ "c_u64", 64 }, .{ "c_u128", 128 }, .{ "c_u1024", 1024 }, .{ "c_u264", 0x108 },
        .{ "c_scope_sub", 3 }, .{ "c_scope_wg", 2 },
    }) |cv| try em.line("%{s} = OpConstant %u32 {d}", .{ cv[0], cv[1] });

    try em.line("%mat_a = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u0", .{});
    try em.line("%mat_b = OpTypeCooperativeMatrixKHR %f16 %c_scope_sub %c_u16 %c_u16 %c_u1", .{});
    try em.line("%mat_c = OpTypeCooperativeMatrixKHR %f32 %c_scope_sub %c_u16 %c_u16 %c_u2", .{});
    try em.line("%c_f32_0 = OpConstant %f32 0", .{});
    try em.line("%c_acc0 = OpConstantComposite %mat_c %c_f32_0", .{});

    // c_k16[ks*16], c_col16[nt*16].
    const c_k16 = [_][]const u8{ "c_u0", "c_u16", "c_u32c", "c_k16_3" };
    try em.line("%c_k16_3 = OpConstant %u32 48", .{});
    var c_col16: [8][]const u8 = .{ "c_u0", "c_u16", "c_u32c", "c_k16_3", "c_u64", "", "", "" };
    for (5..8) |nt| {
        try em.line("%c_col16_{d} = OpConstant %u32 {d}", .{ nt, nt * 16 });
    }
    c_col16[5] = "c_col16_5";
    c_col16[6] = "c_col16_6";
    c_col16[7] = "c_col16_7";

    // --- function ---
    try em.line("%main = OpFunction %void None %fnvoid", .{});
    try em.line("%entry = OpLabel", .{});

    const gidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %gid", .{gidv});
    const tile_r = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ tile_r, gidv });
    const zidx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 2", .{ zidx, gidv });
    const row0 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ row0, tile_r });

    const lidv = em.id();
    try em.line("%t{d} = OpLoad %v3u %lid", .{lidv});
    const lx = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 0", .{ lx, lidv });
    const ly = em.id();
    try em.line("%t{d} = OpCompositeExtract %u32 %t{d} 1", .{ ly, lidv });
    const lymul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u32c", .{ lymul, ly });
    const flat = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ flat, lymul, lx });

    const pnames = [_][]const u8{ "psstride", "psplane", "pheadoff", "pgroup", "pvstride", "postride", "pseq", "pmdplane" };
    for (0..8) |m| {
        const pptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_pc_u32 %vpush %c_u{d}", .{ pptr, m });
        try em.line("%{s} = OpLoad %u32 %t{d}", .{ pnames[m], pptr });
    }

    const head = em.id();
    try em.line("%t{d} = OpIAdd %u32 %pheadoff %t{d}", .{ head, zidx });
    const kvh = em.id();
    try em.line("%t{d} = OpUDiv %u32 %t{d} %pgroup", .{ kvh, head });
    const headmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ headmul, head });
    const kvmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u128", .{ kvmul, kvh });

    // Warp tiling: 2x2 grid of 64x64 tiles.
    const warp_m = em.id();
    try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u1", .{ warp_m, ly });
    const warp_n = em.id();
    try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ warp_n, ly });
    const wm64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wm64, warp_m });
    const row_s = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ row_s, row0, wm64 });
    const wn64 = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ wn64, warp_n });

    // Loop-invariant staging indices (32 f16-pair words per thread).
    const s_zbase = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %psplane", .{ s_zbase, zidx });
    const md_zrow = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %pmdplane", .{ md_zrow, zidx });
    var srow_t: [32]u32 = undefined;
    var mdi_t: [32]u32 = undefined;
    var jcol_t: [32]u32 = undefined;
    var e_t: [32]u32 = undefined;
    var v_prev: u32 = 0;
    for (0..32) |i| {
        var v = flat;
        if (i > 0) {
            const vn = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u128", .{ vn, v_prev });
            v = vn;
        }
        v_prev = v;
        e_t[i] = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u1", .{ e_t[i], v });
        const erow = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u5", .{ erow, v }); // v / 32
        const vmod = em.id();
        try em.line("%t{d} = OpBitwiseAnd %u32 %t{d} %c_u31", .{ vmod, v });
        jcol_t[i] = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u1", .{ jcol_t[i], vmod }); // * 2
        const qrow = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ qrow, row0, erow });
        const qmul = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %psstride", .{ qmul, qrow });
        srow_t[i] = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ srow_t[i], s_zbase, qmul });
        const mdrow = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ mdrow, md_zrow, qrow });
        mdi_t[i] = em.id();
        try em.line("%t{d} = OpShiftLeftLogical %u32 %t{d} %c_u1", .{ mdi_t[i], mdrow }); // *2
    }

    // Pre-allocated ids for the loop-carried values (phi back-edges).
    const k0n = em.id();
    var acc_next: [4][4]u32 = undefined;
    for (&acc_next) |*row| for (row) |*v| {
        v.* = em.id();
    };

    try em.line("OpBranch %head", .{});
    try em.line("%head = OpLabel", .{});
    const k0v = em.id();
    try em.line("%t{d} = OpPhi %u32 %c_u0 %entry %t{d} %cont", .{ k0v, k0n });
    var acc_phi: [4][4]u32 = undefined;
    for (&acc_phi, acc_next) |*prow, nrow| {
        for (prow, nrow) |*ap, an| {
            ap.* = em.id();
            try em.line("%t{d} = OpPhi %mat_c %c_acc0 %entry %t{d} %cont", .{ ap.*, an });
        }
    }
    try em.line("OpLoopMerge %merge %cont None", .{});
    try em.line("OpBranch %cond", .{});
    try em.line("%cond = OpLabel", .{});
    const cmp = em.id();
    try em.line("%t{d} = OpULessThan %bool %t{d} %psstride", .{ cmp, k0v });
    try em.line("OpBranchConditional %t{d} %body %merge", .{cmp});

    try em.line("%body = OpLabel", .{});
    // Stage P = exp(S - m) * invd into the slab.
    for (0..32) |i| {
        const soff = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ soff, srow_t[i], k0v });
        const soff2 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ soff2, soff, jcol_t[i] });
        const widx = em.id();
        try em.line("%t{d} = OpShiftRightLogical %u32 %t{d} %c_u1", .{ widx, soff2 }); // element -> word
        const sptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_ss_f32 %vs %c_u0 %t{d}", .{ sptr, widx });
        const sword = em.id();
        try em.line("%t{d} = OpLoad %u32 %t{d}", .{ sword, sptr });
        const spair = em.id();
        try em.line("%t{d} = OpBitcast %v2f16 %t{d}", .{ spair, sword });
        const mptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_smd_f32 %vmd %c_u0 %t{d}", .{ mptr, mdi_t[i] });
        const mval = em.id();
        try em.line("%t{d} = OpLoad %f32 %t{d}", .{ mval, mptr });
        const mdi1 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ mdi1, mdi_t[i] });
        const dptr = em.id();
        try em.line("%t{d} = OpAccessChain %ptr_smd_f32 %vmd %c_u0 %t{d}", .{ dptr, mdi1 });
        const dval = em.id();
        try em.line("%t{d} = OpLoad %f32 %t{d}", .{ dval, dptr });
        const jj0 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ jj0, k0v, jcol_t[i] });
        for (0..2) |half| {
            const hf = em.id();
            try em.line("%t{d} = OpCompositeExtract %f16 %t{d} {d}", .{ hf, spair, half });
            const sval = em.id();
            try em.line("%t{d} = OpFConvert %f32 %t{d}", .{ sval, hf });
            const shifted = em.id();
            try em.line("%t{d} = OpFSub %f32 %t{d} %t{d}", .{ shifted, sval, mval });
            const eval = em.id();
            try em.line("%t{d} = OpExtInst %f32 %glsl 27 %t{d}", .{ eval, shifted }); // Exp
            const pval = em.id();
            try em.line("%t{d} = OpFMul %f32 %t{d} %t{d}", .{ pval, eval, dval });
            var jj = jj0;
            if (half == 1) {
                const j1 = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ j1, jj0 });
                jj = j1;
            }
            const j_ok = em.id();
            try em.line("%t{d} = OpULessThan %bool %t{d} %pseq", .{ j_ok, jj });
            const pclamp = em.id();
            try em.line("%t{d} = OpSelect %f32 %t{d} %t{d} %c_f32_0", .{ pclamp, j_ok, pval });
            const hval = em.id();
            try em.line("%t{d} = OpFConvert %f16 %t{d}", .{ hval, pclamp });
            var eidx = e_t[i];
            if (half == 1) {
                const e1 = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1", .{ e1, e_t[i] });
                eidx = e1;
            }
            const pptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vpsh %t{d}", .{ pptr, eidx });
            try em.line("OpStore %t{d} %t{d}", .{ pptr, hval });
        }
    }
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});

    // 4 j sub-steps: P tiles from the slab, V tiles from global.
    var acc_cur = acc_phi;
    const a_shbase = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %c_u64", .{ a_shbase, wm64 });
    for (0..4) |ks| {
        var a_off = a_shbase;
        if (ks > 0) {
            const ao = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ ao, a_shbase, c_k16[ks] });
            a_off = ao;
        }
        var ma: [4]u32 = undefined;
        var ao_cur = a_off;
        for (0..4) |r| {
            if (r > 0) {
                const a2 = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u1024", .{ a2, ao_cur });
                ao_cur = a2;
            }
            const aptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_wg_f16 %vpsh %t{d}", .{ aptr, ao_cur });
            ma[r] = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_a %t{d} %c_u0 %c_u64", .{ ma[r], aptr });
        }
        const jrow = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ jrow, k0v, c_k16[ks] });
        const jmul = em.id();
        try em.line("%t{d} = OpIMul %u32 %t{d} %pvstride", .{ jmul, jrow });
        const vb0 = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ vb0, jmul, kvmul });
        const vbase = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ vbase, vb0, wn64 });
        for (0..4) |nt| {
            var v_off = vbase;
            if (nt > 0) {
                const vo = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ vo, vbase, c_col16[nt] });
                v_off = vo;
            }
            const vptr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_sv_f16 %vv %c_u0 %t{d}", .{ vptr, v_off });
            const mb = em.id();
            try em.line("%t{d} = OpCooperativeMatrixLoadKHR %mat_b %t{d} %c_u0 %pvstride", .{ mb, vptr });
            for (0..4) |r| {
                const acc_out = if (ks == 3) acc_next[r][nt] else em.id();
                try em.line("%t{d} = OpCooperativeMatrixMulAddKHR %mat_c %t{d} %t{d} %t{d}", .{ acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try em.line("OpControlBarrier %c_scope_wg %c_scope_wg %c_u264", .{});
    try em.line("OpBranch %cont", .{});

    try em.line("%cont = OpLabel", .{});
    try em.line("%t{d} = OpIAdd %u32 %t{d} %c_u64", .{ k0n, k0v });
    try em.line("OpBranch %head", .{});

    // merge: store 4x4 OUT tiles.
    try em.line("%merge = OpLabel", .{});
    const orn16 = em.id();
    try em.line("%t{d} = OpIMul %u32 %c_u16 %postride", .{orn16});
    var o_rowmul = em.id();
    try em.line("%t{d} = OpIMul %u32 %t{d} %postride", .{ o_rowmul, row_s });
    const o_head = em.id();
    try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ o_head, headmul, wn64 });
    for (0..4) |r| {
        if (r > 0) {
            const nx = em.id();
            try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ nx, o_rowmul, orn16 });
            o_rowmul = nx;
        }
        const o_base = em.id();
        try em.line("%t{d} = OpIAdd %u32 %t{d} %t{d}", .{ o_base, o_rowmul, o_head });
        for (0..4) |nt| {
            var ob = o_base;
            if (nt > 0) {
                const oc = em.id();
                try em.line("%t{d} = OpIAdd %u32 %t{d} %{s}", .{ oc, o_base, c_col16[nt] });
                ob = oc;
            }
            const optr = em.id();
            try em.line("%t{d} = OpAccessChain %ptr_so_f32 %vo %c_u0 %t{d}", .{ optr, ob });
            try em.line("OpCooperativeMatrixStoreKHR %t{d} %t{d} %c_u0 %postride", .{ optr, acc_phi[r][nt] });
        }
    }
    try em.line("OpReturn", .{});
    try em.line("OpFunctionEnd", .{});

    return sasm.assembleChecked(gpa, sasm.version_1_5, t.items);
}
