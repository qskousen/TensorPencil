//! Hand-assembled SPIR-V cooperative-matrix GEMM (tensor cores).
//!
//! Zig's SPIR-V backend cannot express OpTypeCooperativeMatrixKHR, so this
//! module is built instruction-by-instruction at runtime. One subgroup (a
//! 32-thread workgroup) computes a 16x16 f32 tile of C = A @ B with f16
//! operands: A is x in row-major f16 [m][k]; B is the k-major f16 weight
//! layout [k][stride] (exactly our transposed-weight convention, so B needs
//! no further rearranging); C stores f32 directly into the output buffer.
//!
//! All dimensions must be multiples of 16 (callers pad m; n/k already are).
//! Bindings (set 0): 0 = B (weights, f16), 1 = A (x, f16), 2 = C (y, f32) —
//! matching the regular matmul binding order. Push: {m, n, k, stride} u32.

const std = @import("std");

const Asm = struct {
    words: std.ArrayList(u32) = .empty,
    gpa: std.mem.Allocator,
    next: u32 = 1,

    fn id(self: *Asm) u32 {
        const r = self.next;
        self.next += 1;
        return r;
    }

    fn op(self: *Asm, opcode: u16, operands: []const u32) !void {
        try self.words.append(self.gpa, (@as(u32, @intCast(operands.len + 1)) << 16) | opcode);
        try self.words.appendSlice(self.gpa, operands);
    }

    /// Opcode with a trailing string literal (null-terminated, word-padded).
    fn opStr(self: *Asm, opcode: u16, pre: []const u32, s: []const u8) !void {
        const str_words = s.len / 4 + 1;
        try self.words.append(self.gpa, (@as(u32, @intCast(pre.len + str_words + 1)) << 16) | opcode);
        try self.words.appendSlice(self.gpa, pre);
        var i: usize = 0;
        while (i < str_words * 4) : (i += 4) {
            var w: u32 = 0;
            for (0..4) |b| {
                if (i + b < s.len) w |= @as(u32, s[i + b]) << @intCast(8 * b);
            }
            try self.words.append(self.gpa, w);
        }
    }
};

/// Build the module; caller frees the returned words as bytes.
pub fn buildGemm(gpa: std.mem.Allocator) ![]align(4) u8 {
    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    // --- id pre-allocation ---------------------------------------------
    const main_fn = a.id();
    const gid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_f16 = a.id();
    const t_f32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f16 = a.id();
    const t_arr_f32 = a.id();
    const t_sb = a.id(); // struct { [N]f16 } (B / weights)
    const t_sa = a.id(); // struct { [N]f16 } (A / x) — distinct id, same shape
    const t_sc = a.id(); // struct { [N]f32 }
    const t_ptr_sb = a.id();
    const t_ptr_sa = a.id();
    const t_ptr_sc = a.id();
    const v_b = a.id();
    const v_a = a.id();
    const v_c = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_sb_f16 = a.id();
    const t_ptr_sa_f16 = a.id();
    const t_ptr_sc_f32 = a.id();
    const c_u0 = a.id();
    const c_u1 = a.id();
    const c_u2 = a.id();
    const c_u3 = a.id();
    const c_u16 = a.id();
    const c_scope_sub = a.id(); // 3
    const c_use_a = a.id(); // 0 (reuses value of c_u0 but must be distinct? use c_u0)
    _ = c_use_a;
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const c_f32_0 = a.id();
    const c_acc0 = a.id();
    const t_ptr_fn_matc = a.id();
    const t_ptr_fn_u32 = a.id();

    // --- header ----------------------------------------------------------
    // magic, version 1.5, generator 0, bound (patched at end), 0
    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });

    try a.op(17, &.{1}); // OpCapability Shader
    try a.op(17, &.{9}); // Float16
    try a.op(17, &.{5345}); // VulkanMemoryModel
    try a.op(17, &.{6022}); // CooperativeMatrixKHR
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_16bit_storage");
    try a.op(14, &.{ 0, 3 }); // OpMemoryModel Logical Vulkan
    // OpEntryPoint GLCompute %main "main" <interface: gid, buffers, push>
    {
        var pre: [2]u32 = .{ 5, main_fn };
        var post: [5]u32 = .{ gid_var, v_b, v_a, v_c, v_push };
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &pre);
        // name "main\0" = 2 words
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &post);
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, 1, 1 }); // ExecutionMode LocalSize 32 1 1

    // --- decorations -------------------------------------------------------
    try a.op(71, &.{ gid_var, 11, 26 }); // BuiltIn WorkgroupId
    try a.op(71, &.{ t_arr_f16, 6, 2 }); // ArrayStride 2
    try a.op(71, &.{ t_arr_f32, 6, 4 });
    inline for (.{ t_sb, t_sa, t_sc }) |t| {
        try a.op(71, &.{ t, 2 }); // Block
        try a.op(72, &.{ t, 0, 35, 0 }); // member 0 Offset 0
    }
    try a.op(71, &.{ t_push, 2 }); // push Block
    inline for (0..4) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    inline for (.{ v_b, v_a, v_c }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 }); // DescriptorSet 0
        try a.op(71, &.{ v, 33, @intCast(binding) }); // Binding
    }

    // --- types / constants --------------------------------------------------
    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u }); // Input
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f16, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_f32, t_f32, c_arrlen });
    try a.op(30, &.{ t_sb, t_arr_f16 });
    try a.op(30, &.{ t_sa, t_arr_f16 });
    try a.op(30, &.{ t_sc, t_arr_f32 });
    try a.op(32, &.{ t_ptr_sb, 12, t_sb }); // StorageBuffer
    try a.op(32, &.{ t_ptr_sa, 12, t_sa });
    try a.op(32, &.{ t_ptr_sc, 12, t_sc });
    try a.op(59, &.{ t_ptr_sb, v_b, 12 });
    try a.op(59, &.{ t_ptr_sa, v_a, 12 });
    try a.op(59, &.{ t_ptr_sc, v_c, 12 });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push }); // PushConstant
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_sb_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_sa_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_sc_f32, 12, t_f32 });

    try a.op(43, &.{ t_u32, c_u0, 0 });
    try a.op(43, &.{ t_u32, c_u1, 1 });
    try a.op(43, &.{ t_u32, c_u2, 2 });
    try a.op(43, &.{ t_u32, c_u3, 3 });
    try a.op(43, &.{ t_u32, c_u16, 16 });
    try a.op(43, &.{ t_u32, c_scope_sub, 3 }); // Scope Subgroup
    const t_bool = a.id();
    try a.op(20, &.{t_bool});

    // OpTypeCooperativeMatrixKHR: component, scope, rows, cols, use.
    try a.op(4456, &.{ t_mat_a, t_f16, c_scope_sub, c_u16, c_u16, c_u0 });
    try a.op(4456, &.{ t_mat_b, t_f16, c_scope_sub, c_u16, c_u16, c_u1 });
    try a.op(4456, &.{ t_mat_c, t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(43, &.{ t_f32, c_f32_0, 0 }); // f32 0.0 bits
    try a.op(44, &.{ t_mat_c, c_acc0, c_f32_0 }); // OpConstantComposite (replicated)
    try a.op(32, &.{ t_ptr_fn_matc, 7, t_mat_c }); // Function
    try a.op(32, &.{ t_ptr_fn_u32, 7, t_u32 });

    // --- function ------------------------------------------------------------
    const lb_entry = a.id();
    const lb_head = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();

    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid }); // OpFunction
    try a.op(248, &.{lb_entry}); // OpLabel

    const acc_var = a.id();
    const k0_var = a.id();
    try a.op(59, &.{ t_ptr_fn_matc, acc_var, 7 });
    try a.op(59, &.{ t_ptr_fn_u32, k0_var, 7 });

    // gid -> col0/row0
    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_c = a.id();
    const tile_r = a.id();
    try a.op(81, &.{ t_u32, tile_c, gidv, 0 }); // CompositeExtract x
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    const col0 = a.id();
    const row0 = a.id();
    try a.op(132, &.{ t_u32, col0, tile_c, c_u16 }); // IMul
    try a.op(132, &.{ t_u32, row0, tile_r, c_u16 });

    // push loads: m(0) n(1) k(2) stride(3)
    var push_vals: [4]u32 = undefined;
    inline for (0..4) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c_u0,
            1 => c_u1,
            2 => c_u2,
            else => c_u3,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_n = push_vals[1];
    const p_k = push_vals[2];
    const p_stride = push_vals[3];

    // a_row_base = row0 * k; c_base = row0 * n + col0
    const a_row_base = a.id();
    try a.op(132, &.{ t_u32, a_row_base, row0, p_k });
    const c_rowmul = a.id();
    const c_base = a.id();
    try a.op(132, &.{ t_u32, c_rowmul, row0, p_n });
    try a.op(128, &.{ t_u32, c_base, c_rowmul, col0 }); // IAdd

    try a.op(62, &.{ acc_var, c_acc0 }); // OpStore acc = 0
    try a.op(62, &.{ k0_var, c_u0 });
    try a.op(249, &.{lb_head}); // OpBranch

    // loop header
    try a.op(248, &.{lb_head});
    try a.op(246, &.{ lb_merge, lb_cont, 0 }); // OpLoopMerge
    const cond_blk = a.id();
    try a.op(249, &.{cond_blk});
    try a.op(248, &.{cond_blk});
    const k0v = a.id();
    try a.op(61, &.{ t_u32, k0v, k0_var });
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, k0v, p_k }); // ULessThan
    try a.op(250, &.{ cmp, lb_body, lb_merge }); // BranchConditional

    // body
    try a.op(248, &.{lb_body});
    const a_off = a.id();
    try a.op(128, &.{ t_u32, a_off, a_row_base, k0v });
    const a_ptr = a.id();
    try a.op(65, &.{ t_ptr_sa_f16, a_ptr, v_a, c_u0, a_off });
    const ma = a.id();
    // OpCooperativeMatrixLoadKHR: result-type, result, ptr, layout(id), stride(id)
    try a.op(4457, &.{ t_mat_a, ma, a_ptr, c_u0, p_k });
    const b_rowmul = a.id();
    try a.op(132, &.{ t_u32, b_rowmul, k0v, p_stride });
    const b_off = a.id();
    try a.op(128, &.{ t_u32, b_off, b_rowmul, col0 });
    const b_ptr = a.id();
    try a.op(65, &.{ t_ptr_sb_f16, b_ptr, v_b, c_u0, b_off });
    const mb = a.id();
    try a.op(4457, &.{ t_mat_b, mb, b_ptr, c_u0, p_stride });
    const acc_in = a.id();
    try a.op(61, &.{ t_mat_c, acc_in, acc_var });
    const acc_out = a.id();
    try a.op(4459, &.{ t_mat_c, acc_out, ma, mb, acc_in }); // MulAdd
    try a.op(62, &.{ acc_var, acc_out });
    try a.op(249, &.{lb_cont});

    // continue: k0 += 16
    try a.op(248, &.{lb_cont});
    const k0n = a.id();
    try a.op(128, &.{ t_u32, k0n, k0v, c_u16 });
    try a.op(62, &.{ k0_var, k0n });
    try a.op(249, &.{lb_head});

    // merge: store C
    try a.op(248, &.{lb_merge});
    const c_ptr = a.id();
    try a.op(65, &.{ t_ptr_sc_f32, c_ptr, v_c, c_u0, c_base });
    const acc_fin = a.id();
    try a.op(61, &.{ t_mat_c, acc_fin, acc_var });
    // OpCooperativeMatrixStoreKHR: ptr, object, layout(id), stride(id)
    try a.op(4458, &.{ c_ptr, acc_fin, c_u0, p_n });
    try a.op(253, &.{}); // OpReturn
    try a.op(56, &.{}); // OpFunctionEnd

    // patch bound
    a.words.items[3] = a.next;

    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
}

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
    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    // u32 constant pool: SPIR-V forbids duplicate OpConstant, so intern by
    // value. Emitted in the types section below; referenced via `cu`.
    var cvals: [40]u32 = undefined;
    var cids: [40]u32 = undefined;
    var cn: usize = 0;
    const addC = struct {
        fn f(vals: []u32, ids: []u32, n: *usize, asm_: *Asm, v: u32) void {
            for (vals[0..n.*]) |ex| if (ex == v) return;
            vals[n.*] = v;
            ids[n.*] = asm_.id();
            n.* += 1;
        }
    }.f;
    addC(&cvals, &cids, &cn, &a, 0);
    addC(&cvals, &cids, &cn, &a, 1);
    addC(&cvals, &cids, &cn, &a, 2);
    addC(&cvals, &cids, &cn, &a, 3);
    addC(&cvals, &cids, &cn, &a, 16);
    addC(&cvals, &cids, &cn, &a, 32);
    addC(&cvals, &cids, &cn, &a, 16 * mt);
    addC(&cvals, &cids, &cn, &a, 16 * nt);
    for (0..mt) |i| addC(&cvals, &cids, &cn, &a, @intCast(16 * i));
    for (0..nt) |j| addC(&cvals, &cids, &cn, &a, @intCast(16 * j));
    const cu = struct {
        fn f(vals: []const u32, ids: []const u32, n: usize, v: u32) u32 {
            for (0..n) |i| if (vals[i] == v) return ids[i];
            unreachable;
        }
    }.f;

    const main_fn = a.id();
    const gid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_s8 = a.id();
    const t_s32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_s8 = a.id();
    const t_arr_s32 = a.id();
    const t_sb = a.id(); // struct { [N]s8 } (B / weights)
    const t_sa = a.id(); // struct { [N]s8 } (A / x)
    const t_sc = a.id(); // struct { [N]s32 } (C / y)
    const t_ptr_sb = a.id();
    const t_ptr_sa = a.id();
    const t_ptr_sc = a.id();
    const v_b = a.id();
    const v_a = a.id();
    const v_c = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_sb_s8 = a.id();
    const t_ptr_sa_s8 = a.id();
    const t_ptr_sc_s32 = a.id();
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const c_s32_0 = a.id();
    const c_acc0 = a.id();
    const t_ptr_fn_matc = a.id();
    const t_ptr_fn_u32 = a.id();
    const t_bool = a.id();

    // --- header ----------------------------------------------------------
    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });

    try a.op(17, &.{1}); // OpCapability Shader
    try a.op(17, &.{39}); // Int8
    try a.op(17, &.{4448}); // StorageBuffer8BitAccess
    try a.op(17, &.{5345}); // VulkanMemoryModel
    try a.op(17, &.{6022}); // CooperativeMatrixKHR
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_8bit_storage");
    try a.op(14, &.{ 0, 3 }); // OpMemoryModel Logical Vulkan
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, v_b, v_a, v_c, v_push });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, 1, 1 }); // ExecutionMode LocalSize 32 1 1

    // --- decorations -------------------------------------------------------
    try a.op(71, &.{ gid_var, 11, 26 }); // BuiltIn WorkgroupId
    try a.op(71, &.{ t_arr_s8, 6, 1 }); // ArrayStride 1
    try a.op(71, &.{ t_arr_s32, 6, 4 }); // ArrayStride 4
    inline for (.{ t_sb, t_sa, t_sc }) |t| {
        try a.op(71, &.{ t, 2 }); // Block
        try a.op(72, &.{ t, 0, 35, 0 }); // member 0 Offset 0
    }
    try a.op(71, &.{ t_push, 2 });
    inline for (0..4) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    inline for (.{ v_b, v_a, v_c }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 }); // DescriptorSet 0
        try a.op(71, &.{ v, 33, @intCast(binding) }); // Binding
    }

    // --- types / constants --------------------------------------------------
    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 }); // unsigned 32 (indices)
    try a.op(21, &.{ t_s8, 8, 1 }); // signed 8
    try a.op(21, &.{ t_s32, 32, 1 }); // signed 32 (accumulator)
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u }); // Input
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_s8, t_s8, c_arrlen });
    try a.op(28, &.{ t_arr_s32, t_s32, c_arrlen });
    try a.op(30, &.{ t_sb, t_arr_s8 });
    try a.op(30, &.{ t_sa, t_arr_s8 });
    try a.op(30, &.{ t_sc, t_arr_s32 });
    try a.op(32, &.{ t_ptr_sb, 12, t_sb }); // StorageBuffer
    try a.op(32, &.{ t_ptr_sa, 12, t_sa });
    try a.op(32, &.{ t_ptr_sc, 12, t_sc });
    try a.op(59, &.{ t_ptr_sb, v_b, 12 });
    try a.op(59, &.{ t_ptr_sa, v_a, 12 });
    try a.op(59, &.{ t_ptr_sc, v_c, 12 });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push }); // PushConstant
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_sb_s8, 12, t_s8 });
    try a.op(32, &.{ t_ptr_sa_s8, 12, t_s8 });
    try a.op(32, &.{ t_ptr_sc_s32, 12, t_s32 });

    for (0..cn) |i| try a.op(43, &.{ t_u32, cids[i], cvals[i] });
    const c0 = cu(&cvals, &cids, cn, 0);
    const c16 = cu(&cvals, &cids, cn, 16);
    const c32 = cu(&cvals, &cids, cn, 32);
    const c_scope = cu(&cvals, &cids, cn, 3); // Scope Subgroup == 3
    try a.op(20, &.{t_bool});

    // OpTypeCooperativeMatrixKHR: component, scope, rows, cols, use.
    // A = 16x32 (use 0), B = 32x16 (use 1), C = 16x16 (use 2).
    try a.op(4456, &.{ t_mat_a, t_s8, c_scope, c16, c32, c0 });
    try a.op(4456, &.{ t_mat_b, t_s8, c_scope, c32, c16, cu(&cvals, &cids, cn, 1) });
    try a.op(4456, &.{ t_mat_c, t_s32, c_scope, c16, c16, cu(&cvals, &cids, cn, 2) });
    try a.op(43, &.{ t_s32, c_s32_0, 0 });
    try a.op(44, &.{ t_mat_c, c_acc0, c_s32_0 }); // acc filled with 0
    try a.op(32, &.{ t_ptr_fn_matc, 7, t_mat_c }); // Function
    try a.op(32, &.{ t_ptr_fn_u32, 7, t_u32 });

    // --- function ------------------------------------------------------------
    const lb_entry = a.id();
    const lb_head = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();

    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid }); // OpFunction
    try a.op(248, &.{lb_entry});

    // Function-local variables (all in the first block): mt*nt accumulators + k0.
    const nt_tiles = mt * nt;
    var acc_var: [64]u32 = undefined;
    for (0..nt_tiles) |i| {
        acc_var[i] = a.id();
        try a.op(59, &.{ t_ptr_fn_matc, acc_var[i], 7 });
    }
    const k0_var = a.id();
    try a.op(59, &.{ t_ptr_fn_u32, k0_var, 7 });

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_c = a.id();
    const tile_r = a.id();
    try a.op(81, &.{ t_u32, tile_c, gidv, 0 });
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    const col0 = a.id();
    const row0 = a.id();
    try a.op(132, &.{ t_u32, col0, tile_c, cu(&cvals, &cids, cn, 16 * nt) });
    try a.op(132, &.{ t_u32, row0, tile_r, cu(&cvals, &cids, cn, 16 * mt) });

    var push_vals: [4]u32 = undefined;
    inline for (0..4) |m| {
        const pptr = a.id();
        const pval = a.id();
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cu(&cvals, &cids, cn, @intCast(m)) });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_n = push_vals[1];
    const p_k = push_vals[2];
    const p_stride = push_vals[3];

    // Per-tile invariants: A row base [mi] (row*k), B column [nj] (col0+16*nj),
    // C base [mi][nj] (row*n + col). row = row0 + 16*mi.
    var a_row_base: [8]u32 = undefined;
    var c_row_n: [8]u32 = undefined;
    for (0..mt) |mi| {
        const rowmi = a.id();
        try a.op(128, &.{ t_u32, rowmi, row0, cu(&cvals, &cids, cn, @intCast(16 * mi)) });
        a_row_base[mi] = a.id();
        try a.op(132, &.{ t_u32, a_row_base[mi], rowmi, p_k });
        c_row_n[mi] = a.id();
        try a.op(132, &.{ t_u32, c_row_n[mi], rowmi, p_n });
    }
    var col_n: [8]u32 = undefined;
    for (0..nt) |nj| {
        col_n[nj] = a.id();
        try a.op(128, &.{ t_u32, col_n[nj], col0, cu(&cvals, &cids, cn, @intCast(16 * nj)) });
    }

    for (0..nt_tiles) |i| try a.op(62, &.{ acc_var[i], c_acc0 });
    try a.op(62, &.{ k0_var, c0 });
    try a.op(249, &.{lb_head});

    try a.op(248, &.{lb_head});
    try a.op(246, &.{ lb_merge, lb_cont, 0 }); // OpLoopMerge
    const cond_blk = a.id();
    try a.op(249, &.{cond_blk});
    try a.op(248, &.{cond_blk});
    const k0v = a.id();
    try a.op(61, &.{ t_u32, k0v, k0_var });
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, k0v, p_k }); // ULessThan
    try a.op(250, &.{ cmp, lb_body, lb_merge });

    try a.op(248, &.{lb_body});
    // Load mt A fragments and nt B fragments for this k-slab.
    var ma: [8]u32 = undefined;
    for (0..mt) |mi| {
        const a_off = a.id();
        try a.op(128, &.{ t_u32, a_off, a_row_base[mi], k0v });
        const a_ptr = a.id();
        try a.op(65, &.{ t_ptr_sa_s8, a_ptr, v_a, c0, a_off });
        ma[mi] = a.id();
        try a.op(4457, &.{ t_mat_a, ma[mi], a_ptr, c0, p_k }); // Load A RowMajor
    }
    var mb: [8]u32 = undefined;
    const b_rowmul = a.id();
    try a.op(132, &.{ t_u32, b_rowmul, k0v, p_stride });
    for (0..nt) |nj| {
        const b_off = a.id();
        try a.op(128, &.{ t_u32, b_off, b_rowmul, col_n[nj] });
        const b_ptr = a.id();
        try a.op(65, &.{ t_ptr_sb_s8, b_ptr, v_b, c0, b_off });
        mb[nj] = a.id();
        try a.op(4457, &.{ t_mat_b, mb[nj], b_ptr, c0, p_stride }); // Load B RowMajor
    }
    // mt*nt MMAs, reusing the loaded fragments.
    for (0..mt) |mi| {
        for (0..nt) |nj| {
            const ti = mi * nt + nj;
            const acc_in = a.id();
            try a.op(61, &.{ t_mat_c, acc_in, acc_var[ti] });
            const acc_out = a.id();
            // MulAdd, signed operands mask (A|B|C|Result signed = 0xF).
            try a.op(4459, &.{ t_mat_c, acc_out, ma[mi], mb[nj], acc_in, 0xF });
            try a.op(62, &.{ acc_var[ti], acc_out });
        }
    }
    try a.op(249, &.{lb_cont});

    try a.op(248, &.{lb_cont});
    const k0n = a.id();
    try a.op(128, &.{ t_u32, k0n, k0v, c32 }); // k0 += 32
    try a.op(62, &.{ k0_var, k0n });
    try a.op(249, &.{lb_head});

    try a.op(248, &.{lb_merge});
    for (0..mt) |mi| {
        for (0..nt) |nj| {
            const ti = mi * nt + nj;
            const c_base = a.id();
            try a.op(128, &.{ t_u32, c_base, c_row_n[mi], col_n[nj] });
            const c_ptr = a.id();
            try a.op(65, &.{ t_ptr_sc_s32, c_ptr, v_c, c0, c_base });
            const acc_fin = a.id();
            try a.op(61, &.{ t_mat_c, acc_fin, acc_var[ti] });
            try a.op(4458, &.{ c_ptr, acc_fin, c0, p_n }); // Store C RowMajor
        }
    }
    try a.op(253, &.{}); // OpReturn
    try a.op(56, &.{}); // OpFunctionEnd

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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
    const WGM: u32 = 128;
    const WGN: u32 = 128;
    const K_STEP: u32 = 64;
    const NWARPS: u32 = if (warps8) 8 else 4;
    const THREADS: u32 = 32 * NWARPS;
    // Warp grid: 2 warps along m (each 64 rows), NWARPS/2 along n. Each warp
    // is MT=4 x NT fragments (16x16). 4-warp: 2x2 grid, 64x64/warp (NT=4, 16
    // accs/warp = 128 s32/thread, ~1 wg/SM). 8-warp: 2x4 grid, 64x32/warp
    // (NT=2, 8 accs = 64 s32/thread, ~2 wgs/SM — the occupancy lever, since
    // int8 accumulators are forced s32 and can't shrink like fp8's f16 accs).
    const WARP_N: u32 = NWARPS / 2; // warps along n
    const WARP_W: u32 = WGN / WARP_N; // n-cols per warp (64 or 32)
    const MT: u32 = 4;
    const NT: u32 = WARP_W / 16; // 4 or 2
    // Shared u32 layout: A [128 rows][16 u32]=2048, then B [64 krows][32 u32]=2048.
    const A_U32: u32 = WGM * (K_STEP / 4); // 2048; B slab starts here (B_BASE)
    const SH_LEN: u32 = A_U32 + K_STEP * (WGN / 4); // 4096 (16 KB)
    const A_QUADS: u32 = A_U32 / THREADS; // u32/thread for A staging (16 or 8)
    const B_QUADS: u32 = (K_STEP * (WGN / 4)) / THREADS; // (16 or 8) for B staging

    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    // --- constant interning (u32 pool) ---
    var cvals: [96]u32 = undefined;
    var cids: [96]u32 = undefined;
    var cn: usize = 0;
    const addC = struct {
        fn f(vals: []u32, ids: []u32, n: *usize, asm_: *Asm, v: u32) void {
            for (vals[0..n.*]) |ex| if (ex == v) return;
            vals[n.*] = v;
            ids[n.*] = asm_.id();
            n.* += 1;
        }
    }.f;
    // scalars used everywhere
    for ([_]u32{ 0, 1, 2, 3, 4, 5, 8, 15, 16, 31, 32, 48, 64, 128, 256, 512, 768, 1024, 2048, 0x108 }) |v|
        addC(&cvals, &cids, &cn, &a, v);
    // staging per-thread chunk strides t*THREADS
    for (0..@max(A_QUADS, B_QUADS)) |t| addC(&cvals, &cids, &cn, &a, @intCast(t * THREADS));
    // A frag offsets mi*256, ks*8 ; B frag offsets ks*1024, nj*4, B_BASE
    for (0..MT) |mi| addC(&cvals, &cids, &cn, &a, @intCast(mi * 256));
    for (0..NT) |nj| addC(&cvals, &cids, &cn, &a, @intCast(nj * 4));
    addC(&cvals, &cids, &cn, &a, WARP_W / 4); // warp_n u32 column step
    // Stage A (fuse_scale): per-subgroup s32 scratch base ly*256, scratch len,
    // and the 8 interleaved lane strides i*32 (256 elems / 32 lanes).
    if (fuse_scale) {
        addC(&cvals, &cids, &cn, &a, NWARPS * 256);
        for (0..8) |i| addC(&cvals, &cids, &cn, &a, @intCast(i * 32));
    }
    const cu = struct {
        fn f(vals: []const u32, ids: []const u32, n: usize, v: u32) u32 {
            for (0..n) |i| if (vals[i] == v) return ids[i];
            unreachable;
        }
    }.f;

    const main_fn = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_s8 = a.id();
    const t_s32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const t_bool = a.id();
    const c_arrlen = a.id();
    const t_arr_u32 = a.id(); // storage-buffer u32 arrays (A/B views)
    const t_arr_s32 = a.id();
    const t_sb = a.id(); // struct{[N]u32} B view (binding 0)
    const t_sa = a.id(); // struct{[N]u32} A view (binding 1)
    const t_sc = a.id(); // struct{[N]s32} C (binding 2)
    const t_ptr_sb = a.id();
    const t_ptr_sa = a.id();
    const t_ptr_sc = a.id();
    const v_b = a.id();
    const v_a = a.id();
    const v_c = a.id();
    const t_ptr_sb_u32 = a.id();
    const t_ptr_sa_u32 = a.id();
    const t_ptr_sc_s32 = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    // workgroup u32 array
    const c_shlen = a.id();
    const t_sh = a.id();
    const t_ptr_wg_sh = a.id();
    const v_sh = a.id();
    const t_ptr_wg_u32 = a.id();
    // coop matrix types
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const c_s32_0 = a.id();
    const c_acc0 = a.id();
    // Stage A (fuse_scale) extras: C stores f32, binding 3 = scale buffer
    // [act(m_pad) | weight(rows)] f32, and a per-subgroup s32 workgroup scratch
    // that the 16x16 C fragments coop-store into so the 32 lanes can rescale +
    // write f32 element-wise (act[row]*weight[col]).
    const t_f32 = a.id();
    const t_f16 = a.id(); // C store type when c_h16
    const t_arr_f32 = a.id();
    const t_scale = a.id();
    const t_ptr_scale = a.id();
    const v_scale = a.id();
    const t_ptr_scale_f32 = a.id();
    const c_scrlen = a.id();
    const t_scr = a.id();
    const t_ptr_wg_scr = a.id();
    const v_scr = a.id();
    const t_ptr_wg_s32 = a.id();

    // --- header ---
    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1}); // Shader
    try a.op(17, &.{39}); // Int8
    if (c_h16) try a.op(17, &.{9}); // Float16 (C stored f16)
    try a.op(17, &.{5345}); // VulkanMemoryModel
    try a.op(17, &.{6022}); // CooperativeMatrixKHR
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.op(14, &.{ 0, 3 }); // Logical Vulkan
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_b, v_a, v_c, v_push, v_sh });
        if (fuse_scale) try buf.appendSlice(gpa, &.{ v_scale, v_scr });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, NWARPS, 1 }); // LocalSize 32 NWARPS 1

    // --- decorations ---
    try a.op(71, &.{ gid_var, 11, 26 }); // WorkgroupId
    try a.op(71, &.{ lid_var, 11, 27 }); // LocalInvocationId
    try a.op(71, &.{ t_arr_u32, 6, 4 }); // ArrayStride 4
    try a.op(71, &.{ t_arr_s32, 6, if (c_h16) 2 else 4 }); // C stride: f16=2 else 4
    inline for (.{ t_sb, t_sa, t_sc }) |t| {
        try a.op(71, &.{ t, 2 }); // Block
        try a.op(72, &.{ t, 0, 35, 0 }); // member 0 Offset 0
    }
    try a.op(71, &.{ t_push, 2 });
    inline for (0..4) |m| try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    inline for (.{ v_b, v_a, v_c }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 }); // DescriptorSet 0
        try a.op(71, &.{ v, 33, @intCast(binding) });
    }
    if (fuse_scale) {
        try a.op(71, &.{ t_arr_f32, 6, 4 }); // ArrayStride 4
        try a.op(71, &.{ t_scale, 2 }); // Block
        try a.op(72, &.{ t_scale, 0, 35, 0 });
        try a.op(71, &.{ v_scale, 34, 0 });
        try a.op(71, &.{ v_scale, 33, 3 }); // binding 3 = scale buffer
    }

    // --- types / constants ---
    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(21, &.{ t_s8, 8, 1 });
    try a.op(21, &.{ t_s32, 32, 1 });
    try a.op(20, &.{t_bool});
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    if (fuse_scale) try a.op(22, &.{ t_f32, 32 }); // OpTypeFloat 32
    if (c_h16) try a.op(22, &.{ t_f16, 16 }); // OpTypeFloat 16
    const t_c_elem = if (c_h16) t_f16 else if (fuse_scale) t_f32 else t_s32;

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_u32, t_u32, c_arrlen });
    // C array element: f16 (c_h16), f32 (fuse_scale), or s32 (raw).
    try a.op(28, &.{ t_arr_s32, t_c_elem, c_arrlen });
    try a.op(30, &.{ t_sb, t_arr_u32 });
    try a.op(30, &.{ t_sa, t_arr_u32 });
    try a.op(30, &.{ t_sc, t_arr_s32 });
    try a.op(32, &.{ t_ptr_sb, 12, t_sb });
    try a.op(32, &.{ t_ptr_sa, 12, t_sa });
    try a.op(32, &.{ t_ptr_sc, 12, t_sc });
    try a.op(59, &.{ t_ptr_sb, v_b, 12 });
    try a.op(59, &.{ t_ptr_sa, v_a, 12 });
    try a.op(59, &.{ t_ptr_sc, v_c, 12 });
    try a.op(32, &.{ t_ptr_sb_u32, 12, t_u32 });
    try a.op(32, &.{ t_ptr_sa_u32, 12, t_u32 });
    // C element pointer: f16 (c_h16), f32 (fuse_scale), or s32 (raw).
    try a.op(32, &.{ t_ptr_sc_s32, 12, t_c_elem });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push });
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });

    if (fuse_scale) {
        // binding 3 = scale buffer struct{[N]f32}; per-subgroup s32 scratch wg.
        try a.op(28, &.{ t_arr_f32, t_f32, c_arrlen });
        try a.op(30, &.{ t_scale, t_arr_f32 });
        try a.op(32, &.{ t_ptr_scale, 12, t_scale });
        try a.op(59, &.{ t_ptr_scale, v_scale, 12 });
        try a.op(32, &.{ t_ptr_scale_f32, 12, t_f32 });
        try a.op(43, &.{ t_u32, c_scrlen, NWARPS * 256 });
        try a.op(28, &.{ t_scr, t_s32, c_scrlen });
        try a.op(32, &.{ t_ptr_wg_scr, 4, t_scr });
        try a.op(59, &.{ t_ptr_wg_scr, v_scr, 4 });
        try a.op(32, &.{ t_ptr_wg_s32, 4, t_s32 });
    }

    // workgroup u32 array (no ArrayStride — Workgroup class)
    try a.op(43, &.{ t_u32, c_shlen, SH_LEN });
    try a.op(28, &.{ t_sh, t_u32, c_shlen });
    try a.op(32, &.{ t_ptr_wg_sh, 4, t_sh });
    try a.op(59, &.{ t_ptr_wg_sh, v_sh, 4 });
    try a.op(32, &.{ t_ptr_wg_u32, 4, t_u32 });

    for (0..cn) |i| try a.op(43, &.{ t_u32, cids[i], cvals[i] });
    const c0 = cu(&cvals, &cids, cn, 0);
    const c1 = cu(&cvals, &cids, cn, 1);
    const c2 = cu(&cvals, &cids, cn, 2);
    const c3 = cu(&cvals, &cids, cn, 3);
    const c4 = cu(&cvals, &cids, cn, 4);
    const c16 = cu(&cvals, &cids, cn, 16);
    const c32 = cu(&cvals, &cids, cn, 32);
    const c64 = cu(&cvals, &cids, cn, 64);
    const c128 = cu(&cvals, &cids, cn, 128);
    const c1024 = cu(&cvals, &cids, cn, 1024);
    const c2048 = cu(&cvals, &cids, cn, 2048);
    const c_bsem = cu(&cvals, &cids, cn, 0x108);
    const c_scope = c3; // Subgroup scope == 3 (coop matrices)
    const c_scope_wg = c2; // Workgroup scope == 2 (barrier)

    // A=16x32 (use 0), B=32x16 (use 1), C=16x16 (use 2), all s8/s32.
    try a.op(4456, &.{ t_mat_a, t_s8, c_scope, c16, c32, c0 });
    try a.op(4456, &.{ t_mat_b, t_s8, c_scope, c32, c16, c1 });
    try a.op(4456, &.{ t_mat_c, t_s32, c_scope, c16, c16, c2 });
    try a.op(43, &.{ t_s32, c_s32_0, 0 });
    try a.op(44, &.{ t_mat_c, c_acc0, c_s32_0 });

    // --- function ---
    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    // ids / block labels
    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_c = a.id();
    const tile_r = a.id();
    try a.op(81, &.{ t_u32, tile_c, gidv, 0 });
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    const col0 = a.id();
    const row0 = a.id();
    try a.op(132, &.{ t_u32, col0, tile_c, c128 });
    try a.op(132, &.{ t_u32, row0, tile_r, c128 });

    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const lx = a.id();
    const ly = a.id();
    try a.op(81, &.{ t_u32, lx, lidv, 0 });
    try a.op(81, &.{ t_u32, ly, lidv, 1 });
    const flat = a.id();
    const lymul = a.id();
    try a.op(132, &.{ t_u32, lymul, ly, c32 });
    try a.op(128, &.{ t_u32, flat, lymul, lx });

    // push values
    var push_vals: [4]u32 = undefined;
    inline for (0..4) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c0,
            1 => c1,
            2 => c2,
            else => c3,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_n = push_vals[1];
    const p_k = push_vals[2];
    const p_stride = push_vals[3];

    // warp = ly (0..3): warp_m = ly&1, warp_n = ly>>1.
    const warp_m = a.id();
    try a.op(199, &.{ t_u32, warp_m, ly, c1 });
    const warp_n = a.id();
    try a.op(194, &.{ t_u32, warp_n, ly, c1 });
    // A warp base (u32): warp_m*64 rows * 16 u32 = warp_m*1024.
    const a_warp = a.id();
    try a.op(132, &.{ t_u32, a_warp, warp_m, c1024 });
    // B warp col u32 offset: warp_n*WARP_W cols / 4.
    const b_warp = a.id();
    try a.op(132, &.{ t_u32, b_warp, warp_n, cu(&cvals, &cids, cn, WARP_W / 4) });
    // C warp row base (rows): row0 + warp_m*64 ; C warp col base: col0 + warp_n*WARP_W.
    const wm64 = a.id();
    try a.op(132, &.{ t_u32, wm64, warp_m, c64 });
    const c_row0 = a.id();
    try a.op(128, &.{ t_u32, c_row0, row0, wm64 });
    const wn64 = a.id();
    try a.op(132, &.{ t_u32, wn64, warp_n, cu(&cvals, &cids, cn, WARP_W) });
    const c_col0 = a.id();
    try a.op(128, &.{ t_u32, c_col0, col0, wn64 });

    // Loop-invariant A staging indices: quad q = flat + t*128 (0..2048),
    // A_sh row = q/16, u32-col kq = q%16. global u32 = (row0+row)*(k/4) + k0/4 + kq
    // = ((row0+row)*k + k0)/4 + kq. Precompute arowbase_t = (row0+row)*k (s8) and
    // kq_t, and sh index = q.
    var a_shidx: [16]u32 = undefined;
    var a_rowk: [16]u32 = undefined; // (row0+row)*k
    var a_kq: [16]u32 = undefined;
    for (0..A_QUADS) |t| {
        var q = flat;
        if (t > 0) {
            const qn = a.id();
            try a.op(128, &.{ t_u32, qn, flat, cu(&cvals, &cids, cn, @intCast(t * THREADS)) });
            q = qn;
        }
        a_shidx[t] = q;
        const row = a.id();
        try a.op(194, &.{ t_u32, row, q, c4 }); // q/16
        a_kq[t] = a.id();
        try a.op(199, &.{ t_u32, a_kq[t], q, cu(&cvals, &cids, cn, 15) });
        const grow = a.id();
        try a.op(128, &.{ t_u32, grow, row0, row });
        a_rowk[t] = a.id();
        try a.op(132, &.{ t_u32, a_rowk[t], grow, p_k });
    }
    // B staging: quad q, B_sh krow = q/32, u32-col cq = q%32.
    // global u32 = ((k0+krow)*stride + col0)/4 + cq. Precompute krow_t, cq_t.
    var b_shidx: [16]u32 = undefined;
    var b_krow: [16]u32 = undefined;
    var b_cq: [16]u32 = undefined;
    for (0..B_QUADS) |t| {
        var q = flat;
        if (t > 0) {
            const qn = a.id();
            try a.op(128, &.{ t_u32, qn, flat, cu(&cvals, &cids, cn, @intCast(t * THREADS)) });
            q = qn;
        }
        b_shidx[t] = q;
        b_krow[t] = a.id();
        try a.op(194, &.{ t_u32, b_krow[t], q, cu(&cvals, &cids, cn, 5) }); // q/32
        b_cq[t] = a.id();
        try a.op(199, &.{ t_u32, b_cq[t], q, cu(&cvals, &cids, cn, 31) }); // q%32
    }

    // Staging helpers: load(kb) reads the k-slab from global into registers;
    // store(regs) writes them into shared (k0-invariant offsets). Splitting them
    // lets the double-buffered loop issue the next slab's global loads before the
    // current MMA section, hiding load latency under tensor-core work.
    const DB = struct {
        aq: usize,
        bq: usize,
        a_rowk: [16]u32,
        a_kq: [16]u32,
        a_shidx: [16]u32,
        b_krow: [16]u32,
        b_cq: [16]u32,
        b_shidx: [16]u32,
        v_a: u32,
        v_b: u32,
        v_sh: u32,
        t_psa: u32,
        t_psb: u32,
        t_pwg: u32,
        t_u32: u32,
        c0: u32,
        c2: u32,
        c2048: u32,
        p_stride: u32,
        col0: u32,
        fn load(st: @This(), asm_: *Asm, kb: u32) ![32]u32 {
            var regs: [32]u32 = undefined;
            for (0..st.aq) |t| {
                const gs8 = asm_.id();
                try asm_.op(128, &.{ st.t_u32, gs8, st.a_rowk[t], kb });
                const gu = asm_.id();
                try asm_.op(194, &.{ st.t_u32, gu, gs8, st.c2 });
                const gidx = asm_.id();
                try asm_.op(128, &.{ st.t_u32, gidx, gu, st.a_kq[t] });
                const gptr = asm_.id();
                try asm_.op(65, &.{ st.t_psa, gptr, st.v_a, st.c0, gidx });
                regs[t] = asm_.id();
                try asm_.op(61, &.{ st.t_u32, regs[t], gptr });
            }
            for (0..st.bq) |t| {
                const kk = asm_.id();
                try asm_.op(128, &.{ st.t_u32, kk, kb, st.b_krow[t] });
                const kmul = asm_.id();
                try asm_.op(132, &.{ st.t_u32, kmul, kk, st.p_stride });
                const gs8 = asm_.id();
                try asm_.op(128, &.{ st.t_u32, gs8, kmul, st.col0 });
                const gu = asm_.id();
                try asm_.op(194, &.{ st.t_u32, gu, gs8, st.c2 });
                const gidx = asm_.id();
                try asm_.op(128, &.{ st.t_u32, gidx, gu, st.b_cq[t] });
                const gptr = asm_.id();
                try asm_.op(65, &.{ st.t_psb, gptr, st.v_b, st.c0, gidx });
                regs[16 + t] = asm_.id();
                try asm_.op(61, &.{ st.t_u32, regs[16 + t], gptr });
            }
            return regs;
        }
        fn store(st: @This(), asm_: *Asm, regs: [32]u32) !void {
            for (0..st.aq) |t| {
                const sptr = asm_.id();
                try asm_.op(65, &.{ st.t_pwg, sptr, st.v_sh, st.a_shidx[t] });
                try asm_.op(62, &.{ sptr, regs[t] });
            }
            for (0..st.bq) |t| {
                const sidx = asm_.id();
                try asm_.op(128, &.{ st.t_u32, sidx, st.b_shidx[t], st.c2048 });
                const sptr = asm_.id();
                try asm_.op(65, &.{ st.t_pwg, sptr, st.v_sh, sidx });
                try asm_.op(62, &.{ sptr, regs[16 + t] });
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
        .v_a = v_a,
        .v_b = v_b,
        .v_sh = v_sh,
        .t_psa = t_ptr_sa_u32,
        .t_psb = t_ptr_sb_u32,
        .t_pwg = t_ptr_wg_u32,
        .t_u32 = t_u32,
        .c0 = c0,
        .c2 = c2,
        .c2048 = c2048,
        .p_stride = p_stride,
        .col0 = col0,
    };

    // Loop skeleton with phi-carried k0 and 16 accumulators.
    const lb_head = a.id();
    const lb_cond = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();
    const k0n = a.id();
    var acc_next: [4][4]u32 = undefined; // [MT][NT] used
    for (0..MT) |mi| for (0..NT) |nj| {
        acc_next[mi][nj] = a.id();
    };

    // Double-buffer prologue: preload slab 0 into shared before the loop.
    if (double_buf) {
        const regs0 = try db.load(&a, c0);
        try db.store(&a, regs0);
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
    }

    try a.op(249, &.{lb_head});
    try a.op(248, &.{lb_head});
    const k0v = a.id();
    try a.op(245, &.{ t_u32, k0v, c0, lb_entry, k0n, lb_cont });
    var acc_phi: [4][4]u32 = undefined;
    for (0..MT) |mi| for (0..NT) |nj| {
        acc_phi[mi][nj] = a.id();
        try a.op(245, &.{ t_mat_c, acc_phi[mi][nj], c_acc0, lb_entry, acc_next[mi][nj], lb_cont });
    };
    try a.op(246, &.{ lb_merge, lb_cont, 0 });
    try a.op(249, &.{lb_cond});
    try a.op(248, &.{lb_cond});
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, k0v, p_k });
    try a.op(250, &.{ cmp, lb_body, lb_merge });

    try a.op(248, &.{lb_body});
    var regs_next: [32]u32 = undefined;
    if (!double_buf) {
        // Single-buffered: barrier / stage / barrier / MMA.
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
        const regs = try db.load(&a, k0v);
        try db.store(&a, regs);
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
    } else {
        // Double-buffered: issue next slab's global loads (clamped so the last
        // iteration doesn't read OOB — its result is stored but never MMA'd).
        const kb = a.id();
        try a.op(128, &.{ t_u32, kb, k0v, c64 });
        const ok = a.id();
        try a.op(176, &.{ t_bool, ok, kb, p_k });
        const kbc = a.id();
        try a.op(169, &.{ t_u32, kbc, ok, kb, c0 }); // OpSelect
        regs_next = try db.load(&a, kbc);
    }

    // MMA: 2 ks of 32 k each, reading the current shared slab.
    var acc_cur = acc_phi;
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..MT) |mi| {
            const off1 = a.id();
            try a.op(128, &.{ t_u32, off1, a_warp, cu(&cvals, &cids, cn, @intCast(mi * 256)) });
            const off = a.id();
            try a.op(128, &.{ t_u32, off, off1, cu(&cvals, &cids, cn, @intCast(ks * 8)) });
            const aptr = a.id();
            try a.op(65, &.{ t_ptr_wg_u32, aptr, v_sh, off });
            ma[mi] = a.id();
            try a.op(4457, &.{ t_mat_a, ma[mi], aptr, c0, c16 });
        }
        for (0..NT) |nj| {
            const off1 = a.id();
            try a.op(128, &.{ t_u32, off1, c2048, cu(&cvals, &cids, cn, @intCast(ks * 1024)) });
            const off2 = a.id();
            try a.op(128, &.{ t_u32, off2, off1, b_warp });
            const off = a.id();
            try a.op(128, &.{ t_u32, off, off2, cu(&cvals, &cids, cn, @intCast(nj * 4)) });
            const bptr = a.id();
            try a.op(65, &.{ t_ptr_wg_u32, bptr, v_sh, off });
            const mb = a.id();
            try a.op(4457, &.{ t_mat_b, mb, bptr, c0, c32 });
            for (0..MT) |mi| {
                const acc_out = if (ks == 1) acc_next[mi][nj] else a.id();
                try a.op(4459, &.{ t_mat_c, acc_out, ma[mi], mb, acc_cur[mi][nj], 0xF });
                acc_cur[mi][nj] = acc_out;
            }
        }
    }

    if (double_buf) {
        // Barrier: current-slab reads done; store next slab; barrier: visible.
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
        try db.store(&a, regs_next);
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
    }
    try a.op(249, &.{lb_cont});

    try a.op(248, &.{lb_cont});
    try a.op(128, &.{ t_u32, k0n, k0v, c64 });
    try a.op(249, &.{lb_head});

    // merge: store MTxNT C tiles at (c_row0 + mi*16, c_col0 + nj*16), stride p_n.
    try a.op(248, &.{lb_merge});
    if (!fuse_scale) {
        for (0..MT) |mi| {
            const rowmi = a.id();
            try a.op(128, &.{ t_u32, rowmi, c_row0, cu(&cvals, &cids, cn, @intCast(mi * 16)) });
            const rowmul = a.id();
            try a.op(132, &.{ t_u32, rowmul, rowmi, p_n });
            for (0..NT) |nj| {
                const ccol = a.id();
                try a.op(128, &.{ t_u32, ccol, c_col0, cu(&cvals, &cids, cn, @intCast(nj * 16)) });
                const cbase = a.id();
                try a.op(128, &.{ t_u32, cbase, rowmul, ccol });
                const cptr = a.id();
                try a.op(65, &.{ t_ptr_sc_s32, cptr, v_c, c0, cbase });
                try a.op(4458, &.{ cptr, acc_phi[mi][nj], c0, p_n });
            }
        }
    } else {
        // Stage A: rescale + f32 store, per element. Each 16x16 s32 fragment
        // coop-stores into this subgroup's 256-s32 scratch region, then the 32
        // lanes read it back and write y = f32(acc) * act_scale[row] *
        // weight_scale[col]. scale buffer (binding 3) = [act(m_pad) | weight(rows)].
        const p_m = push_vals[0];
        const c15 = cu(&cvals, &cids, cn, 15);
        const c256 = cu(&cvals, &cids, cn, 256);
        const scr_base = a.id();
        try a.op(132, &.{ t_u32, scr_base, ly, c256 }); // ly*256
        for (0..MT) |mi| {
            const rowmi = a.id();
            try a.op(128, &.{ t_u32, rowmi, c_row0, cu(&cvals, &cids, cn, @intCast(mi * 16)) });
            const rowmul = a.id();
            try a.op(132, &.{ t_u32, rowmul, rowmi, p_n });
            for (0..NT) |nj| {
                const ccol = a.id();
                try a.op(128, &.{ t_u32, ccol, c_col0, cu(&cvals, &cids, cn, @intCast(nj * 16)) });
                // coop-store the s32 fragment into scratch (RowMajor, stride 16).
                const scrptr = a.id();
                try a.op(65, &.{ t_ptr_wg_s32, scrptr, v_scr, scr_base });
                try a.op(4458, &.{ scrptr, acc_phi[mi][nj], c0, c16 });
                try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
                for (0..8) |i| {
                    var e = lx;
                    if (i > 0) {
                        const en = a.id();
                        try a.op(128, &.{ t_u32, en, lx, cu(&cvals, &cids, cn, @intCast(i * 32)) });
                        e = en;
                    }
                    const lrow = a.id();
                    try a.op(194, &.{ t_u32, lrow, e, c4 }); // e/16
                    const lcol = a.id();
                    try a.op(199, &.{ t_u32, lcol, e, c15 }); // e%16
                    const grow = a.id();
                    try a.op(128, &.{ t_u32, grow, rowmi, lrow });
                    const gcol = a.id();
                    try a.op(128, &.{ t_u32, gcol, ccol, lcol });
                    // s32 scratch load + convert to f32.
                    const sidx = a.id();
                    try a.op(128, &.{ t_u32, sidx, scr_base, e });
                    const sptr = a.id();
                    try a.op(65, &.{ t_ptr_wg_s32, sptr, v_scr, sidx });
                    const s32v = a.id();
                    try a.op(61, &.{ t_s32, s32v, sptr });
                    const fv = a.id();
                    try a.op(111, &.{ t_f32, fv, s32v }); // OpConvertSToF
                    // act_scale[grow], weight_scale[m_pad + gcol].
                    const aptr = a.id();
                    try a.op(65, &.{ t_ptr_scale_f32, aptr, v_scale, c0, grow });
                    const asv = a.id();
                    try a.op(61, &.{ t_f32, asv, aptr });
                    const widx = a.id();
                    try a.op(128, &.{ t_u32, widx, p_m, gcol });
                    const wptr = a.id();
                    try a.op(65, &.{ t_ptr_scale_f32, wptr, v_scale, c0, widx });
                    const wsv = a.id();
                    try a.op(61, &.{ t_f32, wsv, wptr });
                    const prod = a.id();
                    try a.op(133, &.{ t_f32, prod, asv, wsv }); // OpFMul
                    const yv = a.id();
                    try a.op(133, &.{ t_f32, yv, fv, prod });
                    // y[grow*p_n + gcol] = yv.
                    const lrowmul = a.id();
                    try a.op(132, &.{ t_u32, lrowmul, lrow, p_n });
                    const yb = a.id();
                    try a.op(128, &.{ t_u32, yb, rowmul, lrowmul });
                    const yidx = a.id();
                    try a.op(128, &.{ t_u32, yidx, yb, gcol });
                    const yptr = a.id();
                    try a.op(65, &.{ t_ptr_sc_s32, yptr, v_c, c0, yidx });
                    if (c_h16) {
                        const yh = a.id();
                        try a.op(115, &.{ t_f16, yh, yv }); // OpFConvert f32->f16
                        try a.op(62, &.{ yptr, yh });
                    } else {
                        try a.op(62, &.{ yptr, yv });
                    }
                }
                try a.op(224, &.{ c_scope_wg, c_scope_wg, c_bsem });
            }
        }
    }
    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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

    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    var cvals: [80]u32 = undefined;
    var cids: [80]u32 = undefined;
    var cn: usize = 0;
    const addC = struct {
        fn f(vals: []u32, ids: []u32, n: *usize, asm_: *Asm, v: u32) void {
            for (vals[0..n.*]) |ex| if (ex == v) return;
            vals[n.*] = v;
            ids[n.*] = asm_.id();
            n.* += 1;
        }
    }.f;
    for ([_]u32{ 0, 1, 2, 3, 4, 8, 12, 15, 16, 24, 32, 48, 63, 64, 128, 192, 255, 256, 257, 0xFF, 0x108, NG, ROT, INV, WG, LOADS, WORDS, cols, cols / 4 }) |v|
        addC(&cvals, &cids, &cn, &a, v);
    const cu = struct {
        fn f(vals: []const u32, ids: []const u32, n: usize, v: u32) u32 {
            for (0..n) |i| if (vals[i] == v) return ids[i];
            unreachable;
        }
    }.f;

    const main_fn = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const ext = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_s32 = a.id();
    const t_f32 = a.id();
    const t_f16 = a.id();
    const t_bool = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f32 = a.id();
    const t_arr_u32 = a.id();
    const t_sx = a.id();
    const t_sq = a.id();
    const t_ss = a.id();
    const t_ptr_sx = a.id();
    const t_ptr_sq = a.id();
    const t_ptr_ss = a.id();
    const v_x = a.id();
    const v_q = a.id();
    const v_s = a.id();
    const t_ptr_sx_f32 = a.id();
    const t_ptr_sq_u32 = a.id();
    const t_ptr_ss_f32 = a.id();
    const c_shlen = a.id();
    const t_sh = a.id();
    const t_ptr_wg_sh = a.id();
    const v_sh = a.id();
    const t_ptr_wg_f16 = a.id();
    const cf16_inv16 = a.id();
    const cf16_0 = a.id();
    const cf_inv127 = a.id();
    const cf_tiny = a.id();
    const cf_half = a.id();
    const cf_1 = a.id();
    const cs_127 = a.id();
    const cs_m127 = a.id();

    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1});
    try a.op(17, &.{9}); // Float16
    try a.op(17, &.{5345});
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(11, &.{ext}, "GLSL.std.450");
    try a.op(14, &.{ 0, 3 });
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_x, v_q, v_s, v_sh });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, WG, 1, 1 });

    try a.op(71, &.{ gid_var, 11, 26 });
    try a.op(71, &.{ lid_var, 11, 27 });
    try a.op(71, &.{ t_arr_f32, 6, 4 });
    try a.op(71, &.{ t_arr_u32, 6, 4 });
    inline for (.{ t_sx, t_sq, t_ss }) |t| {
        try a.op(71, &.{ t, 2 });
        try a.op(72, &.{ t, 0, 35, 0 });
    }
    inline for (.{ v_x, v_q, v_s }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 });
        try a.op(71, &.{ v, 33, @intCast(binding) });
    }

    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(21, &.{ t_s32, 32, 1 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(20, &.{t_bool});
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f32, t_f32, c_arrlen });
    try a.op(28, &.{ t_arr_u32, t_u32, c_arrlen });
    try a.op(30, &.{ t_sx, t_arr_f32 });
    try a.op(30, &.{ t_sq, t_arr_u32 });
    try a.op(30, &.{ t_ss, t_arr_f32 });
    try a.op(32, &.{ t_ptr_sx, 12, t_sx });
    try a.op(32, &.{ t_ptr_sq, 12, t_sq });
    try a.op(32, &.{ t_ptr_ss, 12, t_ss });
    try a.op(59, &.{ t_ptr_sx, v_x, 12 });
    try a.op(59, &.{ t_ptr_sq, v_q, 12 });
    try a.op(59, &.{ t_ptr_ss, v_s, 12 });
    try a.op(32, &.{ t_ptr_sx_f32, 12, t_f32 });
    try a.op(32, &.{ t_ptr_sq_u32, 12, t_u32 });
    try a.op(32, &.{ t_ptr_ss_f32, 12, t_f32 });

    try a.op(43, &.{ t_u32, c_shlen, SH_LEN });
    try a.op(28, &.{ t_sh, t_f16, c_shlen });
    try a.op(32, &.{ t_ptr_wg_sh, 4, t_sh });
    try a.op(59, &.{ t_ptr_wg_sh, v_sh, 4 });
    try a.op(32, &.{ t_ptr_wg_f16, 4, t_f16 });

    for (0..cn) |i| try a.op(43, &.{ t_u32, cids[i], cvals[i] });
    try a.op(43, &.{ t_f16, cf16_inv16, @as(u32, @as(u16, @bitCast(@as(f16, 1.0 / 16.0)))) });
    try a.op(43, &.{ t_f16, cf16_0, 0 });
    try a.op(43, &.{ t_f32, cf_inv127, @bitCast(@as(f32, 1.0 / 127.0)) });
    try a.op(43, &.{ t_f32, cf_tiny, @bitCast(@as(f32, 1e-12)) });
    try a.op(43, &.{ t_f32, cf_half, @bitCast(@as(f32, 0.5)) });
    try a.op(43, &.{ t_f32, cf_1, @bitCast(@as(f32, 1.0)) });
    try a.op(43, &.{ t_s32, cs_127, 127 });
    try a.op(43, &.{ t_s32, cs_m127, @bitCast(@as(i32, -127)) });

    const c0 = cu(&cvals, &cids, cn, 0);
    const c1 = cu(&cvals, &cids, cn, 1);
    const c2 = cu(&cvals, &cids, cn, 2);
    const c8 = cu(&cvals, &cids, cn, 8);
    const c255 = cu(&cvals, &cids, cn, 255);
    const c256 = cu(&cvals, &cids, cn, 256);
    const c257 = cu(&cvals, &cids, cn, 257);
    const cff = cu(&cvals, &cids, cn, 0xFF);
    const c_bsem = cu(&cvals, &cids, cn, 0x108);
    const c_wg = cu(&cvals, &cids, cn, WG);
    const c_ng = cu(&cvals, &cids, cn, NG);
    const c_rot = cu(&cvals, &cids, cn, ROT);
    const c_inv = cu(&cvals, &cids, cn, INV);
    const c64 = cu(&cvals, &cids, cn, 64);

    const openLoop = struct {
        fn f(asm_: *Asm, tu: u32, tb: u32, cz: u32, count: u32, pred: u32) !struct { head: u32, cont: u32, merge: u32, iv: u32, inx: u32 } {
            const head = asm_.id();
            const cont = asm_.id();
            const merge = asm_.id();
            const iv = asm_.id();
            const inx = asm_.id();
            const cond = asm_.id();
            const body = asm_.id();
            try asm_.op(249, &.{head});
            try asm_.op(248, &.{head});
            try asm_.op(245, &.{ tu, iv, cz, pred, inx, cont });
            try asm_.op(246, &.{ merge, cont, 0 });
            try asm_.op(249, &.{cond});
            try asm_.op(248, &.{cond});
            const cmp = asm_.id();
            try asm_.op(176, &.{ tb, cmp, iv, count });
            try asm_.op(250, &.{ cmp, body, merge });
            try asm_.op(248, &.{body});
            return .{ .head = head, .cont = cont, .merge = merge, .iv = iv, .inx = inx };
        }
    }.f;
    const closeLoop = struct {
        fn f(asm_: *Asm, tu: u32, one: u32, L: anytype) !void {
            try asm_.op(249, &.{L.cont});
            try asm_.op(248, &.{L.cont});
            try asm_.op(128, &.{ tu, L.inx, L.iv, one });
            try asm_.op(249, &.{L.head});
            try asm_.op(248, &.{L.merge});
        }
    }.f;

    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const row = a.id();
    try a.op(81, &.{ t_u32, row, gidv, 0 });
    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const tid = a.id();
    try a.op(81, &.{ t_u32, tid, lidv, 0 });
    const rowcols = a.id();
    try a.op(132, &.{ t_u32, rowcols, row, cu(&cvals, &cids, cn, cols) });
    var cur = lb_entry;

    // Phase 0: coalesced load f32 -> f16 into padded shared.
    {
        const L = try openLoop(&a, t_u32, t_bool, c0, cu(&cvals, &cids, cn, LOADS), cur);
        const iw = a.id();
        try a.op(132, &.{ t_u32, iw, L.iv, c_wg });
        const j = a.id();
        try a.op(128, &.{ t_u32, j, tid, iw });
        const g = a.id();
        try a.op(194, &.{ t_u32, g, j, c8 });
        const l = a.id();
        try a.op(199, &.{ t_u32, l, j, c255 });
        const gp = a.id();
        try a.op(132, &.{ t_u32, gp, g, c257 });
        const shi = a.id();
        try a.op(128, &.{ t_u32, shi, gp, l });
        const gi = a.id();
        try a.op(128, &.{ t_u32, gi, rowcols, j });
        const xp = a.id();
        try a.op(65, &.{ t_ptr_sx_f32, xp, v_x, c0, gi });
        const valf = a.id();
        try a.op(61, &.{ t_f32, valf, xp });
        const valh = a.id();
        try a.op(115, &.{ t_f16, valh, valf }); // FConvert f32->f16
        const sp = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, sp, v_sh, shi });
        try a.op(62, &.{ sp, valh });
        try closeLoop(&a, t_u32, c1, L);
        cur = L.merge;
    }
    try a.op(224, &.{ c2, c2, c_bsem });

    // Phase 1: FWHT (f16) + normalize (tid < NG).
    {
        const sel = a.id();
        const do = a.id();
        const cmp = a.id();
        try a.op(176, &.{ t_bool, cmp, tid, c_ng });
        try a.op(247, &.{ sel, 0 });
        try a.op(250, &.{ cmp, do, sel });
        try a.op(248, &.{do});
        var cur1 = do;
        const t257 = a.id();
        try a.op(132, &.{ t_u32, t257, tid, c257 });
        inline for (.{ 1, 4, 16, 64 }) |s| {
            const css = cu(&cvals, &cids, cn, s);
            const cs2 = cu(&cvals, &cids, cn, 2 * s);
            const cs3 = cu(&cvals, &cids, cn, 3 * s);
            const L = try openLoop(&a, t_u32, t_bool, c0, c64, cur1);
            const lo = if (s == 1) c0 else blk: {
                const x = a.id();
                try a.op(199, &.{ t_u32, x, L.iv, cu(&cvals, &cids, cn, s - 1) });
                break :blk x;
            };
            const hi = if (s == 1) L.iv else blk: {
                const x = a.id();
                try a.op(130, &.{ t_u32, x, L.iv, lo });
                break :blk x;
            };
            const hi4 = a.id();
            try a.op(196, &.{ t_u32, hi4, hi, c2 });
            const p0 = if (s == 1) hi4 else blk: {
                const x = a.id();
                try a.op(128, &.{ t_u32, x, hi4, lo });
                break :blk x;
            };
            const base = a.id();
            try a.op(128, &.{ t_u32, base, t257, p0 });
            var e: [4]u32 = undefined;
            inline for (.{ c0, css, cs2, cs3 }, 0..) |off, k| {
                const idx = if (k == 0) base else blk: {
                    const x = a.id();
                    try a.op(128, &.{ t_u32, x, base, off });
                    break :blk x;
                };
                const pp = a.id();
                try a.op(65, &.{ t_ptr_wg_f16, pp, v_sh, idx });
                e[k] = a.id();
                try a.op(61, &.{ t_f16, e[k], pp });
            }
            const pq = a.id();
            try a.op(129, &.{ t_f16, pq, e[0], e[1] });
            const qd = a.id();
            try a.op(131, &.{ t_f16, qd, e[2], e[3] });
            const rd = a.id();
            try a.op(131, &.{ t_f16, rd, e[0], e[1] });
            const sd = a.id();
            try a.op(129, &.{ t_f16, sd, e[2], e[3] });
            var nn: [4]u32 = undefined;
            nn[0] = a.id();
            try a.op(129, &.{ t_f16, nn[0], pq, qd });
            nn[1] = a.id();
            try a.op(131, &.{ t_f16, nn[1], pq, qd });
            nn[2] = a.id();
            try a.op(129, &.{ t_f16, nn[2], rd, sd });
            nn[3] = a.id();
            try a.op(131, &.{ t_f16, nn[3], sd, rd });
            inline for (.{ c0, css, cs2, cs3 }, 0..) |off, k| {
                const idx = if (k == 0) base else blk: {
                    const x = a.id();
                    try a.op(128, &.{ t_u32, x, base, off });
                    break :blk x;
                };
                const pp = a.id();
                try a.op(65, &.{ t_ptr_wg_f16, pp, v_sh, idx });
                try a.op(62, &.{ pp, nn[k] });
            }
            try closeLoop(&a, t_u32, c1, L);
            cur1 = L.merge;
        }
        // normalize /16 + abs-max (2-phi loop, f16)
        const nh = a.id();
        const ncond = a.id();
        const nbd = a.id();
        const nct = a.id();
        const nmg = a.id();
        const niv = a.id();
        const ninx = a.id();
        const namax = a.id();
        const namaxn = a.id();
        try a.op(249, &.{nh});
        try a.op(248, &.{nh});
        try a.op(245, &.{ t_u32, niv, c0, cur1, ninx, nct });
        try a.op(245, &.{ t_f16, namax, cf16_0, cur1, namaxn, nct });
        try a.op(246, &.{ nmg, nct, 0 });
        try a.op(249, &.{ncond});
        try a.op(248, &.{ncond});
        const ncmp = a.id();
        try a.op(176, &.{ t_bool, ncmp, niv, c256 });
        try a.op(250, &.{ ncmp, nbd, nmg });
        try a.op(248, &.{nbd});
        const nidx = a.id();
        try a.op(128, &.{ t_u32, nidx, t257, niv });
        const npp = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, npp, v_sh, nidx });
        const raw = a.id();
        try a.op(61, &.{ t_f16, raw, npp });
        const v16 = a.id();
        try a.op(133, &.{ t_f16, v16, raw, cf16_inv16 });
        try a.op(62, &.{ npp, v16 });
        const av = a.id();
        try a.op(12, &.{ t_f16, av, ext, 4, v16 });
        try a.op(12, &.{ t_f16, namaxn, ext, 40, namax, av });
        try a.op(249, &.{nct});
        try a.op(248, &.{nct});
        try a.op(128, &.{ t_u32, ninx, niv, c1 });
        try a.op(249, &.{nh});
        try a.op(248, &.{nmg});
        const gidx = a.id();
        try a.op(128, &.{ t_u32, gidx, c_rot, tid });
        const gpp = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, gpp, v_sh, gidx });
        try a.op(62, &.{ gpp, namax });
        try a.op(249, &.{sel});
        try a.op(248, &.{sel});
        cur = sel;
    }
    try a.op(224, &.{ c2, c2, c_bsem });

    // Phase 2: reduce over gmax[0..NG] -> scale[row], inv (tid == 0).
    {
        const sel = a.id();
        const do = a.id();
        const cmp = a.id();
        try a.op(170, &.{ t_bool, cmp, tid, c0 });
        try a.op(247, &.{ sel, 0 });
        try a.op(250, &.{ cmp, do, sel });
        try a.op(248, &.{do});
        const mh = a.id();
        const mcond = a.id();
        const mbd = a.id();
        const mct = a.id();
        const mmg = a.id();
        const miv = a.id();
        const minx = a.id();
        const mx = a.id();
        const mxn = a.id();
        try a.op(249, &.{mh});
        try a.op(248, &.{mh});
        try a.op(245, &.{ t_u32, miv, c0, do, minx, mct });
        try a.op(245, &.{ t_f16, mx, cf16_0, do, mxn, mct });
        try a.op(246, &.{ mmg, mct, 0 });
        try a.op(249, &.{mcond});
        try a.op(248, &.{mcond});
        const mcmp = a.id();
        try a.op(176, &.{ t_bool, mcmp, miv, c_ng });
        try a.op(250, &.{ mcmp, mbd, mmg });
        try a.op(248, &.{mbd});
        const gidx = a.id();
        try a.op(128, &.{ t_u32, gidx, c_rot, miv });
        const gpp = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, gpp, v_sh, gidx });
        const gv = a.id();
        try a.op(61, &.{ t_f16, gv, gpp });
        try a.op(12, &.{ t_f16, mxn, ext, 40, mx, gv });
        try a.op(249, &.{mct});
        try a.op(248, &.{mct});
        try a.op(128, &.{ t_u32, minx, miv, c1 });
        try a.op(249, &.{mh});
        try a.op(248, &.{mmg});
        const mxf = a.id();
        try a.op(115, &.{ t_f32, mxf, mx }); // f16->f32
        const sc0 = a.id();
        try a.op(133, &.{ t_f32, sc0, mxf, cf_inv127 });
        const sc = a.id();
        try a.op(12, &.{ t_f32, sc, ext, 40, sc0, cf_tiny });
        const scp = a.id();
        try a.op(65, &.{ t_ptr_ss_f32, scp, v_s, c0, row });
        try a.op(62, &.{ scp, sc });
        const invf = a.id();
        try a.op(136, &.{ t_f32, invf, cf_1, sc });
        const invh = a.id();
        try a.op(115, &.{ t_f16, invh, invf }); // f32->f16
        const ip = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, ip, v_sh, c_inv });
        try a.op(62, &.{ ip, invh });
        try a.op(249, &.{sel});
        try a.op(248, &.{sel});
        cur = sel;
    }
    try a.op(224, &.{ c2, c2, c_bsem });

    // Phase 3: coalesced quantize (f16 shared -> f32 math -> packed int8).
    {
        const ip = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, ip, v_sh, c_inv });
        const invh = a.id();
        try a.op(61, &.{ t_f16, invh, ip });
        const inv = a.id();
        try a.op(115, &.{ t_f32, inv, invh }); // f16->f32
        const roww = a.id();
        try a.op(132, &.{ t_u32, roww, row, cu(&cvals, &cids, cn, cols / 4) });
        const L = try openLoop(&a, t_u32, t_bool, c0, cu(&cvals, &cids, cn, WORDS), cur);
        const iw = a.id();
        try a.op(132, &.{ t_u32, iw, L.iv, c_wg });
        const word = a.id();
        try a.op(128, &.{ t_u32, word, tid, iw });
        const be = a.id();
        try a.op(196, &.{ t_u32, be, word, c2 });
        const g = a.id();
        try a.op(194, &.{ t_u32, g, be, c8 });
        const l = a.id();
        try a.op(199, &.{ t_u32, l, be, c255 });
        const gp = a.id();
        try a.op(132, &.{ t_u32, gp, g, c257 });
        const base = a.id();
        try a.op(128, &.{ t_u32, base, gp, l });
        var out = c0;
        inline for (0..4) |k| {
            const idx = if (k == 0) base else blk: {
                const x = a.id();
                try a.op(128, &.{ t_u32, x, base, cu(&cvals, &cids, cn, @intCast(k)) });
                break :blk x;
            };
            const pp = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, pp, v_sh, idx });
            const vh = a.id();
            try a.op(61, &.{ t_f16, vh, pp });
            const v = a.id();
            try a.op(115, &.{ t_f32, v, vh }); // f16->f32
            const r = a.id();
            try a.op(133, &.{ t_f32, r, v, inv });
            const sgn = a.id();
            try a.op(12, &.{ t_f32, sgn, ext, 6, r });
            const hh = a.id();
            try a.op(133, &.{ t_f32, hh, sgn, cf_half });
            const added = a.id();
            try a.op(129, &.{ t_f32, added, r, hh });
            const qi = a.id();
            try a.op(110, &.{ t_s32, qi, added });
            const qc = a.id();
            try a.op(12, &.{ t_s32, qc, ext, 45, qi, cs_m127, cs_127 });
            const qub = a.id();
            try a.op(124, &.{ t_u32, qub, qc });
            const qb = a.id();
            try a.op(199, &.{ t_u32, qb, qub, cff });
            const shk = a.id();
            try a.op(196, &.{ t_u32, shk, qb, cu(&cvals, &cids, cn, @intCast(8 * k)) });
            const no = a.id();
            try a.op(197, &.{ t_u32, no, out, shk });
            out = no;
        }
        const wi = a.id();
        try a.op(128, &.{ t_u32, wi, roww, word });
        const qp = a.id();
        try a.op(65, &.{ t_ptr_sq_u32, qp, v_q, c0, wi });
        try a.op(62, &.{ qp, out });
        try closeLoop(&a, t_u32, c1, L);
        cur = L.merge;
    }

    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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
pub fn buildGemmShared(gpa: std.mem.Allocator, b_f16: bool, warps8: bool, acc_h16: bool, c_h16: bool) ![]align(4) u8 {
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

    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    const main_fn = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_f16 = a.id();
    const t_f32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f32 = a.id();
    const t_sc = a.id(); // struct { [N]f32 }
    const t_ptr_sc = a.id();
    const v_c = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_sc_f32 = a.id();
    const t_bool = a.id();
    // uvec4 views: B is the fp8 weight buffer (binding 3), A the f16
    // activation buffer (binding 0) — both for 128-bit staging loads.
    const t_v4u32 = a.id();
    const t_arr_v4 = a.id();
    const t_sb4 = a.id();
    const t_ptr_sb4 = a.id();
    const v_b4 = a.id();
    const t_ptr_sb4_v4 = a.id();
    const t_sa4 = a.id();
    const t_ptr_sa4 = a.id();
    const v_a4 = a.id();
    // workgroup slab: B two 32x128 f16 sub-slabs, then A two 128x32
    const c_bsh_len = a.id();
    const t_bsh = a.id();
    const t_ptr_wg_bsh = a.id();
    const v_bsh = a.id();
    const t_ptr_wg_f16 = a.id();

    const c_u0 = a.id();
    const c_u1 = a.id();
    const c_u2 = a.id();
    const c_u3 = a.id();
    const c_u4 = a.id();
    const c_u5 = a.id();
    const c_u6 = a.id();
    const c_u7 = a.id();
    const c_u8 = a.id();
    const c_u12 = a.id();
    const c_u31 = a.id();
    const c_u16 = a.id();
    const c_u32c = a.id();
    const c_u64 = a.id();
    const c_u128 = a.id();
    const c_u4096 = a.id(); // parity offset between the two sub-slabs
    const c_mag_mask = a.id(); // 0x007F007F: e4m3 magnitude bits, bytes 0/2
    const c_sgn_mask = a.id(); // 0x00800080: e4m3 sign bits, bytes 0/2
    const c_u264 = a.id(); // barrier semantics: AcquireRelease|WorkgroupMemory
    const c_scope_sub = a.id();
    const c_scope_wg = a.id();
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const t_mat_c32 = a.id(); // f32 store type when acc_h16
    const t_v2f16 = a.id();
    const c_f32_0 = a.id();
    const c_f16_0 = a.id(); // acc_h16 accumulator init
    const c_h256 = a.id(); // f16 256.0: exact e4m3 -> f16 exponent rebias
    const c_v2_256 = a.id();
    const c_acc0 = a.id();

    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1});
    try a.op(17, &.{9});
    try a.op(17, &.{5345});
    try a.op(17, &.{6022});
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_16bit_storage");
    try a.op(14, &.{ 0, 3 });
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_c, v_b4, v_a4, v_push, v_bsh });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, NWARPS, 1 }); // LocalSize 32 N 1

    try a.op(71, &.{ gid_var, 11, 26 }); // WorkgroupId
    try a.op(71, &.{ lid_var, 11, 27 }); // LocalInvocationId
    try a.op(71, &.{ t_arr_f32, 6, if (c_h16) 2 else 4 });
    try a.op(71, &.{ t_sc, 2 });
    try a.op(72, &.{ t_sc, 0, 35, 0 });
    try a.op(71, &.{ t_push, 2 });
    inline for (0..4) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    try a.op(71, &.{ v_c, 34, 0 });
    try a.op(71, &.{ v_c, 33, 2 });
    try a.op(71, &.{ v_b4, 34, 0 });
    try a.op(71, &.{ v_b4, 33, 3 });
    try a.op(71, &.{ v_a4, 34, 0 });
    try a.op(71, &.{ v_a4, 33, 0 });
    try a.op(71, &.{ t_arr_v4, 6, 16 }); // ArrayStride 16
    inline for (.{ t_sb4, t_sa4 }) |t| {
        try a.op(71, &.{ t, 2 });
        try a.op(72, &.{ t, 0, 35, 0 });
    }

    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(20, &.{t_bool});
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f32, if (c_h16) t_f16 else t_f32, c_arrlen });
    try a.op(23, &.{ t_v4u32, t_u32, 4 });
    try a.op(28, &.{ t_arr_v4, t_v4u32, c_arrlen });
    try a.op(30, &.{ t_sb4, t_arr_v4 });
    try a.op(32, &.{ t_ptr_sb4, 12, t_sb4 });
    try a.op(59, &.{ t_ptr_sb4, v_b4, 12 });
    try a.op(32, &.{ t_ptr_sb4_v4, 12, t_v4u32 });
    try a.op(30, &.{ t_sa4, t_arr_v4 });
    try a.op(32, &.{ t_ptr_sa4, 12, t_sa4 });
    try a.op(59, &.{ t_ptr_sa4, v_a4, 12 });
    try a.op(30, &.{ t_sc, t_arr_f32 });
    try a.op(32, &.{ t_ptr_sc, 12, t_sc });
    try a.op(59, &.{ t_ptr_sc, v_c, 12 });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push });
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_sc_f32, 12, if (c_h16) t_f16 else t_f32 });

    // Workgroup slab (no layout decorations): B 2 x [32][WGN] f16 at 0,
    // A 2 x [128][32] f16 at A_BASE.
    try a.op(43, &.{ t_u32, c_bsh_len, A_BASE + 2 * A_SLAB });
    try a.op(28, &.{ t_bsh, t_f16, c_bsh_len });
    try a.op(32, &.{ t_ptr_wg_bsh, 4, t_bsh });
    try a.op(59, &.{ t_ptr_wg_bsh, v_bsh, 4 });
    try a.op(32, &.{ t_ptr_wg_f16, 4, t_f16 });

    try a.op(43, &.{ t_u32, c_u0, 0 });
    try a.op(43, &.{ t_u32, c_u1, 1 });
    try a.op(43, &.{ t_u32, c_u2, 2 });
    try a.op(43, &.{ t_u32, c_u3, 3 });
    try a.op(43, &.{ t_u32, c_u4, 4 });
    try a.op(43, &.{ t_u32, c_u5, 5 });
    try a.op(43, &.{ t_u32, c_u6, 6 });
    try a.op(43, &.{ t_u32, c_u7, 7 });
    try a.op(43, &.{ t_u32, c_u8, 8 });
    try a.op(43, &.{ t_u32, c_u12, 12 });
    try a.op(43, &.{ t_u32, c_u31, 31 });
    try a.op(43, &.{ t_u32, c_u16, 16 });
    try a.op(43, &.{ t_u32, c_u32c, 32 });
    try a.op(43, &.{ t_u32, c_u64, 64 });
    try a.op(43, &.{ t_u32, c_u128, 128 });
    try a.op(43, &.{ t_u32, c_u4096, 4096 });
    try a.op(43, &.{ t_u32, c_mag_mask, 0x007F007F });
    try a.op(43, &.{ t_u32, c_sgn_mask, 0x00800080 });
    try a.op(43, &.{ t_u32, c_u264, 0x108 });
    try a.op(43, &.{ t_u32, c_scope_sub, 3 });
    try a.op(43, &.{ t_u32, c_scope_wg, 2 });

    try a.op(4456, &.{ t_mat_a, t_f16, c_scope_sub, c_u16, c_u16, c_u0 });
    try a.op(4456, &.{ t_mat_b, t_f16, c_scope_sub, c_u16, c_u16, c_u1 });
    try a.op(4456, &.{ t_mat_c, if (acc_h16) t_f16 else t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    if (acc_h16 and !c_h16) try a.op(4456, &.{ t_mat_c32, t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(23, &.{ t_v2f16, t_f16, 2 });
    try a.op(43, &.{ t_f32, c_f32_0, 0 });
    if (acc_h16) try a.op(43, &.{ t_f16, c_f16_0, 0 });
    try a.op(43, &.{ t_f16, c_h256, 0x5C00 });
    try a.op(44, &.{ t_v2f16, c_v2_256, c_h256, c_h256 });
    try a.op(44, &.{ t_mat_c, c_acc0, if (acc_h16) c_f16_0 else c_f32_0 });

    // small u32 constants used in the body (must live in the global section)
    // wg N width / thread count (equal in both configs).
    const c_wgn = if (WGN == 128) c_u128 else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, WGN });
        break :blk c;
    };
    // B sub-slab parity offset and the B chunk row mask/shift.
    const c_bslab = if (B_SLAB == 4096) c_u4096 else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, B_SLAB });
        break :blk c;
    };
    const c_bmask = if (WGN == 128) c_u7 else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, WGN / 16 - 1 });
        break :blk c;
    };
    const c_bshift = if (WGN == 128) c_u3 else c_u4; // log2(WGN/16)
    var c_stage: [4]u32 = undefined; // t*THREADS: staging chunk strides
    c_stage[0] = c_u0;
    c_stage[1] = c_wgn;
    for (2..4) |t| {
        c_stage[t] = a.id();
        try a.op(43, &.{ t_u32, c_stage[t], @intCast(t * THREADS) });
    }
    var c_k16: [4]u32 = undefined; // ks*16: A tile k offsets within a 64 step
    c_k16[0] = c_u0;
    c_k16[1] = c_u16;
    c_k16[2] = c_u32c;
    c_k16[3] = a.id();
    try a.op(43, &.{ t_u32, c_k16[3], 48 });
    var c_col16: [8]u32 = undefined; // nt*16: C col offsets
    c_col16[0] = c_u0;
    c_col16[1] = c_u16;
    c_col16[2] = c_u32c;
    c_col16[3] = c_k16[3];
    c_col16[4] = c_u64;
    for (5..8) |nt| {
        c_col16[nt] = a.id();
        try a.op(43, &.{ t_u32, c_col16[nt], @intCast(nt * 16) });
    }
    // b_sh tile offsets: parity*B_SLAB + ks*(16*WGN) + nt*16.
    var c_bt: [2][2][8]u32 = undefined;
    for (0..2) |par| {
        for (0..2) |ks| {
            for (0..8) |nt| {
                if (par == 0 and ks == 0) {
                    c_bt[par][ks][nt] = c_col16[nt];
                } else {
                    c_bt[par][ks][nt] = a.id();
                    try a.op(43, &.{ t_u32, c_bt[par][ks][nt], @intCast(par * B_SLAB + ks * (16 * WGN) + nt * 16) });
                }
            }
        }
    }
    // A slab constants: row stride and sub-slab size (alias the generic
    // constants when the values coincide), region base, and the a_sh tile
    // offsets 8192 + parity*A_SLAB + r*16*A_STRIDE + ks*16.
    const c_astride = if (A_STRIDE == 32) c_u32c else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, A_STRIDE });
        break :blk c;
    };
    const c_aslab = if (A_SLAB == 4096) c_u4096 else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, A_SLAB });
        break :blk c;
    };
    const c_ash0 = a.id();
    try a.op(43, &.{ t_u32, c_ash0, A_BASE });
    var c_at: [2][2][4]u32 = undefined;
    for (0..2) |par| {
        for (0..2) |ks| {
            for (0..4) |r| {
                if (par == 0 and ks == 0 and r == 0) {
                    c_at[par][ks][r] = c_ash0;
                } else {
                    c_at[par][ks][r] = a.id();
                    try a.op(43, &.{ t_u32, c_at[par][ks][r], @intCast(A_BASE + par * A_SLAB + r * 16 * A_STRIDE + ks * 16) });
                }
            }
        }
    }

    // --- function ---
    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_c = a.id();
    const tile_r = a.id();
    try a.op(81, &.{ t_u32, tile_c, gidv, 0 });
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    const col0 = a.id();
    const row0 = a.id();
    try a.op(132, &.{ t_u32, col0, tile_c, c_wgn }); // WGN cols per wg
    try a.op(132, &.{ t_u32, row0, tile_r, c_u128 }); // 128 rows per wg

    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const lx = a.id();
    const ly = a.id(); // subgroup index 0..7
    try a.op(81, &.{ t_u32, lx, lidv, 0 });
    try a.op(81, &.{ t_u32, ly, lidv, 1 });
    const flat = a.id(); // ly*32 + lx, 0..127
    const lymul = a.id();
    try a.op(132, &.{ t_u32, lymul, ly, c_u32c });
    try a.op(128, &.{ t_u32, flat, lymul, lx });

    var push_vals: [4]u32 = undefined;
    inline for (0..4) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c_u0,
            1 => c_u1,
            2 => c_u2,
            else => c_u3,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_n = push_vals[1];
    const p_k = push_vals[2];
    const p_stride = push_vals[3];

    // Warp tiling: a 2x2 grid of 64x64 tiles per warp — 4 A x 4 B fragments
    // feed 16 MMAs per k-step (2 MMAs per fragment load).
    const warp_m = a.id(); // ly & 1
    try a.op(199, &.{ t_u32, warp_m, ly, c_u1 });
    const warp_n = a.id(); // ly >> 1
    try a.op(194, &.{ t_u32, warp_n, ly, c_u1 });
    const wm64 = a.id();
    try a.op(132, &.{ t_u32, wm64, warp_m, c_u64 });
    const row_s = a.id();
    try a.op(128, &.{ t_u32, row_s, row0, wm64 });
    const wn64 = a.id();
    try a.op(132, &.{ t_u32, wn64, warp_n, c_u64 });
    const col_s = a.id(); // warp's first column tile (of 4)
    try a.op(128, &.{ t_u32, col_s, col0, wn64 });
    const a_shbase = a.id(); // warp's A row-block base within an A sub-slab
    try a.op(132, &.{ t_u32, a_shbase, wm64, c_astride });

    // Loop-invariant B staging indices: thread `flat`, uvec4 t (of 2)
    // covers sub-slab elements 16v..16v+15 with v = flat + t*THREADS, i.e.
    // B k-row v/(WGN/16), columns (v%(WGN/16))*16 .. +15 of the workgroup's
    // column window (consecutive lanes read consecutive 16-byte quads).
    var brow_t: [2]u32 = undefined;
    var bco_t: [2]u32 = undefined;
    var sbase0_t: [2]u32 = undefined;
    var sbase1_t: [2]u32 = undefined;
    for (0..2) |t| {
        var v = flat;
        if (t > 0) {
            const vn = a.id();
            try a.op(128, &.{ t_u32, vn, flat, c_wgn });
            v = vn;
        }
        brow_t[t] = a.id();
        try a.op(194, &.{ t_u32, brow_t[t], v, c_bshift }); // v / (WGN/16)
        const vmod = a.id();
        try a.op(199, &.{ t_u32, vmod, v, c_bmask }); // v % (WGN/16)
        const bcol16 = a.id();
        try a.op(196, &.{ t_u32, bcol16, vmod, c_u4 }); // * 16
        bco_t[t] = a.id();
        try a.op(128, &.{ t_u32, bco_t[t], col0, bcol16 });
        sbase0_t[t] = a.id();
        try a.op(196, &.{ t_u32, sbase0_t[t], v, c_u4 }); // * 16
        sbase1_t[t] = a.id();
        try a.op(128, &.{ t_u32, sbase1_t[t], sbase0_t[t], c_bslab });
    }

    // Loop-invariant A staging indices: thread `flat`, uvec4 t (of AQ)
    // covers sub-slab quad q = flat + t*THREADS -> A row q/4 (of the wg's
    // 128), k-quad q%4 (8 f16 each; consecutive lanes cover a row's 4 quads
    // then step rows). Global uvec4 index = ((row0 + row)*k + kb)/8 + q%4;
    // shared f16 store base = A_BASE + parity*A_SLAB + row*A_STRIDE + (q%4)*8.
    var a_inv_t: [4]u32 = undefined;
    var asb0_t: [4]u32 = undefined;
    var asb1_t: [4]u32 = undefined;
    for (0..AQ) |t| {
        var q = flat;
        if (t > 0) {
            const qn = a.id();
            try a.op(128, &.{ t_u32, qn, flat, c_stage[t] });
            q = qn;
        }
        const arow = a.id();
        try a.op(194, &.{ t_u32, arow, q, c_u2 }); // q / 4
        const aqc = a.id();
        try a.op(199, &.{ t_u32, aqc, q, c_u3 }); // q % 4
        const grow = a.id();
        try a.op(128, &.{ t_u32, grow, row0, arow });
        const gmul = a.id();
        try a.op(132, &.{ t_u32, gmul, grow, p_k });
        const gq = a.id();
        try a.op(194, &.{ t_u32, gq, gmul, c_u3 }); // f16 -> uvec4 index
        a_inv_t[t] = a.id();
        try a.op(128, &.{ t_u32, a_inv_t[t], gq, aqc });
        const srow = a.id();
        try a.op(132, &.{ t_u32, srow, arow, c_astride });
        const scol = a.id();
        try a.op(196, &.{ t_u32, scol, aqc, c_u3 }); // * 8
        const s0 = a.id();
        try a.op(128, &.{ t_u32, s0, srow, scol });
        asb0_t[t] = a.id();
        try a.op(128, &.{ t_u32, asb0_t[t], s0, c_ash0 });
        asb1_t[t] = a.id();
        try a.op(128, &.{ t_u32, asb1_t[t], asb0_t[t], c_aslab });
    }

    // Staging emitters (loads separated from decode+store so the global
    // loads can issue before the MMA section they overlap with).
    const Stage = struct {
        b_f16: bool,
        t_u32: u32,
        t_f16: u32,
        t_v2f16: u32,
        t_ptr_wg_f16: u32,
        v_bsh: u32,
        p_stride: u32,
        c_u0: u32,
        c_u2: u32,
        c_u7: u32,
        c_u8: u32,
        c_mag_mask: u32,
        c_sgn_mask: u32,
        c_v2_256: u32,
        c_j: [4]u32,
        c_j8: [8]u32,
        c_w4: [4]u32,
        c_u4: u32,
        t_v4u32: u32,
        t_ptr_sb4_v4: u32,
        v_b4: u32,
        brow: [2]u32,
        bco: [2]u32,

        /// Global loads for the sub-slab whose first k-row is `kb`: per
        /// 16-element chunk, one uvec4 for fp8 B (16 bytes; entries 2/3
        /// unused) or two for f16 B (32 bytes).
        fn loads(st: @This(), asm_: *Asm, kb: u32) ![4]u32 {
            var quads: [4]u32 = undefined;
            for (0..2) |t| {
                const bk = asm_.id();
                try asm_.op(128, &.{ st.t_u32, bk, kb, st.brow[t] });
                const bmul = asm_.id();
                try asm_.op(132, &.{ st.t_u32, bmul, bk, st.p_stride });
                const boff = asm_.id();
                try asm_.op(128, &.{ st.t_u32, boff, bmul, st.bco[t] });
                if (st.b_f16) {
                    const qidx0 = asm_.id();
                    try asm_.op(194, &.{ st.t_u32, qidx0, boff, st.c_j[3] }); // /8 (f16 elems)
                    for (0..2) |qi| {
                        var qidx = qidx0;
                        if (qi > 0) {
                            const qn = asm_.id();
                            try asm_.op(128, &.{ st.t_u32, qn, qidx0, st.c_j[1] });
                            qidx = qn;
                        }
                        const qptr = asm_.id();
                        try asm_.op(65, &.{ st.t_ptr_sb4_v4, qptr, st.v_b4, st.c_u0, qidx });
                        quads[2 * t + qi] = asm_.id();
                        try asm_.op(61, &.{ st.t_v4u32, quads[2 * t + qi], qptr });
                    }
                } else {
                    const qidx = asm_.id();
                    try asm_.op(194, &.{ st.t_u32, qidx, boff, st.c_u4 }); // /16 (bytes)
                    const qptr = asm_.id();
                    try asm_.op(65, &.{ st.t_ptr_sb4_v4, qptr, st.v_b4, st.c_u0, qidx });
                    quads[t] = asm_.id();
                    try asm_.op(61, &.{ st.t_v4u32, quads[t], qptr });
                }
            }
            return quads;
        }

        /// Stage the quads into the sub-slab whose per-chunk store bases are
        /// `sbase` (parity offset prefolded). fp8 B: SWAR e4m3 -> f16 pair
        /// decode (fields land on f16 layout, exact *256). f16 B: plain
        /// bitcast copy, 8 consecutive f16 per quad (same as the A slab).
        fn stores(st: @This(), asm_: *Asm, quads: [4]u32, sbase: [2]u32) !void {
            if (st.b_f16) {
                for (0..2) |t| {
                    for (0..2) |qi| {
                        var qbase = sbase[t];
                        if (qi > 0) {
                            const qb = asm_.id();
                            try asm_.op(128, &.{ st.t_u32, qb, sbase[t], st.c_u8 });
                            qbase = qb;
                        }
                        for (0..4) |wi| {
                            const wv = asm_.id();
                            try asm_.op(81, &.{ st.t_u32, wv, quads[2 * t + qi], @intCast(wi) });
                            const hv2 = asm_.id();
                            try asm_.op(124, &.{ st.t_v2f16, hv2, wv }); // OpBitcast
                            for (0..2) |j| {
                                const hval = asm_.id();
                                try asm_.op(81, &.{ st.t_f16, hval, hv2, @intCast(j) });
                                var eidx = qbase;
                                if (wi * 2 + j > 0) {
                                    const ei = asm_.id();
                                    try asm_.op(128, &.{ st.t_u32, ei, qbase, st.c_j8[wi * 2 + j] });
                                    eidx = ei;
                                }
                                const bsptr = asm_.id();
                                try asm_.op(65, &.{ st.t_ptr_wg_f16, bsptr, st.v_bsh, eidx });
                                try asm_.op(62, &.{ bsptr, hval });
                            }
                        }
                    }
                }
                return;
            }
            for (0..2) |tq| {
                for (0..4) |wi| {
                const wv = asm_.id();
                try asm_.op(81, &.{ st.t_u32, wv, quads[tq], @intCast(wi) }); // CompositeExtract
                var pair_vals: [2]u32 = undefined; // v2f16: bytes 0/2, bytes 1/3
                for (0..2) |half| {
                    var src = wv;
                    if (half == 1) {
                        const shd = asm_.id();
                        try asm_.op(194, &.{ st.t_u32, shd, wv, st.c_u8 });
                        src = shd;
                    }
                    const magp = asm_.id();
                    try asm_.op(199, &.{ st.t_u32, magp, src, st.c_mag_mask });
                    const sgnp = asm_.id();
                    try asm_.op(199, &.{ st.t_u32, sgnp, src, st.c_sgn_mask });
                    const mag_sh = asm_.id();
                    try asm_.op(196, &.{ st.t_u32, mag_sh, magp, st.c_u7 });
                    const sgn_sh = asm_.id();
                    try asm_.op(196, &.{ st.t_u32, sgn_sh, sgnp, st.c_u8 });
                    const hbits = asm_.id();
                    try asm_.op(197, &.{ st.t_u32, hbits, mag_sh, sgn_sh });
                    const hv2 = asm_.id();
                    try asm_.op(124, &.{ st.t_v2f16, hv2, hbits });
                    pair_vals[half] = asm_.id();
                    try asm_.op(133, &.{ st.t_v2f16, pair_vals[half], hv2, st.c_v2_256 });
                }
                var wbase = sbase[tq];
                if (wi > 0) {
                    const wb = asm_.id();
                    try asm_.op(128, &.{ st.t_u32, wb, sbase[tq], st.c_w4[wi] });
                    wbase = wb;
                }
                for (0..4) |j| {
                    const hval = asm_.id();
                    try asm_.op(81, &.{ st.t_f16, hval, pair_vals[j & 1], @intCast(j >> 1) });
                    var eidx = wbase;
                    if (j > 0) {
                        const ei = asm_.id();
                        try asm_.op(128, &.{ st.t_u32, ei, wbase, st.c_j[j] });
                        eidx = ei;
                    }
                    const bsptr = asm_.id();
                    try asm_.op(65, &.{ st.t_ptr_wg_f16, bsptr, st.v_bsh, eidx });
                    try asm_.op(62, &.{ bsptr, hval });
                }
                }
            }
        }
    };
    const stage: Stage = .{
        .b_f16 = b_f16,
        .t_u32 = t_u32,
        .t_f16 = t_f16,
        .t_v2f16 = t_v2f16,
        .t_ptr_wg_f16 = t_ptr_wg_f16,
        .v_bsh = v_bsh,
        .p_stride = p_stride,
        .c_u0 = c_u0,
        .c_u2 = c_u2,
        .c_u7 = c_u7,
        .c_u8 = c_u8,
        .c_mag_mask = c_mag_mask,
        .c_sgn_mask = c_sgn_mask,
        .c_v2_256 = c_v2_256,
        .c_j = .{ c_u0, c_u1, c_u2, c_u3 },
        .c_j8 = .{ c_u0, c_u1, c_u2, c_u3, c_u4, c_u5, c_u6, c_u7 },
        .c_w4 = .{ c_u0, c_u4, c_u8, c_u12 },
        .c_u4 = c_u4,
        .t_v4u32 = t_v4u32,
        .t_ptr_sb4_v4 = t_ptr_sb4_v4,
        .v_b4 = v_b4,
        .brow = brow_t,
        .bco = bco_t,
    };

    // A staging: plain f16 copy into the shared A region (no decode — the
    // scale is already folded in), same load/store split as B.
    const AStage = struct {
        aq: usize,
        t_u32: u32,
        t_f16: u32,
        t_v2f16: u32,
        t_v4u32: u32,
        t_ptr_sb4_v4: u32,
        v_a4: u32,
        t_ptr_wg_f16: u32,
        v_bsh: u32,
        c_u0: u32,
        c_u3: u32,
        c_j: [8]u32,
        a_inv: [4]u32,

        /// aq uvec4 (128-bit) global loads of the A sub-slab whose first
        /// k-column is `kb`.
        fn loads(st: @This(), asm_: *Asm, kb: u32) ![4]u32 {
            const kbq = asm_.id();
            try asm_.op(194, &.{ st.t_u32, kbq, kb, st.c_u3 }); // kb / 8
            var quads: [4]u32 = undefined;
            for (0..st.aq) |t| {
                const qidx = asm_.id();
                try asm_.op(128, &.{ st.t_u32, qidx, st.a_inv[t], kbq });
                const qptr = asm_.id();
                try asm_.op(65, &.{ st.t_ptr_sb4_v4, qptr, st.v_a4, st.c_u0, qidx });
                quads[t] = asm_.id();
                try asm_.op(61, &.{ st.t_v4u32, quads[t], qptr });
            }
            return quads;
        }

        /// Store the quads' 8 f16 each into the sub-slab whose per-quad
        /// store bases are `sbase` (parity offset prefolded).
        fn stores(st: @This(), asm_: *Asm, quads: [4]u32, sbase: [4]u32) !void {
            for (0..st.aq) |t| {
                for (0..4) |wi| {
                    const wv = asm_.id();
                    try asm_.op(81, &.{ st.t_u32, wv, quads[t], @intCast(wi) });
                    const hv2 = asm_.id();
                    try asm_.op(124, &.{ st.t_v2f16, hv2, wv }); // OpBitcast
                    for (0..2) |j| {
                        const hval = asm_.id();
                        try asm_.op(81, &.{ st.t_f16, hval, hv2, @intCast(j) });
                        var eidx = sbase[t];
                        if (wi * 2 + j > 0) {
                            const ei = asm_.id();
                            try asm_.op(128, &.{ st.t_u32, ei, sbase[t], st.c_j[wi * 2 + j] });
                            eidx = ei;
                        }
                        const bsptr = asm_.id();
                        try asm_.op(65, &.{ st.t_ptr_wg_f16, bsptr, st.v_bsh, eidx });
                        try asm_.op(62, &.{ bsptr, hval });
                    }
                }
            }
        }
    };
    const astage: AStage = .{
        .aq = AQ,
        .t_u32 = t_u32,
        .t_f16 = t_f16,
        .t_v2f16 = t_v2f16,
        .t_v4u32 = t_v4u32,
        .t_ptr_sb4_v4 = t_ptr_sb4_v4,
        .v_a4 = v_a4,
        .t_ptr_wg_f16 = t_ptr_wg_f16,
        .v_bsh = v_bsh,
        .c_u0 = c_u0,
        .c_u3 = c_u3,
        .c_j = .{ c_u0, c_u1, c_u2, c_u3, c_u4, c_u5, c_u6, c_u7 },
        .a_inv = a_inv_t,
    };

    // Prologue: fill sub-slab 0 with the first 32 k-rows/columns.
    {
        const w0 = try stage.loads(&a, c_u0);
        const x0 = try astage.loads(&a, c_u0);
        try stage.stores(&a, w0, sbase0_t);
        try astage.stores(&a, x0, asb0_t);
    }

    const lb_head = a.id();
    const lb_cond = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();
    // Pre-allocated ids for the loop-carried values (phi back-edges).
    const k0n = a.id();
    var acc_next: [4][4]u32 = undefined; // final MulAdd per (row block, col tile)
    for (&acc_next) |*row| for (row) |*v| {
        v.* = a.id();
    };

    try a.op(249, &.{lb_head});
    try a.op(248, &.{lb_head});
    const k0v = a.id();
    try a.op(245, &.{ t_u32, k0v, c_u0, lb_entry, k0n, lb_cont }); // OpPhi
    var acc_phi: [4][4]u32 = undefined;
    for (&acc_phi, acc_next) |*prow, nrow| {
        for (prow, nrow) |*ap, an| {
            ap.* = a.id();
            try a.op(245, &.{ t_mat_c, ap.*, c_acc0, lb_entry, an, lb_cont });
        }
    }
    try a.op(246, &.{ lb_merge, lb_cont, 0 });
    try a.op(249, &.{lb_cond});
    try a.op(248, &.{lb_cond});
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, k0v, p_k });
    try a.op(250, &.{ cmp, lb_body, lb_merge });

    try a.op(248, &.{lb_body});
    var acc_cur = acc_phi;
    // Half-step 0: consume sub-slab 0, prefetch k0+32 into sub-slab 1.
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 }); // OpControlBarrier
    const kb1 = a.id();
    try a.op(128, &.{ t_u32, kb1, k0v, c_u32c });
    const w1 = try stage.loads(&a, kb1);
    const x1 = try astage.loads(&a, kb1);
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..4) |r| {
            const aoff = a.id();
            try a.op(128, &.{ t_u32, aoff, a_shbase, c_at[0][ks][r] });
            const aptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, aptr, v_bsh, aoff });
            ma[r] = a.id();
            try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u0, c_astride });
        }
        for (0..4) |nt| {
            const boff = a.id();
            try a.op(128, &.{ t_u32, boff, c_bt[0][ks][nt], wn64 });
            const bptr2 = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, bptr2, v_bsh, boff });
            const mb = a.id();
            try a.op(4457, &.{ t_mat_b, mb, bptr2, c_u0, c_wgn });
            for (0..4) |r| {
                const acc_out = a.id();
                try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try stage.stores(&a, w1, sbase1_t);
    try astage.stores(&a, x1, asb1_t);
    // Half-step 1: consume sub-slab 1, prefetch k0+64 (clamped to 0 on the
    // last iteration — the garbage fill is never consumed) into sub-slab 0.
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });
    const kb2 = a.id();
    try a.op(128, &.{ t_u32, kb2, k0v, c_u64 });
    const kb2_ok = a.id();
    try a.op(176, &.{ t_bool, kb2_ok, kb2, p_k }); // ULessThan
    const kb2s = a.id();
    try a.op(169, &.{ t_u32, kb2s, kb2_ok, kb2, c_u0 }); // Select
    const w2 = try stage.loads(&a, kb2s);
    const x2 = try astage.loads(&a, kb2s);
    for (0..2) |ks| {
        var ma: [4]u32 = undefined;
        for (0..4) |r| {
            const aoff = a.id();
            try a.op(128, &.{ t_u32, aoff, a_shbase, c_at[1][ks][r] });
            const aptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, aptr, v_bsh, aoff });
            ma[r] = a.id();
            try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u0, c_astride });
        }
        for (0..4) |nt| {
            const boff = a.id();
            try a.op(128, &.{ t_u32, boff, c_bt[1][ks][nt], wn64 });
            const bptr2 = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, bptr2, v_bsh, boff });
            const mb = a.id();
            try a.op(4457, &.{ t_mat_b, mb, bptr2, c_u0, c_wgn });
            for (0..4) |r| {
                const acc_out = if (ks == 1) acc_next[r][nt] else a.id();
                try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try stage.stores(&a, w2, sbase0_t);
    try astage.stores(&a, x2, asb0_t);
    try a.op(249, &.{lb_cont});

    try a.op(248, &.{lb_cont});
    try a.op(128, &.{ t_u32, k0n, k0v, c_u64 });
    try a.op(249, &.{lb_head});

    // merge: store 4x4 C tiles at (row_s + r*16, col_s + nt*16)
    try a.op(248, &.{lb_merge});
    const rn16 = a.id();
    try a.op(132, &.{ t_u32, rn16, c_u16, p_n });
    var c_rowmul = a.id();
    try a.op(132, &.{ t_u32, c_rowmul, row_s, p_n });
    for (0..4) |r| {
        if (r > 0) {
            const nx = a.id();
            try a.op(128, &.{ t_u32, nx, c_rowmul, rn16 });
            c_rowmul = nx;
        }
        for (0..4) |nt| {
            const ccol = a.id();
            try a.op(128, &.{ t_u32, ccol, col_s, c_col16[nt] });
            const cbase = a.id();
            try a.op(128, &.{ t_u32, cbase, c_rowmul, ccol });
            const cptr = a.id();
            try a.op(65, &.{ t_ptr_sc_f32, cptr, v_c, c_u0, cbase });
            var cval = acc_phi[r][nt];
            if (acc_h16 and !c_h16) {
                // f16 accumulators store through an f32 conversion.
                const cv = a.id();
                try a.op(115, &.{ t_mat_c32, cv, cval }); // OpFConvert
                cval = cv;
            }
            try a.op(4458, &.{ cptr, cval, c_u0, p_n });
        }
    }
    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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
    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    const main_fn = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_f16 = a.id();
    const t_f32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f16 = a.id();
    const t_arr_f32b = a.id(); // second f16 array (distinct id for S)
    const t_sk = a.id(); // struct { [N]f16 } (K)
    const t_sq = a.id(); // struct { [N]f16 } (Q)
    const t_ss = a.id(); // struct { [N]f16 } (S, half-precision scores)
    const t_ptr_sk = a.id();
    const t_ptr_sq = a.id();
    const t_ptr_ss = a.id();
    const v_k = a.id();
    const v_q = a.id();
    const v_s = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_sk_f16 = a.id();
    const t_ptr_sq_f16 = a.id();
    const t_ptr_ss_f32 = a.id();
    // u32 view of S (bound again at binding 3) for the coalesced copy-out,
    // plus the 128x128 f16 workgroup bounce slab.
    const t_arr_u32s = a.id();
    const t_ss4 = a.id();
    const t_ptr_ss4 = a.id();
    const v_s4 = a.id();
    const t_ptr_ss4_u32 = a.id();
    const t_v2f16 = a.id();
    const c_ssh_len = a.id();
    const t_ssh = a.id();
    const t_ptr_wg_ssh = a.id();
    const v_ssh = a.id();
    const t_ptr_wg_f16 = a.id();

    const c_u0 = a.id();
    const c_u1 = a.id();
    const c_u2 = a.id();
    const c_u3 = a.id();
    const c_u4 = a.id();
    const c_u5 = a.id();
    const c_u6 = a.id();
    const c_u63 = a.id();
    const c_u16 = a.id();
    const c_u32c = a.id();
    const c_u64 = a.id();
    const c_u128 = a.id();
    const c_u264 = a.id();
    const c_scope_sub = a.id();
    const c_scope_wg = a.id();
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const t_mat_h = a.id(); // f16 store type: S is written half-precision
    const c_f32_0 = a.id();
    const c_acc0 = a.id();

    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1});
    try a.op(17, &.{9});
    try a.op(17, &.{5345});
    try a.op(17, &.{6022});
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_16bit_storage");
    try a.op(14, &.{ 0, 3 });
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_k, v_q, v_s, v_s4, v_push, v_ssh });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, 4, 1 }); // LocalSize 32 4 1

    try a.op(71, &.{ gid_var, 11, 26 }); // WorkgroupId
    try a.op(71, &.{ lid_var, 11, 27 }); // LocalInvocationId
    try a.op(71, &.{ t_arr_f16, 6, 2 });
    try a.op(71, &.{ t_arr_f32b, 6, 2 }); // S is stored f16
    try a.op(71, &.{ t_arr_u32s, 6, 4 });
    inline for (.{ t_sk, t_sq, t_ss, t_ss4 }) |t| {
        try a.op(71, &.{ t, 2 });
        try a.op(72, &.{ t, 0, 35, 0 });
    }
    try a.op(71, &.{ t_push, 2 });
    inline for (0..8) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    inline for (.{ v_k, v_q, v_s }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 });
        try a.op(71, &.{ v, 33, @intCast(binding) });
    }
    try a.op(71, &.{ v_s4, 34, 0 });
    try a.op(71, &.{ v_s4, 33, 3 });

    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f16, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_f32b, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_u32s, t_u32, c_arrlen });
    try a.op(30, &.{ t_sk, t_arr_f16 });
    try a.op(30, &.{ t_sq, t_arr_f16 });
    try a.op(30, &.{ t_ss, t_arr_f32b });
    try a.op(30, &.{ t_ss4, t_arr_u32s });
    try a.op(32, &.{ t_ptr_sk, 12, t_sk });
    try a.op(32, &.{ t_ptr_sq, 12, t_sq });
    try a.op(32, &.{ t_ptr_ss, 12, t_ss });
    try a.op(32, &.{ t_ptr_ss4, 12, t_ss4 });
    try a.op(59, &.{ t_ptr_sk, v_k, 12 });
    try a.op(59, &.{ t_ptr_sq, v_q, 12 });
    try a.op(59, &.{ t_ptr_ss, v_s, 12 });
    try a.op(59, &.{ t_ptr_ss4, v_s4, 12 });
    try a.op(23, &.{ t_v2f16, t_f16, 2 });
    // Push block: 8 words (u0..u5, f0, f1) to match the eltwise layout.
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push });
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_sk_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_sq_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_ss_f32, 12, t_f16 });
    try a.op(32, &.{ t_ptr_ss4_u32, 12, t_u32 });

    // Workgroup bounce slab (no layout decorations): [128][128] f16.
    try a.op(43, &.{ t_u32, c_ssh_len, 16384 });
    try a.op(28, &.{ t_ssh, t_f16, c_ssh_len });
    try a.op(32, &.{ t_ptr_wg_ssh, 4, t_ssh });
    try a.op(59, &.{ t_ptr_wg_ssh, v_ssh, 4 });
    try a.op(32, &.{ t_ptr_wg_f16, 4, t_f16 });

    try a.op(43, &.{ t_u32, c_u0, 0 });
    try a.op(43, &.{ t_u32, c_u1, 1 });
    try a.op(43, &.{ t_u32, c_u2, 2 });
    try a.op(43, &.{ t_u32, c_u3, 3 });
    try a.op(43, &.{ t_u32, c_u4, 4 });
    try a.op(43, &.{ t_u32, c_u5, 5 });
    try a.op(43, &.{ t_u32, c_u6, 6 });
    try a.op(43, &.{ t_u32, c_u63, 63 });
    try a.op(43, &.{ t_u32, c_u16, 16 });
    try a.op(43, &.{ t_u32, c_u32c, 32 });
    try a.op(43, &.{ t_u32, c_u64, 64 });
    try a.op(43, &.{ t_u32, c_u128, 128 });
    try a.op(43, &.{ t_u32, c_u264, 0x108 });
    try a.op(43, &.{ t_u32, c_scope_sub, 3 });
    try a.op(43, &.{ t_u32, c_scope_wg, 2 });

    try a.op(4456, &.{ t_mat_a, t_f16, c_scope_sub, c_u16, c_u16, c_u0 });
    try a.op(4456, &.{ t_mat_b, t_f16, c_scope_sub, c_u16, c_u16, c_u1 });
    try a.op(4456, &.{ t_mat_c, t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(4456, &.{ t_mat_h, t_f16, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(43, &.{ t_f32, c_f32_0, 0 });
    try a.op(44, &.{ t_mat_c, c_acc0, c_f32_0 });

    // ks*16 k-offsets up to hd (also reused as nt*16 column offsets).
    const c_k16 = try gpa.alloc(u32, @max(hd / 16, 4));
    defer gpa.free(c_k16);
    c_k16[0] = c_u0;
    c_k16[1] = c_u16;
    c_k16[2] = c_u32c;
    for (3..c_k16.len) |i| {
        if (i * 16 == 128) {
            c_k16[i] = c_u128;
            continue;
        }
        c_k16[i] = a.id();
        try a.op(43, &.{ t_u32, c_k16[i], @intCast(i * 16) });
    }
    const c_hd = switch (hd) {
        128 => c_u128,
        64 => c_u64,
        else => blk: {
            const c = a.id();
            try a.op(43, &.{ t_u32, c, hd });
            break :blk c;
        },
    };
    // s_sh tile offsets within the bounce slab: r*16*128 + nt*16.
    var c_st: [4][4]u32 = undefined;
    for (0..4) |r| {
        for (0..4) |nt| {
            if (r == 0) {
                c_st[r][nt] = c_k16[nt];
            } else {
                c_st[r][nt] = a.id();
                try a.op(43, &.{ t_u32, c_st[r][nt], @intCast(r * 2048 + nt * 16) });
            }
        }
    }
    // 256 = 16*16 already exists in c_k16 for wide-hd builds.
    const c_u256 = if (c_k16.len > 16) c_k16[16] else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, 256 });
        break :blk c;
    };

    // --- function (straight-line) ---
    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_c = a.id();
    const tile_r = a.id();
    const zidx = a.id();
    try a.op(81, &.{ t_u32, tile_c, gidv, 0 });
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    try a.op(81, &.{ t_u32, zidx, gidv, 2 });
    const col0 = a.id();
    const row0 = a.id();
    try a.op(132, &.{ t_u32, col0, tile_c, c_u128 });
    try a.op(132, &.{ t_u32, row0, tile_r, c_u128 });

    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const lx = a.id();
    const ly = a.id();
    try a.op(81, &.{ t_u32, lx, lidv, 0 });
    try a.op(81, &.{ t_u32, ly, lidv, 1 });
    const flat = a.id(); // ly*32 + lx, 0..127
    const lymul = a.id();
    try a.op(132, &.{ t_u32, lymul, ly, c_u32c });
    try a.op(128, &.{ t_u32, flat, lymul, lx });

    var push_vals: [6]u32 = undefined;
    inline for (0..6) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c_u0,
            1 => c_u1,
            2 => c_u2,
            3 => c_u3,
            4 => c_u4,
            else => c_u5,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_qstride = push_vals[0];
    const p_sstride = push_vals[1];
    const p_headoff = push_vals[2];
    const p_group = push_vals[3];
    const p_khead = push_vals[4];
    const p_splane = push_vals[5];

    // head = head_off + z; kv = head / group.
    const head = a.id();
    try a.op(128, &.{ t_u32, head, p_headoff, zidx });
    const kvh = a.id();
    try a.op(134, &.{ t_u32, kvh, head, p_group }); // UDiv

    // Warp tiling: 2x2 grid of 64x64 tiles — 4 A x 4 B fragments feed 16
    // MMAs per k-step (all loads are global here, so halving them matters
    // double).
    const warp_m = a.id();
    try a.op(199, &.{ t_u32, warp_m, ly, c_u1 }); // ly & 1
    const warp_n = a.id();
    try a.op(194, &.{ t_u32, warp_n, ly, c_u1 }); // ly >> 1
    const wm64 = a.id();
    try a.op(132, &.{ t_u32, wm64, warp_m, c_u64 });
    const row_s = a.id();
    try a.op(128, &.{ t_u32, row_s, row0, wm64 });
    const wn64 = a.id();
    try a.op(132, &.{ t_u32, wn64, warp_n, c_u64 });
    const col_w = a.id();
    try a.op(128, &.{ t_u32, col_w, col0, wn64 });

    // A (Q) base: row_s*q_stride + head*hd; row stride q_stride.
    const a_rowmul = a.id();
    try a.op(132, &.{ t_u32, a_rowmul, row_s, p_qstride });
    const headmul = a.id();
    try a.op(132, &.{ t_u32, headmul, head, c_hd });
    const a_base = a.id();
    try a.op(128, &.{ t_u32, a_base, a_rowmul, headmul });
    const a_row16 = a.id();
    try a.op(132, &.{ t_u32, a_row16, c_u16, p_qstride });

    // K block base: kv*k_head_stride. The global-load path folds col_w in
    // below; the staged path addresses the whole 128-col tile from col0.
    const kvmul = a.id();
    try a.op(132, &.{ t_u32, kvmul, kvh, p_khead });

    // Copy-out invariants: thread `flat` writes u32 words w = flat + i*128
    // of the wg's 128x128 f16 tile (word w = slab row w/64, f16 columns
    // (w%64)*2 — consecutive lanes write consecutive global words). The low
    // 6 bits of w never change across i, so the f16 column pair and the
    // global/shared bases are loop-invariant with fixed steps.
    const zmul = a.id();
    try a.op(132, &.{ t_u32, zmul, zidx, p_splane });
    const srow0 = a.id();
    try a.op(194, &.{ t_u32, srow0, flat, c_u6 }); // flat / 64
    const scol2 = a.id();
    const fmask = a.id();
    try a.op(199, &.{ t_u32, fmask, flat, c_u63 });
    try a.op(196, &.{ t_u32, scol2, fmask, c_u1 }); // * 2
    const grow0 = a.id();
    try a.op(128, &.{ t_u32, grow0, row0, srow0 });
    const growmul = a.id();
    try a.op(132, &.{ t_u32, growmul, grow0, p_sstride });
    const gsum0 = a.id();
    try a.op(128, &.{ t_u32, gsum0, zmul, growmul });
    const gsum1 = a.id();
    try a.op(128, &.{ t_u32, gsum1, gsum0, col0 });
    const gsum2 = a.id();
    try a.op(128, &.{ t_u32, gsum2, gsum1, scol2 });
    const gword0 = a.id();
    try a.op(194, &.{ t_u32, gword0, gsum2, c_u1 }); // f16 elems -> u32 words
    const erow = a.id();
    try a.op(132, &.{ t_u32, erow, srow0, c_u128 });
    const e0 = a.id();
    try a.op(128, &.{ t_u32, e0, erow, scol2 });

    // acc[r][nt] over hd/16 unrolled k-steps.
    var acc: [4][4]u32 = undefined;
    for (&acc) |*row| for (row) |*v| {
        v.* = c_acc0;
    };
    if (stage_k) {
        // K staged through the bounce slab: the 32 KB S tile is dead until
        // after the MMAs, so its first 16 KB holds each 64-deep K slab and
        // the B fragment loads become ldmatrix-from-shared — no extra
        // workgroup memory, occupancy unchanged. (The global K fragment
        // loads were the ~1 TF/s bottleneck: 12-15 KB row strides.)
        //
        // Staging reuses the copy-out invariants: thread `flat` owns slab
        // rows srow0, srow0+2, ... and the f16 column pair at scol2; the
        // matching global elements are kvmul + (k0+row)*s_stride + col0 +
        // scol2 (two scalar f16 loads — all four bindings are taken, so
        // there is no u32 view of K to pair-load through).
        const kbase0 = a.id();
        try a.op(128, &.{ t_u32, kbase0, kvmul, col0 });
        const kbase = a.id();
        try a.op(128, &.{ t_u32, kbase, kbase0, scol2 });
        const c2s = a.id();
        try a.op(128, &.{ t_u32, c2s, p_sstride, p_sstride }); // 2 rows/iter
        for (0..hd / 64) |s| {
            // Stage K rows s*64 .. s*64+64 of the per-head k-major block.
            const krow = a.id();
            try a.op(128, &.{ t_u32, krow, c_k16[s * 4], srow0 });
            const krowmul = a.id();
            try a.op(132, &.{ t_u32, krowmul, krow, p_sstride });
            var g = a.id();
            try a.op(128, &.{ t_u32, g, kbase, krowmul });
            var e = e0;
            for (0..32) |i| {
                if (i > 0) {
                    const gn = a.id();
                    try a.op(128, &.{ t_u32, gn, g, c2s });
                    g = gn;
                    const en = a.id();
                    try a.op(128, &.{ t_u32, en, e, c_u256 });
                    e = en;
                }
                const g1 = a.id();
                try a.op(128, &.{ t_u32, g1, g, c_u1 });
                const e1 = a.id();
                try a.op(128, &.{ t_u32, e1, e, c_u1 });
                const pairs = [2][2]u32{ .{ g, e }, .{ g1, e1 } };
                for (pairs) |pair| {
                    const kp = a.id();
                    try a.op(65, &.{ t_ptr_sk_f16, kp, v_k, c_u0, pair[0] });
                    const kv = a.id();
                    try a.op(61, &.{ t_f16, kv, kp });
                    const sp = a.id();
                    try a.op(65, &.{ t_ptr_wg_f16, sp, v_ssh, pair[1] });
                    try a.op(62, &.{ sp, kv });
                }
            }
            try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 }); // staged
            for (0..4) |kk| {
                const ks = s * 4 + kk;
                var a_off = a_base;
                if (ks > 0) {
                    const ao = a.id();
                    try a.op(128, &.{ t_u32, ao, a_base, c_k16[ks] });
                    a_off = ao;
                }
                var ma: [4]u32 = undefined;
                var ao_cur = a_off;
                for (0..4) |r| {
                    if (r > 0) {
                        const a2 = a.id();
                        try a.op(128, &.{ t_u32, a2, ao_cur, a_row16 });
                        ao_cur = a2;
                    }
                    const aptr = a.id();
                    try a.op(65, &.{ t_ptr_sq_f16, aptr, v_q, c_u0, ao_cur });
                    ma[r] = a.id();
                    try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u0, p_qstride });
                }
                for (0..4) |nt| {
                    // Shared B fragment: slab row kk*16, col wn64 + nt*16 —
                    // c_st[kk][nt] is exactly kk*2048 + nt*16.
                    const bidx = a.id();
                    try a.op(128, &.{ t_u32, bidx, c_st[kk][nt], wn64 });
                    const bptr = a.id();
                    try a.op(65, &.{ t_ptr_wg_f16, bptr, v_ssh, bidx });
                    const mb = a.id();
                    try a.op(4457, &.{ t_mat_b, mb, bptr, c_u0, c_u128 });
                    for (0..4) |r| {
                        const acc_out = a.id();
                        try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc[r][nt] });
                        acc[r][nt] = acc_out;
                    }
                }
            }
            // Protects the next slab's staging — and, after the last slab,
            // the S-tile stores into the same array below.
            try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });
        }
    } else {
        // B (K) base for direct global fragment loads: kv block + col_w.
        const b_base = a.id();
        try a.op(128, &.{ t_u32, b_base, kvmul, col_w });
        for (0..hd / 16) |ks| {
            var a_off = a_base;
            if (ks > 0) {
                const ao = a.id();
                try a.op(128, &.{ t_u32, ao, a_base, c_k16[ks] });
                a_off = ao;
            }
            var ma: [4]u32 = undefined;
            var ao_cur = a_off;
            for (0..4) |r| {
                if (r > 0) {
                    const a2 = a.id();
                    try a.op(128, &.{ t_u32, a2, ao_cur, a_row16 });
                    ao_cur = a2;
                }
                const aptr = a.id();
                try a.op(65, &.{ t_ptr_sq_f16, aptr, v_q, c_u0, ao_cur });
                ma[r] = a.id();
                try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u0, p_qstride });
            }
            // B row block for this k-step starts at b_base + ks*16*s_stride.
            const krowmul = a.id();
            try a.op(132, &.{ t_u32, krowmul, c_k16[ks], p_sstride });
            const b_ks = a.id();
            try a.op(128, &.{ t_u32, b_ks, b_base, krowmul });
            for (0..4) |nt| {
                var b_off = b_ks;
                if (nt > 0) {
                    const bo = a.id();
                    try a.op(128, &.{ t_u32, bo, b_ks, c_k16[nt] });
                    b_off = bo;
                }
                const bptr = a.id();
                try a.op(65, &.{ t_ptr_sk_f16, bptr, v_k, c_u0, b_off });
                const mb = a.id();
                try a.op(4457, &.{ t_mat_b, mb, bptr, c_u0, p_sstride });
                for (0..4) |r| {
                    const acc_out = a.id();
                    try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc[r][nt] });
                    acc[r][nt] = acc_out;
                }
            }
        }
    }

    // Stage the warp's 4x4 f16 tiles into the bounce slab, then copy the
    // whole 128x128 tile out as coalesced u32 words.
    const wbase0 = a.id();
    try a.op(132, &.{ t_u32, wbase0, wm64, c_u128 });
    const wbase = a.id();
    try a.op(128, &.{ t_u32, wbase, wbase0, wn64 });
    for (0..4) |r| {
        for (0..4) |nt| {
            const hacc = a.id();
            try a.op(115, &.{ t_mat_h, hacc, acc[r][nt] }); // OpFConvert
            const sidx = a.id();
            try a.op(128, &.{ t_u32, sidx, wbase, c_st[r][nt] });
            const sptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, sptr, v_ssh, sidx });
            try a.op(4458, &.{ sptr, hacc, c_u0, c_u128 });
        }
    }
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 }); // OpControlBarrier
    // 8192 words for the 128x128 tile -> 64 per thread, 2 rows apart.
    var e = e0;
    var gw = gword0;
    for (0..64) |i| {
        if (i > 0) {
            const en = a.id();
            try a.op(128, &.{ t_u32, en, e, c_u256 });
            e = en;
            const gn = a.id();
            try a.op(128, &.{ t_u32, gn, gw, p_sstride });
            gw = gn;
        }
        const p0 = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, p0, v_ssh, e });
        const h0 = a.id();
        try a.op(61, &.{ t_f16, h0, p0 });
        const e1 = a.id();
        try a.op(128, &.{ t_u32, e1, e, c_u1 });
        const p1 = a.id();
        try a.op(65, &.{ t_ptr_wg_f16, p1, v_ssh, e1 });
        const h1 = a.id();
        try a.op(61, &.{ t_f16, h1, p1 });
        const pair = a.id();
        try a.op(80, &.{ t_v2f16, pair, h0, h1 }); // OpCompositeConstruct
        const word = a.id();
        try a.op(124, &.{ t_u32, word, pair }); // OpBitcast
        const gptr = a.id();
        try a.op(65, &.{ t_ptr_ss4_u32, gptr, v_s4, c_u0, gw });
        try a.op(62, &.{ gptr, word });
    }
    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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
    // Q staged resident in shared vs. cooperative-loaded from global per j
    // block: resident costs 32 KB shared (2 workgroups/SM instead of 6) and
    // measured SLOWER — each workgroup rereads the same 32 KB of Q, so L1
    // serves it either way, and the occupancy matters more.
    const STAGE_Q = false;
    const Q_SH: u32 = if (STAGE_Q) 16384 else 0; // s_sh region base
    const K_SH: u32 = Q_SH + 8192; // k_sh region base (stage_k)

    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    const main_fn = a.id();
    const ext_glsl = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_f16 = a.id();
    const t_f32 = a.id();
    const t_bool = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f16k = a.id();
    const t_arr_f16q = a.id();
    const t_arr_f32o = a.id();
    const t_arr_f16v = a.id();
    const t_sk = a.id();
    const t_sq = a.id();
    const t_so = a.id();
    const t_sv = a.id();
    const t_ptr_sk = a.id();
    const t_ptr_sq = a.id();
    const t_ptr_so = a.id();
    const t_ptr_sv = a.id();
    const v_k = a.id();
    const v_q = a.id();
    const v_o = a.id();
    const v_v = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_g_f16 = a.id(); // storage-buffer f16 element (K/Q/V share)
    const t_ptr_o_f32 = a.id();
    // workgroup slab: q_sh [128][128] f16 at 0, s_sh [64 j][128 q] at 16384
    const c_wsh_len = a.id();
    const t_wsh = a.id();
    const t_ptr_wg_wsh = a.id();
    const v_wsh = a.id();
    const t_ptr_wg_f16 = a.id();

    const c_u0 = a.id();
    const c_u1 = a.id();
    const c_u2 = a.id();
    const c_u3 = a.id();
    const c_u4 = a.id();
    const c_u5 = a.id();
    const c_u6 = a.id();
    const c_u7 = a.id();
    const c_u16 = a.id();
    const c_u32c = a.id();
    const c_u64 = a.id();
    const c_u128 = a.id();
    const c_u264 = a.id();
    const c_scope_sub = a.id();
    const c_scope_wg = a.id();
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const t_mat_h = a.id();
    const c_f32_0 = a.id();
    const c_f32_1 = a.id();
    const c_f32_ninf = a.id(); // -3.4e38 sentinel (matches the softmax kernels)
    const c_f16_0 = a.id();
    const c_acc0 = a.id();

    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1});
    try a.op(17, &.{9});
    try a.op(17, &.{5345});
    try a.op(17, &.{6022});
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_16bit_storage");
    try a.opStr(11, &.{ext_glsl}, "GLSL.std.450");
    try a.op(14, &.{ 0, 3 });
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_k, v_q, v_o, v_v, v_push, v_wsh });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, 4, 1 }); // LocalSize 32 4 1

    try a.op(71, &.{ gid_var, 11, 26 }); // WorkgroupId
    try a.op(71, &.{ lid_var, 11, 27 }); // LocalInvocationId
    inline for (.{ t_arr_f16k, t_arr_f16q, t_arr_f16v }) |t| {
        try a.op(71, &.{ t, 6, 2 });
    }
    try a.op(71, &.{ t_arr_f32o, 6, 4 });
    inline for (.{ t_sk, t_sq, t_so, t_sv }) |t| {
        try a.op(71, &.{ t, 2 });
        try a.op(72, &.{ t, 0, 35, 0 });
    }
    try a.op(71, &.{ t_push, 2 });
    inline for (0..8) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    inline for (.{ v_k, v_q, v_o, v_v }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 });
        try a.op(71, &.{ v, 33, @intCast(binding) });
    }

    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(20, &.{t_bool});
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f16k, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_f16q, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_f32o, t_f32, c_arrlen });
    try a.op(28, &.{ t_arr_f16v, t_f16, c_arrlen });
    try a.op(30, &.{ t_sk, t_arr_f16k });
    try a.op(30, &.{ t_sq, t_arr_f16q });
    try a.op(30, &.{ t_so, t_arr_f32o });
    try a.op(30, &.{ t_sv, t_arr_f16v });
    try a.op(32, &.{ t_ptr_sk, 12, t_sk });
    try a.op(32, &.{ t_ptr_sq, 12, t_sq });
    try a.op(32, &.{ t_ptr_so, 12, t_so });
    try a.op(32, &.{ t_ptr_sv, 12, t_sv });
    try a.op(59, &.{ t_ptr_sk, v_k, 12 });
    try a.op(59, &.{ t_ptr_sq, v_q, 12 });
    try a.op(59, &.{ t_ptr_so, v_o, 12 });
    try a.op(59, &.{ t_ptr_sv, v_v, 12 });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push });
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_g_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_o_f32, 12, t_f32 });

    // Workgroup slab: optional q_sh 128x128 f16, then s_sh 64x128 f16,
    // then (stage_k) k_sh 128 hd x 64 j f16, k-major with row stride 64.
    try a.op(43, &.{ t_u32, c_wsh_len, Q_SH + 8192 + @as(u32, if (stage_k) 8192 else 0) });
    try a.op(28, &.{ t_wsh, t_f16, c_wsh_len });
    try a.op(32, &.{ t_ptr_wg_wsh, 4, t_wsh });
    try a.op(59, &.{ t_ptr_wg_wsh, v_wsh, 4 });
    try a.op(32, &.{ t_ptr_wg_f16, 4, t_f16 });

    try a.op(43, &.{ t_u32, c_u0, 0 });
    try a.op(43, &.{ t_u32, c_u1, 1 });
    try a.op(43, &.{ t_u32, c_u2, 2 });
    try a.op(43, &.{ t_u32, c_u3, 3 });
    try a.op(43, &.{ t_u32, c_u4, 4 });
    try a.op(43, &.{ t_u32, c_u5, 5 });
    try a.op(43, &.{ t_u32, c_u6, 6 });
    try a.op(43, &.{ t_u32, c_u7, 7 });
    try a.op(43, &.{ t_u32, c_u16, 16 });
    try a.op(43, &.{ t_u32, c_u32c, 32 });
    try a.op(43, &.{ t_u32, c_u64, 64 });
    try a.op(43, &.{ t_u32, c_u128, 128 });
    try a.op(43, &.{ t_u32, c_u264, 0x108 });
    try a.op(43, &.{ t_u32, c_scope_sub, 3 });
    try a.op(43, &.{ t_u32, c_scope_wg, 2 });

    try a.op(4456, &.{ t_mat_a, t_f16, c_scope_sub, c_u16, c_u16, c_u0 });
    try a.op(4456, &.{ t_mat_b, t_f16, c_scope_sub, c_u16, c_u16, c_u1 });
    try a.op(4456, &.{ t_mat_c, t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(4456, &.{ t_mat_h, t_f16, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(43, &.{ t_f32, c_f32_0, 0 });
    try a.op(43, &.{ t_f32, c_f32_1, 0x3F800000 });
    try a.op(43, &.{ t_f32, c_f32_ninf, @as(u32, @bitCast(@as(f32, -3.4e38))) });
    const c_f32_minf = a.id(); // true -inf: masks padded j (exp -> exactly 0)
    try a.op(43, &.{ t_f32, c_f32_minf, 0xFF800000 });
    try a.op(43, &.{ t_f16, c_f16_0, 0 });
    try a.op(44, &.{ t_mat_c, c_acc0, c_f32_0 });

    // i*16 offsets (k-steps, tile columns).
    var c_k16: [8]u32 = undefined;
    c_k16[0] = c_u0;
    c_k16[1] = c_u16;
    c_k16[2] = c_u32c;
    for (3..8) |i| {
        c_k16[i] = a.id();
        try a.op(43, &.{ t_u32, c_k16[i], @intCast(i * 16) });
    }
    // Per-column s_sh offsets col*128 (col-major tile rows) — also reused as
    // "col << 7" values for the validity compare against limit*128.
    var c_col: [64]u32 = undefined;
    c_col[0] = c_u0;
    c_col[1] = c_u128;
    for (2..64) |i| {
        c_col[i] = a.id();
        try a.op(43, &.{ t_u32, c_col[i], @intCast(i * 128) });
    }
    // s_sh tile bases for the col-major store/load: Q_SH + (ct*16)*128.
    // Values may collide with c_col entries when Q_SH == 0.
    const c_ssh0 = if (Q_SH == 0) c_u0 else blk: {
        const c = a.id();
        try a.op(43, &.{ t_u32, c, Q_SH });
        break :blk c;
    };
    var c_sct: [4]u32 = undefined;
    c_sct[0] = c_ssh0;
    for (1..4) |ct| {
        const val: u32 = Q_SH + @as(u32, @intCast(ct)) * 16 * 128;
        c_sct[ct] = if (Q_SH == 0) c_col[val / 128] else blk: {
            const c = a.id();
            try a.op(43, &.{ t_u32, c, val });
            break :blk c;
        };
    }
    // k_sh per-k-step tile bases (K_SH + ks*16*64) and the copy's j mask.
    var c_kst: [8]u32 = undefined;
    var c_u63: u32 = undefined;
    if (stage_k) {
        for (0..8) |ks| {
            c_kst[ks] = a.id();
            try a.op(43, &.{ t_u32, c_kst[ks], K_SH + @as(u32, @intCast(ks)) * 1024 });
        }
        c_u63 = a.id();
        try a.op(43, &.{ t_u32, c_u63, 63 });
    }

    // --- function ---
    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_r = a.id();
    const zidx = a.id();
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    try a.op(81, &.{ t_u32, zidx, gidv, 2 });
    const row0 = a.id();
    try a.op(132, &.{ t_u32, row0, tile_r, c_u128 });

    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const lx = a.id();
    const ly = a.id();
    try a.op(81, &.{ t_u32, lx, lidv, 0 });
    try a.op(81, &.{ t_u32, ly, lidv, 1 });
    const flat = a.id();
    const lymul = a.id();
    try a.op(132, &.{ t_u32, lymul, ly, c_u32c });
    try a.op(128, &.{ t_u32, flat, lymul, lx });

    var push_vals: [7]u32 = undefined;
    inline for (0..7) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c_u0,
            1 => c_u1,
            2 => c_u2,
            3 => c_u3,
            4 => c_u4,
            5 => c_u5,
            else => c_u6,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_qstride = push_vals[0];
    const p_sstride = push_vals[1];
    const p_headoff = push_vals[2];
    const p_group = push_vals[3];
    const p_vstride = push_vals[4];
    const p_mdoff = push_vals[5];
    const p_seq = push_vals[6];

    const head = a.id();
    try a.op(128, &.{ t_u32, head, p_headoff, zidx });
    const kvh = a.id();
    try a.op(134, &.{ t_u32, kvh, head, p_group }); // UDiv
    const headmul = a.id();
    try a.op(132, &.{ t_u32, headmul, head, c_u128 });
    // K head base: kv * (128 * s_stride).
    const khead = a.id();
    try a.op(196, &.{ t_u32, khead, p_sstride, c_u7 }); // s_stride << 7
    const kbase = a.id();
    try a.op(132, &.{ t_u32, kbase, kvh, khead });
    const kvmul_v = a.id();
    try a.op(132, &.{ t_u32, kvmul_v, kvh, c_u128 });

    // Per-warp Q row-block bases: warp ly owns q rows [ly*32, ly*32+32).
    // With STAGE_Q the rows copy into shared once (coop load global -> coop
    // store shared); otherwise the S compute cooperative-loads Q straight
    // from global each j block (L1 keeps the workgroup's 32 KB hot).
    const ly32 = a.id();
    try a.op(132, &.{ t_u32, ly32, ly, c_u32c });
    const qrow_g0 = a.id();
    try a.op(128, &.{ t_u32, qrow_g0, row0, ly32 });
    var qg_base: [2]u32 = undefined; // global f16 index of the row block
    for (0..2) |rw| {
        const grow = if (rw == 0) qrow_g0 else blk: {
            const g = a.id();
            try a.op(128, &.{ t_u32, g, qrow_g0, c_u16 });
            break :blk g;
        };
        const gmul = a.id();
        try a.op(132, &.{ t_u32, gmul, grow, p_qstride });
        qg_base[rw] = a.id();
        try a.op(128, &.{ t_u32, qg_base[rw], gmul, headmul });
    }
    if (STAGE_Q) {
        for (0..2) |rw| {
            const smul = a.id();
            try a.op(128, &.{ t_u32, smul, ly32, if (rw == 0) c_u0 else c_u16 });
            const sbase = a.id();
            try a.op(132, &.{ t_u32, sbase, smul, c_u128 });
            for (0..8) |kt| {
                const goff = a.id();
                try a.op(128, &.{ t_u32, goff, qg_base[rw], c_k16[kt] });
                const gptr = a.id();
                try a.op(65, &.{ t_ptr_g_f16, gptr, v_q, c_u0, goff });
                const mq = a.id();
                try a.op(4457, &.{ t_mat_a, mq, gptr, c_u0, p_qstride });
                const soff = a.id();
                try a.op(128, &.{ t_u32, soff, sbase, c_k16[kt] });
                const sptr = a.id();
                try a.op(65, &.{ t_ptr_wg_f16, sptr, v_wsh, soff });
                try a.op(4458, &.{ sptr, mq, c_u0, c_u128 });
            }
        }
    }

    // Per-thread row state (thread owns q row `flat` of the tile).
    // md phase: running online (m, d), loop-carried. out phase: m and 1/d
    // loaded once from the MD table.
    const mdrow = a.id(); // (z*s_stride + row0 + flat)*2 + mdoff
    {
        const zr = a.id();
        try a.op(132, &.{ t_u32, zr, zidx, p_sstride });
        const qr = a.id();
        try a.op(128, &.{ t_u32, qr, zr, row0 });
        const qr2 = a.id();
        try a.op(128, &.{ t_u32, qr2, qr, flat });
        const dbl = a.id();
        try a.op(196, &.{ t_u32, dbl, qr2, c_u1 });
        try a.op(128, &.{ t_u32, mdrow, dbl, p_mdoff });
    }
    var m_hoist: u32 = undefined;
    var invd_hoist: u32 = undefined;
    if (out_phase) {
        const mptr = a.id();
        try a.op(65, &.{ t_ptr_o_f32, mptr, v_o, c_u0, mdrow });
        m_hoist = a.id();
        try a.op(61, &.{ t_f32, m_hoist, mptr });
        const di = a.id();
        try a.op(128, &.{ t_u32, di, mdrow, c_u1 });
        const dptr = a.id();
        try a.op(65, &.{ t_ptr_o_f32, dptr, v_o, c_u0, di });
        invd_hoist = a.id();
        try a.op(61, &.{ t_f32, invd_hoist, dptr });
    }
    // Thread's s_sh column base (col-major tile: element (j, q) at
    // 16384 + j*128 + q).
    const s_thread0 = a.id();
    try a.op(128, &.{ t_u32, s_thread0, c_ssh0, flat });

    // K-staging copy invariants: thread `flat` owns k_sh column flat&63
    // and rows flat>>6 + 2i (consecutive lanes read consecutive K columns
    // — coalesced). Global element = kbase + j0 + (flat&63) +
    // (flat>>6 + 2i)*s_stride; shared element = K_SH + flat + i*128.
    var k_inv: u32 = undefined;
    var k_e0: u32 = undefined;
    var c2s: u32 = undefined;
    if (stage_k) {
        const kj = a.id();
        try a.op(199, &.{ t_u32, kj, flat, c_u63 });
        const kr0 = a.id();
        try a.op(194, &.{ t_u32, kr0, flat, c_u6 });
        const gj = a.id();
        try a.op(128, &.{ t_u32, gj, kbase, kj });
        const gr = a.id();
        try a.op(132, &.{ t_u32, gr, kr0, p_sstride });
        k_inv = a.id();
        try a.op(128, &.{ t_u32, k_inv, gj, gr });
        k_e0 = a.id();
        try a.op(128, &.{ t_u32, k_e0, c_kst[0], flat });
        c2s = a.id();
        try a.op(128, &.{ t_u32, c2s, p_sstride, p_sstride });
    }

    // Warp tiling for S compute: warp ly owns q rows [ly*32, ly*32+32).
    // Warp tiling for P@V (out phase): 2x2 grid, 64q x 64c per warp.
    const warp_m = a.id();
    try a.op(199, &.{ t_u32, warp_m, ly, c_u1 });
    const warp_n = a.id();
    try a.op(194, &.{ t_u32, warp_n, ly, c_u1 });
    const wm64 = a.id();
    try a.op(132, &.{ t_u32, wm64, warp_m, c_u64 });
    const wn64 = a.id();
    try a.op(132, &.{ t_u32, wn64, warp_n, c_u64 });
    // Hoisted per-warp offsets: q/out row-block starts and the V column base.
    var wmr: [4]u32 = undefined;
    wmr[0] = wm64;
    for (1..4) |r| {
        wmr[r] = a.id();
        try a.op(128, &.{ t_u32, wmr[r], wm64, c_k16[r] });
    }
    const vcol_base = a.id();
    try a.op(128, &.{ t_u32, vcol_base, kvmul_v, wn64 });

    // Loop preamble: loop-carried values.
    const lb_head = a.id();
    const lb_cond = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();
    const j0n = a.id();
    // md phase: m/d next values; out phase: 16 acc next values.
    const mn_next = a.id();
    const dn_next = a.id();
    var acc_next: [4][4]u32 = undefined;
    for (&acc_next) |*row| for (row) |*v| {
        v.* = a.id();
    };

    try a.op(249, &.{lb_head});
    try a.op(248, &.{lb_head});
    const j0v = a.id();
    try a.op(245, &.{ t_u32, j0v, c_u0, lb_entry, j0n, lb_cont }); // OpPhi
    var m_phi: u32 = undefined;
    var d_phi: u32 = undefined;
    var acc_phi: [4][4]u32 = undefined;
    if (!out_phase) {
        m_phi = a.id();
        try a.op(245, &.{ t_f32, m_phi, c_f32_ninf, lb_entry, mn_next, lb_cont });
        d_phi = a.id();
        try a.op(245, &.{ t_f32, d_phi, c_f32_0, lb_entry, dn_next, lb_cont });
    } else {
        for (&acc_phi, acc_next) |*prow, nrow| {
            for (prow, nrow) |*ap, an| {
                ap.* = a.id();
                try a.op(245, &.{ t_mat_c, ap.*, c_acc0, lb_entry, an, lb_cont });
            }
        }
    }
    try a.op(246, &.{ lb_merge, lb_cont, 0 });
    try a.op(249, &.{lb_cond});
    try a.op(248, &.{lb_cond});
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, j0v, p_sstride });
    try a.op(250, &.{ cmp, lb_body, lb_merge });

    try a.op(248, &.{lb_body});
    // Barrier: Q staging on the first pass; s_sh reuse on later passes.
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });

    if (stage_k) {
        // Stage this j-block's K tile into k_sh: all four warps consume
        // IDENTICAL K fragments per (ks, ct), so one shared copy replaces
        // 4x-redundant global fragment loads. 64 f16 per thread.
        var g = a.id();
        try a.op(128, &.{ t_u32, g, k_inv, j0v });
        var e = k_e0;
        for (0..64) |i| {
            if (i > 0) {
                const gn = a.id();
                try a.op(128, &.{ t_u32, gn, g, c2s });
                g = gn;
                const en = a.id();
                try a.op(128, &.{ t_u32, en, e, c_u128 });
                e = en;
            }
            const gp = a.id();
            try a.op(65, &.{ t_ptr_g_f16, gp, v_k, c_u0, g });
            const hv = a.id();
            try a.op(61, &.{ t_f16, hv, gp });
            const sp = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, sp, v_wsh, e });
            try a.op(62, &.{ sp, hv });
        }
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });
    }

    // S block: warp ly computes q rows [ly32, ly32+32) x 64 j into s_sh
    // (column-major). 2 row blocks x 4 col tiles, 8 k-steps each.
    for (0..2) |rw| {
        const q_shbase = a.id(); // (ly32 + rw*16) * 128 (STAGE_Q layout)
        {
            const qr = a.id();
            try a.op(128, &.{ t_u32, qr, ly32, if (rw == 0) c_u0 else c_u16 });
            try a.op(132, &.{ t_u32, q_shbase, qr, c_u128 });
        }
        const s_qoff = a.id(); // ly32 + rw*16 (column offset in s_sh)
        try a.op(128, &.{ t_u32, s_qoff, ly32, if (rw == 0) c_u0 else c_u16 });
        for (0..4) |ct| {
            var acc: u32 = c_acc0;
            const jg = a.id(); // global j of this tile: j0 + ct*16
            try a.op(128, &.{ t_u32, jg, j0v, c_k16[ct] });
            for (0..8) |ks| {
                const qoff = a.id();
                try a.op(128, &.{ t_u32, qoff, if (STAGE_Q) q_shbase else qg_base[rw], c_k16[ks] });
                const qptr = a.id();
                if (STAGE_Q) {
                    try a.op(65, &.{ t_ptr_wg_f16, qptr, v_wsh, qoff });
                } else {
                    try a.op(65, &.{ t_ptr_g_f16, qptr, v_q, c_u0, qoff });
                }
                const ma = a.id();
                try a.op(4457, &.{ t_mat_a, ma, qptr, c_u0, if (STAGE_Q) c_u128 else p_qstride });
                var mb: u32 = undefined;
                if (stage_k) {
                    // k_sh tile (ks*16, ct*16), row stride 64.
                    const koff = a.id();
                    try a.op(128, &.{ t_u32, koff, c_kst[ks], c_k16[ct] });
                    const kptr = a.id();
                    try a.op(65, &.{ t_ptr_wg_f16, kptr, v_wsh, koff });
                    mb = a.id();
                    try a.op(4457, &.{ t_mat_b, mb, kptr, c_u0, c_u64 });
                } else {
                    const krow = a.id();
                    try a.op(132, &.{ t_u32, krow, c_k16[ks], p_sstride });
                    const koff0 = a.id();
                    try a.op(128, &.{ t_u32, koff0, kbase, krow });
                    const koff = a.id();
                    try a.op(128, &.{ t_u32, koff, koff0, jg });
                    const kptr = a.id();
                    try a.op(65, &.{ t_ptr_g_f16, kptr, v_k, c_u0, koff });
                    mb = a.id();
                    try a.op(4457, &.{ t_mat_b, mb, kptr, c_u0, p_sstride });
                }
                const acc_out = a.id();
                try a.op(4459, &.{ t_mat_c, acc_out, ma, mb, acc });
                acc = acc_out;
            }
            const hacc = a.id();
            try a.op(115, &.{ t_mat_h, hacc, acc }); // FConvert
            const soff = a.id();
            try a.op(128, &.{ t_u32, soff, c_sct[ct], s_qoff });
            const sptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, sptr, v_wsh, soff });
            try a.op(4458, &.{ sptr, hacc, c_u1, c_u128 }); // ColumnMajor
        }
    }
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });

    // Validity horizon for this block: columns col with col*128 <
    // (seq - j0)*128 are real j positions; the rest mask to -inf / 0.
    const over = a.id();
    try a.op(176, &.{ t_bool, over, j0v, p_seq }); // j0 < seq
    const rem = a.id();
    try a.op(130, &.{ t_u32, rem, p_seq, j0v }); // ISub
    const limit = a.id();
    try a.op(169, &.{ t_u32, limit, over, rem, c_u0 }); // Select
    const limit128 = a.id();
    try a.op(196, &.{ t_u32, limit128, limit, c_u7 });

    if (!out_phase) {
        // Online (m, d) over the 64 shared columns of the thread's row.
        var m_cur = m_phi;
        var d_cur = d_phi;
        for (0..64) |col| {
            var eidx = s_thread0;
            if (col > 0) {
                const ei = a.id();
                try a.op(128, &.{ t_u32, ei, s_thread0, c_col[col] });
                eidx = ei;
            }
            const sptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, sptr, v_wsh, eidx });
            const hval = a.id();
            try a.op(61, &.{ t_f16, hval, sptr });
            const s32 = a.id();
            try a.op(115, &.{ t_f32, s32, hval }); // FConvert
            const valid = a.id();
            try a.op(176, &.{ t_bool, valid, c_col[col], limit128 });
            const s_eff = a.id();
            try a.op(169, &.{ t_f32, s_eff, valid, s32, c_f32_minf });
            const m_new = if (col == 63) mn_next else a.id();
            try a.op(12, &.{ t_f32, m_new, ext_glsl, 40, m_cur, s_eff }); // FMax
            const dm = a.id();
            try a.op(131, &.{ t_f32, dm, m_cur, m_new }); // FSub
            const corr = a.id();
            try a.op(12, &.{ t_f32, corr, ext_glsl, 27, dm }); // Exp
            const ds = a.id();
            try a.op(131, &.{ t_f32, ds, s_eff, m_new });
            const p = a.id();
            try a.op(12, &.{ t_f32, p, ext_glsl, 27, ds });
            const dscaled = a.id();
            try a.op(133, &.{ t_f32, dscaled, d_cur, corr }); // FMul
            const d_new = if (col == 63) dn_next else a.id();
            try a.op(129, &.{ t_f32, d_new, dscaled, p }); // FAdd
            m_cur = m_new;
            d_cur = d_new;
        }
    } else {
        // Transform the shared S block in place to P = exp(S - m) * invd
        // (padded j forced to zero), then accumulate P@V.
        for (0..64) |col| {
            var eidx = s_thread0;
            if (col > 0) {
                const ei = a.id();
                try a.op(128, &.{ t_u32, ei, s_thread0, c_col[col] });
                eidx = ei;
            }
            const sptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, sptr, v_wsh, eidx });
            const hval = a.id();
            try a.op(61, &.{ t_f16, hval, sptr });
            const s32 = a.id();
            try a.op(115, &.{ t_f32, s32, hval });
            const ds = a.id();
            try a.op(131, &.{ t_f32, ds, s32, m_hoist });
            const e = a.id();
            try a.op(12, &.{ t_f32, e, ext_glsl, 27, ds });
            const p = a.id();
            try a.op(133, &.{ t_f32, p, e, invd_hoist });
            const valid = a.id();
            try a.op(176, &.{ t_bool, valid, c_col[col], limit128 });
            const pm = a.id();
            try a.op(169, &.{ t_f32, pm, valid, p, c_f32_0 });
            const hp = a.id();
            try a.op(115, &.{ t_f16, hp, pm });
            try a.op(62, &.{ sptr, hp });
        }
        try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });

        var acc_cur = acc_phi;
        for (0..4) |kk| {
            var ma: [4]u32 = undefined;
            for (0..4) |r| {
                const aoff = a.id();
                try a.op(128, &.{ t_u32, aoff, c_sct[kk], wmr[r] });
                const aptr = a.id();
                try a.op(65, &.{ t_ptr_wg_f16, aptr, v_wsh, aoff });
                ma[r] = a.id();
                try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u1, c_u128 }); // ColMajor
            }
            const jg2 = a.id();
            try a.op(128, &.{ t_u32, jg2, j0v, c_k16[kk] });
            const vrow = a.id();
            try a.op(132, &.{ t_u32, vrow, jg2, p_vstride });
            const vb0 = a.id();
            try a.op(128, &.{ t_u32, vb0, vrow, vcol_base });
            for (0..4) |ct| {
                const voff = a.id();
                try a.op(128, &.{ t_u32, voff, vb0, c_k16[ct] });
                const vptr = a.id();
                try a.op(65, &.{ t_ptr_g_f16, vptr, v_v, c_u0, voff });
                const mb = a.id();
                try a.op(4457, &.{ t_mat_b, mb, vptr, c_u0, p_vstride });
                for (0..4) |r| {
                    const acc_out = if (kk == 3) acc_next[r][ct] else a.id();
                    try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc_cur[r][ct] });
                    acc_cur[r][ct] = acc_out;
                }
            }
        }
    }
    try a.op(249, &.{lb_cont});

    try a.op(248, &.{lb_cont});
    try a.op(128, &.{ t_u32, j0n, j0v, c_u64 });
    try a.op(249, &.{lb_head});

    try a.op(248, &.{lb_merge});
    if (!out_phase) {
        const invd = a.id();
        try a.op(136, &.{ t_f32, invd, c_f32_1, d_phi }); // FDiv
        const mptr = a.id();
        try a.op(65, &.{ t_ptr_o_f32, mptr, v_o, c_u0, mdrow });
        try a.op(62, &.{ mptr, m_phi });
        const di = a.id();
        try a.op(128, &.{ t_u32, di, mdrow, c_u1 });
        const dptr = a.id();
        try a.op(65, &.{ t_ptr_o_f32, dptr, v_o, c_u0, di });
        try a.op(62, &.{ dptr, invd });
    } else {
        for (0..4) |r| {
            const orow = a.id();
            try a.op(128, &.{ t_u32, orow, row0, wmr[r] });
            const omul = a.id();
            try a.op(132, &.{ t_u32, omul, orow, p_qstride });
            const ob = a.id();
            try a.op(128, &.{ t_u32, ob, omul, headmul });
            const ob2 = a.id();
            try a.op(128, &.{ t_u32, ob2, ob, wn64 });
            for (0..4) |ct| {
                const ooff = a.id();
                try a.op(128, &.{ t_u32, ooff, ob2, c_k16[ct] });
                const optr = a.id();
                try a.op(65, &.{ t_ptr_o_f32, optr, v_o, c_u0, ooff });
                try a.op(4458, &.{ optr, acc_phi[r][ct], c_u0, p_qstride });
            }
        }
    }
    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
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
    var a: Asm = .{ .gpa = gpa };
    defer a.words.deinit(gpa);

    const main_fn = a.id();
    const ext_glsl = a.id();
    const gid_var = a.id();
    const lid_var = a.id();
    const t_void = a.id();
    const t_fnvoid = a.id();
    const t_u32 = a.id();
    const t_f16 = a.id();
    const t_f32 = a.id();
    const t_v3u = a.id();
    const t_ptr_in_v3 = a.id();
    const c_arrlen = a.id();
    const t_arr_f16 = a.id();
    const t_arr_f32 = a.id();
    const t_arr_f32b = a.id();
    const t_arr_f32c = a.id();
    const t_ss = a.id(); // struct { [N]u32 } (S as packed f16 pairs)
    const t_sv = a.id(); // struct { [N]f16 } (V)
    const t_so = a.id(); // struct { [N]f32 } (OUT)
    const t_smd = a.id(); // struct { [N]f32 } (MD)
    const t_ptr_ss = a.id();
    const t_ptr_sv = a.id();
    const t_ptr_so = a.id();
    const t_ptr_smd = a.id();
    const v_s = a.id();
    const v_v = a.id();
    const v_o = a.id();
    const v_md = a.id();
    const t_push = a.id();
    const t_ptr_push = a.id();
    const v_push = a.id();
    const t_ptr_pc_u32 = a.id();
    const t_ptr_ss_f32 = a.id(); // u32 element pointer (f16 pair words)
    const t_v2f16 = a.id();
    const t_ptr_sv_f16 = a.id();
    const t_ptr_so_f32 = a.id();
    const t_ptr_smd_f32 = a.id();
    const t_bool = a.id();
    // workgroup P slab: [128 q][64 j] f16
    const c_psh_len = a.id();
    const t_psh = a.id();
    const t_ptr_wg_psh = a.id();
    const v_psh = a.id();
    const t_ptr_wg_f16 = a.id();

    const c_u0 = a.id();
    const c_u1 = a.id();
    const c_u2 = a.id();
    const c_u3 = a.id();
    const c_u4 = a.id();
    const c_u5 = a.id();
    const c_u6 = a.id();
    const c_u7 = a.id();
    const c_u31 = a.id();
    const c_u16 = a.id();
    const c_u32c = a.id();
    const c_u63 = a.id();
    const c_u64 = a.id();
    const c_u128 = a.id();
    const c_u1024 = a.id();
    const c_u264 = a.id();
    const c_scope_sub = a.id();
    const c_scope_wg = a.id();
    const t_mat_a = a.id();
    const t_mat_b = a.id();
    const t_mat_c = a.id();
    const c_f32_0 = a.id();
    const c_acc0 = a.id();

    try a.words.appendSlice(gpa, &.{ 0x0723_0203, 0x0001_0500, 0, 0, 0 });
    try a.op(17, &.{1});
    try a.op(17, &.{9});
    try a.op(17, &.{5345});
    try a.op(17, &.{6022});
    try a.opStr(10, &.{}, "SPV_KHR_cooperative_matrix");
    try a.opStr(10, &.{}, "SPV_KHR_vulkan_memory_model");
    try a.opStr(10, &.{}, "SPV_KHR_16bit_storage");
    try a.opStr(11, &.{ext_glsl}, "GLSL.std.450"); // OpExtInstImport
    try a.op(14, &.{ 0, 3 });
    {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, &.{ 5, main_fn });
        try buf.appendSlice(gpa, &.{ std.mem.bytesToValue(u32, "main"), 0 });
        try buf.appendSlice(gpa, &.{ gid_var, lid_var, v_s, v_v, v_o, v_md, v_push, v_psh });
        try a.op(15, buf.items);
    }
    try a.op(16, &.{ main_fn, 17, 32, 4, 1 }); // LocalSize 32 4 1

    try a.op(71, &.{ gid_var, 11, 26 }); // WorkgroupId
    try a.op(71, &.{ lid_var, 11, 27 }); // LocalInvocationId
    try a.op(71, &.{ t_arr_f16, 6, 2 });
    try a.op(71, &.{ t_arr_f32, 6, 4 });
    try a.op(71, &.{ t_arr_f32b, 6, 4 });
    try a.op(71, &.{ t_arr_f32c, 6, 4 });
    inline for (.{ t_ss, t_sv, t_so, t_smd }) |t| {
        try a.op(71, &.{ t, 2 });
        try a.op(72, &.{ t, 0, 35, 0 });
    }
    try a.op(71, &.{ t_push, 2 });
    inline for (0..8) |m| {
        try a.op(72, &.{ t_push, @intCast(m), 35, @intCast(m * 4) });
    }
    inline for (.{ v_s, v_v, v_o, v_md }, 0..) |v, binding| {
        try a.op(71, &.{ v, 34, 0 });
        try a.op(71, &.{ v, 33, @intCast(binding) });
    }

    try a.op(19, &.{t_void});
    try a.op(33, &.{ t_fnvoid, t_void });
    try a.op(21, &.{ t_u32, 32, 0 });
    try a.op(22, &.{ t_f16, 16 });
    try a.op(22, &.{ t_f32, 32 });
    try a.op(20, &.{t_bool});
    try a.op(23, &.{ t_v3u, t_u32, 3 });
    try a.op(32, &.{ t_ptr_in_v3, 1, t_v3u });
    try a.op(59, &.{ t_ptr_in_v3, gid_var, 1 });
    try a.op(59, &.{ t_ptr_in_v3, lid_var, 1 });

    try a.op(43, &.{ t_u32, c_arrlen, 1 << 28 });
    try a.op(28, &.{ t_arr_f16, t_f16, c_arrlen });
    try a.op(28, &.{ t_arr_f32, t_u32, c_arrlen }); // S words (f16 pairs)
    try a.op(28, &.{ t_arr_f32b, t_f32, c_arrlen });
    try a.op(28, &.{ t_arr_f32c, t_f32, c_arrlen });
    try a.op(23, &.{ t_v2f16, t_f16, 2 });
    try a.op(30, &.{ t_ss, t_arr_f32 });
    try a.op(30, &.{ t_sv, t_arr_f16 });
    try a.op(30, &.{ t_so, t_arr_f32b });
    try a.op(30, &.{ t_smd, t_arr_f32c });
    try a.op(32, &.{ t_ptr_ss, 12, t_ss });
    try a.op(32, &.{ t_ptr_sv, 12, t_sv });
    try a.op(32, &.{ t_ptr_so, 12, t_so });
    try a.op(32, &.{ t_ptr_smd, 12, t_smd });
    try a.op(59, &.{ t_ptr_ss, v_s, 12 });
    try a.op(59, &.{ t_ptr_sv, v_v, 12 });
    try a.op(59, &.{ t_ptr_so, v_o, 12 });
    try a.op(59, &.{ t_ptr_smd, v_md, 12 });
    try a.op(30, &.{ t_push, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32 });
    try a.op(32, &.{ t_ptr_push, 9, t_push });
    try a.op(59, &.{ t_ptr_push, v_push, 9 });
    try a.op(32, &.{ t_ptr_pc_u32, 9, t_u32 });
    try a.op(32, &.{ t_ptr_ss_f32, 12, t_u32 });
    try a.op(32, &.{ t_ptr_sv_f16, 12, t_f16 });
    try a.op(32, &.{ t_ptr_so_f32, 12, t_f32 });
    try a.op(32, &.{ t_ptr_smd_f32, 12, t_f32 });

    // Workgroup P slab (no layout decorations): [128][64] f16.
    try a.op(43, &.{ t_u32, c_psh_len, 8192 });
    try a.op(28, &.{ t_psh, t_f16, c_psh_len });
    try a.op(32, &.{ t_ptr_wg_psh, 4, t_psh });
    try a.op(59, &.{ t_ptr_wg_psh, v_psh, 4 });
    try a.op(32, &.{ t_ptr_wg_f16, 4, t_f16 });

    try a.op(43, &.{ t_u32, c_u0, 0 });
    try a.op(43, &.{ t_u32, c_u1, 1 });
    try a.op(43, &.{ t_u32, c_u2, 2 });
    try a.op(43, &.{ t_u32, c_u3, 3 });
    try a.op(43, &.{ t_u32, c_u4, 4 });
    try a.op(43, &.{ t_u32, c_u5, 5 });
    try a.op(43, &.{ t_u32, c_u6, 6 });
    try a.op(43, &.{ t_u32, c_u7, 7 });
    try a.op(43, &.{ t_u32, c_u31, 31 });
    try a.op(43, &.{ t_u32, c_u16, 16 });
    try a.op(43, &.{ t_u32, c_u32c, 32 });
    try a.op(43, &.{ t_u32, c_u63, 63 });
    try a.op(43, &.{ t_u32, c_u64, 64 });
    try a.op(43, &.{ t_u32, c_u128, 128 });
    try a.op(43, &.{ t_u32, c_u1024, 1024 });
    try a.op(43, &.{ t_u32, c_u264, 0x108 });
    try a.op(43, &.{ t_u32, c_scope_sub, 3 });
    try a.op(43, &.{ t_u32, c_scope_wg, 2 });

    try a.op(4456, &.{ t_mat_a, t_f16, c_scope_sub, c_u16, c_u16, c_u0 });
    try a.op(4456, &.{ t_mat_b, t_f16, c_scope_sub, c_u16, c_u16, c_u1 });
    try a.op(4456, &.{ t_mat_c, t_f32, c_scope_sub, c_u16, c_u16, c_u2 });
    try a.op(43, &.{ t_f32, c_f32_0, 0 });
    try a.op(44, &.{ t_mat_c, c_acc0, c_f32_0 });

    var c_k16: [4]u32 = undefined; // ks*16
    c_k16[0] = c_u0;
    c_k16[1] = c_u16;
    c_k16[2] = c_u32c;
    c_k16[3] = a.id();
    try a.op(43, &.{ t_u32, c_k16[3], 48 });
    var c_col16: [8]u32 = undefined; // nt*16
    c_col16[0] = c_u0;
    c_col16[1] = c_u16;
    c_col16[2] = c_u32c;
    c_col16[3] = c_k16[3];
    c_col16[4] = c_u64;
    for (5..8) |nt| {
        c_col16[nt] = a.id();
        try a.op(43, &.{ t_u32, c_col16[nt], @intCast(nt * 16) });
    }


    // --- function ---
    const lb_entry = a.id();
    try a.op(54, &.{ t_void, main_fn, 0, t_fnvoid });
    try a.op(248, &.{lb_entry});

    const gidv = a.id();
    try a.op(61, &.{ t_v3u, gidv, gid_var });
    const tile_r = a.id();
    const zidx = a.id();
    try a.op(81, &.{ t_u32, tile_r, gidv, 1 });
    try a.op(81, &.{ t_u32, zidx, gidv, 2 });
    const row0 = a.id();
    try a.op(132, &.{ t_u32, row0, tile_r, c_u128 });

    const lidv = a.id();
    try a.op(61, &.{ t_v3u, lidv, lid_var });
    const lx = a.id();
    const ly = a.id();
    try a.op(81, &.{ t_u32, lx, lidv, 0 });
    try a.op(81, &.{ t_u32, ly, lidv, 1 });
    const flat = a.id();
    const lymul = a.id();
    try a.op(132, &.{ t_u32, lymul, ly, c_u32c });
    try a.op(128, &.{ t_u32, flat, lymul, lx });

    var push_vals: [8]u32 = undefined;
    inline for (0..8) |m| {
        const pptr = a.id();
        const pval = a.id();
        const cidx = switch (m) {
            0 => c_u0,
            1 => c_u1,
            2 => c_u2,
            3 => c_u3,
            4 => c_u4,
            5 => c_u5,
            6 => c_u6,
            else => c_u7,
        };
        try a.op(65, &.{ t_ptr_pc_u32, pptr, v_push, cidx });
        try a.op(61, &.{ t_u32, pval, pptr });
        push_vals[m] = pval;
    }
    const p_sstride = push_vals[0];
    const p_splane = push_vals[1];
    const p_headoff = push_vals[2];
    const p_group = push_vals[3];
    const p_vstride = push_vals[4];
    const p_ostride = push_vals[5];
    const p_seq = push_vals[6]; // valid j count (f0 push word, u32 bits)
    const p_mdplane = push_vals[7]; // MD rows per plane (f1 push word)

    const head = a.id();
    try a.op(128, &.{ t_u32, head, p_headoff, zidx });
    const kvh = a.id();
    try a.op(134, &.{ t_u32, kvh, head, p_group }); // UDiv
    const headmul = a.id();
    try a.op(132, &.{ t_u32, headmul, head, c_u128 });
    const kvmul = a.id();
    try a.op(132, &.{ t_u32, kvmul, kvh, c_u128 });

    // Warp tiling: 2x2 grid of 64x64 tiles (out cols = head_dim = 128).
    const warp_m = a.id();
    try a.op(199, &.{ t_u32, warp_m, ly, c_u1 }); // ly & 1
    const warp_n = a.id();
    try a.op(194, &.{ t_u32, warp_n, ly, c_u1 }); // ly >> 1
    const wm64 = a.id();
    try a.op(132, &.{ t_u32, wm64, warp_m, c_u64 });
    const row_s = a.id();
    try a.op(128, &.{ t_u32, row_s, row0, wm64 });
    const wn64 = a.id();
    try a.op(132, &.{ t_u32, wn64, warp_n, c_u64 });

    // Loop-invariant staging indices: thread `flat`, element t (of 64) is
    // slab element e = flat + t*128, i.e. P row e/64 (q = row0 + row),
    // column e%64 (j = k0 + column). Per element the row's {m, 1/d} pair
    // is loaded from MD at (z*s_stride + q)*2.
    const s_zbase = a.id();
    try a.op(132, &.{ t_u32, s_zbase, zidx, p_splane });
    const md_zrow = a.id();
    try a.op(132, &.{ t_u32, md_zrow, zidx, p_mdplane });
    // Per thread: 32 S words, each an f16 pair covering slab elements
    // 2v/2v+1 with v = flat + t*128 (slab row v/32, columns (v%32)*2).
    var srow_t: [32]u32 = undefined; // (row0 + v/32) * s_stride + z*splane
    var mdi_t: [32]u32 = undefined; // ((z*sstride) + row0 + v/32) * 2
    var jcol_t: [32]u32 = undefined; // (v % 32) * 2
    var e_t: [32]u32 = undefined; // slab element base = v*2
    var v_prev: u32 = 0;
    for (0..32) |t| {
        var v = flat;
        if (t > 0) {
            const vn = a.id();
            try a.op(128, &.{ t_u32, vn, v_prev, c_u128 });
            v = vn;
        }
        v_prev = v;
        e_t[t] = a.id();
        try a.op(196, &.{ t_u32, e_t[t], v, c_u1 });
        const erow = a.id();
        try a.op(194, &.{ t_u32, erow, v, c_u5 }); // v / 32
        const vmod = a.id();
        try a.op(199, &.{ t_u32, vmod, v, c_u31 });
        jcol_t[t] = a.id();
        try a.op(196, &.{ t_u32, jcol_t[t], vmod, c_u1 }); // * 2
        const qrow = a.id();
        try a.op(128, &.{ t_u32, qrow, row0, erow });
        const qmul = a.id();
        try a.op(132, &.{ t_u32, qmul, qrow, p_sstride });
        srow_t[t] = a.id();
        try a.op(128, &.{ t_u32, srow_t[t], s_zbase, qmul });
        const mdrow = a.id();
        try a.op(128, &.{ t_u32, mdrow, md_zrow, qrow });
        mdi_t[t] = a.id();
        try a.op(196, &.{ t_u32, mdi_t[t], mdrow, c_u1 }); // ShiftLeftLogical: *2
    }

    const lb_head = a.id();
    const lb_cond = a.id();
    const lb_body = a.id();
    const lb_cont = a.id();
    const lb_merge = a.id();
    const k0n = a.id();
    var acc_next: [4][4]u32 = undefined;
    for (&acc_next) |*row| for (row) |*v| {
        v.* = a.id();
    };

    try a.op(249, &.{lb_head});
    try a.op(248, &.{lb_head});
    const k0v = a.id();
    try a.op(245, &.{ t_u32, k0v, c_u0, lb_entry, k0n, lb_cont }); // OpPhi
    var acc_phi: [4][4]u32 = undefined;
    for (&acc_phi, acc_next) |*prow, nrow| {
        for (prow, nrow) |*ap, an| {
            ap.* = a.id();
            try a.op(245, &.{ t_mat_c, ap.*, c_acc0, lb_entry, an, lb_cont });
        }
    }
    try a.op(246, &.{ lb_merge, lb_cont, 0 });
    try a.op(249, &.{lb_cond});
    try a.op(248, &.{lb_cond});
    const cmp = a.id();
    try a.op(176, &.{ t_bool, cmp, k0v, p_sstride }); // j count == s_stride
    try a.op(250, &.{ cmp, lb_body, lb_merge });

    try a.op(248, &.{lb_body});
    // Stage P = exp(S - m) * invd into the slab, 32 f16-pair words per
    // thread (S is stored half-precision by the scores kernel).
    for (0..32) |t| {
        const soff = a.id();
        try a.op(128, &.{ t_u32, soff, srow_t[t], k0v });
        const soff2 = a.id();
        try a.op(128, &.{ t_u32, soff2, soff, jcol_t[t] });
        const widx = a.id();
        try a.op(194, &.{ t_u32, widx, soff2, c_u1 }); // element -> word
        const sptr = a.id();
        try a.op(65, &.{ t_ptr_ss_f32, sptr, v_s, c_u0, widx });
        const sword = a.id();
        try a.op(61, &.{ t_u32, sword, sptr });
        const spair = a.id();
        try a.op(124, &.{ t_v2f16, spair, sword }); // Bitcast
        const mptr = a.id();
        try a.op(65, &.{ t_ptr_smd_f32, mptr, v_md, c_u0, mdi_t[t] });
        const mval = a.id();
        try a.op(61, &.{ t_f32, mval, mptr });
        const mdi1 = a.id();
        try a.op(128, &.{ t_u32, mdi1, mdi_t[t], c_u1 });
        const dptr = a.id();
        try a.op(65, &.{ t_ptr_smd_f32, dptr, v_md, c_u0, mdi1 });
        const dval = a.id();
        try a.op(61, &.{ t_f32, dval, dptr });
        const jj0 = a.id();
        try a.op(128, &.{ t_u32, jj0, k0v, jcol_t[t] });
        for (0..2) |half| {
            const hf = a.id();
            try a.op(81, &.{ t_f16, hf, spair, @intCast(half) }); // CompositeExtract
            const sval = a.id();
            try a.op(115, &.{ t_f32, sval, hf }); // FConvert f16 -> f32
            const shifted = a.id();
            try a.op(131, &.{ t_f32, shifted, sval, mval }); // FSub
            const eval = a.id();
            try a.op(12, &.{ t_f32, eval, ext_glsl, 27, shifted }); // Exp
            const pval = a.id();
            try a.op(133, &.{ t_f32, pval, eval, dval }); // FMul
            // Padded j columns hold S = 0; with a negative row max that P
            // overflows f16 (Inf * V(0) = NaN), so force it to zero.
            var jj = jj0;
            if (half == 1) {
                const j1 = a.id();
                try a.op(128, &.{ t_u32, j1, jj0, c_u1 });
                jj = j1;
            }
            const j_ok = a.id();
            try a.op(176, &.{ t_bool, j_ok, jj, p_seq }); // ULessThan
            const pclamp = a.id();
            try a.op(169, &.{ t_f32, pclamp, j_ok, pval, c_f32_0 }); // Select
            const hval = a.id();
            try a.op(115, &.{ t_f16, hval, pclamp }); // FConvert
            var eidx = e_t[t];
            if (half == 1) {
                const e1 = a.id();
                try a.op(128, &.{ t_u32, e1, e_t[t], c_u1 });
                eidx = e1;
            }
            const pptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, pptr, v_psh, eidx });
            try a.op(62, &.{ pptr, hval });
        }
    }
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 }); // OpControlBarrier

    // 4 j sub-steps: P tiles from the slab, V tiles straight from global;
    // 2x2 of 64x64 per warp (16 MMAs per 8 fragment loads).
    var acc_cur = acc_phi;
    const a_shbase = a.id();
    try a.op(132, &.{ t_u32, a_shbase, wm64, c_u64 }); // warp_m*64 rows * 64 cols
    for (0..4) |ks| {
        var a_off = a_shbase;
        if (ks > 0) {
            const ao = a.id();
            try a.op(128, &.{ t_u32, ao, a_shbase, c_k16[ks] });
            a_off = ao;
        }
        var ma: [4]u32 = undefined;
        var ao_cur = a_off;
        for (0..4) |r| {
            if (r > 0) {
                const a2 = a.id();
                try a.op(128, &.{ t_u32, a2, ao_cur, c_u1024 }); // +16 rows * 64
                ao_cur = a2;
            }
            const aptr = a.id();
            try a.op(65, &.{ t_ptr_wg_f16, aptr, v_psh, ao_cur });
            ma[r] = a.id();
            try a.op(4457, &.{ t_mat_a, ma[r], aptr, c_u0, c_u64 });
        }
        // V row block: j = k0 + ks*16; warp's column quarter.
        const jrow = a.id();
        try a.op(128, &.{ t_u32, jrow, k0v, c_k16[ks] });
        const jmul = a.id();
        try a.op(132, &.{ t_u32, jmul, jrow, p_vstride });
        const vb0 = a.id();
        try a.op(128, &.{ t_u32, vb0, jmul, kvmul });
        const vbase = a.id();
        try a.op(128, &.{ t_u32, vbase, vb0, wn64 });
        for (0..4) |nt| {
            var v_off = vbase;
            if (nt > 0) {
                const vo = a.id();
                try a.op(128, &.{ t_u32, vo, vbase, c_col16[nt] });
                v_off = vo;
            }
            const vptr = a.id();
            try a.op(65, &.{ t_ptr_sv_f16, vptr, v_v, c_u0, v_off });
            const mb = a.id();
            try a.op(4457, &.{ t_mat_b, mb, vptr, c_u0, p_vstride });
            for (0..4) |r| {
                const acc_out = if (ks == 3) acc_next[r][nt] else a.id();
                try a.op(4459, &.{ t_mat_c, acc_out, ma[r], mb, acc_cur[r][nt] });
                acc_cur[r][nt] = acc_out;
            }
        }
    }
    try a.op(224, &.{ c_scope_wg, c_scope_wg, c_u264 });
    try a.op(249, &.{lb_cont});

    try a.op(248, &.{lb_cont});
    try a.op(128, &.{ t_u32, k0n, k0v, c_u64 });
    try a.op(249, &.{lb_head});

    // merge: store 4x4 OUT tiles at (row_s + r*16, head*128 + wn64 + nt*16).
    try a.op(248, &.{lb_merge});
    const orn16 = a.id();
    try a.op(132, &.{ t_u32, orn16, c_u16, p_ostride });
    var o_rowmul = a.id();
    try a.op(132, &.{ t_u32, o_rowmul, row_s, p_ostride });
    const o_head = a.id();
    try a.op(128, &.{ t_u32, o_head, headmul, wn64 });
    for (0..4) |r| {
        if (r > 0) {
            const nx = a.id();
            try a.op(128, &.{ t_u32, nx, o_rowmul, orn16 });
            o_rowmul = nx;
        }
        const o_base = a.id();
        try a.op(128, &.{ t_u32, o_base, o_rowmul, o_head });
        for (0..4) |nt| {
            var ob = o_base;
            if (nt > 0) {
                const oc = a.id();
                try a.op(128, &.{ t_u32, oc, o_base, c_col16[nt] });
                ob = oc;
            }
            const optr = a.id();
            try a.op(65, &.{ t_ptr_so_f32, optr, v_o, c_u0, ob });
            try a.op(4458, &.{ optr, acc_phi[r][nt], c_u0, p_ostride });
        }
    }
    try a.op(253, &.{});
    try a.op(56, &.{});

    a.words.items[3] = a.next;
    const out = try gpa.alignedAlloc(u8, .of(u32), a.words.items.len * 4);
    @memcpy(out, std.mem.sliceAsBytes(a.words.items));
    return out;
}
