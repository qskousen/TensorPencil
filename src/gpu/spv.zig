//! SPIR-V binary post-processing.
//!
//! Zig 0.16's SPIR-V backend cannot emit `OpExecutionMode LocalSize` (the
//! std.gpu.executionMode helper is rejected by its assembler), so the
//! workgroup size is spliced into the module here at load time.

const std = @import("std");

const magic: u32 = 0x0723_0203;
const op_extension: u16 = 10;
const op_memory_model: u16 = 14;
const op_entry_point: u16 = 15;
const op_execution_mode: u16 = 16;
const op_capability: u16 = 17;
const mode_local_size: u32 = 17;
const cap_physical_storage_buffer: u32 = 5347;
const addressing_physical64: u32 = 5348;
const addressing_logical: u32 = 0;

pub const Error = error{ InvalidSpirv, EntryPointNotFound, OutOfMemory };

const op_name: u16 = 5;
const op_member_name: u16 = 6;
const op_function: u16 = 54;
const op_function_end: u16 = 56;
const op_function_call: u16 = 57;
const op_execution_mode_id: u16 = 331;

/// A copy of `spv` holding one entry point: its function and every function it reaches,
/// all module-scope declarations, and no debug names. A driver processes the whole
/// module it is handed for each pipeline, so one kernel then costs that kernel.
/// Decorations on removed functions go with them; nothing else may target an id
/// inside a removed function (Zig emits no such decoration; `forEntry`'s test checks).
pub fn forEntry(gpa: std.mem.Allocator, spv: []align(4) const u8, entry_name: []const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const w = std.mem.bytesAsSlice(u32, spv);
    if (w[0] != magic) return error.InvalidSpirv;

    // Each function's word range and the functions it calls.
    const Fn = struct { start: usize, end: usize };
    var fns: std.AutoHashMapUnmanaged(u32, Fn) = .empty;
    defer fns.deinit(gpa);
    var calls: std.ArrayList([2]u32) = .empty; // caller, callee
    defer calls.deinit(gpa);
    var entry: ?u32 = null;
    var cur: ?u32 = null;
    var i: usize = 5;
    while (i < w.len) {
        const wc: usize = w[i] >> 16;
        const op: u16 = @truncate(w[i]);
        if (wc == 0 or i + wc > w.len) return error.InvalidSpirv;
        switch (op) {
            op_entry_point => if (wc >= 4 and std.mem.eql(u8, wordString(w[i + 3 .. i + wc]), entry_name)) {
                entry = w[i + 2];
            },
            op_function => {
                cur = w[i + 2];
                try fns.put(gpa, w[i + 2], .{ .start = i, .end = 0 });
            },
            op_function_call => try calls.append(gpa, .{ cur orelse return error.InvalidSpirv, w[i + 3] }),
            op_function_end => {
                fns.getPtr(cur orelse return error.InvalidSpirv).?.end = i + wc;
                cur = null;
            },
            else => {},
        }
        i += wc;
    }
    const root = entry orelse return error.EntryPointNotFound;

    var keep: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer keep.deinit(gpa);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(gpa);
    try stack.append(gpa, root);
    while (stack.pop()) |f| {
        if ((try keep.getOrPut(gpa, f)).found_existing) continue;
        for (calls.items) |c| if (c[0] == f) try stack.append(gpa, c[1]);
    }

    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, w[0..5]);
    i = 5;
    while (i < w.len) {
        const wc: usize = w[i] >> 16;
        const op: u16 = @truncate(w[i]);
        const insn = w[i .. i + wc];
        const drop = switch (op) {
            op_name, op_member_name => true,
            op_entry_point => insn[2] != root,
            op_execution_mode, op_execution_mode_id => insn[1] != root,
            op_decorate => fns.contains(insn[1]) and !keep.contains(insn[1]),
            op_function => {
                const f = fns.get(insn[2]).?;
                if (!keep.contains(insn[2])) {
                    i = f.end;
                    continue;
                }
                try out.appendSlice(gpa, w[f.start..f.end]);
                i = f.end;
                continue;
            },
            else => false,
        };
        if (!drop) try out.appendSlice(gpa, insn);
        i += wc;
    }
    const words = try out.toOwnedSlice(gpa);
    return std.mem.sliceAsBytes(words);
}

/// A literal string operand: bytes up to the first NUL.
fn wordString(ws: []const u32) []const u8 {
    const bytes = std.mem.sliceAsBytes(ws);
    return bytes[0 .. std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len];
}

/// Return a copy of `spv` with `OpExecutionMode <entry> LocalSize x y z`
/// inserted after the OpEntryPoint whose name matches `entry_name`.
pub fn withLocalSize(gpa: std.mem.Allocator, spv: []const u8, entry_name: []const u8, x: u32, y: u32, z: u32) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    const word_at = struct {
        fn get(bytes: []const u8, i: usize) u32 {
            return std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        }
    }.get;
    if (word_at(spv, 0) != magic) return error.InvalidSpirv;

    // Find the OpEntryPoint with the requested name; record its id and the
    // word offset just past the instruction.
    var entry_id: ?u32 = null;
    var insert_at: usize = 0;
    var i: usize = 5;
    while (i < n_words) {
        const first = word_at(spv, i);
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        if (opcode == op_entry_point and wc >= 4) {
            // words: [op] execution_model entry_id name... interface_ids...
            const name_bytes = spv[(i + 3) * 4 .. (i + wc) * 4];
            const name_end = std.mem.indexOfScalar(u8, name_bytes, 0) orelse name_bytes.len;
            if (std.mem.eql(u8, name_bytes[0..name_end], entry_name)) {
                entry_id = word_at(spv, i + 2);
                insert_at = i + wc;
                break;
            }
        }
        i += wc;
    }
    const id = entry_id orelse return error.EntryPointNotFound;

    // OpExecutionMode has 6 words: header, entry, LocalSize, x, y, z.
    const out = try gpa.alignedAlloc(u8, .of(u32), spv.len + 6 * 4);
    @memcpy(out[0 .. insert_at * 4], spv[0 .. insert_at * 4]);
    var insn: [6]u32 = .{ (6 << 16) | @as(u32, op_execution_mode), id, mode_local_size, x, y, z };
    @memcpy(out[insert_at * 4 ..][0 .. 6 * 4], std.mem.sliceAsBytes(&insn));
    @memcpy(out[(insert_at + 6) * 4 ..], spv[insert_at * 4 ..]);
    return out;
}

/// Downgrade PhysicalStorageBuffer64 modules that never use physical
/// pointers to plain Logical addressing: drops the capability and extension
/// and rewrites OpMemoryModel. The Zig backend declares physical addressing
/// unconditionally, which at least one driver (NVIDIA 580) mishandles in
/// combination with workgroup storage.
pub fn withLogicalAddressing(gpa: std.mem.Allocator, spv: []const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, n_words);
    const in_words: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (in_words[0] != magic) return error.InvalidSpirv;
    for (in_words[0..5]) |w| try out.append(gpa, w);

    var i: usize = 5;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        skip: {
            if (opcode == op_capability and in_words[i + 1] == cap_physical_storage_buffer) break :skip;
            if (opcode == op_extension) {
                const name = std.mem.sliceAsBytes(in_words[i + 1 .. i + wc]);
                if (std.mem.startsWith(u8, name, "SPV_KHR_physical_storage_buffer")) break :skip;
            }
            const at = out.items.len;
            for (in_words[i .. i + wc]) |w| try out.append(gpa, w);
            if (opcode == op_memory_model and out.items[at + 1] == addressing_physical64) {
                out.items[at + 1] = addressing_logical;
            }
        }
        i += wc;
    }
    const bytes = try gpa.alignedAlloc(u8, .of(u32), out.items.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out.items));
    return bytes;
}

const op_decorate: u16 = 71;
const op_type_pointer: u16 = 32;
const decoration_array_stride: u32 = 6;
const storage_class_workgroup: u32 = 4;

/// Remove ArrayStride decorations from types pointed to by Workgroup-class
/// pointers. The Zig backend decorates every array type, but Vulkan forbids
/// explicit-layout decorations on workgroup memory (NVIDIA hangs on them;
/// Mesa ignores them). Assumes those types are not shared with
/// explicit-layout storage classes, which holds for our kernels where
/// workgroup tiles have unique array lengths.
pub fn stripWorkgroupStrides(gpa: std.mem.Allocator, spv: []const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    const in_words: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (in_words[0] != magic) return error.InvalidSpirv;

    // Pass 1: collect pointee types of Workgroup pointers.
    var wg_types: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer wg_types.deinit(gpa);
    var i: usize = 5;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        if (opcode == op_type_pointer and wc >= 4 and in_words[i + 2] == storage_class_workgroup) {
            try wg_types.put(gpa, in_words[i + 3], {});
        }
        i += wc;
    }

    // Pass 2: copy, dropping ArrayStride decorations on those types.
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, n_words);
    for (in_words[0..5]) |w| try out.append(gpa, w);
    i = 5;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        const drop = opcode == op_decorate and wc == 4 and
            in_words[i + 2] == decoration_array_stride and wg_types.contains(in_words[i + 1]);
        if (!drop) {
            for (in_words[i .. i + wc]) |w| try out.append(gpa, w);
        }
        i += wc;
    }
    const bytes = try gpa.alignedAlloc(u8, .of(u32), out.items.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out.items));
    return bytes;
}

const op_member_decorate: u16 = 72;

/// Drop exact-duplicate OpDecorate/OpMemberDecorate instructions. The Zig
/// backend emits automatic layout decorations for storage structs and our
/// kernels add the Vulkan-required ones via inline asm, so members end up
/// decorated twice, invalid SPIR-V that NVIDIA rejects at runtime.
pub fn dedupeDecorations(gpa: std.mem.Allocator, spv: []const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    const in_words: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (in_words[0] != magic) return error.InvalidSpirv;

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        seen.deinit(gpa);
    }

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, n_words);
    for (in_words[0..5]) |w| try out.append(gpa, w);

    var i: usize = 5;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        var drop = false;
        if (opcode == op_decorate or opcode == op_member_decorate) {
            const key_src = std.mem.sliceAsBytes(in_words[i .. i + wc]);
            const entry = try seen.getOrPut(gpa, key_src);
            if (entry.found_existing) {
                drop = true;
            } else {
                entry.key_ptr.* = try gpa.dupe(u8, key_src);
            }
        }
        if (!drop) {
            for (in_words[i .. i + wc]) |w| try out.append(gpa, w);
        }
        i += wc;
    }
    const bytes = try gpa.alignedAlloc(u8, .of(u32), out.items.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out.items));
    return bytes;
}

const decoration_no_contraction: u32 = 42;

/// Float arithmetic whose result a driver may otherwise contract into an FMA.
fn contractible(opcode: u16) bool {
    return switch (opcode) {
        127, 129, 131, 133, 136, 142 => true, // FNegate, FAdd, FSub, FMul, FDiv, VectorTimesScalar
        else => false,
    };
}

/// Every float add, subtract, multiply, divide and negate decorated `NoContraction`, so
/// the driver forms no FMA: two kernels summing in the same order then round the same,
/// whatever shape each compiles to. The decorations go before the first type.
pub fn withNoContraction(gpa: std.mem.Allocator, spv: []const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    const in_words: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (in_words[0] != magic) return error.InvalidSpirv;

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var insert_at: ?usize = null;
    var i: usize = 5;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        if (insert_at == null and !preamble(opcode)) insert_at = i;
        // Result type, then result id.
        if (contractible(opcode) and wc >= 3) try ids.append(gpa, in_words[i + 2]);
        i += wc;
    }
    const at = insert_at orelse return error.InvalidSpirv;

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, n_words + ids.items.len * 3);
    for (in_words[0..at]) |w| try out.append(gpa, w);
    for (ids.items) |id| try out.appendSlice(gpa, &.{ (3 << 16) | @as(u32, op_decorate), id, decoration_no_contraction });
    for (in_words[at..]) |w| try out.append(gpa, w);
    const bytes = try gpa.alignedAlloc(u8, .of(u32), out.items.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out.items));
    return bytes;
}

/// The module sections before the first type: capabilities, extensions, imports, the
/// memory model, entry points and execution modes, debug names and annotations.
fn preamble(opcode: u16) bool {
    return switch (opcode) {
        2...8, 10, 11, 14...17, 71...75, 330, 331, 332, 5632, 5633 => true,
        else => false,
    };
}

// SPIR-V capability numbers the Zig backend won't declare for raw inline-asm
// ops, injected here per-module.
pub const cap_dot_product: u32 = 6019; // DotProduct (OpSDot)
pub const cap_dot_product_input_4x8_packed: u32 = 6018; // DotProductInput4x8BitPacked
pub const cap_group_nonuniform: u32 = 61; // GroupNonUniform (subgroup ops base)
pub const cap_group_nonuniform_arithmetic: u32 = 63; // GroupNonUniformArithmetic (Reduce)
pub const cap_group_nonuniform_shuffle: u32 = 65; // GroupNonUniformShuffle (butterfly)
pub const cap_storage_buffer_16bit: u32 = 4433; // StorageBuffer16BitAccess (native u16 loads)

/// Inject `caps` (OpCapability) + optional `ext` (OpExtension) into a module so
/// it can use ops the Zig backend emits via inline asm but never declares
/// (OpSDot, OpGroupNonUniform*, ...). New capability words lead the capability
/// block; the extension follows it, the order SPIR-V requires (all
/// capabilities, then extensions, then the rest). Applied per-module so the
/// shared kernels stay valid on devices lacking the capability.
pub fn withCapabilities(gpa: std.mem.Allocator, spv: []const u8, caps: []const u32, ext: ?[]const u8) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const n_words = spv.len / 4;
    const in_words: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (in_words[0] != magic) return error.InvalidSpirv;

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, n_words + 2 * caps.len + 8);
    for (in_words[0..5]) |w| try out.append(gpa, w);

    // Leading OpCapability instructions (2 words each).
    for (caps) |cap| {
        try out.append(gpa, (2 << 16) | @as(u32, op_capability));
        try out.append(gpa, cap);
    }

    // Optional OpExtension, inserted once the capability block ends.
    var ext_words: [16]u32 = @splat(0);
    var ext_n: u32 = 0;
    if (ext) |e| {
        std.debug.assert(e.len < ext_words.len * 4);
        @memcpy(std.mem.sliceAsBytes(ext_words[0..])[0..e.len], e);
        ext_n = @intCast((e.len + 4) / 4); // + null terminator, rounded up to words
    }
    var i: usize = 5;
    var inserted_ext = ext == null;
    while (i < n_words) {
        const first = in_words[i];
        const opcode: u16 = @truncate(first & 0xFFFF);
        const wc: usize = first >> 16;
        if (wc == 0 or i + wc > n_words) return error.InvalidSpirv;
        if (!inserted_ext and opcode != op_capability) {
            try out.append(gpa, ((ext_n + 1) << 16) | @as(u32, op_extension));
            for (ext_words[0..ext_n]) |w| try out.append(gpa, w);
            inserted_ext = true;
        }
        for (in_words[i .. i + wc]) |w| try out.append(gpa, w);
        i += wc;
    }
    const bytes = try gpa.alignedAlloc(u8, .of(u32), out.items.len * 4);
    @memcpy(bytes, std.mem.sliceAsBytes(out.items));
    return bytes;
}

const op_type_int: u16 = 21;
const op_type_float: u16 = 22;
const op_type_vector: u16 = 23;
const op_type_array: u16 = 28;
const op_type_struct: u16 = 30;
const op_constant: u16 = 43;
const op_variable: u16 = 59;
const cap_float16: u32 = 9;
const cap_rounding_mode_rte: u32 = 4467;
const mode_rounding_mode_rte: u32 = 4462;
const cap_denorm_preserve: u32 = 4464;
const mode_denorm_preserve: u32 = 4459;

/// The float controls `withFloatModes` asks of every entry point.
pub const FloatModes = struct {
    /// `RoundingModeRTE 16`: SPIR-V leaves an undecorated f16 result's rounding to the
    /// driver, and some truncate.
    rte16: bool = false,
    /// `DenormPreserve 16`: gradual underflow, which some drivers only give when asked.
    denorm16: bool = false,
    /// `DenormPreserve 32`, for a device whose float controls cannot differ per width.
    denorm32: bool = false,

    fn any(m: FloatModes) bool {
        return m.rte16 or m.denorm16 or m.denorm32;
    }
};

/// A copy of `spv` with each mode in `modes` on every entry point, its capability and
/// SPV_KHR_float_controls declared. A module that declares no Float16 comes back
/// unchanged, as do `modes` the device does not report.
pub fn withFloatModes(gpa: std.mem.Allocator, spv: []const u8, modes: FloatModes) Error![]align(4) u8 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const w: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (w[0] != magic) return error.InvalidSpirv;
    var has_f16 = false;
    var have = [2]bool{ false, false };
    var entries: std.ArrayList(u32) = .empty;
    defer entries.deinit(gpa);
    var after_entries: usize = 0;
    var i: usize = 5;
    while (i < w.len) {
        const wc: usize = w[i] >> 16;
        const op: u16 = @truncate(w[i]);
        if (wc == 0 or i + wc > w.len) return error.InvalidSpirv;
        if (op == op_capability and wc == 2) {
            if (w[i + 1] == cap_float16) has_f16 = true;
            if (w[i + 1] == cap_rounding_mode_rte) have[0] = true;
            if (w[i + 1] == cap_denorm_preserve) have[1] = true;
        }
        if (op == op_entry_point and wc >= 4) {
            try entries.append(gpa, w[i + 2]);
            after_entries = i + wc;
        }
        i += wc;
    }
    if (!has_f16 or entries.items.len == 0 or !modes.any()) {
        const copy = try gpa.alignedAlloc(u8, .of(u32), spv.len);
        @memcpy(copy, spv);
        return copy;
    }
    var insns: [3][2]u32 = undefined;
    var n_modes: usize = 0;
    var caps: [2]u32 = undefined;
    var n_caps: usize = 0;
    if (modes.rte16) {
        insns[n_modes] = .{ mode_rounding_mode_rte, 16 };
        n_modes += 1;
        if (!have[0]) {
            caps[n_caps] = cap_rounding_mode_rte;
            n_caps += 1;
        }
    }
    for ([2]bool{ modes.denorm16, modes.denorm32 }, [2]u32{ 16, 32 }) |on, width| if (on) {
        insns[n_modes] = .{ mode_denorm_preserve, width };
        n_modes += 1;
    };
    if ((modes.denorm16 or modes.denorm32) and !have[1]) {
        caps[n_caps] = cap_denorm_preserve;
        n_caps += 1;
    }
    // Execution modes follow the last entry point; the capabilities shift that offset.
    const capped = if (n_caps == 0) blk: {
        const copy = try gpa.alignedAlloc(u8, .of(u32), spv.len);
        @memcpy(copy, spv);
        break :blk copy;
    } else try withCapabilities(gpa, spv, caps[0..n_caps], "SPV_KHR_float_controls");
    defer gpa.free(capped);
    const shift = (capped.len - spv.len) / 4;
    const at = (after_entries + shift) * 4;
    const added = entries.items.len * n_modes * 16;
    const out = try gpa.alignedAlloc(u8, .of(u32), capped.len + added);
    @memcpy(out[0..at], capped[0..at]);
    var o = at;
    for (entries.items) |id| for (insns[0..n_modes]) |m| {
        const insn = [4]u32{ (4 << 16) | @as(u32, op_execution_mode), id, m[0], m[1] };
        @memcpy(out[o..][0..16], std.mem.sliceAsBytes(&insn));
        o += 16;
    };
    @memcpy(out[at + added ..], capped[at..]);
    return out;
}

/// The byte size of every sized type in a module, from one pass over its declarations.
/// Members are summed without padding, which is exact for the flat arrays kernels
/// declare.
const Types = struct {
    size: std.AutoHashMapUnmanaged(u32, u64) = .empty,
    constant: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    /// The largest OpTypeArray, in bytes.
    largest_array: u64 = 0,

    fn deinit(t: *Types, gpa: std.mem.Allocator) void {
        t.size.deinit(gpa);
        t.constant.deinit(gpa);
    }

    /// Records instruction `a` if it declares a type or a constant.
    fn add(t: *Types, gpa: std.mem.Allocator, op: u16, a: []align(1) const u32) Error!void {
        switch (op) {
            op_type_int, op_type_float => if (a.len >= 3) try t.size.put(gpa, a[1], a[2] / 8),
            op_type_vector => if (a.len >= 4) try t.size.put(gpa, a[1], (t.size.get(a[2]) orelse 0) * a[3]),
            op_type_array => if (a.len >= 4) {
                const bytes = (t.size.get(a[2]) orelse 0) * (t.constant.get(a[3]) orelse 0);
                try t.size.put(gpa, a[1], bytes);
                t.largest_array = @max(t.largest_array, bytes);
            },
            op_type_struct => if (a.len >= 2) {
                var sum: u64 = 0;
                for (a[2..]) |m| sum += t.size.get(m) orelse 0;
                try t.size.put(gpa, a[1], sum);
            },
            op_constant => if (a.len >= 4) try t.constant.put(gpa, a[2], a[3]),
            else => {},
        }
    }
};

/// The largest fixed-size array `spv` declares, in bytes. One of 4 GiB or more wraps a
/// 32-bit size to zero, and at least one driver then reads every element as element
/// 0: an array over a whole buffer is declared runtime-sized instead.
pub fn largestArrayBytes(gpa: std.mem.Allocator, spv: []const u8) Error!u64 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const w: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (w[0] != magic) return error.InvalidSpirv;
    var types: Types = .{};
    defer types.deinit(gpa);
    var i: usize = 5;
    while (i < w.len) {
        const wc: usize = w[i] >> 16;
        if (wc == 0 or i + wc > w.len) return error.InvalidSpirv;
        try types.add(gpa, @truncate(w[i]), w[i .. i + wc]);
        i += wc;
    }
    return types.largest_array;
}

/// The most workgroup (shared) memory any one entry point of `spv` declares, in
/// bytes, to hold against `maxComputeSharedMemorySize`. From SPIR-V 1.4 an entry
/// point lists every global it uses, so each is counted alone; before that the whole
/// module's workgroup variables are charged to every entry.
pub fn workgroupBytes(gpa: std.mem.Allocator, spv: []const u8) Error!u64 {
    if (spv.len % 4 != 0 or spv.len < 20) return error.InvalidSpirv;
    const w: []align(1) const u32 = std.mem.bytesAsSlice(u32, spv);
    if (w[0] != magic) return error.InvalidSpirv;
    const per_entry = w[1] >= 0x0001_0400;
    var types: Types = .{};
    defer types.deinit(gpa);
    var pointee: std.AutoHashMapUnmanaged(u32, u32) = .empty; // workgroup pointer -> type
    defer pointee.deinit(gpa);
    var var_bytes: std.AutoHashMapUnmanaged(u32, u64) = .empty; // workgroup variable -> bytes
    defer var_bytes.deinit(gpa);
    var interfaces: std.ArrayList([]align(1) const u32) = .empty;
    defer interfaces.deinit(gpa);
    var i: usize = 5;
    while (i < w.len) {
        const wc: usize = w[i] >> 16;
        const op: u16 = @truncate(w[i]);
        if (wc == 0 or i + wc > w.len) return error.InvalidSpirv;
        const a = w[i .. i + wc];
        try types.add(gpa, op, a);
        switch (op) {
            op_entry_point => if (wc >= 4) {
                var n: usize = 3;
                while (n < wc) : (n += 1) if (std.mem.indexOfScalar(u8, std.mem.asBytes(&a[n]), 0) != null) break;
                try interfaces.append(gpa, a[@min(n + 1, wc)..]);
            },
            op_type_pointer => if (wc >= 4 and a[2] == storage_class_workgroup) try pointee.put(gpa, a[1], a[3]),
            op_variable => if (wc >= 4 and a[3] == storage_class_workgroup) {
                const t = pointee.get(a[1]) orelse return error.InvalidSpirv;
                try var_bytes.put(gpa, a[2], types.size.get(t) orelse 0);
            },
            else => {},
        }
        i += wc;
    }
    var all: u64 = 0;
    var it = var_bytes.valueIterator();
    while (it.next()) |b| all += b.*;
    if (!per_entry) return all;
    var worst: u64 = 0;
    for (interfaces.items) |ids| {
        var sum: u64 = 0;
        for (ids) |id| sum += var_bytes.get(id) orelse 0;
        worst = @max(worst, sum);
    }
    return worst;
}

test "patches local size after entry point" {
    const gpa = std.testing.allocator;
    // Minimal synthetic module: header + OpEntryPoint(GLCompute, id 7, "k").
    var module: std.ArrayList(u8) = .empty;
    defer module.deinit(gpa);
    const header = [_]u32{ magic, 0x10000, 0, 10, 0 };
    for (header) |w| try module.appendSlice(gpa, &std.mem.toBytes(w));
    // name "k\0\0\0" = 1 word; wc = 3 fixed + 1 name word = 4... entry point: [op|wc, model, id, name]
    const ep = [_]u32{ (4 << 16) | @as(u32, op_entry_point), 5, 7, std.mem.bytesToValue(u32, "k\x00\x00\x00") };
    for (ep) |w| try module.appendSlice(gpa, &std.mem.toBytes(w));

    const patched = try withLocalSize(gpa, module.items, "k", 64, 2, 1);
    defer gpa.free(patched);
    try std.testing.expectEqual(module.items.len + 24, patched.len);
    const words = std.mem.bytesAsSlice(u32, patched);
    try std.testing.expectEqual((6 << 16) | @as(u32, op_execution_mode), words[9]);
    try std.testing.expectEqual(@as(u32, 7), words[10]); // entry id
    try std.testing.expectEqual(mode_local_size, words[11]);
    try std.testing.expectEqual(@as(u32, 64), words[12]);
    try std.testing.expectEqual(@as(u32, 2), words[13]);

    try std.testing.expectError(error.EntryPointNotFound, withLocalSize(gpa, module.items, "other", 1, 1, 1));
}

test "forEntry cuts each dual entry point to a module that resolves" {
    const gpa = std.testing.allocator;
    const raw = @embedFile("dual_spv");
    const full = try gpa.alignedAlloc(u8, .of(u32), raw.len);
    defer gpa.free(full);
    @memcpy(full, raw);
    const fw = std.mem.bytesAsSlice(u32, full);
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var i: usize = 5;
    while (i < fw.len) : (i += fw[i] >> 16) {
        if (@as(u16, @truncate(fw[i])) == op_entry_point) try names.append(gpa, wordString(fw[i + 3 .. i + (fw[i] >> 16)]));
    }
    try std.testing.expect(names.items.len > 100);
    for (names.items) |name| {
        const one = try forEntry(gpa, full, name);
        defer gpa.free(one);
        errdefer std.debug.print("forEntry({s})\n", .{name});
        try std.testing.expect(one.len < full.len);
        const w = std.mem.bytesAsSlice(u32, one);
        // Every id a call or a decoration names is still defined here.
        var defined: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer defined.deinit(gpa);
        var entries: usize = 0;
        i = 5;
        while (i < w.len) : (i += w[i] >> 16) {
            const op: u16 = @truncate(w[i]);
            const wc = w[i] >> 16;
            if (op == op_entry_point) entries += 1;
            // Result ids: a type's is word 1, a typed result's word 2.
            if (op >= 19 and op <= 39) try defined.put(gpa, w[i + 1], {});
            if ((op >= 41 and op <= 46) or op == 54 or op == 59 or op == 55) try defined.put(gpa, w[i + 2], {});
            _ = wc;
        }
        try std.testing.expectEqual(@as(usize, 1), entries);
        i = 5;
        while (i < w.len) : (i += w[i] >> 16) {
            const op: u16 = @truncate(w[i]);
            if (op == op_function_call) try std.testing.expect(defined.contains(w[i + 3]));
            if (op == op_decorate) try std.testing.expect(defined.contains(w[i + 1]));
        }
    }
    try std.testing.expectError(error.EntryPointNotFound, forEntry(gpa, full, "no_such_kernel"));
}

test "workgroupBytes matches what the validation layer counts, per entry point" {
    const gpa = std.testing.allocator;
    const coopmat = @import("coopmat.zig");
    // The Khronos validation layer reported 49152 B for this module on a 16x16 device.
    const coop = try coopmat.buildGemmShared(gpa, false, coopmat.coop_warps8, coopmat.coop_acc_h16, false, false, 16, 16, 64, 64);
    defer gpa.free(coop);
    try std.testing.expectEqual(@as(u64, 49152), try workgroupBytes(gpa, coop));
    // A module of many kernels is charged its largest one, not their sum.
    const raw = @embedFile("dual_spv");
    const dual = try gpa.alignedAlloc(u8, .of(u32), raw.len);
    defer gpa.free(dual);
    @memcpy(dual, raw);
    const worst = try workgroupBytes(gpa, dual);
    const dw = std.mem.bytesAsSlice(u32, dual);
    var i: usize = 5;
    var name: []const u8 = "";
    while (i < dw.len) : (i += dw[i] >> 16) {
        if (@as(u16, @truncate(dw[i])) == op_entry_point) name = wordString(dw[i + 3 .. i + (dw[i] >> 16)]);
    }
    const one = try forEntry(gpa, dual, name);
    defer gpa.free(one);
    try std.testing.expect(try workgroupBytes(gpa, one) <= worst);
}

test "the weight-only pack tile woPackRows picks fits the shared memory it was picked for" {
    const gpa = std.testing.allocator;
    const coopmat = @import("coopmat.zig");
    try std.testing.expectEqual(@as(?u32, null), coopmat.woPackRows(8 * 1024));
    for ([_]u64{ 16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024 }, [_]u32{ 8, 16, 32, 32 }) |max_shared, want| {
        const r = coopmat.woPackRows(max_shared).?;
        try std.testing.expectEqual(want, r);
        for (0..4) |i| {
            const code = try coopmat.buildIntWoPack(gpa, r, i & 2 != 0, i & 1 != 0);
            defer gpa.free(code);
            try std.testing.expect(try workgroupBytes(gpa, code) <= max_shared);
        }
    }
}

test "withFloatModes marks every entry point of an f16 module and leaves others alone" {
    const gpa = std.testing.allocator;
    const coopmat = @import("coopmat.zig");
    const coop = try coopmat.buildGemmShared(gpa, true, false, false, false, false, 16, 16, 32, 32);
    defer gpa.free(coop);
    const rte = try withFloatModes(gpa, coop, .{ .rte16 = true, .denorm16 = true });
    defer gpa.free(rte);
    const w = std.mem.bytesAsSlice(u32, rte);
    var entries: usize = 0;
    var modes: usize = 0;
    var denorms: usize = 0;
    var cap = false;
    var cap_denorm = false;
    var last_entry: usize = 0;
    var first_mode: usize = 0;
    var i: usize = 5;
    while (i < w.len) : (i += w[i] >> 16) {
        const op: u16 = @truncate(w[i]);
        if (op == op_capability and w[i + 1] == cap_rounding_mode_rte) cap = true;
        if (op == op_capability and w[i + 1] == cap_denorm_preserve) cap_denorm = true;
        if (op == op_execution_mode and w[i + 2] == mode_denorm_preserve) {
            try std.testing.expectEqual(@as(u32, 16), w[i + 3]);
            denorms += 1;
        }
        if (op == op_entry_point) {
            entries += 1;
            last_entry = i;
        }
        if (op == op_execution_mode and w[i + 2] == mode_rounding_mode_rte) {
            try std.testing.expectEqual(@as(u32, 16), w[i + 3]);
            if (modes == 0) first_mode = i;
            modes += 1;
        }
    }
    try std.testing.expect(cap and cap_denorm);
    try std.testing.expect(entries > 0);
    try std.testing.expectEqual(entries, modes);
    try std.testing.expectEqual(entries, denorms);
    try std.testing.expect(first_mode > last_entry);
    // Workgroup accounting is unchanged by the patch.
    try std.testing.expectEqual(try workgroupBytes(gpa, coop), try workgroupBytes(gpa, rte));

    // No Float16 capability: returned as is.
    var plain: std.ArrayList(u8) = .empty;
    defer plain.deinit(gpa);
    for ([_]u32{ magic, 0x10400, 0, 10, 0, (4 << 16) | @as(u32, op_entry_point), 5, 7, std.mem.bytesToValue(u32, "k\x00\x00\x00") }) |x|
        try plain.appendSlice(gpa, &std.mem.toBytes(x));
    const same = try withFloatModes(gpa, plain.items, .{ .rte16 = true, .denorm16 = true });
    defer gpa.free(same);
    try std.testing.expectEqualSlices(u8, plain.items, same);
}

test "largestArrayBytes sees a 4 GiB buffer array that a runtime array avoids" {
    const gpa = std.testing.allocator;
    const coopmat = @import("coopmat.zig");
    // Every hand-written kernel declares its buffers runtime-sized.
    const gemm = try coopmat.buildGemmShared(gpa, true, false, false, false, false, 16, 16, 32, 32);
    defer gpa.free(gemm);
    try std.testing.expect(try largestArrayBytes(gpa, gemm) < 1 << 32);
    // The fixed form this replaced: 2^28 uvec4 is exactly 2^32 bytes.
    var m: std.ArrayList(u8) = .empty;
    defer m.deinit(gpa);
    for ([_]u32{
        magic,                             0x10500, 0,                                   10,                                0,
        (4 << 16) | @as(u32, op_type_int), 1,       32,                                  0,                                 (4 << 16) | @as(u32, op_type_vector),
        2,                                 1,       4,                                   (4 << 16) | @as(u32, op_constant), 1,
        3,                                 1 << 28, (4 << 16) | @as(u32, op_type_array), 4,                                 2,
        3,
    }) |x| try m.appendSlice(gpa, &std.mem.toBytes(x));
    try std.testing.expectEqual(@as(u64, 1 << 32), try largestArrayBytes(gpa, m.items));
}

test "withNoContraction decorates every float multiply and add, before the first type" {
    const gpa = std.testing.allocator;
    const sasm = @import("spirv_asm.zig");
    const code = try sasm.assemble(gpa, sasm.version_1_3,
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "main"
        \\OpExecutionMode %main LocalSize 1 1 1
        \\OpName %main "main"
        \\%void = OpTypeVoid
        \\%fnty = OpTypeFunction %void
        \\%f32 = OpTypeFloat 32
        \\%one = OpConstant %f32 1065353216
        \\%main = OpFunction %void None %fnty
        \\%entry = OpLabel
        \\%m = OpFMul %f32 %one %one
        \\%a = OpFAdd %f32 %m %one
        \\OpReturn
        \\OpFunctionEnd
    );
    defer gpa.free(code);
    const out = try withNoContraction(gpa, code);
    defer gpa.free(out);
    const words = std.mem.bytesAsSlice(u32, out);
    var decorated: usize = 0;
    var seen_type = false;
    var i: usize = 5;
    while (i < words.len) {
        const op: u16 = @truncate(words[i] & 0xFFFF);
        const wc = words[i] >> 16;
        if (op == 19) seen_type = true;
        if (op == op_decorate and words[i + 2] == decoration_no_contraction) {
            try std.testing.expect(!seen_type);
            decorated += 1;
        }
        i += wc;
    }
    try std.testing.expectEqual(@as(usize, 2), decorated);
    try std.testing.expectEqual(code.len + 2 * 12, out.len);
}
