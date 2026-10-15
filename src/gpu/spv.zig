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
