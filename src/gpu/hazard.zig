//! Which buffer ranges a compute dispatch reads and writes, so a batch records a
//! barrier only before a dispatch that touches what an earlier one wrote (or writes
//! what an earlier one read), instead of after every dispatch.
//!
//! `bindingAccess` reads a module's SPIR-V once: a binding is written if any store,
//! atomic or other unrecognised use reaches a pointer derived from its variable. It
//! errs towards "written" and towards `untracked` (a module that can reach memory no
//! binding names, through device addresses or a store whose target it cannot place),
//! which costs only a barrier. Calling a written binding read is the one mistake
//! that produces wrong answers, and every rule here exists to rule it out.

const std = @import("std");

pub const max_bindings = 8;

/// What a module does to each descriptor binding of set 0, one bit per binding.
pub const Access = struct {
    used: u8 = 0,
    written: u8 = 0,
    /// The module may touch memory outside its bindings, or bindings it cannot tell
    /// apart: every dispatch of it is fenced on both sides.
    untracked: bool = false,

    pub const unknown: Access = .{ .untracked = true };
};

const op_type_pointer = 32;
const op_function_parameter = 55;
const op_function_call = 57;
const op_variable = 59;
const op_load = 61;
const op_store = 62;
const op_copy_memory = 63;
const op_copy_memory_sized = 64;
const op_access_chain = 65;
const op_in_bounds_access_chain = 66;
const op_ptr_access_chain = 67;
const op_array_length = 68;
const op_in_bounds_ptr_access_chain = 70;
const op_decorate = 71;
const op_copy_object = 83;
const op_convert_u_to_ptr = 120;
const op_bitcast = 124;
const op_select = 169;
const op_atomic_load = 227;
const op_atomic_store = 228;
const op_phi = 245;
const op_coop_load = 4457;
const op_coop_store = 4458;
const op_line = 8;
const op_ext_inst = 12;
const op_function = 54;
const op_function_end = 56;
const op_vector_shuffle = 79;
const op_composite_extract = 81;
const op_composite_insert = 82;
const op_control_barrier = 224;
const op_memory_barrier = 225;
const op_loop_merge = 246;
const op_selection_merge = 247;
const op_label = 248;
const op_branch = 249;
const op_branch_conditional = 250;
const op_switch = 251;
const op_kill = 252;
const op_return = 253;
const op_unreachable = 255;
const op_no_line = 317;
const op_coop_mul_add = 4459;
const op_coop_length = 4460;
const op_name = 5;
const op_member_name = 6;
const op_entry_point = 15;
const op_member_decorate = 72;
const op_decorate_id = 332;
const op_decorate_string = 5632;

const decoration_binding = 33;
const decoration_descriptor_set = 34;

const sc_input = 1;
const sc_uniform = 2;
const sc_output = 3;
const sc_workgroup = 4;
const sc_private = 6;
const sc_function = 7;
const sc_storage_buffer = 12;
const sc_physical_storage_buffer = 5349;

const none: u8 = 0xff;

/// Per-binding access of every entry point in `code`, unioned.
pub fn bindingAccess(gpa: std.mem.Allocator, code: []align(4) const u8) error{OutOfMemory}!Access {
    if (code.len < 20 or code.len % 4 != 0) return .unknown;
    const words = std.mem.bytesAsSlice(u32, code);
    if (words[0] != 0x0723_0203) return .unknown;
    const bound = words[3];
    if (bound == 0 or bound > 1 << 22) return .unknown;

    // binding[id]: the binding a variable is decorated with; set_ok[id]: set 0.
    const binding = try gpa.alloc(u8, bound);
    defer gpa.free(binding);
    @memset(binding, none);
    const set_of = try gpa.alloc(u32, bound);
    defer gpa.free(set_of);
    @memset(set_of, 0);
    // root[id]: the binding a pointer id derives from.
    const root = try gpa.alloc(u8, bound);
    defer gpa.free(root);
    @memset(root, none);
    // class[id]: the storage class of a pointer TYPE id; ptype[id]: an id's type.
    const class = try gpa.alloc(u32, bound);
    defer gpa.free(class);
    @memset(class, std.math.maxInt(u32));
    const ptype = try gpa.alloc(u32, bound);
    defer gpa.free(ptype);
    @memset(ptype, 0);

    var acc: Access = .{};
    const body = words[5..];

    // Decorations, pointer types and variables come before any function.
    var it = Iter{ .w = body };
    while (it.next()) |ins| {
        if (ins.len == 0) return .unknown;
        switch (ins[0] & 0xffff) {
            op_decorate => if (ins.len >= 4 and ins[1] < bound) switch (ins[2]) {
                decoration_binding => binding[ins[1]] = if (ins[3] < max_bindings) @intCast(ins[3]) else return .unknown,
                decoration_descriptor_set => set_of[ins[1]] = ins[3],
                else => {},
            },
            op_type_pointer => if (ins.len >= 3 and ins[1] < bound) {
                class[ins[1]] = ins[2];
                if (ins[2] == sc_physical_storage_buffer) acc.untracked = true;
            },
            op_convert_u_to_ptr => acc.untracked = true,
            else => {},
        }
    }
    if (it.bad) return .unknown;

    // Variables, then pointer derivations to a fixed point (an OpPhi may name an
    // id defined later in the stream).
    it = .{ .w = body };
    while (it.next()) |ins| {
        if ((ins[0] & 0xffff) != op_variable or ins.len < 4) continue;
        const ty = ins[1];
        const id = ins[2];
        if (id >= bound or ty >= bound) return .unknown;
        ptype[id] = ty;
        const sc = ins[3];
        if (sc != sc_storage_buffer and sc != sc_uniform) continue;
        if (binding[id] == none or set_of[id] != 0) {
            // A buffer variable with no binding, or in another set, cannot be tracked.
            acc.untracked = true;
            continue;
        }
        root[id] = binding[id];
        acc.used |= @as(u8, 1) << @intCast(binding[id]);
    }
    var changed = true;
    while (changed) {
        changed = false;
        it = .{ .w = body };
        while (it.next()) |ins| {
            const op = ins[0] & 0xffff;
            const base_at: usize = switch (op) {
                op_access_chain, op_in_bounds_access_chain, op_ptr_access_chain, op_in_bounds_ptr_access_chain, op_copy_object, op_bitcast => 3,
                else => continue,
            };
            if (ins.len <= base_at) continue;
            const id = ins[2];
            const base = ins[base_at];
            if (id >= bound or base >= bound) return .unknown;
            if (ptype[id] != ins[1]) {
                ptype[id] = ins[1];
                changed = true;
            }
            if (root[base] != none and root[id] != root[base]) {
                if (root[id] != none) {
                    // Derived from two bindings at once: no single answer.
                    acc.untracked = true;
                    continue;
                }
                root[id] = root[base];
                changed = true;
            }
        }
    }
    // Types of the other pointer-producing ids a store target can be.
    it = .{ .w = body };
    while (it.next()) |ins| {
        switch (ins[0] & 0xffff) {
            op_function_parameter, op_function_call, op_load, op_select, op_phi => if (ins.len >= 3 and ins[2] < bound) {
                ptype[ins[2]] = ins[1];
            },
            else => {},
        }
    }

    // Uses, inside function bodies only: ids are unique, so an id operand equal to a
    // pointer's id IS that pointer, and only literal operands can collide with one.
    // The ops below that carry literals are read at their id positions alone.
    it = .{ .w = body };
    var in_fn = false;
    while (it.next()) |ins| {
        const op = ins[0] & 0xffff;
        switch (op) {
            op_function => {
                in_fn = true;
                continue;
            },
            op_function_end => {
                in_fn = false;
                continue;
            },
            else => {},
        }
        if (!in_fn) continue;
        switch (op) {
            // No operand of these can be a buffer pointer.
            op_function_parameter, op_variable, op_label, op_branch, op_branch_conditional, op_switch, op_loop_merge, op_selection_merge, op_return, op_unreachable, op_kill, op_line, op_no_line, op_control_barrier, op_memory_barrier, op_coop_mul_add, op_coop_length => continue,
            333...366 => continue, // OpGroupNonUniform*: values and literal group ops
            // Pointer derivations (handled above) and reads: indices, memory
            // operands, layouts and strides are never buffer pointers.
            op_access_chain, op_in_bounds_access_chain, op_ptr_access_chain, op_in_bounds_ptr_access_chain, op_copy_object, op_bitcast => continue,
            op_load, op_atomic_load, op_coop_load, op_array_length => continue,
            op_composite_extract => {
                if (ins.len > 3) markUnknownUse(&acc, root, ins[3]);
                continue;
            },
            op_composite_insert, op_vector_shuffle => {
                for (ins[3..@min(ins.len, 5)]) |w| markUnknownUse(&acc, root, w);
                continue;
            },
            op_ext_inst => {
                // An out-parameter (modf, frexp) writes through its pointer.
                if (ins.len > 5) for (ins[5..]) |w| markUnknownUse(&acc, root, w);
                continue;
            },
            else => {},
        }
        // A write: its target, and the one operand that could be a stored pointer.
        const target_at: ?usize, const value_at: ?usize = switch (op) {
            op_store, op_coop_store => .{ 1, 2 },
            op_atomic_store => .{ 1, 4 },
            op_copy_memory, op_copy_memory_sized => .{ 1, null },
            // Every other atomic takes its pointer at word 3 and only values after.
            229...242, 318, 319, 5614, 5615, 6035 => .{ 3, null },
            else => .{ null, null },
        };
        if (target_at) |t| {
            if (ins.len <= t) return .unknown;
            const p = ins[t];
            if (p >= bound) return .unknown;
            if (root[p] != none) {
                acc.written |= @as(u8, 1) << @intCast(root[p]);
            } else if (!privateTarget(class, ptype, p)) {
                // Through a pointer this analysis cannot place: any buffer.
                acc.untracked = true;
            }
            if (value_at) |v| if (ins.len > v) markUnknownUse(&acc, root, ins[v]);
            continue;
        }
        // Anything else with a buffer pointer among its operands (a call, a phi, a
        // select, an op this list does not know) may write through it.
        for (ins[1..]) |w| markUnknownUse(&acc, root, w);
    }
    if (it.bad) return .unknown;
    acc.written &= acc.used;
    return acc;
}

/// A literal operand can collide with a pointer id's number; that only adds a write.
fn markUnknownUse(acc: *Access, root: []const u8, w: u32) void {
    if (w < root.len and root[w] != none) acc.written |= @as(u8, 1) << @intCast(root[w]);
}

fn privateTarget(class: []const u32, ptype: []const u32, p: u32) bool {
    const ty = ptype[p];
    if (ty == 0 or ty >= class.len) return false;
    return switch (class[ty]) {
        sc_function, sc_private, sc_workgroup, sc_input, sc_output => true,
        else => false,
    };
}

const Iter = struct {
    w: []const u32,
    at: usize = 0,
    bad: bool = false,

    fn next(it: *Iter) ?[]const u32 {
        if (it.at >= it.w.len) return null;
        const n = it.w[it.at] >> 16;
        if (n == 0 or it.at + n > it.w.len) {
            it.bad = true;
            it.at = it.w.len;
            return null;
        }
        defer it.at += n;
        return it.w[it.at..][0..n];
    }
};

/// What every shader module and pipeline of the process does to its bindings, keyed
/// by device and handle. A creation overwrites, so a recycled handle never answers
/// for its predecessor; a lookup that misses is `untracked`. Filled from the
/// threads `PipeBatch` compiles on, hence the lock.
pub const Registry = struct {
    pub const Kind = enum(u8) { module, pipeline };
    pub const Key = struct { device: u64, handle: u64, kind: Kind };

    lock: std.atomic.Mutex = .unlocked,
    map: std.AutoHashMapUnmanaged(Key, Access) = .empty,

    fn acquire(r: *Registry) void {
        while (!r.lock.tryLock()) std.atomic.spinLoopHint();
    }

    /// An entry that cannot be stored is left out, which reads back as untracked.
    pub fn put(r: *Registry, key: Key, acc: Access) void {
        r.acquire();
        defer r.lock.unlock();
        r.map.put(std.heap.smp_allocator, key, acc) catch {
            _ = r.map.remove(key);
        };
    }

    pub fn get(r: *Registry, key: Key) Access {
        r.acquire();
        defer r.lock.unlock();
        return r.map.get(key) orelse .unknown;
    }
};

pub var registry: Registry = .{};

/// One buffer range a dispatch binds. `hi` is exclusive; a binding with no known
/// extent runs to the end of the buffer.
pub const Range = struct {
    buf: u64,
    lo: u64,
    hi: u64 = std.math.maxInt(u64),

    fn overlaps(a: Range, b: Range) bool {
        return a.buf == b.buf and a.lo < b.hi and b.lo < a.hi;
    }
};

pub const Use = struct { range: Range, write: bool };

/// The ranges dispatches have read and written since the last barrier.
pub const Tracker = struct {
    const cap = 64;
    reads: [cap]Range = undefined,
    writes: [cap]Range = undefined,
    n_reads: usize = 0,
    n_writes: usize = 0,
    /// A dispatch whose accesses are unknown ran since the last barrier.
    poisoned: bool = false,

    pub fn clear(t: *Tracker) void {
        t.n_reads = 0;
        t.n_writes = 0;
        t.poisoned = false;
    }

    pub fn pending(t: *const Tracker) bool {
        return t.poisoned or t.n_reads + t.n_writes > 0;
    }

    /// Whether a dispatch making `uses` must wait for what ran since the last barrier.
    pub fn conflicts(t: *const Tracker, uses: []const Use) bool {
        if (t.poisoned) return true;
        for (uses) |u| {
            for (t.writes[0..t.n_writes]) |w| if (u.range.overlaps(w)) return true;
            if (u.write) for (t.reads[0..t.n_reads]) |r| if (u.range.overlaps(r)) return true;
        }
        return false;
    }

    /// Note `uses` as having run; a full table poisons, which costs one barrier.
    pub fn record(t: *Tracker, uses: []const Use) void {
        for (uses) |u| {
            if (u.write) {
                if (t.n_writes == cap) {
                    t.poisoned = true;
                    continue;
                }
                t.writes[t.n_writes] = u.range;
                t.n_writes += 1;
            } else {
                if (t.n_reads == cap) {
                    t.poisoned = true;
                    continue;
                }
                t.reads[t.n_reads] = u.range;
                t.n_reads += 1;
            }
        }
    }
};

const testing = std.testing;
const asm_ = @import("spirv_asm.zig");

/// A compute module over two bindings of a runtime f32 array, with `body` inside
/// main. `%b0` / `%b1` are the variables, `%u0` a uint zero, `%pf` the element
/// pointer type.
fn module(gpa: std.mem.Allocator, extra_decl: []const u8, body: []const u8) ![]align(4) u8 {
    var a = asm_.Assembler.init(gpa, asm_.version_1_3);
    defer a.deinit();
    try a.add(
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "main"
        \\OpExecutionMode %main LocalSize 1 1 1
        \\OpDecorate %arr ArrayStride 4
        \\OpMemberDecorate %st 0 Offset 0
        \\OpDecorate %st Block
        \\OpDecorate %b0 DescriptorSet 0
        \\OpDecorate %b0 Binding 0
        \\OpDecorate %b1 DescriptorSet 0
        \\OpDecorate %b1 Binding 1
        \\%void = OpTypeVoid
        \\%fnty = OpTypeFunction %void
        \\%f32 = OpTypeFloat 32
        \\%u32 = OpTypeInt 32 0
        \\%u0 = OpConstant %u32 0
        \\%arr = OpTypeRuntimeArray %f32
        \\%st = OpTypeStruct %arr
        \\%pst = OpTypePointer StorageBuffer %st
        \\%pf = OpTypePointer StorageBuffer %f32
        \\%pff = OpTypePointer Function %f32
        \\%b0 = OpVariable %pst StorageBuffer
        \\%b1 = OpVariable %pst StorageBuffer
    );
    try a.add(extra_decl);
    try a.add(
        \\%main = OpFunction %void None %fnty
        \\%entry = OpLabel
    );
    try a.add(body);
    try a.add(
        \\OpReturn
        \\OpFunctionEnd
    );
    return a.finish();
}

test "a binding stored to is written, one only loaded from is read" {
    const gpa = testing.allocator;
    const code = try module(gpa, "",
        \\%p0 = OpAccessChain %pf %b0 %u0 %u0
        \\%v = OpLoad %f32 %p0
        \\%p1 = OpAccessChain %pf %b1 %u0 %u0
        \\OpStore %p1 %v
    );
    defer gpa.free(code);
    const a = try bindingAccess(gpa, code);
    try testing.expect(!a.untracked);
    try testing.expectEqual(@as(u8, 0b11), a.used);
    try testing.expectEqual(@as(u8, 0b10), a.written);
}

test "a store through a copied pointer still writes its binding" {
    const gpa = testing.allocator;
    const code = try module(gpa, "",
        \\%p0 = OpAccessChain %pf %b0 %u0 %u0
        \\%c = OpCopyObject %pf %p0
        \\%v = OpLoad %f32 %c
        \\OpStore %c %v
    );
    defer gpa.free(code);
    const a = try bindingAccess(gpa, code);
    try testing.expectEqual(@as(u8, 0b01), a.written);
}

test "a buffer pointer handed to a function counts as written" {
    const gpa = testing.allocator;
    const code = try module(gpa,
        \\%hfty = OpTypeFunction %void %pf
        \\%h = OpFunction %void None %hfty
        \\%hp = OpFunctionParameter %pf
        \\%hl = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    ,
        \\%p1 = OpAccessChain %pf %b1 %u0 %u0
        \\%r = OpFunctionCall %void %h %p1
    );
    defer gpa.free(code);
    const a = try bindingAccess(gpa, code);
    try testing.expectEqual(@as(u8, 0b10), a.written);
}

test "a store through a buffer pointer of unknown origin makes the module untracked" {
    const gpa = testing.allocator;
    // The parameter's pointer is a buffer pointer the caller never names.
    const code = try module(gpa,
        \\%hfty = OpTypeFunction %void %pf
        \\%h = OpFunction %void None %hfty
        \\%hp = OpFunctionParameter %pf
        \\%hl = OpLabel
        \\OpStore %hp %u0
        \\OpReturn
        \\OpFunctionEnd
    ,
        \\%p0 = OpAccessChain %pf %b0 %u0 %u0
        \\%v = OpLoad %f32 %p0
    );
    defer gpa.free(code);
    const a = try bindingAccess(gpa, code);
    try testing.expect(a.untracked);
}

test "a store to function-local memory touches no binding" {
    const gpa = testing.allocator;
    const code = try module(gpa, "",
        \\%loc = OpVariable %pff Function
        \\%p0 = OpAccessChain %pf %b0 %u0 %u0
        \\%v = OpLoad %f32 %p0
        \\OpStore %loc %v
    );
    defer gpa.free(code);
    const a = try bindingAccess(gpa, code);
    try testing.expect(!a.untracked);
    try testing.expectEqual(@as(u8, 0), a.written);
}

test "device-address pointers make a module untracked" {
    const gpa = testing.allocator;
    const code = try module(gpa,
        \\%ppsb = OpTypePointer 5349 %f32
    , "");
    defer gpa.free(code);
    try testing.expect((try bindingAccess(gpa, code)).untracked);
}

test "bytes that are not SPIR-V are untracked" {
    const junk = [_]u8{0} ** 32;
    const aligned: [32]u8 align(4) = junk;
    try testing.expect((try bindingAccess(testing.allocator, &aligned)).untracked);
}

test "tracker orders read after write, write after read and write after write only" {
    var t: Tracker = .{};
    const a: Range = .{ .buf = 1, .lo = 0, .hi = 100 };
    const a_hi: Range = .{ .buf = 1, .lo = 100, .hi = 200 };
    const b: Range = .{ .buf = 2, .lo = 0 };
    t.record(&.{ .{ .range = a, .write = true }, .{ .range = b, .write = false } });
    // Reading what was written waits; reading what was read does not.
    try testing.expect(t.conflicts(&.{.{ .range = a, .write = false }}));
    try testing.expect(!t.conflicts(&.{.{ .range = b, .write = false }}));
    // Writing what was read waits; a disjoint range of the same buffer does not.
    try testing.expect(t.conflicts(&.{.{ .range = b, .write = true }}));
    try testing.expect(!t.conflicts(&.{.{ .range = a_hi, .write = true }}));
    try testing.expect(t.conflicts(&.{.{ .range = .{ .buf = 1, .lo = 50 }, .write = true }}));
    t.clear();
    try testing.expect(!t.pending());
    try testing.expect(!t.conflicts(&.{.{ .range = a, .write = true }}));
}

test "a full tracker poisons rather than forgetting an access" {
    var t: Tracker = .{};
    for (0..Tracker.cap + 1) |i| t.record(&.{.{ .range = .{ .buf = i + 10, .lo = 0 }, .write = true }});
    try testing.expect(t.poisoned);
    try testing.expect(t.conflicts(&.{.{ .range = .{ .buf = 999, .lo = 0 }, .write = false }}));
}
