//! Minimal PTX text-assembly helper — the PTX analog of `coopmat.zig`'s
//! word-level SPIR-V `Asm` emitter.
//!
//! PTX is human-readable assembly (unlike SPIR-V's binary word stream), so this
//! is deliberately thin: a growing text buffer plus register-file and label
//! management. Kernels are authored as PTX through this builder, which removes
//! the boilerplate of counting `.reg` declarations, generating fresh labels, and
//! unrolling repetitive instruction sequences (fragment loads, MMA grids).
//!
//! Register model: PTX registers are typed at declaration but instructions may
//! reinterpret freely (a `.b32` register is used by `add.s32`, `mul.lo.u32`,
//! etc.). We therefore expose one counter per *storage class* and let callers
//! pick the class by width:
//!   .b32 -> %r    (u32/s32/b32/packed-s8x4/f32-as-bits)
//!   .b64 -> %rd   (pointers, wide math)
//!   .f32 -> %f
//!   .b16 -> %rs
//!   .pred-> %p
//! `reg(.b32)` returns an owned register name like "%r7" and bumps the count so
//! the final module declares exactly `.reg .b32 %r<8>;`.

const std = @import("std");

pub const RegClass = enum { b32, b64, f32, b16, pred };

fn prefix(c: RegClass) []const u8 {
    return switch (c) {
        .b32 => "%r",
        .b64 => "%rd",
        .f32 => "%f",
        .b16 => "%rs",
        .pred => "%p",
    };
}

fn decl(c: RegClass) []const u8 {
    return switch (c) {
        .b32 => ".b32",
        .b64 => ".b64",
        .f32 => ".f32",
        .b16 => ".b16",
        .pred => ".pred",
    };
}

pub const Builder = struct {
    gpa: std.mem.Allocator,
    /// Instruction stream (everything between the register decls and `ret;`).
    body: std.ArrayList(u8) = .empty,
    /// Owned register-name strings, freed on deinit.
    owned: std.ArrayList([]u8) = .empty,
    /// Owned register-name slices handed out by regs(), freed on deinit.
    owned_slices: std.ArrayList([][]const u8) = .empty,
    /// Owned label strings.
    counts: [5]u32 = .{ 0, 0, 0, 0, 0 },
    label_next: u32 = 0,

    pub fn init(gpa: std.mem.Allocator) Builder {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Builder) void {
        for (self.owned.items) |s| self.gpa.free(s);
        self.owned.deinit(self.gpa);
        for (self.owned_slices.items) |s| self.gpa.free(s);
        self.owned_slices.deinit(self.gpa);
        self.body.deinit(self.gpa);
    }

    /// Allocate a fresh virtual register of the given class; returns its PTX name
    /// (e.g. "%r5"). The name is owned by the builder (valid until deinit).
    pub fn reg(self: *Builder, c: RegClass) ![]const u8 {
        const idx = self.counts[@intFromEnum(c)];
        self.counts[@intFromEnum(c)] = idx + 1;
        const name = try std.fmt.allocPrint(self.gpa, "{s}{d}", .{ prefix(c), idx });
        try self.owned.append(self.gpa, name);
        return name;
    }

    /// Allocate `n` consecutive registers of a class, returned as a slice of
    /// names. Useful for MMA fragment register vectors.
    pub fn regs(self: *Builder, c: RegClass, n: usize) ![]const []const u8 {
        const out = try self.gpa.alloc([]const u8, n);
        errdefer self.gpa.free(out);
        for (out) |*r| r.* = try self.reg(c);
        try self.owned_slices.append(self.gpa, out);
        return out;
    }

    /// Fresh unique label name (no leading `%`); emit it with `label()`.
    pub fn newLabel(self: *Builder, comptime tag: []const u8) ![]const u8 {
        const name = try std.fmt.allocPrint(self.gpa, "L_{s}_{d}", .{ tag, self.label_next });
        self.label_next += 1;
        try self.owned.append(self.gpa, name);
        return name;
    }

    /// Append a raw line of PTX (a trailing newline is added).
    pub fn line(self: *Builder, text: []const u8) !void {
        try self.body.appendSlice(self.gpa, "\t");
        try self.body.appendSlice(self.gpa, text);
        try self.body.appendSlice(self.gpa, "\n");
    }

    /// Append a formatted line of PTX (tab-indented, newline-terminated).
    pub fn linef(self: *Builder, comptime fmt: []const u8, args: anytype) !void {
        try self.body.appendSlice(self.gpa, "\t");
        try self.body.print(self.gpa, fmt, args);
        try self.body.appendSlice(self.gpa, "\n");
    }

    /// Emit a label at column 0: "<name>:".
    pub fn label(self: *Builder, name: []const u8) !void {
        try self.body.print(self.gpa, "{s}:\n", .{name});
    }

    /// Emit a comment line.
    pub fn comment(self: *Builder, comptime fmt: []const u8, args: anytype) !void {
        try self.body.appendSlice(self.gpa, "\t// ");
        try self.body.print(self.gpa, fmt, args);
        try self.body.appendSlice(self.gpa, "\n");
    }

    /// Join a slice of register names into a "{a, b, c}" vector operand.
    /// Returned string is owned by the builder.
    pub fn vec(self: *Builder, names: []const []const u8) ![]const u8 {
        var s: std.ArrayList(u8) = .empty;
        defer s.deinit(self.gpa);
        try s.append(self.gpa, '{');
        for (names, 0..) |n, i| {
            if (i != 0) try s.appendSlice(self.gpa, ", ");
            try s.appendSlice(self.gpa, n);
        }
        try s.append(self.gpa, '}');
        const out = try s.toOwnedSlice(self.gpa);
        try self.owned.append(self.gpa, out);
        return out;
    }

    /// Assemble the full module text: preamble, entry signature, register-file
    /// declarations (only for classes that were used), the accumulated body, and
    /// a trailing `ret; }`. `params` is the parenthesised `.param` list contents
    /// (without the surrounding parens), e.g. ".param .u64 p0, .param .u32 p1".
    /// `shared_decls` is any module-scope declarations to emit before the entry
    /// (e.g. `.extern .shared .align 16 .b8 dyn[];` or fixed `.shared` arrays).
    /// Caller owns the returned bytes.
    pub fn build(
        self: *Builder,
        entry_name: []const u8,
        params: []const u8,
        shared_decls: []const u8,
    ) ![:0]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.gpa);
        const w = &out;
        const g = self.gpa;

        try w.print(g,
            \\.version 8.0
            \\.target sm_86
            \\.address_size 64
            \\
        , .{});
        if (shared_decls.len != 0) {
            try w.appendSlice(g, shared_decls);
            try w.append(g, '\n');
        }
        try w.print(g, ".visible .entry {s}(\n", .{entry_name});
        try w.appendSlice(g, params);
        try w.appendSlice(g, "\n)\n{\n");

        // Register-file declarations (skip empty classes).
        inline for ([_]RegClass{ .pred, .b16, .b32, .b64, .f32 }) |c| {
            const n = self.counts[@intFromEnum(c)];
            if (n != 0) try w.print(g, "\t.reg {s} {s}<{d}>;\n", .{ decl(c), prefix(c), n });
        }
        try w.append(g, '\n');
        try w.appendSlice(g, self.body.items);
        try w.appendSlice(g, "\tret;\n}\n");
        return try out.toOwnedSliceSentinel(g, 0);
    }
};

// -------------------------------------------------------------------------
// PTX preamble constants shared by hand-authored kernels.
// -------------------------------------------------------------------------

/// The `.version` string nvcc 13.0 emits for sm_86; 8.0 covers everything we use
/// (mma.m16n8k32, cp.async, ldmatrix). The driver JIT accepts >= its ISA support.
pub const version = "8.0";
pub const target = "sm_86";

test "ptx builder emits a valid vector-add module" {
    const gpa = std.testing.allocator;
    var b = Builder.init(gpa);
    defer b.deinit();

    const a_ptr = try b.reg(.b64);
    const idx = try b.reg(.b32);
    const p = try b.reg(.pred);
    _ = p;

    try b.linef("ld.param.u64 {s}, [p_a];", .{a_ptr});
    try b.linef("mov.u32 {s}, %tid.x;", .{idx});

    const mod = try b.build("vadd", "\t.param .u64 p_a", "");
    defer gpa.free(mod);

    // Sanity: declarations present, entry present, closes cleanly.
    try std.testing.expect(std.mem.indexOf(u8, mod, ".visible .entry vadd(") != null);
    try std.testing.expect(std.mem.indexOf(u8, mod, ".reg .b64 %rd<1>;") != null);
    try std.testing.expect(std.mem.indexOf(u8, mod, ".reg .b32 %r<1>;") != null);
    try std.testing.expect(std.mem.indexOf(u8, mod, ".reg .pred %p<1>;") != null);
    try std.testing.expect(std.mem.endsWith(u8, mod, "\tret;\n}\n"));
}
