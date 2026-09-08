//! Rewrites the LLVM IR Zig emits for an nvptx64 object so the NVPTX backend
//! accepts it. Zig 0.16 exports a kernel as a private function under its
//! mangled name plus an alias carrying the export name, and NVPTX refuses an
//! alias whose target is a kernel. This drops each alias and renames the
//! function to the exported name. Usage: ptx_unalias in.ll out.ll
//!
//! Build wiring: build.zig runs `zig build-obj -femit-llvm-ir`, this, then
//! `zig cc -target nvptx64-cuda -S` to get the PTX that gets embedded.

const std = @import("std");

const Alias = struct { exported: []const u8, internal: []const u8 };

/// `@name = alias <type>, ptr @internal` -> the two names, or null for any other line.
fn parseAlias(line: []const u8) ?Alias {
    if (line.len == 0 or line[0] != '@') return null;
    const eq = std.mem.indexOf(u8, line, " = alias ") orelse return null;
    const comma = std.mem.lastIndexOfScalar(u8, line, ',') orelse return null;
    const tail = std.mem.trim(u8, line[comma + 1 ..], " ");
    if (!std.mem.startsWith(u8, tail, "ptr @")) return null;
    return .{ .exported = line[0..eq], .internal = tail["ptr ".len..] };
}

pub fn rewrite(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    var aliases: std.ArrayList(Alias) = .empty;
    defer aliases.deinit(gpa);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);

    var lines = std.mem.splitScalar(u8, src, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (parseAlias(line)) |al| {
            // NVPTX has no kernel aliases, so one body cannot carry two entry names.
            for (aliases.items) |seen| if (std.mem.eql(u8, seen.internal, al.internal)) return error.KernelExportedTwice;
            try aliases.append(gpa, al);
            continue;
        }
        if (!first) try out.append(gpa, '\n');
        first = false;
        try out.appendSlice(gpa, line);
    }

    var cur = try gpa.dupe(u8, out.items);
    errdefer gpa.free(cur);
    for (aliases.items) |al| {
        const defn = try std.fmt.allocPrint(gpa, "define private ptx_kernel void {s}(", .{al.internal});
        defer gpa.free(defn);
        const defn_new = try std.fmt.allocPrint(gpa, "define ptx_kernel void {s}(", .{al.exported});
        defer gpa.free(defn_new);
        if (std.mem.indexOf(u8, cur, defn) == null) return error.KernelNotFound;
        const a = try std.mem.replaceOwned(u8, gpa, cur, defn, defn_new);
        gpa.free(cur);
        cur = a;
        // Any remaining reference (a call, a metadata node) follows the rename.
        const b = try std.mem.replaceOwned(u8, gpa, cur, al.internal, al.exported);
        gpa.free(cur);
        cur = b;
    }
    return cur;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) {
        std.debug.print("usage: ptx_unalias in.ll out.ll\n", .{});
        return error.BadArgs;
    }
    const io = init.io;
    const src = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(256 << 20));
    const out = try rewrite(arena, src);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out });
}

test "alias to a kernel becomes a renamed definition" {
    const src =
        \\; ModuleID = 'dual'
        \\@add = alias void (ptr addrspace(1), i32), ptr @"dual.Entry((function 'add')).ptx"
        \\@mul = alias void (ptr addrspace(1), i32), ptr @dual.mul
        \\
        \\define private ptx_kernel void @"dual.Entry((function 'add')).ptx"(ptr addrspace(1) %0, i32 %1) unnamed_addr #0 {
        \\  ret void
        \\}
        \\define private ptx_kernel void @dual.mul(ptr addrspace(1) %0, i32 %1) unnamed_addr #0 {
        \\  ret void
        \\}
        \\!0 = !{ptr @dual.mul}
        \\
    ;
    const got = try rewrite(std.testing.allocator, src);
    defer std.testing.allocator.free(got);
    const want =
        \\; ModuleID = 'dual'
        \\
        \\define ptx_kernel void @add(ptr addrspace(1) %0, i32 %1) unnamed_addr #0 {
        \\  ret void
        \\}
        \\define ptx_kernel void @mul(ptr addrspace(1) %0, i32 %1) unnamed_addr #0 {
        \\  ret void
        \\}
        \\!0 = !{ptr @mul}
        \\
    ;
    try std.testing.expectEqualStrings(want, got);
}

test "two exports of one kernel is an error" {
    const src = "@x = alias void (), ptr @m.k\n@y = alias void (), ptr @m.k\ndefine private ptx_kernel void @m.k() {\n  ret void\n}\n";
    try std.testing.expectError(error.KernelExportedTwice, rewrite(std.testing.allocator, src));
}

test "an alias without a kernel definition is an error" {
    const src = "@x = alias void (), ptr @m.x\ndefine void @m.x() {\n  ret void\n}\n";
    try std.testing.expectError(error.KernelNotFound, rewrite(std.testing.allocator, src));
}
