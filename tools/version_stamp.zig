//! Prints the source of the build's `version` module: `git describe`, so a release is
//! named by its tag alone, or `unknown` without a repository or a git binary. build.zig
//! runs it on every build; an unchanged stamp is an unchanged file and recompiles nothing.
//!
//! usage: version_stamp <repository directory>

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 2) return error.InvalidArgs;
    const describe = blk: {
        const r = std.process.run(arena, io, .{
            .argv = &.{ "git", "-C", args[1], "describe", "--tags", "--always", "--dirty", "--abbrev=7" },
        }) catch break :blk "unknown";
        if (!r.term.success()) break :blk "unknown";
        break :blk std.mem.trim(u8, r.stdout, " \t\r\n");
    };
    var buf: [512]u8 = undefined;
    var w: std.Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    try w.interface.print("pub const string = \"{s}\";\n", .{describe});
    try w.interface.flush();
}
