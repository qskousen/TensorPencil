//! Prints the source of the build's `version` module: `<package version> (<git describe>)`,
//! the git half `unknown` without a repository or a git binary. build.zig runs it on
//! every build; an unchanged stamp is an unchanged file and recompiles nothing.
//!
//! usage: version_stamp <package version> <repository directory>

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) return error.InvalidArgs;
    const describe = blk: {
        const r = std.process.run(arena, io, .{
            .argv = &.{ "git", "-C", args[2], "describe", "--tags", "--always", "--dirty", "--abbrev=7" },
        }) catch break :blk "unknown";
        if (!r.term.success()) break :blk "unknown";
        break :blk std.mem.trim(u8, r.stdout, " \t\r\n");
    };
    var buf: [512]u8 = undefined;
    var w: std.Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    try w.interface.print("pub const string = \"{s} ({s})\";\n", .{ args[1], describe });
    try w.interface.flush();
}
