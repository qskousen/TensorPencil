//! The Vulkan pipeline cache on disk, and the list of kernels earlier runs bound.
//!
//! Every pipeline compiles through one `VkPipelineCache` loaded from
//! `vk-pipelines-<pipelineCacheUUID>-<module hash>.bin`, so a later run of the same build
//! on the same driver creates its pipelines from the cache. A file per build matters: the
//! driver hands back only what a process used, so two builds sharing one file (the app
//! and its test binary) would each overwrite the other's; `prune` keeps the newest few.
//! The driver's own disk cache does not cover this: it has a ~1 GB cap shared with
//! everything else on the machine.
//! `vk-kernels-used` names every eltwise kernel a run has bound; `Context.init` compiles
//! those on every core before the first forward, and the rest compile on first use.
//!
//! The pipelines `Context.init` builds outside the eltwise table (the coop GEMMs and the
//! attention kernels) each have a file of their own instead, `vk-pipe-<uuid>-<hash of
//! the pipeline's SPIR-V>.bin`: they take minutes to compile on some drivers, and in the
//! per-build file every edit to an unrelated kernel would throw them away.

const std = @import("std");
const builtin = @import("builtin");

extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// `<XDG_CACHE_HOME, HOME/.cache or LOCALAPPDATA>/tensorpencil/<leaf>`, or null when none
/// of them is set.
pub fn path(buf: []u8, leaf: []const u8) ?[]const u8 {
    const sep = std.Io.Dir.path.sep_str;
    if (getenv("XDG_CACHE_HOME")) |p| return std.mem.print(buf, "{s}" ++ sep ++ "tensorpencil" ++ sep ++ "{s}", .{ std.mem.sliceTo(p, 0), leaf }) catch null;
    if (builtin.target.os.tag == .windows) {
        if (getenv("LOCALAPPDATA")) |p| return std.mem.print(buf, "{s}" ++ sep ++ "tensorpencil" ++ sep ++ "{s}", .{ std.mem.sliceTo(p, 0), leaf }) catch null;
    }
    if (getenv("HOME")) |p| return std.mem.print(buf, "{s}" ++ sep ++ ".cache" ++ sep ++ "tensorpencil" ++ sep ++ "{s}", .{ std.mem.sliceTo(p, 0), leaf }) catch null;
    return null;
}

pub const build_prefix = "vk-pipelines-";
pub const pipe_prefix = "vk-pipe-";

/// The cache file's name for a device's `pipelineCacheUUID` and this build's modules.
pub fn fileName(buf: []u8, uuid: [16]u8, build: u64) []const u8 {
    return std.mem.print(buf, build_prefix ++ "{x}-{x:0>16}.bin", .{ uuid, build }) catch unreachable;
}

/// One pipeline's own cache file, for the device and the pipeline's content hash.
pub fn pipeFileName(buf: []u8, uuid: [16]u8, key: u64) []const u8 {
    return std.mem.print(buf, pipe_prefix ++ "{x}-{x:0>16}.bin", .{ uuid, key }) catch unreachable;
}

/// Delete every `prefix` cache in `dir` but the `keep` most recently written. Best
/// effort: a file that will not go costs disk, not correctness, and one pruned while
/// still in use only costs that pipeline a compile.
pub fn prune(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8, comptime prefix: []const u8, keep: usize) void {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    const F = struct { name: []u8, mtime: i96 };
    var files: std.ArrayList(F) = .empty;
    defer {
        for (files.items) |f| gpa.free(f.name);
        files.deinit(gpa);
    }
    var it = dir.iterate();
    while (it.next(io) catch return) |ent| {
        if (ent.kind != .file or !std.mem.startsWith(u8, ent.name, prefix) or !std.mem.endsWith(u8, ent.name, ".bin")) continue;
        const st = dir.statFile(io, ent.name, .{}) catch continue;
        const name = gpa.dupe(u8, ent.name) catch return;
        files.append(gpa, .{ .name = name, .mtime = st.mtime.nanoseconds }) catch {
            gpa.free(name);
            return;
        };
    }
    if (files.items.len <= keep) return;
    std.mem.sort(F, files.items, {}, struct {
        fn newer(_: void, a: F, b: F) bool {
            return a.mtime > b.mtime;
        }
    }.newer);
    for (files.items[keep..]) |f| dir.deleteFile(io, f.name) catch {};
}

pub const used_list = "vk-kernels-used";

/// The file's bytes, or null when it is absent or unreadable: no file is the normal
/// first run, never an error.
pub fn read(gpa: std.mem.Allocator, io: std.Io, p: []const u8) ?[]u8 {
    const f = std.Io.Dir.cwd().openFile(io, p, .{}) catch return null;
    defer f.close(io);
    const len = f.length(io) catch return null;
    const buf = gpa.alloc(u8, @intCast(len)) catch return null;
    const n = f.readPositionalAll(io, buf, 0) catch {
        gpa.free(buf);
        return null;
    };
    if (n != buf.len) {
        gpa.free(buf);
        return null;
    }
    return buf;
}

/// Write `data` to `p` through a temp file and a rename, so a concurrent reader (another
/// process starting up) never sees a partial file.
pub fn write(io: std.Io, p: []const u8, data: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    if (std.Io.Dir.path.dirname(p)) |d| try cwd.createDirPath(io, d);
    var tmp_buf: [1024]u8 = undefined;
    const tmp = try std.mem.print(&tmp_buf, "{s}.{d}.tmp", .{ p, std.Thread.getCurrentId() });
    try cwd.writeFile(io, .{ .sub_path = tmp, .data = data });
    errdefer cwd.deleteFile(io, tmp) catch {};
    try std.Io.Dir.rename(cwd, tmp, cwd, p, io);
}

test "path names the tensorpencil cache directory" {
    var buf: [512]u8 = undefined;
    const p = path(&buf, used_list) orelse return error.SkipZigTest;
    try std.testing.expect(std.mem.endsWith(u8, p, "tensorpencil" ++ std.Io.Dir.path.sep_str ++ used_list));
    var nb: [96]u8 = undefined;
    try std.testing.expectEqualStrings("vk-pipelines-000102030405060708090a0b0c0d0e0f-00000000000000ab.bin", fileName(&nb, .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, 0xab));
    try std.testing.expectEqualStrings("vk-pipe-000102030405060708090a0b0c0d0e0f-00000000000000ab.bin", pipeFileName(&nb, .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, 0xab));
}

test "prune keeps the newest caches of its kind and nothing else is touched" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // The per-pipeline file is the oldest: a prune of build files must not take it.
    const names = [_][]const u8{ "vk-pipe-a-9.bin", "vk-pipelines-a-1.bin", "vk-pipelines-a-2.bin", "vk-pipelines-a-3.bin", "vk-kernels-used", "coop_tile" };
    for (names, 0..) |n, i| {
        try tmp.dir.writeFile(io, .{ .sub_path = n, .data = "x" });
        // Distinct mtimes, oldest first.
        const t: std.Io.Timestamp = .{ .nanoseconds = @as(i96, @intCast(i + 1)) * std.time.ns_per_s };
        // Through a handle: `Dir.setTimestamps` panics on Windows.
        const f = try tmp.dir.openFile(io, n, .{ .mode = .read_write });
        defer f.close(io);
        try f.setTimestamps(io, .{ .modify_timestamp = .{ .new = t } });
    }
    var pbuf: [512]u8 = undefined;
    const dir_path = pbuf[0..try tmp.dir.realPath(io, &pbuf)];
    prune(std.testing.allocator, io, dir_path, build_prefix, 2);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, names[1], .{}));
    for ([_][]const u8{ names[0], names[2], names[3], names[4], names[5] }) |n| _ = try tmp.dir.statFile(io, n, .{});
    prune(std.testing.allocator, io, dir_path, pipe_prefix, 0);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, names[0], .{}));
    for (names[2..]) |n| _ = try tmp.dir.statFile(io, n, .{});
}
