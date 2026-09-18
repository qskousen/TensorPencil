//! Free space on the filesystem holding a path: what a host checks before it
//! accepts a multi-GB model. std wraps no statfs on any OS, so this is the one
//! place that asks: the raw `statfs` syscall on 64-bit Linux, kernel32 on
//! Windows, and "unknown" (null) elsewhere, which a caller treats as "go ahead".
const std = @import("std");
const builtin = @import("builtin");

/// Bytes an unprivileged writer may still use under `path`, null when the
/// platform has no answer here.
pub fn freeBytes(gpa: std.mem.Allocator, path: []const u8) ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            if (@sizeOf(usize) != 8) return null;
            const z = gpa.dupeZ(u8, path) catch return null;
            defer gpa.free(z);
            var st: LinuxStatfs = undefined;
            // A raw syscall returns -errno in the top range of usize.
            const rc: isize = @bitCast(std.os.linux.syscall2(.statfs, @intFromPtr(z.ptr), @intFromPtr(&st)));
            if (rc < 0) return null;
            if (st.f_bsize <= 0) return null;
            return st.f_bavail * @as(u64, @intCast(st.f_bsize));
        },
        .windows => {
            const w = std.unicode.wtf8ToWtf16LeAllocZ(gpa, path) catch return null;
            defer gpa.free(w);
            var avail: u64 = 0;
            var total: u64 = 0;
            var free: u64 = 0;
            if (GetDiskFreeSpaceExW(w.ptr, &avail, &total, &free) == 0) return null;
            return avail;
        },
        else => return null,
    }
}

/// The kernel's `struct statfs` on 64-bit Linux (x86_64 and aarch64 agree).
const LinuxStatfs = extern struct {
    f_type: i64,
    f_bsize: i64,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_fsid: [2]i32,
    f_namelen: i64,
    f_frsize: i64,
    f_flags: i64,
    f_spare: [4]i64,
};

extern "kernel32" fn GetDiskFreeSpaceExW(
    lpDirectoryName: ?[*:0]const u16,
    lpFreeBytesAvailableToCaller: ?*u64,
    lpTotalNumberOfBytes: ?*u64,
    lpTotalNumberOfFreeBytes: ?*u64,
) callconv(.winapi) c_int;

test "the working directory has a known, non-zero amount of free space on Linux" {
    const free = freeBytes(std.testing.allocator, ".");
    if (builtin.os.tag == .linux) {
        try std.testing.expect(free != null);
        try std.testing.expect(free.? > 0);
    }
    try std.testing.expect(freeBytes(std.testing.allocator, "/nonexistent/tp-diskspace") == null);
}
