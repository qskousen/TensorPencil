//! Runtime library loading for the GPU backends and NVML.
//!
//! `std.DynLib` has no Windows arm in 0.16, so this wraps it and supplies one.
//! The surface is deliberately the same three calls the backends already used
//! (`open` / `lookup` / `close`), plus `openFirst`, because every caller has a
//! list of names to try: a soname, an unversioned fallback, sometimes an
//! absolute path a distro puts the library at.

const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

pub const Error = error{LibraryNotFound};

/// Longest library name or path `open` accepts. Callers pass sonames and a few
/// fixed absolute paths, not user input.
const max_name = 512;

pub const Lib = struct {
    inner: Inner,

    const Inner = if (builtin.os.tag == .windows) windows.HMODULE else std.DynLib;

    pub fn open(name: []const u8) Error!Lib {
        if (builtin.os.tag == .windows) {
            var buf: [max_name]u16 = undefined;
            const n = std.unicode.utf8ToUtf16Le(buf[0 .. buf.len - 1], name) catch
                return error.LibraryNotFound;
            buf[n] = 0;
            const h = LoadLibraryW(buf[0..n :0]) orelse return error.LibraryNotFound;
            return .{ .inner = h };
        }
        return .{ .inner = std.DynLib.open(name) catch return error.LibraryNotFound };
    }

    /// The first name that loads, or null if none does.
    pub fn openFirst(names: []const []const u8) ?Lib {
        for (names) |n| {
            if (open(n)) |lib| return lib else |_| {}
        }
        return null;
    }

    pub fn lookup(self: *Lib, comptime T: type, name: [:0]const u8) ?T {
        if (builtin.os.tag == .windows) {
            const p = GetProcAddress(self.inner, name.ptr) orelse return null;
            return @constCast(@ptrCast(@alignCast(p)));
        }
        return self.inner.lookup(T, name);
    }

    pub fn close(self: *Lib) void {
        if (builtin.os.tag == .windows) {
            _ = FreeLibrary(self.inner);
            return;
        }
        self.inner.close();
    }
};

/// Convenience wrapper so a caller can write `dynlib.openFirst(&.{...})`.
pub fn openFirst(names: []const []const u8) ?Lib {
    return Lib.openFirst(names);
}

extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) callconv(.winapi) ?windows.HMODULE;
extern "kernel32" fn GetProcAddress(hModule: windows.HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?windows.FARPROC;
extern "kernel32" fn FreeLibrary(hLibModule: windows.HMODULE) callconv(.winapi) windows.BOOL;

test "openFirst returns null when no candidate exists" {
    try std.testing.expect(openFirst(&.{ "libnothing-here.so.99", "nothing-here.dll" }) == null);
}
