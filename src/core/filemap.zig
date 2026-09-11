//! Read-only whole-file mapping: the one place either checkpoint reader calls
//! the OS. posix maps with `mmap`; Windows needs a section object plus a view of
//! it, and the section handle is closed immediately because the view holds its
//! own reference.
//!
//! `willNeed` is advisory on both. It exists because a cold multi-GB checkpoint
//! otherwise faults in at page granularity on first touch; `safetensors.ReadMode`
//! documents what that costs.

const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

pub const Error = error{MapFailed};

/// A whole-file read-only mapping. `page_size_min` is what posix `mmap` returns
/// and what a Windows view exceeds (allocation granularity is 64 KiB).
pub const Mapping = []align(std.heap.page_size_min) const u8;

pub fn map(handle: std.Io.File.Handle, len: usize) Error!Mapping {
    if (builtin.os.tag == .windows) {
        const section = CreateFileMappingW(handle, null, PAGE_READONLY, 0, 0, null) orelse
            return error.MapFailed;
        // The view keeps the section alive, so nothing needs the handle after this.
        defer windows.CloseHandle(section);
        const view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0) orelse return error.MapFailed;
        const p: [*]align(std.heap.page_size_min) const u8 = @ptrCast(@alignCast(view));
        return p[0..len];
    }
    return std.posix.mmap(
        null,
        len,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        handle,
        0,
    ) catch error.MapFailed;
}

pub fn unmap(m: Mapping) void {
    if (builtin.os.tag == .windows) {
        _ = UnmapViewOfFile(@ptrCast(m.ptr));
        return;
    }
    std.posix.munmap(@alignCast(m));
}

/// Ask the OS to read the whole mapping in now, in large requests, rather than
/// faulting it a page at a time. Best-effort: failure is not reported anywhere.
pub fn willNeed(m: Mapping) void {
    if (builtin.os.tag == .windows) {
        var range = [1]MemoryRangeEntry{.{ .base = @constCast(@ptrCast(m.ptr)), .len = m.len }};
        // Windows 8+; an older loader without it just faults as before.
        _ = PrefetchVirtualMemory(current_process, 1, &range, 0);
        return;
    }
    std.posix.madvise(@constCast(m.ptr), m.len, std.posix.MADV.WILLNEED) catch {};
}

// ---- Windows section/view API (absent from std.os.windows) -----------------

const PAGE_READONLY: windows.DWORD = 0x02;
const FILE_MAP_READ: windows.DWORD = 4;
/// GetCurrentProcess()'s pseudo-handle, which is a constant, not a call.
const current_process: windows.HANDLE = @ptrFromInt(std.math.maxInt(usize));

const MemoryRangeEntry = extern struct { base: windows.PVOID, len: usize };

extern "kernel32" fn CreateFileMappingW(
    hFile: windows.HANDLE,
    lpAttributes: ?*anyopaque,
    flProtect: windows.DWORD,
    dwMaximumSizeHigh: windows.DWORD,
    dwMaximumSizeLow: windows.DWORD,
    lpName: ?[*:0]const u16,
) callconv(.winapi) ?windows.HANDLE;

extern "kernel32" fn MapViewOfFile(
    hFileMappingObject: windows.HANDLE,
    dwDesiredAccess: windows.DWORD,
    dwFileOffsetHigh: windows.DWORD,
    dwFileOffsetLow: windows.DWORD,
    dwNumberOfBytesToMap: usize,
) callconv(.winapi) ?windows.LPVOID;

extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: windows.LPCVOID) callconv(.winapi) windows.BOOL;

extern "kernel32" fn PrefetchVirtualMemory(
    hProcess: windows.HANDLE,
    NumberOfEntries: usize,
    VirtualAddresses: [*]MemoryRangeEntry,
    Flags: windows.ULONG,
) callconv(.winapi) windows.BOOL;

test "map, read back and unmap a temp file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const body = "safetensors would go here";
    try tmp.dir.writeFile(io, .{ .sub_path = "m.bin", .data = body });

    const f = try tmp.dir.openFile(io, "m.bin", .{ .mode = .read_only });
    defer f.close(io);
    const m = try map(f.handle, body.len);
    defer unmap(m);
    willNeed(m);
    try std.testing.expectEqualStrings(body, m);
}
