//! Raw keyboard input for an interactive terminal: no line buffering, no
//! echo, and Ctrl-C delivered as a byte rather than a signal. POSIX does it
//! with termios; Windows with console modes, where VT input is also switched
//! on so keys and bracketed paste arrive as the same escape sequences a POSIX
//! terminal sends, and VT output so the mode sequences a caller writes are
//! obeyed rather than printed.
const std = @import("std");
const builtin = @import("builtin");

pub const RawTty = switch (builtin.os.tag) {
    .windows => Windows,
    else => Posix,
};

const Posix = struct {
    fd: std.posix.fd_t,
    saved: std.posix.termios,

    pub fn enter(fd: std.posix.fd_t) !Posix {
        const saved = try std.posix.tcgetattr(fd);
        var raw = saved;
        raw.lflag.ICANON = false;
        raw.lflag.ECHO = false;
        raw.lflag.ISIG = false;
        raw.iflag.ICRNL = false;
        raw.iflag.IXON = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 1;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
        try std.posix.tcsetattr(fd, .NOW, raw);
        return .{ .fd = fd, .saved = saved };
    }

    pub fn leave(self: *Posix) void {
        std.posix.tcsetattr(self.fd, .NOW, self.saved) catch {};
    }
};

const Windows = struct {
    in: std.os.windows.HANDLE,
    /// Null when stdout is not a console.
    out: ?std.os.windows.HANDLE,
    in_mode: u32,
    out_mode: u32,
    in_cp: u32,

    const ENABLE_PROCESSED_INPUT = 0x1;
    const ENABLE_LINE_INPUT = 0x2;
    const ENABLE_ECHO_INPUT = 0x4;
    const ENABLE_VIRTUAL_TERMINAL_INPUT = 0x200;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x4;
    const CP_UTF8 = 65001;

    pub fn enter(in: std.os.windows.HANDLE) !Windows {
        const out = std.Io.File.stdout().handle;
        var self: Windows = .{ .in = in, .out = out, .in_mode = 0, .out_mode = 0, .in_cp = GetConsoleCP() };
        if (GetConsoleMode(in, &self.in_mode) == 0) return error.NotATerminal;
        if (GetConsoleMode(out, &self.out_mode) == 0) self.out = null;
        const raw = (self.in_mode & ~@as(u32, ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT)) |
            ENABLE_VIRTUAL_TERMINAL_INPUT;
        if (SetConsoleMode(in, raw) == 0) return error.NotATerminal;
        if (self.out) |o| _ = SetConsoleMode(o, self.out_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        // The editor takes UTF-8 bytes; the console hands out its input code page.
        _ = SetConsoleCP(CP_UTF8);
        return self;
    }

    pub fn leave(self: *Windows) void {
        _ = SetConsoleCP(self.in_cp);
        _ = SetConsoleMode(self.in, self.in_mode);
        if (self.out) |o| _ = SetConsoleMode(o, self.out_mode);
    }

    extern "kernel32" fn GetConsoleMode(h: std.os.windows.HANDLE, mode: *u32) callconv(.winapi) c_int;
    extern "kernel32" fn SetConsoleMode(h: std.os.windows.HANDLE, mode: u32) callconv(.winapi) c_int;
    extern "kernel32" fn GetConsoleCP() callconv(.winapi) u32;
    extern "kernel32" fn SetConsoleCP(cp: u32) callconv(.winapi) c_int;
};
