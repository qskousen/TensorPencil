//! Raw-mode line editor for tp-llm's interactive chat.
//!
//! The cooked-mode line reader treats every '\n' as a submit, so pasting a
//! multi-line message fires one turn per line, and there is no way to type
//! a newline without sending. This editor reads single bytes with the tty
//! in raw mode and understands:
//!
//! - Enter (CR/LF)                      -> submit the message
//! - Shift-Enter (kitty CSI 13;2u — any modified 13u) or Alt-Enter (ESC CR)
//!                                      -> insert a newline, keep editing
//! - bracketed paste (CSI 200~ .. 201~) -> pasted newlines are literal;
//!                                         nothing submits during a paste
//! - Backspace                          -> delete within the current line
//! - Ctrl-C                             -> clear the message (cancel)
//! - Ctrl-D on an empty message         -> end of session
//! - other CSI/SS3 sequences (arrows..) -> swallowed
//!
//! The `Editor` is a pure byte-fed state machine (echo goes to a caller
//! writer), so it is unit-testable without a terminal; `RawTty` holds the
//! termios save/restore. The caller enables the terminal modes around each
//! read (bracketed paste `CSI ?2004 h/l`, kitty keyboard `CSI >1u`/`CSI <u`
//! — unsupported terminals ignore both; Alt-Enter is the newline fallback
//! there) so the tty is cooked again while the model generates and Ctrl-C
//! keeps killing a running generation.

const std = @import("std");

/// Emit before reading a message (bracketed paste on, kitty disambiguate
/// push) and its undo, emitted when the read returns.
pub const enter_seq = "\x1b[?2004h\x1b[>1u";
pub const leave_seq = "\x1b[<u\x1b[?2004l";

pub const Event = enum {
    /// Byte consumed; keep feeding.
    none,
    /// Enter outside a paste: message() is complete.
    submit,
    /// Ctrl-C: message cleared; caller reprompts.
    cancel,
    /// Ctrl-D on an empty message: end the session.
    eof,
};

pub const Editor = struct {
    buf: std.ArrayListUnmanaged(u8) = .empty,
    state: State = .normal,
    /// CSI parameter bytes ("13;2" of CSI 13;2u); overflow abandons the seq.
    csi: [15]u8 = undefined,
    csi_len: usize = 0,
    /// Matched prefix of the paste terminator "\x1b[201~".
    paste_match: usize = 0,

    const State = enum { normal, esc, csi, paste };
    const paste_end = "\x1b[201~";

    pub fn deinit(self: *Editor, gpa: std.mem.Allocator) void {
        self.buf.deinit(gpa);
        self.* = undefined;
    }

    /// The message accumulated so far (valid until the next feed/reset).
    pub fn message(self: *const Editor) []const u8 {
        return self.buf.items;
    }

    pub fn reset(self: *Editor) void {
        self.buf.clearRetainingCapacity();
        self.state = .normal;
        self.csi_len = 0;
        self.paste_match = 0;
    }

    /// Consume one input byte, echoing through `echo` (the caller flushes).
    pub fn feed(self: *Editor, gpa: std.mem.Allocator, byte: u8, echo: *std.Io.Writer) !Event {
        switch (self.state) {
            .normal => switch (byte) {
                '\r', '\n' => {
                    try echo.writeAll("\r\n");
                    return .submit;
                },
                0x7f, 0x08 => { // backspace: within the current line only
                    if (self.buf.items.len > 0 and self.buf.items[self.buf.items.len - 1] != '\n') {
                        // pop one UTF-8 codepoint (continuation bytes 10xxxxxx)
                        var n: usize = self.buf.items.len - 1;
                        while (n > 0 and self.buf.items[n] & 0xc0 == 0x80) n -= 1;
                        self.buf.shrinkRetainingCapacity(n);
                        try echo.writeAll("\x08 \x08");
                    }
                },
                0x03 => { // Ctrl-C: cancel the message
                    self.buf.clearRetainingCapacity();
                    try echo.writeAll("^C\r\n");
                    return .cancel;
                },
                0x04 => { // Ctrl-D: EOF only on an empty message
                    if (self.buf.items.len == 0) return .eof;
                },
                0x1b => self.state = .esc,
                '\t' => {
                    try self.buf.append(gpa, byte);
                    try echo.writeByte(byte);
                },
                else => if (byte >= 0x20) {
                    try self.buf.append(gpa, byte);
                    try echo.writeByte(byte);
                },
            },
            .esc => switch (byte) {
                '[' => {
                    self.state = .csi;
                    self.csi_len = 0;
                },
                '\r', '\n' => { // Alt-Enter: newline without submit
                    self.state = .normal;
                    try self.newline(gpa, echo);
                },
                'O' => self.state = .csi, // SS3 (F-keys): swallow like CSI
                else => self.state = .normal, // other alt-key: ignore
            },
            .csi => {
                if (byte >= 0x40 and byte <= 0x7e) { // final byte
                    self.state = .normal;
                    const params = self.csi[0..self.csi_len];
                    if (byte == '~' and std.mem.eql(u8, params, "200")) {
                        self.state = .paste;
                        self.paste_match = 0;
                    } else if (byte == 'u' and std.mem.startsWith(u8, params, "13;")) {
                        try self.newline(gpa, echo); // modified Enter (kitty)
                    } else if (byte == 'u' and std.mem.eql(u8, params, "13")) {
                        try echo.writeAll("\r\n");
                        return .submit;
                    }
                    // anything else (arrows, home/end, ...): swallowed
                } else if (self.csi_len < self.csi.len) {
                    self.csi[self.csi_len] = byte;
                    self.csi_len += 1;
                } else {
                    self.state = .normal; // runaway sequence
                }
            },
            .paste => {
                if (byte == paste_end[self.paste_match]) {
                    self.paste_match += 1;
                    if (self.paste_match == paste_end.len) {
                        self.state = .normal;
                        self.paste_match = 0;
                    }
                    return .none;
                }
                // mismatch: the held-back prefix was paste data after all
                if (self.paste_match > 0) {
                    for (paste_end[0..self.paste_match]) |b| try self.pasteByte(gpa, b, echo);
                    self.paste_match = 0;
                }
                if (byte == paste_end[0]) {
                    self.paste_match = 1;
                } else {
                    try self.pasteByte(gpa, byte, echo);
                }
            },
        }
        return .none;
    }

    /// One byte of paste payload: newlines are literal (CR / CRLF -> '\n'),
    /// other control bytes are dropped.
    fn pasteByte(self: *Editor, gpa: std.mem.Allocator, byte: u8, echo: *std.Io.Writer) !void {
        switch (byte) {
            '\r' => try self.newline(gpa, echo),
            '\n' => { // CRLF: the CR already inserted the newline
                if (self.buf.items.len == 0 or self.buf.items[self.buf.items.len - 1] != '\n')
                    try self.newline(gpa, echo);
            },
            '\t' => {
                try self.buf.append(gpa, byte);
                try echo.writeByte(byte);
            },
            else => if (byte >= 0x20) {
                try self.buf.append(gpa, byte);
                try echo.writeByte(byte);
            },
        }
    }

    fn newline(self: *Editor, gpa: std.mem.Allocator, echo: *std.Io.Writer) !void {
        try self.buf.append(gpa, '\n');
        try echo.writeAll("\r\n");
    }
};

/// Termios raw-mode guard: no canonical buffering, no echo (the Editor
/// echoes), no ISIG (Ctrl-C is a byte; the editor turns it into a cancel
/// so the tty can't die un-restored mid-input).
pub const RawTty = struct {
    fd: std.posix.fd_t,
    saved: std.posix.termios,

    pub fn enter(fd: std.posix.fd_t) !RawTty {
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

    pub fn leave(self: *RawTty) void {
        std.posix.tcsetattr(self.fd, .NOW, self.saved) catch {};
    }
};

// --- tests -----------------------------------------------------------------

const TestFeed = struct {
    ed: Editor = .{},
    echo_buf: [4096]u8 = undefined,
    echo: std.Io.Writer = undefined,

    fn init(self: *TestFeed) void {
        self.echo = std.Io.Writer.fixed(&self.echo_buf);
    }

    fn deinit(self: *TestFeed) void {
        self.ed.deinit(std.testing.allocator);
    }

    /// Feed bytes expecting no submit/eof mid-stream; returns the last event.
    fn feedAll(self: *TestFeed, bytes: []const u8) !Event {
        var last: Event = .none;
        for (bytes, 0..) |b, i| {
            last = try self.ed.feed(std.testing.allocator, b, &self.echo);
            if (last != .none and i + 1 < bytes.len) return error.EarlyEvent;
        }
        return last;
    }

    fn echoed(self: *const TestFeed) []const u8 {
        return self.echo.buffered();
    }
};

test "enter submits, backspace edits" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    try std.testing.expectEqual(Event.submit, try t.feedAll("hix\x7f\r"));
    try std.testing.expectEqualStrings("hi", t.ed.message());
    try std.testing.expectEqualStrings("hix\x08 \x08\r\n", t.echoed());
}

test "shift-enter and alt-enter insert newlines without submitting" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    // kitty Shift-Enter, then Alt-Enter, then plain Enter submits.
    try std.testing.expectEqual(Event.submit, try t.feedAll("a\x1b[13;2ub\x1b\rc\r"));
    try std.testing.expectEqualStrings("a\nb\nc", t.ed.message());
}

test "bracketed paste keeps newlines literal and never submits" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    // Paste "one\r\ntwo\nthree" (mixed line endings), then Enter to send.
    try std.testing.expectEqual(Event.none, try t.feedAll("\x1b[200~one\r\ntwo\nthree\x1b[201~"));
    try std.testing.expectEqual(Event.submit, try t.feedAll("!\r"));
    try std.testing.expectEqualStrings("one\ntwo\nthree!", t.ed.message());
}

test "paste containing a lone ESC is data, terminator still found" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    try std.testing.expectEqual(Event.none, try t.feedAll("\x1b[200~a\x1b[2b\x1b[201~"));
    try std.testing.expectEqual(Event.submit, try t.feedAll("\r"));
    // "\x1b[2" is a dead prefix of the terminator: flushed as data minus the
    // dropped control ESC byte.
    try std.testing.expectEqualStrings("a[2b", t.ed.message());
}

test "backspace stops at a newline boundary" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    try std.testing.expectEqual(Event.submit, try t.feedAll("a\x1b\rb\x7f\x7f\x7f\r"));
    try std.testing.expectEqualStrings("a\n", t.ed.message());
}

test "utf-8 backspace removes one codepoint" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    try std.testing.expectEqual(Event.submit, try t.feedAll("é\x7f\r")); // C3 A9
    try std.testing.expectEqualStrings("", t.ed.message());
}

test "ctrl-c cancels, ctrl-d is eof only when empty" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    var ev = try t.ed.feed(std.testing.allocator, 'x', &t.echo);
    try std.testing.expectEqual(Event.none, ev);
    ev = try t.ed.feed(std.testing.allocator, 0x04, &t.echo); // ignored: not empty
    try std.testing.expectEqual(Event.none, ev);
    ev = try t.ed.feed(std.testing.allocator, 0x03, &t.echo);
    try std.testing.expectEqual(Event.cancel, ev);
    try std.testing.expectEqualStrings("", t.ed.message());
    ev = try t.ed.feed(std.testing.allocator, 0x04, &t.echo);
    try std.testing.expectEqual(Event.eof, ev);
}

test "arrow keys and other sequences are swallowed" {
    var t: TestFeed = .{};
    t.init();
    defer t.deinit();
    try std.testing.expectEqual(Event.submit, try t.feedAll("a\x1b[Ab\x1b[1;5Cc\x1bOPd\r"));
    try std.testing.expectEqualStrings("abcd", t.ed.message());
}
