//! The image studio's prompt library: every prompt the user has generated from,
//! newest first, grouped by day in the studio's left rail the way conversations
//! are grouped in the chat one.
//!
//! An entry is TEXT ONLY -- the prompt and the negative it was last written
//! against. Not the steps, not the sampler, not the LoRAs: a row restores what
//! you wrote, and every knob stays where you left it. The saved PNG already
//! carries the full recipe in its AUTOMATIC1111 block, and "Open in Studio" is
//! the path that restores one.
//!
//! **The prompt is the identity.** Generating the same prompt twice bumps its
//! timestamp instead of appending a second row, so the list stays a library of
//! distinct prompts rather than a log of presses. The negative stored beside it
//! is whatever was last used with that prompt.
//!
//! One file rather than a directory of them (`history.zig` gives a conversation
//! each): a prompt is a few hundred bytes, and a thousand of them is one 200 KB
//! file against a thousand inodes.
const std = @import("std");
const history = @import("history.zig");

/// Day grouping is the chat sidebar's, reused rather than reimplemented, so the
/// two rails cannot disagree about where "yesterday" ends.
pub const Group = history.Group;
pub const groupOf = history.groupOf;
pub const localOffsetSeconds = history.localOffsetSeconds;

pub const magic = "tp-prompts 1";
pub const file_name = "prompts.tpp";

/// A library longer than this is refused rather than read. A real one is
/// kilobytes; anything at this scale is the wrong file.
pub const max_file_bytes: std.Io.Limit = @enumFromInt(8 << 20);

/// Oldest entries past this are dropped on save, so the file cannot grow without
/// bound on a machine that has been generating for a year.
pub const max_entries: usize = 500;

/// Longest prompt kept. Longer ones are stored truncated at a UTF-8 boundary:
/// the row is a handle for re-running something, and a novel pasted into the box
/// should not make the library unreadable.
pub const max_text: usize = 8 * 1024;

/// One row. `prompt` and `negative` are gpa-owned.
pub const Entry = struct {
    /// Stable across restarts: a hash of the prompt, which IS the identity. The
    /// sidebar needs a u64 row id and this is one that survives a rescan.
    id: u64,
    updated_ms: i64,
    prompt: []u8,
    negative: []u8,
};

pub fn idOf(prompt: []const u8) u64 {
    return std.hash.Wyhash.hash(0x70726f6d, prompt);
}

// ------------------------------------------------------------- serialization

pub fn write(w: *std.Io.Writer, entries: []const Entry) !void {
    try w.print("{s}\n", .{magic});
    for (entries) |e| {
        try w.print("p {d} {d} {d}\n", .{ e.updated_ms, e.prompt.len, e.negative.len });
        try w.writeAll(e.prompt);
        try w.writeAll(e.negative);
        try w.writeAll("\n");
    }
}

pub const ParseError = error{ BadMagic, Truncated, BadHeader } || std.mem.Allocator.Error;

/// Parse the whole library. Entries are gpa-owned; on failure nothing is leaked.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) ParseError![]Entry {
    var i: usize = 0;
    const line = struct {
        fn next(b: []const u8, p: *usize) ParseError![]const u8 {
            if (p.* >= b.len) return error.Truncated;
            const nl = std.mem.indexOfScalarPos(u8, b, p.*, '\n') orelse return error.Truncated;
            const out = b[p.*..nl];
            p.* = nl + 1;
            return out;
        }
    }.next;

    if (!std.mem.eql(u8, try line(bytes, &i), magic)) return error.BadMagic;

    var out: std.ArrayList(Entry) = .empty;
    errdefer {
        for (out.items) |e| {
            gpa.free(e.prompt);
            gpa.free(e.negative);
        }
        out.deinit(gpa);
    }

    while (i < bytes.len) {
        const hdr = line(bytes, &i) catch break;
        if (hdr.len == 0) continue;
        if (!std.mem.startsWith(u8, hdr, "p ")) return error.BadHeader;
        var it = std.mem.tokenizeScalar(u8, hdr["p ".len..], ' ');
        const ms_s = it.next() orelse return error.BadHeader;
        const pl_s = it.next() orelse return error.BadHeader;
        const nl_s = it.next() orelse return error.BadHeader;
        const ms = std.fmt.parseInt(i64, ms_s, 10) catch return error.BadHeader;
        const pl = std.fmt.parseInt(usize, pl_s, 10) catch return error.BadHeader;
        const nl = std.fmt.parseInt(usize, nl_s, 10) catch return error.BadHeader;

        // +1 for the newline the writer puts after the pair.
        if (i + pl + nl + 1 > bytes.len) return error.Truncated;
        const p = try gpa.dupe(u8, bytes[i .. i + pl]);
        errdefer gpa.free(p);
        const n = try gpa.dupe(u8, bytes[i + pl .. i + pl + nl]);
        i += pl + nl + 1;
        try out.append(gpa, .{ .id = idOf(p), .updated_ms = ms, .prompt = p, .negative = n });
    }
    return out.toOwnedSlice(gpa);
}

/// Cut `s` to at most `max_text` bytes without splitting a UTF-8 sequence.
fn clip(s: []const u8) []const u8 {
    if (s.len <= max_text) return s;
    var end = max_text;
    while (end > 0 and (s[end] & 0xC0) == 0x80) end -= 1;
    return s[0..end];
}

// ------------------------------------------------------------------- store

/// The in-memory library, newest first. Owns every entry's text.
pub const Store = struct {
    entries: std.ArrayList(Entry) = .empty,
    /// The library file (gpa-owned), or null when there is no resolvable config
    /// dir. The studio still works; it just does not remember.
    path: ?[]u8 = null,

    pub fn deinit(self: *Store, gpa: std.mem.Allocator) void {
        self.clearEntries(gpa);
        self.entries.deinit(gpa);
        if (self.path) |p| gpa.free(p);
        self.* = .{};
    }

    fn clearEntries(self: *Store, gpa: std.mem.Allocator) void {
        for (self.entries.items) |e| {
            gpa.free(e.prompt);
            gpa.free(e.negative);
        }
        self.entries.clearRetainingCapacity();
    }

    /// Point the store at `dir` (created if absent) and read the library. A
    /// directory we cannot make disables the library rather than failing the app.
    pub fn open(self: *Store, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) void {
        std.Io.Dir.cwd().createDirPath(io, dir) catch |err| {
            std.log.warn("prompts: cannot create {s}: {t} — prompts will not be saved", .{ dir, err });
            return;
        };
        self.path = std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, file_name }) catch return;
        self.rescan(gpa, io);
    }

    /// Re-read the library from disk. A missing file is an empty library, not an
    /// error; a malformed one is reported and left alone rather than overwritten,
    /// so a bad parse does not destroy what it could not read.
    pub fn rescan(self: *Store, gpa: std.mem.Allocator, io: std.Io) void {
        const p = self.path orelse return;
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, max_file_bytes) catch return;
        defer gpa.free(bytes);
        const parsed = parse(gpa, bytes) catch |err| {
            std.log.warn("prompts: cannot parse {s}: {t} — starting empty", .{ p, err });
            return;
        };
        defer gpa.free(parsed);
        self.clearEntries(gpa);
        self.entries.appendSlice(gpa, parsed) catch {
            for (parsed) |e| {
                gpa.free(e.prompt);
                gpa.free(e.negative);
            }
            return;
        };
        std.mem.sort(Entry, self.entries.items, {}, newestFirst);
    }

    fn newestFirst(_: void, a: Entry, b: Entry) bool {
        return a.updated_ms > b.updated_ms;
    }

    /// Record a generated prompt. An existing row for the same prompt is bumped
    /// to `now_ms` and takes the new negative, rather than a duplicate being
    /// appended. An empty prompt is not recorded: it is a legal render (the
    /// unconditional embedding) but not something to keep a row for.
    ///
    /// Returns true when the library changed, so a caller can skip the write.
    pub fn record(self: *Store, gpa: std.mem.Allocator, io: std.Io, now_ms: i64, prompt: []const u8, negative: []const u8) bool {
        const p = clip(std.mem.trim(u8, prompt, " \t\r\n"));
        if (p.len == 0) return false;
        const n = clip(std.mem.trim(u8, negative, " \t\r\n"));
        const id = idOf(p);

        for (self.entries.items) |*e| {
            if (e.id != id or !std.mem.eql(u8, e.prompt, p)) continue;
            e.updated_ms = now_ms;
            if (!std.mem.eql(u8, e.negative, n)) {
                const owned = gpa.dupe(u8, n) catch return false;
                gpa.free(e.negative);
                e.negative = owned;
            }
            std.mem.sort(Entry, self.entries.items, {}, newestFirst);
            self.save(gpa, io);
            return true;
        }

        const op = gpa.dupe(u8, p) catch return false;
        const on = gpa.dupe(u8, n) catch {
            gpa.free(op);
            return false;
        };
        self.entries.append(gpa, .{ .id = id, .updated_ms = now_ms, .prompt = op, .negative = on }) catch {
            gpa.free(op);
            gpa.free(on);
            return false;
        };
        std.mem.sort(Entry, self.entries.items, {}, newestFirst);
        // Oldest out first, which is what the sort just put at the end.
        while (self.entries.items.len > max_entries) {
            const e = self.entries.pop() orelse break;
            gpa.free(e.prompt);
            gpa.free(e.negative);
        }
        self.save(gpa, io);
        return true;
    }

    pub fn remove(self: *Store, gpa: std.mem.Allocator, io: std.Io, id: u64) void {
        for (self.entries.items, 0..) |e, i| {
            if (e.id != id) continue;
            gpa.free(e.prompt);
            gpa.free(e.negative);
            _ = self.entries.orderedRemove(i);
            self.save(gpa, io);
            return;
        }
    }

    /// Find a row by id, for restoring it into the form.
    pub fn find(self: *const Store, id: u64) ?Entry {
        for (self.entries.items) |e| {
            if (e.id == id) return e;
        }
        return null;
    }

    /// Rewrite the whole file. Called from `record`/`remove`; never from a read,
    /// because a write on open is what reshuffled the conversation list once
    /// already (see `history.Store.touch`).
    pub fn save(self: *Store, gpa: std.mem.Allocator, io: std.Io) void {
        const p = self.path orelse return;
        var aw: std.Io.Writer.Allocating = .init(gpa);
        defer aw.deinit();
        write(&aw.writer, self.entries.items) catch return;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = aw.writer.buffered() }) catch |err|
            std.log.warn("prompts: save failed: {t}", .{err});
    }
};

// ------------------------------------------------------------------- tests

fn freeAll(gpa: std.mem.Allocator, entries: []Entry) void {
    for (entries) |e| {
        gpa.free(e.prompt);
        gpa.free(e.negative);
    }
    gpa.free(entries);
}

test "a library round-trips through the length-prefixed format" {
    const gpa = std.testing.allocator;
    var src = [_]Entry{
        // Newlines and quotes inside the text are exactly what a length prefix
        // is for; a line-delimited format would lose the first one.
        .{ .id = idOf("a foggy\npier"), .updated_ms = 5, .prompt = @constCast("a foggy\npier"), .negative = @constCast("blurry") },
        .{ .id = idOf("答え: 42 · \"q\""), .updated_ms = 3, .prompt = @constCast("答え: 42 · \"q\""), .negative = @constCast("") },
    };

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, &src);

    const back = try parse(gpa, aw.writer.buffered());
    defer freeAll(gpa, back);

    errdefer std.debug.print("parsed {d} entries\n", .{back.len});
    try std.testing.expectEqual(@as(usize, 2), back.len);
    for (src, back) |a, b| {
        try std.testing.expectEqual(a.updated_ms, b.updated_ms);
        try std.testing.expectEqualStrings(a.prompt, b.prompt);
        try std.testing.expectEqualStrings(a.negative, b.negative);
        try std.testing.expectEqual(idOf(a.prompt), b.id);
    }
}

test "an empty library round-trips" {
    const gpa = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, &.{});
    const back = try parse(gpa, aw.writer.buffered());
    defer freeAll(gpa, back);
    try std.testing.expectEqual(@as(usize, 0), back.len);
}

test "a truncated or foreign file is refused, not half-read" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadMagic, parse(gpa, "tp-conv 1\n"));
    // A header promising more bytes than the file holds.
    try std.testing.expectError(error.Truncated, parse(gpa, magic ++ "\np 1 99 0\nshort\n"));
}

test "recording the same prompt bumps it instead of appending" {
    const gpa = std.testing.allocator;
    var s: Store = .{}; // no path: nothing is written, everything else runs
    defer s.deinit(gpa);

    try std.testing.expect(s.record(gpa, undefined, 100, "a cat", "blurry"));
    try std.testing.expect(s.record(gpa, undefined, 200, "a dog", ""));
    try std.testing.expect(s.record(gpa, undefined, 300, "a cat", "grainy"));

    errdefer std.debug.print("library holds {d} entries\n", .{s.entries.items.len});
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    // Newest first, and the bumped row carries its NEW negative.
    try std.testing.expectEqualStrings("a cat", s.entries.items[0].prompt);
    try std.testing.expectEqualStrings("grainy", s.entries.items[0].negative);
    try std.testing.expectEqual(@as(i64, 300), s.entries.items[0].updated_ms);
    try std.testing.expectEqualStrings("a dog", s.entries.items[1].prompt);
}

test "an empty or whitespace-only prompt is not recorded" {
    const gpa = std.testing.allocator;
    var s: Store = .{};
    defer s.deinit(gpa);
    try std.testing.expect(!s.record(gpa, undefined, 1, "", "neg"));
    try std.testing.expect(!s.record(gpa, undefined, 1, "  \n\t ", "neg"));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
}

test "surrounding whitespace is not part of the identity" {
    const gpa = std.testing.allocator;
    var s: Store = .{};
    defer s.deinit(gpa);
    _ = s.record(gpa, undefined, 1, "a cat", "");
    _ = s.record(gpa, undefined, 2, "  a cat\n", "");
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
}

test "the library is capped, dropping the oldest" {
    const gpa = std.testing.allocator;
    var s: Store = .{};
    defer s.deinit(gpa);
    var buf: [32]u8 = undefined;
    for (0..max_entries + 10) |i| {
        const p = try std.fmt.bufPrint(&buf, "prompt {d}", .{i});
        _ = s.record(gpa, undefined, @intCast(i + 1), p, "");
    }
    try std.testing.expectEqual(max_entries, s.entries.items.len);
    // The survivors are the newest; the oldest recorded is gone.
    try std.testing.expectEqualStrings("prompt 509", s.entries.items[0].prompt);
    try std.testing.expect(s.find(idOf("prompt 0")) == null);
    try std.testing.expect(s.find(idOf("prompt 509")) != null);
}

test "remove takes a row out by id" {
    const gpa = std.testing.allocator;
    var s: Store = .{};
    defer s.deinit(gpa);
    _ = s.record(gpa, undefined, 1, "keep", "");
    _ = s.record(gpa, undefined, 2, "drop", "");
    s.remove(gpa, undefined, idOf("drop"));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqualStrings("keep", s.entries.items[0].prompt);
}

test "day grouping is the chat sidebar's" {
    const day = 24 * 60 * 60 * 1000;
    const now: i64 = 10 * day + 3600_000;
    try std.testing.expectEqual(Group.today, groupOf(now - 1000, now, 0));
    try std.testing.expectEqual(Group.yesterday, groupOf(now - day, now, 0));
    try std.testing.expectEqual(Group.earlier, groupOf(now - 5 * day, now, 0));
}
