//! Conversation history: an on-disk transcript store beside the config file,
//! plus the index the sidebar renders from.
//!
//! Deliberately knows nothing about `chat.Session`. It stores a flat list of
//! (role, text) turns, and app.zig converts to and from `chat.Message`. That
//! keeps the format, the file I/O and the date grouping unit-testable without a
//! model, a window or an engine.
//!
//! WHAT IS NOT STORED: attachments and generated images. Those are owned by the
//! diffusion engine's own list, which does not persist, so a reloaded
//! conversation restores its text and not its pictures. Saving image bytes into
//! every transcript would make the store grow without bound; the honest version
//! is that reopening an old chat gives you the words back.
//!
//! The file format is length-prefixed rather than JSON: a transcript is
//! arbitrary UTF-8 with newlines, quotes and lone surrogates from a half-decoded
//! stream, and a byte count cannot be broken by any of them.
//!
//!     tp-conv 1\n
//!     updated <unix-ms>\n
//!     title <n>\n<n bytes>\n
//!     turn user <n>\n<n bytes>\n
//!     turn assistant <n>\n<n bytes>\n
//!     image <w> <h> <steps> <seed> <n>\n<n path bytes>\n
//!     meta <open n> <close n> <model n>\n<open><close><model>\n
//!
//! `image` lines belong to the turn above them: the renders that turn produced,
//! by the path they were WRITTEN to. That is what lets a reopened conversation
//! show the picture instead of asking the model's tool call to make it again.
//! Only images that were actually saved get a line (saving is optional), and a
//! path whose file has since moved shows as missing rather than regenerating.
const std = @import("std");

pub const Role = enum { user, assistant };

/// One render a turn produced, by where it was written. Enough to rebuild the
/// card without re-running anything.
pub const ImageRec = struct {
    /// Absolute path the PNG was written to. Owned by the `Loaded` arena when
    /// loading, borrowed from the caller when saving.
    path: []const u8,
    width: usize,
    height: usize,
    steps: usize,
    seed: u64,
};

// NOTE what is deliberately NOT here: the prompt, negative and cfg. The saved
// PNG carries its own AUTOMATIC1111 `parameters` block describing exactly how
// it was made, so "Open in Studio" reads the FILE (see image_view.loadFromFile).
// Keeping a second copy in the transcript would be a record that can disagree
// with the image it describes.

pub const Turn = struct {
    role: Role,
    /// Borrowed from the caller when saving; owned by the `Loaded` arena when
    /// loading.
    text: []const u8,
    /// Whether the prompt this turn was generated from already had the
    /// reasoning block OPEN, so the text starts inside the thought and only
    /// carries the CLOSING marker.
    ///
    /// MUST round-trip. Without it a reloaded conversation parses a primed
    /// reply as if it had no thought at all, and the whole reasoning block
    /// spills into the answer as plain prose. Stored in the role token
    /// (`assistant_primed`) so old files, which have no such token, simply read
    /// as not primed.
    primed: bool = false,
    /// An app-written note rather than something the user typed (an image tool
    /// outcome). Stored in the role token (`note`) so old files, which have no
    /// such token, read as ordinary user turns.
    synthetic: bool = false,
    /// Renders this turn produced that were saved to disk, in emission order.
    /// Borrowed when saving, arena-owned when loading.
    images: []const ImageRec = &.{},
    /// The reasoning markers the generating template actually used, and the
    /// model that produced the turn. Empty means not recorded.
    ///
    /// The markers are stored, not the model family they came from: a family is
    /// a key into a table that still changes, and a fine-tune can emit markers
    /// its base family does not. Without these, reopening a conversation splits
    /// its replies with whatever model happens to be loaded — wrong across
    /// models, and impossible with none.
    reason_open: []const u8 = "",
    reason_close: []const u8 = "",
    model: []const u8 = "",
};

/// One entry in the sidebar index: enough to draw a row, not the transcript.
pub const Entry = struct {
    id: u64,
    /// gpa-owned.
    title: []u8,
    updated_ms: i64,
    turns: usize,
};

pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    turns: []Turn,

    pub fn deinit(self: *Loaded) void {
        self.arena.deinit();
    }
};

pub const magic = "tp-conv 1";
/// A transcript longer than this is refused rather than read into memory. A
/// real chat is kilobytes; anything at this scale is a wrong file.
pub const max_file_bytes: std.Io.Limit = @enumFromInt(32 << 20);

// ------------------------------------------------------------- serialization

/// Write `turns` to `w` in the format above.
pub fn write(w: *std.Io.Writer, title: []const u8, updated_ms: i64, turns: []const Turn) !void {
    try w.print("{s}\n", .{magic});
    try w.print("updated {d}\n", .{updated_ms});
    try w.print("title {d}\n", .{title.len});
    try w.writeAll(title);
    try w.writeAll("\n");
    for (turns) |t| {
        const tag = if (t.role == .assistant and t.primed)
            "assistant_primed"
        else if (t.role == .user and t.synthetic)
            "note"
        else
            @tagName(t.role);
        try w.print("turn {s} {d}\n", .{ tag, t.text.len });
        try w.writeAll(t.text);
        try w.writeAll("\n");
        if (t.reason_open.len > 0 or t.reason_close.len > 0 or t.model.len > 0) {
            try w.print("meta {d} {d} {d}\n", .{ t.reason_open.len, t.reason_close.len, t.model.len });
            try w.writeAll(t.reason_open);
            try w.writeAll(t.reason_close);
            try w.writeAll(t.model);
            try w.writeAll("\n");
        }
        for (t.images) |im| {
            try w.print("image {d} {d} {d} {d} {d}\n", .{ im.width, im.height, im.steps, im.seed, im.path.len });
            try w.writeAll(im.path);
            try w.writeAll("\n");
        }
    }
}

const ParseError = error{ BadMagic, Truncated, BadHeader } || std.mem.Allocator.Error;

/// A cursor over the raw file bytes. Every read is bounds-checked against the
/// buffer, so a truncated or hand-edited file reports `Truncated` rather than
/// slicing past the end.
const Reader = struct {
    b: []const u8,
    i: usize = 0,

    fn line(self: *Reader) ParseError![]const u8 {
        if (self.i >= self.b.len) return error.Truncated;
        const nl = std.mem.indexOfScalarPos(u8, self.b, self.i, '\n') orelse return error.Truncated;
        const out = self.b[self.i..nl];
        self.i = nl + 1;
        return out;
    }

    fn blob(self: *Reader, n: usize) ParseError![]const u8 {
        // +1 for the newline the writer puts after every blob.
        if (self.i + n + 1 > self.b.len) return error.Truncated;
        const out = self.b[self.i .. self.i + n];
        self.i += n + 1;
        return out;
    }
};

/// How many blob bytes follow this header line, or null if it is not one of
/// ours. Shared by both parsers so the header scan and the full parse can never
/// disagree about where the next line starts.
fn blobLen(line: []const u8) ?usize {
    if (std.mem.startsWith(u8, line, "turn ")) {
        const sp = std.mem.lastIndexOfScalar(u8, line, ' ') orelse return null;
        return std.fmt.parseInt(usize, line[sp + 1 ..], 10) catch null;
    }
    // `meta` and `image` end in several lengths whose sum is the blob; a
    // "last number wins" rule would silently read the wrong span.
    const rest = if (std.mem.startsWith(u8, line, "meta "))
        line["meta ".len..]
    else if (std.mem.startsWith(u8, line, "image "))
        line["image ".len..]
    else
        return null;
    const n_lens: usize = if (line[0] == 'm') 3 else 1;
    var it = std.mem.tokenizeScalar(u8, rest, ' ');
    var vals: [8][]const u8 = undefined;
    var n: usize = 0;
    while (it.next()) |t| : (n += 1) {
        if (n == vals.len) return null;
        vals[n] = t;
    }
    if (n < n_lens) return null;
    var total: usize = 0;
    for (vals[n - n_lens .. n]) |v| total += std.fmt.parseInt(usize, v, 10) catch return null;
    return total;
}

/// Parse just the header, for building the index without reading transcripts
/// into memory. Returns (title, updated_ms, turn count).
pub fn parseHeader(gpa: std.mem.Allocator, bytes: []const u8) ParseError!struct { title: []u8, updated_ms: i64, turns: usize } {
    var r: Reader = .{ .b = bytes };
    if (!std.mem.eql(u8, try r.line(), magic)) return error.BadMagic;

    const up = try r.line();
    if (!std.mem.startsWith(u8, up, "updated ")) return error.BadHeader;
    const updated = std.fmt.parseInt(i64, up["updated ".len..], 10) catch return error.BadHeader;

    const tl = try r.line();
    if (!std.mem.startsWith(u8, tl, "title ")) return error.BadHeader;
    const tn = std.fmt.parseInt(usize, tl["title ".len..], 10) catch return error.BadHeader;
    const title = try r.blob(tn);

    var n: usize = 0;
    while (r.i < bytes.len) {
        const hl = r.line() catch break;
        // `blobLen` knows every line type; a "last number on the line" rule read
        // an image line's final length instead of the sum of its three and
        // desynced the whole scan from there on.
        const len = blobLen(hl) orelse break;
        _ = r.blob(len) catch break;
        if (std.mem.startsWith(u8, hl, "turn ")) n += 1;
    }
    return .{ .title = try gpa.dupe(u8, title), .updated_ms = updated, .turns = n };
}

/// Parse a whole transcript. The result owns its own arena.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) ParseError!Loaded {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var r: Reader = .{ .b = bytes };
    if (!std.mem.eql(u8, try r.line(), magic)) return error.BadMagic;
    _ = try r.line(); // updated
    const tl = try r.line();
    if (!std.mem.startsWith(u8, tl, "title ")) return error.BadHeader;
    const tn = std.fmt.parseInt(usize, tl["title ".len..], 10) catch return error.BadHeader;
    _ = try r.blob(tn);

    var out: std.ArrayList(Turn) = .empty;
    var imgs: std.ArrayList(ImageRec) = .empty;
    while (r.i < bytes.len) {
        const hl = r.line() catch break;
        if (std.mem.startsWith(u8, hl, "turn ")) {
            // A new turn closes the previous one's image list.
            if (out.items.len > 0) out.items[out.items.len - 1].images = try imgs.toOwnedSlice(a);
            const rest = hl["turn ".len..];
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse return error.BadHeader;
            const tag = rest[0..sp];
            const is_note = std.mem.eql(u8, tag, "note");
            const role: Role = if (is_note or std.mem.eql(u8, tag, "user")) .user else .assistant;
            const primed = std.mem.eql(u8, tag, "assistant_primed");
            const len = std.fmt.parseInt(usize, rest[sp + 1 ..], 10) catch return error.BadHeader;
            const text = try r.blob(len);
            try out.append(a, .{ .role = role, .text = try a.dupe(u8, text), .primed = primed, .synthetic = is_note });
        } else if (std.mem.startsWith(u8, hl, "meta ")) {
            var mt = std.mem.tokenizeScalar(u8, hl["meta ".len..], ' ');
            const ol = std.fmt.parseInt(usize, mt.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const cl = std.fmt.parseInt(usize, mt.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const ml = std.fmt.parseInt(usize, mt.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const blob = try r.blob(ol + cl + ml);
            if (out.items.len == 0) return error.BadHeader;
            const t = &out.items[out.items.len - 1];
            t.reason_open = try a.dupe(u8, blob[0..ol]);
            t.reason_close = try a.dupe(u8, blob[ol .. ol + cl]);
            t.model = try a.dupe(u8, blob[ol + cl ..]);
        } else if (std.mem.startsWith(u8, hl, "image ")) {
            var it = std.mem.tokenizeScalar(u8, hl["image ".len..], ' ');
            const w = std.fmt.parseInt(usize, it.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const h = std.fmt.parseInt(usize, it.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const st = std.fmt.parseInt(usize, it.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const sd = std.fmt.parseInt(u64, it.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const pl = std.fmt.parseInt(usize, it.next() orelse return error.BadHeader, 10) catch return error.BadHeader;
            const blob = try r.blob(pl);
            // An image line before any turn is a malformed file, not a crash.
            if (out.items.len == 0) return error.BadHeader;
            try imgs.append(a, .{
                .path = try a.dupe(u8, blob),
                .width = w,
                .height = h,
                .steps = st,
                .seed = sd,
            });
        } else break;
    }
    if (out.items.len > 0) out.items[out.items.len - 1].images = try imgs.toOwnedSlice(a);
    return .{ .arena = arena, .turns = try out.toOwnedSlice(a) };
}

// --------------------------------------------------------------- date groups

pub const Group = enum {
    today,
    yesterday,
    earlier,

    pub fn head(self: Group) []const u8 {
        return switch (self) {
            .today => "TODAY",
            .yesterday => "YESTERDAY",
            .earlier => "EARLIER",
        };
    }
};

/// Which group a timestamp falls in, by LOCAL calendar day. `tz_offset_s` is
/// seconds east of UTC (see `localOffsetSeconds`); passing 0 groups by UTC days,
/// which is wrong by at most one boundary and never crashes.
pub fn groupOf(updated_ms: i64, now_ms: i64, tz_offset_s: i64) Group {
    const day_s: i64 = 24 * 60 * 60;
    const d = @divFloor(@divFloor(updated_ms, 1000) + tz_offset_s, day_s);
    const today = @divFloor(@divFloor(now_ms, 1000) + tz_offset_s, day_s);
    if (d >= today) return .today;
    if (d == today - 1) return .yesterday;
    return .earlier;
}

/// Seconds east of UTC for the current local time, or 0 where we cannot tell.
/// `tm_gmtoff` is a POSIX extension present on glibc/musl/macOS/BSD; on other
/// platforms this degrades to UTC-day grouping rather than pulling in a
/// timezone database.
pub fn localOffsetSeconds(now_s: i64) i64 {
    if (@import("builtin").os.tag == .windows) return 0;
    if (!@hasDecl(std.c, "localtime_r")) return 0;
    var t: std.c.time_t = @intCast(now_s);
    var tm: std.c.tm = undefined;
    if (std.c.localtime_r(&t, &tm) == null) return 0;
    return @intCast(tm.tm_gmtoff);
}

/// A conversation's title: the first line of its first user turn, trimmed and
/// capped. A transcript with no user turn yet is "New chat", not "".
pub fn titleFrom(buf: []u8, turns: []const Turn) []const u8 {
    for (turns) |t| {
        if (t.role != .user) continue;
        const s = std.mem.trim(u8, t.text, " \t\r\n");
        if (s.len == 0) continue;
        const line = s[0 .. std.mem.indexOfScalar(u8, s, '\n') orelse s.len];
        const n = @min(line.len, buf.len);
        // Never cut a UTF-8 sequence in half: back up to a boundary.
        var end = n;
        while (end > 0 and (line[end - 1] & 0xC0) == 0x80) end -= 1;
        if (end > 0 and (line[end - 1] & 0x80) != 0 and end < line.len) end -= 1;
        @memcpy(buf[0..end], line[0..end]);
        return buf[0..end];
    }
    return "New chat";
}

// ------------------------------------------------------------------- store

/// The in-memory index, newest first. Owns its entries' titles.
pub const Store = struct {
    entries: std.ArrayList(Entry) = .empty,
    /// Directory transcripts live in (gpa-owned), or null when history is
    /// disabled (no resolvable config dir — the app still works, it just does
    /// not remember).
    dir: ?[]u8 = null,
    current: u64 = 0,

    pub fn deinit(self: *Store, gpa: std.mem.Allocator) void {
        for (self.entries.items) |e| gpa.free(e.title);
        self.entries.deinit(gpa);
        if (self.dir) |d| gpa.free(d);
        self.* = .{};
    }

    /// Point the store at `dir` (created if absent) and scan it. A directory we
    /// cannot make or read disables history instead of failing the app.
    pub fn open(self: *Store, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) void {
        self.dir = gpa.dupe(u8, dir) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch |err| {
            std.log.warn("history: cannot create {s}: {t} — conversations will not be saved", .{ dir, err });
            gpa.free(self.dir.?);
            self.dir = null;
            return;
        };
        self.rescan(gpa, io);
    }

    /// Rebuild the index from the directory. Unreadable or malformed files are
    /// skipped with a warning: one bad transcript must not hide the rest.
    pub fn rescan(self: *Store, gpa: std.mem.Allocator, io: std.Io) void {
        const dir_path = self.dir orelse return;
        for (self.entries.items) |e| gpa.free(e.title);
        self.entries.clearRetainingCapacity();

        var d = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
        defer d.close(io);
        var it = d.iterate();
        var namebuf: [512]u8 = undefined;
        while (it.next(io) catch null) |ent| {
            if (ent.kind != .file) continue;
            if (!std.mem.endsWith(u8, ent.name, ".tpc")) continue;
            const id = std.fmt.parseInt(u64, ent.name[0 .. ent.name.len - 4], 10) catch continue;
            const path = std.fmt.bufPrint(&namebuf, "{s}/{s}", .{ dir_path, ent.name }) catch continue;
            const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, max_file_bytes) catch continue;
            defer gpa.free(bytes);
            const h = parseHeader(gpa, bytes) catch |err| {
                std.log.warn("history: skipping {s}: {t}", .{ ent.name, err });
                continue;
            };
            self.entries.append(gpa, .{
                .id = id,
                .title = h.title,
                .updated_ms = h.updated_ms,
                .turns = h.turns,
            }) catch gpa.free(h.title);
        }
        std.mem.sort(Entry, self.entries.items, {}, newestFirst);
    }

    fn newestFirst(_: void, a: Entry, b: Entry) bool {
        return a.updated_ms > b.updated_ms;
    }

    /// Save `turns` as the current conversation, creating its id on first save.
    /// A transcript with nothing in it is not written: an empty "New chat" the
    /// user never typed into should not accumulate in the list.
    pub fn save(self: *Store, gpa: std.mem.Allocator, io: std.Io, now_ms: i64, turns: []const Turn) void {
        const dir_path = self.dir orelse return;
        if (turns.len == 0) return;
        // The id is the creation time, which also makes the file name sort by
        // age. Allocated on the FIRST save, so a New chat nobody typed into
        // leaves nothing behind.
        if (self.current == 0) self.current = @intCast(@max(now_ms, 1));

        var tbuf: [160]u8 = undefined;
        const title = titleFrom(&tbuf, turns);
        const now = now_ms;

        var aw: std.Io.Writer.Allocating = .init(gpa);
        defer aw.deinit();
        write(&aw.writer, title, now, turns) catch return;

        var namebuf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&namebuf, "{s}/{d}.tpc", .{ dir_path, self.current }) catch return;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = aw.writer.buffered() }) catch |err| {
            std.log.warn("history: save failed: {t}", .{err});
            return;
        };

        // Keep the index in step without a full rescan (this runs after every
        // completed turn).
        for (self.entries.items) |*e| {
            if (e.id != self.current) continue;
            gpa.free(e.title);
            e.title = gpa.dupe(u8, title) catch "";
            e.updated_ms = now;
            e.turns = turns.len;
            std.mem.sort(Entry, self.entries.items, {}, newestFirst);
            return;
        }
        const owned = gpa.dupe(u8, title) catch return;
        self.entries.append(gpa, .{
            .id = self.current,
            .title = owned,
            .updated_ms = now,
            .turns = turns.len,
        }) catch {
            gpa.free(owned);
            return;
        };
        std.mem.sort(Entry, self.entries.items, {}, newestFirst);
    }

    /// Move a conversation to the front of the list without touching its file.
    ///
    /// IN MEMORY ONLY. The list is ordered by `updated_ms`, and opening a
    /// conversation should not modify it — an earlier version rewrote the
    /// outgoing conversation on every switch, so browsing reordered everything.
    /// This gives "the one you are reading is at the top" for the session
    /// without a write on open.
    pub fn touch(self: *Store, id: u64, now_ms: i64) void {
        for (self.entries.items) |*e| {
            if (e.id != id) continue;
            e.updated_ms = now_ms;
            std.mem.sort(Entry, self.entries.items, {}, newestFirst);
            return;
        }
    }

    pub fn load(self: *Store, gpa: std.mem.Allocator, io: std.Io, id: u64) ?Loaded {
        const dir_path = self.dir orelse return null;
        var namebuf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&namebuf, "{s}/{d}.tpc", .{ dir_path, id }) catch return null;
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, max_file_bytes) catch |err| {
            std.log.warn("history: cannot read conversation {d}: {t}", .{ id, err });
            return null;
        };
        defer gpa.free(bytes);
        return parse(gpa, bytes) catch |err| {
            std.log.warn("history: cannot parse conversation {d}: {t}", .{ id, err });
            return null;
        };
    }

    pub fn remove(self: *Store, gpa: std.mem.Allocator, io: std.Io, id: u64) void {
        const dir_path = self.dir orelse return;
        var namebuf: [512]u8 = undefined;
        if (std.fmt.bufPrint(&namebuf, "{s}/{d}.tpc", .{ dir_path, id })) |path| {
            std.Io.Dir.cwd().deleteFile(io, path) catch |err| std.log.warn("history: delete failed: {t}", .{err});
        } else |_| {}
        for (self.entries.items, 0..) |e, i| {
            if (e.id != id) continue;
            gpa.free(e.title);
            _ = self.entries.orderedRemove(i);
            break;
        }
        if (self.current == id) self.current = 0;
    }

    /// Start a fresh conversation. The id is allocated on the first save, so a
    /// New chat the user never speaks into leaves nothing behind.
    pub fn newConversation(self: *Store) void {
        self.current = 0;
    }
};

// ------------------------------------------------------------------- tests

test "a transcript round-trips through the length-prefixed format" {
    const gpa = std.testing.allocator;
    const turns = [_]Turn{
        .{ .role = .user, .text = "first\nline two" }, // newlines inside a turn
        // A primed reply: the flag has to survive, or the reasoning block
        // reads back as ordinary prose.
        .{ .role = .assistant, .text = "答え: 42 · \"quoted\"", .primed = true },
        .{ .role = .assistant, .text = "not primed" },
        .{ .role = .user, .text = "" }, // and an empty one
    };
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, "a title", 1723489200123, &turns);

    var got = try parse(gpa, aw.writer.buffered());
    defer got.deinit();
    try std.testing.expectEqual(turns.len, got.turns.len);
    for (turns, got.turns) |want, have| {
        try std.testing.expectEqual(want.role, have.role);
        try std.testing.expectEqual(want.primed, have.primed);
        try std.testing.expectEqualStrings(want.text, have.text);
    }

    const h = try parseHeader(gpa, aw.writer.buffered());
    defer gpa.free(h.title);
    try std.testing.expectEqualStrings("a title", h.title);
    try std.testing.expectEqual(@as(i64, 1723489200123), h.updated_ms);
    try std.testing.expectEqual(@as(usize, 4), h.turns);
}

test "image records round-trip and attach to their own turn" {
    const gpa = std.testing.allocator;
    const shots = [_]ImageRec{
        .{ .path = "/home/u/Pictures/TensorPencil/tp_1_8812.png", .width = 1216, .height = 832, .steps = 34, .seed = 8812 },
        .{ .path = "/home/u/Pictures/TensorPencil/tp_2_8813.png", .width = 1216, .height = 832, .steps = 34, .seed = 8813 },
    };
    const turns = [_]Turn{
        .{ .role = .user, .text = "two lighthouses" },
        .{ .role = .assistant, .text = "here you go", .images = &shots },
        // A later turn with none of its own must not inherit the pair above.
        .{ .role = .user, .text = "thanks" },
        .{ .role = .assistant, .text = "welcome" },
    };
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, "t", 1, &turns);

    var got = try parse(gpa, aw.writer.buffered());
    defer got.deinit();
    try std.testing.expectEqual(@as(usize, 4), got.turns.len);
    try std.testing.expectEqual(@as(usize, 2), got.turns[1].images.len);
    try std.testing.expectEqual(@as(usize, 0), got.turns[0].images.len);
    try std.testing.expectEqual(@as(usize, 0), got.turns[3].images.len);
    for (shots, got.turns[1].images) |want, have| {
        try std.testing.expectEqualStrings(want.path, have.path);
        try std.testing.expectEqual(want.width, have.width);
        try std.testing.expectEqual(want.seed, have.seed);
    }
    // The header's turn count must not include image lines.
    const h = try parseHeader(gpa, aw.writer.buffered());
    defer gpa.free(h.title);
    try std.testing.expectEqual(@as(usize, 4), h.turns);
}

test "reasoning markers and the generating model round-trip per turn" {
    const gpa = std.testing.allocator;
    const turns = [_]Turn{
        .{ .role = .user, .text = "think about it" },
        // Two turns from DIFFERENT models, which is what a conversation looks
        // like after a model swap. Each has to keep its own markers or the
        // second one's are used to split the first.
        .{
            .role = .assistant,
            .text = "reasoning</think>answer",
            .primed = true,
            .reason_open = "<think>",
            .reason_close = "</think>",
            .model = "qwen3-32b-q4",
        },
        .{ .role = .user, .text = "again" },
        .{
            .role = .assistant,
            .text = "<|channel>thoughtx<channel|>y",
            .reason_open = "<|channel>thought",
            .reason_close = "<channel|>",
            .model = "gemma4-31b",
        },
    };
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, "t", 1, &turns);

    var got = try parse(gpa, aw.writer.buffered());
    defer got.deinit();
    try std.testing.expectEqual(@as(usize, 4), got.turns.len);
    for (turns, got.turns) |want, have| {
        try std.testing.expectEqualStrings(want.reason_open, have.reason_open);
        try std.testing.expectEqualStrings(want.reason_close, have.reason_close);
        try std.testing.expectEqualStrings(want.model, have.model);
        try std.testing.expectEqual(want.primed, have.primed);
    }
    // A user turn records none of it and must not inherit the turn above.
    try std.testing.expectEqualStrings("", got.turns[2].reason_open);
    // And the header count still counts only turns.
    const h = try parseHeader(gpa, aw.writer.buffered());
    defer gpa.free(h.title);
    try std.testing.expectEqual(@as(usize, 4), h.turns);
}

test "an app-written note round-trips as a note, not as the user's words" {
    const gpa = std.testing.allocator;
    const turns = [_]Turn{
        .{ .role = .user, .text = "draw a lighthouse" },
        .{ .role = .assistant, .text = "on it" },
        .{ .role = .user, .text = "[image tool] finished: 1216×832, seed 8812", .synthetic = true },
        .{ .role = .user, .text = "make it darker" },
    };
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, "t", 1, &turns);

    var got = try parse(gpa, aw.writer.buffered());
    defer got.deinit();
    try std.testing.expectEqual(@as(usize, 4), got.turns.len);
    for (turns, got.turns) |want, have| {
        try std.testing.expectEqual(want.role, have.role);
        try std.testing.expectEqual(want.synthetic, have.synthetic);
        try std.testing.expectEqualStrings(want.text, have.text);
    }
    // A note is still a user-role turn: every chat template renders that role,
    // which is the whole reason it is carried as one.
    try std.testing.expectEqual(Role.user, got.turns[2].role);
}

test "a truncated file is reported, not read past the end" {
    const gpa = std.testing.allocator;
    const turns = [_]Turn{.{ .role = .user, .text = "hello there" }};
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try write(&aw.writer, "t", 1, &turns);

    const full = aw.writer.buffered();
    // Every prefix of a valid file must either parse or fail cleanly. A blob
    // read that trusted its length header would slice out of bounds here.
    for (0..full.len) |n| {
        if (parse(gpa, full[0..n])) |*loaded| {
            var l = loaded.*;
            l.deinit();
        } else |err| {
            try std.testing.expect(err == error.Truncated or err == error.BadMagic or err == error.BadHeader);
        }
    }
}

test "a file that is not ours is rejected by magic" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadMagic, parse(gpa, "{\"messages\":[]}\n"));
    // An empty file has no first line at all, so it fails one step earlier.
    try std.testing.expectError(error.Truncated, parse(gpa, ""));
}

test "day grouping is by local calendar day, not elapsed hours" {
    const day: i64 = 24 * 60 * 60 * 1000;
    // 2026-08-12 01:00 UTC, and a "now" later the same UTC day.
    const now = 1786000000000;
    try std.testing.expectEqual(Group.today, groupOf(now - 1000, now, 0));
    try std.testing.expectEqual(Group.today, groupOf(now, now, 0));
    // 23:00 the previous evening is YESTERDAY even though it is hours ago...
    const prev_day = @divFloor(now, day) * day - 1000;
    try std.testing.expectEqual(Group.yesterday, groupOf(prev_day, now, 0));
    // ...and three days back is EARLIER.
    try std.testing.expectEqual(Group.earlier, groupOf(now - 3 * day, now, 0));
    // A clock skew that puts an entry in the future still reads as today
    // rather than falling off the end of the enum.
    try std.testing.expectEqual(Group.today, groupOf(now + day, now, 0));
}

test "title comes from the first user turn and never splits a codepoint" {
    var buf: [8]u8 = undefined;
    try std.testing.expectEqualStrings("New chat", titleFrom(&buf, &.{}));
    try std.testing.expectEqualStrings("New chat", titleFrom(&buf, &.{.{ .role = .assistant, .text = "hi" }}));
    try std.testing.expectEqualStrings("ask", titleFrom(&buf, &.{
        .{ .role = .assistant, .text = "unused" },
        .{ .role = .user, .text = "  ask\nmore  " },
    }));
    // Eight bytes of three-byte ideographs: the cut lands on a boundary, so the
    // result is valid UTF-8 rather than a broken glyph.
    const t = titleFrom(&buf, &.{.{ .role = .user, .text = "日本語のタイトル" }});
    try std.testing.expect(std.unicode.utf8ValidateSlice(t));
    try std.testing.expect(t.len <= buf.len);
}

test "touch moves a conversation to the front without touching disk" {
    const gpa = std.testing.allocator;
    var st: Store = .{};
    defer st.deinit(gpa);
    // No `dir`, so nothing here can write even if it tried.
    try std.testing.expectEqual(@as(?[]u8, null), st.dir);

    for ([_]struct { id: u64, ms: i64 }{
        .{ .id = 1, .ms = 300 },
        .{ .id = 2, .ms = 200 },
        .{ .id = 3, .ms = 100 },
    }) |e| {
        try st.entries.append(gpa, .{
            .id = e.id,
            .title = try gpa.dupe(u8, "t"),
            .updated_ms = e.ms,
            .turns = 1,
        });
    }

    // Opening the oldest puts it at the front, and leaves the rest in order.
    st.touch(3, 400);
    try std.testing.expectEqual(@as(u64, 3), st.entries.items[0].id);
    try std.testing.expectEqual(@as(u64, 1), st.entries.items[1].id);
    try std.testing.expectEqual(@as(u64, 2), st.entries.items[2].id);

    // Touching one that is not there changes nothing.
    st.touch(99, 500);
    try std.testing.expectEqual(@as(usize, 3), st.entries.items.len);
    try std.testing.expectEqual(@as(u64, 3), st.entries.items[0].id);

    // And with no directory a save is a no-op, so reading really cannot write.
    st.save(gpa, std.testing.io, 600, &.{.{ .role = .user, .text = "hi" }});
    try std.testing.expectEqual(@as(u64, 0), st.current);
}
