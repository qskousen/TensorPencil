//! Model files between hosts: content-addressed by BLAKE3, sent in fixed
//! chunks each with its own digest, resumable. The receiver is authoritative:
//! a sender that reconnects asks what the partial file already holds (every
//! chunk of an existing partial is re-hashed at that point), so a sender
//! remembers nothing across a crash. Three checks before a file is offered:
//! each chunk as it lands, the whole file at commit, and the container header
//! parsing (a digest can match a file that was already broken when hashed).
//!
//! Bounded on both ends: after setup, neither `Store.writeChunk` nor
//! `Sender.run` takes an allocator, so a 15 GB transfer costs the buffers on
//! the connection thread's stack and nothing per chunk.
//!
//! On the wire (all under `/v1/blob/<digest hex>/`): `POST manifest` with the
//! `Manifest` JSON answers `Have`, `GET have` answers it again, `PUT chunk/<i>`
//! carries one chunk's bytes, `POST commit` finishes, `DELETE` abandons.
const std = @import("std");
const Io = std.Io;
const tp = @import("TensorPencil");
const link_mod = @import("link.zig");
const httpc = @import("httpc.zig");
const Blake3 = std.crypto.hash.Blake3;

pub const Digest = [32]u8;
pub const default_chunk_bytes: u64 = 8 << 20;
pub const min_chunk_bytes: u64 = 64 << 10;
pub const max_chunk_bytes: u64 = 64 << 20;
pub const max_chunks: usize = 65536;
/// The largest file the chunking can name, and the bound every size off the
/// wire is held to.
pub const max_file_bytes: u64 = max_chunks * max_chunk_bytes;
pub const max_transfers: usize = 2;
/// A transfer nobody touched this long gives its slot to a new manifest; its
/// partial stays on disk for a resume. Tests shorten it.
pub var evict_idle_ns: i96 = 5 * std.time.ns_per_min;
/// A partial file nobody resumed in this long is deleted when the store opens.
pub const partial_max_age_ns: i96 = 7 * std.time.ns_per_day;
/// The stack buffers a chunk streams through, each side.
const io_buf_len: usize = 512 << 10;
/// A manifest names at most `max_chunks` chunks of 64 hex each.
pub const max_manifest_bytes: usize = 8 << 20;

const log = std.log.scoped(.blob);

pub fn digestHex(d: Digest) [64]u8 {
    return std.fmt.bytesToHex(d, .lower);
}

/// What the sender knows about the file before a byte moves.
pub const Manifest = struct {
    /// The file name the receiver stores it under: a basename with a model
    /// extension, never a path.
    name: []const u8 = "",
    size: u64 = 0,
    chunk_bytes: u64 = default_chunk_bytes,
    /// Whole-file BLAKE3, hex.
    digest: []const u8 = "",
    /// One BLAKE3 per chunk, hex, in order.
    chunks: []const []const u8 = &.{},

    /// Both fields come off the wire, so the round-up is written as a divide
    /// rather than `size + chunk_bytes - 1`, which overflows on a hostile pair.
    pub fn chunkCount(m: *const Manifest) usize {
        if (m.chunk_bytes == 0) return 0;
        const n = m.size / m.chunk_bytes + @intFromBool(m.size % m.chunk_bytes != 0);
        return std.math.cast(usize, n) orelse std.math.maxInt(usize);
    }

    pub fn chunkLen(m: *const Manifest, i: usize) u64 {
        const off = @as(u64, i) *| m.chunk_bytes;
        if (off >= m.size) return 0;
        return @min(m.chunk_bytes, m.size - off);
    }

    pub fn validate(m: *const Manifest) error{BadManifest}!void {
        if (m.size == 0 or m.size > max_file_bytes) return error.BadManifest;
        if (m.chunk_bytes < min_chunk_bytes or m.chunk_bytes > max_chunk_bytes) return error.BadManifest;
        if (m.chunkCount() > max_chunks or m.chunks.len != m.chunkCount()) return error.BadManifest;
        if (m.digest.len != 64) return error.BadManifest;
        for (m.chunks) |c| if (c.len != 64) return error.BadManifest;
        if (!nameOk(m.name)) return error.BadManifest;
    }
};

/// A file name and nothing else, of a kind the catalog would scan.
pub fn nameOk(name: []const u8) bool {
    if (name.len == 0 or name.len > 255) return false;
    if (!std.mem.eql(u8, std.fs.path.basename(name), name)) return false;
    if (std.mem.indexOfAny(u8, name, "/\\") != null or std.mem.eql(u8, name, "..") or std.mem.eql(u8, name, ".")) return false;
    const ext = std.fs.path.extension(name);
    return std.ascii.eqlIgnoreCase(ext, ".safetensors") or std.ascii.eqlIgnoreCase(ext, ".gguf");
}

/// Which chunks the receiver holds, as `[start, end)` ranges.
pub const Have = struct {
    chunks: u32 = 0,
    ranges: []const [2]u32 = &.{},

    pub fn has(h: *const Have, i: usize) bool {
        for (h.ranges) |r| if (i >= r[0] and i < r[1]) return true;
        return false;
    }

    pub fn count(h: *const Have) usize {
        var n: usize = 0;
        for (h.ranges) |r| n += r[1] - r[0];
        return n;
    }
};

fn hexToDigest(hex: []const u8) ?Digest {
    if (hex.len != 64) return null;
    var d: Digest = undefined;
    _ = std.fmt.hexToBytes(&d, hex) catch return null;
    return d;
}

/// Whole-file and per-chunk digests in one pass through `buf`.
pub fn hashFile(io: Io, file: Io.File, size: u64, chunk_bytes: u64, chunks: []Digest, buf: []u8, cancel: ?*const Cancel) !Digest {
    var r = file.reader(io, buf);
    try r.seekTo(0);
    var whole = Blake3.init(.{});
    for (chunks, 0..) |*out, i| {
        if (cancel) |c| if (c.stopped()) return error.Cancelled;
        const off = @as(u64, i) * chunk_bytes;
        var left: u64 = @min(chunk_bytes, size - off);
        var h = Blake3.init(.{});
        while (left > 0) {
            const want: usize = @intCast(@min(left, buf.len));
            const got = try r.interface.take(want);
            h.update(got);
            whole.update(got);
            left -= got.len;
        }
        h.final(out);
    }
    var d: Digest = undefined;
    whole.final(&d);
    return d;
}

// ── Receiver ──────────────────────────────────────────────────────────────

pub const StoreError = error{
    BadManifest,
    InsufficientSpace,
    TooManyTransfers,
    UnknownTransfer,
    BadChunk,
    ChunkMismatch,
    Incomplete,
    DigestMismatch,
    NotAModel,
    NameTaken,
    OutOfMemory,
    ReadFailed,
    WriteFailed,
    EndOfStream,
    Unexpected,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    /// Where finished files land, a folder the host scans.
    dir: []u8,
    partial_dir: []u8,
    mu: Io.Mutex = Io.Mutex.init,
    transfers: [max_transfers]?*Transfer = .{null} ** max_transfers,

    const Transfer = struct {
        arena: std.heap.ArenaAllocator,
        man: Manifest,
        key: Digest,
        hex: [64]u8,
        chunk_digests: []Digest,
        have: []bool,
        file: Io.File,
        partial_path: []u8,
        /// Connection threads inside `writeChunk`; `commit` waits for zero.
        writers: u32 = 0,
        /// A DELETE arrived while a chunk was landing; the last writer out drops it.
        aborting: bool = false,
        last_active: Io.Timestamp,

        fn deinit(t: *Transfer, io: Io) void {
            t.file.close(io);
            t.arena.deinit();
        }
    };

    pub fn init(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Store {
        const d = try gpa.dupe(u8, dir);
        errdefer gpa.free(d);
        const p = try std.fs.path.join(gpa, &.{ dir, "partial" });
        errdefer gpa.free(p);
        Io.Dir.cwd().createDirPath(io, p) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        sweepPartials(io, p);
        return .{ .gpa = gpa, .io = io, .dir = d, .partial_dir = p };
    }

    /// Delete partials older than `partial_max_age_ns`.
    fn sweepPartials(io: Io, partial_dir: []const u8) void {
        var d = Io.Dir.cwd().openDir(io, partial_dir, .{ .iterate = true }) catch return;
        defer d.close(io);
        const cutoff = Io.Clock.real.now(io).nanoseconds - partial_max_age_ns;
        var it = d.iterate();
        while (it.next(io) catch null) |ent| {
            if (ent.kind != .file) continue;
            const st = d.statFile(io, ent.name, .{}) catch continue;
            if (st.mtime.nanoseconds < cutoff) d.deleteFile(io, ent.name) catch {};
        }
    }

    fn now(self: *const Store) Io.Timestamp {
        return Io.Clock.real.now(self.io);
    }

    pub fn deinit(self: *Store) void {
        for (&self.transfers) |*slot| if (slot.*) |t| {
            t.deinit(self.io);
            self.gpa.destroy(t);
            slot.* = null;
        };
        self.gpa.free(self.dir);
        self.gpa.free(self.partial_dir);
    }

    fn find(self: *Store, hex: []const u8) ?*Transfer {
        for (self.transfers) |slot| if (slot) |t| if (std.mem.eql(u8, &t.hex, hex)) return t;
        return null;
    }

    /// Take a manifest: a transfer already open answers what it holds; a new
    /// one opens the partial file (checking free space first when the partial
    /// does not exist yet) and re-hashes whatever the partial already has. The
    /// `Have` JSON is gpa-owned.
    pub fn begin(self: *Store, json: []const u8) StoreError![]u8 {
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const man = std.json.parseFromSliceLeaky(Manifest, a, json, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return error.BadManifest;
        try man.validate();
        const key = hexToDigest(man.digest) orelse return error.BadManifest;
        const n = man.chunkCount();
        const chunk_digests = a.alloc(Digest, n) catch return error.OutOfMemory;
        for (man.chunks, chunk_digests) |h, *d| d.* = hexToDigest(h) orelse return error.BadManifest;

        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const t_now = self.now();
        // A sender that reconnects sends the manifest again; the transfer it
        // already has answers, and this parse is done with.
        if (self.find(man.digest)) |t| {
            if (t.aborting) return error.TooManyTransfers;
            const out = try self.haveJson(t);
            t.last_active = t_now;
            arena.deinit();
            return out;
        }
        const slot = self.freeSlot(t_now) orelse return error.TooManyTransfers;

        const partial_path = std.fs.path.join(a, &.{ self.partial_dir, man.digest }) catch return error.OutOfMemory;
        const existing = Io.Dir.cwd().statFile(self.io, partial_path, .{}) catch null;
        if (existing == null) {
            if (tp.diskspace.freeBytes(self.gpa, self.dir)) |free| {
                if (free < man.size + man.size / 20) return error.InsufficientSpace;
            }
        }
        const file = Io.Dir.cwd().createFile(self.io, partial_path, .{ .read = true, .truncate = false }) catch return error.WriteFailed;
        errdefer file.close(self.io);
        if (existing) |st| {
            // The digest covers `size` bytes and a header parse never sees a
            // tail, so a longer partial would commit under the wrong id.
            if (st.size > man.size) file.setLength(self.io, man.size) catch return error.WriteFailed;
        } else preallocate(self.io, file, man.size, man.name);

        // Every arena allocation before the struct literal: `.arena = arena`
        // copies the arena's state, so a field that allocates inside the
        // literal would leave the copy stale.
        const have_flags = a.alloc(bool, n) catch return error.OutOfMemory;
        @memset(have_flags, false);
        const t = self.gpa.create(Transfer) catch return error.OutOfMemory;
        errdefer self.gpa.destroy(t);
        t.* = .{
            .arena = arena,
            .man = man,
            .key = key,
            .hex = undefined,
            .chunk_digests = chunk_digests,
            .have = have_flags,
            .file = file,
            .partial_path = partial_path,
            .last_active = t_now,
        };
        @memcpy(&t.hex, man.digest);
        if (existing != null) verifyExisting(self.io, t);
        // Everything that can fail happens before the slot is taken.
        const out = try self.haveJson(t);
        slot.* = t;
        return out;
    }

    /// An empty slot, or the one freed by evicting the idlest transfer past
    /// `evict_idle_ns` that no writer is inside. Under `mu`.
    ///
    /// A transfer with a writer in it is never evicted, so what bounds the wait
    /// for a sender that stops mid-chunk is the connection watchdog
    /// (`server.body_stall_timeout_ns`): it shuts the socket down, the write
    /// fails, and the writer leaves. Without that bound a dead peer would hold
    /// a slot until the process ends.
    fn freeSlot(self: *Store, t_now: Io.Timestamp) ?*?*Transfer {
        var idlest: ?*?*Transfer = null;
        for (&self.transfers) |*s| {
            const t = s.* orelse return s;
            if (t.writers != 0 or t.aborting) continue;
            if (t_now.nanoseconds - t.last_active.nanoseconds < evict_idle_ns) continue;
            if (idlest == null or t.last_active.nanoseconds < idlest.?.*.?.last_active.nanoseconds) idlest = s;
        }
        const s = idlest orelse return null;
        const t = s.*.?;
        log.info("evicting idle transfer of {s}; its partial stays", .{t.man.name});
        t.deinit(self.io);
        self.gpa.destroy(t);
        s.* = null;
        return s;
    }

    /// Reserve the space up front so ENOSPC lands now, not at chunk 1800.
    /// Linux only; elsewhere the file grows as chunks land.
    fn preallocate(io: Io, file: Io.File, size: u64, name: []const u8) void {
        if (@import("builtin").os.tag == .linux) {
            const rc = std.os.linux.fallocate(file.handle, 0, 0, @intCast(size));
            if (std.posix.errno(rc) != .SUCCESS) log.warn("could not reserve {d} bytes for {s} ({t}); the file grows as chunks land", .{ size, name, std.posix.errno(rc) });
        } else {
            file.setLength(io, size) catch {};
        }
    }

    /// Mark every chunk of a partial that hashes right. The whole partial is
    /// read once; a corrupt chunk is simply sent again.
    fn verifyExisting(io: Io, t: *Transfer) void {
        var buf: [io_buf_len]u8 = undefined;
        var r = t.file.reader(io, &buf);
        for (t.have, 0..) |*flag, i| {
            const off = @as(u64, i) * t.man.chunk_bytes;
            r.seekTo(off) catch return;
            var left = t.man.chunkLen(i);
            var h = Blake3.init(.{});
            while (left > 0) {
                const got = r.interface.take(@intCast(@min(left, buf.len))) catch return;
                h.update(got);
                left -= got.len;
            }
            var d: Digest = undefined;
            h.final(&d);
            flag.* = std.mem.eql(u8, &d, &t.chunk_digests[i]);
        }
    }

    fn haveJson(self: *Store, t: *Transfer) StoreError![]u8 {
        var ranges: std.ArrayList([2]u32) = .empty;
        defer ranges.deinit(self.gpa);
        var i: usize = 0;
        while (i < t.have.len) : (i += 1) {
            if (!t.have[i]) continue;
            const start = i;
            while (i < t.have.len and t.have[i]) i += 1;
            ranges.append(self.gpa, .{ @intCast(start), @intCast(i) }) catch return error.OutOfMemory;
        }
        const h: Have = .{ .chunks = @intCast(t.have.len), .ranges = ranges.items };
        return std.json.Stringify.valueAlloc(self.gpa, h, .{}) catch return error.OutOfMemory;
    }

    pub fn have(self: *Store, hex: []const u8) StoreError![]u8 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const t = self.find(hex) orelse return error.UnknownTransfer;
        if (t.aborting) return error.UnknownTransfer;
        t.last_active = self.now();
        return self.haveJson(t);
    }

    /// Stream one chunk's bytes from `body` into the partial file, hashing as
    /// they land; the chunk counts only when its digest matches. Allocates
    /// nothing: the buffers are this thread's.
    /// Fired before each slice of a chunk body is read, so a caller holding a
    /// deadline can push it out on PROGRESS. Without it the only bound available
    /// is on the whole chunk, which is either too tight for a slow link or too
    /// loose to stop a peer that stopped talking.
    pub const Tick = struct {
        ctx: *anyopaque,
        call: *const fn (*anyopaque) void,
    };

    /// How much of a chunk body is read between two ticks.
    pub const tick_slice: u64 = 256 << 10;

    pub fn writeChunk(self: *Store, hex: []const u8, index: usize, body: *Io.Reader, len: u64, tick: ?Tick) StoreError!void {
        const t = blk: {
            self.mu.lockUncancelable(self.io);
            defer self.mu.unlock(self.io);
            const t = self.find(hex) orelse return error.UnknownTransfer;
            if (t.aborting) return error.UnknownTransfer;
            if (index >= t.have.len or len != t.man.chunkLen(index)) return error.BadChunk;
            t.writers += 1;
            break :blk t;
        };
        defer self.writerDone(t);
        var fbuf: [io_buf_len]u8 = undefined;
        var hbuf: [64 << 10]u8 = undefined;
        var fw = t.file.writer(self.io, &fbuf);
        fw.pos = @as(u64, index) * t.man.chunk_bytes;
        var hashed: Io.Writer.Hashed(Blake3) = .initHasher(&fw.interface, Blake3.init(.{}), &hbuf);
        var left = len;
        while (left > 0) {
            if (tick) |k| k.call(k.ctx);
            const n = @min(left, tick_slice);
            body.streamExact64(&hashed.writer, n) catch |err| return switch (err) {
                error.EndOfStream => error.EndOfStream,
                error.ReadFailed => error.ReadFailed,
                error.WriteFailed => error.WriteFailed,
            };
            left -= n;
        }
        hashed.writer.flush() catch return error.WriteFailed;
        fw.interface.flush() catch return error.WriteFailed;
        var d: Digest = undefined;
        hashed.hasher.final(&d);
        if (!std.mem.eql(u8, &d, &t.chunk_digests[index])) return error.ChunkMismatch;
        self.mu.lockUncancelable(self.io);
        t.have[index] = true;
        self.mu.unlock(self.io);
    }

    /// A writer leaves; the last one out of an aborting transfer drops it.
    fn writerDone(self: *Store, t: *Transfer) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        t.writers -= 1;
        t.last_active = self.now();
        if (t.aborting and t.writers == 0) self.drop(t);
    }

    /// Delete the partial and forget the transfer. Under `mu`; `t` is gone after.
    fn drop(self: *Store, t: *Transfer) void {
        for (&self.transfers) |*s| if (s.* == t) {
            s.* = null;
        };
        Io.Dir.cwd().deleteFile(self.io, t.partial_path) catch {};
        t.deinit(self.io);
        self.gpa.destroy(t);
    }

    /// Every chunk present: re-read the whole file against the manifest's
    /// digest, parse it as a container, then move it into the folder under
    /// its name. Returns the final path, gpa-owned.
    pub fn commit(self: *Store, hex: []const u8) StoreError![]u8 {
        const t = blk: {
            self.mu.lockUncancelable(self.io);
            defer self.mu.unlock(self.io);
            const t = self.find(hex) orelse return error.UnknownTransfer;
            if (t.writers != 0) return error.Incomplete;
            for (t.have) |h| if (!h) return error.Incomplete;
            // Off the list while it is verified, so a chunk cannot land under it.
            for (&self.transfers) |*s| if (s.* == t) {
                s.* = null;
            };
            break :blk t;
        };
        errdefer {
            // A failed commit keeps the partial for the next attempt; the
            // transfer is reopened by the next manifest.
            t.deinit(self.io);
            self.gpa.destroy(t);
        }
        var buf: [io_buf_len]u8 = undefined;
        var r = t.file.reader(self.io, &buf);
        r.seekTo(0) catch return error.ReadFailed;
        var h = Blake3.init(.{});
        var left = t.man.size;
        while (left > 0) {
            const got = r.interface.take(@intCast(@min(left, buf.len))) catch return error.ReadFailed;
            h.update(got);
            left -= got.len;
        }
        var d: Digest = undefined;
        h.final(&d);
        if (!std.mem.eql(u8, &d, &t.key)) return error.DigestMismatch;
        // A short file whose bytes happen to hash right is still not a model.
        var c = tp.pipeline.Container.openHeader(self.gpa, self.io, t.partial_path) catch return error.NotAModel;
        c.deinit();
        t.file.sync(self.io) catch return error.WriteFailed;

        const final = std.fs.path.join(self.gpa, &.{ self.dir, t.man.name }) catch return error.OutOfMemory;
        errdefer self.gpa.free(final);
        // Claim the name with an exclusive create: two commits under one name
        // would otherwise both pass a stat and the second rename would replace
        // the first file silently.
        const claim = Io.Dir.cwd().createFile(self.io, final, .{ .exclusive = true }) catch |err| return switch (err) {
            error.PathAlreadyExists => error.NameTaken,
            else => error.WriteFailed,
        };
        claim.close(self.io);
        Io.Dir.rename(.cwd(), t.partial_path, .cwd(), final, self.io) catch {
            Io.Dir.cwd().deleteFile(self.io, final) catch {};
            return error.WriteFailed;
        };
        t.deinit(self.io);
        self.gpa.destroy(t);
        return final;
    }

    /// Drop a transfer and its partial file. With a chunk still landing the
    /// drop waits for that writer to leave; until then the transfer answers
    /// nothing.
    pub fn abort(self: *Store, hex: []const u8) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const t = self.find(hex) orelse return;
        if (t.writers > 0) {
            t.aborting = true;
            return;
        }
        self.drop(t);
    }
};

// ── Sender ────────────────────────────────────────────────────────────────

pub const Phase = enum(u8) { hashing, sending, committing, done, failed };

/// What a UI reads while a transfer runs; every field is an atomic.
pub const Progress = struct {
    phase: std.atomic.Value(Phase) = .init(.hashing),
    bytes_total: std.atomic.Value(u64) = .init(0),
    bytes_done: std.atomic.Value(u64) = .init(0),
    chunks_total: std.atomic.Value(u32) = .init(0),
    /// Chunk PUTs made, resends included.
    puts: std.atomic.Value(u32) = .init(0),
    /// Connections opened.
    connects: std.atomic.Value(u32) = .init(0),
    /// The error that ended a failed run, "" otherwise.
    err: std.atomic.Value(?[*:0]const u8) = .init(null),

    pub fn errName(p: *const Progress) []const u8 {
        return std.mem.span(p.err.load(.acquire) orelse return "");
    }

    pub fn fraction(p: *const Progress) f32 {
        const total = p.bytes_total.load(.monotonic);
        if (total == 0) return 0;
        return @as(f32, @floatFromInt(p.bytes_done.load(.monotonic))) / @as(f32, @floatFromInt(total));
    }
};

/// Opens a fresh link to the receiver; the sender calls it again after a drop.
pub const Connector = struct {
    ctx: *anyopaque,
    connect: *const fn (ctx: *anyopaque) anyerror!*link_mod.Link,
};

/// Shared between the sending thread and whoever may stop it. `request` ends
/// the run at its next check and wakes a read the link is blocked in; the
/// sender publishes its live link here so that wake reaches the right socket.
pub const Cancel = struct {
    mu: Io.Mutex = .init,
    stop: std.atomic.Value(bool) = .init(false),
    live: ?*link_mod.Link = null,

    pub fn stopped(c: *const Cancel) bool {
        return c.stop.load(.acquire);
    }

    pub fn request(c: *Cancel, io: Io) void {
        c.stop.store(true, .release);
        c.mu.lockUncancelable(io);
        defer c.mu.unlock(io);
        if (c.live) |l| l.shutdown();
    }

    fn publish(c: *Cancel, io: Io, l: ?*link_mod.Link) void {
        c.mu.lockUncancelable(io);
        defer c.mu.unlock(io);
        c.live = l;
    }
};

pub const SendOptions = struct {
    chunk_bytes: u64 = default_chunk_bytes,
    /// Give up after this many failed attempts in a row.
    max_attempts: u32 = 5,
    /// The wait before the second attempt; it doubles after each failure.
    retry_base_ns: u64 = std.time.ns_per_s,
    cancel: ?*Cancel = null,
    /// Test hook: cut the link after this many payload bytes on the first
    /// connection, as a dropped network would.
    cut_after_bytes: ?u64 = null,
};

/// What describing a file costs: the hashing pass and the JSON both a push and
/// a pull name the file with. `chunks` and `json` are the caller's to free.
const Built = struct {
    json: []u8,
    chunks: []Digest,
    hex: [64]u8,
};

fn buildManifest(gpa: std.mem.Allocator, io: Io, file: Io.File, name: []const u8, size: u64, chunk_bytes: u64, cancel: ?*const Cancel) !Built {
    if (size == 0) return error.EmptyFile;
    if (!nameOk(name)) return error.NotAModelName;
    if (chunk_bytes < min_chunk_bytes or chunk_bytes > max_chunk_bytes) return error.BadChunkSize;
    if (size > max_file_bytes) return error.FileTooLarge;
    const n: usize = @intCast(size / chunk_bytes + @intFromBool(size % chunk_bytes != 0));
    if (n > max_chunks) return error.TooManyChunks;

    const chunks = try gpa.alloc(Digest, n);
    errdefer gpa.free(chunks);
    const io_buf = try gpa.alloc(u8, io_buf_len);
    defer gpa.free(io_buf);
    const digest = try hashFile(io, file, size, chunk_bytes, chunks, io_buf, cancel);

    var hexes = try gpa.alloc([]const u8, n);
    defer gpa.free(hexes);
    const hex_store = try gpa.alloc([64]u8, n);
    defer gpa.free(hex_store);
    for (chunks, hex_store, 0..) |c, *h, i| {
        h.* = digestHex(c);
        hexes[i] = h;
    }
    const whole_hex = digestHex(digest);
    const man: Manifest = .{ .name = name, .size = size, .chunk_bytes = chunk_bytes, .digest = &whole_hex, .chunks = hexes };
    const json = try std.json.Stringify.valueAlloc(gpa, man, .{});
    return .{ .json = json, .chunks = chunks, .hex = whole_hex };
}

/// One file to send. `init` does every allocation and the hashing pass; `run`
/// then moves bytes with no allocator at all.
pub const Sender = struct {
    gpa: std.mem.Allocator,
    io: Io,
    file: Io.File,
    size: u64,
    man_json: []u8,
    chunks: []Digest,
    hex: [64]u8,
    opts: SendOptions,
    /// Scratch for parsing the receiver's `Have` (a few KiB at most).
    scratch: []u8,
    io_buf: []u8,

    pub fn init(gpa: std.mem.Allocator, io: Io, path: []const u8, opts: SendOptions) !Sender {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const size = (try file.stat(io)).size;
        if (size == 0) return error.EmptyFile;
        const name = std.fs.path.basename(path);
        const built = try buildManifest(gpa, io, file, name, size, opts.chunk_bytes, opts.cancel);
        errdefer gpa.free(built.json);
        errdefer gpa.free(built.chunks);
        const io_buf = try gpa.alloc(u8, io_buf_len);
        errdefer gpa.free(io_buf);
        const scratch = try gpa.alloc(u8, (64 << 10) + built.chunks.len * 16);
        return .{ .gpa = gpa, .io = io, .file = file, .size = size, .man_json = built.json, .chunks = built.chunks, .hex = built.hex, .opts = opts, .scratch = scratch, .io_buf = io_buf };
    }

    pub fn deinit(self: *Sender) void {
        self.file.close(self.io);
        self.gpa.free(self.man_json);
        self.gpa.free(self.chunks);
        self.gpa.free(self.scratch);
        self.gpa.free(self.io_buf);
    }

    /// Send until the receiver has every chunk and commits. Each attempt is
    /// one connection: manifest, the missing chunks, commit. A dropped link
    /// costs at most the chunk in flight; a failed attempt waits 1, 2, 4, 8 s
    /// before the next, so a daemon restart is not five failures in a row.
    pub fn run(self: *Sender, connector: Connector, prog: *Progress) void {
        prog.bytes_total.store(self.size, .monotonic);
        prog.chunks_total.store(@intCast(self.chunks.len), .monotonic);
        prog.phase.store(.sending, .release);
        var attempts: u32 = 0;
        var wait_ns = self.opts.retry_base_ns;
        var cut = self.opts.cut_after_bytes;
        while (attempts < self.opts.max_attempts) : (attempts += 1) {
            if (attempts > 0) {
                if (!self.pause(wait_ns)) break;
                wait_ns *= 2;
            }
            if (self.stopped()) break;
            const l = connector.connect(connector.ctx) catch |err| {
                prog.err.store(@errorName(err), .release);
                continue;
            };
            _ = prog.connects.fetchAdd(1, .monotonic);
            if (self.opts.cancel) |c| c.publish(self.io, l);
            // A cancel between the connect and the publish above found no
            // link to wake; this check is what catches it.
            const done = if (self.stopped()) false else self.attempt(l, prog, &cut) catch |err| blk: {
                prog.err.store(@errorName(err), .release);
                break :blk false;
            };
            if (self.opts.cancel) |c| c.publish(self.io, null);
            l.close(self.gpa);
            if (done) {
                prog.err.store(null, .release);
                prog.phase.store(.done, .release);
                return;
            }
        }
        if (self.stopped()) prog.err.store("Cancelled", .release);
        prog.phase.store(.failed, .release);
    }

    fn stopped(self: *const Sender) bool {
        const c = self.opts.cancel orelse return false;
        return c.stopped();
    }

    /// Sleep `ns`, in slices so a cancel lands within a tenth of a second.
    /// False when cancelled.
    fn pause(self: *const Sender, ns: u64) bool {
        var left = ns;
        while (left > 0) {
            if (self.stopped()) return false;
            const step: u64 = @min(left, 100 * std.time.ns_per_ms);
            Io.sleep(self.io, .{ .nanoseconds = step }, .real) catch return false;
            left -= step;
        }
        return !self.stopped();
    }

    fn attempt(self: *Sender, l: *link_mod.Link, prog: *Progress, cut: *?u64) !bool {
        var fba = std.heap.FixedBufferAllocator.init(self.scratch);
        var target_buf: [128]u8 = undefined;
        const have = blk: {
            const target = try std.fmt.bufPrint(&target_buf, "/v1/blob/{s}/manifest", .{&self.hex});
            const resp = try httpc.requestInto(l, "POST", target, "application/json", self.man_json, fba.allocator(), 1 << 20);
            if (resp.status == 507) return error.InsufficientSpaceOnHost;
            if (resp.status == 503) return error.HostBusy;
            if (resp.status != 200) return error.ManifestRefused;
            break :blk try std.json.parseFromSliceLeaky(Have, fba.allocator(), resp.body, .{ .ignore_unknown_fields = true });
        };
        var done_bytes: u64 = 0;
        for (0..self.chunks.len) |i| if (have.has(i)) {
            done_bytes += self.chunkLen(i);
        };
        prog.bytes_done.store(done_bytes, .monotonic);

        var fr = self.file.reader(self.io, self.io_buf);
        for (0..self.chunks.len) |i| {
            if (have.has(i)) continue;
            if (self.stopped()) return error.Cancelled;
            const off = @as(u64, i) * self.opts.chunk_bytes;
            const len = self.chunkLen(i);
            try fr.seekTo(off);
            const target = try std.fmt.bufPrint(&target_buf, "/v1/blob/{s}/chunk/{d}", .{ &self.hex, i });
            _ = prog.puts.fetchAdd(1, .monotonic);
            if (cut.*) |at| if (done_bytes + len > at) {
                // Half the chunk, then the wire goes away under us.
                cut.* = null;
                try httpc.writeHead(l, "PUT", target, "application/octet-stream", len);
                try fr.interface.streamExact64(l.writer(), len / 2);
                try l.writer().flush();
                l.shutdown();
                return error.LinkCut;
            };
            const resp = try httpc.requestStreaming(l, "PUT", target, "application/octet-stream", &fr.interface, len, fba.allocator(), 4096);
            if (resp.status != 202) return error.ChunkRefused;
            done_bytes += len;
            prog.bytes_done.store(done_bytes, .monotonic);
        }
        prog.phase.store(.committing, .release);
        const target = try std.fmt.bufPrint(&target_buf, "/v1/blob/{s}/commit", .{&self.hex});
        const resp = try httpc.requestInto(l, "POST", target, "", "", fba.allocator(), 4096);
        return switch (resp.status) {
            200 => true,
            409 => error.NameTakenOnHost,
            422 => error.RejectedByHost,
            else => error.CommitRefused,
        };
    }

    fn chunkLen(self: *const Sender, i: usize) u64 {
        const off = @as(u64, i) *| self.opts.chunk_bytes;
        if (off >= self.size) return 0;
        return @min(self.opts.chunk_bytes, self.size - off);
    }
};

// ── Pull: a host hands a file back ────────────────────────────────────────

/// The read side of a transfer: one file a host OFFERS, described once and
/// held. `Store` is the write side of a push; this is what a pull reads from.
/// Both carry the same manifest and the same per-chunk digests, so a file
/// pulled is checked exactly as a file pushed is, by the same code.
///
/// No file handle is kept: a chunk read opens the path itself, so two
/// connections reading different chunks cannot race on one file offset. An
/// open per 8 MiB chunk is nothing beside the bytes.
pub const Offer = struct {
    gpa: std.mem.Allocator,
    path: []u8,
    man_json: []u8,
    hex: [64]u8,
    size: u64,
    chunk_bytes: u64,
    /// What the file was when it was hashed. A rescan that replaced it must
    /// not be served from the manifest of the file before it.
    mtime_ns: i96,
    /// Connections reading chunks right now. The cache evicts none of these.
    readers: u32 = 0,
    last_active: Io.Timestamp,

    pub fn init(gpa: std.mem.Allocator, io: Io, path: []const u8, chunk_bytes: u64) !Offer {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const st = try file.stat(io);
        const built = try buildManifest(gpa, io, file, std.fs.path.basename(path), st.size, chunk_bytes, null);
        gpa.free(built.chunks); // the JSON carries them; a read needs neither
        errdefer gpa.free(built.json);
        const owned = try gpa.dupe(u8, path);
        return .{
            .gpa = gpa,
            .path = owned,
            .man_json = built.json,
            .hex = built.hex,
            .size = st.size,
            .chunk_bytes = chunk_bytes,
            .mtime_ns = st.mtime.nanoseconds,
            .last_active = Io.Clock.real.now(io),
        };
    }

    pub fn deinit(self: *Offer) void {
        self.gpa.free(self.path);
        self.gpa.free(self.man_json);
        self.* = undefined;
    }

    pub fn chunkCount(self: *const Offer) usize {
        return @intCast(self.size / self.chunk_bytes + @intFromBool(self.size % self.chunk_bytes != 0));
    }

    pub fn chunkLen(self: *const Offer, i: usize) u64 {
        const off = @as(u64, i) *| self.chunk_bytes;
        if (off >= self.size) return 0;
        return @min(self.chunk_bytes, self.size - off);
    }

    /// Chunk `i` into `w`. The digest is not rechecked here: the puller checks
    /// every chunk as it lands and the whole file at commit, which is where a
    /// bad read has to be caught anyway.
    pub fn streamChunk(self: *const Offer, io: Io, i: usize, w: *Io.Writer) !void {
        const len = self.chunkLen(i);
        if (len == 0) return error.BadChunk;
        const file = try Io.Dir.cwd().openFile(io, self.path, .{});
        defer file.close(io);
        var buf: [io_buf_len]u8 = undefined;
        var r = file.reader(io, &buf);
        try r.seekTo(@as(u64, i) * self.chunk_bytes);
        try r.interface.streamExact64(w, len);
    }
};

/// The offers a host has described, so a second pull of the same file does not
/// hash it again. Small: hashing is the cost, and a machine hands over one or
/// two files at a time.
pub const Offers = struct {
    gpa: std.mem.Allocator,
    io: Io,
    chunk_bytes: u64 = default_chunk_bytes,
    mu: Io.Mutex = Io.Mutex.init,
    slots: [max_transfers]?*Offer = .{null} ** max_transfers,

    pub fn init(gpa: std.mem.Allocator, io: Io) Offers {
        return .{ .gpa = gpa, .io = io };
    }

    pub fn deinit(self: *Offers) void {
        for (&self.slots) |*slot| if (slot.*) |o| {
            o.deinit();
            self.gpa.destroy(o);
            slot.* = null;
        };
    }

    /// The offer for `path`, hashing it if this is the first ask. The returned
    /// pointer stands until `release`. A file whose mtime moved is re-hashed.
    pub fn acquire(self: *Offers, path: []const u8) !*Offer {
        {
            self.mu.lockUncancelable(self.io);
            defer self.mu.unlock(self.io);
            for (&self.slots) |*slot| if (slot.*) |o| {
                if (!std.mem.eql(u8, o.path, path)) continue;
                const st = Io.Dir.cwd().statFile(self.io, path, .{}) catch return error.FileNotFound;
                if (st.mtime.nanoseconds != o.mtime_ns or st.size != o.size) break;
                o.readers += 1;
                o.last_active = Io.Clock.real.now(self.io);
                return o;
            };
        }
        // Outside the lock: hashing a checkpoint takes seconds, and holding
        // the lock across it would stall every other connection.
        const made = try self.gpa.create(Offer);
        errdefer self.gpa.destroy(made);
        made.* = try Offer.init(self.gpa, self.io, path, self.chunk_bytes);
        errdefer made.deinit();

        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        // Another connection may have described the same file meanwhile.
        for (&self.slots) |*slot| if (slot.*) |o| {
            if (!std.mem.eql(u8, o.path, path) or o.mtime_ns != made.mtime_ns) continue;
            made.deinit();
            self.gpa.destroy(made);
            o.readers += 1;
            return o;
        };
        made.readers = 1;
        if (self.free()) |slot| {
            slot.* = made;
            return made;
        }
        made.deinit();
        self.gpa.destroy(made);
        return error.TooManyTransfers;
    }

    pub fn release(self: *Offers, o: *Offer) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (o.readers > 0) o.readers -= 1;
        o.last_active = Io.Clock.real.now(self.io);
    }

    /// An empty slot, or one holding an offer nobody is reading and nobody has
    /// touched lately. Called under the lock.
    fn free(self: *Offers) ?*?*Offer {
        for (&self.slots) |*slot| if (slot.* == null) return slot;
        const cutoff = Io.Clock.real.now(self.io).nanoseconds - evict_idle_ns;
        for (&self.slots) |*slot| if (slot.*) |o| {
            if (o.readers > 0 or o.last_active.nanoseconds > cutoff) continue;
            o.deinit();
            self.gpa.destroy(o);
            slot.* = null;
            return slot;
        };
        return null;
    }
};

/// The client half of a pull: a host's `/v1/pull` on one end and a local
/// `Store` on the other, so a pulled file gets every check a pushed one does
/// (each chunk as it lands, the whole file at commit, the header before it is
/// offered) and a broken transfer resumes from what the partial already holds.
///
/// Allocates nothing per chunk: the response body is streamed from the link
/// straight into `Store.writeChunk`.
pub const Fetcher = struct {
    gpa: std.mem.Allocator,
    io: Io,
    store: *Store,
    /// What the host knows the file as: its catalog id text.
    id: []const u8,
    opts: SendOptions = .{},
    /// Filled from the host's manifest on the first attempt.
    hex: [64]u8 = @splat('0'),
    scratch: []u8,

    pub fn init(gpa: std.mem.Allocator, io: Io, store: *Store, id: []const u8, opts: SendOptions) !Fetcher {
        const scratch = try gpa.alloc(u8, max_manifest_bytes);
        return .{ .gpa = gpa, .io = io, .store = store, .id = id, .opts = opts, .scratch = scratch };
    }

    pub fn deinit(self: *Fetcher) void {
        self.gpa.free(self.scratch);
    }

    /// Pull until the local store commits. Same retry shape as `Sender.run`:
    /// each attempt is one connection, and a failed one waits 1, 2, 4, 8 s.
    pub fn run(self: *Fetcher, connector: Connector, prog: *Progress) void {
        prog.phase.store(.sending, .release);
        var attempts: u32 = 0;
        var wait_ns = self.opts.retry_base_ns;
        while (attempts < self.opts.max_attempts) : (attempts += 1) {
            if (attempts > 0) {
                if (!self.pause(wait_ns)) break;
                wait_ns *= 2;
            }
            if (self.stopped()) break;
            const l = connector.connect(connector.ctx) catch |err| {
                prog.err.store(@errorName(err), .release);
                continue;
            };
            _ = prog.connects.fetchAdd(1, .monotonic);
            if (self.opts.cancel) |c| c.publish(self.io, l);
            const done = if (self.stopped()) false else self.attempt(l, prog) catch |err| blk: {
                prog.err.store(@errorName(err), .release);
                break :blk false;
            };
            if (self.opts.cancel) |c| c.publish(self.io, null);
            l.close(self.gpa);
            if (done) {
                prog.err.store(null, .release);
                prog.phase.store(.done, .release);
                return;
            }
        }
        if (self.stopped()) prog.err.store("Cancelled", .release);
        prog.phase.store(.failed, .release);
    }

    fn stopped(self: *const Fetcher) bool {
        const c = self.opts.cancel orelse return false;
        return c.stopped();
    }

    fn pause(self: *const Fetcher, ns: u64) bool {
        var left = ns;
        while (left > 0) {
            if (self.stopped()) return false;
            const step: u64 = @min(left, 100 * std.time.ns_per_ms);
            Io.sleep(self.io, .{ .nanoseconds = step }, .real) catch return false;
            left -= step;
        }
        return !self.stopped();
    }

    fn attempt(self: *Fetcher, l: *link_mod.Link, prog: *Progress) !bool {
        var fba = std.heap.FixedBufferAllocator.init(self.scratch);
        var target_buf: [320]u8 = undefined;

        const target = try std.fmt.bufPrint(&target_buf, "/v1/pull/{s}/manifest", .{self.id});
        const resp = try httpc.requestInto(l, "GET", target, "", "", fba.allocator(), max_manifest_bytes);
        if (resp.status == 404) return error.HostDoesNotHaveIt;
        if (resp.status == 503) return error.HostBusy;
        if (resp.status != 200) return error.ManifestRefused;
        const man_json = resp.body;
        const man = try std.json.parseFromSliceLeaky(Manifest, fba.allocator(), man_json, .{ .ignore_unknown_fields = true });
        if (man.digest.len != 64) return error.BadManifest;
        @memcpy(&self.hex, man.digest);
        const chunks = man.chunkCount();
        prog.bytes_total.store(man.size, .monotonic);
        prog.chunks_total.store(@intCast(chunks), .monotonic);

        // The local store decides what is still wanted; it re-hashes whatever
        // a previous attempt left in the partial.
        const have_json = try self.store.begin(man_json);
        defer self.gpa.free(have_json);
        const have = try std.json.parseFromSliceLeaky(Have, fba.allocator(), have_json, .{ .ignore_unknown_fields = true });

        var done_bytes: u64 = 0;
        for (0..chunks) |i| if (have.has(i)) {
            done_bytes += man.chunkLen(i);
        };
        prog.bytes_done.store(done_bytes, .monotonic);

        for (0..chunks) |i| {
            if (have.has(i)) continue;
            if (self.stopped()) return error.Cancelled;
            const ct = try std.fmt.bufPrint(&target_buf, "/v1/pull/{s}/chunk/{d}", .{ self.id, i });
            _ = prog.puts.fetchAdd(1, .monotonic);
            try httpc.writeHead(l, "GET", ct, "", 0);
            try l.writer().flush();
            const head = try httpc.readHead(l.reader());
            if (head.status != 200) return error.ChunkRefused;
            const len = head.content_length orelse return error.UnsupportedResponse;
            if (len != man.chunkLen(i)) return error.BadChunk;
            try self.store.writeChunk(&self.hex, i, l.reader(), len, null);
            done_bytes += len;
            prog.bytes_done.store(done_bytes, .monotonic);
        }
        prog.phase.store(.committing, .release);
        const path = self.store.commit(&self.hex) catch |err| {
            // A commit that fails on THIS side is final: the bytes are here and
            // they are wrong, so another attempt would fetch the same file.
            prog.err.store(@errorName(err), .release);
            return error.CommitFailed;
        };
        self.gpa.free(path);
        return true;
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

/// A safetensors file: an 8-byte header length, a header naming one F32 tensor
/// that covers the `payload` bytes (a multiple of 4) exactly, then random
/// payload. Returns the file's size.
pub fn writeTestModel(io: Io, dir: Io.Dir, name: []const u8, payload: usize, seed: u64) !usize {
    std.debug.assert(payload % 4 == 0);
    const header = "{{\"w\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}";
    var hbuf: [128]u8 = undefined;
    const h = try std.fmt.bufPrint(&hbuf, header, .{ payload / 4, payload });
    const hlen = (h.len + 7) & ~@as(usize, 7);
    const f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [64 << 10]u8 = undefined;
    var w = f.writer(io, &buf);
    var len8: [8]u8 = undefined;
    std.mem.writeInt(u64, &len8, hlen, .little);
    try w.interface.writeAll(&len8);
    try w.interface.writeAll(h);
    try w.interface.splatByteAll(' ', hlen - h.len);
    var rng = std.Random.DefaultPrng.init(seed);
    var left = payload;
    var chunk: [4096]u8 = undefined;
    while (left > 0) {
        const n = @min(left, chunk.len);
        rng.fill(chunk[0..n]);
        try w.interface.writeAll(chunk[0..n]);
        left -= n;
    }
    try w.interface.flush();
    return 8 + hlen + payload;
}

test "a manifest is checked before a byte moves" {
    var m: Manifest = .{ .name = "a.safetensors", .size = 100, .chunk_bytes = min_chunk_bytes, .digest = "0" ** 64, .chunks = &.{"1" ** 64} };
    try m.validate();
    try testing.expectEqual(@as(usize, 1), m.chunkCount());
    try testing.expectEqual(@as(u64, 100), m.chunkLen(0));
    m.name = "../a.safetensors";
    try testing.expectError(error.BadManifest, m.validate());
    m.name = "a.txt";
    try testing.expectError(error.BadManifest, m.validate());
    m.name = "sub/a.gguf";
    try testing.expectError(error.BadManifest, m.validate());
    m.name = "a.gguf";
    m.chunk_bytes = 10;
    try testing.expectError(error.BadManifest, m.validate());
    m.chunk_bytes = min_chunk_bytes;
    m.chunks = &.{};
    try testing.expectError(error.BadManifest, m.validate());
    // Both sizes come off the wire: the round-up must not overflow, and each
    // must be refused on its own.
    m.chunks = &.{"1" ** 64};
    m.size = std.math.maxInt(u64);
    m.chunk_bytes = std.math.maxInt(u64);
    try testing.expectEqual(@as(usize, 1), m.chunkCount());
    try testing.expectEqual(@as(u64, 0), m.chunkLen(1));
    try testing.expectError(error.BadManifest, m.validate());
    m.chunk_bytes = min_chunk_bytes;
    try testing.expectError(error.BadManifest, m.validate());
    m.size = 100;
    m.chunk_bytes = max_chunk_bytes + 1;
    try testing.expectError(error.BadManifest, m.validate());
    const h: Have = .{ .chunks = 10, .ranges = &.{ .{ 0, 2 }, .{ 5, 6 } } };
    try testing.expect(h.has(1) and !h.has(2) and h.has(5) and !h.has(6));
    try testing.expectEqual(@as(usize, 3), h.count());
}

test "the store re-hashes a partial on resume and refuses a bad chunk, a short commit and a taken name" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const store_dir = try std.fs.path.join(gpa, &.{ root, "models" });
    defer gpa.free(store_dir);
    var store = try Store.init(gpa, io, store_dir);
    defer store.deinit();

    // A 3-chunk model file, hashed as a sender would.
    const cb: u64 = min_chunk_bytes;
    const total = try writeTestModel(io, tmp.dir, "m.safetensors", 2 * @as(usize, @intCast(cb)) + 1000, 7);
    const src_path = try std.fs.path.join(gpa, &.{ root, "m.safetensors" });
    defer gpa.free(src_path);
    var s = try Sender.init(gpa, io, src_path, .{ .chunk_bytes = cb });
    defer s.deinit();
    try testing.expectEqual(@as(usize, 3), s.chunks.len);
    try testing.expectEqual(@as(u64, total), s.size);
    const tail = s.chunkLen(2);

    const have0 = try store.begin(s.man_json);
    defer gpa.free(have0);
    try testing.expect(std.mem.indexOf(u8, have0, "\"ranges\":[]") != null);
    try testing.expectError(error.Incomplete, store.commit(&s.hex));

    // Chunk 1 lands; chunk 0 with chunk 1's bytes is refused and not counted.
    const bytes = try Io.Dir.cwd().readFileAlloc(io, src_path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    const c1: usize = @intCast(cb);
    var r1: Io.Reader = .fixed(bytes[c1 .. 2 * c1]);
    try store.writeChunk(&s.hex, 1, &r1, cb, null);
    var wrong: Io.Reader = .fixed(bytes[c1 .. 2 * c1]);
    try testing.expectError(error.ChunkMismatch, store.writeChunk(&s.hex, 0, &wrong, cb, null));
    var short: Io.Reader = .fixed(bytes[0..10]);
    try testing.expectError(error.BadChunk, store.writeChunk(&s.hex, 0, &short, 10, null));
    const have1 = try store.have(&s.hex);
    defer gpa.free(have1);
    try testing.expect(std.mem.indexOf(u8, have1, "\"ranges\":[[1,2]]") != null);

    // The daemon restarts: a fresh store over the same directory learns chunk
    // 1 from the partial file alone (chunk 0 holds the wrong bytes).
    store.deinit();
    store = try Store.init(gpa, io, store_dir);
    const have2 = try store.begin(s.man_json);
    defer gpa.free(have2);
    try testing.expect(std.mem.indexOf(u8, have2, "\"ranges\":[[1,2]]") != null);

    // The bulk path takes no allocator: with one that refuses everything, the
    // remaining chunks still land. This is what bounds a 15 GB transfer.
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    store.gpa = failing.allocator();
    var r0: Io.Reader = .fixed(bytes[0..c1]);
    try store.writeChunk(&s.hex, 0, &r0, cb, null);
    var r2: Io.Reader = .fixed(bytes[2 * c1 ..]);
    try store.writeChunk(&s.hex, 2, &r2, tail, null);
    store.gpa = gpa;
    const final = try store.commit(&s.hex);
    defer gpa.free(final);
    const got = try Io.Dir.cwd().readFileAlloc(io, final, gpa, .limited(1 << 20));
    defer gpa.free(got);
    try testing.expectEqualSlices(u8, bytes, got);
    try testing.expectError(error.UnknownTransfer, store.have(&s.hex));

    // Sent again: the name is taken, and the partial is left for a retry.
    const have3 = try store.begin(s.man_json);
    defer gpa.free(have3);
    var again0: Io.Reader = .fixed(bytes[0..c1]);
    try store.writeChunk(&s.hex, 0, &again0, cb, null);
    var again1: Io.Reader = .fixed(bytes[c1 .. 2 * c1]);
    try store.writeChunk(&s.hex, 1, &again1, cb, null);
    var again2: Io.Reader = .fixed(bytes[2 * c1 ..]);
    try store.writeChunk(&s.hex, 2, &again2, tail, null);
    try testing.expectError(error.NameTaken, store.commit(&s.hex));

    // Bytes that hash right but are not a container are refused at commit.
    try tmp.dir.writeFile(io, .{ .sub_path = "junk.gguf", .data = "not a gguf at all, just bytes" ** 3000 });
    const junk_path = try std.fs.path.join(gpa, &.{ root, "junk.gguf" });
    defer gpa.free(junk_path);
    var js = try Sender.init(gpa, io, junk_path, .{ .chunk_bytes = cb });
    defer js.deinit();
    const jh = try store.begin(js.man_json);
    defer gpa.free(jh);
    const junk = try Io.Dir.cwd().readFileAlloc(io, junk_path, gpa, .limited(1 << 20));
    defer gpa.free(junk);
    for (0..js.chunks.len) |i| {
        const off = i * c1;
        var jr: Io.Reader = .fixed(junk[off..@min(off + c1, junk.len)]);
        try store.writeChunk(&js.hex, i, &jr, js.chunkLen(i), null);
    }
    try testing.expectError(error.NotAModel, store.commit(&js.hex));
}

test "a longer partial is cut to the manifest's size, and an abort during a chunk waits for the writer" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const store_dir = try std.fs.path.join(gpa, &.{ root, "models" });
    defer gpa.free(store_dir);
    var store = try Store.init(gpa, io, store_dir);
    defer store.deinit();

    const cb: u64 = min_chunk_bytes;
    const total = try writeTestModel(io, tmp.dir, "m.safetensors", 2 * @as(usize, @intCast(cb)) + 1000, 3);
    const src_path = try std.fs.path.join(gpa, &.{ root, "m.safetensors" });
    defer gpa.free(src_path);
    var s = try Sender.init(gpa, io, src_path, .{ .chunk_bytes = cb });
    defer s.deinit();
    const bytes = try Io.Dir.cwd().readFileAlloc(io, src_path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);

    // A partial holding the whole file plus a tail: every chunk hashes right,
    // the tail is dropped, and the committed file is the sender's size.
    const partial = try std.fs.path.join(gpa, &.{ store.partial_dir, &s.hex });
    defer gpa.free(partial);
    {
        const f = try Io.Dir.cwd().createFile(io, partial, .{});
        defer f.close(io);
        var wbuf: [4096]u8 = undefined;
        var w = f.writer(io, &wbuf);
        try w.interface.writeAll(bytes);
        try w.interface.splatByteAll(0xAB, 5000);
        try w.interface.flush();
    }
    const have0 = try store.begin(s.man_json);
    defer gpa.free(have0);
    try testing.expect(std.mem.indexOf(u8, have0, "\"ranges\":[[0,3]]") != null);
    try testing.expectEqual(@as(u64, total), (try Io.Dir.cwd().statFile(io, partial, .{})).size);
    const final = try store.commit(&s.hex);
    defer gpa.free(final);
    try testing.expectEqual(@as(u64, total), (try Io.Dir.cwd().statFile(io, final, .{})).size);

    // A writer is inside the transfer when DELETE arrives: nothing is freed
    // under it, the transfer answers nothing, and the last writer out drops it.
    const have1 = try store.begin(s.man_json);
    defer gpa.free(have1);
    const t = store.find(&s.hex).?;
    t.writers += 1;
    store.abort(&s.hex);
    try testing.expect(store.find(&s.hex) == t and t.aborting);
    try testing.expectError(error.UnknownTransfer, store.have(&s.hex));
    try testing.expectError(error.TooManyTransfers, store.begin(s.man_json));
    var r0: Io.Reader = .fixed(bytes[0..@intCast(cb)]);
    try testing.expectError(error.UnknownTransfer, store.writeChunk(&s.hex, 0, &r0, cb, null));
    store.writerDone(t);
    try testing.expect(store.find(&s.hex) == null);
    try testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, partial, .{}));
    // Gone means gone: a fresh manifest opens a new, empty transfer.
    const have2 = try store.begin(s.man_json);
    defer gpa.free(have2);
    try testing.expect(std.mem.indexOf(u8, have2, "\"ranges\":[]") != null);
}

test "a full table evicts its idlest transfer, keeping the partial, and old partials are swept at init" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const store_dir = try std.fs.path.join(gpa, &.{ root, "models" });
    defer gpa.free(store_dir);
    const partial_dir = try std.fs.path.join(gpa, &.{ store_dir, "partial" });
    defer gpa.free(partial_dir);

    // Two partials from before: one a week and a day old, one fresh.
    try Io.Dir.cwd().createDirPath(io, partial_dir);
    const old_path = try std.fs.path.join(gpa, &.{ partial_dir, "0" ** 64 });
    defer gpa.free(old_path);
    const new_path = try std.fs.path.join(gpa, &.{ partial_dir, "1" ** 64 });
    defer gpa.free(new_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = old_path, .data = "old" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = new_path, .data = "new" });
    const week_ago: Io.Timestamp = .{ .nanoseconds = Io.Clock.real.now(io).nanoseconds - partial_max_age_ns - std.time.ns_per_day };
    try Io.Dir.cwd().setTimestamps(io, old_path, .{ .modify_timestamp = .{ .new = week_ago } });
    var store = try Store.init(gpa, io, store_dir);
    defer store.deinit();
    try testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, old_path, .{}));
    _ = try Io.Dir.cwd().statFile(io, new_path, .{});

    const cb: u64 = min_chunk_bytes;
    var senders: [max_transfers + 1]Sender = undefined;
    var paths: [max_transfers + 1][]u8 = undefined;
    var opened: usize = 0;
    defer for (senders[0..opened], paths[0..opened]) |*x, p| {
        x.deinit();
        gpa.free(p);
    };
    for (&senders, &paths, 0..) |*x, *p, i| {
        var name: [16]u8 = undefined;
        const n = try std.fmt.bufPrint(&name, "m{d}.safetensors", .{i});
        _ = try writeTestModel(io, tmp.dir, n, @intCast(cb), 100 + i);
        p.* = try std.fs.path.join(gpa, &.{ root, n });
        errdefer gpa.free(p.*);
        x.* = try Sender.init(gpa, io, p.*, .{ .chunk_bytes = cb });
        opened += 1;
    }
    for (senders[0..max_transfers]) |*x| gpa.free(try store.begin(x.man_json));
    // Sender 0 has a chunk down; it is the idlest once the others move.
    const bytes0 = try Io.Dir.cwd().readFileAlloc(io, paths[0], gpa, .limited(1 << 20));
    defer gpa.free(bytes0);
    var r0: Io.Reader = .fixed(bytes0);
    try store.writeChunk(&senders[0].hex, 0, &r0, cb, null);
    gpa.free(try store.have(&senders[1].hex));

    // Full, and nothing has been idle long enough.
    try testing.expectError(error.TooManyTransfers, store.begin(senders[max_transfers].man_json));
    const saved = evict_idle_ns;
    defer evict_idle_ns = saved;
    evict_idle_ns = 0;
    gpa.free(try store.begin(senders[max_transfers].man_json));
    try testing.expectError(error.UnknownTransfer, store.have(&senders[0].hex));
    gpa.free(try store.have(&senders[1].hex));
    // The evicted partial is still there: sender 0 resumes with its chunk known.
    const back = try store.begin(senders[0].man_json);
    defer gpa.free(back);
    try testing.expect(std.mem.indexOf(u8, back, "\"ranges\":[[0,1]]") != null);
}
