//! Moving a model file between this machine and a host, either way. One thread
//! per transfer, each opening its own links to that host, so a 15 GB transfer
//! never sits in front of a delta on the control connection. The UI reads
//! atomics.
//!
//! Nothing here decides WHAT to move: `Hosts.missingModel` / `Hosts.pullable`
//! answer that from the two catalogs, by id.
//!
//! A pull lands in a folder this machine scans, so the file shows up in the
//! local catalog on the next scan and the id it was asked for resolves here.
const std = @import("std");
const Io = std.Io;
const blob = @import("serve").blob;
const link = @import("serve").link;
const config = @import("shared").config;

const log = std.log.scoped(.sync);

pub const max_jobs: usize = 2;

/// Which way the bytes go. A push reads a path on this machine; a pull names
/// the file by the id the HOST knows it as, because this machine does not have
/// it and so has no path for it.
pub const Way = enum { push, pull };

pub const Job = struct {
    gpa: std.mem.Allocator,
    io: Io,
    /// The host, by settings name (a slot can be dropped under us).
    host: []u8,
    /// What is moving, for the row.
    stem: []u8,
    /// A push: the file to read. A pull: the folder it lands in.
    path: []u8,
    /// A pull: the host's catalog id text for the file. Empty for a push.
    id: []u8 = &.{},
    way: Way = .push,
    entry: config.HostEntry,
    prog: blob.Progress = .{},
    cancel: blob.Cancel = .{},
    thread: ?std.Thread = null,
    /// Set once the thread is joined; the job is then the caller's to drop.
    reaped: bool = false,
    /// A pull that landed a file, until whoever rescans takes it.
    landed: std.atomic.Value(bool) = .init(false),

    fn connect(ctx: *anyopaque) anyerror!*link.Link {
        const self: *Job = @ptrCast(@alignCast(ctx));
        const ep = self.entry.endpoint() orelse return error.BadPairingString;
        return link.connect(self.gpa, self.io, ep);
    }

    fn main(self: *Job) void {
        switch (self.way) {
            .push => self.push(),
            .pull => self.pull(),
        }
    }

    fn push(self: *Job) void {
        var sender = blob.Sender.init(self.gpa, self.io, self.path, .{ .cancel = &self.cancel }) catch |err| {
            self.fail(err);
            return;
        };
        defer sender.deinit();
        sender.run(.{ .ctx = self, .connect = connect }, &self.prog);
        if (self.prog.phase.load(.acquire) == .done) {
            log.info("{s} is now on {s}", .{ self.stem, self.host });
        } else if (self.cancel.stopped()) {
            log.info("sending {s} to {s} was cancelled; the partial stays on the host", .{ self.stem, self.host });
        } else {
            log.err("sending {s} to {s} failed: {s}", .{ self.stem, self.host, self.prog.errName() });
        }
    }

    fn pull(self: *Job) void {
        var store = blob.Store.init(self.gpa, self.io, self.path) catch |err| {
            self.fail(err);
            return;
        };
        defer store.deinit();
        var fetcher = blob.Fetcher.init(self.gpa, self.io, &store, self.id, .{ .cancel = &self.cancel }) catch |err| {
            self.fail(err);
            return;
        };
        defer fetcher.deinit();
        fetcher.run(.{ .ctx = self, .connect = connect }, &self.prog);
        if (self.prog.phase.load(.acquire) == .done) {
            self.landed.store(true, .release);
            log.info("{s} is now here, from {s}", .{ self.stem, self.host });
        } else if (self.cancel.stopped()) {
            log.info("fetching {s} from {s} was cancelled; the partial stays here", .{ self.stem, self.host });
        } else {
            log.err("fetching {s} from {s} failed: {s}", .{ self.stem, self.host, self.prog.errName() });
        }
    }

    fn fail(self: *Job, err: anyerror) void {
        self.prog.err.store(@errorName(err), .release);
        self.prog.phase.store(.failed, .release);
    }

    pub fn running(self: *const Job) bool {
        return switch (self.prog.phase.load(.acquire)) {
            .done, .failed => false,
            else => true,
        };
    }

    /// "hashing", "42%", "done", or why it failed.
    pub fn status(self: *const Job, buf: []u8) []const u8 {
        return switch (self.prog.phase.load(.acquire)) {
            .hashing => if (self.way == .pull) "asking" else "hashing",
            .sending => std.fmt.bufPrint(buf, "{d}%", .{@as(u32, @intFromFloat(self.prog.fraction() * 100))}) catch "sending",
            .committing => "checking",
            .done => "done",
            .failed => self.prog.errName(),
        };
    }
};

pub const Syncer = struct {
    gpa: std.mem.Allocator,
    io: Io,
    jobs: std.ArrayList(*Job) = .empty,

    pub fn init(gpa: std.mem.Allocator, io: Io) Syncer {
        return .{ .gpa = gpa, .io = io };
    }

    /// Stops every transfer still running (a partial stays on the host for a
    /// resume) and joins its thread.
    pub fn deinit(self: *Syncer) void {
        for (self.jobs.items) |j| j.cancel.request(self.io);
        for (self.jobs.items) |j| {
            if (j.thread) |t| t.join();
            self.free(j);
        }
        self.jobs.deinit(self.gpa);
    }

    fn free(self: *Syncer, j: *Job) void {
        self.gpa.free(j.host);
        self.gpa.free(j.stem);
        self.gpa.free(j.path);
        self.gpa.free(j.id);
        self.gpa.destroy(j);
    }

    /// A pull finished since the last ask: whoever owns the local catalog has
    /// a folder to rescan. Reported once.
    pub fn takeLanded(self: *Syncer) bool {
        var any = false;
        for (self.jobs.items) |j| {
            if (j.landed.swap(false, .acq_rel)) any = true;
        }
        return any;
    }

    /// The job for `host`, running or just finished, null when there is none.
    pub fn forHost(self: *Syncer, host: []const u8) ?*Job {
        for (self.jobs.items) |j| if (std.mem.eql(u8, j.host, host)) return j;
        return null;
    }

    pub fn busy(self: *const Syncer) usize {
        var n: usize = 0;
        for (self.jobs.items) |j| if (j.running()) {
            n += 1;
        };
        return n;
    }

    /// Start sending `path` to `entry`. One transfer per host at a time, and
    /// `max_jobs` overall: a second push would share the same uplink.
    pub fn start(self: *Syncer, host: []const u8, entry: config.HostEntry, path: []const u8) !void {
        try self.spawn(host, entry, .push, path, stemOf(path), "");
    }

    /// Start fetching the file `host` knows as `id` into `dest_dir`, a folder
    /// this machine scans. `stem` is only what the row says while it runs; the
    /// name the file lands under is the host's, out of its manifest.
    pub fn startPull(self: *Syncer, host: []const u8, entry: config.HostEntry, id: []const u8, stem: []const u8, dest_dir: []const u8) !void {
        try self.spawn(host, entry, .pull, dest_dir, stem, id);
    }

    fn spawn(self: *Syncer, host: []const u8, entry: config.HostEntry, way: Way, path: []const u8, stem: []const u8, id: []const u8) !void {
        if (self.forHost(host)) |j| if (j.running()) return error.AlreadySending;
        if (self.busy() >= max_jobs) return error.TooManyTransfers;
        self.dropFinished(host);
        const host_d = try self.gpa.dupe(u8, host);
        errdefer self.gpa.free(host_d);
        const stem_d = try self.gpa.dupe(u8, stem);
        errdefer self.gpa.free(stem_d);
        const path_d = try self.gpa.dupe(u8, path);
        errdefer self.gpa.free(path_d);
        const id_d = try self.gpa.dupe(u8, id);
        errdefer self.gpa.free(id_d);
        const j = try self.gpa.create(Job);
        errdefer self.gpa.destroy(j);
        j.* = .{
            .gpa = self.gpa,
            .io = self.io,
            .host = host_d,
            .stem = stem_d,
            .path = path_d,
            .id = id_d,
            .way = way,
            .entry = entry,
        };
        try self.jobs.append(self.gpa, j);
        j.thread = std.Thread.spawn(.{}, Job.main, .{j}) catch |err| {
            _ = self.jobs.pop();
            return err;
        };
    }

    /// Join what has finished, keeping the last result of each host for its
    /// row. Called once a frame.
    pub fn poll(self: *Syncer) void {
        for (self.jobs.items) |j| {
            if (j.reaped or j.running()) continue;
            if (j.thread) |t| t.join();
            j.thread = null;
            j.reaped = true;
        }
    }

    fn dropFinished(self: *Syncer, host: []const u8) void {
        var i: usize = 0;
        while (i < self.jobs.items.len) {
            const j = self.jobs.items[i];
            if (!j.running() and j.reaped and std.mem.eql(u8, j.host, host)) {
                _ = self.jobs.orderedRemove(i);
                self.free(j);
                continue;
            }
            i += 1;
        }
    }

    fn stemOf(path: []const u8) []const u8 {
        const base = std.fs.path.basename(path);
        return base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "a transfer to an unreachable host fails, keeps its row, and frees on the next start" {
    var s = Syncer.init(testing.allocator, testing.io);
    defer s.deinit();
    var e: config.HostEntry = .{ .spawn = false };
    e.socket.set("127.0.0.1:1");
    e.cert.set("AA");
    e.token.set(&[_]u8{'0'} ** 64);
    try testing.expect(e.remote());

    try s.start("far", e, "/nonexistent/model.safetensors");
    try testing.expectError(error.AlreadySending, s.start("far", e, "/nonexistent/model.safetensors"));
    // The job ends on its own: the file does not open.
    while (s.forHost("far").?.running()) std.Thread.yield() catch {};
    s.poll();
    const j = s.forHost("far").?;
    try testing.expectEqual(blob.Phase.failed, j.prog.phase.load(.acquire));
    try testing.expectEqualStrings("model", j.stem);
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("FileNotFound", j.status(&buf));
    try testing.expectEqual(@as(usize, 0), s.busy());
    // Starting again for the same host replaces the finished row.
    try s.start("far", e, "/nonexistent/other.gguf");
    try testing.expectEqualStrings("other", s.forHost("far").?.stem);
    try testing.expectEqual(@as(usize, 1), s.jobs.items.len);
}

test "a transfer waiting to retry an unreachable host is cancelled at once" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    _ = try blob.writeTestModel(io, tmp.dir, "m.safetensors", 4096, 1);
    const path = try std.fs.path.join(gpa, &.{ root, "m.safetensors" });
    defer gpa.free(path);

    var s = Syncer.init(gpa, io);
    defer s.deinit();
    var e: config.HostEntry = .{ .spawn = false };
    e.socket.set("127.0.0.1:1");
    e.cert.set("AA");
    e.token.set(&[_]u8{'0'} ** 64);
    try s.start("far", e, path);
    const j = s.forHost("far").?;
    // Left alone the job would sit in its 1, 2, 4, 8 s backoff for 15 s.
    while (j.prog.phase.load(.acquire) == .hashing) std.Thread.yield() catch {};
    const t0 = Io.Clock.awake.now(io).nanoseconds;
    j.cancel.request(io);
    while (j.running()) std.Thread.yield() catch {};
    s.poll();
    const took = Io.Clock.awake.now(io).nanoseconds - t0;
    errdefer std.debug.print("cancel took {d} ms\n", .{@divTrunc(took, std.time.ns_per_ms)});
    try testing.expect(took < 5 * std.time.ns_per_s);
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("Cancelled", j.status(&buf));
}
