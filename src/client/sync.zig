//! Moving a model file between this machine and a host, either way. One thread
//! per transfer, each opening its own links to that host, so a 15 GB transfer
//! never sits in front of a delta on the control connection. The UI reads
//! atomics. A host takes one file at a time and the rest wait in order: two
//! at once would only split the same uplink.
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
    /// Read by the UI thread only, in `Syncer.poll`.
    meter: Meter = .{},
    cancel: blob.Cancel = .{},
    thread: ?std.Thread = null,
    /// Waiting its turn: no thread yet, and `prog` says nothing.
    queued: bool = true,
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
        var sender = blob.Sender.init(self.gpa, self.io, self.path, .{ .cancel = &self.cancel, .progress = &self.prog }) catch |err| {
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
        if (self.queued) return false;
        return switch (self.prog.phase.load(.acquire)) {
            .done, .failed => false,
            else => true,
        };
    }

    /// The row's line: how far, how fast, how long; then what it came to, or
    /// why it failed.
    pub fn status(self: *const Job, buf: []u8) []const u8 {
        if (self.queued) return "queued";
        const p = &self.prog;
        const total = p.bytes_total.load(.monotonic);
        const done = @min(p.bytes_done.load(.monotonic), total);
        const m = &self.meter;
        var w: Io.Writer = .fixed(buf);
        (switch (p.phase.load(.acquire)) {
            .hashing => if (self.way == .pull)
                w.writeAll("asking")
            else if (total == 0)
                w.writeAll("hashing")
            else
                w.print("hashing {f} / {f}{f}", .{ Size{ .b = done }, Size{ .b = total }, RateText{ .bps = m.rate } }),
            .sending => blk: {
                // A retry leaves its reason in `err` while it waits to reconnect.
                if (p.errName().len > 0) w.print("retrying ({s}) · ", .{p.errName()}) catch {};
                // A pull learns the size from the host's manifest.
                if (total == 0) break :blk w.writeAll("connecting");
                const pct: u32 = @intFromFloat(p.fraction() * 100);
                w.print("{d}% · {f} / {f}{f}", .{ pct, Size{ .b = done }, Size{ .b = total }, RateText{ .bps = m.rate } }) catch {};
                if (m.rate > 0) w.print(" · {f} left", .{Dur{ .s = @as(f64, @floatFromInt(total - done)) / m.rate }}) catch {};
                break :blk {};
            },
            // The receiver hashes the whole file before it takes it.
            .committing => w.print("checking {f}", .{Size{ .b = total }}),
            .done => blk: {
                const moved = p.bytes_moved.load(.monotonic);
                if (moved == 0) break :blk w.writeAll("done · it was already there");
                const secs = @as(f64, @floatFromInt(m.sending_ns)) / std.time.ns_per_s;
                w.print("done · {f} in {f}", .{ Size{ .b = moved }, Dur{ .s = secs } }) catch {};
                if (secs > 0) w.print("{f}", .{RateText{ .bps = @as(f64, @floatFromInt(moved)) / secs }}) catch {};
                break :blk {};
            },
            .failed => w.writeAll(p.errName()),
        }) catch {};
        return w.buffered();
    }

    /// Fraction of the file handled so far, for a bar; hashing counts too.
    pub fn fraction(self: *const Job) f32 {
        if (self.queued) return 0;
        return switch (self.prog.phase.load(.acquire)) {
            .hashing, .sending => self.prog.fraction(),
            .committing, .done => 1,
            .failed => 0,
        };
    }
};

/// How fast a transfer is going, from samples the UI thread takes. Each phase
/// starts over, since hashing a local disk and sending over a link are two
/// unrelated speeds.
pub const Meter = struct {
    phase: blob.Phase = .hashing,
    /// The previous sample, null until the phase's first.
    at_ns: ?i96 = null,
    at_bytes: u64 = 0,
    /// Bytes per second, smoothed; 0 until two samples are `min_gap_ns` apart.
    rate: f64 = 0,
    sending_from: ?i96 = null,
    /// Wall time spent sending, stalls and retries included.
    sending_ns: i96 = 0,

    const min_gap_ns: i96 = 250 * std.time.ns_per_ms;
    /// How many seconds of history the rate leans on.
    const tau_s: f64 = 3;

    /// `bytes` must only grow within a phase.
    pub fn sample(m: *Meter, now_ns: i96, phase: blob.Phase, bytes: u64) void {
        if (phase != m.phase) {
            if (m.sending_from) |t| m.sending_ns = now_ns - t;
            m.sending_from = if (phase == .sending) now_ns else null;
            m.phase = phase;
            m.at_ns = null;
            m.rate = 0;
        }
        if (m.sending_from) |t| m.sending_ns = now_ns - t;
        const at = m.at_ns orelse {
            m.at_ns = now_ns;
            m.at_bytes = bytes;
            return;
        };
        const dt = now_ns - at;
        if (dt < min_gap_ns) return;
        const dt_s = @as(f64, @floatFromInt(dt)) / std.time.ns_per_s;
        const inst = @as(f64, @floatFromInt(bytes -| m.at_bytes)) / dt_s;
        m.rate = if (m.rate == 0) inst else m.rate + (1 - @exp(-dt_s / tau_s)) * (inst - m.rate);
        m.at_ns = now_ns;
        m.at_bytes = bytes;
    }
};

const Size = struct {
    b: u64,
    pub fn format(self: Size, w: *Io.Writer) Io.Writer.Error!void {
        const f: f64 = @floatFromInt(self.b);
        if (self.b >= 1 << 30) return w.print("{d:.1} GiB", .{f / (1 << 30)});
        if (self.b >= 1 << 20) return w.print("{d:.0} MiB", .{f / (1 << 20)});
        return w.print("{d:.0} KiB", .{f / (1 << 10)});
    }
};

/// " · 112 MiB/s", or nothing before the first rate.
const RateText = struct {
    bps: f64,
    pub fn format(self: RateText, w: *Io.Writer) Io.Writer.Error!void {
        if (self.bps <= 0) return;
        const mib = self.bps / (1 << 20);
        if (mib >= 10) return w.print(" · {d:.0} MiB/s", .{mib});
        return w.print(" · {d:.1} MiB/s", .{mib});
    }
};

const Dur = struct {
    s: f64,
    pub fn format(self: Dur, w: *Io.Writer) Io.Writer.Error!void {
        const t: u64 = @intFromFloat(@min(@max(self.s, 0), 359999) + 0.5);
        if (t >= 3600) return w.print("{d}h{d:0>2}m", .{ t / 3600, t / 60 % 60 });
        if (t >= 60) return w.print("{d}m{d:0>2}s", .{ t / 60, t % 60 });
        return w.print("{d}s", .{t});
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

    /// The job a host's row describes: the one running, else the one that
    /// finished last, else the first waiting. Null when there is none.
    pub fn forHost(self: *Syncer, host: []const u8) ?*Job {
        var last_done: ?*Job = null;
        var first_queued: ?*Job = null;
        for (self.jobs.items) |j| {
            if (!std.mem.eql(u8, j.host, host)) continue;
            if (j.running()) return j;
            if (j.queued) {
                if (first_queued == null) first_queued = j;
            } else last_done = j;
        }
        return last_done orelse first_queued;
    }

    /// How many transfers for `host` wait behind the running one.
    pub fn queuedFor(self: *const Syncer, host: []const u8) usize {
        var n: usize = 0;
        for (self.jobs.items) |j| n += @intFromBool(j.queued and std.mem.eql(u8, j.host, host));
        return n;
    }

    /// The transfer of `stem` with `host`, waiting or running.
    pub fn find(self: *Syncer, host: []const u8, way: Way, stem: []const u8) ?*Job {
        for (self.jobs.items) |j| {
            if (j.way != way or !std.mem.eql(u8, j.host, host) or !std.mem.eql(u8, j.stem, stem)) continue;
            if (j.queued or j.running()) return j;
        }
        return null;
    }

    /// Is `path` on its way to `host`: waiting, moving, or landed and not yet
    /// in that host's catalog, which only a rescan there brings. Such a file is
    /// not "missing" there. A failed one is, so it is offered again.
    pub fn claims(self: *const Syncer, host: []const u8, path: []const u8) bool {
        for (self.jobs.items) |j| {
            if (j.way != .push or !std.mem.eql(u8, j.host, host) or !std.mem.eql(u8, j.path, path)) continue;
            if (j.queued or j.prog.phase.load(.acquire) != .failed) return true;
        }
        return false;
    }

    pub fn busy(self: *const Syncer) usize {
        var n: usize = 0;
        for (self.jobs.items) |j| n += @intFromBool(j.running());
        return n;
    }

    /// Send `path` to `entry`, now or once that host's earlier transfers are
    /// through. Asking again for one already waiting or moving does nothing.
    pub fn start(self: *Syncer, host: []const u8, entry: config.HostEntry, path: []const u8) !void {
        try self.request(host, entry, .push, path, stemOf(path), "");
    }

    /// Fetch the file `host` knows as `id` into `dest_dir`, a folder this
    /// machine scans. `stem` is only what the row says while it runs; the
    /// name the file lands under is the host's, out of its manifest.
    pub fn startPull(self: *Syncer, host: []const u8, entry: config.HostEntry, id: []const u8, stem: []const u8, dest_dir: []const u8) !void {
        try self.request(host, entry, .pull, dest_dir, stem, id);
    }

    fn request(self: *Syncer, host: []const u8, entry: config.HostEntry, way: Way, path: []const u8, stem: []const u8, id: []const u8) !void {
        for (self.jobs.items) |j| {
            if (!(j.queued or j.running())) continue;
            if (j.way == way and std.mem.eql(u8, j.host, host) and std.mem.eql(u8, j.path, path) and std.mem.eql(u8, j.id, id)) return;
        }
        // A host with nothing under way starts a fresh batch; within one,
        // every finished row stays so `claims` still covers what landed.
        if (self.idle(host)) self.dropFinished(host);
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
        self.startReady();
    }

    fn idle(self: *const Syncer, host: []const u8) bool {
        for (self.jobs.items) |j| {
            if ((j.queued or j.running()) and std.mem.eql(u8, j.host, host)) return false;
        }
        return true;
    }

    /// Start every waiting job whose host is free, in the order asked, up to
    /// `max_jobs` at once.
    fn startReady(self: *Syncer) void {
        for (self.jobs.items) |j| {
            if (!j.queued) continue;
            if (self.busy() >= max_jobs) return;
            if (self.hostRunning(j.host)) continue;
            j.queued = false;
            j.thread = std.Thread.spawn(.{}, Job.main, .{j}) catch |err| {
                j.fail(err);
                j.reaped = true;
                continue;
            };
        }
    }

    fn hostRunning(self: *const Syncer, host: []const u8) bool {
        for (self.jobs.items) |j| if (j.running() and std.mem.eql(u8, j.host, host)) return true;
        return false;
    }

    /// Sample the running, join what finished, start what waits. Called once
    /// a frame.
    pub fn poll(self: *Syncer) void {
        const now = Io.Clock.awake.now(self.io).nanoseconds;
        for (self.jobs.items) |j| {
            if (j.queued or j.reaped) continue;
            const phase = j.prog.phase.load(.acquire);
            // Hashing only ever counts up; sending is read off the counter a
            // resume cannot move.
            j.meter.sample(now, phase, if (phase == .hashing) j.prog.bytes_done.load(.monotonic) else j.prog.bytes_moved.load(.monotonic));
            if (j.running()) continue;
            if (j.thread) |t| t.join();
            j.thread = null;
            j.reaped = true;
        }
        self.startReady();
    }

    fn dropFinished(self: *Syncer, host: []const u8) void {
        var i: usize = 0;
        while (i < self.jobs.items.len) {
            const j = self.jobs.items[i];
            if (j.reaped and std.mem.eql(u8, j.host, host)) {
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
    // Asking again for the same file is not a second transfer.
    try s.start("far", e, "/nonexistent/model.safetensors");
    try testing.expectEqual(@as(usize, 1), s.jobs.items.len);
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

test "a second file for a busy host waits its turn, and is not missing there meanwhile" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    _ = try blob.writeTestModel(io, tmp.dir, "a.safetensors", 4096, 1);
    _ = try blob.writeTestModel(io, tmp.dir, "b.safetensors", 4096, 2);
    const pa = try std.fs.path.join(gpa, &.{ root, "a.safetensors" });
    defer gpa.free(pa);
    const pb = try std.fs.path.join(gpa, &.{ root, "b.safetensors" });
    defer gpa.free(pb);

    var s = Syncer.init(gpa, io);
    defer s.deinit();
    // Unreachable: a transfer sits in its retry backoff until cancelled.
    var e: config.HostEntry = .{ .spawn = false };
    e.socket.set("127.0.0.1:1");
    e.cert.set("AA");
    e.token.set(&[_]u8{'0'} ** 64);
    try s.start("far", e, pa);
    try s.start("far", e, pb);
    try s.start("far", e, pb);
    try testing.expectEqual(@as(usize, 2), s.jobs.items.len);
    try testing.expectEqual(@as(usize, 1), s.busy());
    try testing.expectEqual(@as(usize, 1), s.queuedFor("far"));
    const ja = s.find("far", .push, "a").?;
    const jb = s.find("far", .push, "b").?;
    try testing.expect(ja.running() and jb.queued);
    try testing.expectEqual(ja, s.forHost("far").?);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("queued", jb.status(&buf));
    try testing.expect(s.claims("far", pa) and s.claims("far", pb));
    try testing.expect(!s.claims("near", pa));

    // The first ends; the second starts on the next poll.
    ja.cancel.request(io);
    while (ja.running()) std.Thread.yield() catch {};
    s.poll();
    try testing.expect(!jb.queued);
    try testing.expectEqual(@as(usize, 0), s.queuedFor("far"));
    // A failed one is missing again, so it is offered again.
    try testing.expect(!s.claims("far", pa));
    jb.cancel.request(io);
    while (jb.running()) std.Thread.yield() catch {};
    s.poll();
    try testing.expect(!s.claims("far", pb));
}

test "the meter reads a rate from samples, and each phase starts over" {
    const s = std.time.ns_per_s;
    const mib: u64 = 1 << 20;
    var m: Meter = .{};
    m.sample(0, .hashing, 0);
    m.sample(1 * s, .hashing, 500 * mib);
    try testing.expectApproxEqRel(@as(f64, 500 * mib), m.rate, 1e-9);
    // Sending: the first sample only sets the baseline, whatever it reads.
    m.sample(2 * s, .sending, 0);
    try testing.expectEqual(@as(f64, 0), m.rate);
    m.sample(3 * s, .sending, 100 * mib);
    try testing.expectApproxEqRel(@as(f64, 100 * mib), m.rate, 1e-9);
    // Closer than the gap: not a sample.
    m.sample(3 * s + 1000, .sending, 900 * mib);
    try testing.expectApproxEqRel(@as(f64, 100 * mib), m.rate, 1e-9);
    // A stall pulls the rate down, not to zero at once.
    m.sample(4 * s, .sending, 100 * mib);
    try testing.expect(m.rate > 0 and m.rate < 100 * mib);
    m.sample(5 * s, .done, 0);
    try testing.expectEqual(@as(i96, 3 * s), m.sending_ns);
}

test "a transfer's line says how far, how fast and how long, then what it came to" {
    const mib: u64 = 1 << 20;
    var j: Job = .{ .gpa = testing.allocator, .io = testing.io, .host = &.{}, .stem = &.{}, .path = &.{}, .entry = .{}, .queued = false };
    var buf: [128]u8 = undefined;

    j.prog.bytes_total.store(2048 * mib, .monotonic);
    j.prog.bytes_done.store(512 * mib, .monotonic);
    try testing.expectEqualStrings("hashing 512 MiB / 2.0 GiB", j.status(&buf));

    j.prog.phase.store(.sending, .release);
    j.meter.rate = 128 * mib;
    try testing.expectEqualStrings("25% · 512 MiB / 2.0 GiB · 128 MiB/s · 12s left", j.status(&buf));
    j.prog.err.store("ConnectionRefused", .release);
    try testing.expectStringStartsWith(j.status(&buf), "retrying (ConnectionRefused) · 25%");
    j.prog.err.store(null, .release);

    j.prog.phase.store(.done, .release);
    j.prog.bytes_moved.store(1536 * mib, .monotonic);
    j.meter.sending_ns = 90 * std.time.ns_per_s;
    try testing.expectEqualStrings("done · 1.5 GiB in 1m30s · 17 MiB/s", j.status(&buf));
    j.prog.bytes_moved.store(0, .monotonic);
    try testing.expectEqualStrings("done · it was already there", j.status(&buf));

    j.way = .pull;
    j.prog.phase.store(.sending, .release);
    j.prog.bytes_total.store(0, .monotonic);
    try testing.expectEqualStrings("connecting", j.status(&buf));
}
