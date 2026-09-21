//! The client's engine hosts: the local child that always exists, plus every
//! host the settings list, each behind a `Remote` with a `Mirror` of its own.
//! One writer, the frame thread. Chat is pinned to one host. New images join
//! THIS client's queue (`Asked`) and are handed out as hosts free up, one in
//! flight per host; a request naming an image goes to the host that minted
//! it. Connects run on their own threads, and a host that cannot be reached is
//! retried on a widening backoff for as long as the client runs. Only an entry
//! that cannot name an endpoint is given up on.
const std = @import("std");
const Io = std.Io;
const wire = @import("serve").wire;
const link = @import("serve").link;
const config = @import("shared").config;
const catalog = @import("shared").catalog;
const remote = @import("remote.zig");
const mirror = @import("mirror.zig");
const sched = @import("sched.zig");
const models = @import("models.zig");

const log = std.log.scoped(.hosts);

pub const HostId = u32;
pub const Frame = remote.Frame;

pub const Meter = struct { split: f32 = 0.60, limit: f32 = 0.95 };

/// The reply a render was asked in: a host id, and where in that host's
/// transcript the tool call sits.
pub const Bind = struct { host: HostId, msg: u32, variant: u32, index: u32 };

const err_name_max = 48;

pub const Slot = struct {
    id: HostId,
    /// "local" for the child at the default socket, else the entry's name. gpa-owned.
    name: []u8,
    /// null for the local child.
    entry: ?config.HostEntry,
    remote: ?*remote.Remote = null,
    /// A connect in flight, if any.
    connecting: ?*Connecting = null,
    /// Generation of the last connection; a reconnect with the same one is
    /// the same daemon, still holding everything it held.
    last_gen: ?wire.HostGen = null,
    mirror: mirror.Mirror,
    frames: std.ArrayList(Frame) = .empty,
    /// The loss of the current `remote` has been acted on.
    down: bool = false,
    /// Failed connects since this host was last up; picks the backoff.
    tries: u32 = 0,
    /// Times this host has come back after a loss, over the whole session.
    reconnects: u32 = 0,
    /// `Io.Clock.awake` nanoseconds before which no connect is attempted.
    retry_at: i96 = 0,
    /// Gone for good: the entry cannot name an endpoint.
    lost: bool = false,
    /// Why the last connect failed (an error name), empty while it stands.
    status_buf: [err_name_max]u8 = @splat(0),
    status_len: u8 = 0,
    /// `Mirror.catalog_seq` of the LOCAL host when this one was last told the
    /// settings: a remote host's ids are named out of this machine's catalog.
    settings_local_seq: u64 = 0,
    /// The last reason this host was not taking renders, as logged. Static
    /// text, compared by pointer only to notice a change.
    why_logged: []const u8 = "",
    /// `Mirror.catalog_seq` when this host was last told the settings.
    settings_catalog_seq: u64 = 0,
    /// The settings are being held back until this machine can name its files,
    /// and that has been said once.
    settings_held: bool = false,
    /// This host's meter handles, live. Seeded from the settings in `sync`,
    /// written back when a drag settles.
    meter: Meter = .{},
    /// What this host's own finished images cost. In memory only.
    table: sched.Table = .{},
    /// Images already folded into `table`.
    measured: std.AutoHashMapUnmanaged(wire.ImageId, void) = .empty,

    /// Connected and answering.
    pub fn up(self: *const Slot) bool {
        const r = self.remote orelse return false;
        return !self.lost and !r.failed.load(.acquire);
    }

    pub fn status(self: *const Slot) []const u8 {
        return self.status_buf[0..self.status_len];
    }

    fn setStatus(self: *Slot, err_name: []const u8) void {
        const n = @min(err_name.len, self.status_buf.len);
        @memcpy(self.status_buf[0..n], err_name[0..n]);
        self.status_len = @intCast(n);
    }

    pub fn post(self: *Slot, req: wire.Request) void {
        if (self.remote) |r| r.postRequest(req);
    }

    /// `Mirror.pollFetches` callback shape.
    pub fn postCtx(self: *Slot, req: wire.Request) void {
        self.post(req);
    }

    /// Takes ownership of `f` either way.
    pub fn postFrame(self: *Slot, gpa: std.mem.Allocator, f: Frame) void {
        if (self.remote) |r| r.post(f) else f.deinit(gpa);
    }

    /// This host's meter handles, after a drag settled on its bar.
    pub fn postMeter(self: *Slot) void {
        if (!self.up()) return;
        self.post(.{ .meter = .{ .split = self.meter.split, .limit = self.meter.limit } });
    }
};

/// One connect attempt on its own thread: a machine that is off holds a TCP
/// connect for as long as the kernel says, and the window must keep drawing.
/// Shared between the slot and the thread; whichever lets go last frees it,
/// and a connection nobody claimed is closed.
pub const Connecting = struct {
    pub const State = enum(u8) { running, ok, failed };

    const How = union(enum) {
        entry: config.HostEntry,
        local_socket: []const u8,
        local_default,
    };

    gpa: std.mem.Allocator,
    io: Io,
    environ: *const std.process.Environ.Map,
    on_out: *const fn () void,
    how: How,
    /// Command line the child gets, when one is spawned. Owned strings.
    args: []const []const u8,
    state: std.atomic.Value(State) = .init(.running),
    /// Written before `state` becomes `ok`.
    result: ?*remote.Remote = null,
    err_buf: [err_name_max]u8 = @splat(0),
    err_len: std.atomic.Value(usize) = .init(0),
    refs: std.atomic.Value(u8) = .init(2),

    fn run(self: *Connecting) void {
        defer self.release();
        defer self.on_out();
        if (self.open()) |r| {
            self.result = r;
            self.state.store(.ok, .release);
        } else |err| {
            const name = @errorName(err);
            const n = @min(name.len, self.err_buf.len);
            @memcpy(self.err_buf[0..n], name[0..n]);
            self.err_len.store(n, .release);
            self.state.store(.failed, .release);
        }
    }

    fn open(self: *Connecting) !*remote.Remote {
        switch (self.how) {
            .entry => |*e| {
                if (e.remote()) {
                    const ep = e.endpoint() orelse return error.BadPairingString;
                    return remote.Remote.connect(self.gpa, self.io, ep, self.on_out);
                }
                if (e.spawn) return remote.Remote.spawn(self.gpa, self.io, e.socket.slice(), self.args, self.on_out);
                const ep = link.Endpoint.parse(e.socket.slice()) orelse link.Endpoint{ .unix = e.socket.slice() };
                return remote.Remote.connect(self.gpa, self.io, ep, self.on_out);
            },
            .local_socket => |sock| return remote.Remote.spawn(self.gpa, self.io, sock, self.args, self.on_out),
            .local_default => return remote.Remote.connectOrSpawn(self.gpa, self.io, self.environ, self.args, self.on_out),
        }
    }

    fn errName(self: *const Connecting) []const u8 {
        return self.err_buf[0..self.err_len.load(.acquire)];
    }

    fn take(self: *Connecting) ?*remote.Remote {
        const r = self.result;
        self.result = null;
        return r;
    }

    fn release(self: *Connecting) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        if (self.result) |r| r.deinit();
        for (self.args) |a| self.gpa.free(a);
        self.gpa.free(self.args);
        if (self.how == .local_socket) self.gpa.free(self.how.local_socket);
        self.gpa.destroy(self);
    }
};

pub const Hosts = struct {
    gpa: std.mem.Allocator,
    io: Io,
    environ: *const std.process.Environ.Map,
    on_out: *const fn () void,
    /// tp-gui's own `--config`, handed to every child so their catalog caches
    /// sit beside that file rather than the user's.
    config_path: ?[]const u8,
    /// The local child's socket; null for the default path (and reuse of a
    /// daemon already listening there). A probe gives a private one.
    local_socket: ?[]const u8,
    /// Longest side previews are fetched at, set on every mirror.
    preview_max_edge: u32 = 1024,
    /// Heap slots, so a pointer to one (or its mirror) survives the list growing.
    slots: std.ArrayList(*Slot) = .empty,
    chat: HostId = 0,
    next_id: HostId = 0,
    /// The one trial connection in flight, if any (see `Trial`).
    trial: ?*Trial = null,
    /// Renders this client asked for, newest last (see `Asked`).
    asked: std.ArrayList(Asked) = .empty,
    next_ref: u64 = 0,
    /// Scratch for draining a host's parsed tool calls.
    calls: std.ArrayList(mirror.Call) = .empty,
    /// `canRenderAnywhere` as last pushed, so a host coming up or going down
    /// re-tells the chat host whether the image tool is worth offering.
    told_image_tool: ?bool = null,

    pub fn init(gpa: std.mem.Allocator, io: Io, environ: *const std.process.Environ.Map, on_out: *const fn () void, config_path: ?[]const u8, local_socket: ?[]const u8) !Hosts {
        var h: Hosts = .{ .gpa = gpa, .io = io, .environ = environ, .on_out = on_out, .config_path = config_path, .local_socket = local_socket };
        _ = try h.addSlot("local", null);
        return h;
    }

    pub fn deinit(self: *Hosts) void {
        self.dropTrial();
        for (self.asked.items) |*a| a.free(self.gpa);
        self.asked.deinit(self.gpa);
        for (self.calls.items) |*c| c.deinit(self.gpa);
        self.calls.deinit(self.gpa);
        for (self.slots.items) |s| self.freeSlot(s);
        self.slots.deinit(self.gpa);
    }

    // ── Trying a host before it is in force ──────────────────────────────────

    /// Reach a host the settings list but nothing is connected with yet, so a
    /// wrong token is named where it was pasted. Same thread and lifetime
    /// shape as `Connecting`.
    pub const Trial = struct {
        pub const State = enum(u8) { running, reached, refused };

        gpa: std.mem.Allocator,
        io: Io,
        on_out: *const fn () void,
        entry: config.HostEntry,
        state: std.atomic.Value(State) = .init(.running),
        err_buf: [err_name_max]u8 = @splat(0),
        err_len: std.atomic.Value(usize) = .init(0),
        refs: std.atomic.Value(u8) = .init(2),

        fn release(self: *Trial) void {
            if (self.refs.fetchSub(1, .acq_rel) == 1) self.gpa.destroy(self);
        }

        fn run(self: *Trial) void {
            defer self.release();
            defer self.on_out();
            const ep = self.entry.endpoint() orelse return self.fail("BadPairingString");
            if (remote.Remote.check(self.gpa, self.io, ep)) {
                self.state.store(.reached, .release);
            } else |err| self.fail(@errorName(err));
        }

        fn fail(self: *Trial, name: []const u8) void {
            const n = @min(name.len, self.err_buf.len);
            @memcpy(self.err_buf[0..n], name[0..n]);
            self.err_len.store(n, .release);
            self.state.store(.refused, .release);
        }
    };

    /// Reach for a host again right now, dropping any backoff and any give-up.
    /// A host that is already up is left alone.
    pub fn reconnectNamed(self: *Hosts, name: []const u8) void {
        const s = self.slotByName(name) orelse return;
        if (s.up()) return;
        s.tries = 0;
        s.retry_at = 0;
        s.lost = false;
        s.status_len = 0;
        log.info("host {s}: reaching for it now", .{s.name});
    }

    /// Ask a host to scan its model folders again. A remote host scans its OWN
    /// disk, so a file put there by any other means is invisible to this
    /// client until it does.
    pub fn rescanNamed(self: *Hosts, name: []const u8, cfg: *const config.Config) void {
        const s = self.slotByName(name) orelse return;
        if (!s.up()) return;
        postScanTo(s, cfg);
        log.info("host {s}: rescanning its folders", .{s.name});
    }

    /// Start a trial for `e`, replacing any earlier one. A host this client
    /// would START has nothing to reach yet, so it gets none.
    pub fn tryHost(self: *Hosts, e: *const config.HostEntry) void {
        self.dropTrial();
        if (!e.remote() and e.spawn) return;
        const t = self.gpa.create(Trial) catch return;
        t.* = .{ .gpa = self.gpa, .io = self.io, .on_out = self.on_out, .entry = e.* };
        const th = std.Thread.spawn(.{}, Trial.run, .{t}) catch {
            self.gpa.destroy(t);
            return;
        };
        th.detach();
        self.trial = t;
    }

    fn dropTrial(self: *Hosts) void {
        if (self.trial) |t| t.release();
        self.trial = null;
    }

    /// What the trial for `e` has to say, null when none of it is about `e`.
    pub fn trialOf(self: *const Hosts, e: *const config.HostEntry) ?struct { text: []const u8, state: Trial.State } {
        const t = self.trial orelse return null;
        if (!std.mem.eql(u8, t.entry.name.slice(), e.name.slice()) or !t.entry.sameEndpoint(e)) return null;
        const state = t.state.load(.acquire);
        return .{
            .state = state,
            .text = switch (state) {
                .running => "trying it now",
                .reached => "reached it, token accepted: Apply & Reload to use it",
                .refused => troubleText(t.err_buf[0..t.err_len.load(.acquire)]),
            },
        };
    }

    fn addSlot(self: *Hosts, name: []const u8, entry: ?config.HostEntry) !*Slot {
        const s = try self.gpa.create(Slot);
        errdefer self.gpa.destroy(s);
        s.* = .{
            .id = self.next_id,
            .name = try self.gpa.dupe(u8, name),
            .entry = entry,
            .mirror = mirror.Mirror.init(self.gpa),
        };
        s.mirror.preview_max_edge = self.preview_max_edge;
        self.next_id += 1;
        try self.slots.append(self.gpa, s);
        return s;
    }

    fn freeSlot(self: *Hosts, s: *Slot) void {
        if (s.connecting) |c| c.release();
        if (s.remote) |r| r.deinit();
        for (s.frames.items) |f| f.deinit(self.gpa);
        s.frames.deinit(self.gpa);
        s.measured.deinit(self.gpa);
        s.mirror.deinit();
        self.gpa.free(s.name);
        self.gpa.destroy(s);
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    pub fn slotOf(self: *Hosts, id: HostId) ?*Slot {
        for (self.slots.items) |s| if (s.id == id) return s;
        return null;
    }

    pub fn local(self: *Hosts) *Slot {
        return self.slots.items[0];
    }

    pub fn chatSlot(self: *Hosts) *Slot {
        return self.slotOf(self.chat) orelse self.local();
    }

    pub fn chatMirror(self: *Hosts) *mirror.Mirror {
        return &self.chatSlot().mirror;
    }

    /// The host whose mirror lists `id`.
    pub fn imageOwner(self: *Hosts, id: wire.ImageId) ?*Slot {
        for (self.slots.items) |s| if (s.mirror.byId(id) != null) return s;
        return null;
    }

    pub fn imageById(self: *Hosts, id: wire.ImageId) ?*mirror.Image {
        for (self.slots.items) |s| if (s.mirror.byId(id)) |im| return im;
        return null;
    }

    /// The name to show beside a picture, empty while there is only one host.
    pub fn hostOf(self: *Hosts, id: wire.ImageId) []const u8 {
        if (!self.several()) return "";
        const s = self.imageOwner(id) orelse return "";
        return nameFor(s, s.mirror.byId(id) orelse return "", true);
    }

    /// Which host to credit a picture to. Nobody, when this client built it
    /// itself or when there is only one host to be.
    fn nameFor(s: *const Slot, im: *const mirror.Image, many: bool) []const u8 {
        if (!many or im.local) return "";
        return s.name;
    }

    // ── Models across every host ─────────────────────────────────────────────

    /// Each host's catalog, for `models.build`. The LOCAL host is first, which
    /// is what makes a file on this machine keep its real path in the merged
    /// catalog while one only a remote host holds is named by its id.
    pub fn modelSources(self: *Hosts, out: []models.Source) []models.Source {
        var n: usize = 0;
        for (self.slots.items) |s| {
            if (n == out.len) break;
            out[n] = .{
                .name = s.name,
                .up = s.up(),
                .scanned = s.mirror.scannedOnce(),
                .cat = &s.mirror.catalog,
            };
            n += 1;
        }
        return out[0..n];
    }

    /// Everything the merged model view is built from, as one number: it moves
    /// when a catalog does, when a host comes up or goes down, and when the
    /// host list itself changes. Rebuilding the view per frame would re-join
    /// every host's entries to draw a menu that did not move.
    pub fn modelSeq(self: *Hosts) u64 {
        var seq: u64 = self.slots.items.len;
        for (self.slots.items) |s| {
            seq = seq *% 31 +% s.mirror.catalog_seq;
            seq = seq *% 2 +% @intFromBool(s.up());
        }
        return seq;
    }

    /// One render and the host holding it.
    pub const Shot = struct { host: []const u8, im: *mirror.Image };

    /// Finished renders across every host, newest first. Two hosts working at
    /// once finish out of turn, so the library and the viewer order by when a
    /// render was made rather than by which host made it.
    pub fn finished(self: *Hosts, out: []Shot) []Shot {
        var n: usize = 0;
        const name_them = self.several();
        for (self.slots.items) |s| {
            for (s.mirror.images.items) |*im| {
                if (im.status() != .done) continue;
                n = insertRanked(out, n, .{ .host = nameFor(s, im, name_them), .im = im }, newerFirst);
            }
        }
        return out[0..n];
    }

    /// Renders in flight or failed at any host, oldest first: a queue reads
    /// top to bottom. What survives a full buffer is ranked, not taken in slot
    /// order: failures are never cleared on their own, and enough of them would
    /// otherwise crowd out every render actually running.
    ///
    /// A picture whose file is gone is not work and never appears here: the
    /// queue is what a machine is doing, and reopening a conversation asks for
    /// nothing.
    pub fn running(self: *Hosts, out: []Shot) []Shot {
        var n: usize = 0;
        const name_them = self.several();
        for (self.slots.items) |s| {
            for (s.mirror.images.items) |*im| {
                if (im.missing) continue;
                switch (im.status()) {
                    .pending, .generating, .suspended, .failed => {},
                    else => continue,
                }
                n = insertRanked(out, n, .{ .host = nameFor(s, im, name_them), .im = im }, worthMore);
            }
        }
        std.mem.sort(Shot, out[0..n], {}, olderFirst);
        return out[0..n];
    }

    /// When a render was made, for ordering. A picture restored from a file has
    /// no clock of its own and sorts oldest, which is what it is.
    fn madeAt(im: *const mirror.Image) i64 {
        if (im.info.done_ns > 0) return im.info.done_ns;
        return im.info.start_ns;
    }

    fn olderFirst(_: void, a: Shot, b: Shot) bool {
        const ka = madeAt(a.im);
        const kb = madeAt(b.im);
        if (ka != kb) return ka < kb;
        return a.im.info.id < b.im.info.id;
    }

    fn newerFirst(a: Shot, b: Shot) bool {
        return olderFirst({}, b, a);
    }

    /// Which of two rows a full queue keeps: one still running before one that
    /// failed, then the oldest of those running (the next to finish) and the
    /// newest of those that failed (the one just seen).
    fn worthMore(a: Shot, b: Shot) bool {
        const fa = a.im.status() == .failed;
        const fb = b.im.status() == .failed;
        if (fa != fb) return fb;
        if (fa) return newerFirst(a, b);
        return olderFirst({}, a, b);
    }

    /// Place `s` in `out`, which `before` keeps sorted, dropping whatever ranks
    /// last once it is full. Keeps the whole walk bounded by the caller's buffer.
    fn insertRanked(out: []Shot, n: usize, s: Shot, comptime before: fn (Shot, Shot) bool) usize {
        var i: usize = 0;
        while (i < n and before(out[i], s)) i += 1;
        if (i == out.len) return n;
        const end = @min(n + 1, out.len);
        var j = end;
        while (j > i + 1) : (j -= 1) out[j - 1] = out[j - 2];
        out[i] = s;
        return end;
    }

    /// Ask for this render again. The request is the client's, so a retry is a
    /// fresh enqueue that can land on any host, not a replay on the one that
    /// just failed it.
    pub fn retry(self: *Hosts, cfg: *const config.Config, id: wire.ImageId) void {
        const im = self.imageById(id) orelse return;
        // Nothing to ask for: what made this picture was in the file, and the
        // file is what went away. Rendering its empty request would put an
        // unrelated image where the old one was.
        if (im.missing) return;
        // Built into an arena first: enqueueing trims the oldest remembered
        // requests, which can free the very strings being read here.
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var req: wire.ImageRequest = if (self.askedOf(im.info.client_ref)) |asked| asked.req else .{
            // One the model asked a host for: rebuild it from what came back.
            .prompt = im.info.prompt,
            .negative = im.info.negative,
            .width = im.info.req_width,
            .height = im.info.req_height,
            .steps = im.info.req_steps,
            .cfg = im.info.req_cfg,
            .seed = im.info.req_seed,
            .params = im.info.params,
            .from_studio = im.info.from_studio,
        };
        req.prompt = a.dupe(u8, req.prompt) catch return;
        req.negative = a.dupe(u8, req.negative) catch return;
        const loras = a.alloc(wire.LoraSpec, req.loras.len) catch return;
        for (req.loras, loras) |src, *dst| dst.* = .{ .path = a.dupe(u8, src.path) catch "", .strength = src.strength };
        req.loras = loras;
        self.post(cfg, .{ .img_enqueue = req });
    }

    /// Clear away a render that is over, and stop replaying it: the row was the
    /// last thing pointing at it.
    pub fn forget(self: *Hosts, id: wire.ImageId) bool {
        const s = self.imageOwner(id) orelse return false;
        const ref = (s.mirror.byId(id) orelse return false).info.client_ref;
        if (!s.mirror.forget(id)) return false;
        if (self.askedOf(ref)) |a| {
            a.waiting = false;
            a.gone = true;
        }
        return true;
    }

    /// The image this client already holds for `path`, whichever host's mirror
    /// it landed in. A reopened conversation must find its pictures rather than
    /// loading a second copy of each.
    pub fn bySavedPath(self: *Hosts, path: []const u8) ?*mirror.Image {
        for (self.slots.items) |s| if (s.mirror.bySavedPath(path)) |im| return im;
        return null;
    }

    /// Add an image this client built itself. Always the local host's mirror:
    /// it is the one slot that cannot be renamed or unlisted out from under a
    /// transcript that refers to these ids.
    pub fn addLocal(self: *Hosts, info: wire.ImageInfo, pixels: ?[]u8, saved_path: ?[]u8) wire.ImageId {
        return self.local().mirror.addLocal(info, pixels, saved_path);
    }

    /// This picture's file will not open, wherever it is held.
    pub fn markLost(self: *Hosts, id: wire.ImageId, why: []const u8) void {
        const s = self.imageOwner(id) orelse return;
        s.mirror.markLost(id, why);
    }

    /// Longest side previews are fetched at. A view sets it to what it draws:
    /// a 150 px rail thumbnail has no use for a 4 MiB frame, and on a remote
    /// host those frames are what the finished picture queues behind.
    pub fn setPreviewMaxEdge(self: *Hosts, edge: u32) void {
        if (self.preview_max_edge == edge) return;
        self.preview_max_edge = edge;
        for (self.slots.items) |s| s.mirror.preview_max_edge = edge;
    }

    /// The host that would take a render of this shape, for a view that has to
    /// show what it would cost before anyone presses Generate.
    pub fn renderTarget(self: *Hosts, cfg: *const config.Config, width: u32, height: u32, steps: u32) *Slot {
        return switch (self.placeRender(cfg, width, height, steps)) {
            .place => |id| self.slotOf(id) orelse self.chatSlot(),
            else => self.chatSlot(),
        };
    }

    /// More than one host: the rail names the host on each job.
    pub fn several(self: *const Hosts) bool {
        return self.slots.items.len > 1;
    }

    /// A connect is in flight somewhere.
    pub fn connecting(self: *const Hosts) bool {
        for (self.slots.items) |s| if (s.connecting != null) return true;
        return false;
    }

    /// One word for the settings list: what a listed host is doing.
    pub fn statusOf(self: *Hosts, name: []const u8) []const u8 {
        const s = self.slotByName(name) orelse return "";
        return if (troubleOf(s)) |t| t else "up";
    }

    /// What is wrong with this host's connection, in words a person can act
    /// on, or null while it is answering.
    pub fn troubleOf(s: *const Slot) ?[]const u8 {
        if (s.up()) return null;
        if (s.connecting != null) return "connecting";
        if (s.status_len == 0) return if (s.lost) "lost" else "not connected";
        return troubleText(s.status());
    }

    /// A connect error's name said plainly. Anything not listed shows as
    /// itself rather than as a wrong guess.
    pub fn troubleText(err_name: []const u8) []const u8 {
        const map = [_][2][]const u8{
            .{ "Unauthorized", "token refused: re-pair this host" },
            .{ "TlsCertificateNotVerified", "certificate does not match the pairing string" },
            .{ "CertificateSignatureInvalid", "certificate does not match the pairing string" },
            .{ "CertificateExpired", "this host's certificate has expired" },
            .{ "BadPairingString", "the pairing string is not valid" },
            .{ "ProtoMismatch", "this host speaks a different protocol version" },
            .{ "ChildDidNotStart", "the host process did not start" },
            .{ "HelloRefused", "the host refused the connection" },
            .{ "FileNotFound", "nothing is listening there" },
            .{ "ConnectionRefused", "nothing is listening there" },
            .{ "Unexpected", "nothing is listening there" },
            .{ "AddressInUse", "something else holds that socket" },
            .{ "PlainTcpRefused", "a remote host needs a pairing string, not a port" },
        };
        for (map) |m| if (std.mem.eql(u8, err_name, m[0])) return m[1];
        return err_name;
    }

    // ── Connecting ────────────────────────────────────────────────────────────

    /// Bring the slots in line with the settings: the local child, then each
    /// enabled entry, in order. A new host starts connecting; a removed or
    /// renamed one is closed (a child it spawned leaves with it).
    pub fn sync(self: *Hosts, cfg: *const config.Config) void {
        // A host is the same host while it is reachable the same way: a meter
        // handle that moved is not a new host and must not drop the connection.
        var i: usize = 1;
        while (i < self.slots.items.len) {
            const s = self.slots.items[i];
            const keep = if (cfg.hostEntry(s.name)) |e| e.enabled and e.sameEndpoint(&s.entry.?) else false;
            if (keep) {
                i += 1;
                continue;
            }
            log.info("host {s} removed", .{s.name});
            // Its mirror goes with it, so a render it was given is unreachable
            // unless the queue takes it back now.
            self.releasePlaced(s);
            _ = self.slots.orderedRemove(i);
            self.freeSlot(s);
        }
        for (cfg.hosts.slice()) |*e| {
            if (!e.enabled or e.name.opt() == null) continue;
            if (self.slotByName(e.name.slice())) |s| {
                s.entry = e.*;
                s.meter = .{ .split = e.vram_split, .limit = e.vram_limit_frac };
                continue;
            }
            const s = self.addSlot(e.name.slice(), e.*) catch continue;
            s.meter = .{ .split = e.vram_split, .limit = e.vram_limit_frac };
            self.startConnect(s);
        }
        const l = self.local();
        l.meter = .{ .split = cfg.vram_split, .limit = cfg.vram_limit_frac };
        if (l.remote == null and l.connecting == null and !l.lost) self.startConnect(l);
        // The chat pin follows the settings; a name that is not there means local.
        const want: HostId = if (cfg.chat_host.opt()) |n| (if (self.slotByName(n)) |s| s.id else 0) else 0;
        if (want != self.chat) self.setChat(want);
    }

    pub fn slotByName(self: *Hosts, name: []const u8) ?*Slot {
        for (self.slots.items) |s| if (std.mem.eql(u8, s.name, name)) return s;
        return null;
    }

    /// Begin reaching `s` on a thread; `pump` collects the answer.
    fn startConnect(self: *Hosts, s: *Slot) void {
        std.debug.assert(s.connecting == null);
        s.connecting = self.spawnConnect(s) catch |err| {
            s.setStatus(@errorName(err));
            s.tries +|= 1;
            s.retry_at = Io.Clock.awake.now(self.io).nanoseconds + backoff(s.tries);
            return;
        };
    }

    fn spawnConnect(self: *Hosts, s: *Slot) !*Connecting {
        const gpa = self.gpa;
        var args: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (args.items) |a| gpa.free(a);
            args.deinit(gpa);
        }
        if (self.config_path) |p| {
            try args.append(gpa, try gpa.dupe(u8, "--config"));
            try args.append(gpa, try gpa.dupe(u8, p));
        }
        var how: Connecting.How = .local_default;
        if (s.entry) |*e| {
            if (!e.remote() and e.spawn) {
                // Its own catalog cache: two children on one disk finishing a
                // scan together must not write one file.
                var name_buf: [config.max_host_name + 16]u8 = undefined;
                const fname = std.fmt.bufPrint(&name_buf, "catalog-{s}.json", .{s.name}) catch "catalog-host.json";
                if (config.Config.siblingPath(self.io, gpa, self.environ, self.config_path, fname) catch null) |p| {
                    errdefer gpa.free(p);
                    try args.append(gpa, try gpa.dupe(u8, "--index"));
                    try args.append(gpa, p);
                }
            }
            how = .{ .entry = e.* };
        } else if (self.local_socket) |sock| {
            how = .{ .local_socket = try gpa.dupe(u8, sock) };
        }
        errdefer if (how == .local_socket) gpa.free(how.local_socket);
        const c = try gpa.create(Connecting);
        errdefer gpa.destroy(c);
        const owned = try args.toOwnedSlice(gpa);
        c.* = .{
            .gpa = gpa,
            .io = self.io,
            .environ = self.environ,
            .on_out = self.on_out,
            .how = how,
            .args = owned,
        };
        const th = std.Thread.spawn(.{}, Connecting.run, .{c}) catch |err| {
            for (owned) |a| gpa.free(a);
            gpa.free(owned);
            return err;
        };
        th.detach();
        return c;
    }

    /// Take a finished connect: greet a host that answered, or schedule the
    /// next try.
    fn finishConnect(self: *Hosts, s: *Slot, cfg: *const config.Config, now: i96) void {
        const c = s.connecting orelse return;
        switch (c.state.load(.acquire)) {
            .running => return,
            .failed => {
                s.connecting = null;
                s.setStatus(c.errName());
                c.release();
                s.tries +|= 1;
                if (std.mem.eql(u8, s.status(), "BadPairingString")) {
                    s.lost = true;
                    log.err("host {s}: {s}; fix its pairing string in Settings", .{ s.name, s.status() });
                    return;
                }
                const wait = backoff(s.tries);
                s.retry_at = now + wait;
                log.warn("host {s} unreachable ({s}); retrying in {d}s", .{ s.name, s.status(), @divTrunc(wait, std.time.ns_per_s) });
            },
            .ok => {
                s.connecting = null;
                const r = c.take().?;
                c.release();
                if (s.remote) |old| old.deinit();
                s.remote = r;
                s.status_len = 0;
                const restarted = if (s.last_gen) |g| g != r.gen else true;
                if (s.down) {
                    s.reconnects +|= 1;
                    log.info("host {s} back after {d} tries (gen {d}{s})", .{ s.name, s.tries, r.gen, if (restarted) ", restarted" else "" });
                } else log.info("host {s} up (gen {d})", .{ s.name, r.gen });
                s.down = false;
                s.last_gen = r.gen;
                s.tries = 0;
                s.retry_at = 0;
                var arena = std.heap.ArenaAllocator.init(self.gpa);
                defer arena.deinit();
                var adopt: []const wire.Message = &.{};
                if (restarted) {
                    self.hostRestarted(s);
                    if (s.id == self.chat) adopt = s.mirror.toWire(arena.allocator()) catch &.{};
                } else self.cancelStrays(s);
                self.greet(s, cfg, adopt);
            },
        }
    }

    /// A render handed to a host that never listed an image for it, on a link
    /// that has now dropped: the enqueue may never have arrived, so it goes back
    /// in the queue for another host. Left alone it would be lost AND count
    /// against that host's room for as long as the client runs. The host may
    /// still have taken it; `cancelStrays` is what stops it there when the same
    /// daemon answers.
    fn requeueUnlisted(self: *Hosts, s: *Slot) void {
        for (self.asked.items) |*a| {
            if (a.waiting or a.gone or a.at != s.id) continue;
            if (s.mirror.byClientRef(a.ref) != null) continue;
            a.waiting = true;
            log.info("queue: {s} never took a render; back in the queue", .{s.name});
        }
    }

    /// Renders this host is holding that this client has since put elsewhere or
    /// put back in the queue: the enqueue arrived after the link dropped, or the
    /// client gave up on one that was still running. Stop each there, once.
    /// The pass every frame is what catches an enqueue the mirror only learns
    /// about from the reconnect's snapshot.
    fn cancelStrays(self: *Hosts, s: *Slot) void {
        if (!s.up()) return;
        for (s.mirror.images.items) |*im| {
            if (im.local or im.cancel_sent or im.info.client_ref == 0) continue;
            switch (im.info.status) {
                .pending, .generating, .suspended => {},
                // Failed only as this client's guess when the link dropped: the
                // host may well still be rendering it. A real failure is over.
                .failed => if (!std.mem.eql(u8, im.info.failure, "HostLost")) continue,
                .done, .canceled => continue,
            }
            const a = self.askedOf(im.info.client_ref) orelse continue;
            if (!a.waiting and a.at == s.id) continue;
            im.cancel_sent = true;
            s.post(.{ .img_cancel = .{ .image = im.info.id } });
        }
    }

    /// A different daemon answers where one was: its images become this
    /// client's, and it holds none of the renders the old one was given.
    fn hostRestarted(self: *Hosts, s: *Slot) void {
        s.mirror.hostRestarted();
        self.releasePlaced(s);
    }

    /// Records still placed at `s` that it does not hold any more: it restarted,
    /// or it is being unlisted. A render this client already has is over;
    /// anything else goes back in the queue for another host. Left alone they
    /// would count against that host's room for as long as the client runs.
    fn releasePlaced(self: *Hosts, s: *Slot) void {
        for (self.asked.items) |*a| {
            if (a.waiting or a.gone or a.at != s.id) continue;
            const over = if (s.mirror.byRef(a.ref)) |im| switch (im.status()) {
                .done, .canceled => true,
                else => false,
            } else false;
            if (over) a.gone = true else a.waiting = true;
        }
    }

    /// What a freshly connected host is told, in order: the settings, the
    /// transcript it should carry on (a restart, or a re-pin), a snapshot of
    /// what it holds, the meter handles, the folders to scan.
    fn greet(self: *Hosts, s: *Slot, cfg: *const config.Config, adopt: []const wire.Message) void {
        _ = self.postSettingsTo(s, cfg);
        if (adopt.len > 0) s.post(.{ .chat_adopt = .{ .messages = adopt } });
        s.post(.snapshot);
        s.postMeter();
        postScanTo(s, cfg);
    }

    /// False while the settings are being held back, so a caller does not
    /// announce a send that did not happen (and does not say so every frame).
    fn postSettingsTo(self: *Hosts, s: *Slot, cfg: *const config.Config) bool {
        const r = s.remote orelse return false;
        // A remote host is told nothing until this machine can name its files;
        // paths it must drop would leave it with an empty model set.
        if (s.entry != null and s.entry.?.remote() and !self.canNameModels()) {
            if (!s.settings_held) {
                s.settings_held = true;
                log.info("host {s}: holding settings until this machine's catalog arrives", .{s.name});
            }
            return false;
        }
        s.settings_held = false;
        var translated: config.Config = undefined;
        const send: *const config.Config = if (s.entry != null and s.entry.?.remote()) blk: {
            translated = cfg.*;
            self.refsToIds(&translated);
            break :blk &translated;
        } else cfg;
        var hs = config.hostSettings(send);
        // Each host has its own card, so its own handles.
        hs.vram_split = s.meter.split;
        hs.vram_limit_frac = s.meter.limit;
        // Whether the model is offered the image tool is a fact about the whole
        // fleet, and only this client knows it: the host carrying the chat may
        // hold no checkpoint and still have somewhere to send a render.
        hs.image_tool = self.canRenderAnywhere(cfg);
        const json = std.json.Stringify.valueAlloc(r.gpa, hs, .{}) catch |err| {
            log.err("encode settings: {t}", .{err});
            return false;
        };
        defer r.gpa.free(json);
        s.post(.{ .settings = .{ .json = json } });
        s.settings_catalog_seq = s.mirror.catalog_seq;
        return true;
    }

    /// Can any host render at all. What decides whether the model is told the
    /// image tool exists, since the host carrying the chat does not render.
    pub fn canRenderAnywhere(self: *Hosts, cfg: *const config.Config) bool {
        var ids: [4]catalog.ModelId = undefined;
        const need = self.imageIds(cfg, &ids);
        for (self.slots.items) |s| {
            if (!s.up() or !s.mirror.spokeOnce()) continue;
            if (!s.mirror.state.diff_present or !holdsAll(s, need)) continue;
            return true;
        }
        return false;
    }

    /// Can this machine turn the paths it holds into ids yet: only once its
    /// own catalog has arrived.
    pub fn canNameModels(self: *Hosts) bool {
        return self.local().mirror.scannedOnce();
    }

    /// Every model reference that is a path of this machine's becomes its id.
    /// An id, or a path this machine's catalog does not know, is left as is.
    fn refsToIds(self: *Hosts, cfg: *config.Config) void {
        const cat = &self.local().mirror.catalog;
        inline for (config.model_ref_fields) |name| refToId(&@field(cfg, name), cat);
        for (cfg.loras.items[0..cfg.loras.count]) |*l| refToId(&l.path, cat);
    }

    fn refToId(field: *config.PathBuf, cat: *const catalog.Catalog) void {
        const text = field.opt() orelse return;
        if (catalog.parseId(text) != null) return;
        const e = cat.find(text) orelse return;
        var buf: [catalog.id_text_len]u8 = undefined;
        field.set(catalog.idText(e.id(), &buf));
    }

    /// The image model file `cfg` names that this machine has and `s` does
    /// not: what the settings row offers to send. Null for a local host, a
    /// host that is down or has not sent its catalog, or when nothing is missing.
    pub fn missingModel(self: *Hosts, s: *Slot, cfg: *const config.Config) ?[]const u8 {
        const e = s.entry orelse return null;
        if (!e.remote() or !s.up()) return null;
        if (!s.mirror.scannedOnce()) return null;
        // EVERY model the settings name, not just the render's: a host carrying
        // the chat needs the chat model and its vision tower as much as a host
        // rendering needs the VAE, and `model_ref_fields` is the list the
        // engine itself reads.
        inline for (config.model_ref_fields) |name| {
            if (@field(cfg, name).opt()) |ref| {
                if (catalog.parseId(ref) == null) {
                    if (self.local().mirror.catalog.find(ref)) |mine| {
                        if (s.mirror.catalog.byId(mine.id()) == null) return ref;
                    }
                }
            }
        }
        return null;
    }

    /// How many models the settings name that this host does not hold, of
    /// every kind. `missingFiles` answers the narrower question a RENDER asks.
    pub fn missingAll(self: *Hosts, s: *Slot, cfg: *const config.Config) usize {
        if (!s.mirror.scannedOnce()) return 0;
        var n: usize = 0;
        inline for (config.model_ref_fields) |name| {
            if (@field(cfg, name).opt()) |ref| {
                if (catalog.parseId(ref) == null) {
                    if (self.local().mirror.catalog.find(ref)) |mine| {
                        if (s.mirror.catalog.byId(mine.id()) == null) n += 1;
                    }
                }
            }
        }
        return n;
    }

    /// Every file an image render needs. The preview decoder is left out:
    /// without it a render loses its live preview, not its picture.
    fn imageFields(cfg: *const config.Config) [4]*const config.PathBuf {
        return .{ &cfg.diffusion_model, &cfg.text_encoder, &cfg.text_encoder_2, &cfg.vae };
    }

    fn postScanTo(s: *Slot, cfg: *const config.Config) void {
        var dirs: [config.max_model_dirs][]const u8 = undefined;
        var files: [config.max_model_dirs][]const u8 = undefined;
        s.post(.{ .scan = .{ .dirs = paths(&cfg.model_dirs, &dirs), .files = paths(&cfg.model_files, &files) } });
    }

    fn paths(list: *const config.ModelDirList, out: *[config.max_model_dirs][]const u8) []const []const u8 {
        var n: usize = 0;
        for (list.slice()) |*d| {
            out[n] = d.path.opt() orelse continue;
            n += 1;
        }
        return out[0..n];
    }

    /// Every host gets the settings as they stand.
    pub fn postSettings(self: *Hosts, cfg: *const config.Config) void {
        for (self.slots.items) |s| if (s.up()) {
            _ = self.postSettingsTo(s, cfg);
        };
    }

    /// Every host scans its folders again.
    pub fn postScan(self: *Hosts, cfg: *const config.Config) void {
        for (self.slots.items) |s| if (s.up()) postScanTo(s, cfg);
    }

    /// Move the chat to `id`: the new host adopts the transcript as this client
    /// holds it, the old one starts a fresh conversation. A turn running on
    /// either is cut short first.
    pub fn setChat(self: *Hosts, id: HostId) void {
        const to = self.slotOf(id) orelse return;
        const from = self.chatSlot();
        if (to == from) {
            self.chat = id;
            return;
        }
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        to.post(.chat_cancel);
        if (from.mirror.toWire(arena.allocator())) |msgs| {
            if (msgs.len > 0) to.post(.{ .chat_adopt = .{ .messages = msgs } });
        } else |_| {}
        from.post(.chat_cancel);
        from.post(.chat_new);
        // The old host's message indices name nothing here now.
        self.unbindAll();
        self.chat = id;
        log.info("chat pinned to {s}", .{to.name});
    }

    // ── Per frame ─────────────────────────────────────────────────────────────

    /// Every frame each host delivered, into its mirror; connects that finished
    /// are taken, lost hosts are reached for again, and waiting renders are
    /// handed out.
    pub fn pump(self: *Hosts, cfg: *const config.Config) void {
        // Place first, then report: an image has no id until a host takes it.
        defer self.reportModelImages(cfg);
        defer self.dispatch(cfg);
        const local_seq = self.local().mirror.catalog_seq;
        const now = Io.Clock.awake.now(self.io).nanoseconds;
        const can_render = self.canRenderAnywhere(cfg);
        const tool_moved = self.told_image_tool != null and self.told_image_tool.? != can_render;
        self.told_image_tool = can_render;
        for (self.slots.items) |s| {
            self.drain(s);
            self.measure(s);
            self.cancelStrays(s);
            self.takeModelCalls(s);
            // A remote host's ids come from THIS machine's catalog, so a move
            // there is as much a reason to re-send as a move in its own.
            if (s.up() and s.entry != null and s.entry.?.remote() and s.settings_local_seq != local_seq) {
                s.settings_local_seq = local_seq;
                _ = self.postSettingsTo(s, cfg);
            }
            if (s.up() and tool_moved) _ = self.postSettingsTo(s, cfg);
            if (s.up() and s.mirror.catalog_seq != s.settings_catalog_seq) {
                if (self.postSettingsTo(s, cfg)) log.info("host {s}: catalog of {d} entries (rev {d}); settings re-sent", .{
                    s.name, s.mirror.catalog.entries.len, s.mirror.catalog_rev,
                });
            }
            self.logWhy(s, cfg);
            if (s.connecting != null) {
                self.finishConnect(s, cfg, now);
                continue;
            }
            if (s.lost) continue;
            if (s.remote) |r| {
                if (!r.failed.load(.acquire)) continue;
                if (!s.down) {
                    s.down = true;
                    log.warn("host {s} is gone; reconnecting", .{s.name});
                    s.mirror.linkDown();
                    self.requeueUnlisted(s);
                }
            }
            if (now < s.retry_at) continue;
            self.startConnect(s);
        }
    }

    /// Why a host is taking no renders, once per change. The screen says this
    /// in four words; the log gets the sentence and the count behind it.
    fn logWhy(self: *Hosts, s: *Slot, cfg: *const config.Config) void {
        const why: []const u8 = if (self.whyNotRender(s, cfg)) |w| w.long else "";
        if (why.ptr == s.why_logged.ptr and why.len == s.why_logged.len) return;
        s.why_logged = why;
        if (why.len == 0) return log.info("host {s} can take renders", .{s.name});
        var ids: [4]catalog.ModelId = undefined;
        const need = self.imageIds(cfg, &ids);
        log.info("host {s} takes no renders: {s} (needs {d} files, short {d}; image engine {})", .{
            s.name, why, need.len, missingCount(s, need), s.mirror.state.diff_present,
        });
    }

    fn drain(self: *Hosts, s: *Slot) void {
        const r = s.remote orelse return;
        r.take(&s.frames);
        for (s.frames.items) |f| {
            s.mirror.apply(f);
            f.deinit(self.gpa);
        }
        s.frames.clearRetainingCapacity();
    }

    /// How long before the `n`th connect attempt: a second, doubling to half a
    /// minute.
    fn backoff(n: u32) i96 {
        const shift: u6 = @intCast(@min(n -| 1, 5));
        return @min(@as(i96, 1) << shift, 30) * std.time.ns_per_s;
    }

    // ── Placement ─────────────────────────────────────────────────────────────

    /// Fold every finished image this host has not been counted for into its
    /// own table. A failed or cancelled image says nothing about speed.
    fn measure(self: *Hosts, s: *Slot) void {
        for (s.mirror.images.items) |*im| {
            if (im.local or im.status() != .done) continue;
            if (s.measured.contains(im.info.id)) continue;
            s.measured.put(self.gpa, im.info.id, {}) catch return;
            _ = s.table.note(.{
                .family = im.info.family,
                .mpx = megapixels(im.info.width, im.info.height),
                .steps = im.info.total,
                .start_ns = im.info.start_ns,
                .first_step_ns = im.info.first_step_ns,
                .last_step_ns = im.info.last_step_ns,
            });
        }
    }

    fn megapixels(width: u32, height: u32) f64 {
        return @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height)) / 1_000_000;
    }

    /// The render the settings and these dimensions describe. `vram_need` is
    /// the measured peak when the settings hold one for this very model, else
    /// the checkpoint's size.
    pub fn jobFor(self: *Hosts, cfg: *const config.Config, width: u32, height: u32, steps: u32) sched.Job {
        var j: sched.Job = .{ .mpx = megapixels(width, height), .steps = steps };
        const ref = cfg.diffusion_model.opt() orelse return j;
        const e = self.local().mirror.catalog.resolve(ref) orelse return j;
        if (e.ckpt) |c| j.family = @tagName(c.family);
        j.vram_need = if (cfg.diff_peak_resident > 0 and cfg.diff_peak_key == config.modelKey(ref))
            cfg.diff_peak_resident
        else
            e.size;
        return j;
    }

    /// Two lengths of the same fact: the settings row has a line to spend,
    /// the meter has the width of a host name.
    pub const NotRendering = struct {
        long: []const u8,
        short: []const u8,
    };

    /// Why this host is not taking renders, or null when it is a candidate.
    pub fn whyNotRender(self: *Hosts, s: *Slot, cfg: *const config.Config) ?NotRendering {
        if (!s.up()) return null; // its own status already says so
        // Until the first state block lands every field is a default, and a
        // default reads exactly like "no image engine".
        if (!s.mirror.spokeOnce()) return .{
            .long = "connected; waiting for it to describe itself",
            .short = "starting up",
        };
        // First because it is the one the user just did; resume it and
        // whatever else is wrong says so next.
        if (s.mirror.state.diff_paused) return .{
            .long = "paused here: it takes no new renders until it is resumed",
            .short = "paused",
        };
        var ids: [4]catalog.ModelId = undefined;
        if (!holdsAll(s, self.imageIds(cfg, &ids))) return .{
            .long = "does not have every file this model needs",
            .short = "missing files",
        };
        if (!s.mirror.state.diff_present) return .{
            .long = "no image model loaded there",
            .short = "no image model",
        };
        return null;
    }

    /// Every model file an image render needs, as the ids two catalogs agree
    /// on. A host holding the checkpoint but not the text encoder loads nothing.
    fn imageIds(self: *Hosts, cfg: *const config.Config, out: *[4]catalog.ModelId) []const catalog.ModelId {
        var n: usize = 0;
        for (imageFields(cfg)) |field| {
            const ref = field.opt() orelse continue;
            const id = catalog.parseId(ref) orelse blk: {
                const e = self.local().mirror.catalog.find(ref) orelse continue;
                break :blk e.id();
            };
            if (id == 0) continue;
            out[n] = id;
            n += 1;
        }
        return out[0..n];
    }

    /// How many of `ids` this host does not hold. Zero while its catalog has
    /// not arrived, which is also when `holdsAll` assumes the best.
    fn missingCount(s: *Slot, ids: []const catalog.ModelId) usize {
        if (!s.mirror.scannedOnce()) return 0;
        var n: usize = 0;
        for (ids) |id| if (s.mirror.catalog.byId(id) == null) {
            n += 1;
        };
        return n;
    }

    /// How many files this host is short of for the configured render.
    pub fn missingFiles(self: *Hosts, s: *Slot, cfg: *const config.Config) usize {
        var ids: [4]catalog.ModelId = undefined;
        return missingCount(s, self.imageIds(cfg, &ids));
    }

    /// Does this host hold every file that render needs. Before its catalog
    /// arrives nothing is known, and assuming it does is the answer that does
    /// not offer to send a file it may already have.
    fn holdsAll(s: *Slot, ids: []const catalog.ModelId) bool {
        if (!s.mirror.scannedOnce()) return true;
        for (ids) |id| if (s.mirror.catalog.byId(id) == null) return false;
        return true;
    }

    /// Renders handed to `s` that it has not listed yet. They count against
    /// its room, or one free host would be given the whole queue in a frame.
    fn unseen(self: *Hosts, s: *Slot) u32 {
        var n: u32 = 0;
        for (self.asked.items) |*a| {
            if (a.waiting or a.gone or a.at != s.id) continue;
            if (s.mirror.byClientRef(a.ref) == null) n += 1;
        }
        return n;
    }

    /// Seconds left on the render running at `s`, 0 when none or unknown.
    fn remainingOn(s: *Slot) f64 {
        for (s.mirror.images.items) |*im| {
            if (im.local or im.info.status != .generating or im.info.total <= im.info.step) continue;
            const mpx = megapixels(im.info.width, im.info.height);
            const measured = s.table.stepCost(im.info.family, mpx);
            const per_step = if (measured > 0) measured else sched.assumed_s_per_step_per_mpx * mpx;
            return per_step * @as(f64, @floatFromInt(im.info.total - im.info.step));
        }
        return 0;
    }

    /// Each host as the scheduler sees it. `held` names hosts a job earlier in
    /// this pass is waiting for, once per job.
    pub fn facts(self: *Hosts, j: sched.Job, ids: []const catalog.ModelId, held: []const HostId, out: []sched.Host) []sched.Host {
        var n: usize = 0;
        for (self.slots.items) |s| {
            if (n == out.len) break;
            const tel = &s.mirror.telemetry;
            const st = &s.mirror.state;
            var held_here: u32 = 0;
            for (held) |id| held_here += @intFromBool(id == s.id);
            const card = if (tel.limit > 0) tel.limit else tel.vram_total;
            out[n] = .{
                .id = s.id,
                .up = s.up(),
                .ready = st.diff_present and !st.diff_paused,
                .has_model = holdsAll(s, ids),
                .chat = s.id == self.chat,
                .vram_limit = card -| tel.llm_used,
                .busy = st.diff_busy,
                .busy_remaining_s = remainingOn(s),
                .queued = st.pending_images + self.unseen(s) + held_here,
                .resident_family = st.diff_family,
                .s_per_step = s.table.stepCost(j.family, j.mpx),
                .load_s = s.table.loadCost(j.family),
            };
            n += 1;
        }
        return out[0..n];
    }

    // ── The queue ─────────────────────────────────────────────────────────────

    /// One render THIS client asked for. It stays after a host takes it so a
    /// failure can be put to another host exactly as asked: rebuilding the
    /// request from the `ImageInfo` that comes back would drop the LoRA stack.
    ///
    /// An image the chat model asked a host for was never ours; it is not
    /// remembered, not queued here, and never replayed, only reported.
    pub const Asked = struct {
        ref: u64,
        req: wire.ImageRequest,
        /// No host has it yet.
        waiting: bool = true,
        /// The host that took it, once one did.
        at: HostId = 0,
        /// The row is cleared away: nothing is outstanding at `at` any more.
        /// The record itself stays, because it is what stops a re-parsed tool
        /// call from being placed a second time.
        gone: bool = false,
        /// Hosts that have already had this render and failed it.
        tried: [config.max_hosts + 1]HostId = @splat(0),
        n_tried: usize = 0,
        /// Set for a render the model asked for: the conversation to report it
        /// back to, which is not always the host that renders it.
        bind: ?Bind = null,
        /// The image that conversation was last told this call became. A
        /// replay mints a new one somewhere else, and the transcript has to
        /// follow it rather than keep pointing at the render that failed.
        told_id: wire.ImageId = 0,
        /// ...and how it turned out.
        outcome_told: bool = false,

        fn was(self: *const Asked, id: HostId) bool {
            for (self.tried[0..self.n_tried]) |t| if (t == id) return true;
            return false;
        }

        fn mark(self: *Asked, id: HostId) void {
            if (self.was(id) or self.n_tried == self.tried.len) return;
            self.tried[self.n_tried] = id;
            self.n_tried += 1;
        }

        fn free(self: *Asked, gpa: std.mem.Allocator) void {
            gpa.free(self.req.prompt);
            gpa.free(self.req.negative);
            for (self.req.loras) |l| gpa.free(l.path);
            gpa.free(self.req.loras);
        }
    };

    /// Renders a host may hold at once: the one it is rendering. Nothing is
    /// committed to a machine before it is ready for it, so a host that slows
    /// down or dies is never sitting on work.
    pub const depth_per_host: u32 = 1;

    /// Finished jobs kept for replay. A waiting one is never dropped.
    const asked_keep = 16;

    /// Where the references this client mints for jobs no host has yet start.
    /// They share a namespace with image ids in the rail, so they come out of
    /// the queue-reference half of the client's space (`mirror.isQueueRef`),
    /// never the half its own images are minted in.
    pub const client_ref_base: u64 = mirror.queue_ref_base;

    fn remember(self: *Hosts, req: wire.ImageRequest) u64 {
        var kept: usize = 0;
        var i = self.asked.items.len;
        while (i > 0) {
            i -= 1;
            if (self.asked.items[i].waiting) continue;
            kept += 1;
            if (kept > asked_keep) {
                var old = self.asked.orderedRemove(i);
                old.free(self.gpa);
            }
        }
        self.next_ref += 1;
        var owned = req;
        owned.client_ref = client_ref_base | self.next_ref;
        owned.prompt = self.gpa.dupe(u8, req.prompt) catch return 0;
        owned.negative = self.gpa.dupe(u8, req.negative) catch {
            self.gpa.free(owned.prompt);
            return 0;
        };
        const loras = self.gpa.alloc(wire.LoraSpec, req.loras.len) catch {
            self.gpa.free(owned.prompt);
            self.gpa.free(owned.negative);
            return 0;
        };
        for (req.loras, loras) |src, *dst| {
            dst.* = .{ .path = self.gpa.dupe(u8, src.path) catch "", .strength = src.strength };
        }
        owned.loras = loras;
        self.asked.append(self.gpa, .{ .ref = owned.client_ref, .req = owned }) catch {
            var tmp: Asked = .{ .ref = 0, .req = owned };
            tmp.free(self.gpa);
            return 0;
        };
        return owned.client_ref;
    }

    /// Put every render the model asked for into this client's queue, bound to
    /// the reply that asked. The host parses its own tool calls and queues
    /// none of them, so these are placed like any other render and can run on
    /// a machine other than the one carrying the conversation.
    fn takeModelCalls(self: *Hosts, s: *Slot) void {
        s.mirror.takeCalls(&self.calls);
        for (self.calls.items) |*c| {
            defer c.deinit(self.gpa);
            const bind: Bind = .{ .host = s.id, .msg = c.msg, .variant = c.variant, .index = c.index };
            // A host says again what it has not seen answered, so the same call
            // arrives more than once; this client knows what it already took.
            if (self.alreadyTook(bind)) continue;
            const ref = self.remember(c.req);
            const a = self.askedOf(ref) orelse continue;
            a.bind = bind;
        }
        self.calls.clearRetainingCapacity();
    }

    /// Has this exact call already been placed by this client.
    fn alreadyTook(self: *Hosts, bind: Bind) bool {
        for (self.asked.items) |*a| {
            const b = a.bind orelse continue;
            if (std.meta.eql(b, bind)) return true;
        }
        return false;
    }

    /// The image behind a render this client asked for, at the host holding it
    /// now. A replay leaves the failed one behind under the same reference, so
    /// the host it sits at is what picks the right one.
    fn imageFor(self: *Hosts, a: *const Asked) ?*mirror.Image {
        const s = self.slotOf(a.at) orelse return null;
        // A daemon that restarted disowned its images to this client, and the
        // conversation still has to hear how this one ended.
        return s.mirror.byClientRef(a.ref) orelse s.mirror.byRef(a.ref);
    }

    /// Tell a conversation what became of the renders it asked for: which
    /// image each one is, and how it turned out. Both go to the host carrying
    /// that chat, which is not always the host that did the work.
    fn reportModelImages(self: *Hosts, cfg: *const config.Config) void {
        for (self.asked.items) |*a| {
            const bind = a.bind orelse continue;
            if (a.waiting or a.outcome_told) continue;
            const to = self.slotOf(bind.host) orelse continue;
            if (!to.up()) continue;
            const im = self.imageFor(a) orelse continue;
            if (a.told_id != im.info.id) {
                to.post(.{ .chat_image = .{
                    .msg = bind.msg,
                    .variant = bind.variant,
                    .image = im.info.id,
                    .replaces = a.told_id,
                } });
                a.told_id = im.info.id;
            }
            // A failure this client has not looked at yet may still be moved to
            // another host, and the model must not be told it failed if it did.
            if (im.status() == .failed and !im.failure_told) continue;
            if (!cfg.image_tool_result) continue;
            var buf: [220]u8 = undefined;
            const text = outcomeText(&buf, im) orelse continue;
            a.outcome_told = true;
            to.post(.{ .chat_note = .{ .text = text } });
        }
    }

    /// What the model is told about a render it asked for, once it is over.
    /// Null while there is nothing to say yet.
    fn outcomeText(buf: []u8, im: *const mirror.Image) ?[]const u8 {
        return switch (im.status()) {
            .done => std.fmt.bufPrint(buf, "[image tool] finished: {d}×{d}, seed {d}", .{
                im.info.width, im.info.height, im.info.req_seed,
            }) catch null,
            .canceled => "[image tool] canceled by the user",
            .failed => std.fmt.bufPrint(buf, "[image tool] failed: {s}", .{
                mirror.failureText(im.info.failure),
            }) catch null,
            .pending, .generating, .suspended => null,
        };
    }

    /// The conversation these renders were asked in is being left, so nothing
    /// they do should be announced into the next one.
    fn unbindAll(self: *Hosts) void {
        for (self.asked.items) |*a| a.bind = null;
        for (self.slots.items) |s| s.mirror.dropCalls();
    }

    /// Hand waiting jobs to the host that would finish each soonest. A job
    /// whose best host is busy is held for it, and counts as ahead of the next
    /// job there, so the rest of the queue spreads to the others. A job nobody
    /// can take is skipped, not blocking.
    pub fn dispatch(self: *Hosts, cfg: *const config.Config) void {
        var held: [config.max_hosts + 1]HostId = undefined;
        var n_held: usize = 0;
        for (self.asked.items) |*a| {
            if (!a.waiting) continue;
            const target = self.pick(cfg, a, held[0..n_held]) orelse continue;
            if (self.hasRoom(target)) {
                self.handTo(a, target);
            } else if (n_held < held.len) {
                held[n_held] = target.id;
                n_held += 1;
            }
        }
    }

    fn handTo(self: *Hosts, a: *Asked, target: *Slot) void {
        _ = self;
        a.waiting = false;
        a.at = target.id;
        target.post(.{ .img_enqueue = a.req });
        log.info("queue: {s} took a render", .{target.name});
    }

    /// The best host for `a` among those that have not failed it, busy or not.
    fn pick(self: *Hosts, cfg: *const config.Config, a: *const Asked, held: []const HostId) ?*Slot {
        return self.placeExcluding(cfg, a.req, a.tried[0..a.n_tried], held);
    }

    /// Is this host holding fewer renders than it is allowed to, counting what
    /// it was handed and has not listed yet.
    fn hasRoom(self: *Hosts, s: *Slot) bool {
        const st = &s.mirror.state;
        if (!s.up() or !s.mirror.spokeOnce()) return false;
        if (!st.diff_present or st.diff_paused) return false;
        return st.pending_images + @intFromBool(st.diff_busy) + self.unseen(s) < depth_per_host;
    }

    /// Why a waiting render has not gone anywhere yet. A job held for the host
    /// that will finish it soonest looks identical to one nobody can run,
    /// unless the rail says which it is.
    pub const Wait = union(enum) {
        /// A host has room; it goes out this pass.
        ready,
        /// Waiting for the host that would still finish it first.
        behind: []const u8,
        /// Nowhere to put it: nothing is up, or nobody holds the model.
        nowhere,
    };

    pub fn waitReason(self: *Hosts, cfg: *const config.Config, a: *const Asked) Wait {
        const target = self.pick(cfg, a, &.{}) orelse return .nowhere;
        if (self.hasRoom(target)) return .ready;
        return .{ .behind = if (self.several()) target.name else "" };
    }

    /// Renders this reply asked for that are still in the client's own queue,
    /// so no host has minted an image for them and the transcript has nothing
    /// to point at yet. The card holds a slot for each, which is why the count
    /// has to come from here: the reply's own text says how many were asked
    /// for, but not how many are still coming.
    pub fn unplacedFor(self: *Hosts, msg: u32, variant: u32) usize {
        var n: usize = 0;
        const chat = self.chatSlot().id;
        for (self.asked.items) |*a| {
            const b = a.bind orelse continue;
            if (b.host != chat or b.msg != msg or b.variant != variant) continue;
            if (a.gone or a.told_id != 0) continue;
            n += 1;
        }
        return n;
    }

    /// Jobs no host has taken yet, in queue order.
    pub fn waiting(self: *Hosts, out: []*const Asked) []*const Asked {
        var n: usize = 0;
        for (self.asked.items) |*a| {
            if (n == out.len) break;
            if (!a.waiting) continue;
            out[n] = a;
            n += 1;
        }
        return out[0..n];
    }

    fn dropWaiting(self: *Hosts, ref: u64) void {
        for (self.asked.items, 0..) |*a, i| {
            if (a.ref != ref or !a.waiting) continue;
            var gone = self.asked.orderedRemove(i);
            gone.free(self.gpa);
            return;
        }
    }

    fn dropAllWaiting(self: *Hosts) void {
        var i: usize = 0;
        while (i < self.asked.items.len) {
            if (!self.asked.items[i].waiting) {
                i += 1;
                continue;
            }
            var gone = self.asked.orderedRemove(i);
            gone.free(self.gpa);
        }
    }

    /// Move a waiting job before `before`, or to the end of the waiting run.
    fn moveWaiting(self: *Hosts, ref: u64, before: ?u64) void {
        const from = blk: {
            for (self.asked.items, 0..) |*a, i| if (a.ref == ref and a.waiting) break :blk i;
            return;
        };
        const moved = self.asked.orderedRemove(from);
        const to = blk: {
            if (before) |b| {
                for (self.asked.items, 0..) |*a, i| if (a.ref == b) break :blk i;
            }
            break :blk self.asked.items.len;
        };
        self.asked.insert(self.gpa, to, moved) catch {
            self.asked.append(self.gpa, moved) catch {};
        };
    }

    fn askedOf(self: *Hosts, ref: u64) ?*Asked {
        if (ref == 0) return null;
        for (self.asked.items) |*a| if (a.ref == ref) return a;
        return null;
    }

    /// A render that failed, and what was done about it.
    pub const Failure = struct {
        host: []const u8,
        /// Why, in words.
        why: []const u8,
        /// The host it was put to instead, null when it went to none.
        moved_to: ?[]const u8 = null,
        /// Back in the queue: a host can take it, none is free yet.
        requeued: bool = false,
    };

    /// The next failed render nobody has been told about, marked as told, and
    /// put to another host where one can take it. Call until it returns null.
    ///
    /// A picture whose file is gone is not one of them: no host was asked and
    /// none failed, so the card that shows it is the whole of the news.
    pub fn takeFailure(self: *Hosts, cfg: *const config.Config) ?Failure {
        for (self.slots.items) |s| {
            for (s.mirror.images.items) |*im| {
                if (im.missing or im.info.status != .failed or im.failure_told) continue;
                im.failure_told = true;
                var f: Failure = .{ .host = s.name, .why = mirror.failureText(im.info.failure) };
                if (self.replay(cfg, im.info.client_ref, s.id)) |to| {
                    if (to.len == 0) f.requeued = true else f.moved_to = to;
                }
                return f;
            }
        }
        return null;
    }

    /// Put the render behind `ref` to a host that has not already failed it.
    /// Null when this client did not ask for it, or nowhere else can take it.
    fn replay(self: *Hosts, cfg: *const config.Config, ref: u64, from: HostId) ?[]const u8 {
        const a = self.askedOf(ref) orelse return null;
        a.mark(from);
        const target = self.pick(cfg, a, &.{}) orelse return null;
        if (!self.hasRoom(target)) {
            a.waiting = true;
            return "";
        }
        self.handTo(a, target);
        log.info("image failed on host {d}; put to {s}", .{ from, target.name });
        return target.name;
    }

    /// The best host for `req` that is not in `skip`.
    fn placeExcluding(self: *Hosts, cfg: *const config.Config, req: wire.ImageRequest, skip: []const HostId, held: []const HostId) ?*Slot {
        var buf: [config.max_hosts + 1]sched.Host = undefined;
        const j = self.jobFor(cfg, req.width, req.height, req.steps);
        var ids: [4]catalog.ModelId = undefined;
        const f = self.facts(j, self.imageIds(cfg, &ids), held, &buf);
        for (f) |*h| {
            for (skip) |id| if (h.id == id) {
                h.up = false;
            };
        }
        return switch (sched.place(f, j)) {
            .place => |id| self.slotOf(id),
            else => null,
        };
    }

    /// What the scheduler says about this render, as a verdict.
    pub fn placeRender(self: *Hosts, cfg: *const config.Config, width: u32, height: u32, steps: u32) sched.Verdict {
        var buf: [config.max_hosts + 1]sched.Host = undefined;
        const j = self.jobFor(cfg, width, height, steps);
        var ids: [4]catalog.ModelId = undefined;
        return sched.place(self.facts(j, self.imageIds(cfg, &ids), &.{}, &buf), j);
    }

    /// Put a render in this client's queue and answer with the reference it is
    /// known by, for a caller that has to follow this one image. The request's
    /// own `client_ref` is not kept: the reference is this queue's to mint.
    pub fn enqueueImage(self: *Hosts, cfg: *const config.Config, req: wire.ImageRequest) u64 {
        const ref = self.remember(req);
        self.dispatch(cfg);
        return ref;
    }

    /// Send a request where it belongs: an image verb to the image's host, a
    /// new image into this client's queue, settings and the like to every
    /// host, the rest to the chat host.
    pub fn post(self: *Hosts, cfg: *const config.Config, req: wire.Request) void {
        if (req == .img_enqueue) {
            _ = self.enqueueImage(cfg, req.img_enqueue);
            return;
        }
        // A job no host has yet is this client's to cancel or reorder.
        switch (req) {
            .img_cancel => |c| if (mirror.isQueueRef(c.image)) return self.dropWaiting(c.image),
            .img_move => |m| if (mirror.isQueueRef(m.image)) return self.moveWaiting(m.image, m.before),
            .img_cancel_all => self.dropAllWaiting(),
            // A fresh conversation: what the old one asked for is no longer its
            // business, and its message indices name nothing.
            .chat_new => self.unbindAll(),
            else => {},
        }
        const target: ?*Slot = switch (req) {
            .img_fetch => |f| self.imageOwner(f.image),
            .img_cancel => |c| self.imageOwner(c.image),
            .img_move => |m| self.imageOwner(m.image),
            .img_ack => |a| self.imageOwner(a.image),
            // `meter` is per host and never broadcast; `Slot.postMeter` sends it.
            .settings, .scan, .img_cancel_all, .img_pause, .img_eject => {
                for (self.slots.items) |s| if (s.up()) s.post(req);
                return;
            },
            else => self.chatSlot(),
        };
        if (target) |s| s.post(req);
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

fn noop() void {}

/// A private local socket, so a test never reaches a daemon the user runs at
/// the default path; nothing listens there and no sibling tp-serve exists
/// beside a test binary, so every connect fails with FileNotFound.
const test_local_socket = "/nonexistent/tp-hosts-test/local.sock";

/// A remote whose links are never touched: `post` only queues, so what a slot
/// sent can be read back from `outbox`.
fn fakeRemote() remote.Remote {
    return .{ .gpa = testing.allocator, .io = testing.io, .control = undefined, .events = undefined, .on_out = noop };
}

fn newHosts(env: *std.process.Environ.Map) !Hosts {
    return Hosts.init(testing.allocator, testing.io, env, noop, null, test_local_socket);
}

fn ready(s: *Slot) void {
    s.mirror.state.diff_present = true;
    s.mirror.state_seen = true;
}

/// Pump until no connect is in flight.
fn settle(h: *Hosts, cfg: *const config.Config) !void {
    const deadline = Io.Clock.real.now(testing.io).nanoseconds + 10 * std.time.ns_per_s;
    while (h.connecting()) {
        if (Io.Clock.real.now(testing.io).nanoseconds > deadline) return error.ConnectNeverFinished;
        try Io.sleep(testing.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .real);
        h.pump(cfg);
    }
}

/// The nth request a fake remote was handed, decoded.
fn sentRequest(r: *remote.Remote, arena: std.mem.Allocator, n: usize) !wire.Request {
    return wire.decode(wire.Request, arena, r.outbox.items.items[n].text);
}

test "a job goes to the best host that has not failed it, and to nobody when none is up" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});
    const c = try h.addSlot("c", .{});
    const job: wire.ImageRequest = .{ .width = 1024, .height = 1024, .steps = 20 };
    var idle: Hosts.Asked = .{ .ref = 1, .req = job };
    try testing.expect(h.pick(&cfg, &idle, &.{}) == null);
    var ra = fakeRemote();
    var rb = fakeRemote();
    var rc = fakeRemote();
    a.remote = &ra;
    b.remote = &rb;
    c.remote = &rc;
    defer {
        a.remote = null;
        b.remote = null;
        c.remote = null;
    }
    ready(a);
    ready(b);
    // c has no image engine and never wins; a carries the chat, so b does.
    try testing.expectEqual(b, h.pick(&cfg, &idle, &.{}).?);
    // A host at its depth is still the best host; it just has no room now.
    b.mirror.state.pending_images = Hosts.depth_per_host;
    try testing.expect(!h.hasRoom(b));
    try testing.expect(h.hasRoom(a));
    b.mirror.state.pending_images = 0;
    // A host that has already failed this render never gets it again.
    idle.mark(b.id);
    try testing.expectEqual(a, h.pick(&cfg, &idle, &.{}).?);
    b.lost = true;
    try testing.expectEqual(a, h.pick(&cfg, &idle, &.{}).?);
}

test "a host's measured speed and its VRAM room decide, and a model nobody has names a host to send it to" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const gb: u64 = 1 << 30;
    const a = h.local();
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    for ([_]*Slot{ a, b }) |s| {
        ready(s);
        s.mirror.telemetry.limit = 24 * gb;
    }
    try setCatalog(a, &.{.{
        .path = "/m/dream.safetensors",
        .size = 8 * gb,
        .mtime_ns = 1,
        .ckpt = .{ .family = .sd15, .contents = .{} },
    }});
    var cfg: config.Config = .{};
    cfg.diffusion_model.set("/m/dream.safetensors");
    const j = h.jobFor(&cfg, 512, 512, 10);
    try testing.expectEqualStrings("sd15", j.family);
    try testing.expectApproxEqAbs(@as(f64, 0.262144), j.mpx, 1e-6);
    try testing.expectEqual(8 * gb, j.vram_need);

    // B does not have the file and says so; A does, chat or not.
    try setCatalog(b, &.{});
    try testing.expectEqual(sched.Verdict{ .place = a.id }, h.placeRender(&cfg, 512, 512, 10));
    // With A gone, nobody holds it: B is named as the one to send it to.
    a.remote = null;
    try testing.expectEqual(sched.Verdict{ .send_model = b.id }, h.placeRender(&cfg, 512, 512, 10));
    a.remote = &ra;
    // Once B has it, both can run it and the non-chat host wins.
    try setCatalog(b, &.{.{ .path = "id:x", .name = "dream", .size = 8 * gb, .mtime_ns = 2 }});
    try testing.expectEqual(sched.Verdict{ .place = b.id }, h.placeRender(&cfg, 512, 512, 10));
    // B measured slow, A measured fast: the measurement outweighs the chat.
    const s_ns = std.time.ns_per_s;
    try testing.expect(b.table.note(.{ .family = "sd15", .mpx = 0.262144, .steps = 10, .first_step_ns = s_ns, .last_step_ns = 91 * s_ns }));
    try testing.expect(a.table.note(.{ .family = "sd15", .mpx = 0.262144, .steps = 10, .first_step_ns = s_ns, .last_step_ns = 10 * s_ns }));
    try testing.expectEqual(sched.Verdict{ .place = a.id }, h.placeRender(&cfg, 512, 512, 10));
    // B's card is too small: a cost, not a veto. B stays a candidate.
    b.mirror.telemetry.limit = 4 * gb;
    try testing.expectEqual(sched.Verdict{ .place = a.id }, h.placeRender(&cfg, 512, 512, 10));
    a.mirror.telemetry.limit = 4 * gb;
    try testing.expect(h.placeRender(&cfg, 512, 512, 10) == .place);

    // The chat model's VRAM is not room for an image: equal cards and equal
    // speed, but the chat host holds a model that leaves too little.
    a.table = .{};
    b.table = .{};
    a.mirror.telemetry.limit = 24 * gb;
    b.mirror.telemetry.limit = 24 * gb;
    b.mirror.telemetry.llm_used = 20 * gb;
    var buf: [4]sched.Host = undefined;
    var ids: [4]catalog.ModelId = undefined;
    const f = h.facts(j, h.imageIds(&cfg, &ids), &.{}, &buf);
    try testing.expectEqual(24 * gb, f[0].vram_limit);
    try testing.expectEqual(4 * gb, f[1].vram_limit);
    try testing.expectEqual(sched.Verdict{ .place = a.id }, h.placeRender(&cfg, 512, 512, 10));
}

test "an image verb goes to the host that minted the image" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const b = try h.addSlot("b", .{});
    const bytes = try wire.encodeAlloc(testing.allocator, wire.Event{ .img = .{ .id = 0x1234_0000_0007, .status = .done } });
    defer testing.allocator.free(bytes);
    b.mirror.apply(.{ .text = bytes });
    try testing.expectEqual(b, h.imageOwner(0x1234_0000_0007).?);
    try testing.expect(h.imageOwner(5) == null);
    try testing.expect(h.imageById(0x1234_0000_0007) != null);
}

/// Replace a slot's catalog as a `catalog` event would.
fn setCatalog(s: *Slot, entries: []const catalog.Entry) !void {
    const c = try catalog.Catalog.fromEntries(testing.allocator, entries);
    s.mirror.catalog.deinit();
    s.mirror.catalog = c;
    s.mirror.catalog_rev = 1;
    s.mirror.catalog_seq += 1;
}

test "a remote host is not told settings until this machine's own catalog can name the files" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const a = h.local();
    var remote_entry: config.HostEntry = .{ .spawn = false };
    remote_entry.name.set("far");
    remote_entry.socket.set("far:7777");
    remote_entry.cert.set("AA");
    remote_entry.token.set(&[_]u8{'0'} ** 64);
    const b = try h.addSlot("far", remote_entry);

    var cfg: config.Config = .{};
    cfg.diffusion_model.set("/m/anima.safetensors");
    cfg.vae.set("/m/vae.safetensors");

    // The remote host's catalog lands first: nothing can be named yet.
    try setCatalog(b, &.{.{ .path = "id:x", .name = "anima", .size = 10, .mtime_ns = 1 }});
    try testing.expect(!h.canNameModels());

    try setCatalog(a, &.{
        .{ .path = "/m/anima.safetensors", .size = 10, .mtime_ns = 1 },
        .{ .path = "/m/vae.safetensors", .size = 20, .mtime_ns = 1 },
    });
    try testing.expect(h.canNameModels());
    var translated = cfg;
    h.refsToIds(&translated);
    try testing.expect(catalog.parseId(translated.diffusion_model.slice()) != null);
    // It has the checkpoint and not the VAE: short exactly one file.
    var ids: [4]catalog.ModelId = undefined;
    try testing.expectEqual(@as(usize, 2), h.imageIds(&cfg, &ids).len);
    try testing.expectEqual(@as(usize, 1), Hosts.missingCount(b, h.imageIds(&cfg, &ids)));
    b.mirror.state_seen = true;
    var rb = fakeRemote();
    b.remote = &rb;
    defer b.remote = null;
    try testing.expectEqualStrings("missing files", h.whyNotRender(b, &cfg).?.short);
}

test "a remote host is told model ids, a local one paths, and only a remote host is offered the file it lacks" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const a = h.local();
    try setCatalog(a, &.{
        .{ .path = "/home/me/models/dream.safetensors", .size = 2000, .mtime_ns = 1 },
        .{ .path = "/home/me/models/vae.safetensors", .size = 300, .mtime_ns = 1 },
    });
    var remote_entry: config.HostEntry = .{ .spawn = false };
    remote_entry.name.set("far");
    remote_entry.socket.set("lydia:7777");
    remote_entry.cert.set("AA");
    remote_entry.token.set(&[_]u8{'0'} ** 64);
    const b = try h.addSlot("far", remote_entry);
    // It has the VAE (same stem and size) but not the checkpoint.
    try setCatalog(b, &.{.{ .path = "id:x", .name = "vae", .size = 300, .mtime_ns = 9 }});

    var cfg: config.Config = .{};
    cfg.diffusion_model.set("/home/me/models/dream.safetensors");
    cfg.vae.set("/home/me/models/vae.safetensors");
    cfg.text_encoder.set("/home/me/models/gone.safetensors");

    var translated = cfg;
    h.refsToIds(&translated);
    var buf: [catalog.id_text_len]u8 = undefined;
    try testing.expectEqualStrings(catalog.idText(catalog.modelId("dream", 2000, 0), &buf), translated.diffusion_model.slice());
    try testing.expectEqualStrings(catalog.idText(catalog.modelId("vae", 300, 0), &buf), translated.vae.slice());
    // A path this machine's catalog does not know is left alone.
    try testing.expectEqualStrings("/home/me/models/gone.safetensors", translated.text_encoder.slice());
    // An id already there stays put.
    var twice = translated;
    h.refsToIds(&twice);
    try testing.expectEqualStrings(translated.diffusion_model.slice(), twice.diffusion_model.slice());

    // Nothing to offer while the host is down, and nothing for the local one.
    try testing.expect(h.missingModel(b, &cfg) == null);
    var r = fakeRemote();
    b.remote = &r;
    defer b.remote = null;
    try testing.expectEqualStrings("/home/me/models/dream.safetensors", h.missingModel(b, &cfg).?);
    try testing.expect(h.missingModel(a, &cfg) == null);
    try setCatalog(b, &.{
        .{ .path = "id:x", .name = "vae", .size = 300, .mtime_ns = 9 },
        .{ .path = "id:y", .name = "dream", .size = 2000, .mtime_ns = 9 },
    });
    try testing.expect(h.missingModel(b, &cfg) == null);

    // Holding the checkpoint but not the text encoder loads nothing: neither
    // a candidate nor silent about it, and offered the file it lacks.
    try setCatalog(a, &.{
        .{ .path = "/home/me/models/dream.safetensors", .size = 2000, .mtime_ns = 1 },
        .{ .path = "/home/me/models/vae.safetensors", .size = 300, .mtime_ns = 1 },
        .{ .path = "/home/me/models/te.safetensors", .size = 700, .mtime_ns = 1 },
    });
    cfg.text_encoder.set("/home/me/models/te.safetensors");
    ready(b);
    try testing.expectEqualStrings("/home/me/models/te.safetensors", h.missingModel(b, &cfg).?);
    try testing.expectEqualStrings("does not have every file this model needs", h.whyNotRender(b, &cfg).?.long);
    try testing.expectEqualStrings("missing files", h.whyNotRender(b, &cfg).?.short);
    var ids: [4]catalog.ModelId = undefined;
    try testing.expect(!Hosts.holdsAll(b, h.imageIds(&cfg, &ids)));
    try setCatalog(b, &.{
        .{ .path = "id:x", .name = "vae", .size = 300, .mtime_ns = 9 },
        .{ .path = "id:y", .name = "dream", .size = 2000, .mtime_ns = 9 },
        .{ .path = "id:z", .name = "te", .size = 700, .mtime_ns = 9 },
    });
    try testing.expect(h.missingModel(b, &cfg) == null);
    try testing.expect(h.whyNotRender(b, &cfg) == null);
    try testing.expect(Hosts.holdsAll(b, h.imageIds(&cfg, &ids)));
}

test "the queue is pulled from as hosts free up, and an unplaceable job does not block it" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();

    for ([_][]const u8{ "one", "two", "three" }) |p| {
        _ = h.remember(.{ .prompt = p, .width = 512, .height = 512, .steps = 8 });
    }
    var buf: [16]*const Hosts.Asked = undefined;
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 3), h.waiting(&buf).len);
    try testing.expectEqualStrings("one", h.waiting(&buf)[0].req.prompt);

    // A waiting job carries a client id, so the rail can name it; cancelling
    // one takes it out of the queue and leaves the rest in order.
    try testing.expect(mirror.isQueueRef(h.waiting(&buf)[1].ref));
    h.post(&cfg, .{ .img_cancel = .{ .image = h.waiting(&buf)[1].ref } });
    try testing.expectEqual(@as(usize, 2), h.waiting(&buf).len);
    try testing.expectEqualStrings("one", h.waiting(&buf)[0].req.prompt);
    try testing.expectEqualStrings("three", h.waiting(&buf)[1].req.prompt);

    h.post(&cfg, .{ .img_move = .{ .image = h.waiting(&buf)[1].ref, .before = h.waiting(&buf)[0].ref } });
    try testing.expectEqualStrings("three", h.waiting(&buf)[0].req.prompt);

    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;
    // Connected but silent so far: a default must not be read as "no image engine".
    try testing.expect(!h.hasRoom(a));
    try testing.expectEqualStrings("starting up", h.whyNotRender(a, &cfg).?.short);
    ready(a);
    try testing.expect(h.hasRoom(a));
    a.mirror.state.diff_busy = true;
    a.mirror.state.pending_images = 1;
    try testing.expect(!h.hasRoom(a));
    a.mirror.state.diff_busy = false;
    a.mirror.state.pending_images = 0;
    a.mirror.state.diff_paused = true;
    try testing.expect(!h.hasRoom(a));

    // The only host full: nothing is handed out and the queue holds.
    a.mirror.state.diff_paused = false;
    a.mirror.state.diff_busy = true;
    a.mirror.state.pending_images = 1;
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 2), h.waiting(&buf).len);
    try testing.expectEqualStrings("three", h.waiting(&buf)[0].req.prompt);

    // It frees up: it gets ONE job, and the other waits until the host lists
    // that one, however many passes run in between.
    a.mirror.state.diff_busy = false;
    a.mirror.state.pending_images = 0;
    h.dispatch(&cfg);
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    try testing.expectEqualStrings("one", h.waiting(&buf)[0].req.prompt);
    try testing.expectEqual(@as(usize, 1), ra.outbox.items.items.len);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = try sentRequest(&ra, arena.allocator(), 0);
    try testing.expectEqualStrings("three", sent.img_enqueue.prompt);
    // The host lists it as done and idle: the next one goes.
    try feedEvent(a, .{ .img = .{ .id = 9, .client_ref = sent.img_enqueue.client_ref, .status = .done } });
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 0), h.waiting(&buf).len);
    try testing.expectEqual(@as(usize, 2), ra.outbox.items.items.len);

    // Cancel-all empties this client's queue as well as the hosts'.
    _ = h.remember(.{ .prompt = "four", .width = 512, .height = 512, .steps = 8 });
    a.mirror.state.diff_busy = true;
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    h.post(&cfg, .img_cancel_all);
    try testing.expectEqual(@as(usize, 0), h.waiting(&buf).len);
}

test "a render a host never lists again does not wedge that host's queue" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;
    ready(a);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var buf: [4]*const Hosts.Asked = undefined;

    // Taken, failed there, and the row cleared away. The record stays (it is
    // what stops a re-parsed tool call being placed twice) but claims nothing,
    // or this host takes no further render for the life of the client.
    _ = h.remember(.{ .prompt = "one", .width = 512, .height = 512, .steps = 8 });
    h.dispatch(&cfg);
    const one = try sentRequest(&ra, arena.allocator(), 0);
    try feedEvent(a, .{ .img = .{ .id = 9, .client_ref = one.img_enqueue.client_ref, .status = .failed, .failure = "OutOfMemory" } });
    try testing.expect(h.forget(9));
    try testing.expect(h.hasRoom(a));

    // Handed over and NEVER listed, on a link that then drops: the enqueue may
    // not have arrived, so it goes back in the queue rather than being lost AND
    // counted against that host forever.
    _ = h.remember(.{ .prompt = "two", .width = 512, .height = 512, .steps = 8 });
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 0), h.waiting(&buf).len);
    try testing.expect(!h.hasRoom(a)); // outstanding there while the link is up
    a.mirror.linkDown();
    h.requeueUnlisted(a);
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    try testing.expectEqualStrings("two", h.waiting(&buf)[0].req.prompt);
    ready(a);
    try testing.expect(h.hasRoom(a));
}

fn feedEvent(s: *Slot, ev: wire.Event) !void {
    const bytes = try wire.encodeAlloc(testing.allocator, ev);
    defer testing.allocator.free(bytes);
    s.mirror.apply(.{ .text = bytes });
}

test "a job is held for a busy host that will finish sooner, and the next spreads to the other" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    ready(a);
    ready(b);
    const s_ns = std.time.ns_per_s;
    // A: 1 s/step, 2 steps left of 20. B: 10 s/step, idle.
    try testing.expect(a.table.note(.{ .family = "sd15", .mpx = 0.262144, .steps = 20, .first_step_ns = s_ns, .last_step_ns = 20 * s_ns }));
    try testing.expect(b.table.note(.{ .family = "sd15", .mpx = 0.262144, .steps = 20, .first_step_ns = s_ns, .last_step_ns = 191 * s_ns }));
    a.mirror.state.diff_busy = true;
    a.mirror.state.diff_family = try testing.allocator.dupe(u8, "sd15");
    b.mirror.state.diff_family = try testing.allocator.dupe(u8, "sd15");
    try feedEvent(a, .{ .img = .{ .id = 5, .status = .generating, .step = 18, .total = 20, .width = 512, .height = 512, .family = "sd15" } });
    try setCatalog(a, &.{.{ .path = "/m/dream.safetensors", .size = 1, .mtime_ns = 1, .ckpt = .{ .family = .sd15, .contents = .{} } }});
    cfg.diffusion_model.set("/m/dream.safetensors");

    _ = h.remember(.{ .prompt = "one", .width = 512, .height = 512, .steps = 20 });
    _ = h.remember(.{ .prompt = "two", .width = 512, .height = 512, .steps = 20 });
    var buf: [16]*const Hosts.Asked = undefined;
    h.dispatch(&cfg);
    // "one" waits for A (2 s left plus 20 s beats B's 200 s); "two" would
    // wait behind it (42 s) and still beats B, so both hold.
    try testing.expectEqual(@as(usize, 2), h.waiting(&buf).len);
    try testing.expectEqual(@as(usize, 0), ra.outbox.items.items.len);
    try testing.expectEqual(@as(usize, 0), rb.outbox.items.items.len);
    // B measured much closer: the second job spreads to it.
    b.table = .{};
    try testing.expect(b.table.note(.{ .family = "sd15", .mpx = 0.262144, .steps = 20, .first_step_ns = s_ns, .last_step_ns = 39 * s_ns }));
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    try testing.expectEqualStrings("one", h.waiting(&buf)[0].req.prompt);
    try testing.expectEqual(@as(usize, 1), rb.outbox.items.items.len);
    // A frees: "one" goes there.
    a.mirror.state.diff_busy = false;
    try feedEvent(a, .{ .img = .{ .id = 5, .status = .done } });
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 0), h.waiting(&buf).len);
    try testing.expectEqual(@as(usize, 1), ra.outbox.items.items.len);
}

test "a render this client asked for is kept whole, and a failure names a host that has not had it" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});

    // What the client asked for, LoRAs and all.
    const loras = [_]wire.LoraSpec{.{ .path = "/m/ink.safetensors", .strength = 0.7 }};
    const ref = h.remember(.{
        .prompt = "a lighthouse",
        .negative = "blurry",
        .width = 512,
        .height = 512,
        .steps = 8,
        .seed = 99,
        .loras = &loras,
    });
    try testing.expect(ref != 0);
    const asked = h.askedOf(ref).?;
    try testing.expectEqualStrings("a lighthouse", asked.req.prompt);
    try testing.expectEqualStrings("blurry", asked.req.negative);
    try testing.expectEqualStrings("/m/ink.safetensors", asked.req.loras[0].path);
    try testing.expectEqual(@as(f32, 0.7), asked.req.loras[0].strength);
    try testing.expectEqual(ref, asked.req.client_ref);
    try testing.expect(h.askedOf(ref + 1) == null);

    var ra = fakeRemote();
    var rb = fakeRemote();
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    ready(a);
    ready(b);

    try testing.expectEqual(b, h.placeExcluding(&cfg, asked.req, &.{a.id}, &.{}).?);
    try testing.expectEqual(a, h.placeExcluding(&cfg, asked.req, &.{b.id}, &.{}).?);
    try testing.expect(h.placeExcluding(&cfg, asked.req, &.{ a.id, b.id }, &.{}) == null);
    asked.mark(a.id);
    asked.mark(a.id);
    try testing.expectEqual(@as(usize, 1), asked.n_tried);
    try testing.expect(asked.was(a.id) and !asked.was(b.id));

    // A failure with nowhere to go is still reported, in words, exactly once.
    b.remote = null;
    try a.mirror.images.append(testing.allocator, .{ .info = .{
        .id = 7,
        .client_ref = ref,
        .status = .failed,
        .failure = try testing.allocator.dupe(u8, "ComponentNotInCheckpoint"),
    } });
    const f = h.takeFailure(&cfg).?;
    try testing.expectEqualStrings("local", f.host);
    try testing.expectEqualStrings("the checkpoint is missing a component (VAE or text encoder)", f.why);
    try testing.expect(f.moved_to == null);
    try testing.expect(h.takeFailure(&cfg) == null);

    // An image this client never asked for is reported and never replayed.
    try a.mirror.images.append(testing.allocator, .{ .info = .{
        .id = 8,
        .status = .failed,
        .failure = try testing.allocator.dupe(u8, "OutOfMemory"),
    } });
    const g = h.takeFailure(&cfg).?;
    try testing.expectEqualStrings("out of memory", g.why);
    try testing.expect(g.moved_to == null);
}

test "the same daemon coming back is told to stop what this client already put elsewhere" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    ready(a);
    ready(b);
    var buf: [16]*const Hosts.Asked = undefined;
    _ = h.remember(.{ .prompt = "moved", .width = 512, .height = 512, .steps = 8 });
    _ = h.remember(.{ .prompt = "stays", .width = 512, .height = 512, .steps = 8 });
    // A is paused, so both go to B: the first is taken, the second waits.
    a.mirror.state.diff_paused = true;
    h.dispatch(&cfg);
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const first = try sentRequest(&rb, arena.allocator(), 0);
    try feedEvent(b, .{ .img = .{ .id = 40, .client_ref = first.img_enqueue.client_ref, .status = .generating } });
    h.dispatch(&cfg);
    const second = try sentRequest(&rb, arena.allocator(), 1);
    try feedEvent(b, .{ .img = .{ .id = 41, .client_ref = second.img_enqueue.client_ref, .status = .pending } });
    try testing.expectEqual(@as(usize, 0), h.waiting(&buf).len);

    // B's link drops. What ran there is failed as lost and put to A; the one
    // behind it goes back to the queue, since A now holds one unlisted.
    a.mirror.state.diff_paused = false;
    b.mirror.linkDown();
    const f = h.takeFailure(&cfg).?;
    try testing.expectEqualStrings("local", f.moved_to.?);
    try testing.expect(h.takeFailure(&cfg).?.requeued);
    try testing.expectEqual(@as(usize, 1), ra.outbox.items.items.len);
    // Same generation answers: the renders now elsewhere are cancelled there,
    // and nothing else is.
    const before = rb.outbox.items.items.len;
    h.cancelStrays(b);
    try testing.expectEqual(before + 2, rb.outbox.items.items.len);
    const c0 = try sentRequest(&rb, arena.allocator(), before);
    try testing.expectEqual(@as(wire.ImageId, 40), c0.img_cancel.image);
    // Once each: the pass runs every frame, and a host does not need telling
    // twice while it gets around to it.
    h.cancelStrays(b);
    try testing.expectEqual(before + 2, rb.outbox.items.items.len);
}

test "pictures are ordered by when they were made, not by which host made them" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const a = h.local();
    const b = try h.addSlot("b", .{});
    // Two hosts finishing out of turn: b, a, b.
    try feedEvent(a, .{ .img = .{ .id = 1, .status = .done, .done_ns = 200 } });
    try feedEvent(b, .{ .img = .{ .id = 2, .status = .done, .done_ns = 100 } });
    try feedEvent(b, .{ .img = .{ .id = 3, .status = .done, .done_ns = 300 } });
    var buf: [8]Hosts.Shot = undefined;
    const fin = h.finished(&buf);
    try testing.expectEqual(@as(usize, 3), fin.len);
    try testing.expectEqual(@as(wire.ImageId, 3), fin[0].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 1), fin[1].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 2), fin[2].im.info.id);
    // Each one names the host that made it, and only while there is a choice.
    try testing.expectEqualStrings("b", fin[0].host);
    try testing.expectEqualStrings("local", fin[1].host);
    try testing.expectEqualStrings("b", h.hostOf(3));

    // A buffer smaller than the list keeps the newest, still in order.
    var small: [2]Hosts.Shot = undefined;
    const top = h.finished(&small);
    try testing.expectEqual(@as(usize, 2), top.len);
    try testing.expectEqual(@as(wire.ImageId, 3), top[0].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 1), top[1].im.info.id);

    // What is in flight reads the other way: a queue runs oldest first.
    try feedEvent(a, .{ .img = .{ .id = 4, .status = .generating, .start_ns = 500 } });
    try feedEvent(b, .{ .img = .{ .id = 5, .status = .pending, .start_ns = 400 } });
    const busy = h.running(&buf);
    try testing.expectEqual(@as(usize, 2), busy.len);
    try testing.expectEqual(@as(wire.ImageId, 5), busy[0].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 4), busy[1].im.info.id);
}

test "a full queue keeps what is running over what failed, however many failures pile up" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const a = h.local();
    // Failures are never cleared on their own, and they are the OLDEST rows.
    for (0..8) |i| try feedEvent(a, .{ .img = .{
        .id = @intCast(100 + i),
        .status = .failed,
        .start_ns = @intCast(10 + i),
        .failure = "OutOfMemory",
    } });
    try feedEvent(a, .{ .img = .{ .id = 7, .status = .generating, .start_ns = 900 } });
    try feedEvent(a, .{ .img = .{ .id = 8, .status = .pending, .start_ns = 901 } });
    var buf: [4]Hosts.Shot = undefined;
    const rows = h.running(&buf);
    try testing.expectEqual(@as(usize, 4), rows.len);
    // Both live rows are there, and the newest failures fill what is left.
    var live: usize = 0;
    for (rows) |r| if (r.im.status() != .failed) {
        live += 1;
    };
    try testing.expectEqual(@as(usize, 2), live);
    try testing.expectEqual(@as(wire.ImageId, 106), rows[0].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 107), rows[1].im.info.id);
    try testing.expectEqual(@as(wire.ImageId, 8), rows[3].im.info.id);
}

test "a restarted daemon holds none of what it was given, and neither does one that is unlisted" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    var entry: config.HostEntry = .{ .spawn = false };
    entry.name.set("b");
    entry.socket.set("/nonexistent/b.sock");
    cfg.hosts.items[0] = entry;
    cfg.hosts.count = 1;
    const b = try h.addSlot("b", entry);
    var rb = fakeRemote();
    defer rb.outbox.deinit(testing.allocator);
    b.remote = &rb;
    // Guarded: this test unlists b, and a slot that is gone must not be
    // reached for. Without it a failed assertion tears down through a fake
    // remote and the real failure is buried under the crash.
    defer if (h.slotByName("b")) |s| {
        s.remote = null;
    };
    ready(b);
    const done_ref = h.remember(.{ .prompt = "done", .width = 512, .height = 512, .steps = 8 });
    const lost_ref = h.remember(.{ .prompt = "lost", .width = 512, .height = 512, .steps = 8 });
    for (h.asked.items) |*a| {
        a.waiting = false;
        a.at = b.id;
    }
    // One finished and fetched, one still rendering.
    try feedEvent(b, .{ .img = .{ .id = 40, .client_ref = done_ref, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    var px = [_]u8{ 1, 2, 3, 255 };
    b.mirror.apply(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 40, .rev = 1, .w = 1, .h = 1, .len = 4 }, .payload = &px } });
    try feedEvent(b, .{ .img = .{ .id = 41, .client_ref = lost_ref, .status = .generating } });
    try testing.expectEqual(@as(u32, 0), h.unseen(b));

    // A different daemon answers: its images become this client's, so nothing
    // the host lists can account for these records any more. The picture that
    // is already here is over; the render that was in flight has to be made
    // again somewhere.
    h.hostRestarted(b);
    try testing.expectEqual(@as(u32, 0), h.unseen(b));
    try testing.expect(h.askedOf(done_ref).?.gone);
    try testing.expect(h.askedOf(lost_ref).?.waiting);

    // ...and a host taken out of the settings leaves nothing stranded either.
    h.askedOf(lost_ref).?.waiting = false;
    h.askedOf(lost_ref).?.at = b.id;
    b.remote = null;
    cfg.hosts.count = 0;
    h.sync(&cfg);
    try testing.expect(h.slotByName("b") == null);
    try testing.expect(h.askedOf(lost_ref).?.waiting);
    try settle(&h, &cfg);
}

test "a job reference and an image this client minted are never the same number" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const a = h.local();
    var refs: [4]u64 = undefined;
    for (&refs) |*r| r.* = h.remember(.{ .width = 64, .height = 64, .steps = 1 });
    for (refs) |r| {
        try testing.expect(mirror.isQueueRef(r));
        for (0..4) |_| try testing.expect(a.mirror.addLocal(.{}, null, null) != r);
    }
}

test "a waiting render says whether it is held for a host or can go nowhere" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    const asked: Hosts.Asked = .{ .ref = 1, .req = .{ .width = 512, .height = 512, .steps = 8 } };
    // Nothing up yet: nowhere to put it.
    try testing.expectEqual(Hosts.Wait.nowhere, h.waitReason(&cfg, &asked));
    ready(a);
    ready(b);
    try testing.expectEqual(Hosts.Wait.ready, h.waitReason(&cfg, &asked));
    // The best host is busy: it is held for that host, by name.
    b.mirror.state.diff_busy = true;
    a.mirror.state.diff_busy = true;
    const w = h.waitReason(&cfg, &asked);
    try testing.expect(w == .behind);
    try testing.expect(w.behind.len > 0);
}

test "a picture the client built itself lives with the local host and is found by its file" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const b = try h.addSlot("b", .{});
    const path = try testing.allocator.dupe(u8, "/out/a.png");
    const id = h.addLocal(.{ .status = .done, .width = 1, .height = 1 }, null, path);
    try testing.expect(id != 0);
    // It went to the local slot, whichever host carries the chat.
    try testing.expect(h.local().mirror.byId(id) != null);
    try testing.expect(b.mirror.byId(id) == null);
    // And it is found by its file from anywhere, so reopening a conversation
    // twice does not load a second copy.
    try testing.expectEqual(id, h.bySavedPath("/out/a.png").?.info.id);
    try testing.expect(h.bySavedPath("/out/other.png") == null);
    // A client's own picture is nobody's render.
    try testing.expectEqualStrings("", h.hostOf(id));
}

test "a reopened picture whose file is gone is missing, not a failed render" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;
    ready(a);

    const path = try testing.allocator.dupe(u8, "/out/gone.png");
    const id = h.addLocal(.{ .status = .done, .width = 8, .height = 8 }, null, path);
    h.markLost(id, "SavedImageMissing");
    try testing.expect(h.imageById(id).?.missing);

    // It is not work: nothing for the queue rail to draw...
    var shots: [4]Hosts.Shot = undefined;
    try testing.expectEqual(@as(usize, 0), h.running(&shots).len);
    // ...nothing to tell the user a host failed...
    try testing.expect(h.takeFailure(&cfg) == null);
    // ...and nothing to ask for again. The request lived in the file, so a
    // retry here would render an EMPTY prompt at whatever is selected now.
    h.retry(&cfg, id);
    try testing.expectEqual(@as(usize, 0), h.asked.items.len);
    var waiting: [4]*const Hosts.Asked = undefined;
    try testing.expectEqual(@as(usize, 0), h.waiting(&waiting).len);

    // A real render that failed still does all three.
    try a.mirror.images.append(testing.allocator, .{ .info = .{
        .id = 3,
        .status = .failed,
        .failure = try testing.allocator.dupe(u8, "DeviceOutOfMemory"),
    } });
    try testing.expectEqual(@as(usize, 1), h.running(&shots).len);
    try testing.expectEqualStrings("out of VRAM", h.takeFailure(&cfg).?.why);
}

test "a render is cancelled where it is, and a row that only records a failure is cleared away" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;
    ready(a);

    // One waiting here, one running there.
    const ref = h.remember(.{ .prompt = "one", .width = 512, .height = 512, .steps = 8 });
    _ = h.remember(.{ .prompt = "two", .width = 512, .height = 512, .steps = 8 });
    try feedEvent(a, .{ .img = .{ .id = 5, .status = .generating } });
    var buf: [8]*const Hosts.Asked = undefined;
    try testing.expectEqual(@as(usize, 2), h.waiting(&buf).len);

    // A waiting one is this client's to drop, and nothing goes to a host.
    const before = ra.outbox.items.items.len;
    h.post(&cfg, .{ .img_cancel = .{ .image = ref } });
    try testing.expectEqual(@as(usize, 1), h.waiting(&buf).len);
    try testing.expectEqual(before, ra.outbox.items.items.len);
    // A live render is not forgotten; the host is told to stop it.
    try testing.expect(!h.forget(5));
    h.post(&cfg, .{ .img_cancel = .{ .image = 5 } });
    try testing.expectEqual(before + 1, ra.outbox.items.items.len);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = try sentRequest(&ra, arena.allocator(), before);
    try testing.expectEqual(@as(wire.ImageId, 5), sent.img_cancel.image);

    // Once it has failed there is nothing to stop, so the row goes.
    try feedEvent(a, .{ .img = .{ .id = 5, .status = .failed, .failure = "OutOfMemory" } });
    try testing.expect(h.forget(5));
    try testing.expect(h.imageById(5) == null);
    try testing.expect(!h.forget(5));
}

test "the preview size follows what a view actually draws" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    const b = try h.addSlot("b", .{});
    h.setPreviewMaxEdge(192);
    try testing.expectEqual(@as(u32, 192), h.local().mirror.preview_max_edge);
    try testing.expectEqual(@as(u32, 192), b.mirror.preview_max_edge);
    // A host added later starts where the others are.
    const c = try h.addSlot("c", .{});
    try testing.expectEqual(@as(u32, 192), c.mirror.preview_max_edge);
}

test "trying a failed render again asks for it whole, as a job any host can take" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;

    ready(a);
    // Asked for with a LoRA stack, taken by the host, and failed there.
    const loras = [_]wire.LoraSpec{.{ .path = "/m/ink.safetensors", .strength = 0.7 }};
    const ref = h.remember(.{ .prompt = "a lighthouse", .negative = "blurry", .width = 512, .height = 512, .steps = 8, .seed = 99, .loras = &loras });
    h.dispatch(&cfg);
    try feedEvent(a, .{ .img = .{ .id = 9, .client_ref = ref, .status = .failed, .failure = "OutOfMemory" } });

    // The whole request goes out again, LoRAs and seed included, as a new job.
    const before = ra.outbox.items.items.len;
    h.retry(&cfg, 9);
    try testing.expectEqual(before + 1, ra.outbox.items.items.len);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = (try sentRequest(&ra, arena.allocator(), before)).img_enqueue;
    try testing.expectEqualStrings("a lighthouse", sent.prompt);
    try testing.expectEqualStrings("blurry", sent.negative);
    try testing.expectEqualStrings("/m/ink.safetensors", sent.loras[0].path);
    try testing.expectEqual(@as(u64, 99), sent.seed);
    try testing.expect(sent.client_ref != ref);

    // One the model asked a host for has no request of ours behind it, so it is
    // rebuilt from what came back.
    try feedEvent(a, .{ .img = .{ .id = 10, .status = .failed, .prompt = "a mug", .req_width = 256, .req_height = 256, .req_steps = 4, .req_seed = 7 } });
    const asked_before = h.asked.items.len;
    h.retry(&cfg, 10);
    try testing.expectEqual(asked_before + 1, h.asked.items.len);
    const rebuilt = h.asked.items[h.asked.items.len - 1].req;
    try testing.expectEqualStrings("a mug", rebuilt.prompt);
    try testing.expectEqual(@as(u32, 256), rebuilt.width);
    try testing.expectEqual(@as(u64, 7), rebuilt.seed);
    // An id nobody knows asks for nothing.
    h.retry(&cfg, 999);
    try testing.expectEqual(asked_before + 1, h.asked.items.len);
}

test "a render the model asks for is placed like any other and reported back to its conversation" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local(); // carries the chat
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    ready(a);
    ready(b);
    // The chat host is busy, so the model's render goes to the other machine.
    // That is the whole point: it used to be stuck on whoever held the chat.
    a.mirror.state.diff_busy = true;

    try feedEvent(a, .{ .img_requested = .{ .calls = &.{
        .{ .msg = 1, .variant = 0, .index = 0, .req = .{ .prompt = "a lighthouse", .width = 512, .height = 512, .steps = 8 } },
    } } });
    h.pump(&cfg);
    try testing.expectEqual(@as(usize, 1), rb.outbox.items.items.len);

    // The host says it again after a reconnect, not knowing it was heard. The
    // same call is not placed twice.
    try feedEvent(a, .{ .img_requested = .{ .calls = &.{
        .{ .msg = 1, .variant = 0, .index = 0, .req = .{ .prompt = "a lighthouse", .width = 512, .height = 512, .steps = 8 } },
    } } });
    h.pump(&cfg);
    try testing.expectEqual(@as(usize, 1), rb.outbox.items.items.len);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = (try sentRequest(&rb, arena.allocator(), 0)).img_enqueue;
    try testing.expectEqualStrings("a lighthouse", sent.prompt);
    try testing.expect(sent.client_ref != 0);

    // The rendering host mints the id; the conversation is told which image
    // its own reply produced, at the host that carries the conversation.
    try feedEvent(b, .{ .img = .{ .id = 77, .client_ref = sent.client_ref, .status = .generating } });
    h.pump(&cfg);
    const claim = (try sentRequest(&ra, arena.allocator(), 0)).chat_image;
    try testing.expectEqual(@as(u32, 1), claim.msg);
    try testing.expectEqual(@as(u32, 0), claim.variant);
    try testing.expectEqual(@as(wire.ImageId, 77), claim.image);

    // Nothing is said about it while it is still running.
    try testing.expectEqual(@as(usize, 1), ra.outbox.items.items.len);

    // It finishes there, and the model is told here.
    try feedEvent(b, .{ .img = .{ .id = 77, .client_ref = sent.client_ref, .status = .done, .width = 512, .height = 512, .req_seed = 5 } });
    h.pump(&cfg);
    const note = (try sentRequest(&ra, arena.allocator(), 1)).chat_note;
    try testing.expect(std.mem.indexOf(u8, note.text, "finished") != null);
    try testing.expect(std.mem.indexOf(u8, note.text, "512") != null);
    // Once, however many passes run.
    h.pump(&cfg);
    try testing.expectEqual(@as(usize, 2), ra.outbox.items.items.len);
}

test "a reply's renders stay counted as unplaced until the transcript has an image for each" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    var ra = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    a.remote = &ra;
    defer a.remote = null;
    ready(a);

    // Three in one breath, one host: it takes one and the rest wait here.
    try feedEvent(a, .{ .img_requested = .{ .calls = &.{
        .{ .msg = 1, .variant = 0, .index = 0, .req = .{ .prompt = "one", .width = 512, .height = 512, .steps = 8 } },
        .{ .msg = 1, .variant = 0, .index = 1, .req = .{ .prompt = "two", .width = 512, .height = 512, .steps = 8 } },
        .{ .msg = 1, .variant = 0, .index = 2, .req = .{ .prompt = "three", .width = 512, .height = 512, .steps = 8 } },
    } } });
    h.pump(&cfg);
    // The card holds three slots: not one per image, one per call.
    try testing.expectEqual(@as(usize, 3), h.unplacedFor(1, 0));
    // Another reply's card is not told about them.
    try testing.expectEqual(@as(usize, 0), h.unplacedFor(2, 0));

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = (try sentRequest(&ra, arena.allocator(), 0)).img_enqueue;
    try feedEvent(a, .{ .img = .{ .id = 77, .client_ref = sent.client_ref, .status = .generating } });
    h.pump(&cfg);
    // One has an image now, so the card points at it and reserves two.
    try testing.expectEqual(@as(usize, 2), h.unplacedFor(1, 0));
}

test "the model is not told a render failed while this client is still moving it" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local();
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    ready(a);
    ready(b);
    a.mirror.state.diff_busy = true;
    try feedEvent(a, .{ .img_requested = .{ .calls = &.{
        .{ .msg = 1, .variant = 0, .req = .{ .prompt = "a mug", .width = 512, .height = 512, .steps = 8 } },
    } } });
    h.pump(&cfg);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const first = (try sentRequest(&rb, arena.allocator(), 0)).img_enqueue;
    try feedEvent(b, .{ .img = .{ .id = 5, .client_ref = first.client_ref, .status = .failed, .failure = "OutOfMemory" } });

    // Failed, but nobody has looked at it yet: the model is told nothing.
    h.pump(&cfg);
    const after_claim = ra.outbox.items.items.len;
    try testing.expect(after_claim <= 1);

    // The client moves it to the free host instead; still nothing is said.
    a.mirror.state.diff_busy = false;
    _ = h.takeFailure(&cfg);
    h.pump(&cfg);
    for (ra.outbox.items.items) |f| {
        const req = try wire.decode(wire.Request, arena.allocator(), f.text);
        try testing.expect(req != .chat_note);
    }
}

test "the model is offered the image tool when ANY host can render, not the one carrying the chat" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    const a = h.local(); // carries the chat, and renders nothing
    const b = try h.addSlot("b", .{});
    var ra = fakeRemote();
    var rb = fakeRemote();
    defer ra.outbox.deinit(testing.allocator);
    defer rb.outbox.deinit(testing.allocator);
    a.remote = &ra;
    b.remote = &rb;
    defer {
        a.remote = null;
        b.remote = null;
    }
    // Nobody has said anything yet: no tool.
    try testing.expect(!h.canRenderAnywhere(&cfg));
    a.mirror.state_seen = true;
    b.mirror.state_seen = true;
    try testing.expect(!h.canRenderAnywhere(&cfg));
    // The chat host still has no image engine, but the other one does.
    b.mirror.state.diff_present = true;
    try testing.expect(h.canRenderAnywhere(&cfg));
    // What the chat host is told says so, whatever its own checkpoint is.
    _ = h.postSettingsTo(a, &cfg);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sent = try sentRequest(&ra, arena.allocator(), ra.outbox.items.items.len - 1);
    const hs = try std.json.parseFromSliceLeaky(config.HostSettings, arena.allocator(), sent.settings.json, .{ .ignore_unknown_fields = true });
    try testing.expect(hs.image_tool);

    // The only host that could render goes away: the tool goes with it, and the
    // chat host is told again with nothing else having changed.
    h.pump(&cfg); // the answer as it stands is noted
    b.mirror.state.diff_present = false;
    const before = ra.outbox.items.items.len;
    h.pump(&cfg);
    try testing.expect(!h.canRenderAnywhere(&cfg));
    errdefer std.debug.print("pushes before={d} after={d}\n", .{ before, ra.outbox.items.items.len });
    try testing.expect(ra.outbox.items.items.len > before);
    const again = try sentRequest(&ra, arena.allocator(), before);
    const hs2 = try std.json.parseFromSliceLeaky(config.HostSettings, arena.allocator(), again.settings.json, .{ .ignore_unknown_fields = true });
    try testing.expect(!hs2.image_tool);
}

test "a host is tried where it is pasted, before anything is applied" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};

    // A host this client would start has nothing to reach yet.
    _ = cfg.addHost("mine", "/nonexistent/mine.sock", true);
    h.tryHost(cfg.hostEntry("mine").?);
    try testing.expect(h.trial == null);

    _ = cfg.addHost("theirs", "/nonexistent/theirs.sock", false);
    const e = cfg.hostEntry("theirs").?;
    h.tryHost(e);
    try testing.expect(h.trial != null);
    const deadline = Io.Clock.real.now(testing.io).nanoseconds + 10 * std.time.ns_per_s;
    while (h.trialOf(e).?.state == .running) {
        if (Io.Clock.real.now(testing.io).nanoseconds > deadline) return error.TrialNeverFinished;
        try Io.sleep(testing.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .real);
    }
    const got = h.trialOf(e).?;
    try testing.expectEqual(Hosts.Trial.State.refused, got.state);
    try testing.expectEqualStrings("nothing is listening there", got.text);
    try testing.expect(h.trialOf(cfg.hostEntry("mine").?) == null);
    var moved = e.*;
    moved.socket.set("/nonexistent/elsewhere.sock");
    try testing.expect(h.trialOf(&moved) == null);
}

test "sync follows the settings list, a host down at the start is retried, and the chat pin resolves by name" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    _ = cfg.addHost("beta", "/nonexistent/beta.sock", false);
    cfg.chat_host.set("beta");
    // Nothing answers at either socket. The list is still shaped, the pin
    // still resolves, and the connects are in flight.
    h.sync(&cfg);
    try testing.expectEqual(@as(usize, 2), h.slots.items.len);
    const beta = h.slots.items[1];
    try testing.expectEqualStrings("beta", beta.name);
    try testing.expectEqualStrings("connecting", h.statusOf("beta"));
    try settle(&h, &cfg);
    // The failure is not final: it is worded, and the next try is scheduled.
    try testing.expect(!beta.lost);
    try testing.expectEqual(@as(u32, 1), beta.tries);
    try testing.expect(beta.retry_at > 0);
    try testing.expectEqualStrings("nothing is listening there", h.statusOf("beta"));
    try testing.expectEqualStrings("nothing is listening there", Hosts.troubleOf(beta).?);
    try testing.expectEqualStrings("", h.statusOf("nobody"));
    // Asking by hand drops the wait: the next pump reaches for it again.
    beta.retry_at = std.math.maxInt(i96);
    h.reconnectNamed("beta");
    h.pump(&cfg);
    try testing.expect(beta.connecting != null);
    try settle(&h, &cfg);
    try testing.expectEqual(@as(u32, 1), beta.tries);
    // A removed entry closes its slot; the pin falls back to local.
    cfg.removeHost(0);
    cfg.chat_host.set("");
    h.sync(&cfg);
    try testing.expectEqual(@as(usize, 1), h.slots.items.len);
    try testing.expectEqual(@as(HostId, 0), h.chat);
    try settle(&h, &cfg);
}

test "reconnect backoff widens to a cap" {
    try testing.expectEqual(@as(i96, 1 * std.time.ns_per_s), Hosts.backoff(1));
    try testing.expectEqual(@as(i96, 2 * std.time.ns_per_s), Hosts.backoff(2));
    try testing.expectEqual(@as(i96, 16 * std.time.ns_per_s), Hosts.backoff(5));
    try testing.expectEqual(@as(i96, 30 * std.time.ns_per_s), Hosts.backoff(6));
    try testing.expectEqual(@as(i96, 30 * std.time.ns_per_s), Hosts.backoff(50));
}

test "a host that drops is failed once, retried on the backoff, and never given up on" {
    var env: std.process.Environ.Map = .init(testing.allocator);
    defer env.deinit();
    var h = try newHosts(&env);
    defer h.deinit();
    var cfg: config.Config = .{};
    var e: config.HostEntry = .{};
    e.name.set("gone");
    e.socket.set("/nonexistent-dir-for-a-test/tp.sock");
    e.spawn = false;
    const s = try h.addSlot("gone", e);
    var r = fakeRemote();
    r.failed = .init(true);
    s.remote = &r;
    defer s.remote = null;
    try feedEvent(s, .{ .img = .{ .id = 3, .status = .generating } });

    // The first sweep notices, fails what ran there, and reaches out.
    h.pump(&cfg);
    try testing.expect(s.down);
    try testing.expectEqual(wire.ImageStatus.failed, s.mirror.byId(3).?.status());
    try testing.expect(s.connecting != null);
    try settle(&h, &cfg);
    errdefer std.debug.print("after 1: lost={} tries={d} retry_at={d}\n", .{ s.lost, s.tries, s.retry_at });
    try testing.expect(!s.lost);
    try testing.expectEqual(@as(u32, 1), s.tries);
    try testing.expect(s.retry_at > 0);
    try testing.expectEqualStrings("nothing is listening there", Hosts.troubleOf(s).?);

    // Inside the backoff nothing is tried; past it, another try, still not final.
    h.pump(&cfg);
    try testing.expect(s.connecting == null);
    s.retry_at = 0;
    h.pump(&cfg);
    try testing.expect(s.connecting != null);
    try settle(&h, &cfg);
    try testing.expectEqual(@as(u32, 2), s.tries);
    try testing.expect(!s.lost);
    try testing.expectEqual(@as(u32, 0), s.reconnects);
}
