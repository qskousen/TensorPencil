//! tp-serve: the engine host behind a socket. Owns the GPU, the models and the
//! transcript's live copy; keeps nothing on disk but its catalog cache and,
//! for a remote host, its identity. tp-gui spawns one as a child
//! (`--autospawn`: it leaves when its stdin closes or its events client
//! disconnects), connects to one already running, or reaches one on another
//! machine under TLS.
//!
//! `tp-serve [--socket <path>] [--autospawn] [--config <file>] [--index <file>]`
//! serves this machine. Prints one `ready <endpoint>` line on stdout once it
//! listens, then logs on stderr. Without `--config` it starts with default
//! settings and takes the client's `settings` push, which is the normal path.
//! The catalog's cache goes to `--index`, else beside the settings file.
//!
//! `tp-serve --remote <bind>:<port> [--models <dir>]... [--backend <name>]
//! [--state <dir>] [--pair-host <name>] [--new-token] [--incoming <dir>]`
//!
//! A remote host READS as well as writes model files: `/v1/pull/<catalog id>`
//! hands back a file it holds, so a client can take a model it lacks. Only a
//! file in the scanned folders is nameable, and only by the id a scan gave it
//! (`Host.offerPath`, off a table published per scan so a connection thread
//! never touches the live catalog). Nothing a user typed is reachable that way,
//! which is why the receipt below is still about what tp-serve WRITES.
//! serves the network.
//! The state directory (default: `serve/` beside the settings file) holds the
//! certificate, its key and the hash of the token; all three are minted on
//! first start, and the pairing string a client pastes into its Hosts list
//! (`tp://<host>:<port>/<certificate>#<token>`) is printed ONCE, as a `pair`
//! line on stdout, since only the token's hash is kept. `--new-token` mints
//! another. Such a host scans `--models` folders (else the settings file's),
//! names models to clients by id and stem only, and never opens a path a
//! client sends. A client may put a model file it has and the host lacks into
//! `--incoming` (default: the first `--models` folder) over `/v1/blob/*`; the
//! host rescans when one lands.
//!
//! **Everything this process may open for writing**, which is the list a
//! syscall trace is diffed against. The receipt is `zig build serve-probe`
//! under `strace -ff -e trace=openat,creat,rename,unlinkat`, keeping the
//! traces that opened a model or the socket:
//!
//! - the catalog index it was given (`--index`, else beside `--config`), which
//!   holds the OPERATOR's model paths and never crosses the wire as paths;
//! - a remote host's state directory: `cert.pem`, `key.pem`, `token.hash`;
//! - a remote host's incoming model files (`--incoming`): the partial under
//!   `partial/<digest>`, its rename into place, and its unlink when the
//!   sender aborts the transfer (`blob.Store.abort`);
//! - the directory its unix socket sits in (`$XDG_RUNTIME_DIR/tensorpencil-<user>`,
//!   created 0700 by `link.defaultLocalPath`), and the socket itself,
//!   unlinked at shutdown;
//! - a `<model>.sha256` beside a model file that has none, written once so the
//!   AutoV2 hash in a saved PNG's metadata is not recomputed per render (a
//!   digest of the weights: no prompt, no reply, no pixel). `hash_models = false`
//!   turns it off;
//! - thread names (`/proc/self/task/*/comm`) and the GPU device nodes;
//! - `~/.nv/ComputeCache/*`, which the NVIDIA driver writes on its own account
//!   (compiled kernels, no user content) and which no unit test would find.
//!
//! Nothing on that list holds a prompt, a reply or a rendered pixel: the
//! engine writes no files at all, and saving an image is the client's job.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const config = @import("shared").config;
const wire = @import("serve").wire;
const link = @import("serve").link;
const server = @import("serve").server;
const queue = @import("serve").queue;
const pairing = @import("serve").pairing;
const blob = @import("serve").blob;
const host_mod = @import("engine").host;
const serve_tls = @import("serve_tls.zig");

const log = std.log.scoped(.tp_serve);

const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    host: *host_mod.Host,
    listener: *link.Listener,
    autospawn: bool,
    /// Set when serving the network: every connection is wrapped in TLS.
    remote: ?*Remote,
    /// Where a client may put model files; null takes none.
    blobs: ?*blob.Store = null,
    /// The files this host will hand BACK, described once and cached. Only set
    /// for a host that serves folders of its own.
    offers: ?*blob.Offers = null,
    /// Bumped by the host's `on_out`; the events pump parks on it.
    out_gen: std.atomic.Value(u32) = .init(0),
    events_attached: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    /// Connections being served; one past `server.max_connections` is closed unserved.
    conns: std.atomic.Value(u32) = .init(0),
    watchdog: server.Watchdog,

    fn backend(self: *Ctx) server.Backend {
        return .{
            .ctx = self,
            .hello = hello,
            .request = request,
            .upload = upload,
            .outGen = outGen,
            .waitOut = waitOut,
            .wakeOut = wakeOutCb,
            .take = take,
            .eventsAttach = eventsAttach,
            .eventsDetach = eventsDetach,
            .blobs = self.blobs,
            .fileAdded = fileAdded,
            .offers = self.offers,
            .offerPath = if (self.offers != null) offerPath else null,
        };
    }

    /// Where this host holds the file a pull names. Read off the connection
    /// thread, from the host's own locked table, never from the live catalog.
    fn offerPath(ctx: *anyopaque, id: []const u8, buf: []u8) ?[]const u8 {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        return self.host.offerPath(id, buf);
    }

    /// A model file landed: the engine thread rescans (a remote host scans its
    /// own folders, so the empty request is the whole message).
    fn fileAdded(ctx: *anyopaque, path: []const u8) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        log.info("received {s}", .{std.fs.path.basename(path)});
        self.host.postRequest(.{ .scan = .{} });
    }

    fn hello(ctx: *anyopaque, gpa: std.mem.Allocator) anyerror![]u8 {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        return wire.encodeAlloc(gpa, wire.Event{ .hello = .{ .gen = self.host.gen } });
    }
    fn request(ctx: *anyopaque, bytes: []u8) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        if (self.stop.load(.acquire)) return self.gpa.free(bytes);
        self.host.post(.{ .text = bytes });
    }
    fn upload(ctx: *anyopaque, hdr: wire.BinHeader, payload: []u8) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        if (self.stop.load(.acquire)) return self.gpa.free(payload);
        self.host.post(.{ .bin = .{ .hdr = hdr, .payload = payload } });
    }
    fn outGen(ctx: *anyopaque) u32 {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        return self.out_gen.load(.acquire);
    }
    fn waitOut(ctx: *anyopaque, seen: u32, timeout_ns: u64) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        self.io.futexWaitTimeout(u32, &self.out_gen.raw, seen, .{ .duration = .{ .raw = .{ .nanoseconds = timeout_ns }, .clock = .awake } }) catch {};
    }
    fn wakeOutCb(ctx: *anyopaque) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        self.wakeOut();
    }
    fn take(ctx: *anyopaque, out: *std.ArrayList(queue.Frame)) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        if (self.stop.load(.acquire)) return;
        self.host.take(out);
    }
    fn eventsAttach(ctx: *anyopaque) bool {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        return self.events_attached.cmpxchgStrong(false, true, .acq_rel, .acquire) == null;
    }
    fn eventsDetach(ctx: *anyopaque) void {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        // Nothing queued is owed to the next client: it greets with a snapshot.
        self.host.clearOutbox();
        self.events_attached.store(false, .release);
        // Closing tp-gui frees the GPU: the child goes with its one client.
        if (self.autospawn) self.shutdown("client disconnected");
    }

    fn wakeOut(self: *Ctx) void {
        _ = self.out_gen.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.out_gen.raw, std.math.maxInt(u32));
    }

    /// Tear the engines down in order and leave. Runs on whichever thread saw
    /// the reason; the main thread is parked in `accept` and exits with us.
    fn shutdown(self: *Ctx, why: []const u8) noreturn {
        if (self.stop.swap(true, .acq_rel)) {
            // Another thread is already on it.
            while (true) Io.sleep(self.io, .{ .nanoseconds = std.time.ns_per_s }, .real) catch {};
        }
        log.info("shutting down: {s}", .{why});
        self.wakeOut();
        // The socket goes FIRST: freeing the engines takes seconds, and a
        // client arriving meanwhile must spawn a fresh host at this path, not
        // reach a dying one.
        self.listener.deinit(self.gpa);
        self.host.deinit();
        std.process.exit(0);
    }
};

var g_ctx: ?*Ctx = null;

fn onOut() void {
    const c = g_ctx orelse return;
    c.wakeOut();
}

fn connMain(c: *Ctx, stream: Io.net.Stream) void {
    defer _ = c.conns.fetchSub(1, .acq_rel);
    const armed = c.watchdog.arm(stream, server.nowNs(c.io) + server.first_head_timeout_ns);
    const made = if (c.remote) |r| serve_tls.accept(c.gpa, c.io, stream, &r.ident) else link.Link.init(c.gpa, c.io, stream, c.listener.endpoint == .unix);
    const l = made catch |err| {
        if (armed) |a| a.release();
        stream.close(c.io);
        if (c.remote != null) log.warn("tls handshake failed: {t}", .{err});
        return;
    };
    defer l.close(c.gpa);
    // Runs before the close above: a sweep must never reach a closed stream.
    defer if (armed) |a| a.release();
    server.serve(c.gpa, l, c.backend(), &c.stop, .{ .auth = c.listener.authHash(), .watch = armed }) catch |err| log.warn("connection ended: {t}", .{err});
}

/// `--autospawn` liveness: the parent holds our stdin's write end, so its
/// death is our EOF.
fn watchStdin(c: *Ctx) void {
    var buf: [256]u8 = undefined;
    var r = Io.File.stdin().reader(c.io, &buf);
    while (true) _ = r.interface.takeByte() catch break;
    c.shutdown("stdin closed");
}

// ── Remote identity ─────────────────────────────────────────────────────────

const Remote = struct {
    ident: serve_tls.Identity,
    /// What a client's token must hash to.
    auth: link.SecretHash,
};

const RemoteArgs = struct {
    bind: link.HostPort,
    state_dir: ?[]const u8 = null,
    pair_host: ?[]const u8 = null,
    new_token: bool = false,
};

/// The identity and token from the state directory, minted where missing. The
/// pairing string goes to `out` when a token is minted, the only time it can.
fn remoteSetup(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: Io, environ: *const std.process.Environ.Map, cfg_path: ?[]const u8, ra: RemoteArgs, out: *Io.Writer) !*Remote {
    const dir = ra.state_dir orelse (config.Config.siblingPath(io, arena, environ, cfg_path, "serve") catch null) orelse return error.NoStateDir;
    const p = try pairing.Paths.init(arena, dir);
    if (try pairing.ensureIdentity(gpa, io, dir, p, Io.Clock.real.now(io).toSeconds())) log.info("minted a certificate at {s}", .{p.cert});

    const r = try gpa.create(Remote);
    errdefer gpa.destroy(r);
    r.ident = try serve_tls.loadIdentity(gpa, io, p.cert, p.key);
    errdefer r.ident.deinit(gpa);

    if (!ra.new_token) if (pairing.readTokenHash(io, p)) |have| {
        r.auth = have;
        log.info("token {s}: the pairing string was printed when it was minted; --new-token mints another", .{&link.fingerprint(have)});
        return r;
    };
    const token = try pairing.newToken(io, p);
    r.auth = link.hashSecret(token);
    log.info("token {s}: minted, and in the pairing string below", .{&link.fingerprint(r.auth)});
    const host = ra.pair_host orelse try hostName(arena);
    // One write: stderr logging from other threads must not land inside the
    // line a user is about to copy.
    const line = try std.fmt.allocPrint(arena, "pair {s}\n", .{try pairing.pairingAlloc(arena, io, p, host, ra.bind.port, token)});
    try out.writeAll(line);
    try out.flush();
    return r;
}

/// The machine's own name, what a pairing string names when nothing better is
/// given.
fn hostName(arena: std.mem.Allocator) ![]const u8 {
    if (builtin.os.tag == .windows) return "localhost";
    var buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const name = std.posix.gethostname(&buf) catch return "localhost";
    return arena.dupe(u8, name);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = std.heap.smp_allocator;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    var sock_arg: ?[]const u8 = null;
    var cfg_path: ?[]const u8 = null;
    var index_arg: ?[]const u8 = null;
    var autospawn = false;
    var remote_args: ?RemoteArgs = null;
    var models: std.ArrayList([]const u8) = .empty;
    var backend_arg: ?config.Backend = null;
    var state_dir: ?[]const u8 = null;
    var pair_host: ?[]const u8 = null;
    var incoming: ?[]const u8 = null;
    var new_token = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const more = i + 1 < args.len;
        if (std.mem.eql(u8, args[i], "--socket") and more) {
            i += 1;
            sock_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--config") and more) {
            i += 1;
            cfg_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--index") and more) {
            i += 1;
            index_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--autospawn")) {
            autospawn = true;
        } else if (std.mem.eql(u8, args[i], "--remote") and more) {
            i += 1;
            remote_args = .{ .bind = link.splitHostPort(args[i]) orelse return usage() };
        } else if (std.mem.eql(u8, args[i], "--models") and more) {
            i += 1;
            try models.append(arena, args[i]);
        } else if (std.mem.eql(u8, args[i], "--backend") and more) {
            i += 1;
            backend_arg = std.meta.stringToEnum(config.Backend, args[i]) orelse return usage();
        } else if (std.mem.eql(u8, args[i], "--state") and more) {
            i += 1;
            state_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--pair-host") and more) {
            i += 1;
            pair_host = args[i];
        } else if (std.mem.eql(u8, args[i], "--incoming") and more) {
            i += 1;
            incoming = args[i];
        } else if (std.mem.eql(u8, args[i], "--new-token")) {
            new_token = true;
        } else return usage();
    }

    var cfg: config.Config = if (cfg_path) |p| config.Config.load(io, gpa, init.environ_map, p) else .{};
    cfg.applyFraming();
    if (backend_arg) |b| {
        cfg.llm_backend = b;
        cfg.diff_backend = b;
    }
    // The catalog's cache sits beside the settings file, the client's when it
    // named one (a probe's scratch copy keeps its scratch index), unless the
    // client named a file: two hosts on one disk must not share one.
    const index_path: ?[]u8 = if (index_arg) |p| try gpa.dupe(u8, p) else config.Config.siblingPath(io, gpa, init.environ_map, cfg_path, "catalog.json") catch null;
    defer if (index_path) |p| gpa.free(p);

    var out_buf: [4096]u8 = undefined;
    // Streaming: a positional writer redirected into a shared log overwrites it.
    var stdout = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const out = &stdout.interface;

    // A remote host's folders: `--models`, else the settings file's.
    var folders: ?host_mod.Folders = null;
    var remote: ?*Remote = null;
    if (remote_args) |*ra| {
        ra.state_dir = state_dir;
        ra.pair_host = pair_host;
        ra.new_token = new_token;
        remote = try remoteSetup(gpa, arena, io, init.environ_map, cfg_path, ra.*, out);
        var dirs: std.ArrayList([]const u8) = .empty;
        var files: std.ArrayList([]const u8) = .empty;
        try dirs.appendSlice(arena, models.items);
        if (models.items.len == 0) for (cfg.model_dirs.slice()) |*d| if (d.path.opt()) |p| try dirs.append(arena, try arena.dupe(u8, p));
        for (cfg.model_files.slice()) |*f| if (f.path.opt()) |p| try files.append(arena, try arena.dupe(u8, p));
        // A file that lands in `--incoming` must be scanned, wherever that is.
        if (incoming) |inc| if (!folderListed(io, arena, dirs.items, inc)) try dirs.append(arena, inc);
        if (dirs.items.len == 0 and files.items.len == 0) log.warn("no model folders: give --models <dir>, or clients see an empty catalog", .{});
        folders = .{ .dirs = dirs.items, .files = files.items };
    }

    const host = try gpa.create(host_mod.Host);
    host.init(gpa, io, &cfg, .{ .index_path = index_path, .folders = folders }, onOut);

    // Files a client sends land in a folder this host scans, so a finished
    // transfer shows up in the catalog under the id the client named.
    var store: ?blob.Store = null;
    if (folders) |f| {
        if (incoming orelse (if (f.dirs.len > 0) f.dirs[0] else null)) |dir| {
            store = blob.Store.init(gpa, io, dir) catch |err| blk: {
                log.warn("no incoming folder ({t}); this host takes no model files", .{err});
                break :blk null;
            };
            if (store != null) log.info("incoming models: {s}", .{dir});
        }
    }
    defer if (store) |*s| s.deinit();

    // The other direction: a client may take a model this host holds. Only for
    // a host with folders of its own to describe.
    var offers: ?blob.Offers = if (folders != null) blob.Offers.init(gpa, io) else null;
    defer if (offers) |*o| o.deinit();

    var listener = if (remote_args) |ra|
        try link.listenTcp(io, ra.bind.host, ra.bind.port, remote.?.auth)
    else
        try link.listenLocal(gpa, io, sock_arg orelse try link.defaultLocalPath(gpa, io, init.environ_map));
    var ctx: Ctx = .{
        .gpa = gpa,
        .io = io,
        .host = host,
        .listener = &listener,
        .autospawn = autospawn,
        .remote = remote,
        .blobs = if (store) |*s| s else null,
        .offers = if (offers) |*o| o else null,
        .watchdog = .{ .io = io },
    };
    g_ctx = &ctx;
    try host.start();

    // The ready line carries the cookie for the parent that spawned us; the log gets no secret.
    try out.print("ready {f}\n", .{listener.endpoint});
    try out.flush();
    log.info("listening on {f}", .{listener.endpoint.public()});

    if (autospawn) {
        const t = try std.Thread.spawn(.{}, watchStdin, .{&ctx});
        t.detach();
    }
    {
        const t = try std.Thread.spawn(.{}, server.Watchdog.run, .{ &ctx.watchdog, &ctx.stop });
        t.detach();
    }

    while (!ctx.stop.load(.acquire)) {
        const stream = listener.acceptStream() catch |err| {
            if (ctx.stop.load(.acquire)) break;
            log.err("accept: {t}", .{err});
            break;
        };
        if (ctx.conns.load(.acquire) >= server.max_connections) {
            log.warn("refusing a connection: {d} already open", .{server.max_connections});
            stream.close(io);
            continue;
        }
        _ = ctx.conns.fetchAdd(1, .acq_rel);
        const t = std.Thread.spawn(.{}, connMain, .{ &ctx, stream }) catch {
            _ = ctx.conns.fetchSub(1, .acq_rel);
            stream.close(io);
            continue;
        };
        t.detach();
    }
    ctx.shutdown("listener closed");
}

/// Whether `dirs` already names `dir`, by canonical path where both resolve.
fn folderListed(io: Io, arena: std.mem.Allocator, dirs: []const []const u8, dir: []const u8) bool {
    const want = Io.Dir.cwd().realPathFileAlloc(io, dir, arena) catch dir;
    for (dirs) |d| {
        const have = Io.Dir.cwd().realPathFileAlloc(io, d, arena) catch d;
        if (std.mem.eql(u8, have, want)) return true;
    }
    return false;
}

fn usage() error{BadArgs} {
    std.debug.print(
        \\usage: tp-serve [--socket <path>] [--autospawn] [--config <file>] [--index <file>]
        \\       tp-serve --remote <bind>:<port> [--models <dir>]... [--backend cpu|vulkan|zig_cuda|cuda]
        \\                [--state <dir>] [--pair-host <name>] [--new-token] [--incoming <dir>]
        \\                [--config <file>]
        \\
    , .{});
    return error.BadArgs;
}
