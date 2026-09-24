//! The engine host: a `Driver` behind an inbox of wire requests and an outbox
//! of wire events, run by one thread. tp-gui in-process and tp-serve over a
//! socket both drive it the same way, which is what keeps the two from
//! drifting: a client never dereferences an engine pointer, it reads frames.
//!
//! One iteration drains the inbox and handles every request on THIS thread,
//! runs the driver's per-frame hooks, polls the session, then diffs what the
//! engine holds against what was last sent and emits only the changes: a
//! `state` block when any bit of it moved, a whole `transcript` when its shape
//! changed and `delta`s while text streams, an `img` per image that changed,
//! `telemetry` on a cadence. Pixels never travel unasked: a client that sees a
//! new `preview_rev` or a finished image asks with `img_fetch` and gets one
//! binary frame back.
//!
//! One Host per process, like the Driver it owns: the session's wake callback
//! carries no context.
const std = @import("std");
const Io = std.Io;
const tp = @import("TensorPencil");
const wire = @import("serve").wire;
const queue = @import("serve").queue;
const chat = @import("chat.zig");
const diffuser = @import("diffuser.zig");
const Driver = @import("driver.zig").Driver;
const sysmon = @import("sysmon.zig");
const scan = @import("scan.zig");
const config = @import("shared").config;
const catalog = @import("shared").catalog;

const log = std.log.scoped(.host);

pub const Frame = queue.Frame;
pub const Queue = queue.Queue;

/// The folders a host scans on its own account.
pub const Folders = struct { dirs: []const []const u8, files: []const []const u8 };

pub const Options = struct {
    /// Where the catalog scan caches what it found; null for no cache.
    index_path: ?[]const u8 = null,
    /// Set for a host reached over the network. Its catalog then names models
    /// by id and stem only, a client's paths are refused (only id references
    /// resolve, against this catalog), and a `scan` request covers these
    /// folders, never the client's. Must outlive the host.
    folders: ?Folders = null,
    /// Ask the Vulkan loader for the card's heap at start. Off in unit tests,
    /// which must not create a device instance.
    probe_card: bool = true,
};

/// What a client may ask for. One place, so a settings push and a render
/// request refuse the same things; each check names the offending field.
pub const limits = struct {
    pub const min_edge: u64 = 64;
    pub const max_edge: u64 = 8192;
    pub const edge_step: u64 = 8;
    pub const max_pixels: u64 = 64 << 20;
    pub const min_steps: u64 = 1;
    pub const max_steps: u64 = 500;
    pub const max_new_tokens: u64 = 1 << 20;
    pub const max_cfg: f32 = 100;
    pub const max_lora_strength: f32 = 10;
    pub const max_temperature: f32 = 10;
    pub const max_top_k: u64 = 1 << 20;
    pub const max_repeat_penalty: f32 = 100;
    pub const max_repeat_last_n: u64 = 1 << 20;
    pub const max_flat_penalty: f32 = 100;
    pub const max_noise_amount: f32 = 10;

    pub fn checkImage(width: u64, height: u64, steps: u64) ?[]const u8 {
        if (!edgeOk(width)) return "width";
        if (!edgeOk(height)) return "height";
        if (width * height > max_pixels) return "pixels";
        if (steps < min_steps or steps > max_steps) return "steps";
        return null;
    }

    pub fn checkImageRequest(ir: *const wire.ImageRequest) ?[]const u8 {
        if (checkImage(ir.width, ir.height, ir.steps)) |f| return f;
        if (!inRange(ir.cfg, 0, max_cfg)) return "cfg";
        if (ir.loras.len > config.max_family_loras) return "loras";
        for (ir.loras) |l| {
            if (l.path.len == 0 or l.path.len >= config.max_path) return "lora path";
            if (!inRange(l.strength, -max_lora_strength, max_lora_strength)) return "lora strength";
        }
        return null;
    }

    pub fn checkSettings(hs: *const config.HostSettings) ?[]const u8 {
        if (checkImage(hs.width, hs.height, hs.steps)) |f| return f;
        if (hs.max_new_tokens > max_new_tokens) return "max_new_tokens";
        if (!inRange(hs.weight_noise_amount, 0, max_noise_amount)) return "weight_noise_amount";
        return checkSampling(&hs.sampling);
    }

    pub fn checkSampling(s: *const config.Sampling) ?[]const u8 {
        if (!inRange(s.temperature, 0, max_temperature)) return "temperature";
        if (s.top_k > max_top_k) return "top_k";
        if (!inRange(s.top_p, 0, 1)) return "top_p";
        if (!inRange(s.min_p, 0, 1)) return "min_p";
        // Logits are divided by it, so 0 is not a value.
        if (!inRange(s.repeat_penalty, 0, max_repeat_penalty) or s.repeat_penalty == 0) return "repeat_penalty";
        if (s.repeat_last_n > max_repeat_last_n) return "repeat_last_n";
        if (!inRange(s.presence_penalty, -max_flat_penalty, max_flat_penalty)) return "presence_penalty";
        if (!inRange(s.frequency_penalty, -max_flat_penalty, max_flat_penalty)) return "frequency_penalty";
        return null;
    }

    fn edgeOk(e: u64) bool {
        return e >= min_edge and e <= max_edge and e % edge_step == 0;
    }

    /// NaN fails every comparison, so it is refused too.
    fn inRange(x: f32, lo: f32, hi: f32) bool {
        return std.math.isFinite(x) and x >= lo and x <= hi;
    }
};

var g_host: ?*Host = null;

/// The driver's `wake`: bump the loop's counter and wake the engine thread.
fn hostWake() void {
    const h = g_host orelse return;
    _ = h.wake.fetchAdd(1, .release);
    h.io.futexWake(u32, &h.wake.raw, 1);
}

fn noopBeforeLoad() void {}

/// Telemetry cadence: the meters only move while something works.
const telemetry_idle_ns: i96 = 500 * std.time.ns_per_ms;
const telemetry_busy_ns: i96 = 200 * std.time.ns_per_ms;
/// The ticker's period: what bounds the latency of anything cadence-driven
/// (telemetry, the arbiter's retry, a preview poll) when no event wakes the loop.
const tick_ns: u64 = 100 * std.time.ns_per_ms;
/// What a stalled events reader may leave queued before the queue refuses.
const outbox_max_bytes: usize = 256 << 20;

pub const Host = struct {
    gpa: std.mem.Allocator,
    /// First scanned model folder, where a derived direction set is written.
    out_dir: config.PathBuf = .{},
    io: Io,
    drv: Driver,
    inbox: Queue = .{},
    outbox: Queue = .{},
    /// Bumped by `hostWake`; the loop parks on it.
    wake: std.atomic.Value(u32) = .init(0),
    /// Called after frames land in the outbox, so the client can repaint.
    on_out: *const fn () void,
    gen: wire.HostGen,
    stop_flag: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    ticker: ?std.Thread = null,

    // What was last sent, so an iteration emits only changes.
    last_state: []u8 = &.{},
    /// Hash of the transcript's shape (message count, variants, cur, images).
    tx_shape: u64 = 0,
    /// Emitted text length per variant, in (message, variant) order.
    tx_lens: std.ArrayList(usize) = .empty,
    tx_stats: std.ArrayList(wire.TurnStats) = .empty,
    llm_was_busy: bool = false,
    img_seen: std.AutoHashMapUnmanaged(wire.ImageId, ImgPrint) = .empty,
    had_diffuser: bool = false,
    last_telemetry_ns: i96 = 0,
    cpu_meter: sysmon.CpuMeter = .{},
    /// GPU busy and clock where NVML cannot answer (every open driver).
    drm_meter: sysmon.DrmMeter = .{},
    /// The Vulkan card's device-local heap, probed once at start so an idle
    /// host still reports the card it has. 0 when there is no Vulkan device.
    probed_heap: u64 = 0,
    drain_buf: std.ArrayList(Frame) = .empty,
    /// Queued requests taken off the inbox, engine thread only. What an urgent
    /// verb's stamp is compared against (`Driver.Urgent`).
    handled: u64 = 0,
    /// Bumped per finished scan; the client applies a catalog only when it moved.
    catalog_rev: u64 = 0,
    scan_was_running: bool = false,
    /// `Options.folders`: the host is remote to its clients.
    folders: ?Folders = null,
    /// id -> path for every file this host holds, republished after each scan.
    /// Behind its own lock because it is the ONE piece of the catalog a
    /// connection thread reads: a pull is served off the connection, and the
    /// catalog itself belongs to the engine thread.
    offer_mu: Io.Mutex = .init,
    offer_arena: ?std.heap.ArenaAllocator = null,
    offer_rows: []OfferRow = &.{},

    pub const OfferRow = struct { id: catalog.ModelId, path: []const u8 };

    /// Where this host holds `id_text`, copied into `buf`. Safe from any
    /// thread; null when it holds no such file (or the id is not one).
    ///
    /// Only files under the scanned folders are nameable, so this cannot be
    /// asked for a path the host was never told to hold.
    pub fn offerPath(self: *Host, id_text: []const u8, buf: []u8) ?[]const u8 {
        const id = catalog.parseId(id_text) orelse return null;
        self.offer_mu.lockUncancelable(self.io);
        defer self.offer_mu.unlock(self.io);
        for (self.offer_rows) |r| {
            if (r.id != id or r.path.len > buf.len) continue;
            @memcpy(buf[0..r.path.len], r.path);
            return buf[0..r.path.len];
        }
        return null;
    }

    /// Republish the table from the catalog a scan just finished. Engine thread.
    fn publishOffers(self: *Host) void {
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        const a = arena.allocator();
        const rows = a.alloc(OfferRow, scan.cat.entries.len) catch {
            arena.deinit();
            return;
        };
        for (scan.cat.entries, rows) |*e, *r| {
            r.* = .{ .id = e.id(), .path = a.dupe(u8, e.path) catch "" };
        }
        self.offer_mu.lockUncancelable(self.io);
        defer self.offer_mu.unlock(self.io);
        if (self.offer_arena) |*old| old.deinit();
        self.offer_arena = arena;
        self.offer_rows = rows;
    }

    /// The fields of an image a client renders from; a change in any means an
    /// `img` event.
    const ImgPrint = struct {
        status: wire.ImageStatus,
        step: u32,
        total: u32,
        preview_rev: u32,
        has_pixels: bool,
        failure: u16,
    };

    /// In-place: the driver's callbacks hold `self`.
    pub fn init(self: *Host, gpa: std.mem.Allocator, io: Io, settings: *const config.Config, opts: Options, on_out: *const fn () void) void {
        if (g_host != null) @panic("one Host per process");
        self.* = .{
            .gpa = gpa,
            .io = io,
            .drv = undefined,
            .on_out = on_out,
            .gen = @truncate(@as(u96, @bitCast(Io.Clock.real.now(io).toNanoseconds()))),
            .folders = opts.folders,
        };
        g_host = self;
        diffuser.seedImageIds(imageIdBase(self.gen));
        self.drv.init(gpa, io, hostWake, settings);
        scan.init(gpa, io, hostWake, opts.index_path);
        // A remote host's catalog is ready before any client asks.
        if (opts.folders) |f| scan.startScan(f.dirs, f.files);
        // The image engine exists as soon as a model is configured; its pipeline
        // still loads on the first image, so the card is asked about here: an
        // idle host reports what it has rather than nothing.
        if (opts.probe_card and tp.gpu.context.loaderPresent()) self.probed_heap = tp.gpu.context.probeHeapBytes(gpa) orelse 0;
        // The meter's reserve is a fraction of the card, and on a host with no
        // LLM this is the only figure for it until a pipeline is already loading.
        self.drv.card_bytes_hint = self.probed_heap;
        self.outbox.max_bytes = outbox_max_bytes;
        self.drv.syncDiffuser();
    }

    pub fn deinit(self: *Host) void {
        self.stop();
        scan.deinit();
        self.drv.deinit();
        self.inbox.deinit(self.gpa);
        self.outbox.deinit(self.gpa);
        for (self.drain_buf.items) |f| f.deinit(self.gpa);
        self.drain_buf.deinit(self.gpa);
        self.gpa.free(self.last_state);
        self.tx_lens.deinit(self.gpa);
        self.tx_stats.deinit(self.gpa);
        self.img_seen.deinit(self.gpa);
        if (self.offer_arena) |*a| a.deinit();
        if (g_host == self) g_host = null;
    }

    // ── Client side ───────────────────────────────────────────────────────────

    /// Queue a request (any thread). Takes ownership of `f`. Cancel and pause
    /// skip the queue (see `Driver.Urgent`).
    pub fn post(self: *Host, f: Frame) void {
        if (f == .text and self.tryUrgent(f.text)) {
            f.deinit(self.gpa);
            hostWake();
            return;
        }
        // Bumped BEFORE the push, so an urgent verb racing this one stamps
        // itself as behind it rather than ahead.
        _ = self.drv.urgent.posted.fetchAdd(1, .release);
        self.inbox.push(self.io, self.gpa, f) catch |err| {
            log.err("inbox: {t}", .{err});
            _ = self.drv.urgent.posted.fetchSub(1, .release);
            f.deinit(self.gpa);
            return;
        };
        hostWake();
    }

    /// Queue a JSON request (any thread).
    pub fn postRequest(self: *Host, req: wire.Request) void {
        const bytes = wire.encodeAlloc(self.gpa, req) catch |err| {
            log.err("encode request: {t}", .{err});
            return;
        };
        self.post(.{ .text = bytes });
    }

    /// The verbs that must not wait behind the queue, recognised from the
    /// request's first key (the encoder writes `{"tag":...}` with no space).
    /// True when `bytes` was one of them and its flag is set; false sends it
    /// through the inbox, which handles every verb, so a full cancel ring only
    /// costs the wait.
    fn tryUrgent(self: *Host, bytes: []const u8) bool {
        const tag = firstKey(bytes) orelse return false;
        const u = &self.drv.urgent;
        // Stamped before the flag is set, never after: the engine thread reading
        // a flag against a stale stamp is exactly the too-early apply.
        if (std.mem.eql(u8, tag, "chat_cancel")) {
            u.stamp();
            u.llm_cancel.store(true, .release);
        } else if (std.mem.eql(u8, tag, "img_cancel_all")) {
            u.stamp();
            u.img_cancel_all.store(true, .release);
        } else if (std.mem.eql(u8, tag, "img_cancel") or std.mem.eql(u8, tag, "chat_pause") or std.mem.eql(u8, tag, "img_pause")) {
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            const req = wire.decode(wire.Request, arena.allocator(), bytes) catch return false;
            u.stamp();
            switch (req) {
                .img_cancel => |c| return u.pushImgCancel(c.image),
                .chat_pause => |p| u.llm_pause.store(if (p.paused) 1 else 2, .release),
                .img_pause => |p| u.img_pause.store(if (p.paused) 1 else 2, .release),
                else => return false,
            }
        } else return false;
        return true;
    }

    /// Move every emitted frame into `out` (client thread). The caller frees them.
    pub fn take(self: *Host, out: *std.ArrayList(Frame)) void {
        self.outbox.drain(self.io, self.gpa, out);
    }

    /// Free every queued outbox frame (any thread). For when the events client
    /// detaches: the next one greets with a snapshot, so nothing queued is owed.
    pub fn clearOutbox(self: *Host) void {
        self.outbox.clear(self.io, self.gpa);
    }

    // ── Engine thread ─────────────────────────────────────────────────────────

    /// Run the loop on its own thread until `stop`.
    pub fn start(self: *Host) !void {
        // The engine thread parses settings pushes; see `config.parse_stack_size`.
        self.thread = try std.Thread.spawn(.{ .stack_size = config.parse_stack_size }, run, .{self});
        self.ticker = try std.Thread.spawn(.{}, tick, .{self});
    }

    pub fn stop(self: *Host) void {
        self.stop_flag.store(true, .release);
        hostWake();
        if (self.ticker) |t| t.join();
        self.ticker = null;
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    fn tick(self: *Host) void {
        while (!self.stop_flag.load(.acquire)) {
            Io.sleep(self.io, .{ .nanoseconds = tick_ns }, .real) catch {};
            hostWake();
        }
    }

    fn run(self: *Host) void {
        // The engine thread is whichever runs the loop, not the one that built it.
        self.drv.engine_thread = std.Thread.getCurrentId();
        while (true) {
            const seen = self.wake.load(.acquire);
            self.runOnce();
            if (self.stop_flag.load(.acquire)) break;
            self.io.futexWaitUncancelable(u32, &self.wake.raw, seen);
        }
    }

    /// One iteration: requests, hooks, poll, emit. Public so a probe can drive
    /// the host on its own thread, one step at a time.
    pub fn runOnce(self: *Host) void {
        self.drv.applyUrgent(self.handled);
        self.inbox.drain(self.io, self.gpa, &self.drain_buf);
        for (self.drain_buf.items) |f| {
            self.handle(f);
            self.handled += 1;
            f.deinit(self.gpa);
        }
        self.drain_buf.clearRetainingCapacity();
        // Again with the queue empty: a verb stamped behind what was just
        // drained is due now, and waits no further than this pass.
        self.drv.applyUrgent(self.handled);

        if (scan.poll()) {
            self.catalog_rev += 1;
            self.publishOffers();
            self.emitCatalog(true);
        } else if (scan.scanning() != self.scan_was_running) {
            self.emitCatalog(false);
        }
        self.drv.maybeProcessEjects();
        self.drv.maybeStartReload(noopBeforeLoad);
        self.drv.maybeRefreshMeterPolicy();
        self.drv.pumpDiffuser();
        if (self.drv.uiSession()) |s| {
            s.poll();
            self.drv.flushNotesTo(s);
            self.reportImageCalls(s, false);
        }
        self.drv.maybeApplyPendingTranscript();
        self.emitAll();
    }

    fn handle(self: *Host, f: Frame) void {
        switch (f) {
            .bin => |b| self.handleBin(b.hdr, b.payload),
            .text => |t| {
                var arena = std.heap.ArenaAllocator.init(self.gpa);
                defer arena.deinit();
                const req = wire.decode(wire.Request, arena.allocator(), t) catch |err| {
                    self.emitErr(.bad_request, @errorName(err));
                    return;
                };
                self.handleRequest(req);
            },
        }
    }

    fn handleBin(self: *Host, hdr: wire.BinHeader, payload: []const u8) void {
        if (hdr.magic != wire.BinHeader.magic_value) return self.emitErr(.bad_request, "BadMagic");
        switch (hdr.kind) {
            .rgb_upload => {
                const need = @as(u64, hdr.w) *| hdr.h *| 3;
                if (need == 0 or need > payload.len) return self.emitErr(.bad_request, "ShortPayload");
                self.drv.attachOrStage(payload[0..@intCast(need)], hdr.w, hdr.h);
            },
            else => self.emitErr(.bad_request, "UnexpectedBinary"),
        }
    }

    fn handleRequest(self: *Host, req: wire.Request) void {
        const drv = &self.drv;
        switch (req) {
            .hello => |h| {
                if (h.proto != wire.proto) return self.emitErr(.proto_mismatch, "proto");
                self.emit(.{ .hello = .{ .gen = self.gen } });
            },
            .snapshot => self.forgetEmitted(),
            .settings => |s| {
                var arena = std.heap.ArenaAllocator.init(self.gpa);
                defer arena.deinit();
                const a = arena.allocator();
                const hs = std.json.parseFromSliceLeaky(config.HostSettings, a, s.json, .{ .ignore_unknown_fields = true }) catch |err| {
                    return self.emitErr(.bad_request, @errorName(err));
                };
                if (limits.checkSettings(&hs)) |field| return self.emitErr(.bad_request, field);
                // Onto a copy of what is in force: the client's fields never
                // reach the host, so they keep the values this host started with.
                const next = a.create(config.Config) catch return self.emitErr(.internal, "OutOfMemory");
                next.* = drv.settings;
                config.applyHost(next, &hs);
                if (self.folders != null) {
                    resolveModelRefs(next, &scan.cat, true);
                    inline for (config.machine_fields) |name| @field(next, name) = @field(drv.settings, name);
                } else resolveModelRefs(next, &scan.cat, false);
                drv.applySettings(next);
            },
            .meter => |m| {
                drv.split = std.math.clamp(m.split, 0.02, 0.96);
                drv.limit = std.math.clamp(m.limit, 0.10, 0.985);
                drv.applyMeterPolicy();
            },
            .chat_submit => |c| {
                if (!drv.submit(c.text)) self.emitErr(.refused, "submit");
            },
            .chat_regenerate => {
                if (drv.uiSession()) |s| {
                    s.regenerate() catch |err| self.emitErr(.internal, @errorName(err));
                } else drv.requestRegenLoad();
            },
            .chat_cancel => drv.cancelLlm(),
            .chat_pause => |p| drv.setLlmPaused(p.paused),
            .chat_new => drv.newChat(),
            .chat_adopt => |a| self.adopt(a.messages),
            .chat_select_variant => |sv| if (drv.uiSession()) |s| {
                // The session's own API navigates the LAST message only.
                if (@as(usize, sv.msg) + 1 == s.messages.items.len) s.selectVariant(sv.variant);
            } else {
                // Nothing resident: the carried transcript just shows another
                // take; the next session adopts whichever is active.
                drv.carry_mu.lockUncancelable(self.io);
                defer drv.carry_mu.unlock(self.io);
                if (drv.carry) |c| if (sv.msg < c.items.len) {
                    const m = &c.items[sv.msg];
                    if (sv.variant < m.variants.items.len) m.cur = sv.variant;
                };
            },
            .chat_remove_attachment => |r| {
                if (drv.uiSession()) |s| s.removeAttachment(r.index) else drv.removeStaged(r.index);
            },
            .chat_eject => drv.llm_eject_armed = true,
            .chat_image => |ci| self.recordImage(ci),
            .chat_note => |n| if (n.text.len > 0) {
                drv.queueNote(self.gpa.dupe(u8, n.text) catch return);
            },

            .img_enqueue => |ir| self.enqueue(ir),
            .act_derive => |dr| self.actDerive(dr),
            .img_cancel => |c| drv.cancelImage(c.image),
            .img_cancel_all => drv.cancelAllImages(),
            // Its print goes with it: the id is never minted again, so a kept
            // entry is dead weight that a long-lived daemon accumulates.
            .img_ack => |a| if (drv.dropImage(a.image)) {
                _ = self.img_seen.remove(a.image);
            },
            .img_move => |m| if (drv.diffuser) |*d| d.movePending(m.image, m.before),
            .img_pause => |p| drv.setDiffPaused(p.paused),
            .img_eject => drv.diff_eject_armed = true,
            .img_fetch => |f| self.fetch(f),
            .scan => |s| {
                const dirs = if (self.folders) |f| f.dirs else s.dirs;
                // Remembered so a derived direction set has somewhere to land: the
                // scan request's own slices die with the frame.
                if (dirs.len > 0) self.out_dir.set(dirs[0]);
                if (self.folders) |f| scan.startScan(f.dirs, f.files) else scan.startScan(s.dirs, s.files);
            },
        }
    }

    /// Every model reference in `cfg` becomes a path of `cat`'s: an id text
    /// resolves to the file it names here, or to nothing when this host lacks
    /// it. A remote host (`private`) also drops plain paths: a client never
    /// names a file on another machine's disk.
    fn resolveModelRefs(cfg: *config.Config, cat: *const catalog.Catalog, private: bool) void {
        inline for (config.model_ref_fields) |name| resolveRef(&@field(cfg, name), cat, private);
        for (cfg.loras.items[0..cfg.loras.count]) |*l| resolveRef(&l.path, cat, private);
    }

    fn resolveRef(field: *config.PathBuf, cat: *const catalog.Catalog, private: bool) void {
        const text = field.opt() orelse return;
        if (catalog.parseId(text)) |id| {
            field.set(if (cat.byId(id)) |e| e.path else "");
        } else if (private) {
            field.set("");
        }
    }

    /// A render request becomes a queued image. Everything the request carries
    /// is stamped on the image here, so a later settings change cannot reach it.
    /// LoRA paths go through `resolveRef` like a settings push: an id names a
    /// file of this host's, and a remote host takes no plain path.
    /// Run a direction derivation and write the file beside the models.
    ///
    /// Inline on the engine thread: it takes seconds and there is nothing useful to
    /// do meanwhile, so it blocks rather than growing a second kind of queued job.
    fn actDerive(self: *Host, dr: anytype) void {
        const gpa = self.gpa;
        // Every exit answers with `act_derived`: the client is waiting on it, and an
        // `err` event alone leaves it showing "working" for good.
        const d = &(self.drv.diffuser orelse return self.actFailed("no image model"));
        const stem = std.mem.trim(u8, dr.name, " \t");
        if (stem.len == 0) return self.actFailed("needs a name");
        if (std.mem.indexOfAny(u8, stem, "/\\") != null) return self.actFailed("the name cannot be a path");

        const dir = self.out_dir.opt() orelse return self.actFailed("no model folder to write into");
        const path = std.fmt.allocPrint(gpa, "{s}/{s}.actd", .{ dir, stem }) catch return self.actFailed("out of memory");
        defer gpa.free(path);

        var a_list: std.ArrayList([]const u8) = .empty;
        defer a_list.deinit(gpa);
        var b_list: std.ArrayList([]const u8) = .empty;
        defer b_list.deinit(gpa);
        for ([_][]const u8{ dr.set_a, dr.set_b }, [_]*std.ArrayList([]const u8){ &a_list, &b_list }) |set, list| {
            var it = std.mem.splitScalar(u8, set, ';');
            while (it.next()) |raw| {
                const p = std.mem.trim(u8, raw, " \t\r\n");
                if (p.len > 0) list.append(gpa, p) catch return self.actFailed("out of memory");
            }
        }
        if (a_list.items.len == 0 or b_list.items.len == 0) return self.actFailed("needs prompts on both sides");

        d.deriveActDirs(gpa, self.io, a_list.items, b_list.items, dr.size, dr.steps, path) catch |err| {
            std.log.err("act-derive failed: {t}", .{err});
            var buf: [96]u8 = undefined;
            const text = std.fmt.bufPrint(&buf, "{t}", .{err}) catch "failed";
            self.emit(.{ .act_derived = .{ .err = text } });
            return;
        };
        self.emit(.{ .act_derived = .{ .path = path } });
    }

    fn actFailed(self: *Host, text: []const u8) void {
        std.log.err("act-derive: {s}", .{text});
        self.emit(.{ .act_derived = .{ .err = text } });
    }

    fn enqueue(self: *Host, ir: wire.ImageRequest) void {
        const drv = &self.drv;
        if (limits.checkImageRequest(&ir)) |field| return self.emitErr(.bad_request, field);
        var specs: [config.max_family_loras]tp.pipeline.LoraSpec = undefined;
        var paths: [config.max_family_loras]config.PathBuf = undefined;
        for (ir.loras, specs[0..ir.loras.len], paths[0..ir.loras.len]) |l, *s, *p| {
            p.set(l.path);
            resolveRef(p, &scan.cat, self.folders != null);
            const path = p.opt() orelse return self.emitErr(if (catalog.parseId(l.path) != null) .not_found else .bad_request, "lora");
            s.* = .{ .path = path, .strength = l.strength };
        }
        const d = &(drv.diffuser orelse return self.emitErr(.no_model, "no image model"));
        const gpa = self.gpa;
        const gi = gpa.create(diffuser.GenImage) catch return self.emitErr(.internal, "OutOfMemory");
        gi.* = .{
            .client_ref = ir.client_ref,
            .prompt = gpa.dupe(u8, ir.prompt) catch {
                gpa.destroy(gi);
                return self.emitErr(.internal, "OutOfMemory");
            },
            .wake = drv.wake,
            .io = self.io,
            .req_width = ir.width,
            .req_height = ir.height,
            .req_steps = ir.steps,
            .req_cfg = ir.cfg,
            .params = ir.params,
            .from_studio = ir.from_studio,
            .req_seed = if (ir.seed == 0) d.nextSeed() else ir.seed,
        };
        if (ir.negative.len > 0) gi.req_negative = gpa.dupe(u8, ir.negative) catch "";
        if (ir.loras.len > 0) {
            // Before `enqueue`, which only stamps an image with no snapshot of its own.
            d.setImageLoras(gi, specs[0..ir.loras.len]) catch {
                diffuser.freeGenImage(gpa, gi);
                return self.emitErr(.internal, "OutOfMemory");
            };
        }
        d.enqueue(gi) catch {
            diffuser.freeGenImage(gpa, gi);
            return self.emitErr(.internal, "OutOfMemory");
        };
        // Through the driver, not `d.pump()`: a render started while the LLM is
        // loading sees no LLM in the arbiter, takes the whole budget, and the
        // model then loads into a card the image already claimed. The main loop
        // pumps it when the load is done.
        drv.pumpDiffuser();
    }

    /// Tell the client what a finished reply asked for. The host queues none of
    /// it: the client places each call like a render of its own and names what
    /// came back with `chat_image`. The sizes come from the settings, so a chat
    /// host with no image engine still reports the call.
    /// Report the renders the model asked for. `again` re-parses a reply
    /// already reported, for a client that was not listening the first time;
    /// placing one twice is prevented by the client, which knows the calls it
    /// has already taken.
    fn reportImageCalls(self: *Host, s: *chat.Session, again: bool) void {
        var calls: std.ArrayList(chat.ImageCall) = .empty;
        defer calls.deinit(self.gpa);
        const cfg = &self.drv.settings;
        const defaults: chat.CallDefaults = .{ .width = cfg.width, .height = cfg.height, .steps = cfg.steps };
        if (again) s.rescanImages(self.gpa, defaults, &calls) else s.scanNewImages(self.gpa, defaults, &calls);
        if (calls.items.len == 0) return;
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const out = arena.allocator().alloc(wire.ImageCall, calls.items.len) catch return;
        for (calls.items, out, 0..) |c, *w, i| w.* = .{
            .msg = @intCast(c.msg),
            .variant = @intCast(c.variant),
            .index = @intCast(i),
            .req = .{
                .prompt = c.prompt,
                .width = @intCast(c.width),
                .height = @intCast(c.height),
                .steps = @intCast(c.steps),
                .seed = c.seed,
            },
        };
        self.emit(.{ .img_requested = .{ .calls = out } });
    }

    /// Record the image a tool call produced, wherever it ran. The transcript's
    /// shape hash counts each variant's images, so the append is what sends the
    /// transcript again. A pair the transcript no longer has is dropped.
    fn recordImage(self: *Host, ci: @FieldType(wire.Request, "chat_image")) void {
        if (ci.image == 0) return;
        const drv = &self.drv;
        if (drv.uiSession()) |s| {
            const v = variantAt(s.messages.items, ci.msg, ci.variant) orelse return;
            self.setVariantImage(v, ci.image, ci.replaces);
            return;
        }
        drv.carry_mu.lockUncancelable(self.io);
        defer drv.carry_mu.unlock(self.io);
        const c = drv.carry orelse return;
        const v = variantAt(c.items, ci.msg, ci.variant) orelse return;
        self.setVariantImage(v, ci.image, ci.replaces);
    }

    /// The image a tool call became, in the slot of the one it replaces so a
    /// render moved to another host does not leave the dead one behind.
    fn setVariantImage(self: *Host, v: *chat.Variant, id: wire.ImageId, replaces: wire.ImageId) void {
        for (v.images.items) |*have| if (have.* == id) return;
        if (replaces != 0) {
            for (v.images.items) |*have| if (have.* == replaces) {
                have.* = id;
                return;
            };
        }
        v.images.append(self.gpa, id) catch {};
    }

    fn variantAt(msgs: []chat.Message, msg: u32, variant: u32) ?*chat.Variant {
        if (msg >= msgs.len) return null;
        const m = &msgs[msg];
        if (variant >= m.variants.items.len) return null;
        return &m.variants.items[variant];
    }

    /// An image by id: the engine's list first, then the session's attachments
    /// (pending, and on every message), which the session owns.
    fn imageById(self: *Host, id: wire.ImageId) ?*diffuser.GenImage {
        if (id == 0) return null;
        if (self.drv.diffuser) |*d| if (d.byId(id)) |gi| return gi;
        const s = self.drv.uiSession() orelse return null;
        for (s.pendingAttachments()) |gi| if (gi.id == id) return gi;
        for (s.messages.items) |*m| for (m.attachments.items) |gi| if (gi.id == id) return gi;
        return null;
    }

    /// Answer `img_fetch` with one binary frame: the live preview, box-filtered
    /// to `max_edge`, or the finished pixels. Nothing when there is nothing yet.
    fn fetch(self: *Host, f: @FieldType(wire.Request, "img_fetch")) void {
        const gi = self.imageById(f.image) orelse return;
        switch (f.kind) {
            .preview => {
                const pv = gi.preview orelse return;
                const pw = gi.preview_w.load(.acquire);
                const ph = gi.preview_h.load(.acquire);
                const rev = gi.preview_rev.load(.acquire);
                if (pw == 0 or ph == 0 or rev == f.have_rev) return;
                const src = pv[0 .. @as(usize, pw) * ph * 4];
                // Integer box filter: the largest factor that keeps the longer
                // side at or above the asked edge.
                var k: u32 = 1;
                if (f.max_edge > 0) {
                    while (@max(pw, ph) / (k + 1) >= f.max_edge) k += 1;
                }
                const ow = pw / k;
                const oh = ph / k;
                const out = self.gpa.alloc(u8, @as(usize, ow) * oh * 4) catch return;
                boxDown(src, pw, out, ow, oh, k);
                self.pushBin(.{ .kind = .preview_rgba, .id = gi.id, .rev = rev, .w = ow, .h = oh, .len = @intCast(out.len) }, out);
            },
            .pixels => {
                const rgba = gi.rgba orelse return;
                if (gi.get() != .done) return;
                const out = self.gpa.dupe(u8, rgba) catch return;
                self.pushBin(.{ .kind = .image_rgba, .id = gi.id, .rev = 1, .w = @intCast(gi.width), .h = @intCast(gi.height), .len = @intCast(out.len) }, out);
            },
        }
    }

    /// Rebuild the engine's transcript from the wire form. Image ids are kept
    /// as they are. An attachment comes back as an id with no pixels (they are
    /// not on the wire), so it lists and re-adopts but cannot be fetched or
    /// re-encoded for a regenerate.
    fn adopt(self: *Host, msgs: []const wire.Message) void {
        const gpa = self.gpa;
        var list: std.ArrayList(chat.Message) = .empty;
        for (msgs) |wm| {
            var m = chat.Message.init(gpa, if (wm.role == .user) .user else .assistant) catch break;
            m.synthetic = wm.synthetic;
            for (wm.attachments) |id| {
                const gi = gpa.create(diffuser.GenImage) catch break;
                gi.* = .{
                    .id = id,
                    .prompt = gpa.dupe(u8, "") catch {
                        gpa.destroy(gi);
                        break;
                    },
                    .status = .init(@intFromEnum(diffuser.GenStatus.done)),
                    .wake = self.drv.wake,
                    .io = self.io,
                };
                m.attachments.append(gpa, gi) catch {
                    diffuser.freeGenImage(gpa, gi);
                    break;
                };
            }
            // `init` made one empty variant; fill it, then append the rest.
            for (wm.variants, 0..) |wv, i| {
                if (i > 0) m.variants.append(gpa, .{}) catch break;
                const v = &m.variants.items[m.variants.items.len - 1];
                v.text.appendSlice(gpa, wv.text) catch {};
                v.thought_primed = wv.thought_primed;
                // Already acted on: never re-dispatch a stored turn's tool calls.
                v.images_scanned = true;
                v.stats = wv.stats;
                if (wv.reason_open.len > 0 and wv.reason_close.len > 0) {
                    if (gpa.dupe(u8, wv.reason_open)) |o| {
                        if (gpa.dupe(u8, wv.reason_close)) |c| {
                            v.reason_open = o;
                            v.reason_close = c;
                        } else |_| gpa.free(o);
                    } else |_| {}
                }
                if (wv.gen_model.len > 0) v.gen_model = gpa.dupe(u8, wv.gen_model) catch "";
                v.images.appendSlice(gpa, wv.images) catch {};
            }
            m.cur = @min(wm.cur, m.variants.items.len - 1);
            list.append(gpa, m) catch {
                m.deinit(gpa);
                break;
            };
        }
        self.drv.adoptTranscript(list);
    }

    // ── Emission ──────────────────────────────────────────────────────────────

    fn emit(self: *Host, ev: wire.Event) void {
        _ = self.emitOk(ev);
    }

    /// False when the event did not reach the outbox, so a caller keeping a
    /// "already sent" memo can hold it back rather than never sending it.
    fn emitOk(self: *Host, ev: wire.Event) bool {
        const bytes = wire.encodeAlloc(self.gpa, ev) catch |err| {
            log.err("encode event: {t}", .{err});
            return false;
        };
        self.outbox.push(self.io, self.gpa, .{ .text = bytes }) catch {
            self.gpa.free(bytes);
            return false;
        };
        self.on_out();
        return true;
    }

    fn emitErr(self: *Host, code: wire.ErrCode, text: []const u8) void {
        self.emit(.{ .err = .{ .code = code, .text = text } });
    }

    fn pushBin(self: *Host, hdr: wire.BinHeader, payload: []u8) void {
        self.outbox.push(self.io, self.gpa, .{ .bin = .{ .hdr = hdr, .payload = payload } }) catch {
            self.gpa.free(payload);
            return;
        };
        self.on_out();
    }

    /// Make the next `emitAll` send everything, for a snapshot.
    fn forgetEmitted(self: *Host) void {
        self.gpa.free(self.last_state);
        self.last_state = &.{};
        self.tx_shape = 0;
        self.img_seen.clearRetainingCapacity();
        self.last_telemetry_ns = 0;
        // A snapshot must carry the whole list even when nothing is queued.
        self.emit(.{ .queue = .{ .images = &.{} } });
        // A reply that asked for a render while nobody was listening reached
        // no client, and nothing here keeps it: say it again.
        if (self.drv.uiSession()) |s| self.reportImageCalls(s, true);
        self.emitCatalog(true);
    }

    /// The catalog's status, with its entries when `with_entries`.
    fn emitCatalog(self: *Host, with_entries: bool) void {
        self.scan_was_running = scan.scanning();
        const rep = scan.lastReport();
        const json: []u8 = if (with_entries) scan.cat.toWireJsonAlloc(self.gpa, self.folders != null) catch |err| {
            log.err("encode catalog: {t}", .{err});
            return;
        } else &.{};
        defer if (json.len > 0) self.gpa.free(json);
        self.emit(.{ .catalog = .{
            .rev = self.catalog_rev,
            .scanning = self.scan_was_running,
            .files = @intCast(rep.files),
            .probed = @intCast(rep.probed),
            .reused = @intCast(rep.reused),
            .bad_folders = @intCast(rep.bad_folders),
            .json = json,
        } });
    }

    fn emitAll(self: *Host) void {
        self.emitState();
        self.emitTranscript();
        self.emitImages();
        self.emitTelemetry();
        if (self.drv.takePeakUpdate()) |u| self.emit(.{ .diff_peak = .{ .peak = u.peak, .key = u.key } });
        if (self.drv.takeDiffNotice()) |text| {
            defer self.gpa.free(text);
            self.emit(.{ .notice = .{ .tone = .info, .text = text } });
        }
    }

    fn emitState(self: *Host) void {
        const drv = &self.drv;
        var attach_buf: [64]wire.ImageId = undefined;
        var n_attach: usize = 0;
        var st: wire.State = .{
            .loading = drv.loading.load(.acquire),
            .load_err = if (drv.load_err) |e| @errorName(e) else "",
            .llm_paused = drv.llm_paused,
            .llm_eject_armed = drv.llm_eject_armed,
            .pending_submit = drv.pending_submit != null,
            .staged = @intCast(drv.staged.items.len),
            .diff_eject_armed = drv.diff_eject_armed,
            .vision = drv.visionAvailable(),
            .thinking = drv.configuredSupportsThinking(),
            .reasoning_effort = drv.configuredSupportsReasoningEffort(),
            .weight_noise = drv.noiseAvailable(),
        };
        if (drv.uiSession()) |s| {
            st.llm_resident = true;
            st.llm_model = s.model_name;
            if (tp.llm.chat.reasoning()) |r| {
                st.reason_open = r.open;
                st.reason_close = r.close;
            }
            st.gen_err = if (s.gen_err) |e| @errorName(e) else "";
            st.ctx_lost = s.be.ctx.isLost();
            st.llm_busy = s.busy();
            st.turn_pending = s.turnPending();
            for (s.pendingAttachments()) |gi| {
                if (n_attach == attach_buf.len) break;
                attach_buf[n_attach] = gi.id;
                n_attach += 1;
            }
        }
        st.attachments = attach_buf[0..n_attach];
        if (drv.diffuser) |*d| {
            st.diff_present = true;
            st.diff_busy = d.busyNow();
            st.diff_paused = d.isPaused();
            st.diff_load_err = if (d.loadError()) |e| @errorName(e) else "";
            st.diff_family = if (d.loadedFamily()) |f| @tagName(f) else "";
            st.pending_images = @intCast(d.pendingCount());
        }
        const bytes = wire.encodeAlloc(self.gpa, wire.Event{ .state = st }) catch return;
        if (std.mem.eql(u8, bytes, self.last_state)) {
            self.gpa.free(bytes);
            return;
        }
        // The memo moves only once the frame is queued. Committing it first and
        // then failing the push makes this state the baseline that was never
        // sent, and the mirror stays wrong until some later field happens to
        // change.
        const copy = self.gpa.dupe(u8, bytes) catch {
            self.gpa.free(bytes);
            return;
        };
        self.outbox.push(self.io, self.gpa, .{ .text = copy }) catch {
            self.gpa.free(copy);
            self.gpa.free(bytes);
            return;
        };
        self.gpa.free(self.last_state);
        self.last_state = bytes;
        self.on_out();
    }

    /// The transcript the client should mirror: the live session's, or the one
    /// carried across an unload. `carry_mu` is held for the whole walk.
    fn emitTranscript(self: *Host) void {
        const drv = &self.drv;
        var msgs: []chat.Message = &.{};
        var busy = false;
        var locked = false;
        if (drv.uiSession()) |s| {
            msgs = s.messages.items;
            busy = s.busy();
        } else {
            drv.carry_mu.lockUncancelable(self.io);
            locked = true;
            if (drv.carry) |c| msgs = c.items;
        }
        defer if (locked) drv.carry_mu.unlock(self.io);

        var h = std.hash.Wyhash.init(0x7061);
        h.update(std.mem.asBytes(&msgs.len));
        for (msgs) |*m| {
            h.update(std.mem.asBytes(&m.variants.items.len));
            h.update(std.mem.asBytes(&m.cur));
            h.update(std.mem.asBytes(&m.attachments.items.len));
            for (m.variants.items) |*v| h.update(std.mem.asBytes(&v.images.items.len));
        }
        const shape = h.final();

        if (shape != self.tx_shape) {
            // Every memo moves only once its frame is queued: a push the outbox
            // refused with the memo already advanced is text the mirror never
            // sees again, since growth alone does not move the shape.
            if (self.emitWholeTranscript(msgs)) self.tx_shape = shape;
        } else {
            var i: usize = 0;
            for (msgs, 0..) |*m, mi| for (m.variants.items, 0..) |*v, vi| {
                defer i += 1;
                if (i >= self.tx_lens.items.len) continue;
                const have = self.tx_lens.items[i];
                if (v.text.items.len > have) {
                    if (self.emitOk(.{ .delta = .{ .msg = @intCast(mi), .variant = @intCast(vi), .text = v.text.items[have..] } }))
                        self.tx_lens.items[i] = v.text.items.len;
                }
                if (!std.meta.eql(v.stats, self.tx_stats.items[i])) {
                    if (self.emitOk(.{ .stats = .{ .msg = @intCast(mi), .variant = @intCast(vi), .stats = v.stats } }))
                        self.tx_stats.items[i] = v.stats;
                }
            };
        }
        if (self.llm_was_busy and !busy and msgs.len > 0) {
            const last = &msgs[msgs.len - 1];
            if (last.role == .assistant) self.emit(.{ .turn_end = .{ .msg = @intCast(msgs.len - 1), .variant = @intCast(last.cur) } });
        }
        self.llm_was_busy = busy;
    }

    /// False when the transcript did not reach the outbox, so the shape memo
    /// stays behind and the next pass sends it again.
    fn emitWholeTranscript(self: *Host, msgs: []chat.Message) bool {
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        self.tx_lens.clearRetainingCapacity();
        self.tx_stats.clearRetainingCapacity();
        const out = a.alloc(wire.Message, msgs.len) catch return false;
        for (msgs, out) |*m, *wm| {
            const vs = a.alloc(wire.Variant, m.variants.items.len) catch return false;
            for (m.variants.items, vs) |*v, *wv| {
                wv.* = .{
                    .text = v.text.items,
                    .thought_primed = v.thought_primed,
                    .reason_open = v.reason_open,
                    .reason_close = v.reason_close,
                    .gen_model = v.gen_model,
                    .stats = v.stats,
                    .images = v.images.items,
                };
                self.tx_lens.append(self.gpa, v.text.items.len) catch {};
                self.tx_stats.append(self.gpa, v.stats) catch {};
            }
            const att = a.alloc(wire.ImageId, m.attachments.items.len) catch return false;
            for (m.attachments.items, att) |gi, *id| id.* = gi.id;
            wm.* = .{
                .role = if (m.role == .user) .user else .assistant,
                .synthetic = m.synthetic,
                .variants = vs,
                .cur = @intCast(m.cur),
                .attachments = att,
            };
        }
        return self.emitOk(.{ .transcript = .{ .messages = out } });
    }

    fn imageInfo(d: ?*diffuser.Diffuser, gi: *const diffuser.GenImage) wire.ImageInfo {
        const st = gi.get();
        return .{
            .id = gi.id,
            .client_ref = gi.client_ref,
            .status = st,
            .step = gi.step.load(.monotonic),
            .total = gi.total.load(.monotonic),
            .width = @intCast(gi.width),
            .height = @intCast(gi.height),
            .req_width = @intCast(gi.req_width),
            .req_height = @intCast(gi.req_height),
            .req_steps = @intCast(gi.req_steps),
            .req_cfg = gi.req_cfg,
            .req_seed = gi.req_seed,
            .prompt = gi.prompt,
            .negative = gi.req_negative,
            .params = gi.params orelse .{},
            .from_studio = gi.from_studio,
            .failure = if (gi.failure()) |e| @errorName(e) else "",
            .preview_w = gi.preview_w.load(.acquire),
            .preview_h = gi.preview_h.load(.acquire),
            .preview_rev = gi.preview_rev.load(.acquire),
            .pixels_rev = if (st == .done and gi.rgba != null) 1 else 0,
            .start_ns = gi.start_ns.load(.acquire),
            .first_step_ns = gi.first_step_ns.load(.acquire),
            .last_step_ns = gi.last_step_ns.load(.acquire),
            .done_ns = gi.done_ns.load(.acquire),
            .family = if (d) |dd| (if (dd.loadedFamily()) |f| @tagName(f) else "") else "",
            .model_stem = if (gi.model) |m| diffuser.modelStem(m.dit_path) else "",
            .clip1_stem = gi.meta.clip1,
            .clip2_stem = gi.meta.clip2,
            .vae_stem = gi.meta.vae,
            .model_hash = gi.meta.model_hash,
            .vae_hash = gi.meta.vae_hash,
            .weight_dtype = gi.meta.weight_dtype,
            .shift = gi.meta.shift,
            .loras = gi.meta.loras,
        };
    }

    fn emitImageIfChanged(self: *Host, d: ?*diffuser.Diffuser, gi: *const diffuser.GenImage) void {
        const p: ImgPrint = .{
            .status = gi.get(),
            .step = gi.step.load(.monotonic),
            .total = gi.total.load(.monotonic),
            .preview_rev = gi.preview_rev.load(.acquire),
            .has_pixels = gi.rgba != null,
            .failure = gi.gen_error.load(.acquire),
        };
        if (self.img_seen.get(gi.id)) |seen| if (std.meta.eql(seen, p)) return;
        // Memo after the send, never before: a dropped frame with the memo
        // already moved leaves this image's row wrong in the mirror until its
        // step count happens to change again.
        if (!self.emitOk(.{ .img = imageInfo(d, gi) })) return;
        self.img_seen.put(self.gpa, gi.id, p) catch {};
    }

    /// Every image a client can name: the engine's list, and the session's
    /// attachments (whose pixels a client fetches the same way).
    fn emitImages(self: *Host) void {
        const d: ?*diffuser.Diffuser = if (self.drv.diffuser) |*dd| dd else null;
        if (d == null and self.had_diffuser) {
            self.had_diffuser = false;
            self.img_seen.clearRetainingCapacity();
            self.emit(.{ .queue = .{ .images = &.{} } });
        }
        if (d) |dd| {
            self.had_diffuser = true;
            for (dd.items()) |gi| self.emitImageIfChanged(dd, gi);
        }
        if (self.drv.uiSession()) |s| {
            for (s.pendingAttachments()) |gi| self.emitImageIfChanged(d, gi);
            for (s.messages.items) |*m| for (m.attachments.items) |gi| self.emitImageIfChanged(d, gi);
        }
    }

    fn emitTelemetry(self: *Host) void {
        const now = Io.Clock.real.now(self.io).nanoseconds;
        const diff_busy = if (self.drv.diffuser) |*d| d.busyNow() else false;
        const period = if (diff_busy) telemetry_busy_ns else telemetry_idle_ns;
        if (now - self.last_telemetry_ns < period) return;
        self.last_telemetry_ns = now;

        var t: wire.Telemetry = .{};
        t.cpu = self.cpu_meter.sample();
        t.cpu_mhz = sysmon.cpuFreqMhz();
        if (sysmon.nvml()) |nv| {
            if (nv.query()) |g| {
                t.gpu_util = @floatFromInt(g.util);
                t.gpu_mhz = g.clock_mhz;
                t.vram_used = g.mem_used;
                t.vram_total = g.mem_total;
                t.have_gpu = true;
            }
            t.vram_proc = nv.selfUsed() orelse 0;
        }
        if (self.drv.diffuser) |*d| {
            const b = d.vramBreakdown();
            t.diff_te = b.te;
            t.diff_dit = b.dit;
            t.diff_latent = b.latent;
            t.diff_vae = b.vae;
            t.diff_off = d.offloadBytes();
        }
        if (self.drv.uiSession()) |s| {
            t.llm_used = s.be.deviceUsed();
            t.limit = s.vram_limit;
            t.ctx_tokens = s.ctxTokens();
            t.ctx_kv = s.ctxKvBytes();
            const res = s.llmResidency();
            t.layers_gpu = @intCast(res.gpu);
            t.layers_cpu = @intCast(res.cpu);
            t.llm_host = res.host_bytes;
            if (t.vram_total == 0) if (s.be.ctx.memGetInfo()) |mi| {
                t.vram_total = mi.total;
                t.vram_used = mi.total -| mi.free;
                t.have_gpu = true;
            };
        }
        // Still nothing: no NVML and no chat session, which is every host whose
        // card is not NVIDIA's. The image backend opened a device of its own and
        // can say what it holds, and the kernel says how busy we kept it.
        if (t.vram_total == 0) {
            if (self.drv.diffuser) |*d| if (d.vramInfo()) |v| {
                t.vram_total = v.total;
                t.vram_used = v.total -| v.free;
                t.vram_proc = v.proc;
                t.have_gpu = true;
            };
        }
        // Nothing has loaded a model yet, so no device is open to ask. The card
        // is still there and the meter still has to draw it: probe the heap
        // once, without creating a context.
        if (t.vram_total == 0 and self.probed_heap != 0) {
            t.vram_total = self.probed_heap;
            t.have_gpu = true;
        }
        if (!t.have_gpu or t.gpu_mhz == 0) {
            const g = self.drm_meter.sample(self.io, now);
            if (g.found) {
                t.have_gpu = t.have_gpu or t.vram_total > 0;
                if (t.gpu_util == 0) t.gpu_util = g.util;
                if (t.gpu_mhz == 0) t.gpu_mhz = g.clock_mhz;
            }
        }
        self.emit(.{ .telemetry = t });
    }
};

/// Where this host's image ids start: 39 random bits above a 24-bit counter,
/// hashed from the generation so a clock's coarse steps cannot make two hosts
/// agree. The top bit stays clear; a client's own images live there.
pub fn imageIdBase(gen: wire.HostGen) wire.ImageId {
    return std.hash.Wyhash.hash(0x6873, std.mem.asBytes(&gen)) & 0x7FFF_FFFF_FF00_0000;
}

test "image id bases leave the top bit and the counter clear, and differ per generation" {
    const a = imageIdBase(1);
    const b = imageIdBase(2);
    try std.testing.expect(a != b);
    try std.testing.expectEqual(@as(u64, 0), a & 0xFF_FFFF);
    try std.testing.expectEqual(@as(u64, 0), a >> 63);
    try std.testing.expect(a != 0);
}

/// The first object key of a JSON text, or null when it does not start with one.
fn firstKey(bytes: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < bytes.len and std.ascii.isWhitespace(bytes[i])) i += 1;
    if (i >= bytes.len or bytes[i] != '{') return null;
    i += 1;
    while (i < bytes.len and std.ascii.isWhitespace(bytes[i])) i += 1;
    if (i >= bytes.len or bytes[i] != '"') return null;
    const start = i + 1;
    const end = std.mem.indexOfScalarPos(u8, bytes, start, '"') orelse return null;
    return bytes[start..end];
}

test "model references resolve by id against the host's catalog, and a remote host takes no path" {
    const gpa = std.testing.allocator;
    var cat = try catalog.Catalog.fromEntries(gpa, &.{
        .{ .path = "/srv/models/qwen3-4b.gguf", .size = 100, .mtime_ns = 1 },
        .{ .path = "/srv/models/sd15/dreamshaper_8.safetensors", .size = 200, .mtime_ns = 1 },
    });
    defer cat.deinit();
    var id_buf: [catalog.id_text_len]u8 = undefined;
    const llm_id = cat.find("/srv/models/qwen3-4b.gguf").?.id();
    const dit_id = cat.find("/srv/models/sd15/dreamshaper_8.safetensors").?.id();
    var cfg: config.Config = .{};
    cfg.llm_model.set(catalog.idText(llm_id, &id_buf));
    cfg.diffusion_model.set(catalog.idText(dit_id, &id_buf));
    cfg.vae.set(catalog.idText(catalog.modelId("not-here", 7, 0), &id_buf));
    cfg.text_encoder.set("/home/user/models/clip_l.safetensors");
    _ = cfg.addFamilyLora("sd15", catalog.idText(llm_id, &id_buf));

    var local = cfg;
    Host.resolveModelRefs(&local, &cat, false);
    try std.testing.expectEqualStrings("/srv/models/qwen3-4b.gguf", local.llm_model.slice());
    try std.testing.expectEqualStrings("/srv/models/sd15/dreamshaper_8.safetensors", local.diffusion_model.slice());
    try std.testing.expect(local.vae.opt() == null);
    // A local host is the same disk: the client's own path stands.
    try std.testing.expectEqualStrings("/home/user/models/clip_l.safetensors", local.text_encoder.slice());
    try std.testing.expectEqualStrings("/srv/models/qwen3-4b.gguf", local.loras.slice()[0].path.slice());

    var remote = cfg;
    Host.resolveModelRefs(&remote, &cat, true);
    try std.testing.expectEqualStrings("/srv/models/qwen3-4b.gguf", remote.llm_model.slice());
    try std.testing.expect(remote.text_encoder.opt() == null);
}

// ── Host loop tests ───────────────────────────────────────────────────────────
// A Host with no model configured loads nothing and builds no image engine, so
// the loop runs on the test's own thread and every arm answers on the outbox.

const testing = std.testing;

const TestHost = struct {
    fn noop() void {}

    /// Build a model-less Host on a thread with the settings parse stack and
    /// run `body` there (the settings arm parses a big struct in Debug).
    fn run(opts: Options, comptime body: fn (*Host) anyerror!void) !void {
        const Worker = struct {
            fn go(o: Options, result: *anyerror!void) void {
                result.* = inner(o);
            }
            fn inner(o: Options) anyerror!void {
                const gpa = testing.allocator;
                const cfg = try gpa.create(config.Config);
                defer gpa.destroy(cfg);
                cfg.* = .{};
                const h = try gpa.create(Host);
                defer gpa.destroy(h);
                var o2 = o;
                o2.probe_card = false;
                h.init(gpa, testing.io, cfg, o2, noop);
                defer h.deinit();
                try body(h);
            }
        };
        var result: anyerror!void = {};
        const t = try std.Thread.spawn(.{ .stack_size = config.parse_stack_size }, Worker.go, .{ opts, &result });
        t.join();
        return result;
    }

    /// The outbox, decoded (text frames only), into `arena`.
    fn events(h: *Host, arena: std.mem.Allocator) ![]wire.Event {
        var frames: std.ArrayList(Frame) = .empty;
        defer frames.deinit(h.gpa);
        h.take(&frames);
        var evs: std.ArrayList(wire.Event) = .empty;
        for (frames.items) |f| {
            defer f.deinit(h.gpa);
            if (f == .text) try evs.append(arena, try wire.decode(wire.Event, arena, f.text));
        }
        return evs.toOwnedSlice(arena);
    }

    const Err = struct { code: wire.ErrCode, text: []const u8 };

    fn firstErr(evs: []const wire.Event) ?Err {
        for (evs) |e| if (e == .err) return .{ .code = e.err.code, .text = e.err.text };
        return null;
    }

    /// Post `req`, run one pass, and return the first error event, if any.
    fn ask(h: *Host, req: wire.Request) !?Err {
        var arena = std.heap.ArenaAllocator.init(h.gpa);
        defer arena.deinit();
        _ = try events(h, arena.allocator()); // drop what an earlier pass left
        h.postRequest(req);
        h.runOnce();
        const evs = try events(h, arena.allocator());
        const e = firstErr(evs) orelse return null;
        return .{ .code = e.code, .text = try h.gpa.dupe(u8, e.text) };
    }

    fn expectErr(h: *Host, req: wire.Request, code: wire.ErrCode, text: []const u8) !void {
        const e = (try ask(h, req)) orelse {
            std.debug.print("expected {t} '{s}', got no error\n", .{ code, text });
            return error.TestExpectedError;
        };
        defer h.gpa.free(e.text);
        errdefer std.debug.print("expected {t} '{s}', got {t} '{s}'\n", .{ code, text, e.code, e.text });
        try testing.expectEqual(code, e.code);
        try testing.expectEqualStrings(text, e.text);
    }

    fn expectOk(h: *Host, req: wire.Request) !void {
        if (try ask(h, req)) |e| {
            defer h.gpa.free(e.text);
            std.debug.print("expected no error, got {t} '{s}'\n", .{ e.code, e.text });
            return error.TestUnexpectedError;
        }
    }

    fn pushSettings(h: *Host, cfg: *const config.Config) !void {
        const json = try wire.encodeAlloc(h.gpa, config.hostSettings(cfg));
        defer h.gpa.free(json);
        h.postRequest(.{ .settings = .{ .json = json } });
    }

    fn queuedCancels(h: *Host) usize {
        var n: usize = 0;
        for (&h.drv.urgent.img_cancel) |*s| n += @intFromBool(s.load(.acquire) != 0);
        return n;
    }
};

test "image cancels posted in one pass all reach the urgent ring, and a full ring falls through to the inbox" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            for (11..19) |id| h.postRequest(.{ .img_cancel = .{ .image = id } });
            try testing.expectEqual(@as(usize, 8), TestHost.queuedCancels(h));
            try testing.expectEqual(@as(usize, 0), h.inbox.items.items.len);
            h.postRequest(.{ .img_cancel = .{ .image = 19 } });
            try testing.expectEqual(@as(usize, 1), h.inbox.items.items.len);
            h.runOnce();
            try testing.expectEqual(@as(usize, 0), TestHost.queuedCancels(h));
            try testing.expectEqual(@as(usize, 0), h.inbox.items.items.len);
            var arena = std.heap.ArenaAllocator.init(h.gpa);
            defer arena.deinit();
            try testing.expect(TestHost.firstErr(try TestHost.events(h, arena.allocator())) == null);
        }
    }.body);
}

test "a settings push out of range is refused by field and leaves the settings in force alone" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            var cfg: config.Config = .{};
            cfg.steps = 30;
            try TestHost.pushSettings(h, &cfg);
            try TestHost.expectOk(h, .snapshot);
            try testing.expectEqual(@as(usize, 30), h.drv.settings.steps);

            const Case = struct { field: []const u8, apply: *const fn (*config.Config) void };
            const cases = [_]Case{
                .{ .field = "width", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.width = 1004;
                    }
                }.f },
                .{ .field = "height", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.height = 16384;
                    }
                }.f },
                .{ .field = "steps", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.steps = 501;
                    }
                }.f },
                .{ .field = "max_new_tokens", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.max_new_tokens = 1 << 21;
                    }
                }.f },
                .{ .field = "top_p", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.sampling.top_p = 1.5;
                    }
                }.f },
                .{ .field = "temperature", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.sampling.temperature = -1;
                    }
                }.f },
                .{ .field = "repeat_penalty", .apply = &struct {
                    fn f(c: *config.Config) void {
                        c.sampling.repeat_penalty = 0;
                    }
                }.f },
            };
            for (cases) |case| {
                var bad: config.Config = .{};
                bad.steps = 7;
                case.apply(&bad);
                try TestHost.pushSettings(h, &bad);
                try TestHost.expectErr(h, .snapshot, .bad_request, case.field);
                try testing.expectEqual(@as(usize, 30), h.drv.settings.steps);
            }
        }
    }.body);
}

test "a render request out of range is refused before the model is looked for" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .steps = 0 } }, .bad_request, "steps");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .width = 1004 } }, .bad_request, "width");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .height = 32 } }, .bad_request, "height");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .width = 8192, .height = 8200 } }, .bad_request, "height");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .cfg = std.math.nan(f32) } }, .bad_request, "cfg");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = "" }} } }, .bad_request, "lora path");
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = "/l.safetensors", .strength = 50 }} } }, .bad_request, "lora strength");
            // The largest allowed canvas passes the bounds and meets the missing model.
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .width = 8192, .height = 8192, .steps = 500 } }, .no_model, "no image model");
        }
    }.body);
}

test "a remote host refuses a lora named by path" {
    try TestHost.run(.{ .folders = .{ .dirs = &.{}, .files = &.{} } }, struct {
        fn body(h: *Host) !void {
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = "/home/u/lora.safetensors" }} } }, .bad_request, "lora");
        }
    }.body);
}

test "a lora named by id resolves against the host's catalog, or is not found" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            scan.setCanned(try catalog.Catalog.fromEntries(h.gpa, &.{
                .{ .path = "/srv/models/lora.safetensors", .size = 5, .mtime_ns = 1 },
            }));
            var id_buf: [catalog.id_text_len]u8 = undefined;
            const held = catalog.idText(scan.cat.find("/srv/models/lora.safetensors").?.id(), &id_buf);
            // Resolved: the request gets past the lora step to the missing model.
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = held }} } }, .no_model, "no image model");
            var other_buf: [catalog.id_text_len]u8 = undefined;
            const missing = catalog.idText(catalog.modelId("nope", 1, 0), &other_buf);
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = missing }} } }, .not_found, "lora");
            // A local host is the same disk: a plain path stands.
            try TestHost.expectErr(h, .{ .img_enqueue = .{ .prompt = "x", .loras = &.{.{ .path = "/home/u/lora.safetensors" }} } }, .no_model, "no image model");
        }
    }.body);
}

test "an ack drops the image it names, and one for an unknown id is a no-op" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            // No client has acked anything yet, so an id nobody minted must not
            // be an error the client has to handle.
            try TestHost.expectOk(h, .{ .img_ack = .{ .image = 4242 } });

            h.drv.diffuser = diffuser.Diffuser.init(h.gpa, h.io, TestHost.noop, .{ .dit_path = "" }, diffuser.VramCoordinator.none);
            const d = &h.drv.diffuser.?;
            const gi = try h.gpa.create(diffuser.GenImage);
            gi.* = .{ .prompt = try h.gpa.dupe(u8, "p"), .wake = TestHost.noop, .io = h.io };
            gi.status = .init(@intFromEnum(diffuser.GenStatus.done));
            try d.enqueue(gi);
            const id = gi.id;

            // One pass so the host emits the image and remembers its print.
            try TestHost.expectOk(h, .snapshot);
            try testing.expect(h.img_seen.contains(id));

            try TestHost.expectOk(h, .{ .img_ack = .{ .image = id } });
            try testing.expectEqual(@as(?*diffuser.GenImage, null), d.byId(id));
            try testing.expectEqual(@as(usize, 0), d.items().len);
            // The print goes with it; the id is never minted again.
            try testing.expect(!h.img_seen.contains(id));

            // A snapshot after the drop lists the shorter queue.
            var arena = std.heap.ArenaAllocator.init(h.gpa);
            defer arena.deinit();
            h.postRequest(.snapshot);
            h.runOnce();
            for (try TestHost.events(h, arena.allocator())) |e| {
                if (e == .queue) try testing.expectEqual(@as(usize, 0), e.queue.images.len);
                try testing.expect(e != .img);
            }
        }
    }.body);
}

test "an adopted transcript carries its attachment ids and sends them back" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            try TestHost.expectOk(h, .{ .chat_adopt = .{ .messages = &.{
                .{ .role = .user, .variants = &.{.{ .text = "look" }}, .attachments = &.{ 7, 8 } },
                .{ .role = .assistant, .variants = &.{.{ .text = "ok" }} },
            } } });
            {
                h.drv.carry_mu.lockUncancelable(h.io);
                defer h.drv.carry_mu.unlock(h.io);
                const c = h.drv.carry orelse return error.NothingCarried;
                try testing.expectEqual(@as(usize, 2), c.items.len);
                const att = c.items[0].attachments.items;
                try testing.expectEqual(@as(usize, 2), att.len);
                try testing.expectEqual(@as(wire.ImageId, 7), att[0].id);
                try testing.expectEqual(@as(wire.ImageId, 8), att[1].id);
                try testing.expectEqual(wire.ImageStatus.done, att[0].get());
                try testing.expect(att[0].rgba == null);
                try testing.expectEqual(@as(usize, 0), c.items[1].attachments.items.len);
            }
            // The whole transcript went out in that pass; a snapshot sends it again.
            var arena = std.heap.ArenaAllocator.init(h.gpa);
            defer arena.deinit();
            h.postRequest(.snapshot);
            h.runOnce();
            const evs = try TestHost.events(h, arena.allocator());
            for (evs) |e| if (e == .transcript) {
                try testing.expectEqualSlices(wire.ImageId, &.{ 7, 8 }, e.transcript.messages[0].attachments);
                return;
            };
            return error.NoTranscriptEvent;
        }
    }.body);
}

// The render a tool call asked for runs wherever the client placed it, so the
// only way the transcript learns its id is this request coming back.
test "the image a tool call produced lands on the variant that asked, and a stale index does not" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            try TestHost.expectOk(h, .{ .chat_adopt = .{ .messages = &.{
                .{ .role = .user, .variants = &.{.{ .text = "draw a fox" }} },
                .{ .role = .assistant, .cur = 1, .variants = &.{
                    .{ .text = "how about this" },
                    .{ .text = "<image>a red fox in snow</image>" },
                } },
            } } });
            // Nothing the transcript has: no record, and no crash.
            try TestHost.expectOk(h, .{ .chat_image = .{ .msg = 9, .variant = 0, .image = 55 } });
            try TestHost.expectOk(h, .{ .chat_image = .{ .msg = 1, .variant = 7, .image = 55 } });
            try TestHost.expectOk(h, .{ .chat_image = .{ .msg = 1, .variant = 1, .image = 0 } });

            var arena = std.heap.ArenaAllocator.init(h.gpa);
            defer arena.deinit();
            h.postRequest(.{ .chat_image = .{ .msg = 1, .variant = 1, .image = 77 } });
            h.runOnce();
            {
                h.drv.carry_mu.lockUncancelable(h.io);
                defer h.drv.carry_mu.unlock(h.io);
                const c = h.drv.carry orelse return error.NothingCarried;
                try testing.expectEqual(@as(usize, 0), c.items[1].variants.items[0].images.items.len);
                try testing.expectEqualSlices(wire.ImageId, &.{77}, c.items[1].variants.items[1].images.items);
            }
            // The append moved the transcript's shape, so the client gets it.
            var saw_transcript = false;
            for (try TestHost.events(h, arena.allocator())) |e| if (e == .transcript) {
                try testing.expectEqualSlices(wire.ImageId, &.{77}, e.transcript.messages[1].variants[1].images);
                saw_transcript = true;
            };
            if (!saw_transcript) return error.NoTranscriptEvent;

            // The render failed and the client put it to another host: the
            // transcript follows it instead of collecting both attempts. Saying
            // the same thing twice changes nothing either way.
            h.postRequest(.{ .chat_image = .{ .msg = 1, .variant = 1, .image = 78, .replaces = 77 } });
            h.postRequest(.{ .chat_image = .{ .msg = 1, .variant = 1, .image = 78, .replaces = 77 } });
            h.runOnce();
            h.drv.carry_mu.lockUncancelable(h.io);
            defer h.drv.carry_mu.unlock(h.io);
            const c2 = h.drv.carry orelse return error.NothingCarried;
            try testing.expectEqualSlices(wire.ImageId, &.{78}, c2.items[1].variants.items[1].images.items);
        }
    }.body);
}

test "a note for the model with nothing resident is dropped, not an error" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            try TestHost.expectOk(h, .{ .chat_note = .{ .text = "[image tool] failed: out of VRAM" } });
            try TestHost.expectOk(h, .{ .chat_note = .{} });
            try testing.expect(h.drv.carry == null);
        }
    }.body);
}

test "a new chat or an adopt that arrives during a load lands when the load ends" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            const d = &h.drv;
            d.loading.store(true, .release);
            try TestHost.expectOk(h, .{ .chat_adopt = .{ .messages = &.{
                .{ .role = .user, .variants = &.{.{ .text = "hi" }} },
            } } });
            try testing.expect(d.carry == null);
            try testing.expect(d.pending_transcript != null and d.pending_transcript.? == .adopt);
            d.loading.store(false, .release);
            h.runOnce();
            try testing.expect(d.pending_transcript == null);
            try testing.expectEqual(@as(usize, 1), (d.carry orelse return error.NothingCarried).items.len);

            d.loading.store(true, .release);
            try TestHost.expectOk(h, .chat_new);
            try testing.expect(d.carry != null);
            try testing.expect(d.pending_transcript != null and d.pending_transcript.? == .new);
            d.loading.store(false, .release);
            h.runOnce();
            try testing.expect(d.pending_transcript == null);
            try testing.expect(d.carry == null);
        }
    }.body);
}

test "clearOutbox drops everything queued" {
    try TestHost.run(.{}, struct {
        fn body(h: *Host) !void {
            h.postRequest(.{ .hello = .{} });
            h.runOnce();
            try testing.expect(h.outbox.items.items.len > 0);
            h.clearOutbox();
            try testing.expectEqual(@as(usize, 0), h.outbox.items.items.len);
            try testing.expectEqual(@as(usize, 0), h.outbox.bytes);
            var frames: std.ArrayList(Frame) = .empty;
            defer frames.deinit(h.gpa);
            h.take(&frames);
            try testing.expectEqual(@as(usize, 0), frames.items.len);
        }
    }.body);
}

test "the urgent verbs are told apart by their first key" {
    try std.testing.expectEqualStrings("chat_cancel", firstKey("{\"chat_cancel\":{}}").?);
    try std.testing.expectEqualStrings("img_cancel", firstKey(" { \"img_cancel\" : {\"image\":3}}").?);
    try std.testing.expectEqual(@as(?[]const u8, null), firstKey("[1]"));
    try std.testing.expectEqual(@as(?[]const u8, null), firstKey(""));
}

/// RGBA box filter by an integer factor. `dst` is `ow*oh*4`; source pixels
/// past `ow*k` / `oh*k` are dropped.
fn boxDown(src: []const u8, sw: u32, dst: []u8, ow: u32, oh: u32, k: u32) void {
    if (k == 1) {
        @memcpy(dst, src[0..dst.len]);
        return;
    }
    const n: u32 = k * k;
    var y: u32 = 0;
    while (y < oh) : (y += 1) {
        var x: u32 = 0;
        while (x < ow) : (x += 1) {
            var acc: [4]u32 = .{ 0, 0, 0, 0 };
            var dy: u32 = 0;
            while (dy < k) : (dy += 1) {
                const row = (@as(usize, y) * k + dy) * sw;
                var dx: u32 = 0;
                while (dx < k) : (dx += 1) {
                    const p = (row + @as(usize, x) * k + dx) * 4;
                    acc[0] += src[p];
                    acc[1] += src[p + 1];
                    acc[2] += src[p + 2];
                    acc[3] += src[p + 3];
                }
            }
            const o = (@as(usize, y) * ow + x) * 4;
            for (0..4) |c| dst[o + c] = @intCast(acc[c] / n);
        }
    }
}

test "box filter averages each k×k block" {
    // 4×2 source, k=2 -> 2×1
    const src = [_]u8{
        10, 0, 0, 255, 30, 0, 0, 255, 100, 0, 0, 255, 100, 0, 0, 255,
        20, 0, 0, 255, 40, 0, 0, 255, 200, 0, 0, 255, 200, 0, 0, 255,
    };
    var dst: [8]u8 = undefined;
    boxDown(&src, 4, &dst, 2, 1, 2);
    try std.testing.expectEqualSlices(u8, &.{ 25, 0, 0, 255, 150, 0, 0, 255 }, &dst);
}
