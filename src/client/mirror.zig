//! What a client holds of one host, built only from the frames the host sent:
//! the transcript, the image list with whatever pixels have been fetched, the
//! state block and the latest telemetry. One writer, the client's frame
//! thread; the views read it and never touch an engine.
//!
//! Pixels are pulled, never pushed: `pollFetches` asks for a preview when the
//! host reports a newer revision than the one held, at most one outstanding
//! and at most every half second per image, and for a finished image's pixels
//! once. `linkDown` and `hostRestarted` are the two ways a host goes away; the
//! owner tells them apart by the generation the reconnect reports.
const std = @import("std");
const wire = @import("serve").wire;
const catalog = @import("shared").catalog;

const log = std.log.scoped(.mirror);

pub const ImageId = wire.ImageId;
pub const Frame = @import("serve").queue.Frame;

/// The top bit marks a number this client minted rather than a host
/// (`host.imageIdBase` leaves it clear). Under it are TWO spaces that must not
/// meet, since a view puts both in one namespace: images this client owns, and
/// the references it mints for jobs no host has taken.
pub const client_bit: ImageId = 1 << 63;
const space_bit: ImageId = 1 << 62;
pub const local_image_base: ImageId = client_bit | space_bit;
pub const queue_ref_base: ImageId = client_bit;

/// A number in the queue-reference space rather than the local-image one.
pub fn isQueueRef(id: ImageId) bool {
    return id & client_bit != 0 and id & space_bit == 0;
}

/// Ids for images a client builds itself. One counter across every mirror, so
/// two hosts disowning their images cannot mint the same id; the client touches
/// its mirrors from one thread.
var g_next_local: ImageId = local_image_base;

fn mintLocal() ImageId {
    const id = g_next_local;
    g_next_local += 1;
    return id;
}

pub const Variant = struct {
    text: std.ArrayList(u8) = .empty,
    thought_primed: bool = false,
    reason_open: []u8 = "",
    reason_close: []u8 = "",
    gen_model: []u8 = "",
    stats: wire.TurnStats = .{},
    images: std.ArrayList(ImageId) = .empty,

    fn deinit(self: *Variant, gpa: std.mem.Allocator) void {
        self.text.deinit(gpa);
        if (self.reason_open.len > 0) gpa.free(self.reason_open);
        if (self.reason_close.len > 0) gpa.free(self.reason_close);
        if (self.gen_model.len > 0) gpa.free(self.gen_model);
        self.images.deinit(gpa);
    }
};

pub const Message = struct {
    role: wire.Role = .user,
    synthetic: bool = false,
    variants: std.ArrayList(Variant) = .empty,
    cur: usize = 0,
    attachments: std.ArrayList(ImageId) = .empty,

    pub fn active(self: *const Message) *const Variant {
        return &self.variants.items[@min(self.cur, self.variants.items.len - 1)];
    }

    fn deinit(self: *Message, gpa: std.mem.Allocator) void {
        for (self.variants.items) |*v| v.deinit(gpa);
        self.variants.deinit(gpa);
        self.attachments.deinit(gpa);
    }
};

pub const Image = struct {
    info: wire.ImageInfo,
    /// Live preview as last fetched (RGBA), and the revision it is.
    preview: ?[]u8 = null,
    preview_w: u32 = 0,
    preview_h: u32 = 0,
    preview_rev: u32 = 0,
    preview_outstanding: bool = false,
    preview_next_ns: i96 = 0,
    /// Finished pixels (RGBA, `info.width` × `info.height`). Dropped by
    /// `evictPixels` once the file is on disk; `preview` stays as the thumbnail.
    pixels: ?[]u8 = null,
    pixels_requested: bool = false,
    /// Bumped when `preview` or `pixels` changed, so a view can invalidate its
    /// texture exactly once per new picture.
    rev: u32 = 0,
    /// Where the client wrote this render, once it did. gpa-owned. A saved
    /// conversation stores it so a reload shows the picture instead of making
    /// it again.
    saved_path: ?[]u8 = null,
    /// Set once a save was attempted, so a failure is not retried every frame.
    save_tried: bool = false,
    /// Set once the user has been told this render failed, so one failure is
    /// one notice and one replay rather than one per frame.
    failure_told: bool = false,
    /// True for an image this client built itself (a saved render reopened
    /// from disk); the host never had it.
    local: bool = false,
    /// The file this picture lives in will not open. Carried alongside the
    /// failed status rather than as one, because nothing about it is a render
    /// that went wrong: there is nothing to report, nothing queued, and nothing
    /// to try again. The request that made it lived in the PNG, so it cannot be
    /// made a second time either.
    missing: bool = false,
    /// This client has everything it will get for this image and has told the
    /// host so, which is when the host frees it. It stays here, and a later
    /// snapshot that does not list it must not take it away.
    acked: bool = false,
    /// This host has been told to stop rendering it, because the client put
    /// that render somewhere else. One cancel, not one per frame.
    cancel_sent: bool = false,

    fn deinit(self: *Image, gpa: std.mem.Allocator) void {
        freeInfo(gpa, &self.info);
        if (self.preview) |p| gpa.free(p);
        if (self.pixels) |p| gpa.free(p);
        if (self.saved_path) |p| gpa.free(p);
    }

    pub fn status(self: *const Image) wire.ImageStatus {
        return self.info.status;
    }

    /// Finished, with the picture still on its way here.
    pub fn receiving(self: *const Image) bool {
        return self.info.status == .done and self.pixels == null;
    }

    /// Full pixels if here, else the last preview: what a thumbnail draws.
    pub fn thumb(self: *const Image) ?struct { px: []const u8, w: u32, h: u32 } {
        if (self.pixels) |p| return .{ .px = p, .w = self.info.width, .h = self.info.height };
        if (self.preview) |p| {
            if (self.preview_w > 0 and self.preview_h > 0) return .{ .px = p, .w = self.preview_w, .h = self.preview_h };
        }
        return null;
    }

    /// The full picture can be read back from disk.
    pub fn restorable(self: *const Image) bool {
        return self.pixels == null and self.info.status == .done and self.saved_path != null;
    }
};

fn dupeStr(gpa: std.mem.Allocator, s: []const u8) []const u8 {
    if (s.len == 0) return "";
    return gpa.dupe(u8, s) catch "";
}

fn freeStr(gpa: std.mem.Allocator, s: []const u8) void {
    if (s.len > 0) gpa.free(s);
}

/// Copies every string of an `ImageInfo` so it outlives the frame it came in.
fn dupeInfo(gpa: std.mem.Allocator, i: wire.ImageInfo) wire.ImageInfo {
    var out = i;
    out.prompt = dupeStr(gpa, i.prompt);
    out.negative = dupeStr(gpa, i.negative);
    out.failure = dupeStr(gpa, i.failure);
    out.family = dupeStr(gpa, i.family);
    out.model_stem = dupeStr(gpa, i.model_stem);
    return out;
}

fn freeInfo(gpa: std.mem.Allocator, i: *wire.ImageInfo) void {
    freeStr(gpa, i.prompt);
    freeStr(gpa, i.negative);
    freeStr(gpa, i.failure);
    freeStr(gpa, i.family);
    freeStr(gpa, i.model_stem);
    i.* = .{};
}

/// A short human explanation of a generation failure, from the error NAME the
/// host reported. The recognized cases are the ones a user can ACT on;
/// everything else falls back to the name, which is at least specific enough
/// to search for. VRAM exhaustion reaches here under four different names.
pub fn failureText(name: []const u8) []const u8 {
    const T = struct { []const u8, []const u8 };
    const table = [_]T{
        .{ "OutOfMemory", "out of memory" },
        .{ "DeviceOutOfMemory", "out of VRAM" },
        .{ "CudaError", "out of VRAM" },
        .{ "CublasLtError", "out of VRAM" },
        .{ "CudnnError", "out of VRAM" },
        .{ "GpuDecodeNonFinite", "the GPU decode produced invalid pixels" },
        .{ "FileNotFound", "a model file is missing" },
        .{ "UnknownArchitecture", "the checkpoint's architecture is not recognized" },
        .{ "UnsupportedCheckpoint", "this checkpoint cannot run on the selected backend" },
        .{ "ComponentNotInCheckpoint", "the checkpoint is missing a component (VAE or text encoder)" },
        .{ "SavedImageMissing", "the saved image file is gone" },
        .{ "IncompleteMetadata", "the checkpoint file is corrupt" },
    };
    for (table) |row| if (std.mem.eql(u8, row[0], name)) return row[1];
    return name;
}

/// What the ‹ (back) / › (next) buttons on the last assistant response do,
/// carousel-style: back/next step through the existing variants; next pressed
/// on the NEWEST variant means "regenerate a fresh one". Back on the first
/// variant does nothing (the UI disables it). The host applies the
/// result; this only decides what to ask for.
pub const Nav = union(enum) { none, select: usize, regenerate };
pub fn navTarget(cur: usize, n_variants: usize, dir: enum { back, next }) Nav {
    return switch (dir) {
        .back => if (cur == 0) .none else .{ .select = cur - 1 },
        .next => if (cur + 1 < n_variants) .{ .select = cur + 1 } else .regenerate,
    };
}

/// Is there nothing further this image can get from its host: it is over, and
/// either its pixels are here or the host has none to send.
fn holdsAllOf(im: *const Image) bool {
    return switch (im.info.status) {
        .failed, .canceled => true,
        .done => im.pixels != null or im.info.pixels_rev == 0,
        .pending, .generating, .suspended => false,
    };
}

/// One render the model asked for, with the strings owned here so it outlives
/// the frame it arrived in.
pub const Call = struct {
    msg: u32,
    variant: u32,
    /// Which call it is within that reply.
    index: u32,
    req: wire.ImageRequest,

    pub fn deinit(self: *Call, gpa: std.mem.Allocator) void {
        freeStr(gpa, self.req.prompt);
        freeStr(gpa, self.req.negative);
    }
};

pub const HostErr = struct { code: wire.ErrCode, text: []u8 };
pub const HostNotice = struct { tone: wire.Tone, text: []u8 };
pub const PeakUpdate = struct { peak: u64, key: u64 };

pub const Mirror = struct {
    gpa: std.mem.Allocator,
    messages: std.ArrayList(Message) = .empty,
    /// Creation order, like the host's list.
    images: std.ArrayList(Image) = .empty,
    state: wire.State = .{},
    telemetry: wire.Telemetry = .{},
    /// Count of `turn_end` events, so a consumer can act once per finished turn.
    turns_ended: u64 = 0,
    /// The last error the host reported, until the consumer takes it.
    last_err: ?HostErr = null,
    /// The last informational notice, until the consumer takes it. One slot, as
    /// for `last_err`: the notices this carries are seconds apart at least.
    last_notice: ?HostNotice = null,
    /// A grown diffusion peak to persist, until the consumer takes it.
    diff_peak: ?PeakUpdate = null,
    /// Longest side a preview is fetched at (0 = full size). A view sets it to
    /// what it draws, so a 200 px tile does not move a 4 MiB frame.
    preview_max_edge: u32 = 0,
    /// Bumped on every structural transcript change (a `transcript` event), so
    /// a view holding indices knows to drop them.
    transcript_rev: u64 = 0,
    /// Bumped per `telemetry` event, so a meter pushes its history once per sample.
    telemetry_seq: u64 = 0,
    /// The host's model catalog as last sent, and its scan status. Kept across
    /// a host restart: the files are the same until the new host says otherwise.
    catalog: catalog.Catalog,
    catalog_rev: u64 = 0,
    /// A `state` event has arrived (see `spokeOnce`).
    state_seen: bool = false,
    /// Bumped whenever `catalog` was replaced, so a consumer re-resolves once.
    catalog_seq: u64 = 0,
    /// Renders the model asked for and nobody has placed yet. The host parses
    /// its own tool calls but queues none of them; the owner drains this and
    /// puts each through the same queue a user's render goes through.
    asked_for: std.ArrayList(Call) = .empty,
    scanning: bool = false,
    scan_report: catalog.ScanReport = .{},

    pub fn init(gpa: std.mem.Allocator) Mirror {
        return .{ .gpa = gpa, .catalog = catalog.Catalog.init(gpa) };
    }

    pub fn deinit(self: *Mirror) void {
        self.clear();
        self.dropCalls();
        self.asked_for.deinit(self.gpa);
        self.messages.deinit(self.gpa);
        self.images.deinit(self.gpa);
        self.catalog.deinit();
    }

    /// This host's entries have arrived at least once, so an empty catalog
    /// means "nothing there" rather than "not looked yet".
    ///
    /// Keyed on having RECEIVED a list, not on the host's `rev`: a daemon that
    /// loaded its index from cache and has not finished a scan in this process
    /// sends its whole catalog with `rev` still 0, and reading that as "never
    /// looked" makes a host that plainly has files look unknown.
    pub fn scannedOnce(self: *const Mirror) bool {
        return self.catalog_seq > 0;
    }

    /// This host has described itself at least once. Until it has, every field
    /// of `state` is a DEFAULT, and a default reads exactly like a host that
    /// has no image engine and nothing loaded: saying so is a guess presented
    /// as a fact.
    pub fn spokeOnce(self: *const Mirror) bool {
        return self.state_seen;
    }

    /// Take the renders the model has asked for since the last call. The
    /// caller owns the strings and frees them.
    pub fn takeCalls(self: *Mirror, out: *std.ArrayList(Call)) void {
        out.clearRetainingCapacity();
        out.appendSlice(self.gpa, self.asked_for.items) catch return;
        self.asked_for.clearRetainingCapacity();
    }

    /// Drop what the model asked for and nobody placed: the conversation it
    /// belonged to is being left.
    pub fn dropCalls(self: *Mirror) void {
        for (self.asked_for.items) |*c| c.deinit(self.gpa);
        self.asked_for.clearRetainingCapacity();
    }

    /// Forget the host entirely (it restarted, or the link is gone).
    pub fn clear(self: *Mirror) void {
        for (self.messages.items) |*m| m.deinit(self.gpa);
        self.messages.clearRetainingCapacity();
        for (self.images.items) |*im| im.deinit(self.gpa);
        self.images.clearRetainingCapacity();
        self.freeState();
        self.state = .{};
        self.state_seen = false;
        self.telemetry = .{};
        if (self.last_err) |e| self.gpa.free(e.text);
        self.last_err = null;
        if (self.last_notice) |n| self.gpa.free(n.text);
        self.last_notice = null;
        self.transcript_rev += 1;
    }

    fn freeState(self: *Mirror) void {
        freeStr(self.gpa, self.state.load_err);
        freeStr(self.gpa, self.state.diff_load_err);
        freeStr(self.gpa, self.state.diff_family);
        if (self.state.attachments.len > 0) self.gpa.free(self.state.attachments);
    }

    /// The link dropped. Ids are kept: if the same host answers again, its
    /// snapshot overwrites these entries by id. A render that had not
    /// finished is failed as lost so the owner can put it elsewhere.
    pub fn linkDown(self: *Mirror) void {
        const gpa = self.gpa;
        for (self.images.items) |*im| {
            if (im.local) continue;
            im.preview_outstanding = false;
            switch (im.info.status) {
                // A finished render is not lost with the link: the same daemon
                // can still hand over its pixels when it answers again.
                .done => {
                    if (im.pixels == null) im.pixels_requested = false;
                    continue;
                },
                // Over either way. Failing one of these here would toast a
                // failure for a render the user cancelled, and replay it.
                .canceled, .failed => continue,
                .pending, .generating, .suspended => {},
            }
            if (im.preview) |p| gpa.free(p);
            im.preview = null;
            im.info.preview_rev = 0;
            im.info.status = .failed;
            freeStr(gpa, im.info.failure);
            im.info.failure = dupeStr(gpa, "HostLost");
            im.info.pixels_rev = 0;
        }
        self.freeState();
        self.state = .{};
        self.state_seen = false;
        self.telemetry = .{};
    }

    /// A different host answered where this one was. Every image it minted
    /// becomes a local one under a fresh id, so the new host's ids cannot
    /// collide with them, and the transcript's references follow.
    pub fn hostRestarted(self: *Mirror) void {
        self.linkDown();
        const gpa = self.gpa;
        var remap: std.AutoHashMapUnmanaged(ImageId, ImageId) = .empty;
        defer remap.deinit(gpa);
        for (self.images.items) |*im| {
            if (im.local) continue;
            // Nobody can hand over the pixels of a render this daemon never
            // made. One already on disk is not lost: `evictPixels` leaves a
            // saved picture in exactly this state, and failing it here would
            // make `restorable` false and the file unreadable forever.
            if (im.receiving() and im.saved_path == null) {
                im.info.status = .failed;
                freeStr(gpa, im.info.failure);
                im.info.failure = dupeStr(gpa, "HostLost");
            }
            const old = im.info.id;
            im.info.id = mintLocal();
            im.local = true;
            remap.put(gpa, old, im.info.id) catch {};
        }
        for (self.messages.items) |*m| {
            for (m.attachments.items) |*id| id.* = remap.get(id.*) orelse id.*;
            for (m.variants.items) |*v| for (v.images.items) |*id| {
                id.* = remap.get(id.*) orelse id.*;
            };
        }
    }

    /// The transcript in wire form, for handing to a host (`chat_adopt`). The
    /// arrays are `arena`'s; the strings borrow the mirror's, so encode before
    /// the next `apply`.
    pub fn toWire(self: *const Mirror, arena: std.mem.Allocator) ![]wire.Message {
        const out = try arena.alloc(wire.Message, self.messages.items.len);
        for (self.messages.items, out) |*m, *wm| {
            const vs = try arena.alloc(wire.Variant, m.variants.items.len);
            for (m.variants.items, vs) |*v, *wv| wv.* = .{
                .text = v.text.items,
                .thought_primed = v.thought_primed,
                .reason_open = v.reason_open,
                .reason_close = v.reason_close,
                .gen_model = v.gen_model,
                .stats = v.stats,
                .images = v.images.items,
            };
            wm.* = .{
                .role = m.role,
                .synthetic = m.synthetic,
                .variants = vs,
                .cur = @intCast(m.cur),
                .attachments = m.attachments.items,
            };
        }
        return out;
    }

    pub fn byId(self: *Mirror, id: ImageId) ?*Image {
        if (id == 0) return null;
        for (self.images.items) |*im| if (im.info.id == id) return im;
        return null;
    }

    /// The host's entry for a render this client asked for, once it lists it.
    pub fn byClientRef(self: *Mirror, ref: u64) ?*Image {
        if (ref == 0) return null;
        for (self.images.items) |*im| if (!im.local and im.info.client_ref == ref) return im;
        return null;
    }

    /// The same render whether or not the host still owns it: after a restart
    /// its images are this client's, and the owner still has to know how each
    /// one ended.
    pub fn byRef(self: *Mirror, ref: u64) ?*Image {
        if (ref == 0) return null;
        for (self.images.items) |*im| if (im.info.client_ref == ref) return im;
        return null;
    }

    /// The image already here that was saved to `path`, if any. A finished
    /// render's file is its identity: reopening a conversation must find the
    /// image it already has rather than building a second one.
    pub fn bySavedPath(self: *Mirror, path: []const u8) ?*Image {
        if (path.len == 0) return null;
        for (self.images.items) |*im| if (im.saved_path) |p| if (std.mem.eql(u8, p, path)) return im;
        return null;
    }

    /// Add an image this client built itself: a saved render reopened from
    /// disk, finished and with its pixels (or failed, with none). `info.id` is
    /// assigned here; `saved_path` and `pixels` are taken over.
    pub fn addLocal(self: *Mirror, info_in: wire.ImageInfo, pixels: ?[]u8, saved_path: ?[]u8) ImageId {
        var info = dupeInfo(self.gpa, info_in);
        info.id = mintLocal();
        self.images.append(self.gpa, .{
            .info = info,
            .pixels = pixels,
            .saved_path = saved_path,
            .save_tried = true,
            .local = true,
        }) catch {
            freeInfo(self.gpa, &info);
            return 0;
        };
        return info.id;
    }

    /// Apply one frame. A JSON frame that does not decode is logged and dropped:
    /// a newer host may send an event this client does not know.
    pub fn apply(self: *Mirror, f: Frame) void {
        switch (f) {
            .bin => |b| self.applyBin(b.hdr, b.payload),
            .text => |t| {
                var arena = std.heap.ArenaAllocator.init(self.gpa);
                defer arena.deinit();
                const ev = wire.decode(wire.Event, arena.allocator(), t) catch |err| {
                    log.warn("event dropped: {t}", .{err});
                    return;
                };
                self.applyEvent(ev);
            },
        }
    }

    pub fn applyEvent(self: *Mirror, ev: wire.Event) void {
        const gpa = self.gpa;
        switch (ev) {
            .hello => {},
            .state => |s| {
                self.state_seen = true;
                self.freeState();
                self.state = s;
                self.state.load_err = dupeStr(gpa, s.load_err);
                self.state.diff_load_err = dupeStr(gpa, s.diff_load_err);
                self.state.diff_family = dupeStr(gpa, s.diff_family);
                self.state.attachments = if (s.attachments.len > 0) gpa.dupe(ImageId, s.attachments) catch &.{} else &.{};
            },
            .transcript => |t| {
                for (self.messages.items) |*m| m.deinit(gpa);
                self.messages.clearRetainingCapacity();
                for (t.messages) |wm| {
                    var m: Message = .{ .role = wm.role, .synthetic = wm.synthetic, .cur = wm.cur };
                    for (wm.variants) |wv| {
                        var v: Variant = .{
                            .thought_primed = wv.thought_primed,
                            .stats = wv.stats,
                        };
                        v.text.appendSlice(gpa, wv.text) catch {};
                        v.reason_open = @constCast(dupeStr(gpa, wv.reason_open));
                        v.reason_close = @constCast(dupeStr(gpa, wv.reason_close));
                        v.gen_model = @constCast(dupeStr(gpa, wv.gen_model));
                        v.images.appendSlice(gpa, wv.images) catch {};
                        m.variants.append(gpa, v) catch {
                            v.deinit(gpa);
                            break;
                        };
                    }
                    // A message always has a take, so `active` can never index nothing.
                    if (m.variants.items.len == 0) m.variants.append(gpa, .{}) catch {};
                    m.attachments.appendSlice(gpa, wm.attachments) catch {};
                    self.messages.append(gpa, m) catch {
                        m.deinit(gpa);
                        break;
                    };
                }
                self.transcript_rev += 1;
            },
            .delta => |d| if (self.variantAt(d.msg, d.variant)) |v| {
                v.text.appendSlice(gpa, d.text) catch {};
            },
            .stats => |s| if (self.variantAt(s.msg, s.variant)) |v| {
                v.stats = s.stats;
            },
            .turn_end => self.turns_ended += 1,
            .ctx => {},
            .catalog => |c| {
                self.scanning = c.scanning;
                self.scan_report = .{ .files = c.files, .probed = c.probed, .reused = c.reused, .bad_folders = c.bad_folders };
                if (c.json.len == 0) return;
                const next = catalog.Catalog.fromJson(gpa, c.json) catch |err| {
                    log.warn("catalog dropped: {t}", .{err});
                    return;
                };
                self.catalog.deinit();
                self.catalog = next;
                self.catalog_rev = c.rev;
                self.catalog_seq += 1;
            },
            .queue => |q| {
                // Merge: keep the fetched pixels of anything still listed, and
                // every local image, which the host cannot list.
                var kept: std.ArrayList(Image) = .empty;
                // Listed first, or an image the host still names would be taken
                // for kept and then added a second time as a fresh entry.
                for (q.images) |info| {
                    if (self.takeImage(info.id)) |old| {
                        var im = old;
                        freeInfo(gpa, &im.info);
                        im.info = dupeInfo(gpa, info);
                        kept.append(gpa, im) catch {};
                    } else kept.append(gpa, .{ .info = dupeInfo(gpa, info) }) catch {};
                }
                // Then what the host cannot list: this client built it, or the
                // host freed it once this client said it had it.
                var i: usize = 0;
                while (i < self.images.items.len) {
                    if (self.images.items[i].local or self.images.items[i].acked) {
                        kept.append(gpa, self.images.orderedRemove(i)) catch {};
                    } else i += 1;
                }
                for (self.images.items) |*im| im.deinit(gpa);
                self.images.deinit(gpa);
                self.images = kept;
            },
            .img => |info| {
                if (self.byId(info.id)) |im| {
                    freeInfo(gpa, &im.info);
                    im.info = dupeInfo(gpa, info);
                } else {
                    self.images.append(gpa, .{ .info = dupeInfo(gpa, info) }) catch {};
                }
            },
            .img_requested => |r| for (r.calls) |c| {
                var owned = c.req;
                owned.prompt = dupeStr(gpa, c.req.prompt);
                owned.negative = dupeStr(gpa, c.req.negative);
                // LoRAs are the rendering host's own defaults, not the model's.
                owned.loras = &.{};
                self.asked_for.append(gpa, .{ .msg = c.msg, .variant = c.variant, .index = c.index, .req = owned }) catch {
                    var tmp: Call = .{ .msg = c.msg, .variant = c.variant, .index = c.index, .req = owned };
                    tmp.deinit(gpa);
                };
            },
            .telemetry => |t| {
                self.telemetry = t;
                self.telemetry_seq += 1;
            },
            .diff_peak => |p| self.diff_peak = .{ .peak = p.peak, .key = p.key },
            .err => |e| {
                log.warn("host error {t}: {s}", .{ e.code, e.text });
                if (self.last_err) |old| gpa.free(old.text);
                self.last_err = .{ .code = e.code, .text = gpa.dupe(u8, e.text) catch "" };
            },
            .notice => |n| {
                log.info("host: {s}", .{n.text});
                if (self.last_notice) |old| gpa.free(old.text);
                self.last_notice = .{ .tone = n.tone, .text = gpa.dupe(u8, n.text) catch "" };
            },
        }
    }

    fn applyBin(self: *Mirror, hdr: wire.BinHeader, payload: []const u8) void {
        const gpa = self.gpa;
        if (hdr.magic != wire.BinHeader.magic_value) return;
        const im = self.byId(hdr.id) orelse return;
        const need = @as(u64, hdr.w) *| hdr.h *| 4;
        if (need == 0 or need > payload.len) return;
        switch (hdr.kind) {
            .preview_rgba => {
                im.preview_outstanding = false;
                const buf = gpa.dupe(u8, payload[0..@intCast(need)]) catch return;
                if (im.preview) |p| gpa.free(p);
                im.preview = buf;
                im.preview_w = hdr.w;
                im.preview_h = hdr.h;
                im.preview_rev = hdr.rev;
                im.rev += 1;
            },
            .image_rgba => {
                const buf = gpa.dupe(u8, payload[0..@intCast(need)]) catch return;
                if (im.pixels) |p| gpa.free(p);
                im.pixels = buf;
                // The preview stays as this image's thumbnail, so dropping the
                // full pixels later still leaves something to draw.
                im.rev += 1;
            },
            else => {},
        }
    }

    /// Remove and return the image with `id`, if listed.
    fn takeImage(self: *Mirror, id: ImageId) ?Image {
        for (self.images.items, 0..) |*im, i| if (im.info.id == id) return self.images.orderedRemove(i);
        return null;
    }

    fn variantAt(self: *Mirror, msg: u32, variant: u32) ?*Variant {
        if (msg >= self.messages.items.len) return null;
        const m = &self.messages.items[msg];
        if (variant >= m.variants.items.len) return null;
        return &m.variants.items[variant];
    }

    /// The pull side of pixels, and the receipt that follows it. Calls `post`
    /// with each `img_fetch` due now, and with an `img_ack` for every image
    /// this client now holds everything for: the host frees it at that point,
    /// so the ack goes out once and never before the pixels are actually here.
    pub fn pollFetches(self: *Mirror, now_ns: i96, ctx: anytype, post: fn (@TypeOf(ctx), wire.Request) void) void {
        for (self.images.items) |*im| {
            if (!im.local and !im.acked and holdsAllOf(im)) {
                im.acked = true;
                post(ctx, .{ .img_ack = .{ .image = im.info.id } });
            }
            switch (im.info.status) {
                .generating, .suspended => {
                    if (im.info.preview_rev == 0 or im.info.preview_rev == im.preview_rev) continue;
                    if (im.preview_outstanding or now_ns < im.preview_next_ns) continue;
                    im.preview_outstanding = true;
                    im.preview_next_ns = now_ns + 500 * std.time.ns_per_ms;
                    post(ctx, .{ .img_fetch = .{ .image = im.info.id, .kind = .preview, .have_rev = im.preview_rev, .max_edge = self.preview_max_edge } });
                },
                .done => {
                    if (im.info.pixels_rev == 0 or im.pixels != null or im.pixels_requested) continue;
                    im.pixels_requested = true;
                    post(ctx, .{ .img_fetch = .{ .image = im.info.id, .kind = .pixels } });
                },
                else => {},
            }
        }
    }

    /// Drop the full pixels of finished images past the newest `keep` that are
    /// on disk, keeping their thumbnails. A 1024 square is 4 MiB and a long
    /// session finishes a lot of them. Returns the bytes freed.
    pub fn evictPixels(self: *Mirror, keep: usize, pins: []const ImageId) usize {
        var held: usize = 0;
        var freed: usize = 0;
        var i = self.images.items.len;
        while (i > 0) {
            i -= 1;
            const im = &self.images.items[i];
            const px = im.pixels orelse continue;
            held += 1;
            if (held <= keep or im.saved_path == null) continue;
            if (std.mem.indexOfScalar(ImageId, pins, im.info.id) != null) continue;
            freed += px.len;
            self.gpa.free(px);
            im.pixels = null;
            im.rev += 1;
        }
        return freed;
    }

    /// Drop a render that is over and read: the row the user just cleared away.
    /// A live one is cancelled, never forgotten. Local only, which is final for
    /// an acknowledged image, since the host has already freed that one.
    pub fn forget(self: *Mirror, id: ImageId) bool {
        for (self.images.items, 0..) |*im, i| {
            if (im.info.id != id) continue;
            switch (im.info.status) {
                .failed, .canceled => {},
                else => return false,
            }
            var gone = self.images.orderedRemove(i);
            gone.deinit(self.gpa);
            return true;
        }
        return false;
    }

    /// This picture is gone: its file will not open and its pixels are not here.
    pub fn markLost(self: *Mirror, id: ImageId, why: []const u8) void {
        const im = self.byId(id) orelse return;
        im.info.status = .failed;
        im.missing = true;
        freeStr(self.gpa, im.info.failure);
        im.info.failure = dupeStr(self.gpa, why);
        im.rev += 1;
    }

    /// Put back the pixels of an image read again from its file. Takes `px`.
    pub fn restorePixels(self: *Mirror, id: ImageId, px: []u8, w: u32, h: u32) void {
        const im = self.byId(id) orelse return self.gpa.free(px);
        if (im.pixels) |old| self.gpa.free(old);
        im.pixels = px;
        im.info.width = w;
        im.info.height = h;
        im.rev += 1;
    }

    pub fn takeErr(self: *Mirror) ?HostErr {
        const e = self.last_err orelse return null;
        self.last_err = null;
        return e;
    }

    /// The consumer frees `text`, as for `takeErr`.
    pub fn takeNotice(self: *Mirror) ?HostNotice {
        const n = self.last_notice orelse return null;
        self.last_notice = null;
        return n;
    }

    pub fn takeDiffPeak(self: *Mirror) ?PeakUpdate {
        const p = self.diff_peak orelse return null;
        self.diff_peak = null;
        return p;
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "navTarget: carousel semantics for the back/next buttons" {
    // Single variant: back does nothing (the UI disables it), next regenerates.
    try std.testing.expectEqual(Nav.none, navTarget(0, 1, .back));
    try std.testing.expectEqual(Nav.regenerate, navTarget(0, 1, .next));
    // Middle of three: both directions navigate.
    try std.testing.expectEqual(Nav{ .select = 0 }, navTarget(1, 3, .back));
    try std.testing.expectEqual(Nav{ .select = 2 }, navTarget(1, 3, .next));
    // Newest of three: back navigates, next regenerates (appends a fourth).
    try std.testing.expectEqual(Nav{ .select = 1 }, navTarget(2, 3, .back));
    try std.testing.expectEqual(Nav.regenerate, navTarget(2, 3, .next));
}


fn feed(m: *Mirror, ev: wire.Event) !void {
    const bytes = try wire.encodeAlloc(testing.allocator, ev);
    defer testing.allocator.free(bytes);
    m.apply(.{ .text = bytes });
}

test "transcript, deltas and stats build the messages a view reads" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .transcript = .{ .messages = &.{
        .{ .role = .user, .variants = &.{.{ .text = "hi" }} },
        .{ .role = .assistant, .variants = &.{ .{ .text = "a" }, .{ .text = "b", .reason_open = "<t>", .reason_close = "</t>" } }, .cur = 1 },
    } } });
    try testing.expectEqual(@as(usize, 2), m.messages.items.len);
    try feed(&m, .{ .delta = .{ .msg = 1, .variant = 1, .text = "cd" } });
    try feed(&m, .{ .delta = .{ .msg = 5, .variant = 0, .text = "ignored" } });
    try feed(&m, .{ .stats = .{ .msg = 1, .variant = 1, .stats = .{ .gen_tokens = 4 } } });
    const v = m.messages.items[1].active();
    try testing.expectEqualStrings("bcd", v.text.items);
    try testing.expectEqualStrings("</t>", v.reason_close);
    try testing.expectEqual(@as(usize, 4), v.stats.gen_tokens);
    try feed(&m, .{ .turn_end = .{ .msg = 1, .variant = 1 } });
    try testing.expectEqual(@as(u64, 1), m.turns_ended);
}

test "images upsert by id, keep fetched pixels across a queue merge, and pull previews by revision" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .status = .generating, .prompt = "p", .preview_rev = 2 } });
    try feed(&m, .{ .img = .{ .id = 4, .status = .done, .pixels_rev = 1, .width = 2, .height = 1 } });
    try testing.expectEqual(@as(usize, 2), m.images.items.len);

    const Posted = struct {
        var reqs: [8]wire.Request = undefined;
        var n: usize = 0;
        var acks: usize = 0;
        // Fetches only; the receipt has its own test.
        fn post(_: void, r: wire.Request) void {
            if (r == .img_ack) {
                acks += 1;
                return;
            }
            reqs[n] = r;
            n += 1;
        }
    };
    m.pollFetches(1_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 2), Posted.n);
    try testing.expectEqual(wire.PixelKind.preview, Posted.reqs[0].img_fetch.kind);
    try testing.expectEqual(@as(ImageId, 3), Posted.reqs[0].img_fetch.image);
    try testing.expectEqual(wire.PixelKind.pixels, Posted.reqs[1].img_fetch.kind);
    // Outstanding: nothing more is asked until the frame lands.
    m.pollFetches(2_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 2), Posted.n);

    // The preview lands at rev 2; the pixels land for 4.
    var px = [_]u8{ 1, 2, 3, 255, 4, 5, 6, 255 };
    m.apply(.{ .bin = .{ .hdr = .{ .kind = .preview_rgba, .id = 3, .rev = 2, .w = 2, .h = 1, .len = 8 }, .payload = &px } });
    m.apply(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 4, .rev = 1, .w = 2, .h = 1, .len = 8 }, .payload = &px } });
    try testing.expectEqual(@as(u32, 2), m.byId(3).?.preview_rev);
    try testing.expectEqualSlices(u8, &px, m.byId(4).?.pixels.?);
    // Same revision: no new preview fetch even after the backoff.
    m.pollFetches(10_000_000_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 2), Posted.n);
    // A newer revision is fetched once the backoff passes.
    try feed(&m, .{ .img = .{ .id = 3, .status = .generating, .prompt = "p", .preview_rev = 3 } });
    m.pollFetches(10_000_000_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 3), Posted.n);
    try testing.expectEqual(@as(u32, 2), Posted.reqs[2].img_fetch.have_rev);

    // A queue snapshot listing only 4 drops 3 and keeps 4's pixels.
    try feed(&m, .{ .queue = .{ .images = &.{.{ .id = 4, .status = .done, .pixels_rev = 1, .width = 2, .height = 1 }} } });
    try testing.expectEqual(@as(usize, 1), m.images.items.len);
    try testing.expect(m.byId(4).?.pixels != null);
    try testing.expect(m.byId(3) == null);
    // An empty queue (the engine is gone) frees everything the host still had.
    // The finished one stays: it was acknowledged, so the host freed it first
    // and this client is the only place it exists.
    try feed(&m, .{ .queue = .{} });
    try testing.expectEqual(@as(usize, 1), m.images.items.len);
    try testing.expect(m.byId(4).?.acked);
    try testing.expect(Posted.acks > 0);
}

test "the catalog is replaced only when entries arrive, and the scan status always" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    var cat = try catalog.Catalog.fromEntries(testing.allocator, &.{
        .{ .path = "/m/a.gguf", .size = 5, .mtime_ns = 1, .llm = .{ .arch = "qwen3", .size_label = "8B", .width = 4096, .blocks = 36, .supported = true, .vision = false, .class = "Qwen3 8B" } },
    });
    defer cat.deinit();
    const json = try cat.toJsonAlloc(testing.allocator);
    defer testing.allocator.free(json);
    try feed(&m, .{ .catalog = .{ .rev = 0, .scanning = true } });
    try testing.expect(m.scanning);
    try testing.expect(!m.scannedOnce());
    try testing.expectEqual(@as(usize, 0), m.catalog.entries.len);
    try feed(&m, .{ .catalog = .{ .rev = 1, .files = 1, .probed = 1, .json = json } });
    try testing.expect(!m.scanning);
    try testing.expect(m.scannedOnce());
    try testing.expectEqual(@as(u64, 1), m.catalog_seq);
    try testing.expectEqualStrings("a", m.catalog.entries[0].stem());
    try testing.expect(m.catalog.entries[0].isChatModel());
    try testing.expectEqual(@as(usize, 1), m.scan_report.probed);
}

test "a dropped link keeps ids and fails what was running; the same host's next word overwrites it" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    try feed(&m, .{ .img = .{ .id = 4, .client_ref = 77, .status = .generating, .preview_rev = 2, .prompt = "p" } });
    try feed(&m, .{ .img = .{ .id = 6, .status = .generating } });
    try feed(&m, .{ .state = .{ .llm_busy = true } });
    m.linkDown();
    // A finished render is not lost with the link: the same daemon still has it.
    try testing.expectEqual(wire.ImageStatus.done, m.byId(3).?.status());
    try testing.expect(!m.byId(3).?.pixels_requested);
    try testing.expectEqual(wire.ImageStatus.failed, m.byId(4).?.status());
    try testing.expectEqualStrings("HostLost", m.byId(4).?.info.failure);
    try testing.expect(!m.byId(4).?.local);
    try testing.expect(!m.spokeOnce());
    // A render this client asked for is still findable by its reference, which
    // is how its owner decides what to do with it.
    try testing.expectEqual(@as(ImageId, 4), m.byRef(77).?.info.id);
    // The same host, reconnected: its snapshot names the same id and wins.
    try feed(&m, .{ .img = .{ .id = 4, .client_ref = 77, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    try testing.expectEqual(@as(usize, 3), m.images.items.len);
    try testing.expectEqual(wire.ImageStatus.done, m.byId(4).?.status());
}

test "a dropped link leaves a render that was already over exactly as it was" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .client_ref = 7, .status = .canceled } });
    try feed(&m, .{ .img = .{ .id = 4, .client_ref = 8, .status = .failed, .failure = "OutOfMemory" } });
    try feed(&m, .{ .img = .{ .id = 5, .client_ref = 9, .status = .generating } });
    m.linkDown();
    // Failing a cancelled render here toasts a failure the user did not get and
    // replays the render they stopped; an old failure would lose its reason.
    try testing.expectEqual(wire.ImageStatus.canceled, m.byId(3).?.status());
    try testing.expectEqual(wire.ImageStatus.failed, m.byId(4).?.status());
    try testing.expectEqualStrings("OutOfMemory", m.byId(4).?.info.failure);
    try testing.expectEqual(wire.ImageStatus.failed, m.byId(5).?.status());
    try testing.expectEqualStrings("HostLost", m.byId(5).?.info.failure);
}

test "a restart does not fail a finished picture whose pixels were dropped to its file" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    var px = [_]u8{ 1, 2, 3, 255 };
    m.apply(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 3, .rev = 1, .w = 1, .h = 1, .len = 4 }, .payload = &px } });
    // Saved, then evicted: `receiving()` is true and the file is the copy.
    const im = m.byId(3).?;
    im.saved_path = try testing.allocator.dupe(u8, "/tmp/x.png");
    _ = m.evictPixels(0, &.{});
    try testing.expect(m.byId(3).?.receiving() and m.byId(3).?.restorable());

    m.hostRestarted();
    const back = m.images.items[0];
    try testing.expectEqual(wire.ImageStatus.done, back.status());
    try testing.expect(back.restorable()); // else the file is never read again
}

test "a restarted host leaves finished pictures, fails the rest, and the transcript follows the new ids" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    try feed(&m, .{ .img = .{ .id = 4, .status = .generating, .preview_rev = 2, .prompt = "p" } });
    try feed(&m, .{ .img = .{ .id = 5, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } }); // pixels never fetched
    var px = [_]u8{ 1, 2, 3, 255 };
    m.apply(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 3, .rev = 1, .w = 1, .h = 1, .len = 4 }, .payload = &px } });
    try feed(&m, .{ .transcript = .{ .messages = &.{
        .{ .role = .user, .variants = &.{.{ .text = "draw" }}, .attachments = &.{5} },
        .{ .role = .assistant, .variants = &.{.{ .text = "here", .images = &.{ 3, 4 } }} },
    } } });
    try feed(&m, .{ .state = .{ .llm_busy = true, .diff_family = "krea2" } });

    m.hostRestarted();
    try testing.expectEqual(@as(usize, 3), m.images.items.len);
    for (m.images.items) |*im| try testing.expect(im.local and im.info.id >= (1 << 63));
    const a = &m.messages.items[1].variants.items[0].images.items;
    const kept = m.byId(a.*[0]).?;
    try testing.expectEqual(wire.ImageStatus.done, kept.status());
    try testing.expectEqualSlices(u8, &px, kept.pixels.?);
    const lost = m.byId(a.*[1]).?;
    try testing.expectEqual(wire.ImageStatus.failed, lost.status());
    try testing.expectEqualStrings("HostLost", lost.info.failure);
    try testing.expectEqualStrings("p", lost.info.prompt);
    const unfetched = m.byId(m.messages.items[0].attachments.items[0]).?;
    try testing.expectEqual(wire.ImageStatus.failed, unfetched.status());
    try testing.expect(!m.state.llm_busy);
    try testing.expectEqualStrings("", m.state.diff_family);

    // What goes back to the next host names the new ids and the text as is.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const w = try m.toWire(arena.allocator());
    try testing.expectEqual(@as(usize, 2), w.len);
    try testing.expectEqualStrings("here", w[1].variants[0].text);
    try testing.expectEqual(kept.info.id, w[1].variants[0].images[0]);
    try testing.expectEqual(unfetched.info.id, w[0].attachments[0]);
    // And it encodes: the same bytes a client's adopt would carry.
    const bytes = try wire.encodeAlloc(arena.allocator(), wire.Request{ .chat_adopt = .{ .messages = w } });
    try testing.expect(std.mem.indexOf(u8, bytes, "\"here\"") != null);
}

test "a finished render this daemon never made cannot be waited for" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .img = .{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    try testing.expect(m.byId(3).?.receiving());
    m.hostRestarted();
    const im = m.images.items[0];
    try testing.expectEqual(wire.ImageStatus.failed, im.status());
    try testing.expectEqualStrings("HostLost", im.info.failure);
}

test "two mirrors disowning their images mint ids that do not collide" {
    var a = Mirror.init(testing.allocator);
    defer a.deinit();
    var b = Mirror.init(testing.allocator);
    defer b.deinit();
    // Both hosts minted id 7; both go away.
    try feed(&a, .{ .img = .{ .id = 7, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    try feed(&b, .{ .img = .{ .id = 7, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    a.hostRestarted();
    b.hostRestarted();
    try testing.expect(a.images.items[0].info.id != b.images.items[0].info.id);
    try testing.expect(a.images.items[0].info.id >= (1 << 63));
}

test "an image is acknowledged once this client holds all of it, and outlives the host forgetting it" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    const Posted = struct {
        var reqs: [16]wire.Request = undefined;
        var n: usize = 0;
        fn post(_: void, r: wire.Request) void {
            if (n == reqs.len) return;
            reqs[n] = r;
            n += 1;
        }
        fn acks() usize {
            var c: usize = 0;
            for (reqs[0..n]) |r| {
                if (r == .img_ack) c += 1;
            }
            return c;
        }
    };

    // A render still going is not acknowledged.
    try feed(&m, .{ .img = .{ .id = 3, .status = .generating, .preview_rev = 1 } });
    m.pollFetches(1_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 0), Posted.acks());

    // Finished, but its picture is still on the way: still not, or the host
    // would free the pixels this client is waiting for.
    try feed(&m, .{ .img = .{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 } });
    m.pollFetches(2_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 0), Posted.acks());

    // It lands: acknowledged, once, however many passes run.
    var px = [_]u8{ 1, 2, 3, 255 };
    m.apply(.{ .bin = .{ .hdr = .{ .kind = .image_rgba, .id = 3, .rev = 1, .w = 1, .h = 1, .len = 4 }, .payload = &px } });
    m.pollFetches(3_000, {}, Posted.post);
    m.pollFetches(4_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 1), Posted.acks());
    try testing.expectEqual(@as(ImageId, 3), Posted.reqs[Posted.n - 1].img_ack.image);

    // A failure carries no pixels, so it is acknowledged as soon as it lands.
    try feed(&m, .{ .img = .{ .id = 4, .status = .failed, .failure = "OutOfMemory" } });
    m.pollFetches(5_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 2), Posted.acks());

    // So is a finished one the host has no pixels for.
    try feed(&m, .{ .img = .{ .id = 5, .status = .done } });
    m.pollFetches(6_000, {}, Posted.post);
    try testing.expectEqual(@as(usize, 3), Posted.acks());

    // A snapshot that still names an acknowledged image keeps ONE copy of it,
    // with its pixels: it must not be taken aside and then added again.
    try feed(&m, .{ .queue = .{ .images = &.{.{ .id = 3, .status = .done, .pixels_rev = 1, .width = 1, .height = 1 }} } });
    var threes: usize = 0;
    for (m.images.items) |*im| {
        if (im.info.id == 3) threes += 1;
    }
    try testing.expectEqual(@as(usize, 1), threes);
    try testing.expectEqualSlices(u8, &px, m.byId(3).?.pixels.?);

    // The host has freed all three, so its next snapshot lists none of them.
    // They stay here, because this client is now the only place they exist.
    try feed(&m, .{ .queue = .{} });
    try testing.expectEqual(@as(usize, 3), m.images.items.len);
    try testing.expectEqualSlices(u8, &px, m.byId(3).?.pixels.?);
    try testing.expectEqualStrings("OutOfMemory", m.byId(4).?.info.failure);
}

test "the oldest pictures already on disk keep only their thumbnails" {
    const gpa = testing.allocator;
    var m = Mirror.init(gpa);
    defer m.deinit();
    // Four finished renders, each with a preview and full pixels; the last is
    // not saved anywhere.
    for (0..4) |i| {
        const px = try gpa.alloc(u8, 4);
        @memset(px, 7);
        const saved: ?[]u8 = if (i == 3) null else try std.fmt.allocPrint(gpa, "/out/{d}.png", .{i});
        const id = m.addLocal(.{ .status = .done, .width = 1, .height = 1 }, px, saved);
        const im = m.byId(id).?;
        im.preview = try gpa.alloc(u8, 4);
        im.preview_w = 1;
        im.preview_h = 1;
    }
    const ids = [_]ImageId{
        m.images.items[0].info.id,
        m.images.items[1].info.id,
        m.images.items[2].info.id,
        m.images.items[3].info.id,
    };
    // Keep one, pin the oldest: the pinned one and the newest survive, the
    // unsaved one cannot be dropped because nothing could read it back.
    const freed = m.evictPixels(1, &.{ids[0]});
    errdefer std.debug.print("freed={d}\n", .{freed});
    try testing.expect(freed > 0);
    try testing.expect(m.byId(ids[0]).?.pixels != null); // pinned
    try testing.expect(m.byId(ids[1]).?.pixels == null); // dropped
    try testing.expect(m.byId(ids[2]).?.pixels == null); // dropped
    try testing.expect(m.byId(ids[3]).?.pixels != null); // newest, and unsaved
    // Dropped is not gone: the thumbnail still draws, and the file names it.
    const t = m.byId(ids[1]).?.thumb().?;
    try testing.expectEqual(@as(u32, 1), t.w);
    try testing.expect(m.byId(ids[1]).?.restorable());

    // Read one back and it is whole again.
    const back = try gpa.alloc(u8, 16);
    @memset(back, 3);
    m.restorePixels(ids[1], back, 2, 2);
    try testing.expect(!m.byId(ids[1]).?.restorable());
    try testing.expectEqual(@as(u32, 2), m.byId(ids[1]).?.info.width);
    // A file that will not open takes its picture with it, once. It is MISSING,
    // not failed-and-retryable: nothing was rendered and nothing can be.
    m.markLost(ids[2], "SavedImageMissing");
    try testing.expectEqual(wire.ImageStatus.failed, m.byId(ids[2]).?.status());
    try testing.expect(m.byId(ids[2]).?.missing);
    try testing.expect(!m.byId(ids[2]).?.restorable());
}

test "state strings are owned copies and errors are taken once" {
    var m = Mirror.init(testing.allocator);
    defer m.deinit();
    try feed(&m, .{ .state = .{ .loading = true, .load_err = "Boom", .attachments = &.{ 1, 2 } } });
    try feed(&m, .{ .state = .{ .diff_family = "krea2" } });
    try testing.expectEqualStrings("krea2", m.state.diff_family);
    try testing.expectEqualStrings("", m.state.load_err);
    try testing.expectEqual(@as(usize, 0), m.state.attachments.len);
    try feed(&m, .{ .err = .{ .code = .no_model, .text = "no image model" } });
    const e = m.takeErr().?;
    defer testing.allocator.free(e.text);
    try testing.expectEqual(wire.ErrCode.no_model, e.code);
    try testing.expect(m.takeErr() == null);
    try feed(&m, .{ .diff_peak = .{ .peak = 5, .key = 9 } });
    try testing.expectEqual(@as(u64, 9), m.takeDiffPeak().?.key);
}
