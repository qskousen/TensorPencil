//! tp-gui application: window, manual render loop, and the chat frame.
//!
//! The loop mirrors DiffKeep's hand-written SDL loop (rather than
//! `dvui.App.run`) so that secondary windows, notably a zoom/pan viewer for
//! generated images, can be added later without restructuring. For now there
//! is a single window; per-window event routing is introduced when the first
//! secondary window lands.
const std = @import("std");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const tp = @import("TensorPencil");
const vram = tp.vram;
const toolcall = @import("shared").toolcall;
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");
const markdown_view = @import("markdown_view.zig");
const viewer = @import("viewer.zig");
const config = @import("shared").config;
const config_view = @import("config_view.zig");
const selection = @import("client").selection;
const model_spec = @import("shared").model_spec;
const model_lib = @import("model_lib.zig");
const model_menu = @import("model_menu.zig");
const image_view = @import("image_view.zig");
const prompt_history = @import("client").prompt_history;
const pipeline_map = @import("shared").pipeline_map;
const clipboard = @import("clipboard.zig");
const meter = @import("meter.zig");
const status_bar = @import("status_bar.zig");
const toast = @import("toast.zig");
const style = @import("style.zig");
const shell = @import("shell.zig");
const bubbles = @import("bubbles.zig");
const queue_rail = @import("queue_rail.zig");
const history = @import("client").history;
const framing = @import("shared").framing;
const turn_stats = @import("shared").turn_stats;
const hosts = @import("client").hosts;
const models = @import("client").models;
const sync = @import("client").sync;
const mirror = @import("client").mirror;
const save_image = @import("client").save_image;
const wire = @import("serve").wire;
const vips = @import("vips");

/// The engine host: a tp-serve process, spawned beside this binary when none
/// is listening, plus every host the settings list. The app talks to them only
/// in wire requests and reads them only through their mirrors; `g_m` is the
/// chat host's. No engine object exists in this process.
var g_hosts: hosts.Hosts = undefined;
var g_m: *mirror.Mirror = undefined;

/// Every host's catalog as one list, one row per file (`client/models.zig`).
/// Menus, the side-file pickers and the selection all read THIS, never one
/// host's: a model the render host holds and the chat host does not is still a
/// model this client can pick, and the scheduler is what decides where it runs.
var g_models: models.Union = undefined;
/// `Hosts.modelSeq` when `g_models` was built.
var g_model_seq: u64 = 0;

/// Rebuild the merged view. Builds before freeing, so a failure leaves the
/// view that was standing rather than an empty menu.
fn rebuildModels() void {
    var buf: [models.max_sources]models.Source = undefined;
    const next = models.build(g_gpa, g_hosts.modelSources(&buf)) catch |err| {
        std.log.err("merging the hosts' catalogs: {t}", .{err});
        return;
    };
    g_models.deinit();
    g_models = next;
}
/// VRAM meter handle positions (fractions of the card). The meter mutates them
/// in place on drag; release sends them.
/// One bar's sampling state per host, claimed by host id on first sight. The
/// handles themselves live on the slot (`hosts.Slot.meter`), so the host list
/// can greet a host with its own values without asking the view layer.
var g_views: [config.max_hosts + 1]status_bar.View = @splat(.{});
var g_view_ids: [config.max_hosts + 1]?hosts.HostId = @splat(null);
/// Whose bar is being rendered: the meter's callbacks take no argument, so
/// this is how a drag knows which host it moved.
var g_meter_host: hosts.HostId = 0;

fn viewFor(id: hosts.HostId) *status_bar.View {
    for (g_view_ids, 0..) |held, i| if (held == id) return &g_views[i];
    for (g_view_ids, 0..) |held, i| if (held == null) {
        g_view_ids[i] = id;
        g_views[i] = .{};
        return &g_views[i];
    };
    // More hosts than bars: the last one shares, which is only cosmetic.
    return &g_views[g_views.len - 1];
}

/// Forget the sampling history of hosts that are gone, so a later host with a
/// recycled id does not inherit somebody else's sparklines.
fn dropStaleViews() void {
    for (&g_view_ids) |*held| if (held.*) |id| {
        if (g_hosts.slotOf(id) == null) held.* = null;
    };
}
/// The text of a message sent while no LLM was resident, shown as a bubble
/// until the host reports it taken.
var g_pending_text: ?[]u8 = null;
/// Images attached before a session existed: uploaded to the host, kept here
/// for the thumbnail strip until the session has them.
const StagedImage = struct { rgba: []u8, width: usize, height: usize };

/// Wall-clock milliseconds, the stamp the history stores use.
fn nowMs() i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(g_io).nanoseconds, std.time.ns_per_ms));
}
var g_staged: std.ArrayList(StagedImage) = .empty;
/// The configured checkpoint's family, from the FILE (the catalog is empty on
/// a cold start), memoized per path.
var g_family_cache: ?model_spec.Cache = null;

fn post(req: wire.Request) void {
    g_hosts.post(&g_config, req);
}

/// Hand every host its share of the settings as they stand (`config.host_fields`).
fn postSettings() void {
    g_hosts.postSettings(&g_config);
}

fn rescan() void {
    g_hosts.postScan(&g_config);
}

var g_host_status: [96]u8 = undefined;

/// Try a host the settings hold but nothing is connected with yet. Nothing is
/// committed: the answer only decides what the row says.
fn tryHost(e: *const config.HostEntry) void {
    g_hosts.tryHost(e);
}

fn reconnectHost(name: []const u8) void {
    g_hosts.reconnectNamed(name);
}

fn hostStatus(name: []const u8, e: *const config.HostEntry) config_view.HostState {
    // The form's row and the live connection are two different things until
    // Apply runs, and a status from the old connection under a freshly pasted
    // pairing string reads as "the new token was refused too". While a row is
    // ahead of the connection it reports the trial instead.
    const s = g_hosts.slotByName(name);
    const in_force = if (s) |sl| (sl.entry != null and sl.entry.?.sameEndpoint(e)) else false;
    if (!in_force) {
        const t = g_hosts.trialOf(e) orelse return .{ .text = "Apply & Reload to use it", .tone = .pending };
        return .{ .text = t.text, .tone = if (t.state == .refused) .bad else .pending };
    }
    if (hosts.Hosts.troubleOf(s.?)) |t| return .{ .text = t, .tone = .bad, .offer_reconnect = true };
    // A host that is up but never gets a render looks broken; say which.
    const why = g_hosts.whyNotRender(s.?, &g_config) orelse return .{ .text = "up", .tone = .ok };
    const text = std.fmt.bufPrint(&g_host_status, "up · {s}", .{why.long}) catch "up";
    return .{ .text = text, .tone = .pending };
}

var g_sync: sync.Syncer = undefined;
var g_sync_status: [64]u8 = undefined;

fn hostSync(name: []const u8) config_view.HostSync {
    var out: config_view.HostSync = .{};
    // A finished transfer keeps its row until the next send, so its result is
    // readable rather than vanishing on the frame it lands.
    if (g_sync.forHost(name)) |j| {
        out.sending = j.stem;
        out.status = j.status(&g_sync_status);
    }
    const s = g_hosts.slotByName(name) orelse return out;
    // What is still missing, transfer or no transfer: a finished send that
    // hides the next file is why a host stays short after one was sent.
    if (g_hosts.missingModel(s, &g_config)) |path| {
        out.missing = g_models.cat.refName(path);
        out.missing_count = g_hosts.missingAll(s, &g_config);
    }
    return out;
}

fn sendModel(name: []const u8) void {
    const s = g_hosts.slotByName(name) orelse return;
    const path = g_hosts.missingModel(s, &g_config) orelse return;
    sendPath(name, path);
}

/// One named file to one host, from the library grid. `path` is this machine's:
/// the local host is the only one whose files can be read to send.
fn sendPath(name: []const u8, path: []const u8) void {
    const s = g_hosts.slotByName(name) orelse return;
    const entry = s.entry orelse return;
    g_sync.start(name, entry, path) catch |err| std.log.err("sending to {s}: {t}", .{ name, err });
}

/// One file FROM a host into the first model folder, which is a folder the
/// local host scans: the file then shows up in the local catalog and the id it
/// was fetched by resolves here too.
fn pullPath(name: []const u8, id: []const u8, stem: []const u8) void {
    const s = g_hosts.slotByName(name) orelse return;
    const entry = s.entry orelse return;
    const dest = firstModelDir() orelse {
        std.log.err("no model folder to put {s} in; add one in Settings", .{stem});
        return;
    };
    g_sync.startPull(name, entry, id, stem, dest) catch |err| std.log.err("fetching from {s}: {t}", .{ name, err });
}

fn firstModelDir() ?[]const u8 {
    for (g_config.model_dirs.slice()) |*d| if (d.path.opt()) |p| return p;
    return null;
}


/// Apply every frame the host emitted since last time, pull the pixels the
/// mirror wants, persist what the host measured, and save finished renders.
/// Room over what a chat tile draws a preview at, so a hi-dpi screen is not
/// upscaling. No more than that: on a remote host the finished picture queues
/// behind these frames.
const preview_headroom: f32 = 1.5;

/// Full pictures held in memory before the oldest ones already on disk are
/// dropped to their thumbnails. Sixteen 1024 squares is about 64 MiB.
const pixels_kept: usize = 16;

/// The pictures a view is showing at full size, canvas and viewer. Not dropped
/// while shown, and read back from file when they already were. Cleared at the
/// top of every frame and re-asserted by whatever draws.
var g_shown: [2]?wire.ImageId = .{ null, null };

/// Note that `im` is on screen at full size, and ask for its pixels back when
/// they were dropped.
fn shown(slot: usize, im: *mirror.Image) *const mirror.Image {
    g_shown[slot] = im.info.id;
    return im;
}

fn pumpHost() void {
    g_hosts.pump(&g_config);
    g_sync.poll();
    // A pulled file is on this disk but not in this machine's catalog until the
    // local host looks again; until it is, the id it was fetched by resolves
    // nowhere here.
    if (g_sync.takeLanded()) g_hosts.rescanNamed("local", &g_config);
    if (g_retry_request) |id| {
        g_retry_request = null;
        g_hosts.retry(&g_config, id);
    }
    g_m = g_hosts.chatMirror();
    g_hosts.setPreviewMaxEdge(if (g_view == .image)
        image_view.canvasMaxEdge()
    else
        @intFromFloat(@as(f32, @floatFromInt(bubbles.tileMaxEdge())) * preview_headroom));
    const now = std.Io.Clock.real.now(g_io).nanoseconds;
    for (g_hosts.slots.items) |slot| {
        slot.mirror.pollFetches(now, slot, hosts.Slot.postCtx);
        if (slot.mirror.takeDiffPeak()) |u| persistDiffPeak(u.peak, u.key);
        if (slot.mirror.takeErr()) |e| {
            std.log.warn("host {s}: {t}: {s}", .{ slot.name, e.code, e.text });
            if (e.text.len > 0) toast.post(.err, "{s}: {s}", .{ slot.name, e.text });
            g_gpa.free(e.text);
        }
    }
    reportFailures();
    // The host has the message, or the load that will take it: drop the
    // provisional bubble.
    if (g_pending_text != null and (g_m.state.llm_resident and !g_m.state.pending_submit)) {
        g_gpa.free(g_pending_text.?);
        g_pending_text = null;
    }
    if (g_m.state.llm_resident and g_m.state.staged == 0) clearStaged();
    saveNewRenders();
    var pins: [g_shown.len]wire.ImageId = undefined;
    var n_pins: usize = 0;
    for (g_shown) |p| if (p) |id| {
        pins[n_pins] = id;
        n_pins += 1;
    };
    for (g_hosts.slots.items) |slot| _ = slot.mirror.evictPixels(pixels_kept, pins[0..n_pins]);
    restoreShown();
}

/// Read back the file of a picture a view wants at full size whose pixels were
/// dropped. One per frame: decoding is not free and only one can be looked at.
fn restoreShown() void {
    for (g_shown) |p| {
        const id = p orelse continue;
        const slot = g_hosts.imageOwner(id) orelse continue;
        const im = slot.mirror.byId(id) orelse continue;
        // A slot that already has its pixels is not the one to stop at, or a
        // pinned slot below it never gets read back at all.
        if (im.pixels != null or !im.restorable()) continue;
        _ = fullPixels(id);
        return;
    }
}

/// This picture at full size, reading its file again when the pixels were
/// dropped to save memory. Null when there are none and none can be had.
fn fullPixels(id: wire.ImageId) ?[]const u8 {
    const slot = g_hosts.imageOwner(id) orelse return null;
    const im = slot.mirror.byId(id) orelse return null;
    if (im.pixels) |px| return px;
    if (!im.restorable()) return null;
    const path = im.saved_path.?;
    const dec = vips.loadRgb(g_gpa, path) catch |err| {
        std.log.warn("cannot read {s} again: {t}", .{ path, err });
        slot.mirror.markLost(id, "SavedImageMissing");
        return null;
    };
    defer g_gpa.free(dec.pixels);
    const rgba = tp.image.rgbToRgba(g_gpa, dec.pixels, dec.width, dec.height) catch return null;
    slot.mirror.restorePixels(id, rgba, @intCast(dec.width), @intCast(dec.height));
    return slot.mirror.byId(id).?.pixels;
}

/// Say what happened to every render that failed, and where it went. A failed
/// job leaves the queue it was in, so without this the user watches an image
/// disappear and gets nothing to read, on any host.
fn reportFailures() void {
    while (g_hosts.takeFailure(&g_config)) |f| {
        if (f.moved_to) |to| {
            toast.post(.warn, "Image failed on {s}: {s}. Trying {s} instead.", .{ f.host, f.why, to });
        } else if (f.requeued) {
            toast.post(.warn, "Image failed on {s}: {s}. Back in the queue for another host.", .{ f.host, f.why });
        } else {
            toast.post(.err, "Image failed on {s}: {s}. No other host can take it.", .{ f.host, f.why });
        }
        std.log.warn("image failed on host {s}: {s}", .{ f.host, f.why });
    }
}

/// Write every finished render whose pixels have arrived and that has not
/// been written yet, when saving is on. Once per image, success or not.
fn saveNewRenders() void {
    const dir = g_config.output_dir.opt() orelse return;
    for (g_hosts.slots.items) |slot| for (slot.mirror.images.items) |*im| {
        if (im.save_tried or im.local or im.status() != .done) continue;
        const px = im.pixels orelse continue;
        im.save_tried = true;
        const path = save_image.save(g_gpa, g_io, dir, &im.info, px, im.info.width, im.info.height) catch |err| {
            std.log.err("image save failed: {t}", .{err});
            continue;
        };
        std.log.info("saved image to {s}", .{path});
        im.saved_path = path;
    };
}

fn configuredFamily() ?model_spec.Family {
    const path = g_config.diffusion_model.opt() orelse return null;
    if (g_family_cache == null) g_family_cache = model_spec.Cache.init(g_gpa, g_io);
    return (g_family_cache.?.primary(path).info() orelse return null).family;
}

fn llmPaused() bool {
    return g_m.state.llm_paused;
}
fn diffPaused() bool {
    return g_m.state.diff_paused;
}
fn toggleLlmPause() void {
    const s = g_hosts.slotOf(g_meter_host) orelse g_hosts.chatSlot();
    s.post(.{ .chat_pause = .{ .paused = !s.mirror.state.llm_paused } });
}
fn toggleDiffPause() void {
    const s = g_hosts.slotOf(g_meter_host) orelse g_hosts.chatSlot();
    s.post(.{ .img_pause = .{ .paused = !s.mirror.state.diff_paused } });
}

/// The rail's pause button is about the whole queue, which spans every host,
/// so this one broadcasts. The per-host pause lives on that host's meter.
fn toggleDiffPauseEverywhere() void {
    post(.{ .img_pause = .{ .paused = !diffPaused() } });
}

/// Same entry point the send button uses. Returns false when the message went
/// nowhere (empty, or no model to load), so a headless driver does not wait
/// forever. The host may still refuse it; that comes back as an error event.
fn submitChat(text: []const u8) bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return false;
    if (!g_m.state.llm_resident) {
        if (g_config.llm_model.opt() == null) return false;
        if (g_pending_text) |p| g_gpa.free(p);
        g_pending_text = g_gpa.dupe(u8, trimmed) catch null;
    }
    post(.{ .chat_submit = .{ .text = trimmed } });
    return true;
}

/// Start a fresh conversation, clearing the input box. Only the transcript is
/// reset; generated images stay in the studio gallery (and the viewer keeps
/// working).
fn newChat() void {
    post(.chat_new);
    g_handover_rev = g_m.transcript_rev;
    if (g_pending_text) |p| g_gpa.free(p);
    g_pending_text = null;
    clearStaged();
    g_input.clearRetainingCapacity();
    g_follow_bottom = true;
}

/// Attach decoded RGB to the next message: uploaded to the host, and, while no
/// session exists to hold it, kept here for the thumbnail strip.
fn attachImage(rgb: []const u8, w: usize, h: usize) void {
    const payload = g_gpa.dupe(u8, rgb) catch return;
    if (!g_m.state.llm_resident) {
        if (tp.image.rgbToRgba(g_gpa, rgb, w, h)) |rgba| {
            g_staged.append(g_gpa, .{ .rgba = rgba, .width = w, .height = h }) catch g_gpa.free(rgba);
        } else |_| {}
    }
    g_hosts.chatSlot().postFrame(g_gpa, .{ .bin = .{
        .hdr = .{ .kind = .rgb_upload, .id = 0, .rev = 0, .w = @intCast(w), .h = @intCast(h), .len = @intCast(payload.len) },
        .payload = payload,
    } });
}

/// Let the chat model see a picture the client holds: by id when the chat
/// host minted it, else by its pixels (another host's render, or a reopened
/// file).
/// The pixels travel from here, never by naming the id: a host frees an image
/// the moment this client takes delivery, so its own copy is the only one left.
fn attachFromMirror(im: *const mirror.Image) void {
    const rgba = fullPixels(im.info.id) orelse return;
    const w: usize = im.info.width;
    const h: usize = im.info.height;
    const rgb = g_gpa.alloc(u8, w * h * 3) catch return;
    defer g_gpa.free(rgb);
    for (0..w * h) |i| @memcpy(rgb[i * 3 .. i * 3 + 3], rgba[i * 4 .. i * 4 + 3]);
    attachImage(rgb, w, h);
}

fn clearStaged() void {
    for (g_staged.items) |st| g_gpa.free(st.rgba);
    g_staged.clearRetainingCapacity();
}

/// Persist the diffusion pipeline's measured peak residency, so the NEXT run's
/// first image plans against a measurement instead of the file-size bootstrap.
fn persistDiffPeak(peak: u64, key: u64) void {
    g_config.diff_peak_resident = peak;
    g_config.diff_peak_key = key;
    g_config_baseline.diff_peak_resident = peak;
    g_config_baseline.diff_peak_key = key;
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err|
        std.log.warn("[vram] could not persist measured diffusion peak: {t}", .{err});
}

/// on_change: fired every drag-motion frame. The meter already mutated that
/// host's handles in place; motion repaints on its own, so this is a no-op (we
/// deliberately do NOT reshuffle VRAM mid-drag, only on release).
fn meterChanged() void {}

/// on_commit: fired on drag release. Persist the settled fractions where that
/// host's values live, and send them to that host alone, which applies the new
/// policy to its live session.
fn meterCommit() void {
    const s = g_hosts.slotOf(g_meter_host) orelse return;
    if (s.entry == null) {
        // The local child's handles are the settings' own pair.
        g_config.vram_split = s.meter.split;
        g_config.vram_limit_frac = s.meter.limit;
        g_config_baseline.vram_split = s.meter.split;
        g_config_baseline.vram_limit_frac = s.meter.limit;
    } else {
        // A listed host keeps its handles on its own entry. The baseline gets
        // the same write, so a meter drag never reads as an unsaved edit.
        if (g_config.hostEntryMut(s.name)) |e| {
            e.vram_split = s.meter.split;
            e.vram_limit_frac = s.meter.limit;
        }
        if (g_config_baseline.hostEntryMut(s.name)) |e| {
            e.vram_split = s.meter.split;
            e.vram_limit_frac = s.meter.limit;
        }
        if (s.entry) |*entry| {
            entry.vram_split = s.meter.split;
            entry.vram_limit_frac = s.meter.limit;
        }
    }
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    s.postMeter();
}

/// Restore a persisted window geometry onto a freshly created SDL window. Size
/// is applied first, then position (only when one was saved, otherwise SDL's
/// default placement stands), then maximize last so it overrides the rect.
fn applyWindowGeom(window: ?*SDLBackend.c.SDL_Window, w: usize, h: usize, x: i32, y: i32, maximized: bool) void {
    _ = SDLBackend.c.SDL_SetWindowSize(window, @intCast(w), @intCast(h));
    if (x != config.pos_unset and y != config.pos_unset)
        _ = SDLBackend.c.SDL_SetWindowPosition(window, x, y);
    if (maximized) _ = SDLBackend.c.SDL_MaximizeWindow(window);
}

/// Read a window's current geometry into the given config fields, returning
/// whether anything changed. While the window is maximized (or minimized) the
/// size/position are left alone, the stored values keep the last *restored*
/// geometry so un-maximizing (and the next launch) lands on a sensible rect;
/// only the maximized flag tracks the transition.
fn captureGeom(window: ?*SDLBackend.c.SDL_Window, w: *usize, h: *usize, x: *i32, y: *i32, maximized: *bool) bool {
    var changed = false;
    const flags = SDLBackend.c.SDL_GetWindowFlags(window);
    const now_max = (flags & SDLBackend.c.SDL_WINDOW_MAXIMIZED) != 0;
    if (now_max != maximized.*) {
        maximized.* = now_max;
        changed = true;
    }
    if (now_max or (flags & SDLBackend.c.SDL_WINDOW_MINIMIZED) != 0) return changed;

    var cw: c_int = 0;
    var ch: c_int = 0;
    _ = SDLBackend.c.SDL_GetWindowSize(window, &cw, &ch);
    if (cw > 0 and ch > 0) {
        const nw: usize = @intCast(cw);
        const nh: usize = @intCast(ch);
        if (nw != w.* or nh != h.*) {
            w.* = nw;
            h.* = nh;
            changed = true;
        }
    }
    var cx: c_int = 0;
    var cy: c_int = 0;
    _ = SDLBackend.c.SDL_GetWindowPosition(window, &cx, &cy);
    const nx: i32 = @intCast(cx);
    const ny: i32 = @intCast(cy);
    if (nx != x.* or ny != y.*) {
        x.* = nx;
        y.* = ny;
        changed = true;
    }
    return changed;
}

/// Persist window geometry that changed this frame. Geometry is pure view state,
/// so we write the committed baseline (not the live `g_config`, which may hold
/// mid-edit Settings text) with the current geometry overlaid, this keeps
/// Settings -> Cancel able to discard unsaved edits while a concurrent resize
/// still sticks.
fn saveGeometry() void {
    g_config_baseline.win_w = g_config.win_w;
    g_config_baseline.win_h = g_config.win_h;
    g_config_baseline.win_x = g_config.win_x;
    g_config_baseline.win_y = g_config.win_y;
    g_config_baseline.win_max = g_config.win_max;
    g_config_baseline.viewer_w = g_config.viewer_w;
    g_config_baseline.viewer_h = g_config.viewer_h;
    g_config_baseline.viewer_x = g_config.viewer_x;
    g_config_baseline.viewer_y = g_config.viewer_y;
    g_config_baseline.viewer_max = g_config.viewer_max;
    g_config_baseline.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save window geometry failed: {t}", .{err});
}

// The eject and pause buttons live on a host's own bar, so they act on that
// host rather than on whoever carries the chat.
fn meterEjectLlm() void {
    if (g_hosts.slotOf(g_meter_host)) |s| s.post(.chat_eject); // fires once the model is idle
}
fn meterEjectDiff() void {
    if (g_hosts.slotOf(g_meter_host)) |s| s.post(.img_eject);
}
fn meterActions() meter.Actions {
    return .{ .on_change = meterChanged, .on_commit = meterCommit, .on_eject_llm = meterEjectLlm, .on_eject_diff = meterEjectDiff, .on_toggle_pause_llm = toggleLlmPause, .on_toggle_pause_diff = toggleDiffPause };
}

// Conversation history: the on-disk store plus the sidebar's selection state.
// Saved after every completed turn (see maybeSaveHistory) and on exit.
var g_history: history.Store = .{};
/// The image studio's prompt library, beside the transcripts. Text only: a row
/// restores what was written, never the knobs it was written with.
var g_prompts: prompt_history.Store = .{};
/// A sidebar row was clicked; the load happens at the top of the next frame,
/// never mid-render, because it swaps the session's whole transcript.
var g_load_conv: ?u64 = null;
/// Transcript size at the last save, so an idle frame does not rewrite the file
/// 60 times a second.
var g_saved_turns: usize = 0;
var g_saved_tail: usize = 0;
/// Renders on disk at the last save. A turn's images finish AFTER its reply, so
/// without this the newest turn was saved before its files existed and a reload
/// reported them missing until some later turn re-saved the conversation.
var g_saved_images: usize = 0;
/// The mirror's transcript revision when this client handed the host a
/// different conversation. Until it moves, `g_m.messages` still holds the
/// PREVIOUS one while `g_history.current` already names the new file, and a save
/// in that window writes one conversation into the other's file.
var g_handover_rev: ?u64 = null;

var g_rail_tab: queue_rail.Tab = .queue;

// Persistent settings + which full-window view is showing. `g_config_path`
// (from `--config`) overrides the well-known settings-file location; null uses
// the platform config dir.
var g_config: config.Config = .{};
// The config as of the last load/apply, to diff against on the next Apply: only
// a change to a load-affecting field (model path or VRAM limit) forces a session
// reload; everything else is applied live so saving settings never wipes the chat.
var g_config_baseline: config.Config = .{};
var g_config_path: ?[]const u8 = null;
var g_view: View = .chat;
const View = enum { chat, config, image };
// Where Settings returns to (chat or image studio), so opening the gear from
// the studio comes back to the studio, not chat.
var g_return_view: View = .chat;

// Process-lifetime handles the loader thread and config save need.
var g_io: std.Io = undefined;
var g_gpa: std.mem.Allocator = undefined;
var g_environ: *std.process.Environ.Map = undefined;

/// The composer's text. Growable, because dvui's entry silently stops accepting
/// input at the end of a fixed buffer; `input_limit` is the cap instead, and it
/// bounds the per-frame text layout as much as the memory.
var g_input: std.ArrayList(u8) = .empty;
const input_limit = 64 << 10;
var g_wakeup_event_type: u32 = 0;
// Message-list scroll state (persistent so we can follow streaming output).
// g_follow_bottom sticks the view to the newest content; it turns off when the
// user scrolls up and back on when they return to the bottom. g_prev_offset
// tracks last frame's scroll offset to tell a user scroll-up apart from the
// offset drift caused by content growing.
var g_scroll_info: dvui.ScrollInfo = .{};
var g_follow_bottom: bool = true;
var g_prev_offset: f32 = 0;
// The input text entry's id (for Enter-to-send focus check) and measured
// height (so the message list reserves the right amount as the box grows).
var g_input_id: ?dvui.Id = null;
var g_input_h: f32 = 52;
// Full-size image viewer (a second window). g_viewer_request is set when an
// image is clicked; the main loop opens/refocuses the viewer.
var g_viewer: ?*viewer.Viewer = null;
var g_viewer_request: ?wire.ImageId = null;

/// Pushed from worker threads (via the token sink) to unblock
/// `waitEventTimeout` so streamed tokens repaint promptly.
fn wakeupFrame() void {
    var ev: SDLBackend.c.SDL_Event = std.mem.zeroes(SDLBackend.c.SDL_Event);
    ev.type = g_wakeup_event_type;
    _ = SDLBackend.c.SDL_PushEvent(&ev);
}

pub fn run(init: std.process.Init) !void {
    dvui.App.main_init = init;
    SDLBackend.c.SDL_SetMainReady();

    const gpa = init.gpa;
    const arena = init.arena.allocator();
    g_gpa = std.heap.smp_allocator;
    g_io = init.io;
    g_environ = init.environ_map;

    // Parse CLI. `--config <path>` overrides the settings-file location (handy
    // for testing without touching the user's real config); `--model <path>`
    // overrides the saved LLM path for this run (not persisted). Args live in
    // the process arena, so the slices are stable for the whole session.
    const args = try init.minimal.args.toSlice(arena);
    var model_override: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--config") and i + 1 < args.len) {
            i += 1;
            g_config_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_override = args[i];
        }
    }

    // Load persisted settings (all fields default to unset / disabled).
    const first_run = !config.Config.exists(init.io, gpa, init.environ_map, g_config_path);
    g_config = config.Config.load(init.io, gpa, init.environ_map, g_config_path);
    // The compiled-in backend default is NVIDIA's, which is a failed first launch
    // on any other machine. With no settings file yet, take what the box has.
    if (first_run) {
        const be = pipeline_map.fromPipelineBackend(tp.pipeline.detectBackend());
        g_config.llm_backend = be;
        g_config.diff_backend = be;
        std.log.info("no settings file yet: defaulting both backends to {t}", .{be});
    }
    if (model_override) |m| g_config.llm_model.set(m);
    g_sync = sync.Syncer.init(g_gpa, g_io);
    g_hosts = try hosts.Hosts.init(g_gpa, g_io, g_environ, wakeupFrame, g_config_path, null);
    g_m = g_hosts.chatMirror();
    g_models = models.empty(g_gpa);
    // The model folders come from the config (seeded from the configured files'
    // folders the first time); each host scans them and its catalog lands in its
    // mirror. A config from before the catalog existed keeps its exact selection.
    g_config.seedModelDirs();
    selection.resolveAll(&g_config, &g_models.cat);
    g_config_baseline = g_config;
    // Seed the meter handles from the persisted fractions, clamped into the
    // grabbable range (recovers a config that saved a stuck limit at the edge).
    g_config.vram_split = std.math.clamp(g_config.vram_split, 0.02, 0.96);
    g_config.vram_limit_frac = std.math.clamp(g_config.vram_limit_frac, 0.10, 0.985);

    var back = try SDLBackend.initWindow(.{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = @floatFromInt(g_config.win_w), .h = @floatFromInt(g_config.win_h) },
        .min_size = .{ .w = 640, .h = 480 },
        .vsync = true,
        .title = "TensorPencil",
        .environ_map = init.environ_map,
    });
    defer back.deinit();
    // Restore the saved position / maximized state (size is already set above).
    applyWindowGeom(back.window, g_config.win_w, g_config.win_h, g_config.win_x, g_config.win_y, g_config.win_max);

    var win = try dvui.Window.init(@src(), gpa, back.backend(), .{});
    defer win.deinit();

    // Register the bundled broad-coverage font in a bootstrap frame (addFont /
    // themeSet need the current window) so CJK / symbols in LLM output render
    // instead of tofu boxes.
    try win.begin(win.frame_time_ns);
    style.install(); // fonts + palette; both need the current window
    _ = try win.end(.{});

    g_wakeup_event_type = SDLBackend.c.SDL_RegisterEvents(1);

    // The LLM is NOT loaded at startup, it lazy-loads on the first chat message
    // (see submitChat). Build the app-level diffusion engine now if a model is
    // configured (its pipeline still loads lazily on the first image).
    // Derive width/height from the persisted framing before anything reads
    // them: they are a cache of (ratio, megapixels), and an older config may
    // carry a pair that predates the framing fields.
    g_config.applyFraming();
    g_config_baseline.width = g_config.width;
    g_config_baseline.height = g_config.height;

    image_view.setEnv(g_gpa, g_io, wakeupFrame);
    config_view.setEnv(back.window, wakeupFrame, g_gpa, g_io);
    // The hosts: a tp-serve already listening at the local socket, else one
    // spawned now, plus every host the settings list. Connects run on their
    // own threads; a host that is not there yet is retried from `pump`.
    // Each starts on default settings and takes ours; every later change is
    // a settings request from `commitConfig`. A spawned child is told our
    // `--config` so its catalog cache sits beside the same file.
    g_hosts.sync(&g_config);
    g_m = g_hosts.chatMirror();

    // Conversation history lives beside the config file. With `--config <path>`
    // it goes next to THAT file, so a throwaway config gets a throwaway history
    // and a test run never writes into the real one.
    openHistory();

    // Tear down at exit. Stop the diffusion engine FIRST (join its worker) so no
    // diffusion thread is still touching a transcript/gallery image as those are
    // freed; then the LLM, then the gallery.
    defer {
        // The last events land in the mirror, which the flush below writes
        // from; then the host goes (a spawned one exits with us).
        pumpHost();
        saveHistory(true);
        g_history.deinit(g_gpa);
        g_prompts.deinit(g_gpa);
        g_sync.deinit();
        g_models.deinit();
        g_hosts.deinit();
        if (g_pending_text) |p| g_gpa.free(p);
        clearStaged();
        g_staged.deinit(g_gpa);
        if (g_family_cache) |*c| c.deinit();
        image_view.deinit();
        config_view.deinit();
    }
    defer if (g_viewer) |v| v.deinit();

    var interrupted = false;
    main_loop: while (true) {
        pumpHost();
        // A finished folder scan may have found a side file for a slot that had
        // none; resolving fills it and, if that changed a path, applies it. Any
        // host's scan can be the one that did, so the merged view is rebuilt
        // first and the selection resolved against that.
        if (g_hosts.modelSeq() != g_model_seq) {
            g_model_seq = g_hosts.modelSeq();
            rebuildModels();
            selection.resolveAll(&g_config, &g_models.cat);
            if (!g_config.llmReloadEql(&g_config_baseline) or !g_config.diffPathsEql(&g_config_baseline)) commitConfig();
        }
        const nstime = win.beginWait(interrupted);

        // Pump SDL events once, routing each to the window it targets (main or
        // the viewer). File drops (main only) are intercepted here since dvui
        // doesn't surface them.
        var event: SDLBackend.c.SDL_Event = undefined;
        while (SDLBackend.c.SDL_PollEvent(&event)) {
            if (event.type == SDLBackend.c.SDL_EVENT_DROP_FILE) {
                if (event.drop.data != null) handleDropFile(std.mem.span(event.drop.data));
                continue;
            }
            const wid = sdlEventWindowID(event);
            if (g_viewer) |v| {
                if (wid == v.win_id) {
                    _ = v.back.addEvent(&v.win, event) catch {};
                    continue;
                }
            }
            _ = back.addEvent(&win, event) catch {};
        }

        // ── Main window ──────────────────────────────────────────────────
        // A slot is pinned only while a view is actually drawing that picture:
        // both are re-asserted below by whatever renders. Cleared here rather
        // than at each teardown, since a view can stop drawing one without any
        // teardown to hang the clear on.
        g_shown = .{ null, null };
        try win.begin(nstime);
        _ = SDLBackend.c.SDL_SetRenderDrawColor(back.renderer, 0, 0, 0, 255);
        _ = SDLBackend.c.SDL_RenderClear(back.renderer);
        frame();
        var res: dvui.App.Result = .ok;
        for (dvui.events()) |*e| {
            if (e.handled) continue;
            if (e.evt == .window and e.evt.window.action == .close) res = .close;
            if (e.evt == .app and e.evt.app.action == .quit) res = .close;
        }
        var end_micros = try win.end(.{});
        try back.setCursor(win.cursorRequested());
        try back.textInputRect(win.textInputRequested());
        try back.renderPresent();
        if (res != .ok) break :main_loop;

        // A clicked image (chat transcript or studio gallery) opens/refocuses
        // the viewer, which navigates the engine's unified image history.
        var vreq: ?wire.ImageId = null;
        if (g_viewer_request) |id| {
            g_viewer_request = null;
            vreq = id;
        } else if (image_view.viewer_request) |id| {
            image_view.viewer_request = null;
            vreq = id;
        }
        const vsrc = diffuserSource();
        if (vreq) |id| {
            if (g_viewer) |v| {
                v.setImage(id);
                _ = SDLBackend.c.SDL_RaiseWindow(v.back.window);
            } else {
                g_viewer = viewer.Viewer.init(init.gpa, init.io, vsrc, id) catch |err| vblk: {
                    std.log.err("open viewer failed: {t}", .{err});
                    break :vblk null;
                };
                // Restore the viewer window's saved geometry (created hidden, so
                // this lands before it's first shown, no flash).
                if (g_viewer) |v| applyWindowGeom(v.back.window, g_config.viewer_w, g_config.viewer_h, g_config.viewer_x, g_config.viewer_y, g_config.viewer_max);
            }
        }

        // ── Viewer window ────────────────────────────────────────────────
        // Closed (window's X, or its image no longer resolves): tear down
        // without rendering.
        if (g_viewer) |v| if (!v.open) {
            v.deinit();
            g_viewer = null;
        };
        if (g_viewer) |v| {
            try v.win.begin(nstime);
            _ = SDLBackend.c.SDL_SetRenderDrawColor(v.back.renderer, 0, 0, 0, 255);
            _ = SDLBackend.c.SDL_RenderClear(v.back.renderer);
            v.render();
            for (dvui.events()) |*e| {
                if (!e.handled and e.evt == .window and e.evt.window.action == .close) v.open = false;
            }
            const v_end = try v.win.end(.{});
            try v.back.setCursor(v.win.cursorRequested());
            try v.back.textInputRect(v.win.textInputRequested());
            try v.back.renderPresent();
            if (!v.shown) {
                v.shown = true;
                _ = SDLBackend.c.SDL_ShowWindow(v.back.window);
            }
            end_micros = pickMinWait(end_micros, v_end);
            if (!v.open) {
                v.deinit();
                g_viewer = null;
            }
        }

        // Persist any window move/resize/maximize that happened this frame. Both
        // windows are checked; coalesced to at most one config write per frame.
        var geom_changed = captureGeom(back.window, &g_config.win_w, &g_config.win_h, &g_config.win_x, &g_config.win_y, &g_config.win_max);
        if (g_viewer) |v| {
            if (captureGeom(v.back.window, &g_config.viewer_w, &g_config.viewer_h, &g_config.viewer_x, &g_config.viewer_y, &g_config.viewer_max)) geom_changed = true;
        }
        if (geom_changed) saveGeometry();

        const wait_micros = win.waitTime(end_micros);
        interrupted = try back.waitEventTimeout(wait_micros);
    }
}

fn pickMinWait(a: ?u32, b: ?u32) ?u32 {
    if (a == null) return b;
    if (b == null) return a;
    return @min(a.?, b.?);
}

fn sdlEventWindowID(event: SDLBackend.c.SDL_Event) u32 {
    const SDL = SDLBackend.c;
    return switch (event.type) {
        SDL.SDL_EVENT_KEY_DOWN, SDL.SDL_EVENT_KEY_UP => event.key.windowID,
        SDL.SDL_EVENT_TEXT_INPUT, SDL.SDL_EVENT_TEXT_EDITING => event.text.windowID,
        SDL.SDL_EVENT_MOUSE_MOTION => event.motion.windowID,
        SDL.SDL_EVENT_MOUSE_BUTTON_DOWN, SDL.SDL_EVENT_MOUSE_BUTTON_UP => event.button.windowID,
        SDL.SDL_EVENT_MOUSE_WHEEL => event.wheel.windowID,
        SDL.SDL_EVENT_WINDOW_RESIZED,
        SDL.SDL_EVENT_WINDOW_FOCUS_GAINED,
        SDL.SDL_EVENT_WINDOW_FOCUS_LOST,
        SDL.SDL_EVENT_WINDOW_MOUSE_ENTER,
        SDL.SDL_EVENT_WINDOW_MOUSE_LEAVE,
        SDL.SDL_EVENT_WINDOW_CLOSE_REQUESTED,
        SDL.SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED,
        => event.window.windowID,
        else => 0,
    };
}

/// TP_AUTO_IMAGE: render one image through the REAL app path and exit.
///
/// Same argument as `autoMessage`: the CLI's `generate` builds its own
/// `Options`, so it exercises none of what the app wraps around the engine, and
/// the settings-to-engine chain (`selection` -> `modelConfigFromConfig` ->
/// `requestPaths` -> `ModelConfig.applyTo`) is exactly where a configured LoRA
/// would go missing. Run with no display (`DISPLAY= SDL_VIDEODRIVER=dummy`).
///
/// Exits non-zero when the render fails, so a harness cannot mistake a failure
/// for a slow success.
var g_auto_image_sent: bool = false;
/// The queue reference the probe's one render was given. The queue mints it,
/// so it cannot be chosen here.
var g_auto_image_ref: u64 = 0;
fn autoImage() void {
    const prompt_z = getenv("TP_AUTO_IMAGE") orelse return;
    if (!g_m.state.diff_present) {
        if (g_config.diffusion_model.opt() == null) {
            std.log.err("[auto] TP_AUTO_IMAGE with no diffusion_model configured", .{});
            std.process.exit(2);
        }
        return; // the host has not reported its engine yet
    }
    if (!g_auto_image_sent) {
        g_auto_image_sent = true;
        g_auto_image_ref = g_hosts.enqueueImage(&g_config, .{
            .prompt = std.mem.span(prompt_z),
            .width = @intCast(g_config.width),
            .height = @intCast(g_config.height),
            .steps = @intCast(g_config.steps),
            .cfg = 1.0,
            .seed = 1234,
            .from_studio = true,
        });
        return;
    }
    if (g_auto_image_ref == 0) return;
    // Done when the image reports a terminal status AND its pixels have been
    // fetched and (if saving is on) written, which is the whole client path.
    for (g_hosts.slots.items) |slot| for (slot.mirror.images.items) |*im| {
        if (im.info.client_ref != g_auto_image_ref) continue;
        switch (im.status()) {
            .done => {
                if (im.pixels == null) return;
                if (g_config.output_dir.opt() != null and !im.save_tried) return;
                std.log.info("[auto] image done (ok)", .{});
                std.process.exit(0);
            },
            .failed, .canceled => {
                std.log.info("[auto] image done (FAILED: {s})", .{im.info.failure});
                std.process.exit(1);
            },
            else => return,
        }
    };
}

/// TP_AUTO_MESSAGE: send one message through the REAL app path as soon as a
/// session exists, then exit once it finishes. `chat-probe` drives the session
/// directly and so misses everything the app wraps around it (the diffuser, the
/// VRAM arbiter, per-frame `updateSettings`); this reproduces a turn with all of
/// that in place, and pairs with TP_DUMP_REPLY to show what was generated.
/// Run it with no display (`DISPLAY= SDL_VIDEODRIVER=dummy`) so no window opens.
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
var g_auto_sent: bool = false;
fn autoMessage() void {
    const msg_z = getenv("TP_AUTO_MESSAGE") orelse return;
    if (!g_auto_sent) {
        g_auto_sent = true;
        // Same entry point the send button uses. The button can afford to drop a
        // message on the floor; here there is no one to notice, so a refusal has
        // to be loud, otherwise the harness spins until it is killed.
        if (!submitChat(std.mem.span(msg_z))) {
            std.log.err("[auto] message went nowhere (empty, no llm_model configured, or session refused it)", .{});
            std.process.exit(2);
        }
        return;
    }
    // Done when the host reports the turn ended and the reply is mirrored.
    if (g_m.turns_ended == 0) return;
    const msgs = g_m.messages.items;
    if (msgs.len == 0) return;
    const last = &msgs[msgs.len - 1];
    if (last.role != .assistant) return;
    const v = last.active();
    std.log.info("[reply] {d} bytes, thought_len={d}\n{s}\n[reply] end", .{
        v.text.items.len, thoughtLen(v), v.text.items,
    });
    std.process.exit(0);
}

/// Characters of reasoning in a finished take, split with its own markers.
fn thoughtLen(v: *const mirror.Variant) usize {
    const s2 = toolcall.splitThought(v.text.items, markersFor(v), v.thought_primed);
    return if (s2.think) |t| std.mem.trim(u8, t, " \t\r\n").len else 0;
}

/// The markers to split a take with: the ones recorded when it was generated,
/// else the resident model's, else none. With nothing loaded we deliberately
/// do NOT guess: a wrong split silently eats or invents part of a reply.
fn markersFor(v: *const mirror.Variant) ?toolcall.Reasoning {
    if (v.reason_open.len > 0 and v.reason_close.len > 0) return .{ .open = v.reason_open, .close = v.reason_close };
    const st = &g_m.state;
    if (st.reason_open.len > 0 and st.reason_close.len > 0) return .{ .open = st.reason_open, .close = st.reason_close };
    return null;
}

/// A file was dropped on the window: decode it (libvips -> RGB) and attach it
/// to the next message for the model to see.
fn handleDropFile(path: []const u8) void {
    if (g_m.state.loading) return; // the session is being rebuilt
    if (!g_m.state.vision) {
        std.log.warn("dropped {s} but vision is unavailable", .{path});
        return;
    }
    const gpa = std.heap.smp_allocator;
    const dec = vips.loadRgb(gpa, path) catch |err| {
        std.log.err("can't load dropped image {s}: {t}", .{ path, err });
        return;
    };
    defer gpa.free(dec.pixels);
    attachImage(dec.pixels, dec.width, dec.height);
}

/// Ctrl/Cmd+V with an image on the clipboard: decode the raw bytes (any
/// libvips format) and attach it, exactly as a dropped file. Returns true
/// when an image was found on the clipboard (whether or not decoding
/// succeeded), so the caller can consume the event before the text entry
/// treats it as a text paste. Returns false when the clipboard holds no
/// image, letting normal text paste proceed.
fn tryPasteClipboardImage() bool {
    const SDL = SDLBackend.c;
    if (g_m.state.loading) return false; // the session is being rebuilt
    if (!g_m.state.vision) return false;

    var count: usize = 0;
    const mimes = SDL.SDL_GetClipboardMimeTypes(&count);
    if (mimes == null) return false;
    defer SDL.SDL_free(@ptrCast(mimes));

    var mime: [*c]const u8 = null;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const m = mimes[i];
        if (m == null) continue;
        if (std.mem.startsWith(u8, std.mem.span(m), "image/")) {
            mime = m;
            break;
        }
    }
    if (mime == null) return false;

    var size: usize = 0;
    const data = SDL.SDL_GetClipboardData(mime, &size);
    if (data == null or size == 0) return true;
    defer SDL.SDL_free(data);

    const bytes = @as([*]const u8, @ptrCast(data.?))[0..size];
    const gpa = std.heap.smp_allocator;
    const dec = vips.loadRgbFromMemory(gpa, bytes) catch |err| {
        std.log.err("can't decode pasted image ({s}): {t}", .{ std.mem.span(mime), err });
        return true;
    };
    defer gpa.free(dec.pixels);
    attachImage(dec.pixels, dec.width, dec.height);
    return true;
}

fn frame() void {
    // Whatever the host pump learned between frames, now that there is a window.
    toast.pump();
    autoMessage(); // TP_AUTO_MESSAGE; before any early return so it always runs
    autoImage(); // TP_AUTO_IMAGE, likewise

    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = style.C.canvas,
    });
    defer root.deinit();

    // Settings is the one view that takes the whole window: it is a mode, not a
    // place in the workspace, and it has its own way back.
    if (g_view == .config) {
        // The view is handed a config, not a session, so the capability answer is
        // pushed rather than asked for. Same answer the composer gates on.
        config_view.g_noise_supported = g_m.state.weight_noise;
        config_view.render(&g_config, g_m, &g_models, .{
            .apply = applyConfig,
            .cancel = cancelConfig,
            .rescan = rescan,
            .hostStatus = hostStatus,
            .hostSync = hostSync,
            .sendModel = sendModel,
            .sendPath = sendPath,
            .pullPath = pullPath,
            .tryHost = tryHost,
            .reconnectHost = reconnectHost,
        });
        return;
    }

    const bands = shell.Bands.from(root, @as(f32, @floatFromInt(@max(g_hosts.slots.items.len, 1))) * status_bar.bar_outer_height);
    {
        const arena = dvui.currentWindow().arena();
        const llm_resident = g_m.state.llm_resident;
        const t = &g_m.telemetry;
        const image_resident = t.diff_te + t.diff_dit + t.diff_latent + t.diff_vae > 0;
        shell.titleBar(.{
            .tab = if (g_view == .image) .studio else .chat,
            .llm = model_lib.llmChip(arena, &g_config, &g_models, llm_resident),
            .llm_menu = model_lib.llmMenu(arena, &g_config, &g_models),
            .image = model_lib.imageChip(arena, &g_config, &g_models, image_resident),
            .image_menu = model_lib.imageMenu(arena, &g_config, &g_models),
        }, .{
            .on_tab = onTabPicked,
            .on_llm_pick = onLlmPick,
            .on_image_pick = onImagePick,
            .on_settings = openSettings,
        });
    }

    {
        var body = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = bands.body },
            .max_size_content = .height(bands.body),
        });
        defer body.deinit();

        if (g_view == .image) studioBody() else chatBody();
    }

    // One bar per host, in list order (the local child first). Each carries
    // its own card's meter and its own handles.
    dropStaleViews();
    const several = g_hosts.several();
    for (g_hosts.slots.items, 0..) |slot, i| {
        g_meter_host = slot.id;
        status_bar.render(
            viewFor(slot.id),
            &slot.mirror,
            if (several) slot.name else "",
            hosts.Hosts.troubleOf(slot) orelse "",
            if (g_hosts.whyNotRender(slot, &g_config)) |w| w.short else "",
            i,
            &slot.meter.split,
            &slot.meter.limit,
            meterActions(),
        );
    }
}

/// The studio workspace body: prompt rail | canvas + form + composer | queue rail.
///
/// The SAME three bands as chat, at the same widths, so switching tabs does not
/// reflow the window; only the left rail's contents and the middle column
/// differ. The queue rail is shared outright: an image belongs in exactly one
/// place, and Library is where a finished one lives whichever tab made it.
fn studioBody() void {
    recordStudioPrompt();
    renderPromptSidebar();

    {
        // Computed, not left to the box layout, for the reason chatBody states:
        // a child's min size propagates up, so one long unbroken line grows the
        // column and squeezes both rails to slivers.
        const band_w = dvui.parentGet().data().contentRect().w;
        const col_w = @max(320, band_w - style.Layout.sidebar_w - style.Layout.rail_w);
        var col = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .vertical,
            .min_size_content = .{ .w = col_w },
            .max_size_content = .width(col_w),
        });
        defer col.deinit();

        // The host that would take the next render, not the chat host: its
        // models, its queue and its load errors are what this form describes.
        const target = g_hosts.renderTarget(&g_config, @intCast(g_config.width), @intCast(g_config.height), @intCast(g_config.steps));
        image_view.render(&g_config, &target.mirror, &g_models, studioImages(), post, !target.mirror.state.loading, .{
            .settings = openSettings,
            .save_defaults = saveStudioDefaults,
            .cancel = onCancelImage,
        });
    }

    renderQueueRail();
}

/// "Save as defaults": copy the studio form's recipe into the config the form
/// (and the chat tool path) seed from. The one route by which a value the studio
/// edits reaches config, and it is explicit.
fn saveStudioDefaults() void {
    image_view.saveDefaults(&g_config, configuredFamily());
    // Both dependent views seed their buffers once, so both have to be told the
    // numbers under them moved. `commitConfig` saves and reconciles the engines.
    image_view.reseed();
    config_view.reseed();
    commitConfig();
}

/// The chat workspace body: sidebar | transcript + composer | queue rail.
fn chatBody() void {
    // A background (re)load never takes over the screen: the layout stays put,
    // the just-sent message shows as a normal user bubble, and the assistant
    // slot shows a small "Loading…" until the session is live (see
    // renderMessages). No spinner flashing in and out.
    const st = &g_m.state;
    const loading = st.loading;

    // The "no model" notice replaces the TRANSCRIPT, not the workspace: the
    // rails still show, so the queue and the settings shortcut stay reachable
    // while nothing is configured. True when there is genuinely nothing set, or
    // when the last load failed (e.g. a non-mmproj file in the vision-tower
    // slot) and there is no working session at all. During a load `loading` is
    // set, so the transiently-absent session never trips it.
    const no_model = !loading and !st.llm_resident and
        (g_config.llm_model.opt() == null or st.load_err.len > 0);

    applyPendingConversationLoad();
    saveHistory(false);

    renderSidebar();

    {
        // The column's width is COMPUTED from the band, not left to the box
        // layout. `min_size_content.w = 0` is not enough: a child's min size
        // still propagates up, so one long unbroken line (a tool call, a URL)
        // grows the column and squeezes the rails to slivers. Pinning both ends
        // makes the rails' widths the fixed quantity they are supposed to be.
        const band_w = dvui.parentGet().data().contentRect().w;
        const col_w = @max(240, band_w - style.Layout.sidebar_w - style.Layout.rail_w);
        var col = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .vertical,
            .min_size_content = .{ .w = col_w },
            .max_size_content = .width(col_w),
        });
        defer col.deinit();

        if (no_model) {
            renderNoModel();
        } else {
            // Pin the composer to the bottom: cap the transcript's height to
            // what is left. A scrollArea reports its full content height as its
            // min size, so as a plain flex child it would push the composer
            // off-screen.
            const list_h = @max(120, col.data().contentRect().h - g_input_h);
            renderMessages(list_h);
            renderInput();
        }
    }

    renderQueueRail();
}

/// Build the queue rail's model from the live engine. The Queue tab shows what
/// is still waiting for a host, plus what failed; a render in motion is drawn
/// where its pixels are (the studio canvas, or the transcript's tool card). The
/// Library tab shows every finished one, newest first.
fn renderQueueRail() void {
    var jobs_buf: [16]queue_rail.Job = undefined;
    var lib_buf: [24]queue_rail.LibraryItem = undefined;
    var titles: [16][80]u8 = undefined;
    var n_jobs: usize = 0;
    var n_lib: usize = 0;

    // What no host has taken yet, first: it is the front of the queue, and a
    // job invisible until a machine picks it up looks like a click that did
    // nothing. Only these can be reordered; the rest are already somewhere.
    var waiting_buf: [16]*const hosts.Hosts.Asked = undefined;
    var notes: [16][64]u8 = undefined;
    for (g_hosts.waiting(&waiting_buf)) |a| {
        if (n_jobs >= jobs_buf.len) break;
        jobs_buf[n_jobs] = .{
            .id = a.ref,
            .title = style.ellipsize(&titles[n_jobs], a.req.prompt, style.F.row_hi, 150),
            .state = .{ .queued = .{ .note = waitNote(&notes[n_jobs], a) } },
            .from_studio = a.req.from_studio,
            .host = "",
        };
        n_jobs += 1;
    }

    // Then what a host has taken but not started, oldest first however many
    // hosts they are spread over; a job names its host once there is more than
    // one. A render under way is drawn where its pixels are, never here: in the
    // studio that is the canvas, in chat the tool card's tile.
    var run_buf: [16]hosts.Hosts.Shot = undefined;
    for (g_hosts.running(&run_buf)) |sh| {
        if (n_jobs >= jobs_buf.len) break;
        const im = sh.im;
        switch (im.status()) {
            .generating, .suspended => continue,
            else => {},
        }
        jobs_buf[n_jobs] = .{
            .id = im.info.id,
            .title = railTitle(&titles[n_jobs], im),
            .state = switch (im.status()) {
                // A render that failed keeps its row and says why. Dropping it
                // leaves the user watching a job disappear.
                .failed => .{ .failed = .{ .why = mirror.failureText(im.info.failure) } },
                else => .{ .queued = .{ .eta_s = null } },
            },
            .from_studio = im.info.from_studio,
            .host = sh.host,
            .draggable = false,
        };
        n_jobs += 1;
    }

    // Finished images, newest first across every host. Only the Library tab
    // draws these, so nothing here is also sitting inline in the transcript.
    var lib_shots: [lib_buf.len]hosts.Hosts.Shot = undefined;
    for (g_hosts.finished(&lib_shots)) |sh| {
        lib_buf[n_lib] = .{ .id = sh.im.info.id, .thumb = doneThumb(sh.im), .host = sh.host };
        n_lib += 1;
    }

    queue_rail.render(.{
        .tab = g_rail_tab,
        .jobs = jobs_buf[0..n_jobs],
        .library = lib_buf[0..n_lib],
        .paused = diffPaused(),
    }, .{
        .on_tab = onRailTab,
        .on_pause_all = toggleDiffPauseEverywhere,
        .on_open_library = onOpenLibraryImage,
        .on_cancel = onCancelImage,
        .on_retry = onRetryImage,
        .on_reorder = onReorderQueue,
    });
}

/// A queue row's title: the first line of the prompt, since that is what the
/// user recognizes. Truncation is left to the rail, which knows its own width.
fn railTitle(buf: []u8, im: *const mirror.Image) []const u8 {
    const p = std.mem.trim(u8, im.info.prompt, " \t\r\n");
    const line = p[0 .. std.mem.indexOfScalar(u8, p, '\n') orelse p.len];
    if (line.len == 0) return "untitled";
    const n = @min(line.len, buf.len);
    @memcpy(buf[0..n], line[0..n]);
    return buf[0..n];
}

/// Why a waiting render has not gone out, in the rail's few words.
fn waitNote(buf: []u8, a: *const hosts.Hosts.Asked) []const u8 {
    return switch (g_hosts.waitReason(&g_config, a)) {
        .ready => "",
        .behind => |host| if (host.len == 0) "" else std.fmt.bufPrint(buf, "waiting for {s}", .{host}) catch "",
        .nowhere => "no host can run this yet",
    };
}

fn doneThumb(im: *const mirror.Image) queue_rail.Thumb {
    if (im.thumb()) |t| return .{ .rgba = .{ .px = t.px, .w = t.w, .h = t.h } };
    return .loading;
}

// ----------------------------------------------------------- tool-call card

/// Which card has its prompt open, and which tile is selected in it, both keyed
/// by where the run sits in the transcript. One card at a time: an accordion
/// keeps the transcript from growing several screens of prompt text at once.
var g_card_open: ?usize = null;
var g_card_sel_key: ?usize = null;
var g_card_sel_tile: usize = 0;
/// The image run the in-flight callbacks below act on, set immediately before
/// the card renders (dvui callbacks fire inside the same call).
var g_card_imgs: []const wire.ImageId = &.{};

/// This frame's image behind an id, or null once it is gone.
fn imageById(id: wire.ImageId) ?*mirror.Image {
    return g_hosts.imageById(id);
}

/// A card's identity, which it carries through dvui as an opaque context
/// pointer: where the run sits in the transcript, offset so it cannot be null.
/// Not the first image's id, which a retry replaces with one minted elsewhere.
fn cardKey(msg: usize, variant: usize, at: usize) usize {
    return (msg << 20 | variant << 8 | (at & 0xff)) + 1;
}

/// One card over a run of images the model asked for in one breath. `slots` is
/// the whole run, `imgs` the part of it this client has an image for; the rest
/// are still in its queue and draw as reserved slots. An id that no longer
/// resolves (its engine is gone) draws as a failed slot.
fn renderToolCard(imgs: []const wire.ImageId, slots: usize, key: usize, id: usize) void {
    if (slots == 0) return;
    g_card_imgs = imgs;

    var tiles_buf: [12]bubbles.Tile = undefined;
    var step_bufs: [12][32]u8 = undefined;
    const n = @min(slots, tiles_buf.len);
    var n_done: usize = 0;
    var busy = false;
    var first: ?*mirror.Image = null;
    for (tiles_buf[0..n], 0..) |*tile, i| {
        if (i >= imgs.len) {
            // Asked for, and this client has not put it anywhere yet: the rail
            // has the row that says why.
            tile.* = .pending;
            busy = true;
            continue;
        }
        const im = imageById(imgs[i]) orelse {
            tile.* = .failed;
            continue;
        };
        if (first == null) first = im;
        switch (im.status()) {
            .done => {
                n_done += 1;
                tile.* = if (im.thumb()) |t|
                    .{ .rgba = .{ .px = t.px, .w = t.w, .h = t.h } }
                else
                    .pending;
                if (im.receiving()) busy = true;
            },
            .failed, .canceled => tile.* = .failed,
            // A host has it but has not started it. See queue_rail.
            .pending => {
                tile.* = .pending;
                busy = true;
            },
            // Under way: this tile is where it is watched, so the preview and
            // the step count come here and the rail lists nothing.
            .generating, .suspended => {
                tile.* = .{ .rendering = liveTile(im, &step_bufs[i]) };
                busy = true;
            },
        }
    }
    if (busy) dvui.refresh(null, @src(), null);

    var meta_buf: [80]u8 = undefined;
    var cbuf: [8]u8 = undefined;
    const meta = if (first) |f| std.fmt.bufPrint(&meta_buf, "{s}{d}×{d} · seed {d}", .{
        if (n > 1) (std.fmt.bufPrint(&cbuf, "×{d} · ", .{n}) catch "") else "",
        f.info.req_width,
        f.info.req_height,
        f.info.req_seed,
    }) catch "" else "";

    var status_buf: [40]u8 = undefined;
    const status = if (busy)
        (std.fmt.bufPrint(&status_buf, "rendering {d} of {d}", .{ @min(n_done + 1, n), n }) catch "rendering")
    else
        "";

    bubbles.toolCard(@src(), .{
        .meta = meta,
        .tiles = tiles_buf[0..n],
        .aspect = if (first) |f| (if (f.info.req_height > 0)
            @as(f32, @floatFromInt(f.info.req_width)) / @as(f32, @floatFromInt(f.info.req_height))
        else
            1) else 1,
        .selected = if (g_card_sel_key == key) g_card_sel_tile else null,
        .prompt = if (first) |f| f.info.prompt else "",
        .expanded = g_card_open == key,
        .busy = busy,
        .status = status,
        .id_extra = id,
    }, .{
        .ctx = @ptrFromInt(key),
        .on_toggle = cardToggle,
        .on_select = cardSelect,
        .on_open_studio = cardOpenStudio,
        .on_cancel = cardCancel,
    });
}

/// The × on a tile that is rendering.
fn cardCancel(_: *anyopaque, i: usize) void {
    if (i < g_card_imgs.len) onCancelImage(g_card_imgs[i]);
}

/// A tile for a render under way: its last preview frame, where it is, and the
/// host doing it when there is more than one.
fn liveTile(im: *const mirror.Image, buf: []u8) bubbles.Rendering {
    const host = g_hosts.hostOf(im.info.id);
    const label = switch (im.status()) {
        .suspended => std.fmt.bufPrint(buf, "paused · {d}/{d}", .{ im.info.step, im.info.total }) catch "paused",
        else => if (host.len > 0)
            std.fmt.bufPrint(buf, "{d} / {d} · {s}", .{ im.info.step, im.info.total, host }) catch ""
        else
            std.fmt.bufPrint(buf, "step {d} / {d}", .{ im.info.step, im.info.total }) catch "",
    };
    return .{
        .px = if (im.preview) |pv|
            (if (im.preview_w > 0 and im.preview_h > 0)
                bubbles.Px{ .px = pv, .w = im.preview_w, .h = im.preview_h }
            else
                null)
        else
            null,
        .step = im.info.step,
        .steps = im.info.total,
        .label = label,
    };
}

fn cardToggle(ctx: *anyopaque) void {
    const key = @intFromPtr(ctx);
    g_card_open = if (g_card_open == key) null else key;
}

/// Selecting a tile also opens the viewer: "look closer" is what a click on a
/// thumbnail means, and the selection is what makes Open in Studio unambiguous.
fn cardSelect(ctx: *anyopaque, i: usize) void {
    g_card_sel_key = @intFromPtr(ctx);
    g_card_sel_tile = i;
    if (i < g_card_imgs.len) {
        const im = imageById(g_card_imgs[i]) orelse return;
        if (im.status() == .done) g_viewer_request = im.info.id;
    }
}

/// Carry this render's parameters into the studio form and switch to it.
fn cardOpenStudio(ctx: *anyopaque) void {
    const key = @intFromPtr(ctx);
    if (g_card_imgs.len == 0) return;
    const idx = if (g_card_sel_key == key and g_card_sel_tile < g_card_imgs.len) g_card_sel_tile else 0;
    image_view.loadFrom(imageById(g_card_imgs[idx]) orelse return);
    enterImageMode();
}

// ------------------------------------------------------------------- sidebar

/// Draw the conversation list, grouped by local calendar day.
fn renderSidebar() void {
    var rows: [64]shell.ConvRow = undefined;
    var subs: [64][32]u8 = undefined;
    var groups: [3]shell.ConvGroup = undefined;
    var n_groups: usize = 0;

    const now = nowMs();
    const tz = history.localOffsetSeconds(@divTrunc(now, 1000));
    var n: usize = 0;
    // The store is already newest-first, so each group is a contiguous run.
    inline for ([_]history.Group{ .today, .yesterday, .earlier }) |grp| {
        const start = n;
        for (g_history.entries.items) |e| {
            if (n >= rows.len) break;
            if (history.groupOf(e.updated_ms, now, tz) != grp) continue;
            rows[n] = .{
                .id = e.id,
                .title = e.title,
                .sub = std.fmt.bufPrint(&subs[n], "{d} messages", .{e.turns}) catch null,
            };
            n += 1;
        }
        if (n > start) {
            groups[n_groups] = .{ .head = grp.head(), .rows = rows[start..n] };
            n_groups += 1;
        }
    }

    shell.sidebar(.{
        .groups = groups[0..n_groups],
        .selected = if (g_history.current != 0) g_history.current else null,
        .models_pct = null,
    }, .{
        .on_new_chat = onNewConversation,
        .on_select = onSelectConversation,
        .on_delete = onDeleteConversation,
        .on_models = openSettings,
        .on_settings = openSettings,
    });
}

/// The studio's left rail: the prompt library, grouped by day exactly as the
/// conversation list is. A row restores TEXT and nothing else -- the knobs stay
/// where they were, which is what makes the library a place to keep phrasings
/// rather than a second copy of the recipe the PNG already carries.
fn renderPromptSidebar() void {
    var rows: [64]shell.ConvRow = undefined;
    var groups: [3]shell.ConvGroup = undefined;
    var n_groups: usize = 0;

    const now = nowMs();
    const tz = prompt_history.localOffsetSeconds(@divTrunc(now, 1000));
    var n: usize = 0;
    // The store is already newest-first, so each group is a contiguous run.
    inline for ([_]prompt_history.Group{ .today, .yesterday, .earlier }) |grp| {
        const start = n;
        for (g_prompts.entries.items) |e| {
            if (n >= rows.len) break;
            if (prompt_history.groupOf(e.updated_ms, now, tz) != grp) continue;
            // No sub-line: the prompt IS the row, and a second line of it would
            // be the same text twice.
            rows[n] = .{ .id = e.id, .title = e.prompt };
            n += 1;
        }
        if (n > start) {
            groups[n_groups] = .{ .head = grp.head(), .rows = rows[start..n] };
            n_groups += 1;
        }
    }

    shell.sidebar(.{
        .groups = groups[0..n_groups],
        .new_label = "New prompt",
        .empty = "Prompts you generate from are kept here.",
        // No footer: the gear in the title bar is the studio's route to
        // Settings, and a second one a hand's width away is two answers to the
        // same question.
        .footer = false,
    }, .{
        .on_new_chat = image_view.clearPrompt,
        .on_select = onSelectPrompt,
        .on_delete = onDeletePrompt,
        .on_models = openSettings,
        .on_settings = openSettings,
    });
}

fn onSelectPrompt(id: u64) void {
    const e = g_prompts.find(id) orelse return;
    image_view.setPrompt(e.prompt, e.negative);
}

fn onDeletePrompt(id: u64) void {
    g_prompts.remove(g_gpa, g_io, id);
}

/// Take whatever the studio just generated into the prompt library. Polled
/// rather than pushed so `image_view` needs nothing from the store.
fn recordStudioPrompt() void {
    const rec = image_view.recorded_prompt orelse return;
    image_view.recorded_prompt = null;
    _ = g_prompts.record(g_gpa, g_io, nowMs(), rec.prompt, rec.negative);
}

/// Point the store at `<config dir>/conversations` and scan it.
fn openHistory() void {
    const dir: ?[]u8 = blk: {
        if (g_config_path) |p| {
            // An explicit --config: keep the history beside that file.
            const parent = std.fs.path.dirname(p) orelse ".";
            break :blk std.fs.path.join(g_gpa, &.{ parent, "conversations" }) catch null;
        }
        const base = (config.Config.dirPath(g_io, g_gpa, g_environ) catch null) orelse break :blk null;
        defer g_gpa.free(base);
        break :blk std.fs.path.join(g_gpa, &.{ base, "conversations" }) catch null;
    };
    const d = dir orelse {
        std.log.warn("history: no config directory — conversations will not be saved", .{});
        return;
    };
    defer g_gpa.free(d);
    g_history.open(g_gpa, g_io, d);
    // The prompt library lives beside the transcripts, in the same directory an
    // explicit --config redirects, so a throwaway config leaves nothing behind.
    const parent = std.fs.path.dirname(d) orelse ".";
    g_prompts.open(g_gpa, g_io, parent);
}

fn onNewConversation() void {
    saveHistory(true); // don't lose the one we're leaving
    g_history.newConversation();
    g_saved_turns = 0;
    g_saved_tail = 0;
    g_saved_images = 0;
    newChat();
}

fn onSelectConversation(id: u64) void {
    if (id == g_history.current) return;
    g_load_conv = id;
}

fn onDeleteConversation(id: u64) void {
    const was_current = id == g_history.current;
    g_history.remove(g_gpa, g_io, id);
    if (was_current) {
        g_saved_turns = 0;
        g_saved_tail = 0;
        g_saved_images = 0;
        newChat();
    }
}

/// Swap the live transcript for a stored one. Runs at the top of a frame, never
/// mid-render: it resets the session and replays every turn into the model's
/// tokenizer, which is not something to do while widgets hold pointers into the
/// message list.
fn applyPendingConversationLoad() void {
    const id = g_load_conv orelse return;
    if (g_m.state.loading) { // the loader owns the session right now
        g_load_conv = null;
        return;
    }
    // A turn in flight owns both the transcript and the KV it is decoding
    // against: the host would refuse the adopt. Hold the click (keep
    // `g_load_conv`) and apply it when the turn ends, rather than dropping it.
    if (g_m.state.llm_busy) return;
    g_load_conv = null;

    var loaded = g_history.load(g_gpa, g_io, id) orelse return;
    defer loaded.deinit();

    saveHistory(true); // flush the conversation we are leaving

    // The wire form of the stored turns, built in an arena for the one request.
    var arena = std.heap.ArenaAllocator.init(g_gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var msgs: std.ArrayList(wire.Message) = .empty;
    var n_saved: usize = 0;
    for (loaded.turns) |t| {
        const v = a.alloc(wire.Variant, 1) catch break;
        v[0] = .{
            .text = t.text,
            // Without this a primed reply parses as having no thought at all and
            // the whole reasoning block spills into the answer (see history.Turn).
            .thought_primed = t.primed,
            // The markers this turn was generated with, so it splits the same way
            // it did when it was live, with a different model loaded, or none.
            .reason_open = t.reason_open,
            .reason_close = t.reason_close,
            .gen_model = t.model,
            .images = restoreImages(a, t.images),
        };
        n_saved += v[0].images.len;
        msgs.append(a, .{
            .role = if (t.role == .user) .user else .assistant,
            .synthetic = t.synthetic,
            .variants = v,
        }) catch break;
    }
    // The host replaces its transcript (replaying each turn into the model's
    // tokenizer when one is resident) and sends the mirror the result.
    post(.{ .chat_adopt = .{ .messages = msgs.items } });
    // Everything below names the conversation just opened; the mirror still
    // holds the one being left until the host answers the adopt.
    g_handover_rev = g_m.transcript_rev;
    g_input.clearRetainingCapacity();
    g_follow_bottom = true;

    g_history.current = id;
    // The conversation you are looking at belongs at the top of the list. Done
    // in memory only: persisting it would be a write on open, which is the
    // thing that broke the ordering in the first place. Across a restart the
    // list falls back to last-modified order, which is the honest thing for a
    // conversation nobody has touched.
    g_history.touch(id, nowMs());
    g_saved_turns = loaded.turns.len;
    g_saved_tail = if (loaded.turns.len > 0) loaded.turns[loaded.turns.len - 1].text.len else 0;
    // Same count the saver compares against, so opening never rewrites.
    g_saved_images = n_saved;
}

/// Rebuild a turn's finished renders from disk so its card shows the images
/// instead of the model's request being run again.
///
/// A file that has since moved or been deleted becomes a FAILED image carrying
/// `error.SavedImageMissing`, not a fresh generation: the user asked for that
/// picture once, and silently spending VRAM to remake it is the behaviour this
/// whole path exists to remove.
///
/// These are the client's own images (the host never sees them): they live in
/// the mirror beside the host's, so they show up in Library and the viewer,
/// and the transcript carries their ids like any other. Returns the ids, in
/// `a`.
fn restoreImages(a: std.mem.Allocator, recs: []const history.ImageRec) []const wire.ImageId {
    var ids: std.ArrayList(wire.ImageId) = .empty;
    for (recs) |rec| {
        // Already here? Reuse it. The same render can appear in two
        // conversations without being loaded twice, and reopening one
        // conversation repeatedly cannot keep growing the Library.
        if (g_hosts.bySavedPath(rec.path)) |have| {
            ids.append(a, have.info.id) catch return ids.items;
            continue;
        }
        var info: wire.ImageInfo = .{
            // Left EMPTY on purpose: the request lives in the PNG's own
            // metadata, which `image_view.loadFrom` reads when reopening it.
            .req_width = @intCast(rec.width),
            .req_height = @intCast(rec.height),
            .req_steps = @intCast(rec.steps),
            .req_seed = rec.seed,
            .width = @intCast(rec.width),
            .height = @intCast(rec.height),
        };
        var pixels: ?[]u8 = null;
        if (vips.loadRgb(g_gpa, rec.path)) |dec| {
            defer g_gpa.free(dec.pixels);
            if (tp.image.rgbToRgba(g_gpa, dec.pixels, dec.width, dec.height)) |rgba| {
                pixels = rgba;
                info.width = @intCast(dec.width);
                info.height = @intCast(dec.height);
                info.status = .done;
                info.pixels_rev = 1;
            } else |_| {
                info.status = .failed;
                info.failure = "OutOfMemory";
            }
        } else |_| {
            // Moved or deleted since: a FAILED image, not a fresh generation.
            info.status = .failed;
            info.failure = "SavedImageMissing";
        }
        const id = g_hosts.addLocal(info, pixels, g_gpa.dupe(u8, rec.path) catch null);
        if (id != 0) ids.append(a, id) catch return ids.items;
    }
    return ids.items;
}

/// Persist the live transcript if it changed since the last save. Skipped while
/// a turn is streaming (`force` overrides, for a conversation switch or exit):
/// writing on every token would rewrite the file once per frame.
fn saveHistory(force: bool) void {
    const msgs: []mirror.Message = g_m.messages.items;
    if (msgs.len == 0) return;
    // `force` bypasses the BUSY gate only, so a switch or an exit can flush a
    // turn that is still streaming. It does NOT bypass the change check:
    // rewriting an unchanged conversation stamps it with a fresh `updated`, and
    // since the list is ordered by that, merely opening conversations
    // reshuffled them — the one you left jumped to the top. Reading must not
    // write.
    if (!force and g_m.state.llm_busy) return;
    // The host has not swapped the transcript yet, so what is on screen belongs
    // to the conversation we just left, not the one `g_history.current` names.
    if (g_handover_rev) |rev| {
        if (g_m.transcript_rev == rev) return;
        g_handover_rev = null;
    }
    const tail = msgs[msgs.len - 1].active().text.items.len;
    const n_images = savedImageCount(msgs);
    if (msgs.len == g_saved_turns and tail == g_saved_tail and n_images == g_saved_images) return;

    var turns: std.ArrayList(history.Turn) = .empty;
    defer turns.deinit(g_gpa);
    var image_recs: std.ArrayList([]history.ImageRec) = .empty;
    defer {
        for (image_recs.items) |r| g_gpa.free(r);
        image_recs.deinit(g_gpa);
    }
    for (msgs) |*m| {
        // `chat.Role` is user|assistant only; the system prompt is rebuilt
        // from settings on load rather than pinned into every transcript.
        const role: history.Role = switch (m.role) {
            .user => .user,
            .assistant => .assistant,
        };
        // Renders this turn produced that made it to disk. Only saved ones:
        // there is nothing to point at otherwise, and a reload shows the call
        // without the picture rather than generating it again.
        var recs: std.ArrayList(history.ImageRec) = .empty;
        const v = m.active();
        for (v.images.items) |id| {
            const im = imageById(id) orelse continue;
            if (im.status() != .done) continue;
            const path = im.saved_path orelse continue;
            recs.append(g_gpa, .{
                .path = path,
                .width = im.info.width,
                .height = im.info.height,
                .steps = im.info.req_steps,
                .seed = im.info.req_seed,
            }) catch break;
        }
        // Owned by `image_recs` so the slices outlive the save call below.
        const owned: []history.ImageRec = recs.toOwnedSlice(g_gpa) catch &.{};
        image_recs.append(g_gpa, owned) catch {};
        turns.append(g_gpa, .{
            .role = role,
            .text = v.text.items,
            .primed = v.thought_primed,
            .images = owned,
            // What produced this turn. Without the markers a reopened
            // conversation splits its replies with whatever model is loaded now.
            .reason_open = v.reason_open,
            .reason_close = v.reason_close,
            .model = v.gen_model,
            .synthetic = m.synthetic,
        }) catch return;
    }
    if (turns.items.len == 0) return;

    g_history.save(g_gpa, g_io, nowMs(), turns.items);
    g_saved_turns = msgs.len;
    g_saved_tail = msgs[msgs.len - 1].active().text.items.len;
    g_saved_images = n_images;
}

/// Renders that reached disk, the ones `saveHistory` records.
fn savedImageCount(msgs: []const mirror.Message) usize {
    var n: usize = 0;
    for (msgs) |*m| for (m.active().images.items) |id| {
        const im = imageById(id) orelse continue;
        if (im.status() == .done and im.saved_path != null) n += 1;
    };
    return n;
}

// -------------------------------------------------------------- title bar

fn onTabPicked(t: shell.Tab) void {
    switch (t) {
        .chat => if (g_view != .chat) enterChatMode(),
        .studio => if (g_view != .image) enterImageMode(),
    }
}

/// A pick in the chat-model chip applies at once: save, and reload if the
/// resident model changed, the way the thinking toggle applies.
fn onLlmPick(p: model_menu.Pick) void {
    std.log.info("[models] chat pick: {t}", .{p});
    switch (p) {
        .none => selection.clearLlm(&g_config),
        .path => |path| selection.selectLlm(&g_config, &g_models.cat, path),
        .settings => return openSettings(),
    }
    commitConfig();
}

fn onImagePick(p: model_menu.Pick) void {
    std.log.info("[models] image pick: {t}", .{p});
    switch (p) {
        .none => selection.clearCheckpoint(&g_config),
        .path => |path| selection.selectCheckpoint(&g_config, &g_models.cat, path),
        .settings => return openSettings(),
    }
    commitConfig();
}

// ------------------------------------------------------------- queue rail

// ------------------------------------------------------------- composer

/// A stable address for the composer's callback context. The composer acts on
/// process globals, so nothing needs to travel through the pointer, but the
/// shared widget takes one so a caller that DOES have per-instance state can
/// use it.
var g_composer_ctx: u8 = 0;
/// The megapixel field's text, held across frames because the user types into
/// it. Synced from `g_config.framing_mp` whenever it is not focused.
var g_mp_buf: [16]u8 = [_]u8{0} ** 16;

/// Recompute the image dimensions from the framing choice, push them to the
/// engine, and persist.
///
/// The two dependent views seed their form buffers ONCE, so both have to be
/// told: without this, Studio keeps generating at the size it was opened with,
/// and Settings' Apply writes its stale copy straight back over the change.
fn applyFraming() void {
    g_config.applyFraming();
    g_config_baseline.framing_ratio = g_config.framing_ratio;
    g_config_baseline.framing_mp = g_config.framing_mp;
    g_config_baseline.width = g_config.width;
    g_config_baseline.height = g_config.height;
    image_view.reseed();
    config_view.reseed();
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    postSettings();
}

/// The weight-noise curve's shape preview: the curve sampled first-layer-to-head
/// and normalized to its own peak, plus whether it parsed. Recomputed only when
/// the expression text changes, not every frame — `noise_curve` re-parses on each
/// evaluation and says not to call it from a hot path.
var g_noise_shape: [20]f32 = @splat(0);
var g_noise_shape_src: [config.max_noise_curve]u8 = @splat(0);
var g_noise_valid: bool = true;
/// Whether the active shape reads `a`, so the composer can dim an amount field
/// that would do nothing. Recomputed with the shape.
var g_noise_amount_live: bool = true;

fn noiseShape() []const f32 {
    const cur = g_config.weight_noise_curve.slice();
    if (std.mem.eql(u8, std.mem.sliceTo(&g_noise_shape_src, 0), cur)) return &g_noise_shape;
    @memset(&g_noise_shape_src, 0);
    @memcpy(g_noise_shape_src[0..cur.len], cur);

    g_noise_valid = cur.len != 0 and blk: {
        tp.noise_curve.validate(cur) catch break :blk false;
        break :blk true;
    };
    g_noise_amount_live = g_noise_valid and tp.noise_curve.respondsToAmount(cur);
    @memset(&g_noise_shape, 0);
    if (!g_noise_valid) return &g_noise_shape;

    const n: f32 = @floatFromInt(g_noise_shape.len - 1);
    var peak: f32 = 0;
    for (&g_noise_shape, 0..) |*o, i| {
        const t = @as(f32, @floatFromInt(i)) / n;
        // a = 1: the preview is normalized to its own peak anyway, so it draws the
        // SHAPE, which is what the amount field is not telling you.
        o.* = tp.noise_curve.sanitize(tp.noise_curve.eval(cur, .{ .t = t, .a = 1 }) catch 0);
        peak = @max(peak, o.*);
    }
    // Normalized, so a 0.005 curve and a 0.05 one of the same shape draw the same
    // picture. The amplitude is already legible as the leading coefficient.
    if (peak > 0) for (&g_noise_shape) |*o| {
        o.* /= peak;
    };
    return &g_noise_shape;
}

/// Push the configured weight noise to the live backend, and persist.
///
/// Applied LIVE rather than at a turn boundary: the curve lands in a table the
/// decode kernels re-read at every launch, so turning the knob mid-reply takes
/// effect on the next token.
fn applyWeightNoise() void {
    postSettings();
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
}

fn composerNoiseToggle(_: *anyopaque) void {
    g_config.weight_noise = !g_config.weight_noise;
    g_config_baseline.weight_noise = g_config.weight_noise;
    applyWeightNoise();
}

/// Which library entry the composer dropdown has selected. Resynced from the
/// active curve every frame by `noiseNames`, so editing the expression in Settings
/// moves the composer's label without the two needing to talk.
var g_noise_sel: usize = 0;
var g_noise_names: [config.max_noise_curves + 1][]const u8 = undefined;

/// The dropdown's entries: the saved library, plus a trailing "custom" when the
/// active curve is not one of them (typed in Settings). Selecting that entry is a
/// no-op, it exists so the chip can name what is actually live.
fn noiseNames() []const []const u8 {
    const cur = g_config.weight_noise_curve.slice();
    const saved = g_config.noise_curves.slice();
    for (saved, 0..) |*c, i| g_noise_names[i] = c.name.slice();
    if (g_config.noiseCurveMatching(cur)) |i| {
        g_noise_sel = i;
        return g_noise_names[0..saved.len];
    }
    g_noise_names[saved.len] = "custom";
    g_noise_sel = saved.len;
    return g_noise_names[0 .. saved.len + 1];
}

/// The amount field's text, held across frames like `g_mp_buf`.
var g_noise_amt_buf: [16]u8 = [_]u8{0} ** 16;

/// The amount was committed. Clamped generously: past ~0.8 a front-loaded curve
/// only produces fluent nonsense, but watching that happen is the point of the
/// knob being exposed.
fn composerNoiseAmount(_: *anyopaque) void {
    const typed = std.mem.trim(u8, std.mem.sliceTo(&g_noise_amt_buf, 0), " \t");
    const v = std.fmt.parseFloat(f32, typed) catch return;
    const clamped = std.math.clamp(v, 0, 1.0);
    g_config.weight_noise_amount = clamped;
    g_config_baseline.weight_noise_amount = clamped;
    // Applied even when the value did not change. An early return on
    // `clamped == g_config.weight_noise_amount` reads as a harmless optimization
    // and is not: it assumes the backend already holds what the config says, and
    // the one situation where a user retypes the number they can already see is
    // exactly the situation where that has stopped being true. It made a session
    // whose backend had drifted impossible to correct from the composer at all.
    applyWeightNoise();
}

/// A saved curve was picked: it becomes the active one.
fn composerNoisePick(_: *anyopaque) void {
    const saved = g_config.noise_curves.slice();
    if (g_noise_sel >= saved.len) return; // the "custom" entry names, it does not set
    g_config.weight_noise_curve = saved[g_noise_sel].curve;
    g_config_baseline.weight_noise_curve = g_config.weight_noise_curve;
    // Picking a curve turns the noise ON: reaching for a shape is asking for it.
    g_config.weight_noise = true;
    g_config_baseline.weight_noise = true;
    applyWeightNoise();
}

fn composerQuick(_: *anyopaque, _: usize) void {
    openSettings();
}
fn composerAllSettings(_: *anyopaque) void {
    enterImageMode();
}
/// "+ reference": attach an image to the next message. The drop/paste paths
/// already exist (handleDropFile / tryPasteClipboardImage); this opens a picker
/// for the same thing.
fn composerReference(_: *anyopaque) void {
    const path = dvui.dialogNativeFileOpen(g_gpa, .{ .title = "Attach a reference image" }) catch null;
    if (path) |pp| {
        defer g_gpa.free(pp);
        handleDropFile(pp);
    }
}

fn onRailTab(t: queue_rail.Tab) void {
    g_rail_tab = t;
}

/// A failed job asked for another go. Deferred to `pumpHost` rather than run
/// here: `retry` appends an image to a mirror's list, and that list may
/// reallocate under every `*Image` this frame is still drawing from.
var g_retry_request: ?wire.ImageId = null;

fn onRetryImage(id: u64) void {
    g_retry_request = id;
}

/// A library thumbnail was clicked: "look at that picture". Always the viewer,
/// in both views. Sending a finished picture to the studio canvas would park it
/// over whatever is rendering, and the only way back out was to press Generate.
fn onOpenLibraryImage(id: u64) void {
    if (imageById(id) == null) return;
    g_viewer_request = id;
}

/// The × on a queue row, or on a tile that is rendering. A render still to come
/// or under way is cancelled; a row that only records a failure is cleared away,
/// since there is nothing left to stop.
fn onCancelImage(id: u64) void {
    if (g_hosts.forget(id)) return;
    post(.{ .img_cancel = .{ .image = id } });
}

/// Drag-to-reorder. The rail hands back the ids of the row dragged and the row
/// it was dropped before (null for the end).
fn onReorderQueue(move: u64, before: ?u64) void {
    post(.{ .img_move = .{ .image = move, .before = before } });
}

/// Chat view while the session (re)loads on the background thread.
/// A small "Loading..." bubble shown in the assistant response slot while the
/// model (re)loads, same left-leaning neutral style as a real assistant turn,
/// with a spinner. Replaced by the real streaming response the instant the
/// session is live.
fn loadingAssistantBubble() void {
    var bubble = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .margin = .{ .y = 9, .h = 9 },
        .padding = .{},
    });
    defer bubble.deinit();
    dvui.spinner(@src(), .{ .gravity_y = 0.5, .min_size_content = .{ .w = 14, .h = 14 }, .margin = .{ .w = 8 } });
    dvui.labelNoFmt(@src(), "Loading…", .{}, .{ .gravity_y = 0.5, .font = style.F.prose, .color_text = style.C.text_dim });
}

/// Assistant slot for a message sent while the LLM is paused with nothing
/// resident: no spinner (nothing is loading), just a paused hint. The load +
/// generation fire on resume, see toggleLlmPause -> maybeStartReload.
fn queuedAssistantBubble() void {
    var bubble = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .margin = .{ .y = 9, .h = 9 },
        .padding = .{},
    });
    defer bubble.deinit();
    // richLabel (not dvui.label) so the ⏸ routes to the emoji face; no text
    // face we bundle has the media-control glyphs. See fonts.isEmoji.
    fonts.richLabel(@src(), "⏸ queued — resume to generate", .{ .gravity_y = 0.5, .font = style.F.prose, .color_text = style.C.text_dim });
}

/// Chat view when no LLM is configured (or the last load failed): explain and
/// offer a shortcut into settings.
fn renderNoModel() void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5, .padding = dvui.Rect.all(24) });
    defer col.deinit();

    {
        var tl = dvui.textLayout(@src(), .{}, .{ .gravity_x = 0.5, .background = false });
        defer tl.deinit();
        if (g_m.state.load_err.len > 0) {
            const err = g_m.state.load_err;
            var msg: [320]u8 = undefined;
            const text = if (std.mem.eql(u8, err, "MmprojNotVisionTower"))
                "The vision-tower (mmproj) file isn't a vision projector — it looks like an LLM model, not an mmproj.\n\nIn Settings, set the vision tower to an mmproj-*.gguf file, or clear it for a text-only model."
            else
                (std.fmt.bufPrint(&msg, "Failed to load the model: {s}\n\nChoose a different model in Settings.", .{err}) catch "Failed to load the model.");
            fonts.addStyled(tl, text, .{}, .{ .font = style.F.prose, .color_text = style.C.text });
        } else {
            fonts.addStyled(tl, "No LLM model is set.\n\nOpen Settings to choose a model file and get started.", .{}, .{
                .font = style.F.prose,
                .color_text = style.C.text,
            });
        }
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 14 } });
        defer row.deinit();
        if (bubbles.primaryButton(@src(), "Open Settings", true)) openSettings();
        if (bubbles.secondaryButton(@src(), 1, "Image studio", true)) enterImageMode();
    }
}

fn openSettings() void {
    g_return_view = if (g_view == .config) g_return_view else g_view;
    config_view.open();
    g_view = .config;
}

/// What the studio canvas draws from: every host's renders, not just the one
/// the form is configured against.
fn studioImages() image_view.Images {
    return .{ .ctx = undefined, .live = liveImages, .newest = newestImage, .hostOf = hostOfImage };
}

/// Every render in motion, oldest first, across every host. A paused one counts:
/// it is parked, not over, and its preview is still the last thing it drew.
fn liveImages(_: *anyopaque, out: []*const mirror.Image) []const *const mirror.Image {
    var run_buf: [32]hosts.Hosts.Shot = undefined;
    var n: usize = 0;
    for (g_hosts.running(run_buf[0..@min(run_buf.len, out.len)])) |sh| {
        switch (sh.im.status()) {
            .generating, .suspended => {},
            else => continue,
        }
        out[n] = sh.im;
        n += 1;
    }
    return out[0..n];
}

/// The newest finished picture, for a canvas with nothing to watch. Held at
/// full size while it is on screen.
fn newestImage(_: *anyopaque) ?*const mirror.Image {
    var fin_buf: [1]hosts.Hosts.Shot = undefined;
    const fin = g_hosts.finished(&fin_buf);
    if (fin.len == 0) return null;
    return shown(0, fin[0].im);
}

fn hostOfImage(_: *anyopaque, id: wire.ImageId) []const u8 {
    return g_hosts.hostOf(id);
}

/// The viewer navigates every finished render, oldest to newest so the right
/// arrow moves forward in time, and can show any of them, an attachment
/// included. One still arriving is listed too: skipping it would make the
/// arrows jump over a picture that is about to appear.
fn diffuserSource() viewer.ImageSource {
    return .{
        .ctx = undefined,
        .gpa = g_gpa,
        .collect = diffuserCollect,
        .resolve = viewerResolve,
        .hostOf = hostOfImage,
    };
}
fn diffuserCollect(_: *anyopaque, buf: *std.ArrayList(wire.ImageId)) void {
    buf.clearRetainingCapacity();
    var shots: [256]hosts.Hosts.Shot = undefined;
    const fin = g_hosts.finished(&shots);
    var i = fin.len;
    while (i > 0) {
        i -= 1;
        buf.append(g_gpa, fin[i].im.info.id) catch {};
    }
}
fn viewerResolve(_: *anyopaque, id: wire.ImageId) ?*const mirror.Image {
    return shown(1, g_hosts.imageById(id) orelse return null);
}

/// Mode switches are PURE VIEW changes, they never free, unload, or reload a
/// model. Both the LLM and the diffusion engine stay resident across switches;
/// VRAM is shared live via the split (the meter handle), so toggling touches no
/// GPU state and can't leak or mis-budget. The transcript is always intact
/// (the session is never torn down here).
fn enterImageMode() void {
    g_view = .image;
}
fn enterChatMode() void {
    g_view = .chat;
}

/// Apply button: persist settings, reconcile the app-level diffusion engine,
/// and apply everything else WITHOUT wiping the chat. A change that alters the
/// LLM load or the image-tool availability (which changes the system prompt)
/// forces a transcript-preserving reload, but only if the LLM is currently
/// resident; if it hasn't lazy-loaded yet, the new config is simply picked up on
/// the first message.
fn applyConfig() void {
    commitConfig();
    g_view = g_return_view;
}

/// Persist `g_config` and bring the engines in line with it: rebuild or retune
/// the diffuser, reload the LLM when its model set changed, else push the live
/// settings. Settings Apply and a chip pick both end here.
fn commitConfig() void {
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    // Each host diffs the new settings against the ones in force and decides
    // what the change costs (nothing, a live update, a context rebuild, a reload).
    postSettings();
    // The host list and the chat pin may have moved too.
    g_hosts.sync(&g_config);
    g_m = g_hosts.chatMirror();
    g_config_baseline = g_config;
}

/// Toolbar reasoning toggle: flip whether the model reasons before answering,
/// persist it, and push it into the running session live (no reload, it only
/// shapes the next prompt built). Keeps the baseline in sync so a later
/// Settings -> Cancel doesn't resurrect the old value.
fn toggleReasoning() void {
    g_config.reasoning = !g_config.reasoning;
    g_config_baseline.reasoning = g_config.reasoning;
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    postSettings();
}

fn cycleReasoningEffort() void {
    g_config.reasoning_effort = switch (g_config.reasoning_effort) {
        .high => .medium,
        .medium => .low,
        .low => .high,
    };
    g_config_baseline.reasoning_effort = g_config.reasoning_effort;
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    postSettings();
}

/// Cancel button: discard unsaved edits by reloading the on-disk settings, and
/// return to wherever Settings was opened from (chat or the studio).
fn cancelConfig() void {
    g_config = config.Config.load(g_io, g_gpa, g_environ, g_config_path);
    g_config_baseline = g_config;
    g_view = g_return_view;
}

fn renderMessages(list_h: f32) void {
    {
        var scroll = dvui.scrollArea(@src(), .{ .scroll_info = &g_scroll_info }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = list_h },
            .max_size_content = .height(list_h),
        });
        defer scroll.deinit();

        var list = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .horizontal,
            .padding = .{ .x = 30, .y = 24, .w = 30, .h = 24 },
        });
        defer list.deinit();

        // The mirror holds the transcript whether or not a model is resident
        // (the host carries it across an unload), so the conversation stays on
        // screen and is never visually "reset"; only a "new chat" click clears it.
        const st = &g_m.state;
        const loading = st.loading;
        const msgs: []mirror.Message = g_m.messages.items;
        if (msgs.len == 0 and !loading and g_pending_text == null) {
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = dvui.Rect.all(16) });
            defer tl.deinit();
            fonts.addStyled(tl, if (!st.llm_resident)
                "Say something to start — the model loads on your first message."
            else
                "Say something to start the conversation.", .{}, .{
                .font = style.F.prose,
                .color_text = style.C.text_ghost,
            });
        } else {
            for (msgs, 0..) |*m, idx| renderMessage(msgs, m, idx);
        }
        // While the model (re)loads in the background: the just-sent message
        // (not yet in the transcript, it submits once the session is live) shows
        // as a normal user bubble, and the assistant slot shows a small "Loading..."
        // The instant the session is ready these are replaced by the real turn.
        if (loading) {
            if (g_pending_text) |txt| pendingUserBubble(txt);
            loadingAssistantBubble();
        } else if (st.llm_paused and !st.llm_resident) {
            // Paused with nothing resident: the just-sent message is HELD (no load
            // at all) until resume. Show it + a paused hint rather than "Loading..."
            // or the empty-state placeholder. (A resident pause is shown inline
            // per message above.)
            if (g_pending_text) |txt| {
                pendingUserBubble(txt);
                queuedAssistantBubble();
            }
        }
    }

    // The scrollArea has now applied this frame's user scroll (wheel/scrollbar)
    // and knows its virtual size. Decide whether to keep following the bottom:
    // stop if the user moved the view up; resume once they return to the end.
    // A user scroll-up shrinks the offset below last frame's; content growth
    // only raises scrollMax (offset stays), so the two are distinguishable.
    const max = g_scroll_info.scrollMax(.vertical);
    const off = g_scroll_info.offset(.vertical);
    if (g_follow_bottom) {
        if (off + 1.0 < g_prev_offset) g_follow_bottom = false;
    } else if (max - off < 12.0) {
        g_follow_bottom = true;
    }
    if (g_follow_bottom) {
        g_scroll_info.viewport.y = max;
        g_prev_offset = max;
    } else {
        g_prev_offset = off;
    }
}

/// A provisional user bubble for a message that's been sent but not yet in the
/// transcript (the model is still (re)loading). Same right-leaning accent style
/// as a real user turn.
fn pendingUserBubble(text: []const u8) void {
    var bubble = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .gravity_x = 1.0,
        .max_size_content = .width(style.Layout.bubble_max),
        .background = true,
        .color_fill = style.C.bubble_user,
        .border = style.Edge.all,
        .color_border = style.hairline,
        .corner_radius = .{ .x = 10, .y = 10, .w = 3, .h = 10 },
        .padding = .{ .x = 14, .y = 11, .w = 14, .h = 11 },
        .margin = .{ .y = 9, .h = 9 },
    });
    defer bubble.deinit();
    markdown_view.render(@src(), text, .{ .prose = .{
        .background = false,
        .padding = .{},
        .font = style.F.prose,
        .color_text = style.C.text_hi,
    } });
}

/// Spinner + "Processing...": the model is working on this turn but has produced
/// no visible text yet. See `renderMessage` for why this is distinct from the
/// session-load "Loading..." bubble.
fn processingRow() void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{});
    defer row.deinit();
    dvui.spinner(@src(), .{ .gravity_y = 0.5, .min_size_content = .{ .w = 12, .h = 12 }, .margin = .{ .w = 6 } });
    dvui.labelNoFmt(@src(), "Processing…", .{}, .{ .gravity_y = 0.5, .font = style.F.prose, .color_text = style.C.text_dim });
}

/// The assistant bubble's body when there is nothing to show and nothing is
/// running: a generation error, or a turn queued behind a paused LLM.
/// (The live case is the "Processing..." spinner in `renderMessage`.)
fn renderEmptyAssistant(idx: usize) void {
    var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
    defer tl.deinit();
    const st = &g_m.state;
    if (st.llm_resident and st.gen_err.len > 0) {
        // A lost CUDA context fails every turn from here on with the same opaque
        // CudaError. Say what actually happened, once, instead of letting the user
        // retry into it.
        if (st.ctx_lost) {
            fonts.addRich(tl, "the GPU context was lost (a kernel faulted). Restart TensorPencil to use the GPU again; the terminal log names the kernels it was running.");
        } else {
            var msg: [128]u8 = undefined;
            fonts.addRich(tl, std.fmt.bufPrint(&msg, "generation error: {s}", .{st.gen_err}) catch "generation error");
        }
    } else if (st.llm_resident and idx + 1 == g_m.messages.items.len and st.llm_paused) {
        // A turn queued while the LLM is paused, it runs on resume (Tier 2).
        // addStyled (not addText) routes the ⏸ to the emoji face (NotoSansCJK
        // lacks media-control glyphs, see fonts.isEmoji).
        fonts.addStyled(tl, "⏸ queued — resume to generate", .{}, .{ .font = style.F.prose, .color_text = style.C.text_dim });
    }
}

fn renderMessage(msgs: []mirror.Message, m: *mirror.Message, idx: usize) void {
    // An app-written note (an image outcome), not something the user typed:
    // centred, quiet, and no bubble. Giving it a user bubble would put words in
    // the user's mouth in their own transcript.
    if (m.synthetic) {
        dvui.labelNoFmt(@src(), m.active().text.items, .{}, .{
            .id_extra = idx,
            .font = style.F.mono_sm,
            .color_text = style.C.text_ghost,
            .gravity_x = 0.5,
            .padding = .{},
            .margin = .{ .y = 8, .h = 8 },
        });
        return;
    }
    const is_user = m.role == .user;

    // The asymmetry is load-bearing: the user's turn is a CARD ON the page, the
    // assistant's turn IS the page. So the user gets a bubble, a right edge and
    // a tail corner; the assistant gets no bubble, no avatar and no background
    // at all, only a measure limit. Sender is conveyed by shape and side, which
    // is why neither carries a "You"/"Assistant" label.
    var wrap = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .id_extra = idx,
        .expand = .horizontal,
        .margin = .{ .y = 9, .h = 9 },
    });
    defer wrap.deinit();

    var bubble = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = if (is_user) .none else .horizontal,
        .gravity_x = if (is_user) 1.0 else 0.0,
        .max_size_content = .width(if (is_user) style.Layout.bubble_max else style.Layout.prose_max),
        .background = is_user,
        .color_fill = style.C.bubble_user,
        .border = if (is_user) style.Edge.all else .{},
        .color_border = style.hairline,
        // tl, tr, br, bl — the small one is the tail.
        .corner_radius = if (is_user) .{ .x = 10, .y = 10, .w = 3, .h = 10 } else style.R.none,
        .padding = if (is_user) .{ .x = 14, .y = 11, .w = 14, .h = 11 } else .{},
    });
    defer bubble.deinit();

    // Images the user attached to this message.
    for (m.attachments.items, 0..) |id, ai| if (g_hosts.imageById(id)) |im| renderGenImage(im, ai);

    // Everything below shows the message's ACTIVE variant (the ‹/› nav on the
    // last assistant response switches it; older takes stay stored).
    const v = m.active();
    const p = parseThink(v);
    // Only the last assistant message is actively generating; "Thinking..." means
    // the block is still open AND generation is live. A think block left open
    // because generation stopped (e.g. hit max tokens) reads "Thoughts". With no
    // resident model, nothing is generating.
    const live = g_m.state.llm_busy and idx + 1 == msgs.len;

    // The model is working but has not emitted a single token yet: prompt
    // processing (prefill), plus the short gap before the first token.
    //
    // This is tested BEFORE the thought block, and that ordering is the whole
    // point. When reasoning is on the chat template PRIMES `<think>` into the
    // prompt, so `parseThink` reports a non-null (empty) thought and
    // `p.thinking = true` from the very FIRST frame, before generation has
    // produced anything. Checking the thought first therefore jumped straight
    // from "Loading..." to "Thinking..." and displayed prompt processing as
    // reasoning. Raw variant text being empty is the honest signal.
    const nothing_yet = live and v.text.items.len == 0 and m.role == .assistant;

    if (nothing_yet) {
        // Deliberately a DIFFERENT word from the session-load "Loading..."
        // bubble: they are different phases with different costs, and a user
        // reads one unchanging spinner as one thing. "Loading..." is reading the
        // checkpoint and uploading weights to VRAM (chat.zig warms those up front
        // so they land in that phase, not this one); "Processing..." is per-turn
        // work that scales with the prompt.
        processingRow();
    }

    // Reasoning: collapse the thought block behind an expander, default
    // collapsed. The label doubles as a "thinking" indicator while the block is
    // still open. An empty thought (e.g. a model that opened and closed the
    // channel with nothing inside) shows no bubble at all, unless it's still
    // actively streaming, where "Thinking..." is the right cue.
    if (p.think) |think| {
        if (!nothing_yet and (think.len > 0 or (p.thinking and live))) {
            if (dvui.expander(@src(), if (p.thinking and live) "Thinking…" else "Thoughts", .{ .default_expanded = false }, .{})) {
                // Set the reasoning apart from the answer: a dimmer text color on
                // a slightly inset, accent-bordered block (a blockquote look), so
                // it reads as the model's scratch work rather than the reply.
                markdown_view.render(@src(), think, .{ .prose = .{
                    .expand = .horizontal,
                    .background = true,
                    .color_fill = style.C.sunken,
                    .color_text = style.C.text_dim,
                    .color_border = style.tint(style.C.blue, 120),
                    .border = .{ .x = 2 },
                    .corner_radius = .{ .x = 0, .y = 6, .w = 6, .h = 0 },
                    .margin = .{ .y = 4, .h = 6 },
                    .padding = .{ .x = 11, .y = 8, .w = 11, .h = 8 },
                    .font = style.F.prose,
                } });
            }
        }
    }

    if (p.answer.len > 0) {
        renderReply(v, p.answer, if (is_user) style.C.text_hi else style.C.text, idx, @min(m.cur, m.variants.items.len - 1));
        // Selection copies rendered text; this copies the raw markdown of
        // the whole reply (assistant messages only, a user's own text is
        // already in their hands).
        if (!is_user) {
            var wd: dvui.WidgetData = undefined;
            if (dvui.buttonIcon(@src(), "copy markdown", dvui.entypo.clipboard, .{}, .{}, .{
                .gravity_x = 1.0,
                .min_size_content = .{ .h = 12 },
                .color_text = style.C.text_ghost,
                .padding = dvui.Rect.all(2),
                .margin = .{ .y = 2 },
                .data_out = &wd,
            })) dvui.clipboardTextSet(p.answer);
            hint.hover(@src(), &wd, "Copy the reply as markdown");
        }
    } else if (!nothing_yet and m.role == .assistant and p.think == null and v.images.items.len == 0) {
        // Text exists but currently renders to nothing (e.g. a half-streamed
        // marker); keep the working indicator rather than flashing an empty card.
        if (live) processingRow() else renderEmptyAssistant(idx);
    }

    // The turn's measurement (see chat.TurnStats): what the prompt cost under a
    // user message, the rates + this reply's size + the context standing under
    // an assistant one. Updates every frame while the reply streams.
    renderStatsFooter(v.stats, is_user, v.gen_model);

    // ‹ n/m › navigation on the LAST assistant response (TODO #3): ‹ shows the
    // previous take, › the next, or, on the newest take, regenerates a fresh
    // one. Hidden while generating (Stop is the control then). Shown even with
    // NO live session (carried read-only transcript): ‹/› just switch the shown
    // take, and › regenerate lazy-loads the LLM first (see renderVariantNav).
    // Images belonging to a non-active take keep generating in the unified queue.
    if (!is_user) {
        // From `msgs`, the list this message belongs to, rather than re-derived
        // from the session or the carry: the caller already resolved which of the
        // two is live and holds the lock that makes the carried one safe to walk.
        const nmsg = msgs.len;
        const prev_user = nmsg >= 2 and msgs[nmsg - 2].role == .user;
        if (idx + 1 == nmsg and nmsg >= 2 and prev_user and !g_m.state.llm_busy)
            renderVariantNav(m, idx);
    }
}

/// A message's stats line: dim, small, and right-aligned so it reads as a
/// margin note on the bubble rather than as content. Draws nothing at all
/// until there is something measured, a bubble that has only just appeared
/// (or a transcript carried across a model swap) would otherwise show a row of
/// zeros. The strings come from `turn_stats`, which is where they're tested.
fn renderStatsFooter(st: wire.TurnStats, is_user: bool, gen_model: []const u8) void {
    // The model is named only when it is NOT the one loaded now: in a
    // single-model conversation it would be the same string under every turn,
    // and after a swap or a reload it is the thing you actually want to know.
    const live: []const u8 = g_m.state.llm_model;
    const show_model = gen_model.len > 0 and !std.mem.eql(u8, gen_model, live);

    if (!(if (is_user) st.hasPrompt() else st.hasGen())) {
        if (!show_model) return;
        dvui.labelNoFmt(@src(), gen_model, .{}, .{
            .gravity_x = if (is_user) 1.0 else 0.0,
            .font = style.F.mono_sm,
            .color_text = style.C.text_faint,
            .padding = .{},
            .margin = .{ .y = 5 },
        });
        return;
    }
    var buf: [turn_stats.buf_len]u8 = undefined;
    const stats = if (is_user) turn_stats.formatUser(&buf, st) else turn_stats.formatAssistant(&buf, st);
    if (stats.len == 0 and !show_model) return;
    var joined: [turn_stats.buf_len + 96]u8 = undefined;
    const text = if (show_model)
        (std.fmt.bufPrint(&joined, "{s}{s}{s}", .{
            stats,
            if (stats.len > 0) " · " else "",
            gen_model,
        }) catch stats)
    else
        stats;
    if (text.len == 0) return;
    dvui.labelNoFmt(@src(), text, .{}, .{
        // Sized to its text (no expand) so `gravity_x` places it along the
        // bubble's own outer edge: the user's leans right, the assistant's left.
        .gravity_x = if (is_user) 1.0 else 0.0,
        // Telemetry, not prose: mono and small, so a turn's numbers sit under
        // the message without competing with it.
        .font = style.F.mono_sm,
        .color_text = style.C.text_faint,
        .padding = .{},
        .margin = .{ .y = 5 },
    });
}

/// The ‹ n/m › variant-navigation row (see renderMessage). Back is disabled
/// (dimmed, inert) on the first take; next past the newest take regenerates.
/// With no model resident the host switches the shown take on the transcript
/// it carries, and › regenerate lazy-loads the LLM.
fn renderVariantNav(m: *mirror.Message, idx: usize) void {
    const n = m.variants.items.len;
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .y = 2 } });
    defer row.deinit();

    const icon_opts: dvui.Options = .{
        .min_size_content = .{ .w = 16, .h = 16 },
        .padding = dvui.Rect.all(2),
        .gravity_y = 0.5,
    };
    const can_back = m.cur > 0;
    var bwd: dvui.WidgetData = undefined;
    var opts = icon_opts;
    opts.data_out = &bwd;
    if (!can_back) opts.color_text = style.C.text_ghost;
    if (dvui.buttonIcon(@src(), "prev-variant", dvui.entypo.chevron_small_left, .{}, .{}, opts) and can_back) {
        switch (mirror.navTarget(m.cur, n, .back)) {
            // The host re-syncs the KV for the next turn (or just switches the
            // shown take when nothing is resident) and sends the transcript back.
            .select => |i| post(.{ .chat_select_variant = .{ .msg = @intCast(idx), .variant = @intCast(i) } }),
            else => {},
        }
    }
    if (can_back) hint.hover(@src(), &bwd, "Previous response");

    if (n > 1) dvui.label(@src(), "{d}/{d}", .{ m.cur + 1, n }, .{
        .gravity_y = 0.5,
        .color_text = style.C.text_dim,
        .padding = .{ .x = 2, .w = 2 },
    });

    const at_newest = m.cur + 1 == n;
    var nwd: dvui.WidgetData = undefined;
    opts = icon_opts;
    opts.data_out = &nwd;
    if (dvui.buttonIcon(@src(), "next-variant", dvui.entypo.chevron_small_right, .{}, .{}, opts)) {
        switch (mirror.navTarget(m.cur, n, .next)) {
            .select => |i| post(.{ .chat_select_variant = .{ .msg = @intCast(idx), .variant = @intCast(i) } }),
            // Regenerate needs the LLM: the host runs it now if loaded, else
            // lazy-loads and regenerates once the carried transcript is adopted.
            .regenerate => post(.chat_regenerate),
            .none => {},
        }
    }
    hint.hover(@src(), &nwd, if (at_newest)
        "Generate a new response"
    else
        "Next response");
}

/// Render answer text as markdown with `<image>...</image>` tool-call tags
/// hidden (the images render separately). Uses the same line-anchored matcher
/// as the generation scanner (`chat.nextImageCall`), so a call that fires is
/// exactly a call that's hidden, a casual inline mention of the tag stays
/// visible text. A still-streaming, unterminated call hides everything from
/// it onward. Stripped text is assembled in the frame arena so the markdown
/// parser sees one contiguous document (blocks may span a hidden call).
/// Render one reply IN DOCUMENT ORDER: prose, then wherever the model put a
/// tool call, that call's collapsible section and the card for the images it
/// produced, then the prose after it.
///
/// The previous version stripped every call out, rendered the surviving prose
/// as one block and hung all the images off the end, so a reply that talked
/// between generations read out of order.
///
/// Consecutive calls (nothing but whitespace between them) collapse into ONE
/// card, which is what makes a four-image request a 2x2 grid rather than four
/// stacked cards.
fn renderReply(v: *const mirror.Variant, answer: []const u8, color: dvui.Color, msg: usize, variant: usize) void {
    // Calls this client has placed nowhere yet: they have no image to point at,
    // and the card holds a slot for each so the grid does not grow under the
    // reader as they go out.
    const unplaced = g_hosts.unplacedFor(@intCast(msg), @intCast(variant));
    // The walk itself is pure and tested in toolcall.zig; this only draws.
    var segs: [16]toolcall.Segment = undefined;
    for (toolcall.segments(answer, v.images.items.len, &segs), 0..) |seg, i| switch (seg) {
        .prose => |t| renderProse(i, t, color),
        .calls => |c| renderCallRun(
            v.images.items[c.start..][0..c.len],
            c.n_calls,
            @min(c.n_calls, c.len + unplaced),
            c.text,
            cardKey(msg, variant, i),
            i,
        ),
    };
}

fn renderProse(id: usize, text: []const u8, color: dvui.Color) void {
    markdown_view.render(@src(), text, .{
        .id_extra = id * 64,
        .prose = .{
            .background = false,
            .padding = .{},
            .font = style.F.prose,
            .color_text = color,
        },
    });
}

/// One run of calls the model made in one breath: the collapsible call section
/// above, the card below.
///
/// `imgs` can be SHORTER than `n_calls` (or empty): a render still in the client
/// queue has no image yet, and a reopened conversation only rebuilds the ones
/// whose files it can find (files only exist when saving is on). `slots` is how
/// many tiles the card reserves, which is `imgs` plus what is still coming --
/// never `n_calls`, or a conversation reopened without its files would sit
/// waiting on pictures nobody is making. The section always shows, so a reply
/// never loses the fact that it asked for a picture.
fn renderCallRun(imgs: []const wire.ImageId, n_calls: usize, slots: usize, raw: []const u8, key: usize, id: usize) void {
    if (n_calls == 0) return;

    // Same expander treatment as Thoughts: this is the machine's working, and it
    // sits where the model actually emitted it.
    if (dvui.expander(@src(), "Tool call", .{ .default_expanded = false }, .{
        .id_extra = id * 64 + 1,
        .margin = .{ .y = 6 },
    })) {
        var tl = dvui.textLayout(@src(), .{}, .{
            .id_extra = id * 64 + 2,
            .expand = .horizontal,
            .background = true,
            .color_fill = style.C.sunken,
            .color_border = style.tint(style.C.blue, 120),
            .border = .{ .x = 2 },
            .corner_radius = .{ .x = 0, .y = 6, .w = 6, .h = 0 },
            .margin = .{ .h = 6 },
            .padding = .{ .x = 11, .y = 8, .w = 11, .h = 8 },
            .font = style.F.code,
            .color_text = style.C.text_dim,
        });
        defer tl.deinit();
        fonts.addStyled(tl, raw, .{}, .{ .font = style.F.code, .color_text = style.C.text_dim });
    }

    if (slots > 0) {
        renderToolCard(imgs, slots, key, id * 64 + 3);
        return;
    }

    // Nothing to show, and deliberately nothing regenerated. Say which it is:
    // an image the user chose not to keep is a different situation from one that
    // failed, and neither should look like a bug.
    var buf: [96]u8 = undefined;
    const why = if (g_config.output_dir.opt() == null)
        (std.fmt.bufPrint(&buf, "{d} image{s} generated · not kept (image saving is off)", .{
            n_calls,
            if (n_calls == 1) "" else "s",
        }) catch "generated, not kept")
    else if (!g_m.state.diff_present)
        "generated earlier · no image model loaded to show them"
    else
        (std.fmt.bufPrint(&buf, "{d} image{s} generated · the saved file{s} could not be found", .{
            n_calls,
            if (n_calls == 1) "" else "s",
            if (n_calls == 1) "" else "s",
        }) catch "generated, files missing");

    dvui.labelNoFmt(@src(), why, .{}, .{
        .id_extra = id * 64 + 4,
        .font = style.F.mono,
        .color_text = style.C.text_ghost,
        .padding = .{},
        .margin = .{ .h = 8 },
    });
}


/// Display size for an image: downscale so the longer side is `max`, never
/// upscale. Sizing the widget to the actual (aspect-correct) dimensions avoids
/// the letterbox padding a square cap would add for non-square images.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    const mx: f32 = @floatFromInt(@max(w, h));
    const scale = if (mx > max) max / mx else 1.0;
    return .{ .w = @as(f32, @floatFromInt(w)) * scale, .h = @as(f32, @floatFromInt(h)) * scale };
}

fn renderGenImage(im: *const mirror.Image, gi_idx: usize) void {
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = gi_idx, .expand = .horizontal, .margin = .{ .y = 4, .h = 4 } });
    defer b.deinit();

    const info = &im.info;
    switch (im.status()) {
        .pending, .generating, .suspended => {
            const st_now = im.status();
            const generating = st_now == .generating;
            const done = info.step;
            const total = info.total;
            // Live preview: a fetched frame is a new buffer, so dvui's
            // pointer-keyed texture cache re-uploads once per fetch, not per frame.
            if (im.preview) |pv| {
                const pw = im.preview_w;
                const ph = im.preview_h;
                if (pw > 0 and ph > 0) {
                    const sz = fitSize(pw, ph, 200);
                    _ = dvui.image(@src(), .{
                        .source = .{ .pixels = .{ .rgba = pv, .width = pw, .height = ph } },
                        .shrink = .ratio,
                    }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                }
            }
            // Live timing: elapsed since dispatch, average s/step over completed
            // steps (excludes model-load time), and an ETA from that rate. Keep
            // the frame repainting so the elapsed timer ticks between step wakes.
            if (generating) dvui.refresh(null, @src(), null);
            const start = info.start_ns;
            const first = info.first_step_ns;
            const last = info.last_step_ns;
            const now_ns: i64 = @intCast(std.Io.Clock.real.now(g_io).nanoseconds);
            const elapsed_s: f64 = if (start > 0) @as(f64, @floatFromInt(now_ns - start)) / 1e9 else 0;
            const sps: f64 = if (done >= 2 and first > 0 and last > first)
                (@as(f64, @floatFromInt(last - first)) / 1e9) / @as(f64, @floatFromInt(done - 1))
            else
                0;
            const eta_s: f64 = if (sps > 0 and total > done) sps * @as(f64, @floatFromInt(total - done)) else 0;

            var buf: [112]u8 = undefined;
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
            const status = switch (st_now) {
                .suspended => std.fmt.bufPrint(&buf, "⏸  Suspended at step {d}/{d} — resume to continue", .{ done, total }) catch "⏸ Suspended",
                .pending => "🖼  Queued…",
                else => if (sps > 0)
                    std.fmt.bufPrint(&buf, "🖼  Generating…  step {d}/{d}  ·  {d:.2} s/step", .{ done, total, sps }) catch "Generating…"
                else
                    std.fmt.bufPrint(&buf, "🖼  Generating…  step {d}/{d}", .{ done, total }) catch "Generating…",
            };
            fonts.addRich(tl, status);
            tl.deinit();
            const pct: f32 = if (total > 0) @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total)) else 0;
            dvui.progress(@src(), .{ .percent = pct }, .{ .expand = .horizontal, .min_size_content = .{ .h = 6 }, .corner_radius = dvui.Rect.all(3) });
            if (generating and elapsed_s > 0) {
                var tbuf: [80]u8 = undefined;
                const timing = if (eta_s > 0)
                    std.fmt.bufPrint(&tbuf, "{d:.1}s elapsed  ·  ~{d:.1}s left", .{ elapsed_s, eta_s }) catch ""
                else
                    std.fmt.bufPrint(&tbuf, "{d:.1}s elapsed", .{elapsed_s}) catch "";
                dvui.label(@src(), "{s}", .{timing}, .{ .margin = .{ .y = 1 } });
            }
            genInfo(im);
            // Stop this generation (or drop it from the queue): the host sets
            // the flag the pipeline polls between steps.
            if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .{ .y = 4 } })) {
                post(.{ .img_cancel = .{ .image = info.id } });
            }
        },
        .done => {
            if (im.pixels) |rgba| {
                // Wrap in a box so a click anywhere on the image opens the
                // full-size viewer window. Size to the image's own aspect so
                // there's no letterbox padding before the button.
                const sz = fitSize(info.width, info.height, 200);
                var ib = dvui.box(@src(), .{}, .{});
                _ = dvui.image(@src(), .{
                    .source = .{ .pixels = .{ .rgba = rgba, .width = info.width, .height = info.height } },
                    .shrink = .ratio,
                }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                const clicked = dvui.clicked(ib.data(), .{});
                ib.deinit();
                if (clicked) g_viewer_request = info.id;

                genInfo(im);

                {
                    var actions = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .y = 4 } });
                    defer actions.deinit();
                    // Copy the image to the clipboard as a PNG.
                    var cwd: dvui.WidgetData = undefined;
                    if (dvui.buttonIcon(@src(), "copy", dvui.entypo.clipboard, .{}, .{}, .{
                        .min_size_content = .{ .w = 18, .h = 18 },
                        .gravity_y = 0.5,
                        .data_out = &cwd,
                    })) clipboard.copyImage(rgba, info.width, info.height);
                    hint.hover(@src(), &cwd, "Copy image to clipboard");

                    // Let the model see this image: attach it to the next
                    // message. Shown whenever the configured model can see
                    // images (not just while a session is resident); with the
                    // LLM unloaded the host stages it and the lazy load kicks.
                    if (g_m.state.vision and (!im.local or im.pixels != null)) {
                        if (dvui.button(@src(), "Discuss this image", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 6 } })) {
                            attachFromMirror(im);
                        }
                    }
                }
            }
        },
        .failed => {
            // Name the cause. "failed" alone is unactionable, and the most
            // common cause here, VRAM, is one the user can actually fix
            // (unload the LLM, drop the resolution) and then retry into.
            var fbuf: [160]u8 = undefined;
            const msg = if (info.failure.len > 0)
                std.fmt.bufPrint(&fbuf, "image generation failed: {s}", .{mirror.failureText(info.failure)}) catch "image generation failed"
            else
                "image generation failed";
            fonts.richLabel(@src(), msg, .{});
            genInfo(im);
            retryButton(im);
        },
        .canceled => {
            fonts.richLabel(@src(), "image generation canceled", .{});
            genInfo(im);
            retryButton(im);
        },
    }
}

/// "Try again" for a failed or canceled image: asks for the same render again,
/// as a new job that any host can take. It picks up the CURRENT model and
/// backend, which is the point, since the usual fix for the usual failure is to
/// change something first. A local image (a reopened render whose file is gone)
/// has nothing to retry.
fn retryButton(im: *const mirror.Image) void {
    if (im.local) return;
    if (dvui.button(@src(), "Try again", .{}, .{ .margin = .{ .y = 4 } }))
        g_hosts.retry(&g_config, im.info.id);
}

/// A compact metadata line (resolution * seed) plus a collapsed-by-default
/// prompt expander, shown under an image in every state. Uses the actual output
/// dimensions once known, else the requested ones; the seed is shown once
/// assigned (non-zero).
fn genInfo(im: *const mirror.Image) void {
    const info = &im.info;
    const w = if (info.width > 0) info.width else info.req_width;
    const h = if (info.height > 0) info.height else info.req_height;
    var buf: [160]u8 = undefined;
    // Seed is assigned at scan time (never 0 here), so always show it. Once
    // done, append the average s/step (sampling only) and total wall time.
    const meta = if (im.status() == .done) blk: {
        const start = info.start_ns;
        const dn = info.done_ns;
        const first = info.first_step_ns;
        const last = info.last_step_ns;
        const total_s: f64 = if (start > 0 and dn > start) @as(f64, @floatFromInt(dn - start)) / 1e9 else 0;
        const sps: f64 = if (info.req_steps >= 2 and first > 0 and last > first)
            (@as(f64, @floatFromInt(last - first)) / 1e9) / @as(f64, @floatFromInt(info.req_steps - 1))
        else
            0;
        break :blk std.fmt.bufPrint(&buf, "{d}×{d}  ·  {d} steps  ·  seed {d}  ·  {d:.2} s/step  ·  {d:.1}s total", .{ w, h, info.req_steps, info.req_seed, sps, total_s }) catch "";
    } else std.fmt.bufPrint(&buf, "{d}×{d}  ·  {d} steps  ·  seed {d}", .{ w, h, info.req_steps, info.req_seed }) catch "";
    // Rendered through a text layout (not a plain label) so the metadata,
    // seed especially, is mouse-selectable.
    fonts.richLabel(@src(), meta, .{ .margin = .{ .y = 2 } });
    if (dvui.expander(@src(), "Prompt", .{ .default_expanded = false }, .{})) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = .{ .x = 6, .y = 2, .w = 6, .h = 4 } });
        defer tl.deinit();
        fonts.addRich(tl, info.prompt);
    }
}

const Parsed = struct { think: ?[]const u8, answer: []const u8, thinking: bool };

/// Split an assistant message into its reasoning block and the answer, using
/// the active model family's thought markers (chat.reasoning(), e.g. Qwen's
/// `<think>...</think>`, Gemma 4's `<|channel>thought...<channel|>`). The markers
/// themselves are dropped. `thinking` is true while the block is still open (no
/// close marker yet). Families that don't reason return everything as answer.
///
/// `primed` says the prompt already opened the block, so the generated text
/// starts INSIDE the thought and never emits an opening marker, the render-driven
/// (template) path does exactly that. It is recorded per variant at generation
/// time (`Variant.thought_primed`), because a session can render both ways over
/// its lifetime. See `toolcall.splitThought`, which owns the rule so the display
/// and the tool-call scanner cannot disagree about what is inside a thought.
/// Split a variant with ITS OWN markers (see chat.Variant.markersFor): the ones
/// recorded when it was generated, else the live model's, else none. Splitting
/// with the live model's markers is what left a reopened conversation showing a
/// bare `</think>` in its prose.
fn parseThink(v: *const mirror.Variant) Parsed {
    const s = toolcall.splitThought(v.text.items, markersFor(v), v.thought_primed);
    return .{ .think = s.think, .answer = s.answer, .thinking = s.open };
}

/// One pending-attachment thumbnail (56px, RGBA) with a hover-only X. Returns
/// true if its remove button was clicked this frame. Shared by the session and
/// pre-load staging strips (`renderInput`).
fn renderPendingThumb(pi: usize, rgba: []const u8, w: usize, h: usize) bool {
    const sz = fitSize(w, h, 56);
    var ov = dvui.overlay(@src(), .{ .id_extra = pi, .margin = .{ .w = 6 } });
    defer ov.deinit();
    _ = dvui.image(@src(), .{
        .source = .{ .pixels = .{ .rgba = rgba, .width = @intCast(w), .height = @intCast(h) } },
        .shrink = .ratio,
    }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(4) });
    // Non-consuming hover test (leaves the click for the X button).
    const hovered = ov.data().rectScale().r.contains(dvui.currentWindow().mouse_pt);
    if (hovered) {
        if (dvui.buttonIcon(@src(), "remove", dvui.entypo.cross, .{}, .{}, .{
            .gravity_x = 1.0,
            .gravity_y = 0.0,
            .min_size_content = .{ .w = 12, .h = 12 },
            .padding = dvui.Rect.all(2),
            .corner_radius = dvui.Rect.all(3),
        })) return true;
    }
    return false;
}

fn renderInput() void {
    var container = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
    defer container.deinit();

    const st = &g_m.state;
    // Thumbnails of images attached but not yet sent, each with a hover-only X
    // to remove it before sending. Once the LLM is live these are the session's
    // pending attachments (mirrored images whose pixels were fetched); before
    // the lazy first-message load they are this client's own staged copies.
    const n_thumbs = if (st.llm_resident) st.attachments.len else g_staged.items.len;
    if (n_thumbs > 0) {
        var remove_idx: ?usize = null;
        {
            var strip = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 8, .y = 4, .w = 8 } });
            defer strip.deinit();
            if (st.llm_resident) {
                for (st.attachments, 0..) |id, pi| {
                    const im = g_hosts.imageById(id) orelse continue;
                    if (im.pixels) |rgba| if (renderPendingThumb(pi, rgba, im.info.width, im.info.height)) {
                        remove_idx = pi;
                    };
                }
            } else {
                for (g_staged.items, 0..) |sti, pi| {
                    if (renderPendingThumb(pi, sti.rgba, sti.width, sti.height)) remove_idx = pi;
                }
            }
        }
        if (remove_idx) |ri| {
            post(.{ .chat_remove_attachment = .{ .index = @intCast(ri) } });
            if (!st.llm_resident and ri < g_staged.items.len) {
                const gone = g_staged.orderedRemove(ri);
                g_gpa.free(gone.rgba);
            }
        }
    }

    var block = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        // Background stated explicitly: dvui forces one on for a NON-UNIFORM
        // border, and the fill it picks is the theme's, which is the wrong
        // surface anywhere but the canvas. Saying so also silences the warning.
        .background = true,
        .color_fill = style.C.canvas,
        .border = style.Edge.top,
        .color_border = style.hairline_soft,
        .padding = .{ .x = 30, .y = 12, .w = 30, .h = 18 },
    });
    defer block.deinit();

    const busy = st.llm_busy;

    // Quick settings: the three knobs a beginner ever needs. Everything else is
    // in Studio, and the link says so rather than leaving the user to guess.
    {
        var quick = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .h = 10 } });
        defer quick.deinit();

        // Ratio and pixel budget, not raw width/height: those are the two
        // things a person decides, and since Settings no longer has size rows
        // these chips are the single source for it. Both persist.
        var ratio_i: usize = @intFromEnum(g_config.framing_ratio);
        const ratio_before = ratio_i;

        const ratios = comptime blk: {
            var out: [std.enums.values(framing.Ratio).len][]const u8 = undefined;
            for (std.enums.values(framing.Ratio), 0..) |r, i| out[i] = r.label();
            break :blk out;
        };
        if (style.chipDropdown(@src(), &ratios, &ratio_i, .{ .id_extra = 0 }) and ratio_i != ratio_before) {
            g_config.framing_ratio = @enumFromInt(ratio_i);
            applyFraming();
        }

        // Megapixels is TYPED, not picked: the useful values are continuous
        // ("1.8"), and a preset list either omits the one you want or grows
        // until it is a worse menu than a number.
        const mp_res = style.chipInput(@src(), &g_mp_buf, "MP", 30, .{ .id_extra = 1, .margin_x = 8 });
        // COMMIT BEFORE SYNC. On the frame focus leaves, the field is no longer
        // being edited AND has just committed; syncing first would overwrite
        // the typed text with the old config value and then parse that.
        if (mp_res.committed) {
            const typed = std.mem.trim(u8, std.mem.sliceTo(&g_mp_buf, 0), " \t");
            if (std.fmt.parseFloat(f32, typed)) |v| {
                const clamped = std.math.clamp(v, @as(f32, @floatCast(framing.mp_min)), @as(f32, @floatCast(framing.mp_max)));
                if (clamped != g_config.framing_mp) {
                    g_config.framing_mp = clamped;
                    applyFraming();
                }
            } else |_| {}
        }
        if (!mp_res.editing) {
            // Show the live value, but never while it is being typed into.
            // After a commit this also replaces garbage or an out-of-range
            // number with what was actually accepted.
            var tmp: [16]u8 = undefined;
            const want = framing.formatMp(&tmp, g_config.framing_mp);
            if (!std.mem.eql(u8, std.mem.sliceTo(&g_mp_buf, 0), want)) {
                @memset(&g_mp_buf, 0);
                @memcpy(g_mp_buf[0..want.len], want);
            }
        }

        // The reasoning toggle is a quick setting too: it changes what the next
        // turn does, which is what this row is for. Only for a family that can.
        const can_reason = st.thinking;
        if (can_reason) {
            const on = g_config.reasoning;
            if (style.chip(@src(), if (on) "thinking: on" else "thinking: off", .{
                .id_extra = 2,
                .margin_x = 8,
                .fill = if (on) style.over(style.C.chip, style.C.blue, 0.16) else style.C.chip,
                .text = if (on) style.C.blue else style.C.text_dim,
            })) toggleReasoning();

            const has_effort = st.reasoning_effort;
            if (on and has_effort) {
                const label = switch (g_config.reasoning_effort) {
                    .high => "effort: high",
                    .medium => "effort: medium",
                    .low => "effort: low",
                };
                if (style.chip(@src(), label, .{ .id_extra = 3, .margin_x = 8 })) cycleReasoningEffort();
            }
        }

        var link: dvui.ButtonWidget = undefined;
        link.init(@src(), .{}, .{
            .gravity_x = 1.0,
            .gravity_y = 0.5,
            .background = false,
            .color_fill_hover = style.hover_wash,
            .color_fill_press = style.hover_wash,
            .corner_radius = style.R.chip,
            .padding = .{ .x = 5, .y = 3, .w = 5, .h = 3 },
            .margin = .{},
        });
        link.processEvents();
        link.drawBackground();
        {
            var lr = dvui.box(@src(), .{ .dir = .horizontal }, .{});
            defer lr.deinit();
            dvui.labelNoFmt(@src(), "All settings in Studio", .{}, .{
                .font = style.F.ui_sm,
                .color_text = style.C.text_ghost,
                .padding = .{},
                .gravity_y = 0.5,
            });
            style.mark(@src(), .external, 9, style.C.text_ghost, .{ .margin = .{ .x = 4 } });
        }
        const go = link.clicked();
        link.deinit();
        if (go) enterImageMode();
    }

    // Focus from LAST frame's id: the entry is created below, and a one-frame
    // late ring is invisible.
    const input_focused = if (g_input_id) |id| dvui.focusedWidgetId() == id else false;
    var frame_box = bubbles.inputBegin(@src(), input_focused);

    var send = false;

    // Enter (without Shift) on the focused input sends. Consume the key before
    // the multiline entry turns it into a newline; Shift+Enter falls through as
    // a newline. Disabled while generating so the box stays editable.
    if (!busy) {
        // Paste an image from the clipboard: intercept before the text entry so
        // an image on the clipboard attaches instead of a bogus text paste.
        // Text-only clipboards fall through untouched. Handled regardless of
        // focus so it works right after clicking into chat.
        //
        // Matched against dvui's own "paste" bind rather than a literal Ctrl+V,
        // so every chord the text entry pastes on also attaches an image:
        // Ctrl+V, Cmd+V, and Shift+Insert. Spelling the chord out here meant
        // Shift+Insert reached the entry, which pasted the clipboard's TEXT (so
        // an image clipboard did nothing at all).
        for (dvui.events()) |*e| {
            if (e.handled or e.evt != .key) continue;
            const k = e.evt.key;
            if (k.action == .down and k.matchBind("paste")) {
                if (tryPasteClipboardImage()) e.handled = true;
            }
        }
        if (g_input_id) |id| {
            if (dvui.focusedWidgetId()) |fid| {
                if (fid == id) {
                    for (dvui.events()) |*e| {
                        if (e.handled or e.evt != .key) continue;
                        const k = e.evt.key;
                        if ((k.code == .enter or k.code == .kp_enter) and k.action == .down and !k.mod.shift()) {
                            e.handled = true;
                            send = true;
                        }
                    }
                }
            }
        }
    }

    var te = dvui.textEntry(@src(), .{
        .text = .{ .array_list = .{
            .backing = &g_input,
            .allocator = g_gpa,
            .limit = input_limit,
        } },
        .multiline = true,
        .placeholder = "Describe an image, or ask for changes…",
        .scroll_horizontal = false,
        .break_lines = true,
    }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        // The sunken frame around it is the input's chrome (see bubbles.zig);
        // the entry itself draws nothing.
        .background = false,
        .border = .{},
        .padding = .{},
        .font = style.F.input,
        .color_text = style.C.text_hi,
        .theme = style.noFocusTheme(), // the frame shows focus, not the entry
        .min_size_content = .{ .h = 20 },
        // Bound BOTH dimensions of the reported min size. dvui's TextLayout
        // accumulates its min width across every soft-wrapped line (it only
        // resets on a literal '\n', not on a wrap), so a long wrapped message
        // reports a min width proportional to the whole text. With `.expand`
        // horizontal in this row, that ballooning min would overflow the row
        // and shove the Send button off-screen. Capping max_size_content.w
        // clamps the *reported* min (minSizeSetAndRefresh) while expand still
        // stretches the entry to the real available width and break_lines wraps
        // at that width, so no horizontal scroll and Send never collapses.
        .max_size_content = .size(.{ .w = 160, .h = 140 }),
    });
    g_input_id = te.data().id;
    // Reserve for next frame's layout: entry height, plus the frame's padding,
    // the quick-settings row, the block padding, and the attachment strip when
    // present.
    g_input_h = te.data().rect.h + 100 + (if (n_thumbs > 0) @as(f32, 72) else 0);
    te.deinit(); // publishes the text back into `g_input.items`

    const pressed = bubbles.inputEnd(&frame_box, .{
        .busy = busy,
        .can_attach = st.vision,
        // Gated on CAPABILITY, not on a live session: the model is lazy-loaded, so
        // requiring one hid these controls until after the first message. Only a
        // checkpoint whose arch publishes a layer index and whose linears are in a
        // wired dtype gets them, so the affordance never promises an effect it
        // cannot have (see `noiseAvailable`).
        .noise = if (st.weight_noise) .{
            .on = g_config.weight_noise,
            .shape = noiseShape(),
            .valid = g_noise_valid,
            .names = noiseNames(),
            .sel = &g_noise_sel,
            .amount_buf = &g_noise_amt_buf,
            .amount = g_config.weight_noise_amount,
            .amount_live = g_noise_amount_live,
        } else null,
    }, .{
        .ctx = @ptrCast(&g_composer_ctx),
        .on_quick = composerQuick,
        .on_all_settings = composerAllSettings,
        .on_reference = composerReference,
        .on_noise_toggle = composerNoiseToggle,
        .on_noise_pick = composerNoisePick,
        .on_noise_amount = composerNoiseAmount,
    });
    if (pressed) {
        if (busy) post(.chat_cancel) else send = true;
    }

    if (send and !busy) {
        // Read straight from the backing: both `newChat` and `submitChat` copy what
        // they keep, so nothing here outlives the clear below.
        const text = g_input.items;
        // `/new` on its own line starts a fresh chat instead of sending.
        if (std.mem.eql(u8, std.mem.trim(u8, text, " \t\r\n"), "/new")) {
            newChat();
        } else {
            _ = submitChat(text);
        }
        g_input.clearRetainingCapacity();
    }
}
