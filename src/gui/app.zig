//! tp-gui application: window, manual render loop, and the chat frame.
//!
//! The loop mirrors DiffKeep's hand-written SDL loop (rather than
//! `dvui.App.run`) so that secondary windows — notably a zoom/pan viewer for
//! generated images — can be added later without restructuring. For now there
//! is a single window; per-window event routing is introduced when the first
//! secondary window lands.
const std = @import("std");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const tp = @import("TensorPencil");
const vram = tp.vram;
const chat = @import("chat.zig");
const toolcall = @import("toolcall.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");
const markdown_view = @import("markdown_view.zig");
const viewer = @import("viewer.zig");
const config = @import("config.zig");
const config_view = @import("config_view.zig");
const image_view = @import("image_view.zig");
const diffuser = @import("diffuser.zig");
const clipboard = @import("clipboard.zig");
const meter = @import("meter.zig");
const status_bar = @import("status_bar.zig");
const vips = @import("vips");

// dvui frames are argless, so app state is process-global (the dvui idiom).
//
// The session (LLM + optional diffusion/vision) is (re)loaded on a background
// thread whenever settings change, so it can be swapped without a restart and
// the UI stays responsive with a "Loading…" state. `g_session` is only read by
// the UI thread while `g_loading` is false (release/acquire hand-off from the
// loader), so the pointer swap is race-free without a lock.
var g_session: ?*chat.Session = null;
/// The single owner of LLM↔diffusion VRAM arbitration. Its `llm` participant is
/// (re)bound to `g_session` under `g_session_mu` on every load/unload. Driven by
/// the coordinator hooks (`vcEnter`/`vcExit`) and the meter policy.
var g_arbiter: vram.Arbiter = .{};
var g_session_arena: ?*std.heap.ArenaAllocator = null; // load-once weights, freed on reload
var g_loading: std.atomic.Value(bool) = .init(false);
var g_loader: ?std.Thread = null;
var g_reload_requested: bool = false;
// Text stashed when the user sends the first chat message with no LLM resident:
// the model lazy-loads, then this is auto-submitted (see maybeStartReload).
var g_pending_submit: ?[]u8 = null;

// Images dropped/pasted before the LLM has loaded (the lazy first message). Held
// here with their decoded RGB (fed to `attachImage` once a session exists) plus
// a display RGBA (for the pre-load thumbnail strip). `maybeStartReload` drains
// them into the fresh session so the first message carries its attachments.
const StagedImage = struct { rgb: []u8, rgba: []u8, width: usize, height: usize };
var g_staged_images: std.ArrayList(StagedImage) = .empty;

// Cached "does the configured LLM support a reasoning block?" answer, so the
// thinking toggle can show before the model loads. Reading a GGUF header is
// cheap but not per-frame cheap, so we memoize and re-probe only when the
// configured model path changes (including a Settings model swap).
var g_think_probe_path: [config.max_path]u8 = undefined;
var g_think_probe_len: usize = 0;
var g_think_probe_result: bool = false;
var g_think_probe_valid: bool = false;

// The diffusion engine — APP-LEVEL and persistent (survives chat↔image mode
// switches, so the image model isn't reloaded each way). It owns the single
// unified image queue/history shared by the chat tool-call path and the studio.
// Built when a diffusion model is configured; its resident pipeline still loads
// lazily on the first image. The VRAM coordinator it's given dispatches to
// `g_session` when an LLM is resident, else no-ops (diffusion gets everything).
var g_diffuser: ?diffuser.Diffuser = null;
// VRAM meter handle positions (fractions of the card): split = LLM|diffusion
// contention boundary, limit = ceiling. Seeded from config at startup, mutated
// in place on drag, and (on release) persisted + applied live. Defaults are the
// config defaults; run() overwrites them from the loaded config.
var g_split: f32 = 0.60;
var g_limit: f32 = 0.95;
// Eject (⏏) state: set when the user clicks an end button. Fully unloads that
// model to free VRAM. If the model is busy the request stays ARMED and fires
// once it (and the shared image queue) go idle — see maybeProcessEjects. The
// meter renders the armed state (accent border) so the deferral is visible.
var g_llm_eject_armed: bool = false;
var g_diff_eject_armed: bool = false;
// LLM pause is mirrored app-level so it survives an unload (the gate itself lives
// on the Session, which is destroyed on unload — unlike the diffuser's gate,
// which lives on the persistent Diffuser). The button reads this; setPaused keeps
// the session gate in sync while one is resident. (Tier 3.)
var g_llm_paused: bool = false;
// In-flight LLM state saved on an unload-while-paused: the raw `ids` (prompt +
// partial open response) carried across the unload so a reload can reprefill +
// continue that exact response. `g_carry` holds the display transcript alongside.
const LlmSuspend = struct { ids: []u32, midturn: bool };
var g_llm_suspend: ?LlmSuspend = null;
// Set when a reload should CONTINUE a suspended mid-turn response (spawn a decode
// worker after the fresh session adopts the carried `ids`). (Tier 3.)
var g_pending_continue: bool = false;
// Guards `g_session` teardown against the diffusion WORKER thread, which reads
// the session in its VRAM-coordinator hooks (budget/reclaim). Held while a full
// LLM eject frees the session so a concurrent worker can't touch freed memory.
// The UI-thread hooks (enter/exit) never overlap the eject (same thread), so
// only the worker-thread hooks + the teardown take it.
var g_session_mu: std.Io.Mutex = std.Io.Mutex.init;

/// on_change: fired every drag-motion frame. The meter already mutated
/// g_split/g_limit in place; motion repaints on its own, so this is a no-op (we
/// deliberately do NOT reshuffle VRAM mid-drag — only on release).
fn meterChanged() void {}

/// on_commit: fired on drag release — persist the settled fractions and apply
/// the new policy to the live session.
fn meterCommit() void {
    g_config.vram_split = g_split;
    g_config.vram_limit_frac = g_limit;
    g_config_baseline.vram_split = g_split;
    g_config_baseline.vram_limit_frac = g_limit;
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    applyMeterPolicy();
}

/// Apply the meter policy (soft residency): the limit is a WHOLE-CARD ceiling,
/// so our budget for LLM + diffusion is `limit − system` (system = OS/desktop +
/// CUDA-context overhead we don't control). The LLM offloads weights to CPU to
/// fit (cheap; doesn't slow a running image, so it runs even during diffusion
/// generation, serialized with the worker's reclaim hook). Diffusion's resident
/// weights are trimmed to fit ONLY while idle — mid-image eviction would force
/// per-step streaming, which soft residency avoids; the next image re-budgets at
/// its start via imageBudget. No-op with no session or mid-load.
fn applyMeterPolicy() void {
    const s = g_session orelse return;
    if (g_loading.load(.acquire)) return;
    // cuMemGetInfo reads the CALLING thread's current CUDA context, and this
    // runs on the UI thread — which starts with none bound (the LLM's context
    // is created on the loader thread). Bind it first, or the query fails and
    // returns zeros: that silent zero used to skip `setBudgets` entirely,
    // leaving the arbiter uninitialized (the qwen3-32B first-message
    // mass-offload bug). Binding here is the same thing the idle-settle path
    // (`vpApply`) already does on this thread.
    s.be.bindThread();
    const mi = s.be.ctx.memGetInfo();
    const total: u64 = mi.total;
    if (total == 0) {
        std.log.warn("[vram] meter policy skipped: VRAM query failed — arbiter budgets NOT updated", .{});
        return;
    }
    const free_b: u64 = mi.free;
    const used_all: u64 = total -| free_b;
    const llm_res: u64 = s.be.deviceUsed();
    const diff_res: u64 = if (g_diffuser) |*d| d.vramBytes() else 0;
    const system = used_all -| llm_res -| diff_res;
    const tf: f32 = @floatFromInt(total);
    const limit_bytes: u64 = @intFromFloat(g_limit * tf);
    const available: u64 = limit_bytes -| system; // budget for LLM + diffusion
    const share: u64 = @min(@as(u64, @intFromFloat(g_split * tf)), available);
    const diff_busy = if (g_diffuser) |*d| d.busyNow() else false;

    // LLM side: hand the resolved ceiling + share to the arbiter, which settles
    // the LLM to fit (offloading weights to the CPU, or promoting them back).
    // Guarded so a direct (idle) settle can't race the diffusion worker's reclaim
    // hook — both touch the LLM context.
    g_session_mu.lockUncancelable(g_io);
    // Mirror the resolved policy onto the session for the status-bar display and
    // the reset/reclaim ceiling (these fields outlive the arbiter's live state).
    s.vram_limit = available;
    s.vram_share = share;
    s.vram_budget = available;
    g_arbiter.diff_used = diff_res;
    std.log.info("[vram] meter policy: limit {d} MiB − system {d} → budget {d} MiB · LLM share {d} (LLM {d} + diff {d} resident, {d} free)", .{
        limit_bytes >> 20, system >> 20, available >> 20, share >> 20, llm_res >> 20, diff_res >> 20, free_b >> 20,
    });
    g_arbiter.setBudgets(available, share);
    g_session_mu.unlock(g_io);

    // Diffusion side: incrementally free resident weights to fit the room left
    // after the LLM (keeping the rest resident so the next image reloads less),
    // but only when idle (soft residency — no mid-image streaming).
    if (!diff_busy) if (g_diffuser) |*d| {
        _ = d.giveUpToBudget(available -| s.be.deviceUsed());
    };
}

/// Restore a persisted window geometry onto a freshly created SDL window. Size
/// is applied first, then position (only when one was saved — otherwise SDL's
/// default placement stands), then maximize last so it overrides the rect.
fn applyWindowGeom(window: ?*SDLBackend.c.SDL_Window, w: usize, h: usize, x: i32, y: i32, maximized: bool) void {
    _ = SDLBackend.c.SDL_SetWindowSize(window, @intCast(w), @intCast(h));
    if (x != config.pos_unset and y != config.pos_unset)
        _ = SDLBackend.c.SDL_SetWindowPosition(window, x, y);
    if (maximized) _ = SDLBackend.c.SDL_MaximizeWindow(window);
}

/// Read a window's current geometry into the given config fields, returning
/// whether anything changed. While the window is maximized (or minimized) the
/// size/position are left alone — the stored values keep the last *restored*
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
/// mid-edit Settings text) with the current geometry overlaid — this keeps
/// Settings → Cancel able to discard unsaved edits while a concurrent resize
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

fn meterEjectLlm() void {
    g_llm_eject_armed = true; // fires (or fires now) via maybeProcessEjects
}
fn meterEjectDiff() void {
    g_diff_eject_armed = true;
}
fn meterActions() meter.Actions {
    return .{ .on_change = meterChanged, .on_commit = meterCommit, .on_eject_llm = meterEjectLlm, .on_eject_diff = meterEjectDiff, .on_toggle_pause_llm = toggleLlmPause, .on_toggle_pause_diff = toggleDiffPause };
}

/// Diffusion's gate lives on the persistent Diffuser, so it IS the source of
/// truth (queried live, survives an unload). The LLM mirrors its state in
/// `g_llm_paused` because the gate dies with the session on unload.
fn llmPaused() bool {
    return g_llm_paused;
}
fn diffPaused() bool {
    return if (g_diffuser) |*d| d.isPaused() else false;
}
fn toggleLlmPause() void {
    const now_paused = !g_llm_paused;
    g_llm_paused = now_paused;
    if (g_loading.load(.acquire)) return; // applied to the fresh gate on publish
    if (g_session) |s| {
        // Resident: drive the session gate (unpause also dispatches a turn that
        // was queued while paused — see Session.setPaused).
        s.setPaused(now_paused);
    } else if (!now_paused and g_llm_suspend != null) {
        // Unloaded + resuming with a suspended turn: reload, then continue it.
        g_reload_requested = true;
        wakeupFrame();
    }
}
fn toggleDiffPause() void {
    if (g_diffuser) |*d| d.setPaused(!d.isPaused());
}

/// Main-loop hook: carry out any armed eject once ITS OWN model is idle. Each
/// model ejects independently — the LLM can drop while diffusion is still
/// generating (it isn't generating anything, so there's nothing to wait for),
/// and vice versa. A model that's busy when clicked stays armed and ejects the
/// moment it finishes.
fn maybeProcessEjects() void {
    if (g_diff_eject_armed) {
        if (g_diffuser) |*d| {
            if (d.isPaused()) {
                // Unload-while-paused (Tier 3): snapshot the in-flight image to
                // host, then free the weights and KEEP the queue (incl. the
                // suspended image), which resumes on unpause. Free even with
                // pending images — they're parked by the pause gate anyway.
                if (d.busyNow()) {
                    d.requestSuspend(); // worker snapshots + exits; poll next frame
                } else {
                    d.reapAndFree();
                    g_diff_eject_armed = false;
                    applyMeterPolicy();
                }
            } else if (!d.busyNow() and !d.hasPending()) {
                d.freeSession();
                g_diff_eject_armed = false;
                // Diffusion is gone: let the LLM borrow the freed VRAM back.
                applyMeterPolicy();
            }
        } else g_diff_eject_armed = false;
    }
    if (g_llm_eject_armed) {
        if (g_session == null or g_loading.load(.acquire)) {
            g_llm_eject_armed = false; // nothing loaded / a (re)load is in flight
        } else if (g_session) |s| {
            if (g_llm_paused and s.busy()) {
                // Unload-while-paused (Tier 3): the worker is parked mid-decode.
                // Ask it to suspend (stop with the turn left OPEN); we finish the
                // unload once it clears below.
                s.requestSuspend();
            } else if (!s.busy()) {
                // Fire as soon as the LLM itself is idle — do NOT wait on
                // diffusion. The worker only touches the session through the
                // coordinator hooks, which unloadLlm serializes with g_session_mu.
                // If we suspended a mid-turn response, carry its raw `ids` so the
                // reload can reprefill + continue it.
                if (g_llm_paused and s.suspended_midturn) saveLlmSuspend(s);
                unloadLlm();
                g_llm_eject_armed = false;
            }
        }
    }
}

/// Carry the suspended LLM's raw `ids` (prompt + partial open response) across an
/// unload-while-paused so a reload can reprefill + continue it. `g_carry` holds
/// the display transcript alongside. Drains pending bytes first so the displayed
/// text matches the tokens. (Tier 3.)
fn saveLlmSuspend(s: *chat.Session) void {
    s.poll(); // drain streamed bytes into messages so display matches `ids`
    const ids = g_gpa.dupe(u32, s.ids.items) catch |err| {
        std.log.err("save suspend ids: {t}", .{err});
        return;
    };
    if (g_llm_suspend) |old| g_gpa.free(old.ids); // one suspend at a time
    g_llm_suspend = .{ .ids = ids, .midturn = s.suspended_midturn };
}

fn freeLlmSuspend() void {
    if (g_llm_suspend) |sus| {
        g_gpa.free(sus.ids);
        g_llm_suspend = null;
    }
    g_pending_continue = false;
}

/// Fully unload the LLM (free its VRAM) while KEEPING the conversation: the
/// transcript is detached into `g_carry` (rendered read-only until a message
/// reloads + replays it — see renderMessages), never wiped. Runs synchronously
/// on the UI thread (LLM is idle here); the teardown is serialized with the
/// diffusion worker's session access via `g_session_mu`. A model swap / new-chat
/// is unaffected — only a "new chat" click ever resets the transcript.
fn unloadLlm() void {
    const s = g_session orelse return;
    if (s.worker) |t| { // idle here, but be safe
        t.join();
        s.worker = null;
    }
    g_session_mu.lockUncancelable(g_io);
    s.be.bindThread(); // context current on THIS thread to free its device memory
    g_carry = s.detachTranscript();
    s.deinit();
    g_session = null;
    g_arbiter.llm = null; // participant points into the freed session
    g_session_mu.unlock(g_io);
    if (g_session_arena) |a| {
        a.deinit();
        g_gpa.destroy(a);
        g_session_arena = null;
    }
    // Diffusion (if resident) can now borrow the whole card: with no session,
    // imageBudget returns 0 (pin all free VRAM) on the next image.
    wakeupFrame();
}
// The conversation transcript carried across a model-swap reload (detached from
// the old session before teardown, adopted by the new one) so a settings save
// never wipes the chat. Owned by g_gpa; freed if there's no new session to adopt.
var g_carry: ?std.ArrayList(chat.Message) = null;

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

var g_input_buf: [4096]u8 = [_]u8{0} ** 4096;
var g_wakeup_event_type: u32 = 0;
var g_load_err: ?anyerror = null;
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
var g_viewer_request: ?*chat.GenImage = null;

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
    g_config = config.Config.load(init.io, gpa, init.environ_map, g_config_path);
    if (model_override) |m| g_config.llm_model.set(m);
    g_config_baseline = g_config;
    // Seed the meter handles from the persisted fractions, clamped into the
    // grabbable range (recovers a config that saved a stuck limit at the edge).
    g_split = std.math.clamp(g_config.vram_split, 0.02, 0.96);
    g_limit = std.math.clamp(g_config.vram_limit_frac, 0.10, 0.985);

    var back = try SDLBackend.initWindow(.{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = @floatFromInt(g_config.win_w), .h = @floatFromInt(g_config.win_h) },
        .min_size = .{ .w = 640, .h = 480 },
        .vsync = true,
        .title = "tp-gui",
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
    fonts.install();
    _ = try win.end(.{});

    g_wakeup_event_type = SDLBackend.c.SDL_RegisterEvents(1);

    // The LLM is NOT loaded at startup — it lazy-loads on the first chat message
    // (see submitChat). Build the app-level diffusion engine now if a model is
    // configured (its pipeline still loads lazily on the first image).
    image_view.setEnv(g_gpa, g_io, wakeupFrame);
    syncDiffuser();

    // Tear down at exit. Stop the diffusion engine FIRST (join its worker) so no
    // diffusion thread is still touching a transcript/gallery image as those are
    // freed; then the LLM, then the gallery.
    defer {
        if (g_loader) |t| t.join();
        freeDiffuser();
        if (g_session) |s| {
            s.be.bindThread();
            s.deinit();
        }
        if (g_session_arena) |a| {
            a.deinit();
            g_gpa.destroy(a);
        }
        freeCarry();
        freeLlmSuspend();
        if (g_pending_submit) |p| g_gpa.free(p);
        clearStaged();
        g_staged_images.deinit(g_gpa);
        image_view.deinit();
        status_bar.deinit();
    }
    defer if (g_viewer) |v| v.deinit();

    var interrupted = false;
    main_loop: while (true) {
        maybeProcessEjects();
        maybeStartReload();
        // Pump the app-level diffusion engine every frame (both modes; even under
        // Settings) so an in-flight generation finishes — it drains its own
        // unified queue. Gated on !loading so no diffusion worker touches the
        // session while the LLM (re)loads (see maybeStartReload).
        if (!g_loading.load(.acquire)) if (g_diffuser) |*d| d.pump();
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
        var vreq: ?*chat.GenImage = null;
        if (g_viewer_request) |gi| {
            g_viewer_request = null;
            vreq = gi;
        } else if (image_view.viewer_request) |gi| {
            image_view.viewer_request = null;
            vreq = gi;
        }
        const vsrc = diffuserSource();
        if (vreq) |gi| {
            if (g_viewer) |v| {
                v.setImage(gi);
                _ = SDLBackend.c.SDL_RaiseWindow(v.back.window);
            } else {
                g_viewer = viewer.Viewer.init(init.gpa, init.io, vsrc, gi) catch |err| vblk: {
                    std.log.err("open viewer failed: {t}", .{err});
                    break :vblk null;
                };
                // Restore the viewer window's saved geometry (created hidden, so
                // this lands before it's first shown — no flash).
                if (g_viewer) |v| applyWindowGeom(v.back.window, g_config.viewer_w, g_config.viewer_h, g_config.viewer_x, g_config.viewer_y, g_config.viewer_max);
            }
        }

        // ── Viewer window ────────────────────────────────────────────────
        // Closed (window's X, or a "new chat" that freed its image): tear down
        // without rendering, since `v.cur` may now be dangling.
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

/// Can the next message carry an image? True when a vision tower is resident,
/// or — before the lazy first-message load — when the configured model has one
/// (an mmproj path is set alongside an LLM). Lets drop/paste and "Discuss this
/// image" work as the first message, staging into `g_staged_images` until the
/// session comes up. While a load is in flight the session is off-limits to the
/// UI thread, so only the config answers.
fn visionAvailable() bool {
    if (!g_loading.load(.acquire)) if (g_session) |s| return s.visionEnabled();
    return g_config.vision_tower.opt() != null and g_config.llm_model.opt() != null;
}

/// Attach a decoded RGB image to the next message. Hands it to the live session,
/// or (lazy first message) stages it and kicks a load. Callers decode the source
/// and confirm `visionAvailable()` first.
fn attachOrStage(rgb: []const u8, w: usize, h: usize) void {
    if (g_session) |s| {
        s.attachImage(rgb, w, h) catch |err| std.log.err("attach image: {t}", .{err});
        return;
    }
    const rgb_own = g_gpa.dupe(u8, rgb) catch return;
    const rgba = diffuser.rgbToRgba(g_gpa, rgb_own, w, h) catch {
        g_gpa.free(rgb_own);
        return;
    };
    g_staged_images.append(g_gpa, .{ .rgb = rgb_own, .rgba = rgba, .width = w, .height = h }) catch {
        g_gpa.free(rgb_own);
        g_gpa.free(rgba);
        return;
    };
    // Kick the lazy load so the staged image (and any first message) lands in a
    // session; maybeStartReload drains g_staged_images once it's live.
    if (!g_loading.load(.acquire)) g_reload_requested = true;
}

/// RGBA variant of `attachOrStage` for images that already live as display
/// pixels (generated images). `s` is the UI-safe session (null while loading).
fn attachOrStageRgba(s: ?*chat.Session, rgba: []const u8, w: usize, h: usize) void {
    if (s) |ss| {
        ss.attachRgba(rgba, w, h) catch |err| std.log.err("attach image: {t}", .{err});
        return;
    }
    const px = w * h;
    const rgb = g_gpa.alloc(u8, px * 3) catch return;
    defer g_gpa.free(rgb);
    for (0..px) |i| {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    attachOrStage(rgb, w, h);
}

/// Drop a not-yet-loaded staged attachment by index (pre-session mirror of
/// `Session.removeAttachment`).
fn removeStaged(idx: usize) void {
    if (idx >= g_staged_images.items.len) return;
    const st = g_staged_images.orderedRemove(idx);
    g_gpa.free(st.rgb);
    g_gpa.free(st.rgba);
}

/// Free all staged attachments (load failed, or "new chat" before load).
fn clearStaged() void {
    for (g_staged_images.items) |st| {
        g_gpa.free(st.rgb);
        g_gpa.free(st.rgba);
    }
    g_staged_images.clearRetainingCapacity();
}

/// Whether the *configured* LLM (by GGUF architecture) can reason, so the
/// thinking toggle can show before the model loads. A live session's loaded
/// family is authoritative (see `renderInput`); this covers the pre-load window
/// and re-probes whenever the configured model path changes.
fn configuredSupportsThinking() bool {
    const path = g_config.llm_model.opt() orelse {
        g_think_probe_valid = false;
        g_think_probe_len = 0;
        return false;
    };
    if (!(g_think_probe_valid and g_think_probe_len == path.len and
        std.mem.eql(u8, g_think_probe_path[0..g_think_probe_len], path)))
    {
        g_think_probe_result = probeThinking(path);
        @memcpy(g_think_probe_path[0..path.len], path);
        g_think_probe_len = path.len;
        g_think_probe_valid = true;
    }
    return g_think_probe_result;
}

/// Read the configured GGUF's architecture and map it to reasoning support.
/// Any failure (missing/unreadable file, unknown arch) → false.
fn probeThinking(path: []const u8) bool {
    var gg = tp.Gguf.open(g_gpa, g_io, path) catch return false;
    defer gg.deinit();
    const arch = gg.getStr("general.architecture") orelse return false;
    const fam = tp.llm.chat.familyForArch(arch) orelse return false;
    return tp.llm.chat.familySupportsThinking(fam);
}

/// A file was dropped on the window: decode it (libvips → RGB) and attach it
/// to the next message for the model to see.
fn handleDropFile(path: []const u8) void {
    if (g_loading.load(.acquire)) return; // session being rebuilt on the loader thread
    if (!visionAvailable()) {
        std.log.warn("dropped {s} but vision is unavailable", .{path});
        return;
    }
    const gpa = std.heap.smp_allocator;
    const dec = vips.loadRgb(gpa, path) catch |err| {
        std.log.err("can't load dropped image {s}: {t}", .{ path, err });
        return;
    };
    defer gpa.free(dec.pixels);
    attachOrStage(dec.pixels, dec.width, dec.height);
}

/// Ctrl/Cmd+V with an image on the clipboard: decode the raw bytes (any
/// libvips format) and attach it, exactly as a dropped file. Returns true
/// when an image was found on the clipboard (whether or not decoding
/// succeeded), so the caller can consume the event before the text entry
/// treats it as a text paste. Returns false when the clipboard holds no
/// image, letting normal text paste proceed.
fn tryPasteClipboardImage() bool {
    const SDL = SDLBackend.c;
    if (g_loading.load(.acquire)) return false; // session being rebuilt on the loader thread
    if (!visionAvailable()) return false;

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
    attachOrStage(dec.pixels, dec.width, dec.height);
    return true;
}

fn frame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .background = true });
    defer root.deinit();

    if (g_view == .config) {
        config_view.render(&g_config, .{ .apply = applyConfig, .cancel = cancelConfig });
        return;
    }

    if (g_view == .image) {
        // Generation is gated only while an LLM (re)load is in flight (the pump
        // is paused then). The studio doesn't need the LLM — both stay resident.
        const ready = !g_loading.load(.acquire);
        const d: ?*diffuser.Diffuser = if (g_diffuser) |*dd| dd else null;
        // Cap the studio to leave the status bar its row at the bottom (same
        // VRAM/LLM/diffusion readout as chat — we want it in both views).
        const h = @max(120, root.data().contentRect().h - status_bar.bar_height);
        {
            var area = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .min_size_content = .{ .h = h }, .max_size_content = .height(h) });
            defer area.deinit();
            image_view.render(&g_config, d, ready, .{ .to_chat = enterChatMode, .settings = openSettings });
        }
        // `ready` == !g_loading: only touch the session when no reload is tearing
        // it down on the loader thread (same invariant as the chat view).
        status_bar.render(if (ready) g_session else null, diffBusy(), diffVram(), &g_split, &g_limit, g_llm_eject_armed, g_diff_eject_armed, llmPaused(), diffPaused(), meterActions());
        return;
    }

    // Chat view. A background (re)load never takes over the screen: the layout
    // stays put, the just-sent message shows as a normal user bubble, and the
    // assistant slot shows a small "Loading…" until the session is live (see
    // renderMessages). No spinner flashing in and out.
    const loading = g_loading.load(.acquire);
    // While a (re)load is in flight the loader thread is tearing down / rebuilding
    // g_session on its own thread, so the UI MUST NOT dereference the session
    // pointer (the release/acquire hand-off only guarantees it valid when
    // g_loading is false — see the g_session doc comment). Treat it as unavailable
    // and fall back to the carried transcript; otherwise a backend switch that
    // triggers a reload use-after-frees the session mid-frame. (This is why every
    // session-consuming render below takes `s_ui`, not `g_session`.)
    const s_ui: ?*chat.Session = if (loading) null else g_session;

    // Only show the "no model" notice when there's genuinely nothing configured
    // (not during a load, when g_session is transiently null).
    if (!loading and g_session == null and g_config.llm_model.opt() == null) {
        renderNoModel();
        return;
    }

    if (s_ui) |s| {
        s.poll();
        if (g_diffuser) |*d| s.scanNewImages(d);
    }

    // Pin the input strip to the bottom: cap the message list's height to the
    // space left after the (dynamically-measured) input row. A scrollArea
    // reports its full content height as its min size, so as a plain flex child
    // it would push the input off-screen (dvui's box sums every child's min
    // height). max_size_content caps that.
    const list_h = @max(120, root.data().contentRect().h - g_input_h - status_bar.bar_height);

    renderMessages(s_ui, list_h, loading);
    renderInput(s_ui);
    status_bar.render(s_ui, diffBusy(), diffVram(), &g_split, &g_limit, g_llm_eject_armed, g_diff_eject_armed, llmPaused(), diffPaused(), meterActions());
}

fn diffBusy() bool {
    return if (g_diffuser) |*d| d.busyNow() else false;
}
fn diffVram() diffuser.VramBreakdown {
    return if (g_diffuser) |*d| d.vramBreakdown() else .{};
}

/// Chat view while the session (re)loads on the background thread.
/// A small "Loading…" bubble shown in the assistant response slot while the
/// model (re)loads — same left-leaning neutral style as a real assistant turn,
/// with a spinner. Replaced by the real streaming response the instant the
/// session is live.
fn loadingAssistantBubble() void {
    const theme = dvui.themeGet();
    var bubble = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .margin = .{ .x = 8, .y = 3, .w = 96, .h = 3 },
        .background = true,
        .color_fill = theme.fill.lerp(theme.text, 0.08),
        .corner_radius = dvui.Rect.all(10),
        .padding = dvui.Rect.all(10),
    });
    defer bubble.deinit();
    dvui.spinner(@src(), .{ .gravity_y = 0.5, .min_size_content = .{ .w = 14, .h = 14 }, .margin = .{ .w = 8 } });
    dvui.label(@src(), "Loading…", .{}, .{ .gravity_y = 0.5, .color_text = theme.text.lerp(theme.fill, 0.35) });
}

/// Chat view when no LLM is configured (or the last load failed): explain and
/// offer a shortcut into settings.
fn renderNoModel() void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5, .padding = dvui.Rect.all(24) });
    defer col.deinit();

    {
        var tl = dvui.textLayout(@src(), .{}, .{ .gravity_x = 0.5 });
        defer tl.deinit();
        if (g_load_err) |err| {
            var msg: [256]u8 = undefined;
            fonts.addRich(tl, std.fmt.bufPrint(&msg, "Failed to load the model: {t}\n\nChoose a different model in Settings.", .{err}) catch "Failed to load the model.");
        } else {
            fonts.addRich(tl, "No LLM model is set.\n\nOpen Settings (or the ⚙ button) to choose a model file to get started.");
        }
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 12 } });
        defer row.deinit();
        if (dvui.button(@src(), "Open Settings", .{}, .{})) openSettings();
        if (dvui.button(@src(), "Image studio", .{}, .{ .margin = .{ .x = 8 } })) enterImageMode();
    }
}

fn openSettings() void {
    g_return_view = if (g_view == .config) g_return_view else g_view;
    config_view.open();
    g_view = .config;
}

/// The viewer navigates the engine's unified image list (chat + studio).
fn diffuserSource() viewer.ImageSource {
    return .{ .ctx = undefined, .gpa = g_gpa, .collect = diffuserCollect };
}
fn diffuserCollect(_: *anyopaque, buf: *std.ArrayList(*chat.GenImage)) void {
    buf.clearRetainingCapacity();
    if (g_diffuser) |*d| for (d.items()) |gi| {
        if (gi.get() == .done) buf.append(g_gpa, gi) catch {};
    };
}

// ── App-level diffusion engine VRAM coordinator ──────────────────────────────
// The engine calls these through a type-erased ctx pointer; the ctx is unused
// (we dispatch off app globals). Coordination goes to the resident LLM if there
// is one, else no-ops (diffusion has the device to itself). The engine owns its
// queue directly now, so there is no source hook.
fn vcEnter(_: *anyopaque) void {
    // Image queue started → the arbiter drives the LLM down to its share. Under
    // the lock: an idle LLM is settled directly on this (UI) thread, which must
    // not race the diffusion worker's reclaim hook (both touch the LLM context).
    // A busy LLM is only published to its control point (an atomic) and yields at
    // its next token — the fix for the old "no-op while generating" bug.
    g_session_mu.lockUncancelable(g_io);
    defer g_session_mu.unlock(g_io);
    if (g_diffuser) |*d| g_arbiter.diff_used = d.vramBytes();
    g_arbiter.setDiffusionActive(true);
}
fn vcExit(_: *anyopaque) void {
    {
        g_session_mu.lockUncancelable(g_io);
        defer g_session_mu.unlock(g_io);
        if (g_diffuser) |*d| g_arbiter.diff_used = d.vramBytes();
        g_arbiter.setDiffusionActive(false); // LLM may reclaim up to limit − diff_used
    }
    // Queue drained → idle. Re-honor the limit: trim the now-idle diffusion
    // model's resident weights if they push total usage past the ceiling.
    applyMeterPolicy();
}
fn vcBudget(_: *anyopaque) u64 {
    // Called on the diffusion WORKER thread — serialize with a concurrent LLM
    // eject (unloadLlm) that may be freeing the session right now. Pure read.
    g_session_mu.lockUncancelable(g_io);
    defer g_session_mu.unlock(g_io);
    return g_arbiter.diffusionBudget();
}
fn vcReclaim(_: *anyopaque, needed: u64) u64 {
    // Worker thread; the reclaim hook binds the LLM context, so it must not race
    // an eject freeing that context. (LLM idle is checked inside imageReclaim.)
    g_session_mu.lockUncancelable(g_io);
    defer g_session_mu.unlock(g_io);
    return if (g_session) |s| s.imageReclaim(needed) else 0;
}
fn appCoordinator() diffuser.VramCoordinator {
    return .{ .ctx = undefined, .enter = vcEnter, .exit = vcExit, .budget = vcBudget, .reclaim = vcReclaim };
}

/// Whether a diffusion model is fully configured (all three pieces).
fn hasDiffModel(cfg: *const config.Config) bool {
    return cfg.diffusion_model.opt() != null and cfg.vae.opt() != null and cfg.text_encoder.opt() != null;
}

fn dcfgFromConfig() diffuser.DiffConfig {
    return .{
        .dit_path = g_config.diffusion_model.opt().?,
        .vae_path = g_config.vae.opt().?,
        .text_encoder_path = g_config.text_encoder.opt().?,
        .steps = g_config.steps,
        .width = g_config.width,
        .height = g_config.height,
        .backend = diffuser.toPipelineBackend(g_config.diff_backend),
        .vae_decode = diffuser.toPipelineVae(g_config.vae_decode),
        .preview_enabled = g_config.preview != .none,
        .taew_path = if (g_config.preview == .taesd) g_config.taesd.opt() else null,
        .preview_ds = g_config.taesd_size.divisor(),
        .output_dir = g_config.output_dir.opt(),
    };
}

/// Reconcile the app-level engine with the current config: build it when a
/// diffusion model is (newly) configured, free it when cleared, and push
/// path/default/preview changes into a live one (a model swap defers until the
/// queue is idle). Called at startup and on every settings Apply.
fn syncDiffuser() void {
    if (!hasDiffModel(&g_config)) {
        freeDiffuser();
        return;
    }
    if (g_diffuser == null) {
        g_diffuser = diffuser.Diffuser.init(g_gpa, g_io, wakeupFrame, dcfgFromConfig(), appCoordinator());
        g_diffuser.?.seedBase(@truncate(@as(u96, @bitCast(std.Io.Clock.real.now(g_io).nanoseconds))));
    }
    var d = &g_diffuser.?;
    // requestPaths re-dupes the paths into the engine's owned store (nothing
    // aliases the live config buffers) and applies/swaps once the queue is idle.
    d.requestPaths(
        g_config.diffusion_model.opt().?,
        g_config.vae.opt().?,
        g_config.text_encoder.opt().?,
        if (g_config.preview == .taesd) g_config.taesd.opt() else null,
        diffuser.toPipelineBackend(g_config.diff_backend),
        diffuser.toPipelineVae(g_config.vae_decode),
    );
    d.setDefaults(g_config.steps, g_config.width, g_config.height);
    d.setPreview(g_config.preview);
    d.setPreviewSize(g_config.taesd_size.divisor());
    d.setOutputDir(g_config.output_dir.opt());
}

/// Tear down the app-level engine (diffusion model cleared, or at exit): cancel
/// any in-flight/queued generation so the worker aborts instead of blocking the
/// join, then free it (frees the whole image history it owns).
fn freeDiffuser() void {
    if (g_diffuser) |*d| {
        d.cancelAll();
        // Drop every BORROWED reference to the images d.deinit is about to free:
        // the chat transcript (live + carried) and the open viewer. Otherwise a
        // model-clear would leave dangling pointers behind.
        if (g_viewer) |v| v.open = false;
        if (g_session) |s| s.clearImageRefs();
        if (g_carry) |*c| for (c.items) |*m|
            for (m.variants.items) |*v| v.images.clearRetainingCapacity();
        d.deinit();
        g_diffuser = null;
    }
}

/// Mode switches are PURE VIEW changes — they never free, unload, or reload a
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
/// forces a transcript-preserving reload — but only if the LLM is currently
/// resident; if it hasn't lazy-loaded yet, the new config is simply picked up on
/// the first message.
fn applyConfig() void {
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});

    // The diffusion engine is shared by both modes; reconcile it either way.
    syncDiffuser();

    const llm_reload = !g_config.llmReloadEql(&g_config_baseline) or
        (g_config.diffEnabled() != g_config_baseline.diffEnabled()); // tool prompt changes
    // A KV-dtype change needs only a CONTEXT rebuild (weights stay resident),
    // never the full weight reload above.
    const ctx_reload = !g_config.ctxReloadEql(&g_config_baseline);
    if (g_session != null) {
        if (llm_reload) {
            g_reload_requested = true; // transcript-preserving (see loaderMain)
        } else if (!g_loading.load(.acquire)) {
            g_session.?.updateSettings(&g_config); // reasoning / VRAM priority, live
            if (ctx_reload) g_session.?.rebuildContext(toKvDtype(g_config.kv_dtype)) catch |err|
                std.log.err("kv-dtype context rebuild failed: {t}", .{err});
        }
    }
    // If no session is loaded yet (lazy), the new config is used at first chat.

    g_config_baseline = g_config;
    g_view = g_return_view;
}

/// Toolbar reasoning toggle: flip whether the model reasons before answering,
/// persist it, and push it into the running session live (no reload — it only
/// shapes the next prompt built). Keeps the baseline in sync so a later
/// Settings → Cancel doesn't resurrect the old value.
fn toggleReasoning() void {
    g_config.reasoning = !g_config.reasoning;
    g_config_baseline.reasoning = g_config.reasoning;
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    if (g_session) |s| if (!g_loading.load(.acquire)) s.updateSettings(&g_config);
}

/// Cancel button: discard unsaved edits by reloading the on-disk settings, and
/// return to wherever Settings was opened from (chat or the studio).
fn cancelConfig() void {
    g_config = config.Config.load(g_io, g_gpa, g_environ, g_config_path);
    g_config_baseline = g_config;
    g_view = g_return_view;
}

/// Main-loop hook: reap a finished loader, and start a pending (re)load when
/// none is in flight. Runs on the UI thread so the loading-flag hand-off to the
/// loader is well-ordered.
fn maybeStartReload() void {
    if (g_loader) |t| {
        if (!g_loading.load(.acquire)) {
            t.join();
            g_loader = null;
            // Load finished. Apply the current meter policy to the fresh session
            // (settles the LLM to its share if diffusion is already resident).
            if (g_session != null) applyMeterPolicy();
            // Move any images staged before the lazy load into the fresh session
            // (BEFORE the deferred submit, so the first message carries them). If
            // the load failed (no session) or the model has no vision tower, drop
            // them — attachImage no-ops without a tower.
            if (g_session) |s| {
                for (g_staged_images.items) |st|
                    s.attachImage(st.rgb, st.width, st.height) catch |err| std.log.err("attach staged image: {t}", .{err});
            }
            clearStaged();
            // If the first message was stashed while the LLM lazy-loaded, submit
            // it now that the session is live.
            if (g_pending_submit) |text| {
                g_pending_submit = null;
                if (g_session) |s| s.submit(text) catch |err| std.log.err("deferred submit: {t}", .{err});
                g_gpa.free(text);
            }
            // Resume-continue a mid-turn response suspended by unload-while-paused
            // (Tier 3): `ids` was restored verbatim; continue decoding it.
            if (g_pending_continue) {
                g_pending_continue = false;
                if (g_session) |s| s.continueOpenTurn() catch |err| std.log.err("resume continue: {t}", .{err});
            }
        }
    }
    if (!g_reload_requested or g_loading.load(.acquire) or g_loader != null) return;
    // The LLM (re)loads CONCURRENTLY with diffusion — a running image keeps
    // generating on its own context while the LLM builds on a fresh one, so a
    // chat sent mid-image loads and responds right away instead of waiting for
    // the image. The only shared state is the session pointer, which the loader's
    // teardown/publish and the worker-thread coordinator hooks serialize with
    // `g_session_mu`. The pump stays gated while g_loading is set (below), so no
    // NEW image starts mid-load; the in-flight one is left running (not reaped).
    g_reload_requested = false;
    g_load_err = null;
    g_loading.store(true, .release);
    g_loader = std.Thread.spawn(.{}, loaderMain, .{}) catch |err| {
        std.log.err("spawn loader failed: {t}", .{err});
        g_load_err = err;
        g_loading.store(false, .release);
        return;
    };
}

/// Background (re)load: tear down the old session, then build a new one from
/// `g_config`. Runs on its own thread — creates the CUDA context there; the
/// generation/diffusion workers bind to it as before. On completion it publishes
/// `g_session` and clears `g_loading` (release) so the UI thread can adopt it.
fn loaderMain() void {
    // Tear down the previous session first. Its CUDA context must be current on
    // this thread to free device memory, so bind it before deinit. Before
    // freeing it, stop the LLM turn but let any in-flight image FINISH (don't
    // cancel it), then detach the transcript so the chat survives the swap.
    if (g_session) |s| {
        s.be.bindThread();
        s.requestCancel();
        if (s.worker) |t| {
            t.join();
            s.worker = null;
        }
        // Diffusion may be generating concurrently (its worker reads the session
        // via the coordinator hooks), so serialize the teardown with
        // `g_session_mu` — the same guard unloadLlm uses. The transcript is
        // detached (carried) so the chat survives the swap.
        g_session_mu.lockUncancelable(g_io);
        g_carry = s.detachTranscript();
        s.deinit();
        g_session = null;
        g_arbiter.llm = null; // participant points into the freed session
        g_session_mu.unlock(g_io);
    }
    if (g_session_arena) |a| {
        a.deinit();
        g_gpa.destroy(a);
        g_session_arena = null;
    }

    if (g_config.llm_model.opt() == null) {
        // Nothing to load (LLM cleared): drop the carried transcript and leave
        // the notice showing.
        freeCarry();
        freeLlmSuspend();
        g_loading.store(false, .release);
        wakeupFrame();
        return;
    }

    const arena_obj = g_gpa.create(std.heap.ArenaAllocator) catch |err| return finishLoad(err);
    arena_obj.* = std.heap.ArenaAllocator.init(g_gpa);
    g_session_arena = arena_obj;

    const t0 = std.Io.Clock.real.now(g_io).nanoseconds;
    const s = buildSession(arena_obj.allocator()) catch |err| {
        std.log.err("failed to load session: {t}", .{err});
        arena_obj.deinit();
        g_gpa.destroy(arena_obj);
        g_session_arena = null;
        freeCarry(); // no session to adopt the transcript
        freeLlmSuspend();
        return finishLoad(err);
    };
    // Replay the carried transcript into the new model (KV empty; the next turn's
    // prefill replays it) so a model swap keeps the chat.
    if (g_carry) |m| {
        s.be.bindThread();
        if (g_llm_suspend) |sus| {
            // Unload-while-paused resume (Tier 3): restore the exact `ids` (open
            // turn) verbatim instead of replaying (which would close the turn),
            // then continue decoding that response after publish.
            s.adoptSuspended(m, sus.ids) catch |err| std.log.err("adopt suspended failed: {t}", .{err});
            if (sus.midturn) g_pending_continue = true;
            g_gpa.free(sus.ids);
            g_llm_suspend = null;
        } else {
            s.adoptTranscript(m) catch |err| std.log.err("adopt transcript failed: {t}", .{err});
        }
        g_carry = null;
    }
    // A paused reload that is NOT a resume (e.g. a backend switch while paused)
    // starts the fresh gate paused so the state matches the button.
    if (g_llm_paused) s.pause.pause(g_io);
    const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(g_io).nanoseconds - t0)) / 1e9;
    std.log.info("[vram] LLM session loaded/ready in {d:.1}s", .{dt});
    // Publish under the lock: the diffusion worker's coordinator hooks read
    // `g_session` and must see either null or a fully-built session, never a tear.
    g_session_mu.lockUncancelable(g_io);
    g_session = s;
    g_arbiter.llm = s.participant();
    g_session_mu.unlock(g_io);
    finishLoad(null);
}

/// Free a carried transcript that has no destination (LLM cleared or load
/// failed). Messages are gpa-owned.
fn freeCarry() void {
    if (g_carry) |*m| {
        for (m.items) |*msg| msg.deinit(g_gpa);
        m.deinit(g_gpa);
        g_carry = null;
    }
}

/// Publish the load result: record any error, then clear the loading flag
/// (release) as the last write so the UI thread's acquire read sees a settled
/// `g_session`.
fn finishLoad(err: ?anyerror) void {
    g_load_err = err;
    g_loading.store(false, .release);
    wakeupFrame();
}

/// Build the LLM session from the current settings. Vision needs the tower; the
/// image tool is available when a diffusion model is configured (the engine
/// itself is app-level). Paths are duped into `arena` so the session never
/// aliases the live config edit buffers.
fn buildSession(arena: std.mem.Allocator) !*chat.Session {
    const llm = g_config.llm_model.opt().?;
    const mmproj: ?[]const u8 = if (g_config.vision_tower.opt()) |m| try arena.dupe(u8, m) else null;
    const system_prompt = g_config.system_prompt.opt() orelse config.default_system_prompt;

    const s = try chat.Session.init(arena, g_gpa, g_io, wakeupFrame, .{
        .model_path = try arena.dupe(u8, llm),
        .system_prompt = try arena.dupe(u8, system_prompt),
        .seed = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(g_io).nanoseconds))),
        .sampling = chat.samplingParams(&g_config),
        .backend = diffuser.toPipelineBackend(g_config.llm_backend),
        .images_enabled = hasDiffModel(&g_config),
        .mmproj_path = mmproj,
        .vram_split = g_config.vram_split,
        .vram_limit_frac = g_config.vram_limit_frac,
        .reasoning = g_config.reasoning,
        .kv_dtype = toKvDtype(g_config.kv_dtype),
        .regen_cache_mb = g_config.regen_cache_mb,
        .vision_budget_tokens = g_config.vision_budget.tokens(),
    });
    return s;
}

/// Map the GUI's local KV-dtype enum onto the library's `kv_cache.KvDtype`
/// (config.zig stays free of a TensorPencil import). Same field names, explicit
/// so adding a variant is a compile error until both sides agree.
fn toKvDtype(d: config.KvDtype) tp.llm.kv_cache.KvDtype {
    return switch (d) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
    };
}

fn renderMessages(s: ?*chat.Session, list_h: f32, loading: bool) void {
    {
        var scroll = dvui.scrollArea(@src(), .{ .scroll_info = &g_scroll_info }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = list_h },
            .max_size_content = .height(list_h),
        });
        defer scroll.deinit();

        var list = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
        defer list.deinit();

        // With no live session, fall back to the CARRIED transcript (present when
        // the LLM was ejected / is between loads) so the conversation stays on
        // screen read-only and is never visually "reset" — only a "new chat"
        // click clears it. It reloads + replays on the next message.
        const msgs: []chat.Message = if (s) |ss| ss.messages.items else if (g_carry) |c| c.items else &.{};
        if (msgs.len == 0 and !loading) {
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = dvui.Rect.all(16) });
            defer tl.deinit();
            tl.addText(if (s == null)
                "Say something to start — the model loads on your first message."
            else
                "Say something to start the conversation.", .{});
        } else {
            for (msgs, 0..) |*m, idx| renderMessage(s, m, idx);
        }
        // While the model (re)loads in the background: the just-sent message
        // (not yet in the transcript — it submits once the session is live) shows
        // as a normal user bubble, and the assistant slot shows a small "Loading…"
        // The instant the session is ready these are replaced by the real turn.
        if (loading) {
            if (g_pending_submit) |txt| pendingUserBubble(txt);
            loadingAssistantBubble();
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
    const theme = dvui.themeGet();
    var bubble = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .margin = .{ .x = 160, .y = 3, .w = 8, .h = 3 },
        .background = true,
        .color_fill = theme.fill.lerp(theme.focus, 0.30),
        .corner_radius = dvui.Rect.all(10),
        .padding = dvui.Rect.all(10),
    });
    defer bubble.deinit();
    markdown_view.render(@src(), text, .{});
}

fn renderMessage(s: ?*chat.Session, m: *const chat.Message, idx: usize) void {
    const is_user = m.role == .user;
    const theme = dvui.themeGet();

    // Instant-messenger layout: the user's turns lean right in an accent-tinted
    // bubble, the assistant's lean left in a neutral card. The asymmetric side
    // margin does the lean; side + color convey the sender, so there's no
    // per-message "You"/"Assistant" label. Both fills differ from the list
    // background so the bubbles read as cards. The assistant keeps most of the
    // width for images/code; the user's is inset further since it's just text.
    var bubble = dvui.box(@src(), .{ .dir = .vertical }, .{
        .id_extra = idx,
        .expand = .horizontal,
        .margin = if (is_user)
            .{ .x = 160, .y = 3, .w = 8, .h = 3 }
        else
            .{ .x = 8, .y = 3, .w = 96, .h = 3 },
        .background = true,
        .color_fill = if (is_user)
            theme.fill.lerp(theme.focus, 0.30)
        else
            theme.fill.lerp(theme.text, 0.08),
        .corner_radius = dvui.Rect.all(10),
        .padding = dvui.Rect.all(10),
    });
    defer bubble.deinit();

    // Images the user attached to this message.
    for (m.attachments.items, 0..) |gi, ai| renderGenImage(s, gi, ai);

    // Everything below shows the message's ACTIVE variant (the ‹/› nav on the
    // last assistant response switches it; older takes stay stored).
    const v = m.activeConst();
    const p = parseThink(v.text.items);
    // Only the last assistant message is actively generating; "Thinking…" means
    // the block is still open AND generation is live. A think block left open
    // because generation stopped (e.g. hit max tokens) reads "Thoughts". With no
    // live session (carried transcript), nothing is generating.
    const live = if (s) |ss| ss.busy() and idx + 1 == ss.messages.items.len else false;

    // Reasoning: collapse the thought block behind an expander, default
    // collapsed. The label doubles as a "thinking" indicator while the block is
    // still open. An empty thought (e.g. a model that opened and closed the
    // channel with nothing inside) shows no bubble at all — unless it's still
    // actively streaming, where "Thinking…" is the right cue.
    if (p.think) |think| {
        if (think.len > 0 or (p.thinking and live)) {
            if (dvui.expander(@src(), if (p.thinking and live) "Thinking…" else "Thoughts", .{ .default_expanded = false }, .{})) {
                // Set the reasoning apart from the answer: a dimmer text color on
                // a slightly inset, accent-bordered block (a blockquote look), so
                // it reads as the model's scratch work rather than the reply.
                markdown_view.render(@src(), think, .{ .prose = .{
                    .expand = .horizontal,
                    .background = true,
                    .color_fill = theme.fill.lerp(theme.text, 0.15),
                    .color_text = theme.text.lerp(theme.fill, 0.40),
                    .color_border = theme.focus,
                    .border = .{ .x = 3, .y = 0, .w = 0, .h = 0 },
                    .corner_radius = dvui.Rect.all(4),
                    .margin = .{ .x = 2, .y = 4, .w = 2, .h = 4 },
                    .padding = .{ .x = 9, .y = 6, .w = 9, .h = 6 },
                } });
            }
        }
    }

    if (p.answer.len > 0) {
        renderAnswer(p.answer);
        // Selection copies rendered text; this copies the raw markdown of
        // the whole reply (assistant messages only — a user's own text is
        // already in their hands).
        if (!is_user) {
            var wd: dvui.WidgetData = undefined;
            if (dvui.buttonIcon(@src(), "copy markdown", dvui.entypo.clipboard, .{}, .{}, .{
                .gravity_x = 1.0,
                .min_size_content = .{ .h = 12 },
                .color_text = theme.text.lerp(theme.fill, 0.5),
                .padding = dvui.Rect.all(2),
                .margin = .{ .y = 2 },
                .data_out = &wd,
            })) dvui.clipboardTextSet(p.answer);
            hint.hover(@src(), &wd, "Copy the reply as markdown");
        }
    } else if (m.role == .assistant and p.think == null and v.images.items.len == 0) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
        defer tl.deinit();
        if (live) {
            tl.addText("…", .{});
        } else if (if (s) |ss| ss.gen_err else null) |err| {
            var msg: [128]u8 = undefined;
            fonts.addRich(tl, std.fmt.bufPrint(&msg, "⚠ generation error: {t}", .{err}) catch "⚠ generation error");
        } else if (idx + 1 == (if (s) |ss| ss.messages.items.len else 0) and (if (s) |ss| ss.isPaused() else false)) {
            // A turn queued while the LLM is paused — it runs on resume (Tier 2).
            tl.addText("⏸ queued — resume to generate", .{ .color_text = theme.text.lerp(theme.fill, 0.4) });
        }
    }

    // Generated images requested by this variant, below its text. Offset the
    // id base so it can't collide with attachment image ids in this bubble.
    for (v.images.items, 0..) |gi, gi_idx| renderGenImage(s, gi, 100_000 + gi_idx);

    // ‹ n/m › navigation on the LAST assistant response (TODO #3): ‹ shows the
    // previous take, › the next — or, on the newest take, regenerates a fresh
    // one. Hidden while generating (Stop is the control then) and on a carried
    // read-only transcript (no session to regenerate with). Images belonging
    // to a non-active take keep generating in the engine's unified queue.
    if (!is_user) if (s) |ss| {
        const nmsg = ss.messages.items.len;
        if (idx + 1 == nmsg and nmsg >= 2 and !ss.busy() and
            ss.messages.items[nmsg - 2].role == .user)
            renderVariantNav(ss, m);
    };
}

/// The ‹ n/m › variant-navigation row (see renderMessage). Back is disabled
/// (dimmed, inert) on the first take; next past the newest take regenerates.
fn renderVariantNav(s: *chat.Session, m: *const chat.Message) void {
    const theme = dvui.themeGet();
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
    if (!can_back) opts.color_text = theme.text.lerp(theme.fill, 0.75);
    if (dvui.buttonIcon(@src(), "prev-variant", dvui.entypo.chevron_small_left, .{}, .{}, opts) and can_back) {
        switch (chat.navTarget(m.cur, n, .back)) {
            .select => |i| s.selectVariant(i),
            else => {},
        }
    }
    if (can_back) hint.hover(@src(), &bwd, "Previous response");

    if (n > 1) dvui.label(@src(), "{d}/{d}", .{ m.cur + 1, n }, .{
        .gravity_y = 0.5,
        .color_text = theme.text.lerp(theme.fill, 0.4),
        .padding = .{ .x = 2, .w = 2 },
    });

    const at_newest = m.cur + 1 == n;
    var nwd: dvui.WidgetData = undefined;
    opts = icon_opts;
    opts.data_out = &nwd;
    if (dvui.buttonIcon(@src(), "next-variant", dvui.entypo.chevron_small_right, .{}, .{}, opts)) {
        switch (chat.navTarget(m.cur, n, .next)) {
            .select => |i| s.selectVariant(i),
            .regenerate => s.regenerate() catch |err| std.log.err("regenerate failed: {t}", .{err}),
            .none => {},
        }
    }
    hint.hover(@src(), &nwd, if (at_newest)
        "Generate a new response"
    else
        "Next response");
}

/// Render answer text as markdown with `<image>…</image>` tool-call tags
/// hidden (the images render separately). Uses the same line-anchored matcher
/// as the generation scanner (`chat.nextImageCall`), so a call that fires is
/// exactly a call that's hidden — a casual inline mention of the tag stays
/// visible text. A still-streaming, unterminated call hides everything from
/// it onward. Stripped text is assembled in the frame arena so the markdown
/// parser sees one contiguous document (blocks may span a hidden call).
fn renderAnswer(text: []const u8) void {
    // Common case: no tool call anywhere — render the original slice.
    var display = text;
    switch (toolcall.nextImageCall(text)) {
        .none => {},
        .partial => |p| display = p.text_before,
        .call => |first| {
            var out: std.ArrayList(u8) = .empty;
            const arena = dvui.currentWindow().arena();
            out.appendSlice(arena, first.text_before) catch return;
            var rest = first.after;
            strip: while (true) {
                switch (toolcall.nextImageCall(rest)) {
                    .none => {
                        out.appendSlice(arena, rest) catch return;
                        break :strip;
                    },
                    .partial => |p| {
                        out.appendSlice(arena, p.text_before) catch return;
                        break :strip;
                    },
                    .call => |c| {
                        out.appendSlice(arena, c.text_before) catch return;
                        rest = c.after;
                    },
                }
            }
            display = out.items;
        },
    }
    markdown_view.render(@src(), display, .{});
}

/// Display size for an image: downscale so the longer side is `max`, never
/// upscale. Sizing the widget to the actual (aspect-correct) dimensions avoids
/// the letterbox padding a square cap would add for non-square images.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    const mx: f32 = @floatFromInt(@max(w, h));
    const scale = if (mx > max) max / mx else 1.0;
    return .{ .w = @as(f32, @floatFromInt(w)) * scale, .h = @as(f32, @floatFromInt(h)) * scale };
}

fn renderGenImage(s: ?*chat.Session, gi: *chat.GenImage, gi_idx: usize) void {
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = gi_idx, .expand = .horizontal, .margin = .{ .y = 4, .h = 4 } });
    defer b.deinit();

    switch (gi.get()) {
        .pending, .generating, .suspended => {
            const st_now = gi.get();
            const generating = st_now == .generating;
            const done = gi.step.load(.monotonic);
            const total = gi.total.load(.monotonic);
            // Live preview (re-uploaded each frame as its bytes update).
            if (gi.preview) |pv| {
                const pw = gi.preview_w.load(.acquire);
                const ph = gi.preview_h.load(.acquire);
                if (pw > 0 and ph > 0) {
                    const sz = fitSize(pw, ph, 480);
                    _ = dvui.image(@src(), .{
                        .source = .{ .pixels = .{ .rgba = pv[0 .. pw * ph * 4], .width = pw, .height = ph, .invalidation = .always } },
                        .shrink = .ratio,
                    }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                }
            }
            // Live timing: elapsed since dispatch, average s/step over completed
            // steps (excludes model-load time), and an ETA from that rate. Keep
            // the frame repainting so the elapsed timer ticks between step wakes.
            if (generating) dvui.refresh(null, @src(), null);
            const start = gi.start_ns.load(.acquire);
            const first = gi.first_step_ns.load(.acquire);
            const last = gi.last_step_ns.load(.acquire);
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
            genInfo(gi);
            // Stop this generation (or drop it from the queue): the flag is
            // polled by the pipeline between steps and by `nextPending`.
            if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .{ .y = 4 } })) {
                gi.cancel.store(true, .release);
                gi.wake();
                // If we're paused, the worker is parked at the gate — wake it so
                // it re-checks this image's cancel flag now, not on resume.
                if (g_diffuser) |*d| d.wakePaused();
            }
        },
        .done => {
            if (gi.rgba) |rgba| {
                // Wrap in a box so a click anywhere on the image opens the
                // full-size viewer window. Size to the image's own aspect so
                // there's no letterbox padding before the button.
                const sz = fitSize(gi.width, gi.height, 480);
                var ib = dvui.box(@src(), .{}, .{});
                _ = dvui.image(@src(), .{
                    .source = .{ .pixels = .{ .rgba = rgba, .width = @intCast(gi.width), .height = @intCast(gi.height) } },
                    .shrink = .ratio,
                }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                const clicked = dvui.clicked(ib.data(), .{});
                ib.deinit();
                if (clicked) g_viewer_request = gi;

                genInfo(gi);

                {
                    var actions = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .y = 4 } });
                    defer actions.deinit();
                    // Copy the image to the clipboard as a PNG.
                    var cwd: dvui.WidgetData = undefined;
                    if (dvui.buttonIcon(@src(), "copy", dvui.entypo.clipboard, .{}, .{}, .{
                        .min_size_content = .{ .w = 18, .h = 18 },
                        .gravity_y = 0.5,
                        .data_out = &cwd,
                    })) clipboard.copyImage(gi);
                    hint.hover(@src(), &cwd, "Copy image to clipboard");

                    // Let the model see this image: attach it to the next
                    // message. Shown whenever the configured model can see
                    // images (not just while a session is resident); with the
                    // LLM unloaded the image is staged and the lazy load kicks.
                    if (visionAvailable()) {
                        if (dvui.button(@src(), "Discuss this image", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 6 } })) {
                            attachOrStageRgba(s, rgba, gi.width, gi.height);
                        }
                    }
                }
            }
        },
        .failed => {
            fonts.richLabel(@src(), "⚠ image generation failed", .{});
            genInfo(gi);
        },
        .canceled => {
            fonts.richLabel(@src(), "⚠ image generation canceled", .{});
            genInfo(gi);
        },
    }
}

/// A compact metadata line (resolution · seed) plus a collapsed-by-default
/// prompt expander, shown under an image in every state. Uses the actual output
/// dimensions once known, else the requested ones; the seed is shown once
/// assigned (non-zero).
fn genInfo(gi: *chat.GenImage) void {
    const w = if (gi.width > 0) gi.width else gi.req_width;
    const h = if (gi.height > 0) gi.height else gi.req_height;
    var buf: [160]u8 = undefined;
    // Seed is assigned at scan time (never 0 here), so always show it. Once
    // done, append the average s/step (sampling only) and total wall time.
    const meta = if (gi.get() == .done) blk: {
        const start = gi.start_ns.load(.acquire);
        const dn = gi.done_ns.load(.acquire);
        const first = gi.first_step_ns.load(.acquire);
        const last = gi.last_step_ns.load(.acquire);
        const total_s: f64 = if (start > 0 and dn > start) @as(f64, @floatFromInt(dn - start)) / 1e9 else 0;
        const sps: f64 = if (gi.req_steps >= 2 and first > 0 and last > first)
            (@as(f64, @floatFromInt(last - first)) / 1e9) / @as(f64, @floatFromInt(gi.req_steps - 1))
        else
            0;
        break :blk std.fmt.bufPrint(&buf, "{d}×{d}  ·  {d} steps  ·  seed {d}  ·  {d:.2} s/step  ·  {d:.1}s total", .{ w, h, gi.req_steps, gi.req_seed, sps, total_s }) catch "";
    } else std.fmt.bufPrint(&buf, "{d}×{d}  ·  {d} steps  ·  seed {d}", .{ w, h, gi.req_steps, gi.req_seed }) catch "";
    // Rendered through a text layout (not a plain label) so the metadata —
    // seed especially — is mouse-selectable.
    fonts.richLabel(@src(), meta, .{ .margin = .{ .y = 2 } });
    if (dvui.expander(@src(), "Prompt", .{ .default_expanded = false }, .{})) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = .{ .x = 6, .y = 2, .w = 6, .h = 4 } });
        defer tl.deinit();
        fonts.addRich(tl, gi.prompt);
    }
}

const Parsed = struct { think: ?[]const u8, answer: []const u8, thinking: bool };

/// Split an assistant message into its reasoning block and the answer, using
/// the active model family's thought markers (chat.reasoning() — e.g. Qwen's
/// `<think>…</think>`, Gemma 4's `<|channel>thought…<channel|>`). The markers
/// themselves are dropped. `thinking` is true while the block is still open (no
/// close marker yet). Families that don't reason return everything as answer.
fn parseThink(text: []const u8) Parsed {
    const ws = " \n\r\t";
    const r = tp.llm.chat.reasoning() orelse
        return .{ .think = null, .answer = text, .thinking = false };
    const t = std.mem.trimStart(u8, text, ws);
    if (std.mem.startsWith(u8, t, r.open)) {
        const rest = t[r.open.len..];
        if (std.mem.indexOf(u8, rest, r.close)) |end| {
            return .{
                .think = std.mem.trim(u8, rest[0..end], ws),
                .answer = std.mem.trimStart(u8, rest[end + r.close.len ..], ws),
                .thinking = false,
            };
        }
        return .{ .think = std.mem.trimStart(u8, rest, ws), .answer = "", .thinking = true };
    }
    return .{ .think = null, .answer = text, .thinking = false };
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

fn renderInput(s: ?*chat.Session) void {
    var container = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
    defer container.deinit();

    // Thumbnails of images attached but not yet sent, each with a hover-only X
    // to remove it before sending. Once the LLM is live these come from the
    // session's pending attachments; before the lazy first-message load they
    // come from the pre-session staging buffer (dropped/pasted first images).
    const n_thumbs = if (s) |ss| ss.pendingAttachments().len else g_staged_images.items.len;
    if (n_thumbs > 0) {
        var remove_idx: ?usize = null;
        {
            var strip = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 8, .y = 4, .w = 8 } });
            defer strip.deinit();
            if (s) |ss| {
                for (ss.pendingAttachments(), 0..) |gi, pi| {
                    if (gi.rgba) |rgba| if (renderPendingThumb(pi, rgba, gi.width, gi.height)) {
                        remove_idx = pi;
                    };
                }
            } else {
                for (g_staged_images.items, 0..) |st, pi| {
                    if (renderPendingThumb(pi, st.rgba, st.width, st.height)) remove_idx = pi;
                }
            }
        }
        if (remove_idx) |ri| {
            if (s) |ss| ss.removeAttachment(ri) else removeStaged(ri);
        }
    }

    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = dvui.Rect.all(8) });
    defer row.deinit();

    const busy = if (s) |ss| ss.busy() else false;

    // New chat: drop the conversation and start fresh (model stays resident).
    // Disabled mid-turn so it can't race the generation worker.
    var wd: dvui.WidgetData = undefined;
    if (dvui.buttonIcon(@src(), "new-chat", dvui.entypo.plus, .{}, .{}, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 22, .h = 22 },
        .margin = .{ .w = 6 },
        .data_out = &wd,
    }) and !busy) newChat();
    hint.hover(@src(), &wd, "New chat — clears the conversation (model stays loaded)");

    // Gear: switch to the settings view.
    if (dvui.buttonIcon(@src(), "settings", dvui.entypo.cog, .{}, .{}, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 22, .h = 22 },
        .margin = .{ .w = 6 },
        .data_out = &wd,
    })) openSettings();
    hint.hover(@src(), &wd, "Settings");

    // Image studio: leave chat (unloads the LLM) for direct text-to-image.
    // Disabled mid-turn so the unload can't race the generation worker.
    if (dvui.buttonIcon(@src(), "image-studio", dvui.entypo.image, .{}, .{}, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 22, .h = 22 },
        .margin = .{ .w = 6 },
        .data_out = &wd,
    }) and !busy) enterImageMode();
    hint.hover(@src(), &wd, "Image studio — text-to-image without chat (unloads the LLM)");

    // Reasoning toggle (no brain icon in entypo — a lit bulb reads as
    // "thinking"). Shown for any model whose family can reason (Gemma 3 etc.
    // hide it). A live session's loaded family is authoritative; before the
    // lazy first-message load we probe the configured model's GGUF so the toggle
    // is usable up front. Highlighted when on; flips live (or into config) and
    // persists.
    const can_reason = if (s != null) tp.llm.chat.supportsThinking() else configuredSupportsThinking();
    if (can_reason) {
        const on = g_config.reasoning;
        const th = dvui.themeGet();
        if (dvui.buttonIcon(@src(), "reasoning", dvui.entypo.light_bulb, .{}, .{}, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = 22, .h = 22 },
            .margin = .{ .w = 6 },
            .background = on,
            .corner_radius = dvui.Rect.all(5),
            .color_fill = if (on) th.fill.lerp(th.focus, 0.35) else null,
            .color_text = if (on) th.focus else null,
            .data_out = &wd,
        })) toggleReasoning();
        hint.hover(@src(), &wd, if (on)
            "Thinking is on — click to disable reasoning"
        else
            "Thinking is off — click to enable reasoning");
    }

    var send = false;

    // Enter (without Shift) on the focused input sends. Consume the key before
    // the multiline entry turns it into a newline; Shift+Enter falls through as
    // a newline. Disabled while generating so the box stays editable.
    if (!busy) {
        // Paste (Ctrl/Cmd+V) an image from the clipboard: intercept before the
        // text entry so an image on the clipboard attaches instead of a bogus
        // text paste. Text-only clipboards fall through untouched. Handled
        // regardless of focus so it works right after clicking into chat.
        for (dvui.events()) |*e| {
            if (e.handled or e.evt != .key) continue;
            const k = e.evt.key;
            if (k.code == .v and k.action == .down and (k.mod.control() or k.mod.command())) {
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

    var buf: [4096]u8 = undefined;
    var n: usize = 0;

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &g_input_buf },
        .multiline = true,
        .placeholder = "Message…  (Enter to send · Shift+Enter for newline)",
        .scroll_horizontal = false,
        .break_lines = true,
    }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        .min_size_content = .{ .h = 28 },
        .max_size_content = .height(140),
    });
    g_input_id = te.data().id;
    // Reserve for next frame's layout: entry height + row padding, plus the
    // attachment strip when present.
    g_input_h = te.data().rect.h + 24 + (if (n_thumbs > 0) @as(f32, 72) else 0);
    if (send) {
        const t = te.getText();
        n = @min(t.len, buf.len);
        @memcpy(buf[0..n], t[0..n]);
    }
    te.deinit();

    if (dvui.button(@src(), if (busy) "Stop" else "Send", .{}, .{ .gravity_y = 0.5 })) {
        if (busy) {
            if (s) |ss| ss.requestCancel();
        } else if (!send) {
            const t = std.mem.sliceTo(&g_input_buf, 0);
            n = @min(t.len, buf.len);
            @memcpy(buf[0..n], t[0..n]);
            send = true;
        }
    }

    if (send and !busy) {
        // `/new` on its own line starts a fresh chat instead of sending.
        if (std.mem.eql(u8, std.mem.trim(u8, buf[0..n], " \t\r\n"), "/new")) {
            newChat();
        } else {
            submitChat(buf[0..n]);
        }
        @memset(&g_input_buf, 0);
    }
}

/// Send a chat message. If the LLM is resident, submit directly. Otherwise
/// (lazy first chat) stash the text and kick a load; `maybeStartReload`
/// auto-submits it once the model is live.
fn submitChat(text: []const u8) void {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return;
    // Never touch the session while a (re)load is in flight — the loader thread is
    // freeing/rebuilding it. Submit only when it's live; otherwise stash the text
    // and the load-completion path (maybeStartReload) auto-submits it.
    if (!g_loading.load(.acquire)) if (g_session) |s| {
        s.submit(trimmed) catch |err| std.log.err("submit failed: {t}", .{err});
        return;
    };
    if (g_config.llm_model.opt() == null) return; // no model to load
    if (g_pending_submit) |p| g_gpa.free(p);
    g_pending_submit = g_gpa.dupe(u8, trimmed) catch null;
    if (!g_loading.load(.acquire)) g_reload_requested = true; // else the in-flight load will pick it up
}

/// Start a fresh conversation, clearing the input box. Only the transcript is
/// reset; generated images live in the engine's shared history and stay in the
/// studio gallery (and the viewer keeps working). No-op if no session is loaded.
fn newChat() void {
    if (!g_loading.load(.acquire)) if (g_session) |s| s.reset(); // session being rebuilt → g_carry clear below suffices
    // If the LLM is ejected, the transcript lives in g_carry — clear it too so
    // "new chat" starts fresh even while unloaded. (No-op when a session is
    // live, since g_carry is null then.)
    freeCarry();
    freeLlmSuspend(); // drop any suspended-turn carry; a new chat won't resume it
    if (g_llm_paused) { // a fresh chat starts unpaused
        g_llm_paused = false;
        if (!g_loading.load(.acquire)) if (g_session) |s| s.setPaused(false);
    }
    clearStaged(); // drop any images staged for a not-yet-loaded first message
    @memset(&g_input_buf, 0);
    g_follow_bottom = true;
}
