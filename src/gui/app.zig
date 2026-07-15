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
const chat = @import("chat.zig");
const fonts = @import("fonts.zig");
const viewer = @import("viewer.zig");
const config = @import("config.zig");
const config_view = @import("config_view.zig");
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
var g_session_arena: ?*std.heap.ArenaAllocator = null; // load-once weights, freed on reload
var g_loading: std.atomic.Value(bool) = .init(false);
var g_loader: ?std.Thread = null;
var g_reload_requested: bool = false;
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
const View = enum { chat, config };

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

    var back = try SDLBackend.initWindow(.{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = 1100, .h = 820 },
        .min_size = .{ .w = 640, .h = 480 },
        .vsync = true,
        .title = "tp-gui",
        .environ_map = init.environ_map,
    });
    defer back.deinit();

    var win = try dvui.Window.init(@src(), gpa, back.backend(), .{});
    defer win.deinit();

    // Register the bundled broad-coverage font in a bootstrap frame (addFont /
    // themeSet need the current window) so CJK / symbols in LLM output render
    // instead of tofu boxes.
    try win.begin(win.frame_time_ns);
    fonts.install() catch |err| std.log.err("font install failed: {t}", .{err});
    _ = try win.end(.{});

    g_wakeup_event_type = SDLBackend.c.SDL_RegisterEvents(1);

    // Kick off the initial load if an LLM is configured; otherwise the chat
    // view shows a "set a model" notice until the user picks one in settings.
    if (g_config.llm_model.opt() != null) g_reload_requested = true;

    // Tear down the session (on the loader thread's CUDA context) at exit.
    defer {
        if (g_loader) |t| t.join();
        if (g_session) |s| {
            s.be.bindThread();
            s.deinit();
        }
        if (g_session_arena) |a| {
            a.deinit();
            g_gpa.destroy(a);
        }
        freeCarry();
        status_bar.deinit();
    }
    defer if (g_viewer) |v| v.deinit();

    var interrupted = false;
    main_loop: while (true) {
        maybeStartReload();
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

        // A clicked image opens (or refocuses) the viewer window.
        if (g_viewer_request) |gi| {
            g_viewer_request = null;
            if (g_viewer) |v| {
                v.setImage(gi);
                _ = SDLBackend.c.SDL_RaiseWindow(v.back.window);
            } else if (g_session) |s| {
                g_viewer = viewer.Viewer.init(init.gpa, init.io, s, gi) catch |err| vblk: {
                    std.log.err("open viewer failed: {t}", .{err});
                    break :vblk null;
                };
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

/// A file was dropped on the window: decode it (libvips → RGB) and attach it
/// to the next message for the model to see.
fn handleDropFile(path: []const u8) void {
    const s = g_session orelse return;
    if (!s.visionEnabled()) {
        std.log.warn("dropped {s} but vision is unavailable", .{path});
        return;
    }
    const gpa = std.heap.smp_allocator;
    const dec = vips.loadRgb(gpa, path) catch |err| {
        std.log.err("can't load dropped image {s}: {t}", .{ path, err });
        return;
    };
    defer gpa.free(dec.pixels);
    s.attachImage(dec.pixels, dec.width, dec.height) catch |err| std.log.err("attach image: {t}", .{err});
}

/// Ctrl/Cmd+V with an image on the clipboard: decode the raw bytes (any
/// libvips format) and attach it, exactly as a dropped file. Returns true
/// when an image was found on the clipboard (whether or not decoding
/// succeeded), so the caller can consume the event before the text entry
/// treats it as a text paste. Returns false when the clipboard holds no
/// image, letting normal text paste proceed.
fn tryPasteClipboardImage() bool {
    const SDL = SDLBackend.c;
    const s = g_session orelse return false;
    if (!s.visionEnabled()) return false;

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
    s.attachImage(dec.pixels, dec.width, dec.height) catch |err| std.log.err("attach pasted image: {t}", .{err});
    return true;
}

fn frame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .background = true });
    defer root.deinit();

    if (g_view == .config) {
        config_view.render(&g_config, .{ .apply = applyConfig, .cancel = cancelConfig });
        return;
    }

    // Chat view. `g_session` is only touched once the loader has published it
    // (loading flag clear); until then show the load state.
    if (g_loading.load(.acquire)) {
        renderLoading();
        return;
    }
    const s = g_session orelse {
        renderNoModel();
        return;
    };

    s.poll();

    // Pin the input strip to the bottom: cap the message list's height to the
    // space left after the (dynamically-measured) input row. A scrollArea
    // reports its full content height as its min size, so as a plain flex child
    // it would push the input off-screen (dvui's box sums every child's min
    // height). max_size_content caps that.
    const list_h = @max(120, root.data().contentRect().h - g_input_h - status_bar.bar_height);

    renderMessages(s, list_h);
    renderInput(s);
    status_bar.render(s);
}

/// Chat view while the session (re)loads on the background thread.
fn renderLoading() void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5 });
    defer col.deinit();
    dvui.spinner(@src(), .{ .gravity_x = 0.5, .min_size_content = .{ .w = 32, .h = 32 } });
    const name = std.fs.path.basename(g_config.llm_model.slice());
    dvui.label(@src(), "Loading {s}…", .{name}, .{ .gravity_x = 0.5, .padding = .{ .y = 8 } });
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
    if (dvui.button(@src(), "Open Settings", .{}, .{ .gravity_x = 0.5, .margin = .{ .y = 12 } })) openSettings();
}

fn openSettings() void {
    config_view.open();
    g_view = .config;
}

/// Apply button: persist settings, then apply them WITHOUT wiping the chat.
/// - LLM/vision/VRAM-limit change, or an image-tool enable/disable toggle →
///   a transcript-preserving reload (loaderMain carries the chat across).
/// - Everything else (image defaults, preview, priority, and a diffusion-model
///   swap) → applied live to the running session.
fn applyConfig() void {
    g_config.save(g_io, g_gpa, g_environ, g_config_path) catch |err| std.log.err("save settings failed: {t}", .{err});
    const llm_changed = !g_config.llmReloadEql(&g_config_baseline);
    const tool_toggled = g_config.diffEnabled() != g_config_baseline.diffEnabled();
    if (llm_changed or tool_toggled) {
        g_reload_requested = true; // transcript-preserving (see loaderMain)
    } else {
        applyLiveSettings();
    }
    g_config_baseline = g_config;
    g_view = .chat;
}

/// Push live-applicable settings into the running session without a reload:
/// image defaults, preview method, VRAM priority, and — when the diffusion model
/// set changed but the tool stayed enabled — a deferred diffusion-model swap
/// (in-flight/queued images finish on the current model first). Only touches
/// `g_session` while no load is in flight (release/acquire, like the render path).
fn applyLiveSettings() void {
    if (g_loading.load(.acquire)) return;
    const s = g_session orelse return;
    s.updateSettings(&g_config);
    if (!g_config.diffPathsEql(&g_config_baseline) and g_config.diffEnabled()) {
        s.requestDiffPaths(
            g_config.diffusion_model.opt().?,
            g_config.vae.opt().?,
            g_config.text_encoder.opt().?,
            g_config.taesd.opt(),
        );
    }
}

/// Cancel button: discard unsaved edits by reloading the on-disk settings.
fn cancelConfig() void {
    g_config = config.Config.load(g_io, g_gpa, g_environ, g_config_path);
    g_config_baseline = g_config;
    g_view = .chat;
}

/// Main-loop hook: reap a finished loader, and start a pending (re)load when
/// none is in flight. Runs on the UI thread so the loading-flag hand-off to the
/// loader is well-ordered.
fn maybeStartReload() void {
    if (g_loader) |t| {
        if (!g_loading.load(.acquire)) {
            t.join();
            g_loader = null;
        }
    }
    if (!g_reload_requested or g_loading.load(.acquire) or g_loader != null) return;
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
        if (s.diff_thread) |t| {
            t.join();
            s.diff_thread = null;
            s.diff_busy.store(false, .release);
        }
        g_carry = s.detachTranscript();
        s.deinit();
        g_session = null;
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
        return finishLoad(err);
    };
    // Replay the carried transcript into the new model (KV empty; the next turn's
    // prefill replays it) so a model swap keeps the chat.
    if (g_carry) |m| {
        s.be.bindThread();
        s.adoptTranscript(m) catch |err| std.log.err("adopt transcript failed: {t}", .{err});
        g_carry = null;
    }
    const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(g_io).nanoseconds - t0)) / 1e9;
    std.log.info("session ready in {d:.1}s", .{dt});
    g_session = s;
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

/// Build a session from the current settings, mapping unset optional models to
/// disabled features (diffusion needs dit+vae+text-encoder; vision needs the
/// tower). Paths are duped into `arena` so the session never aliases the live
/// config edit buffers.
fn buildSession(arena: std.mem.Allocator) !*chat.Session {
    const llm = g_config.llm_model.opt().?;

    var diff: ?chat.DiffConfig = null;
    if (g_config.diffusion_model.opt()) |dit| {
        if (g_config.vae.opt()) |vae| if (g_config.text_encoder.opt()) |te| {
            diff = .{
                .dit_path = try arena.dupe(u8, dit),
                .vae_path = try arena.dupe(u8, vae),
                .text_encoder_path = try arena.dupe(u8, te),
                .steps = g_config.steps,
                .width = g_config.width,
                .height = g_config.height,
                .backend = .zig_cuda,
                .preview_enabled = g_config.preview != .none,
                .taew_path = if (g_config.preview == .taesd)
                    (if (g_config.taesd.opt()) |t| try arena.dupe(u8, t) else null)
                else
                    null,
            };
        };
    }

    const mmproj: ?[]const u8 = if (g_config.vision_tower.opt()) |m| try arena.dupe(u8, m) else null;

    const system_prompt = g_config.system_prompt.opt() orelse config.default_system_prompt;

    const s = try chat.Session.init(arena, g_gpa, g_io, wakeupFrame, .{
        .model_path = try arena.dupe(u8, llm),
        .system_prompt = try arena.dupe(u8, system_prompt),
        .seed = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(g_io).nanoseconds))),
        .diff = diff,
        .mmproj_path = mmproj,
        .vram_limit_bytes = if (g_config.max_vram_gib > 0)
            @intFromFloat(g_config.max_vram_gib * @as(f32, 1 << 30))
        else
            0,
        .vram_priority = g_config.vram_priority,
    });
    return s;
}

fn renderMessages(s: *chat.Session, list_h: f32) void {
    {
        var scroll = dvui.scrollArea(@src(), .{ .scroll_info = &g_scroll_info }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = list_h },
            .max_size_content = .height(list_h),
        });
        defer scroll.deinit();

        var list = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
        defer list.deinit();

        if (s.messages.items.len == 0) {
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = dvui.Rect.all(16) });
            defer tl.deinit();
            tl.addText("Say something to start the conversation.", .{});
        } else {
            for (s.messages.items, 0..) |*m, idx| renderMessage(s, m, idx);
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

fn renderMessage(s: *chat.Session, m: *const chat.Message, idx: usize) void {
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

    const p = parseThink(m.text.items);
    // Only the last assistant message is actively generating; "Thinking…" means
    // the block is still open AND generation is live. A think block left open
    // because generation stopped (e.g. hit max tokens) reads "Thoughts".
    const live = s.busy() and idx + 1 == s.messages.items.len;

    // Qwen3.5 reasoning: collapse <think>…</think> behind an expander. The
    // label doubles as a "thinking" indicator while the block is still open.
    if (p.think) |think| {
        if (dvui.expander(@src(), if (p.thinking and live) "Thinking…" else "Thoughts", .{ .default_expanded = false }, .{})) {
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
            defer tl.deinit();
            fonts.addRich(tl, think);
        }
    }

    if (p.answer.len > 0) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
        defer tl.deinit();
        addAnswerText(tl, p.answer);
    } else if (m.role == .assistant and p.think == null and m.images.items.len == 0) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
        defer tl.deinit();
        if (live) {
            tl.addText("…", .{});
        } else if (s.gen_err) |err| {
            var msg: [128]u8 = undefined;
            fonts.addRich(tl, std.fmt.bufPrint(&msg, "⚠ generation error: {t}", .{err}) catch "⚠ generation error");
        }
    }

    // Generated images requested by this message, below its text. Offset the
    // id base so it can't collide with attachment image ids in this bubble.
    for (m.images.items, 0..) |gi, gi_idx| renderGenImage(s, gi, 100_000 + gi_idx);
}

/// Add answer text to the layout with `<image>…</image>` tool-call tags hidden
/// (the images render separately). A still-streaming, unterminated open tag
/// hides everything from it onward.
fn addAnswerText(tl: *dvui.TextLayoutWidget, text: []const u8) void {
    const close = "</image>";
    var rest = text;
    // Match the `<image` prefix (tags may carry attributes, e.g.
    // `<image width=1024>`), mirroring the parser in chat.zig.
    while (std.mem.indexOf(u8, rest, "<image")) |a| {
        fonts.addRich(tl, rest[0..a]);
        const after_open = rest[a + "<image".len ..];
        const gt = std.mem.indexOfScalar(u8, after_open, '>') orelse return; // open tag still streaming: hide rest
        const body = after_open[gt + 1 ..];
        const b = std.mem.indexOf(u8, body, close) orelse return; // body still streaming: hide rest
        rest = body[b + close.len ..];
    }
    fonts.addRich(tl, rest);
}

/// Display size for an image: downscale so the longer side is `max`, never
/// upscale. Sizing the widget to the actual (aspect-correct) dimensions avoids
/// the letterbox padding a square cap would add for non-square images.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    const mx: f32 = @floatFromInt(@max(w, h));
    const scale = if (mx > max) max / mx else 1.0;
    return .{ .w = @as(f32, @floatFromInt(w)) * scale, .h = @as(f32, @floatFromInt(h)) * scale };
}

fn renderGenImage(s: *chat.Session, gi: *chat.GenImage, gi_idx: usize) void {
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = gi_idx, .expand = .horizontal, .margin = .{ .y = 4, .h = 4 } });
    defer b.deinit();

    switch (gi.get()) {
        .pending, .generating => {
            const generating = gi.get() == .generating;
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
            const status = if (!generating)
                "🖼  Queued…"
            else if (sps > 0)
                std.fmt.bufPrint(&buf, "🖼  Generating…  step {d}/{d}  ·  {d:.2} s/step", .{ done, total, sps }) catch "Generating…"
            else
                std.fmt.bufPrint(&buf, "🖼  Generating…  step {d}/{d}", .{ done, total }) catch "Generating…";
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

                // Let the model see this image: attach it to the next message.
                if (s.visionEnabled()) {
                    if (dvui.button(@src(), "Discuss this image", .{}, .{ .margin = .{ .y = 4 } })) {
                        s.attachRgba(rgba, gi.width, gi.height) catch |err| std.log.err("attach image: {t}", .{err});
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
    dvui.label(@src(), "{s}", .{meta}, .{ .margin = .{ .y = 2 } });
    if (dvui.expander(@src(), "Prompt", .{ .default_expanded = false }, .{})) {
        var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = .{ .x = 6, .y = 2, .w = 6, .h = 4 } });
        defer tl.deinit();
        fonts.addRich(tl, gi.prompt);
    }
}

const Parsed = struct { think: ?[]const u8, answer: []const u8, thinking: bool };

/// Split an assistant message into its <think>…</think> reasoning and the
/// answer. `thinking` is true while the block is still open (no </think> yet).
fn parseThink(text: []const u8) Parsed {
    const ws = " \n\r\t";
    const t = std.mem.trimStart(u8, text, ws);
    if (std.mem.startsWith(u8, t, "<think>")) {
        const rest = t["<think>".len..];
        if (std.mem.indexOf(u8, rest, "</think>")) |end| {
            return .{
                .think = std.mem.trim(u8, rest[0..end], ws),
                .answer = std.mem.trimStart(u8, rest[end + "</think>".len ..], ws),
                .thinking = false,
            };
        }
        return .{ .think = std.mem.trimStart(u8, rest, ws), .answer = "", .thinking = true };
    }
    return .{ .think = null, .answer = text, .thinking = false };
}

fn renderInput(s: *chat.Session) void {
    var container = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal });
    defer container.deinit();

    // Thumbnails of images attached (dropped) but not yet sent, each with a
    // hover-only X to remove it before sending.
    const pending = s.pendingAttachments();
    if (pending.len > 0) {
        var remove_idx: ?usize = null;
        {
            var strip = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 8, .y = 4, .w = 8 } });
            defer strip.deinit();
            for (pending, 0..) |gi, pi| {
                if (gi.rgba == null) continue;
                const sz = fitSize(gi.width, gi.height, 56);
                var ov = dvui.overlay(@src(), .{ .id_extra = pi, .margin = .{ .w = 6 } });
                _ = dvui.image(@src(), .{
                    .source = .{ .pixels = .{ .rgba = gi.rgba.?, .width = @intCast(gi.width), .height = @intCast(gi.height) } },
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
                    })) remove_idx = pi;
                }
                ov.deinit();
            }
        }
        if (remove_idx) |ri| s.removeAttachment(ri);
    }

    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = dvui.Rect.all(8) });
    defer row.deinit();

    const busy = s.busy();

    // New chat: drop the conversation and start fresh (model stays resident).
    // Disabled mid-turn so it can't race the generation worker.
    if (dvui.buttonIcon(@src(), "new-chat", dvui.entypo.plus, .{}, .{}, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 22, .h = 22 },
        .margin = .{ .w = 6 },
    }) and !busy) newChat();

    // Gear: switch to the settings view.
    if (dvui.buttonIcon(@src(), "settings", dvui.entypo.cog, .{}, .{}, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 22, .h = 22 },
        .margin = .{ .w = 6 },
    })) openSettings();

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
    }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        .min_size_content = .{ .h = 28 },
        .max_size_content = .height(140),
    });
    g_input_id = te.data().id;
    // Reserve for next frame's layout: entry height + row padding, plus the
    // attachment strip when present.
    g_input_h = te.data().rect.h + 24 + (if (pending.len > 0) @as(f32, 72) else 0);
    if (send) {
        const t = te.getText();
        n = @min(t.len, buf.len);
        @memcpy(buf[0..n], t[0..n]);
    }
    te.deinit();

    if (dvui.button(@src(), if (busy) "Stop" else "Send", .{}, .{ .gravity_y = 0.5 })) {
        if (busy) {
            s.requestCancel();
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
            s.submit(buf[0..n]) catch |err| std.log.err("submit failed: {t}", .{err});
        }
        @memset(&g_input_buf, 0);
    }
}

/// Start a fresh conversation on the resident session, clearing the input box.
/// No-op if no session is loaded (nothing to reset).
fn newChat() void {
    // Close the image viewer first: reset frees the GenImages it points at.
    if (g_viewer) |v| v.open = false;
    if (g_session) |s| s.reset();
    @memset(&g_input_buf, 0);
    g_follow_bottom = true;
}
