//! The text-to-image studio: a full-window view for generating images directly
//! from a prompt, with no LLM in the loop. It owns NOTHING persistent — the
//! diffusion engine (`diffuser.Diffuser`) is app-level and owns the single
//! unified image queue/history (shared with the chat tool-call path). This
//! module is just the studio UI: a prompt/param form plus a results grid that
//! renders the engine's full image list (`engine.items()`), newest first.
//!
//! State is process-global (the dvui idiom, mirroring app.zig): only the
//! transient form fields live here.
const std = @import("std");
const dvui = @import("dvui");
const config = @import("config.zig");
const diffuser = @import("diffuser.zig");
const clipboard = @import("clipboard.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");

const GenImage = diffuser.GenImage;

var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_wake: *const fn () void = undefined;
var g_env_ready: bool = false;

/// Set when a done image is clicked; app.zig opens/refocuses the viewer.
pub var viewer_request: ?*GenImage = null;

// Form fields (numeric ones edited as text, like the settings view).
var seeded: bool = false;
var prompt_buf: [4096]u8 = [_]u8{0} ** 4096;
var negative_buf: [1024]u8 = [_]u8{0} ** 1024;
var width_buf: [12]u8 = [_]u8{0} ** 12;
var height_buf: [12]u8 = [_]u8{0} ** 12;
var steps_buf: [8]u8 = [_]u8{0} ** 8;
var cfg_buf: [8]u8 = [_]u8{0} ** 8;
// Wide enough to type any u64 (20 digits + null): the queue count is
// deliberately uncapped — see `generate`.
var count_buf: [24]u8 = [_]u8{0} ** 24;
var seed_buf: [24]u8 = [_]u8{0} ** 24;
var random_seed: bool = true;

/// Give the studio the process allocator/clock/wake it needs to create images.
/// Called once from app startup.
pub fn setEnv(gpa: std.mem.Allocator, io: std.Io, wake: *const fn () void) void {
    g_gpa = gpa;
    g_io = io;
    g_wake = wake;
    g_env_ready = true;
}

/// Nothing persistent to free (the engine owns the images); reset transient UI
/// state at exit.
pub fn deinit() void {
    seeded = false;
    viewer_request = null;
}

fn seed(cfg: *const config.Config) void {
    _ = std.fmt.bufPrintZ(&width_buf, "{d}", .{cfg.width}) catch {};
    _ = std.fmt.bufPrintZ(&height_buf, "{d}", .{cfg.height}) catch {};
    _ = std.fmt.bufPrintZ(&steps_buf, "{d}", .{cfg.steps}) catch {};
    _ = std.fmt.bufPrintZ(&cfg_buf, "{d:.1}", .{@as(f32, 1.0)}) catch {};
    _ = std.fmt.bufPrintZ(&count_buf, "{d}", .{@as(usize, 1)}) catch {};
    _ = std.fmt.bufPrintZ(&seed_buf, "{d}", .{@as(u64, 0)}) catch {};
    seeded = true;
}

fn parseNum(buf: []const u8, fallback: usize) usize {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseInt(usize, s, 10) catch fallback;
}

fn parseFloat(buf: []const u8, fallback: f32) f32 {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseFloat(f32, s) catch fallback;
}

pub const Callbacks = struct {
    to_chat: *const fn () void,
    settings: *const fn () void,
};

/// Render the studio. `d` is the app's diffusion engine (null when no diffusion
/// model is configured → a notice is shown). `ready` is false while the LLM is
/// still being torn down (Generate is disabled until the device is free).
pub fn render(cfg: *const config.Config, d: ?*diffuser.Diffuser, ready: bool, cb: Callbacks) void {
    if (!seeded) seed(cfg);

    // Header: title + Chat / Settings actions.
    {
        var header = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = dvui.Rect.all(10) });
        defer header.deinit();
        dvui.label(@src(), "Image studio", .{}, .{ .font = .theme(.title), .gravity_y = 0.5 });
        {
            var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
            sp.deinit();
        }
        if (dvui.button(@src(), "Chat", .{}, .{ .gravity_y = 0.5 })) cb.to_chat();
        var wd: dvui.WidgetData = undefined;
        if (dvui.buttonIcon(@src(), "settings", dvui.entypo.cog, .{}, .{}, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = 22, .h = 22 },
            .margin = .{ .x = 6 },
            .data_out = &wd,
        })) cb.settings();
        hint.hover(@src(), &wd, "Settings");
    }

    const engine = d orelse {
        // No diffusion model configured: explain + shortcut to settings.
        var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5, .padding = dvui.Rect.all(24) });
        defer col.deinit();
        var tl = dvui.textLayout(@src(), .{}, .{ .gravity_x = 0.5 });
        fonts.addRich(tl, "No diffusion model is set.\n\nOpen Settings and choose a diffusion model, text encoder, and VAE to generate images here.");
        tl.deinit();
        if (dvui.button(@src(), "Open Settings", .{}, .{ .gravity_x = 0.5, .margin = .{ .y = 12 } })) cb.settings();
        return;
    };

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = dvui.Rect.all(8) });
    defer body.deinit();

    renderForm(cfg, engine, ready);
    renderGallery(engine);
}

fn renderForm(cfg: *const config.Config, engine: *diffuser.Diffuser, ready: bool) void {
    // Prompt.
    dvui.label(@src(), "Prompt", .{}, .{ .padding = .{ .x = 2, .y = 4 } });
    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &prompt_buf },
            .multiline = true,
            .placeholder = "Describe the image…",
        }, .{ .expand = .horizontal, .min_size_content = .{ .h = 60 }, .max_size_content = .height(160) });
        te.deinit();
    }

    // Negative prompt (only bites when CFG > 1).
    dvui.label(@src(), "Negative prompt", .{}, .{ .padding = .{ .x = 2, .y = 4 } });
    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &negative_buf },
            .placeholder = "things to avoid (needs CFG > 1)",
        }, .{ .expand = .horizontal, .min_size_content = .{ .h = 24 } });
        te.deinit();
    }

    // Params row.
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 2, .y = 6 } });
        defer row.deinit();
        numField(0, "Width", &width_buf);
        numField(1, "Height", &height_buf);
        numField(2, "Steps", &steps_buf);
        numField(3, "CFG", &cfg_buf);
        numField(4, "Count", &count_buf);
    }

    // Seed row.
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 2, .y = 2 } });
        defer row.deinit();
        _ = dvui.checkbox(@src(), &random_seed, "Random seed", .{ .gravity_y = 0.5 });
        if (!random_seed) {
            dvui.label(@src(), "Seed", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 12 } });
            var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = &seed_buf } }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 160, .h = 20 } });
            te.deinit();
        }
    }

    // Generate / Stop.
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 2, .y = 6 } });
        defer row.deinit();
        const generating = engine.busyNow() or engine.hasPending();
        if (generating) {
            if (dvui.button(@src(), "Stop", .{}, .{ .gravity_y = 0.5 })) engine.cancelAll();
        } else if (!ready) {
            dvui.label(@src(), "Loading model…", .{}, .{ .gravity_y = 0.5 });
        } else {
            if (dvui.button(@src(), "Generate", .{}, .{ .gravity_y = 0.5 })) generate(cfg, engine);
        }
    }
}

fn numField(id: usize, label: []const u8, buf: []u8) void {
    var cell = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = id, .margin = .{ .w = 10 } });
    defer cell.deinit();
    dvui.label(@src(), "{s}", .{label}, .{ .gravity_y = 0.5, .margin = .{ .w = 4 } });
    var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = buf } }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 64, .h = 20 } });
    te.deinit();
}

/// Queue `count` generations from the current form values into the engine's
/// unified queue and start the pump.
fn generate(cfg: *const config.Config, engine: *diffuser.Diffuser) void {
    const prompt = std.mem.trim(u8, std.mem.sliceTo(&prompt_buf, 0), " \t\r\n");
    if (prompt.len == 0) return;
    const neg = std.mem.trim(u8, std.mem.sliceTo(&negative_buf, 0), " \t\r\n");

    const w = diffuser.clampDim(parseNum(&width_buf, cfg.width));
    const h = diffuser.clampDim(parseNum(&height_buf, cfg.height));
    const steps = std.math.clamp(parseNum(&steps_buf, cfg.steps), 1, 100);
    const cfg_scale = std.math.clamp(parseFloat(&cfg_buf, 1.0), 0.0, 30.0);
    // Uncapped by design: queue as many as you ask for (min 1). Each is
    // allocated up front, so a huge count is on you — that's the intent.
    const count = @max(1, parseNum(&count_buf, 1));
    const base_seed: u64 = if (random_seed) 0 else std.fmt.parseInt(u64, std.mem.trim(u8, std.mem.sliceTo(&seed_buf, 0), " \t\r"), 10) catch 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const gi = g_gpa.create(GenImage) catch return;
        gi.* = .{
            .prompt = g_gpa.dupe(u8, prompt) catch {
                g_gpa.destroy(gi);
                return;
            },
            .wake = g_wake,
            .io = g_io,
            .req_width = w,
            .req_height = h,
            .req_steps = steps,
            .req_cfg = cfg_scale,
            // Random: a fresh distinct seed each. Fixed: the entered seed,
            // advanced per image so a batch still varies.
            .req_seed = if (random_seed) engine.nextSeed() else base_seed +% i,
        };
        if (neg.len > 0) gi.req_negative = g_gpa.dupe(u8, neg) catch "";
        engine.enqueue(gi) catch {
            diffuser.freeGenImage(g_gpa, gi);
            return;
        };
    }
    engine.pump();
}

/// Display size for an image: downscale so the longer side is `max`, never up.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    const mx: f32 = @floatFromInt(@max(w, h));
    const scale = if (mx > max) max / mx else 1.0;
    return .{ .w = @as(f32, @floatFromInt(w)) * scale, .h = @as(f32, @floatFromInt(h)) * scale };
}

/// The results grid — the engine's whole image list (chat + studio), newest
/// first, laid out in as many columns as fit.
fn renderGallery(engine: *diffuser.Diffuser) void {
    const imgs = engine.items();
    if (imgs.len == 0) {
        dvui.label(@src(), "Generated images appear here.", .{}, .{ .padding = dvui.Rect.all(16) });
        return;
    }
    var grid = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .y = 8 } });
    defer grid.deinit();

    const cell: f32 = 240;
    const avail = @max(cell, grid.data().contentRect().w);
    const cols = @max(@as(usize, 1), @as(usize, @intFromFloat(avail / (cell + 12))));

    const n = imgs.len;
    var i: usize = 0;
    while (i < n) : (i += cols) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = i, .expand = .horizontal, .margin = .{ .h = 8 } });
        defer row.deinit();
        var c: usize = 0;
        while (c < cols and i + c < n) : (c += 1) {
            // Newest first.
            renderCell(imgs[n - 1 - (i + c)], i + c, cell);
        }
    }
}

fn renderCell(gi: *GenImage, idx: usize, cell: f32) void {
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = idx, .min_size_content = .{ .w = cell }, .margin = .{ .w = 12 } });
    defer b.deinit();

    switch (gi.get()) {
        .pending, .generating => {
            const generating = gi.get() == .generating;
            if (generating) dvui.refresh(null, @src(), null);
            const done = gi.step.load(.monotonic);
            const total = gi.total.load(.monotonic);
            if (gi.preview) |pv| {
                const pw = gi.preview_w.load(.acquire);
                const ph = gi.preview_h.load(.acquire);
                if (pw > 0 and ph > 0) {
                    const sz = fitSize(pw, ph, cell);
                    _ = dvui.image(@src(), .{
                        .source = .{ .pixels = .{ .rgba = pv[0 .. pw * ph * 4], .width = pw, .height = ph, .invalidation = .always } },
                        .shrink = .ratio,
                    }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                }
            }
            const pct: f32 = if (total > 0) @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total)) else 0;
            dvui.progress(@src(), .{ .percent = pct }, .{ .expand = .horizontal, .min_size_content = .{ .h = 6 }, .margin = .{ .y = 3 }, .corner_radius = dvui.Rect.all(3) });
            var buf: [48]u8 = undefined;
            const status = if (!generating) "Queued…" else std.fmt.bufPrint(&buf, "step {d}/{d}", .{ done, total }) catch "…";
            dvui.label(@src(), "{s}", .{status}, .{});
            if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .{ .y = 2 } })) {
                gi.cancel.store(true, .release);
                gi.wake();
            }
        },
        .done => {
            if (gi.rgba) |rgba| {
                const sz = fitSize(gi.width, gi.height, cell);
                var ib = dvui.box(@src(), .{}, .{});
                _ = dvui.image(@src(), .{
                    .source = .{ .pixels = .{ .rgba = rgba, .width = @intCast(gi.width), .height = @intCast(gi.height) } },
                    .shrink = .ratio,
                }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = dvui.Rect.all(6) });
                const clicked = dvui.clicked(ib.data(), .{});
                ib.deinit();
                if (clicked) viewer_request = gi;

                var meta_row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 2 } });
                defer meta_row.deinit();
                var mbuf: [64]u8 = undefined;
                dvui.label(@src(), "{s}", .{std.fmt.bufPrint(&mbuf, "{d}×{d} · seed {d}", .{ gi.width, gi.height, gi.req_seed }) catch ""}, .{ .gravity_y = 0.5 });
                {
                    var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
                    sp.deinit();
                }
                // Copy the image to the clipboard as a PNG.
                var wd: dvui.WidgetData = undefined;
                if (dvui.buttonIcon(@src(), "copy", dvui.entypo.clipboard, .{}, .{}, .{
                    .min_size_content = .{ .w = 16, .h = 16 },
                    .gravity_y = 0.5,
                    .data_out = &wd,
                })) clipboard.copyImage(gi);
                hint.hover(@src(), &wd, "Copy image to clipboard");
            }
        },
        .failed => dvui.label(@src(), "⚠ failed", .{}, .{}),
        .canceled => dvui.label(@src(), "⚠ canceled", .{}, .{}),
    }
}
