//! The text-to-image studio: a full-window view for generating images directly
//! from a prompt, with no LLM in the loop. It owns NOTHING persistent, the
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
const tp = @import("TensorPencil");
const diffuser = @import("diffuser.zig");
const model_spec = @import("model_spec.zig");
const clipboard = @import("clipboard.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");
const style = @import("style.zig");
const bubbles = @import("bubbles.zig");

const GenImage = diffuser.GenImage;

var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_wake: *const fn () void = undefined;
var g_env_ready: bool = false;

/// Set when a done image is clicked; app.zig opens/refocuses the viewer.
pub var viewer_request: ?*GenImage = null;

// Memoized inspection of the configured checkpoint, so the form can default to
// the architecture's own parameters (see `seed`). Live only after `setEnv`.
var g_spec: model_spec.Cache = undefined;
var g_spec_ready: bool = false;

// Form fields (numeric ones edited as text, like the settings view).
var seeded: bool = false;

/// Drop the seeded form fields so the next render re-reads the config. Called
/// when the composer's framing chips change the default size; without it the
/// studio keeps generating at whatever size it was first opened with.
pub fn reseed() void {
    seeded = false;
}
/// The architecture the form was last seeded for. A change here means the user
/// picked a different kind of model, and the CFG default is re-seeded.
var seeded_family: ?model_spec.Family = null;
var prompt_buf: [4096]u8 = [_]u8{0} ** 4096;
var negative_buf: [1024]u8 = [_]u8{0} ** 1024;
var width_buf: [12]u8 = [_]u8{0} ** 12;
var height_buf: [12]u8 = [_]u8{0} ** 12;
var steps_buf: [8]u8 = [_]u8{0} ** 8;
var cfg_buf: [8]u8 = [_]u8{0} ** 8;
// Wide enough to type any u64 (20 digits + null): the queue count is
// deliberately uncapped, see `generate`.
var count_buf: [24]u8 = [_]u8{0} ** 24;
var seed_buf: [24]u8 = [_]u8{0} ** 24;
var random_seed: bool = true;

/// Give the studio the process allocator/clock/wake it needs to create images.
/// Called once from app startup.
pub fn setEnv(gpa: std.mem.Allocator, io: std.Io, wake: *const fn () void) void {
    g_gpa = gpa;
    g_io = io;
    g_wake = wake;
    g_spec = model_spec.Cache.init(gpa, io);
    g_spec_ready = true;
    g_env_ready = true;
}

/// Nothing persistent to free (the engine owns the images); reset transient UI
/// state at exit.
pub fn deinit() void {
    seeded = false;
    seeded_family = null;
    viewer_request = null;
    if (g_spec_ready) {
        g_spec.deinit();
        g_spec_ready = false;
    }
}

/// The architecture of the configured checkpoint, if it can be read. Memoized on
/// the path, so this is one header parse per model change, not per frame.
///
/// A file REPLACED under an unchanged path keeps the old answer here (only the
/// settings screen re-probes, on every open). That is deliberate: the worst
/// consequence is a stale CFG *suggestion* in a form field the user can edit,
/// which does not justify re-parsing a checkpoint header every frame.
fn currentFamily(cfg: *const config.Config) ?model_spec.Family {
    if (!g_spec_ready) return null;
    const path = cfg.diffusion_model.opt() orelse return null;
    const info = g_spec.primary(path).info() orelse return null;
    return info.family;
}

fn seed(cfg: *const config.Config, fam: ?model_spec.Family) void {
    // Width / height come from the composer's framing chips and steps from
    // Settings: both are the user's explicit global defaults, so the form must
    // not overwrite them with an architecture's suggestion. `reseed` re-runs
    // this when the framing changes.
    _ = std.fmt.bufPrintZ(&width_buf, "{d}", .{cfg.width}) catch {};
    _ = std.fmt.bufPrintZ(&height_buf, "{d}", .{cfg.height}) catch {};
    _ = std.fmt.bufPrintZ(&steps_buf, "{d}", .{cfg.steps}) catch {};
    // CFG has no Settings field: it was a hardcoded 1.0, which is krea2's value
    // and *disables* classifier-free guidance. SD1.5 at 1.0 ignores the negative
    // prompt entirely and produces a washed, prompt-adherent-in-name-only image,
    // so the default has to follow the architecture.
    const guidance = if (fam) |f| model_spec.traits(f).cfg else 1.0;
    _ = std.fmt.bufPrintZ(&cfg_buf, "{d:.1}", .{guidance}) catch {};
    _ = std.fmt.bufPrintZ(&count_buf, "{d}", .{@as(usize, 1)}) catch {};
    _ = std.fmt.bufPrintZ(&seed_buf, "{d}", .{@as(u64, 0)}) catch {};
    seeded = true;
    seeded_family = fam;
}

fn famEql(a: ?model_spec.Family, b: ?model_spec.Family) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return a.? == b.?;
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
    settings: *const fn () void,
};

/// Render the studio. `d` is the app's diffusion engine (null when no diffusion
/// model is configured -> a notice is shown). `ready` is false while the LLM is
/// still being torn down (Generate is disabled until the device is free).
pub fn render(cfg: *const config.Config, d: ?*diffuser.Diffuser, ready: bool, cb: Callbacks) void {
    const fam = currentFamily(cfg);
    // Re-seed when the user switches to a different ARCHITECTURE: the form's CFG
    // default is architecture-specific and carrying krea2's over to SD1.5 (or
    // back) is a visibly wrong image, not a preference.
    if (!seeded or !famEql(fam, seeded_family)) seed(cfg, fam);

    // No header here: the window's title bar already names this view and owns
    // the Chat/Studio switch, so a second title row was two things claiming to
    // be the top of the screen.

    const engine = d orelse {
        // No diffusion model configured: explain + shortcut to settings.
        var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5, .padding = dvui.Rect.all(24) });
        defer col.deinit();
        var tl = dvui.textLayout(@src(), .{}, .{ .gravity_x = 0.5, .background = false, .max_size_content = .width(style.Layout.prose_max) });
        fonts.addStyled(tl, "No diffusion model is set.\n\nOpen Settings and choose a diffusion model to generate images here. " ++
            "Most checkpoints bundle everything they need; the text encoder and VAE fields are only for supplying or replacing a piece.", .{}, .{
            .font = style.F.prose,
            .color_text = style.C.text,
        });
        tl.deinit();
        var actions = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 14 } });
        const go = bubbles.primaryButton(@src(), "Open Settings", true);
        actions.deinit();
        if (go) cb.settings();
        return;
    };

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = dvui.Rect.all(8) });
    defer body.deinit();

    renderModelNotice(cfg, engine, fam);
    renderForm(cfg, engine, ready);
    renderGallery(engine);
}

/// Tell the user, BEFORE they hit Generate, when this model set cannot produce an
/// image, and afterwards, why the last load failed.
///
/// A model set can now fail to load for reasons the studio has no other way to
/// show: a missing component the checkpoint does not bundle, an override holding
/// the wrong architecture's weights, or a backend without kernels for it. Without
/// this they surface as a bare "failed" on the image, with the reason only in the
/// terminal.
fn renderModelNotice(cfg: *const config.Config, engine: *diffuser.Diffuser, fam: ?model_spec.Family) void {
    var buf: [320]u8 = undefined;
    const theme = dvui.themeGet();
    const alarm = theme.err.fill orelse theme.text;

    const text: []const u8 = blk: {
        if (engine.loadError()) |err| break :blk std.fmt.bufPrint(
            &buf,
            "The diffusion model failed to load: {t}. Check the model paths and backend in Settings.",
            .{err},
        ) catch "The diffusion model failed to load.";
        const f = fam orelse return; // unreadable/unknown: Settings says why
        const t = model_spec.traits(f);
        const want = diffuser.toPipelineBackend(cfg.diff_backend);
        if (!t.supports(want)) break :blk std.fmt.bufPrint(
            &buf,
            "{s} has no kernels for the {t} backend. Generation will fail to load — change it in Settings.",
            .{ t.label, cfg.diff_backend },
        ) catch "This architecture has no kernels for the selected backend.";
        return;
    };

    var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .padding = .{ .x = 2, .y = 4 }, .color_text = alarm });
    defer tl.deinit();
    tl.addText(text, .{});
}

fn renderForm(cfg: *const config.Config, engine: *diffuser.Diffuser, ready: bool) void {
    // Prompt.
    dvui.label(@src(), "Prompt", .{}, .{ .padding = .{ .x = 2, .y = 4 } });
    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &prompt_buf },
            .multiline = true,
            .placeholder = "Describe the image…",
            .scroll_horizontal = false,
            .break_lines = true,
        }, .{ .expand = .horizontal, .min_size_content = .{ .h = 60 }, .max_size_content = .height(160) });
        te.deinit();
    }

    // Negative prompt (only bites when CFG > 1).
    dvui.label(@src(), "Negative prompt", .{}, .{ .padding = .{ .x = 2, .y = 4 } });
    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &negative_buf },
            .multiline = true,
            .placeholder = "things to avoid (needs CFG > 1)",
            .scroll_horizontal = false,
            .break_lines = true,
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
            if (dvui.button(@src(), "Generate", .{}, .{ .gravity_y = 0.5 })) generate(cfg, engine);
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
    // Empty is allowed: it encodes to the unconditional embedding, which is a render
    // the model can do and some workflows want.
    const prompt = std.mem.trim(u8, std.mem.sliceTo(&prompt_buf, 0), " \t\r\n");
    const neg = std.mem.trim(u8, std.mem.sliceTo(&negative_buf, 0), " \t\r\n");

    const w = diffuser.clampDim(parseNum(&width_buf, cfg.width));
    const h = diffuser.clampDim(parseNum(&height_buf, cfg.height));
    const steps = std.math.clamp(parseNum(&steps_buf, cfg.steps), 1, 100);
    const cfg_scale = std.math.clamp(parseFloat(&cfg_buf, 1.0), 0.0, 30.0);
    // Uncapped by design: queue as many as you ask for (min 1). Each is
    // allocated up front, so a huge count is on you, that's the intent.
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
            .from_studio = true,
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

/// Load a finished image's parameters into the studio form, for the tool
/// card's "Open in Studio". Everything the user would otherwise re-type by
/// reading the metadata line: prompt, negative, size, steps, cfg, and the exact
/// seed, so the first thing Generate does is reproduce what they were looking
/// at, and every edit from there is a deliberate change to a known starting
/// point.
pub fn loadFrom(gi: *const GenImage) void {
    // Prefer the SAVED FILE's own metadata when this image has no live request
    // behind it (one rebuilt from a reopened conversation). The PNG carries the
    // AUTOMATIC1111 block that describes exactly how it was made, which makes
    // the file the record and saves the transcript carrying a second copy that
    // could disagree with it.
    if (gi.prompt.len == 0) if (gi.saved_path) |path| {
        if (loadFromFile(path)) return;
    };

    setBuf(&prompt_buf, gi.prompt);
    setBuf(&negative_buf, gi.req_negative);
    setFmt(&width_buf, "{d}", .{gi.req_width});
    setFmt(&height_buf, "{d}", .{gi.req_height});
    setFmt(&steps_buf, "{d}", .{gi.req_steps});
    setFmt(&cfg_buf, "{d:.1}", .{gi.req_cfg});
    setFmt(&seed_buf, "{d}", .{gi.req_seed});
    setBuf(&count_buf, "1");
    random_seed = false; // an exact seed is the whole point of reopening one
    seeded = true; // do not let `seed()` overwrite what we just put here
}

/// Fill the form from a saved PNG's `parameters` block. Returns false when the
/// file is gone or carries no metadata (an image saved by something that did
/// not write one), so the caller can fall back to whatever it has in memory.
fn loadFromFile(path: []const u8) bool {
    const bytes = std.Io.Dir.cwd().readFileAlloc(g_io, path, g_gpa, @enumFromInt(64 << 20)) catch return false;
    defer g_gpa.free(bytes);
    const text = tp.image.pngText(bytes, "parameters") orelse return false;
    const p = diffuser.parseA1111Params(text);
    if (p.prompt.len == 0) return false;

    setBuf(&prompt_buf, p.prompt);
    setBuf(&negative_buf, p.negative);
    if (p.width) |v| setFmt(&width_buf, "{d}", .{v});
    if (p.height) |v| setFmt(&height_buf, "{d}", .{v});
    if (p.steps) |v| setFmt(&steps_buf, "{d}", .{v});
    if (p.cfg) |v| setFmt(&cfg_buf, "{d:.1}", .{v});
    if (p.seed) |v| setFmt(&seed_buf, "{d}", .{v});
    setBuf(&count_buf, "1");
    random_seed = false;
    seeded = true;
    return true;
}

fn setBuf(buf: []u8, text: []const u8) void {
    @memset(buf, 0);
    const n = @min(text.len, buf.len - 1);
    @memcpy(buf[0..n], text[0..n]);
}

fn setFmt(buf: []u8, comptime fmt: []const u8, args: anytype) void {
    @memset(buf, 0);
    _ = std.fmt.bufPrint(buf, fmt, args) catch {};
}

/// Display size for an image: downscale so the longer side is `max`, never up.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    const mx: f32 = @floatFromInt(@max(w, h));
    const scale = if (mx > max) max / mx else 1.0;
    return .{ .w = @as(f32, @floatFromInt(w)) * scale, .h = @as(f32, @floatFromInt(h)) * scale };
}

/// The results grid, the engine's whole image list (chat + studio), newest
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
            renderCell(engine, imgs[n - 1 - (i + c)], i + c, cell);
        }
    }
}

fn renderCell(engine: *diffuser.Diffuser, gi: *GenImage, idx: usize, cell: f32) void {
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{ .id_extra = idx, .min_size_content = .{ .w = cell }, .margin = .{ .w = 12 } });
    defer b.deinit();

    switch (gi.get()) {
        .pending, .generating, .suspended => {
            const st_now = gi.get();
            const generating = st_now == .generating;
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
            const status = switch (st_now) {
                .suspended => std.fmt.bufPrint(&buf, "⏸ {d}/{d}", .{ done, total }) catch "⏸",
                .pending => "Queued…",
                else => std.fmt.bufPrint(&buf, "step {d}/{d}", .{ done, total }) catch "…",
            };
            // richLabel so the ⏸ (suspended) routes to the emoji face, the prose
            // font lacks media-control glyphs (see fonts.isEmoji).
            fonts.richLabel(@src(), status, .{});
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
        // Both terminal-but-unsuccessful states name themselves and offer a
        // retry. The cell is narrow, so the reason wraps in a textLayout rather
        // than being truncated into a label, an error the user cannot read is
        // the state this replaced.
        .failed => {
            var fbuf: [180]u8 = undefined;
            const msg = if (gi.failure()) |err|
                std.fmt.bufPrint(&fbuf, "failed: {s}", .{diffuser.failureText(err)}) catch "failed"
            else
                "failed";
            var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal, .background = false });
            fonts.addRich(tl, msg);
            tl.deinit();
            retryButton(engine, gi);
        },
        .canceled => {
            dvui.label(@src(), "canceled", .{}, .{});
            retryButton(engine, gi);
        },
    }
}

/// Re-queue this image in place (see `Diffuser.retry`).
fn retryButton(engine: *diffuser.Diffuser, gi: *GenImage) void {
    if (dvui.button(@src(), "Try again", .{}, .{ .margin = .{ .y = 2 } }))
        engine.retry(gi) catch |err| std.log.err("retry image: {t}", .{err});
}
