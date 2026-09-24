//! The image studio's CENTRE COLUMN: a canvas showing the renders in motion,
//! the parameter form in collapsible sections, and the prompt composer at the
//! foot. The prompt rail on the left and the queue rail on the right are
//! `app.zig`'s, the same two it builds for chat, so switching tabs does not
//! reflow the window.
//!
//! **The canvas shows work in motion, never a picture that is already made.**
//! With several hosts there are several renders at once, so it tiles them, and
//! clicking one watches it big until it stops. Only when nothing is rendering
//! does it fall back to the newest finished picture. That split is the answer
//! to a trap this had once: any click sent its image to the canvas, so opening
//! one from the Library parked it over live work and the only way back was to
//! press Generate. Finished pictures belong to the Library and the viewer; a
//! failed one belongs to its rail row, which is where Try again lives.
//!
//! The form sits UNDER the canvas on a narrow window and BESIDE it on a wide
//! one, because the canvas is bound by height: under the form it gets barely
//! half the column, which is enough for two tiles and not four.
//!
//! **The studio edits a per-image RECIPE, not the settings.** Every control here
//! writes a form buffer, and Generate snapshots the lot onto the `GenImage`
//! (`req_*` + `wire.RenderParams` + a LoRA set of its own). Settings holds
//! the DEFAULTS the form is seeded from, and the only things this module writes
//! back to config are the two pieces of view state -- which size control is
//! showing, and which sections are folded -- plus whatever "Save as defaults"
//! is explicitly asked to copy.
//!
//! That split is the answer to a bug this codebase has already had once: two
//! screens each seeding a form buffer from one config field, where whichever
//! was touched last silently won, on the framing chips. One writer
//! per value, and here the writer is the request.
//!
//! State is process-global (the dvui idiom, mirroring app.zig): only the
//! transient form fields live here; the engine owns every image.
const std = @import("std");
const dvui = @import("dvui");
const config = @import("shared").config;
const tp = @import("TensorPencil");
const pipeline_map = @import("shared").pipeline_map;
const save_image = @import("client").save_image;
const mirror = @import("client").mirror;
const wire = @import("serve").wire;
const model_spec = @import("shared").model_spec;
const catalog = @import("shared").catalog;
const models = @import("client").models;
const selection = @import("client").selection;
const framing = @import("shared").framing;
const clipboard = @import("clipboard.zig");
const fonts = @import("fonts.zig");
const hint = @import("hint.zig");
const style = @import("style.zig");
const bubbles = @import("bubbles.zig");

const C = style.C;
const F = style.F;
const R = style.R;

const Image = mirror.Image;
pub const ImageId = wire.ImageId;
/// How the studio reaches the host: every action is a request.
pub const PostFn = *const fn (wire.Request) void;

var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_wake: *const fn () void = undefined;
/// This frame's mirror and request sink, set by `render`.
var g_m: *mirror.Mirror = undefined;
var g_post: PostFn = undefined;

/// Where the canvas gets its pictures. Renders come from every host, not only
/// the one this view is configured against, so a job placed elsewhere still
/// shows up here.
pub const Images = struct {
    ctx: *anyopaque = undefined,
    /// Every render in motion right now, oldest first, filling `out` and
    /// returning the prefix used. This is what the canvas draws: a render is
    /// what is happening, and with several hosts there are several of them.
    live: *const fn (ctx: *anyopaque, out: []*const Image) []const *const Image = noLive,
    /// The newest finished render, for when nothing is in motion.
    newest: *const fn (ctx: *anyopaque) ?*const Image = noNewest,
    /// Which host made `id`, "" when there is only one.
    hostOf: *const fn (ctx: *anyopaque, id: ImageId) []const u8 = noHost,
};

fn noLive(_: *anyopaque, _: []*const Image) []const *const Image {
    return &.{};
}

fn noNewest(_: *anyopaque) ?*const Image {
    return null;
}

fn noHost(_: *anyopaque, _: ImageId) []const u8 {
    return "";
}

var g_images: Images = .{};

/// Set when the canvas is clicked; app.zig opens/refocuses the viewer.
pub var viewer_request: ?ImageId = null;

/// Set when Generate runs, for app.zig to record in the prompt library. Cleared
/// by the reader. Two fields rather than a callback so this module still needs
/// nothing from the store.
pub var recorded_prompt: ?struct { prompt: []const u8, negative: []const u8 } = null;

// Memoized inspection of the configured checkpoint, so the form can default to
// the architecture's own parameters (see `seed`). Live only after `setEnv`.
var g_spec: model_spec.Cache = undefined;
var g_spec_ready: bool = false;

/// Which render the canvas is showing. Null means "the newest finished one",
/// which is what a user wants right after pressing Generate. Cleared whenever
/// the image it names leaves the list.
var g_selected: ?ImageId = null;

// ------------------------------------------------------------- form buffers

var seeded: bool = false;
/// The architecture the form was last seeded for. A change means the user picked
/// a different KIND of model, and the architecture-dependent defaults re-seed.
var seeded_family: ?model_spec.Family = null;

var prompt_buf: [4096]u8 = @splat(0);
var negative_buf: [1024]u8 = @splat(0);
var width_buf: [12]u8 = @splat(0);
var height_buf: [12]u8 = @splat(0);
var mp_buf: [12]u8 = @splat(0);
var steps_buf: [8]u8 = @splat(0);
var cfg_buf: [8]u8 = @splat(0);
// Wide enough to type any u64 (20 digits + null): the queue count is
// deliberately uncapped, see `generate`.
var count_buf: [24]u8 = @splat(0);
var seed_buf: [24]u8 = @splat(0);
var random_seed: bool = true;
var ratio: framing.Ratio = .square;

/// The per-image recipe. Seeded from config, never written back to it.
var f_sampler: config.Sampler = .euler;
var f_scheduler: config.Scheduler = .default;
var f_syntax: config.PromptSyntax = .comfy;
var f_emphasis: config.Emphasis = .original;
var f_compat: config.Compat = .comfy;
var f_cond_on: bool = false;
var f_cond_curve: config.TextBuf(config.max_noise_curve) = .{};
var f_cond_shape: config.CondShape = .shared;
var f_cond_keep_norm: bool = false;
var f_cond_amount_buf: [16]u8 = @splat(0);
var f_cond_negative_buf: [16]u8 = @splat(0);
const max_steers = config.max_steers;
var f_steer_text: [max_steers]config.TextBuf(config.max_steer_text) = @splat(.{});
var f_steer_scale: [max_steers]f32 = @splat(0.25);
var f_steer_mode: [max_steers]config.SteerMode = @splat(.prompt);
var f_steer_n: usize = 0;
var f_act_dirs: config.TextBuf(config.max_path) = .{};
var f_act_scale: f32 = 0;
var f_act_op: config.ActOp = .add;
var f_act_curve: config.TextBuf(config.max_noise_curve) = .{};
var f_act_keep_norm: bool = true;
/// The "create a direction" modal: open flag and its form.
var g_act_modal = false;
var g_act_set_a: [1024]u8 = @splat(0);
var g_act_set_b: [1024]u8 = @splat(0);
var g_act_name: [64]u8 = @splat(0);
var g_act_size: [8]u8 = @splat(0);
var g_act_steps: [8]u8 = @splat(0);
var g_act_sent = false;
/// Whether each term's text field had keyboard focus on the last frame. Read by
/// `ui-probe --click`, which is how a click that never reaches a widget is told
/// apart from one that reaches it and does nothing.
pub var steer_editing: [max_steers]bool = @splat(false);

/// The per-image LoRA set, seeded from the family's configured list. Held flat
/// rather than as a list of `config.FamilyLora` so editing a row here cannot
/// reach the config table by accident.
const max_loras = config.max_family_loras;
var f_lora_path: [max_loras]config.PathBuf = @splat(.{});
var f_lora_strength: [max_loras]f32 = @splat(1.0);
var f_lora_on: [max_loras]bool = @splat(true);
var f_lora_n: usize = 0;

/// The form was filled from one specific image ("Open in Studio"), so the next
/// render adopts it instead of re-seeding. `seeded` alone does not hold it: the
/// family gate re-seeds on top whenever `seeded_family` does not match, and
/// filling the form from an image cannot know what family the view will draw
/// with.
var loaded_from_image: bool = false;

/// Drop the seeded form fields so the next render re-reads the config. Called
/// when the composer's framing chips change the default size, and after "Save as
/// defaults" so the two views agree.
pub fn reseed() void {
    seeded = false;
}

/// Give the studio the process allocator/clock/wake it needs to create images.
/// Called once from app startup.
pub fn setEnv(gpa: std.mem.Allocator, io: std.Io, wake: *const fn () void) void {
    g_gpa = gpa;
    g_io = io;
    g_wake = wake;
    g_spec = model_spec.Cache.init(gpa, io);
    g_spec_ready = true;
}

/// Nothing persistent to free (the engine owns the images); reset transient UI
/// state at exit.
pub fn deinit() void {
    seeded = false;
    seeded_family = null;
    loaded_from_image = false;
    viewer_request = null;
    recorded_prompt = null;
    g_selected = null;
    if (g_spec_ready) {
        g_spec.deinit();
        g_spec_ready = false;
    }
}

/// `ui-probe`'s stand-in for `currentFamily`. The probe's checkpoint paths are
/// canned and no such file exists, so the real answer is null and every
/// architecture-dependent row would be missing from the screenshot -- including
/// the LoRA section, which is most of what there is to look at.
///
/// A seam rather than a catalog fallback on purpose: the family must keep coming
/// from the FILE in the app, because the catalog is scanned on a worker thread
/// and is empty for the first frames of a cold start, which dropped a LoRA once
/// already.
pub var forced_family: ?model_spec.Family = null;

/// The architecture of the configured checkpoint, if it can be read. Memoized on
/// the path, so this is one header parse per model change, not per frame.
///
/// A file REPLACED under an unchanged path keeps the old answer here (only the
/// settings screen re-probes, on every open). That is deliberate: the worst
/// consequence is a stale CFG *suggestion* in a form field the user can edit,
/// which does not justify re-parsing a checkpoint header every frame.
fn currentFamily(cfg: *const config.Config) ?model_spec.Family {
    if (forced_family) |f| return f;
    if (!g_spec_ready) return null;
    const path = cfg.diffusion_model.opt() orelse return null;
    const info = g_spec.primary(path).info() orelse return null;
    return info.family;
}

fn seed(cfg: *const config.Config, fam: ?model_spec.Family) void {
    _ = std.fmt.bufPrintZ(&steps_buf, "{d}", .{cfg.steps}) catch {};
    // CFG 0 in the config means "this architecture's own value", which is the
    // only honest default for a number that cannot be shared: krea2 renders at
    // 1.0, which DISABLES guidance, and SD1.5 at 1.0 ignores the negative
    // prompt entirely and comes out washed.
    const guidance = if (cfg.cfg_scale > 0)
        cfg.cfg_scale
    else if (fam) |f| model_spec.traits(f).cfg else 1.0;
    _ = std.fmt.bufPrintZ(&cfg_buf, "{d:.1}", .{guidance}) catch {};

    // Size comes from the framing pair, which is what the chat composer sets
    // too; the exact fields are the same value spelled in pixels.
    ratio = cfg.framing_ratio;
    setFmt(&mp_buf, "{s}", .{framing.formatMp(&mp_scratch, cfg.framing_mp)});
    setFmt(&width_buf, "{d}", .{cfg.width});
    setFmt(&height_buf, "{d}", .{cfg.height});

    setBuf(&count_buf, "1");
    // Reseeding the form restores its defaults, and the seed's default is
    // "draw a fresh one". Without this the flag survives from a reopened
    // image, leaving the form saying FIXED over a seed of 0, which the host
    // reads as "pick one for me": the field would then be ignored on every
    // render while claiming to be honoured.
    setBuf(&seed_buf, "0");
    random_seed = true;

    f_sampler = cfg.sampler;
    f_scheduler = cfg.scheduler;
    f_syntax = cfg.prompt_syntax;
    f_emphasis = cfg.emphasis;
    f_compat = cfg.compat;
    f_cond_on = cfg.cond_noise;
    f_cond_curve = cfg.cond_noise_curve;
    f_cond_shape = cfg.cond_noise_shape;
    f_cond_keep_norm = cfg.cond_noise_keep_norm;
    setNum(&f_cond_amount_buf, cfg.cond_noise_amount);
    setNum(&f_cond_negative_buf, cfg.cond_noise_negative);
    f_act_dirs = cfg.act_dirs;
    f_act_scale = cfg.act_scale;
    f_act_op = cfg.act_op;
    f_act_curve = cfg.act_curve;
    f_act_keep_norm = cfg.act_keep_norm;
    f_steer_n = cfg.cond_steers.count;
    for (cfg.cond_steers.slice(), 0..) |t, i| {
        f_steer_text[i] = t.text;
        f_steer_scale[i] = t.scale;
        f_steer_mode[i] = t.mode;
    }
    seedLoras(cfg, fam);

    seeded = true;
    seeded_family = fam;
}

var mp_scratch: [16]u8 = undefined;

fn seedLoras(cfg: *const config.Config, fam: ?model_spec.Family) void {
    f_lora_n = 0;
    const f = fam orelse return;
    if (!f.supportsLora()) return;
    for (selection.lorasForFamily(cfg, f)) |l| {
        if (f_lora_n == max_loras) break;
        f_lora_path[f_lora_n].set(l.path.slice());
        f_lora_strength[f_lora_n] = l.strength;
        f_lora_on[f_lora_n] = l.enabled;
        f_lora_n += 1;
    }
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

fn setNum(buf: []u8, v: f32) void {
    @memset(buf, 0);
    _ = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{d}", .{v}) catch {};
}

/// Clamped at the pipeline's own ceiling so a typo cannot ask for something the
/// engine will silently clamp anyway.
fn condAmount() f32 {
    return std.math.clamp(parseFloat(&f_cond_amount_buf, 0.3), 0, tp.pipeline.Session.max_cond_sigma);
}

/// The named shape the live expression matches, or "custom".
fn condShapeName() []const u8 {
    for (tp.noise_curve.cond_shapes) |sh| {
        if (std.mem.eql(u8, sh.expr, f_cond_curve.slice())) return sh.name;
    }
    return "custom";
}

fn condNegative() f32 {
    return std.math.clamp(parseFloat(&f_cond_negative_buf, 0), 0, 1);
}

/// The dimensions the form is currently asking for, however it is spelling them.
fn formDims(cfg: *const config.Config) framing.Dims {
    if (cfg.studio_size_mode == .framing) {
        const mp = parseFloat(&mp_buf, cfg.framing_mp);
        return framing.dimsMp(ratio, mp);
    }
    return .{
        .w = pipeline_map.clampDim(parseNum(&width_buf, cfg.width)),
        .h = pipeline_map.clampDim(parseNum(&height_buf, cfg.height)),
    };
}

pub const Callbacks = struct {
    settings: *const fn () void,
    /// Copy the form's current recipe into the config defaults.
    save_defaults: *const fn () void,
    /// Stop one render. The canvas is where a render in flight is shown in the
    /// studio, so it is also where it can be called off.
    cancel: *const fn (ImageId) void,
};

/// This frame's callbacks, so the canvas tiles can reach `cancel` without every
/// helper taking the struct.
var g_cb: Callbacks = undefined;

// ------------------------------------------------------------------- render

/// Render the studio's centre column. `m` is the host that would take the next
/// render, which is what the model notice, the LoRA list and the queue count
/// describe; `images` reaches every host's renders. Every action goes out
/// through `post`. `ready` is false while the LLM is still being torn down
/// (Generate is disabled until the device is free).
pub fn render(cfg: *config.Config, m: *mirror.Mirror, u: *const models.Union, images: Images, post: PostFn, ready: bool, cb: Callbacks) void {
    g_m = m;
    g_images = images;
    g_post = post;
    g_cb = cb;
    const fam = currentFamily(cfg);
    // Re-seed when the user switches to a different ARCHITECTURE: the CFG
    // default and the LoRA set are both architecture-specific, and carrying
    // krea2's over to SD1.5 (or back) is a visibly wrong image, not a
    // preference.
    if (loaded_from_image) {
        loaded_from_image = false;
        seeded = true;
        seeded_family = fam;
    } else if (!seeded or !famEql(fam, seeded_family)) seed(cfg, fam);

    // No header here: the window's title bar already names this view, owns the
    // Chat/Studio switch and now carries the gear, so a second title row was two
    // things claiming to be the top of the screen.

    if (!m.state.diff_present) return renderNoModel(cb);

    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .min_size_content = .{ .w = 0 } });
    defer col.deinit();

    // The composer is pinned at the foot, the way the chat column's is; what sits
    // above it is either one scrolling stack or two side-by-side panes.
    //
    // The scroll area is given an explicit height rather than left to expand: it
    // reports its whole CONTENT height as its min size, so as a plain flex child
    // it pushes the composer off the bottom of the window (which is exactly what
    // the first build of this did). `g_composer_h` is last frame's measurement,
    // the same trick `app.g_input_h` uses for the chat composer; being one frame
    // late is invisible, and the two text entries only change height when
    // someone is typing into them.
    {
        const rect = col.data().contentRect();
        const band_h = @max(120, rect.h - g_composer_h);
        if (rect.w - form_w >= canvas_pane_min) {
            renderWide(cfg, u, fam, cb, rect.w, band_h);
        } else {
            renderStacked(cfg, u, fam, cb, band_h);
        }
    }

    renderComposer(cfg, ready);
}

/// Canvas over form, both in one scroll. The canvas gives up height so the form
/// stays reachable, which caps the picture at a little over half the column.
fn renderStacked(cfg: *config.Config, u: *const models.Union, fam: ?model_spec.Family, cb: Callbacks, band_h: f32) void {
    g_canvas_max = @min(canvas_max, @max(180, band_h * canvas_share));
    var sc = dvui.scrollArea(@src(), .{ .horizontal = .none }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = band_h },
        .max_size_content = .height(band_h),
        .background = false,
    });
    defer sc.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .x = 18, .y = 10, .w = 18, .h = 6 } });
    defer body.deinit();

    // Inside the scroll, so the column stays exactly two children and the
    // notice cannot squeeze the composer when it appears.
    renderModelNotice(cfg, fam);
    renderCanvas();
    renderForm(cfg, u, fam, cb);
}

/// Canvas beside form. The canvas now gets the column's whole height instead of
/// a share of it, which is what a grid of renders needs to draw them at a size
/// worth looking at; the form keeps a fixed readable width and takes the rest.
fn renderWide(cfg: *config.Config, u: *const models.Union, fam: ?model_spec.Family, cb: Callbacks, col_w: f32, band_h: f32) void {
    var band = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = band_h },
        .max_size_content = .height(band_h),
    });
    defer band.deinit();

    {
        const pane_w = col_w - form_w;
        var pane = dvui.box(@src(), .{ .dir = .vertical }, .{
            .min_size_content = .{ .w = pane_w, .h = band_h },
            .max_size_content = .{ .w = pane_w, .h = band_h },
            .padding = .{ .x = 18, .y = 10, .w = 8, .h = 6 },
        });
        defer pane.deinit();
        g_canvas_max = @min(canvas_max_wide, @max(180, band_h - 70));
        renderModelNotice(cfg, fam);
        renderCanvas();
    }

    var sc = dvui.scrollArea(@src(), .{ .horizontal = .none }, .{
        .min_size_content = .{ .w = form_w, .h = band_h },
        .max_size_content = .{ .w = form_w, .h = band_h },
        .background = false,
    });
    defer sc.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .x = 8, .y = 10, .w = 18, .h = 6 } });
    defer body.deinit();
    renderForm(cfg, u, fam, cb);
}

fn renderForm(cfg: *config.Config, u: *const models.Union, fam: ?model_spec.Family, cb: Callbacks) void {
    renderSizeSection(cfg);
    renderSamplingSection(cfg);
    renderLoraSection(cfg, &u.cat, fam);
    renderAdvancedSection(cfg, cb);
}

/// Last frame's composer height, so this frame can reserve it. Seeded with a
/// plausible value for the very first frame.
var g_composer_h: f32 = 168;
var prompt_h: f32 = 42;
var negative_h: f32 = 20;
/// Everything in the composer that is NOT the two entries: the block's padding,
/// the frame's border and padding, and the button row with its margin.
const composer_chrome: f32 = 86;

fn renderNoModel(cb: Callbacks) void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5, .padding = dvui.Rect.all(24) });
    defer col.deinit();
    var tl = dvui.textLayout(@src(), .{}, .{ .gravity_x = 0.5, .background = false, .max_size_content = .width(style.Layout.prose_max) });
    fonts.addStyled(tl, "No diffusion model is set.\n\nOpen Settings and choose a diffusion model to generate images here. " ++
        "Most checkpoints bundle everything they need; the text encoder and VAE fields are only for supplying or replacing a piece.", .{}, .{
        .font = F.prose,
        .color_text = C.text,
    });
    tl.deinit();
    var actions = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 14 } });
    const go = bubbles.primaryButton(@src(), "Open Settings", true);
    actions.deinit();
    if (go) cb.settings();
}

/// Tell the user, BEFORE they hit Generate, when this model set cannot produce an
/// image, and afterwards, why the last load failed.
///
/// A model set can fail to load for reasons the studio has no other way to show:
/// a missing component the checkpoint does not bundle, an override holding the
/// wrong architecture's weights, or a backend without kernels for it. Without
/// this they surface as a bare "failed" on the image, with the reason only in the
/// terminal.
fn renderModelNotice(cfg: *const config.Config, fam: ?model_spec.Family) void {
    var buf: [320]u8 = undefined;
    const text: []const u8 = blk: {
        if (g_m.state.diff_load_err.len > 0) break :blk std.fmt.bufPrint(
            &buf,
            "The diffusion model failed to load: {s}. Check the model paths and backend in Settings.",
            .{g_m.state.diff_load_err},
        ) catch "The diffusion model failed to load.";
        const f = fam orelse return; // unreadable/unknown: Settings says why
        const t = model_spec.traits(f);
        const want = pipeline_map.toPipelineBackend(cfg.diff_backend);
        if (!t.supports(want)) break :blk std.fmt.bufPrint(
            &buf,
            "{s} has no kernels for the {t} backend. Generation will fail to load — change it in Settings.",
            .{ t.label, cfg.diff_backend },
        ) catch "This architecture has no kernels for the selected backend.";
        return;
    };

    var tl = dvui.textLayout(@src(), .{}, .{
        .expand = .horizontal,
        .background = false,
        .padding = .{ .x = 18, .y = 6, .w = 18, .h = 2 },
        .color_text = C.danger,
    });
    defer tl.deinit();
    fonts.addStyled(tl, text, .{}, .{ .font = F.ui, .color_text = C.danger });
}

// ------------------------------------------------------------------- canvas

/// Longest side the canvas draws ONE image at, and the share of the column it
/// may take when the form sits underneath. A studio wants a picture big, but
/// the form has to stay reachable without a scroll marathon, so the picture
/// yields on a short window rather than the controls doing it.
const canvas_max: f32 = 520;
const canvas_share: f32 = 0.55;

/// Longest side when the form sits BESIDE the canvas: nothing is under it, so
/// it can have the column's height rather than a share of it.
const canvas_max_wide: f32 = 760;

/// The form's width when it sits beside the canvas. Fixed, because a form has a
/// readable width and a picture does not: everything past this goes to the
/// canvas.
const form_w: f32 = 432;

/// Narrowest canvas pane worth splitting the column for. Under this the two
/// panes are both too small and the stacked layout wins.
const canvas_pane_min: f32 = 440;

/// Gap between tiles.
const tile_gap: f32 = 10;

/// Smallest tile worth drawing. Under this a picture stops reading as one and
/// the rail's 52px row would do as well, so a canvas that cannot reach it draws
/// fewer tiles and counts the rest in its caption.
const min_tile: f32 = 150;

/// The canvas's height budget for this frame, from the room the column has.
var g_canvas_max: f32 = canvas_max;

/// Longest side the canvas actually drew an image at last frame, which is what
/// previews want to be fetched at: a grid of four wants quarter-size frames,
/// not four full-size ones it has nowhere to put.
var g_drawn_edge: f32 = canvas_max;

/// Longest side the canvas draws a live preview at, for whoever fetches them.
pub fn canvasMaxEdge() u32 {
    return @intFromFloat(@max(64, g_drawn_edge));
}

/// Watch this one render big instead of tiling every one in flight. Dropped as
/// soon as that render stops moving: the canvas is where work in motion is
/// shown, and it must not get stuck on something that has finished.
pub fn select(id: ImageId) void {
    g_selected = id;
}

/// `w` x `h` scaled to sit exactly inside a `box_w` x `box_h` box, growing as
/// well as shrinking. A preview arrives at a fraction of the final size, so a
/// fit that only ever shrank drew every tile at whatever the sampler happened
/// to decode at and left the canvas's room unused.
fn fitBox(w: usize, h: usize, box_w: f32, box_h: f32) dvui.Size {
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    if (fw <= 0 or fh <= 0) return .{ .w = 0, .h = 0 };
    const scale = @min(box_w / fw, box_h / fh);
    return .{ .w = fw * scale, .h = fh * scale };
}

/// `fitBox` into a square, for the parts of the canvas with one budget rather
/// than two.
fn fitSize(w: usize, h: usize, max: f32) dvui.Size {
    return fitBox(w, h, max, max);
}

/// How the canvas divides itself: `cols` x `rows` cells `cell` px on a side,
/// showing `shown` of the renders in flight.
/// Cells are rectangular, not square: the pane is wider than it is tall once
/// the form moves beside it, and a square cell would leave that width unused
/// for every landscape render.
const Grid = struct {
    cols: usize,
    rows: usize,
    cw: f32,
    ch: f32,
    shown: usize,

    /// What the smallest side of a cell is, which is what decides whether this
    /// shape still draws pictures or thumbnails.
    fn short(self: Grid) f32 {
        return @min(self.cw, self.ch);
    }

    /// Width of a full row, so a row with an empty cell in it still lines up
    /// with the ones above.
    fn rowWidth(self: Grid) f32 {
        return self.cw * @as(f32, @floatFromInt(self.cols)) + tile_gap * @as(f32, @floatFromInt(self.cols - 1));
    }
};

/// Room under each tile for its progress bar and caption, which the picture
/// does not get to use.
const tile_chrome: f32 = 46;

fn cellFor(cols: usize, rows: usize, w: f32, h: f32, n: usize) Grid {
    const fc: f32 = @floatFromInt(cols);
    const fr: f32 = @floatFromInt(rows);
    const cw = (w - tile_gap * (fc - 1)) / fc;
    const ch = (h - (tile_gap + tile_chrome) * fr) / fr;
    return .{
        .cols = cols,
        .rows = rows,
        .cw = @max(64, cw),
        .ch = @max(48, ch),
        .shown = @min(n, cols * rows),
    };
}

/// Grow the grid while there is more to show and the cells stay big enough to
/// be pictures. One big picture beats four thumbnails, which is what the rail
/// is for.
fn gridFor(n: usize, w: f32, h: f32) Grid {
    var best = cellFor(1, 1, w, h, n);
    const shapes = [_][2]usize{ .{ 2, 1 }, .{ 2, 2 } };
    for (shapes) |s| {
        if (best.shown >= n) break;
        const g = cellFor(s[0], s[1], w, h, n);
        if (g.short() < min_tile) break;
        best = g;
    }
    return best;
}

/// Where `id` sits among the renders in motion, or null when it is not one of
/// them. A pin only ever names work in motion: once that render stops the thing
/// worth looking at has moved on, and a canvas that stayed on it is the trap
/// that made a finished picture park over live work with no way back.
fn liveIndex(live: []const *const Image, id: ImageId) ?usize {
    for (live, 0..) |im, i| if (im.info.id == id) return i;
    return null;
}

fn renderCanvas() void {
    var wrap = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .margin = .{ .h = 10 } });
    defer wrap.deinit();
    const avail_w = wrap.data().contentRect().w;

    var buf: [32]*const Image = undefined;
    const live = g_images.live(g_images.ctx, &buf);

    if (g_selected) |id| {
        if (liveIndex(live, id)) |i| {
            if (live.len > 1) return renderPinned(live[i], live.len, avail_w);
        } else g_selected = null;
    }

    if (live.len == 0) return renderIdle();
    if (live.len == 1) return renderOne(live[0], avail_w);
    renderGrid(live, avail_w);
}

/// One render in flight, with the whole pane to itself.
fn renderOne(im: *const Image, avail_w: f32) void {
    g_drawn_edge = @max(avail_w, g_canvas_max);
    renderLive(im, avail_w, g_canvas_max, true);
}

/// One render in flight drawn big, with the way back to the others. Without
/// that way back a pin is a trap: the canvas stays on one picture and nothing
/// says the rest are still going.
fn renderPinned(im: *const Image, n_live: usize, avail_w: f32) void {
    renderOne(im, avail_w);

    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 2 } });
    defer row.deinit();
    var buf: [48]u8 = undefined;
    const label = std.fmt.bufPrint(&buf, "Show all {d} rendering", .{n_live}) catch "Show all";
    if (style.chip(@src(), label, .{ .font = F.mono_row })) g_selected = null;
}

/// Every render in flight, tiled. Clicking one pins it.
fn renderGrid(live: []const *const Image, avail_w: f32) void {
    const g = gridFor(live.len, avail_w, g_canvas_max);
    g_drawn_edge = @max(g.cw, g.ch);
    const row_w = g.rowWidth();

    var i: usize = 0;
    while (i < g.shown) : (i += g.cols) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .id_extra = i,
            .gravity_x = 0.5,
            // Every row is a full row wide, so the last one lines its cells up
            // under the ones above rather than centring an orphan.
            .min_size_content = .{ .w = row_w },
            .max_size_content = .width(row_w),
            .margin = .{ .h = if (i + g.cols < g.shown) tile_gap else 0 },
        });
        defer row.deinit();

        var c: usize = 0;
        while (c < g.cols and i + c < g.shown) : (c += 1) {
            var cellbox = dvui.box(@src(), .{ .dir = .vertical }, .{
                .id_extra = c,
                .min_size_content = .{ .w = g.cw },
                .max_size_content = .width(g.cw),
                .margin = .{ .w = if (c + 1 < g.cols) tile_gap else 0 },
            });
            renderLive(live[i + c], g.cw, g.ch, false);
            const clicked = dvui.clicked(cellbox.data(), .{});
            cellbox.deinit();
            if (clicked) g_selected = live[i + c].info.id;
        }
    }

    // What did not fit is a CHIP, not a label: the studio rail lists no running
    // job, so if this were only a count the renders past the grid would be
    // reachable from nowhere. Clicking it watches the first of them.
    if (g.shown < live.len) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .y = 4 } });
        defer row.deinit();
        var buf: [72]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "+{d} more rendering", .{live.len - g.shown}) catch "+ more";
        if (style.chip(@src(), text, .{ .font = F.mono_row })) g_selected = live[g.shown].info.id;
    }
}

/// Nothing is rendering: the newest finished picture, or a well saying this is
/// where pictures go.
fn renderIdle() void {
    const im = g_images.newest(g_images.ctx) orelse {
        // A dashed well rather than a line of text: the canvas is the biggest
        // thing on the screen and an empty one should still read as the place a
        // picture goes.
        var well = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = 220 },
            .background = true,
            .color_fill = C.sunken,
            .border = style.Edge.all,
            .color_border = style.hairline,
            .corner_radius = R.card,
        });
        defer well.deinit();
        dvui.labelNoFmt(@src(), "Your renders appear here.", .{}, .{
            .font = F.ui,
            .color_text = C.text_ghost,
            .gravity_x = 0.5,
            .gravity_y = 0.5,
        });
        return;
    };
    g_drawn_edge = g_canvas_max;
    if (im.receiving()) renderReceiving(im) else renderDone(im);
}

/// Finished, with the picture still coming over the wire. On a remote host that
/// is seconds, and a blank canvas in the meantime reads as a failure.
fn renderReceiving(im: *const Image) void {
    dvui.refresh(null, @src(), null);
    if (im.thumb()) |t| {
        const sz = fitSize(t.w, t.h, g_canvas_max);
        _ = dvui.image(@src(), .{
            .source = .{ .pixels = .{ .rgba = t.px, .width = t.w, .height = t.h } },
            .shrink = .ratio,
        }, .{ .gravity_x = 0.5, .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = R.card });
    }
    metaLine(metaText(im, "receiving the full picture…"), null, g_canvas_max);
}

/// The canvas's caption, with the host that made it when there is more than one.
fn metaText(im: *const Image, what: []const u8) []const u8 {
    const host = g_images.hostOf(g_images.ctx, im.info.id);
    if (host.len == 0) return what;
    return std.fmt.bufPrint(&meta_buf, "{s} · on {s}", .{ what, host }) catch what;
}

var meta_buf: [200]u8 = undefined;

/// One render in motion, its picture fitted into a `bw` x `bh` box. `big` is
/// the canvas showing a single render; a tile holds its slot open even with
/// nothing in it so the row does not jump when the first preview lands.
fn renderLive(im: *const Image, bw: f32, bh: f32, big: bool) void {
    if (im.status() == .generating) dvui.refresh(null, @src(), null);
    const done = im.info.step;
    const total = im.info.total;

    // A tile's slot is the whole cell whatever shape the picture is: without it
    // a landscape render and a portrait one in the same row put their bars and
    // captions at different heights and the grid reads as broken. The single
    // render has no row to line up with, so it takes only what it needs.
    const blank = im.preview == null or im.preview_w == 0 or im.preview_h == 0;
    var slot = dvui.box(@src(), .{}, .{
        .min_size_content = if (big) .{} else .{ .w = bw, .h = bh },
        .max_size_content = .size(.{ .w = bw, .h = bh }),
        .gravity_x = 0.5,
        // Filled only while there is nothing to put in it: a render whose first
        // preview has not arrived still holds its place in the row.
        .background = !big and blank,
        .color_fill = C.sunken,
        .corner_radius = R.card,
    });
    // A fetched preview is a new buffer each time, so dvui's pointer-keyed
    // texture cache re-uploads exactly once per fetched frame.
    if (im.preview) |pv| {
        const pw = im.preview_w;
        const ph = im.preview_h;
        if (pw > 0 and ph > 0) {
            const sz = fitBox(pw, ph, bw, bh);
            _ = dvui.image(@src(), .{
                .source = .{ .pixels = .{ .rgba = pv, .width = pw, .height = ph } },
                .shrink = .ratio,
            }, .{
                .gravity_x = 0.5,
                .gravity_y = 0.5,
                .min_size_content = sz,
                .max_size_content = .size(sz),
                .corner_radius = R.card,
            });
        }
    }
    slot.deinit();

    // Amber, because this is the machine working. The bar is the only amber
    // thing on the screen while it runs.
    const pct: f32 = if (total > 0) @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total)) else 0;
    {
        var bar = dvui.box(@src(), .{}, .{
            .expand = .horizontal,
            .max_size_content = .width(bw),
            .gravity_x = 0.5,
        });
        defer bar.deinit();
        style.progressTrack(@src(), pct, 4);
    }

    var buf: [64]u8 = undefined;
    const status = switch (im.status()) {
        .suspended => std.fmt.bufPrint(&buf, "paused · step {d}/{d}", .{ done, total }) catch "paused",
        .pending => "queued",
        else => std.fmt.bufPrint(&buf, "step {d} / {d}", .{ done, total }) catch "…",
    };
    liveCaption(im, metaText(im, status), bw);
}

/// A render in motion's caption: what it is doing, and the one thing that can
/// be done to it. The queue rail lists no running job in the studio, so this is
/// the only place to call one off.
fn liveCaption(im: *const Image, text: []const u8, edge: f32) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .max_size_content = .width(edge),
        .gravity_x = 0.5,
        .margin = .{ .y = 6 },
    });
    defer row.deinit();

    var tbuf: [200]u8 = undefined;
    fonts.richLine(@src(), style.ellipsize(&tbuf, text, F.mono, @max(40, edge - 28)), .{
        .font = F.mono,
        .color_text = C.text_dim,
        .padding = .{},
        .margin = .{},
        .gravity_y = 0.5,
    });
    {
        var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
        sp.deinit();
    }
    var wd: dvui.WidgetData = undefined;
    if (dvui.buttonIcon(@src(), "cancel", dvui.entypo.cross, .{}, .{}, .{
        .min_size_content = .{ .w = 14, .h = 14 },
        .gravity_y = 0.5,
        .color_text = C.text_ghost,
        .background = false,
        .corner_radius = R.chip,
        .padding = dvui.Rect.all(2),
        .data_out = &wd,
    })) g_cb.cancel(im.info.id);
    hint.hover(@src(), &wd, "Stop this render");
}

fn renderDone(im: *const Image) void {
    const rgba = im.pixels orelse return;
    const w: usize = im.info.width;
    const h: usize = im.info.height;
    const sz = fitSize(w, h, g_canvas_max);
    var ib = dvui.box(@src(), .{}, .{ .gravity_x = 0.5 });
    _ = dvui.image(@src(), .{
        .source = .{ .pixels = .{ .rgba = rgba, .width = @intCast(w), .height = @intCast(h) } },
        .shrink = .ratio,
    }, .{ .min_size_content = sz, .max_size_content = .size(sz), .corner_radius = R.card });
    const clicked = dvui.clicked(ib.data(), .{});
    ib.deinit();
    if (clicked) viewer_request = im.info.id;

    var buf: [96]u8 = undefined;
    const what = std.fmt.bufPrint(&buf, "{d}×{d} · seed {d}", .{ w, h, im.info.req_seed }) catch "";
    metaLine(metaText(im, what), im, g_canvas_max);
}

/// The line under a finished picture: what it is, and what can be done with it.
fn metaLine(text: []const u8, im: ?*const Image, edge: f32) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .max_size_content = .width(edge),
        .gravity_x = 0.5,
        .margin = .{ .y = 6 },
    });
    defer row.deinit();
    fonts.richLine(@src(), text, .{
        .font = F.mono,
        .color_text = C.text_dim,
        .padding = .{},
        .margin = .{},
        .gravity_y = 0.5,
    });
    const g = im orelse return;
    const px = g.pixels orelse return;
    {
        var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
        sp.deinit();
    }
    var wd: dvui.WidgetData = undefined;
    if (dvui.buttonIcon(@src(), "copy", dvui.entypo.clipboard, .{}, .{}, .{
        .min_size_content = .{ .w = 16, .h = 16 },
        .gravity_y = 0.5,
        .data_out = &wd,
    })) clipboard.copyImage(px, g.info.width, g.info.height);
    hint.hover(@src(), &wd, "Copy image to clipboard");
}

// ----------------------------------------------------------------- sections

/// A labelled form row: a fixed-width label and whatever control follows.
fn rowBegin(src: std.builtin.SourceLocation, id: usize, label: []const u8) *dvui.BoxWidget {
    const row = dvui.box(src, .{ .dir = .horizontal }, .{ .id_extra = id, .expand = .horizontal, .margin = .{ .h = 5 } });
    dvui.labelNoFmt(@src(), label, .{}, .{
        .font = F.ui,
        .color_text = C.text_dim,
        .padding = .{},
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 118 },
        .max_size_content = .width(118),
    });
    return row;
}

/// A dropdown over an enum whose members carry a `label()`. Returns true when
/// the pick changed.
fn enumChip(src: std.builtin.SourceLocation, comptime T: type, cur: *T, id: usize) bool {
    const fields = @typeInfo(T).@"enum".fields;
    var labels: [fields.len][]const u8 = undefined;
    inline for (fields, 0..) |f, i| labels[i] = (@as(T, @enumFromInt(f.value))).label();
    var sel: usize = blk: {
        inline for (fields, 0..) |f, i| if (@intFromEnum(cur.*) == f.value) break :blk i;
        break :blk 0;
    };
    if (!style.chipDropdown(src, &labels, &sel, .{ .id_extra = id })) return false;
    inline for (fields, 0..) |f, i| if (i == sel) {
        cur.* = @enumFromInt(f.value);
    };
    return true;
}

fn numChip(src: std.builtin.SourceLocation, buf: []u8, suffix: []const u8, width: f32, id: usize) void {
    _ = style.chipInput(src, buf, suffix, width, .{ .id_extra = id });
}

fn renderSizeSection(cfg: *config.Config) void {
    var sum_buf: [64]u8 = undefined;
    const d = formDims(cfg);
    var sec = style.collapsibleBegin(@src(), "SIZE & SEED", &cfg.studio_open_size, .{
        .summary = std.fmt.bufPrint(&sum_buf, "{d}×{d} · {s}", .{
            d.w, d.h, if (random_seed) "random seed" else "fixed seed",
        }) catch "",
    });
    defer style.collapsibleEnd(&sec);
    if (!sec.open()) return;

    // The toggle names two spellings of ONE value, so switching converts rather
    // than resets: pick pixels after setting 3:2 at 1 MP and the fields already
    // hold 1216x832, not last week's numbers.
    {
        var row = rowBegin(@src(), 0, "Size as");
        defer row.deinit();
        if (style.segmented(@src(), &.{ "ratio + MP", "width × height" }, @intFromEnum(cfg.studio_size_mode))) |i| {
            const want: config.StudioSizeMode = @enumFromInt(i);
            if (want != cfg.studio_size_mode) {
                switch (want) {
                    .exact => {
                        setFmt(&width_buf, "{d}", .{d.w});
                        setFmt(&height_buf, "{d}", .{d.h});
                    },
                    .framing => {
                        ratio = framing.nearestRatio(d.w, d.h);
                        setFmt(&mp_buf, "{s}", .{framing.formatMp(&mp_scratch, framing.megapixelsOf(d.w, d.h))});
                    },
                }
                cfg.studio_size_mode = want;
            }
        }
    }

    switch (cfg.studio_size_mode) {
        .framing => {
            var row = rowBegin(@src(), 1, "Framing");
            defer row.deinit();
            _ = enumChip(@src(), framing.Ratio, &ratio, 0);
            numChip(@src(), &mp_buf, "MP", 42, 1);
            var px: [32]u8 = undefined;
            dvui.labelNoFmt(@src(), std.fmt.bufPrint(&px, "{d} × {d}", .{ d.w, d.h }) catch "", .{}, .{
                .font = F.mono,
                .color_text = C.text_faint,
                .padding = .{},
                .margin = .{ .x = 10 },
                .gravity_y = 0.5,
            });
        },
        .exact => {
            var row = rowBegin(@src(), 2, "Pixels");
            defer row.deinit();
            numChip(@src(), &width_buf, "w", 52, 0);
            numChip(@src(), &height_buf, "h", 52, 1);
        },
    }

    {
        var row = rowBegin(@src(), 3, "Batch");
        defer row.deinit();
        numChip(@src(), &count_buf, "images", 42, 0);
    }
    {
        var row = rowBegin(@src(), 4, "Seed");
        defer row.deinit();
        if (dvui.checkbox(@src(), &random_seed, "Random", .{ .gravity_y = 0.5 })) {
            // Asking for a fixed seed with nothing in the field would send 0,
            // which the host reads as "draw one" -- fixed in the form and
            // random in fact. So switching to fixed fills in a real seed to
            // edit, the way the last render's number would be.
            if (!random_seed and parseU64(&seed_buf) == 0) setFmt(&seed_buf, "{d}", .{freshSeed()});
        }
        // A fixed seed is the whole point of the field, so it only appears once
        // one is being asked for.
        if (!random_seed) numChip(@src(), &seed_buf, "", 150, 0);
    }
}

fn renderSamplingSection(cfg: *config.Config) void {
    var sum_buf: [96]u8 = undefined;
    var sec = style.collapsibleBegin(@src(), "SAMPLING", &cfg.studio_open_sampling, .{
        .summary = std.fmt.bufPrint(&sum_buf, "{s} · {s} · {s} steps · cfg {s}", .{
            f_sampler.label(),
            f_scheduler.label(),
            std.mem.sliceTo(&steps_buf, 0),
            std.mem.sliceTo(&cfg_buf, 0),
        }) catch "",
    });
    defer style.collapsibleEnd(&sec);
    if (!sec.open()) return;

    {
        var row = rowBegin(@src(), 0, "Sampler");
        defer row.deinit();
        _ = enumChip(@src(), config.Sampler, &f_sampler, 0);
    }
    {
        var row = rowBegin(@src(), 1, "Scheduler");
        defer row.deinit();
        _ = enumChip(@src(), config.Scheduler, &f_scheduler, 0);
    }
    {
        var row = rowBegin(@src(), 2, "Steps");
        defer row.deinit();
        numChip(@src(), &steps_buf, "", 42, 0);
    }
    {
        var row = rowBegin(@src(), 3, "Guidance");
        defer row.deinit();
        numChip(@src(), &cfg_buf, "cfg", 42, 0);
    }
}

/// The per-image LoRA set.
///
/// Hidden outright when the architecture has no sidecar path in this build
/// (`Family.supportsLora`, the same list `Session.attachLoras` refuses by), so a
/// user is never offered a control whose only outcome is a failed load.
fn renderLoraSection(cfg: *config.Config, cat: *const catalog.Catalog, fam: ?model_spec.Family) void {
    const f = fam orelse return;
    if (!f.supportsLora()) return;

    const arena = dvui.currentWindow().arena();
    const cands = cat.lorasFor(arena, f) catch return;

    var active: usize = 0;
    for (0..f_lora_n) |i| {
        if (f_lora_on[i]) active += 1;
    }
    if (cands.len == 0 and f_lora_n == 0) return;

    var sum_buf: [48]u8 = undefined;
    var sec = style.collapsibleBegin(@src(), "LORAS", &cfg.studio_open_loras, .{
        .summary = if (active == 0)
            "none"
        else
            std.fmt.bufPrint(&sum_buf, "{d} active", .{active}) catch "",
    });
    defer style.collapsibleEnd(&sec);
    if (!sec.open()) return;

    help(0, "Applied beside the model's weights, never merged in, so strength stays a dial. " ++
        "These ride on the image you queue: change them and the next render picks them up " ++
        "with no model reload.");

    var remove: ?usize = null;
    for (0..f_lora_n) |i| {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = i, .expand = .horizontal, .margin = .{ .h = 4 } });
        defer row.deinit();

        _ = dvui.checkbox(@src(), &f_lora_on[i], null, .{ .gravity_y = 0.5 });
        // The file name, not the path: it sits next to a slider in a fixed
        // width. Amber when the catalog has never seen the file, which is the
        // difference between "off" and "gone".
        const path = f_lora_path[i].slice();
        const known = cat.resolve(path) != null;
        var nb: [96]u8 = undefined;
        dvui.labelNoFmt(@src(), style.ellipsize(&nb, cat.refName(path), F.ui, 200), .{}, .{
            .font = F.ui,
            .color_text = if (known) C.text else C.amber,
            .padding = .{},
            .margin = .{ .x = 4, .w = 8 },
            .gravity_y = 0.5,
            .min_size_content = .{ .w = 200 },
            .max_size_content = .width(200),
        });
        _ = dvui.sliderEntry(@src(), "{d:0.2}", .{
            .value = &f_lora_strength[i],
            .min = -1.0,
            .max = 2.0,
            .interval = 0.05,
        }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 } });
        if (bubbles.secondaryButton(@src(), i, "Remove", true)) remove = i;
    }
    if (remove) |i| {
        // Shift rather than swap: order is what `applyStrengths`' positional
        // indexes mean, and a reorder is a stack swap the user did not ask for.
        var j = i;
        while (j + 1 < f_lora_n) : (j += 1) {
            f_lora_path[j] = f_lora_path[j + 1];
            f_lora_strength[j] = f_lora_strength[j + 1];
            f_lora_on[j] = f_lora_on[j + 1];
        }
        f_lora_n -= 1;
    }

    if (f_lora_n >= max_loras) {
        help(1, "The LoRA list is full; remove one to add another.");
        return;
    }
    // Only what is not already on, so the menu never offers a duplicate.
    var labels: std.ArrayList([]const u8) = .empty;
    var paths: std.ArrayList([]const u8) = .empty;
    for (cands) |ci| {
        const e = &cat.entries[ci];
        if (hasLora(e.path)) continue;
        const info = e.lora.?.info;
        labels.append(arena, std.fmt.allocPrint(arena, "{s}  ({d} linears, rank {d})", .{ e.stem(), info.targets, info.rank }) catch e.stem()) catch return;
        paths.append(arena, e.path) catch return;
    }
    if (labels.items.len == 0) return;

    var row = rowBegin(@src(), 90, "Add");
    defer row.deinit();
    var pick: usize = 0;
    var entries: std.ArrayList([]const u8) = .empty;
    entries.append(arena, "choose…") catch return;
    entries.appendSlice(arena, labels.items) catch return;
    if (style.chipDropdown(@src(), entries.items, &pick, .{})) {
        if (pick > 0 and f_lora_n < max_loras) {
            f_lora_path[f_lora_n].set(paths.items[pick - 1]);
            f_lora_strength[f_lora_n] = 1.0;
            f_lora_on[f_lora_n] = true;
            f_lora_n += 1;
        }
    }
}

fn hasLora(path: []const u8) bool {
    for (0..f_lora_n) |i| {
        if (std.mem.eql(u8, f_lora_path[i].slice(), path)) return true;
    }
    return false;
}


fn renderAdvancedSection(cfg: *config.Config, cb: Callbacks) void {
    var sum_buf: [96]u8 = undefined;
    // Conditioning noise named in the FOLDED summary: it changes every render and
    // is otherwise invisible the moment the section is collapsed.
    const summary = if (f_cond_on and f_cond_curve.slice().len > 0)
        std.fmt.bufPrint(&sum_buf, "{s} · noise {s} {d}", .{
            f_syntax.label(), condShapeName(), condAmount(),
        }) catch ""
    else
        std.fmt.bufPrint(&sum_buf, "{s} · {s}", .{ f_syntax.label(), f_compat.label() }) catch "";
    var sec = style.collapsibleBegin(@src(), "ADVANCED", &cfg.studio_open_advanced, .{
        .summary = summary,
    });
    defer style.collapsibleEnd(&sec);
    if (!sec.open()) return;

    {
        var row = rowBegin(@src(), 0, "Prompt syntax");
        defer row.deinit();
        _ = enumChip(@src(), config.PromptSyntax, &f_syntax, 0);
    }
    // Only meaningful under the A1111 dialect, so it is hidden rather than
    // greyed: a visible control that does nothing invites the reading that the
    // ComfyUI path has a weighting choice too, and it does not.
    if (f_syntax == .a1111) {
        var row = rowBegin(@src(), 1, "Emphasis");
        defer row.deinit();
        _ = enumChip(@src(), config.Emphasis, &f_emphasis, 0);
    }
    // Deliberately NOT hidden behind the dialect: the two axes are independent,
    // and someone reproducing an A1111 image needs this even when their prompt
    // has no emphasis syntax in it at all.
    {
        var row = rowBegin(@src(), 2, "Sampling compat");
        defer row.deinit();
        _ = enumChip(@src(), config.Compat, &f_compat, 0);
    }

    renderCondNoise();
    renderCondSteer();
    renderActSteer();
    // Drawn from the section so it sits inside the same frame; a floating window
    // is positioned by dvui, not by where it is called.
    if (g_act_modal) renderActModal();

    help(2, "Everything above is per image: it is stamped on each render you queue, " ++
        "and Settings keeps the defaults new renders start from. Live preview, the " ++
        "VAE decode path and the model set itself are Settings' to own.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 4, .h = 4 } });
        defer row.deinit();
        if (bubbles.secondaryButton(@src(), 0, "Save as defaults", true)) cb.save_defaults();
        if (bubbles.secondaryButton(@src(), 1, "Open Settings", true)) cb.settings();
    }
}

/// Conditioning noise: a seeded perturbation of the text conditioning, shaped by a
/// curve over the encoder taps (`t`) and the prompt's tokens (`l`/`n`).
///
/// Rows appear only when the toggle is on. A curve that does not parse leaves the
/// value showing but renders clean, which is the same reading the LLM's curve field
/// takes: someone halfway through typing has not asked for anything yet.
fn renderCondNoise() void {
    {
        var row = rowBegin(@src(), 3, "Cond noise");
        defer row.deinit();
        _ = dvui.checkbox(@src(), &f_cond_on, null, .{ .gravity_y = 0.5 });
    }
    if (!f_cond_on) return;

    {
        var row = rowBegin(@src(), 4, "Taps");
        defer row.deinit();
        _ = enumChip(@src(), config.CondShape, &f_cond_shape, 0);
    }
    {
        var row = rowBegin(@src(), 5, "Amount");
        defer row.deinit();
        numChip(@src(), &f_cond_amount_buf, "", 70, 0);
    }
    // The named shapes come straight from `noise_curve.cond_shapes`, so adding one
    // there adds it here. The selection is DERIVED by matching the live expression
    // rather than stored: two fields for one fact desync, and this one would desync
    // every time the expression is edited by hand.
    const shapes = tp.noise_curve.cond_shapes;
    var sel: usize = shapes.len; // past the end = Custom
    for (shapes, 0..) |sh, i| {
        if (std.mem.eql(u8, sh.expr, f_cond_curve.slice())) {
            sel = i;
            break;
        }
    }
    {
        var row = rowBegin(@src(), 6, "Curve");
        defer row.deinit();
        var labels: [shapes.len + 1][]const u8 = undefined;
        inline for (shapes, 0..) |sh, i| labels[i] = sh.name;
        labels[shapes.len] = "Custom…";
        if (style.chipDropdown(@src(), &labels, &sel, .{ .id_extra = 0 })) {
            // Custom leaves the expression alone; the field below edits it.
            if (sel < shapes.len) f_cond_curve.set(shapes[sel].expr);
        }
    }
    if (sel >= shapes.len) {
        var row = rowBegin(@src(), 9, "Expression");
        defer row.deinit();
        numChip(@src(), &f_cond_curve.data, "", 300, 2);
    }
    {
        var row = rowBegin(@src(), 7, "Keep length");
        defer row.deinit();
        _ = dvui.checkbox(@src(), &f_cond_keep_norm, null, .{ .gravity_y = 0.5 });
    }
    {
        var row = rowBegin(@src(), 8, "On negative");
        defer row.deinit();
        numChip(@src(), &f_cond_negative_buf, "x", 70, 1);
    }

    const curve = f_cond_curve.slice();
    const bad = curve.len > 0 and std.meta.isError(tp.noise_curve.validate(curve));
    if (curve.len == 0) {
        help(3, "No curve, so this render carries no noise. A bare number is a flat curve.");
    } else if (bad) {
        help(4, "That curve does not parse, so this render would carry no noise. It is an " ++
            "expression in t, l, n and a.");
    } else {
        help(5, "Shared moves a token's encoder taps together and SHIFTS the picture; " ++
            "independent moves them apart and ERODES the prompt. The deep taps carry " ++
            "more than the shallow ones. Start at Shared · All taps · 0.3. Custom takes " ++
            "an expression in t (tap depth), l and n (token index and count) and a.");
    }
}

/// Conditioning steering: each row names a direction by text and dials it.
///
/// Independent of the noise toggle above: they are different operators on the same
/// buffer (one moves along a named direction, the other along a random one) and
/// either is useful without the other.
fn renderCondSteer() void {
    {
        var row = rowBegin(@src(), 10, "Steer");
        defer row.deinit();
        if (f_steer_n < max_steers and bubbles.secondaryButton(@src(), 0, "+ term", true)) {
            f_steer_text[f_steer_n] = .{};
            // append at 1.0: the mode that adds a thing without trading the prompt
            // away, at the scale that means "the phrase as written".
            f_steer_scale[f_steer_n] = 1.0;
            f_steer_mode[f_steer_n] = .concat;
            f_steer_n += 1;
        }
        if (f_steer_n == 0) {
            dvui.labelNoFmt(@src(), "none", .{}, .{
                .font = F.ui,
                .color_text = C.text_ghost,
                .padding = .{},
                .margin = .{ .x = 8 },
                .gravity_y = 0.5,
            });
        }
    }

    var remove: ?usize = null;
    for (0..f_steer_n) |i| {
        // Two rows: the text entry alone, then the slider and the button. Keeps the
        // entry wide enough for a phrase.
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = i, .expand = .horizontal, .margin = .{ .h = 2 } });
            defer row.deinit();
            steer_editing[i] = style.chipInput(@src(), &f_steer_text[i].data, "", 250, .{ .id_extra = i }).editing;
        }
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = i, .expand = .horizontal, .margin = .{ .x = 8, .h = 6 } });
            defer row.deinit();
            _ = dvui.sliderEntry(@src(), "{d:0.2}", .{
                .value = &f_steer_scale[i],
                // Wide on purpose. ~1 is already total takeover (a term at 1.0
                // renders ITSELF and discards the prompt), so the headroom past
                // it is for terms fighting each other, and the negative side for
                // suppressing something the prompt keeps re-asserting.
                .min = -3.0,
                .max = 3.0,
                .interval = 0.01,
            }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 110 } });
            _ = enumChip(@src(), config.SteerMode, &f_steer_mode[i], i);
            if (bubbles.secondaryButton(@src(), i, "Remove", true)) remove = i;
        }
    }
    if (remove) |i| {
        var j = i;
        while (j + 1 < f_steer_n) : (j += 1) {
            f_steer_text[j] = f_steer_text[j + 1];
            f_steer_scale[j] = f_steer_scale[j + 1];
        }
        f_steer_n -= 1;
    }
    if (f_steer_n > 0) {
        help(6, "Each term is a direction, measured as the chip says. Positive moves " ++
            "toward the phrase, negative away. Scale is a fraction of " ++
            "the conditioning itself, so it bites fast: 0.1-0.3 nudges, 0.5 changes the " ++
            "look, and 1.0 renders the term INSTEAD of your prompt. Terms are added, so " ++
            "two that mean similar things partly cancel when their signs oppose -- the " ++
            "log says by how much. Each costs one text encode per render.\n\n" ++
            "append adds the phrase's own tokens, so the model can put a new OBJECT in " ++
            "the picture with your prompt untouched -- leave its scale at 1.0, which is " ++
            "the phrase as written. The other three bias the tokens you already have: " ++
            "toward moves your render AT the phrase and gives up your prompt as it goes, " ++
            "trait adds what the phrase IS and leaves your subject alone (more tentacles, " ++
            "less anime), and add is trait's stronger form, stripping out what the phrase " ++
            "shares with your prompt so the subject survives further up the scale.");
    }
}

/// Residual-stream steering: edit the DiT's own activations between blocks, along
/// directions derived by `act-derive`.
///
/// The directions are a file on the HOST's disk, not something the client uploads:
/// a set is ~700 KB and belongs beside the models.
fn renderActSteer() void {
    {
        var row = rowBegin(@src(), 11, "Act steer");
        defer row.deinit();
        numChip(@src(), &f_act_dirs.data, "", 250, 0);
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .x = 8, .h = 4 } });
        defer row.deinit();
        if (bubbles.secondaryButton(@src(), 2, "Create a direction…", true)) {
            if (std.mem.sliceTo(&g_act_size, 0).len == 0) setBuf(&g_act_size, "256");
            if (std.mem.sliceTo(&g_act_steps, 0).len == 0) setBuf(&g_act_steps, "8");
            g_act_sent = false;
            g_act_modal = true;
        }
    }
    if (f_act_dirs.slice().len == 0) return;
    {
        var row = rowBegin(@src(), 12, "Strength");
        defer row.deinit();
        _ = dvui.sliderEntry(@src(), "{d:0.3}", .{
            .value = &f_act_scale,
            .min = -0.5,
            .max = 0.5,
            .interval = 0.001,
        }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 140 } });
        _ = enumChip(@src(), config.ActOp, &f_act_op, 0);
    }
    // Derived by matching the live expression, like the conditioning curve, so
    // editing it cannot desync a remembered name. Empty means flat, which is what
    // "All blocks" is, so it reads as that rather than as Custom.
    const shapes = tp.noise_curve.block_shapes;
    var sel: usize = shapes.len;
    if (f_act_curve.slice().len == 0) {
        sel = 0;
    } else for (shapes, 0..) |sh, i| {
        if (std.mem.eql(u8, sh.expr, f_act_curve.slice())) {
            sel = i;
            break;
        }
    }
    {
        var row = rowBegin(@src(), 13, "Blocks");
        defer row.deinit();
        var labels: [shapes.len + 1][]const u8 = undefined;
        inline for (shapes, 0..) |sh, i| labels[i] = sh.name;
        labels[shapes.len] = "Custom…";
        if (style.chipDropdown(@src(), &labels, &sel, .{ .id_extra = 1 })) {
            // Custom leaves the expression alone; the field below edits it.
            if (sel < shapes.len) f_act_curve.set(shapes[sel].expr);
        }
    }
    if (sel >= shapes.len) {
        {
            var row = rowBegin(@src(), 15, "Expression");
            defer row.deinit();
            numChip(@src(), &f_act_curve.data, "", 250, 3);
        }
        if (std.meta.isError(tp.noise_curve.validate(f_act_curve.slice()))) {
            help(10, "That curve does not parse, so this render would steer nothing. " ++
                "It is an expression in t, 0 at the first block and 1 at the last.");
        }
    }
    {
        var row = rowBegin(@src(), 14, "Keep length");
        defer row.deinit();
        _ = dvui.checkbox(@src(), &f_act_keep_norm, null, .{ .gravity_y = 0.5 });
    }
    help(7, "Edits the DiT's activations between blocks, not the prompt. Strength " ++
        "compounds over the blocks it touches, so it bites far harder than the " ++
        "conditioning knobs: start near 0.02. Early blocks set the composition and " ++
        "late ones the rendering, so which blocks matters as much as how much.");
}

/// Open the create-direction modal, so `ui-probe` can draw it.
pub fn openActModalForProbe() void {
    g_act_modal = true;
    setBuf(&g_act_set_a, "a hill at night with a large full moon; the ocean at night with a large full moon");
    setBuf(&g_act_set_b, "a hill at night; the ocean at night");
    setBuf(&g_act_name, "moon");
    setBuf(&g_act_size, "256");
    setBuf(&g_act_steps, "8");
}

/// Collect two prompt sets and ask the host to derive a direction from them.
///
/// The work happens on the host: it owns the model, and the file it writes lands
/// beside the models there. The client only posts the request and waits.
fn renderActModal() void {
    var win = dvui.floatingWindow(@src(), .{ .modal = true, .open_flag = &g_act_modal }, .{
        .min_size_content = .{ .w = 520, .h = 420 },
    });
    defer win.deinit();
    win.dragAreaSet(dvui.windowHeader("Create an activation direction", "", &g_act_modal));

    help(8, "Two sets of prompts that differ in ONE thing. The direction is what " ++
        "separates them: put the thing you want in every A prompt and leave it out " ++
        "of the matching B prompt. Six matched pairs beat one unmatched pair by a " ++
        "lot. Separate prompts with a semicolon. The name is one word; the file " ++
        "lands in your first model folder as <name>.actd.");

    for ([_][]const u8{ "With the thing (A)", "Without it (B)" }, [_][]u8{ &g_act_set_a, &g_act_set_b }, 0..) |label, buf, i| {
        dvui.labelNoFmt(@src(), label, .{}, .{ .id_extra = i, .font = F.ui, .color_text = C.text_dim, .padding = .{ .y = 6 } });
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = buf },
            .multiline = true,
            .break_lines = true,
            .scroll_horizontal = false,
        }, .{ .id_extra = i, .expand = .horizontal, .min_size_content = .{ .h = 64 } });
        te.deinit();
    }
    {
        var row = rowBegin(@src(), 0, "Name");
        defer row.deinit();
        numChip(@src(), &g_act_name, "", 160, 0);
        numChip(@src(), &g_act_size, "px", 54, 1);
        numChip(@src(), &g_act_steps, "steps", 54, 2);
    }

    const busy = g_m.act_busy;
    if (busy) {
        help(9, "Working. The host is running every prompt through the model, which is " ++
            "seconds per prompt and holds up its renders meanwhile.");
    } else if (g_act_sent) {
        if (g_m.act_err.slice().len > 0) {
            var buf: [160]u8 = undefined;
            help(9, std.fmt.bufPrint(&buf, "That did not work: {s}", .{g_m.act_err.slice()}) catch "That did not work.");
        } else if (g_m.act_path.slice().len > 0) {
            help(9, "Done. The path is filled in above; Generate uses it now.");
        }
    }

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 8 } });
        defer row.deinit();
        if (bubbles.secondaryButton(@src(), 0, if (busy) "Working…" else "Create", !busy)) {
            g_act_sent = true;
            g_m.act_busy = true;
            g_m.act_path.set("");
            g_m.act_err.set("");
            g_post(.{ .act_derive = .{
                .set_a = std.mem.sliceTo(&g_act_set_a, 0),
                .set_b = std.mem.sliceTo(&g_act_set_b, 0),
                .name = std.mem.sliceTo(&g_act_name, 0),
                .size = @intCast(parseNum(&g_act_size, 256)),
                .steps = @intCast(parseNum(&g_act_steps, 8)),
            } });
        }
        if (bubbles.secondaryButton(@src(), 1, "Close", true)) g_act_modal = false;
    }

    // The host answered: take the path straight into the form, so the thing just
    // made is the thing the next render uses.
    if (g_act_sent and !busy and g_m.act_path.slice().len > 0) {
        f_act_dirs.set(g_m.act_path.slice());
        if (f_act_scale == 0) f_act_scale = 0.02;
    }
}

/// A wrapped note under a section head.
///
/// `F.row`, not `F.ui_sm`: the compact UI roles are single-spaced (line 1.0),
/// which is right for a chip or a label and crushes the lines into each other
/// the moment the text wraps. Anything that can reach a second line needs a role
/// with real leading.
fn help(id: usize, text: []const u8) void {
    var tl = dvui.textLayout(@src(), .{}, .{
        .id_extra = id,
        .expand = .horizontal,
        .background = false,
        .max_size_content = .width(style.Layout.prose_max),
        .padding = .{ .y = 2, .h = 8 },
    });
    defer tl.deinit();
    fonts.addStyled(tl, text, .{}, .{ .font = F.row, .color_text = C.text_ghost });
}

// ----------------------------------------------------------------- composer

fn renderComposer(cfg: *const config.Config, ready: bool) void {
    var block = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = C.canvas,
        .border = style.Edge.top,
        .color_border = style.hairline_soft,
        .padding = .{ .x = 18, .y = 10, .w = 18, .h = 12 },
    });
    defer block.deinit();

    var frame = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = C.sunken,
        .border = style.Edge.all,
        .color_border = style.hairline_hi,
        .corner_radius = R.input,
        .padding = .{ .x = 12, .y = 8, .w = 12, .h = 8 },
    });
    defer frame.deinit();

    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &prompt_buf },
            .multiline = true,
            .placeholder = "Describe the image…",
            .scroll_horizontal = false,
            .break_lines = true,
        }, .{
            .expand = .horizontal,
            .background = false,
            .border = .{},
            .padding = .{},
            .min_size_content = .{ .h = 42 },
            .max_size_content = .height(96),
            .font = F.input,
            .theme = style.noFocusTheme(),
        });
        prompt_h = te.data().rect.h;
        te.deinit();
    }
    // The negative is a second prompt, not a knob, so it sits in the same frame
    // rather than in a section. It only bites when guidance is above 1.
    {
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &negative_buf },
            .multiline = true,
            .placeholder = "Negative prompt (needs guidance above 1)",
            .scroll_horizontal = false,
            .break_lines = true,
        }, .{
            .expand = .horizontal,
            .background = false,
            .border = .{},
            .padding = .{},
            .min_size_content = .{ .h = 20 },
            .max_size_content = .height(56),
            .font = F.ui,
            .color_text = C.text_dim,
            .theme = style.noFocusTheme(),
        });
        negative_h = te.data().rect.h;
        te.deinit();
    }
    // The two ENTRIES are measured, not the block around them: an entry is given
    // its natural height, where the block is the flex child the scroll area left
    // over, so feeding the block's height back would converge on whatever the
    // first frame happened to grant it (zero, and it stayed zero). `chrome` is
    // this function's own padding, border and button row, the same shape
    // `app.g_input_h` uses for the chat composer.
    g_composer_h = prompt_h + negative_h + composer_chrome;

    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .y = 4 } });
    defer row.deinit();

    const st = &g_m.state;
    const generating = st.diff_busy or st.pending_images > 0;
    if (!ready) {
        dvui.labelNoFmt(@src(), "Loading model…", .{}, .{
            .font = F.ui,
            .color_text = C.text_dim,
            .padding = .{},
            .gravity_y = 0.5,
        });
    } else if (generating) {
        var n: [40]u8 = undefined;
        dvui.labelNoFmt(@src(), std.fmt.bufPrint(&n, "{d} in the queue", .{st.pending_images + 1}) catch "", .{}, .{
            .font = F.mono,
            .color_text = C.text_faint,
            .padding = .{},
            .gravity_y = 0.5,
        });
    }
    {
        var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
        sp.deinit();
    }
    if (generating and bubbles.secondaryButton(@src(), 0, "Stop", true)) g_post(.img_cancel_all);
    if (bubbles.primaryButton(@src(), "Generate", ready)) generate(cfg);
}

// ----------------------------------------------------------------- generate

/// Ask the host for `count` generations from the current form values.
fn generate(cfg: *const config.Config) void {
    // Empty is allowed: it encodes to the unconditional embedding, which is a
    // render the model can do and some workflows want.
    const prompt = std.mem.trim(u8, std.mem.sliceTo(&prompt_buf, 0), " \t\r\n");
    const neg = std.mem.trim(u8, std.mem.sliceTo(&negative_buf, 0), " \t\r\n");

    const dims = formDims(cfg);
    const steps = std.math.clamp(parseNum(&steps_buf, cfg.steps), 1, 100);
    const cfg_scale = std.math.clamp(parseFloat(&cfg_buf, 1.0), 0.0, 30.0);
    // Uncapped by design: queue as many as you ask for (min 1). Each is
    // allocated up front, so a huge count is on you, that's the intent.
    const count = @max(1, parseNum(&count_buf, 1));
    const base_seed: u64 = if (random_seed) 0 else parseU64(&seed_buf);

    var params: wire.RenderParams = .{
        .sampler = pipeline_map.toPipelineSampler(f_sampler),
        .scheduler = pipeline_map.toPipelineScheduler(f_scheduler),
        .prompt_syntax = pipeline_map.toPipelineSyntax(f_syntax),
        .emphasis = pipeline_map.toPipelineEmphasis(f_emphasis),
        .compat = pipeline_map.toPipelineCompat(f_compat),
        .cond_noise_amount = condAmount(),
        .cond_noise_shape = pipeline_map.toPipelineCondShape(f_cond_shape),
        .cond_noise_keep_norm = f_cond_keep_norm,
        .cond_noise_negative = condNegative(),
    };
    // The toggle is the switch: an empty curve is off on the engine side too, so a
    // disabled section cannot leave a curve applied.
    if (f_cond_on) params.cond_noise.set(f_cond_curve.slice());
    params.act_dirs.set(f_act_dirs.slice());
    params.act_scale = f_act_scale;
    params.act_op = pipeline_map.toPipelineActOp(f_act_op);
    params.act_curve.set(f_act_curve.slice());
    params.act_keep_norm = f_act_keep_norm;
    for (0..f_steer_n) |i| {
        // An empty term or a zero scale is DROPPED rather than sent: each term
        // costs a text-encoder forward on the host, and "off" should cost nothing.
        if (f_steer_text[i].slice().len == 0 or f_steer_scale[i] == 0) continue;
        params.steers[params.steer_count].set(f_steer_text[i].slice());
        params.steers[params.steer_count].scale = f_steer_scale[i];
        params.steers[params.steer_count].mode = pipeline_map.toPipelineSteerMode(f_steer_mode[i]);
        params.steer_count += 1;
    }

    var specs: [max_loras]wire.LoraSpec = undefined;
    var n_specs: usize = 0;
    for (0..f_lora_n) |i| {
        // A disabled row is OMITTED rather than passed at strength 0: a stack
        // entry costs its factors resident, and "off" should cost nothing.
        if (!f_lora_on[i]) continue;
        specs[n_specs] = .{ .path = f_lora_path[i].slice(), .strength = f_lora_strength[i] };
        n_specs += 1;
    }

    var i: usize = 0;
    while (i < count) : (i += 1) {
        g_post(.{ .img_enqueue = .{
            .prompt = prompt,
            .negative = neg,
            .width = @intCast(dims.w),
            .height = @intCast(dims.h),
            .steps = @intCast(steps),
            .cfg = cfg_scale,
            // Random: the host draws a fresh distinct seed each (0). Fixed: the
            // entered seed, advanced per image so a batch still varies.
            .seed = if (random_seed) 0 else base_seed +% i,
            .params = params,
            .loras = specs[0..n_specs],
            .from_studio = true,
        } });
    }
    // The canvas follows the newest render.
    g_selected = null;
    // The prompt library is app.zig's; hand it the text and let it record.
    recorded_prompt = .{ .prompt = prompt, .negative = neg };
}

test "reseeding the studio form asks for a random seed again" {
    // Reopening an image pins its exact seed. Reseeding after that (a new
    // model, re-entering the studio) must not leave the form claiming FIXED
    // over the default 0, because the host reads a seed of 0 as "draw one":
    // the field would be ignored on every render while looking honoured.
    const cfg: config.Config = .{};
    random_seed = false;
    setFmt(&seed_buf, "{d}", .{12345});
    seed(&cfg, null);
    try std.testing.expect(random_seed);
    try std.testing.expectEqual(@as(u64, 0), parseU64(&seed_buf));
}

test "a preview smaller than its tile is grown to fill it, keeping its shape" {
    // A preview arrives at a fraction of the final size. A fit that only ever
    // shrank drew every tile at whatever the sampler decoded at, which left the
    // canvas's room unused and made the grid look broken.
    const sz = fitBox(152, 104, 352, 241);
    errdefer std.debug.print("grown to {d}x{d}\n", .{ sz.w, sz.h });
    try std.testing.expect(sz.w > 340 and sz.w <= 352);
    try std.testing.expect(sz.h > 230 and sz.h <= 241);

    // Portrait in the same cell: bound by height, not width.
    const tall = fitBox(104, 152, 352, 241);
    errdefer std.debug.print("portrait {d}x{d}\n", .{ tall.w, tall.h });
    try std.testing.expectApproxEqAbs(@as(f32, 241), tall.h, 0.5);
    try std.testing.expect(tall.w < 200);
}

test "the canvas grid shows more renders only while the cells stay pictures" {
    // The column with the form under it: ~586 wide, ~294 of height to give. Two
    // fit side by side; a 2x2 would put every tile back at rail-thumbnail size,
    // so the third is counted in the caption instead.
    const narrow = gridFor(3, 586, 294);
    errdefer std.debug.print("narrow {d}x{d} cell {d}x{d} shown {d}\n", .{ narrow.cols, narrow.rows, narrow.cw, narrow.ch, narrow.shown });
    try std.testing.expectEqual(@as(usize, 2), narrow.shown);
    try std.testing.expect(narrow.short() >= min_tile);

    // The form beside the canvas instead: the same three renders all fit.
    const wide = gridFor(3, 786, 594);
    errdefer std.debug.print("wide cell {d}x{d} shown {d}\n", .{ wide.cw, wide.ch, wide.shown });
    try std.testing.expectEqual(@as(usize, 3), wide.shown);
    try std.testing.expectEqual(@as(usize, 2), wide.cols);
    try std.testing.expect(wide.short() >= min_tile);

    // One render never gets a grid, however much room there is.
    try std.testing.expectEqual(@as(usize, 1), gridFor(1, 786, 594).shown);

    // A canvas too small for even two keeps one big picture rather than two
    // thumbnails: that is what the rail is for.
    try std.testing.expectEqual(@as(usize, 1), gridFor(4, 300, 200).shown);
}

test "a pin is dropped as soon as its render is no longer in motion" {
    var a: Image = .{ .info = .{ .id = 7 } };
    var b: Image = .{ .info = .{ .id = 9 } };
    const live = [_]*const Image{ &a, &b };

    try std.testing.expectEqual(@as(?usize, 0), liveIndex(&live, 7));
    try std.testing.expectEqual(@as(?usize, 1), liveIndex(&live, 9));
    // The finished picture a user clicked in the Library: not in motion, so the
    // canvas does not park on it and go on showing it while work continues.
    try std.testing.expectEqual(@as(?usize, null), liveIndex(&live, 11));
    try std.testing.expectEqual(@as(?usize, null), liveIndex(&.{}, 7));
}

/// A seed to start editing from. Only has to be unlikely to repeat and never
/// zero, which the wire reads as "the host draws one"; the frame clock is the
/// only time source a view has, so it is hashed rather than used raw.
fn freshSeed() u64 {
    const t: u64 = @bitCast(@as(i64, @truncate(dvui.frameTimeNS())));
    return std.hash.Wyhash.hash(0x5EED, std.mem.asBytes(&t)) | 1;
}

fn parseU64(buf: []const u8) u64 {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseInt(u64, s, 10) catch 0;
}

/// Copy the form's recipe into the config defaults, for the studio's explicit
/// "Save as defaults". The ONE path by which a generation value this module
/// edits reaches config; everything else here rides on the request.
pub fn saveDefaults(cfg: *config.Config, fam: ?model_spec.Family) void {
    const d = formDims(cfg);
    cfg.steps = std.math.clamp(parseNum(&steps_buf, cfg.steps), 1, 100);
    cfg.cfg_scale = std.math.clamp(parseFloat(&cfg_buf, 1.0), 0.0, 30.0);
    cfg.sampler = f_sampler;
    cfg.scheduler = f_scheduler;
    cfg.prompt_syntax = f_syntax;
    cfg.emphasis = f_emphasis;
    cfg.compat = f_compat;
    cfg.cond_noise = f_cond_on;
    cfg.cond_noise_curve = f_cond_curve;
    cfg.cond_noise_shape = f_cond_shape;
    cfg.cond_noise_keep_norm = f_cond_keep_norm;
    cfg.cond_noise_amount = condAmount();
    cfg.cond_noise_negative = condNegative();
    cfg.act_dirs = f_act_dirs;
    cfg.act_scale = f_act_scale;
    cfg.act_op = f_act_op;
    cfg.act_curve = f_act_curve;
    cfg.act_keep_norm = f_act_keep_norm;
    cfg.cond_steers = .{};
    for (0..f_steer_n) |i| {
        if (f_steer_text[i].slice().len == 0) continue;
        cfg.cond_steers.items[cfg.cond_steers.count] = .{
            .text = f_steer_text[i],
            .scale = f_steer_scale[i],
            .mode = f_steer_mode[i],
        };
        cfg.cond_steers.count += 1;
    }
    // Size goes through the framing pair, which is its single writer; the exact
    // fields are the same value in pixels, so they map back rather than opening
    // a second route to `width`/`height`.
    cfg.framing_ratio = if (cfg.studio_size_mode == .framing) ratio else framing.nearestRatio(d.w, d.h);
    cfg.framing_mp = @floatCast(if (cfg.studio_size_mode == .framing)
        parseFloat(&mp_buf, cfg.framing_mp)
    else
        framing.megapixelsOf(d.w, d.h));
    cfg.applyFraming();

    if (fam) |f| if (f.supportsLora()) {
        const key = selection.familyKey(f);
        // Rewrite the family's list from the form's, in the form's order (order
        // is what the stack's positional dials mean).
        //
        // BOUNDED, not `while (count > 0)`: this runs on the UI thread, and a
        // remove that did not match its own path -- a truncated buffer, a future
        // change to how the table is keyed -- would spin the frame forever
        // instead of losing one row.
        for (0..max_loras) |_| {
            const on = selection.lorasForFamily(cfg, f);
            if (on.len == 0) break;
            // Copied out first: `removeFamilyLora` shifts the table the view is
            // built from.
            var p: config.PathBuf = .{};
            p.set(on[0].path.slice());
            cfg.removeFamilyLora(key, p.slice());
        }
        for (0..f_lora_n) |i| {
            if (cfg.addFamilyLora(key, f_lora_path[i].slice())) |l| {
                l.strength = f_lora_strength[i];
                l.enabled = f_lora_on[i];
            }
        }
    };
}

// --------------------------------------------------------------- form filling

/// Put a prompt (and the negative it was written with) into the composer,
/// leaving every knob alone. This is what a prompt-library row does: the
/// library stores text, and the recipe is whatever the form is set to now.
pub fn setPrompt(prompt: []const u8, negative: []const u8) void {
    setBuf(&prompt_buf, prompt);
    setBuf(&negative_buf, negative);
}

/// Clear the composer for a fresh prompt (the rail's "New prompt").
pub fn clearPrompt() void {
    setBuf(&prompt_buf, "");
    setBuf(&negative_buf, "");
}

/// Load a finished image's parameters into the studio form, for the tool card's
/// "Open in Studio". Everything the user would otherwise re-type by reading the
/// metadata line: prompt, negative, size, steps, cfg, and the exact seed, so the
/// first thing Generate does is reproduce what they were looking at, and every
/// edit from there is a deliberate change to a known starting point.
pub fn loadFrom(im: *const Image) void {
    // Prefer the SAVED FILE's own metadata when this image has no live request
    // behind it (one rebuilt from a reopened conversation). The PNG carries the
    // AUTOMATIC1111 block that describes exactly how it was made, which makes
    // the file the record and saves the transcript carrying a second copy that
    // could disagree with it.
    const info = &im.info;
    if (info.prompt.len == 0) if (im.saved_path) |path| {
        if (loadFromFile(path)) return;
    };

    setBuf(&prompt_buf, info.prompt);
    setBuf(&negative_buf, info.negative);
    setDims(info.req_width, info.req_height);
    setFmt(&steps_buf, "{d}", .{info.req_steps});
    setFmt(&cfg_buf, "{d:.1}", .{info.req_cfg});
    setFmt(&seed_buf, "{d}", .{info.req_seed});
    setBuf(&count_buf, "1");
    random_seed = false; // an exact seed is the whole point of reopening one
    f_sampler = pipeline_map.fromPipelineSampler(info.params.sampler);
    f_scheduler = pipeline_map.fromPipelineScheduler(info.params.scheduler);
    seeded = true;
    loaded_from_image = true; // do not let `seed()` overwrite what we just put here
}

/// Fill the form from a saved PNG's `parameters` block. Returns false when the
/// file is gone or carries no metadata (an image saved by something that did
/// not write one), so the caller can fall back to whatever it has in memory.
fn loadFromFile(path: []const u8) bool {
    const bytes = std.Io.Dir.cwd().readFileAlloc(g_io, path, g_gpa, @enumFromInt(64 << 20)) catch return false;
    defer g_gpa.free(bytes);
    const text = tp.image.pngText(bytes, "parameters") orelse return false;
    const p = save_image.parseA1111Params(text);
    if (p.prompt.len == 0) return false;

    setBuf(&prompt_buf, p.prompt);
    setBuf(&negative_buf, p.negative);
    if (p.width != null and p.height != null) setDims(p.width.?, p.height.?);
    if (p.steps) |v| setFmt(&steps_buf, "{d}", .{v});
    if (p.cfg) |v| setFmt(&cfg_buf, "{d:.1}", .{v});
    if (p.seed) |v| setFmt(&seed_buf, "{d}", .{v});
    setBuf(&count_buf, "1");
    random_seed = false;
    seeded = true;
    loaded_from_image = true;
    return true;
}

/// Put concrete dimensions into whichever size control is showing, so a
/// restored render lands on the size it was made at either way.
fn setDims(w: usize, h: usize) void {
    setFmt(&width_buf, "{d}", .{w});
    setFmt(&height_buf, "{d}", .{h});
    ratio = framing.nearestRatio(w, h);
    setFmt(&mp_buf, "{s}", .{framing.formatMp(&mp_scratch, framing.megapixelsOf(w, h))});
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
