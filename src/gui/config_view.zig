//! The settings screen (a full-window view, toggled from the chat input's gear
//! button). Kept separate from `app.zig` so the main loop stays lean, and from
//! `config.zig` so the persisted data model has no UI dependency.
//!
//! The view owns only the transient numeric edit buffers; every path field
//! binds a `textEntry` directly to the `config.Config`'s own `PathBuf` storage,
//! so there is no edit state to reconcile. `apply`/`cancel` are supplied by the
//! app (save + reload vs. discard).
const std = @import("std");
const dvui = @import("dvui");
const config = @import("config.zig");
const noise_curve = @import("TensorPencil").noise_curve;
const style = @import("style.zig");
const bubbles = @import("bubbles.zig");
const diffuser = @import("diffuser.zig");
const model_spec = @import("model_spec.zig");
const SDLBackend = @import("backend");

// SDL owns file picking: SDL_ShowOpen*Dialog parents the native dialog to the
// main window (so it appears ON TOP, not behind it), unlike tinyfiledialogs,
// which spawns an unparented helper process. The dialogs are ASYNC: the callback
// fires on the main thread during SDL event pumping and writes the chosen path
// straight into the target PathBuf, then wakes a frame to show it.
var g_window: ?*SDLBackend.c.SDL_Window = null;
var g_wakeup: ?*const fn () void = null;
// Used only to check that a browse start directory exists before handing it to
// SDL (see `startDir`).
var g_io: ?std.Io = null;
// One dialog at a time: the SDL pickers are async (unlike the old blocking
// tinyfiledialogs call), so without this a second Browse click would stack a
// second dialog. Cleared in the callback.
var g_dialog_open: bool = false;

// Memoized checkpoint inspection for the three diffusion path rows (see
// `model_spec.Cache`, a probe opens and parses a checkpoint header, far too
// expensive for a per-frame render). Only live once `setEnv` has run.
var g_spec_ready: bool = false;
var g_primary_cache: model_spec.Cache = undefined;
var g_te_cache: model_spec.Cache = undefined;
var g_te2_cache: model_spec.Cache = undefined;
var g_vae_cache: model_spec.Cache = undefined;

/// Wire the SDL window + frame-wakeup (called once from app init). Until set,
/// the pickers no-op (the user can still type a path) and the checkpoint
/// inspection panel stays hidden.
pub fn setEnv(
    window: ?*SDLBackend.c.SDL_Window,
    wakeup: *const fn () void,
    gpa: std.mem.Allocator,
    io: std.Io,
) void {
    g_window = window;
    g_wakeup = wakeup;
    g_io = io;
    g_primary_cache = model_spec.Cache.init(gpa, io);
    g_te_cache = model_spec.Cache.init(gpa, io);
    g_te2_cache = model_spec.Cache.init(gpa, io);
    g_vae_cache = model_spec.Cache.init(gpa, io);
    g_spec_ready = true;
}

/// Release the inspection caches (app teardown).
pub fn deinit() void {
    if (!g_spec_ready) return;
    g_primary_cache.deinit();
    g_te_cache.deinit();
    g_te2_cache.deinit();
    g_vae_cache.deinit();
    g_spec_ready = false;
}

// Static filter sets, SDL keeps the pointer across the async call, so these
// MUST outlive the dialog (module-level const, never a stack local).
const gguf_sdl = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "GGUF models", .pattern = "gguf" },
    .{ .name = "All files", .pattern = "*" },
};
const safetensors_sdl = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "Safetensors", .pattern = "safetensors" },
    .{ .name = "All files", .pattern = "*" },
};
/// Any component that can arrive as its own file, the primary checkpoint, and the
/// text encoders. `pipeline.Container` opens all of them by MAGIC, not by extension,
/// so the picker must not hide GGUFs. A default filter that omits a format the
/// engine supports reads as "unsupported": an encoder row offering `safetensors` only
/// hides perfectly loadable GGUF encoders from the dialog.
const checkpoint_sdl = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "Checkpoints", .pattern = "safetensors;gguf" },
    .{ .name = "Safetensors", .pattern = "safetensors" },
    .{ .name = "GGUF", .pattern = "gguf" },
    .{ .name = "All files", .pattern = "*" },
};

/// SDL open-file / folder callback: `userdata` is the target *PathBuf. Writes
/// the first chosen path and wakes a frame. Cancel (files[0]==null) leaves it.
fn dialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    g_dialog_open = false;
    const pb: *config.PathBuf = @ptrCast(@alignCast(userdata orelse return));
    const files = filelist orelse return; // dialog error
    if (files[0] == null) return; // cancelled
    pb.set(std.mem.span(files[0]));
    if (g_wakeup) |w| w();
}

/// App-supplied actions for the two footer buttons.
pub const Callbacks = struct {
    /// Persist the (already-parsed) config and reload the session.
    apply: *const fn () void,
    /// Discard edits and return to chat.
    cancel: *const fn () void,
};

// Transient view state: numeric fields are edited as text, seeded from the
// config the first frame after `open()`.
var seeded: bool = false;

/// Drop the seeded form buffers so the next render re-reads the config.
///
/// The buffers are seeded ONCE, which means an edit made anywhere else (the
/// composer's framing chips) leaves this view holding the old numbers — and
/// Apply would write them straight back over the change.
pub fn reseed() void {
    seeded = false;
}
var steps_buf: [16]u8 = [_]u8{0} ** 16;
var regen_buf: [16]u8 = [_]u8{0} ** 16;
var noise_amt_buf: [16]u8 = [_]u8{0} ** 16;
// Per-reply token cap. Sits in the sampling SECTION but is committed with the
// other plain integers (commitNumbers), not with commitSampling: it is not part
// of a preset, see config.Config.max_new_tokens.
var maxnew_buf: [16]u8 = [_]u8{0} ** 16;
// LLM sampling controls (same text-buffer pattern; floats parse on commit).
var temp_buf: [16]u8 = [_]u8{0} ** 16;
var topk_buf: [16]u8 = [_]u8{0} ** 16;
var topp_buf: [16]u8 = [_]u8{0} ** 16;
var minp_buf: [16]u8 = [_]u8{0} ** 16;
var rpen_buf: [16]u8 = [_]u8{0} ** 16;
var rlast_buf: [16]u8 = [_]u8{0} ** 16;
var ppen_buf: [16]u8 = [_]u8{0} ** 16;
var fpen_buf: [16]u8 = [_]u8{0} ** 16;
// Preset name entry (save/load/delete target in the sampling section).
var preset_name_buf: [config.max_preset_name]u8 = [_]u8{0} ** config.max_preset_name;
// System-prompt name entry (save/load/delete target in the system-prompt section).
var sys_prompt_name_buf: [config.max_sys_prompt_name]u8 = [_]u8{0} ** config.max_sys_prompt_name;


/// Call when the view is (re)entered so numeric buffers reseed from the config.
pub fn open() void {
    seeded = false;
    // Re-read the checkpoints on every visit to Settings. The memo keys on the
    // path text, so it would otherwise never notice a file REPLACED under an
    // unchanged path, a once-per-open probe is the cheap way to stay honest.
    if (g_spec_ready) {
        g_primary_cache.invalidate();
        g_te_cache.invalidate();
        g_te2_cache.invalidate();
        g_vae_cache.invalidate();
    }
}

fn seed(cfg: *const config.Config) void {
    _ = std.fmt.bufPrintZ(&steps_buf, "{d}", .{cfg.steps}) catch {};
    _ = std.fmt.bufPrintZ(&regen_buf, "{d}", .{cfg.regen_cache_mb}) catch {};
    _ = std.fmt.bufPrintZ(&noise_amt_buf, "{d}", .{cfg.weight_noise_amount}) catch {};
    _ = std.fmt.bufPrintZ(&maxnew_buf, "{d}", .{cfg.max_new_tokens}) catch {};
    seedSampling(cfg);
    seeded = true;
}

/// Reseed just the sampling buffers from the config, also used when a preset
/// is loaded (the studio numeric buffers keep their in-progress edits).
fn seedSampling(cfg: *const config.Config) void {
    const s = &cfg.sampling;
    _ = std.fmt.bufPrintZ(&temp_buf, "{d}", .{s.temperature}) catch {};
    _ = std.fmt.bufPrintZ(&topk_buf, "{d}", .{s.top_k}) catch {};
    _ = std.fmt.bufPrintZ(&topp_buf, "{d}", .{s.top_p}) catch {};
    _ = std.fmt.bufPrintZ(&minp_buf, "{d}", .{s.min_p}) catch {};
    _ = std.fmt.bufPrintZ(&rpen_buf, "{d}", .{s.repeat_penalty}) catch {};
    _ = std.fmt.bufPrintZ(&rlast_buf, "{d}", .{s.repeat_last_n}) catch {};
    _ = std.fmt.bufPrintZ(&ppen_buf, "{d}", .{s.presence_penalty}) catch {};
    _ = std.fmt.bufPrintZ(&fpen_buf, "{d}", .{s.frequency_penalty}) catch {};
}

fn parseNum(buf: []const u8, fallback: usize) usize {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseInt(usize, s, 10) catch fallback;
}

fn parseFloatBuf(buf: []const u8, fallback: f32) f32 {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseFloat(f32, s) catch fallback;
}

/// Round to a multiple of 16 within the pipeline's supported range.
fn clampDim(n: usize) usize {
    return std.math.clamp(n, 256, 2048) / 16 * 16;
}

/// Read the numeric edit buffers back into the config (with clamping). Called
/// on Apply, before the app persists + reloads.
fn commitNumbers(cfg: *config.Config) void {
    cfg.steps = std.math.clamp(parseNum(&steps_buf, cfg.steps), 1, 100);
    // Regen (checkpoint) cache: host RAM, capped at 64 GB to catch typos.
    cfg.regen_cache_mb = @min(parseNum(&regen_buf, cfg.regen_cache_mb), 64 << 10);
    // Weight-noise amount. Clamped to [0,1]: past ~0.8 a front-loaded shape only
    // produces fluent nonsense, and beyond 1 a scale multiplier stops meaning
    // anything, but seeing the top of the range is the point of exposing it.
    cfg.weight_noise_amount = std.math.clamp(parseFloatBuf(&noise_amt_buf, cfg.weight_noise_amount), 0, 1);
    // Max response: 0 keeps its "no explicit cap" meaning, so this only bounds
    // typos at the top end, the real limit is the context window either way.
    cfg.max_new_tokens = @min(parseNum(&maxnew_buf, cfg.max_new_tokens), 1 << 20);
    commitSampling(cfg);
}

/// Read the sampling edit buffers back into the config, clamped to sane ranges
/// (the library additionally caps top_k at its candidate limit and the penalty
/// window at max_penalty_window). Also called when saving a preset, so the
/// preset stores what the controls will actually run with.
fn commitSampling(cfg: *config.Config) void {
    const s = &cfg.sampling;
    s.temperature = std.math.clamp(parseFloatBuf(&temp_buf, s.temperature), 0, 5);
    s.top_k = @min(parseNum(&topk_buf, s.top_k), 512);
    s.top_p = std.math.clamp(parseFloatBuf(&topp_buf, s.top_p), 0, 1);
    s.min_p = std.math.clamp(parseFloatBuf(&minp_buf, s.min_p), 0, 1);
    s.repeat_penalty = std.math.clamp(parseFloatBuf(&rpen_buf, s.repeat_penalty), 0.1, 8);
    s.repeat_last_n = @min(parseNum(&rlast_buf, s.repeat_last_n), 2048);
    s.presence_penalty = std.math.clamp(parseFloatBuf(&ppen_buf, s.presence_penalty), -10, 10);
    s.frequency_penalty = std.math.clamp(parseFloatBuf(&fpen_buf, s.frequency_penalty), -10, 10);
}

pub fn render(cfg: *config.Config, cb: Callbacks) void {
    if (!seeded) seed(cfg);

    // Settings takes the whole window (it is a mode, not a place in the
    // workspace), so unlike Studio it keeps a header of its own — it is the only
    // way back.
    {
        var header = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = style.Layout.title_h },
            .max_size_content = .height(style.Layout.title_h),
            .background = true,
            .color_fill = style.C.chrome,
            .border = style.Edge.bottom,
            .color_border = style.hairline,
            .padding = .{ .x = 14, .w = 14 },
        });
        defer header.deinit();
        dvui.labelNoFmt(@src(), "Settings", .{}, .{
            .font = style.F.app,
            .color_text = style.C.text_hi,
            .gravity_y = 0.5,
            .padding = .{},
        });
        {
            var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
            sp.deinit();
        }
        if (bubbles.secondaryButton(@src(), 0, "Cancel", true)) cb.cancel();
        if (bubbles.primaryButton(@src(), "Apply & Reload", true)) {
            commitNumbers(cfg);
            cb.apply();
        }
    }

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = dvui.Rect.all(6) });
    defer body.deinit();

    section("Models");
    help("The LLM is required for chat. Image generation needs only the diffusion " ++
        "model: many checkpoints bundle their own text encoder(s) and VAE, and the " ++
        "encoder/VAE fields below are OVERRIDES — set one only to supply a piece " ++
        "the checkpoint lacks, or to replace one it has. Vision (chatting about " ++
        "images) needs the vision tower. Any unset feature is simply disabled.");
    pathRow("LLM model", &cfg.llm_model, &gguf_sdl);
    pathRow("Vision tower", &cfg.vision_tower, &gguf_sdl);
    pathRow("Diffusion model", &cfg.diffusion_model, &checkpoint_sdl);
    pathRow("Text encoder", &cfg.text_encoder, &checkpoint_sdl);
    // SDXL's second tower. Always shown rather than revealed only for a detected
    // SDXL checkpoint: a row that appears and disappears as you edit the path
    // above it moves everything below, and the panel already says when it is
    // needed. Ignored by every single-tower architecture.
    pathRow("Text encoder 2 (SDXL)", &cfg.text_encoder_2, &checkpoint_sdl);
    pathRow("VAE", &cfg.vae, &safetensors_sdl);
    pathRow("TAESD preview", &cfg.taesd, &safetensors_sdl);
    checkpointPanel(cfg);

    section("Image generation");
    help("Generated images (chat and the image studio) are saved here as PNGs " ++
        "with AUTOMATIC1111-style metadata embedded. Defaults to a TensorPencil " ++
        "folder in your Pictures directory; clear it to disable saving.");
    dirRow("Output folder", &cfg.output_dir);
    numRow("Default steps", &steps_buf);
    // No width/height here. Image size is the composer's framing chips (aspect
    // ratio + megapixels), which are the single source for it; a second pair of
    // fields would be two controls writing one value, and whichever was touched
    // last would silently win.

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Sampler", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        samplerDropdown(&cfg.sampler);
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Scheduler", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        schedulerDropdown(&cfg.scheduler);
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Prompt syntax", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        promptSyntaxDropdown(&cfg.prompt_syntax);
    }
    // Only meaningful under the A1111 dialect, so it is hidden otherwise rather than
    // shown greyed: a visible control that does nothing invites the reading that the
    // ComfyUI path has a weighting choice too, and it does not.
    if (cfg.prompt_syntax == .a1111) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Emphasis", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        emphasisDropdown(&cfg.emphasis);
    }
    // Deliberately NOT hidden behind `prompt_syntax == .a1111`: the two axes are
    // independent, and a user reproducing an A1111 image needs to find this one even
    // when their prompt has no emphasis syntax in it at all.
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Sampling compat", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        compatDropdown(&cfg.compat);
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Live preview", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.Preview, .{ .choice = &cfg.preview }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }
    if (cfg.preview == .taesd) {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Preview size", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        taesdSizeDropdown(&cfg.taesd_size);
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "VAE decode", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.VaeDecode, .{ .choice = &cfg.vae_decode }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }

    section("Backends");
    help("Compute backend for each engine. The two are independent — e.g. the " ++
        "LLM on CUDA while diffusion runs on Vulkan. The chat LLM currently " ++
        "supports the CUDA backends only (zig_cuda / cuda); picking cpu or vulkan " ++
        "for it fails to load. Diffusion supports all four.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "LLM backend", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.Backend, .{ .choice = &cfg.llm_backend }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "KV cache dtype", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.KvDtype, .{ .choice = &cfg.kv_dtype }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }
    help("KV cache precision. f16 halves the KV-cache VRAM and q8_0 roughly " ++
        "quarters it (longer chats fit), at a small precision cost — output " ++
        "differs slightly from f32. Changing it rebuilds the context; the " ++
        "model weights stay loaded.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Checkpoint reads", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.WeightRead, .{ .choice = &cfg.weight_read }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }
    help("How model files are read. pread (default) maps the file but fetches " ++
        "weight bytes with positional reads, so a cold multi-GB checkpoint does " ++
        "not depend on kernel readahead — which stalls badly when RAM is short. " ++
        "mmap reads straight from the mapping. buffered loads the whole file " ++
        "into RAM up front: use it for checkpoints on ZFS, where mmap faulting " ++
        "can deadlock. Applies to the next model load.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Vision detail (Gemma 4)", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.VisionBudget, .{ .choice = &cfg.vision_budget }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }
    help("Gemma 4 vision token budget: how many tokens (spatial detail) the " ++
        "image tower spends per image. low=70 … high=280 (default) … max=1120. " ++
        "Higher = sharper detail but much more VRAM (ultra/max may not fit " ++
        "alongside a large model). Reloads the model (chat preserved). Ignored " ++
        "by other models.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        _ = dvui.checkbox(@src(), &cfg.gemma4_canonical_template, "Gemma 4 canonical chat template", .{ .gravity_y = 0.5 });
    }
    help("Use Google's current upstream Gemma 4 chat template instead of the " ++
        "one embedded in the model file. Some finetunes ship an older or altered " ++
        "template that can make multi-turn or reasoning chats misbehave; this " ++
        "renders exactly what Google's reference does. Reloads the model (chat " ++
        "preserved). Ignored by non-Gemma-4 models.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        _ = dvui.checkbox(@src(), &cfg.qwen35_fixed_template, "Qwen 3.5/3.6/3.8 fixed chat template", .{ .gravity_y = 0.5 });
    }
    help("Use froggeric's fixed Qwen chat template instead of the one embedded " ++
        "in the model file (on by default). It keeps earlier turns' thinking, so " ++
        "each new turn reuses the cached prompt instead of re-reading the whole " ++
        "chat; it drops the empty thought block the official templates emit; and " ++
        "it lets you turn thinking off on Qwen 3.8, which the official template " ++
        "refuses. Some quantized files ship no template at all, and this is what " ++
        "they get. Reloads the model (chat preserved). Ignored by other models, " ++
        "including plain Qwen3.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        _ = dvui.checkbox(@src(), &cfg.image_tool_result, "Report image results to the model", .{ .gravity_y = 0.5 });
    }
    help("When an image the model asked for finishes or fails, put a short note " ++
        "in the conversation before your next message, so the model knows what " ++
        "happened and can react (offering a smaller retry after running out of " ++
        "VRAM, for instance). Off makes the image tool fire-and-forget: the " ++
        "model asks, and never learns the outcome.");
    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 4 } });
        defer row.deinit();
        dvui.label(@src(), "Diffusion backend", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.Backend, .{ .choice = &cfg.diff_backend }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }

    section("LLM sampling");
    help("How the chat model picks each token. Changes apply on the NEXT reply " ++
        "(no reload). Temperature 0 is greedy; top-k / top-p / min-p trim the " ++
        "candidate pool. The penalties discourage tokens already in the recent " ++
        "window (llama.cpp semantics; while any penalty is active, GPU backends " ++
        "take a slower CPU-sampling path). Save the current values under a name " ++
        "to reuse them; presets persist with the settings on Apply.");
    numRow("Max response (0 = no limit)", &maxnew_buf);
    help("Longest single reply, in tokens. 0 lets a reply run until the model " ++
        "stops on its own or the context window fills. Not saved with presets " ++
        "below. Applies on the next reply.");
    presetRow(cfg);
    numRow("Temperature", &temp_buf);
    numRow("Top-k (0 = off)", &topk_buf);
    numRow("Top-p (1 = off)", &topp_buf);
    numRow("Min-p (0 = off)", &minp_buf);
    numRow("Repeat penalty (1 = off)", &rpen_buf);
    numRow("Penalty window (tokens)", &rlast_buf);
    numRow("Presence penalty (0 = off)", &ppen_buf);
    numRow("Frequency penalty (0 = off)", &fpen_buf);

    // Hidden outright for a checkpoint that cannot honor it, on the same
    // capability answer the composer uses (`app.noiseAvailable`, pushed here
    // because this view is handed a config, not a session). A greyed-out section
    // would still invite the question; an absent one does not.
    if (g_noise_supported) noiseSection(cfg);

    section("VRAM & performance");
    help("VRAM sharing between the chat model and image generation is controlled " ++
        "live from the meter in the status bar: drag the split handle to set each " ++
        "side's guaranteed share, and the limit handle to cap how much of the card " ++
        "is used at all. Both apply instantly — no reload — and persist here.");
    numRow("Regen cache (MB)", &regen_buf);
    help("Host RAM reserved for per-turn context checkpoints, which make " ++
        "\"regenerate response\" and variant switching instant at any context " ++
        "length. More MB keeps more past turns instantly rewindable; the newest " ++
        "turn's checkpoint is always kept, whatever the budget. Applies on the " ++
        "next reply.");

    section("System prompt");
    help("Sent to the model at the start of every conversation. When a diffusion " ++
        "model is configured, the image-tool instructions are appended automatically. " ++
        "Save named prompts and switch between them with the library row below.");
    sysPromptRow(cfg);
    {
        var row = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
        defer row.deinit();
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &cfg.system_prompt.data },
            .multiline = true,
            .scroll_horizontal = false,
            .break_lines = true,
        }, .{ .expand = .horizontal, .min_size_content = .{ .h = 90 }, .max_size_content = .height(220) });
        te.deinit();
    }
}

/// The system-prompt library row: a dropdown that loads a saved prompt into the
/// active editor, a name entry, and Save / Delete buttons operating on that
/// name. Mirrors `presetRow`. Saved prompts live in the Config (persisted on
/// Apply, discarded on Cancel), and "switching" is loading a different one as
/// the active `system_prompt` (which takes effect like any prompt edit).
fn sysPromptRow(cfg: *config.Config) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "Saved", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });

    // Loading copies the saved text into the active editor (and its name into
    // the entry, so Save/Delete target it); it lands via Apply like any edit.
    {
        var dd: dvui.DropdownWidget = undefined;
        dd.init(@src(), .{ .label = if (cfg.sys_prompts.count == 0) "none saved" else "Load…" }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 140 } });
        defer dd.deinit();
        if (dd.dropped()) {
            for (cfg.sys_prompts.slice()) |*sp| {
                if (dd.addChoiceLabel(sp.name.slice())) {
                    cfg.system_prompt.set(sp.text.slice());
                    _ = std.fmt.bufPrintZ(&sys_prompt_name_buf, "{s}", .{sp.name.slice()}) catch {};
                }
            }
        }
    }

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &sys_prompt_name_buf },
        .placeholder = "prompt name",
    }, .{ .expand = .horizontal, .gravity_y = 0.5, .margin = .{ .x = 4 } });
    te.deinit();

    if (dvui.button(@src(), "Save", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } })) {
        _ = cfg.upsertSysPrompt(std.mem.sliceTo(&sys_prompt_name_buf, 0), cfg.system_prompt.slice());
    }
    if (dvui.button(@src(), "Delete", .{}, .{ .gravity_y = 0.5 })) {
        _ = cfg.removeSysPromptNamed(std.mem.sliceTo(&sys_prompt_name_buf, 0));
    }
}

/// The sampling-preset row: a dropdown that loads a saved preset into the
/// controls, a name entry, and Save / Delete buttons operating on that name.
/// Presets live in the Config (persisted on Apply, discarded on Cancel, like
/// every other edit on this screen).
fn presetRow(cfg: *config.Config) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "Preset", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });

    // Loading a preset copies its values into the controls (and its name into
    // the entry, so Save/Delete target it); it still lands via Apply.
    {
        var dd: dvui.DropdownWidget = undefined;
        dd.init(@src(), .{ .label = if (cfg.presets.count == 0) "none saved" else "Load…" }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 140 } });
        defer dd.deinit();
        if (dd.dropped()) {
            for (cfg.presets.slice()) |*pr| {
                if (dd.addChoiceLabel(pr.name.slice())) {
                    cfg.sampling = pr.sampling;
                    _ = std.fmt.bufPrintZ(&preset_name_buf, "{s}", .{pr.name.slice()}) catch {};
                    seedSampling(cfg);
                }
            }
        }
    }

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &preset_name_buf },
        .placeholder = "preset name",
    }, .{ .expand = .horizontal, .gravity_y = 0.5, .margin = .{ .x = 4 } });
    te.deinit();

    if (dvui.button(@src(), "Save", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } })) {
        commitSampling(cfg);
        seedSampling(cfg); // reflect any clamping back into the controls
        _ = cfg.upsertPreset(std.mem.sliceTo(&preset_name_buf, 0), cfg.sampling);
    }
    if (dvui.button(@src(), "Delete", .{}, .{ .gravity_y = 0.5 })) {
        _ = cfg.removePresetNamed(std.mem.sliceTo(&preset_name_buf, 0));
    }
}

/// Whether the configured/loaded LLM honors weight noise. Set by the app each
/// frame before `render`; this view has no session to ask.
pub var g_noise_supported: bool = false;

/// The whole weight-noise section, hidden when nothing can honor it.
fn noiseSection(cfg: *config.Config) void {
    section("Weight noise");
    help("Perturbs the chat model's quantized weights WHILE it generates, so the " ++
        "model itself wobbles rather than the sampler: every 256-weight block's " ++
        "scale is scaled by 1 +- sigma, redrawn each forward pass. Temperature can " ++
        "only reweight the choices the weights already produced; this samples a " ++
        "nearby model, so the wording changes and stays grammatical. Off is exact. " ++
        "CUDA backends only, and today only Gemma 4 honors it.");
    help("Sigma is a CURVE over depth, written as an expression in t: 0 at the " ++
        "first layer, 1 at the LM head. WHERE the noise sits matters more than how " ++
        "much. The same amount late corrupts spelling; early it changes what the " ++
        "model is reasoning about and stays fluent, so a front-loaded curve takes " ++
        "roughly ten times the sigma a flat one survives. Push it far enough and " ++
        "the failure is structural, not lexical: clean sentences, broken plan. " ++
        "Available: t, l, n, abs sqrt exp log sin cos min max clamp pow, pi. There " ++
        "is no if; max() and clamp() gate instead. A bare number is a flat curve.");
    noiseCurveRow(cfg);
    curveField(cfg);
    numRow("Amount (the shape's a)", &noise_amt_buf);
    help("How much noise the shape carries, kept separate from the shape itself so " ++
        "the composer can adjust it without anyone editing an expression. 0.03 " ++
        "rephrases, 0.08 clearly rewords, 0.4 gives a different answer in clean " ++
        "prose, past ~0.8 a front-loaded shape breaks structurally. A shape written " ++
        "with a literal amplitude instead of `a` ignores this.");
    curvePreview(cfg);
}

var curve_name_buf: [config.max_preset_name]u8 = [_]u8{0} ** config.max_preset_name;

/// The noise-curve library row: load a saved curve into the active one, and
/// save/delete under a name. Mirrors `presetRow`.
fn noiseCurveRow(cfg: *config.Config) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "Curve", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
    {
        var dd: dvui.DropdownWidget = undefined;
        dd.init(@src(), .{ .label = if (cfg.noise_curves.count == 0) "none saved" else "Load…" }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 140 } });
        defer dd.deinit();
        if (dd.dropped()) {
            for (cfg.noise_curves.slice()) |*c| {
                if (dd.addChoiceLabel(c.name.slice())) {
                    cfg.weight_noise_curve = c.curve;
                    _ = std.fmt.bufPrintZ(&curve_name_buf, "{s}", .{c.name.slice()}) catch {};
                }
            }
        }
    }

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &curve_name_buf },
        .placeholder = "curve name",
    }, .{ .expand = .horizontal, .gravity_y = 0.5, .margin = .{ .x = 4 } });
    te.deinit();

    if (dvui.button(@src(), "Save", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } }))
        _ = cfg.upsertNoiseCurve(std.mem.sliceTo(&curve_name_buf, 0), cfg.weight_noise_curve.slice());
    if (dvui.button(@src(), "Delete", .{}, .{ .gravity_y = 0.5 }))
        _ = cfg.removeNoiseCurveNamed(std.mem.sliceTo(&curve_name_buf, 0));
}

/// The expression itself, bound straight to the config's own buffer (like the
/// path rows), and the on/off switch beside it. Full width, because this is the
/// one place a curve is meant to be written.
fn curveField(cfg: *config.Config) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();
    dvui.label(@src(), "Enabled", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
    _ = dvui.checkbox(@src(), &cfg.weight_noise, null, .{ .gravity_y = 0.5, .margin = .{ .w = 12 } });
    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &cfg.weight_noise_curve.data },
        .placeholder = "0.08*(1-t)^2",
    }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        .font = style.F.mono,
        // Amber while it does not parse: an unparseable curve is noise OFF, and
        // nothing else on screen would say so.
        .color_text = if (noise_curve.validate(cfg.weight_noise_curve.slice())) style.C.text_hi else |_| style.C.amber,
    });
    te.deinit();
}

/// The curve, drawn first-layer-to-LM-head, with the numbers that decide whether
/// it is the shape you meant: the peak, what reaches the head, and where it dies.
fn curvePreview(cfg: *config.Config) void {
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 2, .w = 4, .h = 8 } });
    defer col.deinit();

    const expr = cfg.weight_noise_curve.slice();
    const n = 48;
    var vals: [n]f32 = @splat(0);
    var peak: f32 = 0;
    var peak_at: usize = 0;
    var zero_from: ?usize = null;
    const ok = expr.len != 0 and blk: {
        noise_curve.validate(expr) catch break :blk false;
        break :blk true;
    };
    if (ok) for (&vals, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, n - 1);
        v.* = noise_curve.sanitize(noise_curve.eval(expr, .{ .t = t, .a = cfg.weight_noise_amount }) catch 0);
        if (v.* > peak) {
            peak = v.*;
            peak_at = i;
        }
        if (v.* == 0 and peak > 0 and zero_from == null) zero_from = i;
        if (v.* != 0) zero_from = null; // only a FINAL run of zeros counts
    };
    // A decaying curve reaching 0 exactly at the head is the ordinary case, and
    // "zero past 100%" says nothing; only a cutoff with real stack after it is
    // worth a line.
    if (zero_from) |z| if (z * 100 / (n - 1) > 95) {
        zero_from = null;
    };

    var norm = vals;
    if (peak > 0) for (&norm) |*v| {
        v.* /= peak;
    };
    // Bounded width, not full width: `sparkline` draws BARS, and 48 of them across
    // the whole form is a solid block you cannot read a shape off. At ~480px they
    // are thin enough to read as a curve.
    {
        var well = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .min_size_content = .{ .w = 480 },
            .max_size_content = .width(480),
        });
        defer well.deinit();
        style.sparkline(@src(), &norm, if (!ok) style.C.amber else if (cfg.weight_noise) style.C.blue else style.C.text_ghost, 40);
    }
    {
        var axis = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .min_size_content = .{ .w = 480 },
            .max_size_content = .width(480),
        });
        defer axis.deinit();
        dvui.labelNoFmt(@src(), "layer 0", .{}, .{ .font = style.F.mono_legend, .color_text = style.C.text_faint, .padding = .{} });
        _ = dvui.spacer(@src(), .{ .expand = .horizontal });
        dvui.labelNoFmt(@src(), "LM head", .{}, .{ .font = style.F.mono_legend, .color_text = style.C.text_faint, .padding = .{} });
    }

    var buf: [192]u8 = undefined;
    const pct = @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(peak_at)) / @as(f32, n - 1) * 100)));
    const txt = if (!ok)
        "does not parse, so noise is off"
    else if (peak == 0)
        "zero everywhere, so noise is off"
    else if (zero_from) |z| std.fmt.bufPrint(&buf, "peak {d:.3} at {d}% of the stack, zero past {d}%, {d:.3} at the LM head", .{
        peak, pct, @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(z)) / @as(f32, n - 1) * 100))), vals[n - 1],
    }) catch ""
    else if (!noise_curve.respondsToAmount(expr))
        std.fmt.bufPrint(&buf, "peak {d:.3} at {d}% of the stack, {d:.3} at the LM head (fixed: this shape ignores Amount)", .{ peak, pct, vals[n - 1] }) catch ""
    else
        std.fmt.bufPrint(&buf, "peak {d:.3} at {d}% of the stack, {d:.3} at the LM head", .{ peak, pct, vals[n - 1] }) catch "";
    dvui.labelNoFmt(@src(), txt, .{}, .{
        .font = style.F.mono_legend,
        .color_text = if (!ok) style.C.amber else style.C.text_faint,
        .padding = .{ .y = 4 },
    });
}

/// Enum dropdown for the TAESD preview size. Unlike `dvui.dropdownEnum` (which
/// shows raw tag names), this renders each option's human-readable `label()`
/// (e.g. "1/6 latent") since the fractions can't be spelled as enum tags.
/// Hand-rolled rather than `dvui.dropdownEnum` for the same reason as
/// `taesdSizeDropdown`: the enum's tag names (`dpmpp_2m_sde_heun`) are the CLI
/// spelling, not something to show a user.
fn promptSyntaxDropdown(choice: *config.PromptSyntax) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    defer dd.deinit();
    if (dd.dropped()) {
        inline for (@typeInfo(config.PromptSyntax).@"enum".fields) |e| {
            const opt: config.PromptSyntax = @field(config.PromptSyntax, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

fn compatDropdown(choice: *config.Compat) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 260 } });
    defer dd.deinit();
    if (dd.dropped()) {
        inline for (@typeInfo(config.Compat).@"enum".fields) |e| {
            const opt: config.Compat = @field(config.Compat, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

fn emphasisDropdown(choice: *config.Emphasis) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 260 } });
    defer dd.deinit();
    if (dd.dropped()) {
        inline for (@typeInfo(config.Emphasis).@"enum".fields) |e| {
            const opt: config.Emphasis = @field(config.Emphasis, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

fn samplerDropdown(choice: *config.Sampler) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    defer dd.deinit();

    if (dd.dropped()) {
        inline for (@typeInfo(config.Sampler).@"enum".fields) |e| {
            const opt: config.Sampler = @field(config.Sampler, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

/// Wider than the others: two entries carry a "(step count may differ)" warning,
/// because `ddim_uniform` and `beta` genuinely return a different number of steps than
/// the Default-steps box asks for.
fn schedulerDropdown(choice: *config.Scheduler) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 260 } });
    defer dd.deinit();

    if (dd.dropped()) {
        inline for (@typeInfo(config.Scheduler).@"enum".fields) |e| {
            const opt: config.Scheduler = @field(config.Scheduler, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

fn taesdSizeDropdown(choice: *config.TaesdSize) void {
    var dd: dvui.DropdownWidget = undefined;
    dd.init(@src(), .{
        .selected_index = @intFromEnum(choice.*),
        .label = choice.label(),
    }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    defer dd.deinit();

    if (dd.dropped()) {
        inline for (@typeInfo(config.TaesdSize).@"enum".fields) |e| {
            const opt: config.TaesdSize = @field(config.TaesdSize, e.name);
            if (dd.addChoiceLabel(opt.label())) choice.* = opt;
        }
    }
}

/// Widget id for a settings row, derived from its visible text. Every
/// section/help/row helper shares a single `@src()` across all its calls, so
/// `id_extra` is the only thing keeping them distinct, and the visible text
/// (section title, row label, help body) is already a unique natural key.
/// Hashing it beats hand-numbered ids (which collide silently on edit) and a
/// per-frame counter (whose ids shift when a row is reordered or becomes
/// conditional, detaching dvui's per-widget state).
fn idFor(text: []const u8) usize {
    return @truncate(std.hash.Wyhash.hash(0, text));
}

fn section(title: []const u8) void {
    dvui.label(@src(), "{s}", .{title}, .{ .id_extra = idFor(title), .font = .theme(.heading), .padding = .{ .x = 4, .y = 10, .h = 4 } });
}

fn help(text: []const u8) void {
    var tl = dvui.textLayout(@src(), .{}, .{ .id_extra = idFor(text), .expand = .horizontal, .padding = .{ .x = 4, .h = 6 } });
    defer tl.deinit();
    tl.addText(text, .{});
}

fn isDir(path: []const u8) bool {
    const io = g_io orelse return false;
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .directory;
}

/// Where a Browse dialog should open: the folder holding the row's current
/// value, so re-picking a model starts where the last one came from. Null means
/// SDL's own default (last used / cwd), which is also what a value that is not
/// under an existing directory gets, since the field is freely typed and may be
/// half-written. Every backend takes `default_location` as a FOLDER for an open
/// dialog (the xdg portal passes it straight through as `current_folder`), so a
/// file path must be reduced to its parent here.
fn startDir(buf: []u8, pb: *const config.PathBuf) ?[*:0]const u8 {
    const p = pb.slice();
    if (p.len == 0) return null;
    const dir = if (isDir(p)) p else (std.fs.path.dirname(p) orelse return null);
    if (dir.len == 0 or dir.len >= buf.len) return null;
    if (!isDir(dir)) return null;
    @memcpy(buf[0..dir.len], dir);
    buf[dir.len] = 0;
    return @ptrCast(buf.ptr);
}

fn pathRow(label: []const u8, pb: *config.PathBuf, filters: []const SDLBackend.c.SDL_DialogFileFilter) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = idFor(label), .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "{s}", .{label}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &pb.data },
        .placeholder = "not set",
    }, .{ .expand = .horizontal, .gravity_y = 0.5 });
    te.deinit();

    if (dvui.button(@src(), "Browse…", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } }) and !g_dialog_open) {
        g_dialog_open = true;
        // SDL copies default_location into its own property store, so this
        // stack buffer only has to live across the call.
        var dir_buf: [config.max_path]u8 = undefined;
        SDLBackend.c.SDL_ShowOpenFileDialog(
            dialogCallback,
            @ptrCast(pb),
            g_window,
            filters.ptr,
            @intCast(filters.len),
            startDir(&dir_buf, pb),
            false, // single selection
        );
    }
    if (dvui.button(@src(), "Clear", .{}, .{ .gravity_y = 0.5 })) {
        pb.set("");
    }
}

/// Like `pathRow`, but browses for a directory (native folder-select dialog).
fn dirRow(label: []const u8, pb: *config.PathBuf) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = idFor(label), .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "{s}", .{label}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &pb.data },
        .placeholder = "saving disabled",
    }, .{ .expand = .horizontal, .gravity_y = 0.5 });
    te.deinit();

    if (dvui.button(@src(), "Browse…", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } }) and !g_dialog_open) {
        g_dialog_open = true;
        var dir_buf: [config.max_path]u8 = undefined;
        SDLBackend.c.SDL_ShowOpenFolderDialog(
            dialogCallback,
            @ptrCast(pb),
            g_window,
            startDir(&dir_buf, pb),
            false, // single selection
        );
    }
    if (dvui.button(@src(), "Clear", .{}, .{ .gravity_y = 0.5 })) {
        pb.set("");
    }
}

// ── Checkpoint inspection panel ───────────────────────────────────────────────
// What the configured diffusion files actually are, shown under the path rows so
// the user finds out here rather than from a failed image. Purely informational:
// it never blocks Apply and never edits the config, `pipeline.Session.init` is
// the authority, and `model_spec` mirrors just enough of it to give a preview
// (see that module's advisory warning).

/// A `...` / `• ...` status line. Colored by severity so the two problem kinds
/// (something is missing, something will refuse to load) stand out from the
/// several ordinary "this is what you have" lines.
fn statusLine(text: []const u8, err: bool) void {
    const theme = dvui.themeGet();
    // `err.fill` is the theme's alarm color (what a dangerous button is painted
    // with); `err.text` is the text drawn ON it, so it would be invisible here.
    const color = if (err) (theme.err.fill orelse theme.text) else theme.text;
    var tl = dvui.textLayout(@src(), .{}, .{
        .id_extra = idFor(text),
        .expand = .horizontal,
        .padding = .{ .x = 12, .h = 2 },
        .color_text = color,
    });
    defer tl.deinit();
    tl.addText(text, .{});
}

/// Same, for a line that needs formatting. `buf` is the caller's scratch.
fn statusFmt(buf: []u8, comptime fmt: []const u8, args: anytype, err: bool) void {
    statusLine(std.fmt.bufPrint(buf, fmt, args) catch fmt, err);
}

fn checkpointPanel(cfg: *config.Config) void {
    if (!g_spec_ready) return;
    const path = cfg.diffusion_model.opt() orelse return;

    var buf: [512]u8 = undefined;
    const probe = g_primary_cache.primary(path);
    const info = switch (probe) {
        .unset => return,
        .failed => |err| {
            statusFmt(&buf, "could not read this checkpoint: {t}", .{err}, true);
            if (err == error.UnknownArchitecture) statusLine(
                "  Its tensor names match no architecture this build knows " ++
                    "(supported: krea2 flow-matching DiT, SD1.5 and SDXL UNets).",
                true,
            );
            return;
        },
        .ok => |i| i,
    };

    const t = model_spec.traits(info.family);
    statusFmt(&buf, "• Architecture: {s}", .{t.label}, false);

    // What the primary file carries, and what the overrides add. An override is
    // reported even when the checkpoint has its own copy, that is the case where
    // knowing which one wins actually matters.
    const te_set = cfg.text_encoder.opt();
    const te2_set = cfg.text_encoder_2.opt();
    const vae_set = cfg.vae.opt();
    if (info.isComplete()) {
        statusLine("• This checkpoint bundles every piece it needs — no other files required.", false);
    } else {
        statusFmt(&buf, "• This checkpoint contains: denoiser{s}{s}{s}", .{
            if (info.contents.conditioner) " + text encoder" else "",
            if (t.dual_conditioner and info.contents.conditioner2) " + text encoder 2" else "",
            if (info.contents.decoder) " + VAE" else "",
        }, false);
    }
    if (te_set != null and info.contents.conditioner)
        statusLine("• Text encoder override set: it replaces the one in the checkpoint.", false);
    if (te2_set != null and t.dual_conditioner and info.contents.conditioner2)
        statusLine("• Text encoder 2 override set: it replaces the one in the checkpoint.", false);
    // A second-tower path set for an architecture that has no second tower is
    // silently unused by the pipeline; say so rather than let it look effective.
    if (te2_set != null and !t.dual_conditioner)
        statusFmt(&buf, "• Text encoder 2 is ignored: {s} has a single text encoder.", .{t.label}, false);
    if (vae_set != null and info.contents.decoder)
        statusLine("• VAE override set: it replaces the one in the checkpoint.", false);

    const miss = model_spec.missing(info, .{
        .conditioner = te_set != null,
        .conditioner2 = te2_set != null,
        .decoder = vae_set != null,
    });
    if (miss.conditioner) statusLine("No text encoder: this checkpoint has none, and no override is set.", true);
    if (miss.conditioner2) statusLine(
        "No second text encoder: SDXL needs both CLIP towers, this checkpoint " ++
            "carries only the first, and no override is set.",
        true,
    );
    if (miss.decoder) statusLine("No VAE: this checkpoint has none, and no override is set.", true);

    // An override that does not hold what it is being asked for. This is the trap
    // that catches an upgrade in particular: switch the primary checkpoint to a
    // bundled SD1.5 file and leave krea2's encoder path set, and the resolver
    // (rightly) prefers the explicit file, opens krea2's qwen3 encoder, hunts for
    // CLIP's tensors in it and reports `ComponentNotInCheckpoint`, an error that
    // names neither the file nor the reason. Say it here instead.
    //
    // Reported, never enforced: silently ignoring a path the user set would be a
    // worse failure than a loud one, and the probe names are only a mirror of the
    // pipeline's (see model_spec's advisory note).
    if (te_set) |p| switch (g_te_cache.side(p, info.family)) {
        .failed => |err| statusFmt(&buf, "the text encoder override could not be read: {t}", .{err}, true),
        .ok => |i| if (!i.contents.conditioner) statusFmt(
            &buf,
            "the text encoder override holds no {s} text encoder. Clear it, or the load will fail.",
            .{t.label},
            true,
        ),
        .unset => {},
    };
    if (te2_set) |p| if (t.dual_conditioner) switch (g_te2_cache.side(p, info.family)) {
        .failed => |err| statusFmt(&buf, "the second text encoder override could not be read: {t}", .{err}, true),
        .ok => |i| if (!i.contents.conditioner2) statusFmt(
            &buf,
            "the second text encoder override holds no {s} OpenCLIP tower. Clear it, or the load will fail.",
            .{t.label},
            true,
        ),
        .unset => {},
    };
    if (vae_set) |p| switch (g_vae_cache.side(p, info.family)) {
        .failed => |err| statusFmt(&buf, "the VAE override could not be read: {t}", .{err}, true),
        .ok => |i| if (!i.contents.decoder) statusFmt(
            &buf,
            "the VAE override holds no {s} decoder. Clear it, or the load will fail.",
            .{t.label},
            true,
        ),
        .unset => {},
    };

    // Backend compatibility. Every architecture runs on every backend today, so
    // this normally says nothing, it stays because "which backends" is a
    // per-family fact (it was cpu-only for the SD family until its device kernels
    // landed) and the next architecture will arrive CPU-first.
    const want = diffuser.toPipelineBackend(cfg.diff_backend);
    if (!t.supports(want)) statusFmt(
        &buf,
        "{s} has no kernels for the {t} backend — pick another below, " ++
            "or generation will fail to load.",
        .{ t.label, cfg.diff_backend },
        true,
    );

    statusFmt(&buf, "• Suggested defaults for this architecture: {d}x{d}, {d} steps, CFG {d:.1}.", .{
        t.width, t.height, t.steps, t.cfg,
    }, false);
}

fn numRow(label: []const u8, buf: []u8) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = idFor(label), .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();
    dvui.label(@src(), "{s}", .{label}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
    // A single-line textEntry with no max_size_content is clamped to its
    // min_size_content, so the height MUST be set here or the box collapses
    // flat (dvui forces max = min for single-line entries).
    var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = buf } }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120, .h = 20 } });
    te.deinit();
}

