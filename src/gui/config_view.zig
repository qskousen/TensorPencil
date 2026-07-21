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
var steps_buf: [16]u8 = [_]u8{0} ** 16;
var width_buf: [16]u8 = [_]u8{0} ** 16;
var height_buf: [16]u8 = [_]u8{0} ** 16;
var regen_buf: [16]u8 = [_]u8{0} ** 16;
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

const gguf_filters = [_][]const u8{"*.gguf"};
const safetensors_filters = [_][]const u8{"*.safetensors"};

/// Call when the view is (re)entered so numeric buffers reseed from the config.
pub fn open() void {
    seeded = false;
}

fn seed(cfg: *const config.Config) void {
    _ = std.fmt.bufPrintZ(&steps_buf, "{d}", .{cfg.steps}) catch {};
    _ = std.fmt.bufPrintZ(&width_buf, "{d}", .{cfg.width}) catch {};
    _ = std.fmt.bufPrintZ(&height_buf, "{d}", .{cfg.height}) catch {};
    _ = std.fmt.bufPrintZ(&regen_buf, "{d}", .{cfg.regen_cache_mb}) catch {};
    seedSampling(cfg);
    seeded = true;
}

/// Reseed just the sampling buffers from the config — also used when a preset
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
    cfg.width = clampDim(parseNum(&width_buf, cfg.width));
    cfg.height = clampDim(parseNum(&height_buf, cfg.height));
    // Regen (checkpoint) cache: host RAM, capped at 64 GB to catch typos.
    cfg.regen_cache_mb = @min(parseNum(&regen_buf, cfg.regen_cache_mb), 64 << 10);
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

    // Header: title + footer actions (kept at top so they're always reachable).
    {
        var header = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = dvui.Rect.all(10) });
        defer header.deinit();
        dvui.label(@src(), "Settings", .{}, .{ .font = .theme(.title), .gravity_y = 0.5 });
        {
            var sp = dvui.box(@src(), .{}, .{ .expand = .horizontal });
            sp.deinit();
        }
        if (dvui.button(@src(), "Cancel", .{}, .{ .gravity_y = 0.5 })) cb.cancel();
        if (dvui.button(@src(), "Apply & Reload", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 6 } })) {
            commitNumbers(cfg);
            cb.apply();
        }
    }

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();
    var body = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = dvui.Rect.all(6) });
    defer body.deinit();

    section("Models");
    help("The LLM is required. Diffusion (image generation) needs all three of " ++
        "diffusion model, text encoder, and VAE. Vision (chatting about images) " ++
        "needs the vision tower. Any unset feature is simply disabled.");
    pathRow("LLM model", &cfg.llm_model, "GGUF models", &gguf_filters);
    pathRow("Vision tower", &cfg.vision_tower, "GGUF mmproj", &gguf_filters);
    pathRow("Diffusion model", &cfg.diffusion_model, "Safetensors", &safetensors_filters);
    pathRow("Text encoder", &cfg.text_encoder, "Safetensors", &safetensors_filters);
    pathRow("VAE", &cfg.vae, "Safetensors", &safetensors_filters);
    pathRow("TAESD preview", &cfg.taesd, "Safetensors", &safetensors_filters);

    section("Image generation");
    help("Generated images (chat and the image studio) are saved here as PNGs " ++
        "with AUTOMATIC1111-style metadata embedded. Defaults to a TensorPencil " ++
        "folder in your Pictures directory; clear it to disable saving.");
    dirRow("Output folder", &cfg.output_dir);
    numRow("Default steps", &steps_buf);
    numRow("Default width", &width_buf);
    numRow("Default height", &height_buf);

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
    presetRow(cfg);
    numRow("Temperature", &temp_buf);
    numRow("Top-k (0 = off)", &topk_buf);
    numRow("Top-p (1 = off)", &topp_buf);
    numRow("Min-p (0 = off)", &minp_buf);
    numRow("Repeat penalty (1 = off)", &rpen_buf);
    numRow("Penalty window (tokens)", &rlast_buf);
    numRow("Presence penalty (0 = off)", &ppen_buf);
    numRow("Frequency penalty (0 = off)", &fpen_buf);

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
        "model is configured, the image-tool instructions are appended automatically.");
    {
        var row = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
        defer row.deinit();
        var te = dvui.textEntry(@src(), .{
            .text = .{ .buffer = &cfg.system_prompt.data },
            .multiline = true,
        }, .{ .expand = .horizontal, .min_size_content = .{ .h = 90 }, .max_size_content = .height(220) });
        te.deinit();
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
        dd.init(@src(), .{ .label = if (cfg.preset_count == 0) "none saved" else "Load…" }, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 140 } });
        defer dd.deinit();
        if (dd.dropped()) {
            for (cfg.presets[0..cfg.preset_count]) |*pr| {
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

/// Enum dropdown for the TAESD preview size. Unlike `dvui.dropdownEnum` (which
/// shows raw tag names), this renders each option's human-readable `label()`
/// (e.g. "1/6 latent") since the fractions can't be spelled as enum tags.
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
/// `id_extra` is the only thing keeping them distinct — and the visible text
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

fn pathRow(label: []const u8, pb: *config.PathBuf, desc: []const u8, filters: []const []const u8) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = idFor(label), .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
    defer row.deinit();

    dvui.label(@src(), "{s}", .{label}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });

    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &pb.data },
        .placeholder = "not set",
    }, .{ .expand = .horizontal, .gravity_y = 0.5 });
    te.deinit();

    if (dvui.button(@src(), "Browse…", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } })) {
        pickFile(pb, label, desc, filters);
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

    if (dvui.button(@src(), "Browse…", .{}, .{ .gravity_y = 0.5, .margin = .{ .x = 4 } })) {
        const arena = dvui.currentWindow().arena();
        const chosen = dvui.dialogNativeFolderSelect(arena, .{
            .title = label,
            .path = pb.opt(),
        }) catch null;
        if (chosen) |p| pb.set(p);
    }
    if (dvui.button(@src(), "Clear", .{}, .{ .gravity_y = 0.5 })) {
        pb.set("");
    }
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

/// Open a blocking native file picker (tinyfiledialogs) and store the choice.
/// If the native dialog is unavailable it returns null and the field is left
/// as-is — the user can still type/paste a path into the text entry.
fn pickFile(pb: *config.PathBuf, title: []const u8, desc: []const u8, filters: []const []const u8) void {
    const arena = dvui.currentWindow().arena();
    const chosen = dvui.dialogNativeFileOpen(arena, .{
        .title = title,
        .path = pb.opt(),
        .filters = filters,
        .filter_description = desc,
    }) catch null;
    if (chosen) |p| pb.set(p);
}
