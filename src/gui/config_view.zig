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
    seeded = true;
}

fn parseNum(buf: []const u8, fallback: usize) usize {
    const s = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " \t\r");
    return std.fmt.parseInt(usize, s, 10) catch fallback;
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

    section(20, "Models");
    help(0, "The LLM is required. Diffusion (image generation) needs all three of " ++
        "diffusion model, text encoder, and VAE. Vision (chatting about images) " ++
        "needs the vision tower. Any unset feature is simply disabled.");
    pathRow(0, "LLM model", &cfg.llm_model, "GGUF models", &gguf_filters);
    pathRow(1, "Vision tower", &cfg.vision_tower, "GGUF mmproj", &gguf_filters);
    pathRow(2, "Diffusion model", &cfg.diffusion_model, "Safetensors", &safetensors_filters);
    pathRow(3, "Text encoder", &cfg.text_encoder, "Safetensors", &safetensors_filters);
    pathRow(4, "VAE", &cfg.vae, "Safetensors", &safetensors_filters);
    pathRow(5, "TAESD preview", &cfg.taesd, "Safetensors", &safetensors_filters);

    section(21, "Image generation");
    help(4, "Generated images (chat and the image studio) are saved here as PNGs " ++
        "with AUTOMATIC1111-style metadata embedded. Defaults to a TensorPencil " ++
        "folder in your Pictures directory; clear it to disable saving.");
    dirRow(6, "Output folder", &cfg.output_dir);
    numRow(10, "Default steps", &steps_buf);
    numRow(11, "Default width", &width_buf);
    numRow(12, "Default height", &height_buf);

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

    section(23, "Backends");
    help(3, "Compute backend for each engine. The two are independent — e.g. the " ++
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
        dvui.label(@src(), "Diffusion backend", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 150 } });
        _ = dvui.dropdownEnum(@src(), config.Backend, .{ .choice = &cfg.diff_backend }, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 200 } });
    }

    section(22, "VRAM & performance");
    help(2, "VRAM sharing between the chat model and image generation is controlled " ++
        "live from the meter in the status bar: drag the split handle to set each " ++
        "side's guaranteed share, and the limit handle to cap how much of the card " ++
        "is used at all. Both apply instantly — no reload — and persist here.");

    section(30, "System prompt");
    help(1, "Sent to the model at the start of every conversation. When a diffusion " ++
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

fn section(id: usize, title: []const u8) void {
    dvui.label(@src(), "{s}", .{title}, .{ .id_extra = id, .font = .theme(.heading), .padding = .{ .x = 4, .y = 10, .h = 4 } });
}

fn help(id: usize, text: []const u8) void {
    var tl = dvui.textLayout(@src(), .{}, .{ .id_extra = id, .expand = .horizontal, .padding = .{ .x = 4, .h = 6 } });
    defer tl.deinit();
    tl.addText(text, .{});
}

fn pathRow(id: usize, label: []const u8, pb: *config.PathBuf, desc: []const u8, filters: []const []const u8) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = id, .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
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
fn dirRow(id: usize, label: []const u8, pb: *config.PathBuf) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = id, .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
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

fn numRow(id: usize, label: []const u8, buf: []u8) void {
    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = id, .expand = .horizontal, .padding = .{ .x = 4, .y = 3 } });
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
