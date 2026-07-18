//! Persistent tp-gui settings: model paths and image-generation defaults.
//!
//! Stored as a tiny `key = value` text file under the platform config dir
//! (`~/.config/tp-gui/config` on Linux), resolved via the `known-folders`
//! package. Every setting has a sane default, so a missing or partial file
//! reads back cleanly — the GUI degrades gracefully when a model is unset
//! (see `PathBuf.opt`, which reports an empty path as "not configured").
//!
//! Path fields double as dvui text-entry buffers: the fixed `PathBuf.data`
//! array is what the config view binds to, so there is no separate edit state
//! to keep in sync.
const std = @import("std");
const known_folders = @import("known-folders");

const Environ = std.process.Environ.Map;

pub const app_dir = "tp-gui";
pub const file_name = "config";

/// Max path length for a model file (also the text-entry buffer capacity).
pub const max_path = 1024;
/// Capacity for the editable system prompt.
pub const max_prompt = 8192;

/// The simple built-in system prompt used when the user hasn't set one. The
/// image-tool instructions (in chat.zig) are appended to this automatically
/// when a diffusion model is configured — this stays tool-agnostic.
pub const default_system_prompt = "You are a helpful assistant.";

/// Which workload gets VRAM preference when chat and image generation compete
/// for the card (see GUI_VRAM.md). `chat` keeps the LLM resident and streams
/// diffusion in the leftover VRAM; `image` evicts LLM layers to the CPU (only as
/// many as needed) so the image model fits resident, then migrates them back
/// once the image queue drains.
/// Live-preview method for image generation.
pub const Preview = enum(u8) {
    none,
    latent2rgb,
    taesd,

    pub fn label(self: Preview) []const u8 {
        return switch (self) {
            .none => "None",
            .latent2rgb => "latent2rgb (fast)",
            .taesd => "TAESD (default)",
        };
    }

    fn fromStr(s: []const u8) ?Preview {
        inline for (@typeInfo(Preview).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// Resolution of the live TAESD preview, as a fraction of the latent grid. A
/// smaller fraction decodes faster (fewer pixels through the approx-VAE) but is
/// blurrier; `full` decodes the whole latent for the sharpest preview. Maps to
/// `pipeline.Options.preview_ds` (the latent-resolution divisor).
pub const TaesdSize = enum(u8) {
    sixth,
    quarter,
    half,
    full,

    pub fn label(self: TaesdSize) []const u8 {
        return switch (self) {
            .sixth => "1/6 latent (fastest)",
            .quarter => "1/4 latent",
            .half => "1/2 latent",
            .full => "Full latent (sharpest)",
        };
    }

    /// Latent-resolution divisor fed to the preview decode.
    pub fn divisor(self: TaesdSize) usize {
        return switch (self) {
            .sixth => 6,
            .quarter => 4,
            .half => 2,
            .full => 1,
        };
    }

    fn fromStr(s: []const u8) ?TaesdSize {
        inline for (@typeInfo(TaesdSize).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// Compute backend. Mirrors `pipeline.Backend`; kept here so the config data
/// model stays free of an engine dependency (app.zig maps it across). Diffusion
/// supports all four; the chat LLM only supports the two CUDA variants today
/// (non-CUDA selections error at load — see chat.Session.init).
pub const Backend = enum(u8) {
    cpu,
    vulkan,
    zig_cuda,
    cuda,

    fn fromStr(s: []const u8) ?Backend {
        inline for (@typeInfo(Backend).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// KV-cache element storage type for the chat LLM. `f32` is the default and
/// bit-exact; `f16` halves the KV footprint (VRAM), at a small precision cost
/// (output is not identical to f32). Mirrors `kv_cache.KvDtype` in the library;
/// mapped to it in `app.buildSession`. Changing it rebuilds the KV context
/// (weights stay resident) — see `ctxReloadEql`.
pub const KvDtype = enum(u8) {
    f32,
    f16,

    fn fromStr(s: []const u8) ?KvDtype {
        inline for (@typeInfo(KvDtype).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// VAE decode-path override for image generation. `auto` is the adaptive chain
/// (whole-image → GPU-tiled → CPU-tiled with OOM fallback); the others force the
/// starting strategy but still degrade gracefully on OOM (see pipeline.zig).
pub const VaeDecode = enum(u8) {
    auto,
    whole,
    gpu_tiled,
    cpu_tiled,

    fn fromStr(s: []const u8) ?VaeDecode {
        inline for (@typeInfo(VaeDecode).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// LLM sampling controls, mirroring the library's `llm.sample.Params` field
/// for field (kept engine-free here — `chat.samplingParams` maps it across).
/// Applied LIVE: a change (or a loaded preset) takes effect on the next chat
/// turn, never a reload.
pub const Sampling = struct {
    /// 0 = greedy; otherwise logits are divided by this.
    temperature: f32 = 0.7,
    /// Keep only the k highest logits (0 = no limit).
    top_k: usize = 20,
    /// Nucleus: smallest cumulative-probability prefix >= top_p (1 = off).
    top_p: f32 = 0.8,
    /// Drop candidates below min_p times the top candidate's probability (0 = off).
    min_p: f32 = 0.0,
    /// Divide a recently-seen token's positive logit by this (1 = off).
    repeat_penalty: f32 = 1.0,
    /// How many trailing context tokens the penalties look at (0 = off).
    repeat_last_n: usize = 64,
    /// Flat logit penalty for every token in the recent window (0 = off).
    presence_penalty: f32 = 0.0,
    /// Per-occurrence logit penalty (0 = off).
    frequency_penalty: f32 = 0.0,
};

/// Named sampling presets, stored inline in the config file (one `preset = ...`
/// line each, `name|temperature|top_k|top_p|min_p|repeat_penalty|repeat_last_n|
/// presence|frequency`). Fixed-capacity so the Config stays a plain value type.
pub const max_presets = 16;
pub const max_preset_name = 48;

pub const Preset = struct {
    name: TextBuf(max_preset_name) = .{},
    sampling: Sampling = .{},
};

/// A fixed-capacity, nul-terminated text buffer. `data` is handed directly to
/// dvui's `textEntry` as its backing store, so the buffer is the single source
/// of truth for the edited value (no separate edit state).
pub fn TextBuf(comptime cap: usize) type {
    return struct {
        const Self = @This();
        data: [cap]u8 = [_]u8{0} ** cap,

        /// Comptime constructor for a default value (asserts it fits).
        pub fn lit(comptime s: []const u8) Self {
            comptime std.debug.assert(s.len < cap);
            var b: Self = .{};
            @memcpy(b.data[0..s.len], s);
            return b;
        }

        /// The current text (up to the first nul).
        pub fn slice(self: *const Self) []const u8 {
            return std.mem.sliceTo(&self.data, 0);
        }

        /// The value, or null when empty ("not configured").
        pub fn opt(self: *const Self) ?[]const u8 {
            const s = self.slice();
            return if (s.len == 0) null else s;
        }

        /// Replace the contents (truncated to capacity, always nul-terminated).
        pub fn set(self: *Self, s: []const u8) void {
            @memset(&self.data, 0);
            const n = @min(s.len, self.data.len - 1);
            @memcpy(self.data[0..n], s[0..n]);
        }

        /// Like `set`, but decodes `\n` / `\\` escapes — used for the multi-line
        /// system prompt, which is stored escaped on a single config line.
        pub fn setUnescaped(self: *Self, s: []const u8) void {
            @memset(&self.data, 0);
            var i: usize = 0;
            var j: usize = 0;
            while (i < s.len and j + 1 < self.data.len) {
                if (s[i] == '\\' and i + 1 < s.len) {
                    switch (s[i + 1]) {
                        'n' => {
                            self.data[j] = '\n';
                            i += 2;
                            j += 1;
                            continue;
                        },
                        '\\' => {
                            self.data[j] = '\\';
                            i += 2;
                            j += 1;
                            continue;
                        },
                        else => {},
                    }
                }
                self.data[j] = s[i];
                i += 1;
                j += 1;
            }
        }
    };
}

pub const PathBuf = TextBuf(max_path);

/// Sentinel for an unsaved window position: SDL places the window itself (the
/// WM's default / centered) instead of us restoring a stored coordinate.
pub const pos_unset: i32 = std.math.minInt(i32);

pub const Config = struct {
    llm_model: PathBuf = .{},
    vision_tower: PathBuf = .{},
    diffusion_model: PathBuf = .{},
    text_encoder: PathBuf = .{},
    vae: PathBuf = .{},
    taesd: PathBuf = .{},
    /// Directory generated images are written to (chat + image studio). Empty
    /// means "not resolved"; `load` fills it with `<Pictures>/TensorPencil`
    /// (`~/Pictures/TensorPencil` on Linux) when unset, so the settings view
    /// shows a concrete default the user can edit. Empty at save-time disables
    /// saving.
    output_dir: PathBuf = .{},
    steps: usize = 20,
    width: usize = 1024,
    height: usize = 1024,
    preview: Preview = .taesd,
    /// Resolution of the live TAESD preview as a fraction of the latent grid.
    /// Applied live (no reload) like the preview method itself.
    taesd_size: TaesdSize = .quarter,
    /// VRAM meter policy as fractions of the whole card (the two draggable
    /// handles; they replace the old max-VRAM cap + priority toggle). `vram_split`
    /// is the LLM|diffusion contention boundary (the LLM's guaranteed share);
    /// `vram_limit_frac` is the ceiling nothing allocates past. Applied LIVE (a
    /// drag reshuffles residency on the fly) — never load-affecting.
    vram_split: f32 = 0.60,
    vram_limit_frac: f32 = 0.95,
    /// Compute backend for the chat LLM. Only the CUDA variants work today
    /// (non-CUDA errors cleanly at load). Changing it forces a reload.
    llm_backend: Backend = .zig_cuda,
    /// Compute backend for image generation. Independent of `llm_backend` — the
    /// two run as separate backend objects, so e.g. LLM on CUDA + diffusion on
    /// Vulkan is fine. Changing it rebuilds the diffusion session live.
    diff_backend: Backend = .zig_cuda,
    /// VAE decode-path override (see `VaeDecode`). Applied live like diff paths.
    vae_decode: VaeDecode = .auto,
    system_prompt: TextBuf(max_prompt) = TextBuf(max_prompt).lit(default_system_prompt),
    /// Whether the chat model reasons (emits a "thought" block) before its
    /// answer, for models that support it (see `chat.supportsThinking`). Applied
    /// live — the GUI toolbar toggle flips this without a reload; the block is
    /// rendered collapsed. No effect on non-reasoning models (e.g. Gemma 3).
    reasoning: bool = true,
    /// KV-cache element storage type (f32 default; f16 halves the KV VRAM
    /// footprint, lossy). Changing it rebuilds the KV context — the weights stay
    /// resident (see `ctxReloadEql`), not a full model reload.
    kv_dtype: KvDtype = .f32,
    /// LLM sampling controls (see `Sampling`). Applied live: pushed into the
    /// running session on Apply and picked up at the next turn — never load-
    /// or context-affecting (not compared by any `*ReloadEql`).
    sampling: Sampling = .{},
    /// Saved sampling presets (`presets[0..preset_count]`). Pure data the
    /// settings view loads/saves by name; persisted with everything else.
    presets: [max_presets]Preset = [_]Preset{.{}} ** max_presets,
    preset_count: usize = 0,

    /// Persisted window geometry for the main window and the image viewer,
    /// restored on the next launch. Size/position track the *restored* (non-
    /// maximized) geometry — while a window is maximized these keep the last
    /// unmaximized values so un-maximizing (and the next launch) lands sensibly.
    /// `*_x`/`*_y` are `pos_unset` until a window has been moved. These are pure
    /// view state: never load- or reload-affecting (not compared by any
    /// `*ReloadEql`), and saved on change directly (see app.captureGeom).
    win_w: usize = 1100,
    win_h: usize = 820,
    win_x: i32 = pos_unset,
    win_y: i32 = pos_unset,
    win_max: bool = false,
    viewer_w: usize = 1000,
    viewer_h: usize = 760,
    viewer_x: i32 = pos_unset,
    viewer_y: i32 = pos_unset,
    viewer_max: bool = false,

    /// The LLM side (model, vision tower, VRAM limit) matches. A change here
    /// needs a reload — but a transcript-preserving one: the chat is carried
    /// across and replayed into the new model, never wiped.
    pub fn llmReloadEql(a: *const Config, b: *const Config) bool {
        return pathEql(&a.llm_model, &b.llm_model) and
            pathEql(&a.vision_tower, &b.vision_tower) and
            a.llm_backend == b.llm_backend;
    }

    /// The LLM's KV-cache CONTEXT config matches. A change here (currently just
    /// `kv_dtype`) needs a context rebuild — the KV cache is re-allocated at the
    /// new dtype and the transcript re-prefilled — but the model WEIGHTS stay
    /// resident (no full `llmReloadEql` reload). Kept separate from
    /// `llmReloadEql` precisely so a dtype flip doesn't reload multi-GB weights.
    pub fn ctxReloadEql(a: *const Config, b: *const Config) bool {
        return a.kv_dtype == b.kv_dtype;
    }

    /// The diffusion model set (dit/text-encoder/vae/taesd) matches. A change
    /// while the image tool stays enabled applies live (deferred until the image
    /// queue drains); no LLM reload.
    pub fn diffPathsEql(a: *const Config, b: *const Config) bool {
        return pathEql(&a.diffusion_model, &b.diffusion_model) and
            pathEql(&a.text_encoder, &b.text_encoder) and
            pathEql(&a.vae, &b.vae) and
            pathEql(&a.taesd, &b.taesd);
    }

    /// The full diffusion config (paths + backend + decode path) matches. A
    /// change while the tool stays enabled rebuilds the diffusion session live
    /// (deferred until the image queue drains); no LLM reload.
    pub fn diffReloadEql(a: *const Config, b: *const Config) bool {
        return diffPathsEql(a, b) and
            a.diff_backend == b.diff_backend and
            a.vae_decode == b.vae_decode;
    }

    /// Find a saved preset by name (name is cleaned the same way `upsertPreset`
    /// cleans it, so lookups match what was stored).
    pub fn findPreset(self: *const Config, raw_name: []const u8) ?usize {
        var buf: [max_preset_name]u8 = undefined;
        const name = cleanPresetName(raw_name, &buf) orelse return null;
        for (self.presets[0..self.preset_count], 0..) |*p, i| {
            if (std.mem.eql(u8, p.name.slice(), name)) return i;
        }
        return null;
    }

    /// Save `s` under `raw_name` (whitespace-trimmed; the reserved '|' and
    /// newlines are dropped), replacing an existing preset of the same name.
    /// Returns false when the name is empty after cleaning or the table is full.
    pub fn upsertPreset(self: *Config, raw_name: []const u8, s: Sampling) bool {
        var buf: [max_preset_name]u8 = undefined;
        const name = cleanPresetName(raw_name, &buf) orelse return false;
        if (self.findPreset(name)) |i| {
            self.presets[i].sampling = s;
            return true;
        }
        if (self.preset_count >= max_presets) return false;
        self.presets[self.preset_count] = .{ .sampling = s };
        self.presets[self.preset_count].name.set(name);
        self.preset_count += 1;
        return true;
    }

    /// Remove the preset named `raw_name` (cleaned like `upsertPreset`).
    /// Returns whether one was removed.
    pub fn removePresetNamed(self: *Config, raw_name: []const u8) bool {
        const i = self.findPreset(raw_name) orelse return false;
        std.mem.copyForwards(Preset, self.presets[i .. self.preset_count - 1], self.presets[i + 1 .. self.preset_count]);
        self.preset_count -= 1;
        self.presets[self.preset_count] = .{};
        return true;
    }

    /// Whether the image-generation tool is enabled: dit + text-encoder + VAE
    /// must all be set (matches app.buildSession). Toggling this needs a reload
    /// (the system prompt gains/loses the image-tool instructions).
    pub fn diffEnabled(self: *const Config) bool {
        return self.diffusion_model.opt() != null and
            self.text_encoder.opt() != null and
            self.vae.opt() != null;
    }

    /// Resolve the config directory (`<config>/tp-gui`); caller frees. Null if
    /// the platform has no known config location.
    fn dirPath(io: std.Io, gpa: std.mem.Allocator, environ: *const Environ) !?[]u8 {
        const base = (try known_folders.getPath(io, gpa, environ, .local_configuration)) orelse return null;
        defer gpa.free(base);
        return try std.fs.path.join(gpa, &.{ base, app_dir });
    }

    /// Resolve the full config file path. An explicit `override` (e.g. from
    /// `--config`) is used verbatim; otherwise it's `<platform config>/tp-gui/
    /// config`. Null only when there's no override and no known config dir.
    /// Caller frees.
    fn filePath(io: std.Io, gpa: std.mem.Allocator, environ: *const Environ, override: ?[]const u8) !?[]u8 {
        if (override) |p| return try gpa.dupe(u8, p);
        const dir = (try dirPath(io, gpa, environ)) orelse return null;
        defer gpa.free(dir);
        return try std.fs.path.join(gpa, &.{ dir, file_name });
    }

    /// The default image-output directory: `<Pictures>/TensorPencil`
    /// (`~/Pictures/TensorPencil` on Linux). Null if the platform has no known
    /// pictures location. Caller frees.
    fn defaultOutputDir(io: std.Io, gpa: std.mem.Allocator, environ: *const Environ) !?[]u8 {
        const base = (try known_folders.getPath(io, gpa, environ, .pictures)) orelse return null;
        defer gpa.free(base);
        return try std.fs.path.join(gpa, &.{ base, "TensorPencil" });
    }

    /// Load settings from disk. A missing file (or missing keys) yields
    /// defaults; a malformed line is skipped rather than failing the load.
    /// `path_override` bypasses the well-known location (used by `--config`).
    pub fn load(io: std.Io, gpa: std.mem.Allocator, environ: *const Environ, path_override: ?[]const u8) Config {
        var cfg: Config = .{};
        const path = (filePath(io, gpa, environ, path_override) catch return cfg) orelse return cfg;
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 * 1024)) catch return cfg;
        defer gpa.free(bytes);

        var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
        while (lines.next()) |line| {
            const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
            const key = std.mem.trim(u8, line[0..eq], " \t\r");
            const val = std.mem.trim(u8, line[eq + 1 ..], " \t\r");
            cfg.apply(key, val);
        }
        // Fill an unset output dir with the platform default so the settings
        // view shows a concrete, editable path (best-effort; left empty if the
        // platform has no known data location, which disables image saving).
        if (cfg.output_dir.opt() == null) {
            if (defaultOutputDir(io, gpa, environ) catch null) |dir| {
                defer gpa.free(dir);
                cfg.output_dir.set(dir);
            }
        }
        return cfg;
    }

    fn apply(self: *Config, key: []const u8, val: []const u8) void {
        if (std.mem.eql(u8, key, "llm_model")) self.llm_model.set(val) //
        else if (std.mem.eql(u8, key, "vision_tower")) self.vision_tower.set(val) //
        else if (std.mem.eql(u8, key, "diffusion_model")) self.diffusion_model.set(val) //
        else if (std.mem.eql(u8, key, "text_encoder")) self.text_encoder.set(val) //
        else if (std.mem.eql(u8, key, "vae")) self.vae.set(val) //
        else if (std.mem.eql(u8, key, "taesd")) self.taesd.set(val) //
        else if (std.mem.eql(u8, key, "output_dir")) self.output_dir.set(val) //
        else if (std.mem.eql(u8, key, "system_prompt")) self.system_prompt.setUnescaped(val) //
        else if (std.mem.eql(u8, key, "steps")) {
            self.steps = std.fmt.parseInt(usize, val, 10) catch self.steps;
        } else if (std.mem.eql(u8, key, "width")) {
            self.width = std.fmt.parseInt(usize, val, 10) catch self.width;
        } else if (std.mem.eql(u8, key, "height")) {
            self.height = std.fmt.parseInt(usize, val, 10) catch self.height;
        } else if (std.mem.eql(u8, key, "preview")) {
            if (Preview.fromStr(val)) |p| self.preview = p;
        } else if (std.mem.eql(u8, key, "taesd_size")) {
            if (TaesdSize.fromStr(val)) |t| self.taesd_size = t;
        } else if (std.mem.eql(u8, key, "vram_split")) {
            self.vram_split = std.math.clamp(std.fmt.parseFloat(f32, val) catch self.vram_split, 0, 1);
        } else if (std.mem.eql(u8, key, "vram_limit_frac")) {
            self.vram_limit_frac = std.math.clamp(std.fmt.parseFloat(f32, val) catch self.vram_limit_frac, 0, 1);
        } else if (std.mem.eql(u8, key, "llm_backend")) {
            if (Backend.fromStr(val)) |b| self.llm_backend = b;
        } else if (std.mem.eql(u8, key, "diff_backend")) {
            if (Backend.fromStr(val)) |b| self.diff_backend = b;
        } else if (std.mem.eql(u8, key, "vae_decode")) {
            if (VaeDecode.fromStr(val)) |v| self.vae_decode = v;
        } else if (std.mem.eql(u8, key, "reasoning")) {
            self.reasoning = std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, key, "kv_dtype")) {
            if (KvDtype.fromStr(val)) |d| self.kv_dtype = d;
        } else if (std.mem.eql(u8, key, "temperature")) {
            self.sampling.temperature = std.fmt.parseFloat(f32, val) catch self.sampling.temperature;
        } else if (std.mem.eql(u8, key, "top_k")) {
            self.sampling.top_k = std.fmt.parseInt(usize, val, 10) catch self.sampling.top_k;
        } else if (std.mem.eql(u8, key, "top_p")) {
            self.sampling.top_p = std.fmt.parseFloat(f32, val) catch self.sampling.top_p;
        } else if (std.mem.eql(u8, key, "min_p")) {
            self.sampling.min_p = std.fmt.parseFloat(f32, val) catch self.sampling.min_p;
        } else if (std.mem.eql(u8, key, "repeat_penalty")) {
            self.sampling.repeat_penalty = std.fmt.parseFloat(f32, val) catch self.sampling.repeat_penalty;
        } else if (std.mem.eql(u8, key, "repeat_last_n")) {
            self.sampling.repeat_last_n = std.fmt.parseInt(usize, val, 10) catch self.sampling.repeat_last_n;
        } else if (std.mem.eql(u8, key, "presence_penalty")) {
            self.sampling.presence_penalty = std.fmt.parseFloat(f32, val) catch self.sampling.presence_penalty;
        } else if (std.mem.eql(u8, key, "frequency_penalty")) {
            self.sampling.frequency_penalty = std.fmt.parseFloat(f32, val) catch self.sampling.frequency_penalty;
        } else if (std.mem.eql(u8, key, "preset")) {
            if (parsePreset(val)) |pr| {
                if (self.preset_count < max_presets) {
                    self.presets[self.preset_count] = pr;
                    self.preset_count += 1;
                }
            }
        } else if (std.mem.eql(u8, key, "win_w")) {
            self.win_w = std.fmt.parseInt(usize, val, 10) catch self.win_w;
        } else if (std.mem.eql(u8, key, "win_h")) {
            self.win_h = std.fmt.parseInt(usize, val, 10) catch self.win_h;
        } else if (std.mem.eql(u8, key, "win_x")) {
            self.win_x = std.fmt.parseInt(i32, val, 10) catch self.win_x;
        } else if (std.mem.eql(u8, key, "win_y")) {
            self.win_y = std.fmt.parseInt(i32, val, 10) catch self.win_y;
        } else if (std.mem.eql(u8, key, "win_max")) {
            self.win_max = std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, key, "viewer_w")) {
            self.viewer_w = std.fmt.parseInt(usize, val, 10) catch self.viewer_w;
        } else if (std.mem.eql(u8, key, "viewer_h")) {
            self.viewer_h = std.fmt.parseInt(usize, val, 10) catch self.viewer_h;
        } else if (std.mem.eql(u8, key, "viewer_x")) {
            self.viewer_x = std.fmt.parseInt(i32, val, 10) catch self.viewer_x;
        } else if (std.mem.eql(u8, key, "viewer_y")) {
            self.viewer_y = std.fmt.parseInt(i32, val, 10) catch self.viewer_y;
        } else if (std.mem.eql(u8, key, "viewer_max")) {
            self.viewer_max = std.mem.eql(u8, val, "true");
        }
    }

    fn pathEql(a: *const PathBuf, b: *const PathBuf) bool {
        return std.mem.eql(u8, a.slice(), b.slice());
    }

    /// Serialize settings to disk (creating the parent dir if needed).
    /// `path_override` bypasses the well-known location (used by `--config`).
    pub fn save(self: *const Config, io: std.Io, gpa: std.mem.Allocator, environ: *const Environ, path_override: ?[]const u8) !void {
        const path = (try filePath(io, gpa, environ, path_override)) orelse return error.NoConfigDir;
        defer gpa.free(path);

        // The system prompt is multi-line; escape it onto one line so the
        // simple key=value format holds.
        const prompt_esc = try escapeAlloc(gpa, self.system_prompt.slice());
        defer gpa.free(prompt_esc);

        const content = try std.fmt.allocPrint(gpa,
            \\llm_model = {s}
            \\vision_tower = {s}
            \\diffusion_model = {s}
            \\text_encoder = {s}
            \\vae = {s}
            \\taesd = {s}
            \\output_dir = {s}
            \\steps = {d}
            \\width = {d}
            \\height = {d}
            \\preview = {s}
            \\taesd_size = {s}
            \\vram_split = {d}
            \\vram_limit_frac = {d}
            \\llm_backend = {s}
            \\diff_backend = {s}
            \\vae_decode = {s}
            \\reasoning = {}
            \\kv_dtype = {s}
            \\win_w = {d}
            \\win_h = {d}
            \\win_x = {d}
            \\win_y = {d}
            \\win_max = {}
            \\viewer_w = {d}
            \\viewer_h = {d}
            \\viewer_x = {d}
            \\viewer_y = {d}
            \\viewer_max = {}
            \\system_prompt = {s}
            \\
        , .{
            self.llm_model.slice(),       self.vision_tower.slice(),
            self.diffusion_model.slice(), self.text_encoder.slice(),
            self.vae.slice(),             self.taesd.slice(),
            self.output_dir.slice(),      self.steps,
            self.width,
            self.height,                  @tagName(self.preview),
            @tagName(self.taesd_size),    self.vram_split,
            self.vram_limit_frac,
            @tagName(self.llm_backend),   @tagName(self.diff_backend),
            @tagName(self.vae_decode),    self.reasoning,
            @tagName(self.kv_dtype),
            self.win_w,                   self.win_h,
            self.win_x,                   self.win_y,
            self.win_max,                 self.viewer_w,
            self.viewer_h,                self.viewer_x,
            self.viewer_y,                self.viewer_max,
            prompt_esc,
        });
        defer gpa.free(content);

        // Sampling + presets are appended after the fixed template — sampling
        // to stay under the 32-arg format limit, presets because they're
        // variable-count (load order doesn't matter: every line is `key = value`).
        var full: std.ArrayList(u8) = .empty;
        defer full.deinit(gpa);
        try full.appendSlice(gpa, content);
        const sampling_lines = try std.fmt.allocPrint(gpa,
            \\temperature = {d}
            \\top_k = {d}
            \\top_p = {d}
            \\min_p = {d}
            \\repeat_penalty = {d}
            \\repeat_last_n = {d}
            \\presence_penalty = {d}
            \\frequency_penalty = {d}
            \\
        , .{
            self.sampling.temperature,    self.sampling.top_k,
            self.sampling.top_p,          self.sampling.min_p,
            self.sampling.repeat_penalty, self.sampling.repeat_last_n,
            self.sampling.presence_penalty,
            self.sampling.frequency_penalty,
        });
        defer gpa.free(sampling_lines);
        try full.appendSlice(gpa, sampling_lines);
        for (self.presets[0..self.preset_count]) |*pr| {
            const line = try std.fmt.allocPrint(gpa, "preset = {s}|{d}|{d}|{d}|{d}|{d}|{d}|{d}|{d}\n", .{
                pr.name.slice(),            pr.sampling.temperature,
                pr.sampling.top_k,          pr.sampling.top_p,
                pr.sampling.min_p,          pr.sampling.repeat_penalty,
                pr.sampling.repeat_last_n,  pr.sampling.presence_penalty,
                pr.sampling.frequency_penalty,
            });
            defer gpa.free(line);
            try full.appendSlice(gpa, line);
        }

        if (std.fs.path.dirname(path)) |dir| {
            std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = full.items });
    }
};

/// Clean a preset name for storage/lookup: trim whitespace and drop the
/// reserved '|' separator (and CR/LF). Returns a slice into `buf`, or null
/// when nothing printable is left.
fn cleanPresetName(raw: []const u8, buf: *[max_preset_name]u8) ?[]const u8 {
    var n: usize = 0;
    for (std.mem.trim(u8, raw, " \t\r\n")) |ch| {
        if (ch == '|' or ch == '\n' or ch == '\r') continue;
        if (n >= max_preset_name - 1) break; // TextBuf needs a trailing nul
        buf[n] = ch;
        n += 1;
    }
    return if (n == 0) null else buf[0..n];
}

/// Parse one `preset = name|t|k|p|minp|rp|rln|pp|fp` value. Null (line skipped)
/// on any malformed field, so a hand-edited config can't half-apply a preset.
fn parsePreset(val: []const u8) ?Preset {
    var it = std.mem.splitScalar(u8, val, '|');
    var buf: [max_preset_name]u8 = undefined;
    const name = cleanPresetName(it.next() orelse return null, &buf) orelse return null;
    var pr: Preset = .{};
    pr.name.set(name);
    const s = &pr.sampling;
    inline for (.{ "temperature", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty" }) |field| {
        const raw = std.mem.trim(u8, it.next() orelse return null, " \t");
        const F = @TypeOf(@field(s, field));
        @field(s, field) = switch (@typeInfo(F)) {
            .float => std.fmt.parseFloat(F, raw) catch return null,
            else => std.fmt.parseInt(F, raw, 10) catch return null,
        };
    }
    return pr;
}

/// Escape backslashes and newlines so a multi-line value survives on one
/// `key = value` line (`\` → `\\`, newline → `\n`, CR dropped). Caller frees.
fn escapeAlloc(gpa: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (s) |ch| switch (ch) {
        '\\' => try out.appendSlice(gpa, "\\\\"),
        '\n' => try out.appendSlice(gpa, "\\n"),
        '\r' => {},
        else => try out.append(gpa, ch),
    };
    return out.toOwnedSlice(gpa);
}

test "PathBuf opt/set round-trips and reports empty as null" {
    var p: PathBuf = .{};
    try std.testing.expect(p.opt() == null);
    p.set("/models/foo.gguf");
    try std.testing.expectEqualStrings("/models/foo.gguf", p.opt().?);
    p.set("");
    try std.testing.expect(p.opt() == null);
}

test "Config.apply parses keys and tolerates junk" {
    var cfg: Config = .{};
    cfg.apply("llm_model", "/a/b.gguf");
    cfg.apply("steps", "30");
    cfg.apply("preview", "latent2rgb");
    cfg.apply("width", "not-a-number");
    cfg.apply("bogus", "ignored");
    try std.testing.expectEqualStrings("/a/b.gguf", cfg.llm_model.opt().?);
    try std.testing.expectEqual(@as(usize, 30), cfg.steps);
    try std.testing.expectEqual(Preview.latent2rgb, cfg.preview);
    try std.testing.expectEqual(@as(usize, 1024), cfg.width); // unchanged on junk
}

test "apply parses taesd_size and maps to a latent divisor" {
    var cfg: Config = .{};
    try std.testing.expectEqual(TaesdSize.quarter, cfg.taesd_size); // default
    try std.testing.expectEqual(@as(usize, 4), cfg.taesd_size.divisor());

    cfg.apply("taesd_size", "sixth");
    try std.testing.expectEqual(TaesdSize.sixth, cfg.taesd_size);
    try std.testing.expectEqual(@as(usize, 6), cfg.taesd_size.divisor());

    cfg.apply("taesd_size", "full");
    try std.testing.expectEqual(@as(usize, 1), cfg.taesd_size.divisor());

    cfg.apply("taesd_size", "garbage"); // unchanged on junk
    try std.testing.expectEqual(TaesdSize.full, cfg.taesd_size);
}

test "apply parses the reasoning flag (default on)" {
    var cfg: Config = .{};
    try std.testing.expect(cfg.reasoning); // on by default
    cfg.apply("reasoning", "false");
    try std.testing.expect(!cfg.reasoning);
    cfg.apply("reasoning", "true");
    try std.testing.expect(cfg.reasoning);
}

test "apply parses kv_dtype (default f32) and ctxReloadEql tracks it" {
    var cfg: Config = .{};
    try std.testing.expectEqual(KvDtype.f32, cfg.kv_dtype); // default
    cfg.apply("kv_dtype", "f16");
    try std.testing.expectEqual(KvDtype.f16, cfg.kv_dtype);
    cfg.apply("kv_dtype", "garbage"); // unchanged on junk
    try std.testing.expectEqual(KvDtype.f16, cfg.kv_dtype);

    // ctxReloadEql: false only when kv_dtype differs (weights stay resident).
    const base: Config = .{};
    try std.testing.expect(base.ctxReloadEql(&base));
    try std.testing.expect(!cfg.ctxReloadEql(&base));
    // A kv_dtype change must NOT trip llmReloadEql (that would reload weights).
    try std.testing.expect(cfg.llmReloadEql(&base));
}

test "system prompt escapes/unescapes newlines round-trip" {
    const gpa = std.testing.allocator;
    var b: TextBuf(max_prompt) = .{};
    b.set("line one\nline two\\end");
    const esc = try escapeAlloc(gpa, b.slice());
    defer gpa.free(esc);
    try std.testing.expectEqualStrings("line one\\nline two\\\\end", esc);

    var back: TextBuf(max_prompt) = .{};
    back.setUnescaped(esc);
    try std.testing.expectEqualStrings("line one\nline two\\end", back.slice());
}

test "default system prompt is populated on a fresh Config" {
    const cfg: Config = .{};
    try std.testing.expectEqualStrings(default_system_prompt, cfg.system_prompt.slice());
}

test "apply parses vram meter fractions, clamps to [0,1], tolerates junk" {
    var cfg: Config = .{};
    cfg.apply("vram_split", "0.5");
    cfg.apply("vram_limit_frac", "0.8");
    try std.testing.expectEqual(@as(f32, 0.5), cfg.vram_split);
    try std.testing.expectEqual(@as(f32, 0.8), cfg.vram_limit_frac);
    cfg.apply("vram_limit_frac", "1.7"); // clamped to 1
    try std.testing.expectEqual(@as(f32, 1.0), cfg.vram_limit_frac);
    cfg.apply("vram_split", "junk"); // unchanged on junk
    try std.testing.expectEqual(@as(f32, 0.5), cfg.vram_split);
}

test "apply parses backends and vae_decode, tolerates junk" {
    var cfg: Config = .{};
    // Defaults.
    try std.testing.expectEqual(Backend.zig_cuda, cfg.llm_backend);
    try std.testing.expectEqual(Backend.zig_cuda, cfg.diff_backend);
    try std.testing.expectEqual(VaeDecode.auto, cfg.vae_decode);

    cfg.apply("llm_backend", "cuda");
    cfg.apply("diff_backend", "vulkan");
    cfg.apply("vae_decode", "cpu_tiled");
    try std.testing.expectEqual(Backend.cuda, cfg.llm_backend);
    try std.testing.expectEqual(Backend.vulkan, cfg.diff_backend);
    try std.testing.expectEqual(VaeDecode.cpu_tiled, cfg.vae_decode);

    // Unrecognized values leave the field unchanged.
    cfg.apply("diff_backend", "bogus");
    cfg.apply("vae_decode", "sideways");
    try std.testing.expectEqual(Backend.vulkan, cfg.diff_backend);
    try std.testing.expectEqual(VaeDecode.cpu_tiled, cfg.vae_decode);
}

test "save/load round-trips the new backend + decode fields" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // `path_override` is used verbatim and bypasses the known-folders lookup, so
    // a plain relative file in cwd exercises the real save/load template path.
    const file = ".gui-config-roundtrip-test";
    defer std.Io.Dir.cwd().deleteFile(io, file) catch {};

    var environ: Environ = .init(gpa); // unused with an override; just needs to exist
    defer environ.deinit();

    var a: Config = .{};
    a.llm_model.set("/m.gguf");
    a.llm_backend = .cuda;
    a.diff_backend = .vulkan;
    a.vae_decode = .gpu_tiled;
    try a.save(io, gpa, &environ, file);

    const b = Config.load(io, gpa, &environ, file);
    try std.testing.expectEqual(Backend.cuda, b.llm_backend);
    try std.testing.expectEqual(Backend.vulkan, b.diff_backend);
    try std.testing.expectEqual(VaeDecode.gpu_tiled, b.vae_decode);
    try std.testing.expectEqualStrings("/m.gguf", b.llm_model.opt().?);
}

test "window geometry round-trips (size, position, maximized)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const file = ".gui-config-geom-test";
    defer std.Io.Dir.cwd().deleteFile(io, file) catch {};

    var environ: Environ = .init(gpa);
    defer environ.deinit();

    // Defaults: position unset, not maximized.
    var a: Config = .{};
    try std.testing.expectEqual(pos_unset, a.win_x);
    try std.testing.expect(!a.win_max);

    a.win_w = 1440;
    a.win_h = 900;
    a.win_x = -30; // negative coords are valid on multi-monitor setups
    a.win_y = 12;
    a.win_max = true;
    a.viewer_w = 800;
    a.viewer_h = 600;
    a.viewer_x = 100;
    a.viewer_y = 200;
    try a.save(io, gpa, &environ, file);

    const b = Config.load(io, gpa, &environ, file);
    try std.testing.expectEqual(@as(usize, 1440), b.win_w);
    try std.testing.expectEqual(@as(usize, 900), b.win_h);
    try std.testing.expectEqual(@as(i32, -30), b.win_x);
    try std.testing.expectEqual(@as(i32, 12), b.win_y);
    try std.testing.expect(b.win_max);
    try std.testing.expectEqual(@as(usize, 800), b.viewer_w);
    try std.testing.expectEqual(@as(usize, 600), b.viewer_h);
    try std.testing.expectEqual(@as(i32, 100), b.viewer_x);
    try std.testing.expectEqual(@as(i32, 200), b.viewer_y);
    try std.testing.expect(!b.viewer_max); // untouched → default
}

test "output_dir round-trips and apply parses it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const file = ".gui-config-outdir-test";
    defer std.Io.Dir.cwd().deleteFile(io, file) catch {};

    var environ: Environ = .init(gpa);
    defer environ.deinit();

    var a: Config = .{};
    a.output_dir.set("/tmp/my images");
    try a.save(io, gpa, &environ, file);

    // With an explicit output_dir set, load leaves it as-is (no default fill).
    const b = Config.load(io, gpa, &environ, file);
    try std.testing.expectEqualStrings("/tmp/my images", b.output_dir.opt().?);

    var c: Config = .{};
    c.apply("output_dir", "/data/out");
    try std.testing.expectEqualStrings("/data/out", c.output_dir.opt().?);
}

test "llmReloadEql: LLM/vision force reload; diff + live fields (incl. VRAM meter) don't" {
    var a: Config = .{};
    var b: Config = .{};
    a.llm_model.set("/m.gguf");
    b.llm_model.set("/m.gguf");
    try std.testing.expect(a.llmReloadEql(&b));

    // Live-only + diffusion-path changes: LLM side still equal (no LLM reload).
    // The VRAM meter fractions apply live (offload/promote on the fly), so a
    // change to them must NOT force a reload.
    b.steps = 40;
    b.vram_split = 0.4;
    b.vram_limit_frac = 0.8;
    b.vae.set("/vae.safetensors");
    b.diffusion_model.set("/dit.safetensors");
    try std.testing.expect(a.llmReloadEql(&b));
    try std.testing.expect(!a.diffPathsEql(&b)); // but the diff set differs

    // Vision tower change: LLM reload required.
    b.vision_tower.set("/mmproj.gguf");
    try std.testing.expect(!a.llmReloadEql(&b));
    b.vision_tower.set("");
    try std.testing.expect(a.llmReloadEql(&b));

    // LLM backend change: LLM reload required.
    b.llm_backend = .cuda;
    try std.testing.expect(!a.llmReloadEql(&b));
}

test "diffReloadEql: diff paths, backend, and decode all trigger a diff rebuild" {
    var a: Config = .{};
    var b: Config = .{};
    try std.testing.expect(a.diffReloadEql(&b));

    // Diffusion-model path change.
    b.diffusion_model.set("/dit.safetensors");
    try std.testing.expect(!a.diffReloadEql(&b));
    b = a;

    // Diffusion backend change (but LLM side unaffected).
    b.diff_backend = .vulkan;
    try std.testing.expect(!a.diffReloadEql(&b));
    try std.testing.expect(a.llmReloadEql(&b));
    b = a;

    // VAE decode-path change.
    b.vae_decode = .cpu_tiled;
    try std.testing.expect(!a.diffReloadEql(&b));
    try std.testing.expect(a.llmReloadEql(&b));
}

test "apply parses sampling keys and tolerates junk" {
    var cfg: Config = .{};
    // Defaults mirror llm.sample.Params.
    try std.testing.expectEqual(@as(f32, 0.7), cfg.sampling.temperature);
    try std.testing.expectEqual(@as(usize, 20), cfg.sampling.top_k);
    try std.testing.expectEqual(@as(f32, 0.8), cfg.sampling.top_p);
    try std.testing.expectEqual(@as(f32, 0.0), cfg.sampling.min_p);
    try std.testing.expectEqual(@as(f32, 1.0), cfg.sampling.repeat_penalty);
    try std.testing.expectEqual(@as(usize, 64), cfg.sampling.repeat_last_n);

    cfg.apply("temperature", "1.2");
    cfg.apply("top_k", "40");
    cfg.apply("top_p", "0.95");
    cfg.apply("min_p", "0.05");
    cfg.apply("repeat_penalty", "1.1");
    cfg.apply("repeat_last_n", "128");
    cfg.apply("presence_penalty", "0.5");
    cfg.apply("frequency_penalty", "-0.25");
    try std.testing.expectEqual(@as(f32, 1.2), cfg.sampling.temperature);
    try std.testing.expectEqual(@as(usize, 40), cfg.sampling.top_k);
    try std.testing.expectEqual(@as(f32, 0.95), cfg.sampling.top_p);
    try std.testing.expectEqual(@as(f32, 0.05), cfg.sampling.min_p);
    try std.testing.expectEqual(@as(f32, 1.1), cfg.sampling.repeat_penalty);
    try std.testing.expectEqual(@as(usize, 128), cfg.sampling.repeat_last_n);
    try std.testing.expectEqual(@as(f32, 0.5), cfg.sampling.presence_penalty);
    try std.testing.expectEqual(@as(f32, -0.25), cfg.sampling.frequency_penalty);

    cfg.apply("temperature", "hot"); // unchanged on junk
    try std.testing.expectEqual(@as(f32, 1.2), cfg.sampling.temperature);

    // Sampling is a LIVE setting: neither a weight reload nor a KV rebuild.
    const base: Config = .{};
    try std.testing.expect(cfg.llmReloadEql(&base));
    try std.testing.expect(cfg.ctxReloadEql(&base));
}

test "preset lines parse, tolerate junk, and cap at max_presets" {
    var cfg: Config = .{};
    cfg.apply("preset", "creative|1.2|40|0.95|0.05|1.1|128|0.5|0.25");
    try std.testing.expectEqual(@as(usize, 1), cfg.preset_count);
    try std.testing.expectEqualStrings("creative", cfg.presets[0].name.slice());
    try std.testing.expectEqual(@as(f32, 1.2), cfg.presets[0].sampling.temperature);
    try std.testing.expectEqual(@as(usize, 128), cfg.presets[0].sampling.repeat_last_n);
    try std.testing.expectEqual(@as(f32, 0.25), cfg.presets[0].sampling.frequency_penalty);

    // Whitespace around fields tolerated (hand-edited config).
    cfg.apply("preset", " precise | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 ");
    try std.testing.expectEqual(@as(usize, 2), cfg.preset_count);
    try std.testing.expectEqualStrings("precise", cfg.presets[1].name.slice());
    try std.testing.expectEqual(@as(f32, 0), cfg.presets[1].sampling.temperature);

    // Malformed lines are skipped whole (no half-applied preset).
    cfg.apply("preset", "broken|1.0|notanumber|1|0|1|64|0|0");
    cfg.apply("preset", "tooshort|1.0|20");
    cfg.apply("preset", "|1.0|20|1|0|1|64|0|0"); // empty name
    try std.testing.expectEqual(@as(usize, 2), cfg.preset_count);

    // The table caps at max_presets; extra lines are dropped.
    for (0..max_presets) |i| {
        var name_buf: [max_preset_name]u8 = undefined;
        const line = std.fmt.bufPrint(&name_buf, "p{d}|1|20|1|0|1|64|0|0", .{i}) catch unreachable;
        cfg.apply("preset", line);
    }
    try std.testing.expectEqual(@as(usize, max_presets), cfg.preset_count);
}

test "upsertPreset adds, replaces by name, sanitizes; removePresetNamed removes" {
    var cfg: Config = .{};
    try std.testing.expect(cfg.upsertPreset("creative", .{ .temperature = 1.3 }));
    try std.testing.expect(cfg.upsertPreset(" pipe|name \n", .{ .temperature = 0.2 }));
    try std.testing.expectEqual(@as(usize, 2), cfg.preset_count);
    try std.testing.expectEqualStrings("pipename", cfg.presets[1].name.slice());

    // Same name replaces in place (no duplicate).
    try std.testing.expect(cfg.upsertPreset("creative", .{ .temperature = 1.5 }));
    try std.testing.expectEqual(@as(usize, 2), cfg.preset_count);
    try std.testing.expectEqual(@as(f32, 1.5), cfg.presets[0].sampling.temperature);

    // Empty / all-reserved names are rejected.
    try std.testing.expect(!cfg.upsertPreset("  ", .{}));
    try std.testing.expect(!cfg.upsertPreset("|||", .{}));

    try std.testing.expect(cfg.findPreset("creative") != null);
    try std.testing.expect(cfg.removePresetNamed("creative"));
    try std.testing.expect(!cfg.removePresetNamed("creative")); // already gone
    try std.testing.expectEqual(@as(usize, 1), cfg.preset_count);
    try std.testing.expectEqualStrings("pipename", cfg.presets[0].name.slice());
}

test "sampling + presets save/load round-trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const file = ".gui-config-sampling-test";
    defer std.Io.Dir.cwd().deleteFile(io, file) catch {};

    var environ: Environ = .init(gpa);
    defer environ.deinit();

    var a: Config = .{};
    a.sampling = .{
        .temperature = 1.1,
        .top_k = 0,
        .top_p = 0.92,
        .min_p = 0.07,
        .repeat_penalty = 1.15,
        .repeat_last_n = 256,
        .presence_penalty = 0.4,
        .frequency_penalty = -0.1,
    };
    try std.testing.expect(a.upsertPreset("creative", .{ .temperature = 1.4, .top_p = 0.98 }));
    try std.testing.expect(a.upsertPreset("greedy", .{ .temperature = 0, .top_k = 1 }));
    try a.save(io, gpa, &environ, file);

    const b = Config.load(io, gpa, &environ, file);
    try std.testing.expectEqual(a.sampling, b.sampling);
    try std.testing.expectEqual(@as(usize, 2), b.preset_count);
    try std.testing.expectEqualStrings("creative", b.presets[0].name.slice());
    try std.testing.expectEqual(a.presets[0].sampling, b.presets[0].sampling);
    try std.testing.expectEqualStrings("greedy", b.presets[1].name.slice());
    try std.testing.expectEqual(a.presets[1].sampling, b.presets[1].sampling);
}

test "diffEnabled requires dit + text-encoder + vae" {
    var c: Config = .{};
    try std.testing.expect(!c.diffEnabled());
    c.diffusion_model.set("/dit.safetensors");
    c.text_encoder.set("/te.safetensors");
    try std.testing.expect(!c.diffEnabled()); // vae still missing
    c.vae.set("/vae.safetensors");
    try std.testing.expect(c.diffEnabled());
}
