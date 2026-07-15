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
pub const Priority = enum(u8) {
    chat,
    balanced,
    image,

    pub fn label(self: Priority) []const u8 {
        return switch (self) {
            .chat => "Chat (keep LLM resident)",
            .balanced => "Balanced (share VRAM, no shuffling)",
            .image => "Image generation",
        };
    }

    fn fromStr(s: []const u8) ?Priority {
        inline for (@typeInfo(Priority).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

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

pub const Config = struct {
    llm_model: PathBuf = .{},
    vision_tower: PathBuf = .{},
    diffusion_model: PathBuf = .{},
    text_encoder: PathBuf = .{},
    vae: PathBuf = .{},
    taesd: PathBuf = .{},
    steps: usize = 20,
    width: usize = 1024,
    height: usize = 1024,
    preview: Preview = .taesd,
    /// Max VRAM (GiB) the chat model may keep resident. 0 = auto (use the live
    /// free VRAM at load). As the conversation grows past this, LLM layers
    /// migrate to the CPU (see GUI_VRAM.md). Load-affecting: changing it reloads.
    max_vram_gib: f32 = 0,
    /// Who gets VRAM preference when chat and image generation compete. Default
    /// `balanced` — keep as much of both resident as fits, no per-image shuffling.
    vram_priority: Priority = .balanced,
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

    /// The LLM side (model, vision tower, VRAM limit) matches. A change here
    /// needs a reload — but a transcript-preserving one: the chat is carried
    /// across and replayed into the new model, never wiped.
    pub fn llmReloadEql(a: *const Config, b: *const Config) bool {
        return pathEql(&a.llm_model, &b.llm_model) and
            pathEql(&a.vision_tower, &b.vision_tower) and
            a.max_vram_gib == b.max_vram_gib and
            a.llm_backend == b.llm_backend;
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
        return cfg;
    }

    fn apply(self: *Config, key: []const u8, val: []const u8) void {
        if (std.mem.eql(u8, key, "llm_model")) self.llm_model.set(val) //
        else if (std.mem.eql(u8, key, "vision_tower")) self.vision_tower.set(val) //
        else if (std.mem.eql(u8, key, "diffusion_model")) self.diffusion_model.set(val) //
        else if (std.mem.eql(u8, key, "text_encoder")) self.text_encoder.set(val) //
        else if (std.mem.eql(u8, key, "vae")) self.vae.set(val) //
        else if (std.mem.eql(u8, key, "taesd")) self.taesd.set(val) //
        else if (std.mem.eql(u8, key, "system_prompt")) self.system_prompt.setUnescaped(val) //
        else if (std.mem.eql(u8, key, "steps")) {
            self.steps = std.fmt.parseInt(usize, val, 10) catch self.steps;
        } else if (std.mem.eql(u8, key, "width")) {
            self.width = std.fmt.parseInt(usize, val, 10) catch self.width;
        } else if (std.mem.eql(u8, key, "height")) {
            self.height = std.fmt.parseInt(usize, val, 10) catch self.height;
        } else if (std.mem.eql(u8, key, "preview")) {
            if (Preview.fromStr(val)) |p| self.preview = p;
        } else if (std.mem.eql(u8, key, "max_vram_gib")) {
            self.max_vram_gib = std.fmt.parseFloat(f32, val) catch self.max_vram_gib;
        } else if (std.mem.eql(u8, key, "vram_priority")) {
            if (Priority.fromStr(val)) |p| self.vram_priority = p;
        } else if (std.mem.eql(u8, key, "llm_backend")) {
            if (Backend.fromStr(val)) |b| self.llm_backend = b;
        } else if (std.mem.eql(u8, key, "diff_backend")) {
            if (Backend.fromStr(val)) |b| self.diff_backend = b;
        } else if (std.mem.eql(u8, key, "vae_decode")) {
            if (VaeDecode.fromStr(val)) |v| self.vae_decode = v;
        } else if (std.mem.eql(u8, key, "reasoning")) {
            self.reasoning = std.mem.eql(u8, val, "true");
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
            \\steps = {d}
            \\width = {d}
            \\height = {d}
            \\preview = {s}
            \\max_vram_gib = {d}
            \\vram_priority = {s}
            \\llm_backend = {s}
            \\diff_backend = {s}
            \\vae_decode = {s}
            \\reasoning = {}
            \\system_prompt = {s}
            \\
        , .{
            self.llm_model.slice(),       self.vision_tower.slice(),
            self.diffusion_model.slice(), self.text_encoder.slice(),
            self.vae.slice(),             self.taesd.slice(),
            self.steps,                   self.width,
            self.height,                  @tagName(self.preview),
            self.max_vram_gib,            @tagName(self.vram_priority),
            @tagName(self.llm_backend),   @tagName(self.diff_backend),
            @tagName(self.vae_decode),    self.reasoning,
            prompt_esc,
        });
        defer gpa.free(content);

        if (std.fs.path.dirname(path)) |dir| {
            std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = content });
    }
};

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

test "apply parses the reasoning flag (default on)" {
    var cfg: Config = .{};
    try std.testing.expect(cfg.reasoning); // on by default
    cfg.apply("reasoning", "false");
    try std.testing.expect(!cfg.reasoning);
    cfg.apply("reasoning", "true");
    try std.testing.expect(cfg.reasoning);
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

test "apply parses max_vram_gib and vram_priority" {
    var cfg: Config = .{};
    cfg.apply("max_vram_gib", "12.5");
    cfg.apply("vram_priority", "image");
    try std.testing.expectEqual(@as(f32, 12.5), cfg.max_vram_gib);
    try std.testing.expectEqual(Priority.image, cfg.vram_priority);
    cfg.apply("vram_priority", "bogus"); // unchanged on junk
    try std.testing.expectEqual(Priority.image, cfg.vram_priority);
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

test "llmReloadEql: LLM/vision/VRAM force reload; diff + live fields don't" {
    var a: Config = .{};
    var b: Config = .{};
    a.llm_model.set("/m.gguf");
    b.llm_model.set("/m.gguf");
    try std.testing.expect(a.llmReloadEql(&b));

    // Live-only + diffusion-path changes: LLM side still equal (no LLM reload).
    b.steps = 40;
    b.vram_priority = .image;
    b.vae.set("/vae.safetensors");
    b.diffusion_model.set("/dit.safetensors");
    try std.testing.expect(a.llmReloadEql(&b));
    try std.testing.expect(!a.diffPathsEql(&b)); // but the diff set differs

    // Vision tower change: LLM reload required.
    b.vision_tower.set("/mmproj.gguf");
    try std.testing.expect(!a.llmReloadEql(&b));
    b.vision_tower.set("");
    try std.testing.expect(a.llmReloadEql(&b));

    // VRAM-limit change: LLM reload required.
    b.max_vram_gib = 8;
    try std.testing.expect(!a.llmReloadEql(&b));
    b.max_vram_gib = 0;
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

test "diffEnabled requires dit + text-encoder + vae" {
    var c: Config = .{};
    try std.testing.expect(!c.diffEnabled());
    c.diffusion_model.set("/dit.safetensors");
    c.text_encoder.set("/te.safetensors");
    try std.testing.expect(!c.diffEnabled()); // vae still missing
    c.vae.set("/vae.safetensors");
    try std.testing.expect(c.diffEnabled());
}
