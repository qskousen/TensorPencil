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
    system_prompt: TextBuf(max_prompt) = TextBuf(max_prompt).lit(default_system_prompt),

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
        }
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
            \\system_prompt = {s}
            \\
        , .{
            self.llm_model.slice(),       self.vision_tower.slice(),
            self.diffusion_model.slice(), self.text_encoder.slice(),
            self.vae.slice(),             self.taesd.slice(),
            self.steps,                   self.width,
            self.height,                  @tagName(self.preview),
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
