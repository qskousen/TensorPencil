//! How a pick becomes the EFFECTIVE model paths the engine reads.
//!
//! The chips and Settings choose by architecture and file; the engine, the reload
//! checks and the diffuser only ever see `Config.llm_model`, `.vision_tower`,
//! `.diffusion_model`, `.text_encoder`, `.text_encoder_2`, `.vae` and `.taesd`.
//! This module is the one place that writes those from a choice, so the rest of
//! the GUI never learned that folders exist.
//!
//! Side files are remembered per diffusion FAMILY and vision towers per LLM CLASS
//! (`Config.family_sides`, `Config.class_towers`). A remembered slot holds a path,
//! `choice_none`, or nothing. Nothing means "follow the checkpoint": use the
//! bundled component when the file has one, else the first compatible file in the
//! catalog, which is then remembered. Picking "bundled" in a dropdown clears the
//! slot back to nothing, so the two read the same and a later checkpoint without
//! the component still gets a file.
//!
//! A configured file the catalog does not know (chosen through "other file…",
//! or a folder since removed) is left exactly as configured. Only what the
//! catalog can see is resolved; nothing here ever blanks a path it cannot judge.
const std = @import("std");
const config = @import("config.zig");
const catalog = @import("catalog.zig");
const model_spec = @import("model_spec.zig");

pub const Family = catalog.Family;
pub const Component = catalog.Component;
pub const Catalog = catalog.Catalog;
pub const Config = config.Config;

/// The side slots a diffusion checkpoint takes.
pub const Slot = enum {
    text_encoder,
    text_encoder_2,
    vae,
    taesd,

    /// The pipeline component the slot supplies; null for the preview decoder,
    /// which the pipeline resolves outside `componentSpec`.
    pub fn component(self: Slot) ?Component {
        return switch (self) {
            .text_encoder => .conditioner,
            .text_encoder_2 => .conditioner2,
            .vae => .decoder,
            .taesd => null,
        };
    }

    /// The effective path field.
    pub fn field(self: Slot, cfg: *Config) *config.PathBuf {
        return switch (self) {
            .text_encoder => &cfg.text_encoder,
            .text_encoder_2 => &cfg.text_encoder_2,
            .vae => &cfg.vae,
            .taesd => &cfg.taesd,
        };
    }

    /// The remembered choice for a family.
    pub fn mem(self: Slot, m: *config.FamilySides) *config.PathBuf {
        return switch (self) {
            .text_encoder => &m.text_encoder,
            .text_encoder_2 => &m.text_encoder_2,
            .vae => &m.vae,
            .taesd => &m.taesd,
        };
    }

    /// Whether the family has this slot at all: only SDXL has a second tower.
    pub fn applies(self: Slot, fam: Family) bool {
        return switch (self) {
            .text_encoder_2 => model_spec.traits(fam).dual_conditioner,
            else => true,
        };
    }

    pub fn label(self: Slot) []const u8 {
        return switch (self) {
            .text_encoder => "Text encoder",
            .text_encoder_2 => "Text encoder 2",
            .vae => "VAE",
            .taesd => "Preview decoder",
        };
    }

    pub const all = [_]Slot{ .text_encoder, .text_encoder_2, .vae, .taesd };
};

pub fn familyKey(fam: Family) []const u8 {
    return @tagName(fam);
}

/// "gemma4|5376": what a vision tower has to match, so the memory is per width,
/// not per file.
pub fn classKey(buf: []u8, l: catalog.Llm) []const u8 {
    return std.fmt.bufPrint(buf, "{s}|{d}", .{ l.arch, l.width }) catch buf[0..0];
}

// ── Diffusion ───────────────────────────────────────────────────────────────

/// The configured checkpoint's catalog entry, when the catalog knows it.
pub fn checkpoint(cfg: *const Config, cat: *const Catalog) ?*const catalog.Entry {
    const e = cat.find(cfg.diffusion_model.slice()) orelse return null;
    return if (e.ckpt != null) e else null;
}

/// A switch never carries the old checkpoint's side files over: the slots are
/// cleared and refilled from the new family's memory.
pub fn selectCheckpoint(cfg: *Config, cat: *const Catalog, path: []const u8) void {
    cfg.diffusion_model.set(path);
    for (Slot.all) |s| s.field(cfg).set("");
    resolveSides(cfg, cat);
}

pub fn clearCheckpoint(cfg: *Config) void {
    cfg.diffusion_model.set("");
    for (Slot.all) |s| s.field(cfg).set("");
}

/// Record `choice` for the checkpoint's family and re-resolve. `choice` is a
/// path, `config.choice_none`, or `config.choice_bundled` (stored as nothing,
/// see the module doc).
pub fn chooseSide(cfg: *Config, cat: *const Catalog, slot: Slot, choice: []const u8) void {
    const e = checkpoint(cfg, cat) orelse return;
    const m = cfg.familySidesMut(familyKey(e.ckpt.?.family)) orelse return;
    slot.mem(m).set(if (std.mem.eql(u8, choice, config.choice_bundled)) "" else choice);
    slot.field(cfg).set(""); // a choice replaces the slot; nothing to adopt
    resolveSides(cfg, cat);
}

/// Fill the four effective side paths for the configured checkpoint from the
/// family's memory and the catalog.
pub fn resolveSides(cfg: *Config, cat: *const Catalog) void {
    const e = checkpoint(cfg, cat) orelse return;
    const ck = e.ckpt.?;
    const m = cfg.familySidesMut(familyKey(ck.family)) orelse return;
    for (Slot.all) |slot| {
        const field = slot.field(cfg);
        if (!slot.applies(ck.family)) {
            field.set("");
            continue;
        }
        const bundled = if (slot.component()) |c| ck.contents.has(c) else false;
        // A slot already holding a path with nothing remembered is a setup from
        // before the memory existed (or from a scan that only now recognizes the
        // checkpoint): adopt it rather than replace it. A switch cleared the slots
        // first, so this never carries another family's file over.
        if (slot.mem(m).slice().len == 0 and field.slice().len > 0) slot.mem(m).set(field.slice());
        const remembered = slot.mem(m).slice();
        if (std.mem.eql(u8, remembered, config.choice_none)) {
            field.set("");
        } else if (remembered.len > 0) {
            field.set(remembered);
        } else if (bundled) {
            field.set("");
        } else if (firstCandidate(cat, ck.family, slot)) |i| {
            field.set(cat.entries[i].path);
            slot.mem(m).set(cat.entries[i].path);
        } else {
            field.set("");
        }
    }
}

fn firstCandidate(cat: *const Catalog, fam: Family, slot: Slot) ?usize {
    return if (slot.component()) |c| cat.firstSide(fam, c) else cat.firstPreview(fam);
}

/// What a slot is set to right now, for a dropdown to show.
pub const SideState = union(enum) {
    /// Using the checkpoint's own copy.
    bundled,
    /// Nothing supplies it (chosen, or nothing compatible exists).
    none,
    /// An explicit file.
    file: []const u8,
};

pub fn sideState(cfg: *const Config, cat: *const Catalog, slot: Slot) SideState {
    const path = @constCast(slot.field(@constCast(cfg))).slice();
    if (path.len > 0) return .{ .file = path };
    if (checkpoint(cfg, cat)) |e| if (slot.component()) |c| if (e.ckpt.?.contents.has(c)) return .bundled;
    return .none;
}

// ── LLM ─────────────────────────────────────────────────────────────────────

pub fn llm(cfg: *const Config, cat: *const Catalog) ?*const catalog.Entry {
    const e = cat.find(cfg.llm_model.slice()) orelse return null;
    return if (e.llm != null) e else null;
}

pub fn selectLlm(cfg: *Config, cat: *const Catalog, path: []const u8) void {
    cfg.llm_model.set(path);
    cfg.vision_tower.set("");
    resolveTower(cfg, cat);
}

pub fn clearLlm(cfg: *Config) void {
    cfg.llm_model.set("");
    cfg.vision_tower.set("");
}

/// Record `choice` (a path or `config.choice_none`) for the configured LLM's
/// class and re-resolve.
pub fn chooseTower(cfg: *Config, cat: *const Catalog, choice: []const u8) void {
    const e = llm(cfg, cat) orelse return;
    var kbuf: [config.max_class_key]u8 = undefined;
    cfg.rememberClassTower(classKey(&kbuf, e.llm.?), choice);
    resolveTower(cfg, cat);
}

/// Fill the effective vision tower for the configured LLM: none for a text-only
/// architecture, the remembered one for its class, else the first that fits.
pub fn resolveTower(cfg: *Config, cat: *const Catalog) void {
    const e = llm(cfg, cat) orelse return;
    const l = e.llm.?;
    if (!l.vision) {
        cfg.vision_tower.set("");
        return;
    }
    var kbuf: [config.max_class_key]u8 = undefined;
    const key = classKey(&kbuf, l);
    // Same adoption as the side slots: a tower already set with no memory for
    // the class is the user's existing pairing, kept as is.
    if (cfg.classTower(key) == null and cfg.vision_tower.opt() != null) cfg.rememberClassTower(key, cfg.vision_tower.slice());
    const remembered = cfg.classTower(key) orelse "";
    if (std.mem.eql(u8, remembered, config.choice_none)) {
        cfg.vision_tower.set("");
    } else if (remembered.len > 0) {
        cfg.vision_tower.set(remembered);
    } else if (cat.firstTower(cat.indexOf(e.path).?)) |i| {
        cfg.vision_tower.set(cat.entries[i].path);
        cfg.rememberClassTower(key, cat.entries[i].path);
    } else {
        cfg.vision_tower.set("");
    }
}

/// After a rescan: whatever the catalog now knows about, re-resolve.
pub fn resolveAll(cfg: *Config, cat: *const Catalog) void {
    resolveSides(cfg, cat);
    resolveTower(cfg, cat);
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

fn ckptEntry(path: []const u8, fam: Family, contents: model_spec.Contents) catalog.Entry {
    return .{ .path = path, .size = 1, .mtime_ns = 1, .ckpt = .{ .family = fam, .contents = contents } };
}

fn sideEntry(path: []const u8, fits: []const struct { Family, Component }) catalog.Entry {
    var e: catalog.Entry = .{ .path = path, .size = 1, .mtime_ns = 1 };
    for (fits) |f| e.side.set(f[0], f[1]);
    return e;
}

fn previewEntry(path: []const u8, fams: []const Family) catalog.Entry {
    var e: catalog.Entry = .{ .path = path, .size = 1, .mtime_ns = 1 };
    for (fams) |f| e.preview.fams[@intFromEnum(f)] = true;
    return e;
}

fn llmEntry(path: []const u8, arch: []const u8, width: u64, vision: bool) catalog.Entry {
    return .{ .path = path, .size = 1, .mtime_ns = 1, .llm = .{
        .arch = arch,
        .size_label = "x",
        .width = width,
        .blocks = 1,
        .supported = true,
        .vision = vision,
        .class = "c",
    } };
}

fn towerEntry(path: []const u8, arch: []const u8, width: u64) catalog.Entry {
    return .{ .path = path, .size = 1, .mtime_ns = 1, .tower = .{ .projector = "p", .arch = arch, .width = width } };
}

fn diffusionCatalog(gpa: std.mem.Allocator) !Catalog {
    return Catalog.fromEntries(gpa, &.{
        ckptEntry("/m/krea2/a.safetensors", .krea2, .{ .denoiser = true }),
        ckptEntry("/m/krea2/b.safetensors", .krea2, .{ .denoiser = true }),
        ckptEntry("/m/sdxl/bundled.safetensors", .sdxl, .{ .denoiser = true, .conditioner = true, .conditioner2 = true, .decoder = true }),
        ckptEntry("/m/sdxl/unet-only.gguf", .sdxl, .{ .denoiser = true }),
        sideEntry("/m/te/qwen3vl-4b.safetensors", &.{.{ .krea2, .conditioner }}),
        sideEntry("/m/vae/wan-a.safetensors", &.{ .{ .krea2, .decoder }, .{ .anima, .decoder } }),
        sideEntry("/m/vae/wan-b.safetensors", &.{ .{ .krea2, .decoder }, .{ .anima, .decoder } }),
        sideEntry("/m/vae/sdxl.vae.safetensors", &.{ .{ .sdxl, .decoder }, .{ .sd15, .decoder } }),
        previewEntry("/m/vae_approx/taew2_1.safetensors", &.{ .krea2, .anima }),
    });
}

test "a denoiser-only checkpoint gets the first compatible file per slot, and remembers it" {
    const gpa = testing.allocator;
    var cat = try diffusionCatalog(gpa);
    defer cat.deinit();
    var cfg: Config = .{};

    selectCheckpoint(&cfg, &cat, "/m/krea2/a.safetensors");
    try testing.expectEqualStrings("/m/te/qwen3vl-4b.safetensors", cfg.text_encoder.slice());
    try testing.expectEqualStrings("/m/vae/wan-a.safetensors", cfg.vae.slice());
    try testing.expectEqualStrings("/m/vae_approx/taew2_1.safetensors", cfg.taesd.slice());
    try testing.expectEqualStrings("", cfg.text_encoder_2.slice()); // not an SDXL slot
    try testing.expectEqualStrings("/m/vae/wan-a.safetensors", cfg.familySides("krea2").?.vae.slice());

    // The other Krea2 file keeps the same sides; a different VAE pick sticks.
    chooseSide(&cfg, &cat, .vae, "/m/vae/wan-b.safetensors");
    try testing.expectEqualStrings("/m/vae/wan-b.safetensors", cfg.vae.slice());
    selectCheckpoint(&cfg, &cat, "/m/krea2/b.safetensors");
    try testing.expectEqualStrings("/m/vae/wan-b.safetensors", cfg.vae.slice());
    try testing.expectEqualStrings("/m/te/qwen3vl-4b.safetensors", cfg.text_encoder.slice());
    try testing.expectEqualStrings("/m/vae/wan-b.safetensors", sideState(&cfg, &cat, .vae).file);

    // "none" is a real choice and survives a switch.
    chooseSide(&cfg, &cat, .taesd, config.choice_none);
    try testing.expectEqualStrings("", cfg.taesd.slice());
    selectCheckpoint(&cfg, &cat, "/m/krea2/a.safetensors");
    try testing.expectEqualStrings("", cfg.taesd.slice());
    try testing.expectEqual(SideState.none, sideState(&cfg, &cat, .taesd));
}

test "a bundled checkpoint uses its own parts until overridden, and 'bundled' clears the override" {
    const gpa = testing.allocator;
    var cat = try diffusionCatalog(gpa);
    defer cat.deinit();
    var cfg: Config = .{};

    selectCheckpoint(&cfg, &cat, "/m/sdxl/bundled.safetensors");
    try testing.expectEqualStrings("", cfg.text_encoder.slice());
    try testing.expectEqualStrings("", cfg.text_encoder_2.slice());
    try testing.expectEqualStrings("", cfg.vae.slice());
    try testing.expectEqual(SideState.bundled, sideState(&cfg, &cat, .vae));
    try testing.expectEqual(SideState.bundled, sideState(&cfg, &cat, .text_encoder_2));
    // No preview decoder exists for SDXL: none, not an error.
    try testing.expectEqual(SideState.none, sideState(&cfg, &cat, .taesd));

    chooseSide(&cfg, &cat, .vae, "/m/vae/sdxl.vae.safetensors");
    try testing.expectEqualStrings("/m/vae/sdxl.vae.safetensors", cfg.vae.slice());
    // A UNet-only SDXL file borrows the bundled checkpoint's CLIPs (donor order:
    // standalone files first, then bundled checkpoints) and keeps the VAE pick.
    selectCheckpoint(&cfg, &cat, "/m/sdxl/unet-only.gguf");
    try testing.expectEqualStrings("/m/sdxl/bundled.safetensors", cfg.text_encoder.slice());
    try testing.expectEqualStrings("/m/sdxl/bundled.safetensors", cfg.text_encoder_2.slice());
    try testing.expectEqualStrings("/m/vae/sdxl.vae.safetensors", cfg.vae.slice());

    // Back on the bundled file, "bundled" for the VAE clears the memory...
    selectCheckpoint(&cfg, &cat, "/m/sdxl/bundled.safetensors");
    chooseSide(&cfg, &cat, .vae, config.choice_bundled);
    try testing.expectEqualStrings("", cfg.vae.slice());
    try testing.expectEqualStrings("", cfg.familySides("sdxl").?.vae.slice());
    // ...so the UNet-only file falls back to the first compatible VAE again.
    selectCheckpoint(&cfg, &cat, "/m/sdxl/unet-only.gguf");
    try testing.expectEqualStrings("/m/vae/sdxl.vae.safetensors", cfg.vae.slice());
}

test "resolving a pre-existing setup adopts it exactly as configured" {
    const gpa = testing.allocator;
    var cat = try diffusionCatalog(gpa);
    defer cat.deinit();
    var cfg: Config = .{};
    // The old config: a Krea2 file with a hand-picked VAE, the second of two.
    cfg.diffusion_model.set("/m/krea2/a.safetensors");
    cfg.text_encoder.set("/m/te/qwen3vl-4b.safetensors");
    cfg.vae.set("/m/vae/wan-b.safetensors");
    resolveAll(&cfg, &cat);
    try testing.expectEqualStrings("/m/vae/wan-b.safetensors", cfg.vae.slice());
    try testing.expectEqualStrings("/m/te/qwen3vl-4b.safetensors", cfg.text_encoder.slice());
    // The one slot that was empty is filled, as it would have been by hand.
    try testing.expectEqualStrings("/m/vae_approx/taew2_1.safetensors", cfg.taesd.slice());
    // Without adoption the first VAE would have won.
    var fresh: Config = .{};
    selectCheckpoint(&fresh, &cat, "/m/krea2/a.safetensors");
    try testing.expectEqualStrings("/m/vae/wan-a.safetensors", fresh.vae.slice());
    // A switch to another family does NOT adopt the old family's file.
    selectCheckpoint(&cfg, &cat, "/m/sdxl/unet-only.gguf");
    try testing.expectEqualStrings("/m/sdxl/bundled.safetensors", cfg.text_encoder.slice());
    try testing.expectEqualStrings("/m/vae/sdxl.vae.safetensors", cfg.vae.slice());
    try testing.expectEqualStrings("", cfg.taesd.slice());
    // The same for a tower: an existing pairing survives the first resolve.
    var tcat = try Catalog.fromEntries(gpa, &.{
        llmEntry("/lm/g.gguf", "gemma4", 5376, true),
        towerEntry("/lm/t1.gguf", "gemma4", 5376),
        towerEntry("/lm/t2.gguf", "gemma4", 5376),
    });
    defer tcat.deinit();
    var lcfg: Config = .{};
    lcfg.llm_model.set("/lm/g.gguf");
    lcfg.vision_tower.set("/lm/t2.gguf");
    resolveAll(&lcfg, &tcat);
    try testing.expectEqualStrings("/lm/t2.gguf", lcfg.vision_tower.slice());
}

test "a checkpoint the catalog does not know: a resolve leaves its sides alone, a switch clears them" {
    const gpa = testing.allocator;
    var cat = try diffusionCatalog(gpa);
    defer cat.deinit();
    var cfg: Config = .{};
    cfg.diffusion_model.set("/elsewhere/model.safetensors");
    cfg.text_encoder.set("/elsewhere/te.safetensors");
    resolveAll(&cfg, &cat);
    try testing.expectEqualStrings("/elsewhere/te.safetensors", cfg.text_encoder.slice());
    try testing.expectEqualStrings("/elsewhere/te.safetensors", sideState(&cfg, &cat, .text_encoder).file);
    selectCheckpoint(&cfg, &cat, "/elsewhere/other.safetensors");
    try testing.expectEqualStrings("/elsewhere/other.safetensors", cfg.diffusion_model.slice());
    try testing.expectEqualStrings("", cfg.text_encoder.slice());
    clearCheckpoint(&cfg);
    try testing.expectEqualStrings("", cfg.diffusion_model.slice());
    try testing.expectEqualStrings("", cfg.text_encoder.slice());
}

test "vision towers: first fit by width, remembered per class, none for text-only" {
    const gpa = testing.allocator;
    var cat = try Catalog.fromEntries(gpa, &.{
        llmEntry("/lm/gemma4-31b.gguf", "gemma4", 5376, true),
        llmEntry("/lm/gemma4-31b-other.gguf", "gemma4", 5376, true),
        llmEntry("/lm/gemma4-12b.gguf", "gemma4", 3840, true),
        llmEntry("/lm/qwen3-8b.gguf", "qwen3", 4096, false),
        towerEntry("/lm/mmproj-12b.gguf", "gemma4", 3840),
        towerEntry("/lm/mmproj-31b.gguf", "gemma4", 5376),
        towerEntry("/lm/mmproj-31b-b.gguf", "gemma4", 5376),
    });
    defer cat.deinit();
    var cfg: Config = .{};

    selectLlm(&cfg, &cat, "/lm/gemma4-31b.gguf");
    try testing.expectEqualStrings("/lm/mmproj-31b-b.gguf", cfg.vision_tower.slice()); // path order
    try testing.expectEqualStrings("/lm/mmproj-31b-b.gguf", cfg.classTower("gemma4|5376").?);
    chooseTower(&cfg, &cat, "/lm/mmproj-31b.gguf");
    try testing.expectEqualStrings("/lm/mmproj-31b.gguf", cfg.vision_tower.slice());

    // Another 31B keeps the class's tower; the 12B gets its own.
    selectLlm(&cfg, &cat, "/lm/gemma4-31b-other.gguf");
    try testing.expectEqualStrings("/lm/mmproj-31b.gguf", cfg.vision_tower.slice());
    selectLlm(&cfg, &cat, "/lm/gemma4-12b.gguf");
    try testing.expectEqualStrings("/lm/mmproj-12b.gguf", cfg.vision_tower.slice());

    // Text-only clears the tower; "none" is remembered for the class.
    selectLlm(&cfg, &cat, "/lm/qwen3-8b.gguf");
    try testing.expectEqualStrings("", cfg.vision_tower.slice());
    selectLlm(&cfg, &cat, "/lm/gemma4-31b.gguf");
    chooseTower(&cfg, &cat, config.choice_none);
    try testing.expectEqualStrings("", cfg.vision_tower.slice());
    selectLlm(&cfg, &cat, "/lm/gemma4-31b-other.gguf");
    try testing.expectEqualStrings("", cfg.vision_tower.slice());

    // Unknown to the catalog: a resolve leaves it alone, a switch clears it.
    cfg.llm_model.set("/elsewhere/model.gguf");
    cfg.vision_tower.set("/elsewhere/mmproj.gguf");
    resolveAll(&cfg, &cat);
    try testing.expectEqualStrings("/elsewhere/mmproj.gguf", cfg.vision_tower.slice());
    selectLlm(&cfg, &cat, "/elsewhere/other.gguf");
    try testing.expectEqualStrings("", cfg.vision_tower.slice());
    clearLlm(&cfg);
    try testing.expectEqualStrings("", cfg.llm_model.slice());
    try testing.expectEqualStrings("", cfg.vision_tower.slice());
}
