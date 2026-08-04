//! What the GUI knows about a diffusion checkpoint before it loads one.
//!
//! The CLI grew two capabilities the GUI did not have: a second architecture
//! (SD1.5 alongside krea2) and **container style as a property of the file, not
//! of the architecture** — any model ships either as one bundled checkpoint
//! (denoiser + text encoder + VAE under their own prefixes) or as separate
//! files. `pipeline.resolveComponent` already handles all four combinations; the
//! GUI's problem was upstream of that: it *demanded* three paths before it would
//! build a diffusion engine at all, so a joined SD1.5 checkpoint could not be
//! configured, and picking one would then have failed on a GPU backend with
//! nothing but a log line.
//!
//! This module is the single place the GUI's architecture knowledge lives:
//!
//! - **What is in a file** (`inspect` / `inspectSide`) — so settings can say
//!   "this checkpoint has its own text encoder and VAE" instead of making the
//!   user guess, and so the two side paths can be presented as the *overrides*
//!   they actually are.
//! - **What a family supports and expects** (`traits`) — backends it has kernels
//!   for, and the resolution / step / CFG defaults that make it produce an image
//!   rather than mush (krea2 at 1024²/CFG 1 and SD1.5 at 512²/CFG 7.5 are not
//!   interchangeable).
//! - **What is still missing** (`missing`) — the readiness predicate the studio,
//!   the chat image tool and the settings screen all share.
//!
//! Adding a third architecture is a `Family` arm in the pipeline plus one entry
//! in each of the two tables here; nothing else in the GUI enumerates families.
//!
//! ⚠️ **Everything here is ADVISORY, never a gate.** The authority on whether a
//! checkpoint can be loaded is `pipeline.Session.init` — it resolves components
//! and rejects an unsupported backend itself. Two things below deliberately
//! mirror private knowledge in `pipeline.zig` (the per-component probe names in
//! `componentSpec`, and the CPU-only rule for SD1.5), and mirrored knowledge
//! drifts. Keeping this side purely advisory bounds that drift to a stale *hint*
//! in the settings screen; it can never turn into a model the GUI refuses to
//! load or a wrong image. If the pipeline ever exports those two facts, delete
//! the tables and call it.
const std = @import("std");
const tp = @import("TensorPencil");

const pipeline = tp.pipeline;

pub const Family = pipeline.Family;
pub const Component = pipeline.Component;
pub const Backend = pipeline.Backend;

/// Which pipeline components a checkpoint carries. A file may hold any subset: a
/// full single-file checkpoint has all of them, ggufy's model-only GGUF has just
/// the denoiser, a VAE export has just the decoder.
///
/// `conditioner2` is SDXL's second text tower (OpenCLIP bigG). It is a real,
/// separately-resolved component — `--text-encoder-2` overrides it independently
/// of the first — so it is modelled here rather than assumed to travel with
/// `conditioner`.
pub const Contents = struct {
    denoiser: bool = false,
    conditioner: bool = false,
    conditioner2: bool = false,
    decoder: bool = false,

    pub fn has(self: Contents, comp: Component) bool {
        return switch (comp) {
            .denoiser => self.denoiser,
            .conditioner => self.conditioner,
            .conditioner2 => self.conditioner2,
            .decoder => self.decoder,
        };
    }
};

/// What a primary checkpoint turned out to be.
pub const Info = struct {
    family: Family,
    contents: Contents,

    /// True when this one file is the whole pipeline (the "checkpoint-style"
    /// case) — the GUI needs no side paths at all. Family-aware, because "every
    /// component" means one more thing for SDXL than for the others.
    pub fn isComplete(self: Info) bool {
        const need2 = traits(self.family).dual_conditioner;
        return self.contents.denoiser and self.contents.conditioner and
            self.contents.decoder and (!need2 or self.contents.conditioner2);
    }
};

// ── The probe table ───────────────────────────────────────────────────────────
// A mirror of `pipeline.componentSpec`: one tensor name per component, tried
// under each prefix spelling that architecture is distributed with. See the
// advisory warning in the module doc — a drifted entry costs a hint, not a load.

const Spec = struct {
    prefixes: []const []const u8,
    probe: []const u8,
};

fn spec(fam: Family, comp: Component) ?Spec {
    return switch (fam) {
        .krea2 => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probe = "blocks.0.attn.wq.weight" },
            .conditioner => .{ .prefixes = &.{ "text_encoders.", "" }, .probe = "model.language_model.embed_tokens.weight" },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "vae.", "" }, .probe = "decoder.conv1.weight" },
            .conditioner2 => null, // single-conditioner architecture
        },
        .sd15 => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probe = "input_blocks.0.0.weight" },
            .conditioner => .{
                .prefixes = &.{ "cond_stage_model.transformer.text_model.", "text_model.", "" },
                .probe = "final_layer_norm.weight",
            },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "" }, .probe = "decoder.conv_in.weight" },
            .conditioner2 => null,
        },
        // ⚠️ SDXL's two towers are spelled DIFFERENTLY inside one file: embedder 0 is a
        // `transformers` CLIPTextModel and embedder 1 an OpenCLIP tower, hence the
        // unrelated probe names. Probing embedder 1 with embedder 0's `text_model.`
        // spelling finds nothing and would report a bundled checkpoint as incomplete.
        .sdxl => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probe = "input_blocks.0.0.weight" },
            .conditioner => .{
                .prefixes = &.{ "conditioner.embedders.0.transformer.text_model.", "text_model.", "" },
                .probe = "final_layer_norm.weight",
            },
            .conditioner2 => .{
                .prefixes = &.{ "conditioner.embedders.1.model.", "" },
                .probe = "ln_final.weight",
            },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "" }, .probe = "decoder.conv_in.weight" },
        },
    };
}

/// Does `store` carry `comp` under any prefix spelling `fam` is distributed with?
pub fn storeHas(store: tp.weights.WeightStore, fam: Family, comp: Component) bool {
    const s = spec(fam, comp) orelse return false;
    for (s.prefixes) |pfx| {
        var buf: [256]u8 = undefined;
        if (pfx.len + s.probe.len > buf.len) continue;
        @memcpy(buf[0..pfx.len], pfx);
        @memcpy(buf[pfx.len..][0..s.probe.len], s.probe);
        if (store.get(buf[0 .. pfx.len + s.probe.len]) != null) return true;
    }
    return false;
}

/// Everything `fam` could contribute, from one open store.
pub fn scan(store: tp.weights.WeightStore, fam: Family) Contents {
    return .{
        .denoiser = storeHas(store, fam, .denoiser),
        .conditioner = storeHas(store, fam, .conditioner),
        .conditioner2 = storeHas(store, fam, .conditioner2),
        .decoder = storeHas(store, fam, .decoder),
    };
}

// ── Per-family traits ─────────────────────────────────────────────────────────

/// GUI-facing properties of an architecture: what it runs on, and the parameters
/// that make it produce an image.
pub const Traits = struct {
    /// Name for the settings/status UI.
    label: []const u8,
    /// Backends the engine has kernels for.
    ///
    /// Every family runs on all four today (`sd_unet{,_gpu,_cuda}` /
    /// `sd_vae{,_gpu,_cuda}` landed the SD family's device paths; see BACKEND.md
    /// §2A). The field stays because "which backends" is genuinely per-family —
    /// it was `cpu` only for SD until those kernels existed, and the next
    /// architecture starts life CPU-only too. Advisory: `Session.init` is what
    /// actually refuses a backend it has no kernels for.
    backends: []const Backend,
    /// Whether this architecture has a SECOND text tower (SDXL's OpenCLIP bigG).
    /// Drives the extra path row and the readiness check; the pipeline resolves
    /// `conditioner2` independently of `conditioner`.
    dual_conditioner: bool = false,
    /// Default generation parameters. These are NOT interchangeable between
    /// families: SD at krea2's CFG 1.0 has no classifier-free guidance at all, and
    /// krea2 at SD1.5's 512² is below the resolution it was trained for.
    width: usize,
    height: usize,
    steps: usize,
    cfg: f32,

    pub fn supports(self: Traits, b: Backend) bool {
        for (self.backends) |x| if (x == b) return true;
        return false;
    }
};

const all_backends = [_]Backend{ .cpu, .vulkan, .zig_cuda, .cuda };

pub fn traits(fam: Family) Traits {
    return switch (fam) {
        .krea2 => .{
            .label = "krea2 (flow-matching DiT)",
            .backends = &all_backends,
            .width = 1024,
            .height = 1024,
            .steps = 8,
            .cfg = 1.0,
        },
        .sd15 => .{
            .label = "SD1.5 (UNet)",
            .backends = &all_backends,
            .width = 512,
            .height = 512,
            .steps = 20,
            .cfg = 7.5,
        },
        // SDXL's native resolution is 1024²: it was trained with the size
        // micro-conditioning telling it so, and asking for 512² gets a model that
        // knows it is being asked for a small image and renders like it.
        .sdxl => .{
            .label = "SDXL (UNet, dual CLIP)",
            .backends = &all_backends,
            .dual_conditioner = true,
            .width = 1024,
            .height = 1024,
            .steps = 20,
            .cfg = 7.5,
        },
    };
}

// ── Readiness ─────────────────────────────────────────────────────────────────

/// Which override paths the user has actually set. A struct rather than a
/// positional list of bools so adding a component cannot silently transpose two
/// arguments at a call site.
pub const Overrides = struct {
    conditioner: bool = false,
    conditioner2: bool = false,
    decoder: bool = false,
};

/// Which components nothing supplies. The GUI shows these; the pipeline is what
/// actually enforces them (`error.ComponentNotInCheckpoint`).
pub const Missing = struct {
    conditioner: bool = false,
    conditioner2: bool = false,
    decoder: bool = false,

    pub fn any(self: Missing) bool {
        return self.conditioner or self.conditioner2 or self.decoder;
    }
};

/// Given what the primary checkpoint carries and which overrides are set, what is
/// still unaccounted for.
///
/// A supplied side file counts even when we could not read it: an unreadable
/// *explicit* path fails loudly in `Session.init` with a real error, and
/// reporting it as "missing" here would name the wrong problem.
///
/// `conditioner2` can only be missing for an architecture that has one — for
/// everything else the pipeline never asks for it.
pub fn missing(info: Info, have: Overrides) Missing {
    const need2 = traits(info.family).dual_conditioner;
    return .{
        .conditioner = !info.contents.conditioner and !have.conditioner,
        .conditioner2 = need2 and !info.contents.conditioner2 and !have.conditioner2,
        .decoder = !info.contents.decoder and !have.decoder,
    };
}

// ── Cached inspection ─────────────────────────────────────────────────────────

/// The result of looking at one path. `unset` for an empty path (the normal
/// state of an unused override), `failed` when the file could not be read or
/// recognized.
pub const Probe = union(enum) {
    unset,
    failed: anyerror,
    ok: Info,

    pub fn info(self: Probe) ?Info {
        return switch (self) {
            .ok => |i| i,
            else => null,
        };
    }
};

/// One path slot's memoized inspection.
///
/// Opening a checkpoint parses (and mmaps) its header, so the settings screen —
/// which re-renders every frame — must not do it per frame. The cache re-probes
/// only when the path text (or, for a side file, the family it is scanned
/// against) actually changes, which is exactly when the user edits the field.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// Owned copy of the path `result` describes ("" when nothing probed yet).
    path: []u8 = &.{},
    /// The family a *side* file was scanned against; null for a primary probe.
    fam_key: ?Family = null,
    result: Probe = .unset,

    pub fn init(gpa: std.mem.Allocator, io: std.Io) Cache {
        return .{ .gpa = gpa, .io = io };
    }

    pub fn deinit(self: *Cache) void {
        if (self.path.len > 0) self.gpa.free(self.path);
        self.* = undefined;
    }

    /// Drop the memo so the next call re-reads the file (e.g. after the user
    /// replaced the file on disk under an unchanged path).
    pub fn invalidate(self: *Cache) void {
        if (self.path.len > 0) self.gpa.free(self.path);
        self.path = &.{};
        self.fam_key = null;
        self.result = .unset;
    }

    /// Inspect a PRIMARY checkpoint: the denoiser decides the architecture, so
    /// this is the only probe that can determine a family.
    pub fn primary(self: *Cache, path: []const u8) Probe {
        return self.probe(path, null);
    }

    /// Inspect a SIDE file (text encoder / VAE) against an already-known family.
    /// A standalone CLIP or VAE export holds no denoiser, so it cannot name its
    /// own architecture — the primary checkpoint's family is the key.
    pub fn side(self: *Cache, path: []const u8, fam: Family) Probe {
        return self.probe(path, fam);
    }

    fn probe(self: *Cache, path: []const u8, fam: ?Family) Probe {
        if (std.mem.eql(u8, path, self.path) and famEql(fam, self.fam_key)) return self.result;

        if (self.path.len > 0) self.gpa.free(self.path);
        self.path = self.gpa.dupe(u8, path) catch &.{};
        self.fam_key = fam;
        self.result = if (path.len == 0) .unset else read(self.gpa, self.io, path, fam);
        return self.result;
    }

    fn famEql(a: ?Family, b: ?Family) bool {
        if (a == null and b == null) return true;
        if (a == null or b == null) return false;
        return a.? == b.?;
    }
};

/// Open `path` and report what it holds. `fam` null means "detect it" (primary);
/// non-null means "scan against this family" (a side file).
fn read(gpa: std.mem.Allocator, io: std.Io, path: []const u8, fam: ?Family) Probe {
    // Opens by MAGIC, so a GGUF denoiser works here exactly as it does in the
    // pipeline — the GUI never has to care which container a file uses.
    var c = pipeline.DitContainer.open(gpa, io, path) catch |err| return .{ .failed = err };
    defer c.deinit();
    const store = c.store();
    const family = fam orelse (pipeline.detectFamily(store) catch |err| return .{ .failed = err });
    return .{ .ok = .{ .family = family, .contents = scan(store, family) } };
}

/// One-shot inspection of a primary checkpoint (no memo). For callers outside a
/// render loop.
pub fn inspect(gpa: std.mem.Allocator, io: std.Io, path: []const u8) Probe {
    return read(gpa, io, path, null);
}

/// One-shot inspection of a side file against a known family (no memo).
pub fn inspectSide(gpa: std.mem.Allocator, io: std.Io, path: []const u8, fam: Family) Probe {
    return read(gpa, io, path, fam);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test "traits: every family runs on every backend" {
    // The SD family's device kernels (sd_unet_gpu/_cuda, sd_vae_gpu/_cuda) landed,
    // so the CPU-only restriction is gone. If a future architecture arrives
    // CPU-first, THIS is the test that should be narrowed — the mechanism stays.
    inline for (@typeInfo(Family).@"enum".fields) |f| {
        const t = traits(@enumFromInt(f.value));
        for ([_]Backend{ .cpu, .vulkan, .zig_cuda, .cuda }) |b|
            try std.testing.expect(t.supports(b));
    }
}

test "traits: each family has a label, a resolution and a positive step count" {
    inline for (@typeInfo(Family).@"enum".fields) |f| {
        const t = traits(@enumFromInt(f.value));
        try std.testing.expect(t.label.len > 0);
        try std.testing.expect(t.width > 0 and t.height > 0);
        try std.testing.expect(t.steps > 0);
        try std.testing.expect(t.cfg >= 1.0);
    }
}

test "traits: the families' defaults differ in the fields that matter" {
    // Swapping one family's CFG or resolution for another's is a visibly wrong
    // image, not a preference — these must not converge by accident.
    try std.testing.expect(traits(.sd15).cfg != traits(.krea2).cfg);
    try std.testing.expect(traits(.sd15).width != traits(.krea2).width);
    try std.testing.expect(traits(.sdxl).width != traits(.sd15).width);
    // SDXL is the only dual-tower architecture today.
    try std.testing.expect(traits(.sdxl).dual_conditioner);
    try std.testing.expect(!traits(.sd15).dual_conditioner);
    try std.testing.expect(!traits(.krea2).dual_conditioner);
}

test "missing: a bundled SD1.5 checkpoint needs no side files" {
    const bundled: Info = .{
        .family = .sd15,
        .contents = .{ .denoiser = true, .conditioner = true, .decoder = true },
    };
    try std.testing.expect(bundled.isComplete());
    try std.testing.expect(!missing(bundled, .{}).any());
}

test "missing: a denoiser-only checkpoint needs both, until they are supplied" {
    const only_unet: Info = .{ .family = .sd15, .contents = .{ .denoiser = true } };
    try std.testing.expect(!only_unet.isComplete());

    const none = missing(only_unet, .{});
    try std.testing.expect(none.conditioner and none.decoder);

    const te_only = missing(only_unet, .{ .conditioner = true });
    try std.testing.expect(!te_only.conditioner and te_only.decoder);

    const both = missing(only_unet, .{ .conditioner = true, .decoder = true });
    try std.testing.expect(!both.any());
}

test "missing: a bundled component is satisfied even with no override set" {
    // ggufy's SD arm: the GGUF holds only the UNet, CLIP+VAE come from the
    // original checkpoint. The mirror case — a checkpoint with a VAE but no
    // text encoder — must ask for exactly one file, not two.
    const unet_and_vae: Info = .{
        .family = .sd15,
        .contents = .{ .denoiser = true, .decoder = true },
    };
    const m = missing(unet_and_vae, .{});
    try std.testing.expect(m.conditioner);
    try std.testing.expect(!m.decoder);
}

test "missing: SDXL's second tower counts, and only for SDXL" {
    // A checkpoint holding both towers is complete...
    const bundled: Info = .{ .family = .sdxl, .contents = .{
        .denoiser = true,
        .conditioner = true,
        .conditioner2 = true,
        .decoder = true,
    } };
    try std.testing.expect(bundled.isComplete());
    try std.testing.expect(!missing(bundled, .{}).any());

    // ...while one holding only CLIP-L is NOT, even though every other
    // architecture would call the same contents complete.
    const half: Info = .{ .family = .sdxl, .contents = .{
        .denoiser = true,
        .conditioner = true,
        .decoder = true,
    } };
    try std.testing.expect(!half.isComplete());
    const m = missing(half, .{});
    try std.testing.expect(m.conditioner2);
    try std.testing.expect(!m.conditioner and !m.decoder);
    // Supplying it as an override resolves it.
    try std.testing.expect(!missing(half, .{ .conditioner2 = true }).any());

    // The identical contents under a single-tower family ask for nothing.
    const sd15: Info = .{ .family = .sd15, .contents = half.contents };
    try std.testing.expect(sd15.isComplete());
    try std.testing.expect(!missing(sd15, .{}).any());
    // ...and an override for a tower the architecture does not have is simply
    // ignored here (the pipeline never resolves `conditioner2` for it).
    try std.testing.expect(!missing(sd15, .{ .conditioner2 = true }).any());
}

test "spec: only SDXL probes a second tower, with its own spelling" {
    try std.testing.expect(spec(.sdxl, .conditioner2) != null);
    try std.testing.expect(spec(.sd15, .conditioner2) == null);
    try std.testing.expect(spec(.krea2, .conditioner2) == null);
    // ⚠️ The two SDXL towers are spelled differently inside one file; reusing
    // embedder 0's probe for embedder 1 would find nothing and report a bundled
    // checkpoint as incomplete.
    try std.testing.expect(!std.mem.eql(
        u8,
        spec(.sdxl, .conditioner).?.probe,
        spec(.sdxl, .conditioner2).?.probe,
    ));
}

test "Contents.has covers every component" {
    const c: Contents = .{ .denoiser = true, .decoder = true, .conditioner2 = true };
    inline for (@typeInfo(Component).@"enum".fields) |f| {
        const comp: Component = @enumFromInt(f.value);
        const want = switch (comp) {
            .denoiser, .decoder, .conditioner2 => true,
            .conditioner => false,
        };
        try std.testing.expectEqual(want, c.has(comp));
    }
}

test "Cache memoizes by path and re-probes when it changes" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var cache = Cache.init(std.testing.allocator, threaded.io());
    defer cache.deinit();

    // An empty path is "not configured", not an error.
    try std.testing.expect(cache.primary("") == .unset);
    try std.testing.expect(cache.primary("") == .unset);

    // A path that cannot exist probes once and memoizes the failure.
    const p = "/nonexistent/tp-gui-test-checkpoint.safetensors";
    try std.testing.expect(cache.primary(p) == .failed);
    // Same path → memo (no second open).
    try std.testing.expect(cache.primary(p) == .failed);
    try std.testing.expectEqualStrings(p, cache.path);
    // Back to empty → unset again.
    try std.testing.expect(cache.primary("") == .unset);
    try std.testing.expectEqual(@as(usize, 0), cache.path.len);
}

test "Cache keys a side probe on the family too" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var cache = Cache.init(std.testing.allocator, threaded.io());
    defer cache.deinit();

    const p = "/nonexistent/tp-gui-test-vae.safetensors";
    _ = cache.side(p, .krea2);
    try std.testing.expectEqual(@as(?Family, .krea2), cache.fam_key);
    // Same path, different family: must re-probe rather than reuse the krea2
    // answer — the probe names are per-family.
    _ = cache.side(p, .sd15);
    try std.testing.expectEqual(@as(?Family, .sd15), cache.fam_key);
}

/// libc getenv (std.posix.getenv is gone in 0.16; the module links libc via
/// TensorPencil). Only used by the opt-in test below.
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

test "probe tables match a real checkpoint (opt-in)" {
    // The probe names here MIRROR `pipeline.componentSpec`, and a mirror drifts.
    // Point TP_SPEC_CHECKPOINT at a real bundled checkpoint to check the tables
    // against one; unset (the default, and CI) skips. Silent on success — a
    // passing test must not write to stderr (see ZIG.md).
    const raw = getenv("TP_SPEC_CHECKPOINT") orelse return;
    const path = std.mem.span(raw);
    if (path.len == 0) return;

    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();

    const r = inspect(std.testing.allocator, threaded.io(), path);
    errdefer std.debug.print("TP_SPEC_CHECKPOINT={s} -> {any}\n", .{ path, r });
    const info = r.info() orelse return error.CheckpointNotRecognized;
    // A bundled checkpoint must read as complete. If this fails the prefix
    // spellings drifted from the pipeline's, and the settings screen is telling
    // users a working model is missing pieces.
    try std.testing.expect(info.contents.denoiser);
    try std.testing.expect(info.isComplete());
}
