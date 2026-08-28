//! What the GUI knows about a diffusion checkpoint before it loads one, and the single
//! place the GUI's architecture knowledge lives:
//!
//! - What is in a file (`inspect` / `inspectSide`), so settings can say "this checkpoint
//!   has its own text encoder and VAE" instead of making the user guess, and so the two
//!   side paths can be presented as the overrides they are.
//! - What a family supports and expects (`traits`): backends it has kernels for, and the
//!   resolution / step / CFG defaults that make it produce an image rather than mush.
//! - What is still missing (`missing`), the readiness predicate the studio, the chat
//!   image tool and the settings screen all share.
//!
//! Container style is a property of the FILE, not of the architecture: any model ships
//! either bundled (denoiser, text encoder and VAE under their own prefixes) or as
//! separate files, and `pipeline.resolveComponent` handles all four combinations. The
//! GUI must not demand three paths before it will build an engine, or a bundled
//! checkpoint cannot be configured at all.
//!
//! Adding an architecture is a `Family` arm in the pipeline plus one entry in each of
//! the two tables here; nothing else in the GUI enumerates families.
//!
//! Everything here is ADVISORY, never a gate. `pipeline.Session.init` is the authority on
//! whether a checkpoint loads: it resolves components and rejects an unsupported backend
//! itself. Two things below deliberately mirror private knowledge in `pipeline.zig` (the
//! per-component probe names in `componentSpec`, and the CPU-only rule for SD1.5), and
//! mirrored knowledge drifts. Staying advisory bounds that drift to a stale hint in the
//! settings screen rather than a model the GUI refuses to load. If the pipeline ever
//! exports those two facts, delete the tables and call it.
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
/// separately-resolved component, `--text-encoder-2` overrides it independently
/// of the first, so it is modelled here rather than assumed to travel with
/// `conditioner`.
pub const Contents = struct {
    denoiser: bool = false,
    conditioner: bool = false,
    conditioner2: bool = false,
    decoder: bool = false,
    /// MiniMax H3's audio VAE. Modelled like `conditioner2` and for the same
    /// reason: it resolves independently, so a checkpoint can carry one decoder
    /// and not the other.
    decoder2: bool = false,

    pub fn has(self: Contents, comp: Component) bool {
        return switch (comp) {
            .denoiser => self.denoiser,
            .conditioner => self.conditioner,
            .conditioner2 => self.conditioner2,
            .decoder => self.decoder,
            .decoder2 => self.decoder2,
        };
    }
};

/// What a primary checkpoint turned out to be.
pub const Info = struct {
    family: Family,
    contents: Contents,

    /// True when this one file is the whole pipeline (the "checkpoint-style"
    /// case), the GUI needs no side paths at all. Family-aware, because "every
    /// component" means one more thing for SDXL than for the others.
    pub fn isComplete(self: Info) bool {
        const t = traits(self.family);
        return self.contents.denoiser and self.contents.conditioner and
            self.contents.decoder and
            (!t.dual_conditioner or self.contents.conditioner2) and
            (!t.dual_decoder or self.contents.decoder2);
    }
};

// ── The probe table ───────────────────────────────────────────────────────────
// A mirror of `pipeline.componentSpec`: one tensor name per component, tried
// under each prefix spelling that architecture is distributed with. See the
// advisory warning in the module doc, a drifted entry costs a hint, not a load.

const Spec = struct {
    prefixes: []const []const u8,
    probe: []const u8,
};

/// Does `store` carry `comp` under any prefix spelling `fam` is distributed with?
/// The prefix/probe table lives in `pipeline`, not here. A second copy of the same
/// rules drifts as soon as the pipeline learns that a component's tensor names differ
/// by CONTAINER. A GGUF text
/// encoder is `embed_tokens.weight` where safetensors is `model.embed_tokens.weight`,
/// so `pipeline.resolveComponent` loaded it happily while this copy, still probing
/// one spelling, reported the file as carrying no conditioner, the GUI refusing a
/// file the engine could open. One table, one answer.
pub fn storeHas(store: tp.weights.WeightStore, fam: Family, comp: Component) bool {
    const s = pipeline.componentSpec(fam, comp) catch return false;
    for (s.prefixes) |pfx| {
        for (s.probes) |probe| {
            var buf: [256]u8 = undefined;
            if (pfx.len + probe.len > buf.len) continue;
            @memcpy(buf[0..pfx.len], pfx);
            @memcpy(buf[pfx.len..][0..probe.len], probe);
            if (store.get(buf[0 .. pfx.len + probe.len]) != null) return true;
        }
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
        .decoder2 = storeHas(store, fam, .decoder2),
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
    /// Every family now runs on all four. The field exists because each new
    /// architecture starts life CPU-first and offering a device it has no kernels for
    /// reads as a hang rather than as a fallback: SD was `cpu` only until
    /// `sd_unet{,_gpu,_cuda}` landed, Z-Image until `zimage_gpu` did, and Anima until
    /// `anima_{gpu,cuda}` did. Advisory, `Session.denoiser` is what actually decides,
    /// and it warns by name when a checkpoint's dtype or a device leaves the trunk on
    /// the host.
    backends: []const Backend,
    /// Whether this architecture has a SECOND text tower (SDXL's OpenCLIP bigG).
    /// Drives the extra path row and the readiness check; the pipeline resolves
    /// `conditioner2` independently of `conditioner`.
    dual_conditioner: bool = false,
    /// Whether this architecture has a SECOND decoder (H3's audio VAE), which
    /// resolves independently of the first.
    dual_decoder: bool = false,
    /// Default pixel frame count. 1 for every still-image family; a video family
    /// SNAPS this to its own grid, so read the shape back from
    /// `Session.latentShape` rather than trusting it.
    frames: usize = 1,
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
        // The official Z-Image Turbo defaults: 1024², 8 steps, cfg 1 (no negative
        // branch), from ComfyUI's own template for it.
        //
        .zimage => .{
            .label = "Z-Image (NextDiT)",
            .backends = &all_backends,
            .width = 1024,
            .height = 1024,
            .steps = 8,
            .cfg = 1.0,
        },
        // ComfyUI's own `image_anima_base_v1` template: 1024x1024, 30 steps, cfg 4,
        // euler + `simple`.
        .anima => .{
            .label = "Anima (Cosmos MiniTrainDIT + LLM adapter)",
            .backends = &all_backends,
            .width = 1024,
            .height = 1024,
            .steps = 30,
            .cfg = 4.0,
        },
        // The node defaults from `comfy_extras/nodes_minimax_h3.py`: 1344x768 and
        // 124 frames (~5s at 24 fps, the bottom of its trained 124-362 range).
        //
        // cfg 1.0 is read off the node signatures, not guessed: every H3
        // conditioning node outputs ONE conditioning (`positive`) and takes no
        // negative, so there is no branch to guide against. No workflow template
        // for it ships locally, so the STEP count is the one value here with no
        // upstream source; the 4-step turbo LoRA is the intended fast path.
        .minimax_h3 => .{
            .label = "MiniMax H3 (joint audio-video DiT)",
            // CPU only until the trunk exists at all, let alone its GPU twins.
            // Offering a device with no kernels reads as a hang, not a fallback.
            .backends = &.{.cpu},
            .dual_decoder = true,
            .width = 1344,
            .height = 768,
            .frames = 124,
            .steps = 30,
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
    decoder2: bool = false,
};

/// Which components nothing supplies. The GUI shows these; the pipeline is what
/// actually enforces them (`error.ComponentNotInCheckpoint`).
pub const Missing = struct {
    conditioner: bool = false,
    conditioner2: bool = false,
    decoder: bool = false,
    decoder2: bool = false,

    pub fn any(self: Missing) bool {
        return self.conditioner or self.conditioner2 or self.decoder or self.decoder2;
    }
};

/// Given what the primary checkpoint carries and which overrides are set, what is
/// still unaccounted for.
///
/// A supplied side file counts even when we could not read it: an unreadable
/// *explicit* path fails loudly in `Session.init` with a real error, and
/// reporting it as "missing" here would name the wrong problem.
///
/// `conditioner2` can only be missing for an architecture that has one, for
/// everything else the pipeline never asks for it.
pub fn missing(info: Info, have: Overrides) Missing {
    const t = traits(info.family);
    return .{
        .conditioner = !info.contents.conditioner and !have.conditioner,
        .conditioner2 = t.dual_conditioner and !info.contents.conditioner2 and !have.conditioner2,
        .decoder = !info.contents.decoder and !have.decoder,
        .decoder2 = t.dual_decoder and !info.contents.decoder2 and !have.decoder2,
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
/// Opening a checkpoint parses (and mmaps) its header, so the settings screen,
/// which re-renders every frame, must not do it per frame. The cache re-probes
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
    /// own architecture, the primary checkpoint's family is the key.
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
    // pipeline, the GUI never has to care which container a file uses.
    var c = pipeline.Container.open(gpa, io, path) catch |err| return .{ .failed = err };
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

test "traits: every family runs on the CPU, and the GPU list is per family" {
    // Narrowed when Z-Image landed, which is exactly what the previous version of
    // this test said should happen if an architecture ever arrived CPU-first. The
    // mechanism is unchanged: the GUI hides backends a family has no kernels for, so
    // a too-generous list here is an offered backend that then fails at render time.
    //
    // A CPU-first arrival gets NARROWED here rather than having its traits entry
    // over-promise, that is what the note on `backends` prescribes, and the
    // opposite of what makes the GUI offer a backend that then fails at render
    // time. Add to this set, do not edit the loop.
    //
    // MiniMax H3 is the current CPU-first arrival, and for a stronger reason than
    // Anima and Z-Image were: it has no trunk at all yet, on any backend. Remove
    // it here when `minimax_h3_cuda` / `minimax_h3_gpu` land.
    var cpu_only = std.EnumSet(Family).initEmpty();
    cpu_only.insert(.minimax_h3);
    inline for (@typeInfo(Family).@"enum".fields) |f| {
        const fam: Family = @enumFromInt(f.value);
        const t = traits(fam);
        // Every family runs on the CPU, without exception, that is the floor.
        try std.testing.expect(t.supports(.cpu));
        for ([_]Backend{ .vulkan, .zig_cuda, .cuda }) |b| {
            errdefer std.debug.print("{t} supports({t}) = {}\n", .{ fam, b, t.supports(b) });
            try std.testing.expectEqual(!cpu_only.contains(fam), t.supports(b));
        }
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
    // image, not a preference, these must not converge by accident.
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
    // original checkpoint. The mirror case, a checkpoint with a VAE but no
    // text encoder, must ask for exactly one file, not two.
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
    // Reads `pipeline`'s table directly, this file no longer keeps a copy, and the
    // point of the test is that the GUI's readiness view and the pipeline's resolver
    // cannot disagree about what a checkpoint contains.
    _ = try pipeline.componentSpec(.sdxl, .conditioner2);
    try std.testing.expectError(error.NoSuchComponent, pipeline.componentSpec(.sd15, .conditioner2));
    try std.testing.expectError(error.NoSuchComponent, pipeline.componentSpec(.krea2, .conditioner2));
    // The two SDXL towers are spelled differently inside one file; reusing
    // embedder 0's probe for embedder 1 would find nothing and report a bundled
    // checkpoint as incomplete.
    try std.testing.expect(!std.mem.eql(
        u8,
        (try pipeline.componentSpec(.sdxl, .conditioner)).probes[0],
        (try pipeline.componentSpec(.sdxl, .conditioner2)).probes[0],
    ));
}

test "the GUI sees a GGUF text encoder as carrying a conditioner" {
    // The regression this file's `storeHas` comment describes: the GUI kept its own
    // probe table with ONE spelling per component, so a `.gguf` encoder, whose tensors
    // are `embed_tokens.weight`, not `model.embed_tokens.weight`, scanned as empty and
    // the settings panel refused a file the engine loads fine.
    const gpa = std.testing.allocator;
    var b = try tp.gguf.TestBuilder.init(gpa, 3, 1, 1);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    // llama.cpp's spelling, which `canonicalName` maps to a BARE `embed_tokens.weight`
    // no `model.` to restore. That is the whole bug in one tensor name.
    try b.tensor("token_embd.weight", &.{ 4, 2 }, 0, 0);
    const file = try b.finish(&([_]u8{0} ** 32));
    defer gpa.free(file);
    var g = try tp.gguf.Gguf.initFromSlice(gpa, file);
    defer g.deinit();
    try std.testing.expect(storeHas(.{ .gguf = &g }, .zimage, .conditioner));
    try std.testing.expect(storeHas(.{ .gguf = &g }, .krea2, .conditioner));
}

test "Contents.has covers every component" {
    const c: Contents = .{ .denoiser = true, .decoder = true, .conditioner2 = true };
    inline for (@typeInfo(Component).@"enum".fields) |f| {
        const comp: Component = @enumFromInt(f.value);
        const want = switch (comp) {
            .denoiser, .decoder, .conditioner2 => true,
            // both left false above, so a `has` that ignored its argument fails
            .conditioner, .decoder2 => false,
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
    // Same path -> memo (no second open).
    try std.testing.expect(cache.primary(p) == .failed);
    try std.testing.expectEqualStrings(p, cache.path);
    // Back to empty -> unset again.
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
    // answer, the probe names are per-family.
    _ = cache.side(p, .sd15);
    try std.testing.expectEqual(@as(?Family, .sd15), cache.fam_key);
}

/// libc getenv (std.posix.getenv is gone in 0.16; the module links libc via
/// TensorPencil). Only used by the opt-in test below.
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

test "probe tables match a real checkpoint (opt-in)" {
    // The probe names here MIRROR `pipeline.componentSpec`, and a mirror drifts.
    // Point TP_SPEC_CHECKPOINT at a real bundled checkpoint to check the tables
    // against one; unset (the default, and CI) skips. Silent on success, a
    // passing test must not write to stderr.
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
