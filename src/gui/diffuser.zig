//! Standalone diffusion engine for tp-gui.
//!
//! Owns a resident `pipeline.Session` (loaded once, kept across a queue of
//! images so the 2nd+ image skips the multi-second reload) and runs ONE image
//! at a time on a background thread + its own CUDA context. It is deliberately
//! LLM-agnostic: the chat session composes one (backing the VRAM hooks with LLM
//! layer eviction), and the no-LLM image studio composes another (with no-op
//! hooks, so diffusion pins all free VRAM). Two things are injected so the
//! engine never has to know who is driving it:
//!
//!  - `Source`, where the next pending `GenImage` comes from (chat scans its
//!    message transcript; the studio scans its gallery list). The engine never
//!    owns the images.
//!  - `VramCoordinator`, how to make room for the image model and how much
//!    resident-weight budget it gets (chat evicts/promotes LLM layers; the
//!    studio no-ops and hands back a 0 "auto / pin all free VRAM" budget).
//!
//! Threading: the UI thread calls `pump` once per frame (reaps a finished
//! worker, starts the next pending image). The worker thread writes the
//! `GenImage` atomics (`onStep`, and the final `rgba`/`status`).
const std = @import("std");
const tp = @import("TensorPencil");
const config = @import("config.zig");

const pipeline = tp.pipeline;
const pause_gate = tp.ops.pause;

/// MEASURED per-component diffusion VRAM (bytes), re-exported from the pipeline
/// for the status-bar meter.
pub const VramBreakdown = pipeline.VramBreakdown;

/// A `std.Io.Writer` that forwards the pipeline's progress lines to `std.log`,
/// so GUI image generation is as observable in the terminal as the CLI is
/// (load / encode / per-step / vae-decode-fallback notes). The CLI passes its
/// stdout writer; the GUI has no stdout for the worker, so it routes here.
/// Line-buffered: one `std.log.info` per '\n'; a line longer than the fixed
/// accumulator is flushed in chunks. Progress notes are short, so `drain` is
/// only ever hit on `flush` (one complete line at a time).
const LogWriter = struct {
    writer: std.Io.Writer,
    line: [512]u8 = undefined,
    len: usize = 0,

    fn init(buffer: []u8) LogWriter {
        return .{ .writer = .{ .vtable = &.{ .drain = drain }, .buffer = buffer } };
    }

    fn emitByte(self: *LogWriter, b: u8) void {
        if (b == '\n' or self.len == self.line.len) {
            if (self.len > 0) std.log.info("gen: {s}", .{self.line[0..self.len]});
            self.len = 0;
            if (b == '\n') return;
        }
        self.line[self.len] = b;
        self.len += 1;
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *LogWriter = @alignCast(@fieldParentPtr("writer", w));
        for (w.buffer[0..w.end]) |b| self.emitByte(b); // buffered bytes first
        w.end = 0;
        var written: usize = 0;
        const slice = data[0 .. data.len - 1];
        for (slice) |bytes| {
            for (bytes) |b| self.emitByte(b);
            written += bytes.len;
        }
        const pattern = data[data.len - 1];
        for (0..splat) |_| {
            for (pattern) |b| self.emitByte(b);
        }
        written += pattern.len * splat;
        return written;
    }
};

pub const GenStatus = enum(u8) { pending, generating, done, failed, canceled, suspended };

/// A short human explanation of a generation failure, for the tile that reports
/// it. The recognized cases are the ones a user can ACT on; everything else
/// falls back to the error name, which is at least specific enough to search
/// for. VRAM exhaustion reaches here under four different names, the CUDA
/// libraries report an out-of-workspace as their own error, and the hand-PTX
/// path surfaces a post-OOM fault as `CudaError` (`pipeline.recoverableDecodeErr`
/// documents the same set), so mapping only `OutOfMemory` would leave the most
/// common failure looking like an internal bug.
pub fn failureText(err: anyerror) []const u8 {
    return switch (err) {
        error.OutOfMemory => "out of memory",
        error.DeviceOutOfMemory,
        error.CudaError,
        error.CublasLtError,
        error.CudnnError,
        => "out of VRAM",
        error.GpuDecodeNonFinite => "the GPU decode produced invalid pixels",
        error.FileNotFound => "a model file is missing",
        error.UnknownArchitecture => "the checkpoint's architecture is not recognized",
        error.UnsupportedCheckpoint => "this checkpoint cannot run on the selected backend",
        error.ComponentNotInCheckpoint => "the checkpoint is missing a component (VAE or text encoder)",
        // A conversation reloaded from disk whose PNG has since been moved or
        // deleted. NOT a generation failure, and deliberately not re-run.
        error.SavedImageMissing => "the saved image file is gone",
        error.IncompleteMetadata => "the checkpoint file is corrupt",
        else => @errorName(err),
    };
}

/// The model-defining config for one generation, exactly the fields that decide
/// which `pipeline.Session` an image needs (paths + backend + VAE decode path).
/// Captured onto each `GenImage` at `enqueue` so a mid-queue backend/model switch
/// only affects images enqueued AFTER it: already-queued images finish on the
/// config they were created with, and the worker reloads the pipeline at the seam
/// where consecutive images disagree. Path strings are gpa-owned dupes (freed in
/// `freeGenImage`); preview/steps/dims stay live (cosmetic / already per-image).
pub const ModelConfig = struct {
    dit_path: []const u8,
    /// Conditioner / decoder OVERRIDES. Empty means "not configured", which is
    /// how `pipeline.Options` spells it too (`openIfGiven` returns null for an empty
    /// path, and the resolver then takes the component out of the primary
    /// checkpoint). Non-empty is an *explicit* request that outranks a bundled copy
    /// see `applyPaths`, which is the single place that turns these into
    /// `explicit_text_encoder` / `explicit_vae`.
    vae_path: []const u8,
    text_encoder_path: []const u8,
    /// SDXL's second text tower (OpenCLIP bigG). Same empty-means-unset rule; the
    /// pipeline resolves it independently of `text_encoder_path` and ignores it
    /// entirely for a single-tower architecture.
    text_encoder_2_path: []const u8,
    backend: pipeline.Backend,
    vae_decode: pipeline.VaeDecode,

    /// Duplicate the path strings into gpa-owned storage. Takes a borrowed
    /// `ModelConfig` rather than a positional path list: with four paths, the
    /// call sites were one transposition away from loading the VAE as a text
    /// encoder, and the compiler could not have caught it.
    fn dupe(gpa: std.mem.Allocator, src: ModelConfig) !ModelConfig {
        var out = src;
        // Allocate into a fixed-size array of the owned slices so a partial
        // failure frees exactly what was taken.
        const fields = [_]*[]const u8{ &out.dit_path, &out.vae_path, &out.text_encoder_path, &out.text_encoder_2_path };
        const srcs = [_][]const u8{ src.dit_path, src.vae_path, src.text_encoder_path, src.text_encoder_2_path };
        var done: usize = 0;
        errdefer for (fields[0..done]) |f| gpa.free(f.*);
        while (done < fields.len) : (done += 1) fields[done].* = try gpa.dupe(u8, srcs[done]);
        return out;
    }

    fn deinit(self: ModelConfig, gpa: std.mem.Allocator) void {
        gpa.free(self.dit_path);
        gpa.free(self.vae_path);
        gpa.free(self.text_encoder_path);
        gpa.free(self.text_encoder_2_path);
    }

    /// Do these two configs need the same resident pipeline? (Paths + backend +
    /// decode path, the fields `pipeline.Session.init` keys the session on.)
    fn eql(a: ModelConfig, b: ModelConfig) bool {
        return a.backend == b.backend and a.vae_decode == b.vae_decode and
            std.mem.eql(u8, a.dit_path, b.dit_path) and
            std.mem.eql(u8, a.vae_path, b.vae_path) and
            std.mem.eql(u8, a.text_encoder_path, b.text_encoder_path) and
            std.mem.eql(u8, a.text_encoder_2_path, b.text_encoder_2_path);
    }

    /// Write this model set onto `opts`, the only place the GUI's paths become
    /// pipeline options, so the explicit-override flags cannot be forgotten at one
    /// of the several call sites that build an `Options`.
    ///
    /// A path is "explicit" exactly when it is non-empty, and that equivalence
    /// only holds because the GUI never pre-fills the two override fields. The
    /// distinction is not cosmetic: the CLI's `Options` ships *defaulted* krea2
    /// side paths, and treating one of those as a request is what broke a joined
    /// SD1.5 checkpoint, the resolver opened krea2's qwen3 encoder, hunted for
    /// CLIP's tensors in it, and reported `ComponentNotInCheckpoint`. If a future
    /// settings screen ever suggests a path, it must leave the buffer empty until
    /// the user accepts it, or carry a separate "asked for" bit.
    pub fn applyTo(self: ModelConfig, opts: *pipeline.Options) void {
        opts.dit_path = self.dit_path;
        opts.text_encoder_path = self.text_encoder_path;
        opts.explicit_text_encoder = self.text_encoder_path.len > 0;
        opts.text_encoder_2_path = self.text_encoder_2_path;
        opts.explicit_text_encoder_2 = self.text_encoder_2_path.len > 0;
        opts.vae_path = self.vae_path;
        opts.explicit_vae = self.vae_path.len > 0;
        opts.backend = self.backend;
        opts.vae_decode = self.vae_decode;
    }
};

/// An image awaiting or undergoing generation. Progress/status fields are
/// atomics written by the diffusion worker and read by the UI thread; `rgba` is
/// published before `status` flips to done (acquire/release), so a done image
/// always has its pixels. Also used (status pre-set to `.done`) for images the
/// user attaches for the model to see.
pub const GenImage = struct {
    prompt: []u8, // owned (gpa)
    status: std.atomic.Value(u8) = .init(@intFromEnum(GenStatus.pending)),
    step: std.atomic.Value(u32) = .init(0),
    total: std.atomic.Value(u32) = .init(0),
    rgba: ?[]u8 = null, // interleaved RGBA, [h][w][4]; gpa-owned
    width: usize = 0,
    height: usize = 0,
    // Live preview (RGBA), allocated (generously) before the diffusion worker
    // starts and overwritten in place each step, the pointer is stable for the
    // whole generation (no fat-pointer race), byte tearing on a UI read is
    // benign (rendered .always). Dimensions vary (latent2rgb ≈ latent res, taew
    // ≈ 256px), published via atomics; 0 means "no preview yet".
    preview: ?[]u8 = null,
    preview_w: std.atomic.Value(u32) = .init(0),
    preview_h: std.atomic.Value(u32) = .init(0),
    /// Set by the UI's Cancel button. Polled by the diffusion pipeline between
    /// steps (a generating image aborts) and by the source's next-pending scan
    /// (a queued image is dropped before it starts).
    cancel: std.atomic.Value(bool) = .init(false),
    /// Why this image failed (`@intFromError`), 0 while it has not. Same shape as
    /// `Diffuser.load_error`, and read through `failure()` so an invalid code can
    /// never reach `@errorFromInt`.
    ///
    /// Recorded because the reason has to leave this thread: the LLM's tool
    /// outcome line reports it back to the model ("out of VRAM" is actionable,
    /// "failed" is not), and the retry UI needs it. Logging it is not enough: that
    /// makes it visible to nobody but a terminal.
    gen_error: std.atomic.Value(u16) = .init(0),
    wake: *const fn () void,
    /// Clock source for the worker's timing timestamps (set at creation).
    io: std.Io,
    // Timing (ns, wall clock; written by the diffusion worker, read by the UI).
    // start/done bracket the whole generation (incl. model load); first/last
    // step bracket the sampling loop for an accurate s/step. 0 = not set yet.
    start_ns: std.atomic.Value(i64) = .init(0),
    first_step_ns: std.atomic.Value(i64) = .init(0),
    last_step_ns: std.atomic.Value(i64) = .init(0),
    done_ns: std.atomic.Value(i64) = .init(0),
    // Requested generation params (from the tool call / studio form; defaults
    // from config). Seed 0 means "assign a fresh one at dispatch".
    req_width: usize = 1024,
    req_height: usize = 1024,
    req_steps: usize = 20,
    req_seed: u64 = 0,
    // Extra studio-only params (unused by the chat tool-call path, which leaves
    // them at these defaults, matching the pre-studio behavior).
    req_negative: []u8 = "", // owned (gpa) when non-empty; "" = none
    req_cfg: f32 = 1.0,
    /// Model/backend snapshot, stamped by `enqueue` (gpa-owned paths). Null for
    /// images that never went through the queue (e.g. user-attached inputs, tests);
    /// the worker falls back to the engine's live config for those.
    model: ?ModelConfig = null,
    /// Where the finished PNG was written, or null when saving was off or the
    /// write failed. gpa-owned. This is what a saved conversation stores so a
    /// reload can show the render instead of producing it again.
    saved_path: ?[]u8 = null,
    /// Set once this image's outcome has been reported to the model, so a
    /// terminal state is announced exactly once. UI-thread only.
    ///
    /// An image REBUILT from a reopened conversation sets this at construction:
    /// it was reported when it was first generated, and re-announcing every old
    /// render on each load would bury the one that just happened.
    outcome_noted: bool = false,
    /// Where this image came from. The queue rail says so on a waiting row,
    /// because "the model asked for this" and "I set this up by hand" are
    /// different enough that a user reordering a queue needs to tell them apart.
    from_studio: bool = false,
    /// In-flight sampler state saved when the image was suspended by an
    /// unload-while-paused (status `.suspended`); gpa-owned. On the next dispatch
    /// the worker passes it as `opts.resume_from` to continue bit-identically,
    /// then frees it. See ops/pause.zig + pipeline.Snapshot. (Tier 3.)
    resume_snapshot: ?pipeline.Snapshot = null,

    pub fn get(self: *const GenImage) GenStatus {
        return @enumFromInt(self.status.load(.acquire));
    }

    /// Why this image failed, or null if it did not (or has not yet).
    pub fn failure(self: *const GenImage) ?anyerror {
        const code = self.gen_error.load(.acquire);
        return if (code == 0) null else @errorFromInt(code);
    }

    /// Record a failure with its cause. Store the reason BEFORE the status: the
    /// UI thread learns of the failure by polling `status`, so the other order
    /// would let it read a fresh `.failed` beside a stale (zero) reason.
    pub fn fail(self: *GenImage, err: anyerror) void {
        self.gen_error.store(@intFromError(err), .release);
        self.status.store(@intFromEnum(GenStatus.failed), .release);
    }
};

pub fn freeGenImage(gpa: std.mem.Allocator, gi: *GenImage) void {
    if (gi.saved_path) |p| gpa.free(p);
    gpa.free(gi.prompt);
    if (gi.req_negative.len > 0) gpa.free(gi.req_negative);
    if (gi.rgba) |r| gpa.free(r);
    if (gi.preview) |p| gpa.free(p);
    if (gi.model) |m| m.deinit(gpa);
    if (gi.resume_snapshot) |*s| s.deinit(gpa);
    gpa.destroy(gi);
}

/// Diffusion configuration for a session. The architecture is detected from the
/// primary checkpoint (`pipeline.detectFamily`), never configured here.
pub const DiffConfig = struct {
    /// Primary checkpoint (safetensors or GGUF, sniffed by magic). The only
    /// required path, it may carry the conditioner and decoder as well.
    dit_path: []const u8,
    /// Conditioner / decoder overrides; "" = take them from the primary
    /// checkpoint. See `ModelConfig`'s note on what non-empty means.
    vae_path: []const u8 = "",
    text_encoder_path: []const u8 = "",
    /// SDXL's second text tower; "" for every other architecture.
    text_encoder_2_path: []const u8 = "",
    steps: usize = 20,
    width: usize = 1024,
    height: usize = 1024,
    /// Sampler for the next image. Load-neutral (per-render state), so this is
    /// reconciled live like the preview settings rather than forcing a rebuild.
    sampler: tp.sampler.Kind = .euler,
    /// Where the steps go; null = the architecture's own default. Load-neutral.
    scheduler: ?tp.sampler.Scheduler = null,
    /// Prompt dialect + (a1111 only) weighting form. Reconciled live like the sampler:
    /// they are per-render parsing choices, not part of the loaded session.
    prompt_syntax: pipeline.PromptSyntax = .comfy,
    emphasis: pipeline.Emphasis = .original,
    compat: pipeline.Compat = .comfy,
    backend: pipeline.Backend = .zig_cuda,
    /// VAE decode-path override (see pipeline.VaeDecode). Default adaptive.
    vae_decode: pipeline.VaeDecode = .auto,
    /// 0 = auto (query live free VRAM); weights past the cap stream per step
    /// so diffusion coexists with the resident LLM.
    vram_budget: u64 = 0,
    /// Optional taew2_1 approx-VAE for a sharper live preview (else latent2rgb).
    taew_path: ?[]const u8 = null,
    /// Latent-resolution divisor for the TAESD preview (see pipeline.Options.
    /// preview_ds). 0 = adaptive default.
    preview_ds: usize = 0,
    /// Show a live preview while sampling. When false, no per-step preview is
    /// computed (the "None" preview method). When true, `taew_path` selects
    /// TAESD vs. the built-in latent2rgb fallback.
    preview_enabled: bool = true,
    /// Directory finished images are written to (with AUTOMATIC1111 metadata).
    /// Null (or empty) disables saving. Duped into the engine on init.
    output_dir: ?[]const u8 = null,
};

/// Injected VRAM coordination. The chat session backs these with LLM layer
/// eviction/promotion; the studio passes `VramCoordinator.none` (no LLM to
/// contend with, so diffusion pins all free VRAM).
pub const VramCoordinator = struct {
    ctx: *anyopaque,
    /// Called as the image queue starts (contention): free room for the image
    /// model. Chat evicts LLM layers per its priority; the studio no-ops.
    enter: *const fn (ctx: *anyopaque) void,
    /// Called when the queue drains: undo `enter` (chat promotes layers back).
    exit: *const fn (ctx: *anyopaque) void,
    /// Resident-weight budget (bytes) for the next image; weights past it
    /// stream per step. 0 = auto / pin all free VRAM (the studio's answer).
    budget: *const fn (ctx: *anyopaque) u64,
    /// VAE-OOM reclaim: free ~`needed` bytes of device memory held elsewhere
    /// (chat migrates just enough LLM layers to the CPU) and return the bytes
    /// actually freed.
    reclaim: *const fn (ctx: *anyopaque, needed: u64) u64,

    fn noEnter(_: *anyopaque) void {}
    fn noExit(_: *anyopaque) void {}
    fn zeroBudget(_: *anyopaque) u64 {
        return 0;
    }
    fn noReclaim(_: *anyopaque, _: u64) u64 {
        return 0;
    }

    /// No-LLM coordinator: diffusion has the device to itself.
    pub const none: VramCoordinator = .{
        .ctx = undefined,
        .enter = noEnter,
        .exit = noExit,
        .budget = zeroBudget,
        .reclaim = noReclaim,
    };
};

/// Map the config's engine-decoupled backend enum onto `pipeline.Backend`.
pub fn toPipelineBackend(b: config.Backend) pipeline.Backend {
    return switch (b) {
        .cpu => .cpu,
        .vulkan => .vulkan,
        .zig_cuda => .zig_cuda,
        .cuda => .cuda,
    };
}

/// Map the config's decode-path enum onto `pipeline.VaeDecode`.
pub fn toPipelineVae(v: config.VaeDecode) pipeline.VaeDecode {
    return switch (v) {
        .auto => .auto,
        .whole => .whole,
        .gpu_tiled => .gpu_tiled,
        .cpu_tiled => .cpu_tiled,
    };
}

/// Map the config's sampler enum onto `sampler.Kind`.
/// Config -> pipeline, for the two prompt-dialect knobs. Separate enums because the
/// config's carry UI labels and a stable serialized spelling.
pub fn toPipelineSyntax(s: config.PromptSyntax) pipeline.PromptSyntax {
    return switch (s) {
        .comfy => .comfy,
        .a1111 => .a1111,
    };
}

pub fn toPipelineCompat(c: config.Compat) pipeline.Compat {
    return switch (c) {
        .comfy => .comfy,
        .a1111 => .a1111,
    };
}

pub fn toPipelineEmphasis(e: config.Emphasis) pipeline.Emphasis {
    return switch (e) {
        .original => .original,
        .no_norm => .no_norm,
        .ignore => .ignore,
    };
}

pub fn toPipelineSampler(s: config.Sampler) tp.sampler.Kind {
    return switch (s) {
        .euler => .euler,
        .dpmpp_2m_sde => .dpmpp_2m_sde,
        .dpmpp_2m_sde_heun => .dpmpp_2m_sde_heun,
    };
}

/// Map the config's scheduler enum onto `?schedule.Scheduler`; `.default` -> null,
/// meaning "the architecture's own", which `Session.scheduleWith` resolves per family.
pub fn toPipelineScheduler(s: config.Scheduler) ?tp.sampler.Scheduler {
    return switch (s) {
        .default => null,
        .normal => .normal,
        .karras => .karras,
        .exponential => .exponential,
        .sgm_uniform => .sgm_uniform,
        .simple => .simple,
        .ddim_uniform => .ddim_uniform,
        .beta => .beta,
        .linear_quadratic => .linear_quadratic,
        .kl_optimal => .kl_optimal,
    };
}

/// Round to a multiple of 16 (pipeline requirement) within sane bounds.
pub fn clampDim(n: usize) usize {
    const c = std.math.clamp(n, 256, 4096);
    return c / 16 * 16;
}

/// Parse `key=value` tokens from an `<image ...>` tag into the GenImage.
pub fn parseGenAttrs(attrs: []const u8, gi: *GenImage) void {
    var it = std.mem.tokenizeAny(u8, attrs, " \t");
    while (it.next()) |tok| {
        const eq = std.mem.indexOfScalar(u8, tok, '=') orelse continue;
        const key = tok[0..eq];
        const val = std.mem.trim(u8, tok[eq + 1 ..], "\"'");
        if (std.mem.eql(u8, key, "width")) {
            if (std.fmt.parseInt(usize, val, 10)) |n| gi.req_width = clampDim(n) else |_| {}
        } else if (std.mem.eql(u8, key, "height")) {
            if (std.fmt.parseInt(usize, val, 10)) |n| gi.req_height = clampDim(n) else |_| {}
        } else if (std.mem.eql(u8, key, "steps")) {
            if (std.fmt.parseInt(usize, val, 10)) |n| gi.req_steps = std.math.clamp(n, 1, 100) else |_| {}
        } else if (std.mem.eql(u8, key, "seed")) {
            if (std.fmt.parseInt(u64, val, 10)) |n| gi.req_seed = n else |_| {}
        }
    }
}

/// Wall-clock now in nanoseconds (real-time ns fit comfortably in i64).
pub fn nowNs(io: std.Io) i64 {
    return @intCast(std.Io.Clock.real.now(io).nanoseconds);
}

/// The AUTOMATIC1111 name for the sigma schedule a family samples on.
///
/// Not cosmetic, this field is what a reader re-renders with. krea2 walks
/// a continuous flow-matching schedule (ComfyUI calls it "Simple"); the SD family
/// walks the discrete beta ladder linearly (A1111's default, "Normal", as opposed
/// to "Karras"). Stamping every render "Simple" would tell anyone reproducing an
/// SDXL image to use the wrong discretization. Null when the architecture is not
/// known, and then the field is omitted rather than guessed.
/// The scheduler a family samples with when the caller did not choose one. Mirrors
/// `schedule.Scheduler.defaultFor`, which takes a sigma table rather than a family.
fn defaultSchedulerFor(fam: pipeline.Family) tp.sampler.Scheduler {
    return switch (fam) {
        // All three flow-matching families; `simple` is what ComfyUI's own templates
        // select for each, including Anima's `image_anima_base_v1`, which pairs it
        // with 30 steps and cfg 4.
        .krea2, .zimage, .anima => .simple,
        .sd15, .sdxl => .normal,
    };
}

/// Build an AUTOMATIC1111-style `parameters` metadata string. Format:
///
///     <prompt>
///     Negative prompt: <negative>            (omitted when empty)
///     Steps: N, Sampler: Euler, Schedule type: S, CFG scale: C, Seed: S, Size: WxH, Model: <name>, Prompt syntax: X
///
/// `model_name` is the diffusion checkpoint's file stem; `fam` names the schedule
/// (the "Schedule type" field is dropped when it is null). Caller frees.
///
/// `Prompt syntax` is not decoration, and the block is wrong without it. A reader
/// (including ComfyUI's own metadata importer) re-renders from these fields, and the very
/// same prompt text means a DIFFERENT image in the two dialects, `(x:1.2)` multiplies in
/// one and replaces in the other, `[x]` is de-emphasis in one and literal text in the
/// other. A1111's own format has no field for this because A1111 only ever has one
/// dialect; here it has to be recorded, on the same reasoning that made `Sampler` and
/// `Schedule type` stop being hardcoded. `Emphasis` rides along only when it can matter.
///
/// `Compat` is the same argument again, and for a bigger effect. It selects whose
/// *sampling* conventions ran, including which RNG drew the noise, which decides whether
/// a seed means the same starting latent at all. `RNG`/`SGM noise multiplier` appear only
/// when they were overridden away from that compat's own defaults, so an ordinary ComfyUI
/// render's block carries neither. `RNG` keeps A1111's
/// own spelling of the field and its values, since that is what a reader will recognize.
pub fn buildA1111Params(
    gpa: std.mem.Allocator,
    prompt: []const u8,
    negative: []const u8,
    steps: usize,
    cfg: f32,
    seed: u64,
    width: usize,
    height: usize,
    model_name: []const u8,
    fam: ?pipeline.Family,
    samp: tp.sampler.Kind,
    sched: ?tp.sampler.Scheduler,
    syntax: pipeline.PromptSyntax,
    emphasis: pipeline.Emphasis,
    compat: pipeline.Compat,
    /// The resolved form, compared against `compat`'s own defaults to decide which
    /// overrides need recording.
    cc: pipeline.CompatConfig,
) ![]u8 {
    const neg_line = if (negative.len > 0)
        try std.fmt.allocPrint(gpa, "Negative prompt: {s}\n", .{negative})
    else
        try gpa.dupe(u8, "");
    defer gpa.free(neg_line);
    // The scheduler actually used, not a guess from the architecture. Deriving it
    // from the family is right only while the scheduler is not selectable, and a
    // reader (including ComfyUI's own metadata importer) re-renders from this field.
    // An explicit choice wins; otherwise name the
    // family's default, and drop the field entirely when even that is unknown.
    const sched_line = if (sched) |sc|
        try std.fmt.allocPrint(gpa, "Schedule type: {s}, ", .{sc.a1111Name()})
    else if (fam) |f|
        try std.fmt.allocPrint(gpa, "Schedule type: {s}, ", .{defaultSchedulerFor(f).a1111Name()})
    else
        try gpa.dupe(u8, "");
    defer gpa.free(sched_line);
    const syntax_name: []const u8 = switch (syntax) {
        .comfy => "ComfyUI",
        .a1111 => "A1111",
    };
    // Only under `.a1111`, under `.comfy` there is exactly one weighting form, so the
    // field would imply a choice that does not exist.
    const emph_line = if (syntax == .a1111)
        try std.fmt.allocPrint(gpa, ", Emphasis: {s}", .{switch (emphasis) {
            .original => "Original",
            .no_norm => "No norm",
            .ignore => "Ignore",
        }})
    else
        try gpa.dupe(u8, "");
    defer gpa.free(emph_line);

    const compat_line = if (compat == .comfy)
        try gpa.dupe(u8, "")
    else
        try gpa.dupe(u8, ", Compat: A1111");
    defer gpa.free(compat_line);

    // Only the knobs that were actually overridden, so the common block is unchanged.
    const base: pipeline.CompatConfig = .of(compat);
    var extra: std.ArrayList(u8) = .empty;
    defer extra.deinit(gpa);
    if (cc.noise_src != base.noise_src) try extra.print(gpa, ", RNG: {s}", .{switch (cc.noise_src) {
        .torch_cpu => "CPU",
        .nv_philox => "NV",
    }});
    if (cc.sgm_noise_multiplier != base.sgm_noise_multiplier) {
        try extra.print(gpa, ", SGM noise multiplier: {s}", .{if (cc.sgm_noise_multiplier) "True" else "False"});
    }
    if (cc.quantize_timestep != base.quantize_timestep) {
        try extra.print(gpa, ", Quantize timesteps: {s}", .{if (cc.quantize_timestep) "True" else "False"});
    }

    return std.fmt.allocPrint(
        gpa,
        "{s}\n{s}Steps: {d}, Sampler: {s}, {s}CFG scale: {d:.1}, Seed: {d}, Size: {d}x{d}, Model: {s}, Prompt syntax: {s}{s}{s}{s}",
        .{ prompt, neg_line, steps, samp.a1111Name(), sched_line, cfg, seed, width, height, model_name, syntax_name, emph_line, compat_line, extra.items },
    );
}

/// The fields of an AUTOMATIC1111 `parameters` block that describe a REQUEST,
/// as read back off a saved PNG. Slices borrow the input.
pub const A1111Params = struct {
    prompt: []const u8 = "",
    negative: []const u8 = "",
    steps: ?usize = null,
    cfg: ?f32 = null,
    seed: ?u64 = null,
    width: ?usize = null,
    height: ?usize = null,
};

/// Parse what `buildA1111Params` wrote. The saved PNG is the record of how an
/// image was made, so reopening one reads it back rather than the transcript
/// carrying a second copy that can disagree with the file.
///
/// Deliberately lenient: this also has to read blocks written by ComfyUI and
/// A1111 themselves, where field order, spelling and which keys are present all
/// vary. Anything unrecognised is skipped and the field stays null.
pub fn parseA1111Params(text: []const u8) A1111Params {
    var out: A1111Params = .{};

    // The settings line is the LAST line that starts with "Steps:"; everything
    // before it is prompt (and negative). Searching from the end is what keeps a
    // prompt that itself contains "Steps:" from splitting the block early.
    const neg_tag = "\nNegative prompt:";
    var head_end = text.len;
    var settings: []const u8 = "";
    var it = std.mem.splitBackwardsScalar(u8, text, '\n');
    var scanned: usize = 0;
    while (it.next()) |line| {
        scanned += line.len + 1;
        if (std.mem.startsWith(u8, line, "Steps:")) {
            settings = line;
            head_end = text.len - scanned + 1;
            break;
        }
    }
    const head = std.mem.trimEnd(u8, text[0..@min(head_end, text.len)], "\n");

    if (std.mem.indexOf(u8, head, neg_tag)) |i| {
        out.prompt = std.mem.trim(u8, head[0..i], " \t\r\n");
        out.negative = std.mem.trim(u8, head[i + neg_tag.len ..], " \t\r\n");
    } else {
        out.prompt = std.mem.trim(u8, head, " \t\r\n");
    }

    var f = std.mem.splitScalar(u8, settings, ',');
    while (f.next()) |field| {
        const colon = std.mem.indexOfScalar(u8, field, ':') orelse continue;
        const key = std.mem.trim(u8, field[0..colon], " \t");
        const val = std.mem.trim(u8, field[colon + 1 ..], " \t");
        if (std.mem.eql(u8, key, "Steps")) {
            out.steps = std.fmt.parseInt(usize, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "CFG scale")) {
            out.cfg = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Seed")) {
            out.seed = std.fmt.parseInt(u64, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "Size")) {
            const x = std.mem.indexOfScalar(u8, val, 'x') orelse continue;
            out.width = std.fmt.parseInt(usize, val[0..x], 10) catch null;
            out.height = std.fmt.parseInt(usize, val[x + 1 ..], 10) catch null;
        }
    }
    return out;
}

/// The file stem of a path (basename minus final extension), the a1111 "Model".
fn modelStem(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    return if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
}

pub fn rgbToRgba(gpa: std.mem.Allocator, rgb: []const u8, w: usize, h: usize) ![]u8 {
    const rgba = try gpa.alloc(u8, w * h * 4);
    for (0..w * h) |i| {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

/// The diffusion engine. Compose one per driver (chat / studio); at most one is
/// ever alive at a time in tp-gui (a mode switch tears one down before building
/// the other), so only one diffusion pipeline is ever resident.
pub const Diffuser = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    wake: *const fn () void,

    /// Base sampling options (paths / backend / preview / default dims); the
    /// per-image params are overlaid from the `GenImage` in `worker`.
    opts: pipeline.Options,
    /// Fresh, distinct per-image seed source (advanced by `nextSeed`).
    seed: u64,
    busy: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// Pause gate for the sampling worker (see ops/pause.zig), driven by the
    /// diffusion pause button (independent of the LLM's). Stable-addressed (the
    /// Diffuser outlives its worker; deinit joins the thread first), so the
    /// per-image `opts.pause` can point at it.
    pause: pause_gate.Gate = .{},
    /// Persistent pipeline (loads the image model once, stays resident across a
    /// queue; freed when the queue drains, on a model swap, or on teardown).
    session: std.atomic.Value(?*pipeline.Session) = .init(null),
    /// taew (approx-VAE) path duped into `path_store`, kept so `setPreview` can
    /// re-enable the TAESD preview without re-duping. Null if none configured.
    taew_owned: ?[]const u8 = null,
    preview_method: config.Preview = .taesd,
    /// Live preview controls, shared with the running worker's sampling loop (via
    /// `opts.preview_live`). `setPreview`/`setPreviewSize` write these so a method
    /// or quality change shows on the next step, even mid-image. Address is stable
    /// (the Diffuser outlives its worker; deinit joins the thread first).
    live_preview: pipeline.LivePreview = .{},
    /// Backs the live path strings after a model swap (reset+re-dupe per swap).
    /// The initial paths point into the caller's arena.
    path_store: std.heap.ArenaAllocator,
    /// The model config the resident `session` is currently loaded for (gpa-owned
    /// dupes; null when no session). The worker compares the next image's snapshot
    /// against this to decide whether to reuse the pipeline or reload at the seam.
    loaded: ?ModelConfig = null,

    /// Directory finished images are saved to (gpa-owned; null = saving off).
    /// Updated live from settings via `setOutputDir` (UI thread, queue idle);
    /// read by the worker after a generation completes.
    output_dir: ?[]u8 = null,

    /// The single unified image queue + history: every generated image (chat
    /// tool-call and studio) lives here, in creation order. The engine OWNS
    /// these (freed on deinit); the chat transcript and the studio gallery both
    /// view into it (borrowed `*GenImage`). Drained FIFO, one at a time.
    queue: std.ArrayList(*GenImage) = .empty,

    vram: VramCoordinator,
    /// True between `vram.enter` (an image started) and the matching
    /// `vram.exit` (queue drained), so `pump` fires the coordinator hooks on
    /// queue EDGES only. `pump` runs every frame, and unconditionally calling
    /// `exit` on every empty-queue frame made the arbiter rebalance (and
    /// publish an LLM residency target) at frame rate for no reason.
    vram_entered: bool = false,
    /// Guards this engine's device RESIDENCY against a foreign thread.
    ///
    /// `trimToBudget`/`giveUpToBudget` bind this engine's device context and free
    /// its weights, and since diffusion became a `tp.vram.Participant` the arbiter
    /// can invoke them from the LLM's worker thread (a starving LLM reclaiming
    /// from an idle image model). That would otherwise race the UI thread's
    /// `pump`, which frees the session at a model seam and flips `busy` + spawns
    /// the worker, the `busy` check alone is a TOCTOU, since the flag is set on
    /// a different thread than the one evicting.
    ///
    /// Held across pump's free/spawn decision and across each yield. Foreign
    /// callers use `tryLock` and DECLINE rather than block, so no lock-ordering
    /// cycle with the LLM-side `g_session_mu` is possible even if a future
    /// reclaim path nests the two.
    res_mu: std.Io.Mutex = std.Io.Mutex.init,
    /// Cross-thread residency intent published by the app's `vram.Arbiter` (the
    /// `tp.vram.Participant` half of `res_mu`'s story). Unlike the LLM, this engine
    /// has no per-step poll of it: mid-image eviction is exactly what soft
    /// residency avoids, so a published target is enacted only while idle. Kept so
    /// both participants have the same shape (and a future step-boundary poll is a
    /// fill-in, not a re-architecture).
    control: tp.vram.ControlPoint = .{},
    /// High-water of MEASURED resident bytes, across generations. Supersedes the
    /// checkpoint-file estimate in `vpDemand` as soon as one image has run.
    peak_resident: std.atomic.Value(u64) = .init(0),

    /// Why the last model LOAD failed, so the UI can say what went wrong instead
    /// of showing a bare "failed" per image. Stored as `@intFromError` (0 = no
    /// failure) because the worker writes it and the UI thread reads it; an error
    /// code is a plain integer, so this needs no lock and no owned string.
    ///
    /// This exists because the SD family made the most likely first-run mistake
    /// invisible: an SD1.5 checkpoint on the GUI's default CUDA backend returns
    /// `error.UnsupportedBackend` (there are no device kernels for its UNet), and a
    /// log line is a trace nobody in the GUI sees.
    load_error: std.atomic.Value(u16) = .init(0),
    /// Architecture of the resident pipeline, for the UI (`@intFromEnum` + 1;
    /// 0 = nothing loaded). Detected by the pipeline from the checkpoint itself.
    loaded_family: std.atomic.Value(u8) = .init(0),
    /// A foreign thread has asked for the pipeline to be dropped
    /// (`requestRelease`); the UI thread enacts it in `fulfillRelease`.
    release_req: std.atomic.Value(bool) = .init(false),

    /// Build the engine from a `DiffConfig`. `wake` repaints the UI on progress;
    /// `vram` is the injected VRAM coordinator (LLM eviction, or `.none`). The
    /// path slices in `cfg` must outlive the diffuser until the first model swap
    /// (they point into the caller's load-once arena).
    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        wake: *const fn () void,
        cfg: DiffConfig,
        vram: VramCoordinator,
    ) Diffuser {
        var opts: pipeline.Options = .{
            .prompt = "",
            .width = cfg.width,
            .height = cfg.height,
            .steps = cfg.steps,
            .sampler = cfg.sampler,
            .scheduler = cfg.scheduler,
            .vram_budget = cfg.vram_budget,
            .prompt_syntax = cfg.prompt_syntax,
            .emphasis = cfg.emphasis,
            .compat = cfg.compat,
            .preview = cfg.preview_enabled,
            .taew_path = cfg.taew_path,
            .preview_ds = cfg.preview_ds,
        };
        // Paths + backend + the explicit-override flags, in one place. Note this
        // OVERWRITES `Options`' defaulted krea2 side paths with "" when nothing is
        // configured, which is what makes a bundled checkpoint resolve out of its
        // own file instead of against a stale default.
        (ModelConfig{
            .dit_path = cfg.dit_path,
            .vae_path = cfg.vae_path,
            .text_encoder_path = cfg.text_encoder_path,
            .text_encoder_2_path = cfg.text_encoder_2_path,
            .backend = cfg.backend,
            .vae_decode = cfg.vae_decode,
        }).applyTo(&opts);
        return .{
            .gpa = gpa,
            .io = io,
            .wake = wake,
            .opts = opts,
            .seed = 0,
            .taew_owned = cfg.taew_path,
            .path_store = std.heap.ArenaAllocator.init(gpa),
            .vram = vram,
            .output_dir = if (cfg.output_dir) |o|
                (if (o.len > 0) gpa.dupe(u8, o) catch null else null)
            else
                null,
        };
    }

    /// Append a generation request to the unified queue (engine takes ownership
    /// of `gi`). The caller may keep a borrowed pointer for its own display.
    ///
    /// Stamps the CURRENT model config (paths / backend / VAE decode) onto the
    /// image so a later backend/model switch only affects images enqueued after
    /// this one, already-queued images finish on the config they were created
    /// with (the worker reloads the pipeline at the seam where they disagree).
    pub fn enqueue(self: *Diffuser, gi: *GenImage) !void {
        if (gi.model == null) gi.model = try ModelConfig.dupe(self.gpa, self.liveConfig());
        try self.queue.append(self.gpa, gi);
    }

    /// The model set the engine would load right now, read back out of `opts`.
    ///
    /// `opts` is the single store for the live config (the worker copies it and
    /// overlays a per-image snapshot), so every "what is currently selected?"
    /// question has to reconstruct a `ModelConfig` from it. Doing that inline at
    /// each of the four sites is how one of them silently misses a newly added
    /// path, which is exactly what a fourth component invites.
    ///
    /// Borrowed: the slices point into `path_store` (or the init-time arena) and
    /// live until the next `requestPaths`.
    fn liveConfig(self: *const Diffuser) ModelConfig {
        return .{
            .dit_path = self.opts.dit_path,
            .vae_path = self.opts.vae_path,
            .text_encoder_path = self.opts.text_encoder_path,
            .text_encoder_2_path = self.opts.text_encoder_2_path,
            .backend = self.opts.backend,
            .vae_decode = self.opts.vae_decode,
        };
    }

    /// Re-queue a failed or canceled image, IN PLACE. The `*GenImage` is unchanged,
    /// so a chat variant holding a borrowed pointer re-renders the same tile going
    /// pending -> generating -> done: "try this one again", not "make another".
    ///
    /// The model snapshot is REFRESHED to the live config, deliberately
    /// inverting `enqueue`'s rule that an image finishes on the config it was
    /// created with. That rule protects a queue from a mid-run switch; a retry is a
    /// fresh request the user is making NOW, and the whole reason to press it after
    /// an OOM is that they just changed something (unloaded the LLM, picked a
    /// smaller model, switched backend). Honouring the old snapshot would retry the
    /// exact configuration that just failed.
    pub fn retry(self: *Diffuser, gi: *GenImage) !void {
        switch (gi.get()) {
            .failed, .canceled => {},
            // Never touch one the worker may be holding, and never re-run a
            // finished one (that would discard a good image in place).
            .pending, .generating, .suspended, .done => return,
        }
        const fresh = try ModelConfig.dupe(self.gpa, self.liveConfig());
        if (gi.model) |m| m.deinit(self.gpa);
        gi.model = fresh;
        // A stale latent from some earlier pause would resume mid-render into a run
        // that has nothing to do with it; a retry starts clean.
        if (gi.resume_snapshot) |*s| {
            s.deinit(self.gpa);
            gi.resume_snapshot = null;
        }
        // Drop the resident pipeline first (only when nothing is in flight,
        // `freeSession` requires that). This is what makes the retry a
        // genuinely different attempt rather than a replay of the conditions
        // that just failed: after a VRAM failure the resident session is
        // holding the memory the retry needs, and after a device fault a fresh
        // session is the only recovery available at all. The cost is one model
        // reload on a path the user explicitly asked for, after a failure.
        //
        // NOT verified to clear a sticky CUDA fault, `CUDA_ERROR_ILLEGAL_
        // ADDRESS` poisons its context, and whether tearing the session down
        // rebuilds far enough to escape that has not been measured here. If it
        // does not, the retry reports the same error again, which is at least
        // honest.
        if (!self.busyNow()) self.freeSession();
        gi.gen_error.store(0, .release);
        gi.step.store(0, .monotonic);
        gi.total.store(0, .monotonic);
        gi.cancel.store(false, .release);
        gi.status.store(@intFromEnum(GenStatus.pending), .release);
        gi.wake();
    }

    /// The unified image list (creation order) for rendering / viewer nav.
    pub fn items(self: *const Diffuser) []*GenImage {
        return self.queue.items;
    }

    /// The image already in `items` that came from `path`, if any.
    ///
    /// A finished render's file is its identity. Reopening a conversation must
    /// find the image it already has rather than building a second one: without
    /// this every load appended another copy and the Library filled with
    /// duplicates of the same picture.
    pub fn findSaved(list: []const *GenImage, path: []const u8) ?*GenImage {
        if (path.len == 0) return null;
        for (list) |gi| {
            const p = gi.saved_path orelse continue;
            if (std.mem.eql(u8, p, path)) return gi;
        }
        return null;
    }

    /// Take ownership of an already-finished image and put it in the unified
    /// list, WITHOUT queueing any work. Used when a reloaded conversation
    /// rebuilds its renders from disk: they are history, so Library and the
    /// viewer should see them like anything else, but there is nothing to
    /// generate. `pump` skips them because they are not `.pending`.
    pub fn adoptFinished(self: *Diffuser, gi: *GenImage) !void {
        try self.queue.append(self.gpa, gi);
    }

    /// Move a still-pending image so it sits before the one currently at
    /// `to_id`, for the queue rail's drag-to-reorder. Both are identified by
    /// pointer, since an index into a filtered view of the queue is not an index
    /// into the queue.
    ///
    /// UI-thread only, which is what makes it safe without a lock: `pump` (also
    /// UI-thread) is the only other mutator, the worker holds one borrowed
    /// `*GenImage` and never touches the list, and a `.generating` image is
    /// refused here so a reorder can never move the one in flight.
    pub fn movePending(self: *Diffuser, move: *GenImage, before: ?*GenImage) void {
        if (move.get() != .pending) return;
        if (before) |b| if (b == move or b.get() != .pending) return;
        const from = std.mem.indexOfScalar(*GenImage, self.queue.items, move) orelse return;
        _ = self.queue.orderedRemove(from);
        const to = if (before) |b|
            (std.mem.indexOfScalar(*GenImage, self.queue.items, b) orelse self.queue.items.len)
        else
            self.queue.items.len;
        self.queue.insert(self.gpa, to, move) catch {
            // Put it back where it was; a failed reorder must not drop the job.
            self.queue.insert(self.gpa, from, move) catch unreachable;
        };
    }

    /// Any image still queued (not yet started)?
    pub fn hasPending(self: *const Diffuser) bool {
        for (self.queue.items) |gi| if (gi.get() == .pending) return true;
        return false;
    }

    /// Cancel every queued/in-flight image (teardown / clear).
    pub fn cancelAll(self: *Diffuser) void {
        for (self.queue.items) |gi| gi.cancel.store(true, .release);
        // A worker parked at the pause gate would never observe the per-image
        // cancel; wake it (without unpausing, the pause button stays in sync)
        // so its checkpoint re-reads `gi.cancel` and returns `.canceled`.
        self.pause.wake(self.io);
    }

    /// Park the sampling worker at the next step boundary (holding the in-flight
    /// latent + resident weights), or release it. UI-thread; driven by the
    /// diffusion pause button.
    pub fn setPaused(self: *Diffuser, paused: bool) void {
        if (paused) self.pause.pause(self.io) else self.pause.unpause(self.io);
    }

    pub fn isPaused(self: *Diffuser) bool {
        return self.pause.isPaused(self.io);
    }

    /// Wake a worker parked at the pause gate so it re-checks its per-image
    /// cancel flag, used by the single-image Cancel button while paused (no-op
    /// when nothing is parked, and never changes the pause state).
    pub fn wakePaused(self: *Diffuser) void {
        self.pause.wake(self.io);
    }

    /// Next image to run, dropping ones canceled before they start. Includes
    /// `.suspended` images (unload-while-paused): the worker resumes them from
    /// their saved latent (opts.resume_from).
    fn nextPending(self: *Diffuser) ?*GenImage {
        for (self.queue.items) |gi| {
            const st = gi.get();
            if (st != .pending and st != .suspended) continue;
            if (gi.cancel.load(.acquire)) {
                gi.status.store(@intFromEnum(GenStatus.canceled), .release);
                if (gi.resume_snapshot) |*s| { // dropping a suspended image: free its state
                    s.deinit(self.gpa);
                    gi.resume_snapshot = null;
                }
                continue;
            }
            return gi;
        }
        return null;
    }

    /// Set the initial per-image seed base (usually derived from the session
    /// seed so repeated runs vary but a session is reproducible).
    pub fn seedBase(self: *Diffuser, base: u64) void {
        self.seed = base;
    }

    /// Advance and return the next distinct seed (unless a caller set one
    /// explicitly on the GenImage).
    pub fn nextSeed(self: *Diffuser) u64 {
        self.seed +%= 0x9E3779B97F4A7C15;
        return self.seed;
    }

    /// Join the worker if one is running (no cancel, an in-flight image is
    /// allowed to FINISH). Used before a session teardown / model swap.
    pub fn stopAndReap(self: *Diffuser) void {
        if (self.thread) |t| {
            t.join();
            self.thread = null;
            self.busy.store(false, .release);
        }
    }

    /// Ask a paused, in-flight worker to suspend: snapshot the in-flight latent
    /// to host, mark the image `.suspended`, and exit. Async, poll `busyNow()`
    /// for completion, then `reapAndFree()`. No-op if nothing is generating. Only
    /// meaningful while paused (the worker must be parked at the step gate to see
    /// the unload request). See ops/pause.zig. (Tier 3 unload-while-paused.)
    pub fn requestSuspend(self: *Diffuser) void {
        self.pause.requestUnload(self.io);
    }

    /// Reap a finished/suspended worker thread and free the resident weights,
    /// KEEPING the queue (incl. any `.suspended` image, which resumes from its
    /// saved latent on the next dispatch). For unload-while-paused: caller must
    /// ensure the worker has stopped (`busyNow() == false`).
    pub fn reapAndFree(self: *Diffuser) void {
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
        self.freeSession();
    }

    /// Free the resident pipeline (returns its VRAM). Binds its own CUDA
    /// context. Caller must ensure no worker is in flight.
    ///
    /// Takes `res_mu` because this is the coarsest residency mutation there is:
    /// the arbiter can be inside `giveUpToBudget` on the LLM's worker thread,
    /// evicting from the very session we are about to destroy. Guarding the
    /// primitive rather than its five call sites means a new caller can't
    /// reintroduce the race. Blocking + uncancelable (the caller owns the
    /// decision); the foreign side tryLocks and declines.
    pub fn freeSession(self: *Diffuser) void {
        self.res_mu.lockUncancelable(self.io);
        defer self.res_mu.unlock(self.io);
        self.freeSessionLocked();
    }

    /// Ask for the resident pipeline to be dropped (the arbiter's release rung,
    /// `vram.Participant.VTable.releaseAll`). Callable from any thread; the free
    /// itself happens in `fulfillRelease` on the UI thread.
    ///
    /// It is a REQUEST and not the free itself because `session` has exactly one
    /// writer by design, and every reader — the status bar's `vramBytes` and
    /// `vramBreakdown` on the frame path, the participant thunks — loads it
    /// unlocked. Freeing it from the LLM worker made all seven of those reads a
    /// use-after-free, and the meter found it within a frame: a SIGSEGV in
    /// `deviceUsed` under `residentLabel`. Locking the readers instead would put a
    /// blocking mutex on every frame, held across a multi-GiB teardown.
    pub fn requestRelease(self: *Diffuser) void {
        if (self.busy.load(.acquire)) return;
        if (self.session.load(.acquire) == null) return;
        self.release_req.store(true, .release);
        self.wake(); // a request made behind a prefill must not wait for a token
    }

    pub fn releaseRequested(self: *const Diffuser) bool {
        return self.release_req.load(.acquire);
    }

    /// Enact a pending `requestRelease`, on the UI thread. The caller holds
    /// whatever guards foreign readers of this engine (the app takes `g_diff_mu`,
    /// which is what the LLM worker holds while it reads the participant).
    /// Re-checks everything: the queue may have moved on since the ask.
    pub fn fulfillRelease(self: *Diffuser) u64 {
        if (!self.release_req.swap(false, .acq_rel)) return 0;
        if (self.busyNow() or self.nextPending() != null) return 0; // work arrived; keep it loaded
        const before = if (self.session.load(.acquire)) |s| s.deviceUsed() else return 0;
        self.freeSession();
        std.log.info("[vram] diffusion released: {d} MiB returned (pipeline unloaded; the next image reloads it)", .{before >> 20});
        return before;
    }

    /// The teardown itself. Callers hold `res_mu`; see `freeSession` for why the
    /// lock belongs on the primitive rather than on each call site.
    fn freeSessionLocked(self: *Diffuser) void {
        if (self.session.load(.acquire)) |s| {
            s.deinit();
            self.session.store(null, .release);
        }
        if (self.loaded) |l| {
            l.deinit(self.gpa);
            self.loaded = null;
        }
        self.loaded_family.store(0, .release);
    }

    /// Why the last model load failed, or null if the last one succeeded (or none
    /// has been tried). Cleared by the next successful load, a stale error would
    /// keep the studio warning about a model the user has since replaced.
    pub fn loadError(self: *const Diffuser) ?anyerror {
        const code = self.load_error.load(.acquire);
        return if (code == 0) null else @errorFromInt(code);
    }

    /// Architecture of the resident pipeline; null when nothing is loaded.
    pub fn loadedFamily(self: *const Diffuser) ?pipeline.Family {
        const v = self.loaded_family.load(.acquire);
        return if (v == 0) null else @enumFromInt(v - 1);
    }

    pub fn deinit(self: *Diffuser) void {
        if (self.thread) |t| t.join();
        self.freeSession();
        if (self.output_dir) |o| self.gpa.free(o);
        self.path_store.deinit();
        // The engine owns every queued image (chat + studio); free them here.
        for (self.queue.items) |gi| freeGenImage(self.gpa, gi);
        self.queue.deinit(self.gpa);
    }

    /// True while an image is generating (status-bar diffusion readout).
    pub fn busyNow(self: *Diffuser) bool {
        return self.busy.load(.acquire);
    }

    /// Device VRAM (bytes) the resident diffusion model actually holds; 0 when
    /// none is loaded.
    pub fn vramBytes(self: *Diffuser) u64 {
        const v = if (self.session.load(.acquire)) |s| s.deviceUsed() else 0;
        // High-water it HERE rather than in the arbiter's `usage` hook: the arbiter
        // only samples on plan events, and there is no plan event between "pipeline
        // loaded" and "queue drained", so it would record the pre-load 0 and miss
        // the peak entirely. The status bar samples this every 500 ms throughout a
        // generation, which is what actually catches it.
        self.notePeak(v);
        return v;
    }

    /// MEASURED per-component VRAM breakdown (TE / DiT / VAE / latent) for the
    /// status-bar meter; all zero when no pipeline is resident.
    pub fn vramBreakdown(self: *Diffuser) VramBreakdown {
        const b: VramBreakdown = if (self.session.load(.acquire)) |s| s.vramBreakdown() else .{};
        // Same high-water as `vramBytes`, because THIS is the accessor the status
        // bar polls every 500 ms, and that poll is the only thing sampling us
        // mid-generation, i.e. the only thing that ever sees the peak. Recording it
        // only in `vramBytes` left the figure at its pre-load 0.
        self.notePeak(b.total());
        return b;
    }

    /// Pipeline bytes NOT on the card: what this model held when it last ran
    /// unconstrained, minus what it holds now. Those weights live in host RAM and
    /// re-upload over PCIe as each step touches them, which is what makes a
    /// squeezed image model slow. 0 while fully resident, and 0 before the first
    /// image (nothing has been measured to compare against).
    ///
    /// Deliberately the same two numbers the arbiter plans with (`vpDemand` minus
    /// `vpUsage`), not a second estimate: the bar then shows what the policy
    /// believes, so a figure that looks wrong on screen is the planner being
    /// wrong, not the display disagreeing with it. Inherits `vpDemand`'s limit,
    /// the peak is a high-water, so an image model that has ONLY ever run
    /// squeezed reads 0 until a roomier image teaches it better.
    pub fn offloadBytes(self: *Diffuser) u64 {
        if (self.session.load(.acquire) == null) return 0; // unloaded, not offloaded
        return self.peak_resident.load(.monotonic) -| self.vramBytes();
    }

    /// High-water the measured resident footprint. Called from every residency
    /// accessor so the peak is caught wherever it happens to be observed.
    fn notePeak(self: *Diffuser, v: u64) void {
        if (v > self.peak_resident.load(.monotonic)) self.peak_resident.store(v, .monotonic);
    }

    /// Free resident diffusion weights to fit `budget` bytes, the GUI VRAM
    /// limit lowered while the queue is idle (soft residency: never mid-image,
    /// which would force per-step streaming). No-op while generating; the next
    /// image re-uploads what fits its budget.
    pub fn trimToBudget(self: *Diffuser, budget: u64) void {
        if (!self.res_mu.tryLock()) return; // pump is deciding / another yield in flight
        defer self.res_mu.unlock(self.io);
        if (self.busy.load(.acquire)) return;
        const s = self.session.load(.acquire) orelse return;
        const before = s.deviceUsed();
        s.trimToBudget(budget);
        const after = s.deviceUsed();
        if (after != before) std.log.info("[vram] diffusion evict (limit {d} MiB): {d}→{d} MiB resident · {d} MiB free", .{
            budget >> 20, before >> 20, after >> 20, s.freeVram() >> 20,
        });
    }

    /// Incrementally free resident diffusion weights to fit `budget` bytes,
    /// keeping the rest resident (unlike `trimToBudget`). Busy-gated (never
    /// mid-image). The arbiter's live "diffusion yields to a growing LLM" lever;
    /// returns bytes freed.
    ///
    /// Binds this engine's device context, so it may only run when no worker owns
    /// it, `res_mu` + the `busy` check together give that guarantee against both
    /// the UI thread's `pump` and the LLM worker's reclaim (see `res_mu`).
    pub fn giveUpToBudget(self: *Diffuser, budget: u64) u64 {
        if (!self.res_mu.tryLock()) return 0;
        defer self.res_mu.unlock(self.io);
        if (self.busy.load(.acquire)) return 0;
        const s = self.session.load(.acquire) orelse return 0;
        const before = s.deviceUsed();
        const freed = s.giveUpToBudget(budget);
        if (freed > 0) std.log.info("[vram] diffusion yield (target {d} MiB): freed {d} MiB ({d}→{d} MiB resident) · {d} MiB free", .{
            budget >> 20, freed >> 20, before >> 20, s.deviceUsed() >> 20, s.freeVram() >> 20,
        });
        return freed;
    }

    // --- tp.vram.Participant adapter --------------------------------------------
    // Makes the image model a first-class arbiter participant alongside the LLM,
    // so `vram.Arbiter.plan` can drive BOTH from one coherent plan. Before this,
    // diffusion was only a budget CONSUMER (it read `diffusionBudget()` at spawn)
    // and the reverse direction, an idle image model giving VRAM back to a
    // growing LLM, was left to a separate app-level call that always cancelled
    // to a no-op. See the `vram.Arbiter` doc comment.

    fn vpUsage(ctx: *anyopaque) u64 {
        return fromCtx(ctx).vramBytes(); // high-waters `peak_resident` internally
    }
    /// Footprint if nothing contended.
    ///
    /// MEASURED once an image has run: the high-water of `vramBytes()`. The
    /// checkpoint-file-size estimate below is only a bootstrap for the very first
    /// image, and it is a poor one, it counts bytes that never become resident
    /// (the pipeline drops the text encoder when the DiT working set won't fit
    /// alongside it). Measured on krea2: files 18961 MiB vs 13052 MiB actually
    /// resident, and since the arbiter turns this number into the LLM's eviction
    /// target, those 6 GiB of phantom demand pushed ~20 extra layers to the host
    /// for memory the image model never used.
    ///
    /// Still nonzero before the first load, deliberately: a diffuser that read as
    /// "wants nothing" would let the LLM plan to keep the whole card and then have
    /// nowhere to load into. `Participant.demand` maxes this with live usage, so an
    /// under-estimate can never shrink below what is actually resident.
    fn vpDemand(ctx: *anyopaque) u64 {
        const self = fromCtx(ctx);
        const peak = self.peak_resident.load(.monotonic);
        return if (peak != 0) peak else self.estimateResidentBytes();
    }
    /// No standing transient reservation. The pipeline sizes each stage against
    /// live free VRAM and has its own reclaim ladder (`vcReclaim` -> tiling ->
    /// CPU), so room it needs above its residency it asks for at the time. The
    /// LLM cannot do that, which is why only the LLM reports headroom.
    fn vpHeadroom(_: *anyopaque) u64 {
        return 0;
    }
    /// Nothing is un-evictable while idle: resident weights are pure cache that
    /// the next image re-uploads. Mid-image the whole working set is the floor,
    /// evicting under a running sampler would force per-step streaming, which soft
    /// residency exists to avoid.
    fn vpFloor(ctx: *anyopaque) u64 {
        const self = fromCtx(ctx);
        return if (self.busyNow()) self.vramBytes() else 0;
    }
    fn vpBusy(ctx: *anyopaque) bool {
        return fromCtx(ctx).busyNow();
    }
    fn vpApply(ctx: *anyopaque, target: u64) void {
        _ = fromCtx(ctx).giveUpToBudget(target);
    }
    /// The arbiter's last rung. `giveUpToBudget` only evicts the weight LRU; the
    /// pipeline underneath it (context, cuBLASLt/cuDNN workspaces, activation and
    /// latent scratch) is not cache and no target reaches it. For an IDLE image
    /// model all of that is recoverable — the next image rebuilds it — so when an
    /// LLM is otherwise going to run layers on the CPU, this hands it back.
    fn vpReleaseAll(ctx: *anyopaque) void {
        fromCtx(ctx).requestRelease();
    }
    fn vpReleasePending(ctx: *anyopaque) bool {
        return fromCtx(ctx).releaseRequested();
    }
    fn fromCtx(ctx: *anyopaque) *Diffuser {
        return @ptrCast(@alignCast(ctx));
    }
    const vp_vtable: tp.vram.Participant.VTable = .{
        .usage = vpUsage,
        .demand = vpDemand,
        .headroom = vpHeadroom,
        .floor = vpFloor,
        .busy = vpBusy,
        .applyBudget = vpApply,
        .releaseAll = vpReleaseAll,
        .releasePending = vpReleasePending,
    };

    /// This image model as a `tp.vram.Participant` the app-level arbiter can drive.
    /// The returned value borrows `self`, so the app must drop it (null out
    /// `Arbiter.diffusion`) before tearing the engine down.
    pub fn participant(self: *Diffuser) tp.vram.Participant {
        return .{ .ctx = self, .control = &self.control, .vtable = &vp_vtable };
    }

    /// Bootstrap estimate of the pipeline's PEAK resident footprint, for a model
    /// no image has run on yet. Superseded by `peak_resident` the moment one has.
    ///
    /// Deliberately EXCLUDES the text encoder. It is the largest checkpoint on
    /// disk here (4999 MiB) but is transient by construction, it encodes the
    /// prompt once and its weights are then droppable, so at the actual peak it
    /// holds 52 MiB. Counting it made the estimate 18961 MiB against a measured
    /// 13055, and since the arbiter turns this into the LLM's eviction target
    /// those 5.9 GiB of phantom demand evicted every layer on the first image.
    /// DiT + VAE gives 13961, within ~900 MiB of the truth.
    ///
    /// Erring LOW is the right direction: an under-estimate means the LLM keeps
    /// more and the pipeline reclaims reactively (`vcReclaim` / the OOM ladder),
    /// whereas an over-estimate is an eviction that was never needed.
    /// A BUNDLED checkpoint therefore over-counts by its text encoder, which the
    /// separate-file case deliberately excludes: the file size is all we have, and
    /// it cannot be broken down per component without parsing the header. Erring
    /// high here is the wrong direction (it over-evicts the LLM on the first
    /// image), but it self-corrects after one generation, `peak_resident`
    /// supersedes this the moment a real measurement exists.
    pub fn estimateResidentBytes(self: *Diffuser) u64 {
        var total: u64 = 0;
        // An unset override contributes nothing (its component is inside the
        // primary checkpoint, already counted).
        for ([_][]const u8{ self.opts.dit_path, self.opts.vae_path }) |p| {
            if (p.len == 0) continue;
            const st = std.Io.Dir.cwd().statFile(self.io, p, .{}) catch continue;
            total += st.size;
        }
        return total;
    }

    /// Seed the measured high-water from a previous session (see
    /// `Config.diff_peak_resident`). Ignored if a bigger figure is already known.
    pub fn seedPeakResident(self: *Diffuser, bytes: u64) void {
        if (bytes > self.peak_resident.load(.monotonic)) self.peak_resident.store(bytes, .monotonic);
    }

    /// The measured peak, for persisting. 0 when no image has run yet. Samples
    /// once more on the way out so a caller at the drain edge still records
    /// something even if nothing polled us during the generation.
    pub fn peakResident(self: *Diffuser) u64 {
        self.notePeak(self.vramBytes());
        return self.peak_resident.load(.monotonic);
    }

    /// Update the default generation params (from settings). Only touched when
    /// no image is in flight, the worker copies `opts` at spawn.
    pub fn setDefaults(self: *Diffuser, steps: usize, width: usize, height: usize) void {
        if (self.busy.load(.acquire)) return;
        self.opts.steps = steps;
        self.opts.width = width;
        self.opts.height = height;
    }

    /// Update the sampler (from settings). Load-neutral, the sampler is per-render
    /// state, not part of the session, so this needs no rebuild and applies to the
    /// next image. Gated on idle like `setDefaults`: the worker reads `opts` at
    /// dispatch, and switching mid-image would not take effect anyway.
    pub fn setSampler(self: *Diffuser, kind: tp.sampler.Kind) void {
        if (self.busy.load(.acquire)) return;
        self.opts.sampler = kind;
    }

    /// Update the prompt dialect and its weighting form (from settings). Load-neutral
    /// like the sampler: the next image parses its prompt the new way, with no reload.
    pub fn setPromptSyntax(
        self: *Diffuser,
        syntax: pipeline.PromptSyntax,
        emph: pipeline.Emphasis,
        compat: pipeline.Compat,
    ) void {
        self.opts.prompt_syntax = syntax;
        self.opts.emphasis = emph;
        self.opts.compat = compat;
    }

    /// Update the scheduler (from settings). Load-neutral, same as `setSampler`.
    pub fn setScheduler(self: *Diffuser, sched: ?tp.sampler.Scheduler) void {
        if (self.busy.load(.acquire)) return;
        self.opts.scheduler = sched;
    }

    /// Update the directory finished images are saved to (from settings). Null
    /// or empty disables saving. Only touched when idle (the worker reads it at
    /// completion); a live change takes effect for the next generation.
    pub fn setOutputDir(self: *Diffuser, dir: ?[]const u8) void {
        if (self.busy.load(.acquire)) return;
        const want: ?[]const u8 = if (dir) |d| (if (d.len > 0) d else null) else null;
        // No change? (both null, or same string) leave the owned copy alone.
        if (want == null and self.output_dir == null) return;
        if (want != null and self.output_dir != null and
            std.mem.eql(u8, want.?, self.output_dir.?)) return;
        if (self.output_dir) |o| self.gpa.free(o);
        self.output_dir = if (want) |w| self.gpa.dupe(u8, w) catch null else null;
    }

    /// Set the live-preview method. Takes effect on the NEXT sampling step, even
    /// mid-image: the running worker reads `live_preview` each step, so there's no
    /// busy-bail. `taehv` is preloaded up front (whenever a taew is configured) so
    /// a mid-image switch to TAESD is instant.
    pub fn setPreview(self: *Diffuser, method: config.Preview) void {
        self.preview_method = method;
        // config.Preview values match pipeline's method encoding (0=none,
        // 1=latent2rgb, 2=taesd).
        self.live_preview.method.store(@intFromEnum(method), .release);
        self.refreshPreview();
    }

    /// Set the TAESD preview resolution (latent-grid divisor; see
    /// pipeline.Options.preview_ds). Applied live, mid-image on the next step.
    pub fn setPreviewSize(self: *Diffuser, preview_ds: usize) void {
        self.opts.preview_ds = preview_ds;
        self.live_preview.ds.store(@intCast(preview_ds), .release);
    }

    /// Reconcile the STATIC preview fields of `opts` (fallback when no live
    /// control) with `preview_method`. The worker always attaches `live_preview`
    /// and passes `taew_owned` as the taew path, so these are just belt-and-braces.
    fn refreshPreview(self: *Diffuser) void {
        self.opts.preview = self.preview_method != .none;
        self.opts.taew_path = if (self.preview_method == .taesd) self.taew_owned else null;
    }

    /// Update the current model config (new paths/backend/decode) for FUTURE
    /// enqueues. Takes effect immediately as the source `enqueue` snapshots onto
    /// each image, already-queued images keep the config they were stamped with,
    /// and the worker reloads the resident pipeline at the seam where consecutive
    /// images disagree. When the queue is fully idle, the now-stale resident
    /// pipeline is dropped here so a switch returns its VRAM promptly.
    ///
    /// `want`'s paths are borrowed (re-duped into `path_store`); any override may
    /// be "" (that component comes from the primary checkpoint). Takes the whole
    /// `ModelConfig` rather than one parameter per path: at four paths plus two
    /// enums a positional list is one transposition away from loading the VAE as
    /// a text encoder, with no type error to catch it.
    pub fn requestPaths(self: *Diffuser, want: ModelConfig, taew: ?[]const u8) void {
        _ = self.path_store.reset(.retain_capacity);
        const a = self.path_store.allocator();
        const owned: ModelConfig = .{
            .dit_path = a.dupe(u8, want.dit_path) catch return,
            .vae_path = a.dupe(u8, want.vae_path) catch return,
            .text_encoder_path = a.dupe(u8, want.text_encoder_path) catch return,
            .text_encoder_2_path = a.dupe(u8, want.text_encoder_2_path) catch return,
            .backend = want.backend,
            .vae_decode = want.vae_decode,
        };
        owned.applyTo(&self.opts);
        // The config just changed, so any previous load failure describes a model
        // set that no longer exists, clear it rather than warn about the old one.
        self.load_error.store(0, .release);
        self.taew_owned = if (taew) |t| (a.dupe(u8, t) catch null) else null;
        self.refreshPreview(); // taew_path follows the (possibly new) taew_owned
        self.dropStaleSession();
    }

    /// Free the resident pipeline when it's loaded for a config the current
    /// defaults no longer match AND the queue is fully idle (no worker, nothing
    /// pending), so a model/backend switch made while idle returns the old VRAM
    /// at once. A mid-queue switch is NOT dropped here: queued images still carry
    /// (and need) their own snapshot, so the worker reloads lazily at the seam.
    fn dropStaleSession(self: *Diffuser) void {
        if (self.busy.load(.acquire)) return;
        if (self.nextPending() != null) return;
        const l = self.loaded orelse return; // nothing resident
        if (!ModelConfig.eql(l, self.liveConfig())) {
            self.freeSession();
            std.log.info("diffusion model switched", .{});
        }
    }

    /// UI-thread, once per frame: reap a finished diffusion, then start the next
    /// pending one (at most one at a time to bound VRAM).
    pub fn pump(self: *Diffuser) void {
        if (self.thread) |t| {
            if (self.busy.load(.acquire)) return; // still running
            t.join();
            self.thread = null;
        }
        // While paused, don't START (or load for) new work, leave it queued
        // until resumed. A generation already in flight is parked at the step
        // gate (Tier 1) and never reaches here (busy -> returned above); this
        // gates the NEXT image so a paused engine neither loads nor begins one.
        if (self.pause.isPaused(self.io)) return;

        const gi = self.nextPending() orelse {
            // Queue drained. Keep the model resident so a later gen reuses it,
            // unless the config was switched mid-queue and it's now stale, in
            // which case drop it here to return its VRAM. Let the coordinator
            // undo any eviction, once, on the drain EDGE (see `vram_entered`).
            self.dropStaleSession();
            if (self.vram_entered) {
                self.vram_entered = false;
                self.vram.exit(self.vram.ctx);
            }
            return;
        };
        // Seam reconcile: if the resident pipeline is loaded for a different config
        // than THIS image needs (a backend/model switch made after it was enqueued),
        // free it now, on the UI thread, where no status-bar reader can race the
        // free, so the worker reloads for gi's snapshot. Images without a snapshot
        // (unqueued attachments) leave the resident session as-is. (`freeSession`
        // takes `res_mu` itself, so a concurrent foreign yield can't interleave.)
        if (gi.model) |m| {
            if (self.loaded) |l| {
                if (!ModelConfig.eql(l, m)) self.freeSession();
            }
        }
        // Make VRAM room for the image model before it loads (the worker
        // auto-budgets from live free VRAM). Deliberately OUTSIDE `res_mu`: this
        // calls out to the app's VRAM coordinator, which rebalances the arbiter and
        // so re-enters this engine's own participant hooks. (Nothing is lost by a
        // yield landing in this window, we hold no session state across it, and
        // the target the arbiter has for us here is a GROW, which is a no-op.)
        self.vram_entered = true;
        self.vram.enter(self.vram.ctx);
        // Preview buffer, sized to the final image resolution, the upper bound
        // on any preview: a full-latent TAESD preview decodes at the image res
        // (lat*spatial_scale), and latent2rgb is smaller (latent res). Sizing it
        // to a fixed 512² silently dropped the larger TAESD sizes (½, full). The
        // pointer stays put for the whole generation; dims published via atomics.
        //
        // Resuming a suspended image (Tier 3 unload-while-paused): KEEP the last
        // published preview + its dims so the parked frame stays on screen until
        // the resumed worker's next onStep overwrites it, instead of blanking to
        // black. The resolution is unchanged (same GenImage), so the existing
        // buffer is still correctly sized.
        if (gi.resume_snapshot == null or gi.preview == null) {
            gi.preview_w = .init(0);
            gi.preview_h = .init(0);
            if (gi.preview) |old| self.gpa.free(old); // free a prior buffer (re-run of a finished image)
            if (self.gpa.alloc(u8, gi.req_width * gi.req_height * 4)) |pb| {
                @memset(pb, 0);
                gi.preview = pb;
            } else |_| {
                gi.preview = null;
            }
        }
        // Hand this engine's device context to the worker under `res_mu`, so the
        // `busy` flag and the spawn are one atomic step. Without it a foreign yield
        // that had already passed its `busy == false` check would evict weights out
        // from under a sampler that has just started (the flag alone is a TOCTOU,
        // it is set on this thread, not the evicting one).
        self.res_mu.lockUncancelable(self.io);
        defer self.res_mu.unlock(self.io);
        gi.status.store(@intFromEnum(GenStatus.generating), .release);
        self.busy.store(true, .release);
        self.thread = std.Thread.spawn(.{}, worker, .{ self, gi }) catch |err| {
            gi.fail(err);
            self.busy.store(false, .release);
            return;
        };
    }

    /// pipeline `Reclaim.call` thunk (recovers the Diffuser from the ctx).
    fn reclaimThunk(ctx: *anyopaque, needed: u64) u64 {
        const self: *Diffuser = @ptrCast(@alignCast(ctx));
        return self.vram.reclaim(self.vram.ctx, needed);
    }

    /// Write a finished image to `output_dir` as a PNG with AUTOMATIC1111
    /// `parameters` metadata. Best-effort: any failure is logged, not fatal (the
    /// image still shows in the UI). Runs on the worker thread. `rgb` is packed
    /// [h][w][3]; `opts` is the worker's per-image options (paths, params).
    fn saveImage(self: *Diffuser, gi: *GenImage, opts: *const pipeline.Options, rgb: []const u8, w: usize, h: usize) void {
        const dir = self.output_dir orelse return; // saving disabled
        const gpa = self.gpa;

        const params = buildA1111Params(
            gpa,
            gi.prompt,
            gi.req_negative,
            gi.req_steps,
            gi.req_cfg,
            gi.req_seed,
            w,
            h,
            modelStem(opts.dit_path),
            self.loadedFamily(),
            opts.sampler,
            opts.scheduler,
            opts.prompt_syntax,
            opts.emphasis,
            opts.compat,
            opts.compatConfig(),
        ) catch |err| {
            std.log.err("image save (metadata) failed: {t}", .{err});
            return;
        };
        defer gpa.free(params);

        var png: std.ArrayList(u8) = .empty;
        defer png.deinit(gpa);
        tp.image.encodePngRgbText(gpa, &png, rgb, w, h, &.{
            .{ .keyword = "parameters", .text = params },
        }) catch |err| {
            std.log.err("image save (encode) failed: {t}", .{err});
            return;
        };

        // Unique, roughly time-sortable filename: tp_<ns>_<seed>.png.
        const name = std.fmt.allocPrint(gpa, "tp_{d}_{d}.png", .{ gi.start_ns.load(.acquire), gi.req_seed }) catch return;
        defer gpa.free(name);
        const path = std.fs.path.join(gpa, &.{ dir, name }) catch return;

        std.Io.Dir.cwd().createDirPath(self.io, dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => {
                std.log.err("image save (mkdir {s}) failed: {t}", .{ dir, err });
                return;
            },
        };
        std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = png.items }) catch |err| {
            std.log.err("image save (write {s}) failed: {t}", .{ path, err });
            gpa.free(path);
            return;
        };
        std.log.info("saved image to {s}", .{path});
        // Kept, not freed: a saved conversation stores this so a reload can show
        // the render rather than generating it again. Written on the worker
        // thread before the image is published as `.done`, so the UI thread only
        // ever sees it fully set.
        if (gi.saved_path) |old| gpa.free(old);
        gi.saved_path = path;
    }

    fn worker(self: *Diffuser, gi: *GenImage) void {
        // Start from the engine's live config (preview method / defaults / output)
        // then override the model-defining fields from THIS image's snapshot, so a
        // backend/model switch made after `gi` was enqueued doesn't retro-apply to
        // it. Images that never went through the queue (no snapshot) use live opts.
        var opts = self.opts;
        const want: ModelConfig = gi.model orelse self.liveConfig();
        want.applyTo(&opts);
        opts.prompt = gi.prompt;
        opts.negative = gi.req_negative;
        opts.cfg = gi.req_cfg;
        opts.width = gi.req_width;
        opts.height = gi.req_height;
        opts.steps = gi.req_steps;
        opts.seed = gi.req_seed;
        opts.on_step = .{ .ctx = gi, .step = onStep };
        // Live preview: the sampling loop reads method + resolution each step so a
        // change made mid-image shows on the next one. Always pass the configured
        // taew path (not just when TAESD is the current method) so the decoder is
        // preloaded and a live switch to TAESD is instant.
        opts.preview_live = &self.live_preview;
        opts.taew_path = self.taew_owned;
        opts.cancel = &gi.cancel; // UI Cancel button aborts sampling
        opts.pause = &self.pause; // UI Pause button parks sampling between steps
        opts.resume_from = gi.resume_snapshot; // continue a suspended image (Tier 3)
        opts.reclaim = .{ .ctx = self, .call = reclaimThunk }; // free LLM VRAM on VAE OOM
        // Resident-weight budget from the coordinator (chat: limit − LLM
        // resident, floored; studio: 0 = auto / pin all free VRAM).
        opts.vram_budget = self.vram.budget(self.vram.ctx);
        gi.total.store(@intCast(opts.steps), .monotonic);
        gi.start_ns.store(nowNs(self.io), .release);

        // Mirror the CLI's progress notes to std.log (load / encode / per-step /
        // vae-decode fallbacks) so terminal output is available for debugging.
        var log_buf: [4096]u8 = undefined;
        var lw = LogWriter.init(&log_buf);
        const progress = &lw.writer;

        // Load the diffusion model when none is resident and keep it across the
        // queue. A mid-queue backend/model switch is reconciled by `pump` on the UI
        // thread BEFORE this worker spawns (it frees the stale session so `session`
        // is null here), freeing on the worker thread would race the status-bar
        // readers, which sample `session` without gating on `busy`.
        var sess = self.session.load(.acquire);
        if (sess == null) {
            const t_load = nowNs(self.io);
            sess = pipeline.Session.init(self.io, self.gpa, opts, progress) catch |err| {
                std.log.err("diffusion model load failed: {t}", .{err});
                self.load_error.store(@intFromError(err), .release);
                gi.fail(err);
                self.busy.store(false, .release);
                self.wake();
                return;
            };
            self.load_error.store(0, .release);
            self.loaded_family.store(@as(u8, @intFromEnum(sess.?.family())) + 1, .release);
            std.log.info("[vram] diffusion model loaded in {d:.1}s ({t}): {d} MiB resident (budget {d} MiB) · {d} MiB free", .{
                @as(f64, @floatFromInt(nowNs(self.io) - t_load)) / 1e9, sess.?.family(),
                sess.?.deviceUsed() >> 20, opts.vram_budget >> 20, sess.?.freeVram() >> 20,
            });
            self.session.store(sess, .release);
            // Record what's resident (gpa-owned; freed on the next reload / free).
            self.loaded = ModelConfig.dupe(self.gpa, want) catch null;
        }
        // Unload-while-paused: generate writes the in-flight latent + step here
        // and returns error.Paused. The worker stores it on the image (status
        // .suspended) and exits; the UI frees the weights, and the next dispatch
        // resumes bit-identically via opts.resume_from.
        var suspend_snap: ?pipeline.Snapshot = null;
        opts.suspend_out = &suspend_snap;
        var img = sess.?.generate(opts, progress) catch |err| {
            if (err == error.Paused) {
                if (gi.resume_snapshot) |*old| old.deinit(self.gpa); // replace any prior
                gi.resume_snapshot = suspend_snap;
                gi.status.store(@intFromEnum(GenStatus.suspended), .release);
                self.busy.store(false, .release);
                self.wake();
                return;
            }
            if (err == error.Canceled) {
                gi.status.store(@intFromEnum(GenStatus.canceled), .release);
            } else {
                std.log.err("image generation failed: {t}", .{err});
                gi.fail(err);
            }
            self.busy.store(false, .release);
            self.wake();
            return;
        };
        defer img.deinit(self.gpa);
        // Completed (resume, if any, is consumed): drop the saved latent.
        if (gi.resume_snapshot) |*s| {
            s.deinit(self.gpa);
            gi.resume_snapshot = null;
        }

        // Persist the finished image (packed RGB) with a1111 metadata before the
        // RGBA conversion. Best-effort, a save failure never fails the gen.
        self.saveImage(gi, &opts, img.rgb, img.width, img.height);

        // The pipeline returns packed RGB; dvui wants RGBA. Convert once.
        const px = img.width * img.height;
        const rgba = self.gpa.alloc(u8, px * 4) catch |err| {
            gi.fail(err);
            self.busy.store(false, .release);
            self.wake();
            return;
        };
        for (0..px) |i| {
            rgba[i * 4 + 0] = img.rgb[i * 3 + 0];
            rgba[i * 4 + 1] = img.rgb[i * 3 + 1];
            rgba[i * 4 + 2] = img.rgb[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        gi.done_ns.store(nowNs(self.io), .release);
        gi.rgba = rgba;
        gi.width = img.width;
        gi.height = img.height;
        gi.status.store(@intFromEnum(GenStatus.done), .release);
        self.busy.store(false, .release);
        self.wake();
    }

    fn onStep(ctx: *anyopaque, done: usize, total: usize, preview: ?pipeline.Preview) void {
        const gi: *GenImage = @ptrCast(@alignCast(ctx));
        // Timestamp the sampling loop: first callback (after step 1) anchors the
        // s/step base; every callback updates the tail. Excludes model-load time.
        const now = nowNs(gi.io);
        if (gi.first_step_ns.load(.monotonic) == 0) gi.first_step_ns.store(now, .release);
        gi.last_step_ns.store(now, .release);
        gi.step.store(@intCast(done), .monotonic);
        gi.total.store(@intCast(total), .monotonic);
        if (preview) |pv| {
            if (gi.preview) |dst| {
                const np = pv.width * pv.height;
                if (np * 4 <= dst.len) {
                    for (0..np) |i| {
                        dst[i * 4 + 0] = pv.rgb[i * 3 + 0];
                        dst[i * 4 + 1] = pv.rgb[i * 3 + 1];
                        dst[i * 4 + 2] = pv.rgb[i * 3 + 2];
                        dst[i * 4 + 3] = 255;
                    }
                    gi.preview_w.store(@intCast(pv.width), .release);
                    gi.preview_h.store(@intCast(pv.height), .release);
                }
            }
        }
        gi.wake();
    }
};

test "applyTo: an unset override is empty and NOT explicit" {
    // The bundled-checkpoint case: one path, nothing else. The empty side paths
    // must overwrite `Options`' defaulted krea2 files, or the resolver would open
    // one of those and hunt for the wrong architecture's tensors in it.
    var opts: pipeline.Options = .{ .prompt = "" };
    (ModelConfig{
        .dit_path = "/sd15.safetensors",
        .vae_path = "",
        .text_encoder_path = "",
        .text_encoder_2_path = "",
        .backend = .cpu,
        .vae_decode = .auto,
    }).applyTo(&opts);

    try std.testing.expectEqualStrings("/sd15.safetensors", opts.dit_path);
    try std.testing.expectEqualStrings("", opts.vae_path);
    try std.testing.expectEqualStrings("", opts.text_encoder_path);
    try std.testing.expectEqualStrings("", opts.text_encoder_2_path);
    try std.testing.expect(!opts.explicit_vae);
    try std.testing.expect(!opts.explicit_text_encoder);
    try std.testing.expect(!opts.explicit_text_encoder_2);
    try std.testing.expectEqual(pipeline.Backend.cpu, opts.backend);
}

test "applyTo: a set override is explicit (it outranks a bundled copy)" {
    var opts: pipeline.Options = .{ .prompt = "" };
    (ModelConfig{
        .dit_path = "/dit.safetensors",
        .vae_path = "/vae.safetensors",
        .text_encoder_path = "",
        .text_encoder_2_path = "",
        .backend = .zig_cuda,
        .vae_decode = .cpu_tiled,
    }).applyTo(&opts);

    // Only the VAE was supplied: it wins over anything in the checkpoint, while
    // the conditioner still resolves out of the checkpoint itself.
    try std.testing.expect(opts.explicit_vae);
    try std.testing.expect(!opts.explicit_text_encoder);
    try std.testing.expectEqualStrings("/vae.safetensors", opts.vae_path);
    try std.testing.expectEqual(pipeline.VaeDecode.cpu_tiled, opts.vae_decode);
}

test "loadError/loadedFamily round-trip through their atomic encodings" {
    var d: Diffuser = tp.init_defaults.of(Diffuser);
    d.load_error = .init(0);
    d.loaded_family = .init(0);
    try std.testing.expectEqual(@as(?anyerror, null), d.loadError());
    try std.testing.expectEqual(@as(?pipeline.Family, null), d.loadedFamily());

    d.load_error.store(@intFromError(error.UnsupportedBackend), .release);
    try std.testing.expectEqual(@as(?anyerror, error.UnsupportedBackend), d.loadError());

    // 0 is reserved for "nothing loaded", so the tag is stored offset by one,
    // without that, family 0 (krea2) would read back as "no model". Every family
    // must survive the round trip, including any added later.
    inline for (@typeInfo(pipeline.Family).@"enum".fields) |f| {
        const fam: pipeline.Family = @enumFromInt(f.value);
        d.loaded_family.store(@as(u8, @intFromEnum(fam)) + 1, .release);
        try std.testing.expectEqual(@as(?pipeline.Family, fam), d.loadedFamily());
    }
}

// The failure reason has to survive the trip from the worker thread to the tile
// that reports it, and `fail` has to leave the two atomics consistent, a UI
// thread that sees `.failed` must never read a zero reason beside it.
test "a failure records its cause, and only a failure does" {
    const nop = struct {
        fn f() void {}
    }.f;
    var gi: GenImage = .{ .prompt = "", .wake = nop, .io = std.testing.io };
    try std.testing.expectEqual(@as(?anyerror, null), gi.failure());
    gi.fail(error.DeviceOutOfMemory);
    try std.testing.expectEqual(GenStatus.failed, gi.get());
    try std.testing.expectEqual(@as(?anyerror, error.DeviceOutOfMemory), gi.failure());
}

// VRAM exhaustion arrives under four different names (the CUDA libraries
// report an out-of-workspace as their own error, and the hand-PTX path surfaces
// a post-OOM fault as `CudaError`), so the one failure a user can actually act
// on must not fall through to a bare error name. Anything unrecognized still
// says *something* specific rather than "failed".
test "failureText names the VRAM family, and falls back to the error name" {
    for ([_]anyerror{ error.DeviceOutOfMemory, error.CudaError, error.CublasLtError, error.CudnnError }) |e| {
        errdefer std.debug.print("error {t}\n", .{e});
        try std.testing.expectEqualStrings("out of VRAM", failureText(e));
    }
    try std.testing.expectEqualStrings("out of memory", failureText(error.OutOfMemory));
    try std.testing.expectEqualStrings("SomethingNovel", failureText(error.SomethingNovel));
}

test "ModelConfig.dupe owns all four paths and eql sees each of them" {
    const gpa = std.testing.allocator;
    const src: ModelConfig = .{
        .dit_path = "/sdxl.safetensors",
        .vae_path = "/vae.safetensors",
        .text_encoder_path = "/clip_l.safetensors",
        .text_encoder_2_path = "/clip_g.safetensors",
        .backend = .cuda,
        .vae_decode = .auto,
    };
    const copy = try ModelConfig.dupe(gpa, src);
    defer copy.deinit(gpa);

    // Equal by value, distinct storage (the source may be a caller's arena that
    // gets reset under a queued image).
    try std.testing.expect(ModelConfig.eql(src, copy));
    try std.testing.expect(src.text_encoder_2_path.ptr != copy.text_encoder_2_path.ptr);

    // Each path participates in the reload decision. The second tower especially:
    // if `eql` ignored it, switching CLIP-G would silently keep the old pipeline.
    var other = src;
    other.text_encoder_2_path = "/other_g.safetensors";
    try std.testing.expect(!ModelConfig.eql(src, other));
}

test "clampDim rounds to multiple of 16 within bounds" {
    try std.testing.expectEqual(@as(usize, 1024), clampDim(1024));
    try std.testing.expectEqual(@as(usize, 1024), clampDim(1030)); // 1030/16*16
    try std.testing.expectEqual(@as(usize, 256), clampDim(10)); // floor
    try std.testing.expectEqual(@as(usize, 4096), clampDim(99999)); // ceil
    try std.testing.expectEqual(@as(usize, 512), clampDim(519));
}

test "parseGenAttrs overrides only provided fields" {
    var gi: GenImage = .{ .prompt = "", .wake = undefined, .io = undefined };
    parseGenAttrs("width=1536 height=1024 steps=12 seed=42", &gi);
    try std.testing.expectEqual(@as(usize, 1536), gi.req_width);
    try std.testing.expectEqual(@as(usize, 1024), gi.req_height);
    try std.testing.expectEqual(@as(usize, 12), gi.req_steps);
    try std.testing.expectEqual(@as(u64, 42), gi.req_seed);

    // Unset fields keep their defaults; bad values are ignored; dims round.
    var gi2: GenImage = .{ .prompt = "", .wake = undefined, .io = undefined };
    parseGenAttrs("width=1000 steps=oops", &gi2);
    try std.testing.expectEqual(@as(usize, 992), gi2.req_width); // 1000/16*16
    try std.testing.expectEqual(@as(usize, 1024), gi2.req_height); // default
    try std.testing.expectEqual(@as(usize, 20), gi2.req_steps); // default (bad ignored)
}

test "modelStem strips directory and extension" {
    try std.testing.expectEqualStrings("krea2", modelStem("/models/diffusion/krea2.safetensors"));
    try std.testing.expectEqualStrings("krea2", modelStem("krea2.safetensors"));
    try std.testing.expectEqualStrings("model", modelStem("/a/b/model")); // no extension
    try std.testing.expectEqualStrings("v1.0", modelStem("/m/v1.0.ckpt")); // last dot only
}

test "buildA1111Params formats prompt, settings, and optional negative" {
    const gpa = std.testing.allocator;

    const with_neg = try buildA1111Params(gpa, "a cat", "blurry", 20, 3.5, 42, 1024, 768, "krea2", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(with_neg);
    try std.testing.expectEqualStrings(
        "a cat\n" ++
            "Negative prompt: blurry\n" ++
            "Steps: 20, Sampler: Euler, Schedule type: Simple, CFG scale: 3.5, Seed: 42, Size: 1024x768, Model: krea2, Prompt syntax: ComfyUI",
        with_neg,
    );

    // No negative -> the "Negative prompt:" line is omitted entirely.
    const no_neg = try buildA1111Params(gpa, "a dog", "", 8, 1.0, 7, 512, 512, "m", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(no_neg);
    try std.testing.expectEqualStrings(
        "a dog\n" ++
            "Steps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 1.0, Seed: 7, Size: 512x512, Model: m, Prompt syntax: ComfyUI",
        no_neg,
    );
}

test "buildA1111Params records the sampler actually used" {
    // A saved PNG is the record a user (or ComfyUI's metadata importer) re-renders
    // from, so a hardcoded sampler name is a wrong answer nothing else would catch,
    // and the a1111 spelling is not the CLI's.
    const gpa = std.testing.allocator;
    for ([_]struct { k: tp.sampler.Kind, want: []const u8 }{
        .{ .k = .euler, .want = "Sampler: Euler," },
        .{ .k = .dpmpp_2m_sde, .want = "Sampler: DPM++ 2M SDE," },
        .{ .k = .dpmpp_2m_sde_heun, .want = "Sampler: DPM++ 2M SDE Heun," },
    }) |c| {
        const s = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, c.k, null, .comfy, .original, .comfy, .{});
        defer gpa.free(s);
        errdefer std.debug.print("{s}\n", .{s});
        try std.testing.expect(std.mem.indexOf(u8, s, c.want) != null);
    }
}

test "buildA1111Params records the sampling compat, and overrides only when overridden" {
    const gpa = std.testing.allocator;

    // An ordinary ComfyUI render's block is byte-for-byte what it was before compat
    // existed, no new fields, so nothing that parses these PNGs has to change.
    const plain = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, .of(.comfy));
    defer gpa.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "Compat") == null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "RNG") == null);

    // A1111's defaults are named once, not spelled out three times.
    const a = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .a1111, .of(.a1111));
    defer gpa.free(a);
    errdefer std.debug.print("{s}\n", .{a});
    try std.testing.expect(std.mem.indexOf(u8, a, "Compat: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "RNG") == null);

    // An override has to be recorded or the block does not describe the render: with
    // `RNG: CPU` this is a different starting latent from the line above, at the same
    // seed. Same reasoning that stopped `Sampler` being hardcoded.
    var cc: pipeline.CompatConfig = .of(.a1111);
    cc.noise_src = .torch_cpu;
    const ov = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .a1111, cc);
    defer gpa.free(ov);
    errdefer std.debug.print("{s}\n", .{ov});
    try std.testing.expect(std.mem.indexOf(u8, ov, "Compat: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, ov, "RNG: CPU") != null);
    // And the two knobs still at A1111's defaults stay out of it.
    try std.testing.expect(std.mem.indexOf(u8, ov, "SGM") == null);

    // The reverse: ComfyUI conventions with A1111's noise, which is the single-variable
    // experiment someone chasing a mismatch would actually run.
    var cn: pipeline.CompatConfig = .of(.comfy);
    cn.noise_src = .nv_philox;
    const nv = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, cn);
    defer gpa.free(nv);
    try std.testing.expect(std.mem.indexOf(u8, nv, "RNG: NV") != null);
    try std.testing.expect(std.mem.indexOf(u8, nv, "Compat") == null);
}

test "buildA1111Params records the prompt dialect, and the emphasis only when it applies" {
    // The same prompt text renders a DIFFERENT image in the two dialects, so a block that
    // does not say which one was used cannot be re-rendered from. `Emphasis` appears only
    // under a1111, where it is a real choice.
    const gpa = std.testing.allocator;
    const a = try buildA1111Params(gpa, "(a:1.2) [b]", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .a1111, .no_norm, .comfy, .{});
    defer gpa.free(a);
    try std.testing.expect(std.mem.indexOf(u8, a, "Prompt syntax: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "Emphasis: No norm") != null);

    const c = try buildA1111Params(gpa, "(a:1.2)", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(c);
    try std.testing.expect(std.mem.indexOf(u8, c, "Prompt syntax: ComfyUI") != null);
    try std.testing.expect(std.mem.indexOf(u8, c, "Emphasis") == null);
}

test "buildA1111Params names the family's own schedule, and omits it when unknown" {
    const gpa = std.testing.allocator;

    // The SD family samples the discrete beta ladder linearly, A1111's "Normal",
    // not krea2's flow-matching "Simple". A reader re-renders from this field.
    for ([_]pipeline.Family{ .sd15, .sdxl }) |f| {
        const s = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", f, .euler, null, .comfy, .original, .comfy, .{});
        defer gpa.free(s);
        try std.testing.expect(std.mem.indexOf(u8, s, "Schedule type: Normal,") != null);
    }
    const k = try buildA1111Params(gpa, "p", "", 8, 1.0, 1, 1024, 1024, "m", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(k);
    try std.testing.expect(std.mem.indexOf(u8, k, "Schedule type: Simple,") != null);

    // An EXPLICIT scheduler wins over the family default, and this is what the
    // field is for: with schedulers selectable, deriving it from the architecture
    // stamps a name the image was not rendered with.
    const karras = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sd15, .euler, .karras, .comfy, .original, .comfy, .{});
    defer gpa.free(karras);
    try std.testing.expect(std.mem.indexOf(u8, karras, "Schedule type: Karras,") != null);
    const kl = try buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .krea2, .euler, .kl_optimal, .comfy, .original, .comfy, .{});
    defer gpa.free(kl);
    try std.testing.expect(std.mem.indexOf(u8, kl, "Schedule type: KL Optimal,") != null);

    // Unknown architecture: drop the field rather than stamp a guess a reader
    // would reproduce with. Everything around it stays well-formed.
    const unknown = try buildA1111Params(gpa, "p", "", 8, 1.0, 1, 512, 512, "m", null, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(unknown);
    try std.testing.expect(std.mem.indexOf(u8, unknown, "Schedule type") == null);
    try std.testing.expectEqualStrings(
        "p\nSteps: 8, Sampler: Euler, CFG scale: 1.0, Seed: 1, Size: 512x512, Model: m, Prompt syntax: ComfyUI",
        unknown,
    );
}

test "nextSeed advances deterministically and distinctly" {
    var d: Diffuser = tp.init_defaults.of(Diffuser);
    d.seed = 0;
    const a = d.nextSeed();
    const b = d.nextSeed();
    try std.testing.expect(a != 0);
    try std.testing.expect(a != b);
}

test "an a1111 parameters block round-trips through its own parser" {
    const gpa = std.testing.allocator;
    const params = try buildA1111Params(
        gpa,
        "a lighthouse in heavy fog\nsecond line of prompt",
        "blurry, low quality",
        34,
        4.2,
        8812,
        1216,
        832,
        "krea2",
        null,
        .euler,
        null,
        .comfy,
        .original,
        .comfy,
        .of(.comfy),
    );
    defer gpa.free(params);

    const got = parseA1111Params(params);
    try std.testing.expectEqualStrings("a lighthouse in heavy fog\nsecond line of prompt", got.prompt);
    try std.testing.expectEqualStrings("blurry, low quality", got.negative);
    try std.testing.expectEqual(@as(?usize, 34), got.steps);
    try std.testing.expectApproxEqAbs(@as(f32, 4.2), got.cfg.?, 0.001);
    try std.testing.expectEqual(@as(?u64, 8812), got.seed);
    try std.testing.expectEqual(@as(?usize, 1216), got.width);
    try std.testing.expectEqual(@as(?usize, 832), got.height);
}

test "parsing tolerates blocks we did not write" {
    // No negative, extra unknown keys, different order: all of these come off
    // images made by ComfyUI or A1111 itself.
    const p = parseA1111Params(
        "just a prompt\nSteps: 20, Sampler: DPM++ 2M, Schedule type: Karras, " ++
            "CFG scale: 7, Seed: 1234, Size: 512x768, Model hash: abc123, Model: sd15",
    );
    try std.testing.expectEqualStrings("just a prompt", p.prompt);
    try std.testing.expectEqualStrings("", p.negative);
    try std.testing.expectEqual(@as(?usize, 20), p.steps);
    try std.testing.expectEqual(@as(?u64, 1234), p.seed);
    try std.testing.expectEqual(@as(?usize, 512), p.width);
    try std.testing.expectEqual(@as(?usize, 768), p.height);

    // A prompt that itself mentions "Steps:" must not be mistaken for the
    // settings line; the LAST one wins.
    const q = parseA1111Params("Steps: how many steps?\nSteps: 12, Seed: 9");
    try std.testing.expectEqualStrings("Steps: how many steps?", q.prompt);
    try std.testing.expectEqual(@as(?usize, 12), q.steps);

    // Nothing at all, and a block with no settings line, must not fault.
    try std.testing.expectEqual(@as(?usize, null), parseA1111Params("").steps);
    try std.testing.expectEqualStrings("only a prompt", parseA1111Params("only a prompt").prompt);
}
