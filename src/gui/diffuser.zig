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
//!  - `Source` — where the next pending `GenImage` comes from (chat scans its
//!    message transcript; the studio scans its gallery list). The engine never
//!    owns the images.
//!  - `VramCoordinator` — how to make room for the image model and how much
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

pub const GenStatus = enum(u8) { pending, generating, done, failed, canceled };

/// The model-defining config for one generation — exactly the fields that decide
/// which `pipeline.Session` an image needs (paths + backend + VAE decode path).
/// Captured onto each `GenImage` at `enqueue` so a mid-queue backend/model switch
/// only affects images enqueued AFTER it: already-queued images finish on the
/// config they were created with, and the worker reloads the pipeline at the seam
/// where consecutive images disagree. Path strings are gpa-owned dupes (freed in
/// `freeGenImage`); preview/steps/dims stay live (cosmetic / already per-image).
pub const ModelConfig = struct {
    dit_path: []const u8,
    vae_path: []const u8,
    text_encoder_path: []const u8,
    backend: pipeline.Backend,
    vae_decode: pipeline.VaeDecode,

    /// Duplicate the path strings into gpa-owned storage.
    fn dupe(
        gpa: std.mem.Allocator,
        dit: []const u8,
        vae: []const u8,
        te: []const u8,
        backend: pipeline.Backend,
        vae_decode: pipeline.VaeDecode,
    ) !ModelConfig {
        const d = try gpa.dupe(u8, dit);
        errdefer gpa.free(d);
        const v = try gpa.dupe(u8, vae);
        errdefer gpa.free(v);
        const t = try gpa.dupe(u8, te);
        return .{ .dit_path = d, .vae_path = v, .text_encoder_path = t, .backend = backend, .vae_decode = vae_decode };
    }

    fn deinit(self: ModelConfig, gpa: std.mem.Allocator) void {
        gpa.free(self.dit_path);
        gpa.free(self.vae_path);
        gpa.free(self.text_encoder_path);
    }

    /// Do these two configs need the same resident pipeline? (Paths + backend +
    /// decode path — the fields `pipeline.Session.init` keys the session on.)
    fn eql(a: ModelConfig, b: ModelConfig) bool {
        return a.backend == b.backend and a.vae_decode == b.vae_decode and
            std.mem.eql(u8, a.dit_path, b.dit_path) and
            std.mem.eql(u8, a.vae_path, b.vae_path) and
            std.mem.eql(u8, a.text_encoder_path, b.text_encoder_path);
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
    // starts and overwritten in place each step — the pointer is stable for the
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
    // them at these defaults — matching the pre-studio behavior).
    req_negative: []u8 = "", // owned (gpa) when non-empty; "" = none
    req_cfg: f32 = 1.0,
    /// Model/backend snapshot, stamped by `enqueue` (gpa-owned paths). Null for
    /// images that never went through the queue (e.g. user-attached inputs, tests);
    /// the worker falls back to the engine's live config for those.
    model: ?ModelConfig = null,

    pub fn get(self: *const GenImage) GenStatus {
        return @enumFromInt(self.status.load(.acquire));
    }
};

pub fn freeGenImage(gpa: std.mem.Allocator, gi: *GenImage) void {
    gpa.free(gi.prompt);
    if (gi.req_negative.len > 0) gpa.free(gi.req_negative);
    if (gi.rgba) |r| gpa.free(r);
    if (gi.preview) |p| gpa.free(p);
    if (gi.model) |m| m.deinit(gpa);
    gpa.destroy(gi);
}

/// Diffusion configuration for a session (krea2 by default).
pub const DiffConfig = struct {
    dit_path: []const u8,
    vae_path: []const u8,
    text_encoder_path: []const u8,
    steps: usize = 20,
    width: usize = 1024,
    height: usize = 1024,
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

/// Build an AUTOMATIC1111-style `parameters` metadata string. Format:
///
///     <prompt>
///     Negative prompt: <negative>            (omitted when empty)
///     Steps: N, Sampler: Euler, Schedule type: Simple, CFG scale: C, Seed: S, Size: WxH, Model: <name>
///
/// `model_name` is the diffusion checkpoint's file stem. Caller frees.
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
) ![]u8 {
    const neg_line = if (negative.len > 0)
        try std.fmt.allocPrint(gpa, "Negative prompt: {s}\n", .{negative})
    else
        try gpa.dupe(u8, "");
    defer gpa.free(neg_line);
    return std.fmt.allocPrint(
        gpa,
        "{s}\n{s}Steps: {d}, Sampler: Euler, Schedule type: Simple, CFG scale: {d:.1}, Seed: {d}, Size: {d}x{d}, Model: {s}",
        .{ prompt, neg_line, steps, cfg, seed, width, height, model_name },
    );
}

/// The file stem of a path (basename minus final extension) — the a1111 "Model".
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
    /// or quality change shows on the next step — even mid-image. Address is stable
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
        return .{
            .gpa = gpa,
            .io = io,
            .wake = wake,
            .opts = .{
                .prompt = "",
                .width = cfg.width,
                .height = cfg.height,
                .steps = cfg.steps,
                .backend = cfg.backend,
                .vae_decode = cfg.vae_decode,
                .vram_budget = cfg.vram_budget,
                .dit_path = cfg.dit_path,
                .vae_path = cfg.vae_path,
                .text_encoder_path = cfg.text_encoder_path,
                .preview = cfg.preview_enabled,
                .taew_path = cfg.taew_path,
                .preview_ds = cfg.preview_ds,
            },
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
    /// this one — already-queued images finish on the config they were created
    /// with (the worker reloads the pipeline at the seam where they disagree).
    pub fn enqueue(self: *Diffuser, gi: *GenImage) !void {
        if (gi.model == null) gi.model = try ModelConfig.dupe(
            self.gpa,
            self.opts.dit_path,
            self.opts.vae_path,
            self.opts.text_encoder_path,
            self.opts.backend,
            self.opts.vae_decode,
        );
        try self.queue.append(self.gpa, gi);
    }

    /// The unified image list (creation order) for rendering / viewer nav.
    pub fn items(self: *const Diffuser) []*GenImage {
        return self.queue.items;
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
        // cancel; wake it (without unpausing — the pause button stays in sync)
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
    /// cancel flag — used by the single-image Cancel button while paused (no-op
    /// when nothing is parked, and never changes the pause state).
    pub fn wakePaused(self: *Diffuser) void {
        self.pause.wake(self.io);
    }

    /// Next pending image to run, dropping ones canceled before they start.
    fn nextPending(self: *Diffuser) ?*GenImage {
        for (self.queue.items) |gi| {
            if (gi.get() != .pending) continue;
            if (gi.cancel.load(.acquire)) {
                gi.status.store(@intFromEnum(GenStatus.canceled), .release);
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

    /// Join the worker if one is running (no cancel — an in-flight image is
    /// allowed to FINISH). Used before a session teardown / model swap.
    pub fn stopAndReap(self: *Diffuser) void {
        if (self.thread) |t| {
            t.join();
            self.thread = null;
            self.busy.store(false, .release);
        }
    }

    /// Free the resident pipeline (returns its VRAM). Binds its own CUDA
    /// context. Caller must ensure no worker is in flight.
    pub fn freeSession(self: *Diffuser) void {
        if (self.session.load(.acquire)) |s| {
            s.deinit();
            self.session.store(null, .release);
        }
        if (self.loaded) |l| {
            l.deinit(self.gpa);
            self.loaded = null;
        }
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
        return if (self.session.load(.acquire)) |s| s.deviceUsed() else 0;
    }

    /// MEASURED per-component VRAM breakdown (TE / DiT / VAE / latent) for the
    /// status-bar meter; all zero when no pipeline is resident.
    pub fn vramBreakdown(self: *Diffuser) VramBreakdown {
        return if (self.session.load(.acquire)) |s| s.vramBreakdown() else .{};
    }

    /// Free resident diffusion weights to fit `budget` bytes — the GUI VRAM
    /// limit lowered while the queue is idle (soft residency: never mid-image,
    /// which would force per-step streaming). No-op while generating; the next
    /// image re-uploads what fits its budget.
    pub fn trimToBudget(self: *Diffuser, budget: u64) void {
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
    pub fn giveUpToBudget(self: *Diffuser, budget: u64) u64 {
        if (self.busy.load(.acquire)) return 0;
        const s = self.session.load(.acquire) orelse return 0;
        const before = s.deviceUsed();
        const freed = s.giveUpToBudget(budget);
        if (freed > 0) std.log.info("[vram] diffusion yield (target {d} MiB): freed {d} MiB ({d}→{d} MiB resident) · {d} MiB free", .{
            budget >> 20, freed >> 20, before >> 20, s.deviceUsed() >> 20, s.freeVram() >> 20,
        });
        return freed;
    }

    /// Estimate the image model's resident footprint (bytes) from its file
    /// sizes — the target VRAM to free for it under image priority. 0 if unknown.
    pub fn estimateResidentBytes(self: *Diffuser) u64 {
        var total: u64 = 0;
        for ([_][]const u8{ self.opts.dit_path, self.opts.vae_path, self.opts.text_encoder_path }) |p| {
            const st = std.Io.Dir.cwd().statFile(self.io, p, .{}) catch continue;
            total += st.size;
        }
        return total;
    }

    /// Update the default generation params (from settings). Only touched when
    /// no image is in flight — the worker copies `opts` at spawn.
    pub fn setDefaults(self: *Diffuser, steps: usize, width: usize, height: usize) void {
        if (self.busy.load(.acquire)) return;
        self.opts.steps = steps;
        self.opts.width = width;
        self.opts.height = height;
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
    /// pipeline.Options.preview_ds). Applied live — mid-image on the next step.
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
    /// each image — already-queued images keep the config they were stamped with,
    /// and the worker reloads the resident pipeline at the seam where consecutive
    /// images disagree. When the queue is fully idle, the now-stale resident
    /// pipeline is dropped here so a switch returns its VRAM promptly. Path args
    /// are borrowed (re-duped into `path_store`).
    pub fn requestPaths(
        self: *Diffuser,
        dit: []const u8,
        vae: []const u8,
        te: []const u8,
        taew: ?[]const u8,
        backend: pipeline.Backend,
        vae_decode: pipeline.VaeDecode,
    ) void {
        _ = self.path_store.reset(.retain_capacity);
        const a = self.path_store.allocator();
        self.opts.dit_path = a.dupe(u8, dit) catch return;
        self.opts.vae_path = a.dupe(u8, vae) catch return;
        self.opts.text_encoder_path = a.dupe(u8, te) catch return;
        self.opts.backend = backend;
        self.opts.vae_decode = vae_decode;
        self.taew_owned = if (taew) |t| (a.dupe(u8, t) catch null) else null;
        self.refreshPreview(); // taew_path follows the (possibly new) taew_owned
        self.dropStaleSession();
    }

    /// Free the resident pipeline when it's loaded for a config the current
    /// defaults no longer match AND the queue is fully idle (no worker, nothing
    /// pending) — so a model/backend switch made while idle returns the old VRAM
    /// at once. A mid-queue switch is NOT dropped here: queued images still carry
    /// (and need) their own snapshot, so the worker reloads lazily at the seam.
    fn dropStaleSession(self: *Diffuser) void {
        if (self.busy.load(.acquire)) return;
        if (self.nextPending() != null) return;
        const l = self.loaded orelse return; // nothing resident
        const cur: ModelConfig = .{
            .dit_path = self.opts.dit_path,
            .vae_path = self.opts.vae_path,
            .text_encoder_path = self.opts.text_encoder_path,
            .backend = self.opts.backend,
            .vae_decode = self.opts.vae_decode,
        };
        if (!ModelConfig.eql(l, cur)) {
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
        // While paused, don't START (or load for) new work — leave it queued
        // until resumed. A generation already in flight is parked at the step
        // gate (Tier 1) and never reaches here (busy → returned above); this
        // gates the NEXT image so a paused engine neither loads nor begins one.
        if (self.pause.isPaused(self.io)) return;

        const gi = self.nextPending() orelse {
            // Queue drained. Keep the model resident so a later gen reuses it —
            // unless the config was switched mid-queue and it's now stale, in
            // which case drop it here to return its VRAM. Let the coordinator
            // undo any eviction — once, on the drain EDGE (see `vram_entered`).
            self.dropStaleSession();
            if (self.vram_entered) {
                self.vram_entered = false;
                self.vram.exit(self.vram.ctx);
            }
            return;
        };
        // Seam reconcile: if the resident pipeline is loaded for a different config
        // than THIS image needs (a backend/model switch made after it was enqueued),
        // free it now — on the UI thread, where no status-bar reader can race the
        // free — so the worker reloads for gi's snapshot. Images without a snapshot
        // (unqueued attachments) leave the resident session as-is.
        if (gi.model) |m| {
            if (self.loaded) |l| {
                if (!ModelConfig.eql(l, m)) self.freeSession();
            }
        }
        // Make VRAM room for the image model before it loads (the worker
        // auto-budgets from live free VRAM).
        self.vram_entered = true;
        self.vram.enter(self.vram.ctx);
        // Preview buffer, sized to the final image resolution — the upper bound
        // on any preview: a full-latent TAESD preview decodes at the image res
        // (lat·spatial_scale), and latent2rgb is smaller (latent res). Sizing it
        // to a fixed 512² silently dropped the larger TAESD sizes (½, full). The
        // pointer stays put for the whole generation; dims published via atomics.
        gi.preview_w = .init(0);
        gi.preview_h = .init(0);
        if (self.gpa.alloc(u8, gi.req_width * gi.req_height * 4)) |pb| {
            @memset(pb, 0);
            gi.preview = pb;
        } else |_| {
            gi.preview = null;
        }
        gi.status.store(@intFromEnum(GenStatus.generating), .release);
        self.busy.store(true, .release);
        self.thread = std.Thread.spawn(.{}, worker, .{ self, gi }) catch {
            gi.status.store(@intFromEnum(GenStatus.failed), .release);
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
        defer gpa.free(path);

        std.Io.Dir.cwd().createDirPath(self.io, dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => {
                std.log.err("image save (mkdir {s}) failed: {t}", .{ dir, err });
                return;
            },
        };
        std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = png.items }) catch |err| {
            std.log.err("image save (write {s}) failed: {t}", .{ path, err });
            return;
        };
        std.log.info("saved image to {s}", .{path});
    }

    fn worker(self: *Diffuser, gi: *GenImage) void {
        // Start from the engine's live config (preview method / defaults / output)
        // then override the model-defining fields from THIS image's snapshot — so a
        // backend/model switch made after `gi` was enqueued doesn't retro-apply to
        // it. Images that never went through the queue (no snapshot) use live opts.
        var opts = self.opts;
        const want: ModelConfig = gi.model orelse .{
            .dit_path = self.opts.dit_path,
            .vae_path = self.opts.vae_path,
            .text_encoder_path = self.opts.text_encoder_path,
            .backend = self.opts.backend,
            .vae_decode = self.opts.vae_decode,
        };
        opts.dit_path = want.dit_path;
        opts.vae_path = want.vae_path;
        opts.text_encoder_path = want.text_encoder_path;
        opts.backend = want.backend;
        opts.vae_decode = want.vae_decode;
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
        // is null here) — freeing on the worker thread would race the status-bar
        // readers, which sample `session` without gating on `busy`.
        var sess = self.session.load(.acquire);
        if (sess == null) {
            const t_load = nowNs(self.io);
            sess = pipeline.Session.init(self.io, self.gpa, opts, progress) catch |err| {
                std.log.err("diffusion model load failed: {t}", .{err});
                gi.status.store(@intFromEnum(GenStatus.failed), .release);
                self.busy.store(false, .release);
                self.wake();
                return;
            };
            std.log.info("[vram] diffusion model loaded in {d:.1}s: {d} MiB resident (budget {d} MiB) · {d} MiB free", .{
                @as(f64, @floatFromInt(nowNs(self.io) - t_load)) / 1e9,
                sess.?.deviceUsed() >> 20, opts.vram_budget >> 20, sess.?.freeVram() >> 20,
            });
            self.session.store(sess, .release);
            // Record what's resident (gpa-owned; freed on the next reload / free).
            self.loaded = ModelConfig.dupe(
                self.gpa,
                want.dit_path,
                want.vae_path,
                want.text_encoder_path,
                want.backend,
                want.vae_decode,
            ) catch null;
        }
        var img = sess.?.generate(opts, progress) catch |err| {
            const st: GenStatus = if (err == error.Canceled) .canceled else blk: {
                std.log.err("image generation failed: {t}", .{err});
                break :blk .failed;
            };
            gi.status.store(@intFromEnum(st), .release);
            self.busy.store(false, .release);
            self.wake();
            return;
        };
        defer img.deinit(self.gpa);

        // Persist the finished image (packed RGB) with a1111 metadata before the
        // RGBA conversion. Best-effort — a save failure never fails the gen.
        self.saveImage(gi, &opts, img.rgb, img.width, img.height);

        // The pipeline returns packed RGB; dvui wants RGBA. Convert once.
        const px = img.width * img.height;
        const rgba = self.gpa.alloc(u8, px * 4) catch {
            gi.status.store(@intFromEnum(GenStatus.failed), .release);
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

    const with_neg = try buildA1111Params(gpa, "a cat", "blurry", 20, 3.5, 42, 1024, 768, "krea2");
    defer gpa.free(with_neg);
    try std.testing.expectEqualStrings(
        "a cat\n" ++
            "Negative prompt: blurry\n" ++
            "Steps: 20, Sampler: Euler, Schedule type: Simple, CFG scale: 3.5, Seed: 42, Size: 1024x768, Model: krea2",
        with_neg,
    );

    // No negative → the "Negative prompt:" line is omitted entirely.
    const no_neg = try buildA1111Params(gpa, "a dog", "", 8, 1.0, 7, 512, 512, "m");
    defer gpa.free(no_neg);
    try std.testing.expectEqualStrings(
        "a dog\n" ++
            "Steps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 1.0, Seed: 7, Size: 512x512, Model: m",
        no_neg,
    );
}

test "nextSeed advances deterministically and distinctly" {
    var d: Diffuser = undefined;
    d.seed = 0;
    const a = d.nextSeed();
    const b = d.nextSeed();
    try std.testing.expect(a != 0);
    try std.testing.expect(a != b);
}
