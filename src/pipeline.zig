//! End-to-end text-to-image pipeline: tokenize -> encode -> sample -> decode.
//!
//! Models load in stages and are freed as soon as their output is captured,
//! bounding peak memory to roughly the DiT mapping (~13 GiB) plus activations.

const std = @import("std");
const gpu_mod = @import("tp_gpu");
const mem_tag = @import("tp_gpu").mem_tag;
pub const MemTag = mem_tag.MemTag;

/// MEASURED per-component diffusion VRAM (bytes), for the GUI meter. `te`/`dit`/
/// `vae` are each stage's device allocations (weights + that stage's scratch);
/// `latent` is everything else (the latent buffer, workspace, init overhead).
/// Sums to the diffusion backend's `deviceUsed()`.
pub const VramBreakdown = struct {
    te: u64 = 0,
    dit: u64 = 0,
    vae: u64 = 0,
    latent: u64 = 0,

    pub fn total(self: VramBreakdown) u64 {
        return self.te + self.dit + self.vae + self.latent;
    }
};
const ops = @import("tp_ops");

/// Thunk wiring the Vulkan device GEMM into `ops.matmul`'s injected dispatch
/// hook, so the ops layer never imports the GPU backend. Registered as
/// `ops.matmul.gpu_dispatch.call` with the `*Context` handed back as `ctx`.
fn gpuMatmulThunk(
    ctx: *anyopaque,
    y: []f32,
    x: []const f32,
    m: usize,
    w_bytes: []const u8,
    dtype_f8: bool,
    rows: usize,
    cols: usize,
    scale: f32,
    bias: ?[]const f32,
) anyerror!void {
    const c: *gpu_mod.Context = @ptrCast(@alignCast(ctx));
    return c.matmul(y, x, m, w_bytes, dtype_f8, rows, cols, scale, bias);
}
const tokenizer_mod = @import("tp_core").tokenizer;
const safetensors = @import("tp_core").safetensors;
const sampler = @import("tp_core").sampler;
const image = @import("tp_core").image;
const qwen3 = @import("tp_models").models.qwen3;
const qwen3_gpu = @import("tp_models").models.qwen3_gpu;
const krea2_text = @import("tp_models").models.krea2_text;
const dit_mod = @import("tp_models").models.dit;
const dit_gpu = @import("tp_models").models.dit_gpu;
const dit_cuda = @import("tp_models").models.dit_cuda;
const qwen3_cuda = @import("tp_models").models.qwen3_cuda;
const cuda = @import("tp_gpu").cuda;
const wan_vae = @import("tp_models").models.wan_vae;
const taehv_mod = @import("tp_models").models.taehv;
const taehv_cuda_mod = @import("tp_models").models.taehv_cuda;
const taehv_gpu_mod = @import("tp_models").models.taehv_gpu;
const vae_gpu = @import("tp_models").models.vae_gpu;
const vae_cuda = @import("tp_models").models.vae_cuda;
const vae_tiled = @import("tp_models").models.vae_tiled;

/// Compute backend for the diffusion model (and, for Vulkan, the encoder + VAE):
///  - cpu:      everything on CPU.
///  - vulkan:   encoder / DiT / VAE GEMMs offloaded to Vulkan (falls back to CPU
///              per-stage when the device is unavailable / out of VRAM).
///  - zig_cuda: whole pipeline on the hand-PTX CUDA backend (pure-Zig; int8
///              convrot checkpoint). `--backend zig-cuda`.
///  - cuda:     whole pipeline on the CUDA backend with NVIDIA's dlopen'd
///              cuBLASLt / cuDNN kernels (Phase 2). `--backend cuda`.
pub const Backend = enum {
    cpu,
    vulkan,
    zig_cuda,
    cuda,

    /// Parse a CLI value ("cpu" / "vulkan" / "zig-cuda" / "cuda"); null if
    /// unrecognized.
    pub fn fromStr(s: []const u8) ?Backend {
        if (std.mem.eql(u8, s, "cpu")) return .cpu;
        if (std.mem.eql(u8, s, "vulkan")) return .vulkan;
        if (std.mem.eql(u8, s, "zig-cuda")) return .zig_cuda;
        if (std.mem.eql(u8, s, "cuda")) return .cuda;
        return null;
    }

    /// True for the CUDA-backed variants (both drive `cuda.Backend`).
    pub fn isCuda(self: Backend) bool {
        return self == .zig_cuda or self == .cuda;
    }
};

/// VAE decode-path override (see `Options.vae_decode`). `auto` runs the adaptive
/// chain; the others force the *starting* strategy but still degrade gracefully
/// on OOM, so a forced path never hard-fails:
///   - `auto`      — whole-image first, then GPU-tiled, then CPU-tiled.
///   - `whole`     — same as `auto` (whole-image is already the first attempt).
///   - `gpu_tiled` — skip the whole-image attempt; tile on the GPU, CPU-tile on OOM.
///   - `cpu_tiled` — go straight to CPU tiling.
/// On a CPU-only backend the GPU options collapse to the CPU paths.
pub const VaeDecode = enum { auto, whole, gpu_tiled, cpu_tiled };

/// A cheap latent2rgb preview of the in-progress latent (RGB8, latent
/// resolution). Valid only for the duration of the `step` callback — copy it.
pub const Preview = struct { rgb: []const u8, width: usize, height: usize };

/// Live preview controls, read once per sampling step so a caller (the GUI) can
/// change the preview METHOD or RESOLUTION mid-generation and see it on the very
/// next completed step — no reload, no waiting for the image to finish. When
/// `Options.preview_live` points at one of these, it OVERRIDES the static
/// `preview`/`preview_ds` fields each step; the static fields are the fallback
/// for callers that don't need live control (the CLI).
pub const LivePreview = struct {
    /// 0 = none, 1 = latent2rgb, 2 = taesd (approx-VAE). Matches the GUI's
    /// `config.Preview` enum values. A `taesd` request falls back to latent2rgb
    /// when no taew decoder loaded. Read/written with acquire/release.
    method: std.atomic.Value(u8) = .init(0),
    /// Latent-resolution divisor for the taesd decode (0 = adaptive default);
    /// same meaning as `Options.preview_ds`. Applied only to the taesd path.
    ds: std.atomic.Value(u32) = .init(0),
};

/// Per-step progress hook. `step(ctx, done, total, preview)` is called once
/// after each sampling step, so a caller (e.g. a GUI) can show a live bar and
/// (when `Options.preview` is set) a live latent2rgb preview.
pub const Progress = struct {
    ctx: *anyopaque,
    step: *const fn (ctx: *anyopaque, done: usize, total: usize, preview: ?Preview) void,
};

/// A suspended generation's in-flight state: the sampler latent and the step it
/// was captured at (the loop checkpoint runs at the TOP of a step, before that
/// step's forward — so the latent is the input to `step`, and resuming re-runs
/// step `step`). Small (one latent, ~1 MB at 1024²). See `Options.suspend_out` /
/// `Options.resume_from`. (Tier 3 unload-while-paused.)
pub const Snapshot = struct {
    /// Host copy of the sampler latent (len == 16·(h/8)·(w/8)); gpa-owned.
    latent: []f32,
    /// Sampling step to resume at.
    step: usize,
};

pub const Options = struct {
    prompt: []const u8,
    negative: []const u8 = "",
    width: usize = 1024,
    height: usize = 1024,
    steps: usize = 8,
    cfg: f32 = 1.0,
    seed: u64 = 0,
    shift: f32 = sampler.default_shift,
    /// Compute backend for the sampling loop (and encoder/VAE where supported).
    backend: Backend = .cpu,
    /// VAE decode-path override (see `VaeDecode`). Default `auto` (adaptive).
    vae_decode: VaeDecode = .auto,
    /// Cap on device memory (bytes; 0 = query the driver's live budget).
    /// Weights past the cap stream per step instead of staying resident.
    vram_budget: u64 = 0,
    /// Run the GPU text encoder's GEMMs on tensor cores (f16). ~0.4s faster
    /// encode but ~doubles its image-delta contribution; default f32.
    encoder_f16: bool = false,
    text_encoder_path: []const u8 = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors",
    dit_path: []const u8 = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors",
    vae_path: []const u8 = "models/vae/krea2RealVae_v10.safetensors",
    /// Optional per-step progress hook (see `Progress`).
    on_step: ?Progress = null,
    /// Compute a latent2rgb preview each step and pass it to `on_step`.
    preview: bool = false,
    /// Optional taew2_1 approx-VAE (TAEHV) checkpoint for a higher-quality
    /// preview; falls back to latent2rgb when null or unloadable.
    taew_path: ?[]const u8 = null,
    /// Latent-resolution divisor for the TAESD preview decode: the preview is
    /// decoded at 1/`preview_ds` of the latent grid. 0 selects the adaptive
    /// default (targets ~256px). Larger = faster but blurrier.
    preview_ds: usize = 0,
    /// Optional live preview controls (see `LivePreview`). When set, the preview
    /// method + resolution are read from it each step (mid-generation switching);
    /// `preview`/`preview_ds` above are the static fallback. The taew decoder is
    /// still loaded up front (from `taew_path`) so switching TO taesd is instant.
    preview_live: ?*const LivePreview = null,
    /// Optional cancel flag, polled throughout generation — between encoder
    /// layers, between DiT blocks (mid-step, on every backend), and between
    /// VAE decode layers/tiles — so a stop lands within a fraction of a step
    /// even on the CPU backend. When it flips true, `generate` unwinds and
    /// returns `error.Canceled` (a caller-driven stop, not a failure).
    cancel: ?*std.atomic.Value(bool) = null,
    /// Optional pause gate, consulted between sampling steps (the same boundary
    /// as `cancel`). While paused the loop parks here — holding the in-flight
    /// latent and the resident DiT weights — until unpaused. See `ops/pause.zig`.
    pause: ?*ops.pause.Gate = null,
    /// Resume a suspended generation: skip noise init, load this latent, and
    /// start the sampling loop at `step` instead of 0. Conditioning + schedule
    /// are recomputed deterministically, so the result is bit-identical to an
    /// uninterrupted run. null = fresh generation. (Tier 3 unload-while-paused.)
    resume_from: ?Snapshot = null,
    /// On a paused unload (the pause gate returns `.unload`), `generate` writes a
    /// host copy of the in-flight latent + current step here (allocated with the
    /// session's gpa; the caller owns and frees it) and returns `error.Paused`,
    /// so the caller can free the model and later resume via `resume_from`. When
    /// null, a requested unload unwinds like a cancel (`error.Canceled`).
    suspend_out: ?*?Snapshot = null,
    /// Optional VRAM-reclaim hook. On a VAE-decode OOM (e.g. a very large image
    /// while the GUI chat model is resident), `generate` calls this to migrate
    /// device memory held by ANOTHER context in the process (the GUI's resident
    /// chat LLM) to the host, freeing room for the decode. `needed` is roughly
    /// how many more bytes the decode wants; the hook frees about that much
    /// (just enough — the LLM layers left resident stay fast) and returns the
    /// bytes actually freed. It may switch the calling thread's current CUDA
    /// context, so the pipeline re-binds its own after. (GUI_VRAM.md Phase 5;
    /// null everywhere else.)
    reclaim: ?Reclaim = null,
};

/// A device-VRAM reclaim callback (see `Options.reclaim`); returns the number of
/// device bytes it actually freed.
pub const Reclaim = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, needed: u64) u64,
};

pub const Image = struct {
    /// Interleaved RGB, [height][width][3].
    rgb: []u8,
    width: usize,
    height: usize,

    pub fn deinit(self: *Image, gpa: std.mem.Allocator) void {
        gpa.free(self.rgb);
        self.* = undefined;
    }
};

/// Stripped conditioning: [seq][12][2560] plus its length.
const Cond = struct {
    data: []f32,
    seq: usize,
};

// --- Tiled VAE decode adapters (see models/vae_tiled.zig) ------------------
// Each decodes a planar [16][th][tw] sub-latent to planar [3][8·th][8·tw]
// pixels on its backend; `vae_tiled.decode` drives the tiling and feather-blends
// the seams so decode VRAM stays bounded regardless of image size.

const CudaTile = struct {
    vae: *const wan_vae.Decoder,
    be: *cuda.Backend,
    cancel: ?*std.atomic.Value(bool) = null,
    fn call(self: CudaTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return vae_cuda.decode(self.vae, self.be, io, gpa, sub, th, tw, self.cancel);
    }
};

const VkTile = struct {
    vae: *const wan_vae.Decoder,
    gc: *gpu_mod.Context,
    cancel: ?*std.atomic.Value(bool) = null,
    fn call(self: VkTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return vae_gpu.decode(self.vae, self.gc, io, gpa, sub, th, tw, self.cancel);
    }
};

const CpuTile = struct {
    vae: *const wan_vae.Decoder,
    cancel: ?*std.atomic.Value(bool) = null,
    fn call(self: CpuTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return self.vae.decode(io, gpa, sub, th, tw, self.cancel);
    }
};

/// Bytes of the whole-image mid-block attention scores plane (seq = zh·zw, one
/// head; `elem` = 2 for f16 GPU scores, 4 for f32 CPU). This term grows
/// quadratically with image area and is what forces tiling on large images.
fn attnPlaneBytes(zh: usize, zw: usize, elem: u64) u64 {
    const seq: u64 = @as(u64, zh) * @as(u64, zw);
    return seq * seq * elem;
}

/// VAE-decode OOM recovery: bytes to free on the first retry round. The decode's
/// exact deficit is unknown, so we free this much, retry, and double each round
/// (see `max_reclaim_rounds`) — small enough to keep resident weights we didn't
/// need to drop, large enough that a multi-GB deficit converges in a few retries
/// (each retry re-runs the decode, so we don't want hundreds of tiny steps).
const reclaim_chunk: u64 = 1 << 30; // 1 GiB

/// Cap on VAE-decode reclaim retries before giving up on the whole-image decode
/// and dropping to tiling. With `reclaim_chunk` doubling each round this frees up
/// to ~1 TiB, far past any card — it's a stall backstop, not a real bound.
const max_reclaim_rounds: usize = 16;

/// Whether a GPU VAE-decode error is one we recover from by freeing VRAM and/or
/// stepping down the fallback ladder (whole-image → reclaim+retry → GPU tiling →
/// CPU tiling), rather than failing the whole image. VRAM exhaustion surfaces as
/// `DeviceOutOfMemory`, but the cuBLASLt / cuDNN libraries report an
/// out-of-workspace as `CublasLtError` / `CudnnError`, and the hand-PTX path can
/// surface a post-OOM stream fault as `CudaError` — all of which used to hit the
/// `else => return err` arm and hard-fail the decode even though a CPU tiled
/// decode would have succeeded. `error.Canceled` and any structural error still
/// propagate (we never want to mask those behind a silent CPU fallback).
fn recoverableDecodeErr(err: anyerror) bool {
    return switch (err) {
        error.DeviceOutOfMemory,
        error.OutOfMemory,
        error.CudaError,
        error.CublasLtError,
        error.CudnnError,
        => true,
        else => false,
    };
}

/// A reusable text-to-image pipeline. `init` creates the backend and loads the
/// text encoder, DiT, and VAE ONCE (their safetensors mappings stay open for the
/// session lifetime), so `generate` can produce many images without reloading —
/// the DiT stays resident in the backend's weight cache across a queue when the
/// VRAM budget allows. The GUI keeps one alive while its image queue is
/// non-empty; the one-shot `generate` free function below wraps init+generate+
/// deinit for the CLI / tests. NOT thread-safe: serialize `generate` calls.
pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    gpu_ctx: ?*gpu_mod.Context,
    cu_be: ?*cuda.Backend,
    // Models + their (kept-open) safetensors mappings. Held at stable addresses
    // inside this heap-allocated struct so the models' pointers into the mmaps
    // stay valid for the whole session.
    tok: tokenizer_mod.Tokenizer,
    enc_st: safetensors.SafeTensors,
    enc: qwen3.TextEncoder,
    dit_st: safetensors.SafeTensors,
    dit: dit_mod.DiT,
    vae_st: safetensors.SafeTensors,
    vae: wan_vae.Decoder,

    /// Create the backend and load all three models once. Heap-allocated (returns
    /// `*Session`) so the models can hold stable pointers into their mappings.
    /// `opts` supplies the paths + backend + initial budget; per-image fields
    /// (prompt/seed/size/steps/cfg/on_step/cancel/reclaim) are ignored here.
    /// `progress` (may be null) receives the load-timing notes.
    pub fn init(io: std.Io, gpa: std.mem.Allocator, opts: Options, progress: ?*std.Io.Writer) !*Session {
        const self = try gpa.create(Session);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.backend = opts.backend;
        self.gpu_ctx = null;
        self.cu_be = null;

        // Vulkan context (--backend vulkan): encoder / DiT / VAE GEMMs on Vulkan.
        if (opts.backend == .vulkan) {
            if (gpu_mod.Context.init(gpa)) |ctx| {
                self.gpu_ctx = ctx;
                ctx.budget_override = opts.vram_budget;
                ops.matmul.gpu_dispatch = .{ .ctx = ctx, .call = gpuMatmulThunk };
                try note(progress, "gpu: {s}\n", .{ctx.deviceName()});
                if (progress) |w| try ctx.writeCoopStatus(w);
            } else |err| {
                try note(progress, "gpu unavailable ({t}); using cpu\n", .{err});
            }
        }
        errdefer if (self.gpu_ctx) |ctx| {
            ops.matmul.gpu_dispatch = null;
            ctx.deinit();
        };

        // Hand-PTX / library CUDA backend (--backend zig-cuda / cuda): the whole
        // pipeline runs on the CUDA backend; all weights stream through its cache
        // (so a tight --vram-budget degrades to weight streaming).
        if (opts.backend.isCuda()) {
            const res = if (opts.backend == .cuda) cuda.Backend.initLibs(gpa) else cuda.Backend.init(gpa);
            if (res) |b| {
                self.cu_be = b;
                b.budget_override = opts.vram_budget;
                if (opts.backend == .cuda) {
                    const L = b.libs.?;
                    try note(progress, "cuda ({s}): cublasLt {d}, cuDNN {d}\n", .{ b.deviceName(), L.lt.cublasLtGetVersion(), L.dnn.cudnnGetVersion() });
                } else {
                    try note(progress, "cuda dit: {s}\n", .{b.deviceName()});
                }
            } else |err| {
                try note(progress, "cuda unavailable ({t}); using cpu dit\n", .{err});
            }
        }
        errdefer if (self.cu_be) |b| b.deinit();

        // MEASURED per-component VRAM attribution for the GUI meter: tag device
        // allocations by pipeline phase (set in generate()). Diffusion-only — the
        // LLM backend leaves this off, so its allocator hot path is untouched.
        if (self.cu_be) |b| b.enableMemTags();
        if (self.gpu_ctx) |c| c.enableMemTags();

        // Load all three models once; keep every mapping OPEN so weight pointers
        // stay valid across images. NO proactive evictWeights anywhere — the
        // encoder/DiT/VAE coexist under the default full-card budget, and a tight
        // budget lets the backend's LRU cache stream the overflow reactively.
        try note(progress, "loading text encoder...\n", .{});
        const t0 = std.Io.Clock.real.now(io).nanoseconds;
        self.tok = try tokenizer_mod.Tokenizer.init(gpa);
        errdefer self.tok.deinit();
        self.enc_st = try safetensors.SafeTensors.open(gpa, io, opts.text_encoder_path);
        errdefer self.enc_st.deinit();
        self.enc = try qwen3.TextEncoder.load(gpa, &self.enc_st);
        errdefer self.enc.deinit();

        try note(progress, "loading diffusion model...\n", .{});
        const t1 = std.Io.Clock.real.now(io).nanoseconds;
        self.dit_st = try safetensors.SafeTensors.open(gpa, io, opts.dit_path);
        errdefer self.dit_st.deinit();
        self.dit = try dit_mod.DiT.load(gpa, &self.dit_st);
        errdefer self.dit.deinit();

        self.vae_st = try safetensors.SafeTensors.open(gpa, io, opts.vae_path);
        errdefer self.vae_st.deinit();
        self.vae = try wan_vae.Decoder.load(gpa, &self.vae_st);
        const t2 = std.Io.Clock.real.now(io).nanoseconds;

        // Async weight streaming via a BOUNDED pinned staging ring (4×128 MB =
        // 512 MB), NOT registerHost — we deliberately do NOT page-lock the ~12 GB
        // checkpoints (that filled RAM and stalled the box). Weights are read from
        // the page-cache-backed mmap (cold→disk, warm→RAM, always reclaimable) and
        // DMA'd off the main thread, so dit_cuda's block-N+1-ahead prefetch
        // overlaps block-N compute instead of a synchronous pageable copy per
        // weight on first touch. Bounded by the existing pin_budget/eviction, so a
        // tight --vram-budget still streams within VRAM bounds. Isolated
        // (cuda-stream-test, DiT-only): cold-load 10.1s→2.4s (cold disk) /
        // 2.8s→2.2s (warm), bit-identical, MemAvailable dropped only ~2 GB.
        if (self.cu_be) |b| b.enableAsyncStreaming(io);
        try note(progress, "models loaded (encoder {d:.1}s, dit+vae {d:.1}s)\n", .{
            @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(t2 - t1)) / 1e9,
        });

        return self;
    }

    /// Device bytes this diffusion session's backend currently holds (weights +
    /// activations). 0 for non-CUDA backends. Read by the GUI status bar.
    pub fn deviceUsed(self: *const Session) u64 {
        if (self.cu_be) |b| return b.deviceUsed();
        if (self.gpu_ctx) |c| return c.device_used;
        return 0;
    }

    /// Free VRAM (bytes) on the card, for the GUI's offload telemetry. 0 on
    /// backends without a mem-info query (Vulkan) or no device. Reads the current
    /// context, so the caller must be on a thread that bound this backend.
    pub fn freeVram(self: *const Session) u64 {
        if (self.cu_be) |b| return b.ctx.memGetInfo().free;
        return 0;
    }

    /// MEASURED per-component VRAM breakdown for the GUI meter. Reads the live
    /// per-tag counters the allocator maintains (both CUDA and Vulkan backends).
    /// `latent` is the measured per-image working set (GPU session, activation
    /// workspace, live-preview decode) plus the small untagged remainder (init
    /// overhead, pools), so the parts always sum to `deviceUsed`.
    pub fn vramBreakdown(self: *const Session) VramBreakdown {
        const total = self.deviceUsed();
        if (total == 0) return .{};
        var b: VramBreakdown = .{};
        if (self.cu_be) |be| {
            b = .{ .te = be.memTagUsed(.te), .dit = be.memTagUsed(.dit), .vae = be.memTagUsed(.vae), .latent = be.memTagUsed(.latent) };
        } else if (self.gpu_ctx) |c| {
            b = .{ .te = c.memTagUsed(.te), .dit = c.memTagUsed(.dit), .vae = c.memTagUsed(.vae), .latent = c.memTagUsed(.latent) };
        }
        return foldUntagged(b, total);
    }

    /// Fold the untagged remainder of `total` into `latent` so the breakdown's
    /// parts sum to `total` (the meter's segments must account for every byte
    /// of `deviceUsed`, or the difference would misrender as "system" VRAM).
    fn foldUntagged(b: VramBreakdown, total: u64) VramBreakdown {
        var out = b;
        out.latent += total -| b.te -| b.dit -| b.vae -| b.latent;
        return out;
    }

    /// Tag subsequent device allocations with the current pipeline phase (on
    /// whichever backend is active). See generate().
    fn setMemTag(self: *Session, tag: MemTag) void {
        if (self.cu_be) |b| b.setMemTag(tag);
        if (self.gpu_ctx) |c| c.setMemTag(tag);
    }

    /// Free resident weights to fit `budget` bytes (GUI VRAM limit lowered while
    /// the queue is idle). The next generate() re-uploads what fits its budget.
    /// Caller must ensure no diffusion worker is in flight.
    pub fn trimToBudget(self: *Session, budget: u64) void {
        if (self.cu_be) |b| {
            b.bindThread();
            b.trimToBudget(budget);
        } else if (self.gpu_ctx) |c| {
            c.trimToBudget(budget);
        }
    }

    /// Incrementally free resident weights down to `target` bytes (LRU), returning
    /// the bytes freed. Unlike `trimToBudget` (all-or-nothing — `evictWeights`
    /// drops the whole cache when over budget), this frees only the excess, so the
    /// rest stays resident and the next image reloads less. This is diffusion's
    /// live VRAM-yield lever — the analog of the LLM stepper's `offloadToBudget`
    /// — so the cross-model `vram.Arbiter` can shrink an idle image model to make
    /// room for a growing LLM. Caller ensures no diffusion worker is in flight;
    /// no-op already under `target` or on a non-device backend.
    pub fn giveUpToBudget(self: *Session, target: u64) u64 {
        const used = self.deviceUsed();
        if (used <= target) return 0;
        const want = used - target;
        if (self.cu_be) |b| {
            b.bindThread();
            return b.evictToFree(want);
        }
        if (self.gpu_ctx) |c| return c.evictToFree(want);
        return 0;
    }

    pub fn deinit(self: *Session) void {
        const gpa = self.gpa;
        // The session may be torn down from a different thread than it was used
        // on (the GUI frees it on the UI thread when the image queue drains);
        // CUDA's "current context" is per-thread, so bind before freeing device
        // memory / destroying the context.
        if (self.cu_be) |b| b.bindThread();
        // Tear the compute backend down FIRST — before unmapping the checkpoint
        // safetensors below. The CUDA backend's prefetch thread streams weights
        // straight from those mmaps and DRAINS its queued requests as it joins
        // (Backend.deinit's contract: "the caller munmaps after Backend.deinit").
        // The old order unmapped first, so when a forward aborted mid-stream
        // (e.g. an OOM under VRAM pressure left a block's prefetches queued) the
        // still-draining thread read the freed DiT mapping → SIGSEGV during
        // teardown. Stopping the thread first keeps the mappings valid until no
        // one reads them.
        if (self.gpu_ctx) |ctx| {
            ops.matmul.gpu_dispatch = null;
            ctx.deinit();
        }
        if (self.cu_be) |b| b.deinit();
        self.vae.deinit();
        self.vae_st.deinit();
        self.dit.deinit();
        self.dit_st.deinit();
        self.enc.deinit();
        self.enc_st.deinit();
        self.tok.deinit();
        gpa.destroy(self);
    }

    /// Generate one image, reusing the loaded models. Rebuilds only the
    /// per-image state: conditioning, the DiT session/workspace (depend on the
    /// prompt AND resolution), the noise latent, and the schedule.
    pub fn generate(self: *Session, opts: Options, progress: ?*std.Io.Writer) !Image {
        const gpa = self.gpa;
        const io = self.io;
        if (opts.width % 16 != 0 or opts.height % 16 != 0) return error.SizeNotMultipleOf16;
        if (opts.steps < 1) return error.NoSteps;
        const lat_h = opts.height / 8;
        const lat_w = opts.width / 8;
        const lat_len = wan_vae.latent_channels * lat_h * lat_w;
        const use_cfg = opts.cfg != 1.0;
        const total_start = std.Io.Clock.real.now(io);

        const gpu_ctx = self.gpu_ctx;
        const cu_be = self.cu_be;
        // A persistent session is reused across successive GUI worker threads
        // (one per queued image); CUDA's current context is per-thread, so bind
        // this session's context to the calling thread before any device op.
        if (cu_be) |b| b.bindThread();
        // Re-apply the (possibly changed) shared VRAM budget each image: the GUI
        // sets it from what the resident LLM currently holds, which varies.
        if (gpu_ctx) |ctx| ctx.budget_override = opts.vram_budget;
        if (cu_be) |b| b.budget_override = opts.vram_budget;
        // Don't pin the transient text encoder (Stage 1): its weights are only
        // needed for this image's encode, so they should cycle out. Pinning is
        // armed for the DiT below (after encode), so the DiT stays resident
        // across queued images while the encoder/VAE cycle in the leftover.
        if (cu_be) |b| b.pin_budget = 0;

        // Stage 1: text encoding (reusing the resident encoder). Tag its device
        // allocations (encoder weights + encode scratch) as TE for the meter.
        self.setMemTag(.te);
        const enc_start = std.Io.Clock.real.now(io);
        const cond_pos = try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &self.tok, &self.enc, opts.prompt, opts.cancel);
        defer gpa.free(cond_pos.data);
        const cond_neg: ?Cond = if (use_cfg)
            try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &self.tok, &self.enc, opts.negative, opts.cancel)
        else
            null;
        defer if (cond_neg) |c| gpa.free(c.data);
        try note(progress, "encoded prompt ({d} tokens{s}) in {d:.1}s\n", .{
            cond_pos.seq,                                                                                 if (use_cfg) " + negative" else "",
            @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - enc_start.nanoseconds)) / 1e9,
        });

        // Pin the DiT across images (GUI_VRAM.md Phase 4): first-touch pinning
        // during sampling keeps the (large) DiT weights resident so a queued
        // image reuses them instead of re-uploading ~13 GB each time. Sized to the
        // available budget minus a reserve for the encoder (~5 GB, unpinned) +
        // sampling activations — so it self-limits: a generous budget (image
        // priority) pins the whole DiT; a tight one pins part (rest streams). The
        // encoder above stayed unpinned (pin_budget 0); the VAE (after the DiT
        // fills pin_budget) stays unpinned too.
        if (cu_be) |b| {
            // Pin as much DiT as fits. Two caps: the shared BUDGET (opts.vram_budget,
            // 0 = no cap; = card limit − LLM resident in the GUI), and — critically
            // — what is PHYSICALLY reachable right now. The budget is blind to other
            // processes on the card (desktop, a running ComfyUI, …); if we pin to it
            // we OOM the moment physical VRAM runs out, and pinned weights can't be
            // evicted to recover. Physical room = live free VRAM + our own unpinned
            // weights (the text encoder), which evict + re-stream as the DiT pins.
            // Reserve the live activation scratch + a small margin on top.
            const free_now = b.ctx.memGetInfo().free;
            const room = free_now + b.evictableWeightBytes();
            const budget = if (opts.vram_budget > 0) @min(opts.vram_budget, room) else room;
            const pin_reserve: u64 = b.attn_scratch_budget + (512 << 20);
            b.pin_budget = budget -| pin_reserve;
            // Same reserve as a LIVE floor for first-touch pinning: pinNew keeps
            // this much VRAM free, so pinning never eats the room the (lazily
            // allocated, per-block) attention scratch + activation workspace need.
            // pin_budget above is blind to the working set not yet allocated at
            // pin time; the floor is the physical backstop that makes streaming
            // actually fit — the whole point of a tight budget.
            b.pin_floor = pin_reserve;
            std.log.info("[diff-vram] budget={d}MB room={d}MB reserve={d}MB pin={d}MB free={d}MB", .{
                opts.vram_budget >> 20, room >> 20, pin_reserve >> 20, b.pin_budget >> 20, free_now >> 20,
            });

            // Proactively drop the transient text encoder when it can't stay
            // resident alongside the DiT's working set. It's evictable but nothing
            // forces it out until the DiT's own allocations reactively reclaim it
            // — which happens DURING step 1, so step 1 pins little (streams around
            // the resident encoder) and runs slow; only once it's cleared do later
            // steps pin properly and speed up (the 11s→3s first-step cliff under a
            // resident LLM). Evicting up front lets step 1 pin from the start. The
            // encode (both CFG passes) is already done, so the encoder has no
            // further use THIS image; on a big card where it all fits we keep it
            // (the next queued image's encode reuses it). At this point — post
            // encode, pre-DiT-Session.init — the weight cache holds ONLY the
            // encoder, so evictWeights drops exactly it. Condition: the DiT weights
            // + activation reserve don't fit in the live free VRAM with the encoder
            // still resident (room already counts the encoder as reclaimable, so
            // the test reduces to dit + reserve > free_now).
            const dit_bytes = self.dit_st.payload.len;
            if (dit_bytes + pin_reserve > free_now) {
                // evictUnpinned (NOT evictWeights): keep a DiT pinned by a previous
                // queued image, drop only the (unpinned) encoder + any stray stream.
                const freed = b.evictUnpinned();
                if (freed > 0) std.log.info("[diff-vram] dropped {d}MB of unpinned weights (text encoder) up front — won't fit alongside the DiT working set (dit={d}MB + reserve={d}MB > free={d}MB)", .{
                    freed >> 20, dit_bytes >> 20, pin_reserve >> 20, free_now >> 20,
                });
            }
        }

        // Stage 2: flow-matching sampling (reusing the resident DiT). DiT weights
        // (streamed lazily during forward) + per-step attention scratch are tagged
        // DiT; the per-image working set (GPU session, activation workspace,
        // preview decode) is tagged `latent` below.
        self.setMemTag(.dit);
        const x = try gpa.alloc(f32, lat_len);
        defer gpa.free(x);
        // Resume: restore the suspended latent instead of drawing fresh noise
        // (the schedule + conditioning above are recomputed identically). A
        // length mismatch (shouldn't happen — resume reuses the same opts) falls
        // back to a fresh draw rather than a bad copy.
        if (opts.resume_from) |r| {
            if (r.latent.len == x.len) @memcpy(x, r.latent) else sampler.fillNoise(x, opts.seed);
        } else sampler.fillNoise(x, opts.seed);

        const sigmas = try sampler.simpleSchedule(gpa, opts.steps, opts.shift);
        defer gpa.free(sigmas);

        {
            const dit = &self.dit;
            const v = try gpa.alloc(f32, lat_len);
            defer gpa.free(v);
            const v_neg = if (use_cfg) try gpa.alloc(f32, lat_len) else null;
            defer if (v_neg) |b| gpa.free(b);

            // Per-image GPU session: text fusion, rope table, timestep vectors
            // computed + uploaded once per image (they depend on the prompt +
            // resolution). The DiT WEIGHTS stay cached in the backend across images.
            // These per-image buffers (session + activation workspace) are the
            // meter's "latent / working" segment — tag them so it's MEASURED, not
            // a remainder.
            self.setMemTag(.latent);
            var sess_pos: ?dit_gpu.Session = null;
            defer if (sess_pos) |*sp| sp.deinit(gpa, gpu_ctx.?);
            var sess_neg: ?dit_gpu.Session = null;
            defer if (sess_neg) |*sn| sn.deinit(gpa, gpu_ctx.?);
            var ws: ?dit_gpu.Workspace = null;
            defer if (ws) |*w| w.deinit(gpu_ctx.?);
            if (cu_be == null) if (gpu_ctx) |gc| {
                sess_pos = try dit_gpu.Session.init(gpa, io, gc, dit, lat_h, lat_w, cond_pos.data, cond_pos.seq, sigmas);
                if (use_cfg) {
                    sess_neg = try dit_gpu.Session.init(gpa, io, gc, dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq, sigmas);
                }
                const seq_txt_cap = @max(cond_pos.seq, if (cond_neg) |c| c.seq else 0);
                ws = try dit_gpu.Workspace.init(gc, lat_h, lat_w, seq_txt_cap);
            };
            var cu_pos: ?dit_cuda.Session = null;
            defer if (cu_pos) |*s| s.deinit(cu_be.?);
            var cu_neg: ?dit_cuda.Session = null;
            defer if (cu_neg) |*s| s.deinit(cu_be.?);
            var cu_ws: ?dit_cuda.Workspace = null;
            defer if (cu_ws) |*w| w.deinit(cu_be.?);
            if (cu_be) |b| {
                cu_pos = try dit_cuda.Session.init(gpa, io, b, dit, lat_h, lat_w, cond_pos.data, cond_pos.seq);
                if (use_cfg) cu_neg = try dit_cuda.Session.init(gpa, io, b, dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq);
                const cu_seq_cap = @max(cond_pos.seq, if (cond_neg) |c| c.seq else 0);
                cu_ws = try dit_cuda.Workspace.init(b, lat_h, lat_w, cu_seq_cap);
            }
            // Back to DiT for the sampling loop: lazily streamed DiT weights and
            // per-step attention scratch belong to the DiT segment.
            self.setMemTag(.dit);

            // A preview can be produced this run when there's a step hook AND
            // either the static preview is on OR a live control is attached (which
            // can toggle it on mid-generation). When a live control is present we
            // allocate the (small) scratch buffers and preload the taew decoder up
            // front regardless of the CURRENT method, so a mid-run switch to any
            // method takes effect on the next step with no reload.
            const preview_active = opts.on_step != null and (opts.preview or opts.preview_live != null);

            // Scratch for the per-step latent2rgb preview (latent resolution RGB8).
            const preview_scratch: ?[]u8 = if (preview_active)
                try gpa.alloc(u8, lat_h * lat_w * 3)
            else
                null;
            defer if (preview_scratch) |ps| gpa.free(ps);

            // Scratch for the per-step denoised (x0) estimate that we preview.
            // We decode the model's clean-image estimate `x - sigma*v`, not the
            // raw noisy latent `x` — matching ComfyUI's preview (it decodes the
            // `denoised` sample), so the preview reads as a blurry image that
            // sharpens rather than noise that resolves only at the last steps.
            const preview_x0: ?[]f32 = if (preview_active)
                try gpa.alloc(f32, x.len)
            else
                null;
            defer if (preview_x0) |p| gpa.free(p);

            // Clamp a latent-resolution divisor to the grid so a large divisor on
            // a small latent can't collapse to a 0-sized decode. 0 → the adaptive
            // default that targets a ~256px preview.
            const clampDs = struct {
                fn f(ds: usize, lh: usize, lw: usize) usize {
                    return @min(@max(1, @min(lh, lw)), if (ds > 0) ds else @max(1, @max(lh, lw) / 32));
                }
            }.f;

            // Optional taew2_1 (TAEHV) approx-VAE for a sharper preview. Loaded up
            // front whenever a preview is active and a taew is configured — even if
            // the current method isn't taesd — so a live switch to taesd is instant.
            var taew_st: ?safetensors.SafeTensors = null;
            defer if (taew_st) |*s| s.deinit();
            var taehv_dec: ?taehv_mod.Decoder = null;
            defer if (taehv_dec) |*d| d.deinit();
            if (preview_active) if (opts.taew_path) |tp| {
                if (safetensors.SafeTensors.open(gpa, io, tp)) |tst| {
                    taew_st = tst;
                    if (taehv_mod.Decoder.load(gpa, &taew_st.?)) |d| {
                        taehv_dec = d;
                        try note(progress, "preview: taew2_1 approx-VAE ready\n", .{});
                    } else |err| try note(progress, "taew2_1 load failed ({t}); latent2rgb preview\n", .{err});
                } else |err| try note(progress, "taew2_1 open failed ({t}); latent2rgb preview\n", .{err});
            };

            const sampling_start = std.Io.Clock.real.now(io);
            const start_step = if (opts.resume_from) |r| @min(r.step, opts.steps) else 0;
            for (start_step..opts.steps) |i| {
                if (opts.cancel) |c| if (c.load(.acquire)) return error.Canceled;
                if (opts.pause) |g| switch (g.checkpoint(io, opts.cancel)) {
                    .proceed => {},
                    .canceled => return error.Canceled,
                    // Unload-while-paused: snapshot the in-flight latent + this
                    // step to host so the caller can free the model and resume
                    // bit-identically later (via resume_from). `x` here is the
                    // input to step `i` (the checkpoint runs before the forward).
                    .unload => {
                        if (opts.suspend_out) |so| {
                            so.* = .{ .latent = try gpa.dupe(f32, x), .step = i };
                            return error.Paused;
                        }
                        return error.Canceled;
                    },
                };
                const start = std.Io.Clock.real.now(io);
                if (cu_be) |b| {
                    try dit_cuda.forward(dit, b, &cu_pos.?, &cu_ws.?, io, gpa, v, x, sigmas[i], opts.cancel);
                } else if (gpu_ctx) |gc| {
                    try dit_gpu.forward(dit, gc, &sess_pos.?, &ws.?, io, gpa, v, x, sigmas[i], opts.cancel);
                } else {
                    try dit.forward(io, gpa, v, x, lat_h, lat_w, sigmas[i], cond_pos.data, cond_pos.seq, opts.cancel);
                }
                if (use_cfg) {
                    if (cu_be) |b| {
                        try dit_cuda.forward(dit, b, &cu_neg.?, &cu_ws.?, io, gpa, v_neg.?, x, sigmas[i], opts.cancel);
                    } else if (gpu_ctx) |gc| {
                        try dit_gpu.forward(dit, gc, &sess_neg.?, &ws.?, io, gpa, v_neg.?, x, sigmas[i], opts.cancel);
                    } else try dit.forward(io, gpa, v_neg.?, x, lat_h, lat_w, sigmas[i], cond_neg.?.data, cond_neg.?.seq, opts.cancel);
                    sampler.applyCfg(v, v_neg.?, opts.cfg);
                }
                sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
                const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - start.nanoseconds)) / 1e6;
                try note(progress, "step {d}/{d}  sigma {d:.3} -> {d:.3}  ({d:.1}s)\n", .{ i + 1, opts.steps, sigmas[i], sigmas[i + 1], ms / 1000.0 });
                if (opts.on_step) |p| {
                    // Live-preview decode allocations (taew weights + scratch) are
                    // working memory, not DiT.
                    self.setMemTag(.latent);
                    defer self.setMemTag(.dit);
                    var pv: ?Preview = null;
                    var taew_rgb: ?[]u8 = null;
                    defer if (taew_rgb) |r| gpa.free(r);

                    // Effective preview method + resolution THIS step. A live
                    // control (the GUI) overrides the static fields, so a method /
                    // quality change made mid-generation shows on the next step. A
                    // taesd request with no decoder loaded degrades to latent2rgb.
                    var method: u8 = if (opts.preview_live) |lp|
                        lp.method.load(.acquire)
                    else if (opts.preview)
                        (if (taehv_dec != null) @as(u8, 2) else 1)
                    else
                        0;
                    if (method == 2 and taehv_dec == null) method = 1;

                    if (method != 0) {
                        // Denoised (x0) estimate = x - sigma*v. `eulerStep` above
                        // set x to x_{i+1} = x_i + (sigma_{i+1} - sigma_i)*v, so the
                        // clean estimate in terms of the post-step latent is
                        // x - sigma_{i+1}*v (collapses to x on the final step where
                        // sigma_{i+1}==0).
                        const x0: []const f32 = if (preview_x0) |px0| blk: {
                            const s_next = sigmas[i + 1];
                            for (px0, x, v) |*o, xi, vi| o.* = xi - s_next * vi;
                            break :blk px0;
                        } else x;
                        if (method == 2) if (taehv_dec) |*d| taew_blk: {
                            const live_ds: usize = if (opts.preview_live) |lp| lp.ds.load(.acquire) else opts.preview_ds;
                            const ds = clampDs(live_ds, lat_h, lat_w);
                            const th = lat_h / ds;
                            const tw = lat_w / ds;
                            const small = downsampleLatent(gpa, x0, lat_h, lat_w, ds) catch break :taew_blk;
                            defer gpa.free(small);
                            const rgb = if (cu_be) |b|
                                (taehv_cuda_mod.decode(d, b, gpa, small, th, tw) catch break :taew_blk)
                            else if (gpu_ctx) |gc|
                                (taehv_gpu_mod.decode(d, gc, gpa, small, th, tw) catch break :taew_blk)
                            else
                                (d.decode(io, gpa, small, th, tw) catch break :taew_blk);
                            taew_rgb = rgb;
                            pv = .{ .rgb = rgb, .width = tw * taehv_mod.spatial_scale, .height = th * taehv_mod.spatial_scale };
                        };
                        if (pv == null) if (preview_scratch) |ps| {
                            wan_vae.latentPreviewInto(ps, x0, lat_h, lat_w);
                            pv = .{ .rgb = ps, .width = lat_w, .height = lat_h };
                        };
                    }
                    p.step(p.ctx, i + 1, opts.steps, pv);
                }
            }
            const sampling_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - sampling_start.nanoseconds)) / 1e9;
            try note(progress, "sampling {d} steps in {d:.1}s ({d:.2}s/step)\n", .{ opts.steps, sampling_s, sampling_s / @as(f64, @floatFromInt(opts.steps)) });
            // Peak-of-sampling attribution (the per-image session/workspace is
            // still alive here) — the same numbers the GUI meter shows.
            if (self.cu_be != null or self.gpu_ctx != null) {
                const bd = self.vramBreakdown();
                std.log.info("[diff-vram] breakdown te={d}MB dit={d}MB latent={d}MB vae={d}MB (used={d}MB)", .{
                    bd.te >> 20, bd.dit >> 20, bd.latent >> 20, bd.vae >> 20, bd.total() >> 20,
                });
            }
        }

        if (opts.cancel) |c| if (c.load(.acquire)) return error.Canceled;

        // Stage 3: denormalize and decode (reusing the resident VAE). Tag its
        // device allocations (VAE weights + decode scratch) as VAE.
        self.setMemTag(.vae);
        // Stage 3: denormalize and decode (reusing the resident VAE). NO
        // evictWeights — the DiT stays resident for the next queued image; under
        // a tight budget the backend's LRU cache streams the overflow reactively.
        {
            const plane = lat_h * lat_w;
            for (0..wan_vae.latent_channels) |c| {
                for (x[c * plane ..][0..plane]) |*val| {
                    val.* = val.* * wan_vae.latents_std[c] + wan_vae.latents_mean[c];
                }
            }
        }
        try note(progress, "decoding latent...\n", .{});
        const dec_start = std.Io.Clock.real.now(io);
        const vae = &self.vae;
        // The whole-image decode is fastest and seamless, but its peak VRAM — in
        // particular the O(seq²) mid-block attention scores plane — grows with
        // image area and OOMs on large images. When it won't fit we decode in
        // overlapping tiles (bounded footprint, feather-blended seams) on the GPU
        // rather than crawling on the CPU; only if even a tile can't fit do we
        // drop to a CPU tiled decode. See models/vae_tiled.zig.
        const tp: vae_tiled.Params = .{};
        // Decode-path override (opts.vae_decode). `force_cpu` short-circuits every
        // backend to CPU tiling; `skip_whole` starts at GPU tiling (no whole-image
        // attempt). `auto`/`whole` leave both false — whole-image is already the
        // first attempt, so the two behave identically.
        const force_cpu = opts.vae_decode == .cpu_tiled;
        const skip_whole = opts.vae_decode == .gpu_tiled or opts.vae_decode == .cpu_tiled;
        const planar = if (force_cpu) planar_blk: {
            try note(progress, "vae decode: tiling on CPU (forced)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae, .cancel = opts.cancel }, CpuTile.call);
        } else if (cu_be) |b| planar_blk: {
            // Attempt ladder — whole-image (fastest, seamless) → GPU tiling
            // (bounded footprint) → CPU (guaranteed). Before EACH GPU retry free
            // JUST ENOUGH VRAM and try again, keeping the rest resident so we
            // reload as little as possible: drop LRU weights from THIS backend's
            // own cache first (the DiT — dead for the rest of this image,
            // re-streams next image), and only when that can't cover the deficit
            // reach into the chat LLM's context. The freed amount escalates so a
            // big deficit converges in a few retries. OOM can arrive as
            // DeviceOutOfMemory OR as a cuBLASLt/cuDNN out-of-workspace error
            // (see recoverableDecodeErr). (skip_whole jumps straight to tiling.)
            try note(progress, "vae decode: mode={s} pinned={d}MB streamed={d}MB free={d}MB\n", .{
                @tagName(opts.vae_decode), b.pinnedWeightBytes() >> 20, b.evictableWeightBytes() >> 20, b.ctx.memGetInfo().free >> 20,
            });
            var want: u64 = reclaim_chunk;
            // Free ~`wnt` bytes across this backend's own cache (LRU incl. the
            // now-dead DiT) then the chat LLM, log it, and report the total freed
            // (0 ⇒ nothing left to free).
            const freeSome = struct {
                fn call(bk: *cuda.Backend, reclaim: ?Reclaim, w: ?*std.Io.Writer, cio: std.Io, wnt: u64) !u64 {
                    const t0 = std.Io.Clock.real.now(cio).nanoseconds;
                    bk.bindThread();
                    const from_self = bk.evictToFree(wnt);
                    var got = from_self;
                    if (got < wnt) if (reclaim) |r| {
                        got += r.call(r.ctx, wnt - got);
                        bk.bindThread(); // reclaim may have switched the context
                    };
                    const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(cio).nanoseconds - t0)) / 1e6;
                    try note(w, "vae decode: freed {d}MB (own {d}MB + llm {d}MB) of {d}MB wanted in {d:.0}ms\n", .{
                        got >> 20, from_self >> 20, (got - from_self) >> 20, wnt >> 20, ms,
                    });
                    return got;
                }
            }.call;
            // PROACTIVE pre-free: estimate the first attempt's peak activation VRAM
            // and free up front so we hit free ≥ 110% of it — avoids burning full
            // failed decodes just to discover the deficit (the reactive loops below
            // still catch any shortfall, since the estimate can't see the opaque
            // cuBLASLt/cuDNN conv workspace). Tiled decode's first attempt is one
            // tile, so estimate at the tile size when skip_whole.
            {
                const est = if (skip_whole) vae.estimatePeakBytes(tp.tile, tp.tile) else vae.estimatePeakBytes(lat_h, lat_w);
                const target = est + est / 10; // 110%
                const free_now = b.ctx.memGetInfo().free;
                try note(progress, "vae decode: est peak {d}MB, want free ≥ {d}MB (have {d}MB)\n", .{ est >> 20, target >> 20, free_now >> 20 });
                if (free_now < target) _ = try freeSome(b, opts.reclaim, progress, io, target - free_now);
            }
            // Phase 1: whole-image with incremental eviction.
            if (!skip_whole) {
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    if (vae_cuda.decode(vae, b, io, gpa, x, lat_h, lat_w, opts.cancel)) |p| {
                        break :planar_blk p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: whole-image OOM ({t}) → freeing VRAM\n", .{err});
                    if (try freeSome(b, opts.reclaim, progress, io, want) == 0) break; // nothing left → tile
                    want *|= 2;
                }
            }
            // Phase 2: GPU tiling (bounded) with the same incremental eviction —
            // after the DiT is evicted a tile easily fits, and it's far faster
            // than the CPU floor below.
            {
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    try note(progress, "vae decode: tiling on GPU ({d}² latent tiles)\n", .{tp.tile});
                    const ct = CudaTile{ .vae = vae, .be = b, .cancel = opts.cancel };
                    if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, CudaTile.call)) |p| {
                        break :planar_blk p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: GPU tiling OOM ({t}) → freeing VRAM\n", .{err});
                    if (try freeSome(b, opts.reclaim, progress, io, want) == 0) break; // nothing left → CPU
                    want *|= 2;
                }
            }
            // Phase 3: CPU tiling — the guaranteed VRAM-can't-OOM floor (slow).
            try note(progress, "vae decode: GPU out of VRAM → CPU tiled decode (slow)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae, .cancel = opts.cancel }, CpuTile.call);
        } else if (gpu_ctx) |gc| planar_blk: {
            // Whole-image decode first. The Vulkan mid-block attention is now
            // query-tiled (flash), so this OOMs only when the conv activation
            // buffers don't fit. On OOM free JUST ENOUGH VRAM and retry, keeping
            // the rest resident: each round drops LRU weights from this context's
            // own cache first (the DiT), and only when that can't cover it
            // reaches into the chat LLM's context. The freed amount escalates so
            // a large deficit converges fast. If nothing more can be freed, tile
            // on the GPU; if even a tile won't fit, decode on the CPU.
            // (skip_whole jumps straight to tiling.)
            try note(progress, "vae decode: mode={s} free={d}MB\n", .{ @tagName(opts.vae_decode), gc.liveVram() >> 20 });
            {
                // Proactive pre-free to ~110% of the estimated first-attempt peak,
                // so we don't burn a full failed decode just to find the deficit.
                const est = if (skip_whole) vae.estimatePeakBytes(tp.tile, tp.tile) else vae.estimatePeakBytes(lat_h, lat_w);
                const target = est + est / 10;
                const free_now = gc.liveVram();
                try note(progress, "vae decode: est peak {d}MB, want free >= {d}MB (have {d}MB)\n", .{ est >> 20, target >> 20, free_now >> 20 });
                if (free_now < target) {
                    _ = gc.evictToFree(target - free_now);
                    if (opts.reclaim) |r| _ = r.call(r.ctx, target -| gc.liveVram());
                }
            }
            if (!skip_whole) {
                var want: u64 = reclaim_chunk;
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    if (vae_gpu.decode(vae, gc, io, gpa, x, lat_h, lat_w, opts.cancel)) |p| {
                        break :planar_blk p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: whole-image OOM ({t}) -> freeing VRAM\n", .{err});
                    const t0 = std.Io.Clock.real.now(io).nanoseconds;
                    const from_self = gc.evictToFree(want); // own resident weights first
                    var got = from_self;
                    if (got < want) if (opts.reclaim) |r| {
                        got += r.call(r.ctx, want - got); // then the chat LLM
                    };
                    const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;
                    try note(progress, "vae decode: freed {d}MB (own {d}MB + llm {d}MB) of {d}MB wanted in {d:.0}ms\n", .{ got >> 20, from_self >> 20, (got - from_self) >> 20, want >> 20, ms });
                    if (got == 0) break; // nothing left to free → tile
                    want *|= 2; // escalate so a big deficit converges fast
                }
            }
            // Phase 2: GPU tiling (bounded) with the same incremental eviction.
            {
                var wt: u64 = reclaim_chunk;
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    try note(progress, "vae decode: tiling on GPU ({d}^2 latent tiles)\n", .{tp.tile});
                    const vt = VkTile{ .vae = vae, .gc = gc, .cancel = opts.cancel };
                    if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, vt, VkTile.call)) |p| {
                        break :planar_blk p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: GPU tiling OOM ({t}) -> freeing VRAM\n", .{err});
                    const t0 = std.Io.Clock.real.now(io).nanoseconds;
                    const from_self = gc.evictToFree(wt);
                    var got = from_self;
                    if (got < wt) if (opts.reclaim) |r| {
                        got += r.call(r.ctx, wt - got);
                    };
                    const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;
                    try note(progress, "vae decode: freed {d}MB (own {d}MB + llm {d}MB) for tiling in {d:.0}ms\n", .{ got >> 20, from_self >> 20, (got - from_self) >> 20, ms });
                    if (got == 0) break; // nothing left → CPU
                    wt *|= 2;
                }
            }
            // Phase 3: CPU tiling — the guaranteed VRAM-can't-OOM floor (slow).
            try note(progress, "vae decode: GPU out of VRAM -> CPU tiled decode (slow)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae, .cancel = opts.cancel }, CpuTile.call);
        } else planar_blk: {
            // CPU-only: tile once the whole-image scores plane (f32) gets large,
            // so a big image doesn't try to allocate tens of GB of host RAM.
            // (skip_whole — e.g. gpu_tiled with no GPU — forces tiling.)
            if (!skip_whole and attnPlaneBytes(lat_h, lat_w, 4) < (1 << 30))
                break :planar_blk try vae.decode(io, gpa, x, lat_h, lat_w, opts.cancel);
            try note(progress, "vae decode: tiling on CPU\n", .{});
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae, .cancel = opts.cancel }, CpuTile.call);
        };
        defer gpa.free(planar);
        try note(progress, "decoded in {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dec_start.nanoseconds)) / 1e9});

        const rgb = try image.planarF32ToRgb8(gpa, planar, opts.width, opts.height);
        try note(progress, "total time {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - total_start.nanoseconds)) / 1e9});
        return .{ .rgb = rgb, .width = opts.width, .height = opts.height };
    }
};

/// One-shot text-to-image: load the models, generate a single image, free.
/// Used by the CLI and tests; the GUI uses a persistent `Session` across a queue.
pub fn generate(io: std.Io, gpa: std.mem.Allocator, opts: Options, progress: ?*std.Io.Writer) !Image {
    // Reject invalid dimensions/steps before loading any models (Session.init
    // maps ~18 GiB of checkpoints); Session.generate re-checks per image.
    if (opts.width % 16 != 0 or opts.height % 16 != 0) return error.SizeNotMultipleOf16;
    if (opts.steps < 1) return error.NoSteps;
    var s = try Session.init(io, gpa, opts, progress);
    defer s.deinit();
    return s.generate(opts, progress);
}

fn encodePrompt(io: std.Io, gpa: std.mem.Allocator, gpu_ctx: ?*gpu_mod.Context, cu_be: ?*cuda.Backend, encoder_f16: bool, tok: *const tokenizer_mod.Tokenizer, enc: *const qwen3.TextEncoder, text: []const u8, cancel: ?*std.atomic.Value(bool)) !Cond {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(tok, gpa, text, &ids);

    // GPU-resident encode (batched, keeps the device saturated): the CUDA
    // backend when active, else Vulkan; the CPU forward is the fallback (and
    // used on any GPU error — except a cancel, which must propagate, not
    // silently restart the encode on the CPU).
    const full = if (cu_be) |b|
        qwen3_cuda.encode(enc, b, io, gpa, ids.items, cancel) catch |err| blk: {
            if (err == error.Canceled) return err;
            std.log.warn("cuda text encode failed ({t}); falling back to CPU (slow)", .{err});
            break :blk try enc.encode(io, gpa, ids.items, cancel);
        }
    else if (gpu_ctx) |gc|
        qwen3_gpu.encode(enc, gc, io, gpa, ids.items, encoder_f16, cancel) catch |err| blk: {
            if (err == error.Canceled) return err;
            std.log.warn("vulkan text encode failed ({t}); falling back to CPU (slow)", .{err});
            break :blk try enc.encode(io, gpa, ids.items, cancel);
        }
    else
        try enc.encode(io, gpa, ids.items, cancel);
    defer gpa.free(full);

    const offset = krea2_text.stripOffset(ids.items);
    const seq = ids.items.len - offset;
    const row = qwen3.tap_count * qwen3.hidden;
    const data = try gpa.alloc(f32, seq * row);
    @memcpy(data, full[offset * row ..][0 .. seq * row]);
    return .{ .data = data, .seq = seq };
}

/// Box-average a planar [C][h][w] latent down by integer factor `f`
/// (→ [C][h/f][w/f]) so the taew preview decode stays cheap.
fn downsampleLatent(gpa: std.mem.Allocator, x: []const f32, h: usize, w: usize, f: usize) ![]f32 {
    const c = wan_vae.latent_channels;
    const th = h / f;
    const tw = w / f;
    const out = try gpa.alloc(f32, c * th * tw);
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(f * f));
    for (0..c) |ch| {
        const src = x[ch * h * w ..];
        const dst = out[ch * th * tw ..];
        for (0..th) |oy| for (0..tw) |ox| {
            var sum: f32 = 0;
            for (0..f) |dy| for (0..f) |dx| {
                sum += src[(oy * f + dy) * w + (ox * f + dx)];
            };
            dst[oy * tw + ox] = sum * inv;
        };
    }
    return out;
}

fn note(progress: ?*std.Io.Writer, comptime fmt: []const u8, args: anytype) !void {
    if (progress) |w| {
        try w.print(fmt, args);
        try w.flush();
    }
}

test "options validation" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expectError(error.SizeNotMultipleOf16, generate(io, gpa, .{ .prompt = "x", .width = 100, .height = 96 }, null));
    try std.testing.expectError(error.NoSteps, generate(io, gpa, .{ .prompt = "x", .steps = 0 }, null));
}

test "vramBreakdown folds only the untagged remainder into latent" {
    const gib_b: u64 = 1 << 30;
    // Typical mid-generation state: all four tags populated plus a little
    // untagged init overhead. The measured latent must be preserved and only
    // the remainder added, so the parts sum to deviceUsed.
    const b = Session.foldUntagged(
        .{ .te = 5 * gib_b, .dit = 13 * gib_b, .vae = 1 * gib_b, .latent = 2 * gib_b },
        21 * gib_b + (100 << 20),
    );
    try std.testing.expectEqual(2 * gib_b + (100 << 20), b.latent);
    try std.testing.expectEqual(21 * gib_b + (100 << 20), b.total());
    // Counters momentarily exceeding deviceUsed (benign cross-thread race) must
    // saturate, not underflow.
    const r = Session.foldUntagged(.{ .te = 2 * gib_b, .dit = 2 * gib_b, .vae = 0, .latent = 0 }, 3 * gib_b);
    try std.testing.expectEqual(@as(u64, 0), r.latent);
}

test "recoverableDecodeErr classifies VAE-decode fallbacks" {
    // Every way a GPU decode can run out of VRAM must trigger the reclaim +
    // tiling ladder — including the cuBLASLt / cuDNN out-of-workspace errors and
    // the hand-PTX post-OOM stream fault, which previously escaped it.
    try std.testing.expect(recoverableDecodeErr(error.DeviceOutOfMemory));
    try std.testing.expect(recoverableDecodeErr(error.OutOfMemory));
    try std.testing.expect(recoverableDecodeErr(error.CudaError));
    try std.testing.expect(recoverableDecodeErr(error.CublasLtError));
    try std.testing.expect(recoverableDecodeErr(error.CudnnError));
    // Cancellation and structural errors must propagate, never be masked by a
    // silent CPU fallback.
    try std.testing.expect(!recoverableDecodeErr(error.Canceled));
    try std.testing.expect(!recoverableDecodeErr(error.SizeNotMultipleOf16));
}
