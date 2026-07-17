//! End-to-end text-to-image pipeline: tokenize -> encode -> sample -> decode.
//!
//! Models load in stages and are freed as soon as their output is captured,
//! bounding peak memory to roughly the DiT mapping (~13 GiB) plus activations.

const std = @import("std");
const gpu_mod = @import("gpu.zig");
const mem_tag = @import("gpu/mem_tag.zig");
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
const ops = @import("ops.zig");
const tokenizer_mod = @import("tokenizer.zig");
const safetensors = @import("safetensors.zig");
const sampler = @import("sampler.zig");
const image = @import("image.zig");
const qwen3 = @import("models/qwen3.zig");
const qwen3_gpu = @import("models/qwen3_gpu.zig");
const krea2_text = @import("models/krea2_text.zig");
const dit_mod = @import("models/dit.zig");
const dit_gpu = @import("models/dit_gpu.zig");
const dit_cuda = @import("models/dit_cuda.zig");
const qwen3_cuda = @import("models/qwen3_cuda.zig");
const cuda = @import("gpu/cuda.zig");
const wan_vae = @import("models/wan_vae.zig");
const taehv_mod = @import("models/taehv.zig");
const taehv_cuda_mod = @import("models/taehv_cuda.zig");
const vae_gpu = @import("models/vae_gpu.zig");
const vae_cuda = @import("models/vae_cuda.zig");
const vae_tiled = @import("models/vae_tiled.zig");

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

/// Per-step progress hook. `step(ctx, done, total, preview)` is called once
/// after each sampling step, so a caller (e.g. a GUI) can show a live bar and
/// (when `Options.preview` is set) a live latent2rgb preview.
pub const Progress = struct {
    ctx: *anyopaque,
    step: *const fn (ctx: *anyopaque, done: usize, total: usize, preview: ?Preview) void,
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
    /// Optional cancel flag, polled between sampling steps and before the VAE
    /// decode. When it flips true, `generate` unwinds and returns
    /// `error.Canceled` (a caller-driven stop, not a failure).
    cancel: ?*std.atomic.Value(bool) = null,
    /// Optional VRAM-reclaim hook. On a CUDA VAE-decode OOM (e.g. a very large
    /// image), `generate` calls this to free device memory held by ANOTHER CUDA
    /// context in the process (the GUI's resident chat model) and retries the
    /// decode. Returns true if it freed anything. It may switch the calling
    /// thread's current CUDA context, so the pipeline re-binds its own after.
    /// (GUI_VRAM.md Phase 5; null everywhere else.)
    reclaim: ?Reclaim = null,
};

/// A device-VRAM reclaim callback (see `Options.reclaim`).
pub const Reclaim = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, needed: u64) bool,
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
    fn call(self: CudaTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return vae_cuda.decode(self.vae, self.be, io, gpa, sub, th, tw);
    }
};

const VkTile = struct {
    vae: *const wan_vae.Decoder,
    gc: *gpu_mod.Context,
    fn call(self: VkTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return vae_gpu.decode(self.vae, self.gc, io, gpa, sub, th, tw);
    }
};

const CpuTile = struct {
    vae: *const wan_vae.Decoder,
    fn call(self: CpuTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return self.vae.decode(io, gpa, sub, th, tw);
    }
};

/// Bytes of the whole-image mid-block attention scores plane (seq = zh·zw, one
/// head; `elem` = 2 for f16 GPU scores, 4 for f32 CPU). This term grows
/// quadratically with image area and is what forces tiling on large images.
fn attnPlaneBytes(zh: usize, zw: usize, elem: u64) u64 {
    const seq: u64 = @as(u64, zh) * @as(u64, zw);
    return seq * seq * elem;
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
                ops.matmul.gpu = ctx;
                try note(progress, "gpu: {s}\n", .{ctx.deviceName()});
                if (progress) |w| try ctx.writeCoopStatus(w);
            } else |err| {
                try note(progress, "gpu unavailable ({t}); using cpu\n", .{err});
            }
        }
        errdefer if (self.gpu_ctx) |ctx| {
            ops.matmul.gpu = null;
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

    /// MEASURED per-component VRAM breakdown for the GUI meter. Reads the live
    /// per-tag counters the allocator maintains (both CUDA and Vulkan backends).
    /// `latent` absorbs the untagged remainder so the parts sum to `deviceUsed`.
    pub fn vramBreakdown(self: *const Session) VramBreakdown {
        const total = self.deviceUsed();
        if (total == 0) return .{};
        var te: u64 = 0;
        var dit: u64 = 0;
        var vae: u64 = 0;
        if (self.cu_be) |b| {
            te = b.memTagUsed(.te);
            dit = b.memTagUsed(.dit);
            vae = b.memTagUsed(.vae);
        } else if (self.gpu_ctx) |c| {
            te = c.memTagUsed(.te);
            dit = c.memTagUsed(.dit);
            vae = c.memTagUsed(.vae);
        }
        return .{ .te = te, .dit = dit, .vae = vae, .latent = total -| te -| dit -| vae };
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

    pub fn deinit(self: *Session) void {
        const gpa = self.gpa;
        // The session may be torn down from a different thread than it was used
        // on (the GUI frees it on the UI thread when the image queue drains);
        // CUDA's "current context" is per-thread, so bind before freeing device
        // memory / destroying the context.
        if (self.cu_be) |b| b.bindThread();
        self.vae.deinit();
        self.vae_st.deinit();
        self.dit.deinit();
        self.dit_st.deinit();
        self.enc.deinit();
        self.enc_st.deinit();
        self.tok.deinit();
        if (self.gpu_ctx) |ctx| {
            ops.matmul.gpu = null;
            ctx.deinit();
        }
        if (self.cu_be) |b| b.deinit();
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
        const cond_pos = try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &self.tok, &self.enc, opts.prompt);
        defer gpa.free(cond_pos.data);
        const cond_neg: ?Cond = if (use_cfg)
            try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &self.tok, &self.enc, opts.negative)
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
            // Pin as much DiT as fits. `avail` is the shared budget (0 = all free
            // VRAM). Reserve only the live activation scratch + a small margin —
            // NOT the encoder (it's unpinned and evicts as the DiT pins), so a big
            // reserve just leaves VRAM sitting empty.
            const avail = if (opts.vram_budget > 0) opts.vram_budget else b.ctx.memGetInfo().free;
            const pin_reserve: u64 = b.attn_scratch_budget + (512 << 20);
            b.pin_budget = avail -| pin_reserve;
            std.log.info("[diff-vram] budget={d}MB avail={d}MB reserve={d}MB pin={d}MB free={d}MB", .{
                opts.vram_budget >> 20, avail >> 20, pin_reserve >> 20, b.pin_budget >> 20, b.ctx.memGetInfo().free >> 20,
            });
        }

        // Stage 2: flow-matching sampling (reusing the resident DiT). Tag its
        // device allocations (DiT weights + per-step scratch + latent) as DiT.
        self.setMemTag(.dit);
        const x = try gpa.alloc(f32, lat_len);
        defer gpa.free(x);
        sampler.fillNoise(x, opts.seed);

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

            // Scratch for the per-step latent2rgb preview (latent resolution RGB8).
            const preview_scratch: ?[]u8 = if (opts.preview and opts.on_step != null)
                try gpa.alloc(u8, lat_h * lat_w * 3)
            else
                null;
            defer if (preview_scratch) |ps| gpa.free(ps);

            // Scratch for the per-step denoised (x0) estimate that we preview.
            // We decode the model's clean-image estimate `x - sigma*v`, not the
            // raw noisy latent `x` — matching ComfyUI's preview (it decodes the
            // `denoised` sample), so the preview reads as a blurry image that
            // sharpens rather than noise that resolves only at the last steps.
            const preview_x0: ?[]f32 = if (opts.preview and opts.on_step != null)
                try gpa.alloc(f32, x.len)
            else
                null;
            defer if (preview_x0) |p| gpa.free(p);

            // Optional taew2_1 (TAEHV) approx-VAE for a sharper preview.
            var taew_st: ?safetensors.SafeTensors = null;
            defer if (taew_st) |*s| s.deinit();
            var taehv_dec: ?taehv_mod.Decoder = null;
            defer if (taehv_dec) |*d| d.deinit();
            const preview_ds: usize = @max(1, @max(lat_h, lat_w) / 32); // downsample so preview ≈ 256px
            if (opts.preview and opts.on_step != null) if (opts.taew_path) |tp| {
                if (safetensors.SafeTensors.open(gpa, io, tp)) |tst| {
                    taew_st = tst;
                    if (taehv_mod.Decoder.load(gpa, &taew_st.?)) |d| {
                        taehv_dec = d;
                        try note(progress, "preview: taew2_1 approx-VAE (1/{d} latent)\n", .{preview_ds});
                    } else |err| try note(progress, "taew2_1 load failed ({t}); latent2rgb preview\n", .{err});
                } else |err| try note(progress, "taew2_1 open failed ({t}); latent2rgb preview\n", .{err});
            };

            const sampling_start = std.Io.Clock.real.now(io);
            for (0..opts.steps) |i| {
                if (opts.cancel) |c| if (c.load(.acquire)) return error.Canceled;
                const start = std.Io.Clock.real.now(io);
                if (cu_be) |b| {
                    try dit_cuda.forward(dit, b, &cu_pos.?, &cu_ws.?, io, gpa, v, x, sigmas[i], opts.cancel);
                } else if (gpu_ctx) |gc| {
                    try dit_gpu.forward(dit, gc, &sess_pos.?, &ws.?, io, gpa, v, x, sigmas[i]);
                } else {
                    try dit.forward(io, gpa, v, x, lat_h, lat_w, sigmas[i], cond_pos.data, cond_pos.seq);
                }
                if (use_cfg) {
                    if (cu_be) |b| {
                        try dit_cuda.forward(dit, b, &cu_neg.?, &cu_ws.?, io, gpa, v_neg.?, x, sigmas[i], opts.cancel);
                    } else if (gpu_ctx) |gc| {
                        try dit_gpu.forward(dit, gc, &sess_neg.?, &ws.?, io, gpa, v_neg.?, x, sigmas[i]);
                    } else try dit.forward(io, gpa, v_neg.?, x, lat_h, lat_w, sigmas[i], cond_neg.?.data, cond_neg.?.seq);
                    sampler.applyCfg(v, v_neg.?, opts.cfg);
                }
                sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
                const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - start.nanoseconds)) / 1e6;
                try note(progress, "step {d}/{d}  sigma {d:.3} -> {d:.3}  ({d:.1}s)\n", .{ i + 1, opts.steps, sigmas[i], sigmas[i + 1], ms / 1000.0 });
                if (opts.on_step) |p| {
                    var pv: ?Preview = null;
                    var taew_rgb: ?[]u8 = null;
                    defer if (taew_rgb) |r| gpa.free(r);
                    // Denoised (x0) estimate = x - sigma*v. `eulerStep` above set
                    // x to x_{i+1} = x_i + (sigma_{i+1} - sigma_i)*v, so the clean
                    // estimate in terms of the post-step latent is x - sigma_{i+1}*v
                    // (collapses to x on the final step where sigma_{i+1}==0).
                    const x0: []const f32 = if (preview_x0) |px0| blk: {
                        const s_next = sigmas[i + 1];
                        for (px0, x, v) |*o, xi, vi| o.* = xi - s_next * vi;
                        break :blk px0;
                    } else x;
                    if (taehv_dec) |*d| taew_blk: {
                        const th = lat_h / preview_ds;
                        const tw = lat_w / preview_ds;
                        const small = downsampleLatent(gpa, x0, lat_h, lat_w, preview_ds) catch break :taew_blk;
                        defer gpa.free(small);
                        const rgb = if (cu_be) |b|
                            (taehv_cuda_mod.decode(d, b, gpa, small, th, tw) catch break :taew_blk)
                        else
                            (d.decode(io, gpa, small, th, tw) catch break :taew_blk);
                        taew_rgb = rgb;
                        pv = .{ .rgb = rgb, .width = tw * taehv_mod.spatial_scale, .height = th * taehv_mod.spatial_scale };
                    }
                    if (pv == null) if (preview_scratch) |ps| {
                        wan_vae.latentPreviewInto(ps, x0, lat_h, lat_w);
                        pv = .{ .rgb = ps, .width = lat_w, .height = lat_h };
                    };
                    p.step(p.ctx, i + 1, opts.steps, pv);
                }
            }
            const sampling_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - sampling_start.nanoseconds)) / 1e9;
            try note(progress, "sampling {d} steps in {d:.1}s ({d:.2}s/step)\n", .{ opts.steps, sampling_s, sampling_s / @as(f64, @floatFromInt(opts.steps)) });
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
            const saved = ops.matmul.gpu;
            ops.matmul.gpu = null;
            defer ops.matmul.gpu = saved;
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae }, CpuTile.call);
        } else if (cu_be) |b| planar_blk: {
            // Whole-image decode first (fastest, seamless). The CUDA mid-block
            // attention is bounded (hand-PTX flash on zig-cuda, cuDNN flash on
            // libs), so this OOMs only when the conv activation buffers don't fit
            // — then reclaim VRAM held by another CUDA context (the GUI chat
            // model) and retry, and finally tile on the GPU. (skip_whole jumps
            // straight to tiling.)
            if (!skip_whole) {
                var attempt: usize = 0;
                while (attempt < 4) : (attempt += 1) {
                    if (vae_cuda.decode(vae, b, io, gpa, x, lat_h, lat_w)) |p| {
                        break :planar_blk p;
                    } else |err| switch (err) {
                        error.DeviceOutOfMemory, error.OutOfMemory => {
                            const freed = opts.reclaim != null and opts.reclaim.?.call(opts.reclaim.?.ctx, 0);
                            if (freed) {
                                b.bindThread(); // reclaim may have switched the current context
                                try note(progress, "vae decode: out of VRAM, reclaimed → retry\n", .{});
                                continue;
                            }
                            break; // nothing more to reclaim → tile
                        },
                        else => return err,
                    }
                }
            }
            try note(progress, "vae decode: tiling on GPU ({d}² latent tiles)\n", .{tp.tile});
            const ct = CudaTile{ .vae = vae, .be = b };
            if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, CudaTile.call)) |p| {
                break :planar_blk p;
            } else |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    try note(progress, "vae decode: GPU tiling out of VRAM → CPU tiled decode\n", .{});
                    const saved = ops.matmul.gpu;
                    ops.matmul.gpu = null;
                    defer ops.matmul.gpu = saved;
                    break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae }, CpuTile.call);
                },
                else => return err,
            }
        } else if (gpu_ctx) |gc| planar_blk: {
            // Whole-image decode first. The Vulkan mid-block attention is now
            // query-tiled (flash), so this OOMs only when the conv activation
            // buffers don't fit — then tile the whole decode on the GPU.
            // (skip_whole jumps straight to tiling.)
            if (!skip_whole) {
                if (vae_gpu.decode(vae, gc, io, gpa, x, lat_h, lat_w)) |p| {
                    break :planar_blk p;
                } else |err| switch (err) {
                    error.DeviceOutOfMemory => {},
                    else => return err,
                }
            }
            try note(progress, "vae decode: tiling on GPU ({d}² latent tiles)\n", .{tp.tile});
            const vt = VkTile{ .vae = vae, .gc = gc };
            if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, vt, VkTile.call)) |p| {
                break :planar_blk p;
            } else |err| switch (err) {
                error.DeviceOutOfMemory => {
                    try note(progress, "vae decode: GPU tiling out of VRAM → CPU tiled decode\n", .{});
                    const saved = ops.matmul.gpu;
                    ops.matmul.gpu = null;
                    defer ops.matmul.gpu = saved;
                    break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae }, CpuTile.call);
                },
                else => return err,
            }
        } else planar_blk: {
            // CPU-only: tile once the whole-image scores plane (f32) gets large,
            // so a big image doesn't try to allocate tens of GB of host RAM.
            // (skip_whole — e.g. gpu_tiled with no GPU — forces tiling.)
            if (!skip_whole and attnPlaneBytes(lat_h, lat_w, 4) < (1 << 30))
                break :planar_blk try vae.decode(io, gpa, x, lat_h, lat_w);
            try note(progress, "vae decode: tiling on CPU\n", .{});
            break :planar_blk try vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, CpuTile{ .vae = vae }, CpuTile.call);
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

fn encodePrompt(io: std.Io, gpa: std.mem.Allocator, gpu_ctx: ?*gpu_mod.Context, cu_be: ?*cuda.Backend, encoder_f16: bool, tok: *const tokenizer_mod.Tokenizer, enc: *const qwen3.TextEncoder, text: []const u8) !Cond {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(tok, gpa, text, &ids);

    // GPU-resident encode (batched, keeps the device saturated): the CUDA
    // backend when active, else Vulkan; the CPU forward is the fallback (and
    // used on any GPU error).
    const full = if (cu_be) |b|
        qwen3_cuda.encode(enc, b, io, gpa, ids.items) catch |err| blk: {
            std.log.warn("cuda text encode failed ({t}); falling back to CPU (slow)", .{err});
            break :blk try enc.encode(io, gpa, ids.items);
        }
    else if (gpu_ctx) |gc|
        qwen3_gpu.encode(enc, gc, io, gpa, ids.items, encoder_f16) catch |err| blk: {
            std.log.warn("vulkan text encode failed ({t}); falling back to CPU (slow)", .{err});
            break :blk try enc.encode(io, gpa, ids.items);
        }
    else
        try enc.encode(io, gpa, ids.items);
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
