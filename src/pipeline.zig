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
const noise_mod = @import("tp_core").noise;
const safetensors = @import("tp_core").safetensors;
const gguf_mod = @import("tp_core").gguf;
const weights_mod = @import("tp_core").weights;
const sampler = @import("tp_core").sampler;
const image = @import("tp_core").image;
const qwen3 = @import("tp_models").models.qwen3;
const qwen3_gpu = @import("tp_models").models.qwen3_gpu;
const krea2_text = @import("tp_models").models.krea2_text;
const dit_mod = @import("tp_models").models.dit;
const zimage = @import("tp_models").models.zimage;
const zimage_text = @import("tp_models").models.zimage_text;
const zimage_gpu = @import("tp_models").models.zimage_gpu;
const zimage_cuda = @import("tp_models").models.zimage_cuda;
const dit_gpu = @import("tp_models").models.dit_gpu;
const dit_cuda = @import("tp_models").models.dit_cuda;
const qwen3_cuda = @import("tp_models").models.qwen3_cuda;
const cuda = @import("tp_gpu").cuda;
/// std.posix.getenv is gone in 0.16 and we link libc (see backend.zig).
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
const wan_vae = @import("tp_models").models.wan_vae;
const clip_tok = @import("tp_core").clip_tokenizer;
const prompt_a1111 = @import("tp_core").prompt_a1111;
const clip_text = @import("tp_models").models.clip_text;
const clip_text_gpu = @import("tp_models").models.clip_text_gpu;
const clip_text_cuda = @import("tp_models").models.clip_text_cuda;
const sd_unet = @import("tp_models").models.sd_unet;
const sd_unet_gpu = @import("tp_models").models.sd_unet_gpu;
const sd_unet_cuda = @import("tp_models").models.sd_unet_cuda;
const sd_vae = @import("tp_models").models.sd_vae;
const sd_vae_gpu = @import("tp_models").models.sd_vae_gpu;
const sd_vae_cuda = @import("tp_models").models.sd_vae_cuda;
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
    /// **Multistep sampler state**, gpa-owned, null for a first-order sampler.
    ///
    /// ⚠️ A latent alone is not enough to resume a DPM++(2M) run bit-identically:
    /// the step after the resume applies a second-order correction built from the
    /// PREVIOUS step's denoised prediction. Dropping it is not a crash — the
    /// resumed step silently degrades to first order and the image differs from an
    /// uninterrupted render, which is exactly the promise `resume_from` makes.
    sde_old_denoised: ?[]f32 = null,
    /// The `h` of the step before the resume point, the other half of that state.
    sde_h_last: f64 = 0,

    pub fn deinit(self: *Snapshot, gpa: std.mem.Allocator) void {
        gpa.free(self.latent);
        if (self.sde_old_denoised) |o| gpa.free(o);
        self.* = undefined;
    }
};

pub const Options = struct {
    prompt: []const u8,
    negative: []const u8 = "",
    /// Which prompt dialect `prompt`/`negative` are written in. See `PromptSyntax` —
    /// the two are not interchangeable spellings of the same thing.
    prompt_syntax: PromptSyntax = .comfy,
    /// How an A1111 attention weight reaches the hidden states. Ignored under `.comfy`.
    emphasis: Emphasis = .original,
    /// Whose **sampling** conventions to follow. Orthogonal to `prompt_syntax`: the same
    /// prompt text can be read either dialect's way under either ecosystem's sampler, and
    /// reproducing an A1111 render needs both set. See `Compat`.
    compat: Compat = .comfy,
    /// Per-knob overrides of `compat`, for an A1111 user who changed one of the three
    /// settings from its default (each is a checkbox there, so this is normal, not
    /// exotic). null = whatever `compat` says.
    rng: ?noise_mod.Source = null,
    sgm_noise_multiplier: ?bool = null,
    quantize_timestep: ?bool = null,
    width: usize = 1024,
    height: usize = 1024,
    steps: usize = 8,
    cfg: f32 = 1.0,
    seed: u64 = 0,
    /// Sigma-schedule shift. ⚠️ **Its default is family-dependent**, so leaving
    /// `explicit_shift` false is not the same as passing this value: krea2 wants
    /// 1.15, Z-Image 3.0, and the SD family ignores it entirely (its ladder comes
    /// from the training betas). `Session.resolvedShift` picks.
    shift: f32 = sampler.default_shift,
    /// True when the caller actually asked for `shift`. Same reasoning as
    /// `explicit_text_encoder`: a *defaulted* value is not a request, and treating
    /// one as such would silently render Z-Image on krea2's shift.
    explicit_shift: bool = false,
    /// Which sampler drives the loop. `euler` is the default and the only
    /// first-order one; see `sampler.Kind`.
    sampler: sampler.Kind = .euler,
    /// Which scheduler places the steps. null = the family's default (`simple` for
    /// krea2, `normal` for the SD family), which is what every render used before
    /// this became selectable. See `sampler.Scheduler`.
    ///
    /// ⚠️ `ddim_uniform` and `beta` can return a different number of steps than
    /// `steps` asks for; `generate` reports and drives off the real count.
    scheduler: ?sampler.Scheduler = null,
    /// SDE noise level (`eta`) and noise multiplier (`s_noise`), ignored by `euler`.
    /// ComfyUI's defaults; `sde_eta = 0` makes an SDE sampler deterministic and
    /// equal to plain DPM++(2M).
    sde_eta: f64 = 1.0,
    sde_s_noise: f64 = 1.0,
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
    /// Where the conditioner and decoder come from **when they are not in the primary
    /// checkpoint**. These defaults are krea2's separate files, and they are a
    /// *fallback*, not an override: `resolveComponent` prefers the primary checkpoint's
    /// own copy over a defaulted path, and prefers an explicitly supplied path over
    /// both (see `explicit_text_encoder` / `explicit_vae`).
    ///
    /// ⚠️ Treating a *defaulted* path as "the user specified this" broke a joined SD1.5
    /// checkpoint: the resolver dutifully opened krea2's qwen3 encoder, looked for
    /// CLIP's tensors in it and reported `ComponentNotInCheckpoint`.
    text_encoder_path: []const u8 = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors",
    dit_path: []const u8 = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors",
    vae_path: []const u8 = "models/vae/krea2RealVae_v10.safetensors",
    /// SDXL's **second** text encoder (OpenCLIP bigG), for the split-file case. Empty by
    /// default rather than pointing anywhere: unlike krea2's components there is no
    /// canonical standalone file for it, and a bundled SDXL checkpoint always carries it.
    text_encoder_2_path: []const u8 = "",
    /// True when the caller actually asked for those paths. A CLI sets this when the
    /// flag was present; a library caller that fills the field in means it.
    explicit_text_encoder: bool = false,
    explicit_text_encoder_2: bool = false,
    explicit_vae: bool = false,
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

    /// `compat` with the per-knob overrides applied — the form the sampling code reads.
    pub fn compatConfig(self: *const Options) CompatConfig {
        var c: CompatConfig = .of(self.compat);
        if (self.rng) |v| c.noise_src = v;
        if (self.sgm_noise_multiplier) |v| c.sgm_noise_multiplier = v;
        if (self.quantize_timestep) |v| c.quantize_timestep = v;
        return c;
    }
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

/// Stripped conditioning: [seq][12][2560] plus its length. Produced by
/// `Session.encode`, consumed by `Session.denoiser` / `Session.predict`; the
/// caller owns `data`.
pub const Cond = struct {
    data: []f32,
    seq: usize,
    /// SDXL's pooled CLIP-G vector (1280). Null for the families that do not use one.
    ///
    /// It rides on the conditioning rather than being folded into it because the vector
    /// the UNet actually consumes (`y`) also carries the *image size*, which `encode` does
    /// not know — so the pooled half is produced here and combined in `Session.denoiser`,
    /// which does.
    pooled: ?[]f32 = null,
    /// Extra conditionings for a prompt whose text CHANGES DURING the render — A1111's
    /// `[a:b:0.5]` and `[a|b]`. Null for every static prompt, which is every prompt in
    /// the ComfyUI dialect.
    ///
    /// ⚠️ **Entry 0 is this `Cond` itself** (`data`/`seq`/`pooled`), so a caller that
    /// ignores `sched` reads the step-0 conditioning and behaves exactly as before. That
    /// is what keeps `Session.predict`, the composed-stages invariant and ggufy's whole
    /// measurement ladder working unchanged.
    sched: ?Schedule = null,

    /// The step → conditioning map for a scheduled prompt.
    pub const Schedule = struct {
        /// Entries 1.., each a complete conditioning with its own `seq` and `pooled`.
        /// Deduplicated by prompt text: `[a|b]` over 35 steps has 35 schedule *entries*
        /// but only 2 distinct texts, and building 35 conditionings for 2 prompts would
        /// cost 33 pointless tower forwards and 33 device sessions.
        extra: []Cond,
        /// Per-step index into `[this] ++ extra`, length = the render's step count.
        at: []u8,

        /// Which conditioning step `i` (0-based) uses. Steps past the end clamp to the
        /// last, so a schedule built for fewer steps than the render still resolves.
        pub fn indexAt(self: Schedule, i: usize) usize {
            if (self.at.len == 0) return 0;
            return self.at[@min(i, self.at.len - 1)];
        }
    };

    /// Total distinct conditionings, including entry 0.
    pub fn entryCount(self: *const Cond) usize {
        return 1 + if (self.sched) |s| s.extra.len else 0;
    }

    /// Conditioning `i`, where 0 is this one.
    pub fn entry(self: *const Cond, i: usize) *const Cond {
        if (i == 0) return self;
        return &self.sched.?.extra[i - 1];
    }

    pub fn deinit(self: *Cond, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
        if (self.pooled) |p| gpa.free(p);
        if (self.sched) |s| {
            // The extras are plain Conds with no schedule of their own, so this recurses
            // exactly one level.
            for (s.extra) |*e| e.deinit(gpa);
            gpa.free(s.extra);
            gpa.free(s.at);
        }
        self.* = undefined;
    }
};

/// Per-call knobs for `Session.encode`. Defaults match `Options`, so
/// `encode(gpa, prompt, .{})` is what `generate` does.
pub const EncodeOptions = struct {
    /// Run the GPU text encoder's GEMMs on tensor cores (f16). See
    /// `Options.encoder_f16`.
    encoder_f16: bool = false,
    /// Which prompt dialect to parse. See `PromptSyntax`.
    prompt_syntax: PromptSyntax = .comfy,
    /// How an attention weight reaches the hidden states. Only read for `.a1111`.
    emphasis: Emphasis = .original,
    /// Steps the render will take — needed only by `.a1111`, whose `[a:b:when]` and
    /// `[a|b]` resolve against the step count. Zero means "one entry", which is what a
    /// caller that is not rendering a schedule wants.
    steps: usize = 0,
    cancel: ?*std.atomic.Value(bool) = null,
};

/// Which prompt dialect to parse. The two are NOT variations on a theme: see
/// `core/prompt_a1111.zig` for the five things that differ.
pub const PromptSyntax = enum {
    /// ComfyUI: `(x:w)` replaces the weight, `[x]` is literal text, `BREAK` is a word,
    /// and weights interpolate away from the empty prompt's hidden states.
    comfy,
    /// AUTOMATIC1111: `(x:w)` multiplies, `[x]` de-emphasizes, `BREAK` splits a chunk, a
    /// chunk boundary backtracks to the last comma, weights multiply the hidden states,
    /// and `[a:b:when]` / `[a|b]` schedule the prompt across steps.
    a1111,
};

/// How A1111 applies an attention weight to the hidden states
/// (`modules/sd_emphasis.py`). Ignored under `.comfy`, which has exactly one form.
pub const Emphasis = enum {
    /// `EmphasisOriginal`: multiply, then one global rescale restoring the chunk mean.
    original,
    /// `EmphasisOriginalNoNorm`: multiply only. Upstream documents this as working
    /// better for SDXL, and it is the form that leaves unweighted tokens untouched.
    no_norm,
    /// `EmphasisIgnore`: parse the syntax and strip it, but weight everything 1.0 —
    /// useful for isolating how much of a difference the weighting itself makes.
    ignore,
};

/// Which ecosystem's **sampling** conventions to reproduce — a separate axis from
/// `PromptSyntax`, which is about the prompt *text*.
///
/// ⚠️ **Three independent conventions, each a documented A1111 option whose default
/// disagrees with ComfyUI's behaviour**, and this engine hardwired ComfyUI's choice for
/// all three. Reproducing an A1111 render needs all of them; getting one wrong is enough
/// to make the image a different image:
///
/// | | A1111 default | ComfyUI (was unconditional here) | cost of the wrong one |
/// |---|---|---|---|
/// | `randn_source` | **`GPU`** — NVIDIA Philox | CPU MT19937 | an *unrelated* image |
/// | `sgm_noise_multiplier` | **`False`** — `x·σ₀` | `x·sqrt(1+σ₀²)` | ~9 dB |
/// | `enable_quantization` | **`False`** — fractional σ→t | nearest trained index | ~25 dB |
///
/// The first is categorically worse than the other two: they perturb a trajectory, it
/// replaces the starting point. Upstream's own note on the option is "changes seeds
/// drastically". The second and third are measured in `sampler.sdScaleInitialNoise` and
/// `sampler.sdModelTimestep`, whose doc comments already name diffusers as the other side
/// of the same disagreement — A1111 happens to sit on diffusers' side of both.
pub const Compat = enum {
    /// ComfyUI's conventions, which every render here used before this was selectable.
    comfy,
    /// AUTOMATIC1111's **defaults**. ⚠️ Not "A1111" in general: each of the three is a
    /// user-visible setting there, so an A1111 user who changed one needs the matching
    /// override below.
    a1111,
};

/// `Compat` resolved into the three switches the sampling code reads, after any
/// per-knob override. Built by `Options.compatConfig`.
pub const CompatConfig = struct {
    /// Which generator the initial latent — and every Brownian-tree node — draws from.
    noise_src: noise_mod.Source = .torch_cpu,
    /// `x·sqrt(1+σ₀²)` when true (ComfyUI / the official SGM implementation), a bare
    /// `x·σ₀` when false (A1111's default).
    sgm_noise_multiplier: bool = true,
    /// Snap the UNet's timestep to the nearest *trained* index (ComfyUI) rather than
    /// passing the fractional one (A1111's `enable_quantization = False`).
    quantize_timestep: bool = true,

    pub fn of(compat: Compat) CompatConfig {
        return switch (compat) {
            .comfy => .{},
            .a1111 => .{
                .noise_src = .nv_philox,
                .sgm_noise_multiplier = false,
                .quantize_timestep = false,
            },
        };
    }
};

/// Per-call knobs for `Session.decode`, mirroring the `Options` fields the VAE
/// stage reads.
pub const DecodeOptions = struct {
    vae_decode: VaeDecode = .auto,
    cancel: ?*std.atomic.Value(bool) = null,
    reclaim: ?Reclaim = null,
};

/// The flow-matching sigma schedule for `steps` steps (length `steps + 1`,
/// ending at 0). Re-exported here so a caller driving its own sampling loop over
/// `Session.predict` gets the same schedule `generate` uses without reaching into
/// `sampler`. Caller frees.
pub fn schedule(gpa: std.mem.Allocator, steps: usize, shift: f32) ![]f32 {
    return sampler.simpleSchedule(gpa, steps, shift);
}

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

/// The SD family's three tile adapters. Same contract as the krea2 ones above —
/// planar `[c][th][tw]` sub-latent in, planar `[3][8·th][8·tw]` pixels out — with
/// the channel-last transposition the SD decoders want folded in on either side,
/// so `vae_tiled` and `Session.decode` stay planar-only.
///
/// ⚠️ The transposition is the layout trap that produced a periodic streak
/// pattern when the UNet landed (see CLAUDE.md): it moves every value's position
/// and none of its magnitude, so getting it wrong is invisible to any norm check.
/// It lives here, once, for exactly that reason.
fn sdTile(
    gpa: std.mem.Allocator,
    sub: []const f32,
    th: usize,
    tw: usize,
    ctx: anytype,
    comptime inner: fn (@TypeOf(ctx), std.mem.Allocator, []const f32, usize, usize) anyerror![]f32,
) anyerror![]f32 {
    const n = th * tw;
    // ⚠️ **DERIVED, not `sd_vae.latent_channels`.** That constant is 4, and the same
    // decoder body also runs Z-Image's 16-channel Flux latent. Hardcoding it
    // transposed only the first quarter of the latent and left the rest reading
    // whatever followed — which rendered as a band of colour noise across the top of
    // an otherwise flat grey image, with the assert below compiled out in ReleaseFast.
    // `vae_tiled.decode` already derives its channel count the same way.
    std.debug.assert(n != 0 and sub.len % n == 0);
    const c = sub.len / n;
    const z = try gpa.alloc(f32, sub.len);
    defer gpa.free(z);
    for (0..n) |px| {
        for (0..c) |ch| z[px * c + ch] = sub[ch * n + px];
    }
    const rows = try inner(ctx, gpa, z, th, tw); // channel-last [ph*pw][3]
    defer gpa.free(rows);
    const pn = n * sd_vae.spatial_scale * sd_vae.spatial_scale;
    std.debug.assert(rows.len == pn * 3);
    const planar = try gpa.alloc(f32, rows.len);
    errdefer gpa.free(planar);
    for (0..pn) |px| {
        for (0..3) |ch| planar[ch * pn + px] = rows[px * 3 + ch];
    }
    return planar;
}

const SdCudaTile = struct {
    vae: *const sd_vae.Decoder,
    be: *cuda.Backend,
    cancel: ?*std.atomic.Value(bool) = null,
    fn inner(self: SdCudaTile, gpa: std.mem.Allocator, z: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return sd_vae_cuda.decode(self.vae, self.be, gpa, z, th, tw, self.cancel);
    }
    fn call(self: SdCudaTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        _ = io;
        return sdTile(gpa, sub, th, tw, self, inner);
    }
};

const SdVkTile = struct {
    vae: *const sd_vae.Decoder,
    gc: *gpu_mod.Context,
    cancel: ?*std.atomic.Value(bool) = null,
    fn inner(self: SdVkTile, gpa: std.mem.Allocator, z: []const f32, th: usize, tw: usize) anyerror![]f32 {
        return sd_vae_gpu.decode(self.vae, self.gc, gpa, z, th, tw, self.cancel);
    }
    fn call(self: SdVkTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        _ = io;
        return sdTile(gpa, sub, th, tw, self, inner);
    }
};

const SdCpuTile = struct {
    vae: *const sd_vae.Decoder,
    io: std.Io,
    cancel: ?*std.atomic.Value(bool) = null,
    fn inner(self: SdCpuTile, gpa: std.mem.Allocator, z: []const f32, th: usize, tw: usize) anyerror![]f32 {
        // The CPU decoder takes no cancel token (unlike wan_vae's), so a stop
        // lands at the next tile boundary rather than mid-tile.
        if (self.cancel) |c| if (c.load(.acquire)) return error.Canceled;
        return self.vae.decode(self.io, gpa, z, th, tw);
    }
    fn call(self: SdCpuTile, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
        _ = io;
        return sdTile(gpa, sub, th, tw, self, inner);
    }
};

/// The family-specific half of the VAE-decode ladder (`Session.decodePlanar`):
/// the per-backend tile context, the peak-VRAM estimate, and the tile geometry.
/// A whole-image decode is just a single tile covering the whole latent, so these
/// three contexts are all the ladder needs to be written once for both families.
const KreaVae = struct {
    vae: *const wan_vae.Decoder,
    /// 128² latent tiles (~1 MP): caps the mid-block scores plane at 512 MiB.
    const tiling: vae_tiled.Params = .{};
    fn estimate(self: KreaVae, zh: usize, zw: usize, scores_resident: bool) u64 {
        _ = scores_resident; // wan_vae's estimator has no scores-plane term
        return self.vae.estimatePeakBytes(zh, zw);
    }
    fn cudaCtx(self: KreaVae, be: *cuda.Backend, cancel: ?*std.atomic.Value(bool)) CudaTile {
        return .{ .vae = self.vae, .be = be, .cancel = cancel };
    }
    fn vkCtx(self: KreaVae, gc: *gpu_mod.Context, cancel: ?*std.atomic.Value(bool)) VkTile {
        return .{ .vae = self.vae, .gc = gc, .cancel = cancel };
    }
    fn cpuCtx(self: KreaVae, io: std.Io, cancel: ?*std.atomic.Value(bool)) CpuTile {
        _ = io;
        return .{ .vae = self.vae, .cancel = cancel };
    }
};

const SdVae = struct {
    vae: *const sd_vae.Decoder,
    /// ⚠️ HALF krea2's tile, and that is the whole point of tiling here. The SD
    /// decoder's widest activation is 256 channels at FULL image resolution, so a
    /// 128² latent tile (1024² pixels) still needs 3 x 1 GiB — barely under a
    /// 1024x1536 whole-image decode, i.e. a tiling that saves nothing. 64² latent
    /// (512² pixels) costs 3 x 256 MiB and is also what ComfyUI's tiled SD decode
    /// defaults to; overlap 8 keeps its 25% seam ratio.
    const tiling: vae_tiled.Params = .{ .tile = 64, .overlap = 8 };
    /// `scores_resident` doubles as "this is the Vulkan arm": it is the only
    /// backend that materializes the scores plane, and also the only one that has
    /// not been taught f16 activation storage. Kept as one flag rather than two so
    /// they cannot be set inconsistently.
    fn estimate(self: SdVae, zh: usize, zw: usize, scores_resident: bool) u64 {
        return self.vae.estimatePeakBytes(zh, zw, scores_resident, self.vae.cfg.act_f16 and !scores_resident);
    }
    fn cudaCtx(self: SdVae, be: *cuda.Backend, cancel: ?*std.atomic.Value(bool)) SdCudaTile {
        return .{ .vae = self.vae, .be = be, .cancel = cancel };
    }
    fn vkCtx(self: SdVae, gc: *gpu_mod.Context, cancel: ?*std.atomic.Value(bool)) SdVkTile {
        return .{ .vae = self.vae, .gc = gc, .cancel = cancel };
    }
    fn cpuCtx(self: SdVae, io: std.Io, cancel: ?*std.atomic.Value(bool)) SdCpuTile {
        return .{ .vae = self.vae, .io = io, .cancel = cancel };
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
/// Why the ladder is stepping down, for the progress log. ⚠️ Not cosmetic: every
/// rung used to say "OOM", which is a wrong diagnosis for a decode that came back
/// non-finite and would send the next reader hunting for a memory problem that is
/// not there.
fn decodeStepReason(err: anyerror) []const u8 {
    return switch (err) {
        error.GpuDecodeNonFinite => "produced non-finite values",
        else => "ran out of VRAM",
    };
}

fn recoverableDecodeErr(err: anyerror) bool {
    return switch (err) {
        error.DeviceOutOfMemory,
        error.OutOfMemory,
        error.CudaError,
        error.CublasLtError,
        error.CudnnError,
        // A device decode whose output came back non-finite (`sd_vae_gpu` checks and
        // reports rather than returning a buffer `planarF32ToRgb8` would clamp to
        // white). The CPU tier of the ladder is exact, so this must reach it.
        error.GpuDecodeNonFinite,
        => true,
        else => false,
    };
}

/// One image's denoiser: everything a DiT forward needs that is built **once per
/// image** rather than once per step — the text fusion, rope table and timestep
/// vectors uploaded to the device, the activation workspace, and (when CFG is on)
/// the second conditioning's session plus a scratch velocity.
///
/// This exists because `predict` cannot be a free function on the GPU backends
/// without rebuilding all of that per step, which would be both slow and a
/// behaviour change. A caller running a normal sampling loop makes one of these
/// and calls `predict` per step, exactly as `generate` does; a caller that wants a
/// single forward (level-2 style: one latent, one sigma, one conditioning) can use
/// `Session.predict`, which wraps init/predict/deinit.
///
/// Borrows its conditionings — they must outlive the `Denoiser`. Bound to one
/// resolution. `sigmas` is a cache, not a constraint: the Vulkan session
/// precomputes a timestep vector per entry, and `predict` at a sigma that is not
/// in the list recomputes one on the fly (correct, just slower) — so pass the
/// schedule you intend to sample with, and an off-schedule probe still works.
/// One conditioning entry's device state for the SD family, per branch.
///
/// `adm` is SDXL's `y` vector, built here rather than in `encode` because it carries the
/// image SIZE — a property of this denoiser's resolution, not of the prompt. It is
/// per-entry because it embeds that entry's pooled CLIP-G vector, which a scheduled
/// prompt changes along with everything else.
pub const SdBranch = struct {
    cu: ?sd_unet_cuda.Session = null,
    vk: ?sd_unet_gpu.Session = null,
    adm: ?[]f32 = null,
};

pub const Denoiser = struct {
    sess: *Session,
    lat_h: usize,
    lat_w: usize,
    /// 1.0 = no classifier-free guidance (single forward per step).
    cfg: f32,
    cond_pos: Cond,
    cond_neg: ?Cond,
    /// Velocity scratch for the negative pass; allocated only under CFG.
    v_neg: ?[]f32 = null,

    vk_pos: ?dit_gpu.Session = null,
    vk_neg: ?dit_gpu.Session = null,
    vk_ws: ?dit_gpu.Workspace = null,
    cu_pos: ?dit_cuda.Session = null,
    cu_neg: ?dit_cuda.Session = null,
    cu_ws: ?dit_cuda.Workspace = null,
    /// SD1.5's per-resolution UNet scratch, for the same reason the GPU sessions
    /// above exist: sized once per image rather than once per step.
    sd_ws: ?sd_unet.Workspace = null,
    /// The SD family's per-conditioning device state, one array element per distinct
    /// scheduled prompt (`Cond.entryCount`). Length 1 for every static prompt.
    ///
    /// One session per (entry, branch) rather than one per branch: a session carries that
    /// conditioning's context on the device plus its folded per-forward ResBlock biases,
    /// so a prompt that changes mid-render needs a session per variant. They are built
    /// eagerly in `denoiser` — deduplication at encode time keeps the count at the number
    /// of distinct prompt *texts* (2 for an `[a|b]`), not the number of steps.
    sd_pos: []SdBranch = &.{},
    sd_neg: []SdBranch = &.{},
    /// Shared across every entry and both branches, sized for the widest conditioning.
    sd_vk_ws: ?sd_unet_gpu.Workspace = null,
    sd_cu_ws: ?sd_unet_cuda.Workspace = null,
    /// Input scratch for SD's `x / sqrt(sigma^2 + 1)` pre-scaling, which krea2 has
    /// no analogue of. Channel-last, since it is what the UNet reads.
    sd_scaled: ?[]f32 = null,
    /// Channel-last scratch for the UNet's output, transposed back to the sampler's
    /// planar layout by `channelLastToPlanar`.
    sd_eps: ?[]f32 = null,

    /// Z-Image's caption half — `cap_embedder` + pad + both `context_refiner`
    /// blocks — one per branch.
    ///
    /// ⚠️ Cached because it is genuinely constant across steps, and that is a
    /// property of the architecture rather than an optimization guess: the
    /// `context_refiner` blocks are built with `modulation=False`, so the text half
    /// never sees the timestep. Recomputing it per step would add two full attention
    /// blocks over the caption to every step for an identical result.
    zi_cap_pos: ?[]f32 = null,
    zi_cap_neg: ?[]f32 = null,
    /// Padded caption length, i.e. how many rows of the joint sequence the text half
    /// occupies. The image half starts here.
    zi_cap_padded: usize = 0,
    /// Z-Image's Vulkan state, when the device can run its GEMMs. Null means the
    /// trunk runs on the CPU — which is also the case on every CUDA backend, which
    /// has no Z-Image forward at all yet.
    zi_vk: ?zimage_gpu.Session = null,
    zi_vk_neg: ?zimage_gpu.Session = null,
    zi_vk_ws: ?zimage_gpu.Workspace = null,
    zi_cu: ?zimage_cuda.Session = null,
    zi_cu_neg: ?zimage_cuda.Session = null,
    zi_cu_ws: ?zimage_cuda.Workspace = null,

    pub fn deinit(self: *Denoiser, gpa: std.mem.Allocator) void {
        if (self.v_neg) |b| gpa.free(b);
        if (self.sd_ws) |*w| w.deinit();
        for ([_][]SdBranch{ self.sd_pos, self.sd_neg }) |arr| {
            for (arr) |*br| {
                if (br.vk) |*x| x.deinit(self.sess.gpu_ctx.?);
                if (br.cu) |*x| x.deinit(self.sess.cu_be.?);
                if (br.adm) |b| gpa.free(b);
            }
            gpa.free(arr);
        }
        if (self.sd_vk_ws) |*w| w.deinit(self.sess.gpu_ctx.?);
        if (self.sd_cu_ws) |*w| w.deinit(self.sess.cu_be.?);
        if (self.sd_scaled) |b| gpa.free(b);
        if (self.sd_eps) |b| gpa.free(b);
        if (self.zi_cap_pos) |b| gpa.free(b);
        if (self.zi_cap_neg) |b| gpa.free(b);
        if (self.zi_vk) |*x| x.deinit(gpa, self.sess.gpu_ctx.?);
        if (self.zi_vk_neg) |*x| x.deinit(gpa, self.sess.gpu_ctx.?);
        if (self.zi_vk_ws) |*w| w.deinit(self.sess.gpu_ctx.?);
        if (self.zi_cu) |*x| x.deinit(gpa, self.sess.cu_be.?);
        if (self.zi_cu_neg) |*x| x.deinit(gpa, self.sess.cu_be.?);
        if (self.zi_cu_ws) |*w| w.deinit(self.sess.cu_be.?);
        if (self.vk_pos) |*s| s.deinit(gpa, self.sess.gpu_ctx.?);
        if (self.vk_neg) |*s| s.deinit(gpa, self.sess.gpu_ctx.?);
        if (self.vk_ws) |*w| w.deinit(self.sess.gpu_ctx.?);
        if (self.cu_pos) |*s| s.deinit(self.sess.cu_be.?);
        if (self.cu_neg) |*s| s.deinit(self.sess.cu_be.?);
        if (self.cu_ws) |*w| w.deinit(self.sess.cu_be.?);
        self.* = undefined;
    }

    /// One denoiser forward at `sigma`, using the STEP-0 conditioning.
    ///
    /// Unchanged in signature and behaviour: for a static prompt — every prompt in the
    /// ComfyUI dialect — there is only one conditioning, so this is the whole story. A
    /// caller driving the stages itself (ggufy's measurement ladder, the composed-stages
    /// invariant test) keeps working verbatim.
    pub fn predict(
        self: *Denoiser,
        gpa: std.mem.Allocator,
        v_out: []f32,
        latent: []const f32,
        sigma: f32,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        return self.predictAt(gpa, v_out, latent, sigma, 0, cancel);
    }

    /// One denoiser forward at `sigma` on the conditioning scheduled for `step` (0-based):
    /// `v_out = model(latent, sigma, cond[step])`, with classifier-free guidance folded in
    /// when `cfg != 1`. `v_out` and `latent` are both `channels·lat_h·lat_w` long.
    ///
    /// `step` selects among A1111's per-step prompt variants and is ignored for a static
    /// prompt. It is the *sampling* step index, not a sigma — the schedule is defined over
    /// step ordinals, so a teacher-forced off-schedule sigma still needs to say which step
    /// it stands for.
    pub fn predictAt(
        self: *Denoiser,
        gpa: std.mem.Allocator,
        v_out: []f32,
        latent: []const f32,
        sigma: f32,
        step: usize,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const s = self.sess;
        const io = s.io;
        if (s.family().isSd()) return self.predictSd(gpa, v_out, latent, sigma, step, cancel);
        if (s.family() == .zimage) return self.predictZImage(gpa, v_out, latent, sigma, cancel);
        const dit = &s.models.krea2.dit;
        std.debug.assert(v_out.len == wan_vae.latent_channels * self.lat_h * self.lat_w);
        std.debug.assert(latent.len == v_out.len);

        if (s.cu_be) |b| {
            try dit_cuda.forward(dit, b, &self.cu_pos.?, &self.cu_ws.?, io, gpa, v_out, latent, sigma, cancel);
        } else if (s.gpu_ctx) |gc| {
            try dit_gpu.forward(dit, gc, &self.vk_pos.?, &self.vk_ws.?, io, gpa, v_out, latent, sigma, cancel);
        } else {
            try dit.forward(io, gpa, v_out, latent, self.lat_h, self.lat_w, sigma, self.cond_pos.data, self.cond_pos.seq, cancel);
        }
        if (self.cfg == 1.0) return;

        const v_neg = self.v_neg.?;
        if (s.cu_be) |b| {
            try dit_cuda.forward(dit, b, &self.cu_neg.?, &self.cu_ws.?, io, gpa, v_neg, latent, sigma, cancel);
        } else if (s.gpu_ctx) |gc| {
            try dit_gpu.forward(dit, gc, &self.vk_neg.?, &self.vk_ws.?, io, gpa, v_neg, latent, sigma, cancel);
        } else {
            try dit.forward(io, gpa, v_neg, latent, self.lat_h, self.lat_w, sigma, self.cond_neg.?.data, self.cond_neg.?.seq, cancel);
        }
        sampler.applyCfg(v_out, v_neg, self.cfg);
    }

    /// Z-Image's forward: the timestep vector, then the image half and the joint
    /// trunk on top of the cached caption half.
    ///
    /// No input pre-scaling and no timestep inversion — it is flow matching, like
    /// krea2, so the sampler's sigma reaches the model directly (`NextDiT` turns it
    /// into `1 - sigma` itself). The output is the trajectory derivative, which is
    /// what makes CFG mixing valid here for the same reason it is for krea2.
    fn predictZImage(
        self: *Denoiser,
        gpa: std.mem.Allocator,
        v_out: []f32,
        latent: []const f32,
        sigma: f32,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const s = self.sess;
        const dit = &s.models.zimage.dit;
        std.debug.assert(v_out.len == zimage.latent_channels * self.lat_h * self.lat_w);
        std.debug.assert(latent.len == v_out.len);

        if (self.zi_cu) |*cu| {
            const b = s.cu_be.?;
            try zimage_cuda.forward(dit, b, cu, &self.zi_cu_ws.?, s.io, gpa, v_out, latent, sigma, cancel);
            if (self.cfg == 1.0) return;
            const v_neg = self.v_neg.?;
            try zimage_cuda.forward(dit, b, &self.zi_cu_neg.?, &self.zi_cu_ws.?, s.io, gpa, v_neg, latent, sigma, cancel);
            sampler.applyCfg(v_out, v_neg, self.cfg);
            return;
        }
        if (self.zi_vk) |*vk| {
            const gc = s.gpu_ctx.?;
            try zimage_gpu.forward(dit, gc, vk, &self.zi_vk_ws.?, s.io, gpa, v_out, latent, sigma, cancel);
            if (self.cfg == 1.0) return;
            const v_neg = self.v_neg.?;
            try zimage_gpu.forward(dit, gc, &self.zi_vk_neg.?, &self.zi_vk_ws.?, s.io, gpa, v_neg, latent, sigma, cancel);
            sampler.applyCfg(v_out, v_neg, self.cfg);
            return;
        }

        const adaln = try dit.adalnInput(s.io, gpa, sigma);
        defer gpa.free(adaln);
        try dit.predict(s.io, gpa, v_out, latent, self.lat_h, self.lat_w, self.zi_cap_pos.?, self.zi_cap_padded, adaln, cancel);
        if (self.cfg == 1.0) return;

        const v_neg = self.v_neg.?;
        try dit.predict(s.io, gpa, v_neg, latent, self.lat_h, self.lat_w, self.zi_cap_neg.?, self.zi_cap_padded, adaln, cancel);
        sampler.applyCfg(v_out, v_neg, self.cfg);
    }

    /// SD1.5's forward: pre-scale the latent, invert sigma back to the timestep the
    /// UNet is conditioned on, run it, and mix the two branches under CFG.
    ///
    /// ⚠️ **Both of the first two steps are silent if omitted.** Feeding the raw
    /// latent runs the model off its training distribution (softer, washed images,
    /// no error), and conditioning on the wrong timestep runs the right model at the
    /// wrong noise level. Neither shows up as anything but "the sampler seems bad".
    fn predictSd(
        self: *Denoiser,
        gpa: std.mem.Allocator,
        v_out: []f32,
        latent: []const f32,
        sigma: f32,
        step: usize,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const s = self.sess;
        const m = s.sd().?;
        const cfg_unet = m.unet.cfg;
        std.debug.assert(v_out.len == cfg_unet.channels * self.lat_h * self.lat_w);
        std.debug.assert(latent.len == v_out.len);

        // ⚠️ **LAYOUT.** The sampler's latent is **planar** `[c][h][w]` — krea2's DiT and
        // both VAEs use that, and `generate`/`decode` are written to it — while
        // `sd_unet.forward` works in **channel-last** `[h*w][c]`, the layout `ops.conv`
        // and the GEMMs want. So this transposes on the way in and back on the way out.
        //
        // Getting this wrong is *rms-preserving*: every value survives, only its
        // position changes, so the per-step magnitudes stay right to a fraction of a
        // percent while the image becomes a periodic streak pattern. It cost a long
        // bisection, and two things made it worse: the UNet parity tests transpose
        // explicitly before calling `forward`, so they never exercised the pipeline's
        // convention, and a dump-and-compare against the reference *interpreted the
        // dumped buffer the same wrong way*, so both sides scrambled identically and
        // agreed to 1e-5.
        const scaled = self.sd_scaled.?;
        const plane = self.lat_h * self.lat_w;
        const ch = cfg_unet.channels;
        for (0..plane) |px| {
            for (0..ch) |c| {
                scaled[px * ch + c] = latent[c * plane + px];
            }
        }
        sampler.sdScaleInput(scaled, scaled, sigma);
        // The nearest *trained* index, not the fractional one — see `sdModelTimestep`,
        // where the measured cost of the fractional form is 25 dB of render agreement.
        // A1111 is the one ecosystem that passes the fractional index by default
        // (`enable_quantization = False`), so `--compat a1111` takes the other branch.
        const t = if (self.sess.compat.quantize_timestep)
            sampler.sdModelTimestep(m.sigma_ladder, sigma)
        else
            sampler.sdTimestepForSigma(m.sigma_ladder, sigma);

        // Which scheduled prompt this step conditions on. Both branches are indexed
        // independently: A1111 schedules the negative prompt too, and the two need not
        // change at the same steps.
        const e_pos = if (self.cond_pos.sched) |sc| sc.indexAt(step) else 0;
        const cl = self.sd_eps.?; // channel-last scratch for the UNet's output
        try self.unetForward(gpa, cl, scaled, t, .positive, e_pos, cancel);
        channelLastToPlanar(v_out, cl, ch, plane);
        if (self.cfg == 1.0) return;

        const v_neg = self.v_neg.?;
        const e_neg = if (self.cond_neg.?.sched) |sc| sc.indexAt(step) else 0;
        try self.unetForward(gpa, cl, scaled, t, .negative, e_neg, cancel);
        channelLastToPlanar(v_neg, cl, ch, plane);
        // Mixing eps is equivalent to mixing denoised predictions at fixed x, the
        // same argument that licenses krea2's velocity CFG.
        sampler.applyCfg(v_out, v_neg, self.cfg);
    }

    /// One SD UNet forward for one conditioning branch, dispatched by backend.
    /// Both arms take and return channel-last `[h*w][c]`.
    fn unetForward(
        self: *Denoiser,
        gpa: std.mem.Allocator,
        eps: []f32,
        scaled: []const f32,
        t: f32,
        branch: enum { positive, negative },
        entry: usize,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const s = self.sess;
        const m = s.sd().?;
        const br = &(switch (branch) {
            .positive => self.sd_pos,
            .negative => self.sd_neg,
        })[entry];
        if (s.cu_be) |b| {
            return sd_unet_cuda.forward(&m.unet, b, &br.cu.?, &self.sd_cu_ws.?, s.io, gpa, eps, scaled, self.lat_h, self.lat_w, t, cancel);
        }
        if (s.gpu_ctx) |gc| {
            return sd_unet_gpu.forward(&m.unet, gc, &br.vk.?, &self.sd_vk_ws.?, s.io, gpa, eps, scaled, self.lat_h, self.lat_w, t, cancel);
        }
        // The CPU UNet's cancel hook rides on ops.cancel, like the DiT's.
        const cond = (switch (branch) {
            .positive => &self.cond_pos,
            .negative => &self.cond_neg.?,
        }).entry(entry);
        return sd_unet.forward(&m.unet, s.io, gpa, &self.sd_ws.?, eps, scaled, self.lat_h, self.lat_w, t, cond.data, cond.seq, br.adm);
    }
};

/// The distinct prompt texts a scheduled prompt passes through, plus the per-step index
/// into them. `texts` lives in `arena`; `at` is allocated from `gpa` because it outlives
/// the call as part of `Cond.Schedule`.
///
/// Split out from `encodeScheduled` so the deduplication is testable without a model —
/// it is the part with the interesting behaviour, and the part whose absence would cost
/// 33 tower forwards on a 35-step `[a|b]`.
pub const SchedulePlan = struct { texts: []const []const u8, at: []u8 };

pub fn planSchedule(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    text: []const u8,
    steps: usize,
) !SchedulePlan {
    const entries = try prompt_a1111.schedule(arena, text, steps);
    var uniq: std.ArrayList([]const u8) = .empty;
    const at = try gpa.alloc(u8, steps);
    errdefer gpa.free(at);
    for (0..steps) |step| {
        // The first entry whose (1-based, inclusive) `until` covers this step.
        const et = for (entries) |e| {
            if (step + 1 <= e.until) break e.text;
        } else entries[entries.len - 1].text;
        const idx = for (uniq.items, 0..) |u, i| {
            if (std.mem.eql(u8, u, et)) break i;
        } else blk: {
            try uniq.append(arena, et);
            break :blk uniq.items.len - 1;
        };
        // 255 distinct prompts in one render is not a thing anyone means to do; refusing
        // beats silently truncating the index.
        if (idx > std.math.maxInt(u8)) return error.TooManyPromptVariants;
        at[step] = @intCast(idx);
    }
    return .{ .texts = uniq.items, .at = at };
}

/// A checkpoint container, which may be either format — used for every component
/// that can arrive as its own file (denoiser, text encoder, VAE).
///
/// ggufy emits both — int4/int8 cluster formats as safetensors, the ggml block
/// quants as GGUF — and until this existed only the safetensors half could be
/// loaded, so the GGUF half of the quantization work could not be run or measured
/// at all. `dit.DiT.load` takes a `WeightStore`, so the only thing missing was
/// opening the file.
/// `Context.WeightReader` over an optional side-file container — the encoder/VAE
/// twin of `Container.weightReader`, which it now simply delegates to. See
/// `SafeTensors.readTo`.
fn storeReader(opt: *?Container) ?cuda.Context.WeightReader {
    const c = if (opt.*) |*v| v else return null;
    return c.weightReader();
}

pub const Container = union(enum) {
    safetensors: safetensors.SafeTensors,
    gguf: gguf_mod.Gguf,

    /// A `Context.WeightReader` over this container, or null when it cannot serve
    /// one. Lets the CUDA staging path pull weight bytes with positional reads
    /// instead of faulting the (multi-GB, read-once) checkpoint mapping — see
    /// `SafeTensors.readTo` for why that matters under host memory pressure.
    ///
    /// GGUF returns null for now: same idea applies, but the DiT GGUF path is
    /// CPU-only today (no GPU block-quant GEMM), so it never reaches this staging
    /// ring and wiring it would be untestable here.
    pub fn weightReader(self: *Container) ?cuda.Context.WeightReader {
        return switch (self.*) {
            .safetensors => |*st| .{
                .ctx = st,
                .read = struct {
                    fn read(ctx: *anyopaque, dst: []u8, src: []const u8) bool {
                        const s2: *const safetensors.SafeTensors = @ptrCast(@alignCast(ctx));
                        return s2.readTo(dst, src);
                    }
                }.read,
            },
            .gguf => null,
        };
    }

    /// Open by **content**, not extension: a leading `GGUF` magic picks the GGUF
    /// reader, anything else the safetensors one. Sniffing rather than trusting the
    /// name because the failure mode of guessing wrong is a baffling error from the
    /// other parser — handing a GGUF to the safetensors reader reports
    /// `InvalidHeader`, which says nothing about what actually happened.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Container {
        return openIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Container {
        var magic: [4]u8 = undefined;
        {
            const f = try dir.openFile(io, path, .{ .mode = .read_only });
            defer f.close(io);
            const n = try f.readPositionalAll(io, &magic, 0);
            if (n < magic.len) return error.CheckpointTooSmall;
        }
        if (std.mem.eql(u8, &magic, "GGUF")) {
            return .{ .gguf = try gguf_mod.Gguf.openIn(gpa, io, dir, path) };
        }
        return .{ .safetensors = try safetensors.SafeTensors.openIn(gpa, io, dir, path) };
    }

    pub fn deinit(self: *Container) void {
        switch (self.*) {
            .safetensors => |*st| st.deinit(),
            .gguf => |*g| g.deinit(),
        }
    }

    /// A store view. Takes a pointer because the containers hold interior
    /// pointers into their own mappings; the `Session` keeps this at a stable
    /// address for exactly that reason.
    pub fn store(self: *const Container) weights_mod.WeightStore {
        return switch (self.*) {
            .safetensors => |*st| .{ .safetensors = st },
            .gguf => |*g| .{ .gguf = g },
        };
    }

    /// Tensor-data bytes — what the VRAM pinning policy sizes itself against.
    pub fn payloadLen(self: *const Container) usize {
        return switch (self.*) {
            .safetensors => |*st| st.payload.len,
            .gguf => |*g| g.payload.len,
        };
    }
};

/// A reusable text-to-image pipeline. `init` creates the backend and loads the
/// text encoder, DiT, and VAE ONCE (their safetensors mappings stay open for the
/// session lifetime), so `generate` can produce many images without reloading —
/// the DiT stays resident in the backend's weight cache across a queue when the
/// VRAM budget allows. The GUI keeps one alive while its image queue is
/// non-empty; the one-shot `generate` free function below wraps init+generate+
/// deinit for the CLI / tests. NOT thread-safe: serialize `generate` calls.
/// Open `path` only when the caller actually gave one: an empty path means "not
/// specified", and the component is then expected in the primary checkpoint.
/// Open a side-file component if one was given. ⚠️ Goes through `Container`, so a
/// side file is opened by **magic** and may be a GGUF — a `.gguf` text encoder used
/// to die with `InvalidHeader` from the safetensors parser, an error that says
/// nothing about what happened (the same trap `Container.open` exists to avoid for
/// the denoiser).
fn openIfGiven(gpa: std.mem.Allocator, io: std.Io, path: []const u8, explicit: bool) !?Container {
    if (path.len == 0) return null;
    if (explicit) return try Container.open(gpa, io, path);
    // A *defaulted* path may simply not exist (a box with only SD checkpoints has no
    // krea2 VAE), and that is not an error as long as the primary checkpoint carries
    // the component. An explicit path that cannot be opened still fails loudly.
    return Container.open(gpa, io, path) catch null;
}

fn storeOf(st: *?Container) ?weights_mod.WeightStore {
    if (st.*) |*s| return s.store();
    return null;
}

/// Channel-last `[h*w][c]` -> planar `[c][h][w]`, the sampler's layout. See the layout
/// note in `Denoiser.predictSd`.
fn channelLastToPlanar(dst: []f32, src: []const f32, ch: usize, plane: usize) void {
    std.debug.assert(dst.len == ch * plane and src.len == dst.len);
    for (0..plane) |px| {
        for (0..ch) |c| dst[c * plane + px] = src[px * ch + c];
    }
}

/// One of the three things a diffusion pipeline is made of. Each may live in the
/// primary checkpoint or in a file of its own — **independently of the architecture**.
/// `conditioner2` exists only for SDXL, which conditions on **two** text towers whose
/// outputs are concatenated. Asking any other family for it is a programming error
/// (`error.NoSuchComponent`), not a missing-weights condition.
pub const Component = enum { denoiser, conditioner, conditioner2, decoder };

/// Where a component's weights were found: a store in which it sits **at the root**
/// (wrapped in a `weights.Prefixed` view when it was nested), plus the file it came
/// from, for the load log.
const Resolved = struct {
    store: weights_mod.WeightStore,
    /// Non-null when a `Prefixed` view had to be created; owned by the Session and
    /// freed in `deinit`.
    view: ?*weights_mod.Prefixed = null,
    from_primary: bool,
};

/// Prefix spellings a component is found under, and tensors that prove it. Both lists
/// are ordered most-specific first, so a bundled checkpoint's nested spelling wins over
/// a bare one that could also match by accident.
///
/// ⚠️ `probes` is a LIST because a component's own tensor names are not the same in
/// every container. A GGUF text encoder is `embed_tokens.weight` where safetensors is
/// `model.embed_tokens.weight` — `canonicalName` strips llama.cpp's `blk.N.` spelling
/// but there is no `model.` to restore, and with one probe a `.gguf` encoder resolved
/// to nothing and reported `ComponentNotInCheckpoint`.
pub const ComponentSpec = struct { prefixes: []const []const u8, probes: []const []const u8 };

pub fn componentSpec(fam: Family, comp: Component) error{NoSuchComponent}!ComponentSpec {
    return switch (fam) {
        .krea2 => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probes = &.{"blocks.0.attn.wq.weight"} },
            .conditioner => .{ .prefixes = &.{ "text_encoders.", "" }, .probes = &.{ "model.language_model.embed_tokens.weight", "embed_tokens.weight" } },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "vae.", "" }, .probes = &.{"decoder.conv1.weight"} },
            .conditioner2 => error.NoSuchComponent,
        },
        .sd15 => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probes = &.{"input_blocks.0.0.weight"} },
            .conditioner => .{
                // LDM bundle, then a bare HF text-encoder export.
                .prefixes = &.{ "cond_stage_model.transformer.text_model.", "text_model.", "" },
                .probes = &.{"final_layer_norm.weight"},
            },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "" }, .probes = &.{"decoder.conv_in.weight"} },
            .conditioner2 => error.NoSuchComponent,
        },
        // Z-Image ships as three separate files in the official layout, but the
        // resolver is the same: each component is looked for in the primary
        // checkpoint first and in a side file second.
        //
        // ⚠️ The conditioner probe is `model.embed_tokens.weight`, NOT krea2's
        // `model.language_model.embed_tokens.weight`: krea2 uses the Qwen3-**VL**
        // checkpoint, whose language model is nested a level deeper. Feeding either
        // encoder to the other family resolves to nothing and reports it, which is
        // the point of probing rather than assuming.
        .zimage => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probes = &.{"cap_embedder.1.weight"} },
            .conditioner => .{ .prefixes = &.{ "text_encoders.", "" }, .probes = &.{ "model.embed_tokens.weight", "embed_tokens.weight" } },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "vae.", "" }, .probes = &.{"decoder.conv_in.weight"} },
            .conditioner2 => error.NoSuchComponent,
        },
        // SDXL bundles its two towers under `conditioner.embedders.{0,1}` — and the two
        // are spelled differently *within one file*: embedder 0 is a `transformers`
        // CLIPTextModel, embedder 1 an OpenCLIP tower. Hence the different probes.
        .sdxl => switch (comp) {
            .denoiser => .{ .prefixes = &.{ "model.diffusion_model.", "" }, .probes = &.{"input_blocks.0.0.weight"} },
            .conditioner => .{
                .prefixes = &.{ "conditioner.embedders.0.transformer.text_model.", "text_model.", "" },
                .probes = &.{"final_layer_norm.weight"},
            },
            .conditioner2 => .{
                .prefixes = &.{ "conditioner.embedders.1.model.", "" },
                .probes = &.{"ln_final.weight"},
            },
            .decoder => .{ .prefixes = &.{ "first_stage_model.", "" }, .probes = &.{"decoder.conv_in.weight"} },
        },
    };
}

/// Resolve where a component's weights are, with the precedence the CLI promises:
///
/// 1. **An explicitly given path wins.** If the user passed `--text-encoder` or
///    `--vae`, that file is used even when the primary checkpoint also carries one —
///    overriding a bundled component is the whole reason to pass the flag.
/// 2. Otherwise the primary checkpoint must contain it.
///
/// Either way the component may sit under a prefix, and either way the loader is
/// handed a store in which it sits at the root (see `weights.Prefixed`), so no loader
/// needs to know which spelling it came from.
fn resolveComponent(
    gpa: std.mem.Allocator,
    fam: Family,
    comp: Component,
    primary: weights_mod.WeightStore,
    side: ?weights_mod.WeightStore,
    side_is_explicit: bool,
) !Resolved {
    const spec = try componentSpec(fam, comp);

    // Order of preference: an explicitly requested file, then the primary checkpoint's
    // own copy, then a defaulted side file. So `--vae x` overrides a bundled VAE, a
    // bundled VAE beats the built-in default path, and nothing silently wins over an
    // explicit request.
    var candidates: [3]?weights_mod.WeightStore = .{ null, null, null };
    var n: usize = 0;
    if (side_is_explicit) {
        if (side) |st| {
            candidates[n] = st;
            n += 1;
        }
    }
    candidates[n] = primary;
    n += 1;
    if (!side_is_explicit) {
        if (side) |st| {
            candidates[n] = st;
            n += 1;
        }
    }

    for (candidates[0..n]) |maybe| {
        const store = maybe.?;
        for (spec.prefixes) |pfx| {
            var buf: [256]u8 = undefined;
            var hit = false;
            for (spec.probes) |probe| {
                if (pfx.len + probe.len > buf.len) continue;
                @memcpy(buf[0..pfx.len], pfx);
                @memcpy(buf[pfx.len..][0..probe.len], probe);
                if (store.get(buf[0 .. pfx.len + probe.len]) != null) {
                    hit = true;
                    break;
                }
            }
            if (!hit) continue;

            if (pfx.len == 0) return .{ .store = store, .from_primary = false };
            const view = try gpa.create(weights_mod.Prefixed);
            errdefer gpa.destroy(view);
            view.* = try weights_mod.Prefixed.init(gpa, store, pfx);
            return .{ .store = view.store(), .view = view, .from_primary = false };
        }
    }
    // Silent: a resolver that both returns a typed error and logs makes every caller's
    // own diagnostic a duplicate, and makes a test of the error path noisy (this suite
    // treats stderr from a passing test as a failure). `reportResolve` is the reporting
    // wrapper the loading paths use.
    return error.ComponentNotInCheckpoint;
}

/// `resolveComponent` plus the diagnostic. The load paths use this; tests of the
/// resolution rules use the silent form.
fn reportResolve(
    gpa: std.mem.Allocator,
    fam: Family,
    comp: Component,
    primary: weights_mod.WeightStore,
    side: ?weights_mod.WeightStore,
    side_is_explicit: bool,
    side_path: []const u8,
) !Resolved {
    return resolveComponent(gpa, fam, comp, primary, side, side_is_explicit) catch |err| {
        if (err == error.ComponentNotInCheckpoint) std.log.err(
            "{t} not found: the checkpoint has no '{s}' under any known prefix, and '{s}' does not supply one either",
            .{ comp, (try componentSpec(fam, comp)).probes[0], side_path },
        );
        return err;
    };
}

/// Which architecture a session is running.
///
/// The engine started as a krea2 pipeline and the stage API (`encode` → `denoiser` →
/// `predict` → `decode`) turned out to be the right shape for both families, so the
/// generalization is a tagged union *inside* `Session` rather than a second pipeline:
/// callers — including ggufy's whole measurement ladder — are unchanged.
///
/// ⚠️ **The two families differ in what a "step" means**, and `sampler.zig` documents
/// it: krea2 predicts a velocity on a continuous flow-matching schedule, SD predicts
/// eps on a discrete ladder, is conditioned on a timestep *index*, and needs its input
/// pre-scaled. What makes one API cover both is that the Euler step consumes the same
/// quantity either way — for eps-prediction the trajectory derivative *is* eps.
pub const Family = enum {
    krea2,
    sd15,
    sdxl,
    /// Z-Image (`NextDiT`), the architecture "zit" checkpoints use. Flow matching
    /// like krea2, and it shares krea2's Qwen3-4B text encoder body — but a
    /// different tap, a different RoPE theta, a 16-channel `AutoencoderKL` rather
    /// than the Wan VAE, and its own 1000-rung sigma table. See `models/zimage.zig`.
    zimage,

    /// Whether this family runs the `sd_unet` / `sd_vae` / CLIP stack, i.e. whether
    /// `Session.sd()` returns a model set. SD1.5 and SDXL differ only in configuration
    /// and in SDXL's second text tower, so nearly every stage is shared.
    pub fn isSd(self: Family) bool {
        return self == .sd15 or self == .sdxl;
    }
};

/// krea2: three separate checkpoints (encoder, DiT, VAE).
pub const Krea2Models = struct {
    tok: tokenizer_mod.Tokenizer,
    /// Null when the conditioner came out of the primary checkpoint.
    enc_st: ?Container,
    enc: qwen3.TextEncoder,
    dit_st: Container,
    dit: dit_mod.DiT,
    /// Null when the decoder came out of the primary checkpoint.
    vae_st: ?Container,
    vae: wan_vae.Decoder,
    /// Prefix views, when a component was nested inside its container.
    enc_view: ?*weights_mod.Prefixed,
    vae_view: ?*weights_mod.Prefixed,
};

/// The SD family (SD1.5 and SDXL): one LDM single-file checkpoint normally holds every
/// model — but the denoiser may come from a *different* file than the conditioner and
/// decoder, which is the case that matters here: ggufy writes a GGUF containing only the
/// UNet (with the container prefix stripped), so a quantized arm is measured with CLIP
/// and the VAE loaded from the original checkpoint.
///
/// One struct for both architectures, because they differ only in `Config`s and in
/// SDXL's second text tower — so `denoiser`, `predict`, `decode` and teardown are shared
/// verbatim rather than duplicated per family.
pub const SdModels = struct {
    tok: clip_tok.Tokenizer,
    unet_st: Container,
    /// Null when that component came out of the primary checkpoint.
    enc_st: ?Container,
    enc2_st: ?Container,
    vae_st: ?Container,
    clip: clip_text.TextEncoder,
    /// SDXL's OpenCLIP bigG tower. Null for SD1.5.
    clip_g: ?clip_text.TextEncoder,
    unet: sd_unet.UNet,
    vae: sd_vae.Decoder,
    clip_view: ?*weights_mod.Prefixed,
    clip_g_view: ?*weights_mod.Prefixed,
    vae_view: ?*weights_mod.Prefixed,
    /// The full per-training-step sigma ladder, for the sigma → timestep inverse a
    /// teacher-forced (off-schedule) sigma needs. Built once.
    sigma_ladder: []f32,
    /// `z_empty` for each tower — the empty prompt's hidden rows at the same capture
    /// layer the conditioning is taken from, which is the reference point attention
    /// weighting interpolates away from (`clip_text.applyWeights`).
    ///
    /// Cached because it depends only on the tower, while `encode` is called at least
    /// twice per image (positive and negative) and a render often reuses a prompt. Filled
    /// lazily, so an unweighted prompt never pays for it at all.
    empty_ref: ?[]f32 = null,
    empty_ref_g: ?[]f32 = null,
};

/// Z-Image: a denoiser-only checkpoint plus Qwen3-4B and the 16-channel Flux VAE,
/// each normally in a file of its own — the shape the official ComfyUI template
/// distributes it in. Structurally krea2's set with two of the three models swapped.
pub const ZImageModels = struct {
    tok: tokenizer_mod.Tokenizer,
    /// Null when that component came out of the primary checkpoint.
    enc_st: ?Container,
    enc: qwen3.TextEncoder,
    dit_st: Container,
    dit: zimage.DiT,
    vae_st: ?Container,
    vae: sd_vae.Decoder,
    /// Prefix views, when a component was nested inside its container.
    enc_view: ?*weights_mod.Prefixed,
    vae_view: ?*weights_mod.Prefixed,
};

/// The loaded model set, tagged by family. A named type rather than an anonymous
/// union in the field: an inline `union(Family)` there makes the compiler derive its
/// name from the first field's type and then report a dependency loop.
pub const Models = union(Family) {
    krea2: Krea2Models,
    sd15: SdModels,
    sdxl: SdModels,
    zimage: ZImageModels,
};

/// Which family a denoiser checkpoint belongs to, from its tensor names alone.
///
/// Name-based, like ggufy's own `ImageArch` detection, because the alternative —
/// asking the caller — puts a mistypeable flag in front of a measurement whose whole
/// point is that the inputs are what they claim to be.
pub fn detectFamily(store: weights_mod.WeightStore) !Family {
    // ⚠️ **SDXL must be tested before SD1.5, and by a tensor SD1.5 does not have.** Both
    // are LDM UNets with the same `input_blocks.0.0` stem, so the stem cannot tell them
    // apart; `label_emb` (the micro-conditioning projection) exists only in SDXL. Reading
    // the stem first would load every SDXL checkpoint as SD1.5, which fails at the fourth
    // level's missing weights rather than saying what it actually is.
    if (store.get("model.diffusion_model.label_emb.0.0.weight") != null) return .sdxl;
    if (store.get("label_emb.0.0.weight") != null) return .sdxl;
    // SD's UNet stem, prefixed (a full single-file checkpoint) or bare (ggufy's
    // model-only GGUF, which strips the container prefix).
    if (store.get("model.diffusion_model.input_blocks.0.0.weight") != null) return .sd15;
    if (store.get("input_blocks.0.0.weight") != null) return .sd15;
    if (store.get("model.diffusion_model.blocks.0.attn.wq.weight") != null) return .krea2;
    if (store.get("blocks.0.attn.wq.weight") != null) return .krea2;
    // Z-Image. ⚠️ Both tensors, which is ComfyUI's own test
    // (`model_detection.py`): `cap_embedder.1.weight` alone is shared with stock
    // Lumina 2 and with OmniGen2, and the `noise_refiner` qk-norm is what says this
    // is the NextDiT shape rather than one of those. The width (3840) is then what
    // distinguishes Z-Image from Lumina 2, and `zimage.DiT.load` checks it.
    for ([_][]const u8{ "model.diffusion_model.", "" }) |pfx| {
        var b1: [96]u8 = undefined;
        var b2: [96]u8 = undefined;
        const cap = std.fmt.bufPrint(&b1, "{s}cap_embedder.1.weight", .{pfx}) catch continue;
        const ref = std.fmt.bufPrint(&b2, "{s}noise_refiner.0.attention.k_norm.weight", .{pfx}) catch continue;
        if (store.get(cap) != null and store.get(ref) != null) return .zimage;
    }
    return error.UnknownArchitecture;
}

/// The `sd_unet` / `sd_vae` configuration pair a family runs.
fn sdConfigs(fam: Family) struct { unet: sd_unet.Config, vae: sd_vae.Config, clip: clip_text.Config } {
    return switch (fam) {
        .sd15 => .{ .unet = sd_unet.sd15, .vae = sd_vae.sd15, .clip = clip_text.clip_l },
        .sdxl => .{ .unet = sd_unet.sdxl, .vae = sd_vae.sdxl, .clip = clip_text.clip_l },
        .krea2, .zimage => unreachable,
    };
}

/// **Whether this family's text encoder can apply per-token prompt weights at all.**
///
/// ⚠️ **Derived from the ENCODER's own declaration, deliberately not from a list of
/// architectures.** SD.Next's equivalent is a hardcoded allow-list matched against pipeline
/// *class names* (`'StableDiffusion' in cls or 'StableCascade' in cls or 'Flux' in cls`), and
/// the cost of that shape is visible in their own file: Chroma and HiDream sit **commented
/// out** of it, and Flux2 needs a `'Flux2' not in cls` special case to avoid being caught by
/// the `Flux` substring. A name-matched list has to be edited for every new pipeline and
/// silently gives the wrong answer until someone does.
///
/// Here the answer comes from `TextEncoder.supports_prompt_weights` on the encoder actually
/// loaded, which cannot disagree with the encoder's real capability and forces a *deliberate*
/// answer from any new one — omitting the declaration is a compile error, not a default. The
/// three properties that decide it are documented on `clip_text.TextEncoder`: a fixed token
/// window, rows the denoiser cross-attends to 1:1, and an empty-prompt encode of the same
/// shape to interpolate against.
/// Where a family's side components live when the caller did not name one.
///
/// ⚠️ `Options`' own `text_encoder_path` / `vae_path` defaults are **krea2's**, and
/// they predate there being more than one family. Handing them to another
/// architecture is exactly the failure the `explicit_*` flags exist to prevent — the
/// resolver would open krea2's Qwen3-**VL** encoder, look for Z-Image's
/// `model.embed_tokens.weight` in it and report `ComponentNotInCheckpoint`. So a
/// defaulted path is chosen per family here, and only an *explicit* one overrides it.
///
/// Both Z-Image paths are what the official ComfyUI template names.
pub fn defaultComponentPath(fam: Family, comp: Component) []const u8 {
    return switch (fam) {
        .zimage => switch (comp) {
            .conditioner => "models/text_encoders/qwen_3_4b.safetensors",
            .decoder => "models/vae/ae.safetensors",
            else => "",
        },
        else => "",
    };
}

/// The sigma-schedule shift a family was trained with, for a caller that did not ask
/// for one. The SD arm's value is unused — its ladder comes from the betas — and is
/// returned only so this is total.
pub fn defaultShift(fam: Family) f32 {
    return switch (fam) {
        .krea2 => sampler.default_shift,
        .zimage => sampler.schedule_mod.z_image_shift,
        .sd15, .sdxl => sampler.default_shift,
    };
}

pub fn supportsPromptWeights(fam: Family) bool {
    return switch (fam) {
        // Z-Image conditions on a Qwen3 hidden state exactly as krea2 does, so it
        // inherits the same answer from the same encoder type — and for the same
        // structural reason, not by analogy.
        .krea2, .zimage => qwen3.TextEncoder.supports_prompt_weights,
        .sd15, .sdxl => clip_text.TextEncoder.supports_prompt_weights,
    };
}

pub const Session = struct {
    models: Models,
    /// Which ecosystem's sampling conventions this session's forwards follow. Set from
    /// `Options` at `init` and refreshed at the top of `generate`, so a caller driving the
    /// public stages itself (ggufy's ladder, the GUI's per-image config) gets the same
    /// behaviour without threading it through four stage signatures.
    compat: CompatConfig = .{},


    gpa: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    gpu_ctx: ?*gpu_mod.Context,
    cu_be: ?*cuda.Backend,
    // Models + their (kept-open) checkpoint mappings. Held at stable addresses
    // inside this heap-allocated struct so the models' pointers into the mmaps
    // stay valid for the whole session. The DiT's container may be safetensors or
    // GGUF (`Container`); the encoder and VAE are safetensors-only.
    tok: tokenizer_mod.Tokenizer,
    enc_st: safetensors.SafeTensors,
    enc: qwen3.TextEncoder,
    dit_st: Container,
    dit: dit_mod.DiT,
    vae_st: safetensors.SafeTensors,
    vae: wan_vae.Decoder,

    /// The loaded family. Shorthand for `@as(Family, self.models)`.
    pub fn family(self: *const Session) Family {
        return self.models;
    }

    /// krea2's model set, asserting the family — every krea2-only path binds this
    /// once at the top and its body is then unchanged from before the split.
    fn k(self: *Session) *Krea2Models {
        return &self.models.krea2;
    }

    /// Z-Image's model set, asserting the family.
    fn zi(self: *Session) *ZImageModels {
        return &self.models.zimage;
    }

    /// The SD-family model set, or null for krea2. Both SD arms hold the same struct
    /// type, so every shared stage binds this once and needs no further family test.
    pub fn sd(self: *Session) ?*SdModels {
        return switch (self.models) {
            .krea2, .zimage => null,
            .sd15 => |*m| m,
            .sdxl => |*m| m,
        };
    }

    /// The store the *denoiser* was loaded from: what a caller overlays to
    /// substitute weights (ggufy's per-tensor attribution arm) without touching the
    /// conditioner or the decoder.
    pub fn denoiserStore(self: *const Session) weights_mod.WeightStore {
        return switch (self.models) {
            .krea2 => |*m| m.dit_st.store(),
            .zimage => |*m| m.dit_st.store(),
            .sd15, .sdxl => |*m| m.unet_st.store(),
        };
    }

    /// Payload bytes of the denoiser's container, for the VRAM meter.
    fn denoiserPayloadLen(self: *const Session) usize {
        return switch (self.models) {
            .krea2 => |*m| m.dit_st.payloadLen(),
            .zimage => |*m| m.dit_st.payloadLen(),
            .sd15, .sdxl => |*m| m.unet_st.payloadLen(),
        };
    }

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
        self.compat = opts.compatConfig();
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
        // The denoiser's container decides the architecture, so it is opened first
        // and its names are what `detectFamily` reads.
        try note(progress, "loading diffusion model...\n", .{});
        const t0 = std.Io.Clock.real.now(io).nanoseconds;
        var den_st = try Container.open(gpa, io, opts.dit_path);
        errdefer den_st.deinit();
        const fam = try detectFamily(den_st.store());

        var t1 = t0;
        var t2 = t0;
        switch (fam) {
            .krea2 => {
                var m: Krea2Models = .{
                    .tok = undefined,
                    .enc_st = null,
                    .enc = undefined,
                    .dit_st = den_st,
                    .dit = undefined,
                    .vae_st = null,
                    .vae = undefined,
                    .enc_view = null,
                    .vae_view = null,
                };
                const den = try reportResolve(gpa, fam, .denoiser, m.dit_st.store(), null, false, opts.dit_path);
                m.dit = try dit_mod.DiT.load(gpa, den.store);
                errdefer m.dit.deinit();
                // The DiT loader detects its own prefix, so the resolver's view is not
                // needed past this point.
                if (den.view) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                }
                t1 = std.Io.Clock.real.now(io).nanoseconds;

                try note(progress, "loading text encoder...\n", .{});
                m.tok = try tokenizer_mod.Tokenizer.init(gpa);
                errdefer m.tok.deinit();

                m.enc_st = try openIfGiven(gpa, io, opts.text_encoder_path, opts.explicit_text_encoder);
                errdefer if (m.enc_st) |*st| st.deinit();
                const enc_r = try reportResolve(gpa, fam, .conditioner, m.dit_st.store(), storeOf(&m.enc_st), opts.explicit_text_encoder, opts.text_encoder_path);
                m.enc_view = enc_r.view;
                m.enc = try qwen3.TextEncoder.load(gpa, enc_r.store);
                errdefer m.enc.deinit();

                m.vae_st = try openIfGiven(gpa, io, opts.vae_path, opts.explicit_vae);
                errdefer if (m.vae_st) |*st| st.deinit();
                const vae_r = try reportResolve(gpa, fam, .decoder, m.dit_st.store(), storeOf(&m.vae_st), opts.explicit_vae, opts.vae_path);
                m.vae_view = vae_r.view;
                m.vae = try wan_vae.Decoder.load(gpa, vae_r.store);
                t2 = std.Io.Clock.real.now(io).nanoseconds;
                self.models = .{ .krea2 = m };
            },
            .zimage => {
                var m: ZImageModels = .{
                    .tok = undefined,
                    .enc_st = null,
                    .enc = undefined,
                    .dit_st = den_st,
                    .dit = undefined,
                    .vae_st = null,
                    .vae = undefined,
                    .enc_view = null,
                    .vae_view = null,
                };
                const den = try reportResolve(gpa, fam, .denoiser, m.dit_st.store(), null, false, opts.dit_path);
                m.dit = try zimage.DiT.load(gpa, den.store, zimage.z_image);
                errdefer m.dit.deinit();
                // The DiT loader detects its own prefix, so the resolver's view is
                // not needed past this point.
                if (den.view) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                }
                t1 = std.Io.Clock.real.now(io).nanoseconds;

                try note(progress, "loading text encoder...\n", .{});
                // The same Qwen2 BPE vocabulary krea2 uses — Z-Image's ComfyUI
                // tokenizer is `Qwen2Tokenizer` over the qwen2.5 vocab.
                m.tok = try tokenizer_mod.Tokenizer.init(gpa);
                errdefer m.tok.deinit();

                const te_path = if (opts.explicit_text_encoder) opts.text_encoder_path else defaultComponentPath(fam, .conditioner);
                m.enc_st = try openIfGiven(gpa, io, te_path, opts.explicit_text_encoder);
                errdefer if (m.enc_st) |*st| st.deinit();
                const enc_r = try reportResolve(gpa, fam, .conditioner, m.dit_st.store(), storeOf(&m.enc_st), opts.explicit_text_encoder, te_path);
                m.enc_view = enc_r.view;
                m.enc = try qwen3.TextEncoder.loadVariant(gpa, enc_r.store, .zimage);
                errdefer m.enc.deinit();

                const vae_path = if (opts.explicit_vae) opts.vae_path else defaultComponentPath(fam, .decoder);
                m.vae_st = try openIfGiven(gpa, io, vae_path, opts.explicit_vae);
                errdefer if (m.vae_st) |*st| st.deinit();
                const vae_r = try reportResolve(gpa, fam, .decoder, m.dit_st.store(), storeOf(&m.vae_st), opts.explicit_vae, vae_path);
                m.vae_view = vae_r.view;
                m.vae = try sd_vae.Decoder.load(gpa, vae_r.store, sd_vae.flux, "");
                t2 = std.Io.Clock.real.now(io).nanoseconds;
                self.models = .{ .zimage = m };
            },
            .sd15, .sdxl => {
                const cfgs = sdConfigs(fam);

                var m: SdModels = .{
                    .tok = undefined,
                    .unet_st = den_st,
                    .enc_st = null,
                    .enc2_st = null,
                    .vae_st = null,
                    .clip = undefined,
                    .clip_g = null,
                    .unet = undefined,
                    .vae = undefined,
                    .clip_view = null,
                    .clip_g_view = null,
                    .vae_view = null,
                    .sigma_ladder = &.{},
                };
                const den = try reportResolve(gpa, fam, .denoiser, m.unet_st.store(), null, false, opts.dit_path);
                m.unet = try sd_unet.UNet.load(gpa, den.store, cfgs.unet, "");
                errdefer m.unet.deinit();
                if (den.view) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                }
                t1 = std.Io.Clock.real.now(io).nanoseconds;

                try note(progress, "loading text encoder...\n", .{});
                m.tok = try clip_tok.Tokenizer.init(gpa);
                errdefer m.tok.deinit();

                m.enc_st = try openIfGiven(gpa, io, opts.text_encoder_path, opts.explicit_text_encoder);
                errdefer if (m.enc_st) |*st| st.deinit();
                const clip_r = try reportResolve(gpa, fam, .conditioner, m.unet_st.store(), storeOf(&m.enc_st), opts.explicit_text_encoder, opts.text_encoder_path);
                m.clip_view = clip_r.view;
                m.clip = try clip_text.TextEncoder.load(gpa, clip_r.store, cfgs.clip, "");
                errdefer m.clip.deinit();

                // SDXL's second tower. It resolves independently — `--text-encoder`
                // overrides the *first* one, and a checkpoint carrying only CLIP-L is a
                // real (if unusual) split, so this gets its own resolution rather than
                // assuming both towers live wherever the first was found.
                //
                // Both `errdefer`s sit at case scope, not inside the `if`: scoped to the
                // block they would stop protecting these on any *later* failure (the VAE
                // load below), which is a leak only the test allocator would ever see.
                if (fam == .sdxl) m.enc2_st = try openIfGiven(gpa, io, opts.text_encoder_2_path, opts.explicit_text_encoder_2);
                errdefer if (m.enc2_st) |*st| st.deinit();
                if (fam == .sdxl) {
                    const g_r = try reportResolve(gpa, fam, .conditioner2, m.unet_st.store(), storeOf(&m.enc2_st), opts.explicit_text_encoder_2, opts.text_encoder_2_path);
                    m.clip_g_view = g_r.view;
                    m.clip_g = try clip_text.TextEncoder.load(gpa, g_r.store, clip_text.clip_g, "");
                }
                errdefer if (m.clip_g) |*c| c.deinit();

                m.vae_st = try openIfGiven(gpa, io, opts.vae_path, opts.explicit_vae);
                errdefer if (m.vae_st) |*st| st.deinit();
                const vae_r = try reportResolve(gpa, fam, .decoder, m.unet_st.store(), storeOf(&m.vae_st), opts.explicit_vae, opts.vae_path);
                m.vae_view = vae_r.view;
                m.vae = try sd_vae.Decoder.load(gpa, vae_r.store, cfgs.vae, "");
                errdefer m.vae.deinit();
                // Both SD architectures were trained on the same beta ladder, so the
                // sampler is shared verbatim.
                m.sigma_ladder = try sampler.sdSigmasFull(gpa);
                t2 = std.Io.Clock.real.now(io).nanoseconds;
                self.models = switch (fam) {
                    .sd15 => .{ .sd15 = m },
                    .sdxl => .{ .sdxl = m },
                    .krea2, .zimage => unreachable,
                };
            },
        }

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
        if (self.cu_be) |b| {
            b.enableAsyncStreaming(io);
            // Feed the staging ring from the checkpoint FILE rather than its
            // mapping. The DiT is the one that matters (~13 GB, read once per
            // image before it is pinned); see Container.weightReader.
            // Every checkpoint this session opened: each reader answers only for
            // its own mapping, so registering all of them just means whichever one
            // a weight came from serves it. Gated by `safetensors.read_mode` inside
            // `readTo`, so `--mmap mmap` turns the whole thing off.
            switch (self.models) {
                .krea2 => |*m| {
                    if (m.dit_st.weightReader()) |wr| b.ctx.addWeightReader(wr);
                    if (storeReader(&m.enc_st)) |wr| b.ctx.addWeightReader(wr);
                    if (storeReader(&m.vae_st)) |wr| b.ctx.addWeightReader(wr);
                },
                .zimage => |*m| {
                    if (m.dit_st.weightReader()) |wr| b.ctx.addWeightReader(wr);
                    if (storeReader(&m.enc_st)) |wr| b.ctx.addWeightReader(wr);
                    if (storeReader(&m.vae_st)) |wr| b.ctx.addWeightReader(wr);
                },
                else => {},
            }
        }
        try note(progress, "models loaded (encoder {d:.1}s, dit+vae {d:.1}s)\n", .{
            @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(t2 - t1)) / 1e9,
        });

        return self;
    }

    /// Swap the DiT for one loaded from `store`, keeping the tokenizer, text
    /// encoder, VAE and backend exactly as they are.
    ///
    /// For sweeps that vary only the diffusion weights — the level-2 per-tensor
    /// attribution arm this was added for substitutes ONE tensor via
    /// `weights.Overlay` and re-measures, once per layer. Reloading a whole
    /// `Session` per point would re-load the text encoder and VAE (tens of seconds
    /// each) and, worse, re-encode the conditioning, which puts a difference in the
    /// *inputs* of a measurement whose entire claim is that the inputs are
    /// identical.
    ///
    /// `store` may be an overlay over `denoiserStore()`: this call does not
    /// touch `dit_st`, so the base container stays open and mapped. The caller owns
    /// `store` and everything it points at, and both must outlive the DiT.
    ///
    /// The new DiT is loaded *before* the old one is freed, so a failed load leaves
    /// the session intact and usable.
    pub fn replaceDenoiser(self: *Session, store: weights_mod.WeightStore) !void {
        // ⚠️ Device weight caches are keyed by HOST POINTER, so they must be dropped
        // whatever the family: the outgoing denoiser's arena is about to be freed,
        // and a later allocation reusing one of those addresses for a different
        // tensor would score a stale cache hit and compute with the wrong weights.
        // Cheap next to a reload, and the next forward re-uploads what fits.
        //
        // This used to sit after an early return for the SD family, which was
        // harmless only while SD was CPU-only. Once it had GPU paths the omission
        // made a per-tensor patch a **silent no-op** — the host UNet held the
        // patched weight, the device served the cached original — reported as a
        // velocity identical to the reference (cos 1.0) rather than as an error.
        if (self.cu_be) |b| {
            b.bindThread();
            b.evictWeights();
        }
        if (self.gpu_ctx) |c| c.evictWeights();

        if (self.sd()) |m| {
            // The substituted store may be prefixed either way (an overlay over a
            // full checkpoint keeps the container prefix; over a ggufy GGUF it does
            // not), so it goes through the same resolver the initial load used.
            const r = try resolveComponent(self.gpa, self.family(), .denoiser, store, null, false);
            defer if (r.view) |v| {
                v.deinit(self.gpa);
                self.gpa.destroy(v);
            };
            const fresh = try sd_unet.UNet.load(self.gpa, r.store, m.unet.cfg, "");
            m.unet.deinit();
            m.unet = fresh;
            return;
        }
        if (self.family() == .zimage) {
            const fresh = try zimage.DiT.load(self.gpa, store, zimage.z_image);
            const m = self.zi();
            m.dit.deinit();
            m.dit = fresh;
            return;
        }
        const fresh = try dit_mod.DiT.load(self.gpa, store);
        const m = self.k();
        m.dit.deinit();
        m.dit = fresh;
    }

    /// The prefix the denoiser's own weights sit under inside `denoiserStore()`,
    /// or `""` when the container holds nothing else (ggufy's model-only GGUF, and
    /// every krea2 DiT).
    ///
    /// A caller that enumerates `denoiserStore().names()` needs this: a bundled SD
    /// checkpoint's container also holds the CLIP and VAE tensors, and patching one
    /// of those through a *denoiser* overlay is a silent no-op that measures as
    /// zero damage rather than as an error.
    pub fn denoiserPrefix(self: *const Session) []const u8 {
        const store = self.denoiserStore();
        const spec = componentSpec(self.family(), .denoiser) catch return "";
        for (spec.prefixes) |pfx| {
            var buf: [256]u8 = undefined;
            if (pfx.len + spec.probe.len > buf.len) continue;
            @memcpy(buf[0..pfx.len], pfx);
            @memcpy(buf[pfx.len..][0..spec.probe.len], spec.probe);
            if (store.get(buf[0 .. pfx.len + spec.probe.len]) != null) return pfx;
        }
        return "";
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
        switch (self.models) {
            .krea2 => |*m| {
                m.vae.deinit();
                m.dit.deinit();
                m.enc.deinit();
                inline for (.{ &m.enc_view, &m.vae_view }) |slot| if (slot.*) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                };
                if (m.enc_st) |*st| st.deinit();
                if (m.vae_st) |*st| st.deinit();
                m.dit_st.deinit();
                m.tok.deinit();
            },
            .zimage => |*m| {
                m.vae.deinit();
                m.dit.deinit();
                m.enc.deinit();
                inline for (.{ &m.enc_view, &m.vae_view }) |slot| if (slot.*) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                };
                if (m.enc_st) |*st| st.deinit();
                if (m.vae_st) |*st| st.deinit();
                m.dit_st.deinit();
                m.tok.deinit();
            },
            .sd15, .sdxl => |*m| {
                gpa.free(m.sigma_ladder);
                if (m.empty_ref) |e| gpa.free(e);
                if (m.empty_ref_g) |e| gpa.free(e);
                m.vae.deinit();
                m.clip.deinit();
                if (m.clip_g) |*c| c.deinit();
                m.unet.deinit();
                inline for (.{ &m.clip_view, &m.clip_g_view, &m.vae_view }) |slot| if (slot.*) |v| {
                    v.deinit(gpa);
                    gpa.destroy(v);
                };
                if (m.enc_st) |*st| st.deinit();
                if (m.enc2_st) |*st| st.deinit();
                if (m.vae_st) |*st| st.deinit();
                m.unet_st.deinit();
                m.tok.deinit();
            },
        }
        gpa.destroy(self);
    }

    /// Generate one image, reusing the loaded models. Rebuilds only the
    /// per-image state: conditioning, the DiT session/workspace (depend on the
    /// prompt AND resolution), the noise latent, and the schedule.
    /// This session's **sigma table** — `model_sampling.sigmas`, the thing a scheduler
    /// reads. krea2's is the shift-parameterized flux formula; the SD family's is the
    /// beta ladder it already caches, borrowed (so this allocates nothing, and the
    /// slice lives as long as the session).
    pub fn sigmaTable(self: *const Session, shift: f32) sampler.SigmaTable {
        return switch (self.models) {
            .krea2 => .{ .flux = shift },
            // ⚠️ A DIFFERENT flow table, not `.flux` with a converted shift: 1000
            // rungs against 10000, and the same formula rounded differently. See
            // `schedule.discreteFlowSigma`.
            .zimage => .{ .discrete_flow = shift },
            .sd15, .sdxl => |*m| .{ .discrete = m.sigma_ladder },
        };
    }

    /// The sigma schedule this session's family samples with, descending and ending at
    /// 0 — using each family's **default** scheduler, which is the behaviour that
    /// predates schedulers being selectable (`simple` for krea2, `normal` for SD).
    /// krea2 reads `shift`; SD1.5 ignores it (its ladder comes from the training betas),
    /// which is why this is a method rather than the free `pipeline.schedule` — a caller
    /// that cannot see the family cannot pick.
    pub fn schedule(self: *const Session, gpa: std.mem.Allocator, steps: usize, shift: f32) ![]f32 {
        return self.scheduleWith(gpa, steps, shift, null);
    }

    /// `schedule` with an explicit scheduler; null means the family default.
    ///
    /// ⚠️ **The result is not always `steps + 1` long.** `ddim_uniform` strides the sigma
    /// table and `beta` de-duplicates repeated rungs, so 30 requested steps can come
    /// back as 31 or fewer. Drive a sampling loop off `sigmas.len - 1` — `generate`
    /// does, and hardcoding `opts.steps` instead would read one past the end.
    pub fn scheduleWith(
        self: *const Session,
        gpa: std.mem.Allocator,
        steps: usize,
        shift: f32,
        sched: ?sampler.Scheduler,
    ) ![]f32 {
        const table = self.sigmaTable(shift);
        return sampler.schedule_mod.build(gpa, table, sched orelse .defaultFor(table), steps);
    }

    /// The shift this session's sigma table should use, honouring an explicit
    /// request and otherwise taking the family's own trained value.
    ///
    /// ⚠️ Z-Image's is **3.0** (`supported_models.py::ZImage.sampling_settings`, and
    /// the official ComfyUI template sets it again explicitly through a
    /// `ModelSamplingAuraFlow` node). Rendering it on krea2's 1.15 is not an error —
    /// it is a valid schedule that puts the steps in the wrong places, which at 8
    /// steps is the difference between a finished image and a smeared one.
    pub fn resolvedShift(self: *const Session, opts: Options) f32 {
        if (opts.explicit_shift) return opts.shift;
        return defaultShift(self.family());
    }

    /// Scale a freshly drawn unit-normal latent to `sigmas[0]`, the way **this family**
    /// starts a trajectory. A method for the same reason `schedule` is one: the two
    /// parameterizations disagree and a caller that cannot see the family cannot pick.
    ///
    /// krea2 (flow matching) multiplies by `sigma0`, which is exactly 1.0 — a
    /// bit-identical no-op. The SD family multiplies by `sqrt(1 + sigma0²)`; see
    /// `sampler.sdScaleInitialNoise` for why that is not the same as `sigma0` and what
    /// getting it wrong costs.
    pub fn scaleInitialNoise(self: *const Session, x: []f32, sigma0: f32) void {
        switch (self.models) {
            // Both flow-matching families: a multiply by `sigma0`, which for both of
            // them is exactly 1.0 at the top of the schedule (Z-Image's
            // `time_snr_shift(3, 1)` is 3/3), so it is a bit-identical no-op.
            .krea2, .zimage => sampler.scaleInitialNoise(x, sigma0),
            // ⚠️ Under `--compat a1111` this is the BARE sigma: A1111's
            // `sgm_noise_multiplier` defaults to False, and its own description of the
            // option ("match initial noise to official SDXL implementation - only useful
            // for reproducing images") is an admission that its default is the odd one
            // out. Both forms render; they render different images.
            .sd15, .sdxl => if (self.compat.sgm_noise_multiplier)
                sampler.sdScaleInitialNoise(x, sigma0)
            else
                sampler.scaleInitialNoise(x, sigma0),
        }
    }

    /// The half-logSNR parameterization this family's denoiser is trained in — what a
    /// higher-order sampler needs and Euler does not. A method for the same reason
    /// `schedule` is one, and the failure mode is worse: `flow` on an SD ladder is
    /// NaN from the first step, `eps` on a krea2 schedule silently integrates the
    /// wrong ODE. See `sampler.Parameterization`.
    pub fn parameterization(self: *const Session) sampler.Parameterization {
        return switch (self.models) {
            .krea2, .zimage => .flow,
            .sd15, .sdxl => .eps,
        };
    }

    /// Latent channel count for this family — 16 for krea2's Wan VAE, 4 for SD's
    /// AutoencoderKL. A caller sizing its own latent needs this.
    pub fn latentChannels(self: *const Session) usize {
        return switch (self.models) {
            .krea2 => wan_vae.latent_channels,
            .zimage => zimage.latent_channels,
            .sd15, .sdxl => |*m| m.unet.cfg.channels,
        };
    }

    /// Fill `rgb_out` (`[lat_h*lat_w][3]` RGB8) with the cheap linear latent2rgb
    /// preview of a planar sampler latent, using THIS family's factors. Each family
    /// has its own matrix over its own channel count (16 for krea2's Wan latent, 4 for
    /// SD's, with different factors for SD1.5 and SDXL); the wrong one is either an
    /// out-of-bounds read or a preview with plausible structure and wrong colours.
    pub fn latentPreviewInto(self: *const Session, rgb_out: []u8, z: []const f32, lat_h: usize, lat_w: usize) void {
        switch (self.models) {
            .krea2 => wan_vae.latentPreviewInto(rgb_out, z, lat_h, lat_w),
            // Also 16 channels, but the Flux matrix and a non-zero bias — a
            // different picture from krea2's Wan matrix, not a shared one.
            .zimage => zimage.latentPreviewInto(rgb_out, z, lat_h, lat_w),
            .sd15 => sd_vae.latentPreviewInto(rgb_out, z, lat_h, lat_w, &sd_vae.latent_rgb_factors_sd15, sd_vae.latent_rgb_bias_sd15),
            .sdxl => sd_vae.latentPreviewInto(rgb_out, z, lat_h, lat_w, &sd_vae.latent_rgb_factors_sdxl, sd_vae.latent_rgb_bias_sdxl),
        }
    }

    /// Stage 1 — text → conditioning, on the resident encoder. Caller frees the
    /// returned `Cond`. Device allocations are tagged `.te` for the VRAM meter.
    pub fn encode(self: *Session, gpa: std.mem.Allocator, text: []const u8, o: EncodeOptions) !Cond {
        self.setMemTag(.te);
        // A1111's `[a:b:when]` / `[a|b]` make the prompt a function of the step, so the
        // result is a schedule of conditionings rather than one. `steps == 0` means the
        // caller is not rendering a schedule (a single measurement forward), so the
        // step-0 prompt is all there is.
        if (o.prompt_syntax == .a1111 and o.steps > 0) return self.encodeScheduled(gpa, text, o);
        return self.encodeText(gpa, text, o);
    }

    /// Resolve A1111 scheduling, deduplicate the resulting prompts by text, and encode
    /// each distinct one.
    ///
    /// ⚠️ **Deduplication is load-bearing, not an optimization.** `[a|b]` contributes an
    /// entry for EVERY step (upstream's `collect_steps` does), so a 35-step render yields
    /// 35 schedule entries over 2 distinct texts. Encoding per entry would mean 33
    /// redundant tower forwards and 33 redundant device sessions.
    fn encodeScheduled(self: *Session, gpa: std.mem.Allocator, text: []const u8, o: EncodeOptions) !Cond {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const plan = try planSchedule(arena, gpa, text, o.steps);
        const uniq = plan.texts;
        const at = plan.at;
        errdefer gpa.free(at);

        var first = try self.encodeText(gpa, uniq[0], o);
        errdefer first.deinit(gpa);
        if (uniq.len == 1) {
            // Scheduling syntax that resolved to one prompt (or none at all): no schedule.
            gpa.free(at);
            return first;
        }

        const extra = try gpa.alloc(Cond, uniq.len - 1);
        errdefer gpa.free(extra);
        var built: usize = 0;
        errdefer for (extra[0..built]) |*e| e.deinit(gpa);
        for (uniq[1..], 0..) |t, i| {
            extra[i] = try self.encodeText(gpa, t, o);
            built += 1;
        }
        first.sched = .{ .extra = extra, .at = at };
        return first;
    }

    /// One conditioning for one concrete prompt text — the whole of `encode` before
    /// scheduling existed, plus the dialect switch in how the text is tokenized.
    fn encodeText(self: *Session, gpa: std.mem.Allocator, text: []const u8, o: EncodeOptions) !Cond {
        // Weights the loaded encoder cannot honour are WARNED about and the prompt encoded
        // verbatim — not refused, and not silently dropped. Generic rather than a branch in
        // the family switch below, because it is a property of the encoder (see
        // `supportsPromptWeights`) and every future family gets it for free. Per-step
        // scheduling is unaffected and works on all of them: `Cond.sched` is family-neutral
        // and this function runs once per resolved text, so krea2 gains something the
        // ComfyUI dialect cannot express, where `[…]` is literal text.
        if (o.prompt_syntax == .a1111 and !supportsPromptWeights(self.family())) {
            if (try weightsWouldBeDropped(gpa, text, o)) std.log.warn(
                "prompt weights are not applied by this model's text encoder: it has no " ++
                    "fixed token window to weight them in (emphasis is a CLIP-only feature " ++
                    "here and in every other implementation). Encoding the prompt verbatim; " ++
                    "--emphasis ignore strips the syntax instead.",
                .{},
            );
        }
        switch (self.models) {
            .krea2 => |*m| {
                return encodePrompt(self.io, gpa, self.gpu_ctx, self.cu_be, o.encoder_f16, &m.tok, &m.enc, text, o.cancel);
            },
            .sd15 => |*m| {
                var p = try self.tokenizePrompt(gpa, &m.tok, text, clip_tok.eos_id, o);
                defer p.deinit(gpa);
                const hidden = try gpa.alloc(f32, p.seq() * m.clip.cfg.hidden);
                errdefer gpa.free(hidden);
                try self.runClip(gpa, &m.clip, hidden, &p, .{
                    .mode = weightMode(o),
                    .empty_cache = &m.empty_ref,
                    .pad_id = clip_tok.eos_id,
                }, o);
                return .{ .data = hidden, .seq = p.seq() };
            },
            .sdxl => |*m| return self.encodeSdxl(gpa, m, o, text),
            .zimage => |*m| {
                var ids: std.ArrayList(u32) = .empty;
                defer ids.deinit(gpa);
                try zimage_text.buildIds(&m.tok, gpa, text, &ids);
                // ⚠️ No prefix strip, unlike krea2: Z-Image conditions on the WHOLE
                // token sequence, chat markers included. See `zimage_text`.
                const data = try self.runQwen3(gpa, &m.enc, ids.items, o);
                return .{ .data = data, .seq = ids.items.len };
            },
        }
    }

    /// One Qwen3 encode, dispatched by backend with the CPU forward as the fallback.
    /// Shared by krea2's `encodePrompt` and Z-Image's arm above — the difference
    /// between the two families is the tap list and the template, both of which are
    /// already decided by the time this runs.
    fn runQwen3(self: *Session, gpa: std.mem.Allocator, enc: *const qwen3.TextEncoder, ids: []const u32, o: EncodeOptions) ![]f32 {
        if (self.cu_be) |b| {
            // ⚠️ Only when the CUDA arm can actually run these weights: its encode
            // GEMM is `opMatmulFp8` unconditionally, and Z-Image's Qwen3-4B ships
            // bf16, which it would read as fp8 bytes and turn into noise.
            if (qwen3_cuda.supportsWeights(enc)) {
                return qwen3_cuda.encode(enc, b, self.io, gpa, ids, o.cancel) catch |err| {
                    if (err == error.Canceled) return err;
                    std.log.warn("cuda text encode failed ({t}); falling back to CPU (slow)", .{err});
                    return enc.encode(self.io, gpa, ids, o.cancel);
                };
            }
        } else if (self.gpu_ctx) |gc| {
            return qwen3_gpu.encode(enc, gc, self.io, gpa, ids, o.encoder_f16, o.cancel) catch |err| {
                if (err == error.Canceled) return err;
                std.log.warn("vulkan text encode failed ({t}); falling back to CPU (slow)", .{err});
                return enc.encode(self.io, gpa, ids, o.cancel);
            };
        }
        return enc.encode(self.io, gpa, ids, o.cancel);
    }

    /// Tokenize one prompt text in the requested dialect. The A1111 path is two passes
    /// (`parseAttention` then `encodeParts`); ComfyUI's is one.
    fn tokenizePrompt(
        self: *Session,
        gpa: std.mem.Allocator,
        tok: *const clip_tok.Tokenizer,
        text: []const u8,
        pad_id: u32,
        o: EncodeOptions,
    ) !clip_tok.Prompt {
        _ = self;
        if (o.prompt_syntax == .comfy) return tok.encodeWeighted(gpa, text, pad_id);

        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const parts = try prompt_a1111.parseAttention(arena_state.allocator(), text);
        if (o.emphasis == .ignore) {
            // `EmphasisIgnore`: the syntax is still parsed and stripped — so `(a:1.5)`
            // conditions on `a`, not on the punctuation — but every weight becomes 1.0.
            for (parts) |*part| if (!part.isBreak()) {
                part.weight = 1.0;
            };
        }
        return tok.encodeParts(gpa, parts, pad_id);
    }

    /// Whether an A1111 prompt carries a weight that an emphasis-incapable encoder would
    /// leave unapplied. Returns a fact rather than logging or erroring, so the caller decides
    /// what to do with it — and so a test can assert it without printing to stderr on success
    /// (any output from a passing test makes the runner print a misleading failure line).
    ///
    /// ⚠️ **This was `error.PromptSyntaxUnsupportedForFamily`, first blanket and then narrowed
    /// to weighted prompts, and both were wrong.** The reasoning was the house rule that
    /// silently dropping weights a caller asked for is this area's recurring failure — but
    /// refusing is not the only alternative to silence, and it is not what the ecosystem does.
    /// **Emphasis weighting is a CLIP-only feature everywhere it is implemented**, checked
    /// against SD.Next (`dev`; A1111 itself never had to decide, having only ever had CLIP):
    ///
    /// - `processing_prompt.set_prompt` gates the emphasis parser behind an allow-list of
    ///   `StableDiffusion*` / `StableCascade` / `Flux`; everything else — Qwen-Image, Wan,
    ///   Lumina, Flux2, and krea2's shape — falls to `prompt_attention = 'fixed'`, i.e. the
    ///   raw string reaches the pipeline unparsed. Chroma and HiDream sit *commented out* of
    ///   that list: tried, then disabled.
    /// - SD3 settles it, having both kinds of encoder at once: the CLIP towers go through the
    ///   weighted provider while the T5 tower is `_get_t5_prompt_embeds(prompt=…)` on the raw
    ///   string, the two then concatenated. Flux is *on* the allow-list and still encodes via
    ///   `pipe.encode_prompt` unweighted ("clip is only used for the pooled embeds").
    ///
    /// So refusing made this engine stricter than any live tool, on a *persistent* setting: a
    /// global choice made for an SDXL render became a hard error on every later krea2 render.
    /// A warning satisfies the house rule (nothing is silent — SD.Next only `debug_log`s its
    /// own fallback) without failing a render nobody else fails.
    ///
    /// The prompt is then encoded **verbatim rather than stripped**, deliberately: it keeps
    /// `.comfy` and `.a1111` byte-identical on such a model, and `--emphasis ignore` is the
    /// explicit way to ask for the syntax to be parsed away instead.
    fn weightsWouldBeDropped(gpa: std.mem.Allocator, text: []const u8, o: EncodeOptions) !bool {
        if (o.emphasis == .ignore) return false; // already asked for the weights to be dropped
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const parts = try prompt_a1111.parseAttention(arena_state.allocator(), text);
        for (parts) |part| {
            // A BREAK is a chunk boundary such a model has no concept of in EITHER dialect —
            // under `.comfy` the word tokenizes literally too — so it is not worth a word.
            if (part.isBreak()) continue;
            if (part.weight != 1.0) return true;
        }
        return false;
    }

    /// The weighting form the encoders should apply for these options.
    fn weightMode(o: EncodeOptions) clip_text.TextEncoder.WeightMode {
        if (o.prompt_syntax == .comfy) return .comfy;
        return switch (o.emphasis) {
            // `ignore` already flattened every weight to 1.0, so the form no longer
            // matters; the cheaper one is picked so no empty-prompt forward is run.
            .no_norm, .ignore => .a1111_no_norm,
            .original => .a1111_original,
        };
    }

    /// Run one CLIP tower on the best available backend, falling back to the CPU only
    /// for errors the CPU can actually recover from.
    ///
    /// ⚠️ **A device FAULT must NOT fall back, and getting this wrong hid a real bug
    /// behind an error naming the wrong stage.** `CUDA_ERROR_ILLEGAL_ADDRESS` (and a
    /// Vulkan device loss) is **sticky**: once a kernel faults, every later call in that
    /// context fails too. The original policy — copied from krea2's `encodePrompt` —
    /// caught *any* error, logged a warning, re-ran the encode on the CPU (which
    /// succeeds, needing no device) and returned normally. The render then died at the
    /// denoiser's first conditioning upload with
    /// `cuMemcpyHtoD failed: CUDA_ERROR_ILLEGAL_ADDRESS`, three stages downstream of the
    /// fault, with the actual error demoted to a warning nobody reads. Reported as
    /// "fails after the TE step", which is exactly where it did *not* fail.
    ///
    /// So the fallback is now allow-listed rather than catch-all:
    ///
    /// - `UnsupportedDType` — this tower's weights have no GPU GEMM (a block-quantized
    ///   text encoder). A real, expected condition; the CPU is the right answer.
    /// - `OutOfMemory` — the device reclaim ladder came up short. The context is intact
    ///   and the CPU path needs no VRAM, so recovering is both safe and useful.
    /// - everything else, `CudaError` above all — **propagate**. A poisoned context
    ///   cannot be recovered by running the next stage anyway, and the error that names
    ///   the faulting stage is worth far more than a render that limps one stage further.
    ///
    /// ⚠️ **A cancel must propagate too**, or cancelling a render silently restarts the
    /// encode on the CPU and takes longer than not cancelling.
    ///
    /// ⚠️ **The three arms are not numerically equal, and CUDA is the loosest.** Vulkan
    /// computes the GEMMs in f32 by default (f16 only under `--encoder-f16`), so it
    /// tracks the CPU forward the ComfyUI fixtures pin to ~1e-6; the CUDA arm has only a
    /// tensor-core entry point at these widths and lands ~1e-3. That is the regime
    /// `sd_unet_cuda` already runs the denoiser in, so it is consistent rather than
    /// newly lossy — but it does mean a `cuda` render and a `cpu` render of the same seed
    /// differ slightly, which is worth knowing before attributing such a difference to
    /// something else.
    fn runClip(
        self: *Session,
        gpa: std.mem.Allocator,
        tower: *const clip_text.TextEncoder,
        out: []f32,
        p: *const clip_tok.Prompt,
        r: clip_text.TextEncoder.PromptRun,
        o: EncodeOptions,
    ) !void {
        if (o.cancel) |c| if (c.load(.monotonic)) return error.Canceled;
        if (self.cu_be) |b| {
            return clip_text_cuda.encodePrompt(tower, b, self.io, gpa, out, p, r) catch |err| {
                if (!clipFallbackOk(err)) return err;
                std.log.warn("cuda clip encode: {t}; falling back to CPU (slow)", .{err});
                return tower.encodePrompt(self.io, gpa, out, p, r);
            };
        }
        if (self.gpu_ctx) |gc| {
            return clip_text_gpu.encodePrompt(tower, gc, self.io, gpa, out, p, r, o.encoder_f16) catch |err| {
                if (!clipFallbackOk(err)) return err;
                std.log.warn("vulkan clip encode: {t}; falling back to CPU (slow)", .{err});
                return tower.encodePrompt(self.io, gpa, out, p, r);
            };
        }
        return tower.encodePrompt(self.io, gpa, out, p, r);
    }

    /// Whether a GPU text-encode error is one the CPU path can recover from. See
    /// `runClip` for why this is an allow-list and not `catch |_|`.
    fn clipFallbackOk(err: anyerror) bool {
        return switch (err) {
            error.UnsupportedDType, error.OutOfMemory => true,
            else => false,
        };
    }

    /// SDXL's conditioning: both towers, concatenated along the feature axis, plus
    /// CLIP-G's projected pooled row.
    ///
    /// Three conventions here are ecosystem choices rather than derivable facts, and each
    /// produces a plausible image if got wrong (see `clip_tokenizer.encodePadded` and
    /// `clip_text.Config.Naming`):
    ///
    /// 1. **The penultimate hidden state, with the final LayerNorm SKIPPED.** Both towers,
    ///    not just one. Using the final state instead is the "clip skip 1 vs 2" difference
    ///    and shifts style across the whole image.
    /// 2. **CLIP-L is padded with EOS and CLIP-G with 0**, so the prompt is tokenized
    ///    twice. The padded slots are conditioning too.
    /// 3. **Pooled comes from CLIP-G only**, from its *final-LayerNormed* last hidden state
    ///    at the first EOS, then through `text_projection` — a different tensor than
    ///    either half of the concatenated context.
    fn encodeSdxl(self: *Session, gpa: std.mem.Allocator, m: *SdModels, o: EncodeOptions, text: []const u8) !Cond {
        const g = &m.clip_g.?;
        const hl = m.clip.cfg.hidden;
        const hg = g.cfg.hidden;
        const clen = clip_tok.context_length;

        // Tokenized twice: same ids, different padding (convention 2 above). The chunk
        // split is length-driven and the two differ only in padding, so the counts always
        // agree — `min` is ComfyUI's own defensive form, not a case that arises here.
        var pl = try self.tokenizePrompt(gpa, &m.tok, text, clip_tok.eos_id, o);
        defer pl.deinit(gpa);
        var pg = try self.tokenizePrompt(gpa, &m.tok, text, 0, o);
        defer pg.deinit(gpa);
        const chunks = @min(pl.chunks, pg.chunks);
        const seq = chunks * clen;

        // `cap_*` take the penultimate state, which is what the UNet cross-attends to.
        // `final_g` takes CLIP-G's chunk-0 final output, the *only* thing pooled reads.
        const cap_l = try gpa.alloc(f32, pl.seq() * hl);
        defer gpa.free(cap_l);
        // `layers - 1` is "after encoder layer layers-2", i.e. the reference's
        // `hidden_states[-2]` — one short of the last block, before the final norm.
        try self.runClip(gpa, &m.clip, cap_l, &pl, .{
            .mode = weightMode(o),
            .empty_cache = &m.empty_ref,
            .pad_id = clip_tok.eos_id,
            .capture_layer = m.clip.cfg.layers - 1,
        }, o);

        const cap_g = try gpa.alloc(f32, pg.seq() * hg);
        defer gpa.free(cap_g);
        const final_g = try gpa.alloc(f32, clen * hg);
        defer gpa.free(final_g);
        try self.runClip(gpa, g, cap_g, &pg, .{
            .mode = weightMode(o),
            .empty_cache = &m.empty_ref_g,
            .pad_id = 0,
            .capture_layer = g.cfg.layers - 1,
            .final_chunk0 = final_g,
        }, o);

        const data = try gpa.alloc(f32, seq * (hl + hg));
        errdefer gpa.free(data);
        for (0..seq) |p| {
            @memcpy(data[p * (hl + hg) ..][0..hl], cap_l[p * hl ..][0..hl]);
            @memcpy(data[p * (hl + hg) + hl ..][0..hg], cap_g[p * hg ..][0..hg]);
        }

        // ⚠️ Pooled comes from CHUNK 0 only, and is never weighted. It is one vector for
        // the whole prompt, so there is nothing to concatenate; taking it from the last
        // chunk instead would condition `y` on the quality tags alone.
        const pooled = try gpa.alloc(f32, hg);
        errdefer gpa.free(pooled);
        var ids0: [clip_tok.context_length]u32 = undefined;
        pg.idsInto(&ids0, 0);
        const row = g.pooled(final_g, &ids0) orelse return error.NoEosInPrompt;
        g.projectPooled(pooled, row);

        return .{ .data = data, .seq = seq, .pooled = pooled };
    }

    /// Stage 2 — build the per-image denoiser (see `Denoiser`). `cond_neg` and a
    /// `cfg` other than 1.0 go together: pass both for classifier-free guidance,
    /// neither for a single forward per step. `sigmas` is the whole schedule (the
    /// Vulkan backend precomputes a timestep vector per entry).
    pub fn denoiser(
        self: *Session,
        gpa: std.mem.Allocator,
        cond_pos: Cond,
        cond_neg: ?Cond,
        cfg: f32,
        lat_h: usize,
        lat_w: usize,
        sigmas: []const f32,
    ) !Denoiser {
        const use_cfg = cfg != 1.0;
        if (use_cfg and cond_neg == null) return error.CfgNeedsNegativeCond;

        var d: Denoiser = .{
            .sess = self,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .cfg = cfg,
            .cond_pos = cond_pos,
            .cond_neg = cond_neg,
        };
        errdefer d.deinit(gpa);

        // The SD family needs the input-scaling scratch and a UNet workspace, both
        // per resolution — plus SDXL's `y`, which is per resolution too. On Vulkan
        // the workspace is device-side and there is one session per conditioning
        // branch; on the CPU it is the host `sd_unet.Workspace`.
        if (self.sd()) |m| {
            const n = m.unet.cfg.channels * lat_h * lat_w;
            if (use_cfg) d.v_neg = try gpa.alloc(f32, n);
            d.sd_scaled = try gpa.alloc(f32, n);
            d.sd_eps = try gpa.alloc(f32, n);
            // ⚠️ **The cap is over EVERY scheduled entry, not just entry 0.** A prompt
            // whose variants tokenize to different chunk counts (`[a:a very long tail
            // that spills past 77 tokens:0.5]`) would otherwise size the shared
            // workspace for the wrong one, and cross-attention writes `ctx_seq` rows
            // into it.
            var seq_cap_sd: usize = 0;
            for (0..cond_pos.entryCount()) |i| seq_cap_sd = @max(seq_cap_sd, cond_pos.entry(i).seq);
            if (cond_neg) |*cn| {
                for (0..cn.entryCount()) |i| seq_cap_sd = @max(seq_cap_sd, cn.entry(i).seq);
            }

            // One branch per distinct conditioning. Built eagerly: dedup at encode time
            // keeps this at the number of distinct prompt TEXTS, so an `[a|b]` over 35
            // steps builds 2, not 35.
            d.sd_pos = try gpa.alloc(SdBranch, cond_pos.entryCount());
            @memset(d.sd_pos, .{});
            d.sd_neg = try gpa.alloc(SdBranch, if (cond_neg) |*cn| cn.entryCount() else 0);
            @memset(d.sd_neg, .{});

            const mc = sd_unet.MicroCond.forSize(lat_h * 8, lat_w * 8);
            {
                self.setMemTag(.latent);
                defer self.setMemTag(.dit);
                for ([_]?*const Cond{ &cond_pos, if (cond_neg) |*cn| cn else null }, 0..) |maybe, side| {
                    const c = maybe orelse continue;
                    const arr = if (side == 0) d.sd_pos else d.sd_neg;
                    for (arr, 0..) |*br, i| {
                        const ce = c.entry(i);
                        // ⚠️ `y` first: a session folds the label embedding into its
                        // per-forward biases and therefore needs the micro-conditioning
                        // already built.
                        //
                        // The micro-conditioning is `original == target == this image, no
                        // crop` — what ComfyUI and diffusers both default to, and not a
                        // free choice: SDXL learned to associate small declared originals
                        // with low-quality training crops, so declaring anything smaller
                        // than the render asks the model for a worse image.
                        if (m.unet.cfg.adm_channels) |adm_len| {
                            br.adm = try gpa.alloc(f32, adm_len);
                            sd_unet.admVector(br.adm.?, ce.pooled orelse return error.MissingPooledCond, mc);
                        }
                        if (self.cu_be) |b| {
                            br.cu = try sd_unet_cuda.Session.init(gpa, b, &m.unet, ce.data, ce.seq, br.adm);
                        } else if (self.gpu_ctx) |gc| {
                            br.vk = try sd_unet_gpu.Session.init(gpa, gc, &m.unet, ce.data, ce.seq, br.adm);
                        }
                    }
                }
            }

            if (self.cu_be) |b| {
                self.setMemTag(.latent);
                defer self.setMemTag(.dit);
                d.sd_cu_ws = try sd_unet_cuda.Workspace.init(gpa, b, &m.unet, lat_h, lat_w, seq_cap_sd);
            } else if (self.gpu_ctx) |gc| {
                self.setMemTag(.latent);
                defer self.setMemTag(.dit);
                d.sd_vk_ws = try sd_unet_gpu.Workspace.init(gpa, gc, &m.unet, lat_h, lat_w, seq_cap_sd);
            } else {
                d.sd_ws = try sd_unet.Workspace.init(gpa, &m.unet, lat_h, lat_w, seq_cap_sd);
            }
            return d;
        }

        if (self.family() == .zimage) {
            const m = self.zi();
            if (use_cfg) d.v_neg = try gpa.alloc(f32, zimage.latent_channels * lat_h * lat_w);
            // ⚠️ Both branches must pad to the SAME caption length, or the image
            // half's RoPE positions (which start at `cap_padded + 1`) would differ
            // between the two CFG passes and the guidance would mix two different
            // geometries. The pad multiple makes that automatic only when the two
            // prompts land in the same bucket, so it is asserted rather than assumed.
            d.zi_cap_pos = try m.dit.capTokens(self.io, gpa, cond_pos.data, cond_pos.seq);
            d.zi_cap_padded = zimage.z_image.padded(cond_pos.seq);
            if (cond_neg) |cn| {
                if (zimage.z_image.padded(cn.seq) != d.zi_cap_padded) return error.CondLengthMismatch;
                d.zi_cap_neg = try m.dit.capTokens(self.io, gpa, cn.data, cn.seq);
            }

            // ⚠️ **Vulkan only** — there is no CUDA Z-Image forward yet, and that case
            // says so rather than being quietly 100x slow: a silent CPU trunk under
            // `--backend cuda` reads as a hang, not as a fallback.
            if (self.gpu_ctx) |gc| {
                if (zimage_gpu.supported(gc, &m.dit)) {
                    self.setMemTag(.latent);
                    defer self.setMemTag(.dit);
                    d.zi_vk = try zimage_gpu.Session.init(gpa, self.io, gc, &m.dit, lat_h, lat_w, d.zi_cap_pos.?, d.zi_cap_padded, sigmas);
                    if (d.zi_cap_neg) |cn| {
                        d.zi_vk_neg = try zimage_gpu.Session.init(gpa, self.io, gc, &m.dit, lat_h, lat_w, cn, d.zi_cap_padded, sigmas);
                    }
                    d.zi_vk_ws = try zimage_gpu.Workspace.init(gc, &m.dit, lat_h, lat_w, d.zi_cap_padded);
                } else {
                    std.log.warn("Z-Image: this device has no tensor-core pipeline for the DiT's " ++
                        "weights; the trunk runs on the CPU. Expect CPU sampling speed.", .{});
                }
            } else if (self.cu_be) |b| {
                if (zimage_cuda.supported(&m.dit)) {
                    self.setMemTag(.latent);
                    defer self.setMemTag(.dit);
                    d.zi_cu = try zimage_cuda.Session.init(gpa, self.io, b, &m.dit, lat_h, lat_w, d.zi_cap_pos.?, d.zi_cap_padded, sigmas);
                    if (d.zi_cap_neg) |cn| {
                        d.zi_cu_neg = try zimage_cuda.Session.init(gpa, self.io, b, &m.dit, lat_h, lat_w, cn, d.zi_cap_padded, sigmas);
                    }
                    d.zi_cu_ws = try zimage_cuda.Workspace.init(b, &m.dit, lat_h, lat_w, d.zi_cap_padded);
                } else {
                    std.log.warn("Z-Image: this checkpoint's weight dtype has no CUDA GEMM path; " ++
                        "the trunk runs on the CPU. Expect CPU sampling speed.", .{});
                }
            }
            return d;
        }

        if (use_cfg) d.v_neg = try gpa.alloc(f32, wan_vae.latent_channels * lat_h * lat_w);

        // The per-image device buffers (session + activation workspace) are the
        // meter's "latent / working" segment — tagged so it is MEASURED rather
        // than left as a remainder. Restored to `.dit` on the way out, since what
        // follows is the sampling loop's lazily streamed weights and attention
        // scratch.
        self.setMemTag(.latent);
        defer self.setMemTag(.dit);

        // One workspace, shared by both conditioning passes, sized to the longer
        // of the two sequences.
        const seq_cap = @max(cond_pos.seq, if (cond_neg) |c| c.seq else 0);
        const dit = &self.k().dit;
        if (self.cu_be) |b| {
            d.cu_pos = try dit_cuda.Session.init(gpa, self.io, b, dit, lat_h, lat_w, cond_pos.data, cond_pos.seq);
            if (use_cfg) d.cu_neg = try dit_cuda.Session.init(gpa, self.io, b, dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq);
            d.cu_ws = try dit_cuda.Workspace.init(b, lat_h, lat_w, seq_cap);
        } else if (self.gpu_ctx) |gc| {
            d.vk_pos = try dit_gpu.Session.init(gpa, self.io, gc, dit, lat_h, lat_w, cond_pos.data, cond_pos.seq, sigmas);
            if (use_cfg) d.vk_neg = try dit_gpu.Session.init(gpa, self.io, gc, dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq, sigmas);
            d.vk_ws = try dit_gpu.Workspace.init(gc, lat_h, lat_w, seq_cap);
        }
        return d;
    }

    /// One denoiser forward, self-contained: `v_out = model(latent, sigma, cond)`.
    ///
    /// This is level 2's primitive (plan §7) — a fixed latent, sigma and
    /// conditioning through the model once, with no sampler and therefore no
    /// trajectory to drift. It builds and tears down a `Denoiser` per call, which
    /// is free on CPU and wasteful on a GPU backend; for many forwards at one
    /// resolution, hold a `Denoiser` and call its `predict` instead.
    pub fn predict(
        self: *Session,
        gpa: std.mem.Allocator,
        v_out: []f32,
        latent: []const f32,
        lat_h: usize,
        lat_w: usize,
        sigma: f32,
        cond: Cond,
    ) !void {
        var d = try self.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, &.{ sigma, 0 });
        defer d.deinit(gpa);
        try d.predict(gpa, v_out, latent, sigma, null);
    }

    /// Stage 3 — latent → pixels on the resident VAE: denormalize, then decode
    /// through the adaptive whole-image → GPU-tiled → CPU-tiled ladder (see
    /// `VaeDecode`), then interleave to RGB8. `latent` is `[16][lat_h][lat_w]`
    /// planar and is NOT modified. The image comes back at `lat_w*8 x lat_h*8`.
    pub fn decode(
        self: *Session,
        latent: []const f32,
        lat_h: usize,
        lat_w: usize,
        o: DecodeOptions,
        progress: ?*std.Io.Writer,
    ) !Image {
        const gpa = self.gpa;
        const io = self.io;

        // The SD family differs from krea2 only in its normalization — its
        // AutoencoderKL takes `z / scaling_factor`, where Wan's is per-channel
        // (mean, std) — and in which decoder the ladder below drives. Both arms
        // end up in `decodePlanar`, so the whole-image → reclaim → GPU-tiled →
        // CPU-tiled recovery is shared.
        //
        // ⚠️ It was NOT shared until 2026-08-03, and that hard-failed every render
        // whose SD VAE decode did not fit: at 1024x1536 the decoder wants 3 x 1.5 GiB
        // of activations, and with a resident LLM or another process on the card
        // that surfaces either as `DeviceOutOfMemory` or — once the OOM ladder has
        // freed memory an in-flight kernel still referenced — as a post-OOM stream
        // fault reported at the next `cuMemcpyHtoD` (`CUDA_ERROR_ILLEGAL_ADDRESS`,
        // i.e. `error.CudaError`). `recoverableDecodeErr` already names that exact
        // case; the SD path just returned before ever reaching it.
        if (self.sd()) |m| {
            const ch = m.unet.cfg.channels;
            if (latent.len != ch * lat_h * lat_w) return error.LatentSizeMismatch;
            self.setMemTag(.vae);
            // Denormalize onto a copy: `decode` must not modify the caller's latent
            // (a caller holding it for a drift curve would otherwise have it
            // silently rescaled — there is a test for it).
            const x = try gpa.dupe(f32, latent);
            defer gpa.free(x);
            const inv = 1.0 / m.vae.cfg.scaling_factor;
            for (x) |*v| v.* *= inv;
            try note(progress, "decoding latent...\n", .{});
            const dec_start = std.Io.Clock.real.now(io);
            const planar = try self.decodePlanar(SdVae{ .vae = &m.vae }, x, lat_h, lat_w, o, progress);
            defer gpa.free(planar);
            try note(progress, "decoded in {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dec_start.nanoseconds)) / 1e9});
            const width = lat_w * 8;
            const height = lat_h * 8;
            return .{
                .rgb = try image.planarF32ToRgb8(gpa, planar, width, height),
                .width = width,
                .height = height,
            };
        }

        if (self.family() == .zimage) {
            const m = self.zi();
            if (latent.len != zimage.latent_channels * lat_h * lat_w) return error.LatentSizeMismatch;
            self.setMemTag(.vae);
            // Denormalize onto a copy (`decode` must not modify the caller's latent).
            // ⚠️ `latent_formats.Flux.process_out` is `z / scale + SHIFT` — the shift
            // term has no SD analogue, and dropping it decodes a latent offset by
            // 0.116, which is a plausible image with a colour cast rather than an error.
            const x = try gpa.dupe(f32, latent);
            defer gpa.free(x);
            const inv = 1.0 / m.vae.cfg.scaling_factor;
            for (x) |*v| v.* = v.* * inv + m.vae.cfg.shift_factor;
            try note(progress, "decoding latent...\n", .{});
            const dec_start = std.Io.Clock.real.now(io);
            const planar = try self.decodePlanar(SdVae{ .vae = &m.vae }, x, lat_h, lat_w, o, progress);
            defer gpa.free(planar);
            try note(progress, "decoded in {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dec_start.nanoseconds)) / 1e9});
            const width = lat_w * 8;
            const height = lat_h * 8;
            return .{
                .rgb = try image.planarF32ToRgb8(gpa, planar, width, height),
                .width = width,
                .height = height,
            };
        }

        if (latent.len != wan_vae.latent_channels * lat_h * lat_w) return error.LatentSizeMismatch;

        // Device allocations here (VAE weights + decode scratch) are tagged VAE.
        // NO evictWeights — the DiT stays resident for the next queued image;
        // under a tight budget the backend's LRU cache streams the overflow
        // reactively.
        self.setMemTag(.vae);

        // Denormalize onto a copy rather than in place: `generate` is done with
        // its latent by now, but a caller driving the stages itself may not be,
        // and silently rescaling their buffer would be a trap. ~2 MB at 1120x1680.
        const x = try gpa.dupe(f32, latent);
        defer gpa.free(x);
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
        const planar = try self.decodePlanar(KreaVae{ .vae = &self.k().vae }, x, lat_h, lat_w, o, progress);
        defer gpa.free(planar);
        try note(progress, "decoded in {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dec_start.nanoseconds)) / 1e9});

        const width = lat_w * 8;
        const height = lat_h * 8;
        return .{ .rgb = try image.planarF32ToRgb8(gpa, planar, width, height), .width = width, .height = height };
    }

    /// Decode an already-denormalized planar `[c][lat_h][lat_w]` latent to planar
    /// `[3][8·lat_h][8·lat_w]` pixels, through the adaptive ladder: whole-image
    /// (fastest, seamless) → free VRAM and retry → GPU tiling (bounded footprint)
    /// → CPU tiling (the floor that cannot OOM). `v` is the family's adapter
    /// (`KreaVae` / `SdVae`), which supplies the per-backend tile context, the
    /// peak-VRAM estimate and the tile geometry — a whole-image decode is just a
    /// single tile covering the whole latent, so nothing else here is
    /// family-specific. Caller frees the result.
    fn decodePlanar(
        self: *Session,
        v: anytype,
        x: []const f32,
        lat_h: usize,
        lat_w: usize,
        o: DecodeOptions,
        progress: ?*std.Io.Writer,
    ) ![]f32 {
        const gpa = self.gpa;
        const io = self.io;
        // The whole-image decode is fastest and seamless, but its peak VRAM — in
        // particular the O(seq²) mid-block attention scores plane — grows with
        // image area and OOMs on large images. When it won't fit we decode in
        // overlapping tiles (bounded footprint, feather-blended seams) on the GPU
        // rather than crawling on the CPU; only if even a tile can't fit do we
        // drop to a CPU tiled decode. See models/vae_tiled.zig.
        const tp = @TypeOf(v).tiling;
        // Decode-path override (o.vae_decode). `force_cpu` short-circuits every
        // backend to CPU tiling; `skip_whole` starts at GPU tiling (no whole-image
        // attempt). `auto`/`whole` leave both false — whole-image is already the
        // first attempt, so the two behave identically.
        const force_cpu = o.vae_decode == .cpu_tiled;
        const skip_whole = o.vae_decode == .gpu_tiled or o.vae_decode == .cpu_tiled;
        if (force_cpu) {
            try note(progress, "vae decode: tiling on CPU (forced)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            const ct = v.cpuCtx(io, o.cancel);
            return vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, @TypeOf(ct).call);
        }
        if (self.cu_be) |b| {
            const wt = v.cudaCtx(b, o.cancel);
            const Whole = @TypeOf(wt);
            // Attempt ladder — whole-image (fastest, seamless) → GPU tiling
            // (bounded footprint) → CPU (guaranteed). Before EACH GPU retry free
            // JUST ENOUGH VRAM and try again, keeping the rest resident so we
            // reload as little as possible: drop LRU weights from THIS backend's
            // own cache first (the denoiser — dead for the rest of this image,
            // re-streams next image), and only when that can't cover the deficit
            // reach into the chat LLM's context. The freed amount escalates so a
            // big deficit converges in a few retries. OOM can arrive as
            // DeviceOutOfMemory OR as a cuBLASLt / cuDNN out-of-workspace error OR
            // as a post-OOM stream fault (see recoverableDecodeErr).
            // (skip_whole jumps straight to tiling.)
            try note(progress, "vae decode: mode={s} pinned={d}MB streamed={d}MB free={d}MB\n", .{
                @tagName(o.vae_decode), b.pinnedWeightBytes() >> 20, b.evictableWeightBytes() >> 20, b.ctx.memGetInfo().free >> 20,
            });
            var want: u64 = reclaim_chunk;
            // Free ~`wnt` bytes across this backend's own cache (LRU incl. the
            // now-dead denoiser) then the chat LLM, log it, and report the total
            // freed (0 ⇒ nothing left to free).
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
                // CUDA never materializes the mid-block scores plane — cuDNN's
                // fused SDPA under `.libs`, `be.attn`'s online softmax otherwise.
                const est = if (skip_whole) v.estimate(tp.tile, tp.tile, false) else v.estimate(lat_h, lat_w, false);
                const target = est + est / 10; // 110%
                const free_now = b.ctx.memGetInfo().free;
                try note(progress, "vae decode: est peak {d}MB, want free ≥ {d}MB (have {d}MB)\n", .{ est >> 20, target >> 20, free_now >> 20 });
                if (free_now < target) _ = try freeSome(b, o.reclaim, progress, io, target - free_now);
            }
            // Phase 1: whole-image with incremental eviction.
            if (!skip_whole) {
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    if (Whole.call(wt, gpa, io, x, lat_h, lat_w)) |p| {
                        return p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: whole-image OOM ({t}) → freeing VRAM\n", .{err});
                    if (try freeSome(b, o.reclaim, progress, io, want) == 0) break; // nothing left → tile
                    want *|= 2;
                }
            }
            // Phase 2: GPU tiling (bounded) with the same incremental eviction —
            // after the denoiser is evicted a tile easily fits, and it's far faster
            // than the CPU floor below.
            {
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    try note(progress, "vae decode: tiling on GPU ({d}² latent tiles)\n", .{tp.tile});
                    if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, wt, Whole.call)) |p| {
                        return p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: GPU tiling {s} ({t}) → freeing VRAM\n", .{ decodeStepReason(err), err });
                    if (try freeSome(b, o.reclaim, progress, io, want) == 0) break; // nothing left → CPU
                    want *|= 2;
                }
            }
            // Phase 3: CPU tiling — the guaranteed VRAM-can't-OOM floor (slow).
            try note(progress, "vae decode: GPU out of VRAM → CPU tiled decode (slow)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            const ct = v.cpuCtx(io, o.cancel);
            return vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, @TypeOf(ct).call);
        }
        if (self.gpu_ctx) |gc| {
            const wt = v.vkCtx(gc, o.cancel);
            const Whole = @TypeOf(wt);
            // Whole-image decode first. The Vulkan mid-block attention is now
            // query-tiled (flash), so this OOMs only when the conv activation
            // buffers don't fit. On OOM free JUST ENOUGH VRAM and retry, keeping
            // the rest resident: each round drops LRU weights from this context's
            // own cache first (the denoiser), and only when that can't cover it
            // reaches into the chat LLM's context. The freed amount escalates so
            // a large deficit converges fast. If nothing more can be freed, tile
            // on the GPU; if even a tile won't fit, decode on the CPU.
            // (skip_whole jumps straight to tiling.)
            try note(progress, "vae decode: mode={s} free={d}MB\n", .{ @tagName(o.vae_decode), gc.liveVram() >> 20 });
            {
                // Proactive pre-free to ~110% of the estimated first-attempt peak,
                // so we don't burn a full failed decode just to find the deficit.
                // Vulkan DOES materialize it, in f32 (see `sd_vae_gpu.attn`).
                const est = if (skip_whole) v.estimate(tp.tile, tp.tile, true) else v.estimate(lat_h, lat_w, true);
                const target = est + est / 10;
                const free_now = gc.liveVram();
                try note(progress, "vae decode: est peak {d}MB, want free >= {d}MB (have {d}MB)\n", .{ est >> 20, target >> 20, free_now >> 20 });
                if (free_now < target) {
                    _ = gc.evictToFree(target - free_now);
                    if (o.reclaim) |r| _ = r.call(r.ctx, target -| gc.liveVram());
                }
            }
            if (!skip_whole) {
                var want: u64 = reclaim_chunk;
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    if (Whole.call(wt, gpa, io, x, lat_h, lat_w)) |p| {
                        return p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: whole-image OOM ({t}) -> freeing VRAM\n", .{err});
                    const t0 = std.Io.Clock.real.now(io).nanoseconds;
                    const from_self = gc.evictToFree(want); // own resident weights first
                    var got = from_self;
                    if (got < want) if (o.reclaim) |r| {
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
                var wnt: u64 = reclaim_chunk;
                var round: usize = 0;
                while (round < max_reclaim_rounds) : (round += 1) {
                    try note(progress, "vae decode: tiling on GPU ({d}^2 latent tiles)\n", .{tp.tile});
                    if (vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, wt, Whole.call)) |p| {
                        return p;
                    } else |err| if (!recoverableDecodeErr(err)) return err else try note(progress, "vae decode: GPU tiling {s} ({t}) -> CPU tier\n", .{ decodeStepReason(err), err });
                    const t0 = std.Io.Clock.real.now(io).nanoseconds;
                    const from_self = gc.evictToFree(wnt);
                    var got = from_self;
                    if (got < wnt) if (o.reclaim) |r| {
                        got += r.call(r.ctx, wnt - got);
                    };
                    const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;
                    try note(progress, "vae decode: freed {d}MB (own {d}MB + llm {d}MB) for tiling in {d:.0}ms\n", .{ got >> 20, from_self >> 20, (got - from_self) >> 20, ms });
                    if (got == 0) break; // nothing left → CPU
                    wnt *|= 2;
                }
            }
            // Phase 3: CPU tiling — the guaranteed VRAM-can't-OOM floor (slow).
            try note(progress, "vae decode: falling back to the CPU tier (exact, slow)\n", .{});
            const saved = ops.matmul.gpu_dispatch;
            ops.matmul.gpu_dispatch = null;
            defer ops.matmul.gpu_dispatch = saved;
            const ct = v.cpuCtx(io, o.cancel);
            return vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, @TypeOf(ct).call);
        }
        // CPU-only: tile once the whole-image scores plane (f32) gets large,
        // so a big image doesn't try to allocate tens of GB of host RAM.
        // (skip_whole — e.g. gpu_tiled with no GPU — forces tiling.)
        const ct = v.cpuCtx(io, o.cancel);
        if (!skip_whole and attnPlaneBytes(lat_h, lat_w, 4) < (1 << 30))
            return @TypeOf(ct).call(ct, gpa, io, x, lat_h, lat_w);
        try note(progress, "vae decode: tiling on CPU\n", .{});
        return vae_tiled.decode(gpa, io, x, lat_h, lat_w, tp, ct, @TypeOf(ct).call);
    }

    pub fn generate(self: *Session, opts: Options, progress: ?*std.Io.Writer) !Image {
        const gpa = self.gpa;
        const io = self.io;
        if (opts.width % 16 != 0 or opts.height % 16 != 0) return error.SizeNotMultipleOf16;
        if (opts.steps < 1) return error.NoSteps;
        // Per-render, not per-session: the GUI snapshots a config per queued image.
        self.compat = opts.compatConfig();
        const lat_h = opts.height / 8;
        const lat_w = opts.width / 8;
        // Family-dependent: 16 latent channels for krea2's Wan VAE, 4 for SD's
        // AutoencoderKL. Hardcoding 16 here ran SD's 4-channel UNet correctly for
        // four steps and then failed in `decode` with `LatentSizeMismatch` — the
        // sampling loop only ever touches whole latents, so the mismatch surfaced as
        // far from its cause as possible.
        const lat_len = self.latentChannels() * lat_h * lat_w;
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

        // The sigma schedule is computed HERE, before the text encode, only because an
        // A1111 prompt schedule is indexed by step and so needs the step count up front.
        // ⚠️ That count is `sigmas.len - 1`, NOT `opts.steps`: `ddim_uniform` strides the
        // sigma table and `beta` de-duplicates rungs, so either can return a different
        // number than requested. Hoisting is safe — `scheduleWith` is pure and touches no
        // RNG, so the noise draw below is unchanged and the render stays bit-identical.
        const shift = self.resolvedShift(opts);
        const sigmas = try self.scheduleWith(gpa, opts.steps, shift, opts.scheduler);
        defer gpa.free(sigmas);
        const nsteps = sigmas.len - 1;
        if (nsteps != opts.steps) {
            try note(progress, "scheduler {t} produced {d} steps for a request of {d}\n", .{
                opts.scheduler orelse sampler.Scheduler.defaultFor(self.sigmaTable(shift)), nsteps, opts.steps,
            });
        }

        // Stage 1: text encoding (reusing the resident encoder).
        const enc_start = std.Io.Clock.real.now(io);
        const enc_opts: EncodeOptions = .{
            .encoder_f16 = opts.encoder_f16,
            .prompt_syntax = opts.prompt_syntax,
            .emphasis = opts.emphasis,
            .steps = nsteps,
            .cancel = opts.cancel,
        };
        var cond_pos = try self.encode(gpa, opts.prompt, enc_opts);
        defer cond_pos.deinit(gpa);
        var cond_neg: ?Cond = if (use_cfg) try self.encode(gpa, opts.negative, enc_opts) else null;
        defer if (cond_neg) |*c| c.deinit(gpa);
        // The variant count is worth reporting: an A1111 `[a|b]` or `[a:b:0.5]` silently
        // becomes several conditionings, and "1 variant" is the difference between a
        // schedule that took effect and syntax that was read as plain text.
        const n_variants = cond_pos.entryCount() + if (cond_neg) |*c| c.entryCount() - 1 else 0;
        try note(progress, "encoded prompt ({d} tokens{s}{s}) in {d:.1}s\n", .{
            cond_pos.seq,
            if (use_cfg) " + negative" else "",
            if (n_variants > 1) ", scheduled" else "",
            @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - enc_start.nanoseconds)) / 1e9,
        });
        if (n_variants > 1) try note(progress, "prompt schedule: {d} distinct conditionings\n", .{n_variants});

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
            const evictable = b.evictableWeightBytes();
            const room = free_now + evictable;
            const budget = if (opts.vram_budget > 0) @min(opts.vram_budget, room) else room;
            const pin_reserve: u64 = b.attn_scratch_budget + (512 << 20);
            // ⚠️ `pin_budget` is a TOTAL cap (`pinNew` tests
            // `pinned_bytes + size > pin_budget`), but `room` is what is ADDITIONALLY
            // reachable — live free VRAM plus our own evictable weights — and so
            // excludes whatever is already pinned. Adding the resident total back is
            // what keeps the two in the same units.
            //
            // Without this, every image after the first stops pinning entirely: a
            // queued image finds ~11.7 GB of DiT already pinned against a `room`-only
            // budget of ~1 GB, so `pinned_bytes > pin_budget` on the very first test
            // and the rest of the DiT streams on every step for the whole image.
            // Image 1 is unaffected (nothing pinned yet, so total == increment).
            //
            // Safe to be generous here: `pin_floor` is the physical backstop and is
            // still checked per pin, so a budget larger than live VRAM cannot
            // over-commit — it only stops the cap from being the binding constraint
            // when it should not be.
            b.pin_budget = b.pinnedWeightBytes() + (budget -| pin_reserve);
            // Same reserve as a LIVE floor for first-touch pinning: pinNew keeps
            // this much VRAM free, so pinning never eats the room the (lazily
            // allocated, per-block) attention scratch + activation workspace need.
            // pin_budget above is blind to the working set not yet allocated at
            // pin time; the floor is the physical backstop that makes streaming
            // actually fit — the whole point of a tight budget.
            b.pin_floor = pin_reserve;
            // ⚠️ `req` is what the caller ASKED for; `eff` is what physical room
            // allows and what actually governs. Printing only `req` next to a much
            // smaller `pin` reads as a bug ("16 GB budget, pinning 689 MB?") when
            // it is the `@min` above doing its job.
            std.log.info("[diff-vram] budget req={d}MB room={d}MB -> eff={d}MB · reserve={d}MB pin={d}MB (free={d}MB + evictable={d}MB)", .{
                opts.vram_budget >> 20, room >> 20, budget >> 20,
                pin_reserve >> 20,      b.pin_budget >> 20,
                free_now >> 20,         evictable >> 20,
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
            // (the next queued image's encode reuses it).
            //
            // ⚠️ **The test must discount the DiT already on the card, and getting
            // that wrong made this fire on every queued image.** On the FIRST image
            // the cache holds only the encoder, so "does dit + reserve fit in free?"
            // is right. On a SUBSEQUENT image the previous DiT is still pinned
            // (evictUnpinned below deliberately keeps it) — which is *why* free VRAM
            // is low — so counting the whole DiT again double-counts it against a
            // `free_now` that already excludes it. Observed: dit=13477 + reserve=2560
            // > free=2628 "true" with the DiT wholly resident, evicting an encoder
            // that fit perfectly well. Subtracting the resident part reduces the
            // second image to `reserve > free`, which is the real question.
            //
            // Pinned bytes are the right proxy for "resident DiT": the encoder and
            // the VAE are both left UNPINNED by design (see pin_budget above), so on
            // this path anything pinned is DiT. Clamped to `dit_bytes` so a stray
            // pin can never make the requirement negative.
            const dit_bytes = self.denoiserPayloadLen();
            const dit_resident = @min(dit_bytes, b.pinnedWeightBytes());
            const dit_to_place = dit_bytes - dit_resident;
            if (dit_to_place + pin_reserve > free_now) {
                // evictUnpinned (NOT evictWeights): keep a DiT pinned by a previous
                // queued image, drop only the (unpinned) encoder + any stray stream.
                const freed = b.evictUnpinned();
                // ⚠️ Says "unpinned weights", not "text encoder": on a subsequent
                // image the VAE is resident and unpinned too, so this drops that as
                // well (harmless — the decode is after sampling, by which point the
                // streaming DiT would have reclaimed it anyway). Naming only the
                // encoder made the line describe less than it did.
                //
                // The inequality is the one actually tested. The old line printed
                // the WHOLE DiT against free VRAM, which framed streaming-by-design
                // as a fit failure and implied dropping ~600 MB could make 13 GB fit.
                if (freed > 0) std.log.info("[diff-vram] dropped {d}MB of unpinned weights (encoder, + VAE if resident) so step 1 pins from the start — DiT still needs {d}MB of {d}MB ({d}MB already resident) + {d}MB reserve > {d}MB free", .{
                    freed >> 20,      dit_to_place >> 20, dit_bytes >> 20,
                    dit_resident >> 20, pin_reserve >> 20, free_now >> 20,
                });
            }
        }

        // Stage 2: flow-matching sampling (reusing the resident DiT). DiT weights
        // (streamed lazily during forward) + per-step attention scratch are tagged
        // DiT; the per-image working set (GPU session, activation workspace,
        // preview decode) is tagged `latent` below.
        self.setMemTag(.dit);
        // `sigmas`/`nsteps` were computed before the encode (see there); the initial
        // latent is scaled by `sigmas[0]`, which is 1.0 for krea2 (a bit-identical no-op)
        // and ~14.6 for SD.

        const x = try gpa.alloc(f32, lat_len);
        defer gpa.free(x);
        // Resume: restore the suspended latent instead of drawing fresh noise
        // (the schedule + conditioning above are recomputed identically). A
        // length mismatch (shouldn't happen — resume reuses the same opts) falls
        // back to a fresh draw rather than a bad copy.
        if (opts.resume_from) |r| {
            if (r.latent.len == x.len) @memcpy(x, r.latent) else {
                sampler.fillNoiseFrom(x, opts.seed, self.compat.noise_src);
                self.scaleInitialNoise(x, sigmas[0]);
            }
        } else {
            sampler.fillNoiseFrom(x, opts.seed, self.compat.noise_src);
            self.scaleInitialNoise(x, sigmas[0]);
        }

        {
            const v = try gpa.alloc(f32, lat_len);
            defer gpa.free(v);

            // The higher-order sampler's per-render state (multistep history + the
            // Brownian noise path). Built HERE, between the initial-noise scaling and
            // the denoiser, because both orderings matter:
            //
            //  - **After** `scaleInitialNoise`: ComfyUI scales the starting latent by
            //    the *unoffset* first sigma (its `noise_scaling` runs before the
            //    sampler function, which offsets a clone), and `init` mutates
            //    `sigmas[0]`.
            //  - **Before** `self.denoiser(...)`: that precomputes a timestep vector
            //    per schedule entry, so it has to see the offset value or step 0 falls
            //    off its own cache.
            var sde: ?sampler.SdeStepper = if (opts.sampler.isSde()) try .init(
                gpa,
                sigmas,
                lat_len,
                self.parameterization(),
                .{
                    .eta = opts.sde_eta,
                    .s_noise = opts.sde_s_noise,
                    .solver = if (opts.sampler == .dpmpp_2m_sde) .midpoint else .heun,
                    // ComfyUI seeds the Brownian path from the render's own seed
                    // (`extra_args["seed"]`), the same one that drew the latent.
                    .seed = opts.seed,
                    // ⚠️ And the same generator, which is the half that is easy to miss:
                    // A1111's pinned k-diffusion builds the tree on the CUDA tensor's
                    // device, so its per-node draws are Philox too. Wiring only the
                    // initial latent would have made euler reproduce and left every SDE
                    // render wrong, with nothing failing.
                    .noise_src = self.compat.noise_src,
                },
                shift,
            ) else null;
            defer if (sde) |*s| s.deinit();
            // Restore the multistep history on a resume, or the first step after it is
            // silently first-order (see `Snapshot.sde_old_denoised`).
            if (sde) |*s| if (opts.resume_from) |r| if (r.sde_old_denoised) |old| {
                if (old.len == lat_len) s.restore(old, r.sde_h_last);
            };

            // The per-image denoiser: text fusion, rope table, timestep vectors and
            // activation workspace, built once here (they depend on the prompt +
            // resolution) and reused every step. The DiT WEIGHTS stay cached in the
            // backend across images, independently of this.
            var den = try self.denoiser(gpa, cond_pos, cond_neg, opts.cfg, lat_h, lat_w, sigmas);
            defer den.deinit(gpa);

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
            //
            // krea2 ONLY: TAEHV is a 16-channel Wan approx-VAE, so it cannot decode an
            // SD latent at all — the SD family's equivalent is TAESD/TAESDXL, a
            // different (4-channel, image) decoder that isn't implemented here. Leaving
            // `taehv_dec` null makes the `method == 2` request degrade to latent2rgb
            // through the existing path, which is the only correct preview for SD today.
            var taew_st: ?safetensors.SafeTensors = null;
            defer if (taew_st) |*s| s.deinit();
            var taehv_dec: ?taehv_mod.Decoder = null;
            defer if (taehv_dec) |*d| d.deinit();
            if (preview_active and self.family() == .krea2) if (opts.taew_path) |tp| {
                if (safetensors.SafeTensors.open(gpa, io, tp)) |tst| {
                    taew_st = tst;
                    if (taehv_mod.Decoder.load(gpa, &taew_st.?)) |d| {
                        taehv_dec = d;
                        try note(progress, "preview: taew2_1 approx-VAE ready\n", .{});
                    } else |err| try note(progress, "taew2_1 load failed ({t}); latent2rgb preview\n", .{err});
                } else |err| try note(progress, "taew2_1 open failed ({t}); latent2rgb preview\n", .{err});
            };

            const sampling_start = std.Io.Clock.real.now(io);
            const start_step = if (opts.resume_from) |r| @min(r.step, nsteps) else 0;
            for (start_step..nsteps) |i| {
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
                            so.* = .{
                                .latent = try gpa.dupe(f32, x),
                                .step = i,
                                // A multistep sampler's history is part of the state a
                                // bit-identical resume needs; null for euler.
                                .sde_old_denoised = if (sde) |*s|
                                    (if (s.have_old) try gpa.dupe(f32, s.old_denoised) else null)
                                else
                                    null,
                                .sde_h_last = if (sde) |*s| s.h_last else 0,
                            };
                            return error.Paused;
                        }
                        return error.Canceled;
                    },
                };
                const start = std.Io.Clock.real.now(io);
                try den.predictAt(gpa, v, x, sigmas[i], i, opts.cancel);
                if (sde) |*s| try s.step(x, v, i) else sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
                const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - start.nanoseconds)) / 1e6;
                try note(progress, "step {d}/{d}  sigma {d:.3} -> {d:.3}  ({d:.1}s)\n", .{ i + 1, nsteps, sigmas[i], sigmas[i + 1], ms / 1000.0 });
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
                        // Denoised (x0) estimate = x_i - sigma_i*v. `eulerStep` above
                        // set x to x_{i+1} = x_i + (sigma_{i+1} - sigma_i)*v, so the
                        // clean estimate in terms of the post-step latent is
                        // x - sigma_{i+1}*v (collapses to x on the final step where
                        // sigma_{i+1}==0).
                        //
                        // That reconstruction is only valid for an Euler step. An SDE
                        // stepper's latent is not `x_i + dt*v` (it has an exponential
                        // drift term and injected noise), so reading the estimate back
                        // out of it would preview a differently-scaled image that
                        // *looks* plausible. It keeps the same quantity to hand.
                        const x0: []const f32 = if (sde) |*s|
                            s.denoised
                        else if (preview_x0) |px0| blk: {
                            const s_next = sigmas[i + 1];
                            for (px0, x, v) |*o, xi, vi| o.* = xi - s_next * vi;
                            break :blk px0;
                        } else x;
                        if (method == 2) if (taehv_dec) |*d| taew_blk: {
                            const live_ds: usize = if (opts.preview_live) |lp| lp.ds.load(.acquire) else opts.preview_ds;
                            const ds = clampDs(live_ds, lat_h, lat_w);
                            const th = lat_h / ds;
                            const tw = lat_w / ds;
                            const small = downsampleLatent(gpa, x0, wan_vae.latent_channels, lat_h, lat_w, ds) catch break :taew_blk;
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
                            self.latentPreviewInto(ps, x0, lat_h, lat_w);
                            pv = .{ .rgb = ps, .width = lat_w, .height = lat_h };
                        };
                    }
                    p.step(p.ctx, i + 1, nsteps, pv);
                }
            }
            const sampling_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - sampling_start.nanoseconds)) / 1e9;
            try note(progress, "sampling {d} steps in {d:.1}s ({d:.2}s/step)\n", .{ nsteps, sampling_s, sampling_s / @as(f64, @floatFromInt(nsteps)) });
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

        var img = try self.decode(x, lat_h, lat_w, .{
            .vae_decode = opts.vae_decode,
            .cancel = opts.cancel,
            .reclaim = opts.reclaim,
        }, progress);
        errdefer img.deinit(gpa);
        try note(progress, "total time {d:.1}s\n", .{@as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - total_start.nanoseconds)) / 1e9});
        return img;
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

/// Box-average a planar [c][h][w] latent down by integer factor `f`
/// (→ [c][h/f][w/f]) so the taew preview decode stays cheap.
///
/// ⚠️ `c` is a parameter, not `wan_vae.latent_channels`: hardcoding krea2's 16 read
/// four times past the end of an SD latent, which crashed the GUI's first previewed
/// SD render (`index out of bounds: index 0, len 0`) rather than producing a bad
/// preview.
fn downsampleLatent(gpa: std.mem.Allocator, x: []const f32, c: usize, h: usize, w: usize, f: usize) ![]f32 {
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

test "the live preview follows the family's own latent format" {
    // Regression: the per-step preview was krea2-only — `downsampleLatent` hardcoded 16
    // channels and the latent2rgb call went straight to `wan_vae` — so the first
    // previewed SD render read four planes past the end of a 4-channel latent and
    // panicked mid-generation (`index out of bounds: index 0, len 0`), for BOTH SD1.5
    // and SDXL. Only the union's tag is read here, so an undefined payload is safe and
    // this stays a fast CPU test.
    const gpa = std.testing.allocator;
    const lat_h = 4;
    const lat_w = 6;
    const rgb = try gpa.alloc(u8, lat_h * lat_w * 3);
    defer gpa.free(rgb);

    var sess: Session = undefined;
    var prev: [4][]u8 = undefined;
    inline for (.{ Family.krea2, Family.sd15, Family.sdxl, Family.zimage }, 0..) |fam, fi| {
        sess.models = switch (fam) {
            .krea2 => .{ .krea2 = undefined },
            .sd15 => .{ .sd15 = undefined },
            .sdxl => .{ .sdxl = undefined },
            .zimage => .{ .zimage = undefined },
        };
        const ch: usize = switch (fam) {
            .krea2 => wan_vae.latent_channels,
            .zimage => zimage.latent_channels,
            .sd15, .sdxl => sd_vae.latent_channels,
        };
        // Exactly the family's channel count — a read one plane past the end is an
        // out-of-bounds panic, which is what the bug was.
        const z = try gpa.alloc(f32, ch * lat_h * lat_w);
        defer gpa.free(z);
        for (z, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
        sess.latentPreviewInto(rgb, z, lat_h, lat_w);
        prev[fi] = try gpa.dupe(u8, rgb);
    }
    defer for (prev) |p| gpa.free(p);
    // Each family uses its OWN factors: sharing SD1.5's for SDXL is not a crash, just a
    // preview with plausible structure and wrong colours. krea2 and Z-Image are the
    // pair that matters most here — both are 16-channel, so the wrong matrix is not
    // even an out-of-bounds read, just quietly wrong colours.
    try std.testing.expect(!std.mem.eql(u8, prev[1], prev[2]));
    try std.testing.expect(!std.mem.eql(u8, prev[0], prev[3]));
}

test "downsampleLatent box-averages every plane of a non-krea2 latent" {
    const gpa = std.testing.allocator;
    const c = sd_vae.latent_channels;
    const h = 4;
    const w = 4;
    const x = try gpa.alloc(f32, c * h * w);
    defer gpa.free(x);
    // Plane `ch` is filled with its own index so a channel mixup shows up as a value.
    for (0..c) |ch| for (x[ch * h * w ..][0 .. h * w]) |*v| {
        v.* = @floatFromInt(ch);
    };
    const small = try downsampleLatent(gpa, x, c, h, w, 2);
    defer gpa.free(small);
    try std.testing.expectEqual(@as(usize, c * 2 * 2), small.len);
    for (0..c) |ch| for (small[ch * 4 ..][0..4]) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(ch)), v, 1e-6);
    };
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

test "the SD tile adapter transposes both ways at every latent width" {
    // `vae_tiled` and `Session.decode` speak planar `[c][h][w]`; the SD decoders
    // speak channel-last `[h*w][c]`. `sdTile` is the single seam between them, and
    // it is the seam because getting this wrong is **rms-preserving** — every value
    // survives, only its position moves — which is how the layout bug that reached a
    // rendered image stayed invisible to every magnitude check (see "the sampler's
    // planar latent survives a round trip through the UNet's layout").
    //
    // ⚠️ **Run at 4 AND 16 channels.** The same decoder body serves SD's 4-channel
    // latent and Z-Image's 16-channel Flux latent, and `sdTile` used to take the
    // count from `sd_vae.latent_channels` — so it transposed the first quarter of a
    // Z-Image latent and read past it for the rest. A 4-channel-only test could not
    // see that (it was the hardcoded value), and the render it produced was a band of
    // colour noise over flat grey rather than an error.
    //
    // The dummy decoder asserts the layout it is HANDED and emits a distinct known
    // pattern, so the two transpositions are pinned independently: a test that only
    // checked the round trip would pass with both of them wrong.
    const gpa = std.testing.allocator;
    const th = 2;
    const tw = 3;
    const n = th * tw;

    inline for (.{ sd_vae.latent_channels, zimage.latent_channels }) |c_in| {
        const sub = try gpa.alloc(f32, c_in * n); // planar [c][th][tw]
        defer gpa.free(sub);
        for (0..c_in) |c| {
            for (0..n) |p| sub[c * n + p] = @floatFromInt(c * 1000 + p);
        }

        const Dummy = struct {
            fn inner(_: @This(), a: std.mem.Allocator, z: []const f32, ith: usize, itw: usize) anyerror![]f32 {
                const nn = ith * itw;
                // Handed channel-last: a position's channels are adjacent.
                try std.testing.expectEqual(nn * c_in, z.len);
                for (0..nn) |p| {
                    for (0..c_in) |c| {
                        try std.testing.expectEqual(@as(f32, @floatFromInt(c * 1000 + p)), z[p * c_in + c]);
                    }
                }
                // Emit channel-last pixels, the shape a real SD decoder returns.
                const pn = nn * sd_vae.spatial_scale * sd_vae.spatial_scale;
                const out = try a.alloc(f32, pn * 3);
                for (0..pn) |p| {
                    for (0..3) |c| out[p * 3 + c] = @floatFromInt(c * 100000 + p);
                }
                return out;
            }
        };

        const planar = try sdTile(gpa, sub, th, tw, Dummy{}, Dummy.inner);
        defer gpa.free(planar);

        const pn = n * sd_vae.spatial_scale * sd_vae.spatial_scale;
        try std.testing.expectEqual(pn * 3, planar.len);
        for (0..3) |c| {
            for (0..pn) |p| {
                try std.testing.expectEqual(@as(f32, @floatFromInt(c * 100000 + p)), planar[c * pn + p]);
            }
        }
    }
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
    // A device decode that came back non-finite: the CPU tier is exact, so the ladder
    // must reach it rather than hand back a white image.
    try std.testing.expect(recoverableDecodeErr(error.GpuDecodeNonFinite));
    // Cancellation and structural errors must propagate, never be masked by a
    // silent CPU fallback.
    try std.testing.expect(!recoverableDecodeErr(error.Canceled));
    try std.testing.expect(!recoverableDecodeErr(error.SizeNotMultipleOf16));
}

const test_gate = @import("tp_models").test_gate;

/// The checkpoints the composed-stages test needs. Defaults, so a machine with
/// the standard `models/` layout runs it and any other self-skips.
const test_paths = struct {
    const dit = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    const te = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    const vae = "models/vae/krea2RealVae_v10.safetensors";
};

test "generate composed from the public stages is bit-identical to Session.generate" {
    // The acceptance test for the stage split: `encode` / `schedule` / `denoiser`
    // + `predict` / `decode` must be able to reproduce `generate` exactly, or they
    // are not the same computation and a caller driving them (level 2, img2img, a
    // custom sampler) is measuring something else.
    //
    // Small and CPU: 128² is 64 DiT tokens, and the CPU path is the one the
    // measurement harnesses anchor on.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    inline for (.{ test_paths.dit, test_paths.te, test_paths.vae }) |p| try test_gate.requireModelFile(io, p);

    const opts: Options = .{
        .prompt = "a copper teapot on a windowsill",
        .width = 128,
        .height = 128,
        .steps = 2,
        .cfg = 1.0,
        .seed = 4242,
        .backend = .cpu,
        .dit_path = test_paths.dit,
        .text_encoder_path = test_paths.te,
        .vae_path = test_paths.vae,
    };
    const lat_h = opts.height / 8;
    const lat_w = opts.width / 8;
    const lat_len = wan_vae.latent_channels * lat_h * lat_w;

    var sess = try Session.init(io, gpa, opts, null);
    defer sess.deinit();

    var whole = try sess.generate(opts, null);
    defer whole.deinit(gpa);

    // The same image, driven stage by stage — this is the reference usage.
    var cond = try sess.encode(gpa, opts.prompt, .{ .encoder_f16 = opts.encoder_f16 });
    defer cond.deinit(gpa);

    const sigmas = try schedule(gpa, opts.steps, opts.shift);
    defer gpa.free(sigmas);
    try std.testing.expectEqual(opts.steps + 1, sigmas.len);

    const x = try gpa.alloc(f32, lat_len);
    defer gpa.free(x);
    // Through `compatConfig` rather than the default-source form: this test's whole claim
    // is that the stages reproduce `generate`, so it has to draw from whatever generator
    // `generate` would have used.
    sampler.fillNoiseFrom(x, opts.seed, opts.compatConfig().noise_src);

    const v = try gpa.alloc(f32, lat_len);
    defer gpa.free(v);

    var den = try sess.denoiser(gpa, cond, null, opts.cfg, lat_h, lat_w, sigmas);
    defer den.deinit(gpa);

    // The one-shot `Session.predict` — level 2's primitive — must agree with the
    // loop's own first forward. If it does not, every level-2 number describes a
    // model nobody renders with.
    const v_oneshot = try gpa.alloc(f32, lat_len);
    defer gpa.free(v_oneshot);
    try sess.predict(gpa, v_oneshot, x, lat_h, lat_w, sigmas[0], cond);

    for (0..opts.steps) |i| {
        try den.predict(gpa, v, x, sigmas[i], null);
        if (i == 0) try std.testing.expectEqualSlices(f32, v, v_oneshot);
        sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
    }

    var staged = try sess.decode(x, lat_h, lat_w, .{}, null);
    defer staged.deinit(gpa);

    try std.testing.expectEqual(whole.width, staged.width);
    try std.testing.expectEqual(whole.height, staged.height);
    try std.testing.expectEqualSlices(u8, whole.rgb, staged.rgb);
}

test "decode does not modify the caller's latent" {
    // `generate` is finished with its latent when it decodes, so denormalizing in
    // place was safe there; a caller holding the latent for a drift curve is not,
    // and a silent rescale would corrupt every subsequent step.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    inline for (.{ test_paths.dit, test_paths.te, test_paths.vae }) |p| try test_gate.requireModelFile(io, p);

    const opts: Options = .{
        .prompt = "x",
        .width = 128,
        .height = 128,
        .backend = .cpu,
        .dit_path = test_paths.dit,
        .text_encoder_path = test_paths.te,
        .vae_path = test_paths.vae,
    };
    var sess = try Session.init(io, gpa, opts, null);
    defer sess.deinit();

    const lat_h = opts.height / 8;
    const lat_w = opts.width / 8;
    const x = try gpa.alloc(f32, wan_vae.latent_channels * lat_h * lat_w);
    defer gpa.free(x);
    sampler.fillNoise(x, 7);
    const before = try gpa.dupe(f32, x);
    defer gpa.free(before);

    var img = try sess.decode(x, lat_h, lat_w, .{}, null);
    defer img.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 128), img.width);
    try std.testing.expectEqualSlices(f32, before, x);
}

test "a CFG denoiser needs a negative conditioning" {
    // The two go together; accepting cfg != 1 with no negative cond would
    // dereference null inside the sampling loop.
    const gpa = std.testing.allocator;
    var sess: Session = undefined;
    sess.gpa = gpa;
    sess.gpu_ctx = null;
    sess.cu_be = null;
    try std.testing.expectError(
        error.CfgNeedsNegativeCond,
        sess.denoiser(gpa, .{ .data = &.{}, .seq = 0 }, null, 2.5, 16, 16, &.{ 1.0, 0.0 }),
    );
}

test "Container picks the reader by magic, not by extension" {
    // A misnamed checkpoint used to reach the wrong parser and report
    // `InvalidHeader`, which says nothing about the real problem. Sniffing means the
    // name is irrelevant; only the bytes decide.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // A minimal valid safetensors, deliberately named `.gguf`.
    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    ;
    var st_bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, st_bytes[0..8], header.len, .little);
    @memcpy(st_bytes[8..][0..header.len], header);
    const vals = [2]f32{ 1.5, -2.5 };
    @memcpy(st_bytes[8 + header.len ..], std.mem.sliceAsBytes(&vals));
    {
        const f = try tmp.dir.createFile(io, "lying.gguf", .{ .truncate = true });
        defer f.close(io);
        var buf: [64]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.interface.writeAll(&st_bytes);
        try w.interface.flush();
    }

    var c = try Container.openIn(gpa, io, tmp.dir, "lying.gguf");
    defer c.deinit();
    try std.testing.expect(c == .safetensors);
    try std.testing.expectEqual(@as(usize, 1), c.store().count());
    try std.testing.expectEqual(@as(usize, 8), c.payloadLen());
}

test "family detection reads the denoiser's own tensor names" {
    // Name-based rather than flag-based on purpose: a mistyped `--arch` in front of a
    // measurement is exactly the kind of input error the whole harness exists to rule
    // out. Both spellings of each family must resolve — a full single-file checkpoint
    // keeps the LDM container prefix, and ggufy's model-only GGUF strips it.
    const gpa = std.testing.allocator;

    // ⚠️ The SDXL cases carry the SD1.5 stem **as well**, because that is what a real
    // SDXL checkpoint looks like: both are LDM UNets with the same `input_blocks.0.0`.
    // So these two cases are the ones that pin the *order* of the checks — testing SDXL
    // with `label_emb` alone would pass even if the stem were read first, and every SDXL
    // checkpoint would then load as SD1.5 and fail on a missing fourth level.
    const cases = [_]struct { names: []const []const u8, want: Family }{
        .{ .names = &.{"model.diffusion_model.input_blocks.0.0.weight"}, .want = .sd15 },
        .{ .names = &.{"input_blocks.0.0.weight"}, .want = .sd15 },
        .{ .names = &.{ "model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.label_emb.0.0.weight" }, .want = .sdxl },
        .{ .names = &.{ "input_blocks.0.0.weight", "label_emb.0.0.weight" }, .want = .sdxl },
        .{ .names = &.{"model.diffusion_model.blocks.0.attn.wq.weight"}, .want = .krea2 },
        .{ .names = &.{"blocks.0.attn.wq.weight"}, .want = .krea2 },
        // ⚠️ Z-Image needs BOTH tensors, which is ComfyUI's own test:
        // `cap_embedder.1.weight` alone is shared with stock Lumina 2 and OmniGen2,
        // and the `noise_refiner` qk-norm is what says this is the NextDiT shape.
        // These cases therefore also pin that a `cap_embedder` on its own is NOT
        // enough (the negative case below).
        .{ .names = &.{ "model.diffusion_model.cap_embedder.1.weight", "model.diffusion_model.noise_refiner.0.attention.k_norm.weight" }, .want = .zimage },
        .{ .names = &.{ "cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight" }, .want = .zimage },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        var fbs = std.Io.Writer.fixed(&buf);
        try fbs.writeByte('{');
        for (c.names, 0..) |nm, i| {
            if (i > 0) try fbs.writeByte(',');
            try fbs.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", .{nm});
        }
        try fbs.writeByte('}');
        const header = fbs.buffered();
        const file = try gpa.alloc(u8, 8 + header.len + 8);
        defer gpa.free(file);
        std.mem.writeInt(u64, file[0..8], header.len, .little);
        @memcpy(file[8..][0..header.len], header);
        @memset(file[8 + header.len ..], 0);

        var st = try safetensors.SafeTensors.initFromSlice(gpa, file);
        defer st.deinit();
        try std.testing.expectEqual(c.want, try detectFamily(.{ .safetensors = &st }));
        // And the component resolver finds the denoiser under whichever spelling the
        // file used, handing back a store in which it sits at the root.
        var r = try resolveComponent(gpa, c.want, .denoiser, .{ .safetensors = &st }, null, false);
        defer if (r.view) |v| {
            v.deinit(gpa);
            gpa.destroy(v);
        };
        const probe = (try componentSpec(c.want, .denoiser)).probes[0];
        try std.testing.expect(r.store.get(probe) != null);
    }

    // A checkpoint of neither family is an error, not a default: silently treating an
    // unknown architecture as krea2 would fail deep inside a weight load.
    {
        const header = "{\"nonsense.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        var file: [8 + header.len + 8]u8 = undefined;
        std.mem.writeInt(u64, file[0..8], header.len, .little);
        @memcpy(file[8..][0..header.len], header);
        @memset(file[8 + header.len ..], 0);
        var st = try safetensors.SafeTensors.initFromSlice(gpa, &file);
        defer st.deinit();
        try std.testing.expectError(error.UnknownArchitecture, detectFamily(.{ .safetensors = &st }));
    }

    // ⚠️ `cap_embedder.1.weight` on its OWN is not Z-Image: stock Lumina 2 and
    // OmniGen2 have it too. This case is what gives the two-tensor probe teeth —
    // without it, a single-tensor test would pass either way.
    {
        const header = "{\"cap_embedder.1.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        var file: [8 + header.len + 8]u8 = undefined;
        std.mem.writeInt(u64, file[0..8], header.len, .little);
        @memcpy(file[8..][0..header.len], header);
        @memset(file[8 + header.len ..], 0);
        var st = try safetensors.SafeTensors.initFromSlice(gpa, &file);
        defer st.deinit();
        try std.testing.expectError(error.UnknownArchitecture, detectFamily(.{ .safetensors = &st }));
    }
}

test "each family's schedule shift and defaulted component paths are its own" {
    // ⚠️ Three constants that produce a plausible image when wrong, so none of them
    // can be left to a shared default:
    //
    //  - the SHIFT. Z-Image trained on 3.0 and krea2 on 1.15, and `Options.shift`
    //    defaults to krea2's. At 8 steps the wrong one is the difference between a
    //    finished image and a smeared one — a valid schedule with the steps in the
    //    wrong places, not an error.
    //  - the two DEFAULTED component paths. `Options`' are krea2's files, and
    //    handing krea2's Qwen3-**VL** encoder to Z-Image resolves to nothing (its
    //    language model is nested a level deeper), which reports
    //    `ComponentNotInCheckpoint` — the exact failure a joined SD1.5 checkpoint hit.
    try std.testing.expectEqual(sampler.default_shift, defaultShift(.krea2));
    try std.testing.expectEqual(@as(f32, 3.0), defaultShift(.zimage));
    try std.testing.expect(defaultShift(.krea2) != defaultShift(.zimage));

    // Only an EXPLICIT request overrides the family's own value.
    var sess: Session = undefined;
    sess.models = .{ .zimage = undefined };
    try std.testing.expectEqual(@as(f32, 3.0), sess.resolvedShift(.{ .prompt = "" }));
    try std.testing.expectEqual(@as(f32, 1.7), sess.resolvedShift(.{ .prompt = "", .shift = 1.7, .explicit_shift = true }));
    sess.models = .{ .krea2 = undefined };
    try std.testing.expectEqual(sampler.default_shift, sess.resolvedShift(.{ .prompt = "" }));

    // Z-Image names its own side files; every other family keeps using `Options`'.
    try std.testing.expect(defaultComponentPath(.zimage, .conditioner).len > 0);
    try std.testing.expect(defaultComponentPath(.zimage, .decoder).len > 0);
    try std.testing.expectEqual(@as(usize, 0), defaultComponentPath(.krea2, .conditioner).len);
}

test "container style is orthogonal to family: bundled, split, and explicit override" {
    // The rule this pins, which an earlier version got wrong by tying container style
    // to the architecture: any family ships joined OR split, an explicitly requested
    // file overrides a bundled component, and a *defaulted* path never does.
    const gpa = std.testing.allocator;

    const Builder = struct {
        /// A one-tensor safetensors file under `name`.
        fn file(buf: []u8, name: []const u8) ![]u8 {
            var hdr: [512]u8 = undefined;
            const header = try std.fmt.bufPrint(&hdr, "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}}}", .{name});
            std.mem.writeInt(u64, buf[0..8], header.len, .little);
            @memcpy(buf[8..][0..header.len], header);
            @memset(buf[8 + header.len ..][0..8], 0);
            return buf[0 .. 8 + header.len + 8];
        }
    };

    // A "bundled" primary: the SD conditioner nested under the LDM prefix.
    var b1: [640]u8 = undefined;
    var bundled = try safetensors.SafeTensors.initFromSlice(gpa, try Builder.file(&b1, "cond_stage_model.transformer.text_model.final_layer_norm.weight"));
    defer bundled.deinit();
    // A separate conditioner file, bare HF spelling.
    var b2: [640]u8 = undefined;
    var side = try safetensors.SafeTensors.initFromSlice(gpa, try Builder.file(&b2, "text_model.final_layer_norm.weight"));
    defer side.deinit();

    const probe = (try componentSpec(.sd15, .conditioner)).probes[0];

    // 1. Bundled only: found inside the primary, through a prefix view.
    {
        var r = try resolveComponent(gpa, .sd15, .conditioner, .{ .safetensors = &bundled }, null, false);
        defer if (r.view) |v| {
            v.deinit(gpa);
            gpa.destroy(v);
        };
        try std.testing.expect(r.view != null);
        try std.testing.expect(r.store.get(probe) != null);
        // The view's namespace is the stripped one, or a consumer enumerating it
        // (the capture sanity gate) would miss on every name.
        try std.testing.expectEqual(@as(usize, 1), r.store.names().len);
        try std.testing.expectEqualStrings(probe, r.store.names()[0]);
    }

    // 2. Split: the primary has no conditioner, the side file does.
    {
        var unet: [640]u8 = undefined;
        var primary = try safetensors.SafeTensors.initFromSlice(gpa, try Builder.file(&unet, "model.diffusion_model.input_blocks.0.0.weight"));
        defer primary.deinit();
        var r = try resolveComponent(gpa, .sd15, .conditioner, .{ .safetensors = &primary }, .{ .safetensors = &side }, false);
        defer if (r.view) |v| {
            v.deinit(gpa);
            gpa.destroy(v);
        };
        try std.testing.expect(r.store.get(probe) != null);
    }

    // 3. Explicit override: the primary DOES carry one, and the flag still wins.
    {
        const r = try resolveComponent(gpa, .sd15, .conditioner, .{ .safetensors = &bundled }, .{ .safetensors = &side }, true);
        defer if (r.view) |v| {
            v.deinit(gpa);
            gpa.destroy(v);
        };
        // Which prefix resolved is how we know WHERE it came from: the side file
        // spells it `text_model.`, the bundle `cond_stage_model.transformer.text_model.`.
        // (Both are candidate spellings, so "no view" would not have distinguished
        // them — the first version of this test asserted that and was simply wrong.)
        try std.testing.expectEqualStrings("text_model.", r.view.?.prefix);
    }

    // 4. A *defaulted* side file does NOT override a bundled component. (This is the
    //    case that broke a joined SD checkpoint: the default krea2 encoder path was
    //    treated as a request, so the resolver looked for CLIP inside qwen3.)
    {
        const r = try resolveComponent(gpa, .sd15, .conditioner, .{ .safetensors = &bundled }, .{ .safetensors = &side }, false);
        defer if (r.view) |v| {
            v.deinit(gpa);
            gpa.destroy(v);
        };
        try std.testing.expectEqualStrings("cond_stage_model.transformer.text_model.", r.view.?.prefix);
    }

    // 5. Nowhere at all is an error, not a silent fallback.
    {
        var empty: [640]u8 = undefined;
        var nothing = try safetensors.SafeTensors.initFromSlice(gpa, try Builder.file(&empty, "unrelated.weight"));
        defer nothing.deinit();
        try std.testing.expectError(
            error.ComponentNotInCheckpoint,
            resolveComponent(gpa, .sd15, .conditioner, .{ .safetensors = &nothing }, null, false),
        );
    }
}

test "the sampler's planar latent survives a round trip through the UNet's layout" {
    // ⚠️ The bug this pins reached a rendered image. The sampler works in planar
    // `[c][h][w]` (krea2's DiT and both VAEs do) while `sd_unet.forward` works in
    // channel-last `[h*w][c]`, and feeding one to the other is **rms-preserving**:
    // every value survives, only its position moves. Per-step magnitudes therefore
    // matched the reference to a fraction of a percent while the image became a
    // periodic streak pattern — and the UNet parity tests could not see it, because
    // they transpose explicitly before calling `forward`.
    const gpa = std.testing.allocator;
    const ch = 4;
    const h = 3;
    const w = 5;
    const plane = h * w;

    const planar = try gpa.alloc(f32, ch * plane);
    defer gpa.free(planar);
    for (planar, 0..) |*v, i| v.* = @floatFromInt(i);

    // Forward transposition, exactly as `predictSd` does it.
    const cl = try gpa.alloc(f32, ch * plane);
    defer gpa.free(cl);
    for (0..plane) |px| {
        for (0..ch) |c| cl[px * ch + c] = planar[c * plane + px];
    }
    // Channel-last means a position's channels are adjacent.
    try std.testing.expectEqual(planar[0], cl[0]);
    try std.testing.expectEqual(planar[plane], cl[1]);
    try std.testing.expectEqual(planar[2 * plane], cl[2]);

    const back = try gpa.alloc(f32, ch * plane);
    defer gpa.free(back);
    channelLastToPlanar(back, cl, ch, plane);
    try std.testing.expectEqualSlices(f32, planar, back);

    // And the property that makes the bug so quiet: the wrong layout has the SAME
    // norm. Any test that only checks magnitudes passes on scrambled data.
    var n_planar: f64 = 0;
    var n_cl: f64 = 0;
    for (planar, cl) |a, b| {
        n_planar += a * a;
        n_cl += b * b;
    }
    try std.testing.expectEqual(n_planar, n_cl);
}

test "a scheduled prompt deduplicates by text and indexes every step" {
    // The dedup is the load-bearing part: `[a|b]` contributes a schedule entry per STEP
    // (upstream's `collect_steps` does), so without it a 35-step render would encode 35
    // conditionings for 2 prompts and build 35 device sessions per branch.
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    {
        const plan = try planSchedule(arena.allocator(), gpa, "[a|b] cat", 35);
        defer gpa.free(plan.at);
        try std.testing.expectEqual(@as(usize, 2), plan.texts.len);
        try std.testing.expectEqual(@as(usize, 35), plan.at.len);
        // Alternating, 1-based: step 1 -> "a", step 2 -> "b", …
        try std.testing.expectEqualStrings("a cat", plan.texts[0]);
        try std.testing.expectEqualStrings("b cat", plan.texts[1]);
        for (plan.at, 0..) |ix, step| try std.testing.expectEqual(@as(u8, @intCast(step % 2)), ix);
    }
    {
        // Prompt editing: one boundary, so two texts and a step index that switches once.
        const plan = try planSchedule(arena.allocator(), gpa, "a [cat:dog:0.5] b", 10);
        defer gpa.free(plan.at);
        try std.testing.expectEqual(@as(usize, 2), plan.texts.len);
        try std.testing.expectEqualStrings("a cat b", plan.texts[0]);
        try std.testing.expectEqualStrings("a dog b", plan.texts[1]);
        for (plan.at[0..5]) |ix| try std.testing.expectEqual(@as(u8, 0), ix);
        for (plan.at[5..]) |ix| try std.testing.expectEqual(@as(u8, 1), ix);
    }
    {
        // No scheduling syntax: one text, every step on it. This is the path every
        // ComfyUI-dialect prompt and most A1111 prompts take, so it must stay trivial.
        const plan = try planSchedule(arena.allocator(), gpa, "1girl, (shiny skin:1.1)", 20);
        defer gpa.free(plan.at);
        try std.testing.expectEqual(@as(usize, 1), plan.texts.len);
        for (plan.at) |ix| try std.testing.expectEqual(@as(u8, 0), ix);
    }
}

test "Cond entry accessors treat the parent as entry 0" {
    // The compatibility claim that keeps ggufy and the composed-stages invariant working:
    // a Cond with no schedule has exactly one entry and it is itself.
    const gpa = std.testing.allocator;
    var plain: Cond = .{ .data = try gpa.alloc(f32, 4), .seq = 1 };
    defer plain.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), plain.entryCount());
    try std.testing.expectEqual(&plain, plain.entry(0));

    const extra = try gpa.alloc(Cond, 1);
    extra[0] = .{ .data = try gpa.alloc(f32, 4), .seq = 2 };
    const at = try gpa.alloc(u8, 4);
    @memcpy(at, &[_]u8{ 0, 1, 1, 0 });
    var sched: Cond = .{ .data = try gpa.alloc(f32, 4), .seq = 1, .sched = .{ .extra = extra, .at = at } };
    defer sched.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), sched.entryCount());
    try std.testing.expectEqual(@as(usize, 2), sched.entry(1).seq);
    try std.testing.expectEqual(@as(usize, 1), sched.sched.?.indexAt(2));
    // Past the end clamps to the last step rather than reading out of bounds — a schedule
    // built for fewer steps than the render still resolves.
    try std.testing.expectEqual(@as(usize, 0), sched.sched.?.indexAt(99));
}

test "compat resolves A1111's three sampling defaults, and a per-knob override wins" {
    // ⚠️ All THREE, which is the point of the aggregate: A1111 disagrees with ComfyUI on
    // every one of them, and the render only reproduces if they all flip together. A
    // future edit that added a fourth convention to `.a1111` while leaving one at
    // ComfyUI's value is exactly what this catches.
    const a: CompatConfig = .of(.a1111);
    try std.testing.expectEqual(noise_mod.Source.nv_philox, a.noise_src);
    try std.testing.expectEqual(false, a.sgm_noise_multiplier);
    try std.testing.expectEqual(false, a.quantize_timestep);

    const c: CompatConfig = .of(.comfy);
    try std.testing.expectEqual(noise_mod.Source.torch_cpu, c.noise_src);
    try std.testing.expectEqual(true, c.sgm_noise_multiplier);
    try std.testing.expectEqual(true, c.quantize_timestep);

    // The default stays ComfyUI's, since that is what every render here reproduced
    // before this was selectable.
    const dflt: Options = .{ .prompt = "x" };
    try std.testing.expectEqual(c, dflt.compatConfig());

    // An override beats the aggregate in both directions — an A1111 user who set
    // `randn_source` to CPU, or a ComfyUI user probing just the RNG.
    var o: Options = .{ .prompt = "x", .compat = .a1111, .rng = .torch_cpu };
    var got = o.compatConfig();
    try std.testing.expectEqual(noise_mod.Source.torch_cpu, got.noise_src);
    try std.testing.expectEqual(false, got.sgm_noise_multiplier); // the others untouched

    o = .{ .prompt = "x", .rng = .nv_philox, .quantize_timestep = false };
    got = o.compatConfig();
    try std.testing.expectEqual(noise_mod.Source.nv_philox, got.noise_src);
    try std.testing.expectEqual(true, got.sgm_noise_multiplier);
    try std.testing.expectEqual(false, got.quantize_timestep);
}

test "the timestep quantization switch is not a no-op on a real schedule" {
    // The cheap half of `--compat a1111` to get wrong silently: both branches return a
    // well-defined timestep and neither errors, so nothing but a render comparison
    // notices. Pin that they actually differ, and that only one of them is integral.
    const gpa = std.testing.allocator;
    const ladder = try sampler.sdSigmasFull(gpa);
    defer gpa.free(ladder);
    const sigmas = try sampler.sdSchedule(gpa, 8);
    defer gpa.free(sigmas);

    var any_differ = false;
    for (sigmas[0..8]) |sg| {
        const q = sampler.sdModelTimestep(ladder, sg);
        const f = sampler.sdTimestepForSigma(ladder, sg);
        try std.testing.expectEqual(@round(q), q);
        if (@abs(q - f) > 0.05) any_differ = true;
    }
    try std.testing.expect(any_differ);
}

test "prompt-weight support is read off the encoder, not a list of architectures" {
    // ⚠️ The predicate must come from the ENCODER. SD.Next's equivalent is a substring match
    // on pipeline class names, and its own file shows the failure mode: Chroma and HiDream
    // commented out of the list, and a `'Flux2' not in cls` guard bolted on so the `Flux`
    // substring does not swallow a different architecture. Ours cannot drift, and a new
    // encoder that declares nothing fails to compile rather than defaulting.
    try std.testing.expect(!supportsPromptWeights(.krea2));
    try std.testing.expect(supportsPromptWeights(.sd15));
    try std.testing.expect(supportsPromptWeights(.sdxl));
    // Read through the encoder types, so this test fails if a declaration is flipped without
    // the pipeline noticing.
    try std.testing.expectEqual(clip_text.TextEncoder.supports_prompt_weights, supportsPromptWeights(.sdxl));
    try std.testing.expectEqual(qwen3.TextEncoder.supports_prompt_weights, supportsPromptWeights(.krea2));
}

test "an unweightable prompt is reported as such, never refused" {
    // History, because the behaviour reversed twice: a BLANKET
    // `error.PromptSyntaxUnsupportedForFamily` (reported as "why does krea2 report
    // PromptSyntaxUnsupportedForFamily?"), then a refusal narrowed to weighted prompts, now a
    // warning. See `weightsWouldBeDropped` for what settled it. Returning a bool rather than
    // logging is also what lets this test stay silent on success.
    const gpa = std.testing.allocator;
    const o: EncodeOptions = .{ .prompt_syntax = .a1111 };

    for ([_][]const u8{
        "a cat sitting on a mat",
        "", // an empty negative goes through the same path
        "a cat, BREAK, best quality", // a boundary krea2 has no concept of either way
        "a cat \\(cute\\) on a mat", // escaped parens are literal text, weight 1.0
    }) |text| {
        errdefer std.debug.print("nothing should be dropped from: '{s}'\n", .{text});
        try std.testing.expect(!try Session.weightsWouldBeDropped(gpa, text, o));
    }

    // Real weights: reported, so the caller can say so — and the render still proceeds.
    for ([_][]const u8{
        "a (cat:1.3) on a mat",
        "a (cat) on a mat", // bare parens are 1.1 under this dialect
        "a [cat] on a mat", // and brackets are 1/1.1 de-emphasis, not literal text
    }) |text| {
        errdefer std.debug.print("weights should have been noticed in: '{s}'\n", .{text});
        try std.testing.expect(try Session.weightsWouldBeDropped(gpa, text, o));
    }

    // `.ignore` already asked for the weights to be dropped, so there is nothing to report.
    try std.testing.expect(!try Session.weightsWouldBeDropped(gpa, "a (cat:1.3) on a mat", .{ .prompt_syntax = .a1111, .emphasis = .ignore }));

    // ⚠️ Per-step scheduling checked THROUGH `planSchedule`, not on the raw text: the
    // scheduling brackets are resolved before `encodeText` ever sees the prompt, so checking
    // `[cat:dog:0.5]` directly would describe a string the pipeline never passes — and would
    // read as de-emphasis, since a bare `[…]` is 1/1.1 once the schedule is stripped.
    for ([_][]const u8{ "a [cat:dog:0.5] on a mat", "a [cat|dog] on a mat" }) |text| {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const plan = try planSchedule(arena_state.allocator(), gpa, text, 4);
        defer gpa.free(plan.at);
        try std.testing.expectEqual(@as(usize, 2), plan.texts.len); // it really did schedule
        for (plan.texts) |t| try std.testing.expect(!try Session.weightsWouldBeDropped(gpa, t, o));
    }
}
