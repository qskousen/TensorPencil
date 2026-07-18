//! Chat session for tp-gui: owns the resident LLM (Qwen3.5 hybrid on the
//! zig-cuda backend) and runs generation on a background thread, streaming
//! decoded tokens back to the UI through a mutex-guarded queue.
//!
//! Threading contract:
//!  - The UI thread calls `submit` (starts a turn) and `poll` (drains streamed
//!    bytes into the live assistant message, joins a finished worker). It only
//!    touches `messages` and `ids` while no worker is running (`submit` refuses
//!    to start a second turn, so the two never race).
//!  - The worker thread runs `engine.generate`, whose per-token writes land in
//!    `TokenSink.drain` — the only place `pending` is written. `drain` fires the
//!    SDL wakeup so the event-driven render loop repaints promptly.
const std = @import("std");
const tp = @import("TensorPencil");
const config = @import("config.zig");
const toolcall = @import("toolcall.zig");
const diffuser = @import("diffuser.zig");

const qwen35 = tp.models.qwen35;
const qwen35_cuda = tp.models.qwen35_cuda;
const vit35 = tp.models.vit35;
const vit35_cuda = tp.models.vit35_cuda;
const gemma3 = tp.models.gemma3;
const gemma3_cuda = tp.models.gemma3_cuda;
const gemma_vit = tp.models.gemma_vit;
const gemma_vit_cuda = tp.models.gemma_vit_cuda;
const gemma4 = tp.models.gemma4;
const gemma4_cuda = tp.models.gemma4_cuda;
const gemma4_vit = tp.models.gemma4_vit;
const gemma4_vit_cuda = tp.models.gemma4_vit_cuda;

/// The resident LLM + its vision tower, one variant per supported GGUF
/// architecture. `model` retains a `*const lm`, so the bundle must live at a
/// stable address (it does — inside the heap-pinned Session) and its tag is
/// never reassigned after init.
const Arch = union(enum) {
    qwen35: struct { lm: qwen35.Model, model: qwen35_cuda.CudaLM, vit: ?vit35.Vit = null },
    gemma3: struct { lm: gemma3.Model, model: gemma3_cuda.CudaLM, vit: ?gemma_vit.Vit = null },
    gemma4: struct { lm: gemma4.Model, model: gemma4_cuda.CudaLM, vit: ?gemma4_vit.Vit = null },
};
const cuda = tp.gpu.cuda;
const engine = tp.llm.engine;
const sample = tp.llm.sample;
const session = tp.llm.session;
const kv_cache = tp.llm.kv_cache;
const chat = tp.llm.chat;
const vram = tp.vram;
const residency = tp.models.residency;
const pipeline = tp.pipeline;
const Tokenizer = tp.tokenizer.Tokenizer;
const Gguf = tp.Gguf;

/// Upper bound on the auto-sized KV window when the caller passes no explicit
/// `max_context` (mirrors the tp-llm CLI). A model can advertise a very large
/// trained context; this keeps the ceiling sane while still far above the old
/// fixed 16384. Growth is lazy, so this only bounds the maximum, not up-front
/// VRAM.
const auto_context_cap: usize = 128 << 10;

/// Family fallback trained context when the GGUF omits `<arch>.context_length`.
const default_trained_context: usize = 32768;

/// Raw decoded image (packed RGB) awaiting vision encoding.
const RawImage = struct { rgb: []u8, width: usize, height: usize };

/// Diffusion helpers now live with the engine (see diffuser.zig). Aliased so
/// this module and its consumers (app.zig, viewer.zig) keep their `chat.*`
/// references working.
const parseGenAttrs = diffuser.parseGenAttrs;
const rgbToRgba = diffuser.rgbToRgba;
const freeGenImage = diffuser.freeGenImage;

/// Tool-only description of the `<image>…</image>` image tool. This is NOT a
/// full system prompt — it carries no persona — and is appended to the user's
/// configured system prompt only when a diffusion model is available.
const image_tool_prompt =
    \\# Image generation tool
    \\You can generate images. When the user asks you to create, draw, generate, paint, or show an image, produce it by writing a tool call on its own line in EXACTLY this format:
    \\<image>a vivid, detailed description of the image</image>
    \\Only a tool call written on its own line generates an image. You may mention the tag inline while explaining, and you may reason about it in your thoughts, without triggering anything — a generation happens ONLY when you commit to it by writing the tag on its own line in your reply.
    \\Write a rich prompt covering subject, setting, style, lighting, mood, and composition. The image is generated and shown to the user automatically. When the user asks for a change, emit a new, updated tool call.
    \\You may optionally set size/steps/seed as tag attributes when the user wants a specific aspect ratio, more detail, or a repeatable result:
    \\<image width=1024 height=1536 steps=12 seed=42>a tall portrait…</image>
    \\Defaults (square, balanced quality) are used when attributes are omitted. width/height are rounded to multiples of 16. Reuse a prior seed when asked to modify a previous generation, or change the seed when asked to generate variations.
;

pub const Role = enum { user, assistant };

/// Re-exported from the diffusion engine so consumers keep using `chat.*`.
pub const GenStatus = diffuser.GenStatus;
pub const GenImage = diffuser.GenImage;

pub const Message = struct {
    role: Role,
    /// UTF-8 text; grows as tokens stream in (assistant) or fixed (user).
    text: std.ArrayList(u8) = .empty,
    /// Images requested by this (assistant) message, in emission order.
    images: std.ArrayList(*GenImage) = .empty,
    /// Images the user attached to this (user) message (display only).
    attachments: std.ArrayList(*GenImage) = .empty,
    /// Set once the completed turn has been scanned for image tool calls.
    images_scanned: bool = false,

    pub fn deinit(self: *Message, gpa: std.mem.Allocator) void {
        self.text.deinit(gpa);
        // `images` are BORROWED (the app-level engine owns generated images and
        // frees them); only free the ArrayList storage. `attachments` (user
        // images) are owned here.
        self.images.deinit(gpa);
        for (self.attachments.items) |gi| freeGenImage(gpa, gi);
        self.attachments.deinit(gpa);
    }
};

pub const DiffConfig = diffuser.DiffConfig;

/// A `std.Io.Writer` sink that appends everything written to a session's
/// `pending` queue under its mutex and wakes the UI. `iface` must be first so
/// `@fieldParentPtr` recovers the sink from the interface pointer.
const TokenSink = struct {
    iface: std.Io.Writer,
    session: *Session,

    const vtable: std.Io.Writer.VTable = .{ .drain = drain };

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *TokenSink = @fieldParentPtr("iface", w);
        const s = self.session;
        s.mu.lockUncancelable(s.io);
        defer s.mu.unlock(s.io);
        // The interface buffer is logically consumed before `data`.
        if (w.end > 0) {
            s.pending.appendSlice(s.gpa, w.buffer[0..w.end]) catch return error.WriteFailed;
            w.end = 0;
        }
        var consumed: usize = 0;
        if (data.len > 0) {
            for (data[0 .. data.len - 1]) |bytes| {
                s.pending.appendSlice(s.gpa, bytes) catch return error.WriteFailed;
                consumed += bytes.len;
            }
            const last = data[data.len - 1];
            var k: usize = 0;
            while (k < splat) : (k += 1) {
                s.pending.appendSlice(s.gpa, last) catch return error.WriteFailed;
            }
            consumed += last.len * splat;
        }
        s.wake();
        return consumed;
    }
};

pub const Options = struct {
    model_path: []const u8,
    /// Base system prompt sent at the start of every conversation. The image
    /// tool description is appended to it when `images_enabled` is set.
    system_prompt: []const u8 = "You are a helpful assistant.",
    /// KV window ceiling; growth commits rows lazily up to this. `null` (the
    /// default) auto-sizes to the model's trained context length (capped at
    /// `auto_context_cap`), so a session isn't silently clipped to a fixed
    /// window — CUDA grows KV rows on demand, so a large ceiling costs VRAM
    /// only as the conversation fills.
    max_context: ?usize = null,
    max_new_tokens: usize = 2048,
    /// BASE sampling seed for the session (the app passes the clock). Every
    /// turn draws its own seed from a sample.SeedSeq over this at submit, so
    /// a new chat / repeated prompt / regenerated reply never replays the
    /// same RNG stream.
    seed: u64 = 0,
    /// Sampling controls (temperature/top-k/top-p/min-p/penalties). Snapshotted
    /// per turn: `updateSettings` stages changes, `submit` applies them, so a
    /// mid-generation edit only affects the NEXT turn.
    sampling: sample.Params = .{},
    /// Compute backend for the chat LLM. Only the CUDA variants are supported
    /// today; `.cpu`/`.vulkan` fail init with `error.UnsupportedLlmBackend`.
    backend: pipeline.Backend = .zig_cuda,
    /// Whether the image tool is available (a diffusion model is configured, so
    /// the app-level engine can render `<image>` calls). Gates the tool prompt.
    images_enabled: bool = false,
    /// mmproj (vision tower) GGUF; enables dropping images to chat about them.
    /// Requires a CUDA backend (which this session always uses).
    mmproj_path: ?[]const u8 = null,
    /// VRAM meter policy, as fractions of the whole card (resolved to bytes
    /// against the live card total at load). `vram_limit_frac` is the ceiling —
    /// the LLM + image model never allocate past it. `vram_split` is the LLM's
    /// guaranteed share under contention: with the image model resident the LLM
    /// settles to this and diffusion gets the rest up to the ceiling; with
    /// diffusion idle/unloaded the LLM borrows all the way up to the ceiling.
    vram_split: f32 = 0.60,
    vram_limit_frac: f32 = 0.95,
    /// Whether the model reasons (emits a thought block) before answering, for
    /// families that support it. Flipped live by the toolbar toggle; no-op for
    /// non-reasoning models. See `chat.setThinking`.
    reasoning: bool = true,
    /// KV-cache element storage type (f32 default; f16 halves the KV footprint).
    kv_dtype: kv_cache.KvDtype = .f32,
};

/// Map the GUI's engine-free `config.Sampling` onto the library's
/// `sample.Params`, field for field — explicit so the two can't drift silently.
pub fn samplingParams(cfg: *const config.Config) sample.Params {
    const s = &cfg.sampling;
    return .{
        .temperature = s.temperature,
        .top_k = s.top_k,
        .top_p = s.top_p,
        .min_p = s.min_p,
        .repeat_penalty = s.repeat_penalty,
        .repeat_last_n = s.repeat_last_n,
        .presence_penalty = s.presence_penalty,
        .frequency_penalty = s.frequency_penalty,
    };
}

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    wake: *const fn () void,

    // Load-once state (backed by the caller's arena; must outlive the session).
    gguf: Gguf,
    arch: Arch,
    tok: Tokenizer,
    be: *cuda.Backend,

    ids: std.ArrayList(u32) = .empty,
    /// The token prefix produced at init (optional BOS + system prompt, with the
    /// image-tool description when diffusion is enabled). `reset` restores `ids`
    /// to this so a "new chat" reuses the exact same prompt as a fresh session.
    initial_ids: std.ArrayList(u32) = .empty,
    opts: engine.Options,
    /// Sampling params staged by `updateSettings` (UI thread) and copied into
    /// `opts.sampling` by `submit` — also UI-thread, and never while a worker
    /// runs (`submit` refuses when busy). Keeps live settings edits from racing
    /// the worker's read of `opts`, and gives "takes effect next turn" exactly.
    pending_sampling: sample.Params = .{},
    /// Per-turn sampling seeds, drawn at each turn boundary (`submit`, UI
    /// thread — same discipline as `pending_sampling`). Deliberately NOT
    /// reset by `reset`: a new chat must not replay the previous chat's
    /// seeds, and a future "regenerate reply" just draws again.
    seeds: sample.SeedSeq = .{ .state = 0 },
    messages: std.ArrayList(Message) = .empty,

    // Worker <-> UI marshalling.
    mu: std.Io.Mutex = std.Io.Mutex.init,
    pending: std.ArrayList(u8) = .empty,
    generating: std.atomic.Value(bool) = .init(false),
    cancel: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,
    gen_err: ?anyerror = null,
    sink_buf: [256]u8 = undefined,

    // Whether the image tool is available (a diffusion model is configured).
    // The diffusion engine itself is owned app-level (persistent across mode
    // switches); this session only PRODUCES tool-call GenImages (scanImageCalls)
    // into its transcript and backs the engine's VRAM coordinator (LLM layer
    // eviction) + queue source (nextPending over the transcript). The flag gates
    // the image-tool system prompt and the post-turn scan.
    images_enabled: bool = false,
    /// The LLM's dynamic-offload budget (bytes) — the ceiling it keeps device
    /// usage under. Equals `vram_limit` (the meter's limit handle); the offload
    /// scheduler migrates layers to the CPU once weights + KV would exceed it.
    vram_budget: u64 = 0,
    /// The VRAM ceiling (bytes) from the meter's limit handle: LLM + image model
    /// together never allocate past it. Diffusion's resident-weight budget is
    /// `vram_limit − (LLM resident)`, so the two share the space below it.
    vram_limit: u64 = 0,
    /// The LLM's guaranteed share (bytes) from the meter's split handle. Under
    /// contention (image model resident) the LLM settles here, freeing the rest
    /// (up to `vram_limit`) for diffusion to borrow.
    vram_share: u64 = 0,
    /// True while LLM layers have been evicted to the CPU to make VRAM room for
    /// image generation (priority = image). Cleared once they've been promoted
    /// back (queue drained) or on a new chat.
    image_evicted: bool = false,
    /// Cross-thread residency intent published by the app's `vram.Arbiter`. The
    /// worker enacts it at each token boundary (`engine.residency_poll`) on its
    /// own context-bound thread; the arbiter enacts it directly while idle. The
    /// single source of truth for this LLM's desired device-residency ceiling.
    control: vram.ControlPoint = .{},

    // Vision (dropped/attached images the model can see). Loaded once; the
    // tower itself lives in `arch` (per-architecture type).
    mmproj_gguf: ?Gguf = null,
    // Images attached for the next message: `attach_view` are display copies
    // (RGBA, shown as a strip + moved into the sent message); `attach_rgb` are
    // the parallel raw RGB the worker encodes. On submit they move to the
    // message and `turn_images` respectively.
    attach_view: std.ArrayList(*GenImage) = .empty,
    attach_rgb: std.ArrayList(RawImage) = .empty,
    turn_text: []u8 = "",
    turn_images: std.ArrayList(RawImage) = .empty,

    /// Loads the model on the zig-cuda backend. `arena` holds load-once data
    /// (weights are mmap views into the GGUF and must outlive the session);
    /// `gpa` is a thread-safe allocator for the churny generation buffers and
    /// the message/token queues. Returns a heap-stable pointer (the worker and
    /// the token sink capture `*Session`).
    pub fn init(
        arena: std.mem.Allocator,
        gpa: std.mem.Allocator,
        io: std.Io,
        wake: *const fn () void,
        cfg: Options,
    ) !*Session {
        // Built in place on the heap: `qwen35_cuda.CudaLM` retains a
        // `*const qwen35.Model`, so `self.lm` must live at a stable address for
        // the whole session (it is re-read when streamed weights re-upload).
        const self = try gpa.create(Session);
        errdefer gpa.destroy(self);

        self.gguf = try Gguf.open(arena, io, cfg.model_path);
        errdefer self.gguf.deinit();
        const arch_str = self.gguf.getStr("general.architecture") orelse "";

        self.tok = try Tokenizer.initFromGguf(arena, &self.gguf);
        errdefer self.tok.deinit();
        chat.applyTokenizer(&self.tok);

        // The chat LLM runs on CUDA only for now. Both CUDA variants work (the
        // decode driver is identical — hand-PTX kernels either way; libs just
        // adds cuBLASLt/cuDNN). Non-CUDA selections error cleanly (the loader
        // surfaces it) rather than silently falling back.
        self.be = switch (cfg.backend) {
            .zig_cuda => try cuda.Backend.init(arena),
            .cuda => try cuda.Backend.initLibs(arena),
            .cpu, .vulkan => return error.UnsupportedLlmBackend,
        };
        errdefer self.be.deinit();

        self.mmproj_gguf = null;
        if (cfg.mmproj_path) |mp| self.mmproj_gguf = try Gguf.open(arena, io, mp);

        // f16 KV is supported per-arch (gemma4 first); the concrete model init
        // (below) returns error.KvDtypeUnsupported for archs not yet wired, so
        // the loader surfaces a clean error rather than corrupting the cache.

        // Context ceiling: honor an explicit request, else auto-size to the
        // model's trained context length (capped) so the window follows the
        // model instead of a fixed 16384. CUDA grows KV lazily, so this bounds
        // the max without paying the VRAM up front.
        const max_context = cfg.max_context orelse
            @min(@as(usize, @intCast(self.gguf.contextLength() orelse default_trained_context)), auto_context_cap);

        // Interactive session: grow from a small floor toward the full window.
        const cap: engine.Capacity = .{
            .initial = @min(max_context, 4096),
            .max = max_context,
            .kv_dtype = cfg.kv_dtype,
        };

        // Reasoning toggle (process-global like the family). No-op for models
        // whose family can't reason; the toolbar toggle flips it live.
        chat.setThinking(cfg.reasoning);

        // Chat template family (process-global, keyed off the same architecture
        // string as the model dispatch below). `familyForArch` is the single
        // source of truth the GUI also uses to probe a configured-but-unloaded
        // model, so keep the mapping there, not duplicated per branch.
        chat.setFamily(chat.familyForArch(arch_str) orelse return error.UnsupportedArchitecture);

        // Architecture dispatch: each variant bundles {lm, model, vit}. The
        // model retains a `*const lm` into the union, which is stable (self is
        // heap-pinned and the tag is set once). Vision towers are scoped
        // per-encode and never stay resident under the LLM.
        if (std.mem.eql(u8, arch_str, "qwen35")) {
            self.arch = .{ .qwen35 = .{ .lm = try qwen35.Model.load(arena, &self.gguf), .model = undefined } };
            const a = &self.arch.qwen35;
            errdefer a.lm.deinit();
            if (self.mmproj_gguf) |*mg| a.vit = try vit35.Vit.load(arena, mg);
            errdefer if (a.vit) |*v| v.deinit();
            a.model = try qwen35_cuda.CudaLM.init(gpa, self.be, &a.lm, cap);
        } else if (std.mem.eql(u8, arch_str, "gemma3")) {
            self.arch = .{ .gemma3 = .{ .lm = try gemma3.Model.load(arena, &self.gguf), .model = undefined } };
            const a = &self.arch.gemma3;
            errdefer a.lm.deinit();
            if (self.mmproj_gguf) |*mg| a.vit = try gemma_vit.Vit.load(arena, mg);
            errdefer if (a.vit) |*v| v.deinit();
            a.model = try gemma3_cuda.CudaLM.init(gpa, self.be, &a.lm, cap);
        } else if (std.mem.eql(u8, arch_str, "gemma4")) {
            self.arch = .{ .gemma4 = .{ .lm = try gemma4.Model.load(arena, &self.gguf), .model = undefined } };
            const a = &self.arch.gemma4;
            errdefer a.lm.deinit();
            // gemma4's "unified" embedder has no ViT — it runs on CPU (cheap);
            // encodes are scoped per image turn (imageTurn).
            if (self.mmproj_gguf) |*mg| a.vit = try gemma4_vit.Vit.load(arena, mg);
            errdefer if (a.vit) |*v| v.deinit();
            a.model = try gemma4_cuda.CudaLM.init(gpa, self.be, &a.lm, cap);
        } else return error.UnsupportedArchitecture;

        self.opts = .{
            .max_new_tokens = cfg.max_new_tokens,
            .max_context = max_context,
            .seed = cfg.seed,
            .sampling = cfg.sampling,
        };
        self.pending_sampling = cfg.sampling;
        self.seeds = sample.SeedSeq.init(cfg.seed);
        // Let the decode loop enact arbiter-published VRAM targets on the worker
        // thread (`self` is heap-pinned, so the captured pointer stays valid).
        self.opts.residency_poll = .{ .ctx = self, .apply = residencyPollThunk };

        self.gpa = gpa;
        self.io = io;
        self.wake = wake;
        self.ids = .empty;
        self.messages = .empty;
        self.mu = std.Io.Mutex.init;
        self.pending = .empty;
        self.generating = .init(false);
        self.cancel = .init(false);
        self.worker = null;
        self.gen_err = null;
        self.opts.cancel = &self.cancel; // stable heap address for the worker

        self.attach_view = .empty;
        self.attach_rgb = .empty;
        self.turn_text = "";
        self.turn_images = .empty;
        self.image_evicted = false;
        self.images_enabled = cfg.images_enabled;

        // Dynamic VRAM offload (GUI_VRAM.md): always arm the dynamic split (free
        // when the model fits — 0 layers on CPU, per-op decode ties the graph;
        // measured 76 vs 77 tok/s on the 9B). Once context + weights outgrow the
        // budget it migrates layers to the CPU, which is ~2.5x FASTER than the
        // weight-streaming fallback on this box (18 vs 7 tok/s). Vision sessions
        // arm it too: text turns run the fast offload path, and an IMAGE turn
        // promotes every layer back to the GPU first (buildAndPrefillTurn) because
        // the host layer path uses scalar RoPE (wrong for image-grid M-RoPE).
        // budget = the configured limit, or the live free VRAM when 0 (auto).
        {
            // Resolve the meter fractions against THIS card's total VRAM. The
            // ceiling (limit handle) is the LLM's offload budget: it loads fully
            // up to the ceiling — the split is not a hard reservation, it only
            // bites when the image model actually loads (imageVramEnter settles
            // the LLM to its share then). With diffusion idle the LLM keeps
            // everything up to the ceiling.
            const total: f32 = @floatFromInt(self.be.ctx.memGetInfo().total);
            self.vram_limit = @intFromFloat(cfg.vram_limit_frac * total);
            self.vram_share = @intFromFloat(cfg.vram_split * total);
            self.vram_budget = self.vram_limit;
            switch (self.arch) {
                inline else => |*a| _ = a.model.autoOffload(self.vram_budget) catch |err|
                    std.log.warn("dynamic offload disabled ({t}); staying resident", .{err}),
            }
        }

        // Gemma prompts begin with a single BOS ({{ bos_token }}); ChatML has none.
        if (self.arch == .gemma3 or self.arch == .gemma4) {
            if (self.tok.specialId("<bos>")) |bos| try self.ids.append(gpa, bos);
        }

        // System prompt: the configured base prompt, with the image-tool
        // description appended when the image tool is available. Prefilled on
        // the first turn (the uncached prefix of `ids`); never shown in the UI.
        if (self.images_enabled) {
            const full = try std.fmt.allocPrint(self.gpa, "{s}\n\n{s}", .{ cfg.system_prompt, image_tool_prompt });
            defer self.gpa.free(full);
            try chat.appendSystem(&self.tok, self.gpa, full, &self.ids);
        } else {
            try chat.appendSystem(&self.tok, self.gpa, cfg.system_prompt, &self.ids);
        }

        // Snapshot the prompt prefix so `reset` can restore it verbatim.
        self.initial_ids = .empty;
        try self.initial_ids.appendSlice(gpa, self.ids.items);
        return self;
    }

    /// Start a fresh conversation (KV + transcript reset, LLM residency re-armed).
    /// CONTRACT: the caller must first stop the app-level diffusion engine (join
    /// its worker) so no diffusion thread is still touching a transcript GenImage
    /// as this frees the messages — the session no longer owns that engine.
    pub fn reset(self: *Session) void {
        if (self.busy()) return;

        for (self.messages.items) |*m| m.deinit(self.gpa);
        self.messages.clearRetainingCapacity();

        for (self.attach_view.items) |gi| freeGenImage(self.gpa, gi);
        self.attach_view.clearRetainingCapacity();
        for (self.attach_rgb.items) |im| self.gpa.free(im.rgb);
        self.attach_rgb.clearRetainingCapacity();

        self.mu.lockUncancelable(self.io);
        self.pending.clearRetainingCapacity();
        self.mu.unlock(self.io);

        self.ids.clearRetainingCapacity();
        self.ids.appendSlice(self.gpa, self.initial_ids.items) catch |err|
            std.log.err("reset: restore prompt prefix failed: {t}", .{err});

        // resetResidency issues device work (shrinks the grown KV back to the
        // initial capacity, brings CPU-offloaded layers back onto the GPU, zeroes
        // recurrent state, and re-arms dynamic offload for the fresh small
        // context), so the LLM context must be current on this (UI) thread. All
        // GPU workers are joined above, so there's no concurrent device access.
        self.be.bindThread();
        self.image_evicted = false; // a fresh chat starts from the armed baseline
        switch (self.arch) {
            inline else => |*a| a.model.resetResidency(self.vram_budget) catch |err|
                std.log.err("reset: residency reset failed: {t}", .{err}),
        }

        self.gen_err = null;
        self.wake();
    }

    /// Rebuild the KV cache at a new element dtype (the config f32<->f16 toggle),
    /// keeping the model WEIGHTS resident — a "context reload", not a model
    /// reload. Frees + re-creates the K/V buffers at `dtype` and resets the
    /// committed length to 0; `ids` still hold the whole transcript, so the next
    /// turn re-prefills it. Runs on the UI thread like `updateSettings`; the
    /// caller must ensure no generation is in flight (returns error.Busy if so).
    /// Errors (error.KvDtypeUnsupported) if the active arch can't do `dtype`.
    pub fn rebuildContext(self: *Session, dtype: kv_cache.KvDtype) !void {
        if (self.busy()) return error.Busy;
        self.be.bindThread();
        switch (self.arch) {
            inline else => |*a| try a.model.reinitCache(dtype),
        }
        self.wake();
    }

    pub fn imagesEnabled(self: *const Session) bool {
        return self.images_enabled;
    }

    /// Ask the running generation to stop at the next token (no-op if idle).
    pub fn requestCancel(self: *Session) void {
        self.cancel.store(true, .release);
    }

    /// Drop the transcript's BORROWED references to engine-owned generated
    /// images (the pointers, not the images). The app calls this right before it
    /// frees the diffusion engine (diffusion model cleared) so no message is
    /// left pointing at freed memory. Text is untouched.
    pub fn clearImageRefs(self: *Session) void {
        for (self.messages.items) |*m| m.images.clearRetainingCapacity();
    }

    /// Apply the non-load-affecting settings live (no reload, so the chat is
    /// preserved): reasoning and the sampling controls. The VRAM meter policy
    /// (split/limit) is applied separately via `applyVramPolicy`; diffusion-
    /// facing settings go to the app-level engine.
    pub fn updateSettings(self: *Session, cfg: *const config.Config) void {
        // Reasoning is process-global (like the family) and only shapes the
        // *next* prompt built, so flipping it mid-conversation is safe without a
        // reload — the current turn already has its ids.
        chat.setThinking(cfg.reasoning);
        // Sampling is staged, not applied: `submit` copies it into `opts` at the
        // next turn boundary (a turn possibly generating right now keeps the
        // params it started with — no racing the worker's read of `opts`).
        self.pending_sampling = samplingParams(cfg);
    }

    // --- vram.Participant adapter --------------------------------------------
    // Lets the app-level `vram.Arbiter` drive this LLM's device residency the
    // same way it (will) drive diffusion. Read thunks are thread-safe (plain
    // field/atomic reads). `applyBudget` mutates residency + binds the context,
    // so it runs ONLY on the worker (via `pollAndApply` at a token boundary) or
    // on the arbiter thread while the LLM is idle — never racing the worker.
    fn vpUsage(ctx: *anyopaque) u64 {
        return fromCtx(ctx).be.deviceUsed();
    }
    fn vpFloor(ctx: *anyopaque) u64 {
        return fromCtx(ctx).ctxKvBytes(); // committed KV can't be evicted
    }
    fn vpBusy(ctx: *anyopaque) bool {
        return fromCtx(ctx).busy();
    }
    fn vpApply(ctx: *anyopaque, target: u64) void {
        const self = fromCtx(ctx);
        self.be.bindThread();
        switch (self.arch) {
            inline else => |*a| {
                // Snapshot residency around the settle so we can log exactly what
                // moved (settleTo is idempotent — an already-satisfied target
                // shifts nothing, so we stay quiet unless the split actually
                // changed, mirroring the per-turn tok/s summary).
                const before = residency.snapshot(&a.model);
                residency.settleTo(&a.model, target) catch |err| {
                    std.log.warn("[vram] LLM settle→{d}MB failed: {t}", .{ target >> 20, err });
                    return;
                };
                const after = residency.snapshot(&a.model);
                if (after.n_cpu != before.n_cpu) {
                    const dir = if (after.n_cpu > before.n_cpu) "offload→CPU" else "promote→GPU";
                    std.log.info("[vram] LLM {s} (target {d} MiB): {d}→{d}/{d} layers on host · device {d}→{d} MiB · {d} MiB free", .{
                        dir, target >> 20, before.n_cpu, after.n_cpu, after.n_layers,
                        before.device_mib, after.device_mib, after.free_mib,
                    });
                }
            },
        }
    }
    fn fromCtx(ctx: *anyopaque) *Session {
        return @ptrCast(@alignCast(ctx));
    }
    const vp_vtable: vram.Participant.VTable = .{ .usage = vpUsage, .floor = vpFloor, .busy = vpBusy, .applyBudget = vpApply };

    /// This LLM as a `vram.Participant` the app-level arbiter can drive.
    pub fn participant(self: *Session) vram.Participant {
        return .{ .ctx = self, .control = &self.control, .vtable = &vp_vtable };
    }

    /// `engine.residency_poll` thunk: at each token boundary, enact any target
    /// the arbiter published to `control` — on the worker's own bound thread.
    fn residencyPollThunk(ctx: *anyopaque) void {
        fromCtx(ctx).participant().pollAndApply();
    }

    /// pipeline reclaim hook (GUI_VRAM.md Phase 5): free LLM VRAM for a large VAE
    /// decode by migrating chat layers to the CPU — done even under chat priority,
    /// as the agreed last resort so a big image never just fails. Migrates JUST
    /// ENOUGH to free ~`needed` bytes (the decode's remaining deficit), leaving
    /// the rest of the LLM resident so it stays fast; `needed == 0` (or more than
    /// the LLM holds) migrates everything it can. Returns the device bytes freed.
    /// Runs on the DIFFUSION thread, so it binds the LLM context and is safe only
    /// when the LLM is idle (else it declines — returns 0 — and the pipeline
    /// falls back to tiling / CPU). The `vram.Arbiter` promotes the layers back
    /// (target = limit − diff_used) when the image queue drains (`vcExit`).
    pub fn imageReclaim(self: *Session, needed: u64) u64 {
        if (self.busy()) {
            std.log.info("[reclaim] declined: LLM is generating (can't migrate mid-decode)", .{});
            return 0;
        }
        if (self.vram_budget == 0) {
            std.log.info("[reclaim] declined: no VRAM budget (auto)", .{});
            return 0;
        }
        self.be.bindThread();
        const before = self.be.deviceUsed();
        const t0 = std.Io.Clock.real.now(self.io).nanoseconds;
        switch (self.arch) {
            inline else => |*a| {
                if (a.model.split == null)
                    a.model.enableCpuSplit(.attn, self.vram_budget, true) catch |err| {
                        std.log.warn("[reclaim] enableCpuSplit failed: {t}", .{err});
                        return 0;
                    };
                if (needed == 0 or needed >= before)
                    a.model.offloadUntilFree(std.math.maxInt(u64)) catch {} // migrate all it can
                else
                    a.model.offloadToBudget(before - needed) catch {}; // just enough
            },
        }
        const freed = before -| self.be.deviceUsed();
        const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(self.io).nanoseconds - t0)) / 1e6;
        std.log.info("[reclaim] needed={d}MB migrated LLM {d}MB→CPU in {d:.0}ms (LLM now {d}MB device)", .{
            needed >> 20, freed >> 20, ms, self.be.deviceUsed() >> 20,
        });
        if (freed > 0) self.image_evicted = true;
        return freed;
    }

    /// Detach the conversation transcript so it survives a session teardown:
    /// returns `messages` and leaves the session with an empty list (so its
    /// `deinit` won't free the carried messages). Messages are gpa-owned, so they
    /// outlive the load-once arena. Caller must have joined all workers first.
    pub fn detachTranscript(self: *Session) std.ArrayList(Message) {
        const m = self.messages;
        self.messages = .empty;
        return m;
    }

    /// Adopt a transcript carried from a previous session and rebuild `ids` by
    /// replaying each turn's text into THIS model's tokenizer/chat template. KV
    /// stays empty (len 0) — the next turn's prefill replays the whole context
    /// into the new model. `ids` must already hold the init prompt prefix
    /// (BOS + system), which a fresh session has. Image attachments are not
    /// re-encoded across a model swap (text is preserved; the model won't re-see
    /// past images). Takes ownership of `msgs`.
    pub fn adoptTranscript(self: *Session, msgs: std.ArrayList(Message)) !void {
        self.messages = msgs;
        for (self.messages.items) |*m| switch (m.role) {
            .user => try chat.appendUser(&self.tok, self.gpa, m.text.items, &self.ids),
            .assistant => {
                try chat.openAssistant(&self.tok, self.gpa, &self.ids);
                if (m.text.items.len > 0) try self.tok.encode(self.gpa, m.text.items, &self.ids);
                try chat.closeAssistant(self.gpa, &self.ids);
            },
        };
        self.wake();
    }

    pub fn deinit(self: *Session) void {
        if (self.worker) |t| t.join();
        // The diffusion engine is owned app-level, not here — the app stops it
        // (so no diffusion worker still touches a transcript GenImage) before
        // tearing the session down.
        for (self.attach_view.items) |gi| freeGenImage(self.gpa, gi);
        self.attach_view.deinit(self.gpa);
        for (self.attach_rgb.items) |im| self.gpa.free(im.rgb);
        self.attach_rgb.deinit(self.gpa);
        for (self.turn_images.items) |im| self.gpa.free(im.rgb);
        self.turn_images.deinit(self.gpa);
        if (self.turn_text.len > 0) self.gpa.free(self.turn_text);
        switch (self.arch) {
            inline else => |*a| {
                a.model.deinit();
                if (a.vit) |*v| v.deinit();
            },
        }
        if (self.mmproj_gguf) |*g| g.deinit();
        for (self.messages.items) |*m| m.deinit(self.gpa);
        self.messages.deinit(self.gpa);
        self.pending.deinit(self.gpa);
        self.ids.deinit(self.gpa);
        self.initial_ids.deinit(self.gpa);
        self.be.deinit();
        self.tok.deinit();
        switch (self.arch) {
            inline else => |*a| a.lm.deinit(),
        }
        self.gguf.deinit();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    /// Vision tower loaded for the active architecture?
    fn hasVit(self: *const Session) bool {
        return switch (self.arch) {
            inline else => |*a| a.vit != null,
        };
    }

    pub fn busy(self: *Session) bool {
        return self.generating.load(.acquire);
    }

    pub fn visionEnabled(self: *const Session) bool {
        return self.hasVit();
    }

    /// Current context length in tokens (KV rows used).
    pub fn ctxTokens(self: *Session) usize {
        return switch (self.arch) {
            inline else => |*a| a.model.cached(),
        };
    }

    /// Total KV-cache bytes for the current context (all attention layers,
    /// logical footprint — K+V, f32) — what the context "costs" in memory
    /// regardless of whether a given layer's KV lives on the GPU or the CPU.
    pub fn ctxKvBytes(self: *Session) u64 {
        return switch (self.arch) {
            .qwen35 => |*a| @as(u64, a.model.cached()) * a.model.cfg.kvDim() * 8 * a.model.cfg.nAttnLayers(),
            // gemma3/gemma4 LOCAL (sliding-window) layers hold only a fixed ring
            // (window + one prefill chunk), so their footprint plateaus instead
            // of growing with the conversation (TODO lever 1); GLOBAL layers hold
            // the full context. gemma4's KV dim also varies per layer.
            .gemma3 => |*a| blk: {
                const cfg = a.model.cfg;
                const cached: u64 = a.model.cached();
                const ring: u64 = if (cfg.sliding_window != 0) cfg.sliding_window + 128 else cached;
                var rows: u64 = 0;
                for (0..cfg.n_layers) |l| rows += if (cfg.isGlobal(l)) cached else @min(cached, ring);
                break :blk rows * cfg.kvDim() * 8;
            },
            .gemma4 => |*a| blk: {
                const cfg = a.model.cfg;
                const cached: u64 = a.model.cached();
                const ring: u64 = if (cfg.sliding_window != 0) cfg.sliding_window + 128 else cached;
                var kv: u64 = 0;
                for (0..cfg.n_layers) |l| {
                    const layer_rows = if (cfg.isGlobal(l)) cached else @min(cached, ring);
                    kv += layer_rows * cfg.kvDim(l);
                }
                break :blk kv * 8;
            },
        };
    }

    /// LLM layer residency: how many layers run on the GPU vs the CPU. `cpu` is 0
    /// when no split is armed (fully resident).
    pub const Residency = struct { gpu: usize, cpu: usize };
    pub fn llmResidency(self: *Session) Residency {
        return switch (self.arch) {
            inline else => |*a| blk: {
                const total = a.model.cfg.n_layers;
                const cpu = if (a.model.split) |sp| sp.n_cpu else 0;
                break :blk .{ .gpu = total - cpu, .cpu = cpu };
            },
        };
    }

    /// Attach a decoded image (packed RGB) to the next message. Shown as a
    /// thumbnail strip now; encoded and interleaved into the turn on send.
    pub fn attachImage(self: *Session, rgb_src: []const u8, w: usize, h: usize) !void {
        if (!self.hasVit()) {
            std.log.warn("dropped image ignored: vision tower not loaded", .{});
            return;
        }
        const rgb = try self.gpa.dupe(u8, rgb_src);
        errdefer self.gpa.free(rgb);
        const gi = try self.gpa.create(GenImage);
        gi.* = .{ .prompt = try self.gpa.dupe(u8, ""), .wake = self.wake, .io = self.io, .width = w, .height = h };
        gi.rgba = try rgbToRgba(self.gpa, rgb, w, h);
        gi.status = .init(@intFromEnum(GenStatus.done));
        try self.attach_view.append(self.gpa, gi);
        try self.attach_rgb.append(self.gpa, .{ .rgb = rgb, .width = w, .height = h });
        self.wake();
    }

    pub fn pendingAttachments(self: *const Session) []const *GenImage {
        return self.attach_view.items;
    }

    /// Drop a not-yet-sent attachment (and its parallel raw RGB) by index.
    pub fn removeAttachment(self: *Session, idx: usize) void {
        if (idx >= self.attach_view.items.len) return;
        freeGenImage(self.gpa, self.attach_view.orderedRemove(idx));
        self.gpa.free(self.attach_rgb.orderedRemove(idx).rgb);
        self.wake();
    }

    /// Attach an RGBA image (e.g. a generated one) to the next message so the
    /// model can see it — converts to the RGB the encoder expects.
    pub fn attachRgba(self: *Session, rgba: []const u8, w: usize, h: usize) !void {
        if (!self.hasVit()) return;
        const px = w * h;
        const rgb = try self.gpa.alloc(u8, px * 3);
        defer self.gpa.free(rgb);
        for (0..px) |i| {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        try self.attachImage(rgb, w, h);
    }

    /// Append a user turn and spawn the worker to stream the reply. No-op if a
    /// turn is already generating or there is nothing to send.
    pub fn submit(self: *Session, text: []const u8) !void {
        if (self.busy()) return;
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len == 0 and self.attach_view.items.len == 0) return;

        // Turn boundary: adopt any sampling changes staged by updateSettings,
        // and draw this turn's sampling seed so no two turns replay the same
        // RNG stream. Safe here (UI thread, no worker running yet).
        self.opts.sampling = self.pending_sampling;
        self.opts.seed = self.seeds.next();

        var um: Message = .{ .role = .user };
        if (trimmed.len > 0) try um.text.appendSlice(self.gpa, trimmed);
        try um.attachments.appendSlice(self.gpa, self.attach_view.items);
        self.attach_view.clearRetainingCapacity();
        try self.messages.append(self.gpa, um);
        try self.messages.append(self.gpa, .{ .role = .assistant });

        // Stash the turn; the worker builds tokens after encoding any images
        // (encoding must run on the worker's CUDA thread).
        self.turn_text = if (trimmed.len > 0) try self.gpa.dupe(u8, trimmed) else "";
        try self.turn_images.appendSlice(self.gpa, self.attach_rgb.items);
        self.attach_rgb.clearRetainingCapacity();

        self.gen_err = null;
        self.cancel.store(false, .release);
        self.generating.store(true, .release);
        self.worker = std.Thread.spawn(.{}, workerMain, .{self}) catch |err| {
            self.generating.store(false, .release);
            return err;
        };
    }

    fn workerMain(self: *Session) void {
        // CUDA contexts are per-thread; the model was loaded on the main
        // thread, so bind the backend's context to this worker before any
        // device op (else cuMemcpyHtoD fails with INVALID_CONTEXT).
        self.be.bindThread();

        self.buildAndPrefillTurn() catch |err| {
            self.gen_err = err;
            std.log.err("turn setup failed: {t}", .{err});
            self.freeTurn();
            self.generating.store(false, .release);
            self.wake();
            return;
        };

        var sink: TokenSink = .{
            .iface = .{ .vtable = &TokenSink.vtable, .buffer = &self.sink_buf },
            .session = self,
        };
        const t0 = std.Io.Clock.real.now(self.io).nanoseconds;
        switch (self.arch) {
            inline else => |*a| {
                const n = engine.generate(&a.model, &self.tok, self.io, self.gpa, &self.ids, self.opts, &sink.iface) catch |err| n: {
                    self.gen_err = err;
                    std.log.err("generation failed: {t}", .{err});
                    break :n 0;
                };
                // End-of-response telemetry, mirroring the CLI's summary line —
                // routed through std.log so it lands on the same channel the GUI
                // already shows diffusion "gen:" progress on (dvui.App.logFn).
                const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(self.io).nanoseconds - t0)) / 1e9;
                const st = session.Stats.of(&a.model);
                // Residency snapshot too: layers migrated to the host grow with
                // the KV cache mid-turn (the ensureCapacity path, not the arbiter),
                // so folding it into the per-turn summary is how those gradual
                // offloads surface (context bound on this worker thread).
                const res = residency.snapshot(&a.model);
                var vbuf: [32]u8 = undefined;
                std.log.info("[llm] {d} tok, {d:.1} tok/s, ctx {d}/{d}{s}, {d}/{d} layers on host, {d} MiB free", .{
                    n, if (dt > 0) @as(f64, @floatFromInt(n)) / dt else 0, st.tokens, st.window, st.vramSuffix(&vbuf),
                    res.n_cpu, res.n_layers, res.free_mib,
                });
            },
        }
        // Close the assistant turn so the next turn's context is well-formed.
        chat.closeAssistant(self.gpa, &self.ids) catch {};
        self.freeTurn();
        self.generating.store(false, .release);
        self.wake();
    }

    /// Build this turn's tokens and prefill them. Text-only turns defer prefill
    /// to `engine.generate`; image turns encode each image (the arch's vision
    /// tower on CUDA), build the interleaved vision token layout, and inject
    /// the embeddings at their pad rows — mirroring `llm_main.imageTurn`.
    fn buildAndPrefillTurn(self: *Session) !void {
        if (self.turn_images.items.len > 0 and self.hasVit()) {
            // Image tokens must run on the GPU: the offloaded-layer host path uses
            // scalar RoPE, wrong for image-grid M-RoPE positions (qwen35). Promote
            // every migrated layer back to the device first (KV-preserving); a huge
            // budget means "bring back everything that physically fits". Runs on
            // the worker's bound LLM context. As context grows afterward,
            // ensureCapacity re-offloads.
            switch (self.arch) {
                inline else => |*a| _ = a.model.promoteLayers(std.math.maxInt(u64)) catch |err|
                    std.log.warn("promote for image turn failed: {t}", .{err}),
            }
            switch (self.arch) {
                .qwen35 => |*a| try self.imageTurn(a, false),
                .gemma3 => |*a| try self.imageTurn(a, true),
                .gemma4 => |*a| try self.imageTurnGemma4(a),
            }
        } else {
            try chat.appendUser(&self.tok, self.gpa, self.turn_text, &self.ids);
            try chat.openAssistant(&self.tok, self.gpa, &self.ids);
        }
    }

    /// Encode this turn's images on the arch's vision tower, build the
    /// interleaved segment layout (family-aware), and inject each image's
    /// embeddings at its placeholder rows. `a` is a pointer to the active
    /// arch bundle; `gemma` selects the (io-taking) gemma vision encoder.
    fn imageTurn(self: *Session, a: anytype, comptime gemma: bool) !void {
        const Enc = if (gemma) gemma_vit.Vit.Encoded else vit35.Vit.Encoded;
        var encs: std.ArrayList(Enc) = .empty;
        defer {
            for (encs.items) |*e| e.deinit(self.gpa);
            encs.deinit(self.gpa);
        }
        var segs: std.ArrayList(chat.Segment) = .empty;
        defer segs.deinit(self.gpa);
        for (self.turn_images.items) |im| {
            const enc = if (gemma)
                try gemma_vit_cuda.encode(&a.vit.?, self.be, self.io, self.gpa, im.rgb, im.width, im.height)
            else
                try vit35_cuda.encode(&a.vit.?, self.be, self.gpa, im.rgb, im.width, im.height);
            try encs.append(self.gpa, enc);
            try segs.append(self.gpa, .{ .image = .{ .grid_w = enc.grid_w, .grid_h = enc.grid_h } });
        }
        if (self.turn_text.len > 0) try segs.append(self.gpa, .{ .text = self.turn_text });
        try session.prefillImageTurn(&a.model, &self.tok, self.gpa, &self.ids, segs.items, encs.items);
    }

    /// imageTurn for gemma4: the "unified" embedder (no ViT transformer) runs
    /// device-side via gemma4_vit_cuda; the projected embeddings then inject
    /// into the LLM at their placeholder rows. Otherwise identical to imageTurn.
    fn imageTurnGemma4(self: *Session, a: anytype) !void {
        var encs: std.ArrayList(gemma4_vit.Vit.Encoded) = .empty;
        defer {
            for (encs.items) |*e| e.deinit(self.gpa);
            encs.deinit(self.gpa);
        }
        var segs: std.ArrayList(chat.Segment) = .empty;
        defer segs.deinit(self.gpa);
        for (self.turn_images.items) |im| {
            const enc = try gemma4_vit_cuda.encode(&a.vit.?, self.be, self.io, self.gpa, im.rgb, im.width, im.height);
            try encs.append(self.gpa, enc);
            try segs.append(self.gpa, .{ .image = .{ .grid_w = enc.grid_w, .grid_h = enc.grid_h } });
        }
        if (self.turn_text.len > 0) try segs.append(self.gpa, .{ .text = self.turn_text });
        try session.prefillImageTurn(&a.model, &self.tok, self.gpa, &self.ids, segs.items, encs.items);
    }

    fn freeTurn(self: *Session) void {
        if (self.turn_text.len > 0) self.gpa.free(self.turn_text);
        self.turn_text = "";
        for (self.turn_images.items) |im| self.gpa.free(im.rgb);
        self.turn_images.clearRetainingCapacity();
    }

    /// UI-thread, once per frame: move streamed bytes into the live assistant
    /// message and reap a finished worker. Tool-call scanning and diffusion
    /// pumping are driven by the app (the engine is app-level).
    pub fn poll(self: *Session) void {
        self.mu.lockUncancelable(self.io);
        if (self.pending.items.len > 0 and self.messages.items.len > 0) {
            const last = &self.messages.items[self.messages.items.len - 1];
            last.text.appendSlice(self.gpa, self.pending.items) catch {};
            self.pending.clearRetainingCapacity();
        }
        self.mu.unlock(self.io);

        if (self.worker) |t| {
            if (!self.busy()) {
                t.join();
                self.worker = null;
            }
        }
    }

    /// Once a turn completes, scan the last assistant message (once) for
    /// `<image>` tool calls and queue a GenImage for each into its transcript,
    /// using the app-level engine's defaults + seed. Called by the app after
    /// `poll` (chat mode). The app then pumps the engine.
    pub fn scanNewImages(self: *Session, d: *diffuser.Diffuser) void {
        if (!self.images_enabled or self.busy() or self.messages.items.len == 0) return;
        const last = &self.messages.items[self.messages.items.len - 1];
        if (last.role != .assistant or last.images_scanned) return;
        last.images_scanned = true;
        self.scanImageCalls(last, d) catch |err| std.log.err("scan image calls: {t}", .{err});
    }

    /// The active family's reasoning-block markers as `toolcall.Reasoning`
    /// (null when the family can't reason), bridging `tp.llm.chat.reasoning()`
    /// to the std-only tool-call module.
    fn reasoningMarkers() ?toolcall.Reasoning {
        const r = chat.reasoning() orelse return null;
        return .{ .open = r.open, .close = r.close };
    }

    /// Extract `<image ...>PROMPT</image>` tool calls from a finished assistant
    /// message and queue a GenImage (status .pending) for each. Optional tag
    /// attributes (width/height/steps/seed) override the engine defaults.
    fn scanImageCalls(self: *Session, msg: *Message, d: *diffuser.Diffuser) !void {
        // Scan only the answer, not the reasoning block, and only line-anchored
        // tags — see toolcall.answerText/nextImageCall for why (spurious fires
        // from the model merely *mentioning* the tag while thinking/explaining).
        var rest = toolcall.answerText(msg.text.items, reasoningMarkers());
        while (true) {
            const c = switch (toolcall.nextImageCall(rest)) {
                .none, .partial => break,
                .call => |c| c,
            };
            if (c.prompt.len > 0) {
                const gi = try self.gpa.create(GenImage);
                gi.* = .{
                    .prompt = try self.gpa.dupe(u8, c.prompt),
                    .wake = self.wake,
                    .io = self.io,
                    .req_width = d.opts.width,
                    .req_height = d.opts.height,
                    .req_steps = d.opts.steps,
                    .req_seed = 0,
                };
                parseGenAttrs(c.attrs, gi);
                // Assign a fresh, distinct seed now (unless the tag set one
                // explicitly) so it's known and displayable immediately — even
                // while the image is still queued. Advancing per image keeps
                // repeated generations varied.
                if (gi.req_seed == 0) gi.req_seed = d.nextSeed();
                // The engine OWNS the image (unified queue + history); the
                // message keeps a borrowed pointer for inline display.
                try d.enqueue(gi);
                try msg.images.append(self.gpa, gi);
            }
            rest = c.after;
        }
    }
};
