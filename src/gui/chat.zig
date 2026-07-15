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
const chat = tp.llm.chat;
const pipeline = tp.pipeline;
const Tokenizer = tp.tokenizer.Tokenizer;
const Gguf = tp.Gguf;

/// Raw decoded image (packed RGB) awaiting vision encoding.
const RawImage = struct { rgb: []u8, width: usize, height: usize };

/// A requested diffusion-model path set awaiting application (gpa-owned dupes).
const PendingDiffPaths = struct { dit: []u8, vae: []u8, te: []u8, taew: ?[]u8 };

/// Round to a multiple of 16 (pipeline requirement) within sane bounds.
fn clampDim(n: usize) usize {
    const c = std.math.clamp(n, 256, 4096);
    return c / 16 * 16;
}

/// The LLM's dynamic-offload budget (bytes): the whole VRAM limit (or the live
/// free VRAM when there's no limit). The LLM always loads fully into its budget
/// when there's room — the priority split is NOT a hard reservation. Balanced's
/// 75/25 only bites when the image model actually loads (contention), handled at
/// image-gen time by `imageVramEnter` (offloadToBudget), not here.
fn llmBudget(limit_bytes: u64, free_vram: u64) u64 {
    return if (limit_bytes == 0) free_vram else limit_bytes;
}

/// Parse `key=value` tokens from an `<image ...>` tag into the GenImage.
fn parseGenAttrs(attrs: []const u8, gi: *GenImage) void {
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
fn nowNs(io: std.Io) i64 {
    return @intCast(std.Io.Clock.real.now(io).nanoseconds);
}

fn rgbToRgba(gpa: std.mem.Allocator, rgb: []const u8, w: usize, h: usize) ![]u8 {
    const rgba = try gpa.alloc(u8, w * h * 4);
    for (0..w * h) |i| {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

/// Tool-only description of the `<image>…</image>` image tool. This is NOT a
/// full system prompt — it carries no persona — and is appended to the user's
/// configured system prompt only when a diffusion model is available.
const image_tool_prompt =
    \\# Image generation tool
    \\You can generate images. When the user asks you to create, draw, generate, paint, or show an image, produce it by writing a tool call on its own line in EXACTLY this format:
    \\<image>a vivid, detailed description of the image</image>
    \\Write a rich prompt covering subject, setting, style, lighting, mood, and composition. The image is generated and shown to the user automatically. When the user asks for a change, emit a new, updated tool call.
    \\You may optionally set size/steps/seed as tag attributes when the user wants a specific aspect ratio, more detail, or a repeatable result:
    \\<image width=1024 height=1536 steps=12 seed=42>a tall portrait…</image>
    \\Defaults (square, balanced quality) are used when attributes are omitted. width/height are rounded to multiples of 16. Reuse a prior seed when asked to modify a previous generation, or change the seed when asked to generate variations.
;

pub const Role = enum { user, assistant };

pub const GenStatus = enum(u8) { pending, generating, done, failed, canceled };

/// An image the assistant asked to generate (via a `<image>…</image>` tool
/// call). Progress/status fields are atomics written by the diffusion worker
/// and read by the UI thread; `rgba` is published before `status` flips to
/// done (acquire/release), so a done image always has its pixels.
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
    /// steps (a generating image aborts) and by `nextPending` (a queued image is
    /// dropped before it starts).
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
    // Requested generation params (from the tool call; defaults from config).
    // Seed 0 means "assign a fresh one at dispatch".
    req_width: usize = 1024,
    req_height: usize = 1024,
    req_steps: usize = 20,
    req_seed: u64 = 0,

    pub fn get(self: *const GenImage) GenStatus {
        return @enumFromInt(self.status.load(.acquire));
    }
};

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
        for (self.images.items) |gi| freeGenImage(gpa, gi);
        for (self.attachments.items) |gi| freeGenImage(gpa, gi);
        self.images.deinit(gpa);
        self.attachments.deinit(gpa);
    }
};

fn freeGenImage(gpa: std.mem.Allocator, gi: *GenImage) void {
    gpa.free(gi.prompt);
    if (gi.rgba) |r| gpa.free(r);
    if (gi.preview) |p| gpa.free(p);
    gpa.destroy(gi);
}

/// Diffusion configuration for the image tool (krea2 by default).
pub const DiffConfig = struct {
    dit_path: []const u8,
    vae_path: []const u8,
    text_encoder_path: []const u8,
    steps: usize = 20,
    width: usize = 1024,
    height: usize = 1024,
    backend: pipeline.Backend = .zig_cuda,
    /// 0 = auto (query live free VRAM); weights past the cap stream per step
    /// so diffusion coexists with the resident LLM.
    vram_budget: u64 = 0,
    /// Optional taew2_1 approx-VAE for a sharper live preview (else latent2rgb).
    taew_path: ?[]const u8 = null,
    /// Show a live preview while sampling. When false, no per-step preview is
    /// computed (the "None" preview method). When true, `taew_path` selects
    /// TAESD vs. the built-in latent2rgb fallback.
    preview_enabled: bool = true,
};

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
    /// tool description is appended to it when `diff` is set.
    system_prompt: []const u8 = "You are a helpful assistant.",
    /// KV window ceiling; growth commits rows lazily up to this.
    max_context: usize = 16384,
    max_new_tokens: usize = 2048,
    seed: u64 = 0,
    temperature: f32 = 0.7,
    /// Image generation config; null disables the image tool.
    diff: ?DiffConfig = null,
    /// mmproj (vision tower) GGUF; enables dropping images to chat about them.
    /// Requires a CUDA backend (which this session always uses).
    mmproj_path: ?[]const u8 = null,
    /// Max VRAM (bytes) the chat model may keep resident before layers migrate
    /// to the CPU as context grows. 0 = auto (use the live free VRAM at load).
    /// See GUI_VRAM.md.
    vram_limit_bytes: u64 = 0,
    /// Who gets VRAM preference when chat + image generation compete.
    vram_priority: config.Priority = .chat,
};

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
    messages: std.ArrayList(Message) = .empty,

    // Worker <-> UI marshalling.
    mu: std.Io.Mutex = std.Io.Mutex.init,
    pending: std.ArrayList(u8) = .empty,
    generating: std.atomic.Value(bool) = .init(false),
    cancel: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,
    gen_err: ?anyerror = null,
    sink_buf: [256]u8 = undefined,

    // Image generation (null diff_opts = tool disabled). One diffusion runs at
    // a time on its own thread + CUDA context, alongside the resident LLM.
    diff_opts: ?pipeline.Options = null,
    diff_seed: u64 = 0,
    diff_busy: std.atomic.Value(bool) = .init(false),
    diff_thread: ?std.Thread = null,
    /// Persistent diffusion pipeline (loads the image model once and stays
    /// resident across a queue of images, so the 2nd+ image skips the reload;
    /// GUI_VRAM.md Phase 4). Created lazily by the first image's worker, reused
    /// by later workers, freed when the queue drains (releasing its VRAM).
    diff_session: std.atomic.Value(?*pipeline.Session) = .init(null),
    /// The taew (approx-VAE) path duped at init, kept so `updateSettings` can
    /// re-enable the TAESD preview live without re-duping (diff_opts.taew_path is
    /// nulled when the preview method isn't TAESD). Null if none was configured.
    taew_owned: ?[]const u8 = null,
    /// Who wins VRAM when chat and image gen compete (see GUI_VRAM.md). Read by
    /// the VRAM coordinator; updatable live from settings.
    vram_priority: config.Priority = .chat,
    /// The resolved LLM VRAM budget (bytes) — the configured limit, or the live
    /// free VRAM when the config limit is 0 (auto). The offload scheduler keeps
    /// LLM device usage under this.
    vram_budget: u64 = 0,
    /// The RAW configured VRAM limit (bytes; 0 = auto/no cap). Unlike
    /// `vram_budget` this is NOT resolved to free VRAM, so it can gate whether a
    /// cap applies. It's a SHARED ceiling: diffusion's resident-weight budget is
    /// `vram_limit − (LLM resident)`, so LLM + image model together stay under it.
    vram_limit: u64 = 0,
    /// Current live-preview method (derived from DiffConfig at init; updated by
    /// settings). Kept so `refreshPreview` can set diff_opts.preview/taew_path
    /// consistently after either a settings change or a diffusion-model swap.
    preview_method: config.Preview = .taesd,
    /// Backs the live diffusion path strings after a model swap (reset+re-dupe
    /// per swap, so it doesn't grow). Initial paths point into the caller's arena.
    diff_path_store: std.heap.ArenaAllocator = undefined,
    /// A requested diffusion-model swap awaiting an idle image queue (in-flight +
    /// queued images finish on the current model, then this applies). gpa-owned.
    diff_paths_pending: ?PendingDiffPaths = null,
    /// True while LLM layers have been evicted to the CPU to make VRAM room for
    /// image generation (priority = image). Cleared once they've been promoted
    /// back (queue drained) or on a new chat.
    image_evicted: bool = false,

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

        self.be = try cuda.Backend.init(arena);
        errdefer self.be.deinit();

        self.mmproj_gguf = null;
        if (cfg.mmproj_path) |mp| self.mmproj_gguf = try Gguf.open(arena, io, mp);

        // Interactive session: grow from a small floor toward the full window.
        const cap: engine.Capacity = .{
            .initial = @min(cfg.max_context, 4096),
            .max = cfg.max_context,
        };

        // Architecture dispatch: each variant bundles {lm, model, vit}. The
        // model retains a `*const lm` into the union, which is stable (self is
        // heap-pinned and the tag is set once). Vision towers are scoped
        // per-encode and never stay resident under the LLM.
        if (std.mem.eql(u8, arch_str, "qwen35")) {
            chat.setFamily(.chatml);
            self.arch = .{ .qwen35 = .{ .lm = try qwen35.Model.load(arena, &self.gguf), .model = undefined } };
            const a = &self.arch.qwen35;
            errdefer a.lm.deinit();
            if (self.mmproj_gguf) |*mg| a.vit = try vit35.Vit.load(arena, mg);
            errdefer if (a.vit) |*v| v.deinit();
            a.model = try qwen35_cuda.CudaLM.init(gpa, self.be, &a.lm, cap);
        } else if (std.mem.eql(u8, arch_str, "gemma3")) {
            chat.setFamily(.gemma);
            self.arch = .{ .gemma3 = .{ .lm = try gemma3.Model.load(arena, &self.gguf), .model = undefined } };
            const a = &self.arch.gemma3;
            errdefer a.lm.deinit();
            if (self.mmproj_gguf) |*mg| a.vit = try gemma_vit.Vit.load(arena, mg);
            errdefer if (a.vit) |*v| v.deinit();
            a.model = try gemma3_cuda.CudaLM.init(gpa, self.be, &a.lm, cap);
        } else if (std.mem.eql(u8, arch_str, "gemma4")) {
            chat.setFamily(.gemma4);
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
            .max_context = cfg.max_context,
            .seed = cfg.seed,
            .sampling = .{ .temperature = cfg.temperature },
        };

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

        self.diff_busy = .init(false);
        self.diff_thread = null;
        self.diff_session = .init(null);
        self.diff_seed = @truncate(cfg.seed +% 0x9E3779B97F4A7C15);
        self.attach_view = .empty;
        self.attach_rgb = .empty;
        self.turn_text = "";
        self.turn_images = .empty;
        self.diff_opts = if (cfg.diff) |d| .{
            .prompt = "",
            .width = d.width,
            .height = d.height,
            .steps = d.steps,
            .backend = d.backend,
            .vram_budget = d.vram_budget,
            .dit_path = d.dit_path,
            .vae_path = d.vae_path,
            .text_encoder_path = d.text_encoder_path,
            .preview = d.preview_enabled, // live preview while sampling (config)
            .taew_path = d.taew_path, // taew2_1 approx-VAE if available, else latent2rgb
        } else null;
        self.taew_owned = if (cfg.diff) |d| d.taew_path else null;
        self.vram_priority = cfg.vram_priority;
        self.preview_method = if (cfg.diff) |d| blk: {
            if (!d.preview_enabled) break :blk config.Preview.none;
            break :blk if (d.taew_path != null) config.Preview.taesd else config.Preview.latent2rgb;
        } else config.Preview.none;
        self.diff_path_store = std.heap.ArenaAllocator.init(gpa);
        self.diff_paths_pending = null;
        self.image_evicted = false;

        // Dynamic VRAM offload (GUI_VRAM.md): always arm the dynamic split (free
        // when the model fits — 0 layers on CPU, per-op decode ties the graph;
        // measured 76 vs 77 tok/s on the 9B). Once context + weights outgrow the
        // budget it migrates layers to the CPU, which is ~2.5x FASTER than the
        // weight-streaming fallback on this box (18 vs 7 tok/s). Vision sessions
        // arm it too: text turns run the fast offload path, and an IMAGE turn
        // promotes every layer back to the GPU first (buildAndPrefillTurn) because
        // the host layer path uses scalar RoPE (wrong for image-grid M-RoPE).
        // budget = the configured limit, or the live free VRAM when 0 (auto).
        self.vram_limit = cfg.vram_limit_bytes; // raw cap (0 = auto); shared with diffusion
        {
            // The LLM loads fully into its budget (the whole limit, or free VRAM
            // when unlimited) — the priority split is not a hard reservation. Under
            // `balanced`, the LLM only gives up its share when the image model
            // actually loads (imageVramEnter → offloadToBudget); until then it uses
            // all the room it has.
            self.vram_budget = llmBudget(cfg.vram_limit_bytes, self.be.ctx.memGetInfo().free);
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
        if (self.diff_opts != null) {
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

    /// Start a fresh conversation, keeping the loaded model resident: drop all
    /// messages/attachments, reset the KV cache, and restore `ids` to the init
    /// prompt prefix. No-op while a turn is generating (the UI disables it then);
    /// any in-flight or queued image generation is canceled first.
    pub fn reset(self: *Session) void {
        if (self.busy()) return;

        // Stop and reap any diffusion so its worker can't touch the messages
        // (and the GenImages it holds) as we free them below.
        for (self.messages.items) |*m| {
            for (m.images.items) |gi| gi.cancel.store(true, .release);
        }
        if (self.diff_thread) |t| {
            t.join();
            self.diff_thread = null;
            self.diff_busy.store(false, .release);
        }
        // Free the resident diffusion model (binds its own CUDA context); the LLM
        // context is re-bound below before resetResidency.
        if (self.diff_session.load(.acquire)) |s| {
            s.deinit();
            self.diff_session.store(null, .release);
        }

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

    pub fn imagesEnabled(self: *const Session) bool {
        return self.diff_opts != null;
    }

    /// Ask the running generation to stop at the next token (no-op if idle).
    pub fn requestCancel(self: *Session) void {
        self.cancel.store(true, .release);
    }

    /// Apply the non-load-affecting settings live (no reload, so the chat is
    /// preserved): image-generation defaults, the preview method, and the VRAM
    /// priority. Called on the UI thread when the user hits Apply and the model
    /// set / VRAM limit are unchanged. The diffusion-facing fields are only
    /// touched when no diffusion is in flight — the worker copies `diff_opts` at
    /// spawn, so skipping a rare concurrent update avoids a torn `taew_path`.
    pub fn updateSettings(self: *Session, cfg: *const config.Config) void {
        // Priority takes effect at the next image-gen contention (imageVramEnter
        // reads it live); the LLM's budget doesn't depend on it (see llmBudget).
        self.vram_priority = cfg.vram_priority;
        if (self.diff_busy.load(.acquire)) return;
        if (self.diff_opts) |*d| {
            d.steps = cfg.steps;
            d.width = cfg.width;
            d.height = cfg.height;
        }
        self.preview_method = cfg.preview;
        self.refreshPreview();
    }

    /// Reconcile the live-preview fields of `diff_opts` with `preview_method` and
    /// the current taew path. Called after a settings change or a diffusion-model
    /// swap so the two derived fields stay consistent.
    fn refreshPreview(self: *Session) void {
        if (self.diff_opts) |*d| {
            d.preview = self.preview_method != .none;
            d.taew_path = if (self.preview_method == .taesd) self.taew_owned else null;
        }
    }

    /// Request a diffusion-model swap (new dit/vae/text-encoder/taesd paths). The
    /// swap is DEFERRED until the image queue is idle, so any in-flight or queued
    /// image finishes on the current model; then `maybeApplyDiffPaths` applies it.
    /// No-op if the image tool isn't enabled (that toggle needs a reload instead).
    /// Path args are borrowed (duped here).
    pub fn requestDiffPaths(self: *Session, dit: []const u8, vae: []const u8, te: []const u8, taew: ?[]const u8) void {
        if (self.diff_opts == null) return;
        self.freePendingDiffPaths();
        const p: PendingDiffPaths = .{
            .dit = self.gpa.dupe(u8, dit) catch return,
            .vae = self.gpa.dupe(u8, vae) catch return,
            .te = self.gpa.dupe(u8, te) catch return,
            .taew = if (taew) |t| (self.gpa.dupe(u8, t) catch null) else null,
        };
        self.diff_paths_pending = p;
        self.maybeApplyDiffPaths();
    }

    fn freePendingDiffPaths(self: *Session) void {
        if (self.diff_paths_pending) |p| {
            self.gpa.free(p.dit);
            self.gpa.free(p.vae);
            self.gpa.free(p.te);
            if (p.taew) |t| self.gpa.free(t);
            self.diff_paths_pending = null;
        }
    }

    /// Apply a pending diffusion-model swap once the image queue has drained.
    /// Called from `requestDiffPaths` and each `pumpDiffusion`. Repoints the
    /// diff_opts path slices at the resettable `diff_path_store` arena.
    fn maybeApplyDiffPaths(self: *Session) void {
        const p = self.diff_paths_pending orelse return;
        if (self.diff_busy.load(.acquire)) return; // an image is generating
        if (self.nextPending() != null) return; // more queued
        var d = &(self.diff_opts orelse return);
        _ = self.diff_path_store.reset(.retain_capacity);
        const a = self.diff_path_store.allocator();
        d.dit_path = a.dupe(u8, p.dit) catch return;
        d.vae_path = a.dupe(u8, p.vae) catch return;
        d.text_encoder_path = a.dupe(u8, p.te) catch return;
        self.taew_owned = if (p.taew) |t| (a.dupe(u8, t) catch null) else null;
        self.refreshPreview(); // taew_path follows the (possibly new) taew_owned
        self.freePendingDiffPaths();
        // Free the resident (old-model) pipeline so the next image reloads with
        // the new paths. Without this, a swap requested while the queue is idle
        // repoints diff_opts but leaves the stale model loaded — the swap would
        // never take effect (short of a new chat). Safe here: this branch only
        // runs with no worker in flight (diff_busy false, nothing queued), so
        // nothing is touching diff_session. deinit binds its own CUDA context.
        if (self.diff_session.load(.acquire)) |s| {
            s.deinit();
            self.diff_session.store(null, .release);
        }
        std.log.info("diffusion model switched", .{});
    }

    /// Estimate the image model's resident footprint (bytes) from its file
    /// sizes — the target VRAM to free for it under image priority. 0 if unknown.
    fn estimateImageBytes(self: *Session) u64 {
        const d = self.diff_opts orelse return 0;
        var total: u64 = 0;
        for ([_][]const u8{ d.dit_path, d.vae_path, d.text_encoder_path }) |p| {
            const st = std.Io.Dir.cwd().statFile(self.io, p, .{}) catch continue;
            total += st.size;
        }
        return total;
    }

    /// Make VRAM room for the image model per the priority, as the image queue
    /// starts (i.e. on CONTENTION — until an image loads, the LLM keeps all its
    /// VRAM). No-op while the LLM is generating (migrations run on this UI thread
    /// and must not race the worker's CUDA context) or when offload is disabled.
    ///  - `image`: evict LLM layers only as needed to fit the image model
    ///    resident, then promote them back when the queue drains (imageVramExit).
    ///  - `balanced`: settle the LLM to 75% of the VRAM limit (ONCE — it stays
    ///    there: no per-image shuffle, no promote-back), leaving ~25% for the
    ///    image model, which streams the rest (diffusion tolerates that well).
    ///    Only under a limit; unlimited/auto keeps both resident, no settling.
    ///  - `chat`: nothing — the LLM keeps its VRAM, the image streams in the rest.
    fn imageVramEnter(self: *Session) void {
        if (self.vram_budget == 0 or self.busy()) return;
        switch (self.vram_priority) {
            .chat => {},
            .image => {
                const need = self.estimateImageBytes();
                if (need == 0) return;
                self.be.bindThread();
                switch (self.arch) {
                    inline else => |*a| {
                        if (a.model.split == null)
                            a.model.enableCpuSplit(.attn, self.vram_budget, true) catch return;
                        a.model.offloadUntilFree(need) catch |err|
                            std.log.warn("image-priority evict failed: {t}", .{err});
                    },
                }
                self.image_evicted = true; // promote back when the queue drains
            },
            .balanced => {
                if (self.vram_limit == 0) return; // no limit → both fit; no settling
                const target = self.vram_limit / 4 * 3; // 75% LLM / 25% image
                self.be.bindThread();
                switch (self.arch) {
                    inline else => |*a| {
                        // Cap the LLM's ONGOING offload ceiling at 75% too, so as
                        // chat continues (KV grows) it stays within its share while
                        // the image model is kept resident — not just a one-time
                        // settle. Restored to the full limit on new-chat (reset).
                        if (a.model.split) |*sp| sp.budget = target;
                        a.model.offloadToBudget(target) catch |err|
                            std.log.warn("balanced settle failed: {t}", .{err});
                    },
                }
                // No image_evicted flag: the LLM stays settled at ~75% (stable).
            },
        }
    }

    /// (image priority) Migrate LLM layers back onto the GPU after the image
    /// queue drains, keeping the image model resident (promoteLayers stops before
    /// overrunning the budget). No-op unless layers were evicted and the LLM is
    /// idle (same CUDA-context constraint as imageVramEnter).
    fn imageVramExit(self: *Session) void {
        if (!self.image_evicted or self.busy()) return;
        self.be.bindThread();
        switch (self.arch) {
            inline else => |*a| _ = a.model.promoteLayers(self.vram_budget) catch |err|
                std.log.warn("image-priority promote failed: {t}", .{err}),
        }
        self.image_evicted = false;
    }

    /// pipeline reclaim hook (GUI_VRAM.md Phase 5): free LLM VRAM for a large VAE
    /// decode by migrating chat layers to the CPU — done even under chat priority,
    /// as the agreed last resort so a big image never just fails. Runs on the
    /// DIFFUSION thread, so it binds the LLM context and is safe only when the LLM
    /// is idle (else it declines and the pipeline falls back to a CPU decode).
    /// `imageVramExit` promotes the layers back once the queue drains.
    fn imageReclaim(self: *Session, needed: u64) bool {
        _ = needed; // free as much as possible
        if (self.busy() or self.vram_budget == 0) return false;
        self.be.bindThread();
        const before = self.be.deviceUsed();
        switch (self.arch) {
            inline else => |*a| {
                if (a.model.split == null)
                    a.model.enableCpuSplit(.attn, self.vram_budget, true) catch return false;
                a.model.offloadUntilFree(std.math.maxInt(u64)) catch {}; // migrate all it can
            },
        }
        const freed = self.be.deviceUsed() < before;
        if (freed) self.image_evicted = true;
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
        if (self.diff_thread) |t| t.join();
        // Free the resident diffusion model first (binds/destroys its own CUDA
        // context), then re-bind the LLM context so the LLM teardown below runs
        // against the right one.
        if (self.diff_session.load(.acquire)) |s| {
            s.deinit();
            self.diff_session.store(null, .release);
            self.be.bindThread();
        }
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
        self.freePendingDiffPaths();
        self.diff_path_store.deinit();
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

    /// True while an image is generating (status-bar diffusion readout).
    pub fn diffusing(self: *Session) bool {
        return self.diff_busy.load(.acquire);
    }

    /// Device VRAM (bytes) the resident diffusion model actually holds — the
    /// accurate figure from the diffusion backend, not an NVML-minus-LLM proxy
    /// (which would also count the desktop's VRAM). 0 when no image model is
    /// loaded. Read by the status bar.
    pub fn diffVramBytes(self: *Session) u64 {
        return if (self.diff_session.load(.acquire)) |s| s.deviceUsed() else 0;
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
            .gemma3 => |*a| @as(u64, a.model.cached()) * a.model.cfg.kvDim() * 8 * a.model.cfg.n_layers,
            // gemma4 KV dim varies per layer (SWA 2048 vs global 512), so sum it.
            .gemma4 => |*a| blk: {
                var kv: u64 = 0;
                for (0..a.model.cfg.n_layers) |l| kv += a.model.cfg.kvDim(l);
                break :blk @as(u64, a.model.cached()) * kv * 8;
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

    /// Collect every finished image in the conversation (attachments then
    /// generated, per message, in order) — the viewer navigates this list.
    pub fn collectImages(self: *const Session, buf: *std.ArrayList(*GenImage)) !void {
        buf.clearRetainingCapacity();
        for (self.messages.items) |*m| {
            for (m.attachments.items) |gi| if (gi.get() == .done) try buf.append(self.gpa, gi);
            for (m.images.items) |gi| if (gi.get() == .done) try buf.append(self.gpa, gi);
        }
    }

    /// Append a user turn and spawn the worker to stream the reply. No-op if a
    /// turn is already generating or there is nothing to send.
    pub fn submit(self: *Session, text: []const u8) !void {
        if (self.busy()) return;
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len == 0 and self.attach_view.items.len == 0) return;

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
        switch (self.arch) {
            inline else => |*a| {
                _ = engine.generate(&a.model, &self.tok, self.io, self.gpa, &self.ids, self.opts, &sink.iface) catch |err| {
                    self.gen_err = err;
                    std.log.err("generation failed: {t}", .{err});
                };
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

        var image_rows: std.ArrayList(usize) = .empty;
        defer image_rows.deinit(self.gpa);
        try chat.appendUserSegments(&self.tok, self.gpa, segs.items, &self.ids, &image_rows);
        try chat.openAssistant(&self.tok, self.gpa, &self.ids);

        if (self.ids.items.len > a.model.cached() + a.model.remaining()) {
            try a.model.ensureCapacity(self.ids.items.len);
        }
        for (image_rows.items, encs.items) |row, e| {
            const before = self.ids.items[a.model.cached()..row];
            if (before.len > 0) try a.model.prefill(before);
            try a.model.prefillImage(e.embeds, e.grid_w, e.grid_h);
        }
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

        var image_rows: std.ArrayList(usize) = .empty;
        defer image_rows.deinit(self.gpa);
        try chat.appendUserSegments(&self.tok, self.gpa, segs.items, &self.ids, &image_rows);
        try chat.openAssistant(&self.tok, self.gpa, &self.ids);

        if (self.ids.items.len > a.model.cached() + a.model.remaining()) {
            try a.model.ensureCapacity(self.ids.items.len);
        }
        for (image_rows.items, encs.items) |row, e| {
            const before = self.ids.items[a.model.cached()..row];
            if (before.len > 0) try a.model.prefill(before);
            try a.model.prefillImage(e.embeds, e.grid_w, e.grid_h);
        }
    }

    fn freeTurn(self: *Session) void {
        if (self.turn_text.len > 0) self.gpa.free(self.turn_text);
        self.turn_text = "";
        for (self.turn_images.items) |im| self.gpa.free(im.rgb);
        self.turn_images.clearRetainingCapacity();
    }

    /// UI-thread, once per frame: move streamed bytes into the live assistant
    /// message and reap a finished worker.
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

        // Once the turn completes, scan it for <image> tool calls, then keep
        // the diffusion queue moving.
        if (self.imagesEnabled() and !self.busy() and self.messages.items.len > 0) {
            const last = &self.messages.items[self.messages.items.len - 1];
            if (last.role == .assistant and !last.images_scanned) {
                last.images_scanned = true;
                self.scanImageCalls(last) catch |err| std.log.err("scan image calls: {t}", .{err});
            }
        }
        self.pumpDiffusion();
    }

    /// Extract `<image ...>PROMPT</image>` tool calls from a finished assistant
    /// message and queue a GenImage (status .pending) for each. Optional tag
    /// attributes (width/height/steps/seed) override the config defaults.
    fn scanImageCalls(self: *Session, msg: *Message) !void {
        const d = self.diff_opts.?;
        const close = "</image>";
        var rest = msg.text.items;
        while (std.mem.indexOf(u8, rest, "<image")) |a| {
            const after_open = rest[a + "<image".len ..];
            const gt = std.mem.indexOfScalar(u8, after_open, '>') orelse break; // incomplete open tag
            const attrs = after_open[0..gt];
            const body = after_open[gt + 1 ..];
            const b = std.mem.indexOf(u8, body, close) orelse break; // incomplete
            const prompt = std.mem.trim(u8, body[0..b], " \n\r\t");
            if (prompt.len > 0) {
                const gi = try self.gpa.create(GenImage);
                gi.* = .{
                    .prompt = try self.gpa.dupe(u8, prompt),
                    .wake = self.wake,
                    .io = self.io,
                    .req_width = d.width,
                    .req_height = d.height,
                    .req_steps = d.steps,
                    .req_seed = 0,
                };
                parseGenAttrs(attrs, gi);
                // Assign a fresh, distinct seed now (unless the tag set one
                // explicitly) so it's known and displayable immediately — even
                // while the image is still queued. Advancing per image keeps
                // repeated generations varied.
                if (gi.req_seed == 0) {
                    self.diff_seed +%= 0x9E3779B97F4A7C15;
                    gi.req_seed = self.diff_seed;
                }
                try msg.images.append(self.gpa, gi);
            }
            rest = body[b + close.len ..];
        }
    }

    fn nextPending(self: *Session) ?*GenImage {
        for (self.messages.items) |*m| {
            for (m.images.items) |gi| {
                if (gi.get() != .pending) continue;
                // Canceled while still queued: drop it before it ever runs.
                if (gi.cancel.load(.acquire)) {
                    gi.status.store(@intFromEnum(GenStatus.canceled), .release);
                    continue;
                }
                return gi;
            }
        }
        return null;
    }

    /// Serialize image generation: reap a finished diffusion, then start the
    /// next pending one (at most one at a time to bound VRAM).
    fn pumpDiffusion(self: *Session) void {
        if (self.diff_thread) |t| {
            if (self.diff_busy.load(.acquire)) return; // still running
            t.join();
            self.diff_thread = null;
        }
        const gi = self.nextPending() orelse {
            // Queue drained. KEEP the diffusion model loaded so a later (not
            // back-to-back) gen reuses it instead of reloading — it's freed ONLY
            // to apply a pending model swap (maybeApplyDiffPaths frees the stale
            // session, so the next gen reloads with the new paths), or on new-chat
            // / reload / teardown (reset/deinit). Under image priority, migrate
            // LLM layers back into whatever room is left beside the still-resident
            // image model (imageVramExit; no-op for chat/balanced).
            self.maybeApplyDiffPaths();
            self.imageVramExit();
            return;
        };
        // Image priority: free VRAM for the image model before it loads (this
        // must precede the diffusion worker, which auto-budgets from live free
        // VRAM). No-op under chat priority or while the LLM is generating.
        self.imageVramEnter();
        // Seed was assigned when the tool call was scanned (see scanImageCalls),
        // so it's already set here.
        // Generous fixed preview buffer (holds any preview up to 512²; taew is
        // ~256px, latent2rgb is latent-res). Dimensions published per step via
        // the atomics; the pointer stays put for the whole generation.
        gi.preview_w = .init(0);
        gi.preview_h = .init(0);
        if (self.gpa.alloc(u8, 512 * 512 * 4)) |pb| {
            @memset(pb, 0);
            gi.preview = pb;
        } else |_| {
            gi.preview = null;
        }
        gi.status.store(@intFromEnum(GenStatus.generating), .release);
        self.diff_busy.store(true, .release);
        self.diff_thread = std.Thread.spawn(.{}, diffWorker, .{ self, gi }) catch {
            gi.status.store(@intFromEnum(GenStatus.failed), .release);
            self.diff_busy.store(false, .release);
            return;
        };
    }

    /// pipeline `Reclaim.call` thunk (recovers the Session from the type-erased ctx).
    fn reclaimThunk(ctx: *anyopaque, needed: u64) bool {
        const self: *Session = @ptrCast(@alignCast(ctx));
        return self.imageReclaim(needed);
    }

    fn diffWorker(self: *Session, gi: *GenImage) void {
        var opts = self.diff_opts.?;
        opts.prompt = gi.prompt;
        opts.width = gi.req_width;
        opts.height = gi.req_height;
        opts.steps = gi.req_steps;
        opts.seed = gi.req_seed;
        opts.on_step = .{ .ctx = gi, .step = onDiffStep };
        opts.cancel = &gi.cancel; // UI Cancel button aborts sampling
        opts.reclaim = .{ .ctx = self, .call = reclaimThunk }; // free LLM VRAM on VAE OOM
        // Share the configured VRAM limit with the LLM: cap diffusion's resident
        // weights at (limit − what the LLM currently holds) so the two together
        // stay under the cap; the rest of the image model streams. Under image
        // priority the LLM was already evicted (imageVramEnter ran before this
        // worker spawned), freeing more headroom. A small floor keeps the budget
        // non-zero even when the LLM fills the limit — 0 would mean "auto / pin
        // all free VRAM" (no cap), so the floor forces streaming instead. When no
        // limit is configured, leave it 0 (auto, as before).
        if (self.vram_limit > 0) {
            opts.vram_budget = @max(256 << 20, self.vram_limit -| self.be.deviceUsed());
        } else {
            opts.vram_budget = 0;
        }
        gi.total.store(@intCast(opts.steps), .monotonic);
        gi.start_ns.store(nowNs(self.io), .release);

        // Load the diffusion model ONCE and keep it resident across the queue:
        // create the session on the first image, reuse it for the rest (the 2nd+
        // image skips the multi-second reload). `pumpDiffusion` frees it when the
        // queue drains, returning its VRAM to the LLM. Reused across successive
        // worker threads — `generate`/`deinit` bind the CUDA context internally.
        var sess = self.diff_session.load(.acquire);
        if (sess == null) {
            sess = pipeline.Session.init(self.io, self.gpa, opts, null) catch |err| {
                std.log.err("diffusion model load failed: {t}", .{err});
                gi.status.store(@intFromEnum(GenStatus.failed), .release);
                self.diff_busy.store(false, .release);
                self.wake();
                return;
            };
            self.diff_session.store(sess, .release);
        }
        var img = sess.?.generate(opts, null) catch |err| {
            const st: GenStatus = if (err == error.Canceled) .canceled else blk: {
                std.log.err("image generation failed: {t}", .{err});
                break :blk .failed;
            };
            gi.status.store(@intFromEnum(st), .release);
            self.diff_busy.store(false, .release);
            self.wake();
            return;
        };
        defer img.deinit(self.gpa);

        // The pipeline returns packed RGB; dvui wants RGBA. Convert once.
        const px = img.width * img.height;
        const rgba = self.gpa.alloc(u8, px * 4) catch {
            gi.status.store(@intFromEnum(GenStatus.failed), .release);
            self.diff_busy.store(false, .release);
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
        self.diff_busy.store(false, .release);
        self.wake();
    }

    fn onDiffStep(ctx: *anyopaque, done: usize, total: usize, preview: ?pipeline.Preview) void {
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
