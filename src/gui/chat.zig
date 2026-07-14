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

const qwen35 = tp.models.qwen35;
const qwen35_cuda = tp.models.qwen35_cuda;
const vit35 = tp.models.vit35;
const vit35_cuda = tp.models.vit35_cuda;
const gemma3 = tp.models.gemma3;
const gemma3_cuda = tp.models.gemma3_cuda;
const gemma_vit = tp.models.gemma_vit;
const gemma_vit_cuda = tp.models.gemma_vit_cuda;

/// The resident LLM + its vision tower, one variant per supported GGUF
/// architecture. `model` retains a `*const lm`, so the bundle must live at a
/// stable address (it does — inside the heap-pinned Session) and its tag is
/// never reassigned after init.
const Arch = union(enum) {
    qwen35: struct { lm: qwen35.Model, model: qwen35_cuda.CudaLM, vit: ?vit35.Vit = null },
    gemma3: struct { lm: gemma3.Model, model: gemma3_cuda.CudaLM, vit: ?gemma_vit.Vit = null },
};
const cuda = tp.gpu.cuda;
const engine = tp.llm.engine;
const chat = tp.llm.chat;
const pipeline = tp.pipeline;
const Tokenizer = tp.tokenizer.Tokenizer;
const Gguf = tp.Gguf;

/// Raw decoded image (packed RGB) awaiting vision encoding.
const RawImage = struct { rgb: []u8, width: usize, height: usize };

/// Round to a multiple of 16 (pipeline requirement) within sane bounds.
fn clampDim(n: usize) usize {
    const c = std.math.clamp(n, 256, 2048);
    return c / 16 * 16;
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

        // Gemma prompts begin with a single BOS ({{ bos_token }}); ChatML has none.
        if (self.arch == .gemma3) {
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
        return self;
    }

    pub fn imagesEnabled(self: *const Session) bool {
        return self.diff_opts != null;
    }

    /// Ask the running generation to stop at the next token (no-op if idle).
    pub fn requestCancel(self: *Session) void {
        self.cancel.store(true, .release);
    }

    pub fn deinit(self: *Session) void {
        if (self.worker) |t| t.join();
        if (self.diff_thread) |t| t.join();
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
            switch (self.arch) {
                .qwen35 => |*a| try self.imageTurn(a, false),
                .gemma3 => |*a| try self.imageTurn(a, true),
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
        const gi = self.nextPending() orelse return;
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

    fn diffWorker(self: *Session, gi: *GenImage) void {
        var opts = self.diff_opts.?;
        opts.prompt = gi.prompt;
        opts.width = gi.req_width;
        opts.height = gi.req_height;
        opts.steps = gi.req_steps;
        opts.seed = gi.req_seed;
        opts.on_step = .{ .ctx = gi, .step = onDiffStep };
        opts.cancel = &gi.cancel; // UI Cancel button aborts sampling
        gi.total.store(@intCast(opts.steps), .monotonic);
        gi.start_ns.store(nowNs(self.io), .release);

        var img = pipeline.generate(self.io, self.gpa, opts, null) catch |err| {
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
