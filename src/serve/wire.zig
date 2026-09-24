//! The tp-serve protocol: what a client asks an engine host and what the host
//! tells every client, as JSON. Additive evolution is free: readers ignore
//! unknown fields and every field has a default, so a new field is a new
//! optional. Binary payloads (pixels, blob chunks) never travel as JSON; they go
//! over the bulk connection behind a `BinHeader`.
//!
//! Every struct here is constructible as `.{}`, which is what lets the
//! round-trip test walk every variant of both unions without knowing them.
const std = @import("std");
const core = @import("tp_core");
// The three prompt and compat enums live in pipeline.zig, so this is the one
// import here that reaches above tp_core.
const pipeline = @import("TensorPencil").pipeline;
const models = @import("TensorPencil").models;

pub const turn_stats = @import("turn_stats.zig");
pub const TurnStats = turn_stats.TurnStats;

/// Bumped when a change is not additive. `hello` refuses a mismatch by name.
pub const proto: u32 = 3;

pub const ImageId = u64;
/// Names a model for selection and enqueue: derived from what the header-only
/// catalog scan already has, so it costs no read and is stable across a restart.
pub const ModelId = u64;
/// Drawn once at host startup. Every id a host mints is only meaningful under
/// the generation that minted it; a different one on reconnect means the host
/// restarted and every held id names nothing.
pub const HostGen = u64;

pub const Role = enum { user, assistant };

pub const ImageStatus = enum(u8) { pending, generating, done, failed, canceled, suspended };

/// The per-render recipe: the choices that decide what the image LOOKS like
/// without deciding which session renders it. Travels on every enqueue, so a
/// sampler picked after an image was queued cannot retro-apply to it.
/// A conditioning-noise curve, inline so a `RenderParams` stays a plain value an
/// enqueued image can hold by copy. `config.TextBuf` is the same idea one layer
/// up and cannot be used here: `shared` imports `serve`, not the other way.
pub const max_curve = 192;
pub const CurveBuf = FixedStr(max_curve);

/// A fixed-capacity string, so a `RenderParams` stays a plain value an enqueued image
/// can hold by copy. `config.TextBuf` is the same idea one layer up and cannot be used
/// here: `shared` imports `serve`, not the other way.
pub fn FixedStr(comptime cap: usize) type {
    return struct {
        const Self = @This();
        data: [cap]u8 = [_]u8{0} ** cap,

        pub fn slice(self: *const Self) []const u8 {
            return std.mem.sliceTo(&self.data, 0);
        }

        pub fn set(self: *Self, s: []const u8) void {
            @memset(&self.data, 0);
            const n = @min(s.len, self.data.len - 1);
            @memcpy(self.data[0..n], s[0..n]);
        }

        /// A plain JSON string, not an array of `cap` bytes.
        pub fn jsonStringify(self: Self, jws: anytype) !void {
            try jws.write(self.slice());
        }

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Self {
            const s = try std.json.innerParse([]const u8, allocator, source, options);
            var b: Self = .{};
            b.set(s);
            return b;
        }
    };
}

pub const max_path = 512;
pub const PathBuf = FixedStr(max_path);
pub const max_steers = 4;
pub const max_steer_text = 96;

/// One conditioning-steering term on the wire. Fixed storage for the same reason
/// `CurveBuf` is: `RenderParams` is held BY VALUE on a queued image.
pub const SteerSpec = struct {
    text: [max_steer_text]u8 = [_]u8{0} ** max_steer_text,
    scale: f32 = 0,
    mode: pipeline.CondNoise.Mode = .prompt,

    pub fn slice(self: *const SteerSpec) []const u8 {
        return std.mem.sliceTo(&self.text, 0);
    }

    pub fn set(self: *SteerSpec, s: []const u8) void {
        @memset(&self.text, 0);
        const n = @min(s.len, self.text.len - 1);
        @memcpy(self.text[0..n], s[0..n]);
    }

    pub fn jsonStringify(self: SteerSpec, jws: anytype) !void {
        try jws.beginObject();
        try jws.objectField("text");
        try jws.write(self.slice());
        try jws.objectField("scale");
        try jws.write(self.scale);
        try jws.objectField("mode");
        try jws.write(self.mode);
        try jws.endObject();
    }

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !SteerSpec {
        const Flat = struct {
            text: []const u8 = "",
            scale: f32 = 0,
            mode: pipeline.CondNoise.Mode = .prompt,
        };
        const f = try std.json.innerParse(Flat, allocator, source, options);
        var out: SteerSpec = .{ .scale = f.scale, .mode = f.mode };
        out.set(f.text);
        return out;
    }
};

pub const RenderParams = struct {
    sampler: core.sampler.Kind = .euler,
    /// null = the architecture's own default schedule.
    scheduler: ?core.sampler.Scheduler = null,
    prompt_syntax: pipeline.PromptSyntax = .comfy,
    emphasis: pipeline.Emphasis = .original,
    compat: pipeline.Compat = .comfy,
    /// Conditioning noise. Empty curve is off, which is the default, so an
    /// ordinary render is unaffected. See `pipeline.CondNoise`.
    cond_noise: CurveBuf = .{},
    cond_noise_amount: f32 = 1,
    cond_noise_shape: pipeline.CondNoise.Shape = .independent,
    cond_noise_keep_norm: bool = false,
    cond_noise_negative: f32 = 0,
    /// Steering terms, `steer_count` of them live. See `pipeline.CondNoise.Steer`.
    steers: [max_steers]SteerSpec = @splat(.{}),
    steer_count: u8 = 0,
    /// Residual-stream steering (`dit.ActSteer`). A PATH, because the directions are
    /// ~700 KB and belong on the host's disk, not in a render request. Empty is off.
    act_dirs: PathBuf = .{},
    act_scale: f32 = 0,
    act_op: models.dit.ActSteer.Op = .add,
    act_curve: CurveBuf = .{},
    act_keep_norm: bool = true,

    /// Read the live recipe back out of `opts`, the single store for it.
    pub fn from(opts: *const pipeline.Options) RenderParams {
        var p: RenderParams = .{
            .sampler = opts.sampler,
            .scheduler = opts.scheduler,
            .prompt_syntax = opts.prompt_syntax,
            .emphasis = opts.emphasis,
            .compat = opts.compat,
            .cond_noise_amount = opts.cond_noise.amount,
            .cond_noise_shape = opts.cond_noise.shape,
            .cond_noise_keep_norm = opts.cond_noise.keep_norm,
            .cond_noise_negative = opts.cond_noise.neg_scale,
        };
        p.cond_noise.set(opts.cond_noise.curve);
        p.act_scale = opts.act_steer.scale;
        p.act_op = opts.act_steer.op;
        p.act_keep_norm = opts.act_steer.keep_norm;
        for (opts.cond_noise.steers) |t| {
            if (p.steer_count == max_steers) break;
            p.steers[p.steer_count].set(t.text);
            p.steers[p.steer_count].scale = t.scale;
            p.steers[p.steer_count].mode = t.mode;
            p.steer_count += 1;
        }
        return p;
    }

    /// ⚠️ `opts.cond_noise.curve` points INTO `self`, so an `opts` built here
    /// must not outlive the `RenderParams` it came from. Every caller holds the
    /// params for the whole render, which is why this is safe and why the
    /// buffer is inline rather than a slice onto someone else's memory.
    /// `scratch` receives the steering terms and `opts` points at it, so it must
    /// live as long as `opts` is used -- same rule as the curve buffer above.
    pub fn applyTo(
        self: *const RenderParams,
        opts: *pipeline.Options,
        scratch: *[max_steers]pipeline.CondNoise.Steer,
    ) void {
        opts.sampler = self.sampler;
        opts.scheduler = self.scheduler;
        opts.prompt_syntax = self.prompt_syntax;
        opts.emphasis = self.emphasis;
        opts.compat = self.compat;
        opts.cond_noise = .{
            .curve = self.cond_noise.slice(),
            .amount = self.cond_noise_amount,
            .shape = self.cond_noise_shape,
            .keep_norm = self.cond_noise_keep_norm,
            .neg_scale = self.cond_noise_negative,
        };
        // `act_steer.dirs` is NOT set here: the directions are a file the host reads,
        // which `applyTo` has no allocator or io for. The caller loads them and fills
        // the slice in.
        opts.act_steer = .{
            .scale = self.act_scale,
            .op = self.act_op,
            .keep_norm = self.act_keep_norm,
        };
        const n = @min(self.steer_count, max_steers);
        for (0..n) |i| scratch[i] = .{
            .text = self.steers[i].slice(),
            .scale = self.steers[i].scale,
            .mode = self.steers[i].mode,
        };
        opts.cond_noise.steers = scratch[0..n];
    }
};

pub const LoraSpec = struct {
    path: []const u8 = "",
    strength: f32 = 1.0,
};

/// One take of a message. Assistant messages accumulate variants as the user
/// regenerates; user messages always have exactly one.
pub const Variant = struct {
    text: []const u8 = "",
    /// The prompt this take was generated from already had the reasoning block
    /// OPEN, so the text starts inside the thought and only emits the close.
    thought_primed: bool = false,
    /// The reasoning markers of the template that generated this text, and the
    /// model it came from. Empty means "not recorded".
    reason_open: []const u8 = "",
    reason_close: []const u8 = "",
    gen_model: []const u8 = "",
    stats: TurnStats = .{},
    /// Images this take asked for, in emission order.
    images: []const ImageId = &.{},
};

pub const Message = struct {
    role: Role = .user,
    /// A note the app wrote, not the user: an image tool outcome.
    synthetic: bool = false,
    variants: []const Variant = &.{},
    cur: u32 = 0,
    /// Images the user attached to this (user) message.
    attachments: []const ImageId = &.{},
};

pub const ImageRequest = struct {
    /// Echoed back on the first `img` event, so the client can tie the id the
    /// host minted to the row it already drew.
    client_ref: u64 = 0,
    prompt: []const u8 = "",
    negative: []const u8 = "",
    width: u32 = 1024,
    height: u32 = 1024,
    steps: u32 = 20,
    cfg: f32 = 1.0,
    /// 0 = the host assigns a fresh one and reports it on `img`.
    seed: u64 = 0,
    params: RenderParams = .{},
    loras: []const LoraSpec = &.{},
    from_studio: bool = false,
};

/// One render the model asked for, parsed from its reply. The host queues none
/// of these: the client places them like any other render, so a picture the
/// model asked for reaches whichever machine is free.
pub const ImageCall = struct {
    /// Where it was asked, so the answer can be tied back to the reply.
    msg: u32 = 0,
    variant: u32 = 0,
    /// Which call it is within that reply, so a client that already placed it
    /// knows this one is the same one arriving again.
    index: u32 = 0,
    req: ImageRequest = .{},
};

/// Everything a client asks. One verb per public method of the engine's
/// session and image queue; a verb that changes nothing is fine to send twice.
pub const Request = union(enum) {
    hello: struct { proto: u32 = proto },
    /// The whole mirrored state: transcript, queue, state. A fresh client (or a
    /// reconnect) asks for this once, then follows the events after it.
    snapshot,
    /// The host's share of the settings (`config.HostSettings`) as JSON. The
    /// wire does not know its shape: both ends use the settings module's own
    /// type, so the two cannot drift, and the client's own fields never travel.
    settings: struct { json: []const u8 = "" },
    meter: struct { split: f32 = 0.60, limit: f32 = 0.95 },

    /// Derive a set of residual-stream directions from two prompt sets and write
    /// them beside the models. Runs on the engine thread and blocks it for the
    /// duration, which is seconds: an explicit, one-off action.
    act_derive: struct {
        /// Prompts separated by ';'.
        set_a: []const u8 = "",
        set_b: []const u8 = "",
        /// File stem; the host picks the folder and adds `.actd`.
        name: []const u8 = "",
        size: u32 = 256,
        steps: u32 = 8,
    },

    chat_submit: struct { text: []const u8 = "" },
    chat_regenerate,
    chat_cancel,
    chat_pause: struct { paused: bool = false },
    chat_new,
    /// Replace the transcript (a stored conversation reopened by the client).
    chat_adopt: struct { messages: []const Message = &.{} },
    chat_select_variant: struct { msg: u32 = 0, variant: u32 = 0 },
    /// Attach an image the host already holds (a render) to the next message.
    chat_remove_attachment: struct { index: u32 = 0 },
    chat_eject,
    /// Record that this variant's tool call produced this image, so the
    /// transcript carries it wherever the render actually ran. `replaces` is
    /// the image this call was last said to be, when a failure moved it to
    /// another host: the transcript follows the render, it does not collect
    /// every attempt at it.
    chat_image: struct { msg: u32 = 0, variant: u32 = 0, image: ImageId = 0, replaces: ImageId = 0 },
    /// Text for the model to read at the next turn boundary, as its own
    /// message. The client sends it: only the client knows what became of a
    /// render it placed on some other host.
    chat_note: struct { text: []const u8 = "" },

    img_enqueue: ImageRequest,
    img_cancel: struct { image: ImageId = 0 },
    img_cancel_all,
    /// The client holds this image's final state, and its pixels when it had
    /// any. Nothing here needs it after this.
    img_ack: struct { image: ImageId = 0 },
    img_move: struct { image: ImageId = 0, before: ?ImageId = null },
    img_pause: struct { paused: bool = false },
    img_eject,
    /// Ask for pixels: the live preview (skipped when the host's `preview_rev`
    /// still equals `have_rev`), box-filtered so the longer side is at most
    /// `max_edge` (0 = full size), or the finished image. Answered with one
    /// binary frame, or nothing when there is nothing yet.
    img_fetch: struct { image: ImageId = 0, kind: PixelKind = .preview, have_rev: u32 = 0, max_edge: u32 = 0 },

    /// Scan these folders and probe these loose files (the host's own disk;
    /// the local child's). A scan already running is finished first.
    scan: struct { dirs: []const []const u8 = &.{}, files: []const []const u8 = &.{} },
};

pub const PixelKind = enum { preview, pixels };

/// The small block of session and engine state a client renders from. Sent
/// whole whenever any of it changes; it is tens of bytes.
pub const State = struct {
    loading: bool = false,
    /// Error name of the last failed load, "" when the last load succeeded.
    load_err: []const u8 = "",
    llm_resident: bool = false,
    /// The resident model's name, "" when none is loaded.
    llm_model: []const u8 = "",
    /// The resident model's reasoning markers, "" when it has none. A variant
    /// that recorded its own uses those instead.
    reason_open: []const u8 = "",
    reason_close: []const u8 = "",
    /// Error name of the last failed turn, "" while the last one succeeded.
    gen_err: []const u8 = "",
    /// The device context is lost: every turn fails until a restart.
    ctx_lost: bool = false,
    llm_busy: bool = false,
    turn_pending: bool = false,
    llm_paused: bool = false,
    llm_eject_armed: bool = false,
    /// A first message is stashed for the lazy load.
    pending_submit: bool = false,
    /// Images staged for the lazy load, before a session exists.
    staged: u32 = 0,
    /// Attachments waiting on the next message.
    attachments: []const ImageId = &.{},
    /// What the configured or resident model can do.
    vision: bool = false,
    thinking: bool = false,
    reasoning_effort: bool = false,
    weight_noise: bool = false,
    diff_present: bool = false,
    diff_busy: bool = false,
    diff_paused: bool = false,
    diff_eject_armed: bool = false,
    diff_load_err: []const u8 = "",
    /// Family tag of the resident pipeline, "" when nothing is loaded.
    diff_family: []const u8 = "",
    pending_images: u32 = 0,
};

/// One image in the unified list, everything but its pixels. `preview_rev` and
/// `pixels_rev` bump when there is something new to fetch over bulk.
pub const ImageInfo = struct {
    id: ImageId = 0,
    client_ref: u64 = 0,
    status: ImageStatus = .pending,
    step: u32 = 0,
    total: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    req_width: u32 = 0,
    req_height: u32 = 0,
    req_steps: u32 = 0,
    req_cfg: f32 = 1.0,
    req_seed: u64 = 0,
    prompt: []const u8 = "",
    negative: []const u8 = "",
    params: RenderParams = .{},
    from_studio: bool = false,
    /// Error name, "" while it has not failed.
    failure: []const u8 = "",
    preview_w: u32 = 0,
    preview_h: u32 = 0,
    preview_rev: u32 = 0,
    pixels_rev: u32 = 0,
    start_ns: i64 = 0,
    first_step_ns: i64 = 0,
    last_step_ns: i64 = 0,
    done_ns: i64 = 0,
    /// Family tag and model stem, for the metadata a client writes into its PNG.
    family: []const u8 = "",
    model_stem: []const u8 = "",
    /// The rest of that metadata: which encoders, VAE and LoRAs a render used,
    /// how the weights were stored, and the shift. The CLIENT writes the file, so
    /// anything the block records has to cross the wire; the host is the only side
    /// that knows what was actually loaded.
    clip1_stem: []const u8 = "",
    clip2_stem: []const u8 = "",
    vae_stem: []const u8 = "",
    model_hash: []const u8 = "",
    vae_hash: []const u8 = "",
    weight_dtype: []const u8 = "",
    shift: f32 = 0,
    /// Name, AutoV2 hash and strength per LoRA, in the order they were applied.
    loras: []const LoraInfo = &.{},
};

pub const LoraInfo = struct {
    name: []const u8 = "",
    hash: []const u8 = "",
    strength: f32 = 1.0,
};

pub const Telemetry = struct {
    cpu: f32 = 0,
    cpu_mhz: f32 = 0,
    gpu_util: f32 = 0,
    gpu_mhz: u32 = 0,
    vram_used: u64 = 0,
    vram_total: u64 = 0,
    vram_proc: u64 = 0,
    have_gpu: bool = false,
    llm_used: u64 = 0,
    llm_host: u64 = 0,
    ctx_tokens: u64 = 0,
    ctx_kv: u64 = 0,
    layers_gpu: u32 = 0,
    layers_cpu: u32 = 0,
    diff_te: u64 = 0,
    diff_dit: u64 = 0,
    diff_latent: u64 = 0,
    diff_vae: u64 = 0,
    diff_off: u64 = 0,
    limit: u64 = 0,
};

pub const ErrCode = enum {
    unauthorized,
    proto_mismatch,
    bad_request,
    no_model,
    out_of_vram,
    load_failed,
    busy,
    refused,
    not_found,
    internal,
};

/// Everything a host tells a client, in the order the host queued it, on the
/// one events socket.
pub const Event = union(enum) {
    hello: struct { proto: u32 = proto, gen: HostGen = 0 },
    state: State,
    /// The whole transcript. Sent after a structural change (adopt, new chat, a
    /// regenerate adding a variant) and in a snapshot; text then streams by `delta`.
    transcript: struct { messages: []const Message = &.{} },
    delta: struct { msg: u32 = 0, variant: u32 = 0, text: []const u8 = "" },
    stats: struct { msg: u32 = 0, variant: u32 = 0, stats: TurnStats = .{} },
    turn_end: struct { msg: u32 = 0, variant: u32 = 0 },
    /// The tool calls a finished reply made, reported once. Nothing is queued
    /// here; the client places each one and answers with `chat_image`.
    img_requested: struct { calls: []const ImageCall = &.{} },
    ctx: struct { tokens: u64 = 0, kv_bytes: u64 = 0 },
    /// The whole unified image list, in creation order (snapshot).
    queue: struct { images: []const ImageInfo = &.{} },
    /// One image changed.
    img: ImageInfo,
    /// A direction set finished (or did not). `path` is where it landed, empty on
    /// failure; `err` says why.
    act_derived: struct {
        path: []const u8 = "",
        err: []const u8 = "",
    },
    telemetry: Telemetry,
    /// The host's model catalog. `json` is the catalog's own index document,
    /// read back by the same code the index file uses, so the wire does not
    /// know its shape; empty when only the scan status moved. `rev` bumps per
    /// finished scan.
    catalog: struct {
        rev: u64 = 0,
        scanning: bool = false,
        files: u32 = 0,
        probed: u32 = 0,
        reused: u32 = 0,
        bad_folders: u32 = 0,
        json: []const u8 = "",
    },
    /// The image pipeline measured a bigger peak than the settings recorded.
    /// The client persists it against the model key; the host keeps no disk.
    diff_peak: struct { peak: u64 = 0, key: u64 = 0 },
    err: struct { code: ErrCode = .internal, text: []const u8 = "" },
    /// Something worth saying that is not a failure: a host-side wait the user
    /// would otherwise read as a hang. The client shows it and keeps nothing.
    notice: struct { tone: Tone = .info, text: []const u8 = "" },
};

/// How loud a `notice` is. Matches the GUI's own toast tones.
pub const Tone = enum { info, warn, err };

/// Header of a binary payload on the bulk connection. Native byte order on both
/// ends; every target we build is little-endian, and the magic is the check.
pub const BinHeader = extern struct {
    magic: u32 = magic_value,
    kind: Kind,
    _pad: u16 = 0,
    id: u64,
    rev: u32,
    w: u32,
    h: u32,
    len: u32,

    pub const magic_value: u32 = 0x31425054; // "TPB1"
    /// The largest edge a header may name. Both fields come off the wire and
    /// every consumer multiplies them by a bytes-per-pixel before it has looked
    /// at the payload, so the frame check holds them to a width that cannot
    /// overflow that product.
    pub const max_edge: u32 = 1 << 16;
    /// `rgb_upload` is the one client-to-host kind: packed RGB to attach to the
    /// next message.
    pub const Kind = enum(u16) { preview_rgba, image_rgba, blob_chunk, rgb_upload };
};

pub fn encodeAlloc(gpa: std.mem.Allocator, value: anytype) ![]u8 {
    return std.json.Stringify.valueAlloc(gpa, value, .{});
}

/// Decodes into `arena`; every string and slice is a copy, so the input may go.
pub fn decode(comptime T: type, arena: std.mem.Allocator, bytes: []const u8) !T {
    return std.json.parseFromSliceLeaky(T, arena, bytes, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

fn roundTrip(comptime U: type) !void {
    inline for (@typeInfo(U).@"union".fields) |f| {
        const v = if (f.type == void) @unionInit(U, f.name, {}) else @unionInit(U, f.name, .{});
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const once = try encodeAlloc(a, v);
        const back = try decode(U, a, once);
        const twice = try encodeAlloc(a, back);
        errdefer std.debug.print("{s}: {s} != {s}\n", .{ f.name, once, twice });
        try testing.expectEqualStrings(once, twice);
        try testing.expectEqualStrings(f.name, @tagName(back));
    }
}

test "every request variant survives the wire" {
    try roundTrip(Request);
}

test "every event variant survives the wire" {
    try roundTrip(Event);
}

test "a populated transcript round-trips with its nested slices" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v: Event = .{ .transcript = .{ .messages = &.{
        .{ .role = .user, .variants = &.{.{ .text = "hi 日本語" }}, .attachments = &.{ 7, 8 } },
        .{ .role = .assistant, .cur = 1, .variants = &.{
            .{ .text = "a" },
            .{ .text = "b", .thought_primed = true, .reason_open = "<think>", .reason_close = "</think>", .images = &.{42}, .stats = .{ .gen_tokens = 3, .decode_ns = 1_000_000_000, .decode_tokens = 3 } },
        } },
    } } };
    const bytes = try encodeAlloc(a, v);
    const back = try decode(Event, a, bytes);
    const msgs = back.transcript.messages;
    try testing.expectEqual(@as(usize, 2), msgs.len);
    try testing.expectEqualStrings("hi 日本語", msgs[0].variants[0].text);
    try testing.expectEqualSlices(ImageId, &.{ 7, 8 }, msgs[0].attachments);
    try testing.expectEqual(@as(u32, 1), msgs[1].cur);
    try testing.expectEqualStrings("</think>", msgs[1].variants[1].reason_close);
    try testing.expectEqual(@as(ImageId, 42), msgs[1].variants[1].images[0]);
    try testing.expectEqual(@as(?f64, 3.0), msgs[1].variants[1].stats.tgRate());
}

test "unknown fields are ignored and missing fields take their defaults" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const req = try decode(Request, a, "{\"chat_submit\":{\"text\":\"x\",\"later_field\":true}}");
    try testing.expectEqualStrings("x", req.chat_submit.text);
    const ev = try decode(Event, a, "{\"img\":{\"id\":9}}");
    try testing.expectEqual(@as(ImageId, 9), ev.img.id);
    try testing.expectEqual(ImageStatus.pending, ev.img.status);
    try testing.expectEqual(core.sampler.Kind.euler, ev.img.params.sampler);
    // Enums travel by name, so a reordered enum is not a wire change.
    const ev2 = try decode(Event, a, "{\"img\":{\"status\":\"suspended\",\"params\":{\"sampler\":\"dpmpp_2m_sde\",\"scheduler\":\"karras\"}}}");
    try testing.expectEqual(ImageStatus.suspended, ev2.img.status);
    try testing.expectEqual(core.sampler.Kind.dpmpp_2m_sde, ev2.img.params.sampler);
    try testing.expectEqual(@as(?core.sampler.Scheduler, .karras), ev2.img.params.scheduler);
}

test "a bad tag or a proto mismatch is an error, not a default" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectError(error.UnknownField, decode(Request, a, "{\"no_such_verb\":{}}"));
    try testing.expectError(error.UnexpectedEndOfInput, decode(Request, a, "{\"hello\":"));
}

test "BinHeader is 32 bytes with the fields where the other end expects them" {
    try testing.expectEqual(@as(usize, 32), @sizeOf(BinHeader));
    try testing.expectEqual(@as(usize, 0), @offsetOf(BinHeader, "magic"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(BinHeader, "kind"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(BinHeader, "id"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(BinHeader, "rev"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(BinHeader, "len"));
    const h: BinHeader = .{ .kind = .image_rgba, .id = 1, .rev = 2, .w = 3, .h = 4, .len = 5 };
    const bytes: [32]u8 = @bitCast(h);
    try testing.expectEqualSlices(u8, "TPB1", bytes[0..4]);
}

test "a conditioning-noise curve survives the wire as a string" {
    // `CurveBuf` is hand-written JSON glue, and the round-trip test above walks
    // every variant at its DEFAULT, which for a curve is empty. A curve with the
    // commas that `clamp`/`min`/`max` need is the case that would break.
    const gpa = testing.allocator;
    var p: RenderParams = .{ .cond_noise_amount = 0.42, .cond_noise_shape = .shared, .cond_noise_keep_norm = true };
    p.cond_noise.set("clamp(a*(1-t),0,1)");

    const json = try encodeAlloc(gpa, p);
    defer gpa.free(json);
    // A plain JSON string, not 192 numbers.
    try testing.expect(std.mem.indexOf(u8, json, "\"clamp(a*(1-t),0,1)\"") != null);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const back = try decode(RenderParams, arena.allocator(), json);
    try testing.expectEqualStrings("clamp(a*(1-t),0,1)", back.cond_noise.slice());
    try testing.expectEqual(@as(f32, 0.42), back.cond_noise_amount);
    try testing.expectEqual(pipeline.CondNoise.Shape.shared, back.cond_noise_shape);
    try testing.expect(back.cond_noise_keep_norm);

    // Over capacity truncates rather than overflowing, and stays nul-terminated.
    var big: CurveBuf = .{};
    big.set("x" ** (max_curve * 2));
    try testing.expectEqual(max_curve - 1, big.slice().len);
}
