//! chat_template.zig — render a chat transcript into a token-id prompt using
//! the model's OWN embedded Jinja `chat_template` (via `jinja.zig`), instead of
//! the hand-maintained per-family glue in `chat.zig`.
//!
//! Why: the hand glue re-serialized every past assistant turn verbatim, so a
//! reasoning model's prior-turn thought blocks accumulated in context forever
//! and the model degraded after a few turns (see TODO #1). The real templates
//! strip prior thoughts (`strip_thinking`), handle the system/`<|think|>` cue,
//! BOS, tool/image placeholders, etc. — rendering the model's template is the
//! single source of truth and fixes that class of drift generically.
//!
//! The rendered string is tokenized with the tokenizer's special-token scanner
//! (one special-aware pass — the same way llama.cpp/transformers tokenize a
//! templated prompt), so template markers (`<bos>`, `<|turn>`, `<|channel>`,
//! `<|im_start|>`, `<think>`, …) map to their special ids.

const std = @import("std");
const tp_core = @import("tp_core");
const jinja = tp_core.jinja;
const Tokenizer = tp_core.tokenizer.Tokenizer;
const Gguf = tp_core.gguf.Gguf;

pub const Role = enum {
    system,
    user,
    assistant,

    pub fn str(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
    }
};

/// An image's ViT token grid (its cache-row footprint is `grid_w*grid_h`).
pub const Grid = struct { grid_w: usize, grid_h: usize };

/// One part of a multimodal user turn: literal text, or an image occupying
/// `grid_w*grid_h` cache rows (the ViT token grid).
pub const Part = union(enum) {
    text: []const u8,
    image: Grid,
};

/// One transcript message. `content` is the RAW text (an assistant message
/// keeps its thought block inline — the template's `strip_thinking` removes it
/// from prior turns; we must NOT pre-strip it here). When `parts` is set the
/// message is multimodal: the template renders each part in order (an image
/// part emits the family's single image-placeholder token, later expanded to
/// the real pad-row block by `renderIdsMM`), and `content` is ignored.
pub const Message = struct {
    role: Role,
    content: []const u8 = "",
    parts: ?[]const Part = null,
};

/// How a family's single image-placeholder token expands into the real image
/// block the model was trained on (mirrors `chat.appendGemma4Image` etc.):
/// `prefix` + `n_tokens` × `pad` + `suffix`. Derived from the tokenizer's
/// special ids, so it stays vocab-driven rather than hard-coded.
pub const ImageExpand = struct {
    placeholder: u32,
    // Inline storage (≤2 markers per side for the families we template) so the
    // descriptor is a plain value with no lifetime dependency.
    prefix_buf: [2]u32 = undefined,
    prefix_len: usize = 0,
    suffix_buf: [2]u32 = undefined,
    suffix_len: usize = 0,
    pad: u32,

    fn prefix(self: *const ImageExpand) []const u32 {
        return self.prefix_buf[0..self.prefix_len];
    }
    fn suffix(self: *const ImageExpand) []const u32 {
        return self.suffix_buf[0..self.suffix_len];
    }

    /// The Qwen3-VL / ChatML layout: the template emits
    /// `<|vision_start|><|image_pad|><|vision_end|>`, so only the single
    /// `<|image_pad|>` expands (to N pad rows); the start/end markers are
    /// already literal text around it.
    pub fn chatml(tok: *const Tokenizer) ?ImageExpand {
        const ph = tok.specialId("<|image_pad|>") orelse return null;
        return .{ .placeholder = ph, .pad = ph };
    }

    /// Gemma 3 layout: `<start_of_image>` → itself + `<image_soft_token>`×N +
    /// `<end_of_image>` (the template emits `<start_of_image>` as the single
    /// placeholder).
    pub fn gemma3(tok: *const Tokenizer) ?ImageExpand {
        const ph = tok.specialId("<start_of_image>") orelse return null;
        const soft = tok.specialId("<image_soft_token>") orelse return null;
        const close = tok.specialId("<end_of_image>") orelse return null;
        var e: ImageExpand = .{ .placeholder = ph, .pad = soft };
        e.prefix_buf[0] = ph; // keep the <start_of_image> marker before the rows
        e.prefix_len = 1;
        e.suffix_buf[0] = close;
        e.suffix_len = 1;
        return e;
    }

    /// The gemma4 layout: `<|image|>` → `<|image>` + pad×N + `<image|>`.
    pub fn gemma4(tok: *const Tokenizer) ?ImageExpand {
        const ph = tok.specialId("<|image|>") orelse return null;
        const open = tok.specialId("<|image>") orelse return null;
        const close = tok.specialId("<image|>") orelse return null;
        var e: ImageExpand = .{ .placeholder = ph, .pad = tok.pad };
        e.prefix_buf[0] = open;
        e.prefix_len = 1;
        e.suffix_buf[0] = close;
        e.suffix_len = 1;
        return e;
    }
};

pub const RenderOpts = struct {
    messages: []const Message,
    /// Append the open assistant turn the model completes.
    add_generation_prompt: bool = true,
    /// Reasoning families honor this (drives `<|think|>` / thought priming).
    enable_thinking: bool = true,
    /// The model's BOS string (templates emit `{{ bos_token }}`); "" for none.
    bos_token: []const u8 = "",
};

pub const ChatTemplate = struct {
    tmpl: jinja.Template,

    /// Load the embedded `tokenizer.chat_template`; null if the GGUF has none
    /// (caller falls back to the hand glue).
    pub fn fromGguf(gpa: std.mem.Allocator, g: *const Gguf) !?ChatTemplate {
        const src = g.getStr("tokenizer.chat_template") orelse return null;
        return .{ .tmpl = try jinja.Template.parse(gpa, src) };
    }

    pub fn fromSource(gpa: std.mem.Allocator, src: []const u8) !ChatTemplate {
        return .{ .tmpl = try jinja.Template.parse(gpa, src) };
    }

    pub fn deinit(self: *ChatTemplate) void {
        self.tmpl.deinit();
    }

    /// Render the transcript to a prompt string appended to `out`.
    pub fn renderString(self: *const ChatTemplate, gpa: std.mem.Allocator, opts: RenderOpts, out: *std.ArrayList(u8)) !void {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const globals = try buildGlobals(arena.allocator(), opts);
        self.tmpl.render(gpa, globals, out) catch |e| return mapErr(e);
    }

    /// Render then tokenize (special-aware) into `out`.
    pub fn renderIds(self: *const ChatTemplate, tok: *const Tokenizer, gpa: std.mem.Allocator, opts: RenderOpts, out: *std.ArrayList(u32)) !void {
        var str: std.ArrayList(u8) = .empty;
        defer str.deinit(gpa);
        try self.renderString(gpa, opts, &str);
        try tok.encode(gpa, str.items, out);
    }

    /// Multimodal render: like `renderIds`, but each image part's placeholder
    /// token (one per image, in transcript order) is expanded in-place into the
    /// real pad-row block per `exp`. The first pad row of each image is recorded
    /// in `image_rows` so the caller can splice the ViT embeddings in with
    /// `model.prefillImage`. `grids` gives every image's `grid_w*grid_h`, in the
    /// same order the parts appear — used for the pad-row counts.
    pub fn renderIdsMM(
        self: *const ChatTemplate,
        tok: *const Tokenizer,
        gpa: std.mem.Allocator,
        opts: RenderOpts,
        exp: ImageExpand,
        grids: []const Grid,
        out: *std.ArrayList(u32),
        image_rows: *std.ArrayList(usize),
    ) !void {
        var raw: std.ArrayList(u32) = .empty;
        defer raw.deinit(gpa);
        try self.renderIds(tok, gpa, opts, &raw);
        var k: usize = 0;
        for (raw.items) |id| {
            if (id == exp.placeholder) {
                if (k >= grids.len) return error.ChatTemplateImageMismatch;
                try out.appendSlice(gpa, exp.prefix());
                try image_rows.append(gpa, out.items.len);
                try out.appendNTimes(gpa, exp.pad, grids[k].grid_w * grids[k].grid_h);
                try out.appendSlice(gpa, exp.suffix());
                k += 1;
            } else {
                try out.append(gpa, id);
            }
        }
        if (k != grids.len) return error.ChatTemplateImageMismatch;
    }
};

// Process-global active template + its BOS string and the session system
// prompt, set once at load by the CLI (mirrors `chat.family`/`chat.bos_token`).
// Null `active` => the model shipped no template; callers fall back to the
// hand glue in `chat.zig`.
pub var active: ?ChatTemplate = null;
pub var bos: []const u8 = "";
pub var system_prompt: ?[]const u8 = null;

fn mapErr(e: jinja.Error) anyerror {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ChatTemplateRender,
    };
}

/// Build the jinja globals dict the templates expect (`messages`, the flags,
/// `bos_token`). Allocated in `a` (a scratch arena owned by the caller).
fn buildGlobals(a: std.mem.Allocator, opts: RenderOpts) !jinja.Value {
    const msgs = try a.create(jinja.List);
    msgs.* = .{};
    for (opts.messages) |m| {
        const d = try a.create(jinja.Dict);
        d.* = .{};
        try d.put(a, "role", .{ .str = m.role.str() });
        if (m.parts) |parts| {
            // Multimodal content: an ordered list of {type:"text",text}/{type:
            // "image"} dicts (matches the templates' content-parts branch).
            const list = try a.create(jinja.List);
            list.* = .{};
            for (parts) |p| {
                const pd = try a.create(jinja.Dict);
                pd.* = .{};
                switch (p) {
                    .text => |t| {
                        try pd.put(a, "type", .{ .str = "text" });
                        try pd.put(a, "text", .{ .str = t });
                    },
                    .image => {
                        try pd.put(a, "type", .{ .str = "image" });
                    },
                }
                try list.items.append(a, .{ .dict = pd });
            }
            try d.put(a, "content", .{ .list = list });
        } else {
            try d.put(a, "content", .{ .str = m.content });
        }
        try msgs.items.append(a, .{ .dict = d });
    }
    const g = try a.create(jinja.Dict);
    g.* = .{};
    try g.put(a, "messages", .{ .list = msgs });
    try g.put(a, "add_generation_prompt", .{ .boolean = opts.add_generation_prompt });
    try g.put(a, "enable_thinking", .{ .boolean = opts.enable_thinking });
    try g.put(a, "bos_token", .{ .str = opts.bos_token });
    return .{ .dict = g };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A tiny ChatML template mirroring the llama/qwen shape, so the render→tokenize
// path is testable with the embedded default (Qwen ChatML) tokenizer — no GGUF.
const chatml_src =
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}" ++
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

test "chat_template: render + tokenize round-trips through special tokens" {
    const gpa = std.testing.allocator;
    var ct = try ChatTemplate.fromSource(gpa, chatml_src);
    defer ct.deinit();

    var tok = try Tokenizer.init(gpa); // embedded Qwen ChatML tokenizer
    defer tok.deinit();

    const msgs = [_]Message{
        .{ .role = .system, .content = "You are terse." },
        .{ .role = .user, .content = "Hi" },
    };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ct.renderIds(&tok, gpa, .{ .messages = &msgs }, &ids);

    // The rendered string tokenized as one special-aware pass must equal a
    // direct encode of the same string.
    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    try tok.encode(gpa, "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", &ref);
    try std.testing.expectEqualSlices(u32, ref.items, ids.items);
}

test "chat_template: prior-turn thoughts are stripped, current turn primed" {
    // Uses the real gemma4 embedded template shape via the reference file if
    // present; otherwise a compact stand-in exercising strip_thinking. This is
    // the crux of TODO #1: past <|channel>…<channel|> blocks must NOT appear.
    const gpa = std.testing.allocator;
    const src =
        "{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m['role']=='assistant' else m['role'] }}\n" ++
        "{%- macro strip(t) -%}{% set ns=namespace(r='') %}{% for p in t.split('<channel|>') %}{% if '<|channel>' in p %}{% set ns.r = ns.r + p.split('<|channel>')[0] %}{% else %}{% set ns.r = ns.r + p %}{% endif %}{% endfor %}{{ ns.r | trim }}{%- endmacro -%}" ++
        "{% if m['role']=='assistant' %}{{ strip(m['content']) }}{% else %}{{ m['content'] | trim }}{% endif %}<turn|>\n{% endfor %}" ++
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}";
    var ct = try ChatTemplate.fromSource(gpa, src);
    defer ct.deinit();

    const think = "<|channel>thought\nlong private reasoning\n<channel|>";
    const msgs = [_]Message{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = think ++ "Hello!" },
        .{ .role = .user, .content = "Bye" },
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .bos_token = "<bos>" }, &out);

    errdefer std.debug.print("rendered:\n{s}\n", .{out.items});
    // The past assistant turn keeps only its answer; the thought is gone.
    try std.testing.expect(std.mem.indexOf(u8, out.items, "long private reasoning") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "<|channel>") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "Hello!") != null);
    // Ends primed for the model to answer the latest user turn.
    try std.testing.expect(std.mem.endsWith(u8, out.items, "<|turn>model\n"));
}

// Real GGUF: the actual gemma4 embedded template + tokenizer. Proves the fix
// on the shipping template — prior-turn thoughts vanish from the TOKEN stream,
// not just the string. Self-skips when the checkpoint is absent (mmaps the
// header + tokenizer only; no weights loaded, so it stays in the fast suite).
test "chat_template: real gemma4 GGUF strips prior thoughts from the token stream" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();
    var ct = (try ChatTemplate.fromGguf(gpa, &g)) orelse return error.SkipZigTest;
    defer ct.deinit();

    const think = "<|channel>thought\nSECRETPRIORREASONING\n<channel|>";
    const msgs = [_]Message{
        .{ .role = .system, .content = "You are terse." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = think ++ "Hello there!" },
        .{ .role = .user, .content = "Bye" },
    };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ct.renderIds(&tok, gpa, .{ .messages = &msgs, .bos_token = "<bos>", .enable_thinking = true }, &ids);

    // Decode the token stream back to text and confirm the prior thought is
    // gone while the answer remains.
    const text = try tok.decodeAlloc(gpa, ids.items);
    defer gpa.free(text);
    errdefer std.debug.print("decoded prompt:\n{s}\n", .{text});
    try std.testing.expect(std.mem.indexOf(u8, text, "SECRETPRIORREASONING") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "Hello there!") != null);
}

// Task #4: the multimodal render must expand the template's single image
// placeholder into the model's real image block — `<|image>` + pad×N +
// `<image|>` for gemma4 — with the recorded row pointing at the first pad, so
// ViT embeddings splice in at the right cache rows. (The block matches the hand
// glue's layout; the render is otherwise the authoritative template output,
// which e.g. trims the space before the tag where the old glue kept it.)
test "chat_template: gemma4 image placeholder expands to the real block at the recorded row" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();
    var ct = (try ChatTemplate.fromGguf(gpa, &g)) orelse return error.SkipZigTest;
    defer ct.deinit();

    const gw: usize = 2;
    const gh: usize = 3;
    const n_rows = gw * gh;
    const parts = [_]Part{ .{ .text = "look " }, .{ .image = .{ .grid_w = gw, .grid_h = gh } } };
    const msgs = [_]Message{.{ .role = .user, .parts = &parts }};
    const exp = ImageExpand.gemma4(&tok).?;
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(gpa);
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    try ct.renderIdsMM(&tok, gpa, .{
        .messages = &msgs,
        .bos_token = if (tok.bos) |b| (try tok.decodeAlloc(gpa, &.{b})) else "",
        .enable_thinking = false,
        .add_generation_prompt = true,
    }, exp, &.{.{ .grid_w = gw, .grid_h = gh }}, &got, &rows);

    const open = tok.specialId("<|image>").?;
    const close = tok.specialId("<image|>").?;
    errdefer std.debug.print("got={any}\nrows={any}\n", .{ got.items, rows.items });
    try std.testing.expectEqual(@as(usize, 1), rows.items.len);
    const r = rows.items[0];
    try std.testing.expectEqual(open, got.items[r - 1]); // block opens just before the row
    for (got.items[r .. r + n_rows]) |t| try std.testing.expectEqual(exp.pad, t);
    try std.testing.expectEqual(close, got.items[r + n_rows]);
    // No bare placeholder token survives the expansion.
    try std.testing.expect(std.mem.indexOfScalar(u32, got.items, exp.placeholder) == null);
}

// Task #4 (gemma3): the gemma3 template's image branch emits a single
// `<start_of_image>`, which `ImageExpand.gemma3` expands into the real block
// `<start_of_image>` + `<image_soft_token>`×N + `<end_of_image>` with the row
// pointing at the first soft token. Unlike gemma4 the placeholder token *is*
// the open marker, so we assert exact marker counts + block layout rather than
// "no placeholder survives". Self-skips when the checkpoint is absent.
test "chat_template: gemma3 image placeholder expands to the real block at the recorded row" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();
    var ct = (try ChatTemplate.fromGguf(gpa, &g)) orelse return error.SkipZigTest;
    defer ct.deinit();

    const gw: usize = 2;
    const gh: usize = 3;
    const n_rows = gw * gh;
    const parts = [_]Part{ .{ .text = "look " }, .{ .image = .{ .grid_w = gw, .grid_h = gh } } };
    const msgs = [_]Message{.{ .role = .user, .parts = &parts }};
    const exp = ImageExpand.gemma3(&tok).?;
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(gpa);
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    try ct.renderIdsMM(&tok, gpa, .{
        .messages = &msgs,
        .bos_token = if (tok.bos) |b| (try tok.decodeAlloc(gpa, &.{b})) else "",
        .enable_thinking = false,
        .add_generation_prompt = true,
    }, exp, &.{.{ .grid_w = gw, .grid_h = gh }}, &got, &rows);

    const open = tok.specialId("<start_of_image>").?;
    const soft = tok.specialId("<image_soft_token>").?;
    const close = tok.specialId("<end_of_image>").?;
    errdefer std.debug.print("got={any}\nrows={any}\n", .{ got.items, rows.items });
    try std.testing.expectEqual(@as(usize, 1), rows.items.len);
    const r = rows.items[0];
    try std.testing.expectEqual(open, got.items[r - 1]); // <start_of_image> just before the row
    for (got.items[r .. r + n_rows]) |t| try std.testing.expectEqual(soft, t);
    try std.testing.expectEqual(close, got.items[r + n_rows]);
    // Exactly one open + one close marker survive (the block, not a stray).
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, got.items, &.{open}));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, got.items, &.{close}));
}
