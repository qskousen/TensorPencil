//! chat_template.zig, render a chat transcript into a token-id prompt using
//! the model's OWN embedded Jinja `chat_template` (via `jinja.zig`), instead of
//! the hand-maintained per-family glue in `chat.zig`.
//!
//! Why: the hand glue re-serialized every past assistant turn verbatim, so a
//! reasoning model's prior-turn thought blocks accumulated in context forever
//! and the model degraded after a few turns (see TODO #1). The real templates
//! strip prior thoughts (`strip_thinking`), handle the system/`<|think|>` cue,
//! BOS, tool/image placeholders, etc., rendering the model's template is the
//! single source of truth and fixes that class of drift generically.
//!
//! The rendered string is tokenized with the tokenizer's special-token scanner
//! (one special-aware pass, the same way llama.cpp/transformers tokenize a
//! templated prompt), so template markers (`<bos>`, `<|turn>`, `<|channel>`,
//! `<|im_start|>`, `<think>`, ...) map to their special ids.

const std = @import("std");
const tp_core = @import("tp_core");
const jinja = tp_core.jinja;
const Tokenizer = tp_core.tokenizer.Tokenizer;
const Gguf = tp_core.gguf.Gguf;

pub const Role = enum {
    system,
    user,
    assistant,
    /// A tool's RESULT, replayed so the model can see what its call returned.
    /// Not every template accepts one: qwen3.5/Bonsai render it as a user
    /// turn wrapped in `<tool_response>`, gemma4 resolves it back to the calling
    /// function through `tool_call_id`, and gemma3 raises ("Conversation
    /// roles must alternate user/assistant"). Probe with `supportsToolRole`
    /// rather than assuming, the answer is a property of the loaded template.
    tool,

    pub fn str(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

/// A function the model may call, rendered into the template's `tools` global
/// as the OpenAI shape every chat template expects:
/// `{"type":"function","function":{"name":...,"description":...,"parameters":...}}`.
///
/// `parameters_json` is JSON text rather than a Zig type because it *is* a
/// JSON Schema, arbitrarily nested, and the templates emit it straight back
/// out (`{{ tool | tojson }}`), so anything less than free-form JSON would
/// silently truncate a caller's schema.
pub const Tool = struct {
    name: []const u8,
    description: []const u8 = "",
    /// JSON Schema object for the arguments, as JSON text.
    parameters_json: []const u8 = "{}",
};

/// One call the assistant made, replayed on the next turn so the model sees its
/// own request alongside the result. Sits on `Message.tool_calls`.
pub const ToolCall = struct {
    /// Correlates this call with the `.tool` message answering it. Emitted only
    /// when non-empty: gemma4 resolves the *name* of the responding function
    /// through it (`tc.get('id') == follow.get('tool_call_id')`), while
    /// qwen3.5/Bonsai ignore it entirely.
    id: []const u8 = "",
    name: []const u8,
    /// The arguments as a JSON object, in JSON text. A dict rather than a
    /// string on purpose: the templates iterate it (`arguments | items`) to emit
    /// one `<parameter=...>` block per key.
    arguments_json: []const u8 = "{}",
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
/// keeps its thought block inline, the template's `strip_thinking` removes it
/// from prior turns; we must NOT pre-strip it here). When `parts` is set the
/// message is multimodal: the template renders each part in order (an image
/// part emits the family's single image-placeholder token, later expanded to
/// the real pad-row block by `renderIdsMM`), and `content` is ignored.
pub const Message = struct {
    role: Role,
    content: []const u8 = "",
    parts: ?[]const Part = null,
    /// Calls this (assistant) message made. Rendered as the template's trained
    /// tool-call format; `content` carries whatever prose came with them.
    tool_calls: ?[]const ToolCall = null,
    /// On a `.tool` message: which `ToolCall.id` this result answers. Emitted
    /// only when non-empty (see `ToolCall.id`).
    tool_call_id: []const u8 = "",
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

    /// Gemma 3 layout: `<start_of_image>` -> itself + `<image_soft_token>`×N +
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

    /// The gemma4 layout: `<|image|>` -> `<|image>` + pad×N + `<image|>`.
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
    /// The model's EOS string (Mistral/llama templates emit `{{ eos_token }}`
    /// to close each assistant turn); "" for none.
    eos_token: []const u8 = "",
    /// Functions the model may call. Null leaves `tools` undefined, which is
    /// what every template tests (`{%- if tools %}`), so a caller that declares
    /// none renders byte-identically to one that never knew about tools.
    tools: ?[]const Tool = null,
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

    /// Google's current upstream "canonical" Gemma 4 chat template (identical
    /// across the 12B/31B releases). Used by the config override that replaces
    /// a model's own embedded template, some finetunes (e.g. DarkIdol 31B)
    /// ship an older/stripped variant; this renders exactly what Google's
    /// `apply_chat_template` would. Byte-exact vs jinja2 (golden fixtures).
    pub const gemma4_canonical_src = jinja.gemma4_canonical_template;

    pub fn gemma4Canonical(gpa: std.mem.Allocator) !ChatTemplate {
        return fromSource(gpa, gemma4_canonical_src);
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
    /// same order the parts appear, used for the pad-row counts.
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

    // --- capability probes --------------------------------------------------
    // Both are MEASURED by rendering, never inferred from the architecture. A
    // template either has a `{% if tools %}` branch and a `role == "tool"` arm
    // or it does not, and the loaded template is the only thing that knows:
    // gemma3 raises on a tool turn, plain llama concatenates any role verbatim,
    // qwen3.5/Bonsai and gemma4 each render their own trained format. A name
    // list would be wrong for the first finetune that strips its tool support.

    const probe_fn = "tp_probe_fn_9a3";
    const probe_result = "tp_probe_result_9a3";

    /// Whether declaring `RenderOpts.tools` actually reaches the model.
    pub fn supportsTools(self: *const ChatTemplate, gpa: std.mem.Allocator) bool {
        const msgs = [_]Message{.{ .role = .user, .content = "hi" }};
        return self.probeContains(gpa, &.{
            .messages = &msgs,
            .tools = &.{.{ .name = probe_fn, .description = "probe" }},
        }, probe_fn);
    }

    /// Whether a `.tool` message can be replayed, the template renders one
    /// without raising AND its content survives into the prompt.
    pub fn supportsToolRole(self: *const ChatTemplate, gpa: std.mem.Allocator) bool {
        const calls = [_]ToolCall{.{ .id = "probe1", .name = probe_fn }};
        const msgs = [_]Message{
            .{ .role = .user, .content = "hi" },
            .{ .role = .assistant, .tool_calls = &calls },
            .{ .role = .tool, .content = probe_result, .tool_call_id = "probe1" },
        };
        return self.probeContains(gpa, &.{
            .messages = &msgs,
            .tools = &.{.{ .name = probe_fn, .description = "probe" }},
        }, probe_result);
    }

    fn probeContains(self: *const ChatTemplate, gpa: std.mem.Allocator, opts: *const RenderOpts, needle: []const u8) bool {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        self.renderString(gpa, opts.*, &out) catch return false;
        return std.mem.indexOf(u8, out.items, needle) != null;
    }
};

// Process-global active template + its BOS string and the session system
// prompt, set once at load by the CLI (mirrors `chat.family`/`chat.bos_token`).
// Null `active` => the model shipped no template; callers fall back to the
// hand glue in `chat.zig`.
pub var active: ?ChatTemplate = null;
pub var bos: []const u8 = "";
pub var eos: []const u8 = "";
pub var system_prompt: ?[]const u8 = null;
/// Functions declared to the model for this session (`--tools`), or null.
pub var tools: ?[]const Tool = null;

/// Parse an OpenAI-style tools declaration (JSON text) into `Tool`s, allocated
/// in `a`, what a `--tools file.json` flag or an API request body carries.
///
/// Both spellings in the wild are accepted, since the file is written by hand as
/// often as it is copied from an API payload:
///   - wrapped:  `[{"type":"function","function":{"name":...,"parameters":{...}}}]`
///   - bare:     `[{"name":...,"description":...,"parameters":{...}}]`
/// A single object rather than an array is taken as a one-element list.
pub fn parseTools(a: std.mem.Allocator, json_text: []const u8) ![]Tool {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, json_text, .{}) catch
        return error.InvalidToolJson;
    const items: []const std.json.Value = switch (root) {
        .array => |arr| arr.items,
        .object => (try a.dupe(std.json.Value, &.{root})),
        else => return error.InvalidToolJson,
    };
    var out = try a.alloc(Tool, items.len);
    for (items, 0..) |it, k| {
        if (it != .object) return error.InvalidToolJson;
        // The wrapped form nests everything under "function"; the bare form is
        // the same object one level up.
        const f = if (it.object.get("function")) |fv| blk: {
            if (fv != .object) return error.InvalidToolJson;
            break :blk fv;
        } else it;
        const name = f.object.get("name") orelse return error.InvalidToolJson;
        if (name != .string) return error.InvalidToolJson;
        const desc = if (f.object.get("description")) |d| (if (d == .string) d.string else "") else "";
        const params = if (f.object.get("parameters")) |p|
            try std.json.Stringify.valueAlloc(a, p, .{})
        else
            "{}";
        out[k] = .{ .name = name.string, .description = desc, .parameters_json = params };
    }
    return out;
}

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
        if (m.tool_calls) |calls| if (calls.len > 0) {
            const list = try a.create(jinja.List);
            list.* = .{};
            for (calls) |c| try list.items.append(a, try toolCallValue(a, c));
            try d.put(a, "tool_calls", .{ .list = list });
        };
        // Absent (not empty-string) when unset: templates test `is defined`.
        if (m.tool_call_id.len > 0) try d.put(a, "tool_call_id", .{ .str = m.tool_call_id });
        try msgs.items.append(a, .{ .dict = d });
    }
    const g = try a.create(jinja.Dict);
    g.* = .{};
    try g.put(a, "messages", .{ .list = msgs });
    try g.put(a, "add_generation_prompt", .{ .boolean = opts.add_generation_prompt });
    try g.put(a, "enable_thinking", .{ .boolean = opts.enable_thinking });
    try g.put(a, "bos_token", .{ .str = opts.bos_token });
    try g.put(a, "eos_token", .{ .str = opts.eos_token });
    if (opts.tools) |decls| {
        const list = try a.create(jinja.List);
        list.* = .{};
        for (decls) |t| try list.items.append(a, try toolValue(a, t));
        try g.put(a, "tools", .{ .list = list });
    }
    return .{ .dict = g };
}

/// Parse a caller-supplied JSON blob (a schema or an argument object) into a
/// jinja value. Reported as `error.InvalidToolJson` rather than a std.json
/// error, so a caller can tell "your tool declaration is malformed" from "the
/// template failed to render".
fn jsonValue(a: std.mem.Allocator, text: []const u8) !jinja.Value {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch
        return error.InvalidToolJson;
    return jinja.fromJson(a, parsed) catch |e| mapErr(e);
}

/// `{"type":"function","function":{"name":...,"description":...,"parameters":...}}`.
/// The key ORDER is load-bearing: qwen3.5/Bonsai emit the declaration with
/// `{{ tool | tojson }}`, so it reaches the model exactly as inserted here.
fn toolValue(a: std.mem.Allocator, t: Tool) !jinja.Value {
    const params = try jsonValue(a, t.parameters_json);
    const f = try a.create(jinja.Dict);
    f.* = .{};
    try f.put(a, "name", .{ .str = t.name });
    try f.put(a, "description", .{ .str = t.description });
    try f.put(a, "parameters", params);
    const d = try a.create(jinja.Dict);
    d.* = .{};
    try d.put(a, "type", .{ .str = "function" });
    try d.put(a, "function", .{ .dict = f });
    return .{ .dict = d };
}

fn toolCallValue(a: std.mem.Allocator, c: ToolCall) !jinja.Value {
    const args = try jsonValue(a, c.arguments_json);
    const f = try a.create(jinja.Dict);
    f.* = .{};
    try f.put(a, "name", .{ .str = c.name });
    try f.put(a, "arguments", args);
    const d = try a.create(jinja.Dict);
    d.* = .{};
    if (c.id.len > 0) try d.put(a, "id", .{ .str = c.id });
    try d.put(a, "type", .{ .str = "function" });
    try d.put(a, "function", .{ .dict = f });
    return .{ .dict = d };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The Mistral v3 `[INST]` chat template (as shipped by Mistral-Small / its
// finetunes, e.g. Pantheon-RP-Pure). Exercises the engine features these
// templates need that simpler ChatML ones don't: a literal `}}` emitted from a
// string (`{{- "}}" }}`, which requires the scanner to skip quoted strings when
// finding the tag close), `messages[1:]` slicing, `selectattr(...,"equalto"...)`,
// the `{{ eos_token }}` global, and the system prompt folded into the LAST user
// turn. The expected string is byte-identical to jinja2's render of the real
// GGUF template (verified against jinja2 3.1.6).
const mistral_v3_src =
    \\{%- if messages[0]["role"] == "system" %}
    \\    {%- set system_message = messages[0]["content"] %}
    \\    {%- set loop_messages = messages[1:] %}
    \\{%- else %}
    \\    {%- set loop_messages = messages %}
    \\{%- endif %}
    \\{%- if not tools is defined %}
    \\    {%- set tools = none %}
    \\{%- endif %}
    \\{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}
    \\{{- bos_token }}
    \\{%- for message in loop_messages %}
    \\    {%- if message["role"] == "user" %}
    \\        {%- if loop.last and system_message is defined %}
    \\            {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
    \\        {%- else %}
    \\            {{- "[INST] " + message["content"] + "[/INST]" }}
    \\        {%- endif %}
    \\    {%- elif message["role"] == "assistant" %}
    \\        {{- " " + message["content"]|trim + eos_token}}
    \\    {%- else %}
    \\        {{- "}}" }}
    \\    {%- endif %}
    \\{%- endfor %}
;

// tp_core's jinja golden corpus: every `expected` in it was rendered by REAL
// jinja2 (`tools/gen_jinja_fixtures.py`). Reusing the Bonsai tool cases here
// tests the one thing those goldens cannot, that the TYPED API above builds
// the same context jinja2 was handed. Embedded (not read from disk) so the test
// does not depend on the cwd; test-only, so it costs shipped binaries nothing.
const jinja_fixtures_json = @embedFile("../core/assets/jinja/fixtures.json");

/// The named template's source out of the golden corpus, in `a`.
fn fixtureTemplate(a: std.mem.Allocator, name: []const u8) ![]const u8 {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, jinja_fixtures_json, .{});
    return root.object.get("templates").?.object.get(name).?.string;
}

/// The named case's jinja2-rendered `expected` output, in `a`.
fn fixtureExpected(a: std.mem.Allocator, key: []const u8) ![]const u8 {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, jinja_fixtures_json, .{});
    for (root.object.get("cases").?.array.items) |c| {
        if (std.mem.eql(u8, c.object.get("key").?.string, key))
            return c.object.get("expected").?.string;
    }
    return error.TestUnexpectedResult;
}

// The end-to-end proof for pieces (1) and (2) of the tool work: `Tool`,
// `ToolCall` and the `.tool` role, rendered through Bonsai's REAL embedded
// template, must reproduce jinja2's output byte for byte. The case is chosen
// because its argument values (bool, null, dict, list, int, string) are exactly
// the ones where the template's own `args_value` spellings disagree, so a
// wrongly-typed argument (e.g. arguments handed over as a JSON *string*, the
// OpenAI wire shape) cannot pass.
test "chat_template: tools + tool_calls + the tool role render byte-exact vs jinja2 (Bonsai)" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "bonsai"));
    defer ct.deinit();

    const calls = [_]ToolCall{.{
        .name = "get_weather",
        .arguments_json =
        \\{"city": "Paris", "opts": {"units": "c"}, "days": [1, 2],
        \\ "count": 3, "verbose": true, "note": null}
        ,
    }};
    const msgs = [_]Message{
        .{ .role = .user, .content = "Hello" },
        .{ .role = .assistant, .tool_calls = &calls },
        .{ .role = .tool, .content = "18C" },
    };
    const decls = [_]Tool{.{
        .name = "get_weather",
        .description = "Look up the weather.",
        .parameters_json =
        \\{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        ,
    }};

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .tools = &decls }, &out);
    try std.testing.expectEqualStrings(try fixtureExpected(a, "bonsai__tool_call_args"), out.items);
}

// The two halves must be INVERSES: what the model emits, parsed by
// `tool_call.zig` and replayed as a `ToolCall`, has to re-render as the same
// block, or the model sees a garbled version of its own request one turn later.
// Rendering with a REAL template rather than a hand-written block is the point:
// it is the template that decides how an argument is stringified.
test "chat_template: a rendered tool call parses back to the arguments it was built from" {
    const tool_call = @import("tool_call.zig");
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "bonsai"));
    defer ct.deinit();

    const args =
        \\{"city": "Paris", "opts": {"units": "c"}, "days": [1, 2], "count": 3, "verbose": true, "note": null}
    ;
    const calls = [_]ToolCall{.{ .name = "get_weather", .arguments_json = args }};
    const msgs = [_]Message{
        .{ .role = .user, .content = "Weather?" },
        .{ .role = .assistant, .tool_calls = &calls },
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .add_generation_prompt = false }, &out);

    const blk = switch (tool_call.nextBlock(out.items)) {
        .block => |b| b,
        else => return error.TestUnexpectedResult,
    };
    var got = try tool_call.parse(gpa, blk.body);
    defer got.deinit(gpa);
    try std.testing.expectEqualStrings("get_weather", got.name);
    try std.testing.expectEqualStrings(args, got.arguments_json);
}

// Declaring no tools must leave `tools` UNDEFINED, so a caller that never heard
// of tool calling renders exactly what it always did. (The templates all test
// `{%- if tools %}`, and an empty list would take the same branch, but a `[]`
// that some future template `| length`s or joins would not.)
test "chat_template: no tools declared renders identically to before tools existed" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "bonsai"));
    defer ct.deinit();
    const msgs = [_]Message{.{ .role = .user, .content = "Hello" }};
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs }, &out);
    try std.testing.expectEqualStrings(try fixtureExpected(a, "bonsai__user_only__default"), out.items);
}

// The load-bearing half of "gemma3 has no tool role": it raises on an
// INSERTED TURN OF ANY ROLE, not just on `tool`. So the obvious fallback for a
// tool result on such a model, slip it in as an extra user message, kills
// every render from that point on. Any caller adding a turn the user did not
// type needs this: the failure is a dead render, not a degraded one.
test "chat_template: gemma3 raises on any inserted turn, whatever its role" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(arena.allocator(), "gemma3"));
    defer ct.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);

    // The alternating baseline renders.
    const ok = [_]Message{
        .{ .role = .user, .content = "draw a fox" },
        .{ .role = .assistant, .content = "here" },
        .{ .role = .user, .content = "thanks" },
    };
    try ct.renderString(gpa, .{ .messages = &ok }, &out);
    // One extra turn between them does not, as a tool result OR as a user one.
    for ([_]Role{ .tool, .user }) |r| {
        const bad = [_]Message{
            .{ .role = .user, .content = "draw a fox" },
            .{ .role = .assistant, .content = "here" },
            .{ .role = r, .content = "image 1: FAILED" },
            .{ .role = .user, .content = "thanks" },
        };
        out.clearRetainingCapacity();
        try std.testing.expectError(error.ChatTemplateRender, ct.renderString(gpa, .{ .messages = &bad }, &out));
    }
}

// Both declaration spellings must reach the model identically, a hand-written
// tools file uses the bare form, an API payload the wrapped one, and a user who
// copies the wrong one would otherwise get a silently tool-less prompt.
test "chat_template: parseTools accepts the bare and the wrapped declaration" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const bare =
        \\[{"name": "get_weather", "description": "Look up the weather.",
        \\  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    ;
    const wrapped =
        \\[{"type": "function", "function": {"name": "get_weather", "description": "Look up the weather.",
        \\  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ;
    var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "bonsai"));
    defer ct.deinit();
    const msgs = [_]Message{.{ .role = .user, .content = "Hello" }};

    var a_out: std.ArrayList(u8) = .empty;
    defer a_out.deinit(gpa);
    var b_out: std.ArrayList(u8) = .empty;
    defer b_out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .tools = try parseTools(a, bare) }, &a_out);
    try ct.renderString(gpa, .{ .messages = &msgs, .tools = try parseTools(a, wrapped) }, &b_out);
    try std.testing.expectEqualStrings(a_out.items, b_out.items);
    // And both are the jinja2 golden for that declaration, so neither spelling
    // is merely "consistently wrong".
    try std.testing.expectEqualStrings(try fixtureExpected(a, "bonsai__tools_decl"), a_out.items);

    try std.testing.expectError(error.InvalidToolJson, parseTools(a, "[{\"description\": \"no name\"}]"));
    try std.testing.expectError(error.InvalidToolJson, parseTools(a, "not json"));
}

// The capability is a property of the TEMPLATE, not of the architecture, and
// gemma3 is the case that makes it matter: it raises `Conversation roles must
// alternate user/assistant` on a tool turn, so a caller that assumes every
// model can take one kills the render. Pinned in both directions so neither
// probe can silently start answering "yes" to everything.
test "chat_template: tool support is measured per template (gemma3 refuses)" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    for ([_][]const u8{ "bonsai", "qwen35", "gemma4" }) |name| {
        var ct = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, name));
        defer ct.deinit();
        errdefer std.debug.print("template {s}\n", .{name});
        try std.testing.expect(ct.supportsTools(gpa));
        try std.testing.expect(ct.supportsToolRole(gpa));
    }
    // gemma3 has neither a tools branch nor a tool role, and RAISES on the
    // latter, which is why the probe renders instead of pattern-matching a name.
    var g3 = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "gemma3"));
    defer g3.deinit();
    try std.testing.expect(!g3.supportsTools(gpa));
    try std.testing.expect(!g3.supportsToolRole(gpa));
    // Plain llama has no tools branch either; its bare role loop does echo a
    // tool turn, which is exactly why the two probes are separate questions.
    var ll = try ChatTemplate.fromSource(gpa, try fixtureTemplate(a, "llama"));
    defer ll.deinit();
    try std.testing.expect(!ll.supportsTools(gpa));
}

test "chat_template: Mistral v3 [INST] template renders byte-exact vs jinja2" {
    const gpa = std.testing.allocator;
    var ct = try ChatTemplate.fromSource(gpa, mistral_v3_src);
    defer ct.deinit();

    const msgs = [_]Message{
        .{ .role = .system, .content = "You are Lyra." },
        .{ .role = .user, .content = "Hello there." },
        .{ .role = .assistant, .content = "Hi!" },
        .{ .role = .user, .content = "How are you?" },
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .bos_token = "<s>", .eos_token = "</s>" }, &out);

    errdefer std.debug.print("rendered: {s}\n", .{out.items});
    // System is folded into the LAST user turn; assistant turn closed by eos.
    try std.testing.expectEqualStrings(
        "<s>[INST] Hello there.[/INST] Hi!</s>[INST] You are Lyra.\n\nHow are you?[/INST]",
        out.items,
    );
}

test "jinja: scanner finds tag close outside quoted strings (literal }} in a string)" {
    const gpa = std.testing.allocator;
    // The `}}` inside the string must NOT be taken as the `{{ }}` terminator.
    var ct = try ChatTemplate.fromSource(gpa, "{{- \"a}}b%}c\" }}X");
    defer ct.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &.{} }, &out);
    try std.testing.expectEqualStrings("a}}b%}cX", out.items);
}

// Real GGUF: parse + render the Mistral-Small finetune's actual embedded
// template. Proves the shipping template loads (no JinjaParse -> no silent
// ChatML fallback) and produces the `[INST]` format. Self-skips when absent;
// mmaps header+tokenizer only (no weights), so it stays in the fast suite.
test "chat_template: real Mistral-Small GGUF renders [INST] (no fallback)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/Pantheon-RP-Pure-1.6.2-22b-Small.i1-Q5_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var ct = (try ChatTemplate.fromGguf(gpa, &g)) orelse return error.TestUnexpectedResult;
    defer ct.deinit();

    const msgs = [_]Message{
        .{ .role = .system, .content = "You are Lyra." },
        .{ .role = .user, .content = "Hi" },
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try ct.renderString(gpa, .{ .messages = &msgs, .bos_token = "<s>", .eos_token = "</s>" }, &out);
    errdefer std.debug.print("rendered: {s}\n", .{out.items});
    try std.testing.expectEqualStrings("<s>[INST] You are Lyra.\n\nHi[/INST]", out.items);
}

// A tiny ChatML template mirroring the llama/qwen shape, so the render->tokenize
// path is testable with the embedded default (Qwen ChatML) tokenizer, no GGUF.
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
    // the crux of TODO #1: past <|channel>...<channel|> blocks must NOT appear.
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
// on the shipping template, prior-turn thoughts vanish from the TOKEN stream,
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

// Gemma 4 gates thinking on a `<|think|>` cue that lives in the SYSTEM turn, so
// the reference template emits a bare system turn holding only the cue when the
// caller supplied no system message. Miss that and a "thinking" session
// silently answers directly, with nothing in the output to say why. Thinking
// OFF is the inverse: no cue, and an empty already-closed thought channel
// primed at the tail so the model skips straight to the answer.
test "chat_template: gemma4 thinking cue survives an absent system message" {
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

    const msgs = [_]Message{.{ .role = .user, .content = "Hi" }};

    for ([_]bool{ true, false }) |think| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ct.renderIds(&tok, gpa, .{
            .messages = &msgs,
            .bos_token = "<bos>",
            .enable_thinking = think,
        }, &ids);
        const text = try tok.decodeAlloc(gpa, ids.items);
        defer gpa.free(text);
        errdefer std.debug.print("enable_thinking={} rendered:\n{s}\n", .{ think, text });

        const has_cue = std.mem.indexOf(u8, text, "<|think|>") != null;
        const primed_closed = std.mem.endsWith(u8, text, "<|channel>thought\n<channel|>");
        try std.testing.expectEqual(think, has_cue);
        try std.testing.expectEqual(!think, primed_closed);
    }
}

// The GUI's "Gemma 4 canonical chat template" option swaps the GGUF's own
// template for Google's upstream copy, which opens with
// `{% set enable_thinking = enable_thinking | default(false) %}`. That
// self-referential set must resolve its right-hand side against the caller's
// global; read it as the not-yet-bound local and thinking silently INVERTS,
// because the tail then primes a closed empty thought channel and the model
// never reasons. The GGUF's own template has no such line, so this path is the
// only one that can regress here.
test "chat_template: gemma4 canonical template honors enable_thinking" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var tok = try Tokenizer.initFromGguf(gpa, &g);
    defer tok.deinit();
    var ct = try ChatTemplate.gemma4Canonical(gpa);
    defer ct.deinit();

    const msgs = [_]Message{
        .{ .role = .system, .content = "You are terse." },
        .{ .role = .user, .content = "Hi" },
    };

    for ([_]bool{ true, false }) |think| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ct.renderIds(&tok, gpa, .{
            .messages = &msgs,
            .bos_token = "<bos>",
            .enable_thinking = think,
        }, &ids);
        const text = try tok.decodeAlloc(gpa, ids.items);
        defer gpa.free(text);
        errdefer std.debug.print("enable_thinking={} rendered:\n{s}\n", .{ think, text });

        const has_cue = std.mem.indexOf(u8, text, "<|think|>") != null;
        const primed_closed = std.mem.endsWith(u8, text, "<|channel>thought\n<channel|>");
        try std.testing.expectEqual(think, has_cue);
        try std.testing.expectEqual(!think, primed_closed);
    }
}

// Regression: Gemma prompts REQUIRE a leading <bos>. The gemma4 tokenizer must
// populate `tok.bos` (from tokenizer.ggml.bos_token_id) so the render-driven
// path, which derives the template's `{{ bos_token }}` string from it, emits
// the BOS. When it was left null, the render dropped BOS and the model
// degenerated into a repeat loop (worst on larger models). Self-skips if absent.
test "chat_template: gemma4 render emits a leading BOS (tok.bos populated)" {
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

    // The tokenizer must know its BOS id (this was the bug: null for gemma4).
    const bos_id = tok.bos orelse return error.TestUnexpectedResult;

    // Derive the BOS string exactly like the render-driven path does, then
    // confirm the rendered token stream actually starts with the BOS id.
    const bos_str = try tok.decodeAlloc(gpa, &.{bos_id});
    defer gpa.free(bos_str);
    const msgs = [_]Message{.{ .role = .user, .content = "Hi" }};
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ct.renderIds(&tok, gpa, .{ .messages = &msgs, .bos_token = bos_str, .enable_thinking = true }, &ids);
    try std.testing.expect(ids.items.len > 0);
    try std.testing.expectEqual(bos_id, ids.items[0]);
}

// Task #4: the multimodal render must expand the template's single image
// placeholder into the model's real image block, `<|image>` + pad×N +
// `<image|>` for gemma4, with the recorded row pointing at the first pad, so
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
    const bos_str = if (tok.bos) |b| (try tok.decodeAlloc(gpa, &.{b})) else "";
    defer if (bos_str.len > 0) gpa.free(bos_str);
    try ct.renderIdsMM(&tok, gpa, .{
        .messages = &msgs,
        .bos_token = bos_str,
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
    const bos_str = if (tok.bos) |b| (try tok.decodeAlloc(gpa, &.{b})) else "";
    defer if (bos_str.len > 0) gpa.free(bos_str);
    try ct.renderIdsMM(&tok, gpa, .{
        .messages = &msgs,
        .bos_token = bos_str,
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
