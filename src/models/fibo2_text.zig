//! Fibo 2.0 prompt conditioning (`comfy/text_encoders/fibo2.py`): the text that reaches
//! Qwen3-VL-4B, and the six context bundles its fifteen tapped states become.
//!
//! A positive prompt that parses as a JSON object is a structured caption and is
//! rewritten the way training saw captions: score keys dropped at the top level and one
//! level down, empty values pruned recursively, `structured_caption` unwrapped, an
//! `edit_instruction` removed, a `{short_description, text_render}` pair kept whole even
//! when empty, then re-serialized with Python's compact separators, unescaped non-ASCII
//! and float repr (`1e-05`, `1e+16`, `2.0`). Anything else passes through as written. A
//! blank prompt is the null caption. The negative is NOT cleaned: it is wrapped in the
//! template as written, or is the null caption when blank.
//!
//! Each `text_render` entry with text gets an `image_tokens` key: empty in the base
//! caption, the vision marker in the caption that shows Qwen3-VL a drawing of the text.
//! `Prepared.draws` lists those strings; with any, ComfyUI conditions on the base caption
//! above sigma 0.9 and on the drawn one below it.

const std = @import("std");
const tp_core = @import("tp_core");
const qwen3 = @import("qwen3.zig");

pub const template_prefix = "<|im_start|>user\n";
pub const template_suffix = "<|im_end|>\n<|im_start|>assistant\n";
pub const vision_marker = "<|vision_start|><|image_pad|><|vision_end|>";

/// The empty caption training used for classifier-free guidance.
pub const null_caption = template_prefix ++
    "{\"short_description\":\"\",\"objects\":[],\"background_setting\":\"\",\"lighting\":{},\"aesthetics\":{},\"photographic_characteristics\":{},\"style_medium\":\"\",\"context\":\"\",\"artistic_style\":\"\"}" ++
    template_suffix;

/// Below this sigma a generation's conditioning shows Qwen3-VL its caption's text drawn
/// as images (`TEXT_IMAGES_START`, 0.9 at the default shift).
pub const text_images_sigma: f32 = 0.9;

const scores = [_][]const u8{ "aesthetic_score", "preference_score" };

/// Which tapped layers make each bundle, in bundle order: the Perceiver's, then one per
/// injection block. Layers are `qwen3.fibo2_taps` entries.
pub const bundle_layers = [6][3]usize{ .{ 9, 20, 31 }, .{ 5, 16, 27 }, .{ 7, 18, 29 }, .{ 9, 20, 31 }, .{ 11, 22, 33 }, .{ 13, 24, 35 } };

/// `bundle_layers` as indices into the tap list.
pub const bundle_index: [18]usize = blk: {
    var out: [18]usize = undefined;
    for (bundle_layers, 0..) |b, i| for (b, 0..) |layer, j| {
        out[i * 3 + j] = std.mem.findScalar(usize, &qwen3.fibo2_taps, layer).?;
    };
    break :blk out;
};

pub const Prepared = struct {
    /// The templated text to tokenize.
    text: []u8,
    /// The `text_render` strings the caption asks to be drawn, in order.
    draws: [][]u8,

    pub fn deinit(self: *Prepared, gpa: std.mem.Allocator) void {
        gpa.free(self.text);
        for (self.draws) |d| gpa.free(d);
        gpa.free(self.draws);
        self.* = undefined;
    }
};

/// The positive prompt as ComfyUI's `Fibo2Tokenizer` prepares it for a generation. With
/// `text_images`, the `text_render` markers are the vision marker, else empty.
pub fn prepare(gpa: std.mem.Allocator, prompt: []const u8, text_images: bool) !Prepared {
    var draws: std.ArrayList([]u8) = .empty;
    errdefer {
        for (draws.items) |d| gpa.free(d);
        draws.deinit(gpa);
    }
    if (isBlank(prompt)) return .{ .text = try gpa.dupe(u8, null_caption), .draws = try draws.toOwnedSlice(gpa) };

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const body: []const u8 = blk: {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, prompt, .{ .duplicate_field_behavior = .use_last }) catch break :blk prompt;
        const caption = (try parseCaption(arena, parsed)) orelse break :blk prompt;
        if (caption != .null) if (textRender(caption)) |list| for (list.items) |*e| {
            if (e.* != .object) continue;
            const t = e.object.get("text") orelse continue;
            if (t != .string or t.string.len == 0) continue;
            try e.object.put(arena, "image_tokens", .{ .string = if (text_images) vision_marker else "" });
            try draws.append(gpa, try gpa.dupe(u8, t.string));
        };
        var out: std.Io.Writer.Allocating = .init(arena);
        try dumps(&out.writer, if (caption == .null) .{ .object = .empty } else caption);
        break :blk out.written();
    };
    const text = if (std.mem.startsWith(u8, body, "<|im_start|>"))
        try gpa.dupe(u8, body)
    else
        try std.mem.concat(gpa, u8, &.{ template_prefix, body, template_suffix });
    return .{ .text = text, .draws = try draws.toOwnedSlice(gpa) };
}

/// The negative prompt: the template around it as written, or the null caption.
pub fn negative(gpa: std.mem.Allocator, prompt: []const u8) ![]u8 {
    if (isBlank(prompt)) return gpa.dupe(u8, null_caption);
    return std.mem.concat(gpa, u8, &.{ template_prefix, prompt, template_suffix });
}

/// `parse_prompt`'s caption: null when the prompt passes through as written, `.null`
/// for an empty caption, else the cleaned object.
fn parseCaption(arena: std.mem.Allocator, parsed: std.json.Value) !?std.json.Value {
    if (parsed != .object) return null;
    var cap = parsed.object;
    _ = cap.orderedRemove("edit_instruction");
    if (cap.get("structured_caption")) |inner| {
        _ = cap.orderedRemove("structured_caption");
        if (inner != .object) return if (isEmpty(inner)) .null else null;
        cap = inner.object;
    }
    const pair = cap.count() == 2 and cap.contains("short_description") and cap.contains("text_render");

    var kept: std.json.ObjectMap = .empty;
    var it = cap.iterator();
    while (it.next()) |e| {
        if (isScore(e.key_ptr.*)) continue;
        var v = e.value_ptr.*;
        if (v == .object) {
            var inner: std.json.ObjectMap = .empty;
            var jt = v.object.iterator();
            while (jt.next()) |f| if (!isScore(f.key_ptr.*)) try inner.put(arena, f.key_ptr.*, f.value_ptr.*);
            v = .{ .object = inner };
        }
        try kept.put(arena, e.key_ptr.*, v);
    }
    var out = try clean(arena, .{ .object = kept });
    if (pair) {
        var p: std.json.ObjectMap = .empty;
        const sd = out.object.get("short_description");
        const tr = out.object.get("text_render");
        try p.put(arena, "short_description", if (sd != null and truthy(sd.?)) sd.? else .{ .string = "" });
        try p.put(arena, "text_render", if (tr != null and truthy(tr.?)) tr.? else .{ .array = .init(arena) });
        out = .{ .object = p };
    }
    return if (isEmpty(out)) .null else out;
}

fn textRender(caption: std.json.Value) ?*std.json.Array {
    if (caption != .object) return null;
    const p = caption.object.getPtr("text_render") orelse return null;
    return if (p.* == .array) &p.array else null;
}

fn isScore(key: []const u8) bool {
    for (scores) |s| if (std.mem.eql(u8, key, s)) return true;
    return false;
}

/// Python's `str.strip() == ""` for the whitespace a prompt carries.
fn isBlank(s: []const u8) bool {
    return std.mem.trim(u8, s, " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f").len == 0;
}

/// `fibo2.empty`: None, or an empty string, dict or list. Numbers and booleans never are.
fn isEmpty(v: std.json.Value) bool {
    return switch (v) {
        .null => true,
        .string => |s| s.len == 0,
        .object => |o| o.count() == 0,
        .array => |a| a.items.len == 0,
        else => false,
    };
}

/// Python truthiness, for the `x or default` of the pair rule.
fn truthy(v: std.json.Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        else => !isEmpty(v),
    };
}

/// `fibo2.clean`: prune empties recursively, keeping order.
fn clean(arena: std.mem.Allocator, v: std.json.Value) !std.json.Value {
    switch (v) {
        .object => |o| {
            var out: std.json.ObjectMap = .empty;
            var it = o.iterator();
            while (it.next()) |e| {
                const c = try clean(arena, e.value_ptr.*);
                if (!isEmpty(c)) try out.put(arena, e.key_ptr.*, c);
            }
            return .{ .object = out };
        },
        .array => |a| {
            var out: std.json.Array = .init(arena);
            for (a.items) |item| {
                const c = try clean(arena, item);
                if (!isEmpty(c)) try out.append(c);
            }
            return .{ .array = out };
        },
        else => return v,
    }
}

/// `json.dumps(v, separators=(",", ":"), ensure_ascii=False)`.
pub fn dumps(w: *std.Io.Writer, v: std.json.Value) !void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .number_string => |s| try w.writeAll(s),
        .float => |f| try pyFloat(w, f),
        .string => |s| try pyString(w, s),
        .array => |a| {
            try w.writeByte('[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try w.writeByte(',');
                try dumps(w, item);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            try w.writeByte('{');
            var it = o.iterator();
            var first = true;
            while (it.next()) |e| {
                if (!first) try w.writeByte(',');
                first = false;
                try pyString(w, e.key_ptr.*);
                try w.writeByte(':');
                try dumps(w, e.value_ptr.*);
            }
            try w.writeByte('}');
        },
    }
}

/// A JSON string as Python escapes it with `ensure_ascii=False`.
fn pyString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0c => try w.writeAll("\\f"),
        0...0x07, 0x0b, 0x0e...0x1f => try w.print("\\u{x:0>4}", .{c}),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, positional when the decimal
/// exponent lies in [-4, 16), else `d.ddde+XX`.
fn pyFloat(w: *std.Io.Writer, f: f64) !void {
    var buf: [64]u8 = undefined;
    const sci = try std.fmt.bufPrint(&buf, "{e}", .{f});
    const neg = sci[0] == '-';
    const body = if (neg) sci[1..] else sci;
    const e_at = std.mem.findScalar(u8, body, 'e').?;
    const exp = try std.fmt.parseInt(i32, body[e_at + 1 ..], 10);
    var digits_buf: [32]u8 = undefined;
    var nd: usize = 0;
    for (body[0..e_at]) |c| if (c != '.') {
        digits_buf[nd] = c;
        nd += 1;
    };
    const digits = digits_buf[0..nd];
    if (neg) try w.writeByte('-');
    if (exp < -4 or exp >= 16) {
        try w.writeByte(digits[0]);
        if (digits.len > 1) {
            try w.writeByte('.');
            try w.writeAll(digits[1..]);
        }
        try w.print("e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
        return;
    }
    if (exp < 0) {
        try w.writeAll("0.");
        for (0..@intCast(-exp - 1)) |_| try w.writeByte('0');
        try w.writeAll(digits);
        return;
    }
    const int_len: usize = @intCast(exp + 1);
    if (digits.len <= int_len) {
        try w.writeAll(digits);
        for (0..int_len - digits.len) |_| try w.writeByte('0');
        try w.writeAll(".0");
    } else {
        try w.writeAll(digits[0..int_len]);
        try w.writeByte('.');
        try w.writeAll(digits[int_len..]);
    }
}

/// The DiT's context from the encoder's tap stack: `taps` is token-major
/// `[text_len][fibo2_taps.len][hidden]`; each state is unit-RMS-normed (eps 1e-6) and
/// rounded to bf16 as the reference's encoder output is, then laid out as
/// `[text_len][bundle_index.len * hidden]`. Caller frees.
pub fn bundles(gpa: std.mem.Allocator, taps: []const f32, text_len: usize, hidden: usize) ![]f32 {
    const n_taps = qwen3.fibo2_taps.len;
    std.debug.assert(taps.len == text_len * n_taps * hidden);
    const normed = try gpa.alloc(f32, taps.len);
    defer gpa.free(normed);
    @import("tp_ops").norm.rmsNormUnit(normed, taps, hidden, 1e-6);
    for (normed) |*v| v.* = tp_core.dtype.bf16ToF32(tp_core.dtype.f32ToBf16(v.*));

    const out = try gpa.alloc(f32, text_len * bundle_index.len * hidden);
    for (0..text_len) |t| for (bundle_index, 0..) |ti, j| {
        @memcpy(out[(t * bundle_index.len + j) * hidden ..][0..hidden], normed[(t * n_taps + ti) * hidden ..][0..hidden]);
    };
    return out;
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const fixture = @embedFile("assets/fibo2_text.json");

test "Fibo 2.0 prompt preparation matches ComfyUI's tokenizer on the corpus" {
    const gpa = testing.allocator;
    const Case = struct { prompt: []const u8, base: []const u8, text_images: []const u8, draws: []const []const u8 };
    const Doc = struct { null_caption: []const u8, template: []const u8, cases: []const Case };
    const doc = try std.json.parseFromSlice(Doc, gpa, fixture, .{});
    defer doc.deinit();
    try testing.expectEqualStrings(doc.value.null_caption, null_caption);

    for (doc.value.cases, 0..) |c, i| {
        errdefer std.debug.print("case {d}: {s}\n", .{ i, c.prompt });
        var base = try prepare(gpa, c.prompt, false);
        defer base.deinit(gpa);
        try testing.expectEqualStrings(c.base, base.text);
        var full = try prepare(gpa, c.prompt, true);
        defer full.deinit(gpa);
        try testing.expectEqualStrings(c.text_images, full.text);
        try testing.expectEqual(c.draws.len, full.draws.len);
        for (c.draws, full.draws) |want, got| try testing.expectEqualStrings(want, got);
    }
}

test "the Fibo 2.0 negative is wrapped as written, never cleaned" {
    const gpa = testing.allocator;
    const n = try negative(gpa, "{\"aesthetic_score\": 1}");
    defer gpa.free(n);
    try testing.expectEqualStrings(template_prefix ++ "{\"aesthetic_score\": 1}" ++ template_suffix, n);
    const z = try negative(gpa, "  ");
    defer gpa.free(z);
    try testing.expectEqualStrings(null_caption, z);
}

test "bundle_index matches ComfyUI's BUNDLE_INDEX" {
    try testing.expectEqualSlices(usize, &.{ 2, 7, 12, 0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14 }, &bundle_index);
}

test "bundles norm each state and place it by bundle" {
    const gpa = testing.allocator;
    const n = qwen3.fibo2_taps.len;
    const hidden = 4;
    var taps: [2 * n * hidden]f32 = undefined;
    for (0..2) |t| for (0..n) |k| for (0..hidden) |h| {
        taps[(t * n + k) * hidden + h] = @floatFromInt((t + 1) * 100 + k * 3 + h + 1);
    };
    const out = try bundles(gpa, &taps, 2, hidden);
    defer gpa.free(out);
    // Token 1, bundle 1's first slot is layer 5, tap 0.
    const slot = out[(1 * 18 + 3) * hidden ..][0..hidden];
    var want: [hidden]f32 = undefined;
    @import("tp_ops").norm.rmsNormUnit(&want, taps[(1 * n + 0) * hidden ..][0..hidden], hidden, 1e-6);
    for (want, slot) |a, b| try testing.expectApproxEqAbs(a, b, 1e-2);
}

const test_gate = @import("../test_gate.zig");
const build_options = @import("build_options");
const te_ref_path = "src/models/assets/fibo2_te_ref.safetensors";
const te_ckpt = build_options.model_root ++ "/text_encoders/qwen3vl_4b_bf16.safetensors";

test "the Fibo 2.0 conditioning matches ComfyUI's on the real Qwen3-VL-4B" {
    const gpa = testing.allocator;
    const io = testing.io;
    const safetensors = tp_core.safetensors;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    var ref = try safetensors.SafeTensors.open(gpa, io, te_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer ck.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &ck }, .fibo2);
    defer enc.deinit();
    var tok = try tp_core.tokenizer.Tokenizer.init(gpa);
    defer tok.deinit();

    const ops = @import("tp_ops");
    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;

    const prompts = [_][]const u8{
        "a red fox in the snow",
        "{\"short_description\": \"a lighthouse at dusk\", \"aesthetic_score\": 8, \"lighting\": {\"conditions\": \"golden hour\"}, \"objects\": []}",
    };
    const hidden = enc.cfg.hidden;
    for (prompts, 0..) |p, i| {
        var kb: [16]u8 = undefined;
        const tv = try ref.require(try std.mem.print(&kb, "tokens.{d}", .{i}));
        const want_ids = try gpa.alloc(u32, tv.bytes.len / 4);
        defer gpa.free(want_ids);
        for (want_ids, 0..) |*id, j| id.* = @intCast(std.mem.readInt(i32, tv.bytes[j * 4 ..][0..4], .little));

        var prep = try prepare(gpa, p, false);
        defer prep.deinit(gpa);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try tok.encode(gpa, prep.text, &ids);
        try testing.expectEqualSlices(u32, want_ids, ids.items);

        const taps = try enc.encode(io, gpa, ids.items, null);
        defer gpa.free(taps);
        const cond = try bundles(gpa, taps, ids.items.len, hidden);
        defer gpa.free(cond);

        const cv = try ref.require(try std.mem.print(&kb, "cond.{d}", .{i}));
        const want = try cv.toF32Alloc(gpa);
        defer gpa.free(want);
        const cols = cv.info.shape.slice()[2];
        // Per slot, so a misplaced tap is named rather than averaged away.
        for (0..bundle_index.len) |slot| {
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..ids.items.len) |t| for (0..cols) |c| {
                const w = want[(t * bundle_index.len + slot) * cols + c];
                const g = cond[(t * bundle_index.len + slot) * hidden + c];
                num += @as(f64, w - g) * (w - g);
                den += @as(f64, w) * w;
            };
            const rel = @sqrt(num / den);
            errdefer std.debug.print("prompt {d} slot {d}: rel L2 {e:.3}\n", .{ i, slot, rel });
            // Both sides round to bf16 (2^-9 relative); a near-tie that rounds the other
            // way is one bf16 step, so the floor is a few 1e-3.
            try testing.expect(rel < 1e-2);
        }
    }
}
