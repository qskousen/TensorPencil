//! Writing a finished render to the user's output folder, with the AUTOMATIC1111
//! `parameters` block a reader re-renders from. The client does this, not the
//! engine: the engine writes no files, and everything the block needs (the
//! prompt the user typed, the request, the model stem and family the host
//! reported) is already on this side of the wire.
const std = @import("std");
const Io = std.Io;
const tp = @import("TensorPencil");
const wire = @import("serve").wire;

pub const Error = error{ OutOfMemory, WriteFailed, EncodeFailed };

/// The `parameters` text for an image, exactly as the engine used to write it.
pub fn paramsAlloc(gpa: std.mem.Allocator, info: *const wire.ImageInfo, w: usize, h: usize) ![]u8 {
    var opts: tp.pipeline.Options = .{ .prompt = "" };
    // Lives as long as `opts` is read below: `applyTo` points into both.
    var steer_scratch: [wire.max_steers]tp.pipeline.CondNoise.Steer = @splat(.{});
    info.params.applyTo(&opts, &steer_scratch);
    const base = try tp.pipeline.buildA1111Params(
        gpa,
        info.prompt,
        info.negative,
        info.req_steps,
        info.req_cfg,
        info.req_seed,
        w,
        h,
        info.model_stem,
        std.meta.stringToEnum(tp.pipeline.Family, info.family),
        opts.sampler,
        opts.scheduler,
        opts.prompt_syntax,
        opts.emphasis,
        opts.compat,
        opts.compatConfig(),
    );
    // Everything the host reported about what it actually loaded. The engine
    // knows it, the client writes the file, so it rides on `ImageInfo`.
    var loras: std.ArrayList(tp.pipeline.LoraRecord) = .empty;
    defer loras.deinit(gpa);
    for (info.loras) |l| try loras.append(gpa, .{ .name = l.name, .hash = l.hash, .strength = l.strength });
    return tp.pipeline.appendExtraParams(gpa, base, .{
        .clip1 = info.clip1_stem,
        .clip2 = info.clip2_stem,
        .vae = info.vae_stem,
        .model_hash = info.model_hash,
        .vae_hash = info.vae_hash,
        .weight_dtype = info.weight_dtype,
        .shift = if (info.shift > 0) info.shift else null,
        // `opts` borrows the curve from `info.params`, which outlives this call.
        .cond_noise = opts.cond_noise.curve,
        .cond_noise_amount = if (opts.cond_noise.on()) opts.cond_noise.amount else null,
        .cond_noise_seed = if (opts.cond_noise.on()) opts.cond_noise.seed orelse info.req_seed else null,
        .cond_noise_negative = if (opts.cond_noise.on() and opts.cond_noise.neg_scale != 0) opts.cond_noise.neg_scale else null,
        // Borrows `steer_scratch`, which outlives this call.
        .cond_steers = opts.cond_noise.steers,
        .loras = loras.items,
    });
}

/// PNG bytes for `rgba` with the metadata block. `rgba` is `w*h*4`.
pub fn encodeAlloc(gpa: std.mem.Allocator, info: *const wire.ImageInfo, rgba: []const u8, w: usize, h: usize) Error![]u8 {
    const params = paramsAlloc(gpa, info, w, h) catch return error.OutOfMemory;
    defer gpa.free(params);
    const rgb = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(rgb);
    for (0..w * h) |i| {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    var png: std.ArrayList(u8) = .empty;
    errdefer png.deinit(gpa);
    tp.image.encodePngRgbText(gpa, &png, rgb, w, h, &.{
        .{ .keyword = "parameters", .text = params },
    }) catch return error.EncodeFailed;
    return png.toOwnedSlice(gpa);
}

/// Unique, roughly time-sortable filename: tp_<ns>_<seed>.png.
pub fn fileName(buf: []u8, info: *const wire.ImageInfo) []const u8 {
    return std.fmt.bufPrint(buf, "tp_{d}_{d}.png", .{ info.start_ns, info.req_seed }) catch "tp_image.png";
}

/// Write the image into `dir` (created if missing) and return the path, gpa-owned.
pub fn save(gpa: std.mem.Allocator, io: Io, dir: []const u8, info: *const wire.ImageInfo, rgba: []const u8, w: usize, h: usize) Error![]u8 {
    return saveIn(gpa, io, Io.Dir.cwd(), dir, info, rgba, w, h);
}

/// `save` relative to `base`: the path returned is `dir/<name>` under it.
pub fn saveIn(gpa: std.mem.Allocator, io: Io, base: Io.Dir, dir: []const u8, info: *const wire.ImageInfo, rgba: []const u8, w: usize, h: usize) Error![]u8 {
    const png = try encodeAlloc(gpa, info, rgba, w, h);
    defer gpa.free(png);
    var name_buf: [64]u8 = undefined;
    const path = std.fs.path.join(gpa, &.{ dir, fileName(&name_buf, info) }) catch return error.OutOfMemory;
    errdefer gpa.free(path);
    base.createDirPath(io, dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return error.WriteFailed,
    };
    base.writeFile(io, .{ .sub_path = path, .data = png }) catch return error.WriteFailed;
    return path;
}

/// The fields of an AUTOMATIC1111 `parameters` block that describe a REQUEST,
/// as read back off a saved PNG. Slices borrow the input.
pub const A1111Params = struct {
    prompt: []const u8 = "",
    negative: []const u8 = "",
    steps: ?usize = null,
    cfg: ?f32 = null,
    seed: ?u64 = null,
    width: ?usize = null,
    height: ?usize = null,
    /// Flow shift. A request field, not a resource one: a reader that falls back
    /// to the family default re-renders a DIFFERENT image when this was set.
    shift: ?f32 = null,
    /// img2img strength.
    denoise: ?f32 = null,
    /// Conditioning-noise curve and its knobs, as written by
    /// `pipeline.appendExtraParams`. The curve is unquoted here.
    cond_noise: []const u8 = "",
    cond_noise_amount: ?f32 = null,
    cond_noise_seed: ?u64 = null,
    cond_noise_negative: ?f32 = null,
};

/// Split a settings line on commas that are OUTSIDE double quotes.
///
/// A1111 quotes any value that contains a comma -- `Lora hashes: "a: 1, b: 2"` has
/// always done this, and a conditioning-noise curve does it too (`clamp(a,0,1)`).
/// Splitting on every comma tears those into fragments that parse as nothing, which
/// is silent: the field simply comes back null.
const FieldIter = struct {
    s: []const u8,
    i: usize = 0,

    fn next(self: *FieldIter) ?[]const u8 {
        if (self.i >= self.s.len) return null;
        const start = self.i;
        var in_q = false;
        while (self.i < self.s.len) : (self.i += 1) {
            switch (self.s[self.i]) {
                '"' => in_q = !in_q,
                ',' => if (!in_q) {
                    const f = self.s[start..self.i];
                    self.i += 1;
                    return f;
                },
                else => {},
            }
        }
        return self.s[start..self.i];
    }
};

/// A value as written: quotes stripped when it carries them.
fn unquote(v: []const u8) []const u8 {
    if (v.len >= 2 and v[0] == '"' and v[v.len - 1] == '"') return v[1 .. v.len - 1];
    return v;
}

/// Parse what `buildA1111Params` wrote. The saved PNG is the record of how an
/// image was made, so reopening one reads it back rather than the transcript
/// carrying a second copy that can disagree with the file.
///
/// Deliberately lenient: this also has to read blocks written by ComfyUI and
/// A1111 themselves, where field order, spelling and which keys are present all
/// vary. Anything unrecognised is skipped and the field stays null.
pub fn parseA1111Params(text: []const u8) A1111Params {
    var out: A1111Params = .{};

    // The settings line is the LAST line that starts with "Steps:"; everything
    // before it is prompt (and negative). Searching from the end is what keeps a
    // prompt that itself contains "Steps:" from splitting the block early.
    const neg_tag = "\nNegative prompt:";
    var head_end = text.len;
    var settings: []const u8 = "";
    var it = std.mem.splitBackwardsScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "Steps:")) {
            settings = line;
            // The line's own offset, not a running total: a block whose first
            // line is the settings line leaves nothing before it, and counting
            // back from the end underflows there.
            head_end = @intFromPtr(line.ptr) - @intFromPtr(text.ptr);
            break;
        }
    }
    const head = std.mem.trimEnd(u8, text[0..head_end], "\n");

    if (std.mem.indexOf(u8, head, neg_tag)) |i| {
        out.prompt = std.mem.trim(u8, head[0..i], " \t\r\n");
        out.negative = std.mem.trim(u8, head[i + neg_tag.len ..], " \t\r\n");
    } else {
        out.prompt = std.mem.trim(u8, head, " \t\r\n");
    }

    var f: FieldIter = .{ .s = settings };
    while (f.next()) |field| {
        const colon = std.mem.indexOfScalar(u8, field, ':') orelse continue;
        const key = std.mem.trim(u8, field[0..colon], " \t");
        const val = std.mem.trim(u8, field[colon + 1 ..], " \t");
        if (std.mem.eql(u8, key, "Steps")) {
            out.steps = std.fmt.parseInt(usize, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "CFG scale")) {
            out.cfg = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Shift")) {
            out.shift = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Denoise")) {
            out.denoise = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Seed")) {
            out.seed = std.fmt.parseInt(u64, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "Cond noise")) {
            out.cond_noise = unquote(val);
        } else if (std.mem.eql(u8, key, "Cond noise amount")) {
            out.cond_noise_amount = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Cond noise seed")) {
            out.cond_noise_seed = std.fmt.parseInt(u64, val, 10) catch null;
        } else if (std.mem.eql(u8, key, "Cond noise negative")) {
            out.cond_noise_negative = std.fmt.parseFloat(f32, val) catch null;
        } else if (std.mem.eql(u8, key, "Size")) {
            const x = std.mem.indexOfScalar(u8, val, 'x') orelse continue;
            out.width = std.fmt.parseInt(usize, val[0..x], 10) catch null;
            out.height = std.fmt.parseInt(usize, val[x + 1 ..], 10) catch null;
        }
    }
    return out;
}

test "the parameters block names the request the way the engine's writer did" {
    const info: wire.ImageInfo = .{
        .prompt = "a red cube",
        .negative = "blur",
        .req_steps = 12,
        .req_cfg = 3.5,
        .req_seed = 1234,
        .model_stem = "cyberChimera_v10Int8",
        .family = "krea2",
        .params = .{ .sampler = .dpmpp_2m_sde, .scheduler = .karras },
    };
    const p = try paramsAlloc(std.testing.allocator, &info, 1152, 1728);
    defer std.testing.allocator.free(p);
    try std.testing.expect(std.mem.startsWith(u8, p, "a red cube\n"));
    try std.testing.expect(std.mem.indexOf(u8, p, "Negative prompt: blur") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "Steps: 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "Seed: 1234") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "Size: 1152x1728") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "Model: cyberChimera_v10Int8") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "Karras") != null);
}

test "a saved PNG carries the parameters chunk and the pixels" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const info: wire.ImageInfo = .{ .prompt = "p", .req_seed = 9, .start_ns = 5, .req_steps = 1 };
    const rgba = [_]u8{ 10, 20, 30, 255, 40, 50, 60, 255 };
    const path = try saveIn(std.testing.allocator, io, tmp.dir, "out", &info, &rgba, 2, 1);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("out/tp_5_9.png", path);
    const bytes = try tmp.dir.readFileAlloc(io, path, std.testing.allocator, .limited(1 << 20));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, "\x89PNG", bytes[0..4]);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "parameters") != null);
}

test "an a1111 parameters block round-trips through its own parser" {
    const gpa = std.testing.allocator;
    const params = try tp.pipeline.buildA1111Params(
        gpa,
        "a lighthouse in heavy fog\nsecond line of prompt",
        "blurry, low quality",
        34,
        4.2,
        8812,
        1216,
        832,
        "krea2",
        null,
        .euler,
        null,
        .comfy,
        .original,
        .comfy,
        .of(.comfy),
    );
    defer gpa.free(params);

    const got = parseA1111Params(params);
    try std.testing.expectEqualStrings("a lighthouse in heavy fog\nsecond line of prompt", got.prompt);
    try std.testing.expectEqualStrings("blurry, low quality", got.negative);
    try std.testing.expectEqual(@as(?usize, 34), got.steps);
    try std.testing.expectApproxEqAbs(@as(f32, 4.2), got.cfg.?, 0.001);
    try std.testing.expectEqual(@as(?u64, 8812), got.seed);
    try std.testing.expectEqual(@as(?usize, 1216), got.width);
    try std.testing.expectEqual(@as(?usize, 832), got.height);
}

test "parsing tolerates blocks we did not write" {
    // No negative, extra unknown keys, different order: all of these come off
    // images made by ComfyUI or A1111 itself.
    const p = parseA1111Params(
        "just a prompt\nSteps: 20, Sampler: DPM++ 2M, Schedule type: Karras, " ++
            "CFG scale: 7, Seed: 1234, Size: 512x768, Model hash: abc123, Model: sd15",
    );
    try std.testing.expectEqualStrings("just a prompt", p.prompt);
    try std.testing.expectEqualStrings("", p.negative);
    try std.testing.expectEqual(@as(?usize, 20), p.steps);
    try std.testing.expectEqual(@as(?u64, 1234), p.seed);
    try std.testing.expectEqual(@as(?usize, 512), p.width);
    try std.testing.expectEqual(@as(?usize, 768), p.height);

    // A prompt that itself mentions "Steps:" must not be mistaken for the
    // settings line; the LAST one wins.
    const q = parseA1111Params("Steps: how many steps?\nSteps: 12, Seed: 9");
    try std.testing.expectEqualStrings("Steps: how many steps?", q.prompt);
    try std.testing.expectEqual(@as(?usize, 12), q.steps);

    // Nothing at all, and a block with no settings line, must not fault.
    try std.testing.expectEqual(@as(?usize, null), parseA1111Params("").steps);
    try std.testing.expectEqualStrings("only a prompt", parseA1111Params("only a prompt").prompt);

    // A foreign file whose block opens with the settings line: no prompt at all
    // before it, which is where counting back from the end underflowed.
    const r = parseA1111Params("Steps: 8, Seed: 3, Size: 64x64");
    try std.testing.expectEqualStrings("", r.prompt);
    try std.testing.expectEqual(@as(?usize, 8), r.steps);
    try std.testing.expectEqual(@as(?u64, 3), r.seed);
    const s = parseA1111Params("Steps: 8, Seed: 3\n");
    try std.testing.expectEqualStrings("", s.prompt);
    try std.testing.expectEqual(@as(?usize, 8), s.steps);
}

test "buildA1111Params formats prompt, settings, and optional negative" {
    const gpa = std.testing.allocator;

    const with_neg = try tp.pipeline.buildA1111Params(gpa, "a cat", "blurry", 20, 3.5, 42, 1024, 768, "krea2", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(with_neg);
    try std.testing.expectEqualStrings(
        "a cat\n" ++
            "Negative prompt: blurry\n" ++
            "Steps: 20, Sampler: Euler, Schedule type: Simple, CFG scale: 3.5, Seed: 42, Size: 1024x768, Model: krea2, Prompt syntax: ComfyUI",
        with_neg,
    );

    // No negative -> the "Negative prompt:" line is omitted entirely.
    const no_neg = try tp.pipeline.buildA1111Params(gpa, "a dog", "", 8, 1.0, 7, 512, 512, "m", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(no_neg);
    try std.testing.expectEqualStrings(
        "a dog\n" ++
            "Steps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 1.0, Seed: 7, Size: 512x512, Model: m, Prompt syntax: ComfyUI",
        no_neg,
    );
}

test "appendExtraParams writes the A1111/Civitai field names" {
    const gpa = std.testing.allocator;
    const base = try gpa.dupe(u8, "p\nSteps: 30, Sampler: Euler, CFG scale: 5.0, Seed: 1, Size: 8x8, Model: m, Prompt syntax: ComfyUI");
    const out = try tp.pipeline.appendExtraParams(gpa, base, .{
        .clip1 = "qwen3vl_4b",
        .vae = "mage_flow_vae",
        .model_hash = "0123456789",
        .vae_hash = "abcdef0123",
        .weight_dtype = "bf16",
        .shift = 6.0,
        .loras = &.{.{ .name = "style", .hash = "9999888877", .strength = 0.8 }},
    });
    defer gpa.free(out);
    errdefer std.debug.print("{s}\n", .{out});
    for ([_][]const u8{
        ", Model hash: 0123456789",
        ", Clip 1: qwen3vl_4b",
        ", VAE: mage_flow_vae",
        ", VAE hash: abcdef0123",
        ", Weight dtype: bf16",
        ", Shift: 6.0000",
        ", Lora hashes: \"style: 9999888877\"",
        ", Lora strengths: \"style: 0.80\"",
    }) |want| try std.testing.expect(std.mem.indexOf(u8, out, want) != null);
    // The Civitai map is built from the same values, so it cannot disagree.
    try std.testing.expect(std.mem.indexOf(u8, out,
        ", Hashes: {\"model\":\"0123456789\",\"vae\":\"abcdef0123\",\"lora:style\":\"9999888877\"}") != null);
    // Absent fields are absent, not empty: `Clip 2` was never set.
    try std.testing.expect(std.mem.indexOf(u8, out, "Clip 2") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Denoise") == null);
    // Conditioning noise was off, so it contributes nothing at all -- not even the
    // amount, which has a non-null default of its own.
    try std.testing.expect(std.mem.indexOf(u8, out, "Cond noise") == null);
}

test "a conditioning-noise curve round-trips with its commas intact" {
    // `clamp`/`min`/`max` are the curve language's only gating constructs and they
    // all take commas, so a settings line that splits on every comma loses the half
    // of the language worth writing. The value is quoted for exactly that reason.
    const gpa = std.testing.allocator;
    const base = try gpa.dupe(u8, "p\nSteps: 8, Sampler: Euler, CFG scale: 5.0, Seed: 77, Size: 8x8, Model: m, Prompt syntax: ComfyUI");
    const out = try tp.pipeline.appendExtraParams(gpa, base, .{
        .cond_noise = "clamp(a*(1-t),0,1)",
        .cond_noise_amount = 0.25,
        .cond_noise_seed = 4242,
        .cond_noise_negative = 0.5,
    });
    defer gpa.free(out);
    errdefer std.debug.print("{s}\n", .{out});
    try std.testing.expect(std.mem.indexOf(u8, out, ", Cond noise: \"clamp(a*(1-t),0,1)\"") != null);

    const got = parseA1111Params(out);
    try std.testing.expectEqualStrings("clamp(a*(1-t),0,1)", got.cond_noise);
    try std.testing.expectEqual(@as(?f32, 0.25), got.cond_noise_amount);
    try std.testing.expectEqual(@as(?u64, 4242), got.cond_noise_seed);
    try std.testing.expectEqual(@as(?f32, 0.5), got.cond_noise_negative);
    // The fields AROUND it still parse: a quoted comma must not swallow the rest of
    // the line either.
    try std.testing.expectEqual(@as(?u64, 77), got.seed);
    try std.testing.expectEqual(@as(?usize, 8), got.steps);
}

test "an empty Resources leaves the block byte-identical" {
    const gpa = std.testing.allocator;
    const text = "p\nSteps: 8, Sampler: Euler, CFG scale: 1.0, Seed: 1, Size: 512x512, Model: m, Prompt syntax: ComfyUI";
    const base = try gpa.dupe(u8, text);
    const out = try tp.pipeline.appendExtraParams(gpa, base, .{});
    defer gpa.free(out);
    try std.testing.expectEqualStrings(text, out);
}

test "autoV2 is the file's own sha256, sidecar or not" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // sha256("hello\n"), which is what `sha256sum` prints for this file.
    const want = "5891b5b522";

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir_path);
    const file = try std.fmt.allocPrint(gpa, "{s}/f.bin", .{dir_path});
    defer gpa.free(file);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = "hello\n" });

    // Hashed here: the streaming read is the part that silently returned the
    // wrong digest when its buffer aliased the reader's own.
    {
        var h: tp.pipeline.HashCache = .{};
        defer h.deinit(gpa);
        try std.testing.expectEqualStrings(want, h.autoV2(gpa, io, file));
    }
    // And read from a sidecar, which must agree rather than merely be accepted.
    {
        const side = try std.fmt.allocPrint(gpa, "{s}/f.sha256", .{dir_path});
        defer gpa.free(side);
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = side,
            .data = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03\n",
        });
        var h: tp.pipeline.HashCache = .{};
        defer h.deinit(gpa);
        try std.testing.expectEqualStrings(want, h.autoV2(gpa, io, file));
    }
    // A file that is not there is "", not an error: the field is then omitted.
    {
        var h: tp.pipeline.HashCache = .{};
        defer h.deinit(gpa);
        try std.testing.expectEqualStrings("", h.autoV2(gpa, io, "no/such/file.bin"));
    }
}

test "a computed hash is SAVED, in the sidecar format the other tools read" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir_path);
    const file = try std.fmt.allocPrint(gpa, "{s}/f.bin", .{dir_path});
    defer gpa.free(file);
    const side = try std.fmt.allocPrint(gpa, "{s}/f.sha256", .{dir_path});
    defer gpa.free(side);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = "hello\n" });

    {
        var h: tp.pipeline.HashCache = .{};
        defer h.deinit(gpa);
        try std.testing.expectEqualStrings("5891b5b522", h.autoV2(gpa, io, file));
    }
    // ⚠️ 64 bytes, bare hex, NO trailing newline: byte-for-byte what the
    // sidecars already beside these checkpoints hold.
    const written = try std.Io.Dir.cwd().readFileAlloc(io, side, gpa, .limited(4096));
    defer gpa.free(written);
    try std.testing.expectEqualStrings(
        "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
        written,
    );
    // And the second reader takes the sidecar rather than the file.
    var h2: tp.pipeline.HashCache = .{ .compute = false };
    defer h2.deinit(gpa);
    try std.testing.expectEqualStrings("5891b5b522", h2.autoV2(gpa, io, file));
}

test "compute = false neither hashes nor writes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir_path);
    const file = try std.fmt.allocPrint(gpa, "{s}/g.bin", .{dir_path});
    defer gpa.free(file);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = "hello\n" });

    var h: tp.pipeline.HashCache = .{ .compute = false };
    defer h.deinit(gpa);
    try std.testing.expectEqualStrings("", h.autoV2(gpa, io, file));

    const side = try std.fmt.allocPrint(gpa, "{s}/g.sha256", .{dir_path});
    defer gpa.free(side);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, side, .{}));
}

test "Shift and Denoise round-trip through the parser" {
    const gpa = std.testing.allocator;
    const base = try gpa.dupe(u8, "p\nSteps: 30, Sampler: Euler, CFG scale: 5.0, Seed: 1, Size: 8x8, Model: m, Prompt syntax: ComfyUI");
    const out = try tp.pipeline.appendExtraParams(gpa, base, .{ .shift = 6.0, .denoise = 0.75 });
    defer gpa.free(out);
    const got = parseA1111Params(out);
    try std.testing.expect(got.shift != null and @abs(got.shift.? - 6.0) < 1e-4);
    try std.testing.expect(got.denoise != null and @abs(got.denoise.? - 0.75) < 1e-4);
    // The fields the block already carried still parse with the tail appended.
    try std.testing.expectEqual(@as(?usize, 30), got.steps);
    try std.testing.expectEqual(@as(?u64, 1), got.seed);
    try std.testing.expectEqualStrings("p", got.prompt);
}

test "buildA1111Params records the sampler actually used" {
    // A saved PNG is the record a user (or ComfyUI's metadata importer) re-renders
    // from, so a hardcoded sampler name is a wrong answer nothing else would catch,
    // and the a1111 spelling is not the CLI's.
    const gpa = std.testing.allocator;
    for ([_]struct { k: tp.sampler.Kind, want: []const u8 }{
        .{ .k = .euler, .want = "Sampler: Euler," },
        .{ .k = .euler_ancestral, .want = "Sampler: Euler a," },
        .{ .k = .heun, .want = "Sampler: Heun," },
        .{ .k = .dpm_2_ancestral, .want = "Sampler: DPM2 a," },
        .{ .k = .dpmpp_2s_ancestral, .want = "Sampler: DPM++ 2S a," },
        .{ .k = .dpmpp_sde, .want = "Sampler: DPM++ SDE," },
        .{ .k = .dpmpp_2m, .want = "Sampler: DPM++ 2M," },
        .{ .k = .dpmpp_2m_sde, .want = "Sampler: DPM++ 2M SDE," },
        .{ .k = .dpmpp_3m_sde, .want = "Sampler: DPM++ 3M SDE," },
        .{ .k = .dpmpp_2m_sde_heun, .want = "Sampler: DPM++ 2M SDE Heun," },
    }) |c| {
        const s = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, c.k, null, .comfy, .original, .comfy, .{});
        defer gpa.free(s);
        errdefer std.debug.print("{s}\n", .{s});
        try std.testing.expect(std.mem.indexOf(u8, s, c.want) != null);
    }
}

test "buildA1111Params records the sampling compat, and overrides only when overridden" {
    const gpa = std.testing.allocator;

    // An ordinary ComfyUI render's block is byte-for-byte what it was before compat
    // existed, no new fields, so nothing that parses these PNGs has to change.
    const plain = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, .of(.comfy));
    defer gpa.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "Compat") == null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "RNG") == null);

    // A1111's defaults are named once, not spelled out three times.
    const a = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .a1111, .of(.a1111));
    defer gpa.free(a);
    errdefer std.debug.print("{s}\n", .{a});
    try std.testing.expect(std.mem.indexOf(u8, a, "Compat: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "RNG") == null);

    // An override has to be recorded or the block does not describe the render: with
    // `RNG: CPU` this is a different starting latent from the line above, at the same
    // seed. Same reasoning that stopped `Sampler` being hardcoded.
    var cc: tp.pipeline.CompatConfig = .of(.a1111);
    cc.noise_src = .torch_cpu;
    const ov = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .a1111, cc);
    defer gpa.free(ov);
    errdefer std.debug.print("{s}\n", .{ov});
    try std.testing.expect(std.mem.indexOf(u8, ov, "Compat: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, ov, "RNG: CPU") != null);
    // And the two knobs still at A1111's defaults stay out of it.
    try std.testing.expect(std.mem.indexOf(u8, ov, "SGM") == null);

    // The reverse: ComfyUI conventions with A1111's noise, which is the single-variable
    // experiment someone chasing a mismatch would actually run.
    var cn: tp.pipeline.CompatConfig = .of(.comfy);
    cn.noise_src = .nv_philox;
    const nv = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, cn);
    defer gpa.free(nv);
    try std.testing.expect(std.mem.indexOf(u8, nv, "RNG: NV") != null);
    try std.testing.expect(std.mem.indexOf(u8, nv, "Compat") == null);
}

test "buildA1111Params records the prompt dialect, and the emphasis only when it applies" {
    // The same prompt text renders a DIFFERENT image in the two dialects, so a block that
    // does not say which one was used cannot be re-rendered from. `Emphasis` appears only
    // under a1111, where it is a real choice.
    const gpa = std.testing.allocator;
    const a = try tp.pipeline.buildA1111Params(gpa, "(a:1.2) [b]", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .a1111, .no_norm, .comfy, .{});
    defer gpa.free(a);
    try std.testing.expect(std.mem.indexOf(u8, a, "Prompt syntax: A1111") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "Emphasis: No norm") != null);

    const c = try tp.pipeline.buildA1111Params(gpa, "(a:1.2)", "", 20, 7.5, 1, 512, 512, "m", .sdxl, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(c);
    try std.testing.expect(std.mem.indexOf(u8, c, "Prompt syntax: ComfyUI") != null);
    try std.testing.expect(std.mem.indexOf(u8, c, "Emphasis") == null);
}

test "buildA1111Params names the family's own schedule, and omits it when unknown" {
    const gpa = std.testing.allocator;

    // The SD family samples the discrete beta ladder linearly, A1111's "Normal",
    // not krea2's flow-matching "Simple". A reader re-renders from this field.
    for ([_]tp.pipeline.Family{ .sd15, .sdxl }) |f| {
        const s = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", f, .euler, null, .comfy, .original, .comfy, .{});
        defer gpa.free(s);
        try std.testing.expect(std.mem.indexOf(u8, s, "Schedule type: Normal,") != null);
    }
    const k = try tp.pipeline.buildA1111Params(gpa, "p", "", 8, 1.0, 1, 1024, 1024, "m", .krea2, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(k);
    try std.testing.expect(std.mem.indexOf(u8, k, "Schedule type: Simple,") != null);

    // An EXPLICIT scheduler wins over the family default, and this is what the
    // field is for: with schedulers selectable, deriving it from the architecture
    // stamps a name the image was not rendered with.
    const karras = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .sd15, .euler, .karras, .comfy, .original, .comfy, .{});
    defer gpa.free(karras);
    try std.testing.expect(std.mem.indexOf(u8, karras, "Schedule type: Karras,") != null);
    const kl = try tp.pipeline.buildA1111Params(gpa, "p", "", 20, 7.5, 1, 512, 512, "m", .krea2, .euler, .kl_optimal, .comfy, .original, .comfy, .{});
    defer gpa.free(kl);
    try std.testing.expect(std.mem.indexOf(u8, kl, "Schedule type: KL Optimal,") != null);

    // Unknown architecture: drop the field rather than stamp a guess a reader
    // would reproduce with. Everything around it stays well-formed.
    const unknown = try tp.pipeline.buildA1111Params(gpa, "p", "", 8, 1.0, 1, 512, 512, "m", null, .euler, null, .comfy, .original, .comfy, .{});
    defer gpa.free(unknown);
    try std.testing.expect(std.mem.indexOf(u8, unknown, "Schedule type") == null);
    try std.testing.expectEqualStrings(
        "p\nSteps: 8, Sampler: Euler, CFG scale: 1.0, Seed: 1, Size: 512x512, Model: m, Prompt syntax: ComfyUI",
        unknown,
    );
}
