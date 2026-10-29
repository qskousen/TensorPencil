//! `quant-attrib <dense> <quant>`: which layers of a quantized checkpoint cost the
//! whole model the most.
//!
//! The dense model's own sampling trajectory is recorded once. Every arm then
//! predicts at those same latents (teacher-forced, never stepping its own), so its
//! velocity error is the weights' alone and cannot compound into a different picture.
//! An arm is a `weights.Mix` of the two files: `restore` serves the quantized file
//! with one group put back to dense and reports how much error that removes;
//! `quantize` serves the dense file with one group quantized.
//!
//! The quantized file runs through its real kernels, so an int8-activation format
//! (W4A8, W6A8, int8 convrot) is measured with its activation quantization, which a
//! weight round-trip cannot see.

const std = @import("std");
const Io = std.Io;
const tp = @import("TensorPencil");
const pipeline = tp.pipeline;
const sampler = tp.sampler;
const Mix = tp.weights.Mix;

pub const By = enum { layer, role, block };
pub const Mode = enum { restore, quantize };

pub const Options = struct {
    dense: []const u8,
    quant: []const u8,
    prompts: []const []const u8,
    by: By = .role,
    mode: Mode = .restore,
    steps: usize = 8,
    width: usize = 512,
    height: usize = 512,
    seed: u64 = 0,
    shift: ?f32 = null,
    backend: pipeline.Backend = .cuda,
    text_encoder_path: ?[]const u8 = null,
    vae_path: ?[]const u8 = null,
    /// Measure only the groups with these labels.
    only: []const []const u8 = &.{},
    csv_path: ?[]const u8 = null,
};

const default_prompts = [_][]const u8{
    "a close-up portrait photograph of an elderly fisherman with a weathered face, soft window light",
    "a vintage travel poster with the bold text \"VISIT THE COAST\" over a red sports car on a cliff road",
};

/// How far one prediction sits from the reference's, at one (prompt, step).
pub const Point = struct {
    prompt: usize,
    step: usize,
    sigma: f32,
    /// ‖v' − v‖ / ‖v‖ over the whole velocity.
    rel_l2: f64,
    cos: f64,
    /// Worst per-position relative error; positions are strided by `positions`.
    max_pos_rel: f64,
};

pub fn point(prompt: usize, step: usize, sigma: f32, v_ref: []const f32, v: []const f32, positions: usize) Point {
    var dot: f64 = 0;
    var nr: f64 = 0;
    var na: f64 = 0;
    var derr: f64 = 0;
    for (v_ref, v) |a, b| {
        const fa: f64 = a;
        const fb: f64 = b;
        dot += fa * fb;
        nr += fa * fa;
        na += fb * fb;
        derr += (fa - fb) * (fa - fb);
    }
    const channels = v_ref.len / positions;
    var max_rel: f64 = 0;
    for (0..positions) |p| {
        var pnr: f64 = 0;
        var pderr: f64 = 0;
        for (0..channels) |c| {
            const fa: f64 = v_ref[c * positions + p];
            const fb: f64 = v[c * positions + p];
            pnr += fa * fa;
            pderr += (fa - fb) * (fa - fb);
        }
        // A zero reference has no scale; counting it would read as perfect.
        if (pnr > 0) max_rel = @max(max_rel, @sqrt(pderr / pnr));
    }
    return .{
        .prompt = prompt,
        .step = step,
        .sigma = sigma,
        .rel_l2 = if (nr > 0) @sqrt(derr / nr) else 0,
        .cos = if (nr > 0 and na > 0) dot / (@sqrt(nr) * @sqrt(na)) else 1,
        .max_pos_rel = max_rel,
    };
}

/// The group a layer belongs to: `blocks.3.attn.wq` is role `blocks.N.attn.wq` and
/// block `blocks.3`. A layer with no numeric component is its own role and block.
pub fn groupLabel(buf: []u8, layer: []const u8, by: By) ![]const u8 {
    if (by == .layer) return layer;
    var it = std.mem.splitScalar(u8, layer, '.');
    var len: usize = 0;
    var replaced = false;
    while (it.next()) |part| {
        const numeric = part.len > 0 and for (part) |ch| {
            if (!std.ascii.isDigit(ch)) break false;
        } else true;
        const piece = if (numeric and !replaced and by == .role) "N" else part;
        if (len > 0) {
            if (len >= buf.len) return error.NameTooLong;
            buf[len] = '.';
            len += 1;
        }
        if (len + piece.len > buf.len) return error.NameTooLong;
        @memcpy(buf[len..][0..piece.len], piece);
        len += piece.len;
        if (numeric and !replaced) {
            replaced = true;
            if (by == .block) return buf[0..len];
        }
    }
    return if (by == .block and !replaced) layer else buf[0..len];
}

/// Layers whose storage differs between the files: a name `b` has and `a` lacks,
/// or one whose dtype or shape differs. Errors if such a layer has no tensor in `a`
/// at all, which means the two files do not name the same model; `stray` then holds
/// the offending name.
pub fn quantizedLayers(arena: std.mem.Allocator, a: tp.weights.WeightStore, b: tp.weights.WeightStore, stray: *[]const u8) ![]const []const u8 {
    var dense_layers: std.StringHashMapUnmanaged(void) = .empty;
    for (a.names()) |n| try dense_layers.put(arena, Mix.layerOf(n), {});
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.ArrayList([]const u8) = .empty;
    for (b.names()) |n| {
        const layer = Mix.layerOf(n);
        if (seen.contains(layer)) continue;
        const differs = if (a.get(n)) |va| blk: {
            const vb = b.get(n).?;
            break :blk va.info.dtype != vb.info.dtype or !va.info.shape.eql(vb.info.shape);
        } else true;
        if (!differs) continue;
        if (!dense_layers.contains(layer)) {
            stray.* = n;
            return error.NamespaceMismatch;
        }
        try seen.put(arena, layer, {});
        try out.append(arena, layer);
    }
    return out.items;
}

const Snapshot = struct { prompt: usize, step: usize, sigma: f32, x: []f32, v: []f32 };

const Arm = struct {
    label: []const u8,
    layers: usize,
    points: []Point,

    fn mean(self: Arm) f64 {
        var s: f64 = 0;
        for (self.points) |p| s += p.rel_l2;
        return s / @as(f64, @floatFromInt(self.points.len));
    }

    fn worst(self: Arm) f64 {
        var w: f64 = 0;
        for (self.points) |p| w = @max(w, p.rel_l2);
        return w;
    }
};

pub fn run(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, opts: Options) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var popts: pipeline.Options = .{
        .prompt = opts.prompts[0],
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .seed = opts.seed,
        .backend = opts.backend,
        .dit_path = opts.dense,
        .preview = false,
    };
    if (opts.text_encoder_path) |p| {
        popts.text_encoder_path = p;
        popts.explicit_text_encoder = true;
    }
    if (opts.vae_path) |p| {
        popts.vae_path = p;
        popts.explicit_vae = true;
    }
    var sess = try pipeline.Session.init(io, gpa, popts, out);
    defer sess.deinit();

    const shape = try sess.latentShape(opts.width, opts.height, 1);
    const lat_len = shape.elems();
    const sigmas = try sess.scheduleWith(arena, opts.steps, opts.shift orelse pipeline.defaultShift(sess.family()), null);

    // Encode everything first: arming residency may drop the text encoder.
    const conds = try arena.alloc(pipeline.Cond, opts.prompts.len);
    for (opts.prompts, conds) |p, *c| c.* = try sess.encode(gpa, p, .{});
    defer for (conds) |*c| c.deinit(gpa);

    var quant_st = try tp.safetensors.SafeTensors.open(gpa, io, opts.quant);
    defer quant_st.deinit();
    var mix = try Mix.init(gpa, sess.denoiserStore(), .{ .safetensors = &quant_st });
    defer mix.deinit(gpa);

    var stray: []const u8 = "";
    const qlayers = quantizedLayers(arena, sess.denoiserStore(), mix.b, &stray) catch |err| {
        if (err == error.NamespaceMismatch) std.log.err("quant-attrib: '{s}' is not a layer of the dense file; the two files name different models or use different prefixes", .{stray});
        return err;
    };
    if (qlayers.len == 0) return error.NothingQuantized;
    const prefix = sess.denoiserPrefix();

    // Groups in first-seen order, each holding its layers.
    var group_names: std.ArrayList([]const u8) = .empty;
    var groups: std.StringHashMapUnmanaged(std.ArrayList([]const u8)) = .empty;
    for (qlayers) |l| {
        var buf: [512]u8 = undefined;
        const stripped = if (std.mem.startsWith(u8, l, prefix)) l[prefix.len..] else l;
        const label = try arena.dupe(u8, try groupLabel(&buf, stripped, opts.by));
        const gop = try groups.getOrPut(arena, label);
        if (!gop.found_existing) {
            gop.value_ptr.* = .empty;
            try group_names.append(arena, label);
        }
        try gop.value_ptr.append(arena, l);
    }
    for (opts.only) |o| if (!groups.contains(o)) {
        std.log.err("quant-attrib: no group '{s}' (by {s})", .{ o, @tagName(opts.by) });
        return error.UnknownGroup;
    };
    try out.print("{d} quantized layers in {d} groups (by {s}), {d} prompts x {d} steps at {d}x{d}\n", .{
        qlayers.len, group_names.items.len, @tagName(opts.by), opts.prompts.len, opts.steps, opts.width, opts.height,
    });
    try out.flush();

    // Reference: the dense model's own trajectory, through the same Mix path every arm takes.
    try mix.select(gpa, &.{});
    try sess.replaceDenoiser(mix.store());
    sess.armDenoiserResidency(null);
    const snaps = try arena.alloc(Snapshot, opts.prompts.len * opts.steps);
    {
        const x = try gpa.alloc(f32, lat_len);
        defer gpa.free(x);
        for (conds, 0..) |cond, pi| {
            sampler.fillNoiseFrom(x, opts.seed + pi, sess.compat.noise_src);
            sess.scaleInitialNoise(x, sigmas[0], shape);
            var den = try sess.denoiser(gpa, cond, null, 1.0, shape.h, shape.w, sigmas);
            defer den.deinit(gpa);
            for (0..opts.steps) |i| {
                const v = try arena.alloc(f32, lat_len);
                try den.predict(gpa, v, x, sigmas[i], null);
                snaps[pi * opts.steps + i] = .{ .prompt = pi, .step = i, .sigma = sigmas[i], .x = try arena.dupe(f32, x), .v = v };
                sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
            }
        }
    }

    var csv_file: ?Io.File = null;
    var csv_buf: [4096]u8 = undefined;
    var csv_w: Io.File.Writer = undefined;
    if (opts.csv_path) |p| {
        csv_file = try Io.Dir.cwd().createFile(io, p, .{ .truncate = true });
        csv_w = csv_file.?.writer(io, &csv_buf);
        try csv_w.interface.writeAll("arm,layers,prompt,step,sigma,rel_l2,cos,max_pos_rel\n");
    }
    defer if (csv_file) |f| f.close(io);

    const measure = struct {
        fn f(s: *pipeline.Session, g: std.mem.Allocator, a: std.mem.Allocator, m: *Mix, layers: []const []const u8, prompts: []const []const u8, c: []pipeline.Cond, sn: []const Snapshot, sg: []const f32, sh: pipeline.LatentShape) ![]Point {
            try m.select(g, layers);
            try s.replaceDenoiser(m.store());
            s.armDenoiserResidency(null);
            const pts = try a.alloc(Point, sn.len);
            const v = try g.alloc(f32, sh.elems());
            defer g.free(v);
            const steps = sn.len / c.len;
            const own = s.family().conditionsFromDenoiser();
            for (c, 0..) |dense_cond, pi| {
                // Here the arm's own weights encode, or its encoder layers would read as free.
                var arm_cond: ?pipeline.Cond = if (own) try s.encode(g, prompts[pi], .{}) else null;
                defer if (arm_cond) |*ac| ac.deinit(g);
                var den = try s.denoiser(g, arm_cond orelse dense_cond, null, 1.0, sh.h, sh.w, sg);
                defer den.deinit(g);
                for (0..steps) |i| {
                    const snap = sn[pi * steps + i];
                    try den.predict(g, v, snap.x, snap.sigma, null);
                    pts[pi * steps + i] = point(pi, i, snap.sigma, snap.v, v, sh.h * sh.w);
                }
            }
            return pts;
        }
    }.f;

    var arms: std.ArrayList(Arm) = .empty;
    const t0 = Io.Clock.real.now(io).nanoseconds;
    // Controls: the reload must read zero, and the whole file is what groups are read against.
    for ([_]struct { []const u8, []const []const u8 }{ .{ "control:dense", &.{} }, .{ "control:all", qlayers } }) |c| {
        try arms.append(arena, .{ .label = c[0], .layers = c[1].len, .points = try measure(sess, gpa, arena, &mix, c[1], opts.prompts, conds, snaps, sigmas, shape) });
        try report(out, arms.last().?, null, t0, io);
    }
    const all_err = arms.items[1].mean();

    var rest: std.ArrayList([]const u8) = .empty;
    for (group_names.items) |label| {
        if (opts.only.len > 0 and !contains(opts.only, label)) continue;
        const members = groups.get(label).?.items;
        const sel = switch (opts.mode) {
            .quantize => members,
            .restore => blk: {
                rest.clearRetainingCapacity();
                for (qlayers) |l| if (!contains(members, l)) try rest.append(arena, l);
                break :blk rest.items;
            },
        };
        try arms.append(arena, .{ .label = label, .layers = members.len, .points = try measure(sess, gpa, arena, &mix, sel, opts.prompts, conds, snaps, sigmas, shape) });
        try report(out, arms.last().?, if (opts.mode == .restore) all_err else null, t0, io);
        if (csv_file != null) try writeCsv(&csv_w.interface, arms.last().?);
    }
    if (csv_file != null) {
        for (arms.items[0..2]) |a| try writeCsv(&csv_w.interface, a);
        try csv_w.interface.flush();
    }

    // Ranked summary: biggest effect first.
    const ranked = arms.items[2..];
    std.mem.sort(Arm, ranked, opts.mode, struct {
        fn lt(mode: Mode, x: Arm, y: Arm) bool {
            return if (mode == .restore) x.mean() < y.mean() else x.mean() > y.mean();
        }
    }.lt);
    try out.print("\n{s} ranking ({s}), whole file mean rel {d:.4}, dense reload {d:.5}\n", .{
        @tagName(opts.mode), if (opts.mode == .restore) "error removed by keeping the group dense" else "error added by quantizing the group alone", all_err, arms.items[0].mean(),
    });
    try out.print("{s:<40} {s:>6} {s:>9} {s:>9} {s:>8}\n", .{ "group", "layers", "mean rel", "worst", if (opts.mode == .restore) "removed" else "of all" });
    for (ranked) |a| {
        const share = if (opts.mode == .restore) (all_err - a.mean()) / all_err else a.mean() / all_err;
        try out.print("{s:<40} {d:>6} {d:>9.4} {d:>9.4} {d:>7.1}%\n", .{ a.label, a.layers, a.mean(), a.worst(), share * 100 });
    }
    try out.flush();
}

fn contains(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

fn report(out: *Io.Writer, a: Arm, all_err: ?f64, t0: i96, io: Io) !void {
    const secs = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t0)) / 1e9;
    try out.print("[{d:>6.0}s] {s:<40} layers {d:>4}  mean rel {d:.4}  worst {d:.4}", .{ secs, a.label, a.layers, a.mean(), a.worst() });
    if (all_err) |e| try out.print("  removes {d:.1}%", .{(e - a.mean()) / e * 100});
    try out.writeAll("\n");
    try out.flush();
}

fn writeCsv(w: *Io.Writer, a: Arm) !void {
    for (a.points) |p| try w.print("{s},{d},{d},{d},{d:.4},{d:.6},{d:.6},{d:.6}\n", .{ a.label, a.layers, p.prompt, p.step, p.sigma, p.rel_l2, p.cos, p.max_pos_rel });
}

pub const usage =
    \\quant-attrib <dense.safetensors> <quant.safetensors> [options]
    \\  Rank which layer groups of a quantized checkpoint damage the whole model,
    \\  teacher-forced on the dense model's own trajectory.
    \\  --by role|block|layer       grouping (default role)
    \\  --mode restore|quantize     restore: one group dense in the quantized model (default)
    \\                              quantize: one group quantized in the dense model
    \\  --prompt P                  repeatable (default: two built-in prompts)
    \\  --steps N --size N | --width N --height N --seed N --shift F
    \\  --backend cuda|zig-cuda|vulkan|cpu --text-encoder P --vae P
    \\  --only G1,G2,...            measure only these groups
    \\  --csv PATH                  every (arm, prompt, step) point
    \\
;

pub fn cli(io: Io, out: *Io.Writer, args: []const []const u8) !void {
    const gpa = std.heap.smp_allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (args.len < 2) {
        try out.writeAll(usage);
        return error.InvalidArgs;
    }
    var opts: Options = .{ .dense = args[0], .quant = args[1], .prompts = &default_prompts };
    var prompts: std.ArrayList([]const u8) = .empty;
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) {
            std.log.err("quant-attrib: {s} needs a value", .{flag});
            return error.InvalidArgs;
        }
        const v = args[i + 1];
        if (std.mem.eql(u8, flag, "--by")) {
            opts.by = std.meta.stringToEnum(By, v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, flag, "--mode")) {
            opts.mode = std.meta.stringToEnum(Mode, v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, flag, "--prompt")) {
            try prompts.append(arena, v);
        } else if (std.mem.eql(u8, flag, "--steps")) {
            opts.steps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, flag, "--size")) {
            opts.width = try std.fmt.parseInt(usize, v, 10);
            opts.height = opts.width;
        } else if (std.mem.eql(u8, flag, "--width")) {
            opts.width = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, flag, "--height")) {
            opts.height = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, flag, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, flag, "--shift")) {
            opts.shift = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, flag, "--backend")) {
            opts.backend = pipeline.Backend.fromStr(v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, flag, "--text-encoder")) {
            opts.text_encoder_path = v;
        } else if (std.mem.eql(u8, flag, "--vae")) {
            opts.vae_path = v;
        } else if (std.mem.eql(u8, flag, "--only")) {
            var list: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |g| try list.append(arena, g);
            opts.only = list.items;
        } else if (std.mem.eql(u8, flag, "--csv")) {
            opts.csv_path = v;
        } else {
            std.log.err("quant-attrib: unknown flag {s}", .{flag});
            try out.writeAll(usage);
            return error.InvalidArgs;
        }
    }
    if (opts.steps < 1) return error.InvalidArgs;
    if (prompts.items.len > 0) opts.prompts = prompts.items;
    try run(io, gpa, out, opts);
}

test "groupLabel: role replaces the block index, block keeps through it" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("blocks.N.attn.wq", try groupLabel(&buf, "blocks.12.attn.wq", .role));
    try std.testing.expectEqualStrings("blocks.12", try groupLabel(&buf, "blocks.12.attn.wq", .block));
    try std.testing.expectEqualStrings("blocks.12.attn.wq", try groupLabel(&buf, "blocks.12.attn.wq", .layer));
    // Only the first index is the block; a later one is part of the role.
    try std.testing.expectEqualStrings("txt.N.mlp.0", try groupLabel(&buf, "txt.3.mlp.0", .role));
    try std.testing.expectEqualStrings("last.linear", try groupLabel(&buf, "last.linear", .role));
    try std.testing.expectEqualStrings("last.linear", try groupLabel(&buf, "last.linear", .block));
    var tiny: [4]u8 = undefined;
    try std.testing.expectError(error.NameTooLong, groupLabel(&tiny, "blocks.1.attn", .role));
}

test "point: identical is zero error, a scaled copy keeps direction, a zero position is skipped" {
    const v_ref = [_]f32{ 1, 0, 2, 0 };
    const same = point(0, 0, 1, &v_ref, &v_ref, 2);
    try std.testing.expectEqual(@as(f64, 0), same.rel_l2);
    try std.testing.expectApproxEqAbs(@as(f64, 1), same.cos, 1e-12);
    const half = [_]f32{ 0.5, 0, 1, 0 };
    const p = point(0, 0, 1, &v_ref, &half, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), p.rel_l2, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), p.cos, 1e-12);
    // Position 1 has a zero reference, so the worst position is position 0's 0.5.
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), p.max_pos_rel, 1e-12);
}

test "quantizedLayers: a changed or extended layer counts, an identical one does not" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const SafeTensors = tp.safetensors.SafeTensors;
    const ha =
        \\{"k.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"l.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},"n.weight":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}}
    ;
    const hb =
        \\{"k.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"l.weight":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},"n.weight":{"dtype":"F32","shape":[1],"data_offsets":[12,16]},"n.scale":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}}
    ;
    var fa: [8 + ha.len + 20]u8 = undefined;
    var fb: [8 + hb.len + 20]u8 = undefined;
    inline for (.{ .{ &fa, ha }, .{ &fb, hb } }) |f| {
        std.mem.writeInt(u64, f[0][0..8], f[1].len, .little);
        @memcpy(f[0][8..][0..f[1].len], f[1]);
        @memset(f[0][8 + f[1].len ..], 0);
    }
    var a = try SafeTensors.initFromSlice(gpa, &fa);
    defer a.deinit();
    var b = try SafeTensors.initFromSlice(gpa, &fb);
    defer b.deinit();
    var stray: []const u8 = "";
    const layers = try quantizedLayers(arena_state.allocator(), .{ .safetensors = &a }, .{ .safetensors = &b }, &stray);
    try std.testing.expectEqual(@as(usize, 2), layers.len);
    try std.testing.expectEqualStrings("l", layers[0]);
    try std.testing.expectEqualStrings("n", layers[1]);

    // A layer the dense file does not have means the files do not line up.
    const hc =
        \\{"z.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    var fc: [8 + hc.len + 4]u8 = undefined;
    std.mem.writeInt(u64, fc[0..8], hc.len, .little);
    @memcpy(fc[8..][0..hc.len], hc);
    @memset(fc[8 + hc.len ..], 0);
    var c = try SafeTensors.initFromSlice(gpa, &fc);
    defer c.deinit();
    try std.testing.expectError(error.NamespaceMismatch, quantizedLayers(arena_state.allocator(), .{ .safetensors = &a }, .{ .safetensors = &c }, &stray));
    try std.testing.expectEqualStrings("z.weight", stray);
}
