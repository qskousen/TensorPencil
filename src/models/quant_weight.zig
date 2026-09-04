//! Container-level loading for the ComfyUI quantized weight formats that more than one
//! model family ships.
//!
//! Exists because NVFP4 arrives on three architectures here (krea2, Anima, Z-Image),
//! each with its own loader, and three copies of "read `weight_scale`, unswizzle it,
//! build the level table" is exactly the drift that makes one family's renders disagree
//! with another's for reasons no shape check can see. One implementation, called from
//! each family's `mat`.
//!
//! Every format ComfyUI's quantizers emit shows up on every family they support, so a
//! container reader for one of them belongs here from the start. A format known to only
//! one family's loader is worse than unsupported: W4A8 and NVFP4 both store
//! `I8 [rows, cols/2]`, which is exactly the int4-convrot signature, so a loader that
//! has not heard of them reads their nibbles as signed int4 times a per-row scale.
//!
//! `int8Scale` is here for the same reason one step on: `int8_tensorwise` ships rotated
//! and unrotated, only the `comfy_quant` blob and the scale's shape tell them apart, and
//! the three families that read it had three copies of the half that handles one of them.

const std = @import("std");
const weights_mod = @import("tp_core").weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// Build a `Weight` for a ComfyUI NVFP4 layer, or null when `name` is not one.
///
/// `name` is the weight tensor's full name (e.g. `model.diffusion_model.blocks.0.attn.wq
/// .weight`); the sidecars are `<name>_scale` (fp8 per-block, swizzled) and
/// `<name>_scale_2` (f32 per-tensor). `rows`/`cols` are the LOGICAL dims, the stored
/// weight is `[rows, cols/2]`.
///
/// Detection is on `_scale_2`, not on `comfy_quant`. Z-Image's NVFP4 checkpoint
/// ships no `comfy_quant` blob at all (krea2's and Anima's carry `{"format": "nvfp4"}`),
/// so a loader keyed on the metadata would silently fail to recognize one of the three
/// files. `_scale_2` is the tensor only this format has.
///
/// Everything allocated, the unswizzled scales, the level table, the `Meta`, lands in
/// `alloc` (the model arena) and must outlive the model. The packed weight bytes are a
/// view into the store, like every other weight here.
pub fn nvfp4(
    alloc: std.mem.Allocator,
    store: WeightStore,
    name: []const u8,
    rows: usize,
    cols: usize,
) !?Weight {
    var buf: [256]u8 = undefined;
    const g_name = std.fmt.bufPrint(&buf, "{s}_scale_2", .{name}) catch return null;
    const gv = store.get(g_name) orelse return null;

    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    // The packed weight is U8 on disk (ComfyUI's `storage_t` for nvfp4). Accept I8 too:
    // the bits, and so the nibble decode, are identical, and other ComfyUI converters
    // have shipped the same bytes under both spellings.
    if ((view.info.dtype != .u8 and view.info.dtype != .i8) or shape.len != 2 or
        cols % 2 != 0 or shape[0] != rows or shape[1] != cols / 2)
    {
        std.log.err("nvfp4: {s} has shape {any} ({t}), expected U8 [{d}, {d}]", .{
            name, shape, view.info.dtype, rows, cols / 2,
        });
        return error.ShapeMismatch;
    }

    var sbuf: [256]u8 = undefined;
    const s_name = std.fmt.bufPrint(&sbuf, "{s}_scale", .{name}) catch return error.ShapeMismatch;
    const sv = store.get(s_name) orelse {
        std.log.err("nvfp4: {s} has a per-tensor scale but no {s}", .{ name, s_name });
        return error.MissingTensor;
    };
    if (sv.info.dtype != .f8_e4m3) {
        std.log.err("nvfp4: {s} is {t}, expected F8_E4M3", .{ s_name, sv.info.dtype });
        return error.ShapeMismatch;
    }
    if (gv.info.elemCount() != 1) {
        std.log.err("nvfp4: {s} has {d} elements, expected a scalar", .{ g_name, gv.info.elemCount() });
        return error.ShapeMismatch;
    }

    const nblk = try ops.nvfp4.validate(rows, cols, view.bytes.len, sv.bytes.len);
    const global = try gv.asScalarF32();

    // Unswizzle ONCE, here, so every consumer (CPU panel, each GPU decode kernel) sees
    // plain row-major `[rows][nblk]`. It is 1/16 of the weight, so the copy is cheap
    // against never putting cuBLAS's five-way tiled index in an inner loop.
    const scales = try alloc.alloc(u8, rows * nblk);
    ops.nvfp4.unswizzleScales(scales, sv.bytes, rows, nblk);

    const levels = try alloc.create(ops.nvfp4.Levels);
    levels.* = ops.nvfp4.Levels.init(global);
    const meta = try alloc.create(ops.nvfp4.Meta);
    meta.* = .{ .scales = scales, .levels = levels };

    var w = Weight.init(view.bytes, .nvfp4, rows, cols);
    w.nvfp4 = meta;
    // Deliberately NOT set: `row_scale` (the scale is per block, not per output row) and
    // `convrot` (this format is not Hadamard-rotated, unlike int8/int4/w4a8 here).
    // `matmul` asserts both are absent for `.nvfp4`.
    return w;
}

/// `blocks.3.attn.wq.weight` -> `blocks.3.attn.wq.comfy_quant`.
///
/// The quant metadata hangs off the *layer*, not off the weight tensor, so it is the one
/// sidecar whose name is not `<weight><suffix>`.
fn quantConfName(buf: []u8, name: []const u8) ![]u8 {
    const base = if (std.mem.endsWith(u8, name, ".weight")) name[0 .. name.len - ".weight".len] else name;
    return std.fmt.bufPrint(buf, "{s}.comfy_quant", .{base});
}

/// The dequant metadata of a ComfyUI `int8_tensorwise` layer: a scale per output row and
/// the rotation group, 0 when the weight is not rotated.
pub const Int8Meta = struct {
    row_scale: []f32,
    convrot: u32,
};

/// The fields of `comfy_quant` this reader uses.
///
/// Both keys may sit at the top level or nested under `params`, and ComfyUI's loader
/// reads them in that order, so this does too. Optional rather than defaulted, because
/// "absent" and "present and false" resolve differently per format below.
const Int8Conf = struct {
    format: []const u8 = "",
    convrot: ?bool = null,
    convrot_groupsize: ?usize = null,
    params: Nested = .{},

    const Nested = struct {
        convrot: ?bool = null,
        convrot_groupsize: ?usize = null,
    };
};

/// Formats with a reader of their own, all of which run BEFORE this one. Seeing one on a
/// weight that reached the int8 arm means the sidecars and the metadata disagree, and
/// reading those bytes as int8 gives finite, plausible, wrong numbers.
const foreign_formats = [_][]const u8{
    "nvfp4", "asym_w4a8_int8", "float8_e4m3fn", "float8_e5m2", "mxfp8",
};

/// ComfyUI's writer emits `convrot` only when it is true, so for `int8_tensorwise` an
/// absent key means unrotated.
const tensorwise_format = "int8_tensorwise";

/// ...but `convrot_w4a4` is ALWAYS rotated and therefore never writes the key at all.
/// Reading its absence as false would unrotate every int4 checkpoint.
const always_rot_format = "convrot_w4a4";

/// Read the `weight_scale` and `comfy_quant` sidecars of a ComfyUI `int8_tensorwise`
/// weight. `name` is the weight tensor's full name; `rows`/`cols` are the LOGICAL dims.
///
/// Two variants ship, and they are different arithmetic, not two spellings of one thing:
///
///   - `weight_scale` `[rows, 1]` with `"convrot": true`. The weight was quantized after
///     a size-256 group Hadamard, so the activation has to be rotated to match; the
///     rotation then cancels across the GEMM.
///   - `weight_scale` a SCALAR with no `convrot`. One scale for the whole tensor and no
///     rotation on either side. Broadcast per row here so every consumer sees the same
///     `row_scale` and only `convrot` distinguishes the two.
///
/// `int8Rotation` decides which, and is where the metadata's edge cases live.
pub fn int8Scale(
    alloc: std.mem.Allocator,
    store: WeightStore,
    name: []const u8,
    rows: usize,
    cols: usize,
) !Int8Meta {
    var buf: [256]u8 = undefined;
    const s_name = try std.fmt.bufPrint(&buf, "{s}_scale", .{name});
    const sv = store.get(s_name) orelse {
        std.log.err("int8: {s} is int8 but has no {s}", .{ name, s_name });
        return error.MissingTensor;
    };
    const n = sv.info.elemCount();
    var nbuf: [256]u8 = undefined;
    const conf_json: ?[]const u8 = if (store.get(try quantConfName(&nbuf, name))) |qv| qv.bytes else null;

    const rot = int8Rotation(alloc, n, rows, cols, conf_json) catch |e| {
        std.log.err("int8: {s} cannot be decoded ({t}): {d} rows, {d} columns, {s} " ++
            "has {d} entries, comfy_quant {?s}", .{ name, e, rows, cols, s_name, n, conf_json });
        return e;
    };

    const row_scale = if (n == rows) try sv.toF32Alloc(alloc) else blk: {
        const s = try sv.asScalarF32();
        const out = try alloc.alloc(f32, rows);
        @memset(out, s);
        break :blk out;
    };
    return .{ .row_scale = row_scale, .convrot = rot };
}

/// `int8Scale` for a family whose activation prep always rotates (Anima, MiniMax H3):
/// an unrotated weight there would be multiplied in a basis it was never quantized in,
/// so refuse it where it is read rather than render noise. `who` names the family in the
/// diagnostic. Drop this in favour of `int8Scale` the day that family's prep can skip
/// the rotation, as krea2's can.
pub fn int8ScaleConvrot(
    alloc: std.mem.Allocator,
    store: WeightStore,
    name: []const u8,
    rows: usize,
    cols: usize,
    who: []const u8,
) !Int8Meta {
    const meta = try int8Scale(alloc, store, name, rows, cols);
    if (meta.convrot == 0) {
        std.log.err("{s}: {s} is int8 with no convrot rotation, which this model's " ++
            "activation prep cannot pair with", .{ who, name });
        return error.UnsupportedCheckpoint;
    }
    return meta;
}

/// The rotation group an `int8_tensorwise` layer decodes with (0 = unrotated), from its
/// `weight_scale`'s element count and its `comfy_quant` bytes, or an error when the two
/// describe a weight this cannot read.
///
/// Separate from `int8Scale` and free of the container so every refusal is reachable
/// from a plain test: each one of them is a file that would otherwise decode to finite,
/// plausible, wrong numbers. `int8Scale` owns the diagnostic, so nothing here logs.
fn int8Rotation(
    alloc: std.mem.Allocator,
    scale_elems: usize,
    rows: usize,
    cols: usize,
    conf_json: ?[]const u8,
) !u32 {
    if (scale_elems != rows and scale_elems != 1) return error.ShapeMismatch;

    // What an ABSENT `convrot` key means. With no blob at all, or a format spelling we do
    // not know (converters invent private ones), the scale's own shape answers, which is
    // sound rather than a guess: ComfyUI refuses convrot unless the scale is per channel.
    var rot = scale_elems == rows;
    var group: usize = ops.convrot.group_size;
    if (conf_json) |bytes| {
        const conf = std.json.parseFromSliceLeaky(Int8Conf, alloc, bytes, .{
            .ignore_unknown_fields = true,
        }) catch return error.UnsupportedCheckpoint;
        for (foreign_formats) |f| {
            if (std.mem.eql(u8, conf.format, f)) return error.UnsupportedCheckpoint;
        }
        if (std.mem.eql(u8, conf.format, tensorwise_format)) rot = false;
        if (std.mem.eql(u8, conf.format, always_rot_format)) rot = true;
        rot = conf.convrot orelse conf.params.convrot orelse rot;
        group = conf.convrot_groupsize orelse conf.params.convrot_groupsize orelse ops.convrot.group_size;
    }
    if (!rot) return 0;

    // ComfyUI's own quantizer rejects convrot on a per-tensor scale, so a file carrying
    // it was built by something else and there is no telling which of the two it meant.
    if (scale_elems != rows) return error.ShapeMismatch;
    // `ops.convrot` implements the size-256 Hadamard only (a comptime table and a
    // radix-4 FWHT over 4 base-4 digits); rotating by another basis is silent noise.
    if (group != ops.convrot.group_size) return error.UnsupportedCheckpoint;
    if (cols % ops.convrot.group_size != 0) return error.ShapeMismatch;
    return ops.convrot.group_size;
}

/// Build a `Weight` for a ComfyUI `asym_w4a8_int8` layer, or null when `name` is not one.
///
/// The sidecars are `<name>_s_rel` (fp8, per `group_size` elements), `<name>_s_channel`
/// (f32, one per output row), an optional `<name>_codebook` (f32[16]) and the layer's
/// `comfy_quant` blob. `rows`/`cols` are the LOGICAL dims, the stored weight is
/// `I8 [rows, cols/2]`, two 4-bit indices per byte. `ops/w4a8.zig` documents the format.
///
/// Detection is on `_s_rel`, and it must be tried BEFORE any int4 heuristic, since
/// `I8 [rows, cols/2]` is exactly the int4-convrot signature. W4A8's nibbles are unsigned
/// indices into a non-uniform Lloyd-Max codebook, so reading them as signed int4 times a
/// per-row scale is finite, plausible and wrong.
///
/// The weight stays PACKED. Materializing the int8 here works on every backend with
/// no kernel, the decode's output is an ordinary int8-convrot weight, but it doubles the
/// footprint (12.2 GB against 6.1 GB for krea2) and turns an mmap'd, evictable checkpoint
/// into that much anonymous RSS, which is precisely the difference between this format and
/// int8. Every consumer decodes on demand instead: the CPU GEMM per k-slice into its
/// existing dequant panel, each GPU backend per GEMM into a device scratch.
///
/// Everything allocated lands in `alloc` (the model arena) and must outlive the model; the
/// packed bytes and `s_rel` are views into the store, like every other weight here.
pub fn w4a8(
    alloc: std.mem.Allocator,
    store: WeightStore,
    name: []const u8,
    rows: usize,
    cols: usize,
) !?Weight {
    var nbuf: [288]u8 = undefined;
    const sr_name = std.fmt.bufPrint(&nbuf, "{s}_s_rel", .{name}) catch return null;
    const sv = store.get(sr_name) orelse return null;

    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (view.info.dtype != .i8 or shape.len != 2 or cols % 2 != 0 or
        shape[0] != rows or shape[1] != cols / 2)
    {
        std.log.err("w4a8: {s} has shape {any} ({t}), expected I8 [{d}, {d}]", .{
            name, shape, view.info.dtype, rows, cols / 2,
        });
        return error.ShapeMismatch;
    }

    // The group size is derived from `_s_rel`'s OWN shape and only then cross-checked
    // against `comfy_quant`. Trusting the JSON alone would let a stale or mistyped
    // `group_size` read every scale from the wrong group, a finite, plausible weight,
    // where a disagreement between the two is an error.
    const sshape = sv.info.shape.slice();
    if (sv.info.dtype != .f8_e4m3 or sshape.len != 2 or sshape[0] != rows or
        sshape[1] == 0 or cols % sshape[1] != 0)
    {
        std.log.err("w4a8: {s}_s_rel has shape {any} ({t}), expected F8_E4M3 [{d}, cols/group]", .{
            name, sshape, sv.info.dtype, rows,
        });
        return error.ShapeMismatch;
    }
    const group_size = cols / sshape[1];

    var convrot_groupsize: usize = ops.w4a8.default_convrot_groupsize;
    if (store.get(try quantConfName(&nbuf, name))) |qv| {
        const conf = ops.w4a8.parseConf(alloc, qv.bytes) catch |e| {
            std.log.err("w4a8: {s}'s comfy_quant is not valid JSON ({t}): '{s}'", .{ name, e, qv.bytes });
            return error.UnsupportedCheckpoint;
        };
        if (!std.mem.eql(u8, conf.format, ops.w4a8.format_name)) {
            std.log.err("w4a8: {s} has a W4A8 scale but comfy_quant says format '{s}'", .{ name, conf.format });
            return error.UnsupportedCheckpoint;
        }
        if (conf.group_size != group_size) {
            std.log.err("w4a8: {s}'s comfy_quant says group_size {d} but weight_s_rel is grouped by {d}", .{
                name, conf.group_size, group_size,
            });
            return error.ShapeMismatch;
        }
        convrot_groupsize = conf.convrot_groupsize;
    }
    // `ops.convrot` implements the size-256 Hadamard only (it is a comptime table and a
    // radix-4 FWHT over 4 base-4 digits). No checkpoint in the wild uses another, and
    // inventing one silently would rotate by the wrong basis.
    if (convrot_groupsize != ops.convrot.group_size) {
        std.log.err("w4a8: {s} declares convrot_groupsize {d}; only {d} is implemented", .{
            name, convrot_groupsize, ops.convrot.group_size,
        });
        return error.UnsupportedCheckpoint;
    }

    const sc_name = try std.fmt.bufPrint(&nbuf, "{s}_s_channel", .{name});
    const scv = store.get(sc_name) orelse {
        std.log.err("w4a8: {s} has a W4A8 group scale but no {s}", .{ name, sc_name });
        return error.MissingTensor;
    };
    if (scv.info.elemCount() != rows) {
        std.log.err("w4a8: {s} has {d} entries, expected {d} (one per output row)", .{
            sc_name, scv.info.elemCount(), rows,
        });
        return error.ShapeMismatch;
    }
    const s_channel = try scv.toF32Alloc(alloc);

    // Absent `weight_codebook` is not an error: a layer quantized with `codebook=False`
    // decodes to the uniform `q - 8` levels instead.
    var cb_store: [16]f32 = undefined;
    var codebook: ?*const [16]f32 = null;
    if (store.get(try std.fmt.bufPrint(&nbuf, "{s}_codebook", .{name}))) |cv| {
        if (cv.info.elemCount() != 16) return error.ShapeMismatch;
        const tmp = try cv.toF32Alloc(alloc);
        @memcpy(&cb_store, tmp[0..16]);
        codebook = &cb_store;
    }

    _ = try ops.w4a8.validate(rows, cols, group_size, convrot_groupsize, view.bytes.len, sv.bytes.len);

    const levels = try alloc.create(ops.w4a8.Levels);
    levels.* = ops.w4a8.Levels.init(codebook);
    const meta = try alloc.create(ops.w4a8.Meta);
    meta.* = .{ .s_rel = sv.bytes, .levels = levels, .group_size = @intCast(group_size) };

    var w = Weight.init(view.bytes, .w4a8, rows, cols);
    w.row_scale = s_channel;
    w.convrot = ops.convrot.group_size;
    w.w4a8 = meta;
    return w;
}

// --- tests -----------------------------------------------------------------

const test_gate = @import("../test_gate.zig");
const real_layers_json = @embedFile("assets/nvfp4_real_layers.json");

fn fnv1a64(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| h = (h ^ b) *% 0x100000001b3;
    return h;
}

test "an NVFP4 layer of each real checkpoint decodes to comfy_kitchen's weight" {
    // The synthetic tier in `ops/nvfp4.zig` pins the arithmetic and the unswizzle; this
    // pins the CONTAINER half, which is what this file adds: the `[rows, cols/2]` shape
    // doubling, `weight_scale` read as fp8 rather than u8, the scalar `weight_scale_2`,
    // and detection by `_scale_2` on a checkpoint (Z-Image's) that has no `comfy_quant`.
    //
    // Goes through `nvfp4()` on ONE tensor per family rather than a whole model load:
    // three multi-GB checkpoints in a Debug test binary is neither affordable nor needed
    // to check a container convention.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    try test_gate.requireIntegration();

    const Layer = struct {
        family: []const u8,
        checkpoint: []const u8,
        layer: []const u8,
        rows: usize,
        cols: usize,
        global_scale: f32,
        packed_fnv1a64: u64,
        scale_fnv1a64: u64,
        expect_f32_fnv1a64: u64,
        expect_head: []const f32,
        expect_tail: []const f32,
    };
    var parsed = try std.json.parseFromSlice(
        struct { layers: []const Layer },
        gpa,
        real_layers_json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var checked: usize = 0;
    for (parsed.value.layers) |L| {
        // The repo reaches checkpoints through the `models/` symlinks; only some
        // directories have one, so a family whose file is not reachable skips rather
        // than failing. `checked` below keeps that from passing green as a whole.
        var pbuf: [512]u8 = undefined;
        const path = blk: {
            // The repo's `models/` symlink set, one per checkpoint directory. `models/`
            // is gitignored, so these are a local convenience, a family without one
            // skips rather than failing.
            for ([_][]const u8{ "models/diffusion_model/", "models/anima/", "models/zit/", "models/" }) |root| {
                const p = std.fmt.bufPrint(&pbuf, "{s}{s}", .{ root, L.checkpoint }) catch continue;
                std.Io.Dir.cwd().access(io, p, .{}) catch continue;
                break :blk p;
            }
            std.debug.print("nvfp4: {s} not reachable under models/, skipping {s}\n", .{ L.checkpoint, L.family });
            break :blk null;
        } orelse continue;

        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };

        // Inputs first: a mismatch here means the checkpoint changed, which is a
        // different fact from "the decode is wrong".
        // The fixture records the LAYER base; the tensors hang off `<layer>.weight`.
        var nbuf: [256]u8 = undefined;
        var sbuf2: [256]u8 = undefined;
        const wn = try std.fmt.bufPrint(&nbuf, "{s}.weight", .{L.layer});
        const sn = try std.fmt.bufPrint(&sbuf2, "{s}_scale", .{wn});
        try std.testing.expectEqual(L.packed_fnv1a64, fnv1a64((try store.require(wn)).bytes));
        try std.testing.expectEqual(L.scale_fnv1a64, fnv1a64((try store.require(sn)).bytes));

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const alloc = arena.allocator();
        const w = (try nvfp4(alloc, store, wn, L.rows, L.cols)) orelse {
            std.debug.print("nvfp4: {s} not recognized as NVFP4\n", .{wn});
            return error.TestUnexpectedResult;
        };
        errdefer std.debug.print("family {s}, layer {s}\n", .{ L.family, L.layer });
        try std.testing.expect(w.dtype == .nvfp4);
        try std.testing.expectEqual(L.rows * L.cols / 2, w.bytes.len);
        // The two things this format does NOT have, asserted rather than assumed.
        try std.testing.expect(w.row_scale == null);
        try std.testing.expectEqual(@as(u32, 0), w.convrot);

        const got = try alloc.alloc(f32, L.rows * L.cols);
        ops.nvfp4.decode(got, w.bytes, w.nvfp4.?.*, L.rows, L.cols);
        for (L.expect_head, 0..) |want, i| try std.testing.expectEqual(want, got[i]);
        for (L.expect_tail, 0..) |want, i|
            try std.testing.expectEqual(want, got[got.len - L.expect_tail.len + i]);
        try std.testing.expectEqual(L.expect_f32_fnv1a64, fnv1a64(std.mem.sliceAsBytes(got)));
        checked += 1;
    }
    if (checked == 0) return error.SkipZigTest;
}

test "nvfp4 returns null for a layer that is not nvfp4, and errors on a broken one" {
    const gpa = std.testing.allocator;
    const safetensors = @import("tp_core").safetensors;

    // A store with one plain bf16 weight: no `_scale_2`, so this is not an nvfp4 layer
    // and the helper must decline rather than guess.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    const hdr =
        \\{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}
    ;
    try buf.appendSlice(gpa, &std.mem.toBytes(@as(u64, hdr.len)));
    try buf.appendSlice(gpa, hdr);
    try buf.appendSlice(gpa, &[_]u8{0} ** 8);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, buf.items);
    defer st.deinit();
    try std.testing.expectEqual(@as(?Weight, null), try nvfp4(gpa, .{ .safetensors = &st }, "w", 2, 2));
}

/// A one-layer store: `w.weight` I8 [rows, cols], its `weight_scale` (per-row when
/// `scales.len == rows`, per tensor when 1) and an optional `comfy_quant` blob.
///
/// The bytes are built by hand rather than through a writer so the header is exactly
/// what a ComfyUI file's is, which is the thing under test.
fn int8Store(
    gpa: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    rows: usize,
    cols: usize,
    scales: []const f32,
    conf: ?[]const u8,
) !void {
    const wbytes = rows * cols;
    const sbytes = scales.len * 4;
    var hdr: std.Io.Writer.Allocating = .init(gpa);
    defer hdr.deinit();
    const w = &hdr.writer;
    try w.print(
        \\{{"w.weight":{{"dtype":"I8","shape":[{d},{d}],"data_offsets":[0,{d}]}},
    , .{ rows, cols, wbytes });
    // A per-tensor scale is a rank-0 tensor, which is exactly what distinguishes the
    // two variants on disk.
    if (scales.len == 1) {
        try w.print(
            \\"w.weight_scale":{{"dtype":"F32","shape":[],"data_offsets":[{d},{d}]}}
        , .{ wbytes, wbytes + sbytes });
    } else {
        try w.print(
            \\"w.weight_scale":{{"dtype":"F32","shape":[{d},1],"data_offsets":[{d},{d}]}}
        , .{ scales.len, wbytes, wbytes + sbytes });
    }
    if (conf) |c| try w.print(
        \\,"w.comfy_quant":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}}
    , .{ c.len, wbytes + sbytes, wbytes + sbytes + c.len });
    try w.writeAll("}");

    const hb = hdr.written();
    try buf.appendSlice(gpa, &std.mem.toBytes(@as(u64, hb.len)));
    try buf.appendSlice(gpa, hb);
    try buf.appendNTimes(gpa, 0, wbytes);
    try buf.appendSlice(gpa, std.mem.sliceAsBytes(scales));
    if (conf) |c| try buf.appendSlice(gpa, c);
}

test "int8Scale reads both int8_tensorwise variants" {
    const gpa = std.testing.allocator;
    const safetensors = @import("tp_core").safetensors;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Rotated: a scale per output row, and the rotation group reported so the
    // activation prep matches the weight.
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try int8Store(gpa, &buf, 4, 256, &.{ 1, 2, 3, 4 },
            \\{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256, "per_row": true}
        );
        var st = try safetensors.SafeTensors.initFromSlice(gpa, buf.items);
        defer st.deinit();
        const m = try int8Scale(alloc, .{ .safetensors = &st }, "w.weight", 4, 256);
        try std.testing.expectEqual(@as(u32, ops.convrot.group_size), m.convrot);
        try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, m.row_scale);
    }

    // Unrotated: one scale for the whole tensor, broadcast per row so every consumer
    // reads the same `row_scale` and only `convrot` tells the two apart. `cols` is not
    // a multiple of 256 here, which the rotated variant would refuse.
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try int8Store(gpa, &buf, 3, 64, &.{0.5},
            \\{"format": "int8_tensorwise"}
        );
        var st = try safetensors.SafeTensors.initFromSlice(gpa, buf.items);
        defer st.deinit();
        const m = try int8Scale(alloc, .{ .safetensors = &st }, "w.weight", 3, 64);
        try std.testing.expectEqual(@as(u32, 0), m.convrot);
        try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0.5 }, m.row_scale);
    }

    // No `comfy_quant` at all: the scale's own shape answers, because ComfyUI refuses
    // convrot unless the scale is per channel. Both directions.
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try int8Store(gpa, &buf, 4, 256, &.{ 1, 1, 1, 1 }, null);
        var st = try safetensors.SafeTensors.initFromSlice(gpa, buf.items);
        defer st.deinit();
        const m = try int8Scale(alloc, .{ .safetensors = &st }, "w.weight", 4, 256);
        try std.testing.expectEqual(@as(u32, ops.convrot.group_size), m.convrot);
    }
    {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try int8Store(gpa, &buf, 4, 256, &.{1}, null);
        var st = try safetensors.SafeTensors.initFromSlice(gpa, buf.items);
        defer st.deinit();
        const m = try int8Scale(alloc, .{ .safetensors = &st }, "w.weight", 4, 256);
        try std.testing.expectEqual(@as(u32, 0), m.convrot);
    }
}

test "int8Rotation refuses metadata it cannot honour" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    const Case = struct {
        scale_elems: usize,
        cols: usize,
        conf: ?[]const u8,
        want: anyerror,
    };
    const rows = 4;
    // Each of these is a file that would otherwise decode to finite, plausible, wrong
    // numbers rather than fail.
    for ([_]Case{
        // convrot on a per-tensor scale: ComfyUI's quantizer cannot emit this, so
        // which of the two the renderer meant is unknowable.
        .{ .scale_elems = 1, .cols = 256, .conf =
            \\{"format": "int8_tensorwise", "convrot": true}
        , .want = error.ShapeMismatch },
        // A rotation group we have no Hadamard for.
        .{ .scale_elems = rows, .cols = 256, .conf =
            \\{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 128}
        , .want = error.UnsupportedCheckpoint },
        // Rotated, but the columns do not tile the group.
        .{ .scale_elems = rows, .cols = 192, .conf =
            \\{"format": "int8_tensorwise", "convrot": true}
        , .want = error.ShapeMismatch },
        // A scale that is neither per row nor per tensor.
        .{ .scale_elems = 2, .cols = 256, .conf = null, .want = error.ShapeMismatch },
        // A format whose bytes this reader would misread (its own reader runs first).
        .{ .scale_elems = rows, .cols = 256, .conf =
            \\{"format": "asym_w4a8_int8"}
        , .want = error.UnsupportedCheckpoint },
        .{ .scale_elems = rows, .cols = 256, .conf = "not json", .want = error.UnsupportedCheckpoint },
    }) |c| {
        errdefer std.debug.print("case conf={?s} scale_elems={d} cols={d}\n", .{ c.conf, c.scale_elems, c.cols });
        try std.testing.expectError(c.want, int8Rotation(alloc, c.scale_elems, rows, c.cols, c.conf));
    }

    // And the unrotated variant is not merely tolerated: a scalar scale over columns the
    // rotation could never tile is exactly what this checkpoint class looks like.
    try std.testing.expectEqual(@as(u32, 0), try int8Rotation(alloc, 1, rows, 64, null));

    // `convrot_w4a4` is always rotated and so writes no `convrot` key; reading its
    // absence as false would unrotate every int4 checkpoint.
    try std.testing.expectEqual(@as(u32, ops.convrot.group_size), try int8Rotation(alloc, rows, rows, 256,
        \\{"format": "convrot_w4a4", "convrot_groupsize": 256}
    ));

    // ComfyUI's loader reads both keys at the top level OR under `params`, in that
    // order, so a file written the nested way must not read as unrotated.
    try std.testing.expectEqual(@as(u32, ops.convrot.group_size), try int8Rotation(alloc, rows, rows, 256,
        \\{"format": "int8_tensorwise", "params": {"convrot": true, "convrot_groupsize": 256}}
    ));
    try std.testing.expectError(error.UnsupportedCheckpoint, int8Rotation(alloc, rows, rows, 256,
        \\{"format": "int8_tensorwise", "params": {"convrot": true, "convrot_groupsize": 128}}
    ));
    // An explicit false beats the format's own default and the scale's shape.
    try std.testing.expectEqual(@as(u32, 0), try int8Rotation(alloc, rows, rows, 256,
        \\{"format": "int8_tensorwise", "convrot": false}
    ));

    // A format spelling this does not know is not an error: a converter may write its
    // own, and the scale's shape still answers. Only the formats with a reader of their
    // own are refused, because reaching this arm with one means the file disagrees with
    // itself.
    try std.testing.expectEqual(@as(u32, ops.convrot.group_size), try int8Rotation(alloc, rows, rows, 256,
        \\{"format": "some_converters_private_name"}
    ));
    try std.testing.expectEqual(@as(u32, 0), try int8Rotation(alloc, 1, rows, 256,
        \\{"format": "some_converters_private_name"}
    ));
}
