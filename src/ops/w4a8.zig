//! ComfyUI's `asym_w4a8_int8` weight format: a 4-bit weight that DECODES to the
//! int8-convrot weight this engine already runs.
//!
//! A layer ships five tensors: `weight` (I8 [N, K/2], two 4-bit codebook indices per
//! byte), `weight_s_rel` (F8_E4M3 [N, K/group_size], the per-group relative scale),
//! `weight_s_channel` (F32 [N], the per-output-row scale), `weight_codebook` (F32 [16],
//! optional) and `comfy_quant` (U8 JSON carrying `group_size` and `convrot_groupsize`).
//! The decode is:
//!
//!     q    = nibble(packed)                        // 0..15, even col = LOW nibble
//!     lvl  = codebook[q]  (or q - 8 with no codebook)
//!     int8 = rint(clamp(lvl * s_rel[row, group], -127, 127))
//!
//! After that it is `ops.matmul`'s existing `.i8` path verbatim, with `row_scale` =
//! `s_channel` and `convrot` = `convrot_groupsize`. That is why this file is small: the
//! novelty is entirely in the level decode, and ComfyUI's own backends do the same thing,
//! so a correct decode reproduces its result bit for bit rather than approximately.
//!
//! The rounding is ties-to-EVEN. torch's `.round()` and the Triton kernel's
//! `libdevice.rint` both are; Zig's `@round` is half-away-from-zero. An exact tie needs
//! `codebook[i] * s_rel` to land on a .5 boundary, which is rare but not negligible
//! because s_rel is fp8, so every checkpoint would carry a handful of off-by-one weights.
//! `roundTiesEven` and the `ties_to_even` fixture case pin it.
//!
//! The codebook is NOT uniform, so this is not `.i4` with a per-group scale. The levels
//! are Lloyd-Max-optimal for a Gaussian (ConvRot makes the rotated groups Gaussian),
//! spaced 0.186 at the tails and 0.103 in the middle. Reading the nibbles as signed int4
//! times a scale, which is what the int4-convrot path does, is finite, plausible and
//! wrong.
//!
//! The decode goes through a 4 KiB lookup table (`Levels`) rather than per-element float
//! arithmetic: `s_rel` is fp8, so there are only 256 possible group scales, and one
//! `[256][16] i8` table holds every value the tensor can decode to. The per-element work
//! becomes one L1 load and one store, which makes a whole-model decode bandwidth-bound.
//! It is exact by construction: the table entries use the same f32 multiply-round-clamp
//! the reference does.

const std = @import("std");
const dtypes = @import("tp_core").dtype;

/// `group_size` when a layer's `comfy_quant` does not say (ComfyUI's default).
pub const default_group_size = 16;

/// `convrot_groupsize` when a layer's `comfy_quant` does not say.
pub const default_convrot_groupsize = 256;

/// The frozen Lloyd-Max codebook `comfy_kitchen` uses unless a tensor's rotated groups
/// are heavy-tailed (`_FIXED_LUT`, gated by an excess-kurtosis probe that real models
/// never trip). All 224 layers of the krea2 W4A8 checkpoint ship exactly this.
///
/// Present for tests and tooling only, a loader must still read each layer's own
/// `weight_codebook`, because the whole point of the kurtosis gate is that a future
/// tensor may carry a fitted table instead. A fast test pins it against the fixture.
pub const fixed_lut: [16]f32 = .{
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541,  0.156985,  0.265284,  0.379533,  0.502636,  0.638953,  0.794876,  0.980671,
};

/// The 16 levels a layer without a `weight_codebook` decodes to: the uniform
/// `q - 8`. ComfyUI's `_dequant_int4_grouped_to_int8` takes this branch when the
/// tensor is absent (a layer quantized with `codebook=False`).
pub const uniform_levels: [16]f32 = blk: {
    var v: [16]f32 = undefined;
    for (&v, 0..) |*e, i| e.* = @as(f32, @floatFromInt(i)) - 8.0;
    break :blk v;
};

/// The `format` string this module handles, as written into `comfy_quant`.
pub const format_name = "asym_w4a8_int8";

/// A quantized layer's `comfy_quant` blob: a JSON object stored as U8 bytes next to
/// the weight. Every field has ComfyUI's own default, since `_load_quantized_module`
/// reads them with `layer_conf.get(..., params_conf.get(..., <default>))`.
pub const QuantConf = struct {
    format: []const u8 = "",
    group_size: usize = default_group_size,
    convrot_groupsize: usize = default_convrot_groupsize,
};

/// Parse a layer's `comfy_quant` blob. Allocations land in `alloc` (the caller's
/// arena), `format` is a slice into the parse.
pub fn parseConf(alloc: std.mem.Allocator, bytes: []const u8) !QuantConf {
    return std.json.parseFromSliceLeaky(QuantConf, alloc, bytes, .{ .ignore_unknown_fields = true });
}

/// Round half to even, matching torch's `Tensor.round` and CUDA's `rint`.
///
/// Zig's `@round` is half-away-from-zero, so it disagrees on every exact tie. Only a
/// value whose fractional part is exactly 0.5 can differ, and there `@round` returns
/// the odd neighbour precisely when ties-to-even wants the other one.
pub inline fn roundTiesEven(x: f32) f32 {
    const r = @round(x);
    if (@abs(r - x) == 0.5) {
        const half = r * 0.5;
        if (half != @trunc(half)) return r - std.math.sign(x); // r is odd
    }
    return r;
}

/// Every int8 value a tensor can decode to, indexed by `[s_rel byte][nibble]`.
///
/// 4 KiB, so it stays in L1 for the whole tensor. Built once per weight; the decode
/// itself then does no floating-point arithmetic at all.
pub const Levels = struct {
    table: [256][16]i8,

    /// `codebook` is the layer's `weight_codebook`, or `null` for the uniform
    /// `q - 8` levels.
    pub fn init(codebook: ?*const [16]f32) Levels {
        const cb: *const [16]f32 = codebook orelse &uniform_levels;
        var lv: Levels = undefined;
        for (0..256) |b| {
            const s = dtypes.f8e4m3ToF32(@intCast(b));
            for (0..16) |i| {
                // The f32 multiply, then ties-to-even, then the clamp, in the
                // reference's order. A NaN group scale is a corrupt checkpoint, the
                // reference's own `.to(torch.int8)` is undefined there and Zig's
                // `@intFromFloat` is illegal behaviour, so it decodes to 0 rather
                // than to whatever the host happens to do.
                const p = cb[i] * s;
                const v = if (std.math.isNan(p)) 0.0 else std.math.clamp(roundTiesEven(p), -127.0, 127.0);
                lv.table[b][i] = @intFromFloat(v);
            }
        }
        return lv;
    }
};

/// The sidecars a PACKED W4A8 weight needs in order to decode, carried on
/// `ops.matmul.Weight.w4a8` so `Weight.bytes` can stay the 4-bit storage.
///
/// Keeping the packed form is the whole point: a decoded krea2 is 12.2 GB of int8
/// against 6.1 GB packed, and on a 24 GB card that difference is the format. Every
/// consumer therefore decodes on demand, the CPU GEMM per k-slice into its existing
/// dequant panel, the GPU backends per GEMM into a device scratch (which is also what
/// ComfyUI's Triton and CUDA backends do).
pub const Meta = struct {
    /// Raw fp8-e4m3 bytes of `weight_s_rel`: `[rows][cols / group_size]`.
    s_rel: []const u8,
    /// This layer's `[256][16]` int8 decode table. A pointer, not a value, so a GPU
    /// backend can key a 4 KiB device copy on its address the same way it keys weights.
    levels: *const Levels,
    group_size: u32,

    /// Group count per row.
    pub fn groups(self: Meta, cols: usize) usize {
        return cols / self.group_size;
    }
};

/// Sanity-check a layer's tensor sizes against each other.
///
/// `cols` is the LOGICAL K (twice the stored column count). Returns the group count.
pub fn validate(rows: usize, cols: usize, group_size: usize, convrot_groupsize: usize, packed_len: usize, s_rel_len: usize) !usize {
    // The reference's own constraints (`validate_w4a8_operands`): K divisible by 16,
    // by the group size and by the ConvRot group size, and a group size that is at
    // least 4 and either divides 16 or is a multiple of it, which makes every legal
    // group size even, and that is what lets a byte's two nibbles share one scale.
    if (group_size < 4 or cols % 16 != 0 or cols % group_size != 0 or
        cols % convrot_groupsize != 0 or (16 % group_size != 0 and group_size % 16 != 0))
        return error.ShapeMismatch;
    std.debug.assert(group_size % 2 == 0);
    const groups = cols / group_size;
    if (packed_len != rows * cols / 2) return error.ShapeMismatch;
    if (s_rel_len != rows * groups) return error.ShapeMismatch;
    return groups;
}

/// Decode rows `[row0, row1)` of one weight into `dst` (the whole `[rows, cols]`
/// int8 buffer, addressed absolutely so the row ranges of concurrent callers do not
/// overlap).
///
/// `packed_bytes` is `weight` as stored (`[rows, cols/2]`), `s_rel` the raw fp8 bytes
/// of `weight_s_rel` (`[rows, cols/group_size]`).
pub fn decodeRows(
    dst: []i8,
    packed_bytes: []const u8,
    s_rel: []const u8,
    lv: *const Levels,
    rows: usize,
    cols: usize,
    group_size: usize,
    row0: usize,
    row1: usize,
) void {
    std.debug.assert(dst.len == rows * cols);
    std.debug.assert(row1 <= rows);
    const groups = cols / group_size;
    const per_group = group_size / 2; // bytes; group_size is even (see validate)
    for (row0..row1) |r| {
        const src = packed_bytes[r * cols / 2 ..][0 .. cols / 2];
        const srow = s_rel[r * groups ..][0..groups];
        const out = dst[r * cols ..][0..cols];
        for (0..groups) |g| {
            const tbl = &lv.table[srow[g]];
            const in = src[g * per_group ..][0..per_group];
            const o = out[g * group_size ..][0..group_size];
            for (in, 0..) |byte, j| {
                o[2 * j] = tbl[byte & 0xF];
                o[2 * j + 1] = tbl[byte >> 4];
            }
        }
    }
}

/// Decode a whole weight, fanning the rows over `std.Thread`s.
///
/// A krea2 W4A8 checkpoint decodes 12.2 GB of int8 at load, which is bandwidth-bound
/// (see the `Levels` note) and so scales with threads. `std.Thread` rather than
/// `std.Io` because the model loaders are synchronous and take no `Io`; a spawn
/// failure just means the caller's thread does that share too, so the decode still
/// completes.
pub fn decode(
    dst: []i8,
    packed_bytes: []const u8,
    s_rel: []const u8,
    lv: *const Levels,
    rows: usize,
    cols: usize,
    group_size: usize,
) void {
    const max_threads = 16;
    const cpus = std.Thread.getCpuCount() catch 1;
    const n = @min(@min(cpus, max_threads), @max(rows / 64, 1));
    if (n <= 1) return decodeRows(dst, packed_bytes, s_rel, lv, rows, cols, group_size, 0, rows);

    const Ctx = struct {
        dst: []i8,
        packed_bytes: []const u8,
        s_rel: []const u8,
        lv: *const Levels,
        rows: usize,
        cols: usize,
        group_size: usize,
        fn run(c: *const @This(), row0: usize, row1: usize) void {
            decodeRows(c.dst, c.packed_bytes, c.s_rel, c.lv, c.rows, c.cols, c.group_size, row0, row1);
        }
    };
    const ctx = Ctx{
        .dst = dst,
        .packed_bytes = packed_bytes,
        .s_rel = s_rel,
        .lv = lv,
        .rows = rows,
        .cols = cols,
        .group_size = group_size,
    };

    var threads: [max_threads]?std.Thread = @splat(null);
    const chunk = (rows + n - 1) / n;
    // Thread i takes chunk i; the caller takes chunk 0 rather than idling, and also
    // takes over any chunk whose spawn failed.
    for (1..n) |i| {
        const lo = i * chunk;
        if (lo >= rows) break;
        const hi = @min(lo + chunk, rows);
        threads[i] = std.Thread.spawn(.{}, Ctx.run, .{ &ctx, lo, hi }) catch blk: {
            Ctx.run(&ctx, lo, hi);
            break :blk null;
        };
    }
    Ctx.run(&ctx, 0, @min(chunk, rows));
    for (threads[1..n]) |t| if (t) |th| th.join();
}

// --- tests -----------------------------------------------------------------

const fixtures_json = @embedFile("assets/w4a8_fixtures.json");

const Case = struct {
    name: []const u8,
    rows: usize,
    cols: usize,
    group_size: usize,
    convrot_groupsize: usize,
    codebook: ?[]const f32,
    packed_hex: []const u8,
    s_rel_hex: []const u8,
    s_channel: []const f32,
    expect_i8: []const i32,
    expect_deq_f32: []const f32,
};

/// Only the synthetic tier lives here; the real-checkpoint tier is
/// `src/models/assets/w4a8_real_layer.json`, next to the loader it exercises (this
/// module cannot reach `test_gate`).
const Fixtures = struct { fixed_lut: []const f32, cases: []const Case };

fn loadFixtures(gpa: std.mem.Allocator) !std.json.Parsed(Fixtures) {
    return std.json.parseFromSlice(Fixtures, gpa, fixtures_json, .{ .ignore_unknown_fields = true });
}

fn unhex(gpa: std.mem.Allocator, hex: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, hex.len / 2);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

test "W4A8 decode matches comfy_kitchen's reference on every fixture case" {
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.cases) |c| {
        const packed_bytes = try unhex(gpa, c.packed_hex);
        defer gpa.free(packed_bytes);
        const s_rel = try unhex(gpa, c.s_rel_hex);
        defer gpa.free(s_rel);

        _ = try validate(c.rows, c.cols, c.group_size, c.convrot_groupsize, packed_bytes.len, s_rel.len);

        var cb: [16]f32 = undefined;
        const cbp: ?*const [16]f32 = if (c.codebook) |v| blk: {
            @memcpy(&cb, v[0..16]);
            break :blk &cb;
        } else null;
        const lv = Levels.init(cbp);

        const dst = try gpa.alloc(i8, c.rows * c.cols);
        defer gpa.free(dst);
        decode(dst, packed_bytes, s_rel, &lv, c.rows, c.cols, c.group_size);

        for (dst, c.expect_i8, 0..) |got, want, i| {
            std.testing.expectEqual(@as(i8, @intCast(want)), got) catch |e| {
                std.debug.print("case {s}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.name, i, i / c.cols, i % c.cols, got, want,
                });
                return e;
            };
        }
    }
}

test "the threaded and single-threaded W4A8 decodes agree" {
    // Bit-identical by construction (the rows are independent), but the row split and
    // the spawn-failure fallback are not, so this pins that a chunk is neither
    // dropped nor decoded twice from the wrong offset.
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    const c = parsed.value.cases[0];
    const packed_bytes = try unhex(gpa, c.packed_hex);
    defer gpa.free(packed_bytes);
    const s_rel = try unhex(gpa, c.s_rel_hex);
    defer gpa.free(s_rel);
    var cb: [16]f32 = undefined;
    @memcpy(&cb, c.codebook.?[0..16]);
    const lv = Levels.init(&cb);

    // Tile the fixture up to enough rows that `decode` really does split.
    const reps = 200;
    const rows = c.rows * reps;
    const packed_big = try gpa.alloc(u8, rows * c.cols / 2);
    defer gpa.free(packed_big);
    const s_rel_big = try gpa.alloc(u8, rows * (c.cols / c.group_size));
    defer gpa.free(s_rel_big);
    for (0..reps) |i| {
        @memcpy(packed_big[i * packed_bytes.len ..][0..packed_bytes.len], packed_bytes);
        @memcpy(s_rel_big[i * s_rel.len ..][0..s_rel.len], s_rel);
    }

    const par = try gpa.alloc(i8, rows * c.cols);
    defer gpa.free(par);
    const seq = try gpa.alloc(i8, rows * c.cols);
    defer gpa.free(seq);
    decode(par, packed_big, s_rel_big, &lv, rows, c.cols, c.group_size);
    decodeRows(seq, packed_big, s_rel_big, &lv, rows, c.cols, c.group_size, 0, rows);
    try std.testing.expectEqualSlices(i8, seq, par);
}

test "a W4A8 weight dequantizes to the reference's f32 weight through the int8 convrot path" {
    // The point of this one is that the decode's *output contract* is the existing
    // int8-convrot Weight: feeding `s_channel` as `row_scale` and un-rotating by the
    // ConvRot group has to reproduce `dequantize_w4a8_int8_weight`, i.e. the physical
    // weight ComfyUI computes. Without this the decode could be right and the
    // hand-off still wrong.
    const gpa = std.testing.allocator;
    const convrot = @import("convrot.zig");
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.cases) |c| {
        const packed_bytes = try unhex(gpa, c.packed_hex);
        defer gpa.free(packed_bytes);
        const s_rel = try unhex(gpa, c.s_rel_hex);
        defer gpa.free(s_rel);
        var cb: [16]f32 = undefined;
        const cbp: ?*const [16]f32 = if (c.codebook) |v| blk: {
            @memcpy(&cb, v[0..16]);
            break :blk &cb;
        } else null;
        const lv = Levels.init(cbp);

        const q = try gpa.alloc(i8, c.rows * c.cols);
        defer gpa.free(q);
        decode(q, packed_bytes, s_rel, &lv, c.rows, c.cols, c.group_size);

        // int8 * per-row scale, then rotate each 256-group back to the original basis
        // exactly what `ops.matmul`'s `.i8` convrot dequant does per k-slice.
        const w = try gpa.alloc(f32, c.rows * c.cols);
        defer gpa.free(w);
        for (0..c.rows) |r| {
            const row = w[r * c.cols ..][0..c.cols];
            for (row, q[r * c.cols ..][0..c.cols]) |*d, s| d.* = @as(f32, @floatFromInt(s)) * c.s_channel[r];
            convrot.rotate(row);
        }

        var max_rel: f32 = 0;
        for (w, c.expect_deq_f32) |got, want| {
            const denom = @max(@abs(want), 1e-6);
            max_rel = @max(max_rel, @abs(got - want) / denom);
        }
        errdefer std.debug.print("case {s}: max rel {d}\n", .{ c.name, max_rel });
        // The only slack is the rotation's summation order (ours is a radix-4 FWHT,
        // torch's a matmul against the materialized Hadamard), which is why this is a
        // tolerance where the int8 comparison above is exact.
        try std.testing.expect(max_rel < 2e-6);
    }
}

test "a matmul through the PACKED W4A8 weight equals one through the materialized int8" {
    // The decode moved out of the loader and into the GEMM so the 4-bit form can stay
    // resident. That is only safe if the two forms compute the same thing, and the
    // decode is exact, so this is `expectEqualSlices` and not a tolerance: the packed
    // path must reproduce the int8 path bit for bit, or a render silently changes
    // depending on how the weight happened to be stored.
    const matmul = @import("matmul.zig");
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.cases) |c| {
        // `cols` must be a multiple of the ConvRot group for the GEMM's un-rotation.
        if (c.cols % 256 != 0) continue;
        const packed_bytes = try unhex(gpa, c.packed_hex);
        defer gpa.free(packed_bytes);
        const s_rel = try unhex(gpa, c.s_rel_hex);
        defer gpa.free(s_rel);
        var cb: [16]f32 = undefined;
        const cbp: ?*const [16]f32 = if (c.codebook) |v| blk: {
            @memcpy(&cb, v[0..16]);
            break :blk &cb;
        } else null;
        const levels = Levels.init(cbp);
        const meta: Meta = .{ .s_rel = s_rel, .levels = &levels, .group_size = @intCast(c.group_size) };

        const q = try gpa.alloc(i8, c.rows * c.cols);
        defer gpa.free(q);
        decode(q, packed_bytes, s_rel, &levels, c.rows, c.cols, c.group_size);

        var w_packed = matmul.Weight.init(packed_bytes, .w4a8, c.rows, c.cols);
        w_packed.row_scale = c.s_channel;
        w_packed.convrot = 256;
        w_packed.w4a8 = &meta;

        var w_i8 = matmul.Weight.init(std.mem.sliceAsBytes(q), .i8, c.rows, c.cols);
        w_i8.row_scale = c.s_channel;
        w_i8.convrot = 256;

        // Both GEMM regimes: below `small_m_max` takes the row-at-a-time path, at or
        // above it the packed outer-product path, and they dequantize at different
        // granularities (a whole row vs a KC k-slice).
        for ([_]usize{ 1, matmul.small_m_max, matmul.small_m_max + 7 }) |m| {
            const x = try gpa.alloc(f32, m * c.cols);
            defer gpa.free(x);
            var rnd = std.Random.DefaultPrng.init(0x4a8);
            for (x) |*v| v.* = rnd.random().float(f32) * 2 - 1;

            const y_p = try gpa.alloc(f32, m * c.rows);
            defer gpa.free(y_p);
            const y_q = try gpa.alloc(f32, m * c.rows);
            defer gpa.free(y_q);
            try matmul.matmul(io, gpa, y_p, x, m, w_packed, null);
            try matmul.matmul(io, gpa, y_q, x, m, w_i8, null);
            errdefer std.debug.print("case {s}, m {d}\n", .{ c.name, m });
            try std.testing.expectEqualSlices(f32, y_q, y_p);
        }
    }
}

test "roundTiesEven matches rint on the cases @round gets wrong" {
    const pairs = [_]struct { x: f32, want: f32 }{
        .{ .x = 0.5, .want = 0 },    .{ .x = -0.5, .want = 0 },
        .{ .x = 1.5, .want = 2 },    .{ .x = -1.5, .want = -2 },
        .{ .x = 2.5, .want = 2 },    .{ .x = -2.5, .want = -2 },
        .{ .x = 3.5, .want = 4 },    .{ .x = -3.5, .want = -4 },
        .{ .x = 126.5, .want = 126 }, .{ .x = 127.5, .want = 128 },
        // Non-ties must be untouched, including just either side of one.
        .{ .x = 0.4999999, .want = 0 }, .{ .x = 0.5000001, .want = 1 },
        .{ .x = 2.49, .want = 2 },   .{ .x = 2.51, .want = 3 },
        .{ .x = 0, .want = 0 },      .{ .x = -0.0, .want = 0 },
        .{ .x = 1e7, .want = 1e7 },
    };
    var differ: usize = 0;
    for (pairs) |p| {
        errdefer std.debug.print("roundTiesEven({d}) = {d}, want {d}\n", .{ p.x, roundTiesEven(p.x), p.want });
        try std.testing.expectEqual(p.want, roundTiesEven(p.x));
        if (@round(p.x) != p.want) differ += 1;
    }
    // Teeth: `@round` must actually disagree somewhere, else this test proves nothing.
    // It disagrees on exactly the ties whose away-from-zero neighbour is odd,
    // ±0.5, ±2.5 and 126.5 here, and agrees on ±1.5, ±3.5, 127.5, where rounding
    // away from zero already lands on the even value.
    try std.testing.expectEqual(@as(usize, 5), differ);
}

test "W4A8 validate refuses the shapes the reference refuses" {
    // A group size that neither divides 16 nor is a multiple of it, K not divisible by
    // the ConvRot group, and mis-sized sidecars. Each of these would otherwise read a
    // scale from the wrong group, finite and plausible, so it has to be an error.
    try std.testing.expectError(error.ShapeMismatch, validate(4, 256, 24, 256, 4 * 128, 4 * 10));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 128, 16, 256, 4 * 64, 4 * 8));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 256, 2, 256, 4 * 128, 4 * 128));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 256, 16, 256, 4 * 127, 4 * 16));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 256, 16, 256, 4 * 128, 4 * 15));
    try std.testing.expectEqual(@as(usize, 16), try validate(4, 256, 16, 256, 4 * 128, 4 * 16));
    try std.testing.expectEqual(@as(usize, 64), try validate(4, 256, 4, 256, 4 * 128, 4 * 64));
}

test "the frozen codebook constant matches comfy_kitchen's _FIXED_LUT" {
    // `fixed_lut` is hand-transcribed, so pin it against the generator's copy of the
    // reference's own table. A typo here would be a plausible codebook and a wrong one.
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();
    try std.testing.expectEqualSlices(f32, parsed.value.fixed_lut, &fixed_lut);
}

test "the uniform levels are the no-codebook branch's q - 8" {
    const lv = Levels.init(null);
    // s_rel = 1.0 (0x38 in e4m3) so the table is the levels themselves.
    try std.testing.expectEqual(@as(f32, 1.0), dtypes.f8e4m3ToF32(0x38));
    for (0..16) |i| {
        const want: i8 = @intCast(@as(i32, @intCast(i)) - 8);
        try std.testing.expectEqual(want, lv.table[0x38][i]);
    }
}
