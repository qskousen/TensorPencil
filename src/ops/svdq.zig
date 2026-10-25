//! Nunchaku's SVDQuant W4A4 weight format (`SVDQW4A4Linear`), precision "int4" or
//! "nvfp4" (Nunchaku's fp4 checkpoints).
//!
//! A layer `X` ships `X.qweight` (I8 [rows, cols/2]), `X.wscales`, `X.smooth_factor`
//! (BF16 [cols]), `X.proj_down` (BF16 [cols, rank]) and `X.proj_up` (BF16 [rows, rank]);
//! fp4 adds `X.wcscales` (BF16 [rows]) and may carry a scalar `X.wtscale` (1 when absent,
//! as Nunchaku's loader has it). Every tensor is in Nunchaku's MMA packing, not row-major,
//! and the header shapes do not say so. The linear is
//!
//!     y = (x / smooth) @ (value(code) * scale * row_mul)^T  +  (x @ down^T) @ up^T
//!
//! with the low-rank branch on the RAW x. int4: `value` is the signed nibble, `scale` a
//! bf16 per 64 inputs, `row_mul` 1. fp4: `value` is E2M1, `scale` an fp8-e4m3 per 16
//! inputs in Nunchaku's micro-scale packing, `row_mul = wcscales * wtscale`, which
//! multiplies the 4-bit product only. Nunchaku quantizes `x / smooth` to 4 bits per group;
//! the CPU here runs it unquantized and a device as int8 per row against `decodeI8`.
//!
//! Codes and scales are read in place (`wordIndex`, `scaleIndex`, `microIndex`); only
//! the small sidecars are unpacked at load (`Meta`). Rows come in tiles of 128 and each
//! tile is contiguous in every packed tensor, so a row view at a multiple of 128 is a
//! slice. `tools/gen_svdq_fixtures.py` runs Nunchaku's own packer for the tests below.

const std = @import("std");
const dtypes = @import("tp_core").dtype;
const w4a8 = @import("w4a8.zig");

pub const tile_rows = 128;

pub const Codes = enum {
    int4,
    fp4,

    pub fn groupSize(c: Codes) usize {
        return switch (c) {
            .int4 => 64,
            .fp4 => 16,
        };
    }

    /// Bytes per stored group scale: bf16 or fp8.
    pub fn scaleBytes(c: Codes) usize {
        return switch (c) {
            .int4 => 2,
            .fp4 => 1,
        };
    }

    /// The largest code magnitude, which sizes the int8 grid so no code clamps.
    pub fn maxValue(c: Codes) f32 {
        return switch (c) {
            .int4 => 8,
            .fp4 => 6,
        };
    }
};

/// E2M1 by bit pattern: bit 3 the sign.
pub const e2m1 = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

/// The sidecars a packed weight needs, carried on `ops.matmul.Weight.svdq`. The codes
/// are `Weight.bytes` and the device int8 GEMM's per-row scale is `Weight.row_scale`.
pub const Meta = struct {
    codes: Codes,
    /// The packed `wscales` of these rows: bf16 (int4) or fp8 (fp4).
    scales: []const u8,
    /// fp4's `wcscales * wtscale` of these rows, unpacked; null for int4.
    row_mul: ?[]const f32 = null,
    /// `1 / smooth_factor`, unpacked `[cols]`; a fused tensor's row views share it.
    inv_smooth: []const f32,
    /// `row_mul / row_scale`, what `decodeI8` multiplies by.
    i8_mul: []const f32,
    /// Unpacked bf16 `[rank][cols]`, shared by row views like `inv_smooth`.
    down: []const u8,
    /// Unpacked bf16 `[rows][rank]`, these rows only.
    up: []const u8,
    rank: u32,

    /// The view of rows `[off, off + n)`, `off` and `n` multiples of `tile_rows`.
    pub fn rows(m: Meta, cols: usize, off: usize, n: usize) Meta {
        std.debug.assert(off % tile_rows == 0 and n % tile_rows == 0);
        var v = m;
        const row_bytes = cols / m.codes.groupSize() * m.codes.scaleBytes();
        v.scales = m.scales[off * row_bytes ..][0 .. n * row_bytes];
        if (m.row_mul) |r| v.row_mul = r[off..][0..n];
        v.i8_mul = m.i8_mul[off..][0..n];
        v.up = m.up[off * m.rank * 2 ..][0 .. n * m.rank * 2];
        return v;
    }

    /// Row `n`'s scale for column `k`, before `row_mul`.
    pub inline fn groupScale(m: Meta, cols: usize, n: usize, k: usize) f32 {
        return switch (m.codes) {
            .int4 => blk: {
                const i = scaleIndex(cols / 64, n, k / 64);
                break :blk dtypes.bf16ToF32(std.mem.readInt(u16, m.scales[i * 2 ..][0..2], .little));
            },
            .fp4 => dtypes.f8e4m3ToF32(m.scales[microIndex(cols / 16, n, k / 16)]),
        };
    }

    pub inline fn rowMul(m: Meta, n: usize) f32 {
        return if (m.row_mul) |r| r[n] else 1;
    }
};

pub fn validate(codes: Codes, rows: usize, cols: usize, rank: usize, code_bytes: usize, scale_bytes: usize) error{ShapeMismatch}!void {
    if (rows % tile_rows != 0 or cols % 128 != 0 or rank % 16 != 0 or code_bytes != rows * cols / 2 or
        scale_bytes != rows * (cols / codes.groupSize()) * codes.scaleBytes())
        return error.ShapeMismatch;
}

/// The u32 holding codes `(n, k .. k + 8)`, `k` a multiple of 8; code `k + i` is nibble
/// `i`, low first. The same order for both code kinds.
pub inline fn wordIndex(cols: usize, n: usize, k: usize) usize {
    const nn = n % tile_rows;
    const kk = k % 64;
    const tile = (n / tile_rows) * (cols / 64) + k / 64;
    return ((((tile * 8 + nn / 16) * 8 + nn % 8) * 4 + (kk % 32) / 8) * 2 + (nn % 16) / 8) * 2 + kk / 32;
}

/// The element of a packed scale `[groups]` per row (or a packed vector, `groups` 1)
/// holding row `n`'s group `g`: int4's bf16 scales, `smooth_factor`, `wcscales`.
pub inline fn scaleIndex(groups: usize, n: usize, g: usize) usize {
    const nn = n % tile_rows;
    return (((((n / tile_rows) * groups + g) * 8 + nn / 16) * 4 + (nn % 8) / 2) * 2 + (nn % 16) / 8) * 2 + nn % 2;
}

/// The byte of fp4's packed fp8 scales (`pack_micro_scale`, 16 inputs a group) holding
/// row `n`'s group `g`.
pub inline fn microIndex(groups: usize, n: usize, g: usize) usize {
    const nn = n % tile_rows;
    return (((((n / tile_rows) * (groups / 4) + g / 4) * 8 + nn % 8) * 4 + (nn % 32) / 8) * 4 + nn / 32) * 4 + g % 4;
}

pub inline fn word(codes: []const u8, cols: usize, n: usize, k: usize) u32 {
    return std.mem.readInt(u32, codes[wordIndex(cols, n, k) * 4 ..][0..4], .little);
}

/// Code `i` of a word as its value: the signed nibble, or E2M1.
pub inline fn value(codes: Codes, w: u32, i: usize) f32 {
    const v: u32 = (w >> @intCast(4 * i)) & 0xF;
    return switch (codes) {
        .int4 => @floatFromInt(@as(i32, @intCast(v)) - @as(i32, if (v >= 8) 16 else 0)),
        .fp4 => e2m1[v],
    };
}

/// A packed bf16 vector (`smooth_factor`, `wcscales`) as plain f32.
pub fn unpackVector(dst: []f32, src: []const u8) void {
    std.debug.assert(src.len == dst.len * 2 and dst.len % tile_rows == 0);
    for (dst, 0..) |*v, k| v.* = dtypes.bf16ToF32(std.mem.readInt(u16, src[scaleIndex(1, k, 0) * 2 ..][0..2], .little));
}

/// A packed low-rank factor as plain bf16: `down` `[cols, rank]` packed to `[rank][cols]`,
/// `up` `[rows, rank]` packed to `[rows][rank]`. `n` is cols or rows.
pub fn unpackLowrank(dst: []u8, src: []const u8, n: usize, rank: usize, down: bool) void {
    std.debug.assert(src.len == n * rank * 2 and dst.len == src.len);
    const np = n / 16;
    const rp = rank / 16;
    var s: usize = 0;
    for (0..np) |a| for (0..rp) |b| for (0..8) |nl| for (0..4) |kl| for (0..2) |npk| for (0..2) |kpk| for (0..2) |rk| {
        const in_n = npk * 8 + nl;
        const in_k = (kpk * 4 + kl) * 2 + rk;
        const d = if (down) (b * 16 + in_n) * n + a * 16 + in_k else (a * 16 + in_n) * rank + b * 16 + in_k;
        @memcpy(dst[d * 2 ..][0..2], src[s * 2 ..][0..2]);
        s += 1;
    };
}

pub const Loaded = struct { meta: *Meta, row_scale: []f32 };

/// The packed tensors of one layer, as the checkpoint stores them.
pub const Packed = struct {
    codes: Codes,
    scales: []const u8,
    smooth: []const u8,
    down: []const u8,
    up: []const u8,
    /// fp4 only.
    wcscales: ?[]const u8 = null,
    wtscale: f32 = 1,
};

/// A layer's `Meta` and the device int8 GEMM's per-row scale from its packed sidecars,
/// everything in `alloc` (the model arena). The per-row scale is `max(value) *
/// max(scale) * row_mul / 127`, so no code clamps on the int8 grid.
pub fn load(alloc: std.mem.Allocator, p: Packed, rows: usize, cols: usize, rank: usize) !Loaded {
    if (p.smooth.len != cols * 2 or p.down.len != cols * rank * 2 or p.up.len != rows * rank * 2) return error.ShapeMismatch;
    if ((p.codes == .fp4) != (p.wcscales != null)) return error.ShapeMismatch;
    const inv_smooth = try alloc.alloc(f32, cols);
    unpackVector(inv_smooth, p.smooth);
    for (inv_smooth) |*v| v.* = 1.0 / v.*;
    const d = try alloc.alloc(u8, p.down.len);
    unpackLowrank(d, p.down, cols, rank, true);
    const u = try alloc.alloc(u8, p.up.len);
    unpackLowrank(u, p.up, rows, rank, false);
    var row_mul: ?[]f32 = null;
    if (p.wcscales) |wc| {
        if (wc.len != rows * 2) return error.ShapeMismatch;
        const r = try alloc.alloc(f32, rows);
        unpackVector(r, wc);
        for (r) |*v| v.* *= p.wtscale;
        row_mul = r;
    }
    const meta = try alloc.create(Meta);
    meta.* = .{ .codes = p.codes, .scales = p.scales, .row_mul = row_mul, .inv_smooth = inv_smooth, .i8_mul = &.{}, .down = d, .up = u, .rank = @intCast(rank) };
    const rs = try alloc.alloc(f32, rows);
    const mul = try alloc.alloc(f32, rows);
    const gs = p.codes.groupSize();
    for (rs, mul, 0..) |*r, *iv, n| {
        var mx: f32 = 0;
        var k: usize = 0;
        while (k < cols) : (k += gs) mx = @max(mx, @abs(meta.groupScale(cols, n, k)));
        const rm = meta.rowMul(n);
        const full = mx * @abs(rm) * p.codes.maxValue() / 127.0;
        r.* = if (full > 0) full else 1;
        iv.* = rm / r.*;
    }
    meta.i8_mul = mul;
    return .{ .meta = meta, .row_scale = rs };
}

/// One code as the int8 the device GEMM runs, ties to even like every int8 decode here:
/// `value * scale * i8_mul`, in that order.
pub inline fn decodeI8(v: f32, scale: f32, i8_mul: f32) i8 {
    const p = v * scale * i8_mul;
    return @intFromFloat(std.math.clamp(w4a8.roundTiesEven(p), -127.0, 127.0));
}

/// `dst[i] = value * scale * row_mul * inv_smooth` for row `n`, columns `[k0, k0 +
/// dst.len)`: the exact weight before the low-rank branch, against the unsmoothed
/// activation. `k0` and the length are multiples of 8.
pub fn decodeSliceF32(dst: []f32, codes: []const u8, meta: Meta, cols: usize, n: usize, k0: usize) void {
    std.debug.assert(k0 % 8 == 0 and dst.len % 8 == 0);
    const rm = meta.rowMul(n);
    var k: usize = 0;
    while (k < dst.len) : (k += 8) {
        const c = k0 + k;
        const w = word(codes, cols, n, c);
        // One scale per 8 columns: both group sizes are multiples of 8.
        const s = meta.groupScale(cols, n, c) * rm;
        inline for (0..8) |i| dst[k + i] = value(meta.codes, w, i) * s * meta.inv_smooth[c + i];
    }
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const safetensors = @import("tp_core").safetensors;

const Fixture = struct {
    st: safetensors.SafeTensors,
    /// `""` for the int4 layer, `"fp4."` for the fp4 one.
    pre: []const u8,
    codes: Codes,
    rows: usize,
    cols: usize,
    rank: usize,

    pub fn open(codes: Codes) !Fixture {
        const st = try safetensors.SafeTensors.initFromSlice(testing.allocator, @embedFile("assets/svdq_fixtures.safetensors"));
        var f: Fixture = .{ .st = st, .pre = if (codes == .fp4) "fp4." else "", .codes = codes, .rows = 0, .cols = 0, .rank = 0 };
        const q = f.view("qweight").info.shape.slice();
        f.rows = q[0];
        f.cols = q[1] * 2;
        f.rank = f.view("proj_up").info.shape.slice()[1];
        return f;
    }

    fn view(f: *const Fixture, name: []const u8) safetensors.TensorView {
        var buf: [64]u8 = undefined;
        return f.st.get(std.mem.print(&buf, "{s}{s}", .{ f.pre, name }) catch unreachable).?;
    }

    pub fn bytes(f: *const Fixture, name: []const u8) []const u8 {
        return f.view(name).bytes;
    }

    pub fn f32s(f: *const Fixture, name: []const u8) []align(1) const f32 {
        return std.mem.bytesAsSlice(f32, f.bytes(name));
    }

    fn meta(f: *const Fixture, alloc: std.mem.Allocator) !Meta {
        return (try f.loaded(alloc)).meta.*;
    }

    pub fn loaded(f: *const Fixture, alloc: std.mem.Allocator) !Loaded {
        return load(alloc, .{
            .codes = f.codes,
            .scales = f.bytes("wscales"),
            .smooth = f.bytes("smooth_factor"),
            .down = f.bytes("proj_down"),
            .up = f.bytes("proj_up"),
            .wcscales = if (f.codes == .fp4) f.bytes("wcscales") else null,
            .wtscale = if (f.codes == .fp4) f.f32s("wtscale")[0] else 1,
        }, f.rows, f.cols, f.rank);
    }
};

/// The fixture, for `matmul`'s test of the whole linear.
pub const TestFixture = Fixture;

test "svdq: codes, scales and the smoothing vector unpack to what Nunchaku's packer packed" {
    for ([_]Codes{ .int4, .fp4 }) |kind| {
        var f = try Fixture.open(kind);
        defer f.st.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const m = try f.meta(arena.allocator());
        try validate(kind, f.rows, f.cols, f.rank, f.bytes("qweight").len, f.bytes("wscales").len);
        // int4 stores the signed code, fp4 its E2M1 bit pattern.
        const want_codes = std.mem.bytesAsSlice(i8, f.bytes("codes"));
        const groups = f.cols / kind.groupSize();
        for (0..f.rows) |n| {
            var k: usize = 0;
            while (k < f.cols) : (k += 8) {
                const w = word(f.bytes("qweight"), f.cols, n, k);
                for (0..8) |i| {
                    const c = want_codes[n * f.cols + k + i];
                    const want: f32 = if (kind == .fp4) e2m1[@as(u8, @bitCast(c))] else @floatFromInt(c);
                    errdefer std.debug.print("{t} code ({d}, {d})\n", .{ kind, n, k + i });
                    try testing.expectEqual(want, value(kind, w, i));
                }
            }
            for (0..groups) |g| {
                errdefer std.debug.print("{t} scale ({d}, {d})\n", .{ kind, n, g });
                try testing.expectEqual(f.f32s("scales")[n * groups + g], m.groupScale(f.cols, n, g * kind.groupSize()));
            }
        }
        for (m.inv_smooth, f.f32s("smooth")) |got, want| try testing.expectEqual(1.0 / want, got);
        if (kind == .fp4) for (m.row_mul.?, f.f32s("row_mul")) |got, want| try testing.expectEqual(want, got);
    }
}

test "svdq: the low-rank factors unpack to plain [rank][cols] and [rows][rank]" {
    for ([_]Codes{ .int4, .fp4 }) |kind| {
        var f = try Fixture.open(kind);
        defer f.st.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const m = try f.meta(arena.allocator());
        try testing.expectEqualSlices(u8, f.bytes("down"), m.down);
        try testing.expectEqualSlices(u8, f.bytes("up"), m.up);
    }
}

test "svdq: decoded rows are the effective weight, and a second-tile row view decodes the same" {
    for ([_]Codes{ .int4, .fp4 }) |kind| {
        var f = try Fixture.open(kind);
        defer f.st.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const m = try f.meta(arena.allocator());
        const weff = f.f32s("weff");
        var row: [256]f32 = undefined;
        // The view starts at the second tile; its codes are the second half of the bytes.
        const view = m.rows(f.cols, tile_rows, f.rows - tile_rows);
        const view_codes = f.bytes("qweight")[tile_rows * f.cols / 2 ..];
        for (0..f.rows) |n| {
            decodeSliceF32(row[0..f.cols], f.bytes("qweight"), m, f.cols, n, 0);
            for (row[0..f.cols], weff[n * f.cols ..][0..f.cols], 0..) |got, want, k| {
                errdefer std.debug.print("{t} weff ({d}, {d}): {d} vs {d}\n", .{ kind, n, k, got, want });
                try testing.expectApproxEqRel(want, got, 1e-6);
            }
            if (n < tile_rows) continue;
            var vrow: [256]f32 = undefined;
            decodeSliceF32(vrow[0..f.cols], view_codes, view, f.cols, n - tile_rows, 0);
            try testing.expectEqualSlices(f32, row[0..f.cols], vrow[0..f.cols]);
        }
    }
}

test "svdq: the int8 decode stays on grid for every code and never clamps" {
    for ([_]Codes{ .int4, .fp4 }) |kind| {
        var f = try Fixture.open(kind);
        defer f.st.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const l = try f.loaded(arena.allocator());
        const m = l.meta.*;
        var worst: f32 = 0;
        for (0..f.rows) |n| {
            const rs = l.row_scale[n];
            var k: usize = 0;
            while (k < f.cols) : (k += 8) {
                const w = word(f.bytes("qweight"), f.cols, n, k);
                const s = m.groupScale(f.cols, n, k);
                for (0..8) |i| {
                    const v = value(kind, w, i);
                    const q = decodeI8(v, s, m.i8_mul[n]);
                    const exact = v * s * m.rowMul(n);
                    errdefer std.debug.print("{t} ({d}, {d}): value {d} is {d} steps\n", .{ kind, n, k + i, v, exact / rs });
                    try testing.expect(@abs(exact / rs) <= 127.0 * (1 + 1e-5));
                    worst = @max(worst, @abs(@as(f32, @floatFromInt(q)) * rs - exact) / rs);
                }
            }
        }
        errdefer std.debug.print("{t}: worst int8 rounding {d} steps\n", .{ kind, worst });
        try testing.expect(worst <= 0.5 + 1e-4);
    }
}
