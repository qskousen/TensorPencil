//! Nunchaku's SVDQuant W4A4 int4 weight format (`SVDQW4A4Linear`, precision "int4").
//!
//! A layer `X` ships `X.qweight` (I8 [rows, cols/2]), `X.wscales` (BF16 [cols/64, rows]),
//! `X.smooth_factor` (BF16 [cols]), `X.proj_down` (BF16 [cols, rank]) and `X.proj_up`
//! (BF16 [rows, rank]). Every one is in Nunchaku's MMA packing, not row-major, and the
//! header shapes do not say so. The linear is
//!
//!     y = (x / smooth) @ (code * scale)^T  +  (x @ down^T) @ up^T
//!
//! with the low-rank branch on the RAW x. Nunchaku quantizes `x / smooth` to 4 bits per
//! 64 inputs; the CPU here runs it unquantized and a device runs it as int8 per row
//! against `decodeI8`'s weight.
//!
//! Codes and scales are read in place (`wordIndex`, `scaleIndex`); only the small
//! sidecars are unpacked at load (`Meta`). Rows come in tiles of 128 and each tile is
//! contiguous in every packed tensor, so a row view at a multiple of 128 is a slice.
//! `tools/gen_svdq_fixtures.py` runs Nunchaku's own packer for the tests below.

const std = @import("std");
const dtypes = @import("tp_core").dtype;
const w4a8 = @import("w4a8.zig");

pub const group_size = 64;
pub const tile_rows = 128;

/// The sidecars a packed weight needs, carried on `ops.matmul.Weight.svdq`. The codes
/// are `Weight.bytes` and the device int8 GEMM's per-row scale is `Weight.row_scale`.
pub const Meta = struct {
    /// Packed bf16 `wscales` of these rows.
    scales: []const u8,
    /// `1 / smooth_factor`, unpacked `[cols]`; a fused tensor's row views share it.
    inv_smooth: []const f32,
    /// `1 / row_scale`, what `decodeI8` multiplies by.
    inv_row_scale: []const f32,
    /// Unpacked bf16 `[rank][cols]`, shared by row views like `inv_smooth`.
    down: []const u8,
    /// Unpacked bf16 `[rows][rank]`, these rows only.
    up: []const u8,
    rank: u32,

    /// The view of rows `[off, off + n)`, `off` and `n` multiples of `tile_rows`.
    pub fn rows(m: Meta, cols: usize, off: usize, n: usize) Meta {
        std.debug.assert(off % tile_rows == 0 and n % tile_rows == 0);
        var v = m;
        const g = cols / group_size;
        v.scales = m.scales[off * g * 2 ..][0 .. n * g * 2];
        v.inv_row_scale = m.inv_row_scale[off..][0..n];
        v.up = m.up[off * m.rank * 2 ..][0 .. n * m.rank * 2];
        return v;
    }
};

pub fn validate(rows: usize, cols: usize, rank: usize, code_bytes: usize, scale_bytes: usize) error{ShapeMismatch}!void {
    if (rows % tile_rows != 0 or cols % (2 * group_size) != 0 or rank % 16 != 0 or
        code_bytes != rows * cols / 2 or scale_bytes != rows * (cols / group_size) * 2)
        return error.ShapeMismatch;
}

/// The u32 holding codes `(n, k .. k + 8)`, `k` a multiple of 8; code `k + i` is
/// nibble `i`, low first, signed.
pub inline fn wordIndex(cols: usize, n: usize, k: usize) usize {
    const nn = n % tile_rows;
    const kk = k % group_size;
    const tile = (n / tile_rows) * (cols / group_size) + k / group_size;
    return ((((tile * 8 + nn / 16) * 8 + nn % 8) * 4 + (kk % 32) / 8) * 2 + (nn % 16) / 8) * 2 + kk / 32;
}

/// The element of a packed scale `[groups]` per row (or a packed vector, `groups` 1)
/// holding row `n`'s group `g`.
pub inline fn scaleIndex(groups: usize, n: usize, g: usize) usize {
    const nn = n % tile_rows;
    return (((((n / tile_rows) * groups + g) * 8 + nn / 16) * 4 + (nn % 8) / 2) * 2 + (nn % 16) / 8) * 2 + nn % 2;
}

pub inline fn word(codes: []const u8, cols: usize, n: usize, k: usize) u32 {
    return std.mem.readInt(u32, codes[wordIndex(cols, n, k) * 4 ..][0..4], .little);
}

pub inline fn nibble(w: u32, i: usize) i32 {
    const v: i32 = @intCast((w >> @intCast(4 * i)) & 0xF);
    return if (v >= 8) v - 16 else v;
}

pub inline fn scaleAt(scales: []const u8, cols: usize, n: usize, g: usize) f32 {
    const i = scaleIndex(cols / group_size, n, g);
    return dtypes.bf16ToF32(std.mem.readInt(u16, scales[i * 2 ..][0..2], .little));
}

/// A packed bf16 vector (`smooth_factor`) as plain f32.
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

/// The per-row scale a device's int8 decode uses, `8 * max(scale) / 127`, so that even a
/// -8 code never clamps; and its reciprocal. `rs` and `inv` are `[rows]`.
pub fn rowScales(rs: []f32, inv: []f32, scales: []const u8, cols: usize) void {
    const groups = cols / group_size;
    for (rs, inv, 0..) |*r, *iv, n| {
        var mx: f32 = 0;
        for (0..groups) |g| mx = @max(mx, @abs(scaleAt(scales, cols, n, g)));
        r.* = if (mx > 0) mx * 8.0 / 127.0 else 1.0;
        iv.* = 1.0 / r.*;
    }
}

/// A layer's `Meta` and per-row scale from its packed sidecars, everything in `alloc`
/// (the model arena). `smooth` is the packed `smooth_factor`, `down` / `up` the packed
/// `proj_down` / `proj_up`.
pub const Loaded = struct { meta: *Meta, row_scale: []f32 };

pub fn load(alloc: std.mem.Allocator, scales: []const u8, smooth: []const u8, down: []const u8, up: []const u8, rows: usize, cols: usize, rank: usize) !Loaded {
    if (smooth.len != cols * 2 or down.len != cols * rank * 2 or up.len != rows * rank * 2) return error.ShapeMismatch;
    const inv_smooth = try alloc.alloc(f32, cols);
    unpackVector(inv_smooth, smooth);
    for (inv_smooth) |*v| v.* = 1.0 / v.*;
    const d = try alloc.alloc(u8, down.len);
    unpackLowrank(d, down, cols, rank, true);
    const u = try alloc.alloc(u8, up.len);
    unpackLowrank(u, up, rows, rank, false);
    const rs = try alloc.alloc(f32, rows);
    const inv = try alloc.alloc(f32, rows);
    rowScales(rs, inv, scales, cols);
    const meta = try alloc.create(Meta);
    meta.* = .{ .scales = scales, .inv_smooth = inv_smooth, .inv_row_scale = inv, .down = d, .up = u, .rank = @intCast(rank) };
    return .{ .meta = meta, .row_scale = rs };
}

/// One code as the int8 the device GEMM runs, ties to even like every int8 decode here.
pub inline fn decodeI8(code: i32, scale: f32, inv_row_scale: f32) i8 {
    const p = @as(f32, @floatFromInt(code)) * scale * inv_row_scale;
    return @intFromFloat(std.math.clamp(w4a8.roundTiesEven(p), -127.0, 127.0));
}

/// `dst[i] = code * scale * inv_smooth` for row `n`, columns `[k0, k0 + dst.len)`: the
/// exact weight before the low-rank branch, against the unsmoothed activation. `k0` and
/// the length are multiples of 8.
pub fn decodeSliceF32(dst: []f32, codes: []const u8, meta: Meta, cols: usize, n: usize, k0: usize) void {
    std.debug.assert(k0 % 8 == 0 and dst.len % 8 == 0);
    var k: usize = 0;
    while (k < dst.len) : (k += 8) {
        const c = k0 + k;
        const w = word(codes, cols, n, c);
        const s = scaleAt(meta.scales, cols, n, c / group_size);
        inline for (0..8) |i| dst[k + i] = @as(f32, @floatFromInt(nibble(w, i))) * s * meta.inv_smooth[c + i];
    }
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const safetensors = @import("tp_core").safetensors;

const Fixture = struct {
    st: safetensors.SafeTensors,
    rows: usize,
    cols: usize,
    rank: usize,

    pub fn open() !Fixture {
        const st = try safetensors.SafeTensors.initFromSlice(testing.allocator, @embedFile("assets/svdq_fixtures.safetensors"));
        const q = st.get("qweight").?.info.shape.slice();
        return .{ .st = st, .rows = q[0], .cols = q[1] * 2, .rank = st.get("proj_up").?.info.shape.slice()[1] };
    }

    pub fn bytes(f: *const Fixture, name: []const u8) []const u8 {
        return f.st.get(name).?.bytes;
    }

    pub fn f32s(f: *const Fixture, name: []const u8) []align(1) const f32 {
        return std.mem.bytesAsSlice(f32, f.bytes(name));
    }

    fn meta(f: *const Fixture, alloc: std.mem.Allocator) !Meta {
        return (try f.loaded(alloc)).meta.*;
    }

    pub fn loaded(f: *const Fixture, alloc: std.mem.Allocator) !Loaded {
        return load(alloc, f.bytes("wscales"), f.bytes("smooth_factor"), f.bytes("proj_down"), f.bytes("proj_up"), f.rows, f.cols, f.rank);
    }
};

/// The fixture, for `matmul`'s test of the whole linear.
pub const TestFixture = Fixture;

test "svdq: codes, scales and the smoothing vector unpack to what Nunchaku's packer packed" {
    var f = try Fixture.open();
    defer f.st.deinit();
    try validate(f.rows, f.cols, f.rank, f.bytes("qweight").len, f.bytes("wscales").len);
    const want_codes = std.mem.bytesAsSlice(i8, f.bytes("codes"));
    for (0..f.rows) |n| {
        var k: usize = 0;
        while (k < f.cols) : (k += 8) {
            const w = word(f.bytes("qweight"), f.cols, n, k);
            for (0..8) |i| {
                errdefer std.debug.print("code ({d}, {d})\n", .{ n, k + i });
                try testing.expectEqual(@as(i32, want_codes[n * f.cols + k + i]), nibble(w, i));
            }
        }
        for (0..f.cols / group_size) |g| {
            errdefer std.debug.print("scale ({d}, {d})\n", .{ n, g });
            try testing.expectEqual(f.f32s("scales")[n * (f.cols / group_size) + g], scaleAt(f.bytes("wscales"), f.cols, n, g));
        }
    }
    var smooth: [256]f32 = undefined;
    unpackVector(smooth[0..f.cols], f.bytes("smooth_factor"));
    for (smooth[0..f.cols], f.f32s("smooth")) |got, want| try testing.expectEqual(want, got);
}

test "svdq: the low-rank factors unpack to plain [rank][cols] and [rows][rank]" {
    var f = try Fixture.open();
    defer f.st.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const m = try f.meta(arena.allocator());
    try testing.expectEqualSlices(u8, f.bytes("down"), m.down);
    try testing.expectEqualSlices(u8, f.bytes("up"), m.up);
}

test "svdq: decoded rows are the effective weight, and a second-tile row view decodes the same" {
    var f = try Fixture.open();
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
            errdefer std.debug.print("weff ({d}, {d}): {d} vs {d}\n", .{ n, k, got, want });
            try testing.expectApproxEqRel(want, got, 1e-6);
        }
        if (n < tile_rows) continue;
        var vrow: [256]f32 = undefined;
        decodeSliceF32(vrow[0..f.cols], view_codes, view, f.cols, n - tile_rows, 0);
        try testing.expectEqualSlices(f32, row[0..f.cols], vrow[0..f.cols]);
    }
}

test "svdq: the int8 decode stays on grid for every code, -8 included" {
    var f = try Fixture.open();
    defer f.st.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const m = try f.meta(arena.allocator());
    var worst: f32 = 0;
    for (0..f.rows) |n| {
        const rs = 1.0 / m.inv_row_scale[n];
        var k: usize = 0;
        while (k < f.cols) : (k += 8) {
            const w = word(f.bytes("qweight"), f.cols, n, k);
            const s = scaleAt(m.scales, f.cols, n, k / group_size);
            for (0..8) |i| {
                const c = nibble(w, i);
                const q = decodeI8(c, s, m.inv_row_scale[n]);
                const exact = @as(f32, @floatFromInt(c)) * s;
                // Inside the int8 range before rounding, so the clamp never bites, and
                // within half an int8 step of the 4-bit value.
                errdefer std.debug.print("({d}, {d}): code {d} is {d} steps\n", .{ n, k + i, c, exact / rs });
                try testing.expect(@abs(exact / rs) <= 127.0 * (1 + 1e-6));
                worst = @max(worst, @abs(@as(f32, @floatFromInt(q)) * rs - exact) / rs);
            }
        }
    }
    errdefer std.debug.print("worst int8 rounding {d} steps\n", .{worst});
    try testing.expect(worst <= 0.5 + 1e-4);
}
