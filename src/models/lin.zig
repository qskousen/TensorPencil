//! What one block linear IS and what a device arm can run, shared by every diffusion
//! family's GPU forwards.
//!
//! A family's loader builds `device_lins`, the flat list of every linear its device
//! forwards run, and every scan here reads that list. A probe of one tensor answers
//! about one tensor: real checkpoints quantize per layer (block 0 dense, blocks 1-27
//! int8), and one GGUF mixes q8_0 and q4_k by width. Linears a family evaluates on the
//! host (AdaLN pairs, Anima's cross k/v on Vulkan) stay out of the list: `ops.matmul`
//! runs every format at any shape there.
//!
//! The CUDA GEMM dispatch built on this is `lin_cuda.zig`.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");

pub const DType = tp_core.dtype.DType;
pub const Weight = ops.matmul.Weight;

/// Storage class of one weight. `blockq` is every ggml block quant; a device arm picks
/// its route per format from there.
pub const Kind = enum { f32, f16, bf16, fp8, i8, i4, w4a8, nvfp4, blockq, other };

pub fn kindOf(dt: DType) Kind {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .f8_e4m3 => .fp8,
        .i8 => .i8,
        .i4 => .i4,
        .w4a8 => .w4a8,
        .nvfp4 => .nvfp4,
        else => if (dt.isBlockQuant()) .blockq else .other,
    };
}

/// Which storage classes a device arm has a GEMM for. W4A8 and int4 each need a
/// decode kernel on top of the int8 GEMM, so an arm with int8 alone still says false.
pub const Caps = struct {
    f32: bool = false,
    f16: bool = false,
    bf16: bool = false,
    fp8: bool = false,
    i8: bool = false,
    i4: bool = false,
    w4a8: bool = false,
    nvfp4: bool = false,
    blockq: bool = false,

    pub fn has(c: Caps, k: Kind) bool {
        return switch (k) {
            .f32 => c.f32,
            .f16 => c.f16,
            .bf16 => c.bf16,
            .fp8 => c.fp8,
            .i8 => c.i8,
            .i4 => c.i4,
            .w4a8 => c.w4a8,
            .nvfp4 => c.nvfp4,
            .blockq => c.blockq,
            .other => false,
        };
    }
};

pub const Bad = struct { tag: []const u8, dtype: DType };

/// The first linear `caps` cannot run, or null.
pub fn unsupported(lins: []const Weight, caps: Caps) ?Bad {
    for (lins) |w| {
        if (!caps.has(kindOf(w.dtype))) return .{ .tag = w.tag orelse "<untagged>", .dtype = w.dtype };
    }
    return null;
}

/// The activation prep a storage class needs on an int8-convrot GEMM surface. W4A8
/// shares int8's: only its weight storage differs, so one prep serves a group holding
/// both.
pub const Prep = enum { none, i8, i4 };

pub fn prepOf(k: Kind) Prep {
    return switch (k) {
        .i8, .w4a8 => .i8,
        .i4 => .i4,
        else => .none,
    };
}

/// Whether the linears sharing the int8 activation prep carry the convrot rotation, or
/// null when they disagree. False when there are none.
///
/// ComfyUI's `int8_tensorwise` ships rotated (a scale per output row) and unrotated
/// (one scale per tensor). The rotation cancels across the GEMM only when both sides
/// apply it, and one prep serves every GEMM in a group, so a checkpoint that mixes the
/// two has no prep that is right for all of them. W4A8 is counted: it decodes to a
/// rotated int8 weight and takes int8's prep.
pub fn convrot(lins: []const Weight) ?bool {
    var seen: ?bool = null;
    for (lins) |w| {
        if (prepOf(kindOf(w.dtype)) == .none) continue;
        const rot = w.convrot != 0;
        if (seen) |s| {
            if (s != rot) return null;
        } else seen = rot;
    }
    return seen orelse false;
}

pub fn any(lins: []const Weight, kind: Kind) bool {
    for (lins) |w| if (kindOf(w.dtype) == kind) return true;
    return false;
}

/// The first linear of this storage class, or null.
pub fn first(lins: []const Weight, kind: Kind) ?Weight {
    for (lins) |w| if (kindOf(w.dtype) == kind) return w;
    return null;
}

/// A packed W4A8 linear whose `group_size` is not a multiple of 8, by name. The CUDA
/// decode kernel reads four packed bytes (8 columns) per thread under one group scale.
/// No shipped checkpoint uses a smaller group, but the format permits 4.
pub fn w4a8SmallGroup(lins: []const Weight) ?[]const u8 {
    for (lins) |w| {
        if (w.dtype == .w4a8 and w.w4a8.?.group_size % 8 != 0) return w.tag orelse "<untagged>";
    }
    return null;
}

/// Largest transient buffer any linear of `kind` decodes into, per the backend's own
/// sizing rule, or 0. A caller pre-sizes with it so the scratch never grows mid-forward.
pub fn maxScratch(lins: []const Weight, kind: Kind, comptime bytesFor: fn (rows: usize, cols: usize) usize) usize {
    var max: usize = 0;
    for (lins) |w| {
        if (kindOf(w.dtype) == kind) max = @max(max, bytesFor(w.rows, w.cols));
    }
    return max;
}

/// DIAGNOSTIC: materialize every block-quant linear to f32 at load, so the same
/// checkpoint runs through the dense path.
///
/// The isolation for a quantized route: a render that is right this way and wrong the
/// packed way puts the defect in the decode or the GEMM, and one that is wrong both ways
/// puts it everywhere else. It costs 4 bytes an element, which is the whole memory the
/// format saves, so it is a debugging tool and never a shipping path.
pub var dequant_at_load: enum { off, f32, bf16 } = .off;

/// `w` with a block quant expanded when `dequant_at_load` is set, else `w`. bf16 is the
/// truer control at real depths: it is what a native dense checkpoint carries, and f32
/// does not fit in VRAM for a whole trunk.
pub fn maybeDequant(alloc: std.mem.Allocator, w: Weight) !Weight {
    if (!w.dtype.isBlockQuant()) return w;
    return switch (dequant_at_load) {
        .off => w,
        .f32 => ops.matmul.materializeF32(alloc, w),
        .bf16 => ops.matmul.materializeBf16(alloc, w),
    };
}

/// Rows `[row0, row0 + nrows)` of `w` as a Weight of its own, sidecars included.
///
/// For a fused qkv weight whose halves the device consumes separately. Every per-row
/// sidecar is sliced with the bytes: a view that kept the parent's `row_scale` or
/// block scales would read q's for k and v, which is finite, plausible and wrong. The
/// sliced sidecar metas land in `alloc`; the tag stays the parent's.
pub fn rowSlice(alloc: std.mem.Allocator, w: Weight, row0: usize, nrows: usize) !Weight {
    std.debug.assert(row0 + nrows <= w.rows);
    if (w.dtype.isBlockQuant()) std.debug.assert(w.cols % w.dtype.blockElems() == 0);
    const row_bytes = w.dtype.storageBytes(w.cols);
    var s = w;
    s.rows = nrows;
    s.bytes = w.bytes[row0 * row_bytes ..][0 .. nrows * row_bytes];
    if (w.row_scale) |rs| s.row_scale = rs[row0..][0..nrows];
    if (w.nvfp4) |m| {
        const nm = try alloc.create(ops.nvfp4.Meta);
        nm.* = m.rowSlice(w.cols, row0, nrows);
        s.nvfp4 = nm;
    }
    if (w.w4a8) |m| {
        const nm = try alloc.create(ops.w4a8.Meta);
        nm.* = m.rowSlice(w.cols, row0, nrows);
        s.w4a8 = nm;
    }
    return s;
}

// --- tests -----------------------------------------------------------------

test "kindOf sorts every dtype into a class" {
    try std.testing.expectEqual(Kind.bf16, kindOf(.bf16));
    try std.testing.expectEqual(Kind.fp8, kindOf(.f8_e4m3));
    try std.testing.expectEqual(Kind.w4a8, kindOf(.w4a8));
    for ([_]DType{ .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs }) |dt|
        try std.testing.expectEqual(Kind.blockq, kindOf(dt));
    try std.testing.expectEqual(Kind.other, kindOf(.u8));
    try std.testing.expectEqual(Kind.other, kindOf(.f64));
}

test "unsupported names the first linear a backend lacks" {
    const a = [_]u8{0} ** (4 * 8 * 2);
    var lins = [_]Weight{
        Weight.init(&a, .bf16, 4, 8),
        Weight.init(a[0 .. 4 * 8], .i8, 4, 8),
    };
    lins[1].tag = "blk.1.wq";
    try std.testing.expect(unsupported(&lins, .{ .bf16 = true, .i8 = true }) == null);
    const bad = unsupported(&lins, .{ .bf16 = true }).?;
    try std.testing.expectEqualStrings("blk.1.wq", bad.tag);
    try std.testing.expectEqual(DType.i8, bad.dtype);
}

test "convrot answers per checkpoint and refuses a mix" {
    const a = [_]u8{0} ** 64;
    var rot = Weight.init(a[0..32], .i8, 4, 8);
    rot.convrot = 256;
    const flat = Weight.init(a[0..32], .i8, 4, 8);
    const dense = Weight.init(&a, .bf16, 4, 8);
    try std.testing.expectEqual(@as(?bool, false), convrot(&.{dense}));
    try std.testing.expectEqual(@as(?bool, true), convrot(&.{ dense, rot }));
    try std.testing.expectEqual(@as(?bool, false), convrot(&.{ flat, dense }));
    try std.testing.expectEqual(@as(?bool, null), convrot(&.{ rot, flat }));
}

test "rowSlice moves the bytes and the per-row scale together" {
    const rows = 6;
    const cols = 32;
    var q: [rows * cols]i8 = undefined;
    for (&q, 0..) |*v, i| v.* = @intCast(i % 100);
    var scale: [rows]f32 = undefined;
    for (&scale, 0..) |*s, r| s.* = @floatFromInt(r + 1);
    var w = Weight.init(std.mem.sliceAsBytes(&q), .i8, rows, cols);
    w.row_scale = &scale;
    w.tag = "qkv";
    const k = try rowSlice(std.testing.allocator, w, 2, 3);
    try std.testing.expectEqual(@as(usize, 3), k.rows);
    try std.testing.expectEqual(@as(usize, cols), k.cols);
    try std.testing.expectEqual(@as(i8, @intCast((2 * cols) % 100)), @as(i8, @bitCast(k.bytes[0])));
    try std.testing.expectEqual(@as(f32, 3), k.row_scale.?[0]);
    try std.testing.expectEqual(@as(usize, 3), k.row_scale.?.len);
    try std.testing.expectEqualStrings("qkv", k.tag.?);

    // A block quant slices by whole rows of blocks: q8_0 is 34 bytes per 32 elements.
    const bq = [_]u8{0} ** (rows * 34);
    const bw = Weight.init(&bq, .q8_0, rows, cols);
    const bs = try rowSlice(std.testing.allocator, bw, 4, 2);
    try std.testing.expectEqual(@as(usize, 2 * 34), bs.bytes.len);
    try std.testing.expectEqual(@intFromPtr(bq[4 * 34 ..].ptr), @intFromPtr(bs.bytes.ptr));
}

test "three row-view GEMMs agree with the fused one the CPU forward runs" {
    // Z-Image's device path does three GEMMs on row slices where the CPU path does one
    // wide GEMM and de-interleaves the result. A swapped or misaligned slice is not an
    // error, q/k/v would simply be each other's, which renders as structured noise.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dim = 8;
    const q_dim = 8;
    const kv_dim = 8;
    const rows = q_dim + 2 * kv_dim;
    const m = 3;

    var wbits: [rows * dim]f32 = undefined;
    for (&wbits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.25 - 1.0;
    const w = Weight.fromF32(&wbits, rows, dim);

    var x: [m * dim]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.5 - 1.5;

    const fused = try gpa.alloc(f32, m * rows);
    defer gpa.free(fused);
    try ops.matmul.matmul(io, gpa, fused, &x, m, w, null);

    inline for (.{ .{ 0, q_dim }, .{ q_dim, kv_dim }, .{ q_dim + kv_dim, kv_dim } }, 0..) |part, pi| {
        const sub = try rowSlice(gpa, w, part[0], part[1]);
        try std.testing.expectEqual(@as(usize, part[1]), sub.rows);
        try std.testing.expectEqual(dim, sub.cols);
        const got = try gpa.alloc(f32, m * part[1]);
        defer gpa.free(got);
        try ops.matmul.matmul(io, gpa, got, &x, m, sub, null);
        for (0..m) |r| {
            for (0..part[1]) |c| {
                errdefer std.debug.print("part {d} row {d} col {d}\n", .{ pi, r, c });
                try std.testing.expectEqual(fused[r * rows + part[0] + c], got[r * part[1] + c]);
            }
        }
    }
}
