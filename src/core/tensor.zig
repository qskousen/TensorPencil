//! Tensor shape/metadata types and a minimal owned f32 tensor.

const std = @import("std");
const dtype = @import("dtype.zig");

pub const max_rank = 8;

/// Fixed-capacity tensor shape. Rank 0 is a scalar (element count 1) — the
/// text encoder's per-tensor `weight_scale` entries use this.
pub const Shape = struct {
    dims: [max_rank]usize,
    rank: usize,

    pub const scalar: Shape = .{ .dims = @splat(0), .rank = 0 };

    pub fn init(dims: []const usize) Shape {
        std.debug.assert(dims.len <= max_rank);
        var s: Shape = .{ .dims = @splat(0), .rank = dims.len };
        @memcpy(s.dims[0..dims.len], dims);
        return s;
    }

    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.rank];
    }

    pub fn count(self: Shape) usize {
        var n: usize = 1;
        for (self.dims[0..self.rank]) |d| n *= d;
        return n;
    }

    pub fn eql(a: Shape, b: Shape) bool {
        return a.rank == b.rank and std.mem.eql(usize, a.dims[0..a.rank], b.dims[0..b.rank]);
    }
};

/// Metadata for one tensor inside a safetensors file. `start`/`end` are byte
/// offsets into the file's data section (after the JSON header).
pub const TensorInfo = struct {
    name: []const u8,
    dtype: dtype.DType,
    shape: Shape,
    start: usize,
    end: usize,

    pub fn byteLen(self: TensorInfo) usize {
        return self.end - self.start;
    }

    pub fn elemCount(self: TensorInfo) usize {
        return self.shape.count();
    }
};

/// Minimal owned, contiguous f32 tensor in host memory.
pub const Tensor = struct {
    data: []f32,
    shape: Shape,

    pub fn init(gpa: std.mem.Allocator, shape: Shape) !Tensor {
        return .{ .data = try gpa.alloc(f32, shape.count()), .shape = shape };
    }

    pub fn initZeros(gpa: std.mem.Allocator, shape: Shape) !Tensor {
        const t = try init(gpa, shape);
        @memset(t.data, 0);
        return t;
    }

    pub fn deinit(self: *Tensor, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
        self.* = undefined;
    }
};

test "shape basics" {
    const s = Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), s.rank);
    try std.testing.expectEqual(@as(usize, 24), s.count());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, s.slice());
    try std.testing.expect(s.eql(Shape.init(&.{ 2, 3, 4 })));
    try std.testing.expect(!s.eql(Shape.init(&.{ 2, 3 })));
    try std.testing.expect(!s.eql(Shape.init(&.{ 2, 3, 5 })));
}

test "scalar shape has one element" {
    try std.testing.expectEqual(@as(usize, 1), Shape.scalar.count());
    try std.testing.expectEqual(@as(usize, 0), Shape.scalar.slice().len);
}

test "tensor alloc and zero" {
    const gpa = std.testing.allocator;
    var t = try Tensor.initZeros(gpa, Shape.init(&.{ 4, 4 }));
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 16), t.data.len);
    for (t.data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
