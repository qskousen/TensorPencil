//! Host halves the encoder tower walks share (`embed_*_dev.zig`, over `tower_cuda.Dev`
//! / `tower_gpu.Dev`): a ragged batch's row layout and its attention bounds, and the
//! rotate-half table the device RoPE reads.

const std = @import("std");
const ops = @import("tp_ops");

/// Items packed back to back: item `i` is rows `[off[i], off[i + 1])`.
pub const Ragged = struct {
    off: []usize,
    total: usize,
    longest: usize,

    pub fn init(gpa: std.mem.Allocator, lens: anytype) !Ragged {
        const off = try gpa.alloc(usize, lens.len + 1);
        off[0] = 0;
        var longest: usize = 0;
        for (lens, 0..) |l, i| {
            const n = if (@TypeOf(l) == usize) l else l.len;
            off[i + 1] = off[i] + n;
            longest = @max(longest, n);
        }
        return .{ .off = off, .total = off[lens.len], .longest = longest };
    }

    pub fn deinit(self: *Ragged, gpa: std.mem.Allocator) void {
        gpa.free(self.off);
        self.* = undefined;
    }

    pub fn items(self: Ragged) usize {
        return self.off.len - 1;
    }

    pub fn len(self: Ragged, i: usize) usize {
        return self.off[i + 1] - self.off[i];
    }

    /// Per-row `[start | end]` of the row's item, the block-diagonal attention's mask.
    /// Caller frees.
    pub fn bounds(self: Ragged, gpa: std.mem.Allocator) ![]u32 {
        const out = try gpa.alloc(u32, 2 * self.total);
        for (0..self.items()) |i| for (self.off[i]..self.off[i + 1]) |r| {
            out[r] = @intCast(self.off[i]);
            out[self.total + r] = @intCast(self.off[i + 1]);
        };
        return out;
    }
};

/// `[cos | sin]` rotate-half table over `rows` positions; the sine half starts at
/// `rows * head_dim / 2`. Caller frees.
pub fn ropeTable(gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64) ![]f32 {
    const half = head_dim / 2;
    var f = try ops.rope.rotateHalfFreqs(gpa, rows, head_dim, theta);
    defer f.deinit(gpa);
    const flat = try gpa.alloc(f32, rows * half * 2);
    @memcpy(flat[0 .. rows * half], f.cos);
    @memcpy(flat[rows * half ..], f.sin);
    return flat;
}
