//! Per-layer K/V cache for autoregressive decode (f32, host memory —
//! LLM_PLAN.md M2; a GPU-resident variant lands with M4).
//!
//! Layout: k/v are [n_layers][capacity][kv_dim] row-major. A forward pass
//! writes the new tokens' K/V at row `len` for every layer, then `commit`s
//! once so all layers observe the same position base.

const std = @import("std");

pub const KvCache = struct {
    k: []f32,
    v: []f32,
    n_layers: usize,
    capacity: usize,
    kv_dim: usize,
    /// Committed token count (positions 0..len are valid in every layer).
    len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, n_layers: usize, capacity: usize, kv_dim: usize) !KvCache {
        const k = try gpa.alloc(f32, n_layers * capacity * kv_dim);
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, n_layers * capacity * kv_dim);
        return .{ .k = k, .v = v, .n_layers = n_layers, .capacity = capacity, .kv_dim = kv_dim };
    }

    pub fn deinit(self: *KvCache, gpa: std.mem.Allocator) void {
        gpa.free(self.k);
        gpa.free(self.v);
        self.* = undefined;
    }

    /// Room left for new tokens.
    pub fn remaining(self: *const KvCache) usize {
        return self.capacity - self.len;
    }

    /// Write `n` tokens' K/V for `layer` at the uncommitted rows [len, len+n).
    pub fn write(self: *KvCache, layer: usize, k_new: []const f32, v_new: []const f32) void {
        std.debug.assert(k_new.len == v_new.len and k_new.len % self.kv_dim == 0);
        std.debug.assert(self.len * self.kv_dim + k_new.len <= self.capacity * self.kv_dim);
        const base = (layer * self.capacity + self.len) * self.kv_dim;
        @memcpy(self.k[base..][0..k_new.len], k_new);
        @memcpy(self.v[base..][0..v_new.len], v_new);
    }

    /// K rows [0, len + extra) of `layer` — `extra` covers written-but-not-yet
    /// committed tokens mid-forward.
    pub fn kView(self: *const KvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.k, layer, extra);
    }

    pub fn vView(self: *const KvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.v, layer, extra);
    }

    fn view(self: *const KvCache, buf: []const f32, layer: usize, extra: usize) []const f32 {
        std.debug.assert(self.len + extra <= self.capacity);
        return buf[layer * self.capacity * self.kv_dim ..][0 .. (self.len + extra) * self.kv_dim];
    }

    /// Advance `len` after all layers have written the same `n` tokens.
    pub fn commit(self: *KvCache, n: usize) void {
        std.debug.assert(self.len + n <= self.capacity);
        self.len += n;
    }

    /// Roll back to `new_len` committed tokens (speculative-decode rejection):
    /// rows past `new_len` are simply overwritten by the next write.
    pub fn truncate(self: *KvCache, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }
};

// --- tests -----------------------------------------------------------------

test "write/commit/view round trip" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 4, 3);
    defer cache.deinit(gpa);

    // Two tokens across both layers, then commit.
    const k0 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const v0 = [_]f32{ -1, -2, -3, -4, -5, -6 };
    const k1 = [_]f32{ 10, 20, 30, 40, 50, 60 };
    cache.write(0, &k0, &v0);
    cache.write(1, &k1, &v0);
    try std.testing.expectEqual(@as(usize, 0), cache.len);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 2));
    try std.testing.expectEqualSlices(f32, &k1, cache.kView(1, 2));
    cache.commit(2);
    try std.testing.expectEqual(@as(usize, 2), cache.len);
    try std.testing.expectEqual(@as(usize, 2), cache.remaining());

    // One more token lands at row 2 of each layer.
    const k2 = [_]f32{ 7, 8, 9 };
    cache.write(0, &k2, &k2);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &(k0 ++ k2), cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &(v0 ++ k2), cache.vView(0, 0));
}

test "truncate rolls back and rows are rewritten" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 1, 4, 2);
    defer cache.deinit(gpa);

    const a = [_]f32{ 1, 2, 3, 4, 5, 6 }; // three tokens
    cache.write(0, &a, &a);
    cache.commit(3);
    cache.truncate(1);
    try std.testing.expectEqual(@as(usize, 1), cache.len);
    try std.testing.expectEqual(@as(usize, 3), cache.remaining());

    // New tokens land at row 1, replacing the rolled-back ones.
    const b = [_]f32{ 9, 10 };
    cache.write(0, &b, &b);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 9, 10 }, cache.kView(0, 0));
}
