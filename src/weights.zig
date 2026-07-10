//! WeightStore — uniform read access over checkpoint containers
//! (safetensors, GGUF), so model loaders don't care which format a
//! checkpoint ships in. Both containers resolve HF-style tensor names
//! (GGUF llama.cpp names are canonicalized at parse time; see gguf.zig).

const std = @import("std");
const safetensors = @import("safetensors.zig");
const gguf = @import("gguf.zig");

pub const TensorView = safetensors.TensorView;

pub const WeightStore = union(enum) {
    safetensors: *const safetensors.SafeTensors,
    gguf: *const gguf.Gguf,

    pub fn get(self: WeightStore, name: []const u8) ?TensorView {
        return switch (self) {
            .safetensors => |st| st.get(name),
            .gguf => |g| g.get(name),
        };
    }

    /// Like `get`, but a missing tensor is an error — for required weights.
    pub fn require(self: WeightStore, name: []const u8) !TensorView {
        return self.get(name) orelse error.MissingTensor;
    }

    pub fn count(self: WeightStore) usize {
        return switch (self) {
            .safetensors => |st| st.count(),
            .gguf => |g| g.count(),
        };
    }

    pub fn names(self: WeightStore) []const []const u8 {
        return switch (self) {
            .safetensors => |st| st.names(),
            .gguf => |g| g.names(),
        };
    }

    /// The whole-file mmap backing the tensor bytes (for page-locked direct
    /// GPU streaming); null on the buffered-read path.
    pub fn mapping(self: WeightStore) ?[]align(std.heap.page_size_min) const u8 {
        return switch (self) {
            .safetensors => |st| st.mapping,
            .gguf => |g| g.mapping,
        };
    }
};

test "weight store dispatches to both containers" {
    const gpa = std.testing.allocator;

    // safetensors arm
    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    ;
    var st_file: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, st_file[0..8], header.len, .little);
    @memcpy(st_file[8..][0..header.len], header);
    const vals = [2]f32{ 3.0, -1.0 };
    @memcpy(st_file[8 + header.len ..], std.mem.sliceAsBytes(&vals));
    var st = try safetensors.SafeTensors.initFromSlice(gpa, &st_file);
    defer st.deinit();

    const store: WeightStore = .{ .safetensors = &st };
    try std.testing.expectEqual(@as(usize, 1), store.count());
    const w = try store.require("w");
    try std.testing.expectEqual(@as(usize, 2), w.info.elemCount());
    try std.testing.expect(store.get("nope") == null);
    try std.testing.expectError(error.MissingTensor, store.require("nope"));
    try std.testing.expect(store.mapping() == null);
}
