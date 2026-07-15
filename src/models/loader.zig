//! Shared weight-loading helpers for the model loaders. Every model built the
//! same `loadMatrix`/`loadVec` (+ indexed `layers.N.` / `v.blk.N.` variants)
//! by hand; the logic — shape/length validation, the name-formatting buffer,
//! the dtype-preserving `Weight` view vs. f32 dequant — lives here once. Model
//! files keep their local `loadMatrix`/`loadLayerMatrix`/… names as thin
//! shims over these, so call sites read naturally and only the logic is shared.

const std = @import("std");
const weights = @import("../weights.zig");
const ops = @import("../ops.zig");

const WeightStore = weights.WeightStore;
const Weight = ops.matmul.Weight;

/// A 2-D matrix weight by exact tensor name, kept in its checkpoint dtype (the
/// GEMM dequantizes inside). Errors if absent or not exactly `rows`×`cols`.
pub fn matrix(store: WeightStore, name: []const u8, rows: usize, cols: usize) !Weight {
    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

/// A 1-D vector weight (norm scales, biases), dequantized to owned f32. Errors
/// if absent or not exactly `len` elements.
pub fn vector(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]f32 {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

/// A matrix under an indexed block prefix: `prefix ++ "{i}." ++ suffix`
/// (e.g. prefix `"layers."`, i 3, suffix `"self_attn.q_proj.weight"` →
/// `"layers.3.self_attn.q_proj.weight"`).
pub fn indexedMatrix(
    store: WeightStore,
    comptime prefix: []const u8,
    i: usize,
    comptime suffix: []const u8,
    rows: usize,
    cols: usize,
) !Weight {
    var buf: [128]u8 = undefined;
    return matrix(store, try std.fmt.bufPrint(&buf, prefix ++ "{d}." ++ suffix, .{i}), rows, cols);
}

/// A vector under an indexed block prefix (see `indexedMatrix`).
pub fn indexedVector(
    alloc: std.mem.Allocator,
    store: WeightStore,
    comptime prefix: []const u8,
    i: usize,
    comptime suffix: []const u8,
    len: usize,
) ![]f32 {
    var buf: [128]u8 = undefined;
    return vector(alloc, store, try std.fmt.bufPrint(&buf, prefix ++ "{d}." ++ suffix, .{i}), len);
}

// --- tests -----------------------------------------------------------------

const safetensors = @import("../safetensors.zig");

// A tiny in-memory safetensors: matrix "w" [2,2]={1,2,3,4} and vector
// "layers.3.b" [2]={5,6}. The file buffer must outlive the SafeTensors (views
// reference it), so the caller owns `file` and passes it in.
const test_header =
    \\{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"layers.3.b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}
;
const test_file_len = 8 + test_header.len + 24;

fn fillTestFile(file: *[test_file_len]u8) void {
    std.mem.writeInt(u64, file[0..8], test_header.len, .little);
    @memcpy(file[8..][0..test_header.len], test_header);
    const w = [4]f32{ 1, 2, 3, 4 };
    const b = [2]f32{ 5, 6 };
    @memcpy(file[8 + test_header.len ..][0..16], std.mem.sliceAsBytes(&w));
    @memcpy(file[8 + test_header.len + 16 ..][0..8], std.mem.sliceAsBytes(&b));
}

test "matrix: shape validated, dtype preserved" {
    const gpa = std.testing.allocator;
    var file: [test_file_len]u8 = undefined;
    fillTestFile(&file);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, &file);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    const w = try matrix(store, "w", 2, 2);
    try std.testing.expectEqual(@as(usize, 2), w.rows);
    try std.testing.expectEqual(@as(usize, 2), w.cols);
    try std.testing.expectError(error.ShapeMismatch, matrix(store, "w", 3, 2));
    try std.testing.expectError(error.MissingTensor, matrix(store, "nope", 2, 2));
}

test "vector: length validated, dequantized to f32" {
    const gpa = std.testing.allocator;
    var file: [test_file_len]u8 = undefined;
    fillTestFile(&file);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, &file);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    const v = try vector(gpa, store, "layers.3.b", 2);
    defer gpa.free(v);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6 }, v);
    try std.testing.expectError(error.ShapeMismatch, vector(gpa, store, "layers.3.b", 3));
}

test "indexed helpers format prefix.i.suffix" {
    const gpa = std.testing.allocator;
    var file: [test_file_len]u8 = undefined;
    fillTestFile(&file);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, &file);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    // "layers." ++ 3 ++ "." ++ "b" == "layers.3.b"
    const v = try indexedVector(gpa, store, "layers.", 3, "b", 2);
    defer gpa.free(v);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6 }, v);
    try std.testing.expectError(error.MissingTensor, indexedVector(gpa, store, "layers.", 9, "b", 2));
}
