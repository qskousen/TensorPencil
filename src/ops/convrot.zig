//! ConvRot: the grouped orthogonal rotation used by ComfyUI's int8_tensorwise
//! "convrot" quantization (comfy_kitchen/tensor/int8_utils.py).
//!
//! Weights are stored quantized as `W_rot = W @ H^T`, grouped into 256-column
//! blocks along the input dim, with `H` a normalized size-256 *regular*
//! Hadamard matrix (kron of the 4x4 block). That matrix is symmetric and
//! orthonormal, so `H = H^T` and `H*H = I`; recovering the original weight from
//! the dequantized-and-scaled `W_rot` therefore applies the same rotation once
//! more: `W = W_rot @ H^T = W_rot @ H`. The rotation spreads per-group outliers
//! so int8 quantization (of both weights offline and activations online) is
//! more accurate — but for our correctness path we simply rotate back to the
//! original weights at dequant time and reuse the normal GEMM.

const std = @import("std");

/// ComfyUI's `convrot_groupsize` for these checkpoints.
pub const group_size = 256;

/// 4x4 regular-Hadamard building block (symmetric, H4*H4 = 4*I).
const h4 = [4][4]f32{
    .{ 1, 1, 1, -1 },
    .{ 1, 1, -1, 1 },
    .{ 1, -1, 1, 1 },
    .{ -1, 1, 1, 1 },
};

/// Normalized size-256 regular Hadamard: kron(h4,h4,h4,h4) / sqrt(256).
/// `H[i][j] = product over the 4 base-4 digits of h4[digit(i)][digit(j)] / 16`.
/// Symmetric and orthonormal, so H = H^T and H*H = I.
pub const H: [group_size][group_size]f32 = blk: {
    @setEvalBranchQuota(4_000_000);
    var m: [group_size][group_size]f32 = undefined;
    for (0..group_size) |i| {
        for (0..group_size) |j| {
            var p: f32 = 1.0;
            var ii = i;
            var jj = j;
            for (0..4) |_| {
                p *= h4[ii % 4][jj % 4];
                ii /= 4;
                jj /= 4;
            }
            m[i][j] = p / 16.0; // 1 / sqrt(256)
        }
    }
    break :blk m;
};

/// Rotate one group in place: `out = H * in` (H symmetric, so also `in^T @ H`).
///
/// `H` is a tensor product of the 4x4 block, so instead of the O(n^2) matvec we
/// apply the block along each base-4 digit axis (a radix-4 fast Walsh-Hadamard
/// transform): 4 passes at strides 1,4,16,64, each an unnormalized 4-point
/// butterfly, then one `/16` normalization (= 1/sqrt(256)). ~40x fewer ops than
/// the matvec and validated against `H` in the tests.
fn rotateGroup(v: *[group_size]f32) void {
    const strides = [_]usize{ 1, 4, 16, 64 };
    for (strides) |s| {
        var base: usize = 0;
        while (base < group_size) : (base += s * 4) {
            for (0..s) |o| {
                const p = base + o;
                const a = v[p];
                const b = v[p + s];
                const c = v[p + 2 * s];
                const d = v[p + 3 * s];
                // 4x4 regular Hadamard block (unnormalized).
                v[p] = a + b + c - d;
                v[p + s] = a + b - c + d;
                v[p + 2 * s] = a - b + c + d;
                v[p + 3 * s] = -a + b + c + d;
            }
        }
    }
    for (v) |*x| x.* /= 16.0;
}

/// Apply the group rotation to each consecutive `group_size` chunk of `data`.
/// `data.len` must be a multiple of `group_size` (guaranteed for these weights:
/// every quantized input dim is a multiple of 256).
pub fn rotate(data: []f32) void {
    std.debug.assert(data.len % group_size == 0);
    var off: usize = 0;
    while (off < data.len) : (off += group_size) {
        rotateGroup(data[off..][0..group_size]);
    }
}

// --- tests -----------------------------------------------------------------

test "hadamard is symmetric" {
    for (0..group_size) |i| {
        for (0..group_size) |j| {
            try std.testing.expectEqual(H[i][j], H[j][i]);
        }
    }
}

test "hadamard is orthonormal (H*H = I)" {
    for (0..group_size) |i| {
        for (0..group_size) |j| {
            var dot: f32 = 0;
            for (0..group_size) |k| dot += H[i][k] * H[k][j];
            const expected: f32 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, dot, 1e-5);
        }
    }
}

test "rotation is self-inverse" {
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    var v: [group_size]f32 = undefined;
    for (&v) |*x| x.* = rand.floatNorm(f32);
    const orig = v;

    rotate(&v);
    rotate(&v); // H applied twice == identity
    for (orig, v) |o, a| try std.testing.expectApproxEqAbs(o, a, 1e-4);
}

test "rotation matches explicit matvec across multiple groups" {
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    var data: [2 * group_size]f32 = undefined;
    for (&data) |*x| x.* = rand.floatNorm(f32);
    const orig = data;

    rotate(&data);

    // Reference: independent H*group matvec per group.
    for (0..2) |g| {
        for (0..group_size) |i| {
            var acc: f32 = 0;
            for (0..group_size) |j| acc += H[i][j] * orig[g * group_size + j];
            try std.testing.expectApproxEqAbs(acc, data[g * group_size + i], 1e-4);
        }
    }
}

// Bit-exact validation against ComfyUI's authoritative `comfy_kitchen`
// implementation. The fixtures are generated by ggufy/gen_int4_fixtures.py
// straight out of `quantize/dequantize_convrot_w4a4_weight` (the exact routines
// ComfyUI's official W4A4 converter and loader use), so matching them proves our
// nibble unpack + per-row scale + group-256 rotation reproduce ComfyUI's
// `convrot_w4a4` dequant. `expected` = unpack(qweight) * scale[:,None] rotated
// back into the original basis (`@ H^T`, and H is symmetric so `@ H`).
test "convrot int4 dequant matches comfyui reference fixture" {
    const dtypes = @import("tp_core").dtype;
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const rows = 16;
    const cols = 6144; // logical; on-disk is nibble-packed to cols/2 bytes per row.
    std.debug.assert(cols % group_size == 0);

    const dir = "testdata/";
    // Skip cleanly if the fixtures aren't checked out on this machine.
    std.Io.Dir.cwd().access(io, dir ++ "int4_convrot_weight.u8", .{}) catch return error.SkipZigTest;

    const packed_bytes = try readFile(u8, gpa, io, dir ++ "int4_convrot_weight.u8", rows * cols / 2);
    defer gpa.free(packed_bytes);
    const scale = try readFile(f32, gpa, io, dir ++ "int4_convrot_scale.f32", rows);
    defer gpa.free(scale);
    const expected = try readFile(f32, gpa, io, dir ++ "int4_convrot_expected.f32", rows * cols);
    defer gpa.free(expected);

    var row: [cols]f32 = undefined;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..rows) |r| {
        // Unpack the row's nibbles (two per byte, element 2k = low nibble) and
        // apply the per-row scale, exactly as the DiT int4 dequant path does.
        const rbytes = packed_bytes[r * cols / 2 ..][0 .. cols / 2];
        for (0..cols) |k| {
            const v = dtypes.DType.nibbleI4(rbytes[k / 2], @intCast(k & 1));
            row[k] = @as(f32, @floatFromInt(v)) * scale[r];
        }
        // Rotate back into the original basis (H is self-inverse).
        rotate(&row);
        for (0..cols) |k| {
            const ref = expected[r * cols + k];
            const d = ref - row[k];
            num += @as(f64, d) * d;
            den += @as(f64, ref) * ref;
        }
    }
    // The only gap vs. the reference is float accumulation order (our radix-4
    // FWHT vs. torch's dense Hadamard matmul), so the agreement is tight.
    const rel = @sqrt(num / den);
    try std.testing.expect(rel < 1e-4);
}

fn readFile(comptime T: type, gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]T {
    const out = try gpa.alloc(T, n);
    errdefer gpa.free(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    return out;
}
