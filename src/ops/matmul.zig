//! Dtype-aware GEMM: y[m, rows] = x[m, cols] @ W^T (+ bias), W stored
//! row-major [rows, cols] as in torch Linear / safetensors.
//!
//! Weights stay in their storage dtype (fp8-e4m3 / bf16 / f16 / f32) and are
//! dequantized into small f32 row panels inside the kernel, so a 12 GiB fp8
//! checkpoint never expands in memory. Work is split over output rows across
//! `std.Io.Group` tasks; accumulation is f32 SIMD.

const std = @import("std");
const dtypes = @import("../dtype.zig");
const gpu_context = @import("../gpu/context.zig");

const DType = dtypes.DType;

/// When set (pipeline --gpu), large f8/f32 GEMMs are dispatched to Vulkan.
/// Single-threaded use only — the pipeline runs matmuls sequentially.
pub var gpu: ?*gpu_context.Context = null;

/// Minimum FLOP count before the GPU path is worth the PCIe round trip.
pub var gpu_min_flops: usize = 1 << 31;

fn gpuEligible(m: usize, w: Weight) bool {
    if (gpu == null) return false;
    if (w.dtype != .f8_e4m3 and w.dtype != .f32) return false;
    return 2 * m * w.rows * w.cols >= gpu_min_flops;
}

/// Rows dequantized together per panel in the small-m path. 8 rows x 16384
/// cols x 4 B = 512 KiB worst case (DiT MLP), which stays comfortably in L2.
const panel_rows = 8;

const vlen = std.simd.suggestVectorLength(f32) orelse 8;
const Vec = @Vector(vlen, f32);

// Packed outer-product path (m >= small_m_max): B subpanels of NR output
// columns are dequantized k-major so the microkernel runs MR x NR register
// tiles of fused multiply-adds with embedded broadcasts of x.
const MR = 6;
const NR = 2 * vlen;
const KC = 512; // k-block: one packed subpanel slice is KC*NR*4 = 128 KiB max
const small_m_max = 16;

/// A weight matrix view over raw checkpoint bytes.
pub const Weight = struct {
    bytes: []const u8,
    dtype: DType,
    rows: usize,
    cols: usize,
    /// Per-tensor dequant scale (ComfyUI fp8 format stores one per weight).
    scale: f32 = 1.0,

    pub fn init(bytes: []const u8, dtype: DType, rows: usize, cols: usize) Weight {
        std.debug.assert(bytes.len == rows * cols * dtype.byteSize());
        return .{ .bytes = bytes, .dtype = dtype, .rows = rows, .cols = cols };
    }

    /// Convenience for tests / f32 weights already in memory.
    pub fn fromF32(data: []const f32, rows: usize, cols: usize) Weight {
        return init(std.mem.sliceAsBytes(data), .f32, rows, cols);
    }
};

pub const Error = error{ UnsupportedDType, OutOfMemory } || std.Io.Cancelable;

/// y[m, w.rows] = x[m, w.cols] @ w^T + bias.
pub fn matmul(
    io: std.Io,
    gpa: std.mem.Allocator,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
) Error!void {
    std.debug.assert(x.len == m * w.cols);
    std.debug.assert(y.len == m * w.rows);
    if (bias) |b| std.debug.assert(b.len == w.rows);
    switch (w.dtype) {
        .f8_e4m3, .bf16, .f16, .f32 => {},
        else => return error.UnsupportedDType,
    }
    if (m == 0 or w.rows == 0) return;

    if (gpuEligible(m, w)) {
        if (gpu.?.matmul(y, x, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, bias)) |_| {
            return;
        } else |err| {
            // Fall back to CPU once and stop routing.
            std.log.warn("gpu matmul failed ({t}); falling back to cpu", .{err});
            gpu = null;
        }
    }

    if (m >= small_m_max) return matmulPacked(io, gpa, y, x, m, w, bias);

    // Small problems are not worth the fork/join overhead.
    const flops = 2 * m * w.rows * w.cols;
    const n_threads = std.Thread.getCpuCount() catch 1;
    const want_tasks: usize = if (flops < (1 << 20) or n_threads == 1) 1 else 4 * n_threads;

    const chunk = chunkRows(w.rows, want_tasks);
    const n_tasks = std.math.divCeil(usize, w.rows, chunk) catch unreachable;

    const scratch = try gpa.alloc(f32, n_tasks * panel_rows * w.cols);
    defer gpa.free(scratch);

    if (n_tasks == 1) {
        runRange(y, x, m, w, bias, 0, w.rows, scratch);
        return;
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    var task: usize = 0;
    var row: usize = 0;
    while (row < w.rows) : (row += chunk) {
        const row_end = @min(row + chunk, w.rows);
        const task_scratch = scratch[task * panel_rows * w.cols ..][0 .. panel_rows * w.cols];
        group.async(io, runRange, .{ y, x, m, w, bias, row, row_end, task_scratch });
        task += 1;
    }
    try group.await(io);
}

fn matmulPacked(
    io: std.Io,
    gpa: std.mem.Allocator,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
) Error!void {
    const n_threads = std.Thread.getCpuCount() catch 1;
    const want_tasks: usize = if (n_threads == 1) 1 else 4 * n_threads;
    const per_task = std.math.divCeil(usize, w.rows, want_tasks) catch unreachable;
    const chunk = std.mem.alignForward(usize, @max(per_task, NR), NR);
    const n_tasks = std.math.divCeil(usize, w.rows, chunk) catch unreachable;

    // Per-task packing buffer: one KC-slice of its row chunk, reused per k block.
    const stride = (chunk / NR) * KC * NR;
    const scratch = try gpa.alloc(f32, n_tasks * stride);
    defer gpa.free(scratch);

    if (n_tasks == 1) {
        packedTask(y, x, m, w, bias, 0, w.rows, scratch[0..stride]);
        return;
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    var task: usize = 0;
    var row: usize = 0;
    while (row < w.rows) : (row += chunk) {
        const row_end = @min(row + chunk, w.rows);
        group.async(io, packedTask, .{ y, x, m, w, bias, row, row_end, scratch[task * stride ..][0..stride] });
        task += 1;
    }
    try group.await(io);
}

fn packedTask(
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    panel: []f32,
) void {
    switch (w.dtype) {
        inline .f8_e4m3, .bf16, .f16, .f32 => |dt| {
            packedTaskTyped(dt, y, x, m, w, bias, row_start, row_end, panel);
        },
        else => unreachable, // validated in matmul()
    }
}

fn packedTaskTyped(
    comptime dt: DType,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    panel: []f32,
) void {
    const cols = w.cols;
    const rows = w.rows;
    const esize = comptime dt.byteSize();
    const n_nr = std.math.divCeil(usize, row_end - row_start, NR) catch unreachable;

    var kc0: usize = 0;
    while (kc0 < cols) : (kc0 += KC) {
        // Note the explicit type: @min with a comptime bound would narrow to
        // u10 and make `kl * NR` overflow (see ZIG.md).
        const kl: usize = @min(KC, cols - kc0);

        // Pack + dequantize this k-slice of the row chunk, k-major per subpanel.
        for (0..n_nr) |nr| {
            const sub = panel[nr * KC * NR ..][0 .. kl * NR];
            for (0..NR) |j| {
                const row = row_start + nr * NR + j;
                if (row < rows) {
                    const src = w.bytes[(row * cols + kc0) * esize ..][0 .. kl * esize];
                    for (0..kl) |k| sub[k * NR + j] = dequantOne(dt, src, k, w.scale);
                } else {
                    for (0..kl) |k| sub[k * NR + j] = 0;
                }
            }
        }

        var t0: usize = 0;
        while (t0 < m) : (t0 += MR) {
            const mr = @min(MR, m - t0);
            for (0..n_nr) |nr| {
                const sub = panel[nr * KC * NR ..][0 .. kl * NR];
                const col0 = row_start + nr * NR;
                switch (mr) {
                    inline 1...MR => |mrc| microKernel(
                        mrc,
                        y,
                        x,
                        sub,
                        t0,
                        col0,
                        rows,
                        cols,
                        kc0,
                        kl,
                        if (kc0 == 0) bias else null,
                        kc0 != 0,
                    ),
                    else => unreachable,
                }
            }
        }
    }
}

inline fn dequantOne(comptime dt: DType, src: []const u8, k: usize, scale: f32) f32 {
    return switch (dt) {
        .f8_e4m3 => dtypes.f8e4m3ToF32(src[k]) * scale,
        .bf16 => dtypes.bf16ToF32(std.mem.readInt(u16, src[k * 2 ..][0..2], .little)) * scale,
        .f16 => dtypes.f16ToF32(std.mem.readInt(u16, src[k * 2 ..][0..2], .little)) * scale,
        .f32 => blk: {
            const v: f32 = @bitCast(std.mem.readInt(u32, src[k * 4 ..][0..4], .little));
            break :blk v * scale;
        },
        else => unreachable,
    };
}

/// MRC x NR register tile over one packed subpanel k-slice, accumulating
/// into y (initialized from bias on the first k block).
fn microKernel(
    comptime mrc: usize,
    y: []f32,
    x: []const f32,
    sub: []const f32,
    t0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
    kc0: usize,
    kl: usize,
    bias: ?[]const f32,
    accumulate: bool,
) void {
    const full = col0 + NR <= rows;
    var acc: [mrc][2]Vec = undefined;
    inline for (0..mrc) |mi| {
        if (accumulate) {
            if (full) {
                acc[mi][0] = y[(t0 + mi) * rows + col0 ..][0..vlen].*;
                acc[mi][1] = y[(t0 + mi) * rows + col0 + vlen ..][0..vlen].*;
            } else {
                var tmp: [NR]f32 = @splat(0);
                for (col0..rows) |c| tmp[c - col0] = y[(t0 + mi) * rows + c];
                acc[mi][0] = tmp[0..vlen].*;
                acc[mi][1] = tmp[vlen..NR].*;
            }
        } else if (bias) |b| {
            var tmp: [NR]f32 = @splat(0);
            for (col0..@min(col0 + NR, rows)) |c| tmp[c - col0] = b[c];
            acc[mi][0] = tmp[0..vlen].*;
            acc[mi][1] = tmp[vlen..NR].*;
        } else {
            acc[mi][0] = @splat(0);
            acc[mi][1] = @splat(0);
        }
    }

    for (0..kl) |k| {
        const b0: Vec = sub[k * NR ..][0..vlen].*;
        const b1: Vec = sub[k * NR + vlen ..][0..vlen].*;
        inline for (0..mrc) |mi| {
            const a: Vec = @splat(x[(t0 + mi) * cols + kc0 + k]);
            acc[mi][0] = @mulAdd(Vec, a, b0, acc[mi][0]);
            acc[mi][1] = @mulAdd(Vec, a, b1, acc[mi][1]);
        }
    }

    inline for (0..mrc) |mi| {
        if (full) {
            y[(t0 + mi) * rows + col0 ..][0..vlen].* = acc[mi][0];
            y[(t0 + mi) * rows + col0 + vlen ..][0..vlen].* = acc[mi][1];
        } else {
            var tmp: [NR]f32 = undefined;
            tmp[0..vlen].* = acc[mi][0];
            tmp[vlen..NR].* = acc[mi][1];
            for (col0..rows) |c| y[(t0 + mi) * rows + c] = tmp[c - col0];
        }
    }
}

/// Round row chunks up to whole panels so tasks never split a panel.
fn chunkRows(rows: usize, want_tasks: usize) usize {
    const per_task = std.math.divCeil(usize, rows, want_tasks) catch unreachable;
    return std.mem.alignForward(usize, @max(per_task, panel_rows), panel_rows);
}

fn runRange(
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    scratch: []f32,
) void {
    switch (w.dtype) {
        inline .f8_e4m3, .bf16, .f16, .f32 => |dt| {
            runRangeTyped(dt, y, x, m, w, bias, row_start, row_end, scratch);
        },
        else => unreachable, // validated in matmul()
    }
}

fn runRangeTyped(
    comptime dt: DType,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    scratch: []f32,
) void {
    const cols = w.cols;
    var r = row_start;
    while (r < row_end) : (r += panel_rows) {
        const nr = @min(panel_rows, row_end - r);
        for (0..nr) |j| {
            const src = w.bytes[(r + j) * cols * comptime dt.byteSize() ..][0 .. cols * comptime dt.byteSize()];
            dequantRow(dt, scratch[j * cols ..][0..cols], src, w.scale);
        }
        for (0..m) |t| {
            const xrow = x[t * cols ..][0..cols];
            var acc: [panel_rows]Vec = @splat(@splat(0));
            var tail: [panel_rows]f32 = @splat(0);
            var k: usize = 0;
            while (k + vlen <= cols) : (k += vlen) {
                const xv: Vec = xrow[k..][0..vlen].*;
                inline for (0..panel_rows) |j| {
                    if (j < nr) {
                        const wv: Vec = scratch[j * cols + k ..][0..vlen].*;
                        acc[j] += xv * wv;
                    }
                }
            }
            while (k < cols) : (k += 1) {
                for (0..nr) |j| tail[j] += xrow[k] * scratch[j * cols + k];
            }
            for (0..nr) |j| {
                var sum = @reduce(.Add, acc[j]) + tail[j];
                if (bias) |b| sum += b[r + j];
                y[t * w.rows + r + j] = sum;
            }
        }
    }
}

fn dequantRow(comptime dt: DType, dst: []f32, src: []const u8, scale: f32) void {
    switch (dt) {
        .f32 => {
            @memcpy(std.mem.sliceAsBytes(dst), src);
            if (scale != 1.0) for (dst) |*v| {
                v.* *= scale;
            };
        },
        .f8_e4m3 => for (dst, src) |*v, b| {
            v.* = dtypes.f8e4m3ToF32(b) * scale;
        },
        .bf16 => for (dst, 0..) |*v, i| {
            v.* = dtypes.bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) * scale;
        },
        .f16 => for (dst, 0..) |*v, i| {
            v.* = dtypes.f16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) * scale;
        },
        else => unreachable,
    }
}

// --- tests ---------------------------------------------------------------

fn naiveMatmul(y: []f32, x: []const f32, m: usize, w_f32: []const f32, rows: usize, cols: usize, bias: ?[]const f32) void {
    for (0..m) |t| {
        for (0..rows) |r| {
            var sum: f64 = 0;
            for (0..cols) |c| sum += @as(f64, x[t * cols + c]) * w_f32[r * cols + c];
            if (bias) |b| sum += b[r];
            y[t * rows + r] = @floatCast(sum);
        }
    }
}

fn testAgainstNaive(m: usize, rows: usize, cols: usize, dt: DType, with_bias: bool, scale: f32) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const bias = if (with_bias) try gpa.alloc(f32, rows) else null;
    defer if (bias) |b| gpa.free(b);
    if (bias) |b| for (b) |*v| {
        v.* = rand.floatNorm(f32);
    };

    // Generate weight bytes in the storage dtype, plus the exact f32 values
    // they decode to (so the reference is bit-faithful).
    const wbytes = try gpa.alloc(u8, rows * cols * dt.byteSize());
    defer gpa.free(wbytes);
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (0..rows * cols) |i| {
        const v = rand.floatNorm(f32);
        switch (dt) {
            .f32 => {
                std.mem.writeInt(u32, wbytes[i * 4 ..][0..4], @bitCast(v), .little);
                w_f32[i] = v * scale;
            },
            .bf16 => {
                const b = dtypes.f32ToBf16(v);
                std.mem.writeInt(u16, wbytes[i * 2 ..][0..2], b, .little);
                w_f32[i] = dtypes.bf16ToF32(b) * scale;
            },
            .f8_e4m3 => {
                const b: u8 = rand.int(u8) & 0x7e; // avoid NaN encodings
                wbytes[i] = b;
                w_f32[i] = dtypes.f8e4m3ToF32(b) * scale;
            },
            else => unreachable,
        }
    }

    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);
    const y_ref = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_ref);

    var w = Weight.init(wbytes, dt, rows, cols);
    w.scale = scale;
    try matmul(io, gpa, y, x, m, w, bias);
    naiveMatmul(y_ref, x, m, w_f32, rows, cols, bias);

    for (y_ref, y) |e, a| {
        const tol = 1e-4 + 1e-5 * @abs(e) * @sqrt(@as(f32, @floatFromInt(cols)));
        try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "matmul f32 small" {
    try testAgainstNaive(3, 5, 7, .f32, true, 1.0);
}

test "matmul f32 vector tail and single token" {
    try testAgainstNaive(1, 9, vlen * 2 + 3, .f32, false, 1.0);
}

test "matmul bf16" {
    try testAgainstNaive(4, 17, 33, .bf16, true, 1.0);
}

test "matmul fp8 with scale" {
    try testAgainstNaive(2, 12, 40, .f8_e4m3, false, 0.03125);
}

test "matmul large enough to spawn tasks" {
    try testAgainstNaive(3, 257, 512, .f32, true, 1.0);
}

test "packed path basic" {
    try testAgainstNaive(32, 64, 128, .f32, true, 1.0);
}

test "packed path with mr, nr, and kc tails" {
    // m % MR != 0, rows % NR != 0, cols > KC with a partial last block.
    try testAgainstNaive(37, 61, KC + 33, .f32, true, 1.0);
    try testAgainstNaive(19, NR + 5, 70, .bf16, false, 1.0);
}

test "packed path fp8 with scale" {
    try testAgainstNaive(24, 90, 130, .f8_e4m3, true, 0.0625);
}

test "packed path single wide row block" {
    try testAgainstNaive(small_m_max, NR, KC * 2, .f32, false, 1.0);
}

test "matmul rejects unsupported dtype" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var y = [_]f32{0};
    const x = [_]f32{ 1, 2 };
    const wb = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const w = Weight.init(&wb, .i64, 1, 2);
    try std.testing.expectError(error.UnsupportedDType, matmul(io, gpa, &y, &x, 1, w, null));
}
