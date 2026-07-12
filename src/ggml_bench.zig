//! ggml-bench: systematic ours-vs-ggml CPU-kernel sweep. `zig build ggml-bench`.
//! Sections: decode GEMV (m=1) per dtype, prefill GEMV (m=128, both threaded),
//! whole-matrix dequant per dtype, fp16->fp32 row conversion. Single-threaded
//! where the op is memory-bound (per-core is the fair comparison); threaded for
//! the compute-bound prefill. Prints a ratio (>1 = ggml faster) per row.

const std = @import("std");
const tp = @import("TensorPencil");
const c = @import("ggml").c;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Kind = struct { dt: tp.DType, g: c.enum_ggml_type, name: []const u8 };
const dtypes = [_]Kind{
    .{ .dt = .q4_k, .g = c.GGML_TYPE_Q4_K, .name = "q4_k" },
    .{ .dt = .q5_k, .g = c.GGML_TYPE_Q5_K, .name = "q5_k" },
    .{ .dt = .q6_k, .g = c.GGML_TYPE_Q6_K, .name = "q6_k" },
    .{ .dt = .q8_0, .g = c.GGML_TYPE_Q8_0, .name = "q8_0" },
};

var rnd_state = std.Random.DefaultPrng.init(0x5EED_1234);
const rnd = rnd_state.random();
const p = std.debug.print;

fn best(comptime runs: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) void) u64 {
    var b: u64 = std.math.maxInt(u64);
    for (0..runs) |_| {
        const t0 = nowNs();
        f(ctx);
        b = @min(b, nowNs() - t0);
    }
    return b;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    c.ggml_cpu_init();

    try decodeSweep(gpa);
    try prefillSweep(gpa, io);
    try dequantSweep(gpa);
    try fp16Sweep(gpa);
}

// ---- Decode GEMV (m=1): our f32 dequant+dot vs ggml vec_dot ----------------
fn decodeSweep(gpa: std.mem.Allocator) !void {
    p("\n=== Decode GEMV m=1, 4096x4096 (single-thread, memory-bound) ===\n", .{});
    const rows = 4096;
    const cols = 4096;
    const x = try gpa.alloc(f32, cols);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    for (dtypes) |k| {
        const wf = try gpa.alloc(f32, rows * cols);
        for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.1;
        const row_b: usize = @intCast(c.ggml_row_size(k.g, @intCast(cols)));
        const q = try gpa.alloc(u8, rows * row_b);
        _ = c.ggml_quantize_chunk(k.g, wf.ptr, q.ptr, 0, @intCast(rows), @intCast(cols), null);

        // ggml activation
        const wt = c.ggml_get_type_traits_cpu(k.g);
        const vdt = wt.*.vec_dot_type;
        const vy_b: usize = @intCast(c.ggml_row_size(vdt, @intCast(cols)));
        const vy = try gpa.alloc(u8, vy_b);
        c.ggml_get_type_traits_cpu(vdt).*.from_float.?(x.ptr, vy.ptr, @intCast(cols));
        const y = try gpa.alloc(f32, rows);

        const GC = struct { q: []u8, vy: []u8, y: []f32, rows: usize, cols: usize, row_b: usize, vd: c.ggml_vec_dot_t };
        const gc = GC{ .q = q, .vy = vy, .y = y, .rows = rows, .cols = cols, .row_b = row_b, .vd = wt.*.vec_dot };
        const t_g = best(6, gc, struct {
            fn f(ctx: GC) void {
                for (0..ctx.rows) |r| ctx.vd.?(@intCast(ctx.cols), &ctx.y[r], 0, ctx.q.ptr + r * ctx.row_b, 0, ctx.vy.ptr, 0, 1);
            }
        }.f);

        // ours: f32 dequant + simd-ish dot, per row
        const OC = struct { q: []u8, x: []f32, y: []f32, rows: usize, cols: usize, row_b: usize, dt: tp.DType, scratch: []f32 };
        const oc = OC{ .q = q, .x = x, .y = y, .rows = rows, .cols = cols, .row_b = row_b, .dt = k.dt, .scratch = try gpa.alloc(f32, cols) };
        const t_o = best(6, oc, struct {
            fn f(ctx: OC) void {
                for (0..ctx.rows) |r| {
                    tp.quants.dequantSlice(ctx.dt, ctx.q[r * ctx.row_b ..][0..ctx.row_b], 0, ctx.cols, ctx.scratch);
                    var s: f32 = 0;
                    for (ctx.scratch, ctx.x) |a, b| s += a * b;
                    ctx.y[r] = s;
                }
            }
        }.f);
        report(k.name, "ours-f32-dequant", t_o, t_g, rows * row_b);
    }
}

// ---- Prefill GEMV (m=128): our threaded matmul vs threaded ggml -------------
fn prefillSweep(gpa: std.mem.Allocator, io: std.Io) !void {
    p("\n=== Prefill GEMV m=128, 4096x4096 (threaded, compute-bound) ===\n", .{});
    const rows = 4096;
    const cols = 4096;
    const M = 128;
    const wf = try gpa.alloc(f32, rows * cols);
    for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.1;
    const xm = try gpa.alloc(f32, M * cols);
    for (xm) |*v| v.* = rnd.floatNorm(f32);

    for (dtypes) |k| {
        const row_b: usize = @intCast(c.ggml_row_size(k.g, @intCast(cols)));
        const q = try gpa.alloc(u8, rows * row_b);
        _ = c.ggml_quantize_chunk(k.g, wf.ptr, q.ptr, 0, @intCast(rows), @intCast(cols), null);
        const w = tp.ops.matmul.Weight.init(q, k.dt, rows, cols);
        const y = try gpa.alloc(f32, M * rows);

        // ours (public matmul → threaded f32 packed path for m>=16)
        const OC = struct { io: std.Io, gpa: std.mem.Allocator, y: []f32, xm: []f32, w: tp.ops.matmul.Weight };
        const oc = OC{ .io = io, .gpa = gpa, .y = y, .xm = xm, .w = w };
        const t_o = best(3, oc, struct {
            fn f(ctx: OC) void {
                tp.ops.matmul.matmul(ctx.io, ctx.gpa, ctx.y, ctx.xm, 128, ctx.w, null) catch @panic("matmul");
            }
        }.f);

        // ggml's real batched mul_mat (tiled + its own threadpool).
        const nth: c_int = @intCast(@min(@as(usize, @intCast(std.Thread.getCpuCount() catch 8)), 16));
        const t_g = ggmlMulMat(k.g, q, xm, rows, cols, M, nth);
        report(k.name, "ours-threaded", t_o, t_g, 0);
    }
}

/// Time ggml's real ggml_mul_mat (tiled kernel + ggml threadpool): out[rows,M] =
/// W[cols,rows] · X[cols,M]. Weight `q` is row-major packed `gtype`.
fn ggmlMulMat(gtype: c.enum_ggml_type, q: []const u8, xm: []const f32, rows: usize, cols: usize, M: usize, nth: c_int) u64 {
    const mem: usize = q.len + M * cols * 4 + M * rows * 4 + (32 << 20);
    const ctx = c.ggml_init(.{ .mem_size = mem, .mem_buffer = null, .no_alloc = false });
    defer c.ggml_free(ctx);
    const w = c.ggml_new_tensor_2d(ctx, gtype, @intCast(cols), @intCast(rows));
    const x = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, @intCast(cols), @intCast(M));
    @memcpy(@as([*]u8, @ptrCast(w.*.data))[0..q.len], q);
    @memcpy(@as([*]u8, @ptrCast(x.*.data))[0 .. M * cols * 4], std.mem.sliceAsBytes(xm));
    const out = c.ggml_mul_mat(ctx, w, x);
    const graph = c.ggml_new_graph(ctx);
    c.ggml_build_forward_expand(graph, out);
    var b: u64 = std.math.maxInt(u64);
    for (0..4) |_| {
        const t0 = nowNs();
        _ = c.ggml_graph_compute_with_ctx(ctx, graph, nth);
        b = @min(b, nowNs() - t0);
    }
    return b;
}

// ---- Whole-matrix dequant: our dequantSlice vs ggml to_float ---------------
fn dequantSweep(gpa: std.mem.Allocator) !void {
    p("\n=== Dequant whole 4096x4096 matrix (single-thread) ===\n", .{});
    const rows = 4096;
    const cols = 4096;
    const out = try gpa.alloc(f32, rows * cols);
    for (dtypes) |k| {
        const wf = try gpa.alloc(f32, rows * cols);
        for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.1;
        const row_b: usize = @intCast(c.ggml_row_size(k.g, @intCast(cols)));
        const q = try gpa.alloc(u8, rows * row_b);
        _ = c.ggml_quantize_chunk(k.g, wf.ptr, q.ptr, 0, @intCast(rows), @intCast(cols), null);

        const OC = struct { q: []u8, out: []f32, rows: usize, cols: usize, row_b: usize, dt: tp.DType };
        const oc = OC{ .q = q, .out = out, .rows = rows, .cols = cols, .row_b = row_b, .dt = k.dt };
        const t_o = best(4, oc, struct {
            fn f(ctx: OC) void {
                for (0..ctx.rows) |r| tp.quants.dequantSlice(ctx.dt, ctx.q[r * ctx.row_b ..][0..ctx.row_b], 0, ctx.cols, ctx.out[r * ctx.cols ..][0..ctx.cols]);
            }
        }.f);

        const to_float = c.ggml_get_type_traits(k.g).*.to_float.?;
        const GC = struct { q: []u8, out: []f32, rows: usize, cols: usize, row_b: usize, tf: c.ggml_to_float_t };
        const gc = GC{ .q = q, .out = out, .rows = rows, .cols = cols, .row_b = row_b, .tf = to_float };
        const t_g = best(4, gc, struct {
            fn f(ctx: GC) void {
                for (0..ctx.rows) |r| ctx.tf.?(ctx.q.ptr + r * ctx.row_b, ctx.out[r * ctx.cols ..].ptr, @intCast(ctx.cols));
            }
        }.f);
        report(k.name, "ours-dequantSlice", t_o, t_g, rows * cols * 4);
    }
}

// ---- fp16 -> fp32 row conversion -------------------------------------------
fn fp16Sweep(gpa: std.mem.Allocator) !void {
    p("\n=== fp16->fp32 row, 16M elems (single-thread) ===\n", .{});
    const n = 16 << 20;
    const in = try gpa.alloc(u16, n);
    for (in) |*v| v.* = rnd.int(u16);
    const out = try gpa.alloc(f32, n);

    const OC = struct { in: []u16, out: []f32 };
    const oc = OC{ .in = in, .out = out };
    const t_o = best(4, oc, struct {
        fn f(ctx: OC) void {
            tp.dtype.f16ToF32Row(std.mem.sliceAsBytes(ctx.in), ctx.out, 1.0);
        }
    }.f);

    const GC = struct { in: []u16, out: []f32, n: usize };
    const gc = GC{ .in = in, .out = out, .n = n };
    const t_g = best(4, gc, struct {
        fn f(ctx: GC) void {
            c.ggml_fp16_to_fp32_row(@ptrCast(ctx.in.ptr), ctx.out.ptr, @intCast(ctx.n));
        }
    }.f);
    report("f16", "ours-f16ToF32", t_o, t_g, n * 2);
}

fn report(name: []const u8, ours_label: []const u8, t_o: u64, t_g: u64, bytes: usize) void {
    const o_ms = @as(f64, @floatFromInt(t_o)) / 1e6;
    const g_ms = @as(f64, @floatFromInt(t_g)) / 1e6;
    const ratio = @as(f64, @floatFromInt(t_o)) / @as(f64, @floatFromInt(t_g));
    if (bytes > 0) {
        const o_gbs = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(t_o));
        const g_gbs = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(t_g));
        p("  {s:<5} {s:<18} {d:8.3} ms ({d:5.1} GB/s)  |  ggml {d:8.3} ms ({d:5.1} GB/s)  |  ggml {d:6.2}x\n", .{ name, ours_label, o_ms, o_gbs, g_ms, g_gbs, ratio });
    } else {
        p("  {s:<5} {s:<18} {d:8.3} ms  |  ggml {d:8.3} ms  |  ggml {d:6.2}x\n", .{ name, ours_label, o_ms, g_ms, ratio });
    }
}
