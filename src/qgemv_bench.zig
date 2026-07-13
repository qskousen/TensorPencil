//! qgemv-bench: grouped-N dp4a quant GEMV vs dequant-to-f16 GEMM, on device.
//! `zig build qgemv-bench`. Answers the question behind "grouped-N": is the
//! grouped multi-input GEMV a real gain over the current m>1 fallback
//! (dequant the whole weight to f16, then a tensor-core GEMM) for the small
//! batches a speculative-verify pass produces — and where does the crossover
//! into the GEMM sit?
//!
//! Both paths are timed with the backend's own sync-per-op CUDA-event
//! profiler (`be.profile` -> `be.prof.ms`), the same methodology the plan's
//! GEMV numbers were measured with. GEMV "grouped ms" is the matmul-bucket
//! time only; the one-time activation quantize (`opGemvQuantizeX`) is reported
//! separately because in a real layer it amortizes across ~7 weights.

const std = @import("std");
const tp = @import("TensorPencil");
const c = @import("ggml").c;

const Backend = tp.gpu.cuda.Backend;
const DeviceBuffer = tp.gpu.cuda.backend.DeviceBuffer;
const Cat = Backend.ProfCat;

const p = std.debug.print;
var rnd_state = std.Random.DefaultPrng.init(0x0B0A710C);
const rnd = rnd_state.random();

const Shape = struct { rows: usize, cols: usize, name: []const u8 };
// rows multiple of 128 (hgemm), cols multiple of 256 (dp4a grouped GEMV).
const shapes = [_]Shape{
    .{ .rows = 4096, .cols = 4096, .name = "4096 x 4096   (attn o_proj)" },
    .{ .rows = 11008, .cols = 4096, .name = "11008 x 4096  (mlp gate/up)" },
    .{ .rows = 4096, .cols = 11008, .name = "4096 x 11008  (mlp down)" },
};
// n=1 is decode; 2-8 is a chain/tree verify batch; 16-64 spans into
// prefill-chunk territory where the GEMM is expected to win.
const batches = [_]usize{ 1, 2, 4, 8, 16, 32, 48, 64, 128 };

const Kind = struct { dt: tp.DType, g: c.enum_ggml_type, name: []const u8 };
// All block-quants now have grouped dp4a kernels.
const kinds = [_]Kind{
    .{ .dt = .q4_k, .g = c.GGML_TYPE_Q4_K, .name = "q4_k" },
    .{ .dt = .q5_k, .g = c.GGML_TYPE_Q5_K, .name = "q5_k" },
    .{ .dt = .q6_k, .g = c.GGML_TYPE_Q6_K, .name = "q6_k" },
    .{ .dt = .q8_0, .g = c.GGML_TYPE_Q8_0, .name = "q8_0" },
};

/// DeviceBuffer sub-view (mirrors qwen*_cuda.zig's private offsetBufSized).
fn offBuf(b: DeviceBuffer, off_bytes: usize, size: u64) DeviceBuffer {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = size };
}

fn grouped(be: *Backend, dt: tp.DType, x_d: DeviceBuffer, y_d: DeviceBuffer, q: []const u8, n: usize, rows: usize, cols: usize) void {
    be.opGemvQuantizeX(x_d, n * cols) catch @panic("quantizeX");
    var off: usize = 0;
    while (off < n) : (off += 8) {
        const ng: usize = @min(8, n - off); // usize annotation: @min range-narrows (ZIG.md)
        be.opGemvQuantQ8N(dt, offBuf(y_d, off * rows * 4, ng * rows * 4), q, 1.0, rows, cols, ng, off, n) catch @panic("q8n");
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    c.ggml_cpu_init();

    const be = Backend.init(gpa) catch {
        p("qgemv-bench: no CUDA device (Backend.init failed) — skipping.\n", .{});
        return;
    };
    defer be.deinit();
    be.profile = true;

    const warmup = 10; // settle clocks + JIT PTX + upload weight (cachedWeight)
    const iters = 40;

    p("grouped-N dp4a GEMV vs dequant->f16 GEMM   (speedup = gemm/grouped; >1 => grouped wins)\n", .{});

    for (shapes) |sh| {
        const rows = sh.rows;
        const cols = sh.cols;
        p("\n=== {s} ===\n", .{sh.name});
        p("  {s:<5} {s:>4}  {s:>11} {s:>10} {s:>10}  {s:>8}  {s:>10}\n", .{ "dt", "n", "grouped ms", "quant ms", "gemm ms", "speedup", "gemv GB/s" });

        const wf = try gpa.alloc(f32, rows * cols);
        for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.1;

        for (kinds) |k| {
            const row_b: usize = @intCast(c.ggml_row_size(k.g, @intCast(cols)));
            const q = try gpa.alloc(u8, rows * row_b);
            _ = c.ggml_quantize_chunk(k.g, wf.ptr, q.ptr, 0, @intCast(rows), @intCast(cols), null);
            const w_bytes = rows * row_b;

            for (batches) |n| {
                const x_d = try be.tensorCreate(n * cols * 4);
                // opMatmulQuant pads m up to 128 and its hgemm writes the full
                // padded row count, so y must be sized for that, not just n.
                const y_d = try be.tensorCreate(std.mem.alignForward(usize, n, 128) * rows * 4);
                const xh = try gpa.alloc(f32, n * cols);
                for (xh) |*v| v.* = rnd.floatNorm(f32);
                try be.tensorUpload(x_d, std.mem.sliceAsBytes(xh));

                for (0..warmup) |_| {
                    grouped(be, k.dt, x_d, y_d, q, n, rows, cols);
                    be.opMatmulQuant(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("gemm");
                }

                be.prof.reset();
                for (0..iters) |_| grouped(be, k.dt, x_d, y_d, q, n, rows, cols);
                const grouped_ms = be.prof.ms[@intFromEnum(Cat.matmul)] / iters;
                const quant_ms = be.prof.ms[@intFromEnum(Cat.elt)] / iters;

                be.prof.reset();
                for (0..iters) |_| be.opMatmulQuant(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("gemm");
                const gemm_ms = be.prof.ms[@intFromEnum(Cat.matmul)] / iters;

                const passes = (n + 7) / 8;
                const gbs = @as(f64, @floatFromInt(w_bytes * passes)) / (grouped_ms * 1e6);
                const speedup = gemm_ms / grouped_ms;
                p("  {s:<5} {d:>4}  {d:>11.4} {d:>10.4} {d:>10.4}  {d:>7.2}x  {d:>10.1}\n", .{ k.name, n, grouped_ms, quant_ms, gemm_ms, speedup, gbs });

                var xd = x_d;
                var yd = y_d;
                be.tensorDestroy(&xd);
                be.tensorDestroy(&yd);
            }
        }
    }
}
