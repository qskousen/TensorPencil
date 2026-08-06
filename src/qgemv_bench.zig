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
    // gemma4 31B (hidden 5376, ffn 21504) — the prefill shapes that matter here.
    .{ .rows = 21504, .cols = 5376, .name = "21504 x 5376  (g4 mlp gate/up)" },
    .{ .rows = 5376, .cols = 21504, .name = "5376 x 21504  (g4 mlp down)" },
    // Bonsai-27B / qwen35 (hidden 5120, ffn 17408) — the q1_0 prefill shapes.
    .{ .rows = 17408, .cols = 5120, .name = "17408 x 5120  (b27 mlp gate/up)" },
    .{ .rows = 5120, .cols = 17408, .name = "5120 x 17408  (b27 mlp down)" },
    // The shape ggml's own `test-backend-ops perf -o MUL_MAT` reports, so our MMQ
    // can be compared to llama.cpp's number apples-to-apples instead of inferred
    // from an end-to-end rate. Theirs: 86.50 TFLOPS at n=512 on a 3090.
    .{ .rows = 4096, .cols = 14336, .name = "4096 x 14336  (ggml perf shape)" },
};
// n=1 is decode; 2-8 is a chain/tree verify batch; 16-64 spans into
// prefill-chunk territory where the GEMM is expected to win.
const batches = [_]usize{ 1, 8, 32, 128, 256, 512 };

const Kind = struct { dt: tp.DType, g: c.enum_ggml_type, name: []const u8 };
// Every block-quant here has a grouped dp4a kernel EXCEPT q1_0 (see `grouped`).
const kinds = [_]Kind{
    .{ .dt = .q4_k, .g = c.GGML_TYPE_Q4_K, .name = "q4_k" },
    .{ .dt = .q5_k, .g = c.GGML_TYPE_Q5_K, .name = "q5_k" },
    .{ .dt = .q6_k, .g = c.GGML_TYPE_Q6_K, .name = "q6_k" },
    .{ .dt = .q8_0, .g = c.GGML_TYPE_Q8_0, .name = "q8_0" },
    .{ .dt = .q1_0, .g = c.GGML_TYPE_Q1_0, .name = "q1_0" },
};

/// Whether `grouped` has a kernel for this dtype (`opGemvQuantQ8N`). q1_0 has no
/// grouped GEMV — it went straight to MMQ for prefill — so the grouped column is
/// skipped rather than panicking.
fn hasGroupedKernel(dt: tp.DType) bool {
    return dt != .q1_0;
}

/// Int8 tensor-core MACs for one GEMM, as TOPS given a millisecond timing —
/// the distance from the 3090's ~284 dense TOPS is what says whether a kernel is
/// worth more work or already near the machine.
fn tops(rows: usize, cols: usize, n: usize, ms: f64) f64 {
    if (ms <= 0) return 0;
    const ops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) * @as(f64, @floatFromInt(n));
    return ops / (ms * 1e-3) / 1e12;
}

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
        p("  {s:<5} {s:>4} {s:>9} {s:>9}  {s:>8} {s:>8}  {s:>8} {s:>9}  {s:>8} {s:>6}\n", .{ "dt", "n", "dequant", "f16 gemm", "deq+gemm", "mmq v1", "pipe ms", "pipe rel", "pipe spd", "TOPS" });

        const wf = try gpa.alloc(f32, rows * cols);
        defer gpa.free(wf);
        for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.1;

        for (kinds) |k| {
            const row_b: usize = @intCast(c.ggml_row_size(k.g, @intCast(cols)));
            const q = try gpa.alloc(u8, rows * row_b);
            _ = c.ggml_quantize_chunk(k.g, wf.ptr, q.ptr, 0, @intCast(rows), @intCast(cols), null);

            for (batches) |n| {
                const x_d = try be.tensorCreate(n * cols * 4);
                // opMatmulQuant pads m up to 128 and its hgemm writes the full
                // padded row count, so y must be sized for that, not just n.
                const y_d = try be.tensorCreate(std.mem.alignForward(usize, n, 128) * rows * 4);
                const xh = try gpa.alloc(f32, n * cols);
                for (xh) |*v| v.* = rnd.floatNorm(f32);
                try be.tensorUpload(x_d, std.mem.sliceAsBytes(xh));

                for (0..warmup) |_| {
                    if (hasGroupedKernel(k.dt)) grouped(be, k.dt, x_d, y_d, q, n, rows, cols);
                    be.opMatmulQuant(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("gemm");
                }

                be.prof.reset();
                if (hasGroupedKernel(k.dt))
                    for (0..iters) |_| grouped(be, k.dt, x_d, y_d, q, n, rows, cols);
                const grouped_ms = be.prof.ms[@intFromEnum(Cat.matmul)] / iters;
                const quant_ms = be.prof.ms[@intFromEnum(Cat.elt)] / iters;

                // MMQ (q4_k only for now): correctness vs the dequant+GEMM path,
                // then timing. Not bit-exact — the activations go through int8 —
                // so compare with a relative-error bound over the whole output.
                var mmq_ms: f64 = 0;
                var mmq_rel: f64 = -1;
                var pipe_ms: f64 = 0;
                var pipe_rel: f64 = -1;
                var den: f64 = 0;
                if (Backend.mmqPipeSupported(k.dt, rows, cols)) {
                    be.opMatmulQuant(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("gemm");
                    const ref = try gpa.alloc(f32, n * rows);
                    defer gpa.free(ref);
                    try be.tensorDownload(y_d, std.mem.sliceAsBytes(ref));
                    if (Backend.mmqSupported(k.dt, rows, cols)) be.opMatmulQuantMmq(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmq");
                    const got = try gpa.alloc(f32, n * rows);
                    defer gpa.free(got);
                    try be.tensorDownload(y_d, std.mem.sliceAsBytes(got));
                    var num: f64 = 0;
                    den = 0;
                    for (ref, got) |r, g| {
                        num += @abs(@as(f64, r) - @as(f64, g));
                        den += @abs(@as(f64, r));
                    }
                    mmq_rel = if (den > 0) num / den else 0;

                    if (Backend.mmqSupported(k.dt, rows, cols)) {
                        for (0..warmup) |_| be.opMatmulQuantMmq(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmq");
                        be.prof.reset();
                        for (0..iters) |_| be.opMatmulQuantMmq(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmq");
                        mmq_ms = (be.prof.ms[@intFromEnum(Cat.matmul)] + be.prof.ms[@intFromEnum(Cat.elt)]) / iters;
                    }

                    // Pipe-tiled MMQ (128x128, MT=4/NT=8) — needs rows % 128 == 0.
                    if (rows % Backend.mmq_pipe_tile == 0) {
                        be.opMatmulQuantMmqPipe(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmqpipe");
                        const got2 = try gpa.alloc(f32, n * rows);
                        defer gpa.free(got2);
                        try be.tensorDownload(y_d, std.mem.sliceAsBytes(got2));
                        var num2: f64 = 0;
                        for (ref, got2) |r, g| num2 += @abs(@as(f64, r) - @as(f64, g));
                        pipe_rel = if (den > 0) num2 / den else 0;

                        for (0..warmup) |_| be.opMatmulQuantMmqPipe(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmqpipe");
                        be.prof.reset();
                        for (0..iters) |_| be.opMatmulQuantMmqPipe(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("mmqpipe");
                        pipe_ms = (be.prof.ms[@intFromEnum(Cat.matmul)] + be.prof.ms[@intFromEnum(Cat.elt)]) / iters;
                    }
                }

                be.prof.reset();
                for (0..iters) |_| be.opMatmulQuant(k.dt, y_d, x_d, n, q, rows, cols) catch @panic("gemm");
                // opMatmulQuant now charges the weight expansion to .dequant and
                // the activation convert + tensor-core GEMM to .matmul, so the
                // two halves of the m>1 fallback are separable.
                const deq_ms = be.prof.ms[@intFromEnum(Cat.dequant)] / iters;
                const mm_ms = be.prof.ms[@intFromEnum(Cat.matmul)] / iters;
                const gemm_ms = deq_ms + mm_ms;

                _ = grouped_ms;
                _ = quant_ms;
                const pipe_spd = if (pipe_ms > 0) gemm_ms / pipe_ms else 0;
                p("  {s:<5} {d:>4} {d:>9.4} {d:>9.4}  {d:>8.4} {d:>8.4}  {d:>8.4} {d:>9.5}  {d:>7.2}x  {d:>6.1}  (v1 rel {d:.5})\n", .{ k.name, n, deq_ms, mm_ms, gemm_ms, mmq_ms, pipe_ms, pipe_rel, pipe_spd, tops(rows, cols, n, pipe_ms), mmq_rel });

                var xd = x_d;
                var yd = y_d;
                be.tensorDestroy(&xd);
                be.tensorDestroy(&yd);
            }
        }
    }
}
