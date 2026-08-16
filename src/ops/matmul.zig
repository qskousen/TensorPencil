//! Dtype-aware GEMM: y[m, rows] = x[m, cols] @ W^T (+ bias), W stored
//! row-major [rows, cols] as in torch Linear / safetensors.
//!
//! Weights stay in their storage dtype (fp8-e4m3 / bf16 / f16 / f32) and are
//! dequantized into small f32 row panels inside the kernel, so a 12 GiB fp8
//! checkpoint never expands in memory. Work is split over output rows across
//! `std.Io.Group` tasks; accumulation is f32 SIMD.

const std = @import("std");
const dtypes = @import("tp_core").dtype;
const quants = @import("tp_core").quants;
const convrot_mod = @import("convrot.zig");
const w4a8_mod = @import("w4a8.zig");
const nvfp4_mod = @import("nvfp4.zig");
const prof = @import("tp_core").prof;
const cancel = @import("cancel.zig");

/// Whether ggml (the GGUF block-quant CPU backend) was linked (`-Dggml`).
/// When false, block-quant weights return `error.QuantBackendUnavailable`.
const have_ggml = quants.have_ggml;

const DType = dtypes.DType;

/// GPU GEMM dispatch, injected by the pipeline. Dependency injection keeps this
/// CPU-ops layer from importing the GPU backend: the pipeline sets `gpu_dispatch`
/// (pipeline --gpu) and large f8/f32 GEMMs are routed to the device via
/// `call(ctx, ...)`. Single-threaded use only, the pipeline runs matmuls
/// sequentially.
pub const GpuDispatch = struct {
    /// Opaque device backend (e.g. a `*gpu.Context`); passed back to `call`.
    ctx: *anyopaque,
    /// Runs y[m, rows] = x[m, cols] @ W^T (+bias) on the device. `dtype_f8`
    /// selects the fp8-e4m3 weight path (else f32). Mirrors gpu `Context.matmul`;
    /// `anyerror` erases the backend's error set so this layer stays agnostic.
    call: *const fn (
        ctx: *anyopaque,
        y: []f32,
        x: []const f32,
        m: usize,
        w_bytes: []const u8,
        dtype_f8: bool,
        rows: usize,
        cols: usize,
        scale: f32,
        bias: ?[]const f32,
    ) anyerror!void,
};

/// Injected by the pipeline; null = CPU only.
pub var gpu_dispatch: ?GpuDispatch = null;

/// Observer called with the INPUT of every GEMM, before it runs. The point is
/// per-layer attribution: `w.tag` names the weight, so a caller can accumulate
/// statistics of the activations each layer actually sees (quantization
/// calibration), time or count GEMMs per layer (profiling), or dump intermediates
/// (debugging) without any of that living in the kernels.
///
/// `x` is `[m, w.cols]` row-major and is only valid for the duration of the call,
/// copy anything you keep. The hook must not mutate it.
pub const Probe = struct {
    ctx: *anyopaque,
    input: *const fn (ctx: *anyopaque, w: Weight, x: []const f32, m: usize) void,
};

/// Installed by a caller; null = no observation, zero cost. Single-threaded use
/// only, exactly like `gpu_dispatch` above: it is read once per `matmul` on the
/// calling thread, so callers that run matmuls concurrently must synchronize the
/// hook themselves.
///
/// Note this sees only GEMMs that go THROUGH this function: the CPU path, plus
/// whatever the GPU dispatch above forwards. Backends with their own GEMM bypass it
/// and carry their own probe points, both DiT backends now do:
///
/// - `dit_cuda`: `lin` (all 224 block linears), `txtGemm` (the txtfusion tower and
///   both txtmlp linears), and the two patch-embed GEMMs.
/// - `dit_gpu`: `Gemm.go` plus the two patch-embed GEMMs; a non-null `probe` also
///   forces the f32 GEMM path there, because the cooperative-matrix fp8 fast path
///   bypasses `Gemm.go` entirely and would feed f16 activations.
///
/// Each stages the activation to host memory and calls this same hook, so the
/// statistics come out of one implementation whichever backend ran the model.
/// int8/int4 checkpoints are skipped on both: their activation is rotated and
/// quantized in place before the GEMM, so there is no f32 activation left to
/// observe.
pub var probe: ?Probe = null;

/// Force block-quantized GEMMs down the exact-activation path at every `m`.
///
/// The CPU has two block-quant implementations that differ in *numerics*, not
/// just speed (see `small_m_max`). At large `m` the weights are dequantized into
/// a packed panel and multiplied in f32, so only the weights are quantized. At
/// small `m` the work is memory-bound instead of compute-bound, and ggml's
/// `vec_dot` GEMV wins by a wide margin, but it reaches that speed by also
/// quantizing the activations to the weight type's `vec_dot_type` (Q8_K for
/// k-quants), which introduces activation quantization error the packed path
/// does not have.
///
/// That trade is the right default for inference. Set this when the *quantity
/// being computed* matters more than throughput: measuring weight-format loss in
/// isolation, comparing a small-`m` result against a large-`m` one, or chasing a
/// numerical discrepancy across backends. Non-block dtypes are unaffected.
///
/// Single-threaded, like `probe` and `gpu_dispatch`: set it around the calls you
/// care about and restore it afterwards.
pub var exact_activations: bool = false;

/// Minimum FLOP count before the GPU path is worth the PCIe round trip.
pub var gpu_min_flops: usize = 1 << 31;

fn gpuEligible(m: usize, w: Weight) bool {
    if (gpu_dispatch == null) return false;
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

/// Token count at which the CPU path switches to the packed outer-product path.
///
/// Public because it changes the *semantics* of a block-quant GEMM, not just its
/// speed: at or above this, block-quant weights are dequantized and multiplied in
/// f32, so only the weights are quantized. Below it, they take ggml's int8
/// `vec_dot` GEMV, which quantizes the activations too, a W8A8-shaped
/// computation. A caller measuring weight-format loss in isolation has to know
/// which side of this line it is on, so it has to be able to read the line.
pub const small_m_max = 16;

/// A weight matrix view over raw checkpoint bytes.
pub const Weight = struct {
    bytes: []const u8,
    dtype: DType,
    rows: usize,
    cols: usize,
    /// Per-tensor dequant scale (ComfyUI fp8 format stores one per weight).
    scale: f32 = 1.0,
    /// Per-output-row dequant scale (ComfyUI int8 `weight_scale`, `[rows]`).
    /// When set, overrides `scale` for `.i8`/`.i4` weights.
    row_scale: ?[]const f32 = null,
    /// ComfyUI `asym_w4a8_int8` sidecars, set iff `dtype == .w4a8`. `bytes` is then
    /// the packed `[rows][cols/2]` nibble storage and `row_scale` is `s_channel`;
    /// decoding a nibble needs the per-group scale and level table in here. Kept
    /// packed on purpose, materializing the int8 doubles the footprint, which is
    /// the entire difference between this format and int8. See `ops/w4a8.zig`.
    w4a8: ?*const w4a8_mod.Meta = null,
    /// ComfyUI NVFP4 sidecars, set iff `dtype == .nvfp4`. `bytes` is then the packed
    /// `[rows][cols/2]` E2M1 storage. Unlike every other 4-bit format here this one has
    /// NO per-output-row scale and NO ConvRot rotation, `row_scale` stays null and
    /// `convrot` 0, because its scale is per 16-element block plus one per tensor. See
    /// `ops/nvfp4.zig`.
    nvfp4: ?*const nvfp4_mod.Meta = null,
    /// ConvRot group size (0 = none). When non-zero, `.i8`/`.i4`/`.w4a8` weights are
    /// stored rotated by a group-wise Hadamard along the input dim and are
    /// un-rotated at dequant time; `cols` must be a multiple of this. i4 packs
    /// two values per byte so `cols` is also even. See ops/convrot.zig.
    convrot: u32 = 0,
    /// Checkpoint tensor name, when the loader knows it (e.g.
    /// "blocks.12.attn.wq.weight"). Purely informational to the kernels, it is
    /// what lets a caller attribute a GEMM to a layer: the `probe` hook below,
    /// profiling breakdowns, and error messages. Owned by whoever built the
    /// Weight (model loaders allocate it in the model arena).
    tag: ?[]const u8 = null,

    pub fn init(bytes: []const u8, dtype: DType, rows: usize, cols: usize) Weight {
        std.debug.assert(bytes.len == dtype.storageBytes(rows * cols));
        return .{ .bytes = bytes, .dtype = dtype, .rows = rows, .cols = cols };
    }

    /// Convenience for tests / f32 weights already in memory.
    pub fn fromF32(data: []const f32, rows: usize, cols: usize) Weight {
        return init(std.mem.sliceAsBytes(data), .f32, rows, cols);
    }
};

pub const Error = error{ UnsupportedDType, QuantBackendUnavailable, OutOfMemory } || std.Io.Cancelable;

/// Whether `matmul` can take a weight of this dtype at all, the same set its own
/// validation switch accepts, exposed so a *loader* can decide up front instead of
/// discovering it mid-forward.
///
/// A model loader that meets an unsupported dtype has one correct move: materialize
/// the weight to f32 once, at load. Checkpoints in the wild do carry surprises (an
/// SD1.5 merge with f64 CLIP linears is the case that prompted this), and the
/// alternative, a `UnsupportedDType` from inside the first forward, reports the
/// problem at the point furthest from its cause.
pub fn supportsDType(dt: DType) bool {
    return switch (dt) {
        .f8_e4m3, .bf16, .f16, .f32, .i8, .i4, .w4a8, .nvfp4 => true,
        .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs, .q1_0, .q2_0_g64 => have_ggml,
        // q2_0_g128 decodes natively (quants.dequantQ2_0G128), so it needs no ggml.
        .q2_0_g128 => true,
        else => false,
    };
}

/// Materialize a weight to f32 once, into `alloc`, folding any per-tensor scale.
///
/// The GPU backends' fused `opMatmul` (the one with bias + destination-offset
/// support, used for a model's small in/out projections) has an f32 pipeline
/// only, handed bf16 bytes it reinterprets them as f32 and the render is pure
/// noise, and handed fp8 it trips an assert. So a loader normalizes those few
/// projections here rather than discovering the problem inside the first forward.
///
/// - `f32` passes through untouched, so the device is fed the original mmap.
/// - `f8_e4m3` / `bf16` / `f16` are converted (fp8 too: the CUDA fused kernel has
///   no fp8 variant).
/// - Anything else, int8/int4 convrot, ggml block quants, refuses loudly.
///   Dequantizing those needs metadata this function does not have (a per-row
///   scale and group rotation, or a ggml block layout), so converting them here
///   would emit silent garbage rather than an error.
///
/// The `tag` survives the copy: materializing must not make a weight
/// unattributable to its checkpoint tensor (`Weight.tag`).
pub fn materializeF32(alloc: std.mem.Allocator, w: Weight) !Weight {
    switch (w.dtype) {
        .f32 => return w,
        .f8_e4m3, .bf16, .f16 => {
            const out = try alloc.alloc(f32, w.rows * w.cols);
            try @import("tp_core").safetensors.convertToF32(w.dtype, w.bytes, out);
            if (w.scale != 1.0) for (out) |*v| {
                v.* *= w.scale;
            };
            var out_w = Weight.fromF32(out, w.rows, w.cols);
            out_w.tag = w.tag;
            return out_w;
        },
        else => return error.UnsupportedCheckpoint,
    }
}

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
    const _pt = prof.tic();
    defer prof.toc(.matmul, _pt);
    std.debug.assert(x.len == m * w.cols);
    std.debug.assert(y.len == m * w.rows);
    if (bias) |b| std.debug.assert(b.len == w.rows);
    switch (w.dtype) {
        .f8_e4m3, .bf16, .f16, .f32, .i8, .i4, .w4a8, .nvfp4 => {},
        .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs, .q1_0, .q2_0_g64 => {
            if (have_ggml) {
                // ggml rows are whole blocks; block-aligned k-slicing depends on it.
                std.debug.assert(w.cols % w.dtype.blockElems() == 0);
            } else {
                // Built with -Dggml=false: block-quant has no CPU backend.
                return error.QuantBackendUnavailable;
            }
        },
        // Rows are whole blocks here too, but the decode is ours, not ggml's.
        .q2_0_g128 => std.debug.assert(w.cols % w.dtype.blockElems() == 0),
        else => return error.UnsupportedDType,
    }
    if (w.dtype == .i8 or w.dtype == .i4 or w.dtype == .w4a8)
        std.debug.assert(w.row_scale != null and w.row_scale.?.len == w.rows);
    // A `.w4a8` weight without its sidecars cannot be decoded at all, and the nibbles
    // would otherwise be read as signed int4, plausible values, wrong weight.
    if (w.dtype == .w4a8) std.debug.assert(w.w4a8 != null);
    // NVFP4 carries no row scale and no rotation; its block scales and level table are
    // the only way to read the nibbles at all.
    if (w.dtype == .nvfp4) {
        std.debug.assert(w.nvfp4 != null);
        std.debug.assert(w.row_scale == null and w.convrot == 0);
    }
    if (m == 0 or w.rows == 0) return;

    // Activation observer (see `probe`): after validation, before any dispatch, so
    // it sees the same `x` every backend path would.
    if (probe) |p| p.input(p.ctx, w, x, m);

    // Cooperative cancel (see ops/cancel.zig): captured once here on the
    // calling thread, polled by the row-chunk tasks so a cancel lands
    // mid-GEMM (a big DiT MLP GEMM is seconds of CPU work).
    const tok = cancel.token;

    if (gpuEligible(m, w)) {
        const gd = gpu_dispatch.?;
        if (gd.call(gd.ctx, y, x, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, bias)) |_| {
            return;
        } else |err| {
            // Fall back to CPU once and stop routing.
            std.log.warn("gpu matmul failed ({t}); falling back to cpu", .{err});
            gpu_dispatch = null;
        }
    }

    // `exact_activations` routes block-quant to the packed path even in the
    // small-m regime, trading the GEMV's speed for f32 activations. The packed
    // path handles any m >= 1 (its MR tiling clamps on the tail), so this is a
    // performance choice, not a correctness one.
    if (m >= small_m_max or (exact_activations and w.dtype.isBlockQuant()))
        return matmulPacked(io, gpa, y, x, m, w, bias, tok);

    // Decode (small m) block-quant GEMV -> ggml's AVX2 quant vec_dot, which is
    // memory-bound (~30x faster than our Zig dequant/int8 kernels). Non-block-
    // quant small-m falls through to the threaded runRange path below.
    //
    // Gated on `quants.usesGgml`, not `isBlockQuant`: q2_0_g128 has no ggml
    // type (its blocks are 128 elements where ggml's type 42 is 64), so it takes
    // the native fused GEMV instead of a call into ggml with the wrong stride.
    if (w.dtype.isBlockQuant() and quants.usesGgml(w.dtype)) {
        if (have_ggml) {
            return ggml_gemv.quantGemv(io, w.dtype, y, x, m, w.bytes, w.rows, w.cols, bias);
        } else {
            return error.QuantBackendUnavailable;
        }
    }
    if (w.dtype == .q2_0_g128) return native_gemv.q2Gemv(io, y, x, m, w.bytes, w.rows, w.cols, bias);
    if (w.dtype.isBlockQuant()) return matmulPacked(io, gpa, y, x, m, w, bias, tok);

    // Small problems are not worth the fork/join overhead.
    const flops = 2 * m * w.rows * w.cols;
    const n_threads = std.Thread.getCpuCount() catch 1;
    const want_tasks: usize = if (flops < (1 << 20) or n_threads == 1) 1 else 4 * n_threads;

    const chunk = chunkRows(w.rows, want_tasks);
    const n_tasks = std.math.divCeil(usize, w.rows, chunk) catch unreachable;

    const scratch = try gpa.alloc(f32, n_tasks * panel_rows * w.cols);
    defer gpa.free(scratch);

    if (n_tasks == 1) {
        runRange(y, x, m, w, bias, 0, w.rows, scratch, tok);
        if (cancel.canceled(tok)) return error.Canceled;
        return;
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    var task: usize = 0;
    var row: usize = 0;
    while (row < w.rows) : (row += chunk) {
        const row_end = @min(row + chunk, w.rows);
        const task_scratch = scratch[task * panel_rows * w.cols ..][0 .. panel_rows * w.cols];
        group.async(io, runRange, .{ y, x, m, w, bias, row, row_end, task_scratch, tok });
        task += 1;
    }
    try group.await(io);
    if (cancel.canceled(tok)) return error.Canceled;
}

fn matmulPacked(
    io: std.Io,
    gpa: std.mem.Allocator,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    tok: cancel.Token,
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
        packedTask(y, x, m, w, bias, 0, w.rows, scratch[0..stride], tok);
        if (cancel.canceled(tok)) return error.Canceled;
        return;
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    var task: usize = 0;
    var row: usize = 0;
    while (row < w.rows) : (row += chunk) {
        const row_end = @min(row + chunk, w.rows);
        group.async(io, packedTask, .{ y, x, m, w, bias, row, row_end, scratch[task * stride ..][0..stride], tok });
        task += 1;
    }
    try group.await(io);
    if (cancel.canceled(tok)) return error.Canceled;
}

// Small-m GEMV for block-quant dtypes ggml cannot serve, today only q2_0_g128,
// whose 128-element blocks ggml's 64-element type 42 cannot address (see
// quants.dequantQ2_0G128). Same shape as `ggml_gemv` below (thread over row
// chunks, weight row read once) minus the activation quantization: the fused dot
// takes f32 activations directly, so there is no `vy` scratch and no from_float.
const native_gemv = struct {
    const Job = struct {
        y: []f32,
        x: []const f32,
        w: []const u8,
        m: usize,
        rows: usize,
        cols: usize,
        row_bytes: usize,
        bias: ?[]const f32,
        tok: cancel.Token,
    };

    fn gemvRows(j: Job, r0: usize, r1: usize) void {
        for (r0..r1) |r| {
            if (cancel.canceled(j.tok)) return; // q2Gemv reports error.Canceled
            const wrow = j.w[r * j.row_bytes ..][0..j.row_bytes];
            for (0..j.m) |t| {
                var s = quants.dotQ2_0G128(wrow, j.x[t * j.cols ..][0..j.cols]);
                if (j.bias) |b| s += b[r];
                j.y[t * j.rows + r] = s;
            }
        }
    }

    /// y[m*rows] = W[rows*cols] (q2_0_g128) * x[m*cols] + bias.
    ///
    /// Threaded over row chunks for the reason `quantGemv` documents: one core
    /// cannot pull the model's weight bandwidth, so rows are split across cores.
    ///
    /// At m > 1 the weight row is re-read per token, so weight traffic scales
    /// with m. That is exactly what `ggml_gemv` does for every other quant (its
    /// `vec_dot` is likewise called per token), so this is parity rather than a
    /// new deficiency, and `small_m_max` hands anything from 16 tokens up to the
    /// packed path, which amortizes the weight over the whole batch.
    pub fn q2Gemv(io: std.Io, y: []f32, x: []const f32, m: usize, w_bytes: []const u8, rows: usize, cols: usize, bias: ?[]const f32) Error!void {
        const row_bytes = DType.q2_0_g128.storageBytes(cols);
        const tok = cancel.token;
        const job = Job{ .y = y, .x = x, .w = w_bytes, .m = m, .rows = rows, .cols = cols, .row_bytes = row_bytes, .bias = bias, .tok = tok };
        const n_threads = std.Thread.getCpuCount() catch 1;
        const want: usize = if (n_threads == 1 or rows < 128) 1 else n_threads;
        if (want == 1) {
            gemvRows(job, 0, rows);
            if (cancel.canceled(tok)) return error.Canceled;
            return;
        }
        const chunk = std.math.divCeil(usize, rows, want) catch unreachable;
        var group: std.Io.Group = .init;
        defer group.cancel(io);
        var r: usize = 0;
        while (r < rows) : (r += chunk) {
            group.async(io, gemvRows, .{ job, r, @min(r + chunk, rows) });
        }
        try group.await(io);
        if (cancel.canceled(tok)) return error.Canceled;
    }
};

// Small-m block-quant GEMV via ggml's AVX2 vec_dot. Behind `have_ggml`: without
// ggml, block-quant weights are rejected in matmul() (error.QuantBackendUnavailable)
// long before reaching here, so this collapses to an empty struct and the
// `@import("ggml")` is never analyzed.
const ggml_gemv = if (have_ggml) struct {
    const ggml = @import("ggml");

    const GemvJob = struct {
        vd: ggml.c.ggml_vec_dot_t,
        y: []f32,
        vy: []const u8,
        w: []const u8,
        m: usize,
        rows: usize,
        cols: usize,
        row_bytes: usize,
        vy_bytes: usize,
        bias: ?[]const f32,
        tok: cancel.Token,
    };

    /// Dot weight rows [r0, r1) against the (shared, pre-quantized) activation.
    fn gemvRows(j: GemvJob, r0: usize, r1: usize) void {
        for (r0..r1) |r| {
            if (cancel.canceled(j.tok)) return; // quantGemv reports error.Canceled
            const wrow = j.w.ptr + r * j.row_bytes;
            for (0..j.m) |t| {
                var s: f32 = 0;
                j.vd.?(@intCast(j.cols), &s, 0, wrow, 0, j.vy.ptr + t * j.vy_bytes, 0, 1);
                if (j.bias) |b| s += b[r];
                j.y[t * j.rows + r] = s;
            }
        }
    }

    /// Decode-path (small m) block-quant GEMV via ggml's AVX2 vec_dot:
    /// y[m*rows] = W[rows*cols] (`dt`) * x[m*cols] + bias. Quantizes each activation
    /// column once to ggml's vec_dot_type (Q8_K for k-quants), then dots every weight
    /// row, threaded over row chunks: a single ggml vec_dot saturates one core, but
    /// the full model's weights exceed single-core memory bandwidth, so splitting
    /// rows across cores pulls more aggregate DRAM bandwidth.
    pub fn quantGemv(io: std.Io, dt: DType, y: []f32, x: []const f32, m: usize, w_bytes: []const u8, rows: usize, cols: usize, bias: ?[]const f32) Error!void {
        quants.ensureGgmlInit();
        const gtype = quants.ggmlType(dt) orelse unreachable; // isBlockQuant gate covers all mapped types
        const wt = ggml.c.ggml_get_type_traits_cpu(gtype);
        const vdt = wt.*.vec_dot_type;
        const from_float = ggml.c.ggml_get_type_traits_cpu(vdt).*.from_float.?;

        const row_bytes: usize = @intCast(ggml.c.ggml_row_size(gtype, @intCast(cols)));
        const vy_bytes: usize = @intCast(ggml.c.ggml_row_size(vdt, @intCast(cols)));

        const alloc = std.heap.c_allocator;
        const vy = alloc.alloc(u8, m * vy_bytes) catch @panic("ggmlQuantGemv: OOM");
        defer alloc.free(vy);
        for (0..m) |t| from_float(x.ptr + t * cols, vy.ptr + t * vy_bytes, @intCast(cols));

        const tok = cancel.token;
        const job = GemvJob{ .vd = wt.*.vec_dot, .y = y, .vy = vy, .w = w_bytes, .m = m, .rows = rows, .cols = cols, .row_bytes = row_bytes, .vy_bytes = vy_bytes, .bias = bias, .tok = tok };
        const n_threads = std.Thread.getCpuCount() catch 1;
        const want: usize = if (n_threads == 1 or rows < 128) 1 else n_threads;
        if (want == 1) {
            gemvRows(job, 0, rows);
            if (cancel.canceled(tok)) return error.Canceled;
            return;
        }

        const chunk = std.math.divCeil(usize, rows, want) catch unreachable;
        var group: std.Io.Group = .init;
        defer group.cancel(io);
        var r: usize = 0;
        while (r < rows) : (r += chunk) {
            group.async(io, gemvRows, .{ job, r, @min(r + chunk, rows) });
        }
        try group.await(io);
        if (cancel.canceled(tok)) return error.Canceled;
    }
} else struct {};

fn packedTask(
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    panel: []f32,
    tok: cancel.Token,
) void {
    // i4 is sub-byte packed (nibbles), so it can't ride the byteSize-based
    // typed path; it has its own kernel that unpacks two values per byte.
    if (w.dtype == .i4 or w.dtype == .w4a8 or w.dtype == .nvfp4) return packedTaskI4(y, x, m, w, bias, row_start, row_end, panel, tok);
    switch (w.dtype) {
        inline .f8_e4m3, .bf16, .f16, .f32, .i8 => |dt| {
            packedTaskTyped(dt, y, x, m, w, bias, row_start, row_end, panel, tok);
        },
        inline .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs, .q1_0, .q2_0_g64, .q2_0_g128 => |dt| {
            packedTaskBlock(dt, y, x, m, w, bias, row_start, row_end, panel, tok);
        },
        else => unreachable, // validated in matmul()
    }
}

/// Packed outer-product path for ggml block-quantized weights. Mirrors
/// `packedTaskI4` (dequant a k-slice of each row into a temp, scatter
/// k-major) with quants.zig doing the block decode. KC is a multiple of every
/// block size we carry (32, 64, 128, 256), so k-slices stay block-aligned, which
/// `dequantSlice` asserts on, and which the comptime check below pins per dtype
/// rather than against a single hardcoded super-block size.
fn packedTaskBlock(
    comptime dt: DType,
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    panel: []f32,
    tok: cancel.Token,
) void {
    comptime std.debug.assert(KC % dt.blockElems() == 0);
    const cols = w.cols;
    const rows = w.rows;
    const row_bytes = dt.storageBytes(cols);
    const n_nr = std.math.divCeil(usize, row_end - row_start, NR) catch unreachable;

    var kc0: usize = 0;
    while (kc0 < cols) : (kc0 += KC) {
        if (cancel.canceled(tok)) return; // matmul() reports error.Canceled
        const kl: usize = @min(KC, cols - kc0);

        for (0..n_nr) |nr| {
            const sub = panel[nr * KC * NR ..][0 .. kl * NR];
            for (0..NR) |j| {
                const row = row_start + nr * NR + j;
                if (row >= rows) {
                    for (0..kl) |k| sub[k * NR + j] = 0;
                    continue;
                }
                var tmp: [KC]f32 = undefined;
                quants.dequantSlice(dt, w.bytes[row * row_bytes ..][0..row_bytes], kc0, kl, tmp[0..kl]);
                for (0..kl) |k| sub[k * NR + j] = tmp[k];
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
    tok: cancel.Token,
) void {
    const cols = w.cols;
    const rows = w.rows;
    const esize = comptime dt.byteSize();
    const n_nr = std.math.divCeil(usize, row_end - row_start, NR) catch unreachable;

    var kc0: usize = 0;
    while (kc0 < cols) : (kc0 += KC) {
        if (cancel.canceled(tok)) return; // matmul() reports error.Canceled
        // Note the explicit type: @min with a comptime bound would narrow to
        // u10 and make `kl * NR` overflow.
        const kl: usize = @min(KC, cols - kc0);

        // Pack + dequantize this k-slice of the row chunk, k-major per subpanel.
        for (0..n_nr) |nr| {
            const sub = panel[nr * KC * NR ..][0 .. kl * NR];
            for (0..NR) |j| {
                const row = row_start + nr * NR + j;
                if (row >= rows) {
                    for (0..kl) |k| sub[k * NR + j] = 0;
                    continue;
                }
                const src = w.bytes[(row * cols + kc0) * esize ..][0 .. kl * esize];
                if (dt == .i8) {
                    // int8 dequant needs the whole 256-group present for the
                    // ConvRot un-rotation, so dequant the k-slice row-major into
                    // a temp (kc0/kl are group-aligned, so it holds whole groups)
                    // then scatter k-major into the subpanel.
                    var tmp: [KC]f32 = undefined;
                    const rs = w.row_scale.?[row];
                    for (0..kl) |k| tmp[k] = @as(f32, @floatFromInt(@as(i8, @bitCast(src[k])))) * rs;
                    if (w.convrot != 0) convrot_mod.rotate(tmp[0..kl]);
                    for (0..kl) |k| sub[k * NR + j] = tmp[k];
                } else {
                    for (0..kl) |k| sub[k * NR + j] = dequantOne(dt, src, k, w.scale);
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

/// Dequant `n` consecutive packed-i4 elements starting at logical element
/// `elem0` (must be even, which holds for row-major weights whose `cols` is a
/// multiple of the 256 ConvRot group) into `dst`: two signed nibbles per byte,
/// element 2k in the low nibble, 2k+1 in the high, scaled by `scale`.
inline fn dequantI4Slice(bytes: []const u8, elem0: usize, n: usize, scale: f32, dst: []f32) void {
    std.debug.assert(elem0 % 2 == 0);
    const byte0 = elem0 / 2;
    for (0..n) |k| {
        const v = dtypes.DType.nibbleI4(bytes[byte0 + k / 2], @intCast(k & 1));
        dst[k] = @as(f32, @floatFromInt(v)) * scale;
    }
}

/// Dequant `n` elements of row `row` starting at column `col0` (even) of a PACKED
/// W4A8 weight: `levels[s_rel[row][group]][nibble] * s_channel[row]`.
///
/// The level table already folds the reference's f32 multiply-round-clamp, so this is
/// two byte lookups and a multiply per element, no floating point until the last
/// step, and bit-identical to the int8 weight ComfyUI would have materialized.
inline fn dequantW4A8Slice(w: Weight, row: usize, col0: usize, n: usize, dst: []f32) void {
    std.debug.assert(col0 % 2 == 0);
    const meta = w.w4a8.?;
    const gs = meta.group_size;
    const srow = meta.s_rel[row * meta.groups(w.cols) ..];
    const rs = w.row_scale.?[row];
    const byte0 = (row * w.cols + col0) / 2;
    for (0..n) |k| {
        const c = col0 + k;
        const tbl = &meta.levels.table[srow[c / gs]];
        const byte = w.bytes[byte0 + k / 2];
        const nib: u4 = @truncate(if (c & 1 == 0) byte else byte >> 4);
        dst[k] = @as(f32, @floatFromInt(tbl[nib])) * rs;
    }
}

/// Packed outer-product path for i4 convrot weights. Mirrors `packedTaskTyped`'s
/// `.i8` branch (dequant a k-slice row-major, un-rotate the whole group, scatter
/// k-major) but unpacks two 4-bit values per byte.
fn packedTaskI4(
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    panel: []f32,
    tok: cancel.Token,
) void {
    const cols = w.cols;
    const rows = w.rows;
    const n_nr = std.math.divCeil(usize, row_end - row_start, NR) catch unreachable;

    var kc0: usize = 0;
    while (kc0 < cols) : (kc0 += KC) {
        if (cancel.canceled(tok)) return; // matmul() reports error.Canceled
        const kl: usize = @min(KC, cols - kc0);

        for (0..n_nr) |nr| {
            const sub = panel[nr * KC * NR ..][0 .. kl * NR];
            for (0..NR) |j| {
                const row = row_start + nr * NR + j;
                if (row >= rows) {
                    for (0..kl) |k| sub[k * NR + j] = 0;
                    continue;
                }
                var tmp: [KC]f32 = undefined;
                if (w.dtype == .w4a8)
                    dequantW4A8Slice(w, row, kc0, kl, tmp[0..kl])
                else if (w.dtype == .nvfp4)
                    nvfp4_mod.decodeSliceF32(tmp[0..kl], w.bytes, w.nvfp4.?.*, cols, row, kc0, kl)
                else
                    dequantI4Slice(w.bytes, row * cols + kc0, kl, w.row_scale.?[row], tmp[0..kl]);
                if (w.convrot != 0) convrot_mod.rotate(tmp[0..kl]);
                for (0..kl) |k| sub[k * NR + j] = tmp[k];
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
    tok: cancel.Token,
) void {
    if (w.dtype == .i4 or w.dtype == .w4a8 or w.dtype == .nvfp4) return runRangeI4(y, x, m, w, bias, row_start, row_end, scratch, tok);
    switch (w.dtype) {
        inline .f8_e4m3, .bf16, .f16, .f32, .i8 => |dt| {
            runRangeTyped(dt, y, x, m, w, bias, row_start, row_end, scratch, tok);
        },
        // Block-quant never reaches runRange: small-m goes to ggmlQuantGemv and
        // large-m to matmulPacked (see matmul()).
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
    tok: cancel.Token,
) void {
    const cols = w.cols;
    var r = row_start;
    while (r < row_end) : (r += panel_rows) {
        if (cancel.canceled(tok)) return; // matmul() reports error.Canceled
        const nr = @min(panel_rows, row_end - r);
        for (0..nr) |j| {
            const src = w.bytes[(r + j) * cols * comptime dt.byteSize() ..][0 .. cols * comptime dt.byteSize()];
            const dst = scratch[j * cols ..][0..cols];
            const rs = if (dt == .i8) w.row_scale.?[r + j] else w.scale;
            dequantRow(dt, dst, src, rs);
            if (dt == .i8 and w.convrot != 0) convrot_mod.rotate(dst);
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

/// Small-m path for i4 convrot weights. Mirrors `runRangeTyped` but dequantizes
/// each weight row from packed nibbles (+ per-row scale + un-rotation) before
/// the dtype-independent GEMM accumulation.
fn runRangeI4(
    y: []f32,
    x: []const f32,
    m: usize,
    w: Weight,
    bias: ?[]const f32,
    row_start: usize,
    row_end: usize,
    scratch: []f32,
    tok: cancel.Token,
) void {
    const cols = w.cols;
    var r = row_start;
    while (r < row_end) : (r += panel_rows) {
        if (cancel.canceled(tok)) return; // matmul() reports error.Canceled
        const nr = @min(panel_rows, row_end - r);
        for (0..nr) |j| {
            const dst = scratch[j * cols ..][0..cols];
            if (w.dtype == .w4a8)
                dequantW4A8Slice(w, r + j, 0, cols, dst)
            else if (w.dtype == .nvfp4)
                nvfp4_mod.decodeSliceF32(dst, w.bytes, w.nvfp4.?.*, cols, r + j, 0, cols)
            else
                dequantI4Slice(w.bytes, (r + j) * cols, cols, w.row_scale.?[r + j], dst);
            if (w.convrot != 0) convrot_mod.rotate(dst);
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
        .i8 => for (dst, src) |*v, b| {
            v.* = @as(f32, @floatFromInt(@as(i8, @bitCast(b)))) * scale;
        },
        .bf16 => dtypes.bf16ToF32Row(src, dst, scale),
        .f16 => dtypes.f16ToF32Row(src, dst, scale),
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

test "probe observes each GEMM's input, attributed by weight tag" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const Recorder = struct {
        const Call = struct { tag: ?[]const u8, m: usize, cols: usize, x0: f32, xn: f32 };
        calls: [4]Call = undefined,
        n: usize = 0,

        fn onInput(ctx: *anyopaque, w: Weight, x: []const f32, m: usize) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            // The hook sees the real input buffer, so record enough to prove it is
            // the right one (not just that something fired).
            self.calls[self.n] = .{ .tag = w.tag, .m = m, .cols = w.cols, .x0 = x[0], .xn = x[x.len - 1] };
            self.n += 1;
        }
    };

    // 2x3 weight, 2 tokens of 3 features.
    const wdata = [_]f32{ 1, 0, 0, 0, 1, 0 };
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var y: [4]f32 = undefined;

    var w = Weight.fromF32(&wdata, 2, 3);
    w.tag = "blocks.0.attn.wq.weight";

    var rec: Recorder = .{};
    probe = .{ .ctx = &rec, .input = Recorder.onInput };
    defer probe = null; // a leaked probe would silently observe every later test

    try matmul(io, gpa, &y, &x, 2, w, null);
    // An untagged weight still fires, the consumer decides what to do with null.
    const untagged = Weight.fromF32(&wdata, 2, 3);
    try matmul(io, gpa, &y, &x, 2, untagged, null);
    // A zero-row GEMM does no work, so there is nothing to observe.
    try matmul(io, gpa, y[0..0], x[0..0], 0, w, null);

    try std.testing.expectEqual(@as(usize, 2), rec.n);
    try std.testing.expectEqualStrings("blocks.0.attn.wq.weight", rec.calls[0].tag.?);
    try std.testing.expectEqual(@as(?[]const u8, null), rec.calls[1].tag);
    for (rec.calls[0..2]) |c| {
        try std.testing.expectEqual(@as(usize, 2), c.m);
        try std.testing.expectEqual(@as(usize, 3), c.cols);
        try std.testing.expectEqual(@as(f32, 1), c.x0);
        try std.testing.expectEqual(@as(f32, 6), c.xn);
    }
    // The GEMM itself is unaffected by observation: y = x @ w^T picks features 0,1.
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 4, 5 }, &y);
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

test "matmul honors the threadlocal cancel token" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const rows = 64;
    const cols = 64;
    const m = small_m_max + 1;
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    @memset(w_f32, 0.5);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    @memset(x, 1.0);
    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);
    const w = Weight.fromF32(w_f32, rows, cols);

    var flag = std.atomic.Value(bool).init(true);
    cancel.token = &flag;
    defer cancel.token = null;
    // Both the packed (large-m) and small-m paths report the cancel.
    try std.testing.expectError(error.Canceled, matmul(io, gpa, y, x, m, w, null));
    try std.testing.expectError(error.Canceled, matmul(io, gpa, y[0..rows], x[0..cols], 1, w, null));

    // Cleared flag: normal completion with a correct result.
    flag.store(false, .release);
    try matmul(io, gpa, y[0..rows], x[0..cols], 1, w, null);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 1e-4);
}

/// int8 ConvRot: quantized bytes + per-row scale, dequantized with the group
/// rotation. The naive reference uses the fully un-rotated f32 weights, so this
/// exercises the matmul's row-scale handling and group-aligned rotation in both
/// the packed and small-m paths (convrot.zig separately validates the matrix).
fn testI8ConvrotAgainstNaive(m: usize, rows: usize, cols: usize, with_bias: bool) !void {
    std.debug.assert(cols % convrot_mod.group_size == 0);
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const bias = if (with_bias) try gpa.alloc(f32, rows) else null;
    defer if (bias) |b| gpa.free(b);
    if (bias) |b| for (b) |*v| {
        v.* = rand.floatNorm(f32);
    };

    const qbytes = try gpa.alloc(u8, rows * cols);
    defer gpa.free(qbytes);
    for (qbytes) |*b| b.* = @bitCast(@as(i8, @intCast(@as(i32, rand.int(i8)))));

    const row_scale = try gpa.alloc(f32, rows);
    defer gpa.free(row_scale);
    for (row_scale) |*s| s.* = 0.001 + rand.float(f32) * 0.01;

    // Reference weights: dequant then un-rotate each row's groups.
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (0..rows) |r| {
        const dst = w_f32[r * cols ..][0..cols];
        for (0..cols) |c| dst[c] = @as(f32, @floatFromInt(@as(i8, @bitCast(qbytes[r * cols + c])))) * row_scale[r];
        convrot_mod.rotate(dst);
    }

    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);
    const y_ref = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_ref);

    var w = Weight.init(qbytes, .i8, rows, cols);
    w.row_scale = row_scale;
    w.convrot = convrot_mod.group_size;
    try matmul(io, gpa, y, x, m, w, bias);
    naiveMatmul(y_ref, x, m, w_f32, rows, cols, bias);

    for (y_ref, y) |e, a| {
        const tol = 1e-3 + 1e-5 * @abs(e) * @sqrt(@as(f32, @floatFromInt(cols)));
        try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "matmul i8 convrot small-m path" {
    try testI8ConvrotAgainstNaive(1, 9, 256, false);
    try testI8ConvrotAgainstNaive(4, 17, 512, true);
}

test "matmul i8 convrot packed path" {
    try testI8ConvrotAgainstNaive(32, 64, 256, true);
    // m % MR, rows % NR, cols spanning multiple KC blocks and groups.
    try testI8ConvrotAgainstNaive(37, 61, 512, true);
    try testI8ConvrotAgainstNaive(small_m_max, NR + 5, 256 * 3, false);
}

/// int4 ConvRot: two signed 4-bit weights packed per byte + per-row scale,
/// dequantized with the group rotation. Same shape as the i8 helper, the
/// naive reference uses the fully un-rotated f32 weights, but weights are
/// int4 [-8,7] packed low-nibble-first.
fn testI4ConvrotAgainstNaive(m: usize, rows: usize, cols: usize, with_bias: bool) !void {
    std.debug.assert(cols % convrot_mod.group_size == 0);
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(123);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const bias = if (with_bias) try gpa.alloc(f32, rows) else null;
    defer if (bias) |b| gpa.free(b);
    if (bias) |b| for (b) |*v| {
        v.* = rand.floatNorm(f32);
    };

    // Random int4 values in [-8, 7], one per logical element.
    const nibbles = try gpa.alloc(i8, rows * cols);
    defer gpa.free(nibbles);
    for (nibbles) |*v| v.* = @as(i8, rand.intRangeAtMost(i4, -8, 7));

    // Pack two per byte, low nibble = even element.
    const qbytes = try gpa.alloc(u8, rows * cols / 2);
    defer gpa.free(qbytes);
    for (qbytes, 0..) |*b, i| {
        const lo: u8 = @as(u4, @bitCast(@as(i4, @intCast(nibbles[2 * i]))));
        const hi: u8 = @as(u4, @bitCast(@as(i4, @intCast(nibbles[2 * i + 1]))));
        b.* = lo | (hi << 4);
    }

    const row_scale = try gpa.alloc(f32, rows);
    defer gpa.free(row_scale);
    for (row_scale) |*s| s.* = 0.001 + rand.float(f32) * 0.01;

    // Reference weights: dequant then un-rotate each row's groups.
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (0..rows) |r| {
        const dst = w_f32[r * cols ..][0..cols];
        for (0..cols) |c| dst[c] = @as(f32, @floatFromInt(nibbles[r * cols + c])) * row_scale[r];
        convrot_mod.rotate(dst);
    }

    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);
    const y_ref = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_ref);

    var w = Weight.init(qbytes, .i4, rows, cols);
    w.row_scale = row_scale;
    w.convrot = convrot_mod.group_size;
    try matmul(io, gpa, y, x, m, w, bias);
    naiveMatmul(y_ref, x, m, w_f32, rows, cols, bias);

    for (y_ref, y) |e, a| {
        const tol = 1e-3 + 1e-5 * @abs(e) * @sqrt(@as(f32, @floatFromInt(cols)));
        try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "matmul i4 convrot small-m path" {
    try testI4ConvrotAgainstNaive(1, 9, 256, false);
    try testI4ConvrotAgainstNaive(4, 17, 512, true);
}

test "matmul i4 convrot packed path" {
    try testI4ConvrotAgainstNaive(32, 64, 256, true);
    // m % MR, rows % NR, cols spanning multiple KC blocks and groups.
    try testI4ConvrotAgainstNaive(37, 61, 512, true);
    try testI4ConvrotAgainstNaive(small_m_max, NR + 5, 256 * 3, false);
}

/// ggml block-quantized weights: random block bytes with pinned-finite f16
/// scales, decoded to f32 by quants.zig for the naive reference, exercises
/// the block dequant plumbing in both the small-m and packed paths.
fn testBlockQuantAgainstNaive(m: usize, rows: usize, cols: usize, dt: DType, with_bias: bool) !void {
    if (!have_ggml) return error.SkipZigTest; // block-quant needs the ggml backend
    const quants_mod = @import("tp_core").quants;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const bias = if (with_bias) try gpa.alloc(f32, rows) else null;
    defer if (bias) |b| gpa.free(b);
    if (bias) |b| for (b) |*v| {
        v.* = rand.floatNorm(f32);
    };

    const row_bytes = dt.storageBytes(cols);
    const wbytes = try gpa.alloc(u8, rows * row_bytes);
    defer gpa.free(wbytes);
    rand.bytes(wbytes);
    // Pin every block's f16 scale fields to small finite values (random u16
    // bit patterns include NaN/inf).
    const d16: u16 = 0x2A66; // 0.05
    const min16: u16 = 0x251F; // 0.02
    const bb = dt.blockBytes();
    var off: usize = 0;
    while (off < wbytes.len) : (off += bb) {
        switch (dt) {
            // q1_0/q2_0's only scale field is `d` at offset 0; the remaining
            // bytes are sign bits / 2-bit codes, so random ones need no pinning
            // (every bit pattern is a valid quant and there is no per-sub-block
            // scale to blow up).
            .q8_0, .q1_0, .q2_0_g64, .q2_0_g128 => std.mem.writeInt(u16, wbytes[off..][0..2], d16, .little),
            .q4_k, .q5_k => {
                std.mem.writeInt(u16, wbytes[off..][0..2], d16, .little);
                std.mem.writeInt(u16, wbytes[off + 2 ..][0..2], min16, .little);
            },
            // q2_k puts d/dmin at the TAIL, after 16 B of scales and 64 B of codes.
            .q2_k => {
                std.mem.writeInt(u16, wbytes[off + 80 ..][0..2], d16, .little);
                std.mem.writeInt(u16, wbytes[off + 82 ..][0..2], min16, .little);
            },
            .q6_k => {
                std.mem.writeInt(u16, wbytes[off + 208 ..][0..2], d16, .little);
                // Pin the 16 i8 sub-block scales to a moderate value: random
                // ±127 scales make pathologically large weights with heavy
                // cancellation, which no relative tolerance survives for the
                // approximate int8 decode path. Varied-scale layout correctness
                // is covered by the fixture-based "q6_k int8 dot" test.
                @memset(wbytes[off + 192 ..][0..16], 8);
            },
            else => unreachable,
        }
    }

    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (0..rows) |r| {
        quants_mod.dequantSlice(dt, wbytes[r * row_bytes ..][0..row_bytes], 0, cols, w_f32[r * cols ..][0..cols]);
    }

    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);
    const y_ref = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_ref);

    const w = Weight.init(wbytes, dt, rows, cols);
    try matmul(io, gpa, y, x, m, w, bias);
    naiveMatmul(y_ref, x, m, w_f32, rows, cols, bias);

    // Small-m block-quant runs through ggml's vec_dot, which quantizes the
    // activation to int8 (Q8_K / Q8_0), approximate by design. Use a robust,
    // dtype-agnostic relative-L2 tolerance over the whole output (a per-row
    // metric blows up on near-zero dots; a per-256-block bound breaks for q8_0).
    //
    // Gated on `usesGgml`, not on `m` alone: a dtype with no ggml vec_dot
    // (q2_0_g128) takes the exact packed path at every m, and holding it to the
    // 5% activation-quantization bound would leave it with no teeth.
    if (m < small_m_max and quants_mod.usesGgml(dt)) {
        var num: f64 = 0;
        var den: f64 = 0;
        for (y_ref, y) |e, a| {
            den += @as(f64, e) * e;
            num += @as(f64, a - e) * (a - e);
        }
        const rel = @sqrt(num / (den + 1e-12));
        std.testing.expect(rel < 0.05) catch |err| {
            std.debug.print("{t} {d}x{d} m={d}: relative-L2 error {d:.5}\n", .{ dt, rows, cols, m, rel });
            return err;
        };
        return;
    }
    for (y_ref, y) |e, a| {
        const tol = 1e-4 + 1e-5 * @abs(e) * @sqrt(@as(f32, @floatFromInt(cols)));
        try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "matmul q8_0 small-m path" {
    try testBlockQuantAgainstNaive(1, 9, 64, .q8_0, false);
    try testBlockQuantAgainstNaive(3, 12, 96, .q8_0, true);
}

test "matmul k-quants small-m path" {
    try testBlockQuantAgainstNaive(1, 9, 256, .q2_k, false);
    try testBlockQuantAgainstNaive(1, 9, 256, .q4_k, false);
    try testBlockQuantAgainstNaive(2, 7, 512, .q5_k, true);
    try testBlockQuantAgainstNaive(1, 11, 256, .q6_k, false);
}

test "matmul block quants packed path" {
    // rows % NR != 0, cols spanning multiple KC slices with a partial last one.
    try testBlockQuantAgainstNaive(32, 61, KC + 256, .q8_0, true);
    try testBlockQuantAgainstNaive(37, NR + 5, KC + 256, .q2_k, true);
    try testBlockQuantAgainstNaive(37, NR + 5, KC + 256, .q4_k, false);
    try testBlockQuantAgainstNaive(19, 33, 512, .q5_k, true);
    try testBlockQuantAgainstNaive(small_m_max, NR, KC + 256, .q6_k, false);
}

test "matmul q1_0 on both paths" {
    // q1_0's 128-element block is the first block size here that is neither 32 nor
    // 256, so the k-slicing is the thing under test as much as the arithmetic:
    // `cols = KC + 128` gives a partial final slice that is still block-aligned.
    try testBlockQuantAgainstNaive(1, 9, 128, .q1_0, false); // GEMV, one block/row
    try testBlockQuantAgainstNaive(3, 12, 512, .q1_0, true); // GEMV, multi-block
    try testBlockQuantAgainstNaive(37, NR + 5, KC + 128, .q1_0, false); // packed, partial slice
    try testBlockQuantAgainstNaive(small_m_max, NR, 5120, .q1_0, true); // packed, Bonsai's hidden
}

test "matmul q2_0 on both paths, both block sizes" {
    // Both variants of GGUF type 42. g64 is a fourth distinct block size (after
    // 32, 256, 128), so as with q1_0 the k-slicing is under test as much as the
    // arithmetic: `cols = KC + blk` gives a partial final slice that is still
    // block-aligned. g128 additionally takes the PACKED path at every m (it has
    // no ggml vec_dot), so its small-m cases exercise a different kernel than
    // every other block quant's do, which is the point of running both here.
    for ([_]DType{ .q2_0_g64, .q2_0_g128 }) |dt| {
        const blk = dt.blockElems();
        try testBlockQuantAgainstNaive(1, 9, blk, dt, false); // one block/row
        try testBlockQuantAgainstNaive(3, 12, 512, dt, true); // multi-block
        try testBlockQuantAgainstNaive(37, NR + 5, KC + blk, dt, false); // packed, partial slice
        try testBlockQuantAgainstNaive(small_m_max, NR, 5120, dt, true); // packed, Bonsai's hidden
    }
}

test "exact_activations removes the small-m path's activation quantization" {
    if (!have_ggml) return error.SkipZigTest;
    const quants_mod = @import("tp_core").quants;
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    // m well below small_m_max, so the default dispatch is the vec_dot GEMV.
    const m = 4;
    const rows = 8;
    const cols = 256;
    const dt: DType = .q4_k;

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const row_bytes = dt.storageBytes(cols);
    const wbytes = try gpa.alloc(u8, rows * row_bytes);
    defer gpa.free(wbytes);
    rand.bytes(wbytes);
    var off: usize = 0;
    while (off < wbytes.len) : (off += dt.blockBytes()) {
        std.mem.writeInt(u16, wbytes[off..][0..2], 0x2A66, .little); // d   = 0.05
        std.mem.writeInt(u16, wbytes[off + 2 ..][0..2], 0x251F, .little); // min = 0.02
    }

    // Reference: the same quantized weights decoded to f32, multiplied exactly.
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (0..rows) |r| {
        quants_mod.dequantSlice(dt, wbytes[r * row_bytes ..][0..row_bytes], 0, cols, w_f32[r * cols ..][0..cols]);
    }
    const y_ref = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_ref);
    naiveMatmul(y_ref, x, m, w_f32, rows, cols, null);

    const w = Weight.init(wbytes, dt, rows, cols);
    const y_gemv = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_gemv);
    const y_exact = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_exact);

    const prev = exact_activations;
    defer exact_activations = prev;

    exact_activations = false;
    try matmul(io, gpa, y_gemv, x, m, w, null);
    exact_activations = true;
    try matmul(io, gpa, y_exact, x, m, w, null);

    const relL2 = struct {
        fn f(ref: []const f32, got: []const f32) f64 {
            var num: f64 = 0;
            var den: f64 = 0;
            for (ref, got) |e, a| {
                den += @as(f64, e) * e;
                num += @as(f64, a - e) * (a - e);
            }
            return @sqrt(num / (den + 1e-12));
        }
    }.f;

    const err_exact = relL2(y_ref, y_exact);
    const err_gemv = relL2(y_ref, y_gemv);

    // With the flag the activations stay f32, so the only difference from the
    // reference is float summation order.
    std.testing.expect(err_exact < 1e-6) catch |err| {
        std.debug.print("exact_activations path rel-L2 {e:.3}, expected < 1e-6\n", .{err_exact});
        return err;
    };
    // Without it, quantizing x to Q8_K costs real accuracy. Asserting the gap
    // (rather than an absolute bound on err_gemv) is what makes this a test of
    // the flag's effect and not of ggml's tolerances.
    std.testing.expect(err_gemv > err_exact * 100) catch |err| {
        std.debug.print("gemv rel-L2 {e:.3} vs exact {e:.3}: activation quantization is not visible\n", .{ err_gemv, err_exact });
        return err;
    };
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
