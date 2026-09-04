//! CUDA-backend Krea 2 DiT forward (int8 convrot checkpoint), producing the
//! same latent as the CPU/Vulkan paths so the hand-PTX backend can generate
//! like-for-like images. Follows the fallback op sequence (int8 tensor-core
//! GEMMs via `opI8Prep`/`opI8Gemm`, f32 eltwise norm/rope/gate, one naive
//! online-softmax GQA attention kernel). Small/CPU-cheap paths (text fusion,
//! timestep MLPs, patchify, unpatchify) stay on the CPU.
//!
//! Numerics match `DiT.forward` up to floating-point reordering (int8 quant +
//! ex2.approx softmax), the same regime the Vulkan int8 path runs in.

const std = @import("std");
const dit = @import("dit.zig");
const cuda = @import("tp_gpu").cuda;
const safetensors = @import("tp_core").safetensors;
const ops = @import("tp_ops");

const DiT = dit.DiT;
const Backend = cuda.Backend;
const DeviceBuffer = cuda.backend.DeviceBuffer;
const DType = @import("tp_core").dtype.DType;

const F = dit.features; // 6144
const heads = dit.n_heads; // 48
const kv_heads = dit.n_kv_heads; // 12
const hd = dit.head_dim; // 128
const half = hd / 2; // 64
const mlp_dim = dit.mlp_dim; // 16384
const n_blocks = dit.n_blocks; // 28
const patch = dit.patch; // 2
const channels = dit.channels; // 16
const txt_dim = dit.txt_dim; // 2560
const txt_heads = dit.txt_heads; // 20
const txt_layers = dit.txt_layers; // 12
const txt_mlp_dim = dit.txt_mlp_dim; // 6912
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)
const eps: f32 = 1e-5;

/// MLP sequence-tile: the gate/up/down GEMMs run over chunks of this many rows
/// so the mg/mu intermediates are [tile][mlp_dim] instead of [seq][mlp_dim]
/// (512 MiB -> 128 MiB each at 1 MP). The MLP is per-row so chunks are independent.
const mlp_tile: usize = 2048;

/// A device-buffer sub-view offset `off_bytes` into `b` (CUDA buffers are raw
/// pointers, so a mid-buffer view is just pointer arithmetic, the eltwise/GEMM
/// kernels index from the given base).
fn offsetBuf(b: DeviceBuffer, off_bytes: usize) DeviceBuffer {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = b.size - off_bytes };
}

/// Zero bias for the block linears (they have none, but opMatmulBf16 always
/// folds a bias in). A file-scope constant so every GEMM width shares one stable
/// host pointer, cachedWeight then caches it once at the widest slice.
const zero_bias: [mlp_dim]f32 = @splat(0);

/// Weight class of the DiT block linears. A convrot checkpoint is int8/int4
/// (per-row scale + prep-once shared-input quant GEMM); a dense checkpoint is
/// bf16 (each linear a standalone f16 tensor-core GEMM); an fp8-e4m3 checkpoint
/// streams each linear through the dequant-to-f16 + hgemm path; a GGUF checkpoint is a
/// ggml block quant, decoded per GEMM either to convrot int8 (`blockq_i8`) or to f16
/// (`blockq_f16`), which is the same choice `blockq_gemm` describes. Uniform across
/// blocks, except that the two blockq arms cover several ggml formats and decode each
/// weight by its own dtype, so a checkpoint that mixes them still computes correctly.
const LinKind = enum { i8, i4, w4a8, nvfp4, bf16, fp8, blockq_i8, blockq_i4, blockq_f16, blockq_mmq };

/// Which GEMM a GGUF block-quant DiT decodes its weights for.
///
/// `int8` rotates and re-quantizes to convrot int8, capping accuracy at int8's; `int4` is
/// that decode one width down for the s4 tensor cores; `f16` expands the weight and keeps
/// the format's own accuracy at about half int8's throughput; `mmq` multiplies the packed
/// weight in place with its scale folded per 32-k substep, re-quantizing nothing.
pub const BlockQGemm = enum { auto, int8, int4, f16, mmq };

/// Default block-quant GEMM target, overridden by `--dit-gguf-gemm`.
///
/// What a re-quantizing route costs a format depends on how the format's own error
/// compares to the regrid's, so `auto` takes int8 for q2_k and q4_k (0.03% and 0.7% more
/// weight error) and f16 for the rest, where q8_0 alone would lose 84%. int4 is never
/// `auto`: it is W4A4, and its 16-level ACTIVATIONS cost more than the weight regrid
/// saves, 2.4 dB for 1.11x the step even on q2_k, whose weight barely notices the regrid.
pub var blockq_gemm: BlockQGemm = .auto;

/// The kind a block-quant weight of this dtype runs as, honouring `blockq_gemm`.
///
/// Only q2_k, q4_k and q8_0 have a convrot decode at all (`Backend.blockQFormat`), so for
/// every other format this is f16 whatever the setting says.
fn blockQKind(dt: DType) LinKind {
    if (blockq_gemm == .mmq and Backend.mmqPipeDtype(dt)) return .blockq_mmq;
    if (Backend.blockQFormat(dt) == null) return .blockq_f16;
    return switch (blockq_gemm) {
        .int8 => .blockq_i8,
        .int4 => .blockq_i4,
        .f16, .mmq => .blockq_f16,
        .auto => switch (dt) {
            .q2_k, .q4_k => .blockq_i8,
            else => .blockq_f16,
        },
    };
}

/// Whether this dtype ends up in a W4A4 GEMM here, activation included. A GGUF's route
/// is a runtime choice its dtype does not show, so a check calibrated on how far a route
/// may drift from the weight-only CPU forward has to ask this, not the storage dtype.
pub fn activationIs4Bit(dt: DType) bool {
    return dt == .i4 or (dt.isBlockQuant() and blockQKind(dt) == .blockq_i4);
}

/// Prep the shared linear input. int8/int4 rotate+quantize `x` in place (all the
/// block's GEMMs then read that internal state); bf16/fp8 GEMMs consume the f32
/// `x` directly, so no prep is needed.
///
/// `rot` is the checkpoint's own convrot state (`dit.i8Convrot`), not a choice: an
/// unrotated `int8_tensorwise` weight needs an unrotated activation and vice versa.
fn linPrep(be: *Backend, kind: LinKind, rot: bool, x: DeviceBuffer, m: usize, cols: usize) !void {
    switch (kind) {
        .i4 => try be.opI4Prep(x, m, cols),
        // W4A8's activation prep IS int8's: the "A8" in the name is exactly that the
        // activation stays 8-bit, so only the WEIGHT's storage differs. That is also
        // what lets a mixed int8/W4A8 checkpoint share one prep per group.
        .i8, .w4a8 => try be.opI8PrepR(x, m, cols, false, rot),
        // A block quant's activation prep IS int8's, because its weights are decoded to
        // convrot int8 per GEMM (`opI8GemmBlockQ`) and fed to the same vendor kernel.
        // The rotation has to match on both sides, so there is nothing format-specific
        // to do here.
        .blockq_i8 => try be.opI8PrepR(x, m, cols, false, blockq_rotate),
        // The int4 decode writes s4, so the activation has to be s4 too. `opI4Prep`
        // always rotates, so this arm ignores `blockq_rotate` rather than silently
        // pairing an unrotated weight with a rotated activation.
        .blockq_i4 => try be.opI4Prep(x, m, cols),
        // MMQ quantizes the activation to q8_1 (a scale per 32 columns) and every linear
        // sharing this `x` then reads it, which is the same prep-once the int8 arm gets
        // and the f16 arm does not.
        .blockq_mmq => try be.opMatmulQuantMmqPipePrep(x, m, cols),
        // NVFP4 needs no prep: it is weight-only here (its own activation quantization is
        // the Blackwell-only path), so the GEMM consumes the f32 `x` like bf16/fp8 do.
        // `blockq_f16` is weight-only for the same reason: `opMatmulQuant` converts the
        // activation itself, and there is no rotation to match.
        .nvfp4, .bf16, .fp8, .blockq_f16 => {},
    }
}

/// One block linear y[m][rows] f32 = x[m][cols] @ Wᵀ. int8/int4 read the prepped
/// activation state (`x` is ignored); bf16 runs the f32-in/f32-out f16
/// tensor-core GEMM (opMatmulBf16, weight bf16->f16 at upload) with a zero bias.
/// fp8 streams the weight, dequants it to an f16 scratch (per-tensor `w.scale`
/// folded in) and runs the same validated hgemm. Block linears carry no bias, so
/// `bias` is unused for the bf16/fp8 GEMMs (they all pass the shared zero_bias).
fn lin(be: *Backend, kind: LinKind, y: DeviceBuffer, x: DeviceBuffer, m: usize, w: anytype, bias: []const f32) !void {
    try probeLinInput(be, kind, x, m, w);
    switch (kind) {
        .i4 => try be.opI4Gemm(y, w.bytes, w.row_scale.?, w.rows),
        .i8, .w4a8 => try i8GemmW(be, y, w, false),
        // Ampere+ feeds raw bf16 straight to the tensor cores (no f16 convert, so
        // a streamed weight is touched once); older cards fall back to the
        // GPU-side bf16->f16 GEMM.
        .bf16 => if (be.ctx.cc_major >= 8)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, bias[0..w.rows])
        else
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, bias[0..w.rows], false, false),
        .fp8 => try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols),
        // Both decode into the shared scratch and read the prepped activation out of
        // backend state, so `x` is ignored here as it is on the other convrot arms.
        .blockq_i8 => try i8GemmW(be, y, w, false),
        .blockq_i4 => try be.opI4GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, true),
        // Expands the weight to f16 and runs the f16 tensor cores, the same shape as the
        // fp8 arm above. Structurally simpler than the int8 arm because nothing has to be
        // rotated or rescaled, so the decode is a streaming dequant.
        .blockq_f16 => try be.opMatmulQuant(w.dtype, y, x, m, w.bytes, w.rows, w.cols),
        // The packed s8 goes straight to the int8 tensor cores, scale folded per 32-k
        // substep, so the weight is used exactly. `x` is ignored: the prepped activation
        // lives in backend state, like the other int8 arms.
        .blockq_mmq => try be.opMatmulQuantMmqPipePrepped(w.dtype, y, m, w.bytes, w.rows, w.cols),
        .nvfp4 => try be.opMatmulNvfp4(
            y,
            x,
            m,
            w.bytes,
            w.nvfp4.?.scales,
            std.mem.asBytes(&w.nvfp4.?.levels.bf16v),
            w.rows,
            w.cols,
            &zero_bias,
        ),
    }
}

/// Feed `ops.matmul.probe` this GEMM's input, so an activation capture works when
/// the DiT runs on CUDA.
///
/// The hook exists here because this backend never goes through `ops.matmul`,
/// it owns its upload and GEMM, so the CPU probe call site sees nothing at all on a
/// GPU run. `lin` is the single choke point for all 224 block linears (the same role
/// `Loader.mat` plays for tagging), so one call covers every one of them.
///
/// int8/int4 are skipped, not captured. `linPrep` has already rotated and
/// quantized `x` in place by the time a GEMM runs, so the buffer no longer holds the
/// f32 activation, recording it would be a W4A4-shaped number filed as a weight-only
/// one (ACTIVATION_AWARE hygiene rule 4). The capture driver refuses those
/// checkpoints outright rather than relying on this returning quietly.
///
/// The activation is downloaded and handed to the same host accumulator a CPU
/// capture uses, rather than reduced on device. That is deliberate for a first
/// version: the statistics then come out of identical f64 code, so a GPU-captured
/// cache differs from a CPU one only by the DiT's own GEMM arithmetic, which is the
/// open question, not a confound in it. It costs one `cuStreamSynchronize` plus
/// `m × cols × 4 B` over PCIe per linear; if that ever dominates a capture, the
/// reduction moves onto the device and this shrinks to a small download.
fn probeInput(be: *Backend, x: DeviceBuffer, m: usize, w: anytype) !void {
    const p = ops.matmul.probe orelse return;
    if (m == 0 or w.cols == 0) return;
    const host = be.gpa.alloc(f32, m * w.cols) catch return error.OutOfMemory;
    defer be.gpa.free(host);
    // `x` must already be a view at the activation's own base (see `offsetBuf`):
    // `tensorDownload` copies from the buffer's start.
    try be.tensorDownload(x, std.mem.sliceAsBytes(host));
    p.input(p.ctx, w, host, m);
}

/// One int8-convrot GEMM, dispatching on how the weight is STORED rather than on the
/// model's `LinKind`: a plain int8 weight goes straight to the GEMM, a packed W4A8 one
/// is decoded into the backend's transient scratch first. Both then run the identical
/// int8 kernel, so this is the only place the two storage forms differ, and dispatching
/// per weight rather than per model means a checkpoint that mixes them computes
/// correctly even though krea2's `LinKind` is still one value for the whole trunk.
fn i8GemmW(be: *Backend, y: DeviceBuffer, w: anytype, c_h16: bool) !void {
    // A block quant decodes to convrot int8 in a scratch and then runs this same GEMM,
    // so it belongs here rather than on a path of its own.
    if (Backend.blockQFormat(w.dtype) != null)
        return be.opI8GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, c_h16, blockq_rotate);
    if (w.dtype == .w4a8) {
        const meta = w.w4a8.?;
        return be.opI8GemmW4A8(
            y,
            w.bytes,
            meta.s_rel,
            std.mem.asBytes(meta.levels),
            w.row_scale.?,
            w.rows,
            w.cols,
            meta.group_size,
            c_h16,
        );
    }
    return be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, c_h16);
}

/// The first block linear the chosen decode cannot cover, by name, or null.
///
/// Scans every weight, not one: a checkpoint quantizes per layer, and the widths differ
/// per layer anyway, so one probe would answer about one shape.
fn blockQUnsupportedLin(model: *const DiT) ?[]const u8 {
    for (model.blocks) |*b| {
        for ([_]ops.matmul.Weight{
            b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate,
            b.mlp.gate, b.mlp.up,  b.mlp.down,
        }) |w| {
            switch (blockQKind(w.dtype)) {
                // 1024 is the convrot FWHT's one-group-per-thread floor, and every format
                // the int8 decode handles divides 256, so a multiple of 1024 is a whole
                // number of blocks too. The f16 route rotates nothing and so has no floor.
                .blockq_i8 => if (w.cols % 1024 != 0) return w.tag orelse "<untagged>",
                // Same FWHT floor, plus the hand s4 GEMM's 128-row tile (cuBLASLt has
                // no s4 kernel, so unlike the int8 arm there is no vendor fallback).
                .blockq_i4 => if (w.cols % 1024 != 0 or w.rows % 128 != 0) return w.tag orelse "<untagged>",
                .blockq_f16 => if (!Backend.quantKernelSupported(w.dtype)) return w.tag orelse "<untagged>",
                // rows % 128 and cols % 256, which krea2's widths all satisfy.
                .blockq_mmq => if (!Backend.mmqPipeSupported(w.dtype, w.rows, w.cols)) return w.tag orelse "<untagged>",
                else => unreachable,
            }
        }
    }
    return null;
}

/// Bytes of decode scratch the block linears need, 0 when none of them decodes into
/// it. Both convrot arms share `bq_i8`; the int4 one writes half as many bytes per
/// weight, so a checkpoint mixing them sizes to the wider.
fn blockQScratchNeed(model: *const DiT) usize {
    var mx: usize = 0;
    for (model.blocks) |*b| {
        for ([_]ops.matmul.Weight{
            b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate,
            b.mlp.gate, b.mlp.up,  b.mlp.down,
        }) |w| mx = @max(mx, switch (blockQKind(w.dtype)) {
            .blockq_i8 => Backend.blockQScratchBytes(w.rows, w.cols),
            .blockq_i4 => Backend.blockQScratchBytesI4(w.rows, w.cols),
            else => 0,
        });
    }
    return mx;
}

/// `probeInput` for the block linears, which are the only GEMMs here whose input
/// may already have been consumed in place.
fn probeLinInput(be: *Backend, kind: LinKind, x: DeviceBuffer, m: usize, w: anytype) !void {
    if (ops.matmul.probe == null) return;
    if (kind == .i8 or kind == .i4 or kind == .w4a8) return; // see the note above
    try probeInput(be, x, m, w);
}

/// Use the tensor-core GQA attention path (hgemm+softmax_row) instead of the
/// naive one-thread-per-(q,head) kernel. On by default, it is O(seq²) faster on
/// the tensor cores and the naive path is O(seq²) latency-bound. Toggle for A/B.
pub var use_tc_attn: bool = true;

/// Whether the block-quant weight decode and the activation prep apply the convrot
/// rotation. Both or neither: the rotation cancels between the two, so dropping it is
/// exact arithmetic and changes only how well ONE scale per row fits the data. Toggle
/// for A/B.
///
/// Leave it on. Off is 91 ms/step faster (1.181 -> 1.090, the FWHT leaving the decode
/// AND the activation prep) and destroys the output: rel RMSE 0.915 against the CPU
/// forward, i.e. uncorrelated, where rotated is 0.044. The weight side barely cares,
/// unrotated per-row int8 costs 0.3% on top of q4_k's own error, measured offline, so
/// all of that is the ACTIVATIONS. Their outliers are what one scale per row cannot
/// span, and spreading them is the whole reason ComfyUI's int8 format is rotated at all.
pub var blockq_rotate: bool = true;

// ==== text fusion (CUDA) =====================================================
//
// The int8/int4 convrot checkpoints leave the text-fusion stack unquantized
// (BF16 attn/mlp, F32 projector/txtmlp). Run on the CPU it is ~2 TFLOP of dense
// GEMM, the entire "loading diffusion model" stall on --backend zig-cuda. This
// mirrors DiT.txtFusion+textTokens on the backend: BF16 weights are dequantized
// to f32 once and fed through the f32 `opMatmul` (no int GEMM applies here); the
// block is the plain (no modulation, no RoPE) variant of the sampling block.

/// f32 weight bytes for `opMatmul`: BF16 and int8 weights dequant into `arena` (kept
/// alive for the whole fusion so the backend's pointer-keyed weight cache stays valid);
/// F32 weights pass through their mmap bytes unchanged.
///
/// int8 reaches here because a `int8_tensorwise` checkpoint may quantize the fusion
/// stack, which the convrot ones leave dense. There is no int GEMM on this path (the
/// widths are small and the prep would not pay), so it dequantizes like bf16 does.
fn txtF32(arena: std.mem.Allocator, w: anytype) ![]const u8 {
    if (w.dtype == .f32) return w.bytes;
    if (w.dtype == .i8) return (try ops.matmul.materializeF32(arena, w)).bytes;
    const out = try arena.alloc(f32, w.rows * w.cols);
    try safetensors.convertToF32(w.dtype, w.bytes, out);
    return std.mem.sliceAsBytes(out);
}

/// Device scratch for the text-fusion blocks, sized for the widest phase (the
/// layerwise blocks over seq_txt*12 rows). Reused by the refiner and txtmlp.
const TxtScratch = struct {
    normed: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    g: DeviceBuffer,
    attn: DeviceBuffer,
    t1: DeviceBuffer,
    gate: DeviceBuffer, // also holds txtmlp mid (seq_txt*6144 ≤ rows_lw*6912)
    up: DeviceBuffer, // also holds txtmlp out

    fn init(be: *Backend, rows_lw: usize) !TxtScratch {
        var s: TxtScratch = undefined;
        s.normed = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.q = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.k = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.v = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.g = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.attn = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.t1 = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.gate = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        s.up = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        return s;
    }
    fn deinit(s: *TxtScratch, be: *Backend) void {
        inline for (@typeInfo(TxtScratch).@"struct".fields) |f| be.tensorDestroy(&@field(s, f.name));
    }
};

/// One f16 tensor-core GEMM y[m][out] = x[m][in] @ Wᵀ (+bias), W dequant to f32.
/// The naive f32 opMatmul is ~50× too slow for these widths; opConvF16 runs the
/// validated hgemm on tensor cores. f16 is far finer than the int8/int4 the DiT
/// blocks downstream run in, so text-fusion precision is not the bottleneck.
/// `bias` is a real bias, or a zeros slice (≥ out long) for the no-bias linears.
fn txtGemm(be: *Backend, arena: std.mem.Allocator, y: DeviceBuffer, x: DeviceBuffer, m: usize, w: anytype, out: usize, in: usize, bias: []const f32) !void {
    // The second probe choke point: every txtfusion linear and both txtmlp linears
    // come through here, and a capture that recorded only the 224 block linears
    // would be missing 34 of the model's 263 measurable layers, a cache that looks
    // complete and is not.
    try probeInput(be, x, m, w);
    try be.opConvF16(y, 0, x, m, try txtF32(arena, w), out, in, bias);
}

/// One TextFusionBlock on the backend: x += attn(rmsnorm(x)); x += mlp(rmsnorm(x)).
/// `x_d` holds n_seqs sequences of seq_len rows of txt_dim (no RoPE, no mask, no
/// modulation). qkNorm with hd=txt_dim is a plain per-row RMS-norm×weight.
fn txtBlockCuda(be: *Backend, arena: std.mem.Allocator, blk: anytype, x_d: DeviceBuffer, s: *const TxtScratch, zero: []const f32, n_seqs: usize, seq_len: usize) !void {
    const rows = n_seqs * seq_len;

    // --- attention ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.prenorm), rows, txt_dim, eps);
    try txtGemm(be, arena, s.q, s.normed, rows, blk.attn.wq, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.k, s.normed, rows, blk.attn.wk, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.v, s.normed, rows, blk.attn.wv, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.g, s.normed, rows, blk.attn.gate, txt_dim, txt_dim, zero);
    try be.qkNorm(s.q, s.q, try normBuf(be, blk.attn.qnorm), rows * txt_heads, hd, eps);
    try be.qkNorm(s.k, s.k, try normBuf(be, blk.attn.knorm), rows * txt_heads, hd, eps);
    // Each of the n_seqs sequences attends only within itself (12-long layerwise,
    // or the whole prompt for the refiner), the naive kernel handles one at a
    // time; seq_len is small so the launch count is cheap and one-time.
    for (0..n_seqs) |i| {
        const off = i * seq_len * txt_dim * 4;
        try be.attn(offsetBuf(s.q, off), offsetBuf(s.k, off), offsetBuf(s.v, off), offsetBuf(s.attn, off), seq_len, seq_len, txt_heads, txt_heads, hd, attn_scale, false);
    }
    try be.sigmoidMul(s.attn, s.g, rows * txt_dim);
    try txtGemm(be, arena, s.t1, s.attn, rows, blk.attn.wo, txt_dim, txt_dim, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);

    // --- mlp (swiglu) ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.postnorm), rows, txt_dim, eps);
    try txtGemm(be, arena, s.gate, s.normed, rows, blk.mlp.gate, txt_mlp_dim, txt_dim, zero);
    try txtGemm(be, arena, s.up, s.normed, rows, blk.mlp.up, txt_mlp_dim, txt_dim, zero);
    try be.siluMul(s.gate, s.up, rows * txt_mlp_dim);
    try txtGemm(be, arena, s.t1, s.gate, rows, blk.mlp.down, txt_dim, txt_mlp_dim, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);
}

/// CUDA port of `DiT.textTokens`: text conditioning [seq_txt*12*txt_dim] -> the
/// combined-sequence tokens [seq_txt*features] the sampler consumes. Runs the
/// whole txtfusion + txtmlp stack on the backend, so it does not stall the CPU.
/// Transient f32 weights are dropped from the cache before returning.
pub fn textTokensCuda(model: *const DiT, be: *Backend, gpa: std.mem.Allocator, cond: []const f32) ![]f32 {
    const seq_txt = cond.len / (txt_layers * txt_dim);
    std.debug.assert(cond.len == seq_txt * txt_layers * txt_dim);
    const rows_lw = seq_txt * txt_layers;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var x_d = try be.tensorCreate(rows_lw * txt_dim * 4);
    defer be.tensorDestroy(&x_d);
    var s = try TxtScratch.init(be, rows_lw);
    defer s.deinit(be);
    // Reclaim the transient f32 text weights (and small norm buffers) from the
    // backend cache; they are unused during sampling and would otherwise pin VRAM.
    // SCOPED (not evictWeights, which nukes ALL cached weights): this runs per
    // image inside a persistent pipeline.Session, and a full evict would drop the
    // resident DiT that the next queued image reuses. The
    // scope drops exactly the weights cached here, the DiT (cached later, in the
    // sampling loop, and pinned) survives.
    be.weightScopeBegin();
    defer be.weightScopeEnd();

    try be.tensorUpload(x_d, std.mem.sliceAsBytes(cond));

    // Zero bias for the no-bias linears (opConvF16 always adds a bias; it reads
    // only the first `out` entries, so one max-width zeros buffer serves all).
    const zero = try arena.alloc(f32, txt_mlp_dim);
    @memset(zero, 0);

    // Layerwise blocks: seq_txt independent 12-long sequences (across the encoder
    // layer axis per token).
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    for (&model.txt_layerwise) |*blk| try txtBlockCuda(be, arena, blk, x_d, &s, zero, seq_txt, txt_layers);
    try be.endBatch();

    // Projector: collapse the 12-layer axis (projected[tok][d] = Σ_l pw[l]*x[tok*12+l][d]).
    // Tiny [1,12] contraction, a host round-trip is simpler than a bespoke kernel.
    {
        const work = try arena.alloc(f32, rows_lw * txt_dim);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(work));
        var pw: [txt_layers]f32 = undefined;
        try safetensors.convertToF32(model.txt_projector.dtype, model.txt_projector.bytes, &pw);
        const projected = try arena.alloc(f32, seq_txt * txt_dim);
        for (0..seq_txt) |tok| {
            const dst = projected[tok * txt_dim ..][0..txt_dim];
            @memset(dst, 0);
            for (0..txt_layers) |l| {
                const src = work[(tok * txt_layers + l) * txt_dim ..][0..txt_dim];
                for (dst, src) |*d, sv| d.* += pw[l] * sv;
            }
        }
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(projected));
    }

    try be.beginBatch();
    // Refiner blocks: one sequence of length seq_txt.
    for (&model.txt_refiner) |*blk| try txtBlockCuda(be, arena, blk, x_d, &s, zero, 1, seq_txt);

    // txtmlp: rmsnorm -> Linear(2560->6144) -> geluTanh -> Linear(6144->6144).
    try be.qkNorm(x_d, s.normed, try normBuf(be, model.txtmlp_norm), seq_txt, txt_dim, eps);
    try txtGemm(be, arena, s.gate, s.normed, seq_txt, model.txtmlp1.w, F, txt_dim, model.txtmlp1.b.?);
    try be.gelu(s.gate, seq_txt * F);
    try txtGemm(be, arena, s.up, s.gate, seq_txt, model.txtmlp3.w, F, F, model.txtmlp3.b.?);
    try be.endBatch();

    const out = try gpa.alloc(f32, seq_txt * F);
    errdefer gpa.free(out);
    try be.tensorDownload(s.up, std.mem.sliceAsBytes(out));
    return out;
}

/// Per-run constants: text-fusion tokens + rope table, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    txt0_d: DeviceBuffer,
    txt_len: usize, // element count (seq_txt * F)
    freqs_d: DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cond: []const f32, seq_txt: usize) !Session {
        _ = io; // text fusion runs on the backend (textTokensCuda), not the CPU
        const h = lat_h / patch;
        const w = lat_w / patch;
        const seq = seq_txt + h * w;

        const txt_tokens = try textTokensCuda(model, be, gpa, cond);
        defer gpa.free(txt_tokens);
        var txt0_d = try be.tensorCreate(txt_tokens.len * 4);
        errdefer be.tensorDestroy(&txt0_d);
        try be.tensorUpload(txt0_d, std.mem.sliceAsBytes(txt_tokens));

        var freqs = try DiT.ropeFreqs(gpa, seq_txt, h, w);
        defer freqs.deinit(gpa);
        std.debug.assert(freqs.half == half);
        const fp = try gpa.alloc(f32, 2 * seq * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. seq * half], freqs.cos);
        @memcpy(fp[seq * half ..], freqs.sin);
        var freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&freqs_d);
        try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));

        return .{
            .seq_txt = seq_txt,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
        };
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.txt0_d);
        be.tensorDestroy(&self.freqs_d);
    }
};

fn normBuf(be: *Backend, w: []const f32) !DeviceBuffer {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// Queue all of a DiT block's streamable weights for async prefetch (called one
/// block ahead so the uploads overlap the previous block's compute). Keys must
/// match the byte slices `forward` later fetches (same host pointers -> cache hit).
fn prefetchBlock(be: *Backend, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    inline for (.{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate, blk.attn.wo }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
    be.prefetchWeight(bytes(blk.attn.qnorm));
    be.prefetchWeight(bytes(blk.attn.knorm));
    inline for (.{ blk.mlp.gate, blk.mlp.up, blk.mlp.down }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
}

/// Per-run device scratch shared across sampler steps (and both CFG sessions):
/// the ~12 activation buffers a forward would otherwise `cuMemAlloc`/free every call.
/// Sized once for the largest sequence (both prompts share n_img, only seq_txt differs).
pub const Workspace = struct {
    x_d: DeviceBuffer = .{},
    imgin_d: DeviceBuffer = .{},
    mv_d: DeviceBuffer = .{},
    fin_d: DeviceBuffer = .{},
    t1_d: DeviceBuffer = .{},
    q_d: DeviceBuffer = .{},
    k_d: DeviceBuffer = .{},
    v_d: DeviceBuffer = .{},
    g_d: DeviceBuffer = .{},
    attn_d: DeviceBuffer = .{},
    mg_d: DeviceBuffer = .{},
    mu_d: DeviceBuffer = .{},

    pub fn init(be: *Backend, lat_h: usize, lat_w: usize, seq_txt_cap: usize) !Workspace {
        const n_img = (lat_h / patch) * (lat_w / patch);
        const mpad = std.mem.alignForward(usize, seq_txt_cap + n_img, 128);
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(mpad * F * 4);
        ws.imgin_d = try be.tensorCreate(n_img * channels * patch * patch * 4);
        ws.mv_d = try be.tensorCreate(n_blocks * 6 * F * 4);
        ws.fin_d = try be.tensorCreate(2 * F * 4);
        ws.t1_d = try be.tensorCreate(mpad * F * 4);
        ws.q_d = try be.tensorCreate(mpad * heads * hd * 4);
        ws.k_d = try be.tensorCreate(mpad * kv_heads * hd * 4);
        ws.v_d = try be.tensorCreate(mpad * kv_heads * hd * 4);
        ws.g_d = try be.tensorCreate(mpad * F * 4);
        ws.attn_d = try be.tensorCreate(mpad * heads * hd * 4);
        // mg/mu hold one MLP tile (see mlp_tile), not the full padded sequence.
        const mlp_rows = @min(mpad, std.mem.alignForward(usize, mlp_tile, 128));
        ws.mg_d = try be.tensorCreate(mlp_rows * mlp_dim * 4);
        ws.mu_d = try be.tensorCreate(mlp_rows * mlp_dim * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            be.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// One DiT forward evaluation: velocity `out` from latent `x_lat` at `sigma`.
/// `ws` must be sized for a sequence ≥ seq_txt + n_img (see Workspace.init).
pub fn forward(model: *const DiT, be: *Backend, sess: *const Session, ws: *const Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32, cancel: ?*std.atomic.Value(bool)) !void {
    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const n_img = (lat_h / patch) * (lat_w / patch);
    const seq = seq_txt + n_img;
    const sin_off = seq * half;

    // ---- CPU: modulation vectors, final-layer vector, patch embed ----
    const tvv = try model.timestepVectors(io, gpa, sigma);
    defer gpa.free(tvv.t);
    defer gpa.free(tvv.tvec);

    const mv = try gpa.alloc(f32, n_blocks * 6 * F);
    defer gpa.free(mv);
    for (model.blocks, 0..) |blk, b| {
        const base = b * 6 * F;
        for (0..6 * F) |i| mv[base + i] = tvv.tvec[i] + blk.mod[i];
        for (0..F) |c| {
            mv[base + c] = (1.0 + mv[base + c]) * blk.prenorm[c];
            mv[base + 3 * F + c] = (1.0 + mv[base + 3 * F + c]) * blk.postnorm[c];
        }
    }
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    for (0..F) |c| {
        fin[c] = (1.0 + tvv.t[c] + model.last_mod[c]) * model.last_norm[c];
        fin[F + c] = tvv.t[c] + model.last_mod[F + c];
    }
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);

    // ---- device buffers (from the per-run Workspace) ----
    const x_d = ws.x_d;
    const imgin_d = ws.imgin_d;
    const mv_d = ws.mv_d;
    const fin_d = ws.fin_d;
    const t1_d = ws.t1_d;
    const q_d = ws.q_d;
    const k_d = ws.k_d;
    const v_d = ws.v_d;
    const g_d = ws.g_d;
    const attn_d = ws.attn_d;
    const mg_d = ws.mg_d;
    const mu_d = ws.mu_d;

    try be.tensorUpload(mv_d, std.mem.sliceAsBytes(mv));
    try be.tensorUpload(fin_d, std.mem.sliceAsBytes(fin));
    try be.tensorCopy(x_d, 0, sess.txt0_d, 0, sess.txt_len * 4);
    try be.tensorUpload(imgin_d, std.mem.sliceAsBytes(img_in));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // Weight class of the DiT block linears, gated once: int8/int4 convrot
    // (per-row scale + packed int weights), dense bf16, or raw fp8-e4m3 (streamed
    // + dequant-to-f16 hgemm). A block-quant / unknown dtype has no GEMM path
    // here, so reject it with a clear error instead of a bad GPU access.
    const wqt = model.blocks[0].attn.wq.dtype;
    if (!dit.gpuLinKindSupported(wqt, .cuda)) return error.UnsupportedCheckpoint;
    const kind: LinKind = switch (wqt) {
        .i8 => .i8,
        .i4 => .i4,
        .w4a8 => .w4a8,
        .nvfp4 => .nvfp4,
        .bf16 => .bf16,
        .f8_e4m3 => .fp8,
        else => blockQKind(wqt), // a block quant; anything else was gated above
    };
    // Whether the int8 activation prep rotates. A property of the checkpoint, not a
    // tuning knob: ComfyUI's `int8_tensorwise` ships rotated (scale per row) and
    // unrotated (one scale per tensor), and the prep has to match the weight it feeds.
    const i8_rot = dit.i8Convrot(model) orelse {
        std.log.err("dit cuda: this checkpoint mixes convrot and plain int8 block linears; " ++
            "one activation prep serves a whole block, so there is no correct one", .{});
        return error.UnsupportedCheckpoint;
    };
    // `opI4Prep` always rotates and has no unrotated build, so an unrotated int4 weight
    // would be multiplied in a basis it was never quantized in.
    if (kind == .i4 and !i8_rot) {
        std.log.err("dit cuda: int4 block linears without convrot are not supported", .{});
        return error.UnsupportedCheckpoint;
    }
    if (kind == .blockq_i8 or kind == .blockq_i4 or kind == .blockq_f16 or kind == .blockq_mmq) {
        // Refuse a shape or format the chosen decode cannot cover here rather than at the
        // launch. krea2's widths (6144, 1536, 16384) clear the int8 arm's floor.
        if (blockQUnsupportedLin(model)) |tag| {
            std.log.err("dit cuda: {s} is a block quant this decode cannot cover " ++
                "(the int8/int4 routes need q2_k, q4_k or q8_0 and cols % 1024 == 0; the f16 route " ++
                "needs a dequant kernel); try --dit-gguf-gemm f16, or --backend cpu", .{tag});
            return error.UnsupportedCheckpoint;
        }
        // Pre-size the int8 decode scratch to the widest linear so it never grows
        // mid-forward, for the reason the W4A8 one does: growth syncs the stream. The f16
        // route's scratch is `fp8_w16`, which the fp8 arm already grows the same way.
        const need = blockQScratchNeed(model);
        if (need != 0) try be.ensureDeviceBuffer(&be.bq_i8, need);
    }
    const zeros: []const f32 = &zero_bias;
    // f16 activation chain (c16): only on the cuBLASLt/irescale int8 libs path.
    // Halves the mlp gate/up/silu/down-input DRAM traffic (the biggest eltwise
    // category). Hand-PTX (igemm_pipe_fused writes f32), int4, and bf16 keep f32.
    const mlp_f16 = (be.kernels == .libs) and (kind == .i8 or kind == .w4a8 or kind == .blockq_i8);
    if (dit.w4a8SmallGroup(model)) |tag| {
        std.log.err("dit cuda: {s} is W4A8 with a group_size that is not a multiple of 8; " ++
            "this backend's decode kernel needs one scale per 8 columns (use --backend cpu or vulkan)", .{tag});
        return error.UnsupportedCheckpoint;
    }
    // Pre-size the W4A8 decode scratch to the model's widest weight, so it never grows
    // mid-forward: growth is safe (ensureDeviceBuffer syncs the stream first) but the
    // first block would pay several syncs for nothing.
    if (dit.anyW4A8(model))
        try be.ensureDeviceBuffer(&be.w4a8_i8, dit.maxW4A8Scratch(model, Backend.w4a8ScratchBytes));

    // patch embed: x[seq_txt..] = img_in @ first^T + bias
    const first_f8 = model.first.w.dtype == .f8_e4m3;
    try probeInput(be, imgin_d, n_img, model.first.w);
    try be.opMatmul(x_d, seq_txt * F * 4, imgin_d, 0, n_img, model.first.w.bytes, first_f8, F, channels * patch * patch, model.first.w.scale, model.first.b);

    // Prefetch block 0's weights before the loop; each iteration prefetches the
    // NEXT block so its upload overlaps this block's compute (async streaming).
    if (be.async_uploads) prefetchBlock(be, model.blocks[0]);

    for (model.blocks, 0..) |blk, b| {
        // Poll cancel between blocks so a stop lands mid-step (≈1/28 of a step)
        // rather than only at step boundaries, matters most under weight
        // streaming, where each block waits on its uploaded weights. The
        // `errdefer` above aborts the in-flight CUDA batch on the way out.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and b + 1 < model.blocks.len) prefetchBlock(be, model.blocks[b + 1]);
        const mb = b * 6 * F;
        // --- attention ---
        try be.rmsMod(x_d, t1_d, mv_d, seq, F, mb + 0 * F, mb + 1 * F, eps);
        try linPrep(be, kind, i8_rot, t1_d, seq, F);
        try lin(be, kind, q_d, t1_d, seq, blk.attn.wq, zeros);
        try lin(be, kind, k_d, t1_d, seq, blk.attn.wk, zeros);
        try lin(be, kind, v_d, t1_d, seq, blk.attn.wv, zeros);
        try lin(be, kind, g_d, t1_d, seq, blk.attn.gate, zeros);
        const qn = try normBuf(be, blk.attn.qnorm);
        const kn = try normBuf(be, blk.attn.knorm);
        try be.qkNorm(q_d, q_d, qn, seq * heads, hd, eps);
        try be.qkNorm(k_d, k_d, kn, seq * kv_heads, hd, eps);
        try be.rope(q_d, sess.freqs_d, seq, heads, half, sin_off);
        try be.rope(k_d, sess.freqs_d, seq, kv_heads, half, sin_off);
        if (use_tc_attn)
            try be.opAttnTC(q_d, k_d, v_d, attn_d, seq, heads, kv_heads, hd, attn_scale)
        else
            try be.attn(q_d, k_d, v_d, attn_d, seq, seq, heads, kv_heads, hd, attn_scale, false);
        try be.sigmoidMul(attn_d, g_d, seq * F);
        try linPrep(be, kind, i8_rot, attn_d, seq, blk.attn.wo.cols);
        try lin(be, kind, t1_d, attn_d, seq, blk.attn.wo, zeros);
        try be.gatedAdd(x_d, t1_d, mv_d, seq * F, F, mb + 2 * F);
        // --- mlp (sequence-tiled: mg/mu are [tile][mlp_dim]; each row-chunk is
        // independent, so we walk seq in mlp_tile-row bands over offset x views) ---
        var c0: usize = 0;
        while (c0 < seq) : (c0 += mlp_tile) {
            const tile: usize = @min(mlp_tile, seq - c0);
            const xo = offsetBuf(x_d, c0 * F * 4);
            try be.rmsMod(xo, t1_d, mv_d, tile, F, mb + 3 * F, mb + 4 * F, eps);
            try linPrep(be, kind, i8_rot, t1_d, tile, F);
            if (mlp_f16) {
                // gate/up GEMMs emit f16 (irescale_h16); silu_mul_h16 reads/writes
                // f16; the down prep reads f16, halving the 16384-dim traffic.
                try i8GemmW(be, mg_d, blk.mlp.gate, true);
                try i8GemmW(be, mu_d, blk.mlp.up, true);
                try be.siluMul16(mg_d, mu_d, tile * mlp_dim);
                try be.opI8PrepR(mg_d, tile, blk.mlp.down.cols, true, i8_rot);
            } else {
                try lin(be, kind, mg_d, t1_d, tile, blk.mlp.gate, zeros);
                try lin(be, kind, mu_d, t1_d, tile, blk.mlp.up, zeros);
                try be.siluMul(mg_d, mu_d, tile * mlp_dim);
                try linPrep(be, kind, i8_rot, mg_d, tile, blk.mlp.down.cols);
            }
            try lin(be, kind, t1_d, mg_d, tile, blk.mlp.down, zeros); // down -> f32 t1_d for gatedAdd
            try be.gatedAdd(xo, t1_d, mv_d, tile * F, F, mb + 5 * F);
        }
    }

    // --- final layer ---
    try be.rmsMod(x_d, t1_d, fin_d, seq, F, 0, F, eps);
    const last_f8 = model.last_linear.w.dtype == .f8_e4m3;
    // Note the offset view: this GEMM reads the image rows only, so the probe must
    // see the same rows the GEMM does and not the text prefix.
    try probeInput(be, offsetBuf(t1_d, seq_txt * F * 4), n_img, model.last_linear.w);
    try be.opMatmul(imgin_d, 0, t1_d, seq_txt * F * 4, n_img, model.last_linear.w.bytes, last_f8, channels * patch * patch, F, model.last_linear.w.scale, model.last_linear.b);
    try be.endBatch();

    const final_rows = try gpa.alloc(f32, n_img * channels * patch * patch);
    defer gpa.free(final_rows);
    try be.tensorDownload(imgin_d, std.mem.sliceAsBytes(final_rows));
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
}

// --- tests -----------------------------------------------------------------

// The block-quant route policy. Pure dispatch, so it runs on the fast suite: it decides
// which GEMM a GGUF DiT uses, and getting it wrong is a silent 2x slowdown or a silent
// accuracy cap rather than a failure.
test "the block-quant route policy sends each format to the GEMM that suits it" {
    const saved = blockq_gemm;
    defer blockq_gemm = saved;

    // `auto`: q4_k to int8, because int8-convrot's ~0.009 error floor is 0.7% on top of
    // q4_k's own and buys 1.85x the speed. Everything else to f16, where q8_0 keeps the
    // accuracy the int8 route would spend (84% more weight error, measured).
    blockq_gemm = .auto;
    try std.testing.expectEqual(LinKind.blockq_i8, blockQKind(.q4_k));
    // q2_k too, even though its WEIGHT could take the s4 regrid: that route is W4A4 and
    // the 4-bit activations, not the weight, are what it costs. int4 stays opt-in.
    try std.testing.expectEqual(LinKind.blockq_i8, blockQKind(.q2_k));
    for ([_]DType{ .q8_0, .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
        try std.testing.expectEqual(LinKind.blockq_f16, blockQKind(dt));

    // An explicit choice is honoured for every format that has a convrot decode.
    for ([_]DType{ .q2_k, .q4_k, .q8_0 }) |dt| {
        blockq_gemm = .int8;
        try std.testing.expectEqual(LinKind.blockq_i8, blockQKind(dt));
        blockq_gemm = .int4;
        try std.testing.expectEqual(LinKind.blockq_i4, blockQKind(dt));
        blockq_gemm = .f16;
        try std.testing.expectEqual(LinKind.blockq_f16, blockQKind(dt));
    }

    // Asking for int8 or int4 on a format with no convrot decode must NOT silently reach
    // `opI8GemmBlockQ`, which would refuse mid-forward; it falls back to f16.
    for ([_]BlockQGemm{ .int8, .int4 }) |g| {
        blockq_gemm = g;
        for ([_]DType{ .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
            try std.testing.expectEqual(LinKind.blockq_f16, blockQKind(dt));
    }
}

// The CUDA W4A8 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed, which is the point: the kernel's only earlier
// validation was that a real render came out bit-identical to the load-time
// materialization, and that check stops being reproducible the moment the checkpoint
// leaves the disk.
//
// Covers what the PTX quietly assumes and a render would not localize: the `prmt.b32`
// byte packing (its selector nibbles must stay under 8 or the instruction
// sign-replicates), the `v2.u32` store's 8-byte alignment, and the one-group-scale-per-
// thread shortcut that makes `group_size % 8 == 0` a requirement.
test "the CUDA W4A8 decode matches ops.w4a8.decode" {
    const gpa = std.testing.allocator;
    const w4a8 = ops.w4a8;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const cases = [_]struct { rows: usize, cols: usize, gs: usize }{
        .{ .rows = 256, .cols = 512, .gs = 16 },
        .{ .rows = 128, .cols = 1024, .gs = 32 },
        .{ .rows = 64, .cols = 256, .gs = 8 }, // the smallest group the kernel allows
    };
    // Every case's buffers stay ALIVE in one arena, and that is load-bearing rather
    // than tidy: both device weight caches key on the HOST POINTER, so freeing a case's
    // arrays and letting the allocator hand the same address to the next case scores a
    // stale cache hit and the kernel reads the previous case's weights. Found the hard
    // way, this test failed on its third case with the first case's data. (A model
    // never hits it: the weights live in the model arena for the model's lifetime.)
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0xA8 + c.cols);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        // Every fp8 byte including NaN (0x7F/0xFF); the level table sends those to 0 on
        // both sides, so this pins that they agree rather than diverging where neither
        // reference defines a value.
        const s_rel = try alloc.alloc(u8, c.rows * c.cols / c.gs);
        r.bytes(s_rel);
        const levels = try alloc.create(w4a8.Levels);
        levels.* = w4a8.Levels.init(&w4a8.fixed_lut);

        const want = try alloc.alloc(i8, c.rows * c.cols);
        w4a8.decode(want, packed_bytes, s_rel, levels, c.rows, c.cols, c.gs);

        const db = try be.w4a8ToI8(packed_bytes, s_rel, std.mem.asBytes(levels), c.rows, c.cols, c.gs);
        const got = try alloc.alloc(i8, c.rows * c.cols);
        try be.tensorDownload(db, std.mem.sliceAsBytes(got));

        for (want, got, 0..) |wv, gv, i| {
            std.testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("rows {d} cols {d} gs {d}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, c.gs, i, i / c.cols, i % c.cols, gv, wv,
                });
                return e;
            };
        }
    }
}

// The CUDA NVFP4 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed.
//
// f16 output, so this is a TOLERANCE against the f32 CPU decode rather than an
// equality, but a tight one, because both sides read the same level table and f16 only
// rounds the store. The bound is set from f16's own quantum, so a nibble-order or
// block-index error (which move values by whole levels) cannot hide inside it.
test "the CUDA NVFP4 decode matches ops.nvfp4.decode" {
    const gpa = std.testing.allocator;
    const nvfp4 = ops.nvfp4;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const cases = [_]struct { rows: usize, cols: usize }{
        .{ .rows = 128, .cols = 256 },
        .{ .rows = 64, .cols = 1024 },
        .{ .rows = 16, .cols = 16 }, // one block per row
    };
    // One arena for every case: the weight cache keys on the HOST POINTER, so reusing an
    // address across cases would score a stale hit and decode the previous case's bytes.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0xF4 + c.cols);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        // Every fp8 byte including NaN: the level table maps those identically on both
        // sides, so this pins that they agree rather than diverging where nothing defines
        // a value.
        const scales = try alloc.alloc(u8, c.rows * c.cols / nvfp4.block_size);
        r.bytes(scales);
        const levels = try alloc.create(nvfp4.Levels);
        levels.* = nvfp4.Levels.init(0.0125);
        const meta: nvfp4.Meta = .{ .scales = scales, .levels = levels };

        const want = try alloc.alloc(f32, c.rows * c.cols);
        nvfp4.decode(want, packed_bytes, meta, c.rows, c.cols);

        const db = try be.nvfp4ToBf16(packed_bytes, scales, std.mem.asBytes(&levels.bf16v), c.rows, c.cols);
        const got_bits = try alloc.alloc(u16, c.rows * c.cols);
        try be.tensorDownload(db, std.mem.sliceAsBytes(got_bits));

        for (want, got_bits, 0..) |wv, gb, i| {
            const gv: f32 = @bitCast(@as(u32, gb) << 16); // bf16 -> f32
            // A NaN block-scale byte (fp8 0x7F/0xFF) PROPAGATES here, unlike `.w4a8`
            // whose int8 target forces it to 0, and it propagates in the reference too,
            // so agreeing on NaN is the correct outcome, not a hole in the check.
            const both_nan = std.math.isNan(wv) and std.math.isNan(gv);
            // Otherwise bf16 rounds the value and nothing else may change it: 8 mantissa
            // bits is 2^-8 relative. A wrong nibble or block index moves by at least one
            // E2M1 level (>= 1/6 of the value), far outside this.
            const tol = @max(@abs(wv) * 5e-3, 1e-30);
            std.testing.expect(both_nan or @abs(gv - wv) <= tol) catch |e| {
                std.debug.print("{d}x{d}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, i, i / c.cols, i % c.cols, gv, wv,
                });
                return e;
            };
        }
    }
}
