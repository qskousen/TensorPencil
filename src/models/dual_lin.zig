//! The one linear dispatcher's routing, for every backend and every caller.
//!
//! Which kernel a linear runs is decided here from four things: the weight's storage
//! and shape, the row count, the caller's `Policy`, and what the backend has (`Caps`,
//! a backend's adapter type, so routing is decided and tested without a device). A
//! backend adapter executes a `Route`; it never chooses one. So a format any caller of
//! a backend runs, every caller runs, and a gap between two backends is a difference
//! between their `Caps`.
//!
//! The two callers' routing used to be two files: a DiT picks a route per weight for
//! its whole forward (`Policy.gemv = false`), an LLM stepper also per row count, taking
//! a GEMV where it decodes one row or verifies a few (`Policy.gemv = true`).

const std = @import("std");
const lin = @import("lin.zig");
const lora_mod = @import("lora.zig");

pub const Weight = lin.Weight;
pub const DType = lin.DType;

/// Which kernel one linear runs.
pub const Route = enum {
    /// Dense weights in their own width. `gemm_bf16` takes the bf16 tensor cores
    /// where the backend has them and the shape tiles, the f16 GEMM otherwise.
    /// `gemm_f32` also takes an f16 or bf16 weight, widened, on a backend with no
    /// 16-bit GEMM (`wideGemm`).
    gemm_f32,
    gemm_f16,
    gemm_bf16,
    gemm_fp8,
    /// fp8 decoded to bf16 with the activation staged as bf16 (`Act16.bf16`).
    gemm_fp8_bf16,
    /// `Dense.f16`: a dense weight on the f16 tensor cores through its cached f16 copy;
    /// the f32 one below `h16_min_co` rows takes the f32 GEMM unless the activation
    /// is f16.
    h16_f32,
    h16_f16,
    h16_bf16,
    /// fp8 against the f32 activation, for a backend whose fp8 GEMM cannot stage it
    /// at the width `Act16` asks for or cannot take the shape.
    gemm_fp8_f32,
    /// Dense GEMVs, for a policy that runs a few rows: one row, or 4 rows a launch.
    gemv_bf16,
    gemv_bf16n,
    gemv_f16,
    gemv_fp8,
    gemv_fp8n,
    gemv_f32,
    gemv_f32n,
    /// A skinny f32 weight (a router), one subgroup a row over every activation row.
    gemv_f32_rows,
    /// Integer convrot and NVFP4. `_wo` expands the weight to bf16 per call against
    /// the unquantized activation (`IntGemm.weight_only`).
    i8,
    i4,
    w4a8,
    /// int4 decoded to int8 per call against the int8 prep (W4A8 arithmetic), for a
    /// backend with no int4 tensor cores.
    i4_i8,
    i8_wo,
    i4_wo,
    w4a8_wo,
    /// The same expansion in f16, against an f16 activation (`IntGemm.weight_only_f16`).
    i8_wo16,
    i4_wo16,
    w4a8_wo16,
    /// The same expansion in f32 through a true-f32 GEMM (`IntGemm.weight_only_f32`).
    i8_wo32,
    i4_wo32,
    w4a8_wo32,
    nvfp4,
    /// Nunchaku SVDQuant decoded to int8 per call against its own prep: the activation
    /// smoothed, then the int8 prep unrotated (`ops/svdq.zig`). The `_wo` twins expand
    /// it exactly against the unquantized activation, at bf16, f16 or f32.
    svdq,
    svdq_wo,
    svdq_wo16,
    svdq_wo32,
    /// Block quants as a batch: re-quantized to int8 or int4 convrot per call (both
    /// quantize the activation too), expanded to bf16 or f16 per call, or MMQ on the
    /// packed weight against a q8_1 activation.
    blockq_i8,
    blockq_i4,
    blockq_bf16,
    blockq_f16,
    blockq_mmq,
    /// Block-quant GEMVs: fused over the f32 activation, dp4a over the q8 one (one
    /// row, 8 rows a launch, or every row in one launch), and the dual row GEMV, one
    /// row or 4 a launch (`gemv_dualn`, each row bit-identical to one).
    gemv_q,
    gemv_q8,
    gemv_q8n,
    gemv_q8batch,
    gemv_dual_q8,
    gemv_dual,
    gemv_dualn,
    /// `gemv_dualn` against the q8_1 activation, each row bit-identical to `gemv_dual_q8`.
    gemv_dualn_q8,

    pub fn prep(r: Route) Prep {
        return switch (r) {
            .i8, .w4a8, .i4_i8, .blockq_i8 => .i8,
            .i4, .blockq_i4 => .i4,
            .blockq_mmq => .mmq,
            .svdq => .svdq,
            .gemv_q8, .gemv_q8n, .gemv_q8batch, .gemv_dual_q8, .gemv_dualn_q8 => .q8,
            else => .none,
        };
    }
};

/// What a route reads besides the weight: only `x`, the int8 or int4 convrot
/// activation, the MMQ tile's q8_1 one, the GEMVs' q8 one, or SVDQuant's smoothed int8.
pub const Prep = enum { none, i8, i4, mmq, q8, svdq };

/// Which GEMM a GGUF block-quant weight takes as a batch. A caller's choice: a DiT and a
/// text encoder want different trades from the same file.
///
/// `int8` rotates and re-quantizes to convrot int8, capping accuracy at int8's; `int4` is
/// that decode one width down for the s4 tensor cores; `bf16` expands the weight and keeps
/// the format's own accuracy at about half int8's throughput (bf16, not f16: Z-Image's
/// activations pass f16's range); `f16` is the same expansion in f16, three more mantissa
/// bits for an activation that fits the range; `mmq` multiplies the packed weight in place
/// with its scale folded per 32-k substep, re-quantizing nothing. `int8`, `int4` and `mmq`
/// quantize the ACTIVATION to 8 or 4 bits as well; `bf16` and `f16` leave it alone.
/// `auto` is the DiT's measured choice (int8 for q2_k and q4_k, bf16 for the rest) and
/// `llm` the LLM prefill's (MMQ where it measured faster than the f16 dequant GEMM).
pub const BlockQGemm = enum { auto, int8, int4, bf16, f16, mmq, llm };

/// What an integer convrot weight (int8, int4, W4A8) multiplies against. `w8a8`
/// quantizes the activation to the integer tensor cores' width per token, which is
/// ComfyUI's arithmetic and the fast one; `weight_only` expands the weight to bf16 per
/// call, rotated back into the checkpoint's basis, and runs the bf16 GEMM on the
/// unquantized activation, so the only quantization left is the file's own. A
/// per-plan choice, because what the activation quantization costs depends on the
/// activation: a Qwen3-style prefill carries an outlier channel on every token that
/// per-token absmax cannot hold.
/// `weight_only_f16` expands to f16 instead, three more mantissa bits for an activation
/// that fits f16's range, which is what a one-shot tower wants; `weight_only_f32` to f32
/// through a true-f32 GEMM (no TF32), the range and the precision both, at f32's speed.
pub const IntGemm = enum { w8a8, weight_only, weight_only_f16, weight_only_f32 };

/// How every conditioning pass (the CLIP towers, the Qwen3 text encoders, SenseNova's
/// prompt pass) runs an integer convrot weight, `--te-int-gemm`, on both backends.
/// Weight-only by default for `lin_cuda.encoder_gemm`'s reason, and because a prompt's
/// residual stream can carry an outlier channel that per-token int8 absmax cannot hold:
/// SenseNova's loses about an eighth of its conditioning to W8A8 at full depth, ~1% to
/// the int8 weight alone, and H3's 32B Qwen3 encoder is 9x closer to its CPU forward.
/// `w8a8` is ComfyUI's arithmetic. A denoiser trunk always keeps W8A8.
///
/// Null follows `encoder_int8_dial`, which `pipeline.Session.encode` sets from the encoder's
/// activation dial (`Session.intGemmFor`): the format's own arithmetic under `auto`,
/// weight-only at an explicit dtype. A set value is a device check's or `--te-int-gemm`'s.
pub var encoder_int8: ?IntGemm = null;
pub var encoder_int8_dial: IntGemm = .w8a8;

/// How a denoiser trunk's integer convrot weights run, for a plan whose `Use` names none;
/// `pipeline.Session` sets it from the denoiser's dial before building one.
pub var trunk_int8: IntGemm = .w8a8;

pub fn encoderInt8() IntGemm {
    return encoder_int8 orelse encoder_int8_dial;
}

/// How a plan computes a DENSE weight (f32, f16 or bf16 storage). `native` runs each
/// in its own width: bf16 on the bf16 tensor cores, f32 through cuBLASLt's f32 GEMM.
/// `f16` runs all three on the f16 tensor cores, the weight converted once in the
/// device cache, which is ComfyUI's fp16 UNet and what the SD family computes; it is
/// also the only dense arm that reads and writes f16 activations.
pub const DenseCompute = enum { native, f16 };

/// The width a route that narrows the f32 activation to 16 bits stages it in. Range
/// decides it, not precision: an activation past 65504 needs `bf16`.
pub const Act16 = enum { f16, bf16 };

/// A caller's routing choices.
pub const Policy = struct {
    /// Whether this caller runs a few rows at a time and takes a GEMV there.
    gemv: bool = false,
    blockq: BlockQGemm = .auto,
    int8: IntGemm = .w8a8,
    dense: DenseCompute = .native,
    act16: Act16 = .f16,
    /// DIAGNOSTICS of the GEMV policy. `decode_dp4a = false` sends block-quant decode
    /// through the f32 GEMV (the isolation for a wrong decode); `dual_decode` sends a
    /// format with both a hand kernel and a dual one through the dual one; `force`
    /// pins the batched block-quant route where the weight can take it.
    decode_dp4a: bool = true,
    dual_decode: bool = false,
    force: Force = .auto,

    pub const llm: Policy = .{ .gemv = true, .blockq = .llm };
};

pub const Force = enum { auto, grouped, mmq, dequant };

/// DIAGNOSTIC: whether block-quant decode takes the dp4a kernels. Off sends every
/// block quant through the f32 `gemv_q` kernel, which is the isolation for a wrong
/// decode: the two differ only by the q8_1 activation quantization.
pub var decode_dp4a: bool = true;

/// DIAGNOSTIC: force the batched (`m > 1`) block-quant route of a GEMV plan, so its
/// crossovers (`grouped_max`, `mmqFaster`) are measurable from one binary
/// (`tp-llm --llm-gemm`). A forced route a weight cannot take, by dtype or by shape,
/// falls back to the automatic choice; `check` still answers for `auto`.
pub var force: Force = .auto;

/// DIAGNOSTIC: route a format that has BOTH a hand-written decode kernel and a dual one
/// (q4_k) through the dual one. The A/B that says whether the hand kernels are still
/// worth their maintenance; `qgemv-bench` alternates it.
pub var dual_decode: bool = false;

/// What `plan` decided for a model: one value per checkpoint, read by every prep and GEMM.
pub const Plan = struct {
    /// Whether the int8 activation prep rotates. A property of the checkpoint
    /// (`lin.convrot`), not a knob.
    rot: bool,
    /// The caller's block-quant route.
    blockq: BlockQGemm,
    /// Whether this caller runs a few rows at a time and takes a GEMV there (an LLM).
    gemv: bool = false,
    /// The LoRA sidecars, or null. Set by the family after `plan`, since it is a
    /// property of the render and not of the checkpoint. `gemm` applies every hit
    /// against the weight it just multiplied, which is what makes one hook reach every
    /// family instead of one per architecture.
    lora: ?*const lora_mod.Stack = null,
    /// The integer convrot weights' GEMM. Set after `plan`, like `lora`: every shape
    /// `check` accepts for the quantized-activation routes runs weight-only too.
    int8: IntGemm = .w8a8,
    /// How dense weights compute. Set after `plan` by the family.
    dense: DenseCompute = .native,
    /// The 16-bit width a route that narrows the f32 activation stages it in. `bf16` for
    /// a family whose activations pass f16's 65504 (Z-Image's MLP down projection reads
    /// ~1e5 and renders solid white at f16); `f16` keeps three more mantissa bits
    /// everywhere else. Set after `plan` by the family.
    act16: Act16 = .f16,

    /// An LLM stepper's plan over weights with no convrot rotation, which is what
    /// `plan` returns for one under `Use.llm`.
    pub const llm: Plan = .{ .rot = false, .blockq = .llm, .gemv = true, .int8 = .weight_only };

    /// A one-shot tower (a ViT, an embedder): every dense weight on the f16 tensor
    /// cores and a block quant on the f16 expansion, the activation never quantized.
    pub const tower: Plan = .{ .rot = false, .blockq = .f16, .dense = .f16, .int8 = .weight_only_f16 };

    /// The routing policy, with the GEMV diagnostics as currently set.
    pub fn policy(p: Plan) Policy {
        return .{ .gemv = p.gemv, .blockq = p.blockq, .int8 = p.int8, .dense = p.dense, .act16 = p.act16, .decode_dp4a = decode_dp4a, .force = force, .dual_decode = dual_decode };
    }
};

/// How a caller runs its linears, for `check`: the plan's routing choices and, for a
/// GEMV plan, the most rows one call passes (a prefill chunk).
pub const Use = struct {
    blockq: BlockQGemm,
    gemv: bool = false,
    /// Null takes `trunk_int8`, the denoiser's activation dial.
    int8: ?IntGemm = null,
    act16: Act16 = .f16,
    max_rows: usize = 1,

    /// An LLM stepper's: integer convrot weights weight-only (they need no prep, and a
    /// prompt's activations carry the outliers per-token int8 cannot hold).
    pub fn llm(max_rows: usize) Use {
        return .{ .blockq = .llm, .gemv = true, .int8 = .weight_only, .max_rows = max_rows };
    }
};

/// Why `check` refused a checkpoint.
/// `too_wide` is a caller's own check: a linear wider than the scratch it sized.
pub const Why = enum { mixed_convrot, no_kernel, int_shape, int4_unrotated, blockq_basis, nvfp4_shape, w4a8_group, gemm_shape, gemv_width, no_row_scale, too_wide };

pub const Refusal = struct { why: Why, tag: []const u8 = "", dtype: DType = .f32, rows: usize = 0, cols: usize = 0 };

pub const Verdict = union(enum) { ok: Plan, refused: Refusal };

/// Accept or refuse a model's device linears for every row count `use` runs on the
/// backend `caps` describes, naming the first problem. What a kernel cannot take about
/// a shape is the backend's answer (`caps.refuse`).
pub fn check(comptime C: type, caps: C, lins: []const Weight, use: Use) Verdict {
    const rot = lin.convrot(lins) orelse return .{ .refused = .{ .why = .mixed_convrot } };
    const p = Plan{ .rot = rot, .blockq = use.blockq, .gemv = use.gemv, .int8 = use.int8 orelse trunk_int8, .act16 = use.act16 };
    const pol = p.policy();
    for (lins) |w| {
        const bad = Refusal{ .why = undefined, .tag = w.tag orelse "<untagged>", .dtype = w.dtype, .rows = w.rows, .cols = w.cols };
        const blockq = lin.kindOf(w.dtype) == .blockq;
        if (use.gemv and blockq and !caps.quantGemv(w.dtype)) return .{ .refused = with(bad, .no_kernel) };
        const r = routeOf(C, caps, w, 1, pol) orelse return .{ .refused = with(bad, if (use.gemv and blockq) .gemv_width else .no_kernel) };
        if (refuseRoute(C, caps, lins, rot, w, r)) |why| return .{ .refused = with(bad, why) };
        // Two rows as well as the most: a small batch can take a route neither end does.
        if (use.gemv and use.max_rows > 1) for ([_]usize{ 2, use.max_rows }) |m| {
            const rm = routeOf(C, caps, w, m, pol) orelse return .{ .refused = with(bad, .gemm_shape) };
            if (refuseRoute(C, caps, lins, rot, w, rm)) |why| return .{ .refused = with(bad, why) };
        };
    }
    if (lin.w4a8SmallGroup(lins)) |tag| return .{ .refused = .{ .why = .w4a8_group, .tag = tag, .dtype = .w4a8 } };
    return .{ .ok = p };
}

fn refuseRoute(comptime C: type, caps: C, lins: []const Weight, rot: bool, w: Weight, r: Route) ?Why {
    // Named here rather than met as `w.row_scale.?` in the GEMM, where the null is a
    // slice at address 0.
    switch (w.dtype) {
        .i8, .i4, .w4a8, .w6a8 => if (w.row_scale == null or w.row_scale.?.len != w.rows) return .no_row_scale,
        else => {},
    }
    return caps.refuse(lins, rot, w, r);
}

fn with(r: Refusal, why: Why) Refusal {
    var out = r;
    out.why = why;
    return out;
}

/// `check`, logging the refusal under `who` and returning it as an error.
pub fn plan(comptime C: type, caps: C, lins: []const Weight, use: Use, who: []const u8) error{UnsupportedCheckpoint}!Plan {
    switch (check(C, caps, lins, use)) {
        .ok => |p| return p,
        .refused => |r| {
            explain(r, who);
            return error.UnsupportedCheckpoint;
        },
    }
}

/// Log why `check` refused, under `who`.
pub fn explain(r: Refusal, who: []const u8) void {
    switch (r.why) {
        .mixed_convrot => std.log.err("{s}: this checkpoint mixes convrot and plain int8 block linears; one activation prep serves a whole group, so there is no correct one", .{who}),
        .no_kernel => std.log.err("{s}: {s} is {t}, which this backend has no kernel for", .{ who, r.tag, r.dtype }),
        .int_shape => std.log.err("{s}: {s} is [{d}, {d}] {t}; this backend's int8 GEMM cannot take that shape", .{ who, r.tag, r.rows, r.cols, r.dtype }),
        .int4_unrotated => std.log.err("{s}: {s} is int4 without convrot, which this backend's int4 path cannot pair with", .{ who, r.tag }),
        .blockq_basis => std.log.err("{s}: {s} decodes to int8 in a different rotation basis than the checkpoint's own int8 linears", .{ who, r.tag }),
        .nvfp4_shape => std.log.err("{s}: {s} is [{d}, {d}] nvfp4; the GEMM it feeds needs rows % 128 == 0 and cols % 32 == 0", .{ who, r.tag, r.rows, r.cols }),
        .w4a8_group => std.log.err("{s}: {s} is W4A8 with a group_size that is not a multiple of 8; the decode kernel needs one scale per 8 columns", .{ who, r.tag }),
        .gemm_shape => std.log.err("{s}: {s} is [{d}, {d}] {t}, a shape no kernel here takes at more than one row", .{ who, r.tag, r.rows, r.cols, r.dtype }),
        .gemv_width => std.log.err("{s}: {s} is [{d}, {d}] {t}; no decode GEMV here takes that shape", .{ who, r.tag, r.rows, r.cols, r.dtype }),
        .too_wide => std.log.err("{s}: {s} has {d} rows, wider than the scratch this caller sized", .{ who, r.tag, r.rows }),
        .no_row_scale => std.log.err("{s}: {s} is {t} with no per-row scale for its {d} rows; its companion weight_scale is missing or the loader did not read it", .{ who, r.tag, r.dtype, r.rows }),
    }
}

/// Rows at or below which a block quant with a grouped kernel takes it over the GEMMs
/// (measured crossover against the f16 dequant GEMM, ~35-48 rows on a 3090).
pub const grouped_max = 40;

/// Rows at or below which a dense bf16 or fp8 weight takes the 4-row grouped GEMV.
pub const dense_grouped_max = @import("tp_core").spec_limits.max_draft + 1;

/// Below this many output rows `h16_f32` takes the f32 GEMM: padding `co` to the tensor
/// core tile costs more than it returns.
pub const h16_min_co: usize = 96;

/// Whether the f16 GEMMs tile this shape: 128-row blocks over 32-wide k slabs.
pub fn gemmShape(w: Weight) bool {
    return w.rows % 128 == 0 and w.cols % 32 == 0;
}

/// The route `w` takes at `m` rows under `pol`, or null when nothing here reads it.
///
/// `C` is the backend's capability adapter. Block quants: a dequant GEMM in f16
/// (`quantKernel`) and bf16 (`dequantBf16`), which first (`dequantF16First`), a decode GEMV (`quantGemv`), an int8
/// convrot decode (`blockQDecode`), MMQ (`mmqDtype`, `mmqShape`, `mmqFaster`), the
/// dp4a GEMVs (`q8Batch`, `q8N`) and the dual row GEMV (`dualOnly`, `dual`, `dualQ8`, `dualN`).
/// Integer convrot: the int8 GEMM (`i8Gemm`), native int4 (`i4Native`) or an int4
/// decode to int8 (`i4Decode`), W4A8 (`w4a8`), weight-only twins (`intWeightOnly`),
/// NVFP4 (`nvfp4`). Dense: the 16-bit GEMMs (`denseGemm`, `denseGemmTakes`), the f16
/// dense arm (`h16`), fp8 (`fp8Gemm`, `fp8Bf16`, `fp8F32`), the grouped GEMVs
/// (`groupedTakes`) and the 16-bit and f32 GEMVs (`gemvF16`, `gemvF32`).
pub fn routeOf(comptime C: type, caps: C, w: Weight, m: usize, pol: Policy) ?Route {
    std.debug.assert(m >= 1);
    switch (lin.kindOf(w.dtype)) {
        .blockq => return if (pol.gemv) blockQGemv(C, caps, w, m, pol) else blockQBatch(C, caps, w, pol),
        .i8 => return intRoute(C, caps, pol, .i8, .{ .i8_wo, .i8_wo16, .i8_wo32 }),
        .i4 => return if (caps.i4Native())
            intRoute(C, caps, pol, .i4, .{ .i4_wo, .i4_wo16, .i4_wo32 })
        else if (pol.int8 != .w8a8)
            woRoute(C, caps, pol, .{ .i4_wo, .i4_wo16, .i4_wo32 })
        else if (caps.i4Decode() and caps.i8Gemm()) .i4_i8 else null,
        .w4a8 => return if (caps.w4a8()) intRoute(C, caps, pol, .w4a8, .{ .w4a8_wo, .w4a8_wo16, .w4a8_wo32 }) else null,
        .nvfp4 => return if (caps.nvfp4()) .nvfp4 else null,
        .svdq => return if (!caps.svdq()) null else intRoute(C, caps, pol, .svdq, .{ .svdq_wo, .svdq_wo16, .svdq_wo32 }),
        .other => return null,
        else => {},
    }
    if (pol.dense == .f16 and caps.h16()) return switch (w.dtype) {
        .f32 => .h16_f32,
        .f16 => .h16_f16,
        .bf16 => .h16_bf16,
        .f8_e4m3 => fp8Route(C, caps, w, pol),
        else => unreachable,
    };
    if (!pol.gemv) return switch (w.dtype) {
        .f32 => .gemm_f32,
        .f16 => if (caps.denseGemm(.f16)) .gemm_f16 else if (caps.wideGemm()) .gemm_f32 else null,
        .bf16 => if (caps.denseGemm(.bf16)) .gemm_bf16 else if (caps.wideGemm()) .gemm_f32 else null,
        .f8_e4m3 => fp8Route(C, caps, w, pol),
        else => unreachable,
    };
    const grouped = caps.groupedTakes(w);
    return switch (w.dtype) {
        .bf16 => if (m == 1)
            .gemv_bf16
        else if ((m <= dense_grouped_max or !caps.denseGemmTakes(w)) and grouped)
            .gemv_bf16n
        else if (caps.denseGemmTakes(w))
            .gemm_bf16
        else if (caps.wideGemm())
            .gemm_f32
        else
            null,
        .f16 => if (m == 1 and caps.gemvF16())
            .gemv_f16
        else if (caps.denseGemm(.f16) and w.cols % 32 == 0) .gemm_f16 else if (caps.wideGemm()) .gemm_f32 else null,
        .f8_e4m3 => if (m == 1)
            .gemv_fp8
        else if (m <= dense_grouped_max and grouped)
            .gemv_fp8n
        else if (caps.fp8Gemm(w) and caps.denseGemmTakes(w))
            fp8Route(C, caps, w, pol)
        else if (caps.fp8F32())
            .gemm_fp8_f32
        else if (grouped) .gemv_fp8n else null,
        .f32 => if (!caps.gemvF32())
            .gemm_f32
        else if (caps.f32Rows(w, m))
            .gemv_f32_rows
        else if (m == 1)
            .gemv_f32
        else if (m <= dense_grouped_max and grouped) .gemv_f32n else .gemm_f32,
        else => unreachable,
    };
}

/// An integer convrot weight: weight-only where the policy asks and the backend has
/// it, else the quantized-activation GEMM.
fn intRoute(comptime C: type, caps: C, pol: Policy, q: Route, wo: [3]Route) ?Route {
    if (pol.int8 != .w8a8) return woRoute(C, caps, pol, wo);
    return if (caps.i8Gemm()) q else null;
}

/// The weight-only route at the width `pol.int8` asks for, `wo` = { bf16, f16, f32 }. A
/// 16-bit width the device has no GEMM for takes f32, which has both its range and its
/// precision; null only where there is no f32 either, so a plan refuses it by name.
fn woRoute(comptime C: type, caps: C, pol: Policy, wo: [3]Route) ?Route {
    const i: usize = switch (pol.int8) {
        .w8a8 => unreachable,
        .weight_only => 0,
        .weight_only_f16 => 1,
        .weight_only_f32 => 2,
    };
    if (caps.intWeightOnly(pol.int8)) return wo[i];
    return if (caps.intWeightOnly(.weight_only_f32)) wo[2] else null;
}

/// fp8 as a batch: staged at the width `Act16` asks for, or against the f32 activation
/// where the backend cannot stage it so.
fn fp8Route(comptime C: type, caps: C, w: Weight, pol: Policy) ?Route {
    if (pol.act16 == .bf16) return if (caps.fp8Bf16()) .gemm_fp8_bf16 else if (caps.fp8F32()) .gemm_fp8_f32 else null;
    if (caps.fp8Gemm(w)) return .gemm_fp8;
    return if (caps.fp8F32()) .gemm_fp8_f32 else null;
}

/// A block quant as a batch. A shape the chosen decode cannot take (`cols % 1024` for
/// the chunked int8/int4 decode, the MMQ tile) takes the 16-bit expansion instead: f16
/// only where the activation fits (`Act16`), and of the two the one the backend tries
/// first (`dequantF16First`).
fn blockQBatch(comptime C: type, caps: C, w: Weight, p: Policy) ?Route {
    const dt = w.dtype;
    const wide_bf: ?Route = if (caps.dequantBf16(dt)) .blockq_bf16 else null;
    const wide_h: ?Route = if (p.act16 == .f16 and caps.quantKernel(dt)) .blockq_f16 else null;
    const wide = if (caps.dequantF16First()) wide_h orelse wide_bf else wide_bf orelse wide_h;
    const r: Route = switch (p.blockq) {
        .f16 => return if (caps.quantKernel(dt)) .blockq_f16 else null,
        .llm => {
            if (!caps.quantKernel(dt)) return null;
            if (caps.mmqFaster(dt, w.rows, w.cols)) return .blockq_mmq;
            return if (gemmShape(w)) .blockq_f16 else null;
        },
        .mmq => if (caps.mmqDtype(dt)) .blockq_mmq else return wide,
        .int8 => if (caps.blockQDecode(dt)) .blockq_i8 else return wide,
        .int4 => if (caps.blockQDecode(dt)) .blockq_i4 else return wide,
        .bf16 => return wide_bf orelse wide_h,
        .auto => if (caps.blockQDecode(dt) and (dt == .q2_k or dt == .q4_k)) .blockq_i8 else return wide,
    };
    return switch (r) {
        .blockq_mmq => if (caps.mmqShape(dt, w.rows, w.cols)) r else wide,
        .blockq_i8, .blockq_i4 => if (w.cols % 1024 == 0 and w.rows % 128 == 0) r else wide,
        else => r,
    };
}

/// A block quant under the GEMV policy: a decode kernel at one row, a grouped one for a
/// few, the batch choice beyond.
fn blockQGemv(comptime C: type, caps: C, w: Weight, m: usize, pol: Policy) ?Route {
    const dt = w.dtype;
    if (!caps.quantGemv(dt)) return null;
    // The dp4a kernels tile 256-column groups and rows in warps of 8.
    const dp4a = w.cols % 256 == 0 and w.rows % 8 == 0;
    if (caps.dualOnly(dt)) {
        // The dual GEMV reads whole codebook and quant words, so a row must be whole
        // blocks; its q8 twin needs the staged activation of a single row. A row that
        // is not takes the GEMM even at one row.
        if (w.cols % dt.blockElems() == 0) {
            // A batch takes decode's activation, so a verify reproduces it.
            const q8 = pol.decode_dp4a and caps.dualQ8(dt);
            if (m == 1) return if (q8) .gemv_dual_q8 else .gemv_dual;
            if (caps.dualN(dt) and m <= grouped_max) return if (q8) .gemv_dualn_q8 else .gemv_dualn;
        }
        return if (caps.quantKernel(dt)) .blockq_f16 else null;
    }
    if (m == 1 and pol.dual_decode and caps.dual(dt) and w.cols % 256 == 0)
        return if (pol.decode_dp4a) .gemv_dual_q8 else .gemv_dual;
    if (m == 1) {
        if (pol.decode_dp4a and dp4a) {
            if (dt == .q5_k or dt == .q6_k or caps.q8Batch(dt)) return .gemv_q8;
            if (caps.q8N(dt)) return .gemv_q8n;
        }
        // The k-quant GEMVs stage per-sub-block scales in an 8 KiB shared table.
        return if (w.cols <= 32768) .gemv_q else null;
    }
    // A weight the GEMM tiles cannot take (a router's expert count, a GDN gate's head
    // count, an odd vocab) is a GEMV question at any m.
    const skinny = !gemmShape(w);
    switch (pol.force) {
        .auto => {},
        .grouped => {
            if (dp4a and caps.q8Batch(dt)) return .gemv_q8batch;
            if (dp4a and caps.q8N(dt)) return .gemv_q8n;
        },
        .mmq => if (!skinny and caps.mmqShape(dt, w.rows, w.cols)) return .blockq_mmq,
        .dequant => if (!skinny) return .blockq_f16,
    }
    if (dp4a and caps.q8Batch(dt) and (skinny or m <= grouped_max)) return .gemv_q8batch;
    if (dp4a and caps.q8N(dt) and (skinny or m <= grouped_max)) return .gemv_q8n;
    if (skinny) return if (w.cols <= 32768) .gemv_q else null;
    return blockQBatch(C, caps, w, pol);
}
