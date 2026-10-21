//! The one CUDA linear dispatcher, on both CUDA arms, for every caller: diffusion trunks,
//! text encoders and LLM steppers.
//!
//! A forward calls `prep` once per activation a group of linears shares and gets a
//! `Group` back, then `gemm` per linear with that group, and names no dtype. The route
//! is chosen PER WEIGHT by `dual_lin.routeOf` from the weight's storage and shape, the
//! row count and the caller's `Plan`, so a checkpoint that quantizes by layer or mixes
//! GGUF formats by width computes correctly, and a format wired here reaches every
//! caller the same day. This file only executes a route (`exec`); it never chooses one.
//!
//! The int8 / int4 convrot, MMQ and dp4a GEMV routes read the activation the prep
//! staged in backend state and ignore `x`; the rest read `x`. A group carries its
//! prep's serial, so a second prep into the same scratch slipped between a prep and its
//! GEMM is an assert, not a silently wrong activation.
//!
//! `plan` accepts or refuses a model's device linears up front, naming the first
//! problem, so a checkpoint fails at load and never inside a forward.
//!
//! A LoRA stack on `Plan.lora` is applied after every `gemm` (`sidecar`); a device GEMM
//! outside this dispatcher must call `sidecarDense` itself, since a linear run bare is
//! a wrong render with no error.
//!
//! `x` and `y` may each be f32, f16 or bf16 on every route: a `routeTakes16` route on the
//! cuBLASLt arm reads and writes them as they are, any other gets an f32 copy of `x` from
//! `prep` and writes through an f32 plane `gemm` narrows into `y` (`Backend.lin_x32`,
//! `lin_y32`).

const std = @import("std");
const lin = @import("lin.zig");
pub const dual_lin = @import("dual_lin.zig");
const lora_mod = @import("lora.zig");
const lora_cuda = @import("lora_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
pub const Weight = lin.Weight;
pub const DType = lin.DType;

pub const Route = dual_lin.Route;
pub const Prep = dual_lin.Prep;
pub const IntGemm = dual_lin.IntGemm;
pub const DenseCompute = dual_lin.DenseCompute;
pub const BlockQGemm = dual_lin.BlockQGemm;
pub const Force = dual_lin.Force;
pub const grouped_max = dual_lin.grouped_max;
pub const dense_grouped_max = dual_lin.dense_grouped_max;

/// The DiT's block-quant route, overridden by `--dit-gguf-gemm`. Every diffusion family
/// passes this to `plan`; the text encoders pass their own.
///
/// What a re-quantizing route costs a format depends on how the format's own error
/// compares to the regrid's, so `auto` takes int8 for q2_k and q4_k (0.03% and 0.7% more
/// weight error) and bf16 for the rest, where q8_0 alone would lose 84%. int4 is never
/// `auto`: it is W4A4, and its 16-level ACTIVATIONS cost more than the weight regrid
/// saves, 2.4 dB for 1.11x the step even on q2_k, whose weight barely notices the regrid.
pub var blockq_gemm: BlockQGemm = .auto;

/// The block-quant route every text encoder passes to `plan`, `--te-gguf-gemm`.
///
/// `f16` by default, not the DiT's `auto`: the int8 and MMQ routes quantize the
/// ACTIVATION to 8 bits, and a conditioning tensor is computed once and then steers every
/// sampling step, so that error is baked into the whole render for a fraction of a second
/// saved once. The other values exist to measure exactly that trade (`te-test
/// --gguf-gemm`).
pub var encoder_gemm: BlockQGemm = .f16;

/// DIAGNOSTIC: whether the block-quant int8 decode and the prep that pairs with it
/// rotate. Skipping it on both sides is exact arithmetic and worth ~8% of a step, and
/// it measures rel RMSE 0.915 against the CPU forward, i.e. output uncorrelated: the
/// activation's per-row absmax quantization needs the rotation to spread outliers.
pub var blockq_rotate: bool = true;

/// DIAGNOSTIC: adds every bias in a separate pass instead of the GEMM's epilogue, so
/// the fold can be timed against itself in one process (`diffusion-profile --ab split-bias`).
pub var split_bias: bool = false;

/// DIAGNOSTIC: false runs a sidecar's A GEMM on the f32 activation even where the
/// base staged an int8 one, the old arm `lora-cuda-bench` times against.
pub var lora_i8_a: bool = true;

pub const Plan = dual_lin.Plan;
pub const Use = dual_lin.Use;

/// The route `w` takes at `m` rows under `p`.
pub fn routeAt(p: Plan, w: Weight, m: usize) ?Route {
    return dual_lin.routeOf(Caps, .{}, w, m, p.policy());
}

/// What the route for `w` at `m` rows does to its activation, for a device check's CPU
/// reference (`ops.matmul.act_quant`). The integer convrot routes only: a block-quant
/// int8 decode also re-quantizes the weight, which this does not model.
pub fn actQuant(_: *Backend, p: Plan, w: Weight, m: usize) ops.matmul.ActQuant {
    const r = routeAt(p, w, m) orelse return .{};
    return switch (r) {
        .i8, .i4_i8, .w4a8 => .{ .bits = 8, .rotate = p.rot },
        .i4 => .{ .bits = 4, .rotate = p.rot },
        else => .{},
    };
}

/// The route `w` takes as a batch; what a DiT plan takes at every row count.
pub fn route(p: Plan, w: Weight) ?Route {
    return routeAt(p, w, 1);
}

/// `p` under `mode`, or unchanged where the device has no bf16 tensor cores.
pub fn withInt8(p: Plan, be: *const Backend, mode: IntGemm) Plan {
    var out = p;
    out.int8 = if (be.ctx.cc_major >= 8) mode else .w8a8;
    return out;
}

/// The route `w` takes under a DiT plan with block-quant route `pol`, the integer
/// weights on their quantized-activation routes (what `check` holds the shapes against).
pub fn routeOf(w: Weight, pol: BlockQGemm) ?Route {
    return route(.{ .rot = false, .blockq = pol }, w);
}

/// Whether this dtype ends up in a W4A4 GEMM here, activation included. A GGUF's route
/// is a runtime choice its dtype does not show, so a check calibrated on how far a route
/// may drift from the weight-only CPU forward has to ask this, not the storage dtype.
pub fn activationIs4Bit(pol: BlockQGemm, dt: DType) bool {
    return dt == .i4 or (dt.isBlockQuant() and routeOf(.{ .bytes = &.{}, .dtype = dt, .rows = 128, .cols = 1024 }, pol) == .blockq_i4);
}

/// Whether route `r` reads and writes 16-bit activations (f16 or bf16): the int8 prep
/// and its GEMMs do, and so do the `h16_*` routes; every other route is f32 on both sides.
pub fn routeTakes16(r: Route) bool {
    return switch (r) {
        .i8, .w4a8, .blockq_i8, .h16_f32, .h16_f16, .h16_bf16 => true,
        else => false,
    };
}

/// Whether every weight here can sit in a 16-bit activation chain, which a family
/// asks once to decide its activation storage.
pub fn takes16(p: Plan, ws: []const Weight) bool {
    for (ws) |w| if (!routeTakes16(route(p, w) orelse return false)) return false;
    return true;
}

/// Whether every weight here runs on the int8 prep, which is what lets a caller chain
/// them through 16-bit activations.
pub fn allI8(p: Plan, ws: []const Weight) bool {
    for (ws) |w| if ((route(p, w) orelse return false).prep() != .i8) return false;
    return true;
}

pub const Why = dual_lin.Why;
pub const Refusal = dual_lin.Refusal;
pub const Verdict = dual_lin.Verdict;

/// Accept or refuse a model's device linears for every row count `use` runs, naming
/// the first problem.
pub fn check(lins: []const Weight, use: Use) Verdict {
    return dual_lin.check(Caps, .{}, lins, use);
}

/// `check`, logging the refusal under `who` and returning it as an error.
pub fn plan(lins: []const Weight, use: Use, who: []const u8) error{UnsupportedCheckpoint}!Plan {
    return dual_lin.plan(Caps, .{}, lins, use, who);
}

/// One activation staged for a group of linears.
pub const Group = struct {
    x: Buf,
    m: usize,
    cols: usize,
    prep: Prep,
    /// Row count the q8 layout was written for: `m`, or `m` padded to the MMQ tile
    /// when a linear in the group runs MMQ. The grouped GEMV reads either.
    layout_rows: usize,
    serial: u32,
    /// `x` widened to f32, for the routes in the group that cannot read it as it is.
    x32: ?Buf = null,
};

/// Bumped by every prep that writes the scratch, so a group can prove its scratch still
/// holds its activation. The int8 and int4 preps share one scratch, MMQ and the q8
/// GEMVs another.
var serial_int: u32 = 0;
var serial_q8: u32 = 0;

/// Stage the activation `x[m][cols]` once for every linear in `group`, if any of them
/// needs it. A prep writes backend state and leaves `x` intact, so a linear in the
/// same group that reads `x` still reads it at its own dtype; only the int8 prep and
/// the `h16_*` routes read a 16-bit one, and the group carries an f32 copy for the rest.
pub fn prep(be: *Backend, p: Plan, x: Buf, m: usize, cols: usize, group: []const Weight) !Group {
    var want: Prep = .none;
    var bq = false;
    var widen = false;
    for (group) |w| {
        const r = routeAt(p, w, m) orelse return error.UnsupportedDType;
        std.debug.assert(w.cols == cols);
        if (x.dt != .f32 and !typedOn(be, r)) widen = true;
        const k = r.prep();
        if (k == .none) continue;
        if (r == .blockq_i8 or r == .blockq_i4) bq = true;
        if (want == .none or want == k) {
            want = k;
        } else if ((want == .q8 or want == .mmq) and (k == .q8 or k == .mmq)) {
            // MMQ's tile-padded q8 layout serves the grouped GEMV too.
            want = .mmq;
        } else {
            // Two preps of one activation would need two live prep states, which the
            // backend does not have. `plan` does not see groups, so this is checked here.
            std.log.err("lin_cuda: a linear group mixes {t} and {t} activation preps; one cannot serve both", .{ want, k });
            return error.UnsupportedCheckpoint;
        }
    }
    var g = Group{ .x = x, .m = m, .cols = cols, .prep = want, .layout_rows = m, .serial = 0 };
    if (widen) {
        try be.ensureDeviceBuffer(&be.lin_x32, m * cols * 4);
        const x32 = be.lin_x32.as(.f32);
        try be.opCast(x, x32, m * cols, 0);
        g.x32 = x32;
    }
    // The int8 prep reads a 16-bit `x` where `typedOn` says; the rest read f32.
    const xf = g.x32 orelse x;
    switch (want) {
        .none => {},
        .i8 => {
            try be.opI8PrepR(if (be.kernels == .libs) x else xf, m, cols, if (bq) blockq_rotate else p.rot);
            serial_int +%= 1;
            g.serial = serial_int;
        },
        .i4 => {
            try be.opI4Prep(xf, m, cols);
            serial_int +%= 1;
            g.serial = serial_int;
        },
        .q8 => {
            try be.opGemvQuantizeX(xf, m * cols);
            serial_q8 +%= 1;
            g.serial = serial_q8;
        },
        .mmq => {
            try be.opMatmulQuantMmqPipePrep(xf, m, cols);
            g.layout_rows = std.mem.alignForward(usize, m, Backend.mmq_pipe_tile);
            serial_q8 +%= 1;
            g.serial = serial_q8;
        },
    }
    return g;
}

/// One linear `y[m][w.rows] = x[m][w.cols] @ Wᵀ (+ bias)` against a prepped group, then
/// its LoRA hits, `y` written at its own dtype. The bias is added in the GEMM's own epilogue
/// where the route's kernel takes one and in a second pass on the rest (fp8, the f16
/// block-quant decode, MMQ, the GEMVs); it is cached on the device by host pointer, so it
/// must outlive the model.
///
/// Buffer requirements the dispatcher cannot check, since it never sees a buffer's
/// size: `y` must hold `align(m, 128)` rows on every route whose prep is not `.none` and
/// on the dequant GEMMs (they launch over padded rows and each block stores a whole
/// tile), and `x` must have `align(m, 4)` rows of backing store on the dense grouped
/// GEMVs (they read 4 rows and predicate the outputs).
pub fn gemm(be: *Backend, p: Plan, g: Group, y: Buf, w: Weight, bias: ?[]const f32) !void {
    const r = routeAt(p, w, g.m) orelse return error.UnsupportedDType;
    std.debug.assert(w.cols == g.cols);
    // A prepped route needs the scratch this group wrote, and nothing since.
    switch (r.prep()) {
        .none => {},
        .i8, .i4 => std.debug.assert(g.prep == r.prep() and g.serial == serial_int),
        .q8 => std.debug.assert((g.prep == .q8 or g.prep == .mmq) and g.serial == serial_q8),
        .mmq => std.debug.assert(g.prep == .mmq and g.serial == serial_q8),
    }
    const typed = typedOn(be, r);
    const x = if (typed) g.x else g.x32 orelse g.x;
    std.debug.assert(x.dt == .f32 or typed);
    const o = Opts{ .rot = p.rot, .bq_rotate = blockq_rotate, .layout_rows = g.layout_rows };
    const split = split_bias and bias != null;
    // A route that writes only f32 fills a plane `y` takes a narrowing copy of; the int8
    // and dequant GEMMs store whole 128-row tiles, so it holds that many.
    const ys = if (y.dt == .f32 or typed) y else blk: {
        try be.ensureDeviceBuffer(&be.lin_y32, std.mem.alignForward(usize, g.m, 128) * w.rows * 4);
        break :blk be.lin_y32.as(.f32);
    };
    try exec(be, r, o, ys, x, g.m, w, if (split) null else bias);
    if (p.lora) |stack| try sidecar(be, p, r, stack, ys, x, g.m, w, if (split) null else bias, true);
    if (split) try addBias(be, ys, g.m, w.rows, bias.?);
    if (ys.buf != y.buf) try be.opCast(ys, y, g.m * w.rows, 0);
}

/// One linear of a one-shot tower under `Plan.tower`, which a ViT, an embedder or a
/// text fusion calls per weight (its prep is the weight's own, so the rotation is too),
/// with `stack`'s LoRA hits on it.
pub fn towerLinear(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32, stack: ?*const lora_mod.Stack) !void {
    var p = Plan.tower;
    p.rot = w.convrot != 0;
    p.lora = stack;
    if (routeAt(p, w, m) == null) {
        std.log.err("lin_cuda: {s} is {t}, which no tower route here takes", .{ w.tag orelse "<untagged>", w.dtype });
        return error.UnsupportedDType;
    }
    return linear(be, p, y, x, m, w, bias);
}

/// Whether route `r` reads and writes 16-bit activations as they are on this arm: the
/// 16-bit GEMMs on both (`opGemm16`), the int8 preps only on the vendor one.
fn typedOn(be: *const Backend, r: Route) bool {
    return switch (r) {
        .h16_f32, .h16_f16, .h16_bf16 => true,
        else => be.kernels == .libs and routeTakes16(r),
    };
}

/// `prep` and `gemm` for one linear on its own.
pub fn linear(be: *Backend, p: Plan, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
    const g = try prep(be, p, x, m, w.cols, &.{w});
    try gemm(be, p, g, y, w, bias);
}

/// The CUDA backend's capabilities, as `dual_lin.routeOf` asks for them.
pub const Caps = struct {
    pub fn quantKernel(_: Caps, dt: DType) bool {
        return Backend.quantKernelSupported(dt);
    }
    /// The block-quant int8 / int4 convrot decode (`opI8GemmBlockQ`).
    pub fn blockQDecode(_: Caps, dt: DType) bool {
        return Backend.blockQFormat(dt) != null;
    }
    pub fn mmqDtype(_: Caps, dt: DType) bool {
        return Backend.mmqPipeDtype(dt);
    }
    pub fn mmqShape(_: Caps, dt: DType, rows: usize, cols: usize) bool {
        return Backend.mmqPipeSupported(dt, rows, cols);
    }
    pub fn mmqFaster(_: Caps, dt: DType, rows: usize, cols: usize) bool {
        return Backend.mmqPipeFaster(dt, rows, cols);
    }
    pub fn dualOnly(_: Caps, dt: DType) bool {
        return Backend.dualGemvOnly(dt);
    }
    pub fn dual(_: Caps, dt: DType) bool {
        return Backend.dualGemvSupported(dt);
    }
    pub fn q8Batch(_: Caps, dt: DType) bool {
        return Backend.quantQ8BatchSupported(dt);
    }
    pub fn q8N(_: Caps, dt: DType) bool {
        return Backend.quantQ8NSupported(dt);
    }
    pub fn quantGemv(_: Caps, dt: DType) bool {
        return Backend.quantKernelSupported(dt);
    }
    pub fn dequantBf16(_: Caps, dt: DType) bool {
        return Backend.quantKernelSupported(dt);
    }
    pub fn dequantF16First(_: Caps) bool {
        return false;
    }
    pub fn dualQ8(_: Caps, dt: DType) bool {
        return Backend.dualGemvSupported(dt);
    }
    /// The dp4a grouped GEMVs take a small batch here.
    pub fn dualN(_: Caps, _: DType) bool {
        return false;
    }
    pub fn i8Gemm(_: Caps) bool {
        return true;
    }
    pub fn i4Native(_: Caps) bool {
        return true;
    }
    pub fn i4Decode(_: Caps) bool {
        return false;
    }
    pub fn w4a8(_: Caps) bool {
        return true;
    }
    /// The weight-only twins need the bf16 tensor cores; `Plan.withInt8` keeps them
    /// off a card without.
    pub fn intWeightOnly(_: Caps, _: dual_lin.IntGemm) bool {
        return true;
    }
    pub fn nvfp4(_: Caps) bool {
        return true;
    }
    pub fn h16(_: Caps) bool {
        return true;
    }
    /// Never needed: every CUDA arm has a 16-bit GEMM.
    pub fn wideGemm(_: Caps) bool {
        return false;
    }
    pub fn denseGemm(_: Caps, _: DType) bool {
        return true;
    }
    pub fn denseGemmTakes(_: Caps, w: Weight) bool {
        return dual_lin.gemmShape(w);
    }
    pub fn fp8Gemm(_: Caps, _: Weight) bool {
        return true;
    }
    pub fn fp8Bf16(_: Caps) bool {
        return true;
    }
    pub fn fp8F32(_: Caps) bool {
        return false;
    }
    /// The grouped dense GEMVs run 8 rows per block.
    pub fn groupedTakes(_: Caps, w: Weight) bool {
        return w.rows % 8 == 0;
    }
    pub fn gemvF16(_: Caps) bool {
        return true;
    }
    pub fn gemvF32(_: Caps) bool {
        return false;
    }
    pub fn f32Rows(_: Caps, _: Weight, _: usize) bool {
        return false;
    }
    /// What the kernel behind route `r` cannot take about `w`, or null.
    pub fn refuse(_: Caps, lins: []const Weight, rot: bool, w: Weight, r: Route) ?Why {
        switch (r.prep()) {
            // The int8 GEMMs launch `rows / 128` blocks and the prep rotates in groups of
            // 256. The weight-only twins need no more than this, which is what lets
            // `Plan.int8` be set after the fact.
            .i8, .i4 => if (w.rows % 128 != 0 or w.cols % 256 != 0) return .int_shape,
            .mmq, .none, .q8 => {},
        }
        return switch (r) {
            // `opI4Prep` always rotates and has no unrotated build.
            .i4 => if (!rot) .int4_unrotated else null,
            // A block-quant decode and a convrot weight in one group would need one prep
            // in two bases.
            .blockq_i8, .blockq_i4 => if (rot != blockq_rotate and (lin.any(lins, .i8) or lin.any(lins, .i4) or lin.any(lins, .w4a8))) .blockq_basis else null,
            .nvfp4 => if (!dual_lin.gemmShape(w)) .nvfp4_shape else null,
            else => null,
        };
    }
};

/// What `exec` needs besides the route and the operands.
pub const Opts = struct {
    /// Whether the checkpoint's integer convrot weights are rotated (`lin.convrot`); the
    /// weight-only routes rotate the expanded weight back by it.
    rot: bool = true,
    /// Whether the block-quant int8 / int4 decode rotates (its prep must agree).
    bq_rotate: bool = true,
    /// Rows the GEMVs' q8 activation was laid out for: `m`, or `m` padded to the MMQ
    /// tile when the same prep also served an MMQ linear.
    layout_rows: ?usize = null,
};

/// `y[m][w.rows] = x[m][w.cols] @ Wᵀ (+ bias)` on route `r`, the bias in the GEMM's
/// epilogue where the route's kernel takes one and in a second pass where it does not
/// (fp8, the f16 block-quant decode, MMQ, the GEMVs).
///
/// A route whose prep is not `.none` reads the activation the prep staged, not `x`, and
/// the int8 / MMQ / dequant GEMMs write `align(m, 128)` rows of `y`.
pub fn exec(be: *Backend, r: dual_lin.Route, o: Opts, y: Buf, x: Buf, m: usize, w: dual_lin.Weight, bias: ?[]const f32) !void {
    if (bias) |b| std.debug.assert(b.len >= w.rows);
    if (r.prep() == .none) try probeInput(be, x, m, w);
    // For the kernels that take no null bias: the backend's zeros, whole, never a slice
    // (a bias is cached by host pointer and sized from the first call's length).
    const zb: []const f32 = bias orelse if (readsZeroBias(r)) try be.zeroBias(w.rows) else &.{};
    var folded = true;
    switch (r) {
        // Routes for kernels this backend lacks; `Caps` never chooses them.
        .gemm_fp8_f32, .gemv_f32, .gemv_f32n, .gemv_f32_rows, .i4_i8, .gemv_dualn, .gemv_dualn_q8 => return error.UnsupportedDType,
        .gemm_f32 => if (w.scale == 1)
            be.opMatmulF32Lt(y, x, m, w.bytes, w.rows, w.cols, bias) catch |err| switch (err) {
                error.UnsupportedKernelArm => try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, bias),
                else => return err,
            }
        else
            try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, bias),
        .gemm_f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, bias),
        // Ampere+ feeds raw bf16 straight to the tensor cores; older cards and shapes
        // off the tile take the f16 GEMM.
        .gemm_bf16 => if (be.ctx.cc_major >= 8 and w.rows % 128 == 0 and w.cols % 32 == 0)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, bias)
        else
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, zb),
        .gemm_fp8 => {
            try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemm_fp8_bf16 => try be.opMatmulFp8Bf16(y, x, m, w.bytes, w.scale, w.rows, w.cols, zb),
        .h16_f32 => if (w.rows >= dual_lin.h16_min_co or x.dt != .f32 or y.dt != .f32)
            try be.opConvF16Scaled(y, 0, x, m, w.bytes, w.rows, w.cols, zb, 1.0)
        else
            try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, 1.0, zb),
        .h16_f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, zb),
        .h16_bf16 => try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, zb),
        .gemv_bf16 => {
            try be.opGemvBf16(y, x, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_bf16n => {
            var off: usize = 0;
            while (off < m) : (off += 4) {
                const n: usize = @min(4, m - off);
                try be.opGemvBf16N(rowsOf(y, w.rows, off, n), rowsOf(x, w.cols, off, 4), w.bytes, w.scale, w.rows, w.cols, n);
            }
            folded = bias == null;
        },
        .gemv_f16 => {
            try be.opGemvF16(y, x, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_fp8 => {
            try be.opGemvFp8(y, x, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_fp8n => {
            var off: usize = 0;
            while (off < m) : (off += 4) {
                const n: usize = @min(4, m - off);
                try be.opGemvFp8N(rowsOf(y, w.rows, off, n), rowsOf(x, w.cols, off, 4), w.bytes, w.scale, w.rows, w.cols, n);
            }
            folded = bias == null;
        },
        .i8 => try be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, bias),
        // The packed 4-bit weight is decoded into the backend's scratch on the way in and
        // runs the same int8 GEMM, so the 4-bit form stays resident.
        .w4a8 => {
            const meta = w.w4a8.?;
            try be.opI8GemmW4A8(y, w.bytes, meta.s_rel, meta.levelBytes(), w.row_scale.?, w.rows, w.cols, meta.group_size, bias);
        },
        .i4 => try be.opI4Gemm(y, w.bytes, w.row_scale.?, w.rows, bias),
        // Expanded to bf16 per call, rotated back into the checkpoint's basis, against
        // the f32 activation.
        .i8_wo => try be.opI8GemmWo(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, o.rot, zb),
        .i4_wo => try be.opI4GemmWo(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, zb),
        .w4a8_wo => {
            const meta = w.w4a8.?;
            try be.opW4A8GemmWo(y, x, m, w.bytes, meta.s_rel, meta.levelBytes(), w.row_scale.?, w.rows, w.cols, meta.group_size, o.rot, zb);
        },
        .i8_wo16 => try be.opI8GemmWoF16(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, o.rot, zb),
        .i8_wo32 => try be.opI8GemmWoF32(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, o.rot, bias),
        .i4_wo32 => try be.opI4GemmWoF32(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, bias),
        .w4a8_wo32 => {
            const meta = w.w4a8.?;
            try be.opW4A8GemmWoF32(y, x, m, w.bytes, meta.s_rel, meta.levelBytes(), w.row_scale.?, w.rows, w.cols, meta.group_size, o.rot, bias);
        },
        .i4_wo16 => try be.opI4GemmWoF16(y, x, m, w.bytes, w.row_scale.?, w.rows, w.cols, zb),
        .w4a8_wo16 => {
            const meta = w.w4a8.?;
            try be.opW4A8GemmWoF16(y, x, m, w.bytes, meta.s_rel, meta.levelBytes(), w.row_scale.?, w.rows, w.cols, meta.group_size, o.rot, zb);
        },
        // Weight-only: decoded to a 16-bit scratch inside the GEMM, the packed form resident.
        .nvfp4 => {
            const meta = w.nvfp4.?;
            try be.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zb);
        },
        .blockq_i8 => try be.opI8GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, o.bq_rotate, bias),
        .blockq_i4 => try be.opI4GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, o.bq_rotate, bias),
        // Expands the weight to bf16 in a scratch and runs the bf16 tensor cores.
        .blockq_bf16 => try be.opMatmulQuantBf16(w.dtype, y, x, m, w.bytes, w.rows, w.cols, zb),
        // The same expansion in f16, and the f16 GEMM the LLM prefill runs.
        .blockq_f16 => {
            try be.opMatmulQuant(w.dtype, y, x, m, w.bytes, w.rows, w.cols);
            folded = bias == null;
        },
        // The packed s8 straight to the int8 tensor cores, scale folded per 32-k substep,
        // against the q8_1 activation the prep staged.
        .blockq_mmq => {
            try be.opMatmulQuantMmqPipePrepped(w.dtype, y, m, w.bytes, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_q => {
            for (0..m) |t| try be.opGemvQuant(w.dtype, rowsOf(y, w.rows, t, 1), rowsOf(x, w.cols, t, 1), w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_dual_q8 => {
            try be.opGemvQuantDualQ8(w.dtype, y, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_dual => {
            try be.opGemvQuantDualF32(w.dtype, y, x, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_q8 => {
            try be.opGemvQuantQ8(w.dtype, y, w.bytes, w.scale, w.rows, w.cols);
            folded = bias == null;
        },
        .gemv_q8n => {
            const layout = o.layout_rows orelse m;
            var off: usize = 0;
            while (off < m) : (off += 8) {
                const ng: usize = @min(8, m - off);
                try be.opGemvQuantQ8N(w.dtype, rowsOf(y, w.rows, off, ng), w.bytes, w.scale, w.rows, w.cols, ng, off, layout);
            }
            folded = bias == null;
        },
        .gemv_q8batch => {
            // The batch kernel's layout is a function of its `n`, so a prep that also
            // served MMQ (tile-padded layout) falls back to the f32 GEMV per row.
            if ((o.layout_rows orelse m) != m) {
                for (0..m) |t| try be.opGemvQuant(w.dtype, rowsOf(y, w.rows, t, 1), rowsOf(x, w.cols, t, 1), w.bytes, w.scale, w.rows, w.cols);
            } else try be.opGemvQuantQ8Batch(w.dtype, y, w.bytes, w.scale, w.rows, w.cols, m);
            folded = bias == null;
        },
    }
    if (!folded) try addBias(be, y, m, w.rows, bias.?);
}

/// A bias added over the whole output plane, for a route whose kernel takes none.
pub fn addBias(be: *Backend, y: Buf, m: usize, rows: usize, b: []const f32) !void {
    const bd: Buf = .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(b)), .size = b.len * 4 };
    try be.opAddBiasRows(y, bd, m, rows, 0);
}

/// Rows `[row0, row0 + n)` of an f32 `[rows][width]` buffer.
fn rowsOf(b: Buf, width: usize, row0: usize, n: usize) Buf {
    return .{ .buf = @fromBackingInt(@intCast(@backingInt(b.buf) + row0 * width * 4)), .mem = b.mem, .size = n * width * 4 };
}

/// Hand the activation, widened to f32, to `ops.matmul.probe` (an activation capture),
/// if one is set.
pub fn probeInput(be: *Backend, x: Buf, m: usize, w: anytype) !void {
    const p = ops.matmul.probe orelse return;
    if (m == 0 or w.cols == 0) return;
    const n = m * w.cols;
    const host = be.gpa.alloc(f32, n) catch return error.OutOfMemory;
    defer be.gpa.free(host);
    if (x.dt == .f32) {
        try be.tensorDownload(x, std.mem.sliceAsBytes(host));
    } else {
        const raw = be.gpa.alloc(u16, n) catch return error.OutOfMemory;
        defer be.gpa.free(raw);
        try be.tensorDownload(x, std.mem.sliceAsBytes(raw));
        for (host, raw) |*h, r| h.* = switch (x.dt) {
            .f16 => @as(f16, @bitCast(r)),
            .bf16 => @bitCast(@as(u32, r) << 16),
            .f32 => unreachable,
        };
    }
    p.input(p.ctx, w, host, m);
}

/// Pre-size the decode scratches to the widest linear, so nothing grows mid-forward.
/// Growth is safe (`ensureDeviceBuffer` syncs the stream first) but the first block
/// would pay a sync per linear for nothing.
/// Whether route `r` takes a bias array even for a linear that has none.
fn readsZeroBias(r: dual_lin.Route) bool {
    return switch (r) {
        .gemm_bf16, .gemm_fp8_bf16, .h16_f32, .h16_f16, .h16_bf16, .i8_wo, .i4_wo, .w4a8_wo, .i8_wo16, .i4_wo16, .w4a8_wo16, .nvfp4, .blockq_bf16 => true,
        else => false,
    };
}

pub fn presize(be: *Backend, p: Plan, lins: []const Weight) !void {
    var bq: usize = 0;
    var w4: usize = 0;
    // The zero bias at its widest before any forward: growing it syncs the stream.
    var zb: usize = 0;
    for (lins) |w| if (route(p, w)) |r| if (readsZeroBias(r)) {
        zb = @max(zb, w.rows);
    };
    if (zb != 0) _ = try be.zeroBias(zb);
    for (lins) |w| switch (route(p, w) orelse continue) {
        .blockq_i8 => bq = @max(bq, Backend.blockQScratchBytes(w.rows, w.cols)),
        .blockq_i4 => bq = @max(bq, Backend.blockQScratchBytesI4(w.rows, w.cols)),
        .w4a8 => w4 = @max(w4, Backend.w4a8ScratchBytes(w.rows, w.cols)),
        // Grown per linear, never shrunk, so the loop leaves the widest.
        .w4a8_wo, .w4a8_wo16 => {
            w4 = @max(w4, Backend.w4a8ScratchBytes(w.rows, w.cols));
            try be.reserveWo(w.rows, w.cols, 0);
        },
        .i8_wo, .i4_wo, .i8_wo16, .i4_wo16 => try be.reserveWo(w.rows, w.cols, 0),
        .i8_wo32, .i4_wo32 => try be.reserveWo32(w.rows, w.cols),
        .w4a8_wo32 => {
            w4 = @max(w4, Backend.w4a8ScratchBytes(w.rows, w.cols));
            try be.reserveWo32(w.rows, w.cols);
        },
        else => {},
    };
    if (bq != 0) try be.ensureDeviceBuffer(&be.bq_i8, bq);
    if (w4 != 0) try be.ensureDeviceBuffer(&be.w4a8_i8, w4);
}

/// `presize` for the scratch that scales with the activation: the weight-only
/// routes' bf16 copy of it. `max_rows` as in `presizeLora`.
pub fn presizeRows(be: *Backend, p: Plan, lins: []const Weight, max_rows: usize) !void {
    for (lins) |w| switch (route(p, w) orelse continue) {
        .i8_wo, .i4_wo, .w4a8_wo => try be.reserveWo(w.rows, w.cols, max_rows),
        else => {},
    };
}

/// Reach the sidecar scratch's final size before the first block, and refuse a
/// factor no kernel here can run.
///
/// Two reasons this is not left to the growth in `sidecar`: growing a device
/// buffer syncs the stream, and `opGemmBf16`'s hand-PTX arm fetches a zero bias
/// per call and GROWS it, whose grown buffer is a NEW host pointer while the old
/// one still sits in the pointer-keyed device weight cache.
///
/// `max_rows` is the most activation rows a render will pass, unpadded.
pub fn presizeLora(be: *Backend, stack: *const lora_mod.Stack, lins: []const Weight, max_rows: usize) !void {
    var widest_out: usize = 0;
    for (lins) |w| {
        for (stack.forWeight(w)) |h| {
            // A trunk that ran the base GEMM alone for one linear is a different
            // model, silently, so one unsupported factor refuses the whole thing
            // rather than falling back per weight.
            if (!lora_cuda.supported(h.target)) {
                std.log.err("lin_cuda: LoRA {s} has a factor shape this backend has no GEMM for", .{h.target.tag});
                return error.UnsupportedCheckpoint;
            }
            widest_out = @max(widest_out, w.rows);
        }
    }
    if (widest_out == 0) return;
    const f = stack.maxFactor();
    const mpad = std.mem.alignForward(usize, max_rows, 128);
    _ = try lora_cuda.borrowed(be, lora_cuda.loElems(max_rows, f.rank), mpad * widest_out);
    // The other terms' planes: a LoKr's per-group product, a diff's or DoRA's base
    // product, DoRA's rescaled input.
    var out_elems: usize = 0;
    var in_elems: usize = 0;
    for (lins) |w| for (stack.forWeight(w)) |h| {
        const t = h.target;
        for (t.kron) |k| out_elems = @max(out_elems, max_rows * k.im * k.w2.rows);
        for (t.diffs) |d| out_elems = @max(out_elems, max_rows * d.w.rows);
        if (t.dora) |d| if (!d.on_out) {
            out_elems = @max(out_elems, mpad * w.rows);
            in_elems = @max(in_elems, max_rows * w.cols);
        };
    };
    if (out_elems != 0) _ = try lora_cuda.outPlane(be, out_elems);
    if (in_elems != 0) _ = try lora_cuda.inPlane(be, in_elems * 4, .f32);
    // The A GEMM's own output width counts too: it is `rank` wide.
    _ = try be.zeroBias(@max(f.rank, widest_out));
}

/// Whether the int8 prep a route reads rotates. `check` refuses a model whose int8
/// linears would need two answers.
fn prepRotates(p: Plan, r: Route) bool {
    return if (r == .blockq_i8 or r == .blockq_i4) blockq_rotate else p.rot;
}

/// Give every LoRA hit whose base linear runs on the int8 prep an int8 A in that
/// prep's basis, so its A GEMM reads the staged activation instead of `x`. Call at
/// attach, per linear list with its own plan; a hit it skips keeps the bf16 A GEMM.
pub fn quantizeLora(gpa: std.mem.Allocator, stack: *lora_mod.Stack, lins: []const Weight, pol: BlockQGemm, int8: IntGemm) !void {
    var p = switch (check(lins, .{ .blockq = pol })) {
        .ok => |p| p,
        .refused => return,
    };
    p.int8 = int8;
    for (lins) |w| {
        const r = route(p, w) orelse continue;
        if (r.prep() != .i8) continue;
        try stack.quantizeHits(gpa, w, prepRotates(p, r));
    }
}

/// The hit's int8 A when this route's staged prep can feed it, else null.
fn i8A(be: *Backend, p: Plan, r: Route, w: Weight, m: usize, h: lora_mod.Stack.Hit) ?Weight {
    if (!lora_i8_a or r.prep() != .i8) return null;
    const a = h.a_i8 orelse return null;
    const want: u32 = if (prepRotates(p, r)) @intCast(ops.convrot.group_size) else 0;
    if (a.convrot != want or be.i8_cols != w.cols or be.i8_m != m) return null;
    return a;
}

/// The sidecar scratch for `w`'s hits, sized by `lora_cuda.loElems`: the int8 A
/// GEMM stores whole tiles, so sizing from `m` and the rank lets its last store
/// run past the buffer.
fn workspace(be: *Backend, hits: []const lora_mod.Stack.Hit, m: usize, w: Weight) !lora_cuda.Workspace {
    var rank: usize = 0;
    for (hits) |h| for (h.target.factors) |f| {
        rank = @max(rank, f.rank());
    };
    return lora_cuda.borrowed(be, lora_cuda.loElems(m, rank), std.mem.alignForward(usize, m, 128) * w.rows);
}

/// Every LoRA hit against `w`, accumulated onto the base GEMM's output.
///
/// A no-op when nothing patches `w`, so it sits beside the one dispatch every
/// family's GEMMs go through and a sidecar cannot be skipped at some call site:
/// that renders a finite, plausible, wrong image with no error anywhere.
///
/// `x` is the activation the base GEMM read, at its own dtype. The int8 arms only READ it
/// during their prep rather than rewriting it, so it is still the unrotated
/// activation the bf16 A GEMM needs; an int8 A reads that prep instead.
///
/// `bias` is what the base added into `y`, which an output-axis DoRA must not rescale.
/// With `dora_base` false the caller has already applied each DoRA's share of the base
/// product (a convolution's, which this cannot compute).
fn sidecar(be: *Backend, p: Plan, r: Route, stack: *const lora_mod.Stack, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32, dora_base: bool) !void {
    const hits = stack.forWeight(w);
    if (hits.len == 0) return;
    var ws = try workspace(be, hits, m, w);
    for (hits) |h| {
        const s = stack.files.items[h.file].strength;
        // Exact: adding `0 * delta` changes nothing, so a dial at zero is free.
        if (s == 0) continue;
        if (dora_base) if (h.target.dora) |d| try doraBase(be, p, y, x, m, w, d, bias, s);
        // A row view of a fused tensor asks for its own rows only.
        try lora_cuda.applyTarget(be, &ws, y, x, m, h.target, s, w.row_off, w.rows, .{ .a_i8 = i8A(be, p, r, w, m, h) });
    }
}

/// A DoRA's share of the base product, onto `y` before its own delta lands there: the
/// output axis rescales what the base wrote, the input axis adds `s * W ((d - 1) x)`,
/// a second base product on a route that stages no prep, so a group's staged int8
/// activation survives it.
fn doraBase(be: *Backend, p: Plan, y: Buf, x: Buf, m: usize, w: Weight, d: lora_mod.Dora, bias: ?[]const f32, s: f32) !void {
    const ratio = try lora_cuda.hostVec(be, d.ratio);
    if (d.on_out) {
        const b: ?Buf = if (bias) |bb| try lora_cuda.hostVec(be, bb[0..w.rows]) else null;
        return be.opDoraRows(y, ratio, b, m, w.rows, s);
    }
    var q = p;
    q.int8 = .weight_only_f16;
    q.blockq = .f16;
    q.lora = null;
    q.gemv = false;
    const r = routeAt(q, w, m) orelse return error.UnsupportedDType;
    if (r.prep() != .none) return error.UnsupportedDType;
    const xd = try lora_cuda.inPlane(be, m * w.cols * 4, .f32);
    try be.opDoraCols(xd, x, ratio, m, w.cols);
    const t = try lora_cuda.outPlane(be, std.mem.alignForward(usize, m, 128) * w.rows);
    try exec(be, r, .{ .rot = q.rot, .bq_rotate = blockq_rotate, .layout_rows = m }, t, xd, m, w, null);
    try lora_cuda.addPlane(be, y, t, m, w.rows, w.rows, 0, s);
}

/// The sidecar for a device GEMM that does not go through `gemm`: `y[m][w.rows]`
/// already holds `x[m][w.cols] @ Wᵀ` (and `bias`) from some other kernel. A DoRA's
/// share of the base product is the caller's to apply first (`doraConv` style).
pub fn sidecarDense(be: *Backend, stack: *const lora_mod.Stack, y: Buf, x: Buf, m: usize, w: Weight) !void {
    try sidecar(be, .{ .rot = false, .blockq = .auto }, .gemm_f32, stack, y, x, m, w, null, false);
}

/// Whether every hit against `lins` has factors this backend runs, logging the
/// first that does not. For a family's `supported`, so such a stack sends the
/// trunk to the CPU rather than failing at session build.
pub fn loraRunnable(stack: ?*const lora_mod.Stack, lins: []const Weight, who: []const u8) bool {
    const s = stack orelse return true;
    for (lins) |w| for (s.forWeight(w)) |h| {
        if (!lora_cuda.supported(h.target)) {
            std.log.warn("{s}: LoRA {s} has a factor shape this backend has no GEMM for; the trunk runs on the CPU", .{ who, h.target.tag });
            return false;
        }
    };
    return true;
}

/// Put `stack` on `plan` and presize for it; after this every `gemm` over `plan`
/// applies it. A no-op with no stack.
pub fn attachLora(be: *Backend, p: *Plan, stack: ?*const lora_mod.Stack, lins: []const Weight, max_rows: usize) !void {
    const s = stack orelse return;
    p.lora = s;
    try presizeLora(be, s, lins, max_rows);
}

// --- tests -----------------------------------------------------------------

// Pure dispatch, so it runs on the fast suite: it decides which GEMM a GGUF DiT uses, and
// getting it wrong is a silent 2x slowdown or a silent accuracy cap rather than a failure.

/// A dtype's route at a shape every decode's floor accepts.
fn routeOfDtype(dt: DType, pol: BlockQGemm) ?Route {
    return routeOf(.{ .bytes = &.{}, .dtype = dt, .rows = 4096, .cols = 4096 }, pol);
}
test "the block-quant route policy sends each format to the GEMM that suits it" {
    // `auto`: q4_k to int8, because int8-convrot's ~0.009 error floor is 0.7% on top of
    // q4_k's own and buys 1.85x the speed. Everything else to bf16, where q8_0 keeps the
    // accuracy the int8 route would spend (84% more weight error, measured).
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(.q4_k, .auto));
    // q2_k too, even though its WEIGHT could take the s4 regrid: that route is W4A4 and
    // the 4-bit activations, not the weight, are what it costs. int4 stays opt-in.
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(.q2_k, .auto));
    for ([_]DType{ .q8_0, .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
        try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt, .auto));

    // An explicit choice is honoured for every format that has a convrot decode.
    for ([_]DType{ .q2_k, .q4_k, .q8_0 }) |dt| {
        try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(dt, .int8));
        try std.testing.expectEqual(@as(?Route, .blockq_i4), routeOfDtype(dt, .int4));
    }
    for ([_]DType{ .q4_k, .q8_0 }) |dt|
        try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt, .bf16));
    // q2_k has the convrot decode AND, now, a dequant kernel, so bf16 and f16 are
    // real routes for it rather than the refusal they once were. What still
    // refuses is a dtype with no decode at all, and the point of the check is that
    // it comes back null rather than falling through to the int8 route.
    try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(.q2_k, .bf16));
    try std.testing.expectEqual(@as(?Route, .blockq_f16), routeOfDtype(.q2_k, .f16));
    for ([_]BlockQGemm{ .bf16, .f16, .int8, .auto }) |g|
        try std.testing.expectEqual(@as(?Route, null), routeOfDtype(.u8, g));

    // Asking for int8 or int4 on a format with no convrot decode must NOT reach
    // `opI8GemmBlockQ`, which would refuse mid-forward; it falls back to bf16.
    for ([_]BlockQGemm{ .int8, .int4 }) |g| {
        for ([_]DType{ .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
            try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt, g));
    }

    // `f16` is the text encoders' route: weight expanded, activation untouched, for
    // every format with a dequant kernel, shape permitting.
    for ([_]DType{ .q4_k, .q5_k, .q6_k, .q8_0, .iq4_xs }) |dt|
        try std.testing.expectEqual(@as(?Route, .blockq_f16), routeOfDtype(dt, .f16));
    try std.testing.expectEqual(Prep.none, Route.blockq_f16.prep());

    // Dense formats route by storage alone, whatever the policy.
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), routeOfDtype(.bf16, .f16));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8), routeOfDtype(.f8_e4m3, .auto));
    try std.testing.expectEqual(@as(?Route, .w4a8), routeOfDtype(.w4a8, .auto));
    try std.testing.expectEqual(@as(?Route, null), routeOfDtype(.u8, .auto));
}

test "a block quant whose shape misses the int8 decode floor takes the bf16 route" {
    // q4_k: 144 bytes per 256 elements. Z-Image's 3840-wide reduction is 15 groups of
    // 256, so the chunked decode (1024-wide chunks) cannot cover it; krea2's 6144 can.
    const wide: [(128 * 6144 / 256 * 144)]u8 = @splat(0);
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOf(Weight.init(&wide, .q4_k, 128, 6144), .auto));
    const narrow: [(128 * 3840 / 256 * 144)]u8 = @splat(0);
    try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOf(Weight.init(&narrow, .q4_k, 128, 3840), .auto));
    // The same shape under `mmq` is fine: the MMQ tile is 256 columns.
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), routeOf(Weight.init(&narrow, .q4_k, 128, 3840), .mmq));
    // The f16 GEMM pads a partial row tile, so an odd row count is no refusal.
    const odd = Weight.init(narrow[0 .. 120 * 3840 / 256 * 144], .q4_k, 120, 3840);
    try std.testing.expectEqual(@as(?Route, .blockq_f16), routeOf(odd, .f16));
    try std.testing.expect(check(&.{odd}, .{ .blockq = .f16 }) == .ok);
}

test "weight-only routes an integer convrot weight to its bf16 twin and nothing else" {
    const bf: [(128 * 256 * 2)]u8 = @splat(0);
    var q = Weight.init(bf[0 .. 128 * 256], .i8, 128, 256);
    q.row_scale = &@as([128]f32, @splat(1));
    q.convrot = 256;
    const dense = Weight.init(&bf, .bf16, 128, 256);
    var p = check(&.{ q, dense }, .{ .blockq = .auto }).ok;
    try std.testing.expectEqual(@as(?Route, .i8), route(p, q));
    p.int8 = .weight_only;
    try std.testing.expectEqual(@as(?Route, .i8_wo), route(p, q));
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), route(p, dense));
    // No int8 prep is staged, which is what keeps a LoRA hit here on the bf16 A GEMM.
    try std.testing.expectEqual(Prep.none, route(p, q).?.prep());
    try std.testing.expect(!allI8(p, &.{q}));
}

// The weight-only decode run on the host through the helpers its dual kernels call:
// row scale, the two-launch rotation back to the checkpoint's basis, the bf16 rounding.
// It must land on the CPU dequant (`materializeF32`) to within bf16, and its product
// with the plain activation on the f32 GEMM; a weight left in the rotated basis, or
// rotated twice, is a GEMM in the wrong basis.
test "weight-only int8 convrot decodes to the CPU dequant" {
    const wm = @import("tp_gpu").cuda.wo_math;
    const gpa = std.testing.allocator;
    const rows = 8;
    const cols = 512;
    var prng = std.Random.DefaultPrng.init(0x5e5e);
    const rnd = prng.random();

    var qb: [rows * cols]i8 = undefined;
    for (&qb) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
    var rs: [rows]f32 = undefined;
    for (&rs) |*v| v.* = 0.002 + rnd.float(f32) * 0.01;
    var w = Weight.init(std.mem.sliceAsBytes(&qb), .i8, rows, cols);
    w.row_scale = &rs;
    w.convrot = @intCast(ops.convrot.group_size);

    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = rnd.floatNorm(f32);
    x[37] = 26; // the outlier channel the route exists for

    const deq = try ops.matmul.materializeF32(gpa, w);
    const wf: []const f32 = @ptrCast(@alignCast(deq.bytes));
    defer gpa.free(wf);

    const bf = struct {
        fn r(v: f32) f32 {
            const bits: u32 = @bitCast(v);
            return @bitCast(((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16) << 16);
        }
    }.r;

    // `i8row_rot_bf16` before its final rounding: 16 lanes a group, 16 values a lane.
    var dev: [rows * cols]f32 = undefined;
    for (0..rows * cols / 4) |t| {
        const word = std.mem.readInt(u32, std.mem.sliceAsBytes(qb[t * 4 ..][0..4])[0..4], .little);
        const quad = wm.i8Quad(word, rs[t * 4 / cols]);
        @memcpy(dev[t * 4 ..][0..4], &quad);
    }
    const rotated = dev;
    for (0..rows * cols / 256) |g| {
        const lanes: *[16][16]f32 = @ptrCast(dev[g * 256 ..][0..256]);
        for (lanes) |*l| wm.radix16(l);
        for ([_]usize{ 1, 4 }) |m| {
            const was = lanes.*;
            for (0..16) |l| for (0..16) |j| {
                lanes[l][j] = wm.bfly4(was[l][j], was[l ^ m][j], was[l ^ (2 * m)][j], was[l ^ (3 * m)][j]);
            };
        }
        for (lanes) |*l| for (l) |*e| {
            e.* /= 16;
        };
    }
    for (wf, dev) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);

    var twice = dev;
    for (0..rows) |r| ops.convrot.rotate(twice[r * cols ..][0..cols]);
    const Case = struct { w: []const f32, ok: bool };
    for ([_]Case{ .{ .w = &dev, .ok = true }, .{ .w = &rotated, .ok = false }, .{ .w = &twice, .ok = false } }) |c| {
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..rows) |r| {
            var ref: f64 = 0;
            var got: f64 = 0;
            for (0..cols) |col| {
                ref += @as(f64, wf[r * cols + col]) * x[col];
                got += @as(f64, bf(c.w[r * cols + col])) * bf(x[col]);
            }
            num += (got - ref) * (got - ref);
            den += ref * ref;
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("weight-only case ok={} rel {d}\n", .{ c.ok, rel });
        try std.testing.expect(if (c.ok) rel < 1e-2 else rel > 0.3);
    }
}

test "the int4 row decode reads nibbles as the CPU does" {
    const wm = @import("tp_gpu").cuda.wo_math;
    for ([_]u8{ 0xF7, 0x80, 0x19, 0x6E }) |byte| {
        for (0..2) |k| {
            const got = wm.sext4(@as(u32, byte) >> @intCast(k * 4));
            try std.testing.expectEqual(@as(i32, DType.nibbleI4(byte, @intCast(k))), got);
        }
    }
}

test "check refuses what the kernels cannot pair and accepts the rest" {
    const bf: [(128 * 256 * 2)]u8 = @splat(0);
    var lins = [_]Weight{ Weight.init(&bf, .bf16, 128, 256), Weight.init(bf[0 .. 128 * 256], .i8, 128, 256) };
    lins[1].row_scale = &@as([128]f32, @splat(1));
    // Plain int8 beside dense: fine, prep unrotated.
    try std.testing.expectEqual(false, check(&lins, .{ .blockq = .auto }).ok.rot);
    lins[1].convrot = 256;
    try std.testing.expectEqual(true, check(&lins, .{ .blockq = .auto }).ok.rot);
    // int4 without the rotation has no prep.
    var w4 = Weight.init(bf[0 .. 128 * 256 / 2], .i4, 128, 256);
    w4.row_scale = &@as([128]f32, @splat(1));
    try std.testing.expectEqual(Why.int4_unrotated, check(&.{w4}, .{ .blockq = .auto }).refused.why);
    w4.convrot = 256;
    try std.testing.expect(check(&.{w4}, .{ .blockq = .auto }) == .ok);
    // A width the int8 GEMM's launch geometry cannot take.
    var odd = Weight.init(bf[0 .. 120 * 256], .i8, 120, 256);
    odd.row_scale = &@as([120]f32, @splat(1));
    odd.tag = "blk.3.wq";
    const r = check(&.{odd}, .{ .blockq = .auto }).refused;
    try std.testing.expectEqual(Why.int_shape, r.why);
    try std.testing.expectEqualStrings("blk.3.wq", r.tag);
    // A format nothing here reads.
    try std.testing.expectEqual(Why.no_kernel, check(&.{Weight.init(bf[0 .. 128 * 256], .u8, 128, 256)}, .{ .blockq = .auto }).refused.why);
    // A rotated and an unrotated int8 linear in one model have no shared prep.
    var flat = Weight.init(bf[0 .. 128 * 256], .i8, 128, 256);
    flat.row_scale = &@as([128]f32, @splat(1));
    try std.testing.expectEqual(Why.mixed_convrot, check(&.{ lins[1], flat }, .{ .blockq = .auto }).refused.why);
}

/// The route an LLM stepper's plan takes for `w` at `m` rows.
fn llmRoute(w: Weight, m: usize) ?Route {
    return routeAt(.llm, w, m);
}

/// A weight with the right dtype and shape and no bytes: routing never reads them,
/// and a real q4_k head at Gemma's vocab would be half a gigabyte of test rodata.
fn fake(dt: DType, rows: usize, cols: usize) Weight {
    return .{ .bytes = &.{}, .dtype = dt, .rows = rows, .cols = cols };
}

test "block quants route by row count: dp4a decode, grouped small batches, MMQ or dequant beyond" {
    const saved = dual_lin.decode_dp4a;
    defer dual_lin.decode_dp4a = saved;
    dual_lin.decode_dp4a = true;
    const q4 = fake(.q4_k, 4096, 4096);
    const q6 = fake(.q6_k, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8), llmRoute(q6, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(fake(.iq4_xs, 4096, 4096), 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(q4, 8));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(q6, grouped_max));
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), llmRoute(q4, grouped_max + 1));
    // q6_k's MMQ kernel exists but loses to dequant+f16 on real shapes.
    try std.testing.expectEqual(@as(?Route, .blockq_f16), llmRoute(q6, 256));
    // The isolation switch sends decode through the f32 kernel.
    dual_lin.decode_dp4a = false;
    try std.testing.expectEqual(@as(?Route, .gemv_q), llmRoute(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q), llmRoute(q6, 1));
}

test "--llm-gemm forces the batched route where the weight can take it" {
    const saved = dual_lin.force;
    defer dual_lin.force = saved;
    const q4 = fake(.q4_k, 4096, 4096);
    const q6 = fake(.q6_k, 4096, 4096);
    dual_lin.force = .grouped;
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(q4, 512));
    try std.testing.expectEqual(@as(?Route, .gemv_q8batch), llmRoute(fake(.q1_0, 4096, 4096), 512));
    dual_lin.force = .mmq;
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), llmRoute(q4, 8));
    // q6_k has a pipe kernel that `mmqPipeFaster` leaves off; forcing reaches it.
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), llmRoute(q6, 256));
    dual_lin.force = .dequant;
    try std.testing.expectEqual(@as(?Route, .blockq_f16), llmRoute(q4, 8));
    // Decode and a skinny weight are untouched by the knob.
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(fake(.q8_0, 320, 4096), 512));
    dual_lin.force = .auto;
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), llmRoute(q4, 512));
}

test "a weight the GEMM tiles cannot take is a GEMV at every row count" {
    // A router: 320 experts over 4096, rows % 128 != 0.
    const router = fake(.q8_0, 320, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(router, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), llmRoute(router, 512));
    // Gemma 3's tied head, 262145 rows: no dp4a tiling either, so the f32 GEMV.
    const head = fake(.q4_k, 262145, 3840);
    try std.testing.expectEqual(@as(?Route, .gemv_q), llmRoute(head, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q), llmRoute(head, 256));
    // A 48-row GDN gate in Bonsai's format takes the batch kernel at any m.
    const gate = fake(.q1_0, 48, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8batch), llmRoute(gate, 512));
    try std.testing.expectEqual(@as(?Route, .gemv_q8), llmRoute(gate, 1));
}

test "dense formats route by row count too" {
    const b16 = fake(.bf16, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_bf16), llmRoute(b16, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), llmRoute(b16, dense_grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), llmRoute(b16, 256));
    const f8 = fake(.f8_e4m3, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_fp8), llmRoute(f8, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_fp8n), llmRoute(f8, 4));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8), llmRoute(f8, 128));
    // A bf16 weight too skinny for the GEMM stays grouped; one the grouped kernel
    // cannot tile either has no batched route, so `check` names it.
    const skinny = fake(.bf16, 48, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), llmRoute(skinny, 512));
    try std.testing.expectEqual(@as(?Route, null), llmRoute(fake(.bf16, 44, 4096), 2));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16), llmRoute(fake(.bf16, 44, 4096), 1));
    // f16 takes its GEMM at any row count, the partial tile padded.
    try std.testing.expectEqual(@as(?Route, .gemm_f16), llmRoute(fake(.f16, 48, 4096), 3));
    try std.testing.expectEqual(@as(?Route, .gemm_f32), llmRoute(fake(.f32, 8, 8), 3));
    try std.testing.expectEqual(@as(?Route, null), llmRoute(fake(.u8, 8, 8), 1));
}

test "check refuses a format with no kernel and names the linear" {
    var ok = fake(.q4_k, 4096, 4096);
    ok.tag = "blk.0.attn_q";
    try std.testing.expect(check(&.{ok}, Use.llm(512)) == .ok);
    // ggml type 42 read as the 64-wide stride: correct bytes, no CUDA kernel.
    try std.testing.expectEqual(@as(?Route, .gemv_q8), llmRoute(fake(.q2_0_g64, 4096, 4096), 1));
    var none = fake(.u8, 8, 8);
    none.tag = "blk.3.ffn_up";
    const r = check(&.{ ok, none }, Use.llm(512)).refused;
    try std.testing.expectEqual(Why.no_kernel, r.why);
    try std.testing.expectEqualStrings("blk.3.ffn_up", r.tag);
}

test "a prep is shared by the group and a stale one is caught" {
    // The pure half of the footgun guard: two groups over the same activation get
    // different serials, and a route that reads no scratch never needs one.
    try std.testing.expectEqual(Prep.none, Route.blockq_f16.prep());
    try std.testing.expectEqual(Prep.q8, Route.gemv_q8n.prep());
    try std.testing.expectEqual(Prep.mmq, Route.blockq_mmq.prep());
}
