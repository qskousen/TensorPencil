//! The one CUDA GEMM dispatcher for diffusion block linears, on both CUDA arms.
//!
//! A family's device forward calls `prep` once per activation a group of linears
//! shares, then `gemm` per linear, and names no dtype. The route is chosen PER WEIGHT
//! from its storage and shape, so a checkpoint that quantizes by layer or mixes GGUF
//! formats by width computes correctly, and a format wired here reaches every family
//! the same day.
//!
//! The int8-convrot arms (int8, W4A8, the block-quant int8 decode) read the activation
//! the prep staged in backend state and ignore `x`; the weight-only arms (bf16, f16,
//! fp8, NVFP4, the block-quant bf16 decode) read `x` directly. MMQ quantizes the
//! activation to q8_1 in its own prep. A block quant with no fast route for its shape
//! degrades to the bf16 dequant route rather than refusing, so "unsupported" here means
//! the format has no dequant kernel at all.
//!
//! `plan` accepts or refuses a model's device linears up front, naming the first
//! problem, so a checkpoint fails at session build and never inside a forward.

const std = @import("std");
const lin = @import("lin.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
pub const Weight = lin.Weight;
pub const DType = lin.DType;

/// Which GEMM one linear runs.
pub const Route = enum { f32, f16, bf16, fp8, i8, i4, w4a8, nvfp4, blockq_i8, blockq_i4, blockq_bf16, blockq_mmq };

/// Which GEMM a GGUF block-quant weight decodes for.
///
/// `int8` rotates and re-quantizes to convrot int8, capping accuracy at int8's; `int4` is
/// that decode one width down for the s4 tensor cores; `bf16` expands the weight and keeps
/// the format's own accuracy at about half int8's throughput (bf16, not f16: Z-Image's
/// activations pass f16's range); `mmq` multiplies the packed weight in place with its
/// scale folded per 32-k substep, re-quantizing nothing.
pub const BlockQGemm = enum { auto, int8, int4, bf16, mmq };

/// Default block-quant route, overridden by `--dit-gguf-gemm`.
///
/// What a re-quantizing route costs a format depends on how the format's own error
/// compares to the regrid's, so `auto` takes int8 for q2_k and q4_k (0.03% and 0.7% more
/// weight error) and bf16 for the rest, where q8_0 alone would lose 84%. int4 is never
/// `auto`: it is W4A4, and its 16-level ACTIVATIONS cost more than the weight regrid
/// saves, 2.4 dB for 1.11x the step even on q2_k, whose weight barely notices the regrid.
pub var blockq_gemm: BlockQGemm = .auto;

/// DIAGNOSTIC: whether the block-quant int8 decode and the prep that pairs with it
/// rotate. Skipping it on both sides is exact arithmetic and worth ~8% of a step, and
/// it measures rel RMSE 0.915 against the CPU forward, i.e. output uncorrelated: the
/// activation's per-row absmax quantization needs the rotation to spread outliers.
pub var blockq_rotate: bool = true;

/// Widest zero bias any diffusion block linear needs (krea2's `mlp_dim`). Passed WHOLE,
/// never sliced: a bias is cached by host pointer and sized from the first call's
/// length, so a narrow layer seen first would leave every wider one reading past the
/// end. `opMatmulNvfp4` wants a stable full-width array for the same reason.
const zero_bias: [16384]f32 = @splat(0);

/// The route a dtype takes, shape aside; null when no kernel reads the format.
pub fn routeOfDtype(dt: DType) ?Route {
    return switch (lin.kindOf(dt)) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .fp8 => .fp8,
        .i8 => .i8,
        .i4 => .i4,
        .w4a8 => .w4a8,
        .nvfp4 => .nvfp4,
        .blockq => blockQRoute(dt),
        .other => null,
    };
}

fn blockQRoute(dt: DType) ?Route {
    if (blockq_gemm == .mmq and Backend.mmqPipeDtype(dt)) return .blockq_mmq;
    if (Backend.blockQFormat(dt) == null) return bf16Route(dt);
    return switch (blockq_gemm) {
        .int8 => .blockq_i8,
        .int4 => .blockq_i4,
        .bf16, .mmq => bf16Route(dt),
        .auto => switch (dt) {
            .q2_k, .q4_k => .blockq_i8,
            else => bf16Route(dt),
        },
    };
}

fn bf16Route(dt: DType) ?Route {
    return if (Backend.quantKernelSupported(dt)) .blockq_bf16 else null;
}

/// The route this weight takes. A block quant whose shape misses its decode's floor
/// (`cols % 1024` for the chunked int8/int4 decode, the MMQ tile for `mmq`) takes the
/// bf16 route instead, which has no floor beyond a dequant kernel.
pub fn routeOf(w: Weight) ?Route {
    const r = routeOfDtype(w.dtype) orelse return null;
    return switch (r) {
        .blockq_mmq => if (Backend.mmqPipeSupported(w.dtype, w.rows, w.cols)) r else bf16Route(w.dtype),
        .blockq_i8, .blockq_i4 => if (w.cols % 1024 == 0 and w.rows % 128 == 0) r else bf16Route(w.dtype),
        else => r,
    };
}

/// Whether this dtype ends up in a W4A4 GEMM here, activation included. A GGUF's route
/// is a runtime choice its dtype does not show, so a check calibrated on how far a route
/// may drift from the weight-only CPU forward has to ask this, not the storage dtype.
pub fn activationIs4Bit(dt: DType) bool {
    return dt == .i4 or routeOfDtype(dt) == .blockq_i4;
}

/// The activation prep a route reads.
pub const Prep = enum { none, i8, i4, mmq };

pub fn prepOf(r: Route) Prep {
    return switch (r) {
        .i8, .w4a8, .blockq_i8 => .i8,
        .i4, .blockq_i4 => .i4,
        .blockq_mmq => .mmq,
        .f32, .f16, .bf16, .fp8, .nvfp4, .blockq_bf16 => .none,
    };
}

/// Whether every weight here runs on the int8 prep, which is what lets a caller chain
/// them through f16 activations (`prep` with `in_f16`, `gemm` with `out_f16`).
pub fn allI8(ws: []const Weight) bool {
    for (ws) |w| if (prepOf(routeOf(w) orelse return false) != .i8) return false;
    return true;
}

/// What `plan` decided for a model: one value per checkpoint, read by every prep.
pub const Plan = struct {
    /// Whether the int8 activation prep rotates. A property of the checkpoint
    /// (`lin.convrot`), not a knob.
    rot: bool,
};

/// Why `check` refused a checkpoint.
pub const Why = enum { mixed_convrot, no_gemm, int_shape, int4_unrotated, blockq_basis, nvfp4_shape, w4a8_group };

pub const Refusal = struct { why: Why, tag: []const u8 = "", dtype: DType = .f32, rows: usize = 0, cols: usize = 0 };

pub const Verdict = union(enum) { ok: Plan, refused: Refusal };

/// Accept or refuse a model's device linears, naming the first problem.
pub fn check(lins: []const Weight) Verdict {
    const rot = lin.convrot(lins) orelse return .{ .refused = .{ .why = .mixed_convrot } };
    for (lins) |w| {
        const bad = Refusal{ .why = undefined, .tag = w.tag orelse "<untagged>", .dtype = w.dtype, .rows = w.rows, .cols = w.cols };
        const r = routeOf(w) orelse return .{ .refused = with(bad, .no_gemm) };
        switch (prepOf(r)) {
            // The int8 GEMMs launch `rows / 128` blocks and the prep rotates in groups of 256.
            .i8, .i4 => if (w.rows % 128 != 0 or w.cols % 256 != 0) return .{ .refused = with(bad, .int_shape) },
            .mmq, .none => {},
        }
        switch (r) {
            // `opI4Prep` always rotates and has no unrotated build.
            .i4 => if (!rot) return .{ .refused = with(bad, .int4_unrotated) },
            // A block-quant decode and a convrot weight in one group would need one prep
            // in two bases.
            .blockq_i8, .blockq_i4 => if (rot != blockq_rotate and (lin.any(lins, .i8) or lin.any(lins, .i4) or lin.any(lins, .w4a8)))
                return .{ .refused = with(bad, .blockq_basis) },
            .nvfp4 => if (w.rows % 128 != 0 or w.cols % 32 != 0) return .{ .refused = with(bad, .nvfp4_shape) },
            else => {},
        }
    }
    if (lin.w4a8SmallGroup(lins)) |tag| return .{ .refused = .{ .why = .w4a8_group, .tag = tag, .dtype = .w4a8 } };
    return .{ .ok = .{ .rot = rot } };
}

fn with(r: Refusal, why: Why) Refusal {
    var out = r;
    out.why = why;
    return out;
}

/// `check`, logging the refusal under `who` and returning it as an error.
pub fn plan(lins: []const Weight, who: []const u8) error{UnsupportedCheckpoint}!Plan {
    switch (check(lins)) {
        .ok => |p| return p,
        .refused => |r| {
            switch (r.why) {
                .mixed_convrot => std.log.err("{s}: this checkpoint mixes convrot and plain int8 block linears; one activation prep serves a whole group, so there is no correct one", .{who}),
                .no_gemm => std.log.err("{s}: {s} is {t}, which this backend has no GEMM for", .{ who, r.tag, r.dtype }),
                .int_shape => std.log.err("{s}: {s} is [{d}, {d}] {t}; the int8/int4 path needs rows % 128 == 0 and cols % 256 == 0", .{ who, r.tag, r.rows, r.cols, r.dtype }),
                .int4_unrotated => std.log.err("{s}: {s} is int4 without convrot, which the s4 prep cannot pair with", .{ who, r.tag }),
                .blockq_basis => std.log.err("{s}: {s} decodes to int8 in a different rotation basis than the checkpoint's own int8 linears", .{ who, r.tag }),
                .nvfp4_shape => std.log.err("{s}: {s} is [{d}, {d}] nvfp4; the f16 GEMM it feeds needs rows % 128 == 0 and cols % 32 == 0", .{ who, r.tag, r.rows, r.cols }),
                .w4a8_group => std.log.err("{s}: {s} is W4A8 with a group_size that is not a multiple of 8; this backend's decode kernel needs one scale per 8 columns", .{ who, r.tag }),
            }
            return error.UnsupportedCheckpoint;
        },
    }
}

/// Pre-size the decode scratches to the widest linear, so nothing grows mid-forward.
/// Growth is safe (`ensureDeviceBuffer` syncs the stream first) but the first block
/// would pay a sync per linear for nothing.
pub fn presize(be: *Backend, lins: []const Weight) !void {
    var bq: usize = 0;
    var w4: usize = 0;
    for (lins) |w| switch (routeOf(w) orelse continue) {
        .blockq_i8 => bq = @max(bq, Backend.blockQScratchBytes(w.rows, w.cols)),
        .blockq_i4 => bq = @max(bq, Backend.blockQScratchBytesI4(w.rows, w.cols)),
        .w4a8 => w4 = @max(w4, Backend.w4a8ScratchBytes(w.rows, w.cols)),
        else => {},
    };
    if (bq != 0) try be.ensureDeviceBuffer(&be.bq_i8, bq);
    if (w4 != 0) try be.ensureDeviceBuffer(&be.w4a8_i8, w4);
}

/// Stage the activation `x[m][cols]` once for every linear in `group`, if any of them
/// needs it. The int8 and MMQ preps write backend state and leave `x` intact, so a
/// weight-only linear in the same group still reads the f32 `x`.
///
/// `in_f16` reads `x` as f16, which only the int8 prep can do (`allI8` says when).
pub fn prep(be: *Backend, p: Plan, x: Buf, m: usize, cols: usize, group: []const Weight, in_f16: bool) !void {
    var want: Prep = .none;
    var bq = false;
    for (group) |w| {
        const r = routeOf(w) orelse return error.UnsupportedDType;
        const k = prepOf(r);
        if (k == .none) continue;
        // Two preps of one activation would need two live prep states, which the
        // backend does not have. `plan` does not see groups, so this is checked here.
        if (want != .none and want != k) {
            std.log.err("lin_cuda: a linear group mixes {t} and {t} activation preps; one cannot serve both", .{ want, k });
            return error.UnsupportedCheckpoint;
        }
        want = k;
        if (r == .blockq_i8 or r == .blockq_i4) bq = true;
    }
    switch (want) {
        .none => {},
        .i8 => try be.opI8PrepR(x, m, cols, in_f16, if (bq) blockq_rotate else p.rot),
        .i4 => {
            std.debug.assert(!in_f16);
            try be.opI4Prep(x, m, cols);
        },
        .mmq => {
            std.debug.assert(!in_f16);
            try be.opMatmulQuantMmqPipePrep(x, m, cols);
        },
    }
}

/// One block linear `y[m][w.rows] f32 = x[m][w.cols] @ Wᵀ`. Block linears carry no bias.
///
/// `out_f16` writes the result as f16 (for an f16 activation chain), which only the int8
/// arms can do.
///
/// ⚠️ `y` must hold `align(m, 128)` rows on every route whose prep is not `.none`: the
/// int8/int4 GEMMs launch over the prep's padded row count and each block stores a whole
/// tile. The weight-only routes write exactly `m`. `plan` cannot check this, since it
/// never sees a buffer, so a family whose sequence padding is coarser than 128 either
/// pads its GEMM outputs or takes only weight-only routes.
pub fn gemm(be: *Backend, p: Plan, y: Buf, x: Buf, m: usize, w: Weight, out_f16: bool) !void {
    _ = p;
    const r = routeOf(w) orelse return error.UnsupportedDType;
    std.debug.assert(!out_f16 or prepOf(r) == .i8);
    try probe(be, r, x, m, w);
    switch (r) {
        .i8 => try be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, out_f16),
        // The packed 4-bit weight is decoded into the backend's scratch on the way in and
        // runs the same int8 GEMM, so the 4-bit form stays resident.
        .w4a8 => {
            const meta = w.w4a8.?;
            try be.opI8GemmW4A8(y, w.bytes, meta.s_rel, std.mem.asBytes(meta.levels), w.row_scale.?, w.rows, w.cols, meta.group_size, out_f16);
        },
        .blockq_i8 => try be.opI8GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, out_f16, blockq_rotate),
        .i4 => try be.opI4Gemm(y, w.bytes, w.row_scale.?, w.rows),
        .blockq_i4 => try be.opI4GemmBlockQ(y, w.dtype, w.bytes, w.rows, w.cols, blockq_rotate),
        // Expands the weight to bf16 in a scratch and runs the bf16 tensor cores.
        .blockq_bf16 => {
            std.debug.assert(w.rows <= zero_bias.len);
            try be.opMatmulQuantBf16(w.dtype, y, x, m, w.bytes, w.rows, w.cols, &zero_bias);
        },
        // The packed s8 goes straight to the int8 tensor cores, scale folded per 32-k
        // substep, against the q8_1 activation `prep` staged.
        .blockq_mmq => try be.opMatmulQuantMmqPipePrepped(w.dtype, y, m, w.bytes, w.rows, w.cols),
        // Weight-only: decoded to an f16 scratch inside the GEMM, the packed form resident.
        .nvfp4 => {
            std.debug.assert(w.rows <= zero_bias.len);
            const meta = w.nvfp4.?;
            try be.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, &zero_bias);
        },
        // Ampere+ feeds raw bf16 straight to the tensor cores. A null bias lets the
        // `.libs` arm write the result straight into `y` instead of staging it and
        // re-reading the whole plane to add zero. Older cards take the f16 GEMM.
        .bf16 => if (be.ctx.cc_major >= 8 and w.rows % 128 == 0 and w.cols % 32 == 0)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, null)
        else {
            std.debug.assert(w.rows <= zero_bias.len);
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, &zero_bias, false, false);
        },
        .f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, null, false, false),
        .fp8 => try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols),
        .f32 => be.opMatmulF32Lt(y, x, m, w.bytes, w.rows, w.cols, null) catch |err| switch (err) {
            error.UnsupportedKernelArm => try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, null),
            else => return err,
        },
    }
}

/// Feed `ops.matmul.probe` a GEMM's input so an activation capture works when a DiT runs
/// on CUDA. This backend never goes through `ops.matmul`, so the CPU probe call site
/// sees nothing on a GPU run.
///
/// Routes that quantize the activation are skipped, not captured: recording their input
/// would file a W8A8- or W4A4-shaped number as a weight-only one. The capture driver
/// refuses those checkpoints outright rather than relying on this returning quietly.
///
/// The activation is downloaded and handed to the same host accumulator a CPU capture
/// uses, so a GPU-captured cache differs from a CPU one only by the GEMM arithmetic. It
/// costs one `cuStreamSynchronize` plus `m x cols x 4 B` over PCIe per linear.
fn probe(be: *Backend, r: Route, x: Buf, m: usize, w: Weight) !void {
    if (ops.matmul.probe == null) return;
    if (prepOf(r) != .none) return;
    try probeInput(be, x, m, w);
}

/// `probe` for a linear outside the block dispatch (a family's patch embed or output
/// projection). `x` must be a view at the activation's own base: `tensorDownload`
/// copies from the buffer's start.
pub fn probeInput(be: *Backend, x: Buf, m: usize, w: anytype) !void {
    const p = ops.matmul.probe orelse return;
    if (m == 0 or w.cols == 0) return;
    const host = be.gpa.alloc(f32, m * w.cols) catch return error.OutOfMemory;
    defer be.gpa.free(host);
    try be.tensorDownload(x, std.mem.sliceAsBytes(host));
    p.input(p.ctx, w, host, m);
}

// --- tests -----------------------------------------------------------------

// Pure dispatch, so it runs on the fast suite: it decides which GEMM a GGUF DiT uses, and
// getting it wrong is a silent 2x slowdown or a silent accuracy cap rather than a failure.
test "the block-quant route policy sends each format to the GEMM that suits it" {
    const saved = blockq_gemm;
    defer blockq_gemm = saved;

    // `auto`: q4_k to int8, because int8-convrot's ~0.009 error floor is 0.7% on top of
    // q4_k's own and buys 1.85x the speed. Everything else to bf16, where q8_0 keeps the
    // accuracy the int8 route would spend (84% more weight error, measured).
    blockq_gemm = .auto;
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(.q4_k));
    // q2_k too, even though its WEIGHT could take the s4 regrid: that route is W4A4 and
    // the 4-bit activations, not the weight, are what it costs. int4 stays opt-in.
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(.q2_k));
    for ([_]DType{ .q8_0, .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
        try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt));

    // An explicit choice is honoured for every format that has a convrot decode.
    for ([_]DType{ .q2_k, .q4_k, .q8_0 }) |dt| {
        blockq_gemm = .int8;
        try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOfDtype(dt));
        blockq_gemm = .int4;
        try std.testing.expectEqual(@as(?Route, .blockq_i4), routeOfDtype(dt));
    }
    blockq_gemm = .bf16;
    for ([_]DType{ .q4_k, .q8_0 }) |dt|
        try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt));
    // q2_k has the convrot decode and no dequant kernel, so the bf16 route is a
    // refusal by name, not a silent fall-through to the int8 one.
    try std.testing.expectEqual(@as(?Route, null), routeOfDtype(.q2_k));

    // Asking for int8 or int4 on a format with no convrot decode must NOT reach
    // `opI8GemmBlockQ`, which would refuse mid-forward; it falls back to bf16.
    for ([_]BlockQGemm{ .int8, .int4 }) |g| {
        blockq_gemm = g;
        for ([_]DType{ .q5_k, .q6_k, .q4_0, .iq4_nl }) |dt|
            try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOfDtype(dt));
    }

    // Dense formats route by storage alone.
    try std.testing.expectEqual(@as(?Route, .bf16), routeOfDtype(.bf16));
    try std.testing.expectEqual(@as(?Route, .fp8), routeOfDtype(.f8_e4m3));
    try std.testing.expectEqual(@as(?Route, .w4a8), routeOfDtype(.w4a8));
    try std.testing.expectEqual(@as(?Route, null), routeOfDtype(.u8));
}

test "a block quant whose shape misses the int8 decode floor takes the bf16 route" {
    const saved = blockq_gemm;
    defer blockq_gemm = saved;
    blockq_gemm = .auto;
    // q4_k: 144 bytes per 256 elements. Z-Image's 3840-wide reduction is 15 groups of
    // 256, so the chunked decode (1024-wide chunks) cannot cover it; krea2's 6144 can.
    const wide = [_]u8{0} ** (128 * 6144 / 256 * 144);
    try std.testing.expectEqual(@as(?Route, .blockq_i8), routeOf(Weight.init(&wide, .q4_k, 128, 6144)));
    const narrow = [_]u8{0} ** (128 * 3840 / 256 * 144);
    try std.testing.expectEqual(@as(?Route, .blockq_bf16), routeOf(Weight.init(&narrow, .q4_k, 128, 3840)));
    // The same shape under `mmq` is fine: the MMQ tile is 256 columns.
    blockq_gemm = .mmq;
    try std.testing.expectEqual(@as(?Route, .blockq_mmq), routeOf(Weight.init(&narrow, .q4_k, 128, 3840)));
}

test "check refuses what the kernels cannot pair and accepts the rest" {
    const bf = [_]u8{0} ** (128 * 256 * 2);
    var lins = [_]Weight{ Weight.init(&bf, .bf16, 128, 256), Weight.init(bf[0 .. 128 * 256], .i8, 128, 256) };
    lins[1].row_scale = &([_]f32{1} ** 128);
    // Plain int8 beside dense: fine, prep unrotated.
    try std.testing.expectEqual(false, check(&lins).ok.rot);
    lins[1].convrot = 256;
    try std.testing.expectEqual(true, check(&lins).ok.rot);
    // int4 without the rotation has no prep.
    var w4 = Weight.init(bf[0 .. 128 * 256 / 2], .i4, 128, 256);
    w4.row_scale = &([_]f32{1} ** 128);
    try std.testing.expectEqual(Why.int4_unrotated, check(&.{w4}).refused.why);
    w4.convrot = 256;
    try std.testing.expect(check(&.{w4}) == .ok);
    // A width the int8 GEMM's launch geometry cannot take.
    var odd = Weight.init(bf[0 .. 120 * 256], .i8, 120, 256);
    odd.row_scale = &([_]f32{1} ** 120);
    odd.tag = "blk.3.wq";
    const r = check(&.{odd}).refused;
    try std.testing.expectEqual(Why.int_shape, r.why);
    try std.testing.expectEqualStrings("blk.3.wq", r.tag);
    // A format nothing here reads.
    try std.testing.expectEqual(Why.no_gemm, check(&.{Weight.init(bf[0 .. 128 * 256], .u8, 128, 256)}).refused.why);
    // A rotated and an unrotated int8 linear in one model have no shared prep.
    var flat = Weight.init(bf[0 .. 128 * 256], .i8, 128, 256);
    flat.row_scale = &([_]f32{1} ** 128);
    try std.testing.expectEqual(Why.mixed_convrot, check(&.{ lins[1], flat }).refused.why);
}
