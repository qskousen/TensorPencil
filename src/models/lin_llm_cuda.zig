//! The one CUDA linear dispatcher for LLM steppers, on both CUDA arms.
//!
//! An LLM runs the same weight at every row count from 1 (decode) through a few
//! (speculative verify, a short follow-up turn) to a whole prefill chunk, and the
//! fastest kernel differs at each: a fused GEMV that reads the weight once, a grouped
//! GEMV that reads it once per 8 rows, the s8 tensor cores straight off the packed
//! weight, or a dequant into an f16 GEMM. The route is chosen PER WEIGHT from its
//! storage, its shape and `m`, so a stepper names no dtype and a kernel wired here
//! reaches every architecture the same day.
//!
//! A stepper calls `prep` once per activation a group of linears shares and gets a
//! `Group` back, then `gemm` per linear with that group. The dp4a and MMQ routes read
//! the q8 activation the prep staged in backend state and ignore the f32 `x`; the rest
//! read `x`. The group carries the prep's serial, so a second prep slipped between a
//! prep and its GEMM is an assert, not a silently wrong activation.
//!
//! `plan` accepts or refuses a model's device linears up front, naming the first
//! problem, so a checkpoint fails at load and never as `unreachable` inside a forward.

const std = @import("std");
const lin = @import("lin.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const spec_limits = @import("tp_core").spec_limits;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
pub const Weight = lin.Weight;
pub const DType = lin.DType;

/// Which kernel one linear runs at one row count.
pub const Route = enum {
    /// Fused block-quant GEMV over the f32 activation, one row per launch.
    gemv_q,
    /// dp4a block-quant GEMV over the q8 activation, one row.
    gemv_q8,
    /// Grouped dp4a GEMV over the q8 activation, up to 8 rows per launch.
    gemv_q8n,
    /// Batched dp4a GEMV, every row in one launch (q1_0, q2_0).
    gemv_q8batch,
    /// Packed weight on the s8 tensor cores against the q8 activation.
    gemm_mmq,
    /// Weight expanded to f16 once, then the f16 tensor-core GEMM.
    gemm_q16,
    gemv_bf16,
    /// Grouped bf16 GEMV, 4 rows per launch, weight read once per group.
    gemv_bf16n,
    gemm_bf16,
    gemv_f16,
    gemm_f16,
    gemv_fp8,
    gemv_fp8n,
    gemm_fp8,
    gemm_f32,

    pub fn prepOf(r: Route) Prep {
        return switch (r) {
            .gemv_q8, .gemv_q8n, .gemv_q8batch => .q8,
            .gemm_mmq => .mmq,
            else => .none,
        };
    }
};

/// What a route reads besides the weight: nothing but `x`, the q8_1 activation laid
/// out for `m` rows, or the same laid out for `m` padded to the MMQ tile.
pub const Prep = enum { none, q8, mmq };

/// Rows at or below which a block quant with a grouped kernel takes it over the GEMMs.
///
/// Measured crossover against the dequant-to-f16 GEMM (qgemv-bench, 3090): ~48 rows
/// q5_k, ~35 q6_k, ~44 for the 27B's shapes; 40 covers every speculative-verify batch
/// (`spec_limits.max_draft + 1`) and most follow-up chat turns.
pub const grouped_max = 40;

/// Rows at or below which a dense bf16 or fp8 weight takes the 4-row grouped GEMV.
/// Every speculative verify batch, where each weight row is read once per 4 inputs
/// instead of the GEMM's dequant-scratch round trip.
pub const dense_grouped_max = spec_limits.max_draft + 1;

/// DIAGNOSTIC: whether block-quant decode takes the dp4a kernels. Off sends every
/// block quant through the f32 `gemv_q` kernel, which is the isolation for a wrong
/// decode: the two differ only by the q8_1 activation quantization.
pub var decode_dp4a: bool = true;

/// Whether the dp4a kernels can tile this weight: 256-column groups, rows in warps of 8.
fn dp4aShape(w: Weight) bool {
    return w.cols % 256 == 0 and w.rows % 8 == 0;
}

/// Whether the f16 GEMMs can take this shape: `launchHgemm` runs `rows / 128` blocks
/// and reads 32-wide k slabs.
fn gemmShape(w: Weight) bool {
    return w.rows % 128 == 0 and w.cols % 32 == 0;
}

/// Whether `opGemvQuantQ8` has a single-row kernel for this dtype.
fn q8Single(dt: DType) bool {
    return dt == .q5_k or dt == .q6_k or Backend.quantQ8BatchSupported(dt);
}

/// The route this weight takes at `m` rows, or null when no kernel here reads it.
pub fn routeOf(w: Weight, m: usize) ?Route {
    std.debug.assert(m >= 1);
    return switch (lin.kindOf(w.dtype)) {
        .blockq => blockQRoute(w, m),
        // The grouped dense kernels run 8 rows per block; a GEMM needs its tile.
        .bf16 => if (m == 1)
            .gemv_bf16
        else if ((m <= dense_grouped_max or !gemmShape(w)) and w.rows % 8 == 0)
            .gemv_bf16n
        else if (gemmShape(w))
            .gemm_bf16
        else
            null,
        .f16 => if (m == 1) .gemv_f16 else if (gemmShape(w)) .gemm_f16 else null,
        .fp8 => if (m == 1)
            .gemv_fp8
        else if ((m <= dense_grouped_max or !gemmShape(w)) and w.rows % 8 == 0)
            .gemv_fp8n
        else if (gemmShape(w))
            .gemm_fp8
        else
            null,
        .f32 => .gemm_f32,
        else => null,
    };
}

fn blockQRoute(w: Weight, m: usize) ?Route {
    const dt = w.dtype;
    // The f32 GEMV and the dequant GEMM switch on dtype with `else => unreachable`.
    if (!Backend.quantKernelSupported(dt)) return null;
    const dp4a = dp4aShape(w);
    if (m == 1) {
        if (decode_dp4a and dp4a) {
            if (q8Single(dt)) return .gemv_q8;
            if (Backend.quantQ8NSupported(dt)) return .gemv_q8n;
        }
        // The k-quant GEMVs stage per-sub-block scales in an 8 KiB shared table.
        return if (w.cols <= 32768) .gemv_q else null;
    }
    // A weight the GEMM tiles cannot take (a router's expert count, a GDN gate's head
    // count, an odd vocab) is a GEMV question at any m.
    const skinny = !gemmShape(w);
    if (dp4a and Backend.quantQ8BatchSupported(dt) and (skinny or m <= grouped_max)) return .gemv_q8batch;
    if (dp4a and Backend.quantQ8NSupported(dt) and (skinny or m <= grouped_max)) return .gemv_q8n;
    if (skinny) return if (w.cols <= 32768) .gemv_q else null;
    if (Backend.mmqPipeFaster(dt, w.rows, w.cols)) return .gemm_mmq;
    return .gemm_q16;
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
};

/// Bumped by every prep that writes the q8 scratch, so a group can prove the scratch
/// still holds its activation.
var prep_serial: u32 = 0;

/// Stage `x[m][cols]` for every linear in `group`, if any of them needs it. Nothing
/// consumes `x`: the q8 form lands in the backend's scratch and the f32 activation is
/// still there for the routes that read it.
pub fn prep(be: *Backend, x: Buf, m: usize, cols: usize, group: []const Weight) !Group {
    var want: Prep = .none;
    for (group) |w| {
        const r = routeOf(w, m) orelse return error.UnsupportedDType;
        std.debug.assert(w.cols == cols);
        switch (r.prepOf()) {
            .none => {},
            .q8 => if (want == .none) {
                want = .q8;
            },
            .mmq => want = .mmq,
        }
    }
    var g = Group{ .x = x, .m = m, .cols = cols, .prep = want, .layout_rows = m, .serial = prep_serial };
    switch (want) {
        .none => {},
        .q8 => {
            try be.opGemvQuantizeX(x, m * cols);
            prep_serial +%= 1;
            g.serial = prep_serial;
        },
        .mmq => {
            try be.opMatmulQuantMmqPipePrep(x, m, cols);
            g.layout_rows = std.mem.alignForward(usize, m, Backend.mmq_pipe_tile);
            prep_serial +%= 1;
            g.serial = prep_serial;
        },
    }
    return g;
}

/// One linear `y[m][w.rows] f32 = x[m][w.cols] @ Wᵀ` against a prepped group. LLM
/// linears carry no bias; `w.scale` multiplies the result where a kernel takes one.
///
/// Buffer requirements the dispatcher cannot check: `y` must hold `align(m, 128)` rows
/// on the GEMM routes (the dequant GEMM and MMQ store whole tiles), and `x` must have
/// `align(m, 4)` rows of backing store on the dense grouped routes (they read 4 rows
/// and predicate the outputs). Every stepper already sizes its buffers that way.
pub fn gemm(be: *Backend, g: Group, y: Buf, w: Weight) !void {
    const r = routeOf(w, g.m) orelse return error.UnsupportedDType;
    std.debug.assert(w.cols == g.cols);
    // A q8 route needs the scratch this group wrote, and nothing since.
    std.debug.assert(r.prepOf() == .none or (g.prep != .none and g.serial == prep_serial));
    const m = g.m;
    const x = g.x;
    switch (r) {
        .gemv_q => {
            for (0..m) |t| {
                try be.opGemvQuant(w.dtype, rowsOf(y, w.rows, t, 1), rowsOf(x, w.cols, t, 1), w.bytes, w.scale, w.rows, w.cols);
            }
        },
        .gemv_q8 => try be.opGemvQuantQ8(w.dtype, y, w.bytes, w.scale, w.rows, w.cols),
        .gemv_q8n => {
            var off: usize = 0;
            while (off < m) : (off += 8) {
                const ng: usize = @min(8, m - off);
                try be.opGemvQuantQ8N(w.dtype, rowsOf(y, w.rows, off, ng), w.bytes, w.scale, w.rows, w.cols, ng, off, g.layout_rows);
            }
        },
        .gemv_q8batch => {
            // The batch kernel's layout is a function of its `n`, so a group that also
            // runs MMQ (tile-padded layout) falls back to the f32 GEMV per row.
            if (g.layout_rows != m) {
                for (0..m) |t| {
                    try be.opGemvQuant(w.dtype, rowsOf(y, w.rows, t, 1), rowsOf(x, w.cols, t, 1), w.bytes, w.scale, w.rows, w.cols);
                }
            } else try be.opGemvQuantQ8Batch(w.dtype, y, w.bytes, w.scale, w.rows, w.cols, m);
        },
        .gemm_mmq => try be.opMatmulQuantMmqPipePrepped(w.dtype, y, m, w.bytes, w.rows, w.cols),
        .gemm_q16 => try be.opMatmulQuant(w.dtype, y, x, m, w.bytes, w.rows, w.cols),
        .gemv_bf16 => try be.opGemvBf16(y, x, w.bytes, w.scale, w.rows, w.cols),
        .gemv_bf16n => {
            var off: usize = 0;
            while (off < m) : (off += 4) {
                const n: usize = @min(4, m - off);
                try be.opGemvBf16N(rowsOf(y, w.rows, off, n), rowsOf(x, w.cols, off, 4), w.bytes, w.scale, w.rows, w.cols, n);
            }
        },
        // Ampere+ feeds raw bf16 to the tensor cores; older cards take the f16 GEMM.
        .gemm_bf16 => if (be.ctx.cc_major >= 8)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, null)
        else
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, null, false, false),
        .gemv_f16 => try be.opGemvF16(y, x, w.bytes, w.scale, w.rows, w.cols),
        .gemm_f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, null, false, false),
        .gemv_fp8 => try be.opGemvFp8(y, x, w.bytes, w.scale, w.rows, w.cols),
        .gemv_fp8n => {
            var off: usize = 0;
            while (off < m) : (off += 4) {
                const n: usize = @min(4, m - off);
                try be.opGemvFp8N(rowsOf(y, w.rows, off, n), rowsOf(x, w.cols, off, 4), w.bytes, w.scale, w.rows, w.cols, n);
            }
        },
        .gemm_fp8 => try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols),
        .gemm_f32 => if (w.scale == 1)
            be.opMatmulF32Lt(y, x, m, w.bytes, w.rows, w.cols, null) catch |err| switch (err) {
                error.UnsupportedKernelArm => try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, null),
                else => return err,
            }
        else
            try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, null),
    }
}

/// `prep` and `gemm` for one linear on its own.
pub fn linear(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight) !void {
    const g = try prep(be, x, m, w.cols, &.{w});
    try gemm(be, g, y, w);
}

/// Rows `[row0, row0 + n)` of an f32 `[rows][width]` buffer.
fn rowsOf(b: Buf, width: usize, row0: usize, n: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + row0 * width * 4), .mem = b.mem, .size = n * width * 4 };
}

/// Why `check` refused a checkpoint.
pub const Why = enum { no_kernel, gemm_shape, gemv_width };

pub const Refusal = struct { why: Why, tag: []const u8, dtype: DType, rows: usize, cols: usize };

pub const Verdict = union(enum) { ok, refused: Refusal };

/// Accept or refuse a model's device linears for every row count a stepper runs,
/// naming the first problem.
pub fn check(lins: []const Weight, prefill_rows: usize) Verdict {
    for (lins) |w| {
        const bad = Refusal{ .why = undefined, .tag = w.tag orelse "<untagged>", .dtype = w.dtype, .rows = w.rows, .cols = w.cols };
        if (lin.kindOf(w.dtype) == .blockq and !Backend.quantKernelSupported(w.dtype)) return .{ .refused = with(bad, .no_kernel) };
        if (routeOf(w, 1) == null) return .{ .refused = with(bad, if (lin.kindOf(w.dtype) == .blockq) .gemv_width else .no_kernel) };
        if (prefill_rows > 1 and routeOf(w, prefill_rows) == null) return .{ .refused = with(bad, .gemm_shape) };
    }
    return .ok;
}

fn with(r: Refusal, why: Why) Refusal {
    var out = r;
    out.why = why;
    return out;
}

/// `check`, logging the refusal under `who` and returning it as an error.
pub fn plan(lins: []const Weight, prefill_rows: usize, who: []const u8) error{UnsupportedCheckpoint}!void {
    switch (check(lins, prefill_rows)) {
        .ok => {},
        .refused => |r| {
            switch (r.why) {
                .no_kernel => std.log.err("{s}: {s} is {t}, which this backend has no kernel for", .{ who, r.tag, r.dtype }),
                .gemv_width => std.log.err("{s}: {s} is [{d}, {d}] {t}; a block-quant decode GEMV needs cols <= 32768, or cols % 256 == 0 and rows % 8 == 0", .{ who, r.tag, r.rows, r.cols, r.dtype }),
                .gemm_shape => std.log.err("{s}: {s} is [{d}, {d}] {t}; the batched GEMM needs rows % 128 == 0 and cols % 32 == 0", .{ who, r.tag, r.rows, r.cols, r.dtype }),
            }
            return error.UnsupportedCheckpoint;
        },
    }
}

// --- tests -----------------------------------------------------------------

// Pure dispatch, so it runs on the fast suite: a wrong route is a silent slowdown or a
// silent numerics change, never a failure.

/// A weight with the right dtype and shape and no bytes: routing never reads them,
/// and a real q4_k head at Gemma's vocab would be half a gigabyte of test rodata.
fn fake(dt: DType, rows: usize, cols: usize) Weight {
    return .{ .bytes = &.{}, .dtype = dt, .rows = rows, .cols = cols };
}

test "block quants route by row count: dp4a decode, grouped small batches, MMQ or dequant beyond" {
    const saved = decode_dp4a;
    defer decode_dp4a = saved;
    decode_dp4a = true;
    const q4 = fake(.q4_k, 4096, 4096);
    const q6 = fake(.q6_k, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), routeOf(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8), routeOf(q6, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), routeOf(q4, 8));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), routeOf(q6, grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemm_mmq), routeOf(q4, grouped_max + 1));
    // q6_k's MMQ kernel exists but loses to dequant+f16 on real shapes.
    try std.testing.expectEqual(@as(?Route, .gemm_q16), routeOf(q6, 256));
    // The isolation switch sends decode through the f32 kernel.
    decode_dp4a = false;
    try std.testing.expectEqual(@as(?Route, .gemv_q), routeOf(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q), routeOf(q6, 1));
}

test "a weight the GEMM tiles cannot take is a GEMV at every row count" {
    // A router: 320 experts over 4096, rows % 128 != 0.
    const router = fake(.q8_0, 320, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), routeOf(router, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q8n), routeOf(router, 512));
    // Gemma 3's tied head, 262145 rows: no dp4a tiling either, so the f32 GEMV.
    const head = fake(.q4_k, 262145, 3840);
    try std.testing.expectEqual(@as(?Route, .gemv_q), routeOf(head, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_q), routeOf(head, 256));
    // A 48-row GDN gate in Bonsai's format takes the batch kernel at any m.
    const gate = fake(.q1_0, 48, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_q8batch), routeOf(gate, 512));
    try std.testing.expectEqual(@as(?Route, .gemv_q8), routeOf(gate, 1));
}

test "dense formats route by row count too" {
    const b16 = fake(.bf16, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_bf16), routeOf(b16, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), routeOf(b16, dense_grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), routeOf(b16, 256));
    const f8 = fake(.f8_e4m3, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_fp8), routeOf(f8, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_fp8n), routeOf(f8, 4));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8), routeOf(f8, 128));
    // A bf16 weight too skinny for the GEMM stays grouped; one the grouped kernel
    // cannot tile either has no batched route, so `check` names it.
    const skinny = fake(.bf16, 48, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), routeOf(skinny, 512));
    try std.testing.expectEqual(@as(?Route, null), routeOf(fake(.bf16, 44, 4096), 2));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16), routeOf(fake(.bf16, 44, 4096), 1));
    try std.testing.expectEqual(@as(?Route, .gemm_f32), routeOf(fake(.f32, 8, 8), 3));
    try std.testing.expectEqual(@as(?Route, null), routeOf(fake(.u8, 8, 8), 1));
}

test "check refuses a format with no kernel and names the linear" {
    var ok = fake(.q4_k, 4096, 4096);
    ok.tag = "blk.0.attn_q";
    try std.testing.expect(check(&.{ok}, 512) == .ok);
    // ggml type 42 read as the 64-wide stride: correct bytes, no CUDA kernel.
    try std.testing.expectEqual(@as(?Route, .gemv_q8), routeOf(fake(.q2_0_g64, 4096, 4096), 1));
    var none = fake(.u8, 8, 8);
    none.tag = "blk.3.ffn_up";
    const r = check(&.{ ok, none }, 512).refused;
    try std.testing.expectEqual(Why.no_kernel, r.why);
    try std.testing.expectEqualStrings("blk.3.ffn_up", r.tag);
}

test "a prep is shared by the group and a stale one is caught" {
    // The pure half of the footgun guard: two groups over the same activation get
    // different serials, and a route that reads no scratch never needs one.
    try std.testing.expectEqual(Prep.none, Route.gemm_q16.prepOf());
    try std.testing.expectEqual(Prep.q8, Route.gemv_q8n.prepOf());
    try std.testing.expectEqual(Prep.mmq, Route.gemm_mmq.prepOf());
}
