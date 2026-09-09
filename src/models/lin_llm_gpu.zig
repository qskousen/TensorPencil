//! The one Vulkan linear dispatcher for LLM steppers.
//!
//! The Vulkan arms decode one token at a time and, where the device has the f16-weight
//! cooperative GEMM, prefill a block-quant model in one batched pass. The route is
//! chosen PER WEIGHT from its storage, its shape and `m`, so a stepper names no dtype,
//! and the four opt-in decode kernels that were A/B knobs on one arm are knobs on every
//! arm. `plan` refuses a checkpoint by name at load instead of `error.UnsupportedDType`
//! from inside a forward.

const std = @import("std");
const lin = @import("lin.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const spec_limits = @import("tp_core").spec_limits;

const Buf = gpu.DeviceBuffer;
pub const Weight = lin.Weight;
pub const DType = lin.DType;

extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Which kernel one linear runs at one row count.
pub const Route = enum {
    /// Per-row dequant GEMV over the transposed 32-row-group layout, k-split.
    gemv_quant_t,
    /// dp4a GEMV over the repacked int8-interleaved layout (q8_0, opt-in iq4_nl).
    gemv_dp4a,
    /// dp4a over the transposed layout, no repack. `TP_VK_T_DP4A`.
    gemv_t_dp4a,
    /// Cooperative subgroup dp4a over the raw weight. `TP_VK_SG_DP4A`.
    gemv_sg_dp4a,
    /// Cooperative subgroup scalar GEMV over the raw weight. `TP_VK_SG_GEMV`.
    gemv_sg,
    /// Weight dequantized to f16 once, then the cooperative-matrix GEMM.
    gemm_quant,
    /// Dense bf16/fp8/f32 k-split GEMV.
    gemv_dense,
    /// Dense grouped GEMV, 4 rows per pass.
    gemv_dense4,
    /// Dense f32-accumulate GEMM (fp8/f32 weights).
    gemm_dense,
};

/// Rows at or below which a dense fp8/f32 weight takes the grouped GEMV.
pub const dense_grouped_max = spec_limits.max_draft + 1;

/// What routing reads about the device and the opt-in knobs, as a plain value so a
/// route is decided, and tested, without a Context.
pub const Knobs = struct {
    /// The device has the integer dot (`OpSDot`) the dp4a kernels need.
    int_dot: bool,
    /// Route iq4_nl through the int8 repack too. Opt-in (`TP_VK_DP4A`): the repack
    /// doubles a 4-bit weight's footprint, where q8_0's grows only ~6%.
    dp4a_iq4: bool,
    sg_gemv: bool,
    sg_dp4a: bool,
    t_dp4a: bool,
    /// The cooperative GEMV covers the formats with no other decode kernel. Not a
    /// knob: for those it is the only alternative to dequantizing the whole weight
    /// per token, so it is on wherever the pipelines built.
    sg_raw: bool,
    /// Block-quant prefill takes the coop GEMM. Off without the f16-weight pipeline,
    /// and off under the raw-reading coop GEMVs: the weight cache keys by host
    /// pointer, so one weight cannot be resident raw for decode and transposed for
    /// prefill. `t_dp4a` reads the same transposed buffer and is exempt.
    gemm_prefill: bool,

    /// Every knob off and no dp4a: the pure dequant routes, for tests.
    pub const plain: Knobs = .{ .int_dot = false, .dp4a_iq4 = false, .sg_gemv = false, .sg_dp4a = false, .t_dp4a = false, .sg_raw = false, .gemm_prefill = false };

    pub fn fromContext(ctx: *const gpu.Context) Knobs {
        const sg_gemv = ctx.hasSubgroupGemv() and getenv("TP_VK_SG_GEMV") != null;
        const sg_dp4a = ctx.hasSubgroupDp4a() and getenv("TP_VK_SG_DP4A") != null;
        return .{
            .int_dot = ctx.hasIntDot(),
            .dp4a_iq4 = ctx.hasIntDot() and getenv("TP_VK_DP4A") != null,
            .sg_gemv = sg_gemv,
            .sg_dp4a = sg_dp4a,
            .t_dp4a = ctx.hasTransposedDp4a() and getenv("TP_VK_T_DP4A") != null,
            .sg_raw = ctx.hasRawSubgroupGemv(),
            .gemm_prefill = ctx.hasQuantPrefillGemm() and !(sg_gemv or sg_dp4a),
        };
    }

    /// Whether this format's resident copy is the int8 repack, which decode and the
    /// prefill GEMM must agree on.
    pub fn dp4aRepack(k: Knobs, dt: DType) bool {
        return switch (dt) {
            .q8_0 => k.int_dot,
            .iq4_nl => k.dp4a_iq4,
            else => false,
        };
    }

    /// The route this weight takes at `m` rows, or null when no kernel here reads it.
    pub fn routeOf(k: Knobs, w: Weight, m: usize) ?Route {
        std.debug.assert(m >= 1);
        switch (lin.kindOf(w.dtype)) {
            .blockq => {
                // These have no TRANSPOSED GEMV; decode is the cooperative one over the
                // raw layout, which is the same buffer the prefill GEMM dequantizes
                // from, so both run off one resident copy.
                if (gpu.Context.dequantOnly(w.dtype)) {
                    if (m == 1 and k.sg_raw and w.cols % w.dtype.blockElems() == 0) return .gemv_sg;
                    return if (k.gemm_prefill) .gemm_quant else null;
                }
                if (!quantKernel(w.dtype)) return null;
                // A Vulkan buffer is an opaque handle, so a GEMV cannot step through the
                // rows of `x`: a batch needs the GEMM or nothing.
                if (m > 1) return if (k.gemm_prefill) .gemm_quant else null;
                const dp4a_fmt = w.dtype == .q8_0 or w.dtype == .iq4_nl;
                if (k.t_dp4a and dp4a_fmt) return .gemv_t_dp4a;
                if (k.sg_dp4a and dp4a_fmt) return .gemv_sg_dp4a;
                if (k.sg_gemv) return .gemv_sg;
                if (dp4a_fmt and k.dp4aRepack(w.dtype)) return .gemv_dp4a;
                return .gemv_quant_t;
            },
            .bf16 => return if (m == 1) .gemv_dense else .gemv_dense4,
            .fp8, .f32 => return if (m == 1) .gemv_dense else if (m <= dense_grouped_max) .gemv_dense4 else .gemm_dense,
            else => return null,
        }
    }

    /// Accept or refuse a model's device linears, naming the first problem. A linear
    /// wider than `max_out` (the GEMV scratch) would write past it.
    pub fn check(k: Knobs, max_out: usize, lins: []const Weight, prefill_rows: usize) Verdict {
        for (lins) |w| {
            const bad = Refusal{ .why = undefined, .tag = w.tag orelse "<untagged>", .dtype = w.dtype, .rows = w.rows };
            if (k.routeOf(w, 1) == null or k.routeOf(w, @max(prefill_rows, 1)) == null) return .{ .refused = with(bad, .no_kernel) };
            if (w.rows > max_out) return .{ .refused = with(bad, .too_wide) };
        }
        return .ok;
    }
};

/// Why `check` refused a checkpoint.
pub const Why = enum { no_kernel, too_wide };

pub const Refusal = struct { why: Why, tag: []const u8, dtype: DType, rows: usize };

pub const Verdict = union(enum) { ok, refused: Refusal };

fn with(r: Refusal, why: Why) Refusal {
    var out = r;
    out.why = why;
    return out;
}

/// Whether the transposed dequant GEMV and the coop dequant GEMM have this format.
pub fn quantKernel(dt: DType) bool {
    return switch (dt) {
        .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl => true,
        else => false,
    };
}

pub fn wcode(dt: DType) gpu.WCode {
    return switch (dt) {
        .f8_e4m3 => .f8,
        .bf16 => .bf16,
        else => .f32,
    };
}

pub const Lin = struct {
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    /// k-split partials for every GEMV route, [nchunk][max_rows] f32 each; one per
    /// member of the widest `linearGroup`, so a group's partial passes need no barrier.
    partials: []Buf,
    /// The f16-weight coop GEMM always folds a bias in and these linears have none;
    /// one full-width buffer for every width, since `smallBuffer` caches by pointer.
    zero_bias: []f32,
    /// Interleaved k chunks per output row in the k-split GEMVs.
    nchunk: usize,
    knobs: Knobs,

    /// `max_out` is the widest output any linear here writes (the head's vocab);
    /// `groups` the widest `linearGroup` (1 for a stepper that never groups).
    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, max_out: usize, nchunk: usize, groups: usize) !Lin {
        const zero_bias = try gpa.alloc(f32, max_out);
        errdefer gpa.free(zero_bias);
        @memset(zero_bias, 0);
        const partials = try gpa.alloc(Buf, @max(groups, 1));
        errdefer gpa.free(partials);
        var made: usize = 0;
        errdefer for (partials[0..made]) |*b| ctx.tensorDestroy(b);
        for (partials) |*b| {
            b.* = try ctx.tensorCreate(max_out * nchunk * 4);
            made += 1;
        }
        return .{
            .ctx = ctx,
            .gpa = gpa,
            .partials = partials,
            .zero_bias = zero_bias,
            .nchunk = nchunk,
            .knobs = Knobs.fromContext(ctx),
        };
    }

    pub fn deinit(self: *Lin) void {
        for (self.partials) |*b| self.ctx.tensorDestroy(b);
        self.gpa.free(self.partials);
        self.gpa.free(self.zero_bias);
    }

    pub fn dp4aRepack(self: *const Lin, dt: DType) bool {
        return self.knobs.dp4aRepack(dt);
    }

    /// The route this weight takes at `m` rows, or null when no kernel here reads it.
    pub fn routeOf(self: *const Lin, w: Weight, m: usize) ?Route {
        return self.knobs.routeOf(w, m);
    }

    /// One linear `y[m][w.rows] f32 = x[m][w.cols] @ Wᵀ`, `y` written at element offset
    /// `y_off` (a stepper packs alpha and beta into one buffer that way; only the
    /// single-row routes honour it).
    pub fn linear(self: *Lin, y: Buf, y_off: usize, x: Buf, m: usize, w: Weight) !void {
        const ctx = self.ctx;
        const r = self.routeOf(w, m) orelse return error.UnsupportedDType;
        std.debug.assert(m == 1 or r == .gemm_quant or r == .gemv_dense4 or r == .gemm_dense);
        switch (r) {
            .gemm_quant => {
                std.debug.assert(w.rows <= self.zero_bias.len);
                try ctx.opMatmulCoopQuant(w.dtype, y, y_off, x, m, w.bytes, w.rows, w.cols, w.scale, self.zero_bias, self.dp4aRepack(w.dtype));
            },
            .gemv_dense4 => {
                std.debug.assert(y_off == 0);
                var g: usize = 0;
                while (g * 4 < m) : (g += 1) {
                    const n: usize = @min(4, m - g * 4);
                    try ctx.opGemvPartial4(x, g * 4 * w.cols, self.partials[0], w.bytes, wcode(w.dtype), w.rows, w.cols, self.nchunk);
                    try ctx.opGemvCombine4(y, g * 4 * w.rows, w.rows, self.partials[0], w.rows, w.scale, self.nchunk, n);
                }
            },
            .gemm_dense => {
                std.debug.assert(y_off == 0);
                try ctx.opMatmul(y, 0, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, null);
            },
            .gemv_quant_t => try ctx.opGemvQuantT(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, self.nchunk, self.partials[0]),
            .gemv_dp4a => try ctx.opGemvDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, self.nchunk, self.partials[0]),
            .gemv_t_dp4a => try ctx.opGemvQuantTDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, self.nchunk, self.partials[0]),
            .gemv_sg_dp4a => try ctx.opGemvQuantSgDp4a(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
            .gemv_sg => try ctx.opGemvQuantSg(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
            .gemv_dense => try ctx.opGemv(y, y_off, x, self.partials[0], w.bytes, wcode(w.dtype), w.rows, w.cols, w.scale, self.nchunk),
        }
    }

    /// Single-row linears over one activation, issued so no barrier drains the GPU
    /// between them: dense weights run all their k-split partial passes, then all their
    /// combines, each into its own scratch. Anything else runs in order, one at a time.
    pub fn linearGroup(self: *Lin, ys: []const Buf, x: Buf, ws: []const Weight) !void {
        std.debug.assert(ys.len == ws.len and ws.len <= self.partials.len);
        const ctx = self.ctx;
        for (ws) |w| {
            if (self.routeOf(w, 1) != .gemv_dense) {
                for (ys, ws) |y, ww| try self.linear(y, 0, x, 1, ww);
                return;
            }
        }
        ctx.independent(ws.len);
        for (ws, 0..) |w, i| try ctx.opGemvPartial(x, self.partials[i], w.bytes, wcode(w.dtype), w.rows, w.cols, self.nchunk);
        ctx.independent(ws.len);
        for (ys, ws, 0..) |y, w, i| try ctx.opGemvCombine(y, 0, self.partials[i], w.rows, w.scale, self.nchunk);
    }

    pub fn check(self: *const Lin, lins: []const Weight, prefill_rows: usize) Verdict {
        return self.knobs.check(self.zero_bias.len, lins, prefill_rows);
    }

    /// `check`, logging the refusal under `who` and returning it as an error.
    pub fn plan(self: *const Lin, lins: []const Weight, prefill_rows: usize, who: []const u8) error{UnsupportedCheckpoint}!void {
        switch (self.check(lins, prefill_rows)) {
            .ok => {
                for (lins) |w| if (gpu.Context.dequantOnly(w.dtype)) {
                    std.log.info("{s}: {t} has no Vulkan decode GEMV; every GEMM dequantizes the weight (slow, but it runs)", .{ who, w.dtype });
                    break;
                };
            },
            .refused => |r| {
                switch (r.why) {
                    .no_kernel => std.log.err("{s}: {s} is {t}, which the Vulkan backend has no kernel for", .{ who, r.tag, r.dtype }),
                    .too_wide => std.log.err("{s}: {s} has {d} rows, wider than the {d} this stepper sized its GEMV scratch for", .{ who, r.tag, r.rows, self.zero_bias.len }),
                }
                return error.UnsupportedCheckpoint;
            },
        }
    }
};

// --- tests -----------------------------------------------------------------

test "quantKernel names the five formats the Vulkan decode GEMV has; the rest of the dequantizers are dequant-only" {
    for ([_]DType{ .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl }) |dt| try std.testing.expect(quantKernel(dt) and !gpu.Context.dequantOnly(dt));
    for ([_]DType{ .q4_0, .iq4_xs, .q1_0, .q2_0_g64, .q2_0_g128 }) |dt| try std.testing.expect(!quantKernel(dt) and gpu.Context.dequantOnly(dt));
    for ([_]DType{ .q2_k, .bf16 }) |dt| try std.testing.expect(!quantKernel(dt) and !gpu.Context.dequantOnly(dt));
}

fn fake(dt: DType, rows: usize, cols: usize) Weight {
    return .{ .bytes = &.{}, .dtype = dt, .rows = rows, .cols = cols };
}

test "block quants route by knob: dequant GEMV by default, dp4a with the integer dot, the coop GEMM only for prefill" {
    const q8 = fake(.q8_0, 4096, 4096);
    const q4 = fake(.q4_k, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_quant_t), Knobs.plain.routeOf(q8, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_quant_t), Knobs.plain.routeOf(q4, 1));
    // A batch is the GEMM or nothing.
    try std.testing.expectEqual(@as(?Route, null), Knobs.plain.routeOf(q4, 8));
    var k = Knobs.plain;
    k.gemm_prefill = true;
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(q4, 8));
    try std.testing.expectEqual(@as(?Route, .gemv_quant_t), k.routeOf(q4, 1));
    // The integer dot takes q8_0 to the repack, iq4_nl only when opted in.
    k.int_dot = true;
    try std.testing.expectEqual(@as(?Route, .gemv_dp4a), k.routeOf(q8, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_quant_t), k.routeOf(fake(.iq4_nl, 4096, 4096), 1));
    k.dp4a_iq4 = true;
    try std.testing.expectEqual(@as(?Route, .gemv_dp4a), k.routeOf(fake(.iq4_nl, 4096, 4096), 1));
    // The cooperative kernels win over the repack; the transposed dp4a over both.
    k.sg_gemv = true;
    try std.testing.expectEqual(@as(?Route, .gemv_sg), k.routeOf(q4, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_sg), k.routeOf(q8, 1));
    k.sg_dp4a = true;
    try std.testing.expectEqual(@as(?Route, .gemv_sg_dp4a), k.routeOf(q8, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_sg), k.routeOf(q4, 1));
    k.t_dp4a = true;
    try std.testing.expectEqual(@as(?Route, .gemv_t_dp4a), k.routeOf(q8, 1));
}

test "a format with no transposed GEMV decodes cooperatively and prefills through the GEMM" {
    const w = fake(.iq4_xs, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, null), Knobs.plain.routeOf(w, 1));
    var k = Knobs.plain;
    k.gemm_prefill = true;
    // Without the cooperative kernels the whole weight dequantizes even to decode.
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(w, 1));
    k.sg_raw = true;
    try std.testing.expectEqual(@as(?Route, .gemv_sg), k.routeOf(w, 1));
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(w, 512));
    for ([_]DType{ .q4_0, .q1_0, .q2_0_g64, .q2_0_g128 }) |dt|
        try std.testing.expectEqual(@as(?Route, .gemv_sg), k.routeOf(fake(dt, 4096, 4096), 1));
    // A row that is not whole blocks has no lane mapping, so it stays on the GEMM.
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(fake(.q1_0, 4096, 4032), 1));
}

test "dense formats route by row count" {
    const b16 = fake(.bf16, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_dense), Knobs.plain.routeOf(b16, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_dense4), Knobs.plain.routeOf(b16, 512));
    const f8 = fake(.f8_e4m3, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_dense4), Knobs.plain.routeOf(f8, dense_grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemm_dense), Knobs.plain.routeOf(f8, dense_grouped_max + 1));
    try std.testing.expectEqual(@as(?Route, null), Knobs.plain.routeOf(fake(.u8, 8, 8), 1));
}

test "check names the first linear with no kernel and one wider than the scratch" {
    var ok = fake(.q4_k, 4096, 4096);
    ok.tag = "blk.0.attn_q";
    var k = Knobs.plain;
    k.gemm_prefill = true;
    try std.testing.expect(k.check(4096, &.{ok}, 512) == .ok);
    // Without the prefill GEMM a batch has no route, so the same weight is refused.
    const r = Knobs.plain.check(4096, &.{ok}, 512).refused;
    try std.testing.expectEqual(Why.no_kernel, r.why);
    try std.testing.expectEqualStrings("blk.0.attn_q", r.tag);
    var wide = fake(.q4_k, 8192, 4096);
    wide.tag = "output";
    const r2 = k.check(4096, &.{ ok, wide }, 1).refused;
    try std.testing.expectEqual(Why.too_wide, r2.why);
    try std.testing.expectEqualStrings("output", r2.tag);
}

test "wcode reads bf16 and fp8 natively and everything else as f32" {
    try std.testing.expectEqual(gpu.WCode.bf16, wcode(.bf16));
    try std.testing.expectEqual(gpu.WCode.f8, wcode(.f8_e4m3));
    try std.testing.expectEqual(gpu.WCode.f32, wcode(.f32));
}
