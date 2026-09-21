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
    /// The dual row GEMV over the raw weight, the same body the CUDA arm runs.
    /// Every block quant's decode on this backend.
    gemv_dual,
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

/// What routing reads about the device, as a plain value so a route is decided,
/// and tested, without a Context.
///
/// This used to carry five knobs selecting between a transposed-layout GEMV, an
/// int8-repack one, and three cooperative ones, four of them opt-in through
/// `TP_VK_*` and defaulted off. The dual GEMV beat every one of them on every
/// format (`TensorPencil vk-gemv-bench`; BACKEND.md has the table), so the
/// choice, the layouts and the kernels are gone and a block quant has one route.
pub const Knobs = struct {
    /// Block-quant prefill takes the coop GEMM. Off without the f16-weight pipeline.
    gemm_prefill: bool,

    /// The pure dequant routes, for tests.
    pub const plain: Knobs = .{ .gemm_prefill = false };

    pub fn fromContext(ctx: *const gpu.Context) Knobs {
        return .{ .gemm_prefill = ctx.hasQuantPrefillGemm() };
    }

    /// The route this weight takes at `m` rows, or null when no kernel here reads it.
    pub fn routeOf(k: Knobs, w: Weight, m: usize) ?Route {
        std.debug.assert(m >= 1);
        switch (lin.kindOf(w.dtype)) {
            .blockq => {
                // The weight is resident RAW, which is what the decode GEMV reads
                // and what the prefill GEMM dequantizes from, so there is one copy
                // and no layout for the two to disagree about.
                //
                // A lane owns whole quant and codebook WORDS, so a row that is not
                // whole blocks has no mapping and stays on the GEMM. A Vulkan
                // buffer is an opaque handle, so a GEMV cannot step through the
                // rows of `x` either: a batch needs the GEMM or nothing.
                if (m == 1 and w.cols % w.dtype.blockElems() == 0) return .gemv_dual;
                return if (k.gemm_prefill) .gemm_quant else null;
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
                try ctx.opMatmulCoopQuant(w.dtype, y, y_off, x, m, w.bytes, w.rows, w.cols, w.scale, self.zero_bias);
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
            .gemv_dual => try ctx.opGemvQuantDual(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
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
                // The notice is about what this checkpoint will actually RUN, so it
                // asks the route, not the layout: a dequant-only format with a raw
                // GEMV (the dual one, or the cooperative one where it built) has a
                // decode kernel and is not what this warns about.
                for (lins) |w| if (self.routeOf(w, 1) == .gemm_quant) {
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

test "every block quant is raw, and the dual GEMV is its decode" {
    for ([_]DType{
        .q8_0,  .q4_0,    .iq4_nl,   .iq4_xs, .q2_k,  .q3_k,
        .q4_k,  .q5_k,    .q6_k,     .q1_0,   .q2_0_g64,
        .q2_0_g128, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq3_s,
    }) |dt|
        try std.testing.expect(gpu.Context.dequantOnly(dt) and gpu.Context.dualGemv(dt));
    try std.testing.expect(!gpu.Context.dequantOnly(.bf16) and !gpu.Context.dualGemv(.bf16));
}

fn fake(dt: DType, rows: usize, cols: usize) Weight {
    return .{ .bytes = &.{}, .dtype = dt, .rows = rows, .cols = cols };
}

test "a block quant decodes through the dual GEMV and prefills through the coop GEMM" {
    // No device feature and no knob: the dual GEMV is the decode with everything
    // else off, which is what let the five other block-quant routes go.
    for ([_]DType{ .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs, .q4_0, .q1_0, .q2_0_g64, .q2_0_g128, .iq3_s }) |dt|
        try std.testing.expectEqual(@as(?Route, .gemv_dual), Knobs.plain.routeOf(fake(dt, 4096, 4096), 1));

    // A batch is the coop GEMM or nothing: a Vulkan buffer is a handle, so a GEMV
    // cannot step through the rows of the activation.
    const q4 = fake(.q4_k, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, null), Knobs.plain.routeOf(q4, 8));
    var k = Knobs.plain;
    k.gemm_prefill = true;
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(q4, 8));
    try std.testing.expectEqual(@as(?Route, .gemv_dual), k.routeOf(q4, 1));

    // A row that is not whole blocks has no lane mapping, so even a single row
    // goes to the GEMM, and to nothing at all without one.
    try std.testing.expectEqual(@as(?Route, .gemm_quant), k.routeOf(fake(.q1_0, 4096, 4032), 1));
    try std.testing.expectEqual(@as(?Route, null), Knobs.plain.routeOf(fake(.q1_0, 4096, 4032), 1));
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
