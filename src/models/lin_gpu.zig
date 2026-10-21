//! The one Vulkan linear dispatcher, for every caller: diffusion trunks, text encoders
//! and LLM steppers.
//!
//! The route is chosen PER WEIGHT by `dual_lin.routeOf` over this backend's `Caps`,
//! read from the device once, so a format any caller here runs, every caller runs, and
//! a gap against CUDA is a difference between two `Caps`. This file only executes a
//! route.
//!
//! A forward calls `prep` once per activation a group of linears shares and gets a
//! `Group` back, then `gemm` per linear. The int8 prep's state lives in the Context;
//! the group carries its serial, so a second prep slipped between a prep and its GEMM
//! is an assert. int4 has no tensor cores here and is decoded to int8 per GEMM, so it
//! runs W4A8 arithmetic on the int8 prep (`i4_i8`).
//!
//! A buffer is a handle, not a pointer. A view (`Context.view`) slices one where the
//! offset meets the storage alignment; `Out.y_off` writes at any element, and the routes
//! whose kernel takes no offset assert it is zero.
//!
//! `x` and `y` may each be f32, f16 or bf16 on every route. The cooperative 16-bit GEMMs
//! read and write them at their own dtype; any other route gets an f32 copy of `x` from
//! `prep` and writes `y` through an f32 staging plane (`readsTyped`, `writesTyped`).

const std = @import("std");
const lin = @import("lin.zig");
pub const dual_lin = @import("dual_lin.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const lora_mod = @import("lora.zig");
const lora_gpu = @import("lora_gpu.zig");

const Buf = gpu.DeviceBuffer;
pub const Weight = lin.Weight;
pub const DType = lin.DType;
pub const Route = dual_lin.Route;
pub const Prep = dual_lin.Prep;
pub const Plan = dual_lin.Plan;
pub const Use = dual_lin.Use;
pub const Why = dual_lin.Why;
pub const Verdict = dual_lin.Verdict;
pub const dense_grouped_max = dual_lin.dense_grouped_max;

/// What this device has, as `dual_lin.routeOf` asks for it.
pub const Caps = struct {
    /// The f16-weight cooperative GEMM, which every 16-bit dense weight, every f32 one
    /// under `Dense.f16` and every block-quant prefill feeds.
    f16w: bool,
    /// The native bf16 cooperative GEMM, preferred for bf16 weights where it built.
    bf16w: bool,
    /// The int8 cooperative GEMM and the decodes that feed it.
    i8: bool,
    i4dec: bool,
    w4a8dec: bool,
    nvfp4dec: bool,
    /// The fp8 cooperative GEMM (`opMatmulCoop`), f16 activation, either build.
    coop_fp8: bool,
    /// What that GEMM's weight rows must be a multiple of (`Context.coop_wgn`).
    fp8_rows: u32 = 256,
    /// The weight-only integer expansion (`opMatmulCoopIntWo`).
    wo_pack: bool = false,
    /// The f32 GEMM the W·A32 weight-only routes end in.
    f32_gemm: bool = false,
    /// `OpSDot`, for the q8_1 decode GEMVs.
    int_dot: bool = false,

    /// No device feature at all: what routing falls back to, for tests.
    pub const none: Caps = .{ .f16w = false, .bf16w = false, .i8 = false, .i4dec = false, .w4a8dec = false, .nvfp4dec = false, .coop_fp8 = false };

    pub fn fromContext(ctx: *const gpu.Context) Caps {
        const has_i8 = ctx.hasI8Gemm();
        return .{
            .f16w = ctx.pipe_coop_f16w != .null_handle,
            .bf16w = ctx.pipe_coop_bf16w != .null_handle,
            .i8 = has_i8,
            .i4dec = has_i8 and ctx.hasI4Decode(),
            .w4a8dec = has_i8 and ctx.hasW4A8Decode(),
            .nvfp4dec = ctx.hasNvfp4Decode(),
            .coop_fp8 = ctx.pipe_coop != .null_handle or ctx.fp8_wide,
            .fp8_rows = ctx.coop_wgn,
            .wo_pack = ctx.pipe_wo_pack[0] != .null_handle,
            .f32_gemm = ctx.pipe_f32 != .null_handle,
            .int_dot = ctx.has_int_dot,
        };
    }

    pub fn quantKernel(c: Caps, dt: DType) bool {
        return c.f16w and gpu.Context.dequantOnly(dt);
    }
    pub fn quantGemv(_: Caps, dt: DType) bool {
        return gpu.Context.dualGemv(dt);
    }
    pub fn dequantBf16(c: Caps, dt: DType) bool {
        return c.bf16w and gpu.Context.dequantOnly(dt);
    }
    /// The f16 expansion is the block-quant GEMM here, bf16 only where the plan needs
    /// its range.
    pub fn dequantF16First(_: Caps) bool {
        return true;
    }
    pub fn blockQDecode(_: Caps, _: DType) bool {
        return false;
    }
    pub fn mmqDtype(_: Caps, _: DType) bool {
        return false;
    }
    pub fn mmqShape(_: Caps, _: DType, _: usize, _: usize) bool {
        return false;
    }
    pub fn mmqFaster(_: Caps, _: DType, _: usize, _: usize) bool {
        return false;
    }
    /// The dual row GEMV is every block quant's decode here.
    pub fn dualOnly(_: Caps, dt: DType) bool {
        return gpu.Context.dualGemv(dt);
    }
    pub fn dual(_: Caps, dt: DType) bool {
        return gpu.Context.dualGemv(dt);
    }
    /// The q8_1 decode GEMV, where the device multiplies with `OpSDot`: without it the
    /// dot is four scalar products and the staging pass costs more than it saves
    /// (measured 9% slower than the f32 GEMV on a 4B decode).
    pub fn dualQ8(c: Caps, _: DType) bool {
        return c.int_dot and !gpu.Context.sdot_off;
    }
    /// A small batch reads each weight once per 4 rows, where the dequant GEMM expands
    /// the whole weight first.
    pub fn dualN(_: Caps, dt: DType) bool {
        return gpu.Context.dualGemv(dt);
    }
    pub fn q8Batch(_: Caps, _: DType) bool {
        return false;
    }
    pub fn q8N(_: Caps, _: DType) bool {
        return false;
    }
    pub fn i8Gemm(c: Caps) bool {
        return c.i8;
    }
    pub fn i4Native(_: Caps) bool {
        return false;
    }
    pub fn i4Decode(c: Caps) bool {
        return c.i4dec;
    }
    pub fn w4a8(c: Caps) bool {
        return c.w4a8dec;
    }
    /// The 16-bit weight-only routes pack for the coop GEMM of their width; f32 runs
    /// the plain f32 GEMM.
    pub fn intWeightOnly(c: Caps, g: dual_lin.IntGemm) bool {
        return switch (g) {
            .w8a8 => unreachable,
            .weight_only => c.bf16w and c.wo_pack,
            .weight_only_f16 => c.f16w and c.wo_pack,
            .weight_only_f32 => c.f32_gemm,
        };
    }
    pub fn nvfp4(c: Caps) bool {
        return c.nvfp4dec;
    }
    /// `Dense.f16`: every dense weight on the f16-weight coop GEMM.
    pub fn h16(c: Caps) bool {
        return c.f16w;
    }
    pub fn denseGemm(c: Caps, _: DType) bool {
        return c.f16w or c.bf16w;
    }
    /// A 16-bit weight widened to f32 for the f32 GEMM, where there is no 16-bit one.
    pub fn wideGemm(c: Caps) bool {
        return c.f32_gemm;
    }
    pub fn denseGemmTakes(c: Caps, _: Weight) bool {
        return c.f16w or c.bf16w;
    }
    /// `opMatmulCoop` tiles `fp8_rows` rows and 64-wide k slabs.
    pub fn fp8Gemm(c: Caps, w: Weight) bool {
        return c.coop_fp8 and w.rows % c.fp8_rows == 0 and w.cols % 64 == 0;
    }
    pub fn fp8Bf16(_: Caps) bool {
        return false;
    }
    pub fn fp8F32(_: Caps) bool {
        return true;
    }
    pub fn groupedTakes(_: Caps, w: Weight) bool {
        return w.rows % 8 == 0;
    }
    pub fn gemvF16(_: Caps) bool {
        return false;
    }
    pub fn gemvF32(_: Caps) bool {
        return true;
    }
    /// The k-split GEMV fills a quarter-row of threads a chunk, which a 100-row router
    /// leaves idle (44 us at one row), and the f32 GEMM tiles waste it (1 ms at 19).
    pub fn f32Rows(_: Caps, w: Weight, m: usize) bool {
        return w.rows <= 256 and w.cols % 4 == 0 and m <= 512;
    }
    /// What the kernel behind route `r` cannot take about `w`, or null.
    pub fn refuse(_: Caps, _: []const Weight, rot: bool, w: Weight, r: Route) ?Why {
        if (r.prep() == .i8) {
            // `opI8GemmBuf` stores whole row tiles; the prep rotates in groups of 256 and
            // packs four int8 a word unrotated.
            if (w.rows % gpu.i8_row_tile != 0) return .int_shape;
            if (w.cols % @as(usize, if (rot) 256 else 4) != 0) return .int_shape;
        }
        return switch (r) {
            // `i4Decode` emits a weight quantized after the rotation and has no
            // unrotated build.
            .i4_i8 => if (!rot) .int4_unrotated else null,
            .nvfp4 => if (!dual_lin.gemmShape(w)) .nvfp4_shape else null,
            // The pack reads four rows a word and rotates whole 256 groups; the int4
            // decode has no unrotated build.
            .i8_wo, .i8_wo16, .i8_wo32 => if (w.cols % @as(usize, if (rot) 256 else 4) != 0) .int_shape else null,
            .i4_wo, .i4_wo16, .i4_wo32 => if (!rot) .int4_unrotated else if (w.cols % 256 != 0) .int_shape else null,
            .w4a8_wo, .w4a8_wo16, .w4a8_wo32 => if (w.cols % 256 != 0) .int_shape else null,
            else => null,
        };
    }
};

pub fn wcode(dt: DType) gpu.WCode {
    return switch (dt) {
        .f8_e4m3 => .f8,
        .bf16 => .bf16,
        else => .f32,
    };
}

/// One activation staged for a group of linears.
pub const Group = struct {
    x: Buf,
    m: usize,
    cols: usize,
    prep: Prep,
    serial: u32,
    /// `x` is the caller's f16 staging of the activation, multiplied by this fp8 scale
    /// (`prepStaged`).
    staged: ?f32 = null,
    /// `x` widened to f32, for the routes in the group that cannot read it as it is.
    x32: ?Buf = null,
};

/// Bumped by every int8 prep, so a group can prove the prep state still holds its
/// activation.
var serial_i8: u32 = 0;
var serial_q8: u32 = 0;

/// Where and how `gemm` writes.
pub const Out = struct {
    /// Element offset into `y`; only the routes whose kernel takes one honour it.
    y_off: usize = 0,
    /// Element offset into the group's `x`; only the f32-activation GEMMs take one
    /// (`linearAt` stages the rest).
    x_off: usize = 0,
    /// Added per output column; must outlive the model (cached by host pointer).
    bias: ?[]const f32 = null,
    /// Rows `y` holds on the routes that store whole 128-row tiles (the fp8 and int8
    /// cooperative GEMMs); `align(m, 128)` when null.
    m_pad: ?usize = null,
    /// A staged group may store an f16 `y` straight from f16 accumulators
    /// (`pipe_coop_c16`), which loses precision over a long reduction; otherwise it
    /// accumulates f32 and narrows.
    acc16: bool = false,
};

/// Accept or refuse a model's device linears on this device, naming the first problem.
pub fn check(ctx: *const gpu.Context, lins: []const Weight, use: Use) Verdict {
    return dual_lin.check(Caps, Caps.fromContext(ctx), lins, use);
}

/// `check`, logging the refusal under `who` and returning it as an error.
/// What the route for `w` at `m` rows does to its activation, for a device check's CPU
/// reference (`ops.matmul.act_quant`). The integer convrot routes only: a block-quant
/// int8 decode also re-quantizes the weight, which this does not model.
pub fn actQuant(ctx: *gpu.Context, p: Plan, w: Weight, m: usize) ops.matmul.ActQuant {
    const r = dual_lin.routeOf(Caps, Caps.fromContext(ctx), w, m, p.policy()) orelse return .{};
    return switch (r) {
        .i8, .i4_i8, .w4a8 => .{ .bits = 8, .rotate = p.rot },
        else => .{},
    };
}

pub fn plan(ctx: *const gpu.Context, lins: []const Weight, use: Use, who: []const u8) error{UnsupportedCheckpoint}!Plan {
    return dual_lin.plan(Caps, Caps.fromContext(ctx), lins, use, who);
}

/// Whether a device forward here can run `lins` and the LoRA stack on them, logging
/// why not under `who`. The base GEMM alone is a different model with no error
/// anywhere, so a factor no kernel here reads sends the trunk to the CPU instead.
pub fn runs(ctx: *const gpu.Context, lins: []const Weight, use: Use, stack: ?*const lora_mod.Stack, who: []const u8) bool {
    if (stack) |st| {
        for (lins) |w| for (st.forWeight(w)) |h| if (!lora_gpu.supported(h.target)) {
            std.log.warn("{s}: LoRA {s} has a factor this backend has no GEMM for; the trunk runs on the CPU instead", .{ who, h.target.tag });
            return false;
        };
    }
    switch (check(ctx, lins, use)) {
        .ok => return true,
        .refused => |r| {
            dual_lin.explain(r, who);
            std.log.warn("{s}: the trunk runs on the CPU instead. Expect CPU sampling speed.", .{who});
            return false;
        },
    }
}

/// The widest output a one-shot tower's `Lin` is sized for: every ViT and embedder
/// here stays well under it, and `gemm` asserts a wider one.
pub const tower_max_out: usize = 32768;

/// One linear of a one-shot tower under `Plan.tower` (dense weights on the f16 coop
/// GEMM), its prep the weight's own, so the rotation is too.
pub fn towerLinear(l: *const Lin, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
    var p = Plan.tower;
    p.rot = w.convrot != 0;
    if (l.routeAt(p, w, m) == null) {
        std.log.err("lin_gpu: {s} is {t}, which no tower route here takes", .{ w.tag orelse "<untagged>", w.dtype });
        return error.UnsupportedDType;
    }
    const g = try l.prep(p, x, m, w.cols, &.{w});
    try l.gemm(p, g, y, w, .{ .bias = bias });
}

/// The widest output any of `lins` writes, which is what `Lin.init` sizes its zero
/// bias and GEMV scratch to.
/// Grow every decode scratch the routes of `lins` under `p` use to its widest before a
/// batch opens: growth is safe mid-batch but flushes the recording, once per growth.
pub fn presize(ctx: *gpu.Context, p: Plan, lins: []const Weight) !void {
    const caps = Caps.fromContext(ctx);
    var nv: usize = 0;
    var w4: usize = 0;
    var bq_rows: usize = 0;
    var bq16: usize = 0;
    var i4s: usize = 0;
    for (lins) |w| switch (dual_lin.routeOf(Caps, caps, w, 128, p.policy()) orelse continue) {
        .nvfp4 => nv = @max(nv, ctx.kmajor16Bytes(w.rows, w.cols)),
        .w4a8 => w4 = @max(w4, gpu.Context.w4a8ScratchBytes(w.rows, w.cols)),
        .blockq_f16, .blockq_bf16 => {
            bq_rows = @max(bq_rows, w.rows * w.cols * 2);
            bq16 = @max(bq16, ctx.kmajor16Bytes(w.rows, w.cols));
        },
        .i8_wo, .i8_wo16 => bq16 = @max(bq16, ctx.kmajor16Bytes(w.rows, w.cols)),
        .i4_wo, .i4_wo16 => {
            i4s = @max(i4s, gpu.Context.i4ScratchBytes(w.rows, w.cols));
            bq16 = @max(bq16, ctx.kmajor16Bytes(w.rows, w.cols));
        },
        .w4a8_wo, .w4a8_wo16 => {
            w4 = @max(w4, gpu.Context.w4a8ScratchBytes(w.rows, w.cols));
            bq16 = @max(bq16, ctx.kmajor16Bytes(w.rows, w.cols));
        },
        .i8_wo32 => try ctx.reserveWo32(w.rows, w.cols),
        .i4_wo32 => {
            i4s = @max(i4s, gpu.Context.i4ScratchBytes(w.rows, w.cols));
            try ctx.reserveWo32(w.rows, w.cols);
        },
        .w4a8_wo32 => {
            w4 = @max(w4, gpu.Context.w4a8ScratchBytes(w.rows, w.cols));
            try ctx.reserveWo32(w.rows, w.cols);
        },
        else => {},
    };
    if (nv != 0) try ctx.ensureDeviceBuffer(&ctx.nvfp4_w16, nv);
    if (w4 != 0) try ctx.ensureDeviceBuffer(&ctx.w4a8_t, w4);
    if (bq_rows != 0) try ctx.ensureDeviceBuffer(&ctx.deq_rows, bq_rows);
    if (bq16 != 0) try ctx.ensureDeviceBuffer(&ctx.deq_f16k, bq16);
    if (i4s != 0) try ctx.ensureDeviceBuffer(&ctx.i4_t, i4s);
}

/// An integer convrot weight as the weight-only GEMMs take it.
fn intWeight(w: Weight) gpu.Context.IntWeight {
    return switch (w.dtype) {
        .i8 => .{ .i8 = w.bytes },
        .i4 => .{ .i4 = w.bytes },
        else => blk: {
            const meta = w.w4a8.?;
            break :blk .{ .w4a8 = .{ .packed_bytes = w.bytes, .s_rel = meta.s_rel, .levels = meta.levelBytes(), .group_size = meta.group_size } };
        },
    };
}

pub fn maxRows(lins: []const Weight) usize {
    var n: usize = 0;
    for (lins) |w| n = @max(n, w.rows);
    return n;
}

pub const Lin = struct {
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    caps: Caps,
    /// k-split partials for every GEMV route, [nchunk][max_out] f32 each; one per
    /// member of the widest `linearGroup`, so a group's partial passes need no barrier.
    /// The first is 4x that: the grouped GEMVs write 4 rows of partials into it.
    partials: []Buf,
    /// For the kernels that always fold a bias, and the widest output any linear here
    /// writes. Passed WHOLE: `smallBuffer` caches by host pointer and sizes from the
    /// first call.
    zero_bias: []f32,
    /// Interleaved k chunks per output row in the k-split GEMVs.
    nchunk: usize,
    /// The LoRA sidecar's scratch, behind a pointer so a const `gemm` can grow it.
    lora: *lora_gpu.Scratch,
    /// Staged input and output: `linearAt`'s offset copies, and `prep`'s widened `x`
    /// and `gemm`'s f32 `y` for a route that cannot take 16-bit. Behind a pointer for
    /// the same reason.
    stage: *[2]Buf,

    /// `max_out` is the widest output any linear here writes; `groups` the widest
    /// `linearGroup`, and 0 for a caller that runs no GEMV.
    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, max_out: usize, nchunk: usize, groups: usize) !Lin {
        const zero_bias = try gpa.alloc(f32, max_out);
        errdefer gpa.free(zero_bias);
        @memset(zero_bias, 0);
        const partials = try gpa.alloc(Buf, groups);
        errdefer gpa.free(partials);
        var made: usize = 0;
        errdefer for (partials[0..made]) |*b| ctx.tensorDestroy(b);
        for (partials, 0..) |*b, i| {
            b.* = try ctx.tensorCreate(max_out * nchunk * 4 * @as(usize, if (i == 0) 4 else 1));
            made += 1;
        }
        const ls = try gpa.create(lora_gpu.Scratch);
        errdefer gpa.destroy(ls);
        ls.* = .{};
        const st = try gpa.create([2]Buf);
        st.* = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        return .{ .ctx = ctx, .gpa = gpa, .caps = Caps.fromContext(ctx), .partials = partials, .zero_bias = zero_bias, .nchunk = nchunk, .lora = ls, .stage = st };
    }

    pub fn deinit(self: *Lin) void {
        self.lora.deinit(self.ctx);
        self.gpa.destroy(self.lora);
        for (self.stage) |*b| self.ctx.tensorDestroy(b);
        self.gpa.destroy(self.stage);
        for (self.partials) |*b| self.ctx.tensorDestroy(b);
        self.gpa.free(self.partials);
        self.gpa.free(self.zero_bias);
    }

    /// The route `w` takes at `m` rows under `p`, or null when no kernel here reads it.
    pub fn routeAt(self: *const Lin, p: Plan, w: Weight, m: usize) ?Route {
        return dual_lin.routeOf(Caps, self.caps, w, m, p.policy());
    }

    /// Whether `w` at `m` rows writes exactly `m` rows of its output, so `y` may be a view
    /// into something larger: the fp8 and int8 coop GEMMs store whole 128-row tiles.
    pub fn writesRows(self: *const Lin, p: Plan, w: Weight, m: usize) bool {
        return switch (self.routeAt(p, w, m) orelse return false) {
            .gemm_fp8, .i8, .i4_i8, .w4a8 => false,
            else => true,
        };
    }

    /// Accept or refuse a model's device linears, naming the first problem. A linear
    /// wider than `max_out` would write past the GEMV scratch and the zero bias.
    pub fn check(self: *const Lin, lins: []const Weight, use: Use) Verdict {
        const v = dual_lin.check(Caps, self.caps, lins, use);
        if (v == .refused) return v;
        for (lins) |w| if (w.rows > self.zero_bias.len)
            return .{ .refused = .{ .why = .too_wide, .tag = w.tag orelse "<untagged>", .dtype = w.dtype, .rows = w.rows, .cols = w.cols } };
        return v;
    }

    /// `check`, logging the refusal under `who` and returning it as an error.
    pub fn plan(self: *const Lin, lins: []const Weight, use: Use, who: []const u8) error{UnsupportedCheckpoint}!Plan {
        switch (self.check(lins, use)) {
            .ok => |p| return p,
            .refused => |r| {
                if (r.why == .too_wide)
                    std.log.err("{s}: {s} has {d} rows, wider than the {d} this caller sized its scratch for", .{ who, r.tag, r.rows, self.zero_bias.len })
                else
                    dual_lin.explain(r, who);
                return error.UnsupportedCheckpoint;
            },
        }
    }

    /// Stage `x[m][cols]` for every linear in `group`, if any of them needs it. The int8
    /// prep writes Context state and leaves `x` intact, so a dense linear in the same
    /// group still reads the f32 activation.
    pub fn prep(self: *const Lin, p: Plan, x: Buf, m: usize, cols: usize, group: []const Weight) !Group {
        var want: Prep = .none;
        var widen = false;
        for (group) |w| {
            const r = self.routeAt(p, w, m) orelse return error.UnsupportedDType;
            std.debug.assert(w.cols == cols);
            if (x.dt != .f32 and !self.readsTyped(r, w, x.dt)) widen = true;
            switch (r.prep()) {
                .none => {},
                .i8 => want = .i8,
                .q8 => want = .q8,
                // `Caps` never routes to the other preps.
                .i4, .mmq => return error.UnsupportedDType,
            }
        }
        var g = Group{ .x = x, .m = m, .cols = cols, .prep = want, .serial = 0 };
        if (widen) {
            const n = m * cols;
            try self.ctx.ensureDeviceBuffer(&self.stage[0], n * 4);
            const x32 = self.stage[0];
            try self.ctx.opElt(gpu.eltCopy(x.dt, .f32), x, x32, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
            g.x32 = x32;
        }
        const xf = g.x32 orelse x;
        switch (want) {
            .i8 => {
                try self.ctx.opI8PrepR(xf, m, cols, p.rot);
                serial_i8 +%= 1;
                g.serial = serial_i8;
            },
            .q8 => {
                try self.ctx.opGemvQuantizeXRows(xf, m, cols);
                serial_q8 +%= 1;
                g.serial = serial_q8;
            },
            else => {},
        }
        return g;
    }

    /// `prep`'s q8 staging of `x[m][cols]`, for a caller that multiplies against it itself
    /// (`opGemvExperts`); counted like `prep`'s, so a held group sees it was replaced.
    pub fn stageQ8(self: *const Lin, x: Buf, m: usize, cols: usize) !Group {
        try self.ctx.opGemvQuantizeXRows(x, m, cols);
        serial_q8 +%= 1;
        return .{ .x = x, .m = m, .cols = cols, .prep = .q8, .serial = serial_q8 };
    }

    /// Whether `g` is the q8 staging of `x[m]` still in place.
    pub fn holdsQ8(_: *const Lin, g: ?Group, x: Buf, m: usize) bool {
        const h = g orelse return false;
        return h.prep == .q8 and h.serial == serial_q8 and h.x32 == null and h.x.buf == x.buf and h.x.off == x.off and h.m == m;
    }

    /// The common fp8 scale when every weight in `ws` runs the cooperative fp8 GEMM with
    /// no LoRA hit, which is what one staged activation can serve; null otherwise.
    pub fn stageScale(self: *const Lin, p: Plan, ws: []const Weight, m: usize) ?f32 {
        if (self.ctx.pipe_coop == .null_handle) return null;
        for (ws) |w| {
            if (self.routeAt(p, w, m) != .gemm_fp8 or w.scale != ws[0].scale) return null;
            if (p.lora) |st| if (st.forWeight(w).len > 0) return null;
        }
        return ws[0].scale;
    }

    /// `prep` for an activation the caller staged itself: `x16` f16 `[align(m, 128)][cols]`,
    /// multiplied by `scale`, zero past row `m` (a fused norm or gate kernel writes it, which
    /// saves the per-GEMM conversion `.gemm_fp8` otherwise runs). `stageScale` must have
    /// returned `scale` for `group`; each `gemm` is then the one cooperative dispatch, and
    /// adds no LoRA sidecar.
    pub fn prepStaged(self: *const Lin, p: Plan, x16: Buf, m: usize, cols: usize, group: []const Weight, scale: f32) Group {
        std.debug.assert(x16.dt == .f16 and self.stageScale(p, group, m) == scale);
        for (group) |w| std.debug.assert(w.cols == cols);
        return .{ .x = x16, .m = m, .cols = cols, .prep = .none, .serial = 0, .staged = scale };
    }

    /// One linear `y[m][w.rows] = x[m][w.cols] @ Wᵀ (+ bias)` against a prepped group, `y`
    /// at its own dtype (16-bit only from the int8 GEMMs and a staged group).
    pub fn gemm(self: *const Lin, p: Plan, g: Group, y: Buf, w: Weight, o: Out) !void {
        const r = self.routeAt(p, w, g.m) orelse return error.UnsupportedDType;
        std.debug.assert(w.cols == g.cols);
        if (g.staged) |sc| {
            std.debug.assert(r == .gemm_fp8 and w.scale == sc and o.y_off == 0);
            const m_pad = o.m_pad orelse std.mem.alignForward(usize, g.m, 128);
            // The coop GEMM stores f32, or f16 from f16 accumulators; anything else lands
            // in f32 and narrows.
            const direct = y.dt == .f32 or (y.dt == .f16 and o.acc16 and self.ctx.pipe_coop_c16 != .null_handle);
            const ys = if (direct) y else try self.stageY(m_pad, w.rows);
            std.debug.assert(ys.size >= m_pad * w.rows * ys.dt.bytes());
            try self.ctx.opMatmulCoopH16(ys, g.x, m_pad, w.bytes, w.rows, w.cols, ys.dt == .f16);
            if (o.bias) |b| try self.addBias(ys, g.m, w.rows, b);
            if (!direct) try self.ctx.opElt(gpu.eltCopy(.f32, y.dt), ys, y, null, null, .{ .u0 = @intCast(g.m * w.rows) }, g.m * w.rows, 1, 1);
            return;
        }
        if (r.prep() == .i8) std.debug.assert(g.prep == .i8 and g.serial == serial_i8);
        if (r.prep() == .q8) std.debug.assert(g.prep == .q8 and g.serial == serial_q8);
        if (o.bias) |b| std.debug.assert(b.len >= w.rows);
        // A real check, not an assert: a release build would read the bias past the end.
        if (o.bias == null and w.rows > self.zero_bias.len) return error.ShapeMismatch;
        const x = if (self.readsTyped(r, w, g.x.dt)) g.x else g.x32 orelse g.x;
        std.debug.assert(x.dt == .f32 or self.readsTyped(r, w, x.dt));
        // `prep` widens from element 0.
        if (g.x32 != null and x.buf == g.x32.?.buf) std.debug.assert(o.x_off == 0);
        if (y.dt == .f32 or self.writesTyped(r, w, y.dt)) {
            try self.exec(r, y, x, g.m, w, o);
            if (p.lora) |stack| try self.sidecar(stack, y, x, g.m, w, o);
            return;
        }
        // An f32 plane, then one narrowing copy to `y` at `o.y_off`.
        const ys = try self.stageY(o.m_pad orelse std.mem.alignForward(usize, g.m, 128), w.rows);
        var so = o;
        so.y_off = 0;
        try self.exec(r, ys, x, g.m, w, so);
        if (p.lora) |stack| try self.sidecar(stack, ys, x, g.m, w, so);
        const n = g.m * w.rows;
        try self.ctx.opElt(gpu.eltCopy(.f32, y.dt), ys, y, null, null, .{ .u0 = @intCast(n), .u2 = @intCast(o.y_off) }, n, 1, 1);
    }

    fn stageY(self: *const Lin, rows: usize, cols: usize) !Buf {
        try self.ctx.ensureDeviceBuffer(&self.stage[1], rows * cols * 4);
        return self.stage[1];
    }

    /// Whether route `r` reads an `x` of dtype `dt` as it is: the cooperative GEMMs whose
    /// operand cast is `pad_h16` / `pad_bf16`, a bf16 one only where the bf16 GEMM built
    /// (`coopOperandBf16`).
    fn readsTyped(self: *const Lin, r: Route, w: Weight, dt: gpu.Dt) bool {
        if (dt == .bf16 and self.ctx.pipe_coop_bf16w == .null_handle) return false;
        return switch (r) {
            .h16_f32 => self.h16Coop(w, dt, dt),
            .h16_f16, .h16_bf16, .gemm_bf16, .gemm_f16, .nvfp4, .blockq_f16, .blockq_bf16 => true,
            .i8_wo, .i4_wo, .w4a8_wo, .i8_wo16, .i4_wo16, .w4a8_wo16 => true,
            else => false,
        };
    }

    /// Whether route `r` writes a `y` of dtype `dt` as it is: those, and the int8 GEMMs.
    fn writesTyped(self: *const Lin, r: Route, w: Weight, dt: gpu.Dt) bool {
        return self.readsTyped(r, w, dt) or r.prep() == .i8;
    }

    /// Whether an `.h16_f32` weight takes the cooperative GEMM rather than the f32 one:
    /// below `h16_min_co` rows only when an activation is 16-bit, which the f32 GEMM
    /// would have to stage.
    fn h16Coop(self: *const Lin, w: Weight, x_dt: gpu.Dt, y_dt: gpu.Dt) bool {
        _ = self;
        const wide = w.rows >= dual_lin.h16_min_co or x_dt != .f32 or y_dt != .f32;
        return wide and std.mem.isAligned(@intFromPtr(w.bytes.ptr), @alignOf(f32));
    }

    /// Every LoRA hit against `w`, added onto the base GEMM's output. A no-op when
    /// nothing patches `w`, so it sits beside the one dispatch every caller goes
    /// through and cannot be skipped at some call site.
    fn sidecar(self: *const Lin, stack: *const lora_mod.Stack, y: Buf, x: Buf, m: usize, w: Weight, o: Out) !void {
        const hits = stack.forWeight(w);
        if (hits.len == 0) return;
        if (o.y_off != 0) {
            std.log.err("lin_gpu: {s} has a LoRA sidecar and an offset output, which the sidecar cannot add onto", .{w.tag orelse "<untagged>"});
            return error.UnsupportedCheckpoint;
        }
        for (hits) |h| {
            const s = stack.files.items[h.file].strength;
            // Exact: adding `0 * delta` changes nothing, so a dial at zero is free.
            if (s == 0) continue;
            if (h.target.dora) |d| try self.doraBase(y, x, m, w, d, o.bias, s);
            try lora_gpu.applyTarget(self.ctx, self.lora, self.zero_bias, y, x, m, h.target, s, w.row_off, w.rows);
        }
    }

    /// A DoRA's share of the base product, onto `y` before its own delta lands there:
    /// the output axis rescales what the base wrote (`bias` included in it), the input
    /// axis adds `s * W ((d - 1) x)`, a second base product on a route that stages no
    /// prep, so a group's staged int8 activation survives it.
    fn doraBase(self: *const Lin, y: Buf, x: Buf, m: usize, w: Weight, d: lora_mod.Dora, bias: ?[]const f32, s: f32) !void {
        const ctx = self.ctx;
        const ratio = try lora_gpu.hostVec(ctx, d.ratio);
        if (d.on_out) {
            const b: ?Buf = if (bias) |bb| try lora_gpu.hostVec(ctx, bb) else null;
            return lora_gpu.doraRows(ctx, y, ratio, b, m, w.rows, s);
        }
        // The weight-only routes rotate an integer weight back by `rot`, which is the
        // checkpoint's, so it comes off the weight rather than a default.
        var q = Plan.tower;
        q.dense = .native;
        q.rot = w.convrot != 0;
        const r = self.routeAt(q, w, m) orelse return error.UnsupportedDType;
        if (r.prep() != .none) return error.UnsupportedDType;
        const xd = try self.lora.inPlane(ctx, m * w.cols * 4, .f32);
        try lora_gpu.doraCols(ctx, xd, x, ratio, m, w.cols);
        const t = try self.lora.outPlane(ctx, std.mem.alignForward(usize, m, 128) * w.rows);
        try self.exec(r, t, xd, m, w, .{});
        try lora_gpu.addPlane(ctx, y, t, m, w.rows, w.rows, 0, s);
    }

    /// Reach the sidecar scratch's final size for `max_rows` rows before the first
    /// block, where growing it would flush a recording batch.
    pub fn presizeLora(self: *const Lin, stack: *const lora_mod.Stack, lins: []const Weight, max_rows: usize) !void {
        var lo: usize = 0;
        var out: usize = 0;
        var in: usize = 0;
        for (lins) |w| for (stack.forWeight(w)) |h| {
            const t = h.target;
            for (t.factors) |f| lo = @max(lo, max_rows * f.a.rows);
            for (t.kron) |k| out = @max(out, max_rows * k.im * k.w2.rows);
            for (t.diffs) |d| out = @max(out, max_rows * d.w.rows);
            if (t.dora) |d| if (!d.on_out) {
                out = @max(out, std.mem.alignForward(usize, max_rows, 128) * w.rows);
                in = @max(in, max_rows * w.cols);
            };
        };
        if (lo != 0) try self.lora.ensure(self.ctx, lo);
        if (out != 0) _ = try self.lora.outPlane(self.ctx, out);
        if (in != 0) _ = try self.lora.inPlane(self.ctx, in * 4, .f32);
    }

    /// `prep` and `gemm` for one linear on its own.
    pub fn linear(self: *const Lin, p: Plan, y: Buf, y_off: usize, x: Buf, m: usize, w: Weight) !void {
        const g = try self.prep(p, x, m, w.cols, &.{w});
        try self.gemm(p, g, y, w, .{ .y_off = y_off });
    }

    /// `linear` reading `x` from element `x_off` and writing `y` from element `y_off`,
    /// with a bias: what a patch embedder or an output head does to one part of a
    /// joint sequence. The f32-activation GEMMs take both offsets directly; any other
    /// route, and a weight with a LoRA hit (the sidecar reads and adds whole planes),
    /// runs on copies staged here, since an element offset need not meet a view's
    /// alignment.
    pub fn linearAt(self: *const Lin, p: Plan, y: Buf, y_off: usize, x: Buf, x_off: usize, m: usize, w: Weight, bias: ?[]const f32) !void {
        const r = self.routeAt(p, w, m) orelse return error.UnsupportedDType;
        const hits = if (p.lora) |st| st.forWeight(w).len else 0;
        if ((r == .gemm_f32 or r == .gemm_fp8_f32) and hits == 0 and x.dt == .f32 and y.dt == .f32) {
            const g = try self.prep(p, x, m, w.cols, &.{w});
            return self.gemm(p, g, y, w, .{ .y_off = y_off, .x_off = x_off, .bias = bias });
        }
        const ctx = self.ctx;
        var xs = x;
        if (x_off != 0) {
            try ctx.ensureDeviceBuffer(&self.stage[0], m * w.cols * 4);
            try ctx.opElt(gpu.eltCopy(x.dt, .f32), x, self.stage[0], null, null, .{ .u0 = @intCast(m * w.cols), .u2 = 0, .u3 = @intCast(x_off) }, m * w.cols, 1, 1);
            xs = self.stage[0];
        }
        var ys = y;
        if (y_off != 0) {
            try ctx.ensureDeviceBuffer(&self.stage[1], std.mem.alignForward(usize, m, 128) * w.rows * 4);
            ys = self.stage[1];
        }
        const g = try self.prep(p, xs, m, w.cols, &.{w});
        try self.gemm(p, g, ys, w, .{ .bias = bias });
        if (y_off != 0) try ctx.opElt(gpu.eltCopy(.f32, y.dt), ys, y, null, null, .{ .u0 = @intCast(m * w.rows), .u2 = @intCast(y_off), .u3 = 0 }, m * w.rows, 1, 1);
    }

    /// Single-row linears over one activation, issued so no barrier drains the GPU
    /// between them: dense weights run all their k-split partial passes, then all their
    /// combines, each into its own scratch. Anything else runs in order, one at a time.
    pub fn linearGroup(self: *const Lin, p: Plan, ys: []const Buf, x: Buf, ws: []const Weight) !void {
        std.debug.assert(ys.len == ws.len and ws.len <= self.partials.len);
        const ctx = self.ctx;
        var dense = true;
        var block = true;
        for (ws) |w| switch (self.routeAt(p, w, 1) orelse return error.UnsupportedDType) {
            .gemv_bf16, .gemv_fp8, .gemv_f32 => block = false,
            .gemv_dual, .gemv_dual_q8 => dense = false,
            else => {
                dense = false;
                block = false;
            },
        };
        // A LoRA hit appends dependent ops, which `independent` would count.
        if (block and p.lora == null) {
            // One staging of x for the group, then the GEMVs side by side.
            const g = try self.prep(p, x, 1, ws[0].cols, ws);
            ctx.independent(ws.len);
            for (ys, ws) |y, w| try self.gemm(p, g, y, w, .{});
            return;
        }
        if (!dense or p.lora != null) {
            for (ys, ws) |y, ww| try self.linear(p, y, 0, x, 1, ww);
            return;
        }
        for (ws) |w| std.debug.assert(w.bytes.len <= max_binding);
        ctx.independent(ws.len);
        for (ws, 0..) |w, i| try ctx.opGemvPartial(x, self.partials[i], w.bytes, wcode(w.dtype), w.rows, w.cols, self.nchunk);
        ctx.independent(ws.len);
        for (ys, ws, 0..) |y, w, i| try ctx.opGemvCombine(y, 0, self.partials[i], w.rows, w.scale, self.nchunk);
    }

    /// Route `r` on this device. A route whose kernel folds no bias adds it in a second
    /// pass, which needs `y_off == 0`.
    fn exec(self: *const Lin, r: Route, y: Buf, x: Buf, m: usize, w: Weight, o: Out) !void {
        const ctx = self.ctx;
        const zb: []const f32 = o.bias orelse self.zero_bias;
        if (o.x_off != 0) std.debug.assert(r == .gemm_f32 or r == .gemm_fp8_f32);
        if (r.prep() == .none and o.x_off == 0) try probeInput(ctx, x, m, w);
        // The int8 and fp8 coop GEMMs store whole 128-row tiles; a `y` sized for `m`
        // rows would be written past its end.
        switch (r) {
            .gemm_fp8, .i8, .i4_i8, .w4a8 => std.debug.assert(y.size >= (o.m_pad orelse std.mem.alignForward(usize, m, 128)) * w.rows * y.dt.bytes()),
            else => {},
        }
        var folded = true;
        switch (r) {
            // Below `h16_min_co` rows the f32 GEMM: padding to the coop tile costs
            // more than it returns. The coop entry wants a real `[]f32`, so an f32
            // tensor at an unaligned offset takes it too.
            .h16_f32 => if (self.h16Coop(w, x.dt, y.dt)) {
                const wf: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w.bytes.ptr)))[0 .. w.bytes.len / 4];
                try ctx.opMatmulCoopF16W(y, o.y_off, x, m, wf, w.rows, w.cols, zb);
            } else try ctx.opMatmul(y, o.y_off * 4, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, o.bias),
            .h16_f16 => try ctx.opMatmulCoopF16Wh(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb),
            .h16_bf16 => if (self.caps.bf16w)
                try ctx.opMatmulCoopBf16(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb)
            else
                try ctx.opMatmulCoopF16Wb(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb),
            .gemm_f32 => if (w.dtype == .f32)
                try ctx.opMatmul(y, o.y_off * 4, x, o.x_off * 4, m, w.bytes, false, w.rows, w.cols, w.scale, o.bias)
            else
                try ctx.opMatmulWide(y, o.y_off * 4, x, o.x_off * 4, m, w.bytes, w.dtype, w.rows, w.cols, w.scale, o.bias),
            .gemm_fp8_f32 => try ctx.opMatmul(y, o.y_off * 4, x, o.x_off * 4, m, w.bytes, true, w.rows, w.cols, w.scale, o.bias),
            .gemm_bf16 => if (self.caps.bf16w)
                try ctx.opMatmulCoopBf16(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb)
            else
                try ctx.opMatmulCoopF16Wb(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb),
            .gemm_f16 => try ctx.opMatmulCoopF16Wh(y, o.y_off, x, m, w.bytes, w.rows, w.cols, zb),
            .gemm_fp8 => if (ctx.fp8_wide)
                try ctx.opMatmulFp8Wide(y, o.y_off, x, m, w.bytes, w.rows, w.cols, w.scale, zb)
            else {
                std.debug.assert(o.y_off == 0);
                try ctx.opMatmulCoop(y, x, m, o.m_pad orelse std.mem.alignForward(usize, m, 128), w.bytes, w.rows, w.cols, w.scale);
                folded = o.bias == null;
            },
            .i8 => {
                std.debug.assert(o.y_off == 0);
                try ctx.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows);
                folded = o.bias == null;
            },
            // The packed 4-bit forms decode into the Context's scratch on the way in and
            // run the same int8 GEMM, so they stay resident packed.
            .i4_i8 => {
                std.debug.assert(o.y_off == 0);
                const wbuf = try ctx.i4Decode(w.bytes, w.rows, w.cols);
                try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows);
                folded = o.bias == null;
            },
            .w4a8 => {
                std.debug.assert(o.y_off == 0);
                const meta = w.w4a8.?;
                const wbuf = try ctx.w4a8Decode(w.bytes, meta.s_rel, meta.levelBytes(), w.rows, w.cols, meta.group_size);
                try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows);
                folded = o.bias == null;
            },
            .nvfp4 => {
                std.debug.assert(o.y_off == 0);
                const meta = w.nvfp4.?;
                try ctx.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zb);
            },
            .blockq_f16 => try ctx.opMatmulCoopQuant(w.dtype, y, o.y_off, x, m, w.bytes, w.rows, w.cols, w.scale, zb, false),
            .blockq_bf16 => try ctx.opMatmulCoopQuant(w.dtype, y, o.y_off, x, m, w.bytes, w.rows, w.cols, w.scale, zb, true),
            inline .i8_wo, .i4_wo, .w4a8_wo, .i8_wo16, .i4_wo16, .w4a8_wo16 => |rt| {
                const iw = intWeight(w);
                const wide = rt == .i8_wo or rt == .i4_wo or rt == .w4a8_wo;
                try ctx.opMatmulCoopIntWo(y, o.y_off, x, m, iw, w.row_scale.?, w.rows, w.cols, w.convrot != 0, zb, wide);
            },
            .i8_wo32, .i4_wo32, .w4a8_wo32 => {
                const iw = intWeight(w);
                try ctx.opMatmulIntWoF32(y, o.y_off, x, m, iw, w.row_scale.?, w.rows, w.cols, w.convrot != 0, o.bias);
            },
            .gemv_dual, .gemv_dualn, .gemv_bf16, .gemv_fp8, .gemv_f32, .gemv_bf16n, .gemv_fp8n, .gemv_f32n => {
                try self.gemv(r, y, o.y_off, x, m, w);
                folded = o.bias == null;
            },
            .gemv_dual_q8, .gemv_dualn_q8 => {
                try self.gemv(r, y, o.y_off, x, m, w);
                folded = o.bias == null;
            },
            .gemv_f32_rows => {
                try ctx.opGemvRowsF32(y, o.y_off, x, w.bytes, w.rows, w.cols, m, w.scale);
                folded = o.bias == null;
            },
            // Routes for kernels this backend lacks; `Caps` never chooses them.
            .gemm_fp8_bf16, .gemv_f16, .i4, .blockq_i8, .blockq_i4, .blockq_mmq, .gemv_q, .gemv_q8, .gemv_q8n, .gemv_q8batch => return error.UnsupportedDType,
        }
        if (folded) return;
        std.debug.assert(o.y_off == 0);
        try self.addBias(y, m, w.rows, o.bias.?);
    }

    /// `y[m][rows] += b`, broadcast over rows, at `y`'s dtype.
    fn addBias(self: *const Lin, y: Buf, m: usize, rows: usize, b: []const f32) !void {
        const bd: Buf = .{ .buf = try self.ctx.smallBuffer(std.mem.sliceAsBytes(b)), .mem = .null_handle, .size = 0 };
        const n = m * rows;
        try self.ctx.opElt(gpu.eltAt("add_bias_rows", y.dt), y, bd, null, null, .{ .u0 = @intCast(n), .u1 = @intCast(rows) }, n, 1, 1);
    }

    /// The GEMV routes. Their kernels type a binding as 2^28 words, so a weight past
    /// `max_binding` bytes runs as row slices, each written in place in the full output.
    fn gemv(self: *const Lin, r: Route, y: Buf, y_off: usize, x: Buf, m: usize, w: Weight) !void {
        const ctx = self.ctx;
        if (m > 1) std.debug.assert(y_off == 0);
        const row_bytes = w.bytes.len / w.rows;
        const per = if (w.bytes.len <= max_binding) w.rows else (max_binding / row_bytes) / 8 * 8;
        var r0: usize = 0;
        while (r0 < w.rows) : (r0 += per) {
            const n = @min(per, w.rows - r0);
            const wb = w.bytes[r0 * row_bytes ..][0 .. n * row_bytes];
            switch (r) {
                .gemv_dual => try ctx.opGemvQuantDual(w.dtype, y, y_off + r0, x, wb, w.scale, n, w.cols),
                // The q8 routes read the activation `prep` staged, not `x`.
                .gemv_dual_q8 => try ctx.opGemvQuantDualQ8(w.dtype, y, y_off + r0, wb, w.scale, n, w.cols),
                .gemv_dualn_q8 => {
                    var t: usize = 0;
                    while (t < m) : (t += 4) try ctx.opGemvQuantDualQ8N(w.dtype, y, t * w.rows + r0, w.rows, wb, w.scale, n, w.cols, t, @min(4, m - t));
                },
                .gemv_bf16, .gemv_fp8, .gemv_f32 => try ctx.opGemv(y, y_off + r0, x, self.partials[0], wb, wcode(w.dtype), n, w.cols, w.scale, self.nchunk),
                .gemv_dualn => {
                    var t: usize = 0;
                    while (t < m) : (t += 4) {
                        const k = @min(4, m - t);
                        const xv = try ctx.view(x, t * w.cols * 4, k * w.cols * 4);
                        try ctx.opGemvQuantDualN(w.dtype, y, t * w.rows + r0, w.rows, xv, wb, w.scale, n, w.cols, k);
                    }
                },
                .gemv_bf16n, .gemv_fp8n, .gemv_f32n => {
                    var t: usize = 0;
                    while (t < m) : (t += 4) {
                        try ctx.opGemvPartial4(x, t * w.cols, self.partials[0], wb, wcode(w.dtype), n, w.cols, self.nchunk);
                        try ctx.opGemvCombine4(y, t * w.rows + r0, w.rows, self.partials[0], n, w.scale, self.nchunk, @min(4, m - t));
                    }
                },
                else => unreachable,
            }
        }
    }
};

/// The kernels' binding bound: a buffer is typed as 2^28 words.
pub const max_binding: usize = 1 << 30;

/// Hands `ops.matmul.probe` the activation a GEMM is about to read, so an activation
/// capture works on this backend. Only for an f32 activation nothing has consumed.
pub fn probeInput(ctx: *gpu.Context, x: Buf, m: usize, w: Weight) !void {
    const p = ops.matmul.probe orelse return;
    if (m == 0 or w.cols == 0) return;
    const n = m * w.cols;
    const host = try ctx.gpa.alloc(f32, n);
    defer ctx.gpa.free(host);
    if (x.dt == .f32) {
        try ctx.tensorDownloadAt(x, 0, std.mem.sliceAsBytes(host));
    } else {
        const raw = try ctx.gpa.alloc(u16, n);
        defer ctx.gpa.free(raw);
        try ctx.tensorDownloadAt(x, 0, std.mem.sliceAsBytes(raw));
        for (host, 0..) |*h, i| h.* = switch (x.dt) {
            .f16 => @as(f16, @bitCast(raw[i])),
            .bf16 => @bitCast(@as(u32, raw[i]) << 16),
            .f32 => unreachable,
        };
    }
    p.input(p.ctx, w, host, m);
}

// --- tests -----------------------------------------------------------------

fn fake(dt: DType, rows: usize, cols: usize) Weight {
    return .{ .bytes = &.{}, .dtype = dt, .rows = rows, .cols = cols };
}

fn llmRoute(c: Caps, w: Weight, m: usize) ?Route {
    return dual_lin.routeOf(Caps, c, w, m, Plan.llm.policy());
}

const full: Caps = .{ .f16w = true, .bf16w = true, .i8 = true, .i4dec = true, .w4a8dec = true, .nvfp4dec = true, .coop_fp8 = true, .wo_pack = true, .f32_gemm = true };

test "every block quant is raw, and the dual GEMV is its decode" {
    for ([_]DType{
        .q8_0,    .q4_0,   .iq4_nl,  .iq4_xs, .q2_k,     .q3_k,
        .q4_k,    .q5_k,   .q6_k,    .q1_0,   .q2_0_g64, .q2_0_g128,
        .iq2_xxs, .iq2_xs, .iq3_xxs, .iq3_s,
    }) |dt|
        try std.testing.expect(gpu.Context.dequantOnly(dt) and gpu.Context.dualGemv(dt));
    try std.testing.expect(!gpu.Context.dequantOnly(.bf16) and !gpu.Context.dualGemv(.bf16));
}

test "a block quant decodes through the dual GEMV and prefills through the coop GEMM" {
    for ([_]DType{ .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl, .iq4_xs, .q4_0, .q1_0, .q2_0_g64, .q2_0_g128, .iq3_s }) |dt|
        try std.testing.expectEqual(@as(?Route, .gemv_dual), llmRoute(Caps.none, fake(dt, 4096, 4096), 1));
    // A small batch takes the grouped dual GEMV, a long one the coop GEMM or nothing.
    const q4 = fake(.q4_k, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_dualn), llmRoute(Caps.none, q4, dual_lin.grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemv_dualn), llmRoute(full, q4, 2));
    try std.testing.expectEqual(@as(?Route, null), llmRoute(Caps.none, q4, dual_lin.grouped_max + 1));
    try std.testing.expectEqual(@as(?Route, .blockq_f16), llmRoute(full, q4, dual_lin.grouped_max + 1));
    try std.testing.expectEqual(@as(?Route, .gemv_dual), llmRoute(full, q4, 1));
    // The dequant GEMM pads any shape, so a skinny weight takes it too: on the grouped
    // GEMV a long batch is a launch per 4 rows.
    try std.testing.expectEqual(@as(?Route, .blockq_f16), llmRoute(full, fake(.q4_k, 40, 4096), 512));
    // A row that is not whole blocks has no lane mapping, so even a single row goes
    // to the GEMM, and to nothing at all without one.
    try std.testing.expectEqual(@as(?Route, .blockq_f16), llmRoute(full, fake(.q1_0, 4096, 4032), 1));
    try std.testing.expectEqual(@as(?Route, null), llmRoute(Caps.none, fake(.q1_0, 4096, 4032), 1));
}

test "dense formats route by row count" {
    const b16 = fake(.bf16, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_bf16), llmRoute(Caps.none, b16, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), llmRoute(Caps.none, b16, 512));
    // With the coop GEMM a long batch takes it rather than 4 rows a pass.
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), llmRoute(full, b16, 512));
    const f8 = fake(.f8_e4m3, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_fp8n), llmRoute(Caps.none, f8, dense_grouped_max));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8_f32), llmRoute(Caps.none, f8, dense_grouped_max + 1));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8), llmRoute(full, f8, dense_grouped_max + 1));
    const f32w = fake(.f32, 4096, 4096);
    try std.testing.expectEqual(@as(?Route, .gemv_f32), llmRoute(Caps.none, f32w, 1));
    try std.testing.expectEqual(@as(?Route, .gemv_f32n), llmRoute(Caps.none, f32w, 3));
    try std.testing.expectEqual(@as(?Route, .gemm_f32), llmRoute(Caps.none, f32w, 512));
    try std.testing.expectEqual(@as(?Route, null), llmRoute(Caps.none, fake(.u8, 8, 8), 1));
}

test "a dense weight whose rows the grouped GEMV cannot tile takes the GEMM" {
    // The grouped kernels decode eight rows a lane group (`opGemvPartial4`); a 100-row
    // router at a 4-row batch computed 96 of them before this rule.
    for ([_]DType{ .f32, .bf16, .f8_e4m3 }) |dt| {
        try std.testing.expect(llmRoute(full, fake(dt, 100, 2560), 4).? != .gemv_f32n);
        try std.testing.expect(llmRoute(full, fake(dt, 100, 2560), 4).? != .gemv_bf16n);
        try std.testing.expect(llmRoute(full, fake(dt, 100, 2560), 4).? != .gemv_fp8n);
    }
    try std.testing.expectEqual(Route.gemv_f32n, llmRoute(full, fake(.f32, 1040, 2560), 4).?);
}

test "a skinny f32 weight takes the row GEMV at any batch it covers" {
    for ([_]usize{ 1, 4, 19, 512 }) |m| try std.testing.expectEqual(Route.gemv_f32_rows, llmRoute(full, fake(.f32, 100, 2560), m).?);
    try std.testing.expectEqual(Route.gemm_f32, llmRoute(full, fake(.f32, 100, 2560), 513).?);
    try std.testing.expectEqual(Route.gemv_f32, llmRoute(full, fake(.f32, 4096, 4096), 1).?);
}

test "a diffusion plan takes every format the device has a kernel for" {
    const p: Plan = .{ .rot = true, .blockq = .auto };
    var q = fake(.i8, 4096, 4096);
    q.row_scale = &.{};
    const pol = p.policy();
    try std.testing.expectEqual(@as(?Route, .i8), dual_lin.routeOf(Caps, full, q, 1, pol));
    try std.testing.expectEqual(@as(?Route, .i4_i8), dual_lin.routeOf(Caps, full, fake(.i4, 4096, 4096), 1, pol));
    try std.testing.expectEqual(@as(?Route, null), dual_lin.routeOf(Caps, Caps.none, fake(.i4, 4096, 4096), 1, pol));
    // A block quant takes the f16 dequant GEMM where the activation fits f16, the bf16
    // one where the family needs bf16 range, and is refused there without that GEMM.
    try std.testing.expectEqual(@as(?Route, .blockq_f16), dual_lin.routeOf(Caps, full, fake(.q4_k, 4096, 4096), 1, pol));
    var wide = p;
    wide.act16 = .bf16;
    try std.testing.expectEqual(@as(?Route, .blockq_bf16), dual_lin.routeOf(Caps, full, fake(.q4_k, 4096, 4096), 1, wide.policy()));
    var no_bf = full;
    no_bf.bf16w = false;
    try std.testing.expectEqual(@as(?Route, null), dual_lin.routeOf(Caps, no_bf, fake(.q4_k, 4096, 4096), 1, wide.policy()));
    var asked = p;
    asked.blockq = .bf16;
    try std.testing.expectEqual(@as(?Route, .blockq_bf16), dual_lin.routeOf(Caps, full, fake(.q4_k, 4096, 4096), 1, asked.policy()));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8_f32), dual_lin.routeOf(Caps, full, fake(.f8_e4m3, 4096, 4096), 1, wide.policy()));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8), dual_lin.routeOf(Caps, full, fake(.f8_e4m3, 4096, 4096), 1, pol));
    try std.testing.expectEqual(@as(?Route, .gemm_fp8_f32), dual_lin.routeOf(Caps, full, fake(.f8_e4m3, 4000, 4096), 1, pol));
}

test "check refuses an int8 shape the GEMM cannot store and an unrotated int4" {
    var q = fake(.i8, 96, 512);
    q.row_scale = &@as([96]f32, @splat(1));
    q.tag = "blk.0.q";
    try std.testing.expectEqual(Why.int_shape, dual_lin.check(Caps, full, &.{q}, .{ .blockq = .auto }).refused.why);
    q = fake(.i8, 128, 512);
    q.row_scale = &@as([128]f32, @splat(1));
    try std.testing.expect(dual_lin.check(Caps, full, &.{q}, .{ .blockq = .auto }) == .ok);
    var q4 = fake(.i4, 128, 512);
    q4.row_scale = &@as([128]f32, @splat(1));
    try std.testing.expectEqual(Why.int4_unrotated, dual_lin.check(Caps, full, &.{q4}, .{ .blockq = .auto }).refused.why);
}

test "wcode reads bf16 and fp8 natively and everything else as f32" {
    try std.testing.expectEqual(gpu.WCode.bf16, wcode(.bf16));
    try std.testing.expectEqual(gpu.WCode.f8, wcode(.f8_e4m3));
    try std.testing.expectEqual(gpu.WCode.f32, wcode(.f32));
}

test "the fp8 GEMM takes only weights whose rows fill its workgroup tile" {
    var c = full;
    c.fp8_rows = 256;
    try std.testing.expect(c.fp8Gemm(fake(.f8_e4m3, 512, 64)));
    try std.testing.expect(!c.fp8Gemm(fake(.f8_e4m3, 384, 64)));
    c.fp8_rows = 128;
    try std.testing.expect(c.fp8Gemm(fake(.f8_e4m3, 384, 64)));
}

test "a 16-bit weight takes the f32 GEMM, widened, only where there is no 16-bit GEMM" {
    const f32_only: Caps = .{ .f16w = false, .bf16w = false, .i8 = false, .i4dec = false, .w4a8dec = false, .nvfp4dec = false, .coop_fp8 = false, .f32_gemm = true };
    const pol = (Plan{ .rot = true, .blockq = .auto }).policy();
    for ([_]DType{ .f16, .bf16 }) |dt| {
        try std.testing.expectEqual(@as(?Route, .gemm_f32), dual_lin.routeOf(Caps, f32_only, fake(dt, 4096, 4096), 1, pol));
        try std.testing.expectEqual(@as(?Route, null), dual_lin.routeOf(Caps, Caps.none, fake(dt, 4096, 4096), 1, pol));
    }
    // An LLM batch: f16 has no other GEMM, bf16 keeps its grouped GEMV.
    try std.testing.expectEqual(@as(?Route, .gemm_f32), llmRoute(f32_only, fake(.f16, 4096, 4096), 512));
    try std.testing.expectEqual(@as(?Route, .gemv_bf16n), llmRoute(f32_only, fake(.bf16, 4096, 4096), 512));
    try std.testing.expectEqual(@as(?Route, .gemm_bf16), dual_lin.routeOf(Caps, full, fake(.bf16, 4096, 4096), 1, pol));
}

test "a weight-only width the device has no GEMM for widens to f32, never narrows" {
    var q = fake(.i8, 4096, 4096);
    q.row_scale = &.{};
    var no_bf = full;
    no_bf.bf16w = false;
    var pol = (Plan{ .rot = true, .blockq = .auto }).policy();
    for ([_]dual_lin.IntGemm{ .weight_only, .weight_only_f16, .weight_only_f32 }, [_]Route{ .i8_wo, .i8_wo16, .i8_wo32 }, [_]Route{ .i8_wo32, .i8_wo16, .i8_wo32 }) |g, on_full, on_no_bf| {
        pol.int8 = g;
        try std.testing.expectEqual(@as(?Route, on_full), dual_lin.routeOf(Caps, full, q, 64, pol));
        try std.testing.expectEqual(@as(?Route, on_no_bf), dual_lin.routeOf(Caps, no_bf, q, 64, pol));
        try std.testing.expectEqual(@as(?Route, null), dual_lin.routeOf(Caps, Caps.none, q, 64, pol));
    }
}
