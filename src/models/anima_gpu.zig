//! GPU-resident Anima (`MiniTrainDIT` + `LLMAdapter`) forward on Vulkan: the whole
//! 28-block trunk runs on the device, with the small per-image paths kept on the host
//! exactly as `zimage_gpu` keeps Z-Image's.
//!
//! **What stays on the CPU, and why none of it is laziness:**
//!
//! - The **entire `llm_adapter`** (6 blocks, 1024 wide, 512 rows). ⚠️ It does not
//!   depend on the timestep at all — its inputs are the two tokenizations and the
//!   Qwen3 encoder's states — so it is computed once per *image*, not per step, and
//!   `pipeline` already runs it inside `encode`. Moving it to the device would buy
//!   ~1/30th of one render and cost a second model's worth of device weights.
//! - The **timestep sinusoid, `t_embedder` MLP and every block's AdaLN pair**,
//!   precomputed for the whole schedule at `Session.init` (see
//!   `anima.DiT.modulationTable`). Anima's modulation is one vector per sublayer per
//!   block for the *whole image* — `emb` is `[B, T, D]` with T = 1, broadcast over h
//!   and w — so the entire AdaLN evaluation is ~0.2 GFLOP against a trunk of
//!   hundreds, and hoisting it out of the step loop is what `dit_gpu` does with its
//!   timestep vectors for the same reason.
//! - **Patchify** and the **final layer**, a transposition and a GEMV apiece.
//!
//! **The one thing that is NOT just "the CPU forward with device ops":**
//!
//! ⚠️ **Cross-attention's K and V are per-IMAGE constants and are precomputed for
//! every block at `Session.init`.** They are projections of the adapter's output,
//! which does not change across steps, so 28 blocks x 2 GEMMs of
//! `[512, 2048] x [2048, 1024]` leave the step loop entirely — along with the f16
//! conversion and k-major gather they would need every step, since they are cached
//! directly in the **attention-operand layout** (K per-head k-major, V row-major).
//! The K norm is per-image too and is applied before caching. Same observation
//! `zimage_gpu` makes about its `modulation=False` caption half.
//!
//! ⚠️ **It is a SMALL win, and an earlier version of this comment claimed 17% of the
//! trunk's per-step GEMM FLOPs — wrong by a factor of 30**, from comparing a 28-block
//! total against a per-block figure. The true share is **0.56%** at 6534 tokens (4.29
//! GFLOP against a block's 767) and 2.4% at 1536, because these GEMMs run over the
//! 512-row *context* while every other one runs over the *sequence*. **Measured** by
//! `anima-cuda-bench`: 0.10 ms/block, so hoisting them saves **0.38% of a step**. The cost
//! is real — 117 MB of f16 operands here (235 MB of f32 on CUDA), doubled under CFG. Kept
//! because it is also what removes the per-step conversion, but it is not why this port
//! is fast.
//!
//! **No new attention kernel was needed** — `coopmat.buildFlashAttn` was already
//! parameterized by query tiles and key length independently; what it lacked was a
//! separate MD plane stride, since `s_stride` served as K's row stride, the j loop
//! bound *and* the MD plane stride at once. Push word 7 now carries it, with 0
//! meaning "= s_stride" so every pre-existing caller is unchanged by construction.
//! One genuinely new kernel: `ln_mod_sg`, the fused weightless LayerNorm +
//! modulation, which is a subgroup kernel because a thread-per-row form at
//! 6534 x 2048 x 3 calls x 28 blocks is the bandwidth trap `qk_rmsnorm_warp`
//! already paid for once.
//!
//! ⚠️ **Head width is exactly 128** — the width `coopmat.buildGemmAttnOut` and the
//! flash kernels tile — so unlike the SD family there is no head padding and no
//! `head_pad`/`head_unpad` round trip. Every GEMM width here (2048 / 8192 / 1024) is
//! a multiple of 128 and every reduction width a multiple of 32, so `opMatmulCoop*`'s
//! shape constraints are met without special-casing.

const std = @import("std");
const anima = @import("anima.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = anima.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the correctness-first `attn_cross` path (one thread per (query, head), any
/// q/kv shape) instead of the tensor-core flash pipeline. For A/B and for reproducing
/// a mismatch.
///
/// ⚠️ **Read once, at `Session.init`** — not per call — because it decides whether the
/// per-image cross-attention K/V are cached as f16 attention operands or as plain f32.
/// Flipping it under a live session would leave the forward reading buffers that were
/// never filled. `Session.tc` is the value actually in force, and `forward` branches on
/// that, never on this.
pub var force_attn_cross: bool = false;

/// Widest zero bias any Anima GEMM needs (the MLP's `mlp_dim`).
///
/// ⚠️ Passed WHOLE, never sliced: `smallBuffer` caches by host POINTER alone and sizes
/// from the first call, so once a 2048-wide GEMM cached it, an 8192-wide one would read
/// off the end. The dispatch reads only `rows` entries, so one full-width buffer serves
/// every GEMM — the same rule `dit_gpu` and `zimage_gpu` record.
const zero_bias: [anima.anima_2b.mlp_dim]f32 = @splat(0);

/// Whether the tensor-core flash pipeline is available and wanted.
fn useFlash(ctx: *gpu.Context) bool {
    return !force_attn_cross and ctx.pipe_flash_md != .null_handle;
}

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: anima.Config,
    lat_h: usize,
    lat_w: usize,
    /// Image tokens, `(lat_h/patch) * (lat_w/patch)`.
    seq: usize,
    seq_pad: usize,
    /// Rows of the adapter's output the trunk cross-attends to.
    ctx_seq: usize,
    ctx_pad: usize,
    /// Whether the flash path is in force for this session — see `force_attn_cross`.
    tc: bool,

    /// 3-axis RoPE table for the image grid: `cos` then `sin`, `seq * half` each.
    freqs_d: Buf,

    /// Cross-attention K/V, precomputed ONE BUFFER PER BLOCK (see the module header).
    /// Under `tc`, `ck_d[i]` is f16 per-head k-major `[heads][hd][ctx_pad]` and
    /// `cv_d[i]` is f16 `[ctx_pad][dim]`; otherwise both are f32 `[ctx_seq][dim]`,
    /// which is what `attn_cross` reads.
    ///
    /// ⚠️ **Per block rather than one buffer with a per-block offset, because on Vulkan
    /// a `DeviceBuffer.buf` is an opaque HANDLE, not a device pointer.** `zimage_cuda`'s
    /// `offsetBuf` trick — adding a byte offset to `.buf` — is meaningful only on CUDA;
    /// doing it here produced an invalid handle and `error_device_lost` on the first
    /// cross-attention, at a latent far too small for the watchdog to be the cause.
    /// A descriptor binds a whole buffer, so the offset has to be an allocation.
    ck_d: []Buf,
    cv_d: []Buf,
    /// How many entries of `ck_d`/`cv_d` hold a live buffer. Equal to `n_layers` after
    /// a successful `init`; smaller only while one is being built, so the errdefer
    /// destroys exactly what exists.
    n_kv: usize,

    /// The schedule this session precomputed for, and the FOLDED per-sigma
    /// modulation tables (see `anima.DiT.foldModulationTable`).
    sigmas: []f32,
    /// Borrowed, model-owned: the FOLDED table for every scheduled sigma, stride
    /// `mod_stride`. ⚠️ Per SIGMA, not per conditioning — see
    /// `anima.DiT.modulationSchedule` for why building it per session cost 0.8 s an image.
    mods: []const f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cond: []const f32,
        ctx_seq: usize,
        sigmas: []const f32,
        /// Model-owned, shared by both conditioning branches (`modulationSchedule`).
        mods: []const f32,
    ) !Session {
        const cfg = model.cfg;
        std.debug.assert(cond.len == ctx_seq * cfg.context_dim);
        std.debug.assert(lat_h % cfg.patch == 0 and lat_w % cfg.patch == 0);
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const seq = h * w;
        const half = cfg.headDim() / 2;

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .seq = seq,
            .seq_pad = std.mem.alignForward(usize, seq, 128),
            .ctx_seq = ctx_seq,
            .ctx_pad = std.mem.alignForward(usize, ctx_seq, 128),
            .tc = useFlash(ctx),
            .freqs_d = undefined,
            .ck_d = &.{},
            .cv_d = &.{},
            .n_kv = 0,
            .sigmas = &.{},
            .mods = mods,
            .mod_stride = anima.modulationTableLen(cfg),
        };
        var made: usize = 0;
        errdefer {
            if (made > 0) ctx.tensorDestroy(&self.freqs_d);
            for (self.ck_d[0..self.n_kv]) |*b| ctx.tensorDestroy(b);
            for (self.cv_d[0..self.n_kv]) |*b| ctx.tensorDestroy(b);
            if (self.ck_d.len != 0) gpa.free(self.ck_d);
            if (self.cv_d.len != 0) gpa.free(self.cv_d);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
        }

        {
            var f = try model.ropeFreqs(gpa, h, w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }

        const t0 = std.Io.Clock.real.now(io);
        try self.buildCrossKv(gpa, ctx, model, cond);
        const t1 = std.Io.Clock.real.now(io);
        if (t1.nanoseconds - t0.nanoseconds > 20_000_000) std.log.info(
            "[anima_gpu] cross K/V ({d} blocks, {d} device buffers): {d:.2}s",
            .{ cfg.n_layers, 2 * cfg.n_layers, @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e9 },
        );

        std.debug.assert(mods.len == sigmas.len * self.mod_stride);
        self.sigmas = try gpa.dupe(f32, sigmas);
        return self;
    }

    /// Project the adapter's output through every block's cross-attention `k`/`v`
    /// (and `knorm`) once, into the layout the active attention path reads.
    ///
    /// Runs on the **host**: it is one 512-row pass per block, done once per image,
    /// and doing it on the device would mean uploading 28 blocks' k/v weights before
    /// the first step rather than letting the step loop stream them in the order it
    /// wants them. ⚠️ Which also means the k/v weights are touched here and again in
    /// the trunk — but only their *device* copies are cached, and this path never
    /// creates one.
    fn buildCrossKv(
        self: *Session,
        gpa: std.mem.Allocator,
        ctx: *gpu.Context,
        model: *const DiT,
        cond: []const f32,
        ) !void {
        const cfg = self.cfg;
        const d = cfg.dim;
        const n = self.ctx_seq;
        const hd = cfg.headDim();
        const heads = cfg.n_heads;
        const nl = cfg.n_layers;

        self.ck_d = try gpa.alloc(Buf, nl);
        self.cv_d = try gpa.alloc(Buf, nl);

        // ⚠️ **On the DEVICE, and it used to be on the host — that was ~0.4 s of a 0.73 s
        // per-image setup**, the Vulkan half of the "Anima images are slow to start"
        // report. The CUDA arm was moved first; this is its twin. What made the host
        // version look cheap was pricing it as a share of a STEP (0.56%) rather than as a
        // one-off before the first one.
        var cond_d = try ctx.tensorCreate(cond.len * 4);
        defer ctx.tensorDestroy(&cond_d);
        try ctx.tensorUpload(cond_d, std.mem.sliceAsBytes(cond));

        // The cross K/V linears go through `lin` like any other, so a W4A8 checkpoint
        // decodes here too and needs its scratch sized to the model-wide maximum first —
        // growing it later would be correct but costs a submit-and-wait per growth.
        {
            const w4 = anima.maxW4A8Scratch(model, gpu.Context.w4a8ScratchBytes);
            if (w4 > 0) try ctx.ensureDeviceBuffer(&ctx.w4a8_t, w4);
        }

        // f32 scratch for one block's projections, reused across blocks. 128-row padded
        // because a quantized GEMM writes `i8_mpad` rows.
        const pad = std.mem.alignForward(usize, n, 128);
        var ks = try ctx.tensorCreate(pad * d * 4);
        defer ctx.tensorDestroy(&ks);
        var vs = try ctx.tensorCreate(pad * d * 4);
        defer ctx.tensorDestroy(&vs);

        for (model.blocks, 0..) |_, bi| {
            const kw = model.blocks[bi].cross_attn.k;
            const vw = model.blocks[bi].cross_attn.v;
            // f16 operands where the flash path reads them; plain f32 for `attn_cross`.
            const kb: usize = if (self.tc) heads * hd * self.ctx_pad * 2 else n * d * 4;
            const vb: usize = if (self.tc) self.ctx_pad * d * 2 else n * d * 4;
            var kbuf = try ctx.tensorCreate(kb);
            errdefer ctx.tensorDestroy(&kbuf);
            var vbuf = try ctx.tensorCreate(vb);
            errdefer ctx.tensorDestroy(&vbuf);

            // k and v share the context activation, so one prep serves both.
            try prepGroup(ctx, cond_d, n, cfg.context_dim, &.{ kw, vw });
            try lin(ctx, ks, cond_d, n, kw);
            try lin(ctx, vs, cond_d, n, vw);
            // ⚠️ K is normed, V is NOT (`v_norm = nn.Identity()`) — the asymmetry
            // `DiT.projectKv` owns on the host path.
            try qkNorm(ctx, ks, model.blocks[bi].cross_attn.knorm, n * heads, hd, cfg.qk_eps);

            if (self.tc) {
                // Straight into the attention-operand layout: K per-head k-major, V
                // row-major, both zero-padded past `ctx_seq` so a padded key scores 0.
                try ctx.opElt(.gather_kmajor_h16, ks, null, null, kbuf, .{
                    .u0 = @intCast(d * self.ctx_pad / 2),
                    .u1 = @intCast(hd),
                    .u2 = @intCast(self.ctx_pad),
                    .u3 = @intCast(n),
                    .u4 = @intCast(heads),
                }, d * self.ctx_pad / 2, 1, 1);
                try ctx.opElt(.f32_to_h16, vs, null, null, vbuf, .{
                    .u0 = @intCast(self.ctx_pad * d / 2),
                    .u1 = @intCast(n * d),
                    .f0 = 1.0,
                }, self.ctx_pad * d / 2, 1, 1);
            } else {
                try ctx.opElt(.copy, ks, kbuf, null, null, .{ .u0 = @intCast(n * d), .u2 = 0, .u3 = 0 }, n * d, 1, 1);
                try ctx.opElt(.copy, vs, vbuf, null, null, .{ .u0 = @intCast(n * d), .u2 = 0, .u3 = 0 }, n * d, 1, 1);
            }
            self.ck_d[bi] = kbuf;
            self.cv_d[bi] = vbuf;
            self.n_kv = bi + 1;
        }
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.freqs_d);
        for (self.ck_d[0..self.n_kv]) |*b| ctx.tensorDestroy(b);
        for (self.cv_d[0..self.n_kv]) |*b| ctx.tensorDestroy(b);
        gpa.free(self.ck_d);
        gpa.free(self.cv_d);
        gpa.free(self.sigmas);
        self.* = undefined;
    }

    /// The precomputed (folded) modulation for `sigma`, or null when the caller probed
    /// a sigma off the schedule this session was built for — a teacher-forced
    /// measurement; `forward` then builds one on the fly.
    fn tv(self: *const Session, sigma: f32) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return self.mods[i * self.mod_stride ..][0..self.mod_stride];
        }
        return null;
    }
};

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// Per-resolution device scratch, shared by both conditioning branches.
pub const Workspace = struct {
    x_d: Buf, // [seq][dim] the running activation
    nrm_d: Buf, // [seq][dim] modulated pre-norm
    dlt_d: Buf, // [seq][dim] sublayer output
    q_d: Buf, // [seq_pad][dim]
    k_d: Buf, // [seq_pad][dim] self-attention only
    v_d: Buf, // [seq_pad][dim] self-attention only
    /// `[seq_pad][dim]` attention output, followed by the flash MD table
    /// (`[heads][seq_pad] x {max, 1/sum}`) in the tail at `mdOffset()`.
    attn_d: Buf,
    mlp_d: Buf, // [seq][mlp_dim]
    mod_d: Buf, // this step's whole folded modulation table
    patch_d: Buf, // [seq][patchDim] raw patches
    qt_d: Buf, // f16 [seq_pad][dim], softmax scale prefolded
    kt_d: Buf, // f16 per-head k-major [heads][hd][seq_pad] (self-attention)
    v16_d: Buf, // f16 [seq_pad][dim] (self-attention)

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mlp_d", "mod_d", "patch_d", "qt_d", "kt_d", "v16_d" };

    /// ⚠️ `tc` comes from the SESSION (`Session.tc`), not from `useFlash(ctx)` again: the
    /// f16 attention scratch exists only on the flash path, and two independent reads of
    /// a mutable global could disagree if it were flipped between the two constructions.
    /// One decision, made once, passed in.
    pub fn init(ctx: *gpu.Context, model: *const DiT, lat_h: usize, lat_w: usize, tc: bool) !Workspace {
        const cfg = model.cfg;
        const seq = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const d = cfg.dim;
        // ⚠️ **Every activation buffer is 128-ROW PADDED.** Two independent reasons, and
        // both are silent when violated: the f16 attention conversion writes whole padded
        // rows and the flash `out` pass writes `seq_pad` of them back; and a quantized
        // GEMM writes `i8_mpad` rows (`opI8Prep` pads m up to 128 and the coop tile is
        // 128x128), not `m`. The CUDA twin learned the second one the hard way —
        // `compute-sanitizer` caught `igemm_pipe_fused` writing past a seq-sized buffer.
        const sizes = [fields.len]usize{
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            (seq_pad * d + cfg.n_heads * seq_pad * 2) * 4,
            seq_pad * cfg.mlp_dim * 4,
            anima.modulationTableLen(cfg) * 4,
            seq * cfg.patchDim() * 4,
            if (tc) seq_pad * d * 2 else 16,
            if (tc) d * seq_pad * 2 else 16,
            if (tc) seq_pad * d * 2 else 16,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether this context can run Anima's block GEMMs at all. The trunk weights are
/// dense bf16 in every checkpoint seen so far, which needs one of the two f16-weight
/// tensor-core pipelines; a device without either has no path and must stay on the
/// CPU rather than silently reading the bytes as something else — the failure
/// `qwen3_gpu.supportsWeights` exists to prevent.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    // ⚠️ The fused norm is checked HERE, not discovered inside the first forward. Every
    // block pre-norm goes through `ln_mod_sg`, and there is no unfused fallback: the
    // shared `modulate` kernel adds its own `1 +`, so it cannot consume the folded table
    // the device path uploads. Without this gate a device lacking subgroup arithmetic
    // would build a whole session and then error three stages in, where `Session.denoiser`
    // can instead warn and leave the trunk on the CPU.
    if (!ctx.hasLnModSg()) return false;
    // ⚠️ EVERY block's linears, not block 0's — see `anima.unsupportedGpuLin`.
    // ⚠️ int8 convrot needs the `sint8` coopmat pipeline; **int4 is absent because no
    // `sint4` coopmat exists on this device** (see the module header).
    // ⚠️ int4 is accepted because `lin` DECODES it to int8 per GEMM (`ctx.i4Decode`), not
    // because a `sint4` coopmat exists — it does not. The packed 4-bit form stays resident,
    // which is what makes int4 cheaper than int8 here rather than the same price.
    const has_i8 = ctx.pipe_coop_i8 != .null_handle;
    // ⚠️ int4 AND W4A8 each need BOTH the int8 pipeline and their own decode kernel that
    // feeds it — each decode's output is an ordinary int8-convrot weight, so neither half
    // alone is a path. (int4 used to need only `has_i8`, because it was widened at load.)
    const support: anima.LinSupport = .{
        .i8 = has_i8,
        .i4 = has_i8 and ctx.hasI4Decode(),
        .w4a8 = has_i8 and ctx.hasW4A8Decode(),
        .nvfp4 = ctx.hasNvfp4Decode(),
    };
    if (anima.unsupportedLin(model, support) != null) return false;
    for (model.blocks) |*b| {
        // A dense block still needs one of the f16-weight pipelines.
        const lins = [_]Weight{ b.self_attn.q, b.cross_attn.q, b.mlp1, b.mlp2 };
        for (lins) |w| {
            if (anima.linKind(w) != .dense) continue;
            if (w.dtype == .bf16 or w.dtype == .f16) {
                if (ctx.pipe_coop_bf16w == .null_handle and ctx.pipe_coop_f16w == .null_handle) return false;
            }
        }
    }
    return true;
}

/// Quantize+rotate the activation once for a group of GEMMs that share it, if any member
/// needs it. Mirrors `anima_cuda.prepGroup`; `Context.opI8Prep` likewise writes its own
/// `i8_x`/`i8_scale` rather than overwriting `x`, so a dense GEMM in the same group is
/// still free to read the f32 activation.
///
/// ⚠️ `cols` must be one of `gpu.i8_prep_cols` or the prep falls back to a 3-pass chain
/// that round-trips a full f32 copy of the activation through global memory. Anima's 2048
/// and 8192 were added to that table for this reason.
fn prepGroup(ctx: *gpu.Context, x: Buf, m: usize, cols: usize, group: []const Weight) !void {
    // ⚠️ Grouped by the PREP a kind needs, not by the kind: int8 and W4A8 share one
    // (`anima.prepKind` says why), so a block mixing them pays a single `opI8Prep`.
    var want: anima.PrepKind = .none;
    for (group) |w| {
        const k = anima.prepKind(anima.linKind(w));
        if (k == .none) continue;
        if (want != .none and want != k) {
            std.log.err("anima_gpu: a linear group mixes {t} and {t} activation preps; one cannot serve both", .{ want, k });
            return error.UnsupportedCheckpoint;
        }
        want = k;
    }
    switch (want) {
        // ⚠️ **`.i4` takes the INT8 prep here, and that is not a bug.** `anima.prepKind` maps
        // int4 to an int4 activation prep because the CUDA arms run true W4A4; Vulkan has no
        // `sint4` coopmat, so its int4 weights are decoded to int8 per GEMM (`lin`) and the
        // activation stays 8-bit. The prep is a property of the BACKEND's GEMM, not of the
        // weight's storage, which is why this mapping is here rather than in `anima.zig`.
        .i8, .i4 => try ctx.opI8Prep(x, m, cols),
        .none => {},
    }
}

/// One block linear. A quantized weight reads the prep state and fuses the per-row
/// rescale; a dense one goes to `gemm`.
///
/// ⚠️ `opI8Gemm` requires `rows % (16 * coopmat.i8_nt) == 0`, i.e. a multiple of 64. Every
/// linear the device runs here is 2048 or 8192; the AdaLN pair's 256/6144 and
/// cross-attention's k/v are evaluated on the HOST.
fn lin(ctx: *gpu.Context, y: Buf, x: Buf, m: usize, w: Weight) !void {
    switch (anima.linKind(w)) {
        .i8 => {
            std.debug.assert(w.rows % 64 == 0);
            try ctx.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
        },
        // ⚠️ **int4 on Vulkan is W4A8-shaped, not W4A4**: there is no `sint4` cooperative
        // matrix, so the nibbles must reach the GEMM as int8 either way and the activation
        // stays int8 (`opI8Prep`). What changed on 2026-08-08 is only WHERE that unpack
        // happens — per GEMM into a scratch, rather than once at load into a resident int8
        // copy. Identical arithmetic (a 4-bit value is exact in 8 bits, and the row scale
        // and rotation are untouched), so renders are bit-identical; it buys 660 MB.
        .i4 => {
            std.debug.assert(w.rows % 64 == 0);
            const wbuf = try ctx.i4Decode(w.bytes, w.rows, w.cols);
            try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows, false);
        },
        // W4A8 reads the SAME prep state and runs the SAME int8 kernel as `.i8` above; the
        // packed 4-bit weight is decoded (and k-major transposed) into the context's
        // scratch on the way in, so the 4-bit form stays resident.
        // Identical in shape to the `.i4` arm above; the only difference is that a W4A8
        // nibble indexes a codebook scaled per group, where an int4 nibble sign-extends.
        .w4a8 => {
            std.debug.assert(w.rows % 64 == 0);
            const meta = w.w4a8.?;
            const wbuf = try ctx.w4a8Decode(w.bytes, meta.s_rel, std.mem.asBytes(meta.levels), w.rows, w.cols, meta.group_size);
            try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows, false);
        },
        // Weight-only NVFP4: decoded to an f16 scratch inside the GEMM, so the 4-bit form
        // stays resident — like both int arms above, decoded per GEMM rather than unpacked
        // at load, since unpacking gives back exactly the memory a 4-bit format buys.
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            // ⚠️ The f16-weight coop GEMM ALWAYS folds a bias in, so it needs the full
            // zero vector — an empty slice would give `smallBuffer` a 0-byte buffer for
            // `bias_compact` to read `rows` floats out of. Same reason `gemm` does it.
            const zeros: []const f32 = &zero_bias;
            std.debug.assert(w.rows <= zeros.len);
            try ctx.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zeros);
        },
        .dense => try gemm(ctx, y, x, m, w),
    }
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = model.cfg;
    const d = cfg.dim;
    const seq = sess.seq;
    const hd = cfg.headDim();
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    std.debug.assert(x_lat.len == cfg.channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == cfg.out_channels * sess.lat_h * sess.lat_w);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const mod = sess.tv(sigma) orelse blk: {
        const tbl = try model.modulationTable(io, gpa, sigma);
        anima.foldModulationTable(cfg, tbl);
        owned = tbl;
        break :blk tbl;
    };

    const patches = try anima.patchify(gpa, cfg, x_lat, sess.lat_h, sess.lat_w);
    defer gpa.free(patches);

    try ctx.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mod));
    try ctx.tensorUpload(ws.patch_d, std.mem.sliceAsBytes(patches));

    // Pre-size the NVFP4 decode scratch to the model's widest linear BEFORE the batch
    // opens: growing it mid-forward is safe (Vulkan's `ensureDeviceBuffer` flushes first)
    // but costs a submit-and-wait per growth, and the first block would pay several.
    {
        const need = anima.maxNvfp4Scratch(model, gpu.Context.nvfp4ScratchBytes);
        if (need > 0) try ctx.ensureDeviceBuffer(&ctx.nvfp4_w16, need);
        const w4 = anima.maxW4A8Scratch(model, gpu.Context.w4a8ScratchBytes);
        if (w4 > 0) try ctx.ensureDeviceBuffer(&ctx.w4a8_t, w4);
    }

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();

    // `x_embedder` is one of the two weights the loader materializes to f32 (it feeds
    // the fused `opMatmul`, which has no bf16 pipeline), so it takes the f32 arm.
    try ctx.opMatmul(ws.x_d, 0, ws.patch_d, 0, seq, model.x_embedder.bytes, model.x_embedder.dtype == .f8_e4m3, d, cfg.patchDim(), model.x_embedder.scale, null);

    for (model.blocks, 0..) |*blk, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try blockForward(ctx, sess, cfg, blk, ws, bi, @intCast(bi * 9 * d), attn_scale);
    }
    try ctx.endBatch();

    // --- the final layer, on the host ------------------------------------------
    // Row-wise, so this is exactly what the CPU `finalize` computes.
    const rows = try gpa.alloc(f32, seq * d);
    defer gpa.free(rows);
    try ctx.tensorDownloadAt(ws.x_d, 0, std.mem.sliceAsBytes(rows));
    try model.finalize(io, gpa, out, rows, mod[cfg.n_layers * 9 * d ..][0 .. 2 * d], sess.lat_h, sess.lat_w);
}

/// One `MiniTrainDIT` `Block` on the device. `ws.x_d` is `[seq][dim]`, modified in
/// place; `mod_base` is this block's offset into the uploaded modulation table.
///
/// The three sublayers are identical in shape — modulated weightless LayerNorm,
/// sublayer, gated residual — and differ only in which third of the table they read
/// and, for attention, whether RoPE applies (Anima's convention 6: self-attention
/// only).
fn blockForward(
    ctx: *gpu.Context,
    sess: *Session,
    cfg: anima.Config,
    blk: anytype,
    ws: *Workspace,
    bi: usize,
    mod_base: u32,
    attn_scale: f32,
) !void {
    const d = cfg.dim;
    const dim32: u32 = @intCast(d);
    const seq = sess.seq;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const half = hd / 2;
    const total: u32 = @intCast(seq * d);

    // --- self-attention --------------------------------------------------------
    // The pre-norm's scale block carries the `1 +` already (`foldModulationTable`).
    try lnMod(ctx, ws, seq, d, mod_base + dim32, mod_base, cfg.norm_eps);
    // q/k/v share one activation, so one prep serves all three.
    try prepGroup(ctx, ws.nrm_d, seq, d, &.{ blk.self_attn.q, blk.self_attn.k, blk.self_attn.v });
    try lin(ctx, ws.q_d, ws.nrm_d, seq, blk.self_attn.q);
    try lin(ctx, ws.k_d, ws.nrm_d, seq, blk.self_attn.k);
    try lin(ctx, ws.v_d, ws.nrm_d, seq, blk.self_attn.v);

    // ⚠️ Anima's Q/K norms take the BLOCKS' 1e-6, not `finfo(f32).eps` — unlike
    // Z-Image, whose `RMSNorm(head_dim)` is built with no `eps` at all. Same kernel,
    // different push constant, and no error either way.
    ctx.independent(2);
    try qkNorm(ctx, ws.q_d, blk.self_attn.qnorm, seq * heads, hd, cfg.qk_eps);
    try qkNorm(ctx, ws.k_d, blk.self_attn.knorm, seq * heads, hd, cfg.qk_eps);
    // ⚠️ SPLIT-half RoPE (pairs `(i, i + 64)`), not the interleaved form Z-Image and
    // krea2 use — hence `rope_half` rather than `rope_inter`.
    ctx.independent(2);
    try ropeHalf(ctx, ws.q_d, sess.freqs_d, seq, heads, half);
    try ropeHalf(ctx, ws.k_d, sess.freqs_d, seq, heads, half);

    try attention(ctx, sess, cfg, ws, .{
        .k = ws.k_d,
        .v = ws.v_d,
        .kv_seq = seq,
        .kv_pad = sess.seq_pad,
        .prepared = false,
    }, attn_scale);
    try prepGroup(ctx, ws.attn_d, seq, d, &.{blk.self_attn.out});
    try lin(ctx, ws.dlt_d, ws.attn_d, seq, blk.self_attn.out);
    try ctx.opElt(.gated_add, ws.x_d, ws.dlt_d, ws.mod_d, null, .{
        .u0 = total,
        .u1 = dim32,
        .u2 = mod_base + 2 * dim32,
    }, seq * d, 1, 1);

    // --- cross-attention onto the adapter's output ------------------------------
    // ⚠️ No RoPE here at all, and K/V come from the session's per-image cache rather
    // than from two GEMMs — see the module header.
    try lnMod(ctx, ws, seq, d, mod_base + 4 * dim32, mod_base + 3 * dim32, cfg.norm_eps);
    try prepGroup(ctx, ws.nrm_d, seq, d, &.{blk.cross_attn.q});
    try lin(ctx, ws.q_d, ws.nrm_d, seq, blk.cross_attn.q);
    try qkNorm(ctx, ws.q_d, blk.cross_attn.qnorm, seq * heads, hd, cfg.qk_eps);
    try attention(ctx, sess, cfg, ws, .{
        .k = sess.ck_d[bi],
        .v = sess.cv_d[bi],
        .kv_seq = sess.ctx_seq,
        .kv_pad = sess.ctx_pad,
        .prepared = true,
    }, attn_scale);
    try prepGroup(ctx, ws.attn_d, seq, d, &.{blk.cross_attn.out});
    try lin(ctx, ws.dlt_d, ws.attn_d, seq, blk.cross_attn.out);
    try ctx.opElt(.gated_add, ws.x_d, ws.dlt_d, ws.mod_d, null, .{
        .u0 = total,
        .u1 = dim32,
        .u2 = mod_base + 5 * dim32,
    }, seq * d, 1, 1);

    // --- MLP -------------------------------------------------------------------
    try lnMod(ctx, ws, seq, d, mod_base + 7 * dim32, mod_base + 6 * dim32, cfg.norm_eps);
    try prepGroup(ctx, ws.nrm_d, seq, d, &.{blk.mlp1});
    try lin(ctx, ws.mlp_d, ws.nrm_d, seq, blk.mlp1);
    // `nn.GELU()` with the default `approximate='none'` — the erf form, not tanh.
    try ctx.opElt(.gelu_erf, ws.mlp_d, null, null, null, .{
        .u0 = @intCast(seq * cfg.mlp_dim),
    }, seq * cfg.mlp_dim, 1, 1);
    // ⚠️ Its own prep: the reduction width here is `mlp_dim`, not `dim`.
    try prepGroup(ctx, ws.mlp_d, seq, cfg.mlp_dim, &.{blk.mlp2});
    try lin(ctx, ws.dlt_d, ws.mlp_d, seq, blk.mlp2);
    try ctx.opElt(.gated_add, ws.x_d, ws.dlt_d, ws.mod_d, null, .{
        .u0 = total,
        .u1 = dim32,
        .u2 = mod_base + 8 * dim32,
    }, seq * d, 1, 1);
}

/// Where a block's cross/self keys and values live, and in which form.
const Kv = struct {
    k: Buf,
    v: Buf,
    kv_seq: usize,
    kv_pad: usize,
    /// True when `k`/`v` are ALREADY the f16 attention operands (the per-image cross
    /// cache); false when they are f32 activations that still need converting.
    prepared: bool,
};

/// Attention from `ws.q_d` (normed, roped, f32) against `kv`, into `ws.attn_d`.
///
/// Both paths are non-causal and compute the same thing; they differ in how the
/// scores are produced. ⚠️ **The flash path is a requirement at production
/// resolutions, not an optimization** — the same statement `zimage_gpu` makes about
/// `attn_full`: a thread-per-(query, head) kernel at ~6800 tokens runs long enough to
/// trip the GPU watchdog (`error_device_lost`, not a slow render), and for
/// cross-attention it would additionally stream a 512-key context per thread.
fn attention(ctx: *gpu.Context, sess: *Session, cfg: anima.Config, ws: *Workspace, kv: Kv, scale: f32) !void {
    const d = cfg.dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const seq = sess.seq;
    const q_pad = sess.seq_pad;

    if (!sess.tc) {
        std.debug.assert(!kv.prepared);
        return ctx.opElt(.attn_cross, ws.q_d, kv.k, kv.v, ws.attn_d, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(heads),
            .u2 = @intCast(hd),
            .u3 = @intCast(kv.kv_seq),
        }, seq * heads, 1, 1);
    }

    // Q always converts: it is a fresh activation every block. K/V convert only for
    // self-attention — the cross cache already holds them in this exact layout.
    try ctx.opElt(.f32_to_h16, ws.q_d, null, null, ws.qt_d, .{
        .u0 = @intCast(q_pad * d / 2),
        .u1 = @intCast(seq * d),
        .f0 = scale,
    }, q_pad * d / 2, 1, 1);
    var k16 = kv.k;
    var v16 = kv.v;
    if (!kv.prepared) {
        ctx.independent(2);
        try ctx.opElt(.gather_kmajor_h16, kv.k, null, null, ws.kt_d, .{
            .u0 = @intCast(d * kv.kv_pad / 2),
            .u1 = @intCast(hd),
            .u2 = @intCast(kv.kv_pad),
            .u3 = @intCast(kv.kv_seq),
            .u4 = @intCast(heads),
        }, d * kv.kv_pad / 2, 1, 1);
        try ctx.opElt(.f32_to_h16, kv.v, null, null, ws.v16_d, .{
            .u0 = @intCast(kv.kv_pad * d / 2),
            .u1 = @intCast(kv.kv_seq * d),
            .f0 = 1.0,
        }, kv.kv_pad * d / 2, 1, 1);
        k16 = ws.kt_d;
        v16 = ws.v16_d;
    }

    const md_off = q_pad * d;
    const push = gpu.EltPush{
        .u0 = @intCast(heads * hd),
        .u1 = @intCast(kv.kv_pad),
        .u2 = 0,
        .u3 = 1, // no GQA: n_heads == n_kv_heads
        .u4 = @intCast(heads * hd),
        .u5 = @intCast(md_off),
        .f0 = @bitCast(@as(u32, @intCast(kv.kv_seq))),
        // ⚠️ The MD table is indexed by QUERY row, so its plane stride is `q_pad` —
        // NOT `u1`, which is the key padded length. 0 would mean "= u1" and is only
        // right for square self-attention; passing it explicitly covers both.
        .f1 = @bitCast(@as(u32, @intCast(q_pad))),
    };
    try ctx.opFlash(.md, ws.qt_d, k16, v16, ws.attn_d, push, q_pad / 128, heads);
    try ctx.opFlash(.out, ws.qt_d, k16, v16, ws.attn_d, push, q_pad / 128, heads);
}

/// Fused weightless LayerNorm + AdaLN modulation, `x_d -> nrm_d`.
///
/// ⚠️ `premul_off` points at the table's SCALE block, which `foldModulationTable`
/// has already turned into `1 + scale`. The two-kernel alternative
/// (`layernorm` + `modulate`) exists on this backend but would read and write the
/// whole activation twice, and its `layernorm` is one thread per row.
fn lnMod(ctx: *gpu.Context, ws: *Workspace, rows: usize, dim: usize, premul_off: u32, shift_off: u32, eps: f32) !void {
    // `supported()` refuses a device without this kernel, so there is no arm to fall back
    // to and no un-folding to attempt.
    std.debug.assert(ctx.hasLnModSg());
    return ctx.opLnModSg(ws.x_d, ws.nrm_d, ws.mod_d, rows, dim, premul_off, shift_off, eps);
}

/// Per-head RMSNorm over `[rows][hd]`, in place.
///
/// ⚠️ **The SUBGROUP kernel, not `Elt.rmsnorm`** — measured **173 -> 358 GB/s** at Anima's
/// geometry (`anima-vk-bench`, seq 1536, 16 heads of 128). `rmsnorm` gives each thread a
/// whole 128-wide head, so a warp's 32 loads land 512 B apart and every one is its own
/// sector fetch; `rmsnorm_sg` gives each head to a subgroup, so the 32 lanes read one
/// contiguous line. Exactly the mistake `qk_rmsnorm_warp` fixed on CUDA, where it was worth
/// 17x — and the subgroup kernel had been sitting here unused. Three calls per block.
fn qkNorm(ctx: *gpu.Context, x: Buf, weights: []const f32, rows: usize, hd: usize, eps: f32) !void {
    const w = try normBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, x, w, rows, hd, eps);
    try ctx.opElt(.rmsnorm, x, x, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(hd),
        .f0 = eps,
    }, rows, 1, 1);
}

fn ropeHalf(ctx: *gpu.Context, x: Buf, freqs: Buf, rows: usize, heads: usize, half: usize) !void {
    const total = rows * heads * half;
    try ctx.opElt(.rope_half, x, null, freqs, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(half),
        .u2 = @intCast(rows * half), // sin_off
        .u3 = @intCast(heads),
    }, total, 1, 1);
}

/// A block GEMM, dispatched by weight dtype — the same routing `zimage_gpu.gemm`
/// uses. Anima ships dense bf16, which takes one of the two f16-weight tensor-core
/// pipelines (native bf16 where the device has a bf16 coop config, else bf16->f16 at
/// upload; both keep the conversion off the host under weight streaming).
fn gemm(ctx: *gpu.Context, y: Buf, x: Buf, m: usize, w: Weight) !void {
    // Every Anima block linear is bias-free, but the f16-weight coop GEMM always folds
    // one in, so hand it the FULL zero vector — see `zero_bias`.
    const zeros: []const f32 = &zero_bias;
    std.debug.assert(w.rows <= zeros.len);
    switch (w.dtype) {
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                try ctx.opMatmulCoopBf16(y, 0, x, m, w.bytes, w.rows, w.cols, zeros);
            } else {
                try ctx.opMatmulCoopF16Wb(y, 0, x, m, w.bytes, w.rows, w.cols, zeros);
            }
        },
        .f16 => try ctx.opMatmulCoopF16Wh(y, 0, x, m, w.bytes, w.rows, w.cols, zeros),
        .f32, .f8_e4m3 => try ctx.opMatmul(y, 0, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, null),
        // `anima.gpuLinKindSupported` gates this before a session is built, so
        // reaching here is a programming error rather than a bad checkpoint.
        else => return error.UnsupportedDType,
    }
}

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
fn normBuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    const h = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .mem = .null_handle, .size = 0 };
}

// --- tests ------------------------------------------------------------------
//
// Two tiers. The first needs no device and pins the modulation-table fold, which is a
// pure convention that yields a plausible image when muddled — and did: folding the
// FINAL layer's scale as well as the blocks' cost a rel L2 of 0.10 against the CPU
// forward, identical on both attention paths. The device tests below it check the
// rectangular attention against `ops.attention` and then the whole forward against
// `anima.DiT.predict` on real weights.

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const safetensors = @import("tp_core").safetensors;

const anima_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors";
/// The same model, quantized to ComfyUI's packed `asym_w4a8_int8`. ⚠️ Block 0 is left
/// dense and block 1 onward is quantized (every real "mixed" checkpoint does this), so a
/// 2-block model is the shallowest one that exercises a quantized block at all.
const anima_w4a8_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/anima/terraRising_20TerraRisingAnima-ASYM_W4A8_INT8.safetensors";

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const dv = @as(f64, e) - @as(f64, a);
        num += dv * dv;
        den += @as(f64, e) * e;
    }
    return if (den > 0) @sqrt(num / den) else @sqrt(num);
}

test "the modulation fold touches every block scale and NOT the final layer's" {
    // ⚠️ The asymmetry is the whole content of this test. The device's fused norm
    // computes `(x-mean)*inv*premul + shift` with no place for the `1 +`, so every
    // BLOCK scale arrives pre-folded — but the final layer runs on the HOST through
    // `DiT.finalize`, whose `modulatedNorm` adds its own 1. Folding it there too was a
    // real bug (rel L2 0.10 vs the CPU forward, identical on both attention paths,
    // which is what said "shared wiring, not attention"). Nothing about it is loud:
    // both tables are finite and the same shape.
    const gpa = testing.allocator;
    var cfg = anima.anima_2b;
    cfg.n_layers = 3;
    const d = cfg.dim;
    const tbl = try gpa.alloc(f32, anima.modulationTableLen(cfg));
    defer gpa.free(tbl);
    @memset(tbl, 0);
    const before = try gpa.dupe(f32, tbl);
    defer gpa.free(before);

    anima.foldModulationTable(cfg, tbl);
    for (0..cfg.n_layers) |bi| {
        const blk = tbl[bi * 9 * d ..][0 .. 9 * d];
        for (0..3) |si| {
            errdefer std.debug.print("block {d} sublayer {d}\n", .{ bi, si });
            // shift and gate untouched, scale +1.
            try testing.expectEqual(@as(f32, 0), blk[si * 3 * d]);
            try testing.expectEqual(@as(f32, 1), blk[si * 3 * d + d]);
            try testing.expectEqual(@as(f32, 0), blk[si * 3 * d + 2 * d]);
        }
    }
    // The final layer's [shift, scale] pair is byte-for-byte what it was.
    const fin = cfg.n_layers * 9 * d;
    try testing.expectEqualSlices(f32, before[fin..], tbl[fin..]);
}

fn gpuCtx(gpa: std.mem.Allocator, io: std.Io) !*gpu.Context {
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return gpu.Context.init(gpa) catch error.SkipZigTest;
}

test "Anima gpu cross-attention matches ops.attention at unequal q/kv lengths" {
    // The rectangular flash path, at Anima's real cross-attention shape (16 heads of
    // 128 against a 512-row context) plus a key length that is NOT a multiple of 128,
    // which is what exercises the padded-key masking. ⚠️ This shape is new to the
    // kernel — `s_stride` used to double as the MD plane stride, so every head past the
    // first read another head's {max, 1/sum} once q_pad exceeded kv_pad.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();
    if (ctx.pipe_flash_md == .null_handle) return error.SkipZigTest;

    const cfg = anima.anima_2b;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const dim = heads * hd;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    var prng = std.Random.DefaultPrng.init(0x5ee);
    const rnd = prng.random();
    for ([_][2]usize{ .{ 300, 512 }, .{ 260, 200 } }) |sh| {
        const nq = sh[0];
        const nkv = sh[1];
        const q_pad = std.mem.alignForward(usize, nq, 128);
        const kv_pad = std.mem.alignForward(usize, nkv, 128);

        const q = try gpa.alloc(f32, nq * dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, nkv * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, nkv * dim);
        defer gpa.free(v);
        for ([_][]f32{ q, k, v }) |buf| for (buf) |*x| {
            x.* = rnd.floatNorm(f32);
        };
        const want = try gpa.alloc(f32, nq * dim);
        defer gpa.free(want);
        try ops.attention.attention(io, gpa, want, q, k, v, .{
            .seq_q = nq,
            .seq_kv = nkv,
            .n_heads = heads,
            .n_kv_heads = heads,
            .head_dim = hd,
        });

        // Drive the same buffers `attention()` builds: Q f16 with the scale prefolded,
        // K per-head k-major f16, V f16 — here from the host so the test exercises the
        // kernel rather than the session's cache builder.
        var q16 = try ctx.tensorCreate(q_pad * dim * 2);
        defer ctx.tensorDestroy(&q16);
        var k16 = try ctx.tensorCreate(dim * kv_pad * 2);
        defer ctx.tensorDestroy(&k16);
        var v16 = try ctx.tensorCreate(kv_pad * dim * 2);
        defer ctx.tensorDestroy(&v16);
        var o_d = try ctx.tensorCreate((q_pad * dim + heads * q_pad * 2) * 4);
        defer ctx.tensorDestroy(&o_d);
        {
            const qb = try gpa.alloc(u16, q_pad * dim);
            defer gpa.free(qb);
            @memset(qb, 0);
            for (0..nq * dim) |i| qb[i] = f16Bits(q[i] * scale);
            try ctx.tensorUpload(q16, std.mem.sliceAsBytes(qb));
            const kb = try gpa.alloc(u16, dim * kv_pad);
            defer gpa.free(kb);
            @memset(kb, 0);
            for (0..nkv) |p| for (0..heads) |hh| for (0..hd) |cc| {
                kb[(hh * hd + cc) * kv_pad + p] = f16Bits(k[p * dim + hh * hd + cc]);
            };
            try ctx.tensorUpload(k16, std.mem.sliceAsBytes(kb));
            const vb = try gpa.alloc(u16, kv_pad * dim);
            defer gpa.free(vb);
            @memset(vb, 0);
            for (0..nkv * dim) |i| vb[i] = f16Bits(v[i]);
            try ctx.tensorUpload(v16, std.mem.sliceAsBytes(vb));
        }

        const push = gpu.EltPush{
            .u0 = @intCast(dim),
            .u1 = @intCast(kv_pad),
            .u2 = 0,
            .u3 = 1,
            .u4 = @intCast(dim),
            .u5 = @intCast(q_pad * dim),
            .f0 = @bitCast(@as(u32, @intCast(nkv))),
            .f1 = @bitCast(@as(u32, @intCast(q_pad))),
        };
        try ctx.beginBatch();
        try ctx.opFlash(.md, q16, k16, v16, o_d, push, q_pad / 128, heads);
        try ctx.opFlash(.out, q16, k16, v16, o_d, push, q_pad / 128, heads);
        try ctx.endBatch();

        const got = try gpa.alloc(f32, q_pad * dim);
        defer gpa.free(got);
        try ctx.tensorDownload(gpu.DeviceBuffer{ .buf = o_d.buf, .mem = .null_handle, .size = got.len * 4 }, std.mem.sliceAsBytes(got));
        const rel = relL2(want, got[0 .. nq * dim]);
        errdefer std.debug.print("cross attn {d}q x {d}kv rel L2 {e:.4}\n", .{ nq, nkv, rel });
        // f16 operands against the CPU's f32 accumulation.
        try testing.expect(rel < 5e-3);
    }
}

test "the Vulkan int4 decode matches the CPU nibble unpack, including the row padding" {
    // ⚠️ **This kernel replaced a load-time widening, so what it must reproduce EXACTLY is
    // that widening** — otherwise the change is a silent quality regression rather than a
    // VRAM win. The reference here is therefore the same sign-extending unpack
    // `anima_gpu.widen` used to do (and that `ops.matmul`'s i4 path does), laid out k-major
    // with zeroed row padding, which is what the GEMM reads.
    //
    // ⚠️ The PADDING is not incidental: `opI8GemmBuf` reads whole 64-row tiles and this
    // scratch is shared between weights of different shapes, so an unwritten pad slot would
    // feed the next GEMM the previous weight's values.
    const gpa = testing.allocator;
    const io = testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();
    if (!ctx.hasI4Decode()) return error.SkipZigTest;

    // One height that needs no row padding and two that do.
    const cases = [_]struct { rows: usize, cols: usize }{
        .{ .rows = 256, .cols = 512 },
        .{ .rows = 2048, .cols = 1024 }, // Anima's cross-attention k/v shape
        .{ .rows = 132, .cols = 256 },
    };
    // ⚠️ Every case's buffers stay ALIVE in one arena: the device weight cache keys on the
    // HOST POINTER, so freeing one case's array and letting the allocator reuse the address
    // scores a stale cache hit and decodes the PREVIOUS case's weights. That is a real
    // failure the W4A8 kernel's test hit first, and it looks like a shape bug.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    for (cases) |c| {
        const stride = gpu.Context.i4ScratchBytes(c.rows, c.cols) / c.cols;
        var prng = std.Random.DefaultPrng.init(0x14 + c.rows);
        const r = prng.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);

        const want = try alloc.alloc(i8, stride * c.cols);
        @memset(want, 0);
        for (packed_bytes, 0..) |byte, i| {
            // ⚠️ LOW nibble = EVEN element, sign-extended — the packing `ops.matmul`'s i4
            // path and ComfyUI's W4A4 converter both use. Swapping the halves is
            // rms-preserving, so only a positional check like this one can see it.
            const row = i / (c.cols / 2);
            const kk = (i % (c.cols / 2)) * 2;
            want[kk * stride + row] = @as(i4, @bitCast(@as(u4, @truncate(byte))));
            want[(kk + 1) * stride + row] = @as(i4, @bitCast(@as(u4, @truncate(byte >> 4))));
        }

        _ = try ctx.i4Decode(packed_bytes, c.rows, c.cols);
        const got = try alloc.alloc(i8, stride * c.cols);
        try ctx.tensorDownload(ctx.i4_t, std.mem.sliceAsBytes(got));

        for (want, got, 0..) |wv, gv, i| {
            testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("rows {d} cols {d} stride {d}: element {d} (k {d}, row {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, stride, i, i / stride, i % stride, gv, wv,
                });
                return e;
            };
        }
    }
}

test "Anima gpu forward matches the CPU forward on a real checkpoint" {
    // The definitive one: same weights, same latent, same conditioning, both forwards.
    try forwardParity(anima_ckpt, 2e-3);
}

test "Anima gpu forward matches the CPU forward on a real W4A8 checkpoint" {
    // ⚠️ **A LOOSER bound, and it is not the dense one relaxed until it passed.** A
    // quantized checkpoint is a different computation on the two sides, not a different
    // rounding of one: the CPU `matmul` decodes the packed weight and multiplies in f32
    // (weight-only), while the device additionally quantizes the ACTIVATION to int8 —
    // W4A8's "A8" — so the residual includes an error the reference never incurs.
    //
    // The receipt that this is quantization and not wiring is that the SAME model
    // quantized int8 measures the same figure: 2.0721e-3 (int8) against 2.0968e-3 (w4a8)
    // through `anima-cuda-test`, a 1.2% gap, with both attention arms agreeing to three
    // digits. W4A8 decodes TO int8-convrot, so landing anywhere else would be the bug.
    try forwardParity(anima_w4a8_ckpt, 6e-3);
}

/// One checkpoint's Vulkan forward against `anima.DiT.predict` on the same weights.
///
/// Truncated to 2 trunk blocks so it loads in seconds — the loop bound is not what a
/// kernel port gets wrong, the block's shape is, and the end-to-end render covers the
/// depth. Two blocks is also the minimum that reaches a quantized one, since every real
/// mixed checkpoint leaves block 0 dense.
fn forwardParity(path: []const u8, tol: f64) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, path);
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var ck = try safetensors.SafeTensors.open(gpa, io, path);
    defer ck.deinit();
    var cfg = anima.anima_2b;
    cfg.n_layers = 2;
    var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();
    if (!supported(ctx, &model)) return error.SkipZigTest;

    const lat_h = 24;
    const lat_w = 32;
    const ctx_seq = cfg.adapter.min_rows; // 512, what a real prompt gives
    var prng = std.Random.DefaultPrng.init(0xa71);
    const rnd = prng.random();
    const cond = try gpa.alloc(f32, ctx_seq * cfg.context_dim);
    defer gpa.free(cond);
    for (cond) |*v| v.* = rnd.floatNorm(f32) * 0.5;
    const x_lat = try gpa.alloc(f32, cfg.channels * lat_h * lat_w);
    defer gpa.free(x_lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);
    const sigma: f32 = 0.75;

    const want = try gpa.alloc(f32, x_lat.len);
    defer gpa.free(want);
    {
        const mod = try model.modulationTable(io, gpa, sigma);
        defer gpa.free(mod);
        try model.predict(io, gpa, want, x_lat, lat_h, lat_w, cond, ctx_seq, mod, null);
    }

    // ⚠️ **The flash path only**, unlike Z-Image's equivalent test. `Session` caches
    // cross-attention's K/V in whichever form the chosen path reads — f16 attention
    // operands or plain f32 — so the two arms cannot share a session, and the flash one
    // is what actually runs at every resolution. `anima-cuda-test` is where both
    // attention arms are compared against the same CPU forward.
    const sched = try model.modulationSchedule(io, gpa, &.{ sigma, 0 });
    defer gpa.free(sched);
    var sess = try Session.init(gpa, io, ctx, &model, lat_h, lat_w, cond, ctx_seq, &.{ sigma, 0 }, sched);
    defer sess.deinit(gpa, ctx);
    if (!sess.tc) return error.SkipZigTest;
    var ws = try Workspace.init(ctx, &model, lat_h, lat_w, sess.tc);
    defer ws.deinit(ctx);

    const got = try gpa.alloc(f32, x_lat.len);
    defer gpa.free(got);
    try forward(&model, ctx, &sess, &ws, io, gpa, got, x_lat, sigma, null);

    const rel = relL2(want, got);
    errdefer std.debug.print("anima gpu forward on {s}: rel L2 {e:.4} (tol {e:.1})\n", .{ path, rel, tol });
    // bf16 GEMMs and f16 attention against the CPU's f32 accumulation — the regime
    // `zimage_gpu` and `sd_unet_cuda` already sit in.
    try testing.expect(rel < tol);
}
