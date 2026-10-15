//! GPU-resident Z-Image (`NextDiT`) forward on the CUDA backends (`zig-cuda`'s hand-PTX
//! and `cuda`'s vendor libraries), the CUDA twin of `zimage_gpu`.
//!
//! The host/device split is identical to the Vulkan arm's; see `zimage_gpu`'s header.
//! Two differences, both forced by the op surface:
//!
//! 1. The modulation table is the FOLDED one (`zimage.modulationTableFolded`). CUDA has
//!    no standalone `modulate`, only `rmsMod` (`out = x*inv*premul[col] + shift[col]`),
//!    so the pre-norm weight must arrive already multiplied into the scale. Vulkan takes
//!    the unfused table because it has a separate `modulate`. Both come out of one AdaLN
//!    evaluation, so they cannot drift.
//! 2. Attention is `opAttnTC`: cuDNN's fused SDPA under `cuda`, hand-PTX tensor cores
//!    under `zig-cuda`. The naive `be.attn` is one thread per (query, head) and trips the
//!    GPU watchdog at production resolutions, so it is the fallback, not the default.
//!
//! Head width is exactly 128, which both `launchHgemmB`'s P@V tiling and cuDNN handle
//! directly, so unlike the SD family there is no head padding and no `head_pad` /
//! `head_unpad` round trip. Every trunk GEMM width (3840 / 10240 / 11520) is a multiple
//! of 128 and every reduction width a multiple of 32.

const std = @import("std");
const zimage = @import("zimage.zig");
const zimage_dev = @import("zimage_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = zimage.DiT;
const File = @This();
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query,head) attention instead of `opAttnTC`. For
/// A/B and for reproducing a mismatch; the device test runs both.
pub var force_naive_attn: bool = false;

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: zimage.Config,
    lat_h: usize,
    lat_w: usize,
    cap_padded: usize,
    n_img: usize,
    img_padded: usize,
    /// `cap_padded + img_padded`, the joint sequence the trunk runs on.
    seq: usize,
    /// What `lin_cuda.plan` decided for this checkpoint's block linears.
    plan: lin_cuda.Plan,

    cap_d: Buf,
    x_pad_d: Buf,
    /// Interleaved-RoPE table for the joint sequence, `cos` then `sin`.
    freqs_d: Buf,
    /// The image half's own table, a separate upload, because the `noise_refiner`
    /// runs on the image tokens alone and needs row 0 of its table to be the image's
    /// first position, not the caption's.
    img_freqs_d: Buf,

    sigmas: []f32,
    /// `[sigmas.len][modulatedBlocks * 4 * dim + dim]`, FOLDED (see the header).
    mods: []f32,
    finals: []f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cap: []const f32,
        cap_padded: usize,
        sigmas: []const f32,
    ) !Session {
        const cfg = model.cfg;
        std.debug.assert(cap.len == cap_padded * cfg.dim);
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const n_img = h * w;
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded + img_padded;
        const half = cfg.head_dim / 2;

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .cap_padded = cap_padded,
            .n_img = n_img,
            .img_padded = img_padded,
            .seq = seq,
            .plan = blk: {
                var p = try lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "zimage cuda");
                // The MLP down projection reads ~1e5, past f16.
                p.act16 = .bf16;
                break :blk p;
            },
            .cap_d = undefined,
            .x_pad_d = undefined,
            .freqs_d = undefined,
            .img_freqs_d = undefined,
            .sigmas = &.{},
            .mods = &.{},
            .finals = &.{},
            .mod_stride = model.modulatedBlocks() * 4 * cfg.dim + cfg.dim,
        };
        var made: usize = 0;
        errdefer {
            const bufs = [_]*Buf{ &self.cap_d, &self.x_pad_d, &self.freqs_d, &self.img_freqs_d };
            for (bufs[0..made]) |b| be.tensorDestroy(b);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.mods.len != 0) gpa.free(self.mods);
            if (self.finals.len != 0) gpa.free(self.finals);
        }

        try lin_cuda.presize(be, self.plan, model.device_lins);
        // The patch embed runs on the fused `opMatmul`.
        try lin_cuda.attachLora(be, &self.plan, model.lora, model.device_lins, seq);
        self.cap_d = try be.tensorCreate(cap.len * 4);
        made += 1;
        try be.tensorUpload(self.cap_d, std.mem.sliceAsBytes(cap));
        self.x_pad_d = try be.tensorCreate(cfg.dim * 4);
        made += 1;
        try be.tensorUpload(self.x_pad_d, std.mem.sliceAsBytes(model.x_pad_token));

        {
            var joint = try model.ropeFreqs(gpa, cap_padded, h, w);
            defer joint.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], joint.cos);
            @memcpy(flat[seq * half ..], joint.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));

            const iflat = try gpa.alloc(f32, 2 * img_padded * half);
            defer gpa.free(iflat);
            @memcpy(iflat[0 .. img_padded * half], joint.cos[cap_padded * half ..]);
            @memcpy(iflat[img_padded * half ..], joint.sin[cap_padded * half ..]);
            self.img_freqs_d = try be.tensorCreate(iflat.len * 4);
            made += 1;
            try be.tensorUpload(self.img_freqs_d, std.mem.sliceAsBytes(iflat));
        }

        self.sigmas = try gpa.dupe(f32, sigmas);
        self.mods = try gpa.alloc(f32, sigmas.len * self.mod_stride);
        self.finals = try gpa.alloc(f32, sigmas.len * cfg.dim);
        for (sigmas, 0..) |s, i| {
            const adaln = try model.adalnInput(io, gpa, s);
            defer gpa.free(adaln);
            const tbl = try model.modulationTableFolded(io, gpa, adaln);
            defer gpa.free(tbl);
            @memcpy(self.mods[i * self.mod_stride ..][0..self.mod_stride], tbl);
            const fin = try model.finalScale(io, gpa, adaln);
            defer gpa.free(fin);
            @memcpy(self.finals[i * cfg.dim ..][0..cfg.dim], fin);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        be.tensorDestroy(&self.cap_d);
        be.tensorDestroy(&self.x_pad_d);
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.img_freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.mods);
        gpa.free(self.finals);
        self.* = undefined;
    }

    fn tv(self: *const Session, sigma: f32) ?StepMod {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return .{
                .mods = self.mods[i * self.mod_stride ..][0..self.mod_stride],
                .final = self.finals[i * self.cfg.dim ..][0..self.cfg.dim],
            };
        }
        return null;
    }
};

const StepMod = struct { mods: []const f32, final: []const f32 };

pub const Workspace = struct {
    x_d: Buf,
    img_d: Buf,
    nrm_d: Buf,
    dlt_d: Buf,
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    mg_d: Buf,
    mu_d: Buf,
    mv_d: Buf,
    imgin_d: Buf,
    /// The storage of the residual (`x_d`, `img_d`), the norm output, the sublayer
    /// delta and the MLP intermediates. The attention block stays f32: V reaches ~1e5 here
    /// and is prescaled on the way into the attention's own f16 cast (`zimage_dev.v_div`).
    act: Dt,

    const fields = [_][]const u8{ "x_d", "img_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mu_d", "mv_d", "imgin_d" };

    pub fn init(be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cap_padded_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const n_img = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        const img_padded = cfg.padded(n_img);
        // Rows, padded to 128 for the buffers a GEMM writes. Z-Image's own
        // `pad_multiple` is 32, but a quantized route's GEMM launches over
        // `align(m, 128)` rows and each block stores a whole tile, so a buffer sized to
        // the model's padding is written off the end (`CUDA_ERROR_ILLEGAL_ADDRESS`, or a
        // solid white render where the overrun lands in another workspace buffer).
        const seq = cap_padded_cap + img_padded;
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const img_pad128 = std.mem.alignForward(usize, img_padded, 128);
        const sizes = [fields.len]usize{
            seq_pad * cfg.dim * 4,
            img_pad128 * cfg.dim * 4,
            seq_pad * cfg.dim * 4,
            seq_pad * cfg.dim * 4,
            seq_pad * cfg.qDim() * 4,
            seq_pad * cfg.kvDim() * 4,
            seq_pad * cfg.kvDim() * 4,
            seq_pad * cfg.qDim() * 4,
            seq_pad * cfg.mlp_dim * 4,
            seq_pad * cfg.mlp_dim * 4,
            (model.modulatedBlocks() * 4 * cfg.dim + cfg.dim) * 4,
            n_img * cfg.patchDim() * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        self.act = act;
        inline for (.{ "x_d", "img_d", "nrm_d", "dlt_d", "mg_d", "mu_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether the CUDA arms can run every block linear this model has. The refusal, if
/// any, is logged by name; the caller says the trunk then runs on the CPU.
pub fn supported(model: *const DiT) bool {
    if (model.layers.len == 0) return false;
    _ = lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "zimage cuda") catch return false;
    if (!lin_cuda.loraRunnable(model.lora, model.device_lins, "zimage cuda")) return false;
    return true;
}

/// DIAGNOSTIC: report a device buffer's magnitude and non-finite count (`TP_ZIMAGE_TRACE`).
///
/// A forward that comes back all-NaN says nothing about where it turned, and the stage
/// boundaries are the only places a host-side check can look without a kernel per op.
/// Costs a sync and a download per call, so it is env-gated and never on in a render.
var trace_on: ?bool = null;

fn trace(be: *Backend, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_ZIMAGE_TRACE") != null;
    if (!trace_on.?) return;
    const host = be.gpa.alloc(f32, elems) catch return;
    defer be.gpa.free(host);
    const batching = be.batching();
    if (batching) be.endBatch() catch return;
    be.downloadF32(buf, host) catch return;
    var mx: f32 = 0;
    var bad: usize = 0;
    for (host) |v| {
        if (!std.math.isFinite(v)) bad += 1 else mx = @max(mx, @abs(v));
    }
    std.debug.print("[zimage-cuda] {s} {d:>2}  max|x| {d:12.1}  non-finite {d}/{d}\n", .{ what, i, mx, bad, elems });
    if (batching) be.beginBatch() catch return;
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    be: *Backend,
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
    const dim = cfg.dim;
    std.debug.assert(x_lat.len == cfg.channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    // The norm weight is folded into the table: `rmsMod` has no place for one.
    var owned_mods: ?[]f32 = null;
    var owned_final: ?[]f32 = null;
    defer if (owned_mods) |b| gpa.free(b);
    defer if (owned_final) |b| gpa.free(b);
    const step = sess.tv(sigma) orelse blk: {
        const adaln = try model.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        owned_mods = try model.modulationTableFolded(io, gpa, adaln);
        owned_final = try model.finalScale(io, gpa, adaln);
        break :blk StepMod{ .mods = owned_mods.?, .final = owned_final.? };
    };
    const patches = try zimage.patchify(gpa, cfg, x_lat, sess.lat_h, sess.lat_w);
    defer gpa.free(patches);
    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.mods));
    try be.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(patches));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    const d: Dev = .{
        .model = model,
        .be = be,
        .sess = sess,
        .ws = ws,
        .geom = .{ .n_img = sess.n_img, .img_padded = sess.img_padded, .seq = sess.seq },
        .zero_off = model.zeroShiftOffset(),
    };
    try zimage_dev.body(d, model, cancel);
    try be.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, sess.seq * dim)) return error.NonFiniteActivation;

    const img_rows = try gpa.alloc(f32, sess.n_img * dim);
    defer gpa.free(img_rows);
    const ew = ws.act.bytes();
    try be.downloadF32(ws.x_d.slice(sess.cap_padded * dim * ew, img_rows.len * ew), img_rows);
    const final = try gpa.alloc(f32, sess.n_img * cfg.patchDim());
    defer gpa.free(final);
    try model.finalLayer(io, gpa, final, img_rows, step.final, sess.n_img);
    zimage.unpatchify(cfg, out, final, sess.lat_h, sess.lat_w);
}

/// `zimage_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    geom: struct { n_img: usize, img_padded: usize, seq: usize },
    /// The modulation table's trailing zero block: Z-Image's modulation has no shift.
    zero_off: usize,

    /// `x_embedder` (the one weight the loader materializes to f32, so it takes the f32
    /// arm), then the learned pad token broadcast into the tail rows.
    pub fn embed(d: Dev) !void {
        const dim = d.model.cfg.dim;
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.img_d, d.ws.imgin_d, d.sess.n_img, d.model.x_embedder.w, d.model.x_embedder.b);
        for (d.sess.n_img..d.sess.img_padded) |r| {
            if (d.ws.act == .f32)
                try d.be.tensorCopy(d.ws.img_d, r * dim * 4, d.sess.x_pad_d, 0, dim * 4)
            else
                try d.be.opCast(d.sess.x_pad_d, d.ws.img_d, dim, r * dim);
        }
    }

    /// Prefetch one block ahead throughout, so each upload overlaps the previous block's
    /// compute; the two stacks are consecutive, so the last refiner primes `layers[0]`.
    pub fn blockBegin(d: Dev, stage: zimage_dev.Stage, i: usize) !void {
        if (!d.be.async_uploads) return;
        const m = d.model;
        switch (stage) {
            .image => {
                if (i == 0) prefetchBlock(d.be, m.noise_refiner[0]);
                if (i + 1 < m.noise_refiner.len) prefetchBlock(d.be, m.noise_refiner[i + 1]) else if (m.layers.len > 0) prefetchBlock(d.be, m.layers[0]);
            },
            .joint => if (i + 1 < m.layers.len) prefetchBlock(d.be, m.layers[i + 1]),
        }
    }

    /// The caption rows, then the image rows, into the joint sequence.
    pub fn join(d: Dev) !void {
        const dim = d.model.cfg.dim;
        const ew = d.ws.act.bytes();
        if (d.ws.act == .f32)
            try d.be.tensorCopy(d.ws.x_d, 0, d.sess.cap_d, 0, d.sess.cap_padded * dim * 4)
        else
            try d.be.opCast(d.sess.cap_d, d.ws.x_d, d.sess.cap_padded * dim, 0);
        try d.be.tensorCopy(d.ws.x_d, d.sess.cap_padded * dim * ew, d.ws.img_d, 0, d.sess.img_padded * dim * ew);
    }

    pub fn trace(d: Dev, what: []const u8, i: usize, buf: Buf, elems: usize) void {
        File.trace(d.be, what, i, buf, elems);
    }

    /// One launch: the norm weight is folded into the table's `premul` slot.
    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, rows: usize, off: usize, _: []const f32) !void {
        const cfg = d.model.cfg;
        try d.be.rmsMod(x, nrm, d.ws.mv_d, rows, cfg.dim, off, d.zero_off, cfg.norm_eps);
    }

    pub fn prep(d: Dev, x: Buf, m: usize, cols: usize, group: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.be, d.sess.plan, x, m, cols, group);
    }

    pub fn gemms(d: Dev, g: lin_cuda.Group, ys: []const Buf, group: []const Weight) !void {
        for (ys, group) |y, w| try lin_cuda.gemm(d.be, d.sess.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try lin_cuda.linear(d.be, d.sess.plan, y, x, m, w, null);
    }

    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, rows: usize, stage: zimage_dev.Stage) !void {
        const cfg = d.model.cfg;
        const be = d.be;
        const half = cfg.head_dim / 2;
        const freqs = if (stage == .image) d.sess.img_freqs_d else d.sess.freqs_d;
        const sin_off = (if (stage == .image) d.sess.img_padded else d.sess.seq) * half;
        try be.qkNorm(d.ws.q_d, d.ws.q_d, try normBuf(be, qnorm), rows * cfg.n_heads, cfg.head_dim, cfg.qk_eps);
        try be.qkNorm(d.ws.k_d, d.ws.k_d, try normBuf(be, knorm), rows * cfg.n_kv_heads, cfg.head_dim, cfg.qk_eps);
        try be.rope(d.ws.q_d, freqs, rows, cfg.n_heads, half, sin_off);
        try be.rope(d.ws.k_d, freqs, rows, cfg.n_kv_heads, half, sin_off);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    pub fn attention(d: Dev, rows: usize) !void {
        const cfg = d.model.cfg;
        const ws = d.ws;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        if (force_naive_attn) return d.be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, rows, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, sc, false);
        try d.be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, sc);
    }

    /// A plain weighted RMSNorm over rows of `dim` (`qkNorm` takes any row width).
    pub fn rowNorm(d: Dev, buf: Buf, w: []const f32, rows: usize) !void {
        const cfg = d.model.cfg;
        try d.be.qkNorm(buf, buf, try normBuf(d.be, w), rows, cfg.dim, cfg.norm_eps);
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, rows: usize, off: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(x, t, d.ws.mv_d, rows * dim, dim, off);
    }

    pub fn siluMul(d: Dev, mg: Buf, mu: Buf, n: usize) !void {
        try d.be.siluMul(mg, mu, n);
    }
};

/// Queue a block's streamable weights for async prefetch, called ONE BLOCK AHEAD so
/// the upload overlaps the previous block's compute. Keys must be the same host
/// pointers `forward` later fetches, or the prefetch is a cache miss and pure waste,
/// hence q/k/v as the loader's row views, the same Weights the forward runs.
///
/// Without this the first step pays the whole ~11.6 GB upload serially: measured
/// 8.0 s for step 1 against a 2.6 s steady state at 1056x1584, which on a 9-step
/// turbo render is a fifth of the total time.
fn prefetchBlock(be: *Backend, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    inline for (.{ blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.out }) |w| be.prefetchWeight(w.bytes);
    be.prefetchWeight(bytes(blk.attn.qnorm));
    be.prefetchWeight(bytes(blk.attn.knorm));
    be.prefetchWeight(bytes(blk.attn_norm2));
    be.prefetchWeight(bytes(blk.ffn_norm2));
    inline for (.{ blk.ffn.w1, blk.ffn.w3, blk.ffn.w2 }) |w| be.prefetchWeight(w.bytes);
}

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
/// `smallBuffer` returns a raw handle; the elt launchers want a `DeviceBuffer`.
fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
