//! CUDA-backend MiniMax H3 trunk forward, the device twin of `minimax_h3.forward`.
//!
//! The split follows `dit_cuda`'s: the 50-block trunk runs on the device and the
//! small host-cheap paths stay on the CPU. Here those are the patch projections,
//! the adaLN projection, the token refiner, the final heads, and the
//! patchify/pack transposes. Their combined cost at the default render is under
//! 0.01% of a step, and several of them have shapes the device GEMMs refuse
//! anyway: `adaln_proj` is `[96768, 8]`, whose 8 columns are below every tiled
//! path's floor, and the output heads are 96 and 32 rows where `opI8Gemm` needs
//! `rows % 128 == 0`.
//!
//! Three things differ from `dit_cuda` and each is load-bearing:
//!
//! 1. **Modulation is PER SEGMENT, not per sequence.** krea2 has one modulation
//!    vector per block; H3 has one per (timestep row, modality tag), and which
//!    one a row reads depends on which segment it is in. Segments are CONTIGUOUS,
//!    so each is a `rmsMod`/`gatedAdd` launch on an offset view with its own
//!    modulation offset. A handful of extra launches per block, no new kernel.
//! 2. **`rms_mod_par` applies no norm weight and no `1 +`**: it computes
//!    `rmsnorm(x) * premul + shift`. So the host folds `norm.weight * (1 + scale)`
//!    into `premul`. Uploading a bare `scale` there drops the norm weight AND the
//!    identity term, which is finite and wrong.
//! 3. **The fused weights are SPLIT BY ROWS on the host.** `qkv_proj` is one
//!    `[3 * inner, hidden]` tensor and `fc1` one `[2 * ffn, hidden]`, but the
//!    device wants separate q/k/v and gate/up buffers. A row range of a row-major
//!    weight is a contiguous byte range and its per-row scales slice with it, so
//!    the split is three (or two) `Weight` views over the same mapping, with no
//!    copy and no de-interleave kernel.
//!
//! Numerics match `minimax_h3.forward` up to int8 quantization and the softmax
//! approximation, the same regime `dit_cuda` runs in. `minimax-h3-cuda-test`
//! checks it against the CPU forward on real weights.

const std = @import("std");
const minimax_h3 = @import("minimax_h3.zig");
const lora_cuda = @import("lora_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = minimax_h3.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

const eps: f32 = 1e-5;

/// Force the naive one-thread-per-(query, head) attention instead of the
/// tensor-core path. For A/B and for reproducing a mismatch.
pub var force_naive_attn: bool = false;

/// A device-buffer sub-view offset `off_bytes` into `b`. CUDA buffers are raw
/// pointers, so this is pointer arithmetic; it does NOT port to Vulkan, where
/// `buf` is an opaque handle.
fn offsetBuf(b: Buf, off_bytes: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = b.size - off_bytes };
}

/// A contiguous row range of a row-major weight, as its own `Weight`.
///
/// Rows of a 2-D weight are `cols` elements apart with no padding, so rows
/// `[from, from + n)` are one byte range, and an int8 weight's per-row scales
/// slice identically. This is what splits the fused `qkv_proj` and `fc1` without
/// a copy.
fn rowSlice(w: Weight, from: usize, n: usize) Weight {
    std.debug.assert(from + n <= w.rows);
    const stride = w.dtype.storageBytes(w.cols);
    var out = w;
    out.bytes = w.bytes[from * stride ..][0 .. n * stride];
    out.rows = n;
    if (w.row_scale) |rs| out.row_scale = rs[from..][0..n];
    return out;
}

/// Whether this checkpoint's trunk can run here.
///
/// int8 convrot only for now, which is what the shipping checkpoint is. Refuse by
/// name rather than at a launch: a dtype with no GEMM path here would otherwise
/// surface as a bad device access several frames deep.
pub fn supported(dit: *const DiT) bool {
    for (dit.blocks) |b| {
        inline for (.{ b.attn.qkv, b.attn.out, b.mlp.fc1, b.mlp.fc2 }) |l| {
            const w = l.w;
            if (w.dtype != .i8) return false;
            if (w.row_scale == null) return false;
            // `opI8Gemm` launches `grid.x = rows / 128`. Every H3 width clears it
            // (21504 / 7168 / 5376 / 28672 / 14336), but a sliced view must too.
            if (w.rows % 128 != 0) return false;
            // A LoRA sidecar this backend cannot apply makes the whole trunk
            // unsupported, not "supported without the LoRA": running the base
            // GEMM alone is a different model, silently.
            if (l.lora) |t| if (!lora_cuda.supported(t)) return false;
        }
    }
    const cfg = dit.cfg;
    // The row splits below must land on 128-row boundaries as well.
    if ((cfg.n_heads * cfg.head_dim) % 128 != 0) return false;
    if (cfg.ffn % 128 != 0) return false;
    return true;
}

/// Per-render device state: everything constant across sampling steps.
pub const Session = struct {
    seq: usize,
    /// `[seq * pairs]` cos then `[seq * pairs]` sin, f32. `sin_off` is the split.
    freqs_d: Buf = .{},
    pairs: usize,

    pub fn init(be: *Backend, gpa: std.mem.Allocator, dit: *const DiT, layout: *const minimax_h3.PackedLayout) !Session {
        const cfg = dit.cfg;
        const pairs = cfg.ropePairs();
        var s: Session = .{ .seq = layout.seq_len, .pairs = pairs };
        errdefer s.deinit(be);

        // The same table the CPU path builds, uploaded once: it depends only on
        // the packed layout's position grid, which is fixed for a render.
        var freqs = try minimax_h3.ropeFreqs(gpa, layout.pos, dit.rope_inv_freq);
        defer freqs.deinit(gpa);
        const host = try gpa.alloc(f32, 2 * layout.seq_len * pairs);
        defer gpa.free(host);
        @memcpy(host[0 .. layout.seq_len * pairs], freqs.cos);
        @memcpy(host[layout.seq_len * pairs ..], freqs.sin);
        s.freqs_d = try be.tensorCreate(host.len * 4);
        try be.tensorUpload(s.freqs_d, std.mem.sliceAsBytes(host));
        return s;
    }

    pub fn sinOff(self: Session) usize {
        return self.seq * self.pairs;
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.freqs_d);
    }
};

/// Per-shape device scratch.
pub const Workspace = struct {
    x_d: Buf = .{},
    t1_d: Buf = .{},
    q_d: Buf = .{},
    k_d: Buf = .{},
    v_d: Buf = .{},
    attn_d: Buf = .{},
    gate_d: Buf = .{},
    up_d: Buf = .{},
    mod_d: Buf = .{},
    /// Per-row modulation LABEL indices (u32) for the two target segments, used
    /// only when a denoise mask relabels rows inside them. `seq` entries is an
    /// over-allocation of a few hundred KB, and sizing it from the mask would mean
    /// the workspace depended on something the layout does not carry.
    vmask_d: Buf = .{},
    amask_d: Buf = .{},
    /// Only allocated when this DiT has a LoRA attached.
    lora: ?lora_cuda.Workspace = null,

    /// Rows the MLP is processed in. The gate/up intermediates are
    /// `[tile][ffn]`, so this bounds the biggest activation: at the default
    /// render an untiled pair would be 2.2 GB each. The MLP is per row, so the
    /// bands are independent.
    pub const mlp_tile: usize = 2048;

    /// Rows every int8 GEMM output must be sized for.
    ///
    /// `opI8Gemm` launches `grid.y = i8_mpad / 128`, i.e. it writes the activation
    /// row count ROUNDED UP to 128, not the count itself. A buffer sized to the
    /// exact row count is written past its end: at the 147-row development shape
    /// the GEMM writes 256 rows. `dit_cuda` pads for the same reason.
    pub fn padRows(rows: usize) usize {
        return std.mem.alignForward(usize, rows, 128);
    }

    /// Sized for `Timesteps.max_labels`, NOT for the count at any one sigma.
    ///
    /// The distinct-timestep count CHANGES along the schedule: at sigma 1.0 the
    /// two streams and their condition pins collapse to 3 labels, and mid-schedule
    /// there are 4. Sizing from one sigma and uploading another's modulation
    /// overruns the buffer, which surfaces as a `CudaError` from the upload two
    /// steps into a render. The bound is small enough that reserving it is free.
    pub fn init(be: *Backend, dit: *const DiT, seq: usize) !Workspace {
        const cfg = dit.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        const mpad = padRows(seq);
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(mpad * cfg.hidden * 4);
        ws.t1_d = try be.tensorCreate(mpad * cfg.hidden * 4);
        ws.q_d = try be.tensorCreate(mpad * inner * 4);
        ws.k_d = try be.tensorCreate(mpad * inner * 4);
        ws.v_d = try be.tensorCreate(mpad * inner * 4);
        ws.attn_d = try be.tensorCreate(mpad * inner * 4);
        const tile = padRows(@min(mlp_tile, seq));
        ws.gate_d = try be.tensorCreate(tile * cfg.ffn * 4);
        ws.up_d = try be.tensorCreate(tile * cfg.ffn * 4);
        // All blocks' modulation in one buffer, so the whole step uploads once.
        ws.mod_d = try be.tensorCreate(dit.blocks.len * minimax_h3.Timesteps.max_labels * 3 * 6 * cfg.hidden * 4);
        ws.vmask_d = try be.tensorCreate(seq * 4);
        ws.amask_d = try be.tensorCreate(seq * 4);
        if (loraScratch(dit, seq)) |sz| {
            ws.lora = try lora_cuda.Workspace.init(be, sz.lo, sz.hi);
            // Pre-size the backend's zero bias to the widest output the sidecar
            // will ask for. `opGemmBf16`'s hand-PTX arm fetches one per call and
            // GROWS it, and the grown buffer is a new host pointer while the old
            // one stays in the pointer-keyed device weight cache. Reaching the
            // final size before the first GEMM means it never grows mid-render.
            _ = try be.zeroBias(sz.widest_out);
        }
        return ws;
    }

    /// The sidecar scratch this DiT needs, or null when no LoRA is attached.
    ///
    /// Sized from the widest range the trunk asks for on each side, not from one
    /// linear: the attention half applies over the whole sequence into an
    /// `inner`-wide plane while the MLP half applies over a row band into an
    /// `ffn`-wide one, and either can be the larger.
    fn loraScratch(dit: *const DiT, seq: usize) ?struct { lo: usize, hi: usize, widest_out: usize } {
        const cfg = dit.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        const mpad = padRows(seq);
        const tile = padRows(@min(mlp_tile, seq));
        var max_rank: usize = 0;
        for (dit.blocks) |b| {
            inline for (.{ b.attn.qkv, b.attn.out, b.mlp.fc1, b.mlp.fc2 }) |l| {
                if (l.lora) |t| for (t.factors) |f| {
                    max_rank = @max(max_rank, f.a.rows);
                };
            }
        }
        if (max_rank == 0) return null;
        return .{
            .lo = mpad * max_rank,
            .hi = @max(mpad * inner, tile * cfg.ffn),
            // The output widths `forward` asks for, plus the rank (the A GEMM's
            // own output width).
            .widest_out = @max(max_rank, @max(inner, cfg.ffn)),
        };
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{ &self.x_d, &self.t1_d, &self.q_d, &self.k_d, &self.v_d, &self.attn_d, &self.gate_d, &self.up_d, &self.mod_d, &self.vmask_d, &self.amask_d }) |b| {
            be.tensorDestroy(b);
        }
        if (self.lora) |*l| l.deinit(be);
        self.* = undefined;
    }
};

/// Element offset of one modulation vector in `mod_d`.
///
/// `[block][t_row][tag][slot][hidden]`, where the six slots are
/// `(premul_msa, shift_msa, gate_msa, premul_mlp, shift_mlp, gate_mlp)`. Note
/// PREMUL, not scale: see the header, the norm weight and the `1 +` are folded in
/// on the host.
fn modOff(cfg: minimax_h3.Config, n_labels: usize, block: usize, t_row: usize, tag: minimax_h3.Tag, slot: usize) usize {
    std.debug.assert(t_row < n_labels and slot < 6);
    return (((block * n_labels + t_row) * 3 + @intFromEnum(tag)) * 6 + slot) * cfg.hidden;
}

/// Build every block's modulation on the host, folded for `rms_mod_par`.
///
/// Cheap: the adaLN projection is `[6 * hidden * 3, time_embed_dim]` against at
/// most a handful of timestep rows, so this is a few million MACs against the
/// trunk's trillions.
fn buildMod(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    t_emb: []const f32,
    n_labels: usize,
) !void {
    const cfg = dit.cfg;
    const h = cfg.hidden;
    const per_block = n_labels * 3 * 6 * h;
    std.debug.assert(out.len == dit.blocks.len * per_block);

    const raw = try gpa.alloc(f32, n_labels * 3 * 6 * h);
    defer gpa.free(raw);

    for (dit.blocks, 0..) |*b, bi| {
        try ops.matmul.matmul(io, gpa, raw, t_emb, n_labels, b.adaln, b.adaln_bias);
        // `raw` is [n_labels][tag][slot][hidden] with slots
        // (shift, scale, gate) x2, which is the reference's chunk order.
        for (0..n_labels) |t_row| {
            for (0..3) |tag| {
                const src = raw[((t_row * 3) + tag) * 6 * h ..][0 .. 6 * h];
                const dst = out[bi * per_block + ((t_row * 3) + tag) * 6 * h ..][0 .. 6 * h];
                inline for (.{ .{ 0, b.norm1 }, .{ 3, b.norm2 } }) |pair| {
                    const base = pair[0];
                    const nw = pair[1];
                    const shift = src[base * h ..][0..h];
                    const scale = src[(base + 1) * h ..][0..h];
                    const gate = src[(base + 2) * h ..][0..h];
                    // premul = norm_weight * (1 + scale): the kernel has neither.
                    for (dst[base * h ..][0..h], nw, scale) |*d, w, sc| d.* = w * (1.0 + sc);
                    @memcpy(dst[(base + 1) * h ..][0..h], shift);
                    @memcpy(dst[(base + 2) * h ..][0..h], gate);
                }
            }
        }
    }
}

fn linPrep(be: *Backend, x: Buf, m: usize, cols: usize) !void {
    try be.opI8Prep(x, m, cols, false);
}

fn lin(be: *Backend, y: Buf, w: Weight) !void {
    std.debug.assert(w.rows % 128 == 0);
    try be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
}

/// `y[m][n] += sidecar(l)` over output rows `[row0, row0 + n)` of `l`.
///
/// `x` is the linear's f32 activation, which `opI8Prep` READS rather than
/// rewrites, so it is still there after the base GEMM: the sidecar works in the
/// unrotated space and must see the same activation the prep did.
///
/// A no-op when `l` has no sidecar, so every base GEMM gets one of these beside
/// it and the pairing is visible at the call site.
fn sidecar(
    be: *Backend,
    ws: *Workspace,
    y: Buf,
    x: Buf,
    m: usize,
    l: minimax_h3.Lin,
    row0: usize,
    n: usize,
) !void {
    const t = l.lora orelse return;
    // The scratch is sized by `loraScratch` from the same DiT, so a missing one
    // means the LoRA was attached after the workspace was built. That would
    // otherwise be a render with no sidecar anywhere.
    if (ws.lora) |*lws| {
        try lora_cuda.applyRange(be, lws, y, x, m, t, row0, n);
    } else {
        std.log.err("minimax_h3_cuda: {s} has a sidecar but the workspace has no LoRA scratch", .{t.tag});
        return error.Unsupported;
    }
}

/// One trunk forward on the device.
///
/// The host does everything outside the 50 blocks; `in` and the outputs are the
/// same shapes `minimax_h3.forward` takes, so the two are drop-in alternatives.
pub fn forward(
    dit: *const DiT,
    be: *Backend,
    sess: *const Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: *const minimax_h3.PackedLayout,
    out_video: []f32,
    out_audio: []f32,
    in: minimax_h3.Inputs,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = dit.cfg;
    const seq = layout.seq_len;
    const h = cfg.hidden;
    const inner = cfg.n_heads * cfg.head_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    std.debug.assert(sess.seq == seq);

    // --- host: embed both streams into the packed sequence -----------------
    const packed_h = try gpa.alloc(f32, seq * h);
    defer gpa.free(packed_h);
    var ts = try minimax_h3.embedPacked(dit, io, gpa, packed_h, layout, in);
    defer ts.deinit(gpa);
    const labels = ts.labels();
    const n_labels = labels.len;

    // --- host: the time embedding and every block's modulation -------------
    const t_emb = try gpa.alloc(f32, n_labels * cfg.time_embed_dim);
    defer gpa.free(t_emb);
    minimax_h3.timeEmbed(t_emb, dit.adaln_t_table, cfg.adaln_curve_grid.?, cfg.time_embed_dim, labels);

    const mod_host = try gpa.alloc(f32, dit.blocks.len * n_labels * 3 * 6 * h);
    defer gpa.free(mod_host);
    try buildMod(dit, io, gpa, mod_host, t_emb, n_labels);

    // --- device: the trunk -------------------------------------------------
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try be.tensorUpload(ws.x_d, std.mem.sliceAsBytes(packed_h));
    try be.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mod_host));

    // A denoise mask relabels rows inside a target segment, so those two segments
    // pick their modulation row PER ROW. The kernels take a u32 index buffer and a
    // stride; the alternative is one launch per run of equal labels, which for a
    // spatial mask is thousands of launches of a few rows each.
    const label_stride = minimax_h3.modality_count * 6 * h;
    var vmask: ?@TypeOf(ws.vmask_d) = null;
    var amask: ?@TypeOf(ws.amask_d) = null;
    // ⚠️ **The staging buffer has to outlive the whole batch.** Uploads inside a
    // batch are queued on the stream, so a host buffer freed at the end of this
    // block is read after it is gone: the two tables are separate allocations
    // living to the end of the function, exactly like `mod_host` above. Getting
    // this wrong hung the device two steps into a render, not at the copy.
    var v_stage: []u32 = &.{};
    defer if (v_stage.len > 0) gpa.free(v_stage);
    var a_stage: []u32 = &.{};
    defer if (a_stage.len > 0) gpa.free(a_stage);
    if (ts.rowsFor(.video).len > 0) {
        const src = ts.rowsFor(.video);
        v_stage = try gpa.alloc(u32, src.len);
        for (v_stage, src) |*o, v| o.* = v;
        try be.tensorUpload(ws.vmask_d, std.mem.sliceAsBytes(v_stage));
        vmask = ws.vmask_d;
    }
    if (ts.rowsFor(.audio).len > 0) {
        const src = ts.rowsFor(.audio);
        a_stage = try gpa.alloc(u32, src.len);
        for (a_stage, src) |*o, v| o.* = v;
        try be.tensorUpload(ws.amask_d, std.mem.sliceAsBytes(a_stage));
        amask = ws.amask_d;
    }
    // The index buffer for a segment, offset to the launch's first row.
    const segIdx = struct {
        fn go(kind: minimax_h3.Kind, vm: ?Buf, am: ?Buf, first: usize) ?Buf {
            const b = switch (kind) {
                .video => vm,
                .audio => am,
                else => null,
            } orelse return null;
            return offsetBuf(b, first * 4);
        }
    }.go;

    for (dit.blocks, 0..) |*b, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;

        // Attention half. Each segment normalizes and modulates with its OWN
        // modulation row; the segments are contiguous, so each is an offset view.
        for (layout.segments) |sg| {
            const t_row = ts.rowFor(sg.kind);
            const tag = sg.kind.tag();
            const idx = segIdx(sg.kind, vmask, amask, 0);
            // With an index buffer the scalar offsets are the LABEL-ZERO ones; the
            // kernel adds `idx[row] * label_stride`.
            const base = if (idx == null) t_row else 0;
            try be.rmsModRows(
                offsetBuf(ws.x_d, sg.start * h * 4),
                offsetBuf(ws.t1_d, sg.start * h * 4),
                ws.mod_d,
                sg.len(),
                h,
                modOff(cfg, n_labels, bi, base, tag, 0),
                modOff(cfg, n_labels, bi, base, tag, 1),
                eps,
                idx,
                label_stride,
            );
        }
        try linPrep(be, ws.t1_d, seq, h);
        // The fused qkv, split by rows: three GEMMs into three planes rather
        // than one GEMM and a de-interleave. The sidecar splits the same way,
        // by row range, which is also how a block-diagonal factor lands.
        try lin(be, ws.q_d, rowSlice(b.attn.qkv.w, 0, inner));
        try lin(be, ws.k_d, rowSlice(b.attn.qkv.w, inner, inner));
        try lin(be, ws.v_d, rowSlice(b.attn.qkv.w, 2 * inner, inner));
        try sidecar(be, ws, ws.q_d, ws.t1_d, seq, b.attn.qkv, 0, inner);
        try sidecar(be, ws, ws.k_d, ws.t1_d, seq, b.attn.qkv, inner, inner);
        try sidecar(be, ws, ws.v_d, ws.t1_d, seq, b.attn.qkv, 2 * inner, inner);

        const qn = try normBuf(be, b.attn.q_norm);
        const kn = try normBuf(be, b.attn.k_norm);
        try be.qkNorm(ws.q_d, ws.q_d, qn, seq * cfg.n_heads, cfg.head_dim, cfg.qk_norm_eps);
        try be.qkNorm(ws.k_d, ws.k_d, kn, seq * cfg.n_heads, cfg.head_dim, cfg.qk_norm_eps);
        // PARTIAL split-half rope: `pairs` pairs of a `head_dim`-wide head, so
        // the tail passes through. `opRopeHalfPart` takes both widths for exactly
        // this; the full-head `rope` would rotate dims that must not move.
        try be.opRopeHalfPart(ws.q_d, sess.freqs_d, seq, cfg.n_heads, sess.pairs, sess.sinOff(), 0, cfg.head_dim);
        try be.opRopeHalfPart(ws.k_d, sess.freqs_d, seq, cfg.n_heads, sess.pairs, sess.sinOff(), 0, cfg.head_dim);
        // Full attention over the whole pack, and MHA: kv_heads == n_heads.
        if (force_naive_attn)
            try be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, scale, false)
        else
            try be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, scale);

        try linPrep(be, ws.attn_d, seq, inner);
        try lin(be, ws.t1_d, b.attn.out.w);
        try sidecar(be, ws, ws.t1_d, ws.attn_d, seq, b.attn.out, 0, h);
        for (layout.segments) |sg| {
            const idx = segIdx(sg.kind, vmask, amask, 0);
            const base = if (idx == null) ts.rowFor(sg.kind) else 0;
            try be.gatedAddRows(
                offsetBuf(ws.x_d, sg.start * h * 4),
                offsetBuf(ws.t1_d, sg.start * h * 4),
                ws.mod_d,
                sg.len() * h,
                h,
                modOff(cfg, n_labels, bi, base, sg.kind.tag(), 2),
                idx,
                label_stride,
            );
        }

        // MLP half, in row bands so the gate/up intermediates stay bounded.
        // A band can straddle segments, so the per-segment modulation is applied
        // as the INTERSECTION of the band with each segment.
        var c0: usize = 0;
        while (c0 < seq) : (c0 += Workspace.mlp_tile) {
            const tile = @min(Workspace.mlp_tile, seq - c0);
            for (layout.segments) |sg| {
                const lo = @max(sg.start, c0);
                const hi = @min(sg.stop, c0 + tile);
                if (lo >= hi) continue;
                const tag = sg.kind.tag();
                // A band starts mid-segment, so the index buffer starts there too.
                const idx = segIdx(sg.kind, vmask, amask, lo - sg.start);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try be.rmsModRows(
                    offsetBuf(ws.x_d, lo * h * 4),
                    offsetBuf(ws.t1_d, (lo - c0) * h * 4),
                    ws.mod_d,
                    hi - lo,
                    h,
                    modOff(cfg, n_labels, bi, base, tag, 3),
                    modOff(cfg, n_labels, bi, base, tag, 4),
                    eps,
                    idx,
                    label_stride,
                );
            }
            try linPrep(be, ws.t1_d, tile, h);
            // fc1 is the fused swiglu gate+value; rows [0, ffn) are the GATE.
            try lin(be, ws.gate_d, rowSlice(b.mlp.fc1.w, 0, cfg.ffn));
            try lin(be, ws.up_d, rowSlice(b.mlp.fc1.w, cfg.ffn, cfg.ffn));
            try sidecar(be, ws, ws.gate_d, ws.t1_d, tile, b.mlp.fc1, 0, cfg.ffn);
            try sidecar(be, ws, ws.up_d, ws.t1_d, tile, b.mlp.fc1, cfg.ffn, cfg.ffn);
            try be.siluMul(ws.gate_d, ws.up_d, tile * cfg.ffn);
            try linPrep(be, ws.gate_d, tile, cfg.ffn);
            try lin(be, ws.t1_d, b.mlp.fc2.w);
            // `gate_d` is the fc2 activation, and `siluMul` wrote it in place,
            // so the sidecar reads the post-swiglu value like the base GEMM.
            try sidecar(be, ws, ws.t1_d, ws.gate_d, tile, b.mlp.fc2, 0, h);
            for (layout.segments) |sg| {
                const lo = @max(sg.start, c0);
                const hi = @min(sg.stop, c0 + tile);
                if (lo >= hi) continue;
                const idx = segIdx(sg.kind, vmask, amask, lo - sg.start);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try be.gatedAddRows(
                    offsetBuf(ws.x_d, lo * h * 4),
                    offsetBuf(ws.t1_d, (lo - c0) * h * 4),
                    ws.mod_d,
                    (hi - lo) * h,
                    h,
                    modOff(cfg, n_labels, bi, base, sg.kind.tag(), 5),
                    idx,
                    label_stride,
                );
            }
        }
    }
    try be.endBatch();

    // --- host: the output heads --------------------------------------------
    const trunk = try gpa.alloc(f32, seq * h);
    defer gpa.free(trunk);
    try be.tensorDownload(ws.x_d, std.mem.sliceAsBytes(trunk));
    try minimax_h3.finalHeads(dit, io, gpa, layout, &ts, t_emb, trunk, out_video, out_audio);
}

fn normBuf(be: *Backend, w: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

// --- tests -----------------------------------------------------------------

test "a fused weight's row slice is a contiguous view with its own scales" {
    // This is what splits `qkv_proj` and `fc1` without a copy: rows of a
    // row-major weight are `cols` apart with no padding, and an int8 weight's
    // per-row scales slice with them. Pairing row i's bytes with row j's scale
    // is finite and wrong, so the two must move together.
    const bytes = [_]u8{0} ** 24;
    const scales = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var w = Weight.init(&bytes, .i8, 6, 4);
    w.row_scale = &scales;

    const mid = rowSlice(w, 2, 3);
    try std.testing.expectEqual(@as(usize, 3), mid.rows);
    try std.testing.expectEqual(@as(usize, 4), mid.cols);
    try std.testing.expectEqual(@as(usize, 12), mid.bytes.len);
    try std.testing.expectEqual(bytes[8..20].ptr, mid.bytes.ptr);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5 }, mid.row_scale.?);

    // The whole thing round-trips, and the three qkv slices tile it exactly.
    const all = rowSlice(w, 0, 6);
    try std.testing.expectEqual(w.bytes.len, all.bytes.len);
    var covered: usize = 0;
    for ([_]usize{ 0, 2, 4 }) |from| covered += rowSlice(w, from, 2).rows;
    try std.testing.expectEqual(w.rows, covered);
}

test "modulation offsets are distinct per block, timestep and tag" {
    // The device reads one modulation vector per (block, timestep row, tag,
    // slot); two of them colliding would silently modulate a segment with
    // another's parameters.
    const cfg: minimax_h3.Config = .{
        .hidden = 8,
        .n_layers = 2,
        .refiner_layers = 1,
        .n_heads = 1,
        .head_dim = 8,
        .ffn = 8,
        .text_dim = 8,
        .time_embed_dim = 8,
        .adaln_curve_grid = 4,
        .rope_inv_freq_len = 1,
    };
    const n_labels = 3;
    var seen = std.AutoHashMap(usize, void).init(std.testing.allocator);
    defer seen.deinit();
    for (0..2) |b| {
        for (0..n_labels) |t| {
            inline for (@typeInfo(minimax_h3.Tag).@"enum".fields) |f| {
                for (0..6) |slot| {
                    const off = modOff(cfg, n_labels, b, t, @enumFromInt(f.value), slot);
                    try std.testing.expect(!seen.contains(off));
                    try seen.put(off, {});
                }
            }
        }
    }
    // ...and they pack the buffer exactly, with no gaps to size around.
    try std.testing.expectEqual(@as(usize, 2 * n_labels * 3 * 6), seen.count());
    try std.testing.expectEqual(@as(usize, 0), modOff(cfg, n_labels, 0, 0, .video, 0));
    try std.testing.expectEqual(
        @as(usize, 2 * n_labels * 3 * 6 * cfg.hidden - cfg.hidden),
        modOff(cfg, n_labels, 1, n_labels - 1, .audio, 5),
    );
}
