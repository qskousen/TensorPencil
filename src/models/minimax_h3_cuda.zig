//! CUDA-backend MiniMax H3 trunk forward, the device twin of `minimax_h3.forward`.
//!
//! The split follows `dit_cuda`'s: the 50-block trunk runs on the device and the
//! small host-cheap paths stay on the CPU. Here those are the patch projections,
//! the adaLN projection, the token refiner, and the patchify/pack transposes.
//! Their combined cost at the default render is under 0.01% of a step, and
//! `adaln_proj` is `[96768, 8]`, whose 8 columns are below every tiled path's
//! floor anyway.
//!
//! The output HEADS are on the device despite being 96 and 32 rows, well under
//! `opI8Gemm`'s 128-row floor: those weights are f32, so they take `opMatmul`,
//! which has no such floor. What made it worth moving is not the projection but
//! what reaching it used to cost -- the whole trunk, `[seq][hidden]` f32,
//! allocated on the host and downloaded every step, 925 MB at 15 seconds of
//! 512x768 against 16 MB for the results.
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
const lora_mod = @import("lora.zig");
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
            for (l.lora) |h| if (!lora_cuda.supported(h.target)) return false;
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
    /// The attention out-projection for one query band. Its own buffer, not a
    /// view into `t1_d`: `opI8Gemm` writes its row count ROUNDED UP to 128, so a
    /// band-offset view would clobber the next band's still-needed normalized
    /// rows, and the last band would run off the end.
    ao_d: Buf = .{},
    /// Q and the attention output for one band, in the attention's own bf16.
    /// Only allocated when `kv_bf16`.
    qb_d: Buf = .{},
    ob_d: Buf = .{},
    /// K and V are stored as bf16 and handed to cuDNN AS the operands, so the
    /// conversion staging that an f32 stream needs does not exist: that staging
    /// is another full copy of both, 2.62 GB at 15 s of 768x1152 on top of the
    /// 2.62 GB the f32 originals cost. Off on the hand-PTX arm, which has no bf16
    /// attention, and off under `force_naive_attn` for the same reason.
    kv_bf16: bool = false,
    gate_d: Buf = .{},
    up_d: Buf = .{},
    mod_d: Buf = .{},
    /// The final layer's modulation, `[max_labels][2][hidden]`, norm folded in.
    fmod_d: Buf = .{},
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

    /// Query rows the attention is processed in. Every band attends to the WHOLE
    /// key/value sequence, so only Q, the attention output and the out-projection
    /// band; K and V stay full. That is what takes `q_d` and `attn_d` off the
    /// sequence: 2.62 GB each at 15 seconds of 768x1152, against 117 MB here.
    pub const attn_band: usize = 4096;

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
        // Band-sized, not sequence-sized: every consumer (the qkv projections, the
        // MLP, the output heads) now works a band at a time, and the widest band
        // is the attention's.
        const tband = padRows(@min(@max(attn_band, mlp_tile), seq));
        ws.t1_d = try be.tensorCreate(tband * cfg.hidden * 4);
        const aband = padRows(@min(attn_band, seq));
        ws.kv_bf16 = be.kernels == .libs and !force_naive_attn;
        const kvw: usize = if (ws.kv_bf16) 2 else 4;
        ws.q_d = try be.tensorCreate(aband * inner * 4);
        ws.k_d = try be.tensorCreate(mpad * inner * kvw);
        ws.v_d = try be.tensorCreate(mpad * inner * kvw);
        ws.attn_d = try be.tensorCreate(aband * inner * 4);
        ws.ao_d = try be.tensorCreate(aband * cfg.hidden * 4);
        if (ws.kv_bf16) {
            ws.qb_d = try be.tensorCreate(aband * inner * 2);
            ws.ob_d = try be.tensorCreate(aband * inner * 2);
        }
        const tile = padRows(@min(mlp_tile, seq));
        ws.gate_d = try be.tensorCreate(tile * cfg.ffn * 4);
        ws.up_d = try be.tensorCreate(tile * cfg.ffn * 4);
        // All blocks' modulation in one buffer, so the whole step uploads once.
        ws.mod_d = try be.tensorCreate(dit.blocks.len * minimax_h3.Timesteps.max_labels * 3 * 6 * cfg.hidden * 4);
        ws.fmod_d = try be.tensorCreate(minimax_h3.Timesteps.max_labels * 2 * cfg.hidden * 4);
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
        // The attention's own device memory, reserved at the shape the forward
        // will ask for (full pack, MHA). cuDNN's plan workspace alone is 1.3 GB at
        // 15 seconds of 768x1152, and allocating it lazily inside the first
        // forward puts it AFTER the residency pass that decides how much of the
        // trunk to pin, which is how a long clip failed where a short one fit.
        if (!force_naive_attn) {
            // Reserved in the SHAPES and the DTYPE the forward will ask for. The
            // forward runs banded, so the plans are (band, seq) and (tail, seq),
            // never (seq, seq); and under bf16 storage the operands arrive in the
            // plan's own type, so there is no staging to reserve. Getting either
            // wrong reserves buffers nothing uses and leaves the real ones to be
            // allocated mid-forward, after residency has already pinned against
            // the space they need -- which is the whole reason this exists.
            const saved_io = be.attn_io_f16;
            const saved_bf = be.attn_bf16;
            be.attn_io_f16 = ws.kv_bf16;
            be.attn_bf16 = ws.kv_bf16;
            defer {
                be.attn_io_f16 = saved_io;
                be.attn_bf16 = saved_bf;
            }
            const full = @min(attn_band, seq);
            try be.reserveAttnCudnn(full, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim);
            const tail = seq % attn_band;
            if (tail != 0 and tail != full) try be.reserveAttnCudnn(tail, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim);
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
                for (l.lora) |h| for (h.target.factors) |f| {
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
        inline for (.{ &self.x_d, &self.t1_d, &self.q_d, &self.k_d, &self.v_d, &self.attn_d, &self.ao_d, &self.qb_d, &self.ob_d, &self.gate_d, &self.up_d, &self.mod_d, &self.fmod_d, &self.vmask_d, &self.amask_d }) |b| {
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
    stack: ?*const lora_mod.Stack,
    y: Buf,
    x: Buf,
    m: usize,
    l: minimax_h3.Lin,
    row0: usize,
    n: usize,
) !void {
    if (l.lora.len == 0) return;
    // The scratch is sized by `loraScratch` from the same DiT, so a missing one
    // means the LoRA was attached after the workspace was built. That would
    // otherwise be a render with no sidecar anywhere.
    const lws = if (ws.lora) |*w| w else {
        std.log.err("minimax_h3_cuda: {s} has a sidecar but the workspace has no LoRA scratch", .{l.lora[0].target.tag});
        return error.Unsupported;
    };
    for (l.lora) |h| {
        const s = stack.?.files.items[h.file].strength;
        // Exact: adding `0 * delta` changes nothing, so a dial at zero is free.
        if (s == 0) continue;
        try lora_cuda.applyRange(be, lws, y, x, m, h.target, s, row0, n);
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
    try minimax_h3.buildModTable(dit, io, gpa, mod_host, t_emb, n_labels);

    // The final layer's own table, norm folded into the premul so the heads use
    // the same weightless norm-and-modulate kernel the blocks do. Allocated out
    // here for the same reason `mod_host` is: its upload is queued on the stream
    // and must outlive the batch.
    const fmod_host = try gpa.alloc(f32, n_labels * 2 * h);
    defer gpa.free(fmod_host);
    try minimax_h3.buildFinalModTable(dit, io, gpa, fmod_host, t_emb, n_labels);

    // --- device: the trunk -------------------------------------------------
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try be.tensorUpload(ws.x_d, std.mem.sliceAsBytes(packed_h));
    try be.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mod_host));
    try be.tensorUpload(ws.fmod_d, std.mem.sliceAsBytes(fmod_host));

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
    // The index buffer for a segment. The launch's first row is a kernel argument,
    // so this is the whole buffer.
    const segIdx = struct {
        fn go(kind: minimax_h3.Kind, vm: ?Buf, am: ?Buf) ?Buf {
            return switch (kind) {
                .video => vm,
                .audio => am,
                else => null,
            };
        }
    }.go;

    // bf16 attention operands. H3's V is UNNORMED, and an unnormed value is the
    // operand that outgrows f16's 65504 ceiling on a real conditioning while a
    // synthetic one stays green -- the failure that renders solid white with no
    // error. bf16 carries f32's exponent, so the range question does not arise;
    // it costs 3 mantissa bits against an f32 accumulation that is unchanged.
    // `TP_ATTN_F16=1` puts it back on f16 for an A/B; there is no reason to pick
    // f16 for a render, since the ranges are what differ and bf16 has the room.
    const saved_io = be.attn_io_f16;
    be.attn_bf16 = ws.kv_bf16;
    // K/V/Q/O are handed over ALREADY in the plan's dtype, which is what this
    // flag means: no staging copies, no conversion inside the attention.
    be.attn_io_f16 = ws.kv_bf16;
    defer {
        be.attn_bf16 = false;
        be.attn_io_f16 = saved_io;
    }

    for (dit.blocks, 0..) |*b, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;

        // Attention half, in bands of query rows. `t1_d` is BAND-SIZED, so the
        // normalize-and-modulate runs once per band into row 0 of it rather than
        // once over the whole sequence: at 15 s of 768x1152 that buffer is 88 MB
        // instead of 1.97 GB, and the int8 prep staging shrinks with it.
        //
        // Each segment modulates with its OWN row, so a band applies the
        // INTERSECTION of itself with each segment, exactly as the MLP does.
        const normBand = struct {
            fn go(be2: *Backend, ws2: *Workspace, lay: *const minimax_h3.PackedLayout, t: *const minimax_h3.Timesteps, c: minimax_h3.Config, nl: usize, blk: usize, b0: usize, n: usize, hh: usize, stride: usize, vm: ?Buf, am: ?Buf) !void {
                for (lay.segments) |sg| {
                    const lo = @max(sg.start, b0);
                    const hi = @min(sg.stop, b0 + n);
                    if (lo >= hi) continue;
                    const idx = switch (sg.kind) {
                        .video => vm,
                        .audio => am,
                        else => null,
                    };
                    // With an index buffer the scalar offsets are the LABEL-ZERO
                    // ones; the kernel adds `idx[row] * label_stride`.
                    const base = if (idx == null) t.rowFor(sg.kind) else 0;
                    try be2.rmsModRowsAt(
                        ws2.x_d,
                        lo,
                        ws2.t1_d,
                        lo - b0,
                        ws2.mod_d,
                        hi - lo,
                        hh,
                        modOff(c, nl, blk, base, sg.kind.tag(), 0),
                        modOff(c, nl, blk, base, sg.kind.tag(), 1),
                        eps,
                        idx,
                        lo - sg.start,
                        stride,
                    );
                }
            }
        }.go;

        const qn = try normBuf(be, b.attn.q_norm);
        const kn = try normBuf(be, b.attn.k_norm);

        // Pass one: K and V over the WHOLE sequence, because every query band
        // attends to all of it. The fused qkv is split by rows, GEMMs into
        // separate planes rather than one GEMM and a de-interleave; the sidecar
        // splits the same way, which is also how a block-diagonal factor lands.
        //
        // This pass reads `x_d` before any of the residual adds below touch it,
        // so K and V see the block's input, which is what they must.
        var k0: usize = 0;
        while (k0 < seq) : (k0 += Workspace.attn_band) {
            const kn_rows = @min(Workspace.attn_band, seq - k0);
            try normBand(be, ws, layout, &ts, cfg, n_labels, bi, k0, kn_rows, h, label_stride, vmask, amask);
            try linPrep(be, ws.t1_d, kn_rows, h);
            // With bf16 storage the GEMM lands in the f32 band scratch (`attn_d`,
            // idle until pass two) and is narrowed into K/V at the band's offset;
            // `qkNorm` and the rope run on the f32 form either way, so neither
            // needs a bf16 variant.
            const kvw: usize = if (ws.kv_bf16) 2 else 4;
            const k_at = if (ws.kv_bf16) ws.attn_d else offsetBuf(ws.k_d, k0 * inner * 4);
            try lin(be, k_at, rowSlice(b.attn.qkv.w, inner, inner));
            try sidecar(be, ws, dit.lora, k_at, ws.t1_d, kn_rows, b.attn.qkv, inner, inner);
            try be.qkNorm(k_at, k_at, kn, kn_rows * cfg.n_heads, cfg.head_dim, cfg.qk_norm_eps);
            try be.opRopeHalfPart(k_at, sess.freqs_d, kn_rows, cfg.n_heads, sess.pairs, sess.sinOff(), k0, cfg.head_dim);
            if (ws.kv_bf16) try be.cvtF32ToBf16(k_at, offsetBuf(ws.k_d, k0 * inner * kvw), kn_rows * inner);

            const v_at = if (ws.kv_bf16) ws.attn_d else offsetBuf(ws.v_d, k0 * inner * 4);
            try lin(be, v_at, rowSlice(b.attn.qkv.w, 2 * inner, inner));
            try sidecar(be, ws, dit.lora, v_at, ws.t1_d, kn_rows, b.attn.qkv, 2 * inner, inner);
            if (ws.kv_bf16) try be.cvtF32ToBf16(v_at, offsetBuf(ws.v_d, k0 * inner * kvw), kn_rows * inner);
        }

        // Q, the attention and the out-projection, one band of query rows at a
        // time, each band folded straight into the residual.
        //
        // ⚠️ The rope table is indexed by GLOBAL position (`pos0`) while the data
        // is band-relative. Passing 0 there rotates every band as if it started at
        // the beginning of the clip, which is finite and wrong.
        var a0: usize = 0;
        while (a0 < seq) : (a0 += Workspace.attn_band) {
            const an = @min(Workspace.attn_band, seq - a0);
            // Re-normalized rather than kept from pass one: holding it would mean
            // a full-sequence `t1_d` again, which is the whole point of banding.
            // The residual adds below only ever touch rows this pass has already
            // read, so a later band still normalizes the block's own input.
            try normBand(be, ws, layout, &ts, cfg, n_labels, bi, a0, an, h, label_stride, vmask, amask);
            try linPrep(be, ws.t1_d, an, h);
            try lin(be, ws.q_d, rowSlice(b.attn.qkv.w, 0, inner));
            try sidecar(be, ws, dit.lora, ws.q_d, ws.t1_d, an, b.attn.qkv, 0, inner);
            try be.qkNorm(ws.q_d, ws.q_d, qn, an * cfg.n_heads, cfg.head_dim, cfg.qk_norm_eps);
            try be.opRopeHalfPart(ws.q_d, sess.freqs_d, an, cfg.n_heads, sess.pairs, sess.sinOff(), a0, cfg.head_dim);
            // MHA: kv_heads == n_heads. Unmasked, so a band of queries over the
            // whole key sequence is the same arithmetic as the square launch.
            if (force_naive_attn) {
                try be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, an, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, scale, false);
            } else if (ws.kv_bf16) {
                try be.cvtF32ToBf16(ws.q_d, ws.qb_d, an * inner);
                try be.opAttnCross(ws.qb_d, ws.k_d, ws.v_d, ws.ob_d, an, seq, cfg.n_heads, cfg.head_dim, scale);
                try be.cvtBf16ToF32(ws.ob_d, ws.attn_d, an * inner);
            } else {
                try be.opAttnCross(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, an, seq, cfg.n_heads, cfg.head_dim, scale);
            }

            try linPrep(be, ws.attn_d, an, inner);
            try lin(be, ws.ao_d, b.attn.out.w);
            try sidecar(be, ws, dit.lora, ws.ao_d, ws.attn_d, an, b.attn.out, 0, h);

            // The residual add, per band, as the INTERSECTION of the band with
            // each segment: a band can straddle segments and each carries its own
            // modulation row.
            for (layout.segments) |sg| {
                const lo = @max(sg.start, a0);
                const hi = @min(sg.stop, a0 + an);
                if (lo >= hi) continue;
                const idx = segIdx(sg.kind, vmask, amask);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try be.gatedAddRowsAt(
                    ws.x_d,
                    lo,
                    ws.ao_d,
                    lo - a0,
                    ws.mod_d,
                    (hi - lo) * h,
                    h,
                    modOff(cfg, n_labels, bi, base, sg.kind.tag(), 2),
                    idx,
                    lo - sg.start,
                    label_stride,
                );
            }
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
                const idx = segIdx(sg.kind, vmask, amask);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try be.rmsModRowsAt(
                    ws.x_d,
                    lo,
                    ws.t1_d,
                    lo - c0,
                    ws.mod_d,
                    hi - lo,
                    h,
                    modOff(cfg, n_labels, bi, base, tag, 3),
                    modOff(cfg, n_labels, bi, base, tag, 4),
                    eps,
                    idx,
                    lo - sg.start,
                    label_stride,
                );
            }
            try linPrep(be, ws.t1_d, tile, h);
            // fc1 is the fused swiglu gate+value; rows [0, ffn) are the GATE.
            try lin(be, ws.gate_d, rowSlice(b.mlp.fc1.w, 0, cfg.ffn));
            try lin(be, ws.up_d, rowSlice(b.mlp.fc1.w, cfg.ffn, cfg.ffn));
            try sidecar(be, ws, dit.lora, ws.gate_d, ws.t1_d, tile, b.mlp.fc1, 0, cfg.ffn);
            try sidecar(be, ws, dit.lora, ws.up_d, ws.t1_d, tile, b.mlp.fc1, cfg.ffn, cfg.ffn);
            try be.siluMul(ws.gate_d, ws.up_d, tile * cfg.ffn);
            try linPrep(be, ws.gate_d, tile, cfg.ffn);
            try lin(be, ws.t1_d, b.mlp.fc2.w);
            // `gate_d` is the fc2 activation, and `siluMul` wrote it in place,
            // so the sidecar reads the post-swiglu value like the base GEMM.
            try sidecar(be, ws, dit.lora, ws.t1_d, ws.gate_d, tile, b.mlp.fc2, 0, h);
            for (layout.segments) |sg| {
                const lo = @max(sg.start, c0);
                const hi = @min(sg.stop, c0 + tile);
                if (lo >= hi) continue;
                const idx = segIdx(sg.kind, vmask, amask);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try be.gatedAddRowsAt(
                    ws.x_d,
                    lo,
                    ws.t1_d,
                    lo - c0,
                    ws.mod_d,
                    (hi - lo) * h,
                    h,
                    modOff(cfg, n_labels, bi, base, sg.kind.tag(), 5),
                    idx,
                    lo - sg.start,
                    label_stride,
                );
            }
        }
    }
    // --- device: the output heads ------------------------------------------
    //
    // On the device even though the projections are 96 and 32 rows, well under
    // `opI8Gemm`'s 128-row floor: these weights are f32, so they take the plain
    // f32 GEMM, which has no such floor. Keeping them here is what removes the
    // whole-trunk download -- `[seq][hidden]` f32 is 925 MB at 15 seconds of
    // 512x768, allocated AND transferred every step, where the results below are
    // 16 MB.
    const v_seg = layout.segmentOf(.video).?;
    const a_seg = layout.segmentOf(.audio).?;
    const v_dim = cfg.videoPatchDim();
    const a_dim = minimax_h3.audio_latent_channels;
    std.debug.assert(dit.final.video_out.dtype == .f32 and dit.final.audio_out.dtype == .f32);

    try finalHeadDev(be, ws, ws.q_d, v_seg, ts.rowFor(.video), segIdx(.video, vmask, amask), h, dit.final.video_out, dit.final.video_bias, dit.cfg.final_norm_eps);
    try finalHeadDev(be, ws, ws.k_d, a_seg, ts.rowFor(.audio), segIdx(.audio, vmask, amask), h, dit.final.audio_out, dit.final.audio_bias, dit.cfg.final_norm_eps);
    try be.endBatch();

    // --- host: unpack and negate -------------------------------------------
    const v_rows = try gpa.alloc(f32, v_seg.len() * v_dim);
    defer gpa.free(v_rows);
    const a_rows = try gpa.alloc(f32, a_seg.len() * a_dim);
    defer gpa.free(a_rows);
    try be.tensorDownload(ws.q_d, std.mem.sliceAsBytes(v_rows));
    try be.tensorDownload(ws.k_d, std.mem.sliceAsBytes(a_rows));

    // The reference's last act is `[-video_out, -audio_out]`, and it is invisible
    // in every norm.
    minimax_h3.unpatchifyVideo(out_video, v_rows, layout.latent_t, layout.latent_h, layout.latent_w);
    for (out_video) |*x| x.* = -x.*;
    minimax_h3.unpackAudio(out_audio, a_rows, layout.audio_t);
    for (out_audio) |*x| x.* = -x.*;
}

/// One output head on the device: weightless norm + modulate from the folded
/// final table, then the f32 projection.
///
/// ⚠️ **The head modulates PER ROW when a denoise mask relabelled the segment.**
/// With an index buffer the scalar offsets are the LABEL-ZERO ones and the kernel
/// adds `idx[row] * stride`, exactly as the trunk's own segments do; passing the
/// segment's own label instead renders a masked clip differently depending on how
/// the mask happened to be expressed.
fn finalHeadDev(
    be: *Backend,
    ws: *Workspace,
    out: Buf,
    seg: minimax_h3.Segment,
    t_row: usize,
    idx: ?Buf,
    h: usize,
    w: Weight,
    bias: []const f32,
    norm_eps: f32,
) !void {
    const n = seg.len();
    const base = if (idx == null) t_row else 0;
    // Banded, because `t1_d` is band-sized and the video segment is most of the
    // sequence. `opMatmul` writes exactly `m * rows` (no 128-row rounding), so a
    // band lands at its own byte offset in `out` with nothing spilling past it.
    var r0: usize = 0;
    while (r0 < n) : (r0 += Workspace.attn_band) {
        const rn = @min(Workspace.attn_band, n - r0);
        try be.rmsModRowsAt(ws.x_d, seg.start + r0, ws.t1_d, 0, ws.fmod_d, rn, h, base * 2 * h, base * 2 * h + h, norm_eps, idx, r0, 2 * h);
        try be.opMatmul(out, r0 * w.rows * 4, ws.t1_d, 0, rn, w.bytes, false, w.rows, h, 1.0, bias);
    }
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
