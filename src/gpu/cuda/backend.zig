//! CUDA `GpuBackend` — the dit_gpu/vae_gpu-facing surface on top of the thin
//! driver `Context` (cu.zig/context.zig) and the hand-PTX `kernels`. Mirrors the
//! method surface of the Vulkan `gpu.Context` so a generic (comptime-dispatched)
//! `dit_gpu.forward` runs on either backend. Buffers are CUDA device pointers
//! wrapped as opaque `Handle`s; a "batch" is just the context stream.
//!
//! Value types (`DeviceBuffer`, `EltPush`, `Elt`) are defined here with the same
//! field/variant names the Vulkan context uses, so dit_gpu's anonymous literals
//! (`.{ .u0 = ... }`) and enum literals (`.rmsnorm`) coerce against `@TypeOf(ctx)`.

const std = @import("std");
const cu = @import("cu.zig");
const ctxmod = @import("context.zig");
const kernels = @import("kernels.zig");
const elt = @import("elt.zig");
const cublaslt = @import("cublaslt.zig");
const cudnn = @import("cudnn.zig");
const dtypes = @import("../../dtype.zig");

const Context = ctxmod.Context;

pub const Error = error{ CudaError, OutOfMemory, NoSuitableDevice, DeviceOutOfMemory };

/// Which set of compute kernels the heavy op methods dispatch to.
///  - hand_ptx: the pure-Zig hand-emitted PTX kernels (`--backend zig-cuda`).
///  - libs:     NVIDIA's cuBLASLt / cuDNN, dlopen'd (`--backend cuda`, Phase 2).
pub const KernelMode = enum { hand_ptx, libs };

/// dlopen'd cuBLASLt + cuDNN handles for the `.libs` kernel mode. Handles are
/// bound to the backend's CUDA stream; the workspace is cuBLASLt's scratch.
pub const Libs = struct {
    lt: cublaslt.Api,
    lt_handle: cublaslt.Handle = null,
    dnn: cudnn.Api,
    dnn_handle: cudnn.Handle = null,
    workspace: ctxmod.Buffer = .{},
};

/// cuBLASLt matmul workspace (device scratch); 32 MiB is ample for the DiT
/// GEMM shapes on sm_86.
const lt_workspace_bytes: usize = 32 << 20;

/// DIAGNOSTIC ONLY: skip the int8 irescale pass (produces garbage output) to
/// measure its batched cost — the ceiling on what fusing dequant could save.
pub var bench_skip_rescale: bool = false;

/// Numeric kind of a cuBLASLt matmul plan: int8 A/B → s32 D (32I compute) or
/// f16 A/B → f32 D (32F compute, HMMA). Both are the TN case with the same
/// layout mapping; only the data/compute/scale types differ.
const LtKind = enum { i8, f16 };

/// A cached cuBLASLt matmul plan for one (kind,n,m,k) shape: the operation
/// descriptor, the three matrix layouts (no data pointer baked in), and the
/// heuristic-selected algo. Destroyed in `Backend.deinit`.
const LtPlan = struct {
    desc: cublaslt.MatmulDesc,
    adesc: cublaslt.MatrixLayout,
    bdesc: cublaslt.MatrixLayout,
    ddesc: cublaslt.MatrixLayout,
    algo: cublaslt.MatmulAlgo,
};

/// Opaque device-buffer / memory token (a CUdeviceptr for `buf`; unused for
/// `mem`). `null_handle` == 0 (a valid alloc never returns device ptr 0).
pub const Handle = enum(u64) { null_handle = 0, _ };

pub const DeviceBuffer = struct {
    buf: Handle = .null_handle,
    mem: Handle = .null_handle,
    size: u64 = 0,

    pub fn ptr(self: DeviceBuffer) cu.CUdeviceptr {
        return @intFromEnum(self.buf);
    }
};

fn dbFromPtr(p: cu.CUdeviceptr, size: u64) DeviceBuffer {
    return .{ .buf = @enumFromInt(p), .mem = .null_handle, .size = size };
}

/// Push constants for eltwise/attention kernels (matches the Vulkan EltPush).
pub const EltPush = extern struct {
    u0: u32 = 0,
    u1: u32 = 0,
    u2: u32 = 0,
    u3: u32 = 0,
    u4: u32 = 0,
    u5: u32 = 0,
    f0: f32 = 0,
    f1: f32 = 0,
};

/// The eltwise kernel selector. Names match the Vulkan `Elt` enum so dit_gpu's
/// enum literals coerce. Only the DiT-path subset is implemented (see opElt).
pub const Elt = enum {
    rms_partial,
    rms_combine,
    rms_apply_mod,
    rms_apply_mod_h16,
    rmsnorm,
    rope_inter,
    qknorm_rope16,
    qknorm_rope_f32,
    gather_kmajor,
    gather_kmajor_h16,
    gather_kmajor16,
    f32_to_h16,
    attn_scores,
    softmax_partial,
    softmax_combine,
    attn_out,
    sigmoid_mul,
    sigmoid_mul_h16,
    sigmoid_mul_g16,
    silu_mul,
    silu_mul_h16,
    silu_mul16,
    gated_add,
    gated_add16,
};

const WeightEntry = struct {
    db: DeviceBuffer,
    last_use: u64 = 0,
    /// Pinned entries (first-touch, up to pin_budget) are immune to eviction.
    pinned: bool = false,
    /// Prefetched but not yet read by any op. Shielded from eviction: evicting
    /// now would discard a transfer that hasn't paid for itself, and the op it
    /// was prefetched for would re-upload it synchronously (double transfer).
    awaiting_use: bool = false,
    upload_ev: cu.CUevent = null,
    /// prefetch generation (0 = not prefetched / synchronously uploaded). A hit
    /// with pf_gen > pf_completed is still in flight on the prefetch thread; the
    /// consumer waits until pf_completed >= pf_gen before using it.
    pf_gen: u64 = 0,
};

/// A queued weight upload for the prefetch thread: memcpy `bytes` (mmap) into a
/// pinned slot then async-DMA into `db`, recording `ev` (compute waits on it).
const PrefetchReq = struct { bytes: []const u8, db: DeviceBuffer, ev: cu.CUevent, gen: u64 };
const pf_ring_sz = 512;

/// Weight-cache entries at or below this size are never evicted (see
/// evictOneWeight): eviction reclaims negligible VRAM and the synchronous
/// re-upload stalls the host behind the full compute queue, which collapses
/// the streaming pipeline.
const evict_min_size: u64 = 4 << 20;

/// A weight buffer evicted but not yet freed: its `ev` (recorded on the compute
/// stream at eviction) signals once the weight's last GEMM finishes, at which
/// point cuMemFree is safe. Deferring the free avoids a per-eviction full sync,
/// so streaming re-uploads overlap compute.
const PendingFree = struct { db: DeviceBuffer, ev: cu.CUevent };

pub const Backend = struct {
    ctx: *Context,
    gpa: std.mem.Allocator,

    // capability booleans read in dit_gpu's hot gate expressions. Start all
    // false (the opMatmul + f32-eltwise parity path); light up as kernels land.
    caps_coop: bool = false,
    caps_coop_c16: bool = false,
    caps_coop_i8_fs16: bool = false,
    caps_scores: bool = false,

    batching_on: bool = false,

    // Compute-kernel dispatch: hand-PTX (default) or the dlopen'd libraries.
    // `libs` is populated by initLibs when `kernels == .libs`.
    kernels: KernelMode = .hand_ptx,
    libs: ?Libs = null,

    // weight cache: host pointer -> uploaded device buffer (LRU-stamped for
    // streaming: under memory pressure the least-recently-used weights are
    // evicted and re-uploaded on next use — see reserveForWeights/evictOneWeight).
    weights: std.AutoHashMapUnmanaged(usize, WeightEntry) = .empty,
    use_counter: u64 = 0,
    /// --vram-budget ceiling on our own device footprint (bytes); 0 = no cap
    /// (only the live cuMemGetInfo headroom bounds the weight cache).
    budget_override: u64 = 0,
    /// Total weights evicted over the backend's lifetime. Nonzero means weight
    /// streaming is (or has been) active — device weight pointers are not
    /// stable, so anything that bakes them in (the CudaLM decode graph) must
    /// stay on per-op launches.
    evictions: u64 = 0,
    /// First-touch weight pinning: newly cached weights are pinned (immune to
    /// eviction) until their total reaches this cap; later weights stream.
    /// For a fixed repeating walk (LLM decode) this turns the LRU cliff —
    /// where any cap below full residency re-uploads EVERYTHING — into cost
    /// proportional to the streamed fraction. 0 = off. Must stay off for the
    /// diffusion pipeline: first-touch would pin the single-use text encoder
    /// and stream the whole DiT.
    pin_budget: u64 = 0,
    /// Bytes currently claimed against pin_budget.
    pinned_bytes: u64 = 0,
    /// async weight uploads: set once a prefetch thread is running, so
    /// re-uploads run on the transfer stream (overlapping compute) with a
    /// per-weight completion event the compute stream waits on.
    async_uploads: bool = false,
    /// Host ranges page-locked via enableDirectStreaming (checkpoint mmaps).
    /// Prefetches whose source lies inside one DMA straight from the mmap,
    /// skipping the staging memcpy. Fixed slots + atomic count so the
    /// prefetch thread can scan while the main thread registers another
    /// model (slot written before the count is published).
    registered: [4][]const u8 = undefined,
    n_registered: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// evicted-but-not-yet-freed weight buffers (deferred free; async path only).
    free_pending: std.ArrayListUnmanaged(PendingFree) = .empty,
    /// Bytes held by free_pending. Counted as available by reserveForWeights:
    /// deferred frees keep device_used inflated until their events signal, and
    /// without this credit the headroom loop would evict the entire unpinned
    /// cache (including just-prefetched weights) on every upload.
    pending_free_bytes: u64 = 0,
    /// Recycled streamed-weight buffers by exact size (async path). Streamed
    /// decode cycles the same handful of weight sizes every token, and
    /// cuMemAlloc/cuMemFree cost ~0.3-1 ms each under an active DMA queue —
    /// per-token churn was 2x slower than the transfers themselves. Buffers
    /// here still count in ctx.device_used (they are still held).
    weight_pool: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(DeviceBuffer)) = .empty,
    /// Total bytes in streamed circulation (unpinned cache entries + deferred
    /// frees + pool). weightBufAcquire paces the main thread against this:
    /// past stream_window it blocks on the oldest deferred free (compute
    /// progress) and recycles, instead of allocating more — bounding streamed
    /// VRAM to the window and keeping steady state free of cuMemAlloc/Free.
    streamed_bytes: u64 = 0,
    /// Cap for streamed_bytes; set alongside pin_budget by the CLI. 0 = no
    /// pacing (weights fully resident or the sync path).
    stream_window: u64 = 0,

    // ---- prefetch thread (block-ahead async weight streaming) ----
    // Lock-free single-producer (main) / single-consumer (thread) ring — Zig 0.16
    // removed std.Thread.Mutex/Condition, so we use atomics + Thread.yield spins.
    pf_thread: ?std.Thread = null,
    pf_ring: [pf_ring_sz]PrefetchReq = undefined,
    pf_head: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // enqueue cursor (main writes)
    pf_tail: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // dequeue cursor (thread writes)
    pf_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // last gen whose DMA+event is queued
    pf_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pf_gen: u64 = 0, // last assigned generation (main-only)

    // fp8-e4m3 GEMM state (opMatmulFp8): the 256-entry e4m3->f32 LUT (uploaded
    // once), and reused f16 scratch for the decoded weight + converted activations.
    // Weights stay fp8 in the cache (streaming-friendly); decoded per GEMM into
    // w16 scratch, activations converted into a16, then the validated f16 buildHgemm.
    fp8_lut: DeviceBuffer = .{},
    fp8_w16: DeviceBuffer = .{},
    fp8_a16: DeviceBuffer = .{},
    /// q8-quantized decode activation (opGemvQuantizeX / opGemvQuantQ8).
    q8_act: DeviceBuffer = .{},

    // Decode-graph state (see stateSetup): device address of g_state and the
    // graph-mode kernel entries sharing it.
    state_ptr: cu.CUdeviceptr = 0,
    f_embed_gather_s: cu.CUfunction = null,
    f_kv_append_s: cu.CUfunction = null,
    f_rope_half_s: cu.CUfunction = null,
    f_attn_split_s: cu.CUfunction = null,
    f_attn_split_h256_s: cu.CUfunction = null,
    f_embed_gather_q8_0: cu.CUfunction = null,
    f_embed_gather_q4_k: cu.CUfunction = null,
    f_embed_gather_q5_k: cu.CUfunction = null,
    f_embed_gather_q6_k: cu.CUfunction = null,

    // f16 tensor-core VAE conv scratch (opConvF16): padded f16 weight + activation
    // and the f32 GEMM output, reused across convs.
    conv_w16: DeviceBuffer = .{},
    conv_a16: DeviceBuffer = .{},
    conv_c: DeviceBuffer = .{},

    // int8 prep state (opI8Prep -> opI8Gemm contract).
    i8_x: DeviceBuffer = .{},
    i8_scale: DeviceBuffer = .{},
    i8_m: usize = 0,
    i8_mpad: usize = 0,
    i8_cols: usize = 0,
    // s32 GEMM accumulator scratch for the cuBLASLt int8 path (.libs mode): the
    // hand-PTX fused kernel folds the rescale into the C-store and needs no acc
    // buffer, but cuBLASLt emits raw s32, rescaled by a separate irescale pass.
    i8_acc: DeviceBuffer = .{},

    // tensor-core attention scratch (per-head, reused across heads/calls; grown
    // to the largest seq seen). f16 Q/K/Vt tiles, f32 scores, f16 probs, f32 out.
    attn_qh: DeviceBuffer = .{},
    attn_kh: DeviceBuffer = .{},
    attn_vth: DeviceBuffer = .{},
    attn_s: DeviceBuffer = .{},
    attn_p: DeviceBuffer = .{}, // materialized softmax probs (non-fused path only)
    attn_md: DeviceBuffer = .{}, // per-row {max, 1/sum} f32 pairs (fused path)
    attn_oh: DeviceBuffer = .{},

    // cuDNN fused-SDPA attention (.libs mode): plan cache keyed by packed
    // (heads,kv_heads,seq,hd), f16 Q/K/V/O scratch, and the SDPA workspace.
    sdpa_plans: std.AutoHashMapUnmanaged(u64, cudnn.SdpaPlan) = .empty,
    // fused int8-GEMM+dequant plans (cuDNN op graph), keyed by packed (m,n,k,d_f16).
    // When `use_fused_i8`, opI8GemmLibs uses these instead of cuBLASLt+irescale —
    // the s32 accumulator never round-trips to DRAM.
    mdq_plans: std.AutoHashMapUnmanaged(u64, cudnn.MatmulDequantPlan) = .empty,
    use_fused_i8: bool = false,
    cudnn_q16: DeviceBuffer = .{},
    cudnn_k16: DeviceBuffer = .{},
    cudnn_v16: DeviceBuffer = .{},
    cudnn_o16: DeviceBuffer = .{},
    cudnn_ws: DeviceBuffer = .{},

    // cached PTX modules / functions.
    prep_mods: std.AutoHashMapUnmanaged(usize, cu.CUfunction) = .empty, // keyed by cols
    prep_owned: std.ArrayListUnmanaged(ctxmod.Module) = .empty,
    fused_mod: ?ctxmod.Module = null,
    fused_fn: cu.CUfunction = null,
    // irescale (s32 acc * act_scale[row] * weight_scale[col] -> f32) for the
    // cuBLASLt int8 path (.libs mode).
    irescale_mod: ?ctxmod.Module = null,
    irescale_fn: cu.CUfunction = null,
    irescale_h16_mod: ?ctxmod.Module = null,
    irescale_h16_fn: cu.CUfunction = null,
    // cuBLASLt int8 matmul plans (desc + layouts + heuristic algo), keyed by
    // packed (n,m,k). Layouts hold no data pointer, so a plan is reused across
    // steps/blocks/weights of the same shape — the expensive heuristic query
    // runs once, and the timed matmul is a pure enqueue.
    lt_plans: std.AutoHashMapUnmanaged(u64, LtPlan) = .empty,
    // int4 (W4A4) variants: prep quantizes to s4 and the GEMM is m16n8k64.s4.
    i4_prep_mods: std.AutoHashMapUnmanaged(usize, cu.CUfunction) = .empty,
    i4_prep_owned: std.ArrayListUnmanaged(ctxmod.Module) = .empty,
    i4_fused_mod: ?ctxmod.Module = null,
    i4_fused_fn: cu.CUfunction = null,
    mm_mod: ?ctxmod.Module = null,
    mm_fn: cu.CUfunction = null,
    hgemm_mod: ?ctxmod.Module = null,
    hgemm_fn: cu.CUfunction = null,
    hgemm_b_mod: ?ctxmod.Module = null,
    hgemm_b_fn: cu.CUfunction = null,
    hgemm_bc16_mod: ?ctxmod.Module = null,
    hgemm_bc16_fn: cu.CUfunction = null,
    hgemm_ao_mod: ?ctxmod.Module = null, // fused attn-out P@V (P computed from S+MD)
    hgemm_ao_fn: cu.CUfunction = null,

    // tensor-core attention: batch `G` heads per launch (grid.z), G derived so the
    // scores+probs scratch stays under `attn_scratch_budget`. attn_batched=false
    // falls back to the per-head loop (A/B reference).
    attn_batched: bool = true,
    /// Fused attention output: the softmax pass emits only per-row {max, 1/sum}
    /// (softmax_md_f16, one S read) and the P@V GEMM recomputes P from S+MD during
    /// A-staging (hgemm_attnout) — no P materialization. Eliminates the P write +
    /// the extra softmax S reads (the Vulkan-parity win). Off = the materialized
    /// softmax_row_f16 → P → hgemm_batched path (kept as the A/B reference).
    attn_fused: bool = true,
    attn_scratch_budget: usize = 2 << 30, // 2 GiB for S+P

    // eltwise module cache: PTX string pointer -> function.
    elt_fns: std.AutoHashMapUnmanaged(usize, cu.CUfunction) = .empty,
    elt_mods: std.ArrayListUnmanaged(ctxmod.Module) = .empty,

    // per-category profiler (sync-per-op device timing; the plan's methodology).
    profile: bool = false,
    prof: Prof = .{},
    ptimer: ?ctxmod.Context.Timer = null,

    pub const ProfCat = enum { matmul, prep, attn, elt, attn_scores, attn_softmax, attn_pv };
    pub const Prof = struct {
        ms: [7]f64 = .{ 0, 0, 0, 0, 0, 0, 0 },
        n: [7]u32 = .{ 0, 0, 0, 0, 0, 0, 0 },
        pub fn reset(self: *Prof) void {
            self.* = .{};
        }
    };

    /// Begin timing the next op (no-op unless `profile`). Pairs with `ptoc`.
    fn ptic(self: *Backend) void {
        if (!self.profile) return;
        if (self.ptimer == null) self.ptimer = self.ctx.timerCreate() catch null;
        if (self.ptimer) |t| self.ctx.timerBegin(t) catch {};
    }
    /// End timing and accumulate into category `c` (syncs the stream — exact
    /// host timing, matching the Vulkan --profile sync-per-op mode).
    fn ptoc(self: *Backend, c: ProfCat) void {
        if (!self.profile) return;
        if (self.ptimer) |t| {
            const ms = self.ctx.timerEndMs(t) catch return;
            self.prof.ms[@intFromEnum(c)] += ms;
            self.prof.n[@intFromEnum(c)] += 1;
        }
    }

    pub fn init(gpa: std.mem.Allocator) Error!*Backend {
        const self = try gpa.create(Backend);
        errdefer gpa.destroy(self);
        const c = try gpa.create(Context);
        errdefer gpa.destroy(c);
        c.* = Context.init(gpa) catch return error.NoSuitableDevice;
        self.* = .{ .ctx = c, .gpa = gpa };
        return self;
    }

    /// Build a backend in `.libs` mode: the normal driver Context plus dlopen'd
    /// cuBLASLt + cuDNN handles bound to the compute stream. Returns an error
    /// (caller falls back to CPU) if the driver, either library, or a handle is
    /// unavailable. The CUDA context is current after `init`, which cuBLASLt /
    /// cuDNN handle creation requires.
    pub fn initLibs(gpa: std.mem.Allocator) Error!*Backend {
        const self = try Backend.init(gpa);
        errdefer self.deinit();
        self.kernels = .libs;
        // Ships OFF: the cuDNN fused int8 GEMM+dequant is bit-exact but measured
        // NEUTRAL vs cuBLASLt+irescale (cuDNN's int8 matmul is ~0.22 s/step slower
        // than cuBLASLt IMMA, exactly canceling the irescale round-trip it removes).
        // cuBLASLt IMMA is the more-proven path; the fused graph stays dormant +
        // documented (MatmulDequantPlan, cuda-libs-i8fused-test). See PLAN.md 2.7.
        self.use_fused_i8 = false;

        var lt = cublaslt.Api.load() catch return error.CudaError;
        errdefer lt.deinit();
        var lt_h: cublaslt.Handle = null;
        if (lt.cublasLtCreate(&lt_h) != cublaslt.SUCCESS) return error.CudaError;
        errdefer _ = lt.cublasLtDestroy(lt_h);

        var dnn = cudnn.Api.load() catch return error.CudaError;
        errdefer dnn.deinit();
        var dnn_h: cudnn.Handle = null;
        if (dnn.cudnnCreate(&dnn_h) != cudnn.SUCCESS) return error.CudaError;
        errdefer _ = dnn.cudnnDestroy(dnn_h);
        _ = dnn.cudnnSetStream(dnn_h, self.ctx.stream);

        const ws = try self.ctx.alloc(lt_workspace_bytes);

        self.libs = .{ .lt = lt, .lt_handle = lt_h, .dnn = dnn, .dnn_handle = dnn_h, .workspace = ws };
        return self;
    }

    pub fn deinit(self: *Backend) void {
        if (self.libs) |*L| {
            var sit = self.sdpa_plans.valueIterator();
            while (sit.next()) |p| p.deinit(&L.dnn);
            self.sdpa_plans.deinit(self.gpa);
            var mit = self.mdq_plans.valueIterator();
            while (mit.next()) |p| p.deinit(&L.dnn);
            self.mdq_plans.deinit(self.gpa);
            var pit = self.lt_plans.valueIterator();
            while (pit.next()) |p| {
                _ = L.lt.cublasLtMatmulDescDestroy(p.desc);
                _ = L.lt.cublasLtMatrixLayoutDestroy(p.adesc);
                _ = L.lt.cublasLtMatrixLayoutDestroy(p.bdesc);
                _ = L.lt.cublasLtMatrixLayoutDestroy(p.ddesc);
            }
            self.lt_plans.deinit(self.gpa);
            self.ctx.free(&L.workspace);
            if (L.dnn_handle != null) _ = L.dnn.cudnnDestroy(L.dnn_handle);
            if (L.lt_handle != null) _ = L.lt.cublasLtDestroy(L.lt_handle);
            L.dnn.deinit();
            L.lt.deinit();
            self.libs = null;
        }
        if (self.pf_thread) |t| {
            self.pf_shutdown.store(true, .release);
            t.join(); // drains queued prefetches, then exits
        }
        // Unregister page-locked checkpoint ranges after all transfer-stream
        // DMAs from them are done (the caller munmaps after Backend.deinit).
        if (self.n_registered.load(.monotonic) > 0) {
            _ = self.ctx.api.cuStreamSynchronize(self.ctx.xfer_stream);
            for (self.registered[0..self.n_registered.load(.monotonic)]) |r| self.ctx.unregisterHost(r);
        }
        self.drainPending();
        self.free_pending.deinit(self.gpa);
        self.drainPool();
        self.weight_pool.deinit(self.gpa);
        var it = self.weights.valueIterator();
        while (it.next()) |e| {
            if (e.upload_ev) |ev| self.ctx.eventDestroy(ev);
            var db = e.db;
            self.tensorDestroy(&db);
        }
        self.weights.deinit(self.gpa);
        self.prep_mods.deinit(self.gpa);
        for (self.prep_owned.items) |m| m.unload(self.ctx);
        self.prep_owned.deinit(self.gpa);
        self.i4_prep_mods.deinit(self.gpa);
        for (self.i4_prep_owned.items) |m| m.unload(self.ctx);
        self.i4_prep_owned.deinit(self.gpa);
        if (self.i4_fused_mod) |m| m.unload(self.ctx);
        if (self.fused_mod) |m| m.unload(self.ctx);
        if (self.mm_mod) |m| m.unload(self.ctx);
        if (self.hgemm_mod) |m| m.unload(self.ctx);
        if (self.hgemm_b_mod) |m| m.unload(self.ctx);
        if (self.hgemm_bc16_mod) |m| m.unload(self.ctx);
        if (self.hgemm_ao_mod) |m| m.unload(self.ctx);
        for (self.elt_mods.items) |m| m.unload(self.ctx);
        self.elt_mods.deinit(self.gpa);
        self.elt_fns.deinit(self.gpa);
        self.tensorDestroy(&self.fp8_lut);
        self.tensorDestroy(&self.fp8_w16);
        self.tensorDestroy(&self.fp8_a16);
        self.tensorDestroy(&self.q8_act);
        self.tensorDestroy(&self.conv_w16);
        self.tensorDestroy(&self.conv_a16);
        self.tensorDestroy(&self.conv_c);
        self.tensorDestroy(&self.i8_x);
        self.tensorDestroy(&self.i8_scale);
        self.tensorDestroy(&self.i8_acc);
        self.tensorDestroy(&self.cudnn_q16);
        self.tensorDestroy(&self.cudnn_k16);
        self.tensorDestroy(&self.cudnn_v16);
        self.tensorDestroy(&self.cudnn_o16);
        self.tensorDestroy(&self.cudnn_ws);
        if (self.irescale_mod) |m| m.unload(self.ctx);
        if (self.irescale_h16_mod) |m| m.unload(self.ctx);
        self.tensorDestroy(&self.attn_qh);
        self.tensorDestroy(&self.attn_kh);
        self.tensorDestroy(&self.attn_vth);
        self.tensorDestroy(&self.attn_s);
        self.tensorDestroy(&self.attn_p);
        self.tensorDestroy(&self.attn_md);
        self.tensorDestroy(&self.attn_oh);
        if (self.ptimer) |t| self.ctx.timerDestroy(t);
        self.ctx.deinit();
        self.gpa.destroy(self.ctx);
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub fn deviceName(self: *Backend) []const u8 {
        return self.ctx.deviceName();
    }

    /// Enable async weight streaming: allocate a pinned staging ring and spawn a
    /// prefetch thread. The DiT driver prefetches block N+1's weights (via
    /// prefetchWeight) while block N computes; the thread does the mmap→pinned
    /// memcpy + async DMA off the main thread, so the upload overlaps compute.
    /// No-op / falls back to synchronous uploads if pinning or the thread fails.
    /// 128 MiB slots cover the largest DiT weight (mlp ~101 MiB).
    pub fn enableAsyncStreaming(self: *Backend) void {
        if (!self.ctx.initStaging(128 << 20)) return;
        self.pf_thread = std.Thread.spawn(.{}, prefetchLoop, .{self}) catch return;
        self.async_uploads = true;
    }

    /// Enable direct async weight streaming from `bytes` (a checkpoint mmap):
    /// page-lock the range and spawn the prefetch thread. Prefetched weights
    /// then DMA straight from the mmap on the transfer stream at full PCIe
    /// bandwidth. (The staging-ring path above measured SLOWER than sync
    /// uploads for the DiT — its extra host memcpy caps throughput below the
    /// driver's pageable copy. Direct DMA has no host copy.) Registration
    /// faults the range in and pins that much host RAM; no-op on failure
    /// (uploads stay synchronous). Call once per checkpoint, before decode.
    pub fn enableDirectStreaming(self: *Backend, bytes: []const u8) void {
        const n = self.n_registered.load(.monotonic);
        if (n == self.registered.len) return;
        if (!self.ctx.registerHost(bytes)) return;
        self.registered[n] = bytes;
        self.n_registered.store(n + 1, .release);
        if (self.pf_thread == null) {
            self.pf_thread = std.Thread.spawn(.{}, prefetchLoop, .{self}) catch null;
        }
        self.async_uploads = self.pf_thread != null;
    }

    /// Whether `bytes` lies inside a page-locked range (prefetch thread or main).
    fn isRegistered(self: *Backend, bytes: []const u8) bool {
        const a = @intFromPtr(bytes.ptr);
        for (self.registered[0..self.n_registered.load(.acquire)]) |r| {
            const r0 = @intFromPtr(r.ptr);
            if (a >= r0 and a + bytes.len <= r0 + r.len) return true;
        }
        return false;
    }

    /// Prefetch-thread body: drain the request ring, uploading each weight through
    /// the pinned staging ring on the transfer stream (blocks THIS thread on the
    /// mmap→pinned memcpy and slot reuse — never the main thread). Advances
    /// pf_completed so consumers know a weight's upload event is recorded.
    fn prefetchLoop(self: *Backend) void {
        _ = self.ctx.api.cuCtxSetCurrent(self.ctx.ctx);
        while (true) {
            const tail = self.pf_tail.load(.monotonic);
            const head = self.pf_head.load(.acquire); // publishes the ring slot
            if (tail == head) {
                if (self.pf_shutdown.load(.acquire)) return; // drained + shutdown
                std.Thread.yield() catch {};
                continue;
            }
            const req = self.pf_ring[tail % pf_ring_sz];
            const cb = ctxmod.Buffer{ .ptr = req.db.ptr(), .bytes = @intCast(req.db.size) };
            if (self.isRegistered(req.bytes)) {
                self.ctx.uploadDirect(cb, req.bytes, req.ev) catch {};
            } else {
                self.ctx.uploadStaged(cb, req.bytes, req.ev) catch {};
            }
            self.pf_tail.store(tail + 1, .release);
            self.pf_completed.store(req.gen, .release); // FIFO: gens complete in order
        }
    }

    /// Upload a weight through the cache immediately (pinned while pin_budget
    /// has room). Lets a caller give specific weights first claim on the pin
    /// budget before the main model's first forward — e.g. the spec-decode
    /// draft model, whose weights are read once per drafted token.
    pub fn warmWeight(self: *Backend, bytes: []const u8) Error!void {
        _ = try self.cachedWeight(bytes);
    }

    /// Queue a weight upload for the prefetch thread (main thread). Allocates the
    /// device buffer + event now (touching the cache/budget, which is main-only)
    /// and enqueues the mmap→device copy for the thread. No-op if already cached.
    pub fn prefetchWeight(self: *Backend, bytes: []const u8) void {
        if (!self.async_uploads) return;
        const key = @intFromPtr(bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return; // already resident or in flight
        }
        self.reserveForWeights(bytes.len);
        const pin = self.pinNew(bytes.len);
        const db = self.weightBufAcquire(bytes.len, pin) catch return; // best-effort; cachedWeight sync-falls-back on miss
        const ev = self.ctx.eventCreate() catch {
            if (pin) self.streamed_bytes += db.size; // returning to circulation
            self.poolPut(db);
            return;
        };
        const head = self.pf_head.load(.monotonic);
        if (head - self.pf_tail.load(.acquire) >= pf_ring_sz) { // ring full — drop (sync fallback later)
            self.ctx.eventDestroy(ev);
            if (pin) self.streamed_bytes += db.size; // returning to circulation
            self.poolPut(db);
            return;
        }
        self.pf_gen += 1;
        const gen = self.pf_gen;
        self.pf_ring[head % pf_ring_sz] = .{ .bytes = bytes, .db = db, .ev = ev, .gen = gen };
        self.pf_head.store(head + 1, .release); // publish the slot to the thread
        self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = pin, .awaiting_use = true, .upload_ev = ev, .pf_gen = gen }) catch {};
    }

    // ---- buffers ------------------------------------------------------------

    pub fn tensorCreate(self: *Backend, size: u64) Error!DeviceBuffer {
        while (true) {
            if (self.ctx.alloc(@intCast(size))) |b| {
                return dbFromPtr(b.ptr, size);
            } else |err| {
                if (err != error.DeviceOutOfMemory) return error.CudaError;
                // Reactive backstop: reclaim deferred-frees (blocking the oldest if
                // needed), else evict a resident weight, then retry. Degrades to
                // streaming instead of failing.
                self.reclaimPending();
                if (self.blockOldestPending()) continue;
                if (self.evictOneWeight()) continue;
                return error.DeviceOutOfMemory;
            }
        }
    }

    /// Synchronously drain all deferred-frees (teardown / evictWeights).
    fn drainPending(self: *Backend) void {
        for (self.free_pending.items) |pf| {
            _ = self.ctx.api.cuEventSynchronize(pf.ev);
            self.ctx.eventDestroy(pf.ev);
            var db = pf.db;
            self.tensorDestroy(&db);
        }
        self.free_pending.clearRetainingCapacity();
        self.pending_free_bytes = 0;
    }

    /// Headroom for a new weight upload (bytes). Bounded by the live device-free
    /// query (cuMemGetInfo — sees other processes) and, if set, our --vram-budget
    /// ceiling minus what we already hold.
    fn budgetHeadroom(self: *Backend) u64 {
        const live = (self.ctx.memGetInfo().free) * 9 / 10; // 10% margin (frag/overhead)
        if (self.budget_override != 0) {
            return @min(self.budget_override -| self.ctx.device_used, live);
        }
        return live;
    }

    /// Free any deferred-free buffers whose last-GEMM event has signaled (async
    /// path). Non-blocking (cuEventQuery); called at the top of reserveForWeights.
    fn reclaimPending(self: *Backend) void {
        var i: usize = 0;
        while (i < self.free_pending.items.len) {
            const pf = self.free_pending.items[i];
            if (self.ctx.api.cuEventQuery(pf.ev) == cu.CUDA_SUCCESS) {
                self.ctx.eventDestroy(pf.ev);
                self.pending_free_bytes -= pf.db.size;
                self.poolPut(pf.db); // recycle — cuMemFree under DMA load costs ~1 ms
                _ = self.free_pending.swapRemove(i);
            } else i += 1;
        }
    }

    /// Block on the oldest deferred-free buffer and reclaim it (the reactive path
    /// when a fresh alloc physically OOMs). Returns false if none pending.
    fn blockOldestPending(self: *Backend) bool {
        if (self.free_pending.items.len == 0) return false;
        const pf = self.free_pending.orderedRemove(0);
        _ = self.ctx.api.cuEventSynchronize(pf.ev);
        self.ctx.eventDestroy(pf.ev);
        self.pending_free_bytes -= pf.db.size;
        self.streamed_bytes -|= pf.db.size;
        var db = pf.db;
        self.tensorDestroy(&db);
        return true;
    }

    /// Evict the least-recently-used cached weight (false if the cache holds only
    /// the protected MRU).
    ///
    /// CRITICAL: never evict the MOST-recently-used weight. An op (e.g. opI8Gemm)
    /// fetches its weight, then fetches weight_scale — that second fetch's reserve
    /// must not free the weight the imminent kernel reads. Unlike Vulkan (where
    /// eviction flushes recorded ops already bound to the weight), CUDA launches
    /// after both fetches, so the just-fetched weight is protected here instead.
    ///
    /// Async path: DEFER the free (record a compute-stream event; reclaim once the
    /// weight's last GEMM signals) so eviction doesn't stall the pipeline — the
    /// buffer lingers a moment (soft-over-budget, physical VRAM has room) and the
    /// next weight uploads concurrently. Sync path: cuStreamSynchronize then free.
    fn evictOneWeight(self: *Backend) bool {
        // In-flight prefetches (pf_gen > pf_completed) must not be freed — the
        // prefetch thread is about to DMA into them.
        const completed = self.pf_completed.load(.acquire);
        var mru_use: u64 = 0;
        var it0 = self.weights.valueIterator();
        while (it0.next()) |e| {
            if (e.last_use > mru_use) mru_use = e.last_use;
        }
        var lru_key: usize = undefined;
        var lru_use: u64 = std.math.maxInt(u64);
        var it = self.weights.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.pinned) continue; // pinned prefix never streams
            if (e.value_ptr.awaiting_use) continue; // prefetched, not yet consumed
            if (e.value_ptr.last_use == mru_use) continue; // protect the MRU
            if (e.value_ptr.pf_gen > completed) continue; // protect in-flight prefetch
            if (e.value_ptr.db.size <= evict_min_size) continue; // norms/scales: not worth a sync stall
            if (e.value_ptr.last_use < lru_use) {
                lru_use = e.value_ptr.last_use;
                lru_key = e.key_ptr.*;
            }
        }
        if (lru_use == std.math.maxInt(u64)) return false; // only the MRU remains
        self.evictions += 1;
        const e = self.weights.fetchRemove(lru_key).?;
        if (e.value.upload_ev) |ev| {
            // The buffer may still be the target of an in-flight prefetch DMA
            // (evicted before any GEMM consumed it): make the compute stream
            // wait on the upload event first, so the deferred-free event below
            // cannot signal before the transfer-stream write has finished.
            self.ctx.computeWaitEvent(ev) catch {};
            self.ctx.eventDestroy(ev);
        }
        const db = e.value.db;
        if (self.async_uploads) {
            if (self.ctx.eventCreate()) |rev| {
                _ = self.ctx.api.cuEventRecord(rev, self.ctx.stream); // after this weight's last GEMM
                if (self.free_pending.append(self.gpa, .{ .db = db, .ev = rev })) |_| {
                    self.pending_free_bytes += db.size;
                    return true; // deferred free; reclaimed lazily
                } else |_| self.ctx.eventDestroy(rev);
            } else |_| {}
        }
        // sync path (async off, or event/append failed): a launch may still
        // reference the buffer, so synchronize before freeing.
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.stream);
        self.streamed_bytes -|= db.size;
        var d = db;
        self.tensorDestroy(&d);
        return true;
    }

    /// Get a device buffer for a weight upload. Exact-size reuse from the
    /// recycling pool first — streamed decode cycles the same handful of
    /// sizes every token, and cuMemAlloc/cuMemFree under an active DMA queue
    /// cost more than the transfers they serve. Past stream_window, block on
    /// the oldest deferred free (i.e. wait for compute to consume the oldest
    /// streamed weight) and recycle it: this paces the enqueue against
    /// compute progress and bounds streamed VRAM to the window. Falls back
    /// to a fresh allocation (soft window) when nothing is pending.
    fn weightBufAcquire(self: *Backend, size: u64, pinned: bool) Error!DeviceBuffer {
        if (self.poolPop(size)) |db| {
            if (pinned) self.streamed_bytes -|= size; // leaves streamed circulation
            return db;
        }
        if (!pinned and self.stream_window != 0) {
            while (self.streamed_bytes + size > self.stream_window) {
                if (self.blockOldestRecycle()) {
                    if (self.poolPop(size)) |db| return db;
                    continue;
                }
                // Nothing pending: push a consumed weight into the deferred
                // queue so the next iteration can wait on and recycle it.
                if (!self.evictOneWeight()) break; // nothing reclaimable: soft overshoot
            }
        }
        const db = try self.tensorCreate(size);
        if (!pinned) self.streamed_bytes += size;
        return db;
    }

    fn poolPop(self: *Backend, size: u64) ?DeviceBuffer {
        const list = self.weight_pool.getPtr(size) orelse return null;
        if (list.items.len == 0) return null;
        list.items.len -= 1;
        return list.items.ptr[list.items.len];
    }

    /// Return a buffer to the recycling pool (real-freed only if bookkeeping
    /// allocation fails).
    fn poolPut(self: *Backend, db: DeviceBuffer) void {
        const g = self.weight_pool.getOrPut(self.gpa, db.size) catch return self.poolPutFailed(db);
        if (!g.found_existing) g.value_ptr.* = .empty;
        g.value_ptr.append(self.gpa, db) catch return self.poolPutFailed(db);
    }

    fn poolPutFailed(self: *Backend, db: DeviceBuffer) void {
        self.streamed_bytes -|= db.size;
        var d = db;
        self.tensorDestroy(&d);
    }

    /// Wait for the oldest deferred free's event (compute progress) and move
    /// its buffer to the recycling pool. False if nothing is pending.
    fn blockOldestRecycle(self: *Backend) bool {
        if (self.free_pending.items.len == 0) return false;
        const pf = self.free_pending.orderedRemove(0);
        _ = self.ctx.api.cuEventSynchronize(pf.ev);
        self.ctx.eventDestroy(pf.ev);
        self.pending_free_bytes -= pf.db.size;
        self.poolPut(pf.db);
        return true;
    }

    /// Free every pooled buffer for real (evictWeights / deinit).
    fn drainPool(self: *Backend) void {
        var it = self.weight_pool.valueIterator();
        while (it.next()) |list| {
            for (list.items) |db| {
                var d = db;
                self.tensorDestroy(&d);
            }
            list.deinit(self.gpa);
        }
        self.weight_pool.clearRetainingCapacity();
    }

    /// Make room for a `need`-byte weight upload: reclaim ready deferred-frees, then
    /// evict LRU weights while the budget lacks headroom. On the DiT's fixed block
    /// walk this becomes sequential weight streaming; createBuffer's OOM retry is
    /// the reactive backstop if this can't free enough.
    fn reserveForWeights(self: *Backend, need: u64) void {
        self.reclaimPending();
        // pending_free_bytes: deferred frees WILL come back without another
        // eviction; counting them stops the loop from over-evicting while
        // their events are still in flight.
        while (self.budgetHeadroom() + self.pending_free_bytes < need) {
            if (!self.evictOneWeight()) return;
        }
    }

    pub fn tensorDestroy(self: *Backend, db: *DeviceBuffer) void {
        if (db.buf != .null_handle) {
            var b = ctxmod.Buffer{ .ptr = db.ptr(), .bytes = @intCast(db.size) };
            self.ctx.free(&b);
        }
        db.* = .{};
    }

    pub fn ensureDeviceBuffer(self: *Backend, db: *DeviceBuffer, size: u64) Error!void {
        if (db.size >= size and db.buf != .null_handle) return;
        self.tensorDestroy(db);
        db.* = try self.tensorCreate(size);
    }

    pub fn tensorUpload(self: *Backend, db: DeviceBuffer, bytes: []const u8) Error!void {
        const b = ctxmod.Buffer{ .ptr = db.ptr(), .bytes = @intCast(db.size) };
        self.ctx.upload(b, bytes) catch return error.CudaError;
    }

    pub fn tensorDownload(self: *Backend, db: DeviceBuffer, out: []u8) Error!void {
        const b = ctxmod.Buffer{ .ptr = db.ptr(), .bytes = @intCast(db.size) };
        self.ctx.download(b, out) catch return error.CudaError;
    }

    pub fn tensorCopy(self: *Backend, dst: DeviceBuffer, dst_off: u64, src: DeviceBuffer, src_off: u64, size: u64) Error!void {
        self.ctx.check(self.ctx.api.cuMemcpyDtoD(dst.ptr() + dst_off, src.ptr() + src_off, @intCast(size)), "cuMemcpyDtoD") catch return error.CudaError;
    }

    /// Cached small upload keyed by host pointer; returns the raw device handle.
    pub fn smallBuffer(self: *Backend, bytes: []const u8) Error!Handle {
        return (try self.cachedWeight(bytes)).buf;
    }

    /// Free the tensor-core attention scratch (the ~seq² scores plane dominates).
    /// The VAE calls this after its mid-block attention so the plane doesn't stay
    /// resident through the 8× upsampling. Syncs first (recorded ops may read it).
    pub fn freeAttnScratch(self: *Backend) void {
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.stream);
        self.tensorDestroy(&self.attn_qh);
        self.tensorDestroy(&self.attn_kh);
        self.tensorDestroy(&self.attn_vth);
        self.tensorDestroy(&self.attn_s);
        self.tensorDestroy(&self.attn_p);
        self.tensorDestroy(&self.attn_md);
        self.tensorDestroy(&self.attn_oh);
    }

    pub fn evictWeights(self: *Backend) void {
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.stream);
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.xfer_stream); // in-flight prefetch DMAs
        self.drainPending();
        var it = self.weights.valueIterator();
        while (it.next()) |e| {
            if (e.upload_ev) |ev| self.ctx.eventDestroy(ev);
            var db = e.db;
            self.tensorDestroy(&db);
        }
        self.weights.clearRetainingCapacity();
        self.drainPool();
        self.pinned_bytes = 0;
        self.streamed_bytes = 0;
    }

    /// Upload+cache a weight blob (int8 weights / scales), keyed by host pointer.
    /// Under memory pressure the LRU weight is evicted to make room; this weight
    /// re-uploads here on its next use (the host bytes are the mmap'd checkpoint).
    /// When async_uploads is on, the (re)upload runs on the transfer stream and
    /// the compute stream waits on a per-weight event before the dependent GEMM —
    /// so a re-upload overlaps the previous op's compute.
    fn cachedWeight(self: *Backend, bytes: []const u8) Error!DeviceBuffer {
        const key = @intFromPtr(bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            e.awaiting_use = false; // consumed: the prefetch paid off, evictable again
            const gen = e.pf_gen;
            const ev = e.upload_ev;
            const db = e.db;
            // If prefetched (in flight), wait until the thread has queued its DMA +
            // recorded the event (block-ahead: usually already done, no wait). The
            // map isn't mutated during the wait (single main thread), so `e` stays
            // valid — but we already copied what we need.
            if (gen != 0) {
                while (self.pf_completed.load(.acquire) < gen) std.Thread.yield() catch {};
            }
            if (ev) |x| self.ctx.computeWaitEvent(x) catch {};
            return db;
        }
        // Miss (not prefetched — e.g. the f32 first/last layers): plain synchronous
        // upload. The staging ring is used ONLY by the prefetch thread, so the main
        // thread never touches it here (no race).
        self.reserveForWeights(bytes.len);
        const pin = self.pinNew(bytes.len);
        const db = try self.weightBufAcquire(bytes.len, pin);
        try self.tensorUpload(db, bytes);
        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = pin });
        return db;
    }

    /// Claim pin residency for a newly cached `size`-byte weight (first-touch
    /// order): true while the claims fit under pin_budget.
    fn pinNew(self: *Backend, size: u64) bool {
        if (self.pinned_bytes + size > self.pin_budget) return false;
        self.pinned_bytes += size;
        return true;
    }

    // ---- submission (single-stream: batch == stream) ------------------------

    pub fn beginBatch(self: *Backend) Error!void {
        self.batching_on = true;
    }
    pub fn endBatch(self: *Backend) Error!void {
        self.batching_on = false;
        self.ctx.synchronize() catch return error.CudaError;
    }
    pub fn abortBatch(self: *Backend) void {
        self.batching_on = false;
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.stream);
    }
    pub fn independent(self: *Backend, n: usize) void {
        _ = self;
        _ = n; // no-op: single-stream ordering is a superset of the elided barriers
    }
    pub fn batching(self: *Backend) bool {
        return self.batching_on;
    }

    // ---- int8 GEMM pair -----------------------------------------------------

    fn prepFn(self: *Backend, cols: usize, in_f16: bool) Error!cu.CUfunction {
        const key = cols | (if (in_f16) @as(usize, 1) << 40 else 0); // f16-input variant keyed separately
        if (self.prep_mods.get(key)) |f| return f;
        const ptx = kernels.buildPrep(self.gpa, cols, 8, in_f16) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        const f = mod.getFunction(self.ctx, "iprep") catch return error.CudaError;
        self.ctx.setMaxDynamicShared(f, kernels.prepSharedBytes(cols)) catch return error.CudaError;
        self.prep_owned.append(self.gpa, mod) catch return error.OutOfMemory;
        self.prep_mods.put(self.gpa, key, f) catch return error.OutOfMemory;
        return f;
    }

    fn fusedFn(self: *Backend) Error!cu.CUfunction {
        if (self.fused_mod != null) return self.fused_fn;
        const ptx = kernels.buildIgemmPipe(self.gpa, 64, true, 8) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.fused_fn = mod.getFunction(self.ctx, "igemm_pipe_fused") catch return error.CudaError;
        self.fused_mod = mod;
        return self.fused_fn;
    }

    fn matmulFn(self: *Backend) Error!cu.CUfunction {
        if (self.mm_mod != null) return self.mm_fn;
        var mod = self.ctx.loadModule(kernels.f32gemm_ptx) catch return error.CudaError;
        self.mm_fn = mod.getFunction(self.ctx, "f32gemm") catch return error.CudaError;
        self.mm_mod = mod;
        return self.mm_fn;
    }

    fn hgemmFn(self: *Backend) Error!cu.CUfunction {
        if (self.hgemm_mod != null) return self.hgemm_fn;
        const ptx = kernels.buildHgemm(self.gpa, false, false, false) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.hgemm_fn = mod.getFunction(self.ctx, "hgemm") catch return error.CudaError;
        self.hgemm_mod = mod;
        return self.hgemm_fn;
    }

    fn hgemmBatchedFn(self: *Backend) Error!cu.CUfunction {
        if (self.hgemm_b_mod != null) return self.hgemm_b_fn;
        const ptx = kernels.buildHgemm(self.gpa, true, false, false) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.hgemm_b_fn = mod.getFunction(self.ctx, "hgemm_batched") catch return error.CudaError;
        self.hgemm_b_mod = mod;
        return self.hgemm_b_fn;
    }

    /// Batched hgemm with f16 C output (scores → softmax path).
    fn hgemmBatchedC16Fn(self: *Backend) Error!cu.CUfunction {
        if (self.hgemm_bc16_mod != null) return self.hgemm_bc16_fn;
        const ptx = kernels.buildHgemm(self.gpa, true, true, false) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.hgemm_bc16_fn = mod.getFunction(self.ctx, "hgemm_batched_c16") catch return error.CudaError;
        self.hgemm_bc16_mod = mod;
        return self.hgemm_bc16_fn;
    }

    /// Fused attention-output GEMM: O = P@V where P = exp(S-max)/sum is recomputed
    /// from S (f16) + the per-row MD table during A-staging (no P materialization).
    fn hgemmAttnOutFn(self: *Backend) Error!cu.CUfunction {
        if (self.hgemm_ao_mod != null) return self.hgemm_ao_fn;
        const ptx = kernels.buildHgemm(self.gpa, true, false, true) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.hgemm_ao_fn = mod.getFunction(self.ctx, "hgemm_attnout") catch return error.CudaError;
        self.hgemm_ao_mod = mod;
        return self.hgemm_ao_fn;
    }

    /// Plain f32 GEMM: y[m][rows] = scale*(x[m][cols] @ Wᵀ) (+bias). y_off/x_off
    /// are BYTE offsets to the first row. Only f32 weights (dtype_f8=false) are
    /// supported here — the int8 checkpoint's non-int8 layers (first/last) are f32.
    pub fn opMatmul(self: *Backend, y: DeviceBuffer, y_off: u64, x: DeviceBuffer, x_off: u64, m: usize, w_bytes: []const u8, dtype_f8: bool, rows: usize, cols: usize, scale: f32, bias: ?[]const f32) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(!dtype_f8); // fp8 path (fp8 checkpoint) not built on CUDA yet
        const f = try self.matmulFn();
        const w_db = try self.cachedWeight(w_bytes);
        var bias_ptr: cu.CUdeviceptr = w_db.ptr(); // dummy (unread when hasbias=0)
        var hasbias: u32 = 0;
        if (bias) |bb| {
            const bdb = try self.cachedWeight(std.mem.sliceAsBytes(bb));
            bias_ptr = bdb.ptr();
            hasbias = 1;
        }
        const total: u32 = @intCast(m * rows);
        var py = y.ptr();
        var px = x.ptr();
        var pw = w_db.ptr();
        var prows: u32 = @intCast(rows);
        var pcols: u32 = @intCast(cols);
        var pscale: f32 = scale;
        var pyoff: u32 = @intCast(y_off / 4);
        var pxoff: u32 = @intCast(x_off / 4);
        var pg = [_]?*anyopaque{
            @ptrCast(&py),    @ptrCast(&px),    @ptrCast(&pw),      @ptrCast(&bias_ptr),
            @ptrCast(&prows), @ptrCast(&pcols), @ptrCast(@constCast(&total)), @ptrCast(&pscale),
            @ptrCast(&pyoff), @ptrCast(&pxoff), @ptrCast(&hasbias),
        };
        self.ctx.launch(f, .{ (total + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, 0, &pg) catch return error.CudaError;
    }

    /// fp8-e4m3 GEMM: y[m][rows] f32 = x[m][cols] f32 @ Wᵀ, W fp8-e4m3 [rows][cols]
    /// with a per-tensor `scale`. The fp8 weight streams through the cache; it is
    /// decoded to an f16 scratch (dequant_fp8_f16, scale folded), the activations
    /// are converted to f16 with the m→128 pad zeroed, and the validated f16
    /// buildHgemm produces the f32 output. rows,cols must be multiples of 128,32.
    /// (No y_off / bias — the text encoder writes each GEMM to its own buffer.)
    pub fn opMatmulFp8(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, m: usize, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        if (self.fp8_lut.buf == .null_handle) {
            self.fp8_lut = try self.tensorCreate(256 * 4);
            try self.tensorUpload(self.fp8_lut, std.mem.sliceAsBytes(&dtypes.f8_e4m3_to_f32_table));
        }
        const w_db = try self.cachedWeight(w_bytes); // fp8 bytes, streams via the cache
        const mpad = std.mem.alignForward(usize, m, 128);
        try self.ensureDeviceBuffer(&self.fp8_w16, rows * cols * 2);
        try self.ensureDeviceBuffer(&self.fp8_a16, mpad * cols * 2);
        const f_deq = try self.eltFn(elt.dequant_fp8_f16_ptx, "dequant_fp8_f16");
        try self.eltLaunch(f_deq, w_db, self.fp8_lut, self.fp8_w16, null, .{ @intCast(rows * cols), 0, 0, 0, 0, 0 }, .{ scale, 0 }, rows * cols);
        const f_cvt = try self.eltFn(elt.f32_to_f16_ptx, "f32_to_f16");
        try self.eltLaunch(f_cvt, x, self.fp8_a16, null, null, .{ @intCast(mpad * cols), @intCast(m * cols), 0, 0, 0, 0 }, .{ 0, 0 }, mpad * cols);
        // C[mpad][rows] = A[mpad][cols] @ B[rows][cols]ᵀ  (m=mpad, n=rows, k=cols)
        if (self.kernels == .libs) {
            try self.ltMatmulF16(y, self.fp8_w16, self.fp8_a16, rows, mpad, cols);
        } else {
            const f_hg = try self.hgemmFn();
            try self.launchHgemm(f_hg, self.fp8_a16, self.fp8_w16, y, mpad, rows, cols);
        }
    }

    /// Fused fp8 GEMV for m=1 decode: y[rows] f32 = scale * (W fp8 [rows][cols] @ x).
    /// One 256-thread block per row, inline LUT dequant — no f16 scratch, each
    /// weight byte is read exactly once (memory-bound optimal for decode).
    pub fn opGemvFp8(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 8 == 0);
        if (self.fp8_lut.buf == .null_handle) {
            self.fp8_lut = try self.tensorCreate(256 * 4);
            try self.tensorUpload(self.fp8_lut, std.mem.sliceAsBytes(&dtypes.f8_e4m3_to_f32_table));
        }
        const w_db = try self.cachedWeight(w_bytes);
        const f = try self.eltFn(elt.gemv_fp8_ptx, "gemv_fp8");
        try self.rowLaunch(f, w_db, x, y, self.fp8_lut, .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0 }, .{ scale, 0 }, rows);
    }

    /// Multi-input fused fp8 GEMV: y[i][rows] f32 = scale * (W @ x_i) for
    /// i < n (n <= 4), reading the fp8 weight once for all inputs — the
    /// small-batch regime (speculative verify, short multi-turn prefills)
    /// where opMatmulFp8's dequant-to-f16-scratch round trip costs ~5x the
    /// weight traffic. x must have 4 rows of backing store; rows beyond n
    /// may be garbage (their outputs are predicated off).
    pub fn opGemvFp8N(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize, n: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 8 == 0 and rows % 8 == 0 and n >= 1 and n <= 4);
        if (self.fp8_lut.buf == .null_handle) {
            self.fp8_lut = try self.tensorCreate(256 * 4);
            try self.tensorUpload(self.fp8_lut, std.mem.sliceAsBytes(&dtypes.f8_e4m3_to_f32_table));
        }
        const w_db = try self.cachedWeight(w_bytes);
        const f = try self.eltFn(elt.gemv_fp8n_ptx, "gemv_fp8n");
        try self.rowLaunch(f, w_db, x, y, self.fp8_lut, .{ @intCast(rows), @intCast(cols), @intCast(n), 0, 0, 0 }, .{ scale, 0 }, rows / 8);
    }

    /// Fused ggml block-quant GEMV for m=1 decode: y[rows] f32 =
    /// scale * (W quant [rows][cols] @ x), inline dequant — each weight byte
    /// read exactly once (memory-bound optimal, like opGemvFp8). cols must be
    /// a whole number of blocks; the k-quant kernels stage per-sub-block
    /// scales in an 8 KiB shared table, capping cols at 32768.
    pub fn opGemvQuant(self: *Backend, dt: dtypes.DType, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % dt.blockElems() == 0 and cols <= 32768);
        const w_db = try self.cachedWeight(w_bytes);
        const f = switch (dt) {
            .q8_0 => try self.eltFn(elt.gemv_q8_0_ptx, "gemv_q8_0"),
            .q4_k => try self.eltFn(elt.gemv_q4_k_ptx, "gemv_q4_k"),
            .q5_k => try self.eltFn(elt.gemv_q5_k_ptx, "gemv_q5_k"),
            .q6_k => try self.eltFn(elt.gemv_q6_k_ptx, "gemv_q6_k"),
            else => unreachable,
        };
        // q5_k/q6_k run warp-per-row (8 rows per block).
        const warp_per_row = dt == .q5_k or dt == .q6_k;
        if (warp_per_row) std.debug.assert(rows % 8 == 0);
        const grid = if (warp_per_row) rows / 8 else rows;
        try self.rowLaunch(f, w_db, x, y, null, .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0 }, .{ scale, 0 }, grid);
    }

    /// Quantize a decode activation vector x (f32[cols]) into the shared q8
    /// scratch (SoA: f32 d[cols/32] then i8 qs[cols], so the GEMVs load
    /// vectors) for the dp4a GEMV path. Call once per distinct x, then any
    /// number of opGemvQuantQ8.
    pub fn opGemvQuantizeX(self: *Backend, x: DeviceBuffer, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        std.debug.assert(cols % 32 == 0);
        const nblk = cols / 32;
        try self.ensureDeviceBuffer(&self.q8_act, nblk * 4 + cols);
        const f = try self.eltFn(elt.quantize_q8_1_ptx, "quantize_q8_1");
        try self.rowLaunch(f, x, self.q8_act, null, null, .{ @intCast(nblk), 0, 0, 0, 0, 0 }, .{ 0, 0 }, (nblk + 7) / 8);
    }

    /// dp4a block-quant GEMV for m=1 decode against the q8 activation
    /// written by opGemvQuantizeX: y[rows] f32 = scale * (W quant @ x̂).
    /// Integer dot products (llama.cpp mmvq math) — ~2.5x less ALU than
    /// opGemvQuant's f32 path, which stays as the q8_0/q4_k fallback.
    pub fn opGemvQuantQ8(self: *Backend, dt: dtypes.DType, y: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 256 == 0 and rows % 8 == 0);
        std.debug.assert(self.q8_act.size >= cols / 32 * 4 + cols);
        const w_db = try self.cachedWeight(w_bytes);
        const f = switch (dt) {
            .q5_k => try self.eltFn(elt.gemv_q5_k_q8_ptx, "gemv_q5_k_q8"),
            .q6_k => try self.eltFn(elt.gemv_q6_k_q8_ptx, "gemv_q6_k_q8"),
            else => unreachable,
        };
        try self.rowLaunch(f, w_db, self.q8_act, y, null, .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0 }, .{ scale, 0 }, rows / 8);
    }

    /// Grouped dp4a GEMV for small-batch prefill: y[i][rows] f32 = scale *
    /// (W quant @ x̂_(row_off+i)) for ng <= 8 activation rows staged by ONE
    /// opGemvQuantizeX(x, n*cols) — the weight streams once per pass instead
    /// of the dequant-to-f16 GEMM's ~6.5x traffic for small chunks.
    pub fn opGemvQuantQ8N(self: *Backend, dt: dtypes.DType, y: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize, ng: usize, row_off: usize, n_total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 256 == 0 and rows % 8 == 0);
        std.debug.assert(ng >= 1 and ng <= 8 and row_off + ng <= n_total);
        std.debug.assert(self.q8_act.size >= n_total * cols / 32 * 4 + n_total * cols);
        const w_db = try self.cachedWeight(w_bytes);
        const f = switch (dt) {
            .q5_k => try self.eltFn(elt.gemv_q5_k_q8n_ptx, "gemv_q5_k_q8n"),
            .q6_k => try self.eltFn(elt.gemv_q6_k_q8n_ptx, "gemv_q6_k_q8n"),
            else => unreachable,
        };
        try self.rowLaunch(f, w_db, self.q8_act, y, null, .{ @intCast(rows), @intCast(cols), @intCast(ng), @intCast(row_off), @intCast(n_total * cols / 32), 0 }, .{ scale, 0 }, rows / 8);
    }

    /// ggml block-quant GEMM (prefill): the opMatmulFp8 shape — dequant the
    /// weight to the shared f16 scratch, convert/pad the activations, run the
    /// f16 tensor-core GEMM. rows,cols must be multiples of 128,32.
    pub fn opMatmulQuant(self: *Backend, dt: dtypes.DType, y: DeviceBuffer, x: DeviceBuffer, m: usize, w_bytes: []const u8, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        const w_db = try self.cachedWeight(w_bytes);
        const mpad = std.mem.alignForward(usize, m, 128);
        try self.ensureDeviceBuffer(&self.fp8_w16, rows * cols * 2);
        try self.ensureDeviceBuffer(&self.fp8_a16, mpad * cols * 2);
        const f_deq = switch (dt) {
            .q8_0 => try self.eltFn(elt.dequant_q8_0_f16_ptx, "dequant_q8_0_f16"),
            .q4_k => try self.eltFn(elt.dequant_q4_k_f16_ptx, "dequant_q4_k_f16"),
            .q5_k => try self.eltFn(elt.dequant_q5_k_f16_ptx, "dequant_q5_k_f16"),
            .q6_k => try self.eltFn(elt.dequant_q6_k_f16_ptx, "dequant_q6_k_f16"),
            else => unreachable,
        };
        try self.eltLaunch(f_deq, w_db, self.fp8_w16, null, null, .{ @intCast(rows * cols), 0, 0, 0, 0, 0 }, .{ 0, 0 }, rows * cols);
        const f_cvt = try self.eltFn(elt.f32_to_f16_ptx, "f32_to_f16");
        try self.eltLaunch(f_cvt, x, self.fp8_a16, null, null, .{ @intCast(mpad * cols), @intCast(m * cols), 0, 0, 0, 0 }, .{ 0, 0 }, mpad * cols);
        if (self.kernels == .libs) {
            try self.ltMatmulF16(y, self.fp8_w16, self.fp8_a16, rows, mpad, cols);
        } else {
            const f_hg = try self.hgemmFn();
            try self.launchHgemm(f_hg, self.fp8_a16, self.fp8_w16, y, mpad, rows, cols);
        }
    }

    /// bf16 GEMV (tied LM head): y[rows] f32 = scale * (W bf16 [rows][cols] @ x).
    pub fn opGemvBf16(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 2 == 0);
        const w_db = try self.cachedWeight(w_bytes);
        const f = try self.eltFn(elt.gemv_bf16_ptx, "gemv_bf16");
        try self.rowLaunch(f, w_db, x, y, null, .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0 }, .{ scale, 0 }, rows);
    }

    /// Multi-input bf16 GEMV (speculative-decode LM head): y[i][rows] f32 =
    /// scale * (W @ x_i) for i < n (n <= 4), reading W once for all inputs.
    /// x must have 4 rows of backing store; rows beyond n may be garbage
    /// (their outputs are predicated off).
    pub fn opGemvBf16N(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize, n: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 2 == 0 and rows % 8 == 0 and n >= 1 and n <= 4);
        const w_db = try self.cachedWeight(w_bytes);
        const f = try self.eltFn(elt.gemv_bf16n_ptx, "gemv_bf16n");
        try self.rowLaunch(f, w_db, x, y, null, .{ @intCast(rows), @intCast(cols), @intCast(n), 0, 0, 0 }, .{ scale, 0 }, rows / 8);
    }

    /// Flash-decoding attention for seq_q consecutive causal queries against
    /// the KV cache (seq_q == 1 is plain decode; > 1 is the speculative
    /// verify batch): a warp per (query, head, KV chunk) in the split pass,
    /// then a merge pass over seq_q*heads rows. Query t sees kv_len0 + t
    /// keys. scratch holds seq_q*heads*nsplit*(hd+4) f32. hd is 128 (4 dims
    /// per lane) or 256 (8 dims per lane, qwen35).
    pub fn opAttnDecode(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, scratch: DeviceBuffer, kv_len0: usize, seq_q: usize, n_heads: usize, kv_heads: usize, hd: usize, nsplit: usize, scale: f32) Error!void {
        self.ptic();
        defer self.ptoc(.attn);
        std.debug.assert((hd == 128 or hd == 256) and seq_q >= 1);
        const f_split = if (hd == 128)
            try self.eltFn(elt.attn_split_ptx, "attn_split")
        else
            try self.eltFn(elt.attn_split_h256_ptx, "attn_split_h256");
        try self.eltLaunch(f_split, q, k, v, scratch, .{ @intCast(kv_len0), @intCast(n_heads), @intCast(kv_heads), @intCast(hd), @intCast(nsplit), @intCast(seq_q) }, .{ scale, 0 }, seq_q * n_heads * nsplit * 32);
        const f_merge = try self.eltFn(elt.attn_merge_ptx, "attn_merge");
        try self.eltLaunch(f_merge, scratch, out, null, null, .{ @intCast(seq_q * n_heads), @intCast(hd), @intCast(nsplit), 0, 0, 0 }, .{ 0, 0 }, seq_q * n_heads * hd);
    }

    /// Partial rotate-half RoPE: rotate the first 2*half dims of
    /// head_dim-wide heads (qwen35: 64 of 256), rows at positions pos0+row.
    pub fn opRopeHalfPart(self: *Backend, qk: DeviceBuffer, freqs: DeviceBuffer, seq: usize, n_heads: usize, half: usize, sin_off: usize, pos0: usize, head_dim: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_half_part_ptx, "rope_half_part");
        const total = seq * n_heads * half;
        try self.eltLaunch(f, qk, null, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), @intCast(pos0), @intCast(head_dim) }, .{ 0, 0 }, total);
    }

    /// Interleaved M-RoPE for one row (qwen35 decode with images): the
    /// position per pair comes from the device pos3 buffer (t, h, w) via
    /// ggml's imrope round-robin; equal positions reproduce opRopeHalfPart.
    pub fn opRopeImrope(self: *Backend, qk: DeviceBuffer, pos3: DeviceBuffer, freqs: DeviceBuffer, n_heads: usize, half: usize, sin_off: usize, sections: [3]u32, head_dim: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_imrope_ptx, "rope_imrope");
        const packed_sections = sections[0] | (sections[1] << 8) | (sections[2] << 16);
        const total = n_heads * half;
        try self.eltLaunch(f, qk, pos3, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), packed_sections, @intCast(head_dim) }, .{ 0, 0 }, total);
    }

    /// rope_imrope over a batch of rows with per-row position triples
    /// (pos3s: [rows][3] u32 on device).
    pub fn opRopeImropePos(self: *Backend, qk: DeviceBuffer, pos3s: DeviceBuffer, freqs: DeviceBuffer, rows: usize, n_heads: usize, half: usize, sin_off: usize, sections: [3]u32, head_dim: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_imrope_pos_ptx, "rope_imrope_pos");
        const packed_sections = sections[0] | (sections[1] << 8) | (sections[2] << 16);
        const total = rows * n_heads * half;
        try self.eltLaunch(f, qk, pos3s, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), packed_sections, @intCast(head_dim) }, .{ 0, 0 }, total);
    }

    /// Deinterleave the qwen35 attention q projection into query and gate
    /// ([q(hd) gate(hd)] per 2*hd-wide head slot).
    pub fn opDeinterleave2(self: *Backend, qg: DeviceBuffer, q: DeviceBuffer, gate: DeviceBuffer, total: usize, hd: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.deinterleave2_ptx, "deinterleave2");
        try self.eltLaunch(f, qg, q, gate, null, .{ @intCast(total), @intCast(hd), 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// a[i] *= sigmoid(b[i]) — the qwen35 attention output gate.
    pub fn opMulSigmoid(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.mul_sigmoid_ptx, "mul_sigmoid");
        try self.eltLaunch(f, a, b, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// Row-wise L2 normalization in place, x_row /= max(|x_row|, eps)
    /// (ggml_l2_norm; rows of dim <= 256).
    pub fn opL2NormRows(self: *Backend, x: DeviceBuffer, rows: usize, dim: usize, eps: f32) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        std.debug.assert(dim <= 256);
        const f = try self.eltFn(elt.l2norm_rows_ptx, "l2norm_rows");
        try self.rowLaunch(f, x, null, null, null, .{ @intCast(rows), @intCast(dim), 0, 0, 0, 0 }, .{ eps, 0 }, rows);
    }

    /// One qwen35 causal-conv step (kernel 4, SiLU) over all channels; the
    /// 3-column per-channel state rolls forward.
    pub fn opGdnConvStep(self: *Backend, conv_state: DeviceBuffer, x: DeviceBuffer, conv_w: DeviceBuffer, out: DeviceBuffer, channels: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.gdn_conv_step_ptx, "gdn_conv_step");
        try self.eltLaunch(f, conv_state, x, conv_w, out, .{ @intCast(channels), 0, 0, 0, 0, 0 }, .{ 0, 0 }, channels);
    }

    /// Per-head delta-net gates: decay = exp(a*softplus(alpha+dt)),
    /// beta = sigmoid(beta_in).
    pub fn opGdnGates(self: *Backend, alpha_beta: DeviceBuffer, a_dt: DeviceBuffer, out: DeviceBuffer, heads: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.gdn_gates_ptx, "gdn_gates");
        try self.eltLaunch(f, alpha_beta, a_dt, out, null, .{ @intCast(heads), 0, 0, 0, 0, 0 }, .{ 0, 0 }, heads);
    }

    /// One decode step of the gated-delta-net recurrence: one 256-thread
    /// block per v-head over its [d][d] state (d <= 128 so the staging
    /// threads fit the block).
    pub fn opGdnDeltaStep(self: *Backend, state: DeviceBuffer, conv_out: DeviceBuffer, gates: DeviceBuffer, o: DeviceBuffer, heads: usize, d: usize, k_heads: usize, scale: f32) Error!void {
        std.debug.assert(d % 4 == 0); // the state walks are x4 unrolled
        self.ptic();
        defer self.ptoc(.attn);
        std.debug.assert(d <= 128);
        const f = try self.eltFn(elt.gdn_delta_step_ptx, "gdn_delta_step");
        try self.rowLaunch(f, state, conv_out, gates, o, .{ @intCast(heads), @intCast(d), @intCast(k_heads), 0, 0, 0 }, .{ scale, 0 }, heads);
    }

    /// Tree-verify flash-decoding attention (LLM_PLAN.md M8): seq_q tree
    /// nodes, node t attending kv rows [0, prefix_len) of the linear cache
    /// plus its ancestor chain stored at rows tree_base+idx of the SAME k/v
    /// buffers. Per-query kv lengths and ancestor row lists live in a meta
    /// table at the scratch tail (see elt.attn_split_tree_ptx; the caller
    /// uploads it before the batch). Chunking matches the decode kernel at
    /// the same kv_len — merged outputs are bitwise-identical to plain
    /// decode. hd must be 128.
    pub fn opAttnDecodeTree(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, scratch: DeviceBuffer, prefix_len: usize, tree_base: usize, seq_q: usize, n_heads: usize, kv_heads: usize, hd: usize, nsplit: usize, scale: f32) Error!void {
        self.ptic();
        defer self.ptoc(.attn);
        std.debug.assert(hd == 128 and seq_q >= 1);
        const f_split = try self.eltFn(elt.attn_split_tree_ptx, "attn_split_tree");
        try self.eltLaunch(f_split, q, k, v, scratch, .{ @intCast(prefix_len), @intCast(n_heads), @intCast(kv_heads), @intCast(tree_base), @intCast(nsplit), @intCast(seq_q) }, .{ scale, 0 }, seq_q * n_heads * nsplit * 32);
        const f_merge = try self.eltFn(elt.attn_merge_ptx, "attn_merge");
        try self.eltLaunch(f_merge, scratch, out, null, null, .{ @intCast(seq_q * n_heads), @intCast(hd), @intCast(nsplit), 0, 0, 0 }, .{ 0, 0 }, seq_q * n_heads * hd);
    }

    /// rotate-half RoPE with an explicit absolute u32 position per row
    /// (tree-verify batches: node positions are depth-based).
    pub fn opRopeHalfPos(self: *Backend, qk: DeviceBuffer, positions: DeviceBuffer, freqs: DeviceBuffer, rows: usize, n_heads: usize, half: usize, sin_off: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_half_pos_ptx, "rope_half_pos");
        const total = rows * n_heads * half;
        try self.eltLaunch(f, qk, positions, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), 0, 0 }, .{ 0, 0 }, total);
    }

    /// eltLaunch variant with one 256-thread block per row (`grid_rows` blocks).
    fn rowLaunch(self: *Backend, f: cu.CUfunction, b0: ?DeviceBuffer, b1: ?DeviceBuffer, b2: ?DeviceBuffer, b3: ?DeviceBuffer, u: [6]u32, fp: [2]f32, grid_rows: usize) Error!void {
        var p0: cu.CUdeviceptr = if (b0) |b| b.ptr() else 0;
        var p1: cu.CUdeviceptr = if (b1) |b| b.ptr() else 0;
        var p2: cu.CUdeviceptr = if (b2) |b| b.ptr() else 0;
        var p3: cu.CUdeviceptr = if (b3) |b| b.ptr() else 0;
        var uu = u;
        var ff = fp;
        var params = [_]?*anyopaque{
            @ptrCast(&p0),    @ptrCast(&p1),    @ptrCast(&p2),    @ptrCast(&p3),
            @ptrCast(&uu[0]), @ptrCast(&uu[1]), @ptrCast(&uu[2]), @ptrCast(&uu[3]),
            @ptrCast(&uu[4]), @ptrCast(&uu[5]), @ptrCast(&ff[0]), @ptrCast(&ff[1]),
        };
        self.ctx.launch(f, .{ @intCast(grid_rows), 1, 1 }, .{ 256, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// f16 tensor-core conv/GEMM for the VAE: y[m][co] f32 = x[m][k] f32 @ Wᵀ + bias,
    /// W f32 [co][k]. Weight and activation convert to f16 zero-padded to the coop
    /// tile (co→128-mult, k→32-mult, m→128), the validated buildHgemm produces a
    /// padded f32 tile, and bias_compact strips the pad + adds bias into `dst` at
    /// `dst_off_elems`. Much faster than the f32 GEMM for the large (co≥96) convs.
    pub fn opConvF16(self: *Backend, dst: DeviceBuffer, dst_off_elems: usize, src: DeviceBuffer, m: usize, w_bytes: []const u8, co: usize, k: usize, bias: []const f32) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        const co_pad = std.mem.alignForward(usize, co, 128);
        const k_pad = std.mem.alignForward(usize, k, 32);
        const m_pad = std.mem.alignForward(usize, m, 128);
        const w_db = try self.cachedWeight(w_bytes);
        const b_db = try self.cachedWeight(std.mem.sliceAsBytes(bias));
        try self.ensureDeviceBuffer(&self.conv_w16, co_pad * k_pad * 2);
        try self.ensureDeviceBuffer(&self.conv_a16, m_pad * k_pad * 2);
        try self.ensureDeviceBuffer(&self.conv_c, m_pad * co_pad * 4);
        const f_pad = try self.eltFn(elt.f32_to_f16_pad2d_ptx, "f32_to_f16_pad2d");
        try self.eltLaunch(f_pad, w_db, self.conv_w16, null, null, .{ @intCast(co_pad * k_pad), @intCast(k_pad), @intCast(co), @intCast(k), 0, 0 }, .{ 0, 0 }, co_pad * k_pad);
        try self.eltLaunch(f_pad, src, self.conv_a16, null, null, .{ @intCast(m_pad * k_pad), @intCast(k_pad), @intCast(m), @intCast(k), 0, 0 }, .{ 0, 0 }, m_pad * k_pad);
        if (self.kernels == .libs) {
            try self.ltMatmulF16(self.conv_c, self.conv_w16, self.conv_a16, co_pad, m_pad, k_pad);
        } else {
            const f_hg = try self.hgemmFn();
            try self.launchHgemm(f_hg, self.conv_a16, self.conv_w16, self.conv_c, m_pad, co_pad, k_pad);
        }
        const f_bc = try self.eltFn(elt.bias_compact_ptx, "bias_compact");
        try self.eltLaunch(f_bc, self.conv_c, b_db, dst, null, .{ @intCast(m * co), @intCast(co), @intCast(co_pad), @intCast(dst_off_elems), 0, 0 }, .{ 0, 0 }, m * co);
    }

    /// cuDNN fused 3×3 NHWC convolution (.libs mode) — the VAE's big convs.
    /// dst[dst_off_elems..][n][co] f32 = conv3x3(src[n][ci] f32, W[co][3][3][ci]
    /// f32) + bias. src/weight convert to f16 (tensor cores, f32 accumulate),
    /// cuDNN writes f16, then `bias_add_f16` adds the per-channel bias into the
    /// f32 dst. No im2col materialization (IMPLICIT_PRECOMP_GEMM). Same numeric
    /// regime as `opConvF16` (f16 conv), validated vs the CPU VAE.
    pub fn opConvCudnn(self: *Backend, dst: DeviceBuffer, dst_off_elems: usize, src: DeviceBuffer, h: usize, w: usize, w_bytes: []const u8, co: usize, ci: usize, bias: []const f32) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        const n = h * w;
        const w_db = try self.cachedWeight(w_bytes);
        const b_db = try self.cachedWeight(std.mem.sliceAsBytes(bias));
        try self.ensureDeviceBuffer(&self.conv_a16, n * ci * 2);
        try self.ensureDeviceBuffer(&self.conv_w16, co * 9 * ci * 2);
        try self.ensureDeviceBuffer(&self.conv_c, n * co * 2);
        const f_cvt = try self.eltFn(elt.f32_to_f16_ptx, "f32_to_f16");
        try self.eltLaunch(f_cvt, src, self.conv_a16, null, null, .{ @intCast(n * ci), @intCast(n * ci), 0, 0, 0, 0 }, .{ 0, 0 }, n * ci);
        try self.eltLaunch(f_cvt, w_db, self.conv_w16, null, null, .{ @intCast(co * 9 * ci), @intCast(co * 9 * ci), 0, 0, 0, 0 }, .{ 0, 0 }, co * 9 * ci);
        const L = &self.libs.?;
        var plan = cudnn.ConvPlan.build(&L.dnn, L.dnn_handle, h, w, ci, co) catch return error.CudaError;
        defer plan.deinit(&L.dnn);
        if (plan.workspace_bytes > 0) try self.ensureDeviceBuffer(&self.cudnn_ws, plan.workspace_bytes);
        plan.execute(&L.dnn, L.dnn_handle, self.conv_a16.ptr(), self.conv_w16.ptr(), self.conv_c.ptr(), self.cudnn_ws.ptr()) catch return error.CudaError;
        const f_bias = try self.eltFn(elt.bias_add_f16_ptx, "bias_add_f16");
        try self.eltLaunch(f_bias, self.conv_c, b_db, dst, null, .{ @intCast(n * co), @intCast(co), @intCast(dst_off_elems), 0, 0, 0 }, .{ 0, 0 }, n * co);
    }

    // ---- cuBLASLt int8 GEMM (.libs mode) ------------------------------------

    fn ltCheck(self: *Backend, s: cublaslt.Status, comptime what: []const u8) Error!void {
        _ = self;
        if (s == cublaslt.SUCCESS) return;
        std.debug.print("cuBLASLt {s} failed: {s} ({d})\n", .{ what, cublaslt.statusName(s), s });
        return error.CudaError;
    }

    fn irescaleFn(self: *Backend, c_h16: bool) Error!cu.CUfunction {
        if (c_h16) {
            if (self.irescale_h16_mod != null) return self.irescale_h16_fn;
            var mod = self.ctx.loadModule(kernels.irescale_h16_ptx) catch return error.CudaError;
            self.irescale_h16_fn = mod.getFunction(self.ctx, "irescale_h16") catch return error.CudaError;
            self.irescale_h16_mod = mod;
            return self.irescale_h16_fn;
        }
        if (self.irescale_mod != null) return self.irescale_fn;
        var mod = self.ctx.loadModule(kernels.irescale_ptx) catch return error.CudaError;
        self.irescale_fn = mod.getFunction(self.ctx, "irescale") catch return error.CudaError;
        self.irescale_mod = mod;
        return self.irescale_fn;
    }

    /// Build (or fetch cached) the cuBLASLt plan for a TN GEMM of shape
    /// D[n,m] = op(W)ᵀ @ op(A) at these (kind,n,m,k). Layouts describe the STORED
    /// (pre-op) col-major matrices: W is row-major [n][k] == col-major [k][n],
    /// A is row-major [m][k] == col-major [k][m], D is col-major [n][m] ==
    /// row-major [m][n] (what `irescale` / the f16 consumers expect). The
    /// heuristic runs once per shape; the returned plan carries no data pointer.
    fn ltPlan(self: *Backend, kind: LtKind, n: usize, m: usize, k: usize) Error!LtPlan {
        const key: u64 = (@as(u64, @intFromEnum(kind)) << 62) | (@as(u64, n) << 42) | (@as(u64, m) << 21) | @as(u64, k);
        if (self.lt_plans.get(key)) |p| return p;

        const ab_type: c_int = switch (kind) {
            .i8 => cublaslt.R_8I,
            .f16 => cublaslt.R_16F,
        };
        const d_type: c_int = switch (kind) {
            .i8 => cublaslt.R_32I,
            .f16 => cublaslt.R_32F,
        };
        const compute: c_int = switch (kind) {
            .i8 => cublaslt.COMPUTE_32I,
            .f16 => cublaslt.COMPUTE_32F,
        };
        const scale: c_int = switch (kind) {
            .i8 => cublaslt.R_32I,
            .f16 => cublaslt.R_32F,
        };

        const L = &self.libs.?;
        const lt = &L.lt;
        var desc: cublaslt.MatmulDesc = null;
        try self.ltCheck(lt.cublasLtMatmulDescCreate(&desc, compute, scale), "MatmulDescCreate");
        errdefer _ = lt.cublasLtMatmulDescDestroy(desc);
        var op_t: c_int = cublaslt.OP_T;
        var op_n: c_int = cublaslt.OP_N;
        try self.ltCheck(lt.cublasLtMatmulDescSetAttribute(desc, cublaslt.DESC_TRANSA, @ptrCast(&op_t), @sizeOf(c_int)), "SetTransA");
        try self.ltCheck(lt.cublasLtMatmulDescSetAttribute(desc, cublaslt.DESC_TRANSB, @ptrCast(&op_n), @sizeOf(c_int)), "SetTransB");

        var adesc: cublaslt.MatrixLayout = null; // W: [k,n] col-major, ld=k
        try self.ltCheck(lt.cublasLtMatrixLayoutCreate(&adesc, ab_type, @intCast(k), @intCast(n), @intCast(k)), "A layout");
        errdefer _ = lt.cublasLtMatrixLayoutDestroy(adesc);
        var bdesc: cublaslt.MatrixLayout = null; // A: [k,m] col-major, ld=k
        try self.ltCheck(lt.cublasLtMatrixLayoutCreate(&bdesc, ab_type, @intCast(k), @intCast(m), @intCast(k)), "B layout");
        errdefer _ = lt.cublasLtMatrixLayoutDestroy(bdesc);
        var ddesc: cublaslt.MatrixLayout = null; // D: [n,m] col-major, ld=n
        try self.ltCheck(lt.cublasLtMatrixLayoutCreate(&ddesc, d_type, @intCast(n), @intCast(m), @intCast(n)), "D layout");
        errdefer _ = lt.cublasLtMatrixLayoutDestroy(ddesc);

        var pref: cublaslt.MatmulPreference = null;
        try self.ltCheck(lt.cublasLtMatmulPreferenceCreate(&pref), "PreferenceCreate");
        defer _ = lt.cublasLtMatmulPreferenceDestroy(pref);
        var ws_bytes: usize = L.workspace.bytes;
        try self.ltCheck(lt.cublasLtMatmulPreferenceSetAttribute(pref, cublaslt.PREF_MAX_WORKSPACE_BYTES, @ptrCast(&ws_bytes), @sizeOf(usize)), "PrefWorkspace");

        var results: [1]cublaslt.HeuristicResult = .{.{}};
        var count: c_int = 0;
        try self.ltCheck(lt.cublasLtMatmulAlgoGetHeuristic(L.lt_handle, desc, adesc, bdesc, ddesc, ddesc, pref, 1, &results, &count), "AlgoGetHeuristic");
        if (count < 1) {
            std.debug.print("cuBLASLt: no {s} algo for n={d} m={d} k={d}\n", .{ @tagName(kind), n, m, k });
            return error.CudaError;
        }

        const plan = LtPlan{ .desc = desc, .adesc = adesc, .bdesc = bdesc, .ddesc = ddesc, .algo = results[0].algo };
        self.lt_plans.put(self.gpa, key, plan) catch return error.OutOfMemory;
        return plan;
    }

    /// Issue a cuBLASLt matmul for a prepared plan against device pointers.
    /// `alpha`/`beta` point to the scale values in the plan's scale type (i32 for
    /// .i8, f32 for .f16). D and C are the same buffer (beta=0).
    fn ltRun(self: *Backend, plan: LtPlan, d: DeviceBuffer, w: DeviceBuffer, a: DeviceBuffer, alpha: *const anyopaque, beta: *const anyopaque) Error!void {
        const L = &self.libs.?;
        const dp: *anyopaque = @ptrFromInt(d.ptr());
        try self.ltCheck(L.lt.cublasLtMatmul(
            L.lt_handle,
            plan.desc,
            alpha,
            @ptrFromInt(w.ptr()),
            plan.adesc,
            @ptrFromInt(a.ptr()),
            plan.bdesc,
            beta,
            dp,
            plan.ddesc,
            dp,
            plan.ddesc,
            &plan.algo,
            @ptrFromInt(L.workspace.ptr),
            L.workspace.bytes,
            self.ctx.stream,
        ), "Matmul");
    }

    /// s32 = Wᵀ @ A on int8 tensor cores via cuBLASLt. `w` is the weight W[n][k]
    /// row-major; `a` the prepped activation A[m][k] row-major; `d` the s32
    /// accumulator (row-major [m][n]). cuBLASLt does ONLY the GEMM — the
    /// per-row×per-col rescale is a separate `irescale` pass.
    pub fn ltMatmulI8(self: *Backend, d: DeviceBuffer, w: DeviceBuffer, a: DeviceBuffer, n: usize, m: usize, k: usize) Error!void {
        const plan = try self.ltPlan(.i8, n, m, k);
        var alpha: i32 = 1;
        var beta: i32 = 0;
        try self.ltRun(plan, d, w, a, @ptrCast(&alpha), @ptrCast(&beta));
    }

    /// f32 D[m][n] = A[m][k](f16) @ W[n][k](f16)ᵀ on the f16 tensor cores (HMMA,
    /// f32 accumulate) via cuBLASLt — the drop-in for the hand-PTX `buildHgemm`
    /// used by the fp8 encoder GEMMs and the VAE convs.
    pub fn ltMatmulF16(self: *Backend, d: DeviceBuffer, w: DeviceBuffer, a: DeviceBuffer, n: usize, m: usize, k: usize) Error!void {
        const plan = try self.ltPlan(.f16, n, m, k);
        var alpha: f32 = 1;
        var beta: f32 = 0;
        try self.ltRun(plan, d, w, a, @ptrCast(&alpha), @ptrCast(&beta));
    }

    /// Fetch/build the cached cuDNN fused int8-GEMM+dequant plan for this shape.
    fn mdqPlan(self: *Backend, m: usize, n: usize, k: usize, d_f16: bool) Error!cudnn.MatmulDequantPlan {
        const key: u64 = (@as(u64, m) << 43) | (@as(u64, n) << 22) | (@as(u64, k) << 1) | @intFromBool(d_f16);
        if (self.mdq_plans.get(key)) |p| return p;
        const L = &self.libs.?;
        const p = cudnn.MatmulDequantPlan.build(&L.dnn, L.dnn_handle, m, n, k, d_f16) catch return error.CudaError;
        self.mdq_plans.put(self.gpa, key, p) catch return error.OutOfMemory;
        return p;
    }

    /// cuBLASLt int8 linear: GEMM into the s32 scratch, then fused rescale to y.
    /// With `use_fused_i8`, replaced by a single cuDNN op graph (GEMM+dequant
    /// fused; no s32 round-trip, no separate irescale).
    fn opI8GemmLibs(self: *Backend, y: DeviceBuffer, w_db: DeviceBuffer, ws_db: DeviceBuffer, rows: usize, c_h16: bool) Error!void {
        const m = self.i8_mpad;
        const k = self.i8_cols;
        if (self.use_fused_i8) {
            const plan = try self.mdqPlan(m, rows, k, c_h16);
            if (plan.workspace_bytes > 0) try self.ensureDeviceBuffer(&self.cudnn_ws, plan.workspace_bytes);
            const L = &self.libs.?;
            plan.execute(&L.dnn, L.dnn_handle, self.i8_x.ptr(), w_db.ptr(), self.i8_scale.ptr(), ws_db.ptr(), y.ptr(), self.cudnn_ws.ptr()) catch return error.CudaError;
            return;
        }
        try self.ensureDeviceBuffer(&self.i8_acc, m * rows * 4);
        try self.ltMatmulI8(self.i8_acc, w_db, self.i8_x, rows, m, k);
        if (bench_skip_rescale) return; // DIAGNOSTIC: measure irescale's batched cost
        const f = try self.irescaleFn(c_h16);
        const total = m * rows;
        var pacc = self.i8_acc.ptr();
        var py = y.ptr();
        var pas = self.i8_scale.ptr();
        var pws = ws_db.ptr();
        var prows: u32 = @intCast(rows);
        var ptot: u32 = @intCast(total);
        var pr = [_]?*anyopaque{ @ptrCast(&pacc), @ptrCast(&py), @ptrCast(&pas), @ptrCast(&pws), @ptrCast(&prows), @ptrCast(&ptot) };
        self.ctx.launch(f, .{ @intCast((total + 255) / 256), 1, 1 }, .{ 256, 1, 1 }, 0, &pr) catch return error.CudaError;
    }

    /// Rotate + per-row dynamic quantize x[m][cols] -> internal i8_x (s8) +
    /// i8_scale (per-row f32), padding m up to 128. Consumed by opI8Gemm.
    pub fn opI8Prep(self: *Backend, x: DeviceBuffer, m: usize, cols: usize, in_f16: bool) Error!void {
        self.ptic();
        defer self.ptoc(.prep);
        const mpad = std.mem.alignForward(usize, m, 128);
        try self.ensureDeviceBuffer(&self.i8_x, mpad * cols);
        try self.ensureDeviceBuffer(&self.i8_scale, mpad * 4);
        // The prep kernel (grid = {m,1,1}) fully overwrites rows 0..m-1 — every
        // column (cols % 1024 == 0, see buildPrep) plus i8_scale[row] — so only
        // the pad rows [m..mpad) need zeroing (GEMM pad rows -> 0 acc, scale 0).
        // When m is already 128-aligned there is no pad, so skip the memset (and
        // its NULL-stream sync bubble) entirely.
        if (mpad > m) {
            const pad = mpad - m;
            self.ctx.memsetD8(.{ .ptr = self.i8_x.ptr() + @as(u64, @intCast(m * cols)), .bytes = @intCast(pad * cols) }, 0, pad * cols) catch return error.CudaError;
            self.ctx.memsetD32(.{ .ptr = self.i8_scale.ptr() + @as(u64, @intCast(m * 4)), .bytes = @intCast(pad * 4) }, 0, pad) catch return error.CudaError;
        }
        const f = try self.prepFn(cols, in_f16);
        var px = x.ptr();
        var pq = self.i8_x.ptr();
        var pas = self.i8_scale.ptr();
        var pp = [_]?*anyopaque{ @ptrCast(&px), @ptrCast(&pq), @ptrCast(&pas) };
        self.ctx.launch(f, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(kernels.prepSharedBytes(cols)), &pp) catch return error.CudaError;
        self.i8_m = m;
        self.i8_mpad = mpad;
        self.i8_cols = cols;
    }

    /// int8 GEMM + fused rescale against the last opI8Prep:
    /// y[m][rows] = prepped @ Wᵀ * act_scale[row] * weight_scale[col] (f32).
    pub fn opI8Gemm(self: *Backend, y: DeviceBuffer, w_bytes: []const u8, weight_scale: []const f32, rows: usize, c_h16: bool) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(!c_h16 or self.kernels == .libs); // f16 output only on the cuBLASLt/irescale path
        const w_db = try self.cachedWeight(w_bytes);
        const ws_db = try self.cachedWeight(std.mem.sliceAsBytes(weight_scale));
        if (self.kernels == .libs) return self.opI8GemmLibs(y, w_db, ws_db, rows, c_h16);
        const f = try self.fusedFn();
        var pa = self.i8_x.ptr();
        var pb = w_db.ptr();
        var pc = y.ptr();
        var pn: u32 = @intCast(rows);
        var pk: u32 = @intCast(self.i8_cols);
        var pas = self.i8_scale.ptr();
        var pws = ws_db.ptr();
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk), @ptrCast(&pas), @ptrCast(&pws) };
        self.ctx.launch(f, .{ @intCast(rows / 128), @intCast(self.i8_mpad / 128), 1 }, .{ 128, 1, 1 }, 0, &pg) catch return error.CudaError;
    }

    // ---- int4 (W4A4) GEMM pair ----------------------------------------------

    fn i4prepFn(self: *Backend, cols: usize) Error!cu.CUfunction {
        if (self.i4_prep_mods.get(cols)) |f| return f;
        const ptx = kernels.buildPrep(self.gpa, cols, 4, false) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        const f = mod.getFunction(self.ctx, "i4prep") catch return error.CudaError;
        self.ctx.setMaxDynamicShared(f, kernels.prepSharedBytes(cols)) catch return error.CudaError;
        self.i4_prep_owned.append(self.gpa, mod) catch return error.OutOfMemory;
        self.i4_prep_mods.put(self.gpa, cols, f) catch return error.OutOfMemory;
        return f;
    }

    fn i4fusedFn(self: *Backend) Error!cu.CUfunction {
        if (self.i4_fused_mod != null) return self.i4_fused_fn;
        const ptx = kernels.buildIgemmPipe(self.gpa, 64, true, 4) catch return error.OutOfMemory;
        defer self.gpa.free(ptx);
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        self.i4_fused_fn = mod.getFunction(self.ctx, "i4gemm_pipe_fused") catch return error.CudaError;
        self.i4_fused_mod = mod;
        return self.i4_fused_fn;
    }

    /// int4 (W4A4) analogue of opI8Prep: rotate + per-row dynamic quantize
    /// x[m][cols] to s4 [-8,7], nibble-packed 2/byte -> i8_x (reused as the
    /// packed-s4 activation scratch) + i8_scale (per-row f32), padding m up to
    /// 128. Consumed by opI4Gemm. (The i8_* state is a homogeneous checkpoint's
    /// single "prepped activation" slot — a run is all-i8 or all-i4.)
    pub fn opI4Prep(self: *Backend, x: DeviceBuffer, m: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.prep);
        const mpad = std.mem.alignForward(usize, m, 128);
        const qbytes = cols / 2; // packed s4 bytes per row
        try self.ensureDeviceBuffer(&self.i8_x, mpad * qbytes);
        try self.ensureDeviceBuffer(&self.i8_scale, mpad * 4);
        if (mpad > m) {
            const pad = mpad - m;
            self.ctx.memsetD8(.{ .ptr = self.i8_x.ptr() + @as(u64, @intCast(m * qbytes)), .bytes = @intCast(pad * qbytes) }, 0, pad * qbytes) catch return error.CudaError;
            self.ctx.memsetD32(.{ .ptr = self.i8_scale.ptr() + @as(u64, @intCast(m * 4)), .bytes = @intCast(pad * 4) }, 0, pad) catch return error.CudaError;
        }
        const f = try self.i4prepFn(cols);
        var px = x.ptr();
        var pq = self.i8_x.ptr();
        var pas = self.i8_scale.ptr();
        var pp = [_]?*anyopaque{ @ptrCast(&px), @ptrCast(&pq), @ptrCast(&pas) };
        self.ctx.launch(f, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, @intCast(kernels.prepSharedBytes(cols)), &pp) catch return error.CudaError;
        self.i8_m = m;
        self.i8_mpad = mpad;
        self.i8_cols = cols;
    }

    /// int4 GEMM + fused rescale against the last opI4Prep. `w_bytes` are the
    /// packed-s4 weight [rows][cols/2]; layout/params match opI8Gemm exactly
    /// (the m16n8k64.s4 kernel derives the k/2 byte stride from the element k).
    pub fn opI4Gemm(self: *Backend, y: DeviceBuffer, w_bytes: []const u8, weight_scale: []const f32, rows: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        const w_db = try self.cachedWeight(w_bytes);
        const ws_db = try self.cachedWeight(std.mem.sliceAsBytes(weight_scale));
        const f = try self.i4fusedFn();
        var pa = self.i8_x.ptr();
        var pb = w_db.ptr();
        var pc = y.ptr();
        var pn: u32 = @intCast(rows);
        var pk: u32 = @intCast(self.i8_cols);
        var pas = self.i8_scale.ptr();
        var pws = ws_db.ptr();
        var pg = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk), @ptrCast(&pas), @ptrCast(&pws) };
        self.ctx.launch(f, .{ @intCast(rows / 128), @intCast(self.i8_mpad / 128), 1 }, .{ 128, 1, 1 }, 0, &pg) catch return error.CudaError;
    }

    // ---- eltwise / attention (correctness-first f32 kernels) ----------------

    // --- decode-graph support (CUDA graphs, LLM_PLAN.md M6) ----------------

    /// Lazy-load the decode-state module (one g_state global shared by the
    /// graph-mode kernel entries) and resolve its pieces.
    fn stateSetup(self: *Backend) Error!void {
        if (self.state_ptr != 0) return;
        var mod = self.ctx.loadModule(elt.decode_state_ptx) catch return error.CudaError;
        self.elt_mods.append(self.gpa, mod) catch return error.OutOfMemory;
        var sz: usize = 0;
        self.ctx.check(self.ctx.api.cuModuleGetGlobal(&self.state_ptr, &sz, mod.mod, "g_state"), "cuModuleGetGlobal") catch return error.CudaError;
        std.debug.assert(sz == 8);
        self.f_embed_gather_s = mod.getFunction(self.ctx, "embed_gather_s") catch return error.CudaError;
        self.f_kv_append_s = mod.getFunction(self.ctx, "kv_append_s") catch return error.CudaError;
        self.f_rope_half_s = mod.getFunction(self.ctx, "rope_half_s") catch return error.CudaError;
        self.f_attn_split_s = mod.getFunction(self.ctx, "attn_split_s") catch return error.CudaError;
        self.f_attn_split_h256_s = mod.getFunction(self.ctx, "attn_split_h256_s") catch return error.CudaError;
        self.f_embed_gather_q8_0 = mod.getFunction(self.ctx, "embed_gather_q8_0") catch return error.CudaError;
        self.f_embed_gather_q4_k = mod.getFunction(self.ctx, "embed_gather_q4_k") catch return error.CudaError;
        self.f_embed_gather_q5_k = mod.getFunction(self.ctx, "embed_gather_q5_k") catch return error.CudaError;
        self.f_embed_gather_q6_k = mod.getFunction(self.ctx, "embed_gather_q6_k") catch return error.CudaError;
    }

    /// Write the per-token dynamic state ({token id, cache position}) the
    /// graph-mode kernels read. Synchronous 8-byte upload; the legacy null
    /// stream orders it before the following graph launch.
    pub fn setDecodeState(self: *Backend, token: u32, pos0: u32) Error!void {
        try self.stateSetup();
        const state = [2]u32{ token, pos0 };
        self.ctx.check(self.ctx.api.cuMemcpyHtoD(self.state_ptr, &state, 8), "cuMemcpyHtoD state") catch return error.CudaError;
    }

    /// Capture everything launched between begin and end on the compute
    /// stream into an executable graph.
    pub fn graphCaptureBegin(self: *Backend) Error!void {
        self.ctx.check(self.ctx.api.cuStreamBeginCapture(self.ctx.stream, 0), "cuStreamBeginCapture") catch return error.CudaError;
    }

    pub fn graphCaptureEnd(self: *Backend) Error!cu.CUgraphExec {
        var graph: cu.CUgraph = null;
        self.ctx.check(self.ctx.api.cuStreamEndCapture(self.ctx.stream, &graph), "cuStreamEndCapture") catch return error.CudaError;
        defer _ = self.ctx.api.cuGraphDestroy(graph);
        var exec: cu.CUgraphExec = null;
        self.ctx.check(self.ctx.api.cuGraphInstantiateWithFlags(&exec, graph, 0), "cuGraphInstantiate") catch return error.CudaError;
        return exec;
    }

    pub fn graphLaunch(self: *Backend, exec: cu.CUgraphExec) Error!void {
        self.ctx.check(self.ctx.api.cuGraphLaunch(exec, self.ctx.stream), "cuGraphLaunch") catch return error.CudaError;
    }

    pub fn graphDestroy(self: *Backend, exec: cu.CUgraphExec) void {
        _ = self.ctx.api.cuGraphExecDestroy(exec);
    }

    /// Graph-mode ops: parameter-identical to their twins except the dynamic
    /// value comes from g_state (see elt.decode_state_ptx).
    pub fn opEmbedGatherS(self: *Backend, x: DeviceBuffer, embed_bytes: []const u8, h: usize) Error!void {
        try self.stateSetup();
        const w_db = try self.cachedWeight(embed_bytes);
        try self.eltLaunch(self.f_embed_gather_s, w_db, x, null, null, .{ @intCast(h), 0, 0, 0, 0, 0 }, .{ 0, 0 }, h);
    }

    /// opEmbedGatherS for a ggml block-quant embedding table: x[i] =
    /// dequant(embed[g_state[0]], i).
    pub fn opEmbedGatherQuant(self: *Backend, dt: dtypes.DType, x: DeviceBuffer, embed_bytes: []const u8, h: usize) Error!void {
        try self.stateSetup();
        const w_db = try self.cachedWeight(embed_bytes);
        const f = switch (dt) {
            .q8_0 => self.f_embed_gather_q8_0,
            .q4_k => self.f_embed_gather_q4_k,
            .q5_k => self.f_embed_gather_q5_k,
            .q6_k => self.f_embed_gather_q6_k,
            else => unreachable,
        };
        try self.eltLaunch(f, w_db, x, null, null, .{ @intCast(h), 0, 0, 0, 0, 0 }, .{ 0, 0 }, h);
    }

    /// dst[base + pos0*stride + i] = src[i], pos0 from g_state (graph-safe
    /// KV appends and tap-row snapshots).
    pub fn opKvAppendS(self: *Backend, dst: DeviceBuffer, src: DeviceBuffer, count: usize, stride: usize, base: usize) Error!void {
        try self.eltLaunch(self.f_kv_append_s, src, dst, null, null, .{ @intCast(count), @intCast(stride), @intCast(base), 0, 0, 0 }, .{ 0, 0 }, count);
    }

    /// dst[dst_off + i] = src[src_off + i] as a kernel — usable inside
    /// recorded batches and graph captures (unlike the null-stream memcpy).
    pub fn opCopyOff(self: *Backend, dst: DeviceBuffer, dst_off_elems: usize, src: DeviceBuffer, src_off_elems: usize, count: usize) Error!void {
        const f = try self.eltFn(elt.copy_off_ptx, "copy_off");
        try self.eltLaunch(f, src, dst, null, null, .{ @intCast(count), @intCast(dst_off_elems), @intCast(src_off_elems), 0, 0, 0 }, .{ 0, 0 }, count);
    }

    pub fn opRopeHalfS(self: *Backend, qk: DeviceBuffer, freqs: DeviceBuffer, n_heads: usize, half: usize, sin_off: usize) Error!void {
        const total = n_heads * half;
        try self.eltLaunch(self.f_rope_half_s, qk, null, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), 0, 0 }, .{ 0, 0 }, total);
    }

    pub fn opAttnDecodeSGraph(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, scratch: DeviceBuffer, n_heads: usize, kv_heads: usize, hd: usize, nsplit: usize, scale: f32) Error!void {
        std.debug.assert(hd == 128 or hd == 256);
        const f_split = if (hd == 128) self.f_attn_split_s else self.f_attn_split_h256_s;
        try self.eltLaunch(f_split, q, k, v, scratch, .{ 0, @intCast(n_heads), @intCast(kv_heads), @intCast(hd), @intCast(nsplit), 1 }, .{ scale, 0 }, n_heads * nsplit * 32);
        const f_merge = try self.eltFn(elt.attn_merge_ptx, "attn_merge");
        try self.eltLaunch(f_merge, scratch, out, null, null, .{ @intCast(n_heads), @intCast(hd), @intCast(nsplit), 0, 0, 0 }, .{ 0, 0 }, n_heads * hd);
    }

    fn eltFn(self: *Backend, ptx: [:0]const u8, entry: [:0]const u8) Error!cu.CUfunction {
        const key = @intFromPtr(ptx.ptr);
        if (self.elt_fns.get(key)) |f| return f;
        var mod = self.ctx.loadModule(ptx) catch return error.CudaError;
        const f = mod.getFunction(self.ctx, entry) catch return error.CudaError;
        self.elt_mods.append(self.gpa, mod) catch return error.OutOfMemory;
        self.elt_fns.put(self.gpa, key, f) catch return error.OutOfMemory;
        return f;
    }

    fn eltLaunch(self: *Backend, f: cu.CUfunction, b0: ?DeviceBuffer, b1: ?DeviceBuffer, b2: ?DeviceBuffer, b3: ?DeviceBuffer, u: [6]u32, fp: [2]f32, total: usize) Error!void {
        var p0: cu.CUdeviceptr = if (b0) |b| b.ptr() else 0;
        var p1: cu.CUdeviceptr = if (b1) |b| b.ptr() else 0;
        var p2: cu.CUdeviceptr = if (b2) |b| b.ptr() else 0;
        var p3: cu.CUdeviceptr = if (b3) |b| b.ptr() else 0;
        var uu = u;
        var ff = fp;
        var params = [_]?*anyopaque{
            @ptrCast(&p0),    @ptrCast(&p1),    @ptrCast(&p2),    @ptrCast(&p3),
            @ptrCast(&uu[0]), @ptrCast(&uu[1]), @ptrCast(&uu[2]), @ptrCast(&uu[3]),
            @ptrCast(&uu[4]), @ptrCast(&uu[5]), @ptrCast(&ff[0]), @ptrCast(&ff[1]),
        };
        const grid: u32 = @intCast((total + 255) / 256);
        self.ctx.launch(f, .{ grid, 1, 1 }, .{ 256, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// out = x*inv*mod[premul+c] + mod[shift+c], inv=1/sqrt(mean(x^2)+eps). Fused
    /// rmsnorm + AdaLN modulation, one thread per row.
    pub fn rmsMod(self: *Backend, x: DeviceBuffer, out: DeviceBuffer, mod: DeviceBuffer, rows: usize, dim: usize, premul_off: usize, shift_off: usize, eps: f32) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        // one block (256 threads) per row, parallel shared reduction.
        const f = try self.eltFn(elt.rms_mod_par_ptx, "rms_mod_par");
        var p0 = x.ptr();
        var p1 = out.ptr();
        var p2 = mod.ptr();
        var p3: cu.CUdeviceptr = 0;
        var uu = [_]u32{ @intCast(rows), @intCast(dim), @intCast(premul_off), @intCast(shift_off), 0, 0 };
        var ff = [_]f32{ eps, 0 };
        var params = [_]?*anyopaque{
            @ptrCast(&p0),    @ptrCast(&p1),    @ptrCast(&p2),    @ptrCast(&p3),
            @ptrCast(&uu[0]), @ptrCast(&uu[1]), @ptrCast(&uu[2]), @ptrCast(&uu[3]),
            @ptrCast(&uu[4]), @ptrCast(&uu[5]), @ptrCast(&ff[0]), @ptrCast(&ff[1]),
        };
        self.ctx.launch(f, .{ @intCast(rows), 1, 1 }, .{ 256, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// per-head RMS norm * weight, one thread per row (rows = seq*n_heads).
    pub fn qkNorm(self: *Backend, x: DeviceBuffer, out: DeviceBuffer, weight: DeviceBuffer, rows: usize, hd: usize, eps: f32) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        // Few wide rows (LLM decode: 1 x 2560) serialize badly at one thread
        // per row; hand them a block per row instead.
        if (rows < 512) {
            const f = try self.eltFn(elt.qk_rmsnorm_par_ptx, "qk_rmsnorm_par");
            try self.rowLaunch(f, x, out, weight, null, .{ @intCast(rows), @intCast(hd), 0, 0, 0, 0 }, .{ eps, 0 }, rows);
            return;
        }
        const f = try self.eltFn(elt.qk_rmsnorm_ptx, "qk_rmsnorm");
        try self.eltLaunch(f, x, out, weight, null, .{ @intCast(rows), @intCast(hd), 0, 0, 0, 0 }, .{ eps, 0 }, rows);
    }

    /// interleaved RoPE in place. total = rows*n_heads*half.
    pub fn rope(self: *Backend, qk: DeviceBuffer, freqs: DeviceBuffer, rows: usize, n_heads: usize, half: usize, sin_off: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_ptx, "rope");
        const total = rows * n_heads * half;
        try self.eltLaunch(f, qk, null, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), 0, 0 }, .{ 0, 0 }, total);
    }

    /// naive GQA attention, online softmax, f32. out[q][h][c]. `causal`
    /// treats the seq_q queries as the last positions of the seq_kv keys
    /// (square when equal — text encoder; seq_q 1 — KV-cached LLM decode);
    /// the DiT passes false.
    pub fn attn(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, seq_q: usize, seq_kv: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32, causal: bool) Error!void {
        self.ptic();
        defer self.ptoc(.attn);
        std.debug.assert(seq_q <= seq_kv);
        const f = try self.eltFn(elt.attn_ptx, "attn");
        try self.eltLaunch(f, q, k, v, out, .{ @intCast(seq_q), @intCast(n_heads), @intCast(kv_heads), @intCast(hd), @intFromBool(causal), @intCast(seq_kv) }, .{ scale, 0 }, seq_q * n_heads);
    }

    /// rotate-half RoPE in place. total = rows*n_heads*half. `pos0` offsets
    /// the freqs row (rows hold absolute positions pos0.. — KV-cached decode).
    pub fn ropeHalf(self: *Backend, qk: DeviceBuffer, freqs: DeviceBuffer, rows: usize, n_heads: usize, half: usize, sin_off: usize, pos0: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.rope_half_ptx, "rope_half");
        const total = rows * n_heads * half;
        try self.eltLaunch(f, qk, null, freqs, null, .{ @intCast(total), @intCast(half), @intCast(sin_off), @intCast(n_heads), @intCast(pos0), 0 }, .{ 0, 0 }, total);
    }

    /// a += b, in place (plain residual add). total = element count.
    pub fn opAdd(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.add_ptx, "add");
        try self.eltLaunch(f, a, b, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// VAE per-position channel L2 norm (+ optional fused silu). x/out [n][c]
    /// channel-last, gamma [c]. One thread per position.
    pub fn opVaeNorm(self: *Backend, x: DeviceBuffer, out: DeviceBuffer, gamma: DeviceBuffer, n: usize, c: usize, silu: bool) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.vae_norm_ptx, "vae_norm");
        try self.eltLaunch(f, x, out, gamma, null, .{ @intCast(n), @intCast(c), @intFromBool(silu), 0, 0, 0 }, .{ 1e-12, 0 }, n);
    }

    /// im2col for a 3x3 conv band: patch[bn][9*ci] from src[h*w][ci], zero-padded;
    /// `up` reads a fused nearest-exact 2x upsample (coords halve). p0 = first
    /// output position of the band. One thread per output f32 (bn*9*ci total).
    pub fn opIm2col(self: *Backend, src: DeviceBuffer, patch: DeviceBuffer, bn: usize, patch_len: usize, ci: usize, w: usize, h: usize, p0: usize, up: bool) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.im2col_ptx, "im2col");
        const total = bn * patch_len;
        try self.eltLaunch(f, src, patch, null, null, .{ @intCast(total), @intCast(patch_len), @intCast(ci), @intCast(w), @intCast(h), @intCast(p0) }, .{ if (up) 1.0 else 0.0, 0 }, total);
    }

    /// Tensor-core GQA attention: out[q][h][hd] = softmax(scale·Q·Kᵀ)·V. Q/K/V are
    /// the interleaved [seq][*][hd] f32 tensors from rope; heads are gathered to
    /// contiguous f16 (V transposed to [hd][mpad]), run through hgemm→softmax→hgemm
    /// on tensor cores, and scattered back. GQA: Q head h reads KV head h/group.
    /// Dispatches to the head-batched path (grid.z fills the GPU) or the per-head
    /// loop reference.
    pub fn opAttnTC(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, seq: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32) Error!void {
        if (self.kernels == .libs) {
            self.ptic();
            defer self.ptoc(.attn);
            return self.opAttnCudnn(q, k, v, out, seq, n_heads, kv_heads, hd, scale);
        }
        if (self.attn_batched) {
            // sub-timed internally (attn_scores/softmax/pv + gather/scatter in .attn)
            try self.opAttnTCBatched(q, k, v, out, seq, n_heads, kv_heads, hd, scale);
        } else {
            self.ptic();
            defer self.ptoc(.attn);
            try self.opAttnTCLoop(q, k, v, out, seq, n_heads, kv_heads, hd, scale);
        }
    }

    /// Build (or fetch cached) a cuDNN fused-SDPA plan for this GQA shape.
    fn sdpaPlan(self: *Backend, n_heads: usize, kv_heads: usize, seq: usize, hd: usize) Error!cudnn.SdpaPlan {
        const key: u64 = (@as(u64, seq) << 24) | (@as(u64, n_heads) << 16) | (@as(u64, kv_heads) << 8) | @as(u64, hd / 8);
        if (self.sdpa_plans.get(key)) |p| return p;
        const L = &self.libs.?;
        const p = cudnn.SdpaPlan.build(&L.dnn, L.dnn_handle, 1, n_heads, kv_heads, seq, hd) catch return error.CudaError;
        self.sdpa_plans.put(self.gpa, key, p) catch return error.OutOfMemory;
        return p;
    }

    /// cuDNN fused flash attention (.libs mode): O = softmax(scale·Q·Kᵀ)·V in one
    /// fused kernel — no per-head gather/scatter, no S materialization, no seq
    /// padding, native GQA. Q/K/V/O are the DiT's f32 [seq][heads][hd] buffers;
    /// they convert to f16 for the op and O converts back to f32. Replaces the
    /// whole hand-PTX scores/softmax/pv pipeline (~80× faster on the GEMMs).
    fn opAttnCudnn(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, seq: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32) Error!void {
        const qn = seq * n_heads * hd;
        const kn = seq * kv_heads * hd;
        try self.ensureDeviceBuffer(&self.cudnn_q16, qn * 2);
        try self.ensureDeviceBuffer(&self.cudnn_k16, kn * 2);
        try self.ensureDeviceBuffer(&self.cudnn_v16, kn * 2);
        try self.ensureDeviceBuffer(&self.cudnn_o16, qn * 2);
        const f_cvt = try self.eltFn(elt.f32_to_f16_ptx, "f32_to_f16");
        try self.eltLaunch(f_cvt, q, self.cudnn_q16, null, null, .{ @intCast(qn), @intCast(qn), 0, 0, 0, 0 }, .{ 0, 0 }, qn);
        try self.eltLaunch(f_cvt, k, self.cudnn_k16, null, null, .{ @intCast(kn), @intCast(kn), 0, 0, 0, 0 }, .{ 0, 0 }, kn);
        try self.eltLaunch(f_cvt, v, self.cudnn_v16, null, null, .{ @intCast(kn), @intCast(kn), 0, 0, 0, 0 }, .{ 0, 0 }, kn);
        const plan = try self.sdpaPlan(n_heads, kv_heads, seq, hd);
        if (plan.workspace_bytes > 0) try self.ensureDeviceBuffer(&self.cudnn_ws, plan.workspace_bytes);
        var sc = scale;
        const L = &self.libs.?;
        plan.execute(&L.dnn, L.dnn_handle, self.cudnn_q16.ptr(), self.cudnn_k16.ptr(), self.cudnn_v16.ptr(), self.cudnn_o16.ptr(), &sc, self.cudnn_ws.ptr()) catch return error.CudaError;
        const f_back = try self.eltFn(elt.f16_to_f32_ptx, "f16_to_f32");
        try self.eltLaunch(f_back, self.cudnn_o16, out, null, null, .{ @intCast(qn), @intCast(qn), 0, 0, 0, 0 }, .{ 0, 0 }, qn);
    }

    /// Head-batched attention: process `G` heads per launch (grid.z=G) so the PV
    /// GEMM (n=hd=128 → grid.x=1) and the whole pipeline fill the SMs, and the
    /// per-head launch count collapses to ~7·(n_heads/G). G is capped so the
    /// scores+probs scratch fits `attn_scratch_budget`.
    fn opAttnTCBatched(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, seq: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32) Error!void {
        const mpad = std.mem.alignForward(usize, seq, 128);
        const group = n_heads / kv_heads;
        const fused = self.attn_fused;
        // scratch/head: S (f16) always; fused adds a tiny MD table, the materialized
        // path adds a full P (f16) — so the fused path fits ~2× the heads per launch.
        const per_head = if (fused) mpad * mpad * 2 + mpad * 8 else mpad * mpad * 4;
        // A --vram-budget shrinks the scores scratch (more head-batches) — the
        // single biggest activation buffer at high resolution.
        const cap = if (self.budget_override != 0) @min(self.attn_scratch_budget, @max(64 << 20, self.budget_override / 4)) else self.attn_scratch_budget;
        var g = cap / per_head;
        if (g < 1) g = 1;
        if (g > n_heads) g = n_heads;

        // scores S are f16 (halves the S write + softmax reads — the memory-bound
        // cost at large seq); O keeps f32. Fused: MD={max,1/sum} f32 pairs replaces
        // the materialized P; non-fused: P is a full f16 [gs·mpad][mpad] tile.
        try self.ensureDeviceBuffer(&self.attn_qh, g * mpad * hd * 2);
        try self.ensureDeviceBuffer(&self.attn_kh, g * mpad * hd * 2);
        try self.ensureDeviceBuffer(&self.attn_vth, g * hd * mpad * 2);
        try self.ensureDeviceBuffer(&self.attn_s, g * mpad * mpad * 2);
        if (fused)
            try self.ensureDeviceBuffer(&self.attn_md, g * mpad * 8)
        else
            try self.ensureDeviceBuffer(&self.attn_p, g * mpad * mpad * 2);
        try self.ensureDeviceBuffer(&self.attn_oh, g * mpad * hd * 4);

        const f_scores = try self.hgemmBatchedC16Fn(); // f16-C scores GEMM
        const f_pv = if (fused) try self.hgemmAttnOutFn() else try self.hgemmBatchedFn();
        const f_sm = if (fused)
            try self.eltFn(kernels.softmax_md_f16_ptx, "softmax_md_f16")
        else
            try self.eltFn(kernels.softmax_row_f16_ptx, "softmax_row_f16");
        const f_ghb = try self.eltFn(elt.gather_head_b_ptx, "gather_head_b");
        const f_gvtb = try self.eltFn(elt.gather_vt_b_ptx, "gather_vt_b");
        const f_scb = try self.eltFn(elt.scatter_head_b_ptx, "scatter_head_b");

        // per-head strides (elements): scores A=Q,B=K stride s_qk, C=S stride s_s;
        // PV A=P/S stride s_s, B=Vt stride s_vt, C=O stride s_o.
        const s_qk: u32 = @intCast(mpad * hd);
        const s_vt: u32 = @intCast(hd * mpad);
        const s_s: u32 = @intCast(mpad * mpad);
        const s_o: u32 = @intCast(mpad * hd);
        const seq32: u32 = @intCast(seq);
        const hd32: u32 = @intCast(hd);
        const nh32: u32 = @intCast(n_heads);
        const kvh32: u32 = @intCast(kv_heads);
        const mpad32: u32 = @intCast(mpad);
        const grp32: u32 = @intCast(group);

        var base: usize = 0;
        while (base < n_heads) : (base += g) {
            const gs = @min(g, n_heads - base);
            const gs32: u32 = @intCast(gs);
            const bh: u32 = @intCast(base);
            // gather Q (group_div=1), K (group_div=group), Vt for gs heads
            self.ptic();
            try self.launch7(f_ghb, .{ q.ptr(), self.attn_qh.ptr() }, .{ seq32, nh32, bh, 1, hd32, mpad32, gs32 * mpad32 * hd32 }, gs * mpad * hd);
            try self.launch7(f_ghb, .{ k.ptr(), self.attn_kh.ptr() }, .{ seq32, kvh32, bh, grp32, hd32, mpad32, gs32 * mpad32 * hd32 }, gs * mpad * hd);
            try self.launch7(f_gvtb, .{ v.ptr(), self.attn_vth.ptr() }, .{ seq32, kvh32, bh, grp32, hd32, mpad32, gs32 * hd32 * mpad32 }, gs * hd * mpad);
            self.ptoc(.attn);
            // scores S[gs][mpad][mpad] f16 = scale·(Q @ Kᵀ)  (scale prefolded in
            // the C-store so f16 S can't overflow; softmax then uses scale=1)
            self.ptic();
            try self.launchHgemmB(f_scores, self.attn_qh, self.attn_kh, self.attn_s, mpad, mpad, hd, gs, s_qk, s_qk, s_s, scale);
            self.ptoc(.attn_scores);
            if (fused) {
                // MD[gs·mpad][2] f32 = {max, 1/sum} per row (one S read, no P write)
                self.ptic();
                try self.launchSoftmaxMd(f_sm, self.attn_s, self.attn_md, gs * mpad, mpad, seq);
                self.ptoc(.attn_softmax);
                // O[gs][mpad][hd] f32 = (softmax S) @ Vtᵀ — P recomputed in-GEMM
                self.ptic();
                try self.launchAttnOut(f_pv, self.attn_s, self.attn_vth, self.attn_oh, self.attn_md, mpad, hd, gs, s_s, s_vt, s_o, seq32, mpad32);
                self.ptoc(.attn_pv);
            } else {
                // P[gs·mpad][mpad] f16 = softmax(S) — flat over all gs heads' rows
                self.ptic();
                try self.launchSoftmax(f_sm, self.attn_s, self.attn_p, gs * mpad, mpad, seq, 1.0);
                self.ptoc(.attn_softmax);
                // O[gs][mpad][hd] f32 = P @ Vtᵀ  (m=mpad, n=hd, k=mpad)
                self.ptic();
                try self.launchHgemmB(f_pv, self.attn_p, self.attn_vth, self.attn_oh, mpad, hd, mpad, gs, s_s, s_vt, s_o, 1.0);
                self.ptoc(.attn_pv);
            }
            // scatter O rows 0..seq into out[q][base+z][hd]
            self.ptic();
            try self.launch7(f_scb, .{ self.attn_oh.ptr(), out.ptr() }, .{ seq32, nh32, bh, hd32, mpad32, gs32 * seq32 * hd32, 0 }, gs * seq * hd);
            self.ptoc(.attn);
        }
    }

    /// Per-head attention reference (A/B against the batched path). Scratch reused
    /// across heads, so VRAM is capped at one head's scores.
    fn opAttnTCLoop(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, seq: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32) Error!void {
        const mpad = std.mem.alignForward(usize, seq, 128);
        const group = n_heads / kv_heads;
        try self.ensureDeviceBuffer(&self.attn_qh, mpad * hd * 2);
        try self.ensureDeviceBuffer(&self.attn_kh, mpad * hd * 2);
        try self.ensureDeviceBuffer(&self.attn_vth, hd * mpad * 2);
        try self.ensureDeviceBuffer(&self.attn_s, mpad * mpad * 4);
        try self.ensureDeviceBuffer(&self.attn_p, mpad * mpad * 2);
        try self.ensureDeviceBuffer(&self.attn_oh, mpad * hd * 4);

        const f_hg = try self.hgemmFn();
        const f_sm = try self.eltFn(kernels.softmax_row_ptx, "softmax_row");
        const f_gh = try self.eltFn(elt.gather_head_ptx, "gather_head");
        const f_gvt = try self.eltFn(elt.gather_vt_ptx, "gather_vt");
        const f_sc = try self.eltFn(elt.scatter_head_ptx, "scatter_head");

        const mpad32: u32 = @intCast(mpad);
        const seq32: u32 = @intCast(seq);
        const hd32: u32 = @intCast(hd);
        const nh32: u32 = @intCast(n_heads);
        const kvh32: u32 = @intCast(kv_heads);

        for (0..n_heads) |h| {
            const kvh: u32 = @intCast(h / group);
            const head32: u32 = @intCast(h);
            try self.launch7(f_gh, .{ q.ptr(), self.attn_qh.ptr() }, .{ seq32, nh32, head32, hd32, mpad32 * hd32, 0, 0 }, mpad * hd);
            try self.launch7(f_gh, .{ k.ptr(), self.attn_kh.ptr() }, .{ seq32, kvh32, kvh, hd32, mpad32 * hd32, 0, 0 }, mpad * hd);
            try self.launch7(f_gvt, .{ v.ptr(), self.attn_vth.ptr() }, .{ seq32, kvh32, kvh, hd32, mpad32, hd32 * mpad32, 0 }, hd * mpad);
            try self.launchHgemm(f_hg, self.attn_qh, self.attn_kh, self.attn_s, mpad, mpad, hd);
            try self.launchSoftmax(f_sm, self.attn_s, self.attn_p, mpad, mpad, seq, scale);
            try self.launchHgemm(f_hg, self.attn_p, self.attn_vth, self.attn_oh, mpad, hd, mpad);
            try self.launch7(f_sc, .{ self.attn_oh.ptr(), out.ptr() }, .{ seq32, nh32, head32, hd32, seq32 * hd32, 0, 0 }, seq * hd);
        }
    }

    /// Launch a 2-buffer / up-to-7-u32 kernel (the gather/scatter signature).
    fn launch7(self: *Backend, f: cu.CUfunction, bufs: [2]cu.CUdeviceptr, u: [7]u32, total: usize) Error!void {
        var p0 = bufs[0];
        var p1 = bufs[1];
        var uu = u;
        var params = [_]?*anyopaque{
            @ptrCast(&p0),    @ptrCast(&p1),    @ptrCast(&uu[0]), @ptrCast(&uu[1]),
            @ptrCast(&uu[2]), @ptrCast(&uu[3]), @ptrCast(&uu[4]), @ptrCast(&uu[5]),
            @ptrCast(&uu[6]),
        };
        // param count is fixed by the entry (gather_vt uses 8, others 7); passing
        // extra pointers is harmless — the driver reads only what the entry declares.
        const grid: u32 = @intCast((total + 255) / 256);
        self.ctx.launch(f, .{ grid, 1, 1 }, .{ 256, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// hgemm: C[m][n] f32 = A[m][k] f16 @ B[n][k] f16ᵀ. m,n multiples of 128.
    fn launchHgemm(self: *Backend, f: cu.CUfunction, a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer, m: usize, n: usize, kk: usize) Error!void {
        var pa = a.ptr();
        var pb = b.ptr();
        var pc = c.ptr();
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(kk);
        var params = [_]?*anyopaque{ @ptrCast(&pa), @ptrCast(&pb), @ptrCast(&pc), @ptrCast(&pn), @ptrCast(&pk) };
        self.ctx.launch(f, .{ @intCast(n / 128), @intCast(m / 128), 1 }, .{ 128, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// batched hgemm: gs independent C[m][n] GEMMs (grid.z=gs), per-head element
    /// strides sa/sb/sc. Same 128×128 tiling as hgemm.
    fn launchHgemmB(self: *Backend, f: cu.CUfunction, a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer, m: usize, n: usize, kk: usize, gs: usize, sa: u32, sb: u32, sc: u32, scale: f32) Error!void {
        var pa = a.ptr();
        var pb = b.ptr();
        var pc = c.ptr();
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(kk);
        var psa = sa;
        var psb = sb;
        var psc = sc;
        var pscale = scale;
        var params = [_]?*anyopaque{
            @ptrCast(&pa),  @ptrCast(&pb),  @ptrCast(&pc),  @ptrCast(&pn),
            @ptrCast(&pk),  @ptrCast(&psa), @ptrCast(&psb), @ptrCast(&psc),
            @ptrCast(&pscale),
        };
        self.ctx.launch(f, .{ @intCast(n / 128), @intCast(m / 128), @intCast(gs) }, .{ 128, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// Fused attention-output GEMM: O[gs][m][n] f32 = softmax(S)[gs][m][k] @ Vt[gs][n][k]ᵀ,
    /// where P = exp(S-max)/sum is recomputed per element from S (f16, A operand,
    /// stride sa) + the MD table (per-row {max,1/sum}, per-head row stride mds=mpad)
    /// during A-staging. m=mpad, n=hd, k=mpad. p_scale is 1 (P is already normalized).
    fn launchAttnOut(self: *Backend, f: cu.CUfunction, s: DeviceBuffer, vt: DeviceBuffer, o: DeviceBuffer, md: DeviceBuffer, m: usize, n: usize, gs: usize, sa: u32, sb: u32, sc: u32, seq: u32, mds: u32) Error!void {
        var pa = s.ptr();
        var pb = vt.ptr();
        var pc = o.ptr();
        var pn: u32 = @intCast(n);
        var pk: u32 = @intCast(m); // k = mpad = m
        var psa = sa;
        var psb = sb;
        var psc = sc;
        var pscale: f32 = 1.0;
        var pmd = md.ptr();
        var pseq = seq;
        var pmds = mds;
        var params = [_]?*anyopaque{
            @ptrCast(&pa),    @ptrCast(&pb),  @ptrCast(&pc),  @ptrCast(&pn),
            @ptrCast(&pk),    @ptrCast(&psa), @ptrCast(&psb), @ptrCast(&psc),
            @ptrCast(&pscale), @ptrCast(&pmd), @ptrCast(&pseq), @ptrCast(&pmds),
        };
        self.ctx.launch(f, .{ @intCast(n / 128), @intCast(m / 128), @intCast(gs) }, .{ 128, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// softmax_md_f16: MD[rows][2] f32 = {max, 1/sum} per row of S[rows][pn] f16
    /// (valid cols 0..seq). One block (256) per row; static 2 KiB shared.
    fn launchSoftmaxMd(self: *Backend, f: cu.CUfunction, s: DeviceBuffer, md: DeviceBuffer, rows: usize, pn: usize, seq: usize) Error!void {
        var ps = s.ptr();
        var pmd = md.ptr();
        var pnn: u32 = @intCast(pn);
        var pseq: u32 = @intCast(seq);
        var params = [_]?*anyopaque{ @ptrCast(&ps), @ptrCast(&pmd), @ptrCast(&pnn), @ptrCast(&pseq) };
        self.ctx.launch(f, .{ @intCast(rows), 1, 1 }, .{ 256, 1, 1 }, 0, &params) catch return error.CudaError;
    }

    /// softmax_row: P[rows][pn] f16 = softmax(scale·S[rows][pn]) over valid cols
    /// 0..seq, pad cols → 0. One block (256) per row; 1 KiB shared for the reduction.
    /// `rows` may span several batched heads (S contiguous [gs·mpad][pn]).
    fn launchSoftmax(self: *Backend, f: cu.CUfunction, s: DeviceBuffer, p: DeviceBuffer, rows: usize, pn: usize, seq: usize, scale: f32) Error!void {
        var ps = s.ptr();
        var pp = p.ptr();
        var pnn: u32 = @intCast(pn); // padded width = mpad
        var pseq: u32 = @intCast(seq);
        var psc: f32 = scale;
        var params = [_]?*anyopaque{ @ptrCast(&ps), @ptrCast(&pp), @ptrCast(&pnn), @ptrCast(&pseq), @ptrCast(&psc) };
        self.ctx.launch(f, .{ @intCast(rows), 1, 1 }, .{ 256, 1, 1 }, 256 * 4, &params) catch return error.CudaError;
    }

    /// a *= sigmoid(b), in place.
    pub fn sigmoidMul(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.sigmoid_mul_ptx, "sigmoid_mul");
        try self.eltLaunch(f, a, b, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// a = silu(a)*b, in place (a=gate, b=up).
    /// f16 SwiGLU gate (c16 chain): a = silu(a)·b, all f16.
    pub fn siluMul16(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.silu_mul_h16_ptx, "silu_mul_h16");
        try self.eltLaunch(f, a, b, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    pub fn siluMul(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.silu_mul_ptx, "silu_mul");
        try self.eltLaunch(f, a, b, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// a = geluTanh(a), in place. total = element count.
    pub fn gelu(self: *Backend, a: DeviceBuffer, total: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.gelu_ptx, "gelu");
        try self.eltLaunch(f, a, null, null, null, .{ @intCast(total), 0, 0, 0, 0, 0 }, .{ 0, 0 }, total);
    }

    /// a += mod[gate_off + col] * b, in place (residual with gate). total=rows*dim.
    pub fn gatedAdd(self: *Backend, a: DeviceBuffer, b: DeviceBuffer, mod: DeviceBuffer, total: usize, dim: usize, gate_off: usize) Error!void {
        self.ptic();
        defer self.ptoc(.elt);
        const f = try self.eltFn(elt.gated_add_ptx, "gated_add");
        try self.eltLaunch(f, a, b, mod, null, .{ @intCast(total), @intCast(dim), @intCast(gate_off), 0, 0, 0 }, .{ 0, 0 }, total);
    }
};

test {
    _ = Backend;
}

/// Random ggml block-quant weight bytes with every block's f16 scale fields
/// pinned to small finite values (random u16 bit patterns include NaN/inf).
fn testQuantWeightBytes(gpa: std.mem.Allocator, dt: dtypes.DType, rows: usize, cols: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    const wbytes = try gpa.alloc(u8, dt.storageBytes(rows * cols));
    rand.bytes(wbytes);
    const d16: u16 = 0x2A66; // 0.05
    const min16: u16 = 0x251F; // 0.02
    const bb = dt.blockBytes();
    var off: usize = 0;
    while (off < wbytes.len) : (off += bb) {
        switch (dt) {
            .q8_0 => std.mem.writeInt(u16, wbytes[off..][0..2], d16, .little),
            .q4_k, .q5_k => {
                std.mem.writeInt(u16, wbytes[off..][0..2], d16, .little);
                std.mem.writeInt(u16, wbytes[off + 2 ..][0..2], min16, .little);
            },
            .q6_k => std.mem.writeInt(u16, wbytes[off + 208 ..][0..2], d16, .little),
            else => unreachable,
        }
    }
    return wbytes;
}

// Gated on a CUDA device: the fused block-quant GEMVs against the CPU
// quants.zig dequant + dot reference, all four formats.
test "gemv quant kernels match CPU reference" {
    const quants = @import("../../quants.zig");
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 16;
    const cols = 512; // two super-blocks: exercises the shared scale table
    var prng = std.Random.DefaultPrng.init(4242);
    const rand = prng.random();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;
    const x_d = try be.tensorCreate(cols * 4);
    const y_d = try be.tensorCreate(rows * 4);
    defer {
        var xd = x_d;
        var yd = y_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));

    const row_f32 = try gpa.alloc(f32, cols);
    defer gpa.free(row_f32);
    const y = try gpa.alloc(f32, rows);
    defer gpa.free(y);

    // All four weight buffers stay alive for the whole test: the device
    // weight cache is keyed by host pointer, so free-then-realloc at the
    // same address would alias a stale upload.
    const dts = [_]dtypes.DType{ .q8_0, .q4_k, .q5_k, .q6_k };
    var ws: [dts.len][]u8 = undefined;
    inline for (dts, 0..) |dt, i| ws[i] = try testQuantWeightBytes(gpa, dt, rows, cols, 100 + i);
    defer for (ws) |w| gpa.free(w);

    inline for (dts, 0..) |dt, i| {
        const w = ws[i];
        try be.opGemvQuant(dt, y_d, x_d, w, 1.0, rows, cols);
        try be.tensorDownload(y_d, std.mem.sliceAsBytes(y));
        const row_bytes = dt.storageBytes(cols);
        for (0..rows) |r| {
            quants.dequantSlice(dt, w[r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
            var acc: f64 = 0;
            for (row_f32, x) |wv, xv| acc += @as(f64, wv) * xv;
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), y[r], 2e-2);
        }
    }
}

// Gated on a CUDA device: the dp4a GEMV path (opGemvQuantizeX +
// opGemvQuantQ8) against a CPU reference that emulates the same q8
// activation quantization (d = amax/127 per 32-elem block); the GPU then
// differs only by accumulation order and rni tie-rounding.
test "dp4a gemv quant kernels match CPU reference" {
    const quants = @import("../../quants.zig");
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 16;
    const cols = 512;
    var prng = std.Random.DefaultPrng.init(1337);
    const rand = prng.random();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;
    const x_d = try be.tensorCreate(cols * 4);
    const y_d = try be.tensorCreate(rows * 4);
    defer {
        var xd = x_d;
        var yd = y_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
    try be.opGemvQuantizeX(x_d, cols);

    // CPU emulation of quantize_q8_1: x̂ = d * rni(x * 127/amax) per block.
    const xq = try gpa.alloc(f32, cols);
    defer gpa.free(xq);
    var blk: usize = 0;
    while (blk < cols / 32) : (blk += 1) {
        var amax: f32 = 0;
        for (x[blk * 32 ..][0..32]) |xi| amax = @max(amax, @abs(xi));
        const d: f32 = amax / 127.0;
        const inv: f32 = if (amax == 0) 0 else 127.0 / amax;
        for (0..32) |i| xq[blk * 32 + i] = d * @round(x[blk * 32 + i] * inv);
    }

    const row_f32 = try gpa.alloc(f32, cols);
    defer gpa.free(row_f32);
    const y = try gpa.alloc(f32, rows);
    defer gpa.free(y);

    const dts = [_]dtypes.DType{ .q5_k, .q6_k };
    var ws: [dts.len][]u8 = undefined;
    inline for (dts, 0..) |dt, i| ws[i] = try testQuantWeightBytes(gpa, dt, rows, cols, 300 + i);
    defer for (ws) |w| gpa.free(w);

    inline for (dts, 0..) |dt, i| {
        const w = ws[i];
        try be.opGemvQuantQ8(dt, y_d, w, 1.0, rows, cols);
        try be.tensorDownload(y_d, std.mem.sliceAsBytes(y));
        const row_bytes = dt.storageBytes(cols);
        for (0..rows) |r| {
            quants.dequantSlice(dt, w[r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
            var acc: f64 = 0;
            for (row_f32, xq) |wv, xv| acc += @as(f64, wv) * xv;
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), y[r], 2e-2);
        }
    }

    // Grouped variant (small-batch prefill): 3 activation rows quantized as
    // one vector, each output row must match the same CPU reference.
    const n = 3;
    const xn = try gpa.alloc(f32, n * cols);
    defer gpa.free(xn);
    for (xn) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;
    const xn_d = try be.tensorCreate(n * cols * 4);
    const yn_d = try be.tensorCreate(n * rows * 4);
    defer {
        var xd = xn_d;
        var yd = yn_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(xn_d, std.mem.sliceAsBytes(xn));
    try be.opGemvQuantizeX(xn_d, n * cols);

    const xnq = try gpa.alloc(f32, n * cols);
    defer gpa.free(xnq);
    blk = 0;
    while (blk < n * cols / 32) : (blk += 1) {
        var amax: f32 = 0;
        for (xn[blk * 32 ..][0..32]) |xi| amax = @max(amax, @abs(xi));
        const d: f32 = amax / 127.0;
        const inv: f32 = if (amax == 0) 0 else 127.0 / amax;
        for (0..32) |j| xnq[blk * 32 + j] = d * @round(xn[blk * 32 + j] * inv);
    }

    const yn = try gpa.alloc(f32, n * rows);
    defer gpa.free(yn);
    inline for (dts, 0..) |dt, i| {
        const w = ws[i];
        try be.opGemvQuantQ8N(dt, yn_d, w, 1.0, rows, cols, n, 0, n);
        try be.tensorDownload(yn_d, std.mem.sliceAsBytes(yn));
        const row_bytes = dt.storageBytes(cols);
        for (0..n) |t| {
            for (0..rows) |r| {
                quants.dequantSlice(dt, w[r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
                var acc: f64 = 0;
                for (row_f32, xnq[t * cols ..][0..cols]) |wv, xv| acc += @as(f64, wv) * xv;
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), yn[t * rows + r], 2e-2);
            }
        }
    }
}

// Gated on a CUDA device: the dequant-to-f16 + tensor-core GEMM prefill path
// against the CPU reference (f16 weight rounding bounds the tolerance).
test "opMatmulQuant matches CPU reference" {
    const quants = @import("../../quants.zig");
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const m = 3;
    const mpad = 128;
    const rows = 128;
    const cols = 512;
    var prng = std.Random.DefaultPrng.init(777);
    const rand = prng.random();

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;
    const x_d = try be.tensorCreate(m * cols * 4);
    const y_d = try be.tensorCreate(mpad * rows * 4);
    defer {
        var xd = x_d;
        var yd = y_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));

    const w = try testQuantWeightBytes(gpa, .q4_k, rows, cols, 55);
    defer gpa.free(w);
    try be.opMatmulQuant(.q4_k, y_d, x_d, m, w, rows, cols);
    const y = try gpa.alloc(f32, mpad * rows);
    defer gpa.free(y);
    try be.tensorDownload(y_d, std.mem.sliceAsBytes(y));

    const row_bytes = dtypes.DType.q4_k.storageBytes(cols);
    const row_f32 = try gpa.alloc(f32, cols);
    defer gpa.free(row_f32);
    for (0..m) |t| {
        for (0..rows) |r| {
            quants.dequantSlice(.q4_k, w[r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
            var acc: f64 = 0;
            for (row_f32, 0..) |wv, c| {
                // The GEMM rounds both operands to f16; emulate it so the
                // reference differs only by accumulation order.
                const wf: f32 = @floatCast(@as(f16, @floatCast(wv)));
                const xf: f32 = @floatCast(@as(f16, @floatCast(x[t * cols + c])));
                acc += @as(f64, wf) * xf;
            }
            const e: f32 = @floatCast(acc);
            try std.testing.expectApproxEqAbs(e, y[t * rows + r], 0.02 + 1e-3 * @abs(e));
        }
    }
}

// Gated on a CUDA device: the graph-mode quant embed gathers against the CPU
// row dequant, all four formats.
test "embed gather quant kernels match CPU reference" {
    const quants = @import("../../quants.zig");
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const vocab = 8;
    const h = 256;
    const token = 5;
    const x_d = try be.tensorCreate(h * 4);
    defer {
        var xd = x_d;
        be.tensorDestroy(&xd);
    }
    const x = try gpa.alloc(f32, h);
    defer gpa.free(x);
    const row_f32 = try gpa.alloc(f32, h);
    defer gpa.free(row_f32);

    try be.setDecodeState(token, 0);
    // Weights outlive the loop: the device weight cache keys by host pointer.
    const dts = [_]dtypes.DType{ .q8_0, .q4_k, .q5_k, .q6_k };
    var ws: [dts.len][]u8 = undefined;
    inline for (dts, 0..) |dt, i| ws[i] = try testQuantWeightBytes(gpa, dt, vocab, h, 900 + i);
    defer for (ws) |w| gpa.free(w);

    inline for (dts, 0..) |dt, i| {
        const w = ws[i];
        try be.opEmbedGatherQuant(dt, x_d, w, h);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(x));
        const row_bytes = dt.storageBytes(h);
        quants.dequantSlice(dt, w[token * row_bytes ..][0..row_bytes], 0, h, row_f32);
        for (row_f32, x) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
    }
}

// Gated on a CUDA device: the qwen35 gated-delta-net op chain (gates,
// conv step, l2norm, delta step) against naive CPU reference math.
test "gdn ops match CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const heads = 4;
    const k_heads = 2;
    const d = 32;
    const channels = 2 * k_heads * d + heads * d; // q|k|v = 256
    const eps: f32 = 1e-6;
    const scale = 1.0 / @sqrt(@as(f32, d));
    var prng = std.Random.DefaultPrng.init(2024);
    const rand = prng.random();

    // Host inputs.
    var qkv: [channels]f32 = undefined;
    var conv_st: [channels * 3]f32 = undefined;
    var conv_w: [channels * 4]f32 = undefined;
    var ab: [2 * heads]f32 = undefined;
    var a_dt: [2 * heads]f32 = undefined;
    var state: [heads * d * d]f32 = undefined;
    for (&qkv) |*x| x.* = rand.floatNorm(f32);
    for (&conv_st) |*x| x.* = rand.floatNorm(f32);
    for (&conv_w) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (&ab) |*x| x.* = rand.floatNorm(f32);
    for (&state) |*x| x.* = rand.floatNorm(f32) * 0.1;
    for (a_dt[0..heads]) |*x| x.* = -rand.float(f32) - 0.1; // a = -exp(A_log) < 0
    for (a_dt[heads..]) |*x| x.* = rand.floatNorm(f32);

    // --- CPU reference ---
    var ref_conv_st = conv_st;
    var ref_conv: [channels]f32 = undefined;
    for (0..channels) |c| {
        const st = ref_conv_st[c * 3 ..][0..3];
        const w = conv_w[c * 4 ..][0..4];
        var acc: f32 = w[3] * qkv[c];
        for (0..3) |k| acc += w[k] * st[k];
        st[0] = st[1];
        st[1] = st[2];
        st[2] = qkv[c];
        ref_conv[c] = acc / (1.0 + @exp(-acc));
    }
    var ref_gates: [2 * heads]f32 = undefined;
    for (0..heads) |h| {
        const x = ab[h] + a_dt[heads + h];
        const sp = if (x > 20.0) x else @log(1.0 + @exp(x));
        ref_gates[h] = @exp(a_dt[h] * sp);
        ref_gates[heads + h] = 1.0 / (1.0 + @exp(-ab[heads + h]));
    }
    // l2norm the q|k head rows.
    for (0..2 * k_heads) |r| {
        const row = ref_conv[r * d ..][0..d];
        var ss: f32 = 0;
        for (row) |x| ss += x * x;
        const sc = 1.0 / @max(@sqrt(ss), eps);
        for (row) |*x| x.* *= sc;
    }
    var ref_state = state;
    var ref_o: [heads * d]f32 = undefined;
    for (0..heads) |h| {
        const qh = ref_conv[(h % k_heads) * d ..][0..d];
        const kh = ref_conv[k_heads * d + (h % k_heads) * d ..][0..d];
        const vh = ref_conv[2 * k_heads * d + h * d ..][0..d];
        const S = ref_state[h * d * d ..][0 .. d * d];
        var m: [d]f32 = @splat(0);
        for (0..d) |i| {
            const row = S[i * d ..][0..d];
            for (row, &m) |*sij, *mj| {
                sij.* *= ref_gates[h];
                mj.* += sij.* * kh[i];
            }
        }
        var dl: [d]f32 = undefined;
        for (0..d) |j| dl[j] = (vh[j] - m[j]) * ref_gates[heads + h];
        const oh = ref_o[h * d ..][0..d];
        @memset(oh, 0);
        for (0..d) |i| {
            const row = S[i * d ..][0..d];
            const qi = qh[i] * scale;
            for (row, dl, oh) |*sij, dj, *oj| {
                sij.* += kh[i] * dj;
                oj.* += sij.* * qi;
            }
        }
    }

    // --- Device ---
    const st_d = try be.tensorCreate(conv_st.len * 4);
    const qkv_d = try be.tensorCreate(channels * 4);
    const cw_d = try be.tensorCreate(conv_w.len * 4);
    const conv_d = try be.tensorCreate(channels * 4);
    const ab_d = try be.tensorCreate(ab.len * 4);
    const adt_d = try be.tensorCreate(a_dt.len * 4);
    const gates_d = try be.tensorCreate(ab.len * 4);
    const state_d = try be.tensorCreate(state.len * 4);
    const o_d = try be.tensorCreate(heads * d * 4);
    defer {
        inline for (.{ st_d, qkv_d, cw_d, conv_d, ab_d, adt_d, gates_d, state_d, o_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(st_d, std.mem.sliceAsBytes(&conv_st));
    try be.tensorUpload(qkv_d, std.mem.sliceAsBytes(&qkv));
    try be.tensorUpload(cw_d, std.mem.sliceAsBytes(&conv_w));
    try be.tensorUpload(ab_d, std.mem.sliceAsBytes(&ab));
    try be.tensorUpload(adt_d, std.mem.sliceAsBytes(&a_dt));
    try be.tensorUpload(state_d, std.mem.sliceAsBytes(&state));

    try be.opGdnGates(ab_d, adt_d, gates_d, heads);
    try be.opGdnConvStep(st_d, qkv_d, cw_d, conv_d, channels);
    try be.opL2NormRows(offsetBufSizedTest(conv_d, 0, 2 * k_heads * d * 4), 2 * k_heads, d, eps);
    try be.opGdnDeltaStep(state_d, conv_d, gates_d, o_d, heads, d, k_heads, scale);

    var got_o: [heads * d]f32 = undefined;
    var got_state: [heads * d * d]f32 = undefined;
    var got_st: [channels * 3]f32 = undefined;
    try be.tensorDownload(o_d, std.mem.sliceAsBytes(&got_o));
    try be.tensorDownload(state_d, std.mem.sliceAsBytes(&got_state));
    try be.tensorDownload(st_d, std.mem.sliceAsBytes(&got_st));

    for (ref_o, got_o) |e, a| try std.testing.expectApproxEqAbs(e, a, 5e-3);
    for (ref_state, got_state) |e, a| try std.testing.expectApproxEqAbs(e, a, 5e-3);
    for (ref_conv_st, got_st) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-6);
}

// Gated on a CUDA device: hd=256 flash-decoding split + deinterleave +
// sigmoid gate + partial rope against CPU references.
test "qwen35 attention ops match CPU reference" {
    const rope = @import("../../ops.zig").rope;
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const heads = 3;
    const kv_heads = 1;
    const hd = 256;
    const kv_len = 7;
    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();

    var q: [heads * hd]f32 = undefined;
    var k: [kv_len * kv_heads * hd]f32 = undefined;
    var v: [kv_len * kv_heads * hd]f32 = undefined;
    for (&q) |*x| x.* = rand.floatNorm(f32);
    for (&k) |*x| x.* = rand.floatNorm(f32);
    for (&v) |*x| x.* = rand.floatNorm(f32);
    const scale = 1.0 / @sqrt(@as(f32, hd));

    // CPU reference attention (seq_q = 1).
    var ref: [heads * hd]f32 = undefined;
    for (0..heads) |h| {
        var scores: [kv_len]f32 = undefined;
        var mx: f32 = -std.math.inf(f32);
        for (0..kv_len) |j| {
            var s: f32 = 0;
            for (0..hd) |c| s += q[h * hd + c] * k[(j * kv_heads) * hd + c];
            scores[j] = s * scale;
            mx = @max(mx, scores[j]);
        }
        var den: f32 = 0;
        for (&scores) |*s| {
            s.* = @exp(s.* - mx);
            den += s.*;
        }
        for (0..hd) |c| {
            var acc: f32 = 0;
            for (0..kv_len) |j| acc += scores[j] * v[(j * kv_heads) * hd + c];
            ref[h * hd + c] = acc / den;
        }
    }

    const q_d = try be.tensorCreate(q.len * 4);
    const k_d = try be.tensorCreate(k.len * 4);
    const v_d = try be.tensorCreate(v.len * 4);
    const out_d = try be.tensorCreate(q.len * 4);
    const scratch_d = try be.tensorCreate(heads * 8 * (hd + 4) * 4);
    defer {
        inline for (.{ q_d, k_d, v_d, out_d, scratch_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(q_d, std.mem.sliceAsBytes(&q));
    try be.tensorUpload(k_d, std.mem.sliceAsBytes(&k));
    try be.tensorUpload(v_d, std.mem.sliceAsBytes(&v));
    try be.opAttnDecode(q_d, k_d, v_d, out_d, scratch_d, kv_len, 1, heads, kv_heads, hd, 8, scale);
    var got: [heads * hd]f32 = undefined;
    try be.tensorDownload(out_d, std.mem.sliceAsBytes(&got));
    for (ref, got) |e, a| try std.testing.expectApproxEqAbs(e, a, 2e-3);

    // deinterleave2 + mul_sigmoid.
    var qg: [2 * heads * hd]f32 = undefined;
    for (&qg) |*x| x.* = rand.floatNorm(f32);
    const qg_d = try be.tensorCreate(qg.len * 4);
    const q2_d = try be.tensorCreate(heads * hd * 4);
    const g2_d = try be.tensorCreate(heads * hd * 4);
    defer {
        inline for (.{ qg_d, q2_d, g2_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(qg_d, std.mem.sliceAsBytes(&qg));
    try be.opDeinterleave2(qg_d, q2_d, g2_d, heads * hd, hd);
    try be.opMulSigmoid(q2_d, g2_d, heads * hd);
    var got_q: [heads * hd]f32 = undefined;
    try be.tensorDownload(q2_d, std.mem.sliceAsBytes(&got_q));
    for (0..heads) |h| {
        for (0..hd) |c| {
            const qv = qg[h * 2 * hd + c];
            const gv = qg[h * 2 * hd + hd + c];
            const e = qv * (1.0 / (1.0 + @exp(-gv)));
            try std.testing.expectApproxEqAbs(e, got_q[h * hd + c], 2e-4 + 1e-3 * @abs(e));
        }
    }

    // Partial rope vs the CPU op (rot 64 of 256 at position 5).
    const rot = 64;
    var freqs = try rope.rotateHalfFreqs(gpa, 8, rot, 1e7);
    defer freqs.deinit(gpa);
    var fp: [8 * rot]f32 = undefined; // [cos | sin] for 8 positions
    @memcpy(fp[0 .. 8 * rot / 2], freqs.cos);
    @memcpy(fp[8 * rot / 2 ..], freqs.sin);
    var xr: [heads * hd]f32 = undefined;
    for (&xr) |*x| x.* = rand.floatNorm(f32);
    var xr_ref = xr;
    rope.applyRotateHalfPartialAt(&xr_ref, freqs, 5, 1, heads, hd, rot);
    const fr_d = try be.tensorCreate(fp.len * 4);
    const xr_d = try be.tensorCreate(xr.len * 4);
    defer {
        inline for (.{ fr_d, xr_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(fr_d, std.mem.sliceAsBytes(&fp));
    try be.tensorUpload(xr_d, std.mem.sliceAsBytes(&xr));
    try be.opRopeHalfPart(xr_d, fr_d, 1, heads, rot / 2, 8 * rot / 2, 5, hd);
    var got_x: [heads * hd]f32 = undefined;
    try be.tensorDownload(xr_d, std.mem.sliceAsBytes(&got_x));
    for (xr_ref, got_x) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
}

fn offsetBufSizedTest(b: DeviceBuffer, off_bytes: usize, size: usize) DeviceBuffer {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

// Gated on a CUDA device: batched flash-decode (seq_q >> 1) at hd=256
// against a CPU causal reference — the qwen35 batched-prefill regime.
test "qkNorm PAR batch rows matches CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 56;
    const hd = 5120;
    const eps: f32 = 1e-6;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const x = try gpa.alloc(f32, rows * hd);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);
    const w = try gpa.alloc(f32, hd);
    defer gpa.free(w);
    for (w) |*v| v.* = 1.0 + 0.1 * rand.floatNorm(f32);

    // Mirror the model: 128-row buffers, partial upload, smallBuffer weight.
    var xd = try be.tensorCreate(128 * hd * 4);
    defer be.tensorDestroy(&xd);
    var od = try be.tensorCreate(128 * hd * 4);
    defer be.tensorDestroy(&od);
    try be.tensorUpload(xd, std.mem.sliceAsBytes(x));
    const wd: DeviceBuffer = .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };

    try be.beginBatch();
    try be.qkNorm(xd, od, wd, rows, hd, eps);
    const got = try gpa.alloc(f32, rows * hd);
    defer gpa.free(got);
    try be.tensorDownload(od, std.mem.sliceAsBytes(got));
    try be.endBatch();

    for (0..rows) |r| {
        var ss: f64 = 0;
        for (x[r * hd ..][0..hd]) |v| ss += @as(f64, v) * v;
        const inv: f32 = @floatCast(1.0 / @sqrt(ss / hd + eps));
        for (0..hd) |c| {
            const want = x[r * hd + c] * inv * w[c];
            if (@abs(want - got[r * hd + c]) > 2e-3) {
                std.debug.print("row {d} col {d}: want {d} got {d}\n", .{ r, c, want, got[r * hd + c] });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "attn decode seq_q batch matches CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const heads = 4;
    const kv_heads = 2;
    const hd = 256;
    const seq_q = 96;
    const kv_len0 = 0; // empty cache (first prefill chunk); query t sees t+1 keys
    const kv_total = kv_len0 + seq_q;
    const nsp = 8;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();

    const q = try gpa.alloc(f32, seq_q * heads * hd);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, kv_total * kv_heads * hd);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, kv_total * kv_heads * hd);
    defer gpa.free(v);
    for (q) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (k) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (v) |*x| x.* = rand.floatNorm(f32);
    const scale = 1.0 / @sqrt(@as(f32, hd));

    // CPU causal reference.
    const ref = try gpa.alloc(f32, seq_q * heads * hd);
    defer gpa.free(ref);
    const scores = try gpa.alloc(f32, kv_total);
    defer gpa.free(scores);
    for (0..seq_q) |t| {
        const klen = kv_len0 + t + 1;
        for (0..heads) |h| {
            const kvh = h / (heads / kv_heads);
            var mx: f32 = -std.math.inf(f32);
            for (0..klen) |j| {
                var sacc: f32 = 0;
                for (0..hd) |c| sacc += q[(t * heads + h) * hd + c] * k[(j * kv_heads + kvh) * hd + c];
                scores[j] = sacc * scale;
                mx = @max(mx, scores[j]);
            }
            var den: f32 = 0;
            for (scores[0..klen]) |*sc| {
                sc.* = @exp(sc.* - mx);
                den += sc.*;
            }
            for (0..hd) |c| {
                var acc: f32 = 0;
                for (0..klen) |j| acc += scores[j] * v[(j * kv_heads + kvh) * hd + c];
                ref[(t * heads + h) * hd + c] = acc / den;
            }
        }
    }

    const q_d = try be.tensorCreate(q.len * 4);
    const k_d = try be.tensorCreate(k.len * 4);
    const v_d = try be.tensorCreate(v.len * 4);
    const o_d = try be.tensorCreate(q.len * 4);
    const s_d = try be.tensorCreate(seq_q * heads * nsp * (hd + 4) * 4);
    defer {
        inline for (.{ q_d, k_d, v_d, o_d, s_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try be.tensorUpload(k_d, std.mem.sliceAsBytes(k));
    try be.tensorUpload(v_d, std.mem.sliceAsBytes(v));
    try be.opAttnDecode(q_d, k_d, v_d, o_d, s_d, kv_len0 + 1, seq_q, heads, kv_heads, hd, nsp, scale);
    const got = try gpa.alloc(f32, q.len);
    defer gpa.free(got);
    try be.tensorDownload(o_d, std.mem.sliceAsBytes(got));
    var worst: f32 = 0;
    var worst_i: usize = 0;
    for (ref, got, 0..) |e, a, i| {
        const diff = @abs(e - a);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
    }
    if (worst > 2e-3) {
        std.debug.print("worst {d} at elem {d} (t={d} h={d} c={d}): want {d} got {d}\n", .{ worst, worst_i, worst_i / (heads * hd), (worst_i / hd) % heads, worst_i % hd, ref[worst_i], got[worst_i] });
        return error.TestExpectedApproxEqAbs;
    }
}

// Gated on a CUDA device: rope_imrope_pos with DIFFERING (t, h, w)
// positions per row against a CPU port of ggml's imrope rule.
test "rope_imrope_pos matches CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 3;
    const heads = 2;
    const hd = 256;
    const half = 32; // rope_dim 64
    const sections = [3]u32{ 11, 11, 10 };
    const cap = 64;
    var prng = std.Random.DefaultPrng.init(4);
    const rand = prng.random();

    // freqs table [cap][half] cos | sin, theta base 1e7.
    var fp: [2 * cap * half]f32 = undefined;
    for (0..cap) |pp| {
        for (0..half) |i| {
            const theta = @as(f64, @floatFromInt(pp)) * std.math.pow(f64, 1e7, -2.0 * @as(f64, @floatFromInt(i)) / 64.0);
            fp[pp * half + i] = @floatCast(@cos(theta));
            fp[cap * half + pp * half + i] = @floatCast(@sin(theta));
        }
    }
    const pos3s = [rows * 3]u32{ 5, 5, 5, 5, 6, 9, 5, 7, 5 };
    var x: [rows * heads * hd]f32 = undefined;
    for (&x) |*v| v.* = rand.floatNorm(f32);

    // CPU reference.
    var ref = x;
    for (0..rows) |r| {
        for (0..heads) |h| {
            const base = (r * heads + h) * hd;
            for (0..half) |pair| {
                var ch: usize = 0;
                if (pair % 3 == 1 and pair < 3 * sections[1]) {
                    ch = 1;
                } else if (pair % 3 == 2 and pair < 3 * sections[2]) {
                    ch = 2;
                }
                const pp = pos3s[r * 3 + ch];
                const c = fp[pp * half + pair];
                const sn = fp[cap * half + pp * half + pair];
                const lo = ref[base + pair];
                const hi = ref[base + half + pair];
                ref[base + pair] = lo * c - hi * sn;
                ref[base + half + pair] = hi * c + lo * sn;
            }
        }
    }

    const x_d = try be.tensorCreate(x.len * 4);
    const p_d = try be.tensorCreate(pos3s.len * 4);
    const f_d = try be.tensorCreate(fp.len * 4);
    defer {
        inline for (.{ x_d, p_d, f_d }) |b| {
            var bb = b;
            be.tensorDestroy(&bb);
        }
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(&x));
    try be.tensorUpload(p_d, std.mem.sliceAsBytes(&pos3s));
    try be.tensorUpload(f_d, std.mem.sliceAsBytes(&fp));
    try be.opRopeImropePos(x_d, p_d, f_d, rows, heads, half, cap * half, sections, hd);
    var got: [rows * heads * hd]f32 = undefined;
    try be.tensorDownload(x_d, std.mem.sliceAsBytes(&got));
    for (ref, got, 0..) |e, a, i| {
        std.testing.expectApproxEqAbs(e, a, 1e-5) catch |err| {
            std.debug.print("elem {d} (row {d} head {d} dim {d}): want {d} got {d}\n", .{ i, i / (heads * hd), (i / hd) % heads, i % hd, e, a });
            return err;
        };
    }
}

// Gated on a CUDA device: gemv_fp8n against a CPU LUT reference, including
// the n < 4 predicated-store tail.
test "gemv_fp8n matches CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 48;
    const cols = 256; // multiple of 8
    const n = 3;
    var prng = std.Random.DefaultPrng.init(999);
    const rand = prng.random();

    const w = try gpa.alloc(u8, rows * cols);
    defer gpa.free(w);
    rand.bytes(w);
    for (w) |*wb| {
        if (wb.* & 0x7F == 0x7F) wb.* = 0; // e4m3 NaN encodings
    }
    const x = try gpa.alloc(f32, 4 * cols);
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;

    const x_d = try be.tensorCreate(4 * cols * 4);
    const y_d = try be.tensorCreate(4 * rows * 4);
    defer {
        var xd = x_d;
        var yd = y_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
    const y = try gpa.alloc(f32, 4 * rows);
    defer gpa.free(y);
    @memset(y, -777.0);
    try be.tensorUpload(y_d, std.mem.sliceAsBytes(y));

    const scale: f32 = 0.25;
    try be.opGemvFp8N(y_d, x_d, w, scale, rows, cols, n);
    try be.tensorDownload(y_d, std.mem.sliceAsBytes(y));

    for (0..n) |i| {
        for (0..rows) |r| {
            var acc: f32 = 0;
            for (0..cols) |c| acc += dtypes.f8_e4m3_to_f32_table[w[r * cols + c]] * x[i * cols + c];
            try std.testing.expectApproxEqAbs(acc * scale, y[i * rows + r], 1e-2);
        }
    }
    for (0..rows) |r| try std.testing.expectEqual(@as(f32, -777.0), y[3 * rows + r]);
}

// Gated on a CUDA device: gemv_bf16n against a CPU reference, including the
// n < 4 predicated-store tail (untouched output rows must stay untouched).
test "gemv_bf16n matches CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const rows = 64;
    const cols = 128;
    const n = 3;
    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    // bf16 weights with exactly-representable values via f32 truncation.
    const w = try gpa.alloc(u16, rows * cols);
    defer gpa.free(w);
    const w_f32 = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w_f32);
    for (w, w_f32) |*wi, *wf| {
        const v = rand.float(f32) * 2.0 - 1.0;
        wi.* = @truncate(@as(u32, @bitCast(v)) >> 16);
        wf.* = @bitCast(@as(u32, wi.*) << 16);
    }
    const x = try gpa.alloc(f32, 4 * cols); // 4 rows of backing store, n=3 live
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;

    const x_d = try be.tensorCreate(4 * cols * 4);
    const y_d = try be.tensorCreate(4 * rows * 4);
    defer {
        var xd = x_d;
        var yd = y_d;
        be.tensorDestroy(&xd);
        be.tensorDestroy(&yd);
    }
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
    // Poison y so the predicated-off 4th row is provably untouched.
    const y = try gpa.alloc(f32, 4 * rows);
    defer gpa.free(y);
    @memset(y, -777.0);
    try be.tensorUpload(y_d, std.mem.sliceAsBytes(y));

    const scale: f32 = 0.5;
    try be.opGemvBf16N(y_d, x_d, std.mem.sliceAsBytes(w), scale, rows, cols, n);
    try be.tensorDownload(y_d, std.mem.sliceAsBytes(y));

    for (0..n) |i| {
        for (0..rows) |r| {
            var acc: f32 = 0;
            for (0..cols) |c| acc += w_f32[r * cols + c] * x[i * cols + c];
            try std.testing.expectApproxEqAbs(acc * scale, y[i * rows + r], 1e-3);
        }
    }
    for (0..rows) |r| try std.testing.expectEqual(@as(f32, -777.0), y[3 * rows + r]);
}

// Gated on a CUDA device: rope_half_pos against the CPU per-row-position
// reference (out-of-order, repeated positions).
test "rope_half_pos matches CPU reference" {
    const gpa = std.testing.allocator;
    const ops = @import("../../ops.zig");
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const n_heads = 2;
    const half = 8;
    const head_dim = 2 * half;
    const positions = [_]usize{ 5, 0, 9, 5 };
    const rows = positions.len;

    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();
    const x = try gpa.alloc(f32, rows * n_heads * head_dim);
    defer gpa.free(x);
    for (x) |*xi| xi.* = rand.float(f32) * 2.0 - 1.0;
    const expected = try gpa.alloc(f32, x.len);
    defer gpa.free(expected);
    @memcpy(expected, x);

    var freqs = try ops.rope.rotateHalfFreqs(gpa, 10, head_dim, 5e6);
    defer freqs.deinit(gpa);
    ops.rope.applyRotateHalfPos(expected, freqs, &positions, n_heads, head_dim);

    const fp = try gpa.alloc(f32, 2 * 10 * half);
    defer gpa.free(fp);
    @memcpy(fp[0 .. 10 * half], freqs.cos);
    @memcpy(fp[10 * half ..], freqs.sin);
    var pos32: [rows]u32 = undefined;
    for (positions, 0..) |p, i| pos32[i] = @intCast(p);

    var x_d = try be.tensorCreate(x.len * 4);
    defer be.tensorDestroy(&x_d);
    var pos_d = try be.tensorCreate(rows * 4);
    defer be.tensorDestroy(&pos_d);
    var f_d = try be.tensorCreate(fp.len * 4);
    defer be.tensorDestroy(&f_d);
    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
    try be.tensorUpload(pos_d, std.mem.sliceAsBytes(&pos32));
    try be.tensorUpload(f_d, std.mem.sliceAsBytes(fp));

    try be.opRopeHalfPos(x_d, pos_d, f_d, rows, n_heads, half, 10 * half);
    try be.tensorDownload(x_d, std.mem.sliceAsBytes(x));
    for (expected, x) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);
}

// Gated on a CUDA device: attn_split_tree + attn_merge against the CPU
// tree-attention reference — branching tree, GQA, prefix + ancestor rows in
// one K/V buffer with the batch rows at tree_base.
test "tree flash-decode attention matches CPU reference" {
    const gpa = std.testing.allocator;
    const ops = @import("../../ops.zig");
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const n_heads = 4;
    const kv_heads = 2;
    const hd = 128;
    const kv_dim = kv_heads * hd;
    const q_dim = n_heads * hd;
    const prefix_len = 10;
    const tree_base = 16;
    const parents = [_]u32{ 0, 0, 1, 1, 0 }; // root, two children, two grandchildren under node 1
    const n = parents.len;
    const nsplit = 4;
    const scale: f32 = 1.0 / 8.0;

    var prng = std.Random.DefaultPrng.init(4242);
    const rand = prng.random();
    const q = try gpa.alloc(f32, n * q_dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, (tree_base + n) * kv_dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, (tree_base + n) * kv_dim);
    defer gpa.free(v);
    for (q) |*e| e.* = rand.float(f32) * 2.0 - 1.0;
    for (k) |*e| e.* = rand.float(f32) * 2.0 - 1.0;
    for (v) |*e| e.* = rand.float(f32) * 2.0 - 1.0;

    const expected = try gpa.alloc(f32, n * q_dim);
    defer gpa.free(expected);
    try ops.attention.attentionTree(
        gpa,
        expected,
        q,
        k[0 .. prefix_len * kv_dim],
        v[0 .. prefix_len * kv_dim],
        k[tree_base * kv_dim ..],
        v[tree_base * kv_dim ..],
        &parents,
        .{ .n_heads = n_heads, .n_kv_heads = kv_heads, .head_dim = hd, .scale = scale },
    );

    // Meta table: [kv_len, anc_0..] per query, stride n+1, at the scratch tail.
    const meta_off = n * n_heads * nsplit * (hd + 4);
    var meta = [_]u32{0} ** (n * (n + 1));
    var depth: [n]usize = undefined;
    depth[0] = 0;
    for (parents[1..], 1..) |p, i| depth[i] = depth[p] + 1;
    for (0..n) |i| {
        meta[i * (n + 1)] = @intCast(prefix_len + depth[i] + 1);
        var j: u32 = @intCast(i);
        var d = depth[i];
        while (true) {
            meta[i * (n + 1) + 1 + d] = j;
            if (j == 0) break;
            j = parents[j];
            d -= 1;
        }
    }

    var q_d = try be.tensorCreate(q.len * 4);
    defer be.tensorDestroy(&q_d);
    var k_d = try be.tensorCreate(k.len * 4);
    defer be.tensorDestroy(&k_d);
    var v_d = try be.tensorCreate(v.len * 4);
    defer be.tensorDestroy(&v_d);
    var out_d = try be.tensorCreate(n * q_dim * 4);
    defer be.tensorDestroy(&out_d);
    var scratch_d = try be.tensorCreate((meta_off + meta.len) * 4);
    defer be.tensorDestroy(&scratch_d);
    try be.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try be.tensorUpload(k_d, std.mem.sliceAsBytes(k));
    try be.tensorUpload(v_d, std.mem.sliceAsBytes(v));
    const meta_dst: DeviceBuffer = .{
        .buf = @enumFromInt(@intFromEnum(scratch_d.buf) + meta_off * 4),
        .mem = scratch_d.mem,
        .size = meta.len * 4,
    };
    try be.tensorUpload(meta_dst, std.mem.sliceAsBytes(&meta));

    try be.opAttnDecodeTree(q_d, k_d, v_d, out_d, scratch_d, prefix_len, tree_base, n, n_heads, kv_heads, hd, nsplit, scale);
    const out = try gpa.alloc(f32, n * q_dim);
    defer gpa.free(out);
    try be.tensorDownload(out_d, std.mem.sliceAsBytes(out));
    for (expected, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 2e-4);
}

