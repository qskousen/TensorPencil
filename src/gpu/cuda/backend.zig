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
    upload_ev: cu.CUevent = null,
    /// prefetch generation (0 = not prefetched / synchronously uploaded). A hit
    /// with pf_gen > pf_completed is still in flight on the prefetch thread; the
    /// consumer waits until pf_completed >= pf_gen before using it.
    pf_gen: u64 = 0,
};

/// A queued weight upload for the prefetch thread: memcpy `bytes` (mmap) into a
/// pinned slot then async-DMA into `db`, recording `ev` (compute waits on it).
const PrefetchReq = struct { bytes: []const u8, db: DeviceBuffer, ev: cu.CUevent, gen: u64 };
const pf_ring_sz = 64;

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
    /// async weight uploads: set once the checkpoint mmap is page-locked, so
    /// re-uploads run on the transfer stream (overlapping compute) with a
    /// per-weight completion event the compute stream waits on.
    async_uploads: bool = false,
    /// evicted-but-not-yet-freed weight buffers (deferred free; async path only).
    free_pending: std.ArrayListUnmanaged(PendingFree) = .empty,

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
        self.drainPending();
        self.free_pending.deinit(self.gpa);
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
            self.ctx.uploadStaged(cb, req.bytes, req.ev) catch {};
            self.pf_tail.store(tail + 1, .release);
            self.pf_completed.store(req.gen, .release); // FIFO: gens complete in order
        }
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
        const db = self.tensorCreate(bytes.len) catch return; // best-effort; cachedWeight sync-falls-back on miss
        const ev = self.ctx.eventCreate() catch {
            var d = db;
            self.tensorDestroy(&d);
            return;
        };
        const head = self.pf_head.load(.monotonic);
        if (head - self.pf_tail.load(.acquire) >= pf_ring_sz) { // ring full — drop (sync fallback later)
            self.ctx.eventDestroy(ev);
            var d = db;
            self.tensorDestroy(&d);
            return;
        }
        self.pf_gen += 1;
        const gen = self.pf_gen;
        self.pf_ring[head % pf_ring_sz] = .{ .bytes = bytes, .db = db, .ev = ev, .gen = gen };
        self.pf_head.store(head + 1, .release); // publish the slot to the thread
        self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .upload_ev = ev, .pf_gen = gen }) catch {};
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
                var db = pf.db;
                self.tensorDestroy(&db);
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
            if (e.value_ptr.last_use == mru_use) continue; // protect the MRU
            if (e.value_ptr.pf_gen > completed) continue; // protect in-flight prefetch
            if (e.value_ptr.last_use < lru_use) {
                lru_use = e.value_ptr.last_use;
                lru_key = e.key_ptr.*;
            }
        }
        if (lru_use == std.math.maxInt(u64)) return false; // only the MRU remains
        const e = self.weights.fetchRemove(lru_key).?;
        if (e.value.upload_ev) |ev| self.ctx.eventDestroy(ev);
        const db = e.value.db;
        if (self.async_uploads) {
            if (self.ctx.eventCreate()) |rev| {
                _ = self.ctx.api.cuEventRecord(rev, self.ctx.stream); // after this weight's last GEMM
                if (self.free_pending.append(self.gpa, .{ .db = db, .ev = rev })) |_| {
                    return true; // deferred free; reclaimed lazily
                } else |_| self.ctx.eventDestroy(rev);
            } else |_| {}
        }
        // sync path (async off, or event/append failed): a launch may still
        // reference the buffer, so synchronize before freeing.
        _ = self.ctx.api.cuStreamSynchronize(self.ctx.stream);
        var d = db;
        self.tensorDestroy(&d);
        return true;
    }

    /// Make room for a `need`-byte weight upload: reclaim ready deferred-frees, then
    /// evict LRU weights while the budget lacks headroom. On the DiT's fixed block
    /// walk this becomes sequential weight streaming; createBuffer's OOM retry is
    /// the reactive backstop if this can't free enough.
    fn reserveForWeights(self: *Backend, need: u64) void {
        self.reclaimPending();
        while (self.budgetHeadroom() < need) {
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
        self.drainPending();
        var it = self.weights.valueIterator();
        while (it.next()) |e| {
            if (e.upload_ev) |ev| self.ctx.eventDestroy(ev);
            var db = e.db;
            self.tensorDestroy(&db);
        }
        self.weights.clearRetainingCapacity();
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
        const db = try self.tensorCreate(bytes.len);
        try self.tensorUpload(db, bytes);
        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter });
        return db;
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

    /// bf16 GEMV (tied LM head): y[rows] f32 = scale * (W bf16 [rows][cols] @ x).
    pub fn opGemvBf16(self: *Backend, y: DeviceBuffer, x: DeviceBuffer, w_bytes: []const u8, scale: f32, rows: usize, cols: usize) Error!void {
        self.ptic();
        defer self.ptoc(.matmul);
        std.debug.assert(cols % 2 == 0);
        const w_db = try self.cachedWeight(w_bytes);
        const f = try self.eltFn(elt.gemv_bf16_ptx, "gemv_bf16");
        try self.rowLaunch(f, w_db, x, y, null, .{ @intCast(rows), @intCast(cols), 0, 0, 0, 0 }, .{ scale, 0 }, rows);
    }

    /// Flash-decoding attention for a single query against seq_kv cached keys:
    /// a warp per (head, KV chunk) in the split pass, then a merge pass.
    /// scratch holds heads*nsplit*(hd+4) f32. hd must be 128.
    pub fn opAttnDecode(self: *Backend, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, scratch: DeviceBuffer, seq_kv: usize, n_heads: usize, kv_heads: usize, hd: usize, nsplit: usize, scale: f32) Error!void {
        self.ptic();
        defer self.ptoc(.attn);
        std.debug.assert(hd == 128);
        const f_split = try self.eltFn(elt.attn_split_ptx, "attn_split");
        try self.eltLaunch(f_split, q, k, v, scratch, .{ @intCast(seq_kv), @intCast(n_heads), @intCast(kv_heads), @intCast(hd), @intCast(nsplit), 0 }, .{ scale, 0 }, n_heads * nsplit * 32);
        const f_merge = try self.eltFn(elt.attn_merge_ptx, "attn_merge");
        try self.eltLaunch(f_merge, scratch, out, null, null, .{ @intCast(n_heads), @intCast(hd), @intCast(nsplit), 0, 0, 0 }, .{ 0, 0 }, n_heads * hd);
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
