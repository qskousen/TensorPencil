//! CUDA Driver-API context: device/context/stream setup, PTX JIT, device
//! buffers, kernel launch, and event-based timing. Thin, explicit, and
//! diagnostic-friendly (the Phase-1 experiment lives on top of this).

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const cu = @import("cu.zig");
const mem_tag = @import("../mem_tag.zig");
pub const MemTag = mem_tag.MemTag;

pub const Error = error{ CudaError, OutOfMemory, DeviceOutOfMemory };

/// Monotonic wall-clock nanoseconds (io-free; std.time.Timer is gone in 0.16).
pub fn monoNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Pinned staging-ring depth (slots in flight for async weight uploads).
const stage_slots = 4;

/// A device allocation. `ptr == 0` means "empty" (never allocated / freed).
pub const Buffer = struct {
    ptr: cu.CUdeviceptr = 0,
    bytes: usize = 0,
    /// Component this allocation is attributed to (for the VRAM meter). Set from
    /// the context's current tag at alloc; read back at free. `.other` unless
    /// tagging is active (diffusion backend only).
    tag: MemTag = .other,

    pub fn isNull(self: Buffer) bool {
        return self.ptr == 0;
    }
};

/// A JIT-compiled PTX module.
pub const Module = struct {
    mod: cu.CUmodule,

    pub fn getFunction(self: Module, ctx: *Context, name: [:0]const u8) Error!cu.CUfunction {
        var f: cu.CUfunction = null;
        try ctx.check(ctx.api.cuModuleGetFunction(&f, self.mod, name.ptr), "cuModuleGetFunction");
        return f;
    }

    pub fn unload(self: Module, ctx: *Context) void {
        _ = ctx.api.cuModuleUnload(self.mod);
    }
};

pub const Context = struct {
    api: cu.Api,
    dev: cu.CUdevice = 0,
    ctx: cu.CUcontext = null,
    stream: cu.CUstream = null,
    xfer_stream: cu.CUstream = null, // async weight uploads (overlaps compute)
    // Pinned staging ring: the checkpoint mmap can't be page-locked directly, so
    // weight uploads memcpy mmap→pinned slot→async DMA. Round-robin slots each
    // with a reuse event (slot free once its DMA signals).
    /// Sources for weight bytes that bypass the mmap — see `WeightReader`. One
    /// slot per open checkpoint (a diffusion session has DiT + encoder + VAE);
    /// each reader answers only for its own file, so they are simply tried in turn.
    weight_readers: [4]?WeightReader = @splat(null),
    /// A single pinned chunk for `uploadWeight`, kept SEPARATE from the staging
    /// ring on purpose: the ring belongs to the prefetch thread, and the
    /// synchronous path can run on the main thread at the same time.
    wr_chunk: ?*anyopaque = null,
    staging: [stage_slots]?*anyopaque = @splat(null),
    staging_ev: [stage_slots]cu.CUevent = @splat(null),
    staging_size: usize = 0,
    staging_next: usize = 0,
    n_staging: usize = 0,
    /// Parallel filler for the staging slots. Inline rather than heap-allocated
    /// because the workers hold `&self.fill`: `initStaging` runs on the
    /// already-heap-resident Context, where `init` returns by value and is copied.
    fill: FillPool = .{ .ctx = undefined },
    fill_ready: bool = false,

    // Device attributes (queried once at init).
    name_buf: [256]u8 = undefined,
    name_len: usize = 0,
    cc_major: c_int = 0,
    cc_minor: c_int = 0,
    sm_count: c_int = 0,
    // Warmup attribution (TP_WARMUP_PROFILE): wall time spent in PTX module
    // JIT/load, summed across the run (all but step 1 hit the module cache).
    jit_ns: u64 = 0,
    jit_count: u32 = 0,
    // Total host->device bytes enqueued (weight residency attribution).
    htod_bytes: u64 = 0,
    /// Prefetch-thread accounting, split so a slow upload can be blamed on the
    /// right half: `stage_read_ns` is the host fill of the pinned slot (pread or
    /// memcpy from the mapping), `stage_wait_ns` the block waiting for a slot's
    /// previous DMA to drain — i.e. the PCIe side pushing back.
    stage_read_ns: u64 = 0,
    stage_wait_ns: u64 = 0,
    shared_optin_max: c_int = 0, // bytes of opt-in dynamic shared per block
    shared_per_sm: c_int = 0,
    clock_khz: c_int = 0,

    device_used: usize = 0,

    /// Per-component device-byte attribution (MEASURED, for the GUI VRAM meter).
    /// Only maintained when `track_tags` is set (diffusion backend) so the LLM
    /// allocator hot path is untouched. `mem_tag` is the tag applied to new
    /// allocations; the pipeline sets it around each phase (encode/denoise/decode).
    track_tags: bool = false,
    mem_tag: MemTag = .other,
    mem_tag_used: [MemTag.count]usize = @splat(0),

    /// Physical-chunk granularity for VMM growable buffers (bytes, typically
    /// 2 MB); 0 when the driver lacks the VMM entry points.
    vmm_granularity: usize = 0,

    /// Last JIT log (error or info) from loadModule; valid until the next call.
    jit_log: [16384]u8 = undefined,
    jit_log_len: usize = 0,

    pub fn init(gpa: std.mem.Allocator) Error!Context {
        // Integration tests are gated behind `-Dintegration`: fail init in test
        // builds when it's off so the CUDA tests self-skip (they `catch SkipZigTest`).
        if (builtin.is_test and !build_options.integration) return error.CudaError;
        _ = gpa;
        var self: Context = .{ .api = cu.Api.load() catch return error.CudaError };
        errdefer self.api.deinit();

        try self.check(self.api.cuInit(0), "cuInit");
        try self.check(self.api.cuDeviceGet(&self.dev, 0), "cuDeviceGet");
        try self.check(self.api.cuCtxCreate(&self.ctx, cu.CU_CTX_SCHED_AUTO, self.dev), "cuCtxCreate");
        errdefer _ = self.api.cuCtxDestroy(self.ctx);
        try self.check(self.api.cuStreamCreate(&self.stream, 0), "cuStreamCreate");
        // Dedicated transfer stream for async weight uploads that overlap compute.
        _ = self.api.cuStreamCreate(&self.xfer_stream, 0);

        // Prefer shared memory for the L1/shared carveout (GEMM is shared-bound).
        _ = self.api.cuCtxSetCacheConfig(cu.CU_FUNC_CACHE_PREFER_SHARED);

        // Device name.
        var nb: [256]u8 = undefined;
        if (self.api.cuDeviceGetName(&nb, nb.len, self.dev) == cu.CUDA_SUCCESS) {
            const n = std.mem.indexOfScalar(u8, &nb, 0) orelse nb.len;
            @memcpy(self.name_buf[0..n], nb[0..n]);
            self.name_len = n;
        }
        _ = self.api.cuDeviceGetAttribute(&self.cc_major, cu.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev);
        _ = self.api.cuDeviceGetAttribute(&self.cc_minor, cu.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev);
        _ = self.api.cuDeviceGetAttribute(&self.sm_count, cu.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev);
        _ = self.api.cuDeviceGetAttribute(&self.shared_optin_max, cu.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self.dev);
        _ = self.api.cuDeviceGetAttribute(&self.shared_per_sm, cu.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, self.dev);
        _ = self.api.cuDeviceGetAttribute(&self.clock_khz, cu.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, self.dev);

        self.vmm_granularity = self.vmmQueryGranularity();

        return self;
    }

    pub fn deinit(self: *Context) void {
        if (self.fill_ready) self.fill.deinit();
        if (self.wr_chunk) |p| _ = self.api.cuMemFreeHost(p);
        for (0..self.n_staging) |i| {
            if (self.staging_ev[i] != null) _ = self.api.cuEventDestroy(self.staging_ev[i]);
            if (self.staging[i]) |p| _ = self.api.cuMemFreeHost(p);
        }
        if (self.xfer_stream != null) _ = self.api.cuStreamDestroy(self.xfer_stream);
        if (self.stream != null) _ = self.api.cuStreamDestroy(self.stream);
        if (self.ctx != null) _ = self.api.cuCtxDestroy(self.ctx);
        self.api.deinit();
    }

    /// Bytes per `uploadWeight` chunk. Bounded so an arbitrarily large weight
    /// (a 300 MB LM head) costs a fixed pinned allocation, not its own size.
    const wr_chunk_size: usize = 32 << 20;

    /// Synchronous upload that sources the bytes through a registered
    /// `WeightReader` when one owns them — read into a pinned chunk, DMA, repeat —
    /// instead of letting `cuMemcpyHtoD` fault them out of the checkpoint mapping.
    ///
    /// This is the path the LLM takes (it has no prefetch thread, so it never
    /// reaches `uploadStaged`). Falls back to a plain `upload` when no reader owns
    /// the bytes or the pinned chunk cannot be allocated, so it is never fatal.
    pub fn uploadWeight(self: *Context, buf: Buffer, data: []const u8) Error!void {
        std.debug.assert(data.len <= buf.bytes);
        var any = false;
        for (self.weight_readers) |maybe| if (maybe != null) {
            any = true;
        };
        if (!any) return self.upload(buf, data);
        if (self.wr_chunk == null) {
            var p: ?*anyopaque = null;
            if (self.api.cuMemAllocHost(&p, wr_chunk_size) != cu.CUDA_SUCCESS) return self.upload(buf, data);
            self.wr_chunk = p;
        }
        const chunk: [*]u8 = @ptrCast(self.wr_chunk.?);
        var off: usize = 0;
        while (off < data.len) {
            const n = @min(wr_chunk_size, data.len - off);
            // A reader that declines mid-weight (not its file) means the whole
            // weight is not ours — finish it with the plain copy rather than
            // interleaving two sources.
            if (!self.readWeight(chunk[0..n], data[off..][0..n])) {
                if (off == 0) return self.upload(buf, data);
                @memcpy(chunk[0..n], data[off..][0..n]);
            }
            try self.check(self.api.cuMemcpyHtoD(buf.ptr + off, chunk, n), "cuMemcpyHtoD(weight)");
            self.htod_bytes += n;
            off += n;
        }
    }

    /// Register a source of weight bytes. Ignored once the slots are full — the
    /// fallback copy is always correct, so running out degrades speed, not results.
    pub fn addWeightReader(self: *Context, wr: WeightReader) void {
        for (&self.weight_readers) |*slot| if (slot.* == null) {
            slot.* = wr;
            return;
        };
    }

    /// Fill `dst` from whichever registered reader owns `src`; false if none does.
    fn readWeight(self: *Context, dst: []u8, src: []const u8) bool {
        for (self.weight_readers) |maybe| if (maybe) |wr| {
            if (wr.read(wr.ctx, dst, src)) return true;
        };
        return false;
    }

/// Lets `uploadStaged` fetch weight bytes with a positional file read instead of
/// copying them out of the checkpoint mapping.
///
/// ⚠️ A read-once multi-GB checkpoint faulted through mmap runs at page-granularity
/// speed (~192 MB/s here) whenever the host is short of free RAM, because
/// `MADV_WILLNEED` is advisory and readahead gets throttled; large explicit reads
/// do not depend on that heuristic firing. It is a callback rather than a file
/// handle so this layer keeps knowing nothing about checkpoint formats.
///
/// Measured worth ~6-9% on a cold 13.5 GB DiT with ample free RAM; the big win it
/// targets (memory-pressured first step) is unproven — see `SafeTensors.readTo`.
///
/// `read` returns false to mean "not mine / could not read" — the caller then
/// falls back to the plain copy, so a missing or failing reader is never fatal.
pub const WeightReader = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *anyopaque, dst: []u8, src: []const u8) bool,
};

/// Fans the host-side fill of one pinned staging slot out over several threads.
///
/// ⚠️ **The fill, not the DMA, is what makes a multi-GB weight upload slow, and
/// nothing in the code said so until it was split out.** Measured on a 3090 with
/// an 11.57 GB Z-Image DiT: `fill 4.88s, slot-wait 0.04s` — i.e. PCIe was idle
/// 99% of the time and one thread reading the checkpoint at **2.37 GB/s** was the
/// whole warm-up. The same file reads at 4.3 GB/s cold and 26 GB/s page-cached
/// under `dd`, so a single reader is leaving most of both on the table: cold it is
/// short on queue depth, warm it is one core doing a 12 GB copy.
///
/// Positional reads are independent by construction (no shared file offset), and
/// the fallback `@memcpy` splits just as freely — so a range split is safe with no
/// ordering between workers. Chunks are claimed rather than pre-assigned so a
/// worker that lands on cold pages cannot hold up the batch.
const FillPool = struct {
    /// Total fillers including the caller, which works alongside them rather than
    /// blocking — so this is threads spawned + 1.
    const workers = 4;
    /// Claim granularity. Small enough that a straggler costs little, large enough
    /// that a claim's mutex round trip is noise against the read it guards.
    const chunk_bytes = 8 << 20;

    ctx: *Context,
    /// `std.Io`'s portable primitives rather than raw futex/pthread — this file
    /// is the one place a macOS/Windows port would otherwise have to rewrite.
    io: std.Io = undefined,
    mu: std.Io.Mutex = .init,
    have_work: std.Io.Condition = .init,
    batch_done: std.Io.Condition = .init,
    threads: [workers - 1]?std.Thread = @splat(null),
    /// Bumped once per batch; a worker waits for it to change.
    gen: u64 = 0,
    shutdown: bool = false,
    dst: []u8 = &.{},
    src: []const u8 = &.{},
    next: usize = 0,
    /// Chunks claimed but not yet finished. The batch is over at 0 with `next`
    /// past the end — counting claims (not completions) is what lets the producer
    /// join without knowing how the work was divided.
    busy: usize = 0,

    fn spawn(self: *FillPool, io: std.Io) void {
        self.io = io;
        for (&self.threads) |*t| t.* = std.Thread.spawn(.{}, loop, .{self}) catch null;
    }

    fn loop(self: *FillPool) void {
        const io = self.io;
        self.mu.lockUncancelable(io);
        var seen = self.gen;
        while (true) {
            while (self.gen == seen and !self.shutdown) self.have_work.waitUncancelable(io, &self.mu);
            if (self.shutdown) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            self.mu.unlock(io);
            self.drain();
            self.mu.lockUncancelable(io);
        }
    }

    /// Claim and fill chunks until none are left. Called by the workers AND by the
    /// producer, which is what keeps `workers` accurate at small sizes: a 29 MB
    /// weight is 4 chunks, and the producer taking one of them beats it idling.
    fn drain(self: *FillPool) void {
        const io = self.io;
        while (true) {
            self.mu.lockUncancelable(io);
            const off = self.next;
            if (off >= self.dst.len) {
                self.mu.unlock(io);
                return;
            }
            const n = @min(chunk_bytes, self.dst.len - off);
            self.next = off + n;
            self.busy += 1;
            const d = self.dst[off..][0..n];
            const s = self.src[off..][0..n];
            self.mu.unlock(io);

            if (!self.ctx.readWeight(d, s)) @memcpy(d, s);

            self.mu.lockUncancelable(io);
            self.busy -= 1;
            if (self.busy == 0 and self.next >= self.dst.len) self.batch_done.signal(io);
            self.mu.unlock(io);
        }
    }

    /// Fill `dst` from `src` across the pool, returning once every byte is in.
    fn run(self: *FillPool, dst: []u8, src: []const u8) void {
        const io = self.io;
        self.mu.lockUncancelable(io);
        self.dst = dst;
        self.src = src;
        self.next = 0;
        self.gen +%= 1;
        self.have_work.broadcast(io);
        self.mu.unlock(io);

        self.drain(); // the producer is a worker too

        self.mu.lockUncancelable(io);
        while (self.busy != 0) self.batch_done.waitUncancelable(io, &self.mu);
        self.mu.unlock(io);
    }

    fn deinit(self: *FillPool) void {
        const io = self.io;
        self.mu.lockUncancelable(io);
        self.shutdown = true;
        self.have_work.broadcast(io);
        self.mu.unlock(io);
        for (self.threads) |t| if (t) |th| th.join();
    }
};

    /// Allocate the pinned staging ring (`stage_slots` × `slot_size` device-pinned
    /// host buffers). Enables async weight uploads. Returns false if unsupported.
    pub fn initStaging(self: *Context, slot_size: usize, io: std.Io) bool {
        if (self.n_staging > 0) return true;
        var i: usize = 0;
        while (i < stage_slots) : (i += 1) {
            var p: ?*anyopaque = null;
            if (self.api.cuMemAllocHost(&p, slot_size) != cu.CUDA_SUCCESS) break;
            self.staging[i] = p;
            var ev: cu.CUevent = null;
            _ = self.api.cuEventCreate(&ev, cu.CU_EVENT_DISABLE_TIMING);
            self.staging_ev[i] = ev;
            self.n_staging = i + 1;
        }
        self.staging_size = slot_size;
        if (self.n_staging > 0 and !self.fill_ready) {
            self.fill.ctx = self;
            self.fill.spawn(io);
            self.fill_ready = true;
        }
        return self.n_staging > 0;
    }

    /// Staged async HtoD upload: memcpy the (pageable mmap) bytes into the next
    /// pinned ring slot, then async-DMA slot→device on the transfer stream, so the
    /// DMA overlaps compute. Records `ev` (weight ready) and the slot's reuse event.
    /// The memcpy runs on the host but overlaps the GPU's prior work. Falls back to
    /// a synchronous upload for data larger than a slot (leaves `ev` unrecorded — a
    /// wait on an unrecorded event is a no-op, and the sync copy already completed).
    pub fn uploadStaged(self: *Context, buf: Buffer, data: []const u8, ev: cu.CUevent) Error!void {
        std.debug.assert(data.len <= buf.bytes);
        if (self.n_staging == 0 or data.len > self.staging_size) return self.upload(buf, data);
        const i = self.staging_next;
        self.staging_next = (i + 1) % self.n_staging;
        const t_wait = monoNs();
        _ = self.api.cuEventSynchronize(self.staging_ev[i]); // slot free (prev DMA done)
        const t_read = monoNs();
        self.stage_wait_ns +%= t_read -% t_wait;
        const slot: [*]u8 = @ptrCast(self.staging[i].?);
        // Prefer a positional read straight into the pinned slot; fall back to the
        // copy when there is no reader, the bytes are not from its file, or the
        // read fails. Same destination either way, so this cannot change results.
        if (self.fill_ready) {
            self.fill.run(slot[0..data.len], data);
        } else if (!self.readWeight(slot[0..data.len], data)) {
            @memcpy(slot[0..data.len], data);
        }
        self.stage_read_ns +%= monoNs() -% t_read;
        self.htod_bytes += data.len;
        try self.check(self.api.cuMemcpyHtoDAsync(buf.ptr, slot, data.len, self.xfer_stream), "cuMemcpyHtoDAsync");
        _ = self.api.cuEventRecord(self.staging_ev[i], self.xfer_stream); // slot reusable after this
        try self.check(self.api.cuEventRecord(ev, self.xfer_stream), "cuEventRecord(xfer)");
    }

    /// Make the compute stream wait for a transfer-stream event (the weight upload)
    /// before subsequent launches read the buffer.
    pub fn computeWaitEvent(self: *Context, ev: cu.CUevent) Error!void {
        try self.check(self.api.cuStreamWaitEvent(self.stream, ev, 0), "cuStreamWaitEvent");
    }

    pub fn eventCreate(self: *Context) Error!cu.CUevent {
        var ev: cu.CUevent = null;
        try self.check(self.api.cuEventCreate(&ev, cu.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        return ev;
    }
    pub fn eventDestroy(self: *Context, ev: cu.CUevent) void {
        _ = self.api.cuEventDestroy(ev);
    }

    pub fn deviceName(self: *const Context) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    pub fn check(self: *Context, r: cu.CUresult, comptime what: []const u8) Error!void {
        if (r == cu.CUDA_SUCCESS) return;
        std.debug.print("CUDA {s} failed: {s} ({s})\n", .{ what, self.api.errName(r), self.api.errString(r) });
        return error.CudaError;
    }

    // ---- Modules ------------------------------------------------------------

    /// JIT-compile a PTX module (the driver's built-in ptxjitcompiler). `ptx_text`
    /// must be NUL-terminated. On failure the JIT error log is captured into
    /// `self.jit_log` and printed. On success any info log is captured too.
    pub fn loadModule(self: *Context, ptx_text: [:0]const u8) Error!Module {
        var info_buf: [16384]u8 = undefined;
        var opts = [_]c_int{
            cu.CU_JIT_INFO_LOG_BUFFER,
            cu.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            cu.CU_JIT_ERROR_LOG_BUFFER,
            cu.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            cu.CU_JIT_TARGET,
        };
        // Size options carry the integer in the pointer slot; the driver writes
        // the used length back into the same slot.
        var vals = [_]?*anyopaque{
            @ptrCast(&info_buf),
            @ptrFromInt(info_buf.len),
            @ptrCast(&self.jit_log),
            @ptrFromInt(self.jit_log.len),
            @ptrFromInt(86), // sm_86
        };
        var mod: cu.CUmodule = null;
        const t0 = monoNs();
        const r = self.api.cuModuleLoadDataEx(&mod, ptx_text.ptr, opts.len, &opts, &vals);
        self.jit_ns += monoNs() -% t0;
        self.jit_count += 1;
        self.jit_log_len = @intFromPtr(vals[3]);
        if (r != cu.CUDA_SUCCESS) {
            std.debug.print(
                "PTX JIT failed: {s}\n--- JIT error log ---\n{s}\n",
                .{ self.api.errName(r), self.jit_log[0..@min(self.jit_log_len, self.jit_log.len)] },
            );
            return error.CudaError;
        }
        return .{ .mod = mod };
    }

    /// Raise a function's dynamic-shared-memory cap above the 48 KB default
    /// (the whole point of the CUDA path). `bytes` may go up to
    /// `shared_optin_max` (~99 KB on sm_86).
    pub fn setMaxDynamicShared(self: *Context, func: cu.CUfunction, bytes: usize) Error!void {
        try self.check(
            self.api.cuFuncSetAttribute(func, cu.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, @intCast(bytes)),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED)",
        );
    }

    // ---- Buffers ------------------------------------------------------------

    pub fn alloc(self: *Context, bytes: usize) Error!Buffer {
        if (bytes == 0) return .{};
        var ptr: cu.CUdeviceptr = 0;
        // Distinguish OOM (so the backend can evict cached weights and retry)
        // from a hard driver error.
        const r = self.api.cuMemAlloc(&ptr, bytes);
        if (r == cu.CUDA_ERROR_OUT_OF_MEMORY) return error.DeviceOutOfMemory;
        try self.check(r, "cuMemAlloc");
        self.device_used += bytes;
        if (self.track_tags) self.mem_tag_used[@intFromEnum(self.mem_tag)] += bytes;
        return .{ .ptr = ptr, .bytes = bytes, .tag = self.mem_tag };
    }

    pub fn free(self: *Context, buf: *Buffer) void {
        if (buf.ptr != 0) {
            _ = self.api.cuMemFree(buf.ptr);
            self.device_used -|= buf.bytes;
            if (self.track_tags) self.mem_tag_used[@intFromEnum(buf.tag)] -|= buf.bytes;
        }
        buf.* = .{};
    }

    /// Set the tag applied to subsequent allocations (the pipeline brackets each
    /// phase). Only meaningful when `track_tags` is on.
    pub fn setMemTag(self: *Context, tag: MemTag) void {
        self.mem_tag = tag;
    }

    /// Live device bytes attributed to `tag` (0 unless tagging is active).
    pub fn memTagUsed(self: *const Context, tag: MemTag) u64 {
        return self.mem_tag_used[@intFromEnum(tag)];
    }

    /// Live device free/total bytes (cuMemGetInfo) — sees OTHER processes'
    /// usage, the CUDA analog of VK_EXT_memory_budget. Used for weight-stream
    /// budgeting; returns free=0 on query failure (forces conservative eviction).
    pub fn memGetInfo(self: *Context) struct { free: usize, total: usize } {
        var free_b: usize = 0;
        var total_b: usize = 0;
        if (self.api.cuMemGetInfo(&free_b, &total_b) != cu.CUDA_SUCCESS) return .{ .free = 0, .total = 0 };
        return .{ .free = free_b, .total = total_b };
    }

    // ---- VMM growable buffers (reserve VA once, commit physical chunks) ------

    fn vmmProp(self: *const Context) cu.CUmemAllocationProp {
        return .{
            .type = cu.CU_MEM_ALLOCATION_TYPE_PINNED,
            .location = .{ .type = cu.CU_MEM_LOCATION_TYPE_DEVICE, .id = self.dev },
        };
    }

    fn vmmQueryGranularity(self: *Context) usize {
        const f = self.api.cuMemGetAllocationGranularity orelse return 0;
        if (self.api.cuMemAddressReserve == null or self.api.cuMemAddressFree == null or
            self.api.cuMemCreate == null or self.api.cuMemRelease == null or
            self.api.cuMemMap == null or self.api.cuMemUnmap == null or
            self.api.cuMemSetAccess == null) return 0;
        var g: usize = 0;
        const prop = self.vmmProp();
        if (f(&g, &prop, cu.CU_MEM_ALLOC_GRANULARITY_MINIMUM) != cu.CUDA_SUCCESS) return 0;
        return g;
    }

    pub fn vmmAvailable(self: *const Context) bool {
        return self.vmm_granularity != 0;
    }

    /// Reserve `size` bytes of device virtual address space (no physical
    /// memory backs it until vmmCommit). `size` must be granularity-aligned.
    pub fn vmmReserve(self: *Context, size: usize) Error!cu.CUdeviceptr {
        var ptr: cu.CUdeviceptr = 0;
        try self.check(self.api.cuMemAddressReserve.?(&ptr, size, 0, 0, 0), "cuMemAddressReserve");
        return ptr;
    }

    /// Commit `size` bytes of physical memory at `base + offset` (both
    /// granularity multiples) inside a vmmReserve'd range. Returns the
    /// allocation handle (needed for release at teardown). OOM is returned
    /// distinctly so the backend can evict cached weights and retry.
    pub fn vmmCommit(self: *Context, base: cu.CUdeviceptr, offset: usize, size: usize) Error!cu.CUmemGenericAllocationHandle {
        const prop = self.vmmProp();
        var h: cu.CUmemGenericAllocationHandle = 0;
        const r = self.api.cuMemCreate.?(&h, size, &prop, 0);
        if (r == cu.CUDA_ERROR_OUT_OF_MEMORY) return error.DeviceOutOfMemory;
        try self.check(r, "cuMemCreate");
        errdefer _ = self.api.cuMemRelease.?(h);
        try self.check(self.api.cuMemMap.?(base + offset, size, 0, h, 0), "cuMemMap");
        errdefer _ = self.api.cuMemUnmap.?(base + offset, size);
        const desc = [1]cu.CUmemAccessDesc{.{
            .location = .{ .type = cu.CU_MEM_LOCATION_TYPE_DEVICE, .id = self.dev },
            .flags = cu.CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        }};
        try self.check(self.api.cuMemSetAccess.?(base + offset, size, &desc, 1), "cuMemSetAccess");
        self.device_used += size;
        return h;
    }

    /// Unmap and release a growable buffer's committed prefix and handles,
    /// then free the VA reservation. All device work reading the range must
    /// have completed (callers sync the stream first).
    pub fn vmmFree(self: *Context, base: cu.CUdeviceptr, va_size: usize, committed: usize, handles: []const cu.CUmemGenericAllocationHandle) void {
        if (committed != 0) _ = self.api.cuMemUnmap.?(base, committed);
        for (handles) |h| _ = self.api.cuMemRelease.?(h);
        _ = self.api.cuMemAddressFree.?(base, va_size);
        self.device_used -|= committed;
    }

    pub fn upload(self: *Context, buf: Buffer, data: []const u8) Error!void {
        std.debug.assert(data.len <= buf.bytes);
        self.htod_bytes += data.len;
        try self.check(self.api.cuMemcpyHtoD(buf.ptr, data.ptr, data.len), "cuMemcpyHtoD");
    }

    pub fn download(self: *Context, buf: Buffer, out: []u8) Error!void {
        std.debug.assert(out.len <= buf.bytes);
        // Ensure prior async work is done before the (blocking) copy.
        try self.check(self.api.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        try self.check(self.api.cuMemcpyDtoH(out.ptr, buf.ptr, out.len), "cuMemcpyDtoH");
    }

    pub fn memsetD8(self: *Context, buf: Buffer, value: u8, bytes: usize) Error!void {
        try self.check(self.api.cuMemsetD8(buf.ptr, value, bytes), "cuMemsetD8");
    }

    pub fn memsetD32(self: *Context, buf: Buffer, value: u32, count: usize) Error!void {
        try self.check(self.api.cuMemsetD32(buf.ptr, value, count), "cuMemsetD32");
    }

    /// Async memsets on the context (compute) stream. Unlike the legacy
    /// null-stream variants above, these stay ordered within the single-stream
    /// batch (no hidden full-stream sync) and, crucially, are legal inside a
    /// CUDA-graph capture — a null-stream memset mid-capture aborts it with
    /// STREAM_CAPTURE_IMPLICIT.
    pub fn memsetD8Async(self: *Context, buf: Buffer, value: u8, bytes: usize) Error!void {
        try self.check(self.api.cuMemsetD8Async(buf.ptr, value, bytes, self.stream), "cuMemsetD8Async");
    }

    pub fn memsetD32Async(self: *Context, buf: Buffer, value: u32, count: usize) Error!void {
        try self.check(self.api.cuMemsetD32Async(buf.ptr, value, count, self.stream), "cuMemsetD32Async");
    }

    // ---- Launch -------------------------------------------------------------

    /// Launch a kernel on the context stream. `params` is a slice of pointers to
    /// each argument value (build with `&arg` casts). Does not synchronize.
    pub fn launch(
        self: *Context,
        func: cu.CUfunction,
        grid: [3]u32,
        block: [3]u32,
        shared_bytes: u32,
        params: []?*anyopaque,
    ) Error!void {
        try self.check(self.api.cuLaunchKernel(
            func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_bytes,
            self.stream,
            if (params.len != 0) params.ptr else null,
            null,
        ), "cuLaunchKernel");
    }

    pub fn synchronize(self: *Context) Error!void {
        try self.check(self.api.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
    }

    // ---- Event timing -------------------------------------------------------

    pub const Timer = struct {
        start: cu.CUevent,
        stop: cu.CUevent,
    };

    pub fn timerCreate(self: *Context) Error!Timer {
        var t: Timer = undefined;
        try self.check(self.api.cuEventCreate(&t.start, 0), "cuEventCreate");
        try self.check(self.api.cuEventCreate(&t.stop, 0), "cuEventCreate");
        return t;
    }

    pub fn timerDestroy(self: *Context, t: Timer) void {
        _ = self.api.cuEventDestroy(t.start);
        _ = self.api.cuEventDestroy(t.stop);
    }

    pub fn timerBegin(self: *Context, t: Timer) Error!void {
        try self.check(self.api.cuEventRecord(t.start, self.stream), "cuEventRecord");
    }

    /// End timing and return elapsed milliseconds (device-measured, avoids host
    /// clock noise — the clock-governor caveat still applies, so take a min).
    pub fn timerEndMs(self: *Context, t: Timer) Error!f32 {
        try self.check(self.api.cuEventRecord(t.stop, self.stream), "cuEventRecord");
        try self.check(self.api.cuEventSynchronize(t.stop), "cuEventSynchronize");
        var ms: f32 = 0;
        try self.check(self.api.cuEventElapsedTime(&ms, t.start, t.stop), "cuEventElapsedTime");
        return ms;
    }
};

test {
    _ = Context;
}
