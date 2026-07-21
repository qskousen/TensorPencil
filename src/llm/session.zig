//! Shared tp-llm session machinery, factored out of the CLI (llm_main.zig) so
//! every architecture/backend path is configured and driven identically:
//!  - weight residency (pin everything; LLM weights never stream),
//!  - the per-run timing summary and CUDA profile dump,
//!  - CUDA backend bring-up (`bringUpCuda`), and
//!  - the generic `run` that owns the cpu/cuda/vulkan construct→prefill→drive→time
//!    skeleton, parameterized by a per-architecture `Spec` (the concrete stepper
//!    types + builders) plus a caller-supplied `prefiller` and `driver`.
//!
//! `run` stays free of model imports: the arch-specific stepper types live in the
//! `Spec` a wrapper hands in, and generation (incl. qwen3 speculative decoding) is
//! delegated to the `driver`, so all model/eagle type references stay in the CLI.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const gpu_context = @import("tp_gpu").context;
const kv_cache = @import("tp_core").kv_cache;
const chat = @import("chat.zig");
const tokenizer = @import("tp_core").tokenizer;

// LLM weights are NEVER streamed: every weight pins on first touch
// (`pinAllWeights` below), and a model that outgrows VRAM degrades via the
// CPU layer split (--cpu-layers/--offload-grow, or the GUI's always-armed
// dynamic offload) — measured ~2.5x faster than the old weight-streaming
// fallback, whose LRU-vs-cyclic-walk pathology re-uploaded ~the whole model
// per token the moment the budget fell short. Weight streaming remains a
// diffusion-only mechanism.

/// End-of-response telemetry a stepper reports: committed context length, the
/// growable window ceiling it can grow to, and device VRAM in use (null on the
/// CPU backend, which has no device). Mirrors the diffusion side's per-image
/// stats line. `model` is the `*Stepper` pointer the generation loop drives.
pub const Stats = struct {
    tokens: usize,
    window: usize,
    vram: ?u64,

    pub fn of(model: anytype) Stats {
        const M = @typeInfo(@TypeOf(model)).pointer.child;
        const tokens = model.cached();
        const window = if (comptime @hasDecl(M, "capacityMax")) model.capacityMax() else tokens + model.remaining();
        const vram = if (comptime @hasDecl(M, "vramUsed")) model.vramUsed() else null;
        return .{ .tokens = tokens, .window = window, .vram = vram };
    }

    /// " , vram 8123 MiB" (leading separator) or "" when there's no device —
    /// so callers can splice it into a stats line uniformly.
    pub fn vramSuffix(self: Stats, buf: []u8) []const u8 {
        const mb = (self.vram orelse return "") >> 20;
        return std.fmt.bufPrint(buf, ", vram {d} MiB", .{mb}) catch "";
    }
};

/// The per-run timing summary. A one-shot (`--prompt`) run prints tokens +
/// tok/s + context + VRAM + setup; an interactive session prints just the token
/// total + setup (elapsed would count the user's typing time between turns, and
/// each turn already reported its own stats line).
pub fn printSummary(stdout: *std.Io.Writer, one_shot: bool, n: usize, setup_s: f64, elapsed_s: f64, stats: Stats) !void {
    if (one_shot) {
        var vbuf: [32]u8 = undefined;
        try stdout.print("\n\n[{d} tokens in {d:.1}s, {d:.2} tok/s; ctx {d}/{d}{s}; setup {d:.1}s]\n", .{
            n, elapsed_s, @as(f64, @floatFromInt(n)) / elapsed_s, stats.tokens, stats.window, stats.vramSuffix(&vbuf), setup_s,
        });
    } else {
        try stdout.print("[session over: {d} tokens generated; setup was {d:.1}s]\n", .{ n, setup_s });
    }
}

/// Per-category device profile rows (`--profile`), one line per non-empty
/// CUDA op category. The caller prints any header and flushes.
pub fn printCudaProfile(stdout: *std.Io.Writer, be: *const cuda.Backend) !void {
    inline for (@typeInfo(cuda.Backend.ProfCat).@"enum".fields, 0..) |f, ci| {
        if (be.prof.n[ci] > 0)
            try stdout.print("  {s:<9} {d:>8.1} ms  ({d} launches)\n", .{ f.name, be.prof.ms[ci], be.prof.n[ci] });
    }
}

/// Compute backend, shared by the CLI dispatch and `run`. Spelled to match the
/// `--backend` CLI values.
pub const BackendKind = enum { cpu, @"zig-cuda", cuda, vulkan };

/// Whether a backend has a working KV cache path for `dt`. f32 is always
/// supported; f16 is enabled per-backend as its attention/append kernels land
/// (CUDA = Phase 1, Vulkan = Phase 2, CPU = Phase 3). The CLI and GUI both
/// gate on this so a mis-set dtype fails cleanly instead of corrupting.
pub fn kvDtypeSupported(backend: BackendKind, dt: kv_cache.KvDtype) bool {
    if (dt == .f32) return true;
    return switch (backend) {
        // CUDA: f16 + q8_0 attention/append/store kernels are wired for every
        // arch (qwen3 hd128, gemma/qwen35 hd256, gemma4 hd512, both graph and
        // non-graph paths). qwen3's EAGLE-tap/tree speculative paths stay
        // f32-only and reject other dtypes at enable time.
        .cuda, .@"zig-cuda" => true,
        // Vulkan: f16 + q8_0 on gemma3/qwen35 (hd256, attn_dsplit_gemma_f16/_q8).
        // qwen3 Vulkan rejects non-f32 in its builder (hd128 path, broken gen).
        .vulkan => true,
        // CPU: f16 packs 2/f32-slot and q8_0 packs ggml 34-byte blocks in
        // KvCache/PerLayerKvCache, expanded on read. All CPU archs support both.
        .cpu => true,
    };
}

/// Device handles a `run` may need: `cu_be` for the CUDA arms, `vk_ctx` for the
/// Vulkan arm; the CPU arm needs neither.
pub const Devices = struct {
    cu_be: ?*cuda.Backend = null,
    vk_ctx: ?*gpu_context.Context = null,
};

/// Result of a `run`: tokens generated and the timestamp generation began
/// (`t0`), so the caller can split setup vs. generation time for the summary.
pub const RunResult = struct { n: usize, t0: i96, stats: Stats = .{ .tokens = 0, .window = 0, .vram = null } };

/// Bring up the CUDA backend for a GPU session: `initLibs` (--backend cuda) or
/// `init` (--backend zig-cuda), set profiling, and pin every weight resident
/// (LLM weights never stream; see the note at the top of this file).
/// Returns null for the cpu / vulkan backends (no CUDA device).
pub fn bringUpCuda(arena: std.mem.Allocator, backend: BackendKind, profile: bool) !?*cuda.Backend {
    const be: ?*cuda.Backend = switch (backend) {
        .cuda => try cuda.Backend.initLibs(arena),
        .@"zig-cuda" => try cuda.Backend.init(arena),
        .cpu, .vulkan => null,
    };
    if (be) |b| {
        b.profile = profile;
        b.pinAllWeights();
    }
    return be;
}

/// Per-architecture descriptor for `run`: the concrete stepper types plus the
/// builders that normalize their `init` signatures. `Vulkan` may be `void` for
/// an arch with no Vulkan backend (its vulkan arm then errors). `UniformSpec`
/// generates one for the common case where CUDA/Vulkan `init` is
/// `(gpa, dev, *LM, cap)` (no per-arch `first_seq`); qwen3 hand-writes a `Spec`
/// whose CUDA/Vulkan builders thread `first_seq` through.
pub fn UniformSpec(
    comptime Model_: type,
    comptime Cpu_: type,
    comptime Cuda_: type,
    comptime Vulkan_: type,
) type {
    return struct {
        pub const Model = Model_;
        pub const Cpu = Cpu_;
        pub const Cuda = Cuda_;
        pub const Vulkan = Vulkan_;

        pub fn buildCpu(gpa: std.mem.Allocator, lm: *const Model, cap: kv_cache.Capacity) !Cpu {
            return Cpu.init(gpa, lm, cap);
        }
        pub fn buildCuda(gpa: std.mem.Allocator, be: *cuda.Backend, lm: *const Model, cap: kv_cache.Capacity, first_seq: usize) !Cuda {
            _ = first_seq; // uniform steppers size KV from `cap`, not the prompt
            return Cuda.init(gpa, be, lm, cap);
        }
        pub fn buildVulkan(gpa: std.mem.Allocator, ctx: *gpu_context.Context, lm: *const Model, cap: kv_cache.Capacity, first_seq: usize) !Vulkan {
            _ = first_seq;
            return Vulkan.init(gpa, ctx, lm, cap);
        }
    };
}

/// Construct the per-backend stepper (`S.build*`), run the arch's one-shot image
/// prefill (`prefiller.prefill`), then drive generation (`driver.drive`) — the
/// skeleton every architecture shares. The stepper type never escapes its arm,
/// so no tagged union is needed; `prefiller`/`driver` are small structs with
/// generic (`model: anytype`) methods. `first_seq` is the prompt length the
/// qwen3 CUDA/Vulkan steppers need to size fixed batch/tap buffers (ignored by
/// the uniform builders).
///
/// `driver.drive(&model) !RunResult` owns the setup-vs-generation boundary: it
/// does any pre-generation work (drafter construction, cpu-split), then stamps
/// `t0` immediately before generating, so the caller's setup/elapsed split
/// attributes that work to setup (a plain driver stamps `t0` right away).
pub fn run(
    comptime S: type,
    dev: Devices,
    backend: BackendKind,
    lm: *const S.Model,
    first_seq: usize,
    prefiller: anytype,
    driver: anytype,
    io: std.Io,
    gpa: std.mem.Allocator,
    cap: kv_cache.Capacity,
    stdout: *std.Io.Writer,
) !RunResult {
    if (!kvDtypeSupported(backend, cap.kv_dtype)) {
        try stdout.print("--kv-dtype {s} is not supported on the {s} backend yet\n", .{ cap.kv_dtype.label(), @tagName(backend) });
        return error.KvDtypeUnsupported;
    }
    switch (backend) {
        .cpu => {
            var model = try S.buildCpu(gpa, lm, cap);
            defer model.deinit();
            // The gemma CPU steppers run image prefill before the first step and
            // need `io`; qwen3/qwen35 steppers take it per-call and have no field.
            if (@hasField(@TypeOf(model), "io")) model.io = io;
            try prefiller.prefill(&model);
            return driver.drive(&model);
        },
        .@"zig-cuda", .cuda => {
            // Weights never stream: an OOM anywhere below means the model (plus
            // KV/activations) genuinely doesn't fit resident — point at the CPU
            // split instead of failing silently.
            errdefer |err| if (err == error.DeviceOutOfMemory) {
                stdout.writeAll("\n[out of device VRAM: LLM weights never stream — rerun with --vram-budget <GiB> and --cpu-layers/--offload-grow to run part of the model on the CPU]\n") catch {};
                stdout.flush() catch {};
            };
            var model = try S.buildCuda(gpa, dev.cu_be.?, lm, cap, first_seq);
            defer model.deinit();
            // Prefill may arm a CPU split on VRAM pressure (auto-offload), whose
            // host layers need `io`; set it before prefill, not just at decode.
            if (@hasField(@TypeOf(model), "io")) model.io = io;
            try prefiller.prefill(&model);
            return driver.drive(&model);
        },
        .vulkan => {
            if (comptime S.Vulkan == void) {
                try stdout.writeAll("this architecture has no vulkan backend yet\n");
                return error.UnsupportedBackend;
            } else {
                const ctx = dev.vk_ctx.?;
                // LLM weights never stream: pin everything; a model that
                // doesn't fit fails cleanly (Vulkan has no CPU-split path).
                ctx.pin_budget = std.math.maxInt(u64);
                errdefer |err| if (err == error.DeviceOutOfMemory) {
                    stdout.writeAll("\n[out of device VRAM: this model doesn't fit resident, and the vulkan backend has no CPU-offload path]\n") catch {};
                    stdout.flush() catch {};
                };
                var model = try S.buildVulkan(gpa, ctx, lm, cap, first_seq);
                defer model.deinit();
                try prefiller.prefill(&model);
                return driver.drive(&model);
            }
        },
    }
}

/// A no-op `prefiller` for text-only sessions (no image to interleave).
pub const no_prefill: struct {
    pub fn prefill(_: @This(), model: anytype) !void {
        _ = model;
    }
} = .{};

/// Append an interleaved image+text user turn and prefill it — the shared core
/// of the CLI's `imageTurn` and the GUI worker's `imageTurn`/`imageTurnGemma4`.
/// Builds the family-aware segment layout, opens the assistant turn, grows the
/// KV window to fit, then injects each encoded image's embeddings at its
/// placeholder rows (text between/around images is prefilled as plain tokens).
/// The engine's later `cached()`-based prefill handles the tail after the last
/// image + `openAssistant`.
///
/// `model` is any stepper exposing cached/remaining/ensureCapacity/prefill/
/// prefillImage; `segs` is the caller-built segment list (image placeholders +
/// text, in display order); `encs` is a slice whose elements have `.embeds`/
/// `.grid_w`/`.grid_h` (the arch ViT's `Encoded`) — one per image segment, in
/// order. The image ENCODE (arch-specific ViT) stays with the caller; this owns
/// only the backend-agnostic layout + interleave. Errors (e.g. ContextFull from
/// ensureCapacity) propagate after `segs`/assistant tokens are already appended,
/// so a caller that wants to bail cleanly should snapshot `ids.items.len` first
/// and `shrinkRetainingCapacity` on error.
pub fn prefillImageTurn(
    model: anytype,
    tok: *const tokenizer.Tokenizer,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    segs: []const chat.Segment,
    encs: anytype,
) !void {
    var image_rows: std.ArrayList(usize) = .empty;
    defer image_rows.deinit(gpa);
    try chat.appendUserSegments(tok, gpa, segs, ids, &image_rows);
    try chat.openAssistant(tok, gpa, ids);

    if (ids.items.len > model.cached() + model.remaining()) {
        try model.ensureCapacity(ids.items.len);
    }
    for (image_rows.items, encs) |row, e| {
        const before = ids.items[model.cached()..row];
        if (before.len > 0) try model.prefill(before);
        try model.prefillImage(e.embeds, e.grid_w, e.grid_h);
    }
}

/// A `prefiller` for a one-shot `--image` turn: prefill the tokens before the
/// image block, then the image embeddings (in place of the placeholder rows),
/// then all but the last token — the engine's generate() prefills that one and
/// samples. A no-op when `img` is null (text-only). `Embeds` is the arch's ViT
/// `Encoded` type (needs `.embeds`, `.grid_w`, `.grid_h`); `ids` is the built
/// prompt and `n_pre`/`n_img` bound the placeholder block within it.
pub fn ImagePrefiller(comptime Embeds: type) type {
    return struct {
        img: ?*const Embeds = null,
        n_pre: usize = 0,
        n_img: usize = 0,
        ids: []const u32 = &.{},
        pub fn prefill(self: @This(), model: anytype) !void {
            const M = @typeInfo(@TypeOf(model)).pointer.child;
            // Steppers without an image path (e.g. qwen35's CPU model — its
            // images are CUDA-only) never receive an image; compile the interleave
            // away for them. `img` is null on those backends at runtime anyway.
            if (comptime !@hasDecl(M, "prefillImage")) return;
            const e = self.img orelse return;
            try model.prefill(self.ids[0..self.n_pre]);
            try model.prefillImage(e.embeds, e.grid_w, e.grid_h);
            try model.prefill(self.ids[self.n_pre + self.n_img .. self.ids.len - 1]);
        }
    };
}

test "bringUpCuda returns null for cpu / vulkan (no device)" {
    // cpu / vulkan never touch a CUDA device, so this is safe without one.
    try std.testing.expect((try bringUpCuda(std.testing.allocator, .cpu, false)) == null);
    try std.testing.expect((try bringUpCuda(std.testing.allocator, .vulkan, false)) == null);
}

// Stub steppers + model to exercise `run`'s CPU arm end-to-end (forcing the
// generic `run`/`UniformSpec`/`no_prefill` to be analyzed on CPU, no device).
// `run` analyzes every arm, so the CPU and CUDA steppers must carry the real
// (differing) init arities — CPU `(gpa, lm, cap)`, CUDA `(gpa, be, lm, cap)`.
const StubModel = struct {};
const StubCpu = struct {
    driven: bool = false,
    pub fn init(gpa: std.mem.Allocator, lm: *const StubModel, cap: kv_cache.Capacity) !StubCpu {
        _ = gpa;
        _ = lm;
        _ = cap;
        return .{};
    }
    pub fn deinit(self: *StubCpu) void {
        _ = self;
    }
};
const StubCuda = struct {
    driven: bool = false,
    pub fn init(gpa: std.mem.Allocator, be: *cuda.Backend, lm: *const StubModel, cap: kv_cache.Capacity) !StubCuda {
        _ = gpa;
        _ = be;
        _ = lm;
        _ = cap;
        return .{};
    }
    pub fn deinit(self: *StubCuda) void {
        _ = self;
    }
};
const CountPrefiller = struct {
    calls: *usize,
    pub fn prefill(self: CountPrefiller, model: anytype) !void {
        _ = model;
        self.calls.* += 1;
    }
};
const StubDriver = struct {
    pub fn drive(_: StubDriver, model: anytype) !RunResult {
        model.driven = true; // proves `run` handed us a mutable stepper pointer
        return .{ .n = 7, .t0 = 0 };
    }
};

test "run cpu arm: build → prefill → drive, returns driver's token count" {
    const S = UniformSpec(StubModel, StubCpu, StubCuda, void);
    var lm: StubModel = .{};
    var calls: usize = 0;
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    const res = try run(
        S,
        .{},
        .cpu,
        &lm,
        0,
        CountPrefiller{ .calls = &calls },
        StubDriver{},
        std.testing.io,
        std.testing.allocator,
        kv_cache.Capacity.fixed(1),
        &w,
    );
    try std.testing.expectEqual(@as(usize, 7), res.n);
    try std.testing.expectEqual(@as(usize, 1), calls); // prefiller ran once
}
