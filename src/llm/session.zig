//! Shared tp-llm session machinery, factored out of the CLI (llm_main.zig) so
//! every architecture/backend path is configured and driven identically:
//!  - weight residency (pin everything; LLM weights never stream),
//!  - the per-run timing summary and CUDA profile dump,
//!  - CUDA backend bring-up (`bringUpCuda`), and
//!  - the generic `run` that owns the cpu/cuda/vulkan construct->prefill->drive->time
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
const gguf_mod = @import("tp_core").gguf;
const chat = @import("chat.zig");
const engine = @import("engine.zig");
const tokenizer = @import("tp_core").tokenizer;

// LLM weights are NEVER streamed: every weight pins on first touch
// (`pinAllWeights` below), and a model that outgrows VRAM degrades via the
// CPU layer split (--cpu-layers/--offload-grow, or the GUI's always-armed
// dynamic offload), measured ~2.5x faster than the old weight-streaming
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

    /// ", vram 8123 MiB" (leading separator) or "" when there's no device,
    /// so callers can splice it into a stats line uniformly.
    pub fn vramSuffix(self: Stats, buf: []u8) []const u8 {
        const mb = (self.vram orelse return "") >> 20;
        return std.fmt.bufPrint(buf, ", vram {d} MiB", .{mb}) catch "";
    }
};

/// Format a generate call's prefill/decode split as ", pp 512 @ 1068.0 tok/s,
/// tg 600 @ 24.2 tok/s" (leading separator), or "" when the engine reported no
/// split (the speculative path). Splitting matters: `elapsed / n` charges the
/// prompt forward and every first-touch device cost to token generation, which
/// on a short run reads far below the real decode rate.
pub fn rateSuffix(t: engine.Timing, buf: []u8) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    if (t.ppRate()) |pp| w.print(", pp {d} @ {d:.1} tok/s", .{ t.prefill_tokens, pp }) catch return "";
    if (t.tgRate()) |tg| w.print(", tg {d} @ {d:.1} tok/s", .{ t.decode_tokens, tg }) catch return "";
    return w.buffered();
}

/// The per-run timing summary. A one-shot (`--prompt`) run prints tokens +
/// the prefill/decode rate split + context + VRAM + setup; an interactive
/// session prints just the token total + setup (elapsed would count the user's
/// typing time between turns, and each turn already reported its own stats
/// line). `timing` is the engine's split, summed over the run's turns.
pub fn printSummary(stdout: *std.Io.Writer, one_shot: bool, n: usize, setup_s: f64, elapsed_s: f64, stats: Stats, timing: engine.Timing) !void {
    if (one_shot) {
        var vbuf: [32]u8 = undefined;
        var rbuf: [96]u8 = undefined;
        try stdout.print("\n\n[{d} tokens in {d:.1}s{s}; ctx {d}/{d}{s}; setup {d:.1}s]\n", .{
            n, elapsed_s, rateSuffix(timing, &rbuf), stats.tokens, stats.window, stats.vramSuffix(&vbuf), setup_s,
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
pub const RunResult = struct {
    n: usize,
    t0: i96,
    stats: Stats = .{ .tokens = 0, .window = 0, .vram = null },
    /// The engine's prefill/decode split, summed over the run's turns.
    timing: engine.Timing = .{},
};

/// Weight-noise CURVE for every GPU session brought up from here on: a
/// `core/noise_curve.zig` expression in the normalized layer depth `t` (a bare
/// number is a flat curve). A global because it is a process-wide CLI choice
/// (`--weight-noise`) that has to reach four separate `bringUpCuda` call sites,
/// exactly like `safetensors.read_mode`. A GUI sets `backend.weight_noise`
/// directly and ignores this.
pub var weight_noise_curve: []const u8 = "";

/// The curve's `a` (`--weight-noise-amount`). 1 by default, so a curve written with
/// literal amplitudes means what it says and only a curve that mentions `a` cares.
pub var weight_noise_amount: f32 = 1;

/// Base of the weight-noise stream (`--weight-noise-seed`). Same seed, curve and
/// prompt reproduce a generation exactly.
pub var weight_noise_seed: u32 = 0;

/// Whether an architecture publishes a per-layer sigma, which is what makes weight
/// noise reach its GEMMs at all. Today exactly one does.
///
/// The coupling is concrete: a stepper honors noise iff it declares `noiseAtLayer`
/// (so `transformer_gpu.decoderLayer*` can tell it which layer is launching) AND
/// ticks the stream in its forward. Extend this list in the same commit that adds
/// the second one, or the GUI will offer the knob for a model that ignores it.
/// `weight noise arch list matches the steppers` in `gui/chat.zig` fails if the two
/// drift.
pub fn archSupportsWeightNoise(arch: []const u8) bool {
    return std.mem.eql(u8, arch, "gemma4");
}

/// Whether weight noise would actually do anything to this checkpoint. Both halves
/// are real and independent: a stepper that never publishes a layer index leaves
/// every sigma at slot 0, and a checkpoint whose linears are in an unwired dtype
/// (a Gemma 4 12B QAT file is q4_0) flows through kernels that never read sigma.
///
/// The dtype test is a majority over the block-quant weights under `blk.`, not a
/// single probe tensor: a mixed file is normal (this repo's 31B q4_k carries 11
/// q5_k tensors), and one supported tensor in an otherwise unsupported model would
/// answer yes to a question the user is really asking about the whole model.
pub fn weightNoiseSupported(gg: *const gguf_mod.Gguf) bool {
    const arch = gg.getStr("general.architecture") orelse return false;
    if (!archSupportsWeightNoise(arch)) return false;
    var quant: usize = 0;
    var wired: usize = 0;
    for (gg.names()) |name| {
        // `Gguf` hands out CANONICAL names, so per-layer tensors are `layers.N.…`,
        // not the file's own `blk.N.…`. Matching the raw spelling counted zero
        // quantized weights and reported every checkpoint unsupported, including
        // the q4_k one this was developed against. Both are accepted so a caller
        // holding raw names is not silently wrong either.
        if (!std.mem.startsWith(u8, name, "layers.") and !std.mem.startsWith(u8, name, "blk.")) continue;
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        const dt = (gg.get(name) orelse continue).info.dtype;
        if (!dt.isBlockQuant()) continue; // norms are f32 and cannot carry noise
        quant += 1;
        if (cuda.Backend.weightNoiseSupported(dt)) wired += 1;
    }
    return quant != 0 and wired * 2 >= quant;
}

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
        b.enableLlmMemTags(); // before any allocation; see enableLlmMemTags
        b.weight_noise.seed = weight_noise_seed;
        // The curve needs the layer count to evaluate; the model supplies that in
        // its own init (`setDepth`), which re-evaluates whatever is set here.
        b.weight_noise.amount = weight_noise_amount;
        b.weight_noise.setCurve(weight_noise_curve);
    }
    return be;
}

/// Point the CUDA weight cache at the checkpoint FILE, so a weight upload reads
/// its bytes positionally instead of faulting them out of the mapping.
///
/// Register every GGUF the session opened (model + mmproj): each reader answers
/// only for its own mapping. No-op without a CUDA backend, and `Gguf.readTo`
/// itself no-ops unless `safetensors.read_mode == .pread`, so `--mmap mmap`
/// disables it without any caller needing to know.
///
/// The `Gguf` must outlive the backend, it is borrowed, not copied. Every
/// caller holds it for the whole session, which is why this takes a pointer.
pub fn useFileReads(be: ?*cuda.Backend, g: *const gguf_mod.Gguf) void {
    const b = be orelse return;
    b.ctx.addWeightReader(.{
        .ctx = @constCast(g),
        .read = struct {
            fn read(ctx: *anyopaque, dst: []u8, src: []const u8) bool {
                const gg: *const gguf_mod.Gguf = @ptrCast(@alignCast(ctx));
                return gg.readTo(dst, src);
            }
        }.read,
    });
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
/// prefill (`prefiller.prefill`), then drive generation (`driver.drive`), the
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
            // KV/activations) genuinely doesn't fit resident, point at the CPU
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
            // Upload the weights BEFORE the first forward. `cachedWeight` is
            // lazy, so without this the one-time host->device copy of every weight
            // is charged to prefill, inflating `pp` and making it incomparable to
            // llama.cpp, which offloads at load time and reports that separately.
            // Opt-in per stepper; those without the decl keep the old behaviour.
            if (@hasDecl(@TypeOf(model), "warmWeights")) model.warmWeights();
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

/// Append an interleaved image+text user turn and prefill it, the shared core
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
/// `.grid_w`/`.grid_h` (the arch ViT's `Encoded`), one per image segment, in
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
/// then all but the last token, the engine's generate() prefills that one and
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
            // Steppers without an image path (e.g. qwen35's CPU model, its
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
// (differing) init arities, CPU `(gpa, lm, cap)`, CUDA `(gpa, be, lm, cap)`.
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

/// Build a minimal in-memory GGUF: `arch`, and `n` per-layer weights of `type_id`
/// plus one f32 norm (which must not be counted as a noisable weight).
fn testGguf(gpa: std.mem.Allocator, arch: []const u8, type_id: u32, n: usize) ![]u8 {
    var b = try gguf_mod.TestBuilder.init(gpa, 3, n + 1, 1);
    errdefer b.deinit();
    try b.kvStr("general.architecture", arch);
    var buf: [64]u8 = undefined;
    for (0..n) |i| {
        const name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{i});
        try b.tensor(name, &.{ 256, 2 }, type_id, 0);
    }
    try b.tensor("blk.0.ffn_norm.weight", &.{256}, 0, 0); // f32
    // One 256x2 q4_k row pair is 2 blocks of 144 bytes; the f32 norm is 1 KiB.
    // Oversized payload is fine, the reader only needs the ranges to fit.
    const payload = try gpa.alloc(u8, 1 << 16);
    defer gpa.free(payload);
    @memset(payload, 0);
    return b.finish(payload);
}

test "weightNoiseSupported gates on both the arch and the weight dtype" {
    const gpa = std.testing.allocator;

    // gemma4 + q4_k: the case this was built for.
    {
        const bytes = try testGguf(gpa, "gemma4", 12, 4); // 12 = q4_k
        defer gpa.free(bytes);
        var gg = try gguf_mod.Gguf.initFromSlice(gpa, bytes);
        defer gg.deinit();
        // Guards the bug this test exists for: `Gguf` canonicalizes `blk.N.…` to
        // `layers.N.…`, and scanning for the raw spelling found nothing at all.
        try std.testing.expect(weightNoiseSupported(&gg));
    }
    // gemma4 + q4_0 (the 12B QAT format): right arch, unwired kernels.
    {
        const bytes = try testGguf(gpa, "gemma4", 2, 4); // 2 = q4_0
        defer gpa.free(bytes);
        var gg = try gguf_mod.Gguf.initFromSlice(gpa, bytes);
        defer gg.deinit();
        try std.testing.expect(!weightNoiseSupported(&gg));
    }
    // Right dtype, wrong arch: no stepper publishes a layer index.
    {
        const bytes = try testGguf(gpa, "qwen35", 12, 4);
        defer gpa.free(bytes);
        var gg = try gguf_mod.Gguf.initFromSlice(gpa, bytes);
        defer gg.deinit();
        try std.testing.expect(!weightNoiseSupported(&gg));
    }
    // A checkpoint with no quantized layer weights at all cannot be perturbed,
    // and must not divide by a zero count.
    {
        const bytes = try testGguf(gpa, "gemma4", 0, 0); // f32 only
        defer gpa.free(bytes);
        var gg = try gguf_mod.Gguf.initFromSlice(gpa, bytes);
        defer gg.deinit();
        try std.testing.expect(!weightNoiseSupported(&gg));
    }
}
