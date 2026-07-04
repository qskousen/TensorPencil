//! End-to-end text-to-image pipeline: tokenize -> encode -> sample -> decode.
//!
//! Models load in stages and are freed as soon as their output is captured,
//! bounding peak memory to roughly the DiT mapping (~13 GiB) plus activations.

const std = @import("std");
const gpu_mod = @import("gpu.zig");
const ops = @import("ops.zig");
const tokenizer_mod = @import("tokenizer.zig");
const safetensors = @import("safetensors.zig");
const sampler = @import("sampler.zig");
const image = @import("image.zig");
const qwen3 = @import("models/qwen3.zig");
const qwen3_gpu = @import("models/qwen3_gpu.zig");
const krea2_text = @import("models/krea2_text.zig");
const dit_mod = @import("models/dit.zig");
const dit_gpu = @import("models/dit_gpu.zig");
const dit_cuda = @import("models/dit_cuda.zig");
const qwen3_cuda = @import("models/qwen3_cuda.zig");
const cuda = @import("gpu/cuda.zig");
const wan_vae = @import("models/wan_vae.zig");
const vae_gpu = @import("models/vae_gpu.zig");
const vae_cuda = @import("models/vae_cuda.zig");

/// Compute backend for the diffusion model (and, for Vulkan, the encoder + VAE):
///  - cpu:      everything on CPU.
///  - vulkan:   encoder / DiT / VAE GEMMs offloaded to Vulkan (falls back to CPU
///              per-stage when the device is unavailable / out of VRAM).
///  - zig_cuda: DiT on the hand-PTX CUDA backend (int8 convrot checkpoint only);
///              encoder + VAE stay on CPU. `--backend zig-cuda` on the CLI.
pub const Backend = enum {
    cpu,
    vulkan,
    zig_cuda,

    /// Parse a CLI value ("cpu" / "vulkan" / "zig-cuda"); null if unrecognized.
    pub fn fromStr(s: []const u8) ?Backend {
        if (std.mem.eql(u8, s, "cpu")) return .cpu;
        if (std.mem.eql(u8, s, "vulkan")) return .vulkan;
        if (std.mem.eql(u8, s, "zig-cuda")) return .zig_cuda;
        return null;
    }
};

pub const Options = struct {
    prompt: []const u8,
    negative: []const u8 = "",
    width: usize = 1024,
    height: usize = 1024,
    steps: usize = 8,
    cfg: f32 = 1.0,
    seed: u64 = 0,
    shift: f32 = sampler.default_shift,
    /// Compute backend for the sampling loop (and encoder/VAE where supported).
    backend: Backend = .cpu,
    /// Cap on device memory (bytes; 0 = query the driver's live budget).
    /// Weights past the cap stream per step instead of staying resident.
    vram_budget: u64 = 0,
    /// Run the GPU text encoder's GEMMs on tensor cores (f16). ~0.4s faster
    /// encode but ~doubles its image-delta contribution; default f32.
    encoder_f16: bool = false,
    text_encoder_path: []const u8 = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors",
    dit_path: []const u8 = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors",
    vae_path: []const u8 = "models/vae/krea2RealVae_v10.safetensors",
};

pub const Image = struct {
    /// Interleaved RGB, [height][width][3].
    rgb: []u8,
    width: usize,
    height: usize,

    pub fn deinit(self: *Image, gpa: std.mem.Allocator) void {
        gpa.free(self.rgb);
        self.* = undefined;
    }
};

/// Stripped conditioning: [seq][12][2560] plus its length.
const Cond = struct {
    data: []f32,
    seq: usize,
};

pub fn generate(io: std.Io, gpa: std.mem.Allocator, opts: Options, progress: ?*std.Io.Writer) !Image {
    if (opts.width % 16 != 0 or opts.height % 16 != 0) return error.SizeNotMultipleOf16;
    if (opts.steps < 1) return error.NoSteps;
    const lat_h = opts.height / 8;
    const lat_w = opts.width / 8;
    const lat_len = wan_vae.latent_channels * lat_h * lat_w;
    const use_cfg = opts.cfg != 1.0;
    const total_start = std.Io.Clock.real.now(io);

    // Vulkan context (--backend vulkan): encoder / DiT / VAE GEMMs on Vulkan.
    var gpu_ctx: ?*gpu_mod.Context = null;
    if (opts.backend == .vulkan) {
        if (gpu_mod.Context.init(gpa)) |ctx| {
            gpu_ctx = ctx;
            ctx.budget_override = opts.vram_budget;
            ops.matmul.gpu = ctx;
            try note(progress, "gpu: {s}\n", .{ctx.deviceName()});
        } else |err| {
            try note(progress, "gpu unavailable ({t}); using cpu\n", .{err});
        }
    }
    defer if (gpu_ctx) |ctx| {
        ops.matmul.gpu = null;
        ctx.deinit();
    };

    // Hand-PTX CUDA backend (--backend zig-cuda): the WHOLE pipeline — text
    // encoder (qwen3_cuda), DiT sampling (dit_cuda), and VAE decode (vae_cuda) —
    // runs on the CUDA backend. All weights stream through its cache, so
    // --vram-budget degrades to weight streaming and it coexists with other GPU
    // workloads via the live cuMemGetInfo budget.
    var cu_be: ?*cuda.Backend = null;
    if (opts.backend == .zig_cuda) {
        if (cuda.Backend.init(gpa)) |b| {
            cu_be = b;
            b.budget_override = opts.vram_budget; // --vram-budget: stream weights past this cap
            try note(progress, "cuda dit: {s}\n", .{b.deviceName()});
        } else |err| {
            try note(progress, "cuda unavailable ({t}); using cpu dit\n", .{err});
        }
    }
    defer if (cu_be) |b| b.deinit();

    // Stage 1: text encoding (encoder freed before the DiT loads).
    var cond_pos: Cond = undefined;
    var cond_neg: ?Cond = null;
    {
        try note(progress, "loading text encoder...\n", .{});
        const load_start = std.Io.Clock.real.now(io);
        var tok = try tokenizer_mod.Tokenizer.init(gpa);
        defer tok.deinit();
        var st = try safetensors.SafeTensors.open(gpa, io, opts.text_encoder_path);
        defer st.deinit();
        var enc = try qwen3.TextEncoder.load(gpa, &st);
        defer enc.deinit();
        const enc_start = std.Io.Clock.real.now(io);
        const load_s = @as(f64, @floatFromInt(enc_start.nanoseconds - load_start.nanoseconds)) / 1e9;
        try note(progress, "text encoder loaded in {d:.1}s\n", .{load_s});

        cond_pos = try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &tok, &enc, opts.prompt);
        if (use_cfg) cond_neg = try encodePrompt(io, gpa, gpu_ctx, cu_be, opts.encoder_f16, &tok, &enc, opts.negative);
        const enc_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - enc_start.nanoseconds)) / 1e9;
        try note(progress, "encoded prompt ({d} tokens{s}) in {d:.1}s\n", .{ cond_pos.seq, if (use_cfg) " + negative" else "", enc_s });
    }
    // The encoder's fp8 weights are stale now (its checkpoint mapping is closed);
    // drop them so they don't linger in the CUDA cache through DiT sampling.
    if (cu_be) |b| b.evictWeights();
    defer gpa.free(cond_pos.data);
    defer if (cond_neg) |c| gpa.free(c.data);

    // Stage 2: flow-matching sampling.
    const x = try gpa.alloc(f32, lat_len);
    defer gpa.free(x);
    sampler.fillNoise(x, opts.seed);

    const sigmas = try sampler.simpleSchedule(gpa, opts.steps, opts.shift);
    defer gpa.free(sigmas);

    {
        try note(progress, "loading diffusion model...\n", .{});
        const dit_start = std.Io.Clock.real.now(io);
        var st = try safetensors.SafeTensors.open(gpa, io, opts.dit_path);
        defer st.deinit();
        var dit = try dit_mod.DiT.load(gpa, &st);
        defer dit.deinit();
        // NOTE: async weight streaming (pinned staging ring) is NOT enabled here —
        // it measured SLOWER than synchronous streaming (the driver's cuMemcpyHtoD
        // already pipelines its internal staging, whereas the explicit ring puts a
        // single-threaded mmap→pinned memcpy on the critical path). True overlap
        // needs a prefetch thread; until then production uses sync streaming, which
        // already matches/beats the Vulkan backend. The infra stays for that work
        // (exercised by `cuda-stream-test`).

        const v = try gpa.alloc(f32, lat_len);
        defer gpa.free(v);
        const v_neg = if (use_cfg) try gpa.alloc(f32, lat_len) else null;
        defer if (v_neg) |b| gpa.free(b);

        // Per-run GPU session: text fusion, rope table, and the schedule's
        // timestep vectors are computed and uploaded once.
        var sess_pos: ?dit_gpu.Session = null;
        defer if (sess_pos) |*sp| sp.deinit(gpa, gpu_ctx.?);
        var sess_neg: ?dit_gpu.Session = null;
        defer if (sess_neg) |*sn| sn.deinit(gpa, gpu_ctx.?);
        // One workspace serves both sessions (sized for the longer text).
        var ws: ?dit_gpu.Workspace = null;
        defer if (ws) |*w| w.deinit(gpu_ctx.?);
        // DiT-on-Vulkan session, only when the DiT actually runs on Vulkan. Under
        // --backend zig-cuda, gpu_ctx exists for the encoder + VAE but the DiT runs
        // on cu_be, so skip this unused session (a full CPU text-fusion pass + its
        // workspace VRAM).
        if (cu_be == null) if (gpu_ctx) |gc| {
            sess_pos = try dit_gpu.Session.init(gpa, io, gc, &dit, lat_h, lat_w, cond_pos.data, cond_pos.seq, sigmas);
            if (use_cfg) {
                sess_neg = try dit_gpu.Session.init(gpa, io, gc, &dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq, sigmas);
            }
            const seq_txt_cap = @max(cond_pos.seq, if (cond_neg) |c| c.seq else 0);
            ws = try dit_gpu.Workspace.init(gc, lat_h, lat_w, seq_txt_cap);
        };
        var cu_pos: ?dit_cuda.Session = null;
        defer if (cu_pos) |*s| s.deinit(cu_be.?);
        var cu_neg: ?dit_cuda.Session = null;
        defer if (cu_neg) |*s| s.deinit(cu_be.?);
        var cu_ws: ?dit_cuda.Workspace = null;
        defer if (cu_ws) |*w| w.deinit(cu_be.?);
        if (cu_be) |b| {
            cu_pos = try dit_cuda.Session.init(gpa, io, b, &dit, lat_h, lat_w, cond_pos.data, cond_pos.seq);
            if (use_cfg) cu_neg = try dit_cuda.Session.init(gpa, io, b, &dit, lat_h, lat_w, cond_neg.?.data, cond_neg.?.seq);
            const cu_seq_cap = @max(cond_pos.seq, if (cond_neg) |c| c.seq else 0);
            cu_ws = try dit_cuda.Workspace.init(b, lat_h, lat_w, cu_seq_cap);
        }
        const dit_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dit_start.nanoseconds)) / 1e9;
        try note(progress, "diffusion model ready in {d:.1}s\n", .{dit_s});

        const sampling_start = std.Io.Clock.real.now(io);
        for (0..opts.steps) |i| {
            const start = std.Io.Clock.real.now(io);
            if (cu_be) |b| {
                try dit_cuda.forward(&dit, b, &cu_pos.?, &cu_ws.?, io, gpa, v, x, sigmas[i]);
            } else if (gpu_ctx) |gc| {
                try dit_gpu.forward(&dit, gc, &sess_pos.?, &ws.?, io, gpa, v, x, sigmas[i]);
            } else {
                try dit.forward(io, gpa, v, x, lat_h, lat_w, sigmas[i], cond_pos.data, cond_pos.seq);
            }
            if (use_cfg) {
                if (cu_be) |b| {
                    try dit_cuda.forward(&dit, b, &cu_neg.?, &cu_ws.?, io, gpa, v_neg.?, x, sigmas[i]);
                } else if (gpu_ctx) |gc| {
                    try dit_gpu.forward(&dit, gc, &sess_neg.?, &ws.?, io, gpa, v_neg.?, x, sigmas[i]);
                } else try dit.forward(io, gpa, v_neg.?, x, lat_h, lat_w, sigmas[i], cond_neg.?.data, cond_neg.?.seq);
                sampler.applyCfg(v, v_neg.?, opts.cfg);
            }
            sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
            const ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - start.nanoseconds)) / 1e6;
            try note(progress, "step {d}/{d}  sigma {d:.3} -> {d:.3}  ({d:.1}s)\n", .{ i + 1, opts.steps, sigmas[i], sigmas[i + 1], ms / 1000.0 });
        }
        const sampling_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - sampling_start.nanoseconds)) / 1e9;
        try note(progress, "sampling {d} steps in {d:.1}s ({d:.2}s/step)\n", .{ opts.steps, sampling_s, sampling_s / @as(f64, @floatFromInt(opts.steps)) });
    }

    // DiT weight buffers are stale after this point; drop them so VAE weights
    // can't collide with a recycled host pointer in the cache, and — under
    // --backend zig-cuda — so the resident DiT int8 weights (~14 GiB) free the
    // VRAM the VAE decode needs (the CUDA VAE re-uploads its own f32 weights).
    if (gpu_ctx) |ctx| ctx.evictWeights();
    if (cu_be) |b| b.evictWeights();

    // Stage 3: denormalize and decode.
    {
        const plane = lat_h * lat_w;
        for (0..wan_vae.latent_channels) |c| {
            for (x[c * plane ..][0..plane]) |*val| {
                val.* = val.* * wan_vae.latents_std[c] + wan_vae.latents_mean[c];
            }
        }
    }
    try note(progress, "decoding latent...\n", .{});
    const dec_start = std.Io.Clock.real.now(io);
    var st = try safetensors.SafeTensors.open(gpa, io, opts.vae_path);
    defer st.deinit();
    var vae = try wan_vae.Decoder.load(gpa, &st);
    defer vae.deinit();
    const planar = if (cu_be) |b|
        try vae_cuda.decode(&vae, b, io, gpa, x, lat_h, lat_w)
    else if (gpu_ctx) |gc|
        vae_gpu.decode(&vae, gc, io, gpa, x, lat_h, lat_w) catch |err| switch (err) {
            // Not enough free VRAM for the decode activations (e.g. another
            // process is holding the GPU). Fall back to a pure-CPU decode:
            // its buffers live in system RAM, so it degrades gracefully
            // instead of crashing. Disable GEMM offload for the duration so
            // the fallback can't bounce straight back into the same OOM.
            error.DeviceOutOfMemory => blk: {
                try note(progress, "vae decode: out of VRAM, falling back to CPU decode\n", .{});
                const saved = ops.matmul.gpu;
                ops.matmul.gpu = null;
                defer ops.matmul.gpu = saved;
                break :blk try vae.decode(io, gpa, x, lat_h, lat_w);
            },
            else => return err,
        }
    else
        try vae.decode(io, gpa, x, lat_h, lat_w);
    defer gpa.free(planar);
    const dec_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - dec_start.nanoseconds)) / 1e9;
    try note(progress, "decoded in {d:.1}s\n", .{dec_s});

    const rgb = try image.planarF32ToRgb8(gpa, planar, opts.width, opts.height);

    const total_s = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - total_start.nanoseconds)) / 1e9;
    try note(progress, "total time {d:.1}s\n", .{total_s});

    return .{ .rgb = rgb, .width = opts.width, .height = opts.height };
}

fn encodePrompt(io: std.Io, gpa: std.mem.Allocator, gpu_ctx: ?*gpu_mod.Context, cu_be: ?*cuda.Backend, encoder_f16: bool, tok: *const tokenizer_mod.Tokenizer, enc: *const qwen3.TextEncoder, text: []const u8) !Cond {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(tok, gpa, text, &ids);

    // GPU-resident encode (batched, keeps the device saturated): the CUDA
    // backend when active, else Vulkan; the CPU forward is the fallback (and
    // used on any GPU error).
    const full = if (cu_be) |b|
        qwen3_cuda.encode(enc, b, io, gpa, ids.items) catch try enc.encode(io, gpa, ids.items)
    else if (gpu_ctx) |gc|
        qwen3_gpu.encode(enc, gc, io, gpa, ids.items, encoder_f16) catch try enc.encode(io, gpa, ids.items)
    else
        try enc.encode(io, gpa, ids.items);
    defer gpa.free(full);

    const offset = krea2_text.stripOffset(ids.items);
    const seq = ids.items.len - offset;
    const row = qwen3.tap_count * qwen3.hidden;
    const data = try gpa.alloc(f32, seq * row);
    @memcpy(data, full[offset * row ..][0 .. seq * row]);
    return .{ .data = data, .seq = seq };
}

fn note(progress: ?*std.Io.Writer, comptime fmt: []const u8, args: anytype) !void {
    if (progress) |w| {
        try w.print(fmt, args);
        try w.flush();
    }
}

test "options validation" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expectError(error.SizeNotMultipleOf16, generate(io, gpa, .{ .prompt = "x", .width = 100, .height = 96 }, null));
    try std.testing.expectError(error.NoSteps, generate(io, gpa, .{ .prompt = "x", .steps = 0 }, null));
}
