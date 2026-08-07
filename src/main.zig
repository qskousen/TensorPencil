//! TensorPencil CLI — thin driver over the TensorPencil library module.

const std = @import("std");
const Io = std.Io;

const TensorPencil = @import("TensorPencil");

/// `--vram-budget min`: hold only the in-flight weights (~2 at a time). 256 MiB
/// comfortably fits the two largest int8 linears (~100 MiB each) plus scales, so
/// no single op sync-thrashes, while every other weight streams per step.
const min_vram_budget: u64 = 256 << 20;

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    if (args.len >= 2 and std.mem.eql(u8, args[1], "gpu-test")) {
        // Survey cooperative-matrix configs (incl. int8 tensor cores) on init.
        TensorPencil.gpu.context.dump_coop_configs = true;
        var ctx = try TensorPencil.gpu.Context.init(arena);
        defer ctx.deinit();
        try stdout.print("device: {s}\n", .{ctx.deviceName()});
        try stdout.print("coop matrix f16->f32:  {d}x{d}x{d}\n", .{ ctx.coop_m, ctx.coop_n, ctx.coop_k });
        try stdout.print("coop matrix i8->i32:   {d}x{d}x{d}  ({s})\n", .{
            ctx.coop_i8_m, ctx.coop_i8_n, ctx.coop_i8_k,
            if (ctx.coop_i8_m != 0) "int8 tensor cores available" else "no int8 coop config",
        });
        // Tiny correctness check: y = x @ W^T with W = 2*I.
        const n = 8;
        var wdata: [n * n]f32 = @splat(0);
        for (0..n) |i| wdata[i * n + i] = 2.0;
        var x: [2 * n]f32 = undefined;
        for (&x, 0..) |*v, i| v.* = @floatFromInt(i);
        var y: [2 * n]f32 = undefined;
        try ctx.matmul(&y, &x, 2, std.mem.sliceAsBytes(&wdata), false, n, n, 1.0, null);
        for (y, 0..) |v, i| {
            if (v != 2.0 * @as(f32, @floatFromInt(i))) {
                try stdout.print("MISMATCH at {d}: {d}\n", .{ i, v });
                try stdout.flush();
                return error.GpuMismatch;
            }
        }
        try stdout.print("gpu matmul OK\n", .{});
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "gpu-i8-test")) {
        try gpuI8Test(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-test")) {
        try cudaTest(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-libs-test")) {
        try cudaLibsTest(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-libs-i8-test")) {
        try cudaLibsI8Test(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-libs-f16-test")) {
        try cudaLibsF16Test(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-libs-attn-test")) {
        try cudaLibsAttnTest(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-libs-i8fused-test")) {
        try cudaLibsI8FusedTest(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-i8-test")) {
        try cudaI8Test(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-i4-test")) {
        try cudaI4Test(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-fp8-test")) {
        try cudaFp8Test(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-encode-test")) {
        try cudaEncodeTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "sd-cuda-test")) {
        // `libs` selects the cuBLASLt/cuDNN arm, which takes a DIFFERENT attention
        // path (true head width, no padding) — so both are worth running.
        var ckpt: []const u8 = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) libs = true else ckpt = a;
        }
        try sdCudaTest(arena, io, stdout, ckpt, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "te-test")) {
        var path: []const u8 = "/home/qt/genai/comfyui/models/text_encoders/Qwen3-4B-Q4_K_M.gguf";
        var ref: []const u8 = "";
        var variant: TensorPencil.models.qwen3.Variant = .zimage;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--ref") and i + 1 < args.len) {
                i += 1;
                ref = args[i];
            } else if (std.mem.eql(u8, args[i], "--krea2")) {
                variant = .krea2;
            } else path = args[i];
        }
        try teTest(arena, io, stdout, path, ref, variant);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "zimage-cuda-test")) {
        var ckpt: []const u8 = "/home/qt/genai/comfyui/models/checkpoints/zit/unstableRevolution_V2Fp16.safetensors";
        var vae: []const u8 = "/home/qt/genai/comfyui/models/vae/z-image-turbo.vae.safetensors";
        var libs = false;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true //
            else if (std.mem.eql(u8, args[i], "--vae") and i + 1 < args.len) {
                i += 1;
                vae = args[i];
            } else ckpt = args[i];
        }
        try zimageCudaTest(arena, io, stdout, ckpt, vae, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "zimage-cuda-bench")) {
        var seq: usize = 6848;
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) libs = true else seq = std.fmt.parseInt(usize, a, 10) catch seq;
        }
        try zimageCudaBench(arena, io, stdout, seq, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-vae-test")) {
        const zh: usize = if (args.len >= 3) (std.fmt.parseInt(usize, args[2], 10) catch 16) else 16;
        try cudaVaeTest(arena, io, stdout, zh);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-dit-test")) {
        const path = if (args.len >= 3) args[2] else "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
        const lat: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 32) else 32;
        var loop = false;
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "loop")) loop = true;
            if (std.mem.eql(u8, a, "libs")) libs = true;
        }
        try cudaDitTest(arena, io, stdout, path, lat, loop, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-attn-test")) {
        const cuda = TensorPencil.gpu.cuda;
        var ctx = cuda.Context.init(arena) catch |err| {
            try stdout.print("cuda unavailable: {t}\n", .{err});
            return;
        };
        defer ctx.deinit();
        try stdout.print("cuda device: {s} (sm_{d}{d})\n", .{ ctx.deviceName(), ctx.cc_major, ctx.cc_minor });
        try cuda.kernels.attnTest(&ctx, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-txtfusion-test")) {
        const path = if (args.len >= 3) args[2] else "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
        const seq_txt: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 448) else 448;
        try cudaTxtFusionTest(arena, io, stdout, path, seq_txt);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-attn-cmp")) {
        try cudaAttnCmp(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-stream-test")) {
        const path = if (args.len >= 3) args[2] else "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
        const lat: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 32) else 32;
        const budget_gib: f64 = if (args.len >= 5) (std.fmt.parseFloat(f64, args[4]) catch 3.0) else 3.0;
        try cudaStreamTest(arena, io, stdout, path, lat, budget_gib);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "generate")) {
        try generate(arena, io, stdout, args[2..]);
    } else if (args.len >= 6 and std.mem.eql(u8, args[1], "decode-latent")) {
        try decodeLatent(arena, io, stdout, args[2], args[3], args[4], args[5]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-matmul")) {
        try benchMatmul(arena, io, stdout);
    } else if (args.len >= 3 and std.mem.eql(u8, args[1], "inspect")) {
        // Placeholder driver: inspect a safetensors file. Replaced by the real
        // `generate` command as the pipeline comes together.
        var st = try TensorPencil.SafeTensors.open(arena, io, args[2]);
        defer st.deinit();
        try stdout.print("{s}: {d} tensors\n", .{ args[2], st.count() });
        for (st.names()) |name| {
            const view = st.get(name).?;
            try stdout.print("  {s}  {any}  {t}\n", .{ name, view.info.shape.slice(), view.info.dtype });
        }
    } else {
        try stdout.print(
            \\TensorPencil — Krea 2 inference engine
            \\usage:
            \\  TensorPencil generate --prompt "..." [options]
            \\      --negative ""      negative prompt (needs --cfg != 1)
            \\      --prompt-syntax    prompt dialect: comfy | a1111. They are NOT
            \\                         spellings of one language: under a1111
            \\                         (x:1.2) MULTIPLIES rather than replaces the
            \\                         weight, [x] de-emphasizes, BREAK forces a
            \\                         chunk boundary, and [a:b:0.5] / [a|b]
            \\                         schedule the prompt across steps
            \\      --emphasis         how an a1111 weight reaches the hidden
            \\                         states: original | no_norm | ignore
            \\                         (ignored under comfy)
            \\      --width 1024       image width  (multiple of 16)
            \\      --height 1024      image height (multiple of 16)
            \\      --steps 8          sampling steps
            \\      --cfg 1.0          guidance scale (1.0 = no negative pass)
            \\      --seed 0           noise seed
            \\      --shift 1.15       flow-matching sigma shift
            \\      --sampler euler    sampler: euler | dpmpp_2m_sde
            \\                         | dpmpp_2m_sde_heun. The two SDE variants are
            \\                         DPM-Solver++(2M) SDE with ComfyUI's midpoint
            \\                         and heun corrections; both inject noise from a
            \\                         seed-determined Brownian tree, so the same
            \\                         --seed reproduces ComfyUI's render
            \\      --sde-eta 1.0      SDE noise level (0 = deterministic DPM++(2M))
            \\      --sde-s-noise 1.0  SDE noise multiplier
            \\      --scheduler        where the steps go: normal | karras
            \\                         | exponential | sgm_uniform | simple
            \\                         | ddim_uniform | beta | linear_quadratic
            \\                         | kl_optimal. Default is the architecture's
            \\                         own (simple for krea2, normal for SD).
            \\                         ddim_uniform and beta may return a step count
            \\                         different from --steps
            \\      --compat comfy     whose SAMPLING conventions to reproduce:
            \\                         comfy | a1111. Orthogonal to
            \\                         --prompt-syntax; reproducing an a1111 seed
            \\                         needs both. a1111 sets all three of the
            \\                         overrides below to ITS defaults, which
            \\                         differ from ComfyUI on every one
            \\      --rng cpu          noise generator: cpu (torch's CPU MT19937,
            \\                         what ComfyUI uses) | nv (NVIDIA Philox,
            \\                         what a1111 uses by default, since its
            \\                         randn_source defaults to "GPU"). This one
            \\                         changes the image entirely, not slightly
            \\      --sgm-noise-mult   scale the initial latent by sqrt(1+s0^2)
            \\                         (on, ComfyUI/SGM) or by a bare s0 (off,
            \\                         a1111's default) (on/off)
            \\      --quantize-t       condition the UNet on the nearest TRAINED
            \\                         timestep (on, ComfyUI) or on the fractional
            \\                         one (off, a1111's default) (on/off)
            \\      --backend cpu      compute backend: cpu | vulkan | zig-cuda
            \\                         | cuda. vulkan offloads encoder/DiT/VAE
            \\                         GEMMs to Vulkan; zig-cuda runs the whole
            \\                         pipeline on the pure-Zig hand-PTX CUDA
            \\                         backend; cuda runs it on NVIDIA's dlopen'd
            \\                         cuBLASLt/cuDNN kernels (both take an fp8 /
            \\                         int8|int4 convrot / dense bf16 --dit ckpt)
            \\      --vram-budget 0    GiB of device memory to use (0 = ask the
            \\                         driver); weights past it stream per step.
            \\                         "min" holds only the in-flight weights
            \\                         (~2 at a time) — lowest VRAM, but streams
            \\                         every weight each step (slow; pair w/ a
            \\                         small image for sub-GiB total)
            \\      --encoder-f16 off  run the text encoder GEMMs on tensor
            \\                         cores (f16): ~0.4s faster, slightly less
            \\                         exact conditioning (on/off)
            \\      --dit-f32 off      run the diffusion model in full f32
            \\                         instead of the f16 tensor-core path
            \\                         (slower, more exact) (on/off)
            \\      --dit <path>       diffusion checkpoint (fp8 / int8 / int4
            \\                         convrot / dense bf16; auto-detected). The
            \\                         "model.diffusion_model." key prefix is also
            \\                         auto-detected. Default: krea2 Fp8
            \\      --vae <path>       VAE decoder checkpoint
            \\      --text-encoder <path>  text-encoder checkpoint (qwen3 for krea2,
            \\                         CLIP-L for the SD family)
            \\      --text-encoder-2 <path>  SDXL's second text encoder (OpenCLIP
            \\                         bigG), only for a split-file checkpoint
            \\      --mmap pread       how checkpoint bytes are read:
            \\                         pread (default) = mapped, but weight bytes
            \\                         fetched with positional reads — does not
            \\                         depend on kernel readahead, which collapses
            \\                         to page-granularity speed on a cold
            \\                         multi-GB file when RAM is short;
            \\                         mmap = read straight from the mapping;
            \\                         buffered = read the whole file into RAM up
            \\                         front. Use buffered on ZFS (mmap faulting
            \\                         can deadlock there under memory pressure).
            \\      --out out.png      output file
            \\  TensorPencil inspect <file.safetensors>   list tensors in a checkpoint
            \\  TensorPencil bench-matmul                 time a DiT-sized fp8 GEMM
            \\  TensorPencil decode-latent <z.bin> <zh> <zw> <out.png>
            \\  TensorPencil sd-cuda-test [<sd1.5 ckpt>] [libs]
            \\      check the CUDA SD kernels against the CPU ops they reproduce
            \\      ("libs" runs the cuBLASLt/cuDNN arm, whose attention path
            \\      differs); exits non-zero if any check fails
            \\  TensorPencil te-test [<encoder>] [--ref <encoder>] [--krea2]
            \\      check a Qwen3 text encoder on every GPU backend against its
            \\      own CPU forward (kernel check), and optionally against a
            \\      second encoder (quantization check). Accepts .gguf
            \\  TensorPencil zimage-cuda-test [<zimage ckpt>] [libs]
            \\      check zimage_cuda's device forward against the CPU forward on
            \\      real weights, on both attention paths; non-zero if any fails
            \\  TensorPencil zimage-cuda-bench [<seq>] [libs]
            \\      time one Z-Image step's worth of bf16 GEMMs, with and without
            \\      the f32<->bf16 conversion passes, to isolate where the time goes
            \\
        , .{});
    }

    try stdout.flush();
}

/// Validate + time the raw int8 tensor-core GEMM (s8*s8->s32) against a CPU
/// reference. Correctness first (small shape), then DiT-block-sized timing.
fn gpuI8Test(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    var ctx = try TensorPencil.gpu.Context.init(arena);
    defer ctx.deinit();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    if (ctx.pipe_coop_i8 == .null_handle) {
        try stdout.print("no int8 cooperative-matrix pipeline (coop_i8 {d}x{d}x{d})\n", .{ ctx.coop_i8_m, ctx.coop_i8_n, ctx.coop_i8_k });
        return;
    }
    try stdout.print("int8 coop: {d}x{d}x{d}\n", .{ ctx.coop_i8_m, ctx.coop_i8_n, ctx.coop_i8_k });

    const Case = struct { m: usize, n: usize, k: usize, check: bool };
    // m a multiple of 16*i8_mt (32), n a multiple of 16*i8_nt (64), k of 32.
    const cases = [_]Case{
        .{ .m = 64, .n = 64, .k = 64, .check = true }, // register kernel
        .{ .m = 64, .n = 256, .k = 128, .check = true }, // register kernel
        .{ .m = 128, .n = 128, .k = 64, .check = true }, // shared kernel (128-mult)
        .{ .m = 256, .n = 384, .k = 320, .check = true }, // shared kernel
        .{ .m = 4224, .n = 6144, .k = 6144, .check = false }, // DiT-sized timing
        .{ .m = 7680, .n = 6144, .k = 6144, .check = false }, // DiT qkv @1120x1680
        .{ .m = 7680, .n = 16384, .k = 6144, .check = false }, // DiT mlp gate/up
        .{ .m = 7680, .n = 6144, .k = 16384, .check = false }, // DiT mlp.down
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    for (cases) |c| {
        const m = c.m;
        const n = c.n;
        const k = c.k;
        const xb = try arena.alloc(u8, m * k); // s8 activations
        for (xb) |*v| v.* = @bitCast(rand.int(i8));
        const wb = try arena.alloc(u8, n * k); // s8 weights [n][k]
        for (wb) |*v| v.* = @bitCast(rand.int(i8));

        var x_d = try ctx.tensorCreate(m * k);
        defer ctx.tensorDestroy(&x_d);
        var y_d = try ctx.tensorCreate(m * n * 4);
        defer ctx.tensorDestroy(&y_d);
        try ctx.tensorUpload(x_d, xb);

        const iters: usize = if (c.check) 1 else 6;
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * n * k));
        for (0..iters) |it| {
            const start = std.Io.Clock.real.now(io);
            try ctx.opMatmulCoopI8(y_d, x_d, m, wb, n, k);
            const end = std.Io.Clock.real.now(io);
            const ns: f64 = @floatFromInt(end.nanoseconds - start.nanoseconds);
            if (!c.check) {
                const tag: []const u8 = if (it == 0) " (incl. weight upload)" else "";
                try stdout.print("i8 coop GEMM {d}x{d}x{d}: {d:.2} ms, {d:.1} GFLOP/s{s}\n", .{ m, n, k, ns / 1e6, flops / ns, tag });
                try stdout.flush();
            }
        }

        if (c.check) {
            const y = try arena.alloc(u8, m * n * 4);
            try ctx.tensorDownload(y_d, y);
            const yi: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, y));
            var mism: usize = 0;
            for (0..m) |i| {
                for (0..n) |j| {
                    var acc: i32 = 0;
                    for (0..k) |kk| {
                        const xv: i32 = @as(i8, @bitCast(xb[i * k + kk]));
                        const wv: i32 = @as(i8, @bitCast(wb[j * k + kk]));
                        acc += xv * wv;
                    }
                    if (yi[i * n + j] != acc) {
                        if (mism < 5) try stdout.print("  MISMATCH [{d},{d}]: gpu={d} cpu={d}\n", .{ i, j, yi[i * n + j], acc });
                        mism += 1;
                    }
                }
            }
            try stdout.print("i8 coop GEMM {d}x{d}x{d}: {d} / {d} mismatches\n", .{ m, n, k, mism, m * n });
            try stdout.flush();
            if (mism != 0) return error.GpuMismatch;
        }
    }

    // --- full int8 linear: rotate + dynamic quantize + GEMM + rescale -------
    const convrot = TensorPencil.ops.convrot;
    const LCase = struct { m: usize, rows: usize, cols: usize, check: bool };
    const lcases = [_]LCase{
        .{ .m = 48, .rows = 64, .cols = 256, .check = true },
        .{ .m = 48, .rows = 128, .cols = 512, .check = true },
        .{ .m = 128, .rows = 128, .cols = 6144, .check = true }, // Stage B fused prep (f16)
        .{ .m = 128, .rows = 128, .cols = 16384, .check = true }, // Stage B fused prep, mlp.down cols

        .{ .m = 4224, .rows = 6144, .cols = 6144, .check = false }, // DiT-sized
    };
    for (lcases) |c| {
        const m = c.m;
        const rows = c.rows;
        const cols = c.cols;
        const xf = try arena.alloc(f32, m * cols);
        for (xf) |*v| v.* = rand.floatNorm(f32);
        const wb = try arena.alloc(u8, rows * cols); // pre-rotated int8 weight
        for (wb) |*v| v.* = @bitCast(rand.int(i8));
        const wscale = try arena.alloc(f32, rows);
        for (wscale) |*s| s.* = 0.001 + rand.float(f32) * 0.02;

        var x_d = try ctx.tensorCreate(m * cols * 4);
        defer ctx.tensorDestroy(&x_d);
        var y_d = try ctx.tensorCreate(m * rows * 4);
        defer ctx.tensorDestroy(&y_d);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(xf));

        const iters: usize = if (c.check) 1 else 6;
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * rows * cols));
        for (0..iters) |it| {
            const start = std.Io.Clock.real.now(io);
            // Batch the prep+GEMM+scale chain into one submission (as the DiT
            // does) so the timing isn't dominated by per-op submit+wait.
            try ctx.beginBatch();
            try ctx.opMatmulI8(y_d, x_d, m, wb, wscale, rows, cols);
            try ctx.endBatch();
            const end = std.Io.Clock.real.now(io);
            const ns: f64 = @floatFromInt(end.nanoseconds - start.nanoseconds);
            if (!c.check) {
                const tag: []const u8 = if (it == 0) " (incl. weight upload)" else "";
                try stdout.print("i8 linear {d}x{d}x{d}: {d:.2} ms, {d:.1} GFLOP/s{s}\n", .{ m, rows, cols, ns / 1e6, flops / ns, tag });
                try stdout.flush();
            }
        }
        if (!c.check) continue;

        const yg = try arena.alloc(u8, m * rows * 4);
        try ctx.tensorDownload(y_d, yg);
        const y_gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yg));

        // CPU replica of the same int8 pipeline, plus an f32 "truth" from the
        // rotated activations against the dequantized weight.
        const xr = try arena.dupe(f32, xf);
        const xi8 = try arena.alloc(i8, m * cols);
        const ascale = try arena.alloc(f32, m);
        for (0..m) |i| {
            convrot.rotate(xr[i * cols ..][0..cols]);
            var amax: f32 = 0;
            for (xr[i * cols ..][0..cols]) |v| amax = @max(amax, @abs(v));
            const s = @max(amax / 127.0, 1e-12);
            ascale[i] = s;
            for (0..cols) |k| {
                var qi: i32 = @intFromFloat(@round(xr[i * cols + k] / s));
                qi = @max(@as(i32, -127), @min(@as(i32, 127), qi));
                xi8[i * cols + k] = @intCast(qi);
            }
        }
        var num_sim: f64 = 0;
        var num_truth: f64 = 0;
        var den: f64 = 0;
        for (0..m) |i| {
            for (0..rows) |j| {
                var acc: i32 = 0;
                var truth: f64 = 0;
                for (0..cols) |k| {
                    acc += @as(i32, xi8[i * cols + k]) * @as(i32, @as(i8, @bitCast(wb[j * cols + k])));
                    truth += @as(f64, xr[i * cols + k]) * (@as(f64, @floatFromInt(@as(i8, @bitCast(wb[j * cols + k])))) * wscale[j]);
                }
                const sim: f64 = @as(f64, @floatFromInt(acc)) * ascale[i] * wscale[j];
                const g: f64 = y_gpu[i * rows + j];
                num_sim += (g - sim) * (g - sim);
                num_truth += (g - truth) * (g - truth);
                den += truth * truth;
            }
        }
        const rel_sim = @sqrt(num_sim / den);
        const rel_truth = @sqrt(num_truth / den);
        try stdout.print("i8 linear {d}x{d}x{d}: rel vs cpu-sim {d:.5} (wiring), rel vs f32 {d:.4} (int8 accuracy)\n", .{ m, rows, cols, rel_sim, rel_truth });
        try stdout.flush();
        // Stage B (cols>=6144) rotates in f16 shared, so GPU-vs-f32-CPU-replica
        // diverges ~0.4% — but that f16 rotation error stays WITHIN int8 quant
        // noise (rel-vs-f32 is unchanged, ~0.9%), so the real accuracy gate is
        // rel_truth. Small cols use the f32 register path (bit-close, 1e-3).
        const sim_gate: f64 = if (cols >= 6144) 1e-2 else 1e-3;
        if (rel_sim > sim_gate or rel_truth > 0.03) return error.GpuMismatch;
    }
    // --- Stage A: fused-rescale shared GEMM (y = s32 * act[row] * weight[col]) ---
    if (ctx.pipe_coop_i8_fs != .null_handle) {
        const FCase = struct { m: usize, rows: usize, cols: usize };
        const fcases = [_]FCase{ .{ .m = 128, .rows = 128, .cols = 64 }, .{ .m = 256, .rows = 384, .cols = 320 } };
        for (fcases) |c| {
            const m = c.m;
            const rows = c.rows;
            const cols = c.cols;
            const xb = try arena.alloc(u8, m * cols); // s8 activations [m][cols]
            for (xb) |*v| v.* = @bitCast(rand.int(i8));
            const wb = try arena.alloc(u8, rows * cols); // s8 weights [rows][cols]
            for (wb) |*v| v.* = @bitCast(rand.int(i8));
            const ascale = try arena.alloc(f32, m);
            for (ascale) |*s| s.* = 0.001 + rand.float(f32) * 0.02;
            const wscale = try arena.alloc(f32, rows);
            for (wscale) |*s| s.* = 0.001 + rand.float(f32) * 0.02;
            // scale buffer = [act(m_pad) | weight(rows)]; m already %128 here.
            const scat = try arena.alloc(f32, m + rows);
            @memcpy(scat[0..m], ascale);
            @memcpy(scat[m..], wscale);

            var x_d = try ctx.tensorCreate(m * cols);
            defer ctx.tensorDestroy(&x_d);
            var y_d = try ctx.tensorCreate(m * rows * 4);
            defer ctx.tensorDestroy(&y_d);
            var s_d = try ctx.tensorCreate((m + rows) * 4);
            defer ctx.tensorDestroy(&s_d);
            try ctx.tensorUpload(x_d, xb);
            try ctx.tensorUpload(s_d, std.mem.sliceAsBytes(scat));

            const ok = try ctx.opMatmulCoopI8Fused(y_d, x_d, m, wb, rows, cols, s_d.buf, false);
            if (!ok) {
                try stdout.print("fused i8 {d}x{d}x{d}: not dispatched (shape/pipe)\n", .{ m, rows, cols });
                continue;
            }
            const yg = try arena.alloc(u8, m * rows * 4);
            try ctx.tensorDownload(y_d, yg);
            const y_gpu: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, yg));
            var max_rel: f64 = 0;
            for (0..m) |i| {
                for (0..rows) |j| {
                    var acc: i32 = 0;
                    for (0..cols) |k| acc += @as(i32, @as(i8, @bitCast(xb[i * cols + k]))) * @as(i32, @as(i8, @bitCast(wb[j * cols + k])));
                    const want: f64 = @as(f64, @floatFromInt(acc)) * ascale[i] * wscale[j];
                    const got: f64 = y_gpu[i * rows + j];
                    const rel = @abs(got - want) / (@abs(want) + 1e-9);
                    max_rel = @max(max_rel, rel);
                }
            }
            try stdout.print("fused i8 {d}x{d}x{d}: max rel err vs cpu {d:.6}\n", .{ m, rows, cols, max_rel });
            if (max_rel > 1e-4) return error.GpuMismatch;
        }
    }

    try stdout.print("int8 coop GEMM + linear OK\n", .{});
}

/// CUDA backend bring-up: load the driver, report device caps (incl. the
/// >48 KB opt-in shared the Vulkan path cannot reach), and run the PTX vadd
/// smoke test end to end.
fn cudaTest(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var ctx = cuda.Context.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("cuda device: {s} (sm_{d}{d}), {d} SMs, opt-in shared {d} KB/block, {d} KB/SM, {d} MHz\n", .{
        ctx.deviceName(),   ctx.cc_major,                    ctx.cc_minor,
        ctx.sm_count,       @divTrunc(ctx.shared_optin_max, 1024),
        @divTrunc(ctx.shared_per_sm, 1024), @divTrunc(ctx.clock_khz, 1000),
    });
    try cuda.kernels.smokeTest(&ctx);
    try stdout.print("cuda vadd smoke test OK\n", .{});
}

/// Phase-2 library backend bring-up (`--backend cuda`): dlopen cuBLASLt + cuDNN,
/// create handles bound to the compute stream, report their versions. Validates
/// bindings→handle end to end (the analog of `cuda-test` for the driver path).
fn cudaLibsTest(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.initLibs(arena) catch |err| {
        try stdout.print("cuda libs unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    const L = be.libs.?;
    try stdout.print("cuda device: {s}\n", .{be.deviceName()});
    try stdout.print("cublasLt version: {d} (cudart {d})\n", .{ L.lt.cublasLtGetVersion(), L.lt.cublasLtGetCudartVersion() });
    try stdout.print("cuDNN version: {d}\n", .{L.dnn.cudnnGetVersion()});
    try stdout.print("cuda libs smoke test OK\n", .{});
}

/// Phase-2 milestone 2.1: cuBLASLt int8 IMMA GEMM. Bit-exact validation vs a CPU
/// integer matmul (s8·s8→s32 is exact, so 0 mismatches expected), then a min-of-N
/// TOP/s benchmark at the DiT GEMM shapes to compare against the hand-PTX
/// `igemm_pipe` (~135 TOP/s) and the Vulkan coopmat (~85).
fn cudaLibsI8Test(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.initLibs(arena) catch |err| {
        try stdout.print("cuda libs unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("cuda device: {s}, cublasLt {d}\n", .{ be.deviceName(), be.libs.?.lt.cublasLtGetVersion() });

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();

    // ---- bit-exact validation vs a CPU integer matmul ----
    const Case = struct { m: usize, n: usize, k: usize };
    const checks = [_]Case{ .{ .m = 256, .n = 512, .k = 6144 }, .{ .m = 128, .n = 256, .k = 16384 } };
    for (checks) |c| {
        const a = try arena.alloc(i8, c.m * c.k);
        const w = try arena.alloc(i8, c.n * c.k);
        defer arena.free(a);
        defer arena.free(w);
        for (a) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
        for (w) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
        var da = try be.tensorCreate(c.m * c.k);
        defer be.tensorDestroy(&da);
        var dw = try be.tensorCreate(c.n * c.k);
        defer be.tensorDestroy(&dw);
        var dacc = try be.tensorCreate(c.m * c.n * 4);
        defer be.tensorDestroy(&dacc);
        try be.tensorUpload(da, std.mem.sliceAsBytes(a));
        try be.tensorUpload(dw, std.mem.sliceAsBytes(w));
        try be.ltMatmulI8(dacc, dw, da, c.n, c.m, c.k);
        const acc = try arena.alloc(i32, c.m * c.n);
        defer arena.free(acc);
        try be.tensorDownload(dacc, std.mem.sliceAsBytes(acc));
        var mism: usize = 0;
        for (0..c.m) |i| {
            for (0..c.n) |j| {
                var s: i32 = 0;
                for (0..c.k) |kk| s += @as(i32, a[i * c.k + kk]) * @as(i32, w[j * c.k + kk]);
                if (s != acc[i * c.n + j]) mism += 1;
            }
        }
        try stdout.print("i8 gemm {d}x{d}x{d}: {d} mismatches vs cpu s32\n", .{ c.m, c.n, c.k, mism });
        if (mism != 0) return error.GpuMismatch;
    }

    // ---- min-of-N TOP/s at the DiT GEMM shapes ----
    const Shape = struct { m: usize, n: usize, k: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .m = 4224, .n = 6144, .k = 6144, .name = "square  " },
        .{ .m = 7680, .n = 6144, .k = 6144, .name = "qkv     " },
        .{ .m = 7680, .n = 16384, .k = 6144, .name = "mlp up  " },
        .{ .m = 7680, .n = 6144, .k = 16384, .name = "mlp down" },
    };
    for (shapes) |s| {
        var da = try be.tensorCreate(s.m * s.k);
        defer be.tensorDestroy(&da);
        var dw = try be.tensorCreate(s.n * s.k);
        defer be.tensorDestroy(&dw);
        var dacc = try be.tensorCreate(s.m * s.n * 4);
        defer be.tensorDestroy(&dacc);
        const timer = try be.ctx.timerCreate();
        defer be.ctx.timerDestroy(timer);
        try be.ltMatmulI8(dacc, dw, da, s.n, s.m, s.k); // warm: builds+caches the plan
        var best: f32 = std.math.floatMax(f32);
        for (0..12) |_| {
            try be.ctx.timerBegin(timer);
            try be.ltMatmulI8(dacc, dw, da, s.n, s.m, s.k);
            const ms = try be.ctx.timerEndMs(timer);
            best = @min(best, ms);
        }
        const macs: f64 = @floatFromInt(s.m * s.n * s.k);
        const tops = 2.0 * macs / (@as(f64, best) / 1000.0) / 1e12;
        try stdout.print("i8 gemm {s} {d}x{d}x{d}: {d:.3} ms, {d:.1} TOP/s\n", .{ s.name, s.m, s.n, s.k, best, tops });
    }
}

/// Phase-2 milestone 2.2: cuBLASLt f16 GEMM (HMMA, f32 accumulate) — the drop-in
/// for the hand-PTX `buildHgemm` behind the fp8 encoder GEMMs and the VAE convs.
/// Validates D[m][n] f32 = A[m][k] @ W[n][k]ᵀ vs a CPU f32-accumulate reference
/// (f16 inputs widen exactly; only the reduction order differs → f16 regime),
/// then a min-of-N TFLOP/s bench.
fn cudaLibsF16Test(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.initLibs(arena) catch |err| {
        try stdout.print("cuda libs unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("cuda device: {s}, cublasLt {d}\n", .{ be.deviceName(), be.libs.?.lt.cublasLtGetVersion() });

    var prng = std.Random.DefaultPrng.init(0xF16F16);
    const rnd = prng.random();

    // ---- validation vs CPU f32-accumulate reference ----
    const Case = struct { m: usize, n: usize, k: usize };
    const checks = [_]Case{ .{ .m = 256, .n = 512, .k = 2560 }, .{ .m = 128, .n = 256, .k = 6144 } };
    for (checks) |c| {
        const a = try arena.alloc(f16, c.m * c.k);
        const w = try arena.alloc(f16, c.n * c.k);
        defer arena.free(a);
        defer arena.free(w);
        for (a) |*v| v.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);
        for (w) |*v| v.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);
        var da = try be.tensorCreate(c.m * c.k * 2);
        defer be.tensorDestroy(&da);
        var dw = try be.tensorCreate(c.n * c.k * 2);
        defer be.tensorDestroy(&dw);
        var dd = try be.tensorCreate(c.m * c.n * 4);
        defer be.tensorDestroy(&dd);
        try be.tensorUpload(da, std.mem.sliceAsBytes(a));
        try be.tensorUpload(dw, std.mem.sliceAsBytes(w));
        try be.ltMatmulF16(dd, dw, da, c.n, c.m, c.k);
        const d = try arena.alloc(f32, c.m * c.n);
        defer arena.free(d);
        try be.tensorDownload(dd, std.mem.sliceAsBytes(d));
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..c.m) |i| {
            for (0..c.n) |j| {
                var s: f32 = 0;
                for (0..c.k) |kk| s += @as(f32, a[i * c.k + kk]) * @as(f32, w[j * c.k + kk]);
                const diff = @as(f64, d[i * c.n + j]) - @as(f64, s);
                num += diff * diff;
                den += @as(f64, s) * @as(f64, s);
            }
        }
        const rel = @sqrt(num / den);
        try stdout.print("f16 gemm {d}x{d}x{d}: rel vs cpu-f32 {d:.6}\n", .{ c.m, c.n, c.k, rel });
        if (rel > 5e-3) return error.GpuMismatch;
    }

    // ---- min-of-N TFLOP/s ----
    const Shape = struct { m: usize, n: usize, k: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .m = 4224, .n = 6144, .k = 6144, .name = "square    " },
        .{ .m = 448, .n = 9728, .k = 2560, .name = "enc mlp   " },
        .{ .m = 5376, .n = 384, .k = 3456, .name = "vae conv  " },
    };
    for (shapes) |s| {
        var da = try be.tensorCreate(s.m * s.k * 2);
        defer be.tensorDestroy(&da);
        var dw = try be.tensorCreate(s.n * s.k * 2);
        defer be.tensorDestroy(&dw);
        var dd = try be.tensorCreate(s.m * s.n * 4);
        defer be.tensorDestroy(&dd);
        const timer = try be.ctx.timerCreate();
        defer be.ctx.timerDestroy(timer);
        try be.ltMatmulF16(dd, dw, da, s.n, s.m, s.k); // warm
        var best: f32 = std.math.floatMax(f32);
        for (0..12) |_| {
            try be.ctx.timerBegin(timer);
            try be.ltMatmulF16(dd, dw, da, s.n, s.m, s.k);
            const ms = try be.ctx.timerEndMs(timer);
            best = @min(best, ms);
        }
        const flops: f64 = @floatFromInt(s.m * s.n * s.k);
        const tflops = 2.0 * flops / (@as(f64, best) / 1000.0) / 1e12;
        try stdout.print("f16 gemm {s} {d}x{d}x{d}: {d:.3} ms, {d:.1} TFLOP/s\n", .{ s.name, s.m, s.n, s.k, best, tflops });
    }
}

/// Phase-2 milestone 2.7: cuDNN fused int8 GEMM + per-row×per-col dequant (one
/// op graph). Validates D = (A·B in s32)·act_scale·weight_scale vs a CPU
/// reference (s32 matmul exact; dequant in f32), then a min-of-N bench vs the
/// separate ltMatmulI8 + irescale path.
fn cudaLibsI8FusedTest(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    const cudnn = cuda.cudnn;
    var be = cuda.Backend.initLibs(arena) catch |err| {
        try stdout.print("cuda libs unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    const api = &be.libs.?.dnn;
    const handle = be.libs.?.dnn_handle;
    try stdout.print("cuda device: {s}, cuDNN {d}\n", .{ be.deviceName(), api.cudnnGetVersion() });

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{ .{ .m = 256, .n = 512, .k = 6144 }, .{ .m = 128, .n = 256, .k = 16384 } };
    var prng = std.Random.DefaultPrng.init(0x1F00D);
    const rnd = prng.random();
    for (cases) |c| {
        const a = try arena.alloc(i8, c.m * c.k);
        const w = try arena.alloc(i8, c.n * c.k);
        const asc = try arena.alloc(f32, c.m);
        const wsc = try arena.alloc(f32, c.n);
        defer arena.free(a);
        defer arena.free(w);
        defer arena.free(asc);
        defer arena.free(wsc);
        for (a) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
        for (w) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
        for (asc) |*v| v.* = 0.001 + rnd.float(f32) * 0.01;
        for (wsc) |*v| v.* = 0.001 + rnd.float(f32) * 0.01;

        var da = try be.tensorCreate(a.len);
        defer be.tensorDestroy(&da);
        var dw = try be.tensorCreate(w.len);
        defer be.tensorDestroy(&dw);
        var das = try be.tensorCreate(asc.len * 4);
        defer be.tensorDestroy(&das);
        var dws = try be.tensorCreate(wsc.len * 4);
        defer be.tensorDestroy(&dws);
        var dd = try be.tensorCreate(c.m * c.n * 4);
        defer be.tensorDestroy(&dd);
        try be.tensorUpload(da, std.mem.sliceAsBytes(a));
        try be.tensorUpload(dw, std.mem.sliceAsBytes(w));
        try be.tensorUpload(das, std.mem.sliceAsBytes(asc));
        try be.tensorUpload(dws, std.mem.sliceAsBytes(wsc));

        var plan = cudnn.MatmulDequantPlan.build(api, handle, c.m, c.n, c.k, false) catch |err| {
            try stdout.print("fused build failed ({t}) for {d}x{d}x{d}\n", .{ err, c.m, c.n, c.k });
            return;
        };
        defer plan.deinit(api);
        var ws: cuda.backend.DeviceBuffer = .{};
        if (plan.workspace_bytes > 0) ws = try be.tensorCreate(plan.workspace_bytes);
        defer be.tensorDestroy(&ws);
        try plan.execute(api, handle, da.ptr(), dw.ptr(), das.ptr(), dws.ptr(), dd.ptr(), ws.ptr());
        try be.ctx.synchronize();

        const d = try arena.alloc(f32, c.m * c.n);
        defer arena.free(d);
        try be.tensorDownload(dd, std.mem.sliceAsBytes(d));
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..c.m) |i| {
            for (0..c.n) |j| {
                var s: i32 = 0;
                for (0..c.k) |kk| s += @as(i32, a[i * c.k + kk]) * @as(i32, w[j * c.k + kk]);
                const ref = @as(f32, @floatFromInt(s)) * asc[i] * wsc[j];
                const diff = @as(f64, d[i * c.n + j]) - @as(f64, ref);
                num += diff * diff;
                den += @as(f64, ref) * @as(f64, ref);
            }
        }
        const rel = @sqrt(num / den);
        try stdout.print("fused i8+dequant {d}x{d}x{d}: rel vs cpu {d:.6} (ws {d} B)\n", .{ c.m, c.n, c.k, rel, plan.workspace_bytes });
        if (rel > 1e-3) return error.GpuMismatch;
    }
    try stdout.print("cuda libs fused int8+dequant test OK\n", .{});
}

/// Phase-2 milestone 2.4: cuDNN fused SDPA (flash attention). Validates the
/// backend-graph SDPA op in ISOLATION — synthetic f16 Q/K/V (GQA), non-causal —
/// vs a CPU softmax-attention reference, before any DiT wiring. Tensors stored
/// [s,h,d] (the DiT layout). Scale = 1/sqrt(d).
fn cudaLibsAttnTest(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    const cudnn = cuda.cudnn;
    var be = cuda.Backend.initLibs(arena) catch |err| {
        try stdout.print("cuda libs unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    const api = &be.libs.?.dnn;
    const handle = be.libs.?.dnn_handle;
    try stdout.print("cuda device: {s}, cuDNN {d}\n", .{ be.deviceName(), api.cudnnGetVersion() });

    const Case = struct { hq: usize, hkv: usize, s: usize, d: usize };
    const cases = [_]Case{
        .{ .hq = 4, .hkv = 4, .s = 256, .d = 128 }, // MHA baseline
        .{ .hq = 8, .hkv = 2, .s = 256, .d = 128 }, // GQA (group 4, the DiT ratio)
    };
    var prng = std.Random.DefaultPrng.init(0x5D9A);
    const rnd = prng.random();
    for (cases) |c| {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.d)));
        const group = c.hq / c.hkv;
        const q = try arena.alloc(f16, c.s * c.hq * c.d);
        const k = try arena.alloc(f16, c.s * c.hkv * c.d);
        const v = try arena.alloc(f16, c.s * c.hkv * c.d);
        defer arena.free(q);
        defer arena.free(k);
        defer arena.free(v);
        for (q) |*x| x.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);
        for (k) |*x| x.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);
        for (v) |*x| x.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);

        var dq = try be.tensorCreate(q.len * 2);
        defer be.tensorDestroy(&dq);
        var dk = try be.tensorCreate(k.len * 2);
        defer be.tensorDestroy(&dk);
        var dv = try be.tensorCreate(v.len * 2);
        defer be.tensorDestroy(&dv);
        var do_ = try be.tensorCreate(c.s * c.hq * c.d * 2);
        defer be.tensorDestroy(&do_);
        try be.tensorUpload(dq, std.mem.sliceAsBytes(q));
        try be.tensorUpload(dk, std.mem.sliceAsBytes(k));
        try be.tensorUpload(dv, std.mem.sliceAsBytes(v));

        var plan = cudnn.SdpaPlan.build(api, handle, 1, c.hq, c.hkv, c.s, c.d) catch |err| {
            try stdout.print("SDPA build failed ({t}) for hq={d} hkv={d} s={d} d={d}\n", .{ err, c.hq, c.hkv, c.s, c.d });
            return;
        };
        defer plan.deinit(api);
        var ws: cuda.backend.DeviceBuffer = .{};
        if (plan.workspace_bytes > 0) ws = try be.tensorCreate(plan.workspace_bytes);
        defer be.tensorDestroy(&ws);
        var sc = scale;
        try plan.execute(api, handle, dq.ptr(), dk.ptr(), dv.ptr(), do_.ptr(), &sc, ws.ptr());
        try be.ctx.synchronize();

        const o = try arena.alloc(f16, c.s * c.hq * c.d);
        defer arena.free(o);
        try be.tensorDownload(do_, std.mem.sliceAsBytes(o));

        // CPU reference: full softmax attention (f32), GQA via kv = h/group.
        const row = try arena.alloc(f32, c.s);
        defer arena.free(row);
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..c.hq) |h| {
            const kv = h / group;
            for (0..c.s) |i| {
                var mx: f32 = -std.math.inf(f32);
                for (0..c.s) |j| {
                    var dot: f32 = 0;
                    for (0..c.d) |dd| dot += @as(f32, q[i * c.hq * c.d + h * c.d + dd]) * @as(f32, k[j * c.hkv * c.d + kv * c.d + dd]);
                    row[j] = dot * scale;
                    mx = @max(mx, row[j]);
                }
                var sum: f32 = 0;
                for (0..c.s) |j| {
                    row[j] = @exp(row[j] - mx);
                    sum += row[j];
                }
                for (0..c.d) |dd| {
                    var acc: f32 = 0;
                    for (0..c.s) |j| acc += row[j] * @as(f32, v[j * c.hkv * c.d + kv * c.d + dd]);
                    const ref = acc / sum;
                    const got = @as(f32, o[i * c.hq * c.d + h * c.d + dd]);
                    const diff = @as(f64, got) - @as(f64, ref);
                    num += diff * diff;
                    den += @as(f64, ref) * @as(f64, ref);
                }
            }
        }
        const rel = @sqrt(num / den);
        try stdout.print("sdpa hq={d} hkv={d} s={d} d={d}: rel vs cpu {d:.6} (ws {d} B)\n", .{ c.hq, c.hkv, c.s, c.d, rel, plan.workspace_bytes });
        if (rel > 2e-2) return error.GpuMismatch;
    }

    // ---- min-of-N timing at DiT attention shapes (GQA 48/12, hd 128) ----
    // Compare against the hand-PTX two-pass attention (2.3 profile @1024px:
    // scores 263 + softmax 55 + pv 163 = ~481 ms; @1408px ~1.6 s).
    const TShape = struct { s: usize, name: []const u8 };
    const tshapes = [_]TShape{ .{ .s = 4104, .name = "1024px" }, .{ .s = 7752, .name = "1408px" } };
    for (tshapes) |t| {
        const hq: usize = 48;
        const hkv: usize = 12;
        const d: usize = 128;
        var dq = try be.tensorCreate(t.s * hq * d * 2);
        defer be.tensorDestroy(&dq);
        var dk = try be.tensorCreate(t.s * hkv * d * 2);
        defer be.tensorDestroy(&dk);
        var dv = try be.tensorCreate(t.s * hkv * d * 2);
        defer be.tensorDestroy(&dv);
        var do2 = try be.tensorCreate(t.s * hq * d * 2);
        defer be.tensorDestroy(&do2);
        var plan = try cudnn.SdpaPlan.build(api, handle, 1, hq, hkv, t.s, d);
        defer plan.deinit(api);
        var ws: cuda.backend.DeviceBuffer = .{};
        if (plan.workspace_bytes > 0) ws = try be.tensorCreate(plan.workspace_bytes);
        defer be.tensorDestroy(&ws);
        var sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const timer = try be.ctx.timerCreate();
        defer be.ctx.timerDestroy(timer);
        try plan.execute(api, handle, dq.ptr(), dk.ptr(), dv.ptr(), do2.ptr(), &sc, ws.ptr()); // warm
        var best: f32 = std.math.floatMax(f32);
        for (0..12) |_| {
            try be.ctx.timerBegin(timer);
            try plan.execute(api, handle, dq.ptr(), dk.ptr(), dv.ptr(), do2.ptr(), &sc, ws.ptr());
            const ms = try be.ctx.timerEndMs(timer);
            best = @min(best, ms);
        }
        try stdout.print("sdpa {s} (s={d}, 48/12 hd128): {d:.3} ms  (hand-PTX ~{s})\n", .{ t.name, t.s, best, if (t.s == 4104) "481 ms" else "1600 ms" });
    }
    try stdout.print("cuda libs SDPA test OK\n", .{});
}

/// CUDA int8 GEMM validation + benchmark against the Vulkan ~85 TOPS baseline.
/// Uses the same CPU oracle as `gpu-i8-test`. (Filled in as the hand-PTX GEMM
/// lands; for now reports device caps so the harness path is wired.)
fn cudaI8Test(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var ctx = cuda.Context.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("cuda device: {s} (sm_{d}{d})\n", .{ ctx.deviceName(), ctx.cc_major, ctx.cc_minor });
    try cuda.kernels.i8GemmTest(&ctx, io, stdout);
    try cuda.kernels.i8LinearTest(&ctx, io, stdout);
}

/// int4 (W4A4) tensor-core validation, staged like `cuda-i8-test`: raw s4*s4
/// GEMM against a CPU oracle first, then (as they land) the full convrot linear.
fn cudaI4Test(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var ctx = cuda.Context.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("cuda device: {s} (sm_{d}{d})\n", .{ ctx.deviceName(), ctx.cc_major, ctx.cc_minor });
    try cuda.kernels.i4GemmTest(&ctx, io, stdout);
    try cuda.kernels.i4LinearTest(&ctx, io, stdout);
}

/// Validate the CUDA DiT forward against the CPU int8 forward on the same
/// (random) inputs: proves the hand-PTX backend computes a like-for-like DiT
/// velocity. Requires the int8 convrot checkpoint.
/// Compare the batched tensor-core attention (opAttnTCBatched) against the
/// validated per-head loop (opAttnTCLoop) on identical random Q/K/V, at a
/// no-padding size (seq=256) and a padded one (seq=264). Both should match a CPU
/// GQA attention reference. Pinpoints the batched-path bug.
/// Validate the CUDA fp8-e4m3 GEMM (opMatmulFp8) against a CPU reference that
/// dequantizes the same fp8 bytes through the e4m3 LUT: y[m][rows] = x @ Wᵀ.
fn cudaFp8Test(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("cuda device: {s}\n", .{be.deviceName()});
    const cases = [_][3]usize{ .{ 130, 512, 256 }, .{ 264, 2560, 4096 }, .{ 8, 1024, 2560 } };
    for (cases) |c| {
        const m = c[0];
        const rows = c[1];
        const cols = c[2];
        const mpad = std.mem.alignForward(usize, m, 128);
        var prng = std.Random.DefaultPrng.init(99);
        const rnd = prng.random();
        const x = try arena.alloc(f32, m * cols);
        for (x) |*v| v.* = rnd.floatNorm(f32) * 0.5;
        const w = try arena.alloc(u8, rows * cols);
        for (w) |*b| {
            b.* = rnd.int(u8);
            if (b.* == 0x7f) b.* = 0; // avoid the two e4m3 NaN encodings
            if (b.* == 0xff) b.* = 0x80;
        }
        const scale: f32 = 0.35;

        var xd = try be.tensorCreate(x.len * 4);
        defer be.tensorDestroy(&xd);
        try be.tensorUpload(xd, std.mem.sliceAsBytes(x));
        var yd = try be.tensorCreate(mpad * rows * 4);
        defer be.tensorDestroy(&yd);
        try be.opMatmulFp8(yd, xd, m, w, scale, rows, cols);
        try be.endBatch();
        const y = try arena.alloc(f32, mpad * rows);
        try be.tensorDownload(yd, std.mem.sliceAsBytes(y));

        var num: f64 = 0;
        var den: f64 = 0;
        for (0..m) |i| {
            for (0..rows) |r| {
                var acc: f64 = 0;
                for (0..cols) |cc| {
                    const wv = TensorPencil.dtype.f8e4m3ToF32(w[r * cols + cc]) * scale;
                    acc += @as(f64, x[i * cols + cc]) * wv;
                }
                const d = @as(f64, y[i * rows + r]) - acc;
                num += d * d;
                den += acc * acc;
            }
        }
        try stdout.print("fp8 gemm m={d} rows={d} cols={d}: rel RMSE {d:.5}\n", .{ m, rows, cols, @sqrt(num / den) });
        try stdout.flush();
    }
}

/// Validate the CUDA text encoder (qwen3_cuda) against the CPU encode on the
/// same prompt: proves the fp8 GEMM + rope_half + causal attention + norm chain
/// produce a like-for-like Krea 2 conditioning stack.
/// Validate the CUDA SD-family kernels against the CPU ops they reproduce, and
/// then both whole stages against their CPU forwards.
///
/// This exists because the hand-PTX kernels cannot have unit tests the way their
/// SPIR-V twins do (`sd_unet_gpu`'s test block): the test binary brings up no
/// CUDA context. Without it the CUDA path's only evidence is that a whole render
/// lands within 0.3 dB of the Vulkan arm — real, but it localizes nothing.
///
/// Every check is against the CPU op, on random data, and the command exits with
/// an error if any fails, so it is usable as a gate.
fn sdCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const ops = TensorPencil.ops;
    const sd_unet = TensorPencil.models.sd_unet;
    const sd_unet_cuda = TensorPencil.models.sd_unet_cuda;
    const sd_vae = TensorPencil.models.sd_vae;
    const sd_vae_cuda = TensorPencil.models.sd_vae_cuda;
    const Buf = cuda.backend.DeviceBuffer;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== sd-cuda-test ==\ncuda device: {s} (kernels: {t})\n\n", .{ be.deviceName(), be.kernels });

    var prng = std.Random.DefaultPrng.init(4242);
    const rnd = prng.random();
    var failures: usize = 0;

    // Report one check. `tol` is on relative L2 unless the check is exact.
    const R = struct {
        out: *Io.Writer,
        fails: *usize,
        fn check(self: @This(), name: []const u8, want: []const f32, got: []const f32, tol: f64) !void {
            var num: f64 = 0;
            var den: f64 = 0;
            var max_err: f64 = 0;
            var nan = false;
            // Which SIDE went non-finite, and how big each side's values get. A bare
            // `rel L2 nan` says a check failed but not whether the reference or the
            // kernel produced it — and "the reference overflowed" is a completely
            // different bug from "the kernel did".
            var bad_want: usize = 0;
            var bad_got: usize = 0;
            var mag_want: f64 = 0;
            var mag_got: f64 = 0;
            for (want, got) |e, a| {
                if (!std.math.isFinite(e)) bad_want += 1 else mag_want = @max(mag_want, @abs(@as(f64, e)));
                if (!std.math.isFinite(a)) bad_got += 1 else mag_got = @max(mag_got, @abs(@as(f64, a)));
                if (std.math.isNan(a)) nan = true;
                const d = @as(f64, e) - @as(f64, a);
                num += d * d;
                den += @as(f64, e) * @as(f64, e);
                max_err = @max(max_err, @abs(d));
            }
            const rel = if (den == 0) @sqrt(num) else @sqrt(num / den);
            const ok = !nan and bad_want == 0 and bad_got == 0 and rel <= tol;
            if (!ok) self.fails.* += 1;
            try self.out.print("{s:<44} rel L2 {d:.7}  max {d:.6}  {s}\n", .{
                name, rel, max_err, if (ok) "ok" else "FAIL",
            });
            if (bad_want != 0 or bad_got != 0) {
                try self.out.print("{s:<44}   non-finite: want {d}/{d} (max |v| {d:.1}), got {d}/{d} (max |v| {d:.1})\n", .{
                    "", bad_want, want.len, mag_want, bad_got, got.len, mag_got,
                });
            }
            // Flush per check: the whole-stage sweep takes minutes and can be
            // killed by the host OOM killer mid-run, and a buffered report loses
            // every line it had already produced — i.e. exactly the diagnostic.
            try self.out.flush();
        }
    };
    const rep: R = .{ .out = stdout, .fails = &failures };

    var a_d: Buf = .{};
    var b_d: Buf = .{};
    var c_d: Buf = .{};
    var d_d: Buf = .{};
    var e_d: Buf = .{};
    defer inline for (.{ &a_d, &b_d, &c_d, &d_d, &e_d }) |x| be.tensorDestroy(x);

    const up = struct {
        fn go(bk: *cuda.Backend, buf: *Buf, host: []const f32) !void {
            try bk.ensureDeviceBuffer(buf, host.len * 4);
            try bk.tensorUpload(buf.*, std.mem.sliceAsBytes(host));
        }
    }.go;

    // --- GroupNorm -----------------------------------------------------------
    // Two regimes: mean 0 is what a UNet activation looks like; mean 400 is the
    // one `ops.norm.groupNorm` warns about (a late VAE block), where the shifted
    // sum-of-squares form dies and Welford does not. The looser bound there is
    // f32's representation of `x` itself cancelling to a quantity of size 1.
    {
        const groups = 32;
        const n = 130; // not a multiple of the 256 statistics chunks
        const ch = 320;
        const x = try arena.alloc(f32, n * ch);
        const cat = try arena.alloc(f32, 2 * ch);
        for (cat) |*v| v.* = rnd.floatNorm(f32);
        const want = try arena.alloc(f32, n * ch);
        const got = try arena.alloc(f32, n * ch);
        try be.ensureDeviceBuffer(&d_d, groups * 256 * 3 * 4);
        try be.ensureDeviceBuffer(&e_d, groups * 2 * 4);
        try up(be, &c_d, cat);
        for ([2]f32{ 0, 400 }) |mean| {
            for (x) |*v| v.* = mean + rnd.floatNorm(f32);
            try up(be, &a_d, x);
            try be.ensureDeviceBuffer(&b_d, n * ch * 4);
            for ([2]bool{ false, true }) |silu| {
                try be.opGroupNorm(a_d, b_d, c_d, d_d, e_d, n, ch, groups, 256, 1e-5, silu, false);
                try be.tensorDownload(b_d, std.mem.sliceAsBytes(got));
                ops.norm.groupNorm(want, x, ch, groups, cat[0..ch], cat[ch..], 1e-5);
                if (silu) ops.act.silu(want);
                var name_buf: [64]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "group_norm mean={d:.0} silu={}", .{ mean, silu });
                try rep.check(name, want, got, if (mean == 0) 2e-6 else 2e-4);
            }
        }
    }

    // --- GEGLU ---------------------------------------------------------------
    // The halves are (value, gate) in that order and the gate takes the ERF gelu;
    // swapping either is a silent quality loss, so this compares element by
    // element rather than in aggregate.
    {
        const n = 37;
        const inner = 96;
        const src = try arena.alloc(f32, n * 2 * inner);
        for (src) |*v| v.* = rnd.floatNorm(f32) * 3;
        const want = try arena.alloc(f32, n * inner);
        for (0..n) |p| {
            const row = src[p * 2 * inner ..][0 .. 2 * inner];
            for (0..inner) |j| want[p * inner + j] = row[j] * ops.act.geluErfScalar(row[inner + j]);
        }
        try up(be, &a_d, src);
        try be.ensureDeviceBuffer(&b_d, n * inner * 4);
        try be.opGeglu(a_d, b_d, n, inner);
        const got = try arena.alloc(f32, n * inner);
        try be.tensorDownload(b_d, std.mem.sliceAsBytes(got));
        // 2e-5, not the 1e-6 the SPIR-V twin holds: every kernel in `cuda/elt.zig`
        // computes exp as `ex2.approx(x*log2e)` (its header says so), which is an
        // approximate instruction, where SPIR-V's `@exp` is not. Measured 6.6e-6 —
        // two orders below the f16 GEMM error either side of it, so the two
        // backends legitimately disagree here and neither is wrong.
        try rep.check("geglu (erf gate, value-first halves)", want, got, 2e-5);
    }

    // --- cross-attention (seq_kv != seq_q) -----------------------------------
    {
        const n = 100;
        const ctx_seq = 77; // the SD family's conditioning length
        const heads = 8;
        const hd = 40; // SD1.5's outermost level
        const dim = heads * hd;
        const q = try arena.alloc(f32, n * dim);
        const k = try arena.alloc(f32, ctx_seq * dim);
        const v = try arena.alloc(f32, ctx_seq * dim);
        for (q) |*x| x.* = rnd.floatNorm(f32);
        for (k) |*x| x.* = rnd.floatNorm(f32);
        for (v) |*x| x.* = rnd.floatNorm(f32);
        try up(be, &a_d, q);
        try up(be, &b_d, k);
        try up(be, &c_d, v);
        try be.ensureDeviceBuffer(&d_d, n * dim * 4);
        try be.opAttnCross(a_d, b_d, c_d, d_d, n, ctx_seq, heads, hd, 1.0 / @sqrt(@as(f32, hd)));
        const got = try arena.alloc(f32, n * dim);
        try be.tensorDownload(d_d, std.mem.sliceAsBytes(got));
        const want = try arena.alloc(f32, n * dim);
        try ops.attention.attention(io, arena, want, q, k, v, .{
            .seq_q = n,
            .seq_kv = ctx_seq,
            .n_heads = heads,
            .n_kv_heads = heads,
            .head_dim = hd,
        });
        try rep.check("attn_cross (q=100, kv=77, hd=40)", want, got, 1e-5);
    }

    // --- channel concat ------------------------------------------------------
    {
        const n = 17;
        const ch_a = 12;
        const ch_b = 20;
        const total = ch_a + ch_b;
        const av = try arena.alloc(f32, n * ch_a);
        const bv = try arena.alloc(f32, n * ch_b);
        for (av, 0..) |*x, i| x.* = @floatFromInt(i);
        for (bv, 0..) |*x, i| x.* = @floatFromInt(1000 + i);
        try up(be, &a_d, av);
        try up(be, &b_d, bv);
        try be.ensureDeviceBuffer(&c_d, n * total * 4);
        try be.opConcatCh(a_d, c_d, n, ch_a, total, 0);
        try be.opConcatCh(b_d, c_d, n, ch_b, total, ch_a);
        const got = try arena.alloc(f32, n * total);
        try be.tensorDownload(c_d, std.mem.sliceAsBytes(got));
        const want = try arena.alloc(f32, n * total);
        for (0..n) |p| {
            @memcpy(want[p * total ..][0..ch_a], av[p * ch_a ..][0..ch_a]);
            @memcpy(want[p * total + ch_a ..][0..ch_b], bv[p * ch_b ..][0..ch_b]);
        }
        try rep.check("concat_ch (exact)", want, got, 0);
    }

    // --- broadcast bias add --------------------------------------------------
    // How a ResBlock's timestep projection reaches the activation on this
    // backend; the offset argument is what lets one packed buffer serve them all.
    {
        const n = 23;
        const ch = 64;
        const off = 64;
        const x = try arena.alloc(f32, n * ch);
        const bias = try arena.alloc(f32, off + ch);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        for (bias) |*v| v.* = rnd.floatNorm(f32);
        try up(be, &a_d, x);
        try up(be, &b_d, bias);
        try be.opAddBiasRows(a_d, b_d, n, ch, off);
        const got = try arena.alloc(f32, n * ch);
        try be.tensorDownload(a_d, std.mem.sliceAsBytes(got));
        const want = try arena.alloc(f32, n * ch);
        for (0..n) |p| for (0..ch) |cc| {
            want[p * ch + cc] = x[p * ch + cc] + bias[off + cc];
        };
        try rep.check("add_bias_rows (offset into a packed buffer)", want, got, 1e-7);
    }

    // --- head pad / unpad round trip ----------------------------------------
    // The layout contract the tensor-core attention sits between. f16 storage, so
    // the bound is f16-scale; what matters is that no value moves.
    {
        const n = 300;
        const heads = 8;
        const hd = 40;
        const hp = 128;
        const dim = heads * hd;
        const seq_pad = std.mem.alignForward(usize, n, 128);
        const rows = seq_pad * heads * hp;
        const src = try arena.alloc(f32, n * dim);
        for (src) |*v| v.* = rnd.floatNorm(f32);
        try up(be, &a_d, src);
        try be.ensureDeviceBuffer(&b_d, rows * 2);
        try be.opHeadPadH16(a_d, b_d, n, seq_pad, heads, hd, hp, 1.0);
        const words = try arena.alloc(u32, rows / 2);
        try be.tensorDownload(b_d, std.mem.sliceAsBytes(words));
        const widened = try arena.alloc(f32, rows);
        for (words, 0..) |wd, i| {
            widened[i * 2] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(wd)))));
            widened[i * 2 + 1] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(wd >> 16)))));
        }
        // Every padding lane must be a hard zero, or the scores GEMM picks up
        // whatever the buffer held before.
        var pad_nonzero: usize = 0;
        for (0..seq_pad) |r| for (0..heads) |hh| for (hd..hp) |t| {
            if (widened[r * heads * hp + hh * hp + t] != 0) pad_nonzero += 1;
        };
        if (pad_nonzero != 0) failures += 1;
        try stdout.print("{s:<44} {d} non-zero lanes  {s}\n", .{
            "head_pad_h16 padding lanes", pad_nonzero, if (pad_nonzero == 0) "ok" else "FAIL",
        });
        try up(be, &c_d, widened);
        try be.ensureDeviceBuffer(&d_d, n * dim * 4);
        try be.opHeadUnpad(c_d, d_d, n, heads, hd, hp);
        const got = try arena.alloc(f32, n * dim);
        try be.tensorDownload(d_d, std.mem.sliceAsBytes(got));
        try rep.check("head_pad_h16 -> head_unpad round trip", src, got, 1e-3);
    }

    // --- self-attention at every SD head width -------------------------------
    // 40/80/160 are SD1.5's three attending levels, 64 is SDXL's. Under cuDNN all
    // four run at the true width; under hand-PTX the narrow ones pad to 128 and
    // 160 pads to 256, so this is where `attnHd` gets exercised.
    {
        const n = 300;
        const cases = [_]struct { hd: usize, heads: usize }{
            .{ .hd = 40, .heads = 8 },
            .{ .hd = 64, .heads = 10 },
            .{ .hd = 80, .heads = 8 },
            .{ .hd = 160, .heads = 8 },
        };
        for (cases) |cse| {
            const dim = cse.heads * cse.hd;
            const q = try arena.alloc(f32, n * dim);
            const k = try arena.alloc(f32, n * dim);
            const v = try arena.alloc(f32, n * dim);
            for (q) |*x| x.* = rnd.floatNorm(f32);
            for (k) |*x| x.* = rnd.floatNorm(f32);
            for (v) |*x| x.* = rnd.floatNorm(f32);

            var ws: sd_unet_cuda.Workspace = .{
                .gpa = arena,
                .skips = try arena.alloc(cuda.backend.DeviceBuffer, 0),
                .skip_ch = try arena.alloc(usize, 0),
            };
            defer ws.deinit(be);
            try up(be, &ws.q, q);
            try up(be, &ws.k, k);
            try up(be, &ws.v, v);
            try be.ensureDeviceBuffer(&ws.ao, n * dim * 4);
            const hp = sd_unet_cuda.attnHd(be, cse.hd);
            if (hp != cse.hd) {
                inline for (.{ "qp", "kp", "vp", "op" }) |f| {
                    try be.ensureDeviceBuffer(&@field(ws, f), n * cse.heads * hp * 4);
                }
            }
            try sd_unet_cuda.selfAttn(be, &ws, n, cse.heads, cse.hd, 1.0 / @sqrt(@as(f32, @floatFromInt(cse.hd))));
            const got = try arena.alloc(f32, n * dim);
            try be.tensorDownload(ws.ao, std.mem.sliceAsBytes(got));
            const want = try arena.alloc(f32, n * dim);
            try ops.attention.attention(io, arena, want, q, k, v, .{
                .seq_q = n,
                .seq_kv = n,
                .n_heads = cse.heads,
                .n_kv_heads = cse.heads,
                .head_dim = cse.hd,
            });
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "self attention hd={d} heads={d} (runs at {d})", .{ cse.hd, cse.heads, hp });
            // f16 tensor cores on a seq-length reduction.
            try rep.check(name, want, got, 5e-3);
        }
    }

    // --- the VAE mid-block's attention shape ---------------------------------
    // ONE head 512 wide over every latent position — nothing like the UNet's
    // 40..160-wide multi-head cases above, and the shape that makes `opAttnTC`
    // switch to its query-tiled (flash) path once the scores plane outgrows the
    // scratch budget. That switch is between seq 384 and seq 4096, i.e. between a
    // 24x16 latent and a 64² one, so the sizes here straddle it deliberately: the
    // whole-decoder check below passed at 12x10 for months while every 512-square
    // and larger SDXL decode came out as a solid white image.
    //
    // The magnitude arm matters for the same reason it does in the decoder sweep:
    // an online-softmax kernel's numerics depend on the score range, so a unit
    // gaussian is not a sufficient input.
    {
        const ch: usize = 512;
        const seqs = [_]usize{ 384, 4096, 9216 };
        const mags = [_]f32{ 1.0, 8.0 };
        for (seqs) |n| {
            for (mags) |mag| {
                var it_arena = std.heap.ArenaAllocator.init(arena);
                defer it_arena.deinit();
                const a = it_arena.allocator();
                const q = try a.alloc(f32, n * ch);
                const k = try a.alloc(f32, n * ch);
                const v = try a.alloc(f32, n * ch);
                for (q) |*x| x.* = rnd.floatNorm(f32) * mag;
                for (k) |*x| x.* = rnd.floatNorm(f32) * mag;
                for (v) |*x| x.* = rnd.floatNorm(f32) * mag;
                var bq: Buf = .{};
                var bk: Buf = .{};
                var bv: Buf = .{};
                var bo: Buf = .{};
                defer inline for (.{ &bq, &bk, &bv, &bo }) |b| be.tensorDestroy(b);
                try up(be, &bq, q);
                try up(be, &bk, k);
                try up(be, &bv, v);
                try be.ensureDeviceBuffer(&bo, n * ch * 4);
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
                try be.opAttnTC(bq, bk, bv, bo, n, 1, 1, ch, scale);
                const got = try a.alloc(f32, n * ch);
                try be.tensorDownload(bo, std.mem.sliceAsBytes(got));
                const want = try a.alloc(f32, n * ch);
                try ops.attention.attention(io, a, want, q, k, v, .{
                    .seq_q = n,
                    .seq_kv = n,
                    .n_heads = 1,
                    .n_kv_heads = 1,
                    .head_dim = ch,
                });
                var name_buf: [72]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "vae mid attention seq={d} hd=512 (x{d:.0})", .{ n, mag });
                // The x8 arm is looser BY MEASUREMENT, not to make it pass: scores
                // grow with |q|·|k|, so an f16 GEMM feeding a softmax lands at ~6e-3
                // relative at every seq length tested (1.0 holds 6e-4). What this arm
                // is for is catching `inf`/NaN, which is what the SDXL VAE actually
                // hit — a rel-L2 bound would report that as `nan`, not as 0.006.
                try rep.check(name, want, got, if (mag > 1.0) 1e-2 else 5e-3);
            }
        }
    }

    // --- convolution at all three sampling modes -----------------------------
    // Odd extents so stride 2's ceil(h/2) output size and the pad edges both
    // matter. The upsample case is compared against an explicitly doubled
    // tensor, which is exactly the equivalence the fused gather claims.
    {
        const h: usize = 11;
        const w: usize = 7;
        const ci: usize = 32;
        const co: usize = 128; // >= 96, so the tensor-core arm that ships
        const torch_w = try arena.alloc(f32, co * ci * 9);
        for (torch_w) |*x| x.* = rnd.floatNorm(f32) * 0.1;
        const bias = try arena.alloc(f32, co);
        for (bias) |*x| x.* = rnd.floatNorm(f32);
        const packed_w = try ops.conv.packWeight(arena, torch_w, co, ci, 3);
        const in = try arena.alloc(f32, h * w * ci);
        for (in) |*x| x.* = rnd.floatNorm(f32);
        try up(be, &a_d, in);
        var patch: Buf = .{};
        defer be.tensorDestroy(&patch);

        for ([3]sd_unet_cuda.SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
            const oh = switch (mode) {
                .stride1 => h,
                .stride2 => (h + 1) / 2,
                .upsample2x => 2 * h,
            };
            const ow = switch (mode) {
                .stride1 => w,
                .stride2 => (w + 1) / 2,
                .upsample2x => 2 * w,
            };
            const cv: TensorPencil.ops.conv.Conv2d = .{
                .w = packed_w,
                .b = bias,
                .co = co,
                .ci = ci,
                .k = 3,
                .stride = if (mode == .stride2) @as(usize, 2) else 1,
                .pad = 1,
            };
            try be.ensureDeviceBuffer(&b_d, oh * ow * co * 4);
            try sd_unet_cuda.convInto(be, &patch, &b_d, &a_d, h, w, cv, mode);
            const got = try arena.alloc(f32, oh * ow * co);
            try be.tensorDownload(b_d, std.mem.sliceAsBytes(got));
            const want = try arena.alloc(f32, oh * ow * co);
            if (mode == .upsample2x) {
                const expanded = try arena.alloc(f32, oh * ow * ci);
                for (0..oh) |y| for (0..ow) |xx| {
                    @memcpy(expanded[(y * ow + xx) * ci ..][0..ci], in[((y / 2) * w + (xx / 2)) * ci ..][0..ci]);
                };
                try ops.conv.conv2d(io, arena, want, expanded, oh, ow, cv);
            } else {
                try ops.conv.conv2d(io, arena, want, in, h, w, cv);
            }
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "conv 3x3 {t} (im2col_sd + f16 GEMM)", .{mode});
            try rep.check(name, want, got, 2e-3);
        }
    }

    // --- CLIP's two GELU forms -----------------------------------------------
    // Checked value by value (max, not just rel L2): the three GELU forms in this
    // codebase agree to ~1e-2, so an aggregate bound loose enough to pass f32 noise
    // would also pass the wrong kernel. quick-GELU is CLIP-L's, erf-GELU is CLIP-G's,
    // and swapping them is a style shift with no error anywhere.
    {
        const n = 4096;
        const x = try arena.alloc(f32, n);
        // Spread over the range a real FFN sees, tails included.
        for (x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / @as(f32, n) - 0.5) * 24.0;
        const want = try arena.alloc(f32, n);
        const got = try arena.alloc(f32, n);
        // `2e-5` rather than the Vulkan arm's 2e-6 because every exp in `cuda/elt.zig`
        // is `ex2.approx` — an approximate instruction, per that file's header. Do not
        // "fix" it by tightening the bound.
        for ([_]struct { []const u8, bool }{ .{ "gelu_quick (CLIP-L)", true }, .{ "gelu_erf (CLIP-G)", false } }) |arm| {
            const name, const is_quick = arm;
            @memcpy(want, x);
            if (is_quick) ops.act.geluQuick(want) else ops.act.geluErf(want);
            try up(be, &a_d, x);
            if (is_quick) try be.geluQuick(a_d, n) else try be.geluErf(a_d, n);
            try be.tensorDownload(a_d, std.mem.sliceAsBytes(got));
            try rep.check(name, want, got, 2e-5);
        }
    }

    // --- whole stages against their CPU forwards -----------------------------
    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("\nskipping the whole-stage checks: no checkpoint at {s}\n", .{ckpt});
        try summarize(stdout, failures);
        return;
    };
    var st = try TensorPencil.SafeTensors.open(arena, io, ckpt);
    defer st.deinit();
    const store: TensorPencil.weights.WeightStore = .{ .safetensors = &st };
    try stdout.print("\nwhole stages against the CPU forward ({s}):\n", .{ckpt});

    // The UNet check is SD1.5-specific (config + a 768-wide context and no `y`);
    // the VAE below is not — the two families' decoders are architecturally
    // IDENTICAL and differ only in weights and in `scaling_factor`, which `decode`
    // never sees — so an SDXL checkpoint still exercises the sweep, which is how
    // that family's fp16 behaviour gets measured at all.
    const family = TensorPencil.pipeline.detectFamily(store) catch .sd15;
    try stdout.print("checkpoint family: {t}\n", .{family});

    // --- the CLIP text tower, whole, on a weighted two-chunk prompt -----------
    // Both towers when the checkpoint has both, since they differ in naming (fused
    // q/k/v), depth, head count AND activation — the SDXL arm is the one that
    // exercises the erf-GELU path and the captured-penultimate-layer path.
    {
        const clip_text = TensorPencil.models.clip_text;
        const clip_text_cuda = TensorPencil.models.clip_text_cuda;
        const clip_tok = TensorPencil.clip_tokenizer;
        const Tower = struct { name: []const u8, cfg: clip_text.Config, prefix: []const u8, pad: u32, capture: ?usize };
        const towers: []const Tower = switch (family) {
            .sd15 => &.{.{
                .name = "clip_l",
                .cfg = clip_text.clip_l,
                .prefix = "cond_stage_model.transformer.text_model.",
                .pad = clip_tok.eos_id,
                .capture = null,
            }},
            .sdxl => &.{
                .{
                    .name = "clip_l",
                    .cfg = clip_text.clip_l,
                    .prefix = "conditioner.embedders.0.transformer.text_model.",
                    .pad = clip_tok.eos_id,
                    .capture = clip_text.clip_l.layers - 1,
                },
                .{
                    .name = "clip_g",
                    .cfg = clip_text.clip_g,
                    .prefix = "conditioner.embedders.1.model.",
                    .pad = 0,
                    .capture = clip_text.clip_g.layers - 1,
                },
            },
            // This command is the SD-family kernel gate; other architectures have
            // their own text towers and are not exercised here.
            .krea2, .zimage => &.{},
        };
        for (towers) |tw| {
            var enc = clip_text.TextEncoder.load(arena, store, tw.cfg, tw.prefix) catch |err| {
                try stdout.print("{s:<44} load failed: {t}\n", .{ tw.name, err });
                failures += 1;
                continue;
            };
            defer enc.deinit();

            // Two chunks with a mix of weighted and unweighted rows, so the chunk
            // batching, the causal mask per chunk and `applyWeights` are all live.
            const clen = clip_tok.context_length;
            const tokens = try arena.alloc(clip_tok.Weighted, 2 * clen);
            for (tokens, 0..) |*t, i| {
                const slot = i % clen;
                t.* = .{
                    .id = if (slot == 0) clip_tok.bos_id else if (slot > 60) clip_tok.eos_id else rnd.intRangeLessThan(u32, 0, 49000),
                    .weight = if (slot % 5 == 0) 1.0 else if (slot % 3 == 0) 1.15 else 0.8,
                };
            }
            const p: clip_tok.Prompt = .{ .tokens = tokens, .chunks = 2 };

            const want = try arena.alloc(f32, p.seq() * tw.cfg.hidden);
            const got = try arena.alloc(f32, p.seq() * tw.cfg.hidden);
            // Separate caches: each side must compute its own `z_empty`, or the device
            // arm is handed the host's and that half goes unchecked.
            var cache_cpu: ?[]f32 = null;
            var cache_dev: ?[]f32 = null;
            const run: clip_text.TextEncoder.PromptRun = .{
                .empty_cache = &cache_cpu,
                .pad_id = tw.pad,
                .capture_layer = tw.capture,
            };
            try enc.encodePrompt(io, arena, want, &p, run);
            var dev_run = run;
            dev_run.empty_cache = &cache_dev;
            try clip_text_cuda.encodePrompt(&enc, be, io, arena, got, &p, dev_run);

            var name_buf: [80]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "clip_text_cuda {s} (2 chunks, weighted)", .{tw.name});
            // f16 tensor-core GEMMs against an f32 CPU forward over 12/32 layers —
            // the same regime as the whole-UNet check below, and the per-kernel checks
            // above are what pin the arithmetic.
            try rep.check(name, want, got, 2e-2);
        }
    }
    if (family == .sd15) {
        var unet = try sd_unet.UNet.load(arena, store, sd_unet.sd15, "model.diffusion_model.");
        defer unet.deinit();
        const lat_h = 16;
        const lat_w = 16;
        const ctx_seq = 77;
        const n = lat_h * lat_w;
        const x = try arena.alloc(f32, n * 4);
        const context = try arena.alloc(f32, ctx_seq * 768);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        for (context) |*v| v.* = rnd.floatNorm(f32) * 0.5;
        const t: f32 = 481;

        var cpu_ws = try sd_unet.Workspace.init(arena, &unet, lat_h, lat_w, ctx_seq);
        defer cpu_ws.deinit();
        const want = try arena.alloc(f32, n * 4);
        try sd_unet.forward(&unet, io, arena, &cpu_ws, want, x, lat_h, lat_w, t, context, ctx_seq, null);

        var sess = try sd_unet_cuda.Session.init(arena, be, &unet, context, ctx_seq, null);
        defer sess.deinit(be);
        var ws = try sd_unet_cuda.Workspace.init(arena, be, &unet, lat_h, lat_w, ctx_seq);
        defer ws.deinit(be);
        const got = try arena.alloc(f32, n * 4);
        try sd_unet_cuda.forward(&unet, be, &sess, &ws, io, arena, got, x, lat_h, lat_w, t, null);
        // A whole UNet of f16 GEMMs; the per-kernel checks above are what pin the
        // arithmetic, this pins the wiring.
        try rep.check("sd_unet_cuda.forward (SD1.5, 16x16 latent)", want, got, 2e-2);
    }

    {
        var dec = try sd_vae.Decoder.load(arena, store, if (family == .sdxl) sd_vae.sdxl else sd_vae.sd15, "first_stage_model.");
        defer dec.deinit();
        // ⚠️ A SWEEP, not one size, and that is the point: this check passed at
        // 12x10 for months while a 96x96 latent (a 768-square render) decoded to a
        // solid white image. Anything that reduces over positions — GroupNorm's
        // chunked Welford statistics, the conv's banding, the attention tiling —
        // can be correct on a small tile and wrong once the position count crosses
        // a kernel's own blocking, so the sizes below deliberately straddle the
        // render sizes people actually use (64 = 512², 96 = 768², 128 = 1024²) and
        // the tiled-decode tile sizes (64) plus its ragged edge tiles.
        const sizes = [_][2]usize{
            .{ 12, 10 }, // tiny, not square, not powers of two
            .{ 24, 16 }, // a tiled decode's ragged edge tile
            .{ 64, 64 }, // 512² render / one whole tile of a tiled decode
            .{ 96, 64 }, // non-square, both large
            .{ 96, 96 }, // 768² render
            .{ 128, 128 }, // 1024² render
        };
        // ⚠️ NOT `arena`: the CPU reference frees every intermediate it allocates,
        // but an arena only returns memory at scope exit, so the sweep accumulates
        // every buffer of every size and the OOM killer takes the process at 128²
        // (~2 GB per activation there). A freeing allocator keeps the peak at the
        // few live buffers the decoder actually holds.
        const ref_gpa = std.heap.page_allocator;
        // ⚠️ The MAGNITUDE arm is as load-bearing as the size arm. `decode` is
        // handed `z / scaling_factor`, so a real latent arrives ~5.5x (SD1.5) to
        // ~7.7x (SDXL) wider than the unit gaussian a fixture naturally uses — and
        // a unit-gaussian check passed at every size while a REAL 768-square render
        // decoded to a solid white image.
        const mags = [_]f32{ 1.0, 1.0 / 0.13025 };
        for (sizes) |s| {
            for (mags) |mag| {
                const lat_h = s[0];
                const lat_w = s[1];
                const z = try ref_gpa.alloc(f32, lat_h * lat_w * 4);
                defer ref_gpa.free(z);
                for (z) |*v| v.* = rnd.floatNorm(f32) * mag;
                const want = try dec.decode(io, ref_gpa, z, lat_h, lat_w);
                defer ref_gpa.free(want);
                const got = try sd_vae_cuda.decode(&dec, be, ref_gpa, z, lat_h, lat_w, null);
                defer ref_gpa.free(got);
                var name_buf: [80]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "sd_vae_cuda.decode ({d}x{d} latent, z x{d:.1})", .{ lat_h, lat_w, mag });
                try rep.check(name, want, got, 5e-3);
            }
        }
    }

    try summarize(stdout, failures);
}

fn summarize(stdout: *Io.Writer, failures: usize) !void {
    if (failures == 0) {
        try stdout.print("\nall checks ok\n", .{});
        try stdout.flush();
        return;
    }
    try stdout.print("\n{d} check(s) FAILED\n", .{failures});
    try stdout.flush();
    return error.ValidationFailed;
}

/// Check a Qwen3 text encoder end to end: every GPU backend against the CPU
/// forward of the SAME file, and optionally a second encoder against the first
/// (which is how a quantized GGUF is compared to its bf16 original).
///
/// A CLI command rather than a test for the usual reason — the test binary brings
/// up no CUDA/Vulkan context — and because an 8 GB encoder is not a unit test.
///
/// ⚠️ The two questions it separates are genuinely different and were being
/// conflated by looking at renders: "does this backend compute what the CPU
/// computes" (a kernel question, expect ~1e-3) and "does this quantization change
/// the conditioning" (a format question, expect much more). A render comparison
/// answers neither on its own.
fn teTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, ref_path: []const u8, variant: TensorPencil.models.qwen3.Variant) !void {
    const qwen3 = TensorPencil.models.qwen3;
    const qwen3_cuda = TensorPencil.models.qwen3_cuda;
    const qwen3_gpu = TensorPencil.models.qwen3_gpu;
    const zimage_text = TensorPencil.models.zimage_text;
    const cuda = TensorPencil.gpu.cuda;
    const tokenizer = TensorPencil.tokenizer;

    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("te-test needs an encoder ({s})\n", .{path});
        return;
    };
    // A prompt long enough that the GEMMs are real (m in the hundreds), since a
    // 5-token encode exercises a different regime from any actual render.
    const text = "A sophisticated, high-end vector and layered graphic art piece on a pure " ++
        "white background, an ethereal angelic figure crouched in profile, translucent " ++
        "wings of layered blue floral petals, soft teals and pale pinks, clean precise " ++
        "vector lines, two dark navy hummingbird silhouettes, restricted harmonious palette.";

    var tok = try tokenizer.Tokenizer.init(arena);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(arena);
    try zimage_text.buildIds(&tok, arena, text, &ids);

    var ct = try TensorPencil.pipeline.Container.open(arena, io, path);
    defer ct.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(arena, ct.store(), variant);
    defer enc.deinit();
    try stdout.print("== te-test ==\nencoder : {s}\n          {d} layers, hidden {d}, rope_theta {d:.0}, q dtype {t}, {d} tokens\n", .{
        path, enc.cfg.n_layers, enc.cfg.hidden, enc.cfg.rope_theta, enc.layers[0].q.dtype, ids.items.len,
    });

    var t0 = std.Io.Clock.real.now(io);
    const want = try enc.encode(io, arena, ids.items, null);
    var ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
    try stdout.print("\ncpu       {d:8.0} ms   (reference)\n", .{ms});

    var failures: usize = 0;
    const R = struct {
        fn rel(a: []const f32, b: []const f32) f64 {
            var num: f64 = 0;
            var den: f64 = 0;
            for (a, b) |e, g| {
                num += (@as(f64, e) - g) * (@as(f64, e) - g);
                den += @as(f64, e) * e;
            }
            return if (den == 0) 0 else @sqrt(num / den);
        }
        fn finite(a: []const f32) bool {
            for (a) |v| if (!std.math.isFinite(v)) return false;
            return true;
        }
    };

    // Each CUDA arm, then Vulkan. A backend that is absent simply reports so.
    inline for (.{ true, false }) |libs| {
        if (cuda.Backend.initLibs(arena)) |_| {} else |_| {}
        const be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch null;
        if (be) |b| {
            defer b.deinit();
            const name = if (libs) "cuda" else "zig-cuda";
            if (!qwen3_cuda.supportsWeights(&enc)) {
                try stdout.print("{s:<9} REFUSED (supportsWeights false — would fall back to CPU)\n", .{name});
                failures += 1;
            } else {
                _ = try qwen3_cuda.encode(&enc, b, io, arena, ids.items, null); // warm
                t0 = std.Io.Clock.real.now(io);
                const got = try qwen3_cuda.encode(&enc, b, io, arena, ids.items, null);
                ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                const r = R.rel(want, got);
                const ok = R.finite(got) and r < 5e-3;
                if (!ok) failures += 1;
                try stdout.print("{s:<9} {d:8.0} ms   rel L2 {e:.4}  {s}\n", .{ name, ms, r, if (ok) "ok" else "FAILED" });
            }
        } else {
            try stdout.print("{s:<9} unavailable\n", .{if (libs) "cuda" else "zig-cuda"});
        }
    }
    if (TensorPencil.gpu.Context.init(arena)) |gc| {
        defer gc.deinit();
        if (!qwen3_gpu.supportsWeights(gc, &enc)) {
            try stdout.print("{s:<9} REFUSED (supportsWeights false — would fall back to CPU)\n", .{"vulkan"});
            failures += 1;
        } else {
            _ = try qwen3_gpu.encode(&enc, gc, io, arena, ids.items, false, null);
            t0 = std.Io.Clock.real.now(io);
            const got = try qwen3_gpu.encode(&enc, gc, io, arena, ids.items, false, null);
            ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
            const r = R.rel(want, got);
            const ok = R.finite(got) and r < 5e-3;
            if (!ok) failures += 1;
            try stdout.print("{s:<9} {d:8.0} ms   rel L2 {e:.4}  {s}\n", .{ "vulkan", ms, r, if (ok) "ok" else "FAILED" });
        }
    } else |err| try stdout.print("{s:<9} unavailable ({t})\n", .{ "vulkan", err });

    // The format question, kept separate from the kernel question above.
    if (ref_path.len != 0) {
        var rt = try TensorPencil.pipeline.Container.open(arena, io, ref_path);
        defer rt.deinit();
        var ref = try qwen3.TextEncoder.loadVariant(arena, rt.store(), variant);
        defer ref.deinit();
        const rwant = try ref.encode(io, arena, ids.items, null);
        try stdout.print("\nvs {s}\n  ({t} weights)  rel L2 {e:.4}  — the QUANTIZATION delta, not a kernel error\n", .{
            ref_path, ref.layers[0].q.dtype, R.rel(rwant, want),
        });
    }
    try summarize(stdout, failures);
}

/// Check `zimage_cuda`'s device forward against `zimage.DiT.predict` on real
/// checkpoint weights — the CUDA twin of `zimage_gpu`'s gated Vulkan test, and a
/// CLI command for the reason `sd-cuda-test` and `cuda-dit-test` are: the test
/// binary brings up no CUDA context, so a `test` block here would only ever skip.
///
/// Truncated to a few trunk layers so it loads in seconds. The loop bound is not
/// what a kernel port gets wrong; the block's shape is. Exits non-zero on failure
/// so it works as a gate.
fn zimageCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, vae_path: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const zimage = TensorPencil.models.zimage;
    const zimage_cuda = TensorPencil.models.zimage_cuda;
    const safetensors = TensorPencil.SafeTensors;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("zimage-cuda-test needs a Z-Image checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try safetensors.open(arena, io, ckpt);
    defer ck.deinit();
    var cfg = zimage.z_image;
    cfg.n_layers = 2;
    var model = try zimage.DiT.load(arena, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== zimage-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });
    if (!zimage_cuda.supported(&model)) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }

    const lat = 16; // 8x8 = 64 image tokens, exactly two pad buckets
    const seq_txt = 20;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const ctxv = try arena.alloc(f32, seq_txt * cfg.cap_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, cfg.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);

    const sigma: f32 = 0.75;
    const cap = try model.capTokens(io, arena, ctxv, seq_txt);
    const cap_padded = cfg.padded(seq_txt);

    const want = try arena.alloc(f32, x_lat.len);
    {
        const adaln = try model.adalnInput(io, arena, sigma);
        try model.predict(io, arena, want, x_lat, lat, lat, cap, cap_padded, adaln, null);
    }

    var failures: usize = 0;
    // ⚠️ BOTH attention paths. `opAttnTC` is what runs; the naive one is the
    // fallback and the reference the fast path was validated against. Checking
    // only the fast one leaves the fallback free to rot — and checking only the
    // slow one is what let the watchdog problem reach a render on Vulkan.
    const saved = zimage_cuda.force_naive_attn;
    defer zimage_cuda.force_naive_attn = saved;
    for ([_]bool{ false, true }) |naive| {
        zimage_cuda.force_naive_attn = naive;
        var sess = try zimage_cuda.Session.init(arena, io, be, &model, lat, lat, cap, cap_padded, &.{ sigma, 0 });
        defer sess.deinit(arena, be);
        var ws = try zimage_cuda.Workspace.init(be, &model, lat, lat, cap_padded);
        defer ws.deinit(be);
        const got = try arena.alloc(f32, x_lat.len);
        try zimage_cuda.forward(&model, be, &sess, &ws, io, arena, got, x_lat, sigma, null);

        var num: f64 = 0;
        var den: f64 = 0;
        var nonfinite: usize = 0;
        for (want, got) |e, a| {
            if (!std.math.isFinite(a)) nonfinite += 1;
            num += (@as(f64, e) - a) * (@as(f64, e) - a);
            den += @as(f64, e) * e;
        }
        const rel = if (den == 0) 0 else @sqrt(num / den);
        // The GEMMs run tensor cores against the CPU's f32 accumulation — the same
        // regime `sd_unet_cuda` sits in, and `zimage_gpu` measures 2.2e-4 there.
        const ok = nonfinite == 0 and rel < 1e-3;
        if (!ok) failures += 1;
        try stdout.print("forward vs CPU ({s:<11})  rel L2 {e:.4}  {s}{s}\n", .{
            if (naive) "naive attn" else "opAttnTC", rel,
            if (ok) "ok" else "FAILED",
            if (nonfinite != 0) " (non-finite output)" else "",
        });
    }
    // --- the VAE decode ---------------------------------------------------------
    // ⚠️ Checked at a SMALL and a LARGE latent, because the mid-block attention takes
    // a different path at each and the small one is where the f16-logit overflow that
    // once rendered solid white actually shows. A single size is how that bug hid.
    if (vae_path.len != 0) {
        std.Io.Dir.cwd().access(io, vae_path, .{}) catch {
            try stdout.print("\n(skipping the VAE: {s} not found)\n", .{vae_path});
            try summarize(stdout, failures);
            return;
        };
        var vt = try safetensors.open(arena, io, vae_path);
        defer vt.deinit();
        const store: TensorPencil.weights.WeightStore = .{ .safetensors = &vt };
        const vcfg = TensorPencil.models.sd_vae.flux;
        var vdec = try TensorPencil.models.sd_vae.Decoder.load(arena, store, vcfg, "");
        defer vdec.deinit();
        try stdout.print("\n-- VAE decode ({d}-channel latent) --\n", .{vcfg.z_channels});

        const sizes = [_][2]usize{ .{ 12, 10 }, .{ 40, 32 } };
        const saved_vae = TensorPencil.models.sd_vae_cuda.force_naive_attn;
        defer TensorPencil.models.sd_vae_cuda.force_naive_attn = saved_vae;
        for (sizes) |sz| {
            const zh = sz[0];
            const zw = sz[1];
            const z = try arena.alloc(f32, vcfg.z_channels * zh * zw);
            for (z) |*v| v.* = rnd.floatNorm(f32);
            const ref = try vdec.decode(io, arena, z, zh, zw);
            for ([_]bool{ false, true }) |naive| {
                TensorPencil.models.sd_vae_cuda.force_naive_attn = naive;
                const t0 = std.Io.Clock.real.now(io);
                const got = try TensorPencil.models.sd_vae_cuda.decode(&vdec, be, arena, z, zh, zw, null);
                const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                var num: f64 = 0;
                var den: f64 = 0;
                var bad: usize = 0;
                for (ref, got) |e, a| {
                    if (!std.math.isFinite(a)) bad += 1;
                    num += (@as(f64, e) - a) * (@as(f64, e) - a);
                    den += @as(f64, e) * e;
                }
                const rel = if (den == 0) 0 else @sqrt(num / den);
                const ok = bad == 0 and rel < 5e-3;
                if (!ok) failures += 1;
                try stdout.print("  {d}x{d} {s:<12} {d:7.0} ms  rel L2 {e:.4}  {s}{s}\n", .{
                    zh, zw, if (naive) "be.attn" else "opAttnTC", dt, rel,
                    if (ok) "ok" else "FAILED",
                    if (bad != 0) " (non-finite)" else "",
                });
            }
        }
    }
    try summarize(stdout, failures);
}

/// Best-of-`n` wall time in ms for one device-work closure, warming once first
/// (PTX JIT, cuBLASLt heuristics and cuDNN plans are all first-call costs).
fn timeBest(io: Io, n: usize, comptime go: anytype, be: anytype, args: anytype) !f64 {
    try go(be, args);
    var best: f64 = std.math.inf(f64);
    for (0..n) |_| {
        const a = std.Io.Clock.real.now(io);
        try go(be, args);
        const b = std.Io.Clock.real.now(io);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    return best;
}

/// Time ONE Z-Image trunk step's worth of bf16 GEMMs in isolation, with and
/// without the two streaming passes `opGemmBf16` wraps around every call (the
/// f32→bf16 activation pad, and `bias_compact` on the f32 output).
///
/// The point is a receipt rather than an estimate. A whole-render profile puts
/// ~88% of a step under `.matmul`, but that bucket spans the op, not the GEMM —
/// `opGemmBf16` is `ptic()`-scoped, so the conversion passes are counted inside
/// it. Without this split, "cuBLASLt is as fast as it gets" and "we spend a
/// third of the step converting operands" are indistinguishable.
///
/// `seq` is the joint (caption ++ image) token count; 6848 is a 1056x1584 render.
/// Weights are random and each shape is uploaded once — the working set is far
/// past L2 either way, so reuse costs nothing in fidelity and 11.6 GB in VRAM.
fn zimageCudaBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, seq: usize, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const zimage = TensorPencil.models.zimage;
    const Buf = cuda.backend.DeviceBuffer;
    const cfg = zimage.z_image;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();

    // The seven GEMMs of one block, in forward order: q, k, v, attn-out, w1, w3, w2.
    const Shape = struct { co: usize, k: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .co = cfg.qDim(), .k = cfg.dim, .name = "wq" },
        .{ .co = cfg.kvDim(), .k = cfg.dim, .name = "wk" },
        .{ .co = cfg.kvDim(), .k = cfg.dim, .name = "wv" },
        .{ .co = cfg.dim, .k = cfg.dim, .name = "out" },
        .{ .co = cfg.mlp_dim, .k = cfg.dim, .name = "w1" },
        .{ .co = cfg.mlp_dim, .k = cfg.dim, .name = "w3" },
        .{ .co = cfg.dim, .k = cfg.mlp_dim, .name = "w2" },
    };
    // 30 trunk layers + the 2 noise refiners, which are the same block shape.
    const blocks = cfg.n_layers + cfg.n_refiner_layers;

    try stdout.print("== zimage-cuda-bench seq={d} ==\ncuda device: {s} (kernels: {t})\n", .{ seq, be.deviceName(), be.kernels });
    try stdout.print("{d} blocks x {d} GEMMs = {d} calls/step\n\n", .{ blocks, shapes.len, blocks * shapes.len });

    var max_k: usize = 0;
    var max_co: usize = 0;
    for (shapes) |s| {
        max_k = @max(max_k, s.k);
        max_co = @max(max_co, s.co);
    }
    const m_pad = std.mem.alignForward(usize, seq, 128);

    var x: Buf = .{};
    var y: Buf = .{};
    try be.ensureDeviceBuffer(&x, m_pad * max_k * 4);
    try be.ensureDeviceBuffer(&y, m_pad * max_co * 4);
    {
        const host = try arena.alloc(f32, seq * max_k);
        defer arena.free(host);
        var prng = std.Random.DefaultPrng.init(99);
        for (host) |*v| v.* = prng.random().floatNorm(f32) * 0.02;
        try be.tensorUpload(x, std.mem.sliceAsBytes(host));
    }
    // One host-side bf16 weight per distinct (co,k); `cachedWeight` keys by host
    // pointer, so a shared buffer would also share the device copy — which is what
    // we want, since the real weights are read from VRAM either way.
    var wbytes: [shapes.len][]u8 = undefined;
    var seen: [shapes.len]?usize = @splat(null);
    for (shapes, 0..) |s, i| {
        for (shapes[0..i], 0..) |t, j| {
            if (t.co == s.co and t.k == s.k) {
                seen[i] = j;
                break;
            }
        }
        if (seen[i]) |j| {
            wbytes[i] = wbytes[j];
            continue;
        }
        const raw = try arena.alloc(u16, s.co * s.k);
        var prng = std.Random.DefaultPrng.init(7 + i);
        // bf16 = the top 16 bits of an f32, so build it that way rather than
        // by hand: a denormal or a NaN pattern here would time differently.
        for (raw) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().floatNorm(f32) * 0.02)) >> 16);
        wbytes[i] = std.mem.sliceAsBytes(raw);
    }

    const zeros = try arena.alloc(f32, max_co);
    @memset(zeros, 0);

    // One step of GEMMs, timed as a whole: individual calls are 0.5-13 ms, so a
    // per-call clock would mostly measure the sync.
    const Run = struct {
        fn go(b: *cuda.Backend, sh: []const Shape, wb: [][]u8, nb: usize, m: usize, xb: Buf, yb: Buf, z: []const f32) !void {
            try b.beginBatch();
            for (0..nb) |_| {
                // `null` bias, exactly as `zimage_cuda.gemm` passes it.
                _ = z;
                for (sh, 0..) |s, i| try b.opGemmBf16(yb, xb, m, wb[i], s.co, s.k, null);
            }
            try b.endBatch();
        }
    };

    var t_full: f64 = std.math.inf(f64);
    var t_gemm: f64 = std.math.inf(f64);
    inline for (.{ false, true }) |gemm_only| {
        cuda.backend.bench_gemm_only = gemm_only;
        try Run.go(be, &shapes, &wbytes, 2, seq, x, y, zeros); // warm: JIT + plans
        for (0..3) |_| {
            const a = std.Io.Clock.real.now(io);
            try Run.go(be, &shapes, &wbytes, blocks, seq, x, y, zeros);
            const c = std.Io.Clock.real.now(io);
            const ms = @as(f64, @floatFromInt(c.nanoseconds - a.nanoseconds)) / 1e6;
            if (gemm_only) t_gemm = @min(t_gemm, ms) else t_full = @min(t_full, ms);
        }
    }
    cuda.backend.bench_gemm_only = false;

    // FLOPs are counted at the PADDED m: that arithmetic is really performed, and
    // charging only the useful rows would report a rate the hardware never hit.
    var flop: f64 = 0;
    for (shapes) |s| flop += 2.0 * @as(f64, @floatFromInt(m_pad * s.co * s.k));
    flop *= @floatFromInt(blocks);

    const conv = t_full - t_gemm;
    try stdout.print("full opGemmBf16 : {d:.1} ms   ({d:.1} TFLOP/s effective)\n", .{ t_full, flop / (t_full / 1e3) / 1e12 });
    try stdout.print("GEMM only       : {d:.1} ms   ({d:.1} TFLOP/s)\n", .{ t_gemm, flop / (t_gemm / 1e3) / 1e12 });
    try stdout.print("conversion pass : {d:.1} ms   ({d:.1}% of the op)\n", .{ conv, 100.0 * conv / t_full });
    try stdout.print("\narithmetic: {d:.1} TFLOP/step at m_pad={d} (m={d}, {d:.1}% pad waste)\n", .{
        flop / 1e12, m_pad, seq,
        100.0 * @as(f64, @floatFromInt(m_pad - seq)) / @as(f64, @floatFromInt(m_pad)),
    });

    // --- the rest of the block, so the three parts can be added up -------------
    // Everything a block does apart from its GEMMs, at the same shapes and in the
    // same order. Without these the GEMM figure above is only half an answer: it
    // says how fast the GEMMs are, not what fraction of the step they are.
    const heads = cfg.n_heads;
    const hd = cfg.head_dim;
    var q: Buf = .{};
    var kk: Buf = .{};
    var v: Buf = .{};
    var ao: Buf = .{};
    var mg: Buf = .{};
    var mu: Buf = .{};
    var mv: Buf = .{};
    var nrm: Buf = .{};
    var frq: Buf = .{};
    var nw: Buf = .{};
    try be.ensureDeviceBuffer(&q, seq * cfg.qDim() * 4);
    try be.ensureDeviceBuffer(&kk, seq * cfg.kvDim() * 4);
    try be.ensureDeviceBuffer(&v, seq * cfg.kvDim() * 4);
    try be.ensureDeviceBuffer(&ao, seq * cfg.qDim() * 4);
    try be.ensureDeviceBuffer(&mg, seq * cfg.mlp_dim * 4);
    try be.ensureDeviceBuffer(&mu, seq * cfg.mlp_dim * 4);
    try be.ensureDeviceBuffer(&nrm, seq * cfg.dim * 4);
    try be.ensureDeviceBuffer(&mv, blocks * 4 * cfg.dim * 4);
    try be.ensureDeviceBuffer(&frq, seq * hd * 4);
    try be.ensureDeviceBuffer(&nw, cfg.mlp_dim * 4);

    const t_attn = try timeBest(io, 3, struct {
        fn go(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| try b.opAttnTC(a.q, a.k, a.v, a.o, a.seq, a.h, a.h, a.hd, 0.088388);
            try b.endBatch();
        }
    }.go, be, .{ .q = q, .k = kk, .v = v, .o = ao, .seq = seq, .h = heads, .hd = hd, .nb = blocks });

    // ⚠️ Timed per op FAMILY, not as one lump. "elementwise is 30% of the step"
    // is not actionable — these are pure-bandwidth kernels, so the question is
    // which of them is far from the 936 GB/s the card can do, and a single total
    // averages the answer away.
    const ea = .{
        .x = y, .nrm = nrm, .mv = mv, .q = q, .k = kk, .nw = nw, .frq = frq,
        .mg = mg, .mu = mu, .seq = seq, .dim = cfg.dim, .mlp = cfg.mlp_dim,
        .h = heads, .hd = hd, .zoff = blocks * 4 * cfg.dim - cfg.dim, .nb = blocks,
    };
    const E = struct {
        // ⚠️ Each is the exact call `zimage_cuda.blockForward` makes, per block.
        fn rmsmod(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| {
                try b.rmsMod(a.x, a.nrm, a.mv, a.seq, a.dim, 0, a.zoff, 1e-5);
                try b.rmsMod(a.x, a.nrm, a.mv, a.seq, a.dim, 2 * a.dim, a.zoff, 1e-5);
            }
            try b.endBatch();
        }
        /// The per-HEAD q/k norms: many narrow (hd=128) rows.
        fn qk(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| {
                try b.qkNorm(a.q, a.q, a.nw, a.seq * a.h, a.hd, 1e-7);
                try b.qkNorm(a.k, a.k, a.nw, a.seq * a.h, a.hd, 1e-7);
            }
            try b.endBatch();
        }
        /// The two sandwich norms, through the SAME entry point but at full model
        /// width: few rows, each 3840 wide. A different regime entirely.
        fn sandwich(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| {
                try b.qkNorm(a.nrm, a.nrm, a.nw, a.seq, a.dim, 1e-5);
                try b.qkNorm(a.nrm, a.nrm, a.nw, a.seq, a.dim, 1e-5);
            }
            try b.endBatch();
        }
        fn ropes(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| {
                try b.rope(a.q, a.frq, a.seq, a.h, a.hd / 2, a.seq * a.hd / 2);
                try b.rope(a.k, a.frq, a.seq, a.h, a.hd / 2, a.seq * a.hd / 2);
            }
            try b.endBatch();
        }
        fn gated(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| {
                try b.gatedAdd(a.x, a.nrm, a.mv, a.seq * a.dim, a.dim, a.dim);
                try b.gatedAdd(a.x, a.nrm, a.mv, a.seq * a.dim, a.dim, 3 * a.dim);
            }
            try b.endBatch();
        }
        fn silu(b: *cuda.Backend, a: anytype) !void {
            try b.beginBatch();
            for (0..a.nb) |_| try b.siluMul(a.mg, a.mu, a.seq * a.mlp);
            try b.endBatch();
        }
    };
    const sd: f64 = @floatFromInt(seq * cfg.dim);
    const parts = [_]struct { name: []const u8, t: f64, gb: f64 }{
        .{ .name = "rmsMod x2", .t = try timeBest(io, 3, E.rmsmod, be, ea), .gb = 2 * sd * 8 },
        .{ .name = "qkNorm q,k (hd=128)", .t = try timeBest(io, 3, E.qk, be, ea), .gb = 2 * sd * 8 },
        .{ .name = "qkNorm sandwich (w=3840)", .t = try timeBest(io, 3, E.sandwich, be, ea), .gb = 2 * sd * 8 },
        .{ .name = "rope q,k", .t = try timeBest(io, 3, E.ropes, be, ea), .gb = 2 * sd * 8 },
        .{ .name = "gatedAdd x2", .t = try timeBest(io, 3, E.gated, be, ea), .gb = 2 * sd * 12 },
        .{ .name = "siluMul", .t = try timeBest(io, 3, E.silu, be, ea), .gb = @as(f64, @floatFromInt(seq * cfg.mlp_dim)) * 12 },
    };
    var t_elt: f64 = 0;
    for (parts) |p| t_elt += p.t;

    const total = t_full + t_attn + t_elt;
    // Attention is 4·seq²·hd per head (QK then PV), both at full width — Z-Image
    // has no causal mask, so nothing is skipped.
    const aflop = 4.0 * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(seq)) *
        @as(f64, @floatFromInt(hd * heads * blocks));
    try stdout.print("\n-- one step's device work, by part --\n", .{});
    try stdout.print("GEMMs (with conv) : {d:7.1} ms  {d:5.1}%\n", .{ t_full, 100 * t_full / total });
    try stdout.print("attention         : {d:7.1} ms  {d:5.1}%   ({d:.1} TFLOP/step, {d:.1} TFLOP/s)\n", .{ t_attn, 100 * t_attn / total, aflop / 1e12, aflop / (t_attn / 1e3) / 1e12 });
    try stdout.print("elementwise       : {d:7.1} ms  {d:5.1}%\n", .{ t_elt, 100 * t_elt / total });
    try stdout.print("                    {d:7.1} ms total device work\n", .{total});
    // ⚠️ The achieved bandwidth is the number that says whether a kernel is
    // *finished* or merely *working*: these move a known number of bytes and do
    // almost no arithmetic, so anything far under the card's ~936 GB/s is a
    // kernel problem, not a workload.
    try stdout.print("\n-- elementwise, per op (x{d} blocks) --\n", .{blocks});
    for (parts) |p| {
        const gb = p.gb * @as(f64, @floatFromInt(blocks)) / 1e9;
        try stdout.print("{s:<26} {d:6.1} ms   {d:6.1} GB/s\n", .{ p.name, p.t, gb / (p.t / 1e3) });
    }
    try stdout.flush();
}

/// Validate the CUDA VAE decode (vae_cuda) against the CPU decode on a random
/// latent: proves the im2col conv + vae_norm + tensor-core mid attention chain.
fn cudaVaeTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, zh: usize) !void {
    const cuda = TensorPencil.gpu.cuda;
    const wan_vae = TensorPencil.models.wan_vae;
    const vae_cuda = TensorPencil.models.vae_cuda;
    const vae_path = "models/vae/krea2RealVae_v10.safetensors";
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch {
        try stdout.print("cuda-vae-test needs the VAE checkpoint\n", .{});
        return;
    };
    const zw = zh;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const z = try arena.alloc(f32, wan_vae.latent_channels * zh * zw);
    for (z) |*v| v.* = rnd.floatNorm(f32);

    var st = try TensorPencil.SafeTensors.open(arena, io, vae_path);
    defer st.deinit();
    var dec = try wan_vae.Decoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();

    const want = try dec.decode(io, arena, z, zh, zw, null);

    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== cuda-vae-test zh={d} ({d}x{d} px) ==\ncuda device: {s}\n", .{ zh, zh * 8, zw * 8, be.deviceName() });

    const got0 = try vae_cuda.decode(&dec, be, io, arena, z, zh, zw, null);
    arena.free(got0);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        const g = try vae_cuda.decode(&dec, be, io, arena, z, zh, zw, null);
        const b = std.Io.Clock.real.now(io);
        arena.free(g);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    const got = try vae_cuda.decode(&dec, be, io, arena, z, zh, zw, null);

    var max_err: f32 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
        num += (@as(f64, e) - a) * (@as(f64, e) - a);
        den += @as(f64, e) * e;
    }
    try stdout.print("cuda decode: {d:.3} s (best of 3)\n", .{best / 1000.0});
    try stdout.print("cuda-vs-cpu: max_err={d:.6} rel RMSE={d:.6}\n", .{ max_err, @sqrt(num / den) });
    try stdout.flush();
}

fn cudaEncodeTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    const qwen3 = TensorPencil.models.qwen3;
    const qwen3_cuda = TensorPencil.models.qwen3_cuda;
    const krea2_text = TensorPencil.models.krea2_text;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, te_path, .{}) catch {
        try stdout.print("cuda-encode-test needs the text encoder checkpoint\n", .{});
        return;
    };
    var tok = try TensorPencil.tokenizer.Tokenizer.init(arena);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    try krea2_text.buildIds(&tok, arena, "a fluffy orange cat sitting on a windowsill", &ids);

    var st = try TensorPencil.SafeTensors.open(arena, io, te_path);
    defer st.deinit();
    var enc = try qwen3.TextEncoder.load(arena, .{ .safetensors = &st });
    defer enc.deinit();

    const want = try enc.encode(io, arena, ids.items, null);

    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== cuda-encode-test ({d} tokens) ==\ncuda device: {s}\n", .{ ids.items.len, be.deviceName() });

    // warm-up (JIT + weight upload), then timed.
    const got0 = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, null);
    arena.free(got0);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        const g = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, null);
        const b = std.Io.Clock.real.now(io);
        arena.free(g);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    const got = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, null);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        num += (@as(f64, e) - a) * (@as(f64, e) - a);
        den += @as(f64, e) * e;
    }
    try stdout.print("cuda encode: {d:.3} s (best of 3)\n", .{best / 1000.0});
    try stdout.print("cuda-vs-cpu: max_err={d:.5} rel RMSE={d:.5} (max|v|={d:.1})\n", .{ max_err, @sqrt(num / den), max_val });
    try stdout.flush();
}

/// Validate the CUDA text-fusion port against the CPU reference: same DiT, same
/// synthetic conditioning, compare `DiT.textTokens` (CPU) vs `textTokensCuda`.
fn cudaTxtFusionTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, seq_txt: usize) !void {
    const cuda = TensorPencil.gpu.cuda;
    const dit = TensorPencil.models.dit;
    const dit_cuda = TensorPencil.models.dit_cuda;
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("cuda-txtfusion-test needs an int8/int4 convrot checkpoint: {s}\n", .{path});
        return;
    };
    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var model = try dit.DiT.load(arena, .{ .safetensors = &st });
    defer model.deinit();

    // Deterministic conditioning [seq_txt, 12, txt_dim] (encoder-output-scale ~O(1)).
    const cond = try arena.alloc(f32, seq_txt * dit.txt_layers * dit.txt_dim);
    for (cond, 0..) |*c, i| {
        const z: u32 = @truncate(i *% 2654435761 +% 40503);
        c.* = (@as(f32, @floatFromInt(z >> 8)) / @as(f32, 1 << 24) - 0.5) * 2.0;
    }

    const want = try model.textTokens(io, arena, cond, seq_txt);

    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== cuda-txtfusion-test (seq_txt={d}) ==\ncuda device: {s}\n", .{ seq_txt, be.deviceName() });

    const warm = try dit_cuda.textTokensCuda(&model, be, arena, cond); // JIT + upload
    arena.free(warm);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        const g = try dit_cuda.textTokensCuda(&model, be, arena, cond);
        const b = std.Io.Clock.real.now(io);
        arena.free(g);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    const got = try dit_cuda.textTokensCuda(&model, be, arena, cond);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        num += (@as(f64, e) - a) * (@as(f64, e) - a);
        den += @as(f64, e) * e;
    }
    try stdout.print("cuda txtfusion: {d:.3} s (best of 3)\n", .{best / 1000.0});
    try stdout.print("cuda-vs-cpu: max_err={d:.5} rel RMSE={d:.5} (max|v|={d:.1})\n", .{ max_err, @sqrt(num / den), max_val });
    try stdout.flush();
}

fn cudaAttnCmp(arena: std.mem.Allocator, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    const heads = 48;
    const kv = 12;
    const hd = 128;
    const grp = heads / kv;
    const scale: f32 = 1.0 / @sqrt(@as(f32, hd));
    const cfgs = [_]struct { seq: usize, std: f32 }{
        .{ .seq = 256, .std = 0.3 }, .{ .seq = 264, .std = 0.3 },
        .{ .seq = 264, .std = 1.0 }, .{ .seq = 264, .std = 3.0 },
    };
    for (cfgs) |cfg| {
        const seq = cfg.seq;
        const mpad = std.mem.alignForward(usize, seq, 128);
        var prng = std.Random.DefaultPrng.init(7);
        const rnd = prng.random();
        const q = try arena.alloc(f32, mpad * heads * hd);
        const k = try arena.alloc(f32, mpad * kv * hd);
        const v = try arena.alloc(f32, mpad * kv * hd);
        @memset(q, 0);
        @memset(k, 0);
        @memset(v, 0);
        for (0..seq) |s| {
            for (0..heads * hd) |i| q[s * heads * hd + i] = rnd.floatNorm(f32) * cfg.std;
            for (0..kv * hd) |i| {
                k[s * kv * hd + i] = rnd.floatNorm(f32) * cfg.std;
                v[s * kv * hd + i] = rnd.floatNorm(f32) * cfg.std;
            }
        }
        var qd = try be.tensorCreate(q.len * 4);
        defer be.tensorDestroy(&qd);
        var kd = try be.tensorCreate(k.len * 4);
        defer be.tensorDestroy(&kd);
        var vd = try be.tensorCreate(v.len * 4);
        defer be.tensorDestroy(&vd);
        var ob = try be.tensorCreate(mpad * heads * hd * 4);
        defer be.tensorDestroy(&ob);
        var ol = try be.tensorCreate(mpad * heads * hd * 4);
        defer be.tensorDestroy(&ol);
        try be.tensorUpload(qd, std.mem.sliceAsBytes(q));
        try be.tensorUpload(kd, std.mem.sliceAsBytes(k));
        try be.tensorUpload(vd, std.mem.sliceAsBytes(v));
        be.attn_batched = true;
        try be.opAttnTC(qd, kd, vd, ob, seq, heads, kv, hd, scale);
        be.attn_batched = false;
        try be.opAttnTC(qd, kd, vd, ol, seq, heads, kv, hd, scale);
        try be.endBatch();
        const gb = try arena.alloc(f32, mpad * heads * hd);
        const gl = try arena.alloc(f32, mpad * heads * hd);
        try be.tensorDownload(ob, std.mem.sliceAsBytes(gb));
        try be.tensorDownload(ol, std.mem.sliceAsBytes(gl));
        // CPU reference + rel of batched/loop vs cpu, and batched vs loop.
        var nb: f64 = 0;
        var nl: f64 = 0;
        var nbl: f64 = 0;
        var den: f64 = 0;
        const prow = try arena.alloc(f32, seq);
        for (0..seq) |qi| {
            for (0..heads) |h| {
                const kvh = h / grp;
                var mx: f32 = -std.math.inf(f32);
                for (0..seq) |j| {
                    var dot: f32 = 0;
                    for (0..hd) |c| dot += q[(qi * heads + h) * hd + c] * k[(j * kv + kvh) * hd + c];
                    prow[j] = dot * scale;
                    mx = @max(mx, prow[j]);
                }
                var sum: f32 = 0;
                for (0..seq) |j| {
                    prow[j] = @exp(prow[j] - mx);
                    sum += prow[j];
                }
                for (0..seq) |j| prow[j] /= sum;
                for (0..hd) |c| {
                    var acc: f32 = 0;
                    for (0..seq) |j| acc += prow[j] * v[(j * kv + kvh) * hd + c];
                    const idx = (qi * heads + h) * hd + c;
                    const db = @as(f64, gb[idx]) - acc;
                    const dl = @as(f64, gl[idx]) - acc;
                    const dbl = @as(f64, gb[idx]) - gl[idx];
                    nb += db * db;
                    nl += dl * dl;
                    nbl += dbl * dbl;
                    den += @as(f64, acc) * acc;
                }
            }
        }
        try stdout.print("seq={d} std={d:.1} mpad={d}: batched rel {d:.5}, loop rel {d:.5}, batched-vs-loop {d:.5}\n", .{ seq, cfg.std, mpad, @sqrt(nb / den), @sqrt(nl / den), @sqrt(nbl / den) });
        try stdout.flush();
    }
}

/// Validate weight streaming: the CUDA DiT forward with a small --vram-budget
/// (weights evicted+re-uploaded per step) must be BIT-IDENTICAL to the resident
/// forward, and reports the s/step perf loss.
fn cudaStreamTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, lat: usize, budget_gib: f64) !void {
    const dit_mod = TensorPencil.models.dit;
    const dit_cuda = TensorPencil.models.dit_cuda;
    const cuda = TensorPencil.gpu.cuda;

    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var model = try dit_mod.DiT.load(arena, .{ .safetensors = &st });
    defer model.deinit();
    if (model.blocks[0].attn.wq.dtype != .i8) {
        try stdout.print("cuda-stream-test needs the int8 convrot checkpoint\n", .{});
        return;
    }
    const seq_txt: usize = 8;
    const sigma: f32 = 0.7;
    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();
    const cond = try arena.alloc(f32, seq_txt * dit_mod.txt_layers * dit_mod.txt_dim);
    for (cond) |*v| v.* = rand.floatNorm(f32) * 0.5;
    const x = try arena.alloc(f32, dit_mod.channels * lat * lat);
    for (x) |*v| v.* = rand.floatNorm(f32);

    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    // Test the production (synchronous) streaming path.
    // (enableAsyncStreaming() would exercise the dormant, measured-slower async path.)
    try stdout.print("== cuda-stream-test lat={d} ({d}px), budget {d:.2} GiB (sync) ==\ncuda device: {s}\n", .{ lat, lat * 8, budget_gib, be.deviceName() });
    var sess = try dit_cuda.Session.init(arena, io, be, &model, lat, lat, cond, seq_txt);
    defer sess.deinit(be);
    var ws = try dit_cuda.Workspace.init(be, lat, lat, seq_txt);
    defer ws.deinit(be);

    const out_res = try arena.alloc(f32, x.len);
    const out_str = try arena.alloc(f32, x.len);
    const budget: u64 = @intFromFloat(budget_gib * (1 << 30));

    // COLD-LOAD (the real "step 1" cost): first forward after evict, budget=0 so
    // the whole model pins in — this is the ~12 GB host->VRAM upload. SYNC pageable
    // path first.
    be.budget_override = 0;
    be.evictWeights();
    const cold_sync_a = std.Io.Clock.real.now(io);
    try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_res, x, sigma, null);
    const t_cold_sync = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - cold_sync_a.nanoseconds)) / 1e6;
    // Resident timed (weights now loaded): steady per-step.
    var t_res: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_res, x, sigma, null);
        const b = std.Io.Clock.real.now(io);
        t_res = @min(t_res, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }

    // Streamed (small budget): evictWeights first so nothing is pre-resident.
    be.evictWeights();
    be.budget_override = budget;
    try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_str, x, sigma, null);
    var t_str: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_str, x, sigma, null);
        const b = std.Io.Clock.real.now(io);
        t_str = @min(t_str, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    be.budget_override = 0;

    // Async staging-ring COLD-LOAD: bounded 512 MB pinned ring (NOT registerHost —
    // no whole-file page-lock), weights read from the page-cache-backed mmap and
    // DMA'd off the main thread with block-N+1-ahead prefetch overlapping block-N
    // compute. "Lean on the mmap": cold reads fault from disk, warm reads hit page
    // cache (reclaimable, never stalls). Watch RAM externally: VmLck must stay tiny
    // (~512 MB), RSS/page-cache growth is fine. budget=0 pins all (like a real gen).
    const out_astr = try arena.alloc(f32, x.len);
    be.enableAsyncStreaming(io);
    be.evictWeights();
    be.budget_override = 0;
    const cold_a_a = std.Io.Clock.real.now(io);
    try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_astr, x, sigma, null);
    const t_cold_async = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - cold_a_a.nanoseconds)) / 1e6;
    var t_astr: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_astr, x, sigma, null);
        const b = std.Io.Clock.real.now(io);
        t_astr = @min(t_astr, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    be.budget_override = 0;

    var maxdiff: f32 = 0;
    var ndiff: usize = 0;
    for (out_res, out_str) |a, b| {
        const d = @abs(a - b);
        if (d != 0) ndiff += 1;
        maxdiff = @max(maxdiff, d);
    }
    var andiff: usize = 0;
    for (out_res, out_astr) |a, b| if (a != b) {
        andiff += 1;
    };
    try stdout.print("COLD-LOAD (12GB host->VRAM): sync-pageable {d:.3} s  vs  async-staging-ring {d:.3} s ({d:.1}% of sync)\n", .{ t_cold_sync / 1000.0, t_cold_async / 1000.0, t_cold_async / t_cold_sync * 100.0 });
    try stdout.print("resident {d:.3} s/step, streamed(sync) {d:.3} s/step ({d:.1}% slower), streamed(async-ring) {d:.3} s/step ({d:.1}% slower)\n", .{ t_res / 1000.0, t_str / 1000.0, (t_str / t_res - 1.0) * 100.0, t_astr / 1000.0, (t_astr / t_res - 1.0) * 100.0 });
    try stdout.print("streamed vs resident: sync {d} / async {d} / {d} elems differ, max abs diff {d}\n", .{ ndiff, andiff, out_res.len, maxdiff });
    try stdout.flush();
    if (ndiff != 0 or andiff != 0) return error.StreamMismatch; // must be bit-identical
    try stdout.print("cuda weight streaming OK (bit-identical)\n", .{});
}

fn cudaDitTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, lat: usize, use_loop: bool, use_libs: bool) !void {
    const dit_mod = TensorPencil.models.dit;
    const dit_cuda = TensorPencil.models.dit_cuda;
    const cuda = TensorPencil.gpu.cuda;

    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var model = try dit_mod.DiT.load(arena, .{ .safetensors = &st });
    defer model.deinit();
    const wqt = model.blocks[0].attn.wq.dtype;
    if (wqt != .i8 and wqt != .i4 and wqt != .bf16) {
        try stdout.print("cuda-dit-test needs an int8/int4 convrot or bf16 checkpoint (wq.dtype={t})\n", .{wqt});
        return;
    }
    const qtag: []const u8 = switch (wqt) {
        .i4 => "int4",
        .i8 => "int8",
        .bf16 => "bf16",
        else => unreachable,
    };

    const seq_txt: usize = 8;
    const sigma: f32 = 0.7;
    // CPU reference is O(seq^2) and unusably slow past ~256px; gate the rel check.
    const check_cpu = lat <= 32;
    try stdout.print("== cuda-dit-test lat={d} ({d}px), seq~{d} ==\n", .{ lat, lat * 8, seq_txt + (lat / 2) * (lat / 2) });
    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();
    const cond = try arena.alloc(f32, seq_txt * dit_mod.txt_layers * dit_mod.txt_dim);
    for (cond) |*v| v.* = rand.floatNorm(f32) * 0.5;
    const x = try arena.alloc(f32, dit_mod.channels * lat * lat);
    for (x) |*v| v.* = rand.floatNorm(f32);

    const out_cpu = try arena.alloc(f32, x.len);
    if (check_cpu) {
        const t0 = std.Io.Clock.real.now(io);
        try model.forward(io, arena, out_cpu, x, lat, lat, sigma, cond, seq_txt, null);
        const t1 = std.Io.Clock.real.now(io);
        try stdout.print("cpu {s} forward: {d:.1} s\n", .{ qtag, @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e9 });
        try stdout.flush();
    }

    var be = (if (use_libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("cuda device: {s} (kernels: {s})\n", .{ be.deviceName(), if (use_libs) "cuBLASLt/cuDNN libs" else "hand-PTX" });
    var sess = try dit_cuda.Session.init(arena, io, be, &model, lat, lat, cond, seq_txt);
    defer sess.deinit(be);
    var ws = try dit_cuda.Workspace.init(be, lat, lat, seq_txt);
    defer ws.deinit(be);
    if (use_loop) be.attn_batched = false;
    const out_cuda = try arena.alloc(f32, x.len);

    // Warm-up pass (uploads weights, JITs modules).
    try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_cuda, x, sigma, null);
    const reps: usize = if (lat <= 64) 4 else 2;
    // Batched (profile off) timing — the real steady-state s/step.
    var best_ms: f64 = std.math.inf(f64);
    for (0..reps) |_| {
        const ta = std.Io.Clock.real.now(io);
        try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_cuda, x, sigma, null);
        const tb = std.Io.Clock.real.now(io);
        best_ms = @min(best_ms, @as(f64, @floatFromInt(tb.nanoseconds - ta.nanoseconds)) / 1e6);
    }
    try stdout.print("cuda {s} forward: {d:.3} s/step (best of {d}, batched)\n", .{ qtag, best_ms / 1000.0, reps });
    // One profiled (sync-per-op) pass for the per-category breakdown.
    be.profile = true;
    be.prof.reset();
    try dit_cuda.forward(&model, be, &sess, &ws, io, arena, out_cuda, x, sigma, null);
    be.profile = false;
    // matmul/prep/elt + attention (gather/scatter in `attn`, GEMMs/softmax split out).
    const cats = [_][]const u8{ "matmul", "prep", "attn(g/s)", "elt", "  scores", "  softmax", "  pv" };
    var sync_total: f64 = 0;
    for (cats, 0..) |name, i| {
        sync_total += be.prof.ms[i];
        try stdout.print("  {s:<9} {d:>8.1} ms  ({d} launches)\n", .{ name, be.prof.ms[i], be.prof.n[i] });
    }
    try stdout.print("  {s:<9} {d:>8.1} ms  (sync-per-op sum)\n", .{ "total", sync_total });
    try stdout.flush();

    if (check_cpu) {
        var num: f64 = 0;
        var den: f64 = 0;
        var maxabs: f32 = 0;
        for (out_cpu, out_cuda) |a, b| {
            const d = @as(f64, a) - b;
            num += d * d;
            den += @as(f64, a) * a;
            maxabs = @max(maxabs, @abs(a));
        }
        const rel = @sqrt(num / den);
        try stdout.print("DiT velocity: rel RMSE cuda-vs-cpu {d:.5} (max|v|={d:.3})\n", .{ rel, maxabs });
        try stdout.flush();
        // The CPU reference is weight-only (W{4,8}A16: dequant weights, f32
        // activations); CUDA also quantizes activations (W4A4 / W8A8). int8's
        // activation-quant error is tiny (<0.08); int4's 16-level activation
        // quant diverges ~0.15-0.20 from the f32-activation reference — that's
        // the price of W4A4, not a wiring bug (the per-linear sim is bit-exact).
        const tol: f32 = if (wqt == .i4) 0.25 else 0.08;
        if (rel > tol) return error.GpuMismatch;
        try stdout.print("cuda DiT forward OK ({s}; int4 vs W4A16 ref includes activation-quant)\n", .{qtag});
    }
}

fn generate(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    // Flush buffered output on any error return so arg-validation messages
    // ("unknown backend ...", "--prompt is required", ...) actually reach the
    // terminal before the error unwinds past main's final flush.
    errdefer stdout.flush() catch {};
    var opts: TensorPencil.pipeline.Options = .{ .prompt = "" };
    var out_path: []const u8 = "out.png";
    var repeat: usize = 1; // --repeat N: reuse one pipeline.Session for N images (bench cross-queue residency)
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const flag = args[i];
        if (i + 1 >= args.len) {
            try stdout.print("missing value for {s}\n", .{flag});
            return error.InvalidArgs;
        }
        const val = args[i + 1];
        if (std.mem.eql(u8, flag, "--prompt")) {
            opts.prompt = val;
        } else if (std.mem.eql(u8, flag, "--repeat")) {
            repeat = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, flag, "--negative")) {
            opts.negative = val;
        } else if (std.mem.eql(u8, flag, "--width")) {
            opts.width = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, flag, "--height")) {
            opts.height = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, flag, "--steps")) {
            opts.steps = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, flag, "--cfg")) {
            opts.cfg = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, flag, "--shift")) {
            opts.shift = try std.fmt.parseFloat(f32, val);
            opts.explicit_shift = true;
        } else if (std.mem.eql(u8, flag, "--profile")) {
            TensorPencil.models.dit_gpu.profile = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1");
        } else if (std.mem.eql(u8, flag, "--prompt-syntax")) {
            if (std.mem.eql(u8, val, "comfy")) {
                opts.prompt_syntax = .comfy;
            } else if (std.mem.eql(u8, val, "a1111")) {
                opts.prompt_syntax = .a1111;
            } else {
                try stdout.print("unknown prompt syntax '{s}' (expected: comfy, a1111)\n", .{val});
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, flag, "--emphasis")) {
            if (std.mem.eql(u8, val, "original")) {
                opts.emphasis = .original;
            } else if (std.mem.eql(u8, val, "no_norm")) {
                opts.emphasis = .no_norm;
            } else if (std.mem.eql(u8, val, "ignore")) {
                opts.emphasis = .ignore;
            } else {
                try stdout.print("unknown emphasis '{s}' (expected: original, no_norm, ignore)\n", .{val});
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, flag, "--compat")) {
            if (std.mem.eql(u8, val, "comfy")) {
                opts.compat = .comfy;
            } else if (std.mem.eql(u8, val, "a1111")) {
                opts.compat = .a1111;
            } else {
                try stdout.print("unknown compat '{s}' (expected: comfy, a1111)\n", .{val});
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, flag, "--rng")) {
            if (std.mem.eql(u8, val, "cpu")) {
                opts.rng = .torch_cpu;
            } else if (std.mem.eql(u8, val, "nv") or std.mem.eql(u8, val, "gpu")) {
                // "gpu" is accepted because that is what A1111's own dropdown says; its
                // "NV" setting is the same generator computed on the host.
                opts.rng = .nv_philox;
            } else {
                try stdout.print("unknown rng '{s}' (expected: cpu, nv)\n", .{val});
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, flag, "--sgm-noise-mult")) {
            opts.sgm_noise_multiplier = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--quantize-t")) {
            opts.quantize_timestep = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--sampler")) {
            opts.sampler = TensorPencil.sampler.Kind.parse(val) orelse {
                try stdout.print(
                    "unknown sampler '{s}' (expected: euler, dpmpp_2m_sde, dpmpp_2m_sde_heun)\n",
                    .{val},
                );
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--sde-eta")) {
            opts.sde_eta = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, flag, "--sde-s-noise")) {
            opts.sde_s_noise = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, flag, "--scheduler")) {
            opts.scheduler = TensorPencil.sampler.Scheduler.parse(val) orelse {
                try stdout.print("unknown scheduler '{s}' (expected: normal, karras, " ++
                    "exponential, sgm_uniform, simple, ddim_uniform, beta, " ++
                    "linear_quadratic, kl_optimal)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--backend")) {
            opts.backend = TensorPencil.pipeline.Backend.fromStr(val) orelse {
                try stdout.print("unknown backend '{s}' (expected: cpu, vulkan, zig-cuda)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--vae-decode")) {
            opts.vae_decode = std.meta.stringToEnum(TensorPencil.pipeline.VaeDecode, val) orelse {
                try stdout.print("unknown vae decode mode '{s}' (expected: auto, whole, gpu_tiled, cpu_tiled)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--vram-budget")) {
            // "min" = bare-minimum weight residency: hold only the in-flight
            // weights (~2 at a time) and stream everything else. Activations
            // aren't streamable, so total VRAM is still weight-min + activations
            // (resolution-bound); use a small image to get well under 1 GiB.
            if (std.mem.eql(u8, val, "min")) {
                opts.vram_budget = min_vram_budget;
            } else {
                const gib = try std.fmt.parseFloat(f64, val);
                opts.vram_budget = @intFromFloat(gib * (1 << 30));
            }
        } else if (std.mem.eql(u8, flag, "--encoder-f16")) {
            opts.encoder_f16 = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--dit-f32")) {
            TensorPencil.models.dit_gpu.force_f32 = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--dit")) {
            opts.dit_path = val;
        } else if (std.mem.eql(u8, flag, "--vae")) {
            opts.vae_path = val;
            opts.explicit_vae = true;
        } else if (std.mem.eql(u8, flag, "--text-encoder")) {
            opts.text_encoder_path = val;
            opts.explicit_text_encoder = true;
        } else if (std.mem.eql(u8, flag, "--text-encoder-2")) {
            // SDXL's second tower, for a split-file checkpoint. A bundled SDXL
            // checkpoint carries it, so this is an override, not a requirement.
            opts.text_encoder_2_path = val;
            opts.explicit_text_encoder_2 = true;
        } else if (std.mem.eql(u8, flag, "--mmap")) {
            // pread (default) | mmap | buffered — see safetensors.ReadMode.
            TensorPencil.safetensors.read_mode = TensorPencil.safetensors.parseReadMode(val) orelse {
                std.log.err("--mmap: expected pread|mmap|buffered (got '{s}')", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--out")) {
            out_path = val;
        } else if (std.mem.eql(u8, flag, "--taew")) {
            // Preview mode: "none"/"latent2rgb" => latent2rgb; else a taew2_1
            // path => taesd approx-VAE. Saves the last step's preview to
            // <out>.preview.png. (No --taew flag at all => no preview.)
            opts.preview = true;
            if (!std.mem.eql(u8, val, "none") and !std.mem.eql(u8, val, "latent2rgb"))
                opts.taew_path = val;
        } else {
            try stdout.print("unknown flag {s}\n", .{flag});
            return error.InvalidArgs;
        }
    }
    if (opts.prompt.len == 0) {
        try stdout.print("--prompt is required\n", .{});
        return error.InvalidArgs;
    }

    var cap: PreviewCap = .{ .arena = arena };
    if (opts.preview) opts.on_step = .{ .ctx = &cap, .step = PreviewCap.onStep };

    // --repeat N>1: keep ONE pipeline.Session resident across N images (the GUI's
    // cross-queue path) and time each — the model loads only on image 1; images
    // 2..N should skip the reload/weight-upload warmup. seed advances per image.
    if (repeat > 1) {
        var sess = try TensorPencil.pipeline.Session.init(io, arena, opts, stdout);
        defer sess.deinit();
        for (0..repeat) |n| {
            var per = opts;
            per.seed = opts.seed +% n;
            const t0 = std.Io.Clock.real.now(io);
            var im = try sess.generate(per, stdout);
            defer im.deinit(arena);
            const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e9;
            var png: std.ArrayList(u8) = .empty;
            defer png.deinit(arena);
            try TensorPencil.image.encodePngRgb(arena, &png, im.rgb, im.width, im.height);
            const p = try std.fmt.allocPrint(arena, "{s}.{d}.png", .{ out_path, n });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = png.items });
            try stdout.print("[repeat] image {d}/{d} in {d:.1}s -> {s}\n", .{ n + 1, repeat, dt, p });
        }
        return;
    }

    var img = try TensorPencil.pipeline.generate(io, arena, opts, stdout);
    defer img.deinit(arena);

    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(arena);
    try TensorPencil.image.encodePngRgb(arena, &png, img.rgb, img.width, img.height);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = png.items });
    try stdout.print("wrote {s} ({d}x{d})\n", .{ out_path, img.width, img.height });

    if (cap.rgb) |rgb| {
        var ppng: std.ArrayList(u8) = .empty;
        defer ppng.deinit(arena);
        try TensorPencil.image.encodePngRgb(arena, &ppng, rgb, cap.w, cap.h);
        const ppath = try std.fmt.allocPrint(arena, "{s}.preview.png", .{out_path});
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ppath, .data = ppng.items });
        try stdout.print("wrote {s} ({d}x{d}) [taew preview]\n", .{ ppath, cap.w, cap.h });
    }
}

/// Captures the latest per-step preview (for --taew validation).
const PreviewCap = struct {
    arena: std.mem.Allocator,
    rgb: ?[]u8 = null,
    w: usize = 0,
    h: usize = 0,

    fn onStep(ctx: *anyopaque, done: usize, total: usize, preview: ?TensorPencil.pipeline.Preview) void {
        _ = done;
        _ = total;
        const self: *PreviewCap = @ptrCast(@alignCast(ctx));
        const pv = preview orelse return;
        const buf = self.arena.alloc(u8, pv.rgb.len) catch return;
        @memcpy(buf, pv.rgb);
        self.rgb = buf;
        self.w = pv.width;
        self.h = pv.height;
    }
};

/// Decode a planar f32 [16][zh][zw] latent file (VAE space) to a PNG.
fn decodeLatent(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, z_path: []const u8, zh_s: []const u8, zw_s: []const u8, out_path: []const u8) !void {
    const vae = TensorPencil.models.wan_vae;
    const zh = try std.fmt.parseInt(usize, zh_s, 10);
    const zw = try std.fmt.parseInt(usize, zw_s, 10);

    const z = try arena.alloc(f32, vae.latent_channels * zh * zw);
    {
        const file = try std.Io.Dir.cwd().openFile(io, z_path, .{ .mode = .read_only });
        defer file.close(io);
        const got = try file.readPositionalAll(io, std.mem.sliceAsBytes(z), 0);
        if (got != z.len * 4) return error.ShortRead;
    }

    var st = try TensorPencil.SafeTensors.open(arena, io, "models/vae/krea2RealVae_v10.safetensors");
    defer st.deinit();
    var dec = try vae.Decoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();

    const start = std.Io.Clock.real.now(io);
    const planar = try dec.decode(io, arena, z, zh, zw, null);
    const end = std.Io.Clock.real.now(io);

    const px = try TensorPencil.image.planarF32ToRgb8(arena, planar, zw * vae.spatial_scale, zh * vae.spatial_scale);
    var png: std.ArrayList(u8) = .empty;
    try TensorPencil.image.encodePngRgb(arena, &png, px, zw * vae.spatial_scale, zh * vae.spatial_scale);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = png.items });

    const ms = @as(f64, @floatFromInt(end.nanoseconds - start.nanoseconds)) / 1e6;
    try stdout.print("decoded {d}x{d} -> {s} ({d:.0} ms)\n", .{ zw * 8, zh * 8, out_path, ms });
}

/// One DiT-block-sized fp8 GEMM: [m=1024, 6144] x [6144, 6144]^T, CPU and GPU.
fn benchMatmul(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const m = 1024;
    const rows = 6144;
    const cols = 6144;
    var prng = std.Random.DefaultPrng.init(1);
    const rand = prng.random();

    const wbytes = try arena.alloc(u8, rows * cols);
    for (wbytes) |*b| b.* = rand.int(u8) & 0x7e; // fp8 e4m3, NaN-free
    const x = try arena.alloc(f32, m * cols);
    for (x) |*v| v.* = rand.floatNorm(f32);
    const y = try arena.alloc(f32, m * rows);

    const w = TensorPencil.ops.matmul.Weight.init(wbytes, .f8_e4m3, rows, cols);
    const flops: f64 = 2.0 * m * rows * cols;

    for (0..3) |_| {
        const start = std.Io.Clock.real.now(io);
        try TensorPencil.ops.matmul.matmul(io, arena, y, x, m, w, null);
        const end = std.Io.Clock.real.now(io);
        const ns: f64 = @floatFromInt(end.nanoseconds - start.nanoseconds);
        try stdout.print("cpu fp8 GEMM {d}x{d}x{d}: {d:.1} ms, {d:.1} GFLOP/s\n", .{ m, rows, cols, ns / 1e6, flops / ns });
        try stdout.flush();
    }

    var ctx = TensorPencil.gpu.Context.init(arena) catch |err| {
        try stdout.print("gpu unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("gpu: {s}\n", .{ctx.deviceName()});
    const y_ref = try arena.dupe(f32, y);
    for (0..3) |i| {
        const start = std.Io.Clock.real.now(io);
        try ctx.matmul(y, x, m, wbytes, true, rows, cols, 1.0, null);
        const end = std.Io.Clock.real.now(io);
        const ns: f64 = @floatFromInt(end.nanoseconds - start.nanoseconds);
        const tag: []const u8 = if (i == 0) " (incl. weight upload)" else "";
        try stdout.print("gpu fp8 GEMM {d}x{d}x{d}: {d:.1} ms, {d:.1} GFLOP/s{s}\n", .{ m, rows, cols, ns / 1e6, flops / ns, tag });
        try stdout.flush();
    }
    var max_err: f32 = 0;
    for (y_ref, y) |a, b| max_err = @max(max_err, @abs(a - b));
    try stdout.print("gpu vs cpu max err: {d:.6}\n", .{max_err});

    // Cooperative-matrix path at DiT-step shapes (1024px: m_pad 4224).
    if (ctx.pipe_coop != .null_handle) {
        const cm = 4224;
        const xc = try arena.alloc(f32, cm * cols);
        for (xc) |*v| v.* = rand.floatNorm(f32);
        var x_d = try ctx.tensorCreate(cm * cols * 4);
        defer ctx.tensorDestroy(&x_d);
        var y_d = try ctx.tensorCreate(cm * rows * 4);
        defer ctx.tensorDestroy(&y_d);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(xc));
        const cflops: f64 = 2.0 * cm * rows * cols;
        for (0..8) |i| {
            const start = std.Io.Clock.real.now(io);
            try ctx.opMatmulCoop(y_d, x_d, cm, cm, wbytes, rows, cols, 1.0);
            const end = std.Io.Clock.real.now(io);
            const ns: f64 = @floatFromInt(end.nanoseconds - start.nanoseconds);
            const tag: []const u8 = if (i == 0) " (incl. weight upload)" else "";
            try stdout.print("gpu coop GEMM {d}x{d}x{d}: {d:.1} ms, {d:.1} GFLOP/s{s}\n", .{ cm, rows, cols, ns / 1e6, cflops / ns, tag });
            try stdout.flush();
        }
    }
}

test "library module is reachable" {
    try std.testing.expectEqual(@as(usize, 4), TensorPencil.DType.f32.byteSize());
}
