//! TensorPencil CLI, thin driver over the TensorPencil library module.

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
    // Flush on the ERROR path too. The `flush` at the bottom of this function is
    // never reached when a subcommand returns, so every "print a diagnostic then
    // return the error" site printed into a buffer that was thrown away, and the
    // user saw a bare stack trace instead of the sentence explaining it.
    defer stdout.flush() catch {};

    if (args.len >= 2 and std.mem.eql(u8, args[1], "gpu-test")) {
        // Survey cooperative-matrix configs (incl. int8 tensor cores) on init.
        TensorPencil.gpu.context.dump_coop_configs = true;
        var ctx = try TensorPencil.gpu.Context.init(arena, io);
        defer ctx.deinit();
        try stdout.print("device: {s}\n", .{ctx.deviceName()});
        try stdout.print("coop matrix f16->f32:  {d}x{d}x{d}\n", .{ ctx.coop_m, ctx.coop_n, ctx.coop_k });
        try stdout.print("warp tile f16: {d}x{d} (wg {d}x{d}); int8: wg {d}x{d}\n", .{
            ctx.coop_wg_m / 2, ctx.coop_wg_n / 2, ctx.coop_wg_m, ctx.coop_wg_n, ctx.coop_i8_wg_m, ctx.coop_i8_wg_n,
        });
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-bqdec-test")) {
        const cuda = TensorPencil.gpu.cuda;
        var ctx = cuda.Context.init(arena) catch |err| {
            try stdout.print("cuda unavailable: {t}\n", .{err});
            return;
        };
        defer ctx.deinit();
        try stdout.print("cuda device: {s}\n", .{ctx.deviceName()});
        try cuda.kernels.blockQDecodeTest(&ctx, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-fp8-test")) {
        try cudaFp8Test(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-encode-test")) {
        try cudaEncodeTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "sd-cuda-test")) {
        // `libs` selects the cuBLASLt/cuDNN arm, which takes a DIFFERENT attention
        // path (true head width, no padding), so both are worth running.
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
            } else if (std.mem.eql(u8, args[i], "--anima")) {
                // The variant decides the CONFIG (0.6B: 28 layers, hidden 1024),
                // so a 0.6B encoder under the default `.zimage` variant is a
                // `ShapeMismatch` at load rather than a wrong answer. This flag is how
                // Anima's encoder gets checked on every backend.
                variant = .anima;
            } else if (std.mem.eql(u8, args[i], "--minimax-h3")) {
                // 50 layers, hidden 5120, theta 5e6 and NO final norm. Pair with
                // `TP_TE_VISION=1` to compare the deepstack/mrope path too.
                variant = .minimax_h3;
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "anima-cuda-test")) {
        var ckpt: []const u8 = "/home/qt/genai/comfyui/models/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors";
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) libs = true else ckpt = a;
        }
        try animaCudaTest(arena, io, stdout, ckpt, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "tune-coop")) {
        try tuneCoop(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-norm-bench")) {
        try vkNormBench(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "anima-vk-bench")) {
        var seq: usize = 6534;
        for (args[2..]) |a| seq = std.fmt.parseInt(usize, a, 10) catch seq;
        try animaVkBench(arena, io, stdout, seq);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "anima-cuda-bench")) {
        var seq: usize = 6534;
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) libs = true else seq = std.fmt.parseInt(usize, a, 10) catch seq;
        }
        try animaCudaBench(arena, io, stdout, seq, libs);
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "dump-ptx")) {
        try dumpPtx(arena, io, stdout, if (args.len >= 3) args[2] else "", if (args.len >= 4) args[3] else "");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-dit-test")) {
        const path = if (args.len >= 3) args[2] else "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
        const lat: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 32) else 32;
        var loop = false;
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "loop")) loop = true;
            if (std.mem.eql(u8, a, "libs")) libs = true;
            // A GGUF DiT has two decode routes and they are different GEMMs, so the
            // device check has to be able to name one rather than only testing `auto`.
            if (std.mem.eql(u8, a, "int8")) TensorPencil.models.dit_cuda.blockq_gemm = .int8;
            if (std.mem.eql(u8, a, "int4")) TensorPencil.models.dit_cuda.blockq_gemm = .int4;
            if (std.mem.eql(u8, a, "f16")) TensorPencil.models.dit_cuda.blockq_gemm = .f16;
            if (std.mem.eql(u8, a, "mmq")) TensorPencil.models.dit_cuda.blockq_gemm = .mmq;
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "generate-clip")) {
        try generateClip(arena, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vae-encode-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else "/home/qt/genai/comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors";
        const libs = for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) break true;
        } else false;
        try minimaxH3VaeEncodeCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-audio-encode-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else "/home/qt/genai/comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors";
        const libs = for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) break true;
        } else false;
        try minimaxH3AudioEncodeCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-audio-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else "/home/qt/genai/comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors";
        var libs = false;
        for (args[2..]) |a| if (std.mem.eql(u8, a, "libs")) {
            libs = true;
        };
        try minimaxH3AudioCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lora-cuda-test")) {
        try loraCudaTest(arena, io, stdout, args.len >= 3 and std.mem.eql(u8, args[2], "libs"));
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vae-cuda-test")) {
        try minimaxH3VaeCudaTest(arena, io, stdout, args.len >= 3 and std.mem.eql(u8, args[2], "libs"));
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-cuda-test")) {
        const ck = if (args.len >= 3) args[2] else "/home/qt/genai/comfyui/models/diffusion_models/h3/10erosMaxInt8Ref2va_v10Beta.safetensors";
        const libs = args.len >= 4 and std.mem.eql(u8, args[3], "libs");
        try minimaxH3CudaTest(arena, io, stdout, ck, libs);
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
            \\                         int8|int4 convrot / w4a8 / dense bf16
            \\                         --dit ckpt)
            \\      --vram-budget 0    GiB of device memory to use (0 = ask the
            \\                         driver); weights past it stream per step.
            \\                         "min" holds only the in-flight weights
            \\                         (~2 at a time) — lowest VRAM, but streams
            \\                         every weight each step (slow; pair w/ a
            \\                         small image for sub-GiB total)
            \\      --encoder-f16 off  run the text encoder GEMMs on tensor
            \\                         cores (f16): ~0.4s faster, slightly less
            \\                         exact conditioning (on/off)
            \\      --dit-gguf-gemm auto  which GEMM a GGUF block-quant DiT
            \\                         decodes its weights for:
            \\                         auto | int8 | int4 | f16.
            \\                         int8 rotates and re-quantizes to convrot
            \\                         int8, which is ~2x faster but caps accuracy
            \\                         at int8's; int4 does the same one width down
            \\                         for the s4 tensor cores; f16 expands the
            \\                         weight and keeps the format's own. auto picks
            \\                         int8 for q2_k/q4_k (whose own error already
            \\                         dominates the regrid) and f16 for the rest.
            \\                         int4 is opt-in even for q2_k: it is W4A4,
            \\                         and the 4-bit activations cost more than the
            \\                         weight regrid saves. Only q2_k/q4_k/q8_0
            \\                         have an int8 or int4 decode
            \\      --dit-f32 off      run the krea2 DiT in full f32 instead of the
            \\                         f16 tensor-core path (slower, more exact)
            \\                         (on/off). VULKAN and krea2 ONLY: it sets
            \\                         models.dit_gpu.force_f32, so it is a no-op on
            \\                         the CUDA backends and on every other family
            \\      --dit <path>       diffusion checkpoint (fp8 / int8 / int4
            \\                         convrot / asym_w4a8_int8 / nvfp4 / dense
            \\                         bf16 / GGUF; auto-detected). w4a8 and nvfp4
            \\                         keep 4-bit weights resident and decode per
            \\                         GEMM (w4a8 -> int8, nvfp4 -> f16, so nvfp4
            \\                         is ~2x slower at the same VRAM on pre-
            \\                         Blackwell cards). A GGUF block quant does
            \\                         the same on --backend cuda / zig-cuda,
            \\                         decoding per GEMM (see --dit-gguf-gemm). The
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
            \\  TensorPencil te-test [<encoder>] [--ref <encoder>] [--krea2|--anima]
            \\      check a Qwen3 text encoder on every GPU backend against its
            \\      own CPU forward (kernel check), and optionally against a
            \\      second encoder (quantization check). Accepts .gguf
            \\  TensorPencil zimage-cuda-test [<zimage ckpt>] [libs]
            \\      check zimage_cuda's device forward against the CPU forward on
            \\      real weights, on both attention paths; non-zero if any fails
            \\  TensorPencil anima-cuda-test [<anima ckpt>] [libs]
            \\      check anima_cuda's new kernels (lnMod, rectangular TC
            \\      attention) against the CPU ops they reproduce, then its whole
            \\      device forward against anima.DiT.predict on real weights, on
            \\      both attention paths; non-zero if any fails
            \\  TensorPencil vk-norm-bench
            \\      thread-per-row vs subgroup weighted RMSNorm at every shape the
            \\      three Vulkan DiTs use, at the sizes they render at
            \\  TensorPencil anima-vk-bench [<seq>]
            \\      the Vulkan counterpart: bf16 vs int8 GEMMs at identical shapes,
            \\      the int8 prep, attention and elementwise, each against a
            \\      ceiling (TFLOP/s or GB/s)
            \\  TensorPencil anima-cuda-bench [<seq>] [libs]
            \\      time one Anima trunk step's device work at real shapes, split
            \\      GEMM / attention / elementwise, each against a ceiling
            \\      (TFLOP/s or GB/s) rather than a share of the total
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
    var ctx = try TensorPencil.gpu.Context.init(arena, io);
    defer ctx.deinit();
    try stdout.print("device: {s}\n", .{ctx.deviceName()});
    if (ctx.pipe_coop_i8 == .null_handle and ctx.pipe_coop_i8_sh == .null_handle) {
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
            try ctx.opMatmulCoopI8(y_d, x_d, m, null, wb, n, k);
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
        // diverges ~0.4%, but that f16 rotation error stays WITHIN int8 quant
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

            const ok = try ctx.opMatmulCoopI8Fused(y_d, x_d, m, null, wb, rows, cols, s_d.buf, false);
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
/// bindings->handle end to end (the analog of `cuda-test` for the driver path).
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
/// integer matmul (s8*s8->s32 is exact, so 0 mismatches expected), then a min-of-N
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

/// Phase-2 milestone 2.2: cuBLASLt f16 GEMM (HMMA, f32 accumulate), the drop-in
/// for the hand-PTX `buildHgemm` behind the fp8 encoder GEMMs and the VAE convs.
/// Validates D[m][n] f32 = A[m][k] @ W[n][k]ᵀ vs a CPU f32-accumulate reference
/// (f16 inputs widen exactly; only the reduction order differs -> f16 regime),
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
/// op graph). Validates D = (A*B in s32)*act_scale*weight_scale vs a CPU
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
/// backend-graph SDPA op in ISOLATION, synthetic f16 Q/K/V (GQA), non-causal,
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

        var plan = cudnn.SdpaPlan.build(api, handle, 1, c.hq, c.hkv, c.s, c.s, c.d) catch |err| {
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
        var plan = try cudnn.SdpaPlan.build(api, handle, 1, hq, hkv, t.s, t.s, d);
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
/// lands within 0.3 dB of the Vulkan arm, real, but it localizes nothing.
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
            // kernel produced it, and "the reference overflowed" is a completely
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
            // every line it had already produced, i.e. exactly the diagnostic.
            try self.out.flush();
        }
    };
    const rep: R = .{ .out = stdout, .fails = &failures };

    var a_d: Buf = .{};
    var b_d: Buf = .{};
    var c_d: Buf = .{};
    var d_d: Buf = .{};
    var e_d: Buf = .{};
    // The `inline for` is evaluated against the whole (long) rest of the function.
    @setEvalBranchQuota(10000);
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

    // --- f16 activation-storage twins ----------------------------------------
    // The SD UNet carries its stream as f16 under `--backend cuda`
    // (`sd_unet_cuda.Workspace.act_f16`), which routes every op below through a
    // separate kernel. Each is checked against the SAME CPU reference as its f32
    // twin, at f16's own tolerance: a wrong one shows up as a whole-forward NaN
    // with nothing to say which kernel produced it.
    {
        const h16 = struct {
            fn upH(bk: *cuda.Backend, buf: *Buf, host: []const f32, arena2: std.mem.Allocator) !void {
                const half = try arena2.alloc(u16, host.len);
                for (host, half) |v, *o| o.* = @bitCast(@as(f16, @floatCast(v)));
                try bk.ensureDeviceBuffer(buf, half.len * 2);
                try bk.tensorUpload(buf.*, std.mem.sliceAsBytes(half));
            }
            fn down(bk: *cuda.Backend, buf: Buf, out: []f32, arena2: std.mem.Allocator) !void {
                const half = try arena2.alloc(u16, out.len);
                try bk.tensorDownload(buf, std.mem.sliceAsBytes(half));
                for (half, out) |v, *o| o.* = @as(f16, @bitCast(v));
            }
        };

        // group_norm over f16 storage (stats stay f32).
        {
            const groups = 32;
            const n = 130;
            const ch = 320;
            const x = try arena.alloc(f32, n * ch);
            for (x) |*v| v.* = rnd.floatNorm(f32);
            const cat = try arena.alloc(f32, 2 * ch);
            for (cat) |*v| v.* = rnd.floatNorm(f32);
            try h16.upH(be, &a_d, x, arena);
            try up(be, &c_d, cat);
            try be.ensureDeviceBuffer(&b_d, n * ch * 2);
            try be.ensureDeviceBuffer(&d_d, groups * 32 * 3 * 4);
            try be.ensureDeviceBuffer(&e_d, groups * 2 * 4);
            try be.opGroupNorm(a_d, b_d, c_d, d_d, e_d, n, ch, groups, 32, 1e-5, true, true);
            const got = try arena.alloc(f32, n * ch);
            try h16.down(be, b_d, got, arena);
            const want = try arena.alloc(f32, n * ch);
            // The reference reads the SAME f16 values the kernel does, so this
            // measures the kernel, not the round trip into f16.
            const xh = try arena.alloc(f32, n * ch);
            for (x, xh) |v, *o| o.* = @floatCast(@as(f16, @floatCast(v)));
            ops.norm.groupNorm(want, xh, ch, groups, cat[0..ch], cat[ch..], 1e-5);
            ops.act.silu(want);
            try rep.check("group_norm h16 (f16 storage, silu)", want, got, 2e-3);
        }

        // layer_norm over f16 storage (weight/bias stay f32).
        {
            const rows = 61;
            const dim = 320;
            const x = try arena.alloc(f32, rows * dim);
            for (x) |*v| v.* = rnd.floatNorm(f32);
            const wv = try arena.alloc(f32, dim);
            const bv = try arena.alloc(f32, dim);
            for (wv) |*v| v.* = rnd.floatNorm(f32);
            for (bv) |*v| v.* = rnd.floatNorm(f32);
            try h16.upH(be, &a_d, x, arena);
            try be.ensureDeviceBuffer(&b_d, rows * dim * 2);
            try be.opLayerNorm(a_d, b_d, wv, bv, rows, dim, 1e-5, true);
            const got = try arena.alloc(f32, rows * dim);
            try h16.down(be, b_d, got, arena);
            const want = try arena.alloc(f32, rows * dim);
            for (0..rows) |r| {
                const row = x[r * dim ..][0..dim];
                var mean: f64 = 0;
                for (row) |v| mean += @as(f32, @floatCast(@as(f16, @floatCast(v))));
                mean /= @floatFromInt(dim);
                var vr: f64 = 0;
                for (row) |v| {
                    const d = @as(f64, @as(f32, @floatCast(@as(f16, @floatCast(v))))) - mean;
                    vr += d * d;
                }
                vr /= @floatFromInt(dim);
                const inv = 1.0 / @sqrt(vr + 1e-5);
                for (0..dim) |j| {
                    const v = @as(f64, @as(f32, @floatCast(@as(f16, @floatCast(row[j])))));
                    want[r * dim + j] = @floatCast((v - mean) * inv * wv[j] + bv[j]);
                }
            }
            try rep.check("layer_norm h16 (f16 storage)", want, got, 3e-3);
        }

        // geglu over f16 storage.
        {
            const n = 37;
            const inner = 96;
            const src = try arena.alloc(f32, n * 2 * inner);
            for (src) |*v| v.* = rnd.floatNorm(f32) * 3;
            try h16.upH(be, &a_d, src, arena);
            try be.ensureDeviceBuffer(&b_d, n * inner * 2);
            try be.opGeglu(a_d, b_d, n, inner, true);
            const got = try arena.alloc(f32, n * inner);
            try h16.down(be, b_d, got, arena);
            const want = try arena.alloc(f32, n * inner);
            for (0..n) |pi| {
                const row = src[pi * 2 * inner ..][0 .. 2 * inner];
                for (0..inner) |j| {
                    const val: f32 = @floatCast(@as(f16, @floatCast(row[j])));
                    const gate: f32 = @floatCast(@as(f16, @floatCast(row[inner + j])));
                    want[pi * inner + j] = val * ops.act.geluErfScalar(gate);
                }
            }
            try rep.check("geglu h16 (f16 storage)", want, got, 3e-3);
        }

        // add, concat_ch, add_bias_rows over f16 storage.
        {
            const n = 41;
            const ch = 48;
            const av = try arena.alloc(f32, n * ch);
            const bv = try arena.alloc(f32, n * ch);
            for (av) |*v| v.* = rnd.floatNorm(f32);
            for (bv) |*v| v.* = rnd.floatNorm(f32);
            try h16.upH(be, &a_d, av, arena);
            try h16.upH(be, &b_d, bv, arena);
            try be.opAddH16(a_d, b_d, n * ch);
            const got = try arena.alloc(f32, n * ch);
            try h16.down(be, a_d, got, arena);
            const want = try arena.alloc(f32, n * ch);
            for (av, bv, want) |x, y, *o| o.* = @floatCast(@as(f16, @floatCast(x)) + @as(f16, @floatCast(y)));
            try rep.check("add h16 (f16 storage)", want, got, 2e-3);

            // concat_ch: the skip and the stream side by side in one row.
            const total = ch * 2;
            try h16.upH(be, &a_d, av, arena);
            try h16.upH(be, &b_d, bv, arena);
            try be.ensureDeviceBuffer(&c_d, n * total * 2);
            try be.opConcatCh(a_d, c_d, n, ch, total, 0, true);
            try be.opConcatCh(b_d, c_d, n, ch, total, ch, true);
            const cgot = try arena.alloc(f32, n * total);
            try h16.down(be, c_d, cgot, arena);
            const cwant = try arena.alloc(f32, n * total);
            for (0..n) |r| {
                for (0..ch) |j| {
                    cwant[r * total + j] = @floatCast(@as(f16, @floatCast(av[r * ch + j])));
                    cwant[r * total + ch + j] = @floatCast(@as(f16, @floatCast(bv[r * ch + j])));
                }
            }
            try rep.check("concat_ch h16 (f16 storage)", cwant, cgot, 0);

            // add_bias_rows: the ResBlock timestep projection, f32, broadcast down
            // an f16 activation.
            const off = 7;
            const proj = try arena.alloc(f32, off + ch);
            for (proj) |*v| v.* = rnd.floatNorm(f32);
            try h16.upH(be, &a_d, av, arena);
            try up(be, &d_d, proj);
            try be.opAddBiasRows(a_d, d_d, n, ch, off, true);
            try h16.down(be, a_d, got, arena);
            for (0..n) |r| for (0..ch) |j| {
                want[r * ch + j] = @floatCast(@as(f16, @floatCast(av[r * ch + j])) + @as(f16, @floatCast(proj[off + j])));
            };
            try rep.check("add_bias_rows h16 (f32 bias, f16 dst)", want, got, 2e-3);
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
        try be.opGeglu(a_d, b_d, n, inner, false);
        const got = try arena.alloc(f32, n * inner);
        try be.tensorDownload(b_d, std.mem.sliceAsBytes(got));
        // 2e-5, not the 1e-6 the SPIR-V twin holds: every kernel in `cuda/elt.zig`
        // computes exp as `ex2.approx(x*log2e)` (its header says so), which is an
        // approximate instruction, where SPIR-V's `@exp` is not. Measured 6.6e-6,
        // two orders below the f16 GEMM error either side of it, so the two
        // backends legitimately disagree here and neither is wrong.
        try rep.check("geglu (erf gate, value-first halves)", want, got, 2e-5);
    }

    // --- cross-attention (seq_kv != seq_q) -----------------------------------
    // Both entry points over every head width the SD family uses, at a query count
    // past one tile of the rectangular path. The naive kernel is f32 throughout;
    // `opAttnCross` runs tensor cores (cuDNN here, hand-PTX under zig-cuda) and
    // carries f16's error, hence the two tolerances.
    {
        const ctx_seq = 77; // the SD family's conditioning length
        const Shape = struct { n: usize, heads: usize, hd: usize };
        for ([_]Shape{
            .{ .n = 100, .heads = 8, .hd = 40 }, // SD1.5 outermost
            .{ .n = 300, .heads = 8, .hd = 80 },
            .{ .n = 300, .heads = 8, .hd = 160 }, // SD1.5 innermost, pads to 256
            .{ .n = 700, .heads = 10, .hd = 64 }, // SDXL, every level
        }) |s| {
            const dim = s.heads * s.hd;
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(s.hd)));
            const q = try arena.alloc(f32, s.n * dim);
            const k = try arena.alloc(f32, ctx_seq * dim);
            const v = try arena.alloc(f32, ctx_seq * dim);
            for (q) |*x| x.* = rnd.floatNorm(f32);
            for (k) |*x| x.* = rnd.floatNorm(f32);
            for (v) |*x| x.* = rnd.floatNorm(f32);
            try up(be, &a_d, q);
            try up(be, &b_d, k);
            try up(be, &c_d, v);
            try be.ensureDeviceBuffer(&d_d, s.n * dim * 4);
            const want = try arena.alloc(f32, s.n * dim);
            try ops.attention.attention(io, arena, want, q, k, v, .{
                .seq_q = s.n,
                .seq_kv = ctx_seq,
                .n_heads = s.heads,
                .n_kv_heads = s.heads,
                .head_dim = s.hd,
            });
            const got = try arena.alloc(f32, s.n * dim);
            var name_buf: [96]u8 = undefined;

            try be.opAttnCrossNaive(a_d, b_d, c_d, d_d, s.n, ctx_seq, s.heads, s.hd, scale);
            try be.tensorDownload(d_d, std.mem.sliceAsBytes(got));
            try rep.check(try std.fmt.bufPrint(&name_buf, "attn_cross naive (q={d}, kv=77, hd={d})", .{ s.n, s.hd }), want, got, 1e-5);

            try be.opAttnCross(a_d, b_d, c_d, d_d, s.n, ctx_seq, s.heads, s.hd, scale);
            try be.tensorDownload(d_d, std.mem.sliceAsBytes(got));
            try rep.check(try std.fmt.bufPrint(&name_buf, "attn_cross tensor-core (q={d}, kv=77, hd={d})", .{ s.n, s.hd }), want, got, 3e-3);
        }
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
        try be.opConcatCh(a_d, c_d, n, ch_a, total, 0, false);
        try be.opConcatCh(b_d, c_d, n, ch_b, total, ch_a, false);
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
        try be.opAddBiasRows(a_d, b_d, n, ch, off, false);
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
                .skip_hw = try arena.alloc([2]usize, 0),
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
    // ONE head 512 wide over every latent position, nothing like the UNet's
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
                // grow with |q|*|k|, so an f16 GEMM feeding a softmax lands at ~6e-3
                // relative at every seq length tested (1.0 holds 6e-4). What this arm
                // is for is catching `inf`/NaN, which is what the SDXL VAE actually
                // hit, a rel-L2 bound would report that as `nan`, not as 0.006.
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
            try sd_unet_cuda.convInto(be, &patch, &b_d, &a_d, h, w, cv, mode, null);
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
            var name_buf: [80]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "conv 3x3 {t} (im2col_sd + f16 GEMM)", .{mode});
            try rep.check(name, want, got, 2e-3);

            // The same convolution with f16 activation STORAGE on both sides, which
            // is how the SD UNet runs it under `--backend cuda`. Both the cuDNN and
            // the banded arm are covered: `convIntoPrec` picks by mode.
            {
                const inh = try arena.alloc(u16, in.len);
                for (in, inh) |v, *o| o.* = @bitCast(@as(f16, @floatCast(v)));
                try be.ensureDeviceBuffer(&c_d, inh.len * 2);
                try be.tensorUpload(c_d, std.mem.sliceAsBytes(inh));
                try be.ensureDeviceBuffer(&d_d, oh * ow * co * 2);
                try sd_unet_cuda.convIntoPrec(be, &patch, &d_d, &c_d, h, w, cv, mode, null, 1.0, true, true);
                const oh16 = try arena.alloc(u16, oh * ow * co);
                try be.tensorDownload(d_d, std.mem.sliceAsBytes(oh16));
                const goth = try arena.alloc(f32, oh * ow * co);
                for (oh16, goth) |v, *o| o.* = @as(f16, @bitCast(v));
                const nameh = try std.fmt.bufPrint(&name_buf, "conv 3x3 {t} h16 (f16 storage)", .{mode});
                try rep.check(nameh, want, goth, 5e-3);
            }
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
        // is `ex2.approx`, an approximate instruction, per that file's header. Do not
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
    // the VAE below is not, the two families' decoders are architecturally
    // IDENTICAL and differ only in weights and in `scaling_factor`, which `decode`
    // never sees, so an SDXL checkpoint still exercises the sweep, which is how
    // that family's fp16 behaviour gets measured at all.
    const family = TensorPencil.pipeline.detectFamily(store) catch .sd15;
    try stdout.print("checkpoint family: {t}\n", .{family});

    // --- the CLIP text tower, whole, on a weighted two-chunk prompt -----------
    // Both towers when the checkpoint has both, since they differ in naming (fused
    // q/k/v), depth, head count AND activation, the SDXL arm is the one that
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
            .krea2, .zimage, .anima, .minimax_h3 => &.{},
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
            // f16 tensor-core GEMMs against an f32 CPU forward over 12/32 layers,
            // the same regime as the whole-UNet check below, and the per-kernel checks
            // above are what pin the arithmetic.
            try rep.check(name, want, got, 2e-2);
        }
    }
    if (family == .sd15) {
        var unet = try sd_unet.UNet.load(arena, store, sd_unet.sd15, "model.diffusion_model.");
        defer unet.deinit();
        const ctx_seq = 77;
        const context = try arena.alloc(f32, ctx_seq * 768);
        for (context) |*v| v.* = rnd.floatNorm(f32) * 0.5;
        const t: f32 = 481;
        var sess = try sd_unet_cuda.Session.init(arena, be, &unet, context, ctx_seq, null);
        defer sess.deinit(be);

        // 16x16 halves cleanly at every level; 22x18 does not (22 -> 11 -> 6 -> 3 and
        // 18 -> 9 -> 5 -> 3), so the decoder's upsample has to land on the skip's grid
        // rather than twice its own. Only the second shape exercises that.
        //
        // 17x23 is ODD on both axes, which is what a real ComfyUI resolution preset
        // produces (1032x1552 -> 129x194) and what `pipeline` used to reject outright.
        // It halves rounding UP from the very first level, so every skip in the decoder
        // is one row or column short of twice its own grid.
        for ([_][2]usize{ .{ 16, 16 }, .{ 16, 24 }, .{ 22, 18 }, .{ 17, 23 } }) |lat| {
            const lat_h = lat[0];
            const lat_w = lat[1];
            const n = lat_h * lat_w;
            const x = try arena.alloc(f32, n * 4);
            for (x) |*v| v.* = rnd.floatNorm(f32);

            var cpu_ws = try sd_unet.Workspace.init(arena, &unet, lat_h, lat_w, ctx_seq);
            defer cpu_ws.deinit();
            const want = try arena.alloc(f32, n * 4);
            try sd_unet.forward(&unet, io, arena, &cpu_ws, want, x, lat_h, lat_w, t, context, ctx_seq, null);

            var ws = try sd_unet_cuda.Workspace.init(arena, be, &unet, lat_h, lat_w, ctx_seq);
            defer ws.deinit(be);
            const got = try arena.alloc(f32, n * 4);
            try sd_unet_cuda.forward(&unet, be, &sess, &ws, io, arena, got, x, lat_h, lat_w, t, null);
            // A whole UNet of f16 GEMMs; the per-kernel checks above are what pin the
            // arithmetic, this pins the wiring.
            var name: [64]u8 = undefined;
            try rep.check(
                try std.fmt.bufPrint(&name, "sd_unet_cuda.forward (SD1.5, {d}x{d} latent)", .{ lat_h, lat_w }),
                want,
                got,
                2e-2,
            );
        }
    }

    {
        var dec = try sd_vae.Decoder.load(arena, store, if (family == .sdxl) sd_vae.sdxl else sd_vae.sd15, "first_stage_model.");
        defer dec.deinit();
        // A SWEEP, not one size, and that is the point: this check passed at
        // 12x10 for months while a 96x96 latent (a 768-square render) decoded to a
        // solid white image. Anything that reduces over positions, GroupNorm's
        // chunked Welford statistics, the conv's banding, the attention tiling,
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
        // NOT `arena`: the CPU reference frees every intermediate it allocates,
        // but an arena only returns memory at scope exit, so the sweep accumulates
        // every buffer of every size and the OOM killer takes the process at 128²
        // (~2 GB per activation there). A freeing allocator keeps the peak at the
        // few live buffers the decoder actually holds.
        const ref_gpa = std.heap.page_allocator;
        // The MAGNITUDE arm is as load-bearing as the size arm. `decode` is
        // handed `z / scaling_factor`, so a real latent arrives ~5.5x (SD1.5) to
        // ~7.7x (SDXL) wider than the unit gaussian a fixture naturally uses, and
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
/// A CLI command rather than a test for the usual reason, the test binary brings
/// up no CUDA/Vulkan context, and because an 8 GB encoder is not a unit test.
///
/// The two questions it separates are genuinely different and were being
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
    // `TP_TE_TAP=n` truncates the encode to n layers. The ISOLATION for a
    // device-vs-CPU number that looks large: if the divergence grows smoothly with
    // depth it is int8 accumulation, and if it jumps at a particular depth it is a
    // defect in one layer's arithmetic. A single figure at the full depth cannot
    // tell those apart.
    var enc = if (std.c.getenv("TP_TE_TAP")) |v| blk: {
        const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch variant.config().n_layers;
        const taps = try arena.alloc(usize, 1);
        taps[0] = n;
        var cfg = variant.config();
        cfg.n_layers = n;
        try stdout.print("(diagnostic: truncated to {d} layers)\n", .{n});
        break :blk try qwen3.TextEncoder.loadVariantOverride(arena, ct.store(), variant, .{ .cfg = cfg, .taps = taps });
    } else try qwen3.TextEncoder.loadVariant(arena, ct.store(), variant);
    defer enc.deinit();
    try stdout.print("== te-test ==\nencoder : {s}\n          {d} layers, hidden {d}, rope_theta {d:.0}, q dtype {t}, {d} tokens\n", .{
        path, enc.cfg.n_layers, enc.cfg.hidden, enc.cfg.rope_theta, enc.layers[0].q.dtype, ids.items.len,
    });

    // `TP_TE_VISION=1` adds a spliced vision payload: pre-embedded block rows over
    // the placeholder tokens, multimodal rope, and DeepStack features. That is the
    // fl2va/ref2va path, and it exercises three things the text-only compare cannot
    // -- the row splice, the mrope table, and the per-layer injection at image rows.
    //
    // Synthetic rather than from a real ViT on purpose: this compares the DEVICE
    // encode against the HOST one on the same payload, so where the rows came from
    // is irrelevant, and a real tower would drag a 27 GB vision load in.
    var vision: qwen3.TextEncoder.Vision = .{};
    const want_vision = std.c.getenv("TP_TE_VISION") != null;
    if (want_vision) {
        const h = enc.cfg.hidden;
        const at: usize = @min(4, ids.items.len - 1);
        const n_rows: usize = @min(16, ids.items.len - at);
        var prng = std.Random.DefaultPrng.init(0x7e5100);
        const rnd = prng.random();

        const rows = try arena.alloc(f32, n_rows * h);
        for (rows) |*v| v.* = rnd.floatNorm(f32) * 0.5;
        const blocks = try arena.alloc(qwen3.TextEncoder.Vision.Block, 1);
        blocks[0] = .{ .at = at, .rows = rows };

        const spans = try arena.alloc(qwen3.TextEncoder.Vision.Span, 1);
        spans[0] = .{ .start = at, .len = n_rows };

        // Three features, the real DeepStack count, injected into layers 0/1/2.
        const n_ds: usize = 3;
        const ds = try arena.alloc([]const f32, n_ds);
        for (ds) |*d| {
            const buf = try arena.alloc(f32, n_rows * h);
            for (buf) |*v| v.* = rnd.floatNorm(f32) * 0.3;
            d.* = buf;
        }

        // mrope positions for one block, from the same builder the pipeline uses.
        const pos = try arena.alloc(f32, 3 * ids.items.len);
        // A square merged grid covering the block, which is what a real reference
        // image of this token count would have produced.
        var g: usize = 1;
        while (g * g < n_rows) g += 1;
        const img = [_]TensorPencil.models.minimax_h3_vit.ImageSpan{.{
            .index = at,
            .size = n_rows,
            .grid_h = 2 * g,
            .grid_w = 2 * g,
        }};
        try TensorPencil.models.minimax_h3_vit.mropePositions(pos, ids.items.len, &img);

        vision = .{
            .blocks = blocks,
            .positions = pos,
            .rope_dims = enc.cfg.rope_dims,
            .deepstack = ds,
            .inject = spans,
        };
        try stdout.print("vision  : 1 block of {d} rows at {d}, {d} deepstack features, mrope on\n", .{ n_rows, at, n_ds });
    }

    var t0 = std.Io.Clock.real.now(io);
    const want = try enc.encodeVision(io, arena, ids.items, vision, null);
    var ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
    try stdout.print("\ncpu       {d:8.0} ms   (reference)\n", .{ms});

    // The payload must not be inert, or the device compare below proves only that
    // the text path runs twice.
    if (want_vision) {
        const plain = try enc.encode(io, arena, ids.items, null);
        var num: f64 = 0;
        var den: f64 = 0;
        for (want, plain) |e, g| {
            num += (@as(f64, e) - g) * (@as(f64, e) - g);
            den += @as(f64, e) * e;
        }
        const d = if (den == 0) 0 else @sqrt(num / den);
        try stdout.print("          vision payload moves the conditioning by {d:.4}\n", .{d});
        if (!(d > 0.05)) {
            try stdout.print("FAILED: the vision payload is inert\n", .{});
            return error.GpuMismatch;
        }
    }

    // int8's device-vs-CPU floor is far above fp8's or a block quant's, and it is
    // MEASURED, not assumed: one isolated int8 GEMM is 0.0088 in
    // `minimax-h3-cuda-test`, one encoder layer here is 0.0065, and 50 layers
    // saturate at 0.018. fp8 and block-quant encoders land at 1.3e-4. One tolerance
    // for both would either pass a broken int8 path or fail a correct one.
    const tol: f64 = if (enc.layers[0].q.dtype == .i8) 3e-2 else 5e-3;
    if (enc.layers[0].q.dtype == .i8) {
        try stdout.print("tolerance: {d:.4} (int8's own floor; fp8/block-quant use 0.005)\n", .{tol});
    }

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
            if (!qwen3_cuda.supportsWeightsOn(b, &enc)) {
                try stdout.print("{s:<9} REFUSED (supportsWeights false — would fall back to CPU)\n", .{name});
                failures += 1;
            } else {
                _ = try qwen3_cuda.encodeVision(&enc, b, io, arena, ids.items, vision, null); // warm
                t0 = std.Io.Clock.real.now(io);
                const got = try qwen3_cuda.encodeVision(&enc, b, io, arena, ids.items, vision, null);
                ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                const r = R.rel(want, got);
                const ok = R.finite(got) and r < tol;
                if (!ok) failures += 1;
                try stdout.print("{s:<9} {d:8.0} ms   rel L2 {e:.4}  {s}\n", .{ name, ms, r, if (ok) "ok" else "FAILED" });
            }
        } else {
            try stdout.print("{s:<9} unavailable\n", .{if (libs) "cuda" else "zig-cuda"});
        }
    }
    if (want_vision) {
        try stdout.print("{s:<9} skipped (no deepstack/mrope path; the pipeline falls back to CPU)\n", .{"vulkan"});
    } else if (TensorPencil.gpu.Context.init(arena, io)) |gc| {
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
            const ok = R.finite(got) and r < tol;
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
/// checkpoint weights, the CUDA twin of `zimage_gpu`'s gated Vulkan test, and a
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
    // BOTH attention paths. `opAttnTC` is what runs; the naive one is the
    // fallback and the reference the fast path was validated against. Checking
    // only the fast one leaves the fallback free to rot, and checking only the
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
        // The GEMMs run tensor cores against the CPU's f32 accumulation, the same
        // regime `sd_unet_cuda` sits in, and `zimage_gpu` measures 2.2e-4 there.
        const ok = nonfinite == 0 and rel < 1e-3;
        if (!ok) failures += 1;
        try stdout.print("forward vs CPU ({s:<11})  rel L2 {e:.4}  {s}{s}\n", .{
            if (naive) "naive attn" else "opAttnTC", rel,
            if (ok) "ok" else "FAILED",              if (nonfinite != 0) " (non-finite output)" else "",
        });
    }
    // --- the VAE decode ---------------------------------------------------------
    // Checked at a SMALL and a LARGE latent, because the mid-block attention takes
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
/// f32->bf16 activation pad, and `bias_compact` on the f32 output).
///
/// The point is a receipt rather than an estimate. A whole-render profile puts
/// ~88% of a step under `.matmul`, but that bucket spans the op, not the GEMM,
/// `opGemmBf16` is `ptic()`-scoped, so the conversion passes are counted inside
/// it. Without this split, "cuBLASLt is as fast as it gets" and "we spend a
/// third of the step converting operands" are indistinguishable.
///
/// `seq` is the joint (caption ++ image) token count; 6848 is a 1056x1584 render.
/// Weights are random and each shape is uploaded once, the working set is far
/// past L2 either way, so reuse costs nothing in fidelity and 11.6 GB in VRAM.
/// Time ONE Anima trunk step's device work at real shapes, with no checkpoint and no
/// sampler, split GEMM / attention / elementwise, and give each part a CEILING to be
/// judged against (achieved TFLOP/s for the arithmetic, achieved GB/s for the
/// bandwidth-bound kernels).
///
/// The point is a receipt rather than an estimate, and the ceiling is the part that
/// matters. A share-of-time table has no notion of what an op *should* have cost, which
/// is exactly how Z-Image's two 37 GB/s norm kernels hid inside a 7% `elt` bucket on a
/// 936 GB/s card while the profile pointed at a `matmul` bucket that was already at 92%
/// of the roof. `bench_gemm_only` additionally splits `opGemmBf16`'s own f32<->bf16
/// conversion passes out of the GEMM they wrap, since `ptic()` scopes the op, not the
/// kernel.
///
/// `seq` is the image token count: 6534 is a 1056x1584 render, 1536 a 512x768 one.
/// Weights are random and each shape is allocated once, the working set is far past L2
/// either way, so the reuse costs nothing in fidelity and several GB in VRAM.
fn animaCudaBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, seq: usize, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const anima = TensorPencil.models.anima;
    const cfg = anima.anima_2b;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== anima-cuda-bench seq={d} ==\ncuda device: {s} (kernels: {t})\n", .{ seq, be.deviceName(), be.kernels });

    const d = cfg.dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const half = hd / 2;
    const ctx_seq = cfg.adapter.min_rows;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const iters = 10;

    // Random bf16 weights for the three distinct GEMM shapes.
    const wsq = try arena.alloc(u16, d * d); // [2048, 2048]
    const wup = try arena.alloc(u16, cfg.mlp_dim * d); // [8192, 2048]
    const wdn = try arena.alloc(u16, d * cfg.mlp_dim); // [2048, 8192]
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    for ([_][]u16{ wsq, wup, wdn }) |w| for (w) |*v| {
        v.* = @bitCast(@as(u16, @truncate(@as(u32, @bitCast(@as(f32, rnd.floatNorm(f32) * 0.02))) >> 16)));
    };

    var x_d = try be.tensorCreate(seq * d * 4);
    defer be.tensorDestroy(&x_d);
    var y_d = try be.tensorCreate(seq * d * 4);
    defer be.tensorDestroy(&y_d);
    var k_d = try be.tensorCreate(seq * d * 4);
    defer be.tensorDestroy(&k_d);
    var v_d = try be.tensorCreate(seq * d * 4);
    defer be.tensorDestroy(&v_d);
    var big_d = try be.tensorCreate(seq * cfg.mlp_dim * 4);
    defer be.tensorDestroy(&big_d);
    var mod_d = try be.tensorCreate(anima.modulationTableLen(cfg) * 4);
    defer be.tensorDestroy(&mod_d);
    var frq_d = try be.tensorCreate(2 * seq * half * 4);
    defer be.tensorDestroy(&frq_d);
    var ck_d = try be.tensorCreate(ctx_seq * d * 4);
    defer be.tensorDestroy(&ck_d);
    var cv_d = try be.tensorCreate(ctx_seq * d * 4);
    defer be.tensorDestroy(&cv_d);
    {
        const fill = try arena.alloc(f32, seq * cfg.mlp_dim);
        for (fill) |*t| t.* = rnd.floatNorm(f32);
        try be.tensorUpload(big_d, std.mem.sliceAsBytes(fill));
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(fill[0 .. seq * d]));
        try be.tensorUpload(k_d, std.mem.sliceAsBytes(fill[0 .. seq * d]));
        try be.tensorUpload(v_d, std.mem.sliceAsBytes(fill[0 .. seq * d]));
        // Modulation: the scale blocks are folded, so ~1.
        const m = try arena.alloc(f32, anima.modulationTableLen(cfg));
        for (m) |*t| t.* = 1.0 + rnd.floatNorm(f32) * 0.1;
        try be.tensorUpload(mod_d, std.mem.sliceAsBytes(m));
        const f = try arena.alloc(f32, 2 * seq * half);
        for (f) |*t| t.* = rnd.floatNorm(f32) * 0.5;
        try be.tensorUpload(frq_d, std.mem.sliceAsBytes(f));
        try be.tensorUpload(ck_d, std.mem.sliceAsBytes(fill[0 .. ctx_seq * d]));
        try be.tensorUpload(cv_d, std.mem.sliceAsBytes(fill[0 .. ctx_seq * d]));
    }

    const Timer = struct {
        io: Io,
        be: *cuda.Backend,
        fn run(self: @This(), n: usize, comptime f: anytype, args: anytype) !f64 {
            // One warm-up so PTX JIT and any plan build are outside the measurement.
            try @call(.auto, f, args);
            try self.be.ctx.synchronize();
            const t0 = std.Io.Clock.real.now(self.io);
            for (0..n) |_| try @call(.auto, f, args);
            try self.be.ctx.synchronize();
            const ns = std.Io.Clock.real.now(self.io).nanoseconds - t0.nanoseconds;
            return @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(n));
        }
    };
    const T = Timer{ .io = io, .be = be };

    // --- GEMMs, with and without the conversion passes --------------------------
    try stdout.print("\n-- GEMMs (per call; x{d} per block => the block total) --\n", .{1});
    const Gemm = struct { name: []const u8, y: cuda.backend.DeviceBuffer, x: cuda.backend.DeviceBuffer, w: []const u16, co: usize, k: usize, per_block: usize, m: usize };
    const gemms = [_]Gemm{
        .{ .name = "q/k/v/out  2048<-2048", .y = y_d, .x = x_d, .w = wsq, .co = d, .k = d, .per_block = 6, .m = seq },
        .{ .name = "mlp1       8192<-2048", .y = big_d, .x = x_d, .w = wup, .co = cfg.mlp_dim, .k = d, .per_block = 1, .m = seq },
        .{ .name = "mlp2       2048<-8192", .y = y_d, .x = big_d, .w = wdn, .co = d, .k = cfg.mlp_dim, .per_block = 1, .m = seq },
        // NOT part of the step: these are the cross-attention K/V projections, which
        // `Session.init` hoists out of the step loop because they depend only on the
        // context. Timed here so the hoist's worth is MEASURED rather than asserted,
        // and it is small (see the roll-up), which a FLOP count off by 30x once claimed
        // otherwise.
        .{ .name = "cross k/v  2048<-1024 (hoisted)", .y = y_d, .x = x_d, .w = wsq, .co = d, .k = 1024, .per_block = 2, .m = ctx_seq },
    };
    var gemm_ms: f64 = 0;
    var gemm_flop: f64 = 0;
    var hoisted_ms: f64 = 0;
    for (gemms) |g| {
        const wb = std.mem.sliceAsBytes(g.w);
        const flop = 2.0 * @as(f64, @floatFromInt(g.m * g.co * g.k));
        cuda.backend.bench_gemm_only = false;
        const full = try T.run(iters, cuda.Backend.opGemmBf16, .{ be, g.y, g.x, g.m, wb, g.co, g.k, @as(?[]const f32, null) });
        cuda.backend.bench_gemm_only = true;
        const only = try T.run(iters, cuda.Backend.opGemmBf16, .{ be, g.y, g.x, g.m, wb, g.co, g.k, @as(?[]const f32, null) });
        cuda.backend.bench_gemm_only = false;
        if (g.m == seq) {
            gemm_ms += full * @as(f64, @floatFromInt(g.per_block));
            gemm_flop += flop * @as(f64, @floatFromInt(g.per_block));
        } else {
            hoisted_ms += full * @as(f64, @floatFromInt(g.per_block));
        }
        try stdout.print("  {s}  x{d}  {d:7.2} ms  ({d:6.2} gemm + {d:5.2} convert)  {d:6.1} TFLOP/s pure\n", .{
            g.name, g.per_block, full, only, full - only, flop / only / 1e9,
        });
    }

    // --- attention --------------------------------------------------------------
    try stdout.print("\n-- attention --\n", .{});
    const self_flop = 4.0 * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(d));
    const self_ms = try T.run(iters, cuda.Backend.opAttnTC, .{ be, x_d, k_d, v_d, y_d, seq, heads, heads, hd, scale });
    try stdout.print("  self  {d:>5}q x {d:>5}kv  x1  {d:7.2} ms  {d:6.1} TFLOP/s\n", .{ seq, seq, self_ms, self_flop / self_ms / 1e9 });
    const cross_flop = 4.0 * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(ctx_seq)) * @as(f64, @floatFromInt(d));
    const cross_ms = try T.run(iters, cuda.Backend.opAttnTCRect, .{ be, x_d, ck_d, cv_d, y_d, seq, ctx_seq, heads, heads, hd, scale });
    try stdout.print("  cross {d:>5}q x {d:>5}kv  x1  {d:7.2} ms  {d:6.1} TFLOP/s\n", .{ seq, ctx_seq, cross_ms, cross_flop / cross_ms / 1e9 });

    // --- elementwise, judged by achieved BANDWIDTH ------------------------------
    // This is the column that makes a broken kernel legible: these do almost no
    // arithmetic, so bytes-moved / seconds has a known ceiling (~936 GB/s on a 3090,
    // ~700 achievable), and a kernel at a tenth of it is broken rather than merely
    // small. Share-of-time cannot say that.
    try stdout.print("\n-- elementwise (bytes moved / time; the 3090's roof is 936 GB/s) --\n", .{});
    const fseq: f64 = @floatFromInt(seq);
    const fd: f64 = @floatFromInt(d);
    const Elt = struct { name: []const u8, ms: f64, bytes: f64, per_block: usize };
    var elts: [5]Elt = undefined;
    // lnMod: two read passes over x plus a read+write in the apply pass.
    elts[0] = .{ .name = "lnMod", .ms = try T.run(iters, cuda.Backend.lnMod, .{ be, x_d, y_d, mod_d, seq, d, d, @as(usize, 0), cfg.norm_eps }), .bytes = fseq * fd * 4 * 4, .per_block = 3 };
    elts[1] = .{ .name = "gatedAdd", .ms = try T.run(iters, cuda.Backend.gatedAdd, .{ be, x_d, y_d, mod_d, seq * d, d, @as(usize, 0) }), .bytes = fseq * fd * 4 * 3, .per_block = 3 };
    var nrm_d = try be.tensorCreate(hd * 4);
    defer be.tensorDestroy(&nrm_d);
    elts[2] = .{ .name = "qkNorm", .ms = try T.run(iters, cuda.Backend.qkNorm, .{ be, x_d, x_d, nrm_d, seq * heads, hd, cfg.qk_eps }), .bytes = fseq * fd * 4 * 2, .per_block = 3 };
    elts[3] = .{ .name = "ropeHalf", .ms = try T.run(iters, cuda.Backend.ropeHalf, .{ be, x_d, frq_d, seq, heads, half, seq * half, @as(usize, 0) }), .bytes = fseq * fd * 4 * 2, .per_block = 2 };
    elts[4] = .{ .name = "geluErf (mlp width)", .ms = try T.run(iters, cuda.Backend.geluErf, .{ be, big_d, seq * cfg.mlp_dim }), .bytes = fseq * @as(f64, @floatFromInt(cfg.mlp_dim)) * 4 * 2, .per_block = 1 };
    var elt_ms: f64 = 0;
    for (elts) |e| {
        elt_ms += e.ms * @as(f64, @floatFromInt(e.per_block));
        try stdout.print("  {s:<20} x{d}  {d:7.2} ms  {d:6.0} GB/s\n", .{ e.name, e.per_block, e.ms, e.bytes / e.ms / 1e6 });
    }

    // --- the roll-up ------------------------------------------------------------
    const nl: f64 = @floatFromInt(cfg.n_layers);
    const block_ms = gemm_ms + self_ms + cross_ms + elt_ms;
    const step_ms = block_ms * nl;
    try stdout.print("\n-- one block, then one {d}-block step --\n", .{cfg.n_layers});
    try stdout.print("  GEMM        {d:8.2} ms/block  {d:8.0} ms/step  ({d:4.1}%)  {d:5.1} TFLOP/s effective\n", .{ gemm_ms, gemm_ms * nl, 100 * gemm_ms / block_ms, gemm_flop / gemm_ms / 1e9 });
    try stdout.print("  self attn   {d:8.2} ms/block  {d:8.0} ms/step  ({d:4.1}%)\n", .{ self_ms, self_ms * nl, 100 * self_ms / block_ms });
    try stdout.print("  cross attn  {d:8.2} ms/block  {d:8.0} ms/step  ({d:4.1}%)\n", .{ cross_ms, cross_ms * nl, 100 * cross_ms / block_ms });
    try stdout.print("  elementwise {d:8.2} ms/block  {d:8.0} ms/step  ({d:4.1}%)\n", .{ elt_ms, elt_ms * nl, 100 * elt_ms / block_ms });
    try stdout.print("  TOTAL       {d:8.2} ms/block  {d:8.0} ms/step\n", .{ block_ms, step_ms });
    try stdout.print("\n  hoisted out of the step loop (cross k/v, per IMAGE not per step):\n", .{});
    try stdout.print("  {d:8.2} ms/block  {d:8.1} ms once  = {d:5.2}% of a step if it were NOT hoisted\n", .{
        hoisted_ms, hoisted_ms * nl, 100 * hoisted_ms * nl / step_ms,
    });
}

/// Sweep the weighted-RMSNorm shapes the three Vulkan DiTs actually use, timing the
/// thread-per-row `Elt.rmsnorm` against the subgroup `rmsnorm_sg` at each.
///
/// Exists because the choice between them is shape-dependent and the fast one is not
/// always the subgroup one: `rmsnorm` gives a thread a whole row, which is catastrophic
/// for a narrow row (a warp's loads land `dim*4` bytes apart) but not obviously so for a
/// wide one, where a single thread's own reads are at least sequential. Anima's 128-wide
/// heads measured 45 GB/s against 511 at production size, and only 176 against 355
/// at a third of it, so a small-shape measurement understates the gap by 3x. Judge each
/// geometry, and judge it at the size the model runs.
/// Persist the measured tiles and say where. The format and the merge live in
/// `context.writeCoopTileCache`, shared with the init-time screen.
fn writeCoopTileCache(io: Io, stdout: *Io.Writer, ctx: anytype, f16w: [2]u32, i8t: [2]u32) void {
    const gpu = TensorPencil.gpu.context;
    var pbuf: [512]u8 = undefined;
    const path = gpu.coopTileCachePath(&pbuf) orelse {
        stdout.print("no cache path (set TP_COOP_TILE_CACHE, XDG_CACHE_HOME or HOME); not persisted\n", .{}) catch {};
        return;
    };
    gpu.writeCoopTileCache(io, ctx.coopCacheKey(), f16w, i8t);
    stdout.print("persisted to {s}; future runs pick this up automatically\n", .{path}) catch {};
}

/// int8 half of `tune-coop`. Same discipline as the f16 sweep and for the same reasons:
/// warm up before timing anything, score a SET of shapes rather than one, give each shape
/// its own weight allocation (the weight cache keys on the host pointer with no shape in
/// the key, so slices of one buffer collide and every shape after the first would time a
/// read of a wrongly-transposed weight), and batch the probes into one submit.
fn tuneCoopI8(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ctx: anytype) ![2]u32 {
    const gpu = TensorPencil.gpu.context;
    if (ctx.pipe_coop_i8_sh == .null_handle) {
        try stdout.print("\nint8: no shared cooperative-matrix pipeline; nothing to tune\n", .{});
        return .{ 64, 64 };
    }
    const nwarps: u32 = ctx.coopI8Warps();
    var buf: [9][2]u32 = undefined;
    const cands = gpu.coopI8TileCandidates(ctx.coop_i8_m, ctx.coop_i8_n, nwarps, ctx.max_shared_bytes, ctx.max_wg_invocations, &buf);
    try stdout.print("\nint8 fragment {d}x{d}x{d}, {d} expressible tile(s)\n\n", .{ ctx.coop_i8_m, ctx.coop_i8_n, ctx.coop_i8_k, cands.len });
    if (cands.len == 0) return .{ 64, 64 };

    // DiT-block shapes, where int8 convrot actually runs. m and n are multiples of 128
    // and k of 64 so every candidate tile can express them, which keeps the comparison
    // about speed rather than about which tiles had to fall back.
    const Shape = struct { m: usize, n: usize, k: usize };
    const shapes = [_]Shape{
        .{ .m = 2048, .n = 1280, .k = 1280 },
        .{ .m = 2048, .n = 5120, .k = 1280 },
        .{ .m = 2048, .n = 1280, .k = 5120 },
        .{ .m = 1024, .n = 2560, .k = 2560 },
    };
    var max_x: usize = 0;
    var max_y: usize = 0;
    for (shapes) |sh| {
        max_x = @max(max_x, sh.m * sh.k);
        max_y = @max(max_y, sh.m * sh.n);
    }
    var x_d = try ctx.tensorCreate(max_x);
    defer ctx.tensorDestroy(&x_d);
    var y_d = try ctx.tensorCreate(max_y * 4);
    defer ctx.tensorDestroy(&y_d);
    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    {
        const xb = try arena.alloc(u8, max_x);
        for (xb) |*v| v.* = @bitCast(rand.int(i8));
        try ctx.tensorUpload(x_d, xb);
    }
    var w_of: [shapes.len][]u8 = undefined;
    for (shapes, 0..) |sh, i| {
        w_of[i] = try arena.alloc(u8, sh.n * sh.k);
        for (w_of[i]) |*v| v.* = @bitCast(rand.int(i8));
    }

    ctx.setCoopI8WarpTile(arena, cands[0][0], cands[0][1]) catch {};
    for (0..4) |_| {
        try ctx.beginBatch();
        for (shapes, 0..) |sh, i| try ctx.opMatmulCoopI8(y_d, x_d, sh.m, null, w_of[i], sh.n, sh.k);
        try ctx.endBatch();
    }

    var ms_of = try arena.alloc(f64, cands.len);
    @memset(ms_of, std.math.inf(f64));
    for (0..2) |_| for (cands, 0..) |c, ci| {
        ctx.setCoopI8WarpTile(arena, c[0], c[1]) catch continue;
        const reps: usize = 2;
        var iter: usize = 0;
        while (iter < 5) : (iter += 1) {
            const t0 = std.Io.Clock.real.now(io);
            try ctx.beginBatch();
            for (0..reps) |_| for (shapes, 0..) |sh, i| {
                try ctx.opMatmulCoopI8(y_d, x_d, sh.m, null, w_of[i], sh.n, sh.k);
            };
            try ctx.endBatch();
            const el = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
            if (iter >= 2) ms_of[ci] = @min(ms_of[ci], el / @as(f64, @floatFromInt(reps)));
        }
    };

    var flops: f64 = 0;
    for (shapes) |sh| flops += 2.0 * @as(f64, @floatFromInt(sh.m * sh.n * sh.k));
    var best_ms: f64 = std.math.inf(f64);
    var base_ms: f64 = std.math.inf(f64);
    var best: [2]u32 = cands[0];
    for (cands, 0..) |c, ci| {
        const ms = ms_of[ci];
        if (!std.math.isFinite(ms)) continue;
        try stdout.print("  i8 warp {d:>3}x{d:<3} wg {d:>3}x{d:<3}  {d:>9.2} ms  {d:>8.1} GFLOP/s\n", .{
            c[0], c[1], 2 * c[0], (nwarps / 2) * c[1], ms, flops / (ms * 1e6),
        });
        if (c[0] == 64 and c[1] == 64) base_ms = ms;
        if (ms < best_ms) {
            best_ms = ms;
            best = c;
        }
    }
    if (std.math.isFinite(base_ms) and best_ms > base_ms * 0.95) {
        best = .{ 64, 64 };
        best_ms = base_ms;
    }
    try stdout.print("\nfastest int8: warp {d}x{d}", .{ best[0], best[1] });
    if (best[0] == 64 and best[1] == 64) {
        try stdout.print(" (the default; nothing to set)\n", .{});
    } else {
        if (std.math.isFinite(base_ms)) {
            try stdout.print(", {d:.2}x the 64x64 default\n", .{base_ms / best_ms});
        } else {
            try stdout.print(" (64x64 is not expressible here)\n", .{});
        }
        try stdout.print("set it with: TP_COOP_I8_TILE={d} TP_COOP_I8_TILE_N={d}\n", .{ best[0], best[1] });
    }
    // Leave the context on the winner so a caller that keeps using it gets the good one.
    ctx.setCoopI8WarpTile(arena, best[0], best[1]) catch {};
    return best;
}

/// Measure every warp tile the device can express for the f16 coop GEMM and report
/// which is fastest. This is a measurement and not a derivation on purpose: the tile
/// is decided by the register budget per lane, Vulkan exposes no query for that, and
/// the legal-but-slow tiles are indistinguishable from the legal-and-fast ones by any
/// limit we can read. So no vendor table — an unknown GPU measures itself.
fn tuneCoop(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const gpu = TensorPencil.gpu.context;
    const ctx = gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== tune-coop ==\ndevice: {s}\n", .{ctx.device_name[0..ctx.device_name_len]});
    if (ctx.coop_m == 0) {
        try stdout.print("no f16 cooperative-matrix pipeline on this device; nothing to tune\n", .{});
        return;
    }
    var buf: [9][2]u32 = undefined;
    const cands = gpu.coopTileCandidates(ctx.coop_m, ctx.coop_n, ctx.max_shared_bytes, ctx.max_wg_invocations, &buf);
    try stdout.print("fragment {d}x{d}x{d}, subgroup {d}, {d} expressible tile(s)\n\n", .{
        ctx.coop_m, ctx.coop_n, ctx.coop_k, ctx.subgroup_size, cands.len,
    });
    if (cands.len == 0) return;

    // A SET of shapes, not one. A single representative GEMM is not representative:
    // measured on m=2048/512/1152 alone this picks warp 16x32 on an A310, and the real
    // render is then 1.9x SLOWER than at 32x32. These span what an SD render issues —
    // a short CLIP sequence, a mid UNet linear, a 3x3 im2col conv, and a late VAE conv
    // over a large pixel band — and the winner is the one with the best TOTAL time.
    const Shape = struct { m: usize, rows: usize, cols: usize, what: []const u8 };
    const shapes = [_]Shape{
        .{ .m = 77, .rows = 768, .cols = 768, .what = "clip seq" },
        .{ .m = 1024, .rows = 1280, .cols = 1280, .what = "unet lin" },
        .{ .m = 4096, .rows = 640, .cols = 2880, .what = "unet conv" },
        .{ .m = 16384, .rows = 128, .cols = 1152, .what = "vae conv" },
    };
    var max_x: usize = 0;
    var max_y: usize = 0;
    for (shapes) |sh| {
        max_x = @max(max_x, sh.m * sh.cols);
        max_y = @max(max_y, sh.m * sh.rows);
    }
    var x_d = try ctx.tensorCreate(max_x * 4);
    defer ctx.tensorDestroy(&x_d);
    var y_d = try ctx.tensorCreate(max_y * 4);
    defer ctx.tensorDestroy(&y_d);
    {
        var prng = std.Random.DefaultPrng.init(7);
        const rnd = prng.random();
        const fill = try arena.alloc(f32, max_x);
        for (fill) |*t| t.* = rnd.floatNorm(f32);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(fill));
    }
    // ONE ALLOCATION PER SHAPE. The weight cache is keyed by host pointer alone
    // (context.weightBuffer), with no shape in the key, so slices of a shared buffer
    // would collide on a single entry and three of the four shapes would be timing
    // reads of a buffer transposed for the first shape's dimensions.
    var w_of: [shapes.len][]u16 = undefined;
    {
        var prng = std.Random.DefaultPrng.init(11);
        const rnd = prng.random();
        for (shapes, 0..) |sh, i| {
            w_of[i] = try arena.alloc(u16, sh.rows * sh.cols);
            for (w_of[i]) |*t| t.* = @bitCast(@as(f16, @floatCast(rnd.floatNorm(f32) * 0.05)));
        }
    }
    const bias = try arena.alloc(f32, 1280);
    @memset(bias, 0);

    // Warm up before ANY candidate is timed. This GPU idles its clocks, and the first
    // candidate would otherwise be measured cold and look catastrophically slow purely
    // for being first — which is exactly what a naive sweep reports.
    ctx.setCoopWarpTile(arena, cands[0][0], cands[0][1]) catch {};
    for (0..8) |_| {
        try ctx.beginBatch();
        for (shapes, 0..) |sh, i| {
            try ctx.opMatmulCoopF16Wh(y_d, 0, x_d, sh.m, std.mem.sliceAsBytes(w_of[i]), sh.rows, sh.cols, bias[0..sh.rows]);
        }
        try ctx.endBatch();
    }

    var ms_of = try arena.alloc(f64, cands.len);
    @memset(ms_of, std.math.inf(f64));
    // Two passes over the whole list, keeping the min per candidate: a single pass
    // cannot distinguish a slow tile from a tile that happened to run while the clocks
    // were still ramping.
    for (0..2) |_| for (cands, 0..) |c, ci| {
        ctx.setCoopWarpTile(arena, c[0], c[1]) catch continue;
        // All probes in ONE submit, repeated, and divided back out. One submit-and-wait
        // per GEMM measures per-dispatch LATENCY; a render records many ops per submit,
        // so it lives in the throughput regime instead. The two rank tiles differently
        // (that is what this loop exists to expose), so the tuner has to measure the
        // regime the workload actually runs in.
        // setCoopWarpTile evicts the weight cache, so the first passes re-upload and
        // re-transpose every weight; discard them.
        const reps: usize = 4;
        var iter: usize = 0;
        while (iter < 6) : (iter += 1) {
            const t0 = std.Io.Clock.real.now(io);
            try ctx.beginBatch();
            for (0..reps) |_| for (shapes, 0..) |sh, si| {
                try ctx.opMatmulCoopF16Wh(y_d, 0, x_d, sh.m, std.mem.sliceAsBytes(w_of[si]), sh.rows, sh.cols, bias[0..sh.rows]);
            };
            try ctx.endBatch();
            const el = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
            if (iter >= 2) ms_of[ci] = @min(ms_of[ci], el / @as(f64, @floatFromInt(reps)));
        }
    };

    var best_ms: f64 = std.math.inf(f64);
    var base_ms: f64 = std.math.inf(f64);
    var best: [2]u32 = cands[0];
    for (cands, 0..) |c, ci| {
        const ms = ms_of[ci];
        if (!std.math.isFinite(ms)) continue;
        try stdout.print("  warp {d:>3}x{d:<3} wg {d:>3}x{d:<3}  {d:>9.2} ms total over {d} shapes\n", .{
            c[0], c[1], 2 * c[0], 2 * c[1], ms, shapes.len,
        });
        if (c[0] == 64 and c[1] == 64) base_ms = ms;
        if (ms < best_ms) {
            best_ms = ms;
            best = c;
        }
    }
    const i8_best = try tuneCoopI8(arena, io, stdout, ctx);

    // A tie is not a reason to change geometry: require a real margin over the default
    // before recommending a switch, or the tuner churns on run-to-run noise.
    const margin = 0.95;
    if (std.math.isFinite(base_ms) and best_ms > base_ms * margin) {
        best = .{ 64, 64 };
        best_ms = base_ms;
    }
    try stdout.print("\nfastest: warp {d}x{d}", .{ best[0], best[1] });
    if (best[0] == 64 and best[1] == 64) {
        try stdout.print(" (the default; nothing to set)\n", .{});
    } else {
        if (std.math.isFinite(base_ms)) {
            try stdout.print(", {d:.2}x the 64x64 default\n", .{base_ms / best_ms});
        } else {
            try stdout.print(" (64x64 is not expressible here)\n", .{});
        }
        try stdout.print("set it with: TP_COOP_WARP_TILE={d} TP_COOP_WARP_TILE_N={d}\n", .{ best[0], best[1] });
    }
    try stdout.print("\n", .{});
    writeCoopTileCache(io, stdout, ctx, best, i8_best);
}

fn vkNormBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const gpu = TensorPencil.gpu.context;

    const ctx = gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== vk-norm-bench ==\nvulkan device: {s}\n", .{ctx.device_name[0..ctx.device_name_len]});
    if (!ctx.hasSubgroupNorm()) {
        try stdout.print("rmsnorm_sg unavailable on this device; nothing to compare\n", .{});
        return;
    }

    // Real geometries, at the token counts these models are actually rendered at.
    const Case = struct { who: []const u8, rows: usize, dim: usize, per_step: usize };
    const cases = [_]Case{
        .{ .who = "anima  Q/K      @1056x1584", .rows = 6534 * 16, .dim = 128, .per_step = 3 * 28 },
        .{ .who = "anima  Q/K      @512x768  ", .rows = 1536 * 16, .dim = 128, .per_step = 3 * 28 },
        .{ .who = "zimage Q/K      @1056x1584", .rows = 6848 * 30, .dim = 128, .per_step = 2 * 32 },
        .{ .who = "zimage sandwich @1056x1584", .rows = 6848, .dim = 3840, .per_step = 4 * 32 },
        // krea2's Q/K `Elt.rmsnorm` sits in an ELSE branch behind two fused
        // alternatives (`qknorm_rope16`, `qknorm_rope_f32`). It is live for a bf16 or fp8
        // checkpoint, `qkv_shared` requires `!is_bf16`, so those fall through, and NOT for
        // an int8 one, which takes the fused f32 path. So this row is a real saving for the
        // dense checkpoints only.
        .{ .who = "krea2  Q/K bf16 @1120x1680", .rows = 7350 * 48, .dim = 128, .per_step = 2 * 28 },
        // Reference only, `per_step = 0`. A 6144-wide row is krea2's block-norm
        // shape, but `dit_gpu` routes those through the already-parallel
        // `rms_partial`/`rms_combine`/`rms_apply_mod` chain, NOT `Elt.rmsnorm`. Timed here
        // so nobody reads the wide-row numbers as an available krea2 win; there is none.
        .{ .who = "  (ref) wide row, no such site", .rows = 8400, .dim = 6144, .per_step = 0 },
    };

    var max_elems: usize = 0;
    var max_dim: usize = 0;
    for (cases) |c| {
        max_elems = @max(max_elems, c.rows * c.dim);
        max_dim = @max(max_dim, c.dim);
    }
    var x_d = try ctx.tensorCreate(max_elems * 4);
    defer ctx.tensorDestroy(&x_d);
    var y_d = try ctx.tensorCreate(max_elems * 4);
    defer ctx.tensorDestroy(&y_d);
    var w_d = try ctx.tensorCreate(max_dim * 4);
    defer ctx.tensorDestroy(&w_d);
    {
        var prng = std.Random.DefaultPrng.init(9);
        const rnd = prng.random();
        const fill = try arena.alloc(f32, max_elems);
        for (fill) |*t| t.* = rnd.floatNorm(f32);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(fill));
        const w = try arena.alloc(f32, max_dim);
        for (w) |*t| t.* = 1.0;
        try ctx.tensorUpload(w_d, std.mem.sliceAsBytes(w));
    }

    // Warm on the op, then best-of-rounds: the 3090 idles its clocks and a cold short run
    // reads ~2x slow (see anima-vk-bench).
    const T = struct {
        io: Io,
        fn run(self: @This(), comptime f: anytype, args: anytype) !f64 {
            const w0 = std.Io.Clock.real.now(self.io);
            var n: usize = 0;
            while (n < 200) : (n += 1) {
                try @call(.auto, f, args);
                const el = @as(f64, @floatFromInt(std.Io.Clock.real.now(self.io).nanoseconds - w0.nanoseconds)) / 1e6;
                if (el >= 150) break;
            }
            var best: f64 = std.math.inf(f64);
            for (0..5) |_| {
                const t0 = std.Io.Clock.real.now(self.io);
                for (0..8) |_| try @call(.auto, f, args);
                const ns = std.Io.Clock.real.now(self.io).nanoseconds - t0.nanoseconds;
                best = @min(best, @as(f64, @floatFromInt(ns)) / 1e6 / 8.0);
            }
            return best;
        }
    }{ .io = io };

    try stdout.print("\n{s:<28} {s:>10} {s:>12} {s:>12} {s:>7}  {s}\n", .{
        "shape", "rows x dim", "thread/row", "subgroup", "speedup", "per-step saving",
    });
    for (cases) |c| {
        const bytes = @as(f64, @floatFromInt(c.rows * c.dim)) * 4 * 2; // read + write
        const thr = try T.run(gpu.Context.opElt, .{ ctx, gpu.Elt.rmsnorm, x_d, @as(?gpu.DeviceBuffer, y_d), @as(?gpu.DeviceBuffer, w_d), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
            .u0 = @intCast(c.rows),
            .u1 = @intCast(c.dim),
            .f0 = 1e-6,
        }, c.rows, @as(usize, 1), @as(usize, 1) });
        const sub = try T.run(gpu.Context.opRmsNormSg, .{ ctx, x_d, y_d, w_d, c.rows, c.dim, @as(f32, 1e-6) });
        try stdout.print("{s} {d:6}x{d:<4} {d:7.2} ms {d:4.0} GB/s {d:7.2} ms {d:4.0} GB/s {d:6.1}x  {d:6.0} -> {d:4.0} ms\n", .{
            c.who,      c.rows,                    c.dim,
            thr,        bytes / thr / 1e6,
            sub,        bytes / sub / 1e6,
            thr / sub,  thr * @as(f64, @floatFromInt(c.per_step)),
            sub * @as(f64, @floatFromInt(c.per_step)),
        });
    }
    try stdout.print("\n`per-step saving` assumes every call site switches; it is the CEILING on\n", .{});
    try stdout.print("    the change, not a prediction — a render also has to be re-measured.\n", .{});
}

/// Time ONE Anima trunk step's VULKAN device work at real shapes, split GEMM / int8 prep /
/// attention / elementwise, each against a CEILING, the Vulkan counterpart of
/// `anima-cuda-bench`, and the harness whose absence made "Vulkan's int8 speedup is 1.2x
/// where CUDA's is 1.9x" an observation rather than a diagnosis.
///
/// Timing is sync-per-op and that is exact here, not an approximation: outside
/// `beginBatch` every `opBegin`/`opEnd` pair ends in `submitAndWait`, which is the same
/// methodology `dit_gpu`'s `--profile` uses. It does mean the numbers include per-submit
/// overhead, which a batched render elides, so the roll-up is an upper bound on the step
/// and the PER-OP ratios are the trustworthy part.
///
/// Prints bf16 and int8 side by side at identical shapes, because the question is not "how
/// fast is the int8 GEMM" but "is it faster than the bf16 one it replaces, and by enough to
/// pay for the prep".
fn animaVkBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, seq: usize) !void {
    const gpu = TensorPencil.gpu.context;
    const anima = TensorPencil.models.anima;
    const cfg = anima.anima_2b;

    const ctx = gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== anima-vk-bench seq={d} ==\nvulkan device: {s}\n", .{ seq, ctx.device_name[0..ctx.device_name_len] });
    try stdout.print("  coop f16 {s} · coop bf16w {s} · coop i8 {s} · ln_mod_sg {s} · flash {s}\n", .{
        if (ctx.pipe_coop_f16w != .null_handle) "yes" else "NO",
        if (ctx.pipe_coop_bf16w != .null_handle) "yes" else "NO",
        if (ctx.pipe_coop_i8 != .null_handle) "yes" else "NO",
        if (ctx.hasLnModSg()) "yes" else "NO",
        if (ctx.pipe_flash_md != .null_handle) "yes" else "NO",
    });
    // Stated, not inferred: a width with no fused prep silently takes the 3-pass chain.
    for ([_]usize{ cfg.dim, cfg.mlp_dim }) |c| {
        try stdout.print("  fused int8 prep @ cols {d}: {s}\n", .{ c, if (ctx.hasFusedI8Prep(c)) "yes" else "NO (3-pass fallback)" });
    }

    const d = cfg.dim;
    const mlp = cfg.mlp_dim;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const half = hd / 2;
    const ctx_seq = cfg.adapter.min_rows;
    const seq_pad = std.mem.alignForward(usize, seq, 128);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const iters = 8;

    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();

    // Weights at the three device shapes, in both dtypes.
    const Shape = struct { name: []const u8, rows: usize, cols: usize, per_block: usize };
    const shapes = [_]Shape{
        .{ .name = "q/k/v/out+cross 2048<-2048", .rows = d, .cols = d, .per_block = 6 },
        .{ .name = "mlp1            8192<-2048", .rows = mlp, .cols = d, .per_block = 1 },
        .{ .name = "mlp2            2048<-8192", .rows = d, .cols = mlp, .per_block = 1 },
    };
    var wbf: [shapes.len][]u16 = undefined;
    var wi8: [shapes.len][]u8 = undefined;
    var wsc: [shapes.len][]f32 = undefined;
    for (shapes, 0..) |sh, i| {
        wbf[i] = try arena.alloc(u16, sh.rows * sh.cols);
        for (wbf[i]) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.floatNorm(f32) * 0.02)) >> 16);
        wi8[i] = try arena.alloc(u8, sh.rows * sh.cols);
        for (wi8[i]) |*v| v.* = @bitCast(@as(i8, @intCast(rnd.intRangeAtMost(i8, -127, 127))));
        wsc[i] = try arena.alloc(f32, sh.rows);
        for (wsc[i]) |*v| v.* = 0.0005 + rnd.float(f32) * 0.002;
    }

    // Buffers, all padded so a quant GEMM's i8_mpad rows fit.
    var x_d = try ctx.tensorCreate(seq_pad * mlp * 4);
    defer ctx.tensorDestroy(&x_d);
    var y_d = try ctx.tensorCreate(seq_pad * mlp * 4);
    defer ctx.tensorDestroy(&y_d);
    var k_d = try ctx.tensorCreate(seq_pad * d * 4);
    defer ctx.tensorDestroy(&k_d);
    var v_d = try ctx.tensorCreate(seq_pad * d * 4);
    defer ctx.tensorDestroy(&v_d);
    var mod_d = try ctx.tensorCreate(anima.modulationTableLen(cfg) * 4);
    defer ctx.tensorDestroy(&mod_d);
    var frq_d = try ctx.tensorCreate(2 * seq * half * 4);
    defer ctx.tensorDestroy(&frq_d);
    var nrm_d = try ctx.tensorCreate(hd * 4);
    defer ctx.tensorDestroy(&nrm_d);
    var q16 = try ctx.tensorCreate(seq_pad * d * 2);
    defer ctx.tensorDestroy(&q16);
    var k16 = try ctx.tensorCreate(d * seq_pad * 2);
    defer ctx.tensorDestroy(&k16);
    var v16 = try ctx.tensorCreate(seq_pad * d * 2);
    defer ctx.tensorDestroy(&v16);
    var o_d = try ctx.tensorCreate((seq_pad * d + heads * seq_pad * 2) * 4);
    defer ctx.tensorDestroy(&o_d);
    {
        const fill = try arena.alloc(f32, seq_pad * mlp);
        for (fill) |*t| t.* = rnd.floatNorm(f32);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(fill));
        try ctx.tensorUpload(k_d, std.mem.sliceAsBytes(fill[0 .. seq_pad * d]));
        try ctx.tensorUpload(v_d, std.mem.sliceAsBytes(fill[0 .. seq_pad * d]));
        const m = try arena.alloc(f32, anima.modulationTableLen(cfg));
        for (m) |*t| t.* = 1.0 + rnd.floatNorm(f32) * 0.1;
        try ctx.tensorUpload(mod_d, std.mem.sliceAsBytes(m));
        const f = try arena.alloc(f32, 2 * seq * half);
        for (f) |*t| t.* = rnd.floatNorm(f32) * 0.5;
        try ctx.tensorUpload(frq_d, std.mem.sliceAsBytes(f));
        const nw = try arena.alloc(f32, hd);
        for (nw) |*t| t.* = 1.0;
        try ctx.tensorUpload(nrm_d, std.mem.sliceAsBytes(nw));
    }
    const zeros = try arena.alloc(f32, mlp);
    @memset(zeros, 0);

    // BEST-of-rounds after a timed warm-up, not a single mean, the 3090 idles its
    // clocks and a cold short run reads 2x slow. A first version of this harness used one
    // 8-iteration mean and produced numbers that moved by 2.2x between runs (bf16 mlp1 read
    // 2.29 ms then 5.06 ms; `gelu_erf` 672 then 342 GB/s), which is enough to invert a
    // conclusion. The warm-up spins the clocks up on the op being measured, and the minimum
    // over rounds is the estimate least polluted by interference, the same reason
    // `gpu-perf-lab-notes` says never to trust a short run on this card.
    const T = struct {
        io: Io,
        const warm_ms: f64 = 150;
        const rounds: usize = 5;
        fn run(self: @This(), n: usize, comptime f: anytype, args: anytype) !f64 {
            const w0 = std.Io.Clock.real.now(self.io);
            var warmed: usize = 0;
            while (true) {
                try @call(.auto, f, args);
                warmed += 1;
                const el = @as(f64, @floatFromInt(std.Io.Clock.real.now(self.io).nanoseconds - w0.nanoseconds)) / 1e6;
                if (el >= warm_ms or warmed >= 200) break;
            }
            var best: f64 = std.math.inf(f64);
            for (0..rounds) |_| {
                const t0 = std.Io.Clock.real.now(self.io);
                for (0..n) |_| try @call(.auto, f, args);
                const ns = std.Io.Clock.real.now(self.io).nanoseconds - t0.nanoseconds;
                best = @min(best, @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(n)));
            }
            return best;
        }
    }{ .io = io };

    // --- the sync-per-op floor -------------------------------------------------
    // Measure the per-submit cost before anything else, so every number below can be
    // read against it. Outside a batch each op is its own submit+fence, and a render
    // BATCHES, so an op whose time is close to this floor is being measured, not timed.
    const floor = try T.run(64, gpu.Context.opElt, .{ ctx, gpu.Elt.copy, x_d, @as(?gpu.DeviceBuffer, y_d), @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
        .u0 = 256,
        .u2 = 0,
        .u3 = 0,
    }, @as(usize, 256), @as(usize, 1), @as(usize, 1) });
    try stdout.print("\n-- submit+fence floor (a 256-element copy): {d:.3} ms/op --\n", .{floor});
    try stdout.print("   Subtract it to judge a kernel; a batched render pays it ~once per 512 dispatches.\n", .{});

    // --- GEMMs: bf16 vs int8 at identical shapes -------------------------------
    try stdout.print("\n-- GEMMs, bf16 vs int8 at the same shape (per call) --\n", .{});
    var bf_ms: f64 = 0;
    var i8_ms: f64 = 0;
    var prep_ms: f64 = 0;
    for (shapes, 0..) |sh, i| {
        const flop = 2.0 * @as(f64, @floatFromInt(seq * sh.rows * sh.cols));
        const bw = std.mem.sliceAsBytes(wbf[i]);
        const bf = try T.run(iters, gpu.Context.opMatmulCoopBf16, .{ ctx, y_d, @as(usize, 0), x_d, seq, bw, sh.rows, sh.cols, @as([]const f32, zeros) });
        // int8 needs a prep of the activation at this reduction width first.
        const pr = try T.run(iters, gpu.Context.opI8Prep, .{ ctx, x_d, seq, sh.cols });
        const q8 = try T.run(iters, gpu.Context.opI8Gemm, .{ ctx, y_d, @as([]const u8, wi8[i]), @as([]const f32, wsc[i]), sh.rows, false });
        bf_ms += bf * @as(f64, @floatFromInt(sh.per_block));
        i8_ms += q8 * @as(f64, @floatFromInt(sh.per_block));
        // One prep per GROUP, not per GEMM: q/k/v/out/cross share cols=2048 (4 groups in a
        // block: qkv, self-out, cross-q, cross-out) and mlp1/mlp2 have one each.
        const groups: f64 = if (sh.cols == d and sh.rows == d) 4 else 1;
        prep_ms += pr * groups;
        // TFLOP/s is computed on the time NET of the submit floor, so a small GEMM is not
        // credited with the harness's own overhead.
        try stdout.print("  {s} x{d}  bf16 {d:7.2} ms ({d:5.1} TF/s net)  int8 {d:7.2} ms ({d:5.1} TF/s net)  {d:4.2}x   prep {d:6.2} ms\n", .{
            sh.name,      sh.per_block,
            bf,           flop / @max(bf - floor, 1e-6) / 1e9,
            q8,           flop / @max(q8 - floor, 1e-6) / 1e9,
            bf / q8,      pr,
        });
    }

    // --- attention -------------------------------------------------------------
    try stdout.print("\n-- attention (f16 operand prep + flash md/out) --\n", .{});
    var attn_ms: f64 = 0;
    inline for (.{ .{ "self ", true }, .{ "cross", false } }) |cs| {
        const square = cs[1];
        const kv_seq: usize = if (square) seq else ctx_seq;
        const kv_pad = std.mem.alignForward(usize, kv_seq, 128);
        const cvt = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.f32_to_h16, x_d, @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, null), q16, gpu.EltPush{
            .u0 = @intCast(seq_pad * d / 2),
            .u1 = @intCast(seq * d),
            .f0 = scale,
        }, seq_pad * d / 2, @as(usize, 1), @as(usize, 1) });
        const gth = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.gather_kmajor_h16, k_d, @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, null), k16, gpu.EltPush{
            .u0 = @intCast(d * kv_pad / 2),
            .u1 = @intCast(hd),
            .u2 = @intCast(kv_pad),
            .u3 = @intCast(kv_seq),
            .u4 = @intCast(heads),
        }, d * kv_pad / 2, @as(usize, 1), @as(usize, 1) });
        const push = gpu.EltPush{
            .u0 = @intCast(heads * hd),
            .u1 = @intCast(kv_pad),
            .u3 = 1,
            .u4 = @intCast(heads * hd),
            .u5 = @intCast(seq_pad * d),
            .f0 = @bitCast(@as(u32, @intCast(kv_seq))),
            .f1 = @bitCast(@as(u32, @intCast(seq_pad))),
        };
        const md = try T.run(iters, gpu.Context.opFlash, .{ ctx, gpu.FlashPass.md, q16, k16, v16, o_d, push, seq_pad / 128, heads });
        const out = try T.run(iters, gpu.Context.opFlash, .{ ctx, gpu.FlashPass.out, q16, k16, v16, o_d, push, seq_pad / 128, heads });
        const flop = 4.0 * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(kv_seq)) * @as(f64, @floatFromInt(d));
        const tot = cvt + 2 * gth + md + out;
        attn_ms += tot;
        try stdout.print("  {s} {d:>5}q x {d:>5}kv  cvt {d:5.2} + gather {d:5.2} + md {d:6.2} + out {d:6.2} = {d:7.2} ms  {d:5.1} TFLOP/s\n", .{
            cs[0], seq, kv_seq, cvt, gth, md, out, tot, flop / (md + out) / 1e9,
        });
    }

    // --- elementwise, judged by achieved BANDWIDTH -----------------------------
    try stdout.print("\n-- elementwise (bytes/time; the 3090's roof is 936 GB/s) --\n", .{});
    const fseq: f64 = @floatFromInt(seq);
    const fd: f64 = @floatFromInt(d);
    var elt_ms: f64 = 0;
    {
        const ln = try T.run(iters, gpu.Context.opLnModSg, .{ ctx, x_d, y_d, mod_d, seq, d, @as(usize, d), @as(usize, 0), cfg.norm_eps });
        const ga = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.gated_add, x_d, @as(?gpu.DeviceBuffer, y_d), @as(?gpu.DeviceBuffer, mod_d), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
            .u0 = @intCast(seq * d),
            .u1 = @intCast(d),
            .u2 = 0,
        }, seq * d, @as(usize, 1), @as(usize, 1) });
        const qn = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.rmsnorm, x_d, @as(?gpu.DeviceBuffer, x_d), @as(?gpu.DeviceBuffer, nrm_d), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
            .u0 = @intCast(seq * heads),
            .u1 = @intCast(hd),
            .f0 = cfg.qk_eps,
        }, seq * heads, @as(usize, 1), @as(usize, 1) });
        const rp = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.rope_half, x_d, @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, frq_d), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
            .u0 = @intCast(seq * heads * half),
            .u1 = @intCast(half),
            .u2 = @intCast(seq * half),
            .u3 = @intCast(heads),
        }, seq * heads * half, @as(usize, 1), @as(usize, 1) });
        const gl = try T.run(iters, gpu.Context.opElt, .{ ctx, gpu.Elt.gelu_erf, x_d, @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, null), @as(?gpu.DeviceBuffer, null), gpu.EltPush{
            .u0 = @intCast(seq * mlp),
        }, seq * mlp, @as(usize, 1), @as(usize, 1) });
        // The subgroup RMSNorm exists on this backend and the qk-norm was not using it.
        // Timed alongside the thread-per-row one so the choice is measured, not assumed:
        // `rmsnorm`'s thread owns a whole 128-wide head, so a warp's 32 loads land 512 B
        // apart and each is its own sector, the identical mistake `qk_rmsnorm_warp` fixed
        // on CUDA, where it was worth 17x.
        const qs = if (ctx.hasSubgroupNorm())
            try T.run(iters, gpu.Context.opRmsNormSg, .{ ctx, x_d, x_d, nrm_d, seq * heads, hd, cfg.qk_eps })
        else
            0;
        const rows = [_]struct { n: []const u8, ms: f64, bytes: f64, per: usize }{
            .{ .n = "ln_mod_sg", .ms = ln, .bytes = fseq * fd * 4 * 4, .per = 3 },
            .{ .n = "gated_add", .ms = ga, .bytes = fseq * fd * 4 * 3, .per = 3 },
            .{ .n = "rmsnorm (qk) thread/row", .ms = qn, .bytes = fseq * fd * 4 * 2, .per = 3 },
            .{ .n = "rmsnorm (qk) subgroup", .ms = qs, .bytes = fseq * fd * 4 * 2, .per = 0 },
            .{ .n = "rope_half", .ms = rp, .bytes = fseq * fd * 4 * 2, .per = 2 },
            .{ .n = "gelu_erf (mlp)", .ms = gl, .bytes = fseq * @as(f64, @floatFromInt(mlp)) * 4 * 2, .per = 1 },
        };
        for (rows) |r| {
            if (r.ms == 0) continue;
            elt_ms += r.ms * @as(f64, @floatFromInt(r.per));
            try stdout.print("  {s:<24} x{d}  {d:7.2} ms  {d:6.0} GB/s\n", .{ r.n, r.per, r.ms, r.bytes / r.ms / 1e6 });
        }
    }

    // --- roll-up ---------------------------------------------------------------
    const nl: f64 = @floatFromInt(cfg.n_layers);
    const bf_step = (bf_ms + attn_ms + elt_ms) * nl;
    const i8_step = (i8_ms + prep_ms + attn_ms + elt_ms) * nl;
    try stdout.print("\n-- one {d}-block step (sync-per-op, so an UPPER bound) --\n", .{cfg.n_layers});
    try stdout.print("  bf16: gemm {d:6.0} + attn {d:6.0} + elt {d:5.0} = {d:6.0} ms\n", .{ bf_ms * nl, attn_ms * nl, elt_ms * nl, bf_step });
    try stdout.print("  int8: gemm {d:6.0} + prep {d:5.0} + attn {d:6.0} + elt {d:5.0} = {d:6.0} ms\n", .{ i8_ms * nl, prep_ms * nl, attn_ms * nl, elt_ms * nl, i8_step });
    try stdout.print("  => int8 is {d:4.2}x the bf16 step; the GEMMs alone are {d:4.2}x, and prep gives back {d:4.1}%%\n", .{
        bf_step / i8_step, bf_ms / i8_ms, 100 * prep_ms / (i8_ms + prep_ms),
    });
}

/// Validate `anima_cuda` against the CPU ops and the CPU forward. Two tiers on
/// purpose: the KERNEL checks need no checkpoint and localize a mismatch to one op,
/// and the whole-forward check on real weights is what catches a wiring mistake that
/// every individual kernel survives. Exits non-zero if anything fails, so it works as
/// a gate. (A CLI command rather than a unit test because the test binary brings up no
/// CUDA context, matching `zimage-cuda-test` / `sd-cuda-test`.)
fn animaCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const anima = TensorPencil.models.anima;
    const anima_cuda = TensorPencil.models.anima_cuda;
    const ops = TensorPencil.ops;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== anima-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    var prng = std.Random.DefaultPrng.init(0xa71a);
    const rnd = prng.random();
    var failures: usize = 0;

    // --- kernel: fused weightless LayerNorm + AdaLN modulation ------------------
    // Includes a row whose MEAN dwarfs its spread, which is where the shifted
    // `E[x^2]-E[x]^2` variance cancels catastrophically in f32. A mean-0 case alone
    // cannot tell the two forms apart, and `ops.norm.groupNorm` records what that
    // costs. The reference is f64 for the same reason: at mean 400 the f32 host path's
    // own serial sum carries more error than the device's block reduction.
    try stdout.print("\n-- kernels --\n", .{});
    for ([_]f32{ 0, 400 }) |bias| {
        const rows = 300;
        const dim = anima.anima_2b.dim;
        const x = try arena.alloc(f32, rows * dim);
        for (x) |*v| v.* = rnd.floatNorm(f32) * 1.5 + bias;
        const mod = try arena.alloc(f32, 2 * dim);
        for (mod[0..dim]) |*v| v.* = 1.0 + rnd.floatNorm(f32) * 0.3; // 1 + scale, folded
        for (mod[dim..]) |*v| v.* = rnd.floatNorm(f32) * 0.2;

        var x_d = try be.tensorCreate(x.len * 4);
        defer be.tensorDestroy(&x_d);
        var m_d = try be.tensorCreate(mod.len * 4);
        defer be.tensorDestroy(&m_d);
        var y_d = try be.tensorCreate(x.len * 4);
        defer be.tensorDestroy(&y_d);
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
        try be.tensorUpload(m_d, std.mem.sliceAsBytes(mod));
        try be.lnMod(x_d, y_d, m_d, rows, dim, 0, dim, anima.anima_2b.norm_eps);
        const got = try arena.alloc(f32, x.len);
        try be.tensorDownload(y_d, std.mem.sliceAsBytes(got));

        var num: f64 = 0;
        var den: f64 = 0;
        for (0..rows) |r| {
            const xr = x[r * dim ..][0..dim];
            var sum: f64 = 0;
            for (xr) |v| sum += v;
            const mean = sum / @as(f64, @floatFromInt(dim));
            var vs: f64 = 0;
            for (xr) |v| vs += (@as(f64, v) - mean) * (@as(f64, v) - mean);
            const inv = 1.0 / @sqrt(vs / @as(f64, @floatFromInt(dim)) + anima.anima_2b.norm_eps);
            for (0..dim) |i| {
                const want = (@as(f64, xr[i]) - mean) * inv * mod[i] + mod[dim + i];
                const d = @as(f64, got[r * dim + i]) - want;
                num += d * d;
                den += want * want;
            }
        }
        const rel = @sqrt(num / den);
        const ok = rel < 1e-4;
        if (!ok) failures += 1;
        try stdout.print("  lnMod (row mean {d:>3})       rel L2 {e:.4} vs f64  {s}\n", .{ bias, rel, if (ok) "ok" else "FAILED" });
    }

    // --- kernel: rectangular tensor-core attention ------------------------------
    // This is the new capability the cross-attention path rests on, so it is
    // checked at Anima's real shape (512-row context, 16 heads of 128) AND at a
    // non-multiple-of-128 key length, which is what exercises the padded-key masking.
    {
        const heads = anima.anima_2b.n_heads;
        const hd = anima.anima_2b.headDim();
        const dim = heads * hd;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const shapes = [_][2]usize{ .{ 300, 512 }, .{ 260, 200 }, .{ 128, 128 } };
        for (shapes) |sh| {
            const nq = sh[0];
            const nkv = sh[1];
            const q = try arena.alloc(f32, nq * dim);
            const k = try arena.alloc(f32, nkv * dim);
            const v = try arena.alloc(f32, nkv * dim);
            for (q) |*t| t.* = rnd.floatNorm(f32);
            for (k) |*t| t.* = rnd.floatNorm(f32);
            for (v) |*t| t.* = rnd.floatNorm(f32);
            const want = try arena.alloc(f32, nq * dim);
            try ops.attention.attention(io, arena, want, q, k, v, .{
                .seq_q = nq,
                .seq_kv = nkv,
                .n_heads = heads,
                .n_kv_heads = heads,
                .head_dim = hd,
            });

            var q_d = try be.tensorCreate(q.len * 4);
            defer be.tensorDestroy(&q_d);
            var k_d = try be.tensorCreate(k.len * 4);
            defer be.tensorDestroy(&k_d);
            var v_d = try be.tensorCreate(v.len * 4);
            defer be.tensorDestroy(&v_d);
            var o_d = try be.tensorCreate(q.len * 4);
            defer be.tensorDestroy(&o_d);
            try be.tensorUpload(q_d, std.mem.sliceAsBytes(q));
            try be.tensorUpload(k_d, std.mem.sliceAsBytes(k));
            try be.tensorUpload(v_d, std.mem.sliceAsBytes(v));
            try be.opAttnTCRect(q_d, k_d, v_d, o_d, nq, nkv, heads, heads, hd, scale);
            const got = try arena.alloc(f32, q.len);
            try be.tensorDownload(o_d, std.mem.sliceAsBytes(got));

            var num: f64 = 0;
            var den: f64 = 0;
            var bad: usize = 0;
            for (want, got) |e, a| {
                if (!std.math.isFinite(a)) bad += 1;
                num += (@as(f64, e) - a) * (@as(f64, e) - a);
                den += @as(f64, e) * e;
            }
            const rel = if (den == 0) 0 else @sqrt(num / den);
            // f16 operands against the CPU's f32 accumulation, the regime every
            // tensor-core attention here sits in.
            const ok = bad == 0 and rel < 5e-3;
            if (!ok) failures += 1;
            try stdout.print("  opAttnTCRect {d:>4}q x {d:>4}kv  rel L2 {e:.4}  {s}{s}\n", .{
                nq, nkv, rel, if (ok) "ok" else "FAILED", if (bad != 0) " (non-finite)" else "",
            });
        }
    }

    // --- the whole forward on real weights --------------------------------------
    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("\n(skipping the forward: {s} not found)\n", .{ckpt});
        try summarize(stdout, failures);
        return;
    };
    var ck = try safetensors.open(arena, io, ckpt);
    defer ck.deinit();
    try stdout.print("\n-- forward (real weights) --\n", .{});
    // Depth 1 AND depth 2, because that is the ATTRIBUTION on a mixed checkpoint.
    // A quantized Anima file leaves block 0 dense and quantizes block 1 onward, so depth 1
    // measures the dense pipeline alone and depth 2 adds exactly one quantized block. Two
    // numbers separate "the port is wrong" from "int8 changes the answer", which one number
    // cannot.
    //
    // The deeper points exist because "depth adds no new WIRING" is a trap: it is true
    // of the wiring and false of a defect that only manifests with depth. A broken int4
    // activation prep reads 3.86e-2 at depth 2, exactly int4's expected coarseness, so a
    // single depth cannot tell "coarse" from "accumulating". What such a defect changes
    // is the SHAPE of the curve, not its first point: int8 and W4A8 stay flat from 2 to 28
    // (2.1e-3 -> 7-9e-3) while broken int4 climbed 3.9e-2 -> 5.2e-1 and healthy int4 is
    // flat at ~3.9e-2. A single depth cannot distinguish "coarse" from "accumulating",
    // and that distinction is the whole diagnostic. Cheap: the deep CPU reference at a
    // 24x32 latent is seconds.
    for ([_]usize{ 1, 2, 8, 28 }) |depth| {
        var cfg = anima.anima_2b;
        cfg.n_layers = depth;
        var model = try anima.DiT.load(arena, .{ .safetensors = &ck }, cfg);
        defer model.deinit();
        if (!anima_cuda.supported(&model)) {
            try stdout.print("  checkpoint dtype unsupported on this backend\n", .{});
            break;
        }
        // `unsupportedLin` with the no-convrot support set reports the first int8/int4
        // linear, which is exactly "is this depth quantized, and as what".
        const qdt: ?TensorPencil.dtype.DType = if (anima.unsupportedLin(&model, .{})) |q| q.dtype else null;
        failures += try animaForwardCheck(arena, io, stdout, be, &model, depth, qdt, rnd);
    }

    try summarize(stdout, failures);
}

/// One depth's device-vs-CPU forward check, on both attention arms. Returns the failure
/// count. `quant` widens the tolerance for the reason printed alongside it.
fn animaForwardCheck(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    be: *TensorPencil.gpu.cuda.Backend,
    model: *const TensorPencil.models.anima.DiT,
    depth: usize,
    qdt: ?TensorPencil.dtype.DType,
    rnd: std.Random,
) !usize {
    const anima = TensorPencil.models.anima;
    const anima_cuda = TensorPencil.models.anima_cuda;
    const cfg = model.cfg;
    const lat_h = 24;
    const lat_w = 32;
    const ctx_seq = cfg.adapter.min_rows; // 512, what a real prompt gives
    const cond = try arena.alloc(f32, ctx_seq * cfg.context_dim);
    for (cond) |*t| t.* = rnd.floatNorm(f32) * 0.5;
    const x_lat = try arena.alloc(f32, cfg.channels * lat_h * lat_w);
    for (x_lat) |*t| t.* = rnd.floatNorm(f32);
    const sigma: f32 = 0.75;

    const want = try arena.alloc(f32, x_lat.len);
    {
        const mod = try model.modulationTable(io, arena, sigma);
        try model.predict(io, arena, want, x_lat, lat_h, lat_w, cond, ctx_seq, mod, null);
    }
    // The device sessions borrow one folded schedule, shared by both branches.
    const sched = try model.modulationSchedule(io, arena, &.{ sigma, 0 });

    // Tolerances, and why each is the value it is.
    //
    // A quantized checkpoint is a DIFFERENT computation on the two sides, not a different
    // rounding of the same one: the CPU `matmul` dequantizes the weight and multiplies in
    // f32 (W8A16, weight-only) while `opI8Prep`/`opI8Gemm` also quantize the ACTIVATION
    // (W8A8), so the residual includes an activation quantization the reference never
    // does.
    //
    // Depth 1 is looser than depth 2, which looks backwards and is not: with one block
    // the trunk barely transforms `x`, so the magnitude the relative error is measured
    // against is smaller (1.42e-3 against 8.0e-4 on a dense checkpoint). Those depth-1
    // figures are IDENTICAL between a dense file and its quantized sibling, block 0 being
    // bf16 in both, which is the cross-check that quantization changed nothing it should
    // not have.
    //
    // int4 gets ~16x int8's bound because it has 16 levels against 256, and that RATIO is
    // the receipt: predicted 16x from the bit width, measured 18x (3.80e-2 against
    // 2.09e-3). A wiring error would not scale with the level count, and both attention
    // arms agree to three digits, so it is not attention either.
    //
    // W4A8 takes int8's bound because it IS int8 on the device: its decode emits
    // int8-convrot values that run the same prep and GEMM, so it must land where int8
    // lands. The same model quantized both ways measures 2.0721e-3 against 2.0968e-3, a
    // 1.2% gap. The dense bound of 1e-3 fails a correct implementation here.
    //
    // The growth with depth is MEASURED, not chosen to make the deep points pass: every
    // healthy arm accumulates by ~4x from depth 2 to 28 (int8 2.07e-3 -> 7.34e-3, W4A8
    // 2.10e-3 -> 8.91e-3, int4 3.80e-2 -> 1.59e-1), so `1 + depth/8` tracks the real
    // curve with roughly 3x headroom throughout.
    //
    // It still has teeth at the depth that matters, which is what makes a depth-scaled
    // bound honest: an `opI4Prep` truncation reads 2.96e-1 at depth 8 against this rule's
    // 1.2e-1, and would PASS a bound keyed on depth 28 alone (2.13e-1 against 2.7e-1)
    // because that defect saturates rather than diverging. Hence the intermediate depths.
    const base: f64 = switch (qdt orelse .f32) {
        .i4 => 6e-2,
        .i8, .w4a8 => 6e-3,
        else => if (depth == 1) 2.5e-3 else 1e-3,
    };
    const tol: f64 = base * (1.0 + @as(f64, @floatFromInt(depth)) / 8.0);
    var failures: usize = 0;
    const saved = anima_cuda.force_naive_attn;
    defer anima_cuda.force_naive_attn = saved;
    for ([_]bool{ false, true }) |naive| {
        anima_cuda.force_naive_attn = naive;
        var sess = try anima_cuda.Session.init(arena, be, model, lat_h, lat_w, cond, ctx_seq, &.{ sigma, 0 }, sched);
        defer sess.deinit(arena, be);
        var ws = try anima_cuda.Workspace.init(be, model, lat_h, lat_w);
        defer ws.deinit(be);
        const got = try arena.alloc(f32, x_lat.len);
        const t0 = std.Io.Clock.real.now(io);
        try anima_cuda.forward(model, be, &sess, &ws, io, arena, got, x_lat, sigma, null);
        const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;

        var num: f64 = 0;
        var den: f64 = 0;
        var bad: usize = 0;
        for (want, got) |e, a| {
            if (!std.math.isFinite(a)) bad += 1;
            num += (@as(f64, e) - a) * (@as(f64, e) - a);
            den += @as(f64, e) * e;
        }
        const rel = if (den == 0) 0 else @sqrt(num / den);
        const ok = bad == 0 and rel < tol;
        if (!ok) failures += 1;
        try stdout.print("  {d} block{s} {s:<7} ({s:<11}) {d:7.0} ms  rel L2 {e:.4}  {s}{s}\n", .{
            depth,
            if (depth == 1) " " else "s",
            if (qdt) |d| @tagName(d) else "dense",
            if (naive) "naive attn" else "tensor core",
            dt,
            rel,
            if (ok) "ok" else "FAILED",
            if (bad != 0) " (non-finite output)" else "",
        });
    }
    _ = anima;
    return failures;
}

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
    // pointer, so a shared buffer would also share the device copy, which is what
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

    // Timed per op FAMILY, not as one lump. "elementwise is 30% of the step"
    // is not actionable, these are pure-bandwidth kernels, so the question is
    // which of them is far from the 936 GB/s the card can do, and a single total
    // averages the answer away.
    const ea = .{
        .x = y,
        .nrm = nrm,
        .mv = mv,
        .q = q,
        .k = kk,
        .nw = nw,
        .frq = frq,
        .mg = mg,
        .mu = mu,
        .seq = seq,
        .dim = cfg.dim,
        .mlp = cfg.mlp_dim,
        .h = heads,
        .hd = hd,
        .zoff = blocks * 4 * cfg.dim - cfg.dim,
        .nb = blocks,
    };
    const E = struct {
        // Each is the exact call `zimage_cuda.blockForward` makes, per block.
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

    // --- the VAE mid-block attention, isolated -----------------------------------
    // Its own section because it is a DIFFERENT shape from the DiT's: one head
    // over 512 channels attending across every latent position, so seq is ~26k where
    // the DiT's heads are 30x128. That difference is why the naive per-query kernel
    // is merely slow in the DiT and catastrophic here.
    {
        // The VAE attends at LATENT resolution over every position, so its seq is
        // the latent size, not the DiT token count. Pass the bench a latent-sized seq
        // (26136 = 132x198, a 1056x1584 render) to see the shape that actually runs.
        const vseq = seq;
        const vhd: usize = 512;
        var vq: Buf = .{};
        var vk: Buf = .{};
        var vv: Buf = .{};
        var vo: Buf = .{};
        inline for (.{ &vq, &vk, &vv, &vo }) |bp| try be.ensureDeviceBuffer(bp, vseq * vhd * 4);
        {
            const host = try arena.alloc(f32, vseq * vhd);
            defer arena.free(host);
            var pr = std.Random.DefaultPrng.init(5);
            for (host) |*hv| hv.* = pr.random().floatNorm(f32);
            inline for (.{ vq, vk, vv }) |bb| try be.tensorUpload(bb, std.mem.sliceAsBytes(host));
        }
        const vscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(vhd)));
        const A = struct {
            fn naive(b: *cuda.Backend, ar: anytype) !void {
                try b.beginBatch();
                try b.attn(ar.q, ar.k, ar.v, ar.o, ar.n, ar.n, 1, 1, ar.hd, ar.sc, false);
                try b.endBatch();
            }
            fn tc(b: *cuda.Backend, ar: anytype) !void {
                try b.beginBatch();
                try b.opAttnTC(ar.q, ar.k, ar.v, ar.o, ar.n, 1, 1, ar.hd, ar.sc);
                try b.endBatch();
            }
        };
        const va = .{ .q = vq, .k = vk, .v = vv, .o = vo, .n = vseq, .hd = vhd, .sc = vscale };
        // 4*seq²*hd: QK then PV, no mask.
        const vflop = 4.0 * @as(f64, @floatFromInt(vseq)) * @as(f64, @floatFromInt(vseq)) * @as(f64, @floatFromInt(vhd));
        try stdout.print("\n-- VAE mid-block attention (seq={d}, 1 head x {d}) --\n", .{ vseq, vhd });
        const t_naive = try timeBest(io, 1, A.naive, be, va);
        const t_tc = timeBest(io, 2, A.tc, be, va) catch std.math.nan(f64);
        try stdout.print("be.attn (thread/query) {d:8.0} ms  {d:6.2} TFLOP/s\n", .{ t_naive, vflop / (t_naive / 1e3) / 1e12 });
        try stdout.print("opAttnTC               {d:8.0} ms  {d:6.2} TFLOP/s  ({d:.1}x)\n", .{ t_tc, vflop / (t_tc / 1e3) / 1e12, t_naive / t_tc });
    }

    const total = t_full + t_attn + t_elt;
    // Attention is 4*seq²*hd per head (QK then PV), both at full width, Z-Image
    // has no causal mask, so nothing is skipped.
    const aflop = 4.0 * @as(f64, @floatFromInt(seq)) * @as(f64, @floatFromInt(seq)) *
        @as(f64, @floatFromInt(hd * heads * blocks));
    try stdout.print("\n-- one step's device work, by part --\n", .{});
    try stdout.print("GEMMs (with conv) : {d:7.1} ms  {d:5.1}%\n", .{ t_full, 100 * t_full / total });
    try stdout.print("attention         : {d:7.1} ms  {d:5.1}%   ({d:.1} TFLOP/step, {d:.1} TFLOP/s)\n", .{ t_attn, 100 * t_attn / total, aflop / 1e12, aflop / (t_attn / 1e3) / 1e12 });
    try stdout.print("elementwise       : {d:7.1} ms  {d:5.1}%\n", .{ t_elt, 100 * t_elt / total });
    try stdout.print("                    {d:7.1} ms total device work\n", .{total});
    // The achieved bandwidth is the number that says whether a kernel is
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
    const sigma: f32 = if (std.c.getenv("TP_H3_SIGMA")) |v| (std.fmt.parseFloat(f32, std.mem.span(v)) catch 0.7) else 0.7;
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
    // the whole model pins in, this is the ~12 GB host->VRAM upload. SYNC pageable
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

    // Async staging-ring COLD-LOAD: bounded 512 MB pinned ring (NOT registerHost,
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

/// Write one hand-PTX kernel's generated source to a file, so it can be fed to
/// `ptxas -arch=sm_86 -v` offline.
///
/// The driver JITs these at load time and reports neither the register count nor
/// spill traffic, and `ncu` needs the GPU performance counters unlocked by root.
/// ptxas on the dumped file needs neither, and spill stores/loads are the number
/// that decides whether a register-hungry GEMM kernel can take another live value.
/// Needs no device: these builders are pure string generation.
fn dumpPtx(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, which: []const u8, out_path: []const u8) !void {
    const kern = TensorPencil.gpu.cuda.kernels;
    const known = "mmq_pipe_q4_k | mmq_pipe_q5_k | mmq_pipe_q6_k | mmq_pipe_q1_0 | mmq_pipe_q8_0 | mmq_q4_k | igemm_pipe_fused | hgemm";
    if (which.len == 0 or out_path.len == 0) {
        try stdout.print("usage: dump-ptx <kernel> <out.ptx>\n  kernels: {s}\n", .{known});
        return;
    }
    const eq = std.mem.eql;
    const ptx = if (eq(u8, which, "mmq_pipe_q4_k")) try kern.buildMmqPipeQ4K(arena, .q4_k) else if (eq(u8, which, "mmq_pipe_q5_k")) try kern.buildMmqPipeQ4K(arena, .q5_k) else if (eq(u8, which, "mmq_pipe_q6_k")) try kern.buildMmqPipeQ6K(arena) else if (eq(u8, which, "mmq_pipe_q1_0")) try kern.buildMmqPipeQ1_0(arena) else if (eq(u8, which, "mmq_pipe_q8_0")) try kern.buildMmqPipeQ8_0(arena) else if (eq(u8, which, "mmq_q4_k")) try kern.buildMmqQ4K(arena, 16, 4) else if (eq(u8, which, "igemm_pipe_fused")) try kern.buildIgemmPipe(arena, 64, true, 8, true) else if (eq(u8, which, "hgemm")) try kern.buildHgemm(arena, false, false, false, false, true, false) else {
        try stdout.print("unknown kernel '{s}'\n  kernels: {s}\n", .{ which, known });
        return;
    };
    var f = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var w = f.writer(io, &buf);
    try w.interface.writeAll(ptx);
    try w.interface.flush();
    try stdout.print("wrote {s} ({d} bytes)\n", .{ out_path, ptx.len });
}

fn cudaDitTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, lat: usize, use_loop: bool, use_libs: bool) !void {
    const dit_mod = TensorPencil.models.dit;
    const dit_cuda = TensorPencil.models.dit_cuda;
    const cuda = TensorPencil.gpu.cuda;

    // Container.open, not SafeTensors.open: a q4_k DiT is a GGUF, and handing one to
    // the safetensors reader reports `InvalidHeader`, which says nothing.
    var container = try TensorPencil.pipeline.Container.open(arena, io, path);
    defer container.deinit();
    var model = try dit_mod.DiT.load(arena, container.store());
    defer model.deinit();
    const wqt = model.blocks[0].attn.wq.dtype;
    if (!dit_mod.gpuLinKindSupported(wqt, .cuda)) {
        try stdout.print("cuda-dit-test needs a checkpoint the CUDA DiT has a GEMM for (wq.dtype={t})\n", .{wqt});
        return;
    }
    const qtag: []const u8 = @tagName(wqt);

    const seq_txt: usize = 8;
    const sigma: f32 = if (std.c.getenv("TP_H3_SIGMA")) |v| (std.fmt.parseFloat(f32, std.mem.span(v)) catch 0.7) else 0.7;
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
    // Batched (profile off) timing, the real steady-state s/step.
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
    // Names come off ProfCat itself. A hand-written list silently mislabels every row
    // after the first one it is short of: it omitted `dequant`, so `prep` printed the
    // weight-dequant total, `attn` printed prep, `elt` printed attn, and `attn_pv` was
    // never printed at all.
    var sync_total: f64 = 0;
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| {
        const i = @intFromEnum(cat);
        const name = switch (cat) {
            .attn => "attn(g/s)",
            .attn_scores => "  scores",
            .attn_softmax => "  softmax",
            .attn_pv => "  pv",
            else => @tagName(cat),
        };
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
        // The CPU reference is weight-only, so this measures activation quantization
        // amplified over 28 blocks, and the bound scales with the WEIGHT's coarseness:
        // int8 activations put q4_k at 0.044 and q2_k at 0.094, int4's at 0.15-0.20.
        // Keyed on the ROUTE, not the storage, since a GGUF can take either GEMM.
        //
        // These bounds are wide because the metric's spread is wide, and both causes
        // are properties of the harness rather than of the kernels:
        //
        //   - `x` and `cond` are random normals, so the forward runs far OFF the data
        //     manifold, where reduced-precision activations diverge much more than on
        //     a real latent. A checkpoint measuring 0.13 here renders cleanly.
        //   - the spread across CHECKPOINTS of one architecture and format is ~2x
        //     (int8-convrot: 0.068 animosity, 0.108 gonzalomo, 0.131 center-semiraw),
        //     and a checkpoint that is a requantization of an already-quantized model
        //     sits at the top of it whatever format it was requantized INTO: that one
        //     base measures 0.117 (nvfp4), 0.125 (fp8), 0.129 (w4a8), 0.131 (int8),
        //     five unrelated weight paths inside one narrow band, which is the tell
        //     that the weights, not the route, set the level.
        //
        // So this gate catches a wiring break (which lands near 1.0, not 0.13) and
        // deliberately does not try to bound quantization quality; `--dit` renders and
        // the PSNR tables in BACKEND.md are what measure that.
        const tol: f32 = if (dit_cuda.activationIs4Bit(wqt))
            0.25
        else if (wqt == .q2_k)
            0.15
        else
            0.18;
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
    // Tracked separately from `opts.prompt.len`: an EMPTY prompt is a valid render
    // (it encodes to the unconditional embedding), so only the absent flag is an error.
    var prompt_given = false;
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
            prompt_given = true;
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
        } else if (std.mem.eql(u8, flag, "--dit-gguf-gemm")) {
            TensorPencil.models.dit_cuda.blockq_gemm = std.meta.stringToEnum(TensorPencil.models.dit_cuda.BlockQGemm, val) orelse {
                try stdout.print("unknown gguf gemm target '{s}' (expected: auto, int8, int4, f16, mmq)\n", .{val});
                return error.InvalidArgs;
            };
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
            // pread (default) | mmap | buffered, see safetensors.ReadMode.
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
    if (!prompt_given) {
        try stdout.print("--prompt is required (it may be empty: --prompt \"\")\n", .{});
        return error.InvalidArgs;
    }

    var cap: PreviewCap = .{ .arena = arena };
    if (opts.preview) opts.on_step = .{ .ctx = &cap, .step = PreviewCap.onStep };

    // --repeat N>1: keep ONE pipeline.Session resident across N images (the GUI's
    // cross-queue path) and time each, the model loads only on image 1; images
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

    var ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
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

/// `generate-clip`: a MiniMax H3 render, video plus its soundtrack.
///
/// Separate from `generate` rather than a flag on it, because the result is a
/// different thing: a clip has frames, a frame rate and audio, and no single
/// image to write. The default output is a muxed MP4 (`src/av.zig`); `--frames`
/// writes a PNG sequence plus a WAV instead, for inspecting individual frames.
fn generateClip(gpa: std.mem.Allocator, io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const av = @import("av");
    var opts: pipeline.Options = .{ .prompt = "" };
    var clip_opts: pipeline.Session.ClipOptions = .{ .prompt = "" };
    var out_dir: []const u8 = "scratch_out/clip";
    // Default is an MP4; `--frames` writes a PNG sequence + WAV instead, which is
    // what you want when inspecting individual frames.
    var frames_only = false;
    // Reference pictures, in REQUEST order: the presentation's `<Picture i>`
    // ordinals count in the order they arrive on the command line.
    var ref_paths: std.ArrayList([]const u8) = .empty;
    defer ref_paths.deinit(gpa);
    // The reference's `ref_image_size` policy. `match` keeps a reference no more
    // expensive than a frame; `max` spends a 2048 px short edge on identity
    // fidelity. Reference tokens ride through EVERY sampling step, so `max` can be
    // several times slower.
    var ref_match = true;
    // Keyframes, which are NOT references: they anchor at a pixel frame of the
    // target and share its canvas. The first frame is a plain stretch (the
    // geometry the clip is built on); a last frame is a cover crop.
    var first_frame: ?[]const u8 = null;
    var last_frame: ?[]const u8 = null;
    // Reference videos, each a DIRECTORY of `frame_NNNNN.png`, which is exactly
    // the layout `--frames` writes: a rendered clip round-trips as a reference.
    var ref_video_dirs: std.ArrayList([]const u8) = .empty;
    defer ref_video_dirs.deinit(gpa);
    // One entry per `--ref-video`, null unless a `--ref-video-audio` followed it.
    var ref_video_audio: std.ArrayList(?[]const u8) = .empty;
    defer ref_video_audio.deinit(gpa);
    var ref_audio_paths: std.ArrayList([]const u8) = .empty;
    defer ref_audio_paths.deinit(gpa);
    // Continuation: a frames directory (plus its audio.wav if present) and how much
    // of it to hold fixed.
    var continue_dir: ?[]const u8 = null;
    var preserve_frames: usize = 0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        const val: ?[]const u8 = if (i + 1 < args.len) args[i + 1] else null;
        if (std.mem.eql(u8, flag, "--prompt")) {
            clip_opts.prompt = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--dit")) {
            opts.dit_path = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--text-encoder")) {
            opts.text_encoder_path = val orelse return error.MissingValue;
            opts.explicit_text_encoder = true;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--vae")) {
            opts.vae_path = val orelse return error.MissingValue;
            opts.explicit_vae = true;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--audio-vae")) {
            opts.vae_2_path = val orelse return error.MissingValue;
            opts.explicit_vae_2 = true;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--ref-video")) {
            try ref_video_dirs.append(gpa, val orelse return error.MissingValue);
            try ref_video_audio.append(gpa, null);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--ref-video-audio")) {
            // Attaches to the MOST RECENT --ref-video, so the two are paired
            // positionally rather than by a name the caller has to invent.
            if (ref_video_dirs.items.len == 0) {
                try stdout.print("--ref-video-audio must follow a --ref-video\n", .{});
                return error.BadArgs;
            }
            ref_video_audio.items[ref_video_audio.items.len - 1] = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--continue-from")) {
            continue_dir = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--preserve-frames")) {
            preserve_frames = std.fmt.parseInt(usize, val orelse return error.MissingValue, 10) catch return error.BadArgs;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--ref-audio")) {
            try ref_audio_paths.append(gpa, val orelse return error.MissingValue);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--first-frame")) {
            first_frame = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--last-frame")) {
            last_frame = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--ref-image")) {
            try ref_paths.append(gpa, val orelse return error.MissingValue);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--ref-size")) {
            const v = val orelse return error.MissingValue;
            if (std.mem.eql(u8, v, "match")) {
                ref_match = true;
            } else if (std.mem.eql(u8, v, "max")) {
                ref_match = false;
            } else return error.BadArgs;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--lora")) {
            opts.lora_path = val orelse return error.MissingValue;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--lora-strength")) {
            opts.lora_strength = try std.fmt.parseFloat(f32, val orelse return error.MissingValue);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--width")) {
            clip_opts.width = try std.fmt.parseInt(usize, val orelse return error.MissingValue, 10);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--height")) {
            clip_opts.height = try std.fmt.parseInt(usize, val orelse return error.MissingValue, 10);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--length")) {
            clip_opts.length = try std.fmt.parseInt(usize, val orelse return error.MissingValue, 10);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--steps")) {
            clip_opts.steps = try std.fmt.parseInt(usize, val orelse return error.MissingValue, 10);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--seed")) {
            clip_opts.seed = try std.fmt.parseInt(u64, val orelse return error.MissingValue, 10);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--shift")) {
            clip_opts.shift = try std.fmt.parseFloat(f32, val orelse return error.MissingValue);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--shift-audio")) {
            clip_opts.shift_audio = try std.fmt.parseFloat(f32, val orelse return error.MissingValue);
            i += 1;
        } else if (std.mem.eql(u8, flag, "--backend")) {
            opts.backend = std.meta.stringToEnum(@TypeOf(opts.backend), val orelse return error.MissingValue) orelse return error.BadBackend;
            i += 1;
        } else if (std.mem.eql(u8, flag, "--frames")) {
            frames_only = true;
        } else if (std.mem.eql(u8, flag, "--out")) {
            out_dir = val orelse return error.MissingValue;
            i += 1;
        } else {
            try stdout.print("unknown flag: {s}\n", .{flag});
            return error.BadArgs;
        }
    }
    opts.prompt = clip_opts.prompt;

    if (std.c.getenv("TP_LORA_NO_FUSE") != null) {
        TensorPencil.models.lora_cuda.bench_no_fuse = true;
        try stdout.print("(A/B: sidecar accumulate NOT fused into the B GEMM)\n", .{});
    }
    if (std.c.getenv("TP_LORA_SKIP_ADD") != null) {
        TensorPencil.models.lora_cuda.bench_skip_add = true;
        try stdout.print("(DIAGNOSTIC: sidecar accumulate skipped, this render is WRONG)\n", .{});
    }

    // PNG only, through the engine's own decoder: the diffusion executable links
    // libav for muxing but not libvips, so there is no general image decode here.
    // PNG is what this CLI's own `--frames` output writes, so a rendered frame
    // round-trips as a reference.
    const loadPng = struct {
        fn go(a: std.mem.Allocator, ioo: Io, path: []const u8) !pipeline.Session.RefImage {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(ioo, path, a, .limited(64 << 20));
            defer a.free(bytes);
            const png = try TensorPencil.image.decodePngRgb(a, bytes);
            defer a.free(png.pixels);
            const rgb = try a.alloc(f32, 3 * png.height * png.width);
            for (0..png.height) |y| {
                for (0..png.width) |x| {
                    for (0..3) |c| {
                        rgb[c * png.height * png.width + y * png.width + x] =
                            @as(f32, @floatFromInt(png.pixels[(y * png.width + x) * 3 + c])) / 255.0;
                    }
                }
            }
            return .{ .rgb = rgb, .width = png.width, .height = png.height };
        }
    }.go;

    const refs = try gpa.alloc(pipeline.Session.RefImage, ref_paths.items.len);
    defer {
        for (refs) |r| gpa.free(@constCast(r.rgb));
        gpa.free(refs);
    }
    for (ref_paths.items, refs, 0..) |path, *out, ri| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        const png = TensorPencil.image.decodePngRgb(gpa, bytes) catch |err| {
            try stdout.print("reference {d} ({s}) is not a PNG this build can read: {t}\n", .{ ri + 1, path, err });
            return err;
        };
        defer gpa.free(png.pixels);
        // Planar f32 in [0, 1], which is what both towers take.
        const rgb = try gpa.alloc(f32, 3 * png.height * png.width);
        for (0..png.height) |y| {
            for (0..png.width) |x| {
                for (0..3) |c| {
                    rgb[c * png.height * png.width + y * png.width + x] =
                        @as(f32, @floatFromInt(png.pixels[(y * png.width + x) * 3 + c])) / 255.0;
                }
            }
        }
        out.* = .{ .rgb = rgb, .width = png.width, .height = png.height };
        try stdout.print("reference {d}: {s} ({d}x{d})\n", .{ ri + 1, path, png.width, png.height });
    }
    clip_opts.refs = refs;
    clip_opts.ref_area = if (ref_match) clip_opts.width * clip_opts.height else 0;

    // Keyframes. The LAST frame's index is the SNAPPED frame count minus one, not
    // `--length - 1`: the request snaps up to the model's 17k+5 grid, so an index
    // taken from the request would land short of the clip's real end.
    const snapped = TensorPencil.models.minimax_h3.alignFrameCount(@max(5, clip_opts.length));
    var kfs: std.ArrayList(pipeline.Session.KeyframeReq) = .empty;
    defer {
        for (kfs.items) |k| gpa.free(@constCast(k.img.rgb));
        kfs.deinit(gpa);
    }
    if (first_frame) |path| {
        const img = try loadPng(gpa, io, path);
        try kfs.append(gpa, .{ .img = img, .frame_index = 0, .cover_crop = false });
        try stdout.print("first frame: {s} ({d}x{d})\n", .{ path, img.width, img.height });
    }
    if (last_frame) |path| {
        const img = try loadPng(gpa, io, path);
        try kfs.append(gpa, .{ .img = img, .frame_index = snapped - 1, .cover_crop = true });
        try stdout.print("last frame: {s} ({d}x{d}), anchored at frame {d}\n", .{ path, img.width, img.height, snapped - 1 });
    }
    clip_opts.keyframes = kfs.items;

    // WAV only, through the engine's own reader: this executable links libav for
    // muxing but decoding an arbitrary container here would mean a second
    // dependency. `--frames` writes a WAV beside the frames, so a rendered clip's
    // soundtrack round-trips as a reference.
    const loadWav = struct {
        fn go(a: std.mem.Allocator, ioo: Io, path: []const u8) !TensorPencil.audio.Wave {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(ioo, path, a, .limited(512 << 20));
            defer a.free(bytes);
            return TensorPencil.audio.decodeWav(a, bytes);
        }
    }.go;

    // Reference videos, each with its optional soundtrack.
    const vid_waves = try gpa.alloc(?TensorPencil.audio.Wave, ref_video_dirs.items.len);
    @memset(vid_waves, null);
    defer {
        for (vid_waves) |*w| if (w.*) |*ww| ww.deinit(gpa);
        gpa.free(vid_waves);
    }
    const vids = try gpa.alloc(pipeline.Session.RefVideo, ref_video_dirs.items.len);
    defer {
        for (vids) |v| {
            for (v.frames) |f| gpa.free(@constCast(f.rgb));
            gpa.free(@constCast(v.frames));
        }
        gpa.free(vids);
    }
    for (ref_video_dirs.items, vids, 0..) |dir_path, *out, vi| {
        var frames: std.ArrayList(pipeline.Session.RefImage) = .empty;
        errdefer {
            for (frames.items) |f| gpa.free(@constCast(f.rgb));
            frames.deinit(gpa);
        }
        // Sequential, not a directory listing: `frame_00010` must follow
        // `frame_00009`, and a listing's order is the filesystem's, not the
        // clip's. Stops at the first gap.
        var fi: usize = 0;
        while (true) : (fi += 1) {
            var nb: [512]u8 = undefined;
            const fp = try std.fmt.bufPrint(&nb, "{s}/frame_{d:0>5}.png", .{ dir_path, fi });
            const img = loadPng(gpa, io, fp) catch break;
            try frames.append(gpa, img);
        }
        if (frames.items.len == 0) {
            try stdout.print("reference video {d} ({s}) has no frame_NNNNN.png files\n", .{ vi + 1, dir_path });
            return error.BadArgs;
        }
        try stdout.print("reference video {d}: {s} ({d} frames, {d}x{d})\n", .{
            vi + 1, dir_path, frames.items.len, frames.items[0].width, frames.items[0].height,
        });
        out.* = .{ .frames = try frames.toOwnedSlice(gpa) };
        if (ref_video_audio.items[vi]) |wav_path| {
            const w = loadWav(gpa, io, wav_path) catch |err| {
                try stdout.print("reference video {d}'s soundtrack ({s}) is not a WAV this build can read: {t}\n", .{ vi + 1, wav_path, err });
                return err;
            };
            vid_waves[vi] = w;
            out.soundtrack = .{ .samples = w.samples, .channels = w.channels, .sample_rate = w.sample_rate };
            try stdout.print("  soundtrack: {s} ({d:.2} s, {d} ch @ {d} Hz)\n", .{
                wav_path, w.seconds(), w.channels, w.sample_rate,
            });
        }
    }
    clip_opts.ref_videos = vids;

    // Standalone reference soundtracks.
    const aud_waves = try gpa.alloc(TensorPencil.audio.Wave, ref_audio_paths.items.len);
    var aud_done: usize = 0;
    defer {
        for (aud_waves[0..aud_done]) |*w| w.deinit(gpa);
        gpa.free(aud_waves);
    }
    const auds = try gpa.alloc(pipeline.Session.RefAudio, ref_audio_paths.items.len);
    defer gpa.free(auds);
    for (ref_audio_paths.items, aud_waves, auds, 0..) |path, *w, *out, ai| {
        w.* = loadWav(gpa, io, path) catch |err| {
            try stdout.print("reference audio {d} ({s}) is not a WAV this build can read: {t}\n", .{ ai + 1, path, err });
            return err;
        };
        aud_done += 1;
        out.* = .{ .samples = w.samples, .channels = w.channels, .sample_rate = w.sample_rate };
        try stdout.print("reference audio {d}: {s} ({d:.2} s, {d} ch @ {d} Hz)\n", .{
            ai + 1, path, w.seconds(), w.channels, w.sample_rate,
        });
    }
    clip_opts.ref_audios = auds;

    // A continuation source: the same `frame_NNNNN.png` layout `--frames` writes,
    // plus the `audio.wav` beside it when there is one, so a rendered clip
    // continues itself.
    var cont_frames: std.ArrayList(pipeline.Session.RefImage) = .empty;
    defer {
        for (cont_frames.items) |f| gpa.free(@constCast(f.rgb));
        cont_frames.deinit(gpa);
    }
    var cont_wave: ?TensorPencil.audio.Wave = null;
    defer if (cont_wave) |*w| w.deinit(gpa);
    if (continue_dir) |dir_path| {
        if (preserve_frames == 0) {
            try stdout.print("--continue-from needs --preserve-frames N (how much to hold fixed)\n", .{});
            return error.BadArgs;
        }
        var fi: usize = 0;
        while (true) : (fi += 1) {
            var nb: [512]u8 = undefined;
            const fp = try std.fmt.bufPrint(&nb, "{s}/frame_{d:0>5}.png", .{ dir_path, fi });
            const img = loadPng(gpa, io, fp) catch break;
            try cont_frames.append(gpa, img);
        }
        if (cont_frames.items.len == 0) {
            try stdout.print("--continue-from {s} has no frame_NNNNN.png files\n", .{dir_path});
            return error.BadArgs;
        }
        var ab: [512]u8 = undefined;
        const ap = try std.fmt.bufPrint(&ab, "{s}/audio.wav", .{dir_path});
        cont_wave = loadWav(gpa, io, ap) catch null;
        try stdout.print("continue from: {s} ({d} frames, {d}x{d}{s}), preserving {d}\n", .{
            dir_path,           cont_frames.items.len, cont_frames.items[0].width,
            cont_frames.items[0].height, if (cont_wave != null) ", with audio" else ", no audio.wav",
            preserve_frames,
        });
        clip_opts.continue_from = .{
            .frames = cont_frames.items,
            .audio = if (cont_wave) |w| .{
                .samples = w.samples,
                .channels = w.channels,
                .sample_rate = w.sample_rate,
            } else null,
            .preserve_frames = preserve_frames,
        };
    }

    var sess = try pipeline.Session.init(io, gpa, opts, stdout);
    defer sess.deinit();

    var clip = try sess.generateClip(gpa, clip_opts, stdout);
    defer clip.deinit(gpa);

    try stdout.print("clip: {d} frames {d}x{d} @ {d} fps, {d} audio samples @ {d} Hz\n", .{
        clip.frames, clip.width, clip.height, clip.fps_num, clip.audioFrames(), clip.audio_sample_rate,
    });

    // `--out` names the clip, with or without the extension: MP4 mode appends
    // `.mp4` if it is not already there, frames mode strips it and uses the rest
    // as a directory. Blindly appending gave `fox.mp4.mp4`, and creating the
    // directory unconditionally left an empty one beside every MP4.
    const dir = std.Io.Dir.cwd();
    const out_stem = if (std.mem.endsWith(u8, out_dir, ".mp4")) out_dir[0 .. out_dir.len - 4] else out_dir;
    // Frames mode writes INTO `out_stem`; MP4 mode writes a file beside it, so it
    // needs the PARENT. Creating neither is how a finished render died at the mux
    // with `MuxOpenFailed`, having done all the work.
    if (frames_only) {
        try dir.createDirPath(io, out_stem);
    } else if (std.fs.path.dirname(out_stem)) |parent| {
        try dir.createDirPath(io, parent);
    }
    var buf: [512]u8 = undefined;

    // The AUTOMATIC1111 `parameters` block, in the container's own metadata. The
    // same builder the GUI writes into a PNG text chunk, because a reader
    // re-renders from it and two copies of the format would drift.
    const model_name = std.fs.path.stem(opts.dit_path);
    const params_base = try pipeline.buildA1111Params(
        gpa,
        clip_opts.prompt,
        "",
        clip_opts.steps,
        1.0, // no CFG branch: every H3 conditioning node emits one conditioning
        clip_opts.seed,
        clip.width,
        clip.height,
        model_name,
        sess.family(),
        .euler,
        null,
        opts.prompt_syntax,
        opts.emphasis,
        opts.compat,
        opts.compatConfig(),
    );
    const params = try pipeline.appendClipParams(gpa, params_base, sess.clipShifts(clip_opts));
    defer gpa.free(params);
    const params_z = try gpa.dupeZ(u8, params);
    defer gpa.free(params_z);

    if (!frames_only) {
        const mp4 = try std.fmt.bufPrintZ(&buf, "{s}.mp4", .{out_stem});
        var mux = av.Muxer.open(mp4, .{
            .width = clip.width,
            .height = clip.height,
            .fps_num = clip.fps_num,
            .fps_den = clip.fps_den,
            .audio_channels = clip.audio_channels,
            .audio_sample_rate = clip.audio_sample_rate,
            .meta_key = "parameters",
            .meta_value = params_z,
        }) catch |err| {
            try stdout.print("mux open failed ({t}): {s}\n", .{ err, av.lastError() });
            return err;
        };
        errdefer mux.abort();
        for (0..clip.frames) |f| try mux.writeFrame(clip.frame(f));
        if (clip.hasAudio()) try mux.writeAudio(clip.audio, clip.audioFrames());
        try mux.finish();
        try stdout.print("wrote {s}\n", .{mp4});
        try stdout.flush();
        return;
    }

    for (0..clip.frames) |f| {
        var png: std.ArrayList(u8) = .empty;
        defer png.deinit(gpa);
        try TensorPencil.image.encodePngRgbText(gpa, &png, clip.frame(f), clip.width, clip.height, &.{
            .{ .keyword = "parameters", .text = params },
        });
        const path = try std.fmt.bufPrint(&buf, "{s}/frame_{d:0>5}.png", .{ out_stem, f });
        try dir.writeFile(io, .{ .sub_path = path, .data = png.items });
    }
    if (clip.hasAudio()) {
        const wav = try TensorPencil.audio.encodeWavPcm16(gpa, clip.audio, clip.audio_channels, clip.audio_sample_rate);
        defer gpa.free(wav);
        const path = try std.fmt.bufPrint(&buf, "{s}/audio.wav", .{out_stem});
        try dir.writeFile(io, .{ .sub_path = path, .data = wav });
    }
    try stdout.print("wrote {s}/\n", .{out_stem});
    try stdout.flush();
}

/// `minimax-h3-audio-cuda-test`: the BigVGAN audio decode on the device against
/// its CPU reference, on real weights. Non-zero exit if it disagrees.
///
/// Reports a per-sample maximum alongside the relative L2, because this is a
/// VOCODER: a norm can look fine while a handful of samples clip or click, and
/// the device path runs its GEMMs through f16 where the reference is f32.
/// `minimax-h3-vae-encode-cuda-test`: the video VAE's 3-D causal ENCODER on the
/// device against its CPU reference, on real weights. Non-zero exit if it
/// disagrees.
///
/// A CLIP by default, not a single frame: GroupNorm statistics are per FRAME, and a
/// one-frame encode is blind to that axis, so a single-frame check would pass with
/// the statistics taken over the whole volume.
fn minimaxH3VaeEncodeCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const ve = TensorPencil.models.minimax_h3_vae_encode;
    const ve_cuda = TensorPencil.models.minimax_h3_vae_encode_cuda;
    const h3vae = TensorPencil.models.minimax_h3_vae;
    const h3 = TensorPencil.models.minimax_h3;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== minimax-h3-vae-encode-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("checkpoint not found: {s}\n", .{ckpt});
        return;
    };
    var st = try safetensors.open(arena, io, ckpt);
    defer st.deinit();
    var enc = ve.VideoEncoder.load(arena, .{ .safetensors = &st }, null) catch |err| {
        try stdout.print("this VAE carries no usable encoder: {t}\n", .{err});
        return;
    };
    defer enc.deinit();
    try stdout.print("encoder: {d} levels, {d} norm groups, {d} -> {d} ch\n", .{
        enc.levels.len, enc.cfg.norm_groups, enc.conv_in.in_ch, enc.cfg.embed_dim,
    });
    if (!ve_cuda.supported(&enc)) {
        try stdout.print("FAIL: this encoder's shapes have no device path here\n", .{});
        return error.UnsupportedCheckpoint;
    }

    // Small by default. The CPU reference is the slow side and every convention
    // (causal temporal padding, reflect spatial padding, the asymmetric downsample
    // pad, per-frame statistics) runs at any extent.
    const px: usize = if (std.c.getenv("TP_H3_VENC_PX")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 64) else 64;
    const frames: usize = if (std.c.getenv("TP_H3_VENC_T")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 5) else 5;
    try stdout.print("{d} frames at {d}x{d}\n", .{ frames, px, px });

    var prng = std.Random.DefaultPrng.init(0x7e3c0);
    const rnd = prng.random();
    const pix = try arena.alloc(f32, 3 * frames * px * px);
    for (pix) |*v| v.* = rnd.floatNorm(f32) * 0.5;

    const sp: h3vae.Spatial = .{ .ratio = h3.spatial_downscale };
    const vol: ve.Vol = .{ .d = pix, .ch = 3, .t = frames, .h = px, .w = px };

    var sess = try ve_cuda.Session.init(arena, &enc);
    defer sess.deinit();

    {
        const sh = ve_cuda.Shapes.of(&enc, @min(frames, enc.cfg.clip_length), px, px);
        try stdout.print("shapes: vol {d}, patch {d}, gemm {d} -> {d} MB\n", .{
            sh.vol, sh.patch, sh.gemm, sh.deviceBytes() >> 20,
        });
        try stdout.flush();
    }

    const t0 = std.Io.Clock.real.now(io).nanoseconds;
    const want = try ve.encode(&enc, io, arena, vol, sp, null);
    const t1 = std.Io.Clock.real.now(io).nanoseconds;
    try stdout.print("host encode done\n", .{});
    try stdout.flush();
    var ctx: ve_cuda.Ctx = .{ .enc = &enc, .sess = &sess, .be = be, .gpa = arena };
    defer ctx.deinit();
    const got = try ve.encode(&enc, io, arena, vol, sp, ctx.moments());
    const t2 = std.Io.Clock.real.now(io).nanoseconds;

    if (want.ch != got.ch or want.t != got.t or want.h != got.h or want.w != got.w) {
        try stdout.print("FAIL: shapes differ, host [{d}][{d}][{d}][{d}] device [{d}][{d}][{d}][{d}]\n", .{
            want.ch, want.t, want.h, want.w, got.ch, got.t, got.h, got.w,
        });
        return error.GpuMismatch;
    }
    try stdout.print("latent [{d}][{d}][{d}][{d}]\n", .{ got.ch, got.t, got.h, got.w });
    try stdout.print("device workspace peak {d} MB (host peak would be {d} MB)\n", .{
        ctx.peak.deviceBytes() >> 20, ve.peakBytesFor(enc.cfg, @min(frames, enc.cfg.clip_length), px, px) >> 20,
    });

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f64 = 0;
    for (want.d[0..want.elems()], got.d[0..got.elems()]) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
        max_abs = @max(max_abs, @abs(@as(f64, e - a)));
    }
    const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
    for (got.d[0..got.elems()]) |a| if (!std.math.isFinite(a)) {
        try stdout.print("FAIL: the device latent is not finite\n", .{});
        return error.GpuMismatch;
    };

    const cpu_ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6;
    const dev_ms = @as(f64, @floatFromInt(t2 - t1)) / 1e6;
    try stdout.print("cpu {d:.0} ms, device {d:.0} ms ({d:.1}x)\n", .{ cpu_ms, dev_ms, cpu_ms / @max(dev_ms, 1e-9) });

    // The frames must genuinely differ, or a device path that took its statistics
    // over the whole volume (or encoded one frame and copied it) would pass.
    var t_diff: f64 = 0;
    var t_ref: f64 = 0;
    if (got.t > 1) {
        const plane = got.h * got.w;
        for (0..got.ch) |c| {
            for (0..plane) |i| {
                const a0 = got.d[(c * got.t + 0) * plane + i];
                const a1 = got.d[(c * got.t + 1) * plane + i];
                t_diff += @as(f64, a0 - a1) * (a0 - a1);
                t_ref += @as(f64, a0) * a0;
            }
        }
    }
    const t_rel = if (t_ref > 0) @sqrt(t_diff / t_ref) else 1.0;

    const tol = 2e-3;
    const ok = rel < tol and t_rel > 0.05;
    try stdout.print("latent rel L2 {d:.6}  max |dz| {d:.6}  (tol {d:.4})  frame split {d:.3}  {s}\n", .{
        rel, max_abs, tol, t_rel, if (ok) "ok" else "FAIL",
    });
    if (!ok) return error.GpuMismatch;
}

/// `minimax-h3-audio-encode-cuda-test`: the DAC audio ENCODER on the device
/// against its CPU reference, on real weights. Non-zero exit if it disagrees.
///
/// Reports the latent's relative L2 and a per-element maximum. The tolerance is
/// tighter than the decode side's because a latent is not a waveform: an error
/// here rides through every sampling step as conditioning.
fn minimaxH3AudioEncodeCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const ae = TensorPencil.models.minimax_h3_audio_encode;
    const ae_cuda = TensorPencil.models.minimax_h3_audio_encode_cuda;
    const av = TensorPencil.models.minimax_h3_audio;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== minimax-h3-audio-encode-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("checkpoint not found: {s}\n", .{ckpt});
        return;
    };
    var st = try safetensors.open(arena, io, ckpt);
    defer st.deinit();
    var enc = try ae.AudioEncoder.load(arena, .{ .safetensors = &st }, .{});
    defer enc.deinit();
    try stdout.print("encoder: {d} stages, hop {d}, latent {d} ch, head {d} wide / {d} heads\n", .{
        enc.stages.len, enc.hop(), enc.latentChannels(), enc.head.in_dim, enc.head.heads,
    });
    if (!ae_cuda.supported(&enc)) {
        try stdout.print("FAIL: this encoder's shapes have no device path here\n", .{});
        return error.UnsupportedCheckpoint;
    }

    // A quarter second by default: the CPU reference is the slow side, and every
    // code path (all five stages, both padding relations, the causal attention)
    // runs at any length. The attention is O(t^2), so a long clip is the device's
    // advantage rather than a different test.
    const ms: usize = if (std.c.getenv("TP_H3_AUDIO_MS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 250) else 250;
    const len = av.sample_rate * ms / 1000;
    const t = enc.latentFrames(len);
    try stdout.print("{d} ms -> {d} samples per channel -> {d} latent frames\n", .{ ms, len, t });

    var prng = std.Random.DefaultPrng.init(0xae1c0);
    const rnd = prng.random();
    const wav = try arena.alloc(f32, len * av.stereo);
    // A tone per channel plus noise: the two stereo halves must differ, and a tone
    // is what makes a pitch error visible if anyone plots the disagreement.
    for (0..len) |i| {
        const tt = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(av.sample_rate));
        wav[i * 2 + 0] = 0.5 * @sin(2.0 * std.math.pi * 220.0 * tt) + 0.05 * rnd.floatNorm(f32);
        wav[i * 2 + 1] = 0.5 * @sin(2.0 * std.math.pi * 660.0 * tt) + 0.05 * rnd.floatNorm(f32);
    }

    // ISOLATION: `TP_H3_AUDIO_BAND=<elems>` sets the im2col band. Setting it huge
    // removes banding entirely, which is the one thing that turns on with LENGTH --
    // so if a long clip disagrees and an unbanded one does not, the band arithmetic
    // is the cause and not the GEMM or the kernels.
    if (std.c.getenv("TP_H3_AUDIO_BAND")) |v| {
        ae_cuda.patch_band = std.fmt.parseInt(usize, std.mem.span(v), 10) catch ae_cuda.patch_band;
        try stdout.print("(diagnostic: im2col band {d} elements)\n", .{ae_cuda.patch_band});
    }

    var sess = try ae_cuda.Session.init(arena, &enc);
    defer sess.deinit();
    var ws = try ae_cuda.Workspace.init(be, &enc, len);
    defer ws.deinit(be);
    try stdout.print("workspace: {d} MB of device buffers (host peak would be {d} MB)\n", .{
        ws.shapes.deviceBytes() >> 20, enc.peakBytesFor(len) >> 20,
    });

    const n = enc.latentChannels() * av.stereo * t;
    const want = try arena.alloc(f32, n);
    const got = try arena.alloc(f32, n);

    const t0 = std.Io.Clock.real.now(io).nanoseconds;
    try ae.encode(&enc, io, arena, want, wav, len, av.stereo);
    const t1 = std.Io.Clock.real.now(io).nanoseconds;
    try ae_cuda.encode(&enc, &sess, be, &ws, arena, got, wav, len, av.stereo);
    const t2 = std.Io.Clock.real.now(io).nanoseconds;

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
        max_abs = @max(max_abs, @abs(@as(f64, e - a)));
    }
    const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
    for (got) |a| if (!std.math.isFinite(a)) {
        try stdout.print("FAIL: the device latent is not finite\n", .{});
        return error.GpuMismatch;
    };

    const cpu_ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6;
    const dev_ms = @as(f64, @floatFromInt(t2 - t1)) / 1e6;
    try stdout.print("cpu {d:.0} ms, device {d:.0} ms ({d:.1}x)\n", .{ cpu_ms, dev_ms, cpu_ms / @max(dev_ms, 1e-9) });

    // The two stereo halves must genuinely differ, or a device path that encoded
    // one and copied it would pass every norm above.
    var ch_diff: f64 = 0;
    var ch_ref: f64 = 0;
    for (0..enc.latentChannels()) |c| {
        for (0..t) |i| {
            const l = got[(c * av.stereo + 0) * t + i];
            const r = got[(c * av.stereo + 1) * t + i];
            ch_diff += @as(f64, l - r) * (l - r);
            ch_ref += @as(f64, l) * l;
        }
    }
    const ch_rel = if (ch_ref > 0) @sqrt(ch_diff / ch_ref) else 0;

    const tol = 2e-3;
    const ok = rel < tol and ch_rel > 0.1;
    try stdout.print("latent rel L2 {d:.6}  max |dz| {d:.6}  (tol {d:.4})  stereo split {d:.3}  {s}\n", .{
        rel, max_abs, tol, ch_rel, if (ok) "ok" else "FAIL",
    });
    if (!ok) return error.GpuMismatch;
}

fn minimaxH3AudioCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const av = TensorPencil.models.minimax_h3_audio;
    const av_cuda = TensorPencil.models.minimax_h3_audio_cuda;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== minimax-h3-audio-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("checkpoint not found: {s}\n", .{ckpt});
        return;
    };
    var st = try safetensors.open(arena, io, ckpt);
    defer st.deinit();
    var dec = try av.AudioDecoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();
    try stdout.print("vocoder: {d} stages, {d} kernels, x{d}, latent {d} -> {d} ch\n", .{
        dec.nStages(), dec.n_kernels, dec.upsampleFactor(), dec.dec_in.in_ch, dec.dec_in.out_ch,
    });
    if (!av_cuda.supported(&dec)) {
        try stdout.print("FAIL: this decoder's shapes have no device path here\n", .{});
        return error.UnsupportedCheckpoint;
    }

    // Latent frames. Small by default: the CPU reference is the slow side, and
    // every code path (all seven stages, both kernel/rate pairings, the temporal
    // extent of the kaiser filters) runs at any length.
    const t: usize = if (std.c.getenv("TP_H3_AUDIO_T")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4) else 4;
    const samples = t * dec.upsampleFactor();
    try stdout.print("t = {d} latent frames -> {d} samples per channel\n", .{ t, samples });

    if (std.c.getenv("TP_H3_AUDIO_F16") != null) {
        av_cuda.f16_gemm = true;
        try stdout.print("(diagnostic: conv GEMMs on tensor cores through f16)\n", .{});
    }

    var prng = std.Random.DefaultPrng.init(0xa0d10);
    const rnd = prng.random();
    const z = try arena.alloc(f32, av.latent_channels * av.stereo * t);
    for (z) |*v| v.* = rnd.floatNorm(f32);

    var sess = try av_cuda.Session.init(arena, &dec);
    defer sess.deinit();
    try stdout.print("session: {d} MB of permuted weights\n", .{sess.bytes() >> 20});
    var ws = try av_cuda.Workspace.init(be, &dec, t);
    defer ws.deinit(be);
    try stdout.print("workspace: {d} MB\n", .{av_cuda.Workspace.bytesFor(&dec, t) >> 20});

    const want = try arena.alloc(f32, samples * av.stereo);
    const got = try arena.alloc(f32, samples * av.stereo);

    const t0 = std.Io.Clock.real.now(io).nanoseconds;
    try av.decode(&dec, io, arena, want, z, t);
    const t1 = std.Io.Clock.real.now(io).nanoseconds;
    try av_cuda.decode(&dec, &sess, be, &ws, arena, got, z, t, null);
    const t2 = std.Io.Clock.real.now(io).nanoseconds;

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f64 = 0;
    var max_at: usize = 0;
    for (want, got, 0..) |e, a, i| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
        const d = @abs(@as(f64, e - a));
        if (d > max_abs) {
            max_abs = d;
            max_at = i;
        }
    }
    const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
    var peak: f64 = 0;
    for (want) |e| peak = @max(peak, @abs(@as(f64, e)));

    const cpu_ns: u64 = @intCast(t1 - t0);
    const dev_ns: u64 = @intCast(t2 - t1);
    try stdout.print("cpu {d} ms, device {d} ms ({d:.1}x)\n", .{
        cpu_ns / std.time.ns_per_ms,
        dev_ns / std.time.ns_per_ms,
        @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(@max(dev_ns, 1))),
    });
    try stdout.print("rel L2 {e}  max |dv| {e} at sample {d}  (reference peak {e})\n", .{ rel, max_abs, max_at, peak });
    // The two stereo channels must genuinely differ, or a device path that
    // decoded one and copied it would pass everything above.
    var ch_l2_ref: f64 = 0;
    var ch_l2_err: f64 = 0;
    for (0..samples) |i| {
        const l = @as(f64, got[i * av.stereo]);
        const r = @as(f64, got[i * av.stereo + 1]);
        ch_l2_ref += l * l;
        ch_l2_err += (l - r) * (l - r);
    }
    const ch_rel = if (ch_l2_ref > 0) @sqrt(ch_l2_err / ch_l2_ref) else 0;
    try stdout.print("stereo channels differ by rel {d:.4}\n", .{ch_rel});

    // Concentrated or spread? A single spike is a boundary bug; a broad tail is
    // precision. A max alone cannot tell them apart.
    for ([_]f64{ 1e-4, 3e-4, 1e-3, 3e-3 }) |thr| {
        var n: usize = 0;
        for (want, got) |e, a| if (@abs(@as(f64, e - a)) > thr) {
            n += 1;
        };
        try stdout.print("  |dv| > {e}: {d} of {d} ({d:.2}%)\n", .{
            thr, n, want.len, 100.0 * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(want.len)),
        });
    }

    // A vocoder's error budget is per sample, not per norm: 16-bit PCM's own
    // quantum is 3e-5, so anything under ~1e-3 absolute is inaudible.
    if (rel > 5e-3 or max_abs > 2e-3 or ch_rel < 0.05) {
        try stdout.print("FAIL\n", .{});
        // Flush before unwinding: returning an error skips main's own flush, and
        // a failure whose numbers never reach the terminal is not a diagnostic.
        stdout.flush() catch {};
        return error.Unsupported;
    }
    try stdout.print("OK\n", .{});
}

/// `lora-cuda-test`: the device LoRA sidecar against its host twin, at the real
/// H3 factor shapes. Non-zero exit if they disagree.
///
/// Isolates exactly the sidecar: the factors are synthetic bf16 and there is no
/// checkpoint, no trunk and no int8, so a disagreement here is `lora_cuda` and
/// nothing else. `lora.zig`'s unit tests already pin the host path against
/// ComfyUI's merged LoRA, so host-agrees-with-device closes the loop.
fn loraCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const lora = TensorPencil.models.lora;
    const lora_cuda = TensorPencil.models.lora_cuda;
    const Weight = TensorPencil.ops.matmul.Weight;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== lora-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    // The real turbo LoRA's shapes, plus the fused qkv split into its three
    // block-diagonal factors. `m` is deliberately not 128-aligned: `opGemmBf16`
    // pads m internally, and a case that only ever ran aligned would not show a
    // pad row leaking into the accumulate.
    const hidden: usize = 5376;
    const inner: usize = 7168;
    const ffn: usize = 14336;
    const cases = [_]struct { name: []const u8, in_dim: usize, out: usize, rank: usize, groups: usize }{
        .{ .name = "attn.qkv_proj", .in_dim = hidden, .out = 3 * inner, .rank = 384, .groups = 3 },
        .{ .name = "attn.out_proj", .in_dim = inner, .out = hidden, .rank = 128, .groups = 1 },
        .{ .name = "mlp.fc1", .in_dim = hidden, .out = 2 * ffn, .rank = 128, .groups = 1 },
        .{ .name = "mlp.fc2", .in_dim = ffn, .out = hidden, .rank = 128, .groups = 1 },
    };
    const m: usize = 150;
    const scale: f32 = 0.0625;

    var prng = std.Random.DefaultPrng.init(0x10ca);
    const rnd = prng.random();

    var worst: f64 = 0;
    for (cases) |c| {
        // Per-group factors, contiguous, exactly as `lora.loadTarget` leaves
        // them after a block-diagonal split.
        const gr = c.rank / c.groups;
        const go = c.out / c.groups;
        const factors = try arena.alloc(lora.Factor, c.groups);
        for (factors, 0..) |*f, g| {
            const a_b = try arena.alloc(u8, gr * c.in_dim * 2);
            const b_b = try arena.alloc(u8, go * gr * 2);
            for (0..a_b.len / 2) |k| std.mem.writeInt(u16, a_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            for (0..b_b.len / 2) |k| std.mem.writeInt(u16, b_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            f.* = .{
                .a = Weight.init(a_b, .bf16, gr, c.in_dim),
                .b = Weight.init(b_b, .bf16, go, gr),
                .out_off = g * go,
                .scale = scale,
            };
        }
        const t: lora.Target = .{ .factors = factors, .in_dim = c.in_dim, .out_dim = c.out, .tag = c.name };
        if (!lora_cuda.supported(&t)) {
            try stdout.print("FAIL: {s} shapes have no device path\n", .{c.name});
            return error.Unsupported;
        }

        const x = try arena.alloc(f32, m * c.in_dim);
        for (x) |*v| v.* = rnd.floatNorm(f32);

        // The ranges the trunk actually asks for: qkv splits three ways, fc1
        // two, the rest whole. Same list as `minimax_h3_cuda.forward`.
        const ranges: []const struct { off: usize, n: usize } = switch (c.groups) {
            3 => &.{ .{ .off = 0, .n = inner }, .{ .off = inner, .n = inner }, .{ .off = 2 * inner, .n = inner } },
            else => if (c.out == 2 * ffn)
                &.{ .{ .off = 0, .n = ffn }, .{ .off = ffn, .n = ffn } }
            else
                &.{.{ .off = 0, .n = c.out }},
        };

        // Host: the full delta, then read the same ranges out of it.
        const want = try arena.alloc(f32, m * c.out);
        @memset(want, 0);
        try t.applyHost(io, arena, want, c.out, x, m);

        // The control row. The device GEMM rounds the activation to bf16 before
        // multiplying, so a residual of about bf16's own 2^-9 is the FLOOR, not a
        // defect. Running the host path on a pre-rounded activation says how much
        // of the residual that accounts for; without this the number below is
        // uninterpretable.
        const x_bf = try arena.alloc(f32, m * c.in_dim);
        for (x_bf, x) |*d, v| d.* = @bitCast(@as(u32, f32ToBf16(v)) << 16);
        const floor_ref = try arena.alloc(f32, m * c.out);
        @memset(floor_ref, 0);
        try t.applyHost(io, arena, floor_ref, c.out, x_bf, m);

        var ws = try lora_cuda.Workspace.init(be, m * c.rank, m * c.out);
        defer ws.deinit(be);
        const mpad = std.mem.alignForward(usize, m, 128);
        var xd = try be.tensorCreate(mpad * c.in_dim * 4);
        defer be.tensorDestroy(&xd);
        const xpad = try arena.alloc(f32, mpad * c.in_dim);
        @memset(xpad, 0);
        @memcpy(xpad[0 .. m * c.in_dim], x);
        try be.tensorUpload(xd, std.mem.sliceAsBytes(xpad));

        for (ranges) |r| {
            var yd = try be.tensorCreate(mpad * r.n * 4);
            defer be.tensorDestroy(&yd);
            // Start from zero, so what comes back IS the sidecar's contribution.
            const zeros = try arena.alloc(f32, mpad * r.n);
            @memset(zeros, 0);
            try be.tensorUpload(yd, std.mem.sliceAsBytes(zeros));

            try be.beginBatch();
            errdefer if (be.batching()) be.abortBatch();
            try lora_cuda.applyRange(be, &ws, yd, xd, m, &t, r.off, r.n);
            try be.endBatch();

            const got = try arena.alloc(f32, m * r.n);
            try be.tensorDownload(yd, std.mem.sliceAsBytes(got));

            var l2_ref: f64 = 0;
            var l2_err: f64 = 0;
            var l2_floor: f64 = 0;
            for (0..m) |i| {
                for (0..r.n) |j| {
                    const e = want[i * c.out + r.off + j];
                    const a = got[i * r.n + j];
                    const fl = floor_ref[i * c.out + r.off + j];
                    l2_ref += @as(f64, e) * e;
                    l2_err += @as(f64, e - a) * (e - a);
                    l2_floor += @as(f64, e - fl) * (e - fl);
                }
            }
            const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
            const floor = if (l2_ref > 0) @sqrt(l2_floor / l2_ref) else @sqrt(l2_floor);
            worst = @max(worst, rel);
            try stdout.print("  {s:<14} rows [{d:>5},{d:>5})  rel {e}  (bf16 activation floor {e})\n", .{ c.name, r.off, r.off + r.n, rel, floor });
            // The delta must be nonzero, or the comparison proves nothing.
            if (l2_ref == 0) {
                try stdout.print("FAIL: {s} produced a zero delta\n", .{c.name});
                return error.Unsupported;
            }
        }
    }

    // bf16 factors against an f32 host reference: the tolerance is the bf16
    // GEMM's own, ~1e-3 relative at these widths, not an exactness claim.
    try stdout.print("worst rel {e}\n", .{worst});
    if (worst > 5e-3) {
        try stdout.print("FAIL: the device sidecar disagrees with the host one\n", .{});
        return error.Unsupported;
    }
    try stdout.print("OK\n", .{});
}

/// Round to bf16 (ties to even), the same rounding the CUDA converts use.
fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const lsb = (bits >> 16) & 1;
    const rounded = bits + 0x7fff + lsb;
    return @truncate(rounded >> 16);
}

/// `minimax-h3-cuda-test`: the H3 trunk on the device against its CPU reference,
/// on real weights. Non-zero exit if it disagrees.
///
/// A CLI command rather than a unit test because the test binary brings up no
/// CUDA context (see CLAUDE.md). It runs at the DEVELOPMENT shape: 147 packed
/// rows, which exercises every code path the full render does and costs seconds
/// rather than the ~20 minutes a full-resolution CPU reference would.
fn minimaxH3CudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const h3 = TensorPencil.models.minimax_h3;
    const h3_cuda = TensorPencil.models.minimax_h3_cuda;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== minimax-h3-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("checkpoint not found: {s}\n", .{ckpt});
        return;
    };
    var st = try safetensors.open(arena, io, ckpt);
    defer st.deinit();
    var dit = try h3.DiT.load(arena, .{ .safetensors = &st });
    defer dit.deinit();
    try stdout.print("dit: {d} blocks, hidden {d}, {d}x{d} heads\n", .{
        dit.cfg.n_layers, dit.cfg.hidden, dit.cfg.n_heads, dit.cfg.head_dim,
    });
    if (std.c.getenv("TP_H3_NAIVE") != null) {
        h3_cuda.force_naive_attn = true;
        try stdout.print("(naive attention)\n", .{});
    }
    {
        const off = if (std.c.getenv("TP_H3_BLOCK_OFF")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0) else 0;
        const n = if (std.c.getenv("TP_H3_BLOCKS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch dit.blocks.len) else dit.blocks.len;
        if (off != 0 or n != dit.blocks.len) {
            const lo = @min(off, dit.blocks.len);
            dit.blocks = dit.blocks[lo..@min(lo + n, dit.blocks.len)];
            try stdout.print("(blocks [{d},{d}))\n", .{ lo, lo + dit.blocks.len });
        }
    }
    if (!h3_cuda.supported(&dit)) {
        try stdout.print("FAIL: this checkpoint's weights have no device path here\n", .{});
        return error.UnsupportedCheckpoint;
    }

    // The development shape: 256x256, 5 frames.
    const text_len: usize = if (std.c.getenv("TP_H3_TEXT")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 6) else 6;
    const lat_t: usize = 2;
    const lat_h: usize = if (std.c.getenv("TP_H3_LH")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 16) else 16;
    const lat_w: usize = if (std.c.getenv("TP_H3_LW")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 16) else 16;
    const audio_t: usize = 8;
    const sigma: f32 = if (std.c.getenv("TP_H3_SIGMA")) |v| (std.fmt.parseFloat(f32, std.mem.span(v)) catch 0.7) else 0.7;

    var layout = try h3.PackedLayout.build(arena, .{
        .text_len = text_len,
        .latent_t = lat_t,
        .latent_h = lat_h,
        .latent_w = lat_w,
        .audio_t = audio_t,
    }, &.{}, &.{}, &.{});
    defer layout.deinit();
    try stdout.print("packed sequence: {d} rows\n", .{layout.seq_len});

    var prng = std.Random.DefaultPrng.init(0x431a);
    const rnd = prng.random();
    const video = try arena.alloc(f32, h3.latent_channels * lat_t * lat_h * lat_w);
    for (video) |*v| v.* = rnd.floatNorm(f32);
    const audio = try arena.alloc(f32, h3.audio_latent_channels * h3.audio_channels * audio_t);
    for (audio) |*v| v.* = rnd.floatNorm(f32);
    const text = try arena.alloc(f32, text_len * dit.cfg.hidden);
    for (text) |*v| v.* = rnd.floatNorm(f32) * 0.5;

    // A DENOISE MASK when asked for, because it is the one thing that makes the two
    // target segments modulate PER ROW rather than once per segment, and the device
    // path reaches that through a different kernel argument (an index buffer) than
    // the host does. `TP_H3_MASK=binary` is the video-continuation shape;
    // `graded` adds timestep labels of its own, which is what stresses the index
    // stride; `spatial` alternates row by row, the case a run-length split could
    // not have served.
    const frame_rows = (lat_h / h3.patch_h) * (lat_w / h3.patch_w);
    var video_mask: []f32 = &.{};
    var audio_mask: []f32 = &.{};
    if (std.c.getenv("TP_H3_MASK")) |v| {
        const mode = std.mem.span(v);
        video_mask = try arena.alloc(f32, lat_t * frame_rows);
        audio_mask = try arena.alloc(f32, h3.audio_channels * audio_t);
        for (video_mask, 0..) |*m, i| m.* = if (std.mem.eql(u8, mode, "binary"))
            (if (i < frame_rows) 0.0 else 1.0)
        else if (std.mem.eql(u8, mode, "spatial"))
            (if (i % 2 == 0) 0.0 else 1.0)
        else
            @as(f32, @floatFromInt(i % 5)) / 4.0;
        // The audio stream masked too, so its own sigma and pin are exercised.
        for (audio_mask, 0..) |*m, i| m.* = if (i < audio_t) 0.0 else 1.0;
        try stdout.print("denoise mask: {s} ({d} video rows, {d} audio rows)\n", .{
            mode, video_mask.len, audio_mask.len,
        });
    }

    // `TP_H3_MASK=uniform` is the ISOLATION for the device's per-row index
    // arithmetic: a mask whose rows all agree normally COLLAPSES to a scalar
    // modulation offset, and forcing the per-row table instead must give the same
    // answer to the last bit. Device against device, so no int8 or CPU error is in
    // the way -- unlike the masked device-vs-CPU figures, where a pinned row's
    // larger modulation genuinely amplifies the existing divergence.
    if (video_mask.len > 0 and std.mem.eql(u8, std.mem.span(std.c.getenv("TP_H3_MASK").?), "uniform")) {
        for (video_mask) |*m| m.* = 0.5;
        for (audio_mask) |*m| m.* = 0.5;
        const in_u: h3.Inputs = .{
            .video = video,
            .audio = audio,
            .text = text,
            .sigma = sigma,
            .video_mask = video_mask,
            .audio_mask = audio_mask,
        };
        var sess_u = try h3_cuda.Session.init(be, arena, &dit, &layout);
        defer sess_u.deinit(be);
        var dws_u = try h3_cuda.Workspace.init(be, &dit, layout.seq_len);
        defer dws_u.deinit(be);

        const n_v = h3.latent_channels * lat_t * lat_h * lat_w;
        const n_a = h3.audio_latent_channels * h3.audio_channels * audio_t;
        const scalar_v = try arena.alloc(f32, n_v);
        const scalar_a = try arena.alloc(f32, n_a);
        const rows_v = try arena.alloc(f32, n_v);
        const rows_a = try arena.alloc(f32, n_a);

        h3.force_row_labels = false;
        try h3_cuda.forward(&dit, be, &sess_u, &dws_u, io, arena, &layout, scalar_v, scalar_a, in_u, null);
        h3.force_row_labels = true;
        try h3_cuda.forward(&dit, be, &sess_u, &dws_u, io, arena, &layout, rows_v, rows_a, in_u, null);
        h3.force_row_labels = false;

        const relOf = struct {
            fn go(want: []const f32, got: []const f32) f64 {
                var l2_ref: f64 = 0;
                var l2_err: f64 = 0;
                for (want, got) |e, g| {
                    l2_ref += @as(f64, e) * e;
                    l2_err += @as(f64, e - g) * (e - g);
                }
                return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
            }
        }.go;
        const dev = @max(relOf(scalar_v, rows_v), relOf(scalar_a, rows_a));

        // The same comparison on the HOST forward, which bisects the two: the
        // per-row modulation is shared logic in `Timesteps` and separate code in
        // the kernels, and only running both says which one moved.
        const cs_v = try arena.alloc(f32, n_v);
        const cs_a = try arena.alloc(f32, n_a);
        const cr_v = try arena.alloc(f32, n_v);
        const cr_a = try arena.alloc(f32, n_a);
        var ws_u = try h3.Workspace.init(arena, dit.cfg, &layout);
        defer ws_u.deinit(arena);
        h3.force_row_labels = false;
        try h3.forward(&dit, io, arena, &ws_u, &layout, cs_v, cs_a, in_u);
        h3.force_row_labels = true;
        try h3.forward(&dit, io, arena, &ws_u, &layout, cr_v, cr_a, in_u);
        h3.force_row_labels = false;
        const host = @max(relOf(cs_v, cr_v), relOf(cs_a, cr_a));

        try stdout.print("per-row vs scalar modulation, same labels:\n", .{});
        try stdout.print("  host   rel L2 {e}  {s}\n", .{ host, if (host < 1e-6) "ok" else "FAIL" });
        try stdout.print("  device rel L2 {e}  {s}\n", .{ dev, if (dev < 1e-6) "ok" else "FAIL" });
        if (!(host < 1e-6) or !(dev < 1e-6)) return error.GpuMismatch;
        return;
    }

    // `TP_H3_GLOBAL_PREP=1` is the ISOLATION for the int8 prep's global row staging.
    //
    // Every reduction in this DiT fits in shared memory, so forcing the global path
    // runs the same arithmetic in the same order on the same data and must agree to
    // the LAST BIT. A device-vs-CPU tolerance cannot see a defect here: int8 over 50
    // blocks has a ~1e-2 floor, and the address rewrite this validates is exactly
    // the kind of thing that hides under it.
    if (std.c.getenv("TP_H3_GLOBAL_PREP") != null) {
        const n_v = h3.latent_channels * lat_t * lat_h * lat_w;
        const n_a = h3.audio_latent_channels * h3.audio_channels * audio_t;
        const sh_v = try arena.alloc(f32, n_v);
        const sh_a = try arena.alloc(f32, n_a);
        const gl_v = try arena.alloc(f32, n_v);
        const gl_a = try arena.alloc(f32, n_a);

        var sess_p = try h3_cuda.Session.init(be, arena, &dit, &layout);
        defer sess_p.deinit(be);
        var dws_p = try h3_cuda.Workspace.init(be, &dit, layout.seq_len);
        defer dws_p.deinit(be);

        // A local `Inputs`, since this runs before the shared one is built.
        const in_p: h3.Inputs = .{ .video = video, .audio = audio, .text = text, .sigma = sigma };

        try stdout.print("shared-staging ceiling on this device: {d} columns\n", .{be.i8PrepMaxSharedCols()});
        cuda.Backend.force_global_prep = false;
        try h3_cuda.forward(&dit, be, &sess_p, &dws_p, io, arena, &layout, sh_v, sh_a, in_p, null);
        cuda.Backend.force_global_prep = true;
        try h3_cuda.forward(&dit, be, &sess_p, &dws_p, io, arena, &layout, gl_v, gl_a, in_p, null);
        cuda.Backend.force_global_prep = false;

        var worst: f64 = 0;
        for ([_][2][]const f32{ .{ sh_v, gl_v }, .{ sh_a, gl_a } }) |pair| {
            var l2_ref: f64 = 0;
            var l2_err: f64 = 0;
            for (pair[0], pair[1]) |e, g| {
                l2_ref += @as(f64, e) * e;
                l2_err += @as(f64, e - g) * (e - g);
            }
            worst = @max(worst, if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err));
        }
        try stdout.print("int8 prep, global vs shared row staging: rel L2 {e}  {s}\n", .{
            worst, if (worst == 0) "ok (bit-identical)" else "FAIL",
        });
        if (worst != 0) return error.GpuMismatch;
        return;
    }


    const in: h3.Inputs = .{
        .video = video,
        .audio = audio,
        .text = text,
        .sigma = sigma,
        .video_mask = video_mask,
        .audio_mask = audio_mask,
    };

    // --- isolation: ONE int8 GEMM, device against host ----------------------
    // The whole-forward figure cannot say whether a 10% error is the GEMM, the
    // norm, the rope or the attention. This removes exactly one component.
    for ([_]usize{ 128, 150, 256 }) |m| {
        const h = dit.cfg.hidden;
        const x = try arena.alloc(f32, m * h);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const w = dit.blocks[0].attn.qkv.w;

        const want = try arena.alloc(f32, m * w.rows);
        try TensorPencil.ops.matmul.matmul(io, arena, want, x, m, w, null);

        const mpad = std.mem.alignForward(usize, m, 128);
        var xd = try be.tensorCreate(mpad * h * 4);
        defer be.tensorDestroy(&xd);
        var yd = try be.tensorCreate(mpad * w.rows * 4);
        defer be.tensorDestroy(&yd);
        const xpad = try arena.alloc(f32, mpad * h);
        @memset(xpad, 0);
        @memcpy(xpad[0 .. m * h], x);
        try be.beginBatch();
        try be.tensorUpload(xd, std.mem.sliceAsBytes(xpad));
        try be.opI8Prep(xd, m, h, false);
        try be.opI8Gemm(yd, w.bytes, w.row_scale.?, w.rows, false);
        try be.endBatch();
        const got = try arena.alloc(f32, mpad * w.rows);
        try be.tensorDownload(yd, std.mem.sliceAsBytes(got));

        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        for (want, got[0 .. m * w.rows]) |e, a| {
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
        }
        const rel = @sqrt(l2_err / l2_ref);
        try stdout.print("qkv GEMM [{d}x{d}] x [{d}]  rel L2 {d:.5}\n", .{ w.rows, w.cols, m, rel });
    }

    // --- the CPU reference --------------------------------------------------
    var ws = try h3.Workspace.init(arena, dit.cfg, &layout);
    defer ws.deinit(arena);
    const cpu_v = try arena.alloc(f32, video.len);
    const cpu_a = try arena.alloc(f32, audio.len);
    var t0 = std.Io.Clock.real.now(io).nanoseconds;
    try h3.forward(&dit, io, arena, &ws, &layout, cpu_v, cpu_a, in);
    const cpu_ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;

    // --- the device path ----------------------------------------------------
    var sess = try h3_cuda.Session.init(be, arena, &dit, &layout);
    defer sess.deinit(be);
    var dws = try h3_cuda.Workspace.init(be, &dit, layout.seq_len);
    defer dws.deinit(be);
    const gpu_v = try arena.alloc(f32, video.len);
    const gpu_a = try arena.alloc(f32, audio.len);
    t0 = std.Io.Clock.real.now(io).nanoseconds;
    try h3_cuda.forward(&dit, be, &sess, &dws, io, arena, &layout, gpu_v, gpu_a, in, null);
    const gpu_ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;

    // --- localize: compare the TRUNK output per segment ---------------------
    // `ws.h` still holds the CPU trunk after `forward` (the heads use their own
    // scratch), and `dws.x_d` is the device's. Comparing per segment says which
    // rows diverge, which a single output figure cannot.
    {
        const dev_trunk = try arena.alloc(f32, layout.seq_len * dit.cfg.hidden);
        try be.tensorDownload(dws.x_d, std.mem.sliceAsBytes(dev_trunk));
        try stdout.print("\n-- trunk, per segment --\n", .{});
        for (layout.segments) |sg| {
            const n = sg.len() * dit.cfg.hidden;
            const off = sg.start * dit.cfg.hidden;
            var l2_ref: f64 = 0;
            var l2_err: f64 = 0;
            for (ws.h[off..][0..n], dev_trunk[off..][0..n]) |e, a| {
                l2_ref += @as(f64, e) * e;
                l2_err += @as(f64, e - a) * (e - a);
            }
            const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
            try stdout.print("  {s:<10} rows [{d:>5},{d:>5})  rel L2 {d:.5}\n", .{ @tagName(sg.kind), sg.start, sg.stop, rel });
        }
    }

    // --- compare ------------------------------------------------------------
    // Relative L2, not a per-element bound: the device path quantizes activations
    // to int8 and approximates the softmax, so individual elements move while the
    // field as a whole must not.
    var failures: usize = 0;
    for ([_]struct { name: []const u8, want: []const f32, got: []const f32 }{
        .{ .name = "video", .want = cpu_v, .got = gpu_v },
        .{ .name = "audio", .want = cpu_a, .got = gpu_a },
    }) |c| {
        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        var max_abs: f64 = 0;
        for (c.want, c.got) |e, a| {
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
            max_abs = @max(max_abs, @abs(@as(f64, e - a)));
        }
        const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
        // Depth-scaled, the same form `anima-cuda-test` uses and for the same
        // reason: the device path is W8A8 where this reference is W8A32 (exact
        // weights, f32 activations), so the two diverge with depth by design.
        //
        // The base is 1.5e-2 against anima's 6e-3, and that is a MEASURED
        // allowance, not a guess. At 150 packed rows the curve is flat at 0.003
        // through 45 blocks and then jumps: 0.005 at 48 and 0.009 video / 0.061
        // audio at 50. Isolations that rule out a defect: one qkv GEMM alone is
        // 0.0088 (the expected W8A8 level), and swapping the tensor-core
        // attention for the naive kernel moves the total by under 6%. What is
        // left is the last block, whose `fc2` scales average 3x smaller than
        // every other block's, so its contribution to the velocity is small in
        // magnitude and a fixed absolute quantization error there shows up as a
        // large RELATIVE one in the output.
        //
        // This bound says "no defect", not "matches the reference render". The
        // acceptance test is a render compared against ComfyUI, which itself runs
        // this checkpoint W8A8 and so may agree with the device path more closely
        // than this f32 host reference does. That comparison is owed.
        const tol = 1.5e-2 * (1.0 + @as(f64, @floatFromInt(dit.blocks.len)) / 8.0);
        const ok = rel < tol and std.math.isFinite(rel);
        if (!ok) failures += 1;
        try stdout.print("{s:<8} rel L2 {d:.5}  max |dv| {d:.5}  (tol {d:.4})  {s}\n", .{
            c.name, rel, max_abs, tol, if (ok) "ok" else "FAIL",
        });
    }
    try stdout.print("\ncpu {d:.0} ms, device {d:.0} ms ({d:.1}x)\n", .{ cpu_ms, gpu_ms, cpu_ms / gpu_ms });
    try stdout.flush();
    if (failures != 0) return error.DeviceMismatch;
}

/// `minimax-h3-vae-cuda-test`: the video VAE's ViT3D on the device against its
/// CPU reference.
///
/// Runs on the toy-width FIXTURE the CPU path is pinned with, not the real 5.2 GB
/// VAE: the fixture is f32 with the real head_dim and patch geometry, so it
/// exercises every device path at a size where the CPU side is instant. A real
/// checkpoint arm would add a 7-minute CPU reference for no extra coverage.
fn minimaxH3VaeCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const h3vae = TensorPencil.models.minimax_h3_vae;
    const h3vae_cuda = TensorPencil.models.minimax_h3_vae_cuda;
    const safetensors = TensorPencil.SafeTensors;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== minimax-h3-vae-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    const fixture = @embedFile("models/assets/minimax_h3_vae.safetensors");
    var st = try safetensors.initFromSlice(arena, fixture);
    defer st.deinit();
    var dec = try h3vae.VideoDecoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();
    if (!h3vae_cuda.supported(&dec)) {
        try stdout.print("FAIL: this VAE's weights have no device path here\n", .{});
        return error.UnsupportedCheckpoint;
    }
    try stdout.print("vae: {d} blocks, dim {d}, {d}x{d} heads, patch {d}x{d}\n", .{
        dec.cfg.n_layers, dec.cfg.dim, dec.cfg.heads, dec.cfg.head_dim, dec.cfg.patch_t, dec.cfg.patch,
    });

    // Shape override: the 512x512 render showed patch-grid artifacts the tiny
    // default shape does not, so the same sequence length has to be reachable
    // here. `in.z` is regenerated as noise when the shape is not the fixture's.
    const t: usize = if (std.c.getenv("TP_VAE_T")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2) else 2;
    const h: usize = if (std.c.getenv("TP_VAE_H")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 3) else 3;
    const w: usize = if (std.c.getenv("TP_VAE_W")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4) else 4;
    const z = if (t == 2 and h == 3 and w == 4)
        try (try st.require("in.z")).toF32Alloc(arena)
    else blk: {
        var prng = std.Random.DefaultPrng.init(0x5ae);
        const rr = prng.random();
        const zz = try arena.alloc(f32, dec.cfg.in_channels * t * h * w);
        for (zz) |*v| v.* = rr.floatNorm(f32);
        break :blk zz;
    };
    try stdout.print("shape t={d} h={d} w={d} -> {d} grid tokens, seq {d}\n", .{
        t, h, w, t * h * w, t * h * w + dec.cfg.n_register + 1,
    });

    const s = h3vae.outputShape(dec.cfg, t, h, w);
    const n = dec.cfg.out_channels * s.frames * s.height * s.width;
    const want = try arena.alloc(f32, n);
    const got = try arena.alloc(f32, n);

    // RAW volumes on both sides: the ImageNet finalize clamps, and a clamp hides
    // exactly the disagreement this is looking for.
    try h3vae.decodeVolume(&dec, io, arena, want, z, t, h, w);

    var sess = try h3vae_cuda.Session.init(be, arena, &dec, t, h, w);
    defer sess.deinit(be);
    var ws = try h3vae_cuda.Workspace.init(be, &dec, sess.seq, sess.grid);
    defer ws.deinit(be);
    try h3vae_cuda.decodeVolume(&dec, be, &sess, &ws, io, arena, got, z, t, h, w);

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f64 = 0;
    var nonfinite: usize = 0;
    for (want, got) |e, aa| {
        if (!std.math.isFinite(aa)) nonfinite += 1;
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - aa) * (e - aa);
        max_abs = @max(max_abs, @abs(@as(f64, e - aa)));
    }
    const rel = @sqrt(l2_err / l2_ref);
    // Dense f16/f32 GEMMs and an approximated softmax, no activation
    // quantization anywhere, so this is a much tighter regime than the int8 trunk.
    const ok = nonfinite == 0 and rel < 2e-3;
    try stdout.print("volume decode  rel L2 {d:.6}  max |dv| {d:.6}  {s}\n", .{ rel, max_abs, if (ok) "ok" else "FAIL" });
    try stdout.flush();
    if (!ok) return error.DeviceMismatch;
}
