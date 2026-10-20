//! TensorPencil CLI, thin driver over the TensorPencil library module.

const std = @import("std");
const Io = std.Io;

const TensorPencil = @import("TensorPencil");
const quant_attrib = @import("quant_attrib.zig");
const conv_cuda = TensorPencil.models.conv_cuda;

/// `--act f32|f16|bf16` on a device check: the width its device forwards store
/// activations at, null for each check's own. f32 and the default are held to the
/// check's tolerance; a 16-bit run is opt-in and fails only on a non-finite output
/// (`checkTol`).
var test_act: ?TensorPencil.pipeline.Act = null;

fn testAct() TensorPencil.pipeline.Act {
    return test_act orelse .f32;
}

fn checkTol(tol: f64) f64 {
    return if (testAct() == .f32) tol else std.math.inf(f64);
}

/// `args` without a device check's `--act X`, which `test_act` takes: each command's
/// own parser reads an unknown argument as a checkpoint path.
fn takeTestAct(arena: std.mem.Allocator, args: []const [:0]const u8) ![]const [:0]const u8 {
    if (args.len < 2 or !std.mem.endsWith(u8, args[1], "-test")) return args;
    var out: std.ArrayList([:0]const u8) = .empty;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--act") and i + 1 < args.len) {
            test_act = std.meta.stringToEnum(TensorPencil.pipeline.Act, args[i + 1]) orelse return error.InvalidArgs;
            i += 1;
        } else try out.append(arena, args[i]);
    }
    return out.items;
}

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try takeTestAct(arena, try init.minimal.args.toSlice(arena));
    // Per-step buffers churn (alloc/free every forward pass); the process arena
    // never frees, so a render gets a real allocator. A clip makes the difference
    // impossible to miss: three [seq][hidden] f32 buffers a step, 2.8 GB per step
    // at 15 seconds of 512x768, none of it coming back until the process exits.
    const gpa = std.heap.smp_allocator;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;
    // Flush on the ERROR path too. The `flush` at the bottom of this function is
    // never reached when a subcommand returns, so every "print a diagnostic then
    // return the error" site printed into a buffer that was thrown away, and the
    // user saw a bare stack trace instead of the sentence explaining it.
    defer stdout.flush() catch {};
    // A forward that checks its own output aborts the device check it runs in.
    errdefer |err| if (err == error.NonFiniteActivation) if (test_act) |a|
        stdout.print("FAILED: a device forward went non-finite at --act {t}\n", .{a}) catch {};

    // In a device check the CPU side is the reference. Below `small_m_max` rows the
    // CPU block-quant GEMM also quantizes the activation (ggml's vec_dot), which on an
    // outlier-heavy model made the reference itself several percent off.
    if (args.len >= 2 and std.mem.endsWith(u8, args[1], "-test")) TensorPencil.ops.matmul.exact_activations = true;

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
            ctx.coop_i8_m,                                                                    ctx.coop_i8_n, ctx.coop_i8_k,
            if (ctx.coop_i8_m != 0) "int8 tensor cores available" else "no int8 coop config",
        });
        if (ctx.coop_m == 16 and ctx.coop_n == 16 and ctx.coop_k == 16) {
            const P = struct { lanes: u32, pin: bool, sgs: u32, shared: bool, stride: u32, iters: u32, off: u32 = 0, bitcast: bool = false };
            const n = ctx.subgroup_size;
            for ([_]P{
                .{ .lanes = 32, .pin = true, .sgs = 1, .shared = false, .stride = 16, .iters = 1 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 16, .iters = 1 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 1 },
                .{ .lanes = 32, .pin = true, .sgs = 1, .shared = false, .stride = 16, .iters = 2 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 2 },
                .{ .lanes = n, .pin = false, .sgs = 4, .shared = true, .stride = 32, .iters = 2 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 1, .off = 16 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 1, .off = 512 },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 1, .off = 8192 },
                .{ .lanes = 32, .pin = true, .sgs = 1, .shared = true, .stride = 16, .iters = 1, .bitcast = true },
                .{ .lanes = 32, .pin = true, .sgs = 4, .shared = true, .stride = 32, .iters = 1, .off = 512, .bitcast = true },
            }) |p| {
                const how = .{ p.lanes, if (p.pin) "pinned" else "unpinned", p.sgs, if (p.shared) "shared" else "storage", p.stride, p.iters, p.off, if (p.bitcast) " bitcast" else "" };
                if (ctx.coopProbe(p.lanes, p.pin, p.sgs, p.shared, p.stride, p.iters, p.off, p.bitcast)) |r| {
                    try stdout.print("coop probe {d} lanes {s} x{d} {s} stride {d} iters {d} offset {d}{s}: {s}, ran at {d}\n", how ++ .{ if (r.correct) "correct" else "WRONG", r.lanes });
                } else |err| try stdout.print("coop probe {d} lanes {s} x{d} {s} stride {d} iters {d} offset {d}{s}: {t}\n", how ++ .{err});
                try stdout.flush();
            }
        }
        if (ctx.pipe_coop_f16w != .null_handle) {
            // The f16 coop GEMM on exact small integers, its weight converted on the host
            // and, separately, transposed on the device: which half a wrong answer is in.
            const m = 32;
            const rows = 64;
            const cols = 64;
            var xs: [m * cols]f32 = undefined;
            var ws: [rows * cols]f32 = undefined;
            var wh: [rows * cols]f16 = undefined;
            for (0..m) |t| for (0..cols) |k| {
                xs[t * cols + k] = @floatFromInt(@as(i32, @intCast((t * 7 + k * 3) % 9)) - 4);
            };
            for (0..rows) |r| for (0..cols) |k| {
                ws[r * cols + k] = @floatFromInt(@as(i32, @intCast((r * 5 + k) % 7)) - 3);
                wh[r * cols + k] = @floatCast(ws[r * cols + k]);
            };
            const zeros: [rows]f32 = @splat(0);
            var x_d = try ctx.tensorCreate(xs.len * 4);
            defer ctx.tensorDestroy(&x_d);
            try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(&xs));
            var y_d = try ctx.tensorCreate(m * rows * 4);
            defer ctx.tensorDestroy(&y_d);
            for ([_][]const u8{ "host-converted weight", "device-transposed weight" }, 0..) |label, route| {
                if (route == 0) try ctx.opMatmulCoopF16W(y_d, 0, x_d, m, &ws, rows, cols, &zeros) else try ctx.opMatmulCoopF16Wh(y_d, 0, x_d, m, std.mem.sliceAsBytes(&wh), rows, cols, &zeros);
                var y: [m * rows]f32 = undefined;
                try ctx.tensorDownload(y_d, std.mem.sliceAsBytes(&y));
                var exact: usize = 0;
                var zero: usize = 0;
                for (0..m) |t| for (0..rows) |r| {
                    var want: f32 = 0;
                    for (0..cols) |k| want += xs[t * cols + k] * ws[r * cols + k];
                    if (y[t * rows + r] == want) exact += 1;
                    if (y[t * rows + r] == 0) zero += 1;
                };
                try stdout.print("coop f16 GEMM, {s}: {d}/{d} exact, {d} zero, y[0..3] = {d} {d} {d}\n", .{ label, exact, y.len, zero, y[0], y[1], y[2] });
                try stdout.flush();
            }
        }
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "act-derive")) {
        try actDerive(gpa, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-fp8-test")) {
        try cudaFp8Test(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-encode-test")) {
        try cudaEncodeTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "sd-cuda-test")) {
        // `libs` selects the cuBLASLt/cuDNN arm, which takes a DIFFERENT attention
        // path (true head width, no padding), so both are worth running.
        var ckpt: []const u8 = TensorPencil.model_root ++ "/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";
        var libs = false;
        for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) libs = true else ckpt = a;
        }
        try sdCudaTest(arena, io, stdout, ckpt, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "te-test")) {
        var path: []const u8 = TensorPencil.model_root ++ "/text_encoders/Qwen3-4B-Q4_K_M.gguf";
        var ref: []const u8 = "";
        var variant: TensorPencil.models.qwen3.Variant = .zimage;
        var dump: []const u8 = "";
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--ref") and i + 1 < args.len) {
                i += 1;
                ref = args[i];
            } else if (std.mem.eql(u8, args[i], "--te-int-gemm") and i + 1 < args.len) {
                i += 1;
                TensorPencil.models.dual_lin.encoder_int8 = std.meta.stringToEnum(TensorPencil.models.lin_cuda.IntGemm, args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--gguf-gemm") and i + 1 < args.len) {
                // The encoder's block-quant route (`lin_cuda.encoder_gemm`), so the
                // activation-quantizing routes can be measured against the CPU encode.
                i += 1;
                TensorPencil.models.lin_cuda.encoder_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--dump") and i + 1 < args.len) {
                // Raw f32 conditioning per arm, `<prefix>.<arm>.f32`, for a byte
                // compare of two builds: rel L2 to four digits cannot tell "same
                // kernels" from "same to 1e-5".
                i += 1;
                dump = args[i];
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
        try teTest(arena, io, stdout, path, ref, variant, dump);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "sensenova-cuda-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/sensenova/sensenova_u15_q4_k.gguf";
        // A wiring check: the default route quantizes nothing on the device, so the CPU
        // reference is the exact answer; `--gguf-gemm` picks another.
        TensorPencil.models.lin_cuda.blockq_gemm = .bf16;
        var libs = false;
        var layers: usize = 2;
        var toks: usize = 20;
        var sigma_arg: f32 = 0.75;
        var canvas: ?[2]usize = null;
        var lora_path: []const u8 = "";
        var bench: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true //
            else if (std.mem.eql(u8, args[i], "--prefix-bench") and i + 1 < args.len) {
                i += 1;
                bench = std.fmt.parseInt(usize, args[i], 10) catch 5;
            } else if (std.mem.eql(u8, args[i], "--lora") and i + 1 < args.len) {
                i += 1;
                lora_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--lora-bf16-a")) {
                // The control row: the sidecar's A GEMM on the bf16 activation.
                TensorPencil.models.lin_cuda.lora_i8_a = false;
            } else if ((std.mem.eql(u8, args[i], "--te-int-gemm") or std.mem.eql(u8, args[i], "--prefix-gemm")) and i + 1 < args.len) {
                i += 1;
                TensorPencil.models.dual_lin.encoder_int8 = std.meta.stringToEnum(TensorPencil.models.lin_cuda.IntGemm, args[i]) orelse {
                    try stdout.print("unknown text encoder int gemm '{s}' (expected: w8a8, weight_only)\n", .{args[i]});
                    return error.InvalidArgs;
                };
            } else if (std.mem.eql(u8, args[i], "--gguf-gemm") and i + 1 < args.len) {
                // The block-quant route, to tell a route's own arithmetic from a defect.
                i += 1;
                TensorPencil.models.lin_cuda.blockq_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch 2;
            } else if (std.mem.eql(u8, args[i], "--prompt-tokens") and i + 1 < args.len) {
                i += 1;
                toks = std.fmt.parseInt(usize, args[i], 10) catch 20;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch 0.75;
            } else if (std.mem.eql(u8, args[i], "--canvas") and i + 2 < args.len) {
                canvas = .{ try std.fmt.parseInt(usize, args[i + 1], 10), try std.fmt.parseInt(usize, args[i + 2], 10) };
                i += 2;
            } else ckpt = args[i];
        }
        try sensenovaCudaTest(arena, io, stdout, ckpt, libs, layers, toks, sigma_arg, canvas, lora_path, bench);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "sensenova-vk-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors";
        var layers: usize = 2;
        var toks: usize = 20;
        var sigma_arg: f32 = 0.75;
        var canvas: ?[2]usize = null;
        var lora_path: []const u8 = "";
        var bench: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--lora") and i + 1 < args.len) {
                i += 1;
                lora_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch 2;
            } else if (std.mem.eql(u8, args[i], "--prompt-tokens") and i + 1 < args.len) {
                i += 1;
                toks = std.fmt.parseInt(usize, args[i], 10) catch 20;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch 0.75;
            } else if (std.mem.eql(u8, args[i], "--prefix-bench") and i + 1 < args.len) {
                i += 1;
                bench = std.fmt.parseInt(usize, args[i], 10) catch 0;
            } else if (std.mem.eql(u8, args[i], "--canvas") and i + 2 < args.len) {
                canvas = .{ try std.fmt.parseInt(usize, args[i + 1], 10), try std.fmt.parseInt(usize, args[i + 2], 10) };
                i += 2;
            } else ckpt = args[i];
        }
        try sensenovaVkTest(arena, io, stdout, ckpt, layers, toks, sigma_arg, canvas, lora_path, bench);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "zimage-cuda-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/checkpoints/zit/unstableRevolution_V2Fp16.safetensors";
        var vae: []const u8 = TensorPencil.model_root ++ "/vae/z-image-turbo.vae.safetensors";
        var lora: TestLoraArg = .{};
        var libs = false;
        var layers: usize = 0; // 0 = the full trunk
        var lat_arg: usize = 16;
        var cap_arg: usize = 20;
        var sigma_arg: f32 = 0.75;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true //
            else if (std.mem.eql(u8, args[i], "--vae") and i + 1 < args.len) {
                i += 1;
                vae = args[i];
            } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch 0;
            } else if (std.mem.eql(u8, args[i], "--dequant-at-load") and i + 1 < args.len) {
                i += 1;
                TensorPencil.models.lin.dequant_at_load = std.meta.stringToEnum(@TypeOf(TensorPencil.models.lin.dequant_at_load), args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch sigma_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--gguf-gemm") and i + 1 < args.len) {
                // A GGUF trunk has several decode routes and they are different GEMMs,
                // so the device check has to be able to name one rather than only `auto`.
                i += 1;
                TensorPencil.models.lin_cuda.blockq_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--lora") and i + 1 < args.len) {
                i += 1;
                lora.path = args[i];
            } else if (std.mem.eql(u8, args[i], "--strength") and i + 1 < args.len) {
                i += 1;
                lora.strength = try std.fmt.parseFloat(f32, args[i]);
            } else if (std.mem.eql(u8, args[i], "--lora-bf16-a")) {
                TensorPencil.models.lin_cuda.lora_i8_a = false;
            } else ckpt = args[i];
        }
        try zimageCudaTest(arena, io, stdout, ckpt, vae, libs, layers, lat_arg, cap_arg, sigma_arg, lora);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "mage-vae-vk-test")) {
        var vae: []const u8 = TensorPencil.model_root ++ "/vae/mage_flow_vae_bf16.safetensors";
        // Two rows of three windows, the last of each padded.
        var lat_h: usize = 40;
        var lat_w: usize = 72;
        var bench: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--bench") and i + 1 < args.len) {
                bench = std.fmt.parseInt(usize, args[i + 1], 10) catch 0;
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 2 < args.len) {
                lat_h = std.fmt.parseInt(usize, args[i + 1], 10) catch lat_h;
                lat_w = std.fmt.parseInt(usize, args[i + 2], 10) catch lat_w;
                i += 2;
            } else vae = args[i];
        }
        try mageVaeVkTest(arena, io, stdout, vae, lat_h, lat_w, bench);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "mage-vae-cuda-test")) {
        var vae: []const u8 = TensorPencil.model_root ++ "/vae/mage_flow_vae_bf16.safetensors";
        var libs = false;
        // Two rows of three windows, the last of each padded.
        var lat_h: usize = 40;
        var lat_w: usize = 72;
        var bench: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true //
            else if (std.mem.eql(u8, args[i], "--bench") and i + 1 < args.len) {
                bench = std.fmt.parseInt(usize, args[i + 1], 10) catch 0;
                i += 1;
            }
            else if (std.mem.eql(u8, args[i], "--lat") and i + 2 < args.len) {
                lat_h = std.fmt.parseInt(usize, args[i + 1], 10) catch lat_h;
                lat_w = std.fmt.parseInt(usize, args[i + 2], 10) catch lat_w;
                i += 2;
            } else vae = args[i];
        }
        try mageVaeCudaTest(arena, io, stdout, vae, libs, lat_h, lat_w, bench);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "mageflow-bench")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors";
        var backend: []const u8 = "cuda";
        var lat_arg: usize = 64;
        var cap_arg: usize = 64;
        var reps: usize = 5;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
                i += 1;
                backend = args[i];
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--reps") and i + 1 < args.len) {
                i += 1;
                reps = std.fmt.parseInt(usize, args[i], 10) catch reps;
            } else ckpt = args[i];
        }
        try mageflowBench(arena, io, stdout, ckpt, backend, lat_arg, cap_arg, reps);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "mageflow-vk-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors";
        var layers: usize = 0; // 0 = the full trunk
        var lat_arg: usize = 16;
        var cap_arg: usize = 20;
        var refs_arg: usize = 0;
        var sigma_arg: f32 = 0.75;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch 0;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch sigma_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--refs") and i + 1 < args.len) {
                i += 1;
                refs_arg = std.fmt.parseInt(usize, args[i], 10) catch refs_arg;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--unbatched")) {
                TensorPencil.models.mageflow_gpu.force_unbatched = true;
            } else ckpt = args[i];
        }
        try mageflowVkTest(arena, io, stdout, ckpt, layers, lat_arg, cap_arg, refs_arg, sigma_arg);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "moss-image-te-test")) {
        var te: []const u8 = TensorPencil.model_root ++ "/text_encoders/qwen3.5_9b-INT8_CONVROT.safetensors";
        var backend: []const u8 = "zig-cuda";
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
                i += 1;
                backend = args[i];
            } else te = args[i];
        }
        try mossImageTeTest(arena, io, stdout, te, backend);
    } else if (args.len >= 2 and (std.mem.eql(u8, args[1], "moss-image-cuda-test") or std.mem.eql(u8, args[1], "moss-image-vk-test"))) {
        const vk = std.mem.eql(u8, args[1], "moss-image-vk-test");
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/moss/moss_image_sft-INT8_CONVROT.safetensors";
        var libs = false;
        var layers: usize = 0;
        var lat_arg: usize = 16;
        var cap_arg: usize = 20;
        var sigma_arg: f32 = 0.75;
        var prof = false;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) {
                libs = true;
            } else if (std.mem.eql(u8, args[i], "--prof")) {
                prof = true;
            } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch layers;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch sigma_arg;
            } else ckpt = args[i];
        }
        if (prof and !vk) return mossImageCudaProf(arena, io, stdout, ckpt, libs, lat_arg, cap_arg);
        try mossImageTest(arena, io, stdout, ckpt, if (vk) .vulkan else if (libs) .cuda else .zig_cuda, layers, lat_arg, cap_arg, sigma_arg);
    } else if (args.len >= 2 and (std.mem.eql(u8, args[1], "qwen-image21-cuda-test") or std.mem.eql(u8, args[1], "qwen-image21-vk-test"))) {
        const vk = std.mem.eql(u8, args[1], "qwen-image21-vk-test");
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/qwen2.1/qwen_image_2.1_turbo_int8_convrot.safetensors";
        var libs = false;
        var layers: usize = 0;
        var lat_arg: usize = 16;
        var cap_arg: usize = 20;
        var sigma_arg: f32 = 0.75;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) {
                libs = true;
            } else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch layers;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch sigma_arg;
            } else ckpt = args[i];
        }
        try qwenImage21DevTest(arena, io, stdout, ckpt, if (vk) .vulkan else if (libs) .cuda else .zig_cuda, layers, lat_arg, cap_arg, sigma_arg);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "mageflow-cuda-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors";
        var libs = false;
        var layers: usize = 0; // 0 = the full trunk
        var lat_arg: usize = 16;
        var cap_arg: usize = 20;
        var refs_arg: usize = 0;
        var sigma_arg: f32 = 0.75;
        var lora: TestLoraArg = .{};
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true //
            else if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
                i += 1;
                layers = std.fmt.parseInt(usize, args[i], 10) catch 0;
            } else if (std.mem.eql(u8, args[i], "--sigma") and i + 1 < args.len) {
                i += 1;
                sigma_arg = std.fmt.parseFloat(f32, args[i]) catch sigma_arg;
            } else if (std.mem.eql(u8, args[i], "--cap") and i + 1 < args.len) {
                i += 1;
                cap_arg = std.fmt.parseInt(usize, args[i], 10) catch cap_arg;
            } else if (std.mem.eql(u8, args[i], "--lat") and i + 1 < args.len) {
                i += 1;
                lat_arg = std.fmt.parseInt(usize, args[i], 10) catch lat_arg;
            } else if (std.mem.eql(u8, args[i], "--refs") and i + 1 < args.len) {
                i += 1;
                refs_arg = std.fmt.parseInt(usize, args[i], 10) catch refs_arg;
            } else if (std.mem.eql(u8, args[i], "--dequant-at-load") and i + 1 < args.len) {
                i += 1;
                TensorPencil.models.lin.dequant_at_load = std.meta.stringToEnum(@TypeOf(TensorPencil.models.lin.dequant_at_load), args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--gguf-gemm") and i + 1 < args.len) {
                i += 1;
                TensorPencil.models.lin_cuda.blockq_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, args[i]) orelse return error.InvalidArgs;
            } else if (std.mem.eql(u8, args[i], "--lora") and i + 1 < args.len) {
                i += 1;
                lora.path = args[i];
            } else if (std.mem.eql(u8, args[i], "--strength") and i + 1 < args.len) {
                i += 1;
                lora.strength = try std.fmt.parseFloat(f32, args[i]);
            } else if (std.mem.eql(u8, args[i], "--lora-bf16-a")) {
                TensorPencil.models.lin_cuda.lora_i8_a = false;
            } else ckpt = args[i];
        }
        try mageflowCudaTest(arena, io, stdout, ckpt, libs, layers, lat_arg, cap_arg, refs_arg, sigma_arg, lora);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "anima-cuda-test")) {
        var ckpt: []const u8 = TensorPencil.model_root ++ "/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors";
        var libs = false;
        var lora: TestLoraArg = .{};
        var layers: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const a = args[i];
            if (std.mem.eql(u8, a, "libs")) {
                libs = true;
            } else if (std.mem.eql(u8, a, "--lora-bf16-a")) {
                TensorPencil.models.lin_cuda.lora_i8_a = false;
            } else if (std.mem.eql(u8, a, "--lora") and i + 1 < args.len) {
                i += 1;
                lora.path = args[i];
            } else if (std.mem.eql(u8, a, "--strength") and i + 1 < args.len) {
                i += 1;
                lora.strength = try std.fmt.parseFloat(f32, args[i]);
            } else if (std.mem.eql(u8, a, "--layers") and i + 1 < args.len) {
                i += 1;
                layers = try std.fmt.parseInt(usize, args[i], 10);
            } else ckpt = a;
        }
        try animaCudaTest(arena, io, stdout, ckpt, libs, lora, layers);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "tune-coop")) {
        try tuneCoop(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-norm-bench")) {
        try vkNormBench(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "dual-gemv-bench")) {
        try dualGemvBench(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-bq-bench")) {
        try vkBqBench(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-wo-bench")) {
        try vkWoBench(arena, io, stdout, if (args.len >= 3) try std.fmt.parseInt(usize, args[2], 10) else 4096);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-prep-bench")) {
        try vkPrepBench(arena, io, stdout, if (args.len >= 3) try std.fmt.parseInt(usize, args[2], 10) else 4096);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vk-gemv-bench")) {
        try vkGemvBench(arena, io, stdout);
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
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "gemma-vit-test")) {
        try gemmaVitTest(arena, io, stdout, if (args.len >= 3) args[2] else "/mnt/watch/models/llm/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/mmproj-F16.gguf");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "gemma4-vit-test")) {
        try gemma4VitTest(arena, io, stdout, if (args.len >= 3) args[2] else "/mnt/watch/models/llm/mmproj-gemma-4-12b-it-qat-q4_0.gguf");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "vit35-test")) {
        try vit35Test(arena, io, stdout, if (args.len >= 3) args[2] else "/mnt/watch/models/llm/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/mmproj-Qwen3.6-27B-BF16.gguf");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "taehv-test")) {
        try taehvTest(arena, io, stdout, if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/vae_approx/taew2_1.safetensors");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "wan-vae-test")) {
        // `wan-vae-test [latent side] [vae path] [naive-attn]`: the last runs the CUDA
        // arms' mid-block attention through the f32 kernel, to isolate its share.
        const zh: usize = if (args.len >= 3) (std.fmt.parseInt(usize, args[2], 10) catch 16) else 16;
        if (args.len >= 5 and std.mem.eql(u8, args[4], "naive-attn")) TensorPencil.models.sd_vae_cuda.force_naive_attn = true;
        try wanVaeTest(arena, io, stdout, zh, if (args.len >= 4) args[3] else TensorPencil.model_root ++ "/vae/krea2RealVae_v10.safetensors");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "dump-coop-spv")) {
        const frag: u32 = if (args.len >= 4) std.fmt.parseInt(u32, args[3], 10) catch 16 else 16;
        const tile: u32 = if (args.len >= 5) std.fmt.parseInt(u32, args[4], 10) catch 64 else 64;
        try dumpCoopSpv(arena, io, stdout, if (args.len >= 3) args[2] else "", frag, tile);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "dump-ptx")) {
        try dumpPtx(arena, io, stdout, if (args.len >= 3) args[2] else "", if (args.len >= 4) args[3] else "");
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-dit-test")) {
        var path: []const u8 = TensorPencil.model_root ++ "/diffusion_models/krea2/animosity_krea2Ver10-INT8_CONVROT.safetensors";
        var lat: usize = 32;
        var loop = false;
        var libs = false;
        var layers: usize = 0;
        var lora_path: []const u8 = "";
        var strength: f32 = 1;
        var positional: usize = 0;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            const a = args[i];
            // A GGUF DiT has two decode routes and they are different GEMMs, so the
            // device check has to be able to name one rather than only testing `auto`.
            const routes = [_]struct { []const u8, TensorPencil.models.lin_cuda.BlockQGemm }{ .{ "int8", .int8 }, .{ "int4", .int4 }, .{ "bf16", .bf16 }, .{ "mmq", .mmq } };
            const route = for (routes) |r| {
                if (std.mem.eql(u8, a, r[0])) break r[1];
            } else null;
            if (route) |r| {
                TensorPencil.models.lin_cuda.blockq_gemm = r;
            } else if (std.mem.eql(u8, a, "loop")) {
                loop = true;
            } else if (std.mem.eql(u8, a, "libs")) {
                libs = true;
            } else if (std.mem.eql(u8, a, "--lora-bf16-a")) {
                TensorPencil.models.lin_cuda.lora_i8_a = false;
            } else if (std.mem.eql(u8, a, "--layers") and i + 1 < args.len) {
                i += 1;
                layers = try std.fmt.parseInt(usize, args[i], 10);
            } else if (std.mem.eql(u8, a, "--lora") and i + 1 < args.len) {
                i += 1;
                lora_path = args[i];
            } else if (std.mem.eql(u8, a, "--strength") and i + 1 < args.len) {
                i += 1;
                strength = try std.fmt.parseFloat(f32, args[i]);
            } else {
                if (positional == 0) path = a else lat = std.fmt.parseInt(usize, a, 10) catch lat;
                positional += 1;
            }
        }
        try cudaDitTest(arena, io, stdout, path, lat, loop, libs, layers, lora_path, strength);
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
        const path = if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/diffusion_models/krea2/animosity_krea2Ver10-INT8_CONVROT.safetensors";
        const seq_txt: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 448) else 448;
        try cudaTxtFusionTest(arena, io, stdout, path, seq_txt);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-attn-cmp")) {
        try cudaAttnCmp(arena, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "cuda-stream-test")) {
        const path = if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/diffusion_models/krea2/animosity_krea2Ver10-INT8_CONVROT.safetensors";
        const lat: usize = if (args.len >= 4) (std.fmt.parseInt(usize, args[3], 10) catch 32) else 32;
        const budget_gib: f64 = if (args.len >= 5) (std.fmt.parseFloat(f64, args[4]) catch 3.0) else 3.0;
        try cudaStreamTest(arena, io, stdout, path, lat, budget_gib);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "generate")) {
        try generate(gpa, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "generate-clip")) {
        try generateClip(gpa, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vae-encode-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else TensorPencil.model_root ++ "/vae/minimax_h3_video_vae_fp16.safetensors";
        const libs = for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) break true;
        } else false;
        try minimaxH3VaeEncodeCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-audio-encode-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else TensorPencil.model_root ++ "/vae/minimax_h3_audio_vae_fp32.safetensors";
        const libs = for (args[2..]) |a| {
            if (std.mem.eql(u8, a, "libs")) break true;
        } else false;
        try minimaxH3AudioEncodeCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-audio-vk-test")) {
        const ck = if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/vae/minimax_h3_audio_vae_fp32.safetensors";
        try minimaxH3AudioVkTest(arena, io, stdout, ck);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-audio-cuda-test")) {
        const ck = if (args.len >= 3 and !std.mem.eql(u8, args[2], "libs")) args[2] else TensorPencil.model_root ++ "/vae/minimax_h3_audio_vae_fp32.safetensors";
        var libs = false;
        for (args[2..]) |a| if (std.mem.eql(u8, a, "libs")) {
            libs = true;
        };
        try minimaxH3AudioCudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 8 and std.mem.eql(u8, args[1], "quant-gemm-test")) {
        try quantGemmTest(arena, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "quant-attrib")) {
        try quant_attrib.cli(io, stdout, args[2..]);
    } else if (args.len >= 3 and std.mem.eql(u8, args[1], "velocity-diff")) {
        try velocityDiff(io, stdout, args[2..]);
    } else if (args.len >= 4 and std.mem.eql(u8, args[1], "cond-diff")) {
        try condDiff(io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lin-vk-test")) {
        var shape: [3]usize = .{ 154, 256, 1024 };
        if (args.len >= 6 and std.mem.eql(u8, args[2], "--shape")) {
            for (&shape, args[3..6]) |*v, a| v.* = try std.fmt.parseInt(usize, a, 10);
        }
        try linVkTest(arena, io, stdout, shape);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lin-cuda-test")) {
        try linCudaTest(arena, io, stdout, args.len >= 3 and std.mem.eql(u8, args[2], "libs"));
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lora-vk-test")) {
        try loraVkTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lora-cuda-test")) {
        try loraCudaTest(arena, io, stdout, args.len >= 3 and std.mem.eql(u8, args[2], "libs"));
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "lora-cuda-bench")) {
        var libs = false;
        var m: usize = 4096;
        var rank_buf: [4]usize = .{ 128, 0, 0, 0 };
        var nranks: usize = 1;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "libs")) libs = true else if (std.mem.eql(u8, args[i], "--m") and i + 1 < args.len) {
                i += 1;
                m = try std.fmt.parseInt(usize, args[i], 10);
            } else if (std.mem.eql(u8, args[i], "--rank") and i + 1 < args.len) {
                i += 1;
                nranks = 0;
                var it = std.mem.splitScalar(u8, args[i], ',');
                while (it.next()) |r| : (nranks += 1) {
                    if (nranks == rank_buf.len) return error.InvalidArgument;
                    rank_buf[nranks] = try std.fmt.parseInt(usize, r, 10);
                }
            }
        }
        try loraCudaBench(arena, io, stdout, libs, m, rank_buf[0..nranks]);
    } else if (args.len >= 4 and std.mem.eql(u8, args[1], "lora-sidecar-bench")) {
        var size: usize = 1024;
        var rounds: usize = 7;
        var backend: TensorPencil.pipeline.Backend = .cuda;
        var te: []const u8 = "";
        var vae: []const u8 = "";
        var i: usize = 4;
        while (i + 1 < args.len) : (i += 2) {
            if (std.mem.eql(u8, args[i], "--size")) size = try std.fmt.parseInt(usize, args[i + 1], 10);
            if (std.mem.eql(u8, args[i], "--rounds")) rounds = try std.fmt.parseInt(usize, args[i + 1], 10);
            if (std.mem.eql(u8, args[i], "--backend")) backend = TensorPencil.pipeline.Backend.fromStr(args[i + 1]) orelse return error.InvalidArgs;
            if (std.mem.eql(u8, args[i], "--text-encoder")) te = args[i + 1];
            if (std.mem.eql(u8, args[i], "--vae")) vae = args[i + 1];
        }
        try loraSidecarBench(arena, io, stdout, args[2], args[3], size, rounds, backend, te, vae);
    } else if (args.len >= 4 and std.mem.eql(u8, args[1], "lora-restack-test")) {
        var te: []const u8 = "";
        var vae: []const u8 = "";
        var backend: TensorPencil.pipeline.Backend = .cuda;
        var i: usize = 4;
        while (i + 1 < args.len) : (i += 2) {
            if (std.mem.eql(u8, args[i], "--text-encoder")) te = args[i + 1] else if (std.mem.eql(u8, args[i], "--vae")) vae = args[i + 1] else if (std.mem.eql(u8, args[i], "--backend")) backend = TensorPencil.pipeline.Backend.fromStr(args[i + 1]) orelse return error.InvalidArgs;
        }
        try loraRestackTest(arena, io, stdout, args[2], args[3], te, vae, backend);
    } else if (args.len >= 3 and std.mem.eql(u8, args[1], "lora-info")) {
        try loraInfo(arena, io, stdout, args[2..]);
    } else if (args.len >= 3 and std.mem.eql(u8, args[1], "diffusion-profile")) {
        try diffusionProfile(arena, io, stdout, args[2..]);
    } else if (args.len >= 4 and std.mem.eql(u8, args[1], "lora-merge-check")) {
        try loraMergeCheck(arena, io, stdout, args[2..]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "dual-cuda-test")) {
        try dualCudaTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vae-vk-test")) {
        try minimaxH3VaeVkTest(arena, io, stdout);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vae-cuda-test")) {
        try minimaxH3VaeCudaTest(arena, io, stdout, args.len >= 3 and std.mem.eql(u8, args[2], "libs"));
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-vk-test")) {
        const ck = if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/diffusion_models/h3/10erosMaxInt8Ref2va_v10Beta.safetensors";
        try minimaxH3VkTest(arena, io, stdout, ck);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "minimax-h3-cuda-test")) {
        const ck = if (args.len >= 3) args[2] else TensorPencil.model_root ++ "/diffusion_models/h3/10erosMaxInt8Ref2va_v10Beta.safetensors";
        const libs = args.len >= 4 and std.mem.eql(u8, args[3], "libs");
        try minimaxH3CudaTest(arena, io, stdout, ck, libs);
    } else if (args.len >= 6 and std.mem.eql(u8, args[1], "decode-latent")) {
        try decodeLatent(arena, io, stdout, args[2], args[3], args[4], args[5]);
    } else if (args.len >= 2 and std.mem.eql(u8, args[1], "bench-matmul")) {
        try benchMatmul(arena, io, stdout);
    } else if (args.len >= 3 and std.mem.eql(u8, args[1], "inspect")) {
        // Container.open, not SafeTensors.open: the reader is picked by magic, so
        // `inspect` works on a GGUF too. Handing one to the safetensors reader reports
        // `InvalidHeader`, which says nothing about what happened.
        var ct = try TensorPencil.pipeline.Container.open(arena, io, args[2]);
        defer ct.deinit();
        const store = ct.store();
        const names = store.names();
        try stdout.print("{s}: {d} tensors\n", .{ args[2], names.len });
        for (names) |name| {
            const view = store.get(name).?;
            try stdout.print("  {s}  {any}  {t}{s}\n", .{
                name,            view.info.shape.slice(),
                view.info.dtype, if (view.info.flat_blocks) "  (flat blocks)" else "",
            });
        }
    } else {
        try stdout.print(
            \\TensorPencil — diffusion inference engine
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
            \\      --sampler euler    sampler, each ComfyUI's of that name:
            \\                         euler | euler_ancestral | heun | dpm_2
            \\                         | dpm_2_ancestral | dpmpp_2s_ancestral
            \\                         | dpmpp_sde | dpmpp_2m | dpmpp_2m_sde
            \\                         | dpmpp_2m_sde_heun | dpmpp_3m_sde.
            \\                         The ancestral ones step short of the next sigma
            \\                         and add the gap back as noise; the SDE ones
            \\                         take their noise from a seed-determined
            \\                         Brownian tree instead. heun, dpm_2 and the two
            \\                         2S/2-ancestral ones plus dpmpp_sde run the
            \\                         model TWICE per step, so the same --steps is
            \\                         twice the time
            \\      --eta 1.0          noise level for a stochastic sampler
            \\                         (0 = deterministic DPM++(2M), or plain euler)
            \\      --s-noise 1.0      multiplier on that noise
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
            \\      --cond-noise       perturb the text conditioning by a curve.
            \\                         An expression in t (depth over the encoder
            \\                         taps: krea2 stacks 12, SDXL is CLIP-L then
            \\                         CLIP-G, SenseNova is its layers; families
            \\                         with one tap read t = 0), l and n (token
            \\                         index and count, so a*(1-l/n) fronts the
            \\                         prompt) and a (--cond-noise-amount). A bare
            \\                         number is a flat curve. Same expression
            \\                         language as tp-llm's --weight-noise. Absent
            \\                         is off, and off is bit-identical
            \\      --cond-noise-amount  the curve's `a` (default 1). Noise is
            \\                         scaled by the conditioning's own RMS, so one
            \\                         amount means the same on every family
            \\      --cond-noise-seed  pin the perturbation. Without it the field
            \\                         follows --seed, so --repeat draws a fresh
            \\                         one each image; set it to hold the damage
            \\                         still while sweeping something else
            \\      --cond-noise-negative  scale on the CFG negative branch
            \\                         (default 0, leaving it alone: one field on
            \\                         both branches partly cancels)
            \\      --cond-noise-shape   independent | shared (default independent).
            \\                         shared gives a token's encoder taps ONE draw
            \\                         instead of one each. krea2's tap projector is a
            \\                         signed contrast between layer groups, so it
            \\                         amplifies noise that differs between taps and
            \\                         passes noise they share: independent ERODES the
            \\                         prompt, shared SHIFTS it
            \\      --cond-noise-keep-norm  rescale each perturbed vector back to the
            \\                         length it had (on/off, default off). Without it
            \\                         sigma 1 also makes the conditioning 41% louder
            \\      --cond-steer <text>  steer the conditioning TOWARD another
            \\                         prompt's direction. The direction is that
            \\                         text's conditioning minus the EMPTY prompt's,
            \\                         per encoder tap, normalized -- the subtraction
            \\                         is what makes it a direction rather than a
            \\                         second prompt
            \\      --cond-steer-scale   how far, in units of the conditioning's own
            \\                         RMS (so it reads like --cond-noise-amount).
            \\                         Negative steers AWAY. --cond-noise shapes it
            \\                         per tap when both are set
            \\      --cond-steer-mode    how the term reaches the conditioning:
            \\                         prompt (default; the direction FROM this
            \\                         render toward the phrase -- adds it and
            \\                         gives up the prompt), empty (the phrase
            \\                         against nothing, an attribute), orthogonal
            \\                         (the phrase with the prompt's own direction
            \\                         stripped out) or concat (APPEND the phrase's
            \\                         tokens; the only one that adds without
            \\                         trading the prompt away. Not on zimage,
            \\                         anima or sensenova)
            \\      --cond-noise-pooled  perturb SDXL's pooled vector too (on/off,
            \\                         default on). Off leaves the UNet's global
            \\                         style conditioning untouched
            \\      --backend B        compute backend: cpu | vulkan | zig-cuda
            \\                         | cuda; unnamed, zig-cuda with an NVIDIA
            \\                         device, else vulkan, else cpu. vulkan
            \\                         offloads encoder/DiT/VAE GEMMs to
            \\                         Vulkan; zig-cuda runs the whole
            \\                         pipeline on the pure-Zig hand-PTX CUDA
            \\                         backend; cuda runs it on NVIDIA's dlopen'd
            \\                         cuBLASLt/cuDNN kernels (both take an fp8 /
            \\                         int8 (convrot or tensorwise) / int4 convrot
            \\                         / w4a8 / dense bf16 --dit ckpt)
            \\      --vram-budget 0    GiB of device memory to use (0 = ask the
            \\                         driver); weights past it stream per step.
            \\                         A HARD ceiling: allocations past it are
            \\                         refused, not merely discouraged. Only
            \\                         weights stream, so a budget under what the
            \\                         activations alone need is refused (~4 GiB
            \\                         at 1.8MP) — render smaller to go lower.
            \\                         "min" is not a number but the floor: keep
            \\                         no weight resident past the set in use, so
            \\                         the footprint is the activations plus that
            \\                         set. Streams every weight every step
            \\      --te-gguf-gemm f16  which GEMM a GGUF block-quant TEXT ENCODER
            \\                         decodes its weights for (same values as
            \\                         --dit-gguf-gemm). f16 expands the weight and
            \\                         leaves the activation alone; int8/int4/mmq
            \\                         quantize the activation too, and a
            \\                         conditioning is computed once and steers
            \\                         every step, so that error is baked into the
            \\                         whole render. Exists to measure that trade
            \\      --dit-gguf-gemm auto  which GEMM a GGUF block-quant DiT
            \\                         decodes its weights for:
            \\                         auto | int8 | int4 | bf16 | f16 | mmq.
            \\                         int8 rotates and re-quantizes to convrot
            \\                         int8, which is ~2x faster but caps accuracy
            \\                         at int8's; int4 does the same one width down
            \\                         for the s4 tensor cores; bf16 expands the
            \\                         weight and keeps the format's own. auto picks
            \\                         int8 for q2_k/q4_k (whose own error already
            \\                         dominates the regrid) and bf16 for the rest.
            \\                         int4 is opt-in even for q2_k: it is W4A4,
            \\                         and the 4-bit activations cost more than the
            \\                         weight regrid saves. Only q2_k/q4_k/q8_0
            \\                         have an int8 or int4 decode
            \\      --dit-f32 off      run the krea2 DiT in full f32 instead of the
            \\                         f16 tensor-core path (slower, more exact)
            \\                         (on/off). VULKAN and krea2 ONLY: it sets
            \\                         models.dit_gpu.force_f32, so it is a no-op on
            \\                         the CUDA backends and on every other family
            \\      --dit <path>       diffusion checkpoint (fp8 / int8 convrot
            \\                         or tensorwise / int4 convrot /
            \\                         asym_w4a8_int8 / nvfp4 / dense
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
            \\      --te-act / --dit-act / --vae-act <auto|f32|f16|bf16>
            \\                         activation precision of the text encoder, the
            \\                         denoiser (--unet-act is the same) and the VAE.
            \\                         auto (default): integer weights run the format's
            \\                         own arithmetic (int8 W8A8, int4 W4A4, W4A8), GGUF
            \\                         its --dit-gguf-gemm route, and the stored
            \\                         activations start at the family's dtype and
            \\                         widen f16 -> f32 after a non-finite forward,
            \\                         remembered for the session. An explicit dtype
            \\                         is used everywhere, integer weights included
            \\                         (int8 at f16 is W8A16, at f32 W8A32), and fails
            \\                         instead of widening. A dtype a component's
            \\                         kernels on this backend do not take runs at f32
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
            \\      --ref-image <png>  reference picture for image editing (SenseNova
            \\                         and Mage-Flow-Edit); repeat for more than one.
            \\                         SenseNova presents them at the size given
            \\                         (32 px per token), so resize those before
            \\                         passing them; Mage-Flow resizes each twice
            \\                         itself, to 384 px for the vision tower and to
            \\                         the RENDER's size for the denoiser.
            \\                         Ignored by every other family
            \\      --out out.png      output file
            \\  TensorPencil inspect <file.safetensors>   list tensors in a checkpoint
            \\  TensorPencil bench-matmul                 time a DiT-sized fp8 GEMM
            \\  TensorPencil decode-latent <z.bin> <zh> <zw> <out.png>
            \\  Any *-test command takes --act f32|f16|bf16: the activation width its
            \\      device forwards store at (default: each check's own). A 16-bit run
            \\      reports its error against the CPU and fails only when non-finite.
            \\  TensorPencil sd-cuda-test [<sd1.5 ckpt>] [libs]
            \\      check the CUDA SD kernels against the CPU ops they reproduce
            \\      ("libs" runs the cuBLASLt/cuDNN arm, whose attention path
            \\      differs); exits non-zero if any check fails
            \\  TensorPencil te-test [<encoder>] [--ref <encoder>] [--krea2|--anima]
            \\      check a Qwen3 text encoder on every GPU backend against its
            \\      own CPU forward (kernel check), and optionally against a
            \\      second encoder (quantization check). Accepts .gguf
            \\  TensorPencil sensenova-cuda-test [<ckpt>] [libs] [--layers N] [--canvas H W]
            \\      [--prompt-tokens N] [--sigma S] [--lora F [--lora-bf16-a]]
            \\      [--te-int-gemm w8a8|weight_only] [--prefix-bench N]
            \\      check sensenova_cuda's PREFIX pass and generation forward
            \\      against the CPU ones on real weights, on both attention paths
            \\      and at a padded and an aligned canvas; non-zero if any fails
            \\  TensorPencil sensenova-vk-test [<ckpt>] [--layers N] [--canvas H W] [--prefix-bench N]
            \\      [--prompt-tokens N] [--sigma S]
            \\      the same checks against sensenova_gpu on Vulkan
            \\  TensorPencil lora-cuda-bench [libs] [--m ROWS] [--rank R[,R...]]
            \\      the LoRA sidecar on an int8 base through lin_cuda, the bf16-A
            \\      path against the int8 A: numerics, then an interleaved timing
            \\      table; non-zero if a numerics check fails
            \\  TensorPencil lora-sidecar-bench <ckpt> <lora> [--size N] [--rounds R] [--backend B] [--text-encoder P] [--vae P]
            \\      s/step with the LoRA off, the bf16-A sidecar and the int8-A one,
            \\      interleaved on one session, then one render each to scratch_out/
            \\  TensorPencil lora-restack-test <ckpt> <lora> [--backend cuda|zig-cuda|vulkan]
            \\      swap a LoRA stack on a live session: the image must come back
            \\      bit-identical and the factors' VRAM be handed back
            \\  TensorPencil lora-info <lora>...
            \\      load each file as a sidecar stack would: targets, ranks and the
            \\      resident bytes, rank padding included
            \\  TensorPencil mage-vae-cuda-test [<mage vae>] [libs] [--lat H W] [--bench N]
            \\  TensorPencil mage-vae-vk-test [<mage vae>] [--lat H W] [--bench N]
            \\        Mage-VAE's CUDA decode against its CPU decode, stage by
            \\        stage. --lat takes BOTH extents: the 32x32 attention window
            \\        pads a smaller latent UP and splits a larger one, so one
            \\        size answers about one of those
            \\  TensorPencil gemma-vit-test [<mmproj gguf>]
            \\        Gemma 3's vision tower on every GPU arm against the CPU one
            \\  TensorPencil gemma4-vit-test [<mmproj gguf>]
            \\        Gemma 4's vision tower (the 12B embedder or the 31B gemma4v, by the
            \\        file's projector type) the same way
            \\  TensorPencil vit35-test [<mmproj gguf>]
            \\        the Qwen3.5/3.6 vision tower the same way, with a hash per arm
            \\  TensorPencil taehv-test [<taew2_1>]
            \\        the live-preview decoder on every GPU arm against the CPU one,
            \\        with a hash per arm for comparing two builds
            \\  TensorPencil mageflow-bench [<mage ckpt>] [--backend cuda|zig-cuda|vulkan]
            \\                 [--lat N] [--cap N] [--reps N]
            \\      one Mage-Flow forward at real shapes: batched s/step best-of-N,
            \\      then one sync-per-op pass for the per-category split
            \\  TensorPencil mageflow-vk-test [<mage ckpt>]
            \\  TensorPencil moss-image-te-test [<qwen3.5 te>] [--backend zig-cuda|cuda|vulkan]
            \\      the Qwen3.5 text encoder on the device against the exact ComfyUI
            \\      reference in moss_image_te_ref: weight-only, and W8A8 by its noise
            \\  TensorPencil moss-image-cuda-test | moss-image-vk-test [<ckpt>] [libs] [--layers N] [--lat N] [--cap N] [--sigma S]
            \\      MOSS-Image's device forward against its CPU forward: weight-only to the
            \\      exact forward, W8A8 by the size of its activation noise
            \\  TensorPencil mageflow-cuda-test [<mage ckpt>] [libs]
            \\        [--layers N] [--lat N] [--cap N] [--refs N] [--sigma S]
            \\        Mage-Flow's CUDA forward against its CPU forward. The axes
            \\        are separate because a defect rides one: --layers is depth
            \\        (a per-block error accumulates), --lat the canvas token
            \\        count, --cap the TEXT half's length (the other half of the
            \\        joint sequence, and the one with its own projections),
            \\        --refs how many reference images the edit path appends, and
            \\        --sigma the activation magnitude
            \\  TensorPencil zimage-cuda-test [<zimage ckpt>] [libs]
            \\      check zimage_cuda's device forward against the CPU forward on
            \\      real weights, on both attention paths; non-zero if any fails
            \\  TensorPencil anima-cuda-test [<anima ckpt>] [libs]
            \\      check anima_cuda's new kernels (lnMod, rectangular TC
            \\      attention) against the CPU ops they reproduce, then its whole
            \\      device forward against anima.DiT.predict on real weights, on
            \\      both attention paths; non-zero if any fails
            \\  TensorPencil dual-cuda-test
            \\      the dual-target kernels (gpu/kernels/dual.zig, compiled to PTX)
            \\      against their hand-PTX twins and an f64 host reference, then
            \\      bandwidth of both; non-zero if any result is off
            \\  TensorPencil vk-norm-bench
            \\      thread-per-row vs subgroup weighted RMSNorm at every shape the
            \\      three Vulkan DiTs use, at the sizes they render at
            \\  TensorPencil vk-wo-bench [m]
            \\      Vulkan int8 linears at DiT shapes: W8A8, weight-only W8A16 and a
            \\      dense f16 weight on the same GEMM, interleaved, at m rows (4096)
            \\  TensorPencil vk-prep-bench [m]
            \\      the Vulkan int8 activation prep per width, fused against the
            \\      3-pass chain, interleaved, at m rows (default 4096), with GB/s
            \\  TensorPencil vk-gemv-bench
            \\      the Vulkan block-quant decode GEMV per format, in GB/s and
            \\      Gelem/s over the weight
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

    case: for (cases) |c| {
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
            ctx.opMatmulCoopI8(y_d, x_d, m, null, wb, n, k) catch |err| switch (err) {
                // A shape only the register kernel takes, on a device without it.
                error.UnsupportedDType => {
                    try stdout.print("i8 coop GEMM {d}x{d}x{d}: no kernel for this shape here, skipped\n", .{ m, n, k });
                    continue :case;
                },
                else => return err,
            };
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
        ctx.deviceName(),               ctx.cc_major,                          ctx.cc_minor,
        ctx.sm_count,                   @divTrunc(ctx.shared_optin_max, 1024), @divTrunc(ctx.shared_per_sm, 1024),
        @divTrunc(ctx.clock_khz, 1000),
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

        var plan = cudnn.MatmulDequantPlan.build(api, handle, c.m, c.n, c.k, .f32) catch |err| {
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

        var plan = cudnn.SdpaPlan.build(api, handle, 1, c.hq, c.hkv, c.s, c.s, c.d, cudnn.b.DATA_HALF) catch |err| {
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
        var plan = try cudnn.SdpaPlan.build(api, handle, 1, hq, hkv, t.s, t.s, d, cudnn.b.DATA_HALF);
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

/// CUDA int8 GEMM validation and timing, on the same CPU oracle as `gpu-i8-test`.
fn cudaI8Test(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    var ctx = cuda.Context.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("cuda device: {s} (sm_{d}{d})\n", .{ ctx.deviceName(), ctx.cc_major, ctx.cc_minor });
    try cuda.kernels.i8GemmTest(&ctx, stdout);
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
                try be.opGroupNorm(a_d, b_d, c_d, d_d, e_d, n, ch, groups, 256, 1e-5, silu);
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
            try be.opGroupNorm(a_d.as(.f16), b_d.as(.f16), c_d, d_d, e_d, n, ch, groups, 32, 1e-5, true);
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
            try be.opLayerNorm(a_d.as(.f16), b_d.as(.f16), wv, bv, rows, dim, 1e-5);
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
            try be.opGeglu(a_d.as(.f16), b_d.as(.f16), n, inner);
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
            try be.opAdd(a_d.as(.f16), b_d.as(.f16), n * ch);
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
            try be.opConcatCh(a_d.as(.f16), c_d.as(.f16), n, ch, total, 0);
            try be.opConcatCh(b_d.as(.f16), c_d.as(.f16), n, ch, total, ch);
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
            try be.opAddBiasRows(a_d.as(.f16), d_d, n, ch, off);
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
        try be.opGeglu(a_d, b_d, n, inner);
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

        for ([3]conv_cuda.SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
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
                .w = .{ .krsc = packed_w },
                .b = bias,
                .co = co,
                .ci = ci,
                .k = 3,
                .stride = if (mode == .stride2) @as(usize, 2) else 1,
                .pad = 1,
            };
            try be.ensureDeviceBuffer(&b_d, oh * ow * co * 4);
            try conv_cuda.conv2d(be, &patch, &b_d, &a_d, h, w, cv, mode, null, 1.0);
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

            // The same convolution with 16-bit activation STORAGE on both sides, which
            // is how the SD UNet and VAE run it. Both the cuDNN and the banded arm are
            // covered: `conv2d` picks by mode.
            inline for (.{ cuda.backend.Dt.f16, .bf16 }, .{ 5e-3, 2e-2 }) |dt, tol| {
                const bits = struct {
                    fn to(v: f32) u16 {
                        return if (dt == .f16) @bitCast(@as(f16, @floatCast(v))) else blk: {
                            const u: u32 = @bitCast(v);
                            break :blk @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
                        };
                    }
                    fn from(v: u16) f32 {
                        return if (dt == .f16) @as(f16, @bitCast(v)) else @bitCast(@as(u32, v) << 16);
                    }
                };
                const inh = try arena.alloc(u16, in.len);
                for (in, inh) |v, *o| o.* = bits.to(v);
                try be.ensureDeviceBuffer(&c_d, inh.len * 2);
                try be.tensorUpload(c_d, std.mem.sliceAsBytes(inh));
                try be.ensureDeviceBuffer(&d_d, oh * ow * co * 2);
                const src16 = c_d.as(dt);
                var dst16 = d_d.as(dt);
                try conv_cuda.conv2d(be, &patch, &dst16, &src16, h, w, cv, mode, null, 1.0);
                const oh16 = try arena.alloc(u16, oh * ow * co);
                try be.tensorDownload(d_d, std.mem.sliceAsBytes(oh16));
                const goth = try arena.alloc(f32, oh * ow * co);
                for (oh16, goth) |v, *o| o.* = bits.from(v);
                const nameh = try std.fmt.bufPrint(&name_buf, "conv 3x3 {t} {t} storage", .{ mode, dt });
                try rep.check(nameh, want, goth, tol);
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
            .krea2, .zimage, .anima, .minimax_h3, .sensenova, .mageflow, .moss_image, .qwen_image21 => &.{},
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
            // Both integer modes: on an int8 tower they differ by the activation
            // quantization alone, which is the isolation for whatever gap W8A8 shows.
            const lin_cuda = TensorPencil.models.lin_cuda;
            const saved = TensorPencil.models.dual_lin.encoder_int8;
            defer TensorPencil.models.dual_lin.encoder_int8 = saved;
            // Every encoder dial width on each integer mode.
            for ([_]cuda.backend.Dt{ .f32, .f16, .bf16 }) |act| for ([_]lin_cuda.IntGemm{ .weight_only, .w8a8 }) |mode| {
                TensorPencil.models.dual_lin.encoder_int8 = mode;
                if (cache_dev) |c| arena.free(c);
                cache_dev = null;
                var dev_run = run;
                dev_run.empty_cache = &cache_dev;
                try clip_text_cuda.encodePrompt(&enc, be, io, arena, got, &p, dev_run, act);

                var name_buf: [96]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "clip_text_cuda {s} {t} {t} (2 chunks, weighted)", .{ tw.name, act, mode });
                // f16 tensor-core GEMMs against an f32 CPU forward over 12/32 layers,
                // the same regime as the whole-UNet check below, and the per-kernel
                // checks above are what pin the arithmetic.
                try rep.check(name, want, got, if (act == .bf16) 5e-2 else 2e-2);
            };
        }
    }
    if (family == .sd15) {
        var unet = try sd_unet.UNet.load(arena, store, sd_unet.sd15, "model.diffusion_model.");
        defer unet.deinit();
        const ctx_seq = 77;
        const context = try arena.alloc(f32, ctx_seq * 768);
        for (context) |*v| v.* = rnd.floatNorm(f32) * 0.5;
        const t: f32 = 481;
        if (test_act) |a| unet.act = switch (a) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
        };
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
                checkTol(2e-2),
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
                const got = try sd_vae_cuda.decode(&dec, be, ref_gpa, z, lat_h, lat_w, null, test_act);
                defer ref_gpa.free(got);
                var name_buf: [80]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "sd_vae_cuda.decode ({d}x{d} latent, z x{d:.1})", .{ lat_h, lat_w, mag });
                try rep.check(name, want, got, checkTol(5e-3));
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
fn teTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, ref_path: []const u8, variant: TensorPencil.models.qwen3.Variant, dump: []const u8) !void {
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
    try dumpF32(io, arena, dump, "cpu", want);

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

    // The format question: the reference encoder (normally the dense source), against
    // which each arm below is also held, since that is what a render actually sees.
    const rwant: ?[]f32 = if (ref_path.len == 0) null else blk: {
        var rt = try TensorPencil.pipeline.Container.open(arena, io, ref_path);
        defer rt.deinit();
        var ref = try qwen3.TextEncoder.loadVariant(arena, rt.store(), variant);
        defer ref.deinit();
        const rw = try ref.encode(io, arena, ids.items, null);
        try stdout.print("\nvs {s}\n  ({t} weights)  cpu rel L2 {e:.4}  (the weights' own quantization)\n", .{
            ref_path, ref.layers[0].q.dtype, R.rel(rw, want),
        });
        break :blk rw;
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
                for (te_acts) |act| {
                    _ = try qwen3_cuda.encodeVision(&enc, b, io, arena, ids.items, vision, act, null); // warm
                    t0 = std.Io.Clock.real.now(io);
                    const got = try qwen3_cuda.encodeVision(&enc, b, io, arena, ids.items, vision, act, null);
                    ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                    if (act == .f32) try dumpF32(io, arena, dump, name, got);
                    failures += try teArm(stdout, R, name, act, ms, want, got, rwant, tol);
                }
            }
        } else {
            try stdout.print("{s:<9} unavailable\n", .{if (libs) "cuda" else "zig-cuda"});
        }
    }
    if (TensorPencil.gpu.Context.init(arena, io)) |gc| {
        defer gc.deinit();
        if (!qwen3_gpu.supportsWeights(gc, &enc)) {
            try stdout.print("{s:<9} REFUSED (supportsWeights false — would fall back to CPU)\n", .{"vulkan"});
            failures += 1;
        } else {
            for (te_acts) |act| {
                _ = try qwen3_gpu.encodeVision(&enc, gc, io, arena, ids.items, vision, act, null);
                t0 = std.Io.Clock.real.now(io);
                const got = try qwen3_gpu.encodeVision(&enc, gc, io, arena, ids.items, vision, act, null);
                ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                if (act == .f32) try dumpF32(io, arena, dump, "vulkan", got);
                failures += try teArm(stdout, R, "vulkan", act, ms, want, got, rwant, tol);
            }
        }
    } else |err| try stdout.print("{s:<9} unavailable ({t})\n", .{ "vulkan", err });

    try summarize(stdout, failures);
}

/// The encoder dial's widths `te-test` runs every arm at.
const te_acts = [_]TensorPencil.gpu.context.Dt{ .f32, .f16, .bf16 };

/// One `te-test` row: f32 against `tol`, a 16-bit width only for being finite (its
/// rel L2 is what a default is chosen from). Returns 1 on a failure.
fn teArm(stdout: *Io.Writer, comptime R: type, name: []const u8, act: TensorPencil.gpu.context.Dt, ms: f64, want: []const f32, got: []const f32, rwant: ?[]f32, tol: f64) !usize {
    const r = R.rel(want, got);
    const ok = R.finite(got) and (act != .f32 or r < tol);
    try stdout.print("{s:<9} {t:<4} {d:8.0} ms   rel L2 {e:.4}  {s}\n", .{ name, act, ms, r, if (ok) "ok" else if (R.finite(got)) "FAILED" else "NON-FINITE" });
    if (rwant) |rw| try stdout.print("{s:<14} vs ref   rel L2 {e:.4}  (what a render sees)\n", .{ "", R.rel(rw, got) });
    return @intFromBool(!ok);
}

/// `<prefix>.<arm>.f32`, raw little-endian f32, or nothing when `prefix` is empty.
fn dumpF32(io: Io, arena: std.mem.Allocator, prefix: []const u8, arm: []const u8, data: []const f32) !void {
    if (prefix.len == 0) return;
    const path = try std.fmt.allocPrint(arena, "{s}.{s}.f32", .{ prefix, arm });
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var buf: [1 << 16]u8 = undefined;
    var w = f.writer(io, &buf);
    try w.interface.writeAll(std.mem.sliceAsBytes(data));
    try w.interface.flush();
}

/// Check a SenseNova device arm against the CPU forward on real checkpoint weights:
/// the PREFIX pass (the conditioning, which is a whole second 8B stack) and then the
/// generation forward, at two canvas shapes.
///
/// A CLI command rather than a test for the reason every device check here is one:
/// the test binary brings up no device context. Truncated to a few layers by default
/// so it loads in seconds; `--layers` is a real axis, because a defect that
/// accumulates reads healthy at depth 2 and as noise at 42.
///
/// The two canvas shapes are not a cost knob either. One is a multiple of 32 and one
/// is not, so the pad-and-crop path is live in exactly one of them, and a case that
/// happened to be aligned would let a missing crop pass.
fn sensenovaCudaTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    libs: bool,
    layers: usize,
    toks: usize,
    sigma: f32,
    canvas: ?[2]usize,
    lora_path: []const u8,
    bench: usize,
) !void {
    const cuda = TensorPencil.gpu.cuda;
    const arm = TensorPencil.models.sensenova_cuda;

    var ld = (try sensenovaTestLoad(arena, io, stdout, ckpt, "sensenova-cuda-test", layers, lora_path)) orelse return;
    defer ld.deinit();
    try ld.attachLora(arena, stdout);

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== sensenova-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });
    if (bench > 0) return sensenovaPrefixBench(arm, be, arena, io, stdout, &ld.model, toks, bench, arm.supported(&ld.model));
    try sensenovaArmTest(arm, be, arena, io, stdout, &ld.model, toks, sigma, canvas, arm.supported(&ld.model));
}

/// The two prefix GEMM policies on one device in one process: each against the CPU
/// prefix, then `rounds` interleaved timings of the pass (A, B, A, B, ...), since this
/// card's clock drifts between runs and only a same-process alternation compares.
fn sensenovaPrefixBench(
    comptime arm: type,
    be: anytype,
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    model: *TensorPencil.models.sensenova.Model,
    toks: usize,
    rounds: usize,
    supported: bool,
) !void {
    const sensenova = TensorPencil.models.sensenova;
    const IntGemm = TensorPencil.models.dual_lin.IntGemm;
    const saved = TensorPencil.models.dual_lin.encoder_int8;
    defer TensorPencil.models.dual_lin.encoder_int8 = saved;
    if (!supported) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }
    const cfg = model.cfg;
    const ids = try arena.alloc(u32, toks);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 2731 + 11) % cfg.vocab);
    var layout = try sensenova.plainLayout(arena, ids);
    defer layout.deinit(arena);
    try stdout.print("trunk depth {d}, prompt {d} tokens\n", .{ cfg.n_layers, toks });

    var want = try model.prefixForward(io, arena, layout, &.{}, null);
    defer want.deinit(arena);
    const pols = [_]IntGemm{ .w8a8, .weight_only };
    for (pols) |pol| {
        TensorPencil.models.dual_lin.encoder_int8 = pol;
        // The first pass also uploads the stack; it is the accuracy row, not a timing.
        var got = try arm.prefixForward(model, be, io, arena, layout, &.{}, .f32, null);
        defer got.deinit(arena);
        try stdout.print("prefix KV vs CPU  {t:<12} rel L2 {e:.4}\n", .{ pol, relL2f(want.kv, got.kv) });
    }

    const ms = try arena.alloc(f64, 2 * rounds);
    for (0..rounds) |r| {
        for (pols, 0..) |pol, pi| {
            TensorPencil.models.dual_lin.encoder_int8 = pol;
            const t0 = std.Io.Clock.awake.now(io);
            var got = try arm.prefixForward(model, be, io, arena, layout, &.{}, .f32, null);
            const t1 = std.Io.Clock.awake.now(io);
            got.deinit(arena);
            ms[pi * rounds + r] = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
        }
        try stdout.print("round {d}: w8a8 {d:8.1} ms  weight_only {d:8.1} ms\n", .{ r, ms[r], ms[rounds + r] });
    }
    for (pols, 0..) |pol, pi| {
        const xs = ms[pi * rounds ..][0..rounds];
        std.mem.sort(f64, xs, {}, std.sort.asc(f64));
        try stdout.print("median {t:<12} {d:8.1} ms  (min {d:.1}, max {d:.1})\n", .{ pol, xs[rounds / 2], xs[0], xs[rounds - 1] });
    }
    if (arm == TensorPencil.models.sensenova_cuda)
        try stdout.print("device scratch after both: fp8_w16 {d} MB, fp8_a16 {d} MB\n", .{ be.fp8_w16.size >> 20, be.fp8_a16.size >> 20 })
    else
        try stdout.print("device scratch after both: deq_rows {d} MB, deq_f16k {d} MB\n", .{ be.deq_rows.size >> 20, be.deq_f16k.size >> 20 });
}

/// The Vulkan twin of `sensenovaCudaTest`, over `sensenova_gpu`.
fn sensenovaVkTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    layers: usize,
    toks: usize,
    sigma: f32,
    canvas: ?[2]usize,
    lora_path: []const u8,
    bench: usize,
) !void {
    const gpu_mod = TensorPencil.gpu.context;
    const arm = TensorPencil.models.sensenova_gpu;

    var ld = (try sensenovaTestLoad(arena, io, stdout, ckpt, "sensenova-vk-test", layers, lora_path)) orelse return;
    defer ld.deinit();
    try ld.attachLora(arena, stdout);

    const ctx = gpu_mod.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== sensenova-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});
    if (bench > 0) return sensenovaPrefixBench(arm, ctx, arena, io, stdout, &ld.model, toks, bench, arm.supported(ctx, &ld.model));
    try sensenovaArmTest(arm, ctx, arena, io, stdout, &ld.model, toks, sigma, canvas, arm.supported(ctx, &ld.model));
}

/// The checkpoint and the model that reads it. The container OWNS the mapping every
/// weight points into, so it must outlive the model, which is why it is handed back
/// rather than closed here.
const SenseNovaTestLoad = struct {
    ck: TensorPencil.pipeline.Container,
    model: TensorPencil.models.sensenova.Model,
    /// The LoRA file, which must outlive `stack` (its factors are views into the
    /// mapping), which must outlive `model`.
    lora_ck: ?TensorPencil.pipeline.Container = null,
    stack: ?TensorPencil.models.lora.Stack = null,
    lora_path: []const u8 = "",

    /// Point the model at the stack and report what attached. Called on the
    /// FINAL address, after `sensenovaTestLoad`'s return has moved this struct.
    ///
    /// A real axis for a device check: the host and device applies are separate
    /// code, so a whole-render PSNR cannot say which of the two moved, and
    /// `--layers` sweeps the depth that tells a precision floor from a defect
    /// that accumulates.
    fn attachLora(self: *SenseNovaTestLoad, gpa: std.mem.Allocator, stdout: *Io.Writer) !void {
        if (self.stack == null) return;
        const st = &self.stack.?;
        st.beginAttach();
        for (self.model.device_lins) |w| _ = try st.resolve(w);
        for (self.model.prefix_lins) |w| _ = try st.resolve(w);
        const n = try st.finishAttach(gpa);
        try TensorPencil.models.sensenova_cuda.quantizeLora(gpa, &self.model, st);
        self.model.lora = st;
        try stdout.print("lora: {s}\n  {d} sidecars over {d} layers, {d} MB\n", .{
            self.lora_path, n, self.model.cfg.n_layers, st.bytes() >> 20,
        });
    }

    fn deinit(self: *SenseNovaTestLoad) void {
        self.model.deinit();
        // The stack was built on the caller's arena, so its maps go with that;
        // only the mappings need closing here.
        self.ck.deinit();
        if (self.lora_ck) |*c| c.deinit();
    }
};

/// Open the checkpoint and load the trunk at `layers` depth, or print why not.
fn sensenovaTestLoad(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    cmd: []const u8,
    layers: usize,
    lora_path: []const u8,
) !?SenseNovaTestLoad {
    const sensenova = TensorPencil.models.sensenova;
    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("{s} needs a SenseNova checkpoint ({s})\n", .{ cmd, ckpt });
        return null;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    errdefer ck.deinit();
    var cfg = sensenova.u15_8b;
    if (layers != 0) cfg.n_layers = layers;
    var out: SenseNovaTestLoad = .{
        .ck = ck,
        .model = try sensenova.Model.load(arena, try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store()), cfg),
    };
    // Loaded here but ATTACHED by the caller: this struct is returned by value,
    // and the model holds a pointer to the stack, so attaching before the move
    // leaves it pointing at the old address.
    if (lora_path.len > 0) {
        out.lora_ck = try TensorPencil.pipeline.Container.open(arena, io, lora_path);
        var stack: TensorPencil.models.lora.Stack = .{};
        _ = try stack.add(arena, out.lora_ck.?.store(), 1.0, lora_path, .{});
        out.stack = stack;
        out.lora_path = lora_path;
    }
    return out;
}

/// A device check's CPU reference hook: each of `lins` has its activation quantized as
/// the device route for that weight quantizes it. Every other linear stays exact, since
/// the device runs it on the host or not at all.
fn ActEmu(comptime Dev: type, comptime of: anytype) type {
    const mm = TensorPencil.ops.matmul;
    return struct {
        dev: Dev,
        plan: TensorPencil.models.dual_lin.Plan,
        lins: []const mm.Weight,

        const Self = @This();

        fn hook(c: *const anyopaque, w: mm.Weight, m: usize) mm.ActQuant {
            const self: *const Self = @ptrCast(@alignCast(c));
            for (self.lins) |l| {
                if (l.bytes.ptr == w.bytes.ptr) return of(self.dev, self.plan, w, m);
            }
            return .{};
        }

        fn quantizes(self: *const Self) bool {
            for (self.lins) |w| if (of(self.dev, self.plan, w, 1).bits != 0) return true;
            return false;
        }

        /// The caller resets `mm.act_quant` to null when its reference is done.
        fn install(self: *const Self) void {
            mm.act_quant = .{ .ctx = self, .of = hook };
        }
    };
}

/// The checks themselves, over whichever arm `arm` is: the two differ only in what
/// they take as a device handle and in `supported`'s arity, which the callers above
/// have already resolved.
fn sensenovaArmTest(
    comptime arm: type,
    dev: anytype,
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    model: *TensorPencil.models.sensenova.Model,
    toks: usize,
    sigma: f32,
    canvas: ?[2]usize,
    supported: bool,
) !void {
    const sensenova = TensorPencil.models.sensenova;
    const sensenova_dev = TensorPencil.models.sensenova_dev;
    const cfg = model.cfg;
    // The prefix KV is held to the exact CPU forward, so the prompt pass runs weight-only
    // unless `--te-int-gemm` names the format's own arithmetic.
    const dual_lin = TensorPencil.models.dual_lin;
    if (dual_lin.encoder_int8 == null) dual_lin.encoder_int8 = .weight_only_f16;
    try stdout.print("trunk depth {d} of {d}, prompt {d} tokens, prefix gemm {t}\n", .{ cfg.n_layers, sensenova.u15_8b.n_layers, toks, dual_lin.encoderInt8() });
    if (!supported) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }
    // The Vulkan arm reads repacked f32.
    if (arm == TensorPencil.models.sensenova_gpu) try model.repackConvs();

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const ids = try arena.alloc(u32, toks);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 2731 + 11) % cfg.vocab);

    var failures: usize = 0;

    // --- the prefix pass -------------------------------------------------------
    var layout = try sensenova.plainLayout(arena, ids);
    defer layout.deinit(arena);
    var want_pre = try model.prefixForward(io, arena, layout, &.{}, null);
    defer want_pre.deinit(arena);
    var got_pre = try arm.prefixForward(model, dev, io, arena, layout, &.{}, .f32, null);
    defer got_pre.deinit(arena);
    {
        const rel = relL2f(want_pre.kv, got_pre.kv);
        const ok = rel < 5e-3 and std.math.isFinite(rel);
        if (!ok) failures += 1;
        try stdout.print("prefix KV vs CPU              rel L2 {e:.4}  {s}\n", .{ rel, if (ok) "ok" else "FAILED" });
    }
    // A 16-bit stream is opt-in, so it is reported against the CPU and fails only
    // when it overflows.
    for ([_]TensorPencil.gpu.context.Dt{ .f16, .bf16 }) |act| {
        var pre = arm.prefixForward(model, dev, io, arena, layout, &.{}, act, null) catch |err| {
            failures += 1;
            try stdout.print("prefix KV vs CPU ({t:<4} stream) {t}  FAILED\n", .{ act, err });
            continue;
        };
        defer pre.deinit(arena);
        const rel = relL2f(want_pre.kv, pre.kv);
        const ok = std.math.isFinite(rel);
        if (!ok) failures += 1;
        try stdout.print("prefix KV vs CPU ({t:<4} stream) rel L2 {e:.4}  {s}\n", .{ act, rel, if (ok) "ok" else "FAILED" });
    }

    // --- the generation forward ------------------------------------------------
    // A route that quantizes the activation is measured against a CPU reference that
    // quantizes it the same way; against the f32 one it would measure the format.
    const lin_mod = if (arm == TensorPencil.models.sensenova_gpu) TensorPencil.models.lin_gpu else TensorPencil.models.lin_cuda;
    const emu: ActEmu(@TypeOf(dev), lin_mod.actQuant) = .{ .dev = dev, .plan = try arm.canvasPlan(dev, model), .lins = model.device_lins };
    const quantized = emu.quantizes();

    // 64x64 is 2x2 tokens and aligned; 96x40 pads to 96x64 and crops back.
    const defaults = [_][2]usize{ .{ 64, 64 }, .{ 96, 40 } };
    const asked = [1][2]usize{canvas orelse .{ 0, 0 }};
    const shapes: []const [2]usize = if (canvas != null) &asked else &defaults;
    const t = 1.0 - sigma;
    const saved = arm.force_naive_attn;
    defer arm.force_naive_attn = saved;
    for (shapes) |hw| {
        const h = hw[0];
        const w = hw[1];
        const x = try arena.alloc(f32, 3 * h * w);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const exact = try arena.alloc(f32, x.len);
        try model.predict(io, arena, exact, x, h, w, want_pre, t, null);
        // An activation-quantized forward cannot be held to an emulated one element by
        // element (a rounding pattern reshuffles once its input moves by a fraction of
        // an int8 step), so as in `mageflow-*-test`: the walk weight-only against the
        // exact forward, and the activation noise against the format's own.
        // Linear in depth, fitted to a measured sweep of BOTH arms (1, 2, 4, 8, 16 and
        // 32 layers), within 1.6x of it everywhere past 4: the GEMMs run tensor cores
        // against the CPU's f32 accumulation, so the gap grows with the layer count.
        const tol: f64 = 2e-3 + 1e-3 * @as(f64, @floatFromInt(cfg.n_layers));
        const emulated: []f32 = if (!quantized) &.{} else blk: {
            const q = try arena.alloc(f32, x.len);
            emu.install();
            defer TensorPencil.ops.matmul.act_quant = null;
            try model.predict(io, arena, q, x, h, w, want_pre, t, null);
            try stdout.print("forward {d}x{d} emulated activation quant vs f32 rel L2 {e:.4}  (the format's own error)\n", .{ h, w, relL2f(exact, q) });
            break :blk q;
        };
        const weight_only: []f32 = if (!quantized) &.{} else blk: {
            arm.force_naive_attn = false;
            arm.force_weight_only = true;
            defer arm.force_weight_only = false;
            var sess = try arm.Session.init(arena, dev, model, h, w, want_pre);
            defer sess.deinit(dev);
            var ws = try arm.Workspace.init(dev, model, &sess, want_pre.seq, .f32);
            defer ws.deinit(dev);
            const wo = try arena.alloc(f32, x.len);
            try arm.forward(model, dev, &sess, &ws, io, arena, wo, x, t, null);
            const rel = relL2f(exact, wo);
            const ok = std.math.isFinite(rel) and rel < tol;
            if (!ok) failures += 1;
            try stdout.print("forward {d}x{d} (weight-only) rel L2 {e:.4}  {s}\n", .{ h, w, rel, if (ok) "ok" else "FAILED" });
            break :blk wo;
        };

        for ([_]bool{ false, true }) |naive| {
            arm.force_naive_attn = naive;
            var sess = try arm.Session.init(arena, dev, model, h, w, want_pre);
            defer sess.deinit(dev);
            var ws = try arm.Workspace.init(dev, model, &sess, want_pre.seq, .f32);
            defer ws.deinit(dev);
            const got = try arena.alloc(f32, x.len);
            try arm.forward(model, dev, &sess, &ws, io, arena, got, x, t, null);
            if (quantized) {
                const label = try std.fmt.allocPrint(arena, "forward {d}x{d} ({s})", .{ h, w, if (naive) "naive attn" else "tc attn" });
                try checkQuantNoise(stdout, &failures, label, exact, emulated, weight_only, got);
                continue;
            }

            var nonfinite: usize = 0;
            for (got) |v| {
                if (!std.math.isFinite(v)) nonfinite += 1;
            }
            const rel = relL2f(exact, got);
            // The control row. `rel` divides by the CPU velocity's own magnitude, and
            // that magnitude is not fixed: at a truncated depth on random input the
            // prediction can land near the input, which shrinks the velocity and
            // inflates the ratio without the absolute error moving. Printing both is
            // what tells a real divergence from a small denominator. The 64x64 canvas
            // runs ~25x the 96x40 case's error at the same depth on both arms alike,
            // with the same reference magnitude; unexplained, and not tuned around.
            var wsum: f64 = 0;
            for (exact) |v| wsum += @as(f64, v) * v;
            const wrms = @sqrt(wsum / @as(f64, @floatFromInt(exact.len)));
            const ok = nonfinite == 0 and rel < tol;
            if (!ok) failures += 1;
            try stdout.print("forward {d}x{d} ({s:<10}) rel L2 {e:.4}  (cpu v rms {d:.4})  {s}{s}\n", .{
                h,                          w,
                if (naive) "naive attn" else "tc attn", rel,
                wrms,                       if (ok) "ok" else "FAILED",
                if (nonfinite != 0) " (non-finite output)" else "",
            });
        }
        arm.force_naive_attn = false;
        for ([_]TensorPencil.gpu.context.Dt{ .f16, .bf16 }) |act| {
            var sess = try arm.Session.init(arena, dev, model, h, w, want_pre);
            defer sess.deinit(dev);
            var ws = try arm.Workspace.init(dev, model, &sess, want_pre.seq, sensenova_dev.streamDtFor(act));
            defer ws.deinit(dev);
            const got = try arena.alloc(f32, x.len);
            arm.forward(model, dev, &sess, &ws, io, arena, got, x, t, null) catch |err| {
                failures += 1;
                try stdout.print("forward {d}x{d} ({t:<4} stream) {t}  FAILED\n", .{ h, w, act, err });
                continue;
            };
            const rel = relL2f(exact, got);
            const ok = std.math.isFinite(rel);
            if (!ok) failures += 1;
            try stdout.print("forward {d}x{d} ({t:<4} stream) rel L2 {e:.4}  {s}\n", .{ h, w, act, rel, if (ok) "ok" else "FAILED" });
        }
    }
    try summarize(stdout, failures);
}

fn relL2f(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        num += (@as(f64, e) - a) * (@as(f64, e) - a);
        den += @as(f64, e) * e;
    }
    return if (den == 0) @sqrt(num) else @sqrt(num / den);
}

/// Check `zimage_cuda`'s device forward against `zimage.DiT.predict` on real
/// checkpoint weights, the CUDA twin of `zimage_gpu`'s gated Vulkan test, and a
/// CLI command for the reason `sd-cuda-test` and `cuda-dit-test` are: the test
/// binary brings up no CUDA context, so a `test` block here would only ever skip.
///
/// Truncated to a few trunk layers so it loads in seconds. The loop bound is not
/// what a kernel port gets wrong; the block's shape is. Exits non-zero on failure
/// so it works as a gate.
/// Mage-VAE's CUDA decode against its CPU decode, staged.
///
/// Staged rather than end to end because the halves fail for different reasons:
/// the CoD decoder is convolutions, GroupNorm and a WINDOWED attention with
/// replicate padding, while the per-pixel MLP is a gather, a per-ROW AdaLN and a
/// fold. One final comparison would say only "wrong".
fn mageVaeCudaTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    vae_path: []const u8,
    libs: bool,
    lat_h: usize,
    lat_w: usize,
    bench: usize,
) !void {
    const cuda = TensorPencil.gpu.cuda;
    const mage_vae = TensorPencil.models.mage_vae;
    const mage_vae_cuda = TensorPencil.models.mage_vae_cuda;

    std.Io.Dir.cwd().access(io, vae_path, .{}) catch {
        try stdout.print("mage-vae-cuda-test needs a Mage-VAE ({s})\n", .{vae_path});
        return;
    };
    var vt = try TensorPencil.SafeTensors.open(arena, io, vae_path);
    defer vt.deinit();
    var model = try mage_vae.MageVae.load(arena, .{ .safetensors = &vt }, "");
    defer model.deinit();

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== mage-vae-cuda-test ==\ncuda device: {s} (kernels: {t})\nlatent {d}x{d} -> {d}x{d} px\n", .{
        be.deviceName(), be.kernels, lat_h, lat_w, lat_h * 16, lat_w * 16,
    });

    try mageVaeArmTest(mage_vae_cuda, be, arena, io, stdout, &model, lat_h, lat_w, bench);
}

/// One Mage-Flow forward's device work at real shapes, on whichever backend is
/// named, so the two arms are measured by the same harness.
///
/// Two numbers, and they answer different questions. The BATCHED best-of-N is the
/// steady-state s/step a render actually pays. The sync-per-op pass that follows
/// serializes every op to attribute the time, so its total is an upper bound and
/// only the RATIOS between its rows mean anything.
/// The AUTOMATIC1111 `parameters` block for a CLI render, resources and all.
///
/// The paths come off `opts`, filtered by `img.sources` (a side path the checkpoint
/// overrode is not what ran); the weight
/// dtype is not asked for here because this runs after `generate` has returned
/// and the session is gone. A file hash is read from a `<path>.sha256` sidecar
/// when one exists and computed once otherwise.
fn renderParams(
    arena: std.mem.Allocator,
    io: Io,
    opts: *const TensorPencil.pipeline.Options,
    /// What the render resolved that `opts` does not hold: the shift (unless
    /// `--shift` was given, the family default ran) and the weight dtype (a
    /// property of what loaded).
    img: *const TensorPencil.pipeline.Image,
    w: usize,
    h: usize,
) ![]u8 {
    const base = try TensorPencil.pipeline.buildA1111Params(
        arena,
        opts.prompt,
        opts.negative,
        opts.steps,
        opts.cfg,
        opts.seed,
        w,
        h,
        std.fs.path.stem(opts.dit_path),
        img.family,
        opts.sampler,
        opts.scheduler,
        opts.prompt_syntax,
        opts.emphasis,
        opts.compat,
        opts.compatConfig(),
    );
    var hashes: TensorPencil.pipeline.HashCache = .{ .compute = opts.hash_models };
    defer hashes.deinit(arena);
    var loras: std.ArrayList(TensorPencil.pipeline.LoraRecord) = .empty;
    defer loras.deinit(arena);
    for (opts.loras) |l| try loras.append(arena, .{
        .name = std.fs.path.stem(l.path),
        .hash = hashes.autoV2(arena, io, l.path),
        .strength = l.strength,
    });
    const used = TensorPencil.pipeline.usedPath;
    // A VAE baked into the checkpoint gets no field, as in A1111.
    const vae_path = if (img.sources.get(.decoder) == .side) opts.vae_path else "";
    return TensorPencil.pipeline.appendExtraParams(arena, base, .{
        .clip1 = std.fs.path.stem(used(img.sources, .conditioner, opts.dit_path, opts.text_encoder_path)),
        .clip2 = std.fs.path.stem(used(img.sources, .conditioner2, opts.dit_path, opts.text_encoder_2_path)),
        .vae = std.fs.path.stem(vae_path),
        .model_hash = hashes.autoV2(arena, io, opts.dit_path),
        .vae_hash = hashes.autoV2(arena, io, vae_path),
        .shift = img.shift,
        .weight_dtype = img.weight_dtype,
        .act_dtype = if (img.acts) |a| try std.fmt.allocPrint(arena, "{f}", .{a}) else "",
        // Only when moved off ComfyUI's defaults, so an ordinary block is
        // unchanged and an overridden one re-renders.
        .eta = if (opts.eta != 1.0) opts.eta else null,
        .s_noise = if (opts.s_noise != 1.0) opts.s_noise else null,
        // The seed is recorded RESOLVED, because the field follows `--seed` when the
        // knob carries none and a reader has no way to derive that.
        .cond_noise = opts.cond_noise.curve,
        .cond_noise_amount = if (opts.cond_noise.on()) opts.cond_noise.amount else null,
        .cond_noise_seed = if (opts.cond_noise.on()) opts.cond_noise.seed orelse opts.seed else null,
        .cond_noise_negative = if (opts.cond_noise.on() and opts.cond_noise.neg_scale != 0) opts.cond_noise.neg_scale else null,
        .cond_steers = opts.cond_noise.steers,
        .loras = loras.items,
    });
}

/// Derive residual-stream steering directions by running two prompt sets and taking
/// the difference of their mean per-block activations.
///
/// File is `ACTD1` + blocks + features + `[blocks][features]` f32 unit vectors. Plain
/// because nothing here writes safetensors.
/// Takes a real allocator, never the process gpa: this runs one sampling pass per
/// prompt, so per-step buffers that never come back add up over a whole prompt set.
fn actDerive(gpa: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    var opts: TensorPencil.pipeline.Options = .{ .prompt = "" };
    opts.width = 512;
    opts.height = 512;
    opts.steps = 4;
    // The capture runs on whatever backend renders, so default to the fast one.
    opts.backend = .cuda;
    var set_a: []const u8 = "";
    var set_b: []const u8 = "";
    var out_path: []const u8 = "dirs.actd";

    var i: usize = 0;
    while (i + 1 < args.len) : (i += 2) {
        const flag = args[i];
        const val = args[i + 1];
        if (std.mem.eql(u8, flag, "--dit")) {
            opts.dit_path = val;
        } else if (std.mem.eql(u8, flag, "--text-encoder")) {
            opts.text_encoder_path = val;
            opts.explicit_text_encoder = true;
        } else if (std.mem.eql(u8, flag, "--vae")) {
            opts.vae_path = val;
            opts.explicit_vae = true;
        } else if (std.mem.eql(u8, flag, "--set-a")) {
            set_a = val;
        } else if (std.mem.eql(u8, flag, "--set-b")) {
            set_b = val;
        } else if (std.mem.eql(u8, flag, "--out")) {
            out_path = val;
        } else if (std.mem.eql(u8, flag, "--size")) {
            opts.width = try std.fmt.parseInt(usize, val, 10);
            opts.height = opts.width;
        } else if (std.mem.eql(u8, flag, "--steps")) {
            opts.steps = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, flag, "--backend")) {
            opts.backend = TensorPencil.pipeline.Backend.fromStr(val) orelse return error.InvalidArgs;
        } else {
            try stdout.print("unknown flag {s}\n", .{flag});
            return error.InvalidArgs;
        }
    }
    if (set_a.len == 0 or set_b.len == 0) {
        try stdout.print("act-derive needs --set-a and --set-b (prompts separated by ';')\n", .{});
        return error.InvalidArgs;
    }

    var sess = try TensorPencil.pipeline.Session.init(io, gpa, opts, stdout);
    defer sess.deinit();
    const shape = sess.actShape() orelse {
        try stdout.print("this family has no residual-stream hook\n", .{});
        return error.InvalidArgs;
    };

    var a_list: std.ArrayList([]const u8) = .empty;
    defer a_list.deinit(gpa);
    var b_list: std.ArrayList([]const u8) = .empty;
    defer b_list.deinit(gpa);
    for ([_][]const u8{ set_a, set_b }, [_]*std.ArrayList([]const u8){ &a_list, &b_list }) |set, list| {
        var it = std.mem.splitScalar(u8, set, ';');
        while (it.next()) |raw| {
            const p = std.mem.trim(u8, raw, " \t");
            if (p.len > 0) try list.append(gpa, p);
        }
    }

    const Note = struct {
        fn on(ctx: *anyopaque, prompt: []const u8, done: usize, total: usize) void {
            const w: *Io.Writer = @ptrCast(@alignCast(ctx));
            w.print("  [{d}/{d}] {s}\n", .{ done, total, prompt }) catch {};
            w.flush() catch {};
        }
    };
    const dirs = try TensorPencil.pipeline.ActDerive.run(sess, gpa, opts, a_list.items, b_list.items, .{
        .ctx = @ptrCast(stdout),
        .on = Note.on,
    });
    defer gpa.free(dirs);
    const bytes = try TensorPencil.pipeline.ActDerive.encodeFile(gpa, shape.blocks, shape.features, dirs);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = bytes });
    try stdout.print("wrote {s} ({d} blocks x {d})\n", .{ out_path, shape.blocks, shape.features });
    try stdout.flush();
}

fn mageflowBench(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    backend: []const u8,
    lat: usize,
    seq_txt: usize,
    reps: usize,
) !void {
    const mageflow = TensorPencil.models.mageflow;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("mageflow-bench needs a Mage-Flow checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    const store = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    var model = try mageflow.DiT.load(arena, store);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const cond = try arena.alloc(f32, seq_txt * mageflow.txt_dim);
    for (cond) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, mageflow.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);
    const out = try arena.alloc(f32, x_lat.len);
    const sigma: f32 = 0.75;

    try stdout.print("== mageflow-bench ==\nbackend {s}  latent {d}x{d} ({d}x{d} px)  text {d} rows\n", .{
        backend, lat, lat, lat * 16, lat * 16, seq_txt,
    });

    if (std.mem.eql(u8, backend, "vulkan")) {
        const gpu_mod = TensorPencil.gpu.context;
        const arm = TensorPencil.models.mageflow_gpu;
        const ctx = gpu_mod.Context.init(arena, io) catch |err| {
            try stdout.print("vulkan unavailable: {t}\n", .{err});
            return;
        };
        defer ctx.deinit();
        try stdout.print("vulkan device: {s}\n", .{ctx.deviceName()});
        if (!arm.supported(ctx, &model)) {
            try stdout.print("checkpoint dtype unsupported on this device\n", .{});
            return;
        }
        var sess = try arm.Session.init(arena, io, ctx, &model, lat, lat, cond, seq_txt, &.{}, &.{ sigma, 0 });
        defer sess.deinit(arena, ctx);
        var ws = try arm.Workspace.init(ctx, &model, seq_txt, sess.n_tok, .f32);
        defer ws.deinit(ctx);

        try arm.forward(&model, ctx, &sess, &ws, io, arena, out, x_lat, sigma, null);
        // Interleaved over the heads-per-batch knob, for the same clock-drift
        // reason the CUDA arm interleaves its bias A/B.
        var best: f64 = std.math.inf(f64);
        var best_unbatched: f64 = std.math.inf(f64);
        for (0..reps) |_| {
            for ([_]bool{ false, true }) |unb| {
                arm.force_unbatched = unb;
                const ta = std.Io.Clock.real.now(io);
                try arm.forward(&model, ctx, &sess, &ws, io, arena, out, x_lat, sigma, null);
                const tb = std.Io.Clock.real.now(io);
                const ms = @as(f64, @floatFromInt(tb.nanoseconds - ta.nanoseconds)) / 1e6;
                if (unb) best_unbatched = @min(best_unbatched, ms) else best = @min(best, ms);
            }
        }
        arm.force_unbatched = false;
        try stdout.print("batched forward: {d:.1} ms (best of {d}, one submission)\n", .{ best, reps });
        try stdout.print("                 {d:.1} ms (best of {d}, submit per op)  batching saves {d:.1}%\n", .{
            best_unbatched, reps, 100 * (best_unbatched - best) / best_unbatched,
        });

        arm.profile = true;
        arm.prof.reset();
        try arm.forward(&model, ctx, &sess, &ws, io, arena, out, x_lat, sigma, null);
        arm.profile = false;
        const p = arm.prof;
        const rows = [_]struct { name: []const u8, ns: i128 }{
            .{ .name = "gemm", .ns = p.gemm_ns },
            .{ .name = "gemm/txt", .ns = p.gemm_txt_ns },
            .{ .name = "attn", .ns = p.attn_ns },
            .{ .name = "norm", .ns = p.norm_ns },
            .{ .name = "elt", .ns = p.elt_ns },
            .{ .name = "copy", .ns = p.copy_ns },
            .{ .name = "xfer", .ns = p.xfer_ns },
            .{ .name = "cpu", .ns = p.cpu_ns },
        };
        const tot = p.totalNs();
        for (rows) |r| try printRow(stdout, r.name, r.ns, tot);
        try stdout.print("  {s:<7} {d:>8.1} ms  (sync-per-op sum, upper bound)\n", .{ "total", nsMs(tot) });
        return;
    }

    const cuda = TensorPencil.gpu.cuda;
    const arm = TensorPencil.models.mageflow_cuda;
    const libs = std.mem.eql(u8, backend, "cuda");
    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("cuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });
    if (!arm.supported(&model)) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }
    var sess = try arm.Session.init(arena, io, be, &model, lat, lat, cond, seq_txt, &.{}, &.{ sigma, 0 });
    defer sess.deinit(arena, be);
    var ws = try arm.Workspace.init(be, &model, seq_txt, sess.n_tok, .f32);
    defer ws.deinit(be);

    try arm.forward(&model, be, &sess, &ws, io, arena, out, x_lat, sigma, null);
    // INTERLEAVED A/B in one process. This card's clock drifts ~3% between runs,
    // which is larger than most of the changes worth making, so two separate
    // invocations cannot tell a 4% win from noise; alternating inside one run
    // makes the drift common-mode.
    var best: f64 = std.math.inf(f64);
    var best_split: f64 = std.math.inf(f64);
    for (0..reps) |_| {
        for ([_]bool{ false, true }) |split| {
            TensorPencil.models.lin_cuda.split_bias = split;
            const ta = std.Io.Clock.real.now(io);
            try arm.forward(&model, be, &sess, &ws, io, arena, out, x_lat, sigma, null);
            const tb = std.Io.Clock.real.now(io);
            const ms = @as(f64, @floatFromInt(tb.nanoseconds - ta.nanoseconds)) / 1e6;
            if (split) best_split = @min(best_split, ms) else best = @min(best, ms);
        }
    }
    TensorPencil.models.lin_cuda.split_bias = false;
    try stdout.print("batched forward: {d:.1} ms (best of {d}, bias folded)\n", .{ best, reps });
    try stdout.print("                 {d:.1} ms (best of {d}, bias in a separate pass)  fold saves {d:.1}%\n", .{
        best_split, reps, 100 * (best_split - best) / best_split,
    });

    be.profile = true;
    be.prof.reset();
    try arm.forward(&model, be, &sess, &ws, io, arena, out, x_lat, sigma, null);
    be.profile = false;
    var tot: f64 = 0;
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| tot += be.prof.ms[@intFromEnum(cat)];
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| {
        const i = @intFromEnum(cat);
        const ms = be.prof.ms[i];
        try stdout.print("  {s:<7} {d:>8.1} ms  {d:>5.1}%  ({d} launches)\n", .{
            @tagName(cat), ms, if (tot == 0) 0 else 100 * ms / tot, be.prof.n[i],
        });
    }
    try stdout.print("  {s:<7} {d:>8.1} ms  (sync-per-op sum, upper bound)\n", .{ "total", tot });
}

/// `diffusion-profile <ckpt> [--text-encoder P] [--text-encoder-2 P] [--vae P]
/// [--size N] [--reps N] [--lora P] [--backend B] [--dit-act D]
/// [--ab split-bias|act:D|perturb|prep3|convpatch|fullbar] [--dump P]`: one denoiser forward of any family,
/// through the same `Session` a render uses. The batched best-of-N is the steady-state
/// s/step; on CUDA the sync-per-op pass after it attributes the time, so its total is an
/// upper bound and only the ratios between rows mean anything. `--ab split-bias`
/// alternates each rep with a forward that adds every bias in a separate pass, an
/// interleaved A/B of the epilogue fold; `--ab act:D` with a second denoiser whose dial
/// is D; `--ab perturb` with the same denoiser on a latent nudged by 1e-5, the floor a
/// dtype A/B is read against (krea2 fp8 512²: 5.4e-4 on cuda, 7.2e-3 on Vulkan). `--dump`
/// writes both arms' velocities, raw f32, for comparing across backends.
fn diffusionProfile(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const cuda = TensorPencil.gpu.cuda;
    var opts: pipeline.Options = .{ .prompt = "a red fox sitting on a mossy rock in a misty pine forest, morning light", .dit_path = args[0], .backend = .cuda, .preview = false };
    var size: usize = 1024;
    var reps: usize = 5;
    var loras: std.ArrayList(pipeline.LoraSpec) = .empty;
    var ab_split = false;
    // A second arm of the same denoiser with one global switch flipped.
    var toggle: ?struct { flag: *bool, what: []const u8 } = null;
    var ab_act: ?pipeline.Act = null;
    var dump: ?[]const u8 = null;
    var ab_perturb = false;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--text-encoder")) {
            opts.text_encoder_path = v;
            opts.explicit_text_encoder = true;
        } else if (std.mem.eql(u8, args[i], "--text-encoder-2")) {
            opts.text_encoder_2_path = v;
            opts.explicit_text_encoder_2 = true;
        } else if (std.mem.eql(u8, args[i], "--vae")) {
            opts.vae_path = v;
            opts.explicit_vae = true;
        } else if (std.mem.eql(u8, args[i], "--size")) {
            size = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, args[i], "--reps")) {
            reps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, args[i], "--lora")) {
            try loras.append(arena, .{ .path = v });
        } else if (std.mem.eql(u8, args[i], "--backend")) {
            opts.backend = pipeline.Backend.fromStr(v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, args[i], "--dit-act")) {
            opts.dit_act = std.meta.stringToEnum(pipeline.Act, v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, args[i], "--dump")) {
            dump = v;
        } else if (std.mem.eql(u8, args[i], "--ab")) {
            if (std.mem.eql(u8, v, "split-bias")) {
                ab_split = true;
            } else if (std.mem.eql(u8, v, "prep3")) {
                // Vulkan's fused int8 prep against the 3-pass chain it replaces.
                toggle = .{ .flag = &TensorPencil.gpu.context.force_i8_prep_3pass, .what = "3-pass int8 prep" };
            } else if (std.mem.eql(u8, v, "fullbar")) {
                // Vulkan's hazard-tracked barriers against one after every dispatch.
                toggle = .{ .flag = &TensorPencil.gpu.context.Context.full_barriers, .what = "full barriers" };
            } else if (std.mem.eql(u8, v, "convpatch")) {
                // Vulkan's 3x3 conv gathered into the GEMM operand against the f32 patch.
                toggle = .{ .flag = &TensorPencil.gpu.dual_table.force_conv_patch, .what = "f32 conv patch" };
            } else if (std.mem.eql(u8, v, "perturb")) {
                // DIAGNOSTIC: the same denoiser on a latent nudged by 1e-5, the floor a
                // dtype A/B is read against.
                ab_perturb = true;
            } else if (std.mem.startsWith(u8, v, "act:")) {
                ab_act = std.meta.stringToEnum(pipeline.Act, v[4..]) orelse return error.InvalidArgs;
            } else return error.UnknownAbArm;
        }
    }
    opts.loras = loras.items;
    opts.width = size;
    opts.height = size;
    // A real allocator: a session's model memory must come back on teardown.
    const gpa = std.heap.smp_allocator;
    var sess = try pipeline.Session.init(io, gpa, opts, stdout);
    defer sess.deinit();
    if (ab_split and sess.cu_be == null) return error.NoCudaBackend;
    sess.setDials(opts.te_act, opts.dit_act, opts.vae_act);

    var cond = try sess.encode(gpa, opts.prompt, .{});
    defer cond.deinit(gpa);
    // Without it nothing pins, and a denoiser that does not fit beside the encoder
    // re-streams on every forward: the profile then times the transfer.
    sess.armDenoiserResidency();
    const sigmas = try sess.schedule(gpa, 8, opts.shift);
    defer gpa.free(sigmas);
    const ds = sess.spatialDownscale();
    const lat_h = size / ds;
    const lat_w = size / ds;
    const n = sess.latentChannels() * lat_h * lat_w;
    var den = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
    defer den.deinit(gpa);
    const act_a = sess.actFor(.denoiser);
    // The second arm's dtype is fixed when its denoiser is built, so both live at once.
    var den_b: ?pipeline.Denoiser = null;
    defer if (den_b) |*db| db.deinit(gpa);
    if (ab_act) |a| {
        sess.setDials(opts.te_act, a, opts.vae_act);
        den_b = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
    }
    const act_b = sess.actFor(.denoiser);
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(7);
    for (x) |*v| v.* = prng.random().floatNorm(f32);
    const v = try gpa.alloc(f32, n);
    defer gpa.free(v);
    const v_b = try gpa.alloc(f32, n);
    defer gpa.free(v_b);
    const x_b = try gpa.dupe(f32, x);
    defer gpa.free(x_b);
    if (ab_perturb) for (x_b) |*e| {
        e.* += 1e-5 * prng.random().floatNorm(f32);
    };

    defer if (toggle) |t| {
        t.flag.* = false;
    };
    // Two warm forwards: the first uploads every weight (and compiles the other arm's
    // kernels).
    for (0..2) |_| {
        try den.predict(gpa, v, x, sigmas[1], null);
        if (den_b) |*db| try db.predict(gpa, v_b, x, sigmas[1], null);
        if (toggle) |t| {
            t.flag.* = true;
            try den.predict(gpa, v_b, x, sigmas[1], null);
            t.flag.* = false;
        }
    }
    const lin_cuda = TensorPencil.models.lin_cuda;
    var best = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
    const arms: usize = if (ab_split or den_b != null or ab_perturb or toggle != null) 2 else 1;
    // The order flips every rep: whichever arm runs first measured 2-3% slower on a
    // 512² SD1.5 UNet whichever path it was.
    for (0..reps) |rep| {
        for (0..arms) |k| {
            const arm = if (rep % 2 == 0) k else arms - 1 - k;
            lin_cuda.split_bias = ab_split and arm == 1;
            if (toggle) |t| t.flag.* = arm == 1;
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            if (arm == 1 and den_b != null) try den_b.?.predict(gpa, v_b, x, sigmas[1], null) else if (arm == 1 and ab_perturb) try den.predict(gpa, v_b, x_b, sigmas[1], null) else if (arm == 1 and toggle != null) try den.predict(gpa, v_b, x, sigmas[1], null) else try den.predict(gpa, v, x, sigmas[1], null);
            best[arm] = @min(best[arm], @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6);
        }
    }
    lin_cuda.split_bias = false;
    if (toggle) |t| {
        t.flag.* = false;
        var se: f64 = 0;
        var ss: f64 = 0;
        for (v, v_b) |a, b| {
            se += (a - b) * (a - b);
            ss += a * a;
        }
        try stdout.print("  {s}: {d:.1} ms, the default path saves {d:.1}% (velocity rel L2 between them {e:.2})\n", .{ t.what, best[1], 100 * (best[1] - best[0]) / best[1], @sqrt(se / ss) });
    }
    try stdout.print("{t} {d}x{d}, weights {s}, denoiser {t}: batched forward {d:.1} ms (best of {d})\n", .{ sess.family(), size, size, sess.weightDtype(), act_a, best[0], reps });
    if (dump) |path| {
        // DIAGNOSTIC: both arms' velocities, raw f32, the second after the first.
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, std.mem.sliceAsBytes(v));
        if (den_b != null) try f.writeStreamingAll(io, std.mem.sliceAsBytes(v_b));
    }
    if (den_b != null or ab_perturb) {
        var se: f64 = 0;
        var ss: f64 = 0;
        for (v, v_b) |a, b| {
            se += (a - b) * (a - b);
            ss += a * a;
        }
        try stdout.print("  denoiser {t}: {d:.1} ms ({d:.1}% vs the first), velocity rel L2 vs {t} {e:.3}\n", .{ act_b, best[1], 100 * (best[1] - best[0]) / best[0], act_a, @sqrt(se / ss) });
    }
    if (ab_split) {
        try stdout.print("  bias in a separate pass: {d:.1} ms, the fold saves {d:.1}%\n", .{ best[1], 100 * (best[1] - best[0]) / best[1] });
        // The split arm is the unfused arithmetic, so an f32 output must match it bit
        // for bit; an f16 one rounds once fewer.
        const v_split = try gpa.alloc(f32, n);
        defer gpa.free(v_split);
        try den.predict(gpa, v, x, sigmas[1], null);
        lin_cuda.split_bias = true;
        try den.predict(gpa, v_split, x, sigmas[1], null);
        lin_cuda.split_bias = false;
        var differ: usize = 0;
        var max_d: f32 = 0;
        var max_v: f32 = 0;
        for (v, v_split) |a, b| {
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) differ += 1;
            max_d = @max(max_d, @abs(a - b));
            max_v = @max(max_v, @abs(b));
        }
        try stdout.print("  fold vs split output: {d} of {d} values differ, max |d| {e:.3} (max |v| {d:.3})\n", .{ differ, n, max_d, max_v });
    }

    const be = sess.cu_be orelse return stdout.flush();
    be.profile = true;
    be.prof.reset();
    try den.predict(gpa, v, x, sigmas[1], null);
    be.profile = false;
    var tot: f64 = 0;
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| tot += be.prof.ms[@intFromEnum(cat)];
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| {
        const ci = @intFromEnum(cat);
        const ms = be.prof.ms[ci];
        if (be.prof.n[ci] != 0) try stdout.print("  {s:<12} {d:>8.1} ms  {d:>5.1}%  ({d} launches)\n", .{ @tagName(cat), ms, if (tot == 0) 0 else 100 * ms / tot, be.prof.n[ci] });
    }
    try stdout.print("  {s:<12} {d:>8.1} ms  (sync-per-op sum, upper bound)\n", .{ "total", tot });
    try stdout.flush();
}

fn nsMs(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn printRow(out: *Io.Writer, name: []const u8, ns: i128, tot: i128) !void {
    const pct = if (tot == 0) 0 else 100 * nsMs(ns) / nsMs(tot);
    try out.print("  {s:<9} {d:>8.1} ms  {d:>5.1}%\n", .{ name, nsMs(ns), pct });
}

/// Mage-VAE's Vulkan decode against its CPU decode, staged like its CUDA twin
/// and for the same reason: the CoD decoder and the per-pixel MLP fail for
/// different reasons, and one final comparison would say only "wrong".
fn mageVaeVkTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    vae_path: []const u8,
    lat_h: usize,
    lat_w: usize,
    bench: usize,
) !void {
    const gpu_mod = TensorPencil.gpu.context;
    const mage_vae = TensorPencil.models.mage_vae;
    const mage_vae_gpu = TensorPencil.models.mage_vae_gpu;

    std.Io.Dir.cwd().access(io, vae_path, .{}) catch {
        try stdout.print("mage-vae-vk-test needs a Mage-VAE ({s})\n", .{vae_path});
        return;
    };
    var vt = try TensorPencil.SafeTensors.open(arena, io, vae_path);
    defer vt.deinit();
    var model = try mage_vae.MageVae.load(arena, .{ .safetensors = &vt }, "");
    defer model.deinit();
    const ctx = gpu_mod.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    if (TensorPencil.models.conv_gpu.needsRepack(ctx)) try model.repackConvs(arena);
    try stdout.print("== mage-vae-vk-test ==\nvulkan device: {s}\nlatent {d}x{d} -> {d}x{d} px\n", .{
        ctx.deviceName(), lat_h, lat_w, lat_h * 16, lat_w * 16,
    });

    try mageVaeArmTest(mage_vae_gpu, ctx, arena, io, stdout, &model, lat_h, lat_w, bench);
}

/// The Mage-VAE checks over whichever arm `arm` is. The CoD conditioning is checked
/// first: a windowing or padding mistake shows there, where the RGB only says "wrong".
/// `bench` > 0 skips the CPU and times that many device decodes instead.
fn mageVaeArmTest(
    comptime arm: type,
    dev: anytype,
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    model: *const TensorPencil.models.mage_vae.MageVae,
    lat_h: usize,
    lat_w: usize,
    bench: usize,
) !void {
    var prng = std.Random.DefaultPrng.init(91);
    const rnd = prng.random();
    const z = try arena.alloc(f32, lat_h * lat_w * TensorPencil.models.mage_vae.latent_channels);
    for (z) |*v| v.* = rnd.floatNorm(f32);

    const Dt = TensorPencil.gpu.context.Dt;
    if (bench > 0) {
        // Interleaved, the order flipped each round: this card's clock drifts.
        const acts = [_]Dt{ .f32, .f16 };
        const ms = try arena.alloc(f64, bench * acts.len);
        for (0..bench) |r| {
            for (0..acts.len) |j| {
                const ai = if (r % 2 == 0) j else acts.len - 1 - j;
                const t0 = std.Io.Clock.awake.now(io);
                const got = try arm.decode(model, dev, io, arena, z, lat_h, lat_w, acts[ai], null);
                ms[ai * bench + r] = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds)) / 1e6;
                arena.free(got);
            }
        }
        for (acts, 0..) |act, ai| {
            const xs = ms[ai * bench ..][0..bench];
            std.mem.sort(f64, xs, {}, std.sort.asc(f64));
            try stdout.print("decode {t:<4} median {d:.1} ms (min {d:.1}, max {d:.1}) over {d}\n", .{ act, xs[bench / 2], xs[0], xs[bench - 1], bench });
        }
        return;
    }

    var failures: usize = 0;
    const want_cod = try model.codForward(io, arena, z, lat_h, lat_w);
    const want = try model.decode(io, arena, z, lat_h, lat_w);
    // f32 is held to the f16 the convolution GEMM narrows its operands to, the floor
    // every VAE arm here sits at (measured 2.4e-4); a 16-bit row fails only when it
    // overflows, since it is opt-in.
    for ([_]Dt{ .f32, .f16, .bf16 }) |act| {
        const tol: f64 = if (act == .f32) 1e-3 else std.math.inf(f64);
        const got_cod = try arm.codForward(model, dev, arena, z, lat_h, lat_w, act);
        try checkRel(stdout, &failures, try std.fmt.allocPrint(arena, "cod conditioning {t}", .{act}), want_cod, got_cod, tol);
        const got = try arm.decode(model, dev, io, arena, z, lat_h, lat_w, act, null);
        try checkRel(stdout, &failures, try std.fmt.allocPrint(arena, "decode (whole) {t}", .{act}), want, got, tol);
    }
    try summarize(stdout, failures);
}

/// Mage-Flow's Vulkan forward against its CPU forward, on the real checkpoint.
/// Same five axes as `mageflow-cuda-test`, for the same reason.
fn mageflowVkTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    layers: usize,
    lat_arg: usize,
    cap_arg: usize,
    refs_arg: usize,
    sigma_arg: f32,
) !void {
    const gpu_mod = TensorPencil.gpu.context;
    const mageflow = TensorPencil.models.mageflow;
    const mageflow_gpu = TensorPencil.models.mageflow_gpu;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("mageflow-vk-test needs a Mage-Flow checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    const store = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    const depth = if (layers != 0) layers else mageflow.DiT.depthIn(store);
    var model = try mageflow.DiT.loadDepth(arena, store, depth);
    defer model.deinit();

    const ctx = gpu_mod.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== mageflow-vk-test ==\nvulkan device: {s}\ntrunk depth {d}\n", .{ ctx.deviceName(), depth });
    if (!mageflow_gpu.supported(ctx, &model)) {
        try stdout.print("checkpoint dtype unsupported on this device\n", .{});
        return;
    }

    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const lat = lat_arg;
    const seq_txt = cap_arg;
    const ctxv = try arena.alloc(f32, seq_txt * mageflow.txt_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, mageflow.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);

    // References at a DIFFERENT extent from the canvas: a reference is resized to
    // the render's size upstream, but the forward must not assume it.
    const refs = try arena.alloc(mageflow.Ref, refs_arg);
    for (refs, 0..) |*r, i| {
        const rh = lat - 1 + i % 2;
        const rw = lat + 1 - i % 2;
        const buf = try arena.alloc(f32, mageflow.channels * rh * rw);
        for (buf) |*v| v.* = rnd.floatNorm(f32);
        r.* = .{ .lat = buf, .h = rh, .w = rw };
    }
    if (refs_arg != 0) try stdout.print("edit path: {d} reference image(s)\n", .{refs_arg});

    const sigma: f32 = sigma_arg;
    const emu: ActEmu(*gpu_mod.Context, TensorPencil.models.lin_gpu.actQuant) = .{ .dev = ctx, .plan = try mageflow_gpu.linPlan(ctx, &model), .lins = model.device_lins };
    const ref = try mageflowRef(io, arena, stdout, &model, emu, x_lat, lat, sigma, ctxv, seq_txt, refs);
    // As `mageflow-cuda-test`: the shallow bound covers the whole --sigma range, where
    // the f16 GEMM's error against f32 accumulation is 4x what it is at the default.
    const tol = checkTol(if (depth <= 4) 1e-2 else 3e-2);

    var failures: usize = 0;
    const weight_only: []f32 = if (ref.emulated == null) &.{} else blk: {
        mageflow_gpu.force_weight_only = true;
        defer mageflow_gpu.force_weight_only = false;
        var sess = try mageflow_gpu.Session.init(arena, io, ctx, &model, lat, lat, ctxv, seq_txt, refs, &.{ sigma, 0 });
        defer sess.deinit(arena, ctx);
        var ws = try mageflow_gpu.Workspace.init(ctx, &model, seq_txt, sess.n_tok, TensorPencil.models.mageflow_dev.streamDtFor(testAct()));
        defer ws.deinit(ctx);
        const wo = try arena.alloc(f32, x_lat.len);
        try mageflow_gpu.forward(&model, ctx, &sess, &ws, io, arena, wo, x_lat, sigma, null);
        try checkRel(stdout, &failures, "forward (weight-only)", ref.exact, wo, tol);
        break :blk wo;
    };
    // BOTH attention paths, so the `attn_full` fallback cannot rot behind the
    // tensor-core one that actually runs.
    const saved = mageflow_gpu.force_attn_full;
    defer mageflow_gpu.force_attn_full = saved;
    for ([_]bool{ false, true }) |full| {
        mageflow_gpu.force_attn_full = full;
        var sess = try mageflow_gpu.Session.init(arena, io, ctx, &model, lat, lat, ctxv, seq_txt, refs, &.{ sigma, 0 });
        defer sess.deinit(arena, ctx);
        var ws = try mageflow_gpu.Workspace.init(ctx, &model, seq_txt, sess.n_tok, TensorPencil.models.mageflow_dev.streamDtFor(testAct()));
        defer ws.deinit(ctx);
        const got = try arena.alloc(f32, x_lat.len);
        try mageflow_gpu.forward(&model, ctx, &sess, &ws, io, arena, got, x_lat, sigma, null);
        const label = if (full) "forward (attn_full)" else "forward (tensor core)";
        if (ref.emulated == null)
            try checkRel(stdout, &failures, label, ref.exact, got, tol)
        else
            try checkQuantNoise(stdout, &failures, label, ref.exact, ref.emulated.?, weight_only, got);
    }
    try summarize(stdout, failures);
}

/// Relative L2 of `got` against `want`, printed with a pass/fail and a non-finite
/// count, for the device commands that stage several comparisons.
fn checkRel(out: *Io.Writer, fails: *usize, what: []const u8, want: []const f32, got: []const f32, tol: f64) !void {
    var num: f64 = 0;
    var den: f64 = 0;
    var nonfinite: usize = 0;
    for (want, got) |e, a| {
        if (!std.math.isFinite(a)) nonfinite += 1;
        num += (@as(f64, e) - a) * (@as(f64, e) - a);
        den += @as(f64, e) * e;
    }
    const rel = if (den == 0) 0 else @sqrt(num / den);
    const ok = nonfinite == 0 and rel < tol;
    if (!ok) fails.* += 1;
    try out.print("{s:<22} rel L2 {e:.4}  {s}{s}\n", .{
        what,                                               rel, if (ok) "ok" else "FAILED",
        if (nonfinite != 0) " (non-finite output)" else "",
    });
}

/// The CPU forwards a Mage-Flow device check compares against: exact, and when the
/// device quantizes activations, emulated as its routes quantize them.
const MageRef = struct { exact: []f32, emulated: ?[]f32 };

fn mageflowRef(
    io: Io,
    arena: std.mem.Allocator,
    stdout: *Io.Writer,
    model: *const TensorPencil.models.mageflow.DiT,
    emu: anytype,
    x_lat: []const f32,
    lat: usize,
    sigma: f32,
    ctxv: []const f32,
    seq_txt: usize,
    refs: []const TensorPencil.models.mageflow.Ref,
) !MageRef {
    const exact = try arena.alloc(f32, x_lat.len);
    try model.forward(io, arena, exact, x_lat, lat, lat, sigma, ctxv, seq_txt, refs, null);
    if (!emu.quantizes()) return .{ .exact = exact, .emulated = null };
    const q = try arena.alloc(f32, x_lat.len);
    emu.install();
    defer TensorPencil.ops.matmul.act_quant = null;
    try model.forward(io, arena, q, x_lat, lat, lat, sigma, ctxv, seq_txt, refs, null);
    try stdout.print("emulated activation quant vs f32  rel L2 {e:.4}  (the format's own error)\n", .{relL2f(exact, q)});
    return .{ .exact = exact, .emulated = q };
}

/// A W8A8 forward against its reference. It cannot be held to the emulated forward
/// element by element: a quantizer's rounding pattern is reshuffled once its input
/// moves by a fraction of an int8 step, which the device's own dense rounding does. So
/// the walk is checked weight-only, each route's quantizer against its emulation in
/// `lin-cuda-test` / `lin-vk-test`, and here the size of the activation noise
/// (`got - weight_only`) against the format's (`emulated - exact`).
fn checkQuantNoise(out: *Io.Writer, fails: *usize, what: []const u8, exact: []const f32, emulated: []const f32, weight_only: []const f32, got: []const f32) !void {
    var dev: f64 = 0;
    var fmt: f64 = 0;
    var den: f64 = 0;
    var nonfinite: usize = 0;
    for (exact, emulated, weight_only, got) |x, q, w, g| {
        if (!std.math.isFinite(g)) nonfinite += 1;
        dev += (@as(f64, g) - w) * (@as(f64, g) - w);
        fmt += (@as(f64, q) - x) * (@as(f64, q) - x);
        den += @as(f64, x) * x;
    }
    const ratio = @sqrt(dev / fmt);
    const ok = nonfinite == 0 and ratio > 1.0 / 1.5 and ratio < 1.5;
    if (!ok) fails.* += 1;
    try out.print("{s:<22} activation noise {e:.4}, format's {e:.4} (x{d:.2})  {s}{s}\n", .{
        what,                                               @sqrt(dev / den), @sqrt(fmt / den), ratio, if (ok) "ok" else "FAILED",
        if (nonfinite != 0) " (non-finite output)" else "",
    });
}

/// Mage-Flow's CUDA forward against its CPU forward, on the real checkpoint.
///
/// Five axes, because a defect rides one of them and a check pinned to one shape
/// answers about one shape: depth (an error that accumulates reads healthy at 2
/// blocks and garbage at 12), the canvas token count, the TEXT half's length
/// (which is the other half of the joint sequence and has its own projections,
/// its own modulation and its own QK-norms), how many reference images the edit
/// path appends, and the sigma the activations scale with.
fn mageflowCudaTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    libs: bool,
    layers: usize,
    lat_arg: usize,
    cap_arg: usize,
    refs_arg: usize,
    sigma_arg: f32,
    lora: TestLoraArg,
) !void {
    const cuda = TensorPencil.gpu.cuda;
    const mageflow = TensorPencil.models.mageflow;
    const mageflow_cuda = TensorPencil.models.mageflow_cuda;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("mageflow-cuda-test needs a Mage-Flow checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    const store = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    const depth = if (layers != 0) layers else mageflow.DiT.depthIn(store);
    var model = try mageflow.DiT.loadDepth(arena, store, depth);
    defer model.deinit();

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== mageflow-cuda-test ==\ncuda device: {s} (kernels: {t})\ntrunk depth {d}\n", .{ be.deviceName(), be.kernels, depth });
    if (!mageflow_cuda.supported(&model)) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }

    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const lat = lat_arg;
    const seq_txt = cap_arg;
    const ctxv = try arena.alloc(f32, seq_txt * mageflow.txt_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, mageflow.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);

    // References at a DIFFERENT extent from the canvas, because a reference is
    // resized to the render's size upstream but the forward must not assume it:
    // its own grid is what its RoPE positions are centered on.
    const refs = try arena.alloc(mageflow.Ref, refs_arg);
    for (refs, 0..) |*r, i| {
        const rh = lat - 1 + i % 2;
        const rw = lat + 1 - i % 2;
        const buf = try arena.alloc(f32, mageflow.channels * rh * rw);
        for (buf) |*v| v.* = rnd.floatNorm(f32);
        r.* = .{ .lat = buf, .h = rh, .w = rw };
    }
    if (refs_arg != 0) try stdout.print("edit path: {d} reference image(s)\n", .{refs_arg});

    const sigma: f32 = sigma_arg;
    const emu: ActEmu(*cuda.Backend, TensorPencil.models.lin_cuda.actQuant) = .{ .dev = be, .plan = try mageflow_cuda.linPlan(&model), .lins = model.device_lins };
    const ref = try mageflowRef(io, arena, stdout, &model, emu, x_lat, lat, sigma, ctxv, seq_txt, refs);
    const want = ref.exact;
    // Tensor cores against the CPU's f32 accumulation, the regime every other family's
    // CUDA check sits in, and the residual carries each block's rounding into the next
    // so the bound grows with depth. The shallow bound covers the whole --sigma range:
    // both device arms read 1.6e-3 at sigma 0.75 and 7.3e-3 at sigma 1.0, agreeing with
    // each other to four digits at every point.
    const tol = checkTol(if (depth <= 4) 1e-2 else 3e-2);

    var failures: usize = 0;
    const weight_only: []f32 = if (ref.emulated == null) &.{} else blk: {
        mageflow_cuda.force_weight_only = true;
        defer mageflow_cuda.force_weight_only = false;
        var sess = try mageflow_cuda.Session.init(arena, io, be, &model, lat, lat, ctxv, seq_txt, refs, &.{ sigma, 0 });
        defer sess.deinit(arena, be);
        var ws = try mageflow_cuda.Workspace.init(be, &model, seq_txt, sess.n_tok, TensorPencil.models.mageflow_dev.streamDtFor(testAct()));
        defer ws.deinit(be);
        const wo = try arena.alloc(f32, x_lat.len);
        try mageflow_cuda.forward(&model, be, &sess, &ws, io, arena, wo, x_lat, sigma, null);
        try checkRel(stdout, &failures, "forward (weight-only)", ref.exact, wo, tol);
        break :blk wo;
    };
    var got0: []f32 = &.{};
    // BOTH attention paths: `opAttnTC` is what runs, and the naive one is the
    // fallback the fast path was validated against. Checking only one leaves the
    // other free to rot.
    const saved = mageflow_cuda.force_naive_attn;
    defer mageflow_cuda.force_naive_attn = saved;
    for ([_]bool{ false, true }) |naive| {
        mageflow_cuda.force_naive_attn = naive;
        var sess = try mageflow_cuda.Session.init(arena, io, be, &model, lat, lat, ctxv, seq_txt, refs, &.{ sigma, 0 });
        defer sess.deinit(arena, be);
        var ws = try mageflow_cuda.Workspace.init(be, &model, seq_txt, sess.n_tok, TensorPencil.models.mageflow_dev.streamDtFor(testAct()));
        defer ws.deinit(be);
        const got = try arena.alloc(f32, x_lat.len);
        try mageflow_cuda.forward(&model, be, &sess, &ws, io, arena, got, x_lat, sigma, null);
        if (!naive) got0 = got;
        if (ref.emulated != null) {
            try checkQuantNoise(stdout, &failures, if (naive) "forward (naive attn)" else "forward (opAttnTC)", ref.exact, ref.emulated.?, weight_only, got);
            continue;
        }
        var num: f64 = 0;
        var den: f64 = 0;
        var nonfinite: usize = 0;
        for (want, got) |e, a| {
            if (!std.math.isFinite(a)) nonfinite += 1;
            num += (@as(f64, e) - a) * (@as(f64, e) - a);
            den += @as(f64, e) * e;
        }
        const rel = if (den == 0) 0 else @sqrt(num / den);
        const ok = nonfinite == 0 and rel < tol;
        if (!ok) failures += 1;
        try stdout.print("forward vs CPU ({s:<11})  rel L2 {e:.4}  {s}{s}\n", .{
            if (naive) "naive attn" else "opAttnTC", rel,
            if (ok) "ok" else "FAILED",              if (nonfinite != 0) " (non-finite output)" else "",
        });
    }
    mageflow_cuda.force_naive_attn = saved;
    if (try openTestLora(arena, io, stdout, lora.path, lora.strength, .{})) |st| {
        failures += try reportAttach(stdout, st, try model.attachLora(arena, st), layers != 0);
        try TensorPencil.models.lin_cuda.quantizeLora(arena, st, model.device_lins, TensorPencil.models.lin_cuda.blockq_gemm, .w8a8);
        const want_l = try arena.alloc(f32, x_lat.len);
        try model.forward(io, arena, want_l, x_lat, lat, lat, sigma, ctxv, seq_txt, refs, null);
        if (!mageflow_cuda.supported(&model)) return error.ValidationFailed;
        var sess = try mageflow_cuda.Session.init(arena, io, be, &model, lat, lat, ctxv, seq_txt, refs, &.{ sigma, 0 });
        defer sess.deinit(arena, be);
        var ws = try mageflow_cuda.Workspace.init(be, &model, seq_txt, sess.n_tok, TensorPencil.models.mageflow_dev.streamDtFor(testAct()));
        defer ws.deinit(be);
        const got_l = try arena.alloc(f32, x_lat.len);
        try mageflow_cuda.forward(&model, be, &sess, &ws, io, arena, got_l, x_lat, sigma, null);
        failures += try loraReport(stdout, want, want_l, got0, got_l, tol, true);
    }
    try summarize(stdout, failures);
}

/// MOSS-Image's device forward against its CPU forward, on the real checkpoint, over
/// depth, canvas tokens, text length and sigma. Weight-only is held to the exact
/// forward; W8A8 by the size of its activation noise against the format's.
fn mossImageTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    backend: enum { cuda, zig_cuda, vulkan },
    layers: usize,
    lat: usize,
    seq_txt: usize,
    sigma: f32,
) !void {
    const cuda = TensorPencil.gpu.cuda;
    const moss = TensorPencil.models.moss_image;
    const dev = TensorPencil.models.moss_image_dev;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("moss-image test needs a MOSS-Image checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    const store = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    var cfg = try moss.Config.detect(store);
    if (layers != 0) cfg.layers = @min(layers, cfg.layers);
    var model = try moss.DiT.loadWith(arena, store, cfg);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const ctxv = try arena.alloc(f32, seq_txt * cfg.txt_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, cfg.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);
    const exact = try arena.alloc(f32, x_lat.len);
    try model.forward(io, arena, exact, x_lat, lat, lat, sigma, ctxv, seq_txt, null);
    const tol = checkTol(if (cfg.layers <= 4) 1e-2 else 3e-2);
    var failures: usize = 0;

    if (backend == .vulkan) {
        const arm = TensorPencil.models.moss_image_gpu;
        const ctx = TensorPencil.gpu.context.Context.init(arena, io) catch |err| {
            try stdout.print("vulkan unavailable: {t}\n", .{err});
            return;
        };
        defer ctx.deinit();
        try stdout.print("== moss-image-vk-test ==\nvulkan device: {s}\ntrunk depth {d}, latent {d}x{d}, text {d}, sigma {d}\n", .{ ctx.deviceName(), cfg.layers, lat, lat, seq_txt, sigma });
        if (!arm.supported(ctx, &model)) {
            try stdout.print("checkpoint dtype unsupported on this device\n", .{});
            return;
        }
        const emu: ActEmu(*TensorPencil.gpu.context.Context, TensorPencil.models.lin_gpu.actQuant) = .{ .dev = ctx, .plan = try arm.linPlan(ctx, &model), .lins = model.device_lins };
        const run = struct {
            fn f(m: *const moss.DiT, c: *TensorPencil.gpu.context.Context, i: Io, a: std.mem.Allocator, l: usize, t: []const f32, st: usize, s: f32, x: []const f32) ![]f32 {
                var sess = try arm.Session.init(a, i, c, m, l, l, t, st, &.{ s, 0 });
                defer sess.deinit(a, c);
                var ws = try arm.Workspace.init(c, m, l * l, st, dev.streamDtFor(testAct()));
                defer ws.deinit(c);
                const got = try a.alloc(f32, x.len);
                try arm.forward(m, c, &sess, &ws, i, a, got, x, s, null);
                return got;
            }
        }.f;
        try mossImageChecks(arm, emu, stdout, &failures, &model, io, arena, ctx, run, lat, ctxv, seq_txt, sigma, x_lat, exact, tol);
        if (lat <= 16) {
            arm.force_attn_full = true;
            defer arm.force_attn_full = false;
            arm.force_weight_only = true;
            defer arm.force_weight_only = false;
            try checkRel(stdout, &failures, "forward (attn_full)", exact, try run(&model, ctx, io, arena, lat, ctxv, seq_txt, sigma, x_lat), tol);
        }
        return summarize(stdout, failures);
    }

    const arm = TensorPencil.models.moss_image_cuda;
    var be = (if (backend == .cuda) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== moss-image-cuda-test ==\ncuda device: {s} (kernels: {t})\ntrunk depth {d}, latent {d}x{d}, text {d}, sigma {d}\n", .{
        be.deviceName(), be.kernels, cfg.layers, lat, lat, seq_txt, sigma,
    });
    if (!arm.supported(&model)) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }
    const emu: ActEmu(*cuda.Backend, TensorPencil.models.lin_cuda.actQuant) = .{ .dev = be, .plan = try arm.linPlan(&model), .lins = model.device_lins };
    const run = struct {
        fn f(m: *const moss.DiT, b: *cuda.Backend, i: Io, a: std.mem.Allocator, l: usize, t: []const f32, st: usize, s: f32, x: []const f32) ![]f32 {
            var sess = try arm.Session.init(a, i, b, m, l, l, t, st, &.{ s, 0 });
            defer sess.deinit(a, b);
            var ws = try arm.Workspace.init(b, m, l * l, st, dev.streamDtFor(testAct()));
            defer ws.deinit(b);
            const got = try a.alloc(f32, x.len);
            try arm.forward(m, b, &sess, &ws, i, a, got, x, s, null);
            return got;
        }
    }.f;
    try mossImageChecks(arm, emu, stdout, &failures, &model, io, arena, be, run, lat, ctxv, seq_txt, sigma, x_lat, exact, tol);
    try summarize(stdout, failures);
}

/// Qwen-Image 2.1's device forward against its CPU forward, on the real checkpoint, over
/// depth, canvas tokens, text length and sigma. One session runs two sigmas, so the
/// second reads the prefix keys and values the first left behind.
fn qwenImage21DevTest(
    arena: std.mem.Allocator,
    io: Io,
    stdout: *Io.Writer,
    ckpt: []const u8,
    backend: enum { cuda, zig_cuda, vulkan },
    layers: usize,
    lat: usize,
    seq_txt: usize,
    sigma: f32,
) !void {
    const cuda = TensorPencil.gpu.cuda;
    const qi = TensorPencil.models.qwen_image21;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("qwen-image21 device test needs a Qwen-Image 2.1 checkpoint ({s})\n", .{ckpt});
        return;
    };
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    const store = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    var cfg = try qi.Config.detect(store);
    if (layers != 0) cfg.layers = @min(layers, cfg.layers);
    var model = try qi.DiT.loadWith(arena, store, cfg);
    defer model.deinit();

    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const ctxv = try arena.alloc(f32, seq_txt * cfg.txt_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, cfg.in_ch * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);
    const sigmas = [_]f32{ sigma, sigma * 0.5 };
    var exact: [2][]f32 = undefined;
    for (&exact, sigmas) |*e, s| {
        e.* = try arena.alloc(f32, cfg.out_ch * lat * lat);
        try model.forward(io, arena, e.*, x_lat, lat, lat, s, ctxv, seq_txt, &.{}, null);
    }
    const inp: Qi21In = .{ .model = &model, .lat = lat, .ctxv = ctxv, .seq_txt = seq_txt, .sigmas = &sigmas, .x_lat = x_lat, .exact = exact, .tol = checkTol(if (cfg.layers <= 4) 1e-2 else 3e-2) };
    var failures: usize = 0;

    if (backend == .vulkan) {
        const arm = TensorPencil.models.qwen_image21_gpu;
        const ctx = TensorPencil.gpu.context.Context.init(arena, io) catch |err| {
            try stdout.print("vulkan unavailable: {t}\n", .{err});
            return;
        };
        defer ctx.deinit();
        try stdout.print("== qwen-image21-vk-test ==\nvulkan device: {s}\ntrunk depth {d}, latent {d}x{d}, text {d}, sigma {d} then {d}\n", .{ ctx.deviceName(), cfg.layers, lat, lat, seq_txt, sigmas[0], sigmas[1] });
        if (!arm.supported(ctx, &model)) {
            try stdout.print("checkpoint dtype unsupported on this device\n", .{});
            return;
        }
        const emu: ActEmu(*TensorPencil.gpu.context.Context, TensorPencil.models.lin_gpu.actQuant) = .{ .dev = ctx, .plan = try arm.linPlan(ctx, &model), .lins = model.device_lins };
        try qwenImage21Checks(arm, ctx, emu, stdout, &failures, io, arena, inp);
        return summarize(stdout, failures);
    }

    const arm = TensorPencil.models.qwen_image21_cuda;
    var be = (if (backend == .cuda) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== qwen-image21-cuda-test ==\ncuda device: {s} (kernels: {t})\ntrunk depth {d}, latent {d}x{d}, text {d}, sigma {d} then {d}\n", .{
        be.deviceName(), be.kernels, cfg.layers, lat, lat, seq_txt, sigmas[0], sigmas[1],
    });
    if (!arm.supported(arena, &model)) {
        try stdout.print("checkpoint dtype unsupported on this backend\n", .{});
        return;
    }
    const emu: ActEmu(*cuda.Backend, TensorPencil.models.lin_cuda.actQuant) = .{ .dev = be, .plan = try arm.linPlan(arena, &model), .lins = model.device_lins };
    try qwenImage21Checks(arm, be, emu, stdout, &failures, io, arena, inp);
    try summarize(stdout, failures);
}

const Qi21In = struct {
    model: *const TensorPencil.models.qwen_image21.DiT,
    lat: usize,
    ctxv: []const f32,
    seq_txt: usize,
    sigmas: []const f32,
    x_lat: []const f32,
    exact: [2][]f32,
    tol: f64,
};

/// Both arms' checks: exact attention against the exact forward (the walk itself),
/// weight-only, then W8A8 by its noise when the checkpoint is integer.
fn qwenImage21Checks(comptime arm: type, device: anytype, emu: anytype, stdout: *Io.Writer, failures: *usize, io: Io, arena: std.mem.Allocator, in: Qi21In) !void {
    const dev = TensorPencil.models.qwen_image21_dev;
    const run = struct {
        fn f(m: *const TensorPencil.models.qwen_image21.DiT, b: @TypeOf(device), i: Io, a: std.mem.Allocator, l: usize, t: []const f32, st: usize, ss: []const f32, x: []const f32) ![2][]f32 {
            var sess = try arm.Session.init(a, i, b, m, l, l, t, st, &.{ ss[0], ss[1], 0 });
            defer sess.deinit(a, b);
            var ws = try arm.Workspace.init(b, m, l * l, sess.pre, dev.streamDtFor(testAct()));
            defer ws.deinit(b);
            var got: [2][]f32 = undefined;
            for (&got, ss) |*g, s| {
                g.* = try a.alloc(f32, m.cfg.out_ch * l * l);
                try arm.forward(m, b, &sess, &ws, i, a, g.*, x, s, null);
            }
            return got;
        }
    }.f;
    const names = [2][]const u8{ "first step (prefix pass)", "second step (cached prefix)" };
    if (in.lat <= 16 and emu.quantizes()) {
        // Exact attention and weight-only linears leave the walk itself, which must
        // hold to the CPU forward at f32 rounding. Integer weights only: a dense or
        // NVFP4 weight's GEMM stages its activation at 16 bits.
        arm.force_attn_f32 = true;
        arm.force_weight_only = true;
        defer arm.force_attn_f32 = false;
        defer arm.force_weight_only = false;
        const ex = try run(in.model, device, io, arena, in.lat, in.ctxv, in.seq_txt, in.sigmas, in.x_lat);
        for (names, in.exact, ex) |n, e, g| try checkRel(stdout, failures, try std.fmt.allocPrint(arena, "{s} (f32 attention)", .{n}), e, g, checkTol(1e-4));
    }
    if (!emu.quantizes()) {
        const got = try run(in.model, device, io, arena, in.lat, in.ctxv, in.seq_txt, in.sigmas, in.x_lat);
        for (names, in.exact, got) |n, e, g| try checkRel(stdout, failures, n, e, g, in.tol);
        return;
    }
    arm.force_weight_only = true;
    const wo = run(in.model, device, io, arena, in.lat, in.ctxv, in.seq_txt, in.sigmas, in.x_lat);
    arm.force_weight_only = false;
    const wog = try wo;
    for (names, in.exact, wog) |n, e, g| try checkRel(stdout, failures, try std.fmt.allocPrint(arena, "{s} (weight-only)", .{n}), e, g, in.tol);
    const q = try arena.alloc(f32, in.x_lat.len);
    {
        emu.install();
        defer TensorPencil.ops.matmul.act_quant = null;
        try in.model.forward(io, arena, q, in.x_lat, in.lat, in.lat, in.sigmas[0], in.ctxv, in.seq_txt, &.{}, null);
    }
    try stdout.print("emulated activation quant vs f32  rel L2 {e:.4}  (the format's own error)\n", .{relL2f(in.exact[0], q)});
    const got = try run(in.model, device, io, arena, in.lat, in.ctxv, in.seq_txt, in.sigmas, in.x_lat);
    try checkQuantNoise(stdout, failures, "first step (W8A8)", in.exact[0], q, wog[0], got[0]);
}

/// MOSS-Image's Qwen3.5 encoder on a device against the fixture's exact conditioning.
fn mossImageTeTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, te: []const u8, backend: []const u8) !void {
    const mm = TensorPencil.models;
    const dual_lin = mm.dual_lin;
    const ref_path = "src/models/assets/moss_image_te_ref.safetensors";
    std.Io.Dir.cwd().access(io, te, .{}) catch {
        try stdout.print("moss-image-te-test needs the Qwen3.5 text encoder ({s})\n", .{te});
        return;
    };
    var ref = try TensorPencil.safetensors.SafeTensors.open(arena, io, ref_path);
    defer ref.deinit();
    var ck = try TensorPencil.pipeline.Container.open(arena, io, te);
    defer ck.deinit();
    var enc = try mm.moss_image_text.Encoder.load(arena, ck.store());
    defer enc.deinit();
    const hidden = enc.hidden();

    const Arm = enum { cuda, libs, vulkan };
    const arm: Arm = if (std.mem.eql(u8, backend, "vulkan")) .vulkan else if (std.mem.eql(u8, backend, "cuda")) .libs else .cuda;
    var be: ?*TensorPencil.gpu.cuda.Backend = null;
    var ctx: ?*TensorPencil.gpu.context.Context = null;
    defer if (be) |b| b.deinit();
    defer if (ctx) |c| c.deinit();
    switch (arm) {
        .vulkan => ctx = TensorPencil.gpu.context.Context.init(arena, io) catch |err| {
            try stdout.print("vulkan unavailable: {t}\n", .{err});
            return;
        },
        .cuda, .libs => be = (if (arm == .libs) TensorPencil.gpu.cuda.Backend.initLibs(arena) else TensorPencil.gpu.cuda.Backend.init(arena)) catch |err| {
            try stdout.print("cuda unavailable: {t}\n", .{err});
            return;
        },
    }
    try stdout.print("== moss-image-te-test ==\n{s}: {s}\n", .{ backend, if (be) |b| b.deviceName() else ctx.?.deviceName() });

    const encodeOn = struct {
        fn f(a: std.mem.Allocator, b: ?*TensorPencil.gpu.cuda.Backend, c: ?*TensorPencil.gpu.context.Context, e: *const mm.moss_image_text.Encoder, ids: []const u32, mode: dual_lin.IntGemm) ![]f32 {
            const out = try a.alloc(f32, ids.len * e.hidden());
            const cap = TensorPencil.kv_cache.Capacity.fixed(ids.len);
            if (b) |bb| {
                var use = mm.lin_cuda.Use.llm(mm.qwen35_cuda.CudaLM.prefill_batch);
                use.int8 = mode;
                var lm = try mm.qwen35_cuda.CudaLM.initUse(a, bb, &e.lm, cap, use);
                defer lm.deinit();
                try lm.encode(ids, out);
            } else {
                var use = mm.lin_gpu.Use.llm(mm.qwen35_gpu.VulkanLM.prefill_batch);
                use.int8 = mode;
                var lm = try mm.qwen35_gpu.VulkanLM.initUse(a, c.?, &e.lm, cap, use);
                defer lm.deinit();
                try lm.encode(ids, out);
            }
            return out;
        }
    }.f;

    var failures: usize = 0;
    for (0..2) |p| {
        var kb: [16]u8 = undefined;
        const tv = try ref.require(try std.fmt.bufPrint(&kb, "tokens.{d}", .{p}));
        const raw = std.mem.bytesAsSlice(i32, tv.bytes);
        const ids = try arena.alloc(u32, raw.len);
        for (ids, raw) |*d, s| d.* = @intCast(s);
        const want = try (try ref.require(try std.fmt.bufPrint(&kb, "cond.{d}", .{p}))).toF32Alloc(arena);
        std.debug.assert(want.len == ids.len * hidden);
        try stdout.print("prompt {d}: {d} tokens\n", .{ p, ids.len });

        const wo = try encodeOn(arena, be, ctx, &enc, ids, .weight_only_f32);
        try checkRel(stdout, &failures, "  weight-only f32", want, wo, checkTol(2e-3));
        const w8 = try encodeOn(arena, be, ctx, &enc, ids, .w8a8);
        // W8A8 is the format's own arithmetic: held by the size of its noise against
        // the CPU forward with each linear's activation quantized as the device does.
        const lins = try enc.lm.deviceLins(arena);
        // The hook holds a pointer to `emu`, so each encode runs inside its branch.
        defer TensorPencil.ops.matmul.act_quant = null;
        const emulated = if (be) |b| blk: {
            var use = mm.lin_cuda.Use.llm(mm.qwen35_cuda.CudaLM.prefill_batch);
            use.int8 = .w8a8;
            const emu: ActEmu(*TensorPencil.gpu.cuda.Backend, mm.lin_cuda.actQuant) = .{ .dev = b, .plan = try mm.lin_cuda.plan(lins, use, "te-test"), .lins = lins };
            emu.install();
            break :blk try enc.encodeIds(io, arena, ids);
        } else blk: {
            var use = mm.lin_gpu.Use.llm(mm.qwen35_gpu.VulkanLM.prefill_batch);
            use.int8 = .w8a8;
            const emu: ActEmu(*TensorPencil.gpu.context.Context, mm.lin_gpu.actQuant) = .{ .dev = ctx.?, .plan = try mm.lin_gpu.plan(ctx.?, lins, use, "te-test"), .lins = lins };
            emu.install();
            break :blk try enc.encodeIds(io, arena, ids);
        };
        TensorPencil.ops.matmul.act_quant = null;
        try stdout.print("  emulated W8A8 vs exact rel L2 {e:.4}  (the format's own error)\n", .{relL2f(want, emulated)});
        try checkQuantNoise(stdout, &failures, "  w8a8", want, emulated, wo, w8);
    }
    try summarize(stdout, failures);
}

/// One MOSS-Image CUDA forward, timed warm and then split by the backend's profiler.
fn mossImageCudaProf(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool, lat: usize, seq_txt: usize) !void {
    const cuda = TensorPencil.gpu.cuda;
    const moss = TensorPencil.models.moss_image;
    const arm = TensorPencil.models.moss_image_cuda;
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    var model = try moss.DiT.load(arena, try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store()));
    defer model.deinit();
    var be = try (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena));
    defer be.deinit();
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const ctxv = try arena.alloc(f32, seq_txt * model.cfg.txt_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x = try arena.alloc(f32, model.cfg.channels * lat * lat);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const out = try arena.alloc(f32, x.len);
    const sigma: f32 = 0.75;
    var sess = try arm.Session.init(arena, io, be, &model, lat, lat, ctxv, seq_txt, &.{ sigma, 0 });
    defer sess.deinit(arena, be);
    var ws = try arm.Workspace.init(be, &model, lat * lat, seq_txt, .f32);
    defer ws.deinit(be);
    try arm.forward(&model, be, &sess, &ws, io, arena, out, x, sigma, null);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const ta = std.Io.Clock.real.now(io);
        try arm.forward(&model, be, &sess, &ws, io, arena, out, x, sigma, null);
        best = @min(best, @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - ta.nanoseconds)) / 1e6);
    }
    try stdout.print("== moss-image cuda ({s}) {d}x{d} latent, text {d} ==\nforward {d:.1} ms (best of 3)\n", .{ if (libs) "libs" else "hand_ptx", lat, lat, seq_txt, best });
    be.profile = true;
    be.prof.reset();
    try arm.forward(&model, be, &sess, &ws, io, arena, out, x, sigma, null);
    be.profile = false;
    var tot: f64 = 0;
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| tot += be.prof.ms[@intFromEnum(cat)];
    inline for (comptime std.enums.values(cuda.Backend.ProfCat)) |cat| {
        const i = @intFromEnum(cat);
        try stdout.print("  {s:<7} {d:>8.1} ms  {d:>5.1}%  ({d} launches)\n", .{ @tagName(cat), be.prof.ms[i], if (tot == 0) 0 else 100 * be.prof.ms[i] / tot, be.prof.n[i] });
    }
}

/// The two checks both arms run: weight-only against the exact forward, and the
/// format's W8A8 by its noise, when the device quantizes activations at all.
fn mossImageChecks(
    comptime arm: type,
    emu: anytype,
    stdout: *Io.Writer,
    failures: *usize,
    model: *const TensorPencil.models.moss_image.DiT,
    io: Io,
    arena: std.mem.Allocator,
    device: anytype,
    run: anytype,
    lat: usize,
    ctxv: []const f32,
    seq_txt: usize,
    sigma: f32,
    x_lat: []const f32,
    exact: []const f32,
    tol: f64,
) !void {
    if (!emu.quantizes()) {
        try checkRel(stdout, failures, "forward", exact, try run(model, device, io, arena, lat, ctxv, seq_txt, sigma, x_lat), tol);
        return;
    }
    const q = try arena.alloc(f32, x_lat.len);
    {
        emu.install();
        defer TensorPencil.ops.matmul.act_quant = null;
        try model.forward(io, arena, q, x_lat, lat, lat, sigma, ctxv, seq_txt, null);
    }
    const fmt = relL2f(exact, q);
    try stdout.print("emulated activation quant vs f32  rel L2 {e:.4}  (the format's own error)\n", .{fmt});
    arm.force_weight_only = true;
    const wo = run(model, device, io, arena, lat, ctxv, seq_txt, sigma, x_lat);
    arm.force_weight_only = false;
    try checkRel(stdout, failures, "forward (weight-only)", exact, try wo, tol);
    const got = try run(model, device, io, arena, lat, ctxv, seq_txt, sigma, x_lat);
    // Past ~5% the trunk amplifies a quantizer's noise chaotically (42 blocks at sigma
    // 1 reach 17% where 12 blocks hold 7e-4), and two draws of it differ in size by
    // half on BOTH arms: the ratio then measures the draw, not the route.
    var noise_fail: usize = 0;
    try checkQuantNoise(stdout, &noise_fail, "forward (W8A8)", exact, q, try wo, got);
    if (fmt < 5e-2) failures.* += noise_fail else if (noise_fail != 0)
        try stdout.print("  (not gated: the format's own error is past 5%, where the ratio is the draw's)\n", .{});
}

/// `--lora <file> [--strength s]` on a device check.
const TestLoraArg = struct { path: []const u8 = "", strength: f32 = 1 };

fn zimageCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, vae_path: []const u8, libs: bool, layers: usize, lat_arg: usize, cap_arg: usize, sigma_arg: f32, lora: TestLoraArg) !void {
    const cuda = TensorPencil.gpu.cuda;
    const zimage = TensorPencil.models.zimage;
    const zimage_cuda = TensorPencil.models.zimage_cuda;
    const safetensors = TensorPencil.SafeTensors;

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("zimage-cuda-test needs a Z-Image checkpoint ({s})\n", .{ckpt});
        return;
    };
    // Container.open, not SafeTensors.open: a GGUF Z-Image is a real checkpoint here.
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
    defer ck.deinit();
    var cfg = zimage.z_image;
    // Depth is a real axis here, not a cost knob: a defect that accumulates rather than
    // being wrong at once reads as a healthy residual at depth 2 and garbage at 30, which
    // is exactly what a solid white render off a q8_0 GGUF turned out to be. The default
    // is the full trunk; `--layers N` picks one depth.
    if (layers != 0) cfg.n_layers = layers;
    var model = try zimage.DiT.load(arena, try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store()), cfg);
    defer model.deinit();
    try stdout.print("trunk depth {d} (+{d}+{d} refiners)\n", .{ cfg.n_layers, cfg.n_refiner_layers, cfg.n_refiner_layers });

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

    // 16 is 8x8 = 64 image tokens, exactly two pad buckets. `--lat` raises it: the token
    // count is what the padding rules and every GEMM's `m` key on, so a check at one
    // latent size answers about one `m`.
    const lat = lat_arg;
    // The caption length is the other half of the joint sequence, and it is padded on its
    // own (`cfg.padded`), so it is its own shape axis. `--cap` raises it: a real prompt is
    // far longer than 20 tokens.
    const seq_txt = cap_arg;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const ctxv = try arena.alloc(f32, seq_txt * cfg.cap_dim);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try arena.alloc(f32, cfg.channels * lat * lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);

    // The sigma is a real axis too, not a fixed detail: the trunk's activations scale
    // with it, and a device buffer that only overflows at a magnitude is fine at 0.75 and
    // all-NaN at the 1.0 a render's first step actually uses.
    const sigma: f32 = sigma_arg;
    const cap = try model.capTokens(io, arena, ctxv, seq_txt);
    const cap_padded = cfg.padded(seq_txt);

    const want = try arena.alloc(f32, x_lat.len);
    {
        const adaln = try model.adalnInput(io, arena, sigma);
        try model.predict(io, arena, want, x_lat, lat, lat, cap, cap_padded, adaln, null);
    }

    var failures: usize = 0;
    var got0: []f32 = &.{};
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
        var ws = try zimage_cuda.Workspace.init(be, &model, lat, lat, cap_padded, TensorPencil.models.zimage_dev.streamDtFor(testAct()));
        defer ws.deinit(be);
        const got = try arena.alloc(f32, x_lat.len);
        try zimage_cuda.forward(&model, be, &sess, &ws, io, arena, got, x_lat, sigma, null);
        if (!naive) got0 = got;

        var num: f64 = 0;
        var den: f64 = 0;
        var nonfinite: usize = 0;
        for (want, got) |e, a| {
            if (!std.math.isFinite(a)) nonfinite += 1;
            num += (@as(f64, e) - a) * (@as(f64, e) - a);
            den += @as(f64, e) * e;
        }
        const rel = if (den == 0) 0 else @sqrt(num / den);
        // The GEMMs run tensor cores against the CPU's f32 accumulation, the same regime
        // `sd_unet_cuda` sits in. A dense bf16 checkpoint measures 2.2e-4 at the 2 layers
        // this ran at before it could vary depth; the bound grows with depth because the
        // residual carries each block's rounding into the next.
        //
        // The growth is MEASURED, not chosen to make deep runs pass: a q8_0 GGUF reads
        // 2.3e-4 flat from depth 2 to 24 and 1.4e-2 at 30, and the same file dequantized
        // to dense bf16 at load reads 1.1e-2 at 30, i.e. the deep figure is the bf16
        // operand floor and not the quantization. For scale, krea2's equivalent check
        // passes at 0.18. It still has teeth: the f16 overflow this test was extended to
        // catch lands at non-finite, and a mis-wired route lands near 1.0.
        const tol = checkTol(if (model.layers.len <= 8) 1e-3 else 5e-2);
        const ok = nonfinite == 0 and rel < tol;
        if (!ok) failures += 1;
        try stdout.print("forward vs CPU ({s:<11})  rel L2 {e:.4}  {s}{s}\n", .{
            if (naive) "naive attn" else "opAttnTC", rel,
            if (ok) "ok" else "FAILED",              if (nonfinite != 0) " (non-finite output)" else "",
        });
    }
    zimage_cuda.force_naive_attn = saved;
    if (try openTestLora(arena, io, stdout, lora.path, lora.strength, .{ .renames = &zimage.lora_keys })) |st| {
        failures += try reportAttach(stdout, st, try model.attachLora(arena, st), layers != 0);
        try TensorPencil.models.lin_cuda.quantizeLora(arena, st, model.device_lins, TensorPencil.models.lin_cuda.blockq_gemm, .w8a8);
        // The caption half and the AdaLN linears are patched too, so both are rebuilt.
        const cap_l = try model.capTokens(io, arena, ctxv, seq_txt);
        const want_l = try arena.alloc(f32, x_lat.len);
        try model.predict(io, arena, want_l, x_lat, lat, lat, cap_l, cap_padded, try model.adalnInput(io, arena, sigma), null);
        if (!zimage_cuda.supported(&model)) return error.ValidationFailed;
        var sess = try zimage_cuda.Session.init(arena, io, be, &model, lat, lat, cap_l, cap_padded, &.{ sigma, 0 });
        defer sess.deinit(arena, be);
        var ws = try zimage_cuda.Workspace.init(be, &model, lat, lat, cap_padded, TensorPencil.models.zimage_dev.streamDtFor(testAct()));
        defer ws.deinit(be);
        const got_l = try arena.alloc(f32, x_lat.len);
        try zimage_cuda.forward(&model, be, &sess, &ws, io, arena, got_l, x_lat, sigma, null);
        failures += try loraReport(stdout, want, want_l, got0, got_l, if (model.layers.len <= 8) 1e-3 else 5e-2, true);
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
                const got = try TensorPencil.models.sd_vae_cuda.decode(&vdec, be, arena, z, zh, zw, null, test_act);
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
                const ok = bad == 0 and rel < checkTol(5e-3);
                if (!ok) failures += 1;
                try stdout.print("  {d}x{d} {s:<12} {d:7.0} ms  rel L2 {e:.4}  {s}{s}\n", .{
                    zh,                         zw,                                    if (naive) "be.attn" else "opAttnTC", dt, rel,
                    if (ok) "ok" else "FAILED", if (bad != 0) " (non-finite)" else "",
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

/// The Vulkan block-quant decode GEMV, per format, in GB/s over the weight.
///
/// There used to be five routes here (a 32-row-group transposed GEMV with
/// k-split partials, an int8-repack one, and three cooperative ones, four of
/// them behind `TP_VK_*` knobs that defaulted off). This harness is what
/// compared them, on one weight copy per route, and the dual body won every
/// format by 1.06x to 2.88x, so the rest are gone and BACKEND.md carries the
/// table. What is left is the per-format rate, which is what says whether a
/// format is at the machine or leaving it on the table. `git log` has the
/// comparison if it ever needs re-running.
///
/// ⚠️ Absolute times include Vulkan's per-submit cost, since every op here ends
/// in `submitAndWait`, and an LLM decode on this backend is SUBMIT-bound rather
/// than kernel-bound: switching every format to a 1.7x faster GEMV moved a 9B
/// q6_k model's tok/s not at all. Compare rows to each other, not to CUDA.
/// `dual-gemv-bench`: the dual q8 decode GEMV, one source, on Vulkan and on zig-cuda at
/// the shapes of a 4B decode: one activation staging, then 50 GEMVs with no barrier
/// between them, best of 5. What the two report apart is the compile and the launch,
/// not the kernel's design.
fn dualGemvBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const DType = TensorPencil.DType;
    const ctx = TensorPencil.gpu.context.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    // The CUDA columns are left empty on a card without it.
    const be_opt: ?*TensorPencil.gpu.cuda.Backend = TensorPencil.gpu.cuda.Backend.init(arena) catch null;
    defer if (be_opt) |be| be.deinit();
    const Shape = struct { rows: usize, cols: usize };
    const shapes = [_]Shape{ .{ .rows = 4096, .cols = 2560 }, .{ .rows = 1024, .cols = 2560 }, .{ .rows = 2560, .cols = 4096 }, .{ .rows = 9728, .cols = 2560 }, .{ .rows = 2560, .cols = 9728 }, .{ .rows = 17408, .cols = 5120 } };
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const reps = 50;
    try stdout.print("GB/s, f32 activation (decode's route on Vulkan) and q8_1 staged\n{s:<6} {s:>14} {s:>8} {s:>8} {s:>8} {s:>8}\n", .{ "dtype", "shape", "vk f32", "vk q8", "cu f32", "cu q8" });
    for ([_]DType{ .q4_0, .q4_k, .q5_k, .q6_k, .q8_0, .q2_k, .q3_k, .iq4_nl, .iq4_xs, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq3_s, .q1_0, .q2_0_g64, .q2_0_g128 }) |dt| for (shapes) |sh| {
        const w = try arena.alloc(u8, dt.storageBytes(sh.rows * sh.cols));
        rnd.bytes(w);
        var off: usize = 0;
        while (off < w.len) : (off += dt.blockBytes()) switch (dt) {
            .q8_0 => std.mem.writeInt(u16, w[off..][0..2], 0x2A66, .little),
            .q4_k, .q5_k => {
                std.mem.writeInt(u16, w[off..][0..2], 0x2A66, .little);
                std.mem.writeInt(u16, w[off + 2 ..][0..2], 0x251F, .little);
            },
            else => std.mem.writeInt(u16, w[off + 208 ..][0..2], 0x2A66, .little),
        };
        const xh = try arena.alloc(f32, sh.cols);
        for (xh) |*v| v.* = rnd.floatNorm(f32);
        var us: [4]f64 = @splat(std.math.inf(f64));
        inline for (0..4) |arm| arm: {
            if (arm >= 2 and be_opt == null) break :arm;
            const dev = if (arm < 2) ctx else be_opt.?;
            const q8 = arm % 2 == 1;
            var x_d = try dev.tensorCreate(sh.cols * 4);
            defer dev.tensorDestroy(&x_d);
            var y_d = try dev.tensorCreate(sh.rows * 4);
            defer dev.tensorDestroy(&y_d);
            try dev.tensorUpload(x_d, std.mem.sliceAsBytes(xh));
            for (0..6) |round| {
                const t0 = std.Io.Clock.awake.now(io).nanoseconds;
                try dev.beginBatch();
                if (q8) try dev.opGemvQuantizeX(x_d, sh.cols);
                dev.independent(reps);
                for (0..reps) |_| switch (arm) {
                    0 => try dev.opGemvQuantDual(dt, y_d, 0, x_d, w, 1.0, sh.rows, sh.cols),
                    1 => try dev.opGemvQuantDualQ8(dt, y_d, 0, w, 1.0, sh.rows, sh.cols),
                    2 => try dev.opGemvQuantDualF32(dt, y_d, x_d, w, 1.0, sh.rows, sh.cols),
                    else => try dev.opGemvQuantDualQ8(dt, y_d, w, 1.0, sh.rows, sh.cols),
                };
                try dev.endBatch();
                const t: f64 = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e3 / reps;
                if (round > 0) us[arm] = @min(us[arm], t);
            }
        }
        const mb: f64 = @floatFromInt(w.len);
        var shape_buf: [32]u8 = undefined;
        try stdout.print("{t:<6} {s:>14} {d:8.0} {d:8.0} {d:8.0} {d:8.0}\n", .{ dt, try std.fmt.bufPrint(&shape_buf, "{d}x{d}", .{ sh.rows, sh.cols }), mb / us[0] / 1e3, mb / us[1] / 1e3, mb / us[2] / 1e3, mb / us[3] / 1e3 });
    };
    try stdout.flush();
}

/// `vk-bq-bench`: Vulkan's block-quant GEMM split into its stages (dequant to 16-bit
/// rows, the k-major transpose, the coop GEMM) at diffusion and prefill shapes, each stage's
/// cost by subtraction from variants that skip it, interleaved in one process.
fn vkBqBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const gpu = TensorPencil.gpu.context;
    const DType = TensorPencil.DType;
    const ctx = gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    defer gpu.bench_bq_skip = .none;
    try stdout.print("== vk-bq-bench ==\nvulkan device: {s}\n", .{ctx.device_name[0..ctx.device_name_len]});

    const Shape = struct { rows: usize, cols: usize, m: usize, who: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 3072, .cols = 3072, .m = 4096, .who = "krea2 qkv/o, 1024^2" },
        .{ .rows = 12288, .cols = 3072, .m = 4096, .who = "krea2 mlp up" },
        .{ .rows = 3072, .cols = 12288, .m = 4096, .who = "krea2 mlp down" },
        .{ .rows = 14336, .cols = 4096, .m = 512, .who = "LLM prefill 512" },
    };
    var prng = std.Random.DefaultPrng.init(0xB0B);
    const rnd = prng.random();
    const Skip = @TypeOf(gpu.bench_bq_skip);
    const skips = [_]Skip{ .none, .pack, .expand };
    const reps = 5;
    const rounds = 7;
    for ([_]DType{ .q4_k, .q8_0 }) |dt| for (shapes) |sh| {
        const wf = try arena.alloc(f32, sh.rows * sh.cols);
        for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.02;
        const w = try arena.alloc(u8, dt.storageBytes(sh.rows * sh.cols));
        _ = try TensorPencil.quants.quantizeChunk(dt, wf, w, sh.rows, sh.cols, null);
        const xh = try arena.alloc(f32, sh.m * sh.cols);
        for (xh) |*v| v.* = rnd.floatNorm(f32);
        var xd = try ctx.tensorCreate(xh.len * 4);
        defer ctx.tensorDestroy(&xd);
        try ctx.tensorUpload(xd, std.mem.sliceAsBytes(xh));
        var yd = try ctx.tensorCreate(sh.m * sh.rows * 4);
        defer ctx.tensorDestroy(&yd);
        const bias = try arena.alloc(f32, sh.rows);
        @memset(bias, 0);

        var t: [skips.len][rounds]f64 = undefined;
        for (skips) |s| {
            gpu.bench_bq_skip = s;
            try ctx.opMatmulCoopQuant(dt, yd, 0, xd, sh.m, w, sh.rows, sh.cols, 1.0, bias, false);
        }
        for (0..rounds) |r| for (0..skips.len) |j| {
            const i = (j + r) % skips.len;
            gpu.bench_bq_skip = skips[i];
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            try ctx.beginBatch();
            for (0..reps) |_| try ctx.opMatmulCoopQuant(dt, yd, 0, xd, sh.m, w, sh.rows, sh.cols, 1.0, bias, false);
            try ctx.endBatch();
            t[i][r] = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6 / reps;
        };
        var med: [skips.len]f64 = undefined;
        for (&t, &med) |*col, *md| {
            std.mem.sort(f64, col, {}, std.sort.asc(f64));
            md.* = col[rounds / 2];
        }
        gpu.bench_bq_skip = .none;
        const elems: f64 = @floatFromInt(sh.rows * sh.cols);
        const pack = med[0] - med[1];
        const deq = med[1] - med[2];
        // Stage figures are differences of medians, so a small stage can read negative.
        try stdout.print("{t:<5} {d:>5}x{d:<5} m {d:<4} {s:<22} total {d:7.3} ms | dequant {d:6.3} ({d:5.0} GB/s out) | transpose {d:6.3} ({d:5.0} GB/s) | gemm {d:6.3}\n", .{
            dt, sh.rows, sh.cols, sh.m, sh.who, med[0], deq, elems * 2 / deq / 1e6, pack, elems * 4 / pack / 1e6, med[2],
        });
    };
}

fn vkGemvBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const gpu = TensorPencil.gpu.context;
    const DType = TensorPencil.DType;

    const ctx = gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== vk-gemv-bench ==\nvulkan device: {s}\n", .{ctx.device_name[0..ctx.device_name_len]});

    const dts = [_]DType{
        .q8_0,    .q4_0,   .iq4_nl,  .iq4_xs, .q2_k,     .q3_k,
        .q4_k,    .q5_k,   .q6_k,    .q1_0,   .q2_0_g64, .q2_0_g128,
        .iq2_xxs, .iq2_xs, .iq3_xxs, .iq3_s,
    };

    const Shape = struct { rows: usize, cols: usize, who: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 17408, .cols = 5120, .who = "17408 x 5120 (mlp gate/up)" },
        .{ .rows = 5120, .cols = 17408, .who = "5120 x 17408 (mlp down)" },
    };

    var prng = std.Random.DefaultPrng.init(0xB10C);
    const rnd = prng.random();

    // `batched` runs the twenty iterations inside ONE beginBatch, which is how a
    // stepper issues them; unbatched is one submit-and-fence per op. The gap
    // between the two columns is what per-submission cost a decode actually pays.
    const T = struct {
        io: Io,
        ctx: *gpu.Context,
        fn run(self: @This(), comptime batched: bool, comptime f: anytype, args: anytype) !f64 {
            for (0..20) |_| try @call(.auto, f, args);
            var best: f64 = std.math.inf(f64);
            for (0..5) |_| {
                const t0 = std.Io.Clock.real.now(self.io);
                if (batched) try self.ctx.beginBatch();
                for (0..20) |_| try @call(.auto, f, args);
                if (batched) try self.ctx.endBatch();
                const ns = std.Io.Clock.real.now(self.io).nanoseconds - t0.nanoseconds;
                best = @min(best, @as(f64, @floatFromInt(ns)) / 1e6 / 20.0);
            }
            return best;
        }
    }{ .io = io, .ctx = ctx };

    for (shapes) |sh| {
        var x_d = try ctx.tensorCreate(sh.cols * 4);
        var y_d = try ctx.tensorCreate(sh.rows * 4);
        defer {
            ctx.tensorDestroy(&x_d);
            ctx.tensorDestroy(&y_d);
        }
        const xh = try arena.alloc(f32, sh.cols);
        for (xh) |*v| v.* = rnd.floatNorm(f32);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(xh));

        try stdout.print("\n=== {s} ===\n{s:<11} {s:>10} {s:>10} {s:>10} {s:>11} {s:>8}\n", .{
            sh.who, "dtype", "1-per-submit", "batched", "GB/s", "Gelem/s", "MiB",
        });
        for (dts) |dt| {
            // Random block bytes with the scales pinned, not a real quantization:
            // every kernel here is straight-line arithmetic over the block, so the
            // VALUES cannot move the timing, and correctness is the device tests'
            // job, not this one's.
            const q = try arena.alloc(u8, dt.storageBytes(sh.rows * sh.cols));
            rnd.bytes(q);
            var off: usize = 0;
            while (off < q.len) : (off += dt.blockBytes()) switch (dt) {
                .q4_k, .q5_k => {
                    std.mem.writeInt(u16, q[off..][0..2], 0x2A66, .little);
                    std.mem.writeInt(u16, q[off + 2 ..][0..2], 0x251F, .little);
                },
                .q2_k => {
                    std.mem.writeInt(u16, q[off + 80 ..][0..2], 0x2A66, .little);
                    std.mem.writeInt(u16, q[off + 82 ..][0..2], 0x251F, .little);
                },
                .q3_k => std.mem.writeInt(u16, q[off + 108 ..][0..2], 0x2A66, .little),
                .q6_k => std.mem.writeInt(u16, q[off + 208 ..][0..2], 0x2A66, .little),
                else => std.mem.writeInt(u16, q[off..][0..2], 0x2A66, .little),
            };

            const args = .{ ctx, dt, y_d, @as(usize, 0), x_d, @as([]const u8, q), @as(f32, 1.0), sh.rows, sh.cols };
            const ms = try T.run(false, gpu.Context.opGemvQuantDual, args);
            const bms = try T.run(true, gpu.Context.opGemvQuantDual, args);
            try stdout.print("{s:<11} {d:7.3} ms {d:7.3} ms {d:9.1} {d:11.1} {d:8.0}\n", .{
                @tagName(dt),
                ms,
                bms,
                @as(f64, @floatFromInt(q.len)) / (bms * 1e-3) / 1e9,
                @as(f64, @floatFromInt(sh.rows * sh.cols)) / (bms * 1e-3) / 1e9,
                @as(f64, @floatFromInt(q.len)) / (1 << 20),
            });
        }
    }
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
        // alternatives (`qknorm_rope`, its f16 instance). It is live for a bf16 or fp8
        // checkpoint, `qkv_shared` requires `!is_bf16`, so those fall through, and NOT for
        // an int8 one, which takes the fused f32 path. So this row is a real saving for the
        // dense checkpoints only.
        .{ .who = "krea2  Q/K bf16 @1120x1680", .rows = 7350 * 48, .dim = 128, .per_step = 2 * 28 },
        // Reference only, `per_step = 0`. A 6144-wide row is krea2's block-norm
        // shape, but `dit_gpu` fuses those with the AdaLN modulation (`rms_mod`),
        // NOT `Elt.rmsnorm`. Timed here so nobody reads the wide-row numbers as an
        // available krea2 win; there is none.
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
            c.who,                                     c.rows,            c.dim,
            thr,                                       bytes / thr / 1e6, sub,
            bytes / sub / 1e6,                         thr / sub,         thr * @as(f64, @floatFromInt(c.per_step)),
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
        const q8 = try T.run(iters, gpu.Context.opI8Gemm, .{ ctx, y_d, @as([]const u8, wi8[i]), @as([]const f32, wsc[i]), sh.rows });
        bf_ms += bf * @as(f64, @floatFromInt(sh.per_block));
        i8_ms += q8 * @as(f64, @floatFromInt(sh.per_block));
        // One prep per GROUP, not per GEMM: q/k/v/out/cross share cols=2048 (4 groups in a
        // block: qkv, self-out, cross-q, cross-out) and mlp1/mlp2 have one each.
        const groups: f64 = if (sh.cols == d and sh.rows == d) 4 else 1;
        prep_ms += pr * groups;
        // TFLOP/s is computed on the time NET of the submit floor, so a small GEMM is not
        // credited with the harness's own overhead.
        try stdout.print("  {s} x{d}  bf16 {d:7.2} ms ({d:5.1} TF/s net)  int8 {d:7.2} ms ({d:5.1} TF/s net)  {d:4.2}x   prep {d:6.2} ms\n", .{
            sh.name, sh.per_block,
            bf,      flop / @max(bf - floor, 1e-6) / 1e9,
            q8,      flop / @max(q8 - floor, 1e-6) / 1e9,
            bf / q8, pr,
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
fn animaCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, libs: bool, lora: TestLoraArg, lora_layers: usize) !void {
    const cuda = TensorPencil.gpu.cuda;
    const anima = TensorPencil.models.anima;
    const anima_cuda = TensorPencil.models.anima_cuda;
    const ops = TensorPencil.ops;

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
    var ck = try TensorPencil.pipeline.Container.open(arena, io, ckpt);
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
    const dstore = try TensorPencil.pipeline.denoiserStoreIn(arena, ck.store());
    const full = try anima.detectConfig(dstore);
    for ([_]usize{ 1, 2, 8, full.n_layers }) |depth| {
        var cfg = full;
        cfg.n_layers = depth;
        var model = try anima.DiT.load(arena, dstore, cfg);
        defer model.deinit();
        if (!anima_cuda.supported(&model)) {
            try stdout.print("  checkpoint dtype unsupported on this backend\n", .{});
            break;
        }
        // The first non-dense linear says "is this depth quantized, and as what".
        const dense: TensorPencil.models.lin.Caps = .{ .f32 = true, .f16 = true, .bf16 = true, .fp8 = true };
        const qdt: ?TensorPencil.dtype.DType = if (TensorPencil.models.lin.unsupported(model.device_lins, dense)) |q| q.dtype else null;
        failures += try animaForwardCheck(arena, io, stdout, be, &model, depth, qdt, rnd);
    }

    if (try openTestLora(arena, io, stdout, lora.path, lora.strength, .{})) |st| {
        var cfg = full;
        if (lora_layers != 0) cfg.n_layers = lora_layers;
        var model = try anima.DiT.load(arena, dstore, cfg);
        defer model.deinit();
        const dense: TensorPencil.models.lin.Caps = .{ .f32 = true, .f16 = true, .bf16 = true, .fp8 = true };
        const qdt: ?TensorPencil.dtype.DType = if (TensorPencil.models.lin.unsupported(model.device_lins, dense)) |q| q.dtype else null;
        const lat_h = 24;
        const lat_w = 32;
        const ctx_seq = cfg.adapter.min_rows;
        const cond = try arena.alloc(f32, ctx_seq * cfg.context_dim);
        for (cond) |*t| t.* = rnd.floatNorm(f32) * 0.5;
        const x_lat = try arena.alloc(f32, cfg.channels * lat_h * lat_w);
        for (x_lat) |*t| t.* = rnd.floatNorm(f32);
        const sigma: f32 = 0.75;
        var want: [2][]f32 = undefined;
        var got: [2][]f32 = undefined;
        for (0..2) |pass| {
            if (pass == 1) {
                failures += try reportAttach(stdout, st, try model.attachLora(arena, st), lora_layers != 0);
                try TensorPencil.models.lin_cuda.quantizeLora(arena, st, model.device_lins, TensorPencil.models.lin_cuda.blockq_gemm, .w8a8);
                if (!anima_cuda.supported(&model)) return error.ValidationFailed;
            }
            want[pass] = try arena.alloc(f32, x_lat.len);
            try model.predict(io, arena, want[pass], x_lat, lat_h, lat_w, cond, ctx_seq, try model.modulationTable(io, arena, sigma), null);
            const sched = try model.modulationSchedule(io, arena, &.{ sigma, 0 });
            var sess = try anima_cuda.Session.init(arena, be, &model, lat_h, lat_w, cond, ctx_seq, &.{ sigma, 0 }, sched);
            defer sess.deinit(arena, be);
            var ws = try anima_cuda.Workspace.init(be, &model, lat_h, lat_w, TensorPencil.models.anima_dev.streamDtFor(testAct()));
            defer ws.deinit(be);
            got[pass] = try arena.alloc(f32, x_lat.len);
            try anima_cuda.forward(&model, be, &sess, &ws, io, arena, got[pass], x_lat, sigma, null);
        }
        try stdout.print("lora check at {d} blocks\n", .{cfg.n_layers});
        failures += try loraReport(stdout, want[0], want[1], got[0], got[1], checkTol(animaTol(qdt, cfg.n_layers)), true);
    }

    try summarize(stdout, failures);
}

/// `animaForwardCheck`'s bound for a device forward against the CPU at `depth`,
/// `qdt` the first non-dense linear's storage (reasoning at the call site).
fn animaTol(qdt: ?TensorPencil.dtype.DType, depth: usize) f64 {
    const base: f64 = switch (qdt orelse .f32) {
        .i4 => 6e-2,
        .i8, .w4a8, .w6a8 => 6e-3,
        // A GGUF block quant on the device decodes to convrot int8 per GEMM
        // (`lin_cuda`), so its residual against the weight-only CPU reference is int8's
        // plus the format's own regrid, not the dense one. Measured on a q4_k Anima:
        // 2.28e-3 at depth 2 rising to 1.14e-2 at 28, i.e. within int8's own band. The
        // dense bound would fail every block-quant checkpoint for being quantized.
        else => |dt| if (dt.isBlockQuant()) 6e-3 else if (depth == 1) 2.5e-3 else 1e-3,
    };
    return base * (1.0 + @as(f64, @floatFromInt(depth)) / 8.0);
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
    const tol = checkTol(animaTol(qdt, depth));
    var failures: usize = 0;
    const saved = anima_cuda.force_naive_attn;
    defer anima_cuda.force_naive_attn = saved;
    for ([_]bool{ false, true }) |naive| {
        anima_cuda.force_naive_attn = naive;
        var sess = try anima_cuda.Session.init(arena, be, model, lat_h, lat_w, cond, ctx_seq, &.{ sigma, 0 }, sched);
        defer sess.deinit(arena, be);
        var ws = try anima_cuda.Workspace.init(be, model, lat_h, lat_w, TensorPencil.models.anima_dev.streamDtFor(testAct()));
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
        flop / 1e12,                                                                   m_pad, seq,
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

/// Gemma 3's vision tower on every GPU arm against the CPU one, from one gradient
/// image: the worst token cosine and an FNV hash of each arm's embeddings, so two
/// builds can be compared byte for byte. Exits non-zero below 0.999.
/// `quant-gemm-test <file> <tensor> <rows> <cols> <x.f32> <y.f32>`: one weight through
/// the CPU GEMM as every family loads it (`quant_weight.load`), `x` ([m][cols] f32) in
/// and `y` ([m][rows]) out, so a decoder outside this codebase can check what the
/// reader made of the file.
fn quantGemmTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, a: []const []const u8) !void {
    var st = try TensorPencil.SafeTensors.open(arena, io, a[0]);
    defer st.deinit();
    const rows = try std.fmt.parseInt(usize, a[2], 10);
    const cols = try std.fmt.parseInt(usize, a[3], 10);
    const w = try TensorPencil.models.quant_weight.load(arena, .{ .safetensors = &st }, a[1], rows, cols, .{ .who = "quant-gemm-test" });
    const f = try std.Io.Dir.cwd().openFile(io, a[4], .{});
    const len: usize = @intCast(try f.length(io));
    const x = try arena.alloc(f32, len / 4);
    _ = try f.readPositionalAll(io, std.mem.sliceAsBytes(x), 0);
    f.close(io);
    const m = x.len / cols;
    const y = try arena.alloc(f32, m * rows);
    try TensorPencil.ops.matmul.matmul(io, arena, y, x[0 .. m * cols], m, w, null);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a[5], .data = std.mem.sliceAsBytes(y) });
    try stdout.print("{s}: {t} [{d}, {d}], {d} rows through the CPU GEMM\n", .{ a[1], w.dtype, rows, cols, m });
    try stdout.flush();
}

/// One vision arm against the CPU: the worst per-token cosine (failing below `floor`)
/// and an FNV hash of its embeddings, so two builds can be compared byte for byte.
fn vitReport(out: *Io.Writer, name: []const u8, ref: []const f32, got: []const f32, dim: usize, floor: f64, fail: *bool) !void {
    var min_cos: f64 = 1;
    var t: usize = 0;
    while (t * dim < ref.len) : (t += 1) {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (ref[t * dim ..][0..dim], got[t * dim ..][0..dim]) |a, b| {
            dot += @as(f64, a) * b;
            na += @as(f64, a) * a;
            nb += @as(f64, b) * b;
        }
        min_cos = @min(min_cos, dot / (@sqrt(na) * @sqrt(nb)));
    }
    try out.print("{s:<9} min_token_cos {d:.6}  fnv {x:0>16}\n", .{ name, min_cos, std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(got)) });
    if (!(min_cos > floor)) fail.* = true;
}

/// The Qwen3.5/3.6 tower on every GPU arm against the CPU one, at a non-square size so
/// the rope's row and column axes differ.
fn vit35Test(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8) !void {
    const models = TensorPencil.models;
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("vit35-test needs a Qwen3.5/3.6 mmproj ({s})\n", .{path});
        std.process.exit(1);
    };
    var g = try TensorPencil.gguf.Gguf.open(arena, io, path);
    defer g.deinit();
    var vit = try models.vit35.Vit.load(arena, &g);
    defer vit.deinit();

    const w = 160;
    const h = 96;
    var pixels: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 3;
        pixels[i] = @intCast((x * 3 + y) & 0xff);
        pixels[i + 1] = @intCast((y * 2) & 0xff);
        pixels[i + 2] = @intCast((x * y) & 0xff);
    };
    var want = try vit.encode(io, arena, &pixels, w, h);
    defer want.deinit(arena);
    try stdout.print("grid {d}x{d}\n", .{ want.grid_w, want.grid_h });
    const pd = vit.cfg.proj_dim;

    var bad = false;
    const cuda = TensorPencil.gpu.cuda;
    for ([_]bool{ false, true }) |libs| {
        var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
            try stdout.print("{s}: unavailable ({t})\n", .{ if (libs) "cuda" else "zig-cuda", err });
            continue;
        };
        defer be.deinit();
        var got = try models.vit35_cuda.encode(&vit, be, arena, &pixels, w, h);
        defer got.deinit(arena);
        try vitReport(stdout, if (libs) "cuda" else "zig-cuda", want.embeds, got.embeds, pd, 0.999, &bad);
    }
    if (TensorPencil.gpu.Context.init(arena, io)) |ctx| {
        defer ctx.deinit();
        const used0 = ctx.device_used;
        var got = try models.vit35_gpu.encode(&vit, ctx, arena, &pixels, w, h);
        defer got.deinit(arena);
        try vitReport(stdout, "vulkan", want.embeds, got.embeds, pd, 0.999, &bad);
        // A grown scratch buffer is freed whole, so the encode may leave less than it found.
        const held = @as(i64, @intCast(ctx.device_used)) - @as(i64, @intCast(used0));
        try stdout.print("vulkan    held after encode {d} B\n", .{held});
        if (held > 0) bad = true;
    } else |err| try stdout.print("vulkan: unavailable ({t})\n", .{err});
    try stdout.flush();
    if (bad) std.process.exit(1);
}

/// Gemma 4's vision tower, whichever the mmproj's projector type names, on every GPU
/// arm against the CPU one. The full `gemma4v` tower's floor is its CUDA arm's measured
/// one: f16 operand rounding compounds through 27 blocks (`gemma4v_vit_cuda`'s test).
fn gemma4VitTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8) !void {
    const models = TensorPencil.models;
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("gemma4-vit-test needs a Gemma 4 mmproj ({s})\n", .{path});
        std.process.exit(1);
    };
    var g = try TensorPencil.gguf.Gguf.open(arena, io, path);
    defer g.deinit();
    const full = std.mem.eql(u8, g.getStr("clip.vision.projector_type") orelse "", "gemma4v");
    const w = 256;
    const h = 192;
    var pixels: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 3;
        pixels[i] = @intCast(x & 0xff);
        pixels[i + 1] = @intCast(y & 0xff);
        pixels[i + 2] = 100;
    };
    const Arms = struct {
        fn run(comptime Tower: type, comptime Cuda: type, comptime Gpu: type, a: std.mem.Allocator, io_: Io, out: *Io.Writer, gg: *const TensorPencil.gguf.Gguf, px: []const u8, floor: f64) !bool {
            var vit = try Tower.Vit.load(a, gg);
            defer vit.deinit();
            var want = try vit.encode(io_, a, px, w, h);
            defer want.deinit(a);
            const pd = vit.cfg.proj_dim;
            try out.print("{s}: {d} tokens\n", .{ @typeName(Tower), want.embeds.len / pd });
            var bad = false;
            const cuda = TensorPencil.gpu.cuda;
            for ([_]bool{ false, true }) |libs| {
                var be = (if (libs) cuda.Backend.initLibs(a) else cuda.Backend.init(a)) catch |err| {
                    try out.print("{s}: unavailable ({t})\n", .{ if (libs) "cuda" else "zig-cuda", err });
                    continue;
                };
                defer be.deinit();
                var got = try Cuda.encode(&vit, be, io_, a, px, w, h);
                defer got.deinit(a);
                try vitReport(out, if (libs) "cuda" else "zig-cuda", want.embeds, got.embeds, pd, floor, &bad);
            }
            if (TensorPencil.gpu.Context.init(a, io_)) |ctx| {
                defer ctx.deinit();
                const used0 = ctx.device_used;
                var got = try Gpu.encode(&vit, ctx, io_, a, px, w, h);
                defer got.deinit(a);
                try vitReport(out, "vulkan", want.embeds, got.embeds, pd, floor, &bad);
                const held = @as(i64, @intCast(ctx.device_used)) - @as(i64, @intCast(used0));
                try out.print("vulkan    held after encode {d} B\n", .{held});
                if (held > 0) bad = true;
            } else |err| try out.print("vulkan: unavailable ({t})\n", .{err});
            return bad;
        }
    };
    const bad = if (full)
        try Arms.run(models.gemma4v_vit, models.gemma4v_vit_cuda, models.gemma4v_vit_gpu, arena, io, stdout, &g, &pixels, 0.80)
    else
        try Arms.run(models.gemma4_vit, models.gemma4_vit_cuda, models.gemma4_vit_gpu, arena, io, stdout, &g, &pixels, 0.999);
    try stdout.flush();
    if (bad) std.process.exit(1);
}

fn gemmaVitTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8) !void {
    const models = TensorPencil.models;
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("gemma-vit-test needs the Gemma 3 mmproj ({s})\n", .{path});
        std.process.exit(1);
    };
    var g = try TensorPencil.gguf.Gguf.open(arena, io, path);
    defer g.deinit();
    var vit = try models.gemma_vit.Vit.load(arena, &g);
    defer vit.deinit();

    const w = 128;
    const h = 96;
    var pixels: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 3;
        pixels[i] = @intCast((x * 2) & 0xff);
        pixels[i + 1] = @intCast((y * 2) & 0xff);
        pixels[i + 2] = 100;
    };
    var want = try vit.encode(io, arena, &pixels, w, h);
    defer want.deinit(arena);
    const pd = vit.cfg.proj_dim;

    var bad = false;

    const cuda = TensorPencil.gpu.cuda;
    for ([_]bool{ false, true }) |libs| {
        var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
            try stdout.print("{s}: unavailable ({t})\n", .{ if (libs) "cuda" else "zig-cuda", err });
            continue;
        };
        defer be.deinit();
        var got = try models.gemma_vit_cuda.encode(&vit, be, io, arena, &pixels, w, h);
        defer got.deinit(arena);
        try vitReport(stdout, if (libs) "cuda" else "zig-cuda", want.embeds, got.embeds, pd, 0.999, &bad);
    }
    if (TensorPencil.gpu.Context.init(arena, io)) |ctx| {
        defer ctx.deinit();
        var dev = try models.gemma_vit_gpu.VitGpu.load(arena, &vit);
        defer dev.deinit();
        const used0 = ctx.device_used;
        var got = try dev.encode(ctx, io, arena, &pixels, w, h);
        defer got.deinit(arena);
        try vitReport(stdout, "vulkan", want.embeds, got.embeds, pd, 0.999, &bad);
        // A grown scratch buffer is freed whole, so the encode may leave less than it found.
        const held = @as(i64, @intCast(ctx.device_used)) - @as(i64, @intCast(used0));
        try stdout.print("vulkan    held after encode {d} B\n", .{held});
        if (held > 0) bad = true;
    } else |err| try stdout.print("vulkan: unavailable ({t})\n", .{err});
    try stdout.flush();
    if (bad) std.process.exit(1);
}

/// The live-preview decoder on every GPU arm against the CPU one, from one fixed latent:
/// the worst 8-bit difference and an FNV hash of each arm's RGB, so two builds can be
/// compared byte for byte. Exits non-zero past 3 levels.
fn taehvTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8) !void {
    const taehv = TensorPencil.models.taehv;
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        try stdout.print("taehv-test needs the TAEHV checkpoint ({s})\n", .{path});
        std.process.exit(1);
    };
    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var dec = try taehv.Decoder.load(arena, &st);
    defer dec.deinit();

    const zh = 24;
    const zw = 32;
    var prng = std.Random.DefaultPrng.init(7);
    const z = try arena.alloc(f32, taehv.latent_channels * zh * zw);
    for (z) |*v| v.* = prng.random().floatNorm(f32);
    const want = try dec.decode(io, arena, z, zh, zw);

    var bad = false;
    const report = struct {
        fn f(out: *Io.Writer, name: []const u8, ref: []const u8, got: []const u8, fail: *bool) !void {
            var worst: u32 = 0;
            for (ref, got) |e, a| worst = @max(worst, @abs(@as(i32, e) - @as(i32, a)));
            try out.print("{s:<9} max_u8_diff {d}  fnv {x:0>16}\n", .{ name, worst, std.hash.Fnv1a_64.hash(got) });
            if (worst > 3) fail.* = true;
        }
    }.f;
    try stdout.print("cpu       fnv {x:0>16}\n", .{std.hash.Fnv1a_64.hash(want)});

    const cuda = TensorPencil.gpu.cuda;
    for ([_]bool{ false, true }) |libs| {
        var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
            try stdout.print("{s}: unavailable ({t})\n", .{ if (libs) "cuda" else "zig-cuda", err });
            continue;
        };
        defer be.deinit();
        const got = try TensorPencil.models.taehv_cuda.decode(&dec, be, arena, z, zh, zw);
        try report(stdout, if (libs) "cuda" else "zig-cuda", want, got, &bad);
    }
    if (TensorPencil.gpu.Context.init(arena, io)) |ctx| {
        defer ctx.deinit();
        const got = try TensorPencil.models.taehv_gpu.decode(&dec, ctx, arena, z, zh, zw);
        try report(stdout, "vulkan", want, got, &bad);
    } else |err| try stdout.print("vulkan: unavailable ({t})\n", .{err});
    try stdout.flush();
    if (bad) std.process.exit(1);
}

/// The Wan 2.1 decoder on every GPU arm at every activation width against the CPU
/// decoder, from one denormalized random latent: rel L2, worst error and time per arm.
/// Exits non-zero on a non-finite arm or one past its bound.
fn wanVaeTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, zh: usize, vae_path: []const u8) !void {
    const cuda = TensorPencil.gpu.cuda;
    const wan_vae = TensorPencil.models.wan_vae;
    const sd_vae_cuda = TensorPencil.models.sd_vae_cuda;
    const sd_vae_gpu = TensorPencil.models.sd_vae_gpu;
    const Dt = TensorPencil.gpu.context.Dt;
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch {
        try stdout.print("wan-vae-test needs the VAE checkpoint ({s})\n", .{vae_path});
        std.process.exit(1);
    };
    const zw = zh;
    const plane = zh * zw;
    var st = try TensorPencil.SafeTensors.open(arena, io, vae_path);
    defer st.deinit();
    var dec = try wan_vae.Decoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();

    // The latent at its own format's statistics: Wan 2.1's 16 channels, or Qwen-Image
    // 2.1's 64 for the Wan 2.2 layout.
    const qi = TensorPencil.models.qwen_image21;
    const zc = dec.latentChannels();
    const mean: []const f32 = if (zc == qi.latent_channels) &qi.latents_mean else &wan_vae.latents_mean;
    const sdev: []const f32 = if (zc == qi.latent_channels) &qi.latents_std else &wan_vae.latents_std;
    var prng = std.Random.DefaultPrng.init(7);
    const z = try arena.alloc(f32, zc * plane);
    for (z, 0..) |*v, i| v.* = prng.random().floatNorm(f32) * sdev[i / plane] + mean[i / plane];
    const want = try dec.decode(io, arena, z, zh, zw, null);
    // Vulkan reads repacked weights, and keys them on host addresses.
    try dec.repackConvs(arena);
    try stdout.print("== wan-vae-test zh={d} ({d}x{d} px, {d} out channels) ==\n", .{ zh, zh * dec.scale(), zw * dec.scale(), dec.outChannels() });

    var bad = false;
    const Arm = struct {
        fn run(out: *Io.Writer, io_: Io, name: []const u8, act: Dt, ref: []const f32, ctx: anytype, fail: *bool) !void {
            var best: f64 = std.math.inf(f64);
            var got: []f32 = &.{};
            for (0..2) |_| {
                const a = std.Io.Clock.real.now(io_);
                got = try ctx.decode(act);
                best = @min(best, @as(f64, @floatFromInt(std.Io.Clock.real.now(io_).nanoseconds - a.nanoseconds)) / 1e6);
            }
            var worst: f32 = 0;
            var num: f64 = 0;
            var den: f64 = 0;
            var finite = true;
            for (ref, got) |e, g| {
                if (!std.math.isFinite(g)) finite = false;
                worst = @max(worst, @abs(e - g));
                num += (@as(f64, e) - g) * (@as(f64, e) - g);
                den += @as(f64, e) * e;
            }
            const rel = @sqrt(num / den);
            const bound: f64 = if (act == .f32) 5e-3 else 2e-2;
            const ok = finite and rel < bound;
            try out.print("{s:<9} {t:<5} rel L2 {e:.3}  max {d:.4}  {d:.1} ms{s}\n", .{ name, act, rel, worst, best, if (ok) "" else if (finite) "  OVER" else "  NON-FINITE" });
            if (!ok) fail.* = true;
        }

        /// The conv gathered into the GEMM operand against the f32 patch, interleaved.
        fn convAb(out: *Io.Writer, io_: Io, a: std.mem.Allocator, name: []const u8, ctx: anytype) !void {
            const patch = &TensorPencil.gpu.dual_table.force_conv_patch;
            defer patch.* = false;
            var best = [_][2]f64{.{ std.math.inf(f64), std.math.inf(f64) }} ** 2;
            for (0..4) |_| for ([_]Dt{ .f32, .f16 }, 0..) |act, ai| for (0..2) |pi| {
                patch.* = pi == 1;
                const t0 = std.Io.Clock.real.now(io_);
                const g = try ctx.decode(act);
                best[ai][pi] = @min(best[ai][pi], @as(f64, @floatFromInt(std.Io.Clock.real.now(io_).nanoseconds - t0.nanoseconds)) / 1e6);
                a.free(g);
            };
            for ([_]Dt{ .f32, .f16 }, best) |act, b| try out.print("{s:<9} {t:<5} conv into operand {d:.1} ms, f32 patch {d:.1} ms ({d:.0}%)\n", .{ name, act, b[0], b[1], 100 * (b[0] - b[1]) / b[1] });
        }
    };
    const acts = [_]Dt{ .f32, .f16, .bf16 };
    for ([_]bool{ false, true }) |libs| {
        var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
            try stdout.print("{s}: unavailable ({t})\n", .{ if (libs) "cuda" else "zig-cuda", err });
            continue;
        };
        defer be.deinit();
        const C = struct {
            d: *const wan_vae.Decoder,
            be: *cuda.Backend,
            a: std.mem.Allocator,
            z: []const f32,
            n: usize,
            fn decode(c: @This(), act: Dt) ![]f32 {
                return sd_vae_cuda.decodeWan(c.d, c.be, c.a, c.z, c.n, c.n, null, act);
            }
        };
        for (acts) |act| try Arm.run(stdout, io, if (libs) "cuda" else "zig-cuda", act, want, C{ .d = &dec, .be = be, .a = arena, .z = z, .n = zh }, &bad);
        try Arm.convAb(stdout, io, arena, if (libs) "cuda" else "zig-cuda", C{ .d = &dec, .be = be, .a = arena, .z = z, .n = zh });
    }
    if (TensorPencil.gpu.Context.init(arena, io)) |ctx| {
        defer ctx.deinit();
        const V = struct {
            d: *const wan_vae.Decoder,
            ctx: *TensorPencil.gpu.Context,
            a: std.mem.Allocator,
            z: []const f32,
            n: usize,
            fn decode(c: @This(), act: Dt) ![]f32 {
                return sd_vae_gpu.decodeWan(c.d, c.ctx, c.a, c.z, c.n, c.n, null, act);
            }
        };
        for (acts) |act| try Arm.run(stdout, io, "vulkan", act, want, V{ .d = &dec, .ctx = ctx, .a = arena, .z = z, .n = zh }, &bad);
        try Arm.convAb(stdout, io, arena, "vulkan", V{ .d = &dec, .ctx = ctx, .a = arena, .z = z, .n = zh });
    } else |err| try stdout.print("vulkan: unavailable ({t})\n", .{err});
    try stdout.flush();
    if (bad) std.process.exit(1);
}

fn cudaEncodeTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const cuda = TensorPencil.gpu.cuda;
    const qwen3 = TensorPencil.models.qwen3;
    const qwen3_cuda = TensorPencil.models.qwen3_cuda;
    const krea2_text = TensorPencil.models.krea2_text;
    const te_path = TensorPencil.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
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
    const got0 = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, .f32, null);
    arena.free(got0);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        const g = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, .f32, null);
        const b = std.Io.Clock.real.now(io);
        arena.free(g);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    const got = try qwen3_cuda.encode(&enc, be, io, arena, ids.items, .f32, null);

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

    const warm = try dit_cuda.textTokensCuda(&model, be, io, arena, cond); // JIT + upload
    arena.free(warm);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const a = std.Io.Clock.real.now(io);
        const g = try dit_cuda.textTokensCuda(&model, be, io, arena, cond);
        const b = std.Io.Clock.real.now(io);
        arena.free(g);
        best = @min(best, @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6);
    }
    const got = try dit_cuda.textTokensCuda(&model, be, io, arena, cond);

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
    var ws = try dit_cuda.Workspace.init(be, lat, lat, seq_txt, .f32);
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

/// Writes the Vulkan coop shaders `Context.init` builds, for `spirv-val`, `spirv-dis`
/// or an offline compiler. `frag` and `tile` stand in for the device's fragment side
/// and the screened warp tile, which only a live device decides.
fn dumpCoopSpv(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, dir: []const u8, frag: u32, tile: u32) !void {
    const cm = TensorPencil.gpu.context.coopmat;
    if (dir.len == 0) {
        try stdout.print("usage: dump-coop-spv <dir> [frag 16] [warp tile 64]\n", .{});
        return;
    }
    const Shader = struct { name: []const u8, code: []align(4) const u8 };
    const shaders = [_]Shader{
        .{ .name = "coop", .code = try cm.buildGemmShared(arena, false, cm.coop_warps8, cm.coop_acc_h16, false, false, frag, frag, 64, 64) },
        .{ .name = "coop_f16w", .code = try cm.buildGemmShared(arena, true, false, false, false, false, frag, frag, tile, tile) },
        .{ .name = "coop_c16", .code = try cm.buildGemmShared(arena, false, cm.coop_warps8, true, true, false, frag, frag, 64, 64) },
        .{ .name = "coop_bf16w", .code = try cm.buildGemmShared(arena, true, false, false, false, true, frag, frag, tile, tile) },
        .{ .name = "coop_i8", .code = try cm.buildGemmI8(arena, cm.i8_mt, cm.i8_nt) },
        .{ .name = "scores", .code = try cm.buildGemmScores(arena, 128, cm.scores_stage_k) },
        .{ .name = "attn_out", .code = try cm.buildGemmAttnOut(arena) },
        .{ .name = "flash_md", .code = try cm.buildFlashMd(arena) },
        .{ .name = "flash_out", .code = try cm.buildFlashOut(arena) },
        .{ .name = "coop_i8_shared", .code = try cm.buildGemmSharedI8(arena, cm.coop_i8_warps8, false, cm.coop_i8_double_buf, false, frag, frag, 16, tile, tile) },
        .{ .name = "coop_i8_shared_fs", .code = try cm.buildGemmSharedI8(arena, cm.coop_i8_warps8, true, cm.coop_i8_double_buf, false, frag, frag, 16, tile, tile) },
        .{ .name = "fused_prep_i8", .code = try cm.buildFusedPrepI8(arena, 3072, 48 * 1024) },
        // Padded plans: no chunk divides these rows.
        .{ .name = "fused_prep_i8_2816", .code = try cm.buildFusedPrepI8(arena, 2816, 32 * 1024) },
        .{ .name = "fused_prep_i8_9728", .code = try cm.buildFusedPrepI8(arena, 9728, 32 * 1024) },
        .{ .name = "fused_prep_i8_25600", .code = try cm.buildFusedPrepI8(arena, 25600, 32 * 1024) },
        .{ .name = "wo_pack", .code = try cm.buildIntWoPack(arena, 32, true, false) },
        .{ .name = "add_rms", .code = try cm.buildAddRms(arena, 256) },
        .{ .name = "probe_storage", .code = try cm.buildCoopProbe(arena, 32, 1, false, 16, 1, 0, false) },
        .{ .name = "probe_shared_bitcast", .code = try cm.buildCoopProbe(arena, 32, 4, true, 32, 2, 512, true) },
    };
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
    defer d.close(io);
    for (shaders) |s| {
        const name = try std.fmt.allocPrint(arena, "{s}.spv", .{s.name});
        try d.writeFile(io, .{ .sub_path = name, .data = s.code });
        try stdout.print("wrote {s} ({d} bytes)\n", .{ name, s.code.len });
    }
}

/// A LoRA stack for a device check, or null when `path` is empty. Lives in `arena`
/// at a stable address, since a model holds a pointer to it; the container is
/// never closed, because the factors are views into its mapping.
fn openTestLora(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, strength: f32, keys: TensorPencil.models.lora.KeyMap) !?*TensorPencil.models.lora.Stack {
    if (path.len == 0) return null;
    const ck = try arena.create(TensorPencil.pipeline.Container);
    ck.* = try TensorPencil.pipeline.Container.openNamed(arena, io, path);
    const st = try arena.create(TensorPencil.models.lora.Stack);
    st.* = .{};
    _ = try st.add(arena, ck.store(), strength, path, keys);
    try stdout.print("lora: {s} at strength {d}, {d} keys\n", .{ path, strength, st.files.items[0].side.keyCount() });
    return st;
}

/// Print what `attachLora` applied, and fail unless it is every key in the file.
/// A `truncated` trunk runs fewer blocks than the file patches, so there a
/// partial count is the expected answer and is only printed.
fn reportAttach(stdout: *Io.Writer, st: *const TensorPencil.models.lora.Stack, applied: usize, truncated: bool) !usize {
    const c = st.coverage(0);
    const ok = applied == c.total or truncated;
    try stdout.print("lora attach: {d} of {d} file targets applied  {s}\n", .{
        applied, c.total, if (applied == c.total) "ok" else if (truncated) "(truncated trunk)" else "PARTIAL",
    });
    return if (ok) 0 else 1;
}

/// The LoRA's share of a device check. `want0`/`got0` are the CPU and device
/// forwards without the stack, `want1`/`got1` with it. The whole forward is gated
/// like the base check; the DELTA (with minus without) on each side is the
/// sidecar's own path with most of the base's error cancelled, and a sidecar that
/// is skipped, doubled or read at the wrong rows lands at 1 or more.
///
/// The delta bound is loose because the harness is: random inputs off the data
/// manifold make a device forward reproduce a small perturbation's response only
/// to ~0.13 (dense krea2, no LoRA involved, `cuda-dit-test`'s control row reads
/// 0.26 for an input perturbation) and W8A8 activations push that to ~0.27, while
/// the trunk sidecar skipped reads 0.997.
/// `loraReport`'s delta figure alone.
fn deltaRel(want0: []const f32, want1: []const f32, got0: []const f32, got1: []const f32) f64 {
    var dw2: f64 = 0;
    var dd2: f64 = 0;
    for (want0, want1, got0, got1) |a0, a1, b0, b1| {
        const dw = @as(f64, a1) - a0;
        const dg = @as(f64, b1) - b0;
        dw2 += dw * dw;
        dd2 += (dg - dw) * (dg - dw);
    }
    return @sqrt(dd2 / dw2);
}

fn loraReport(stdout: *Io.Writer, want0: []const f32, want1: []const f32, got0: []const f32, got1: []const f32, tol_whole: ?f64, gate_delta: bool) !usize {
    var dw2: f64 = 0;
    var dd2: f64 = 0;
    var w2: f64 = 0;
    var e2: f64 = 0;
    var bad: usize = 0;
    for (want0, want1, got0, got1) |a0, a1, b0, b1| {
        if (!std.math.isFinite(b1)) bad += 1;
        const dw = @as(f64, a1) - a0;
        const dg = @as(f64, b1) - b0;
        dw2 += dw * dw;
        dd2 += (dg - dw) * (dg - dw);
        w2 += @as(f64, a1) * a1;
        e2 += (@as(f64, b1) - a1) * (@as(f64, b1) - a1);
    }
    const whole = @sqrt(e2 / w2);
    const delta = @sqrt(dd2 / dw2);
    const moved = @sqrt(dw2 / w2);
    // Null is the control row: printed, not gated.
    const ok = if (tol_whole) |t| bad == 0 and whole < t and (!gate_delta or delta < 0.5) else true;
    try stdout.print("lora: CPU delta |with - without| / |with| = {e:.3}\n", .{moved});
    try stdout.print("lora: forward with LoRA vs CPU  rel L2 {e:.4}   delta vs CPU delta rel L2 {e:.4}  {s}{s}\n", .{
        whole,                                                                                                           delta,
        if (tol_whole == null) "(control)" else if (!ok) "FAILED" else if (gate_delta) "ok" else "ok (delta not gated)", if (bad != 0) " (non-finite)" else "",
    });
    return if (ok) 0 else 1;
}

fn cudaDitTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, path: []const u8, lat: usize, use_loop: bool, use_libs: bool, layers: usize, lora_path: []const u8, strength: f32) !void {
    const dit_mod = TensorPencil.models.dit;
    const dit_cuda = TensorPencil.models.dit_cuda;
    const cuda = TensorPencil.gpu.cuda;

    // Container.open, not SafeTensors.open: a q4_k DiT is a GGUF, and handing one to
    // the safetensors reader reports `InvalidHeader`, which says nothing.
    var container = try TensorPencil.pipeline.Container.open(arena, io, path);
    defer container.deinit();
    var model = try dit_mod.DiT.load(arena, try TensorPencil.pipeline.denoiserStoreIn(arena, container.store()));
    defer model.deinit();
    // A truncated trunk keeps the CPU reference bounded on a 26 GB dense file.
    if (layers != 0 and layers < model.blocks.len) {
        model.blocks = model.blocks[0..layers];
        model.device_lins = model.device_lins[0 .. layers * 8];
        try stdout.print("(first {d} of {d} blocks)\n", .{ layers, dit_mod.n_blocks });
    }
    const wqt = model.blocks[0].attn.wq.dtype;
    switch (TensorPencil.models.lin_cuda.check(model.device_lins, .{ .blockq = TensorPencil.models.lin_cuda.blockq_gemm })) {
        .ok => {},
        .refused => |r| {
            try stdout.print("cuda-dit-test needs a checkpoint the CUDA DiT has a GEMM for ({s}: {t} {s})\n", .{ r.tag, r.dtype, @tagName(r.why) });
            return;
        },
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
    var ws = try dit_cuda.Workspace.init(be, lat, lat, seq_txt, TensorPencil.models.dit_cuda.streamDtFor(be, testAct()));
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
        // The bounds are wide because the harness is, not because the kernels are:
        // `x` and `cond` are random normals, so the forward runs far OFF the data
        // manifold where reduced-precision activations diverge much more than on a
        // real latent, and the spread across checkpoints of one architecture and
        // format is ~2x on its own, set by the weights rather than by the route (a
        // requantized base measures the same level through five unrelated formats).
        //
        // So this gate catches a wiring break (which lands near 1.0, not 0.13) and
        // deliberately does not try to bound quantization quality; that is what a
        // whole `--dit` render and its PSNR measure.
        const tol: f32 = if (TensorPencil.models.lin_cuda.activationIs4Bit(TensorPencil.models.lin_cuda.blockq_gemm, wqt))
            0.25
        else if (wqt == .q2_k)
            0.15
        else
            0.18;
        if (!std.math.isFinite(rel) or rel > checkTol(tol)) return error.GpuMismatch;
        try stdout.print("cuda DiT forward OK ({s}; int4 vs W4A16 ref includes activation-quant)\n", .{qtag});

        const st = (try openTestLora(arena, io, stdout, lora_path, strength, .{})) orelse return;
        var control: f64 = 0;
        // The control row: the same delta comparison for a small change of the INPUT,
        // with no sidecar anywhere. The LoRA's delta figure means something only
        // beside what this harness gives for a perturbation of like size.
        {
            const x2 = try arena.alloc(f32, x.len);
            var prng2 = std.Random.DefaultPrng.init(99);
            for (x2, x) |*d, v| d.* = v + prng2.random().floatNorm(f32) * 0.02;
            const cpu_x2 = try arena.alloc(f32, x.len);
            try model.forward(io, arena, cpu_x2, x2, lat, lat, sigma, cond, seq_txt, null);
            const cuda_x2 = try arena.alloc(f32, x.len);
            try dit_cuda.forward(&model, be, &sess, &ws, io, arena, cuda_x2, x2, sigma, null);
            try stdout.print("control, an input perturbation and no LoRA:\n", .{});
            control = deltaRel(out_cpu, cpu_x2, out_cuda, cuda_x2);
            _ = try loraReport(stdout, out_cpu, cpu_x2, out_cuda, cuda_x2, null, false);
        }
        const txt_cpu0 = try model.textTokens(io, arena, cond, seq_txt);
        const txt_cuda0 = try dit_cuda.textTokensCuda(&model, be, io, arena, cond);
        var failures = try reportAttach(stdout, st, try model.attachLora(arena, st), layers != 0);
        // The text-fusion stack is its own device path (`txtGemm`, not `lin_cuda.gemm`),
        // so its delta is checked on its own before the whole forward.
        {
            const txt_cpu1 = try model.textTokens(io, arena, cond, seq_txt);
            const txt_cuda1 = try dit_cuda.textTokensCuda(&model, be, io, arena, cond);
            try stdout.print("text tokens:\n", .{});
            failures += try loraReport(stdout, txt_cpu0, txt_cpu1, txt_cuda0, txt_cuda1, 5e-3, true);
        }
        try TensorPencil.models.lin_cuda.quantizeLora(arena, st, model.device_lins, TensorPencil.models.lin_cuda.blockq_gemm, .w8a8);
        try stdout.print("lora A GEMM: {s}\n", .{if (TensorPencil.models.lin_cuda.lora_i8_a) "int8 where the base stages int8" else "bf16 (control)"});
        const cpu_l = try arena.alloc(f32, x.len);
        try model.forward(io, arena, cpu_l, x, lat, lat, sigma, cond, seq_txt, null);
        var sess_l = try dit_cuda.Session.init(arena, io, be, &model, lat, lat, cond, seq_txt);
        defer sess_l.deinit(be);
        const cuda_l = try arena.alloc(f32, x.len);
        try dit_cuda.forward(&model, be, &sess_l, &ws, io, arena, cuda_l, x, sigma, null);
        // Past where the control resolves a perturbation at all (a W8A8 trunk at full
        // depth), a delta figure measures the base, not the sidecar.
        if (control > 0.5) try stdout.print("  the control does not resolve a perturbation at this depth; the shallow runs gate the delta\n", .{});
        failures += try loraReport(stdout, out_cpu, cpu_l, out_cuda, cuda_l, tol, control <= 0.5);
        try summarize(stdout, failures);
    }
}

fn generate(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    // Flush buffered output on any error return so arg-validation messages
    // ("unknown backend ...", "--prompt is required", ...) actually reach the
    // terminal before the error unwinds past main's final flush.
    errdefer stdout.flush() catch {};
    // Unnamed, the backend is the machine's (`pipeline.detectBackend`).
    var opts: TensorPencil.pipeline.Options = .{ .prompt = "", .backend = TensorPencil.pipeline.detectBackend() };
    var out_path: []const u8 = "out.png";
    // Tracked separately from `opts.prompt.len`: an EMPTY prompt is a valid render
    // (it encodes to the unconditional embedding), so only the absent flag is an error.
    var prompt_given = false;
    var repeat: usize = 1; // --repeat N: reuse one pipeline.Session for N images (bench cross-queue residency)
    // SenseNova reference pictures, in the order they were given, which is the
    // order they are presented to the model in.
    var ref_paths: std.ArrayList([]const u8) = .empty;
    defer ref_paths.deinit(arena);
    // LoRA sidecars, in the order given (which only fixes the float sum: the
    // deltas add).
    var loras: std.ArrayList(TensorPencil.pipeline.LoraSpec) = .empty;
    defer loras.deinit(arena);
    var steers: [8]TensorPencil.pipeline.CondNoise.Steer = @splat(.{});
    var n_steers: usize = 0;
    var act_path: []const u8 = "";
    var act_scale: f32 = 0;
    var act_op: TensorPencil.models.dit.ActSteer.Op = .gain;
    var act_keep_norm = true;
    var act_curve: []const u8 = "";
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
        } else if (std.mem.eql(u8, flag, "--ref-image")) {
            try ref_paths.append(arena, val);
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
        } else if (std.mem.eql(u8, flag, "--model-hash")) {
            opts.hash_models = !(std.mem.eql(u8, val, "off") or std.mem.eql(u8, val, "0"));
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
                // Listed off the enum, so a sampler cannot be missing from the message
                // the day it is added.
                try stdout.print("unknown sampler '{s}' (expected one of:", .{val});
                inline for (comptime std.enums.values(TensorPencil.sampler.Kind)) |k| {
                    try stdout.print(" {s}", .{k.label()});
                }
                try stdout.print(")\n", .{});
                return error.InvalidArgs;
            };
            // `--sde-*` are the spellings from when the SDE samplers were the only
            // stochastic ones; they mean the same thing to every sampler that draws.
        } else if (std.mem.eql(u8, flag, "--eta") or std.mem.eql(u8, flag, "--sde-eta")) {
            opts.eta = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, flag, "--s-noise") or std.mem.eql(u8, flag, "--sde-s-noise")) {
            opts.s_noise = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, flag, "--cond-noise")) {
            // An empty curve is the off switch, not a malformed expression: a sweep
            // wants a control row it can write as `--cond-noise ""` rather than by
            // dropping the flag and everything that depends on it. Anything else is
            // validated here rather than at encode time, with the message tp-llm
            // gives for `--weight-noise`, so the two CLIs reject the same strings
            // the same way.
            if (val.len > 0) TensorPencil.noise_curve.validate(val) catch {
                try stdout.print("--cond-noise '{s}' is not a valid curve. It is an " ++
                    "expression in t (depth over the encoder taps), l and n (token index " ++
                    "and count) and a (the amount), e.g. 'a*(1-t)^2' or a bare number.\n", .{val});
                return error.InvalidArgs;
            };
            opts.cond_noise.curve = val;
        } else if (std.mem.eql(u8, flag, "--cond-noise-amount")) {
            opts.cond_noise.amount = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--cond-noise-seed")) {
            opts.cond_noise.seed = try std.fmt.parseInt(u64, val, 10);
        } else if (std.mem.eql(u8, flag, "--cond-noise-negative")) {
            opts.cond_noise.neg_scale = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--cond-noise-shape")) {
            opts.cond_noise.shape = if (std.mem.eql(u8, val, "shared"))
                .shared
            else if (std.mem.eql(u8, val, "independent"))
                .independent
            else {
                try stdout.print("unknown --cond-noise-shape '{s}' (expected: independent, shared)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--cond-noise-keep-norm")) {
            opts.cond_noise.keep_norm = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--act-steer")) {
            act_path = val;
        } else if (std.mem.eql(u8, flag, "--act-scale")) {
            act_scale = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--act-keep-norm")) {
            act_keep_norm = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--act-curve")) {
            TensorPencil.noise_curve.validate(val) catch {
                try stdout.print("--act-curve '{s}' is not a valid curve\n", .{val});
                return error.InvalidArgs;
            };
            act_curve = val;
        } else if (std.mem.eql(u8, flag, "--act-op")) {
            act_op = std.meta.stringToEnum(TensorPencil.models.dit.ActSteer.Op, val) orelse {
                try stdout.print("unknown --act-op '{s}' (expected: gain, add)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--cond-steer")) {
            // Repeatable: each one opens a term, and the `--cond-steer-scale` after
            // it sets that term's scale. Terms are independent, so "more tentacles"
            // and "less anime" are two of these rather than one compromise.
            if (n_steers == steers.len) {
                try stdout.print("at most {d} --cond-steer terms\n", .{steers.len});
                return error.InvalidArgs;
            }
            steers[n_steers] = .{ .text = val, .scale = 1 };
            n_steers += 1;
        } else if (std.mem.eql(u8, flag, "--cond-steer-mode")) {
            if (n_steers == 0) {
                try stdout.print("--cond-steer-mode needs a --cond-steer before it\n", .{});
                return error.InvalidArgs;
            }
            steers[n_steers - 1].mode = if (std.mem.eql(u8, val, "prompt"))
                .prompt
            else if (std.mem.eql(u8, val, "empty"))
                .empty
            else if (std.mem.eql(u8, val, "orthogonal"))
                .orthogonal
            else if (std.mem.eql(u8, val, "concat"))
                .concat
            else {
                try stdout.print("unknown --cond-steer-mode '{s}' (expected: prompt, empty, orthogonal, concat)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--cond-steer-scale")) {
            if (n_steers == 0) {
                try stdout.print("--cond-steer-scale needs a --cond-steer before it\n", .{});
                return error.InvalidArgs;
            }
            steers[n_steers - 1].scale = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--cond-noise-pooled")) {
            opts.cond_noise.pooled = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
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
        } else if (std.mem.eql(u8, flag, "--te-act") or std.mem.eql(u8, flag, "--dit-act") or
            std.mem.eql(u8, flag, "--unet-act") or std.mem.eql(u8, flag, "--vae-act"))
        {
            const a: ?TensorPencil.pipeline.Act = if (std.mem.eql(u8, val, "auto")) null else std.meta.stringToEnum(TensorPencil.pipeline.Act, val) orelse {
                try stdout.print("unknown activation type '{s}' for {s} (expected: auto, f32, f16, bf16)\n", .{ val, flag });
                return error.InvalidArgs;
            };
            if (std.mem.eql(u8, flag, "--te-act")) opts.te_act = a else if (std.mem.eql(u8, flag, "--vae-act")) opts.vae_act = a else opts.dit_act = a;
        } else if (std.mem.eql(u8, flag, "--inject-nonfinite")) {
            // DIAGNOSTIC: <component>:<dtype>, exercises the precision fallback.
            const colon = std.mem.indexOfScalar(u8, val, ':') orelse return error.InvalidArgs;
            TensorPencil.precision.inject = .{
                .component = std.meta.stringToEnum(TensorPencil.precision.Component, val[0..colon]) orelse return error.InvalidArgs,
                .act = std.meta.stringToEnum(TensorPencil.pipeline.Act, val[colon + 1 ..]) orelse return error.InvalidArgs,
            };
        } else if (std.mem.eql(u8, flag, "--vae-decode")) {
            opts.vae_decode = std.meta.stringToEnum(TensorPencil.pipeline.VaeDecode, val) orelse {
                try stdout.print("unknown vae decode mode '{s}' (expected: auto, whole, gpu_tiled, cpu_tiled)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--vram-budget")) {
            // "min" = bare-minimum weight residency: hold only the weights a
            // forward is using and stream everything else. Not a byte ceiling:
            // activations aren't streamable, so the smallest workable figure is
            // resolution-bound, and a fixed one is either above the floor (not
            // minimal) or below it (refuses every render).
            if (std.mem.eql(u8, val, "min")) {
                opts.vram_min_weights = true;
            } else {
                const gib = try std.fmt.parseFloat(f64, val);
                opts.vram_budget = @intFromFloat(gib * (1 << 30));
            }
        } else if (std.mem.eql(u8, flag, "--dit-dequant-at-load")) {
            // Diagnostic: expand every block-quant block linear to f32 at load, so the
            // same file runs through the dense path (`lin.dequant_at_load`).
            TensorPencil.models.lin.dequant_at_load = std.meta.stringToEnum(@TypeOf(TensorPencil.models.lin.dequant_at_load), val) orelse {
                std.log.err("--dit-dequant-at-load: expected off|f32|bf16 (got '{s}')", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--lora-host-bf16")) {
            // DIAGNOSTIC: the control row for a device-vs-host sidecar
            // comparison. See `lora.host_bf16_act`.
            TensorPencil.models.lora.host_bf16_act = std.mem.eql(u8, val, "on");
        } else if (std.mem.eql(u8, flag, "--lora")) {
            try loras.append(arena, parseLoraSpec(val));
        } else if (std.mem.eql(u8, flag, "--lora-strength")) {
            // Applies to the most recent `--lora`, like the `path:strength` form.
            if (loras.items.len == 0) {
                try stdout.print("--lora-strength before any --lora\n", .{});
                return error.InvalidArgs;
            }
            loras.items[loras.items.len - 1].strength = try std.fmt.parseFloat(f32, val);
        } else if (std.mem.eql(u8, flag, "--dit-gguf-gemm")) {
            TensorPencil.models.lin_cuda.blockq_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, val) orelse {
                try stdout.print("unknown gguf gemm target '{s}' (expected: auto, int8, int4, bf16, f16, mmq)\n", .{val});
                return error.InvalidArgs;
            };
        } else if (std.mem.eql(u8, flag, "--te-gguf-gemm")) {
            TensorPencil.models.lin_cuda.encoder_gemm = std.meta.stringToEnum(TensorPencil.models.lin_cuda.BlockQGemm, val) orelse {
                try stdout.print("unknown gguf gemm target '{s}' (expected: auto, int8, int4, bf16, f16, mmq)\n", .{val});
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

    // SenseNova reference pictures. PNG only, through the engine's own decoder:
    // the diffusion executable links libav for muxing but not libvips, so there is
    // no general image decode here, and a rendered frame round-trips as a
    // reference. They are NOT resized: the token count follows the extent, which
    // is the workflow's call upstream too.
    opts.loras = loras.items;
    opts.cond_noise.steers = steers[0..n_steers];
    if (act_path.len > 0) {
        const dirs = TensorPencil.pipeline.loadActDirs(arena, io, act_path) catch {
            try stdout.print("{s} is not a readable ACTD1 direction file\n", .{act_path});
            return error.InvalidArgs;
        };
        var curve_at: []f32 = &.{};
        if (act_curve.len > 0) {
            curve_at = try arena.alloc(f32, dirs.len / TensorPencil.models.dit.features);
            TensorPencil.noise_curve.fill(act_curve, 1, curve_at);
        }
        opts.act_steer = .{
            .dirs = dirs,
            .scale = act_scale,
            .op = act_op,
            .curve_at = curve_at,
            .keep_norm = act_keep_norm,
        };
    }
    if (ref_paths.items.len != 0) {
        // Both families' arrays from one decode: each ignores the other's field,
        // so the CLI does not have to know the architecture before the session
        // that detects it exists.
        const refs = try arena.alloc(TensorPencil.models.sensenova.RefImage, ref_paths.items.len);
        const mf_refs = try arena.alloc(TensorPencil.pipeline.Session.RefImage, ref_paths.items.len);
        for (ref_paths.items, refs, mf_refs, 0..) |path, *out, *mf_out, ri| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(256 << 20));
            defer arena.free(bytes);
            const png = TensorPencil.image.decodePngRgb(arena, bytes) catch |err| {
                try stdout.print("reference {d} ({s}) is not a PNG this build can read: {t}\n", .{ ri + 1, path, err });
                return err;
            };
            defer arena.free(png.pixels);
            const plane = png.height * png.width;
            const rgb = try arena.alloc(f32, 3 * plane);
            for (0..plane) |pix| {
                for (0..3) |c| rgb[c * plane + pix] = @as(f32, @floatFromInt(png.pixels[pix * 3 + c])) / 255.0;
            }
            out.* = .{ .rgb = rgb, .h = png.height, .w = png.width };
            mf_out.* = .{ .rgb = rgb, .height = png.height, .width = png.width };
            try stdout.print("reference {d}: {s} ({d}x{d})\n", .{ ri + 1, path, png.width, png.height });
        }
        opts.sn_ref_images = refs;
        opts.mf_ref_images = mf_refs;
        try stdout.flush();
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

    // The `parameters` block, the same builder the GUI writes: a saved PNG is the
    // record of how it was made, and a CLI render that carries none is a render
    // nobody can reproduce.
    const params = try renderParams(arena, io, &opts, &img, img.width, img.height);
    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(arena);
    try img.encodePng(arena, &png, &.{
        .{ .keyword = "parameters", .text = params },
    });
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

    var st = try TensorPencil.SafeTensors.open(arena, io, TensorPencil.model_root ++ "/vae/krea2RealVae_v10.safetensors");
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

test {
    _ = quant_attrib;
}

test "library module is reachable" {
    try std.testing.expectEqual(@as(usize, 4), TensorPencil.DType.f32.byteSize());
}

// The clip writer is the one part of the video path that runs on the linked FFmpeg (the
// system's, or the pinned package under -Dbundle-libs and on Windows).
test "the clip muxer writes an MP4 through the linked FFmpeg" {
    const av = @import("av");
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}{c}clip.mp4", .{ dir, std.fs.path.sep }, 0);
    defer gpa.free(path);

    const w = 64;
    const h = 64;
    const n = 8;
    const rate = 48000;
    const m = try av.Muxer.open(path, .{ .width = w, .height = h, .audio_channels = 2, .audio_sample_rate = rate, .meta_key = "parameters", .meta_value = "test" });
    var rgb: [w * h * 3]u8 = undefined;
    for (0..n) |f| {
        for (&rgb, 0..) |*p, i| p.* = @truncate(i * 7 + f * 31);
        m.writeFrame(&rgb) catch |err| {
            m.abort();
            return err;
        };
    }
    const frames = rate * n / 24;
    const audio = try gpa.alloc(f32, frames * 2);
    defer gpa.free(audio);
    for (audio, 0..) |*v, i| v.* = 0.25 * @sin(@as(f32, @floatFromInt(i / 2)) * 0.05);
    m.writeAudio(audio, frames) catch |err| {
        m.abort();
        return err;
    };
    try m.finish();

    const bytes = try tmp.dir.readFileAlloc(io, "clip.mp4", gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    errdefer std.debug.print("clip.mp4: {d} bytes, box {s}\n", .{ bytes.len, bytes[4..@min(8, bytes.len)] });
    try std.testing.expect(bytes.len > 1000);
    try std.testing.expectEqualStrings("ftyp", bytes[4..8]);
}

/// `generate-clip`: a MiniMax H3 render, video plus its soundtrack.
///
/// Separate from `generate` rather than a flag on it, because the result is a
/// different thing: a clip has frames, a frame rate and audio, and no single
/// `--lora <path>[:<strength>]`, one file per flag.
///
/// The strength is split from the RIGHT and only taken as one if it parses as a
/// float, so a path that happens to contain a colon still resolves as a path.
fn parseLoraSpec(arg: []const u8) TensorPencil.pipeline.LoraSpec {
    if (std.mem.lastIndexOfScalar(u8, arg, ':')) |c| {
        if (std.fmt.parseFloat(f32, arg[c + 1 ..])) |v| {
            return .{ .path = arg[0..c], .strength = v };
        } else |_| {}
    }
    return .{ .path = arg };
}

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
    // LoRA sidecars, in the order given (which only fixes the float sum: the
    // deltas add).
    var loras: std.ArrayList(pipeline.LoraSpec) = .empty;
    defer loras.deinit(gpa);
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
            try loras.append(gpa, parseLoraSpec(val orelse return error.MissingValue));
            i += 1;
        } else if (std.mem.eql(u8, flag, "--lora-strength")) {
            // Applies to the most recent `--lora`, which is also what the
            // `path:strength` form does; either spelling works.
            if (loras.items.len == 0) return error.BadArgs;
            loras.items[loras.items.len - 1].strength = try std.fmt.parseFloat(f32, val orelse return error.MissingValue);
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
            const v = val orelse return error.MissingValue;
            opts.backend = pipeline.Backend.fromStr(v) orelse {
                try stdout.print("unknown backend '{s}' (expected: cpu, vulkan, zig-cuda, cuda)\n", .{v});
                return error.InvalidArgs;
            };
            i += 1;
        } else if (std.mem.eql(u8, flag, "--te-act") or std.mem.eql(u8, flag, "--dit-act") or std.mem.eql(u8, flag, "--vae-act")) {
            const v = val orelse return error.MissingValue;
            const a: ?pipeline.Act = if (std.mem.eql(u8, v, "auto")) null else std.meta.stringToEnum(pipeline.Act, v) orelse {
                try stdout.print("unknown activation type '{s}' for {s} (expected: auto, f32, f16, bf16)\n", .{ v, flag });
                return error.InvalidArgs;
            };
            if (std.mem.eql(u8, flag, "--te-act")) clip_opts.te_act = a else if (std.mem.eql(u8, flag, "--vae-act")) clip_opts.vae_act = a else clip_opts.dit_act = a;
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

    opts.loras = loras.items;
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
            dir_path,                    cont_frames.items.len,                                       cont_frames.items[0].width,
            cont_frames.items[0].height, if (cont_wave != null) ", with audio" else ", no audio.wav", preserve_frames,
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

fn minimaxH3AudioVkTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8) !void {
    const av = TensorPencil.models.minimax_h3_audio;
    const av_cuda = TensorPencil.models.minimax_h3_audio_gpu;
    const safetensors = TensorPencil.SafeTensors;

    var ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== minimax-h3-audio-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});

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

    var prng = std.Random.DefaultPrng.init(0xa0d10);
    const rnd = prng.random();
    const z = try arena.alloc(f32, av.latent_channels * av.stereo * t);
    for (z) |*v| v.* = rnd.floatNorm(f32);

    var sess = try av_cuda.Session.init(arena, &dec);
    defer sess.deinit();
    try stdout.print("session: {d} MB of permuted weights\n", .{sess.bytes() >> 20});
    var ws = try av_cuda.Workspace.init(ctx, arena, &dec, t);
    defer ws.deinit(ctx);

    const want = try arena.alloc(f32, samples * av.stereo);
    const got = try arena.alloc(f32, samples * av.stereo);

    const t0 = std.Io.Clock.real.now(io).nanoseconds;
    try av.decode(&dec, io, arena, want, z, t);
    const t1 = std.Io.Clock.real.now(io).nanoseconds;
    try av_cuda.decode(&dec, &sess, ctx, &ws, arena, got, z, t, null);
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

/// `dual-cuda-test`: a sample of the dual-target kernels (gpu/kernels/dual.zig,
/// compiled to PTX) against an f64 host evaluation of the same formula, then their
/// achieved bandwidth. The hand-PTX twins are gone; correctness of the rest of the
/// set is the whole-forward device parity commands.
fn dualCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    _ = io;
    const cuda = TensorPencil.gpu.cuda;
    var be = cuda.Backend.init(arena) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== dual-cuda-test ==\ncuda device: {s}\n", .{be.deviceName()});

    const Op = enum { add, silu_mul, sigmoid_mul, gelu, gelu_erf, relu, l2norm_rows };
    const ops = [_]Op{ .add, .silu_mul, .sigmoid_mul, .gelu, .gelu_erf, .relu, .l2norm_rows };
    // A DiT MLP activation's size: 4096 tokens x 8192 wide. Row kernels run over
    // it as rows x l2_dim.
    const n: usize = 4096 * 8192;
    const l2_dim: usize = 128;
    const a0 = try arena.alloc(f32, n);
    const b0 = try arena.alloc(f32, n);
    var prng = std.Random.DefaultPrng.init(0xd0a1);
    const rnd = prng.random();
    for (a0) |*v| v.* = rnd.floatNorm(f32) * 3.0;
    for (b0) |*v| v.* = rnd.floatNorm(f32) * 3.0;
    const got = try arena.alloc(f32, n);

    var da = try be.tensorCreate(n * 4);
    defer be.tensorDestroy(&da);
    var db = try be.tensorCreate(n * 4);
    defer be.tensorDestroy(&db);
    try be.tensorUpload(db, std.mem.sliceAsBytes(b0));

    const timer = try be.ctx.timerCreate();
    defer be.ctx.timerDestroy(timer);

    var failed = false;
    try stdout.print("{s:<12} {s:>10} {s:>10}\n", .{ "op", "max err", "GB/s" });
    for (ops) |op| {
        var err: f64 = 0;
        var ms: f64 = 0;
        const reps = 20;
        var rep: usize = 0;
        // Rep 0 is the untimed warm-up: it carries the module's JIT.
        while (rep <= reps) : (rep += 1) {
            try be.tensorUpload(da, std.mem.sliceAsBytes(a0));
            try be.ctx.synchronize();
            try be.ctx.timerBegin(timer);
            switch (op) {
                .add => try be.opAdd(da, db, n),
                .silu_mul => try be.siluMul(da, db, n),
                .sigmoid_mul => try be.sigmoidMul(da, db, n),
                .gelu => try be.gelu(da, n),
                .gelu_erf => try be.geluErf(da, n),
                .relu => try be.opRelu(da, n),
                .l2norm_rows => try be.opL2NormRows(da, n / l2_dim, l2_dim, 1e-6),
            }
            const t = try be.ctx.timerEndMs(timer);
            if (rep > 0) ms += t;
            if (rep == 0) {
                try be.tensorDownload(da, std.mem.sliceAsBytes(got));
                err = if (op == .l2norm_rows) l2normMaxErr(a0, got, l2_dim) else dualMaxErr(op, a0, b0, got);
            }
        }
        // Two-input ops read a and b and write a; one-input ops read and write a.
        const two = op == .add or op == .silu_mul or op == .sigmoid_mul;
        const bytes: f64 = @floatFromInt(n * 4 * @as(usize, if (two) 3 else 2) + if (op == .l2norm_rows) n * 4 else 0);
        const gbs = bytes / (ms / reps * 1e-3) / 1e9;
        // f32 transcendental paths through ex2.approx: a few ulp. add and relu are exact.
        const tol: f64 = if (op == .add or op == .relu) 0 else 4e-6;
        const ok = err <= tol;
        if (!ok) failed = true;
        try stdout.print("{t:<12} {e:>10.2} {d:>10.1}{s}\n", .{ op, err, gbs, if (ok) "" else "  FAIL" });
    }

    // The weighted rmsnorm at the shapes that matter: an LLM decode hidden norm
    // (one wide row), a decode qk norm (many short rows) and a DiT trunk norm.
    // Hand PTX (block per row under 512 rows, else warp per row) against the shared
    // subgroup-per-row kernel, interleaved.
    try stdout.print("\n{s:<22} {s:>10} {s:>10} {s:>8}\n", .{ "rmsnorm rows x dim", "hand us", "dual us", "ratio" });
    const shapes = [_][2]usize{ .{ 1, 4096 }, .{ 1, 5376 }, .{ 32, 128 }, .{ 256, 128 }, .{ 4096, 3072 }, .{ 32768, 128 } };
    for (shapes) |sh| {
        const rows = sh[0];
        const dim = sh[1];
        var w = try be.tensorCreate(dim * 4);
        defer be.tensorDestroy(&w);
        try be.tensorUpload(w, std.mem.sliceAsBytes(b0[0..dim]));
        var us: [2]f64 = .{ 0, 0 };
        const reps = 50;
        var rep: usize = 0;
        while (rep <= reps) : (rep += 1) {
            for ([_]bool{ false, true }, 0..) |dual, kk| {
                cuda.backend.rms_dual = dual;
                try be.ctx.synchronize();
                try be.ctx.timerBegin(timer);
                try be.qkNorm(da, db, w, rows, dim, 1e-6);
                const t = try be.ctx.timerEndMs(timer);
                if (rep > 0) us[kk] += t * 1000.0;
            }
        }
        cuda.backend.rms_dual = true;
        try stdout.print("{d:>8} x {d:<11} {d:>10.1} {d:>10.1} {d:>8.2}\n", .{ rows, dim, us[0] / reps, us[1] / reps, us[0] / us[1] });
    }
    if (failed) return error.DualKernelMismatch;
}

fn l2normMaxErr(a: []const f32, got: []const f32, dim: usize) f64 {
    var worst: f64 = 0;
    var r: usize = 0;
    while (r < a.len / dim) : (r += 1) {
        var ss: f64 = 0;
        for (a[r * dim ..][0..dim]) |v| ss += @as(f64, v) * v;
        const scale = 1.0 / @max(@sqrt(ss), 1e-6);
        for (a[r * dim ..][0..dim], got[r * dim ..][0..dim]) |v, g| {
            const ref = v * scale;
            worst = @max(worst, @abs(@as(f64, g) - ref) / (1.0 + @abs(ref)));
        }
    }
    return worst;
}

/// max |got - ref| / (1 + |ref|) over all elements, the reference evaluated in f64.
fn dualMaxErr(op: anytype, a: []const f32, b: []const f32, got: []const f32) f64 {
    var worst: f64 = 0;
    for (a, b, got) |ai, bi, gi| {
        const x: f64 = ai;
        const y: f64 = bi;
        const ref: f64 = switch (op) {
            // The f32 sum itself: an add has one correctly rounded answer.
            .add => ai + bi,
            .silu_mul => x / (1.0 + @exp(-x)) * y,
            .sigmoid_mul => x / (1.0 + @exp(-y)),
            .gelu => x / (1.0 + @exp(-(1.5957691216057308 * (x + 0.044715 * x * x * x)))),
            .gelu_erf => 0.5 * x * (1.0 + erf64(x * 0.7071067811865476)),
            .relu => @max(0.0, x),
            .l2norm_rows => unreachable,
        };
        const e = @abs(@as(f64, gi) - ref) / (1.0 + @abs(ref));
        if (e > worst) worst = e;
    }
    return worst;
}

/// A&S 7.1.26 erf, the same approximation the kernels use, so the check measures
/// the f32 evaluation and not the approximation itself.
fn erf64(x: f64) f64 {
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    const er = 1.0 - poly * @exp(-ax * ax);
    return if (x < 0) -er else er;
}

/// `lora-info <lora>...`: each file through `Stack.add`, with no model. What loads,
/// at which ranks, and what it costs resident before and after the rank padding.
/// `lora-info [--family zimage] <file>...`: what each file holds, read through the
/// family's key map when one is named, as the attach reads it.
/// `lora-merge-check <sd ckpt> <lora>... [--strength S] [--dump <out.safetensors> <tensor>...]`:
/// merge each LoRA ALONE over an SD checkpoint, with no model loaded, and report
/// what it reached: keys applied, keys nothing here reads, tensors written, bytes
/// and time. `--dump` writes the named merged tensors (UNet names, or `clip_l:` /
/// `clip_g:` prefixed) for `tools/check_lora_merge_real.py` to compare against
/// ComfyUI on the real weights. Non-zero exit if a file fails to merge.
fn loraMergeCheck(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const lora_sd = TensorPencil.models.lora_sd;
    var ck = try pipeline.Container.open(arena, io, args[0]);
    defer ck.deinit();
    var loras: std.ArrayList([]const u8) = .empty;
    var strength: f32 = 1.0;
    var dump_path: ?[]const u8 = null;
    var dump_names: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--strength") and i + 1 < args.len) {
            i += 1;
            strength = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--dump") and i + 1 < args.len) {
            i += 1;
            dump_path = args[i];
            while (i + 1 < args.len) : (i += 1) try dump_names.append(arena, args[i + 1]);
        } else try loras.append(arena, args[i]);
    }
    var failures: usize = 0;
    for (loras.items) |path| {
        var lc = pipeline.Container.open(arena, io, path) catch |err| {
            try stdout.print("{s}: cannot open ({t})\n", .{ path, err });
            failures += 1;
            continue;
        };
        defer lc.deinit();
        const files = [_]lora_sd.File{.{ .store = lc.store(), .strength = strength, .path = path }};
        const t0 = std.Io.Clock.real.now(io).nanoseconds;
        // A real allocator, not `arena`: a merge can be gigabytes, and each file's
        // must be handed back before the next one is made.
        const gpa = std.heap.smp_allocator;
        var chk = pipeline.SdMergeCheck.init(gpa, io, &ck, &files) catch |err| {
            try stdout.print("{s}: FAILED ({t})\n", .{ std.fs.path.basename(path), err });
            failures += 1;
            continue;
        };
        defer chk.deinit(gpa);
        const dt = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e9;
        const mg = &chk.lora.merged;
        var per: [3]usize = .{ 0, 0, 0 };
        for (mg.overlays, 0..) |o, ci| if (o) |ov| {
            per[ci] = ov.patch.count();
        };
        try stdout.print("{s}: {t}, {d} keys -> {d} tensors (unet {d}, clip_l {d}, clip_g {d}), {d} MB, {d} file keys, {d:.2}s\n", .{
            std.fs.path.basename(path), chk.family, mg.applied[0], mg.tensors, per[0], per[1], per[2], mg.bytes >> 20, lc.store().names().len, dt,
        });
        if (dump_path) |dp| try dumpMerged(arena, io, dp, mg, dump_names.items);
        try stdout.flush();
    }
    if (failures > 0) {
        try stdout.flush();
        std.process.exit(1);
    }
}

fn dumpMerged(arena: std.mem.Allocator, io: Io, path: []const u8, mg: *const TensorPencil.models.lora_sd.Merged, names: []const []const u8) !void {
    var hdr: std.ArrayList(u8) = .empty;
    var payload: std.ArrayList(u8) = .empty;
    try hdr.append(arena, '{');
    var first = true;
    for (names) |full| {
        var comp: usize = 0;
        var name = full;
        for ([_][]const u8{ "clip_l:", "clip_g:" }, 1..) |p, ci| if (std.mem.startsWith(u8, full, p)) {
            comp = ci;
            name = full[p.len..];
        };
        const ov = mg.overlays[comp] orelse continue;
        const v = ov.patch.get(name) orelse continue;
        if (!first) try hdr.append(arena, ',');
        first = false;
        const dt = switch (v.info.dtype) {
            .f16 => "F16",
            .f32 => "F32",
            else => return error.UnsupportedDType,
        };
        try hdr.print(arena, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ full, dt });
        for (v.info.shape.slice(), 0..) |d, k| try hdr.print(arena, "{s}{d}", .{ if (k > 0) "," else "", d });
        try hdr.print(arena, "],\"data_offsets\":[{d},{d}]}}", .{ payload.items.len, payload.items.len + v.bytes.len });
        try payload.appendSlice(arena, v.bytes);
    }
    try hdr.append(arena, '}');
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var buf: [1 << 16]u8 = undefined;
    var w = f.writer(io, &buf);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, hdr.items.len, .little);
    try w.interface.writeAll(&len);
    try w.interface.writeAll(hdr.items);
    try w.interface.writeAll(payload.items);
    try w.interface.flush();
}

fn loraInfo(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    const lora = TensorPencil.models.lora;
    var renames: []const lora.Rename = &.{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const path = args[i];
        if (std.mem.eql(u8, path, "--family") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "zimage")) renames = &TensorPencil.models.zimage.lora_keys //
            else renames = &.{};
            continue;
        }
        var ct = try TensorPencil.pipeline.Container.openNamed(arena, io, path);
        defer ct.deinit();
        var stack: lora.Stack = .{};
        defer stack.deinit(arena);
        const n = try stack.add(arena, ct.store(), 1.0, path, .{ .renames = renames });
        var hist: [8]struct { rank: usize = 0, n: usize = 0 } = @splat(.{});
        var it = stack.files.items[0].side.index.valueIterator();
        while (it.next()) |t| for (t.factors) |f| {
            for (&hist) |*h| {
                if (h.n == 0 or h.rank == f.rank()) {
                    h.rank = f.rank();
                    h.n += 1;
                    break;
                }
            }
        };
        const b = stack.bytes();
        const pad = stack.padBytes();
        try stdout.print("{s}\n  {d} targets from {d} file keys, {d} keys in no known spelling\n  factor ranks after padding:", .{ path, n, stack.files.items[0].side.keyCount(), stack.files.items[0].side.unclaimed });
        for (hist) |h| if (h.n > 0) try stdout.print(" {d} x{d}", .{ h.rank, h.n });
        try stdout.print("\n  resident {d} bytes ({d:.1} MB), of which padding {d} ({d:.1} MB); file factors {d:.1} MB\n", .{
            b, @as(f64, @floatFromInt(b)) / 1048576.0, pad, @as(f64, @floatFromInt(pad)) / 1048576.0, @as(f64, @floatFromInt(b - pad)) / 1048576.0,
        });
    }
}

/// `lora-restack-test <ckpt> <lora>`: swap a LoRA stack on a LIVE session and
/// prove the swap is both real and clean. Non-zero exit on either failure.
///
/// Three renders of one prompt at one seed through ONE resident session:
///
///   1. with the sidecar   2. after `setLoras(&.{})`   3. after putting it back
///   4. with the dial moved on the live session   5. a stack installed at that dial
///
/// and the two checks that matter are different defects.
///
/// **1 vs 2 must DIFFER**, or the removal did nothing and the whole per-image
/// LoRA feature is decoration.
///
/// **1 vs 3 must be BIT-IDENTICAL**, so a re-attach cannot come back subtly
/// unlike what it replaced. Bytes rather than a tolerance because the two runs
/// are the same weights at the same seed and anything but equality is a defect.
///
/// **And the device must give the factors BACK.** This is the check that earns
/// the command, because the pixels cannot see it: the device weight cache is
/// keyed on the host POINTER of each factor (`lora_cuda`'s header), so a restack
/// that frees a stack without evicting its factors leaves the old device copies
/// resident forever. Re-attaching the SAME file renders identically either way
/// -- stale bytes and fresh bytes are the same bytes -- so run 3 passes both
/// image checks while VRAM has silently grown by a whole stack. MEASURED on a
/// SenseNova int4 + the 8-step turbo LoRA (777 MB of factors), with the eviction
/// deliberately disabled: `dit` went 4789 -> 4789 -> 5566 MB where the correct
/// build goes 4789 -> 4012 -> 4789. That is a leak of the entire stack per swap,
/// and a user flipping a LoRA a few times would meet it as an OOM with no
/// explanation. Hence the two memory assertions below.
fn loraRestackTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, lora_path: []const u8, te: []const u8, vae: []const u8, backend: TensorPencil.pipeline.Backend) !void {
    const pipeline = TensorPencil.pipeline;
    const specs = [_]pipeline.LoraSpec{.{ .path = lora_path, .strength = 0.8 }};

    // Small and short on purpose: this is about the stack, not the picture, and
    // three renders of a full canvas is minutes of nothing being tested.
    var opts: pipeline.Options = .{
        .prompt = "a red ceramic cube on a wooden table",
        .dit_path = ckpt,
        .width = 256,
        .height = 256,
        .steps = 4,
        .seed = 42,
        .backend = backend,
        .preview = false,
        .loras = &specs,
    };

    if (te.len > 0) {
        opts.text_encoder_path = te;
        opts.explicit_text_encoder = true;
    }
    if (vae.len > 0) {
        opts.vae_path = vae;
        opts.explicit_vae = true;
    }
    try stdout.print("== lora-restack-test ==\nckpt: {s}\nlora: {s}\n", .{ ckpt, lora_path });
    // A real allocator, not `arena`: a merged family re-merges gigabytes per swap
    // and every one must come back, or the test measures its own harness.
    var sess = pipeline.Session.init(io, std.heap.smp_allocator, opts, stdout) catch |err| {
        // Not a pass: a test that could not load has tested nothing.
        try stdout.print("cannot load: {t}\n", .{err});
        try stdout.flush();
        std.process.exit(1);
    };
    defer sess.deinit();
    if (!sess.family().supportsLora()) {
        try stdout.print("{t} has no sidecar path in this build; nothing to test\n", .{sess.family()});
        return;
    }

    const stack_bytes = sess.loraBytes();

    var with1 = try sess.generate(opts, stdout);
    defer with1.deinit(arena);
    const a = try arena.dupe(u8, with1.rgb);
    const vram_with1 = sess.deviceUsed();

    try sess.setLoras(io, &.{});
    var without = try sess.generate(opts, stdout);
    defer without.deinit(arena);
    const vram_without = sess.deviceUsed();

    try sess.setLoras(io, &specs);
    var with2 = try sess.generate(opts, stdout);
    defer with2.deinit(arena);
    const vram_with2 = sess.deviceUsed();

    // The dial: moved on the live session it must render exactly what a stack
    // installed at that strength renders, and not what the old dial did. A
    // merged family re-merges on the move, so this is its whole dial path.
    sess.setLoraStrength(0, 0.4);
    var dialed = try sess.generate(opts, stdout);
    defer dialed.deinit(arena);
    const at04 = [_]pipeline.LoraSpec{.{ .path = lora_path, .strength = 0.4 }};
    try sess.setLoras(io, &at04);
    var fresh = try sess.generate(opts, stdout);
    defer fresh.deinit(arena);
    const dial_exact = std.mem.eql(u8, dialed.rgb, fresh.rgb);
    const dial_moves = !std.mem.eql(u8, dialed.rgb, a);

    // Half the stack, so ordinary workspace jitter cannot trip either way. The
    // defect this separates from noise is a WHOLE stack, not a few MB.
    const slack = stack_bytes / 2;
    const removed_differs = !std.mem.eql(u8, a, without.rgb);
    const restored_exact = std.mem.eql(u8, a, with2.rgb);
    const freed = vram_without + slack < vram_with1;
    const no_leak = vram_with2 < vram_with1 + slack;

    try stdout.print("\nstack: {d} MB of factors\n", .{stack_bytes >> 20});
    try stdout.print("device used: with {d} MB · without {d} MB · with again {d} MB\n", .{
        vram_with1 >> 20, vram_without >> 20, vram_with2 >> 20,
    });
    try stdout.print("removing the sidecar changes the image: {s}\n", .{if (removed_differs) "yes" else "NO"});
    try stdout.print("putting it back reproduces it exactly: {s}\n", .{if (restored_exact) "yes" else "NO"});
    try stdout.print("removing it returns its VRAM: {s}\n", .{if (freed) "yes" else "NO"});
    try stdout.print("putting it back does not grow VRAM: {s}\n", .{if (no_leak) "yes" else "NO"});
    try stdout.print("moving the dial changes the image: {s}\n", .{if (dial_moves) "yes" else "NO"});
    try stdout.print("a moved dial renders what a fresh stack at that strength renders: {s}\n", .{if (dial_exact) "yes" else "NO"});

    if (!removed_differs or !restored_exact or !freed or !no_leak or !dial_moves or !dial_exact) {
        // Say WHICH: the four mean different things, and only the last two can
        // see a missing eviction.
        if (!removed_differs) try stdout.print("FAIL: the LoRA is not reaching the forward at all\n", .{});
        if (!restored_exact) try stdout.print("FAIL: the re-attached stack does not match the original\n", .{});
        if (!freed) try stdout.print("FAIL: the removed factors are still resident (missing device eviction)\n", .{});
        if (!no_leak) try stdout.print("FAIL: the swap leaked a stack's worth of VRAM (missing device eviction)\n", .{});
        if (!dial_moves) try stdout.print("FAIL: moving the dial does not reach the forward\n", .{});
        if (!dial_exact) try stdout.print("FAIL: a moved dial is not the strength it claims\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }
    try stdout.print("ok\n", .{});
}

/// `lora-vk-test`: the Vulkan sidecar (`lora_gpu.apply`) against the host apply, on the
/// fused-linear shapes `lora-cuda-test` uses and every rank real files ship. No
/// checkpoint. Exits 1 on any failure.
fn loraVkTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const lora = TensorPencil.models.lora;
    const lora_gpu = TensorPencil.models.lora_gpu;
    const Weight = TensorPencil.ops.matmul.Weight;
    const ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        std.process.exit(1);
    };
    defer ctx.deinit();
    try stdout.print("== lora-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});

    const m: usize = 150;
    const mpad = std.mem.alignForward(usize, m, 128);
    const scale: f32 = 0.0625;
    const strength: f32 = 0.75;
    var prng = std.Random.DefaultPrng.init(0x10ca);
    const rnd = prng.random();
    var scratch: lora_gpu.Scratch = .{};
    defer scratch.deinit(ctx);
    const zb = try arena.alloc(f32, 32768);
    @memset(zb, 0);

    const Case = struct { name: []const u8, in_dim: usize, out: usize, rank: usize, groups: usize, split: usize };
    var cases: std.ArrayList(Case) = .empty;
    // A fused qkv split three ways into block-diagonal factors, an fc1 asked for in two
    // halves, plain linears; then every rank real files ship, padded as `Sidecar.load`
    // pads it, including SD1.5's 320-wide output off the 128 tile.
    try cases.appendSlice(arena, &.{
        .{ .name = "qkv", .in_dim = 3072, .out = 3 * 3072, .rank = 96, .groups = 3, .split = 3 },
        .{ .name = "fc1", .in_dim = 3072, .out = 2 * 8192, .rank = 64, .groups = 1, .split = 2 },
        .{ .name = "out", .in_dim = 3072, .out = 3072, .rank = 128, .groups = 1, .split = 1 },
    });
    for ([_]usize{ 8, 32, 96, 128 }) |rank| for ([_][2]usize{ .{ 3072, 3072 }, .{ 4096, 1024 }, .{ 1280, 320 } }) |sh|
        try cases.append(arena, .{ .name = "rank sweep", .in_dim = sh[0], .out = sh[1], .rank = rank, .groups = 1, .split = 1 });

    var worst: f64 = 0;
    for (cases.items) |c| {
        const gr = c.rank / c.groups;
        const go = c.out / c.groups;
        const file_f = try arena.alloc(lora.Factor, c.groups);
        const dev_f = try arena.alloc(lora.Factor, c.groups);
        for (file_f, dev_f, 0..) |*f, *d, g| {
            const a_b = try arena.alloc(u8, gr * c.in_dim * 2);
            const b_b = try arena.alloc(u8, go * gr * 2);
            for (0..a_b.len / 2) |k| std.mem.writeInt(u16, a_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            for (0..b_b.len / 2) |k| std.mem.writeInt(u16, b_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            f.* = .{ .a = Weight.init(a_b, .bf16, gr, c.in_dim), .b = Weight.init(b_b, .bf16, go, gr), .out_off = g * go };
            d.* = try lora.padRank(arena, f.*);
            d.out_off = f.out_off;
        }
        const host_t: lora.Target = .{ .factors = file_f, .in_dim = c.in_dim, .out_dim = c.out, .scale = scale, .tag = c.name };
        const t: lora.Target = .{ .factors = dev_f, .in_dim = c.in_dim, .out_dim = c.out, .scale = scale, .tag = c.name };
        if (!lora_gpu.supported(&t)) {
            try stdout.print("FAIL: {s} has no device path\n", .{c.name});
            std.process.exit(1);
        }
        const x = try arena.alloc(f32, mpad * c.in_dim);
        @memset(x, 0);
        for (x[0 .. m * c.in_dim]) |*v| v.* = rnd.floatNorm(f32);
        const want = try arena.alloc(f32, m * c.out);
        @memset(want, 0);
        try host_t.applyHost(io, arena, want, c.out, x[0 .. m * c.in_dim], m, strength);
        // The control row: the device rounds the activation to 16 bits first.
        const x_bf = try arena.alloc(f32, m * c.in_dim);
        for (x_bf, x[0 .. m * c.in_dim]) |*d, v| d.* = @bitCast(@as(u32, f32ToBf16(v)) << 16);
        const floor_ref = try arena.alloc(f32, m * c.out);
        @memset(floor_ref, 0);
        try host_t.applyHost(io, arena, floor_ref, c.out, x_bf, m, strength);

        var xd = try ctx.tensorCreate(mpad * c.in_dim * 4);
        defer ctx.tensorDestroy(&xd);
        try ctx.tensorUpload(xd, std.mem.sliceAsBytes(x));
        const n = c.out / c.split;
        for (0..c.split) |r| {
            var yd = try ctx.tensorCreate(mpad * n * 4);
            defer ctx.tensorDestroy(&yd);
            // A nonzero base, so a GEMM that overwrote instead of adding is off by it.
            const base = try arena.alloc(f32, mpad * n);
            for (base) |*v| v.* = rnd.floatNorm(f32) * 0.01;
            try ctx.tensorUpload(yd, std.mem.sliceAsBytes(base));
            try lora_gpu.apply(ctx, &scratch, zb, yd, xd, m, &t, strength, r * n, n);
            const got = try arena.alloc(f32, mpad * n);
            try ctx.tensorDownload(yd, std.mem.sliceAsBytes(got));
            var l2_ref: f64 = 0;
            var l2_err: f64 = 0;
            var l2_floor: f64 = 0;
            for (0..m) |i| for (0..n) |j| {
                const e = want[i * c.out + r * n + j];
                const a = got[i * n + j] - base[i * n + j];
                const fl = floor_ref[i * c.out + r * n + j];
                l2_ref += @as(f64, e) * e;
                l2_err += @as(f64, e - a) * (e - a);
                l2_floor += @as(f64, e - fl) * (e - fl);
            };
            if (l2_ref == 0) {
                try stdout.print("FAIL: {s} produced a zero delta\n", .{c.name});
                std.process.exit(1);
            }
            const rel = @sqrt(l2_err / l2_ref);
            worst = @max(worst, rel);
            try stdout.print("  {s:<10} rank {d:>3} {d:>5} -> {d:>5} rows [{d:>5},{d:>5})  rel {e}  (16-bit activation floor {e})\n", .{ c.name, c.rank, c.in_dim, c.out, r * n, (r + 1) * n, rel, @sqrt(l2_floor / l2_ref) });
            try stdout.flush();
        }
    }
    try stdout.print("worst rel {e}\n", .{worst});
    if (worst > 5e-3) {
        try stdout.print("FAILED\n", .{});
        std.process.exit(1);
    }
    try stdout.print("OK\n", .{});
}

/// `lora-cuda-test`: the device LoRA sidecar against its host twin, at the real
/// H3 factor shapes, then at ranks 8 to 128 through the rank padding and both A
/// GEMMs. Non-zero exit if they disagree.
///
/// Isolates exactly the sidecar: the factors are synthetic bf16 and there is no
/// checkpoint and no trunk, so a disagreement here is `lora_cuda` and nothing
/// else. `lora.zig`'s unit tests already pin the host path against ComfyUI's
/// merged LoRA, so host-agrees-with-device closes the loop.
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
    // The runtime dial, which is NOT folded into a factor: the device apply
    // multiplies it on top of `Target.scale`, so a value other than 1 is what
    // shows the two are actually multiplied rather than one of them dropped.
    const strength: f32 = 0.75;

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
            };
        }
        const t: lora.Target = .{ .factors = factors, .in_dim = c.in_dim, .out_dim = c.out, .scale = scale, .tag = c.name };
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
        try t.applyHost(io, arena, want, c.out, x, m, strength);

        // The control row. The device GEMM rounds the activation to bf16 before
        // multiplying, so a residual of about bf16's own 2^-9 is the FLOOR, not a
        // defect. Running the host path on a pre-rounded activation says how much
        // of the residual that accounts for; without this the number below is
        // uninterpretable.
        const x_bf = try arena.alloc(f32, m * c.in_dim);
        for (x_bf, x) |*d, v| d.* = @bitCast(@as(u32, f32ToBf16(v)) << 16);
        const floor_ref = try arena.alloc(f32, m * c.out);
        @memset(floor_ref, 0);
        try t.applyHost(io, arena, floor_ref, c.out, x_bf, m, strength);

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
            try lora_cuda.applyRange(be, &ws, yd, xd, m, &t, strength, r.off, r.n);
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

    // Every rank real files ship, padded as `Sidecar.load` pads it, on the device
    // against the host apply of the FILE's unpadded factor: the padding must add
    // nothing, and a stride slip between the A and B GEMMs shows here. Both A
    // GEMMs: bf16 on the f32 activation, and the int8 A on a staged int8 prep,
    // whose `lo` is 128 wide whatever the rank.
    var worst_i8: f64 = 0;
    var worst_bf: f64 = 0;
    try stdout.print("rank sweep, m {d} (device: padded factor; host: the file's):\n", .{m});
    for ([_]usize{ 8, 32, 96, 128 }) |rank| {
        // 1280 -> 320 is SD1.5's outermost width, off the 128 tile.
        for ([_]struct { in: usize, out: usize }{ .{ .in = 3072, .out = 3072 }, .{ .in = 4096, .out = 1024 }, .{ .in = 1280, .out = 320 } }) |sh| {
            const a_b = try arena.alignedAlloc(u8, .of(u32), rank * sh.in * 2);
            const b_b = try arena.alignedAlloc(u8, .of(u32), sh.out * rank * 2);
            for (0..a_b.len / 2) |k| std.mem.writeInt(u16, a_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            for (0..b_b.len / 2) |k| std.mem.writeInt(u16, b_b[k * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * 0.02), .little);
            const file_f = try arena.alloc(lora.Factor, 1);
            file_f[0] = .{ .a = Weight.init(a_b, .bf16, rank, sh.in), .b = Weight.init(b_b, .bf16, sh.out, rank), .out_off = 0 };
            const pad_f = try arena.alloc(lora.Factor, 1);
            pad_f[0] = try lora.padRank(arena, file_f[0]);
            const file_t: lora.Target = .{ .factors = file_f, .in_dim = sh.in, .out_dim = sh.out, .scale = scale, .tag = "file" };
            const t: lora.Target = .{ .factors = pad_f, .in_dim = sh.in, .out_dim = sh.out, .scale = scale, .tag = "padded" };
            const rank_pad = pad_f[0].rank();
            if (!lora_cuda.supported(&t)) {
                try stdout.print("FAIL: rank {d} padded to {d} has no device path\n", .{ rank, rank_pad });
                return error.Unsupported;
            }

            const x = try arena.alloc(f32, m * sh.in);
            for (x) |*v| v.* = rnd.floatNorm(f32);
            const want = try arena.alloc(f32, m * sh.out);
            @memset(want, 0);
            try file_t.applyHost(io, arena, want, sh.out, x, m, strength);

            const mpad = std.mem.alignForward(usize, m, 128);
            var ws = try lora_cuda.Workspace.init(be, lora_cuda.loElems(m, rank_pad), mpad * sh.out);
            defer ws.deinit(be);
            var xd = try be.tensorCreate(mpad * sh.in * 4);
            defer be.tensorDestroy(&xd);
            const xpad = try arena.alloc(f32, mpad * sh.in);
            @memset(xpad, 0);
            @memcpy(xpad[0 .. m * sh.in], x);
            try be.tensorUpload(xd, std.mem.sliceAsBytes(xpad));
            var yd = try be.tensorCreate(mpad * sh.out * 4);
            defer be.tensorDestroy(&yd);
            const zeros = try arena.alloc(f32, mpad * sh.out);
            @memset(zeros, 0);
            const aq = try lora.quantizeA(arena, pad_f[0].a, true);
            const got = try arena.alloc(f32, m * sh.out);

            for ([_]bool{ false, true }) |int8_a| {
                try be.tensorUpload(yd, std.mem.sliceAsBytes(zeros));
                try be.beginBatch();
                errdefer if (be.batching()) be.abortBatch();
                if (int8_a) try be.opI8PrepR(xd, m, sh.in, true);
                try lora_cuda.apply(be, &ws, yd, xd, m, &t, strength, 0, sh.out, .{ .a_i8 = if (int8_a) aq.weight(sh.in, true) else null });
                try be.endBatch();
                try be.tensorDownload(yd, std.mem.sliceAsBytes(got));
                var l2_ref: f64 = 0;
                var l2_err: f64 = 0;
                for (want, got) |e, a| {
                    l2_ref += @as(f64, e) * e;
                    l2_err += @as(f64, e - a) * (e - a);
                }
                if (l2_ref == 0) {
                    try stdout.print("FAIL: rank {d} produced a zero delta\n", .{rank});
                    return error.Unsupported;
                }
                const rel = @sqrt(l2_err / l2_ref);
                if (int8_a) worst_i8 = @max(worst_i8, rel) else worst = @max(worst, rel);
                try stdout.print("  rank {d:>3} (pad {d:>3})  {d:>5} -> {d:>5}  {s:<6} A  rel {e}\n", .{ rank, rank_pad, sh.in, sh.out, if (int8_a) "int8" else "bf16", rel });
            }

            // A 16-bit activation stream: x stored at that width, and the delta
            // accumulated onto a NONZERO base of it at its own magnitude, so a GEMM that
            // overwrote instead of adding would be off by the whole base.
            for ([_]cuda.backend.Dt{ .f16, .bf16 }) |sdt| {
                const toRow = struct {
                    fn go(dt: cuda.backend.Dt, src: []const f32, dst: []u8) void {
                        if (dt == .f16) TensorPencil.dtype.f32ToF16Row(src, dst) else TensorPencil.dtype.f32ToBf16Row(src, dst);
                    }
                    fn at(dt: cuda.backend.Dt, b: []const u8, i: usize) f32 {
                        const u = std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
                        return if (dt == .f16) @as(f16, @bitCast(u)) else @bitCast(@as(u32, u) << 16);
                    }
                };
                var x16d = try be.tensorCreate(mpad * sh.in * 2);
                defer be.tensorDestroy(&x16d);
                const x16 = try arena.alloc(u8, mpad * sh.in * 2);
                toRow.go(sdt, xpad, x16);
                try be.tensorUpload(x16d, x16);
                var y16d = try be.tensorCreate(mpad * sh.out * 2);
                defer be.tensorDestroy(&y16d);
                var rms: f64 = 0;
                for (want) |v| rms += @as(f64, v) * v;
                rms = @sqrt(rms / @as(f64, @floatFromInt(want.len)));
                const base = try arena.alloc(f32, mpad * sh.out);
                for (base) |*v| v.* = @floatCast(rms * rnd.floatNorm(f64));
                const base16 = try arena.alloc(u8, mpad * sh.out * 2);
                toRow.go(sdt, base, base16);
                const got16 = try arena.alloc(u8, mpad * sh.out * 2);
                for ([_]bool{ false, true }) |int8_a| {
                    try be.tensorUpload(y16d, base16);
                    try be.beginBatch();
                    errdefer if (be.batching()) be.abortBatch();
                    if (int8_a) try be.opI8PrepR(x16d.as(sdt), m, sh.in, true);
                    try lora_cuda.apply(be, &ws, y16d.as(sdt), x16d.as(sdt), m, &t, strength, 0, sh.out, .{ .a_i8 = if (int8_a) aq.weight(sh.in, true) else null });
                    try be.endBatch();
                    try be.tensorDownload(y16d, got16);
                    var l2_ref: f64 = 0;
                    var l2_err: f64 = 0;
                    for (want, 0..) |e, i| {
                        const a = toRow.at(sdt, got16, i) - toRow.at(sdt, base16, i);
                        l2_ref += @as(f64, e) * e;
                        l2_err += @as(f64, e - a) * (e - a);
                    }
                    const rel = @sqrt(l2_err / l2_ref);
                    // bf16 rounds the base-plus-delta sum to 8 bits, ~4e-3 of a delta
                    // as large as its base; scored on its own.
                    if (sdt == .bf16) worst_bf = @max(worst_bf, rel) else if (int8_a) worst_i8 = @max(worst_i8, rel) else worst = @max(worst, rel);
                    try stdout.print("  rank {d:>3} (pad {d:>3})  {d:>5} -> {d:>5}  {s:<6} A  {t} in/out  rel {e}\n", .{ rank, rank_pad, sh.in, sh.out, if (int8_a) "int8" else "bf16", sdt, rel });
                }
            }
            be.evictWeights();
        }
    }

    // bf16 factors against an f32 host reference: the tolerance is the bf16
    // GEMM's own, ~1e-3 relative at these widths, not an exactness claim. The
    // int8 A quantizes A and the activation per row, ~1e-2.
    try stdout.print("worst rel {e} (bf16 A), {e} (int8 A), {e} (bf16 storage)\n", .{ worst, worst_i8, worst_bf });
    if (worst > 5e-3 or worst_i8 > 3e-2 or worst_bf > 3e-2) {
        try stdout.print("FAIL: the device sidecar disagrees with the host one\n", .{});
        return error.Unsupported;
    }
    try stdout.print("OK\n", .{});
}

/// `velocity-diff <ckpt> [--text-encoder P] [--text-encoder-2 P] [--vae P] [--size N]
/// [--backend cuda|zig-cuda|vulkan] [--lora P] [--lora-bf16-a] [--dit-act D]
/// [--cpu-round-residual f16|bf16]`: one denoiser forward on the CPU and one on the
/// device, from ONE conditioning (encoded on the CPU) and one latent, so the rel L2
/// is the denoiser's alone. Run the dense checkpoint first: that is the floor a
/// quantized one is read against. `--cpu-round-residual` (krea2) rounds the CPU's
/// residual stream after every add, which is what a 16-bit device stream costs alone.
fn velocityDiff(io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const gpa = std.heap.smp_allocator;
    var opts: pipeline.Options = .{ .prompt = "a red fox sitting on a mossy rock in a misty pine forest, morning light", .dit_path = args[0], .backend = .cpu, .preview = false };
    var size: usize = 512;
    var device: pipeline.Backend = .cuda;
    var dit_act: pipeline.Act = .f32;
    var loras: [1]pipeline.LoraSpec = undefined;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        // The control row for a LoRA on an int8 trunk: CUDA's sidecar A on the
        // unquantized activation.
        if (std.mem.eql(u8, args[i], "--lora-bf16-a")) {
            TensorPencil.models.lin_cuda.lora_i8_a = false;
            i -= 1;
            continue;
        }
        if (i + 1 >= args.len) return error.InvalidArgs;
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--lora")) {
            loras[0] = .{ .path = v };
            opts.loras = &loras;
        } else if (std.mem.eql(u8, args[i], "--text-encoder")) {
            opts.text_encoder_path = v;
            opts.explicit_text_encoder = true;
        } else if (std.mem.eql(u8, args[i], "--text-encoder-2")) {
            opts.text_encoder_2_path = v;
            opts.explicit_text_encoder_2 = true;
        } else if (std.mem.eql(u8, args[i], "--vae")) {
            opts.vae_path = v;
            opts.explicit_vae = true;
        } else if (std.mem.eql(u8, args[i], "--size")) {
            size = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, args[i], "--backend")) {
            device = pipeline.Backend.fromStr(v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, args[i], "--dit-act")) {
            dit_act = std.meta.stringToEnum(pipeline.Act, v) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, args[i], "--cpu-round-residual")) {
            TensorPencil.models.dit.round_residual = std.meta.stringToEnum(@TypeOf(TensorPencil.models.dit.round_residual.?), v) orelse return error.InvalidArgs;
        } else return error.InvalidArgs;
    }
    opts.width = size;
    opts.height = size;

    var cpu = try pipeline.Session.init(io, gpa, opts, null);
    defer cpu.deinit();
    var cond = try cpu.encode(gpa, opts.prompt, .{});
    defer cond.deinit(gpa);
    const sigmas = try cpu.schedule(gpa, 8, opts.shift);
    defer gpa.free(sigmas);
    const ds = cpu.spatialDownscale();
    const lat_h = size / ds;
    const lat_w = size / ds;
    const n = cpu.latentChannels() * lat_h * lat_w;
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(7);
    for (x) |*v| v.* = prng.random().floatNorm(f32);
    cpu.scaleInitialNoise(x, sigmas[0], .{ .channels = cpu.latentChannels(), .h = lat_h, .w = lat_w });
    // A mid-schedule sigma, where the prediction carries the most structure.
    const sigma = sigmas[sigmas.len / 2];

    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    {
        var den = try cpu.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
        defer den.deinit(gpa);
        try den.predict(gpa, want, x, sigma, null);
    }
    const got = try gpa.alloc(f32, n);
    defer gpa.free(got);
    var dev_act: pipeline.Act = undefined;
    {
        var dopts = opts;
        dopts.backend = device;
        var dev = try pipeline.Session.init(io, gpa, dopts, null);
        defer dev.deinit();
        dev.setDials(null, dit_act, null);
        // The device's own conditioning, on its own row: the denoiser above reads the
        // CPU's, so this is where a text-encoder sidecar on the device is checked.
        {
            var dc = try dev.encode(gpa, opts.prompt, .{});
            defer dc.deinit(gpa);
            try stdout.print("conditioning: {t} vs cpu encoder rel L2 {e:.4}", .{ device, relL2f(cond.data, dc.data) });
            if (cond.pooled) |cp| if (dc.pooled) |dp| try stdout.print(", pooled {e:.4}", .{relL2f(cp, dp)});
            try stdout.print("\n", .{});
        }
        var den = try dev.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
        defer den.deinit(gpa);
        try den.predict(gpa, got, x, sigma, null);
        dev_act = dev.actFor(.denoiser);
    }
    var bad = [2]usize{ 0, 0 };
    var mag = [2]f32{ 0, 0 };
    for ([2][]const f32{ want, got }, 0..) |side, si| for (side) |v| {
        if (!std.math.isFinite(v)) bad[si] += 1 else mag[si] = @max(mag[si], @abs(v));
    };
    try stdout.print("{s}: {t} at {t} vs cpu denoiser at sigma {d:.3}, {d}x{d}: rel L2 {e:.4}\n", .{ std.fs.path.stem(args[0]), device, dev_act, sigma, size, size, relL2f(want, got) });
    if (bad[0] + bad[1] != 0) try stdout.print("  non-finite: cpu {d}/{d} (max |v| {d:.3}), {t} {d}/{d} (max |v| {d:.3})\n", .{ bad[0], n, mag[0], device, bad[1], n, mag[1] });
    try stdout.flush();
}

/// `cond-diff <ckpt A> <ckpt B> [--prompt P]... [--te-int-gemm M]`: the same prompts
/// through each checkpoint's own text encoder on CUDA, and the rel L2 of B's
/// conditioning (and SDXL's pooled vector) against A's. What a quantized checkpoint is
/// judged on: a conditioning that has lost the prompt still renders a clean picture.
fn condDiff(io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const gpa = std.heap.smp_allocator;
    var prompts: std.ArrayList([]const u8) = .empty;
    defer prompts.deinit(gpa);
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--prompt")) {
            try prompts.append(gpa, args[i + 1]);
        } else if (std.mem.eql(u8, args[i], "--te-int-gemm")) {
            TensorPencil.models.dual_lin.encoder_int8 = std.meta.stringToEnum(TensorPencil.models.lin_cuda.IntGemm, args[i + 1]) orelse return error.InvalidArgs;
        } else return error.InvalidArgs;
    }
    if (prompts.items.len == 0) {
        try prompts.append(gpa, "a red fox sitting on a mossy rock in a misty pine forest, morning light");
        try prompts.append(gpa, "an old lighthouse keeper repairing a brass lantern, oil painting, warm interior");
    }

    // A's conditionings, held while B loads: one session at a time.
    const Got = struct { data: []f32, pooled: ?[]f32 };
    const held = try gpa.alloc(Got, prompts.items.len);
    defer gpa.free(held);
    for ([2][]const u8{ args[0], args[1] }, 0..) |path, side| {
        const opts: pipeline.Options = .{ .prompt = prompts.items[0], .dit_path = path, .backend = .cuda, .preview = false };
        var sess = try pipeline.Session.init(io, gpa, opts, null);
        defer sess.deinit();
        for (prompts.items, held) |prompt, *h| {
            var cond = try sess.encode(gpa, prompt, .{});
            defer cond.deinit(gpa);
            if (side == 0) {
                h.* = .{ .data = try gpa.dupe(f32, cond.data), .pooled = if (cond.pooled) |p| try gpa.dupe(f32, p) else null };
                continue;
            }
            defer {
                gpa.free(h.data);
                if (h.pooled) |p| gpa.free(p);
            }
            if (cond.data.len != h.data.len) {
                try stdout.print("conditioning shapes differ: {d} vs {d}\n", .{ h.data.len, cond.data.len });
                return error.ShapeMismatch;
            }
            try stdout.print("{s}\n  conditioning rel L2 {d:.5}", .{ prompt, relL2f(h.data, cond.data) });
            if (h.pooled) |p| if (cond.pooled) |q| try stdout.print("   pooled rel L2 {d:.5}", .{relL2f(p, q)});
            try stdout.print("\n", .{});
        }
    }
    try stdout.print("text encoder int gemm: {t}\n", .{TensorPencil.models.dual_lin.encoderInt8()});
    try stdout.flush();
}

const LinTestW = struct { name: []const u8, w: TensorPencil.ops.matmul.Weight };

/// The synthetic weights `lin-cuda-test` and `lin-vk-test` run: one per storage form,
/// each built the way its loader would hand it over from the f32 `wf`, then a set whose
/// rows are off the GEMMs' 128 tile.
fn linTestWeights(arena: std.mem.Allocator, rnd: std.Random, wf: []const f32, rows: usize, cols: usize, row_scale: []f32) ![]LinTestW {
    const ops = TensorPencil.ops;
    const dtype = TensorPencil.dtype;
    const Weight = ops.matmul.Weight;
    var ws: std.ArrayList(LinTestW) = .empty;
    try ws.append(arena, .{ .name = "f32", .w = Weight.fromF32(wf, rows, cols) });
    {
        const b = try arena.alloc(u8, rows * cols * 2);
        dtype.f32ToF16Row(wf, b);
        try ws.append(arena, .{ .name = "f16", .w = Weight.init(b, .f16, rows, cols) });
        const c = try arena.alloc(u8, rows * cols * 2);
        dtype.f32ToBf16Row(wf, c);
        try ws.append(arena, .{ .name = "bf16", .w = Weight.init(c, .bf16, rows, cols) });
    }
    {
        // e4m3 bytes with exponents 4..8, magnitudes 2^-3..2^1.
        const b = try arena.alloc(u8, rows * cols);
        for (b) |*v| v.* = (rnd.int(u8) & 0x87) | (@as(u8, rnd.intRangeAtMost(u8, 4, 8)) << 3);
        var w = Weight.init(b, .f8_e4m3, rows, cols);
        w.scale = 0.01;
        try ws.append(arena, .{ .name = "fp8", .w = w });
    }
    {
        // Every e4m3 code but the two NaNs, subnormals included, as a real checkpoint has.
        const b = try arena.alloc(u8, rows * cols);
        for (b) |*v| {
            v.* = rnd.int(u8);
            while (v.* & 0x7f == 0x7f) v.* = rnd.int(u8);
        }
        var w = Weight.init(b, .f8_e4m3, rows, cols);
        // A real checkpoint's per-tensor scale, which folded into an f16 activation
        // takes it below f16's smallest normal.
        w.scale = 4e-4;
        try ws.append(arena, .{ .name = "fp8 wide", .w = w });
    }
    {
        const b = try arena.alloc(u8, rows * cols);
        for (b) |*v| v.* = @bitCast(rnd.intRangeAtMost(i8, -127, 127));
        var w = Weight.init(b, .i8, rows, cols);
        w.row_scale = row_scale;
        w.convrot = ops.convrot.group_size;
        try ws.append(arena, .{ .name = "int8 convrot", .w = w });
        w.convrot = 0;
        try ws.append(arena, .{ .name = "int8 plain", .w = w });
    }
    {
        const b = try arena.alloc(u8, rows * cols / 2);
        rnd.bytes(b);
        var w = Weight.init(b, .i4, rows, cols);
        w.row_scale = row_scale;
        w.convrot = ops.convrot.group_size;
        try ws.append(arena, .{ .name = "int4 convrot", .w = w });
    }
    {
        const group = 64;
        const b = try arena.alloc(u8, rows * cols / 2);
        rnd.bytes(b);
        const s_rel = try arena.alloc(u8, rows * cols / group);
        for (s_rel) |*v| v.* = 0x30 | (rnd.int(u8) & 0x07);
        const levels = try arena.create(ops.w4a8.Levels);
        levels.* = ops.w4a8.Levels.init(null);
        const meta = try arena.create(ops.w4a8.Meta);
        meta.* = .{ .s_rel = s_rel, .codes = .{ .w4 = levels }, .group_size = group };
        var w = Weight.init(b, .w4a8, rows, cols);
        w.row_scale = row_scale;
        w.w4a8 = meta;
        w.convrot = ops.convrot.group_size;
        try ws.append(arena, .{ .name = "w4a8", .w = w });
    }
    {
        const group = 64;
        const b = try arena.alloc(u8, rows * cols * 3 / 4);
        rnd.bytes(b);
        const s_rel = try arena.alloc(u8, rows * cols / group);
        for (s_rel) |*v| v.* = 0x28 | (rnd.int(u8) & 0x07);
        const levels = try arena.create(ops.w4a8.Levels6);
        levels.* = ops.w4a8.Levels6.init();
        const meta = try arena.create(ops.w4a8.Meta);
        meta.* = .{ .s_rel = s_rel, .codes = .{ .w6 = levels }, .group_size = group };
        var w = Weight.init(b, .w6a8, rows, cols);
        w.row_scale = row_scale;
        w.w4a8 = meta;
        w.convrot = ops.convrot.group_size;
        try ws.append(arena, .{ .name = "w6a8", .w = w });
    }
    {
        const b = try arena.alloc(u8, rows * cols / 2);
        rnd.bytes(b);
        const scales = try arena.alloc(u8, rows * ops.nvfp4.Meta.blocks(cols));
        for (scales) |*v| v.* = 0x30 | (rnd.int(u8) & 0x07);
        const levels = try arena.create(ops.nvfp4.Levels);
        levels.* = ops.nvfp4.Levels.init(0.01);
        const meta = try arena.create(ops.nvfp4.Meta);
        meta.* = .{ .scales = scales, .levels = levels };
        var w = Weight.init(b, .nvfp4, rows, cols);
        w.nvfp4 = meta;
        try ws.append(arena, .{ .name = "nvfp4", .w = w });
    }
    for ([_]TensorPencil.dtype.DType{ .q4_k, .q5_k, .q8_0, .q6_k, .q3_k, .q4_0, .iq4_nl, .iq4_xs, .q2_k }) |dt| {
        const b = try arena.alloc(u8, dt.storageBytes(rows * cols));
        _ = try TensorPencil.quants.quantizeChunk(dt, wf, b, rows, cols, null);
        try ws.append(arena, .{ .name = @tagName(dt), .w = Weight.init(b, dt, rows, cols) });
    }
    // Rows off the GEMMs' 128 tile, which the hand kernels pad to and compact back.
    const rows_odd = 200;
    // The odd-row case cuts its rows out of `wf`; a caller with fewer would read past it.
    if (rows < rows_odd) return error.InvalidArgs;
    {
        const src = wf[0 .. rows_odd * cols];
        const b = try arena.alloc(u8, rows_odd * cols * 2);
        dtype.f32ToF16Row(src, b);
        try ws.append(arena, .{ .name = "f16 odd", .w = Weight.init(b, .f16, rows_odd, cols) });
        const c = try arena.alloc(u8, rows_odd * cols * 2);
        dtype.f32ToBf16Row(src, c);
        try ws.append(arena, .{ .name = "bf16 odd", .w = Weight.init(c, .bf16, rows_odd, cols) });
        // Its own bytes: the device weight caches key on the host pointer, and a slice
        // of the 256-row weight would be served that weight's upload.
        var w = ws.items[3].w;
        w.bytes = try arena.dupe(u8, w.bytes[0 .. rows_odd * cols]);
        w.rows = rows_odd;
        try ws.append(arena, .{ .name = "fp8 odd", .w = w });
        for ([_]TensorPencil.dtype.DType{ .q4_k, .q3_k }) |dt| {
            const q = try arena.alloc(u8, dt.storageBytes(rows_odd * cols));
            _ = try TensorPencil.quants.quantizeChunk(dt, src, q, rows_odd, cols, null);
            try ws.append(arena, .{ .name = if (dt == .q4_k) "q4_k odd" else "q3_k odd", .w = Weight.init(q, dt, rows_odd, cols) });
        }
        // Off the weight-only pack's 32-row tile too.
        for ([_][]const u8{ "int8 convrot", "int4 convrot" }, [_][]const u8{ "int8 odd", "int4 odd" }) |from, name| {
            var iw = for (ws.items) |it| {
                if (std.mem.eql(u8, it.name, from)) break it.w;
            } else unreachable;
            iw.bytes = try arena.dupe(u8, iw.bytes[0 .. iw.bytes.len / iw.rows * rows_odd]);
            iw.row_scale = try arena.dupe(f32, iw.row_scale.?[0..rows_odd]);
            iw.rows = rows_odd;
            try ws.append(arena, .{ .name = name, .w = iw });
        }
    }

    return ws.items;
}

/// `y1 - y0` must be the bias to rounding whatever the route's own accuracy, and `y0`
/// within a route's quantization of the CPU GEMM. Prints the row; false on a failure.
/// `store_tol` is the rounding a 16-bit activation storage adds to both checks (0 for f32).
fn linTestScore(stdout: *Io.Writer, name: []const u8, label: []const u8, r: TensorPencil.models.dual_lin.Route, y0: []const f32, y1: []const f32, want: []const f32, bias: []const f32, mm: usize, n: usize, store_tol: f64) !bool {
    var e_num: f64 = 0;
    var b_den: f64 = 0;
    var c_num: f64 = 0;
    var c_den: f64 = 0;
    for (0..mm) |i| for (0..n) |j| {
        const d = @as(f64, y1[i * n + j]) - y0[i * n + j] - bias[j];
        e_num += d * d;
        b_den += @as(f64, bias[j]) * bias[j];
        const g = @as(f64, y0[i * n + j]) - want[i * n + j];
        c_num += g * g;
        c_den += @as(f64, want[i * n + j]) * want[i * n + j];
    };
    const bias_rel = @sqrt(e_num / b_den);
    const cpu_rel = @sqrt(c_num / c_den);
    const cpu_tol: f64 = switch (r.prep()) {
        .i4 => 0.3,
        .i8, .mmq, .q8 => 0.05,
        .none => 0.02,
    };
    const ok = bias_rel <= 1e-4 + store_tol and cpu_rel <= cpu_tol + store_tol;
    try stdout.print("{s:<13} {s:<15} route {t:<13} bias {e:.2}  vs cpu {d:.5} (tol {d})  {s}\n", .{
        name, label, r, bias_rel, cpu_rel, cpu_tol, if (ok) "ok" else "FAIL",
    });
    try stdout.flush();
    return ok;
}

/// A quantizing route against the CPU GEMM fed the activation that route's prep makes
/// (`ops.matmul.act_quant`). Against the exact GEMM a wrong quantizer hides inside the
/// format's own error; against this one it is all that is left.
fn linTestEmulated(stdout: *Io.Writer, io: Io, gpa: std.mem.Allocator, arena: std.mem.Allocator, emu: anytype, x: []const f32, m: usize, w: TensorPencil.ops.matmul.Weight, got: []const f32) !bool {
    const want = try arena.alloc(f32, m * w.rows);
    emu.install();
    defer TensorPencil.ops.matmul.act_quant = null;
    try TensorPencil.ops.matmul.matmul(io, gpa, want, x, m, w, null);
    const rel = relL2f(want, got);
    const ok = rel <= 1e-3;
    try stdout.print("{s:<29} vs emulated activation quant {e:.2} (tol 1e-3)  {s}\n", .{ "", rel, if (ok) "ok" else "FAIL" });
    return ok;
}

/// `lin-vk-test`: `lin-cuda-test` over `lin_gpu`, every route this device takes, with
/// and without a bias, f32 and 16-bit activations in and out, against the CPU GEMM.
/// Exits 1 on any failure.
/// `shape` is { m, rows, cols } of the main weight set (`--shape M ROWS COLS`).
fn linVkTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, shape: [3]usize) !void {
    const gpu_mod = TensorPencil.gpu;
    const ops = TensorPencil.ops;
    const lin_gpu = TensorPencil.models.lin_gpu;
    const gpa = std.heap.smp_allocator;

    const ctx = gpu_mod.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        std.process.exit(1);
    };
    defer ctx.deinit();
    try stdout.print("== lin-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});
    try ctx.writeCoopStatus(stdout);

    const m, const rows, const cols = shape;
    try stdout.print("m {d}, rows {d}, cols {d}\n", .{ m, rows, cols });
    const mpad = std.mem.alignForward(usize, m, 128);
    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();
    const wf = try arena.alloc(f32, rows * cols);
    for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.03;
    const x = try arena.alloc(f32, mpad * cols);
    @memset(x, 0);
    for (x[0 .. m * cols]) |*v| v.* = rnd.floatNorm(f32);
    const row_scale = try arena.alloc(f32, rows);
    for (row_scale) |*v| v.* = 0.0005 + 0.0005 * rnd.float(f32);
    const ws = try linTestWeights(arena, rnd, wf, rows, cols, row_scale);

    var l = try lin_gpu.Lin.init(gpa, ctx, rows, 16, 1);
    defer l.deinit();

    const Dt = gpu_mod.context.Dt;
    const Arm = struct { label: []const u8, dense: TensorPencil.models.dual_lin.DenseCompute = .native, act16: TensorPencil.models.dual_lin.Act16 = .f16, int8: TensorPencil.models.dual_lin.IntGemm = .w8a8, llm_m: usize = 0, input: Dt = .f32, store: Dt = .f32, staged: bool = false };
    const arms = [_]Arm{
        .{ .label = "llm m=1", .llm_m = 1 },
        .{ .label = "llm m=3", .llm_m = 3 },
        .{ .label = "llm m=5", .llm_m = 5 },
        .{ .label = "llm m=154", .llm_m = m },
        .{ .label = "native" },
        .{ .label = "dense f16", .dense = .f16 },
        .{ .label = "act bf16", .act16 = .bf16 },
        .{ .label = "weight-only", .int8 = .weight_only },
        .{ .label = "weight-only f16", .int8 = .weight_only_f16 },
        .{ .label = "weight-only f32", .int8 = .weight_only_f32 },
        // 16-bit activations: the cooperative GEMMs cast at the buffer's own dtype, every
        // other route stages through f32.
        .{ .label = "store f16", .store = .f16 },
        .{ .label = "store bf16", .store = .bf16 },
        .{ .label = "x f16", .input = .f16 },
        .{ .label = "x bf16", .input = .bf16 },
        .{ .label = "x+store f16", .input = .f16, .store = .f16 },
        .{ .label = "x+store bf16", .input = .bf16, .store = .bf16 },
        .{ .label = "llm m=1 x f16", .llm_m = 1, .input = .f16 },
        // fp8 against an activation the caller staged (`prepStaged`), as krea2's fused
        // norm does, into f32, f16 from f16 accumulators, and bf16 through f32.
        .{ .label = "staged", .staged = true },
        .{ .label = "staged f16", .staged = true, .store = .f16 },
        .{ .label = "staged bf16", .staged = true, .store = .bf16 },
    };

    var xd = try ctx.tensorCreate(mpad * cols * 4);
    defer ctx.tensorDestroy(&xd);
    try ctx.tensorUpload(xd, std.mem.sliceAsBytes(x));
    var yd = try ctx.tensorCreate(mpad * rows * 4);
    defer ctx.tensorDestroy(&yd);
    const y16 = try arena.alloc(u8, mpad * rows * 2);
    var x16d = try ctx.tensorCreate(mpad * cols * 2);
    defer ctx.tensorDestroy(&x16d);
    // x at each 16-bit dtype, for the arms that read one.
    var xin = [2]gpu_mod.context.DeviceBuffer{ try ctx.tensorCreate(mpad * cols * 2), try ctx.tensorCreate(mpad * cols * 2) };
    defer for (&xin) |*b| ctx.tensorDestroy(b);
    {
        const h = try arena.alloc(u8, mpad * cols * 2);
        TensorPencil.dtype.f32ToF16Row(x, h);
        try ctx.tensorUpload(xin[0], h);
        TensorPencil.dtype.f32ToBf16Row(x, h);
        try ctx.tensorUpload(xin[1], h);
    }
    const y0 = try arena.alloc(f32, mpad * rows);
    const y1 = try arena.alloc(f32, mpad * rows);
    const want = try arena.alloc(f32, m * rows);

    var failures: usize = 0;
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    for (ws, 0..) |c, wi| {
        const n = c.w.rows;
        const want_c = want[0 .. m * n];
        try ops.matmul.matmul(io, gpa, want_c, x[0 .. m * cols], m, c.w, null);
        var rms: f64 = 0;
        for (want_c) |v| rms += @as(f64, v) * v;
        rms = @sqrt(rms / @as(f64, @floatFromInt(want_c.len)));
        // A fresh array per weight, kept to the end: the device caches key a bias on
        // its host pointer, so a refilled one is served its first contents.
        const bias = try arena.alloc(f32, n);
        for (bias) |*v| v.* = @floatCast(rms * rnd.floatNorm(f64));
        for (arms) |arm| {
            const mm = if (arm.llm_m != 0) arm.llm_m else m;
            const use: lin_gpu.Use = if (arm.llm_m != 0) .llm(mm) else .{ .blockq = .auto, .act16 = arm.act16, .int8 = arm.int8 };
            var p = switch (lin_gpu.check(ctx, &.{c.w}, use)) {
                .ok => |p| p,
                .refused => continue,
            };
            if (arm.llm_m == 0) p.dense = arm.dense;
            const r = l.routeAt(p, c.w, mm) orelse continue;
            if (arm.staged and l.stageScale(p, &.{c.w}, mm) == null) continue;
            const key = @as(u64, @intFromEnum(arm.input)) << 44 | @as(u64, @intFromBool(arm.staged)) << 42 | @as(u64, @intFromEnum(arm.store)) << 40 | @as(u64, @intFromEnum(r)) << 16 | @as(u64, @intFromBool(n != rows)) << 15 | @as(u64, @intFromBool(c.w.convrot != 0)) << 14 | @as(u64, @intFromBool(mm > 4)) << 13 | @as(u64, @intFromEnum(c.w.dtype)) << 2 | @as(u64, wi) << 48;
            if ((try seen.getOrPut(arena, key)).found_existing) continue;
            try lin_gpu.presize(ctx, p, &.{c.w});
            for ([2]?[]const f32{ null, bias }, [2][]f32{ y0, y1 }) |b, dst| {
                const g = if (arm.staged) blk: {
                    const sc = c.w.scale;
                    const pairs = mpad * cols / 2;
                    try ctx.opElt(.f32_to_h16, xd, null, null, x16d, .{ .u0 = @intCast(pairs), .u1 = @intCast(mm * cols), .f0 = sc }, pairs, 1, 1);
                    break :blk l.prepStaged(p, x16d.as(.f16), mm, cols, &.{c.w}, sc);
                } else try l.prep(p, switch (arm.input) {
                    .f32 => xd,
                    .f16 => xin[0].as(.f16),
                    .bf16 => xin[1].as(.bf16),
                }, mm, cols, &.{c.w});
                try l.gemm(p, g, yd.as(arm.store), c.w, .{ .bias = b, .m_pad = mpad, .acc16 = arm.staged });
                switch (arm.store) {
                    .f32 => try ctx.tensorDownload(yd, std.mem.sliceAsBytes(dst)),
                    .f16 => {
                        try ctx.tensorDownload(try ctx.view(yd, 0, y16.len), y16);
                        TensorPencil.dtype.f16ToF32Row(y16, dst, 1.0);
                    },
                    .bf16 => {
                        try ctx.tensorDownload(try ctx.view(yd, 0, y16.len), y16);
                        TensorPencil.dtype.bf16ToF32Row(y16, dst, 1.0);
                    },
                }
            }
            const tolOf = struct {
                fn f(dt: Dt) f64 {
                    return switch (dt) {
                        .f32 => 0,
                        .f16 => 2e-3,
                        .bf16 => 1.5e-2,
                    };
                }
            }.f;
            const store_tol = tolOf(arm.store) + tolOf(arm.input);
            if (!try linTestScore(stdout, c.name, arm.label, r, y0, y1, want_c, bias, mm, n, store_tol)) failures += 1;
            if (arm.input == .f32 and arm.store == .f32 and !arm.staged and lin_gpu.actQuant(ctx, p, c.w, mm).bits != 0) {
                if (!try linTestEmulated(stdout, io, gpa, arena, ActEmu(*gpu_mod.Context, lin_gpu.actQuant){ .dev = ctx, .plan = p, .lins = &.{c.w} }, x[0 .. mm * cols], mm, c.w, y0[0 .. mm * n])) failures += 1;
            }
            // A grouped GEMV is a speculative verify's route, which must reproduce decode
            // exactly: each row against a one-row run, bit for bit.
            if (arm.input == .f32) switch (r) {
                .gemv_dualn, .gemv_dualn_q8, .gemv_bf16n, .gemv_fp8n, .gemv_f32n => for (0..mm) |t| {
                    const g = try l.prep(p, try ctx.view(xd, t * cols * 4, cols * 4), 1, cols, &.{c.w});
                    try l.gemm(p, g, yd, c.w, .{ .m_pad = mpad });
                    try ctx.tensorDownload(try ctx.view(yd, 0, n * 4), std.mem.sliceAsBytes(y1[0..n]));
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(y1[0..n]), std.mem.sliceAsBytes(y0[t * n ..][0..n]))) {
                        try stdout.print("  FAIL {s} {s} {t}: row {d} differs from a one-row run\n", .{ c.name, arm.label, r, t });
                        failures += 1;
                        break;
                    }
                },
                else => {},
            };
        }
    }
    failures += try attnRowsGroupedCheck(ctx, arena, stdout);
    failures += try fusedPrepCheck(ctx, io, arena, stdout);
    try stdout.print("{s}\n", .{if (failures == 0) "OK" else "FAILED"});
    try stdout.flush();
    if (failures != 0) std.process.exit(1);
}

/// `vk-wo-bench`: one int8 convrot linear three ways through `lin_gpu`, alternating in
/// one process: W8A8, weight-only W8A16 (the pack, then the f16 GEMM) and a dense f16
/// weight of the same shape on that f16 GEMM. W8A16 less dense f16 is what the pack costs;
/// W8A8 against dense f16 is the two GEMMs' own rates.
fn vkWoBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, m: usize) !void {
    const lin_gpu = TensorPencil.models.lin_gpu;
    const Weight = TensorPencil.ops.matmul.Weight;
    const gpa = std.heap.smp_allocator;
    const ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== vk-wo-bench ==\nvulkan device: {s}, m = {d}\n", .{ ctx.deviceName(), m });
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const mpad = std.mem.alignForward(usize, m, 128);
    // krea2's (6144, 16384) and Mage-Flow's (3072, 12288) block linears, rows x cols.
    for ([_][2]usize{ .{ 6144, 6144 }, .{ 16384, 6144 }, .{ 6144, 16384 }, .{ 3072, 3072 }, .{ 12288, 3072 }, .{ 3072, 12288 } }) |shape| {
        const rows = shape[0];
        const cols = shape[1];
        const ib = try arena.alloc(u8, rows * cols);
        rnd.bytes(ib);
        const scale = try arena.alloc(f32, rows);
        @memset(scale, 1e-3);
        var wi = Weight.init(ib, .i8, rows, cols);
        wi.row_scale = scale;
        wi.convrot = TensorPencil.ops.convrot.group_size;
        const wf32 = try arena.alloc(f32, rows * cols);
        for (wf32) |*v| v.* = rnd.floatNorm(f32) * 0.03;
        const fb = try arena.alloc(u8, rows * cols * 2);
        TensorPencil.dtype.f32ToF16Row(wf32, fb);
        const wf = Weight.init(fb, .f16, rows, cols);
        const x = try arena.alloc(f32, mpad * cols);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        var xd = try ctx.tensorCreate(x.len * 4);
        defer ctx.tensorDestroy(&xd);
        try ctx.tensorUpload(xd, std.mem.sliceAsBytes(x));
        var yd = try ctx.tensorCreate(mpad * rows * 4);
        defer ctx.tensorDestroy(&yd);
        var l = try lin_gpu.Lin.init(gpa, ctx, rows, 0, 0);
        defer l.deinit();

        const Arm = struct { label: []const u8, w: Weight, int8: TensorPencil.models.dual_lin.IntGemm };
        const arms = [_]Arm{ .{ .label = "W8A8", .w = wi, .int8 = .w8a8 }, .{ .label = "W8A16", .w = wi, .int8 = .weight_only_f16 }, .{ .label = "dense f16", .w = wf, .int8 = .w8a8 } };
        var plans: [arms.len]lin_gpu.Plan = undefined;
        for (arms, &plans) |a, *p| {
            p.* = switch (lin_gpu.check(ctx, &.{a.w}, .{ .blockq = .auto, .int8 = a.int8 })) {
                .ok => |pl| pl,
                .refused => return error.Unsupported,
            };
            try lin_gpu.presize(ctx, p.*, &.{a.w});
            const g = try l.prep(p.*, xd, m, cols, &.{a.w});
            try l.gemm(p.*, g, yd, a.w, .{ .m_pad = mpad });
        }
        var best: [arms.len]f64 = @splat(std.math.inf(f64));
        for (0..5) |_| for (arms, plans, 0..) |a, p, ai| {
            try ctx.beginBatch();
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            for (0..5) |_| {
                const g = try l.prep(p, xd, m, cols, &.{a.w});
                try l.gemm(p, g, yd, a.w, .{ .m_pad = mpad });
            }
            try ctx.endBatch();
            best[ai] = @min(best[ai], @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 5e6);
        };
        const flop: f64 = @floatFromInt(2 * m * rows * cols);
        try stdout.print("{d:>5}x{d:<5}", .{ rows, cols });
        for (arms, best) |a, b| try stdout.print("  {s} {d:6.2} ms ({d:5.1} TF/s)", .{ a.label, b, flop / (b * 1e9) });
        try stdout.print("  pack ~{d:.2} ms\n", .{best[1] - best[2]});
        try stdout.flush();
    }
}

/// `vk-prep-bench`: the int8 activation prep at each model width, the fused kernel and
/// the 3-pass chain alternating in one process (this card's clock drifts between runs).
/// GB/s counts the f32 read and the int8 write once.
fn vkPrepBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, m: usize) !void {
    const C = TensorPencil.gpu.context;
    const ctx = C.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    defer C.force_i8_prep_3pass = false;
    try stdout.print("== vk-prep-bench ==\nvulkan device: {s}, m = {d}\n", .{ ctx.deviceName(), m });
    for ([_]usize{ 2048, 2816, 3072, 3840, 4096, 5376, 6144, 7168, 8192, 9728, 12288, 14336, 16384, 25600 }) |cols| {
        if (!ctx.hasFusedI8Prep(cols)) continue;
        var xd = try ctx.tensorCreate(m * cols * 4);
        defer ctx.tensorDestroy(&xd);
        var best = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
        for (0..2) |mode| {
            C.force_i8_prep_3pass = mode == 1;
            try ctx.opI8PrepR(xd, m, cols, true);
        }
        for (0..5) |_| for (0..2) |mode| {
            C.force_i8_prep_3pass = mode == 1;
            try ctx.beginBatch();
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            for (0..10) |_| try ctx.opI8PrepR(xd, m, cols, true);
            try ctx.endBatch();
            const ns: f64 = @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0);
            best[mode] = @min(best[mode], ns / 10e3);
        };
        const bytes: f64 = @floatFromInt(m * cols * 5);
        try stdout.print("cols {d:>5}  fused {d:8.1} us ({d:5.0} GB/s)   3-pass {d:8.1} us   x{d:.2}\n", .{ cols, best[0], bytes / (best[0] * 1e3), best[1], best[1] / best[0] });
        try stdout.flush();
    }
}

/// The fused int8 prep (`coopmat.buildFusedPrepI8`) at each width it builds for: a row past
/// f16's range and a row far below it against the emulated GEMM, each row on its own,
/// since one row's error would swamp the rest.
fn fusedPrepCheck(ctx: *TensorPencil.gpu.Context, io: Io, arena: std.mem.Allocator, stdout: *Io.Writer) !usize {
    var failures: usize = 0;
    // Every model width this box runs, single-chunk and chunked: krea2, Anima, SenseNova,
    // Mage-Flow (3072), Z-Image (3840), H3's trunk (5376, 7168, 14336) and its 32B encoder
    // (25600); the Qwen3(-VL) 4B encoders' MLP (9728) and 11 groups (2816) take a padded plan.
    for ([_]usize{ 2048, 2816, 3072, 3840, 4096, 5376, 6144, 7168, 8192, 9728, 12288, 14336, 16384, 25600 }) |cols| {
        if (!ctx.hasFusedI8Prep(cols)) {
            try stdout.print("fused int8 prep {d}   no build (3-pass chain)\n", .{cols});
            continue;
        }
        failures += try fusedPrepCheckAt(ctx, io, arena, stdout, cols);
    }
    return failures;
}

fn fusedPrepCheckAt(ctx: *TensorPencil.gpu.Context, io: Io, arena: std.mem.Allocator, stdout: *Io.Writer, cols: usize) !usize {
    const lin_gpu = TensorPencil.models.lin_gpu;
    const gpa = std.heap.smp_allocator;
    const m = 37;
    const n = 256;
    const mpad = std.mem.alignForward(usize, m, 128);
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const wf = try arena.alloc(f32, n * cols);
    for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.03;
    const row_scale = try arena.alloc(f32, n);
    for (row_scale) |*v| v.* = 0.0005 + 0.0005 * rnd.float(f32);
    const w = for (try linTestWeights(arena, rnd, wf, n, cols, row_scale)) |c| {
        if (std.mem.eql(u8, c.name, "int8 convrot")) break c.w;
    } else unreachable;
    const x = try arena.alloc(f32, mpad * cols);
    @memset(x, 0);
    for (x[0 .. m * cols]) |*v| v.* = rnd.floatNorm(f32);
    for (x[3 * cols ..][0..cols]) |*v| v.* *= 3e4;
    for (x[5 * cols ..][0..cols]) |*v| v.* *= 1e-3;
    // An outlier channel at column 0: a padded plan's loads past the row read that
    // element, and the rotation maps a constant group to itself, so only a row whose
    // first value tops its rotated max shows a padding group that was not zeroed.
    x[7 * cols] = 60;

    const p = switch (lin_gpu.check(ctx, &.{w}, .{ .blockq = .auto })) {
        .ok => |p| p,
        .refused => return 0,
    };
    if (lin_gpu.actQuant(ctx, p, w, m).bits == 0) return 0;
    var l = try lin_gpu.Lin.init(gpa, ctx, n, 16, 1);
    defer l.deinit();
    var xd = try ctx.tensorCreate(x.len * 4);
    defer ctx.tensorDestroy(&xd);
    try ctx.tensorUpload(xd, std.mem.sliceAsBytes(x));
    var yd = try ctx.tensorCreate(mpad * n * 4);
    defer ctx.tensorDestroy(&yd);
    try lin_gpu.presize(ctx, p, &.{w});
    const g = try l.prep(p, xd, m, cols, &.{w});
    try l.gemm(p, g, yd, w, .{ .m_pad = mpad });
    const got = try arena.alloc(f32, mpad * n);
    try ctx.tensorDownload(yd, std.mem.sliceAsBytes(got));

    const want = try arena.alloc(f32, m * n);
    const emu: ActEmu(*TensorPencil.gpu.Context, lin_gpu.actQuant) = .{ .dev = ctx, .plan = p, .lins = &.{w} };
    emu.install();
    defer TensorPencil.ops.matmul.act_quant = null;
    try TensorPencil.ops.matmul.matmul(io, gpa, want, x[0 .. m * cols], m, w, null);
    var worst: f64 = 0;
    var nonfinite: usize = 0;
    for (0..m) |r| {
        for (got[r * n ..][0..n]) |v| {
            if (!std.math.isFinite(v)) nonfinite += 1;
        }
        worst = @max(worst, relL2f(want[r * n ..][0..n], got[r * n ..][0..n]));
    }
    const ok = nonfinite == 0 and worst <= 1e-3;
    try stdout.print("fused int8 prep {d}   worst row vs emulated activation quant {e:.2} (tol 1e-3)  {s}{s}\n", .{ cols, worst, if (ok) "ok" else "FAIL", if (nonfinite != 0) " (non-finite output)" else "" });
    return @intFromBool(!ok);
}

/// `opAttnRows`' grouped kernel against the per-head one, bit for bit, for every KV
/// format and group size, a decode row and a batch: the steppers mix the two freely.
fn attnRowsGroupedCheck(ctx: *TensorPencil.gpu.Context, arena: std.mem.Allocator, stdout: *Io.Writer) !usize {
    const Ctx = TensorPencil.gpu.Context;
    const hd = 128;
    const kv_heads = 2;
    const kv_len0 = 190;
    const nsplit = 8;
    const max_q = 5;
    var rnd = std.Random.DefaultPrng.init(7);
    const r = rnd.random();
    var failures: usize = 0;
    for ([_]usize{ 2, 4, 8 }) |group| {
        const heads = kv_heads * group;
        const kv = try arena.alloc(f32, (kv_len0 + max_q) * kv_heads * hd);
        const qh = try arena.alloc(f32, max_q * heads * hd);
        for (kv) |*v| v.* = r.floatNorm(f32);
        for (qh) |*v| v.* = r.floatNorm(f32);
        var kv_f = try ctx.tensorCreate(kv.len * 4);
        defer ctx.tensorDestroy(&kv_f);
        try ctx.tensorUpload(kv_f, std.mem.sliceAsBytes(kv));
        var q_d = try ctx.tensorCreate(qh.len * 4);
        defer ctx.tensorDestroy(&q_d);
        try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(qh));
        var scratch = try ctx.tensorCreate(max_q * heads * nsplit * (hd + 2) * 4);
        defer ctx.tensorDestroy(&scratch);
        var out = try ctx.tensorCreate(max_q * heads * hd * 4);
        defer ctx.tensorDestroy(&out);
        const got = [2][]f32{ try arena.alloc(f32, max_q * heads * hd), try arena.alloc(f32, max_q * heads * hd) };
        for ([_]TensorPencil.gpu.context.KvFmt{ .f32, .f16, .q8_0 }) |fmt| {
            const dt: TensorPencil.kv_cache.KvDtype = switch (fmt) {
                .f32 => .f32,
                .f16 => .f16,
                .q8_0 => .q8_0,
            };
            var cache = try ctx.tensorCreate(dt.sizeBytes(kv.len));
            defer ctx.tensorDestroy(&cache);
            switch (fmt) {
                .f32 => try ctx.tensorUpload(cache, std.mem.sliceAsBytes(kv)),
                .f16 => try ctx.opStoreKvF16(cache, 0, kv_f, 0, kv.len),
                .q8_0 => try ctx.opStoreKvQ8(cache, 0, kv_f, 0, kv.len),
            }
            const spans = [_]Ctx.AttnSpan{ .{}, .{ .window = 64, .ring = 128 }, .{ .window = 64, .ring = 128, .bidir_end = kv_len0 + max_q - 1 } };
            for (spans, [_][]const u8{ "causal", "window+ring", "+bidir" }) |sp, sp_name| for ([_]usize{ 1, 2, max_q }) |seq_q| {
                // A decode row takes the per-head kernel on these formats.
                if (seq_q == 1 and fmt != .f32) continue;
                const n = seq_q * heads * hd;
                for ([2]bool{ false, true }, got) |grouped, dst| {
                    Ctx.attn_rows_grouped = grouped;
                    try ctx.opAttnRows(q_d, cache, cache, out, scratch, kv_len0, seq_q, heads, kv_heads, hd, nsplit, 0.088, fmt, sp);
                    try ctx.tensorDownload(try ctx.view(out, 0, n * 4), std.mem.sliceAsBytes(dst[0..n]));
                }
                Ctx.attn_rows_grouped = true;
                const same = std.mem.eql(u8, std.mem.sliceAsBytes(got[0][0..n]), std.mem.sliceAsBytes(got[1][0..n]));
                try stdout.print("attn rows   group {d} {t:<5} {s:<11} q {d}  grouped {s}\n", .{ group, fmt, sp_name, seq_q, if (same) "bit-identical  ok" else "DIFFERS  FAIL" });
                if (!same) failures += 1;
            };
        }
    }
    return failures;
}

/// `lin-cuda-test [libs]`: every `lin_cuda` route on synthetic weights, each run with
/// and without a bias. `y_bias - y_plain` must equal the bias to rounding whatever the
/// route's own accuracy, which is what catches a route that drops or misplaces it; the
/// output is also held against the CPU GEMM at a tolerance set by what the route
/// quantizes, to catch one that returns garbage. Exits 1 on any failure.
fn linCudaTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, libs: bool) !void {
    const cuda = TensorPencil.gpu.cuda;
    const ops = TensorPencil.ops;
    const lin_cuda = TensorPencil.models.lin_cuda;
    const dtype = TensorPencil.dtype;
    const Weight = ops.matmul.Weight;
    const gpa = std.heap.smp_allocator;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        std.process.exit(1);
    };
    defer be.deinit();
    try stdout.print("== lin-cuda-test ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });

    // An out-of-memory eviction must not take a weight an op still reads: one held
    // across launches (a conv's bands), or one fetched since the last launch.
    {
        const blob = try arena.alloc(f32, 3 * (1 << 18));
        @memset(blob, 1);
        const wa = std.mem.sliceAsBytes(blob[0 .. 1 << 18]);
        const wb = std.mem.sliceAsBytes(blob[1 << 18 .. 2 << 18]);
        const wc = std.mem.sliceAsBytes(blob[2 << 18 ..]);
        var tick: cuda.backend.DeviceBuffer = .{};
        defer be.tensorDestroy(&tick);
        try be.ensureDeviceBuffer(&tick, 1024);
        const da = cuda.backend.DeviceBuffer{ .buf = try be.smallBuffer(wa), .mem = .null_handle, .size = 0 };
        _ = try be.smallBuffer(wc);
        try be.opAdd(tick, tick, 256);
        be.hold(da);
        _ = try be.smallBuffer(wb);
        _ = be.evictToFree(1 << 40);
        const held_ok = be.resident(wa) and be.resident(wb) and !be.resident(wc);
        be.unhold();
        try be.opAdd(tick, tick, 256);
        _ = try be.smallBuffer(wb);
        _ = be.evictToFree(1 << 40);
        const released_ok = !be.resident(wa);
        try be.endBatch();
        try stdout.print("eviction holds: held {s}, released {s}\n", .{ if (held_ok) "kept" else "LOST", if (released_ok) "evicted" else "KEPT" });
        if (!held_ok or !released_ok) {
            try stdout.flush();
            std.process.exit(1);
        }
    }

    // m is not a multiple of 128 (a CLIP prompt is 154 rows), so every padded route
    // writes past it and every unpadded one must not need to.
    const m = 154;
    const rows = 256;
    const cols = 1024;
    const mpad = std.mem.alignForward(usize, m, 128);
    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();

    const wf = try arena.alloc(f32, rows * cols);
    for (wf) |*v| v.* = rnd.floatNorm(f32) * 0.03;
    const x = try arena.alloc(f32, mpad * cols);
    @memset(x, 0);
    for (x[0 .. m * cols]) |*v| v.* = rnd.floatNorm(f32);
    const row_scale = try arena.alloc(f32, rows);
    for (row_scale) |*v| v.* = 0.0005 + 0.0005 * rnd.float(f32);

    const ws = try linTestWeights(arena, rnd, wf, rows, cols, row_scale);

    // Each weight under every plan that changes its route. The LLM plan's route also
    // moves with the row count: a GEMV at one row, a grouped one for a few, a GEMM beyond.
    const Arm = struct { label: []const u8, pol: lin_cuda.BlockQGemm = .auto, int8: lin_cuda.IntGemm = .w8a8, dense: lin_cuda.DenseCompute = .native, act16: @TypeOf(@as(lin_cuda.Plan, undefined).act16) = .f16, llm_m: usize = 0, input: cuda.backend.Dt = .f32, store: cuda.backend.Dt = .f32 };
    const arms = [_]Arm{
        .{ .label = "llm m=1", .llm_m = 1 },
        .{ .label = "llm m=3", .llm_m = 3 },
        .{ .label = "llm m=154", .llm_m = m },
        .{ .label = "native" },
        .{ .label = "dense f16", .dense = .f16 },
        .{ .label = "weight-only", .int8 = .weight_only },
        .{ .label = "weight-only f16", .int8 = .weight_only_f16 },
        .{ .label = "weight-only f32", .int8 = .weight_only_f32 },
        .{ .label = "bq int8", .pol = .int8 },
        .{ .label = "bq int4", .pol = .int4 },
        .{ .label = "bq bf16", .pol = .bf16 },
        .{ .label = "bq f16", .pol = .f16 },
        .{ .label = "bq mmq", .pol = .mmq },
        .{ .label = "act bf16", .act16 = .bf16 },
        // Activation STORAGE, in and out, on every route: the 16-bit ones read and write
        // it as it is, the rest stage through f32.
        .{ .label = "store f16", .input = .f16, .store = .f16 },
        .{ .label = "store bf16", .input = .bf16, .store = .bf16 },
        .{ .label = "store f16 dense", .dense = .f16, .input = .f16, .store = .f16 },
        .{ .label = "store bf16 dense", .dense = .f16, .input = .bf16, .store = .bf16 },
        .{ .label = "store bf16 bq8", .pol = .int8, .input = .bf16, .store = .bf16 },
        .{ .label = "x f16", .input = .f16 },
        .{ .label = "x bf16", .input = .bf16 },
        .{ .label = "y f16", .store = .f16 },
        .{ .label = "y bf16", .store = .bf16 },
        .{ .label = "llm m=1 x f16", .llm_m = 1, .input = .f16 },
    };

    var xd = try be.tensorCreate(mpad * cols * 4);
    defer be.tensorDestroy(&xd);
    try be.tensorUpload(xd, std.mem.sliceAsBytes(x));
    var yd = try be.tensorCreate(mpad * rows * 4);
    defer be.tensorDestroy(&yd);
    // `x` rounded to each 16-bit storage, and one 16-bit output, for the store arms.
    var xd16: [2]cuda.backend.DeviceBuffer = undefined;
    for (&xd16, [2]cuda.backend.Dt{ .f16, .bf16 }) |*b, dt| {
        const bytes = try arena.alloc(u8, mpad * cols * 2);
        if (dt == .f16) dtype.f32ToF16Row(x, bytes) else dtype.f32ToBf16Row(x, bytes);
        b.* = try be.tensorCreate(bytes.len);
        try be.tensorUpload(b.*, bytes);
        b.dt = dt;
    }
    defer for (&xd16) |*b| be.tensorDestroy(b);
    var yd16 = try be.tensorCreate(mpad * rows * 2);
    defer be.tensorDestroy(&yd16);
    const y16 = try arena.alloc(u8, mpad * rows * 2);
    const y0 = try arena.alloc(f32, mpad * rows);
    const y1 = try arena.alloc(f32, mpad * rows);
    const want = try arena.alloc(f32, m * rows);

    var failures: usize = 0;
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    for (ws, 0..) |c, wi| {
        const n = c.w.rows;
        const want_c = want[0 .. m * n];
        try ops.matmul.matmul(io, gpa, want_c, x[0 .. m * cols], m, c.w, null);
        var rms: f64 = 0;
        for (want_c) |v| rms += @as(f64, v) * v;
        rms = @sqrt(rms / @as(f64, @floatFromInt(want_c.len)));
        // A fresh array per weight, kept to the end: the device caches key a bias on
        // its host pointer, so a refilled one is served its first contents.
        const bias = try arena.alloc(f32, n);
        for (bias) |*v| v.* = @floatCast(rms * rnd.floatNorm(f64));
        for (arms) |arm| {
            const mm = if (arm.llm_m != 0) arm.llm_m else m;
            const use: lin_cuda.Use = if (arm.llm_m != 0) .llm(mm) else .{ .blockq = arm.pol };
            var p = lin_cuda.plan(&.{c.w}, use, "lin-cuda-test") catch continue;
            if (arm.llm_m == 0) {
                p = lin_cuda.withInt8(p, be, arm.int8);
                p.dense = arm.dense;
                p.act16 = arm.act16;
            }
            const r = lin_cuda.routeAt(p, c.w, mm) orelse continue;
            // One row per distinct route, shape and arm of the f16 dense dispatch.
            const key = @as(u64, @intFromEnum(arm.input)) << 42 | @as(u64, @intFromEnum(arm.store)) << 40 | @as(u64, @intFromEnum(r)) << 16 | @as(u64, @intFromBool(n != rows)) << 15 | @as(u64, @intFromBool(c.w.convrot != 0)) << 14 | @as(u64, @intFromEnum(c.w.dtype)) << 2 | @as(u64, @intFromBool(arm.act16 == .bf16)) << 1 | @intFromBool(arm.dense == .f16) | @as(u64, wi) << 48;
            if ((try seen.getOrPut(arena, key)).found_existing) continue;

            try lin_cuda.presize(be, p, &.{c.w});
            try lin_cuda.presizeRows(be, p, &.{c.w}, mm);
            const xin = if (arm.input == .f32) xd else xd16[@intFromBool(arm.input == .bf16)];
            const yout = if (arm.store == .f32) yd else yd16.as(arm.store);
            for ([2]?[]const f32{ null, bias }, [2][]f32{ y0, y1 }) |b, dst| {
                const g = try lin_cuda.prep(be, p, xin, mm, cols, &.{c.w});
                try lin_cuda.gemm(be, p, g, yout, c.w, b);
                if (arm.store == .f32) {
                    try be.tensorDownload(yd, std.mem.sliceAsBytes(dst));
                } else {
                    try be.tensorDownload(yd16, y16);
                    if (arm.store == .f16) dtype.f16ToF32Row(y16, dst, 1.0) else dtype.bf16ToF32Row(y16, dst, 1.0);
                }
            }
            const tolOf = struct {
                fn f(dt: cuda.backend.Dt) f64 {
                    return switch (dt) {
                        .f32 => 0,
                        .f16 => 2e-3,
                        .bf16 => 1.5e-2,
                    };
                }
            }.f;
            const store_tol = tolOf(arm.store) + tolOf(arm.input);
            if (!try linTestScore(stdout, c.name, arm.label, r, y0, y1, want_c, bias, mm, n, store_tol)) failures += 1;
            if (arm.input == .f32 and arm.store == .f32 and lin_cuda.actQuant(be, p, c.w, mm).bits != 0) {
                if (!try linTestEmulated(stdout, io, gpa, arena, ActEmu(*cuda.Backend, lin_cuda.actQuant){ .dev = be, .plan = p, .lins = &.{c.w} }, x[0 .. mm * cols], mm, c.w, y0[0 .. mm * n])) failures += 1;
            }
        }
    }
    // The conv weight resolver (`Backend.weightF16`) through the real device conv
    // entry, each storage form in each order, against the CPU conv of the same weights.
    // 64 channels make a 3x3 row 576 wide, so the q4_k case is flat-block.
    {
        const Conv2d = ops.conv.Conv2d;
        const ch = 12;
        const ci = 64;
        const co = 128;
        const xin = try arena.alloc(f32, ch * ch * ci);
        for (xin) |*v| v.* = rnd.floatNorm(f32);
        const cbias = try arena.alloc(f32, co);
        for (cbias) |*v| v.* = rnd.floatNorm(f32) * 0.1;
        var cx = try be.tensorCreate(xin.len * 4);
        defer be.tensorDestroy(&cx);
        try be.tensorUpload(cx, std.mem.sliceAsBytes(xin));
        var cy = try be.tensorCreate(ch * ch * co * 4);
        defer be.tensorDestroy(&cy);
        var patch: cuda.backend.DeviceBuffer = .{};
        defer be.tensorDestroy(&patch);
        const cwant = try arena.alloc(f32, ch * ch * co);
        const cgot = try arena.alloc(f32, ch * ch * co);
        for ([_]usize{ 3, 1 }) |k| {
            const n = co * ci * k * k;
            const tw = try arena.alloc(f32, n);
            for (tw) |*v| v.* = rnd.floatNorm(f32) * 0.05;
            const forms = [_]struct { name: []const u8, dt: TensorPencil.dtype.DType }{
                .{ .name = "f32", .dt = .f32 }, .{ .name = "f16", .dt = .f16 }, .{ .name = "bf16", .dt = .bf16 }, .{ .name = "q4_k", .dt = .q4_k },
            };
            for (forms) |fm| {
                const bytes = try arena.alloc(u8, fm.dt.storageBytes(n));
                switch (fm.dt) {
                    .f32 => @memcpy(bytes, std.mem.sliceAsBytes(tw)),
                    .f16 => dtype.f32ToF16Row(tw, bytes),
                    .bf16 => dtype.f32ToBf16Row(tw, bytes),
                    else => _ = try TensorPencil.quants.quantizeChunk(fm.dt, tw, bytes, n / 256, 256, null),
                }
                const torch: Conv2d = .{ .w = .{ .torch = .{ .w = Weight.init(bytes, fm.dt, co, ci * k * k) } }, .b = cbias, .co = co, .ci = ci, .k = k, .pad = k / 2 };
                // The repacked twin, of the weights the stored form actually holds.
                var repacked = torch;
                repacked.w = .{ .krsc = try torch.materialize(arena) };
                try ops.conv.conv2d(io, gpa, cwant, xin, ch, ch, torch);
                for ([2]Conv2d{ torch, repacked }, [2][]const u8{ "torch", "krsc" }) |cv, order| {
                    try conv_cuda.conv2d(be, &patch, &cy, &cx, ch, ch, cv, .stride1, null, 1.0);
                    try be.tensorDownload(cy, std.mem.sliceAsBytes(cgot));
                    const rel = relL2f(cwant, cgot);
                    const ok = rel <= 2e-3;
                    if (!ok) failures += 1;
                    try stdout.print("conv {d}x{d} {s:<5} {s:<5}  vs cpu {e:.3}  {s}\n", .{ k, k, fm.name, order, rel, if (ok) "ok" else "FAIL" });
                }
            }
        }
    }

    // The conv gathered straight into the GEMM operand against the f32 patch: both
    // hand the GEMM the same 16-bit values, so any differing byte is a gather defect.
    {
        const dual_table = TensorPencil.gpu.dual_table;
        const Conv2d = ops.conv.Conv2d;
        defer dual_table.force_conv_patch = false;
        const h = 11;
        const w = 7;
        const co = 128;
        var patch: cuda.backend.DeviceBuffer = .{};
        defer be.tensorDestroy(&patch);
        var cases: usize = 0;
        var differ: usize = 0;
        for ([_]usize{ 32, 12 }) |ci| {
            const wt = try arena.alloc(f32, co * ci * 9);
            for (wt) |*v| v.* = rnd.floatNorm(f32) * 0.1;
            const cb = try arena.alloc(f32, co);
            for (cb) |*v| v.* = rnd.floatNorm(f32);
            const xin = try arena.alloc(f32, h * w * ci);
            for (xin) |*v| v.* = rnd.floatNorm(f32) * 40;
            var x32 = try be.tensorCreate(xin.len * 4);
            defer be.tensorDestroy(&x32);
            try be.tensorUpload(x32, std.mem.sliceAsBytes(xin));
            const cv: Conv2d = .{ .w = .{ .krsc = wt }, .b = cb, .co = co, .ci = ci, .k = 3, .pad = 1 };
            for ([_]cuda.backend.Dt{ .f32, .f16, .bf16 }) |sdt| {
                var src = x32;
                var x16: cuda.backend.DeviceBuffer = .{};
                defer be.tensorDestroy(&x16);
                if (sdt != .f32) {
                    x16 = try be.tensorCreate(xin.len * 2);
                    x16.dt = sdt;
                    try be.opCast(x32, x16, xin.len, 0);
                    src = x16;
                }
                for ([_]cuda.backend.Dt{ .f32, .f16, .bf16 }) |ddt| for ([_]f32{ 1, 256 }) |div| for ([_]conv_cuda.SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
                    if (sdt == .f16 and div != 1) continue;
                    var c = cv;
                    if (mode == .stride2) c.stride = 2;
                    const oh: usize = switch (mode) {
                        .stride1 => h,
                        .stride2 => (h + 1) / 2,
                        .upsample2x => 2 * h,
                    };
                    const ow: usize = switch (mode) {
                        .stride1 => w,
                        .stride2 => (w + 1) / 2,
                        .upsample2x => 2 * w,
                    };
                    const bytes = oh * ow * co * ddt.bytes();
                    var outs: [2][]u8 = undefined;
                    for (0..2) |path| {
                        dual_table.force_conv_patch = path == 1;
                        var y = try be.tensorCreate(bytes);
                        defer be.tensorDestroy(&y);
                        y.dt = ddt;
                        conv_cuda.conv2d(be, &patch, &y, &src, h, w, c, mode, null, div) catch |err| {
                            try stdout.print("conv gather ci {d} src {t} dst {t} div {d} {t} {s}: {t}\n", .{ ci, sdt, ddt, div, mode, if (path == 1) "patch" else "gather", err });
                            try stdout.flush();
                            return err;
                        };
                        outs[path] = try arena.alloc(u8, bytes);
                        try be.tensorDownload(y, outs[path]);
                    }
                    cases += 1;
                    if (!std.mem.eql(u8, outs[0], outs[1])) {
                        differ += 1;
                        try stdout.print("conv gather ci {d} src {t} dst {t} div {d} {t}: DIFFERS from the f32 patch\n", .{ ci, sdt, ddt, div, mode });
                    }
                };
            }
        }
        failures += differ;
        try stdout.print("conv gather vs f32 patch: {d} of {d} cases bit-identical  {s}\n", .{ cases - differ, cases, if (differ == 0) "ok" else "FAIL" });
    }

    try stdout.print("{s}\n", .{if (failures == 0) "OK" else "FAILED"});
    try stdout.flush();
    if (failures != 0) std.process.exit(1);
}

/// `lora-cuda-bench [libs] [--rank R[,R...]]`: the LoRA sidecar through `lin_cuda.gemm`
/// on an int8 convrot base at DiT shapes, in arms: the base alone, then per rank the
/// bf16-A sidecar and the int8 A (`lin_cuda.lora_i8_a`). Numerics first: each arm's
/// delta against the host f32 sidecar, beside the base's own W8A8 error on the same
/// activation. Then timing, every arm of every rank interleaved in one process, so two
/// ranks compare within a round. Non-zero exit on a delta over its limit, a wrong-basis
/// A the check does not catch, or a nonzero pad row.
fn loraCudaBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, libs: bool, m: usize, ranks: []const usize) !void {
    const cuda = TensorPencil.gpu.cuda;
    const lora = TensorPencil.models.lora;
    const lin_cuda = TensorPencil.models.lin_cuda;
    const Weight = TensorPencil.ops.matmul.Weight;
    const page = std.heap.page_allocator;

    var be = (if (libs) cuda.Backend.initLibs(arena) else cuda.Backend.init(arena)) catch |err| {
        try stdout.print("cuda unavailable: {t}\n", .{err});
        return;
    };
    defer be.deinit();
    try stdout.print("== lora-cuda-bench ==\ncuda device: {s} (kernels: {t})\n", .{ be.deviceName(), be.kernels });
    defer lin_cuda.lora_i8_a = true;

    const max_ranks = 4;
    if (ranks.len == 0 or ranks.len > max_ranks) return error.InvalidArgument;
    // A rank is timed as the file ships it: `Sidecar.load` pads it, so this does too.
    for (ranks) |r| if (r == 0) return error.InvalidArgument;
    const Arm = struct { name: []const u8, rank: usize = 0, lora: bool, i8a: bool };
    var arm_buf: [1 + 2 * max_ranks]Arm = undefined;
    arm_buf[0] = .{ .name = "base alone", .lora = false, .i8a = false };
    for (ranks, 0..) |r, i| {
        arm_buf[1 + 2 * i] = .{ .name = "bf16 A", .rank = r, .lora = true, .i8a = false };
        arm_buf[2 + 2 * i] = .{ .name = "int8 A", .rank = r, .lora = true, .i8a = true };
    }
    const arms = arm_buf[0 .. 1 + 2 * ranks.len];
    const Shape = struct { in: usize, out: usize };
    // Square DiT widths and an MLP pair, then SenseNova's turbo LoRA targets
    // (q/o, k/v, gate/up, down).
    const shapes = [_]Shape{
        .{ .in = 4096, .out = 4096 }, .{ .in = 6144, .out = 6144 },  .{ .in = 6144, .out = 16384 }, .{ .in = 16384, .out = 6144 },
        .{ .in = 4096, .out = 1024 }, .{ .in = 4096, .out = 12288 }, .{ .in = 12288, .out = 4096 },
    };
    if (m < 256 or m % 32 != 0) return error.InvalidArgument;
    const strength: f32 = 0.8;
    const nref: usize = 256; // host-reference rows
    const tol_i8a: f64 = 3e-2;
    const tol_bf16: f64 = 5e-3;
    const rounds: usize = 30;
    const calls: usize = 20;

    var prng = std.Random.DefaultPrng.init(0x10cab);
    const rnd = prng.random();
    var failed = false;

    const timer = try be.ctx.timerCreate();
    defer be.ctx.timerDestroy(timer);

    try stdout.print("m {d}, ranks {any}, strength {d}, base int8 convrot; delta rel is against the host f32 sidecar on the first {d} rows\n", .{ m, ranks, strength, nref });

    for (shapes, 0..) |sh, si| {
        var sa = std.heap.ArenaAllocator.init(page);
        defer sa.deinit();
        const a = sa.allocator();
        const randBf16 = struct {
            fn f(al: std.mem.Allocator, r: std.Random, n: usize, sd: f32) ![]u8 {
                const b = try al.alignedAlloc(u8, .of(u32), n * 2);
                for (0..n) |i| std.mem.writeInt(u16, b[i * 2 ..][0..2], f32ToBf16(r.floatNorm(f32) * sd), .little);
                return b;
            }
        }.f;

        // An int8 convrot checkpoint's linear: a gaussian weight rotated and
        // quantized per row, which is what `quantizeA` does.
        const w_bf = try randBf16(a, rnd, sh.out * sh.in, 0.02);
        const wq = try lora.quantizeA(a, Weight.init(w_bf, .bf16, sh.out, sh.in), true);
        var w = wq.weight(sh.in, true);
        w.tag = "bench.weight";
        const lins = [_]Weight{w};

        // One stack of two files per rank; the second file is only read by the
        // two-file check. `t` is the file's own factor, padded as `Sidecar.load` pads.
        const PerRank = struct { t: lora.Target, t2: lora.Target, stack: lora.Stack };
        var per: [max_ranks]PerRank = undefined;
        for (ranks, 0..) |rank, ri| {
            const pr = &per[ri];
            var ts: [2]lora.Target = undefined;
            for (&ts, [_]f32{ 1.0, 0.5 }, [_][]const u8{ "bench", "bench2" }) |*tt, sc, tag| {
                const fac = try a.alloc(lora.Factor, 1);
                fac[0] = try lora.padRank(a, .{
                    .a = Weight.init(try randBf16(a, rnd, rank * sh.in, 0.02), .bf16, rank, sh.in),
                    .b = Weight.init(try randBf16(a, rnd, sh.out * rank, 0.02), .bf16, sh.out, rank),
                    .out_off = 0,
                });
                tt.* = .{ .factors = fac, .in_dim = sh.in, .out_dim = sh.out, .scale = sc, .tag = tag };
            }
            pr.t = ts[0];
            pr.t2 = ts[1];
            pr.stack = .{};
            for ([_]f32{ strength, 0.6 }) |s| try pr.stack.files.append(a, .{
                .side = .{ .arena = std.heap.ArenaAllocator.init(a), .index = .empty, .unclaimed = 0 },
                .strength = s,
                .path = "synthetic",
            });
        }
        // Pointers into `per` only once it has stopped moving.
        for (per[0..ranks.len]) |*pr| {
            var hit_list: std.ArrayListUnmanaged(lora.Stack.Hit) = .empty;
            try hit_list.append(a, .{ .target = &pr.t, .file = 0 });
            try hit_list.append(a, .{ .target = &pr.t2, .file = 1 });
            try pr.stack.index.put(a, "bench.weight", hit_list);
            const hits = &pr.stack.index.getPtr("bench.weight").?.items;
            // The production attach, not a copy of it.
            try lin_cuda.quantizeLora(a, &pr.stack, &lins, .auto, .w8a8);
            if (hits.*[0].a_i8 == null) {
                try stdout.print("FAIL: quantizeLora gave the int8 route no int8 A\n", .{});
                std.process.exit(1);
            }
            hits.len = 1; // one file unless a check asks for two
        }

        var plan = try lin_cuda.plan(&lins, .{ .blockq = .auto }, "lora-cuda-bench");
        try lin_cuda.presize(be, plan, &lins);
        for (per[0..ranks.len]) |*pr| try lin_cuda.presizeLora(be, &pr.stack, &lins, m);

        const mpad = std.mem.alignForward(usize, m, 128);
        var xd = try be.tensorCreate(mpad * sh.in * 4);
        defer be.tensorDestroy(&xd);
        var yd = try be.tensorCreate(mpad * sh.out * 4);
        defer be.tensorDestroy(&yd);
        const x = try a.alloc(f32, mpad * sh.in);
        const plane = mpad * sh.out;
        const p_buf = try page.alloc(f32, plane);
        defer page.free(p_buf);
        const base_rows = try a.alloc(f32, nref * sh.out);
        const want = try a.alloc(f32, nref * sh.out);
        const base_ref = try a.alloc(f32, nref * sh.out);

        const Run = struct {
            fn go(b: *cuda.Backend, pl: *lin_cuda.Plan, st: ?*const lora.Stack, i8a: bool, xb: cuda.backend.DeviceBuffer, yb: cuda.backend.DeviceBuffer, rows: usize, ws: []const Weight) !void {
                lin_cuda.lora_i8_a = i8a;
                pl.lora = st;
                const g = try lin_cuda.prep(b, pl.*, xb, rows, ws[0].cols, ws);
                try lin_cuda.gemm(b, pl.*, g, yb, ws[0], null);
            }
            fn once(b: *cuda.Backend, pl: *lin_cuda.Plan, st: ?*const lora.Stack, i8a: bool, xb: cuda.backend.DeviceBuffer, yb: cuda.backend.DeviceBuffer, rows: usize, ws: []const Weight, dst: []f32) !void {
                try b.beginBatch();
                errdefer if (b.batching()) b.abortBatch();
                try go(b, pl, st, i8a, xb, yb, rows, ws);
                try b.endBatch();
                try b.tensorDownload(yb, std.mem.sliceAsBytes(dst));
            }
        };
        const relOf = struct {
            fn f(ref: []const f32, got: []const f32, base: ?[]const f32) f64 {
                var l2r: f64 = 0;
                var l2e: f64 = 0;
                for (ref, got, 0..) |e, g, i| {
                    const d = if (base) |bb| g - bb[i] else g;
                    l2r += @as(f64, e) * e;
                    l2e += @as(f64, e - d) * (e - d);
                }
                return @sqrt(l2e / l2r);
            }
        }.f;
        const stackOf = struct {
            fn f(p: []PerRank, rs: []const usize, arm: Arm) ?*const lora.Stack {
                if (!arm.lora) return null;
                for (rs, 0..) |r, i| if (r == arm.rank) return &p[i].stack;
                unreachable;
            }
        }.f;

        try stdout.print("\n-- shape in {d} -> out {d} --\n", .{ sh.in, sh.out });

        // Numerics: gaussian and heavy-tailed activations, all m rows through the
        // device, the first `nref` against the host.
        for ([_][]const u8{ "gaussian", "heavy-tailed" }) |dist| {
            @memset(x, 0);
            for (x[0 .. m * sh.in]) |*v| {
                v.* = rnd.floatNorm(f32);
                if (dist[0] == 'h' and rnd.float(f32) < 0.005) v.* *= 30;
            }
            try be.tensorUpload(xd, std.mem.sliceAsBytes(x));
            const xr = x[0 .. nref * sh.in];
            try TensorPencil.ops.matmul.matmul(io, a, base_ref, xr, nref, w, null);

            try Run.once(be, &plan, null, false, xd, yd, m, &lins, p_buf);
            @memcpy(base_rows, p_buf[0 .. nref * sh.out]);
            const base_rel = relOf(base_ref, base_rows, null);
            try stdout.print("  {s:<12} base W8A8 rel {e:.3}\n", .{ dist, base_rel });

            for (per[0..ranks.len], ranks) |*pr, rank| {
                @memset(want, 0);
                try pr.t.applyHost(io, a, want, sh.out, xr, nref, strength);
                for ([_]bool{ false, true }) |i8a| {
                    try Run.once(be, &plan, &pr.stack, i8a, xd, yd, m, &lins, p_buf);
                    const tol_d = if (i8a) tol_i8a else tol_bf16;
                    const rel = relOf(want, p_buf[0 .. nref * sh.out], base_rows);
                    const ok = rel < tol_d;
                    if (!ok) failed = true;
                    try stdout.print("  {s:<12} rank {d:>3} {s:<7} delta rel {e:.3}  (limit {e:.1}){s}\n", .{ dist, rank, if (i8a) "int8 A" else "bf16 A", rel, tol_d, if (ok) "" else "  FAIL" });
                }
            }
        }

        for (per[0..ranks.len], ranks) |*pr, rank| {
            const hits = &pr.stack.index.getPtr("bench.weight").?.items;
            @memset(want, 0);
            try pr.t.applyHost(io, a, want, sh.out, x[0 .. nref * sh.in], nref, strength);

            // The check has teeth: an A quantized unrotated but labelled rotated pairs
            // the wrong basis with the prep, and must fail the delta limit.
            {
                const good = hits.*[0].a_i8.?;
                const bad = try lora.quantizeA(a, pr.t.factors[0].a, false);
                var lie = bad.weight(sh.in, false);
                lie.convrot = @intCast(TensorPencil.ops.convrot.group_size);
                hits.*[0].a_i8 = lie;
                try Run.once(be, &plan, &pr.stack, true, xd, yd, m, &lins, p_buf);
                hits.*[0].a_i8 = good;
                be.evictWeightRange(std.mem.sliceAsBytes(bad.q));
                be.evictWeightRange(std.mem.sliceAsBytes(bad.scale));
                const rel = relOf(want, p_buf[0 .. nref * sh.out], base_rows);
                const bites = rel >= tol_i8a;
                if (!bites) failed = true;
                try stdout.print("  rank {d:>3} teeth: wrong-basis int8 A gives delta rel {e:.3}, {s}\n", .{ rank, rel, if (bites) "the check fails it" else "FAIL: the check would pass it" });
            }

            if (si == 0) {
                // A stack of two files, each hit with its own int8 A.
                hits.len = 2;
                try pr.t2.applyHost(io, a, want, sh.out, x[0 .. nref * sh.in], nref, 0.6);
                for ([_]bool{ false, true }) |i8a| {
                    try Run.once(be, &plan, &pr.stack, i8a, xd, yd, m, &lins, p_buf);
                    const rel = relOf(want, p_buf[0 .. nref * sh.out], base_rows);
                    const ok = rel < (if (i8a) tol_i8a else tol_bf16);
                    if (!ok) failed = true;
                    try stdout.print("  rank {d:>3} 2 files: {s:<7} delta rel {e:.3}{s}\n", .{ rank, if (i8a) "int8 A" else "bf16 A", rel, if (ok) "" else "  FAIL" });
                }
                hits.len = 1;

                // A row count off the tile: the int8 A GEMM stores every padded row, and
                // the pad rows of `y` must still come back zero.
                const m_odd: usize = m - 96;
                for ([_]bool{ false, true }) |i8a| {
                    try Run.once(be, &plan, &pr.stack, i8a, xd, yd, m_odd, &lins, p_buf);
                    var pad_nonzero: usize = 0;
                    for (p_buf[m_odd * sh.out ..]) |v| pad_nonzero += @intFromBool(v != 0);
                    if (pad_nonzero != 0) failed = true;
                    try stdout.print("  rank {d:>3} m {d}: {s:<7} {d} nonzero pad values{s}\n", .{ rank, m_odd, if (i8a) "int8 A" else "bf16 A", pad_nonzero, if (pad_nonzero == 0) "" else "  FAIL" });
                }
            }
        }

        // Timing. Warm every arm, then each round runs every arm once, in an order
        // rotated by one per round.
        for (0..2) |_| for (arms) |arm| try Run.once(be, &plan, stackOf(&per, ranks, arm), arm.i8a, xd, yd, m, &lins, p_buf[0..1]);
        var ms: [arm_buf.len][rounds]f32 = undefined;
        for (0..rounds) |r| {
            for (0..arms.len) |j| {
                const i = (j + r) % arms.len;
                const arm = arms[i];
                try be.beginBatch();
                lin_cuda.lora_i8_a = arm.i8a;
                plan.lora = stackOf(&per, ranks, arm);
                const g = try lin_cuda.prep(be, plan, xd, m, sh.in, &lins);
                try be.ctx.timerBegin(timer);
                for (0..calls) |_| try lin_cuda.gemm(be, plan, g, yd, w, null);
                const t_ms = try be.ctx.timerEndMs(timer);
                try be.endBatch();
                ms[i][r] = t_ms / @as(f32, @floatFromInt(calls));
            }
        }
        var med: [arm_buf.len]f32 = undefined;
        var mins: [arm_buf.len]f32 = undefined;
        for (ms[0..arms.len], 0..) |*row, i| {
            std.mem.sort(f32, row, {}, std.sort.asc(f32));
            med[i] = row[rounds / 2];
            mins[i] = row[0];
        }
        try stdout.print("  {s:<10} {s:>4} {s:>9} {s:>9} {s:>10} {s:>9}\n", .{ "arm", "rank", "median ms", "min ms", "sidecar ms", "overhead" });
        for (arms, 0..) |arm, i| {
            const side = med[i] - med[0];
            try stdout.print("  {s:<10} {d:>4} {d:>9.3} {d:>9.3} {d:>10.3} {d:>8.1}%\n", .{ arm.name, arm.rank, med[i], mins[i], side, 100 * side / med[0] });
        }
        be.evictWeights();
    }
    try stdout.print("\ntiming: {d} rounds; each round runs every arm once, starting one arm later than the round before; a sample is {d} back-to-back gemm calls between two stream events, after one untimed prep of the activation; median and min over rounds. Base GEMM excludes its prep.\n", .{ rounds, calls });
    try stdout.flush();
    if (failed) {
        try stdout.print("FAIL\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }
    try stdout.print("OK\n", .{});
}

/// `lora-sidecar-bench <ckpt> <lora> [--size N] [--rounds R] [--backend B]
/// [--text-encoder P] [--vae P]`: one live session, every variant on it, the dial and
/// the sidecar's diagnostics flipped between forwards. Times `Denoiser.predict`
/// interleaved, compares the velocities, then renders one image per rendering variant
/// to scratch_out/. CUDA's variants add the bf16 A against the int8 one and two skipped
/// A GEMMs; Vulkan has one sidecar.
fn loraSidecarBench(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8, lora_path: []const u8, size: usize, rounds: usize, backend: TensorPencil.pipeline.Backend, te: []const u8, vae: []const u8) !void {
    const pipeline = TensorPencil.pipeline;
    const lin_cuda = TensorPencil.models.lin_cuda;
    const lora_cuda = TensorPencil.models.lora_cuda;
    const specs = [_]pipeline.LoraSpec{.{ .path = lora_path, .strength = 1.0 }};
    var opts: pipeline.Options = .{
        .prompt = "a red fox curled up asleep in fresh snow, soft morning light, detailed fur",
        .dit_path = ckpt,
        .width = size,
        .height = size,
        .steps = 8,
        .seed = 42,
        .backend = backend,
        .preview = false,
        .loras = &specs,
    };
    if (te.len != 0) opts.text_encoder_path = te;
    if (vae.len != 0) opts.vae_path = vae;
    try stdout.print("== lora-sidecar-bench ==\nckpt: {s}\nlora: {s}\n{d}x{d}, {d} steps, {t}\n", .{ ckpt, lora_path, size, size, opts.steps, backend });
    var sess = pipeline.Session.init(io, arena, opts, stdout) catch |err| {
        try stdout.print("cannot load: {t}\n", .{err});
        return;
    };
    defer sess.deinit();
    defer lin_cuda.lora_i8_a = true;
    defer lora_cuda.bench_skip_a = .none;
    defer TensorPencil.gpu.cuda.Backend.bench_acc_overwrite = false;

    // `no_lora` runs a denoiser built before the stack was attached, which every
    // device arm reads at that point: the true base, where `lora_off` is the
    // stack at strength 0 and keeps whatever the attach changed (SD's f32 chain).
    // `old` and `new` are the two sidecars compared; CUDA's skip_a rows time the A
    // GEMMs (`lora_cuda.bench_skip_a`) and render nothing.
    const SkipA = @TypeOf(lora_cuda.bench_skip_a);
    const Variant = struct { name: []const u8, strength: f32, i8a: bool = true, plain: bool = false, skip_a: SkipA = .none, overwrite: bool = false };
    const cuda_variants = [_]Variant{
        .{ .name = "no_lora", .strength = 0, .plain = true },
        .{ .name = "lora_off", .strength = 0 },
        .{ .name = "old_sidecar", .strength = 1, .i8a = false },
        .{ .name = "new_sidecar", .strength = 1 },
        .{ .name = "skip_a_all", .strength = 1, .skip_a = .all },
        .{ .name = "skip_a_shared", .strength = 1, .skip_a = .shared },
        // The B GEMM's read of the output plane removed, its result wrong: what
        // running the sidecar before the base GEMM could save at most.
        .{ .name = "b_overwrite", .strength = 1, .overwrite = true },
    };
    const vk_variants = [_]Variant{
        .{ .name = "no_lora", .strength = 0, .plain = true },
        .{ .name = "lora_off", .strength = 0 },
        .{ .name = "sidecar", .strength = 1 },
    };
    const variants: []const Variant = if (backend == .vulkan) &vk_variants else &cuda_variants;
    const set = struct {
        fn f(se: *pipeline.Session, v: Variant) void {
            se.setLoraStrength(0, v.strength);
            lin_cuda.lora_i8_a = v.i8a;
            lora_cuda.bench_skip_a = v.skip_a;
            TensorPencil.gpu.cuda.Backend.bench_acc_overwrite = v.overwrite;
        }
    }.f;

    var cond = try sess.encode(arena, opts.prompt, .{});
    defer cond.deinit(arena);
    sess.armDenoiserResidency();
    const sigmas = try sess.schedule(arena, opts.steps, opts.shift);
    const ds = sess.spatialDownscale();
    const lat_h = size / ds;
    const lat_w = size / ds;
    const n = sess.latentChannels() * lat_h * lat_w;
    try sess.setLoras(io, &.{});
    var den_plain = try sess.denoiser(arena, cond, null, 1.0, lat_h, lat_w, sigmas);
    defer den_plain.deinit(arena);
    try sess.setLoras(io, &specs);
    var den = try sess.denoiser(arena, cond, null, 1.0, lat_h, lat_w, sigmas);
    defer den.deinit(arena);

    var prng = std.Random.DefaultPrng.init(42);
    const x = try arena.alloc(f32, n);
    for (x) |*v| v.* = prng.random().floatNorm(f32);
    sess.scaleInitialNoise(x, sigmas[0], .{ .channels = sess.latentChannels(), .h = lat_h, .w = lat_w });

    const vel = try arena.alloc([]f32, variants.len);
    for (vel, variants) |*v, vr| {
        v.* = try arena.alloc(f32, n);
        set(sess, vr);
        try (if (vr.plain) &den_plain else &den).predict(arena, v.*, x, sigmas[0], null);
    }
    const relOf = struct {
        fn f(ref: []const f32, got: []const f32) f64 {
            var a: f64 = 0;
            var e: f64 = 0;
            for (ref, got) |r, g| {
                a += @as(f64, r) * r;
                e += @as(f64, r - g) * (r - g);
            }
            return @sqrt(e / a);
        }
    }.f;
    // `lora off vs no lora` includes anything the denoiser computed once at build time
    // with the stack at full strength, krea2's text fusion among it.
    try stdout.print("velocity at sigma {d:.3}: sidecar vs lora off rel {e:.3} (what the LoRA moves); lora off vs no lora rel {e:.3}\n", .{
        sigmas[0], relOf(vel[2], vel[1]), relOf(vel[0], vel[1]),
    });
    if (variants.len > 3) try stdout.print("new vs old sidecar rel {e:.3}\n", .{relOf(vel[2], vel[3])});
    if (variants.len > 4)
        try stdout.print("skipped A changes the velocity: all rel {e:.3}, shared rel {e:.3}\n", .{ relOf(vel[3], vel[4]), relOf(vel[3], vel[5]) });

    // Interleaved: each round runs every variant once, starting one later per round.
    const per: usize = 3;
    const t = try arena.alloc(f64, rounds * variants.len);
    for (variants) |vr| {
        set(sess, vr);
        try (if (vr.plain) &den_plain else &den).predict(arena, vel[0], x, sigmas[1], null);
    }
    for (0..rounds) |r| {
        for (0..variants.len) |j| {
            const i = (j + r) % variants.len;
            set(sess, variants[i]);
            const t0 = std.Io.Clock.awake.now(io).nanoseconds;
            const d = if (variants[i].plain) &den_plain else &den;
            for (0..per) |_| try d.predict(arena, vel[i], x, sigmas[1], null);
            const t1 = std.Io.Clock.awake.now(io).nanoseconds;
            t[i * rounds + r] = @as(f64, @floatFromInt(t1 - t0)) / 1e9 / @as(f64, @floatFromInt(per));
        }
    }
    const med = try arena.alloc(f64, variants.len);
    for (variants, 0..) |vr, i| {
        const col = t[i * rounds ..][0..rounds];
        std.mem.sort(f64, col, {}, std.sort.asc(f64));
        med[i] = col[rounds / 2];
        try stdout.print("  {s:<12} {d:.4} s/step median, {d:.4} min, {d:.4} max\n", .{ vr.name, med[i], col[0], col[rounds - 1] });
    }
    try stdout.print("  over no LoRA: attached at 0 +{d:.1}%, {s} +{d:.1}%\n", .{ 100 * (med[1] - med[0]) / med[0], variants[2].name, 100 * (med[2] - med[0]) / med[0] });
    if (variants.len > 3) try stdout.print("  over no LoRA: {s} +{d:.1}%\n", .{ variants[3].name, 100 * (med[3] - med[0]) / med[0] });
    if (variants.len > 4) try stdout.print("  A GEMMs: all {d:.1} ms/step, the ones a shared A would absorb {d:.1} ms/step\n", .{
        1e3 * (med[3] - med[4]), 1e3 * (med[3] - med[5]),
    });
    if (variants.len > 6) try stdout.print("  B's read of the output plane: {d:.1} ms/step ({d:.1}% of new_sidecar's step), the most a sidecar-first order saves\n", .{
        1e3 * (med[3] - med[6]), 100 * (med[3] - med[6]) / med[3],
    });
    try stdout.print("  timing: {d} rounds, each round every variant once in an order rotated by one per round, {d} predict calls per sample, same denoiser and latent\n", .{ rounds, per });

    // One render per variant that renders, the whole 8-step trajectory.
    try std.Io.Dir.cwd().createDirPath(io, "scratch_out");
    const renders = @min(variants.len, 4);
    var rgb: [4][]u8 = undefined;
    for (variants[0..renders], 0..) |vr, i| {
        set(sess, vr);
        const img = try sess.generate(opts, null);
        rgb[i] = img.rgb;
        var png: std.ArrayList(u8) = .empty;
        try TensorPencil.image.encodePngRgb(arena, &png, img.rgb, img.width, img.height);
        var name_buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "scratch_out/lora_e2e_{d}_{s}.png", .{ size, vr.name });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = png.items });
        try stdout.print("  wrote {s}\n", .{name});
    }
    const psnr = struct {
        fn f(p: []const u8, q: []const u8) f64 {
            var se: f64 = 0;
            for (p, q) |u, v| {
                const d = @as(f64, @floatFromInt(u)) - @as(f64, @floatFromInt(v));
                se += d * d;
            }
            const mse = se / @as(f64, @floatFromInt(p.len));
            return if (mse == 0) std.math.inf(f64) else 10 * std.math.log10(255.0 * 255.0 / mse);
        }
    }.f;
    try stdout.print("image PSNR: sidecar vs lora off {d:.2} dB (what the LoRA moves)\n", .{psnr(rgb[2], rgb[1])});
    if (renders > 3) try stdout.print("image PSNR: new vs old sidecar {d:.2} dB\n", .{psnr(rgb[2], rgb[3])});
    try stdout.flush();
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
/// CUDA context. It runs at the DEVELOPMENT shape: 147 packed
/// rows, which exercises every code path the full render does and costs seconds
/// rather than the ~20 minutes a full-resolution CPU reference would.
/// `TP_H3_LORA=<file>`: attach that LoRA to an H3 device test's DiT, so the device
/// sidecar is held against the CPU forward's host apply of the same stack.
fn h3TestLora(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, dit: *TensorPencil.models.minimax_h3.DiT) !void {
    const path = std.mem.span(std.c.getenv("TP_H3_LORA") orelse return);
    const lora = TensorPencil.models.lora;
    const ct = try arena.create(TensorPencil.pipeline.Container);
    ct.* = try TensorPencil.pipeline.Container.openNamed(arena, io, path);
    const stack = try arena.create(lora.Stack);
    stack.* = .{};
    _ = try stack.add(arena, ct.store(), 1.0, path, .{});
    const n = try dit.attachLora(arena, stack);
    try stdout.print("lora: {s}, {d} targets attached\n", .{ path, n });
}

/// The residual's storage in an H3 device test, from `--act`. `TP_H3_WEIGHT_ONLY`:
/// int8 weights weight-only, so dtypes compare without W8A8 noise.
fn h3TestAct() TensorPencil.pipeline.Act {
    if (std.c.getenv("TP_H3_WEIGHT_ONLY") != null) {
        TensorPencil.models.minimax_h3_cuda.force_weight_only = true;
        TensorPencil.models.minimax_h3_gpu.force_weight_only = true;
    }
    return TensorPencil.models.minimax_h3_dev.streamDtFor(testAct());
}

/// `minimax-h3-vk-test`: the H3 trunk on Vulkan against its CPU reference, on
/// real weights. The twin of `minimax-h3-cuda-test` and deliberately the same
/// shapes, seeds and tolerance, so the two arms' figures sit side by side.
///
/// Takes the same `TP_H3_*` knobs: `BLOCKS`/`BLOCK_OFF` to narrow, `MASK` for the
/// per-row modulation index, `NAIVE` for the reference attention, `LH`/`LW`/`TEXT`
/// for the shape. Exits non-zero on a mismatch so it works as a gate.
fn minimaxH3VkTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, ckpt: []const u8) !void {
    const h3 = TensorPencil.models.minimax_h3;
    const h3_gpu = TensorPencil.models.minimax_h3_gpu;
    const safetensors = TensorPencil.SafeTensors;

    var ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== minimax-h3-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});

    std.Io.Dir.cwd().access(io, ckpt, .{}) catch {
        try stdout.print("checkpoint not found: {s}\n", .{ckpt});
        return;
    };
    var st = try safetensors.open(arena, io, ckpt);
    defer st.deinit();
    var dit = try h3.DiT.load(arena, .{ .safetensors = &st });
    defer dit.deinit();
    try h3TestLora(arena, io, stdout, &dit);
    if (std.c.getenv("TP_H3_NAIVE") != null) {
        h3_gpu.force_naive_attn = true;
        try stdout.print("(naive attention)\n", .{});
    }
    {
        const off = if (std.c.getenv("TP_H3_BLOCK_OFF")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0) else 0;
        const n = if (std.c.getenv("TP_H3_BLOCKS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch dit.blocks.len) else dit.blocks.len;
        if (off != 0 or n != dit.blocks.len) {
            const lo = @min(off, dit.blocks.len);
            dit.sliceBlocks(lo, @min(lo + n, dit.blocks.len));
            try stdout.print("(blocks [{d},{d}))\n", .{ lo, lo + dit.blocks.len });
        }
    }
    try stdout.print("dit: {d} blocks, hidden {d}, {d}x{d} heads\n", .{
        dit.cfg.n_layers, dit.cfg.hidden, dit.cfg.n_heads, dit.cfg.head_dim,
    });
    if (!h3_gpu.supported(ctx, &dit)) {
        try stdout.print("FAIL: this checkpoint's weights have no Vulkan path here\n", .{});
        return error.UnsupportedCheckpoint;
    }

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

    // The same seed the CUDA test uses, so the two arms' figures compare directly.
    var prng = std.Random.DefaultPrng.init(0x431a);
    const rnd = prng.random();
    const video = try arena.alloc(f32, h3.latent_channels * lat_t * lat_h * lat_w);
    for (video) |*v| v.* = rnd.floatNorm(f32);
    const audio = try arena.alloc(f32, h3.audio_latent_channels * h3.audio_channels * audio_t);
    for (audio) |*v| v.* = rnd.floatNorm(f32);
    const text = try arena.alloc(f32, text_len * dit.cfg.hidden);
    for (text) |*v| v.* = rnd.floatNorm(f32) * 0.5;

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
        for (audio_mask, 0..) |*m, i| m.* = if (i < audio_t) 0.0 else 1.0;
        try stdout.print("denoise mask: {s} ({d} video rows, {d} audio rows)\n", .{
            mode, video_mask.len, audio_mask.len,
        });
    }

    const in: h3.Inputs = .{
        .video = video,
        .audio = audio,
        .text = text,
        .sigma = sigma,
        .video_mask = video_mask,
        .audio_mask = audio_mask,
    };

    var ws = try h3.Workspace.init(arena, dit.cfg, &layout);
    defer ws.deinit(arena);
    const cpu_v = try arena.alloc(f32, video.len);
    const cpu_a = try arena.alloc(f32, audio.len);
    var t0 = std.Io.Clock.real.now(io).nanoseconds;
    try h3.forward(&dit, io, arena, &ws, &layout, cpu_v, cpu_a, in);
    const cpu_ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;

    var sess = try h3_gpu.Session.init(ctx, arena, &dit, &layout);
    defer sess.deinit(ctx);
    var dws = try h3_gpu.Workspace.init(ctx, &dit, layout.seq_len, h3TestAct());
    defer dws.deinit(ctx);
    const gpu_v = try arena.alloc(f32, video.len);
    const gpu_a = try arena.alloc(f32, audio.len);
    t0 = std.Io.Clock.real.now(io).nanoseconds;
    try h3_gpu.forward(&dit, ctx, &sess, &dws, io, arena, &layout, gpu_v, gpu_a, in, null);
    const gpu_ms = @as(f64, @floatFromInt(std.Io.Clock.real.now(io).nanoseconds - t0)) / 1e6;

    // Per segment, because one output figure cannot say which rows diverged.
    {
        const dev_trunk = try arena.alloc(f32, layout.seq_len * dit.cfg.hidden);
        try ctx.downloadF32At(dws.x_d, 0, dev_trunk);
        for (dev_trunk) |*v| v.* *= TensorPencil.models.minimax_h3_dev.streamDiv(dws.x_d.dt);
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
        // Same FORM as the CUDA arm's and for the same reason (W8A8 against a
        // W8A32 host reference diverges with depth by design), but a 2e-2 base
        // rather than 1.5e-2. That is Vulkan's own int8 rounding in the last
        // block, and it is measured, not a fudge: the two arms agree to 3% of
        // each other through 25 blocks (audio 0.00206 here against 0.00213 on
        // CUDA) and both then jump at 50, where this DiT's final `fc2` scales
        // average 3x smaller than every other block's, so a fixed absolute
        // quantization error reads as a large RELATIVE one. At that depth the
        // trunk agrees between the arms (video 0.0616 against 0.0606, text
        // 0.431 against 0.448) while only the amplified output differs, and
        // swapping the tensor-core attention for `attn_full` moves the figure
        // by 1% -- so the divergence is the quantization, not this port.
        //
        // Like the CUDA bound this says "no defect", not "matches the reference
        // render". The acceptance test is the two device arms compared on a
        // whole render, which is owed.
        const tol = 2.0e-2 * (1.0 + @as(f64, @floatFromInt(dit.blocks.len)) / 8.0);
        const ok = rel < tol and std.math.isFinite(rel);
        if (!ok) failures += 1;
        try stdout.print("{s:<8} rel L2 {d:.5}  max |dv| {d:.5}  (tol {d:.4})  {s}\n", .{
            c.name, rel, max_abs, tol, if (ok) "ok" else "FAIL",
        });
    }
    try stdout.print("\ncpu {d:.0} ms, device {d:.0} ms ({d:.1}x)\n", .{ cpu_ms, gpu_ms, cpu_ms / gpu_ms });
    if (failures != 0) return error.GpuMismatch;
}

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
    try h3TestLora(arena, io, stdout, &dit);
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
            dit.sliceBlocks(lo, @min(lo + n, dit.blocks.len));
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
        var dws_u = try h3_cuda.Workspace.init(be, &dit, layout.seq_len, .f32);
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
        var dws_p = try h3_cuda.Workspace.init(be, &dit, layout.seq_len, .f32);
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
        try be.opI8Prep(xd, m, h);
        try be.opI8Gemm(yd, w.bytes, w.row_scale.?, w.rows, null);
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
    var dws = try h3_cuda.Workspace.init(be, &dit, layout.seq_len, h3TestAct());
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
        try be.downloadF32(dws.x_d, dev_trunk);
        for (dev_trunk) |*v| v.* *= TensorPencil.models.minimax_h3_dev.streamDiv(dws.x_d.dt);
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
/// `minimax-h3-vae-vk-test`: H3's video VAE decode on Vulkan against its CPU
/// decoder, on real weights. The twin of `minimax-h3-vae-cuda-test`, same fixture
/// shape, same tolerance. `TP_VAE_T`/`_H`/`_W` override the grid.
fn minimaxH3VaeVkTest(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer) !void {
    const h3vae = TensorPencil.models.minimax_h3_vae;
    const h3vae_gpu = TensorPencil.models.minimax_h3_vae_gpu;
    const safetensors = TensorPencil.SafeTensors;

    var ctx = TensorPencil.gpu.Context.init(arena, io) catch |err| {
        try stdout.print("vulkan unavailable: {t}\n", .{err});
        return;
    };
    defer ctx.deinit();
    try stdout.print("== minimax-h3-vae-vk-test ==\nvulkan device: {s}\n", .{ctx.deviceName()});

    // The same embedded fixture the CUDA arm checks against, weights and `in.z`
    // in one file, so neither arm needs a checkpoint on disk.
    const fixture = @embedFile("models/assets/minimax_h3_vae.safetensors");
    var st = try safetensors.initFromSlice(arena, fixture);
    var dec = try h3vae.VideoDecoder.load(arena, .{ .safetensors = &st });
    defer dec.deinit();
    try stdout.print("decoder: {d} blocks, dim {d}, {d}x{d} heads, patch {d}x{d}\n", .{
        dec.cfg.n_layers, dec.cfg.dim, dec.cfg.heads, dec.cfg.head_dim, dec.cfg.patch_t, dec.cfg.patch,
    });
    if (!h3vae_gpu.supported(&dec)) {
        try stdout.print("FAIL: this decoder's weights have no Vulkan path here\n", .{});
        return error.UnsupportedCheckpoint;
    }
    if (std.c.getenv("TP_VAE_NAIVE") != null) {
        h3vae_gpu.force_naive_attn = true;
        try stdout.print("(naive attention)\n", .{});
    }

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

    const sh = h3vae.outputShape(dec.cfg, t, h, w);
    const n = dec.cfg.out_channels * sh.frames * sh.height * sh.width;
    const want = try arena.alloc(f32, n);
    const got = try arena.alloc(f32, n);

    try h3vae.decodeVolume(&dec, io, arena, want, z, t, h, w);

    var sess = try h3vae_gpu.Session.init(ctx, arena, &dec, t, h, w);
    defer sess.deinit(ctx);
    var all_ok = true;
    for ([_]TensorPencil.gpu.context.Dt{ .f32, .f16, .bf16 }) |act| {
        var ws = try h3vae_gpu.Workspace.init(ctx, arena, &dec, sess.seq, sess.grid, act);
        defer ws.deinit(ctx);
        try h3vae_gpu.decodeVolume(&dec, ctx, &sess, &ws, io, arena, got, z, t, h, w);
        if (!try h3VaeReport(stdout, act, want, got)) all_ok = false;
    }
    try stdout.flush();
    if (!all_ok) return error.DeviceMismatch;
}

fn h3VaeReport(stdout: *Io.Writer, act: TensorPencil.gpu.context.Dt, want: []const f32, got: []const f32) !bool {
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
    const bound: f64 = if (act == .f32) 2e-3 else 2e-2;
    const ok = nonfinite == 0 and rel < bound;
    try stdout.print("volume decode {t:<5} rel L2 {d:.6}  max |dv| {d:.6}  {s}\n", .{ act, rel, max_abs, if (ok) "ok" else "FAIL" });
    return ok;
}

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
    var all_ok = true;
    for ([_]TensorPencil.gpu.context.Dt{ .f32, .f16, .bf16 }) |act| {
        var ws = try h3vae_cuda.Workspace.init(be, &dec, sess.seq, sess.grid, act);
        defer ws.deinit(be);
        try h3vae_cuda.decodeVolume(&dec, be, &sess, &ws, io, arena, got, z, t, h, w);
        if (!try h3VaeReport(stdout, act, want, got)) all_ok = false;
    }
    try stdout.flush();
    if (!all_ok) return error.DeviceMismatch;
}
