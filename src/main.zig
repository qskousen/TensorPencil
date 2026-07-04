//! TensorPencil CLI — thin driver over the TensorPencil library module.

const std = @import("std");
const Io = std.Io;

const TensorPencil = @import("TensorPencil");

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
            \\      --width 1024       image width  (multiple of 16)
            \\      --height 1024      image height (multiple of 16)
            \\      --steps 8          sampling steps
            \\      --cfg 1.0          guidance scale (1.0 = no negative pass)
            \\      --seed 0           noise seed
            \\      --shift 1.15       flow-matching sigma shift
            \\      --gpu off          offload large GEMMs to Vulkan (on/off)
            \\      --vram-budget 0    GiB of device memory to use (0 = ask the
            \\                         driver); weights past it stream per step
            \\      --encoder-f16 off  run the text encoder GEMMs on tensor
            \\                         cores (f16): ~0.4s faster, slightly less
            \\                         exact conditioning (on/off)
            \\      --dit-f32 off      run the diffusion model in full f32
            \\                         instead of the f16 tensor-core path
            \\                         (slower, more exact) (on/off)
            \\      --dit <path>       diffusion checkpoint (fp8 or int8 convrot;
            \\                         auto-detected). Default: krea2 ...Fp8
            \\      --out out.png      output file
            \\  TensorPencil inspect <file.safetensors>   list tensors in a checkpoint
            \\  TensorPencil bench-matmul                 time a DiT-sized fp8 GEMM
            \\  TensorPencil decode-latent <z.bin> <zh> <zw> <out.png>
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

fn generate(arena: std.mem.Allocator, io: Io, stdout: *Io.Writer, args: []const []const u8) !void {
    var opts: TensorPencil.pipeline.Options = .{ .prompt = "" };
    var out_path: []const u8 = "out.png";
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
        } else if (std.mem.eql(u8, flag, "--profile")) {
            TensorPencil.models.dit_gpu.profile = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1");
        } else if (std.mem.eql(u8, flag, "--gpu")) {
            opts.use_gpu = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--vram-budget")) {
            const gib = try std.fmt.parseFloat(f64, val);
            opts.vram_budget = @intFromFloat(gib * (1 << 30));
        } else if (std.mem.eql(u8, flag, "--encoder-f16")) {
            opts.encoder_f16 = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--dit-f32")) {
            TensorPencil.models.dit_gpu.force_f32 = std.mem.eql(u8, val, "on") or std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else if (std.mem.eql(u8, flag, "--dit")) {
            opts.dit_path = val;
        } else if (std.mem.eql(u8, flag, "--out")) {
            out_path = val;
        } else {
            try stdout.print("unknown flag {s}\n", .{flag});
            return error.InvalidArgs;
        }
    }
    if (opts.prompt.len == 0) {
        try stdout.print("--prompt is required\n", .{});
        return error.InvalidArgs;
    }

    var img = try TensorPencil.pipeline.generate(io, arena, opts, stdout);
    defer img.deinit(arena);

    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(arena);
    try TensorPencil.image.encodePngRgb(arena, &png, img.rgb, img.width, img.height);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = png.items });
    try stdout.print("wrote {s} ({d}x{d})\n", .{ out_path, img.width, img.height });
}

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
    var dec = try vae.Decoder.load(arena, &st);
    defer dec.deinit();

    const start = std.Io.Clock.real.now(io);
    const planar = try dec.decode(io, arena, z, zh, zw);
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
