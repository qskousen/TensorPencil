//! tp-llm — LLM chat CLI, a thin driver over the TensorPencil library.
//!
//! M1 (LLM_PLAN.md): greedy decoding with full-sequence recompute per token.
//! Use -Doptimize=ReleaseFast; Debug is far too slow for a 4B model.

const std = @import("std");
const Io = std.Io;

const TensorPencil = @import("TensorPencil");
const qwen3 = TensorPencil.models.qwen3;
const qwen3_cuda = TensorPencil.models.qwen3_cuda;
const llm = TensorPencil.llm;

const usage =
    \\usage: tp-llm --model <qwen3.safetensors> --prompt <text>
    \\              [--backend cpu|zig-cuda|cuda|vulkan]
    \\              [--system <text>] [--max-tokens <n>] [--max-context <n>]
    \\              [--temperature <t>] [--top-k <n>] [--top-p <p>]
    \\              [--repeat-penalty <r>] [--seed <n>] [--greedy]
    \\
    \\Sampling defaults follow Qwen3 non-thinking recommendations:
    \\temperature 0.7, top-k 20, top-p 0.8. --greedy = --temperature 0.
    \\Seed defaults to the clock; pass --seed for reproducible output.
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    // Per-token buffers churn (alloc/free every forward pass); the process
    // arena never frees, so the generation loop gets a real allocator.
    const gpa = std.heap.smp_allocator;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var system: ?[]const u8 = null;
    var backend: BackendKind = .cpu;
    var profile = false;
    var opts: llm.engine.Options = .{ .seed = @truncate(@as(u96, @bitCast(Io.Clock.real.now(io).nanoseconds))) };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--model")) {
            model_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--prompt")) {
            prompt = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--system")) {
            system = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            opts.max_new_tokens = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--max-context")) {
            opts.max_context = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--temperature")) {
            opts.sampling.temperature = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--top-k")) {
            opts.sampling.top_k = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--top-p")) {
            opts.sampling.top_p = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--repeat-penalty")) {
            opts.sampling.repeat_penalty = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--greedy")) {
            opts.sampling.temperature = 0;
        } else if (std.mem.eql(u8, a, "--profile")) {
            profile = true;
        } else if (std.mem.eql(u8, a, "--backend")) {
            const name = try nextArg(args, &i);
            backend = std.meta.stringToEnum(BackendKind, name) orelse {
                try stdout.print("unknown backend: {s} (cpu | zig-cuda | cuda | vulkan)\n", .{name});
                try stdout.flush();
                return error.InvalidArgument;
            };
        } else {
            try stdout.writeAll(usage);
            try stdout.flush();
            return error.InvalidArgument;
        }
    }
    const path = model_path orelse {
        try stdout.writeAll(usage);
        try stdout.flush();
        return error.MissingArgument;
    };
    const user_prompt = prompt orelse {
        try stdout.writeAll(usage);
        try stdout.flush();
        return error.MissingArgument;
    };

    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(arena, &st);
    defer lm.deinit();
    var tok = try TensorPencil.tokenizer.Tokenizer.init(arena);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    if (system) |s| try llm.chat.appendSystem(&tok, gpa, s, &ids);
    try llm.chat.appendUser(&tok, gpa, user_prompt, &ids);
    try llm.chat.openAssistant(&tok, gpa, &ids);

    try stdout.print("[{s} backend, {d} prompt tokens, max {d} new, temp {d:.2}, seed {d}]\n\n", .{
        @tagName(backend), ids.items.len, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    try stdout.flush();

    const capacity = try llm.engine.capacityFor(opts, ids.items.len);
    const prompt_len = ids.items.len;
    const t_init = Io.Clock.real.now(io).nanoseconds;
    var t0: i96 = undefined; // set after backend/model setup: generation only
    const n = switch (backend) {
        .cpu => blk: {
            var model = try llm.engine.CpuModel.init(gpa, &lm, capacity);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try llm.engine.generate(&model, &tok, io, gpa, &ids, opts, stdout);
        },
        .@"zig-cuda", .cuda => blk: {
            const cuda_be = TensorPencil.gpu.cuda.Backend;
            const be = if (backend == .cuda) try cuda_be.initLibs(arena) else try cuda_be.init(arena);
            defer be.deinit();
            be.profile = profile;
            var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, capacity, prompt_len);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            const count = try llm.engine.generate(&model, &tok, io, gpa, &ids, opts, stdout);
            if (profile) {
                try stdout.print("\n\nprofile (device, sync-per-op):\n", .{});
                inline for (@typeInfo(cuda_be.ProfCat).@"enum".fields, 0..) |f, ci| {
                    if (be.prof.n[ci] > 0)
                        try stdout.print("  {s:<9} {d:>8.1} ms  ({d} launches)\n", .{ f.name, be.prof.ms[ci], be.prof.n[ci] });
                }
            }
            break :blk count;
        },
        .vulkan => blk: {
            var ctx = try TensorPencil.gpu.Context.init(arena);
            defer ctx.deinit();
            var model = try TensorPencil.models.qwen3_gpu.VulkanLM.init(gpa, ctx, &lm, capacity, prompt_len);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try llm.engine.generate(&model, &tok, io, gpa, &ids, opts, stdout);
        },
    };
    const t_end = Io.Clock.real.now(io).nanoseconds;
    const setup_s = @as(f64, @floatFromInt(t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(t_end - t0)) / 1e9;

    // Generation time includes the prefill + first-use weight upload; setup
    // is backend/model initialization.
    try stdout.print("\n\n[{d} tokens in {d:.1}s, {d:.2} tok/s; setup {d:.1}s]\n", .{
        n, elapsed_s, @as(f64, @floatFromInt(n)) / elapsed_s, setup_s,
    });
    try stdout.flush();
}

const BackendKind = enum { cpu, @"zig-cuda", cuda, vulkan };

fn nextArg(args: []const [:0]const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgument;
    return args[i.*];
}
