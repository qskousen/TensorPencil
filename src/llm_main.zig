//! tp-llm — LLM chat CLI, a thin driver over the TensorPencil library.
//!
//! One-shot (--prompt) or interactive multi-turn chat (no --prompt) over the
//! Qwen3-VL-4B stack on the cpu / vulkan / zig-cuda / cuda backends; see
//! LLM_PLAN.md. Use -Doptimize=ReleaseFast; Debug is far too slow for a 4B
//! model.

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
    \\              [--spec-k <n>] [--draft-model <qwen3.safetensors>]
    \\              [--eagle <eagle3.safetensors>] [--tree <nodes>]
    \\              [--vram-budget <GiB>|min]
    \\
    \\Sampling defaults follow Qwen3 non-thinking recommendations:
    \\temperature 0.7, top-k 20, top-p 0.8. --greedy = --temperature 0.
    \\Seed defaults to the clock; pass --seed for reproducible output.
    \\--spec-k enables speculative decoding (prompt-lookup drafting, up to n
    \\tokens verified per forward; lossless — same output distribution).
    \\--draft-model drafts with a smaller model instead (e.g. Qwen3-0.6B;
    \\cpu / zig-cuda / cuda backends; implies --spec-k 4 unless given).
    \\--eagle drafts with a trained EAGLE-3 head reading the target's own
    \\hidden states (zig-cuda / cuda; implies --spec-k 4 unless given).
    \\--tree drafts a branching token TREE per verify forward instead of a
    \\chain (up to <nodes> tree nodes, e.g. 16; requires --eagle and
    \\--greedy; still lossless — byte-identical to vanilla greedy).
    \\Without --prompt, tp-llm runs an interactive multi-turn chat (one user
    \\turn per stdin line; /exit or Ctrl-D quits).
    \\--vram-budget caps device memory for the WEIGHTS (GPU backends). The
    \\first weights touched are pinned resident up to the cap (minus slack
    \\for in-flight streamed weights); the remainder streams from the
    \\mmapped file every token at PCIe speed, so decode slows in proportion
    \\to the streamed fraction. "min" pins nothing and streams everything.
    \\KV cache and activations are not streamed and live outside the cap.
    \\0 (default) = driver-managed.
    \\
;

/// `--vram-budget min`: hold only the in-flight weights. The budget is a soft
/// ceiling (a single weight larger than it still uploads while physical VRAM
/// has room), so this streams everything, embed/LM-head included.
const min_vram_budget: u64 = 256 << 20;

/// How far short of --vram-budget the pinned weight prefix stops. The gap is
/// the streaming window: in-flight streamed weights live in it (it must
/// comfortably exceed the largest streamed weight, the ~780 MiB embed/LM-head
/// table, to keep the DMA pipeline deep) plus GEMM scratch. The budget is
/// soft, so a transient overshoot degrades gracefully rather than failing.
const pin_slack: u64 = 1 << 30;

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
    var draft_path: ?[]const u8 = null;
    var eagle_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var system: ?[]const u8 = null;
    var backend: BackendKind = .cpu;
    var profile = false;
    var vram_budget: u64 = 0;
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
        } else if (std.mem.eql(u8, a, "--spec-k")) {
            opts.spec_k = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--tree")) {
            opts.tree_nodes = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--draft-model")) {
            draft_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--eagle")) {
            eagle_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--vram-budget")) {
            const val = try nextArg(args, &i);
            if (std.mem.eql(u8, val, "min")) {
                vram_budget = min_vram_budget;
            } else {
                const gib = try std.fmt.parseFloat(f64, val);
                vram_budget = @intFromFloat(gib * (1 << 30));
            }
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

    var st = try TensorPencil.SafeTensors.open(arena, io, path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(arena, &st);
    defer lm.deinit();
    var tok = try TensorPencil.tokenizer.Tokenizer.init(arena);
    defer tok.deinit();

    // Draft model for speculative decoding (same tokenizer family; its own
    // KV cache and stepper). Implies spec decoding. The mapped safetensors
    // must outlive the session — weights are views into it.
    var draft_st: ?TensorPencil.SafeTensors = null;
    defer if (draft_st) |*s| s.deinit();
    var draft_lm: ?qwen3.CausalLM = null;
    defer if (draft_lm) |*d| d.deinit();
    if (draft_path) |dp| {
        if (backend == .vulkan) {
            try stdout.writeAll("--draft-model is not supported on the vulkan backend yet (use --spec-k for n-gram drafting)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        draft_st = try TensorPencil.SafeTensors.open(arena, io, dp);
        draft_lm = try qwen3.CausalLM.load(arena, &draft_st.?);
        if (opts.spec_k == 0) opts.spec_k = 4;
    }
    // EAGLE-3 head (reads the target's tapped hidden states; CUDA only).
    var eagle_st: ?TensorPencil.SafeTensors = null;
    defer if (eagle_st) |*s| s.deinit();
    if (eagle_path) |ep| {
        if (backend != .@"zig-cuda" and backend != .cuda) {
            try stdout.writeAll("--eagle requires the zig-cuda or cuda backend\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (draft_path != null) {
            try stdout.writeAll("--eagle and --draft-model are mutually exclusive\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        eagle_st = try TensorPencil.SafeTensors.open(arena, io, ep);
        if (opts.spec_k == 0) opts.spec_k = 4;
    }
    // Tree drafting (v1): EAGLE head trees, greedy only.
    if (opts.tree_nodes > 0) {
        if (eagle_path == null) {
            try stdout.writeAll("--tree requires --eagle (tree drafting needs a proposeTree drafter)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (opts.sampling.temperature != 0) {
            try stdout.writeAll("--tree requires --greedy (v1 trees are greedy-only)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
    }
    var spec_stats: llm.spec.Stats = .{};
    if (opts.spec_k > 0 or opts.tree_nodes > 0) opts.spec_stats = &spec_stats;

    // With --prompt: one-shot. Without: interactive chat (one user turn per
    // line; the KV cache carries the whole conversation across turns).
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    if (system) |s| try llm.chat.appendSystem(&tok, gpa, s, &ids);
    if (prompt) |p| {
        try llm.chat.appendUser(&tok, gpa, p, &ids);
        try llm.chat.openAssistant(&tok, gpa, &ids);
    }

    try stdout.print("[{s} backend, {d} prompt tokens, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), ids.items.len, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) try stdout.writeAll("[interactive chat: empty line to re-prompt, /exit or Ctrl-D to quit]\n");
    try stdout.writeAll("\n");
    try stdout.flush();

    // One-shot sizes the cache to the request; a chat session gets the whole
    // window. GPU activation buffers cover the first prefill (turn one for
    // one-shot; longer inputs chunk).
    const capacity = if (prompt == null) opts.max_context else try llm.engine.capacityFor(opts, ids.items.len);
    const prompt_len = if (prompt == null) @min(512, capacity) else ids.items.len;
    const t_init = Io.Clock.real.now(io).nanoseconds;
    var t0: i96 = undefined; // set after backend/model setup: generation only
    const n = switch (backend) {
        .cpu => blk: {
            if (vram_budget != 0) try stdout.writeAll("[--vram-budget ignored on the cpu backend]\n");
            var model = try llm.engine.CpuModel.init(gpa, &lm, capacity);
            defer model.deinit();
            if (draft_lm) |*dlm| {
                var draft = try llm.engine.CpuModel.init(gpa, dlm, capacity);
                defer draft.deinit();
                var drafter = try llm.spec.ModelDrafter(llm.engine.CpuModel).init(gpa, io, &draft);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                break :blk try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt);
            }
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt);
        },
        .@"zig-cuda", .cuda => blk: {
            const cuda_be = TensorPencil.gpu.cuda.Backend;
            const be = if (backend == .cuda) try cuda_be.initLibs(arena) else try cuda_be.init(arena);
            defer be.deinit();
            be.profile = profile;
            be.budget_override = vram_budget; // --vram-budget: stream weights past this cap
            be.pin_budget = vram_budget -| pin_slack; // pin the first-touched prefix, stream the rest
            be.stream_window = @min(pin_slack, vram_budget); // in-flight streamed-weight cap / enqueue pacing
            if (vram_budget != 0) {
                // Page-lock the checkpoint mmaps so streamed weights DMA
                // directly at full PCIe bandwidth, prefetched one layer
                // ahead of compute (skipped when everything stays resident:
                // registration pins host RAM for no benefit).
                if (st.mapping) |m| be.enableDirectStreaming(m);
                if (draft_st) |ds| if (ds.mapping) |m| be.enableDirectStreaming(m);
                if (eagle_st) |es| if (es.mapping) |m| be.enableDirectStreaming(m);
            }
            var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, capacity, prompt_len);
            defer model.deinit();
            var count: usize = undefined;
            if (eagle_st) |*est| {
                try model.enableTaps(TensorPencil.models.eagle3.default_tap_layers);
                if (opts.tree_nodes > 0) try model.enableTree();
                var head = try TensorPencil.models.eagle3.Eagle3Head.load(gpa, be, est, &lm, capacity);
                defer head.deinit();
                var drafter = try TensorPencil.models.eagle3.Eagle3Drafter.init(gpa, &head, &model);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt);
            } else if (draft_lm) |*dlm| {
                var draft = try qwen3_cuda.CudaLM.init(gpa, be, dlm, capacity, prompt_len);
                defer draft.deinit();
                // First claim on the pin budget goes to the draft: its weights
                // are read once per drafted token, the target's only once per
                // ~k accepted tokens, so pinned draft bytes save more PCIe.
                if (vram_budget != 0) try draft.prewarmWeights();
                var drafter = try llm.spec.ModelDrafter(qwen3_cuda.CudaLM).init(gpa, io, &draft);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt);
            } else {
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt);
            }
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
            ctx.budget_override = vram_budget; // --vram-budget: stream weights past this cap
            ctx.pin_budget = vram_budget -| pin_slack; // pin the first-touched prefix, stream the rest
            var model = try TensorPencil.models.qwen3_gpu.VulkanLM.init(gpa, ctx, &lm, capacity, prompt_len);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt);
        },
    };
    const t_end = Io.Clock.real.now(io).nanoseconds;
    const setup_s = @as(f64, @floatFromInt(t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(t_end - t0)) / 1e9;

    // One-shot summary (generation time includes the prefill + first-use
    // weight upload; setup is backend/model initialization). Chat sessions
    // print per-turn stats instead — elapsed here would count typing time.
    if (prompt != null) {
        try stdout.print("\n\n[{d} tokens in {d:.1}s, {d:.2} tok/s; setup {d:.1}s]\n", .{
            n, elapsed_s, @as(f64, @floatFromInt(n)) / elapsed_s, setup_s,
        });
    } else {
        try stdout.print("[session over: {d} tokens generated; setup was {d:.1}s]\n", .{ n, setup_s });
    }
    if (opts.spec_k > 0 or opts.tree_nodes > 0) {
        const pct = if (spec_stats.drafted > 0)
            100.0 * @as(f64, @floatFromInt(spec_stats.accepted)) / @as(f64, @floatFromInt(spec_stats.drafted))
        else
            0.0;
        try stdout.print("[spec: {d}/{d} drafts accepted ({d:.0}%), {d} verify forwards for {d} tokens]\n", .{
            spec_stats.accepted, spec_stats.drafted, pct, spec_stats.forwards, n,
        });
    }
    try stdout.flush();
}

/// Drive one generation (--prompt) or an interactive chat loop (no
/// --prompt): each stdin line becomes a user turn, the assistant's reply
/// streams back, and the turn is sealed so the KV cache carries the whole
/// conversation. `drafter` is null (vanilla / n-gram speculative via
/// engine.generate) or a *spec.ModelDrafter. Returns total tokens generated.
fn runSession(
    model: anytype,
    drafter: anytype,
    tok: *const TensorPencil.tokenizer.Tokenizer,
    io: Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: llm.engine.Options,
    stdout: *Io.Writer,
    prompt: ?[]const u8,
) !usize {
    if (prompt != null) {
        return doGenerate(model, drafter, tok, io, gpa, ids, opts, stdout);
    }

    var stdin_buffer: [64 * 1024]u8 = undefined;
    var stdin_reader: Io.File.Reader = .initStreaming(.stdin(), io, &stdin_buffer);
    const stdin = &stdin_reader.interface;

    var total: usize = 0;
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const raw = (try stdin.takeDelimiter('\n')) orelse break; // null = EOF
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "/exit")) break;

        try llm.chat.appendUser(tok, gpa, line, ids);
        try llm.chat.openAssistant(tok, gpa, ids);
        const t0 = Io.Clock.real.now(io).nanoseconds;
        const n = doGenerate(model, drafter, tok, io, gpa, ids, opts, stdout) catch |err| switch (err) {
            error.ContextFull => {
                try stdout.writeAll("\n[context window full]\n");
                try stdout.flush();
                break;
            },
            else => return err,
        };
        try llm.chat.closeAssistant(gpa, ids);
        total += n;

        const dt = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t0)) / 1e9;
        try stdout.print("\n[{d} tok, {d:.1} tok/s, ctx {d}/{d}]\n", .{
            n, @as(f64, @floatFromInt(n)) / dt, model.cached(), model.cached() + model.remaining(),
        });
        try stdout.flush();
        if (model.remaining() == 0) {
            try stdout.writeAll("[context window full]\n");
            try stdout.flush();
            break;
        }
    }
    return total;
}

/// One turn's generation: engine.generate for null drafters (vanilla, or
/// n-gram speculative when spec_k > 0), spec.generate for a model drafter.
fn doGenerate(
    model: anytype,
    drafter: anytype,
    tok: *const TensorPencil.tokenizer.Tokenizer,
    io: Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: llm.engine.Options,
    stdout: *Io.Writer,
) !usize {
    if (comptime @TypeOf(drafter) == @TypeOf(null)) {
        return llm.engine.generate(model, tok, io, gpa, ids, opts, stdout);
    } else {
        return llm.spec.generate(model, drafter, tok, io, gpa, ids, opts, stdout);
    }
}

const BackendKind = enum { cpu, @"zig-cuda", cuda, vulkan };

fn nextArg(args: []const [:0]const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgument;
    return args[i.*];
}
