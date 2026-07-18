//! tp-llm — LLM chat CLI, a thin driver over the TensorPencil library.
//!
//! One-shot (--prompt) or interactive multi-turn chat (no --prompt) over the
//! Qwen3-VL-4B stack on the cpu / vulkan / zig-cuda / cuda backends; see
//! LLM_PLAN.md. Use -Doptimize=ReleaseFast; Debug is far too slow for a 4B
//! model.

const std = @import("std");
const Io = std.Io;

const TensorPencil = @import("TensorPencil");
const vips = @import("vips");
const qwen3 = TensorPencil.models.qwen3;
const qwen3_cuda = TensorPencil.models.qwen3_cuda;
const llm = TensorPencil.llm;

const usage =
    \\usage: tp-llm --model <qwen3.safetensors> --prompt <text>
    \\              [--backend cpu|zig-cuda|cuda|vulkan]
    \\              [--system <text>] [--max-tokens <n>] [--max-context <n>]
    \\              [--kv-dtype f32|f16]
    \\              [--temperature <t>] [--top-k <n>] [--top-p <p>] [--min-p <p>]
    \\              [--repeat-penalty <r>] [--repeat-last-n <n>]
    \\              [--presence-penalty <p>] [--frequency-penalty <p>]
    \\              [--seed <n>] [--greedy] [--no-think]
    \\              [--spec-k <n>] [--draft-model <qwen3.safetensors>]
    \\              [--eagle <eagle3.safetensors>] [--tree <nodes>]
    \\              [--vram-budget <GiB>|min] [--cpu-layers tail|attn] [--offload-grow]
    \\              [--image <image> --mmproj <mmproj.gguf>]
    \\
    \\--max-context caps the context window; the default is the model's
    \\trained context length (up to 128k). Only a small initial slice of KV
    \\rows (4096, or the prompt size) is committed up front; the cache grows
    \\in place on demand, so a large window costs VRAM only as the
    \\conversation actually fills it. Under VRAM pressure growth evicts
    \\least-recently-used weights into the streaming path (slower decode)
    \\instead of failing, and the session ends only when nothing more can be
    \\freed. Speculative runs (--spec-k/--draft-model/--eagle/--tree) reserve
    \\the whole window physically up front, so their default window stays 4096.
    \\--backend vulkan also reserves the whole window up front (no growable
    \\buffers), so without an explicit --max-context it auto-sizes to the
    \\largest window whose KV reservation fits in VRAM beside the weights,
    \\capped at the trained length.
    \\Sampling defaults follow Qwen3 non-thinking recommendations:
    \\temperature 0.7, top-k 20, top-p 0.8. --greedy = --temperature 0.
    \\--min-p drops candidates below <p> times the top candidate's
    \\probability (default 0 = off). The penalties scan the last
    \\--repeat-last-n context tokens (default 64; 0 disables them):
    \\--repeat-penalty divides a seen token's positive logit (default 1 =
    \\off), --presence-penalty subtracts a flat amount from every seen
    \\token, --frequency-penalty subtracts per occurrence (defaults 0 =
    \\off; llama.cpp formulas). Any active penalty needs the full logits,
    \\so it takes the CPU-sampling path on GPU backends (slower per token).
    \\Seed defaults to the clock; pass --seed for reproducible output.
    \\Reasoning is on by default for models that support it (Qwen3.5, Gemma 4):
    \\the model emits a thought block before its answer. --no-think disables it
    \\(the turn is primed with an empty thought so the model answers directly);
    \\--think forces it on. No effect on non-reasoning models (e.g. Gemma 3).
    \\--spec-k enables speculative decoding (prompt-lookup drafting, up to n
    \\tokens verified per forward; lossless — same output distribution).
    \\--draft-model drafts with a smaller model instead (e.g. Qwen3-0.6B;
    \\cpu / zig-cuda / cuda backends; implies --spec-k 4 unless given).
    \\--eagle drafts with a trained EAGLE-3 head reading the target's own
    \\hidden states (zig-cuda / cuda; implies --spec-k 4 unless given).
    \\--tree drafts a branching token TREE per verify forward instead of a
    \\chain (up to <nodes> tree nodes, e.g. 16; requires --eagle and
    \\--greedy; still lossless — byte-identical to vanilla greedy).
    \\Without --prompt, tp-llm runs an interactive multi-turn chat: Enter
    \\sends the message, Shift-Enter (or Alt-Enter) inserts a newline, and
    \\pasted multi-line text stays one message (bracketed paste); /exit or
    \\Ctrl-D quits. Piped stdin reads one message per line. With --mmproj
    \\on the zig-cuda/cuda backends, chat turns may attach images anywhere
    \\in the message as @path.jpg or @"path with spaces.png" mentions
    \\(multiple per turn; qwen35 GGUF models). Images decode via system
    \\libvips: jpeg/png/webp/gif/tiff/bmp, EXIF rotation applied.
    \\--vram-budget caps device memory for the WEIGHTS (GPU backends). The
    \\first weights touched are pinned resident up to the cap (minus slack
    \\for in-flight streamed weights); the remainder streams from the
    \\mmapped file every token at PCIe speed, so decode slows in proportion
    \\to the streamed fraction. "min" pins nothing and streams everything.
    \\KV cache and activations are not streamed and live outside the cap.
    \\0 (default) = pin as much of the model as the free VRAM holds, so a
    \\model that fits stays fully resident (streaming a model that fits is
    \\~3.6x slower); only a model larger than VRAM streams its tail.
    \\--cpu-layers (qwen35 GGUF, zig-cuda/cuda, text-only) fits the model under
    \\--vram-budget a different way: instead of streaming weights, it runs whole
    \\layers on the CPU (their weights never touch the device). The layer count
    \\is derived from the budget; "tail" keeps a contiguous device prefix (the
    \\last N layers go to CPU), "attn" evicts the KV-growing attention layers
    \\first. Requires --vram-budget. Trades PCIe streaming for host compute —
    \\faster than streaming only when the CPU keeps up (slow PCIe / fast CPU);
    \\degrades gradually with the budget and avoids the graph-eviction cliff.
    \\--offload-grow (implies the split, attn policy unless --cpu-layers given)
    \\is the DYNAMIC form: it starts with as many layers on the GPU as fit under
    \\--vram-budget and migrates more to the CPU on demand as the KV cache grows,
    \\so short contexts run at full speed and long ones degrade gradually instead
    \\of hitting the streaming cliff.
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
    var debug_batch: usize = 0;
    var image_path: ?[]const u8 = null;
    var mmproj_path: ?[]const u8 = null;
    var draft_path: ?[]const u8 = null;
    var eagle_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var system: ?[]const u8 = null;
    var backend: BackendKind = .cpu;
    var profile = false;
    var vram_budget: u64 = 0;
    var cpu_split: ?TensorPencil.models.qwen35_cuda.CpuSplitPolicy = null;
    var dynamic_offload = false;
    var max_context_arg: ?usize = null;
    var opts: llm.engine.Options = .{ .seed = @truncate(@as(u96, @bitCast(Io.Clock.real.now(io).nanoseconds))) };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--model")) {
            model_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--debug-batch")) {
            debug_batch = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--image")) {
            image_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--mmproj")) {
            mmproj_path = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--prompt")) {
            prompt = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--system")) {
            system = try nextArg(args, &i);
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            opts.max_new_tokens = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--max-context")) {
            max_context_arg = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--kv-dtype")) {
            const name = try nextArg(args, &i);
            opts.kv_dtype = llm.kv_cache.KvDtype.parse(name) orelse {
                try stdout.print("unknown --kv-dtype: {s} (f32 | f16)\n", .{name});
                try stdout.flush();
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, a, "--temperature")) {
            opts.sampling.temperature = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--top-k")) {
            opts.sampling.top_k = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--top-p")) {
            opts.sampling.top_p = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--min-p")) {
            opts.sampling.min_p = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--repeat-penalty")) {
            opts.sampling.repeat_penalty = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--repeat-last-n")) {
            opts.sampling.repeat_last_n = try std.fmt.parseInt(usize, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--presence-penalty")) {
            opts.sampling.presence_penalty = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--frequency-penalty")) {
            opts.sampling.frequency_penalty = try std.fmt.parseFloat(f32, try nextArg(args, &i));
        } else if (std.mem.eql(u8, a, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, try nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, a, "--greedy")) {
            opts.sampling.temperature = 0;
        } else if (std.mem.eql(u8, a, "--think")) {
            llm.chat.setThinking(true);
        } else if (std.mem.eql(u8, a, "--no-think")) {
            llm.chat.setThinking(false);
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
                vram_budget = llm.session.min_vram_budget;
            } else {
                const gib = try std.fmt.parseFloat(f64, val);
                vram_budget = @intFromFloat(gib * (1 << 30));
            }
        } else if (std.mem.eql(u8, a, "--cpu-layers")) {
            const name = try nextArg(args, &i);
            cpu_split = std.meta.stringToEnum(TensorPencil.models.qwen35_cuda.CpuSplitPolicy, name) orelse {
                try stdout.print("unknown --cpu-layers policy: {s} (tail | attn)\n", .{name});
                try stdout.flush();
                return error.InvalidArgument;
            };
        } else if (std.mem.eql(u8, a, "--offload-grow")) {
            dynamic_offload = true;
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

    // Container chosen by extension: .gguf or safetensors.
    var st = try ModelFile.open(arena, io, path);
    defer st.deinit();

    // Default context ceiling: the model's trained context length (growth
    // commits KV rows only as the conversation fills, so a big ceiling costs
    // VRAM lazily — and stops gracefully when even weight eviction can't
    // free enough). Speculative sessions allocate the whole window physically
    // up front, so without an explicit --max-context they keep the 4096
    // default. Vulkan also reserves up front but AUTO-SIZES to what VRAM holds
    // in its backend block below; 4096 here is a transient placeholder it
    // overrides (it must be set before capacityPlan runs).
    const fixed_session = opts.spec_k > 0 or opts.tree_nodes > 0 or draft_path != null or eagle_path != null;
    opts.max_context = max_context_arg orelse
        (if (fixed_session or backend == .vulkan) 4096 else @min(trainedContext(&st), auto_context_cap));

    // Architecture dispatch: qwen35 (hybrid DeltaNet) has its own model and
    // steppers (cpu / zig-cuda / cuda), and no speculative decoding (the
    // recurrent state cannot roll back past rejected drafts).
    if (st == .gguf and st.gguf.getStr("general.architecture") != null and
        std.mem.eql(u8, st.gguf.getStr("general.architecture").?, "qwen35"))
    {
        if (draft_path != null or eagle_path != null or opts.spec_k > 0 or opts.tree_nodes > 0) {
            try stdout.writeAll("speculative decoding is not supported for qwen35 models (recurrent state cannot roll back)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (image_path != null) {
            if (backend == .cpu or backend == .vulkan) {
                try stdout.writeAll("--image requires the zig-cuda or cuda backend\n");
                try stdout.flush();
                return error.InvalidArgument;
            }
            if (mmproj_path == null) {
                try stdout.writeAll("--image requires --mmproj <mmproj.gguf> (the vision tower ships separately)\n");
                try stdout.flush();
                return error.InvalidArgument;
            }
            if (prompt == null) {
                try stdout.writeAll("--image requires --prompt (one-shot); in interactive chat, attach images with @path.png mentions instead\n");
                try stdout.flush();
                return error.InvalidArgument;
            }
        }
        return runQwen35(arena, gpa, io, &st.gguf, backend, vram_budget, cpu_split, dynamic_offload, image_path, mmproj_path, prompt, system, opts, profile, debug_batch, stdout);
    }

    // Gemma 3 (sandwich norms, dual local/global RoPE): its own CPU model.
    if (st == .gguf and st.gguf.getStr("general.architecture") != null and
        std.mem.eql(u8, st.gguf.getStr("general.architecture").?, "gemma3"))
    {
        if (draft_path != null or eagle_path != null or opts.spec_k > 0 or opts.tree_nodes > 0) {
            try stdout.writeAll("speculative decoding is not supported for gemma3 models yet\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (cpu_split != null or dynamic_offload) {
            try stdout.writeAll("--cpu-layers / --offload-grow is only supported for qwen35 models\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        return runGemma3(arena, gpa, io, &st.gguf, backend, vram_budget, image_path, mmproj_path, prompt, system, opts, profile, stdout);
    }

    // Gemma 4 (per-layer local/global attention geometry, V-norm, out_scale,
    // proportional RoPE): text-only CPU model for now (GPU + vision are
    // follow-ups).
    if (st == .gguf and st.gguf.getStr("general.architecture") != null and
        std.mem.eql(u8, st.gguf.getStr("general.architecture").?, "gemma4"))
    {
        if (draft_path != null or eagle_path != null or opts.spec_k > 0 or opts.tree_nodes > 0) {
            try stdout.writeAll("speculative decoding is not supported for gemma4 models yet\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (cpu_split != null or dynamic_offload) {
            try stdout.writeAll("--cpu-layers / --offload-grow is only supported for qwen35 models\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        return runGemma4(arena, gpa, io, &st.gguf, backend, image_path, mmproj_path, prompt, system, opts, profile, stdout);
    }

    if (cpu_split != null or dynamic_offload) {
        try stdout.writeAll("--cpu-layers / --offload-grow (hybrid CPU/GPU split) is only supported for qwen35 (Qwen3.5/3.6) models for now\n");
        try stdout.flush();
        return error.InvalidArgument;
    }

    if (image_path != null) {
        try stdout.writeAll("--image is only supported for qwen35 (Qwen3.5/3.6) GGUF models\n");
        try stdout.flush();
        return error.InvalidArgument;
    }

    return runQwen3(arena, gpa, io, &st, backend, vram_budget, draft_path, eagle_path, max_context_arg, prompt, system, opts, profile, stdout);
}

/// One-shot / chat session for a Qwen3 model (the Krea 2 text-encoder stack /
/// the tp-llm target): cpu / zig-cuda / cuda / vulkan, plus speculative decoding
/// (n-gram --spec-k, a --draft-model, or an --eagle head with optional --tree).
fn runQwen3(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: Io,
    st: *ModelFile,
    backend: BackendKind,
    vram_budget: u64,
    draft_path: ?[]const u8,
    eagle_path: ?[]const u8,
    max_context_arg: ?usize,
    prompt: ?[]const u8,
    system: ?[]const u8,
    opts_in: llm.engine.Options,
    profile: bool,
    stdout: *Io.Writer,
) !void {
    var opts = opts_in;
    var lm = try qwen3.CausalLM.load(arena, st.store());
    defer lm.deinit();
    if (backend == .vulkan and lm.hasBlockQuantWeights()) {
        try stdout.writeAll("GGUF block-quantized weights (Q8_0/Q4_K/Q5_K/Q6_K) run on cpu / zig-cuda / cuda only for now\n");
        try stdout.flush();
        return error.InvalidArgument;
    }
    // Tokenizer: a GGUF checkpoint carries its own vocab (which may differ from
    // the embedded Qwen3 one — e.g. Qwen3.6's 248k tokens); fall back to the
    // embedded tokenizer when the file has none (ComfyUI-style conversions strip it).
    var tok = switch (st.*) {
        .gguf => |*g| TensorPencil.tokenizer.Tokenizer.initFromGguf(arena, g) catch |err| switch (err) {
            error.MissingTokenizer => try TensorPencil.tokenizer.Tokenizer.init(arena),
            else => return err,
        },
        .safetensors => try TensorPencil.tokenizer.Tokenizer.init(arena),
    };
    defer tok.deinit();
    llm.chat.applyTokenizer(&tok);

    // Draft model for speculative decoding (same tokenizer family; its own KV
    // cache and stepper). Implies spec decoding. The mapped checkpoint must
    // outlive the session — weights are views into it.
    var draft_st: ?ModelFile = null;
    defer if (draft_st) |*s| s.deinit();
    var draft_lm: ?qwen3.CausalLM = null;
    defer if (draft_lm) |*d| d.deinit();
    if (draft_path) |dp| {
        if (backend == .vulkan) {
            try stdout.writeAll("--draft-model is not supported on the vulkan backend yet (use --spec-k for n-gram drafting)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        draft_st = try ModelFile.open(arena, io, dp);
        draft_lm = try qwen3.CausalLM.load(arena, draft_st.?.store());
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

    // Vulkan reserves the whole KV window up front (no growable buffers), so
    // bring its device up now and auto-size the context to what VRAM holds
    // before the banner and capacityPlan below read opts.max_context (an
    // explicit --max-context always wins). CUDA defaults to the trained length
    // and grows KV lazily. The context is reused by session.run's vulkan arm.
    var vk_ctx: ?*TensorPencil.gpu.Context = null;
    defer if (vk_ctx) |c| c.deinit();
    if (backend == .vulkan) {
        const ctx = try TensorPencil.gpu.Context.init(arena);
        vk_ctx = ctx;
        if (max_context_arg == null) {
            const free = ctx.liveVram();
            const weight_est: u64 = if (st.mapping()) |m| m.len else 0;
            const per_tok = TensorPencil.models.qwen3_gpu.VulkanLM.kvWindowBytes(1);
            const avail = free -| weight_est -| (llm.session.pin_slack * 2); // activations + freqs + margin
            const trained = @min(trainedContext(st), auto_context_cap);
            opts.max_context = std.math.clamp(avail / per_tok, 4096, trained);
        }
    }

    try stdout.print("[{s} backend, {d} prompt tokens, ctx window {d}, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), ids.items.len, opts.max_context, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) try stdout.writeAll("[interactive chat: Enter sends, Shift- or Alt-Enter adds a line, /exit or Ctrl-D quits]\n");
    try stdout.writeAll("\n");
    try stdout.flush();

    // One-shot caps the window at the request; a chat session gets the whole
    // --max-context window. Speculative runs stay fixed-capacity (the tree
    // batch region and EAGLE tap buffers stride by capacity).
    const cap = try capacityPlan(opts, prompt, ids.items.len, opts.spec_k > 0 or opts.tree_nodes > 0 or draft_path != null or eagle_path != null);
    const prompt_len = if (prompt == null) @min(512, cap.max) else ids.items.len;

    const t_init = Io.Clock.real.now(io).nanoseconds;
    // Dense CUDA bring-up (qwen3 has no CPU/GPU split): pin under the budget and
    // page-lock the target/draft/eagle mmaps so a streamed tail DMAs at full PCIe.
    const be_cuda = try llm.session.bringUpCuda(arena, backend, profile, vram_budget);
    defer if (be_cuda) |b| b.deinit();
    if (be_cuda) |b| if (vram_budget != 0) {
        if (st.mapping()) |m| b.enableDirectStreaming(m);
        if (draft_st) |ds| if (ds.mapping()) |m| b.enableDirectStreaming(m);
        if (eagle_st) |es| if (es.mapping) |m| b.enableDirectStreaming(m);
    };

    const dev: llm.session.Devices = .{ .cu_be = be_cuda, .vk_ctx = vk_ctx };
    const driver: Qwen3Driver = .{
        .be_cuda = be_cuda,
        .draft_lm = if (draft_lm) |*d| d else null,
        .eagle_st = if (eagle_st) |*e| e else null,
        .target_lm = &lm,
        .cap = cap,
        .prompt_len = prompt_len,
        .vram_budget = vram_budget,
        .tok = &tok,
        .io = io,
        .gpa = gpa,
        .ids = &ids,
        .opts = opts,
        .stdout = stdout,
        .prompt = prompt,
    };
    const res = try llm.session.run(Qwen3Spec, dev, backend, &lm, prompt_len, llm.session.no_prefill, driver, io, gpa, cap, vram_budget, stdout);
    if (profile) if (be_cuda) |b| {
        try stdout.print("\n\nprofile (device, sync-per-op):\n", .{});
        try llm.session.printCudaProfile(stdout, b);
    };

    const setup_s = @as(f64, @floatFromInt(res.t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - res.t0)) / 1e9;
    try llm.session.printSummary(stdout, prompt != null, res.n, setup_s, elapsed_s, res.stats);
    if (opts.spec_k > 0 or opts.tree_nodes > 0) {
        const pct = if (spec_stats.drafted > 0)
            100.0 * @as(f64, @floatFromInt(spec_stats.accepted)) / @as(f64, @floatFromInt(spec_stats.drafted))
        else
            0.0;
        try stdout.print("[spec: {d}/{d} drafts accepted ({d:.0}%), {d} verify forwards for {d} tokens]\n", .{
            spec_stats.accepted, spec_stats.drafted, pct, spec_stats.forwards, res.n,
        });
    }
    try stdout.flush();
}

/// Session-lifetime vision context for @image mentions in interactive
/// turns (qwen35 on the CUDA backends).
/// The session's vision tower + backend for interactive @image mentions,
/// one variant per architecture (Qwen3-VL's ViT vs Gemma 3's SigLIP tower).
const ImageChat = union(enum) {
    qwen35: struct { vit: *const TensorPencil.models.vit35.Vit, be: *TensorPencil.gpu.cuda.Backend },
    gemma3: struct { vit: *const TensorPencil.models.gemma_vit.Vit, be: *TensorPencil.gpu.cuda.Backend, io: Io },
};

/// Build and prefill an interactive turn containing @image mentions:
/// encode each mentioned PNG on the session ViT, append the interleaved
/// user turn + assistant open, then prefill everything through the last
/// image (text via prefill, image rows via prefillImage) so the engine's
/// cached()-based prefill takes over from the tail. Returns false — with
/// a message, and ids untouched or rolled back — when images are
/// unavailable, a file fails to decode, or the turn doesn't fit the
/// remaining context.
fn imageTurn(
    model: anytype,
    img_chat: ?ImageChat,
    tok: *const TensorPencil.tokenizer.Tokenizer,
    gpa: std.mem.Allocator,
    parts: []const llm.chat.Part,
    ids: *std.ArrayList(u32),
    stdout: *Io.Writer,
) !bool {
    const M = switch (@typeInfo(@TypeOf(model))) {
        .pointer => |p| p.child,
        else => @TypeOf(model),
    };
    if (comptime !@hasDecl(M, "prefillImage")) {
        try stdout.writeAll("[@image mentions need the zig-cuda or cuda backend]\n");
        return false;
    } else {
        const ic = img_chat orelse {
            try stdout.writeAll("[@image mentions need --mmproj <mmproj.gguf> and the zig-cuda or cuda backend]\n");
            return false;
        };

        // Encoded image (embeds owned by gpa), arch-independent so both
        // vision towers feed the same prefill loop.
        const Img = struct { embeds: []f32, grid_w: usize, grid_h: usize };
        var imgs: std.ArrayList(Img) = .empty;
        defer {
            for (imgs.items) |im| gpa.free(im.embeds);
            imgs.deinit(gpa);
        }
        var segs: std.ArrayList(llm.chat.Segment) = .empty;
        defer segs.deinit(gpa);
        for (parts) |p| switch (p) {
            .text => |t| try segs.append(gpa, .{ .text = t }),
            .image => |path| {
                const dec = vips.loadRgb(gpa, path) catch |err| {
                    try stdout.print("[can't load {s}: {t}]\n", .{ path, err });
                    return false;
                };
                defer gpa.free(dec.pixels);
                const im: Img = switch (ic) {
                    .qwen35 => |q| blk: {
                        const e = try TensorPencil.models.vit35_cuda.encode(q.vit, q.be, gpa, dec.pixels, dec.width, dec.height);
                        break :blk .{ .embeds = e.embeds, .grid_w = e.grid_w, .grid_h = e.grid_h };
                    },
                    .gemma3 => |g| blk: {
                        const e = try TensorPencil.models.gemma_vit_cuda.encode(g.vit, g.be, g.io, gpa, dec.pixels, dec.width, dec.height);
                        break :blk .{ .embeds = e.embeds, .grid_w = e.grid_w, .grid_h = e.grid_h };
                    },
                };
                try imgs.append(gpa, im);
                try segs.append(gpa, .{ .image = .{ .grid_w = im.grid_w, .grid_h = im.grid_h } });
                try stdout.print("[{s}: {d}x{d} -> {d} rows]\n", .{ path, dec.width, dec.height, im.grid_w * im.grid_h });
                try stdout.flush();
            },
        };

        // Append + interleave via the shared helper (same core the GUI worker
        // uses). On a context-full growth failure, roll the turn's tokens back
        // and bail gracefully instead of erroring the session.
        const ids_before = ids.items.len;
        llm.session.prefillImageTurn(model, tok, gpa, ids, segs.items, imgs.items) catch |err| {
            if (err == error.ContextFull) {
                try stdout.print("[turn needs {d} rows, only {d} left in context]\n", .{ ids.items.len - model.cached(), model.remaining() });
                ids.shrinkRetainingCapacity(ids_before);
                return false;
            }
            return err;
        };
        return true;
    }
}

/// Read one interactive message with the raw-mode editor (Shift/Alt-Enter
/// insert newlines, bracketed paste keeps pasted newlines literal). The
/// tty is raw and the terminal modes are enabled only inside this call —
/// generation runs with the terminal cooked again, so Ctrl-C still kills
/// a running reply. Returns null at end of session (Ctrl-D, closed stdin).
fn readEditorMessage(ed: *llm.repl.Editor, fd: std.posix.fd_t, stdin: *Io.Reader, gpa: std.mem.Allocator, stdout: *Io.Writer) !?[]const u8 {
    var raw = try llm.repl.RawTty.enter(fd);
    defer raw.leave();
    try stdout.writeAll(llm.repl.enter_seq);
    try stdout.flush();
    defer {
        stdout.writeAll(llm.repl.leave_seq) catch {};
        stdout.flush() catch {};
    }
    ed.reset();
    while (true) {
        const b = stdin.takeByte() catch |err| switch (err) {
            error.EndOfStream => return null,
            else => |e| return e,
        };
        switch (try ed.feed(gpa, b, stdout)) {
            .none => try stdout.flush(),
            .cancel => {
                try stdout.writeAll("> ");
                try stdout.flush();
            },
            .submit => return ed.message(),
            .eof => return null,
        }
    }
}

/// Drive one generation (--prompt) or an interactive chat loop (no
/// --prompt): each message becomes a user turn, the assistant's reply
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
    img_chat: ?ImageChat,
) !usize {
    if (prompt != null) {
        return doGenerate(model, drafter, tok, io, gpa, ids, opts, stdout);
    }

    const stdin_file: Io.File = .stdin();
    var stdin_buffer: [64 * 1024]u8 = undefined;
    var stdin_reader: Io.File.Reader = .initStreaming(stdin_file, io, &stdin_buffer);
    const stdin = &stdin_reader.interface;
    // A real terminal gets the raw-mode editor (multi-line paste stays one
    // message, Shift/Alt-Enter insert newlines); piped stdin keeps one
    // message per line so scripted sessions are unchanged.
    const tty = stdin_file.isTty(io) catch false;
    var ed: llm.repl.Editor = .{};
    defer ed.deinit(gpa);

    var total: usize = 0;
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const msg = if (tty)
            (try readEditorMessage(&ed, stdin_file.handle, stdin, gpa, stdout)) orelse break
        else
            (try stdin.takeDelimiter('\n')) orelse break; // null = EOF
        const line = std.mem.trim(u8, msg, " \t\r\n");
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "/exit")) break;

        // @image mentions become interleaved vision blocks, prefilled
        // (with prefillImage) before the engine sees the turn's tail.
        var parts: std.ArrayList(llm.chat.Part) = .empty;
        defer parts.deinit(gpa);
        var has_image = false;
        if (std.mem.indexOfScalar(u8, line, '@') != null) {
            try llm.chat.parseImageMentions(gpa, line, &parts);
            for (parts.items) |p| {
                if (p == .image) has_image = true;
            }
        }
        if (has_image) {
            if (!try imageTurn(model, img_chat, tok, gpa, parts.items, ids, stdout)) continue;
        } else {
            try llm.chat.appendUser(tok, gpa, line, ids);
            try llm.chat.openAssistant(tok, gpa, ids);
        }
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

        // The window a growable cache reports is its ceiling, not the
        // currently committed slice; the session is only over when the
        // ceiling itself is reached. `Stats.of` also carries device VRAM.
        const st = llm.session.Stats.of(model);
        const window = st.window;
        const dt = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t0)) / 1e9;
        var vbuf: [32]u8 = undefined;
        try stdout.print("\n[{d} tok, {d:.1} tok/s, ctx {d}/{d}{s}]\n", .{
            n, @as(f64, @floatFromInt(n)) / dt, st.tokens, window, st.vramSuffix(&vbuf),
        });
        try stdout.flush();
        if (model.cached() >= window) {
            try stdout.writeAll("[context window full]\n");
            try stdout.flush();
            break;
        }
    }
    return total;
}

/// The generation `driver` for the non-speculative architectures (gemma3,
/// gemma4, qwen35): drive one turn / the chat loop with no model drafter.
/// `session.run` calls `drive` on the constructed stepper. (qwen3's driver,
/// which selects a draft / EAGLE drafter, lives with the qwen3 path.)
const SimpleDriver = struct {
    tok: *const TensorPencil.tokenizer.Tokenizer,
    io: Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: llm.engine.Options,
    stdout: *Io.Writer,
    prompt: ?[]const u8,
    img_chat: ?ImageChat,
    pub fn drive(self: SimpleDriver, model: anytype) !llm.session.RunResult {
        const t0 = Io.Clock.real.now(self.io).nanoseconds;
        const n = try runSession(model, null, self.tok, self.io, self.gpa, self.ids, self.opts, self.stdout, self.prompt, self.img_chat);
        return .{ .n = n, .t0 = t0, .stats = llm.session.Stats.of(model) };
    }
};

/// Vulkan KV-window size for the gemma3 weight budget (2 x n_layers x cap x
/// kvDim x f32). gemma4 has no Vulkan backend, so its window fn is never called.
fn gemma3KvWindow(lm: *const TensorPencil.models.gemma3.Model, cap: llm.engine.Capacity) u64 {
    return 2 * @as(u64, lm.cfg.n_layers) * cap.max * lm.cfg.kvDim() * 4;
}
fn gemma4KvWindow(lm: *const TensorPencil.models.gemma4.Model, cap: llm.engine.Capacity) u64 {
    _ = lm;
    _ = cap;
    return 0; // gemma4 Spec has Vulkan = void; never reached
}
fn qwen35KvWindow(lm: *const TensorPencil.models.qwen35.Model, cap: llm.engine.Capacity) u64 {
    _ = lm;
    _ = cap;
    return 0; // qwen35 Vulkan pins what fits from full live VRAM (no KV reserve)
}

/// The generation `driver` for qwen35: like `SimpleDriver`, but first applies a
/// hybrid CPU/GPU layer split (`--cpu-layers` / `--offload-grow`) to the built
/// CUDA model and prints the split banner. The split is CUDA-only, so the setup
/// is `@hasDecl`-guarded — it compiles away for the cpu / vulkan steppers (which
/// the earlier feature gate has already restricted to no-split).
const Qwen35Driver = struct {
    cpu_split: ?TensorPencil.models.qwen35_cuda.CpuSplitPolicy,
    dynamic_offload: bool,
    vram_budget: u64,
    n_layers: usize,
    tok: *const TensorPencil.tokenizer.Tokenizer,
    io: Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: llm.engine.Options,
    stdout: *Io.Writer,
    prompt: ?[]const u8,
    img_chat: ?ImageChat,
    pub fn drive(self: Qwen35Driver, model: anytype) !llm.session.RunResult {
        const M = @typeInfo(@TypeOf(model)).pointer.child;
        if (comptime @hasDecl(M, "enableCpuSplit")) {
            if (self.cpu_split != null or self.dynamic_offload) {
                // Dynamic offload defaults to the attn policy (frees KV-growing
                // layers first, recovering the most VRAM per migration).
                const pol = self.cpu_split orelse .attn;
                try model.enableCpuSplit(pol, self.vram_budget, self.dynamic_offload);
                if (model.split) |sp| {
                    const mode = if (sp.dynamic) "offload-grow" else "cpu-layers";
                    try self.stdout.print("[--{s} {s}: {d}/{d} layers on CPU at start, {d} on GPU]\n", .{ mode, @tagName(sp.policy), sp.n_cpu, self.n_layers, self.n_layers - sp.n_cpu });
                } else {
                    try self.stdout.writeAll("[--cpu-layers: model fits under --vram-budget; no split needed]\n");
                }
                try self.stdout.flush();
            }
        }
        // t0 after the (setup) cpu-split migration, so it counts as setup not generation.
        const t0 = Io.Clock.real.now(self.io).nanoseconds;
        const n = try runSession(model, null, self.tok, self.io, self.gpa, self.ids, self.opts, self.stdout, self.prompt, self.img_chat);
        return .{ .n = n, .t0 = t0, .stats = llm.session.Stats.of(model) };
    }
};

/// qwen3's `Spec`: unlike the uniform archs, its CUDA/Vulkan steppers size
/// fixed batch/tap buffers from the prompt length, so the builders thread
/// `first_seq` through (and Vulkan takes `cap.max`, not the whole `Capacity`).
const Qwen3Spec = struct {
    pub const Model = qwen3.CausalLM;
    pub const Cpu = llm.engine.CpuModel;
    pub const Cuda = qwen3_cuda.CudaLM;
    pub const Vulkan = TensorPencil.models.qwen3_gpu.VulkanLM;
    pub fn kvWindow(lm: *const Model, cap: llm.engine.Capacity) u64 {
        _ = lm;
        return Vulkan.kvWindowBytes(cap.max);
    }
    pub fn buildCpu(gpa: std.mem.Allocator, lm: *const Model, cap: llm.engine.Capacity) !Cpu {
        return Cpu.init(gpa, lm, cap);
    }
    pub fn buildCuda(gpa: std.mem.Allocator, be: *TensorPencil.gpu.cuda.Backend, lm: *const Model, cap: llm.engine.Capacity, first_seq: usize) !Cuda {
        return Cuda.init(gpa, be, lm, cap, first_seq);
    }
    pub fn buildVulkan(gpa: std.mem.Allocator, ctx: *TensorPencil.gpu.Context, lm: *const Model, cap: llm.engine.Capacity, first_seq: usize) !Vulkan {
        // qwen3 Vulkan uses the hd128 attn_dsplit path (no f16 variant) and its
        // generation is a known-broken path; f16 KV is gemma3/qwen35-only on Vulkan.
        if (cap.kv_dtype != .f32) return error.KvDtypeUnsupported;
        return Vulkan.init(gpa, ctx, lm, cap.max, first_seq);
    }
};

/// The generation `driver` for qwen3: selects a speculative drafter and drives
/// generation. EAGLE (a trained head over the target's tapped hidden states)
/// and a CUDA draft model are CUDA-only; a CPU draft model runs on the CPU
/// stepper; everything else (n-gram `--spec-k`, or vanilla) uses a null drafter
/// (engine.generate handles n-gram internally). Branch selection is by exact
/// stepper type, so the CUDA/CPU-only construction compiles away for the other
/// backends (Vulkan falls through to the plain path).
const Qwen3Driver = struct {
    be_cuda: ?*TensorPencil.gpu.cuda.Backend,
    draft_lm: ?*qwen3.CausalLM,
    eagle_st: ?*TensorPencil.SafeTensors,
    target_lm: *const qwen3.CausalLM,
    cap: llm.engine.Capacity,
    prompt_len: usize,
    vram_budget: u64,
    tok: *const TensorPencil.tokenizer.Tokenizer,
    io: Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: llm.engine.Options,
    stdout: *Io.Writer,
    prompt: ?[]const u8,

    // Stamp t0 here (after any drafter/EAGLE construction in `drive`) so that
    // setup work is not attributed to generation time.
    fn gen(self: Qwen3Driver, model: anytype, drafter: anytype) !llm.session.RunResult {
        const t0 = Io.Clock.real.now(self.io).nanoseconds;
        const n = try runSession(model, drafter, self.tok, self.io, self.gpa, self.ids, self.opts, self.stdout, self.prompt, null);
        return .{ .n = n, .t0 = t0, .stats = llm.session.Stats.of(model) };
    }

    pub fn drive(self: Qwen3Driver, model: anytype) !llm.session.RunResult {
        const M = @typeInfo(@TypeOf(model)).pointer.child;
        if (comptime M == Qwen3Spec.Cuda) {
            if (self.eagle_st) |est| {
                try model.enableTaps(TensorPencil.models.eagle3.default_tap_layers);
                if (self.opts.tree_nodes > 0) try model.enableTree();
                var head = try TensorPencil.models.eagle3.Eagle3Head.load(self.gpa, self.be_cuda.?, est, self.target_lm, self.cap.max);
                defer head.deinit();
                var drafter = try TensorPencil.models.eagle3.Eagle3Drafter.init(self.gpa, &head, model);
                defer drafter.deinit();
                return self.gen(model, &drafter);
            }
            if (self.draft_lm) |dlm| {
                var draft = try M.init(self.gpa, self.be_cuda.?, dlm, self.cap, self.prompt_len);
                defer draft.deinit();
                // First claim on the pin budget goes to the draft: its weights are
                // read once per drafted token, the target's only once per ~k accepted
                // tokens, so pinned draft bytes save more PCIe.
                if (self.vram_budget != 0) try draft.prewarmWeights();
                var drafter = try llm.spec.ModelDrafter(M).init(self.gpa, self.io, &draft);
                defer drafter.deinit();
                return self.gen(model, &drafter);
            }
        } else if (comptime M == Qwen3Spec.Cpu) {
            if (self.draft_lm) |dlm| {
                var draft = try M.init(self.gpa, dlm, self.cap);
                defer draft.deinit();
                var drafter = try llm.spec.ModelDrafter(M).init(self.gpa, self.io, &draft);
                defer drafter.deinit();
                return self.gen(model, &drafter);
            }
        }
        return self.gen(model, null);
    }
};

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

/// One-shot / chat session for a qwen35 hybrid model (cpu stepper).
fn runQwen35(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: Io,
    g: *const TensorPencil.Gguf,
    backend: BackendKind,
    vram_budget: u64,
    cpu_split: ?TensorPencil.models.qwen35_cuda.CpuSplitPolicy,
    dynamic_offload: bool,
    image_path: ?[]const u8,
    mmproj_path: ?[]const u8,
    prompt: ?[]const u8,
    system: ?[]const u8,
    opts: llm.engine.Options,
    profile: bool,
    debug_batch: usize,
    stdout: *Io.Writer,
) !void {
    const qwen35 = TensorPencil.models.qwen35;
    var lm = try qwen35.Model.load(arena, g);
    defer lm.deinit();
    var tok = try TensorPencil.tokenizer.Tokenizer.initFromGguf(arena, g);
    defer tok.deinit();
    llm.chat.applyTokenizer(&tok);

    // Hybrid CPU/GPU layer split (--cpu-layers / --offload-grow): CUDA-only,
    // needs a budget as the VRAM ceiling, and text-only (the CPU path uses
    // scalar RoPE, which matches M-RoPE for text but not image positions).
    if (cpu_split != null or dynamic_offload) {
        if (backend != .@"zig-cuda" and backend != .cuda) {
            try stdout.writeAll("--cpu-layers / --offload-grow requires the zig-cuda or cuda backend\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (vram_budget == 0) {
            try stdout.writeAll("--cpu-layers needs --vram-budget <GiB> to size the CPU/GPU split\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (image_path != null or mmproj_path != null) {
            try stdout.writeAll("--cpu-layers is not supported with images yet (CPU layers use scalar RoPE; M-RoPE positions differ per axis)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
    }

    // Debug escape hatch: force the CPU ViT on GPU backends (A/B the CUDA
    // vision tower against the reference path).
    const debug_cpu_vit = false;

    // CUDA backends create the device context up front so the ViT (image
    // encode) can run on it before the LLM claims the VRAM. (bringUpCuda isn't
    // used here: under a CPU/GPU split the weight budget is NOT applied — the
    // split keeps its GPU layers fully resident and sizes itself after the
    // model is built, in Qwen35Driver.)
    const cuda_be = TensorPencil.gpu.cuda.Backend;
    const be_cuda: ?*cuda_be = switch (backend) {
        .cuda => try cuda_be.initLibs(arena),
        .@"zig-cuda" => try cuda_be.init(arena),
        else => null,
    };
    defer if (be_cuda) |be| be.deinit();
    if (be_cuda) |be| {
        be.profile = profile;
        if (cpu_split == null and !dynamic_offload) {
            _ = llm.session.applyCuda(be, vram_budget);
            // Page-lock the checkpoint mmap so a streamed tail DMAs at full PCIe
            // bandwidth, prefetched one layer ahead (skipped when fully resident).
            if (vram_budget != 0) if (g.mapping) |m| be.enableDirectStreaming(m);
        }
    }
    // Vulkan context up front (the qwen35 LLM runs on it; there is no Vulkan ViT).
    const vk_ctx: ?*TensorPencil.gpu.Context = if (backend == .vulkan) try TensorPencil.gpu.Context.init(arena) else null;
    defer if (vk_ctx) |c| c.deinit();

    // The vision tower stays loaded for the whole session when --mmproj is
    // given: --image encodes up front (one-shot), and interactive turns can
    // attach images with @path.png mentions at any point. The tower's host
    // side is cheap (mmap + a small arena); its device weights are scoped
    // per encode and never stay resident under the LLM.
    var mmg: ?TensorPencil.Gguf = null;
    defer if (mmg) |*mg| mg.deinit();
    var vit: ?TensorPencil.models.vit35.Vit = null;
    defer if (vit) |*v| v.deinit();
    if (mmproj_path) |mp| {
        mmg = try TensorPencil.Gguf.open(arena, io, mp);
        vit = try TensorPencil.models.vit35.Vit.load(arena, &mmg.?);
    }
    const img_chat: ?ImageChat = if (vit != null and be_cuda != null and !debug_cpu_vit)
        .{ .qwen35 = .{ .vit = &vit.?, .be = be_cuda.? } }
    else
        null;

    // Encode a --image up front; the embeddings are injected during
    // prefill. Wrapping mirrors llama.cpp mtmd: the image leads the user
    // turn as <|vision_start|>[embeddings]<|vision_end|>.
    var img: ?TensorPencil.models.vit35.Vit.Encoded = null;
    defer if (img) |*e| e.deinit(gpa);
    if (image_path) |ip| {
        const dec = try vips.loadRgb(gpa, ip);
        defer gpa.free(dec.pixels);
        const t_vit = Io.Clock.real.now(io).nanoseconds;
        img = if (img_chat) |ic|
            try TensorPencil.models.vit35_cuda.encode(ic.qwen35.vit, ic.qwen35.be, gpa, dec.pixels, dec.width, dec.height)
        else
            try vit.?.encode(io, gpa, dec.pixels, dec.width, dec.height);
        const vit_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t_vit)) / 1e9;
        try stdout.print("[image {d}x{d} -> {d}x{d} tokens; vit {d:.1}s ({s})]\n", .{
            dec.width, dec.height, img.?.grid_w, img.?.grid_h, vit_s, if (img_chat != null) "gpu" else "cpu",
        });
        try stdout.flush();
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    // For an image one-shot, the tokens before/after the embedding block are
    // tracked so prefill can interleave them with prefillImage.
    var n_pre: usize = 0;
    if (system) |s| try llm.chat.appendSystem(&tok, gpa, s, &ids);
    if (img) |*e| {
        try tok.encode(gpa, "<|im_start|>user\n<|vision_start|>", &ids);
        n_pre = ids.items.len;
        // Placeholders keep ids aligned with cache rows (sampling penalties
        // and the engine's cached()-based prefill both index by row).
        const pad_id = tok.specialId("<|image_pad|>") orelse tok.pad;
        try ids.appendNTimes(gpa, pad_id, e.grid_w * e.grid_h);
        try tok.encode(gpa, "<|vision_end|>", &ids);
        try tok.encode(gpa, prompt.?, &ids);
        try tok.encode(gpa, "<|im_end|>\n<|im_start|>assistant\n", &ids);
    } else if (prompt) |p| {
        try llm.chat.appendUser(&tok, gpa, p, &ids);
        try llm.chat.openAssistant(&tok, gpa, &ids);
    }

    try stdout.print("[{s} backend, qwen35 {d}L hybrid, {d} prompt tokens, ctx window {d}, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), lm.cfg.n_layers, ids.items.len, opts.max_context, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) {
        try stdout.writeAll("[interactive chat: Enter sends, Shift- or Alt-Enter adds a line, /exit or Ctrl-D quits]\n");
        if (img_chat != null) try stdout.writeAll("[attach images with @path.jpg / @\"path with spaces.png\" (png/jpeg/webp/gif/tiff), anywhere in a message]\n");
    }
    try stdout.writeAll("\n");
    try stdout.flush();

    // Commit a small initial KV slice and grow toward the window on demand
    // (no speculative decoding for qwen35, so sessions are always dynamic).
    const cap = try capacityPlan(opts, prompt, ids.items.len, false);

    // --debug-batch <n>: diff sequential vs batched prefill of the first
    // <n> prompt tokens, layer by layer, then exit.
    if (debug_batch > 0) {
        const dbg_n = debug_batch;
        const be = be_cuda orelse try cuda_be.init(arena);
        defer if (be_cuda == null) be.deinit();
        var model = try TensorPencil.models.qwen35_cuda.CudaLM.init(gpa, be, &lm, cap);
        defer model.deinit();
        const nl = lm.cfg.n_layers;
        const dump_seq = try gpa.alloc(f32, nl * lm.cfg.hidden);
        defer gpa.free(dump_seq);
        const dump_bat = try gpa.alloc(f32, nl * lm.cfg.hidden);
        defer gpa.free(dump_bat);
        const op_seq = try gpa.alloc(f32, dbg_n * lm.cfg.hidden);
        defer gpa.free(op_seq);
        const op_bat = try gpa.alloc(f32, dbg_n * lm.cfg.hidden);
        defer gpa.free(op_bat);
        @memset(op_seq, 0);
        @memset(op_bat, 0);
        const toks = ids.items[0..dbg_n];

        // Sequential reference: capture layer-0 normed per token.
        model.op_dump = op_seq;
        model.op_dump_row = 0;
        for (toks[0 .. dbg_n - 1]) |id| try model.debugStepOne(id);
        model.layer_dump = dump_seq;
        try model.debugStepOne(toks[dbg_n - 1]);
        model.layer_dump = null;
        model.op_dump = null;

        try model.resetCache();
        model.layer_dump = dump_bat;
        model.op_dump = op_bat;
        try model.prefill(toks);
        model.layer_dump = null;
        model.op_dump = null;

        for (0..nl) |l| {
            var worst: f32 = 0;
            var rel: f32 = 0;
            for (dump_seq[l * lm.cfg.hidden ..][0..lm.cfg.hidden], dump_bat[l * lm.cfg.hidden ..][0..lm.cfg.hidden]) |a, bb| {
                worst = @max(worst, @abs(a - bb));
                rel = @max(rel, @abs(a - bb) / (@abs(a) + 1e-3));
            }
            try stdout.print("layer {d:>2} ({s}): max abs diff {d:.6} rel {d:.4}\n", .{ l, if (lm.cfg.isRecurrent(l)) "lin " else "attn", worst, rel });
        }
        for (0..dbg_n) |r| {
            var worst: f32 = 0;
            for (op_seq[r * lm.cfg.hidden ..][0..lm.cfg.hidden], op_bat[r * lm.cfg.hidden ..][0..lm.cfg.hidden]) |a, bb| {
                worst = @max(worst, @abs(a - bb));
            }
            if (worst > 1e-4) try stdout.print("layer0 normed row {d:>3}: max diff {d:.6}\n", .{ r, worst });
        }
        try stdout.flush();
        return;
    }

    // --profile: wall-clock forward-pass breakdown on cpu, device op timings on
    // the CUDA backends. Enable the cpu profiler before generation; reported
    // after `run` (before the summary, matching the old per-arm ordering).
    if (profile and backend == .cpu) {
        TensorPencil.prof.reset();
        TensorPencil.prof.enabled = true;
    }

    const dev: llm.session.Devices = .{ .cu_be = be_cuda, .vk_ctx = vk_ctx };
    const Spec = llm.session.UniformSpec(
        qwen35.Model,
        qwen35.CpuModel,
        TensorPencil.models.qwen35_cuda.CudaLM,
        TensorPencil.models.qwen35_gpu.VulkanLM,
        qwen35KvWindow,
    );
    const n_img: usize = if (img) |*e| e.grid_w * e.grid_h else 0;
    const prefiller: llm.session.ImagePrefiller(TensorPencil.models.vit35.Vit.Encoded) = .{
        .img = if (img) |*e| e else null,
        .n_pre = n_pre,
        .n_img = n_img,
        .ids = ids.items,
    };
    const driver: Qwen35Driver = .{
        .cpu_split = cpu_split,
        .dynamic_offload = dynamic_offload,
        .vram_budget = vram_budget,
        .n_layers = lm.cfg.n_layers,
        .tok = &tok,
        .io = io,
        .gpa = gpa,
        .ids = &ids,
        .opts = opts,
        .stdout = stdout,
        .prompt = prompt,
        .img_chat = img_chat,
    };

    const t_init = Io.Clock.real.now(io).nanoseconds;
    const res = try llm.session.run(Spec, dev, backend, &lm, ids.items.len, prefiller, driver, io, gpa, cap, vram_budget, stdout);
    if (profile) {
        if (backend == .cpu) {
            TensorPencil.prof.report(stdout) catch {};
        } else if (be_cuda) |be| {
            llm.session.printCudaProfile(stdout, be) catch {};
            stdout.flush() catch {};
        }
    }
    const setup_s = @as(f64, @floatFromInt(res.t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - res.t0)) / 1e9;
    try llm.session.printSummary(stdout, prompt != null, res.n, setup_s, elapsed_s, res.stats);
    try stdout.flush();
}

/// One-shot / chat session for a Gemma 3 model (cpu / zig-cuda / cuda,
/// text-only).
fn runGemma3(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: Io,
    g: *const TensorPencil.Gguf,
    backend: BackendKind,
    vram_budget: u64,
    image_path: ?[]const u8,
    mmproj_path: ?[]const u8,
    prompt: ?[]const u8,
    system: ?[]const u8,
    opts: llm.engine.Options,
    profile: bool,
    stdout: *Io.Writer,
) !void {
    if (image_path != null) {
        if (mmproj_path == null) {
            try stdout.writeAll("--image requires --mmproj <mmproj.gguf> (the SigLIP vision tower ships separately)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (prompt == null) {
            try stdout.writeAll("--image requires --prompt (one-shot captioning)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
    }
    const gemma3 = TensorPencil.models.gemma3;
    var lm = try gemma3.Model.load(arena, g);
    defer lm.deinit();
    var tok = try TensorPencil.tokenizer.Tokenizer.initFromGguf(arena, g);
    defer tok.deinit();
    llm.chat.applyTokenizer(&tok);
    llm.chat.setFamily(.gemma);

    // CUDA backend created up front (so the ViT can encode before the LLM
    // claims VRAM); null on cpu. Bring-up pins weights under the budget; then
    // page-lock the checkpoint mmap so a streamed tail DMAs at full bandwidth.
    const be_cuda = try llm.session.bringUpCuda(arena, backend, profile, vram_budget);
    defer if (be_cuda) |b| b.deinit();
    if (be_cuda) |b| if (vram_budget != 0) if (g.mapping) |m| b.enableDirectStreaming(m);
    // Vulkan context created up front too (shared by the ViT and the LLM).
    const vk_ctx: ?*TensorPencil.gpu.Context = if (backend == .vulkan) try TensorPencil.gpu.Context.init(arena) else null;
    defer if (vk_ctx) |c| c.deinit();

    // Vision tower stays loaded for the whole session (--mmproj): --image
    // encodes up front (one-shot). The projected embeddings are injected
    // UNSCALED during prefill. On CUDA the tower runs device-side (GPU ViT)
    // and interactive @path.png mentions are supported; on Vulkan --image
    // runs the SigLIP blocks device-side too (gemma_vit_gpu).
    var mmg: ?TensorPencil.Gguf = null;
    defer if (mmg) |*mg| mg.deinit();
    var vit: ?TensorPencil.models.gemma_vit.Vit = null;
    defer if (vit) |*v| v.deinit();
    var vit_gpu: ?TensorPencil.models.gemma_vit_gpu.VitGpu = null;
    defer if (vit_gpu) |*v| v.deinit();
    if (mmproj_path) |mp| {
        mmg = try TensorPencil.Gguf.open(arena, io, mp);
        vit = try TensorPencil.models.gemma_vit.Vit.load(arena, &mmg.?);
        if (vk_ctx != null) vit_gpu = try TensorPencil.models.gemma_vit_gpu.VitGpu.load(arena, &vit.?);
    }
    const img_chat: ?ImageChat = if (vit != null and be_cuda != null)
        .{ .gemma3 = .{ .vit = &vit.?, .be = be_cuda.?, .io = io } }
    else
        null;

    var img: ?TensorPencil.models.gemma_vit.Vit.Encoded = null;
    defer if (img) |*e| e.deinit(gpa);
    if (image_path) |ip| {
        const dec = try vips.loadRgb(gpa, ip);
        defer gpa.free(dec.pixels);
        const t_vit = Io.Clock.real.now(io).nanoseconds;
        img = if (be_cuda) |b|
            try TensorPencil.models.gemma_vit_cuda.encode(&vit.?, b, io, gpa, dec.pixels, dec.width, dec.height)
        else if (vk_ctx) |c|
            try vit_gpu.?.encode(c, io, gpa, dec.pixels, dec.width, dec.height)
        else
            try vit.?.encode(io, gpa, dec.pixels, dec.width, dec.height);
        const dev = if (be_cuda != null or vk_ctx != null) "gpu" else "cpu";
        const vit_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t_vit)) / 1e9;
        try stdout.print("[image {d}x{d} -> {d} tokens; vit {d:.1}s ({s})]\n", .{ dec.width, dec.height, img.?.grid_w * img.?.grid_h, vit_s, dev });
        try stdout.flush();
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    // Gemma prompts begin with a single BOS ({{ bos_token }} in the template).
    if (tok.specialId("<bos>")) |bos| try ids.append(gpa, bos);
    // Rows before the image block, and the block's soft-token count — the
    // one-shot prefill interleaves text / image / text around them.
    var n_pre: usize = 0;
    var n_img: usize = 0;
    if (img) |*e| {
        n_img = e.grid_w * e.grid_h;
        try tok.encode(gpa, "<start_of_turn>user\n", &ids);
        try tok.encode(gpa, "<start_of_image>", &ids);
        n_pre = ids.items.len;
        try ids.appendNTimes(gpa, tok.specialId("<image_soft_token>") orelse return error.MissingImageToken, n_img);
        try tok.encode(gpa, "<end_of_image>", &ids);
        try tok.encode(gpa, prompt.?, &ids);
        try tok.encode(gpa, "<end_of_turn>\n", &ids);
        try llm.chat.openAssistant(&tok, gpa, &ids);
    } else {
        if (system) |s| try llm.chat.appendSystem(&tok, gpa, s, &ids);
        if (prompt) |p| {
            try llm.chat.appendUser(&tok, gpa, p, &ids);
            try llm.chat.openAssistant(&tok, gpa, &ids);
        }
    }

    try stdout.print("[{s} backend, gemma3 {d}L, {d} prompt tokens, ctx window {d}, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), lm.cfg.n_layers, ids.items.len, opts.max_context, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) {
        try stdout.writeAll("[interactive chat: Enter sends, Shift- or Alt-Enter adds a line, /exit or Ctrl-D quits]\n");
        if (img_chat != null) try stdout.writeAll("[attach images with @path.jpg / @\"path with spaces.png\" (png/jpeg/webp/gif/tiff), anywhere in a message]\n");
    }
    try stdout.writeAll("\n");
    try stdout.flush();

    const cap = try capacityPlan(opts, prompt, ids.items.len, false);
    if (profile) {
        TensorPencil.prof.reset();
        TensorPencil.prof.enabled = true;
    }
    defer if (profile) TensorPencil.prof.report(stdout) catch {};

    const dev: llm.session.Devices = .{ .cu_be = be_cuda, .vk_ctx = vk_ctx };
    const Spec = llm.session.UniformSpec(
        gemma3.Model,
        gemma3.CpuModel,
        TensorPencil.models.gemma3_cuda.CudaLM,
        TensorPencil.models.gemma3_gpu.VulkanLM,
        gemma3KvWindow,
    );
    const prefiller: llm.session.ImagePrefiller(TensorPencil.models.gemma_vit.Vit.Encoded) = .{
        .img = if (img) |*e| e else null,
        .n_pre = n_pre,
        .n_img = n_img,
        .ids = ids.items,
    };
    // On Vulkan there is no CUDA backend, so img_chat is already null (interactive
    // @image is CUDA-only) — the driver passes it uniformly.
    const driver: SimpleDriver = .{ .tok = &tok, .io = io, .gpa = gpa, .ids = &ids, .opts = opts, .stdout = stdout, .prompt = prompt, .img_chat = img_chat };

    const t_init = Io.Clock.real.now(io).nanoseconds;
    const res = try llm.session.run(Spec, dev, backend, &lm, ids.items.len, prefiller, driver, io, gpa, cap, vram_budget, stdout);
    const setup_s = @as(f64, @floatFromInt(res.t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - res.t0)) / 1e9;
    try llm.session.printSummary(stdout, prompt != null, res.n, setup_s, elapsed_s, res.stats);
    try stdout.flush();
}

fn runGemma4(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: Io,
    g: *const TensorPencil.Gguf,
    backend: BackendKind,
    image_path: ?[]const u8,
    mmproj_path: ?[]const u8,
    prompt: ?[]const u8,
    system: ?[]const u8,
    opts: llm.engine.Options,
    profile: bool,
    stdout: *Io.Writer,
) !void {
    if (backend == .vulkan) {
        try stdout.writeAll("gemma4 runs on cpu / zig-cuda / cuda (the vulkan backend is a follow-up)\n");
        try stdout.flush();
        return error.InvalidArgument;
    }
    if (image_path != null) {
        if (mmproj_path == null) {
            try stdout.writeAll("--image requires --mmproj <mmproj.gguf> (the vision embedder ships separately)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (prompt == null) {
            try stdout.writeAll("--image requires --prompt (one-shot; gemma4 interactive @image is not wired yet)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
    }
    const gemma4 = TensorPencil.models.gemma4;
    var lm = try gemma4.Model.load(arena, g);
    defer lm.deinit();
    var tok = try TensorPencil.tokenizer.Tokenizer.initFromGguf(arena, g);
    defer tok.deinit();
    llm.chat.applyTokenizer(&tok);
    llm.chat.setFamily(.gemma4);

    // CUDA backend created up front so the vision embedder can encode
    // device-side (the LLM claims VRAM after). Null on cpu. A 0 budget pins the
    // whole model resident (the 12B Q4_0 fits the 3090).
    const be_cuda = try llm.session.bringUpCuda(arena, backend, profile, 0);
    defer if (be_cuda) |bb| bb.deinit();

    // Vision embedder (--mmproj): the gemma4uv "unified" embedder (no ViT
    // blocks) runs device-side on the CUDA backends (host on cpu); --image
    // encodes up front (one-shot) and the embeddings inject UNSCALED at prefill.
    var mmg: ?TensorPencil.Gguf = null;
    defer if (mmg) |*mg| mg.deinit();
    var vit: ?TensorPencil.models.gemma4_vit.Vit = null;
    defer if (vit) |*v| v.deinit();
    if (mmproj_path) |mp| {
        mmg = try TensorPencil.Gguf.open(arena, io, mp);
        vit = try TensorPencil.models.gemma4_vit.Vit.load(arena, &mmg.?);
    }
    var img: ?TensorPencil.models.gemma4_vit.Vit.Encoded = null;
    defer if (img) |*e| e.deinit(gpa);
    if (image_path) |ip| {
        const dec = try vips.loadRgb(gpa, ip);
        defer gpa.free(dec.pixels);
        const t_vit = Io.Clock.real.now(io).nanoseconds;
        img = if (be_cuda) |bb|
            try TensorPencil.models.gemma4_vit_cuda.encode(&vit.?, bb, io, gpa, dec.pixels, dec.width, dec.height)
        else
            try vit.?.encode(io, gpa, dec.pixels, dec.width, dec.height);
        const vit_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t_vit)) / 1e9;
        try stdout.print("[image {d}x{d} -> {d} tokens; vit {d:.1}s ({s})]\n", .{ dec.width, dec.height, img.?.grid_w * img.?.grid_h, vit_s, if (be_cuda == null) "cpu" else "gpu" });
        try stdout.flush();
    }

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    // Gemma 4 prompts begin with a single BOS.
    if (tok.specialId("<bos>")) |bos| try ids.append(gpa, bos);
    // Rows before the image block, and the block's token count — the one-shot
    // prefill interleaves text / image / text around them.
    var n_pre: usize = 0;
    var n_img: usize = 0;
    if (img) |*e| {
        n_img = e.grid_w * e.grid_h;
        try tok.encode(gpa, "<|turn>user\n", &ids);
        try ids.append(gpa, tok.specialId("<|image>") orelse return error.MissingImageToken);
        n_pre = ids.items.len;
        // Placeholder rows (overwritten by prefillImage; keep ids aligned with
        // cache rows). The id is immaterial to the forward.
        try ids.appendNTimes(gpa, tok.pad, n_img);
        try ids.append(gpa, tok.specialId("<image|>") orelse return error.MissingImageToken);
        try tok.encode(gpa, prompt.?, &ids);
        try tok.encode(gpa, "<turn|>\n", &ids);
        try llm.chat.openAssistant(&tok, gpa, &ids);
    } else {
        if (system) |s| try llm.chat.appendSystem(&tok, gpa, s, &ids);
        if (prompt) |p| {
            try llm.chat.appendUser(&tok, gpa, p, &ids);
            try llm.chat.openAssistant(&tok, gpa, &ids);
        }
    }

    try stdout.print("[{s} backend, gemma4 {d}L, {d} prompt tokens, ctx window {d}, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), lm.cfg.n_layers, ids.items.len, opts.max_context, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) {
        try stdout.writeAll("[interactive chat: Enter sends, Shift- or Alt-Enter adds a line, /exit or Ctrl-D quits]\n");
    }
    try stdout.writeAll("\n");
    try stdout.flush();

    const cap = try capacityPlan(opts, prompt, ids.items.len, false);
    if (profile) {
        TensorPencil.prof.reset();
        TensorPencil.prof.enabled = true;
    }
    defer if (profile) TensorPencil.prof.report(stdout) catch {};

    // CUDA backend + pinning were set up front (before the vision encode). No
    // CPU-split / offload / streaming — the 12B fits. gemma4 has no Vulkan
    // backend (Spec.Vulkan = void); main() rejects --backend vulkan before here.
    const dev: llm.session.Devices = .{ .cu_be = be_cuda };
    const Spec = llm.session.UniformSpec(
        gemma4.Model,
        gemma4.CpuModel,
        TensorPencil.models.gemma4_cuda.CudaLM,
        void,
        gemma4KvWindow,
    );
    const prefiller: llm.session.ImagePrefiller(TensorPencil.models.gemma4_vit.Vit.Encoded) = .{
        .img = if (img) |*e| e else null,
        .n_pre = n_pre,
        .n_img = n_img,
        .ids = ids.items,
    };
    const driver: SimpleDriver = .{ .tok = &tok, .io = io, .gpa = gpa, .ids = &ids, .opts = opts, .stdout = stdout, .prompt = prompt, .img_chat = null };

    const t_init = Io.Clock.real.now(io).nanoseconds;
    // gemma4 ignores --vram-budget (it always pins the whole model), so pass 0.
    const res = try llm.session.run(Spec, dev, backend, &lm, ids.items.len, prefiller, driver, io, gpa, cap, 0, stdout);
    const setup_s = @as(f64, @floatFromInt(res.t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - res.t0)) / 1e9;
    try llm.session.printSummary(stdout, prompt != null, res.n, setup_s, elapsed_s, res.stats);
    try stdout.flush();
}

/// Ceiling for the auto (no --max-context) window: bounds the up-front RoPE
/// table (rows x head_dim/2 x 2 f32 — 64 MB at 128k for head_dim 128) and
/// the per-layer VA reservations. Physical VRAM, not this, is what actually
/// limits a session; pass --max-context to raise it.
const auto_context_cap: usize = 128 << 10;

/// The model's trained context length (`<arch>.context_length`), when the
/// container records one. Safetensors checkpoints carry no metadata — fall
/// back to the native Qwen3 window.
fn trainedContext(st: *const ModelFile) usize {
    switch (st.*) {
        .gguf => |*g| return @intCast(g.contextLength() orelse 32768),
        .safetensors => return 32768,
    }
}

/// KV-cache sizing for a session: the ceiling is the old fixed capacity
/// (one-shot request size, or the whole --max-context window for chat);
/// dynamic sessions commit only min(4096, prompt + 1) rows up front and grow
/// toward it. `fixed` forces initial == max (speculative decoding).
fn capacityPlan(opts: llm.engine.Options, prompt: ?[]const u8, n_prompt: usize, fixed: bool) !llm.engine.Capacity {
    const max = if (prompt == null) opts.max_context else try llm.engine.capacityFor(opts, n_prompt);
    if (fixed) return .{ .initial = max, .max = max, .kv_dtype = opts.kv_dtype };
    return .{
        .initial = @min(max, @max(n_prompt + 1, llm.kv_cache.initial_context)),
        .max = max,
        .kv_dtype = opts.kv_dtype,
    };
}

const BackendKind = llm.session.BackendKind;

/// A checkpoint opened from disk, container chosen by file extension
/// (".gguf" or safetensors otherwise).
const ModelFile = union(enum) {
    safetensors: TensorPencil.SafeTensors,
    gguf: TensorPencil.Gguf,

    fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !ModelFile {
        if (std.ascii.endsWithIgnoreCase(path, ".gguf"))
            return .{ .gguf = try TensorPencil.Gguf.open(gpa, io, path) };
        return .{ .safetensors = try TensorPencil.SafeTensors.open(gpa, io, path) };
    }

    fn store(self: *const ModelFile) TensorPencil.WeightStore {
        return switch (self.*) {
            .safetensors => |*s| .{ .safetensors = s },
            .gguf => |*g| .{ .gguf = g },
        };
    }

    fn mapping(self: *const ModelFile) ?[]align(std.heap.page_size_min) const u8 {
        return self.store().mapping();
    }

    fn deinit(self: *ModelFile) void {
        switch (self.*) {
            .safetensors => |*s| s.deinit(),
            .gguf => |*g| g.deinit(),
        }
    }
};

fn nextArg(args: []const [:0]const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgument;
    return args[i.*];
}
