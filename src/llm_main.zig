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
    \\              [--temperature <t>] [--top-k <n>] [--top-p <p>]
    \\              [--repeat-penalty <r>] [--seed <n>] [--greedy]
    \\              [--spec-k <n>] [--draft-model <qwen3.safetensors>]
    \\              [--eagle <eagle3.safetensors>] [--tree <nodes>]
    \\              [--vram-budget <GiB>|min]
    \\              [--image <image> --mmproj <mmproj.gguf>]
    \\
    \\--max-context caps the context window; the default is the model's
    \\trained context length (up to 128k). Only a small initial slice of KV
    \\rows (4096, or the prompt size) is committed up front; the cache grows
    \\in place on demand, so a large window costs VRAM only as the
    \\conversation actually fills it. Under VRAM pressure growth evicts
    \\least-recently-used weights into the streaming path (slower decode)
    \\instead of failing, and the session ends only when nothing more can be
    \\freed. Speculative runs (--spec-k/--draft-model/--eagle/--tree) and
    \\--backend vulkan reserve the whole window physically up front, so
    \\their default window stays 4096.
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

    // Container chosen by extension: .gguf or safetensors.
    var st = try ModelFile.open(arena, io, path);
    defer st.deinit();

    // Default context ceiling: the model's trained context length (growth
    // commits KV rows only as the conversation fills, so a big ceiling costs
    // VRAM lazily — and stops gracefully when even weight eviction can't
    // free enough). Fixed-capacity sessions (speculative decoding, vulkan)
    // allocate the whole window physically up front, so without an explicit
    // --max-context they keep the old 4096 default.
    const fixed_session = opts.spec_k > 0 or opts.tree_nodes > 0 or draft_path != null or eagle_path != null;
    opts.max_context = max_context_arg orelse
        (if (fixed_session or backend == .vulkan) 4096 else @min(trainedContext(&st), auto_context_cap));

    // Architecture dispatch: qwen35 (hybrid DeltaNet) has its own model and
    // steppers (cpu / zig-cuda / cuda), and no speculative decoding (the
    // recurrent state cannot roll back past rejected drafts).
    if (st == .gguf and st.gguf.getStr("general.architecture") != null and
        std.mem.eql(u8, st.gguf.getStr("general.architecture").?, "qwen35"))
    {
        if (backend == .vulkan) {
            try stdout.writeAll("qwen35 (hybrid DeltaNet) models run on cpu / zig-cuda / cuda only for now\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (draft_path != null or eagle_path != null or opts.spec_k > 0 or opts.tree_nodes > 0) {
            try stdout.writeAll("speculative decoding is not supported for qwen35 models (recurrent state cannot roll back)\n");
            try stdout.flush();
            return error.InvalidArgument;
        }
        if (vram_budget != 0) try stdout.writeAll("[--vram-budget ignored for qwen35 (weights stay resident)]\n");
        if (image_path != null) {
            if (backend == .cpu) {
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
        return runQwen35(arena, gpa, io, &st.gguf, backend, image_path, mmproj_path, prompt, system, opts, profile, debug_batch, stdout);
    }

    if (image_path != null) {
        try stdout.writeAll("--image is only supported for qwen35 (Qwen3.5/3.6) GGUF models\n");
        try stdout.flush();
        return error.InvalidArgument;
    }

    var lm = try qwen3.CausalLM.load(arena, st.store());
    defer lm.deinit();
    if (backend == .vulkan and lm.hasBlockQuantWeights()) {
        try stdout.writeAll("GGUF block-quantized weights (Q8_0/Q4_K/Q5_K/Q6_K) run on cpu / zig-cuda / cuda only for now\n");
        try stdout.flush();
        return error.InvalidArgument;
    }
    // Tokenizer: a GGUF checkpoint carries its own vocab (which may differ
    // from the embedded Qwen3 one — e.g. Qwen3.6's 248k tokens); fall back
    // to the embedded tokenizer when the file has none (ComfyUI-style
    // conversions strip it).
    var tok = switch (st) {
        .gguf => |*g| TensorPencil.tokenizer.Tokenizer.initFromGguf(arena, g) catch |err| switch (err) {
            error.MissingTokenizer => try TensorPencil.tokenizer.Tokenizer.init(arena),
            else => return err,
        },
        .safetensors => try TensorPencil.tokenizer.Tokenizer.init(arena),
    };
    defer tok.deinit();
    llm.chat.applyTokenizer(&tok);

    // Draft model for speculative decoding (same tokenizer family; its own
    // KV cache and stepper). Implies spec decoding. The mapped checkpoint
    // must outlive the session — weights are views into it.
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

    try stdout.print("[{s} backend, {d} prompt tokens, ctx window {d}, max {d} new/turn, temp {d:.2}, seed {d}]\n", .{
        @tagName(backend), ids.items.len, opts.max_context, opts.max_new_tokens, opts.sampling.temperature, opts.seed,
    });
    if (prompt == null) try stdout.writeAll("[interactive chat: Enter sends, Shift- or Alt-Enter adds a line, /exit or Ctrl-D quits]\n");
    try stdout.writeAll("\n");
    try stdout.flush();

    // One-shot caps the window at the request; a chat session gets the whole
    // --max-context window. Either way only a small initial slice of KV rows
    // is committed up front — the cache grows in place as generation fills it
    // (evicting weights into the streaming path under VRAM pressure).
    // Speculative runs stay fixed-capacity: the tree batch region and EAGLE
    // tap buffers stride by capacity, so it cannot move mid-session.
    const cap = try capacityPlan(opts, prompt, ids.items.len, opts.spec_k > 0 or opts.tree_nodes > 0 or draft_path != null or eagle_path != null);
    const prompt_len = if (prompt == null) @min(512, cap.max) else ids.items.len;
    const t_init = Io.Clock.real.now(io).nanoseconds;
    var t0: i96 = undefined; // set after backend/model setup: generation only
    const n = switch (backend) {
        .cpu => blk: {
            if (vram_budget != 0) try stdout.writeAll("[--vram-budget ignored on the cpu backend]\n");
            var model = try llm.engine.CpuModel.init(gpa, &lm, cap);
            defer model.deinit();
            if (draft_lm) |*dlm| {
                var draft = try llm.engine.CpuModel.init(gpa, dlm, cap);
                defer draft.deinit();
                var drafter = try llm.spec.ModelDrafter(llm.engine.CpuModel).init(gpa, io, &draft);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                break :blk try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt, null);
            }
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt, null);
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
                if (st.mapping()) |m| be.enableDirectStreaming(m);
                if (draft_st) |ds| if (ds.mapping()) |m| be.enableDirectStreaming(m);
                if (eagle_st) |es| if (es.mapping) |m| be.enableDirectStreaming(m);
            }
            var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, cap, prompt_len);
            defer model.deinit();
            var count: usize = undefined;
            if (eagle_st) |*est| {
                try model.enableTaps(TensorPencil.models.eagle3.default_tap_layers);
                if (opts.tree_nodes > 0) try model.enableTree();
                var head = try TensorPencil.models.eagle3.Eagle3Head.load(gpa, be, est, &lm, cap.max);
                defer head.deinit();
                var drafter = try TensorPencil.models.eagle3.Eagle3Drafter.init(gpa, &head, &model);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt, null);
            } else if (draft_lm) |*dlm| {
                var draft = try qwen3_cuda.CudaLM.init(gpa, be, dlm, cap, prompt_len);
                defer draft.deinit();
                // First claim on the pin budget goes to the draft: its weights
                // are read once per drafted token, the target's only once per
                // ~k accepted tokens, so pinned draft bytes save more PCIe.
                if (vram_budget != 0) try draft.prewarmWeights();
                var drafter = try llm.spec.ModelDrafter(qwen3_cuda.CudaLM).init(gpa, io, &draft);
                defer drafter.deinit();
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, &drafter, &tok, io, gpa, &ids, opts, stdout, prompt, null);
            } else {
                t0 = Io.Clock.real.now(io).nanoseconds;
                count = try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt, null);
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
            // Vulkan has no growable buffers yet: reserve the whole window.
            var model = try TensorPencil.models.qwen3_gpu.VulkanLM.init(gpa, ctx, &lm, cap.max, prompt_len);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt, null);
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

/// Session-lifetime vision context for @image mentions in interactive
/// turns (qwen35 on the CUDA backends).
const ImageChat = struct {
    vit: *const TensorPencil.models.vit35.Vit,
    be: *TensorPencil.gpu.cuda.Backend,
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

        var encs: std.ArrayList(TensorPencil.models.vit35.Vit.Encoded) = .empty;
        defer {
            for (encs.items) |*e| e.deinit(gpa);
            encs.deinit(gpa);
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
                try encs.ensureUnusedCapacity(gpa, 1);
                const enc = try TensorPencil.models.vit35_cuda.encode(ic.vit, ic.be, gpa, dec.pixels, dec.width, dec.height);
                encs.appendAssumeCapacity(enc);
                try segs.append(gpa, .{ .image = .{ .grid_w = enc.grid_w, .grid_h = enc.grid_h } });
                try stdout.print("[{s}: {d}x{d} -> {d} rows]\n", .{ path, dec.width, dec.height, enc.grid_w * enc.grid_h });
                try stdout.flush();
            },
        };

        const ids_before = ids.items.len;
        var image_rows: std.ArrayList(usize) = .empty;
        defer image_rows.deinit(gpa);
        try llm.chat.appendUserSegments(tok, gpa, segs.items, ids, &image_rows);
        try llm.chat.openAssistant(tok, gpa, ids);
        if (ids.items.len - model.cached() > model.remaining()) {
            // Growable caches commit more rows first (image turns are the
            // usual way a turn outgrows the committed slice).
            if (comptime @hasDecl(M, "ensureCapacity")) model.ensureCapacity(ids.items.len) catch {};
            if (ids.items.len - model.cached() > model.remaining()) {
                try stdout.print("[turn needs {d} rows, only {d} left in context]\n", .{ ids.items.len - model.cached(), model.remaining() });
                ids.shrinkRetainingCapacity(ids_before);
                return false;
            }
        }
        // Interleave text prefill with the image embeddings, exactly like
        // the one-shot --image path; the engine's generate() prefills the
        // remaining tail from cached().
        for (image_rows.items, encs.items) |row, e| {
            const pending = ids.items[model.cached()..row];
            if (pending.len > 0) try model.prefill(pending);
            try model.prefillImage(e.embeds, e.grid_w, e.grid_h);
        }
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
        // ceiling itself is reached.
        const M = switch (@typeInfo(@TypeOf(model))) {
            .pointer => |p| p.child,
            else => @TypeOf(model),
        };
        const window = if (comptime @hasDecl(M, "capacityMax")) model.capacityMax() else model.cached() + model.remaining();
        const dt = @as(f64, @floatFromInt(Io.Clock.real.now(io).nanoseconds - t0)) / 1e9;
        try stdout.print("\n[{d} tok, {d:.1} tok/s, ctx {d}/{d}]\n", .{
            n, @as(f64, @floatFromInt(n)) / dt, model.cached(), window,
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

    // Debug escape hatch: force the CPU ViT on GPU backends (A/B the CUDA
    // vision tower against the reference path).
    const debug_cpu_vit = false;

    // CUDA backends create the device context up front so the ViT (image
    // encode) can run on it before the LLM claims the VRAM.
    const cuda_be = TensorPencil.gpu.cuda.Backend;
    const be_cuda: ?*cuda_be = switch (backend) {
        .cuda => try cuda_be.initLibs(arena),
        .@"zig-cuda" => try cuda_be.init(arena),
        else => null,
    };
    defer if (be_cuda) |be| be.deinit();

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
        .{ .vit = &vit.?, .be = be_cuda.? }
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
            try TensorPencil.models.vit35_cuda.encode(ic.vit, ic.be, gpa, dec.pixels, dec.width, dec.height)
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

        try model.debugReset();
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

    const t_init = Io.Clock.real.now(io).nanoseconds;
    var t0: i96 = undefined;
    const n = switch (backend) {
        .cpu => blk: {
            var model = try qwen35.CpuModel.init(gpa, &lm, cap);
            defer model.deinit();
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt, null);
        },
        .@"zig-cuda", .cuda => blk: {
            const be = be_cuda.?;
            be.profile = profile;
            defer if (profile) {
                inline for (@typeInfo(cuda_be.ProfCat).@"enum".fields, 0..) |f, ci| {
                    if (be.prof.n[ci] > 0)
                        stdout.print("  {s:<9} {d:>8.1} ms  ({d} launches)\n", .{ f.name, be.prof.ms[ci], be.prof.n[ci] }) catch {};
                }
                stdout.flush() catch {};
            };
            var model = try TensorPencil.models.qwen35_cuda.CudaLM.init(gpa, be, &lm, cap);
            defer model.deinit();
            if (img) |*e| {
                // Mixed prefill: tokens, then the image embeddings (in place
                // of the placeholder rows), then all but the last token —
                // the engine's generate() prefills that one and samples.
                const n_img = e.grid_w * e.grid_h;
                try model.prefill(ids.items[0..n_pre]);
                try model.prefillImage(e.embeds, e.grid_w, e.grid_h);
                try model.prefill(ids.items[n_pre + n_img .. ids.items.len - 1]);
            }
            t0 = Io.Clock.real.now(io).nanoseconds;
            break :blk try runSession(&model, null, &tok, io, gpa, &ids, opts, stdout, prompt, img_chat);
        },
        .vulkan => unreachable, // rejected in main
    };
    const t_end = Io.Clock.real.now(io).nanoseconds;
    const setup_s = @as(f64, @floatFromInt(t0 - t_init)) / 1e9;
    const elapsed_s = @as(f64, @floatFromInt(t_end - t0)) / 1e9;
    if (prompt != null) {
        try stdout.print("\n\n[{d} tokens in {d:.1}s, {d:.2} tok/s; setup {d:.1}s]\n", .{
            n, elapsed_s, @as(f64, @floatFromInt(n)) / elapsed_s, setup_s,
        });
    } else {
        try stdout.print("[session over: {d} tokens generated; setup was {d:.1}s]\n", .{ n, setup_s });
    }
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
        .gguf => |*g| {
            const arch = g.getStr("general.architecture") orelse return 32768;
            var buf: [64]u8 = undefined;
            const key = std.fmt.bufPrint(&buf, "{s}.context_length", .{arch}) catch return 32768;
            return @intCast(g.getUint(key) orelse 32768);
        },
        .safetensors => return 32768,
    }
}

/// KV-cache sizing for a session: the ceiling is the old fixed capacity
/// (one-shot request size, or the whole --max-context window for chat);
/// dynamic sessions commit only min(4096, prompt + 1) rows up front and grow
/// toward it. `fixed` forces initial == max (speculative decoding).
fn capacityPlan(opts: llm.engine.Options, prompt: ?[]const u8, n_prompt: usize, fixed: bool) !llm.engine.Capacity {
    const max = if (prompt == null) opts.max_context else try llm.engine.capacityFor(opts, n_prompt);
    if (fixed) return .fixed(max);
    return .{ .initial = @min(max, @max(n_prompt + 1, llm.kv_cache.initial_context)), .max = max };
}

const BackendKind = enum { cpu, @"zig-cuda", cuda, vulkan };

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
