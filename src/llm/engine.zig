//! Autoregressive generation loop: one full-sequence prefill into the KV
//! cache, then seq-1 decode steps (LLM_PLAN.md M2).

const std = @import("std");
const qwen3 = @import("../models/qwen3.zig");
const test_gate = @import("../test_gate.zig");
const tokenizer_mod = @import("tp_core").tokenizer;
const ops = @import("tp_ops");
const chat = @import("chat.zig");
const sample = @import("tp_core").sample;
const spec = @import("spec.zig");
const kv_cache_mod = @import("tp_core").kv_cache;

const Tokenizer = tokenizer_mod.Tokenizer;
const KvCache = kv_cache_mod.KvCache;

pub const Capacity = kv_cache_mod.Capacity;

pub const Options = struct {
    max_new_tokens: usize = 256,
    /// Context-window ceiling. The KV cache's growth limit is
    /// min(max_context, prompt + max_new_tokens) for one-shot prompts and
    /// max_context for chat sessions; only a small initial slice is
    /// committed up front (see capacityPlanFor / kv_cache.Capacity).
    max_context: usize = 4096,
    sampling: sample.Params = .{},
    /// RNG seed for sampling THIS generate call (irrelevant when temperature
    /// = 0). The sampler is constructed fresh per call, so a multi-turn
    /// driver must supply a fresh seed each turn (see sample.SeedSeq) or
    /// every turn replays the same RNG stream — a repeated prompt would then
    /// reproduce the identical "random" response.
    seed: u64 = 0,
    /// Speculative decoding: max drafted tokens per verify forward
    /// (0 = off). Requires a backend stepper with stepAll + truncate.
    spec_k: usize = 0,
    /// Tree drafting (LLM_PLAN.md M8): total tree size per verify forward —
    /// the root (pending token) plus up to tree_nodes-1 drafted branch nodes
    /// (0 = chain drafting). Greedy-only in v1; requires a drafter exposing
    /// proposeTree and a stepper exposing stepAllTree + commitTreePath
    /// (error.TreeUnsupported otherwise).
    tree_nodes: usize = 0,
    /// When set, speculative decoding accumulates draft/accept counts here.
    spec_stats: ?*spec.Stats = null,
    /// Cooperative cancellation: checked before each decode step; when it
    /// reads true, generation stops and returns the tokens produced so far
    /// (a clean stop, not an error). Lets a UI interrupt a long reply.
    cancel: ?*std.atomic.Value(bool) = null,
    /// Optional pause gate, consulted before each decode step (the same boundary
    /// as `cancel`). While paused the loop parks here — holding the KV cache and
    /// resident weights — until unpaused. See `ops/pause.zig`.
    pause: ?*ops.pause.Gate = null,
    /// Set true (when non-null) if the loop stops because the pause gate returned
    /// `.unload` — i.e. an unload-while-paused suspend, distinct from a cancel or
    /// a natural stop. Lets the caller keep the assistant turn OPEN (no
    /// closeAssistant) so a later reload can reprefill `ids` and continue. (Tier 3.)
    suspended_out: ?*bool = null,
    /// KV-cache element storage type (f32 default; f16 halves the footprint,
    /// lossy). Copied onto the Capacity that reaches every model init.
    kv_dtype: kv_cache_mod.KvDtype = .f32,
    /// Optional per-token residency hook. A cross-thread VRAM arbiter (the GUI,
    /// when a diffusion model contends for VRAM) publishes a new device-residency
    /// ceiling to this session's `vram.ControlPoint` from another thread; the
    /// decode loop invokes this at each token boundary so the target is enacted
    /// on the worker's OWN (context-bound) thread instead of raced from the
    /// arbiter thread. null on the CLI / studio (no coordinator).
    residency_poll: ?ResidencyPoll = null,
};

/// A `residency_poll`: `apply(ctx)` enacts any pending residency target on the
/// calling (worker) thread. Typically `ctx` is a session and `apply` is a thunk
/// over `vram.Participant.pollAndApply`.
pub const ResidencyPoll = struct {
    ctx: *anyopaque,
    apply: *const fn (ctx: *anyopaque) void,
};

/// KV-cache capacity ceiling for a given prompt; errors when the prompt
/// alone overflows the window.
pub fn capacityFor(opts: Options, prompt_len: usize) !usize {
    if (prompt_len >= opts.max_context) return error.PromptTooLong;
    return @min(opts.max_context, prompt_len + opts.max_new_tokens);
}

/// Dynamic sizing plan for a one-shot prompt: commit only enough rows for
/// the prompt (or the kv_cache.initial_context floor) and grow toward the
/// capacityFor ceiling as generation actually uses it. Backends without
/// growth support treat `.max` as the (old, fixed) capacity.
pub fn capacityPlanFor(opts: Options, prompt_len: usize) !Capacity {
    const max = try capacityFor(opts, prompt_len);
    return .{
        .initial = @min(max, @max(prompt_len + 1, kv_cache_mod.initial_context)),
        .max = max,
        .kv_dtype = opts.kv_dtype,
    };
}

/// Make room for `need` more rows, growing dynamic-capacity models (steppers
/// exposing ensureCapacity). error.ContextFull when the window — or the
/// memory backing its growth — can't cover the rows.
fn ensureRoom(model: anytype, need: usize) !void {
    if (need <= model.remaining()) return;
    const M = switch (@typeInfo(@TypeOf(model))) {
        .pointer => |p| p.child,
        else => @TypeOf(model),
    };
    if (comptime !@hasDecl(M, "ensureCapacity")) return error.ContextFull;
    const before = model.cached() + model.remaining(); // window before growth
    try model.ensureCapacity(model.cached() + need);
    const after = model.cached() + model.remaining();
    if (after != before)
        std.log.info("[ctx] KV window grew {d} -> {d} rows ({d} tokens committed)", .{ before, after, model.cached() });
}

/// Extend `ids` in place until a stop token, the token budget, or a full
/// context window. Decoded bytes stream to `out` (flushed per token, held
/// back until UTF-8-complete) when non-null; the stop token is not appended.
/// Returns the number of tokens generated.
///
/// `model` is a pointer to any backend stepper exposing
///   step(io, ids_new: []const u32, logits: []f32) — forward the new tokens
///     at the next cache positions and write last-position vocab logits,
///   cached() usize — committed cache length, and
///   remaining() usize — cache room left.
/// Speculative decoding (opts.spec_k > 0) additionally requires
///   stepAll(io, ids_new, logits) — one vocab row per new token, and
///   truncate(new_len) — roll the cache back to `new_len` tokens;
/// backends without them return error.SpecUnsupported.
/// Tree drafting (opts.tree_nodes > 0, spec.generateTree) requires
///   stepAllTree(io, tokens, parents, logits) — tree-verify forward, and
///   commitTreePath(path) — append the accepted root path to the cache;
/// plus a drafter exposing proposeTree (error.TreeUnsupported otherwise).
/// The first step call carries the not-yet-cached prompt suffix (prefill —
/// the whole prompt on turn one, only the new turn's tokens on later turns
/// of a multi-turn session); each later call carries the single sampled
/// token (decode).
pub fn generate(
    model: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: Options,
    out: ?*std.Io.Writer,
) !usize {
    if (opts.spec_k > 0) {
        const M = switch (@typeInfo(@TypeOf(model))) {
            .pointer => |p| p.child,
            else => @TypeOf(model),
        };
        if (comptime !@hasDecl(M, "stepAll")) return error.SpecUnsupported;
        var drafter: spec.NgramDrafter = .{};
        return spec.generate(model, &drafter, tok, io, gpa, ids, opts, out);
    }
    // Greedy on a GPU backend that exposes stepArgmax: pick the token on-device
    // and download just the id, skipping the full ~vocab logit transfer + host
    // argmax each step.
    {
        const M = switch (@typeInfo(@TypeOf(model))) {
            .pointer => |p| p.child,
            else => @TypeOf(model),
        };
        // Recent-window penalties (repetition/presence/frequency) modify the
        // full logits. Steppers exposing the *Pen variants scatter them
        // on-device (opPenalize) so the GPU sampling paths survive a penalty;
        // anything else falls through to the download + CPU-sample path while
        // one is active. min-p and top-p never need a fallback: they are
        // prefix cuts inside the shared distFromSorted tail.
        const pen_on = opts.sampling.penaltiesActive();
        if (opts.sampling.temperature <= 0) {
            if (comptime @hasDecl(M, "stepArgmax")) {
                if (!pen_on or comptime @hasDecl(M, "stepArgmaxPen"))
                    return generateGreedyArgmax(model, tok, io, gpa, ids, opts, out);
            }
        } else if (comptime @hasDecl(M, "stepSelect")) {
            if (!pen_on or comptime @hasDecl(M, "stepSelectPen"))
                return generateGpuSample(model, tok, io, gpa, ids, opts, out);
        }
    }

    const logits = try gpa.alloc(f32, model.vocab());
    defer gpa.free(logits);

    var sampler = sample.Sampler.init(opts.sampling, opts.seed);
    var stream: Utf8Stream = .{};

    const new = ids.items[model.cached()..];
    if (new.len == 0) return error.ContextFull;
    try ensureRoom(model, new.len);
    try model.step(io, new, logits);
    var n: usize = 0;
    while (n < opts.max_new_tokens) {
        if (opts.residency_poll) |rp| rp.apply(rp.ctx); // enact any arbiter-published VRAM target on this thread

        if (opts.cancel) |c| if (c.load(.acquire)) break;
        // Pause parks at the token boundary. `.canceled` stops like a cancel;
        // `.unload` also stops but flags a suspend so the caller keeps the turn
        // open and reprefill-resumes it on reload (Tier 3 unload-while-paused).
        if (opts.pause) |g| switch (g.checkpoint(io, opts.cancel)) {
            .proceed => {},
            .canceled => break,
            .unload => {
                if (opts.suspended_out) |so| so.* = true;
                break;
            },
        };
        const next = sampler.next(logits, ids.items);
        if (chat.isStop(next)) break;
        try ids.append(gpa, next);
        n += 1;
        if (out) |w| {
            const bytes = try tok.decodeAlloc(gpa, &.{next});
            defer gpa.free(bytes);
            try stream.write(w, bytes);
            try w.flush();
        }
        if (n == opts.max_new_tokens) break; // budget spent: skip the forward whose logits nobody reads
        if (model.remaining() == 0) {
            // Dynamic capacity: commit more rows (possibly evicting weights
            // into the streaming path) before declaring the window full.
            // (Runtime comparison, not a switch: models whose ensureCapacity
            // can only fail with ContextFull would make an else unreachable.)
            ensureRoom(model, 1) catch |err| {
                if (err == error.ContextFull) break;
                return err;
            };
        }
        try model.step(io, &.{next}, logits);
    }
    return n;
}

/// Dispatch to the model's penalized stepper variant when it has one (the
/// penalties are scattered onto the device logits before the argmax); with an
/// empty entry list the two variants are identical. generate()'s dispatch only
/// routes active penalties to models exposing the Pen variants, so the plain
/// fallback here never drops a penalty.
fn stepArgmaxOf(model: anytype, io: std.Io, new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
    const M = switch (@typeInfo(@TypeOf(model))) {
        .pointer => |p| p.child,
        else => @TypeOf(model),
    };
    if (comptime @hasDecl(M, "stepArgmaxPen")) return model.stepArgmaxPen(io, new, pen, sp);
    std.debug.assert(pen.len == 0);
    return model.stepArgmax(io, new);
}

/// stepSelect twin of `stepArgmaxOf`.
fn stepSelectOf(model: anytype, io: std.Io, new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
    const M = switch (@typeInfo(@TypeOf(model))) {
        .pointer => |p| p.child,
        else => @TypeOf(model),
    };
    if (comptime @hasDecl(M, "stepSelectPen")) return model.stepSelectPen(io, new, pen, sp, out_id, out_logit);
    std.debug.assert(pen.len == 0);
    return model.stepSelect(io, new, out_id, out_logit);
}

/// Greedy generation via on-device argmax (`model.stepArgmax`): no host logits
/// buffer, no per-step vocab download — just the sampled id comes back. Emits
/// identical tokens to the full-vocab greedy path (both are argmax, lowest
/// index on ties). Active penalties ride along as a per-token entry upload to
/// the device penalize kernel (same window the CPU path scans: `ids` so far).
fn generateGreedyArgmax(
    model: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: Options,
    out: ?*std.Io.Writer,
) !usize {
    var stream: Utf8Stream = .{};
    var pen_buf: [sample.max_penalty_window]sample.PenaltyEntry = undefined;
    const new = ids.items[model.cached()..];
    if (new.len == 0) return error.ContextFull;
    try ensureRoom(model, new.len);
    var next = try stepArgmaxOf(model, io, new, sample.collectPenalties(ids.items, opts.sampling, &pen_buf), opts.sampling);
    var n: usize = 0;
    while (n < opts.max_new_tokens) {
        if (opts.residency_poll) |rp| rp.apply(rp.ctx); // enact any arbiter-published VRAM target on this thread

        if (opts.cancel) |c| if (c.load(.acquire)) break;
        // Pause parks at the token boundary. `.canceled` stops like a cancel;
        // `.unload` also stops but flags a suspend so the caller keeps the turn
        // open and reprefill-resumes it on reload (Tier 3 unload-while-paused).
        if (opts.pause) |g| switch (g.checkpoint(io, opts.cancel)) {
            .proceed => {},
            .canceled => break,
            .unload => {
                if (opts.suspended_out) |so| so.* = true;
                break;
            },
        };
        if (chat.isStop(next)) break;
        try ids.append(gpa, next);
        n += 1;
        if (out) |w| {
            const bytes = try tok.decodeAlloc(gpa, &.{next});
            defer gpa.free(bytes);
            try stream.write(w, bytes);
            try w.flush();
        }
        if (n == opts.max_new_tokens) break;
        if (model.remaining() == 0) {
            ensureRoom(model, 1) catch |err| {
                if (err == error.ContextFull) break;
                return err;
            };
        }
        next = try stepArgmaxOf(model, io, &.{next}, sample.collectPenalties(ids.items, opts.sampling, &pen_buf), opts.sampling);
    }
    return n;
}

/// Stochastic generation via on-device top-k (`model.stepSelect`): the device
/// selects the top-k candidates and returns just those (id,logit) pairs; the
/// CPU Sampler runs the identical softmax/min-p/top-p/RNG tail over them
/// (bit-identical to the full-vocab path for the same logits + seed). No
/// per-step vocab download. Active penalties are scattered onto the device
/// logits BEFORE the top-k (stepSelectPen), so the candidates are the true
/// post-penalty top set and the CPU tail never re-applies them.
fn generateGpuSample(
    model: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: Options,
    out: ?*std.Io.Writer,
) !usize {
    const cap = model.maxSelect();
    const out_id = try gpa.alloc(u32, cap);
    defer gpa.free(out_id);
    const out_logit = try gpa.alloc(f32, cap);
    defer gpa.free(out_logit);
    const cands = try gpa.alloc(sample.Candidate, cap);
    defer gpa.free(cands);

    var sampler = sample.Sampler.init(opts.sampling, opts.seed);
    var stream: Utf8Stream = .{};
    var pen_buf: [sample.max_penalty_window]sample.PenaltyEntry = undefined;
    const new = ids.items[model.cached()..];
    if (new.len == 0) return error.ContextFull;
    try ensureRoom(model, new.len);
    var count = try stepSelectOf(model, io, new, sample.collectPenalties(ids.items, opts.sampling, &pen_buf), opts.sampling, out_id, out_logit);
    var n: usize = 0;
    while (n < opts.max_new_tokens) {
        if (opts.residency_poll) |rp| rp.apply(rp.ctx); // enact any arbiter-published VRAM target on this thread

        if (opts.cancel) |c| if (c.load(.acquire)) break;
        // Pause parks at the token boundary. `.canceled` stops like a cancel;
        // `.unload` also stops but flags a suspend so the caller keeps the turn
        // open and reprefill-resumes it on reload (Tier 3 unload-while-paused).
        if (opts.pause) |g| switch (g.checkpoint(io, opts.cancel)) {
            .proceed => {},
            .canceled => break,
            .unload => {
                if (opts.suspended_out) |so| so.* = true;
                break;
            },
        };
        for (cands[0..count], out_id[0..count], out_logit[0..count]) |*c, id, lg| c.* = .{ .id = id, .logit = lg };
        const next = sampler.nextFromCandidates(cands[0..count]);
        if (chat.isStop(next)) break;
        try ids.append(gpa, next);
        n += 1;
        if (out) |w| {
            const bytes = try tok.decodeAlloc(gpa, &.{next});
            defer gpa.free(bytes);
            try stream.write(w, bytes);
            try w.flush();
        }
        if (n == opts.max_new_tokens) break;
        if (model.remaining() == 0) {
            ensureRoom(model, 1) catch |err| {
                if (err == error.ContextFull) break;
                return err;
            };
        }
        count = try stepSelectOf(model, io, &.{next}, sample.collectPenalties(ids.items, opts.sampling, &pen_buf), opts.sampling, out_id, out_logit);
    }
    return n;
}

/// CPU backend stepper: qwen3.CausalLM + host KvCache; the LM head is the
/// tied bf16 embedding GEMV.
pub const CpuModel = struct {
    lm: *const qwen3.CausalLM,
    gpa: std.mem.Allocator,
    cache: KvCache,
    rope: ops.rope.RopeTables(1),
    last_hidden: []f32,
    /// Growth ceiling (rows); the cache starts at cap.initial and grows here.
    max_capacity: usize,
    /// Retained per-layer batch K/V of the last tree verify
    /// ([n_layers][tree_n][kv_dim]); lazily allocated to
    /// n_layers * spec.max_tree_nodes rows on first stepAllTree.
    tree_k: ?[]f32 = null,
    tree_v: ?[]f32 = null,
    tree_n: usize = 0,

    pub fn init(gpa: std.mem.Allocator, lm: *const qwen3.CausalLM, cap: Capacity) !CpuModel {
        var cache = try KvCache.init(gpa, lm.cfg.n_layers, cap.initial, lm.cfg.kvDim(), cap.kv_dtype);
        errdefer cache.deinit(gpa);
        var rope = try ops.rope.RopeTables(1).init(gpa, .{
            .{ .head_dim = qwen3.head_dim, .theta = lm.cfg.rope_theta },
        }, cap.initial);
        errdefer rope.deinit(gpa);
        const last_hidden = try gpa.alloc(f32, lm.cfg.hidden);
        return .{ .lm = lm, .gpa = gpa, .cache = cache, .rope = rope, .last_hidden = last_hidden, .max_capacity = cap.max };
    }

    pub fn capacityMax(self: *const CpuModel) usize {
        return self.max_capacity;
    }

    /// Grow the cache (and the RoPE table) to hold at least `min_rows`.
    /// error.ContextFull past the window or when host memory runs out.
    pub fn ensureCapacity(self: *CpuModel, min_rows: usize) !void {
        const target = (try kv_cache_mod.growPlan(self.cache.capacity, self.max_capacity, min_rows)) orelse return;
        self.cache.grow(self.gpa, target) catch return error.ContextFull;
        self.rope.regrow(self.gpa, target) catch return error.ContextFull;
    }

    pub fn deinit(self: *CpuModel) void {
        self.cache.deinit(self.gpa);
        self.rope.deinit(self.gpa);
        self.gpa.free(self.last_hidden);
        if (self.tree_k) |t| self.gpa.free(t);
        if (self.tree_v) |t| self.gpa.free(t);
        self.* = undefined;
    }

    pub fn cached(self: *const CpuModel) usize {
        return self.cache.len;
    }

    pub fn remaining(self: *const CpuModel) usize {
        return self.cache.remaining();
    }

    pub fn vocab(self: *const CpuModel) usize {
        _ = self;
        return qwen3.vocab_size;
    }

    pub fn step(self: *CpuModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        try self.lm.forwardCached(io, self.gpa, ids_new, &self.cache, self.rope.get(0), self.last_hidden);
        try ops.matmul.matmul(io, self.gpa, logits, self.last_hidden, 1, self.lm.lmHead(), null);
    }

    /// step, but with vocab logits for every new token ([ids_new.len, vocab]
    /// row-major) — the speculative-decode verify forward.
    pub fn stepAll(self: *CpuModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        const n = ids_new.len;
        std.debug.assert(logits.len == n * qwen3.vocab_size);
        const hid = try self.gpa.alloc(f32, n * self.lm.cfg.hidden);
        defer self.gpa.free(hid);
        try self.lm.forwardCached(io, self.gpa, ids_new, &self.cache, self.rope.get(0), hid);
        try ops.matmul.matmul(io, self.gpa, logits, hid, n, self.lm.lmHead(), null);
    }

    pub fn truncate(self: *CpuModel, new_len: usize) void {
        self.cache.truncate(new_len);
    }

    /// Tree-verify forward (spec.generateTree): vocab logits for every tree
    /// node; the batch K/V rows are retained (NOT committed to the cache)
    /// until commitTreePath copies the accepted path in.
    pub fn stepAllTree(self: *CpuModel, io: std.Io, tokens: []const u32, parents: []const u32, logits: []f32) !void {
        const cfg = self.lm.cfg;
        const n = tokens.len;
        std.debug.assert(n >= 1 and n <= spec.max_tree_nodes);
        std.debug.assert(logits.len == n * qwen3.vocab_size);
        if (self.tree_k == null) {
            self.tree_k = try self.gpa.alloc(f32, cfg.n_layers * spec.max_tree_nodes * cfg.kvDim());
            self.tree_v = try self.gpa.alloc(f32, cfg.n_layers * spec.max_tree_nodes * cfg.kvDim());
        }
        const hid = try self.gpa.alloc(f32, n * cfg.hidden);
        defer self.gpa.free(hid);
        try self.lm.forwardTree(io, self.gpa, tokens, parents, &self.cache, self.rope.get(0), self.tree_k.?, self.tree_v.?, hid);
        try ops.matmul.matmul(io, self.gpa, logits, hid, n, self.lm.lmHead(), null);
        self.tree_n = n;
    }

    /// Copy the accepted root path's K/V rows from the retained tree batch
    /// into the cache (path[0] == 0; strictly ascending node indices).
    pub fn commitTreePath(self: *CpuModel, path: []const usize) !void {
        const kvd = self.lm.cfg.kvDim();
        for (path) |idx| {
            std.debug.assert(idx < self.tree_n);
            for (0..self.lm.cfg.n_layers) |l| {
                const row = (l * self.tree_n + idx) * kvd;
                self.cache.write(l, self.tree_k.?[row..][0..kvd], self.tree_v.?[row..][0..kvd]);
            }
            self.cache.commit(1);
        }
    }
};

/// Byte-level BPE tokens can split multi-byte UTF-8 sequences; hold back an
/// incomplete trailing sequence until the token that completes it.
pub const Utf8Stream = struct {
    pending: [3]u8 = undefined,
    pending_len: usize = 0,

    pub fn write(self: *Utf8Stream, w: *std.Io.Writer, bytes: []const u8) !void {
        var buf: [3 + 128]u8 = undefined; // decoded token bytes are <= 128
        std.debug.assert(bytes.len <= 128);
        @memcpy(buf[0..self.pending_len], self.pending[0..self.pending_len]);
        @memcpy(buf[self.pending_len..][0..bytes.len], bytes);
        const all = buf[0 .. self.pending_len + bytes.len];

        const complete = completeUtf8Prefix(all);
        try w.writeAll(all[0..complete]);
        self.pending_len = all.len - complete;
        @memcpy(self.pending[0..self.pending_len], all[complete..]);
    }
};

/// Length of the longest prefix that does not end mid-codepoint. Invalid
/// bytes are passed through rather than held forever.
fn completeUtf8Prefix(bytes: []const u8) usize {
    var back: usize = 0;
    var i = bytes.len;
    while (i > 0 and back < 4) {
        i -= 1;
        back += 1;
        const b = bytes[i];
        if (b & 0x80 == 0) return i + 1; // ASCII tail: complete
        if (b & 0xC0 == 0xC0) { // leading byte of a multi-byte sequence
            const need = std.unicode.utf8ByteSequenceLength(b) catch return bytes.len;
            return if (bytes.len - i >= need) bytes.len else i;
        }
        // 0b10xxxxxx continuation: keep scanning backwards.
    }
    return bytes.len; // >= 4 trailing continuations: invalid, flush as-is
}

// --- tests -----------------------------------------------------------------

test "utf8 stream holds back split codepoints" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var stream: Utf8Stream = .{};

    // "é" (0xC3 0xA9) split across two writes; "😀" (4 bytes) split 1+3.
    try stream.write(&w, "a\xC3");
    try std.testing.expectEqualStrings("a", w.buffered());
    try stream.write(&w, "\xA9b");
    try std.testing.expectEqualStrings("a\xC3\xA9b", w.buffered());
    try stream.write(&w, "\xF0");
    try std.testing.expectEqualStrings("a\xC3\xA9b", w.buffered());
    try stream.write(&w, "\x9F\x98\x80!");
    try std.testing.expectEqualStrings("a\xC3\xA9b\xF0\x9F\x98\x80!", w.buffered());
}

test "utf8 prefix scanner" {
    try std.testing.expectEqual(@as(usize, 3), completeUtf8Prefix("abc"));
    try std.testing.expectEqual(@as(usize, 1), completeUtf8Prefix("a\xC3"));
    try std.testing.expectEqual(@as(usize, 4), completeUtf8Prefix("ab\xC3\xA9"));
    try std.testing.expectEqual(@as(usize, 1), completeUtf8Prefix("a\xF0\x9F\x98"));
    try std.testing.expectEqual(@as(usize, 0), completeUtf8Prefix("\xF0\x9F"));
}

// Multi-turn cache continuation: a second generate() on the same model must
// prefill only the new turn's tokens (kept tiny — gated on the model).
test "multi-turn generation continues the cache" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    try test_gate.requireModelFile(io, te_path);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: Options = .{ .max_new_tokens = 1, .sampling = .{ .temperature = 0 } };
    var model = try CpuModel.init(gpa, &lm, .fixed(128));
    defer model.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Say hi.", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    _ = try generate(&model, &tok, io, gpa, &ids, opts, null);
    // A max-tokens-truncated turn leaves exactly the last sampled token
    // uncached (it is prefilled with the next turn instead).
    try std.testing.expectEqual(ids.items.len - 1, model.cached());

    // Second turn: only the new tokens are prefilled, on top of the cache.
    try chat.closeAssistant(gpa, &ids);
    try chat.appendUser(&tok, gpa, "Again.", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const before = model.cached();
    _ = try generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(model.cached() > before);
    try std.testing.expectEqual(ids.items.len - 1, model.cached());
}

// Dynamic-capacity growth is transparent: a cache that starts far too small
// for the prompt (growing during prefill) must produce the same greedy token
// as a full-size cache. Gated on the checkpoint; kept to 1 token.
test "generation with a growing cache matches fixed capacity" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    try test_gate.requireModelFile(io, te_path);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Say hi.", &prompt);
    try chat.openAssistant(&tok, gpa, &prompt);

    const opts: Options = .{ .max_new_tokens = 1, .sampling = .{ .temperature = 0 } };
    var out_fixed: u32 = undefined;
    var out_grown: u32 = undefined;
    for ([2]Capacity{ .fixed(128), .{ .initial = 4, .max = 128 } }, [2]*u32{ &out_fixed, &out_grown }) |cap, out| {
        var model = try CpuModel.init(gpa, &lm, cap);
        defer model.deinit();
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ids.appendSlice(gpa, prompt.items);
        const n = try generate(&model, &tok, io, gpa, &ids, opts, null);
        try std.testing.expectEqual(@as(usize, 1), n);
        out.* = ids.items[ids.items.len - 1];
    }
    try std.testing.expectEqual(out_fixed, out_grown);
}

// End-to-end mechanics (forward -> lm_head -> argmax -> append) against the
// real checkpoint; skipped when the model is absent. Kept to 2 tokens — each
// costs a full 36-layer forward.
test "greedy generation produces valid tokens" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    try test_gate.requireModelFile(io, te_path);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Say hi.", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const prompt_len = ids.items.len;

    const opts: Options = .{ .max_new_tokens = 2, .sampling = .{ .temperature = 0 } };
    var model = try CpuModel.init(gpa, &lm, try capacityPlanFor(opts, prompt_len));
    defer model.deinit();
    const n = try generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expectEqual(prompt_len + n, ids.items.len);
    try std.testing.expect(n > 0); // "Say hi." should not stop on token one
    for (ids.items[prompt_len..]) |id| try std.testing.expect(id < qwen3.vocab_size);
}
