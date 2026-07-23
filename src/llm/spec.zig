//! Speculative decoding: a cheap drafter proposes up to k tokens, the target
//! model verifies them all in one batched forward (one KV-cache append, one
//! multi-row LM head), and the cache is rolled back past the first rejection.
//!
//! Lossless: a drafted token is accepted with its probability under the
//! target's fully processed sampling distribution, and a rejection resamples
//! from the renormalized residual — so emitted tokens follow exactly the
//! distribution vanilla sampling draws from (byte-identical for greedy,
//! where no randomness is consumed at all).
//!
//! The win comes from decode being weight-bandwidth-bound: a k+1-row verify
//! forward reads the same weights as a 1-row decode step, so accepted
//! drafts are nearly free tokens.

const std = @import("std");
const qwen3 = @import("../models/qwen3.zig");
const test_gate = @import("../test_gate.zig");
const tokenizer_mod = @import("tp_core").tokenizer;
const engine = @import("engine.zig");
const chat = @import("chat.zig");
const sample = @import("tp_core").sample;
const spec_limits = @import("tp_core").spec_limits;

const Tokenizer = tokenizer_mod.Tokenizer;

/// Speculative-decode size caps. Defined at the core level
/// (`src/spec_limits.zig`) so model backends can size their K/V + verify
/// buffers from them without importing this driver (which depends upward on the
/// generation engine); re-exported here so `spec.max_draft` /
/// `spec.max_tree_nodes` keep working.
pub const max_draft = spec_limits.max_draft;
pub const max_tree_nodes = spec_limits.max_tree_nodes;

pub const Stats = struct {
    /// Tokens proposed by the drafter.
    drafted: usize = 0,
    /// Drafted tokens accepted by the target model.
    accepted: usize = 0,
    /// Verify forwards run (each also carries the pending sampled token,
    /// so this is the count of target-model forwards after prefill).
    forwards: usize = 0,
};

/// Prompt-lookup drafter: find the longest trailing n-gram (max_n down to
/// min_n) that occurred earlier in the context and propose the tokens that
/// followed its most recent occurrence. No model, no weights — pays off on
/// repetition (code, structured output, multi-turn chat restating context)
/// and costs only an already-needed forward when wrong.
pub const NgramDrafter = struct {
    max_n: usize = 4,
    /// Bigram floor: 1-gram matches draft near-noise (~10% acceptance) and
    /// every drafted row makes the verify forward wider.
    min_n: usize = 2,

    pub fn propose(self: *const NgramDrafter, ids: []const u32, buf: []u32) usize {
        std.debug.assert(self.min_n >= 1 and self.max_n >= self.min_n);
        if (buf.len == 0 or ids.len < 2) return 0;
        var n = @min(self.max_n, ids.len - 1);
        while (n >= self.min_n) : (n -= 1) {
            const suffix = ids[ids.len - n ..];
            var start = ids.len - n; // candidate match start, scanned backwards
            while (start > 0) {
                start -= 1;
                if (!std.mem.eql(u32, ids[start..][0..n], suffix)) continue;
                const count = @min(buf.len, ids.len - (start + n));
                @memcpy(buf[0..count], ids[start + n ..][0..count]);
                return count;
            }
        }
        return 0;
    }
};

/// Draft-model drafter (LLM_PLAN.md M5): a second, smaller CausalLM on its
/// own backend stepper proposes greedy continuations. It mirrors the token
/// history its KV cache reflects, so after the target rejects drafts it
/// re-syncs by truncating to the longest common prefix and prefilling the
/// rest. Draft errors and a full draft context degrade to proposing nothing
/// (the target just decodes vanilla) — the drafter can never corrupt output.
pub fn ModelDrafter(comptime Stepper: type) type {
    return struct {
        model: *Stepper,
        gpa: std.mem.Allocator,
        io: std.Io,
        /// Tokens whose K/V the draft cache currently holds.
        hist: std.ArrayList(u32) = .empty,
        logits: []f32,

        const Self = @This();

        pub fn init(gpa: std.mem.Allocator, io: std.Io, model: *Stepper) !Self {
            return .{ .model = model, .gpa = gpa, .io = io, .logits = try gpa.alloc(f32, qwen3.vocab_size) };
        }

        pub fn deinit(self: *Self) void {
            self.hist.deinit(self.gpa);
            self.gpa.free(self.logits);
        }

        pub fn propose(self: *Self, ids: []const u32, buf: []u32) usize {
            return self.proposeInner(ids, buf) catch 0;
        }

        fn proposeInner(self: *Self, ids: []const u32, buf: []u32) !usize {
            if (buf.len == 0) return 0;
            // Re-sync the draft cache with reality: keep the longest common
            // prefix of what we cached and what the target committed, then
            // prefill the rest (rejected drafts fall off; on full acceptance
            // one token is recomputed to obtain fresh logits).
            var p = commonPrefix(self.hist.items, ids);
            if (p == ids.len) p -= 1;
            if (p < self.model.cached()) self.model.truncate(p);
            self.hist.clearRetainingCapacity();
            try self.hist.appendSlice(self.gpa, ids);

            const new = ids[self.model.cached()..];
            if (new.len > self.model.remaining()) return 0; // draft context full
            try self.model.step(self.io, new, self.logits);

            // Greedy rollout; the last proposal is never fed back (its
            // logits would go unread).
            var m: usize = 0;
            while (true) {
                const d = sample.argmax(self.logits);
                buf[m] = d;
                m += 1;
                if (m == buf.len or chat.isStop(d) or self.model.remaining() == 0) break;
                try self.model.step(self.io, &.{d}, self.logits);
                try self.hist.append(self.gpa, d);
            }
            return m;
        }

        fn commonPrefix(a: []const u32, b: []const u32) usize {
            const n = @min(a.len, b.len);
            for (0..n) |i| {
                if (a[i] != b[i]) return i;
            }
            return n;
        }
    };
}

/// Adapt a chain drafter to the tree interface as a single-path tree — no
/// acceptance gain over chain mode (a chain is a degenerate tree), but it
/// lets any drafter drive the tree-verify path (parity tests, plumbing).
pub fn ChainAsTree(comptime D: type) type {
    return struct {
        inner: *D,

        pub fn proposeTree(self: @This(), ids: []const u32, tokens: []u32, parents: []u32, max_depth: usize) usize {
            const m = self.inner.propose(ids, tokens[0..@min(tokens.len, max_depth)]);
            // Slice index i is node i+1; a chain's node i+1 hangs off node i.
            for (parents[0..m], 0..) |*p, i| p.* = @intCast(i);
            return m;
        }
    };
}

/// Speculative counterpart of engine.generate — same contract (extends `ids`
/// in place, streams to `out`, returns tokens generated), driven by a
/// `drafter` exposing propose(ids: []const u32, buf: []u32) usize.
///
/// Positions per iteration, with L = ids.len at the verify forward: the
/// pending token sits at L-1 (sampled but not yet cached), drafts at
/// L..L+m-1. Logits row i predicts position L+i: rows 0..m-1 judge the
/// drafts, row m yields a free bonus token when everything is accepted.
pub fn generate(
    model: anytype,
    drafter: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: engine.Options,
    out: ?*std.Io.Writer,
) !usize {
    const D = PtrChild(@TypeOf(drafter));
    const M = PtrChild(@TypeOf(model));
    if (opts.tree_nodes > 0) {
        if (comptime @hasDecl(D, "proposeTree") and @hasDecl(M, "stepAllTree")) {
            // v1 trees are greedy-only: temperature > 0 needs recursive
            // multi-child rejection sampling (SpecInfer-style residuals).
            if (opts.sampling.temperature == 0)
                return generateTree(model, drafter, tok, io, gpa, ids, opts, out);
        }
        return error.TreeUnsupported;
    }
    // A tree-only drafter cannot serve chain mode (prunes the chain path,
    // which calls drafter.propose, at comptime).
    if (comptime !@hasDecl(D, "propose")) return error.TreeUnsupported;
    // Greedy on a GPU stepper: verify by per-row on-device argmax, downloading
    // just the ids instead of (k+1)*vocab logits. Acceptance is identical to
    // the download path — the greedy dist is a point mass at the argmax.
    // Repetition penalty needs the full logits, so it stays on the download path.
    if (comptime @hasDecl(M, "stepAllArgmax")) {
        if (opts.sampling.temperature == 0 and opts.sampling.repeat_penalty == 1.0)
            return generateChainGreedy(model, drafter, tok, io, gpa, ids, opts, out);
    }
    const vocab = qwen3.vocab_size;
    const k_max: usize = @min(opts.spec_k, max_draft);
    std.debug.assert(k_max > 0);
    const logits = try gpa.alloc(f32, (k_max + 1) * vocab);
    defer gpa.free(logits);

    var sampler = sample.Sampler.init(opts.sampling, opts.seed);
    var stream: engine.Utf8Stream = .{};

    // Prefill the uncached suffix; only the last position's logits matter.
    const new = ids.items[model.cached()..];
    if (new.len == 0 or new.len > model.remaining()) return error.ContextFull;
    try model.step(io, new, logits[0..vocab]);

    var n: usize = 0;
    var pending = sampler.next(logits[0..vocab], ids.items);
    while (true) {
        // Cooperative stop/pause at the token boundary (mirrors engine.zig).
        if (opts.cancel) |c| if (c.load(.acquire)) break;
        if (opts.pause) |pg| if (pg.checkpoint(io, opts.cancel) != .proceed) break;
        if (chat.isStop(pending)) break;
        try ids.append(gpa, pending);
        n += 1;
        try emit(tok, gpa, &stream, out, pending);
        if (n == opts.max_new_tokens) break; // pending stays uncached, like vanilla
        if (model.remaining() == 0) break;

        // Draft, then verify [pending] ++ draft in one forward. Cap the
        // draft so accepted tokens can neither overflow the cache (pending
        // takes one slot) nor the token budget (bonus token takes one).
        var buf: [max_draft + 1]u32 = undefined;
        const k_eff: usize = @min(k_max, @min(model.remaining() - 1, opts.max_new_tokens - n - 1));
        const m = drafter.propose(ids.items, buf[1 .. 1 + k_eff]);
        buf[0] = pending;
        const draft = buf[1 .. 1 + m];
        if (opts.spec_stats) |s| {
            s.drafted += m;
            s.forwards += 1;
        }
        try model.stepAll(io, buf[0 .. 1 + m], logits[0 .. (m + 1) * vocab]);

        // Accept left to right; the first rejection resamples and rolls back.
        var next_tok: ?u32 = null;
        for (draft, 0..) |d, i| {
            const row = logits[i * vocab ..][0..vocab];
            const dist = sampler.dist(row, ids.items);
            if (!sampler.accept(&dist, d)) {
                next_tok = dist.sampleExcluding(sampler.rng.random(), d);
                break;
            }
            if (opts.spec_stats) |s| s.accepted += 1;
            if (chat.isStop(d)) {
                model.truncate(ids.items.len); // drop the stop's row + rejected tail
                return n;
            }
            try ids.append(gpa, d);
            n += 1;
            try emit(tok, gpa, &stream, out, d);
            if (n == opts.max_new_tokens) {
                model.truncate(ids.items.len);
                return n;
            }
        }

        if (next_tok) |t| {
            // Rejected at some draft position: everything past the accepted
            // prefix (== ids) is invalid cache.
            model.truncate(ids.items.len);
            pending = t;
        } else {
            // All drafts accepted (or none proposed): the last row is a free
            // extra sample.
            const row = logits[m * vocab ..][0..vocab];
            pending = sampler.next(row, ids.items);
        }
    }
    return n;
}

/// Greedy chain speculative decoding via on-device per-row argmax
/// (`model.stepAllArgmax`): the verify forward returns just the target's argmax
/// token per row (a few ids) instead of the (k+1)*vocab logit block. Acceptance
/// is byte-identical to the download path: the greedy target distribution is a
/// point mass at the argmax, so a draft is accepted iff it equals g[i], a
/// rejection yields g[i], and full acceptance takes g[m] as the bonus token.
fn generateChainGreedy(
    model: anytype,
    drafter: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: engine.Options,
    out: ?*std.Io.Writer,
) !usize {
    const k_max: usize = @min(opts.spec_k, max_draft);
    std.debug.assert(k_max > 0);
    var stream: engine.Utf8Stream = .{};

    const new = ids.items[model.cached()..];
    if (new.len == 0 or new.len > model.remaining()) return error.ContextFull;
    var pending = try model.stepArgmax(io, new);

    var n: usize = 0;
    var g: [max_draft + 1]u32 = undefined;
    while (true) {
        // Cooperative stop/pause at the token boundary (mirrors engine.zig).
        if (opts.cancel) |c| if (c.load(.acquire)) break;
        if (opts.pause) |pg| if (pg.checkpoint(io, opts.cancel) != .proceed) break;
        if (chat.isStop(pending)) break;
        try ids.append(gpa, pending);
        n += 1;
        try emit(tok, gpa, &stream, out, pending);
        if (n == opts.max_new_tokens) break; // pending stays uncached, like vanilla
        if (model.remaining() == 0) break;

        var buf: [max_draft + 1]u32 = undefined;
        const k_eff: usize = @min(k_max, @min(model.remaining() - 1, opts.max_new_tokens - n - 1));
        const m = drafter.propose(ids.items, buf[1 .. 1 + k_eff]);
        buf[0] = pending;
        const draft = buf[1 .. 1 + m];
        if (opts.spec_stats) |s| {
            s.drafted += m;
            s.forwards += 1;
        }
        // g[i] = the target's greedy token given the prefix through row i.
        try model.stepAllArgmax(io, buf[0 .. 1 + m], g[0 .. 1 + m]);

        var next_tok: ?u32 = null;
        for (draft, 0..) |d, i| {
            if (d != g[i]) {
                next_tok = g[i]; // reject → resample = the point-mass argmax
                break;
            }
            if (opts.spec_stats) |s| s.accepted += 1;
            if (chat.isStop(d)) {
                model.truncate(ids.items.len);
                return n;
            }
            try ids.append(gpa, d);
            n += 1;
            try emit(tok, gpa, &stream, out, d);
            if (n == opts.max_new_tokens) {
                model.truncate(ids.items.len);
                return n;
            }
        }
        if (next_tok) |t| {
            model.truncate(ids.items.len);
            pending = t;
        } else {
            pending = g[m]; // all accepted → the row-m bonus token
        }
    }
    return n;
}

/// Tree-drafted speculative decoding (LLM_PLAN.md M8), greedy-only. The
/// drafter proposes a branching tree of candidate continuations instead of a
/// single chain, one forward verifies every node under a tree-attention
/// mask, and the walk keeps the deepest root path whose tokens match the
/// target's greedy choices — a chain dies at its first wrong token, a tree
/// survives as long as ANY branch guessed right.
///
/// Node layout: node 0 is the root (the pending token, already emitted, at
/// position ids.len - 1); draft node i > 0 carries tokens[i] with
/// parents[i] < i, at position ids.len - 1 + depth(i). Verify logits row i
/// predicts the token FOLLOWING node i given (committed prefix + node i's
/// root path). Tree nodes never touch the linear KV cache during the
/// forward; commitTreePath copies the accepted path's rows in afterwards.
///
/// Byte-identical to vanilla greedy by induction: the walk emits exactly
/// argmax chains of the verify logits, and each row's logits equal what
/// vanilla decode would produce for the same committed context.
pub fn generateTree(
    model: anytype,
    drafter: anytype,
    tok: *const Tokenizer,
    io: std.Io,
    gpa: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    opts: engine.Options,
    out: ?*std.Io.Writer,
) !usize {
    const vocab = qwen3.vocab_size;
    const n_max: usize = @min(@max(opts.tree_nodes, 1), max_tree_nodes);
    const logits = try gpa.alloc(f32, n_max * vocab);
    defer gpa.free(logits);

    var sampler = sample.Sampler.init(opts.sampling, opts.seed);
    var stream: engine.Utf8Stream = .{};

    const new = ids.items[model.cached()..];
    if (new.len == 0 or new.len > model.remaining()) return error.ContextFull;
    try model.step(io, new, logits[0..vocab]);

    var n: usize = 0;
    var pending = sampler.next(logits[0..vocab], ids.items);
    var node_tokens: [max_tree_nodes]u32 = undefined;
    var node_parents: [max_tree_nodes]u32 = undefined;
    while (true) {
        // Cooperative stop/pause at the token boundary (mirrors engine.zig).
        if (opts.cancel) |c| if (c.load(.acquire)) break;
        if (opts.pause) |pg| if (pg.checkpoint(io, opts.cancel) != .proceed) break;
        if (chat.isStop(pending)) break;
        try ids.append(gpa, pending);
        n += 1;
        try emit(tok, gpa, &stream, out, pending);
        if (n == opts.max_new_tokens) break; // pending stays uncached, like vanilla
        if (model.remaining() == 0) break;

        // Draft a tree rooted at pending. Depth is capped so a fully
        // accepted path can neither overflow the cache (the root takes one
        // slot, an accepted node at depth d lands at position len + d) nor
        // the token budget.
        node_tokens[0] = pending;
        node_parents[0] = 0;
        const max_depth = @min(model.remaining() - 1, opts.max_new_tokens - n);
        const cap = if (max_depth == 0) 0 else n_max - 1;
        const m = if (cap == 0) 0 else drafter.proposeTree(ids.items, node_tokens[1 .. 1 + cap], node_parents[1 .. 1 + cap], max_depth);
        if (std.debug.runtime_safety) validateTree(node_parents[0 .. 1 + m], max_depth);
        if (opts.spec_stats) |s| {
            s.drafted += m;
            s.forwards += 1;
        }
        try model.stepAllTree(io, node_tokens[0 .. 1 + m], node_parents[0 .. 1 + m], logits[0 .. (1 + m) * vocab]);

        // Greedy acceptance walk: descend from the root as long as some
        // child guessed the target's argmax; the first miss (or a leaf)
        // yields the correction/bonus token as the next pending.
        var path: [max_tree_nodes]usize = undefined;
        path[0] = 0;
        var plen: usize = 1;
        var cur: usize = 0;
        var next_pending: ?u32 = null;
        walk: while (true) {
            const row = logits[cur * vocab ..][0..vocab];
            const dist = sampler.dist(row, ids.items);
            const t = dist.ids[0]; // greedy argmax (repetition penalty applied)
            var child: ?usize = null;
            for (node_tokens[1 .. 1 + m], node_parents[1 .. 1 + m], 1..) |dt, dp, i| {
                if (dp == cur and dt == t) {
                    child = i;
                    break;
                }
            }
            const c = child orelse {
                next_pending = t;
                break :walk;
            };
            if (opts.spec_stats) |s| s.accepted += 1;
            if (chat.isStop(t)) break :walk; // accepted stop: not emitted, not committed
            try ids.append(gpa, t);
            n += 1;
            try emit(tok, gpa, &stream, out, t);
            path[plen] = c;
            plen += 1;
            cur = c;
            if (n == opts.max_new_tokens) break :walk;
        }
        try model.commitTreePath(path[0..plen]);
        pending = next_pending orelse return n; // stop or budget hit mid-walk
    }
    return n;
}

fn validateTree(parents: []const u32, max_depth: usize) void {
    var depth: [max_tree_nodes]usize = undefined;
    depth[0] = 0;
    std.debug.assert(parents[0] == 0);
    for (parents[1..], 1..) |p, i| {
        std.debug.assert(p < i);
        depth[i] = depth[p] + 1;
        std.debug.assert(depth[i] <= max_depth);
    }
}

fn PtrChild(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.child,
        else => T,
    };
}

fn emit(tok: *const Tokenizer, gpa: std.mem.Allocator, stream: *engine.Utf8Stream, out: ?*std.Io.Writer, id: u32) !void {
    const w = out orelse return;
    const bytes = try tok.decodeAlloc(gpa, &.{id});
    defer gpa.free(bytes);
    try stream.write(w, bytes);
    try w.flush();
}

// --- tests -----------------------------------------------------------------

test "ngram drafter proposes the continuation of a repeated pattern" {
    const d: NgramDrafter = .{ .max_n = 3, .min_n = 1 };
    var buf: [4]u32 = undefined;

    // "5 6 7 8 ... 5 6 7" — trigram match, continuation is 8 then 9.
    const ids = [_]u32{ 5, 6, 7, 8, 9, 1, 5, 6, 7 };
    const m = d.propose(&ids, &buf);
    try std.testing.expectEqual(@as(usize, 4), m);
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 1, 5 }, buf[0..m]);

    // No earlier occurrence of any suffix: nothing proposed.
    const fresh = [_]u32{ 1, 2, 3, 4 };
    try std.testing.expectEqual(@as(usize, 0), d.propose(&fresh, &buf));

    // Most recent match wins: "1 2" appears twice with different followers.
    const twice = [_]u32{ 1, 2, 9, 1, 2, 4, 1, 2 };
    const m2 = d.propose(&twice, buf[0..1]);
    try std.testing.expectEqual(@as(usize, 1), m2);
    try std.testing.expectEqual(@as(u32, 4), buf[0]);
}

/// Deterministic stand-in for a backend stepper: the next token is a
/// function of the ENTIRE committed history (a rolling sum), so any
/// mishandled truncate/rollback poisons every subsequent prediction and
/// diverges the output. Greedy logits: point mass at rule(hist).
const ToyModel = struct {
    gpa: std.mem.Allocator,
    hist: std.ArrayList(u32) = .empty,
    capacity: usize,
    rule: *const fn (hist: []const u32) u32,
    /// Last verified tree's tokens (commitTreePath resolves node indices
    /// against it), mirroring the retained batch K/V of a real backend.
    tree_tokens: [max_tree_nodes]u32 = undefined,

    fn deinit(self: *ToyModel) void {
        self.hist.deinit(self.gpa);
    }

    pub fn cached(self: *const ToyModel) usize {
        return self.hist.items.len;
    }

    pub fn vocab(self: *const ToyModel) usize {
        _ = self;
        return qwen3.vocab_size;
    }

    pub fn remaining(self: *const ToyModel) usize {
        return self.capacity - self.hist.items.len;
    }

    pub fn truncate(self: *ToyModel, new_len: usize) void {
        std.debug.assert(new_len <= self.hist.items.len);
        self.hist.shrinkRetainingCapacity(new_len);
    }

    pub fn step(self: *ToyModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(logits.len == qwen3.vocab_size);
        try self.hist.appendSlice(self.gpa, ids_new);
        fillRow(logits, self.rule(self.hist.items));
    }

    pub fn stepAll(self: *ToyModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(logits.len == ids_new.len * qwen3.vocab_size);
        for (ids_new, 0..) |id, i| {
            try self.hist.append(self.gpa, id);
            fillRow(logits[i * qwen3.vocab_size ..][0..qwen3.vocab_size], self.rule(self.hist.items));
        }
    }

    /// Tree verify: node i's logits are the rule applied to the committed
    /// history plus the tokens along node i's root path. Nothing is
    /// committed — any acceptance-walk/commit bug leaves hist poisoned and
    /// diverges the output.
    pub fn stepAllTree(self: *ToyModel, io: std.Io, tokens: []const u32, parents: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(tokens.len == parents.len);
        std.debug.assert(logits.len == tokens.len * qwen3.vocab_size);
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(self.gpa);
        for (0..tokens.len) |i| {
            var chain: [max_tree_nodes]u32 = undefined;
            var d: usize = 0;
            var j = i;
            while (true) {
                chain[d] = tokens[j];
                d += 1;
                if (j == 0) break;
                j = parents[j];
            }
            ctx.clearRetainingCapacity();
            try ctx.appendSlice(self.gpa, self.hist.items);
            while (d > 0) {
                d -= 1;
                try ctx.append(self.gpa, chain[d]);
            }
            fillRow(logits[i * qwen3.vocab_size ..][0..qwen3.vocab_size], self.rule(ctx.items));
        }
        @memcpy(self.tree_tokens[0..tokens.len], tokens);
    }

    pub fn commitTreePath(self: *ToyModel, path: []const usize) !void {
        std.debug.assert(self.hist.items.len + path.len <= self.capacity);
        for (path) |idx| try self.hist.append(self.gpa, self.tree_tokens[idx]);
    }

    fn fillRow(row: []f32, target: u32) void {
        @memset(row, -100.0);
        row[target] = 0.0;
    }
};

/// Greedy reference: what vanilla decoding of `rule` emits.
fn ruleRollout(rule: *const fn ([]const u32) u32, gpa: std.mem.Allocator, prompt: []const u32, max_new: usize, out: *std.ArrayList(u32)) !void {
    try out.appendSlice(gpa, prompt);
    for (0..max_new) |_| {
        const next = rule(out.items);
        if (chat.isStop(next)) return;
        try out.append(gpa, next);
    }
}

fn sumRule(hist: []const u32) u32 {
    var s: u32 = 7;
    for (hist) |t| s = (s *% 31 +% t) % 97;
    return s;
}

fn periodicRule(hist: []const u32) u32 {
    return @intCast(hist.len % 7);
}

/// Always disagrees with sumRule on the same history (never a stop token).
fn offByOneRule(hist: []const u32) u32 {
    return (sumRule(hist) + 1) % 97;
}

fn stopAfter20Rule(hist: []const u32) u32 {
    if (hist.len >= 20) return tokenizer_mod.im_end;
    return @intCast(hist.len % 5);
}

const SpecCase = struct {
    rule: *const fn ([]const u32) u32,
    drafter_kind: enum { cheat, adversarial, ngram },
    max_new: usize = 24,
    spec_k: usize = 4,
};

/// A drafter that knows the toy rule (perfect proposals), or proposes
/// always-wrong tokens; both must leave output identical to vanilla greedy.
const TestDrafter = struct {
    rule: *const fn ([]const u32) u32,
    wrong: bool,
    scratch: std.ArrayList(u32) = .empty,
    gpa: std.mem.Allocator,

    fn deinit(self: *TestDrafter) void {
        self.scratch.deinit(self.gpa);
    }

    pub fn propose(self: *TestDrafter, ids: []const u32, buf: []u32) usize {
        self.scratch.clearRetainingCapacity();
        self.scratch.appendSlice(self.gpa, ids) catch return 0;
        for (buf) |*b| {
            var t = self.rule(self.scratch.items);
            if (self.wrong) t = (t + 1) % 90; // never a stop token, never right
            b.* = t;
            self.scratch.append(self.gpa, t) catch return 0;
        }
        return buf.len;
    }
};

fn runSpecCase(case: SpecCase) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const opts: engine.Options = .{
        .max_new_tokens = case.max_new,
        .sampling = .{ .temperature = 0 },
        .spec_k = case.spec_k,
    };

    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(gpa);
    try ruleRollout(case.rule, gpa, &prompt, case.max_new, &expected);

    var stats: Stats = .{};
    var opts_s = opts;
    opts_s.spec_stats = &stats;

    var model: ToyModel = .{ .gpa = gpa, .capacity = 128, .rule = case.rule };
    defer model.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ids.appendSlice(gpa, &prompt);

    const n = switch (case.drafter_kind) {
        .cheat, .adversarial => blk: {
            var drafter: TestDrafter = .{ .rule = case.rule, .wrong = case.drafter_kind == .adversarial, .gpa = gpa };
            defer drafter.deinit();
            break :blk try generate(&model, &drafter, undefined, io, gpa, &ids, opts_s, null);
        },
        .ngram => blk: {
            var drafter: NgramDrafter = .{};
            break :blk try generate(&model, &drafter, undefined, io, gpa, &ids, opts_s, null);
        },
    };

    try std.testing.expectEqualSlices(u32, expected.items, ids.items);
    try std.testing.expectEqual(expected.items.len - prompt.len, n);
    // The cache never holds tokens that are not in ids.
    try std.testing.expect(model.cached() <= ids.items.len);
    try std.testing.expectEqualSlices(u32, ids.items[0..model.cached()], model.hist.items);

    switch (case.drafter_kind) {
        .cheat => try std.testing.expectEqual(stats.drafted, stats.accepted),
        .adversarial => try std.testing.expectEqual(@as(usize, 0), stats.accepted),
        .ngram => {},
    }
}

test "spec greedy is byte-identical: perfect drafter accepts everything" {
    try runSpecCase(.{ .rule = sumRule, .drafter_kind = .cheat });
    // Fewer forwards than tokens: drafts actually amortized.
    // (24 tokens with k=4 needs ~5 verify forwards.)
}

test "spec greedy is byte-identical: adversarial drafter rejects everything" {
    try runSpecCase(.{ .rule = sumRule, .drafter_kind = .adversarial });
}

test "spec greedy is byte-identical: ngram drafter on periodic output" {
    try runSpecCase(.{ .rule = periodicRule, .drafter_kind = .ngram, .max_new = 30 });
}

test "spec greedy is byte-identical: drafted stop token" {
    try runSpecCase(.{ .rule = stopAfter20Rule, .drafter_kind = .cheat, .max_new = 40 });
    try runSpecCase(.{ .rule = stopAfter20Rule, .drafter_kind = .ngram, .max_new = 40 });
}

test "spec greedy is byte-identical: budget lands mid-acceptance" {
    // 24-token budget with k=4 and everything accepted: the budget check
    // fires inside the acceptance loop.
    try runSpecCase(.{ .rule = sumRule, .drafter_kind = .cheat, .max_new = 22 });
    try runSpecCase(.{ .rule = sumRule, .drafter_kind = .cheat, .max_new = 23 });
}

test "spec fills the context window exactly like vanilla" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    // Capacity 16 with a 5-token prompt: the window, not the budget, ends it.
    var model: ToyModel = .{ .gpa = gpa, .capacity = 16, .rule = sumRule };
    defer model.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ids.appendSlice(gpa, &prompt);
    var drafter: TestDrafter = .{ .rule = sumRule, .wrong = false, .gpa = gpa };
    defer drafter.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 100, .sampling = .{ .temperature = 0 }, .spec_k = 4 };
    const n = try generate(&model, &drafter, undefined, io, gpa, &ids, opts, null);

    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(gpa);
    // Vanilla stops once the cache is full: it caches every emitted token,
    // so it emits capacity - prompt + 1 tokens (the last stays uncached).
    try ruleRollout(sumRule, gpa, &prompt, 16 - prompt.len + 1, &expected);
    try std.testing.expectEqualSlices(u32, expected.items, ids.items);
    try std.testing.expectEqual(expected.items.len - prompt.len, n);
    try std.testing.expect(model.cached() <= 16);
}

const TreeDraftMode = enum { chain, branch, adversarial };

/// Tree drafters for the toy tests: a single-path chain of correct tokens,
/// a binary tree pairing the correct token with a wrong sibling at every
/// level (the walk must pick the right branch), or a chain of always-wrong
/// tokens. Output must be byte-identical to vanilla greedy in every case.
const TreeTestDrafter = struct {
    rule: *const fn ([]const u32) u32,
    mode: TreeDraftMode,
    gpa: std.mem.Allocator,
    scratch: std.ArrayList(u32) = .empty,

    fn deinit(self: *TreeTestDrafter) void {
        self.scratch.deinit(self.gpa);
    }

    pub fn proposeTree(self: *TreeTestDrafter, ids: []const u32, tokens: []u32, parents: []u32, max_depth: usize) usize {
        self.scratch.clearRetainingCapacity();
        self.scratch.appendSlice(self.gpa, ids) catch return 0;
        var m: usize = 0;
        var parent: u32 = 0; // node index of the previous on-path node
        var depth: usize = 0;
        while (depth < max_depth) : (depth += 1) {
            const t = self.rule(self.scratch.items);
            const wrong = (t + 1) % 90; // never right, never a stop token
            switch (self.mode) {
                .chain => {
                    if (m == tokens.len) break;
                    tokens[m] = t;
                    parents[m] = parent;
                    parent = @intCast(m + 1);
                    m += 1;
                },
                .branch => {
                    if (m + 2 > tokens.len) break;
                    tokens[m] = wrong; // wrong sibling first: the walk must skip it
                    parents[m] = parent;
                    tokens[m + 1] = t;
                    parents[m + 1] = parent;
                    parent = @intCast(m + 2);
                    m += 2;
                },
                .adversarial => {
                    if (m == tokens.len) break;
                    tokens[m] = wrong;
                    parents[m] = parent;
                    parent = @intCast(m + 1);
                    m += 1;
                },
            }
            self.scratch.append(self.gpa, t) catch return 0;
        }
        return m;
    }
};

const TreeCase = struct {
    rule: *const fn ([]const u32) u32,
    mode: TreeDraftMode,
    max_new: usize = 24,
    tree_nodes: usize = 9,
};

fn runTreeCase(case: TreeCase) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };

    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(gpa);
    try ruleRollout(case.rule, gpa, &prompt, case.max_new, &expected);

    var stats: Stats = .{};
    const opts: engine.Options = .{
        .max_new_tokens = case.max_new,
        .sampling = .{ .temperature = 0 },
        .tree_nodes = case.tree_nodes,
        .spec_stats = &stats,
    };

    var model: ToyModel = .{ .gpa = gpa, .capacity = 128, .rule = case.rule };
    defer model.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ids.appendSlice(gpa, &prompt);

    var drafter: TreeTestDrafter = .{ .rule = case.rule, .mode = case.mode, .gpa = gpa };
    defer drafter.deinit();
    const n = try generate(&model, &drafter, undefined, io, gpa, &ids, opts, null);

    try std.testing.expectEqualSlices(u32, expected.items, ids.items);
    try std.testing.expectEqual(expected.items.len - prompt.len, n);
    // The cache never holds tokens that are not in ids.
    try std.testing.expect(model.cached() <= ids.items.len);
    try std.testing.expectEqualSlices(u32, ids.items[0..model.cached()], model.hist.items);

    switch (case.mode) {
        .chain => if (case.rule == sumRule) try std.testing.expectEqual(stats.drafted, stats.accepted),
        .branch => if (case.rule == sumRule) try std.testing.expectEqual(stats.drafted, 2 * stats.accepted),
        .adversarial => try std.testing.expectEqual(@as(usize, 0), stats.accepted),
    }
}

test "tree greedy is byte-identical: chain-shaped tree accepts everything" {
    try runTreeCase(.{ .rule = sumRule, .mode = .chain });
}

test "tree greedy is byte-identical: the walk picks the right branch" {
    try runTreeCase(.{ .rule = sumRule, .mode = .branch });
    try runTreeCase(.{ .rule = periodicRule, .mode = .branch, .max_new = 30 });
}

test "tree greedy is byte-identical: adversarial tree rejects everything" {
    try runTreeCase(.{ .rule = sumRule, .mode = .adversarial });
}

test "tree greedy is byte-identical: drafted stop token" {
    try runTreeCase(.{ .rule = stopAfter20Rule, .mode = .chain, .max_new = 40 });
    try runTreeCase(.{ .rule = stopAfter20Rule, .mode = .branch, .max_new = 40 });
}

test "tree greedy is byte-identical: budget lands mid-walk" {
    try runTreeCase(.{ .rule = sumRule, .mode = .chain, .max_new = 22 });
    try runTreeCase(.{ .rule = sumRule, .mode = .branch, .max_new = 23 });
}

test "tree spec fills the context window exactly like vanilla" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    var model: ToyModel = .{ .gpa = gpa, .capacity = 16, .rule = sumRule };
    defer model.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ids.appendSlice(gpa, &prompt);
    var drafter: TreeTestDrafter = .{ .rule = sumRule, .mode = .branch, .gpa = gpa };
    defer drafter.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 100, .sampling = .{ .temperature = 0 }, .tree_nodes = 9 };
    const n = try generate(&model, &drafter, undefined, io, gpa, &ids, opts, null);

    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(gpa);
    try ruleRollout(sumRule, gpa, &prompt, 16 - prompt.len + 1, &expected);
    try std.testing.expectEqualSlices(u32, expected.items, ids.items);
    try std.testing.expectEqual(expected.items.len - prompt.len, n);
    try std.testing.expect(model.cached() <= 16);
}

test "tree dispatch rejects unsupported drafters and sampling" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var model: ToyModel = .{ .gpa = gpa, .capacity = 16, .rule = sumRule };
    defer model.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try ids.appendSlice(gpa, &.{ 3, 1, 4 });

    // A chain-only drafter cannot serve tree mode.
    var ngram: NgramDrafter = .{};
    const opts: engine.Options = .{ .sampling = .{ .temperature = 0 }, .tree_nodes = 8 };
    try std.testing.expectError(error.TreeUnsupported, generate(&model, &ngram, undefined, io, gpa, &ids, opts, null));

    // v1 trees are greedy-only.
    var drafter: TreeTestDrafter = .{ .rule = sumRule, .mode = .chain, .gpa = gpa };
    defer drafter.deinit();
    var opts_t = opts;
    opts_t.sampling = .{ .temperature = 1.0 };
    try std.testing.expectError(error.TreeUnsupported, generate(&model, &drafter, undefined, io, gpa, &ids, opts_t, null));
}

// Sampled speculative decoding must draw from the same distribution as
// vanilla sampling. The toy rule's logits are a point mass, which would make
// this trivial, so use a model with a genuinely spread distribution.
const SpreadToy = struct {
    gpa: std.mem.Allocator,
    hist: std.ArrayList(u32) = .empty,
    capacity: usize,

    fn deinit(self: *SpreadToy) void {
        self.hist.deinit(self.gpa);
    }
    pub fn cached(self: *const SpreadToy) usize {
        return self.hist.items.len;
    }
    pub fn vocab(self: *const SpreadToy) usize {
        _ = self;
        return qwen3.vocab_size;
    }

    pub fn remaining(self: *const SpreadToy) usize {
        return self.capacity - self.hist.items.len;
    }
    pub fn truncate(self: *SpreadToy, new_len: usize) void {
        self.hist.shrinkRetainingCapacity(new_len);
    }
    pub fn step(self: *SpreadToy, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        try self.hist.appendSlice(self.gpa, ids_new);
        fillSpread(logits, self.hist.items);
    }
    pub fn stepAll(self: *SpreadToy, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        for (ids_new, 0..) |id, i| {
            try self.hist.append(self.gpa, id);
            fillSpread(logits[i * qwen3.vocab_size ..][0..qwen3.vocab_size], self.hist.items);
        }
    }
    /// Four live candidates whose order depends on the history sum.
    fn fillSpread(row: []f32, hist: []const u32) void {
        @memset(row, -100.0);
        var s: u32 = 0;
        for (hist) |t| s +%= t;
        for (0..4) |c| row[(s + c) % 8] = 1.5 - 0.5 * @as(f32, @floatFromInt(c));
    }
};

test "spec sampling matches the vanilla distribution" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 2, 3 };
    const trials = 400;
    const steps = 4;
    // Token histogram over all emitted positions, vanilla vs spec.
    var hist_vanilla = [_]usize{0} ** 8;
    var hist_spec = [_]usize{0} ** 8;

    for (0..trials) |seed| {
        const opts: engine.Options = .{
            .max_new_tokens = steps,
            .sampling = .{ .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
            .seed = seed,
        };
        {
            var model: SpreadToy = .{ .gpa = gpa, .capacity = 32 };
            defer model.deinit();
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(gpa);
            try ids.appendSlice(gpa, &prompt);
            _ = try engine.generate(&model, undefined, io, gpa, &ids, opts, null);
            for (ids.items[prompt.len..]) |t| hist_vanilla[t] += 1;
        }
        {
            var opts_s = opts;
            opts_s.spec_k = 2;
            opts_s.seed = seed + 1_000_000; // spec consumes rng differently; only distributions match
            var model: SpreadToy = .{ .gpa = gpa, .capacity = 32 };
            defer model.deinit();
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(gpa);
            try ids.appendSlice(gpa, &prompt);
            var drafter: NgramDrafter = .{};
            _ = try generate(&model, &drafter, undefined, io, gpa, &ids, opts_s, null);
            for (ids.items[prompt.len..]) |t| hist_spec[t] += 1;
        }
    }

    const total = trials * steps;
    for (hist_vanilla, hist_spec) |a, b| {
        const fa = @as(f64, @floatFromInt(a)) / total;
        const fb = @as(f64, @floatFromInt(b)) / total;
        try std.testing.expectApproxEqAbs(fa, fb, 0.05);
    }
}

// ModelDrafter driving spec.generate, with ToyModels as both target and
// draft: a same-rule draft accepts everything, a divergent-rule draft
// rejects (exercising the cache re-sync: truncate to common prefix +
// re-prefill). Output must equal vanilla greedy either way.
test "model drafter is byte-identical and re-syncs after rejections" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const opts: engine.Options = .{ .max_new_tokens = 24, .sampling = .{ .temperature = 0 }, .spec_k = 4 };

    var expected: std.ArrayList(u32) = .empty;
    defer expected.deinit(gpa);
    try ruleRollout(sumRule, gpa, &prompt, opts.max_new_tokens, &expected);

    inline for (.{ sumRule, offByOneRule }) |draft_rule| {
        var stats: Stats = .{};
        var opts_s = opts;
        opts_s.spec_stats = &stats;

        var target: ToyModel = .{ .gpa = gpa, .capacity = 128, .rule = sumRule };
        defer target.deinit();
        var draft: ToyModel = .{ .gpa = gpa, .capacity = 128, .rule = draft_rule };
        defer draft.deinit();
        var drafter = try ModelDrafter(ToyModel).init(gpa, io, &draft);
        defer drafter.deinit();

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ids.appendSlice(gpa, &prompt);
        _ = try generate(&target, &drafter, undefined, io, gpa, &ids, opts_s, null);

        try std.testing.expectEqualSlices(u32, expected.items, ids.items);
        // The draft's cache mirror never desyncs from its stepper.
        try std.testing.expectEqual(drafter.hist.items.len, draft.cached());
        try std.testing.expectEqualSlices(u32, drafter.hist.items, draft.hist.items);
        if (draft_rule == sumRule) {
            try std.testing.expectEqual(stats.drafted, stats.accepted); // perfect draft
        } else {
            try std.testing.expectEqual(@as(usize, 0), stats.accepted); // rules differ everywhere
        }
    }
}

// Gated on the real checkpoint: spec-k greedy output must equal vanilla
// greedy output through the CpuModel path. Kept tiny — every token is a full
// 36-layer forward in Debug.
test "spec matches vanilla greedy on the real model" {
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

    var opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };

    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);

    var ids_spec: std.ArrayList(u32) = .empty;
    defer ids_spec.deinit(gpa);
    try ids_spec.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.spec_k = 2;
        var stats: Stats = .{};
        opts.spec_stats = &stats;
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_spec.items.len)));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_spec, opts, null);
        try std.testing.expect(stats.forwards > 0);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_spec.items);
}

// Gated on the real checkpoint: tree-verify greedy output through the
// CpuModel path (forwardTree + attentionTree + commitTreePath) must equal
// vanilla greedy. The n-gram chain rides ChainAsTree so real drafts (and
// real rejections) flow through the tree machinery. Kept tiny — every
// forward is a full 36-layer pass in Debug.
test "tree spec matches vanilla greedy on the real model" {
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

    var opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };

    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_tree: std.ArrayList(u32) = .empty;
    defer ids_tree.deinit(gpa);
    try ids_tree.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.tree_nodes = 4;
        var stats: Stats = .{};
        opts.spec_stats = &stats;
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_tree.items.len)));
        defer model.deinit();
        var ngram: NgramDrafter = .{};
        var drafter: ChainAsTree(NgramDrafter) = .{ .inner = &ngram };
        _ = try generate(&model, &drafter, &tok, io, gpa, &ids_tree, opts, null);
        try std.testing.expect(stats.forwards > 0);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_tree.items);
}

// Gated on both checkpoints: the real 4B target with the real 0.6B draft
// model on the CPU stepper, greedy, byte-identical to vanilla. Kept tiny —
// Debug forwards are slow.
test "model drafter matches vanilla greedy on the real models" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    const draft_path = "models/text_encoders/qwen_3_06b_base.safetensors";
    try test_gate.requireModelFile(io, te_path);
    try test_gate.requireModelFile(io, draft_path);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var dst = try safetensors.SafeTensors.open(gpa, io, draft_path);
    defer dst.deinit();
    var dlm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &dst });
    defer dlm.deinit();
    try std.testing.expectEqual(@as(usize, 28), dlm.cfg.n_layers);
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };

    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Say hi.", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_spec: std.ArrayList(u32) = .empty;
    defer ids_spec.deinit(gpa);
    try ids_spec.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.spec_k = 2;
        var model = try engine.CpuModel.init(gpa, &lm, .fixed(try engine.capacityFor(opts, ids_spec.items.len)));
        defer model.deinit();
        var draft = try engine.CpuModel.init(gpa, &dlm, .fixed(try engine.capacityFor(opts, ids_spec.items.len)));
        defer draft.deinit();
        var drafter = try ModelDrafter(engine.CpuModel).init(gpa, io, &draft);
        defer drafter.deinit();
        _ = try generate(&model, &drafter, &tok, io, gpa, &ids_spec, opts, null);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_spec.items);
}
