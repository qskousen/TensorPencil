//! K2 Horizon dense, MoE, and MoVA language models.

const std = @import("std");
const Gguf = @import("tp_core").gguf.Gguf;
const kvmod = @import("tp_core").kv_cache;
const Tokenizer = @import("tp_core").tokenizer.Tokenizer;
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");

const Weight = ops.matmul.Weight;

pub const Config = struct {
    n_layers: usize,
    context: usize,
    hidden: usize,
    intermediate: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rms_eps: f32,
    norm_groups: usize,
    n_experts: usize,
    n_experts_used: usize,
    expert_intermediate: usize,
    dense_layers: usize,
    shared_intermediate: usize,
    normalize_weights: bool,
    expert_scale: f32,
    gating: Gating,
    n_value_experts: usize,
    n_value_experts_used: usize,
    vocab: usize,

    pub const Gating = enum { softmax, sigmoid };

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }

    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }

    pub fn load(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "k2-horizon")) return error.UnknownModelConfig;
        const embed = g.get("embed_tokens.weight") orelse return error.MissingTensor;
        const shape = embed.info.shape.slice();
        if (shape.len != 2) return error.ShapeMismatch;
        const gating_raw = g.getUint("k2-horizon.expert_gating_func") orelse 2;
        const gating: Gating = switch (gating_raw) {
            1 => .softmax,
            2 => .sigmoid,
            else => return error.UnsupportedModelConfig,
        };
        return .{
            .n_layers = getUsize(g, "k2-horizon.block_count") orelse return error.UnknownModelConfig,
            .context = getUsize(g, "k2-horizon.context_length") orelse return error.UnknownModelConfig,
            .hidden = getUsize(g, "k2-horizon.embedding_length") orelse return error.UnknownModelConfig,
            .intermediate = getUsize(g, "k2-horizon.feed_forward_length") orelse return error.UnknownModelConfig,
            .n_heads = getUsize(g, "k2-horizon.attention.head_count") orelse return error.UnknownModelConfig,
            .n_kv_heads = getUsize(g, "k2-horizon.attention.head_count_kv") orelse return error.UnknownModelConfig,
            .head_dim = getUsize(g, "k2-horizon.attention.key_length") orelse return error.UnknownModelConfig,
            .rope_theta = g.getFloat("k2-horizon.rope.freq_base") orelse 10_000_000,
            .rms_eps = @floatCast(g.getFloat("k2-horizon.attention.layer_norm_rms_epsilon") orelse 1e-6),
            .norm_groups = getUsize(g, "k2-horizon.attention.group_norm_groups") orelse 1,
            .n_experts = getUsize(g, "k2-horizon.expert_count") orelse 0,
            .n_experts_used = getUsize(g, "k2-horizon.expert_used_count") orelse 0,
            .expert_intermediate = getUsize(g, "k2-horizon.expert_feed_forward_length") orelse 0,
            .dense_layers = getUsize(g, "k2-horizon.leading_dense_block_count") orelse 0,
            .shared_intermediate = getUsize(g, "k2-horizon.expert_shared_feed_forward_length") orelse 0,
            .normalize_weights = g.getBool("k2-horizon.expert_weights_norm") orelse false,
            .expert_scale = @floatCast(g.getFloat("k2-horizon.expert_weights_scale") orelse 1),
            .gating = gating,
            .n_value_experts = getUsize(g, "k2-horizon.attention.value_expert_count") orelse 0,
            .n_value_experts_used = getUsize(g, "k2-horizon.attention.value_expert_used_count") orelse 0,
            .vocab = shape[0],
        };
    }
};

fn getUsize(g: *const Gguf, key: []const u8) ?usize {
    return if (g.getUint(key)) |v| @intCast(v) else null;
}

pub const Experts = struct {
    gate: []Weight,
    up: []Weight,
    down: []Weight,
    router: Weight,
    bias: []const f32,
    shared_gate: ?Weight,
    shared_up: ?Weight,
    shared_down: ?Weight,
};

pub const Values = struct {
    weights: []Weight,
    router: Weight,
    bias: []const f32,
};

pub const Layer = struct {
    attn_norm: []const f32,
    q: Weight,
    k: Weight,
    v: ?Weight,
    values: ?Values,
    attn_gate: ?Weight,
    o: Weight,
    ffn_norm: []const f32,
    dense_gate: ?Weight,
    dense_up: ?Weight,
    dense_down: ?Weight,
    experts: ?Experts,
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    embed: Weight,
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Model {
        const cfg = try Config.load(g);
        if (cfg.hidden % cfg.norm_groups != 0 or cfg.head_dim == 0 or cfg.qDim() == 0 or cfg.kvDim() == 0)
            return error.UnsupportedModelConfig;
        if (cfg.n_experts > 0 and (cfg.n_experts_used == 0 or cfg.expert_intermediate == 0))
            return error.UnsupportedModelConfig;
        if (cfg.n_value_experts > 0 and cfg.n_value_experts_used == 0)
            return error.UnsupportedModelConfig;

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const embed = try loadWeight(g, "embed_tokens.weight", cfg.vocab, cfg.hidden);
        const head = try loadWeight(g, "lm_head.weight", cfg.vocab, cfg.hidden);
        const final_norm = try loadF32(a, g, "norm.weight", cfg.hidden);
        const layers = try a.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| layer.* = try loadLayer(a, g, cfg, l);
        return .{ .arena = arena, .cfg = cfg, .embed = embed, .head = head, .layers = layers, .final_norm = final_norm };
    }


    /// Every linear the device forwards run, experts included, tagged for a backend's
    /// route check at load.
    pub fn deviceLins(self: *const Model, alloc: std.mem.Allocator) ![]Weight {
        var lins: std.ArrayList(Weight) = .empty;
        const T = struct {
            fn add(list: *std.ArrayList(Weight), a: std.mem.Allocator, w: ?Weight, comptime fmt: []const u8, args: anytype) !void {
                var t = w orelse return;
                t.tag = try a.print(fmt, args);
                try list.append(a, t);
            }
        };
        for (self.layers, 0..) |layer, l| {
            try T.add(&lins, alloc, layer.q, "layers.{d}.q", .{l});
            try T.add(&lins, alloc, layer.k, "layers.{d}.k", .{l});
            try T.add(&lins, alloc, layer.v, "layers.{d}.v", .{l});
            try T.add(&lins, alloc, layer.attn_gate, "layers.{d}.attn_gate", .{l});
            try T.add(&lins, alloc, layer.o, "layers.{d}.o", .{l});
            try T.add(&lins, alloc, layer.dense_gate, "layers.{d}.dense_gate", .{l});
            try T.add(&lins, alloc, layer.dense_up, "layers.{d}.dense_up", .{l});
            try T.add(&lins, alloc, layer.dense_down, "layers.{d}.dense_down", .{l});
            if (layer.values) |vals| {
                try T.add(&lins, alloc, vals.router, "layers.{d}.values.router", .{l});
                for (vals.weights, 0..) |w, e| try T.add(&lins, alloc, w, "layers.{d}.values.{d}", .{ l, e });
            }
            if (layer.experts) |ex| {
                try T.add(&lins, alloc, ex.router, "layers.{d}.experts.router", .{l});
                try T.add(&lins, alloc, ex.shared_gate, "layers.{d}.experts.shared_gate", .{l});
                try T.add(&lins, alloc, ex.shared_up, "layers.{d}.experts.shared_up", .{l});
                try T.add(&lins, alloc, ex.shared_down, "layers.{d}.experts.shared_down", .{l});
                for (ex.gate, ex.up, ex.down, 0..) |g, u, d, e| {
                    try T.add(&lins, alloc, g, "layers.{d}.experts.{d}.gate", .{ l, e });
                    try T.add(&lins, alloc, u, "layers.{d}.experts.{d}.up", .{ l, e });
                    try T.add(&lins, alloc, d, "layers.{d}.experts.{d}.down", .{ l, e });
                }
            }
        }
        try T.add(&lins, alloc, self.head, "lm_head", .{});
        return lins.toOwnedSlice(alloc);
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn forwardCached(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        cache: *kvmod.KvCache,
        freqs: ops.rope.Freqs,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        if (seq == 0 or seq > cache.remaining()) return error.ContextFull;
        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.embed, ids, x);
        var scratch = try Scratch.init(gpa, cfg, seq);
        defer scratch.deinit(gpa);
        const pos0 = cache.len;
        for (self.layers, 0..) |layer, l| try layerForward(io, gpa, cfg, layer, x, seq, cache, freqs, l, pos0, &scratch);
        cache.commit(seq);
        groupRmsNorm(out, x[(seq - 1) * cfg.hidden ..][0..cfg.hidden], self.final_norm, cfg.norm_groups, cfg.rms_eps);
    }

    pub fn forwardLayer(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        x: []f32,
        seq: usize,
        cache: *kvmod.KvCache,
        freqs: ops.rope.Freqs,
        layer: usize,
        pos0: usize,
        scratch: *Scratch,
    ) !void {
        try layerForward(io, gpa, self.cfg, self.layers[layer], x, seq, cache, freqs, layer, pos0, scratch);
    }
};

fn loadLayer(a: std.mem.Allocator, g: *const Gguf, cfg: Config, l: usize) !Layer {
    var buf: [96]u8 = undefined;
    const name = struct {
        fn f(b: []u8, layer: usize, suffix: []const u8) ![]const u8 {
            return std.mem.print(b, "layers.{d}.{s}", .{ layer, suffix });
        }
    }.f;
    const sparse = cfg.n_experts > 0 and l >= cfg.dense_layers;
    const mova = sparse and cfg.n_value_experts > 0;
    const attn_norm = try loadF32(a, g, try name(&buf, l, "input_layernorm.weight"), cfg.hidden);
    const q = try loadWeight(g, try name(&buf, l, "self_attn.q_proj.weight"), cfg.qDim(), cfg.hidden);
    const k = try loadWeight(g, try name(&buf, l, "self_attn.k_proj.weight"), cfg.kvDim(), cfg.hidden);
    const v = if (!mova) try loadWeight(g, try name(&buf, l, "self_attn.v_proj.weight"), cfg.kvDim(), cfg.hidden) else null;
    const values: ?Values = if (mova) .{
        .weights = try loadExpertWeights(a, g, try name(&buf, l, "attn_v_exps.weight"), cfg.n_value_experts, cfg.kvDim(), cfg.hidden),
        .router = try loadRouter(a, g, try name(&buf, l, "attn_v_gate.weight"), cfg.n_value_experts, cfg.hidden),
        .bias = try loadF32(a, g, try name(&buf, l, "attn_v_gate.bias"), cfg.n_value_experts),
    } else null;
    const attn_gate = if (g.get(try name(&buf, l, "attn_gate.weight")) != null)
        try loadWeight(g, try name(&buf, l, "attn_gate.weight"), cfg.qDim(), cfg.hidden)
    else
        null;
    const o = try loadWeight(g, try name(&buf, l, "self_attn.o_proj.weight"), cfg.hidden, cfg.qDim());
    const ffn_norm = try loadF32(a, g, try name(&buf, l, "post_attention_layernorm.weight"), cfg.hidden);

    if (!sparse) return .{
        .attn_norm = attn_norm,
        .q = q,
        .k = k,
        .v = v,
        .values = values,
        .attn_gate = attn_gate,
        .o = o,
        .ffn_norm = ffn_norm,
        .dense_gate = try loadWeight(g, try name(&buf, l, "mlp.gate_proj.weight"), cfg.intermediate, cfg.hidden),
        .dense_up = try loadWeight(g, try name(&buf, l, "mlp.up_proj.weight"), cfg.intermediate, cfg.hidden),
        .dense_down = try loadWeight(g, try name(&buf, l, "mlp.down_proj.weight"), cfg.hidden, cfg.intermediate),
        .experts = null,
    };

    const shared = cfg.shared_intermediate > 0;
    return .{
        .attn_norm = attn_norm,
        .q = q,
        .k = k,
        .v = v,
        .values = values,
        .attn_gate = attn_gate,
        .o = o,
        .ffn_norm = ffn_norm,
        .dense_gate = null,
        .dense_up = null,
        .dense_down = null,
        .experts = .{
            .gate = try loadExpertWeights(a, g, try name(&buf, l, "ffn_gate_exps.weight"), cfg.n_experts, cfg.expert_intermediate, cfg.hidden),
            .up = try loadExpertWeights(a, g, try name(&buf, l, "ffn_up_exps.weight"), cfg.n_experts, cfg.expert_intermediate, cfg.hidden),
            .down = try loadExpertWeights(a, g, try name(&buf, l, "ffn_down_exps.weight"), cfg.n_experts, cfg.hidden, cfg.expert_intermediate),
            .router = try loadRouter(a, g, try name(&buf, l, "ffn_gate_inp.weight"), cfg.n_experts, cfg.hidden),
            .bias = try loadF32(a, g, try name(&buf, l, "exp_probs_b.bias"), cfg.n_experts),
            .shared_gate = if (shared) try loadWeight(g, try name(&buf, l, "ffn_gate_shexp.weight"), cfg.shared_intermediate, cfg.hidden) else null,
            .shared_up = if (shared) try loadWeight(g, try name(&buf, l, "ffn_up_shexp.weight"), cfg.shared_intermediate, cfg.hidden) else null,
            .shared_down = if (shared) try loadWeight(g, try name(&buf, l, "ffn_down_shexp.weight"), cfg.hidden, cfg.shared_intermediate) else null,
        },
    };
}

/// Routers are tiny and their GEMM runs at every batch width; a quantized one
/// would take the 8-token GEMV path, so keep them f32 whatever the file holds.
fn loadRouter(a: std.mem.Allocator, g: *const Gguf, name: []const u8, rows: usize, cols: usize) !Weight {
    const w = try loadWeight(g, name, rows, cols);
    if (w.dtype == .f32) return w;
    const view = g.get(name) orelse return error.MissingTensor;
    const f = try view.toF32Alloc(a);
    return Weight.init(std.mem.sliceAsBytes(f), .f32, rows, cols);
}

fn loadWeight(g: *const Gguf, name: []const u8, rows: usize, cols: usize) !Weight {
    const view = g.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

fn loadExpertWeights(a: std.mem.Allocator, g: *const Gguf, name: []const u8, count: usize, rows: usize, cols: usize) ![]Weight {
    const view = g.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 3 or shape[0] != count or shape[1] != rows or shape[2] != cols) return error.ShapeMismatch;
    const bytes_per = view.info.dtype.storageBytes(rows * cols);
    if (view.bytes.len != count * bytes_per) return error.ShapeMismatch;
    const out = try a.alloc(Weight, count);
    for (out, 0..) |*w, i| {
        w.* = Weight.init(view.bytes[i * bytes_per ..][0..bytes_per], view.info.dtype, rows, cols);
    }
    return out;
}

fn loadF32(a: std.mem.Allocator, g: *const Gguf, name: []const u8, len: usize) ![]const f32 {
    const view = g.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return try view.toF32Alloc(a);
}

pub const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    attn_gate: []f32,
    route_logits: []f32,
    selected: []usize,
    selected_weights: []f32,
    token_ids: []usize,
    token_weights: []f32,
    expert_in: []f32,
    expert_a: []f32,
    expert_b: []f32,
    expert_out: []f32,
    expert_counts: []usize,
    expert_starts: []usize,
    /// One row per (token, selected expert) slot, packed by expert (`packSlots`).
    slot_row: []usize,
    slot_in: []f32,
    slot_a: []f32,
    slot_b: []f32,
    slot_out: []f32,
    jobs: []ops.matmul.BatchGemv,

    pub fn init(gpa: std.mem.Allocator, cfg: Config, seq: usize) !Scratch {
        const max_experts: usize = @max(cfg.n_experts, cfg.n_value_experts);
        const max_used: usize = @max(cfg.n_experts_used, cfg.n_value_experts_used);
        const max_inner: usize = @max(@max(cfg.intermediate, cfg.expert_intermediate), @max(cfg.shared_intermediate, cfg.kvDim()));
        const max_out: usize = @max(cfg.hidden, cfg.kvDim());
        return .{
            .normed = try gpa.alloc(f32, seq * cfg.hidden),
            .tmp = try gpa.alloc(f32, seq * cfg.hidden),
            .q = try gpa.alloc(f32, seq * cfg.qDim()),
            .k = try gpa.alloc(f32, seq * cfg.kvDim()),
            .v = try gpa.alloc(f32, seq * cfg.kvDim()),
            .attn = try gpa.alloc(f32, seq * cfg.qDim()),
            .attn_gate = try gpa.alloc(f32, seq * cfg.qDim()),
            .route_logits = try gpa.alloc(f32, seq * max_experts),
            .selected = try gpa.alloc(usize, seq * max_used),
            .selected_weights = try gpa.alloc(f32, seq * max_used),
            .token_ids = try gpa.alloc(usize, seq),
            .token_weights = try gpa.alloc(f32, seq),
            .expert_in = try gpa.alloc(f32, seq * cfg.hidden),
            .expert_a = try gpa.alloc(f32, @max(seq * max_inner, max_used * @max(cfg.expert_intermediate, cfg.kvDim()))),
            .expert_b = try gpa.alloc(f32, @max(seq * max_inner, max_used * @max(cfg.expert_intermediate, cfg.kvDim()))),
            .expert_out = try gpa.alloc(f32, @max(seq * max_out, max_used * max_out)),
            .expert_counts = try gpa.alloc(usize, max_experts),
            .expert_starts = try gpa.alloc(usize, max_experts),
            .slot_row = try gpa.alloc(usize, seq * max_used),
            .slot_in = try gpa.alloc(f32, seq * max_used * cfg.hidden),
            .slot_a = try gpa.alloc(f32, seq * max_used * @max(cfg.expert_intermediate, cfg.kvDim())),
            .slot_b = try gpa.alloc(f32, seq * max_used * cfg.expert_intermediate),
            .slot_out = try gpa.alloc(f32, seq * max_used * cfg.hidden),
            .jobs = try gpa.alloc(ops.matmul.BatchGemv, 2 * max_experts),
        };
    }

    pub fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        inline for (.{ "normed", "tmp", "q", "k", "v", "attn", "attn_gate", "route_logits", "selected", "selected_weights", "token_ids", "token_weights", "expert_in", "expert_a", "expert_b", "expert_out", "expert_counts", "expert_starts", "slot_row", "slot_in", "slot_a", "slot_b", "slot_out", "jobs" }) |field|
            gpa.free(@field(self, field));
        self.* = undefined;
    }
};

fn groupRmsNorm(out: []f32, x: []const f32, weight: []const f32, groups: usize, eps: f32) void {
    const hidden = weight.len;
    const group = hidden / groups;
    const rows = x.len / hidden;
    for (0..rows) |r| for (0..groups) |g| {
        const off = r * hidden + g * group;
        ops.norm.rmsNorm(out[off..][0..group], x[off..][0..group], weight[g * group ..][0..group], eps);
    };
}

fn layerForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: Layer,
    x: []f32,
    seq: usize,
    cache: *kvmod.KvCache,
    freqs: ops.rope.Freqs,
    l: usize,
    pos0: usize,
    s: *Scratch,
) !void {
    const normed = s.normed[0 .. seq * cfg.hidden];
    const q = s.q[0 .. seq * cfg.qDim()];
    const k = s.k[0 .. seq * cfg.kvDim()];
    const v = s.v[0 .. seq * cfg.kvDim()];
    groupRmsNorm(normed, x, layer.attn_norm, cfg.norm_groups, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, q, normed, seq, layer.q, null);
    try ops.matmul.matmul(io, gpa, k, normed, seq, layer.k, null);
    if (layer.values) |values|
        try routedValues(io, gpa, cfg, values, normed, seq, v, s)
    else
        try ops.matmul.matmul(io, gpa, v, normed, seq, layer.v.?, null);
    ops.rope.applyRotateHalfAt(q, freqs, pos0, seq, cfg.n_heads, cfg.head_dim);
    ops.rope.applyRotateHalfAt(k, freqs, pos0, seq, cfg.n_kv_heads, cfg.head_dim);
    cache.write(l, k, v);
    const attn = s.attn[0 .. seq * cfg.qDim()];
    try ops.attention.attention(io, gpa, attn, q, cache.kView(l, seq), cache.vView(l, seq), .{
        .seq_q = seq,
        .seq_kv = pos0 + seq,
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
        .causal = true,
    });
    if (layer.attn_gate) |gate_w| {
        const gate = s.attn_gate[0 .. seq * cfg.qDim()];
        try ops.matmul.matmul(io, gpa, gate, normed, seq, gate_w, null);
        const ln2: f32 = 0.6931471805599453;
        for (attn, gate) |*a, gv| {
            const z = gv * ln2;
            const sp = if (z > 0) z + @log(1 + @exp(-z)) else @log(1 + @exp(z));
            a.* *= sp / ln2;
        }
    }
    const tmp = s.tmp[0 .. seq * cfg.hidden];
    try ops.matmul.matmul(io, gpa, tmp, attn, seq, layer.o, null);
    for (x, tmp) |*dst, add| dst.* += add;

    groupRmsNorm(normed, x, layer.ffn_norm, cfg.norm_groups, cfg.rms_eps);
    if (layer.experts) |experts| {
        try routedFfn(io, gpa, cfg, experts, normed, seq, tmp, s);
    } else {
        const gate = s.expert_a[0 .. seq * cfg.intermediate];
        const up = s.expert_b[0 .. seq * cfg.intermediate];
        try ops.matmul.matmul(io, gpa, gate, normed, seq, layer.dense_gate.?, null);
        try ops.matmul.matmul(io, gpa, up, normed, seq, layer.dense_up.?, null);
        ops.act.siluMul(gate, up);
        try ops.matmul.matmul(io, gpa, tmp, gate, seq, layer.dense_down.?, null);
    }
    for (x, tmp) |*dst, add| dst.* += add;
}

/// Gate `logits` (per row of `n_experts`) in place and pick each row's `n_used` best by
/// score plus `bias`, weighted as the config says.
pub fn route(
    cfg: Config,
    logits: []f32,
    bias: []const f32,
    seq: usize,
    n_experts: usize,
    n_used: usize,
    selected: []usize,
    weights: []f32,
) void {
    for (0..seq) |t| {
        const row = logits[t * n_experts ..][0..n_experts];
        switch (cfg.gating) {
            .sigmoid => ops.act.sigmoid(row),
            .softmax => {
                var mx = -std.math.inf(f32);
                for (row) |v| mx = @max(mx, v);
                var sum: f32 = 0;
                for (row) |*v| {
                    v.* = @exp(v.* - mx);
                    sum += v.*;
                }
                for (row) |*v| v.* /= sum;
            },
        }
        const si = selected[t * n_used ..][0..n_used];
        const sw = weights[t * n_used ..][0..n_used];
        @memset(sw, -std.math.inf(f32));
        for (row, 0..) |p, e| {
            const score = p + bias[e];
            var at: usize = 0;
            while (at < n_used and score <= sw[at]) : (at += 1) {}
            if (at == n_used) continue;
            var j = n_used - 1;
            while (j > at) : (j -= 1) {
                sw[j] = sw[j - 1];
                si[j] = si[j - 1];
            }
            sw[at] = score;
            si[at] = e;
        }
        var sum: f32 = 0;
        for (si, sw) |e, *w| {
            w.* = row[e];
            sum += w.*;
        }
        if (cfg.normalize_weights) {
            for (sw) |*w| w.* /= @max(sum, 6.103515625e-5);
        }
        if (cfg.expert_scale != 1) {
            for (sw) |*w| w.* *= cfg.expert_scale;
        }
    }
}

pub fn routedValues(io: std.Io, gpa: std.mem.Allocator, cfg: Config, values: Values, x: []const f32, seq: usize, out: []f32, s: *Scratch) !void {
    const logits = s.route_logits[0 .. seq * cfg.n_value_experts];
    try ops.matmul.matmul(io, gpa, logits, x, seq, values.router, null);
    const selected = s.selected[0 .. seq * cfg.n_value_experts_used];
    const weights = s.selected_weights[0 .. seq * cfg.n_value_experts_used];
    route(cfg, logits, values.bias, seq, cfg.n_value_experts, cfg.n_value_experts_used, selected, weights);
    try routedValuesSelected(io, gpa, cfg, values, x, seq, out, s, selected, weights);
}

pub fn routedValuesSelected(io: std.Io, gpa: std.mem.Allocator, cfg: Config, values: Values, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    if (routedValuesSlots(io, gpa, cfg, values, x, seq, out, s, selected, weights)) |_| return else |err| switch (err) {
        error.QuantBackendUnavailable => {},
        else => return err,
    }
    try routedValuesPerExpert(io, gpa, cfg, values, x, seq, out, s, selected, weights);
}

fn routedValuesPerExpert(io: std.Io, gpa: std.mem.Allocator, cfg: Config, values: Values, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    @memset(out, 0);
    for (values.weights, 0..) |w, e| {
        const n = gatherExpert(cfg.hidden, x, seq, selected, weights, cfg.n_value_experts_used, e, s);
        if (n == 0) continue;
        const y = s.expert_a[0 .. n * cfg.kvDim()];
        try expertMatmul(io, gpa, y, s.expert_in[0 .. n * cfg.hidden], n, w);
        ops.act.silu(y);
        scatterExpert(out, cfg.kvDim(), y, n, s.token_ids, s.token_weights);
    }
}

pub fn routedFfnSparse(io: std.Io, gpa: std.mem.Allocator, cfg: Config, experts: Experts, x: []const f32, seq: usize, out: []f32, s: *Scratch) !void {
    const logits = s.route_logits[0 .. seq * cfg.n_experts];
    try ops.matmul.matmul(io, gpa, logits, x, seq, experts.router, null);
    const selected = s.selected[0 .. seq * cfg.n_experts_used];
    const weights = s.selected_weights[0 .. seq * cfg.n_experts_used];
    route(cfg, logits, experts.bias, seq, cfg.n_experts, cfg.n_experts_used, selected, weights);
    try routedFfnSparseSelected(io, gpa, cfg, experts, x, seq, out, s, selected, weights);
}

pub fn routedFfnSparseSelected(io: std.Io, gpa: std.mem.Allocator, cfg: Config, experts: Experts, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    if (routedFfnSlots(io, gpa, cfg, experts, x, seq, out, s, selected, weights)) |_| return else |err| switch (err) {
        error.QuantBackendUnavailable => {},
        else => return err,
    }
    try routedFfnPerExpert(io, gpa, cfg, experts, x, seq, out, s, selected, weights);
}

fn routedFfnPerExpert(io: std.Io, gpa: std.mem.Allocator, cfg: Config, experts: Experts, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    @memset(out, 0);
    for (experts.gate, experts.up, experts.down, 0..) |gate_w, up_w, down_w, e| {
        const n = gatherExpert(cfg.hidden, x, seq, selected, weights, cfg.n_experts_used, e, s);
        if (n == 0) continue;
        const gate = s.expert_a[0 .. n * cfg.expert_intermediate];
        const up = s.expert_b[0 .. n * cfg.expert_intermediate];
        try expertMatmul(io, gpa, gate, s.expert_in[0 .. n * cfg.hidden], n, gate_w);
        try expertMatmul(io, gpa, up, s.expert_in[0 .. n * cfg.hidden], n, up_w);
        ops.act.siluMul(gate, up);
        const y = s.expert_out[0 .. n * cfg.hidden];
        try expertMatmul(io, gpa, y, gate, n, down_w);
        scatterExpert(out, cfg.hidden, y, n, s.token_ids, s.token_weights);
    }
}

/// One token: the selected experts' GEMVs as two threaded passes (gate+up, then
/// down) instead of one fork/join per matrix. Block-quant experts only; anything
/// else reports QuantBackendUnavailable and takes the general path.



/// Which way a host layer's routed blocks take a batch: on the CPU, or uploaded to the
/// device. Calibrated once before the first batch that has to choose, then fixed, so every
/// forward of a process runs the same arithmetic; `TP_K2_HOST_ROWS=N` pins it to "the
/// CPU up to N rows" for an A/B across processes.
pub const HostCost = struct {
    cpu_row_ns: f64,
    upload_ns: f64,

    pub fn onCpu(self: HostCost, rows: usize) bool {
        return self.cpu_row_ns * @as(f64, @floatFromInt(rows)) <= self.upload_ns;
    }

    /// The pinned policy, if the environment names one.
    pub fn pinned() ?HostCost {
        const raw = std.c.getenv("TP_K2_HOST_ROWS") orelse return null;
        const rows = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch return null;
        return .{ .cpu_row_ns = 1, .upload_ns = @floatFromInt(rows) };
    }

    /// The CPU side per row, timed on `rows` synthetic rows through `layer`'s routed
    /// blocks with the experts spread evenly, as a real batch's are. The second of two
    /// passes: the first faults the mapped weights in, a cost only a process's first
    /// batch pays.
    pub fn cpuRowNs(io: std.Io, gpa: std.mem.Allocator, cfg: Config, layer: Layer, s: *Scratch, rows: usize) !f64 {
        _ = try cpuPass(io, gpa, cfg, layer, s, rows);
        return @as(f64, @floatFromInt(try cpuPass(io, gpa, cfg, layer, s, rows))) / @as(f64, @floatFromInt(rows));
    }

    fn cpuPass(io: std.Io, gpa: std.mem.Allocator, cfg: Config, layer: Layer, s: *Scratch, rows: usize) !i96 {
        const x = s.normed[0 .. rows * cfg.hidden];
        for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.01 - 0.08;
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        if (layer.values) |values| {
            const used = cfg.n_value_experts_used;
            synthRoutes(s, rows, used, cfg.n_value_experts);
            try routedValuesSelected(io, gpa, cfg, values, x, rows, s.v[0 .. rows * cfg.kvDim()], s, s.selected[0 .. rows * used], s.selected_weights[0 .. rows * used]);
        }
        if (layer.experts) |experts| {
            const used = cfg.n_experts_used;
            synthRoutes(s, rows, used, cfg.n_experts);
            try routedFfnSparseSelected(io, gpa, cfg, experts, x, rows, s.tmp[0 .. rows * cfg.hidden], s, s.selected[0 .. rows * used], s.selected_weights[0 .. rows * used]);
        }
        return std.Io.Clock.awake.now(io).nanoseconds - t0;
    }

    fn synthRoutes(s: *Scratch, rows: usize, used: usize, n_experts: usize) void {
        for (0..rows) |t| for (0..used) |k| {
            s.selected[t * used + k] = (t * 37 + k * (n_experts / used + 1)) % n_experts;
            s.selected_weights[t * used + k] = 1.0 / @as(f32, @floatFromInt(used));
        };
    }
};

/// Rows `HostCost.cpuRowNs` times: near where the two ways cross, so a fixed cost does
/// not read as a per-row one.
pub const host_calibration_rows = 128;

test "a host layer's batch takes the cheaper way" {
    const hc: HostCost = .{ .cpu_row_ns = 300_000, .upload_ns = 60_000_000 };
    try std.testing.expect(hc.onCpu(200));
    try std.testing.expect(!hc.onCpu(201));
}

/// The routed blocks with the batch's slots packed by expert: each expert with fewer than
/// `small_m_max` rows is one job of one threaded GEMV pass per linear (`quantGemvBatch`,
/// every weight row read once for its rows), where a loop over experts paid a fork and
/// join per matrix; one with more takes the GEMM. The slots then combine in each token's
/// selection order, so a single row is exactly the decode step. QuantBackendUnavailable
/// (ggml absent, `exact_activations`) comes before `out` is written.
fn routedFfnSlots(io: std.Io, gpa: std.mem.Allocator, cfg: Config, experts: Experts, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    const inter = cfg.expert_intermediate;
    const hidden = cfg.hidden;
    const dt = experts.gate[0].dtype;
    if (!dt.isBlockQuant() or experts.up[0].dtype != dt or experts.down[0].dtype != dt) return error.QuantBackendUnavailable;
    const pk = packSlots(s, x, hidden, selected[0 .. seq * cfg.n_experts_used], cfg.n_experts_used, cfg.n_experts);
    var n: usize = 0;
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m > 0 and m < ops.matmul.small_m_max) {
        s.jobs[2 * n] = .{ .y = s.slot_a[r * inter ..][0 .. m * inter], .x = s.slot_in[r * hidden ..][0 .. m * hidden], .w = experts.gate[e].bytes, .m = m };
        s.jobs[2 * n + 1] = .{ .y = s.slot_b[r * inter ..][0 .. m * inter], .x = s.slot_in[r * hidden ..][0 .. m * hidden], .w = experts.up[e].bytes, .m = m };
        n += 1;
    };
    if (n > 0) try ops.matmul.quantGemvBatch(io, dt, s.jobs[0 .. 2 * n], inter, hidden);
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m >= ops.matmul.small_m_max) {
        const xin = s.slot_in[r * hidden ..][0 .. m * hidden];
        try expertMatmul(io, gpa, s.slot_a[r * inter ..][0 .. m * inter], xin, m, experts.gate[e]);
        try expertMatmul(io, gpa, s.slot_b[r * inter ..][0 .. m * inter], xin, m, experts.up[e]);
    };
    ops.act.siluMul(s.slot_a[0 .. pk.rows * inter], s.slot_b[0 .. pk.rows * inter]);
    n = 0;
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m > 0 and m < ops.matmul.small_m_max) {
        s.jobs[n] = .{ .y = s.slot_out[r * hidden ..][0 .. m * hidden], .x = s.slot_a[r * inter ..][0 .. m * inter], .w = experts.down[e].bytes, .m = m };
        n += 1;
    };
    if (n > 0) try ops.matmul.quantGemvBatch(io, dt, s.jobs[0..n], hidden, inter);
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m >= ops.matmul.small_m_max) {
        try expertMatmul(io, gpa, s.slot_out[r * hidden ..][0 .. m * hidden], s.slot_a[r * inter ..][0 .. m * inter], m, experts.down[e]);
    };
    combineSlots(out, s.slot_out, s.slot_row, seq, cfg.n_experts_used, hidden, weights);
}

fn routedValuesSlots(io: std.Io, gpa: std.mem.Allocator, cfg: Config, values: Values, x: []const f32, seq: usize, out: []f32, s: *Scratch, selected: []const usize, weights: []const f32) !void {
    const kv = cfg.kvDim();
    const hidden = cfg.hidden;
    const dt = values.weights[0].dtype;
    if (!dt.isBlockQuant()) return error.QuantBackendUnavailable;
    const pk = packSlots(s, x, hidden, selected[0 .. seq * cfg.n_value_experts_used], cfg.n_value_experts_used, cfg.n_value_experts);
    var n: usize = 0;
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m > 0 and m < ops.matmul.small_m_max) {
        s.jobs[n] = .{ .y = s.slot_a[r * kv ..][0 .. m * kv], .x = s.slot_in[r * hidden ..][0 .. m * hidden], .w = values.weights[e].bytes, .m = m };
        n += 1;
    };
    if (n > 0) try ops.matmul.quantGemvBatch(io, dt, s.jobs[0..n], kv, hidden);
    for (pk.counts, pk.starts, 0..) |m, r, e| if (m >= ops.matmul.small_m_max) {
        try expertMatmul(io, gpa, s.slot_a[r * kv ..][0 .. m * kv], s.slot_in[r * hidden ..][0 .. m * hidden], m, values.weights[e]);
    };
    ops.act.silu(s.slot_a[0 .. pk.rows * kv]);
    combineSlots(out, s.slot_a, s.slot_row, seq, cfg.n_value_experts_used, kv, weights);
}

const Packed = struct { rows: usize, counts: []const usize, starts: []const usize };

/// Pack the slots by expert, token order within one: `slot_row[slot]` is the slot's row,
/// and row r of `slot_in` the input of the token it routes.
fn packSlots(s: *Scratch, x: []const f32, hidden: usize, selected: []const usize, used: usize, n_experts: usize) Packed {
    const counts = s.expert_counts[0..n_experts];
    const starts = s.expert_starts[0..n_experts];
    @memset(counts, 0);
    for (selected) |e| counts[e] += 1;
    var acc: usize = 0;
    for (counts, starts) |c, *st| {
        st.* = acc;
        acc += c;
    }
    @memset(counts, 0);
    for (selected, 0..) |e, at| {
        const r = starts[e] + counts[e];
        counts[e] += 1;
        s.slot_row[at] = r;
        @memcpy(s.slot_in[r * hidden ..][0..hidden], x[at / used * hidden ..][0..hidden]);
    }
    return .{ .rows = selected.len, .counts = counts, .starts = starts };
}

/// out[t] = sum over k of weights[t * used + k] * rows[slot_row[t * used + k]], in k order.
fn combineSlots(out: []f32, rows: []const f32, slot_row: []const usize, seq: usize, used: usize, width: usize, weights: []const f32) void {
    @memset(out[0 .. seq * width], 0);
    for (0..seq) |t| {
        const dst = out[t * width ..][0..width];
        for (0..used) |k| {
            const scale = weights[t * used + k];
            for (dst, rows[slot_row[t * used + k] * width ..][0..width]) |*d, v| d.* += v * scale;
        }
    }
}

fn expertMatmul(io: std.Io, gpa: std.mem.Allocator, y: []f32, x: []const f32, rows: usize, w: ops.matmul.Weight) !void {
    if (!w.dtype.isBlockQuant() or rows < ops.matmul.small_m_max) {
        return ops.matmul.matmul(io, gpa, y, x, rows, w, null);
    }
    const chunk = ops.matmul.small_m_max - 1;
    var off: usize = 0;
    while (off < rows) : (off += chunk) {
        const n = @min(chunk, rows - off);
        try ops.matmul.matmul(
            io,
            gpa,
            y[off * w.rows ..][0 .. n * w.rows],
            x[off * w.cols ..][0 .. n * w.cols],
            n,
            w,
            null,
        );
    }
}

pub fn routedFfn(io: std.Io, gpa: std.mem.Allocator, cfg: Config, experts: Experts, x: []const f32, seq: usize, out: []f32, s: *Scratch) !void {
    try routedFfnSparse(io, gpa, cfg, experts, x, seq, out, s);
    if (experts.shared_gate) |gate_w| {
        const gate = s.expert_a[0 .. seq * cfg.shared_intermediate];
        const up = s.expert_b[0 .. seq * cfg.shared_intermediate];
        try ops.matmul.matmul(io, gpa, gate, x, seq, gate_w, null);
        try ops.matmul.matmul(io, gpa, up, x, seq, experts.shared_up.?, null);
        ops.act.siluMul(gate, up);
        const shared = s.expert_out[0 .. seq * cfg.hidden];
        try ops.matmul.matmul(io, gpa, shared, gate, seq, experts.shared_down.?, null);
        for (out, shared) |*dst, add| dst.* += add;
    }
}

fn gatherExpert(hidden: usize, x: []const f32, seq: usize, selected: []const usize, weights: []const f32, used: usize, expert: usize, s: *Scratch) usize {
    var n: usize = 0;
    for (0..seq) |t| for (0..used) |k| {
        const at = t * used + k;
        if (selected[at] != expert) continue;
        s.token_ids[n] = t;
        s.token_weights[n] = weights[at];
        @memcpy(s.expert_in[n * hidden ..][0..hidden], x[t * hidden ..][0..hidden]);
        n += 1;
    };
    return n;
}

fn scatterExpert(out: []f32, width: usize, values: []const f32, n: usize, ids: []const usize, weights: []const f32) void {
    for (0..n) |i| {
        const dst = out[ids[i] * width ..][0..width];
        const src = values[i * width ..][0..width];
        const scale = weights[i];
        for (dst, src) |*d, v| d.* += v * scale;
    }
}

pub const CpuModel = struct {
    lm: *const Model,
    gpa: std.mem.Allocator,
    cache: kvmod.KvCache,
    freqs: ops.rope.Freqs,
    last_hidden: []f32,
    max_capacity: usize,

    pub fn init(gpa: std.mem.Allocator, lm: *const Model, cap: kvmod.Capacity) !CpuModel {
        var cache = try kvmod.KvCache.init(gpa, lm.cfg.n_layers, cap.initial, lm.cfg.kvDim(), cap.kv_dtype);
        errdefer cache.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.initial, lm.cfg.head_dim, lm.cfg.rope_theta);
        errdefer freqs.deinit(gpa);
        return .{
            .lm = lm,
            .gpa = gpa,
            .cache = cache,
            .freqs = freqs,
            .last_hidden = try gpa.alloc(f32, lm.cfg.hidden),
            .max_capacity = cap.max,
        };
    }

    pub fn deinit(self: *CpuModel) void {
        self.cache.deinit(self.gpa);
        self.freqs.deinit(self.gpa);
        self.gpa.free(self.last_hidden);
        self.* = undefined;
    }

    pub fn cached(self: *const CpuModel) usize {
        return self.cache.len;
    }
    pub fn remaining(self: *const CpuModel) usize {
        return self.cache.remaining();
    }
    pub fn capacityMax(self: *const CpuModel) usize {
        return self.max_capacity;
    }
    pub fn vocab(self: *const CpuModel) usize {
        return self.lm.cfg.vocab;
    }

    pub fn ensureCapacity(self: *CpuModel, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.cache.capacity, self.max_capacity, min_rows)) orelse return;
        var new_freqs = ops.rope.rotateHalfFreqs(self.gpa, target, self.lm.cfg.head_dim, self.lm.cfg.rope_theta) catch return error.ContextFull;
        errdefer new_freqs.deinit(self.gpa);
        self.cache.grow(self.gpa, target) catch return error.ContextFull;
        self.freqs.deinit(self.gpa);
        self.freqs = new_freqs;
    }

    pub fn step(self: *CpuModel, io: std.Io, ids: []const u32, logits: []f32) !void {
        try self.lm.forwardCached(io, self.gpa, ids, &self.cache, self.freqs, self.last_hidden);
        try ops.matmul.matmul(io, self.gpa, logits, self.last_hidden, 1, self.lm.head, null);
    }
};

test "K2 Horizon config loads from the 36B GGUF" {
    const path = "/home/qt/genai/lmstudio/models/K2-Horizon-MoVA-36B-A4B-Q6_K.gguf";
    var g = Gguf.open(std.testing.allocator, std.testing.io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer g.deinit();
    var model = try Model.load(std.testing.allocator, &g);
    defer model.deinit();
    try std.testing.expectEqual(@as(usize, 48), model.cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 100), model.cfg.n_experts);
    try std.testing.expectEqual(@as(usize, 64), model.cfg.n_value_experts);
    try std.testing.expectEqual(@as(usize, 3), model.cfg.dense_layers);

    var tok = try Tokenizer.initFromGguf(std.testing.allocator, &g);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(std.testing.allocator);
    try tok.encode(std.testing.allocator, "Cafe\u{301} عربي\u{200d}test 1234\n\n", &ids);
    try std.testing.expectEqualSlices(u32, &.{ 164494, 53986, 12447, 16727, 2749, 222, 4018, 21, 503 }, ids.items);
    const tags = try tok.decodeAlloc(std.testing.allocator, &.{ 250054, 250043, 250056, 250057, 250060, 250061, 250044, 250055 });
    defer std.testing.allocator.free(tags);
    try std.testing.expectEqualStrings("<ifm|tool_calls><ifm|tool_call><ifm|arg_key></ifm|arg_key><ifm|arg_value></ifm|arg_value></ifm|tool_call></ifm|tool_calls>", tags);
}
