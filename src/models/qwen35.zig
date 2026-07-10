//! Qwen3.5/3.6 hybrid language model (GGUF arch "qwen35"): a stack where
//! every `full_attention_interval`-th layer is gated full attention and the
//! rest are gated-DeltaNet linear attention. Ported from llama.cpp
//! (src/models/qwen35.cpp + delta-net-base.cpp `build_delta_net_autoregressive`).
//!
//! Full-attention layers (blk 3, 7, ... with interval 4): the q projection
//! emits query and output-gate interleaved per head ([256 q][256 gate] per
//! 512-wide head slot), per-head QK RMS-norm at head_dim 256, partial neox
//! RoPE over the first `rope_dim` (64) head dims (text-only collapses the
//! interleaved M-RoPE sections to standard RoPE), causal GQA attention, then
//! out = wo(attn * sigmoid(gate)).
//!
//! Linear-attention layers: qkv = wqkv(x) runs through a per-channel causal
//! conv (kernel 4, SiLU) whose 3-column tail is recurrent state; q/k are
//! L2-normalized per 128-dim head and tiled from 16 k-heads to 48 v-heads
//! (v-head j uses k-head j % 16). Per v-head delta rule over a 128x128
//! state S (k-dim i, v-dim j):
//!   S *= exp(a * softplus(alpha + dt_bias))    (a stored as -exp(A_log))
//!   m_j = sum_i S[i,j] k_i
//!   d_j = (v_j - m_j) * sigmoid(beta)
//!   S[i,j] += k_i d_j
//!   o_j = sum_i S[i,j] q_i / sqrt(128)
//! then o = rmsnorm_per_head(o; ssm_norm) * silu(z), out = ssm_out(o), with
//! z = attn_gate(x). Prefill runs the same recurrence sequentially per token
//! (llama.cpp's chunked path is an optimization, not different math).
//!
//! Weights stay in checkpoint dtype (GGUF block quants) and dequantize
//! inside the GEMM; the Gguf mapping must outlive the model.

const std = @import("std");
const gguf_mod = @import("../gguf.zig");
const weights_mod = @import("../weights.zig");
const qwen3 = @import("qwen3.zig");
const ops = @import("../ops.zig");
const kv_cache_mod = @import("../llm/kv_cache.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const KvCache = kv_cache_mod.KvCache;

pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    intermediate: usize,
    rope_theta: f64,
    rms_eps: f32,
    vocab: usize,
    full_attn_interval: usize,
    /// Interleaved M-RoPE section widths (t, h, w) in rotary pairs; text-only
    /// input makes them irrelevant (all position channels equal).
    rope_sections: [3]u32,
    // Gated-DeltaNet dims.
    lin_k_heads: usize,
    lin_v_heads: usize,
    lin_head_dim: usize,
    conv_kernel: usize,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
    pub fn linQKDim(self: Config) usize {
        return self.lin_k_heads * self.lin_head_dim;
    }
    pub fn linVDim(self: Config) usize {
        return self.lin_v_heads * self.lin_head_dim;
    }
    /// Channels through the causal conv: q + k + v concatenated.
    pub fn convChannels(self: Config) usize {
        return 2 * self.linQKDim() + self.linVDim();
    }
    pub fn isRecurrent(self: Config, l: usize) bool {
        return (l + 1) % self.full_attn_interval != 0;
    }
    pub fn nAttnLayers(self: Config) usize {
        return self.n_layers / self.full_attn_interval;
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "qwen35")) return error.UnknownModelConfig;
        const key = struct {
            fn f(gg: *const Gguf, comptime name: []const u8) !usize {
                return @intCast(gg.getUint("qwen35." ++ name) orelse return error.UnknownModelConfig);
            }
        }.f;
        const head_dim = try key(g, "attention.key_length");
        if (try key(g, "attention.value_length") != head_dim) return error.UnknownModelConfig;
        const state_size = try key(g, "ssm.state_size");
        const v_heads = try key(g, "ssm.time_step_rank");
        const inner = try key(g, "ssm.inner_size");
        if (inner % v_heads != 0 or inner / v_heads != state_size) return error.UnknownModelConfig;
        const embed = g.get("embed_tokens.weight") orelse return error.UnknownModelConfig;
        const eshape = embed.info.shape.slice();
        if (eshape.len != 2) return error.UnknownModelConfig;
        var sections: [3]u32 = undefined;
        {
            const arr = g.getArr("qwen35.rope.dimension_sections") orelse return error.UnknownModelConfig;
            var it = arr.iterate();
            for (&sections) |*s| {
                const v = it.next() orelse return error.UnknownModelConfig;
                s.* = @intCast(switch (v) {
                    .int => |i| i,
                    .uint => |u| @as(i64, @intCast(u)),
                    else => return error.UnknownModelConfig,
                });
            }
        }
        return .{
            .n_layers = try key(g, "block_count"),
            .hidden = try key(g, "embedding_length"),
            .n_heads = try key(g, "attention.head_count"),
            .n_kv_heads = try key(g, "attention.head_count_kv"),
            .head_dim = head_dim,
            .rope_dim = try key(g, "rope.dimension_count"),
            .intermediate = try key(g, "feed_forward_length"),
            .rope_theta = g.getFloat("qwen35.rope.freq_base") orelse 1e7,
            .rms_eps = @floatCast(g.getFloat("qwen35.attention.layer_norm_rms_epsilon") orelse 1e-6),
            .vocab = eshape[0],
            .full_attn_interval = try key(g, "full_attention_interval"),
            .rope_sections = sections,
            .lin_k_heads = try key(g, "ssm.group_count"),
            .lin_v_heads = v_heads,
            .lin_head_dim = state_size,
            .conv_kernel = try key(g, "ssm.conv_kernel"),
        };
    }
};

const Mlp = struct {
    post_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

const AttnLayer = struct {
    input_norm: []const f32,
    /// [n_heads * head_dim * 2, hidden]: per-head-interleaved query + gate.
    qg: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: []const f32, // [head_dim]
    k_norm: []const f32,
    mlp: Mlp,
};

const LinLayer = struct {
    input_norm: []const f32,
    /// [convChannels, hidden]: q | k | v.
    qkv: Weight,
    /// [linVDim, hidden]: the gated-norm gate z (GGUF "attn_gate").
    z: Weight,
    alpha: Weight, // [v_heads, hidden]
    beta: Weight, // [v_heads, hidden]
    a: []const f32, // [v_heads], stored as -exp(A_log)
    dt_bias: []const f32, // [v_heads]
    /// [convChannels][kernel] causal conv taps, w[0] on the oldest token.
    conv_w: []const f32,
    ssm_norm: []const f32, // [lin_head_dim]
    out: Weight, // [hidden, linVDim]
    mlp: Mlp,
};

const Layer = union(enum) {
    attn: AttnLayer,
    linear: LinLayer,
};

/// Recurrent + attention caches for one session.
pub const State = struct {
    /// K/V rows for the full-attention layers only (indexed by attn slot).
    kv: KvCache,
    /// [n_lin_layers][convChannels][kernel-1] rolling conv tail.
    conv: []f32,
    /// [n_lin_layers][v_heads][d][d] delta-rule states (k-dim major).
    ssm: []f32,
    len: usize = 0,
    capacity: usize,

    pub fn init(gpa: std.mem.Allocator, cfg: Config, capacity: usize) !State {
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        var kv = try KvCache.init(gpa, cfg.nAttnLayers(), capacity, cfg.kvDim());
        errdefer kv.deinit(gpa);
        const conv = try gpa.alloc(f32, n_lin * cfg.convChannels() * (cfg.conv_kernel - 1));
        errdefer gpa.free(conv);
        const ssm = try gpa.alloc(f32, n_lin * cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim);
        errdefer gpa.free(ssm);
        @memset(conv, 0);
        @memset(ssm, 0);
        return .{ .kv = kv, .conv = conv, .ssm = ssm, .capacity = capacity };
    }

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        self.kv.deinit(gpa);
        gpa.free(self.conv);
        gpa.free(self.ssm);
        self.* = undefined;
    }

    fn convState(self: *State, cfg: Config, lin_idx: usize) []f32 {
        const n = cfg.convChannels() * (cfg.conv_kernel - 1);
        return self.conv[lin_idx * n ..][0..n];
    }

    fn ssmState(self: *State, cfg: Config, lin_idx: usize) []f32 {
        const n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
        return self.ssm[lin_idx * n ..][0..n];
    }
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    embed: Weight,
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Model {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const embed = try loadMatrix(store, "embed_tokens.weight", cfg.vocab, cfg.hidden);
        const head = try loadMatrix(store, "lm_head.weight", cfg.vocab, cfg.hidden);
        const final_norm = try loadVec(alloc, store, "norm.weight", cfg.hidden);

        const layers = try alloc.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            const mlp: Mlp = .{
                .post_norm = try loadLayerVec(alloc, store, l, "post_attention_layernorm.weight", cfg.hidden),
                .gate = try loadLayerMatrix(store, l, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
                .up = try loadLayerMatrix(store, l, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
                .down = try loadLayerMatrix(store, l, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
            };
            const input_norm = try loadLayerVec(alloc, store, l, "input_layernorm.weight", cfg.hidden);
            if (cfg.isRecurrent(l)) {
                layer.* = .{ .linear = .{
                    .input_norm = input_norm,
                    .qkv = try loadLayerMatrix(store, l, "attn_qkv.weight", cfg.convChannels(), cfg.hidden),
                    .z = try loadLayerMatrix(store, l, "attn_gate.weight", cfg.linVDim(), cfg.hidden),
                    .alpha = try loadLayerMatrix(store, l, "ssm_alpha.weight", cfg.lin_v_heads, cfg.hidden),
                    .beta = try loadLayerMatrix(store, l, "ssm_beta.weight", cfg.lin_v_heads, cfg.hidden),
                    .a = try loadLayerVec(alloc, store, l, "ssm_a", cfg.lin_v_heads),
                    .dt_bias = try loadLayerVec(alloc, store, l, "ssm_dt.bias", cfg.lin_v_heads),
                    .conv_w = try loadLayerVec(alloc, store, l, "ssm_conv1d.weight", cfg.convChannels() * cfg.conv_kernel),
                    .ssm_norm = try loadLayerVec(alloc, store, l, "ssm_norm.weight", cfg.lin_head_dim),
                    .out = try loadLayerMatrix(store, l, "ssm_out.weight", cfg.hidden, cfg.linVDim()),
                    .mlp = mlp,
                } };
            } else {
                layer.* = .{ .attn = .{
                    .input_norm = input_norm,
                    .qg = try loadLayerMatrix(store, l, "self_attn.q_proj.weight", cfg.qDim() * 2, cfg.hidden),
                    .k = try loadLayerMatrix(store, l, "self_attn.k_proj.weight", cfg.kvDim(), cfg.hidden),
                    .v = try loadLayerMatrix(store, l, "self_attn.v_proj.weight", cfg.kvDim(), cfg.hidden),
                    .o = try loadLayerMatrix(store, l, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim()),
                    .q_norm = try loadLayerVec(alloc, store, l, "self_attn.q_norm.weight", cfg.head_dim),
                    .k_norm = try loadLayerVec(alloc, store, l, "self_attn.k_norm.weight", cfg.head_dim),
                    .mlp = mlp,
                } };
            }
        }

        return .{
            .arena = arena,
            .cfg = cfg,
            .embed = embed,
            .head = head,
            .layers = layers,
            .final_norm = final_norm,
        };
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Forward `ids` at positions [state.len, state.len + ids.len). When
    /// `out` ([n * hidden], n <= ids.len) is set, it receives the
    /// final-normed hidden states of the last n positions (LM-head ready).
    pub fn forwardCached(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        state: *State,
        freqs: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0 and state.len + seq <= state.capacity);

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.embed, ids, x);

        var s = try Scratch.init(gpa, seq, cfg);
        defer s.deinit(gpa);

        for (self.layers, 0..) |*layer, l| {
            switch (layer.*) {
                .attn => |*al| {
                    const slot = l / cfg.full_attn_interval;
                    try attnLayerForward(io, gpa, cfg, al, x, seq, freqs, &state.kv, slot, &s);
                },
                .linear => |*ll| {
                    // Linear layers before l: l minus the attn layers before l.
                    const lin_idx = l - l / cfg.full_attn_interval;
                    try linLayerForward(io, gpa, cfg, ll, x, seq, state.convState(cfg, lin_idx), state.ssmState(cfg, lin_idx), &s);
                },
            }
            try mlpForward(io, gpa, cfg, layerMlp(layer), x, seq, &s);
        }
        state.kv.commit(seq);
        state.len += seq;

        if (out) |o| {
            std.debug.assert(o.len % cfg.hidden == 0);
            const n = o.len / cfg.hidden;
            std.debug.assert(n >= 1 and n <= seq);
            ops.norm.rmsNorm(o, x[(seq - n) * cfg.hidden ..][0 .. n * cfg.hidden], self.final_norm, cfg.rms_eps);
        }
    }
};

/// Pointer into the layer's stored Mlp (the layer must be passed by
/// pointer — a by-value union would hand back a dangling stack address).
fn layerMlp(layer: *const Layer) *const Mlp {
    return switch (layer.*) {
        .attn => |*al| &al.mlp,
        .linear => |*ll| &ll.mlp,
    };
}

/// Gated full attention (llama.cpp build_layer_attn): interleaved q+gate,
/// per-head QK-norm, partial RoPE, causal GQA, sigmoid output gate.
fn attnLayerForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: *const AttnLayer,
    x: []f32,
    seq: usize,
    freqs: ops.rope.Freqs,
    cache: *KvCache,
    slot: usize,
    s: *Scratch,
) !void {
    const pos0 = cache.len;
    const hd = cfg.head_dim;

    ops.norm.rmsNorm(s.normed, x, layer.input_norm, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, s.qg, s.normed, seq, layer.qg, null);
    try ops.matmul.matmul(io, gpa, s.k, s.normed, seq, layer.k, null);
    try ops.matmul.matmul(io, gpa, s.v, s.normed, seq, layer.v, null);

    // Split the interleaved [q(hd) gate(hd)] head slots.
    for (0..seq) |t| {
        for (0..cfg.n_heads) |h| {
            const src = s.qg[(t * cfg.n_heads + h) * hd * 2 ..];
            @memcpy(s.q[(t * cfg.n_heads + h) * hd ..][0..hd], src[0..hd]);
            @memcpy(s.gate[(t * cfg.n_heads + h) * hd ..][0..hd], src[hd .. hd * 2]);
        }
    }

    ops.norm.rmsNorm(s.q[0 .. seq * cfg.qDim()], s.q[0 .. seq * cfg.qDim()], layer.q_norm, cfg.rms_eps);
    ops.norm.rmsNorm(s.k[0 .. seq * cfg.kvDim()], s.k[0 .. seq * cfg.kvDim()], layer.k_norm, cfg.rms_eps);
    ops.rope.applyRotateHalfPartialAt(s.q[0 .. seq * cfg.qDim()], freqs, pos0, seq, cfg.n_heads, hd, cfg.rope_dim);
    ops.rope.applyRotateHalfPartialAt(s.k[0 .. seq * cfg.kvDim()], freqs, pos0, seq, cfg.n_kv_heads, hd, cfg.rope_dim);

    cache.write(slot, s.k[0 .. seq * cfg.kvDim()], s.v[0 .. seq * cfg.kvDim()]);
    try ops.attention.attention(io, gpa, s.attn_out, s.q[0 .. seq * cfg.qDim()], cache.kView(slot, seq), cache.vView(slot, seq), .{
        .seq_q = seq,
        .seq_kv = pos0 + seq,
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_kv_heads,
        .head_dim = hd,
        .causal = true,
    });

    // Output gating: attn * sigmoid(gate), elementwise per head dim.
    for (s.attn_out[0 .. seq * cfg.qDim()], s.gate[0 .. seq * cfg.qDim()]) |*a, g| {
        a.* *= 1.0 / (1.0 + @exp(-g));
    }

    try ops.matmul.matmul(io, gpa, s.tmp, s.attn_out, seq, layer.o, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;
}

/// Gated DeltaNet linear attention (llama.cpp build_layer_attn_linear):
/// batched projections, then the sequential per-token conv + delta rule.
fn linLayerForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: *const LinLayer,
    x: []f32,
    seq: usize,
    conv_state: []f32,
    ssm_state: []f32,
    s: *Scratch,
) !void {
    const channels = cfg.convChannels();
    const d = cfg.lin_head_dim;
    const vdim = cfg.linVDim();
    const qkdim = cfg.linQKDim();
    const heads = cfg.lin_v_heads;
    const taps = cfg.conv_kernel;
    const readout_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    ops.norm.rmsNorm(s.normed, x, layer.input_norm, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, s.lin_qkv, s.normed, seq, layer.qkv, null);
    try ops.matmul.matmul(io, gpa, s.lin_z, s.normed, seq, layer.z, null);
    try ops.matmul.matmul(io, gpa, s.lin_alpha, s.normed, seq, layer.alpha, null);
    try ops.matmul.matmul(io, gpa, s.lin_beta, s.normed, seq, layer.beta, null);

    for (0..seq) |t| {
        const qkv_t = s.lin_qkv[t * channels ..][0..channels];
        const conv_t = s.lin_conv;

        // Depthwise causal conv over [state | current], SiLU; the state
        // rolls forward one column. w[0] hits the oldest tap.
        for (0..channels) |c| {
            const st = conv_state[c * (taps - 1) ..][0 .. taps - 1];
            const w = layer.conv_w[c * taps ..][0..taps];
            var acc: f32 = w[taps - 1] * qkv_t[c];
            for (0..taps - 1) |k| acc += w[k] * st[k];
            for (0..taps - 2) |k| st[k] = st[k + 1];
            st[taps - 2] = qkv_t[c];
            conv_t[c] = acc / (1.0 + @exp(-acc)); // SiLU
        }

        // Split and L2-normalize q/k per head (clamped, ggml_l2_norm).
        const qc = conv_t[0..qkdim];
        const kc = conv_t[qkdim .. 2 * qkdim];
        const vc = conv_t[2 * qkdim ..][0..vdim];
        l2NormRows(qc, d, cfg.rms_eps);
        l2NormRows(kc, d, cfg.rms_eps);

        // Per-head delta rule; v-head h uses k-head h % lin_k_heads.
        const o_t = s.lin_o[t * vdim ..][0..vdim];
        for (0..heads) |h| {
            const decay = @exp(layer.a[h] * softplus(s.lin_alpha[t * heads + h] + layer.dt_bias[h]));
            const beta = 1.0 / (1.0 + @exp(-s.lin_beta[t * heads + h]));
            const kh = kc[(h % cfg.lin_k_heads) * d ..][0..d];
            const qh = qc[(h % cfg.lin_k_heads) * d ..][0..d];
            const vh = vc[h * d ..][0..d];
            const S = ssm_state[h * d * d ..][0 .. d * d];

            // Pass 1: decay the state and read the memory m = S^T k.
            const m = s.lin_m;
            @memset(m, 0);
            for (0..d) |i| {
                const row = S[i * d ..][0..d];
                const ki = kh[i];
                for (row, m) |*sij, *mj| {
                    sij.* *= decay;
                    mj.* += sij.* * ki;
                }
            }
            // Pass 2: rank-1 update + readout o = S^T q / sqrt(d).
            const dl = s.lin_d;
            for (0..d) |j| dl[j] = (vh[j] - m[j]) * beta;
            const oh = o_t[h * d ..][0..d];
            @memset(oh, 0);
            for (0..d) |i| {
                const row = S[i * d ..][0..d];
                const ki = kh[i];
                const qi = qh[i] * readout_scale;
                for (row, dl, oh) |*sij, dj, *oj| {
                    sij.* += ki * dj;
                    oj.* += sij.* * qi;
                }
            }
        }
    }

    // Gated per-head RMS norm: rmsnorm(o; ssm_norm) * silu(z).
    ops.norm.rmsNorm(s.lin_o[0 .. seq * vdim], s.lin_o[0 .. seq * vdim], layer.ssm_norm, cfg.rms_eps);
    for (s.lin_o[0 .. seq * vdim], s.lin_z[0 .. seq * vdim]) |*o, z| {
        o.* *= z / (1.0 + @exp(-z)); // SiLU(z)
    }

    try ops.matmul.matmul(io, gpa, s.tmp, s.lin_o, seq, layer.out, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;
}

fn mlpForward(io: std.Io, gpa: std.mem.Allocator, cfg: Config, mlp: *const Mlp, x: []f32, seq: usize, s: *Scratch) !void {
    ops.norm.rmsNorm(s.normed, x, mlp.post_norm, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, s.mlp_gate, s.normed, seq, mlp.gate, null);
    try ops.matmul.matmul(io, gpa, s.mlp_up, s.normed, seq, mlp.up, null);
    ops.act.siluMul(s.mlp_gate, s.mlp_up);
    try ops.matmul.matmul(io, gpa, s.tmp, s.mlp_gate, seq, mlp.down, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;
}

/// x_row / max(|x_row|, eps) for each `dim`-wide row (ggml_l2_norm).
fn l2NormRows(x: []f32, dim: usize, eps: f32) void {
    var off: usize = 0;
    while (off < x.len) : (off += dim) {
        const row = x[off..][0..dim];
        var ss: f32 = 0;
        for (row) |v| ss += v * v;
        const scale = 1.0 / @max(@sqrt(ss), eps);
        for (row) |*v| v.* *= scale;
    }
}

fn softplus(v: f32) f32 {
    return if (v > 20.0) v else @log(1.0 + @exp(v));
}

/// Per-forward activation buffers for `seq` tokens.
const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    // Full attention.
    qg: []f32,
    q: []f32,
    gate: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    // Linear attention.
    lin_qkv: []f32,
    lin_conv: []f32,
    lin_z: []f32,
    lin_o: []f32,
    lin_alpha: []f32,
    lin_beta: []f32,
    lin_m: []f32,
    lin_d: []f32,
    // MLP.
    mlp_gate: []f32,
    mlp_up: []f32,

    fn init(gpa: std.mem.Allocator, seq: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        var done: usize = 0;
        errdefer { // free the fields allocated before the failing one
            inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
                if (i < done) gpa.free(@field(s, f.name));
            }
        }
        const sizes = [_]usize{
            seq * cfg.hidden, // normed
            seq * cfg.hidden, // tmp
            seq * cfg.qDim() * 2, // qg
            seq * cfg.qDim(), // q
            seq * cfg.qDim(), // gate
            seq * cfg.kvDim(), // k
            seq * cfg.kvDim(), // v
            seq * cfg.qDim(), // attn_out
            seq * cfg.convChannels(), // lin_qkv
            cfg.convChannels(), // lin_conv
            seq * cfg.linVDim(), // lin_z
            seq * cfg.linVDim(), // lin_o
            seq * cfg.lin_v_heads, // lin_alpha
            seq * cfg.lin_v_heads, // lin_beta
            cfg.lin_head_dim, // lin_m
            cfg.lin_head_dim, // lin_d
            seq * cfg.intermediate, // mlp_gate
            seq * cfg.intermediate, // mlp_up
        };
        inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try gpa.alloc(f32, sizes[i]);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        inline for (@typeInfo(Scratch).@"struct".fields) |f| {
            gpa.free(@field(s, f.name));
        }
        s.* = undefined;
    }
};

fn loadMatrix(store: WeightStore, name: []const u8, rows: usize, cols: usize) !Weight {
    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

fn loadLayerMatrix(store: WeightStore, l: usize, comptime suffix: []const u8, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    return loadMatrix(store, try std.fmt.bufPrint(&buf, "layers.{d}." ++ suffix, .{l}), rows, cols);
}

fn loadVec(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]f32 {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

fn loadLayerVec(alloc: std.mem.Allocator, store: WeightStore, l: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    return loadVec(alloc, store, try std.fmt.bufPrint(&buf, "layers.{d}." ++ suffix, .{l}), len);
}

/// CPU stepper for the engine loop (mirrors engine.CpuModel). Speculative
/// decoding is unsupported: the recurrent state cannot be truncated.
pub const CpuModel = struct {
    lm: *const Model,
    gpa: std.mem.Allocator,
    state: State,
    freqs: ops.rope.Freqs,
    last_hidden: []f32,

    pub fn init(gpa: std.mem.Allocator, lm: *const Model, capacity: usize) !CpuModel {
        var state = try State.init(gpa, lm.cfg, capacity);
        errdefer state.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, capacity, lm.cfg.rope_dim, lm.cfg.rope_theta);
        errdefer freqs.deinit(gpa);
        const last_hidden = try gpa.alloc(f32, lm.cfg.hidden);
        return .{ .lm = lm, .gpa = gpa, .state = state, .freqs = freqs, .last_hidden = last_hidden };
    }

    pub fn deinit(self: *CpuModel) void {
        self.state.deinit(self.gpa);
        self.freqs.deinit(self.gpa);
        self.gpa.free(self.last_hidden);
        self.* = undefined;
    }

    pub fn cached(self: *const CpuModel) usize {
        return self.state.len;
    }

    pub fn remaining(self: *const CpuModel) usize {
        return self.state.capacity - self.state.len;
    }

    pub fn vocab(self: *const CpuModel) usize {
        return self.lm.cfg.vocab;
    }

    pub fn step(self: *CpuModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        try self.lm.forwardCached(io, self.gpa, ids_new, &self.state, self.freqs, self.last_hidden);
        try ops.matmul.matmul(io, self.gpa, logits, self.last_hidden, 1, self.lm.head, null);
    }
};

// --- tests -----------------------------------------------------------------

// The delta-rule recurrence against an independent naive f64 reference
// (same formulas, straightforward loops) over a few tokens — catches
// state-layout / decay / tiling mistakes in the optimized two-pass sweep.
test "delta rule recurrence matches naive reference" {
    const d = 4;
    const heads = 2;
    const k_heads = 1; // both v-heads share k-head 0 (tiling j % k_heads)
    const n_tokens = 5;

    var prng = std.Random.DefaultPrng.init(31337);
    const rand = prng.random();

    // Optimized-path state and naive-path state.
    var S = [_]f32{0} ** (heads * d * d);
    var S_ref = [_]f64{0} ** (heads * d * d);
    var o = [_]f32{0} ** (heads * d);
    var o_ref = [_]f64{0} ** (heads * d);
    const scale = 1.0 / @sqrt(@as(f32, d));

    var m_buf: [d]f32 = undefined;
    var d_buf: [d]f32 = undefined;

    for (0..n_tokens) |_| {
        var q: [k_heads * d]f32 = undefined;
        var k: [k_heads * d]f32 = undefined;
        var v: [heads * d]f32 = undefined;
        for (&q) |*x| x.* = rand.floatNorm(f32);
        for (&k) |*x| x.* = rand.floatNorm(f32);
        for (&v) |*x| x.* = rand.floatNorm(f32);
        l2NormRows(&q, d, 1e-6);
        l2NormRows(&k, d, 1e-6);
        const decays = [heads]f32{ 0.9, 0.5 };
        const betas = [heads]f32{ 0.7, 1.0 };

        for (0..heads) |h| {
            const kh = k[(h % k_heads) * d ..][0..d];
            const qh = q[(h % k_heads) * d ..][0..d];
            const vh = v[h * d ..][0..d];

            // Optimized two-pass sweep (mirrors linLayerForward).
            {
                const Sh = S[h * d * d ..][0 .. d * d];
                const m = &m_buf;
                @memset(m, 0);
                for (0..d) |i| {
                    const row = Sh[i * d ..][0..d];
                    for (row, m) |*sij, *mj| {
                        sij.* *= decays[h];
                        mj.* += sij.* * kh[i];
                    }
                }
                for (0..d) |j| d_buf[j] = (vh[j] - m[j]) * betas[h];
                const oh = o[h * d ..][0..d];
                @memset(oh, 0);
                for (0..d) |i| {
                    const row = Sh[i * d ..][0..d];
                    const qi = qh[i] * scale;
                    for (row, d_buf, oh) |*sij, dj, *oj| {
                        sij.* += kh[i] * dj;
                        oj.* += sij.* * qi;
                    }
                }
            }
            // Naive reference straight from the formulas.
            {
                const Sh = S_ref[h * d * d ..][0 .. d * d];
                for (Sh) |*sv| sv.* *= decays[h];
                var m: [d]f64 = @splat(0);
                for (0..d) |i| {
                    for (0..d) |j| m[j] += Sh[i * d + j] * kh[i];
                }
                var delta: [d]f64 = undefined;
                for (0..d) |j| delta[j] = (vh[j] - m[j]) * betas[h];
                for (0..d) |i| {
                    for (0..d) |j| Sh[i * d + j] += kh[i] * delta[j];
                }
                const oh = o_ref[h * d ..][0..d];
                @memset(oh, 0);
                for (0..d) |i| {
                    for (0..d) |j| oh[j] += Sh[i * d + j] * qh[i] * scale;
                }
            }
        }
        for (o, o_ref) |a, e| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(e)), a, 1e-4);
        for (S, S_ref) |a, e| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(e)), a, 1e-4);
    }
}

// Config + weight wiring against the real Qwen3.6 checkpoint; skipped when
// absent. Load-only — the 27B forward is validated end-to-end via tp-llm
// against llama.cpp.
test "qwen35 loads from real qwen3.6 gguf" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Q5_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try Model.load(gpa, &g);
    defer lm.deinit();

    const cfg = lm.cfg;
    try std.testing.expectEqual(@as(usize, 64), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 5120), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 24), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 4), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 64), cfg.rope_dim);
    try std.testing.expectEqual(@as(usize, 248320), cfg.vocab);
    try std.testing.expectEqual(@as(usize, 16), cfg.nAttnLayers());
    try std.testing.expectEqual(@as(usize, 10240), cfg.convChannels());
    try std.testing.expect(cfg.isRecurrent(0) and !cfg.isRecurrent(3) and !cfg.isRecurrent(63));
    try std.testing.expect(lm.layers[3] == .attn and lm.layers[0] == .linear);
    // Untied head: output.weight is its own tensor.
    try std.testing.expect(lm.head.bytes.ptr != lm.embed.bytes.ptr);
}
