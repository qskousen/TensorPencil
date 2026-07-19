//! EAGLE-3 draft head (LLM_PLAN.md M7): a trained one-layer drafter that
//! reads the TARGET model's own hidden states instead of running a second
//! full model. Per committed position the target taps its residual stream
//! entering three layers (CudaLM.enableTaps); the head fuses them
//! (fc: 3*hidden -> hidden), pairs the fused feature with the NEXT token's
//! embedding, and one llama-style decoder layer + a reduced 32k-vocab LM
//! head (d2t delta-maps draft ids to target ids) drafts greedily — its own
//! hidden output feeds subsequent rollout steps (EAGLE-3 "training-time
//! test"). ~218M bf16 params: a draft step is one layer instead of the 0.6B
//! draft model's 28.
//!
//! The head weights here (AngelSlim/Qwen3-4B_eagle3) were trained on vanilla
//! Qwen3-4B-Instruct; running them against the Heretic-abliterated VL
//! checkpoint at fp8 is an experiment — acceptance measures the feature
//! drift. Verification stays lossless either way: a bad drafter only costs
//! speed, never correctness.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const qwen3_cuda = @import("qwen3_cuda.zig");
const cuda = @import("../gpu/cuda.zig");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");
const spec = @import("../llm/spec.zig");
const chat = @import("../llm/chat.zig");
const sample = @import("../llm/sample.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;
const SafeTensors = safetensors.SafeTensors;

const hidden = 2560; // == the 4B target's hidden (fc/embed dims must match)
const cat_dim = 2 * hidden; // (normed embedding | normed hidden)
const fused_dim = 3 * hidden; // three tapped layers
const n_heads = 32;
const kv_heads = 8;
const hd = 128;
const half = hd / 2;
const q_dim = n_heads * hd; // 4096
const kv_dim = kv_heads * hd; // 1024
const intermediate = 9728;
const draft_vocab = 32000;
const rope_theta: f64 = 1000000.0;
const eps: f32 = 1e-6;
const attn_scale: f32 = 1.0 / @sqrt(@as(f32, hd));
const nsplit = 32;

/// Rows per head forward: covers a full verify round's grounding batch;
/// longer (prefill) groundings chunk.
const max_rows = spec.max_draft + 2;

/// Tree drafting (LLM_PLAN.md M8): widest tree level the head forwards in
/// one batch, and the beam-search shape — top candidates per node, each
/// level keeping the `beam` best by cumulative draft log-probability.
/// Deep-narrow (2/2/2) measured best: this head's acceptance concentrates
/// in its top-1, so a tree is worth most as "chain + second-chance
/// siblings", and every 4 verify rows cost one full target weight pass.
const tree_level_cap = 8;
const tree_root_top = 2;
const tree_branch_top = 2;
const tree_beam = 2;

/// Target layers whose incoming residual stream feeds fc, low/mid/high —
/// outputs of layers (2, N/2, N-3) for the 36-layer target (measured best
/// acceptance among the tap conventions on this checkpoint pair).
pub const default_tap_layers: [3]usize = .{ 3, 19, 34 };

pub const Eagle3Head = struct {
    arena: std.heap.ArenaAllocator,
    be: *Backend,
    /// Target embedding matrix (the head has none of its own).
    embed_bytes: []const u8,

    fc: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    lm_head: Weight,
    input_norm: []const f32,
    hidden_norm: []const f32,
    post_norm: []const f32,
    final_norm: []const f32,
    /// draft id -> target id (delta table resolved at load).
    d2t: []const u32,

    capacity: usize,
    /// Head rows cached (row q pairs target-position-q features with token q+1).
    len: usize = 0,
    k_cache: Buf,
    v_cache: Buf,
    freqs_d: Buf,
    b: HeadBufs,

    pub fn load(gpa: std.mem.Allocator, be: *Backend, st: *const SafeTensors, target: *const qwen3.CausalLM, capacity: usize) !Eagle3Head {
        if (target.cfg.hidden != hidden) return error.UnsupportedModelConfig;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: Eagle3Head = undefined;
        self.be = be;
        self.embed_bytes = target.embed.bytes;
        self.capacity = capacity;
        self.len = 0;

        self.fc = try weight(st, "fc.weight", hidden, fused_dim);
        self.q = try weight(st, "midlayer.self_attn.q_proj.weight", q_dim, cat_dim);
        self.k = try weight(st, "midlayer.self_attn.k_proj.weight", kv_dim, cat_dim);
        self.v = try weight(st, "midlayer.self_attn.v_proj.weight", kv_dim, cat_dim);
        self.o = try weight(st, "midlayer.self_attn.o_proj.weight", hidden, q_dim);
        self.gate = try weight(st, "midlayer.mlp.gate_proj.weight", intermediate, hidden);
        self.up = try weight(st, "midlayer.mlp.up_proj.weight", intermediate, hidden);
        self.down = try weight(st, "midlayer.mlp.down_proj.weight", hidden, intermediate);
        self.lm_head = try weight(st, "lm_head.weight", draft_vocab, hidden);
        self.input_norm = try norm(alloc, st, "midlayer.input_layernorm.weight");
        self.hidden_norm = try norm(alloc, st, "midlayer.hidden_norm.weight");
        self.post_norm = try norm(alloc, st, "midlayer.post_attention_layernorm.weight");
        self.final_norm = try norm(alloc, st, "norm.weight");

        // d2t ships as i64 deltas: target_id = draft_id + d2t[draft_id].
        const d2t_view = try st.require("d2t");
        if (d2t_view.info.dtype != .i64 or d2t_view.info.elemCount() != draft_vocab) return error.ShapeMismatch;
        const d2t = try alloc.alloc(u32, draft_vocab);
        for (d2t, 0..) |*t, i| {
            const delta = std.mem.readInt(i64, d2t_view.bytes[i * 8 ..][0..8], .little);
            t.* = @intCast(@as(i64, @intCast(i)) + delta);
            if (t.* >= qwen3.vocab_size) return error.ShapeMismatch;
        }
        self.d2t = d2t;

        var freqs = try ops.rope.rotateHalfFreqs(alloc, capacity + 1, hd, rope_theta);
        defer freqs.deinit(alloc);
        const fp = try alloc.alloc(f32, 2 * (capacity + 1) * half);
        @memcpy(fp[0 .. (capacity + 1) * half], freqs.cos);
        @memcpy(fp[(capacity + 1) * half ..], freqs.sin);
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&self.freqs_d);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        // Rows [capacity, capacity + max_tree_nodes) are the tree batch
        // region: sibling branches collide at the same position, so tree
        // rollout K/V never appends linearly.
        self.k_cache = try be.tensorCreate((capacity + spec.max_tree_nodes) * kv_dim * 4);
        errdefer be.tensorDestroy(&self.k_cache);
        self.v_cache = try be.tensorCreate((capacity + spec.max_tree_nodes) * kv_dim * 4);
        errdefer be.tensorDestroy(&self.v_cache);

        self.b = try HeadBufs.init(be);
        // Moved last: the arena's state mutates on every allocation above,
        // and an early copy would leak everything allocated after it.
        self.arena = arena;
        return self;
    }

    pub fn deinit(self: *Eagle3Head) void {
        self.b.deinit(self.be);
        self.be.tensorDestroy(&self.k_cache);
        self.be.tensorDestroy(&self.v_cache);
        self.be.tensorDestroy(&self.freqs_d);
        self.arena.deinit();
        self.* = undefined;
    }

    /// One decoder-layer forward over `rows` prepared inputs at head rows
    /// [len, len+rows): caller has filled b.feat (fused features for
    /// grounding / previous hidden for rollout) and b.emb (next-token
    /// embeddings). Rope position of row q is q+1 (its input token's
    /// position). Leaves each row's output hidden state in b.feat.
    pub fn forward(self: *Eagle3Head, rows: usize) !void {
        const be = self.be;
        const b = &self.b;
        const p0 = self.len;
        std.debug.assert(rows >= 1 and rows <= max_rows and p0 + rows <= self.capacity);

        try be.qkNorm(b.emb, b.embn, try nbuf(be, self.input_norm), rows, hidden, eps);
        try be.qkNorm(b.feat, b.hidn, try nbuf(be, self.hidden_norm), rows, hidden, eps);
        for (0..rows) |r| {
            try be.opCopyOff(b.cat, r * cat_dim, b.embn, r * hidden, hidden);
            try be.opCopyOff(b.cat, r * cat_dim + hidden, b.hidn, r * hidden, hidden);
        }
        try self.linear(b.q, b.cat, self.q, q_dim, cat_dim, rows);
        try self.linear(b.k, b.cat, self.k, kv_dim, cat_dim, rows);
        try self.linear(b.v, b.cat, self.v, kv_dim, cat_dim, rows);
        try be.ropeHalf(b.q, self.freqs_d, rows, n_heads, half, (self.capacity + 1) * half, p0 + 1);
        try be.ropeHalf(b.k, self.freqs_d, rows, kv_heads, half, (self.capacity + 1) * half, p0 + 1);
        try be.tensorCopy(self.k_cache, p0 * kv_dim * 4, b.k, 0, rows * kv_dim * 4);
        try be.tensorCopy(self.v_cache, p0 * kv_dim * 4, b.v, 0, rows * kv_dim * 4);
        try be.opAttnDecode(b.q, self.k_cache, self.v_cache, b.attn, b.attn_scratch, p0 + 1, rows, n_heads, kv_heads, hd, nsplit, attn_scale, 0, 0, false, .f32);
        try self.linear(b.t, b.attn, self.o, hidden, q_dim, rows);
        try be.opAdd(b.feat, b.t, rows * hidden);

        try be.qkNorm(b.feat, b.normed, try nbuf(be, self.post_norm), rows, hidden, eps);
        try self.linear(b.gate, b.normed, self.gate, intermediate, hidden, rows);
        try self.linear(b.up, b.normed, self.up, intermediate, hidden, rows);
        try be.siluMul(b.gate, b.up, rows * intermediate);
        try self.linear(b.t, b.gate, self.down, hidden, intermediate, rows);
        try be.opAdd(b.feat, b.t, rows * hidden);
        self.len = p0 + rows;
    }

    /// Greedy draft from row `row`'s hidden output: 32k draft logits ->
    /// argmax -> d2t -> target token id.
    pub fn draftToken(self: *Eagle3Head, row: usize, logits: []f32) !u32 {
        const be = self.be;
        const b = &self.b;
        try be.qkNorm(offsetBufSized(b.feat, row * hidden * 4, hidden * 4), b.t, try nbuf(be, self.final_norm), 1, hidden, eps);
        try be.opGemvBf16(b.logits, b.t, self.lm_head.bytes, 1.0, draft_vocab, hidden);
        try be.tensorDownload(offsetBufSized(b.logits, 0, draft_vocab * 4), std.mem.sliceAsBytes(logits[0..draft_vocab]));
        return self.d2t[sample.argmax(logits[0..draft_vocab])];
    }

    pub fn truncate(self: *Eagle3Head, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    /// One tree-LEVEL forward (tree drafting): `rows` draft nodes, all at
    /// depth `depth` (rope position len + depth), K/V stored at batch rows
    /// capacity + first_node.. (nodes of a level are contiguous), attention
    /// over the grounded prefix plus each row's draft-ancestor chain
    /// (`meta`: [rows][rows+1] u32 rows of kv_len + ancestor node indices,
    /// as attn_split_tree expects). The caller has filled b.feat (parent
    /// hidden states) and b.emb (node token embeddings). Row hidden outputs
    /// land in b.feat and node_hidden rows first_node..; draft logits for
    /// every row land in b.logits. self.len is untouched.
    pub fn forwardTreeLevel(self: *Eagle3Head, first_node: usize, rows: usize, depth: usize, meta: []const u32) !void {
        const be = self.be;
        const b = &self.b;
        const p0 = self.len;
        std.debug.assert(rows >= 1 and rows <= tree_level_cap);
        std.debug.assert(first_node >= 1 and first_node + rows <= spec.max_tree_nodes);
        std.debug.assert(meta.len == rows * (rows + 1) and p0 + depth <= self.capacity);

        var pos: [tree_level_cap]u32 = undefined;
        for (pos[0..rows]) |*p| p.* = @intCast(p0 + depth);
        try be.tensorUpload(offsetBufSized(b.tree_pos, 0, rows * 4), std.mem.sliceAsBytes(pos[0..rows]));
        const meta_off = rows * n_heads * nsplit * (hd + 4);
        try be.tensorUpload(offsetBufSized(b.tree_scratch, meta_off * 4, meta.len * 4), std.mem.sliceAsBytes(meta));

        try be.qkNorm(b.emb, b.embn, try nbuf(be, self.input_norm), rows, hidden, eps);
        try be.qkNorm(b.feat, b.hidn, try nbuf(be, self.hidden_norm), rows, hidden, eps);
        for (0..rows) |r| {
            try be.opCopyOff(b.cat, r * cat_dim, b.embn, r * hidden, hidden);
            try be.opCopyOff(b.cat, r * cat_dim + hidden, b.hidn, r * hidden, hidden);
        }
        try self.linear(b.q, b.cat, self.q, q_dim, cat_dim, rows);
        try self.linear(b.k, b.cat, self.k, kv_dim, cat_dim, rows);
        try self.linear(b.v, b.cat, self.v, kv_dim, cat_dim, rows);
        try be.opRopeHalfPos(b.q, b.tree_pos, self.freqs_d, rows, n_heads, half, (self.capacity + 1) * half);
        try be.opRopeHalfPos(b.k, b.tree_pos, self.freqs_d, rows, kv_heads, half, (self.capacity + 1) * half);
        try be.tensorCopy(self.k_cache, (self.capacity + first_node) * kv_dim * 4, b.k, 0, rows * kv_dim * 4);
        try be.tensorCopy(self.v_cache, (self.capacity + first_node) * kv_dim * 4, b.v, 0, rows * kv_dim * 4);
        try be.opAttnDecodeTree(b.q, self.k_cache, self.v_cache, b.attn, b.tree_scratch, p0, self.capacity, rows, n_heads, kv_heads, hd, nsplit, attn_scale);
        try self.linear(b.t, b.attn, self.o, hidden, q_dim, rows);
        try be.opAdd(b.feat, b.t, rows * hidden);

        try be.qkNorm(b.feat, b.normed, try nbuf(be, self.post_norm), rows, hidden, eps);
        try self.linear(b.gate, b.normed, self.gate, intermediate, hidden, rows);
        try self.linear(b.up, b.normed, self.up, intermediate, hidden, rows);
        try be.siluMul(b.gate, b.up, rows * intermediate);
        try self.linear(b.t, b.gate, self.down, hidden, intermediate, rows);
        try be.opAdd(b.feat, b.t, rows * hidden);

        try be.opCopyOff(b.node_hidden, first_node * hidden, b.feat, 0, rows * hidden);
        try self.logitsFromFeat(0, rows);
    }

    /// Draft logits for b.feat rows [row0, row0+rows) into b.logits rows
    /// [0, rows): final norm + the reduced 32k LM head in 4-input groups.
    pub fn logitsFromFeat(self: *Eagle3Head, row0: usize, rows: usize) !void {
        const be = self.be;
        const b = &self.b;
        std.debug.assert(rows >= 1 and rows <= tree_level_cap and row0 + rows <= max_rows);
        try be.qkNorm(offsetBufSized(b.feat, row0 * hidden * 4, rows * hidden * 4), b.t, try nbuf(be, self.final_norm), rows, hidden, eps);
        var off: usize = 0;
        while (off < rows) : (off += 4) {
            const g: usize = @min(4, rows - off); // annotated: @min would narrow
            try be.opGemvBf16N(
                offsetBufSized(b.logits, off * draft_vocab * 4, g * draft_vocab * 4),
                offsetBufSized(b.t, off * hidden * 4, 4 * hidden * 4),
                self.lm_head.bytes,
                self.lm_head.scale,
                draft_vocab,
                hidden,
                g,
            );
        }
    }

    /// bf16 linear, grouped 4-input GEMVs (the head never sees big batches).
    fn linear(self: *Eagle3Head, y: Buf, x: Buf, w: Weight, rows_out: usize, cols: usize, seq: usize) !void {
        const be = self.be;
        if (seq == 1) {
            try be.opGemvBf16(y, x, w.bytes, w.scale, rows_out, cols);
            return;
        }
        var off: usize = 0;
        while (off < seq) : (off += 4) {
            const n: usize = @min(4, seq - off); // annotated: @min would narrow to u3
            try be.opGemvBf16N(
                offsetBufSized(y, off * rows_out * 4, n * rows_out * 4),
                offsetBufSized(x, off * cols * 4, 4 * cols * 4),
                w.bytes,
                w.scale,
                rows_out,
                cols,
                n,
            );
        }
    }

    fn weight(st: *const SafeTensors, name: []const u8, rows: usize, cols: usize) !Weight {
        const view = st.get(name) orelse return error.MissingTensor;
        const shape = view.info.shape.slice();
        if (view.info.dtype != .bf16 or shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
        return Weight.init(view.bytes, .bf16, rows, cols);
    }

    fn norm(alloc: std.mem.Allocator, st: *const SafeTensors, name: []const u8) ![]f32 {
        const view = st.get(name) orelse return error.MissingTensor;
        if (view.info.elemCount() != hidden) return error.ShapeMismatch;
        return view.toF32Alloc(alloc);
    }
};

const HeadBufs = struct {
    feat: Buf, // fused feature in / hidden out, [max_rows][hidden]
    fc_in: Buf, // packed 3-tap fusion input, [max_rows][3*hidden]
    emb: Buf,
    embn: Buf,
    hidn: Buf,
    cat: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    normed: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    attn_scratch: Buf,
    logits: Buf,
    /// Tree drafting: per-node hidden outputs (rollout inputs), uniform
    /// per-level rope positions, and the tree-attention scratch (+ meta tail).
    node_hidden: Buf,
    tree_pos: Buf,
    tree_scratch: Buf,

    fn init(be: *Backend) !HeadBufs {
        const r4 = std.mem.alignForward(usize, max_rows, 4); // grouped-GEMV inputs read 4 rows at a time
        var self: HeadBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        const sizes = [fields.len]usize{
            r4 * hidden * 4, // feat (rollout copies from arbitrary rows)
            r4 * fused_dim * 4, // fc_in
            max_rows * hidden * 4, // emb
            max_rows * hidden * 4, // embn
            max_rows * hidden * 4, // hidn
            r4 * cat_dim * 4, // cat
            r4 * q_dim * 4, // q (attn out feeds o_proj via b.attn)
            max_rows * kv_dim * 4, // k
            max_rows * kv_dim * 4, // v
            r4 * q_dim * 4, // attn
            r4 * hidden * 4, // normed
            r4 * intermediate * 4, // gate
            max_rows * intermediate * 4, // up
            max_rows * hidden * 4, // t
            max_rows * n_heads * nsplit * (hd + 4) * 4, // attn_scratch
            tree_level_cap * draft_vocab * 4, // logits (a row per tree-level node)
            spec.max_tree_nodes * hidden * 4, // node_hidden
            tree_level_cap * 4, // tree_pos
            (tree_level_cap * n_heads * nsplit * (hd + 4) + tree_level_cap * (tree_level_cap + 1)) * 4, // tree_scratch
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *HeadBufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "feat", "fc_in", "emb", "embn", "hidn", "cat", "q", "k", "v", "attn", "normed", "gate", "up", "t", "attn_scratch", "logits", "node_hidden", "tree_pos", "tree_scratch" };
};

/// offsetBuf carrying an explicit size (tensorUpload/Download use db.size).
fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// EAGLE-3 drafter for spec.generate: propose(ids, buf). Head row q is
/// "grounded" when built from the target's features at position q paired
/// with committed token q+1; rollout rows (built from the head's own hidden
/// states) are re-grounded next round once the verify forward supplies real
/// features. Degrades to proposing nothing on any failure.
pub const Eagle3Drafter = struct {
    head: *Eagle3Head,
    target: *qwen3_cuda.CudaLM,
    gpa: std.mem.Allocator,
    hist: std.ArrayList(u32) = .empty,
    /// Head rows [0, grounded) are feature-grounded.
    grounded: usize = 0,
    logits: []f32,
    emb_row: []f32,

    pub fn init(gpa: std.mem.Allocator, head: *Eagle3Head, target: *qwen3_cuda.CudaLM) !Eagle3Drafter {
        std.debug.assert(target.taps_on);
        const logits = try gpa.alloc(f32, tree_level_cap * draft_vocab);
        errdefer gpa.free(logits);
        const emb_row = try gpa.alloc(f32, max_rows * hidden);
        return .{ .head = head, .target = target, .gpa = gpa, .logits = logits, .emb_row = emb_row };
    }

    pub fn deinit(self: *Eagle3Drafter) void {
        self.hist.deinit(self.gpa);
        self.gpa.free(self.logits);
        self.gpa.free(self.emb_row);
    }

    pub fn propose(self: *Eagle3Drafter, ids: []const u32, buf: []u32) usize {
        return self.proposeInner(ids, buf) catch 0;
    }

    fn proposeInner(self: *Eagle3Drafter, ids: []const u32, buf: []u32) !usize {
        const head = self.head;
        const be = head.be;
        if (buf.len == 0 or ids.len < 2) return 0;
        var rows_in_buf = try self.ground(ids);
        if (rows_in_buf == 0) return 0; // nothing new grounded: no fresh tip hidden

        // Draft 1 comes from the grounded tip; rollout feeds the head its own
        // hidden states.
        var m: usize = 0;
        while (m < buf.len) {
            const tok = try head.draftToken(rows_in_buf - 1, self.logits);
            buf[m] = tok;
            m += 1;
            if (m == buf.len or chat.isStop(tok) or head.len >= head.capacity) break;
            // Next rollout row: feature = previous hidden, input token = tok.
            if (rows_in_buf > 1) try be.tensorCopy(head.b.feat, 0, head.b.feat, (rows_in_buf - 1) * hidden * 4, hidden * 4);
            try self.uploadEmbeds(&.{tok});
            try head.forward(1);
            rows_in_buf = 1;
        }
        return m;
    }

    /// Re-sync the head with the committed ids and ground every position's
    /// head row from the target's tap features; returns the row count of
    /// the freshest forward (b.feat occupancy — the tip hidden lives at row
    /// count-1), or 0 when no new row was grounded (no fresh tip to draft
    /// from) or the target hasn't forwarded the features yet.
    fn ground(self: *Eagle3Drafter, ids: []const u32) !usize {
        const head = self.head;
        const be = head.be;

        // Drop head rows invalidated by rejections (row q pairs feature q
        // with token q+1) plus any un-grounded rollout tail.
        const cp = commonPrefix(self.hist.items, ids);
        const valid = if (cp == 0) 0 else @min(self.grounded, cp - 1);
        head.truncate(valid);
        self.hist.clearRetainingCapacity();
        try self.hist.appendSlice(self.gpa, ids);

        // Ground rows [valid, ids.len-2]: target features exist for every
        // forwarded position (everything but the pending token).
        const want = ids.len - 1; // rows after grounding
        if (want > self.target.cached()) return 0; // feature not forwarded yet
        if (want > head.capacity) return 0;
        var rows_in_buf: usize = 0; // rows of the most recent forward (b.feat occupancy)
        while (head.len < want) {
            const a = head.len;
            const rows = @min(max_rows, want - a);
            // fc inputs: pack the 3 tap rows per position into [rows][3*hidden].
            for (0..3) |j| {
                for (0..rows) |r| {
                    try be.opCopyOff(head.b.fc_in, r * fused_dim + j * hidden, self.target.tap_d, (j * self.target.capacity + a + r) * hidden, hidden);
                }
            }
            try head.linear(head.b.feat, head.b.fc_in, head.fc, hidden, fused_dim, rows);
            try self.uploadEmbeds(ids[a + 1 ..][0..rows]);
            try head.forward(rows);
            rows_in_buf = rows;
        }
        self.grounded = head.len;
        return rows_in_buf;
    }

    /// Tree proposal (spec.generateTree, LLM_PLAN.md M8): level-synchronous
    /// beam expansion. The root's top tree_root_top candidates seed level 1;
    /// each level is forwarded through the head in ONE batch (uniform depth,
    /// tree attention over the grounded prefix + draft-ancestor chains),
    /// its rows' top tree_branch_top continuations are ranked by cumulative
    /// draft log-probability, and the best tree_beam become the next level.
    /// The deepest level is never forwarded (its logits would go unread).
    /// Slice index i is node i+1; parent values are node indices (0 = root).
    pub fn proposeTree(self: *Eagle3Drafter, ids: []const u32, tokens: []u32, parents: []u32, max_depth: usize) usize {
        return self.proposeTreeInner(ids, tokens, parents, max_depth) catch 0;
    }

    fn proposeTreeInner(self: *Eagle3Drafter, ids: []const u32, tokens: []u32, parents: []u32, max_depth: usize) !usize {
        const head = self.head;
        const be = head.be;
        if (tokens.len == 0 or ids.len < 2 or max_depth == 0) return 0;
        const budget = @min(tokens.len, spec.max_tree_nodes - 1);
        const rows_in_buf = try self.ground(ids);
        if (rows_in_buf == 0) return 0;
        if (head.len + max_depth > head.capacity) return 0;

        // Root: hidden feeds level 1's rollout rows; logits seed the beam.
        try be.opCopyOff(head.b.node_hidden, 0, head.b.feat, (rows_in_buf - 1) * hidden, hidden);
        try head.logitsFromFeat(rows_in_buf - 1, 1);
        try be.tensorDownload(offsetBufSized(head.b.logits, 0, draft_vocab * 4), std.mem.sliceAsBytes(self.logits[0..draft_vocab]));

        // Per-node bookkeeping (node 0 = root; slice index i is node i+1).
        var node_parent: [spec.max_tree_nodes]u32 = undefined;
        var cum_lp: [spec.max_tree_nodes]f32 = undefined;
        var level: [tree_level_cap]u32 = undefined; // node ids of the current level
        var level_n: usize = 0;
        var m: usize = 0;
        node_parent[0] = 0;
        cum_lp[0] = 0;

        // Level 1: the root's widest fan-out.
        {
            const want = @min(tree_root_top, budget);
            var ct: [tree_root_top]u32 = undefined;
            var cl: [tree_root_top]f32 = undefined;
            const c = topNLogits(self.logits[0..draft_vocab], ct[0..want], cl[0..want]);
            for (ct[0..c], cl[0..c]) |t, lp| {
                const node: u32 = @intCast(m + 1);
                tokens[m] = head.d2t[t];
                parents[m] = 0;
                node_parent[node] = 0;
                cum_lp[node] = lp;
                level[level_n] = node;
                level_n += 1;
                m += 1;
            }
        }

        var depth: usize = 1;
        while (level_n > 0 and depth < max_depth and m < budget) {
            // Forward the level: feat rows = parent hidden, emb = node tokens.
            var level_tokens: [tree_level_cap]u32 = undefined;
            for (level[0..level_n], 0..) |node, r| {
                level_tokens[r] = tokens[node - 1];
                try be.opCopyOff(head.b.feat, r * hidden, head.b.node_hidden, node_parent[node] * hidden, hidden);
            }
            try self.uploadEmbeds(level_tokens[0..level_n]);
            // Attention meta: kv_len = grounded prefix + depth; the ancestor
            // list holds only DRAFT nodes (the root's row is the grounded
            // tip, already inside the prefix), depth order.
            var meta: [tree_level_cap * (tree_level_cap + 1)]u32 = undefined;
            for (level[0..level_n], 0..) |node, r| {
                meta[r * (level_n + 1)] = @intCast(head.len + depth);
                var j = node;
                var d = depth;
                while (j != 0) {
                    d -= 1;
                    meta[r * (level_n + 1) + 1 + d] = j;
                    j = node_parent[j];
                }
                std.debug.assert(d == 0);
            }
            try head.forwardTreeLevel(level[0], level_n, depth, meta[0 .. level_n * (level_n + 1)]);
            try be.tensorDownload(offsetBufSized(head.b.logits, 0, level_n * draft_vocab * 4), std.mem.sliceAsBytes(self.logits[0 .. level_n * draft_vocab]));

            // Rank all continuations by cumulative log-probability.
            const Cand = struct { parent: u32, tok: u32, lp: f32 };
            var cands: [tree_level_cap * tree_branch_top]Cand = undefined;
            var nc: usize = 0;
            for (level[0..level_n], 0..) |node, r| {
                if (chat.isStop(tokens[node - 1])) continue; // never expand past a stop
                var ct: [tree_branch_top]u32 = undefined;
                var cl: [tree_branch_top]f32 = undefined;
                const c = topNLogits(self.logits[r * draft_vocab ..][0..draft_vocab], &ct, &cl);
                for (ct[0..c], cl[0..c]) |t, lp| {
                    cands[nc] = .{ .parent = node, .tok = head.d2t[t], .lp = cum_lp[node] + lp };
                    nc += 1;
                }
            }
            if (nc == 0) break;
            std.mem.sort(Cand, cands[0..nc], {}, struct {
                fn gt(_: void, a: Cand, b: Cand) bool {
                    return a.lp > b.lp;
                }
            }.gt);

            const keep = @min(@min(tree_beam, nc), budget - m);
            level_n = 0;
            for (cands[0..keep]) |cand| {
                const node: u32 = @intCast(m + 1);
                tokens[m] = cand.tok;
                parents[m] = cand.parent;
                node_parent[node] = cand.parent;
                cum_lp[node] = cand.lp;
                level[level_n] = node;
                level_n += 1;
                m += 1;
            }
            depth += 1;
        }
        return m;
    }

    /// Top-|out| draft-vocab candidates by logit, converted to
    /// log-probabilities (log-softmax over the row). Strict-greater
    /// insertion keeps the lowest index on ties, matching argmax — the
    /// beam's best candidate is exactly the chain drafter's greedy pick.
    fn topNLogits(row: []const f32, out_tok: []u32, out_lp: []f32) usize {
        std.debug.assert(out_tok.len == out_lp.len);
        const n = @min(out_tok.len, row.len);
        var count: usize = 0;
        for (row, 0..) |v, i| {
            if (count == n and v <= out_lp[n - 1]) continue;
            var j = if (count < n) count else n - 1;
            while (j > 0 and out_lp[j - 1] < v) : (j -= 1) {
                out_lp[j] = out_lp[j - 1];
                out_tok[j] = out_tok[j - 1];
            }
            out_lp[j] = v;
            out_tok[j] = @intCast(i);
            if (count < n) count += 1;
        }
        var mx = -std.math.inf(f32);
        for (row) |v| mx = @max(mx, v);
        var se: f32 = 0;
        for (row) |v| se += @exp(v - mx);
        const lse = mx + @log(se);
        for (out_lp[0..count]) |*lp| lp.* -= lse;
        return count;
    }

    fn uploadEmbeds(self: *Eagle3Drafter, tokens: []const u32) !void {
        const head = self.head;
        for (tokens, 0..) |id, r| {
            if (id >= qwen3.vocab_size) return error.TokenIdOutOfRange;
            const row = head.embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
            try safetensors.convertToF32(.bf16, row, self.emb_row[r * hidden ..][0..hidden]);
        }
        try head.be.tensorUpload(offsetBufSized(head.b.emb, 0, tokens.len * hidden * 4), std.mem.sliceAsBytes(self.emb_row[0 .. tokens.len * hidden]));
    }

    fn commonPrefix(a: []const u32, b: []const u32) usize {
        const n = @min(a.len, b.len);
        for (0..n) |i| {
            if (a[i] != b[i]) return i;
        }
        return n;
    }
};

// Gated on the GPU marker + both checkpoints: EAGLE-drafted greedy output
// must equal vanilla greedy (the lossless-verification contract holds no
// matter what the head proposes). Kept tiny — Debug forwards are slow.
test "eagle drafter matches vanilla greedy on the real models" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const tokenizer_mod = @import("../tokenizer.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    const head_path = "models/text_encoders/qwen3_4b_eagle3.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, head_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var est = try SafeTensors.open(gpa, io, head_path);
    defer est.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
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
        var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)), ids_vanilla.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.spec_k = 2;
        const cap = try engine.capacityFor(opts, ids_spec.items.len);
        var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, .fixed(cap), ids_spec.items.len);
        defer model.deinit();
        try model.enableTaps(default_tap_layers);
        var head = try Eagle3Head.load(gpa, be, &est, &lm, cap);
        defer head.deinit();
        var drafter = try Eagle3Drafter.init(gpa, &head, &model);
        defer drafter.deinit();
        _ = try spec.generate(&model, &drafter, &tok, io, gpa, &ids_spec, opts, null);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_spec.items);
}

// Gated like the chain test above: EAGLE TREE drafting (proposeTree beam +
// the target's tree-verify forward) stays byte-identical to vanilla greedy —
// the lossless contract holds no matter what tree the head proposes.
test "eagle tree drafter matches vanilla greedy on the real models" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const tokenizer_mod = @import("../tokenizer.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    const head_path = "models/text_encoders/qwen3_4b_eagle3.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, head_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var est = try SafeTensors.open(gpa, io, head_path);
    defer est.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    var opts: engine.Options = .{ .max_new_tokens = 4, .sampling = .{ .temperature = 0 } };
    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Say hi.", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_tree: std.ArrayList(u32) = .empty;
    defer ids_tree.deinit(gpa);
    try ids_tree.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)), ids_vanilla.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.tree_nodes = 16;
        var stats: spec.Stats = .{};
        opts.spec_stats = &stats;
        const cap = try engine.capacityFor(opts, ids_tree.items.len);
        var model = try qwen3_cuda.CudaLM.init(gpa, be, &lm, .fixed(cap), ids_tree.items.len);
        defer model.deinit();
        try model.enableTaps(default_tap_layers);
        try model.enableTree();
        var head = try Eagle3Head.load(gpa, be, &est, &lm, cap);
        defer head.deinit();
        var drafter = try Eagle3Drafter.init(gpa, &head, &model);
        defer drafter.deinit();
        _ = try spec.generate(&model, &drafter, &tok, io, gpa, &ids_tree, opts, null);
        try std.testing.expect(stats.forwards > 0);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_tree.items);
}
