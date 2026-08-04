//! CLIP text encoder — the SD-family conditioner (`cond_stage_model` in an LDM
//! single-file checkpoint), mirroring `transformers.CLIPTextModel`.
//!
//! A plain pre-LN transformer, but four details are load-bearing and each one
//! produces a *plausible* wrong conditioning rather than an obvious failure:
//!
//! 1. **The attention is causal.** A text encoder that reads bidirectionally is a
//!    different model; every prompt still encodes, and every image still renders.
//! 2. **The MLP activation is `quick_gelu`** (`x·σ(1.702x)`), not tanh-gelu and not
//!    erf-gelu. CLIP-L (SD1.5) uses quick_gelu; CLIP-G (SDXL's second tower) uses
//!    the erf form — `Config.act` carries which.
//! 3. **Positions are a learned table**, not RoPE and not sinusoidal, and the table
//!    is exactly `max_positions` long: SD's 77-token window is a property of the
//!    weights, not a convention we choose.
//! 4. **The output SD conditions on is `final_layer_norm(x)`** — the last hidden
//!    state, layer-normed. Skipping that norm leaves activations off by a scale the
//!    UNet's cross-attention silently absorbs into softer images.
//!
//! Loads from any `WeightStore` and keeps large weights in their checkpoint dtype
//! (so a quantized text tower runs, and so `ops.matmul.probe` can attribute its
//! GEMMs — ggufy quantizes this encoder today and could not measure it before).
//! The token embedding is the exception: it is a lookup table, never a GEMM, and is
//! read a row at a time.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const DType = tp_core.dtype.DType;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    hidden: usize,
    layers: usize,
    heads: usize,
    intermediate: usize,
    max_positions: usize,
    eps: f32,
    act: Act,
    /// Token id whose row becomes the pooled output. For SD's CLIP vocabularies
    /// `<|endoftext|>`; also the padding token for CLIP-L, so "first occurrence" is
    /// what selects the real end of the prompt.
    eos_id: u32,
    /// Which spelling the checkpoint stores this tower under. See `Naming`.
    naming: Naming = .hf,
    /// Whether a `text_projection` matrix is present — CLIP-G's, which turns its
    /// pooled row into the vector SDXL's `y` carries. CLIP-L has none in SD
    /// checkpoints (nothing consumes it).
    projection: bool = false,

    pub const Act = enum { quick_gelu, gelu_erf };

    /// ⚠️ **The two towers of one SDXL checkpoint are stored under different
    /// conventions**, and this is not a quirk of one file — it is what the ecosystem
    /// ships, because CLIP-L came through `transformers` and CLIP-G through
    /// `open_clip`:
    ///
    /// | | `.hf` (CLIP-L) | `.open_clip` (CLIP-G) |
    /// |---|---|---|
    /// | block | `encoder.layers.N` | `transformer.resblocks.N` |
    /// | norms | `layer_norm1` / `layer_norm2` | `ln_1` / `ln_2` |
    /// | MLP | `mlp.fc1` / `mlp.fc2` | `mlp.c_fc` / `mlp.c_proj` |
    /// | q/k/v | three `self_attn.{q,k,v}_proj` | **one fused `attn.in_proj_weight`** |
    /// | positions | `embeddings.position_embedding.weight` | `positional_embedding` |
    /// | final norm | `final_layer_norm` | `ln_final` |
    ///
    /// The fused `[3·hidden, hidden]` matrix is the load-bearing difference: q, k and v
    /// are its three row blocks, in that order.
    pub const Naming = enum { hf, open_clip };

    pub fn headDim(self: Config) usize {
        return self.hidden / self.heads;
    }
};

/// SD1.5's conditioner: ViT-L/14's text tower.
pub const clip_l: Config = .{
    .hidden = 768,
    .layers = 12,
    .heads = 12,
    .intermediate = 3072,
    .max_positions = 77,
    .eps = 1e-5,
    .act = .quick_gelu,
    .eos_id = 49407,
};

/// SDXL's second tower (OpenCLIP ViT-bigG/14): wider, deeper, erf-GELU instead of
/// quick-GELU, stored under OpenCLIP's names with fused q/k/v, and carrying the
/// `text_projection` that produces SDXL's pooled conditioning.
pub const clip_g: Config = .{
    .hidden = 1280,
    .layers = 32,
    .heads = 20,
    .intermediate = 5120,
    .max_positions = 77,
    .eps = 1e-5,
    .act = .gelu_erf,
    .eos_id = 49407,
    .naming = .open_clip,
    .projection = true,
};

const Layer = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    q: Weight,
    q_b: []const f32,
    k: Weight,
    k_b: []const f32,
    v: Weight,
    v_b: []const f32,
    o: Weight,
    o_b: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    fc1: Weight,
    fc1_b: []const f32,
    fc2: Weight,
    fc2_b: []const f32,
};

pub const TextEncoder = struct {
    /// **This tower can apply per-token prompt weights.** Read by `pipeline` instead of a
    /// list of architecture names, so the capability cannot drift from the encoder that
    /// has it, and a newly added encoder has to state its answer or fail to compile.
    ///
    /// What makes it true here, and these are the actual requirements rather than "it is
    /// CLIP": the conditioning is a **fixed 77-slot window** whose rows correspond 1:1 to
    /// prompt tokens, the denoiser cross-attends to those rows, and an **empty-prompt
    /// encode of the same shape** exists to interpolate against (`applyWeights`). An
    /// encoder producing a variable-length LLM tap state has none of the three.
    pub const supports_prompt_weights = true;

    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// The embedding table, kept as a view: `[vocab][hidden]`, read one row per
    /// token. Materializing it would cost 151 MB of f32 for CLIP-L to serve 77 rows.
    tok_embed: safetensors.TensorView,
    vocab: usize,
    pos_embed: []const f32,
    layers: []Layer,
    final_ln_w: []const f32,
    final_ln_b: []const f32,
    /// CLIP-G's `text_projection`, `[hidden][hidden]` and applied as `x @ T` (see
    /// `projectPooled`). Null unless `cfg.projection`.
    text_projection: ?[]const f32,

    /// `prefix` is where the text model sits in the store. For an LDM single-file
    /// checkpoint that is `cond_stage_model.transformer.text_model.`; for a bare
    /// HF export, `text_model.`.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config, prefix: []const u8) !TextEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const l: Loader = .{ .store = store, .alloc = alloc, .pfx = prefix, .cfg = cfg };

        const emb_name = try std.fmt.allocPrint(alloc, "{s}{s}", .{ prefix, switch (cfg.naming) {
            .hf => "embeddings.token_embedding.weight",
            .open_clip => "token_embedding.weight",
        } });
        const tok_view = store.get(emb_name) orelse {
            std.log.err("clip_text: missing {s}", .{emb_name});
            return error.MissingTensor;
        };
        const emb_shape = tok_view.info.shape.slice();
        if (emb_shape.len != 2 or emb_shape[1] != cfg.hidden) {
            std.log.err("clip_text: {s} has shape {any}, expected [vocab, {d}]", .{ emb_name, emb_shape, cfg.hidden });
            return error.ShapeMismatch;
        }

        const layers = try alloc.alloc(Layer, cfg.layers);
        for (layers, 0..) |*layer, i| {
            layer.* = switch (cfg.naming) {
                .hf => try l.layerHf(i),
                .open_clip => try l.layerOpenClip(i),
            };
        }

        // ⚠️ Every allocation must happen BEFORE the struct literal. `.arena = arena`
        // copies the arena's state by value, so an allocation made by a *later* field
        // initializer lands in the local arena's chunk list and is invisible to the
        // copy the caller will `deinit()` — a leak the test allocator catches and
        // nothing else would. Same reason `dit.zig` builds its fields up front.
        const pos_embed = switch (cfg.naming) {
            .hf => try l.vec("embeddings.position_embedding.weight", .{}, cfg.max_positions * cfg.hidden),
            .open_clip => try l.vec("positional_embedding", .{}, cfg.max_positions * cfg.hidden),
        };
        const final_ln_w = switch (cfg.naming) {
            .hf => try l.vec("final_layer_norm.weight", .{}, cfg.hidden),
            .open_clip => try l.vec("ln_final.weight", .{}, cfg.hidden),
        };
        const final_ln_b = switch (cfg.naming) {
            .hf => try l.vec("final_layer_norm.bias", .{}, cfg.hidden),
            .open_clip => try l.vec("ln_final.bias", .{}, cfg.hidden),
        };
        const text_projection: ?[]const f32 = if (cfg.projection)
            try l.vec("text_projection", .{}, cfg.hidden * cfg.hidden)
        else
            null;

        return .{
            .arena = arena,
            .cfg = cfg,
            .tok_embed = tok_view,
            .vocab = emb_shape[0],
            .pos_embed = pos_embed,
            .layers = layers,
            .final_ln_w = final_ln_w,
            .final_ln_b = final_ln_b,
            .text_projection = text_projection,
        };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// `out` is `[ids.len][hidden]`, the final hidden state after
    /// `final_layer_norm` — what SD's cross-attention consumes.
    ///
    /// `hidden_out`, when given, receives the same-shaped activation at
    /// `capture_layer`: 0 = the embedding output, n = after encoder layer n−1. That
    /// is what makes a mismatch localize to a layer instead of just failing at the
    /// end, and it is also how "CLIP skip" would be implemented.
    pub fn forward(
        self: *const TextEncoder,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        ids: []const u32,
        capture: ?Capture,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        if (seq == 0 or seq > cfg.max_positions) return error.InvalidSequenceLength;
        std.debug.assert(out.len == seq * cfg.hidden);

        // Embedding lookup + learned positional table, straight into `out`, which is
        // the residual stream for the whole forward.
        const row_bytes = self.tok_embed.info.dtype.storageBytes(cfg.hidden);
        for (ids, 0..) |id, p| {
            if (id >= self.vocab) return error.TokenOutOfRange;
            const src = self.tok_embed.bytes[id * row_bytes ..][0..row_bytes];
            const dst = out[p * cfg.hidden ..][0..cfg.hidden];
            try safetensors.convertToF32(self.tok_embed.info.dtype, src, dst);
            for (dst, self.pos_embed[p * cfg.hidden ..][0..cfg.hidden]) |*o, pe| o.* += pe;
        }
        if (capture) |c| if (c.layer == 0) @memcpy(c.dst, out);

        const hd = cfg.headDim();
        const norm_buf = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(norm_buf);
        const q = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(v);
        const attn_out = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(attn_out);
        const ff = try gpa.alloc(f32, seq * cfg.intermediate);
        defer gpa.free(ff);

        for (self.layers, 0..) |layer, li| {
            // --- attention block: x = x + o(attn(ln1(x))) ---
            ops.norm.layerNorm(norm_buf, out, layer.ln1_w, layer.ln1_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, q, norm_buf, seq, layer.q, layer.q_b);
            try ops.matmul.matmul(io, gpa, k, norm_buf, seq, layer.k, layer.k_b);
            try ops.matmul.matmul(io, gpa, v, norm_buf, seq, layer.v, layer.v_b);
            try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
                .seq_q = seq,
                .seq_kv = seq,
                .n_heads = cfg.heads,
                .n_kv_heads = cfg.heads,
                .head_dim = hd,
                // The one non-obvious flag in this file. CLIP's text tower is a
                // causal LM body used as an encoder.
                .causal = true,
            });
            try ops.matmul.matmul(io, gpa, norm_buf, attn_out, seq, layer.o, layer.o_b);
            for (out, norm_buf) |*o, r| o.* += r;

            // --- mlp block: x = x + fc2(act(fc1(ln2(x)))) ---
            ops.norm.layerNorm(norm_buf, out, layer.ln2_w, layer.ln2_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, ff, norm_buf, seq, layer.fc1, layer.fc1_b);
            switch (cfg.act) {
                .quick_gelu => ops.act.geluQuick(ff),
                .gelu_erf => ops.act.geluErf(ff),
            }
            try ops.matmul.matmul(io, gpa, norm_buf, ff, seq, layer.fc2, layer.fc2_b);
            for (out, norm_buf) |*o, r| o.* += r;

            if (capture) |c| if (c.layer == li + 1) @memcpy(c.dst, out);
        }

        ops.norm.layerNorm(out, out, self.final_ln_w, self.final_ln_b, cfg.eps);
    }

    /// How to run one tower over a whole multi-chunk prompt. A struct rather than five
    /// positional arguments because SDXL's two towers differ in four of them.
    /// How a prompt's attention weights become changes to the hidden states. The two
    /// dialects disagree, and the difference is visible: see `applyWeights` (ComfyUI's
    /// interpolate-away-from-empty) versus `applyWeightsA1111` (multiply, optionally
    /// rescaled to preserve the chunk mean).
    pub const WeightMode = enum {
        comfy,
        a1111_original,
        a1111_no_norm,

        /// Whether the empty-prompt reference has to be computed at all. Only ComfyUI's
        /// form needs it — so an A1111 render skips that tower forward entirely.
        pub fn needsEmpty(self: WeightMode) bool {
            return self == .comfy;
        }
    };

    pub const PromptRun = struct {
        /// Which dialect's weighting to apply.
        mode: WeightMode = .comfy,
        /// Caller-owned slot holding this tower's `z_empty`, filled on first need and
        /// reused after. See `applyWeights` for what it is.
        empty_cache: *?[]f32,
        /// Trailing filler for a short chunk *and* for the empty reference sequence —
        /// EOS for CLIP-L, 0 for CLIP-G.
        pad_id: u32,
        /// Null when the conditioning is this tower's final output (SD1.5); otherwise
        /// the layer to capture (SDXL's penultimate, final LayerNorm skipped).
        capture_layer: ?usize = null,
        /// When given, receives chunk 0's post-final-LN rows — the only thing `pooled`
        /// may read, and only from the first chunk.
        final_chunk0: ?[]f32 = null,
    };

    /// Run this tower over every chunk of a tokenized prompt and apply the attention
    /// weights, filling `out` (`[p.seq()][hidden]`).
    ///
    /// The chunks are **independent forwards concatenated along the sequence axis** —
    /// each carries its own BOS/EOS and its own 77 positional embeddings, so a 154-row
    /// conditioning is two separate encodings of a *causal* tower rather than one long
    /// one. That is why a chunk boundary is a real semantic seam, and why
    /// `clip_tokenizer.encodeWeighted` works to keep short segments from straddling it.
    pub fn encodePrompt(
        self: *const TextEncoder,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        p: *const tp_core.clip_tokenizer.Prompt,
        r: PromptRun,
    ) !void {
        const clen = tp_core.clip_tokenizer.context_length;
        const h = self.cfg.hidden;
        std.debug.assert(out.len == p.seq() * h);

        // Only needed when the conditioning is a *captured* layer, in which case the
        // final output still has to go somewhere.
        const scratch = try gpa.alloc(f32, if (r.capture_layer != null) clen * h else 0);
        defer gpa.free(scratch);

        var ids: [tp_core.clip_tokenizer.context_length]u32 = undefined;
        for (0..p.chunks) |c| {
            p.idsInto(&ids, c);
            const rows = out[c * clen * h ..][0 .. clen * h];
            const final = if (r.capture_layer) |layer| blk: {
                try self.forward(io, gpa, scratch, &ids, .{ .layer = layer, .dst = rows });
                break :blk scratch;
            } else blk: {
                try self.forward(io, gpa, rows, &ids, null);
                break :blk rows;
            };
            if (c == 0) if (r.final_chunk0) |dst| @memcpy(dst, final);
        }

        if (!p.hasWeights()) return;
        const empty = if (r.mode.needsEmpty()) try self.emptyRef(io, gpa, r) else &[_]f32{};
        for (0..p.chunks) |c| {
            self.applyMode(out[c * clen * h ..][0 .. clen * h], empty, p.chunk(c), r.mode);
        }
    }

    /// Apply one chunk's weights in whichever dialect's form. `empty` is read only by
    /// `.comfy` and may be an empty slice otherwise.
    pub fn applyMode(
        self: *const TextEncoder,
        rows: []f32,
        empty: []const f32,
        w: []const tp_core.clip_tokenizer.Weighted,
        mode: WeightMode,
    ) void {
        switch (mode) {
            .comfy => self.applyWeights(rows, empty, w),
            .a1111_original => self.applyWeightsA1111(rows, w, true),
            .a1111_no_norm => self.applyWeightsA1111(rows, w, false),
        }
    }

    /// This tower's `z_empty`, computed once and cached in `r.empty_cache`.
    fn emptyRef(self: *const TextEncoder, io: std.Io, gpa: std.mem.Allocator, r: PromptRun) ![]const f32 {
        if (r.empty_cache.*) |e| return e;
        const clen = tp_core.clip_tokenizer.context_length;
        const h = self.cfg.hidden;

        var ids: [tp_core.clip_tokenizer.context_length]u32 = undefined;
        tp_core.clip_tokenizer.emptyIds(&ids, r.pad_id);

        const e = try gpa.alloc(f32, clen * h);
        errdefer gpa.free(e);
        if (r.capture_layer) |layer| {
            const scratch = try gpa.alloc(f32, clen * h);
            defer gpa.free(scratch);
            try self.forward(io, gpa, scratch, &ids, .{ .layer = layer, .dst = e });
        } else {
            try self.forward(io, gpa, e, &ids, null);
        }
        r.empty_cache.* = e;
        return e;
    }

    /// Apply the prompt's per-token attention weights to one chunk's hidden rows,
    /// in place.
    ///
    /// ⚠️ **This is an interpolation away from the EMPTY prompt's hidden state, not a
    /// multiply**, and not the mean-renormalized form A1111 uses either:
    ///
    /// ```
    /// z[j] = (z[j] - z_empty[j]) * w + z_empty[j]
    /// ```
    ///
    /// `z_empty` is this same tower run on `[BOS] [EOS] pad…` (`clip_tokenizer.emptyIds`)
    /// at the same capture layer, indexed by the **same position** j. So a weight of 0
    /// means "whatever this slot would say with no prompt at all" rather than a zero
    /// vector, and a weight of 2 extrapolates along that direction. Multiplying the row
    /// instead is a plausible-looking image with the wrong emphasis everywhere, since a
    /// CLIP hidden row is nowhere near zero-centred.
    ///
    /// Rows whose weight is exactly 1.0 are skipped rather than computed, so the
    /// arithmetic is bit-identical to not having weights at all — which is what lets the
    /// unweighted path stay untouched. `empty` may therefore be `undefined` when the
    /// caller knows `Prompt.hasWeights()` is false.
    pub fn applyWeights(
        self: *const TextEncoder,
        rows: []f32,
        empty: []const f32,
        w: []const tp_core.clip_tokenizer.Weighted,
    ) void {
        const h = self.cfg.hidden;
        std.debug.assert(rows.len == w.len * h);
        for (w, 0..) |t, p| {
            if (t.weight == 1.0) continue;
            const z = rows[p * h ..][0..h];
            const ze = empty[p * h ..][0..h];
            for (z, ze) |*zi, e| zi.* = (zi.* - e) * t.weight + e;
        }
    }

    /// A1111's emphasis application — the alternative to `applyWeights` above, and a
    /// genuinely different picture from the same numbers.
    ///
    /// ```
    /// z[j] *= w[j]                        (per position)
    /// z    *= original_mean / new_mean    (ONE scalar over the whole chunk)
    /// ```
    ///
    /// ⚠️ **The rescale is global, so emphasising one tag rescales every other token in
    /// that chunk** — including the ones at weight 1.0. That is the substantive
    /// difference from ComfyUI's form, which only moves the weighted positions. The mean
    /// is `z.mean()` over all positions × all channels of this chunk, matching
    /// `EmphasisOriginal`.
    ///
    /// `renormalize = false` is upstream's `EmphasisOriginalNoNorm`, a bare multiply,
    /// which upstream itself documents as working better for SDXL.
    ///
    /// The mean is accumulated in f64 where torch uses f32. That is deliberately *more*
    /// accurate than the reference rather than differently accurate: the quantity is a
    /// sum over ~100k values whose result is then divided out again, so an f32
    /// accumulation would add error the reference's own bound cannot describe.
    pub fn applyWeightsA1111(
        self: *const TextEncoder,
        rows: []f32,
        w: []const tp_core.clip_tokenizer.Weighted,
        renormalize: bool,
    ) void {
        const h = self.cfg.hidden;
        std.debug.assert(rows.len == w.len * h);

        var sum: f64 = 0;
        for (rows) |v| sum += v;
        const original_mean = sum / @as(f64, @floatFromInt(rows.len));

        for (w, 0..) |t, p| {
            if (t.weight == 1.0) continue; // exact, so an unweighted row is untouched
            for (rows[p * h ..][0..h]) |*z| z.* *= t.weight;
        }
        if (!renormalize) return;

        var sum2: f64 = 0;
        for (rows) |v| sum2 += v;
        const new_mean = sum2 / @as(f64, @floatFromInt(rows.len));
        // A chunk whose weighted mean lands exactly on zero has no recoverable scale;
        // leaving it alone beats multiplying by an infinity.
        if (new_mean == 0) return;
        const k: f32 = @floatCast(original_mean / new_mean);
        for (rows) |*z| z.* *= k;
    }

    /// Which intermediate activation `forward` should copy out, and where to.
    pub const Capture = struct {
        /// 0 = embedding output, n = after encoder layer n−1.
        layer: usize,
        /// `[seq][hidden]`, caller-owned.
        dst: []f32,
    };

    /// The pooled vector SDXL conditions on: the hidden row at the **first**
    /// `eos_id`. HF selects it with `argmax(input_ids)`, which is equivalent only
    /// because `<|endoftext|>` is the highest id in SD's vocabularies and every
    /// padding slot holds it — so "first eos" is the well-defined form of the same
    /// choice. Returns null when the sequence has no eos (an unterminated prompt).
    pub fn pooled(self: *const TextEncoder, hidden: []const f32, ids: []const u32) ?[]const f32 {
        for (ids, 0..) |id, p| {
            if (id == self.cfg.eos_id) return hidden[p * self.cfg.hidden ..][0..self.cfg.hidden];
        }
        return null;
    }

    /// `dst = row @ text_projection` — the vector SDXL's `y` carries, from CLIP-G's
    /// pooled row.
    ///
    /// ⚠️ **The stored matrix is `[hidden][hidden]` in `x @ T` orientation**, which is
    /// the *transpose* of an `nn.Linear` weight; the reference implementations transpose
    /// it on load. Since the transpose is the whole content of this function it is done
    /// here explicitly rather than by materializing a flipped copy — `dst[j]` sums over
    /// the **column** j. Getting it backwards is a 1280-vector of the right magnitude
    /// pointing somewhere else entirely.
    pub fn projectPooled(self: *const TextEncoder, dst: []f32, row: []const f32) void {
        const h = self.cfg.hidden;
        const t = self.text_projection.?;
        std.debug.assert(dst.len == h and row.len == h);
        @memset(dst, 0);
        for (row, 0..) |x, i| {
            const tr = t[i * h ..][0..h];
            for (dst, tr) |*d, w| d.* += x * w;
        }
    }
};

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    /// `transformers` naming: three separate projections per attention.
    fn layerHf(l: Loader, i: usize) !Layer {
        const h = l.cfg.hidden;
        return .{
            .ln1_w = try l.vec("encoder.layers.{d}.layer_norm1.weight", .{i}, h),
            .ln1_b = try l.vec("encoder.layers.{d}.layer_norm1.bias", .{i}, h),
            .q = try l.mat("encoder.layers.{d}.self_attn.q_proj.weight", .{i}, h, h),
            .q_b = try l.vec("encoder.layers.{d}.self_attn.q_proj.bias", .{i}, h),
            .k = try l.mat("encoder.layers.{d}.self_attn.k_proj.weight", .{i}, h, h),
            .k_b = try l.vec("encoder.layers.{d}.self_attn.k_proj.bias", .{i}, h),
            .v = try l.mat("encoder.layers.{d}.self_attn.v_proj.weight", .{i}, h, h),
            .v_b = try l.vec("encoder.layers.{d}.self_attn.v_proj.bias", .{i}, h),
            .o = try l.mat("encoder.layers.{d}.self_attn.out_proj.weight", .{i}, h, h),
            .o_b = try l.vec("encoder.layers.{d}.self_attn.out_proj.bias", .{i}, h),
            .ln2_w = try l.vec("encoder.layers.{d}.layer_norm2.weight", .{i}, h),
            .ln2_b = try l.vec("encoder.layers.{d}.layer_norm2.bias", .{i}, h),
            .fc1 = try l.mat("encoder.layers.{d}.mlp.fc1.weight", .{i}, l.cfg.intermediate, h),
            .fc1_b = try l.vec("encoder.layers.{d}.mlp.fc1.bias", .{i}, l.cfg.intermediate),
            .fc2 = try l.mat("encoder.layers.{d}.mlp.fc2.weight", .{i}, h, l.cfg.intermediate),
            .fc2_b = try l.vec("encoder.layers.{d}.mlp.fc2.bias", .{i}, h),
        };
    }

    /// OpenCLIP naming: `resblocks`, `ln_1`/`ln_2`, `c_fc`/`c_proj`, and **one fused
    /// `attn.in_proj_weight`** whose three row blocks are q, k, v in that order.
    fn layerOpenClip(l: Loader, i: usize) !Layer {
        const h = l.cfg.hidden;
        const qkv = try l.fusedQkv("transformer.resblocks.{d}.attn.in_proj_weight", .{i});
        const qkv_b = try l.vec("transformer.resblocks.{d}.attn.in_proj_bias", .{i}, 3 * h);
        return .{
            .ln1_w = try l.vec("transformer.resblocks.{d}.ln_1.weight", .{i}, h),
            .ln1_b = try l.vec("transformer.resblocks.{d}.ln_1.bias", .{i}, h),
            .q = qkv[0],
            .q_b = qkv_b[0..h],
            .k = qkv[1],
            .k_b = qkv_b[h..][0..h],
            .v = qkv[2],
            .v_b = qkv_b[2 * h ..][0..h],
            .o = try l.mat("transformer.resblocks.{d}.attn.out_proj.weight", .{i}, h, h),
            .o_b = try l.vec("transformer.resblocks.{d}.attn.out_proj.bias", .{i}, h),
            .ln2_w = try l.vec("transformer.resblocks.{d}.ln_2.weight", .{i}, h),
            .ln2_b = try l.vec("transformer.resblocks.{d}.ln_2.bias", .{i}, h),
            .fc1 = try l.mat("transformer.resblocks.{d}.mlp.c_fc.weight", .{i}, l.cfg.intermediate, h),
            .fc1_b = try l.vec("transformer.resblocks.{d}.mlp.c_fc.bias", .{i}, l.cfg.intermediate),
            .fc2 = try l.mat("transformer.resblocks.{d}.mlp.c_proj.weight", .{i}, h, l.cfg.intermediate),
            .fc2_b = try l.vec("transformer.resblocks.{d}.mlp.c_proj.bias", .{i}, h),
        };
    }

    /// Split a `[3·hidden, hidden]` fused projection into q, k, v.
    ///
    /// The three blocks are contiguous rows of one row-major tensor, so this is a view
    /// per block and copies nothing — the same zero-copy path an unfused checkpoint
    /// takes. (A quantized store would hand us a dtype the GEMM reads directly too; only
    /// the dtypes it cannot read are materialized, and then the split is over f32.)
    ///
    /// All three carry the **fused tensor's** name as their probe tag, deliberately: q, k
    /// and v are three GEMMs over the *same* input, so per-input-column activation
    /// statistics are identical for all three, and the fused tensor is also the unit
    /// ggufy quantizes. Three contributions instead of one scales every column of that
    /// row's importance equally, which a per-row-relative imatrix is invariant to.
    fn fusedQkv(l: Loader, comptime fmt: []const u8, args: anytype) ![3]Weight {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("clip_text: missing {s}", .{nm});
            return error.MissingTensor;
        };
        const h = l.cfg.hidden;
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != 3 * h or shape[1] != h) {
            std.log.err("clip_text: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, view.info.dtype, 3 * h, h });
            return error.ShapeMismatch;
        }
        const tag = try l.alloc.dupe(u8, nm);
        var out: [3]Weight = undefined;
        // `!flat_blocks`: a shape-fixed block-quantized tensor's blocks tile its flat
        // element sequence, so neither the byte split below nor `Weight.init`'s
        // row-aligned blocking is valid for it — materialize instead.
        if (ops.matmul.supportsDType(view.info.dtype) and !view.info.flat_blocks) {
            // A block-quantized tower would put `h*h` elements per third, and the split
            // is only byte-addressable if that is a whole number of blocks. It is for
            // every real CLIP (h is a multiple of 64, so h² is a multiple of 256), but
            // `storageBytes` only *asserts* it — and asserts are gone in ReleaseFast,
            // where the result would be a silently wrong byte count.
            if (view.info.dtype.info().block_elems > 1 and (h * h) % view.info.dtype.info().block_elems != 0) {
                std.log.err("clip_text: {s} is {t} and hidden² ({d}) is not a whole number of blocks", .{ nm, view.info.dtype, h * h });
                return error.UnsupportedFusedQkvLayout;
            }
            const block = view.info.dtype.storageBytes(h * h);
            for (&out, 0..) |*w, b| {
                w.* = Weight.init(view.bytes[b * block ..][0..block], view.info.dtype, h, h);
                w.tag = tag;
            }
        } else {
            const f32s = try view.toF32Alloc(l.alloc);
            for (&out, 0..) |*w, b| {
                w.* = Weight.fromF32(f32s[b * h * h ..][0 .. h * h], h, h);
                w.tag = tag;
            }
        }
        return out;
    }

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        return fbs.buffered();
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("clip_text: missing {s}", .{nm});
            return error.MissingTensor;
        };
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != rows or shape[1] != cols) {
            std.log.err("clip_text: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, view.info.dtype, rows, cols });
            return error.ShapeMismatch;
        }
        // Most checkpoints hand us a dtype the GEMM reads directly (f32/f16/bf16/fp8,
        // or a quantized text tower). The rest are materialized to f32 once, here:
        // this SD1.5 merge stores its CLIP linears as **f64**, which ComfyUI also
        // casts on load, and discovering that from inside the first forward would
        // report the problem as far from its cause as possible.
        var w = if (ops.matmul.supportsDType(view.info.dtype) and !view.info.flat_blocks) blk: {
            break :blk Weight.init(view.bytes, view.info.dtype, rows, cols);
        } else blk: {
            std.log.debug("clip_text: materializing {s} ({t}) to f32", .{ nm, view.info.dtype });
            const f32s = try view.toF32Alloc(l.alloc);
            break :blk Weight.fromF32(f32s, rows, cols);
        };
        // The checkpoint name travels with the weight so `ops.matmul.probe` can
        // attribute a GEMM to a layer — which is what makes the text tower
        // measurable at all.
        w.tag = try l.alloc.dupe(u8, nm);
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("clip_text: missing {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("clip_text: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");

/// The reference fixture (`tools/gen_sd15_fixtures.py`): a real SD1.5 checkpoint
/// driven through `transformers.CLIPTextModel`. Gated, because it needs the 5 GB
/// checkpoint the fixture was generated against.
const ref_path = "src/models/assets/sd15_ref.safetensors";
const sd15_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";

test "CLIP-L matches transformers.CLIPTextModel on a real SD1.5 checkpoint" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_l, "cond_stage_model.transformer.text_model.");
    defer enc.deinit();

    // Token ids for two prompts (one of them empty — the CFG branch), [2][77] i32.
    // Read as raw i32: `convertToF32` deliberately refuses integer dtypes, since
    // silently turning token ids into floats is never what a caller wants.
    const tokens_view = ref.get("clip.tokens").?;
    const tokens = std.mem.bytesAsSlice(i32, tokens_view.bytes);
    const expected = try ref.get("clip.hidden").?.toF32Alloc(gpa);
    defer gpa.free(expected);
    const expected_embed = try ref.get("clip.embed_out").?.toF32Alloc(gpa);
    defer gpa.free(expected_embed);
    const expected_l0 = try ref.get("clip.layer0").?.toF32Alloc(gpa);
    defer gpa.free(expected_l0);
    const expected_pooled = try ref.get("clip.pooled").?.toF32Alloc(gpa);
    defer gpa.free(expected_pooled);

    const seq = clip_l.max_positions;
    const hid = clip_l.hidden;
    const n_prompts = tokens.len / seq;

    const ids = try gpa.alloc(u32, seq);
    defer gpa.free(ids);
    const out = try gpa.alloc(f32, seq * hid);
    defer gpa.free(out);
    const cap = try gpa.alloc(f32, seq * hid);
    defer gpa.free(cap);

    for (0..n_prompts) |p| {
        for (ids, 0..) |*id, i| id.* = @intCast(tokens[p * seq + i]);

        // The embedding output first: a wrong positional table or a wrong lookup
        // shows up here, before any attention can smear it.
        try enc.forward(io, gpa, out, ids, .{ .layer = 0, .dst = cap });
        for (expected_embed[p * seq * hid ..][0 .. seq * hid], cap, 0..) |e, a, i| {
            errdefer std.debug.print("clip embed_out prompt {d} idx {d}: {d:.6} vs {d:.6}\n", .{ p, i, e, a });
            try testing.expectApproxEqAbs(e, a, 2e-4);
        }

        // Then layer 0, which is where a non-causal mask or the wrong activation
        // first becomes visible.
        try enc.forward(io, gpa, out, ids, .{ .layer = 1, .dst = cap });
        for (expected_l0[p * seq * hid ..][0 .. seq * hid], cap, 0..) |e, a, i| {
            errdefer std.debug.print("clip layer0 prompt {d} idx {d}: {d:.6} vs {d:.6}\n", .{ p, i, e, a });
            try testing.expectApproxEqAbs(e, a, 2e-3);
        }

        // And the whole tower. Tolerance is loose in absolute terms because CLIP's
        // final hidden states run to ~30 in magnitude; what matters is that it is
        // ~1e-4 relative across 59k values, not that it is 1e-6.
        try enc.forward(io, gpa, out, ids, null);
        var max_abs: f32 = 0;
        var max_rel: f32 = 0;
        for (expected[p * seq * hid ..][0 .. seq * hid], out) |e, a| {
            max_abs = @max(max_abs, @abs(e - a));
            const scale = @max(@abs(e), 1.0);
            max_rel = @max(max_rel, @abs(e - a) / scale);
        }
        errdefer std.debug.print("clip hidden prompt {d}: max_abs {d:.6} max_rel {d:.6}\n", .{ p, max_abs, max_rel });
        try testing.expect(max_rel < 5e-3);

        // The pooled row SDXL will need, selected by first-eos rather than argmax.
        const pooled = enc.pooled(out, ids).?;
        for (expected_pooled[p * hid ..][0..hid], pooled) |e, a| {
            try testing.expectApproxEqAbs(e, a, 5e-2);
        }
    }
}

const sdxl_ref_path = "src/models/assets/sdxl_ref.safetensors";
const sdxl_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sdxl/blackMAGICXL_v145.safetensors";

test "SDXL's two towers match ComfyUI on a real SDXL checkpoint" {
    // The reference is ComfyUI (`tools/gen_sdxl_fixtures.py`), which is both the
    // compatibility target and the thing that settles SDXL's three conventions: the
    // penultimate hidden state with **no** final LayerNorm, CLIP-G padded with 0, and
    // pooled taken from CLIP-G's projected final state.
    //
    // The two towers are checked through the concatenated context ComfyUI hands the UNet,
    // sliced back into halves — `[0..768)` is CLIP-L's, `[768..2048)` is CLIP-G's — so
    // this pins the concatenation order too, which is otherwise a coin flip that
    // produces a plausible image either way.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sdxl_ckpt);
    try test_gate.requireModelFile(io, sdxl_ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, sdxl_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sdxl_ckpt);
    defer ck.deinit();

    var l_enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_l, "conditioner.embedders.0.transformer.text_model.");
    defer l_enc.deinit();
    var g_enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_g, "conditioner.embedders.1.model.");
    defer g_enc.deinit();

    const seq = 77;
    const hl = clip_l.hidden;
    const hg = clip_g.hidden;
    const wide = hl + hg;

    const tok_l = ref.get("clip.tokens_l").?;
    const tok_g = ref.get("clip.tokens_g").?;
    const ids_l_i32 = std.mem.bytesAsSlice(i32, tok_l.bytes);
    const ids_g_i32 = std.mem.bytesAsSlice(i32, tok_g.bytes);
    const context = try ref.get("clip.context").?.toF32Alloc(gpa);
    defer gpa.free(context);
    const want_pooled = try ref.get("clip.pooled").?.toF32Alloc(gpa);
    defer gpa.free(want_pooled);
    const want_embed = try ref.get("clip_g.embed_out").?.toF32Alloc(gpa);
    defer gpa.free(want_embed);
    const want_l0 = try ref.get("clip_g.layer0").?.toF32Alloc(gpa);
    defer gpa.free(want_l0);
    const want_final = try ref.get("clip_g.final_ln").?.toF32Alloc(gpa);
    defer gpa.free(want_final);

    const n_prompts = ids_l_i32.len / seq;
    const ids_l = try gpa.alloc(u32, seq);
    defer gpa.free(ids_l);
    const ids_g = try gpa.alloc(u32, seq);
    defer gpa.free(ids_g);
    const out_l = try gpa.alloc(f32, seq * hl);
    defer gpa.free(out_l);
    const cap_l = try gpa.alloc(f32, seq * hl);
    defer gpa.free(cap_l);
    const out_g = try gpa.alloc(f32, seq * hg);
    defer gpa.free(out_g);
    const cap_g = try gpa.alloc(f32, seq * hg);
    defer gpa.free(cap_g);
    const pooled = try gpa.alloc(f32, hg);
    defer gpa.free(pooled);

    for (0..n_prompts) |p| {
        for (ids_l, 0..) |*id, i| id.* = @intCast(ids_l_i32[p * seq + i]);
        for (ids_g, 0..) |*id, i| id.* = @intCast(ids_g_i32[p * seq + i]);

        // CLIP-G first, and its localizers before its output: this tower is the new
        // code path (OpenCLIP names, fused q/k/v split into three views), and a
        // mis-split shows up at the embedding output or after layer 0 rather than as a
        // vague disagreement 32 layers later.
        try g_enc.forward(io, gpa, out_g, ids_g, .{ .layer = 0, .dst = cap_g });
        try expectRel("clip_g.embed_out", want_embed[p * seq * hg ..][0 .. seq * hg], cap_g, 3e-3);
        try g_enc.forward(io, gpa, out_g, ids_g, .{ .layer = 1, .dst = cap_g });
        try expectRel("clip_g.layer0", want_l0[p * seq * hg ..][0 .. seq * hg], cap_g, 3e-3);

        // `layers - 1` = "after encoder layer layers-2" = the reference's
        // `hidden_states[-2]`, the penultimate state SDXL conditions on.
        try g_enc.forward(io, gpa, out_g, ids_g, .{ .layer = clip_g.layers - 1, .dst = cap_g });
        try expectRel("clip_g.final_ln", want_final[p * seq * hg ..][0 .. seq * hg], out_g, 3e-3);
        try l_enc.forward(io, gpa, out_l, ids_l, .{ .layer = clip_l.layers - 1, .dst = cap_l });

        // The concatenated context, one row at a time so a failure names the position.
        for (0..seq) |r| {
            const want_row = context[(p * seq + r) * wide ..][0..wide];
            try expectRel("context[clip_l half]", want_row[0..hl], cap_l[r * hl ..][0..hl], 3e-3);
            try expectRel("context[clip_g half]", want_row[hl..], cap_g[r * hg ..][0..hg], 3e-3);
        }

        // And the pooled vector, from the FINAL (LayerNormed) state through
        // `text_projection` — a different quantity than either context half.
        const row = g_enc.pooled(out_g, ids_g).?;
        g_enc.projectPooled(pooled, row);
        try expectRel("clip.pooled", want_pooled[p * hg ..][0..hg], pooled, 3e-3);
    }
}

/// Relative-L2 comparison with a named diagnostic. CLIP-G's activations run to ~600 in
/// magnitude (its well-known outlier channels), so an absolute tolerance would be either
/// meaningless there or impossibly tight on the small values in the same tensor.
fn expectRel(what: []const u8, want: []const f32, got: []const f32, tol: f64) !void {
    try testing.expectEqual(want.len, got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f64 = 0;
    for (want, got) |e, a| {
        max_abs = @max(max_abs, @abs(e - a));
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    const rel = if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
    errdefer std.debug.print("{s}: rel L2 {e:.4} (tol {e:.4}) max_abs {d:.5}\n", .{ what, rel, tol, max_abs });
    try testing.expect(rel < tol);
}

test "the causal mask is what makes this an encoder and not a bidirectional one" {
    // A property test that needs no checkpoint: with causal attention, appending a
    // token cannot change the rows before it. If the mask were dropped, every row
    // would move — and the model would still produce plausible conditioning, which is
    // why this is worth pinning separately from the fixture.
    const gpa = testing.allocator;
    const io = testing.io;

    const cfg: Config = .{
        .hidden = 8, .layers = 2, .heads = 2, .intermediate = 16,
        .max_positions = 6, .eps = 1e-5, .act = .quick_gelu, .eos_id = 3,
    };
    var ckpt = try TinyCheckpoint.init(gpa, cfg, 16);
    defer ckpt.deinit(gpa);
    var enc = try TextEncoder.load(gpa, ckpt.store(), cfg, "");
    defer enc.deinit();

    const ids4 = [_]u32{ 1, 5, 9, 2 };
    const ids5 = [_]u32{ 1, 5, 9, 2, 7 };
    const out4 = try gpa.alloc(f32, ids4.len * cfg.hidden);
    defer gpa.free(out4);
    const out5 = try gpa.alloc(f32, ids5.len * cfg.hidden);
    defer gpa.free(out5);
    try enc.forward(io, gpa, out4, &ids4, null);
    try enc.forward(io, gpa, out5, &ids5, null);

    // final_layer_norm is per row, so the shared prefix must match exactly.
    for (out4, out5[0..out4.len], 0..) |a, b, i| {
        errdefer std.debug.print("causal prefix diverged at {d}: {d:.6} vs {d:.6}\n", .{ i, a, b });
        try testing.expectApproxEqAbs(a, b, 1e-5);
    }
}

/// A deterministic in-memory **safetensors** checkpoint for the small property
/// tests: every weight is a fixed pseudo-random f32, serialized into a caller-owned
/// buffer and opened through the real parser. Built as a real container rather than
/// a stub store because `WeightStore` is a closed union over the two formats we
/// actually read — and because a test that bypasses the parser would not catch a
/// loader that mis-reads shapes.
///
/// `pub` so the device-parity tests in `clip_text_gpu` can build the same tower the CPU
/// forward is checked against; `.hf` naming only, which is all those tests need.
pub const TinyCheckpoint = struct {
    buf: []u8,
    st: safetensors.SafeTensors,

    pub fn init(gpa: std.mem.Allocator, cfg: Config, vocab: usize) !TinyCheckpoint {
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var shapes: std.ArrayList([2]usize) = .empty; // [rows, cols]; cols 0 = rank-1
        defer shapes.deinit(gpa);

        const add = struct {
            fn f(g: std.mem.Allocator, ns: *std.ArrayList([]const u8), ss: *std.ArrayList([2]usize), comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !void {
                try ns.append(g, try std.fmt.allocPrint(g, fmt, args));
                try ss.append(g, .{ rows, cols });
            }
        }.f;

        try add(gpa, &names, &shapes, "embeddings.token_embedding.weight", .{}, vocab, cfg.hidden);
        try add(gpa, &names, &shapes, "embeddings.position_embedding.weight", .{}, cfg.max_positions, cfg.hidden);
        for (0..cfg.layers) |i| {
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm1.weight", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm1.bias", .{i}, cfg.hidden, 0);
            inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |proj| {
                try add(gpa, &names, &shapes, "encoder.layers.{d}.self_attn." ++ proj ++ ".weight", .{i}, cfg.hidden, cfg.hidden);
                try add(gpa, &names, &shapes, "encoder.layers.{d}.self_attn." ++ proj ++ ".bias", .{i}, cfg.hidden, 0);
            }
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm2.weight", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm2.bias", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc1.weight", .{i}, cfg.intermediate, cfg.hidden);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc1.bias", .{i}, cfg.intermediate, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc2.weight", .{i}, cfg.hidden, cfg.intermediate);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc2.bias", .{i}, cfg.hidden, 0);
        }
        try add(gpa, &names, &shapes, "final_layer_norm.weight", .{}, cfg.hidden, 0);
        try add(gpa, &names, &shapes, "final_layer_norm.bias", .{}, cfg.hidden, 0);

        // Header first, so the data offsets are known before anything is written.
        var hdr: std.Io.Writer.Allocating = .init(gpa);
        defer hdr.deinit();
        try hdr.writer.writeByte('{');
        var offset: usize = 0;
        for (names.items, shapes.items, 0..) |nm, sh, idx| {
            const n = sh[0] * (if (sh[1] == 0) 1 else sh[1]);
            const bytes = n * @sizeOf(f32);
            if (idx > 0) try hdr.writer.writeByte(',');
            if (sh[1] == 0) {
                try hdr.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ nm, sh[0], offset, offset + bytes });
            } else {
                try hdr.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ nm, sh[0], sh[1], offset, offset + bytes });
            }
            offset += bytes;
        }
        try hdr.writer.writeByte('}');
        const header = hdr.written();

        const buf = try gpa.alloc(u8, 8 + header.len + offset);
        errdefer gpa.free(buf);
        std.mem.writeInt(u64, buf[0..8], header.len, .little);
        @memcpy(buf[8..][0..header.len], header);

        // Deterministic values: an LCG, so the fixture is the same on every machine.
        var seed: u64 = 0x5EED;
        const data = std.mem.bytesAsSlice(f32, buf[8 + header.len ..]);
        for (data) |*x| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            x.* = (@as(f32, @floatFromInt(@as(u32, @truncate(seed >> 33)))) / 2147483648.0 - 1.0) * 0.2;
        }

        return .{ .buf = buf, .st = try safetensors.SafeTensors.initFromSlice(gpa, buf) };
    }

    pub fn deinit(self: *TinyCheckpoint, gpa: std.mem.Allocator) void {
        self.st.deinit();
        gpa.free(self.buf);
        self.* = undefined;
    }

    pub fn store(self: *const TinyCheckpoint) WeightStore {
        return .{ .safetensors = &self.st };
    }
};

// --- weighted, multi-chunk prompts -----------------------------------------

/// `tools/gen_clip_cond_ref.py`: ComfyUI's `CLIPTextEncode` output for seven prompts on
/// two real checkpoints. Pins the *encoder* half of long-prompt support — the chunk
/// forwards and `applyWeights` — where `clip_tokenizer`'s own fixture pins the tokenizer
/// half. Both are needed: the ids can be right while the weighting form is wrong, and
/// that combination is a plausible image with the wrong emphasis everywhere.
const cond_ref_path = "src/models/assets/clip_cond_ref.safetensors";
const cond_sd15_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";
const cond_sdxl_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sdxl/perfectdeliberate_v10.safetensors";

const cond_cases = [_][]const u8{ "real_pos", "real_neg", "plain", "empty", "zero_w", "big_w", "three" };

/// The prompt strings the fixture was generated from, in the same order. Kept here
/// rather than in the fixture because the *point* is that this engine tokenizes them
/// itself — a fixture supplying the ids would leave the tokenizer untested from here.
const cond_prompts = [_][]const u8{
    "1girl, original, general, blonde hair, long hair, ponytail, light blue eyes, " ++
        "(shiny skin:1.1), gentle smile, bare shoulders, collarbone, " ++
        "[white|light blue] off-shoulder shirt, sheer sleeves, denim short shorts, " ++
        "holding straw hat, looking back, side angle, upper body, beach, ocean, summer, " ++
        "sunny day, (lens flare:0.75), breeze, BREAK, masterpiece, best quality, " ++
        "very aesthetic, high contrast, vibrant, highres, year 2024, newest,",
    "(worst quality, low quality:1.2), (nsfw, sexually suggestive:1.2), lowres, " ++
        "(monochrome:1.1), wide shot, multiple views, watermark, signature, " ++
        "1980s \\(style\\), 4koma, serafuku, (text:1.1),",
    "a photograph of an astronaut riding a horse",
    "",
    "a (red:0) cat",
    "a (red:2.0) cat",
    "masterpiece, " ++ ("very detailed cat portrait, " ** 25) ++ "(sunset:1.4)",
};

/// Relative L2 of `got` against `want`, which is the figure these tolerances are in:
/// CLIP hidden states run to ~30 in magnitude and a weight of 2 extrapolates further,
/// so an absolute bound would be meaningless across the seven cases.
///
/// Measured across all 14 prompt x tower comparisons: CLIP-L 1.2e-6 to 2.3e-6, CLIP-G
/// 1.6e-5 to 2.7e-5 (32 layers of f32 reduction-order difference), pooled 4.4e-6 to
/// 2.1e-5. The 5e-5 bound below is ~2x the worst of those, not a round number chosen
/// for comfort — a weighting form that is merely *plausible* misses by 1e-1, so the
/// bound has room to be this tight.
fn relL2(want: []const f32, got: []const f32) f32 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        num += @as(f64, e - a) * @as(f64, e - a);
        den += @as(f64, e) * @as(f64, e);
    }
    if (den == 0) return @floatCast(@sqrt(num));
    return @floatCast(@sqrt(num / den));
}

test "SD1.5 weighted multi-chunk conditioning matches ComfyUI's CLIPTextEncode" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, cond_sd15_ckpt);
    try test_gate.requireModelFile(io, cond_ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, cond_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, cond_sd15_ckpt);
    defer ck.deinit();

    var enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_l, "cond_stage_model.transformer.text_model.");
    defer enc.deinit();

    var tok = try tp_core.clip_tokenizer.Tokenizer.init(gpa);
    defer tok.deinit();

    // One cache across all seven prompts, which is also what `SdModels` does — so this
    // incidentally covers the cached arm rather than only the first-fill one.
    var empty_cache: ?[]f32 = null;
    defer if (empty_cache) |e| gpa.free(e);

    for (cond_cases, cond_prompts) |name, text| {
        var buf: [64]u8 = undefined;
        const want = try ref.get(try std.fmt.bufPrint(&buf, "sd15.{s}.cond", .{name})).?.toF32Alloc(gpa);
        defer gpa.free(want);

        var p = try tok.encodeWeighted(gpa, text, tp_core.clip_tokenizer.eos_id);
        defer p.deinit(gpa);
        // The row count is itself a claim: 154 where ComfyUI chunked and 77 where it
        // did not. Getting this wrong used to be silent truncation.
        try testing.expectEqual(want.len, p.seq() * clip_l.hidden);

        const got = try gpa.alloc(f32, p.seq() * clip_l.hidden);
        defer gpa.free(got);
        try enc.encodePrompt(io, gpa, got, &p, .{
            .empty_cache = &empty_cache,
            .pad_id = tp_core.clip_tokenizer.eos_id,
        });

        const rel = relL2(want, got);
        errdefer std.debug.print("sd15 '{s}': {d} rows, rel L2 {d:.6}\n", .{ name, p.seq(), rel });
        try testing.expect(rel < 5e-5);
    }
}

test "SDXL weighted multi-chunk conditioning matches ComfyUI's CLIPTextEncode" {
    // Strictly more than the SD1.5 case: two towers, CLIP-G padded with 0, the
    // conditioning taken from the *penultimate* layer with the final LayerNorm skipped,
    // and a pooled vector that must come from chunk 0 alone.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, cond_sdxl_ckpt);
    try test_gate.requireModelFile(io, cond_ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, cond_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, cond_sdxl_ckpt);
    defer ck.deinit();

    var l_enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_l, "conditioner.embedders.0.transformer.text_model.");
    defer l_enc.deinit();
    var g_enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_g, "conditioner.embedders.1.model.");
    defer g_enc.deinit();

    var tok = try tp_core.clip_tokenizer.Tokenizer.init(gpa);
    defer tok.deinit();

    const clen = tp_core.clip_tokenizer.context_length;
    const hl = clip_l.hidden;
    const hg = clip_g.hidden;
    const wide = hl + hg;

    var cache_l: ?[]f32 = null;
    defer if (cache_l) |e| gpa.free(e);
    var cache_g: ?[]f32 = null;
    defer if (cache_g) |e| gpa.free(e);

    for (cond_cases, cond_prompts) |name, text| {
        var buf: [64]u8 = undefined;
        const want = try ref.get(try std.fmt.bufPrint(&buf, "sdxl.{s}.cond", .{name})).?.toF32Alloc(gpa);
        defer gpa.free(want);
        const want_pooled = try ref.get(try std.fmt.bufPrint(&buf, "sdxl.{s}.pooled", .{name})).?.toF32Alloc(gpa);
        defer gpa.free(want_pooled);

        var pl = try tok.encodeWeighted(gpa, text, tp_core.clip_tokenizer.eos_id);
        defer pl.deinit(gpa);
        var pg = try tok.encodeWeighted(gpa, text, 0);
        defer pg.deinit(gpa);
        // The two paddings must never change the chunk split — that is what makes the
        // halves of the concatenated context line up row for row.
        try testing.expectEqual(pl.chunks, pg.chunks);
        try testing.expectEqual(want.len, pl.seq() * wide);

        const cap_l = try gpa.alloc(f32, pl.seq() * hl);
        defer gpa.free(cap_l);
        try l_enc.encodePrompt(io, gpa, cap_l, &pl, .{
            .empty_cache = &cache_l,
            .pad_id = tp_core.clip_tokenizer.eos_id,
            .capture_layer = clip_l.layers - 1,
        });

        const cap_g = try gpa.alloc(f32, pg.seq() * hg);
        defer gpa.free(cap_g);
        const final_g = try gpa.alloc(f32, clen * hg);
        defer gpa.free(final_g);
        try g_enc.encodePrompt(io, gpa, cap_g, &pg, .{
            .empty_cache = &cache_g,
            .pad_id = 0,
            .capture_layer = clip_g.layers - 1,
            .final_chunk0 = final_g,
        });

        // Compared as the two halves of ComfyUI's concatenated context rather than
        // rebuilt into one — the concat order is pinned by the SDXL test above, and
        // slicing keeps a failure attributable to a specific tower.
        const l_rows = try gpa.alloc(f32, pl.seq() * hl);
        defer gpa.free(l_rows);
        const g_rows = try gpa.alloc(f32, pg.seq() * hg);
        defer gpa.free(g_rows);
        for (0..pl.seq()) |row| {
            @memcpy(l_rows[row * hl ..][0..hl], want[row * wide ..][0..hl]);
            @memcpy(g_rows[row * hg ..][0..hg], want[row * wide + hl ..][0..hg]);
        }

        const rel_l = relL2(l_rows, cap_l);
        const rel_g = relL2(g_rows, cap_g);
        errdefer std.debug.print(
            "sdxl '{s}': {d} rows, rel L2 clip_l {d:.6} clip_g {d:.6}\n",
            .{ name, pl.seq(), rel_l, rel_g },
        );
        try testing.expect(rel_l < 5e-5);
        try testing.expect(rel_g < 5e-5);

        // Pooled: chunk 0's final-LayerNormed row at the first EOS, projected — and
        // never weighted. Taking it from the last chunk would condition `y` on the
        // quality tags alone.
        var ids0: [tp_core.clip_tokenizer.context_length]u32 = undefined;
        pg.idsInto(&ids0, 0);
        const row = g_enc.pooled(final_g, &ids0).?;
        const got_pooled = try gpa.alloc(f32, hg);
        defer gpa.free(got_pooled);
        g_enc.projectPooled(got_pooled, row);
        const rel_p = relL2(want_pooled, got_pooled);
        errdefer std.debug.print("sdxl '{s}': pooled rel L2 {d:.6}\n", .{ name, rel_p });
        try testing.expect(rel_p < 5e-5);
    }
}

test "A1111 emphasis restores the chunk mean and so touches unweighted rows too" {
    // The property that distinguishes the two dialects' APPLICATION (as opposed to their
    // parsing): ComfyUI's form leaves a weight-1.0 row bit-identical, A1111's does not,
    // because its rescale is one scalar over the whole chunk.
    const gpa = testing.allocator;
    const cfg: Config = .{
        .hidden = 8, .layers = 1, .heads = 2, .intermediate = 16,
        .max_positions = tp_core.clip_tokenizer.context_length,
        .eps = 1e-5, .act = .quick_gelu, .eos_id = tp_core.clip_tokenizer.eos_id,
    };
    var ckpt = try TinyCheckpoint.init(gpa, cfg, 64);
    defer ckpt.deinit(gpa);
    var enc = try TextEncoder.load(gpa, ckpt.store(), cfg, "");
    defer enc.deinit();

    const n = 4;
    const w = [_]tp_core.clip_tokenizer.Weighted{
        .{ .id = 1, .weight = 1.0 },
        .{ .id = 2, .weight = 1.5 },
        .{ .id = 3, .weight = 1.0 },
        .{ .id = 4, .weight = 0.5 },
    };
    const base = try gpa.alloc(f32, n * cfg.hidden);
    defer gpa.free(base);
    var prng = std.Random.DefaultPrng.init(5);
    // Deliberately off-centre: a zero-mean chunk would make the rescale a no-op and the
    // test vacuous.
    for (base) |*v| v.* = 2.0 + prng.random().floatNorm(f32);

    const meanOf = struct {
        fn f(xs: []const f32) f64 {
            var s: f64 = 0;
            for (xs) |v| s += v;
            return s / @as(f64, @floatFromInt(xs.len));
        }
    }.f;
    const want_mean = meanOf(base);

    // `Original`: the chunk mean survives, and the unweighted rows have MOVED.
    const orig = try gpa.dupe(f32, base);
    defer gpa.free(orig);
    enc.applyWeightsA1111(orig, &w, true);
    try testing.expectApproxEqRel(want_mean, meanOf(orig), 1e-5);
    var unweighted_moved = false;
    for (0..cfg.hidden) |i| if (orig[i] != base[i]) {
        unweighted_moved = true;
    };
    try testing.expect(unweighted_moved);

    // `No norm`: a bare multiply, so row 0 is bit-identical and the mean is free to move.
    const nonorm = try gpa.dupe(f32, base);
    defer gpa.free(nonorm);
    enc.applyWeightsA1111(nonorm, &w, false);
    for (0..cfg.hidden) |i| try testing.expectEqual(base[i], nonorm[i]);
    try testing.expect(@abs(meanOf(nonorm) - want_mean) > 1e-3);

    // And ComfyUI's form for contrast: local, so row 0 is untouched there as well but the
    // weighted rows go somewhere else entirely (toward z_empty, not toward zero).
    const empty = try gpa.alloc(f32, n * cfg.hidden);
    defer gpa.free(empty);
    for (empty) |*v| v.* = 1.0;
    const comfy = try gpa.dupe(f32, base);
    defer gpa.free(comfy);
    enc.applyWeights(comfy, empty, &w);
    for (0..cfg.hidden) |i| try testing.expectEqual(base[i], comfy[i]);
    var differs = false;
    for (cfg.hidden..2 * cfg.hidden) |i| if (@abs(comfy[i] - nonorm[i]) > 1e-4) {
        differs = true;
    };
    try testing.expect(differs);
}
