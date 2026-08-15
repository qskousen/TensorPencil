# DIFFKEEP.md — Semantic-search encoders for DiffKeep

TensorPencil's first external consumer is **DiffKeep** (`../DiffKeep`), an image-library
app with semantic search. Today DiffKeep's entire search stack runs on **ONNX
Runtime**; this plan extends TensorPencil to cover those models **natively**
(safetensors + hand-implemented architectures, TP's existing pattern) so DiffKeep
can drop the `onnxruntime` dependency and inherit TP's CPU/CUDA/Vulkan backends
and quant for free.

**We are NOT building an ONNX graph runtime.** Native reimplementation only.

## Scope boundary — TP is the *encoder*, nothing else

DiffKeep's semantic search has three parts. Only the middle one is TP's job:

| Part | Owner | Notes |
|---|---|---|
| Vector storage + KNN + fusion | **DiffKeep (unchanged)** | sqlite-vec `vec0` tables, brute-force `MATCH`/`k`, Reciprocal-Rank-Fusion of the two spaces, debounced search flow |
| **`text → vec` / `image → vec` encoders** | **TensorPencil (this plan)** | replaces `src/onnx_embedder.zig` + `src/onnx_tokenizers.zig` |
| Image decode/resize | **DiffKeep (unchanged)** | libvips (`vips_helper.c`), 224² square. TP receives an already-decoded RGB buffer |

Everything below the encoder line (sqlite-vec, RRF, min-similarity thresholds,
FTS keyword mode) stays in DiffKeep and is out of scope.

## The models to cover

All produce **L2-normalized 768-d** vectors. There are **two independent
embedding spaces** (a prompt/query *text* space; SigLIP2 for image ↔ cross-modal
text) — DiffKeep keeps them in separate tables and fuses ranks, so TP just needs to
reproduce each encoder faithfully.

The **prompt-text space (role A)** has **two candidate models** — see the
"Text-encoder candidates" section below. DiffKeep will benchmark both on its
`search_quality_corpus` eval and pick one; TP implements whichever survives (and can
carry both, since they share the M2 encoder tier). The image space (B/C) is SigLIP2
only — no better drop-in exists.

| # | Model | Role | Arch | Tokenizer | Input | HF source (verify commit) |
|---|---|---|---|---|---|---|
| A1 | **Snowflake Arctic Embed M v2.0** | prompt-text docs + semantic text query | **GTE-multilingual-base**: bidirectional encoder, **RoPE**, 768-d, pooled | SentencePiece **Unigram** | text, ≤512 tok, query gets `"query: "` prefix | `Snowflake/snowflake-arctic-embed-m-v2.0` |
| A2 | **EmbeddingGemma 300M** | prompt-text docs + semantic text query | **Gemma 3** bidirectional encoder (T5Gemma-init), mean pool + 2-layer Dense projection, 768-d + **MRL** (512/256/128) | **Gemma tokenizer** (already in TP) | text, ≤2048 tok, task-prefixed (see below) | `google/embeddinggemma-300m` |
| B | **SigLIP2 ViT-B-16 (webli) — text tower** | cross-modal text→image query | bidirectional text transformer, `no_causal_mask`, pool + projection, 768-d | **Gemma-style BPE** (256k) | text, ≤64 tok | `timm/ViT-B-16-SigLIP2` (text tower) |
| C | **SigLIP2 ViT-B-16 (webli) — visual tower** | image index-time encoder | ViT patch-16, **attention-pool (MAP) head** + projection, 768-d | — | RGB 224² (see open Q), `/255`, mean/std 0.5 | `timm/ViT-B-16-SigLIP2` (visual tower) |

Asymmetry to preserve per model: **A1 (Snowflake)** prefixes queries `"query: "`,
documents raw. **A2 (EmbeddingGemma)** prefixes queries `task: search result | query: {q}`
and documents `title: none | text: {doc}`. SigLIP2 is symmetric per-tower.

### Text-encoder candidates (role A) — Snowflake vs EmbeddingGemma

Snowflake M v2.0 (Dec 2024) is the current DiffKeep model. EmbeddingGemma (Sept 2025)
is the newer SOTA-under-500M option and is added here as a candidate because it lands
on TP's most mature codepath at a fraction of the implementation cost:

| | Snowflake Arctic Embed M v2.0 (A1) | EmbeddingGemma 300M (A2) |
|---|---|---|
| Base arch | GTE-multilingual-base — **new to TP** | **Gemma 3** — TP has it on all 4 backends |
| Tokenizer | SentencePiece **Unigram/Viterbi** — **new algorithm** (see new-work #2) | **Gemma tokenizer** — already in TP (gemma3/gemma4) |
| Pooling/head | pooled (confirm CLS vs mean) | mean pool + 2-layer Dense projection |
| Output | 768-d | 768-d + **Matryoshka** (512/256/128) |
| Size | ~305M (113M non-embed) | 308M (~100M non-embed + 200M embed) |
| License | Apache 2.0 | Gemma license (TP already ships Gemma models) |
| Quality | baseline (DiffKeep today) | top open multilingual <500M on MTEB, beats ~2× models |

**Implementation leverage of A2:** it reuses TP's Gemma-3 forward + the existing Gemma
tokenizer, so it needs *no* new tokenizer and only the M2 bidirectional-encoder cap
(bidirectional attention already landed for gemma image tokens; add mean pool + Dense
head + task prefixes). Whether A2 also lets us skip the `tokenizer.json` loader / Unigram
work entirely depends on whether SigLIP2's text tokenizer (B) also maps onto TP's Gemma
tokenizer — verify in M0/M1.

**Cost of switching A1→A2 (DiffKeep-side, one-time):** different model = different
embedding space, so DiffKeep must **re-embed its text index once** on migration. This is
DiffKeep's call after its quality eval — TP just implements the winner.

## Target consumer API

The surface DiffKeep will call (exact names TBD; lives in the `TensorPencil` umbrella):

```zig
// One loaded encoder per model; caller owns the returned normalized vector.
const enc = try tp.embed.TextEncoder.open(gpa, io, .arctic_embed_m_v2, dir, backend);
const v: []f32 = try enc.embedText(gpa, "a query", .{ .role = .query }); // len 768, L2-normed

const senc = try tp.embed.TextEncoder.open(gpa, io, .siglip2_b16_text, dir, backend);
const ienc = try tp.embed.ImageEncoder.open(gpa, io, .siglip2_b16_visual, dir, backend);
const iv: []f32 = try ienc.embedImage(gpa, rgb_224x224); // len 768, L2-normed
```

Design notes:
- Output is **always L2-normalized** (DiffKeep relies on `sim = 1 - d²/2` under sqlite-vec L2).
- `role` handles Snowflake's query/document prefix asymmetry.
- Batch variants (`embedTextBatch`) matter for index-time throughput (DiffKeep batches 8).
- Backend selection reuses TP's existing `gpu.Context` / backend enum.

## What TP already has vs. what's new

**Reuse (big leverage):**
- Tensor infra, safetensors loader, `WeightStore`, dtype/quant — all there.
- GEMM / attention / norm / activation kernels on CPU + CUDA + Vulkan.
- **Bidirectional attention** — landed for gemma image tokens; the encoders need the
  same non-causal mask.
- **RoPE** — Snowflake/GTE uses it; TP has rope ops.
- **ViT towers** — gemma3 vision *is* SigLIP; the transformer body of tower C is close
  to existing code.
- **Gemma-style BPE algorithm** — TP's `gemma4` tokenizer kind is merge-by-rank BPE,
  matching SigLIP2's text tokenizer *algorithm*.

**Genuinely new work:**
1. **Bidirectional *encoder* tier** — every TP model today is a causal generative
   decoder ending in per-token logits. These are encoder-only, bidirectional, and end
   in **pooling + a projection head → one vector**. One reusable shape covers A and B
   (and A2/EmbeddingGemma: Gemma-3 body + bidirectional mask + mean pool + Dense head).
2. **Unigram/Viterbi tokenizer** — **only needed for A1 (Snowflake).** TP's `spm` is
   llama.cpp score-ranked *bigram-merge* SPM, a **different algorithm** from Snowflake's
   SentencePiece **Unigram lattice** (verified). New tokenizer required *if* we ship A1.
   **A2 (EmbeddingGemma) uses TP's existing Gemma tokenizer and needs none of this.**
3. **`tokenizer.json` loader** — TP loads vocab from GGUF / embedded assets only.
   DiffKeep's models ship HuggingFace `tokenizer.json`. Needed for the Unigram vocab+scores
   (A1) and SigLIP2 BPE vocab+merges (B). **May be avoidable if we ship A2 instead of A1
   *and* SigLIP2's tokenizer maps onto TP's Gemma tokenizer — verify in M0/M1.**
4. **Attention-pool (MAP) head + projection** — SigLIP's single-vector head; existing
   ViT towers emit per-patch features, not a pooled contrastive vector. New for tower C,
   and analogous pooling (CLS/mean) + projection for A/B.
5. **HF weight-name → `WeightStore` mapping** per model (GTE, SigLIP2 towers).

## Where it lives in the layer stack

- `tokenizer.json` loader + Unigram tokenizer → **`tp_core`** (pure-std, alongside the
  existing tokenizer).
- Encoder architectures + pooling/projection heads → **`tp_models`** (new `models/embed/`
  or similar).
- High-level `TextEncoder`/`ImageEncoder` façade → re-exported from the **umbrella**
  `root.zig` (e.g. `tp.embed`).
- Image preprocessing (resize/normalize) stays in DiffKeep (vips); TP takes a decoded
  RGB buffer, so no new image-decode dep in the library.

## Milestones

- [x] **M0 — Weights & config acquisition.** ✅ Done (see M0 findings appendix). Pull the candidate models' safetensors +
      configs from HF — **both A1 (Snowflake) and A2 (EmbeddingGemma)** for the text role,
      plus B/C (SigLIP2). Extract the exact hyperparameters (layers, width, heads, RoPE
      base, FFN type/gating, norm type/eps, pooling mode, projection shapes). For A2 also
      capture the Dense-head shapes + task prefixes and confirm its tokenizer == TP's Gemma
      tokenizer. Confirm the SigLIP2 resolution DiffKeep actually ships (open Q below).
      Decide compute dtype (start f32/f16; int8 later). Produce a one-page arch spec per model.
- [~] **M1 — Tokenizers.** **BPE side DONE (2026-07-22):** `tokenizer.json` BPE loader
      `Tokenizer.initGemma4FromTokenizerJson` in `src/core/tokenizer.zig` reuses the
      existing `encodeGemma4` merge path — **verified bit-identical to HF `tokenizers`**
      for both **A2/EmbeddingGemma (262144)** and **B/SigLIP2 text (256000)** across a
      12-case corpus (test `"gemma4 tokenizer.json matches HF tokenizers"`; golden in
      `testdata/embed_tokenizer_golden.json`, generator uses HF `tokenizers` 0.22.2).
      Post-processor frame (`<bos>…<eos>` vs `…<eos>`) is added by the embed façade, not
      the tokenizer. **A1 Unigram DONE (2026-07-22):** `Tokenizer.initUnigramFromTokenizerJson`
      + `.unigram` kind (whitespace-split → ▁-prefix → Viterbi over log-scores, whole-word
      `<unk>` fallback). **Bit-exact vs HF `tokenizers`** on the corpus (test `"unigram
      tokenizer.json matches HF tokenizers"`). The `Precompiled`/NFKC normalizer is
      intentionally NOT applied — matches DiffKeep's deployed `onnx_tokenizers.zig` (verified
      0/12 corpus divergence), so TP reproduces DiffKeep's existing index vectors.
- [~] **M2 — Bidirectional encoder tier.** **CPU DONE for A2 (2026-07-22):** reused
      `transformer.gemma3_spec` `layerForward(.fresh)` — extended `.fresh` to honor the
      `bidirectional` flag (`causal = !bidirectional`) so it drives a non-causal encoder;
      added mean-pool + L2-normalize + the no-bias Dense head in `src/models/embed_gemma.zig`.
      Remaining tier work for the other models: **CLS pooling** (A1), **MAP attention-pool
      head** (SigLIP visual C), and the SigLIP `text_projection`/`last`-token pool (B).
- [ ] **M3 — Text encoder (role A) end-to-end.** Implement the chosen candidate(s) on the
      M2 tier:
      - **A1 — Snowflake Arctic Embed:** ✅ **DONE + VALIDATED (2026-07-22)** in
        `src/models/embed_snowflake.zig` (`Model.open`/`embed`): GTE arch (post-LayerNorm,
        packed qkv, NEOX RoPE θ=160000, non-causal attention, GeGLU with exact-erf gelu,
        CLS pool → L2-norm). **Cosine vs Snowflake f32 ONNX = 0.9999998** across the 12-case
        corpus (test `"snowflake arctic embed matches ONNX reference"`; ref
        `testdata/snowflake_ref_vectors.json`). Added `ops.act.geluErf*` (exact-erf gelu).
        `"query: "` prefix applied by the caller/façade. CPU-only (M6 = GPU).
      - **A2 — EmbeddingGemma:** ✅ **DONE + VALIDATED (2026-07-22)** in
        `src/models/embed_gemma.zig` (`Model.open`/`embed`): Gemma-3 body reuse + bidirectional
        mask + mean-pool + 2 Dense + L2-norm. **Cosine vs onnx-community ONNX = 0.999985**
        across the 12-case corpus (test `"embeddinggemma matches ONNX reference"`; reference
        `testdata/embeddinggemma_ref_vectors.json`). Prompt prefixes (`task: search result |
        query:` / `title: none | text:`) are applied by the caller/façade, not the model.
        **Remaining:** CPU-only so far (M6 = CUDA/Vulkan); façade `tp.embed` + batch (M6);
        seq > 512 bidirectional sliding window (currently asserts seq ≤ 512).
      **Validation floor:** cosine vs ONNX ≥ ~0.999 (A2 far exceeds). Unlocks DiffKeep's
      `.semantic` mode. DiffKeep runs its quality eval to pick A1 vs A2.
- [x] **M4 — SigLIP2 text tower (model B).** ✅ **DONE + VALIDATED (2026-07-22)** in
      `src/models/embed_siglip.zig` (`TextModel.open`/`embed`): open_clip CLIP text —
      token+learned-pos embeddings, 12 **pre-LN** blocks, non-causal attention over a fixed
      64-token zero-padded window, gelu-tanh plain FFN, `ln_final` → **last-token pool** (pos
      63) → `text_projection` (+bias) → L2-norm. Reuses the SigLIP2 BPE tokenizer. **Cosine
      vs immich textual ONNX > 0.999999** (test `"siglip2 text tower matches ONNX reference"`;
      ref `testdata/siglip2_text_ref_vectors.json`). Caller frames ids `[content…, <eos>=1]`
      (≤60 tokens) and the model pads to 64. Unlocks the cross-modal text query side.
- [x] **M5 — SigLIP2 visual tower (model C).** ✅ **DONE + VALIDATED (2026-07-22)** in
      `src/models/embed_siglip.zig` (`VisualModel.open`/`embed`): timm ViT-B/16 — conv
      patch-embed (as matmul) → +pos_embed (196 patches) → 12 pre-LN blocks → `trunk.norm`
      → **MAP attention-pool head** (`AttentionPoolLatent`: latent query, kv over tokens,
      `proj` + residual pre-norm MLP) → L2-norm. **Cosine vs immich visual ONNX = 0.99999833**
      on a fixed input tensor (test `"siglip2 visual tower matches ONNX reference"`; input
      `testdata/siglip2_visual_input.f32`, ref `testdata/siglip2_visual_ref.json`). Input =
      decoded RGB 224², `/255`, mean/std 0.5, CHW (preprocessing stays in DiffKeep). Validate
      image embeddings' cosine vs ONNX; confirm image↔text cross-modal retrieval works.
- [x] **M6 — Backends + API polish. ✅ DONE (2026-07-22).** `tp.embed` façade
      (`src/embed.zig`, re-exported from `root.zig`): `TextEncoder.open(gpa, io, kind, dir,
      backend)` + `.embedText(gpa, text, .{.role})` / `.embedTextBatch`; `ImageEncoder.open(…,
      backend)` + `.embedImage` / `.embedImageBatch`. Handles tokenizer load, role prefixes,
      framing, pooling, L2-norm; caller-owned unit 768-d (`tp.embed.dim`). `backend` =
      `.cpu` | `.{ .vulkan = *gpu.Context }` | `.{ .cuda = *cuda.Backend }`.
      **All four encoders run on BOTH Vulkan and CUDA**, each on-device-validated vs the CPU
      forward on an RTX 3090 (8 device forwards; parity cos ≥ 0.999, gated by `-Dintegration`):
      `embed_siglip_gpu`/`_cuda` (Text+Visual), `embed_gemma_gpu`/`_cuda`, `embed_snowflake_gpu`/`_cuda`.
      GPU forwards borrow the CPU model's f32 weights (no dequant) and run the transformer
      body device-side (opMatmul/opConvF16 + norm/attn/rope/gelu ops), head on host. Façade
      GPU dispatch validated (`"embed façade Vulkan matches CPU"`). **Deferred:** true
      batched forwards (see M8; batch APIs currently loop per item), int8.
- [ ] **M7 — DiffKeep integration.** Swap `onnx_embedder.zig` for the TP encoders behind
      DiffKeep's existing embed call sites (`scan.zig`, `search.zig`); **drop the
      `onnxruntime` dependency** and its build wiring. Re-run DiffKeep's
      `search_quality_corpus` eval to confirm no retrieval-quality regression.
- [x] **M8 — True batched forwards (throughput, optional). DONE + VALIDATED (2026-07-22),
      all three backends.** **CPU:** All four encoders now have a fused `embedBatch` that packs B items into
      one contiguous (ragged, no-padding) activation: every GEMM (`m = sum(seq_i)`), LayerNorm/
      RMSNorm, RoPE, GeGLU, patch-embed, residual runs **once** over all rows; only attention
      and RoPE loop per item (each attends only itself). Bit-identical to per-item (tests
      `"* embedBatch matches per-item"`, max-abs-diff < 1e-4). The gemma path added a shared
      `transformer.layerForwardBatchedFresh` (reuses the tested `qkvProject`/`mlpBlock`).
      Façade `embedTextBatch`/`embedImageBatch` route the whole slice to one CPU forward (B=1
      and all GPU backends keep the per-item path for now).
      - **Measured CPU throughput (RTX-less; 9800X3D, ReleaseFast, seq~16, `zig build
        embed-bench`), ms/item @ batch 1 → 16 → 32:** snowflake 28.1 → 10.1 → 9.8 (**2.9×**);
        embeddinggemma 22.0 → 7.0 → 6.5 (**3.4×**); siglip-text 34.4 → 15.1 → 15.8 (**2.3×**);
        siglip-visual 80.2 → 67.2 → 70.5 (**~1.15×**, already compute-bound on 196-patch GEMMs).
        Text encoders gain most (small seq → fork/join + weight-reuse amortization dominates).
      - **GPU (Vulkan + CUDA) DONE + ON-DEVICE VALIDATED (2026-07-22, RTX 3090).** All 8
        device forwards (`embed_{snowflake,gemma,siglip}_{gpu,cuda}`) gained an `embedBatch`
        mirroring the CPU packing: GEMMs run once over `total` rows (`opMatmul`/`opConvF16`
        take `m`); RoPE + attention loop per item. **CUDA** uses a new `DeviceBuffer.viewF32`
        (non-owning offset view, built on the existing `dbOffset`) — no kernel change.
        **Vulkan** gained element-offset push fields on the `rope_half` (`u5`) and `attn_full`
        (`u4`=q/out, `u5`=k/v) SPIR-V kernels (default 0 = unchanged for all existing callers,
        verified). q_dim ≠ kv_dim (gemma GQA) handled with distinct q/kv offsets. Bit-close to
        per-item on-device (8 gated tests `"* {Vulkan,CUDA} embedBatch matches per-item"`,
        max-abs-diff < 1e-3/1e-4; ran green under `-Dintegration`). Façade routes every backend
        through the batched forward for B > 1.
      - **Measured GPU throughput (RTX 3090, best-of-4 via `zig build embed-bench` — NEVER the
        cache-glob binary, see CLAUDE.md), ms/item @ batch 1 → 8 → 16:**
        **Vulkan** (with `beginBatch`) snowflake 33.6 → 7.5 → 5.3 (**6.3×**), gemma
        89 → 38 → 35 (**2.6×**), siglip-text 43 → 17 → 15 (**2.9×**), siglip-visual 70 → 43 → 42
        (**1.68×**). **CUDA** snowflake 52 → — → 7.9 (**6.6×**), gemma 82 → — → 35 (**2.3×**),
        siglip-text 59 → — → 23 (**2.6×**), siglip-visual 100 → — → 65 (**1.5×**). (Batch=1 GPU is
        ~1.5× slower than CPU — launch-bound — but batching more than recovers it.)
      - **Why (from the `--prof` per-op device profiler):** the path is **per-op-overhead-
        bound**, not compute-bound — ms/op is ~constant as the batch widens the matmuls (CUDA
        snowflake matmul 0.45 ms/op @ m=18 vs 0.42 ms/op @ m=144), i.e. ~1000× the actual FLOPs'
        cost. Batching amortizes the **GEMM** ops (one wide matmul for B items instead of B) but
        the **attention + RoPE ops stay a per-item loop**. So the win tracks how GEMM-dominated
        the model is: snowflake (12 layers) → attention is 48% of device ops at B=8 → biggest
        batch win; **gemma (24 layers) → attention is 82%** (192 ops = 8×24) → smallest win;
        siglip-visual's 196-token attention is heavy per op → smallest win.
      - **Vulkan `beginBatch` landed (2026-07-22):** the Vulkan embed forwards previously
        submitted+fenced **every** op (`opEnd`→`submitAndWait`, no `beginBatch` — unlike the
        CUDA forwards); they now wrap the layer loop in `beginBatch`/`endBatch` so the whole
        body is one submission (norm-weight `nbuf` uploads are pointer-cached, so they don't
        flush after warmup). Correctness re-validated (21/21 Vulkan parity tests). Clean idle-GPU
        reading on snowflake: batch 1/16 41.7→36 / 6.7→5.85 ms/item (~13%). Full clean sweep
        pending a quiet GPU (dev card shared with a live tp-gui session — verify with
        `nvidia-smi` before trusting GPU perf; see the stale-binary + contention caveats).
      - **GPU deep-dive (2026-07-22) — two real bottlenecks found; the GPU was NOT compute-bound.**
        1. **Per-call weight re-upload (CUDA) — FIXED.** Every CUDA embed forward wrapped itself
           in `weightScopeBegin/End`, and `weightScopeEnd` *frees* the cached weights — so the whole
           ~300M model was re-converted-to-f16 + re-uploaded to VRAM **every call** (the scope is
           for the transient diffusion ViT, wrong for a persistent encoder). Removed it from all 6
           CUDA forwards (weights borrow the mmap-stable CPU model → the pointer-keyed cache is valid
           indefinitely; parity 20/20 green). Batch-1 (query latency) dropped hard: CUDA snowflake
           52→13, gemma 82→44, siglip-text 59→26, siglip-visual 100→65 ms/item. (Vulkan doesn't have
           this bug — its `opMatmul` cache persists.)
        2. **Attention kernel was parallelism-starved — FIXED with a block-diagonal batched kernel.**
           Probe (attention disabled, resident weights): gemma CUDA batch-16 = **1.55 ms/item** —
           the entire model minus attention. So attention was ~95% of gemma's GPU time (~29 ms/item),
           constant per-item (the B per-item launches serialize). Cause: the old `attn` kernel is
           one-thread-per-(query,head); gemma's short seq (~18) × 3 heads ≈ 54 threads on a
           10,496-core GPU. **Fix (built + validated, both backends):** a block-diagonal batched
           kernel — all B items' query rows in ONE launch (B× threads fill the GPU), each attending
           only its own item's keys via a per-item bounds table. **CUDA:** `attn_batched` PTX +
           `Backend.opAttnBatched`. **Vulkan:** a standalone SPIR-V module `kernels/attn_batched.zig`
           (the eltwise module is at the Zig-SPIR-V per-module entry-point limit — adding any kernel
           there segfaults the compiler) + a dedicated 5-buffer pipeline + `Context.attnBatchedBind`/
           `attnBatchedDispatch`. Parity 16/16 per backend.
      - **FINAL batch-16 throughput, ms/item — GPU wins every encoder on both backends** (GPU under
        some GUI contention, so conservative): snowflake CPU 9.8 / CUDA 0.97 / Vulkan 2.6; gemma
        7.0 / 3.9 / 6.7; siglip-text 15 / 2.2 / 3.9; siglip-visual 67 / 11.1 / 14.8. The batched
        kernel gave 5–13× over the per-item attention loop. (siglip-visual's 196×196 attention is
        heavier per op; a tiled attention kernel could push it further, but it already beats CPU.)

## Validation strategy

DiffKeep already ships the ONNX models and an eval corpus — use them as ground truth:
- **Per-model numeric parity:** feed identical inputs to ONNX (in DiffKeep) and TP;
  require cosine(TP, ONNX) above a high threshold (~0.999) per vector. Small drift is
  fine — DiffKeep's search is rank-based with min-similarity floors.
- **Tokenizer parity:** exact token-id match vs `onnx_tokenizers.zig` on a corpus.
- **End-to-end:** DiffKeep's `search_quality_corpus.zig` / `test_search_quality`
  before/after the swap — retrieval quality must not regress.

## Open questions / decisions to lock in M0

0. **Text encoder: A1 (Snowflake) vs A2 (EmbeddingGemma).** DiffKeep runs its
   `search_quality_corpus` eval on both and picks the one with sufficient retrieval
   quality; A2 is far cheaper for TP to implement (reuses Gemma-3 + Gemma tokenizer) and
   is newer/higher-MTEB, but switching costs DiffKeep a one-time full re-embed of its text
   index. Also decide whether TP ships *both* (they share the M2 tier) or only the winner.
1. **SigLIP2 resolution.** ✅ **RESOLVED (M0): 224.** DiffKeep's immich export ==
   `timm/ViT-B-16-SigLIP2` (base, no suffix) — byte-identical tokenizer.json + config
   (`image_size:224`, `vit_base_patch16_siglip_224`). 14×14 = **196 patches**, squash
   resize, mean/std 0.5, bicubic. See M0 spec appendix.
2. **GTE specifics for Snowflake.** ✅ **RESOLVED (M0):** CLS pooling, **GeGLU** MLP
   (packed `up_gate_proj` 768→6144→split, `down_proj` 3072→768), **LayerNorm w/ bias**
   eps 1e-12 (not RMSNorm), RoPE θ=160000, no projection head. See M0 spec appendix.
3. **Compute dtype.** Start **f32/f16** (all three checkpoints ship f32 safetensors; a
   768-d encoder is cheap). TP's int8 scheme ≠ ONNX's — pursue int8 only if throughput
   demands it.
4. **Model-file layout / distribution.** Weights staged under `../DiffKeep/Models/`
   (gitignored) next to DiffKeep's ONNX ground-truth. Ownership of HF→safetensors still
   TBD, but no *conversion* is needed — all three ship native safetensors (see appendix).

## Dead ends already ruled out
- **ONNX graph runtime in TP** — rejected; against TP's "own the whole tower" design and
  a much larger project than three native archs.
- **llama.cpp / mtmd path** — DiffKeep already abandoned its old Qwen3-VL + mtmd
  captioning/embedding path (db schema v8: "switched from Qwen3-VL to SigLIP2/snowflake").
  Not relevant to this plan.

---

## M0 findings — acquired weights & arch specs

**Status: M0 essentially DONE.** All candidate checkpoints acquired as native
safetensors (no ONNX conversion needed); arch hyperparameters + tensor names extracted
below. Remaining M0 decision is open-Q #0 (DiffKeep's A1-vs-A2 quality eval).

### Acquired weights (all under `../DiffKeep/Models/`, gitignored)

| Model | Dir | Source | Notes |
|---|---|---|---|
| A2 EmbeddingGemma | `embeddinggemma-300m/` | **`unsloth/embeddinggemma-300m`** (ungated mirror, byte-identical to gated `google/…`) | full ST package: base + `1_Pooling` + `2_Dense` + `3_Dense` |
| A1 Snowflake | `snowflake-arctic-embed-m-v2.0/` | `Snowflake/snowflake-arctic-embed-m-v2.0` | added `model.safetensors`+configs; kept existing tokenizer + `onnx/model_int8.onnx` |
| B+C SigLIP2 | `ViT-B-16-SigLIP2-timm/` | `timm/ViT-B-16-SigLIP2` (base=224) | one `open_clip_model.safetensors` holds **both** towers + MAP head + text projection |

**Parity ground-truth (already in DiffKeep):** `snowflake-arctic-embed-m-v2.0/onnx/model_int8.onnx`,
`ViT-B-16-SigLIP2__webli/{visual,textual}/model.onnx`. (Note: DiffKeep's ONNX is int8;
TP computes f32/f16 — expect small numeric drift, hence the ~0.999 cosine floor.)

*Gate note:* `google/embeddinggemma-300m` is gated (manual approval / contact-sharing).
We deliberately route around it via the `unsloth` mirror (Gemma license permits
redistribution; file list + sizes verified byte-identical to the official repo).

### A2 — EmbeddingGemma 300M (`Gemma3TextModel`, bidirectional)
- 24 layers · hidden 768 · **GQA** 3 Q-heads / 1 KV-head · **head_dim 256** ·
  `query_pre_attn_scalar` 256 (attn scale 1/√256) · **QK-norm** (`q_norm`/`k_norm` [256]).
- MLP: **GeGLU** `gate_proj`+`up_proj` 768→1152, `down_proj` 1152→768, `gelu_pytorch_tanh`.
- Gemma-3 **4-norm** layout (`input`/`post_attention`/`pre_feedforward`/`post_feedforward_layernorm`),
  RMSNorm eps 1e-6.
- **Dual RoPE**: θ=1e6 on `full_attention` layers (every 6th: 5/11/17/23), θ=1e4 +
  sliding_window 512 on the rest. `use_bidirectional_attention: true`.
- Vocab **262144** (Gemma-3 tokenizer, `tokenizer.model` 4,689,074 B). max_pos 2048.
- **Head pipeline** (`modules.json`): Transformer → **mean pool** (`include_prompt:true`)
  → Dense 768→3072 (no bias/act) → Dense 3072→768 (no bias/act) → **L2 normalize**.
  MRL: truncate final vec to 512/256/128 then renormalize.
- **Prompts:** query `task: search result | query: {q}` · document `title: none | text: {doc}`.
- **TP reuse:** ≈ existing gemma3 text forward. New only: bidirectional mask (have it),
  mean-pool, 2 plain GEMMs, drop LM head.

### A1 — Snowflake Arctic Embed M v2.0 (`GteModel`, bidirectional)
- 12 layers · hidden 768 · 12 heads (head_dim 64) · **packed `qkv_proj` [2304,768]+bias**,
  `o_proj`+bias.
- MLP: **GeGLU** packed `up_gate_proj` [6144,768] (→ split 3072 gate/3072 up), `down_proj`
  3072→768, all +bias · `hidden_act` gelu.
- **LayerNorm (with bias)** eps **1e-12** (`attn_ln`, `mlp_ln`, `embeddings.LayerNorm`) —
  *not* RMSNorm. Single **RoPE θ=160000**. Learned `token_type_embeddings` [1,768] (single type).
- Vocab **250048** · XLM-R **SentencePiece Unigram** tokenizer (`XLMRobertaTokenizer`) ·
  specials `<s>`=0(CLS)/`<pad>`=1/`</s>`=2/`<unk>`=3/`<mask>`=250001 · max 512 (model 32768).
- **Head:** **CLS pooling** (token 0) → **L2 normalize**. No dense projection. MRL dim [256].
- **Prompts:** query `query: {q}` · document raw.
- **TP reuse:** needs a LayerNorm+GELU+CLS encoder body (new-ish) **and** the Unigram
  tokenizer + `tokenizer.json` loader (new-work #2/#3).

### B — SigLIP2 ViT-B-16 text tower (open_clip, bidirectional)
- 12 resblocks · width 768 · 12 heads · packed `attn.in_proj_weight` [2304,768]+bias,
  `attn.out_proj`+bias · MLP `c_fc` 768→3072 + `c_proj` 3072→768 (gelu-tanh) · `ln_1`/`ln_2`
  **LayerNorm w/ bias**, `ln_final`, eps 1e-6.
- **Learned `positional_embedding` [64,768]** (NOT RoPE) · context_length 64 · `no_causal_mask`.
- **Pooling `last`** (take last/EOS-position token) → **`text_projection` [768,768]+bias** → 768-d.
- Vocab **256000** — Gemma tokenizer, **but the 256k variant** (`timm/ViT-B-16-SigLIP2`,
  `clean:canonicalize`), **≠ EmbeddingGemma's 262144 Gemma-3 tokenizer.** ⚠️ So B likely
  still needs the `tokenizer.json` loader even if A2 wins — verify TP's gemma tokenizer
  coverage of the 256k vocab in M1.

### C — SigLIP2 ViT-B-16 visual tower (open_clip / timm ViT)
- `visual.trunk.blocks.N` timm ViT-B/16, patch 16, **image 224 → 196 patches**, width 768,
  12 layers/heads · **MAP attention-pool head** (`timm_pool: map`) → 768-d (proj `none`).
- Preprocess (DiffKeep owns): RGB 224², `/255`, mean/std 0.5, bicubic, **squash** (no crop), CHW.
- **TODO:** dump exact `visual.*` head/pos-embed tensor names before M5 (MAP head layout).
