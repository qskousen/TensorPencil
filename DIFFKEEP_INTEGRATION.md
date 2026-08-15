# Integrating TensorPencil embedding encoders into DiffKeep

Handover for the DiffKeep-side agent. This explains how to call TensorPencil (TP)
as a native Zig library to produce **all** of DiffKeep's semantic-search embedding
vectors — the text space (Snowflake / EmbeddingGemma) **and** the SigLIP2
cross-modal image/text space — fully replacing the `onnxruntime` path
(`src/onnx_embedder.zig` + `src/onnx_tokenizers.zig`).

TP and DiffKeep are sibling checkouts (`../TensorPencil`, `../DiffKeep`), both on
**Zig 0.16.0**.

---

## 1. Status — what's ready

**All four encoders** — both text-space candidates AND both SigLIP2 towers — are
implemented natively in TP and validated on CPU, so DiffKeep can drop
`onnxruntime` **entirely**:

| Model | Role | TP type | Tokenizer | Validated (cosine vs ONNX) |
|---|---|---|---|---|
| **Snowflake Arctic Embed M v2.0** (current text model) | text `.semantic` | `models.embed_snowflake.Model` | Unigram | **0.9999998** |
| **EmbeddingGemma 300m** (newer text candidate) | text `.semantic` | `models.embed_gemma.Model` | BPE | **0.999985** |
| **SigLIP2 ViT-B-16 text tower** | cross-modal query | `models.embed_siglip.TextModel` | BPE (256k) | **>0.999999** |
| **SigLIP2 ViT-B-16 visual tower** | image index | `models.embed_siglip.VisualModel` | — (RGB in) | **0.99999833** |

All produce **L2-normalized 768-d `f32`** vectors, matching what DiffKeep stores in
its sqlite-vec `vec0` tables. Tokenizers are **bit-exact** vs HuggingFace
`tokenizers`; the Snowflake tokenizer also reproduces DiffKeep's existing
`onnx_tokenizers.zig` output (0 divergences on the test corpus), so **switching to
TP-Snowflake does not require re-embedding** the existing text index. (The SigLIP2
towers are numerically ~identical too, but re-embedding the image index is cheap
and safest when you cut over — verify with the eval corpus.)

**Backends:** all four encoders run on **CPU, Vulkan, and CUDA** (each GPU path is
on-device-validated vs CPU). Pick per encoder via the `backend` argument (§4). CPU
is plenty for indexing (~300M params, f32); GPU is there if you want it.

**NOT yet done** (do not depend on these):
- **True batched GEMM** — `embedTextBatch`/`embedImageBatch` loop per item
  (throughput comes from each forward's internal parallelism); int8 is out of scope.
- **Image decode / resize** stays in DiffKeep (libvips): the image encoder takes an
  already-decoded, preprocessed `f32` CHW tensor (see §4).

---

## 2. Model files on disk

TP loads **safetensors + `tokenizer.json`** (it does not use ONNX). DiffKeep
currently ships the ONNX + tokenizers but **not** the safetensors, so add them to
the model directories (the `tokenizer.json` you already have is reused as-is):

- **Snowflake** dir must contain: `model.safetensors` (+ existing `tokenizer.json`).
  Source: `Snowflake/snowflake-arctic-embed-m-v2.0` (`model.safetensors`, ~1.2 GB f32,
  Apache-2.0).
- **EmbeddingGemma** dir must contain the sentence-transformers package:
  `model.safetensors`, `2_Dense/model.safetensors`, `3_Dense/model.safetensors`
  (+ its `tokenizer.json`). Source: `unsloth/embeddinggemma-300m` (ungated mirror of
  the gated `google/embeddinggemma-300m`; byte-identical, Gemma license).
- **SigLIP2 (both towers)** dir must contain a single `open_clip_model.safetensors`
  (both towers + MAP head + text projection in one file, ~1.5 GB f32) plus the text
  tower's `tokenizer.json`. Source: `timm/ViT-B-16-SigLIP2` (the base = 224 variant;
  its `tokenizer.json` is byte-identical to the immich export DiffKeep already ships).
  Confirm resolution **224** (14×14 = 196 patches).

Compute dtype is `f32` (the encoders are cheap; int8 is out of scope).

---

## 3. Build wiring

TP publishes a module named **`TensorPencil`** (package name `.TensorPencil`,
`build.zig.zon`).

In DiffKeep's `build.zig.zon`, add a path dependency:

```zig
.dependencies = .{
    // ... existing ...
    .TensorPencil = .{ .path = "../TensorPencil" },
},
```

In DiffKeep's `build.zig`, import the module into the exe/lib that does embedding:

```zig
const tp_dep = b.dependency("TensorPencil", .{
    .target = target,
    .optimize = optimize,
    // The f32 embedding path does NOT need ggml's quant kernels. Try disabling
    // ggml to avoid pulling in its C library + libc++:
    .ggml = false,
});
exe.root_module.addImport("TensorPencil", tp_dep.module("TensorPencil"));
```

> **ggml note:** ggml is an optional TP dependency (default **on**) providing fast
> AVX2 block-quant kernels. The embedding models compute in `f32`, whose GEMM path
> is native Zig, so `.ggml = false` should build and run fine and keeps DiffKeep's
> link simple. If the build complains, drop `.ggml = false` (default on) — then TP
> links `libggml` (needs `libc`/`libc++`), fetched automatically via `zig fetch`.

Build in **ReleaseFast** for indexing throughput (Debug f32 matmul is very slow).

---

## 4. API — the `tp.embed` façade

Use the high-level façade `tp.embed`. It handles the per-model tokenizer, role
prefixes, frame tokens, pooling, and L2 normalization for you — you pass text (or
a decoded image) and get a caller-owned unit-norm 768-d `[]f32`.

```zig
const tp = @import("TensorPencil");

// backend: .cpu | .{ .vulkan = ctx } | .{ .cuda = be }  (GPU handle must outlive the encoder)
const backend: tp.embed.Backend = .cpu;

// TEXT — pick one kind: .arctic_embed_m_v2 | .embeddinggemma | .siglip2_text
var enc = try tp.embed.TextEncoder.open(gpa, io, .arctic_embed_m_v2,
    "Models/snowflake-arctic-embed-m-v2.0", backend); // dir holds safetensors + tokenizer.json
defer enc.deinit();

const q = try enc.embedText(gpa, "a red bicycle", .{ .role = .query });    // 768-d, L2-normed
defer gpa.free(q);
const d = try enc.embedText(gpa, some_document_text, .{ .role = .document });
defer gpa.free(d);

// Batch (DiffKeep indexes ~8 at a time): one caller-owned vector per input.
const vecs = try enc.embedTextBatch(gpa, &.{ "cat", "dog" }, .{ .role = .document });
defer { for (vecs) |v| gpa.free(v); gpa.free(vecs); }

// IMAGE — SigLIP2 visual. Feed your existing libvips CHW tensor unchanged.
var ienc = try tp.embed.ImageEncoder.open(gpa, io, "Models/ViT-B-16-SigLIP2", backend);
defer ienc.deinit();
const img: []f32 = try loadImageAsTensor(...);   // 3*224*224, /255, mean/std 0.5, CHW
const iv = try ienc.embedImage(gpa, img);         // 768-d, L2-normed
defer gpa.free(iv);
```

**Backend:** pass `.cpu` (default choice — plenty for indexing), or a live GPU
handle: `.{ .vulkan = vk_ctx }` (`*tp.gpu.context.Context`) or `.{ .cuda = cu_be }`
(`*tp.gpu.cuda.Backend`). The handle must outlive the encoder; open the encoder
once and reuse. The GPU paths run the transformer body on-device and are
validated bit-close to CPU.

Mapping to DiffKeep's roles:
- **text `.semantic` space** → `TextEncoder(.arctic_embed_m_v2)` (or `.embeddinggemma`);
  use `.role = .query` for search text, `.role = .document` for indexed prompt text.
- **cross-modal query** → `TextEncoder(.siglip2_text)` (role ignored — symmetric).
- **image index** → `ImageEncoder`.

Notes:
- `io: std.Io` + `gpa` come from your `main`/worker. `TextEncoder`/`ImageEncoder`
  own their model + tokenizer (mmap ~1.2 GB) — **open once, reuse**. They are
  `*const` for the embed calls but not internally synchronized: one per worker
  thread, or guard with a mutex. `embedTextBatch` currently loops (throughput comes
  from each forward's internal threading).
- Output is **always L2-normalized 768-d** (`tp.embed.dim == 768`). Do **not**
  re-normalize.

<details><summary>Low-level API (only if you need to bypass the façade)</summary>

`tp.tokenizer.Tokenizer` (`initUnigramFromTokenizerJson` / `initGemma4FromTokenizerJson`,
`.encode` → content ids) + `tp.models.embed_snowflake.Model` /
`embed_gemma.Model` / `embed_siglip.{TextModel,VisualModel}` (`.open(gpa, io, dir)`,
`.embed(io, gpa, ids, out)`). You then apply prefixes + frames yourself:

| | Snowflake | EmbeddingGemma | SigLIP2 text |
|---|---|---|---|
| query prefix | `"query: "` | `"task: search result \| query: "` | *(none)* |
| document prefix | *(none)* | `"title: none \| text: "` | *(none)* |
| frame | `<s>=0 … </s>=2` | `<bos>=2 … <eos>=1` | `… <eos>=1` (pad→64) |
| pooling | CLS | mean | last token |

</details>

---

## 5. Where to wire it in DiffKeep

- Replace **all** embedding call sites in `onnx_embedder.zig` / `onnx_tokenizers.zig`
  (`scan.zig` indexing, `search.zig` queries) with `tp.embed.TextEncoder` /
  `ImageEncoder` (§4) — one encoder per space, opened once and reused.
- Keep everything **below** the encoder unchanged: sqlite-vec storage, KNN/`MATCH`,
  Reciprocal-Rank-Fusion, min-similarity floors, FTS keyword mode. Keep your libvips
  image decode/resize (feed its output tensor to `VisualModel.embed`).
- **`onnxruntime` can be dropped entirely** — all four encoders are now native TP.
  Remove the onnxruntime build wiring and the `*/onnx/*.onnx` model files; keep the
  `tokenizer.json` files (TP reads them) and add the safetensors from §2.

---

## 6. Choosing the model (A1 vs A2)

This is DiffKeep's call, via its `search_quality_corpus` / `test_search_quality`
eval:
- **Snowflake (A1)** — the current model; TP reproduces its exact tokenization, so
  it drops in with **no re-embed**. Safest.
- **EmbeddingGemma (A2)** — newer, higher MTEB, Matryoshka-truncatable (768→512/256/128).
  Switching means a **one-time full re-embed** of the text index (different embedding
  space) and using the A2 prefixes above.

TP can carry **both** (they share the tokenizer/encoder infra), so you can A/B them
on the eval corpus before committing.

---

## 7. Limitations / gotchas

- **Backends:** CPU / Vulkan / CUDA all work (per-encoder `backend` arg). CPU is
  fine for indexing; build ReleaseFast either way.
- **EmbeddingGemma asserts sequence length ≤ 512 tokens** (post-framing). Its local
  layers use a 512 sliding window that TP currently runs as full attention (correct
  for ≤512). Truncate longer inputs, or file a TP request for the long-doc path.
  Snowflake's own limit is 512 tokens (XLM-R), so cap inputs at 512 either way.
- **Output is already L2-normalized** — do not normalize again (DiffKeep's
  `sim = 1 - d²/2` under sqlite-vec L2 relies on unit vectors).
- **No batch API** yet — loop per text/image (threads for parallelism).
- **SigLIP2 text** caps at a 64-token window (right-padded) per SigLIP; pass ≤ ~60
  content tokens + `<eos>`. **SigLIP2 visual** is fixed 224×224 → 196 patches; keep
  DiffKeep's libvips preprocessing (`/255`, mean/std 0.5, CHW) unchanged.
- **Output is already L2-normalized** — do not normalize again (DiffKeep's
  `sim = 1 - d²/2` under sqlite-vec L2 relies on unit vectors).
- Parity floor is cosine ≥ 0.999 vs the reference ONNX; measured is ≥ 0.99998 for
  every encoder, so retrieval quality is effectively identical. Re-run DiffKeep's
  eval after the swap to confirm no regression.
