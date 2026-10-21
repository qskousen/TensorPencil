# DIFFKEEP_GEMMA2.md: EmbeddingGemma 2 for DiffKeep

Goal: replace DiffKeep's encoder set (A1 Snowflake, A2 EmbeddingGemma 300M, B/C SigLIP2,
see `DIFFKEEP.md`) with ONE model, EmbeddingGemma 2, after it passes a retrieval-quality
check on DiffKeep's own `search_quality_corpus`.

Scope: TP computes vectors and nothing else. Storage, indexing, KNN and ranking stay in
DiffKeep, as in `DIFFKEEP.md`.

## Status

Text and image are implemented on CPU, Vulkan and CUDA, behind
`tp.embed.MultimodalEncoder` (text and images into one space) and
`tp.embed.TextEncoder(.embeddinggemma2)` (text only). Audio and video are not.

| check | result |
|---|---|
| text vs sentence-transformers, 7 cases incl. 1378 tokens | cosine >= 0.9999999 (bf16 control: 0.99995) |
| tokenizer + prompt + frame vs transformers ids | exact |
| image vs sentence-transformers, 3 PNGs (down, up, 1:4.7) | cosine >= 0.99999994 |
| resize vs torchvision | byte-exact |
| device vs CPU, both backends | text >= 0.9999 every case (0.999999 on the one read out), images >= 0.999996 |

Weights: `google/embeddinggemma-2` (Apache 2.0, ungated, 1.49 GB bf16) in
`../DiffKeep/Models/embeddinggemma-2/`. The fixture generators need transformers from main
and sentence-transformers >= 6.1; the overlay venv at `/dump/venv-eg2` has them.

## The model, from its config and transformers' code

- Text core (`models/embed_gemma2.zig`): 24 bidirectional Gemma 4 layers, width 512, 4
  query heads. Local layers: head 256, 2 KV heads, rope 1e4, keys within 512 positions
  either side (`|q - k| <= 512`). Every 6th layer global: head 512, 1 KV head, rope 1e6.
  Attention scale 1.0, weightless V norm, norm weights without Gemma 3's +1.
- Per-layer inputs: one projection of the input embeddings to 24 x 512, normed, gating a
  third residual block per layer, then a per-layer scalar. No per-layer token table.
- Head: final norm, 512 -> 768 projection, mean pool over every token, L2 normalize.
- Image (`models/gemma4v_vit.zig`, shared with Gemma 4 chat): 16-block Gemma 4 ViT, patch
  16, tanh GELU, 3x3 pool to 280 tokens by default, weightless RMS norm, 768 -> 512 into the
  text model as `<bos> <|image> soft... <image|> <eos>`. Images take no prompt.
- Matryoshka: truncate to 512/256/128 and RE-NORMALIZE (`tp.embed.truncate`). The model
  card puts 256 near-lossless; 128 is text-only quality.
- Prompts: query `task: search result | query: {q}`, document `title: none | text: {doc}`.

## Measured cost (RTX 3090 shared with another process, ReleaseFast, ms/item)

| | batch 1 | batch 16 |
|---|---|---|
| CPU | 26.4 | 8.4 |
| CUDA | 55.7 | 4.8 |
| Vulkan | 58.6 | 4.5 |

Text at ~16 tokens via `zig build embed-bench -- --model gemma2`. About 20-25% over
EmbeddingGemma 300M. Single queries are fastest on the CPU; bulk encoding on the GPU.

Images, 1024x768 RGB8 in (`--model gemma2-image [--budget N] [--stages]`), ms/image:

| tokens | CPU | CUDA batch 1 | CUDA batch 8 | Vulkan batch 1 | Vulkan batch 8 |
|---|---|---|---|---|---|
| 280 (default) | 1657 | 123 | 110 | 145 | 129 |
| 140 | 642 | | 55 | | 61 |
| 70 | 279 | | 32 | | 36 |

At 280 a CUDA image is preprocessing 14 ms (CPU resize), the vision tower 69 ms and the
text body 34 ms; batching only amortizes the text body, since the tower runs per image.
A budget changes the vectors, so an index is built at one budget; its retrieval cost
is unmeasured.

## Next

1. Retrieval-quality A/B against the current four encoders on `search_quality_corpus`
   (DiffKeep side). Replace only on a win.
2. DiffKeep swaps to `MultimodalEncoder`: one table, no rank fusion, and a re-embed of
   both indexes (a new space).
3. Image throughput: batch the vision tower (one image per call now) and overlap the CPU
   resize with the device. Pick a token budget by retrieval quality, not speed alone.
4. Audio (Gemma 4 conformer, 300M) and video (frames through the image tower), if DiffKeep
   wants them.
