# DIFFKEEP_GEMMA2.md: EmbeddingGemma 2 for DiffKeep (deferred)

Status: ON HOLD until the LoRA-for-every-family/backend hoist is done.

Goal: replace DiffKeep's encoder set (A1 Snowflake, A2 EmbeddingGemma 300M, B/C SigLIP2,
see `DIFFKEEP.md`) with ONE model, EmbeddingGemma 2, after it passes a retrieval-quality
check on DiffKeep's own `search_quality_corpus`.

Scope: TP computes vectors and nothing else. Storage, indexing, KNN and ranking stay in
DiffKeep, as in `DIFFKEEP.md`.

## The model

Google DeepMind, released 2026-10-06, Apache 2.0, `google/embeddinggemma-2` (safetensors).
Facts below are from the model card and launch coverage; verify against `config.json`
before porting.

- One 768-d space for text, code, image, video, audio. A text query retrieves a picture
  directly, so the two-space Reciprocal-Rank-Fusion setup DiffKeep runs today is no longer needed.
- Modular, one checkpoint: 270M text core (about 130M transformer + 140M embedding table),
  optional 170M vision encoder, optional 300M audio encoder. Any subset shares the space.
- Text core: Gemma 4 lineage, BIDIRECTIONAL, 24 layers, width 512, 5:1 local:global with a
  1024 window, mean pool, dense 512 -> 768. Our A2 port (`Gemma3TextModel`, bidirectional)
  and the Gemma 4 LLM path are the starting points.
- 8K context. Image: 280 tokens by default (70-1120 configurable). Video: 140/frame at
  1 fps. Audio: 25 tokens/s, mono 16 kHz.
- Matryoshka: truncate to 512/256/128, then RE-NORMALIZE. 256 is near-lossless
  (MTEB en 68.46 -> 67.78); 128 collapses multimodal (MMEB 59.01 -> 45.65). 256 is the likely output size.
- Task prefixes are required; leaving them off degrades quality silently.
  Query: `task: search result | query: {q}`. Document: `title: {t} | text: {body}`.
  Symmetric: `task: classification | query: {x}` on both sides.
- f16 gives NaN or silently degraded embeddings (activation range). bf16 on GPU, f32 on CPU.

Open questions:
- Does the vision tower match Gemma 4's? If yes, mostly reuse; if not, a new ViT.
- Audio and video: in scope for DiffKeep, or text + image only (440M)?
- Does TP take image preprocessing (variable token budget) or does DiffKeep's libvips
  path keep it?

## Where it runs

- Query text: CPU. About 130M of transformer on a few dozen tokens should be a few ms
  (estimated, not measured), avoids contention with diffusion/LLM on the GPU, and
  needs no device load.
- Bulk encoding: GPU through the VRAM arbiter, CPU fallback. Google quotes 37 ms/image
  on an M5 Pro GPU at a 70-token budget.
- Image-to-image queries: either.
- CPU and GPU vectors differ by float noise only; encoding the library on one and querying on the
  other is fine with the same weights.

## First steps when this resumes

1. `tools/gen_embeddinggemma2.py`: run sentence-transformers on a fixed corpus (text,
   images; prefixes on; 768 and 256), write the fixture, as for A1/A2/B/C.
2. Text core on CPU against it, reusing the A2 port; then vision.
3. Retrieval-quality A/B vs the current four encoders on `search_quality_corpus`. Replace
   only on a win.
4. GPU path for bulk encoding; update `BACKEND.md` and `DIFFKEEP_INTEGRATION.md`.
