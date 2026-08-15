# Handoff: llama/Mistral (Mistral-Nemo, IQ4_NL) — Vulkan stepper

Continuation point for making `Impish_Bloodmoon_12B-ARM_HA_NL.gguf` (Mistral-Nemo,
GGUF arch `llama`, mostly IQ4_NL) run on **every backend**. CPU + CUDA are **done,
validated token-identical to llama.cpp, and merged into the working tree** (build
+ `zig build test` green on `master`, no regressions). Only the **Vulkan LLM
stepper** remains.

Model: `/home/qt/genai/lmstudio/models/Impish_Bloodmoon_12B-ARM_HA_NL.gguf`
(40 layers, hidden 5120, FFN 14336, GQA 32/8, head_dim 128, rope θ=1e6, vocab
131074, rms_eps **1e-5**, ChatML, tekken BPE; quants IQ4_NL(196)+Q8_0(80)+Q5_K(5)
+F16 embd+F32 norms).

## Status

| Backend | State |
|---|---|
| CPU | ✅ done + validated (`"Thursday, Friday, Saturday, Sunday."` token-identical to llama.cpp; open prompts coherent) |
| CUDA (3090) | ✅ done + validated + optimized (**35 tok/s**; llama.cpp is 81 — the residual 2.3× is TP's *general* decode gap, same as q4_k, out of scope) |
| Vulkan | ✅ **done + validated + prefill-optimized** — `VulkanLM` is config-driven with a `gemvW` block-quant path (decode → `opGemvQuantT`), optional QK-norm, F16 embed, runtime vocab/eps. Nemo greedy = `"Thursday, Friday, Saturday, Sunday."`; qwen3-VL bf16 Vulkan path still coherent + gated tests pass. **Decode ~16.5 tok/s**; **prefill via tensor-core GEMM** (`opMatmulCoopQuant`) — a 411-token prompt prefills in ~0.1 s of marginal work (was ~37 s single-token), so first-token latency on a long prompt dropped from ~39 s to ~6 s (now dominated by the one-time ~4.5 s weight transpose/upload) |

## Vulkan: how it landed

`src/models/qwen3_gpu.zig VulkanLM` now serves two regimes off `lm.cfg` + a `quant` flag (`lm.head.dtype.isBlockQuant()`):
- **Dense** (bf16/fp8, tied head — Qwen3-VL): unchanged batched square-attention prefill + grouped-GEMV / GEMM + spec decode; the 4-vocab-chunk tied head reads `embed_f32`.
- **Block-quant** (llama/Mistral): `linearQuant` per projection — **decode/short-follow-up (m==1)** → `gemvW`/`opGemvQuantT` (per-row fused dequant, exact); **fresh-prompt prefill (m>1)** → `opMatmulCoopQuant` (tensor-core GEMM, see below). QK-norm skipped via `if (!cfg.qk_norm) return;`. F16 embed host-gathered to `embed_f32` (`convertToF32` handles f16/bf16). Fixed `[Config.max_layers]` KV arrays; all dims/eps/θ/vocab from `cfg`. `LmBufs` cfg-sized + a `quant_partials` buffer. `kvWindowBytes(cfg, cap)` takes cfg. `chunkRows` batches the fresh prompt (len==0) into one square-attention chunk when `can_gemm_prefill`, else 1-token; `stepAll` (spec) verifies each draft with its own 1-token forward.
- **Prefill tensor-core GEMM** (the speed win): `context.opMatmulCoopQuant(dt, …)` dequants a block-quant weight to f16 k-major **on the GPU** and runs the existing f16-weight coopmat GEMM (`coopF16WDispatch`). New SPIR-V kernels in `eltwise.zig`: `dequant_{q8_0,q4_k,q5_k,q6_k,iq4_nl}_f32` (read the resident 32-row-group transposed weight — same buffer decode uses, no 2nd copy — write f32 row-major) + generic `pack_h16_kmajor` (f32 row-major → f16 k-major padded). All `opElt` calls, so the dequant→pack→GEMM chain stays batched and the per-op barriers serialize the reused `deq_f32`/`deq_f16k` scratch (no 32 GB f16 cache — re-dequant per prefill into one buffer). Guarded by `ctx.hasQuantPrefillGemm()` (f16 coopmat present); falls back to per-token GEMV otherwise. Unit test: `"gpu block-quant prefill GEMM matches cpu reference"` (all 5 formats, f16-rounded reference).
- `llm_main.zig` guard relaxed: block-quant layer weights + head allowed on Vulkan; only a block-quant **embedding** is still rejected (no Vulkan gather kernel).
- `encode()` (the Krea2 diffusion text encoder) left on the module consts — untouched.
- **Weight transpose GPU-ified** (2026-07-23): `weightBufferRawT` no longer does the single-thread ~8 GiB CPU byte-scatter into the 32-row-group `_t` layout — it uploads the raw weight to a device scratch and runs a `transpose_grp32` compute pass (`transpose.zig`, byte-permutation, one output u32/invocation). Byte-identical to the old layout (gated GEMV/GEMM tests pass). Also added a `prefill_gemm_min` (32) threshold: fresh prompts shorter than that keep the per-token GEMV (the GEMM's dequant-all-weights pass isn't worth it below the crossover). Net cold-start: short-prompt first token ~3.6 s → ~2.2 s; long-prompt (411 tok) ~5.1 s → ~4.3 s.
- **dp4a decode — DONE, ~2.2× faster, opt-in** (2026-07-23): decode is compute-bound (~12% of bandwidth), so int8 dp4a is the lever. `OpSDot` emits from Zig-SPIR-V inline asm (`dp4a.zig`, separate module); `spv.zig withDotProduct` injects the `DotProduct`/`DotProductInput4x8BitPacked` capabilities + `SPV_KHR_integer_dot_product` extension; the device enables the feature. The first cut (dp4a over the `_t` layout) was **slower** — the 32-apart bytes forced a per-group gather+pack. The fix is a **repacked int8-interleaved layout** (`repack_q8_0`/`repack_iq4_nl` in `transpose.zig`): 4 contiguous quant int8 per `u32`, 32 rows warp-interleaved, f32 row-scale/block, **iq4_nl codebook pre-applied** so q8_0 and iq4_nl share one `gemv_repack_dp4a` kernel (and one `dequant_repack_f32` for the prefill GEMM). Now **9.6 → 20.7 tok/s (~2.2×)** on the 3090, output correct, validated by `"gpu dp4a decode GEMV matches cpu reference"`. **Opt-in via `TP_VK_DP4A=1`**: the repacked weight ~doubles the iq4_nl VRAM (Nemo 7→13 GB resident; fits 24 GB but can OOM tighter configs), so it's not yet the default — making it default needs VRAM-aware auto-context sizing. When on, the repacked layout serves BOTH decode (dp4a) and prefill dequant, so only one resident copy.
- **Remaining perf**: with `TP_VK_DP4A=1` decode is ~20 tok/s (approaching CUDA's 35). Default-on needs the VRAM-budget accounting; f16-MAC (~1.3–1.7×, no VRAM cost) stays an option for the non-dp4a default path.

## What's already landed (context for the Vulkan work)

- **IQ4_NL quant**: `dtype.zig` (enum + info, 32 elems/18 B), `quants.zig`
  (`GGML_TYPE_IQ4_NL`), `gguf.zig` (id 20), `safetensors.zig`, `matmul.zig`.
- **llama arch = generalized qwen3.zig**: `Config` gained `vocab`/`qk_norm`/
  `permute_qk`/`rms_eps`; `detectGguf` reads `llama.*` keys; `loadQK`
  **un-permutes q/k at load** (llama.cpp NORM-rope layout → our rotate-half, so
  every rope/attn kernel is reused unchanged); `forwardCached`/`forwardTree`
  branch `qwen3_spec` vs `transformer.llama_spec` (new `qk_norm=false` spec).
- **tekken tokenizer**: `tokenizer.zig` `pretokenEndTekken`/`caseRun`;
  `unicode_tables.zig` regenerated with lowercase/uppercase ranges
  (`gen_unicode_tables.py`). BOS support (`chat.bos_token`/`appendBosIfStart`).
- **CUDA**: `gemv_iq4_nl` (shared-memory LUT — see gotcha below) + `dequant_iq4_nl_f16`
  in `elt.zig`, `embed_gather_h` (f16 embed), all wired in `backend.zig`;
  `qwen3_cuda.zig` `CudaLM` generalized (cfg vocab/eps, qk_norm guards, gate).

## Vulkan: DONE (device side, builds clean)

- `src/gpu/kernels/eltwise.zig`: **`gemv_iq4_nl` + `gemv_iq4_nl_t`** SPIR-V
  (one-thread-per-row; **no workgroup memory** — Zig-SPIR-V constraint;
  `kvalues_iq4nl` is a module-const `[16]i8`; `_t` uses the `tf16`/`tbyte`
  32-row-group readers, `row_bytes=(cols/32)*18`).
- `src/gpu/context.zig`: `Elt` enum `gemv_iq4_nl`/`_t` (appended at END — the
  dispatch **table is positional**, matching entries appended at END too);
  `opGemvQuant` + `opGemvQuantT` switch arms + `row_bytes=(cols/32)*18` +
  relaxed the `cols%256` assert for iq4_nl.

## Vulkan: VulkanLM config-driven rewrite (DONE — original plan below, all landed)

`src/models/qwen3_gpu.zig` `VulkanLM` is **hardcoded to qwen3-4B / bf16 / tied
head** and is **GEMM-oriented with NO block-quant path**. Unlike CUDA's `CudaLM`
(already cfg-driven), this needs a real rewrite. The clean reference is
**`src/models/gemma3_gpu.zig`** — already cfg-driven + block-quant on Vulkan.

Concrete changes:
1. **Config-drive**: add `cfg: qwen3.Config` field; replace module consts
   (`hidden`, `n_heads`, `kv_heads`, `intermediate`, `n_layers`, `eps`,
   `qwen3.vocab_size`) with `self.cfg.*` throughout `VulkanLM` (`head_dim`/`hd`
   stays 128). **Leave `encode()` on the consts** — it's the Krea2 diffusion
   text-encoder, a separate fn; don't touch it.
2. **Fixed arrays**: `k_cache`/`v_cache` `[n_layers]Buf` → `[qwen3.Config.max_layers]Buf`,
   loops over `cfg.n_layers`; size all buffers from `cfg`.
3. **Block-quant matmul**: add a `gemvW` helper like gemma3_gpu.zig:205 —
   `.q8_0/.q4_k/.q5_k/.q6_k/.iq4_nl => opGemvQuantT(...)`, else `opGemvQuant`/
   existing bf16 path. Route all 7 weight matmuls (q,k,v,o,gate,up,down) — which
   currently call `opGemvPartial`/`self.gemm` — plus the untied head through it.
   Vulkan block-quant uses per-row GEMV for BOTH decode and prefill (no GEMM).
4. **qk_norm guard** in `normQK` (`if (!cfg.qk_norm) return;`).
5. **F16 embed**: host-gather via `safetensors.convertToF32(.f16, ...)` (like the
   CPU/CUDA prefill embed) — the LM head is a separate IQ4_NL tensor (untied),
   handled by `gemvW`; the embed table is F16.
6. **Relax `init()` gate**: currently rejects `cfg.n_layers!=n_layers` /
   `hidden!=hidden` / non-bf16 embed / untied head.
7. **`src/llm_main.zig` `runQwen3`**: the vulkan block-quant guard
   ("GGUF block-quant … run on cpu/zig-cuda/cuda only") must allow iq4_nl (or the
   llama arch) through.

## Validate

```
# Nemo on Vulkan (expect coherent; token-identical to CPU on the constrained one):
zig build run-llm -Doptimize=ReleaseFast -- --model <nemo.gguf> --backend vulkan \
  --greedy --max-tokens 40 --prompt "Continue the sequence with no extra words: Monday, Tuesday, Wednesday,"
#   → "Thursday, Friday, Saturday, Sunday."

# qwen3-VL Vulkan regression (must stay working — bf16 path):
zig build run-llm -Doptimize=ReleaseFast -- \
  --model /home/qt/genai/comfyui/models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors \
  --backend vulkan --greedy --max-tokens 40 --prompt "Name three primary colors and one fact about the sky."
```

Reference perf: `llama.cpp` on this model+3090 ≈ 81 tok/s decode (`llama-bench
-ngl 99`); TP CUDA ≈ 35. Vulkan is correctness-first (memory: TP Vulkan trails).

## Gotchas
- **rms_eps = 1e-5** for Nemo (qwen3 is 1e-6) — read from `<arch>.attention.layer_norm_rms_epsilon`.
- **q/k un-permute** already done at load (`qwen3.loadQK`, `cfg.permute_qk`) → Vulkan reuses rotate-half rope as-is. Don't re-permute.
- CUDA LUT gotcha (context): divergent `ld.const` **serializes** on CUDA → shared-mem LUT was the 2.8× fix. Vulkan uses a module-const array (no equivalent penalty observed); if Vulkan decode is slow, revisit.
- `llama-tokenize`'s `cat -n` miscounts `\n`-valued tokens as extra display lines.
- CLI auto-routes arch `llama` → `runQwen3` (default fall-through) → `qwen3.CausalLM`.
