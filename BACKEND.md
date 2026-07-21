# BACKEND.md — Backend / Feature / Format Support Grid

TensorPencil has **four compute backends**. This document is the support matrix:
what each backend can do, from top-level capabilities (diffusion, LLM) down to
individual kernels (relu, rope, gemv), and which **data formats** each operation
runs in.

> Generated from a code survey; cite the listed files as the source of truth.
> When you add a backend path or kernel, update the relevant table here.

## The four backends

| Backend | CLI (`--backend`) | What it is | Selected in |
|---|---|---|---|
| **cpu** | `cpu` | Pure-Zig reference + ggml for CPU block-quant GEMV/dequant. Correctness baseline; slow. | default fallback |
| **vulkan** | `vulkan` | Zig→SPIR-V compute kernels + cooperative-matrix (tensor-core) GEMM, `dlopen libvulkan`. | `pipeline.zig`, `llm/session.zig` |
| **zig-cuda** | `zig-cuda` | Hand-emitted PTX, JIT'd through the CUDA driver API. No vendor math libs. | `gpu/cuda.zig` (`.hand_ptx`) |
| **cuda** | `cuda` | Same driver Context as zig-cuda, but prefill GEMM → **cuBLASLt** and prefill attention → **cuDNN SDPA** (`dlopen`'d). Fastest. | `gpu/cuda.zig` (`.libs`) |

**Key structural fact:** `zig-cuda` and `cuda` **share the same code path** (`*_cuda.zig`
model steppers, one `cuda.Backend`). They differ *only* in the prefill/batched
GEMM and prefill attention: `.libs` routes those to cuBLASLt/cuDNN, `.hand_ptx`
uses hand-written kernels. **Decode (m=1) GEMV, flash-decode attention, RoPE,
RMSNorm, GDN, and embedding are hand-PTX in both** — cuDNN is never used at decode.

Legend: ✅ full · ⚠️ works but slow / limited · ❌ unsupported · — not applicable

---

## 1. Top-level capabilities

| Capability | cpu | vulkan | zig-cuda | cuda | Notes |
|---|---|---|---|---|---|
| **Diffusion txt2img** | ⚠️ ref | ✅ | ✅ | ✅ **primary** | cuda ≈ 1.42× ComfyUI gap; vulkan targets ~1.2× of cuda |
| **LLM text generation** | ⚠️ ref | ✅¹ | ✅ | ✅ **primary** | dispatched per-arch in `llm_main.zig` |
| **LLM vision (ViT/mmproj)** | ✅ | ⚠️ gemma3 only | ✅ | ✅ | see §5 |
| **GPU init failure** | — | → CPU fallback | → CPU fallback | → CPU fallback | logs + degrades, never hard-fails |

¹ vulkan LLM excludes **gemma4** entirely and **rejects GGUF block-quant weights for qwen3** (see §4–§5).

---

## 2. Diffusion pipeline (`src/pipeline.zig`)

Per-stage dispatch order everywhere is `if (cu_be)` → CUDA, `else if (gpu_ctx)` → Vulkan, `else` → CPU.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-4B) | ✅ f32 | ✅ f32 (⚠️ f16 opt via `--encoder-f16`) | ✅ fp8→f16 TC | ✅ fp8→f16 TC | `krea2_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (Krea 2, 28 blocks) | ✅ all dtypes | ✅ fp8 / int8 / bf16 | ✅ int8 / int4 / bf16 | ✅ int8 / int4 / bf16 | `dit{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1) | ✅ | ✅ | ✅ | ✅ (+cuDNN conv avail.) | `wan_vae.zig`, `vae_{gpu,cuda}.zig` |
| **VAE tiling** | CPU-tile | GPU-tile + CPU floor | GPU-tile + CPU floor | GPU-tile + CPU floor | `vae_tiled.zig` |
| **TAEHV preview** (taew2_1) | ✅ | ✅ *(new)* | ✅ | ✅ | `taehv{,_gpu,_cuda}.zig` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `wan_vae.latentPreviewInto` (fallback when no `--taew`) |

**Cancellation** (`Options.cancel`): polled between sampling steps everywhere, plus mid-stage on every backend — between DiT blocks, between text-encoder layers, and between VAE decode layers (and per tile in `vae_tiled`). On the **cpu** backend the threaded matmul/attention kernels additionally poll a threadlocal token (`src/ops/cancel.zig`, armed by `dit.forward` / `wan_vae.Decoder.decode` / `qwen3.TextEncoder.encode`) per row-panel / k-block / query row, so a cancel lands in milliseconds even when a single CPU GEMM takes seconds. `error.Canceled` is never swallowed: the VAE OOM-retry ladder and the GPU→CPU encode fallback both propagate it.

### DiT block weight-dtype support

| DiT block dtype | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **fp8-e4m3** *(default ckpt)* | ✅ | ✅ (fast coop) | ❌ blocks¹ | ❌ blocks¹ |
| **int8-convrot** | ✅ | ✅ | ✅ | ✅ (+f16 MLP) |
| **int4-convrot** | ✅ | ❌² | ✅ | ✅ |
| **bf16 dense** | ✅ | ✅ native/f16 | ✅ native/f16 | ✅ cuBLASLt R_16BF |
| **f32** | ✅ | ✅ (offload) | — | — |

¹ The default checkpoint is fp8; **CUDA backends require an int8/int4/bf16 DiT checkpoint** (fp8 blocks → `error.UnsupportedCheckpoint`, `dit_cuda.zig:436`). First/last patch-embed projections still accept fp8.
² No sint4 cooperative-matrix on Vulkan (see §7).

---

## 3. Diffusion ops & kernels

| Op | cpu | vulkan | zig-cuda | cuda | Formats |
|---|---|---|---|---|---|
| **GEMM / linear** | `ops/matmul.zig` (dequant→f32 SIMD) | `opMatmul`/`opMatmulCoop*` | hand hgemm/igemm/i4gemm | cuBLASLt | see §7 |
| **Conv2d** (im2col + GEMM, fused 2× upsample) | ✅ f32 | ✅ f16-TC (co≥96) / f32 | ✅ | ✅ (cuDNN conv avail.) | f32 weights; f16 TC wide-co |
| **Attention** (DiT GQA 48/12, hd128) | f32 | f16 two-pass TC (+flash) | `opAttnTC` (hgemm+softmax) | f16 TC | f16 scores |
| **VAE mid-block attn** (1 head, hd384) | f32 | ⚠️ CPU round-trip | TC | TC | — |
| RMSNorm + AdaLN modulate | ✅ | `rms_apply_mod`/`modulate` | `rms_mod_par` | ↤ shared | f32 (+h16) |
| QK-norm | ✅ | `qknorm_rope16` | `qk_rmsnorm` | ↤ | f32/f16 |
| RoPE (3-axis interleaved) | ✅ | `rope_inter` | `rope` | ↤ | f32 |
| SwiGLU / silu-mul | ✅ | `silu_mul{,16,_h16}` | `silu_mul{,_h16}` | ↤ | f32/f16 |
| sigmoid-gated add | ✅ | `sigmoid_mul` | `mul_sigmoid` | ↤ | f32 |
| gated residual add | ✅ | `gated_add{,16}` | `gated_add` | ↤ | f32/f16 |
| **relu** *(new)* | ✅ | `relu` | `relu` | ↤ | f32 |
| **add_relu** (TAEHV residual, new) | ✅ (fused loop) | `add_relu` | `add`+`relu` | ↤ | f32 |
| im2col | ✅ | `im2col` | `im2col` | ↤ | f32 |
| nearest-2× upsample | explicit | fused into im2col | fused into im2col | ↤ | — |
| convrot (Hadamard un-rotate) | `ops/convrot.zig` | `rotate`/`rotate_fwht` | `buildPrep` | ↤ | int8/int4, group 256 |

`↤ shared` = cuda reuses zig-cuda's hand-PTX kernel for this op (only prefill GEMM/attention differ).

---

## 4. LLM models (text) — `tp-llm` (`src/llm_main.zig`, `llm/session.zig`)

| Model | cpu | vulkan | zig-cuda | cuda | Files | Notes |
|---|---|---|---|---|---|---|
| **qwen3** (Qwen3 / Qwen3-VL) | ✅ | ✅¹ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig` | only arch with spec decode; primary path fp8 safetensors; GGUF metadata config-detect up to 64 layers (Qwen3-32B); hybrid CPU split like qwen35 (32B: offload measured ~1.7x over the removed streaming path at equal budget) |
| **qwen35** (hybrid DeltaNet: GDN+attn) | ✅ | ✅ | ✅ | ✅ | `qwen35{,_gpu,_cuda}.zig` | GGUF block-quant on vulkan too; no spec (recurrent state) |
| **gemma3** (sandwich norms, dual RoPE) | ✅ | ✅ | ✅ | ✅ | `gemma3{,_gpu,_cuda}.zig` | entirely GGUF block-quant |
| **gemma4** (per-layer geom, factored RoPE) | ✅ | ❌² | ✅ | ✅ | `gemma4{,_cuda}.zig` | `--backend vulkan` rejected; hybrid CPU split like gemma3 (GUI-driven; per-layer-KV host shadow keeps the device ring layout). Config is fully metadata-driven: 12B (Q4_0, 48L) **and 31B** (mixed Q4_K/Q6_K + tied Q6_K head, 60L, hidden 5376, kv 16↔4) both load with no code change; vision via `gemma4uv`/`gemma4v` towers (see §5) |

¹ **qwen3 on vulkan rejects GGUF block-quant** (`llm_main.zig`): dense fp8/bf16/f32 only. bf16 weights are read **natively** (2-byte, `transpose_bf16`/`pipe_tr_bf16` + a bf16 branch in `gemv_partial`/`gemv_partial4`, weight code `context.WCode.bf16`), like CUDA's `gemv_bf16` — no widening, weights stay 8GB. bf16 has no tiled GEMM so prefill streams through grouped GEMV. Generation is coherent (4B ~32 tok/s) but not token-identical to CPU (GEMV reduction-order drift).
² No `gemma4_gpu.zig`; `Spec.Vulkan = void`.

---

## 5. LLM vision towers

| Tower | cpu | vulkan | zig-cuda | cuda | Files | Vision dtypes |
|---|---|---|---|---|---|---|
| **Qwen3-VL `vit35`** (SigLIP, 2-D RoPE) | ✅ | ❌ | ✅ | ✅ | `vit35{,_cuda}.zig` | bf16 blocks/proj, f32 patch |
| **Gemma 3 `gemma_vit`** (SigLIP-So400m) | ✅ | ✅ | ✅ | ✅ | `gemma_vit{,_gpu,_cuda}.zig` | f16 blocks (vulkan: →f32 at load) |
| **Gemma 4 `gemma4_vit`** (shallow `gemma4uv` embedder, 12B) | ✅ | ❌ | ✅ | ✅ | `gemma4_vit{,_cuda}.zig` | f32 patch, bf16 proj |
| **Gemma 4 `gemma4v_vit`** (full SigLIP tower, 31B) | ✅ | ❌ | ✅ | ✅ | `gemma4v_vit{,_cuda}.zig` | Q8_0/F16 blocks, f32 patch |

- Only **gemma3** has a Vulkan ViT. Interactive `@image` chat mentions are **CUDA-only**; one-shot `--image` falls back to CPU (all towers) or Vulkan (gemma3 only).
- ³ The **`gemma4v`** tower (31B `DarkIdol-Gemma-4-31B` mmproj: `projector_type "gemma4v"` — full 27-block SigLIP with per-head QK-RMSNorm, 2-D neox RoPE θ=100, weightless V-norm, `kq_scale=1.0`, GeGLU-quick FFN, RMS sandwich norms, 3×3 avg-pool merge, `std_bias`/`std_scale` affine, single `mm.input_projection`). CPU forward `gemma4v_vit.zig`; CUDA/zig-cuda device tower `gemma4v_vit_cuda.zig` runs the 27 blocks device-side (~1.0 s at 512² vs ~3.6 s CPU) and projects on host — reuses `opAttnTC`/`opHeadPad`/`qkNorm` + three gemma4v-specific ops (`opRopeVisionGemma4` 2-D neox rope, `geluQuickMul`, and qkNorm-as-RMS over `dim`). Vulkan not built (gemma4 has no Vulkan LLM). Preprocess follows **Google's `gemma4_vision_token_budget`**: aspect-preserving resize (NO crop, NO letterbox/pad) to a 48-aligned grid sized so post-merge tokens target a settable budget `nMax` — `f = sqrt(nMax·48²/(w·h))`, each dim floored to /48; `2·p/255 − 1` normalize. Budget is runtime-settable: CLI `--vision-budget low|medium|high|ultra|max|<tokens>` and tp-gui "Vision detail" dropdown (`config.VisionBudget`), presets 70/140/280/560/1120, default `high` (280). The budget also sizes the **LLM's** image-prefill scratch + LOCAL KV-ring slack (`gemma4.Config.image_budget` → `maxBatch()`/`bufRows()`/`localRingRows`), since a bidirectional image block prefills in ONE pass — so it's fixed at load (reload-gated in the GUI via `llmReloadEql`), and `high`/default stays lean (no regression). `ultra`/`max` (~540/~1080 tokens) roughly triple that scratch + ring; on a 24 GB card + 31B they need headroom (`--kv-dtype q8_0` and/or a smaller `--max-context`) or they OOM cleanly (no corruption). `TP_VIT_DUMP=<path.png>` writes the exact pixels the tower ingests. Dispatch by `clip.vision.projector_type` picks `gemma4v_vit` vs the shallow `gemma4uv` `gemma4_vit` (both CLI `runGemma4` and tp-gui). GPU parity vs the f32 CPU tower is looser than gemma3's (min-token cos ~0.96): the `kq_scale=1.0` peaked softmax makes the f16 tensor-core attention more divergent — semantically preserved (image-accurate captions match the CPU tower).

---

## 6. LLM ops, decode features & advanced paths

| Op / feature | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| Attention prefill | `ops/attention.zig` | `attn_full` + coopmat | hgemm+softmax TC | **cuDNN SDPA** |
| Attention decode (flash) | (same fn) | `attn_dsplit`/`attn_dmerge`, `attn_decode_q35` | `attn_split`/`_merge`/`_h256`/`_h512` | ↤ hand-PTX |
| GQA / windowed (Gemma) local attn | ✅ | ✅ | ✅ | ✅ |
| Bidirectional image-block attn (Gemma vision) | ✅ | ✅ (gemma3) | ✅ | ✅ |
| KV cache | host slices | device + gather | `opKvAppendS` + gather/scatter | ↤ |
| **Growable VMM KV context** (cuMemMap in-place) | ⚠️ host arrays | ❌ (reserves window up front) | ✅ | ✅ |
| RoPE (half/partial/interleaved/dual/factored) | ✅ | ✅ (no vision/M-RoPE) | ✅ (+M-RoPE, vision) | ↤ |
| RMSNorm / LayerNorm / sandwich | ✅ | `rmsnorm`/`layernorm` | `qk_rmsnorm`/`ln_bias_par` | ↤ |
| **Decode GEMV (dp4a int8)** | ggml `vec_dot` (no dp4a) | ⚠️ **f32 scalar, no dp4a** | ✅ `gemv_q*_q8n` grouped-N dp4a | ↤ |
| Prefill GEMM | `matmul.zig` microkernel | coopmat bf16/f16 + int8 s8→s32 | hand hgemm/igemm/i4gemm | **cuBLASLt** |
| **GDN / gated DeltaNet** (qwen35) | ✅ | `gdn_gates`/`gdn_conv_step`/`gdn_delta_step` | `gdn_*` | ↤ |
| Embedding gather | model | ⚠️ host-side | on-device `opEmbedGather*` | ↤ |
| **Sampling** (argmax/temp/top-k/top-p/min-p + repeat/presence/frequency penalties) | ✅ `llm/sample.zig` | ✅ on-device argmax/top-k select (qwen3) | ✅ on-device argmax/top-k select (qwen3/qwen35/gemma3/gemma4) | ↤ |
| **Turn-boundary checkpoint / rollback** (`checkpoint`/`restoreCheckpoint` — tp-gui regenerate) | ❌ | ❌ | ✅ qwen3/qwen35/gemma3/gemma4 | ↤ |

GPU sampling is a candidate select, not a full sampler: the device runs argmax (greedy) or a top-k reduce (`stepArgmax`/`stepSelect`) and downloads only the candidates; the CPU `llm/sample.zig` tail (temperature softmax, top-p, min-p, RNG) runs over them, bit-identical to full-vocab CPU sampling. The recent-window penalties (repeat/presence/frequency) also run on-device: the `penalize` kernel (PTX + SPIR-V, `opPenalize`) scatters the host-collected (unique id, subtract) entries onto the resident logits BEFORE the argmax/top-k (`stepArgmaxPen`/`stepSelectPen` on qwen3 all-GPU-backends, qwen35_cuda, gemma3_cuda, gemma4_cuda; bit-identical to CPU `applyPenalties` on CUDA via `div.rn`, ~2.5-ULP division tolerance on Vulkan; validated token-identical e2e vs the CPU-sampled spec-verify path). gemma4's logit finalization needs no device tanh: its softcap is strictly MONOTONIC, so the device selects over the RAW logits (after an on-device suppress mask — the penalize scatter with an infinite presence penalty), and only the downloaded candidates get the exact host softcap + penalties (`gemma4.finalizeCandidates`; validated token-identical to the old download path in all four sampling modes). Only the vulkan qwen35/gemma3 steppers (no stepSelect yet) still take the full logit download + CPU path.

Turn-boundary checkpoints back tp-gui's O(snapshot) "regenerate response" / variant-switch rollback: `checkpoint(out)` captures the non-append-only context state at a turn boundary and `restoreCheckpoint(snap, q)` truncates back to it — append-only attention KV is never copied. Per arch the snapshot is: **qwen3** nothing at all (uniform full attention is entirely append-only, so the snapshot is zero bytes and restore is a pure `truncate(q)`); **qwen35** the DeltaNet conv/ssm recurrent state + M-RoPE position (tens of MB); **gemma3/gemma4** the LOCAL layers' sliding-window KV rings (the response overwrites their oldest rows, so `len` rollback alone can't rewind past the ring slack; a few hundred MB, f16-aware on gemma4). Snapshot/restore are residency-aware (qwen35/gemma3/gemma4 read/write each layer's CURRENT owner — device buffer or CPU-split host shadow — so layers may migrate between snapshot and restore). Validated token-identical A/B on the real checkpoints (`-Dintegration '-Dtest-filter=checkpoint restore'`). Vulkan/CPU steppers don't expose the API (the GUI chat runs CUDA-only; its session falls back to a full transcript re-prefill for any arch without it).

### Speculative decoding (qwen3 only)

| Feature (flag) | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| n-gram prompt-lookup (`--spec-k`) | ✅ | ✅ | ✅ | ✅ |
| draft-model (`--draft-model`) | ✅ | ❌ | ✅ | ✅ |
| EAGLE-3 head (`--eagle`) | ❌ | ❌ | ✅ | ✅ |
| tree drafting (`--tree`) | ❌ (verify only) | ❌ | ✅ | ✅ |

### Weight residency / offload

**LLM weights NEVER stream.** Every LLM weight pins device-resident on first
touch (`Backend.pinAllWeights` / Vulkan `pin_budget = maxInt`, set by
`llm/session.zig` bring-up and the GUI session loader) and is immune to LRU
eviction. A model that outgrows VRAM degrades by migrating **whole layers to
the CPU** (the hybrid split below) — measured ~2.5x faster than the removed
weight-streaming fallback, whose LRU-vs-cyclic-walk pathology re-uploaded ~the
whole model per token the moment the budget fell short (the 31B 0.1 tok/s
cliff). `--vram-budget` now only sizes the split planners; on Vulkan (no split)
a model that doesn't fit fails with a clean error. Per-step weight streaming
remains a **diffusion-only** mechanism (`pin_floor` + prefetch staging ring).

| Feature | cpu | vulkan | zig-cuda | cuda | Models |
|---|---|---|---|---|---|
| pin-all weight residency | — | ✅ | ✅ | ✅ | all |
| `--cpu-layers` static split | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |
| `--offload-grow` dynamic offload | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |

The hybrid CPU/GPU split works with **every KV dtype** (`--kv-dtype f32|f16|q8_0`): the offloaded layers' host shadow (`llm/kv_cache.zig KvCache`, or `PerLayerKvCache` for gemma4's per-layer geometry) stores the same storage format as the device caches — packed-f16 slots or raw ggml `block_q8_0` bytes, byte-identical to the device layout — so migrate/promote (and the ring checkpoint copies — gemma3 translates ring↔linear, gemma4's shadow keeps the device ring layout so rings move wholesale) are raw, lossless copies (`kRowBytes`/`vRowBytes`). f16/q8_0 KV therefore keep their reduced footprint on both sides of the split AND the offload safety net at once. The GUI dtype toggle (`reinitCache`) also survives an armed split: the host shadow is rebuilt at the new dtype and host-resident layers keep no device KV. Applies to qwen3/qwen35/gemma3/gemma4 (the gemma models' split is GUI-driven — `autoOffload`/`settleTo`/`imageReclaim` — not exposed as CLI flags); qwen3's EAGLE-tap/tree paths remain f32-only.

**Automatic offload-on-OOM (gemma4, CLI + GUI):** `gemma4_cuda` never dead-ends on a device OOM while a layer can still move to the host. On `DeviceOutOfMemory` during a prefill forward (`forwardRows` retry wrapper) or KV growth (`ensureCapacity`), it arms a dynamic split **on demand** (`ensureOffloadArmed` — all layers resident, `n_cpu=0`, so `migrateNext` can then offload incrementally; it deliberately skips `enableCpuSplit`'s budget planner, which mid-life would offload almost everything at once), migrates a few layers to the CPU, and retries. Safe at the forward boundary: a failed forward aborts its batch and does NOT advance `self.len`, so migrating (which copies only committed KV `[0,len)`) and re-running is idempotent. This makes **tp-llm** (which never pre-arms a split, unlike the GUI's Arbiter) able to run/prefill a model or a large `--vision-budget` that doesn't fully fit — it degrades to a hybrid split (slower) instead of crashing. `session.zig` sets `model.io` before the CUDA prefill (not just decode) so the host layer path works. Only engages under real pressure — `high`/fitting workloads stay fully resident (`self.split == null`, zero overhead). Pattern is gemma4-only for now; drops into the other `*_cuda.zig` steppers via the shared `runtime/residency.zig` as a follow-up.

**q8_0 KV** (`--kv-dtype q8_0`, GUI dropdown): the ggml `block_q8_0` format — 34 bytes per 32 elements (f16 scale `d = absmax/127` + 32 × i8), ~3.8× smaller than f32. Rows are quantized once on write/append (CPU `packQ80`, CUDA `f32_to_q8_0`/`kv_append_s_q8`, Vulkan `kv_store_q8_0`) and dequantized inside the attention kernels (CUDA `attn_split_q8`/`attn_split_h256_q8`/`attn_split_h512_q8` + graph `_s_q8` twins, Vulkan `attn_dsplit_gemma_q8`, CPU dequant-on-view). Quantization rounds ties-to-EVEN on every engine (host 2^23 trick / CUDA `cvt.rni` / Vulkan floor+compare — deliberately diverging from ggml's `roundf` only on exact .5 ties) so host- and device-quantized bytes are **bit-identical**: a row quantized on either side of an offload split, checkpoint, or migrate/promote round trip produces the same cache bytes (see the `opStoreKvQ8 ... bit-identically` device test). Every model kv_dim is a multiple of 64, so rows never split blocks and the byte math stays 4-aligned. Backend coverage matches f16: all four backends; on Vulkan gemma3/qwen35 only (qwen3's hd128 Vulkan path stays f32); qwen3 spec-decode tree/EAGLE stay f32-only. Like f16, q8_0 is lossy — output is not token-identical to f32 — and a dtype toggle rebuilds the context.

---

## 7. Data-format support matrix

DType enum: `src/dtype.zig:11` — `f8_e4m3, f16, bf16, f32, i8, i4, q4_0, q8_0, q4_k, q5_k, q6_k`.
(`i8`/`i4` are the ComfyUI "convrot" formats for the **image/DiT** path; GGUF `q*` are the **LLM** path.)

| Format | cpu | vulkan | zig-cuda | cuda | How it computes |
|---|---|---|---|---|---|
| **f32** | ✅ | ✅ scalar / VAE→f16 coop | ✅ fallback | ⤷ via f16/bf16 | dtype-aware GEMM; f32 SIMD accumulate |
| **f16** | ✅ vectorized Zig | ✅ f16 coopmat | ✅ `buildHgemm` m16n8k16, `gemv_f16` | ✅ cuBLASLt `R_16F` | tensor-core coop / mma |
| **bf16** | ✅ | ✅ **native bf16 coopmat** + f16 fallback | ✅ bf16 mma (Ampere+) + f16 | ✅ cuBLASLt `R_16BF` | native bf16 on all GPUs |
| **fp8-e4m3** | ✅ (LUT) | ✅ `pipe_f8` → f16 coop | ✅ `gemv_fp8`, `dequant_fp8_f16` | ✅ dequant→f16, `R_16F` | 1-byte weights, dequant in kernel |
| **int8 (+convrot)** | ✅ | ✅ **s8→s32 tensor cores** | ✅ **m16n8k32 s8** IMMA | ✅ cuBLASLt `R_8I`/`COMPUTE_32I` | Hadamard un-rotate at dequant |
| **int4 (+convrot)** | ✅ | ❌ **no sint4 coopmat** | ✅ **m16n8k64 s4** IMMA (W4A4) | ❌ (no cuBLASLt s4) | nibble-packed 2/byte |
| **GGUF q4_0** | ✅ ggml | ❌ (no GEMV kernel) | ✅ `gemv_q4_0(_q8n)` | ⤷ dequant→f16 | — |
| **GGUF q8_0** | ✅ ggml | ✅ `gemv_q8_0`/`_t` (scalar) | ✅ `gemv_q8_0(_q8n)` | ⤷ dequant→f16 | — |
| **GGUF q4_k** | ✅ ggml | ✅ `gemv_q4_k`/`_t` (scalar) | ✅ `gemv_q4_k(_q8n)` | ⤷ dequant→f16 | — |
| **GGUF q5_k** | ✅ ggml | ✅ `gemv_q5_k`/`_t` (scalar) | ✅ `gemv_q5_k(_q8/_q8n)` | ⤷ dequant→f16 | — |
| **GGUF q6_k** | ✅ ggml | ✅ `gemv_q6_k`/`_t` (scalar) | ✅ `gemv_q6_k(_q8/_q8n)` | ⤷ dequant→f16 | — |

Notes:
- **int4 / W4A4 is CUDA-hand-PTX-only** (`s4 m16n8k64` tensor cores). CPU has a correctness path; Vulkan and cuBLASLt cannot do sint4.
- **GGUF block-quant on GPU dequants on-the-fly inside the GEMV** — never expanded to VRAM. Vulkan's GEMV is **scalar f32 (no dp4a)** and lacks `q4_0`. The `cuda` (libs) arm dequants GGUF to f16 for prefill GEMM but uses the shared **hand-PTX** GEMV at decode (cuBLASLt/cuDNN never consume GGUF block-quant directly).
- **convrot** (`src/ops/convrot.zig`): size-256 Hadamard rotation, applied at int8/int4 dequant. `cols` must be a multiple of 256 (i4 also even from nibble-packing).

---

## 8. Kernel inventory (appendix)

### Vulkan — `Elt` compute kernels (`src/gpu/context.zig:70`, bodies in `src/gpu/kernels/eltwise.zig`)
`rmsnorm` · `rms_partial` · `rms_combine` · `rms_apply_mod{,_h16}` · `rms_apply_w` · `modulate` ·
`gated_add{,16}` · `add` · `relu` · `add_relu` · `silu_mul{,_h16,16}` · `sigmoid_mul{,_h16,_g16}` ·
`gelu` · `gelu_mul` · `layernorm` · `vae_norm` · `l2norm_rows` · `qknorm_rope16` · `qknorm_rope_f32` ·
`rope_inter` · `rope_half` · `rope_qwen35` · `attention` · `attn_scores` · `softmax_partial` ·
`softmax_combine` · `softmax_rows` · `attn_out` · `attn_dsplit` · `attn_dmerge` · `attn_full` ·
`attn_decode_q35` · `gather_kmajor{,_h16,16}` · `f32_to_h16{,_pad}` · `f32_to_bf16_pad` · `copy` ·
`deinterleave2` · `scale_concat` · `scale_i32` · `bias_compact` · `im2col` · `rotate` · `rotate_fwht` ·
`rowmax_i8` · `rowscale_i8` · `quantize_i8` · `gemv_partial{,4}` · `gemv_combine{,4}` ·
`gemv_q8_0{,_t}` · `gemv_q4_k{,_t}` · `gemv_q5_k{,_t}` · `gemv_q6_k{,_t}` · `gdn_gates` ·
`gdn_conv_step` · `gdn_delta_step`

**Vulkan GEMM entry points** (`context.zig`): `opMatmul` (f32/fp8) · `opGemv{,Partial,Quant,QuantT}` ·
`opMatmulCoop{,H16}` (fp8→f16) · `opMatmulCoopF16W{,b}` (f32/bf16→f16) · `opMatmulCoopBf16` (native bf16) ·
`opMatmulCoopI8{,Fused}` / `opMatmulI8` / `opI8Gemm` (int8 s8→s32) · `opAttnScores{,Vae}` / `opFlash` / `opAttnOut`.
Coopmat SPIR-V builders in `src/gpu/coopmat.zig` (`buildGemmShared` f16/bf16, `buildGemmI8` int8; **no s4**).

### zig-cuda — hand-PTX (`src/gpu/cuda/kernels.zig` GEMM, `elt.zig` elementwise/attn)
GEMM builders: `buildHgemm` (f16/bf16 mma m16n8k16) · `buildIgemmSmem`/`buildIgemmPipe` (int8 m16n8k32, int4 m16n8k64) · `buildPrep` (quant/rotate). **All DiT GEMM formats** use warp-cooperative `ldmatrix.x4`/`.x2` frag loads + an XOR-swizzled (`off^=(row&7)<<4`) conflict-free shared layout (`use_ldmatrix` flag on `buildIgemmPipe` — int8/int4 — AND `buildHgemm` — f16/bf16 dense + attention; on at every runtime site; pure permutation → bit-exact, ~+1–3% on qkv/attn-proj shapes, flat on BW-bound MLP shapes).
GEMV: `gemv_{fp8,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k}` + `_q8`/grouped-N `_q8n` dp4a variants.
Attn: `attn`, `attn_split`/`_merge`/`_h256`/`_h512`/`_tree`. GDN: `gdn_{conv_step,gates,delta_step}`.
Plus `im2col`, dtype-pad converts, `dequant_*_f16`, rope/norm/act kernels.
Vision: `rope_vision` (Qwen3-VL 2-D rope) · `rope_vision_gemma4` (gemma4v per-head-half x/y neox rope) · `gelu_quick_mul` (gemma4v GeGLU-quick FFN).

### cuda — vendor libs (`.libs` mode)
**cuBLASLt** (`src/gpu/cuda/cublaslt.zig`): int8 `R_8I`/`COMPUTE_32I`, f16 `R_16F`, bf16 `R_16BF` (prefill GEMM only).
**cuDNN** (`src/gpu/cuda/cudnn.zig`): fused SDPA-forward attention (prefill), legacy conv-forward (VAE).
Everything else (decode GEMV, convrot prep, elementwise, RoPE, GDN) stays hand-PTX.

### cpu — `src/ops/*.zig` + ggml (fetched dep, optional `-Dggml`, default on)
`matmul.zig` (dtype-aware microkernel; block-quant decode GEMV → ggml `vec_dot`) · `attention.zig`
(+`attentionTree`) · `norm.zig` · `act.zig` (`silu`/`geluTanh`/`sigmoid` + `*Mul`) · `rope.zig` ·
`convrot.zig` · `vmath.zig` (SIMD `expVec`). ggml owns CPU block-quant dequant + decode `vec_dot`;
everything else (GGUF parse, f16/bf16/fp8 conversion, GEMM threading, convrot, tokenizer, sampling) is in-house Zig.
With `-Dggml=false`, ggml is not fetched/linked and block-quant (q4_0/q8_0/q4_k/q5_k/q6_k) is unavailable
(`matmul` → `error.QuantBackendUnavailable`); all other dtypes and backends are unaffected.
