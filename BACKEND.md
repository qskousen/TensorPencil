# BACKEND.md — backend / feature / format support grid

What each backend can do, from top-level capabilities down to individual kernels, and which
data formats each operation runs in. **This file is a state-of-the-world grid, not a
history** — when you add a backend path, kernel or dtype, update the relevant row; when a
gap closes, delete it from §9. Keep entries to a line or two and cite the listed files as
the source of truth.

## The four backends

| Backend | `--backend` | What it is | Selected in |
|---|---|---|---|
| **cpu** | `cpu` | Pure-Zig reference + ggml for CPU block-quant GEMV/dequant. Correctness baseline; slow. | default fallback |
| **vulkan** | `vulkan` | Zig→SPIR-V compute kernels + cooperative-matrix (tensor-core) GEMM, `dlopen libvulkan`. | `pipeline.zig`, `llm/session.zig` |
| **zig-cuda** | `zig-cuda` | Hand-emitted PTX, JIT'd through the CUDA driver API. No vendor math libs. | `gpu/cuda.zig` (`.hand_ptx`) |
| **cuda** | `cuda` | Same driver Context as zig-cuda, but batched/prefill GEMM → **cuBLASLt** and prefill attention → **cuDNN SDPA**. Fastest. | `gpu/cuda.zig` (`.libs`) |

**`zig-cuda` and `cuda` share one code path** (`*_cuda.zig` steppers, one `cuda.Backend`).
They differ *only* in batched GEMM and prefill attention. Decode (m=1) GEMV, flash-decode
attention, RoPE, RMSNorm, GDN, convrot prep and embedding are hand-PTX in both.

Legend: ✅ full · ⚠️ works but slow / limited · ❌ unsupported · — not applicable

---

## 1. Top-level capabilities

| Capability | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **Diffusion txt2img** (all five families) | ⚠️ ref | ✅ | ✅ | ✅ **primary** |
| **LLM text generation** | ⚠️ ref | ✅¹ | ✅ | ✅ **primary** |
| **LLM vision (ViT/mmproj)** | ✅ | ⚠️ gemma3 only | ✅ | ✅ |
| **GPU init failure** | — | → CPU fallback | → CPU fallback | → CPU fallback |

¹ vulkan LLM excludes **gemma4** entirely; a block-quant token *embedding* is also rejected
(no Vulkan block-quant gather kernel). See §4–§5.

---

## 2. Diffusion (`src/pipeline.zig`)

Per-stage dispatch order everywhere is `if (cu_be)` → CUDA, `else if (gpu_ctx)` → Vulkan,
`else` → CPU. Five families; every stage of every family runs on every backend.

**Cancellation** (`Options.cancel`) is polled between sampling steps, and mid-stage on every
backend: between DiT blocks, between text-encoder layers, between VAE decode layers, and per
tile in `vae_tiled`. On **cpu** the threaded matmul/attention kernels additionally poll a
threadlocal token (`src/ops/cancel.zig`) per row-panel / k-block / query row, so a cancel
lands in milliseconds even when one GEMM takes seconds. `error.Canceled` is never swallowed:
the VAE OOM ladder and the GPU→CPU encode fallback both propagate it.

### 2A. krea2

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-4B) | ✅ f32 | ✅ f32 (f16 via `--encoder-f16`) | ✅ fp8→f16 TC | ✅ fp8→f16 TC | `krea2_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (28 blocks) | ✅ all dtypes | ✅ fp8/int8/w4a8/nvfp4/bf16 | ✅ + int4 | ✅ + int4 | `dit{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1) | ✅ | ✅ | ✅ | ✅ (+cuDNN conv) | `wan_vae.zig`, `vae_{gpu,cuda}.zig` |
| **VAE tiling** | CPU-tile | GPU-tile + CPU floor | ↤ | ↤ | `vae_tiled.zig` |
| **TAEHV preview** | ✅ | ✅ | ✅ | ✅ | `taehv{,_gpu,_cuda}.zig` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `wan_vae.latentPreviewInto` |

### 2B. The SD family (SD1.5 / SDXL)

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **CLIP-L / CLIP-G** | ✅ ref | ✅ | ✅ | ✅ | `clip_text{,_gpu,_cuda}.zig` |
| **UNet** (LDM `UNetModel`) | ✅ ref | ✅ | ✅ | ✅ | `sd_unet{,_gpu,_cuda}.zig` |
| **VAE decode** (AutoencoderKL) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` |

The CLIP batch axis on the device is the **chunk**, since one 77-row window gives only
`77·heads` threads. The empty-prompt reference `z_empty` rides along as one more batch item
and is cached per tower.

⚠️ **Precision is not symmetric between the GPU arms.** Vulkan follows `qwen3_gpu`'s
convention (f32 by default, tensor-core f16 only under `--encoder-f16`) and holds 1.1e-5 rel
L2 against the CPU forward; CUDA has only a tensor-core entry point at these widths, so it
runs f16 regardless and lands ~1e-3. A `cuda` and a `cpu` render of one seed therefore differ
slightly at the conditioning.

⚠️ **The two towers of one SDXL checkpoint use different activations** (CLIP-L quick-GELU,
CLIP-G erf-GELU), and the three GELU forms here agree to ~1e-2 — close enough to look right,
far enough to shift style. `cfg.act` carries which; parity tests compare value by value.

**Attention head width is the one shape the three GPU arms do not share.** SD1.5 attends at
head_dim 40/80/160 (8 fixed heads), SDXL at 64 (fixed width, growing head count):

| backend | how it attends | cost |
|---|---|---|
| **cuda** | cuDNN fused SDPA at the true width | none |
| **zig-cuda** | zero-padded to a multiple of 128 | `128/hd` on the attention |
| **vulkan** | zero-padded to 128; hd 160 falls back to the scalar kernel | as above |

The padding is exact (a zero dimension contributes nothing to a dot product, and V's zero
columns give output columns that are dropped), so it costs only arithmetic. Both GPU limits
are the same one: the P@V GEMM tiles the head dimension in 128-wide blocks
(`coopmat.buildGemmAttnOut`, `launchHgemmB`'s `grid.x = n/128`). Removing the padding means
parameterizing those two builders by head width, not changing the model code.

⚠️ **The SDXL VAE overflows f16 and every GPU arm divides the residual stream before the
cast.** Its decoder residual reaches 4.2e5 against f16's 65504 ceiling; unscaled, the cast
gives `inf`, the next GroupNorm spreads NaN through its mean, and the render is solid white
with no error. `residual_act_div` (256, a power of two, so exact) is applied only to the
residual-*reading* convolutions — the 1×1 shortcut and the level upsamples — via
`opConvF16Scaled` / `opMatmulCoopF16WScaled` / `convIntoScaled`. ⚠️ Those kernels' spare `f32`
push constant is load-bearing: **every caller must pass 1.0, not 0.0.** SD1.5's VAE peaks near
7e3 and never shows it.

### 2C. Z-Image ("zit", `NextDiT`)

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-4B, penultimate) | ✅ f32 | ✅ | ✅ | ✅ | `zimage_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (30 blocks + 2+2 refiners) | ✅ bf16/f16/f32/fp8/GGUF | ✅ + nvfp4 | ✅ | ✅ | `zimage{,_gpu,_cuda}.zig` |
| **VAE decode** (`AutoencoderKL`, 16-ch) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` (`sd_vae.flux`) |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `zimage.latentPreviewInto` (Flux factors + bias) |

**How `zimage_cuda` differs from `zimage_gpu`**, both forced by the op surface:

- ⚠️ **CUDA takes the FOLDED modulation table** (`zimage.modulationTableFolded`): it has no
  standalone `modulate`, only `rmsMod`, so the pre-norm weight must arrive multiplied into the
  scale. Vulkan takes the unfused table. Both derive from one AdaLN evaluation so they cannot
  drift.
- Attention is `opAttnTC` (cuDNN SDPA under `cuda`, hand-PTX otherwise); `be.attn` is the
  fallback behind `force_naive_attn`.

Structural facts for this trunk: head width is exactly 128, so unlike the SD family there is
**no head padding**, and every trunk GEMM width (3840 / 10240 / 11520) is a multiple of 128.
The fused `qkv` splits into three zero-copy **row views** — ⚠️ the device weight cache keys on
the host pointer and part 0 shares the fused tensor's pointer, so never upload both. The
timestep MLP, all AdaLN linears (precomputed for the whole schedule at `Session.init`), the
whole caption half, patchify and the final layer stay on the host. ⚠️ **Two sequence lengths
per forward**: the `noise_refiner` blocks run on the image half alone at the positions it will
hold in the joint sequence, so the rope table is *sliced*, not rebuilt.

### 2D. Anima (Cosmos-Predict2 `MiniTrainDIT` + `llm_adapter`)

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **T5 tokenizer** (SentencePiece Unigram) | ✅ | — | — | — | `core/t5_tokenizer.zig` |
| **Text encoder** (Qwen3-0.6B, final state + `model.norm`) | ✅ | ✅ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig`, `Variant.anima` |
| **`llm_adapter`** (6 blocks, 1024, T5-indexed) | ✅ | host | host | host | `anima.zig` (`Adapter`) |
| **DiT** (28 blocks, 2048) | ✅ bf16/f16/f32/fp8/int8/int4/w4a8/nvfp4 | ✅ int8/int4/w4a8/nvfp4/bf16 | ✅ | ✅ | `anima{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1, 16-ch) | ✅ | ✅ | ✅ | ✅ | `wan_vae.zig`, `vae_{gpu,cuda}.zig` |
| **TAEHV / latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | krea2's Wan matrix, **not** Z-Image's Flux one |

⚠️ **The `llm_adapter` stays on the host on purpose**: it does not depend on the timestep, so
it is computed once per *image* inside `encode`, and a device port would buy ~1/30th of a render.

**Cross-attention K and V are per-image constants**, precomputed for every block at
`Session.init` (28 blocks × 2 GEMMs of `[512, 2048] × [2048, 1024]` leave the step loop).
Vulkan caches them in the f16 attention-operand layout (117 MB), CUDA in f32 (235 MB, its
entry points convert internally) — doubled under CFG, since a session is bound to one
conditioning. ⚠️ **One buffer per block on Vulkan**, not one buffer with a per-block offset: a
Vulkan `DeviceBuffer.buf` is an opaque handle, so `zimage_cuda`'s `offsetBuf` trick is
CUDA-only and gives `error_device_lost` here.

⚠️ **Under `--backend cuda`, self-attention takes cuDNN's fused SDPA but cross-attention takes
the hand-PTX rectangular path**, because `SdpaPlan` is built for a single sequence length.
Teaching it a rectangular shape is a real but separate change (~6% of the step).

⚠️ **Rectangular tensor-core attention is a requirement, not an optimization.**
Cross-attention is `seq × 512` — 2.4% of a step's FLOPs — but the naive
thread-per-(query, head) kernels stream the whole 512-key context per thread, costing an
estimated ~0.6 s/step at 1056x1584. The FLOP share is not what decides; the data reuse is.

⚠️ **Quantization kind is resolved PER BLOCK** (`anima.linKind` / `prepGroup`). Real mixed
checkpoints leave block 0 entirely dense, quantize block 1's ten attention/MLP linears, and
quantize all sixteen in blocks 2-27, so a one-tensor probe says "GPU ok" and then panics.
`anima.deviceLins` is the single list every support scan reads. `dit_cuda`'s per-*model*
`LinKind` would be wrong for at least one block of such a file.

⚠️ **Vulkan's int4 is W4A8-shaped, not W4A4** — no `sint4` coopmat, so the weight decodes to
int8 per GEMM (`i4_decode_t`) and the activation stays int8. That is more accurate than CUDA's
true W4A4, so the same file scores differently per backend. Both 4-bit formats decode **per
GEMM**, not at load, so a 4-bit checkpoint costs 4-bit memory on every backend.

Anima's reduction widths (2048, 8192) are in `gpu.i8_prep_cols`; ⚠️ a missing width silently
falls back to a 3-pass prep that round-trips a full f32 activation copy through global memory.

**What the trunk port added** — two kernels and one op generalization:

| new | where | why |
|---|---|---|
| `ln_mod_sg` | `kernels/subgroup.zig` | fused weightless LayerNorm + AdaLN modulation, one subgroup per row. Two-pass deviation variance, matching `ops.norm.layerNormUnit` (not the shifted form, which cancels catastrophically at large row means) |
| `ln_mod_par` | `cuda/elt.zig` | the CUDA twin, derived from `ln_bias_par` by asserted substitution (`replaceOnce`) so the reduction and variance stay literally shared |
| `pmdplane` (push word 7) | `coopmat.buildFlashAttn` | lets the flash kernel run **rectangular** q × kv. `s_stride` was K's row stride, the j loop bound *and* the MD plane stride at once, and the MD table is indexed by QUERY row. **0 means "= s_stride"**, so pre-existing callers are unchanged by construction |
| `opAttnTCRect` | `cuda/backend.zig` | `opAttnTCBatched` with the two sequence lengths pulled apart |

### 2E. Diffusion speed snapshot

RTX 3090, ReleaseFast. Read a PSNR against its model's own precision floor, not in
isolation — these models disagree with themselves across dtypes by 23-25 dB.

| family / config | cpu | vulkan | zig-cuda | cuda | reference |
|---|---|---|---|---|---|
| SD1.5 512², 8 steps, CFG 7.5 | 6.91 s/step | 0.92 | 0.46 | **0.27** | 25× end to end on `cuda` |
| SDXL 1024², 8 steps, CFG 7.5 | 50.06 | 2.20 | 1.83 | **1.11** | 41× end to end |
| krea2 1120x1680, cfg 1, int8 | — | — | — | **2.25** | w4a8 2.26, int4 1.95, nvfp4 4.36 |
| Z-Image 1056x1584, 9 steps, cfg 1 | — | 3.80 | — | **2.00** | ComfyUI 1.836 (92%) |
| Anima 512x768, 30 steps, cfg 5 | 43.60 | 0.59 | 0.42 | **0.37** | ComfyUI bf16 0.33 (89%) |
| Anima 1056x1584, same | — | 3.11 | 2.32 | **1.62** | ComfyUI bf16 1.31 (81%) |

The CPU path is the reference; no GPU arm is bit-identical to it. All three GPU arms agree
with each other inside their models' own precision envelopes.

### 2F. DiT block weight-dtype support

| DiT block dtype | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **fp8-e4m3** | ✅ | ✅ (fast coop) | ✅ stream+dequant¹ | ✅ stream+dequant¹ |
| **int8-convrot** | ✅ | ✅ | ✅ | ✅ (+f16 MLP) |
| **int4-convrot** | ✅ | ✅ (decodes to int8) | ✅ W4A4 | ✅ W4A4 |
| **w4a8** | ✅ | ✅ | ✅ | ✅ |
| **nvfp4** | ✅ | ✅ | ✅ | ✅ |
| **bf16 dense** | ✅ | ✅ native/f16 | ✅ native/f16 | ✅ cuBLASLt `R_16BF` |
| **f32** | ✅ | ✅ (offload) | — | — |
| **GGUF block quants** | ✅ | ❌ | ❌ | ❌ |

¹ fp8 block linears stream through `opMatmulFp8`: the weight decodes to an f16 scratch
(`dequant_fp8_f16`, per-tensor scale folded) and runs through `buildHgemm` (hand-PTX) or
`ltMatmulF16` (cuBLASLt). The scratch is re-materialized per GEMM (no fp8 tensor-core GEMM),
so fp8 on the CUDA backends is correctness-first and slower per step than int8.
⚠️ The **CUDA fused `opMatmul`** (bias + destination offset, used only by `first`/`last.linear`)
has no fp8 variant, so those two projections — like bf16 — are materialized to f32 once at load
(`DiT.opMatmulF32`); otherwise the run aborts on the fp8 assert or reads packed bytes as f32.

⚠️ **A GGUF DiT is CPU-only and that is enforced** (`dit.gpuLinKindSupported`,
`anima`/`zimage` equivalents). Both GPU forwards return `error.UnsupportedCheckpoint`; before
the gate, Vulkan treated any unrecognized dtype as raw fp8 and rendered a blank white image
with no error.

---

## 3. Diffusion ops and kernels

| Op | cpu | vulkan | zig-cuda | cuda | Formats |
|---|---|---|---|---|---|
| **GEMM / linear** | `ops/matmul.zig` | `opMatmul`/`opMatmulCoop*` | hand hgemm/igemm/i4gemm | cuBLASLt | see §7 |
| **Conv2d** (im2col + GEMM, fused 2× upsample) | ✅ f32 | ✅ f16-TC (co≥96) / f32 | ✅ | ✅ (+cuDNN conv) | f32 weights; f16 TC wide-co |
| **Attention** (DiT GQA 48/12, hd128) | f32 | f16 two-pass TC (+flash) | `opAttnTC` | f16 TC | f16 scores |
| **VAE mid-block attn** | f32 | f32 scores plane, query-banded | TC flash, f32 scores band | cuDNN SDPA | see below |
| RMSNorm + AdaLN modulate | ✅ | `rms_apply_mod`/`modulate` | `rms_mod_par` | ↤ | f32 (+h16) |
| Weighted RMSNorm (Q/K, sandwich) | ✅ | `rmsnorm_sg` (subgroup) | `qk_rmsnorm_warp` | ↤ | f32/f16 |
| Weightless LayerNorm + modulate | ✅ | `ln_mod_sg` | `ln_mod_par` | ↤ | f32 |
| RoPE (3-axis interleaved) | ✅ | `rope_inter` | `rope` | ↤ | f32 |
| SwiGLU / silu-mul | ✅ | `silu_mul{,16,_h16}` | `silu_mul{,_h16}` | ↤ | f32/f16 |
| sigmoid-gated add | ✅ | `sigmoid_mul` | `mul_sigmoid` | ↤ | f32 |
| gated residual add | ✅ | `gated_add{,16}` | `gated_add` | ↤ | f32/f16 |
| relu / add_relu | ✅ | `relu` / `add_relu` | `relu` / `add`+`relu` | ↤ | f32 |
| GroupNorm (Welford) | ✅ | `gn_stats`/`gn_combine`/`gn_apply` | ↤ same names | ↤ | f32/h16 |
| im2col | ✅ | `im2col`, `im2col_sd` | ↤ | ↤ | f32/h16 |
| nearest-2× upsample | explicit | fused into im2col | fused into im2col | ↤ | — |
| convrot (Hadamard un-rotate) | `ops/convrot.zig` | `rotate`/`rotate_fwht` | `buildPrep` | ↤ | int8/int4, group 256 |
| 4-bit decode | `ops/{w4a8,nvfp4}.zig` | `w4a8_decode_t`/`nvfp4_decode_t`/`i4_decode_t` | `w4a8_decode`/`nvfp4_decode` PTX | ↤ | per GEMM into scratch |

`↤ shared` = `cuda` reuses zig-cuda's hand-PTX kernel (only batched GEMM and prefill
attention differ).

**Norms are bandwidth-bound and must be subgroup/warp-per-row, not thread-per-row.** A
thread-per-row weighted RMSNorm puts a warp's 32 loads `4*dim` apart, one sector each: measured
29-45 GB/s against 460-550 for the subgroup form on a 936 GB/s card. All eight live diffusion
sites use the parallel form. ⚠️ **Not bit-identical** — the row sum becomes a tree where it was
serial (the more accurate of the two), so gated forward-parity tolerances move with it.

**VAE mid-block attention has a range problem, not a precision one.** The Flux/Z-Image VAE's
logits reach 9.95e6 where SD1.5's reach 8.3 — 152× past f16's ceiling, and f16's quantum up
there is ~8000 against O(1) softmax differences. So the scores must stay f32: Vulkan
materializes an f32 plane in query bands (`sd_vae.scoresBand`), `cuda` uses cuDNN's fused SDPA
(softmax internal, f32), `zig-cuda` uses `opAttnTCFlash` with an f32 scores band.
⚠️ `scoresBand` is shared by the decoder and `estimatePeakBytes` — two copies of that constant
is what makes a peak estimate lie.

⚠️ **`attn_out` runs its OWN online softmax.** There is no softmax pass between it and
`attn_scores`; inserting one exponentiates twice — finite, plausible, wrong by rel L2 0.26.
That absence reads as a missing step until you read the kernel.

⚠️ **`ctx.independent(n)` counts DISPATCHES, not calls**, and a coop GEMM is three of them
sharing scratch. Marking three q/k/v GEMMs independent both lets them clobber each other's
scratch and drops the barriers inside the first one.

**VAE activation storage** (`sd_vae.Config.act_f16`) is f16 for the `flux` and `sd15` configs
and f32 for `sdxl`. ⚠️ **Range gates it, not precision**: an f16 buffer cannot hold a value
that `residual_act_div` exists to sneak through a cast, so the two are alternatives, and
`convIntoPrec` asserts they are never combined. ⚠️ f16 storage also **forces the cooperative
conv path** — `coop_min_co` (96) is a performance threshold, but the f32 arm has no f16
storage form, and a narrow conv falling to it writes f32 into an f16 buffer.

---

## 4. LLM models (text) — `tp-llm` (`src/llm_main.zig`, `llm/session.zig`)

| Model | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **qwen3** (Qwen3 / Qwen3-VL / llama-arch Mistral-Nemo) | ✅ | ✅¹ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig` |
| **qwen35** (hybrid DeltaNet: GDN + attn) | ✅ | ✅ | ✅ | ✅ | `qwen35{,_gpu,_cuda}.zig` |
| **gemma3** (sandwich norms, dual RoPE) | ✅ | ✅ | ✅ | ✅ | `gemma3{,_gpu,_cuda}.zig` |
| **gemma4** (per-layer geometry, factored RoPE) | ✅ | ❌² | ✅ | ✅ | `gemma4{,_cuda}.zig` |

- **qwen3** is the only arch with speculative decoding. It shares its stepper with the
  generalized `llama`/Mistral arch (un-permuted q/k at load, optional QK-norm, untied head,
  runtime vocab/eps). GGUF metadata config-detect up to 64 layers.
- **qwen35** has no spec decode (recurrent state).
- **gemma3** is entirely GGUF block-quant.
- **gemma4** config is fully metadata-driven: 12B (Q4_0, 48L) and 31B (mixed Q4_K/Q6_K, tied
  Q6_K head, 60L, hidden 5376, kv 16↔4) load with no code change. Vision via `gemma4uv` /
  `gemma4v` towers (§5).

¹ **qwen3 on vulkan** (`qwen3_gpu.zig VulkanLM`, config-driven) runs two regimes. **Dense**
(fp8/bf16/f32, tied head): batched square-attention prefill + spec decode, with bf16 read
natively (2-byte `transpose_bf16`/`pipe_tr_bf16` plus a bf16 branch in `gemv_partial{,4}`,
weight code `context.WCode.bf16`) so weights are never widened; bf16 has no tiled GEMM, so
prefill streams through grouped GEMV. Generation is coherent but not token-identical to CPU
(GEMV reduction-order drift). **GGUF block-quant** (q8_0/q4_k/q5_k/q6_k/iq4_nl + untied
block-quant head): decode is the per-row fused-dequant GEMV (`gemvW` → `opGemvQuantT`);
prefill runs the tensor-core GEMM via `opMatmulCoopQuant`, which dequants each weight to f16
k-major on the GPU and reuses `coopF16WDispatch`, so the whole prompt prefills in one batched
pass. The f16 form is 4× the block-quant size and is re-dequanted per prefill into one reused
scratch. Falls back to per-token GEMV when the device lacks the f16 coopmat pipeline. QK-norm
is skipped (`cfg.qk_norm = false`); the F16 embedding is host-gathered to f32.
**Opt-in int8 dp4a decode** (`TP_VK_DP4A=1`, q8_0/iq4_nl, needs
`VK_KHR_shader_integer_dot_product`) repacks the weight into an int8-interleaved layout so a
warp reads 4 contiguous quants per `u32` and dots via `OpSDot` — ~2.2× decode, but the
repacked weight roughly doubles VRAM, hence opt-in until VRAM-aware auto-sizing lands. The
`OpSDot` capability is injected by `spv.zig withDotProduct` into a separate `dp4a` module.

² No `gemma4_gpu.zig`; `Spec.Vulkan = void`.

### LLM speed against llama.cpp

Same prompt, same token count, same GPU, matched settings. ⚠️ **The 3090 drifts several
percent between sessions and llama.cpp drifts with it, so only an interleaved same-session
ratio is meaningful** — never compare tok/s across sessions.

| model / format | decode | prefill |
|---|---|---|
| Bonsai-27B q1_0, `zig-cuda` | **1.30×** (75.8 vs 58.3 tok/s) | **0.86×** (1097 vs 1276) |
| Bonsai-27B q2_0 g128, `zig-cuda` | 0.92× | ~0.88× |
| Mistral-Nemo IQ4_NL, `vulkan` | ~3.5× slower than llama.cpp's own Vulkan | — |

⚠️ **Prefill tok/s is flat in prompt length only when the weight upload is outside the timer.**
`Backend.cachedWeight` uploads lazily, so a one-shot run charges host→device copy to the first
forward, i.e. to `pp`, where llama.cpp reports it as load time. `CudaLM.warmWeights` (opt-in via
`@hasDecl`, called from `session.run`) uploads up front. ⚠️ **Only qwen35 has it** — qwen3,
gemma3 and gemma4 still charge the upload to their prefill and their prefill numbers are
understated. A rising prefill rate with prompt length is the signature of this.

⚠️ **`engine.prefill_gate_chunk` and a stepper's `prefill_chunk` must change together or not at
all.** The engine slices prefill into `prefill_gate_chunk` pieces so the pause/cancel gate can
land mid-prefill, so a larger stepper chunk is dead code that only enlarges buffers. A stepper
that knows what its kernels want declares `prefill_batch` (`engine.prefillBatchOf`); qwen35_cuda
ties `prefill_batch = prefill_chunk`. **Pause and cancel are polled at different granularities
on purpose**: pause parks only at batch boundaries (parking mid-kernel would strand
half-computed state), while cancel unwinds between layers via `engine.publishCancel` →
`ops.cancel`'s threadlocal token.

---

## 5. LLM vision towers

| Tower | cpu | vulkan | zig-cuda | cuda | Files | Vision dtypes |
|---|---|---|---|---|---|---|
| **Qwen3-VL `vit35`** (SigLIP, 2-D RoPE) | ✅ | ❌ | ✅ | ✅ | `vit35{,_cuda}.zig` | bf16 blocks/proj, f32 patch |
| **Gemma 3 `gemma_vit`** (SigLIP-So400m) | ✅ | ✅ | ✅ | ✅ | `gemma_vit{,_gpu,_cuda}.zig` | f16 blocks (vulkan: →f32 at load) |
| **Gemma 4 `gemma4_vit`** (shallow `gemma4uv` embedder, 12B) | ✅ | ❌ | ✅ | ✅ | `gemma4_vit{,_cuda}.zig` | f32 patch, bf16 proj |
| **Gemma 4 `gemma4v_vit`** (full SigLIP tower, 31B) | ✅ | ❌ | ✅ | ✅ | `gemma4v_vit{,_cuda}.zig` | Q8_0/F16 blocks, f32 patch |

- Only **gemma3** has a Vulkan ViT. Interactive `@image` chat mentions are **CUDA-only**;
  one-shot `--image` falls back to CPU (all towers) or Vulkan (gemma3 only).
- **`gemma4v`** (`projector_type "gemma4v"`) is a 27-block SigLIP with per-head QK-RMSNorm, 2-D
  neox RoPE θ=100, weightless V-norm, `kq_scale = 1.0`, GeGLU-quick FFN, RMS sandwich norms,
  3×3 avg-pool merge, `std_bias`/`std_scale` affine and a single `mm.input_projection`. The
  device tower runs the 27 blocks device-side and projects on host, reusing
  `opAttnTC`/`opHeadPad`/`qkNorm` plus three gemma4v-specific ops (`opRopeVisionGemma4`,
  `geluQuickMul`, qkNorm-as-RMS over `dim`). ⚠️ GPU parity against the f32 CPU tower is looser
  than gemma3's (min-token cos ~0.96): the `kq_scale = 1.0` peaked softmax makes f16
  tensor-core attention more divergent. Semantically preserved.
- **Preprocess follows Google's `gemma4_vision_token_budget`**: aspect-preserving resize (no
  crop, no letterbox) to a 48-aligned grid sized so post-merge tokens hit a budget —
  `f = sqrt(nMax·48²/(w·h))`, each dim floored to /48, then `2·p/255 − 1`. Runtime-settable via
  `--vision-budget low|medium|high|ultra|max|<tokens>` and the tp-gui "Vision detail" dropdown
  (`config.VisionBudget`; presets 70/140/280/560/1120, default `high`). ⚠️ The budget also sizes
  the **LLM's** image-prefill scratch and local KV-ring slack (`gemma4.Config.image_budget` →
  `maxBatch()`/`bufRows()`/`localRingRows`), because a bidirectional image block prefills in one
  pass — so it is fixed at load and reload-gated in the GUI (`llmReloadEql`). `ultra`/`max`
  roughly triple that scratch; on a 24 GB card with the 31B they need `--kv-dtype q8_0` and/or a
  smaller `--max-context` or they OOM cleanly.
- `TP_VIT_DUMP=<path.png>` writes the exact pixels the tower ingests.

---

## 6. LLM ops, decode features and residency

| Op / feature | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| Attention prefill | `ops/attention.zig` | `attn_full` + coopmat | hgemm+softmax TC, `attn_split_g` | **cuDNN SDPA** |
| Attention decode (flash) | (same fn) | `attn_dsplit`/`attn_dmerge`, `attn_decode_q35` | `attn_split`/`_merge`/`_h256`/`_h512` | ↤ hand-PTX |
| GQA / windowed (Gemma) local attn | ✅ | ✅ | ✅ | ✅ |
| Bidirectional image-block attn | ✅ | ✅ (gemma3) | ✅ | ✅ |
| KV cache | host slices | device + gather | `opKvAppendS` + gather/scatter | ↤ |
| **Growable VMM KV context** (cuMemMap in place) | ⚠️ host arrays | ❌ (reserves window up front) | ✅ | ✅ |
| RoPE (half/partial/interleaved/dual/factored) | ✅ | ✅ (no vision/M-RoPE) | ✅ (+M-RoPE, vision) | ↤ |
| RMSNorm / LayerNorm / sandwich | ✅ | `rmsnorm`/`layernorm` | `qk_rmsnorm`/`ln_bias_par` | ↤ |
| **Decode GEMV (dp4a int8)** | ggml `vec_dot` | ⚠️ opt-in only (`TP_VK_DP4A`) | ✅ `gemv_q*_q8n` grouped-N | ↤ |
| Prefill GEMM | `matmul.zig` microkernel | coopmat bf16/f16 + int8 s8→s32 | hand hgemm/igemm/i4gemm, MMQ pipes | **cuBLASLt** |
| **GDN / gated DeltaNet** (qwen35) | ✅ | `gdn_gates`/`gdn_conv_step`/`gdn_delta_step` | + batched `gdn_conv_batch`, `opGdnDeltaChunk` | ↤ |
| Embedding gather | model | ⚠️ host-side | on-device `opEmbedGather*` | ↤ |
| **Sampling** (argmax/temp/top-k/top-p/min-p + penalties) | ✅ `llm/sample.zig` | ✅ argmax/top-k select (qwen3) | ✅ (qwen3/qwen35/gemma3/gemma4) | ↤ |
| **Turn-boundary checkpoint / rollback** | ❌ | ❌ | ✅ qwen3/qwen35/gemma3/gemma4 | ↤ |

**GPU sampling is a candidate select, not a full sampler.** The device runs argmax or a top-k
reduce (`stepArgmax`/`stepSelect`) and downloads only the candidates; the CPU tail (temperature
softmax, top-p, min-p, RNG) runs over them, bit-identical to full-vocab CPU sampling. Recent-window
penalties also run on-device (`opPenalize` scatters host-collected (unique id, subtract) entries
onto resident logits before the select) — bit-identical to CPU on CUDA via `div.rn`, ~2.5 ULP
tolerance on Vulkan. gemma4 needs no device tanh: its softcap is strictly **monotonic**, so the
device selects over raw logits and only the downloaded candidates get the exact host softcap
(`gemma4.finalizeCandidates`). Only the vulkan qwen35/gemma3 steppers still take the full logit
download.

**Turn-boundary checkpoints** back tp-gui's O(snapshot) regenerate / variant-switch rollback.
`checkpoint(out)` captures the non-append-only context state at a turn boundary;
`restoreCheckpoint(snap, q)` truncates back to it, and append-only attention KV is never copied.
Per arch: **qwen3** nothing at all (uniform full attention is entirely append-only, so restore is
a pure `truncate(q)`); **qwen35** the DeltaNet conv/ssm state + M-RoPE position (tens of MB);
**gemma3/gemma4** the local layers' sliding-window KV rings (a response overwrites their oldest
rows, so `len` rollback alone cannot rewind past the ring slack). Snapshot and restore are
residency-aware — layers may migrate between the two. Steppers without the API fall back to a
full transcript re-prefill.

### Speculative decoding (qwen3 only)

| Feature | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| n-gram prompt-lookup (`--spec-k`) | ✅ | ✅ | ✅ | ✅ |
| draft-model (`--draft-model`) | ✅ | ❌ | ✅ | ✅ |
| EAGLE-3 head (`--eagle`) | ❌ | ❌ | ✅ | ✅ |
| tree drafting (`--tree`) | ❌ (verify only) | ❌ | ✅ | ✅ |

### Weight residency and offload

⚠️ **LLM weights NEVER stream.** Every LLM weight pins device-resident on first touch
(`Backend.pinAllWeights` / Vulkan `pin_budget = maxInt`) and is immune to LRU eviction. A model
that outgrows VRAM degrades by migrating **whole layers to the CPU**, measured ~2.5× faster than
the removed weight-streaming fallback, whose LRU-vs-cyclic-walk pathology re-uploaded roughly the
whole model per token once the budget fell short. `--vram-budget` only sizes the split planners;
on Vulkan (no split) a model that does not fit fails with a clean error. **Per-step weight
streaming is a diffusion-only mechanism** (`pin_floor` + prefetch staging ring).

| Feature | cpu | vulkan | zig-cuda | cuda | Models |
|---|---|---|---|---|---|
| pin-all weight residency | — | ✅ | ✅ | ✅ | all |
| `--cpu-layers` static split | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |
| `--offload-grow` dynamic offload | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |

The hybrid split works with **every KV dtype** (`--kv-dtype f32|f16|q8_0`): the offloaded layers'
host shadow (`llm/kv_cache.zig`, or `PerLayerKvCache` for gemma4's per-layer geometry) stores the
same format as the device caches, byte-identical to the device layout, so migrate/promote and the
ring checkpoint copies are raw lossless copies. gemma3 translates ring↔linear; gemma4's shadow
keeps the device ring layout so rings move wholesale. The GUI dtype toggle survives an armed split.
The gemma models' split is GUI-driven (`autoOffload`/`settleTo`/`imageReclaim`), not CLI flags;
qwen3's EAGLE-tap/tree paths remain f32-only.

**Automatic offload-on-OOM (gemma4, CLI + GUI).** `gemma4_cuda` never dead-ends on a device OOM
while a layer can still move to the host: on `DeviceOutOfMemory` during a prefill forward or KV
growth it arms a dynamic split on demand (`ensureOffloadArmed` — all layers resident, `n_cpu = 0`,
so `migrateNext` offloads incrementally; it deliberately skips `enableCpuSplit`'s budget planner,
which mid-life would offload almost everything at once), migrates a few layers and retries. Safe at
the forward boundary: a failed forward aborts its batch and does not advance `self.len`, so
migrating and re-running is idempotent. Only engages under real pressure. gemma4-only for now;
drops into the other `*_cuda.zig` steppers via `runtime/residency.zig`.

**q8_0 KV** (`--kv-dtype q8_0`): the ggml `block_q8_0` format, 34 bytes per 32 elements
(f16 `d = absmax/127` + 32 × i8), ~3.8× smaller than f32. Rows are quantized once on write
(`packQ80` / `f32_to_q8_0` / `kv_store_q8_0`) and dequantized inside the attention kernels.
⚠️ Quantization rounds **ties-to-even** on every engine (host 2^23 trick, CUDA `cvt.rni`, Vulkan
floor+compare — deliberately diverging from ggml's `roundf` only on exact .5 ties) so host- and
device-quantized bytes are bit-identical across an offload split, checkpoint or migrate round trip.
Every model kv_dim is a multiple of 64, so rows never split blocks. Coverage matches f16: on Vulkan
gemma3/qwen35 only; qwen3 spec-decode tree/EAGLE stay f32-only. Like f16, q8_0 is lossy — output is
not token-identical to f32 — and a dtype toggle rebuilds the context.

---

## 7. Data-format support matrix

DType enum (`src/dtype.zig`): `f8_e4m3, f16, bf16, f32, i8, i4, q4_0, q8_0, q4_k, q5_k, q6_k,
iq4_nl, q1_0, q2_0_g64, q2_0_g128`. `i8`/`i4` are the ComfyUI convrot formats for the **image**
path; GGUF `q*` are the **LLM** path.

| Format | cpu | vulkan | zig-cuda | cuda | How it computes |
|---|---|---|---|---|---|
| **f32** | ✅ | ✅ scalar / VAE→f16 coop | ✅ fallback | ⤷ via f16/bf16 | dtype-aware GEMM, f32 SIMD accumulate |
| **f16** | ✅ vectorized Zig | ✅ f16 coopmat | ✅ `buildHgemm` m16n8k16, `gemv_f16` | ✅ cuBLASLt `R_16F` | tensor-core coop / mma |
| **bf16** | ✅ | ✅ native bf16 coopmat + f16 fallback | ✅ bf16 mma (Ampere+) | ✅ cuBLASLt `R_16BF` | native bf16 on all GPUs |
| **fp8-e4m3** | ✅ (LUT) | ✅ `pipe_f8` → f16 coop | ✅ `gemv_fp8`, `dequant_fp8_f16` | ✅ dequant→f16 | 1-byte weights, dequant in kernel |
| **int8 (+convrot)** | ✅ | ✅ s8→s32 tensor cores | ✅ m16n8k32 s8 IMMA | ✅ cuBLASLt `R_8I`/`COMPUTE_32I` | Hadamard un-rotate at dequant |
| **int4 (+convrot)** | ✅ | ✅ decode→int8 per GEMM | ✅ m16n8k64 s4 IMMA (W4A4) | ✅ hand-PTX (no cuBLASLt s4) | nibble-packed 2/byte |
| **w4a8** (`asym_w4a8_int8`) | ✅ | ✅ | ✅ | ✅ | 4-bit codebook indices + fp8 per-group scale; stays packed, decodes to int8 **per GEMM** then runs the ordinary int8 convrot GEMM. `ops/w4a8.zig` |
| **nvfp4** (E2M1) | ✅ | ✅ | ✅ | ✅ | 4-bit E2M1 + fp8 per-16-block scale + per-tensor scale; stays packed, decodes to **bf16** per GEMM then the existing bf16 tensor-core GEMM. Weight-only; the native W4A4-fp4 GEMM is sm_100+. `ops/nvfp4.zig` |
| **GGUF q4_0** | ✅ ggml | ❌ | ✅ `gemv_q4_0(_q8n)` | ⤷ dequant→f16 | |
| **GGUF q8_0** | ✅ ggml | ✅ `gemv_q8_0{,_t}` (scalar) | ✅ `gemv_q8_0(_q8n)` | ⤷ dequant→f16 | |
| **GGUF q4_k / q5_k / q6_k** | ✅ ggml | ✅ scalar `gemv_*{,_t}` | ✅ `gemv_*(_q8/_q8n)` + MMQ | ⤷ dequant→f16 | |
| **GGUF iq4_nl** | ✅ ggml | ✅ scalar (module-const LUT) | ✅ shared-mem LUT | ⤷ dequant→f16 | 32 elems / 18 B, non-linear `kvalues_iq4nl` |
| **GGUF q1_0** | ✅ ggml | ❌ | ✅ `gemv_q1_0{,_q8}`, `mmq_pipe_q1_0` | ⤷ dequant→f16 | 128 elems / 18 B; **1 sign bit per weight**, `v = bit ? d : -d`, `d = mean\|x\|` |
| **GGUF q2_0 g128** | ✅ **native** `dotQ2_0G128` (not ggml) | ❌ | ✅ `gemv_q2_0_g128{,_q8}`, `mmq_pipe_q2_0` | ⤷ dequant→f16 | 128 elems / 34 B; 2 bits/weight, `v = (code − 1)·d`, codes → {−1, 0, +1, +2} |
| **GGUF q2_0 g64** | ✅ ggml | ❌ | ✅ built, ⚠️ **never executed** (no g64 file here) | ⤷ dequant→f16 | 64 elems / 18 B, ggml's own `QK2_0` |

**Notes:**

- **GGUF block-quant on GPU dequants on the fly inside the GEMV** — never expanded to VRAM.
  Vulkan's GEMV is scalar f32 unless `TP_VK_DP4A` is set, and it lacks `q4_0`, `q1_0` and both
  `q2_0`s. The `cuda` arm dequants GGUF to f16 for prefill GEMM but uses the shared hand-PTX GEMV
  at decode; cuBLASLt/cuDNN never consume block quants directly.
- **convrot** (`src/ops/convrot.zig`): size-256 Hadamard rotation applied at int8/int4 dequant.
  `cols` must be a multiple of 256 (i4 also even, from nibble packing).
- ⚠️ **The packed 4-bit formats share ONE container loader** (`models/quant_weight.zig`), called
  from krea2's, Anima's and Z-Image's `mat`. All store `[rows, cols/2]`, which is *exactly* the
  int4-convrot signature, so each is detected by its own sidecar (`_scale_2` for NVFP4, `_s_rel`
  for W4A8) and **before** the int4 heuristic. Every format ComfyUI's quantizers emit reaches every
  family they support, so a reader for one belongs in the shared module from the start.
- **`w4a8` shares int8's activation prep and GEMM** (`opI8Prep`/`opI8Gemm`) — the "A8" is exactly
  that the activation stays 8-bit — so only the weight's *storage* differs and dispatch is per
  weight (`i8GemmW`; `anima.prepKind` on the Anima arm, which is what lets one prep serve a block
  mixing int8 and W4A8). A mixed int8/W4A8 checkpoint therefore works.
- ⚠️ **Both 4-bit decode kernels are bandwidth-bound with a hard ceiling** (~20 ms/step for a whole
  model — the level table makes a decode pure byte lookup). Judge them by achieved GB/s, never by
  share of the step. CUDA's does **four packed bytes per thread**, because the level lookups are
  dependent loads and one byte per thread is latency-bound; that makes `group_size % 8 == 0` a
  requirement there (`dit.w4a8SmallGroup` refuses a smaller group by name). Vulkan's is general over
  group size but requires its inputs **pre-transposed through `weightBuffer`** — a transpose's
  strided side is 32 sectors per warp-load, and paying it per GEMM rather than once at load was a
  50× gap.
- ⚠️ **The device weight caches key on HOST POINTER.** Anything that reuses a weight-shaped buffer
  (a test allocating per case, a per-session shadow of a per-model weight) scores a stale cache hit.
  `Backend.zeroBias` in particular must not be handed to `cachedWeight` — it reallocates on growth,
  so its pointer moves.
- **q1_0 is the first 128-element block** (every other quant here is 32 or 256), so anything that
  hardcoded those sizes is wrong for it; `matmul.packedTaskBlock`'s k-slice assert is per-dtype.
- ⚠️ **GGUF type id 42 is claimed by TWO formats and picking the wrong one is silent.** Both spell
  themselves "Q2_0", both compute `v = (code − 1)·d` from 2-bit codes packed 4 per byte LSB-first,
  and they differ **only in block size**: upstream ggml's `QK2_0 = 64` (18 B) against the PrismML
  fork's `prism` branch at `QK2_0 = 128` (34 B). Nothing inside a file distinguishes them — not the
  type id, not `general.file_type` (41 for both).
  - The only signal is the on-disk row length, differing by 17/18. `gguf.detectQ2_0Variant` resolves
    it from the gap to the next tensor in offset order, accepting a candidate only on
    `alignForward(size) == gap`; a gap fitting both or neither casts no vote rather than guessing.
    No vote, or two tensors disagreeing, is `AmbiguousQ2_0Variant`.
  - ⚠️ **The failure is asymmetric.** Reading a g64 file as g128 computes *smaller* spans, so the
    bounds check passes and every tensor view is silently short and misaligned. The reverse overruns
    and surfaces as `InvalidOffsets`. A `ne[0] % blockElems()` check does not save you: real hidden
    dims (5120, 4096, 14336) are multiples of 128 as well as 64.
  - **g128 is decoded natively, not by ggml** (`quants.dequantQ2_0G128` + `quants.dotQ2_0G128` +
    `matmul.native_gemv`), because the vendored ggml's `GGML_TYPE_Q2_0` is g64 and `QK2_0` is a
    compile-time define. ⚠️ `quants.ggmlType(.q2_0_g128)` returns **null** as a hard interlock,
    making every ggml entry point unreachable for the dtype. The native GEMV does **not** quantize
    the activation, so it is exact where ggml's `vec_dot` is not, at ~0.76× llama.cpp's int8 kernel.

---

## 8. Kernel inventory (appendix)

### Vulkan — `Elt` compute kernels (`src/gpu/context.zig`, bodies in `src/gpu/kernels/eltwise.zig`)

`rmsnorm` · `rmsnorm_sg` · `rms_partial` · `rms_combine` · `rms_apply_mod{,_h16}` · `rms_apply_w` ·
`modulate` · `ln_mod_sg` · `gated_add{,16}` · `add{,_h16}` · `relu` · `add_relu` ·
`silu_mul{,_h16,16}` · `sigmoid_mul{,_h16,_g16}` · `gelu` · `gelu_mul` · `gelu_quick` · `gelu_erf` ·
`layernorm` · `vae_norm` · `l2norm_rows` · `qknorm_rope16` · `qknorm_rope_f32` · `rope_inter` ·
`rope_half` · `rope_qwen35` · `attention` · `attn_scores` · `softmax_partial` · `softmax_combine` ·
`softmax_rows` · `attn_out` · `attn_dsplit` · `attn_dmerge` · `attn_full` · `attn_decode_q35` ·
`attn_causal_batched` · `attn_cross` · `gather_kmajor{,_h16,16}` · `f32_to_h16{,_pad}` ·
`h16_to_h16_pad` · `f32_to_bf16_pad` · `copy` · `deinterleave2` · `scale_concat` · `scale_i32` ·
`bias_compact{,_h16}` · `im2col` · `im2col_sd{,_h16}` · `rotate` · `rotate_fwht` · `rowmax_i8` ·
`rowscale_i8` · `quantize_i8` · `gemv_partial{,4}` · `gemv_combine{,4}` · `gn_stats{,_h16}` ·
`gn_combine` · `gn_apply{,_h16}` · `silu` · `geglu` · `concat_ch` · `head_pad_h16` · `head_unpad` ·
`i4_decode_t` · `w4a8_decode_t` · `nvfp4_decode_t` · `gemv_q8_0{,_t}` · `gemv_q4_k{,_t}` ·
`gemv_q5_k{,_t}` · `gemv_q6_k{,_t}` · `gemv_iq4_nl{,_t}` · `gdn_gates` · `gdn_conv_step` ·
`gdn_delta_step`

**Vulkan GEMM entry points** (`context.zig`): `opMatmul` (f32/fp8) · `opGemv{,Partial,Quant,QuantT}` ·
`opMatmulCoop{,H16}` (fp8→f16) · `opMatmulCoopF16W{,b,h,Dev}` (f32/bf16/f16→f16; `Dev` takes a
device-resident bias, for the SD UNet's per-forward folded ResBlock bias) · `opMatmulCoopBf16`
(native bf16) · `opMatmulCoopQuant` (GGUF block quants, prefill) · `opMatmulCoopI8{,Fused}` /
`opMatmulI8` / `opI8Gemm` (int8 s8→s32) · `opAttnScores{,Vae,Sd}` / `opFlash` / `opAttnOut`.

Coopmat SPIR-V builders in `src/gpu/coopmat.zig` (`buildGemmShared` f16/bf16, `buildGemmI8` int8,
`buildFlashAttn`, `buildGemmAttnOut`; **no s4**). Scores shaders are compiled per head width —
128 (DiT, head-padded SD UNet), 384 (Wan VAE mid-block), 512 (SD VAE mid-block) — since
`buildGemmScores` unrolls the k-depth.

### zig-cuda — hand-PTX (`src/gpu/cuda/kernels.zig` GEMM, `elt.zig` elementwise/attn)

GEMM builders: `buildHgemm` (f16/bf16 mma m16n8k16, optional f32 A/C) · `buildIgemmSmem` /
`buildIgemmPipe` (int8 m16n8k32, int4 m16n8k64) · `buildPrep` (quant/rotate) ·
`buildMmqPipeQ{1_0,2_0}` and the q4_k/q6_k MMQ pipes. All DiT GEMM formats use warp-cooperative
`ldmatrix.x4`/`.x2` fragment loads with an XOR-swizzled conflict-free shared layout
(`use_ldmatrix`; a pure permutation, so bit-exact).

GEMV: `gemv_{fp8,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k,iq4_nl,q1_0,q2_0_g64,q2_0_g128}` plus `_q8` and
grouped-N `_q8n` dp4a variants.

Attention: `attn` · `attn_split`/`_merge`/`_h256`/`_h512`/`_tree` · `attn_split_g` (group-shared KV
fragment, entry for hd ∈ {128, 256}, full causal, `heads > kv_heads`; bit-identical per head to
`attn_split`, with `_f16`/`_q8` KV variants) · `softmax_md_{f16,f32}`.

GDN: `gdn_{conv_step,gates,delta_step}` · `gdn_conv_batch` · `opGdnDeltaChunk` (state in registers
across the chunk, so bit-identical to the per-token form — ⚠️ decode can only run the per-token
form, so a re-prefill must match it exactly).

SD family: `gn_stats`/`gn_combine`/`gn_apply` (Welford) · `geglu` · `concat_ch` · `attn_cross` ·
`im2col_sd` · `head_pad_h16`/`head_unpad` · `add_bias_rows` · `gelu_quick`/`gelu_erf` ·
`f16_pad2d`. Diffusion decode: `i4`/`w4a8_decode`/`nvfp4_decode`. Vision: `rope_vision` ·
`rope_vision_gemma4` · `gelu_quick_mul`. Plus `im2col`, dtype-pad converts, `dequant_*_f16`, and
the rope/norm/act kernels.

### cuda — vendor libs (`.libs` mode)

**cuBLASLt** (`src/gpu/cuda/cublaslt.zig`): int8 `R_8I`/`COMPUTE_32I`, f16 `R_16F`, bf16 `R_16BF`,
batched/prefill GEMM only. **cuDNN** (`src/gpu/cuda/cudnn.zig`): fused SDPA-forward attention,
legacy conv-forward (VAE). Everything else stays hand-PTX.

### cpu — `src/ops/*.zig` + ggml (fetched dep, `-Dggml`, default on)

`matmul.zig` (dtype-aware microkernel; block-quant decode GEMV → ggml `vec_dot`, plus
`native_gemv` for q2_0 g128) · `attention.zig` (+`attentionTree`) · `norm.zig` · `act.zig` ·
`rope.zig` · `conv.zig` · `convrot.zig` · `w4a8.zig` · `nvfp4.zig` · `vmath.zig` (SIMD `expVec`).
ggml owns CPU block-quant dequant and decode `vec_dot`; everything else (GGUF parse, f16/bf16/fp8
conversion, GEMM threading, convrot, tokenizer, sampling) is in-house Zig. With `-Dggml=false`,
block quants are unavailable (`error.QuantBackendUnavailable`); all other dtypes and backends are
unaffected.

---

## 9. Known gaps

Delete a row when it closes.

| gap | detail |
|---|---|
| `cuda-dit-test` fails on krea2 | rel L2 0.141 (bf16) / 0.132 (int8 convrot) against the CPU forward, and the bf16 arm takes ~11.5 s for a 256px forward. krea2's bf16 CUDA arm has no working validation. |
| Vulkan attention is the diffusion perf gap | 16.0 TFLOP/s against `opAttnTC`'s 44.3 for the same work; inside it the P@V pass costs 2.1× the scores pass. Orthogonal to quantization; the GEMMs are fine (bf16 37-45, int8 77-86 TFLOP/s). |
| Vulkan weight upload | CUDA fans pinned staging over `Context.FillPool` (4 threads); Vulkan has no equivalent, so its first-image setup is ~0.69 s where CUDA's whole build is 0.32 s. Helps every model. |
| `warmWeights` is qwen35-only | qwen3, gemma3 and gemma4 charge the lazy weight upload to their prefill timer. ~30 lines each, mirroring their own `layerDeviceBytes`. |
| No Vulkan ViT except gemma3 | `vit35`, `gemma4_vit`, `gemma4v_vit` are CPU/CUDA only. |
| No Vulkan gemma4 | `Spec.Vulkan = void`; `--backend vulkan` is rejected for the arch. |
| Vulkan block-quant embedding | rejected in `llm_main.zig` — no block-quant gather kernel, so an f16/bf16 embed table is required. |
| Vulkan dp4a decode is opt-in | `TP_VK_DP4A=1`; the repacked int8 weight roughly doubles VRAM, so it stays opt-in until VRAM-aware auto-sizing lands. |
| `opMatmulFp8` writes `y` directly | unlike `opGemmBf16`/`opMatmulNvfp4` it carries `launchHgemm`'s `mpad`-rows requirement implicitly. Its zimage/anima `.f8_e4m3` arms have never been exercised and would hit it the day an fp8 checkpoint for either shows up. |
| No GPU GEMM for GGUF block quants in diffusion | a GGUF DiT is CPU-only on every backend; `gpuLinKindSupported` is the one place to widen. |
| krea2 has no Vulkan int4 path | `dit_gpu` never accepted it. `i4_decode_t` is now most of what it would need. |
| q2_0 g64 kernels unexecuted | GEMV and MMQ are generated from the same templates as g128 but no g64 file exists here to run them against. |
| `--text-encoder-2` split path unexercised | every SDXL checkpoint here is bundled, so the flag is built and reviewed but not measured. |
