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

- ⚠️ **A cuBLASLt GEMM writing f16 D needs an f16 bias**; an f32 bias over f16 D gets no algo
  (`INVALID_VALUE`). `cachedBiasF16` holds the converted copy.
- Under `.libs`, `opMatmulF16` / `opMatmulBf16` / `opConvF16Prec` go through `ltGemmHalfBias`
  (no padding, bias in the epilogue) when `k % 8 == 0`. Otherwise, and on `zig-cuda`, the
  padded `[m_pad][k_pad]` path runs, padded to the hand-PTX tile.
- A bf16 weight feeding an f16 GEMM is rewritten to f16 in place on first use (`cachedWeightF16`).

**Dual-target kernels** (`src/gpu/kernels/dual.zig`, bodies in `dual/*.zig`, entry table in
`dual_table.zig`): one Zig source compiled to SPIR-V for the Vulkan `Context` and to PTX for
both CUDA arms, over one launch ABI (four f32 buffers, seven u32, two f32). Every entry has a
Vulkan `Elt` tag and a CUDA `dualElems` / `dualPairs` / `dualRows` launch; layouts are the
Vulkan push layouts and the CUDA wrappers remap. It holds:
- the elementwise set: activations, copies, modulation, residual gates, every RoPE variant
  (M-RoPE and the two vision ropes included), the tensor-core attention head gathers, f16/bf16
  conversions and padded copies, im2col, MoE row routing, sampling passes, GDN gates and conv,
  the H3 audio kernels;
- every block-quant dequantizer (f16, bf16 and f32 output, raw row-major input);
- the row reductions (rmsnorm, grouped rmsnorm, layernorm, the two fused norm+AdaLN kernels,
  l2norm, GroupNorm statistics, int8 row max), one subgroup per row.

Per backend: the tensor-core GEMMs and MMQ pipes (`cuda/kernels.zig`, `coopmat.zig`), the dp4a
GEMVs, the attention decompositions, the graph-capture kernels that read `g_state`, and anything
needing workgroup memory (Zig-emitted workgroup storage hangs NVIDIA under Vulkan, ZIG.md). The
weighted rmsnorm keeps one hand CUDA twin, `qk_rmsnorm_par` (a block per row), used below 64 rows
at width 1024 or more (the LLM decode hidden norm). `dual-cuda-test` checks both.

Legend: ✅ full · ⚠️ works but slow / limited · ❌ unsupported · — not applicable

---

## 1. Top-level capabilities

| Capability | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **Diffusion txt2img** (every family) | ⚠️ ref | ✅ | ✅ | ✅ **primary** |
| **LLM text generation** | ⚠️ ref | ✅¹ | ✅ | ✅ **primary** |
| **LLM vision (ViT/mmproj)** | ✅ | ✅ | ✅ | ✅ |
| **GPU init failure** | — | → CPU fallback | → CPU fallback | → CPU fallback |

¹ A block-quant token embedding is host-gathered. The `Context.dequantOnly` formats have no
transposed GEMV and prefill through the dequant GEMM: q4_0 / iq4_xs / q1_0 / q2_0 decode through
the cooperative subgroup GEMV, q2_k / q3_k / iq2_xxs / iq2_xs / iq3_xxs / iq3_s through the dual
one. See §4–§5.

---

## 2. Diffusion (`src/pipeline.zig`)

Per-stage dispatch order everywhere is `if (cu_be)` → CUDA, `else if (gpu_ctx)` → Vulkan,
`else` → CPU. Every stage of every family runs on every backend.

**Cancellation** (`Options.cancel`) is polled between sampling steps and mid-stage on every
backend: between DiT blocks, text-encoder layers, VAE decode layers, and per tile in
`vae_tiled`. On **cpu** the threaded matmul/attention kernels also poll a threadlocal token
(`src/ops/cancel.zig`) per row-panel / k-block / query row. `error.Canceled` is never
swallowed: the VAE OOM ladder and the GPU→CPU encode fallback both propagate it.

### 2A. krea2

Single-stream DiT over a Qwen3-VL-4B conditioning, Wan 2.1 VAE.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-4B) | ✅ f32 | ✅ follows `--te-act` | ✅ fp8→f16 TC, follows `--te-act` | ✅ fp8→f16 TC, follows `--te-act` | `krea2_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (28 blocks) | ✅ all dtypes | ✅ fp8/int8/int4/w4a8/nvfp4/bf16 | ✅ + GGUF block quants | ✅ + GGUF block quants | `dit{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1) | ✅ | ✅ | ✅ | ✅ (+cuDNN conv) | `wan_vae{,_dev}.zig` over `sd_vae_{gpu,cuda}.zig` |
| **VAE tiling** | CPU-tile | GPU-tile + CPU floor | ↤ | ↤ | `vae_tiled.zig` |
| **TAEHV preview** | ✅ | ✅ | ✅ | ✅ | `taehv{,_gpu,_cuda}.zig` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `wan_vae.latentPreviewInto` |
| **LoRA sidecar** | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J; text fusion included. |

- **DiT activation default is f16** (`Session.defaultAct`), not ComfyUI's bf16.
- The stream follows the dial on cuda and Vulkan (`dit_cuda.streamDtFor`, `dit_gpu.streamDtFor`):
  residual, norm outputs and `wo` / `down` outputs at f16 or bf16; the attention block runs f16
  under either 16-bit stream. **zig-cuda keeps the stream f32.**
- Vulkan's fp8 cooperative GEMM accumulates in f16 (`coopmat.coop_acc_h16`). A staged fp8 group
  stores f16 output from f16 accumulators only when the caller asks (`lin_gpu.Out.acc16`), which
  krea2 does under the f32 stream only.

Gates: `cuda-dit-test`; `velocity-diff <ckpt> --dit-act`; `diffusion-profile --ab act:D` against `--ab perturb`.

### 2B. The SD family (SD1.5 / SDXL)

LDM UNets over CLIP-L (SD1.5) or CLIP-L + CLIP-G (SDXL), AutoencoderKL VAE.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **CLIP-L / CLIP-G** | ✅ ref | ✅ dense; integer weights on host | ✅ | ✅ | `clip_text{,_gpu,_cuda}.zig`. Every packed format loads through `quant_weight.load`. On CUDA every linear goes through `lin_cuda` (`clip_text_cuda.planFor`, `Plan.dense = .f16`; integer weights W8A8 under `auto`, weight-only at an explicit `--te-act`) |
| **UNet** (LDM `UNetModel`) | ✅ ref | ✅ | ✅ | ✅ | `sd_unet{,_gpu,_cuda}.zig`. On CUDA every linear goes through `lin_cuda` (`Plan.dense = .f16`), so int8 convrot and GGUF block quants run. Convs stay in checkpoint order and dtype; see "Convolution weights" |
| **VAE decode** (AutoencoderKL) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` |
| **LoRA sidecar** (LoRA, LoCon, Tucker, LoHa, LoKr, DoRA, diffs; UNet and both towers) | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J. |

**UNet activation storage** (`sd_unet_cuda.streamDt`, `sd_unet_gpu.streamDt` → `Workspace.act`):
f16 where `sd_unet.Config.act_f16` says, else what `--dit-act f32|f16|bf16` (alias `--unet-act`)
asks. Kernels compute in f32 and per-channel parameters stay f32; only the big buffers narrow.
- zig-cuda keeps Q, K, V and the attention output f32. Vulkan keeps the attention operands f32
  and needs `pipe_coop_f16w` (bf16: `pipe_coop_bf16w`) for a 16-bit stream, else runs f32.
- ⚠️ **Every buffer a GEMM reads must be at the stream's width**, including the text
  conditioning (`Session.ctx_d`, a cross-attention K/V source). Read as f32 it renders NaN.
- ⚠️ **An activation capture (`ops.matmul.probe`) and a 16-bit stream are mutually exclusive**;
  the probe is gated off.
- ⚠️ **A bf16 chain runs bf16 GEMM operands everywhere** (`Backend.operand16`: bf16 when either
  side is bf16). The dense-16 GEMM (`opGemm16`), cuDNN conv and SDPA, LoRA A/B and the int8
  routes all honour it. Vulkan's coop GEMM does the same (`coopOperandBf16`) and refuses a bf16
  chain on a device without the bf16 cooperative pipeline.
- ⚠️ **The SDXL VAE overflows f16**: every GPU arm divides the residual by `residual_act_div`
  (256) before the cast, on the residual-reading convs only (`opConvF16Scaled`,
  `opMatmulCoopF16WScaled`, `conv2d`). Those kernels' spare f32 push constant must be 1.0, not 0.0.

**Attention head width.** SD1.5 attends at head_dim 40/80/160, SDXL at 64:

| backend | how it attends |
|---|---|
| **cuda** | cuDNN fused SDPA at the true width, self and cross (`opAttnCross` → rectangular SDPA) |
| **zig-cuda** | zero-padded to a multiple of 128 (`opAttnTCRect` for cross) |
| **vulkan** | zero-padded to 128; hd 160 uses the scalar kernel |

The padding is exact. The 128 limit is the P@V GEMM's head tile (`coopmat.buildGemmAttnOut`,
`launchHgemmB`).

- **CLIP batches over prompt chunks** on the device; the empty-prompt `z_empty` rides as one more
  item, cached per tower.
- ⚠️ **CLIP precision differs per GPU arm.** Vulkan computes f32 at the default width; CUDA has
  only a tensor-core GEMM there and computes f16 whatever the storage width, so a `cuda` and a
  `cpu` render of one seed differ slightly at the conditioning.
- ⚠️ **The two SDXL towers use different GELUs** (CLIP-L quick-GELU, CLIP-G erf-GELU), close
  enough to look right. `cfg.act` carries which.

**Convolution weights.** One conv module per backend (`conv_cuda.zig`, `conv_gpu.zig`) runs every
convolution: `conv2d` (1x1, 3x3 at stride 1 or 2, fused 2x upsample; cuDNN on `cuda` where it can,
else a gather into the 16-bit operand), `patchGemm` / `convStride` (SenseNova patch embeds),
`conv1d` (vocoder, audio encoder) and `conv3d` (CUDA, H3 video encoder). The 1-D and 3-D forms
gather an f32 patch and run an f32 plan through `lin_cuda` / `lin_gpu`.
- Loader rule (`quant_weight.convWeights`, every VAE family, the SD UNet, SenseNova's vision
  embeds and `fm_head`): checkpoint order for `co >= 64` and any dense or block-quant dtype.
  LPIPS is the only repacking loader.
- CUDA (`Backend.weightF16`): every f16 conv kernel reads an f16 `[co][k][k][ci]` device weight.
  A dense one is converted once (`conv_w_h16`) and cached; a block quant stays packed and is
  decoded per call (`opDequantF16`, so GGUF `flat_blocks` convs work).
- Vulkan decodes a checkpoint-order conv once at first use and keeps only the 16-bit device
  weight (`Context.convWeight16`); it repacks only on a device without the f16 coop GEMM
  (`conv_gpu.needsRepack`).
- `wan_vae` keeps its 5-D causal kernels whole in checkpoint order (`Weights.Torch.kt`; the
  repack and `conv_w_h16` read slice `kt - 1`). TAEHV repacks everything; `mage_vae` repacks on
  Vulkan (`MageVae.repackConvs`). A CPU-tiled VAE decode uses a decode-scoped f32 copy.
- ⚠️ A `conv1d` whose `dst` aliases `src` must fit one band; it refuses to split.

Gates: `sd-cuda-test`; `lin-cuda-test` (the conv resolver through `conv_cuda.conv2d`, f32/f16/bf16/flat-block q4_k, both orders, both arms; `store f16` / `store bf16` arms); `lin-vk-test`; `lora-cuda-test`; `cuda-vae-test`; `velocity-diff <ckpt>`.

### 2C. Z-Image ("zit", `NextDiT`)

NextDiT over a Qwen3-4B penultimate-layer conditioning, Flux AutoencoderKL.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-4B, penultimate) | ✅ f32 | ✅ | ✅ | ✅ | `zimage_text.zig`, `qwen3{,_gpu,_cuda}.zig` |
| **DiT** (30 blocks + 2+2 refiners) | ✅ bf16/f16/f32/fp8/GGUF | ✅ + nvfp4 | ✅ | ✅ | `zimage{,_gpu,_cuda}.zig` |
| **VAE decode** (`AutoencoderKL`, 16-ch) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` (`sd_vae.flux`) |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `zimage.latentPreviewInto` (Flux factors + bias) |
| **LoRA sidecar** | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J; split `to_q/to_k/to_v` keys land on the fused `qkv` (`zimage.lora_keys`). |

- **DiT activation default is f32** (`zimage_dev.streamDtFor`): explicit f16 goes non-finite at
  step 1 and fails with `NonFiniteActivation`; bf16 works. The attention block stays f32.
- ⚠️ **fp8 needs `lin_cuda.Plan.act16 = .bf16` on both CUDA arms**: the MLP down projection reads
  ~1e5, and f16 staging renders white. NVFP4 decodes to bf16.
- ⚠️ **V is prescaled by 2^-6 across attention's f16 cast** (`zimage_cuda.v_div`, `zimage_gpu.v_div`;
  the two must stay equal). V is unnormed and overflows f16 at depth; unscaled, the render is solid
  white with no error.
- ⚠️ **CUDA takes the folded modulation table** (`zimage.modulationTableFolded`, for `rmsMod`);
  Vulkan takes the unfused one.
- ⚠️ The fused `qkv` splits into row views at load (`lin.rowSlice`); `q` shares the fused tensor's
  host pointer, which keys the device weight cache, so never upload both. The `noise_refiner` blocks
  run on the image half alone with a sliced rope table.
- Head width is 128 (no padding). The timestep MLP, AdaLN linears (precomputed at `Session.init`),
  the caption half, patchify and the final layer run on the host.

Gates: `zimage-cuda-test --layers --lat --cap --sigma` (overflow only shows at depth and real scale), `--dequant-at-load f32|bf16` (dense path for the same file); `zimage-cuda-bench`.

### 2D. Anima (Cosmos-Predict2 `MiniTrainDIT` + `llm_adapter`)

Cosmos-Predict2 trunk conditioned through a T5-indexed LLM adapter over Qwen3-0.6B, Wan 2.1 VAE.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **T5 tokenizer** (SentencePiece Unigram) | ✅ | — | — | — | `core/t5_tokenizer.zig` |
| **Text encoder** (Qwen3-0.6B, final state + `model.norm`) | ✅ | ✅ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig`, `Variant.anima` |
| **`llm_adapter`** (6 blocks, 1024, T5-indexed) | ✅ | host | host | host | `anima.zig` (`Adapter`) |
| **DiT** (2048 wide; depth counted from the checkpoint) | ✅ bf16/f16/f32/fp8/int8/int4/w4a8/nvfp4 | ✅ int8/int4/w4a8/nvfp4/bf16 | ✅ | ✅ | `anima{,_gpu,_cuda}.zig` |
| **VAE decode** (Wan 2.1, 16-ch) | ✅ | ✅ | ✅ | ✅ | `wan_vae{,_dev}.zig` over `sd_vae_{gpu,cuda}.zig` |
| **TAEHV / latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | krea2's Wan matrix, **not** Z-Image's Flux one |
| **LoRA sidecar** | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J; trunk, AdaLN and `llm_adapter`. |

- **DiT activation default is f32** (`anima_dev.streamDtFor`): explicit f16 goes non-finite at
  step 1. bf16 works. The attention block stays f32.
- The `llm_adapter` is timestep-independent and runs once per image inside `encode`.
- Cross-attention K/V are precomputed per block at `Session.init` (Vulkan f16 in the attention
  layout, CUDA f32; doubled under CFG). ⚠️ On Vulkan that is **one buffer per block**: offsetting a
  `DeviceBuffer.buf` gives `error_device_lost`; use `Context.view`.
- Both CUDA arms run the hand-PTX `opAttnTCRect` on the `seq × 512` cross-attention (Anima does not
  go through `opAttnCross`). The naive kernel `opAttnCrossNaive` is the validation reference only.
- ⚠️ On Intel Alchemist the 8x8x16 coop config is the only f16/bf16 GEMM; without it `anima_gpu`
  refuses a bf16 weight.

**Kernels this trunk relies on:**

| kernel | where | what |
|---|---|---|
| `ln_mod` | `kernels/dual/rows.zig` | fused weightless LayerNorm + AdaLN modulation, one subgroup per row on both arms. Two-pass variance, matching `ops.norm.layerNormUnit` |
| `pmdplane` (push word 7) | `coopmat.buildFlashAttn` | rectangular q × kv flash attention. `s_stride` is K's row stride and the j bound; the MD table is indexed by query row. **0 means "= s_stride"** (square callers) |
| `opAttnTCRect` | `cuda/backend.zig` | `opAttnTCBatched` with the two sequence lengths separate |

**Per-weight quantization and the int8 prep** (every family):
- ⚠️ **Quantization kind is resolved per weight** (`lin_cuda.routeOf`, `lin_cuda.prep` per shared
  activation). Real checkpoints mix dense and quantized blocks; `DiT.device_lins` is the one list
  every support scan reads.
- ⚠️ **Vulkan int4 is W4A8-shaped, not W4A4** (no `sint4` coopmat): the weight decodes to int8 per
  GEMM (`i4_decode_t`). The same file scores differently per backend. Both 4-bit formats decode
  per GEMM, never at load.
- The Vulkan int8 activation prep is one fused f32 kernel (`coopmat.buildFusedPrepI8`, built per
  width by `Context.i8PrepPipe`, plan from `coopmat.fusedPrepPlan`) at any width that is a multiple
  of 256 up to 32768; past that it takes the 3-pass chain. It matches `ops.matmul.emulateActQuant`.

Gates: `anima-cuda-test`; `lin-vk-test` (every model width of the int8 prep against the emulation); `anima-cuda-bench`, `anima-vk-bench`, `vk-prep-bench`, `diffusion-profile --ab prep3`.

### 2E. MiniMax H3 (joint audio-video DiT)

Video and stereo audio denoised in one packed token sequence on two sigma schedules.
VIDEO_PLAN.md is the roadmap; the conventions live in the `minimax_h3*` module headers.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-32B, 50 layers, UNNORMALIZED last state) | ✅ int8-convrot | ✅ int8-convrot | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig`, `Variant.minimax_h3`. Does not fit a 24 GB card: CUDA streams a layer ahead (`prefetchLayer`), Vulkan through the context's weight LRU |
| **DiT trunk** (50 blocks, 5376, 56x128) | ✅ int8 | ✅ int8 | ✅ int8 | ✅ int8 | `minimax_h3{,_gpu,_cuda}.zig`, one walk in `minimax_h3_dev.zig`. `DiT.device_lins` (fused qkv and fc1 as row views) through `lin_cuda` / `lin_gpu`, so any dispatcher format and LoRA reaches H3. Attention in 4096-row query bands; output heads on the device |
| **Video VAE decode** (ViT3D, 36 blocks, temporal chunking) | ✅ | ✅ | ✅ | ✅ | `minimax_h3_vae{,_gpu,_cuda,_dev}.zig`. Head width 64 padded to 128. GEMMs through `lin_cuda` / `lin_gpu.towerLinear` |
| **Audio VAE decode** (BigVGAN, 32 kHz stereo) | ✅ | ✅ f32 | ✅ f32 | ✅ f32 | `minimax_h3_audio{,_gpu,_cuda}.zig`. Device prep (`DevSession`) is in the model file, shared by both arms |
| **Video VAE encode** (3-D causal CNN, banded im2col) | ✅ | host | ✅ | ✅ | `minimax_h3_vae_encode{,_cuda}.zig`, references and keyframes |
| **Audio VAE encode** (DAC + causal-attention posterior head) | ✅ | host | ✅ | ✅ | `minimax_h3_audio_encode{,_cuda}.zig`, reference soundtracks |
| **Vision blocks** (fl2va / ref2va conditioning) | ✅ | ✅ (tower on host) | ✅ | ✅ | `minimax_h3_vit.zig` + `qwen3{,_gpu,_cuda}.encodeVision`, one walk in `qwen3_enc_dev.zig` |
| **Denoise masks** (per-row timesteps) | ✅ | ✅ | ✅ | ✅ | `Timesteps.initMasked`; per-row index buffer into the shared `rms_mod` / `gated_add` |
| **Spatial VAE tiling** | ✅ | ✅ | ✅ | ✅ | required: the device session is sized for one tile; a window-sized session misses every tile and falls back to the host |
| **MP4 muxing** (H.264 + AAC, `parameters` tag) | ✅ | ✅ | ✅ | ✅ | `lib/video/av_helper.c`, `src/av.zig` (exe only) |
| **LoRA sidecar** | ✅ any dtype | ✅ (`lora_gpu`) | ✅ bf16, any rank | ✅ bf16, any rank | §2J. `TP_H3_LORA=<file>` attaches one in both device tests |

- `generate-clip --backend vulkan` renders text-to-clip on the device; the encode sides and the
  vision tower fall back to the host.
- **Trunk activation default is f32** (`minimax_h3_dev.streamDtFor`): `x_d` and the norm output at
  the dial's width, deltas and MLP planes f32. An f16 residual is stored divided by 256 with every
  RMSNorm reading it at `eps / 65536`. The clip path takes the dials (`ClipOptions`,
  `generate-clip --dit-act`) and reruns one dtype wider on a non-finite step.
- **Video VAE default is f16**; QKV through the attention output stay f32. A non-finite window
  reruns only the video decode wider (`Session.decodeClip`).
- ⚠️ **The audio VAE runs f32 GEMMs, never tensor cores** (`opMatmulF32Lt`, `COMPUTE_32F`, not
  `_FAST_TF32`; `kernels.buildSgemm` on zig-cuda): the f16 route is an audible noise floor.
  `TP_H3_AUDIO_F16=1` runs the f16 plan as a diagnostic.
- ⚠️ The trunk's device-vs-CPU error jumps at the last block (its `fc2` scales are ~3x smaller), so
  read that figure as "no defect", not a precision bound.

Gates: `minimax-h3-cuda-test` / `minimax-h3-vk-test` (`TP_H3_MASK=binary|graded|spatial`, `TP_H3_BLOCKS` depth, `TP_H3_ACT` residual dtype, `TP_H3_WEIGHT_ONLY`); `minimax-h3-vae-cuda-test` / `minimax-h3-vae-vk-test` (`TP_VAE_MAG`); `minimax-h3-audio-cuda-test` / `minimax-h3-audio-vk-test`; `minimax-h3-vae-encode-cuda-test`, `minimax-h3-audio-encode-cuda-test`; `te-test <file> --minimax-h3`, and `TP_TE_VISION=1 te-test` for the vision path.

### 2F. Diffusion speed snapshot

Seconds per step, ReleaseFast, RTX 3090. "—" is unmeasured. ComfyUI is the same render in
ComfyUI on the same card.

| card | family | checkpoint format | size | steps, cfg | cpu | vulkan | zig-cuda | cuda | ComfyUI |
|---|---|---|---|---|---|---|---|---|---|
| RTX 3090 | SD1.5 | — | 512² | 8, 7.5 | 6.91 | 0.92 | 0.46 | 0.27 | — |
| RTX 3090 | SDXL | — | 1024² | 8, 7.5 | 50.06 | 2.20 | 1.83 | 1.11 | — |
| RTX 3090 | krea2 | int8 | 1120x1680 | —, 1 | — | — | — | 2.25 | — |
| RTX 3090 | krea2 | w4a8 | 1120x1680 | —, 1 | — | — | — | 2.26 | — |
| RTX 3090 | krea2 | int4 | 1120x1680 | —, 1 | — | — | — | 1.95 | — |
| RTX 3090 | krea2 | nvfp4 | 1120x1680 | —, 1 | — | — | — | 4.36 | — |
| RTX 3090 | krea2 | bf16 (streams) | 1024² | —, 1 | — | — | — | 3.3-7.0 | — |
| RTX 3090 | krea2 | fp8 | 1024² | —, 1 | — | — | — | 2.45 | — |
| RTX 3090 | krea2 | q8_0 via f16 | 1024² | —, 1 | — | — | — | 2.46 | — |
| RTX 3090 | krea2 | q8_0 via int8 | 1024² | —, 1 | — | — | — | 1.37 | — |
| RTX 3090 | krea2 | int8 | 1024² | —, 1 | — | — | — | 1.14 | — |
| RTX 3090 | krea2 | q4_k via f16 | 1024² | —, 1 | — | — | — | 2.40 | — |
| RTX 3090 | krea2 | q4_k via int8 | 1024² | —, 1 | — | — | — | 1.33 | — |
| RTX 3090 | krea2 | q2_k via int8 | 1024² | —, 1 | — | — | — | 1.29 | — |
| RTX 3090 | krea2 | q2_k via int4 | 1024² | —, 1 | — | — | — | 1.21 | — |
| RTX 3090 | krea2 | int4 | 1024² | —, 1 | — | — | — | 0.95 | — |
| RTX 3090 | Z-Image | — | 1056x1584 | 9, 1 | — | 3.80 | — | 2.00 | 1.836 |
| RTX 3090 | Anima | — | 512x768 | 30, 5 | 43.60 | 0.59 | 0.42 | 0.37 | 0.33 (bf16) |
| RTX 3090 | Anima | — | 1056x1584 | 30, 5 | — | 3.11 | 2.32 | 1.62 | 1.31 (bf16) |
| RTX 3090 | SenseNova | — | 1024² | 8, 1 | 30.30 | — | — | 0.48 | — |
| RTX 3090 | MiniMax H3 | int8 | 256², 5 frames | 8, — | — | 0.88 | — | 0.14 | — |
| RTX 3090 | Mage-Flow | — | 1024² | 30, 5 | — | — | — | 0.755 | 0.621 |
| RTX 3090 | MOSS-Image | int8 | 1024² | 25, 2.5 | — | 2.29 | 1.70 | 1.17 | 0.694 (int8 attention) |

### 2G. SenseNova U1.5 (pixel-space MoT)

Not a DiT: one Qwen3-shaped 8B trunk carrying two weight copies per layer. The prompt
runs the base copy causally and leaves a KV cache (the conditioning); the canvas, in
pixel space at one token per 32x32 px with no VAE, runs the `_mot_gen` copy against it,
unmasked. `models/sensenova.zig`'s header lists the silent conventions, `sensenova_gpu.zig`'s
what the Vulkan arm does differently.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Prefix pass** (42 layers, base copy) | ✅ | ✅ | ✅ | ✅ | `sensenova{,_gpu,_cuda}.zig`; the CUDA arms prefetch it a layer ahead and leave it to the LRU |
| **Generation trunk** (42 layers, `_mot_gen` copy) | ✅ | ✅ | ✅ | ✅ | `sensenova{,_gpu,_cuda}.zig` |
| **Vision patch embedder** (16x16 conv, interleaved 2-D rope, 2x2 merge) | ✅ | ✅ | ✅ | ✅ | `im2col_stride` + `rope_inter_span_pos` (`dual/`) |
| **`fm_head`** (shuffle 2, 3x3, shuffle 2, 3x3, shuffle 8) | ✅ | ✅ | ✅ | ✅ | `pixel_shuffle` + banded `im2col_sd` (cuDNN on `cuda`) |
| **Understanding tower** (reference pictures) | ✅ | host | host | host | once per conditioning |
| **Image editing** (1..n reference pictures) | ✅ | ✅ | ✅ | ✅ | block-causal prefix via `attention.kv_end`; `opAttnBatched` / `attnBatched` take the per-query key range as a buffer |
| **Decode** | ✅ | ✅ | ✅ | ✅ | no VAE: `clamp((x+1)/2)` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | exact |
| **LoRA sidecar** (8-step turbo) | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J; the shipped turbo LoRA patches only the `_mot_gen` copy |

- ⚠️ **The base copy's `mlp.down_proj` cannot be q4_k.** With it quantized the prefix KV
  is gone by layer 5 and every prompt renders the same prompt-blind picture. q5_k or
  better works; `mlp_mot_gen.down_proj` at q4_k is fine.
- The prompt pass takes the text encoders' block-quant route (`--te-gguf-gemm`, default
  f16) and follows `--te-act` (default f32); the KV is handed over in f32.
- The canvas stream follows `--dit-act` on every arm (`sensenova_dev.streamDtFor`, default
  f32). At f16 the residual is stored divided by 16 with every RMSNorm reading it at
  `eps / 256`; raw f16 overflows at the last step. Sublayer delta, attention planes,
  vision tower and `fm_head` stay f32.
- Integer weights run W8A8 under `auto`; an explicit `--te-act` runs the prefix
  weight-only (§2I), which is much closer to the dense model.
- When the canvas quantizes an activation, the device check's reference is the CPU
  emulation of that route (`ops.matmul.act_quant`, per weight via each arm's
  `canvasActQuant`). Block-quant int8 decodes are not emulated.

Gates: `sensenova-cuda-test` / `sensenova-vk-test` (prefix KV and the generation forward,
both attention paths, aligned and padded canvas, every activation width, `--layers`,
`--prefix-bench N`).

### 2H. Mage-Flow (double-stream MMDiT + a one-step diffusion codec)

Double-stream: text and image tokens keep their own modulation, norms and MLP and meet
only inside the attention. The silent conventions live in the module headers:
`mageflow.zig` and `mage_vae.zig` for the model and codec, `mageflow_cuda.zig` and
`mageflow_gpu.zig` for each device arm.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-4B, normed final state) | ✅ | ❌ | ✅ | ✅ | `mageflow_text.zig`, `qwen3{,_gpu,_cuda}.zig`, `Variant.mageflow` |
| **DiT** (12 double-stream blocks) | ✅ all dtypes | ✅ every format `lin_gpu` routes | ✅ + int8/int4/w4a8/GGUF | ✅ + int8/int4/w4a8/GGUF | `mageflow{,_gpu,_cuda}.zig` |
| **VAE decode** (Mage-VAE, 128-ch at 16x) | ✅ | ✅ | ✅ | ✅ | `mage_vae{,_dev,_gpu,_cuda}.zig` |
| **VAE decode** (Flux2 anchor, same latent) | ✅ | ✅ | ✅ | ✅ | `sd_vae{,_gpu,_cuda}.zig` + `sd_vae.PackedLatent`; `pipeline.MageFlowVae` picks by file contents |
| **VAE encode** (Mage-VAE only) | ✅ | host | host | host | `mage_vae.zig`, once per reference image |
| **Vision tower** (Qwen3-VL-4B ViT + DeepStack) | ✅ | ❌ | host | host | `minimax_h3_vit.zig` (`Config.qwen3vl_4b`) + `qwen3{,_cuda}.encodeVision` |
| **Image editing** (1..n reference pictures) | ✅ | ✅ | ✅ | ✅ | `mageflow_text.prepare` + `Session.mageflowRefLatents` |
| **latent2rgb preview** | ✅ | ✅ | ✅ | ✅ | `mage_vae.latentPreviewInto` |
| **LoRA sidecar** | ✅ | ✅ (`lora_gpu`) | ✅ | ✅ | §2J; both streams and the modulation linears |

- ⚠️ The Flux2 anchor cannot encode, so a reference image with that VAE is refused by name.
- The MLP down projection reads past f16 at sigma 1, so both device plans stage 16-bit
  operands at bf16 (`act16`).
- The codec follows `--vae-act` on every arm, default f16. The windowed attention's
  operands, the latent, the RGB and the per-pixel MLP residual stay f32 at every width.
  The CUDA window attention is `opAttnTCFlash` per window.
- An activation-quantized checkpoint is checked as a weight-only walk against the exact
  forward (`force_weight_only`), plus activation noise within 1.5x of the emulated
  format's, plus each route's quantizer in `lin-cuda-test` / `lin-vk-test`. The emulation
  covers only `device_lins`; the modulation runs on the host.

Gates: `mageflow-cuda-test` / `mageflow-vk-test` and `mage-vae-cuda-test` /
`mage-vae-vk-test` (`--layers`, `--lat`, `--cap`, `--refs`, `--sigma`; `--bench N` times
the decode at both widths). ComfyUI fixtures: `tools/gen_mageflow_*.py`,
`gen_mage_vae_fixtures.py`.

### 2L. MOSS-Image (single-stream DiT over Flux 2's latent, Qwen3.5-9B encoder)

42 single-stream blocks, patch 1 over Mage-Flow's 128-channel latent (either codec
decodes it), one modulation shared by every block, a soft-capped `situ` FFN and
split-half 4-axis RoPE. Silent conventions: `moss_image.zig`'s and
`moss_image_text.zig`'s headers.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3.5-9B, HF safetensors, final normed state) | ✅ | ✅ | ✅ | ✅ | `moss_image_text.zig`, `qwen35.Model.loadHf`, `{CudaLM,VulkanLM}.encode` |
| **Text refiner** (2 blocks, once per conditioning) | ✅ | host | host | host | `DiT.textTokens` |
| **DiT** (42 blocks) | ✅ all dtypes | ✅ every format `lin_gpu` routes | ✅ | ✅ | `moss_image{,_dev,_gpu,_cuda}.zig`; fused `qkv` / `ffn_up_gate` run as row views |
| **VAE decode** | ✅ | ✅ | ✅ | ✅ | Mage-Flow's, `pipeline.MageFlowVae` |
| **LoRA sidecar** | ✅ | ✅ | ✅ | ✅ | §2J; split q/k/v keys land on thirds of a fused `qkv` (`lora_keys_fused`) |

- ⚠️ The HF encoder's DeltaNet V heads are GROUPED (v-head h reads k-head h / (v/k)),
  a GGUF's tiled (h % k). Every GDN kernel takes the group (`Config.v_group`); permuting
  the heads instead would break an int8 convrot `out_proj`.
- The `1 + w` norms and `A = -exp(A_log)` are applied at load, in f32.
- ComfyUI runs this checkpoint with int8 attention, which nothing here runs; compare
  against `tools/gen_moss_image_render.py --no-int8-attention`.
- At full depth near sigma 1 the W8A8 noise ratio is reported, not gated.

Gates: `moss-image-cuda-test` / `moss-image-vk-test` (`--layers`, `--lat`, `--cap`,
`--sigma`; `--prof` profiles one CUDA forward) and `moss-image-te-test --backend`.

### 2M. Qwen-Image 2.1 (block-causal single-stream DiT, Qwen3-VL-8B, RGBA Wan 2.2 VAE)

32 single-stream blocks at 4096, patch 1 on a 64-channel latent at 16x. The sequence is
the prompt (reference latents spliced at their slots) then the target; text is causal and
an image block sees through its own end. The prefix is modulated at t = 0, so the device
arms run it once per conditioning and keep each block's K/V; a step runs only the target
rows against `[cached prefix | own]`, unmasked. Silent conventions: `qwen_image21.zig`'s
and `qwen_image21_text.zig`'s headers.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-8B, last state unnormed, system turn stripped) | ✅ | ✅ | ✅ | ✅ | `qwen3.Variant.qwen_image21`, `qwen_image21_text.zig` |
| **DiT** (32 blocks) | ✅ all dtypes | ✅ every format `lin_gpu` routes | ✅ | ✅ | `qwen_image21{,_dev,_gpu,_cuda}.zig`; base files fuse the FFN (`gate_up`), turbo files ship it split |
| **Prefix attention** | `kv_end` | `attn_causal_batched` (text), rect `attn_gpu.core` (image blocks) | `opAttnTCCausal` / `opAttnTCRect` | same | one call per layout part |
| **VAE decode** (Wan 2.2 layout, temporal kernel 1, RGBA) | ✅ | ✅ f32 only | ✅ f32 only | ✅ f32 only | `wan_vae.loadNested`, `dup_up_add` |
| **LoRA sidecar** | ✅ | ✅ | ✅ | ✅ | §2J; `qwen_image21.loraKeys` |

- ⚠️ The VAE runs at f32 only: at f16 its activations outgrow 65504 (NaN).
- The schedule is diffusers', not ComfyUI's (`schedule.DiffusersFlow`): mu linear in the
  target's token count, exponential shift, stretch to `shift_terminal` 0.02. The
  `reference` scheduler is it; the family's flux table takes the same mu, so every
  ComfyUI scheduler also moves with resolution.
- Turbo is the `turbo` scheduler's 8 fixed sigmas. Base and turbo checkpoints cannot be
  told apart by weights.
- Released formats: bf16, int8, W4A8, NVFP4, W6A8, and Nunchaku's SVDQuant int4 and fp4
  under diffusers names (`Config.nunchaku_names`: `to_qkv` fused, `net.0.proj` rows
  `[up | gate]`, the reverse of `gate_up`). Nunchaku runs the fp4 file on Blackwell only;
  here it decodes like int4, so every arm takes it.

Gates: `qwen-image21-cuda-test` / `qwen-image21-vk-test` (`--layers`, `--lat`, `--cap`,
`--sigma`, a checkpoint path; two sigmas on one session, so the second reads the cached
prefix), `wan-vae-test <qwen vae>`.

### 2I. DiT block weight-dtype support

The CUDA columns hold for every family: one dispatcher, `lin_cuda`, routes each block
linear by its own dtype and shape (`dual_lin.routeOf`, executed by `lin_cuda.exec`, which
the LLM steppers use too). The Vulkan column holds the same way through
`models/lin_gpu.zig` over the device's `Caps`. A gap between backends is a difference
between `lin_cuda.Caps` and `lin_gpu.Caps`. `plan` refuses an unsupported checkpoint by
tensor name at session build.

| DiT block dtype | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| **fp8-e4m3** | ✅ | ✅ (coop) | ✅ stream+dequant¹ | ✅ stream+dequant¹ |
| **int8-convrot** | ✅ | ✅ | ✅ | ✅ (+f16 MLP²) |
| **int8-tensorwise (no convrot)** | ✅ | ✅ | ✅ | ✅ (+f16 MLP²) |
| **int4-convrot** | ✅ | ✅ (decodes to int8: W4A8, `i4_i8`) | ✅ W4A4 | ✅ W4A4 |
| **w4a8** | ✅ | ✅ | ✅ | ✅ |
| **int8 / int4 / w4a8, weight-only** | ❌ | ✅ (`opMatmulCoopIntWo`) | ✅ →bf16 / →f16 / →f32 | ✅ →bf16 / →f16 / →f32 |
| **nvfp4** | ✅ | ✅ | ✅ | ✅ |
| **svdq4** (Nunchaku int4 / fp4) | ✅ | ✅ W8A8 + weight-only | ✅ W8A8 + weight-only | ✅ ditto |
| **bf16 dense** | ✅ | ✅ native/f16 | ✅ native/f16 | ✅ cuBLASLt `R_16BF` |
| **f32** | ✅ | ✅ (offload) | ❌ | ❌ |
| **GGUF q2_k** | ✅ | ✅ →f16 / →bf16³ | ✅ →int8-convrot or →int4-convrot | ✅ ditto |
| **GGUF q4_k / q8_0** | ✅ | ✅ →f16 / →bf16³ | ✅ →int8-convrot or →bf16 | ✅ →int8-convrot or →bf16 |
| **GGUF q4_0/q5_k/q6_k/iq4_nl/iq4_xs** | ✅ | ✅ →f16 / →bf16³ | ✅ →bf16 | ✅ →bf16 |

¹ fp8 weights decode per GEMM to an f16 scratch (`dequant_fp8_f16`) and run the f16
GEMM; there is no fp8 tensor-core GEMM. The CUDA fused `opMatmul` (`first`/`last.linear`)
has no fp8 form, so those two are materialized to f32 at load (`DiT.opMatmulF32`).

² krea2's f16 MLP chain is off in any block whose MLP a LoRA patches.

³ Vulkan's block-quant GEMM (`opMatmulCoopQuant`) dequantizes to f16 and stages the
activation at f16, or both at bf16 (`blockq_bf16`) for a family that needs bf16 range
(`Plan.act16 = .bf16`, Z-Image). f16 is tried first (`Caps.dequantF16First`); a device
without the bf16 coop GEMM refuses a Z-Image GGUF.

**Integer weights.**
- Under `auto` an integer convrot weight runs the format's arithmetic: int8 W8A8, int4
  W4A4 (W4A8 on Vulkan), `asym_w4a8_int8` W4A8. An explicit `--te-act` / `--dit-act`
  runs it weight-only at that width: f16 (`weight_only_f16`), bf16 (`weight_only`), f32
  (`weight_only_f32`, a true-f32 GEMM). `Session.intGemmFor` resolves it; the denoiser's
  choice reaches every plan through `dual_lin.trunk_int8`, the encoder's through
  `dual_lin.encoderInt8`. The saved `Activation dtype` field names the arithmetic.
- Weight-only routes (`i8_wo*`, `i4_wo*`, `w4a8_wo*`) expand the weight per GEMM into a
  scratch with its row scale and rotation undone; a LoRA hit there takes the bf16 A GEMM.
  CUDA needs Ampere or later; Vulkan needs both 16-bit coop GEMMs. bf16 expansion rounds
  the weight to 8 bits; f16 is far closer to the CPU.
- ⚠️ ComfyUI's `int8_tensorwise` has two variants: rotated (`"convrot": true`, `[rows,1]`
  scale) and unrotated (scalar scale). `quant_weight.int8Scale` reads which, and
  `lin.convrot` decides whether the prep rotates; its scan must include `.w4a8`, whose
  weights are rotated, or the render is noise. Unrotated is less accurate; that is the
  format. int4 is refused unrotated on both backends.
- ⚠️ A route that takes a bias reads `Backend.zeroBias` for a bias-free linear, sized to
  the widest such linear in `lin_cuda.presize`; a fixed-size array there reads past its end.
- Bias rides the GEMM epilogue on every CUDA route but fp8, the f16 block-quant decode and
  MMQ, which add it in a separate pass (`lin_cuda.gemmBias`).

**GGUF block quants on CUDA.**
- The block-quant route is the caller's: DiTs pass `--dit-gguf-gemm auto|int8|int4|bf16`
  (`lin_cuda.blockq_gemm`), text encoders `--te-gguf-gemm` (`qwen3_cuda.encoder_gemm`,
  default f16). `int8`, `int4` and `mmq` quantize the activation; `bf16` and `f16` do not.
  Encoders default to f16 because any activation-quantizing route bakes its error into the
  whole render.
- `auto` sends q2_k and q4_k to int8-convrot and every other format to bf16. A weight
  whose width misses the int8 decode's `cols % 1024` floor (Z-Image's 3840) takes bf16.
- The int8 route (`opI8GemmBlockQ`) decodes each packed weight into one scratch, rotates,
  computes per-row scales (cached per weight, `bq_scales`) and runs the vendor int8 GEMM.
  It re-quantizes, so it caps accuracy at int8-convrot's. ⚠️ Do not use it for q8_0: a
  native int8 checkpoint is smaller and equal.
- ⚠️ q2_k is the low-VRAM option, not a quality one; on a 2B trunk it renders a different
  image. `--dit-gguf-gemm int4` is available for it.
- A new block format is `buildPrep`'s `PrepBlock` plus a block walk, and its block must
  divide 256 elements. ⚠️ The decode chunk must divide `cols` (`bqChunk`), or each row's
  tail is silently left undecoded.
- q4_0/q5_k/q6_k/iq4_nl are wired but no render has been made with one.

Gates: `lin-cuda-test [libs]` / `lin-vk-test` (every route, with and without bias,
weight-only f32 exact), `cuda-bqdec-test`, `te-test --gguf-gemm <route>`, `vk-wo-bench`,
`vk-bq-bench`, `diffusion-profile --ab split-bias`.
### 2J. LoRA sidecar support

`--lora <path>[:<strength>]`, repeatable. Every family takes a LoRA as a runtime sidecar,
`y = W x + s B (A x)`: the base weights are never touched, `strength` is a live dial
(`Session.setLoraStrength`), and a stack swaps on a live session (`Session.setLoras`).

| family | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| `sensenova`, `minimax_h3`, `krea2`, `zimage`, `anima`, `mageflow`, `moss_image`, `qwen_image21`, `sd15`, `sdxl` | ✅ | ✅ | ✅ | ✅ |

Adapter forms (one reader, `lora_adapter.parse`; the same apply on every backend):

| adapter | SD (`lora_sd.zig`) | DiT families (`lora.Stack`) |
|---|---|---|
| plain LoRA | ✅ | ✅ |
| LoCon 3x3 | ✅, `A` runs as a conv | refused (adapter on a convolution) |
| Tucker `lora_mid` | ✅, folded into `A` | ✅ |
| LoHa | ✅, CP LoHa too | ✅ |
| LoKr (`lora.Kron`) | ✅ linear and 3x3, no weight-sized delta | ✅ |
| whole diff (`lora.Diff`) | ✅ | ✅ |
| DoRA (`lora.Dora`), output or input axis | ✅ | ✅ |
| refused by name | `set_weight`, Tucker LoKr, DoRA over LoKr, DoRA on a row block of a fused tensor | DoRA over LoKr, `reshape_weight` |

Where the apply runs:

| piece | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| linear sidecar | each model's host funnel (`mm` / `linear`) | inside `lin_gpu.gemm`, two 16-bit coop GEMMs, B accumulating onto the output | inside `lin_cuda.gemm` | same, B folded into cuBLASLt's epilogue | `lora.zig`, `lora_gpu.zig`, `lora_cuda.zig` |
| f32 / f16 / bf16 factors (`--lora-math op16`) | ✅ any | ✅ converted once to the coop operand | ✅ `weight16` | ✅ `weight16` | |
| true-f32 factor GEMMs (`--lora-math f32`) | always | ✅ f32 GEMM + `add_scaled` | falls back to `op16` | ✅ `COMPUTE_32F`, beta into an f32 output | |
| patch embeds, output heads | ✅ | ✅ (`lin_gpu.linearAt`) | ✅ offset views | ✅ | |
| SD 3x3 conv `A` | ✅ | conv (`sd_unet_gpu.convLora`) | ✅ | ✅ | `lora_sd.zig` |
| int8 `A` on an int8-prep route (int8, W4A8, block-quant int8) | | | ✅ `opI8Gemm` | ✅ | `lin_cuda.quantizeLora` |
| SD f16 activation chain | | | | ✅ `opGemmHalfIn` / `opGemmHalfAcc` | |

- **A factor keeps its file's dtype** (f32, f16, bf16; fp8 and f64 widen to f32 at load),
  and `lora.device_math` picks the device arithmetic. `op16` (default) converts it once,
  on the device, to the GEMM's 16-bit operand; `f32` keeps it and the activation exact.
  Measured on MOSS-Image's f32 DMD LoRA at 1024² (int8 trunk, 3090): `f32` costs +32% a
  step over no LoRA against `op16`'s +19% on cuda (+64% against +21% on vulkan), and moves
  the image less than the control, the same arm at strength 1.003 (39.0 dB against 35.4 dB
  on cuda, 37.2 against 35.4 on vulkan). Per linear (`lora-cuda-test`): `op16` is 2-3e-3
  relative on an f32 activation (bf16 operands), 4e-4 on an f16 one, `f32` 1e-6.
  Unmeasured: a dense bf16 trunk, whose own noise floor is lower than W8A8's.
- **`op16`'s operand width is `lora.operandFor`**, the same on both backends: f16 where
  the GEMM's 16-bit side (A's input, B's output) already is f16, bf16 elsewhere, since an
  f32 activation can pass 65504. Forced f16 on MOSS's f32 activations: 2.9e-4 per linear
  against bf16's 2.3e-3, the same speed (±1%), and no difference past the control end to
  end (37.8 / 38.9 dB on cuda, 37.9 / 37.2 on vulkan, both below the 35.4 dB nudge).
  cuda keeps a 16-bit output's own width for B: cuBLASLt has no bf16 GEMM writing f16.
- **Key spellings are the family's `lora.KeyMap`**: ComfyUI's `diffusion_model.` always,
  plus its trainers' prefixes; a bundled checkpoint's `model.diffusion_model.` also matches
  (`lora.baseName`). Qwen-Image 2.1 takes diffusers' `transformer.` and bare names, and its
  fused FFN-half keys (`gate_layer`, `proj`), which ComfyUI drops. The catalog offers a file
  by the same list (`loraFits`, `model_spec.loraSpec`) plus a block-depth bound.
- **SD** keeps three stacks (UNet, CLIP-L, CLIP-G). Keys: LDM, diffusers, `lycoris_*`
  (`lora_sd.unetToDiffusers`), `lora_te`/`lora_te1` to CLIP-L, `lora_te2` to CLIP-G,
  including row blocks of the fused `in_proj_weight`.
- **Z-Image** takes diffusers' split `to_q`/`to_k`/`to_v`/`to_out.0` as row blocks of the
  fused `attention.qkv` (`zimage.lora_keys`); a view across two factors is refused at
  attach. Krea2's diffusers spelling (`transformer_blocks.N.attn.to_q`) is not loaded.
- **Coverage**: a file that patches nothing is refused; one that patches part applies that
  part and warns per unmatched target plus `ONLY n of m`. A family not in
  `Family.supportsLora` refuses `--lora`.
- **CUDA factors are bf16.** Any rank runs: `Sidecar.load` zero-pads it to a multiple of 32
  (`lora.rank_align`, exact), so a low-rank file resides larger than it is (`lora-info`
  prints both). `lora_cuda.supported` needs A's input width on 32 and B's output on 128;
  a factor of another dtype is not padded and stays host-only.
- **int8 `A`** is quantized in the prep's basis, rotated exactly when the prep rotates
  (`lora.quantizeA`), and the bf16 `A` is then not uploaded. MiniMax H3 keeps the bf16 `A`
  (`lora_cuda.applyRange`). `--lora-bf16-a` forces it for comparison.
- ⚠️ In a stack, a DoRA's norms read the base weight, where ComfyUI's merge reads it with
  the earlier files merged in.
- ⚠️ A whole-render device-vs-host PSNR is no reading on the sidecar (a turbo trajectory
  amplifies the arm's own divergence); check the apply with `--layers N`.
  `lora_sd.merge` is the ComfyUI-exact merge, kept as the oracle; it renders nothing.

Gates: `lora-cuda-test`, `lora-vk-test`, `lora-restack-test [--backend]`, `lora-merge-check`
(`tools/check_lora_merge_real.py`), `velocity-diff --lora`, `cuda-dit-test` /
`zimage-cuda-test` / `anima-cuda-test` / `mageflow-cuda-test` / `sensenova-cuda-test`
`--lora <file> [--lora-bf16-a] [--strength s]`; timing: `lora-sidecar-bench`, `lora-cuda-bench`.

---

### 2K. Conditioning edits (`--cond-noise`, `--cond-steer`)

Two operators on the text conditioning, applied inside `Session.encode` on the host
`Cond.data`, so they reach every family and backend with no device code. Applied any
later, they reach only the CPU forward.

`Session.condLayout` describes each family's buffer; `segs` is the depth axis:

| family | rows | segments |
|---|---|---|
| krea2 | tokens | 12 encoder taps |
| zimage / mageflow / h3 | tokens | 1 |
| anima | adapter rows (pad rows included; they are attended) | 1 |
| sd15 | tokens | 1 |
| sdxl | tokens | 2, CLIP-L then CLIP-G, unequal widths |
| sensenova | tokens | 2 per layer (K, V) |

**Noise** adds a seeded perturbation shaped by a `core/noise_curve.zig` curve over `t`
(segment depth), `l`/`n` (token index and count) and `a` (`--cond-noise-amount`), scaled
by each segment's RMS. It is drawn once before shaping, so an amplitude sweep is a ray.
Amount 0 and an empty curve are bit-identical to off. `--cond-noise-shape shared` gives a
token's taps one draw; `--cond-noise-keep-norm` holds each vector's length. Sigma caps at
`Session.max_cond_sigma` = 8.

**Steering** moves the conditioning along a direction named by a phrase; `--cond-steer` is
repeatable (the studio's Steer rows), `--cond-steer-mode` picks how:

| mode | direction | effect |
|---|---|---|
| `concat` | none: APPENDS the phrase's token rows | adds an object, prompt untouched |
| `prompt` | `steer - <this conditioning>` | moves at the phrase, gives up the prompt |
| `empty` | `steer - encode("")` | adds the phrase as an attribute |
| `orthogonal` | `steer` with the prompt's direction projected out | as `empty`, subject survives further |

- ⚠️ SenseNova alone is laid out `[seg][rows][w]`; `CondLayout.segs_outer` keeps a walk
  from transposing depth onto tokens, which renders plausible and wrong.
- ⚠️ `scale` multiplies the ROW NORM, not the per-element RMS. 0.1-0.3 nudges, ~1 replaces
  the prompt. For `concat` it multiplies the appended rows, 1 being the phrase as encoded.
- `concat` grows `Cond.seq`, so `CondNoise.concatSupported` refuses it on zimage, anima
  and sensenova.
- Directions are per segment. Terms add, so overlapping terms of opposite sign cancel; the
  engine logs an overlap above 0.25. Only the positive branch is steered, so CFG amplifies
  the move; a negative-scale term stands in for a negative prompt at cfg 1.
- Both are per image on `wire.RenderParams` (fixed buffers `wire.CurveBuf`,
  `wire.SteerSpec`), so a render the chat model asks for takes neither. ⚠️ `applyTo`
  points `opts.cond_noise.curve` and `.steers` into the params; the `Options` must not
  outlive them.
- Recorded in the PNG block as `Cond noise` and `Cond steer`, both quoted.

Gates: `zig build test` (SenseNova's layout offsets pinned against
`sensenova.Prefix.key`/`.value`).

## 3. Diffusion ops and kernels

| Op | cpu | vulkan | zig-cuda | cuda | Formats |
|---|---|---|---|---|---|
| **GEMM / linear** | `ops/matmul.zig` | `opMatmul`/`opMatmulCoop*` | hand hgemm/igemm/i4gemm | cuBLASLt | see §7 |
| **Conv2d** (patch gather + GEMM, fused 2× upsample) | ✅ f32 | ✅ f16-TC (co≥96) / f32 | ✅ | ✅ (cuDNN NHWC conv for stride-1 3×3; gather for stride-2 and upsample) | f32 weights; f16 TC wide-co; f16 or bf16 activation storage either side |
| **Attention** (DiT GQA 48/12, hd128) | f32 | f16 two-pass TC (+flash), `models/attn_gpu.zig` | `opAttnTC` | f16 TC | f16 scores |
| **VAE mid-block attn** | f32 | f32 scores plane, query-banded | TC flash, f32 scores band | cuDNN SDPA | see below |
| RMSNorm + AdaLN modulate | ✅ | `rms_apply_mod`/`modulate` or `rms_mod` (dual) | `rms_mod` (dual) | ↤ | f32 (+h16) |
| Weighted RMSNorm (Q/K, sandwich) | ✅ | `rmsnorm` (dual, subgroup per row) | `rmsnorm` (dual); hand `qk_rmsnorm_par` for <64 rows at ≥1024 wide | ↤ | f32/f16 |
| Weightless LayerNorm + modulate | ✅ | `ln_mod` (dual) | ↤ | ↤ | f32 |
| RoPE (3-axis interleaved) | ✅ | `rope_inter` (dual) | ↤ | ↤ | f32 |
| SwiGLU / silu-mul | ✅ | `silu_mul{,16,_h16}` (dual) | ↤ | ↤ | f32/f16 |
| sigmoid-gated add | ✅ | `sigmoid_mul` (dual) | ↤ | ↤ | f32 |
| gated residual add | ✅ | `gated_add{,16}` (dual) | ↤ | ↤ | f32/f16 |
| relu / add_relu | ✅ | `relu` / `add_relu` | `relu` / `add`+`relu` | ↤ | f32 |
| GroupNorm (Welford) | ✅ | `gn_stats`/`gn_combine`/`gn_apply` | ↤ same names | ↤ | f32/h16 |
| im2col | ✅ | `im2col`, `im2col_sd` | ↤ | ↤ | f32/h16 |
| 3x3 patch into 16-bit operand | | `im2col_h16`/`im2col_bf16` (`opConvCoop`) | `opConvGemm16` | ↤ | f16/bf16 |
| nearest-2× upsample | explicit | fused into im2col | fused into im2col | ↤ | — |
| convrot (Hadamard un-rotate) | `ops/convrot.zig` | `rotate`/`rotate_fwht` | `buildPrep` | ↤ | int8/int4, group 256 |
| 4-bit decode | `ops/{w4a8,nvfp4,svdq}.zig` | `w4a8_decode_t`/`nvfp4_decode_t`/`i4_decode_t`/`svdq_decode_t`/`svdq_unpack*` | `w4a8_decode`/`nvfp4_decode` PTX, `svdq_decode`/`svdq_unpack*` dual | ↤ | per GEMM into scratch |

`↤` = `cuda` reuses zig-cuda's hand-PTX kernel (only batched GEMM and prefill attention
differ).

Every diffusion family's Vulkan tensor-core attention is `models/attn_gpu.zig` (`core`,
`stage`, `run` / `attn_full`); the SD VAE's single wide head (`opAttnScoresVae`) is its own
kernel. A dense weight on the f16 plan (`h16_*`) reads and writes a 16-bit activation as is
on every arm (`opGemm16`, `opMatmulCoopF16W`); below `h16_min_co` output rows only an
all-f32 linear takes the f32 GEMM, and only the int8 preps widen a 16-bit activation off the
vendor arm. A device buffer carries its dtype (`DeviceBuffer.dt`), and conv, GroupNorm, add,
im2col, pad, `bias_compact` and `bias_add` pick their `dual.zig` instance from it.

VAE activation storage (`sd_vae.Config.act_f16`, overridden by `--vae-act f32|f16|bf16` on
Vulkan and both CUDA arms):

| VAE | default | notes |
|---|---|---|
| `flux`, `sd15` | f16 | |
| `sdxl` | f32 | f16 goes non-finite; bf16 works |
| Wan 2.1 (krea2, Anima) | f16 | same adapters (`wan_vae_dev.zig`), own `vae_norm`; Vulkan attention on the TC path with f16 scores (`Dev.attnCoop`) |

- Norms are subgroup/warp per row, never thread per row. ⚠️ The row sum is a tree, so it is
  not bit-identical to a serial norm.
- The SD/Flux VAE mid-block logits exceed f16's range, so its scores stay f32 on every arm.
  ⚠️ `sd_vae.scoresBand` is shared by the decoder and `estimatePeakBytes`; keep it one constant.
- ⚠️ `attn_out` runs its OWN online softmax; a softmax pass between it and `attn_scores`
  exponentiates twice and renders plausible and wrong.
- ⚠️ `opMatmul`'s `y_off` / `x_off` are BYTES on Vulkan and elements on CUDA.
- ⚠️ `sd_unet_gpu.groupNormInto` uses its own chunk count (256) and the caller sizes `gstat`;
  the CUDA arm's 32 under-allocates it.
- ⚠️ `ctx.independent(n)` counts DISPATCHES, and a coop GEMM is three sharing scratch. Read
  only under `full_barriers`.
- ⚠️ 16-bit VAE storage and `residual_act_div` are alternatives (`conv2d` asserts); bf16 keeps
  the divisor. 16-bit storage forces the coop conv path, since the f32 arm cannot write 16-bit.
  Every GPU decode tier is finite-checked (`pipeline.finiteDecode`); a non-finite one goes to
  the CPU tier.
- The 16-bit 3x3 patch gather must be bit-identical to the f32 patch;
  `dual_table.force_conv_patch` switches it off.
- ⚠️ A cached CUDA weight is never evicted while an op reads it: eviction skips entries
  fetched since the last launch and those under `Backend.hold`.

Gates: `lin-cuda-test`, `lin-vk-test`, the gated `sd_unet_gpu` test, `wan-vae-test [side]`,
`diffusion-profile --ab convpatch`; timing: `vk-norm-bench`.

---
## 4. LLM models (text) — `tp-llm` (`src/llm_main.zig`, `llm/session.zig`)

The LLM architectures `tp-llm` and tp-gui chat run, per backend.

| Model | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **qwen3** (Qwen3 / Qwen3-VL / llama-arch Mistral-Nemo) | ✅ | ✅¹ | ✅ | ✅ | `qwen3{,_gpu,_cuda}.zig` |
| **qwen35** (hybrid DeltaNet: GDN + attn) | ✅ | ✅ | ✅ | ✅ | `qwen35{,_dev,_gpu,_cuda}.zig` |
| **k2-horizon** (MoE + MoVA) | ✅ | ✅ | ✅ | ✅ | `k2_horizon{,_dev,_gpu,_cuda}.zig` |
| **gemma3** (sandwich norms, dual RoPE) | ✅ | ✅ | ✅ | ✅ | `gemma3{,_gpu,_cuda}.zig` |
| **gemma4** (per-layer geometry, factored RoPE) | ✅ | ✅ | ✅ | ✅ | `gemma4{,_gpu,_cuda}.zig` |

- **qwen3** also runs the generalized `llama`/Mistral arch (un-permuted q/k at load,
  optional QK-norm, untied head, runtime vocab/eps); GGUF config-detect up to 64 layers.
  It is the only arch with speculative decoding. On Vulkan it is one batched forward at any
  position in 2048-row chunks, attention against the cache for every KV dtype.
- **qwen35** has no spec decode (recurrent state). Its device forward is `qwen35_dev.zig`
  over each backend's `Dev`; Vulkan prefills in 512-token batches like CUDA. Vulkan runs
  the dual attention and chunked DeltaNet kernels (`dual/attn.zig`); CUDA keeps its own
  grouped attention.
- **k2-horizon**: grouped RMSNorm, sigmoid routing, shared experts, attention gates, MoVA
  value experts, reasoning effort. Both GPU backends pin the fixed path and as many
  routed-expert groups as fit; the rest are host layers (`migrateLayer`/`promoteLayer`,
  `autoOffload`, `offloadToBudget`; KV growth migrates layers, never streams). A host
  layer runs a batch on the CPU or stages its groups to the GPU per prefill, chosen once
  per process by `k2.HostCost` (`TP_K2_HOST_ROWS=N` pins it). On Vulkan a resident block of
  up to 64 rows (`device_route_max`) routes on the device (`moe_route`, `gemvx_dual_<fmt>`,
  `moe_combine`); larger batches route on the host (`VulkanLM.device_route_off` forces
  host). CUDA routed GEMMs: grouped q8 GEMV up to 256 rows (q6_k only), else
  `hgemm_experts` (`TP_K2_EXPERT_GEMM=mmq|gemv` selects). Prompts use the model's own
  template; tools go through it and `<ifm|tool_call>` is parsed by `llm/tool_call.zig`.
- **gemma3** is GGUF block-quant only. Vulkan runs 256-row chunks, the local layers' KV
  ring sized window plus one chunk, and an image prefilled as one batch.
- **gemma4** config is metadata-driven (12B Q4_0 48L, 31B mixed Q4_K/Q6_K 60L load with no
  code change). Vision via `gemma4uv` / `gemma4v` towers (§5).
- ⚠️ `engine.prefill_gate_chunk` and a stepper's `prefill_chunk` change together or not at
  all: the engine slices prefill at the gate chunk, so a larger stepper chunk is dead code.
  A stepper declares its preferred size as `prefill_batch` (`engine.prefillBatchOf`). Pause
  parks only at batch boundaries; cancel unwinds between layers (`engine.publishCancel`).

¹ **qwen3 on vulkan** (`qwen3_gpu.zig VulkanLM`): any weight `lin_gpu` takes, dense or GGUF
block-quant, tied or untied head (a GEMV over a weight past the 1 GiB binding runs as row
slices); bf16 read natively. A batch up to `gemv_batch_max` rows (spec verify) is
bit-identical per row to decode. ⚠️ Every dense weight is cached twice (GEMV and coop
layouts): a bf16 4B holds 16 GB. QK-norm is skipped where the arch has none.

Gates: `tp-llm --bench` (logits hash per backend), `lin-vk-test`, `lin-cuda-test`.

### LLM linear dispatch (`models/lin_cuda.zig` and `models/lin_gpu.zig` under `Plan.llm`)

One dispatcher per backend for every LLM stepper. Each `Model` exposes `deviceLins` and the
stepper `plan`s it at init, so a format with no kernel is refused by tensor name at load. A
stepper `prep`s an activation once per group of linears sharing it (q/k/v, gate/up) and
`gemm`s each; CUDA picks the route per weight from storage, shape and row count `m`:

| weight | `m = 1` | `1 < m <= 40` | `m > 40` |
|---|---|---|---|
| q5_k, q6_k, q1_0, q2_0 | `gemv_q*_q8` dp4a | `gemv_q*_q8n` grouped (q1_0/q2_0: one batched launch) | MMQ pipe where `mmqPipeFaster`, else dequant-f16 GEMM |
| q4_0, q8_0, q4_k | `gemv_q*_q8n` (ng = 1) dp4a | grouped | same |
| iq4_xs | `gemv_iq4_xs_q8n` (ng = 1) dp4a | grouped | MMQ pipe |
| iq4_nl | `gemv_iq4_nl` f32 activation | dequant-f16 GEMM | same |
| q2_k, q3_k, iq2_xxs, iq2_xs, iq3_xxs, iq3_s | `gemv_dual_*_q8` | dequant-f16 GEMM | same |
| any block quant, `rows % 128 != 0` or `cols % 256 != 0` | f32 GEMV | grouped if tileable, else per-row f32 GEMV | same |
| bf16 | `gemv_bf16` | `gemv_bf16n`, 4 rows per launch | `opGemmBf16` (Ampere+) / f16 GEMM |
| fp8 | `gemv_fp8` | `gemv_fp8n` | `opMatmulFp8` |
| f16 / f32 | f16 GEMV / f32 GEMM | GEMM | GEMM |

- The dequant-f16, bf16 and fp8 GEMMs take any row count (`hgemmPadN` pads a weight off the
  128-wide N tile). dp4a and MMQ read the q8_1 activation the prep staged; the rest read f32.
- Overrides: `lin_cuda.decode_dp4a = false` sends every block quant through the f32 GEMV;
  `tp-llm --llm-gemm grouped|mmq|dequant` forces the batched route (a weight that cannot
  take it keeps its automatic one).
- **On Vulkan a block quant has one route**: decode is the dual row GEMV over the raw weight
  (`opGemvQuantDual`), a batch the coop dequant GEMM (`opMatmulCoopQuant`), both reading the
  same resident copy. A row that is not whole blocks goes to the GEMM.
- **Vulkan `OpSDot` decode** where the device has `VK_KHR_shader_integer_dot_product`
  (`Context.has_int_dot`, `lin_gpu.Caps.dualQ8`): `gemv_dual_<fmt>_sdot` at one row,
  `gemv4_dual_<fmt>_sdot` for a 2-4 row spec verify, each row bit-identical to decode.
  ⚠️ That match needs `NoContraction` on every float op of both cut modules
  (`spv.withNoContraction`) and one-segment-per-row q8_1 staging; without it two kernel
  shapes agree only by luck. `TP_VK_NO_SDOT` and the `--bench` toggle put the f32 GEMV back.
- ⚠️ **A batched Vulkan dispatch waits only for what it touches** (`gpu/hazard.zig`): a
  barrier is recorded only on a real read/write hazard since the last one, from each
  module's SPIR-V write set. Two offsets of one buffer always conflict (a binding's extent
  is unknown). `TP_VK_FULLBAR`, the `--bench` toggle and `diffusion-profile --ab fullbar`
  put a barrier after every dispatch and must give identical bytes; `TP_VK_NOBAR` gives
  wrong answers. Sync validation is clean on AMD; the NVIDIA-only coop paths are unvalidated.

Gates: `lin-cuda-test`, `lin-vk-test`, `gpu dual gemv covers the codebook and low-bit
k-quant formats` (q4_0 also via the gemma4 Vulkan stepper tests), `dual-gemv-bench`,
`vk-gemv-bench`.

### LLM speed against llama.cpp

RTX 3090, warm (weights uploaded before timing), greedy. ⚠️ The 3090 drifts several percent
between sessions; only a same-session ratio against llama.cpp is meaningful, so compare rows
within one line, not across rows.

| card | model | quant | backend | prefill tok/s | decode tok/s | llama.cpp same | note |
|---|---|---|---|---|---|---|---|
| 3090 | Bonsai-27B | q1_0 | zig-cuda | 1097 | 75.8 | pp 1276, tg 58.3 | |
| 3090 | Bonsai-27B | q2_0 g128 | zig-cuda | — | — | ours 0.88x pp, 0.92x tg | ratio only |
| 3090 | Mistral-Nemo | IQ4_NL | vulkan | — | — | llama.cpp Vulkan ~3.5x faster tg | ratio only |
| 3090 | Qwen3-4B | q4_k | vulkan | 1951 | 63 | — | 1309-token prompt |
| 3090 | Qwen3-4B | bf16 | vulkan | 2344 | 45 | — | 1309-token prompt |
| 3090 | Qwen3.5-9B | q6_k | vulkan | 1074 | 37.5 | — | 1313-token prompt |
| 3090 | Qwen3.5-9B | q6_k | cuda | 1550 | 84 | — | 1313-token prompt |
| 3090 | Gemma 3 12B | Q4_K_M | vulkan | — | 23.6 | — | 1310-token prompt |
| 3090 | Gemma 3 12B | Q4_K_M | cuda | 763 | 53.1 | — | pp at 917 tokens, tg at 1310 |
| 3090 | Gemma 4 31B | q4_k | cuda | ~270 | ~25 | — | 1824-token prompt |
| 3090 | K2 Horizon | Q6_K | cuda | 106 / 223 / 406 | 32.4 | — | pp at 19 / 242 / 489 tokens; 16 MoE layers on host |
| 3090 | K2 Horizon | Q6_K | vulkan | 78 / 112 / 181 | 27.9 | — | same; 16 MoE layers on host |

- ⚠️ `Backend.cachedWeight` uploads lazily, so a one-shot run charges the upload to prefill.
  A stepper's `warmWeights` (called from `session.run`) uploads first; every `*_cuda.zig`
  stepper and the gemma3, gemma4, qwen3, qwen35 and k2-horizon Vulkan ones have it. A
  prefill rate that rises with prompt length means a stepper lacks it.
- ⚠️ A process's first prefill also page-faults mmapped host layers; time with `--bench`.

---

## 5. LLM vision towers

The image encoders behind multimodal chat, each device forward written once over a
per-backend adapter.

| Tower | cpu | vulkan | zig-cuda | cuda | Files | Vision dtypes |
|---|---|---|---|---|---|---|
| **Qwen3-VL `vit35`** (SigLIP, 2-D RoPE) | ✅ | ✅ | ✅ | ✅ | `vit35{,_dev,_gpu,_cuda}.zig` | bf16 blocks/proj, f32 patch |
| **Gemma 3 `gemma_vit`** (SigLIP-So400m) | ✅ | ✅ | ✅ | ✅ | `gemma_vit{,_dev,_gpu,_cuda}.zig` | f16 blocks (vulkan: →f32 at load) |
| **Gemma 4 `gemma4_vit`** (shallow `gemma4uv` embedder, 12B) | ✅ | ✅ | ✅ | ✅ | `gemma4_vit{,_dev,_gpu,_cuda}.zig` | f32 patch, bf16 proj |
| **Gemma 4 `gemma4v_vit`** (full SigLIP tower, 31B; EmbeddingGemma 2's) | ✅ | ✅ | ✅ | ✅ | `gemma4v_vit{,_dev,_gpu,_cuda}.zig` | Q8_0/F16 blocks (GGUF), bf16 (safetensors), f32 patch |

- tp-gui image chat runs on Vulkan for qwen35, gemma3 and gemma4. tp-llm's interactive
  `@image` is CUDA-only; a one-shot `--image` takes the Vulkan tower.
- A Vulkan tower leaves nothing resident after an encode (`tower_gpu.Scope`).
- **`gemma4v`** (`projector_type "gemma4v"`): 27-block SigLIP, per-head QK-RMSNorm, 2-D neox
  RoPE θ=100, weightless V-norm, `kq_scale = 1.0`, GeGLU FFN, RMS sandwich norms, 3×3
  avg-pool merge, `std_bias`/`std_scale` affine, one `mm.input_projection`. Blocks run on
  the device, the projection on the host. ⚠️ Its GPU output is looser against the f32 CPU
  tower than gemma3's: the f16 GEMM rounding compounds over 27 blocks (measured per depth in
  `gemma4v_vit_cuda.zig`'s test). EmbeddingGemma 2's 16-block tower agrees to 0.999997.
- The same tower loads from a transformers `Gemma4VisionModel` (`Vit.loadSafetensors`,
  EmbeddingGemma 2), which differs in three settings (`Config`): tanh GELU where the GGUF
  path runs llama.cpp's quick GELU, torchvision's antialiased bicubic resize
  (`image.resizeAntialiasBicubic`) where the GGUF path runs an align-corners bilinear, and
  no std affine. ⚠️ On EmbeddingGemma 2's tower, quick GELU in place of tanh moves the worst
  soft token to cosine 0.9934 against transformers; the GGUF 31B path is unmeasured
  against transformers.
- **Preprocess follows `gemma4_vision_token_budget`**: aspect-preserving resize to a
  48-aligned grid sized to a post-merge token budget, then `2·p/255 − 1`. Set by
  `--vision-budget low|medium|high|ultra|max|<tokens>` or tp-gui "Vision detail" (presets
  70/140/280/560/1120, default `high`). ⚠️ The budget also sizes the LLM's image-prefill
  scratch and KV-ring slack, so it is fixed at load (reload-gated in the GUI); `ultra`/`max`
  on the 31B at 24 GB need `--kv-dtype q8_0` and/or a smaller `--max-context`.
- `TP_VIT_DUMP=<path.png>` writes the pixels the tower ingests.

Gates: `vit35-test`, `gemma-vit-test`, `gemma4-vit-test` (every arm against the CPU, a hash
per arm).

### 5A. Embedding encoders (`tp.embed`, `src/embed.zig`)

Encoder-only towers ending in one L2-normalized 768-d vector, each device forward written
once over `tower_cuda.Dev` / `tower_gpu.Dev`. Linears run in the checkpoint's dtype; the
pooling heads run on the host.

| Encoder | cpu | vulkan | zig-cuda | cuda | Files | Reference |
|---|---|---|---|---|---|---|
| Snowflake Arctic Embed M v2.0 | ✅ | ✅ | ✅ | ✅ | `embed_snowflake{,_dev,_gpu,_cuda}.zig` | ONNX |
| EmbeddingGemma 300M | ✅ | ✅ | ✅ | ✅ | `embed_gemma{,_dev,_gpu,_cuda}.zig` | ONNX |
| **EmbeddingGemma 2**, text | ✅ | ✅ | ✅ | ✅ | `embed_gemma2{,_dev,_gpu,_cuda}.zig` | sentence-transformers (`tools/gen_embeddinggemma2.py`) |
| **EmbeddingGemma 2**, image | ✅ | ✅ | ✅ | ✅ | + `gemma4v_vit*` | (`tools/gen_embeddinggemma2_image.py`) |
| SigLIP2 ViT-B-16 text / visual | ✅ | ✅ | ✅ | ✅ | `embed_siglip{,_dev,_gpu,_cuda}.zig` | ONNX |

- Attention: Snowflake, EmbeddingGemma 300M and SigLIP2 run the one-thread-per-row
  `attn_batched` kernel (heads up to 512). EmbeddingGemma 2 runs `attn_bounded` (`dual/`,
  both arms): one subgroup per (row, head, key chunk) over the row's `[start | end)` from a
  table at the scratch's tail, then `attn_merge`. The table carries both the item bounds
  of a ragged batch and the bidirectional sliding window (`Ragged.boundsInto`). On a
  284-row image body it is 0.42 ms a layer against `attn_batched`'s 24.5.
- The Gemma 4 vision tower's Vulkan attention runs `attn_gpu`'s tensor-core path with
  heads zero-padded to 128 (`opAttnBounded` without tensor cores), like CUDA's: 1.7 s ->
  84 ms on a 2520-patch image. On the 31B gate it moves Vulkan's min token cosine from
  0.9955 (f32) to 0.9926, CUDA's arms read 0.9936 / 0.9954.
- CPU: a plain non-causal attention of 64k+ scores runs per head as two GEMMs around a
  threaded row softmax (`ops/attention.zig` `attentionGemm`): EmbeddingGemma 2's CPU
  image tower 5.9 s -> 1.45 s.
- A GPU encoder's weights stay resident; `deinit` evicts them (`tower.forEachHostBuffer`),
  because both device caches key on host addresses the model then frees.
- EmbeddingGemma 2's audio tower and video input are not implemented.

Gates: the `embed*` tests under `-Dintegration` (CPU against the reference, each device
forward against the CPU, batched against per-item); `zig build embed-bench` times them.

---

## 6. LLM ops, decode features and residency

| Op / feature | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| Attention prefill | `ops/attention.zig` | `attn_full` + coopmat | hgemm+softmax TC, `attn_split_g` | **cuDNN SDPA** |
| Attention decode (flash) | (same fn) | `attn_rows{,_g2,_g4,_g8}_*` + `attn_merge` (dual) | `attn_split`/`_merge`/`_h256`/`_h512` | ↤ hand-PTX |
| GQA / windowed (Gemma) local attn | ✅ | ✅ | ✅ | ✅ |
| Bidirectional image-block attn | ✅ | ✅ (gemma3/gemma4) | ✅ | ✅ |
| KV cache | host slices | device + gather | `opKvAppendS` + gather/scatter | ↤ |
| **Growable KV context** | ⚠️ host arrays | ✅ qwen3/qwen35/gemma3/gemma4 (`kv_gpu.Kv.grow`, copy into a bigger buffer; rings fixed, gemma4's regrown once on the first image) | ✅ VMM in place | ✅ |
| RoPE (half/partial/interleaved/dual/factored) | ✅ | ✅ (+M-RoPE; vision as a per-token `rope_half` table) | ✅ (+M-RoPE, vision) | ↤ |
| RMSNorm / LayerNorm / sandwich | ✅ | `rmsnorm`/`layernorm` | `qk_rmsnorm`/`ln_bias_par` | ↤ |
| **Decode GEMV (int8 dot)** | ggml `vec_dot` | ✅ every block quant where the device has `OpSDot` (`gemv_dual_*_sdot`, `gemv4_dual_*_sdot`, experts `gemvx_dual_*_sdot`) | ✅ every arch via `lin_cuda` (`gemv_q*_q8`, `_q8n`, `gemv_dual_*_q8`) | ↤ |
| Prefill GEMM | `matmul.zig` microkernel | coopmat bf16/f16 + int8 s8→s32 | hand hgemm/igemm/i4gemm, MMQ pipes | **cuBLASLt** |
| **GDN / gated DeltaNet** (qwen35) | ✅ | `gdn_gates`/`gdn_conv_step`/`gdn_delta_step` + dual chunk | + batched `gdn_conv_batch`, `opGdnDeltaChunk` | ↤ |
| **GDN grouped v-heads** (HF checkpoint, `Config.v_group`) | ✅ | ✅ | ✅ | ↤ |
| Embedding gather | model | ⚠️ host-side | on-device `opEmbedGather*` | ↤ |
| **Sampling** (argmax/temp/top-k/top-p/min-p + penalties) | ✅ `llm/sample.zig` | ✅ argmax/top-k select (qwen3/qwen35/gemma3/gemma4/k2) | ✅ (qwen3/qwen35/gemma3/gemma4) | ↤ |
| **Turn-boundary checkpoint / rollback** | ❌ | ✅ qwen3/qwen35/gemma3/gemma4/k2 | ✅ qwen3/qwen35/gemma3/gemma4 | ↤ |
| **Weight noise** (`--weight-noise`) | ❌ | ✅ qwen3/qwen35/gemma3/gemma4, every block quant | ✅ qwen3/qwen35/gemma3/gemma4, block-quant GEMVs + MMQ pipes | ↤ |
| **CPU layer split / dynamic offload** (`residency.zig`) | — | ✅ qwen3/qwen35/gemma3/gemma4/k2 | ✅ qwen3/qwen35/gemma3/gemma4/k2 | ↤ |
| **tp-gui chat** (`engine/chat.zig`) | ❌ | ✅ every arch | ✅ every arch | ↤ |

- **GPU sampling is a candidate select.** The device runs argmax or a top-k reduce
  (`stepArgmax`/`stepSelect`), downloads the candidates, and the CPU applies temperature,
  top-p, min-p and the RNG over them, bit-identical to full-vocab CPU sampling. Penalties
  run on-device (`opPenalize`): bit-identical to CPU on CUDA, ~2.5 ULP on Vulkan. gemma4's
  softcap is monotonic, so it is applied on the host to the candidates only
  (`gemma4.finalizeCandidates`).
- **Turn-boundary checkpoints** back tp-gui's regenerate / variant switch:
  `checkpoint(out)` captures the non-append-only state, `restoreCheckpoint(snap, q)`
  truncates back. qwen3 stores nothing (pure `truncate`); qwen35 the DeltaNet conv/ssm state
  and M-RoPE position; gemma3/gemma4 the local layers' KV rings. Residency-aware. A stepper
  without the API falls back to a full re-prefill.

### Weight noise (`--weight-noise`, tp-gui's composer toggle)

Perturbs block-quant weights while generating: each super-block's `d` (and `dmin`) is
multiplied by `1 + sigma*u`, u uniform in [-1,1], redrawn every forward. The hash lives in
`cuda/wnoise.zig` (PTX) and `dual/quant.zig`'s `jitter` (dual kernels, Vulkan).

- **Sigma is a curve over depth**: `core/noise_curve.zig` evaluates an expression in `t` (0
  at the first decoder layer, 1 at the LM head) into `WeightNoise.table`; a bare number is
  flat. The curve may read `a`, set by `--weight-noise-amount` (default 1) or the composer;
  `noise_curve.respondsToAmount` says whether a curve uses it. tp-gui keeps named shapes in
  Settings; the CLI takes the same expression.
- `--weight-noise-seed` makes a run reproducible on one backend; weight ids are first-touch
  order per backend, so a seed does not reproduce across backends. Sigma 0 is bit-identical
  to off.
- **Support is two predicates, use both**: `cuda.Backend.weightNoiseSupported(dt)` (dtype)
  and `llm.session.archSupportsWeightNoise(arch)` (stepper declares `noiseAtLayer`);
  `llm.session.weightNoiseSupported(gguf)` combines them by majority over the block-quant
  weights under `layers.`. ⚠️ `Gguf.names()` returns canonical `layers.N.…`, not `blk.N.…`.
- Steppers: gemma3, gemma4, qwen3 (and llama) and qwen35; not k2-horizon. ⚠️ A stepper that
  captures a decode graph must refuse capture while noise is on (a graph freezes key and
  sigma).
- ⚠️ Not perturbed: `buildMmqQ4K`, the dequant-f16 fallback (so q4_0 and iq4_nl only at
  decode, not in a long prefill), the fp8/bf16/int8-convrot GEMMs, `opGemvQuantQ8Batch`
  (q1_0/q2_0 prefill chunks off MMQ), and the embedding gather (a tied `token_embd` is
  perturbed as the head only). Noise on `attn_k`/`attn_v` persists in the KV cache. Sigma
  tolerance differs by dtype (ternary q2_0 needs far more than q4_k).

| kernel | path | dtypes |
|---|---|---|
| `gemv_{q4_0,q8_0,q4_k,q5_k,q6_k,iq4_xs}_q8n` | decode + grouped prefill | those six |
| `gemv_{q5_k,q6_k,q1_0,q2_0_g64,q2_0_g128}_q8`, `_q8x2` | dp4a decode | those five |
| `gemv_{q4_0,q8_0,q1_0,q2_0_g64,q2_0_g128,iq4_nl,q6_k}` | f32-activation decode, LM head (tied `token_embd`) | those seven |
| `buildMmqPipeQ{4K,6K,8_0,1_0,2_0}` | batched prefill MMQ | q4_k, q5_k, iq4_xs, q6_k, q8_0, q1_0, q2_0 |

Gates: the `gui/chat.zig` test that the arch list and the steppers' `noiseAtLayer`
declarations agree.

### Reasoning markers

The thought/answer delimiters come from the model's own chat template
(`chat.observeReasoning`, which needs both halves), else from the architecture
(`chat.reasoningFor`). `tp-llm --reasoning-markers "<open>,<close>"` overrides both. A family
with selectable effort (K2 Horizon) keeps the family answer. Defaults: qwen35 `<think>`,
gemma4 `<|channel>thought`, gemma3 none.

### Speculative decoding (qwen3 only)

| Feature | cpu | vulkan | zig-cuda | cuda |
|---|---|---|---|---|
| n-gram prompt-lookup (`--spec-k`) | ✅ | ✅ | ✅ | ✅ |
| draft-model (`--draft-model`) | ✅ | ❌ | ✅ | ✅ |
| EAGLE-3 head (`--eagle`) | ❌ | ❌ | ✅ | ✅ |
| tree drafting (`--tree`) | ❌ (verify only) | ❌ | ✅ | ✅ |

- EAGLE tap and tree paths are f32-KV only.

### Weight residency and offload

⚠️ **LLM weights never stream.** Every LLM weight pins device-resident on first touch
(`Backend.pinAllWeights` / Vulkan `pin_budget = maxInt`); a model that outgrows VRAM moves
whole layers to the CPU instead. Per-step weight streaming is diffusion-only.

| Feature | cpu | vulkan | zig-cuda | cuda | Models |
|---|---|---|---|---|---|
| pin-all weight residency | — | ✅ | ✅ | ✅ | all |
| `--cpu-layers` static split | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |
| `--offload-grow` dynamic offload | ❌ | ❌ | ✅ | ✅ | qwen3/qwen35 |
| GUI-driven split (`autoOffload`/`settleTo`/`imageReclaim`) | — | ✅ | ✅ | ✅ | gemma3/gemma4/k2 |
| automatic offload on OOM | — | — | ✅ | ✅ | gemma4 |

- `--vram-budget` only sizes the split planners. A Vulkan model that does not fit even after
  splitting fails with a clean error.
- The split works with every KV dtype: the host shadow (`llm/kv_cache.zig`,
  `PerLayerKvCache` for gemma4) stores the device format byte for byte, so migrate, promote
  and ring checkpoints are lossless copies. A KV dtype toggle survives an armed split.
- **Offload on OOM (gemma4_cuda)**: a `DeviceOutOfMemory` during a prefill or KV growth arms
  a dynamic split (`ensureOffloadArmed`), migrates a few layers and retries the forward.
  Other `*_cuda.zig` steppers can take it via `runtime/residency.zig`.
- **q8_0 KV** (`--kv-dtype q8_0`): ggml `block_q8_0`, 34 bytes per 32 elements, quantized on
  write, dequantized in the attention kernels. Coverage matches f16 (every Vulkan stepper via
  `kv_gpu.Kv`). ⚠️ It rounds ties-to-even on every engine (unlike ggml's `roundf`) so host and
  device bytes match across a split. Like f16 it is lossy; a dtype toggle rebuilds the context.

---
## 7. Data-format support matrix

DType enum (`src/dtype.zig`): `f8_e4m3, f16, bf16, f32, i8, i4, w4a8, nvfp4, svdq4, q4_0, q8_0, q2_k, q3_k, q4_k, q5_k,
q6_k, iq4_nl, iq4_xs, iq2_xxs, iq2_xs, iq3_xxs, iq3_s, q1_0, q2_0_g64, q2_0_g128, w6a8`. `i8`/`i4` are the ComfyUI convrot formats for the **image**
path; GGUF `q*` are the **LLM** path.

| Format | cpu | vulkan | zig-cuda | cuda | How it computes |
|---|---|---|---|---|---|
| **f32** | ✅ | ✅ scalar / VAE→f16 coop | ✅ fallback | ⤷ via f16/bf16 | dtype-aware GEMM, f32 accumulate |
| **f16** | ✅ vectorized Zig | ✅ f16 coopmat | ✅ `buildHgemm` m16n8k16, `gemv_f16` | ✅ cuBLASLt `R_16F` | tensor-core coop / mma |
| **bf16** | ✅ | ✅ native bf16 coopmat + f16 fallback | ✅ bf16 mma (Ampere+) | ✅ cuBLASLt `R_16BF` | native bf16 where the device has it |
| **fp8-e4m3** | ✅ (LUT) | ✅ `pipe_f8` → f16 coop | ✅ `gemv_fp8`, `dequant_fp8_f16` | ✅ dequant→f16 | 1-byte weights, dequant in kernel; ComfyUI's scaled fp8 (rank-0 `weight_scale`) multiplied into every GEMM |
| **int8 (+convrot)** | ✅ | ✅ s8→s32 tensor cores | ✅ m16n8k32 s8 IMMA | ✅ cuBLASLt `R_8I`/`COMPUTE_32I` | W8A8, Hadamard un-rotate at dequant |
| **int4 (+convrot)** | ✅ | ✅ decode→int8 per GEMM | ✅ m16n8k64 s4 IMMA (W4A4) | ✅ hand-PTX (no cuBLASLt s4) | nibble-packed 2/byte |
| **w4a8** (`asym_w4a8_int8`) | ✅ | ✅ | ✅ | ✅ | 4-bit codebook indices + fp8 per-group scale; decodes to int8 per GEMM, then the int8 convrot GEMM (`ops/w4a8.zig`) |
| **w6a8** (`w6a8_int8`) | ✅ | ✅ `w6a8_decode_t` | ✅ `w6a8_decode` | ✅ ditto | 6-bit uniform `(q − 32)·s_rel`, nibble plane then top-2-bit plane, `I8 [rows, 3·cols/4]`; same routes as w4a8 |
| **nvfp4** (E2M1) | ✅ | ✅ | ✅ | ✅ | E2M1 + fp8 per-16 scale + per-tensor scale; decodes to bf16 per GEMM, then the bf16 GEMM. Weight-only (`ops/nvfp4.zig`) |
| **svdq4** (Nunchaku SVDQuant int4 / fp4) | ✅ | ✅ | ✅ | ✅ | 4-bit codes in Nunchaku's MMA tile order (int4, or E2M1 with an fp8 per-16 micro-scale and a per-row `wcscales * wtscale`) + bf16 per-64 scale for int4 + per-input smoothing + rank-32 low-rank branch (`ops/svdq.zig`). W8A8: the activation times `1/smooth`, the unrotated int8 prep, the weight decoded per GEMM to int8 on a per-row grid. Weight-only (bf16/f16/f32): the exact weight expanded per GEMM. The low-rank branch runs on the raw activation through the LoRA sidecar's GEMMs, in true f32 on the f32 route |
| **GGUF q4_0** | ✅ ggml | ✅ `gemv_dual_q4_0` | ✅ `gemv_q4_0(_q8n)` | ⤷ dequant→f16 | |
| **GGUF q8_0** | ✅ ggml | ✅ `gemv_dual_q8_0` | ✅ `gemv_q8_0(_q8n)` | ⤷ dequant→f16 | |
| **GGUF q4_k / q5_k / q6_k** | ✅ ggml | ✅ `gemv_dual_*` | ✅ `gemv_*(_q8/_q8n)`; MMQ `mmq_pipe_q{4,5,6}_k` | ⤷ dequant→f16 | MMQ prefill routed for q4_k/q5_k only (`mmqPipeFaster`) |
| **GGUF q2_k** | ✅ ggml | ✅ `gemv_dual_q2_k` | ✅ decode→int8/int4 convrot (`buildPrep`); `gemv_q2_k_q8` + `dequant_q2_k_*` | ✅ ditto | 256 elems / 84 B |
| **GGUF q3_k** | ✅ ggml | ✅ `gemv_dual_q3_k` | ✅ `gemv_q3_k_q8` + `dequant_q3_k_*` | ⤷ dequant→f16 | 256 elems / 110 B; high-bit mask is inverted |
| **GGUF iq2_xxs / iq2_xs / iq3_xxs / iq3_s** | ✅ ggml | ✅ `gemv_dual_<fmt>` | ✅ `gemv_<fmt>_q8` + `dequant_<fmt>_*` | ⤷ dequant→f16 | codebook formats, 256-elem super-blocks of 66 / 74 / 98 / 110 B; tables in `src/gpu/kernels/iq_grid.zig` (`tools/gen_iq_grid.py`), one blob per backend (`Backend.iqTable`, `Context.iqTable`) |
| **GGUF iq4_nl** | ✅ ggml | ✅ `gemv_dual_iq4_nl` | ✅ shared-mem LUT | ⤷ dequant→f16 | 32 elems / 18 B, `kvalues_iq4nl` |
| **GGUF iq4_xs** | ✅ ggml | ✅ `gemv_dual_iq4_xs` | ✅ `gemv_iq4_xs`, `gemv_iq4_xs_q8n`, `embed_gather_iq4_xs`, `mmq_pipe_iq4_xs` | ⤷ dequant→f16 | 256 elems / 136 B; odd super-blocks only 8-byte aligned |
| **GGUF q1_0** | ✅ ggml | ✅ `gemv_dual_q1_0` | ✅ `gemv_q1_0{,_q8}`, `mmq_pipe_q1_0` | ⤷ dequant→f16 | 128 elems / 18 B; 1 sign bit per weight, `v = bit ? d : -d` |
| **GGUF q2_0 g128** | ✅ native `dotQ2_0G128` (not ggml) | ✅ `gemv_dual_q2_0_g128` | ✅ `gemv_q2_0_g128{,_q8}`, `mmq_pipe_q2_0` | ⤷ dequant→f16 | 128 elems / 34 B; `v = (code − 1)·d` |
| **GGUF q2_0 g64** | ✅ ggml | ✅ `gemv_dual_q2_0_g64`, ⚠️ device test only | ✅ built, ⚠️ never executed (no g64 file) | ⤷ dequant→f16 | 64 elems / 18 B, ggml's `QK2_0` |

**Notes:**

- **GGUF block quants on GPU decode inside the GEMV**, never expanded to VRAM. The `cuda` arm
  dequants to f16 for the prefill GEMM. cuBLASLt/cuDNN never read block quants.
- **convrot** (`src/ops/convrot.zig`): `cols` must be a multiple of 256 (i4 also even).
- ⚠️ **The packed 4-bit formats share one loader** (`models/quant_weight.zig`) and all store
  `[rows, cols/2]`, the int4-convrot shape, so each is detected by its sidecar (`_scale_2` for
  NVFP4, `_s_rel` for W4A8) BEFORE the int4 heuristic. W6A8 goes through the W4A8 reader, bit
  depth from the stored row width, cross-checked against `comfy_quant`'s `format`.
- **w4a8 shares int8's activation prep and GEMM**, so a block mixing int8 and W4A8 works.
- **svdq4 has its own prep** (`Prep.svdq`): one smoothing vector per group, which a fused
  tensor's row views share; a group mixing it with another prep is refused. An SVDQuant
  layer has no `.weight` tensor, so its reader runs first (`<base>.qweight`). Row views must
  sit on its 128-row tiles.
- ⚠️ **CUDA's W4A8 decode needs `group_size % 8 == 0`** (`dit.w4a8SmallGroup` refuses smaller by
  name). Vulkan's takes any group but requires its input pre-transposed through `weightBuffer`.
- ⚠️ **The device weight caches key on HOST POINTER.** A reused weight-shaped buffer scores a stale
  hit. Never hand `Backend.zeroBias` to `cachedWeight`: it reallocates on growth.
- **q1_0 and q2_0 g128 use 128-element blocks**; code that hardcodes 32 or 256 is wrong for them.
- ⚠️ **GGUF type id 42 is two formats** (ggml's g64, 18 B; PrismML's g128, 34 B), with nothing in
  the file to tell them apart. `gguf.detectQ2_0Variant` resolves it from tensor offset gaps;
  ambiguity is `AmbiguousQ2_0Variant`. Reading g64 as g128 passes bounds checks silently.
- **q2_0 g128 is decoded natively** (`quants.dequantQ2_0G128`, `dotQ2_0G128`, `matmul.native_gemv`),
  with exact activations; `quants.ggmlType(.q2_0_g128)` returns null so no ggml path can reach it.

---

## 8. Kernel inventory (appendix)

### Both GPU arms — dual-target kernels (`src/gpu/kernels/dual.zig`, table `dual_table.zig`)

One source, SPIR-V and PTX; the full list is `dual_table.entries`. By area:

- activations: `add{,_relu,_scaled,_h16}` · `relu` · `silu{,_mul,_mul16,_mul_h16}` · `sigmoid_mul{,_h16,_g16}` · `gelu{,_mul,_quick,_quick_mul,_erf}` · `geglu{,_h16}` · `softplus_gate`
- svdq: `mul_cols{,_f16,_bf16}` (the smoothing) · `svdq_decode` (row-major int8, CUDA) ·
  `svdq_decode_t` (k-major int8, Vulkan) · `svdq_unpack{,_f16,_bf16}` (exact weight-only)
- moves: `copy` · `scale_f32` · `concat_ch` · `scale_concat` · `scale_i32` · `quantize_i8` · `kv_store_f16` · `transpose_kmajor16` · `gather_kmajor{,_h16,16}` · `gather_head{,_b}` · `gather_vt{,_b}` · `scatter_head{,_b}` · `head_pad{,_h16}` · `head_unpad` · `deinterleave{2,3}` · `gather_rows` · `scatter_add_rows` · `moe_combine` · `moe_route`
- converts: `f32_to_h16{,_pad}` · `f32_to_bf16_pad` · `h16_to_h16_pad` · `bf16_to_h16_pad` · `f16_to_f32`
- modulation: `modulate` · `gated_add{,16}` · `rms_apply_mod{,_h16}` · `rms_apply_w` · `rms_partial` · `rms_combine`
- rope: `rope_inter` · `rope_half{,_pos,_part}` · `rope_imrope{,_pos}` · `rope_vision{,_gemma4}`
- row reductions (one subgroup per row): `rmsnorm` · `group_rmsnorm` · `gemv_rows_f32` (`Caps.f32Rows`) · `rms_mod` · `layernorm{,_h16}` · `ln_mod` · `l2norm_rows{,_g}` · `gn_stats{,_h16}` · `rowmax_i8`
- GroupNorm / VAE: `gn_combine` · `gn_apply{,_h16}` · `vae_norm` · `dup_up_add` · `bias_compact{,_h16}` · `bias_add_{f16,h16}` · `add_bias_rows{,_h16}` · `im2col` · `im2col_sd{,_h16}` · `im2col1d`
- Mage-VAE: `dw_conv3` · `col_mean` · `mul_cols_sigmoid` · `nerf_feat` · `modulate_pr` · `gated_add_pr` · `layernorm_f32_{f16,bf16}` · `patch_scatter` · `win_gather` · `win_scatter`
- softmax: `softmax_partial` · `softmax_combine`
- fused: `qknorm_rope16` · `qknorm_rope_f32`
- sampling: `argmax_reduce` · `argmax_final` · `topk_reduce` · `penalize`
- GDN: `gdn_gates{,_batch}` · `gdn_conv_{step,batch,state}` · `gdn_delta_chunk_d{64,128}`
- attention: `attn_rows_{f32,f16,q8}` · `attn_rows_g{2,4,8}_{f32,f16,q8}` (grouped KV heads) · `attn_merge`
- k-split: `gemv_combine{,4}`
- H3 audio: `aa_up_snake` · `aa_down` · `convt1d_ca` · `snake1d_ca` · `mean_heads_pool`
- dequantizers: `dequant_<fmt>_{f16,bf16}` for fp8, q8_0, q4_0, q1_0, q2_0_g64, q2_0_g128, iq4_nl, iq4_xs, q2_k, q3_k, q4_k, q5_k, q6_k, iq2_xxs, iq2_xs, iq3_xxs, iq3_s
- integer decodes: `w6a8_decode` (CUDA) · `w6a8_decode_t` (Vulkan) · `dequant_i8row_bf16` · `i{8,4}row_rot_bf16` (PTX only; the SPIR-V build is empty)
- decode GEMVs: `gemv_{q2_k,q3_k,iq2_xxs,iq2_xs,iq3_xxs,iq3_s}` (f32 activation) and `_q8` (q8_1 activation), the only decode kernel those six formats have · `gemv4_dual_<fmt>` (up to 4 activation rows, each bit-identical to a one-row launch) · `gemvx_dual_<fmt>{,_sdot}` (routed MoE slot)

⚠️ **A dequantizer and its GEMV decode the same format through different code** (`value` vs
`group`), so they can drift; the device tests score both against ggml.

⚠️ **The shared dual module runs `dp4a` as arithmetic on SPIR-V**, since carrying the integer-dot
capability would fail every kernel on a device without it. The `OpSDot` GEMVs are separate `_sdot`
entries, taken when the device has integer dot (`lin_gpu.Caps.dualQ8`; `sdot_off` disables them).

**Routing.** CUDA uses its hand GEMVs where one exists and the dual GEMV for the six formats
without one. Vulkan routes every block quant to `.gemv_dual` (`dequantOnly`), and a batch of up to
`dual_lin.grouped_max` rows to `.gemv_dualn`. Harnesses: `zig build qgemv-bench` (CUDA),
`TensorPencil vk-gemv-bench` (Vulkan).

⚠️ **On SPIR-V a quad load needs an alias binding** (a second `extern var` on binding 1 typed
`@Vector(4, f32)`); without it `Env.ld4` is four scalar loads.

⚠️ **A 16-entry codebook belongs in the table**: `iq4Codes` reads `kvalues_iq4nl` from
`iq_grid.blob`, not from a switch.

### Vulkan-only — `Elt` compute kernels (`src/gpu/kernels/eltwise.zig`)

`attn_full` · `attn_causal_batched` · `attn_cross` · `attn_scores` · `attn_out` ·
`kv_store_q8_0` · `gemv_partial{,4}` · `gdn_delta_step` · `rotate_fwht` · `w4a8_decode_t` ·
`i4_decode_t` · `nvfp4_decode_t`. LLM attention against its cache is the dual `attn_rows*`.

**Vulkan GEMM entry points** (`context.zig`): `opMatmul` (f32/fp8) · `opGemv{,Partial}` ·
`opGemvQuantDual` (block-quant decode) · `opMatmulCoop{,H16}` (fp8→f16) ·
`opMatmulCoopF16W{,b,h,Dev}` (f32/bf16/f16→f16; `Dev` takes a device-resident bias) ·
`opMatmulCoopBf16` (native bf16) · `opMatmulCoopQuant` (GGUF prefill) · `opMatmulFp8Wide` (fp8 as
f16 weights) · `opMatmulCoopI8{,Fused}` / `opMatmulI8` / `opI8Gemm` (int8 s8→s32) ·
`opAttnScores{,Vae,Sd}` / `opFlash` / `opAttnOut`.

Coopmat SPIR-V builders (`src/gpu/coopmat.zig`): `buildGemmShared` (f16/bf16), `buildGemmSharedI8`,
`buildGemmI8` (register-tiled int8 fallback), `buildFlashAttn`, `buildGemmAttnOut`,
`buildGemmScores`, `buildIntWoPack` (weight-only integer pack, tile sized by `woPackRows`). No s4.

- **Fragment shape**: `buildGemmShared` and `buildGemmSharedI8` take the device's M/N (16×16 on
  NVIDIA/AMD, 8×8 on Intel Alchemist); every other builder is 16-only and is gated off where 16
  is unavailable. K is 16 (f16/bf16) or 32 (int8). `Context.init` picks the widest usable config
  and probes f16-accumulate and bf16 at that shape. An inexpressible shape leaves the pipeline null
  and the caller falls back.
- **Warp tile** (M and N separately): `TP_COOP_WARP_TILE` / `_N` (f16) and `TP_COOP_I8_TILE` / `_N`
  (int8). ⚠️ Validate M and N as a pair; a rectangle can be legal where its square is not.
  `coop_i8_wg_m/n` drive both the dispatch grid and which shapes may take the shared path;
  `opI8Gemm` returns `UnsupportedDType` for a shape neither int8 kernel can express.
- **Tile selection** precedence: env, then the cache, then an init screen, then 64×64. The cache is
  `<XDG_CACHE_HOME or HOME/.cache>/tensorpencil/coop_tile` (`TP_COOP_TILE_CACHE`), one line per
  `vendor:device:driver`. `tune-coop` runs the full sweep and writes it; `Context.init` screens and
  saves when no entry matches (`context.autotune_tiles = false` / `TP_COOP_AUTOTUNE=0` disable,
  `context.tile_cache_path` redirects). The init screen switches only on a large win, so it may
  stop short of `tune-coop`'s pick. `gpu-test` prints the tile in force. `Context.init` takes an `Io`.
- ⚠️ **`Context.coopNPad` is the only place that may compute the coop GEMM's padded N**; every
  k-major weight producer (`weightBufferF16From32`, `weightBufferF16FromBf16`, `nvfp4Decode`) pads
  to it, or the GEMM reads the wrong offsets.
- ⚠️ **In `buildGemmShared` the B staging mask/shift follow the 16-element per-thread chunk, not
  `frag_n`**; a mistake is invisible at 16×16 and garbage at 8×8.
- `spirv_asm.assembleChecked` rejects a `%name` referenced but never defined.
- **No coop GEMM** (`TP_VK_COOP=0` or a device without one): everything still runs on the GPU; a
  16-bit weight takes the f32 GEMM widened once (`weightBufferWidened`, twice the memory), and
  convs stage around it (`conv_gpu.f32Out` / `f32Finish`).
- **Weight-only integer** widths each ask for their own GEMM (`Caps.intWeightOnly`); a width with
  no GEMM widens to the f32 expansion.
- **fp8 wide route**: where the device flushes f16 subnormals in the coop multiply (`verifyGemms`
  runs fp8 at a 2^-17 scale to find out), fp8 runs as f16 weights with the scale in the epilogue
  (`Context.fp8_wide`). `TP_VK_FP8_WIDE` forces it, `TP_VK_FP8_F32` takes the f32 GEMM. Every
  module asks for f16 gradual underflow (`spv.withFloatModes`).
- **AMD RDNA2**: wave64, coop shaders pinned to subgroup 32 (`VK_EXT_subgroup_size_control`); no
  bf16 coop, so bf16 weights take the f16 coop GEMM; int8 at the driver's K=16 as four sub-steps.
  ⚠️ `createModule` refuses any buffer array of 4 GiB or more (the driver reads it as element 0);
  `adaptBatchCap` sizes submissions under Windows' TDR.
- **Intel Alchemist**: 8×8 fragments (`8x8x16` f16/bf16, `8x8x32` s8, f32/s32 accumulate only);
  the 16-only attention builders are off, and the SD UNet uses `attn_batched` instead.
- Pipelines are cached per SPIR-V (`pipe_cache.zig`). `buildGemmScores` loops its k-depth past
  head 128 and is compiled per head width (128, 384, 512).

### zig-cuda — hand-PTX (`src/gpu/cuda/kernels.zig` GEMM, `elt.zig` elementwise/attn)

GEMM builders: `buildHgemm` (f16/bf16 mma m16n8k16, optional f32 A/C) · `buildIgemmSmem` /
`buildIgemmPipe` (int8 m16n8k32, int4 m16n8k64; `_fused` rescales and adds an optional bias) ·
`irescale`/`irescale_h16` (the `.libs` int8 epilogue) · `buildPrep` (quant/rotate) ·
`buildMmqPipeQ{1_0,2_0}`, the q4_k/q5_k/q6_k/iq4_xs and q8_0 MMQ pipes (`mmq_pipe_q8_0` opt-in via
`--dit-gguf-gemm mmq`). DiT GEMMs load fragments with `ldmatrix` over an XOR-swizzled shared layout
(`use_ldmatrix`, bit-exact).

GEMV: `gemv_{fp8,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k,iq4_nl,iq4_xs,q1_0,q2_0_g64,q2_0_g128}` plus `_q8`
and grouped-N `_q8n` dp4a variants (none for iq4_nl).

Attention: `attn` · `attn_split`/`_merge`/`_h256`/`_h512`/`_tree` · `attn_split_g` (group-shared KV,
hd ∈ {128, 256}, causal, `heads > kv_heads`, `_f16`/`_q8` KV variants) · `softmax_md_{f16,f32}`.

GDN: `gdn_delta_step` (block per head) · `opGdnDeltaChunk` (state in registers; ⚠️ must stay
bit-identical to the per-token form, which decode runs).

Still hand PTX in `elt.zig`: `qk_rmsnorm_par` · `attn_cross` · `attn_batched` · `quantize_q8_1` ·
`w4a8_decode` · `nvfp4_decode` · `f32_to_q8_0` · `im2col3d` · `gdn_delta_step` · the graph-capture
`decode_state` set. Everything else elementwise is the dual-target module.

### cuda — vendor libs (`.libs` mode)

**cuBLASLt** (`src/gpu/cuda/cublaslt.zig`): int8 `R_8I`/`COMPUTE_32I`, f16 `R_16F`, bf16 `R_16BF`,
batched/prefill GEMM only. **cuDNN** (`src/gpu/cuda/cudnn.zig`): fused SDPA-forward attention,
legacy conv-forward (VAE). Everything else is hand-PTX.

### cpu — `src/ops/*.zig` + ggml (fetched dep, `-Dggml`, default on)

`matmul.zig` (dtype-aware microkernel; block-quant decode GEMV → ggml `vec_dot`, `native_gemv` for
q2_0 g128) · `attention.zig` (+`attentionTree`) · `norm.zig` · `act.zig` · `rope.zig` · `conv.zig` ·
`convrot.zig` · `w4a8.zig` · `nvfp4.zig` · `vmath.zig` (SIMD `expVec`). ggml owns CPU block-quant
dequant and `vec_dot`; the rest is in-house Zig. With `-Dggml=false`, block quants return
`error.QuantBackendUnavailable`; other dtypes are unaffected.

---

## 9. Known gaps

Delete a row when it closes.

| gap | detail |
|---|---|
| `cuda-dit-test` fails on krea2 | bf16 and int8 convrot arms disagree with the CPU forward; krea2's bf16 CUDA arm has no working validation. |
| Vulkan diffusion attention | slower than CUDA's for the same work, the P@V pass most of all; the GEMMs are not the gap. |
| Vulkan weight upload | no parallel pinned staging like CUDA's `Context.FillPool`, so first-image setup is slower. |
| Vulkan gemma4 decode | behind zig-cuda and uses more VRAM; the q4_0 `_sdot` GEMV is the slower kernel. |
| Vulkan k2-horizon speed | behind CUDA on decode and prefill; GPU kernel time (q6_k `_sdot` GEMVs), not barriers. Untried: split a row over more subgroups; an expert-packed `gemvx`. |
| Vulkan LLM decode barriers | most remaining barriers are real RAW/WAW hazards on reused scratch; fusing dependent ops and alternating scratch are the levers. |
| `opMatmulFp8` writes `y` directly | carries `launchHgemm`'s `mpad`-rows requirement implicitly; zimage/anima `.f8_e4m3` arms never exercised. |
| `mmq_pipe_q4_k` far below int8 peak | LLM q4_k prefill only (not diffusion); bound by shared-memory bank conflicts on fragment loads. Read its block comment before optimizing. |
| q8_0 MMQ slower than dequant | `mmq_pipe_q8_0` is correct and opt-in (`--dit-gguf-gemm mmq`) but loses to the dequant route and cuBLASLt int8. |
| GGUF int8 decode in diffusion covers only q2_k/q4_k/q8_0 | q5_k/q6_k/q4_0/iq4_nl DiTs take the bf16 dequant route on CUDA and are CPU-only on Vulkan; widen at `Backend.blockQFormat` plus a block walk in `buildPrep`. |
| q2_0 g64 kernels unexecuted | no g64 file here to run them against. |
| Qwen-Image 2.1 reference editing | the DiT takes reference latents (`Ref`), but nothing produces them: no 2.2-layout VAE encoder, no vision tower for the 8B encoder, no device path for an image prefix. |
| Qwen-Image 2.1 f16 DiT stream | finite at 1024², but default stays f32 until an interleaved A/B. |
| `--text-encoder-2` split path unexercised | every SDXL checkpoint here is bundled. |
