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

⚠️ **A GEMM writing f16 D needs an f16 BIAS.** cuBLASLt's epilogue bias type follows D, and
the heuristic rejects an f32 bias over an f16 D outright (`INVALID_VALUE`, no algo) rather
than converting. `cachedBiasF16` keeps the converted vector; a bias is `co` floats, so unlike
a weight the second copy costs nothing.

⚠️ **The `m`/`co`/`k` padding around the f16 GEMMs is the hand-PTX tile's requirement, not
cuBLASLt's, and it is not free.** The padded form converts the activation into an
`[m_pad][k_pad]` plane, accumulates into an `[m_pad][co_pad]` f32 plane, then reads that whole
plane back to compact it and add the bias. Under `.libs`, `opMatmulF16` / `opMatmulBf16` /
`opConvF16Prec` instead go through `ltGemmHalfBias`: nothing padded, and the bias added in the
GEMM's own epilogue (`CUBLASLT_EPILOGUE_BIAS`, which is why the chosen kernels have `relu` in
their names). Needs `k % 8 == 0`; the padded path stays for everything else and for
`zig-cuda`. A bf16 weight is rewritten to f16 **in place** on first use
(`cachedWeightF16`) rather than re-converted per call — both are 16 bits, so a second copy
would cost the weight's size again in VRAM for nothing.

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
| **DiT** (28 blocks) | ✅ all dtypes | ✅ fp8/int8/w4a8/nvfp4/bf16 | ✅ + int4 + GGUF block quants | ✅ + int4 + GGUF block quants | `dit{,_gpu,_cuda}.zig` |
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
| **cuda** | cuDNN fused SDPA at the true width, self and cross alike | none |
| **zig-cuda** | zero-padded to a multiple of 128 | `128/hd` on the attention |
| **vulkan** | zero-padded to 128; hd 160 falls back to the scalar kernel | as above |

Cross-attention onto the 77-row text conditioning goes through `opAttnCross`, which picks
cuDNN's rectangular SDPA under `cuda` and the head-padded `opAttnTCRect` under `zig-cuda`.

The padding is exact (a zero dimension contributes nothing to a dot product, and V's zero
columns give output columns that are dropped), so it costs only arithmetic. Both GPU limits
are the same one: the P@V GEMM tiles the head dimension in 128-wide blocks
(`coopmat.buildGemmAttnOut`, `launchHgemmB`'s `grid.x = n/128`). Removing the padding means
parameterizing those two builders by head width, not changing the model code.

⚠️ **Under `--backend cuda` the SD UNet carries its activations as f16 in DRAM**
(`sd_unet.Config.act_f16` → `sd_unet_cuda.Workspace.act_f16`). Every kernel still computes in
f32 and every per-channel parameter (norm weight/bias, timestep projection, convolution bias)
stays f32; only the big buffers narrow. It is worth 1.2-1.3x on the whole step at every
resolution, from two effects: the per-GEMM and per-attention conversions disappear outright
rather than shrinking, and the bandwidth-bound half of the step (GroupNorm, LayerNorm, GEGLU,
the residual adds) moves half the bytes. `zig-cuda` stays f32: its narrow-head attention pads
through `opHeadPad`, which has no f16-to-f16 form, and its hand-PTX GEMM has no f16-D form.

Two things this forces, both of which are silent wrong answers when missed:
- **Every buffer a GEMM reads has to be at the stream's width, including the ones that are
  not activations.** The per-image text conditioning is a GEMM source for the cross-attention
  K/V projections, so `Session.ctx_d` is uploaded as f16 too. Read as f32 it renders NaN.
- **An activation capture (`ops.matmul.probe`) and an f16 stream are mutually exclusive.** The
  probe accumulates f32, and a convolution's probe wants the `[n][9*ci]` im2col matrix that
  the cuDNN path never builds; both are gated off rather than fed the wrong thing.

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

⚠️ **Anima calls `opAttnTCRect` directly, so both CUDA arms run the same hand-PTX kernels on
its `seq × 512` cross-attention.** `SdpaPlan` does take a rectangular shape now, and the
dispatching `opAttnCross` would send `--backend cuda` to cuDNN; Anima deliberately does not
go through it.

⚠️ **Rectangular tensor-core attention is a requirement, not an optimization.**
Cross-attention is `seq × 512` — 2.4% of a step's FLOPs — but the naive
thread-per-(query, head) kernel (`opAttnCrossNaive`, kept as the validation reference)
streams the whole context per thread through a local-memory accumulator, costing an
estimated ~0.6 s/step at 1056x1584. The FLOP share is not what decides; the data reuse is.
The SD family paid the same bill for real: its 70 cross-attentions onto a 77-row text
conditioning were **45% of an SDXL step** at 1120x1680 until they were routed through
`opAttnCross`.

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

### 2E. MiniMax H3 (joint audio-video DiT)

The first family that is not a still image: video and stereo audio are denoised in
ONE packed token sequence, on two sigma schedules. See VIDEO_PLAN.md.

| Stage | cpu | vulkan | zig-cuda | cuda | Files |
|---|---|---|---|---|---|
| **Text encoder** (Qwen3-VL-32B, 50 layers, UNNORMALIZED last state) | ✅ int8-convrot | — | ✅ | ✅ | `qwen3{,_cuda}.zig`, `Variant.minimax_h3`. The device arms have the int8 GEMM and the vision path, and are **60x** where the weights fit (25 layers: 7421 -> 124 ms). ⚠️ They do not fit a 3090: 23250 MB of weights against 21451 MB free, and past that the per-layer re-upload makes it SLOWER than the CPU (50 layers: 20840 ms vs 18384 ms). `supportsWeightsOn` gates on measured free VRAM, so this encoder runs on the CPU here |
| **DiT trunk** (50 blocks, 5376, 56x128) | ✅ int8 | — | ✅ int8 | ✅ int8 | `minimax_h3{,_cuda}.zig` |
| **Video VAE decode** (ViT3D, 36 blocks, temporal chunking) | ✅ | — | ✅ | ✅ | `minimax_h3_vae{,_cuda}.zig` |
| **Audio VAE decode** (BigVGAN, 32 kHz stereo) | ✅ | — | ✅ f32 | ✅ f32 | `minimax_h3_audio{,_cuda}.zig` |
| **Video VAE encode** (3-D causal CNN, banded im2col) | ✅ | — | ✅ | ✅ | `minimax_h3_vae_encode{,_cuda}.zig`, references and keyframes |
| **Audio VAE encode** (DAC + causal-attention posterior head) | ✅ | — | ✅ | ✅ | `minimax_h3_audio_encode{,_cuda}.zig`, reference soundtracks |
| **Vision blocks** (fl2va / ref2va conditioning) | ✅ | — | ✅ | ✅ | `minimax_h3_vit.zig` + `qwen3{,_cuda}.encodeVision`. Vulkan has no deepstack/mrope path and falls back to the CPU, loudly |
| **Denoise masks** (per-row timesteps) | ✅ | — | ✅ | ✅ | `Timesteps.initMasked`; the device path adds a per-row index buffer to `rms_mod_par` / `gated_add` |
| **Spatial VAE tiling** | ✅ | — | ✅ | ✅ | load-bearing, not a memory bound |
| **MP4 muxing** (H.264 + AAC, `parameters` tag) | ✅ | ✅ | ✅ | ✅ | `lib/video/av_helper.c`, `src/av.zig` (exe only) |
| **LoRA sidecar** (`--lora`, never merged) | ✅ any dtype | — | ✅ bf16 | ✅ bf16 | `lora{,_cuda}.zig`, architecture-independent |

Both VAE ENCODE sides are now on the device, validated by
`minimax-h3-vae-encode-cuda-test` and `minimax-h3-audio-encode-cuda-test` against
their CPU references on real weights (rel L2 ~2e-6 in both cases). Measured on a
3090 with the vendor-library arm:

| | CPU | device | | host peak | device peak |
|---|---|---|---|---|---|
| video, 17 frames @ 256px | 47.9 s | **1.11 s** | 43x | 4352 MB | 1824 MB |
| video, 17 frames @ 128px | 10.1 s | 0.33 s | 31x | 1088 MB | 573 MB |
| audio, 6 s stereo | 4.22 s | 0.20 s | 22x | 297 MB | 260 MB |
| audio, 2 s stereo | 1.42 s | 0.09 s | 16x | 109 MB | 129 MB |

`qwen3_cuda.encodeVision` is the device vision path: the block rows are pasted
before the upload, mrope replaces the 1-D rope table, and DeepStack is added per
span with an offset `opAdd` (an injection span is a contiguous row run, so it needs
no kernel). Validated by `te-test` with `TP_TE_VISION=1` on a real block-quant
encoder: **1.44e-4** device-vs-CPU against a 1.29e-4 text-only control, 5.7x faster,
with the payload moving the conditioning by 0.119. Teeth confirmed by disabling the
device injection, which fails at 4.19e-2.

**The int8 activation prep no longer has a width ceiling.** It stages the row in
dynamic SHARED memory where that fits and in a GLOBAL buffer where it does not
(`buildPrep`'s `stage_global`, chosen by `kernels.prepNeedsGlobalStage` against the
device's opt-in shared limit). Same algorithm, same arithmetic, same instruction
order -- only the address space of the row accesses changes. The isolation is
`minimax-h3-cuda-test TP_H3_GLOBAL_PREP=1`, which forces the global path at a width
that fits and requires the two to agree **to the last bit**: measured rel L2 exactly
0. `Backend.force_global_prep` is the flag.

Before this, a reduction wider than 25024 columns on sm_86 could not run at all: the
prep needs `cols * 4 + 1280` bytes of shared and the limit is 101376, so MiniMax
H3's 25600-wide `down_proj` missed by exactly one 256-wide convrot group.

⚠️ **Both are about EVEN on the hand-PTX arm**, which has no tiled f32 GEMM and
falls back to a one-thread-per-output kernel. They stay in f32 there rather than
dropping to f16: the decode side measured f16 at about -42 dB, and a latent's error
rides through every sampling step as conditioning. `--backend cuda` (vendor
libraries) is what the speedups above need.

The video port replaces only `encodeMoments`; the temporal clip chunking and the
spatial tiling above it are shared with the CPU path through a `Moments` hook, so
the intricate part exists once. Its workspace GROWS per call rather than being sized
for the untiled extent, because tiling and chunking both hand it something smaller.

The CUDA trunk keeps the host-cheap paths on the CPU (patch projections, the adaLN
projection, the token refiner, the output heads): several have shapes the device
GEMMs refuse, and together they are under 0.01% of a step. Measured 129x over the
CPU trunk at 150 packed rows.

Measured end to end, 256x256 / 5 frames / 8 steps on a 3090, as each piece moved:

| | CPU everything | + CUDA trunk | + CUDA video VAE |
|---|---|---|---|
| video decode | — | 420 s | **4.9 s** |
| whole render | ~13 min (1 step) | 10 min | **2 min 21 s** |

The audio VAE (BigVGAN) is now on the device too: **47.8x** at 37 latent frames
(3591 ms -> 75 ms), which takes a 22-frame clip's audio decode from ~9 s to ~0.2 s.
Four new kernels, all in `elt.zig`: `im2col1d`, `aa_up_snake`, `aa_down` and
`convt1d_ca`. Signals are CHANNEL-LAST there, unlike the CPU reference's planar
layout — that is what makes every kernel coalesced (a warp covers consecutive
channels at one time step) and removes the transpose the CPU im2col pays on both
sides of its GEMM. The conv weights are permuted once per session to match.

⚠️ **Its GEMMs run in f32, not on tensor cores, and that is measured.** Against the
f32 CPU reference on the real vocoder, the f16 tensor-core route is 2.4e-3
relative with a 8.3e-3 worst sample — about -42 dB, with 74% of samples past 1e-4,
which for a VOCODER is an audible noise floor. Plain f32 through cuBLASLt
(`opMatmulF32Lt`, `COMPUTE_32F`, deliberately not `_FAST_TF32`) is 9.5e-6 / 2.8e-5,
under 16-bit PCM's own quantum. At the real shape f16 was also **not faster**
(82 ms against 75 ms): the per-conv weight-pad and activation-convert passes cost
more than the tensor cores win at these widths. `TP_H3_AUDIO_F16=1` on
`minimax-h3-audio-cuda-test` reproduces both numbers. `zig-cuda` has no tiled f32
GEMM, so it keeps the one-thread-per-output kernel: correct, 6.9x instead of 47.8x.

The video VAE's device path needs two things the DiT's did not: `opDeinterleave3` (its
`to_qkv` is fused PER HEAD, so the planes are not row ranges) and the SD family's
head padding (`opAttnTC`'s P@V GEMM tiles at 128 and this VAE is 32 heads of
**64**, which launches a zero-sized grid rather than computing a wrong answer).

The LoRA sidecar goes through `opGemmBf16` twice per output range (A then B) plus a
scaled accumulate, so its factors sit in the same pointer-keyed device weight cache
as the trunk's own and the VRAM arbiter sees them. That fixes the shape limits: the
factor's output width must be a multiple of 128 and its contracted width a multiple
of 32, which every real rank (128, 384) clears. `lora_cuda.supported` refuses
anything else by name, and the whole trunk then falls back rather than running the
base GEMM alone — a sidecar applied nowhere is a different model, silently.

Device residual against the f32 host apply is **2.3e-3** relative, entirely
explained: 1.66e-3 from rounding the activation to bf16 for each of the two GEMMs,
in quadrature. That is under the int8 base path's own ~4e-3.

The accumulate onto the base GEMM's output is folded into cuBLASLt's epilogue
(`opGemmBf16Acc`, `beta = 1` with `C == D`), because materializing the delta and
adding it separately cost MORE than the two GEMMs it served: at 512x512 the two
GEMMs are +7% of a step and an unfused accumulate another +19%, against +9% fused.
`zig-cuda` keeps the unfused route, since the hand-PTX `hgemm` writes its C tiles
unconditionally; `Workspace.fused` decides once per session.

⚠️ **The activation prep's rotation used to truncate.** `buildPrep` computed its
FWHT butterflies per thread as `ngroups * 64 / 256`, which rounds DOWN whenever
`cols % 1024 != 0`, leaving the tail of every row in its unrotated basis: no
error, no assert, a GEMM in the wrong basis. Every width in the engine happened to
be a multiple of 1024 until H3's 5376-wide hidden (21 groups, needing 5.25
iterations and getting 5), which was 22% wrong on every linear reading that width.
Fixed by rounding up and guarding the tail; `prepButterflyIters` is the exposed
form and a device-free test pins it. Any future model whose hidden width is not a
multiple of 1024 would have hit the same thing.

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
| **int8-tensorwise (no convrot)** | ✅ | ✅ | ✅ | ✅ (+f16 MLP) |
| **int4-convrot** | ✅ | ✅ (decodes to int8) | ✅ W4A4 | ✅ W4A4 |
| **w4a8** | ✅ | ✅ | ✅ | ✅ |
| **nvfp4** | ✅ | ✅ | ✅ | ✅ |
| **bf16 dense** | ✅ | ✅ native/f16 | ✅ native/f16 | ✅ cuBLASLt `R_16BF` |
| **f32** | ✅ | ✅ (offload) | — | — |
| **GGUF q2_k** | ✅ | ❌ | ✅ →int8-convrot or →int4-convrot | ✅ ditto |
| **GGUF q4_k / q8_0** | ✅ | ❌ | ✅ →int8-convrot or →f16 | ✅ →int8-convrot or →f16 |
| **GGUF q4_0/q5_k/q6_k/iq4_nl** | ✅ | ❌ | ✅ →f16 (unmeasured) | ✅ →f16 (unmeasured) |

¹ fp8 block linears stream through `opMatmulFp8`: the weight decodes to an f16 scratch
(`dequant_fp8_f16`, per-tensor scale folded) and runs through `buildHgemm` (hand-PTX) or
`ltMatmulF16` (cuBLASLt). The scratch is re-materialized per GEMM (no fp8 tensor-core GEMM),
so fp8 on the CUDA backends is correctness-first and slower per step than int8.
⚠️ The **CUDA fused `opMatmul`** (bias + destination offset, used only by `first`/`last.linear`)
has no fp8 variant, so those two projections — like bf16 and int8 — are materialized to f32 once
at load (`DiT.opMatmulF32`); otherwise the run aborts on the fp8 assert or reads packed bytes as f32.

**ComfyUI's `int8_tensorwise` ships two variants and they are different arithmetic.**
Rotated (`"convrot": true`, `weight_scale` `[rows,1]`) quantizes the weight after a size-256
group Hadamard, so the activation prep must rotate too. Unrotated (`weight_scale` a scalar,
broadcast per row at load) rotates neither side. `quant_weight.int8Scale` reads which; whether
the prep rotates then comes from `dit.i8Convrot`, whose answer must cover every storage form
that shares that prep — `.w4a8` decodes to a *rotated* int8 weight, so omitting it pairs a
rotated weight with an unrotated activation and the render is uncorrelated noise
(rel RMSE 1.00 vs the CPU forward, measured). An unrotated checkpoint also tends to quantize
the projections and the whole text-fusion stack, which the convrot files leave dense.
The unrotated prep is `rowmax_i8` + `quantize_i8` on Vulkan and `iprep_nr` on CUDA.

⚠️ **Unrotated costs accuracy, and that is the format, not the port.** Spreading activation
outliers is the entire reason ComfyUI's int8 format is rotated at all. Measured on krea2 at
256px against the same CPU forward: convrot **0.033** rel RMSE, `asym_w4a8_int8` 0.049,
q8_0 GGUF 0.047, unrotated int8-tensorwise **0.108**. ComfyUI pays the same, so this
matches it rather than diverging from it.

**A q4_k or q8_0 GGUF krea2 DiT runs on both CUDA arms**, and it reaches the ordinary
int8 path rather than a block-quant GEMM of its own. `opI8GemmBlockQ` decodes each packed
weight into ONE reusable scratch — dequantize, rotate by the convrot FWHT, take each
output row's absmax, quantize to int8 with it — and then runs the same vendor kernel
int8 uses. This is `opI8GemmW4A8`'s trick: the packed weight stays resident, so a q4_k
krea2 costs 6.9 GB of VRAM against int8's 11.9, and the GEMM measures 489 ms against
int8's 494 — the same kernel, so the same speed. `linPrep` is int8's, unchanged; the
rotation has to match on both sides.

The two formats share everything after the load stage, so a new one is `buildPrep`'s
`PrepBlock` plus a block walk: block stride, where the scale sits, and which element of
a block a thread reads. It must divide 256 elements per block, because the load walks the
row in strides of 256 columns and relies on a thread's position inside a block being
constant across the blocks it touches. A checkpoint may mix the formats: `i8GemmW`
dispatches per WEIGHT dtype, not on the model's one `LinKind`.

⚠️ **The int8 decode re-quantizes, so it caps accuracy at int8-convrot's whatever the
source format carried**, which is why there is a second route (`--dit-gguf-gemm`, below).
q8_0 stores a scale per 32 columns and arrives at the GEMM with one per row: swept over 4
blocks × 4 linears its derived int8 weight carries **1.19x** the error of a native ComfyUI
int8 checkpoint's (0.0107 vs 0.0090 relative, uniform across every layer, with the
computed row scales agreeing with ComfyUI's stored ones to under 1%).

⚠️ **The per-row scales are computed here, not read from the file.** int8 and w4a8
carry theirs from the quantizer; a block quant's scale is the absmax of the ROTATED
weight and nothing in the GGUF knows it. It falls out for free because the weight
decode IS `buildPrep` with a packed input mode — rotating a row, taking its absmax and
quantizing to int8 is the same operation whether the row is a token's activations or a
weight's output row.

**Two decode routes, `--dit-gguf-gemm auto|int8|f16`** (`dit_cuda.blockQKind`). `int8` is
the convrot path above. `f16` expands the weight to f16 and runs the f16 tensor cores via
`opMatmulQuant`, the same op the LLM prefill uses, so it needs no rotation, no per-row
scale and no absmax reduction: those exist only to make ONE int8 scale per row viable.
That also makes it the wider route, covering every format with a dequant kernel
(q4_0/q8_0/q4_k/q5_k/q6_k/iq4_nl) where int8 covers only q4_k and q8_0, and it drops the
convrot `cols % 1024` floor.

⚠️ **Which route wins is a property of the FORMAT, and the deciding number is the format's
own weight error, not its bit width.** The two errors add roughly in quadrature and
int8-convrot's floor is ~0.009 relative, so it vanishes under a 4-bit format's error and
dominates an 8-bit one's. Measured against the dense checkpoint over 4 blocks × 4 linears,
choosing int8 over f16 costs:

| format | own error | via int8 | cost of int8 |
|---|---|---|---|
| q4_k | 0.0917 | 0.0923 | **+0.7%** |
| q8_0 | 0.0073 | 0.0134 | **+84%** |

So `auto` sends q4_k to int8 and everything else to f16. Pushing q4_k through f16 is
1.85x the time (2.40 vs 1.30 s/step) for nothing: over 4 seeds it measures **-1.44 dB
±1.14**, i.e. zero, and the 3-7% it does take off the DiT-vs-CPU velocity error is mostly
the ACTIVATIONS (which f16 does not quantize), not the weights.

⚠️ **Do not run q8_0 through the int8 route.** It is strictly dominated: a native int8
checkpoint is SMALLER (13.2 vs 15.6 GB) and equal in accuracy, so the only reason to hold
a q8_0 file is the accuracy the f16 route keeps.

**q8_0 on f16 is how to run a 24 GB card at near-dense accuracy**, and it costs almost
nothing over a dense f16 model that fit: the weight expansion is 61 ms/step, measured on
the `dequant` bucket, and it is resolution-INDEPENDENT (224 launches over the same weights
whatever the sequence length), so at 1024² it is **2.6%** of the 2.34 s step. The rest is
the f16 GEMM, which measures **3.06x** the int8 one at identical shapes (441 vs 144 ms at
lat=64) against a 4x spec ratio for f32 accumulate. That 3x is the whole gap between the
two routes, so no amount of decode tuning closes it.

⚠️ **Do not compare against the dense bf16 checkpoint's wall time on a 24 GB card.** It
needs 25.1 GB, so it streams and measures 3.3-7.0 s/step depending on what else is
resident. It is a correctness reference, not a speed baseline.

⚠️ **Every other GGUF block quant is still CPU-only, and that is enforced**
(`dit.gpuLinKindSupported`, which takes which GPU arm is asking, and whose list is the
union of `Backend.blockQFormat` and `Backend.quantKernelSupported`, plus the
`anima`/`zimage` equivalents). Vulkan has neither decode.

**q2_k is the smallest krea2 that renders**, 6.6 GB on disk and 4.0-4.2 GB of DiT VRAM,
and it renders a coherent image. Its weight error is 4x q4_k's, so treat it as the low-VRAM
option, not a quality one. The int8 regrid is free on it (0.3086 against the file's own
0.3085), and it is the only format for which the int4 route is even arguable, because
int4-convrot's own ~0.17 floor sits below what q2_k already costs. It is still not the
default: `cuda-dit-test <q2_k> 128 libs int4` improves the matmul bucket 502 -> 404 ms but
leaves `prep` and attention untouched, so the step only moves 1.11x, and one seed rendered
2.4 dB below the int8 route. `--dit-gguf-gemm int4` for it.

⚠️ **q4_0/q5_k/q6_k/iq4_nl are wired but UNMEASURED here** — no such diffusion checkpoint
is on this box. They reach `opMatmulQuant`, which is covered per-dtype by its own device
test, but no render has been made with one.

**Measured, krea2 1024², 3090, against the bf16 checkpoint:**

| format | file | DiT VRAM | s/step | weight err | PSNR |
|---|---|---|---|---|---|
| dense bf16 | 26.3 GB | 25.1 GB (streams) | 3.3-7.0 | 0 | reference |
| **q8_0 GGUF, f16** | **15.6 GB** | **12.7 GB** | **2.46** | **0.0059** | **34.6** |
| q8_0 GGUF, int8 | 15.6 GB | 12.7 GB | 1.37 | 0.0109 | 34.5 |
| int8 convrot | 13.2 GB | 11.9 GB | 1.14 | 0.0091 | 31.8 |
| fp8 e4m3 | 13.2 GB | 6.9 GB | 2.45 | - | 28.40 |
| **q4_k GGUF, int8** | **9.5 GB** | **6.9 GB** | **1.33** | **0.0768** | **21.9** |
| q4_k GGUF, f16 | 9.5 GB | 6.9 GB | 2.40 | 0.0762 | - |
| int4 convrot | 8.1 GB | 5.9 GB | 0.95 | 0.1657 | 19.2 |
| **q2_k GGUF, int8** | **6.6 GB** | **4.2 GB** | **1.29** | **0.3086** | **-** |
| q2_k GGUF, int4 | 6.6 GB | 4.0 GB | 1.21 | 0.3348 | - |

s/step and VRAM at 1024², `--backend cuda`. **`weight err`** is the relative Frobenius
error of what the GEMM actually multiplies, against the dense checkpoint, over 4 blocks × 4
linears. **PSNR** is the mean over 4 prompts at one seed.

⚠️ **Rank close formats by `weight err`, never by PSNR.** PSNR over a 20-step trajectory
has a per-prompt sd of ~11 dB here, so paired over 4 prompts it resolves q4_k vs int4
(+2.71 dB, SE 0.82) and NOTHING inside the near-lossless group: q8_0's two routes differ by
1.84x in weight error and measure +0.06 dB apart with SE 2.69. The weight error is stable
to a few percent across every layer and is what generalizes across prompts. Do not read a
PSNR gap under ~3 dB in this table as real.

q4_k is the quality leader of the 4-bit class and within 1.17x of int8 on time.

The q8_0 PSNRs are the two routes on the same file: on int8 it costs more than a native
int8 checkpoint on both axes and matches it on quality, which is why `auto` does not send
it there.

⚠️ **Read that column as a multi-seed mean: one seed's PSNR is worth about ±8 dB.**
20 steps of a sampler amplify a weight perturbation chaotically, so a single render cannot
rank two formats this close. Over 4 seeds int8 alone spans 28.08 to 45.00 (16.9 dB) and
crosses q8_0's line in both directions, while their means sit 0.05 dB apart. Rank two
close formats by weight error against the dense checkpoint instead, which is stable to
0.01x across every layer.

The decode costs ~0.2 s/step and that is nearly INDEPENDENT of the source format (q8_0 and
q4_k measure the same per step despite q8_0 reading 1.9x the packed bytes). Its extra DRAM
traffic over a plain int8 GEMM is only ~25 GB/step, ~31 ms at the 3090's bandwidth, so the
kernel is ~6x off its roofline: the cost is the FWHT and the shared-memory traffic, not
reading the weight. That is also why a fatter block quant is not proportionally slower.

The decode is chunked and takes its per-row scale as an INPUT (`buildPrep`'s `chunk`
argument), which is what got it from ~300 to ~180 ms/step. The two go together: a chunk
cannot compute a whole row's absmax, and without that reduction nothing forces the row
to be resident, so shared drops from `cols*4` (64 KB at mlp.down's cols=16384, ONE block
per SM) to `chunk*4`. Recomputing the scale per GEMM was pure waste — a weight's scale is
a property of the weight, so it is computed once on first use and cached per weight
(`bq_scales`, ~7 MB for krea2). An activation's is data and still costs a reduction.

⚠️ **The chunk must DIVIDE cols** (`bqChunk`). The kernel derives its chunk count as
`cols / chunk`, so a chunk that does not divide truncates it and silently leaves the tail
of every row undecoded: 4096 against krea2's 6144 covers two thirds of the weight and
still renders a plausible image. The builder asserts it and asserts are off in the builds
this runs in. ⚠️ The whole-row path normalizes shared by the FWHT's 1/16 gain before
taking the absmax; the chunked path folds that 16 into the incoming scale instead, and
omitting it gives a fully saturated int8 weight (rel RMSE 0.97).

⚠️ **The convrot rotation is load-bearing and cannot be dropped to close the last 91 ms**
against w4a8 (`dit_cuda.blockq_rotate`). Skipping it on BOTH sides is exact arithmetic — the
rotation cancels between weight and activation — and it is 1.181 -> 1.090 s/step because
the FWHT leaves the decode and the activation prep at once. It also destroys the output:
rel RMSE 0.915 against the CPU forward, where rotated is 0.044. The weight side barely
cares (unrotated per-row int8 costs 0.3% on top of q4_k's own error); it is entirely the
ACTIVATIONS, whose outliers one scale per row cannot span. That is what w4a8's format buys
by quantizing in the rotated domain up front, and it is why q4_k pays a decode w4a8 does
not. Closing that gap needs precomputed rotated weights, which costs the 6.9 -> 12.2 GB
that the packed form exists to avoid.

Real eltwise is ~90 ms/step at 75-93% of DRAM bandwidth (`gated_add` 844 GB/s,
`sigmoid_mul` 841, `rms_mod_par` 833, `rope` 829), already at the roofline, so only fusion
could help.

⚠️ **`cuda-dit-test`'s bucket labels were misaligned with `ProfCat`**, which omitted
`dequant`, so every row after `matmul` printed the NEXT category's total: the row labelled
`prep` was the weight dequant, `attn(g/s)` was prep, `elt` was attn, and `attn_pv` never
printed. Any earlier reading of this breakdown that named a bucket by its printed label is
off by one. Launch counts identify the rows if you meet an old figure: 224 is one per block
linear (dequant), 336 is prep, 28 is attn, 281 is elt. The names now come off the enum.

---

## 3. Diffusion ops and kernels

| Op | cpu | vulkan | zig-cuda | cuda | Formats |
|---|---|---|---|---|---|
| **GEMM / linear** | `ops/matmul.zig` | `opMatmul`/`opMatmulCoop*` | hand hgemm/igemm/i4gemm | cuBLASLt | see §7 |
| **Conv2d** (im2col + GEMM, fused 2× upsample) | ✅ f32 | ✅ f16-TC (co≥96) / f32 | ✅ | ✅ (cuDNN NHWC conv for stride-1 3×3; im2col for stride-2 and upsample) | f32 weights; f16 TC wide-co; f16 activation storage either side |
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
| **Weight noise** (`--weight-noise`, per-layer curve) | ❌ | ❌ | ✅ gemma4, q4_k/q5_k/q6_k/iq4_xs | ↤ |

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

### Weight noise (`--weight-noise`, tp-gui's composer toggle)

Perturbs the block-quant weights while generating: each super-block's `d` (and `dmin`) is
multiplied by `1 + sigma*u`, u uniform in [-1,1], redrawn every forward pass. One draw covers
a block's 256 weights, so it moves the model's FUNCTION rather than its logits, and the token
ranking itself can change while grammar stays intact. `cuda/wnoise.zig` holds the one hash and
the PTX snippet every kernel below shares.

**Shape and amount are separate knobs.** A curve may read `a`, which
`--weight-noise-amount` (default 1) and tp-gui's composer field set, so the
expression says WHERE the noise goes and one number says HOW MUCH — nobody should
have to edit an expression to turn the amount down. `a*(1-t)^2` at amount 0.4 is
bit-identical to the literal `0.4*(1-t)^2` (verified: same greedy output). A curve
that never mentions `a` ignores the knob, which `noise_curve.respondsToAmount`
detects by MEASUREMENT (evaluating at two amounts) rather than by scanning for the
token, so the UI can dim an inert field instead of letting it do nothing quietly.

In tp-gui the shapes are a saved library: the composer carries a toggle, a shape
sparkline, a dropdown of named shapes and the amount field, while the expression
itself is written in Settings (UI.md 13). `--weight-noise` takes the same
expression, so the CLI and the GUI describe schedules identically.

**Sigma is a curve over depth, not a constant.** `core/noise_curve.zig` evaluates an
expression in `t` (0 at the first decoder layer, exactly 1 at the LM head) once per layer into
`WeightNoise.table`; a launch reads the slot for the layer it belongs to, published by
`transformer_gpu.decoderLayer*` via the duck-typed `noiseAtLayer`, and `lmHead` selects the
head slot. A bare number is a flat curve.

**MEASURED, gemma4 31B q4_k, greedy, matched noise budget** (mirror-image curves: same peak,
same total, opposite placement):

| shape / amount | result |
|---|---|
| `a*t^2` @ 0.2 (back-loaded) | token corruption: `ownS-white`, `thePLC aingSL {L de gloom` |
| `a*(1-t)^2` @ 0.2 (front-loaded) | fluent, and rewords vs the control |
| `a*(1-t)^2` @ 0.03 | indistinguishable from the control on this prompt |
| `a*(1-t)^2` @ 0.4 | a materially different answer, still clean prose |
| `max(0, a*(1-t/0.4))` @ 0.6 | coherent with noise on the first 40% of the stack only |
| `a*(1-t)^2` @ 0.8 | collapses to token salad |
| `a*(1-t)^3` @ 0.8 | breaks STRUCTURALLY instead: fluent fragments in a loop |

So placement dominates amplitude, and the two failure modes are distinct: noise near the head
lands on token selection and corrupts glyphs, while noise early destroys the plan and the late
layers faithfully render the wreckage. A front-loaded curve therefore takes a peak sigma
~10x what a flat one tolerates (~0.4-0.6 vs ~0.05), which is the reason the knob is a curve.

sigma 0 everywhere leaves `d * 1.0`, so off is bit-identical (verified token-identical against
the pre-change kernels on a 31B q4_k greedy run), and neither the off nor the on path costs
measurable throughput: sigma 0 and sigma 0.02 interleave inside run-to-run scatter against the
pre-change build (pp ~270 tok/s, tg ~25 tok/s, 1824-token prompt, RTX 3090). No separate
noise-free kernel variant is warranted.

The draw keys on the block's byte offset WITHIN its weight plus a host-folded
(seed, forward index, weight id), NOT an absolute device address: `cuMemAlloc` shuffles
addresses per process, and an address-keyed sweep produces a fresh unrepeatable model at every
sigma. `--weight-noise-seed` makes a run reproducible.

**Two predicates answer "would this do anything", and callers must use both.**
`cuda.Backend.weightNoiseSupported(dt)` is the dtype half and lives with the
kernels, because that is exactly what it describes: which kernels were wired.
`llm.session.archSupportsWeightNoise(arch)` is the arch half — a stepper honors
noise only if it declares `noiseAtLayer` (so `transformer_gpu.decoderLayer*` can
tell it which layer is launching) and ticks the stream in its forward.
`llm.session.weightNoiseSupported(gguf)` combines them, taking a majority over the
block-quant weights under `layers.` rather than probing one tensor, since a mixed
file is normal (this repo's 31B q4_k carries 11 q5_k tensors). tp-gui hides the
controls when it is false; tp-llm warns. A test in `gui/chat.zig` fails if the arch
list and the steppers' `noiseAtLayer` declarations disagree in either direction.

⚠️ `Gguf.names()` returns CANONICAL names, so per-layer tensors are `layers.N.…`,
not the file's own `blk.N.…`. Scanning for the raw spelling matches nothing and
reports every checkpoint unsupported.

Honoring it today:

| kernel | path | dtypes |
|---|---|---|
| `gemv_q4_k_q8n`, `gemv_q5_k_q8n` | decode + grouped prefill | q4_k, q5_k |
| `gemv_q5_k_q8` | decode | q5_k |
| `gemv_q6_k` | LM head (tied `token_embd`) | q6_k |
| `buildMmqPipeQ4K` | batched prefill MMQ | q4_k, q5_k, iq4_xs |

Every MMQ entry declares the two parameters (`mmq_params`, so one launcher fits all), but only
`buildMmqPipeQ4K` reads them: the q6_k / q8_0 / q1_0 / q2_0 pipes and the older
`buildMmqQ4K` ignore sigma, as do the dequant-to-f16 fallback, the fp8/bf16/int8-convrot GEMMs,
and every non-gemma4 stepper (only `gemma4_cuda.forwardRows` ticks the stream, and only
`gemma4_cuda` declares `noiseAtLayer`). The embedding GATHER is a separate kernel from
`gemv_q6_k` — gemma4 embeds host-side entirely — so a tied `token_embd` is perturbed as the LM
head and left alone as the embedding table. ⚠️ Noise on `attn_k`/`attn_v` enters the KV cache
and persists for the rest of the sequence, so those two drift cumulatively while everything
else is resampled per forward.

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

DType enum (`src/dtype.zig`): `f8_e4m3, f16, bf16, f32, i8, i4, q4_0, q8_0, q2_k, q4_k, q5_k,
q6_k, iq4_nl, iq4_xs, q1_0, q2_0_g64, q2_0_g128`. `i8`/`i4` are the ComfyUI convrot formats for the **image**
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
| **GGUF q4_k / q5_k / q6_k** | ✅ ggml | ✅ scalar `gemv_*{,_t}` | ✅ `gemv_*(_q8/_q8n)`; MMQ `mmq_pipe_q{4,5,6}_k` | ⤷ dequant→f16 | q6_k's MMQ is correct but loses to dequant+f16, so `mmqPipeFaster` routes only q4_k/q5_k on. `TP_NO_MMQ5` / `TP_NO_MMQ_IQ4` A/B the routing, `TP_MMQ_NOSTAGE` isolates A-staging cost (garbage output, valid timing) |
| **GGUF q2_k** | ✅ ggml | ❌ | ✅ decode→int8/int4 convrot (`buildPrep`) | ✅ ditto | 256 elems / 84 B; 2-bit codes, 4-bit scale+min per 16, f16 d/dmin at the block TAIL. Diffusion only so far: no GEMV, so no LLM decode path |
| **GGUF iq4_nl** | ✅ ggml | ✅ scalar (module-const LUT) | ✅ shared-mem LUT | ⤷ dequant→f16 | 32 elems / 18 B, non-linear `kvalues_iq4nl` |
| **GGUF iq4_xs** | ✅ ggml | ❌ | ✅ `gemv_iq4_xs` (shared-mem LUT), `embed_gather_iq4_xs`, `mmq_pipe_iq4_xs` | ⤷ dequant→f16 | 256 elems / 136 B; `kvalues_iq4nl` over a k-quant super-block, 6-bit sub-block scale split across `scales_h`/`scales_l`, biased −32. No dp4a GEMV. Its 136-byte block leaves odd super-blocks only 8-byte aligned, so staging loads are `v2` |
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

**Fragment shapes: `buildGemmShared` takes the device's M/N; every other coop builder is
16-only.** NVIDIA and AMD advertise 16×16, Intel Alchemist only 8×8 (`8x8x16` f16 and bf16,
`8x8x32` s8/u8, f32/s32 accumulate only — no f16 C at all). K is always 16, which is what lets
the 32-deep staging stay two k sub-steps. The warp tile is fixed at 64×64, so a smaller fragment
means more fragments per warp (8×8 instead of 4×4), same accumulator registers per lane and 4×
the MulAdds — not a smaller warp, so staging and host dispatch geometry are shape-independent.
`Context.init` picks the widest usable config, and probes f16-accumulate and native bf16 *at that
shape* rather than assuming they exist. Shapes the kernels cannot express leave the coop pipelines
null and the caller falls back.

**Warp tile: `buildGemmShared` also takes the per-warp output tile (M and N separately).**
Accumulators per lane are `warp_m * warp_n / subgroup_size` — independent of the fragment
shape — so 64×64 is 128 f32/lane, fine where a thread has ~255 registers (NVIDIA) and far past
the ~32 f32/lane an Intel Xe lane gets at SIMD32. On an A310 the 64×64 tile makes the coop GEMM
slow enough that `bench-matmul` returns `error_device_lost`; **16×32 is 4.35× faster end to end
(VAE decode 10×)** with output identical to 1 LSB. Vulkan exposes no register-budget query, so
this cannot be derived from limits the way the fragment shape can — `tune-coop` measures every
expressible tile and reports the winner. `TP_COOP_WARP_TILE` / `TP_COOP_WARP_TILE_N` set it.

⚠️ **A rectangular tile can be legal where its square form is not**, so M and N must be validated
as a PAIR. 16×32 is expressible on an A310 and 16×16 is not (B staging would need fewer chunks
than the workgroup has threads). Validating M against itself demotes M to the default alone, and
the run then measures a tile nobody asked for — which is how 64×32 spent a while masquerading as
16×32 here and made the tuner look wrong.

The `tune-coop` harness earned three fixes worth not repeating: measure a WARMUP pass first (this
3090 idles its clocks, so the first candidate measured otherwise reports ~18× low purely for being
first); score a SET of shapes, since ranking on one GEMM picks a tile the render does not want; and
give each probe shape its OWN weight allocation, because `weightBuffer` keys the cache on
`@intFromPtr(bytes.ptr)` with no shape in the key, so slices of one buffer collide and every shape
after the first times a read of a wrongly-transposed weight. Batching all probes into one submit
was measured and made no difference on Alchemist, so per-dispatch latency was not the issue.

⚠️ **`Context.coopNPad` is the only place that may compute the coop GEMM's padded N.** The shader
is told that value as its B row stride, so every producer of a k-major weight it reads
(`weightBufferF16From32`, `weightBufferF16FromBf16`, `nvfp4Decode`) must pad to the same multiple.
A producer that keeps its own literal lays the weight out at offsets the GEMM never reads — a
wrong image, not a crash, and invisible while the wg tile happens to be 128. `nvfp4ScratchBytes`
is exempt because it only sizes scratch, where a larger alignment is an over-estimate.

⚠️ **Per-thread staging arithmetic is not fragment arithmetic.** In `buildGemmShared` the B
staging mask/shift divide a WGN-wide row by the 16-element chunk each *thread* stages (the `<< 4`
beside them), so they must not follow `frag_n`. 16×16 cannot expose the difference, because
`WGN/frag_n` and `WGN/16` are then the same number: getting it wrong renders byte-identically on
NVIDIA and produces garbage only on an 8×8 device. `src/gpu/spirv_asm.zig` `assembleChecked` now
rejects a `%name` that is referenced but never defined, which is the other way a hand-emitted
kernel fails silently (the assembler allocates ids on first mention so forward references work,
and a typo therefore used to assemble into a dangling id the driver answers with a fault).

**int8 (`buildGemmSharedI8`) takes the device s8 fragment M/N and its own warp tile too**, via
`TP_COOP_I8_TILE` / `TP_COOP_I8_TILE_N` and `coopI8TileFits`. K is always 32, so K_STEP stays two
sub-steps. `coop_i8_wg_m/n` drive both the dispatch grid **and** the divisibility guard that
decides whether a shape may take the shared path at all — a smaller tile makes MORE shapes
eligible. `buildGemmI8` (the register-tiled fallback) is still 16-only and explicitly gated, so
`opI8Gemm`/`opI8GemmBuf` require *either* kernel and return `UnsupportedDType` for a shape neither
can express rather than asserting on the absent one.

Intel Alchemist status: SD1.5 renders on an Arc A310 (4 GB) at 8×8, verified 64 dB / max 1 LSB
against the same seed on a 3090, and int8 s8→s32 runs at 8×8×32 with `gpu-i8-test` clean (four
shapes at 0 mismatches, fused-scale at 0.000000 rel err, the same 0.00427 wiring residual the 3090
reports). The 16-only attention builders are gated off there; the SD UNet does not use them (its
attention is the `attn_batched` Zig shader).

`tune-coop` sweeps the int8 tile as well as the f16 one, and the int8 fit admits 16 in the N
position, so it has 9 candidates where f16 has 6. **It persists both winners, and `Context.init`
applies them, so tuning is a one-off per machine and every later run is fast with no flags.**
Precedence is env override, then the measured cache, then the 64×64 default, resolved per path
(f16 and int8 independently) and re-validated by the fit checks before use.

The cache is `<XDG_CACHE_HOME or HOME/.cache>/tensorpencil/coop_tile` (or `TP_COOP_TILE_CACHE`),
one line per device keyed `vendor:device:driver` — the driver version is in the key so a driver
update re-measures rather than trusting a number produced by different codegen. Reading it at init,
rather than re-tiling afterwards, is what lets the pipelines be built at the chosen tile the first
time. `tune-coop` writes it, preserving other devices' lines so one cache serves a multi-GPU box.
`gpu-test` prints the tile in force, which is how you check what a machine actually resolved.

**`Context.init` takes an `Io`** (`init(gpa, io)`), and keeps it, because 0.16 puts both file access
and clocks behind one. Tests pass `std.testing.io`; every other caller already had one in scope.

**A device with no cache entry screens its own tiles at init and saves the result**, so the first
run on a machine self-tunes and every later run just loads. Matching is on `vendor:device:driver`,
so a cache from a different GPU (or a different driver) is ignored rather than applied. Precedence
stays env → cache → screen → 64×64 default. This lives in `Context.init`, so a library consumer
gets it from `gpu.Context.init` or `pipeline.Session.init` with no extra call; `context.autotune_tiles = false`
disables it programmatically and `context.tile_cache_path` redirects the file, the two things an
embedder needs (`TP_COOP_AUTOTUNE=0` / `TP_COOP_TILE_CACHE` are the env equivalents).

⚠️ **The init screen is coarse on purpose and must only be trusted to find the LARGE effect.** It
uses small shapes and runs in 0.3s on a 3090, 13s on an A310. Getting there took three corrections
worth not repeating: without a warmup it measured the first candidate (the 64×64 default) cold and
picked the wrong tile; with one pass it was not even self-consistent, since back-to-back runs
disagreed about the winner; and a tight margin let it prefer a tile the full sweep shows is 14%
worse. It now runs **two passes and only switches when a candidate beats the default by ≥1.67× in
both**, so a noisy screen can decline to change a tile but never commit to a wrong one. On an A310
it finds f16 16×32 (what `tune-coop` finds) and int8 16×16 (`tune-coop` finds 16×32, 1.3× better) —
i.e. it captures ~13× of the ~15.7× and leaves the remainder to the explicit command.

⚠️ **On Alchemist the best int8 tile is warp 16×32 at 781 GFLOP/s — 15.7× the 64×64 default's
49.7, and 1.5× the 599 that square 16×16 gives.** The rectangle wins on both the f16 and int8
paths, so never sweep only square tiles: a hand sweep of squares picked 16×16 and left 1.5× on the
floor. For scale the 3090 does ~80,000 GFLOP/s int8 here at 64×64, so Alchemist int8 remains ~100×
behind — far more than its ~5-8× specification gap, i.e. correct and much improved but not tuned.
(An earlier note here claimed int8 and f16 both plateaued at ~524 and inferred the matrix units
were therefore not the limit. That was measured on int8's square tile only; at 16×32 int8 reaches
781 and clearly exceeds the f16 path, so the convergence argument does not hold.)

### zig-cuda — hand-PTX (`src/gpu/cuda/kernels.zig` GEMM, `elt.zig` elementwise/attn)

GEMM builders: `buildHgemm` (f16/bf16 mma m16n8k16, optional f32 A/C) · `buildIgemmSmem` /
`buildIgemmPipe` (int8 m16n8k32, int4 m16n8k64) · `buildPrep` (quant/rotate) ·
`buildMmqPipeQ{1_0,2_0}` and the q4_k/q6_k MMQ pipes. All DiT GEMM formats use warp-cooperative
`ldmatrix.x4`/`.x2` fragment loads with an XOR-swizzled conflict-free shared layout
(`use_ldmatrix`; a pure permutation, so bit-exact).

GEMV: `gemv_{fp8,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k,iq4_nl,iq4_xs,q1_0,q2_0_g64,q2_0_g128}` plus `_q8`
and grouped-N `_q8n` dp4a variants (iq4_xs has neither: it decodes f32 straight from the block).

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
| `mmq_pipe_q4_k` at ~24% of int8 peak | **Not on the diffusion path** (a q4_k/q8_0 DiT decodes to int8-convrot and uses the vendor GEMM); it is the LLM q4_k prefill kernel. 369 ms/step at lat=64, down from 434, all of it from shared-memory BANK CONFLICTS on the fragment loads. ⚠️ SEVEN plausible causes measured NOT to be it: ALU (4%), spill (`kstep` 128 spills zero, 24% slower), occupancy (forcing 3-4 blocks/SM is 10x WORSE — the 128 f32 accumulators spill per mma), cp.async double-buffering (10% slower), the s32→f32 `cvt`, DRAM (6%), ldmatrix (50% slower). Nsight: latency bound at 1.93 warps/scheduler of 12, ~1.5x ceiling. Read the block comment before optimizing. |
| q8_0 MMQ built and LOST | `mmq_pipe_q8_0` exists, is correct (device test against an exact f64 reference, teeth checked by mis-wiring the per-substep scale) and is opt-in via `--dit-gguf-gemm mmq`. It measures **566 ms** of GEMM per step against the f16 route's **440** and cuBLASLt int8's **141**. ⚠️ Do not retry it expecting the estimate that motivated it: the premise was that `igemm_pipe` runs ~1.68x cuBLASLt, but igemm_pipe chains the mma's s32 C operand across k and NO MMQ can, because the scale changes every 32 elements. Isolation: A staging is 225 of the 566 (`TP_MMQ8_NOSTAGE` gives 342), and even at 342 it loses, because q4_k's nibble packing feeds TWO substeps from one 32-byte A fragment where 8-bit weights need their own, doubling shared A-load traffic on the one axis this kernel family responds to. The only real lever left is a one-time repack to planar qs + a scale plane, worth ~5% end-to-end on this card. |
| No GPU GEMM for GGUF block quants other than q4_k/q8_0 in diffusion | q5_k/q6_k/q4_0/iq4_nl DiTs are CPU-only on every backend; `gpuLinKindSupported` + `Backend.blockQFormat` are the two places to widen, and each needs only a load-stage block walk in `buildPrep`. |
| krea2 has no Vulkan int4 path | `dit_gpu` never accepted it. `i4_decode_t` is now most of what it would need. |
| q2_0 g64 kernels unexecuted | GEMV and MMQ are generated from the same templates as g128 but no g64 file exists here to run them against. |
| `--text-encoder-2` split path unexercised | every SDXL checkpoint here is bundled, so the flag is built and reviewed but not measured. |
